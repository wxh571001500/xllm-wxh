# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable routed-expert backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from xllm.python.ascend_custom_ops import ensure_ascend_custom_ops
from xllm.python.layers.moe.types import (
    MoEExpertsConfig,
    MoETokenDispatchOutput,
)

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


_ACL_FORMAT_FRACTAL_NZ = 29


def _select_tensor_shard(
    tensor: torch.Tensor,
    dim: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    if world_size == 1:
        return tensor
    if tensor.shape[dim] % world_size != 0:
        raise ValueError("MoE weight dimension must divide parallel size")
    return torch.tensor_split(tensor, world_size, dim=dim)[rank].contiguous()


def _load_packed_expert_shard(
    state_dict: StateDict,
    name: str,
    config: MoEExpertsConfig,
    tp_dim: int | None,
) -> torch.Tensor:
    tensor = state_dict.get_sharded_tensor(
        name,
        0,
        config.ep_rank,
        config.ep_size,
    )
    if tp_dim is not None:
        tensor = _select_tensor_shard(
            tensor,
            tp_dim,
            config.tp_rank,
            config.tp_size,
        )
    return tensor


class RoutedExperts(nn.Module, ABC):
    """Common interface implemented by all routed-experts backends."""

    @abstractmethod
    def forward(
        self,
        dispatch_output: MoETokenDispatchOutput,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_weights(
        self,
        state_dict: StateDict,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def finish_weight_loading(self) -> None:
        raise NotImplementedError


class UnquantizedRoutedExperts(RoutedExperts):
    """TP-sharded routed experts executed with standard linear operations."""

    def __init__(
        self,
        config: MoEExpertsConfig,
        activation: nn.Module,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if config.intermediate_size % config.tp_size != 0:
            raise ValueError("MoE expert intermediate size must divide tp_size")
        self._config = config
        intermediate_per_rank = config.intermediate_size // config.tp_size
        self.global_num_experts = config.num_experts
        self.num_experts = config.num_local_experts
        self.first_expert_id = config.first_expert_id
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tp_size = config.tp_size
        self.activation = activation
        self._loaded_mask = torch.zeros(
            (config.num_local_experts, 3),
            dtype=torch.bool,
            device="cpu",
        )
        self._packed_loaded: set[str] = set()
        self.w13_weight = nn.Parameter(
            torch.empty(
                config.num_local_experts,
                2 * intermediate_per_rank,
                config.hidden_size,
                dtype=dtype,
                device=device,
            )
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                config.num_local_experts,
                config.hidden_size,
                intermediate_per_rank,
                dtype=dtype,
                device=device,
            )
        )
        nn.init.normal_(self.w13_weight, std=0.02)
        nn.init.normal_(self.w2_weight, std=0.02)

    def forward(
        self,
        dispatch_output: MoETokenDispatchOutput,
    ) -> torch.Tensor:
        hidden_states = dispatch_output.hidden_states
        if hidden_states.shape[0] == 0:
            return hidden_states.new_empty((0, self.hidden_size))
        if dispatch_output.group_list.numel() != self.num_experts:
            raise ValueError("MoE group_list must contain one value per expert")
        if dispatch_output.group_list_type == 1:
            group_boundaries = dispatch_output.group_list.cumsum(dim=0)
        elif dispatch_output.group_list_type == 0:
            group_boundaries = dispatch_output.group_list
        else:
            raise ValueError(f"Unsupported MoE group_list_type: {dispatch_output.group_list_type}")

        expert_outputs: list[torch.Tensor] = []
        group_start = 0
        for expert_id, group_end_tensor in enumerate(group_boundaries):
            group_end = int(group_end_tensor.item())
            if group_end == group_start:
                continue
            expert_input = hidden_states[group_start:group_end]
            gate_up = F.linear(expert_input, self.w13_weight[expert_id])
            expert_output = self.activation(gate_up)
            expert_output = F.linear(expert_output, self.w2_weight[expert_id])
            expert_outputs.append(expert_output)
            group_start = group_end
        if group_start != hidden_states.shape[0]:
            raise ValueError("MoE group_list does not match dispatched tokens")
        return torch.cat(expert_outputs, dim=0)

    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        packed_name = name.removesuffix(".weight")
        packed_targets = {
            "w13_weight": self.w13_weight,
            "w2_weight": self.w2_weight,
        }
        packed_target = packed_targets.get(packed_name)
        if packed_target is not None:
            if tensor.shape != packed_target.shape:
                raise ValueError(f"MoE {packed_name} expects {packed_target.shape}, got {tensor.shape}")
            packed_target.data.copy_(tensor.to(packed_target))
            self._packed_loaded.add(packed_name)
            return True

        parts = name.split(".")
        if len(parts) != 3 or parts[2] != "weight":
            return False
        try:
            expert_id = int(parts[0])
        except ValueError:
            return False
        if not self.first_expert_id <= expert_id < (self.first_expert_id + self.num_experts):
            return False
        local_expert_id = expert_id - self.first_expert_id
        projection_group = {
            "w1": "gate",
            "gate_proj": "gate",
            "w3": "up",
            "up_proj": "up",
            "w2": "down",
            "down_proj": "down",
        }.get(parts[1])
        if projection_group is None:
            return False

        if projection_group in ("gate", "up"):
            half = self.w13_weight.shape[1] // 2
            start = 0 if projection_group == "gate" else half
            target = self.w13_weight.data[
                local_expert_id,
                start : start + half,
            ]
        else:
            target = self.w2_weight.data[local_expert_id]
        if tensor.shape != target.shape:
            raise ValueError(f"MoE expert {name} expects {target.shape}, got {tensor.shape}")
        target.copy_(tensor.to(target))
        projection_index = {"gate": 0, "up": 1, "down": 2}[projection_group]
        self._loaded_mask[local_expert_id, projection_index] = True
        return True

    def load_weights(
        self,
        state_dict: StateDict,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        del tp_rank, tp_size
        loaded: set[str] = set()
        for name in state_dict.keys():
            packed_name = name.removesuffix(".weight")
            if packed_name in ("w13_weight", "w2_weight"):
                tp_dim = 1 if packed_name == "w13_weight" else 2
                tensor = _load_packed_expert_shard(
                    state_dict,
                    name,
                    self._config,
                    tp_dim,
                )
            elif len(name.split(".")) == 3:
                parts = name.split(".")
                try:
                    expert_id = int(parts[0])
                except ValueError:
                    continue
                if not self.first_expert_id <= expert_id < (self.first_expert_id + self.num_experts):
                    continue
                shard_dim = 0 if parts[1] in ("w1", "w3", "gate_proj", "up_proj") else 1
                tensor = state_dict.get_sharded_tensor(
                    name,
                    shard_dim,
                    self._config.tp_rank,
                    self._config.tp_size,
                )
            else:
                tensor = state_dict.get_tensor(name)
            if self.load_weight(name, tensor):
                loaded.add(name)
        return loaded

    def finish_weight_loading(self) -> None:
        if self._packed_loaded:
            required_packed = {"w13_weight", "w2_weight"}
            missing = required_packed.difference(self._packed_loaded)
            if missing:
                raise KeyError(f"Packed expert weights are missing: {sorted(missing)}")
            return
        if not bool(self._loaded_mask.all()):
            raise KeyError("Routed expert weights are incomplete")


def _ensure_fused_w4a8_custom_op() -> None:
    ensure_ascend_custom_ops(("dequant_situ_quant",))


def _dequant_situ_quant(
    hidden_states: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ensure_fused_w4a8_custom_op()
    return torch.ops._C_ascend.dequant_situ_quant(
        x=hidden_states,
        weight_scale=None,
        activation_scale=None,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=None,
        beta=beta,
        linear_beta=linear_beta,
        activate_left=True,
        quant_mode="dynamic",
    )


class QuantizedExpertsActivation(nn.Module, ABC):
    """Activation contract for dynamically quantized expert pipelines."""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class FusedQuantizedSituAndMul(QuantizedExpertsActivation):
    """Fused SiTU activation producing quantized output and token scales."""

    def __init__(self, beta: float, linear_beta: float | None) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _dequant_situ_quant(
            hidden_states,
            self.beta,
            self.linear_beta,
        )


class FusedW4A8RoutedExperts(RoutedExperts):
    """ModelSlim W4A8 routed experts implemented with fused grouped matmul."""

    def __init__(
        self,
        config: MoEExpertsConfig,
        activation: QuantizedExpertsActivation,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if config.intermediate_size % config.tp_size != 0:
            raise ValueError("MoE expert intermediate size must divide tp_size")
        self._config = config
        intermediate_per_rank = config.intermediate_size // config.tp_size
        if intermediate_per_rank % 2 != 0 or config.hidden_size % 2 != 0 or 16 % config.tp_size != 0:
            raise ValueError("Packed W4A8 expert dimensions are invalid")
        self.global_num_experts = config.num_experts
        self.num_experts = config.num_local_experts
        self.first_expert_id = config.first_expert_id
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tp_size = config.tp_size
        self.activation = activation
        self.output_dtype = dtype
        self._runtime_weights_ready = False
        if device.type in ("npu", "privateuseone"):
            _ensure_fused_w4a8_custom_op()
        self._loaded_mask = torch.zeros(
            (config.num_local_experts, 3, 4),
            dtype=torch.bool,
            device="cpu",
        )
        self._packed_loaded: set[str] = set()
        self.w13_weight = nn.Parameter(
            torch.empty(
                config.num_local_experts,
                intermediate_per_rank,
                config.hidden_size,
                dtype=torch.int8,
                device=device,
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                config.num_local_experts,
                config.hidden_size // 2,
                intermediate_per_rank,
                dtype=torch.int8,
                device=device,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "w13_weight_scale",
            torch.empty(
                config.num_local_experts,
                2 * intermediate_per_rank,
                1,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "w13_weight_offset",
            torch.empty_like(self.w13_weight_scale),
        )
        self.register_buffer(
            "w13_scale_bias",
            torch.empty_like(self.w13_weight_scale),
        )
        self.register_buffer(
            "w2_weight_scale",
            torch.empty(
                config.num_local_experts,
                config.hidden_size,
                1,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "w2_weight_offset",
            torch.empty_like(self.w2_weight_scale),
        )
        self.register_buffer(
            "w2_scale_bias",
            torch.empty(
                config.num_local_experts,
                config.hidden_size,
                16 // config.tp_size,
                dtype=torch.float32,
                device=device,
            ),
        )

    def forward(
        self,
        dispatch_output: MoETokenDispatchOutput,
    ) -> torch.Tensor:
        if not self._runtime_weights_ready:
            raise RuntimeError("W4A8 expert weights are not ready")
        hidden_states = dispatch_output.hidden_states
        if hidden_states.shape[0] == 0:
            return torch.empty(
                (0, self.hidden_size),
                dtype=self.output_dtype,
                device=hidden_states.device,
            )
        if dispatch_output.dynamic_scale is None:
            raise ValueError("W4A8 experts require dispatched token scales")
        gate_up = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[self.w13_weight],
            scale=[self.w13_weight_scale.unsqueeze(-2)],
            bias=[self.w13_scale_bias],
            per_token_scale=[dispatch_output.dynamic_scale],
            split_item=2,
            group_list_type=dispatch_output.group_list_type,
            group_type=0,
            group_list=dispatch_output.group_list,
            output_dtype=torch.bfloat16,
        )[0]
        activated, activated_scale = self.activation(gate_up)
        expert_output = torch_npu.npu_grouped_matmul(
            x=[activated],
            weight=[self.w2_weight],
            scale=[self.w2_weight_scale],
            bias=[self.w2_scale_bias],
            per_token_scale=[activated_scale],
            split_item=2,
            group_list_type=dispatch_output.group_list_type,
            group_type=0,
            group_list=dispatch_output.group_list,
            output_dtype=self.output_dtype,
        )[0]
        return expert_output

    @staticmethod
    def _encode_per_channel_scale(scale: torch.Tensor) -> torch.Tensor:
        transposed = scale.transpose(1, 2).contiguous()
        encoded = transposed.cpu().view(torch.int32).to(torch.int64)
        return encoded.to(device=scale.device)

    def _process_quantized_weights(self) -> None:
        if self._runtime_weights_ready:
            return
        if self.w13_weight.shape[-2] % 4 != 0:
            raise ValueError("W4A8 w13 packed dimension must divide 4")
        if self.w2_weight.shape[-2] % 4 != 0:
            raise ValueError("W4A8 w2 packed dimension must divide 4")

        self.w13_weight.data = self.w13_weight.data.transpose(1, 2).contiguous()
        self.w2_weight.data = self.w2_weight.data.transpose(1, 2).contiguous()
        if self.w13_weight.device.type in ("npu", "privateuseone"):
            self.w13_weight.data = torch_npu.npu_format_cast(
                self.w13_weight.data,
                _ACL_FORMAT_FRACTAL_NZ,
            )
            self.w2_weight.data = torch_npu.npu_format_cast(
                self.w2_weight.data,
                _ACL_FORMAT_FRACTAL_NZ,
            )
        self.w13_weight.data = self.w13_weight.data.view(torch.int32).contiguous()
        self.w2_weight.data = self.w2_weight.data.view(torch.int32).contiguous()
        self.w13_weight_scale.data = self._encode_per_channel_scale(self.w13_weight_scale.data).squeeze(1)
        self.w2_weight_scale.data = self._encode_per_channel_scale(self.w2_weight_scale.data)
        self.w13_scale_bias.data = self.w13_scale_bias.data.transpose(1, 2).contiguous().sum(dim=1)
        self.w2_scale_bias.data = self.w2_scale_bias.data.transpose(1, 2).contiguous().sum(dim=1)
        self._runtime_weights_ready = True

    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        packed_targets = {
            "w13_weight": self.w13_weight,
            "w2_weight": self.w2_weight,
            "w13_weight_scale": self.w13_weight_scale,
            "w13_weight_offset": self.w13_weight_offset,
            "w13_scale_bias": self.w13_scale_bias,
            "w2_weight_scale": self.w2_weight_scale,
            "w2_weight_offset": self.w2_weight_offset,
            "w2_scale_bias": self.w2_scale_bias,
        }
        packed_name = name.removesuffix(".weight")
        packed_target = packed_targets.get(packed_name)
        if packed_target is not None:
            if tensor.shape != packed_target.shape:
                raise ValueError(f"MoE {packed_name} expects {packed_target.shape}, got {tensor.shape}")
            packed_target.data.copy_(tensor.to(packed_target))
            self._packed_loaded.add(packed_name)
            return True

        parts = name.split(".")
        if len(parts) != 3:
            return False
        try:
            expert_id = int(parts[0])
        except ValueError:
            return False
        if not self.first_expert_id <= expert_id < (self.first_expert_id + self.num_experts):
            return False
        local_expert_id = expert_id - self.first_expert_id
        projection_group = {
            "w1": "gate",
            "gate_proj": "gate",
            "w3": "up",
            "up_proj": "up",
            "w2": "down",
            "down_proj": "down",
        }.get(parts[1])
        if projection_group is None:
            return False
        suffix = parts[2]

        if projection_group in ("gate", "up"):
            target_tensor = self._w13_target(suffix)
            if target_tensor is None:
                return False
            half = target_tensor.shape[1] // 2
            start = 0 if projection_group == "gate" else half
            target = target_tensor.data[
                local_expert_id,
                start : start + half,
            ]
        else:
            target_tensor = self._w2_target(suffix)
            if target_tensor is None:
                return False
            target = target_tensor.data[local_expert_id]
        if tensor.shape != target.shape:
            raise ValueError(f"MoE expert {name} expects {target.shape}, got {tensor.shape}")
        target.copy_(tensor.to(target))
        projection_index = {"gate": 0, "up": 1, "down": 2}[projection_group]
        suffix_index = {
            "weight": 0,
            "weight_scale": 1,
            "weight_offset": 2,
            "scale_bias": 3,
        }[suffix]
        self._loaded_mask[
            local_expert_id,
            projection_index,
            suffix_index,
        ] = True
        return True

    def _w13_target(self, suffix: str) -> torch.Tensor | None:
        return {
            "weight": self.w13_weight,
            "weight_scale": self.w13_weight_scale,
            "weight_offset": self.w13_weight_offset,
            "scale_bias": self.w13_scale_bias,
        }.get(suffix)

    def _w2_target(self, suffix: str) -> torch.Tensor | None:
        return {
            "weight": self.w2_weight,
            "weight_scale": self.w2_weight_scale,
            "weight_offset": self.w2_weight_offset,
            "scale_bias": self.w2_scale_bias,
        }.get(suffix)

    def load_weights(
        self,
        state_dict: StateDict,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        del tp_rank, tp_size
        loaded: set[str] = set()
        for name in state_dict.keys():
            packed_name = name.removesuffix(".weight")
            if packed_name in ("w13_weight", "w2_weight"):
                tensor = _load_packed_expert_shard(
                    state_dict,
                    name,
                    self._config,
                    1 if packed_name == "w13_weight" else 2,
                )
            elif packed_name in (
                "w13_weight_scale",
                "w13_weight_offset",
                "w13_scale_bias",
            ):
                tensor = _load_packed_expert_shard(
                    state_dict,
                    name,
                    self._config,
                    1,
                )
            elif packed_name == "w2_scale_bias":
                tensor = _load_packed_expert_shard(
                    state_dict,
                    name,
                    self._config,
                    2,
                )
            elif packed_name in ("w2_weight_scale", "w2_weight_offset"):
                tensor = _load_packed_expert_shard(
                    state_dict,
                    name,
                    self._config,
                    None,
                )
            elif len(name.split(".")) == 3:
                parts = name.split(".")
                try:
                    expert_id = int(parts[0])
                except ValueError:
                    continue
                if not self.first_expert_id <= expert_id < (self.first_expert_id + self.num_experts):
                    continue
                projection = parts[1]
                suffix = parts[2]
                if projection in ("w1", "w3", "gate_proj", "up_proj"):
                    tensor = state_dict.get_sharded_tensor(
                        name,
                        0,
                        self._config.tp_rank,
                        self._config.tp_size,
                    )
                elif suffix in ("weight", "scale_bias"):
                    tensor = state_dict.get_sharded_tensor(
                        name,
                        1,
                        self._config.tp_rank,
                        self._config.tp_size,
                    )
                else:
                    tensor = state_dict.get_tensor(name)
            else:
                continue
            if self.load_weight(name, tensor):
                loaded.add(name)
        return loaded

    def finish_weight_loading(self) -> None:
        if self._packed_loaded:
            required_packed = {
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w13_weight_offset",
                "w13_scale_bias",
                "w2_weight_scale",
                "w2_weight_offset",
                "w2_scale_bias",
            }
            missing = required_packed.difference(self._packed_loaded)
            if missing:
                raise KeyError(f"Packed expert weights are missing: {sorted(missing)}")
        elif not bool(self._loaded_mask.all()):
            raise KeyError("Routed expert weights are incomplete")

        if self.w13_weight.device.type in ("npu", "privateuseone"):
            self._process_quantized_weights()


__all__ = [
    "FusedQuantizedSituAndMul",
    "FusedW4A8RoutedExperts",
    "QuantizedExpertsActivation",
    "RoutedExperts",
    "UnquantizedRoutedExperts",
]
