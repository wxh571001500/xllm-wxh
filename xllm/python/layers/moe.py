# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kimi K3 MoE layers.

The implementation is deliberately self-contained.  Kimi K3's routed experts
use a latent projection around the expert MLP, and the checkpoint stores the
three expert projections separately while the runtime keeps the two gated
projections packed.  Keeping the conversion here lets the text model dispatch
weights to the owning layer instead of maintaining a model-wide parameter map.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from xllm.python import ops


_ACL_FORMAT_FRACTAL_NZ = 29
_KIMI_K3_CUSTOM_OP_HANDLES: list[Any] = []


def _prepend_env_path(name: str, path: str) -> None:
    entries = [entry for entry in os.environ.get(name, "").split(":") if entry]
    if path not in entries:
        entries.insert(0, path)
        os.environ[name] = ":".join(entries)


def _ensure_kimi_k3_w4a8_custom_op() -> None:
    if hasattr(torch.ops._C_ascend, "dequant_situ_quant"):
        return
    try:
        distribution = importlib.metadata.distribution("vllm-ascend")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Kimi K3 W4A8 execution requires the vllm-ascend package"
        ) from error

    package_dir = Path(distribution.locate_file("vllm_ascend"))
    vendor_dir = (
        package_dir
        / "_cann_ops_custom"
        / "vendors"
        / "custom_transformer"
    )
    vendor_library = vendor_dir / "op_api" / "lib" / "libcust_opapi.so"
    kernels_library = package_dir / "libvllm_ascend_kernels.so"
    extension_paths = sorted(package_dir.glob("vllm_ascend_C.*.so"))
    required_paths = [vendor_library, kernels_library]
    if not extension_paths or any(not path.is_file() for path in required_paths):
        raise RuntimeError(
            "Installed vllm-ascend does not contain the Kimi K3 custom op libraries"
        )

    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", str(vendor_dir))
    _KIMI_K3_CUSTOM_OP_HANDLES.extend(
        [
            ctypes.CDLL(str(vendor_library), mode=ctypes.RTLD_GLOBAL),
            ctypes.CDLL(str(kernels_library), mode=ctypes.RTLD_GLOBAL),
        ]
    )
    torch.ops.load_library(str(extension_paths[0]))
    if not hasattr(torch.ops._C_ascend, "dequant_situ_quant"):
        raise RuntimeError(
            "vllm-ascend did not register _C_ascend.dequant_situ_quant"
        )


def _dequant_situ_quant(
    hidden_states: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ensure_kimi_k3_w4a8_custom_op()
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


def _state_dict_tensor(state_dict: Any, name: str) -> torch.Tensor | None:
    if not state_dict.has(name):
        return None
    return state_dict.get_tensor(name)


def _state_dict_with_prefix(state_dict: Any, prefix: str) -> Any:
    return state_dict.get_dict_with_prefix(prefix)


def _state_dict_size(state_dict: Any) -> int:
    if hasattr(state_dict, "size"):
        return int(state_dict.size())
    return len(state_dict.keys())


def _state_dict_sharded_tensor(
    state_dict: Any,
    name: str,
    dim: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor | None:
    if not state_dict.has(name):
        return None
    if hasattr(state_dict, "get_sharded_tensor"):
        return state_dict.get_sharded_tensor(name, dim, tp_rank, tp_size)
    tensor = state_dict.get_tensor(name)
    if tp_size == 1:
        return tensor
    if tensor.shape[dim] % tp_size != 0:
        raise ValueError(
            f"Kimi K3 tensor dimension {tensor.shape[dim]} is not divisible "
            f"by tp_size {tp_size}"
        )
    shard_size = tensor.shape[dim] // tp_size
    return tensor.narrow(dim, tp_rank * shard_size, shard_size).contiguous()


def _situ_and_mul(
    tensor: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    if tensor.shape[-1] % 2 != 0:
        raise ValueError("Kimi K3 SiTU input must have an even last dimension")
    width = tensor.shape[-1] // 2
    gate, up = tensor[..., :width], tensor[..., width:]
    gate = gate.float()
    up = up.float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(dtype=tensor.dtype)


class KimiK3RoutedExperts(nn.Module):
    """TP-sharded routed expert matrices used by :class:`KimiK3MoE`."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int,
        dtype: torch.dtype,
        device: torch.device,
        quantized: bool = False,
    ) -> None:
        super().__init__()
        if intermediate_size % tp_size != 0:
            raise ValueError("Kimi K3 expert intermediate size must divide tp_size")
        intermediate_per_rank = intermediate_size // tp_size
        if quantized and (
            intermediate_per_rank % 2 != 0
            or hidden_size % 2 != 0
            or 16 % tp_size != 0
        ):
            raise ValueError("Kimi K3 packed W4A8 dimensions are invalid")
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = tp_size
        self.quantized = quantized
        self._runtime_weights_ready = False
        if quantized and device.type in ("npu", "privateuseone"):
            _ensure_kimi_k3_w4a8_custom_op()
        self._loaded_mask = torch.zeros(
            (num_experts, 3, 4), dtype=torch.bool, device="cpu"
        )
        self._packed_loaded: set[str] = set()
        w13_output = intermediate_per_rank if quantized else 2 * intermediate_per_rank
        w2_output = hidden_size // 2 if quantized else hidden_size
        weight_dtype = torch.int8 if quantized else dtype
        self.w13_weight = nn.Parameter(
            torch.empty(
                num_experts,
                w13_output,
                hidden_size,
                dtype=weight_dtype,
                device=device,
            ),
            requires_grad=not quantized,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                num_experts,
                w2_output,
                intermediate_per_rank,
                dtype=weight_dtype,
                device=device,
            ),
            requires_grad=not quantized,
        )
        if quantized:
            self.register_buffer(
                "w13_weight_scale",
                torch.empty(
                    num_experts,
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
                    num_experts,
                    hidden_size,
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
                    num_experts,
                    hidden_size,
                    16 // tp_size,
                    dtype=torch.float32,
                    device=device,
                ),
            )
        else:
            nn.init.normal_(self.w13_weight, std=0.02)
            nn.init.normal_(self.w2_weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        beta: float,
        linear_beta: float | None,
    ) -> torch.Tensor:
        if self.quantized:
            return self._forward_quantized(
                hidden_states,
                topk_ids,
                topk_weights,
                beta,
                linear_beta,
            )
        output = torch.zeros_like(hidden_states)
        for expert_id in range(self.num_experts):
            token_ids, choices = torch.where(topk_ids == expert_id)
            if token_ids.numel() == 0:
                continue
            expert_input = hidden_states.index_select(0, token_ids)
            gate_up = F.linear(expert_input, self.w13_weight[expert_id])
            expert_output = _situ_and_mul(gate_up, beta, linear_beta)
            expert_output = F.linear(expert_output, self.w2_weight[expert_id])
            expert_output = expert_output * topk_weights[token_ids, choices].unsqueeze(-1)
            output.index_add_(0, token_ids, expert_output)
        if self.tp_size > 1:
            ops.all_reduce_(output)
        return output

    def _forward_quantized(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        beta: float,
        linear_beta: float | None,
    ) -> torch.Tensor:
        if not self._runtime_weights_ready:
            raise RuntimeError("Kimi K3 W4A8 expert weights are not ready")
        if hidden_states.shape[0] == 0:
            return torch.zeros_like(hidden_states)

        num_tokens = hidden_states.shape[0]
        (
            sorted_hidden_states,
            expanded_row_indices,
            expert_tokens,
            per_token_scale,
        ) = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids.to(torch.int32),
            scale=None,
            active_num=num_tokens * topk_ids.shape[1],
            expert_num=self.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self.num_experts],
            quant_mode=1,
        )
        group_list = expert_tokens.to(torch.int64)
        gate_up = torch_npu.npu_grouped_matmul(
            x=[sorted_hidden_states],
            weight=[self.w13_weight],
            scale=[self.w13_weight_scale.unsqueeze(-2)],
            bias=[self.w13_scale_bias],
            per_token_scale=[per_token_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.bfloat16,
        )[0]
        activated, activated_scale = _dequant_situ_quant(
            gate_up,
            beta,
            linear_beta,
        )
        expert_output = torch_npu.npu_grouped_matmul(
            x=[activated],
            weight=[self.w2_weight],
            scale=[self.w2_weight_scale],
            bias=[self.w2_scale_bias],
            per_token_scale=[activated_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
            output_dtype=hidden_states.dtype,
        )[0]
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=expert_output,
            sorted_indices=expanded_row_indices.abs(),
            probs=topk_weights.to(expert_output.dtype),
        )
        if self.tp_size > 1:
            ops.all_reduce_(output)
        return output

    @staticmethod
    def _encode_per_channel_scale(scale: torch.Tensor) -> torch.Tensor:
        transposed = scale.transpose(1, 2).contiguous()
        encoded = transposed.cpu().view(torch.int32).to(torch.int64)
        return encoded.to(device=scale.device)

    def _process_quantized_weights(self) -> None:
        if self._runtime_weights_ready:
            return
        if self.w13_weight.shape[-2] % 4 != 0:
            raise ValueError("Kimi K3 W4A8 w13 packed dimension must divide 4")
        if self.w2_weight.shape[-2] % 4 != 0:
            raise ValueError("Kimi K3 W4A8 w2 packed dimension must divide 4")

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
        self.w13_weight_scale.data = self._encode_per_channel_scale(
            self.w13_weight_scale.data
        ).squeeze(1)
        self.w2_weight_scale.data = self._encode_per_channel_scale(
            self.w2_weight_scale.data
        )
        self.w13_scale_bias.data = (
            self.w13_scale_bias.data.transpose(1, 2).contiguous().sum(dim=1)
        )
        self.w2_scale_bias.data = (
            self.w2_scale_bias.data.transpose(1, 2).contiguous().sum(dim=1)
        )
        self._runtime_weights_ready = True

    def load_weight(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> bool:
        """Load one packed or per-expert checkpoint tensor."""
        packed_targets = {
            "w13_weight": self.w13_weight,
            "w2_weight": self.w2_weight,
        }
        if self.quantized:
            packed_targets.update(
                {
                    "w13_weight_scale": self.w13_weight_scale,
                    "w13_weight_offset": self.w13_weight_offset,
                    "w13_scale_bias": self.w13_scale_bias,
                    "w2_weight_scale": self.w2_weight_scale,
                    "w2_weight_offset": self.w2_weight_offset,
                    "w2_scale_bias": self.w2_scale_bias,
                }
            )
        packed_name = name.removesuffix(".weight")
        target = packed_targets.get(packed_name)
        if target is not None:
            if tensor.shape != target.shape:
                raise ValueError(
                    f"Kimi K3 {packed_name} expects {target.shape}, got {tensor.shape}"
                )
            target.data.copy_(tensor.to(target))
            self._packed_loaded.add(packed_name)
            return True

        parts = name.split(".")
        if len(parts) != 3:
            return False
        try:
            expert_id = int(parts[0])
        except ValueError:
            return False
        if not 0 <= expert_id < self.num_experts:
            return False
        projection = parts[1]
        suffix = parts[2]
        projection_group = {
            "w1": "gate",
            "gate_proj": "gate",
            "w3": "up",
            "up_proj": "up",
            "w2": "down",
            "down_proj": "down",
        }.get(projection)
        if projection_group is None:
            return False
        if not self.quantized and suffix != "weight":
            return False

        if projection_group in ("gate", "up"):
            if suffix == "weight":
                target_tensor = self.w13_weight
            elif suffix == "weight_scale":
                target_tensor = self.w13_weight_scale
            elif suffix == "weight_offset":
                target_tensor = self.w13_weight_offset
            elif suffix == "scale_bias":
                target_tensor = self.w13_scale_bias
            else:
                return False
            half = target_tensor.shape[1] // 2
            start = 0 if projection_group == "gate" else half
            target = target_tensor.data[expert_id, start : start + half]
        else:
            if suffix == "weight":
                target_tensor = self.w2_weight
            elif suffix == "weight_scale":
                target_tensor = self.w2_weight_scale
            elif suffix == "weight_offset":
                target_tensor = self.w2_weight_offset
            elif suffix == "scale_bias":
                target_tensor = self.w2_scale_bias
            else:
                return False
            target = target_tensor.data[expert_id]
        if tensor.shape != target.shape:
            raise ValueError(
                f"Kimi K3 expert {name} expects {target.shape}, got {tensor.shape}"
            )
        target.copy_(tensor.to(target))
        projection_index = {"gate": 0, "up": 1, "down": 2}[projection_group]
        suffix_index = {
            "weight": 0,
            "weight_scale": 1,
            "weight_offset": 2,
            "scale_bias": 3,
        }[suffix]
        self._loaded_mask[expert_id, projection_index, suffix_index] = True
        return True

    def load_weights(
        self,
        state_dict: Any,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        loaded: set[str] = set()
        for name in state_dict.keys():
            packed_name = name.removesuffix(".weight")
            if packed_name in ("w13_weight", "w2_weight"):
                shard_dim = 1 if packed_name == "w13_weight" else 2
                tensor = _state_dict_sharded_tensor(
                    state_dict,
                    name,
                    shard_dim,
                    tp_rank,
                    tp_size,
                )
            elif packed_name in (
                "w13_weight_scale",
                "w13_weight_offset",
                "w13_scale_bias",
            ):
                tensor = _state_dict_sharded_tensor(
                    state_dict,
                    name,
                    1,
                    tp_rank,
                    tp_size,
                )
            elif packed_name == "w2_scale_bias":
                tensor = _state_dict_sharded_tensor(
                    state_dict,
                    name,
                    2,
                    tp_rank,
                    tp_size,
                )
            elif len(name.split(".")) == 3:
                parts = name.split(".")
                projection = parts[1]
                suffix = parts[2]
                if projection in ("w1", "w3", "gate_proj", "up_proj"):
                    tensor = _state_dict_sharded_tensor(
                        state_dict,
                        name,
                        0,
                        tp_rank,
                        tp_size,
                    )
                elif suffix == "weight" or suffix == "scale_bias":
                    tensor = _state_dict_sharded_tensor(
                        state_dict,
                        name,
                        1,
                        tp_rank,
                        tp_size,
                    )
                else:
                    tensor = _state_dict_tensor(state_dict, name)
            else:
                continue
            if tensor is not None and self.load_weight(name, tensor):
                loaded.add(name)
        return loaded

    def finish_weight_loading(self) -> None:
        suffix_count = 4 if self.quantized else 1
        if self._packed_loaded:
            required_packed = {"w13_weight", "w2_weight"}
            if self.quantized:
                required_packed.update(
                    {
                        "w13_weight_scale",
                        "w13_weight_offset",
                        "w13_scale_bias",
                        "w2_weight_scale",
                        "w2_weight_offset",
                        "w2_scale_bias",
                    }
                )
            missing = required_packed.difference(self._packed_loaded)
            if missing:
                raise KeyError(
                    f"Kimi K3 packed expert weights are missing: {sorted(missing)}"
                )
            if self.quantized and self.w13_weight.device.type in (
                "npu",
                "privateuseone",
            ):
                self._process_quantized_weights()
            return
        if not bool(self._loaded_mask[:, :, :suffix_count].all()):
            raise KeyError("Kimi K3 routed expert weights are incomplete")
        if self.quantized and self.w13_weight.device.type in (
            "npu",
            "privateuseone",
        ):
            self._process_quantized_weights()


class KimiK3MoE(nn.Module):
    """Latent routed MoE with optional shared experts."""

    def __init__(
        self,
        config: Any,
        dtype: torch.dtype,
        device: torch.device,
        tp_size: int,
        tp_rank: int,
        routed_expert_down_proj: nn.Module,
        routed_expert_up_proj: nn.Module,
        shared_experts: nn.Module | None = None,
        quantized: bool = False,
    ) -> None:
        super().__init__()
        num_experts = int(config.num_experts)
        top_k = int(config.num_experts_per_token)
        routed_hidden_size = int(config.routed_expert_hidden_size)
        moe_intermediate_size = int(config.moe_intermediate_size)
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("Kimi K3 num_experts_per_token is invalid")
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = int(config.hidden_size)
        self.routed_hidden_size = routed_hidden_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.renormalize = bool(config.moe_renormalize)
        self.router_activation = str(config.moe_router_activation_func)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.situ_beta = float(getattr(config, "activation_situ_beta", None) or 1.0)
        self.situ_linear_beta = getattr(config, "activation_situ_linear_beta", None)
        self.quantized = quantized
        self._loaded_components: set[str] = set()
        self.gate = nn.Linear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=dtype, device=device)
        )
        self.routed_expert_down_proj = routed_expert_down_proj
        self.routed_expert_up_proj = routed_expert_up_proj
        self.routed_expert_norm = (
            nn.RMSNorm(
                routed_hidden_size,
                eps=float(config.rms_norm_eps),
                dtype=dtype,
                device=device,
            )
            if bool(getattr(config, "latent_moe_use_norm", False))
            else None
        )
        self.experts = KimiK3RoutedExperts(
            self.num_experts,
            routed_hidden_size,
            moe_intermediate_size,
            tp_size,
            dtype,
            device,
            quantized=quantized,
        )
        self.shared_experts = shared_experts

    def _topk(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.router_activation == "softmax":
            scores = torch.softmax(router_logits, dim=-1)
        elif self.router_activation == "sigmoid":
            scores = torch.sigmoid(router_logits)
        else:
            raise ValueError(f"Unsupported Kimi K3 router activation: {self.router_activation}")
        selection = scores + self.gate.e_score_correction_bias.to(dtype=scores.dtype)
        topk_selection = torch.topk(selection, self.top_k, dim=-1)
        topk_ids = topk_selection.indices
        topk_weights = scores.gather(-1, topk_ids)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        return topk_ids, topk_weights * self.routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        routed_input = self.routed_expert_down_proj(hidden_states)
        topk_ids, topk_weights = self._topk(self.gate(hidden_states))
        routed_output = self.experts(
            routed_input,
            topk_ids,
            topk_weights,
            self.situ_beta,
            self.situ_linear_beta,
        )
        if self.routed_expert_norm is not None:
            routed_output = self.routed_expert_norm(routed_output)
        output = self.routed_expert_up_proj(routed_output)
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        return output.reshape(original_shape)

    def load_weight(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> bool:
        if name == "gate.weight":
            if tensor.shape != self.gate.weight.shape:
                raise ValueError(
                    f"Kimi K3 gate expects {self.gate.weight.shape}, got {tensor.shape}"
                )
            self.gate.weight.data.copy_(tensor.to(self.gate.weight))
            self._loaded_components.add(name)
            return True
        if name in ("e_score_correction_bias", "gate.e_score_correction_bias"):
            target = self.gate.e_score_correction_bias
            if tensor.shape != target.shape:
                raise ValueError(
                    "Kimi K3 correction bias expects "
                    f"{target.shape}, got {tensor.shape}"
                )
            target.data.copy_(tensor.to(target))
            self._loaded_components.add("gate.e_score_correction_bias")
            return True
        for prefix, module in (
            ("routed_expert_down_proj.", self.routed_expert_down_proj),
            ("routed_expert_up_proj.", self.routed_expert_up_proj),
        ):
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if hasattr(module, "load_weight"):
                loaded = bool(module.load_weight(suffix, tensor))
                if loaded:
                    self._loaded_components.add(name)
                return loaded
            if suffix != "weight":
                return False
            target = module.weight
            if tensor.shape != target.shape:
                raise ValueError(
                    f"Kimi K3 weight {name} expects {target.shape}, "
                    f"got {tensor.shape}"
                )
            target.data.copy_(tensor.to(target))
            self._loaded_components.add(name)
            return True
        if name == "routed_expert_norm.weight" and self.routed_expert_norm is not None:
            self.routed_expert_norm.weight.data.copy_(tensor.to(self.routed_expert_norm.weight))
            self._loaded_components.add(name)
            return True
        return False

    def load_weights(
        self,
        state_dict: Any,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        loaded: set[str] = set()

        direct_names = [
            "gate.weight",
            "gate.e_score_correction_bias",
            "routed_expert_down_proj.weight",
            "routed_expert_up_proj.weight",
        ]
        if self.quantized:
            for projection in (
                "routed_expert_down_proj",
                "routed_expert_up_proj",
            ):
                direct_names.extend(
                    f"{projection}.{suffix}"
                    for suffix in ("weight_scale", "weight_offset")
                )
        if self.routed_expert_norm is not None:
            direct_names.append("routed_expert_norm.weight")
        for name in direct_names:
            tensor = _state_dict_tensor(state_dict, name)
            if tensor is not None and self.load_weight(name, tensor):
                loaded.add(name)

        shared_state_dict = _state_dict_with_prefix(state_dict, "shared_experts.")
        if self.shared_experts is not None and _state_dict_size(shared_state_dict) > 0:
            child_loaded = self.shared_experts.load_weights(
                shared_state_dict,
                tp_rank,
                tp_size,
            )
            loaded.update(f"shared_experts.{name}" for name in child_loaded)

        experts_state_dict = _state_dict_with_prefix(state_dict, "experts.")
        if _state_dict_size(experts_state_dict) > 0:
            child_loaded = self.experts.load_weights(
                experts_state_dict,
                tp_rank,
                tp_size,
            )
            loaded.update(f"experts.{name}" for name in child_loaded)
        return loaded

    def finish_weight_loading(self) -> None:
        required = {
            "gate.weight",
            "gate.e_score_correction_bias",
            "routed_expert_down_proj.weight",
            "routed_expert_up_proj.weight",
        }
        if self.quantized:
            for projection in (
                "routed_expert_down_proj",
                "routed_expert_up_proj",
            ):
                required.update(
                    f"{projection}.{suffix}"
                    for suffix in ("weight_scale", "weight_offset")
                )
        if self.routed_expert_norm is not None:
            required.add("routed_expert_norm.weight")
        missing = required.difference(self._loaded_components)
        if missing:
            raise KeyError(f"Kimi K3 MoE weights are missing: {sorted(missing)}")
        for module in (
            self.routed_expert_down_proj,
            self.routed_expert_up_proj,
            self.shared_experts,
            self.experts,
        ):
            if module is not None and hasattr(module, "finish_weight_loading"):
                module.finish_weight_loading()


__all__ = ["KimiK3MoE", "KimiK3RoutedExperts"]
