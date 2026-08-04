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

"""Kimi K3 text graph for xLLM's Python model executor.

KDA, MLA and vision execution are intentionally left outside this first text
graph.  The decoder still exposes an ``Attention`` layer so the executor can
construct its normal runtime contract; the placeholder preserves tensor
shapes until those two attention implementations are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import ops
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    HiddenParallelEmbedding,
    KimiK3MoE,
    RMSNorm,
    RowParallelLinear,
)
from xllm.python.models.base import PyModelBase


def _tp_rank_from_device(device: object) -> int:
    value = str(device)
    if ":" not in value:
        return 0
    try:
        return int(value.rsplit(":", 1)[-1])
    except ValueError:
        return 0


def _copy_parameter(parameter: torch.Tensor, tensor: torch.Tensor) -> None:
    if parameter.shape != tensor.shape:
        raise ValueError(
            f"Kimi K3 parameter expects {parameter.shape}, got {tensor.shape}"
        )
    parameter.data.copy_(tensor.to(dtype=parameter.dtype, device=parameter.device))


def _state_dict_size(state_dict: Any) -> int:
    if hasattr(state_dict, "size"):
        return int(state_dict.size())
    return len(state_dict.keys())


def _state_dict_with_prefix(
    state_dict: Any,
    prefix: str | list[str],
) -> Any:
    if isinstance(prefix, list):
        return state_dict.get_dict_with_prefixes(prefix)
    return state_dict.get_dict_with_prefix(prefix)


def _state_dict_tensor(state_dict: Any, name: str) -> torch.Tensor | None:
    if not state_dict.has(name):
        return None
    return state_dict.get_tensor(name)


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


def _layer_ids(state_dict: Any) -> list[int]:
    layer_ids: set[int] = set()
    for name in state_dict.keys():
        parts = name.split(".", 2)
        if len(parts) < 2 or parts[0] != "layers":
            continue
        try:
            layer_ids.add(int(parts[1]))
        except ValueError:
            continue
    return sorted(layer_ids)


@dataclass
class KimiK3TextConfig:
    hidden_size: int = 7168
    n_layers: int = 93
    n_heads: int = 96
    n_kv_heads: int = 96
    head_dim: int = 128
    intermediate_size: int = 33792
    vocab_size: int = 163840
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 1048576
    hidden_act: str = "situ"
    tie_word_embeddings: bool = False
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    num_experts: int | None = 896
    num_experts_per_token: int | None = 16
    num_shared_experts: int = 2
    moe_intermediate_size: int | None = 3072
    routed_expert_hidden_size: int | None = 3584
    moe_renormalize: bool = True
    moe_router_activation_func: str = "sigmoid"
    routed_scaling_factor: float = 1.0
    latent_moe_use_norm: bool = True
    activation_situ_beta: float | None = 4.0
    activation_situ_linear_beta: float | None = 25.0
    attn_res_block_size: int = 12
    quantize_type: str = ""
    quant_method: str = ""
    quant_version: str = ""
    quant_group_size: int = 0
    tp_size: int = 1
    tp_rank: int = 0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "KimiK3TextConfig":
        raw = config.get("text_config", config)
        if not isinstance(raw, dict):
            raise TypeError("Kimi K3 text_config must be a dictionary")

        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in raw and raw[key] is not None:
                    return raw[key]
            for key in keys:
                if key in config and config[key] is not None:
                    return config[key]
            return default

        linear_attention = raw.get("linear_attn_config", {})
        if not isinstance(linear_attention, dict):
            linear_attention = {}
        n_heads = int(pick("n_heads", "num_attention_heads", default=96))
        hidden_size = int(pick("hidden_size", default=7168))
        configured_head_dim = pick("head_dim")
        head_dim = int(
            linear_attention.get("head_dim", 128)
            if configured_head_dim is None
            else configured_head_dim
        )
        return cls(
            hidden_size=hidden_size,
            n_layers=int(pick("n_layers", "num_hidden_layers", default=93)),
            n_heads=n_heads,
            n_kv_heads=int(pick("n_kv_heads", "num_key_value_heads", default=n_heads)),
            head_dim=head_dim,
            intermediate_size=int(pick("intermediate_size", default=33792)),
            vocab_size=int(pick("vocab_size", default=163840)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-5)),
            max_position_embeddings=int(
                pick("max_position_embeddings", default=1048576)
            ),
            hidden_act=str(pick("hidden_act", default="situ")),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=False)),
            first_k_dense_replace=int(pick("first_k_dense_replace", default=1)),
            moe_layer_freq=int(pick("moe_layer_freq", default=1)),
            num_experts=(
                None
                if pick("num_experts", "n_routed_experts", default=896) is None
                else int(pick("num_experts", "n_routed_experts", default=896))
            ),
            num_experts_per_token=(
                None
                if pick("num_experts_per_token", "num_experts_per_tok", default=16) is None
                else int(pick("num_experts_per_token", "num_experts_per_tok", default=16))
            ),
            num_shared_experts=int(pick("num_shared_experts", "n_shared_experts", default=2)),
            moe_intermediate_size=(
                None
                if pick("moe_intermediate_size", default=3072) is None
                else int(pick("moe_intermediate_size", default=3072))
            ),
            routed_expert_hidden_size=(
                None
                if pick("routed_expert_hidden_size", default=3584) is None
                else int(pick("routed_expert_hidden_size", default=3584))
            ),
            moe_renormalize=bool(pick("moe_renormalize", "norm_topk_prob", default=True)),
            moe_router_activation_func=str(
                pick("moe_router_activation_func", default="sigmoid")
            ),
            routed_scaling_factor=float(pick("routed_scaling_factor", default=1.0)),
            latent_moe_use_norm=bool(pick("latent_moe_use_norm", default=True)),
            activation_situ_beta=(
                None
                if pick("activation_situ_beta", default=4.0) is None
                else float(pick("activation_situ_beta", default=4.0))
            ),
            activation_situ_linear_beta=(
                None
                if pick("activation_situ_linear_beta", default=25.0) is None
                else float(pick("activation_situ_linear_beta", default=25.0))
            ),
            attn_res_block_size=int(pick("attn_res_block_size", default=12)),
            quantize_type=str(config.get("quantize_type", "")),
            quant_method=str(config.get("quant_method", "")),
            quant_version=str(config.get("quant_version", "")),
            quant_group_size=int(config.get("quant_group_size", 0)),
            tp_size=int(config.get("tp_size", raw.get("tp_size", 1))),
            tp_rank=int(config.get("tp_rank", raw.get("tp_rank", 0))),
        )

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.hidden_size % self.tp_size != 0:
            raise ValueError("Kimi K3 hidden_size must be positive and divisible by tp_size")
        if self.vocab_size <= 0 or self.vocab_size % self.tp_size != 0:
            raise ValueError("Kimi K3 vocab_size must be positive and divisible by tp_size")
        if self.n_layers <= 0 or self.n_heads <= 0 or self.head_dim <= 0:
            raise ValueError("Kimi K3 layer and attention dimensions must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("Kimi K3 rms_norm_eps must be positive")
        if self.n_heads % self.tp_size != 0:
            raise ValueError("Kimi K3 attention heads must be divisible by tp_size")
        if self.n_kv_heads % self.tp_size != 0:
            raise ValueError("Kimi K3 key/value heads must be divisible by tp_size")
        if self.intermediate_size <= 0 or self.intermediate_size % self.tp_size != 0:
            raise ValueError("Kimi K3 intermediate_size must be divisible by tp_size")
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("Kimi K3 TP rank and size are invalid")
        if self.moe_layer_freq <= 0 or self.first_k_dense_replace < 0:
            raise ValueError("Kimi K3 MoE layer placement is invalid")
        if self.attn_res_block_size <= 0:
            raise ValueError("Kimi K3 attn_res_block_size must be positive")
        if self.activation_situ_beta is not None and self.activation_situ_beta <= 0:
            raise ValueError("Kimi K3 activation_situ_beta must be positive")
        if (
            self.activation_situ_linear_beta is not None
            and self.activation_situ_linear_beta <= 0
        ):
            raise ValueError("Kimi K3 activation_situ_linear_beta must be positive")
        if self.hidden_act not in ("situ", "silu"):
            raise ValueError(f"Unsupported Kimi K3 activation: {self.hidden_act}")
        if self.moe_router_activation_func not in ("sigmoid", "softmax"):
            raise ValueError(
                "Kimi K3 router activation must be sigmoid or softmax"
            )
        if self.num_experts is not None:
            if self.num_experts_per_token is None or self.moe_intermediate_size is None:
                raise ValueError("Kimi K3 MoE dimensions are incomplete")
            if self.routed_expert_hidden_size is None:
                raise ValueError("Kimi K3 routed_expert_hidden_size is required")
            if self.moe_intermediate_size % self.tp_size != 0:
                raise ValueError("Kimi K3 MoE intermediate_size must divide tp_size")
            if not 0 < self.num_experts_per_token <= self.num_experts:
                raise ValueError(
                    "Kimi K3 num_experts_per_token must be within num_experts"
                )
        if self.uses_quantized_weights:
            if self.quant_version != "1.0.0":
                raise ValueError(
                    "Kimi K3 W4A8 weights require quant_version 1.0.0"
                )
            if self.quant_group_size != 0:
                raise ValueError(
                    "Kimi K3 currently supports per-channel W4A8 weights only"
                )
            if (
                self.num_experts is None
                or self.routed_expert_hidden_size is None
                or self.moe_intermediate_size is None
            ):
                raise ValueError("Kimi K3 W4A8 requires routed experts")
            if 16 % self.tp_size != 0:
                raise ValueError("Kimi K3 W4A8 scale_bias requires tp_size <= 16")
            if self.routed_expert_hidden_size % 2 != 0:
                raise ValueError(
                    "Kimi K3 W4A8 routed hidden size must be even"
                )
            if self.moe_intermediate_size % (2 * self.tp_size) != 0:
                raise ValueError(
                    "Kimi K3 W4A8 expert size must be divisible by 2 * tp_size"
                )

    @property
    def uses_quantized_weights(self) -> bool:
        quantize_type = self.quantize_type.lower()
        quant_method = self.quant_method.lower()
        return quantize_type == "w4a8_dynamic" or quant_method == "ascend_int4"

    def is_moe_layer(self, layer_id: int) -> bool:
        return (
            self.num_experts is not None
            and layer_id >= self.first_k_dense_replace
            and layer_id % self.moe_layer_freq == 0
        )


class KimiK3W8A8DynamicLinear(nn.Module):
    """Kimi-only dynamic W8A8 linear with module-local loading."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device,
        tp_size: int = 1,
        reduce_results: bool = False,
        gather_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.reduce_results = reduce_results
        self.gather_output = gather_output
        self._processed = False
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=torch.int8,
                device=device,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(out_features, 1, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "weight_offset",
            torch.empty(out_features, 1, dtype=torch.float32, device=device),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._processed:
            raise RuntimeError("Kimi K3 W8A8 weights have not finished loading")
        quantized, per_token_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
        output = ops.quant_matmul(
            quantized,
            self.weight,
            False,
            self.weight_scale,
            None,
            per_token_scale,
            None,
            hidden_states.dtype,
        )
        if self.reduce_results and self.tp_size > 1:
            ops.all_reduce_(output)
        if self.gather_output and self.tp_size > 1:
            output = ops.all_gather(output, dim=-1, world_size=self.tp_size)
        return output

    def load_weight(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> bool:
        targets = {
            "weight": self.weight,
            "weight_scale": self.weight_scale,
            "weight_offset": self.weight_offset,
        }
        target = targets.get(name)
        if target is None:
            return False
        _copy_parameter(target, tensor)
        return True

    def finish_weight_loading(self) -> None:
        if self._processed:
            return
        self.weight.data = self.weight.data.transpose(0, 1).contiguous()
        self.weight_scale.data = self.weight_scale.data.flatten().contiguous()
        self.weight_offset.data = self.weight_offset.data.flatten().contiguous()
        self._processed = True


class KimiK3MLP(nn.Module):
    """Dense Kimi gated MLP with module-local weight loading."""

    def __init__(
        self,
        config: KimiK3TextConfig,
        dtype: torch.dtype,
        device: torch.device,
        intermediate_size: int | None = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        if intermediate_size % config.tp_size != 0:
            raise ValueError("Kimi K3 MLP intermediate_size must divide tp_size")
        intermediate_per_rank = intermediate_size // config.tp_size
        self.tp_size = config.tp_size
        self.quantized = config.uses_quantized_weights
        if self.quantized:
            self.gate_up_proj = KimiK3W8A8DynamicLinear(
                config.hidden_size,
                2 * intermediate_per_rank,
                device,
                tp_size=config.tp_size,
            )
            self.down_proj = KimiK3W8A8DynamicLinear(
                intermediate_per_rank,
                config.hidden_size,
                device,
                tp_size=config.tp_size,
                reduce_results=reduce_results,
            )
        else:
            self.gate_up_proj = ColumnParallelLinear(
                config.hidden_size,
                2 * intermediate_per_rank,
                config.tp_size,
                dtype=dtype,
                device=device,
            )
            self.down_proj = RowParallelLinear(
                intermediate_per_rank,
                config.hidden_size,
                config.tp_size,
                dtype=dtype,
                device=device,
            )
        self.reduce_results = reduce_results
        self.hidden_act = config.hidden_act
        self.situ_beta = float(config.activation_situ_beta or 1.0)
        self.situ_linear_beta = config.activation_situ_linear_beta
        self._loaded_components: set[str] = set()

    def _activation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.hidden_act == "situ":
            width = tensor.shape[-1] // 2
            gate, up = tensor[..., :width], tensor[..., width:]
            gate = gate.float()
            up = up.float()
            gate = self.situ_beta * torch.tanh(gate / self.situ_beta) * torch.sigmoid(gate)
            if self.situ_linear_beta is not None:
                up = self.situ_linear_beta * torch.tanh(up / self.situ_linear_beta)
            return (gate * up).to(tensor.dtype)
        if self.hidden_act == "silu":
            return ops.silu_and_mul(tensor)
        raise ValueError(f"Unsupported Kimi K3 activation: {self.hidden_act}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        activated = self._activation(self.gate_up_proj(hidden_states))
        if self.quantized or self.reduce_results:
            return self.down_proj(activated)
        output = F.linear(activated, self.down_proj.weight, self.down_proj.bias)
        return output

    def load_weight(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> bool:
        projection, separator, suffix = name.partition(".")
        if projection in ("gate_proj", "up_proj") and separator:
            if suffix not in ("weight", "weight_scale", "weight_offset"):
                return False
            if not self.quantized and suffix != "weight":
                return False
            target_tensor = getattr(self.gate_up_proj, suffix)
            half = target_tensor.shape[0] // 2
            target = (
                target_tensor.data[:half]
                if projection == "gate_proj"
                else target_tensor.data[half:]
            )
            _copy_parameter(target, tensor)
            self._loaded_components.add(f"{projection}.{suffix}")
            return True
        if projection == "gate_up_proj" and separator:
            if self.quantized:
                loaded = self.gate_up_proj.load_weight(
                    suffix,
                    tensor,
                )
                if loaded:
                    self._loaded_components.add(f"{projection}.{suffix}")
                return loaded
            if suffix != "weight":
                return False
            _copy_parameter(self.gate_up_proj.weight, tensor)
            self._loaded_components.update(
                {"gate_proj.weight", "up_proj.weight"}
            )
            return True
        if projection == "down_proj" and separator:
            if self.quantized:
                loaded = self.down_proj.load_weight(
                    suffix,
                    tensor,
                )
                if loaded:
                    self._loaded_components.add(f"down_proj.{suffix}")
                return loaded
            if suffix != "weight":
                return False
            _copy_parameter(self.down_proj.weight, tensor)
            self._loaded_components.add("down_proj.weight")
            return True
        return False

    def load_weights(
        self,
        state_dict: Any,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        loaded: set[str] = set()
        suffixes = ["weight"]
        if self.quantized:
            suffixes.extend(["weight_scale", "weight_offset"])
        for projection in ("gate_proj", "up_proj"):
            for suffix in suffixes:
                name = f"{projection}.{suffix}"
                tensor = _state_dict_sharded_tensor(
                    state_dict,
                    name,
                    0,
                    tp_rank,
                    tp_size,
                )
                if tensor is not None and self.load_weight(name, tensor):
                    loaded.add(name)
        for suffix in suffixes:
            name = f"down_proj.{suffix}"
            tensor = (
                _state_dict_sharded_tensor(
                    state_dict,
                    name,
                    1,
                    tp_rank,
                    tp_size,
                )
                if suffix == "weight"
                else _state_dict_tensor(state_dict, name)
            )
            if tensor is not None and self.load_weight(name, tensor):
                loaded.add(name)
        if state_dict.has("gate_up_proj.weight"):
            tensor = _state_dict_sharded_tensor(
                state_dict,
                "gate_up_proj.weight",
                0,
                tp_rank,
                tp_size,
            )
            if tensor is not None and self.load_weight("gate_up_proj.weight", tensor):
                loaded.add("gate_up_proj.weight")
        return loaded

    def finish_weight_loading(self) -> None:
        suffixes = ["weight"]
        if self.quantized:
            suffixes.extend(["weight_scale", "weight_offset"])
        required = {
            f"{projection}.{suffix}"
            for projection in ("gate_proj", "up_proj", "down_proj")
            for suffix in suffixes
        }
        if "gate_up_proj.weight" in self._loaded_components:
            required.difference_update({"gate_proj.weight", "up_proj.weight"})
        missing = required.difference(self._loaded_components)
        if missing:
            raise KeyError(f"Kimi K3 MLP weights are missing: {sorted(missing)}")
        if not self.quantized:
            if self.down_proj.tp_size > 1 and self.down_proj.weight.device.type in (
                "npu",
                "privateuseone",
            ):
                self.down_proj.format_npu_weight_()
            return
        self.gate_up_proj.finish_weight_loading()
        self.down_proj.finish_weight_loading()


class KimiK3AttentionPlaceholder(Attention):
    """Shape-compatible attention shell until KDA/MLA are implemented."""

    def __init__(self, config: KimiK3TextConfig, layer_id: int) -> None:
        super().__init__(
            num_heads=config.n_heads // config.tp_size,
            num_kv_heads=config.n_kv_heads // config.tp_size,
            head_dim=config.head_dim,
            scale=config.head_dim**-0.5,
            sliding_window=0,
            layer_id=layer_id,
        )

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del positions
        return torch.zeros_like(hidden_states)


def _apply_attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: nn.Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_float = values.float()
    normed = values_float * torch.rsqrt(
        values_float.square().mean(dim=-1, keepdim=True) + norm.eps
    )
    normed = normed * norm.weight.float()
    scores = F.linear(normed, projection.weight.float()).squeeze(-1)
    probabilities = torch.softmax(scores, dim=-1).unsqueeze(-1)
    return (probabilities * values_float).sum(dim=1).to(values.dtype)


class KimiK3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiK3TextConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.self_attn = KimiK3AttentionPlaceholder(config, layer_id)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        if config.is_moe_layer(layer_id):
            if config.uses_quantized_weights:
                routed_expert_down_proj = KimiK3W8A8DynamicLinear(
                    config.hidden_size,
                    config.routed_expert_hidden_size,
                    device,
                )
                routed_expert_up_proj = KimiK3W8A8DynamicLinear(
                    config.routed_expert_hidden_size,
                    config.hidden_size,
                    device,
                )
            else:
                routed_expert_down_proj = nn.Linear(
                    config.hidden_size,
                    config.routed_expert_hidden_size,
                    bias=False,
                    dtype=dtype,
                    device=device,
                )
                routed_expert_up_proj = nn.Linear(
                    config.routed_expert_hidden_size,
                    config.hidden_size,
                    bias=False,
                    dtype=dtype,
                    device=device,
                )
            shared_experts = (
                KimiK3MLP(
                    config,
                    dtype,
                    device,
                    intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
                    reduce_results=False,
                )
                if config.num_shared_experts
                else None
            )
            self.block_sparse_moe = KimiK3MoE(
                config,
                dtype,
                device,
                config.tp_size,
                config.tp_rank,
                routed_expert_down_proj=routed_expert_down_proj,
                routed_expert_up_proj=routed_expert_up_proj,
                shared_experts=shared_experts,
                quantized=config.uses_quantized_weights,
            )
        else:
            self.mlp = KimiK3MLP(config, dtype, device)
        self.attn_res_block_size = config.attn_res_block_size
        self.self_attention_res_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.mlp_res_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.self_attention_res_proj = nn.Linear(
            config.hidden_size,
            1,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.mlp_res_proj = nn.Linear(
            config.hidden_size,
            1,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self._loaded_components: set[str] = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_sum: torch.Tensor | None = hidden_states
        if block_residual.shape[1] > 0:
            hidden_states = _apply_attention_residual(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            )
        if self.layer_id % self.attn_res_block_size == 0:
            block_residual = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
            prefix_sum = None
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.self_attn(hidden_states, positions)
        prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output
        hidden_states = _apply_attention_residual(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = (
            self.block_sparse_moe(hidden_states)
            if hasattr(self, "block_sparse_moe")
            else self.mlp(hidden_states)
        )
        return prefix_sum + hidden_states, block_residual

    def load_weights(
        self,
        state_dict: Any,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        loaded: set[str] = set()
        targets: dict[str, torch.Tensor] = {
            "input_layernorm.weight": self.input_layernorm.weight,
            "post_attention_layernorm.weight": self.post_attention_layernorm.weight,
            "self_attention_res_norm.weight": self.self_attention_res_norm.weight,
            "mlp_res_norm.weight": self.mlp_res_norm.weight,
            "self_attention_res_proj.weight": self.self_attention_res_proj.weight,
            "mlp_res_proj.weight": self.mlp_res_proj.weight,
        }
        for name, target in targets.items():
            tensor = _state_dict_tensor(state_dict, name)
            if tensor is not None:
                _copy_parameter(target, tensor)
                self._loaded_components.add(name)
                loaded.add(name)

        if hasattr(self, "mlp"):
            child_state_dict = _state_dict_with_prefix(state_dict, "mlp.")
            if _state_dict_size(child_state_dict) > 0:
                loaded.update(
                    f"mlp.{name}"
                    for name in self.mlp.load_weights(
                        child_state_dict,
                        tp_rank,
                        tp_size,
                    )
                )
        else:
            child_state_dict = _state_dict_with_prefix(
                state_dict,
                "block_sparse_moe.",
            )
            if _state_dict_size(child_state_dict) > 0:
                loaded.update(
                    f"block_sparse_moe.{name}"
                    for name in self.block_sparse_moe.load_weights(
                        child_state_dict,
                        tp_rank,
                        tp_size,
                    )
                )
        return loaded

    def finish_weight_loading(self) -> None:
        required = {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attention_res_norm.weight",
            "mlp_res_norm.weight",
            "self_attention_res_proj.weight",
            "mlp_res_proj.weight",
        }
        missing = required.difference(self._loaded_components)
        if missing:
            raise KeyError(
                f"Kimi K3 decoder layer {self.layer_id} weights are missing: "
                f"{sorted(missing)}"
            )
        if hasattr(self, "mlp"):
            self.mlp.finish_weight_loading()
        else:
            self.block_sparse_moe.finish_weight_loading()


class KimiK3TextModel(nn.Module):
    def __init__(self, config: KimiK3TextConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = HiddenParallelEmbedding(
            config.vocab_size,
            config.hidden_size // config.tp_size,
            config.tp_size,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList(
            [KimiK3DecoderLayer(config, i, dtype, device) for i in range(config.n_layers)]
        )
        self.output_attn_res_norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.output_attn_res_proj = nn.Linear(
            config.hidden_size,
            1,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)
        self._loaded_weights: set[str] = set()

    def initial_block_count(self) -> int:
        return sum(i % self.config.attn_res_block_size == 0 for i in range(self.config.n_layers))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        block_residual = hidden_states.new_zeros(
            (hidden_states.shape[0], 0, hidden_states.shape[-1])
        )
        for layer in self.layers:
            hidden_states, block_residual = layer(hidden_states, positions, block_residual)
        hidden_states = _apply_attention_residual(
            hidden_states,
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        )
        return self.norm(hidden_states)

    def load_weights(
        self,
        state_dict: Any,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        loaded: set[str] = set()
        direct_targets = {
            "embed_tokens.weight": self.embed_tokens.weight,
            "output_attn_res_norm.weight": self.output_attn_res_norm.weight,
            "output_attn_res_proj.weight": self.output_attn_res_proj.weight,
            "norm.weight": self.norm.weight,
        }
        for name, target in direct_targets.items():
            tensor = (
                _state_dict_sharded_tensor(
                    state_dict,
                    name,
                    1,
                    tp_rank,
                    tp_size,
                )
                if name == "embed_tokens.weight"
                else _state_dict_tensor(state_dict, name)
            )
            if tensor is not None:
                _copy_parameter(target, tensor)
                self._loaded_weights.add(name)
                loaded.add(f"model.{name}")

        for layer_id in _layer_ids(state_dict):
            if not 0 <= layer_id < len(self.layers):
                continue
            layer_state_dict = _state_dict_with_prefix(
                state_dict,
                f"layers.{layer_id}.",
            )
            if _state_dict_size(layer_state_dict) == 0:
                continue
            loaded.update(
                f"model.layers.{layer_id}.{name}"
                for name in self.layers[layer_id].load_weights(
                    layer_state_dict,
                    tp_rank,
                    tp_size,
                )
            )
        return loaded

    def finish_weight_loading(self) -> None:
        required = {
            "embed_tokens.weight",
            "output_attn_res_norm.weight",
            "output_attn_res_proj.weight",
            "norm.weight",
        }
        missing = required.difference(self._loaded_weights)
        if missing:
            raise KeyError(f"Kimi K3 text model weights are missing: {sorted(missing)}")
        for layer in self.layers:
            layer.finish_weight_loading()


class KimiK3ForCausalLM(PyModelBase):
    """Top-level Python model entry for Kimi K3 text checkpoints."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = KimiK3TextConfig.from_dict(config)
        self.cfg.tp_size = int(config.get("tp_size", self.cfg.tp_size))
        self.cfg.tp_rank = int(
            config.get("tp_rank", _tp_rank_from_device(config.get("device", "npu:0")))
        )
        self.cfg.validate()
        self.dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        self.device = torch.device(config.get("device", "cuda"))
        self.model = KimiK3TextModel(self.cfg, self.dtype, self.device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size,
            gather_output=True,
            dtype=self.dtype,
            device=self.device,
        )
        self._lm_head_loaded = False

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def load_weights(self, state_dicts: list[Any], tp_rank: int, tp_size: int) -> set[str]:
        if tp_rank != self.cfg.tp_rank or tp_size != self.cfg.tp_size:
            raise ValueError("Kimi K3 loader TP rank/size must match model construction")
        loaded: set[str] = set()
        model_prefixes = ["language_model.model.", "model.", ""]
        for state_dict in state_dicts:
            model_state_dict = _state_dict_with_prefix(
                state_dict,
                model_prefixes,
            )
            loaded.update(
                self.model.load_weights(model_state_dict, tp_rank, tp_size)
            )

            if self.cfg.tie_word_embeddings:
                lm_state_dict = model_state_dict
                lm_weight_name = "embed_tokens.weight"
            else:
                lm_state_dict = _state_dict_with_prefix(
                    state_dict,
                    [
                        "language_model.lm_head.",
                        "lm_head.",
                        "model.lm_head.",
                        "head.",
                        "",
                    ],
                )
                lm_weight_name = "weight"
            tensor = _state_dict_sharded_tensor(
                lm_state_dict,
                lm_weight_name,
                0,
                tp_rank,
                tp_size,
            )
            if tensor is not None:
                _copy_parameter(self.lm_head.weight, tensor)
                self._lm_head_loaded = True
                loaded.add("lm_head.weight")

        self.model.finish_weight_loading()
        if not self._lm_head_loaded:
            raise KeyError("Kimi K3 lm_head weight is missing")
        return loaded


__all__ = [
    "KimiK3AttentionPlaceholder",
    "KimiK3DecoderLayer",
    "KimiK3ForCausalLM",
    "KimiK3MLP",
    "KimiK3TextConfig",
    "KimiK3TextModel",
]
