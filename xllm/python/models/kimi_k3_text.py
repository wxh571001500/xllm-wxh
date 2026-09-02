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

"""Kimi K3 text graph for xLLM's Python model executor.

The decoder supports heterogeneous KDA and Gated-MLA layers. Full-attention
layers use the shared Gated-MLA math through the runtime attention backend;
a placeholder remains only as a defensive fallback for incomplete layer maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn

from xllm.python import ops
from xllm.python.attention.backend import MlaUnabsorbedPrefill
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    HiddenParallelEmbedding,
    KimiK3MoE,
    RMSNorm,
    RowParallelLinear,
    W8A8DynamicLinearMethod,
)
from xllm.python.layers.attention import AttentionRuntimeLayer
from xllm.python.layers.kda import KimiK3DeltaAttention, KimiK3KDAMetadata
from xllm.python.layers.moe.types import MoECommType
from xllm.python.model_executor.forward_context import get_forward_context
from xllm.python.models.base import PyModelBase
from xllm.python.models.kimi_k3_gated_mla import (
    KimiK3GatedMLA,
    KimiK3GatedMLAConfig,
)

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


def _tp_rank_from_device(device: object) -> int:
    """Extract TP rank from device (e.g., 'npu:3' -> 3).

    Raises ValueError if the device does not include a rank index.
    """
    dev = torch.device(device)
    if dev.index is None:
        raise ValueError(f"Device must include rank index (e.g., 'npu:0'), got: {device!r}")
    return dev.index


def _resolve_dp_rank(config: dict[str, Any]) -> int:
    """Derive the DP rank from the C++ parallel properties in the config dict.

    The C++ side defines the TP group as a contiguous block of
    ``world_size // dp_size`` ranks, so the DP rank is the global rank divided
    by that block size. Returns 0 when DP is disabled or the world layout is
    absent (e.g. CPU test configs).
    """
    raw = config.get("text_config", config)
    dp_size = int(config.get("dp_size", raw.get("dp_size", 1)) or 1)
    if dp_size <= 1:
        return 0
    world_size = int(config.get("world_size", raw.get("world_size", 0)) or 0)
    global_rank = int(config.get("rank", raw.get("rank", 0)) or 0)
    tp_block = world_size // dp_size if world_size > 0 else 0
    if tp_block <= 0:
        return 0
    return global_rank // tp_block


def _copy_parameter(parameter: torch.Tensor, tensor: torch.Tensor) -> None:
    if parameter.shape != tensor.shape:
        raise ValueError(f"Kimi K3 parameter expects {parameter.shape}, got {tensor.shape}")
    parameter.data.copy_(tensor.to(dtype=parameter.dtype, device=parameter.device))


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
        raise ValueError(f"Kimi K3 tensor dimension {tensor.shape[dim]} is not divisible by tp_size {tp_size}")
    shard_size = tensor.shape[dim] // tp_size
    return tensor.narrow(dim, tp_rank * shard_size, shard_size).contiguous()


def _layer_ids(state_dict: StateDict) -> list[int]:
    layer_ids: set[int] = set()
    for name in state_dict.keys():  # noqa: SIM118
        parts = name.split(".", 2)
        if len(parts) < 2 or parts[0] != "layers":
            continue
        try:
            layer_ids.add(int(parts[1]))
        except ValueError:
            continue
    return sorted(layer_ids)


class _MergedStateDict:
    """Cross-shard view over several per-file ``StateDict`` objects.

    The checkpoint splits a single layer's tensors across many safetensors
    shards (e.g. KDA ``self_attn.q_proj`` and ``o_proj`` live in different
    files). Loaders such as KDA require every tensor of a layer in one pass,
    so we expose the union of all shards behind the same interface a single
    ``StateDict`` provides and load each weight exactly once.

    A name -> (shard, full_name) index is built once and filtered per prefix
    view, so lookups stay O(1) instead of scanning every shard per access.
    """

    def __init__(
        self,
        shards: list[Any] | None = None,
        index: dict[str, tuple[Any, str]] | None = None,
    ) -> None:
        self._shards = shards or []
        # Maps the exposed (prefix-stripped) name to (shard, full name).
        self._index = index

    def _ensure_index(self) -> dict[str, tuple[Any, str]]:
        if self._index is None:
            self._index = {}
            for shard in self._shards:
                for name in shard.keys():  # noqa: SIM118
                    self._index.setdefault(name, (shard, name))
        return self._index

    def has(self, name: str) -> bool:
        return name in self._ensure_index()

    def _resolve(self, name: str) -> tuple[Any, str]:
        entry = self._ensure_index().get(name)
        if entry is None:
            raise KeyError(f"missing checkpoint weight: {name}")
        return entry

    def get_tensor(self, name: str) -> torch.Tensor:
        shard, full_name = self._resolve(name)
        return shard.get_tensor(full_name)

    def get_sharded_tensor(
        self,
        name: str,
        dim: int,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        shard, full_name = self._resolve(name)
        return shard.get_sharded_tensor(full_name, dim, rank, world_size)

    def keys(self) -> list[str]:
        return list(self._ensure_index())

    def size(self) -> int:
        return len(self._ensure_index())

    def get_dict_with_prefix(self, prefix: str) -> _MergedStateDict:
        if not prefix:
            return _MergedStateDict(index=dict(self._ensure_index()))
        index = {name[len(prefix) :]: entry for name, entry in self._ensure_index().items() if name.startswith(prefix)}
        return _MergedStateDict(index=index)

    def get_dict_with_prefixes(self, prefixes: list[str]) -> _MergedStateDict:
        for prefix in prefixes:
            merged = self.get_dict_with_prefix(prefix)
            if merged.size() > 0:
                return merged
        return _MergedStateDict(index={})


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
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1
    topk_method: str = "noaux_tc"
    moe_intermediate_size: int | None = 3072
    routed_expert_hidden_size: int | None = 3584
    moe_renormalize: bool = True
    moe_router_activation_func: str = "sigmoid"
    routed_scaling_factor: float = 1.0
    latent_moe_use_norm: bool = True
    activation_situ_beta: float | None = 4.0
    activation_situ_linear_beta: float | None = 25.0
    attn_res_block_size: int = 12
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    mla_use_rope: bool = False
    mla_use_output_gate: bool = True
    num_nextn_predict_layers: int = 0
    logit_scale: float | None = None
    linear_attn_config: dict = field(default_factory=dict)
    quantize_type: str = ""
    quant_method: str = ""
    quant_version: str = ""
    quant_group_size: int = 0
    tp_size: int = 1
    tp_rank: int = 0
    world_size: int = 1
    rank: int = 0
    ep_size: int = 1
    dp_size: int = 1
    dp_rank: int = 0
    moe_comm_type: str = "all_gather"
    mc2_tokens_capacity: int = 512
    enable_flashcomm1: bool = False
    enable_prefix_cache: bool = True

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> KimiK3TextConfig:
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
        head_dim = int(linear_attention.get("head_dim", 128) if configured_head_dim is None else configured_head_dim)
        _ws = int(config.get("world_size", raw.get("world_size", 0)) or 0)
        _dp = int(config.get("dp_size", raw.get("dp_size", 1)) or 1)
        _tp = int(config.get("tp_size") or raw.get("tp_size") or 0)
        if _tp <= 1 and _ws > 1 and _dp > 0:
            _tp = _ws // _dp
        _g_rank = int(config.get("rank") or raw.get("rank") or 0)
        _tp_rank = int(config.get("tp_rank") or raw.get("tp_rank") or 0)
        if _tp_rank == 0 and _g_rank > 0 and _tp > 1:
            _tp_rank = _g_rank % _tp
        return cls(
            hidden_size=hidden_size,
            n_layers=int(pick("n_layers", "num_hidden_layers", default=93)),
            n_heads=n_heads,
            n_kv_heads=int(pick("n_kv_heads", "num_key_value_heads", default=n_heads)),
            head_dim=head_dim,
            intermediate_size=int(pick("intermediate_size", default=33792)),
            vocab_size=int(pick("vocab_size", default=163840)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-5)),
            max_position_embeddings=int(pick("max_position_embeddings", default=1048576)),
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
            use_grouped_topk=bool(pick("use_grouped_topk", default=True)),
            num_expert_group=int(pick("num_expert_group", "n_group", default=1)),
            topk_group=int(pick("topk_group", default=1)),
            topk_method=str(pick("topk_method", default="noaux_tc")),
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
            moe_router_activation_func=str(pick("moe_router_activation_func", default="sigmoid")),
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
            q_lora_rank=int(pick("q_lora_rank", default=1536)),
            kv_lora_rank=int(pick("kv_lora_rank", default=512)),
            qk_nope_head_dim=int(pick("qk_nope_head_dim", default=128)),
            qk_rope_head_dim=int(pick("qk_rope_head_dim", default=64)),
            v_head_dim=int(pick("v_head_dim", default=128)),
            mla_use_nope=bool(pick("mla_use_nope", default=True)),
            mla_use_rope=bool(pick("mla_use_rope", default=False)),
            mla_use_output_gate=bool(pick("mla_use_output_gate", default=True)),
            num_nextn_predict_layers=int(pick("num_nextn_predict_layers", default=0)),
            logit_scale=(
                None if pick("logit_scale", default=None) is None else float(pick("logit_scale", default=None))
            ),
            linear_attn_config=dict(linear_attention),
            quantize_type=str(config.get("quantize_type", "")),
            quant_method=str(config.get("quant_method", "")),
            quant_version=str(config.get("quant_version", "")),
            quant_group_size=int(config.get("quant_group_size", 0)),
            tp_size=_tp,
            tp_rank=_tp_rank,
            world_size=int(
                config.get(
                    "world_size",
                    raw.get(
                        "world_size",
                        config.get("tp_size", raw.get("tp_size", 1)),
                    ),
                )
            ),
            rank=int(
                config.get(
                    "rank",
                    raw.get(
                        "rank",
                        config.get("tp_rank", raw.get("tp_rank", 0)),
                    ),
                )
            ),
            ep_size=int(config.get("ep_size", raw.get("ep_size", 1))),
            dp_size=int(config.get("dp_size", raw.get("dp_size", 1))),
            dp_rank=_resolve_dp_rank(config),
            moe_comm_type=str(pick("moe_comm_type", "moe_communication", default="all_gather")),
            mc2_tokens_capacity=int(pick("mc2_tokens_capacity", default=512)),
            enable_flashcomm1=bool(pick("enable_flashcomm1", default=False)),
            enable_prefix_cache=bool(pick("enable_prefix_cache", default=True)),
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
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("Kimi K3 global rank and size are invalid")
        if self.ep_size <= 0 or self.world_size % self.ep_size != 0:
            raise ValueError("Kimi K3 EP size must divide world size")
        if self.dp_size <= 0 or self.world_size % self.dp_size != 0:
            raise ValueError("Kimi K3 DP size must divide world size")
        if not 0 <= self.dp_rank < self.dp_size:
            raise ValueError("Kimi K3 DP rank and size are invalid")
        if self.ep_size > 1 and self.dp_size != self.ep_size:
            supported = self.dp_size == 1 and self.tp_size == self.ep_size and self.world_size == self.ep_size
            if not supported:
                raise ValueError("Kimi K3 supports EP with either dp=ep or dp=1, attention_tp=ep")
        MoECommType.from_value(self.moe_comm_type)
        if self.moe_layer_freq <= 0 or self.first_k_dense_replace < 0:
            raise ValueError("Kimi K3 MoE layer placement is invalid")
        if self.attn_res_block_size <= 0:
            raise ValueError("Kimi K3 attn_res_block_size must be positive")
        if self.activation_situ_beta is not None and self.activation_situ_beta <= 0:
            raise ValueError("Kimi K3 activation_situ_beta must be positive")
        if self.activation_situ_linear_beta is not None and self.activation_situ_linear_beta <= 0:
            raise ValueError("Kimi K3 activation_situ_linear_beta must be positive")
        if self.hidden_act not in ("situ", "silu"):
            raise ValueError(f"Unsupported Kimi K3 activation: {self.hidden_act}")
        if self.moe_router_activation_func not in ("sigmoid", "softmax"):
            raise ValueError("Kimi K3 router activation must be sigmoid or softmax")
        if self.num_experts is not None:
            if self.num_experts_per_token is None or self.moe_intermediate_size is None:
                raise ValueError("Kimi K3 MoE dimensions are incomplete")
            if self.routed_expert_hidden_size is None:
                raise ValueError("Kimi K3 routed_expert_hidden_size is required")
            if self.moe_intermediate_size % self.tp_size != 0:
                raise ValueError("Kimi K3 MoE intermediate_size must divide tp_size")
            if not 0 < self.num_experts_per_token <= self.num_experts:
                raise ValueError("Kimi K3 num_experts_per_token must be within num_experts")
            if self.use_grouped_topk:
                if self.num_expert_group <= 0:
                    raise ValueError("Kimi K3 num_expert_group must be positive")
                if self.num_experts % self.num_expert_group != 0:
                    raise ValueError("Kimi K3 experts must divide evenly into expert groups")
                if not 0 < self.topk_group <= self.num_expert_group:
                    raise ValueError("Kimi K3 topk_group must be within expert groups")
        if self.num_nextn_predict_layers < 0:
            raise ValueError("Kimi K3 num_nextn_predict_layers must be non-negative")
        if self.logit_scale is not None and self.logit_scale <= 0:
            raise ValueError("Kimi K3 logit_scale must be positive")
        mla_dimensions = {
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
        }
        invalid_mla = [name for name, value in mla_dimensions.items() if value <= 0]
        if invalid_mla:
            raise ValueError(f"Kimi K3 MLA dimensions must be positive: {invalid_mla}")
        if not self.mla_use_nope:
            raise ValueError("Kimi K3 MLA requires mla_use_nope")
        if self.mla_use_rope:
            raise ValueError("Kimi K3 MLA does not apply RoPE")
        if not self.mla_use_output_gate:
            raise ValueError("Kimi K3 MLA requires output gating")
        kda_layers = tuple(int(layer) for layer in self.linear_attn_config.get("kda_layers", ()))
        full_attn_layers = tuple(int(layer) for layer in self.linear_attn_config.get("full_attn_layers", ()))
        layer_numbers = kda_layers + full_attn_layers
        if len(set(layer_numbers)) != len(layer_numbers):
            raise ValueError("Kimi K3 attention layer lists must not overlap or repeat")
        if any(layer < 1 or layer > self.n_layers for layer in layer_numbers):
            raise ValueError("Kimi K3 attention layer numbers must be within [1, n_layers]")
        if full_attn_layers and len(layer_numbers) != self.n_layers:
            raise ValueError("Kimi K3 kda_layers and full_attn_layers must cover all layers")
        if self.uses_quantized_weights:
            if self.quant_version != "1.0.0":
                raise ValueError("Kimi K3 W4A8 weights require quant_version 1.0.0")
            if self.quant_group_size != 0:
                raise ValueError("Kimi K3 currently supports per-channel W4A8 weights only")
            if self.num_experts is None or self.routed_expert_hidden_size is None or self.moe_intermediate_size is None:
                raise ValueError("Kimi K3 W4A8 requires routed experts")
            if 16 % self.tp_size != 0:
                raise ValueError("Kimi K3 W4A8 scale_bias requires tp_size <= 16")
            if self.routed_expert_hidden_size % 2 != 0:
                raise ValueError("Kimi K3 W4A8 routed hidden size must be even")
            if self.moe_intermediate_size % (2 * self.tp_size) != 0:
                raise ValueError("Kimi K3 W4A8 expert size must be divisible by 2 * tp_size")

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

    @property
    def use_sequence_parallel(self) -> bool:
        """FlashComm1 sequence parallelism gate.

        The residual trunk is sharded over the attention TP group, so it only
        applies when TP is active. It composes with DP and EP: TP is the
        innermost rank dimension, so a TP group stays inside one DP cell and
        shares its batch; and the MoE always returns output replicated across
        the attention TP group (the trunk layout contract), so the EP path is
        treated as a black box — FlashComm1 optimizes only the TP part
        (attention o_proj + residual trunk) and leaves the MoE's own EP
        communication untouched.
        """
        return self.enable_flashcomm1 and self.tp_size > 1

    def is_kda_layer(self, layer_id: int) -> bool:
        """``layer_id`` is 0-based; the config's ``kda_layers`` is 1-based."""
        kda_layers = self.linear_attn_config.get("kda_layers") or []
        return (layer_id + 1) in kda_layers

    def is_mla_layer(self, layer_id: int) -> bool:
        if not 0 <= layer_id < self.n_layers:
            raise ValueError(f"Kimi K3 layer id is out of range: {layer_id}")
        full_attn_layers = self.linear_attn_config.get("full_attn_layers") or []
        if full_attn_layers:
            return (layer_id + 1) in full_attn_layers
        return not self.is_kda_layer(layer_id)


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
        quant_method = W8A8DynamicLinearMethod() if self.quantized else None
        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size,
            2 * intermediate_per_rank,
            config.tp_size,
            dtype=dtype,
            device=device,
            quant_method=quant_method,
        )
        self.down_proj = RowParallelLinear(
            intermediate_per_rank,
            config.hidden_size,
            config.tp_size,
            dtype=dtype,
            device=device,
            reduce_results=reduce_results,
            quant_method=(W8A8DynamicLinearMethod() if self.quantized else None),
        )
        self.reduce_results = reduce_results
        self.hidden_act = config.hidden_act
        self.situ_beta = float(config.activation_situ_beta or 1.0)
        self.situ_linear_beta = config.activation_situ_linear_beta
        self._loaded_components: set[str] = set()

        # Resolve activation function once at construction to avoid repeated
        # string lookups in the hot path.
        if self.hidden_act == "situ":
            self._activation_fn = self._situ_activation
        elif self.hidden_act == "silu":
            self._activation_fn = self._silu_activation
        else:
            raise ValueError(f"Unsupported Kimi K3 activation: {self.hidden_act}")

    def _situ_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        width = tensor.shape[-1] // 2
        gate, up = tensor[..., :width], tensor[..., width:]
        gate = gate.float()
        up = up.float()
        gate = self.situ_beta * torch.tanh(gate / self.situ_beta) * torch.sigmoid(gate)
        if self.situ_linear_beta is not None:
            up = self.situ_linear_beta * torch.tanh(up / self.situ_linear_beta)
        return (gate * up).to(tensor.dtype)

    def _silu_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        return ops.silu_and_mul(tensor)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        activated = self._activation_fn(self.gate_up_proj(hidden_states))
        return self.down_proj(activated)

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
            target = target_tensor.data[:half] if projection == "gate_proj" else target_tensor.data[half:]
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
            self._loaded_components.update({"gate_proj.weight", "up_proj.weight"})
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
        state_dict: StateDict,
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
                if not state_dict.has(name):
                    continue
                tensor = state_dict.get_sharded_tensor(
                    name,
                    0,
                    tp_rank,
                    tp_size,
                )
                if self.load_weight(name, tensor):
                    loaded.add(name)
        for suffix in suffixes:
            name = f"down_proj.{suffix}"
            if not state_dict.has(name):
                continue
            tensor = (
                state_dict.get_sharded_tensor(
                    name,
                    1,
                    tp_rank,
                    tp_size,
                )
                if suffix == "weight"
                else state_dict.get_tensor(name)
            )
            if self.load_weight(name, tensor):
                loaded.add(name)
        if state_dict.has("gate_up_proj.weight"):
            tensor = state_dict.get_sharded_tensor(
                "gate_up_proj.weight",
                0,
                tp_rank,
                tp_size,
            )
            if self.load_weight("gate_up_proj.weight", tensor):
                loaded.add("gate_up_proj.weight")
        return loaded

    def finish_weight_loading(self) -> None:
        suffixes = ["weight"]
        if self.quantized:
            suffixes.extend(["weight_scale", "weight_offset"])
        required = {
            f"{projection}.{suffix}" for projection in ("gate_proj", "up_proj", "down_proj") for suffix in suffixes
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


class KimiK3FusedQKVAProjection(nn.Module):
    """Merged Kimi K3 q/kv A projection with checkpoint-local loading."""

    def __init__(
        self,
        config: KimiK3TextConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.query_size = config.q_lora_rank
        self.key_value_size = config.kv_lora_rank + config.qk_rope_head_dim
        self.output_size = self.query_size + self.key_value_size
        self.quantized = config.uses_quantized_weights
        if self.quantized:
            self.projection = ColumnParallelLinear(
                config.hidden_size,
                self.output_size,
                1,
                dtype=dtype,
                device=device,
                quant_method=W8A8DynamicLinearMethod(),
            )
        else:
            self.projection = nn.Linear(
                config.hidden_size,
                self.output_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
        self._loaded_components: set[str] = set()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)

    def _load_component(
        self,
        projection_name: str,
        suffix: str,
        tensor: torch.Tensor,
    ) -> None:
        if projection_name == "q_a_proj":
            offset = 0
            size = self.query_size
        elif projection_name == "kv_a_proj_with_mqa":
            offset = self.query_size
            size = self.key_value_size
        else:
            raise KeyError(f"Unsupported Kimi K3 fused projection: {projection_name}")

        target = getattr(self.projection, suffix)
        target_slice = target.data.narrow(0, offset, size)
        _copy_parameter(target_slice, tensor)
        self._loaded_components.add(f"{projection_name}.{suffix}")

    def load_weights(
        self,
        state_dict: Any,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        del tp_rank, tp_size
        suffixes = ["weight"]
        if self.quantized:
            suffixes.extend(("weight_scale", "weight_offset"))
        loaded: set[str] = set()
        for projection_name in ("q_a_proj", "kv_a_proj_with_mqa"):
            for suffix in suffixes:
                name = f"{projection_name}.{suffix}"
                if not state_dict.has(name):
                    continue
                self._load_component(
                    projection_name,
                    suffix,
                    state_dict.get_tensor(name),
                )
                loaded.add(name)
        return loaded

    def finish_weight_loading(self) -> None:
        suffixes = ["weight"]
        if self.quantized:
            suffixes.extend(("weight_scale", "weight_offset"))
        required = {
            f"{projection_name}.{suffix}"
            for projection_name in ("q_a_proj", "kv_a_proj_with_mqa")
            for suffix in suffixes
        }
        missing = required.difference(self._loaded_components)
        if missing:
            raise KeyError(f"Kimi K3 fused q/kv A projection weights are missing: {sorted(missing)}")
        if self.quantized:
            self.projection.finish_weight_loading()


class KimiK3MLAAttention(KimiK3GatedMLA, AttentionRuntimeLayer):
    """xLLM TP/backend adapter for the shared Kimi K3 Gated-MLA core."""

    attention_kind = "mla"
    use_vllm_fia_v2_decode = True

    _REPLICATED = (
        "q_a_layernorm.weight",
        "kv_a_layernorm.weight",
    )
    _COLUMN_SHARDED = (
        "q_b_proj.weight",
        "kv_b_proj.weight",
        "g_proj.weight",
    )

    def __init__(
        self,
        config: KimiK3TextConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        num_heads = config.n_heads // config.tp_size
        query_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        scale = query_head_dim**-0.5
        super().__init__(
            KimiK3GatedMLAConfig(
                hidden_size=config.hidden_size,
                num_attention_heads=num_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                rms_norm_eps=config.rms_norm_eps,
            ),
            dtype=dtype,
            device=device,
        )
        # Attention backends consume these layer-local runtime attributes.
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.head_dim = config.kv_lora_rank
        self.scale = scale
        self.sliding_window = 0
        self.layer_id = layer_id
        self.num_heads_local = num_heads
        self.query_head_dim = query_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        output_size = num_heads * config.v_head_dim
        self.quantized = config.uses_quantized_weights
        self.fused_qkv_a_proj = KimiK3FusedQKVAProjection(
            config,
            dtype,
            device,
        )

        if self.quantized:
            self.q_b_proj = ColumnParallelLinear(
                config.q_lora_rank,
                num_heads * query_head_dim,
                config.tp_size,
                dtype=dtype,
                device=device,
                quant_method=W8A8DynamicLinearMethod(),
            )
        else:
            self.q_b_proj = ColumnParallelLinear(
                config.q_lora_rank,
                num_heads * query_head_dim,
                config.tp_size,
                dtype=dtype,
                device=device,
            )
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, config.rms_norm_eps, dtype=dtype, device=device)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, config.rms_norm_eps, dtype=dtype, device=device)
        self.kv_b_proj = ColumnParallelLinear(
            config.kv_lora_rank,
            num_heads * (config.qk_nope_head_dim + config.v_head_dim),
            config.tp_size,
            dtype=dtype,
            device=device,
        )
        self.g_proj = ColumnParallelLinear(
            config.hidden_size,
            output_size,
            config.tp_size,
            dtype=dtype,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            output_size,
            config.hidden_size,
            config.tp_size,
            dtype=dtype,
            device=device,
            reduce_results=not config.use_sequence_parallel,
        )
        self.register_buffer(
            "W_UK",
            torch.empty(
                num_heads,
                config.qk_nope_head_dim,
                config.kv_lora_rank,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "W_UV",
            torch.empty(
                num_heads,
                config.kv_lora_rank,
                config.v_head_dim,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )
        self._loaded_components: set[str] = set()

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del positions  # Kimi K3 retains the positional slice but does not rotate it.
        num_tokens = hidden_states.shape[0]
        qkv_lora = self.fused_qkv_a_proj(hidden_states)
        q_lora, compressed_kv = qkv_lora.split(
            [
                self.config.q_lora_rank,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ],
            dim=-1,
        )
        q_c = self.q_a_layernorm(q_lora)
        q = self.q_b_proj(q_c).view(num_tokens, self.num_heads_local, self.query_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_latent = torch.bmm(q_nope.transpose(0, 1), self.W_UK).transpose(0, 1)

        k_latent_raw, k_pe = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_latent = self.kv_a_layernorm(k_latent_raw).view(num_tokens, 1, self.kv_lora_rank)
        backend = get_forward_context().attention_backend
        unabsorbed_prefill = None
        if backend.use_unabsorbed_mla_prefill():
            kv_projection = self.kv_b_proj(k_latent.squeeze(1)).view(
                num_tokens,
                self.num_heads_local,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            k_nope, value = kv_projection.split(
                [self.qk_nope_head_dim, self.v_head_dim],
                dim=-1,
            )
            unabsorbed_prefill = MlaUnabsorbedPrefill(
                query_nope=q_nope,
                key_nope=k_nope,
                value=value,
            )
        attn_out = backend.execute_mla(
            q_latent,
            q_pe,
            k_latent,
            k_pe.view(num_tokens, 1, self.qk_rope_head_dim),
            self,
            topk=None,
            unabsorbed_prefill=unabsorbed_prefill,
        )
        if attn_out.shape[-1] == self.v_head_dim:
            values = attn_out
        elif attn_out.device.type in ("npu", "privateuseone"):
            values = torch_npu.npu_transpose_batchmatmul(
                attn_out.transpose(0, 1).contiguous(),
                self.W_UV,
                perm_y=(1, 0, 2),
            )
        else:
            values = torch.bmm(
                attn_out.transpose(0, 1),
                self.W_UV,
            ).transpose(0, 1)
        values = values.reshape(num_tokens, self.num_heads_local * self.v_head_dim)
        return self.apply_output_gate(values, hidden_states)

    def _load_projection_weight(
        self,
        state_dict: Any,
        name: str,
        parameter: torch.Tensor,
        tp_rank: int,
        tp_size: int,
        shard_dim: int | None = None,
    ) -> set[str]:
        tensor = (
            _state_dict_tensor(state_dict, name)
            if shard_dim is None
            else _state_dict_sharded_tensor(state_dict, name, shard_dim, tp_rank, tp_size)
        )
        if tensor is None:
            return set()

        loaded = {name}
        if not tensor.is_floating_point():
            scale_name = f"{name}_scale"
            offset_name = f"{name}_offset"
            scale = (
                _state_dict_tensor(state_dict, scale_name)
                if shard_dim is None
                else _state_dict_sharded_tensor(state_dict, scale_name, 0, tp_rank, tp_size)
            )
            offset = (
                _state_dict_tensor(state_dict, offset_name)
                if shard_dim is None
                else _state_dict_sharded_tensor(state_dict, offset_name, 0, tp_rank, tp_size)
            )
            missing = [
                companion_name
                for companion_name, companion in (
                    (scale_name, scale),
                    (offset_name, offset),
                )
                if companion is None
            ]
            if missing:
                raise KeyError(f"Kimi K3 quantized MLA weight {name} is missing companions: {missing}")
            tensor = (tensor.float() - offset.float()) * scale.float()
            loaded.update((scale_name, offset_name))

        _copy_parameter(parameter, tensor)
        return loaded

    def _load_w8a8_projection(
        self,
        state_dict: Any,
        projection: str,
        module: ColumnParallelLinear,
        tp_rank: int,
        tp_size: int,
        shard_output: bool = False,
    ) -> set[str]:
        weight_name = f"{projection}.weight"
        if not state_dict.has(weight_name):
            return set()

        loaded: set[str] = set()
        for suffix in ("weight", "weight_scale", "weight_offset"):
            name = f"{projection}.{suffix}"
            if not state_dict.has(name):
                raise KeyError(f"Kimi K3 quantized MLA weight {weight_name} is missing companion: {name}")
            tensor = (
                state_dict.get_sharded_tensor(name, 0, tp_rank, tp_size)
                if shard_output
                else state_dict.get_tensor(name)
            )
            if not module.load_weight(suffix, tensor):
                raise KeyError(f"Unsupported Kimi K3 MLA weight: {name}")
            loaded.add(name)
        return loaded

    def load_weights(self, state_dict: Any, tp_rank: int, tp_size: int) -> set[str]:
        loaded: set[str] = set()
        parameters = dict(self.named_parameters())
        loaded.update(
            self.fused_qkv_a_proj.load_weights(
                state_dict,
                tp_rank,
                tp_size,
            )
        )
        replicated = self._REPLICATED
        column_sharded = self._COLUMN_SHARDED
        if self.quantized:
            loaded.update(
                self._load_w8a8_projection(
                    state_dict,
                    "q_b_proj",
                    self.q_b_proj,
                    tp_rank,
                    tp_size,
                    shard_output=True,
                )
            )
            replicated = (
                "q_a_layernorm.weight",
                "kv_a_layernorm.weight",
            )
            column_sharded = (
                "kv_b_proj.weight",
                "g_proj.weight",
            )
        for name in replicated:
            loaded.update(
                self._load_projection_weight(
                    state_dict,
                    name,
                    parameters[name],
                    tp_rank,
                    tp_size,
                )
            )
        for name in column_sharded:
            loaded.update(
                self._load_projection_weight(
                    state_dict,
                    name,
                    parameters[name],
                    tp_rank,
                    tp_size,
                    shard_dim=0,
                )
            )
        loaded.update(
            self._load_projection_weight(
                state_dict,
                "o_proj.weight",
                self.o_proj.weight,
                tp_rank,
                tp_size,
                shard_dim=1,
            )
        )
        self._loaded_components.update(loaded)
        return loaded

    def finish_weight_loading(self) -> None:
        required = set(self._REPLICATED + self._COLUMN_SHARDED + ("o_proj.weight",))
        if self.quantized:
            required.update(
                f"{projection}.{suffix}" for projection in ("q_b_proj",) for suffix in ("weight_scale", "weight_offset")
            )
        missing = required.difference(self._loaded_components)
        if missing:
            raise KeyError(f"Kimi K3 MLA weights are missing: {sorted(missing)}")
        self.fused_qkv_a_proj.finish_weight_loading()
        if self.quantized:
            self.q_b_proj.finish_weight_loading()
        weight = self.kv_b_proj.weight.data.view(
            self.num_heads_local,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_uk, w_uv = weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self.W_UK.copy_(w_uk.contiguous())
        self.W_UV.copy_(w_uv.transpose(1, 2).contiguous())
        self.o_proj.format_npu_weight_()


class KimiK3KDARuntime:
    """Per-step execution context for KDA layers.

    The C++ executor, before every model forward:
      1. sets ``metadata`` to a :class:`KimiK3KDAMetadata` for the batch
         via :meth:`xllm.python.model_executor.executor.ModelExecutor.set_kda_metadata`;
      2. populates ``caches[layer_id] = (conv_state, recurrent_state)`` for
         every KDA layer via :meth:`xllm.python.model_executor.executor.ModelExecutor.bind_kda_caches`,
         with shapes / dtypes per
         :meth:`KimiK3DeltaAttention.conv_state_shape` /
         :meth:`KimiK3DeltaAttention.recurrent_state_shape` /
         :meth:`KimiK3DeltaAttention.state_dtypes`.

    The C++ ``PyExecutorImpl`` wires both steps so the KDA layers read state
    directly from this runtime object; no further C++-side plumbing is needed.
    """

    def __init__(self) -> None:
        self.metadata: KimiK3KDAMetadata | None = None
        self.caches: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def require(self, layer_id: int) -> tuple[KimiK3KDAMetadata, torch.Tensor, torch.Tensor]:
        if self.metadata is None or layer_id not in self.caches:
            raise RuntimeError(
                "KDA runtime is not initialized: the executor must set "
                "metadata and per-layer state caches before the forward pass "
                f"(layer {layer_id})."
            )
        conv_state, recurrent_state = self.caches[layer_id]
        return self.metadata, conv_state, recurrent_state


class KimiK3AttentionPlaceholder(Attention):
    """Shape-compatible attention shell until MLA is implemented.

    Used for the full-attention (non-KDA) layers. KDA layers use
    :class:`KimiK3DeltaAttention` instead.
    """

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
    normed = values_float * torch.rsqrt(values_float.square().mean(dim=-1, keepdim=True) + norm.eps)
    normed = normed * norm.weight.float()
    scores = F.linear(normed, projection.weight.float()).squeeze(-1)
    probabilities = torch.softmax(scores, dim=-1).unsqueeze(-1)
    return (probabilities * values_float).sum(dim=1).to(values.dtype)


def _flashcomm1_pad_size(num_tokens: int, tp_size: int) -> int:
    return (-num_tokens) % tp_size


def _flashcomm1_gather(shard: torch.Tensor, num_tokens: int, tp_size: int) -> torch.Tensor:
    """All-gather a sequence-parallel shard back to the full ``num_tokens`` rows."""
    full = ops.all_gather(shard, dim=0, world_size=tp_size, group_name="tp")
    return full[:num_tokens]


def _flashcomm1_reduce_scatter(partial: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Reduce a TP partial-sum and keep only this rank's token shard.

    Replaces the row-parallel all-reduce so the output stays sequence-parallel.
    """
    pad = _flashcomm1_pad_size(partial.shape[0], tp_size)
    if pad > 0:
        partial = F.pad(partial, (0, 0, 0, pad))
    return ops.reduce_scatter(partial, dim=0, world_size=tp_size, group_name="tp")


def _flashcomm1_shard(full: torch.Tensor, num_tokens: int, tp_size: int, tp_rank: int) -> torch.Tensor:
    """Take this rank's token shard of an already-replicated full tensor."""
    pad = _flashcomm1_pad_size(num_tokens, tp_size)
    if pad > 0:
        full = F.pad(full, (0, 0, 0, pad))
    shard = full.shape[0] // tp_size
    return full[tp_rank * shard : (tp_rank + 1) * shard].contiguous()


def _sp_active(sp_flag: bool) -> bool:
    """FlashComm1 sequence-parallel is prefill-only (matching the C++ gate).
    During ACL graph capture/warmup (decode) it must be disabled so that
    graph-incompatible collective ops (TP all-gather/reduce-scatter) are
    not recorded into the static graph."""
    if not sp_flag:
        return False
    try:
        ctx = get_forward_context()
        return ctx.acl_graph is None and not ctx.graph_warmup
    except RuntimeError:
        return True


class KimiK3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiK3TextConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        kda_runtime: KimiK3KDARuntime,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.kda_runtime = kda_runtime
        self.is_kda = config.is_kda_layer(layer_id)
        self._sp = config.use_sequence_parallel
        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        if self.is_kda:
            self.self_attn = KimiK3DeltaAttention(
                config.hidden_size,
                config.linear_attn_config,
                layer_id=layer_id,
                tp_size=config.tp_size,
                tp_rank=config.tp_rank,
                rms_norm_eps=config.rms_norm_eps,
                quantized=config.uses_quantized_weights,
                reduce_o_proj=not config.use_sequence_parallel,
                dtype=dtype,
                device=device,
            )
        elif config.is_mla_layer(layer_id):
            self.self_attn = KimiK3MLAAttention(config, layer_id, dtype, device)
        else:
            self.self_attn = KimiK3AttentionPlaceholder(config, layer_id)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        if config.is_moe_layer(layer_id):
            if config.uses_quantized_weights:
                routed_expert_down_proj = ColumnParallelLinear(
                    config.hidden_size,
                    config.routed_expert_hidden_size,
                    1,
                    dtype=dtype,
                    device=device,
                    quant_method=W8A8DynamicLinearMethod(),
                )
                routed_expert_up_proj = ColumnParallelLinear(
                    config.routed_expert_hidden_size,
                    config.hidden_size,
                    1,
                    dtype=dtype,
                    device=device,
                    quant_method=W8A8DynamicLinearMethod(),
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
            self.mlp = KimiK3MLP(
                config,
                dtype,
                device,
                reduce_results=not config.use_sequence_parallel,
            )
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
        # With FlashComm1 the residual trunk is sequence-parallel: hidden_states
        # and block_residual carry this rank's token shard. Per-token ops
        # (norms, hyper-connection residual math) run on the shard directly;
        # attention and the FFN need the full token set, so we gather before
        # them and shard their outputs back.
        num_tokens = positions.shape[0]
        prefix_sum: torch.Tensor | None = hidden_states
        sp = _sp_active(self._sp)
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
        attention_input = _flashcomm1_gather(hidden_states, num_tokens, self.tp_size) if sp else hidden_states
        if self.is_kda:
            metadata, conv_state, recurrent_state = self.kda_runtime.require(self.layer_id)
            attention_output = self.self_attn(attention_input, metadata, conv_state, recurrent_state)
        else:
            attention_output = self.self_attn(attention_input, positions)
        if sp:
            attention_output = _flashcomm1_reduce_scatter(attention_output, self.tp_size)
        prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output
        hidden_states = _apply_attention_residual(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            # The MoE keeps its own replicated TP/EP reductions, so gather to
            # full tokens and shard the result back.
            if sp:
                moe_input = _flashcomm1_gather(hidden_states, num_tokens, self.tp_size)
                hidden_states = _flashcomm1_shard(
                    self.block_sparse_moe(moe_input),
                    num_tokens,
                    self.tp_size,
                    self.tp_rank,
                )
            else:
                hidden_states = self.block_sparse_moe(hidden_states)
        else:
            if sp:
                mlp_input = _flashcomm1_gather(hidden_states, num_tokens, self.tp_size)
                hidden_states = _flashcomm1_reduce_scatter(self.mlp(mlp_input), self.tp_size)
            else:
                hidden_states = self.mlp(hidden_states)
        return prefix_sum + hidden_states, block_residual

    def load_weights(
        self,
        state_dict: StateDict,
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
            if not state_dict.has(name):
                continue
            _copy_parameter(target, state_dict.get_tensor(name))
            self._loaded_components.add(name)
            loaded.add(name)

        if self.is_kda:
            # KDA applies its own TP sharding, so hand it the full tensors.
            consumed = self.self_attn.load_weights(
                "self_attn",
                lambda name: (state_dict.get_tensor(name) if state_dict.has(name) else None),
            )
            self.self_attn.process_weights_after_loading()
            for name in consumed:
                self._loaded_components.add(f"self_attn.{name}")
                loaded.add(f"self_attn.{name}")
        elif isinstance(self.self_attn, KimiK3MLAAttention):
            child_state_dict = state_dict.get_dict_with_prefix("self_attn.")
            if child_state_dict.size() > 0:
                loaded.update(
                    f"self_attn.{name}"
                    for name in self.self_attn.load_weights(
                        child_state_dict,
                        tp_rank,
                        tp_size,
                    )
                )

        if hasattr(self, "mlp"):
            child_state_dict = state_dict.get_dict_with_prefix("mlp.")
            if child_state_dict.size() > 0:
                loaded.update(
                    f"mlp.{name}"
                    for name in self.mlp.load_weights(
                        child_state_dict,
                        tp_rank,
                        tp_size,
                    )
                )
        else:
            child_state_dict = state_dict.get_dict_with_prefix(
                "block_sparse_moe.",
            )
            if child_state_dict.size() > 0:
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
            raise KeyError(f"Kimi K3 decoder layer {self.layer_id} weights are missing: {sorted(missing)}")
        if isinstance(self.self_attn, KimiK3MLAAttention):
            self.self_attn.finish_weight_loading()
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
        self.kda_runtime = KimiK3KDARuntime()
        self.layers = nn.ModuleList(
            [KimiK3DecoderLayer(config, i, dtype, device, self.kda_runtime) for i in range(config.n_layers)]
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
        self._sp = config.use_sequence_parallel
        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank
        self._loaded_weights: set[str] = set()

    def initial_block_count(self) -> int:
        return sum(i % self.config.attn_res_block_size == 0 for i in range(self.config.n_layers))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        num_tokens = hidden_states.shape[0]
        sp = _sp_active(self._sp)
        if sp:
            # FlashComm1: shard the token dimension so the residual trunk runs
            # sequence-parallel; the embedding output is replicated, so this is
            # a local slice.
            hidden_states = _flashcomm1_shard(hidden_states, num_tokens, self.tp_size, self.tp_rank)
        block_residual = hidden_states.new_zeros((hidden_states.shape[0], 0, hidden_states.shape[-1]))
        for layer in self.layers:
            hidden_states, block_residual = layer(hidden_states, positions, block_residual)
        hidden_states = _apply_attention_residual(
            hidden_states,
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        )
        if sp:
            hidden_states = _flashcomm1_gather(hidden_states, num_tokens, self.tp_size)
        return self.norm(hidden_states)

    def load_weights(
        self,
        state_dict: StateDict,
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
            if not state_dict.has(name):
                continue
            tensor = (
                state_dict.get_sharded_tensor(
                    name,
                    1,
                    tp_rank,
                    tp_size,
                )
                if name == "embed_tokens.weight"
                else state_dict.get_tensor(name)
            )
            _copy_parameter(target, tensor)
            self._loaded_weights.add(name)
            loaded.add(f"model.{name}")

        for layer_id in _layer_ids(state_dict):
            if not 0 <= layer_id < len(self.layers):
                continue
            layer_state_dict = state_dict.get_dict_with_prefix(
                f"layers.{layer_id}.",
            )
            if layer_state_dict.size() == 0:
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
        _tp_override = int(config.get("tp_size") or 0)
        if _tp_override > 1:
            self.cfg.tp_size = _tp_override
        elif self.cfg.tp_size <= 1:
            _ws = int(config.get("world_size") or 0)
            _dp = int(config.get("dp_size") or 1)
            if _ws > 1:
                self.cfg.tp_size = _ws // _dp
        _tp_rank_override = config.get("tp_rank")
        if _tp_rank_override is not None and int(_tp_rank_override) > 0:
            self.cfg.tp_rank = int(_tp_rank_override)
        else:
            _dev_rank = _tp_rank_from_device(config.get("device", "npu:0"))
            self.cfg.tp_rank = _dev_rank % self.cfg.tp_size if self.cfg.tp_size > 0 else 0
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

    @property
    def supports_prefix_cache(self) -> bool:
        """Kimi K3 participates in the C++ prefix cache.

        The KDA layers restore their conv/recurrent state from checkpoints
        identified by the C++ ``LinearStatePrefixCache``, and the Gated-MLA
        layers reuse KV blocks through the standard ``PrefixCache``. Both
        paths are driven by the C++ scheduler with no Python-side save/restore
        logic needed.
        """
        return self.cfg.enable_prefix_cache

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def compute_logits(
        self,
        hidden: torch.Tensor,
        selected_idxes: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = super().compute_logits(hidden, selected_idxes)
        if self.cfg.logit_scale is not None:
            logits = logits * self.cfg.logit_scale
        return logits

    def load_weights(
        self,
        state_dicts: list[StateDict],
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        tp_rank = self.cfg.tp_rank
        tp_size = self.cfg.tp_size
        loaded: set[str] = set()

        # A single layer's tensors are spread across shards, so merge every
        # shard into one cross-shard view and load each weight exactly once.
        merged = _MergedStateDict(list(state_dicts))
        model_state_dict = merged.get_dict_with_prefixes(["language_model.model.", "model.", ""])
        loaded.update(self.model.load_weights(model_state_dict, tp_rank, tp_size))

        if self.cfg.tie_word_embeddings:
            lm_state_dict = model_state_dict
            lm_weight_name = "embed_tokens.weight"
        else:
            lm_state_dict = merged.get_dict_with_prefixes(
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
    "KimiK3KDARuntime",
    "KimiK3MLAAttention",
    "KimiK3MLP",
    "KimiK3TextConfig",
    "KimiK3TextModel",
]
