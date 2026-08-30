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

"""Kimi K3 dense-MLA DSpark draft model.

The graph follows vLLM-Ascend: latent MLA KV cache, YaRN RoPE on the
positional slice, no output gate, and non-causal speculative blocks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch_npu

from xllm.python import kernels
from xllm.python.attention.backend import MlaUnabsorbedPrefill
from xllm.python.layers import ColumnParallelLinear, GatedMLP, RMSNorm, RowParallelLinear
from xllm.python.layers.attention import AttentionRuntimeLayer
from xllm.python.model_executor.forward_context import get_forward_context, record_layer_event
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v32 import DeepseekYarnRotaryEmbedding, _yarn_get_mscale
from xllm.python.models.dspark_accuracy import (
    dump_dspark_tensors,
    is_dspark_accuracy_dump_enabled,
    snapshot_for_dump,
)
from xllm.python.models.kimi_k3_text import (
    _layer_ids,
    _MergedStateDict,
    _state_dict_sharded_tensor,
    _state_dict_tensor,
)

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


def _copy_parameter(parameter: torch.Tensor, tensor: torch.Tensor) -> None:
    if parameter.shape != tensor.shape:
        raise ValueError(f"Kimi K3 DSpark parameter expects {parameter.shape}, got {tensor.shape}")
    parameter.data.copy_(tensor.to(dtype=parameter.dtype, device=parameter.device))


def _first_tensor(state_dict: Any, names: tuple[str, ...]) -> torch.Tensor | None:
    for name in names:
        tensor = _state_dict_tensor(state_dict, name)
        if tensor is not None:
            return tensor
    return None


def _first_sharded_tensor(
    state_dict: Any,
    names: tuple[str, ...],
    dim: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor | None:
    for name in names:
        tensor = _state_dict_sharded_tensor(state_dict, name, dim, tp_rank, tp_size)
        if tensor is not None:
            return tensor
    return None


def _resize_context_projection(model: nn.Module, weight: torch.Tensor, target_hidden_size: int) -> None:
    if weight.ndim != 2 or weight.shape[1] % target_hidden_size != 0:
        raise ValueError(
            "Kimi K3 DSpark context_proj.weight must be "
            f"[hidden, target_hidden_size * layers], got {tuple(weight.shape)}"
        )
    model.config.num_target_layers = weight.shape[1] // target_hidden_size
    if model.context_proj.weight.shape == weight.shape:
        return
    parameter = model.context_proj.weight
    model.context_proj = nn.Linear(
        weight.shape[1],
        model.config.hidden_size,
        bias=False,
        dtype=parameter.dtype,
        device=parameter.device,
    )


def _shard_fused_gate_up(
    weight: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    if weight.shape[0] % (2 * tp_size) != 0:
        raise ValueError(
            "Kimi K3 DSpark fused gate/up dimension must divide "
            f"2 * tp_size, got {weight.shape[0]} and tp_size={tp_size}"
        )
    gate, up = weight.chunk(2, dim=0)
    shard_size = gate.shape[0] // tp_size
    return torch.cat(
        (
            gate.narrow(0, tp_rank * shard_size, shard_size),
            up.narrow(0, tp_rank * shard_size, shard_size),
        ),
        dim=0,
    ).contiguous()


def _rope_cos_sin(
    positions: torch.Tensor,
    rotary: DeepseekYarnRotaryEmbedding,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions.to(torch.int64).contiguous()
    cos_sin = rotary.cos_sin_cache[positions]
    half = cos_sin.shape[-1] // 2
    cos_half, sin_half = cos_sin[..., :half], cos_sin[..., half:]
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1).unsqueeze(1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1).unsqueeze(1)
    return cos, sin


def _apply_interleaved_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    rotary: DeepseekYarnRotaryEmbedding,
) -> torch.Tensor:
    cos, sin = _rope_cos_sin(positions, rotary)
    if tensor.device.type in ("npu", "privateuseone"):
        tokens, heads, dim = tensor.shape
        return torch_npu.npu_interleave_rope(tensor.view(tokens, heads, 1, dim), cos, sin).view(tokens, heads, dim)
    cos_sin = rotary.cos_sin_cache[positions.to(torch.int64).contiguous()]
    half = cos_sin.shape[-1] // 2
    cos_half, sin_half = cos_sin[..., :half], cos_sin[..., half:]
    pairs = tensor.unflatten(-1, (-1, 2))
    even, odd = pairs.unbind(dim=-1)
    cos_half, sin_half = cos_half.unsqueeze(1), sin_half.unsqueeze(1)
    return torch.stack((even * cos_half - odd * sin_half, odd * cos_half + even * sin_half), dim=-1).flatten(-2)


@dataclass
class K3DSparkConfig:
    hidden_size: int = 7168
    intermediate_size: int = 14336
    n_layers: int = 5
    n_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    target_hidden_size: int = 7168
    num_target_layers: int = 5
    vocab_size: int = 163840
    markov_rank: int = 256
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 32768
    rope_theta: float = 1e6
    rope_scaling_factor: float = 32.0
    rope_original_max_position_embeddings: int = 32768
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 0.0
    tp_size: int = 1
    tp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> K3DSparkConfig:
        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in config and config[key] is not None:
                    return config[key]
            return default

        rope = config.get("rope_parameters", {})
        if not isinstance(rope, dict):
            rope = {}

        def rpick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in rope and rope[key] is not None:
                    return rope[key]
                if f"rope_scaling_{key}" in config and config[f"rope_scaling_{key}"] is not None:
                    return config[f"rope_scaling_{key}"]
                if key in config and config[key] is not None:
                    return config[key]
            return default

        dflash = config.get("dflash_config", {})
        if not isinstance(dflash, dict):
            dflash = {}
        target_ids = dflash.get("target_layer_ids") or config.get("target_layer_ids") or []
        target_layers = len(target_ids) or int(pick("dspark_num_target_layers", "num_target_layers", default=5))
        if target_layers > 32:
            target_layers = 5
        max_pos = int(pick("max_position_embeddings", default=32768))
        return cls(
            hidden_size=int(pick("hidden_size", default=7168)),
            intermediate_size=int(pick("intermediate_size", default=14336)),
            n_layers=int(pick("n_layers", "num_hidden_layers", default=5)),
            n_heads=int(pick("n_heads", "num_attention_heads", default=64)),
            q_lora_rank=int(pick("q_lora_rank", default=1536)),
            kv_lora_rank=int(pick("kv_lora_rank", default=512)),
            qk_nope_head_dim=int(pick("qk_nope_head_dim", default=128)),
            qk_rope_head_dim=int(pick("qk_rope_head_dim", default=64)),
            v_head_dim=int(pick("v_head_dim", default=128)),
            target_hidden_size=int(pick("dspark_target_hidden_size", "target_hidden_size", default=7168)),
            num_target_layers=target_layers,
            vocab_size=int(pick("vocab_size", default=163840)),
            markov_rank=int(pick("markov_rank", default=256)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-5)),
            max_position_embeddings=max_pos,
            rope_theta=float(rpick("rope_theta", default=dflash.get("rope_theta", 1e6))),
            rope_scaling_factor=float(rpick("factor", "rope_scaling_factor", default=32.0)),
            rope_original_max_position_embeddings=int(rpick("original_max_position_embeddings", default=max_pos)),
            rope_beta_fast=int(rpick("beta_fast", default=32)),
            rope_beta_slow=int(rpick("beta_slow", default=1)),
            rope_mscale=float(rpick("mscale", default=1.0)),
            rope_mscale_all_dim=float(rpick("mscale_all_dim", default=0.0)),
            tp_size=int(pick("tp_size", default=1)),
            tp_rank=int(pick("tp_rank", default=0)),
            dp_size=int(pick("dp_size", default=1)),
            dp_rank=int(pick("dp_rank", default=0)),
        )

    def validate(self) -> None:
        values = (
            self.hidden_size,
            self.intermediate_size,
            self.n_layers,
            self.n_heads,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.target_hidden_size,
            self.num_target_layers,
            self.vocab_size,
            self.markov_rank,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Kimi K3 DSpark dimensions must be positive")
        if self.n_heads % self.tp_size or self.intermediate_size % self.tp_size or self.vocab_size % self.tp_size:
            raise ValueError("Kimi K3 DSpark dimensions must divide tp_size")
        if self.qk_rope_head_dim % 2:
            raise ValueError("Kimi K3 DSpark RoPE dimension must be even")
        if not 0 <= self.tp_rank < self.tp_size or not 0 <= self.dp_rank < self.dp_size:
            raise ValueError("Kimi K3 DSpark parallel rank is invalid")


class K3DSparkFusedQKVAProjection(nn.Module):
    def __init__(self, config: K3DSparkConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.query_size = config.q_lora_rank
        self.kv_size = config.kv_lora_rank + config.qk_rope_head_dim
        self.projection = nn.Linear(
            config.hidden_size,
            self.query_size + self.kv_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)

    def load_weights(self, state_dict: Any) -> set[str]:
        fused = _first_tensor(
            state_dict,
            (
                "fused_qkv_a_proj.weight",
                "fused_qkv_a_proj.projection.weight",
            ),
        )
        if fused is not None:
            _copy_parameter(self.projection.weight, fused)
            return {"fused_qkv_a_proj.weight"}

        query = _first_tensor(state_dict, ("q_a_proj.weight",))
        kv = _first_tensor(state_dict, ("kv_a_proj_with_mqa.weight",))
        if query is None or kv is None:
            available = sorted(name for name in state_dict.keys() if "proj" in name or "attn" in name)
            raise KeyError(
                "Kimi K3 DSpark q/kv A weights are incomplete: expected "
                "q_a_proj.weight and kv_a_proj_with_mqa.weight, or "
                "fused_qkv_a_proj.weight; available projection keys: "
                f"{available[:32]}"
            )
        _copy_parameter(self.projection.weight, torch.cat((query, kv), dim=0))
        return {"q_a_proj.weight", "kv_a_proj_with_mqa.weight"}


class K3DSparkMLAAttention(AttentionRuntimeLayer, nn.Module):
    attention_kind = "mla"
    use_vllm_fia_v2_decode = True

    def __init__(self, config: K3DSparkConfig, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.config, self.layer_id = config, layer_id
        self.num_heads, self.num_kv_heads = config.n_heads // config.tp_size, 1
        self.head_dim, self.sliding_window = config.kv_lora_rank, 0
        self.non_causal_block = True
        self.q_lora_rank, self.kv_lora_rank = config.q_lora_rank, config.kv_lora_rank
        self.qk_nope_head_dim, self.qk_rope_head_dim = config.qk_nope_head_dim, config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        query_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.scale = query_dim**-0.5 * _yarn_get_mscale(config.rope_scaling_factor, config.rope_mscale_all_dim) ** 2
        self.fused_qkv_a_proj = K3DSparkFusedQKVAProjection(config, dtype, device)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, config.rms_norm_eps, dtype=dtype, device=device)
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank,
            self.num_heads * query_dim,
            config.tp_size,
            dtype=dtype,
            device=device,
        )
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, config.rms_norm_eps, dtype=dtype, device=device)
        self.kv_b_proj = ColumnParallelLinear(
            config.kv_lora_rank,
            self.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
            config.tp_size,
            dtype=dtype,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * config.v_head_dim,
            config.hidden_size,
            config.tp_size,
            dtype=dtype,
            device=device,
        )
        self.rotary = DeepseekYarnRotaryEmbedding(
            config.qk_rope_head_dim,
            config.rope_original_max_position_embeddings,
            config.rope_scaling_factor,
            config.rope_theta,
            config.rope_beta_fast,
            config.rope_beta_slow,
            config.rope_mscale,
            config.rope_mscale_all_dim,
            dtype=dtype,
            device=device,
            cache_max_position_embeddings=config.max_position_embeddings,
        )
        self.register_buffer(
            "W_UK",
            torch.empty(
                self.num_heads,
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
                self.num_heads,
                config.kv_lora_rank,
                config.v_head_dim,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )

    def _project_context_kv(
        self,
        qkv_lora: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressed = qkv_lora[..., self.q_lora_rank :]
        latent, rope = compressed.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent = self.kv_a_layernorm(latent).view(-1, 1, self.kv_lora_rank)
        return latent, _apply_interleaved_rope(rope.unsqueeze(1), positions, self.rotary)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        qkv = self.fused_qkv_a_proj(hidden_states)
        q = self.q_b_proj(self.q_a_layernorm(qkv[..., : self.q_lora_rank])).view(
            tokens,
            self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_latent = torch.bmm(q_nope.transpose(0, 1), self.W_UK).transpose(0, 1)
        q_pe = _apply_interleaved_rope(q_rope, positions, self.rotary)
        k_latent, k_pe = self._project_context_kv(qkv, positions)
        backend = get_forward_context().attention_backend
        unabsorbed_prefill = None
        if backend.use_unabsorbed_mla_prefill():
            projected_kv = self.kv_b_proj(k_latent.squeeze(1)).view(
                tokens,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            key_nope, value = projected_kv.split(
                [self.qk_nope_head_dim, self.v_head_dim],
                dim=-1,
            )
            unabsorbed_prefill = MlaUnabsorbedPrefill(
                query_nope=q_nope,
                key_nope=key_nope,
                value=value,
            )
        output = backend.execute_mla(
            q_latent,
            q_pe,
            k_latent,
            k_pe,
            self,
            unabsorbed_prefill=unabsorbed_prefill,
        )
        if output.shape[-1] == self.v_head_dim:
            values = output
        elif output.device.type in ("npu", "privateuseone"):
            values = torch_npu.npu_transpose_batchmatmul(
                output.transpose(0, 1).contiguous(),
                self.W_UV,
                perm_y=(1, 0, 2),
            )
        else:
            values = torch.bmm(output.transpose(0, 1), self.W_UV).transpose(0, 1)
        return self.o_proj(values.reshape(tokens, self.num_heads * self.v_head_dim))

    def project_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._project_context_kv(self.fused_qkv_a_proj(hidden_states), positions)

    def load_weights(self, state_dict: Any, tp_rank: int, tp_size: int) -> set[str]:
        loaded = self.fused_qkv_a_proj.load_weights(state_dict)
        for name, parameter in (
            ("q_a_layernorm.weight", self.q_a_layernorm.weight),
            ("kv_a_layernorm.weight", self.kv_a_layernorm.weight),
        ):
            tensor = _first_tensor(state_dict, (name,))
            if tensor is None:
                raise KeyError(f"missing Kimi K3 DSpark MLA weight: {name}")
            _copy_parameter(parameter, tensor)
            loaded.add(name)
        for name, parameter in (
            ("q_b_proj.weight", self.q_b_proj.weight),
            ("kv_b_proj.weight", self.kv_b_proj.weight),
        ):
            tensor = _first_sharded_tensor(state_dict, (name,), 0, tp_rank, tp_size)
            if tensor is None:
                raise KeyError(f"missing Kimi K3 DSpark MLA weight: {name}")
            _copy_parameter(parameter, tensor)
            loaded.add(name)
        output = _first_sharded_tensor(state_dict, ("o_proj.weight",), 1, tp_rank, tp_size)
        if output is None:
            raise KeyError("missing Kimi K3 DSpark MLA weight: o_proj.weight")
        _copy_parameter(self.o_proj.weight, output)
        loaded.add("o_proj.weight")
        weight = self.kv_b_proj.weight.data.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_uk, w_uv = weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self.W_UK.copy_(w_uk.contiguous())
        self.W_UV.copy_(w_uv.transpose(1, 2).contiguous())
        self.o_proj.format_npu_weight_()
        return loaded


class K3DSparkDecoderLayer(nn.Module):
    def __init__(self, config: K3DSparkConfig, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = K3DSparkMLAAttention(config, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)
        self.mlp = GatedMLP(config.hidden_size, config.intermediate_size, config.tp_size, dtype, device)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual, hidden_states = hidden_states, self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual

    def load_weights(self, state_dict: Any, tp_rank: int, tp_size: int) -> set[str]:
        loaded: set[str] = set()
        for name, parameter in (
            ("input_layernorm.weight", self.input_layernorm.weight),
            ("post_attention_layernorm.weight", self.post_attention_layernorm.weight),
        ):
            tensor = _first_tensor(state_dict, (name,))
            if tensor is None:
                raise KeyError(f"missing Kimi K3 DSpark weight: {name}")
            _copy_parameter(parameter, tensor)
            loaded.add(name)
        attention = state_dict.get_dict_with_prefix("self_attn.")
        loaded.update(f"self_attn.{name}" for name in self.self_attn.load_weights(attention, tp_rank, tp_size))
        mlp = state_dict.get_dict_with_prefix("mlp.")
        gate_up = _first_tensor(
            mlp,
            ("gate_up_proj.weight",),
        )
        gate = _first_sharded_tensor(
            mlp,
            ("gate_proj.weight",),
            0,
            tp_rank,
            tp_size,
        )
        up = _first_sharded_tensor(
            mlp,
            ("up_proj.weight",),
            0,
            tp_rank,
            tp_size,
        )
        down = _first_sharded_tensor(mlp, ("down_proj.weight",), 1, tp_rank, tp_size)
        if gate_up is None and (gate is None or up is None):
            raise KeyError("Kimi K3 DSpark MLP weights are incomplete")
        if down is None:
            raise KeyError("Kimi K3 DSpark MLP down projection weight is missing")
        if gate_up is not None:
            gate_up = _shard_fused_gate_up(gate_up, tp_rank, tp_size)
            loaded.add("mlp.gate_up_proj.weight")
        else:
            if gate is None or up is None:
                raise KeyError("Kimi K3 DSpark MLP weights are incomplete")
            gate_up = torch.cat((gate, up), dim=0)
            loaded.update(("mlp.gate_proj.weight", "mlp.up_proj.weight"))
        _copy_parameter(self.mlp.gate_up_proj.weight, gate_up)
        _copy_parameter(self.mlp.down_proj.weight, down)
        self.mlp.down_proj.format_npu_weight_()
        loaded.add("mlp.down_proj.weight")
        return loaded


class K3DSparkModel(nn.Module):
    def __init__(self, config: K3DSparkConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.config, self.embed_tokens = config, None
        self.context_proj = nn.Linear(
            config.target_hidden_size * config.num_target_layers,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.context_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)
        self.layers = nn.ModuleList([K3DSparkDecoderLayer(config, i, dtype, device) for i in range(config.n_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)
        self.markov_w1 = nn.Embedding(config.vocab_size, config.markov_rank, dtype=dtype, device=device)
        self.markov_w2 = nn.Linear(config.markov_rank, config.vocab_size, bias=False, dtype=dtype, device=device)
        self._accuracy_base_logits: torch.Tensor | None = None
        self._accuracy_markov_step = 0

    _inject_call_count = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Inject vLLM's input_ids for the first draft call to enable
        # like-for-like precision comparison (debug only).
        if is_dspark_accuracy_dump_enabled():
            inject_path = os.getenv("XLLM_DSPARK_INJECT_PATH", "")
            if inject_path and os.path.isfile(inject_path) and K3DSparkModel._inject_call_count == 0:
                K3DSparkModel._inject_call_count += 1
                inj = torch.load(inject_path, weights_only=False)
                inj_ids = inj["input_ids"].to(device=input_ids.device, dtype=input_ids.dtype)
                inj_pos = inj["positions"].to(device=positions.device, dtype=positions.dtype)
                print(f"INJECT: overriding draft input_ids {input_ids.tolist()} -> {inj_ids.tolist()}", flush=True)
                input_ids = inj_ids
                positions = inj_pos
        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise RuntimeError("Kimi K3 DSpark target embedding is not shared")
            inputs_embeds = self.embed_tokens(input_ids)
        hidden, residual = inputs_embeds, None
        trace_tensors: dict[str, torch.Tensor | None] = {
            "draft.input_ids": snapshot_for_dump(input_ids),
            "draft.positions": snapshot_for_dump(positions),
            "draft.inputs_embeds": snapshot_for_dump(inputs_embeds),
        }
        for layer_index, layer in enumerate(self.layers):
            hidden, residual = layer(positions, hidden, residual)
            trace_tensors[f"draft.layer.{layer_index}.hidden"] = snapshot_for_dump(hidden)
            trace_tensors[f"draft.layer.{layer_index}.residual"] = snapshot_for_dump(residual)
            record_layer_event(layer.layer_id)
        hidden, _ = self.final_norm(hidden, residual)
        trace_tensors["draft.final_hidden"] = snapshot_for_dump(hidden)
        dump_dspark_tensors("draft_forward", trace_tensors)
        return hidden

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError("Kimi K3 DSpark target embedding is not shared")
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, target_hidden: torch.Tensor) -> torch.Tensor:
        squeeze = target_hidden.dim() == 1
        if squeeze:
            target_hidden = target_hidden.unsqueeze(0)
        expected = self.context_proj.in_features
        if target_hidden.shape[-1] != expected:
            raise ValueError(
                f"Kimi K3 DSpark context hidden size mismatch: expected {expected}, got {target_hidden.shape[-1]}"
            )
        hidden = self.context_norm(self.context_proj(target_hidden))
        output = hidden.squeeze(0) if squeeze else hidden
        dump_dspark_tensors(
            "context_projection",
            {
                "context.target_hidden": target_hidden,
                "context.projected_hidden": output,
            },
        )
        return output

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if len(kv_caches) != len(self.layers):
            raise ValueError("Kimi K3 DSpark KV cache count must match draft layers")
        hidden = self.combine_hidden_states(target_hidden)
        for layer, (latent_cache, rope_cache, _) in zip(self.layers, kv_caches, strict=True):
            attention = layer.self_attn
            qkv_lora = attention.fused_qkv_a_proj(hidden)
            raw_kv = qkv_lora[..., attention.q_lora_rank :].contiguous()
            rope_cos, rope_sin = _rope_cos_sin(positions, attention.rotary)
            kernels.write_mla_kv_cache(
                raw_kv,
                attention.kv_a_layernorm.weight,
                rope_cos,
                rope_sin,
                slot_mapping,
                latent_cache,
                rope_cache,
                attention.kv_lora_rank,
                attention.qk_rope_head_dim,
                attention.config.rms_norm_eps,
            )
        return hidden

    def dspark_markov_bias(self, previous_token_ids: torch.Tensor) -> torch.Tensor:
        markov_embed = self.markov_w1(previous_token_ids)
        markov_bias = self.markov_w2(markov_embed)
        corrected_logits = None
        predicted_token_ids = None
        if self._accuracy_base_logits is not None:
            base_logits = self._accuracy_base_logits.view(
                previous_token_ids.shape[0],
                -1,
                self._accuracy_base_logits.shape[-1],
            )
            if self._accuracy_markov_step < base_logits.shape[1]:
                corrected_logits = base_logits[:, self._accuracy_markov_step] + markov_bias
                predicted_token_ids = corrected_logits.argmax(dim=-1)
                self._accuracy_markov_step += 1
        dump_dspark_tensors(
            "draft_markov",
            {
                "markov.previous_token_ids": previous_token_ids,
                "markov.embedding": markov_embed,
                "markov.bias": markov_bias,
                "markov.corrected_logits": corrected_logits,
                "markov.predicted_token_ids": predicted_token_ids,
            },
        )
        return markov_bias

    def set_accuracy_base_logits(self, base_logits: torch.Tensor) -> None:
        self._accuracy_base_logits = base_logits
        self._accuracy_markov_step = 0

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [f"layers.{layer.layer_id}.self_attn" for layer in self.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * len(self.layers)


class K3DSparkForCausalLM(PyModelBase):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = K3DSparkConfig.from_dict(config)
        self.cfg.validate()
        self.dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        self.device = torch.device(config.get("device", "npu"))
        self.model = K3DSparkModel(self.cfg, self.dtype, self.device)
        self.lm_head: nn.Module | None = None

    def compute_logits(self, hidden: torch.Tensor, selected_idxes: torch.Tensor | None) -> torch.Tensor:
        if selected_idxes is not None and selected_idxes.numel() > 0:
            hidden = hidden.index_select(0, selected_idxes)
        if self.lm_head is None:
            raise RuntimeError("Kimi K3 DSpark target LM head is not shared")
        logits = self.lm_head(hidden)
        if is_dspark_accuracy_dump_enabled():
            self.model.set_accuracy_base_logits(logits)
        dump_dspark_tensors(
            "draft_logits",
            {
                "draft.sample_hidden": hidden,
                "draft.base_logits": logits,
            },
        )
        return logits

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(target_hidden)

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        return self.model.write_context_kv(target_hidden, positions, slot_mapping, kv_caches)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def get_draft_attn_causal(self) -> list[bool]:
        return self.model.get_draft_attn_causal()

    def dspark_markov_bias(self, previous_token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.dspark_markov_bias(previous_token_ids)

    def load_weights(self, state_dicts: list[StateDict], tp_rank: int, tp_size: int) -> set[str]:
        del tp_rank, tp_size
        merged = _MergedStateDict(list(state_dicts))
        state_dict = merged.get_dict_with_prefixes(["model.", ""])
        loaded: set[str] = set()
        direct = {
            "context_proj.weight": ("context_proj.weight", "fc.weight"),
            "context_norm.weight": ("context_norm.weight", "hidden_norm.weight"),
            "final_norm.weight": ("final_norm.weight", "norm.weight"),
            "markov_w1.weight": ("markov_head.markov_w1.weight", "markov_w1.weight"),
            "markov_w2.weight": ("markov_head.markov_w2.weight", "markov_w2.weight"),
        }
        for name, aliases in direct.items():
            tensor = _first_tensor(state_dict, aliases)
            if tensor is None:
                raise KeyError(f"missing Kimi K3 DSpark weight: {name}")
            if name == "context_proj.weight":
                _resize_context_projection(self.model, tensor, self.cfg.target_hidden_size)
            _copy_parameter(getattr(self.model, name.rsplit(".", 1)[0]).weight, tensor)
            loaded.add(name)
        layer_ids = _layer_ids(state_dict)
        if len(layer_ids) != len(self.model.layers):
            raise KeyError(
                "Kimi K3 DSpark checkpoint layer count does not match config: "
                f"got {layer_ids}, expected {len(self.model.layers)}"
            )
        for layer, layer_id in zip(self.model.layers, layer_ids, strict=True):
            layer_state = state_dict.get_dict_with_prefix(f"layers.{layer_id}.")
            loaded.update(
                f"layers.{layer_id}.{name}"
                for name in layer.load_weights(
                    layer_state,
                    self.cfg.tp_rank,
                    self.cfg.tp_size,
                )
            )
        return loaded


K3DSparkAttention = K3DSparkMLAAttention
KimiK3DSparkConfig = K3DSparkConfig
KimiK3DSparkAttention = K3DSparkMLAAttention
KimiK3DSparkDecoderLayer = K3DSparkDecoderLayer
KimiK3DSparkModel = K3DSparkModel
KimiK3DSparkForCausalLM = K3DSparkForCausalLM

__all__ = [
    "K3DSparkAttention",
    "K3DSparkConfig",
    "K3DSparkMLAAttention",
    "K3DSparkDecoderLayer",
    "K3DSparkModel",
    "K3DSparkForCausalLM",
    "KimiK3DSparkConfig",
    "KimiK3DSparkAttention",
    "KimiK3DSparkDecoderLayer",
    "KimiK3DSparkModel",
    "KimiK3DSparkForCausalLM",
]
