# Copyright 2026 The xLLM Authors.
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

"""Qwen2 dense decoder tower (Python model executor target).

Embedded as the LLM half of JoyaiASR (checkpoint prefix ``llm.``). Weight
loading implements the shared ``ScopedWeightLoader`` protocol:
``load_gqa_fused_attention`` with ``qk_norm=False`` (HF Qwen2 has no
per-head QK-norm) — its ``attention_bias`` covers q/k/v only, and the
helper skips the absent o_proj bias on its own. RoPE uses the shared
half-cache helpers in :mod:`xllm.python.layers.rotary_embedding`.

``Qwen2Model.forward`` consumes ``self._inputs_embeds`` when set (multimodal
merge path), mirroring ``Qwen3VLModel``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    GatedMLP,
    HiddenParallelEmbedding,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
    apply_rotary_half,
    gather_half_rope_cos_sin,
)
from xllm.python.model_executor.forward_context import record_layer_event
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    gqa_head_split,
    load_gqa_fused_attention,
)


@dataclass
class Qwen2Config:
    hidden_size: int = 3584
    n_layers: int = 28
    n_heads: int = 28
    n_kv_heads: int = 4
    head_dim: int = 128
    intermediate_size: int = 18944
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 32768
    vocab_size: int = 152064
    tie_word_embeddings: bool = False
    sliding_window: int = 0
    attention_bias: bool = True
    tp_size: int = 1
    tp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0

    # Keys mapping 1:1 onto fields; defaults live only on the dataclass.
    _DIRECT_KEYS = (
        "hidden_size",
        "head_dim",
        "intermediate_size",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
        "vocab_size",
        "tie_word_embeddings",
        "sliding_window",
        "attention_bias",
        "tp_size",
        "tp_rank",
        "dp_size",
        "dp_rank",
    )
    # Fields whose flat-ModelArgs key differs from the HF config key.
    _ALIASED_KEYS = {
        "n_layers": ("n_layers", "num_hidden_layers"),
        "n_heads": ("n_heads", "num_attention_heads"),
        "n_kv_heads": ("n_kv_heads", "num_key_value_heads"),
    }

    @classmethod
    def from_dict(cls, d: dict) -> Qwen2Config:
        """First present key wins for aliased fields; other fields map 1:1."""

        def pick(*keys: str):
            for key in keys:
                if d.get(key) is not None:
                    return d[key]
            return None

        kwargs: dict = {key: d[key] for key in cls._DIRECT_KEYS if d.get(key) is not None}
        for field, keys in cls._ALIASED_KEYS.items():
            value = pick(*keys)
            if value is not None:
                kwargs[field] = value
        cfg = cls(**kwargs)
        # HF fallbacks: absent kv heads means MHA, absent head_dim is derived.
        if "n_kv_heads" not in kwargs:
            cfg.n_kv_heads = cfg.n_heads
        if d.get("head_dim") is None:
            cfg.head_dim = cfg.hidden_size // cfg.n_heads
        return cfg

    def head_split(self) -> tuple[int, int]:
        """Per-rank ``(num_heads, num_kv_heads)``."""
        return gqa_head_split(self.n_heads, self.n_kv_heads, self.tp_size)


class Qwen2Attention(nn.Module):
    def __init__(self, cfg: Qwen2Config, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.layer_id = layer_id
        self._use_fused_rope = kernels.has_fused_qk_rotary(cfg.head_dim)
        num_heads, num_kv_heads = cfg.head_split()
        tp = cfg.tp_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        # Unsharded kv head count (the fused loader derives the split from it).
        self.n_kv_heads_full = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim

        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_size,
            self.q_size + 2 * self.kv_size,
            tp,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            cfg.hidden_size,
            tp,
            bias=False,  # HF Qwen2 never has o_proj bias
            dtype=dtype,
            device=device,
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.head_dim**-0.5,
            sliding_window=cfg.sliding_window,
            layer_id=layer_id,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)
        num_tokens = qkv.size(0)
        q = qkv[:, : self.q_size].view(num_tokens, self.num_heads, self.head_dim)
        k = qkv[:, self.q_size : self.q_size + self.kv_size].view(num_tokens, self.num_kv_heads, self.head_dim)
        v = qkv[:, self.q_size + self.kv_size :]

        # Manual RoPE; the table pair is built once per forward (see Qwen2Model)
        # in the layout the active path expects.
        if self._use_fused_rope:
            q, k = kernels.apply_qk_rotary(q, k, rope_cos, rope_sin)
        else:
            q = apply_rotary_half(q, rope_cos, rope_sin)
            k = apply_rotary_half(k, rope_cos, rope_sin)

        attn_out = self.attn(q.reshape(num_tokens, self.q_size), k.reshape(num_tokens, self.kv_size), v)
        return self.o_proj(attn_out)

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        # HF Qwen2 has no per-head q/k RMSNorm.
        load_gqa_fused_attention(self, weights, context, self.n_kv_heads_full, qk_norm=False)
        self.o_proj.process_weights_after_loading()


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Qwen2Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = Qwen2Attention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.mlp = GatedMLP(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.tp_size,
            dtype,
            device,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)

        hidden = self.self_attn(hidden, rope_cos, rope_sin)

        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        weights.load_tensor(self.input_layernorm.weight, "input_layernorm.weight")
        weights.load_tensor(self.post_attention_layernorm.weight, "post_attention_layernorm.weight")
        self.self_attn.load_weights(weights.with_prefix("self_attn."), context)
        self.mlp.load_weights(weights.with_prefix("mlp."), context)
        self.mlp.process_weights_after_loading()


class Qwen2Model(nn.Module):
    def __init__(self, cfg: Qwen2Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self._use_fused_rope = kernels.has_fused_qk_rotary(cfg.head_dim)
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // tp, tp, dtype=dtype, device=device
        )
        self.rotary = RotaryEmbedding(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList([Qwen2DecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        # Set by get_input_embeddings (multimodal merge); None on text-only steps.
        self._inputs_embeds: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self._inputs_embeds is not None:
            hidden = self._inputs_embeds
            self._inputs_embeds = None
        else:
            hidden = self.embed_tokens(input_ids)

        positions = positions.to(torch.int64).contiguous()
        # positions and the rotary cache are invariant across the layer
        # loop: build the rope table pair once here instead of once per
        # layer — n_layers identical [num_tokens, head_dim] gathers (and
        # recorded decode-graph ops) collapse to one. The fused path expands
        # the gathered half rows to full-width tables once here; the
        # reference path consumes the half rows directly.
        rope_cos, rope_sin = gather_half_rope_cos_sin(self.rotary.cos_sin_cache, positions)
        if self._use_fused_rope:
            rope_cos = kernels.expand_half_rope_table(rope_cos, self.rotary.head_dim)
            rope_sin = kernels.expand_half_rope_table(rope_sin, self.rotary.head_dim)
        residual: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            hidden, residual = layer(hidden, residual, rope_cos, rope_sin)
            record_layer_event(i)
        hidden, _ = self.norm(hidden, residual)
        return hidden
