# Copyright 2025-2026 The xLLM Authors.
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

"""Attention layer that delegates runtime execution to the current backend.

The layer owns layer-local attention semantics. Wrapper state, plan, workspace
and KV cache reside in the FlashInferBackend, accessed via ForwardContext.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python.model_executor.forward_context import get_forward_context


@dataclass(frozen=True, slots=True)
class AttentionLayerSpec:
    """Layer-local runtime contract used for cache and backend dispatch."""

    layer_id: int
    kind: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float
    sliding_window: int


class AttentionRuntimeLayer:
    """Mixin implemented by every layer that owns a runtime KV cache."""

    attention_kind = "mha"

    def attention_layer_spec(self) -> AttentionLayerSpec:
        return AttentionLayerSpec(
            layer_id=self.layer_id,
            kind=self.attention_kind,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            sliding_window=self.sliding_window,
        )


class Attention(AttentionRuntimeLayer, nn.Module):
    """Thin attention layer that dispatches to the current backend."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        layer_id: int,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.sliding_window = sliding_window
        self.layer_id = layer_id
        self.causal = causal

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        backend = get_forward_context().attention_backend
        return backend.execute(q, k, v, self)
