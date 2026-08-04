# Copyright 2026 The xLLM Authors.
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

"""Python entry point for Kimi K3 multimodal generation.

The vision tower is implemented in :mod:`kimi_k3_vit`.  The language side is
intentionally a small executor-compatible shell while the K3 decoder is being
ported; it preserves the real C++ VLM/mm_data/PyExecutor control flow so the
vision output can be compared independently.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from xllm.python.layers.attention import Attention
from xllm.python.models.base import PyCausalVLMBase, PyModelBase
from xllm.python.models.kimi_k3_vit import KimiK3VisionModel


class _ZeroLMHead(nn.Module):
    def __init__(self, vocab_size: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.device = device

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (hidden_states.shape[0], self.vocab_size),
            dtype=self.dtype,
            device=hidden_states.device,
        )


class _KimiK3TextShell(nn.Module):
    """Executor contract for the temporary language-model placeholder."""

    def __init__(
        self, config: dict[str, Any], dtype: torch.dtype, device: torch.device
    ):
        super().__init__()
        self.hidden_size = int(config.get("hidden_size", 7168))
        num_layers = int(config.get("n_layers", 93))
        head_dim = int(config.get("head_dim", 1))
        self.dummy_attentions = nn.ModuleList(
            [
                Attention(
                    num_heads=int(config.get("n_heads", 1)),
                    num_kv_heads=int(config.get("n_kv_heads", 1) or 1),
                    head_dim=head_dim,
                    scale=head_dim**-0.5,
                    sliding_window=int(config.get("sliding_window", -1)),
                    layer_id=layer_id,
                )
                for layer_id in range(num_layers)
            ]
        )
        self.dtype = dtype
        self.device = device

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if inputs_embeds is not None:
            return inputs_embeds
        return torch.zeros(
            (input_ids.shape[0], self.hidden_size),
            dtype=self.dtype,
            device=input_ids.device,
        )


class KimiK3ForCausalLM(PyModelBase):
    """Temporary K3 language-model shell driven by ``PyExecutorImpl``."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.model = _KimiK3TextShell(config, dtype, device)
        self.lm_head = _ZeroLMHead(
            int(config.get("vocab_size", 163840)), dtype, device
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (input_ids.shape[0], self.model.hidden_size),
            dtype=self.model.dtype,
            device=input_ids.device,
        )

    def load_weights(
        self, state_dicts: list[Any], tp_rank: int, tp_size: int
    ) -> set[str]:
        del state_dicts, tp_rank, tp_size
        return set()


class KimiK3ForConditionalGeneration(PyCausalVLMBase):
    """K3 CausalVLM composed from the vision path and causal LM."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.vision_model = KimiK3VisionModel(config)
        self.language_model = KimiK3ForCausalLM(config)

    def encode_multimodal(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> list[torch.Tensor]:
        return self.vision_model(pixel_values, grid_thws)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeds: torch.Tensor | None,
        multimodal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeds is None or multimodal_mask is None:
            return inputs_embeds
        if multimodal_embeds.shape[0] != int(multimodal_mask.sum().item()):
            raise ValueError("Kimi K3 image embedding and mask sizes do not match")
        inputs_embeds[multimodal_mask] = multimodal_embeds
        return inputs_embeds

    def load_weights(
        self, state_dicts: list[Any], tp_rank: int, tp_size: int
    ) -> set[str]:
        loaded = self.vision_model.load_weights(state_dicts, tp_rank, tp_size)
        loaded.update(
            self.language_model.load_weights(state_dicts, tp_rank, tp_size)
        )
        return loaded


__all__ = ["KimiK3ForCausalLM", "KimiK3ForConditionalGeneration"]
