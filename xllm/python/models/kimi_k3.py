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

"""Python entry point for Kimi K3 multimodal generation."""

from __future__ import annotations

from typing import Any

import torch

from xllm.python.models.base import PyCausalVLMBase
from xllm.python.models.kimi_k3_text import KimiK3ForCausalLM
from xllm.python.models.kimi_k3_vit import KimiK3VisionModel


class KimiK3ForConditionalGeneration(PyCausalVLMBase):
    """K3 CausalVLM composed from the vision encoder and text model."""

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
