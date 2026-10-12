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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn

from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
)
from xllm.python.model_executor.execution_context import ExecutionMetadataBuilder
from xllm.python.model_executor.forward_context import EplbRuntimeState, LayerSynchronizer
from xllm.python.model_executor.input_batch import InputBatch

ModelExecutionOutput = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class BaseRunner(ABC):
    execution_metadata_builders: tuple[ExecutionMetadataBuilder, ...] = ()

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
    ) -> None:
        self.model = model
        self.attention_backend = attention_backend
        self.device = device
        self.layer_caches: list[LayerCache] = []
        self.execution_metadata_builders = ()

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        self.layer_caches = layer_caches

    def bind_execution_metadata_builders(
        self,
        builders: Sequence[ExecutionMetadataBuilder],
    ) -> None:
        self.execution_metadata_builders = tuple(builders)

    @staticmethod
    def _slice_output(
        output: ModelExecutionOutput,
        batch_size: int,
        *,
        clone: bool = False,
    ) -> ModelExecutionOutput:
        """Trim graph padding, optionally copying outputs out of graph storage."""

        def _trim(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor[:batch_size]
            return tensor.clone() if clone else tensor

        if isinstance(output, tuple):
            return _trim(output[0]), _trim(output[1])
        return _trim(output)

    @abstractmethod
    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        eplb: EplbRuntimeState | None = None,
        input_batch: InputBatch | None = None,
    ) -> ModelExecutionOutput:
        pass
