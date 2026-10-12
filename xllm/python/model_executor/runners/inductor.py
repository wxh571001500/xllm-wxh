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

import torch

from xllm.python.attention.backend import AttentionMetadata
from xllm.python.model_executor.forward_context import (
    EplbRuntimeState,
    ForwardContext,
    LayerSynchronizer,
    forward_context,
)
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.runners.base import BaseRunner, ModelExecutionOutput


class InductorRunner(BaseRunner):
    def __init__(self, model, attention_backend, device, backend: str) -> None:
        super().__init__(model, attention_backend, device)
        self.compiled_model = torch.compile(model, backend=backend)

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
        execution_contexts = {}
        if self.execution_metadata_builders:
            if input_batch is None:
                raise RuntimeError("execution metadata builders require upstream InputBatch metadata")
            for builder in self.execution_metadata_builders:
                metadata_type = builder.metadata_type
                if metadata_type in execution_contexts:
                    raise RuntimeError(f"duplicate execution metadata builder for {metadata_type.__name__}")
                execution_contexts[metadata_type] = builder.build(input_batch, metadata)
        self.attention_backend.prepare(metadata)
        with forward_context(
            ForwardContext(
                self.attention_backend,
                self.device,
                metadata,
                self.layer_caches,
                layer_synchronizer=layer_synchronizer,
                eplb=eplb,
                execution_contexts=execution_contexts,
            )
        ):
            if input_embedding is None:
                return self.compiled_model(input_ids, positions)
            return self.compiled_model(input_ids, positions, input_embedding)
