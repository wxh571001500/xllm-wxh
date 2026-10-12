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

"""Forward-scoped execution contexts contributed by optional features."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from xllm.python.model_executor.input_batch import InputBatch

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata, LayerCache

ExecutionContexts = dict[type[object], object]


class ExecutionMetadataBuilder(Protocol):
    """Builds model-visible metadata for eager and graph execution."""

    metadata_type: type[object]

    def build(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> object: ...

    def allocate_persistent(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> object: ...

    def update_persistent(
        self,
        persistent_metadata: object,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> None: ...


class RegisteredExecutionMetadataBuilder(ExecutionMetadataBuilder, Protocol):
    """Execution metadata builder constructible from model runtime config."""

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> RegisteredExecutionMetadataBuilder | None: ...


@runtime_checkable
class LayerCacheAwareExecutionMetadataBuilder(Protocol):
    """Optional builder capability for binding long-lived layer caches."""

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None: ...
