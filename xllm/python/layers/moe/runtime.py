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

"""Per-forward runtime metadata shared by reusable MoE layers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class MoEBatchMetadata:
    """Token counts prepared once by the scheduler for one model forward."""

    local_num_tokens: int
    local_actual_tokens: int
    max_num_tokens: int

    def __post_init__(self) -> None:
        if self.local_num_tokens < 0:
            raise ValueError("MoE local token count must be non-negative")
        if not 0 <= self.local_actual_tokens <= self.local_num_tokens:
            raise ValueError(
                "MoE actual token count must be within the local token count"
            )
        if self.max_num_tokens < self.local_num_tokens:
            raise ValueError(
                "MoE maximum token count must cover the local token count"
            )


_CURRENT_BATCH_METADATA: ContextVar[MoEBatchMetadata | None] = ContextVar(
    "_CURRENT_BATCH_METADATA",
    default=None,
)


@contextmanager
def moe_batch_context(metadata: MoEBatchMetadata | None) -> Iterator[None]:
    """Make scheduler token metadata visible to every MoE layer."""

    token = _CURRENT_BATCH_METADATA.set(metadata)
    try:
        yield
    finally:
        _CURRENT_BATCH_METADATA.reset(token)


def get_moe_batch_metadata() -> MoEBatchMetadata | None:
    """Return metadata for the active model forward, if provided."""

    return _CURRENT_BATCH_METADATA.get()


__all__ = [
    "MoEBatchMetadata",
    "get_moe_batch_metadata",
    "moe_batch_context",
]
