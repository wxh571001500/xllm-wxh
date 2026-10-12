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

"""Materialized, model-execution view of one scheduler step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class InputBatchMetadata(Protocol):
    """Request-scoped metadata supplied by the C++ execution pipeline."""

    num_reqs: int
    num_tokens: int
    num_scheduled_tokens: list[int]
    num_computed_tokens: list[int]
    query_start_loc: list[int]
    is_prefilling: list[int]


@dataclass(frozen=True, slots=True)
class InputBatch:
    """Inputs and authoritative batch semantics consumed by Python execution.

    Request-scoped fields are copied from the C++ producer. Graph padding is
    added later by the graph runner and never changes the actual request or
    token counts. This initial contract covers non-speculative execution, where
    rows are packed contiguously by request.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    num_reqs: int
    num_tokens: int
    num_tokens_after_padding: int
    num_scheduled_tokens: tuple[int, ...]
    num_computed_tokens: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    is_prefilling: tuple[bool, ...]
    is_padding: torch.Tensor | None
    is_dummy: bool

    @classmethod
    def from_runtime(
        cls,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: InputBatchMetadata,
        *,
        is_dummy: bool,
    ) -> InputBatch:
        """Build an unpadded batch without inferring request semantics."""
        num_reqs = int(metadata.num_reqs)
        num_tokens = int(metadata.num_tokens)
        num_scheduled_tokens = tuple(int(value) for value in metadata.num_scheduled_tokens)
        num_computed_tokens = tuple(int(value) for value in metadata.num_computed_tokens)
        query_start_loc = tuple(int(value) for value in metadata.query_start_loc)
        is_prefilling = tuple(bool(value) for value in metadata.is_prefilling)

        cls._validate_runtime_fields(
            input_ids=input_ids,
            positions=positions,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            query_start_loc=query_start_loc,
            is_prefilling=is_prefilling,
            is_dummy=is_dummy,
        )
        return cls(
            input_ids=input_ids,
            positions=positions,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            query_start_loc=query_start_loc,
            is_prefilling=is_prefilling,
            is_padding=None,
            is_dummy=is_dummy,
        )

    def bind_graph_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_padding: torch.Tensor,
    ) -> InputBatch:
        """Bind this batch to a graph bucket's persistent input addresses."""
        token_capacity = input_ids.shape[0]
        if token_capacity < self.num_tokens:
            raise ValueError("graph token capacity is smaller than the actual token count")
        if self._position_token_count(positions) != token_capacity:
            raise ValueError("graph positions do not match the token capacity")
        if is_padding.shape != (token_capacity,):
            raise ValueError("graph padding mask does not match the token capacity")
        if is_padding.dtype != torch.bool:
            raise TypeError("graph padding mask must use bool")
        if is_padding.device != input_ids.device:
            raise ValueError("graph padding mask must be on the input device")

        return InputBatch(
            input_ids=input_ids,
            positions=positions,
            num_reqs=self.num_reqs,
            num_tokens=self.num_tokens,
            num_tokens_after_padding=token_capacity,
            num_scheduled_tokens=self.num_scheduled_tokens,
            num_computed_tokens=self.num_computed_tokens,
            query_start_loc=self.query_start_loc,
            is_prefilling=self.is_prefilling,
            is_padding=is_padding,
            is_dummy=self.is_dummy,
        )

    @classmethod
    def _validate_runtime_fields(
        cls,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
        num_scheduled_tokens: tuple[int, ...],
        num_computed_tokens: tuple[int, ...],
        query_start_loc: tuple[int, ...],
        is_prefilling: tuple[bool, ...],
        is_dummy: bool,
    ) -> None:
        if input_ids.dim() != 1:
            raise ValueError("InputBatch input_ids must be one-dimensional")
        if num_reqs < 0 or num_tokens < 0:
            raise ValueError("InputBatch counts must be non-negative")
        if input_ids.shape[0] != num_tokens:
            raise ValueError("upstream num_tokens does not match input_ids")
        if cls._position_token_count(positions) != num_tokens:
            raise ValueError("positions do not match upstream num_tokens")

        request_fields = (
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
        )
        if any(len(values) != num_reqs for values in request_fields):
            raise ValueError("request-scoped InputBatch metadata has inconsistent lengths")
        if any(value <= 0 for value in num_scheduled_tokens):
            raise ValueError("each non-dummy request must schedule at least one token")
        if any(value < 0 for value in num_computed_tokens):
            raise ValueError("num_computed_tokens must be non-negative")

        dummy_layout = is_dummy and num_reqs == 0 and num_tokens > 0
        if sum(num_scheduled_tokens) != num_tokens and not dummy_layout:
            raise ValueError("num_scheduled_tokens disagrees with num_tokens")

        if len(query_start_loc) != num_reqs + 1 or not query_start_loc or query_start_loc[0] != 0:
            raise ValueError("query_start_loc must contain one request boundary plus a leading zero")
        query_widths = tuple(end - start for start, end in zip(query_start_loc, query_start_loc[1:]))
        if query_widths != num_scheduled_tokens:
            raise ValueError("query_start_loc disagrees with num_scheduled_tokens")
        logical_num_tokens = query_start_loc[-1]
        if logical_num_tokens != num_tokens and not dummy_layout:
            raise ValueError("upstream request layout disagrees with num_tokens")

    @staticmethod
    def _position_token_count(positions: torch.Tensor) -> int:
        if positions.dim() == 1:
            return positions.shape[0]
        if positions.dim() == 2:
            return positions.shape[1]
        raise ValueError("InputBatch positions must be one- or two-dimensional")
