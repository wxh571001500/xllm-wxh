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
# ==============================================================================

"""Execution metadata owned by the GLM5.3 EPLv2 model path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from xllm.python.model_executor.input_batch import InputBatch

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata


@dataclass(slots=True)
class Glm5NextEplv2Metadata:
    """Stable DP-local activity mask consumed by GLM EPLv2."""

    local_token_mask: torch.Tensor


class Glm5NextEplv2MetadataBuilder:
    """Build and update GLM EPLv2 execution metadata for eager and Graph."""

    metadata_type = Glm5NextEplv2Metadata

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> Glm5NextEplv2MetadataBuilder | None:
        if (
            str(config.get("model_type", "")) not in ("glm5_next", "glm5_next_text")
            or int(config.get("expert_parallel_degree", 0)) != 2
            or int(config.get("ep_size", 1)) <= 1
        ):
            return None
        return cls(
            dp_size=int(config.get("dp_size", 1)),
            dp_rank=int(config.get("dp_rank", 0)),
        )

    def __init__(self, dp_size: int, dp_rank: int) -> None:
        if dp_size <= 0 or not 0 <= dp_rank < dp_size:
            raise ValueError(f"invalid GLM EPLv2 DP geometry: size={dp_size}, rank={dp_rank}")
        self._dp_size = dp_size
        self._dp_rank = dp_rank

    def build(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> Glm5NextEplv2Metadata:
        mask = torch.zeros(
            input_batch.num_tokens,
            dtype=torch.bool,
            device=input_batch.input_ids.device,
        )
        self._update_mask(mask, input_batch, metadata)
        return Glm5NextEplv2Metadata(mask)

    def allocate_persistent(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> Glm5NextEplv2Metadata:
        capacity = input_batch.num_tokens_after_padding
        if capacity <= 0:
            raise ValueError("GLM EPLv2 graph token capacity must be positive")
        return Glm5NextEplv2Metadata(
            torch.zeros(
                capacity,
                dtype=torch.bool,
                device=input_batch.input_ids.device,
            )
        )

    def update_persistent(
        self,
        persistent_metadata: object,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> None:
        if not isinstance(persistent_metadata, Glm5NextEplv2Metadata):
            raise TypeError("GLM EPLv2 received invalid persistent execution metadata")
        self._update_mask(persistent_metadata.local_token_mask, input_batch, metadata)

    def _update_mask(
        self,
        mask: torch.Tensor,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> None:
        if mask.dim() != 1 or mask.device != input_batch.input_ids.device:
            raise ValueError("GLM EPLv2 local token mask must be a device one-dimensional tensor")
        if input_batch.num_tokens > mask.numel():
            raise RuntimeError(
                "GLM EPLv2 local token mask capacity is smaller than the execution batch: "
                f"tokens={input_batch.num_tokens}, capacity={mask.numel()}"
            )
        if self._dp_size > 1:
            counts = tuple(int(count) for count in getattr(metadata, "raw_dp_execution_token_counts", ()))
            if len(counts) != self._dp_size:
                raise RuntimeError("GLM EPLv2 local masks require raw execution counts for every DP rank")
            valid_rows = counts[self._dp_rank]
            if valid_rows < 0 or valid_rows > input_batch.num_tokens:
                raise RuntimeError(
                    "GLM EPLv2 local active-token rows exceed the execution batch: "
                    f"rows={valid_rows}, tokens={input_batch.num_tokens}"
                )
        else:
            valid_rows = 0 if input_batch.is_dummy else input_batch.num_tokens
        mask.zero_()
        mask[:valid_rows].fill_(True)
