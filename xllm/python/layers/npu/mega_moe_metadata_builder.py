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

"""Metadata builder for NPU Token Owner MegaMoe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from xllm.python.layers.npu.mega_moe_metadata import MEGA_MOE_MAX_TOKENS, MegaMoeMetadata
from xllm.python.model_executor.input_batch import InputBatch

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata


class TokenOwnerMegaMoeMetadataBuilder:
    """Build eager metadata and maintain its persistent Graph representation."""

    metadata_type = MegaMoeMetadata

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> TokenOwnerMegaMoeMetadataBuilder | None:
        if not bool(config.get("enable_mega_moe", False)):
            return None
        return cls(
            is_token_owner=int(config.get("tp_rank", 0)) == 0,
            dp_size=int(config.get("dp_size", 1)),
            dp_rank=int(config.get("dp_rank", 0)),
            hidden_size=int(config.get("hidden_size", 0)),
            top_k=int(
                config.get(
                    "num_experts_per_tok",
                    config.get("num_experts_per_token", 0),
                )
            ),
            token_limit=int(config.get("mega_moe_num_max_tokens_per_rank", 0)),
        )

    def __init__(
        self,
        is_token_owner: bool,
        dp_size: int,
        dp_rank: int,
        hidden_size: int,
        top_k: int,
        token_limit: int,
    ) -> None:
        if dp_size <= 0:
            raise ValueError("MegaMoe DP size must be positive")
        if dp_rank < 0 or dp_rank >= dp_size:
            raise ValueError(f"MegaMoe DP rank {dp_rank} is outside DP size {dp_size}")
        if hidden_size <= 0:
            raise ValueError("MegaMoe hidden size must be positive")
        if top_k <= 0:
            raise ValueError("MegaMoe top-k must be positive")
        if token_limit <= 0:
            raise ValueError("MegaMoe token limit must be positive")
        self._is_token_owner = is_token_owner
        self._dp_size = dp_size
        self._dp_rank = dp_rank
        self._hidden_size = hidden_size
        self._top_k = top_k
        self._token_limit = token_limit

    def build(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> MegaMoeMetadata:
        execution_token_counts = self._execution_token_counts(
            input_batch,
            metadata,
        )
        token_capacity = max(execution_token_counts, default=input_batch.num_tokens)
        self._validate_token_count(input_batch.num_tokens, token_capacity)
        return self._build_metadata(
            token_capacity,
            input_batch.input_ids.device,
            input_batch.num_tokens,
        )

    def allocate_persistent(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> MegaMoeMetadata:
        token_capacity = input_batch.num_tokens_after_padding
        if token_capacity <= 0:
            raise ValueError("MegaMoe graph token capacity must be positive")
        return self._build_metadata(
            token_capacity,
            input_batch.input_ids.device,
            None,
        )

    def update_persistent(
        self,
        persistent_metadata: object,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> None:
        if not isinstance(persistent_metadata, MegaMoeMetadata):
            raise TypeError("Token Owner MegaMoe received invalid persistent metadata")

        token_capacity = persistent_metadata.active_token_mask.shape[0]
        execution_token_counts = self._execution_token_counts(
            input_batch,
            metadata,
        )
        if any(count > token_capacity for count in execution_token_counts):
            raise RuntimeError(
                "DP execution token counts must fit the MegaMoe graph capacity: "
                f"counts={execution_token_counts}, capacity={token_capacity}"
            )
        self._validate_token_count(input_batch.num_tokens, token_capacity)
        if input_batch.num_tokens_after_padding != token_capacity:
            raise RuntimeError(
                "MegaMoe metadata capacity does not match the execution batch: "
                f"metadata={token_capacity}, batch={input_batch.num_tokens_after_padding}"
            )
        if input_batch.is_padding is None:
            raise RuntimeError("MegaMoe graph execution requires a padding mask")
        if input_batch.is_padding.shape != (token_capacity,):
            raise RuntimeError("MegaMoe padding mask does not match the metadata capacity")

        if self._is_token_owner:
            persistent_metadata.active_token_mask.copy_(~input_batch.is_padding)

    def _build_metadata(
        self,
        token_capacity: int,
        device: torch.device,
        local_token_count: int | None,
    ) -> MegaMoeMetadata:
        active_token_mask = torch.zeros(
            token_capacity,
            dtype=torch.int8,
            device=device,
        )
        if self._is_token_owner:
            if local_token_count is None:
                active_token_mask.fill_(1)
            else:
                active_token_mask[:local_token_count].fill_(1)
        else:
            active_token_mask[0] = 1

        if self._is_token_owner or token_capacity > min(MEGA_MOE_MAX_TOKENS, self._token_limit):
            return MegaMoeMetadata(active_token_mask=active_token_mask)

        # MegaMoe treats these non-owner inputs as immutable, so all MoE
        # layers can safely share the same storage for one execution.
        return MegaMoeMetadata(
            active_token_mask=active_token_mask,
            dummy_input=torch.zeros(
                token_capacity,
                self._hidden_size,
                dtype=torch.bfloat16,
                device=device,
            ),
            dummy_topk_weights=torch.full(
                (token_capacity, self._top_k),
                1.0 / self._top_k,
                dtype=torch.float32,
                device=device,
            ),
            dummy_topk_ids=(
                torch.arange(
                    self._top_k,
                    dtype=torch.int32,
                    device=device,
                )
                .view(1, self._top_k)
                .expand(token_capacity, self._top_k)
                .contiguous()
            ),
        )

    def _execution_token_counts(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> tuple[int, ...]:
        execution_token_counts = tuple(
            int(count)
            for count in getattr(
                metadata,
                "dp_execution_token_counts",
                (),
            )
        )
        if not execution_token_counts and self._dp_size == 1:
            execution_token_counts = (input_batch.num_tokens,)
        if len(execution_token_counts) != self._dp_size:
            raise RuntimeError(f"expected {self._dp_size} DP execution token counts, got {execution_token_counts}")
        if any(count <= 0 for count in execution_token_counts):
            raise RuntimeError(f"MegaMoe requires positive DP execution token counts, got {execution_token_counts}")
        if execution_token_counts[self._dp_rank] != input_batch.num_tokens:
            raise RuntimeError(
                "DP execution token count does not match the local input: "
                f"rank={self._dp_rank}, rows={input_batch.num_tokens}, "
                f"token_counts={execution_token_counts}"
            )
        return execution_token_counts

    @staticmethod
    def _validate_token_count(
        local_token_count: int,
        token_capacity: int,
    ) -> None:
        if local_token_count <= 0 or local_token_count > token_capacity:
            raise RuntimeError(
                "local token count must fit the MegaMoe token capacity: "
                f"count={local_token_count}, capacity={token_capacity}"
            )
