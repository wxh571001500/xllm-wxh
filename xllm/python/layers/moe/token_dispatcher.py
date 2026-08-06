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

"""Token dispatch and combine stages for routed experts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch_npu

from xllm.python.layers.moe.types import (
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
)


@dataclass(frozen=True)
class _NativeCombineMetadata:
    token_indices: torch.Tensor
    routing_weights: torch.Tensor
    num_tokens: int


@dataclass(frozen=True)
class _FusedCombineMetadata:
    expanded_row_indices: torch.Tensor
    routing_weights: torch.Tensor


class MoETokenDispatcher(ABC):
    """Dispatch tokens to experts and combine expert outputs."""

    @abstractmethod
    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput:
        raise NotImplementedError

    @abstractmethod
    def token_combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: object,
    ) -> torch.Tensor:
        raise NotImplementedError


class NativeTokenDispatcher(MoETokenDispatcher):
    """Portable token permutation used by the unquantized fallback path."""

    def __init__(self, num_experts: int) -> None:
        self._num_experts = num_experts

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput:
        hidden_states = token_dispatch_input.hidden_states
        routing = token_dispatch_input.routing
        num_tokens = hidden_states.shape[0]
        top_k = routing.topk_ids.shape[1]
        token_indices = (
            torch.arange(num_tokens, device=hidden_states.device)
            .unsqueeze(1)
            .expand(num_tokens, top_k)
            .reshape(-1)
        )
        expert_ids = routing.topk_ids.reshape(-1).to(torch.int64)
        sort_order = torch.argsort(expert_ids, stable=True)
        sorted_token_indices = token_indices.index_select(0, sort_order)
        sorted_expert_ids = expert_ids.index_select(0, sort_order)
        sorted_hidden_states = hidden_states.index_select(
            0,
            sorted_token_indices,
        )
        sorted_weights = routing.topk_weights.reshape(-1).index_select(
            0,
            sort_order,
        )
        group_list = torch.bincount(
            sorted_expert_ids,
            minlength=self._num_experts,
        ).to(torch.int64)
        return MoETokenDispatchOutput(
            hidden_states=sorted_hidden_states,
            group_list=group_list,
            group_list_type=1,
            combine_metadata=_NativeCombineMetadata(
                token_indices=sorted_token_indices,
                routing_weights=sorted_weights,
                num_tokens=num_tokens,
            ),
        )

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: object,
    ) -> torch.Tensor:
        if not isinstance(combine_metadata, _NativeCombineMetadata):
            raise TypeError("Native dispatcher received incompatible metadata")
        output = hidden_states.new_zeros(
            (combine_metadata.num_tokens, hidden_states.shape[-1])
        )
        weighted_output = hidden_states * combine_metadata.routing_weights.to(
            hidden_states
        ).unsqueeze(-1)
        output.index_add_(
            0,
            combine_metadata.token_indices,
            weighted_output,
        )
        return output


class FusedAllGatherTokenDispatcher(MoETokenDispatcher):
    """Fused token permutation matching the all-gather dispatcher path."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        quantized: bool,
    ) -> None:
        self._num_experts = num_experts
        self._top_k = top_k
        self._quantized = quantized

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput:
        hidden_states = token_dispatch_input.hidden_states
        routing = token_dispatch_input.routing
        num_tokens = hidden_states.shape[0]
        (
            sorted_hidden_states,
            expanded_row_indices,
            expert_tokens,
            dynamic_scale,
        ) = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            routing.topk_ids.to(torch.int32),
            scale=None,
            active_num=num_tokens * self._top_k,
            expert_num=self._num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self._num_experts],
            quant_mode=1 if self._quantized else -1,
        )
        return MoETokenDispatchOutput(
            hidden_states=sorted_hidden_states,
            group_list=expert_tokens.to(torch.int64),
            group_list_type=1,
            combine_metadata=_FusedCombineMetadata(
                expanded_row_indices=expanded_row_indices,
                routing_weights=routing.topk_weights,
            ),
            dynamic_scale=dynamic_scale if self._quantized else None,
        )

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: object,
    ) -> torch.Tensor:
        if not isinstance(combine_metadata, _FusedCombineMetadata):
            raise TypeError("Fused dispatcher received incompatible metadata")
        return torch_npu.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=combine_metadata.expanded_row_indices.abs(),
            probs=combine_metadata.routing_weights.to(hidden_states.dtype),
        )


__all__ = [
    "FusedAllGatherTokenDispatcher",
    "MoETokenDispatcher",
    "NativeTokenDispatcher",
]
