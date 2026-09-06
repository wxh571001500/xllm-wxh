# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Common configuration and stage contracts used by MoE layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class MoECommType(str, Enum):
    """Supported routed-expert communication strategies."""

    ALL_GATHER = "all_gather"
    ALL_TO_ALL = "all_to_all"
    MC2 = "mc2"
    AUTO = "auto"

    @classmethod
    def from_value(cls, value: str | MoECommType) -> MoECommType:
        if isinstance(value, cls):
            return value
        normalized = str(value).lower().replace("-", "_")
        aliases = {
            "allgather": cls.ALL_GATHER,
            "alltoall": cls.ALL_TO_ALL,
            "all2all": cls.ALL_TO_ALL,
        }
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)


@dataclass(frozen=True)
class MoEParallelConfig:
    """Parallel topology and communication policy for routed experts."""

    tp_size: int
    tp_rank: int
    ep_size: int
    ep_rank: int
    dp_size: int = 1
    dp_rank: int = 0
    comm_type: MoECommType = MoECommType.ALL_GATHER
    mc2_tokens_capacity: int = 512
    tp_group_name: str = "moe_tp"
    ep_group_name: str = "moe_ep"
    input_tp_size: int = 0
    input_tp_rank: int = -1
    input_tp_group_name: str = "tp"

    def __post_init__(self) -> None:
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("MoE TP rank and size are invalid")
        if self.ep_size <= 0 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("MoE EP rank and size are invalid")
        if self.dp_size <= 0 or not 0 <= self.dp_rank < self.dp_size:
            raise ValueError("MoE DP rank and size are invalid")
        if self.mc2_tokens_capacity <= 0:
            raise ValueError("MoE MC2 token capacity must be positive")
        input_tp_size = self.input_tp_size or self.tp_size
        input_tp_rank = self.tp_rank if self.input_tp_rank < 0 else self.input_tp_rank
        object.__setattr__(self, "input_tp_size", input_tp_size)
        object.__setattr__(self, "input_tp_rank", input_tp_rank)
        if input_tp_size <= 0 or not 0 <= input_tp_rank < input_tp_size:
            raise ValueError("MoE input TP rank and size are invalid")
        if input_tp_size != self.tp_size:
            supported = self.dp_size == 1 and self.tp_size == 1 and input_tp_size == self.ep_size
            if not supported:
                raise ValueError("MoE currently supports distinct input TP only for dp=1, moe_tp=1, and input_tp=ep")

    @property
    def partitions_replicated_input(self) -> bool:
        """Whether EP ranks receive replicated attention-TP tokens."""
        return self.input_tp_size != self.tp_size


@dataclass(frozen=True)
class MoERouterConfig:
    """Configuration for reusable expert routing strategies."""

    num_experts: int
    top_k: int
    scoring_func: str
    renormalize: bool
    routed_scaling_factor: float
    use_grouped_topk: bool = False
    num_expert_group: int = 1
    topk_group: int = 1


@dataclass(frozen=True)
class MoEExpertsConfig:
    """Shape and tensor-parallel configuration for routed experts."""

    num_experts: int
    hidden_size: int
    intermediate_size: int
    tp_size: int
    tp_rank: int
    ep_size: int = 1
    ep_rank: int = 0

    def __post_init__(self) -> None:
        if self.ep_size <= 0 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("MoE expert EP rank and size are invalid")
        if self.num_experts <= 0 or self.num_experts % self.ep_size != 0:
            raise ValueError("MoE experts must divide evenly across EP ranks")

    @property
    def num_local_experts(self) -> int:
        return self.num_experts // self.ep_size

    @property
    def first_expert_id(self) -> int:
        return self.ep_rank * self.num_local_experts


@dataclass(frozen=True)
class MoERoutingResult:
    """Selected experts and their contribution weights."""

    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass(frozen=True)
class MoEPrepareOutput:
    """Tensors prepared for routing and routed-expert execution."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    padded_hidden_states_shape: torch.Size | None = None
    active_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class MoETokenDispatchInput:
    """Input consumed by a token dispatcher."""

    hidden_states: torch.Tensor
    routing: MoERoutingResult
    active_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class MoETokenDispatchOutput:
    """Expert-grouped tokens and metadata required for combination."""

    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    combine_metadata: object
    dynamic_scale: torch.Tensor | None = None


@dataclass(frozen=True)
class MoEFusedExpertsResult:
    """Result returned by the routed-experts communication pipeline."""

    routed_out: torch.Tensor


__all__ = [
    "MoECommType",
    "MoEExpertsConfig",
    "MoEFusedExpertsResult",
    "MoEParallelConfig",
    "MoEPrepareOutput",
    "MoERouterConfig",
    "MoERoutingResult",
    "MoETokenDispatchInput",
    "MoETokenDispatchOutput",
]
