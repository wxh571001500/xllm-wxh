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

"""Common configuration and stage contracts used by MoE layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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


@dataclass(frozen=True)
class MoETokenDispatchInput:
    """Input consumed by a token dispatcher."""

    hidden_states: torch.Tensor
    routing: MoERoutingResult


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
    "MoEExpertsConfig",
    "MoEFusedExpertsResult",
    "MoEPrepareOutput",
    "MoERouterConfig",
    "MoERoutingResult",
    "MoETokenDispatchInput",
    "MoETokenDispatchOutput",
]
