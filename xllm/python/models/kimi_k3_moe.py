# Copyright 2026 The xLLM Authors.
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

"""Kimi K3 MoE routing, expert computation, and communication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Integral
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from xllm.python import distributed
from xllm.python.model_executor.forward_context import get_execution_context, get_forward_context
from xllm.python.model_executor.forward_context import in_acl_graph as _in_acl_graph_capture

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


# torch_npu uses numeric SOC_VERSION values for the A3/A5 dispatch-v2 API.
_SOC_VERSION_A3_MIN = 250
_SOC_VERSION_A3_MAX = 255
_SOC_VERSION_A5 = 260
_SOC_VERSION_A5_MIN = 300


# Configuration and tensor metadata


class MoECommType(str, Enum):
    """Supported routed-expert communication strategies."""

    ALL_GATHER = "all_gather"
    ALL_TO_ALL = "all_to_all"
    MC2 = "mc2"
    AUTO = "auto"


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
            supported = supported or (self.tp_size == 1 and self.ep_size == self.dp_size * input_tp_size)
            if not supported:
                raise ValueError(
                    "MoE supports distinct input TP only for dp=1 with input_tp=ep, or global EP with ep=dp*input_tp"
                )

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


@dataclass(frozen=True, slots=True)
class MC2ExecutionContext:
    """Static MC2 bucket and live mask owned by a decode graph entry."""

    padded_num_tokens: int
    active_mask: torch.Tensor


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


# Expert activation


def _validate_situ_activation(beta: float, linear_beta: float | None) -> None:
    if beta <= 0:
        raise ValueError("SiTU beta must be positive")
    if linear_beta is not None and linear_beta <= 0:
        raise ValueError("SiTU linear_beta must be positive when enabled")


class SituAndMul(nn.Module):
    """SiTU gated activation used by routed and dense expert MLPs."""

    def __init__(self, beta: float, linear_beta: float | None) -> None:
        super().__init__()
        _validate_situ_activation(beta, linear_beta)
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] % 2 != 0:
            raise ValueError("SiTU input must have an even last dimension")
        width = tensor.shape[-1] // 2
        gate = tensor[..., :width].float()
        up = tensor[..., width:].float()
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (gate * up).to(dtype=tensor.dtype)


# Expert routing


class MoERouter(ABC):
    """Select routed experts from model-produced router logits."""

    @abstractmethod
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
    ) -> MoERoutingResult:
        raise NotImplementedError


class GroupedTopKRouter(MoERouter):
    """Grouped top-k router using PyTorch on CPU and torch_npu on NPU."""

    def __init__(self, config: MoERouterConfig) -> None:
        if config.top_k <= 0 or config.top_k > config.num_experts:
            raise ValueError("MoE top_k must be within the number of experts")
        if config.use_grouped_topk:
            if config.num_expert_group <= 0:
                raise ValueError("MoE num_expert_group must be positive")
            if config.num_experts % config.num_expert_group != 0:
                raise ValueError("MoE experts must divide evenly into groups")
            if not 0 < config.topk_group <= config.num_expert_group:
                raise ValueError("MoE topk_group must be within expert groups")
            if config.top_k > config.topk_group * (config.num_experts // config.num_expert_group):
                raise ValueError("MoE top_k exceeds the number of experts in selected groups")
        self.config = config

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
    ) -> MoERoutingResult:
        del hidden_states
        if router_logits.device.type in ("npu", "privateuseone"):
            return self._select_experts_npu(router_logits, correction_bias)
        return self._select_experts_native(router_logits, correction_bias)

    def _select_experts_npu(
        self,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
    ) -> MoERoutingResult:
        if self.config.scoring_func == "softmax":
            norm_type = 0
        elif self.config.scoring_func == "sigmoid":
            norm_type = 1
        else:
            raise ValueError(f"Unsupported MoE router activation: {self.config.scoring_func}")
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
            router_logits,
            k=self.config.top_k,
            k_group=self.config.topk_group if self.config.use_grouped_topk else 1,
            group_count=(self.config.num_expert_group if self.config.use_grouped_topk else 1),
            group_select_mode=1,
            renorm=int(self.config.renormalize),
            norm_type=norm_type,
            out_flag=False,
            routed_scaling_factor=self.config.routed_scaling_factor,
            eps=1e-20,
            bias=(correction_bias.to(router_logits) if correction_bias is not None else None),
        )
        return MoERoutingResult(
            topk_ids=topk_ids.to(torch.int32),
            topk_weights=topk_weights,
        )

    def _select_experts_native(
        self,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None,
    ) -> MoERoutingResult:
        if self.config.scoring_func == "softmax":
            scores = torch.softmax(router_logits, dim=-1)
        elif self.config.scoring_func == "sigmoid":
            scores = torch.sigmoid(router_logits)
        else:
            raise ValueError(f"Unsupported MoE router activation: {self.config.scoring_func}")

        selection = scores
        if correction_bias is not None:
            selection = selection + correction_bias.to(dtype=scores.dtype)
        if self.config.use_grouped_topk:
            selection = self._mask_unselected_groups(selection)

        topk_ids = torch.topk(
            selection.to(torch.float32),
            self.config.top_k,
            dim=-1,
        ).indices
        topk_weights = scores.gather(-1, topk_ids)
        if self.config.renormalize:
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1,
                keepdim=True,
            )
        topk_weights = topk_weights * self.config.routed_scaling_factor
        return MoERoutingResult(
            topk_ids=topk_ids.to(torch.int32),
            topk_weights=topk_weights,
        )

    def _mask_unselected_groups(self, selection: torch.Tensor) -> torch.Tensor:
        num_tokens = selection.shape[0]
        experts_per_group = self.config.num_experts // self.config.num_expert_group
        grouped = selection.view(
            num_tokens,
            self.config.num_expert_group,
            experts_per_group,
        )
        group_score_width = min(2, experts_per_group)
        group_scores = torch.topk(
            grouped,
            k=group_score_width,
            dim=-1,
        ).values.sum(dim=-1)
        selected_groups = torch.topk(
            group_scores,
            k=self.config.topk_group,
            dim=-1,
        ).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, selected_groups, True)
        expert_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(selection)
        return selection.masked_fill(~expert_mask, torch.finfo(selection.dtype).min)


# Token preparation and output reduction


def _all_gather_rows(
    tensor: torch.Tensor,
    *,
    dim: int,
    world_size: int,
    group_name: str,
) -> torch.Tensor:
    # The main collective uses all_gather_into_tensor for dim=0. Kimi's
    # validated ACL capture uses the list-based all_gather instead: a negative
    # leading dimension selects that existing main path without changing
    # token order, output shape, or the behavior of any other model.
    gather_dim = -tensor.ndim if dim == 0 and _in_acl_graph_capture() else dim
    return distributed.all_gather(tensor, dim=gather_dim, world_size=world_size, group_name=group_name)


class PrepareAndFinalize(ABC):
    """Prepare routed-expert inputs and finalize their output."""

    @abstractmethod
    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> MoEPrepareOutput:
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class _ExpertParallelPrepareAndFinalize(PrepareAndFinalize):
    """Shared EP token-padding and MoE-TP reduction helpers."""

    def __init__(self, config: MoEParallelConfig) -> None:
        self._config = config
        self._num_tokens = 0
        self._original_num_tokens = 0

    def _gather_token_counts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local_count = torch.tensor(
            [hidden_states.shape[0]],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        return _all_gather_rows(
            local_count,
            dim=0,
            world_size=self._config.ep_size,
            group_name=self._config.ep_group_name,
        )

    def _max_tokens_from_metadata(
        self,
        hidden_states: torch.Tensor,
    ) -> int | None:
        """Return the scheduler's DP max without a device-to-host sync.

        Global EP receives the same local DP token set on every attention-TP
        rank.  The scheduler already publishes those per-DP counts in host
        metadata, so repeating an HCCL all-gather for every MoE layer is
        unnecessary.  The counts are authoritative even when an empty DP
        shard carries one fake input row: its physical shape is intentionally
        different from the reported logical count.  Avoid a shape-dependent
        fallback, which would make only that rank enter the collective.
        """
        del hidden_states
        try:
            metadata = get_forward_context().metadata
        except (RuntimeError, AttributeError):
            return None
        raw_counts = getattr(metadata, "dp_execution_token_counts", None)
        if raw_counts is None:
            return None
        try:
            raw_counts = tuple(raw_counts)
        except TypeError as error:
            raise ValueError("dp_execution_token_counts must be an integer sequence") from error
        # Mainline metadata uses an empty tuple when scheduler counts are
        # unavailable. Preserve the collective fallback for that sentinel.
        if not raw_counts:
            return None
        if len(raw_counts) != self._config.dp_size:
            raise ValueError("dp_execution_token_counts length must match MoE DP size")
        if any(not isinstance(count, Integral) or isinstance(count, bool) for count in raw_counts):
            raise ValueError("dp_execution_token_counts must contain only integers")
        token_counts = tuple(int(count) for count in raw_counts)
        if any(count < 0 for count in token_counts):
            raise ValueError("dp_execution_token_counts cannot contain negative values")
        if not 0 <= self._config.dp_rank < len(token_counts):
            raise ValueError("MoE DP rank is outside dp_execution_token_counts")
        return max(token_counts, default=0)

    def _max_tokens_across_dp(self, hidden_states: torch.Tensor) -> int:
        max_tokens = self._max_tokens_from_metadata(hidden_states)
        if max_tokens is not None:
            return max_tokens
        token_counts = self._gather_token_counts(hidden_states)
        return int(token_counts.max().item())

    def _reduce_tp(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
    ) -> torch.Tensor:
        if reduce_results and self._config.tp_size > 1:
            if self._config.tp_group_name == "tp":
                distributed.all_reduce_(hidden_states)
            else:
                distributed.all_reduce_(
                    hidden_states,
                    group_name=self._config.tp_group_name,
                )
        return hidden_states

    def _partition_replicated_input(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> MoEPrepareOutput:
        self._num_tokens = hidden_states.shape[0]
        pad_size = (-self._num_tokens) % self._config.input_tp_size
        if pad_size > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
            router_logits = F.pad(router_logits, (0, 0, 0, pad_size))
        padded_hidden_states_shape = hidden_states.shape
        hidden_states = torch.tensor_split(
            hidden_states,
            self._config.input_tp_size,
            dim=0,
        )[self._config.input_tp_rank]
        router_logits = torch.tensor_split(
            router_logits,
            self._config.input_tp_size,
            dim=0,
        )[self._config.input_tp_rank]
        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            padded_hidden_states_shape=padded_hidden_states_shape,
        )

    def _gather_partitioned_output(
        self,
        hidden_states: torch.Tensor,
        padded_hidden_states_shape: torch.Size | None,
    ) -> torch.Tensor:
        if padded_hidden_states_shape is None:
            raise ValueError("MoE partitioned input requires its padded shape")
        hidden_states = _all_gather_rows(
            hidden_states,
            dim=0,
            world_size=self._config.input_tp_size,
            group_name=self._config.input_tp_group_name,
        )
        if hidden_states.shape != padded_hidden_states_shape:
            raise RuntimeError(
                "MoE input TP all-gather returned an unexpected shape: "
                f"expected {padded_hidden_states_shape}, got "
                f"{hidden_states.shape}"
            )
        return hidden_states[: self._num_tokens]


class AllGatherPrepareAndFinalize(_ExpertParallelPrepareAndFinalize):
    """Gather EP tokens before experts and reduce-scatter their outputs."""

    def __init__(self, config: MoEParallelConfig) -> None:
        super().__init__(config)
        self._compact_dp_tokens = False

    def _use_compact_dp_tokens(self) -> bool:
        if not (self._config.partitions_replicated_input and self._config.dp_size > 1):
            return False
        # Warmup and capture must record the same collectives. Equal physical
        # buckets allow a static DP gather; they do not make distinct DP token
        # sets identical. This path takes precedence over partitioning below
        # in both prepare and finalize, including on idle DP ranks.
        return _in_acl_graph_capture()

    def _use_partitioned_global_tokens(self) -> bool:
        """Gather distinct DP token sets before evaluating local experts.

        An EP all-reduce alone is valid only when all EP ranks process the
        same tokens. With DP > 1, equal graph bucket sizes still contain
        distinct DP token sets, so the replicated-token shortcut is invalid
        in both eager and graph execution, including on idle DP shards.
        The compact graph path is selected first in prepare/finalize and
        gathers DP tokens without partitioning attention-TP replicas. The
        capture check below applies only to DP=1, where all ranks share the
        same tokens and an EP all-reduce is valid.
        """
        if self._config.dp_size > 1:
            return True
        if _in_acl_graph_capture():
            return False
        return _in_prefill()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> MoEPrepareOutput:
        self._num_tokens = hidden_states.shape[0]
        self._compact_dp_tokens = self._use_compact_dp_tokens()
        if self._compact_dp_tokens:
            # Attention-TP ranks already hold replicas. Gathering only across
            # DP avoids padding a one-token batch to input_tp_size on each DP.
            hidden_states = _all_gather_rows(hidden_states, dim=0, world_size=self._config.dp_size, group_name="dp")
            router_logits = _all_gather_rows(router_logits, dim=0, world_size=self._config.dp_size, group_name="dp")
            return MoEPrepareOutput(hidden_states, router_logits, padded_hidden_states_shape=hidden_states.shape)
        if self._config.ep_size == 1:
            return MoEPrepareOutput(hidden_states, router_logits)

        if self._config.partitions_replicated_input:
            if not self._use_partitioned_global_tokens():
                # Legacy decode path: each EP rank executes its local experts
                # on replicated tokens and sums with an EP all-reduce.
                return MoEPrepareOutput(hidden_states, router_logits)
            # With global EP, each attention-TP rank starts with a replica of
            # the same DP token set. Partition it first so the subsequent EP
            # all-gather reconstructs the global token set exactly once.
            self._original_num_tokens = hidden_states.shape[0]
            if self._config.dp_size > 1:
                if _in_acl_graph_capture():
                    # Graph warmup/capture uses a static bucket on every DP
                    # rank.  The physical input shape is already equal, and
                    # querying token counts here would record a host sync in
                    # the decode graph.
                    max_tokens = self._num_tokens
                else:
                    # Token counts can differ across DP groups. HCCL
                    # all-gather requires identical input shapes, so pad
                    # every rank's full DP token set to the global maximum
                    # before splitting.
                    max_tokens = self._max_tokens_across_dp(hidden_states)
                pad_size = max_tokens - hidden_states.shape[0]
                if pad_size > 0:
                    hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
                    router_logits = F.pad(router_logits, (0, 0, 0, pad_size))
            prepared = self._partition_replicated_input(
                hidden_states,
                router_logits,
            )
            prepared = replace(
                prepared,
                hidden_states=_all_gather_rows(
                    prepared.hidden_states,
                    dim=0,
                    world_size=self._config.ep_size,
                    group_name=self._config.ep_group_name,
                ),
                router_logits=_all_gather_rows(
                    prepared.router_logits,
                    dim=0,
                    world_size=self._config.ep_size,
                    group_name=self._config.ep_group_name,
                ),
            )
            prepared = replace(
                prepared,
                padded_hidden_states_shape=prepared.hidden_states.shape,
            )
            return prepared

        if _in_acl_graph_capture():
            # Decode Graph pads every DP rank to the same static bucket.
            max_tokens = self._num_tokens
        else:
            max_tokens = self._max_tokens_across_dp(hidden_states)
        pad_size = max_tokens - self._num_tokens
        if pad_size > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
            router_logits = F.pad(router_logits, (0, 0, 0, pad_size))
        hidden_states = _all_gather_rows(
            hidden_states,
            dim=0,
            world_size=self._config.ep_size,
            group_name=self._config.ep_group_name,
        )
        router_logits = _all_gather_rows(
            router_logits,
            dim=0,
            world_size=self._config.ep_size,
            group_name=self._config.ep_group_name,
        )
        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            padded_hidden_states_shape=hidden_states.shape,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        del padded_hidden_states_shape
        if self._compact_dp_tokens:
            # Global EP partitions experts across both attention TP and DP.
            # Sum local-node expert partitions, then sum/scatter across nodes.
            distributed.all_reduce_(hidden_states, group_name=self._config.input_tp_group_name)
            return distributed.reduce_scatter(hidden_states, world_size=self._config.dp_size, group_name="dp")[
                : self._num_tokens
            ]
        if self._config.partitions_replicated_input:
            if not self._use_partitioned_global_tokens():
                # Legacy decode path: sum partial expert outputs across EP and
                # keep the replicated token layout expected by the graph.
                if self._config.ep_size > 1:
                    distributed.all_reduce_(
                        hidden_states,
                        group_name=self._config.ep_group_name,
                    )
                return hidden_states
            if self._config.ep_size > 1:
                hidden_states = distributed.reduce_scatter(
                    hidden_states,
                    world_size=self._config.ep_size,
                    group_name=self._config.ep_group_name,
                )
            hidden_states = _all_gather_rows(
                hidden_states,
                dim=0,
                world_size=self._config.input_tp_size,
                group_name=self._config.input_tp_group_name,
            )
            return hidden_states[: self._original_num_tokens]
        if self._config.ep_size > 1:
            hidden_states = distributed.reduce_scatter(
                hidden_states,
                world_size=self._config.ep_size,
                group_name=self._config.ep_group_name,
            )
            hidden_states = hidden_states[: self._num_tokens]
        return self._reduce_tp(hidden_states, reduce_results)


class AllToAllPrepareAndFinalize(_ExpertParallelPrepareAndFinalize):
    """Keep local tokens for explicit EP all-to-all dispatch/combine."""

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> MoEPrepareOutput:
        if self._config.partitions_replicated_input:
            return self._partition_replicated_input(
                hidden_states,
                router_logits,
            )
        self._num_tokens = hidden_states.shape[0]
        return MoEPrepareOutput(hidden_states, router_logits)

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        hidden_states = self._reduce_tp(hidden_states, reduce_results)
        if self._config.partitions_replicated_input:
            return self._gather_partitioned_output(
                hidden_states,
                padded_hidden_states_shape,
            )
        return hidden_states


class MC2PrepareAndFinalize(_ExpertParallelPrepareAndFinalize):
    """Pad uneven EP batches and provide MC2's active-token mask."""

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> MoEPrepareOutput:
        self._num_tokens = hidden_states.shape[0]
        context = get_forward_context()
        mc2_context = get_execution_context(MC2ExecutionContext)
        context_pad = mc2_context.padded_num_tokens if mc2_context is not None else None
        context_mask = mc2_context.active_mask if mc2_context is not None else None
        if self._config.partitions_replicated_input:
            tp_size = self._config.input_tp_size
            tp_rank = self._config.input_tp_rank
            if context_pad is None:
                if _in_acl_graph_capture():
                    raise ValueError("MC2 graph execution requires a static token bucket and mask")
                max_tokens = max(self._max_tokens_across_dp(hidden_states), self._num_tokens)
                padded_tokens = (max_tokens + tp_size - 1) // tp_size * tp_size
            else:
                padded_tokens = int(context_pad)
            if padded_tokens < self._num_tokens or padded_tokens % tp_size:
                raise ValueError("MC2 token bucket must cover the input and be divisible by input TP")
            if padded_tokens > self._config.mc2_tokens_capacity:
                raise ValueError(f"MC2 token count {padded_tokens} exceeds capacity {self._config.mc2_tokens_capacity}")
            if context_mask is not None:
                if context_mask.dtype != torch.bool or tuple(context_mask.shape) != (padded_tokens,):
                    raise ValueError("MC2 mask must cover the complete padded input before TP partitioning")
                active_mask = context_mask
            else:
                if _in_acl_graph_capture():
                    raise ValueError("MC2 graph execution requires a static active-token mask")
                counts = getattr(
                    context.metadata,
                    "dp_execution_token_counts",
                    (),
                )
                logical_tokens = (
                    int(counts[self._config.dp_rank]) if len(counts) == self._config.dp_size else self._num_tokens
                )
                active_mask = torch.arange(padded_tokens, device=hidden_states.device) < logical_tokens
            pad_size = padded_tokens - self._num_tokens
            if pad_size:
                hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = F.pad(router_logits, (0, 0, 0, pad_size))
            padded_shape = hidden_states.shape
            return MoEPrepareOutput(
                hidden_states=torch.tensor_split(hidden_states, tp_size, dim=0)[tp_rank],
                router_logits=torch.tensor_split(router_logits, tp_size, dim=0)[tp_rank],
                padded_hidden_states_shape=padded_shape,
                active_mask=torch.tensor_split(active_mask, tp_size, dim=0)[tp_rank],
            )
        if context_pad is not None:
            max_tokens = int(context_pad)
            if max_tokens < self._num_tokens or max_tokens > self._config.mc2_tokens_capacity:
                raise ValueError("MC2 static token bucket must cover the input and fit capacity")
            if context_mask is None or context_mask.dtype != torch.bool or tuple(context_mask.shape) != (max_tokens,):
                raise ValueError("MC2 static active-token mask must cover the token bucket")
            pad_size = max_tokens - self._num_tokens
            return MoEPrepareOutput(
                hidden_states=F.pad(hidden_states, (0, 0, 0, pad_size)),
                router_logits=F.pad(router_logits, (0, 0, 0, pad_size)),
                padded_hidden_states_shape=torch.Size((max_tokens, hidden_states.shape[1])),
                active_mask=context_mask,
            )
        if _in_acl_graph_capture():
            raise ValueError("MC2 graph execution requires a static token bucket and mask")
        if self._config.ep_size == 1:
            active_mask = torch.ones(
                self._num_tokens,
                dtype=torch.bool,
                device=hidden_states.device,
            )
            return MoEPrepareOutput(
                hidden_states,
                router_logits,
                active_mask=active_mask,
            )

        token_counts = self._gather_token_counts(hidden_states)
        max_tokens = int(token_counts.max().item())
        if max_tokens > self._config.mc2_tokens_capacity:
            raise ValueError(f"MC2 token count {max_tokens} exceeds capacity {self._config.mc2_tokens_capacity}")
        active_mask = (
            torch.arange(
                max_tokens,
                device=hidden_states.device,
            )
            < self._num_tokens
        )
        pad_size = max_tokens - self._num_tokens
        if pad_size > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
            router_logits = F.pad(router_logits, (0, 0, 0, pad_size))
        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            padded_hidden_states_shape=hidden_states.shape,
            active_mask=active_mask,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        hidden_states = self._reduce_tp(hidden_states, reduce_results)
        if self._config.partitions_replicated_input:
            return self._gather_partitioned_output(
                hidden_states,
                padded_hidden_states_shape,
            )
        return hidden_states[: self._num_tokens]


# Token dispatch and combine


def _in_prefill() -> bool:
    """Return whether the active eager step is a prefill step."""
    try:
        metadata = get_forward_context().metadata
        return bool(metadata is not None and (metadata.is_prefill or metadata.is_chunked_prefill))
    except (RuntimeError, AttributeError):
        return False


@dataclass(frozen=True)
class _NativeCombineMetadata:
    token_indices: torch.Tensor
    routing_weights: torch.Tensor
    num_tokens: int


@dataclass(frozen=True)
class _FusedCombineMetadata:
    expanded_row_indices: torch.Tensor
    routing_weights: torch.Tensor


@dataclass(frozen=True)
class _AllToAllCombineMetadata:
    input_splits: list[int]
    output_splits: list[int]
    topk_weights: torch.Tensor
    reversed_local_input_permutation_mapping: torch.Tensor
    reversed_global_input_permutation_mapping: torch.Tensor | None
    hidden_shape: torch.Size
    hidden_shape_before_permute: torch.Size


@dataclass(frozen=True)
class _LegacyAllToAllCombineMetadata:
    input_splits: list[int]
    output_splits: list[int]
    token_indices: torch.Tensor
    routing_weights: torch.Tensor
    inverse_receive_order: torch.Tensor
    num_tokens: int


def _all_to_all(
    x: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group_name: str,
) -> torch.Tensor:
    """Exchange token rows eagerly using PyTorch without a shared custom op."""
    if group_name != "moe_ep":
        raise ValueError("Kimi MoE uses the mainline moe_ep group")
    group = distributed.moe_ep_group(x.device)
    world_size = group.size()
    if len(output_split_sizes) != world_size:
        raise ValueError("all-to-all output splits must match group size")
    if len(input_split_sizes) != world_size:
        raise ValueError("all-to-all input splits must match group size")
    if any(size < 0 for size in (*input_split_sizes, *output_split_sizes)):
        raise ValueError("all-to-all splits must be nonnegative")
    if sum(input_split_sizes) != x.shape[0]:
        raise ValueError("all-to-all input splits must match tensor size")
    output = x.new_empty((sum(output_split_sizes), *x.shape[1:]))
    if world_size == 1:
        if output_split_sizes != input_split_sizes:
            raise ValueError("single-rank all-to-all input and output splits must match")
        output.copy_(x)
        return output
    dist.all_to_all_single(
        output,
        x.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


def _async_all_to_all(
    input_tensor: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group_name: str,
) -> tuple[torch.Tensor, object | None]:
    """vLLM-style all-to-all-v wrapper used by eager prefill only."""
    if group_name != "moe_ep":
        raise ValueError("Kimi MoE uses the mainline moe_ep group")
    group = distributed.moe_ep_group(input_tensor.device)
    if group.size() == 1:
        # Reuse the synchronous adapter's split validation and local copy.
        return _all_to_all(input_tensor, output_split_sizes, input_split_sizes, group_name), None
    output = input_tensor.new_empty(
        (sum(output_split_sizes), *input_tensor.shape[1:]),
    )
    handle = dist.all_to_all_single(
        output,
        input_tensor.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=True,
    )
    return output, handle


@dataclass(frozen=True)
class _MC2CombineMetadata:
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    ep_recv_counts: torch.Tensor
    tp_recv_counts: torch.Tensor
    assist_info: torch.Tensor
    expand_scales: torch.Tensor
    active_mask: torch.Tensor | None


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

    def __init__(
        self,
        num_experts: int,
        first_expert_id: int = 0,
        num_local_experts: int | None = None,
    ) -> None:
        self._num_experts = num_experts
        self._first_expert_id = first_expert_id
        self._num_local_experts = num_local_experts or num_experts

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput:
        hidden_states = token_dispatch_input.hidden_states
        routing = token_dispatch_input.routing
        num_tokens = hidden_states.shape[0]
        top_k = routing.topk_ids.shape[1]
        token_indices = (
            torch.arange(num_tokens, device=hidden_states.device).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
        )
        expert_ids = routing.topk_ids.reshape(-1).to(torch.int64)
        local_mask = expert_ids.ge(self._first_expert_id) & expert_ids.lt(
            self._first_expert_id + self._num_local_experts
        )
        token_indices = token_indices[local_mask]
        expert_ids = expert_ids[local_mask] - self._first_expert_id
        routing_weights = routing.topk_weights.reshape(-1)[local_mask]
        sort_order = torch.argsort(expert_ids, stable=True)
        sorted_token_indices = token_indices.index_select(0, sort_order)
        sorted_expert_ids = expert_ids.index_select(0, sort_order)
        sorted_hidden_states = hidden_states.index_select(
            0,
            sorted_token_indices,
        )
        sorted_weights = routing_weights.index_select(0, sort_order)
        group_list = torch.bincount(
            sorted_expert_ids,
            minlength=self._num_local_experts,
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
        output = hidden_states.new_zeros((combine_metadata.num_tokens, hidden_states.shape[-1]))
        weighted_output = hidden_states * combine_metadata.routing_weights.to(hidden_states).unsqueeze(-1)
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
        first_expert_id: int = 0,
        num_local_experts: int | None = None,
    ) -> None:
        self._num_experts = num_experts
        self._top_k = top_k
        self._quantized = quantized
        self._first_expert_id = first_expert_id
        self._num_local_experts = num_local_experts or num_experts

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput:
        hidden_states = token_dispatch_input.hidden_states
        routing = token_dispatch_input.routing
        local_mask = routing.topk_ids.ge(self._first_expert_id) & routing.topk_ids.lt(
            self._first_expert_id + self._num_local_experts
        )
        routing_weights = routing.topk_weights * local_mask.to(routing.topk_weights.dtype)
        num_tokens = hidden_states.shape[0]
        # N * K is the full routing capacity, as in the mainline MoE kernels.
        # An active expert range does not bound how many routes land locally:
        # all N * K routes may select this rank's experts. Keep the worst-case
        # capacity for both eager execution and graph capture. Local group
        # counts delimit the valid expert rows; unselected routes have zero weight.
        # Computing a smaller value from device routing would add a host sync
        # and make output shapes data-dependent during graph capture.
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
            active_expert_range=[
                self._first_expert_id,
                self._first_expert_id + self._num_local_experts,
            ],
            quant_mode=1 if self._quantized else -1,
        )
        # The API returns a global-sized count buffer with the active range
        # packed into its prefix, as in the mainline DeepSeek V4 path.
        expert_tokens = expert_tokens[: self._num_local_experts]
        # Keep the fixed active_num capacity for eager and graph execution.
        # The grouped matmul consumes the device-side group counts, so slicing
        # to their sum here would add a per-layer device-to-host sync.
        return MoETokenDispatchOutput(
            hidden_states=sorted_hidden_states,
            group_list=expert_tokens.to(torch.int64),
            group_list_type=1,
            combine_metadata=_FusedCombineMetadata(
                expanded_row_indices=expanded_row_indices,
                routing_weights=routing_weights,
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


class VllmAllToAllTokenDispatcher(MoETokenDispatcher):
    """vLLM-Ascend all-to-all dispatcher for eager sequence-parallel prefill."""

    def __init__(
        self,
        config: MoEParallelConfig,
        num_experts: int,
        quantized: bool,
    ) -> None:
        if num_experts % config.ep_size != 0:
            raise ValueError("MoE experts must divide evenly across EP ranks")
        self._config = config
        self._num_experts = num_experts
        self._num_local_experts = num_experts // config.ep_size
        self._first_expert_id = config.ep_rank * self._num_local_experts
        self._quantized = quantized
        self._expert_ids_per_ep_rank_by_device: dict[str, torch.Tensor] = {}
        if self._num_local_experts > 1:
            self._expert_ids_per_ep_rank = (
                torch.arange(
                    self._num_experts,
                    dtype=torch.int32,
                )
                % self._num_local_experts
            )

    def _expert_ids_for_device(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        indices = self._expert_ids_per_ep_rank_by_device.get(key)
        if indices is None:
            indices = self._expert_ids_per_ep_rank.to(device=device, dtype=torch.int64)
            self._expert_ids_per_ep_rank_by_device[key] = indices
        return indices

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput:
        if _in_acl_graph_capture():
            raise RuntimeError(
                "Variable-split All2All dispatch is eager-only; use static AllGather for graph execution"
            )
        hidden_states = token_dispatch_input.hidden_states
        routing = token_dispatch_input.routing
        # npu_moe_token_permute uses the fused path only with int64 expert
        # indices on this Ascend runtime.  The router keeps int32 ids for
        # MC2/routing kernels, so widen only in the vLLM-style prefill path.
        topk_ids = routing.topk_ids.to(torch.int64)
        topk_weights = routing.topk_weights
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        # Match vLLM-Ascend: compute variable splits from token counts, then
        # use the fused NPU permutation instead of Python sorting and a second
        # expert-id all-to-all.
        local_counts = torch.histc(
            topk_ids.to(torch.float32),
            bins=self._num_experts,
            min=0,
            max=self._num_experts,
        )
        # histc returns exactly num_experts bins. The frozen EP config and
        # constructor divisibility check guarantee
        # num_experts == ep_size * num_local_experts for this reshape.
        # Variable All2All splits must be host lists. This intentional sync
        # is restricted to eager execution by the guard above.
        input_splits = (
            local_counts.reshape(self._config.ep_size, self._num_local_experts)
            .sum(dim=1)
            .to(device="cpu", dtype=torch.int64)
            .tolist()
        )
        global_counts = distributed.all_gather(
            local_counts,
            dim=0,
            world_size=self._config.ep_size,
            group_name=self._config.ep_group_name,
        ).reshape(self._config.ep_size, self._num_experts)
        local_slice = slice(self._first_expert_id, self._first_expert_id + self._num_local_experts)
        global_local_counts = global_counts[:, local_slice]
        output_splits = global_local_counts.sum(dim=1).to(device="cpu", dtype=torch.int64).tolist()
        group_list = global_local_counts.sum(dim=0).to(torch.int64)

        num_out_tokens = topk_ids.numel()
        permuted, reversed_local = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids,
            num_out_tokens=num_out_tokens,
        )

        dynamic_scale = None
        scale_handle = None
        if self._quantized:
            if permuted.device.type not in ("npu", "privateuseone"):
                raise RuntimeError("Quantized All2All MoE requires an NPU")
            permuted, dynamic_scale = torch_npu.npu_dynamic_quant(permuted)
            dynamic_scale, scale_handle = _async_all_to_all(
                dynamic_scale,
                output_splits,
                input_splits,
                self._config.ep_group_name,
            )
        grouped_hidden_states, token_handle = _async_all_to_all(
            permuted,
            output_splits,
            input_splits,
            self._config.ep_group_name,
        )
        if scale_handle is not None:
            scale_handle.wait()
        if token_handle is not None:
            token_handle.wait()

        reversed_global = None
        if self._num_local_experts > 1:
            global_expert_indices = torch.repeat_interleave(
                self._expert_ids_for_device(grouped_hidden_states.device),
                global_local_counts.reshape(-1).to(torch.int64),
            )
            grouped_hidden_states, reversed_global = torch_npu.npu_moe_token_permute(
                tokens=grouped_hidden_states,
                indices=global_expert_indices,
                num_out_tokens=grouped_hidden_states.shape[0],
            )
            if dynamic_scale is not None:
                dynamic_scale, _ = torch_npu.npu_moe_token_permute(
                    tokens=dynamic_scale.unsqueeze(-1),
                    indices=global_expert_indices,
                )
                dynamic_scale = dynamic_scale.squeeze(-1)
        return MoETokenDispatchOutput(
            hidden_states=grouped_hidden_states,
            group_list=group_list,
            group_list_type=1,
            combine_metadata=_AllToAllCombineMetadata(
                input_splits=input_splits,
                output_splits=output_splits,
                topk_weights=topk_weights,
                reversed_local_input_permutation_mapping=reversed_local,
                reversed_global_input_permutation_mapping=reversed_global,
                hidden_shape=hidden_shape,
                hidden_shape_before_permute=hidden_states.shape,
            ),
            dynamic_scale=dynamic_scale,
        )

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: object,
    ) -> torch.Tensor:
        if not isinstance(combine_metadata, _AllToAllCombineMetadata):
            raise TypeError("All2All dispatcher received incompatible metadata")

        if self._num_local_experts > 1 and combine_metadata.reversed_global_input_permutation_mapping is not None:
            hidden_states = torch_npu.npu_moe_token_unpermute(
                permuted_tokens=hidden_states,
                sorted_indices=combine_metadata.reversed_global_input_permutation_mapping,
            )
        returned, handle = _async_all_to_all(
            hidden_states,
            combine_metadata.input_splits,
            combine_metadata.output_splits,
            self._config.ep_group_name,
        )
        if handle is not None:
            handle.wait()
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=returned,
            sorted_indices=combine_metadata.reversed_local_input_permutation_mapping.to(torch.int32),
            probs=combine_metadata.topk_weights,
            restore_shape=combine_metadata.hidden_shape_before_permute,
        )
        return output.view(combine_metadata.hidden_shape)


class AllToAllTokenDispatcher(MoETokenDispatcher):
    """Legacy token-level all-to-all dispatcher used by decode."""

    def __init__(self, config: MoEParallelConfig, num_experts: int, quantized: bool) -> None:
        if num_experts % config.ep_size != 0:
            raise ValueError("MoE experts must divide evenly across EP ranks")
        self._config = config
        self._num_experts = num_experts
        self._num_local_experts = num_experts // config.ep_size
        self._first_expert_id = config.ep_rank * self._num_local_experts
        self._quantized = quantized

    def token_dispatch(self, token_dispatch_input: MoETokenDispatchInput) -> MoETokenDispatchOutput:
        hidden_states = token_dispatch_input.hidden_states
        routing = token_dispatch_input.routing
        num_tokens = hidden_states.shape[0]
        top_k = routing.topk_ids.shape[1]
        token_indices = (
            torch.arange(num_tokens, device=hidden_states.device).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
        )
        expert_ids = routing.topk_ids.reshape(-1).to(torch.int64)
        routing_weights = routing.topk_weights.reshape(-1)
        send_order = torch.argsort(expert_ids, stable=True)
        sent_expert_ids = expert_ids.index_select(0, send_order)
        sent_hidden_states = hidden_states.index_select(0, token_indices.index_select(0, send_order))
        local_counts = torch.histc(
            expert_ids.to(torch.float32), bins=self._num_experts, min=0, max=self._num_experts
        ).to(torch.int64)
        global_counts = distributed.all_gather(
            local_counts,
            dim=0,
            world_size=self._config.ep_size,
            group_name=self._config.ep_group_name,
        ).view(self._config.ep_size, self._num_experts)
        input_splits = local_counts.view(self._config.ep_size, self._num_local_experts).sum(dim=1).to("cpu").tolist()
        local_slice = slice(self._first_expert_id, self._first_expert_id + self._num_local_experts)
        output_splits = global_counts[:, local_slice].sum(dim=1).to("cpu").tolist()
        received_hidden_states = _all_to_all(
            sent_hidden_states,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group_name=self._config.ep_group_name,
        )
        received_expert_ids = _all_to_all(
            sent_expert_ids,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group_name=self._config.ep_group_name,
        )
        local_expert_ids = received_expert_ids - self._first_expert_id
        receive_order = torch.argsort(local_expert_ids, stable=True)
        grouped_hidden_states = received_hidden_states.index_select(0, receive_order)
        group_list = torch.bincount(
            local_expert_ids.index_select(0, receive_order), minlength=self._num_local_experts
        ).to(torch.int64)
        dynamic_scale = None
        if self._quantized:
            grouped_hidden_states, dynamic_scale = torch_npu.npu_dynamic_quant(grouped_hidden_states)
        return MoETokenDispatchOutput(
            hidden_states=grouped_hidden_states,
            group_list=group_list,
            group_list_type=1,
            combine_metadata=_LegacyAllToAllCombineMetadata(
                input_splits=input_splits,
                output_splits=output_splits,
                token_indices=token_indices.index_select(0, send_order),
                routing_weights=routing_weights.index_select(0, send_order),
                inverse_receive_order=torch.argsort(receive_order, stable=True),
                num_tokens=num_tokens,
            ),
            dynamic_scale=dynamic_scale,
        )

    def token_combine(self, hidden_states: torch.Tensor, combine_metadata: object) -> torch.Tensor:
        if not isinstance(combine_metadata, _LegacyAllToAllCombineMetadata):
            raise TypeError("Legacy All2All dispatcher received incompatible metadata")
        received_order = hidden_states.index_select(0, combine_metadata.inverse_receive_order)
        returned = _all_to_all(
            received_order,
            output_split_sizes=combine_metadata.input_splits,
            input_split_sizes=combine_metadata.output_splits,
            group_name=self._config.ep_group_name,
        )
        output = returned.new_zeros((combine_metadata.num_tokens, returned.shape[-1]))
        weighted = returned * combine_metadata.routing_weights.to(returned.dtype).unsqueeze(-1)
        output.index_add_(0, combine_metadata.token_indices, weighted)
        return output


class MC2TokenDispatcher(MoETokenDispatcher):
    """Use Ascend MC2 dispatch/combine operators over the EP group."""

    def __init__(
        self,
        config: MoEParallelConfig,
        num_experts: int,
        quantized: bool,
        device: torch.device,
    ) -> None:
        if device.type not in ("npu", "privateuseone"):
            raise RuntimeError("MC2 MoE requires an NPU")
        if config.ep_group_name != "moe_ep":
            raise ValueError("Kimi MC2 requires the mainline moe_ep group")
        self._config = config
        self._num_experts = num_experts
        self._quantized = quantized
        self._enable_v2 = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
        soc_version = int(torch_npu.npu.get_soc_version())
        # A3/A5 dispatch-v2 requires the explicit TP arguments even when the
        # MoE TP group is logically size one. This matches vLLM-Ascend's
        # A3/A5 path and avoids dispatch tiling with an incomplete contract.
        self._need_tp_arguments = (
            _SOC_VERSION_A3_MIN <= soc_version <= _SOC_VERSION_A3_MAX
            or soc_version == _SOC_VERSION_A5
            or soc_version >= _SOC_VERSION_A5_MIN
        )
        self._need_expert_scales = soc_version == _SOC_VERSION_A5
        self._group_name, rank, world_size = distributed.moe_ep_hccl_info(device)
        if rank != config.ep_rank or world_size != config.ep_size:
            raise RuntimeError("MC2 communicator does not match the expert partition")

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput:
        routing = token_dispatch_input.routing
        kwargs = {
            "x": token_dispatch_input.hidden_states,
            "expert_ids": routing.topk_ids.to(torch.int32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self._num_experts,
            "global_bs": 0,
            "expert_token_nums_type": 1,
            "x_active_mask": token_dispatch_input.active_mask,
            "scales": None,
            "quant_mode": 2 if self._quantized else 0,
            "group_ep": self._group_name,
            "ep_world_size": self._config.ep_size,
            "ep_rank_id": self._config.ep_rank,
        }
        if self._need_tp_arguments:
            kwargs.update(
                {
                    "group_tp": self._group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )
        if self._need_expert_scales:
            kwargs["expert_scales"] = routing.topk_weights.to(torch.float32)
        dispatch = (
            torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)
            if self._enable_v2
            else torch_npu.npu_moe_distribute_dispatch(**kwargs)
        )
        (
            expanded,
            dynamic_scale,
            assist_info,
            expert_token_nums,
            ep_recv_counts,
            tp_recv_counts,
            expand_scales,
        ) = dispatch[:7]
        return MoETokenDispatchOutput(
            hidden_states=expanded,
            dynamic_scale=dynamic_scale if self._quantized else None,
            group_list=expert_token_nums.to(torch.int64),
            group_list_type=1,
            combine_metadata=_MC2CombineMetadata(
                topk_ids=routing.topk_ids.to(torch.int32),
                topk_weights=routing.topk_weights,
                ep_recv_counts=ep_recv_counts,
                tp_recv_counts=tp_recv_counts,
                assist_info=assist_info,
                expand_scales=expand_scales,
                active_mask=token_dispatch_input.active_mask,
            ),
        )

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: object,
    ) -> torch.Tensor:
        if not isinstance(combine_metadata, _MC2CombineMetadata):
            raise TypeError("MC2 dispatcher received incompatible metadata")
        kwargs = {
            "expand_x": hidden_states,
            "expert_ids": combine_metadata.topk_ids,
            "expert_scales": combine_metadata.topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self._num_experts,
            "global_bs": 0,
            "x_active_mask": combine_metadata.active_mask,
            "ep_send_counts": combine_metadata.ep_recv_counts,
            "group_ep": self._group_name,
            "ep_world_size": self._config.ep_size,
            "ep_rank_id": self._config.ep_rank,
            "expand_scales": combine_metadata.expand_scales,
            "comm_quant_mode": 0,
        }
        if self._enable_v2:
            kwargs["assist_info_for_combine"] = combine_metadata.assist_info
        else:
            kwargs["expand_idx"] = combine_metadata.assist_info
        if self._need_tp_arguments:
            kwargs.update(
                {
                    "tp_send_counts": combine_metadata.tp_recv_counts,
                    "group_tp": self._group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )
        if self._enable_v2:
            return torch_npu.npu_moe_distribute_combine_v2(**kwargs)
        return torch_npu.npu_moe_distribute_combine(**kwargs)


# Expert computation

_ACL_FORMAT_FRACTAL_NZ = 29


def _select_tensor_shard(
    tensor: torch.Tensor,
    dim: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    if world_size == 1:
        return tensor
    if tensor.shape[dim] % world_size != 0:
        raise ValueError("MoE weight dimension must divide parallel size")
    return torch.tensor_split(tensor, world_size, dim=dim)[rank].contiguous()


def _load_packed_expert_shard(
    state_dict: StateDict,
    name: str,
    config: MoEExpertsConfig,
    tp_dim: int | None,
) -> torch.Tensor:
    tensor = state_dict.get_sharded_tensor(
        name,
        0,
        config.ep_rank,
        config.ep_size,
    )
    if tp_dim is not None:
        tensor = _select_tensor_shard(
            tensor,
            tp_dim,
            config.tp_rank,
            config.tp_size,
        )
    return tensor


class RoutedExperts(nn.Module, ABC):
    """Common interface implemented by all routed-experts backends."""

    @abstractmethod
    def forward(
        self,
        dispatch_output: MoETokenDispatchOutput,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_weights(
        self,
        state_dict: StateDict,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def finish_weight_loading(self) -> None:
        raise NotImplementedError


class UnquantizedRoutedExperts(RoutedExperts):
    """TP-sharded routed experts executed with standard linear operations."""

    def __init__(
        self,
        config: MoEExpertsConfig,
        activation: nn.Module,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if config.intermediate_size % config.tp_size != 0:
            raise ValueError("MoE expert intermediate size must divide tp_size")
        self._config = config
        intermediate_per_rank = config.intermediate_size // config.tp_size
        self.global_num_experts = config.num_experts
        self.num_experts = config.num_local_experts
        self.first_expert_id = config.first_expert_id
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tp_size = config.tp_size
        self.activation = activation
        self._loaded_mask = torch.zeros(
            (config.num_local_experts, 3),
            dtype=torch.bool,
            device="cpu",
        )
        self._packed_loaded: set[str] = set()
        self.w13_weight = nn.Parameter(
            torch.empty(
                config.num_local_experts,
                2 * intermediate_per_rank,
                config.hidden_size,
                dtype=dtype,
                device=device,
            )
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                config.num_local_experts,
                config.hidden_size,
                intermediate_per_rank,
                dtype=dtype,
                device=device,
            )
        )
        nn.init.normal_(self.w13_weight, std=0.02)
        nn.init.normal_(self.w2_weight, std=0.02)

    def forward(
        self,
        dispatch_output: MoETokenDispatchOutput,
    ) -> torch.Tensor:
        hidden_states = dispatch_output.hidden_states
        if hidden_states.shape[0] == 0:
            return hidden_states.new_empty((0, self.hidden_size))
        if dispatch_output.group_list.numel() != self.num_experts:
            raise ValueError("MoE group_list must contain one value per expert")
        gate_up = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[self.w13_weight.transpose(-1, -2)],
            split_item=2,
            group_list_type=dispatch_output.group_list_type,
            group_type=0,
            group_list=dispatch_output.group_list,
            output_dtype=hidden_states.dtype,
        )[0]
        activated = self.activation(gate_up)
        expert_output = torch_npu.npu_grouped_matmul(
            x=[activated],
            weight=[self.w2_weight.transpose(-1, -2)],
            split_item=2,
            group_list_type=dispatch_output.group_list_type,
            group_type=0,
            group_list=dispatch_output.group_list,
            output_dtype=hidden_states.dtype,
        )[0]
        return expert_output

    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        packed_name = name.removesuffix(".weight")
        packed_targets = {
            "w13_weight": self.w13_weight,
            "w2_weight": self.w2_weight,
        }
        packed_target = packed_targets.get(packed_name)
        if packed_target is not None:
            if tensor.shape != packed_target.shape:
                raise ValueError(f"MoE {packed_name} expects {packed_target.shape}, got {tensor.shape}")
            packed_target.data.copy_(tensor.to(packed_target))
            self._packed_loaded.add(packed_name)
            return True

        parts = name.split(".")
        if len(parts) != 3 or parts[2] != "weight":
            return False
        try:
            expert_id = int(parts[0])
        except ValueError:
            return False
        if not self.first_expert_id <= expert_id < (self.first_expert_id + self.num_experts):
            return False
        local_expert_id = expert_id - self.first_expert_id
        projection_group = {
            "w1": "gate",
            "gate_proj": "gate",
            "w3": "up",
            "up_proj": "up",
            "w2": "down",
            "down_proj": "down",
        }.get(parts[1])
        if projection_group is None:
            return False

        if projection_group in ("gate", "up"):
            half = self.w13_weight.shape[1] // 2
            start = 0 if projection_group == "gate" else half
            target = self.w13_weight.data[
                local_expert_id,
                start : start + half,
            ]
        else:
            target = self.w2_weight.data[local_expert_id]
        if tensor.shape != target.shape:
            raise ValueError(f"MoE expert {name} expects {target.shape}, got {tensor.shape}")
        target.copy_(tensor.to(target))
        projection_index = {"gate": 0, "up": 1, "down": 2}[projection_group]
        self._loaded_mask[local_expert_id, projection_index] = True
        return True

    def load_weights(
        self,
        state_dict: StateDict,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        del tp_rank, tp_size
        loaded: set[str] = set()
        for name in state_dict.keys():
            packed_name = name.removesuffix(".weight")
            if packed_name in ("w13_weight", "w2_weight"):
                tp_dim = 1 if packed_name == "w13_weight" else 2
                tensor = _load_packed_expert_shard(
                    state_dict,
                    name,
                    self._config,
                    tp_dim,
                )
            elif len(name.split(".")) == 3:
                parts = name.split(".")
                try:
                    expert_id = int(parts[0])
                except ValueError:
                    continue
                if not self.first_expert_id <= expert_id < (self.first_expert_id + self.num_experts):
                    continue
                shard_dim = 0 if parts[1] in ("w1", "w3", "gate_proj", "up_proj") else 1
                tensor = state_dict.get_sharded_tensor(
                    name,
                    shard_dim,
                    self._config.tp_rank,
                    self._config.tp_size,
                )
            else:
                tensor = state_dict.get_tensor(name)
            if self.load_weight(name, tensor):
                loaded.add(name)
        return loaded

    def finish_weight_loading(self) -> None:
        if self._packed_loaded:
            required_packed = {"w13_weight", "w2_weight"}
            missing = required_packed.difference(self._packed_loaded)
            if missing:
                raise KeyError(f"Packed expert weights are missing: {sorted(missing)}")
            return
        if not bool(self._loaded_mask.all()):
            raise KeyError("Routed expert weights are incomplete")


def _dequant_situ_quant(
    hidden_states: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The fused kernel uses linear_beta=0 to leave the up half unchanged.
    return torch.ops.xllm_ops.dequant_situ_quant(
        hidden_states,
        None,
        None,
        None,
        None,
        None,
        None,
        beta,
        linear_beta if linear_beta is not None else 0.0,
        True,
        "dynamic",
    )


class QuantizedExpertsActivation(nn.Module, ABC):
    """Activation contract for dynamically quantized expert pipelines."""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class FusedQuantizedSituAndMul(QuantizedExpertsActivation):
    """Fused SiTU activation producing quantized output and token scales."""

    def __init__(self, beta: float, linear_beta: float | None) -> None:
        super().__init__()
        _validate_situ_activation(beta, linear_beta)
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _dequant_situ_quant(
            hidden_states,
            self.beta,
            self.linear_beta,
        )


class FusedW4A8RoutedExperts(RoutedExperts):
    """ModelSlim W4A8 routed experts implemented with fused grouped matmul."""

    def __init__(
        self,
        config: MoEExpertsConfig,
        activation: QuantizedExpertsActivation,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if config.intermediate_size % config.tp_size != 0:
            raise ValueError("MoE expert intermediate size must divide tp_size")
        self._config = config
        intermediate_per_rank = config.intermediate_size // config.tp_size
        if intermediate_per_rank % 2 != 0 or config.hidden_size % 2 != 0 or 16 % config.tp_size != 0:
            raise ValueError("Packed W4A8 expert dimensions are invalid")
        self.global_num_experts = config.num_experts
        self.num_experts = config.num_local_experts
        self.first_expert_id = config.first_expert_id
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.tp_size = config.tp_size
        self.activation = activation
        self.output_dtype = dtype
        self._runtime_weights_ready = False
        self._loaded_mask = torch.zeros(
            (config.num_local_experts, 3, 4),
            dtype=torch.bool,
            device="cpu",
        )
        self._packed_loaded: set[str] = set()
        self.w13_weight = nn.Parameter(
            torch.empty(
                config.num_local_experts,
                intermediate_per_rank,
                config.hidden_size,
                dtype=torch.int8,
                device=device,
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                config.num_local_experts,
                config.hidden_size // 2,
                intermediate_per_rank,
                dtype=torch.int8,
                device=device,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "w13_weight_scale",
            torch.empty(
                config.num_local_experts,
                2 * intermediate_per_rank,
                1,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "w13_weight_offset",
            torch.empty_like(self.w13_weight_scale),
        )
        self.register_buffer(
            "w13_scale_bias",
            torch.empty_like(self.w13_weight_scale),
        )
        self.register_buffer(
            "w2_weight_scale",
            torch.empty(
                config.num_local_experts,
                config.hidden_size,
                1,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "w2_weight_offset",
            torch.empty_like(self.w2_weight_scale),
        )
        self.register_buffer(
            "w2_scale_bias",
            torch.empty(
                config.num_local_experts,
                config.hidden_size,
                16 // config.tp_size,
                dtype=torch.float32,
                device=device,
            ),
        )

    def forward(
        self,
        dispatch_output: MoETokenDispatchOutput,
    ) -> torch.Tensor:
        if not self._runtime_weights_ready:
            raise RuntimeError("W4A8 expert weights are not ready")
        hidden_states = dispatch_output.hidden_states
        if hidden_states.shape[0] == 0:
            return torch.empty(
                (0, self.hidden_size),
                dtype=self.output_dtype,
                device=hidden_states.device,
            )
        if dispatch_output.dynamic_scale is None:
            raise ValueError("W4A8 experts require dispatched token scales")

        gate_up = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[self.w13_weight],
            scale=[self.w13_weight_scale.unsqueeze(-2)],
            bias=[self.w13_scale_bias],
            per_token_scale=[dispatch_output.dynamic_scale],
            split_item=2,
            group_list_type=dispatch_output.group_list_type,
            group_type=0,
            group_list=dispatch_output.group_list,
            output_dtype=torch.bfloat16,
        )[0]
        activated, activated_scale = self.activation(gate_up)
        expert_output = torch_npu.npu_grouped_matmul(
            x=[activated],
            weight=[self.w2_weight],
            scale=[self.w2_weight_scale],
            bias=[self.w2_scale_bias],
            per_token_scale=[activated_scale],
            split_item=2,
            group_list_type=dispatch_output.group_list_type,
            group_type=0,
            group_list=dispatch_output.group_list,
            output_dtype=self.output_dtype,
        )[0]
        return expert_output

    @staticmethod
    def _encode_per_channel_scale(scale: torch.Tensor) -> torch.Tensor:
        transposed = scale.transpose(1, 2).contiguous()
        encoded = transposed.cpu().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        return encoded.to(device=scale.device)

    def _process_quantized_weights(self) -> None:
        if self._runtime_weights_ready:
            return
        if self.w13_weight.shape[-2] % 4 != 0:
            raise ValueError("W4A8 w13 packed dimension must divide 4")
        if self.w2_weight.shape[-2] % 4 != 0:
            raise ValueError("W4A8 w2 packed dimension must divide 4")

        self.w13_weight.data = self.w13_weight.data.transpose(1, 2).contiguous()
        self.w2_weight.data = self.w2_weight.data.transpose(1, 2).contiguous()
        if self.w13_weight.device.type in ("npu", "privateuseone"):
            self.w13_weight.data = torch_npu.npu_format_cast(
                self.w13_weight.data,
                _ACL_FORMAT_FRACTAL_NZ,
            )
            self.w2_weight.data = torch_npu.npu_format_cast(
                self.w2_weight.data,
                _ACL_FORMAT_FRACTAL_NZ,
            )
        self.w13_weight.data = self.w13_weight.data.view(torch.int32).contiguous()
        self.w2_weight.data = self.w2_weight.data.view(torch.int32).contiguous()
        self.w13_weight_scale.data = self._encode_per_channel_scale(self.w13_weight_scale.data).squeeze(1)
        self.w2_weight_scale.data = self._encode_per_channel_scale(self.w2_weight_scale.data)
        self.w13_scale_bias.data = self.w13_scale_bias.data.transpose(1, 2).contiguous().sum(dim=1)
        self.w2_scale_bias.data = self.w2_scale_bias.data.transpose(1, 2).contiguous().sum(dim=1)
        self._runtime_weights_ready = True

    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        packed_targets = {
            "w13_weight": self.w13_weight,
            "w2_weight": self.w2_weight,
            "w13_weight_scale": self.w13_weight_scale,
            "w13_weight_offset": self.w13_weight_offset,
            "w13_scale_bias": self.w13_scale_bias,
            "w2_weight_scale": self.w2_weight_scale,
            "w2_weight_offset": self.w2_weight_offset,
            "w2_scale_bias": self.w2_scale_bias,
        }
        packed_name = name.removesuffix(".weight")
        packed_target = packed_targets.get(packed_name)
        if packed_target is not None:
            if tensor.shape != packed_target.shape:
                raise ValueError(f"MoE {packed_name} expects {packed_target.shape}, got {tensor.shape}")
            packed_target.data.copy_(tensor.to(packed_target))
            self._packed_loaded.add(packed_name)
            return True

        parts = name.split(".")
        if len(parts) != 3:
            return False
        try:
            expert_id = int(parts[0])
        except ValueError:
            return False
        if not self.first_expert_id <= expert_id < (self.first_expert_id + self.num_experts):
            return False
        local_expert_id = expert_id - self.first_expert_id
        projection_group = {
            "w1": "gate",
            "gate_proj": "gate",
            "w3": "up",
            "up_proj": "up",
            "w2": "down",
            "down_proj": "down",
        }.get(parts[1])
        if projection_group is None:
            return False
        suffix = parts[2]

        if projection_group in ("gate", "up"):
            target_tensor = self._w13_target(suffix)
            if target_tensor is None:
                return False
            half = target_tensor.shape[1] // 2
            start = 0 if projection_group == "gate" else half
            target = target_tensor.data[
                local_expert_id,
                start : start + half,
            ]
        else:
            target_tensor = self._w2_target(suffix)
            if target_tensor is None:
                return False
            target = target_tensor.data[local_expert_id]
        if tensor.shape != target.shape:
            raise ValueError(f"MoE expert {name} expects {target.shape}, got {tensor.shape}")
        target.copy_(tensor.to(target))
        projection_index = {"gate": 0, "up": 1, "down": 2}[projection_group]
        suffix_index = {
            "weight": 0,
            "weight_scale": 1,
            "weight_offset": 2,
            "scale_bias": 3,
        }[suffix]
        self._loaded_mask[
            local_expert_id,
            projection_index,
            suffix_index,
        ] = True
        return True

    def _w13_target(self, suffix: str) -> torch.Tensor | None:
        return {
            "weight": self.w13_weight,
            "weight_scale": self.w13_weight_scale,
            "weight_offset": self.w13_weight_offset,
            "scale_bias": self.w13_scale_bias,
        }.get(suffix)

    def _w2_target(self, suffix: str) -> torch.Tensor | None:
        return {
            "weight": self.w2_weight,
            "weight_scale": self.w2_weight_scale,
            "weight_offset": self.w2_weight_offset,
            "scale_bias": self.w2_scale_bias,
        }.get(suffix)

    def load_weights(
        self,
        state_dict: StateDict,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        del tp_rank, tp_size
        loaded: set[str] = set()
        for name in state_dict.keys():
            packed_name = name.removesuffix(".weight")
            if packed_name in ("w13_weight", "w2_weight"):
                tensor = _load_packed_expert_shard(
                    state_dict,
                    name,
                    self._config,
                    1 if packed_name == "w13_weight" else 2,
                )
            elif packed_name in (
                "w13_weight_scale",
                "w13_weight_offset",
                "w13_scale_bias",
            ):
                tensor = _load_packed_expert_shard(
                    state_dict,
                    name,
                    self._config,
                    1,
                )
            elif packed_name == "w2_scale_bias":
                tensor = _load_packed_expert_shard(
                    state_dict,
                    name,
                    self._config,
                    2,
                )
            elif packed_name in ("w2_weight_scale", "w2_weight_offset"):
                tensor = _load_packed_expert_shard(
                    state_dict,
                    name,
                    self._config,
                    None,
                )
            elif len(name.split(".")) == 3:
                parts = name.split(".")
                try:
                    expert_id = int(parts[0])
                except ValueError:
                    continue
                if not self.first_expert_id <= expert_id < (self.first_expert_id + self.num_experts):
                    continue
                projection = parts[1]
                suffix = parts[2]
                if projection in ("w1", "w3", "gate_proj", "up_proj"):
                    tensor = state_dict.get_sharded_tensor(
                        name,
                        0,
                        self._config.tp_rank,
                        self._config.tp_size,
                    )
                elif suffix in ("weight", "scale_bias"):
                    tensor = state_dict.get_sharded_tensor(
                        name,
                        1,
                        self._config.tp_rank,
                        self._config.tp_size,
                    )
                else:
                    tensor = state_dict.get_tensor(name)
            else:
                continue
            if self.load_weight(name, tensor):
                loaded.add(name)
        return loaded

    def finish_weight_loading(self) -> None:
        if self._packed_loaded:
            required_packed = {
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w13_weight_offset",
                "w13_scale_bias",
                "w2_weight_scale",
                "w2_weight_offset",
                "w2_scale_bias",
            }
            missing = required_packed.difference(self._packed_loaded)
            if missing:
                raise KeyError(f"Packed expert weights are missing: {sorted(missing)}")
        elif not bool(self._loaded_mask.all()):
            raise KeyError("Routed expert weights are incomplete")

        if self.w13_weight.device.type in ("npu", "privateuseone"):
            self._process_quantized_weights()


# Communication policies


class MoECommMethod:
    """Compose prepare/finalize and token dispatch into one MoE pipeline."""

    def __init__(
        self,
        token_dispatcher: MoETokenDispatcher,
        prepare_finalize: PrepareAndFinalize,
    ) -> None:
        self.token_dispatcher = token_dispatcher
        self.prepare_finalize = prepare_finalize

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> MoEPrepareOutput:
        return self.prepare_finalize.prepare(hidden_states, router_logits)

    def fused_experts(
        self,
        experts: RoutedExperts,
        prepare_output: MoEPrepareOutput,
        routing: MoERoutingResult,
    ) -> MoEFusedExpertsResult:
        dispatch_output = self.token_dispatcher.token_dispatch(
            MoETokenDispatchInput(
                hidden_states=prepare_output.hidden_states,
                routing=routing,
                active_mask=prepare_output.active_mask,
            )
        )

        expert_output = experts(dispatch_output)
        if not isinstance(expert_output, torch.Tensor):
            raise TypeError("Routed experts must return a tensor")
        routed_out = self.token_dispatcher.token_combine(
            expert_output,
            dispatch_output.combine_metadata,
        )
        return MoEFusedExpertsResult(routed_out=routed_out)

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        return self.prepare_finalize.finalize(
            hidden_states,
            reduce_results,
            padded_hidden_states_shape,
        )


class AllGatherCommMethod(MoECommMethod):
    """EP all-gather/reduce-scatter with local expert execution."""

    def __init__(
        self,
        config: MoEParallelConfig,
        num_experts: int,
        top_k: int,
        quantized: bool,
        device: torch.device,
    ) -> None:
        num_local_experts = num_experts // config.ep_size
        first_expert_id = config.ep_rank * num_local_experts
        if quantized or device.type in ("npu", "privateuseone"):
            dispatcher: MoETokenDispatcher = FusedAllGatherTokenDispatcher(
                num_experts=num_experts,
                top_k=top_k,
                quantized=quantized,
                first_expert_id=first_expert_id,
                num_local_experts=num_local_experts,
            )
        else:
            dispatcher = NativeTokenDispatcher(
                num_experts=num_experts,
                first_expert_id=first_expert_id,
                num_local_experts=num_local_experts,
            )
        super().__init__(dispatcher, AllGatherPrepareAndFinalize(config))


class AllToAllCommMethod(MoECommMethod):
    """Explicit EP all-to-all dispatch/combine with MoE-TP reduction."""

    def __init__(
        self,
        config: MoEParallelConfig,
        num_experts: int,
        quantized: bool,
        vllm_prefill: bool = False,
    ) -> None:
        super().__init__(
            (
                VllmAllToAllTokenDispatcher(config, num_experts, quantized)
                if vllm_prefill
                else AllToAllTokenDispatcher(config, num_experts, quantized)
            ),
            AllToAllPrepareAndFinalize(config),
        )


class MC2CommMethod(MoECommMethod):
    """Ascend MC2 dispatch/combine with MoE-TP reduction."""

    def __init__(
        self,
        config: MoEParallelConfig,
        num_experts: int,
        quantized: bool,
        device: torch.device,
    ) -> None:
        super().__init__(
            MC2TokenDispatcher(config, num_experts, quantized, device),
            MC2PrepareAndFinalize(config),
        )


class AdaptiveMoECommMethod(MoECommMethod):
    """Use All2All eagerly and static AllGather for decode graph execution."""

    def __init__(
        self,
        config: MoEParallelConfig,
        num_experts: int,
        top_k: int,
        quantized: bool,
        device: torch.device,
        all_gather_config: MoEParallelConfig | None = None,
        vllm_prefill: bool = False,
    ) -> None:
        self._config = config
        self._all_gather = AllGatherCommMethod(
            all_gather_config or config,
            num_experts,
            top_k,
            quantized,
            device,
        )
        self._all_to_all = (
            AllToAllCommMethod(config, num_experts, quantized, vllm_prefill=vllm_prefill)
            if config.ep_size > 1
            else None
        )
        self._active: MoECommMethod | None = None

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> MoEPrepareOutput:
        # Dynamic all-to-all split metadata cannot be captured in an ACL graph.
        if _in_acl_graph_capture():
            self._active = self._all_gather
        elif self._all_to_all is not None:
            self._active = self._all_to_all
        else:
            self._active = self._all_gather
        return self._active.prepare(hidden_states, router_logits)

    def fused_experts(
        self,
        experts: RoutedExperts,
        prepare_output: MoEPrepareOutput,
        routing: MoERoutingResult,
    ) -> MoEFusedExpertsResult:
        if self._active is None:
            raise RuntimeError("MoE communication prepare must run first")
        return self._active.fused_experts(experts, prepare_output, routing)

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        if self._active is None:
            raise RuntimeError("MoE communication prepare must run first")
        output = self._active.finalize(
            hidden_states,
            reduce_results,
            padded_hidden_states_shape,
        )
        self._active = None
        return output


def build_moe_comm_method(
    config: MoEParallelConfig,
    num_experts: int,
    top_k: int,
    quantized: bool,
    device: torch.device,
    all_gather_config: MoEParallelConfig | None = None,
    vllm_prefill: bool = False,
) -> MoECommMethod:
    """Build the configured reusable MoE communication method."""
    comm_type = config.comm_type
    # Honor the explicit policy; prepare/finalize handles the global EP layout.
    if config.ep_size == 1 or comm_type == MoECommType.ALL_GATHER:
        return AllGatherCommMethod(
            config,
            num_experts,
            top_k,
            quantized,
            device,
        )
    if comm_type == MoECommType.ALL_TO_ALL:
        # Explicit all-to-all is used for eager prefill. During ACL graph
        # capture AdaptiveMoECommMethod falls back to static all-gather because
        # all-to-all split metadata is materialized on the host.
        return AdaptiveMoECommMethod(
            config,
            num_experts,
            top_k,
            quantized,
            device,
            all_gather_config=all_gather_config,
            vllm_prefill=vllm_prefill,
        )
    if comm_type == MoECommType.MC2:
        return MC2CommMethod(config, num_experts, quantized, device)
    if comm_type == MoECommType.AUTO:
        return AdaptiveMoECommMethod(
            config,
            num_experts,
            top_k,
            quantized,
            device,
        )
    raise ValueError(f"Unsupported MoE communication type: {comm_type}")


# MoE execution

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


class MoERunner:
    """Coordinate routing, communication, experts, and shared branches."""

    def __init__(
        self,
        router: MoERouter,
        comm_method: MoECommMethod,
    ) -> None:
        self.router = router
        self.comm_method = comm_method

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        experts: RoutedExperts,
        correction_bias: torch.Tensor | None = None,
        shared_experts: torch.nn.Module | None = None,
        routed_input_transform: TensorTransform | None = None,
        routed_output_transform: TensorTransform | None = None,
    ) -> torch.Tensor:
        shared_input = hidden_states
        routed_input = hidden_states
        if routed_input_transform is not None:
            routed_input = self._unwrap_tensor(routed_input_transform(routed_input))

        prepare_output = self.comm_method.prepare(routed_input, router_logits)
        routing = self.router.select_experts(
            hidden_states=prepare_output.hidden_states,
            router_logits=prepare_output.router_logits,
            correction_bias=correction_bias,
        )

        fused_result = self.comm_method.fused_experts(
            experts=experts,
            prepare_output=prepare_output,
            routing=routing,
        )
        routed_output = self.comm_method.finalize(
            hidden_states=fused_result.routed_out,
            reduce_results=self._reduce_routed_results(),
            padded_hidden_states_shape=(prepare_output.padded_hidden_states_shape),
        )
        if routed_output_transform is not None:
            routed_output = self._unwrap_tensor(routed_output_transform(routed_output))

        shared_output = None
        if shared_experts is not None:
            shared_output = self._unwrap_tensor(shared_experts(shared_input))
            shared_output = self._finalize_shared_expert_output(shared_output)
        output = routed_output if shared_output is None else routed_output + shared_output

        return self._finalize_output(output)

    def _reduce_routed_results(self) -> bool:
        return True

    def _finalize_shared_expert_output(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states

    def _finalize_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    @staticmethod
    def _unwrap_tensor(
        result: torch.Tensor | tuple[torch.Tensor, object],
    ) -> torch.Tensor:
        if isinstance(result, tuple):
            return result[0]
        return result


# Kimi K3 model integration


def _global_prefill_step(dp_size: int) -> bool:
    """Whether the global DP step contains prefill work.

    EP collectives span all DP groups for Kimi's EP=world topology.  A local
    ``metadata.is_prefill`` value is therefore insufficient: an idle/decode
    DP peer must use the same routed-MoE protocol as the active prefill peers.
    The C++ input builder publishes the per-DP batch-type vector in
    ``dp_is_decode`` on every rank.
    """
    # Decode graph warmup/capture has no request metadata. All ranks use
    # the same static decode protocol, including the metadata-free warmup.
    if dp_size <= 1 or _in_acl_graph_capture():
        return False
    metadata = get_forward_context().metadata
    if metadata is None:
        raise RuntimeError("Kimi MoE requires forward metadata for global DP phase selection")
    raw_values = getattr(metadata, "dp_is_decode", None)
    if raw_values is None:
        raise RuntimeError("Kimi MoE requires dp_is_decode metadata for global DP phase selection")
    try:
        values = tuple(int(value) for value in raw_values)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Kimi MoE received invalid dp_is_decode metadata") from exc
    if len(values) != dp_size:
        raise RuntimeError(f"Kimi MoE requires {dp_size} DP phase values, received {len(values)}")
    if any(value not in (0, 1) for value in values):
        raise RuntimeError(f"Kimi MoE received invalid dp_is_decode values: {values!r}")
    # xLLM uses 0 for prefill (including an empty DP rank inherited from a
    # prefill batch) and 1 for decode.  If any peer has prefill work, every
    # rank in the global EP communicator must enter the prefill protocol.
    return len(values) > 1 and any(value == 0 for value in values)


class MoE(nn.Module):
    """Reusable model-level MoE with common routing and weight ownership."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        router_config: MoERouterConfig,
        experts: RoutedExperts,
        comm_method: MoECommMethod,
        dtype: torch.dtype,
        device: torch.device,
        shared_experts: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if router_config.num_experts != num_experts:
            raise ValueError("MoE router and expert counts must match")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.gate = nn.Linear(
            hidden_size,
            num_experts,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.gate.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts, dtype=dtype, device=device))
        self.experts = experts
        self.shared_experts = shared_experts
        self._router = GroupedTopKRouter(router_config)
        self._runner = MoERunner(self._router, comm_method)
        self._loaded_components: set[str] = set()

    def _routed_input_transform(self) -> TensorTransform | None:
        return None

    def _routed_output_transform(self) -> TensorTransform | None:
        return None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        output = self._runner.forward(
            hidden_states=hidden_states,
            router_logits=router_logits,
            correction_bias=self.gate.e_score_correction_bias,
            experts=self.experts,
            shared_experts=self.shared_experts,
            routed_input_transform=self._routed_input_transform(),
            routed_output_transform=self._routed_output_transform(),
        )
        return output.reshape(original_shape)

    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        if name == "gate.weight":
            target = self.gate.weight
            # 减专家验证：权重含 num_experts 个专家，config 减到更少时截断前 N 行。
            if tensor.shape[0] > target.shape[0]:
                tensor = tensor[: target.shape[0]]
            if tensor.shape != target.shape:
                raise ValueError(f"MoE gate expects {target.shape}, got {tensor.shape}")
            target.data.copy_(tensor.to(target))
            self._loaded_components.add(name)
            return True
        if name in ("e_score_correction_bias", "gate.e_score_correction_bias"):
            target = self.gate.e_score_correction_bias
            if tensor.shape[0] > target.shape[0]:
                tensor = tensor[: target.shape[0]]
            if tensor.shape != target.shape:
                raise ValueError(f"MoE correction bias expects {target.shape}, got {tensor.shape}")
            target.data.copy_(tensor.to(target))
            self._loaded_components.add("gate.e_score_correction_bias")
            return True
        return False

    def _shared_experts_tp_coordinates(self, tp_rank: int, tp_size: int) -> tuple[int, int]:
        return tp_rank, tp_size

    def _experts_tp_coordinates(self, tp_rank: int, tp_size: int) -> tuple[int, int]:
        return tp_rank, tp_size

    def load_weights(
        self,
        state_dict: StateDict,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        loaded: set[str] = set()
        for name in ("gate.weight", "gate.e_score_correction_bias"):
            if state_dict.has(name) and self.load_weight(
                name,
                state_dict.get_tensor(name),
            ):
                loaded.add(name)

        shared_state_dict = state_dict.get_dict_with_prefix("shared_experts.")
        if self.shared_experts is not None and shared_state_dict.size() > 0:
            shared_tp_rank, shared_tp_size = self._shared_experts_tp_coordinates(tp_rank, tp_size)
            child_loaded = self.shared_experts.load_weights(
                shared_state_dict,
                shared_tp_rank,
                shared_tp_size,
            )
            loaded.update(f"shared_experts.{name}" for name in child_loaded)

        experts_state_dict = state_dict.get_dict_with_prefix("experts.")
        if experts_state_dict.size() > 0:
            experts_tp_rank, experts_tp_size = self._experts_tp_coordinates(tp_rank, tp_size)
            child_loaded = self.experts.load_weights(
                experts_state_dict,
                experts_tp_rank,
                experts_tp_size,
            )
            loaded.update(f"experts.{name}" for name in child_loaded)
        return loaded

    def _required_weight_names(self) -> set[str]:
        return {
            "gate.weight",
            "gate.e_score_correction_bias",
        }

    def _weight_modules_to_finalize(
        self,
    ) -> tuple[nn.Module | None, ...]:
        return self.shared_experts, self.experts

    def finish_weight_loading(self) -> None:
        missing = self._required_weight_names().difference(self._loaded_components)
        if missing:
            raise KeyError(f"MoE weights are missing: {sorted(missing)}")
        for module in self._weight_modules_to_finalize():
            if module is not None and hasattr(module, "finish_weight_loading"):
                module.finish_weight_loading()


class KimiK3MoERunner(MoERunner):
    """Kimi policy for latent routed and shared expert TP reductions."""

    def __init__(
        self,
        router: GroupedTopKRouter,
        comm_method: MoECommMethod,
        tp_size: int,
        tp_group_name: str,
    ) -> None:
        super().__init__(router, comm_method)
        self._tp_size = tp_size
        self._tp_group_name = tp_group_name

    def _finalize_shared_expert_output(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self._tp_size > 1:
            if self._tp_group_name == "tp":
                distributed.all_reduce_(hidden_states)
            else:
                distributed.all_reduce_(
                    hidden_states,
                    group_name=self._tp_group_name,
                )
        return hidden_states


class KimiK3MoE(MoE):
    """Kimi latent MoE extending the reusable model-level implementation."""

    def __init__(
        self,
        config: Any,
        dtype: torch.dtype,
        device: torch.device,
        tp_size: int,
        tp_rank: int,
        routed_expert_down_proj: nn.Module,
        routed_expert_up_proj: nn.Module,
        shared_experts: nn.Module | None = None,
        quantized: bool = False,
    ) -> None:
        num_experts = int(config.num_experts)
        top_k = int(config.num_experts_per_token)
        hidden_size = int(config.hidden_size)
        routed_hidden_size = int(config.routed_expert_hidden_size)
        tp_group_name = "moe_tp"
        # The model receives these coordinates from the native ParallelArgs,
        # as other Python models do. Communication uses the mainline groups.
        input_tp_size = tp_size
        input_tp_rank = tp_rank
        ep_size = int(getattr(config, "ep_size", 1))
        dp_size = int(getattr(config, "dp_size", 1))
        if ep_size == 1:
            moe_tp_size = tp_size
            moe_tp_rank = tp_rank
            ep_rank = 0
            dp_rank = int(getattr(config, "rank", tp_rank)) // tp_size
            tp_group_name = "tp"
        else:
            world_size = int(getattr(config, "world_size", tp_size * ep_size))
            global_rank = int(getattr(config, "rank", tp_rank))
            if world_size % ep_size != 0:
                raise ValueError("Kimi K3 world size must divide MoE EP size")
            moe_tp_size = world_size // ep_size
            moe_tp_rank = global_rank % moe_tp_size
            ep_rank = global_rank // moe_tp_size
            dp_rank = global_rank // (world_size // dp_size)
        parallel_config = MoEParallelConfig(
            tp_size=moe_tp_size,
            tp_rank=moe_tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            comm_type=MoECommType.ALL_TO_ALL,
            mc2_tokens_capacity=int(getattr(config, "mc2_tokens_capacity", 512)),
            tp_group_name=tp_group_name,
            input_tp_size=input_tp_size,
            input_tp_rank=input_tp_rank,
        )
        # FlashComm1 gives each attention-TP rank a distinct token shard.  In
        # the explicit all-to-all path those rows are already the dispatch
        # input; partitioning them once more would reduce 508 rows to 32.
        # Use a local-token MoE topology for routed experts and let the
        # sequence-parallel wrapper handle the shared expert branch below.
        self.sequence_parallel_routed = (
            parallel_config.comm_type == MoECommType.ALL_TO_ALL
            and input_tp_size > 1
            and ep_size > 1
            and moe_tp_size == 1
        )
        comm_parallel_config = (
            replace(
                parallel_config,
                tp_size=1,
                tp_rank=0,
                input_tp_size=1,
                input_tp_rank=0,
            )
            if self.sequence_parallel_routed
            else parallel_config
        )
        experts_config = MoEExpertsConfig(
            num_experts=num_experts,
            hidden_size=routed_hidden_size,
            intermediate_size=int(config.moe_intermediate_size),
            tp_size=moe_tp_size,
            tp_rank=moe_tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
        )
        configured_situ_beta = getattr(config, "activation_situ_beta", None)
        situ_beta = 1.0 if configured_situ_beta is None else float(configured_situ_beta)
        situ_linear_beta = getattr(config, "activation_situ_linear_beta", None)
        if quantized:
            experts: RoutedExperts = FusedW4A8RoutedExperts(
                config=experts_config,
                activation=FusedQuantizedSituAndMul(
                    beta=situ_beta,
                    linear_beta=situ_linear_beta,
                ),
                dtype=dtype,
                device=device,
            )
        else:
            experts = UnquantizedRoutedExperts(
                config=experts_config,
                activation=SituAndMul(
                    beta=situ_beta,
                    linear_beta=situ_linear_beta,
                ),
                dtype=dtype,
                device=device,
            )

        router_config = MoERouterConfig(
            num_experts=num_experts,
            top_k=top_k,
            scoring_func=str(config.moe_router_activation_func),
            renormalize=bool(config.moe_renormalize),
            routed_scaling_factor=float(config.routed_scaling_factor),
            use_grouped_topk=bool(getattr(config, "use_grouped_topk", True)),
            num_expert_group=int(getattr(config, "num_expert_group", 1)),
            topk_group=int(getattr(config, "topk_group", 1)),
        )
        # Static AllGather is the validated CANN 9.0.0 decode path.
        decode_comm_method = AllGatherCommMethod(parallel_config, num_experts, top_k, quantized, device)
        if self.sequence_parallel_routed:
            prefill_comm_method = build_moe_comm_method(
                config=comm_parallel_config,
                num_experts=num_experts,
                top_k=top_k,
                quantized=quantized,
                device=device,
                all_gather_config=parallel_config,
                vllm_prefill=True,
            )
        else:
            prefill_comm_method = decode_comm_method
        super().__init__(
            hidden_size=hidden_size,
            num_experts=num_experts,
            router_config=router_config,
            experts=experts,
            comm_method=decode_comm_method,
            dtype=dtype,
            device=device,
            shared_experts=shared_experts,
        )

        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.dp_size = dp_size
        self.moe_tp_size = moe_tp_size
        self.moe_tp_rank = moe_tp_rank
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.quantized = quantized
        self.routed_expert_down_proj = routed_expert_down_proj
        self.routed_expert_up_proj = routed_expert_up_proj
        self.routed_expert_norm = (
            nn.RMSNorm(
                routed_hidden_size,
                eps=float(config.rms_norm_eps),
                dtype=dtype,
                device=device,
            )
            if bool(getattr(config, "latent_moe_use_norm", False))
            else None
        )
        self._runner = KimiK3MoERunner(
            self._router,
            decode_comm_method,
            input_tp_size,
            parallel_config.input_tp_group_name,
        )
        self._prefill_runner = (
            KimiK3MoERunner(
                self._router,
                prefill_comm_method,
                1,
                parallel_config.input_tp_group_name,
            )
            if self.sequence_parallel_routed
            else self._runner
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_parallel_tokens: int | None = None,
    ) -> torch.Tensor:
        # EP protocol selection follows the global DP step, regardless of
        # whether this rank has a local sequence-parallel token shard.
        # Otherwise prefill peers use All2All while idle/decode peers use
        # AllGather on the same EP communicator.
        local_prefill = _in_prefill()
        use_prefill_path = self.sequence_parallel_routed and (local_prefill or _global_prefill_step(self.dp_size))

        if not use_prefill_path:
            return super().forward(hidden_states)

        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routed_output = self._prefill_runner.forward(
            hidden_states=hidden_states,
            router_logits=router_logits,
            correction_bias=self.gate.e_score_correction_bias,
            experts=self.experts,
            shared_experts=None,
            routed_input_transform=self._routed_input_transform(),
            routed_output_transform=self._routed_output_transform(),
        )

        # The shared MLP is TP-sharded.  Its token layout depends on the
        # *local* forward type, even when the routed EP collective is chosen
        # from the global DP state above. A local prefill rank with explicit
        # FlashComm1 token metadata needs the sequence-parallel
        # all-gather/reduce-scatter pair.  A local decode/empty rank is still
        # in the ordinary replicated-token layout; gathering it as if it were
        # a sequence shard would divide the shared result by TP and make it
        # impossible to add to the routed output (the 4-vs-1/2 row mismatch
        # seen under mixed prefill+decode load).
        if self.shared_experts is not None:
            if local_prefill and sequence_parallel_tokens is not None:
                full_input = distributed.all_gather(
                    hidden_states,
                    dim=0,
                    world_size=self.tp_size,
                    group_name="tp",
                )[:sequence_parallel_tokens]
                shared_output = self._runner._unwrap_tensor(self.shared_experts(full_input))
                if self.tp_size > 1:
                    # This is vLLM's maybe_pad_and_reduce semantics: sum the
                    # TP partials while returning this rank's sequence shard.
                    pad_size = (-shared_output.shape[0]) % self.tp_size
                    if pad_size > 0:
                        shared_output = F.pad(shared_output, (0, 0, 0, pad_size))
                    shared_output = distributed.reduce_scatter(
                        shared_output,
                        world_size=self.tp_size,
                        group_name="tp",
                    )
                    shared_output = shared_output[: hidden_states.shape[0]]
            else:
                # Ranks without a local sequence shard still join the global
                # prefill routed protocol. Their token rows remain replicated
                # across TP, so use the ordinary all-reduce layout here.
                shared_output = self._runner._unwrap_tensor(self.shared_experts(hidden_states))
                if self.tp_size > 1:
                    distributed.all_reduce_(shared_output, group_name="tp")
            routed_output = routed_output + shared_output
        return routed_output.reshape(original_shape)

    def _routed_input_transform(self) -> TensorTransform:
        return self.routed_expert_down_proj

    def _routed_output_transform(self) -> TensorTransform:
        return self._transform_routed_output

    def _transform_routed_output(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.routed_expert_norm is not None:
            hidden_states = self.routed_expert_norm(hidden_states)
        result = self.routed_expert_up_proj(hidden_states)
        if isinstance(result, tuple):
            return result[0]
        return result

    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        if super().load_weight(name, tensor):
            return True
        for prefix, module in (
            ("routed_expert_down_proj.", self.routed_expert_down_proj),
            ("routed_expert_up_proj.", self.routed_expert_up_proj),
        ):
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if hasattr(module, "load_weight"):
                loaded = bool(module.load_weight(suffix, tensor))
                if loaded:
                    self._loaded_components.add(name)
                return loaded
            if suffix != "weight":
                return False
            target = module.weight
            if tensor.shape != target.shape:
                raise ValueError(f"Kimi K3 weight {name} expects {target.shape}, got {tensor.shape}")
            target.data.copy_(tensor.to(target))
            self._loaded_components.add(name)
            return True
        if name == "routed_expert_norm.weight" and self.routed_expert_norm is not None:
            target = self.routed_expert_norm.weight
            if tensor.shape != target.shape:
                raise ValueError(f"Kimi K3 weight {name} expects {target.shape}, got {tensor.shape}")
            target.data.copy_(tensor.to(target))
            self._loaded_components.add(name)
            return True
        return False

    def _shared_experts_tp_coordinates(self, tp_rank: int, tp_size: int) -> tuple[int, int]:
        del tp_rank, tp_size
        return self.tp_rank, self.tp_size

    def _experts_tp_coordinates(self, tp_rank: int, tp_size: int) -> tuple[int, int]:
        del tp_rank, tp_size
        return self.moe_tp_rank, self.moe_tp_size

    def load_weights(
        self,
        state_dict: StateDict,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        loaded = super().load_weights(state_dict, tp_rank, tp_size)
        direct_names = [
            "routed_expert_down_proj.weight",
            "routed_expert_up_proj.weight",
        ]
        if self.quantized:
            for projection in (
                "routed_expert_down_proj",
                "routed_expert_up_proj",
            ):
                direct_names.extend(f"{projection}.{suffix}" for suffix in ("weight_scale", "weight_offset"))
        if self.routed_expert_norm is not None:
            direct_names.append("routed_expert_norm.weight")
        for name in direct_names:
            if state_dict.has(name) and self.load_weight(
                name,
                state_dict.get_tensor(name),
            ):
                loaded.add(name)
        return loaded

    def _required_weight_names(self) -> set[str]:
        required = super()._required_weight_names()
        required.update(
            {
                "routed_expert_down_proj.weight",
                "routed_expert_up_proj.weight",
            }
        )
        if self.quantized:
            for projection in (
                "routed_expert_down_proj",
                "routed_expert_up_proj",
            ):
                required.update(f"{projection}.{suffix}" for suffix in ("weight_scale", "weight_offset"))
        if self.routed_expert_norm is not None:
            required.add("routed_expert_norm.weight")
        return required

    def _weight_modules_to_finalize(
        self,
    ) -> tuple[nn.Module | None, ...]:
        return (
            self.routed_expert_down_proj,
            self.routed_expert_up_proj,
            *super()._weight_modules_to_finalize(),
        )


__all__ = [
    "AdaptiveMoECommMethod",
    "AllGatherCommMethod",
    "AllGatherPrepareAndFinalize",
    "AllToAllCommMethod",
    "AllToAllPrepareAndFinalize",
    "AllToAllTokenDispatcher",
    "FusedAllGatherTokenDispatcher",
    "FusedQuantizedSituAndMul",
    "FusedW4A8RoutedExperts",
    "GroupedTopKRouter",
    "KimiK3MoE",
    "KimiK3MoERunner",
    "MC2CommMethod",
    "MC2ExecutionContext",
    "MC2PrepareAndFinalize",
    "MC2TokenDispatcher",
    "MoE",
    "MoECommType",
    "MoECommMethod",
    "MoEExpertsConfig",
    "MoEFusedExpertsResult",
    "MoEParallelConfig",
    "MoEPrepareOutput",
    "MoERouter",
    "MoERouterConfig",
    "MoERoutingResult",
    "MoERunner",
    "MoETokenDispatchInput",
    "MoETokenDispatchOutput",
    "MoETokenDispatcher",
    "NativeTokenDispatcher",
    "PrepareAndFinalize",
    "QuantizedExpertsActivation",
    "RoutedExperts",
    "SituAndMul",
    "UnquantizedRoutedExperts",
    "build_moe_comm_method",
]
