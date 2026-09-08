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

"""Preparation and finalization stages for MoE communication methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

import torch
import torch.nn.functional as F

from xllm.python import ops
from xllm.python.layers.moe.types import MoEParallelConfig, MoEPrepareOutput
from xllm.python.model_executor.forward_context import get_forward_context


def _in_acl_graph_capture() -> bool:
    try:
        return get_forward_context().acl_graph is not None
    except RuntimeError:
        return False


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


class TensorParallelPrepareAndFinalize(PrepareAndFinalize):
    """Identity preparation and optional TP reduction for replicated tokens."""

    def __init__(
        self,
        tp_size: int,
        tp_group_name: str = "tp",
    ) -> None:
        self._tp_size = tp_size
        self._tp_group_name = tp_group_name

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> MoEPrepareOutput:
        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        del padded_hidden_states_shape
        if reduce_results and self._tp_size > 1:
            if self._tp_group_name == "tp":
                ops.all_reduce_(hidden_states)
            else:
                ops.all_reduce_(hidden_states, group_name=self._tp_group_name)
        return hidden_states


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
        return ops.all_gather(
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
            token_counts = tuple(int(count) for count in getattr(metadata, "dp_token_counts", ()))
        except (RuntimeError, TypeError, ValueError):
            return None
        if len(token_counts) != self._config.dp_size:
            return None
        if not 0 <= self._config.dp_rank < len(token_counts):
            return None
        if any(count < 0 for count in token_counts):
            return None
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
                ops.all_reduce_(hidden_states)
            else:
                ops.all_reduce_(
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
        hidden_states = ops.all_gather(
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

    def _use_prefill_fast_path(self) -> bool:
        """Whether this step may use the prefill-only global-EP fast path.

        Decode must keep the commit-c26 graph-safe behavior. The optimized
        path is enabled only for explicit prefill/chunked-prefill steps and
        falls back to the legacy path whenever phase metadata is unavailable.
        """
        if _in_acl_graph_capture():
            return False
        try:
            metadata = get_forward_context().metadata
        except (RuntimeError, AttributeError):
            return False
        return bool(metadata is not None and (metadata.is_prefill or metadata.is_chunked_prefill))

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> MoEPrepareOutput:
        self._num_tokens = hidden_states.shape[0]
        if self._config.ep_size == 1:
            return MoEPrepareOutput(hidden_states, router_logits)

        if self._config.partitions_replicated_input:
            if not self._use_prefill_fast_path():
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
                hidden_states=ops.all_gather(
                    prepared.hidden_states,
                    dim=0,
                    world_size=self._config.ep_size,
                    group_name=self._config.ep_group_name,
                ),
                router_logits=ops.all_gather(
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
        hidden_states = ops.all_gather(
            hidden_states,
            dim=0,
            world_size=self._config.ep_size,
            group_name=self._config.ep_group_name,
        )
        router_logits = ops.all_gather(
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
        if self._config.partitions_replicated_input:
            if not self._use_prefill_fast_path():
                # Legacy decode path: sum partial expert outputs across EP and
                # keep the replicated token layout expected by the graph.
                if self._config.ep_size > 1:
                    ops.all_reduce_(
                        hidden_states,
                        group_name=self._config.ep_group_name,
                    )
                return hidden_states
            if self._config.ep_size > 1:
                hidden_states = ops.reduce_scatter(
                    hidden_states,
                    dim=0,
                    world_size=self._config.ep_size,
                    group_name=self._config.ep_group_name,
                )
            hidden_states = ops.all_gather(
                hidden_states,
                dim=0,
                world_size=self._config.input_tp_size,
                group_name=self._config.input_tp_group_name,
            )
            return hidden_states[: self._original_num_tokens]
        if self._config.ep_size > 1:
            hidden_states = ops.reduce_scatter(
                hidden_states,
                dim=0,
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
        if self._config.partitions_replicated_input:
            prepared = self._partition_replicated_input(
                hidden_states,
                router_logits,
            )
            local_tokens = prepared.hidden_states.shape[0]
            if local_tokens > self._config.mc2_tokens_capacity:
                raise ValueError(f"MC2 token count {local_tokens} exceeds capacity {self._config.mc2_tokens_capacity}")
            active_mask = (
                torch.arange(
                    prepared.padded_hidden_states_shape[0],
                    device=hidden_states.device,
                )
                < self._num_tokens
            )
            active_mask = torch.tensor_split(
                active_mask,
                self._config.input_tp_size,
                dim=0,
            )[self._config.input_tp_rank]
            return MoEPrepareOutput(
                hidden_states=prepared.hidden_states,
                router_logits=prepared.router_logits,
                padded_hidden_states_shape=(prepared.padded_hidden_states_shape),
                active_mask=active_mask,
            )
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


__all__ = [
    "AllGatherPrepareAndFinalize",
    "AllToAllPrepareAndFinalize",
    "MC2PrepareAndFinalize",
    "PrepareAndFinalize",
    "TensorParallelPrepareAndFinalize",
]
