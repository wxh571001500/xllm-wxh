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

import torch
import torch.nn.functional as F

from xllm.python import ops
from xllm.python.layers.moe.runtime import get_moe_batch_metadata
from xllm.python.layers.moe.types import MoEParallelConfig, MoEPrepareOutput


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

    def _runtime_max_tokens(
        self,
        local_tokens: int,
        partitioned_input: bool,
    ) -> int | None:
        metadata = get_moe_batch_metadata()
        if metadata is None:
            return None
        if metadata.local_num_tokens != self._num_tokens:
            raise ValueError(
                "MoE runtime metadata does not match the input token count: "
                f"expected {self._num_tokens}, got "
                f"{metadata.local_num_tokens}"
            )
        max_tokens = metadata.max_num_tokens
        if partitioned_input:
            max_tokens = (
                max_tokens + self._config.input_tp_size - 1
            ) // self._config.input_tp_size
        if max_tokens < local_tokens:
            raise ValueError(
                "MoE runtime padding target is smaller than the local input: "
                f"target {max_tokens}, local {local_tokens}"
            )
        return max_tokens

    def _runtime_actual_tokens(self) -> int:
        metadata = get_moe_batch_metadata()
        if metadata is None:
            return self._num_tokens
        if metadata.local_num_tokens != self._num_tokens:
            raise ValueError(
                "MoE runtime metadata does not match the input token count"
            )
        return metadata.local_actual_tokens

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

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> MoEPrepareOutput:
        if self._config.partitions_replicated_input:
            prepared = self._partition_replicated_input(
                hidden_states,
                router_logits,
            )
            hidden_states = prepared.hidden_states
            router_logits = prepared.router_logits
            input_shape = prepared.padded_hidden_states_shape
        else:
            self._num_tokens = hidden_states.shape[0]
            input_shape = None
        if self._config.ep_size == 1:
            return MoEPrepareOutput(hidden_states, router_logits)

        local_tokens = hidden_states.shape[0]
        max_tokens = self._runtime_max_tokens(
            local_tokens,
            self._config.partitions_replicated_input,
        )
        if max_tokens is None:
            token_counts = self._gather_token_counts(hidden_states)
            max_tokens = int(token_counts.max().item())
        pad_size = max_tokens - local_tokens
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
            padded_hidden_states_shape=input_shape,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        if self._config.ep_size > 1:
            hidden_states = ops.reduce_scatter(
                hidden_states,
                dim=0,
                world_size=self._config.ep_size,
                group_name=self._config.ep_group_name,
            )
            local_tokens = (
                padded_hidden_states_shape[0] // self._config.input_tp_size
                if self._config.partitions_replicated_input
                and padded_hidden_states_shape is not None
                else self._num_tokens
            )
            hidden_states = hidden_states[:local_tokens]
        if self._config.partitions_replicated_input:
            return self._gather_partitioned_output(
                hidden_states,
                padded_hidden_states_shape,
            )
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
            hidden_states = prepared.hidden_states
            router_logits = prepared.router_logits
            local_tokens = hidden_states.shape[0]
            max_tokens = self._runtime_max_tokens(
                local_tokens,
                partitioned_input=True,
            )
            if max_tokens is None:
                token_counts = self._gather_token_counts(hidden_states)
                max_tokens = int(token_counts.max().item())
            if max_tokens > self._config.mc2_tokens_capacity:
                raise ValueError(
                    f"MC2 token count {max_tokens} exceeds capacity "
                    f"{self._config.mc2_tokens_capacity}"
                )
            actual_tokens = self._runtime_actual_tokens()
            active_mask = torch.arange(
                prepared.padded_hidden_states_shape[0],
                device=hidden_states.device,
            ) < actual_tokens
            active_mask = torch.tensor_split(
                active_mask,
                self._config.input_tp_size,
                dim=0,
            )[self._config.input_tp_rank]
            pad_size = max_tokens - local_tokens
            if pad_size > 0:
                hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = F.pad(router_logits, (0, 0, 0, pad_size))
                active_mask = F.pad(active_mask, (0, pad_size), value=False)
            return MoEPrepareOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                padded_hidden_states_shape=(
                    prepared.padded_hidden_states_shape
                ),
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

        max_tokens = self._runtime_max_tokens(
            self._num_tokens,
            partitioned_input=False,
        )
        if max_tokens is None:
            token_counts = self._gather_token_counts(hidden_states)
            max_tokens = int(token_counts.max().item())
        if max_tokens > self._config.mc2_tokens_capacity:
            raise ValueError(
                f"MC2 token count {max_tokens} exceeds capacity "
                f"{self._config.mc2_tokens_capacity}"
            )
        active_mask = torch.arange(
            max_tokens,
            device=hidden_states.device,
        ) < self._runtime_actual_tokens()
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
            if padded_hidden_states_shape is None:
                raise ValueError(
                    "MC2 partitioned input requires its padded shape"
                )
            local_tokens = (
                padded_hidden_states_shape[0]
                // self._config.input_tp_size
            )
            return self._gather_partitioned_output(
                hidden_states[:local_tokens],
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
