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

"""Token dispatch and combine stages for routed experts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch_npu

from xllm.python import ops
from xllm.python.distributed import get_parallel_group
from xllm.python.layers.moe.types import (
    MoEParallelConfig,
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


@dataclass(frozen=True)
class _AllToAllCombineMetadata:
    input_splits: list[int]
    output_splits: list[int]
    token_indices: torch.Tensor
    routing_weights: torch.Tensor
    inverse_receive_order: torch.Tensor
    num_tokens: int


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


class AllToAllTokenDispatcher(MoETokenDispatcher):
    """Exchange routed tokens with contiguous expert owners over the EP group."""

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
        routing_weights = routing.topk_weights.reshape(-1)
        send_order = torch.argsort(expert_ids, stable=True)
        sent_expert_ids = expert_ids.index_select(0, send_order)
        sent_hidden_states = hidden_states.index_select(
            0,
            token_indices.index_select(0, send_order),
        )

        local_counts = torch.histc(
            expert_ids.to(torch.float32),
            bins=self._num_experts,
            min=0,
            max=self._num_experts,
        ).to(torch.int64)
        global_counts = ops.all_gather(
            local_counts,
            dim=0,
            world_size=self._config.ep_size,
            group_name=self._config.ep_group_name,
        ).view(self._config.ep_size, self._num_experts)
        input_splits = local_counts.view(self._config.ep_size, self._num_local_experts).sum(dim=1).to("cpu").tolist()
        local_slice = slice(
            self._first_expert_id,
            self._first_expert_id + self._num_local_experts,
        )
        output_splits = global_counts[:, local_slice].sum(dim=1).to("cpu").tolist()
        received_hidden_states = ops.all_to_all_single(
            sent_hidden_states,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group_name=self._config.ep_group_name,
        )
        received_expert_ids = ops.all_to_all_single(
            sent_expert_ids,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group_name=self._config.ep_group_name,
        )
        local_expert_ids = received_expert_ids - self._first_expert_id
        receive_order = torch.argsort(local_expert_ids, stable=True)
        inverse_receive_order = torch.argsort(receive_order, stable=True)
        grouped_hidden_states = received_hidden_states.index_select(
            0,
            receive_order,
        )
        grouped_expert_ids = local_expert_ids.index_select(0, receive_order)
        group_list = torch.bincount(
            grouped_expert_ids,
            minlength=self._num_local_experts,
        ).to(torch.int64)
        dynamic_scale = None
        if self._quantized:
            if grouped_hidden_states.device.type not in ("npu", "privateuseone"):
                raise RuntimeError("Quantized All2All MoE requires an NPU")
            grouped_hidden_states, dynamic_scale = torch_npu.npu_dynamic_quant(grouped_hidden_states)
        return MoETokenDispatchOutput(
            hidden_states=grouped_hidden_states,
            group_list=group_list,
            group_list_type=1,
            combine_metadata=_AllToAllCombineMetadata(
                input_splits=input_splits,
                output_splits=output_splits,
                token_indices=token_indices.index_select(0, send_order),
                routing_weights=routing_weights.index_select(0, send_order),
                inverse_receive_order=inverse_receive_order,
                num_tokens=num_tokens,
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
        received_order = hidden_states.index_select(
            0,
            combine_metadata.inverse_receive_order,
        )
        returned = ops.all_to_all_single(
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
        self._config = config
        self._num_experts = num_experts
        self._quantized = quantized
        self._enable_v2 = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
        soc_version = int(torch_npu.npu.get_soc_version())
        self._need_tp_arguments = 250 <= soc_version <= 255 or soc_version == 260
        group = get_parallel_group(config.ep_group_name, device)
        if group.process_group is None:
            raise RuntimeError("MC2 MoE requires an EP process-group backend")
        process_group = group.process_group
        if hasattr(process_group, "_get_backend"):
            backend = process_group._get_backend(torch.device("npu"))
        elif hasattr(process_group, "get_hccl_comm_name"):
            backend = process_group
        else:
            raise RuntimeError("MC2 MoE requires an HCCL process-group backend")
        self._group_name = backend.get_hccl_comm_name(group.local_rank)

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
                    "expert_scales": routing.topk_weights.to(torch.float32),
                    "group_tp": self._group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )
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


__all__ = [
    "AllToAllTokenDispatcher",
    "FusedAllGatherTokenDispatcher",
    "MC2TokenDispatcher",
    "MoETokenDispatcher",
    "NativeTokenDispatcher",
]
