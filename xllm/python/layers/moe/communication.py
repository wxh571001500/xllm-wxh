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

"""MoE communication-method orchestration."""

from __future__ import annotations

import torch
import torch_npu

from xllm.python.layers.moe.experts import RoutedExperts
from xllm.python.layers.moe.prepare_finalize import (
    AllGatherPrepareAndFinalize,
    AllToAllPrepareAndFinalize,
    MC2PrepareAndFinalize,
    PrepareAndFinalize,
    TensorParallelPrepareAndFinalize,
)
from xllm.python.layers.moe.token_dispatcher import (
    AllToAllTokenDispatcher,
    FusedAllGatherTokenDispatcher,
    MC2TokenDispatcher,
    MoETokenDispatcher,
    NativeTokenDispatcher,
)
from xllm.python.layers.moe.types import (
    MoECommType,
    MoEFusedExpertsResult,
    MoEParallelConfig,
    MoEPrepareOutput,
    MoERoutingResult,
    MoETokenDispatchInput,
)


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


class TensorParallelCommMethod(MoECommMethod):
    """MoE communication method for the current replicated-token TP path."""

    def __init__(
        self,
        tp_size: int,
        token_dispatcher: MoETokenDispatcher,
    ) -> None:
        super().__init__(
            token_dispatcher=token_dispatcher,
            prepare_finalize=TensorParallelPrepareAndFinalize(tp_size),
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
    ) -> None:
        super().__init__(
            AllToAllTokenDispatcher(config, num_experts, quantized),
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
    """Select MC2 for small batches and All2All for larger EP batches."""

    def __init__(
        self,
        config: MoEParallelConfig,
        num_experts: int,
        top_k: int,
        quantized: bool,
        device: torch.device,
    ) -> None:
        self._config = config
        self._all_gather = AllGatherCommMethod(
            config,
            num_experts,
            top_k,
            quantized,
            device,
        )
        self._all_to_all = (
            AllToAllCommMethod(config, num_experts, quantized)
            if config.ep_size > 1
            else None
        )
        has_mc2 = hasattr(torch_npu, "npu_moe_distribute_dispatch") and hasattr(
            torch_npu,
            "npu_moe_distribute_combine",
        )
        self._mc2 = (
            MC2CommMethod(config, num_experts, quantized, device)
            if config.ep_size > 1
            and device.type in ("npu", "privateuseone")
            and has_mc2
            else None
        )
        self._active: MoECommMethod | None = None

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> MoEPrepareOutput:
        if self._mc2 is not None and (
            hidden_states.shape[0] <= self._config.mc2_tokens_capacity
        ):
            self._active = self._mc2
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
) -> MoECommMethod:
    """Build the configured reusable MoE communication method."""
    comm_type = config.comm_type
    if config.ep_size == 1 or comm_type == MoECommType.ALL_GATHER:
        return AllGatherCommMethod(
            config,
            num_experts,
            top_k,
            quantized,
            device,
        )
    if comm_type == MoECommType.ALL_TO_ALL:
        return AllToAllCommMethod(config, num_experts, quantized)
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


__all__ = [
    "AdaptiveMoECommMethod",
    "AllGatherCommMethod",
    "AllToAllCommMethod",
    "MC2CommMethod",
    "MoECommMethod",
    "TensorParallelCommMethod",
    "build_moe_comm_method",
]
