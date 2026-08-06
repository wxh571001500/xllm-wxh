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

from xllm.python.layers.moe.experts import RoutedExperts
from xllm.python.layers.moe.prepare_finalize import (
    PrepareAndFinalize,
    TensorParallelPrepareAndFinalize,
)
from xllm.python.layers.moe.token_dispatcher import MoETokenDispatcher
from xllm.python.layers.moe.types import (
    MoEFusedExpertsResult,
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


__all__ = ["MoECommMethod", "TensorParallelCommMethod"]
