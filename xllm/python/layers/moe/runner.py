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

"""Weightless execution coordinator for reusable MoE components."""

from __future__ import annotations

from typing import Callable

import torch

from xllm.python.layers.moe.communication import MoECommMethod
from xllm.python.layers.moe.experts import RoutedExperts
from xllm.python.layers.moe.router import MoERouter

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


__all__ = ["MoERunner"]
