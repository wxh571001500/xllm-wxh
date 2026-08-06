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

from xllm.python import ops
from xllm.python.layers.moe.types import MoEPrepareOutput


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

    def __init__(self, tp_size: int) -> None:
        self._tp_size = tp_size

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
            ops.all_reduce_(hidden_states)
        return hidden_states


__all__ = ["PrepareAndFinalize", "TensorParallelPrepareAndFinalize"]
