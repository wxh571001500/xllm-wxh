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

"""Reusable MoE activation modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class SituAndMul(nn.Module):
    """SiTU gated activation used by routed and dense expert MLPs."""

    def __init__(self, beta: float, linear_beta: float | None) -> None:
        super().__init__()
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


__all__ = ["SituAndMul"]
