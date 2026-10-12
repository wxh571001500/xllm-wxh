# Copyright 2026 The xLLM Authors.
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

"""Backend-neutral checkpoint helpers for Qwen3.5 gated delta nets."""

from __future__ import annotations

import torch

from xllm.python.layers.qwen3_5.common import Qwen3_5GatedDeltaNetConfig
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader


def shard_qkv_rows(
    state: ScopedWeightLoader,
    tensor: torch.Tensor,
    local_name: str,
    cfg: Qwen3_5GatedDeltaNetConfig,
    context: ParallelLoadContext,
) -> torch.Tensor:
    """Split checkpoint Q/K/V rows and pack this rank's local TP shard."""
    global_key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    global_value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    q, k, v = tensor.split(
        (global_key_dim, global_key_dim, global_value_dim),
        dim=0,
    )
    return torch.cat(
        [
            state.shard_value(
                part,
                f"{local_name}[{tag}]",
                0,
                context.tp_rank,
                context.tp_size,
            )
            for part, tag in ((q, "q"), (k, "k"), (v, "v"))
        ]
    )
