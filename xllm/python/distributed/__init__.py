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

"""Distributed execution for the Python model executor."""

from __future__ import annotations

import importlib
from typing import Any

from xllm.python.distributed.collectives import (
    all_gather,
    all_gather_variable,
    all_reduce_,
    broadcast_,
    cp_rank,
    cp_world_size,
    dcp_group,
    dp_all_gather,
    gather_dp_execution_tokens,
    init_process_group,
    init_tp_group,
    layerwise_rank,
    moe_ep_all_reduce,
    moe_ep_group,
    moe_ep_hccl_info,
    moe_tp_all_reduce,
    reduce_scatter,
    tp_all_gather,
    tp_all_reduce,
    tp_rank,
)

__all__ = [
    "init_process_group",
    "init_tp_group",
    "tp_rank",
    "cp_rank",
    "cp_world_size",
    "layerwise_rank",
    "broadcast_",
    "tp_all_reduce",
    "tp_all_gather",
    "dp_all_gather",
    "moe_tp_all_reduce",
    "moe_ep_all_reduce",
    "moe_ep_group",
    "moe_ep_hccl_info",
    "dcp_group",
    "all_reduce_",
    "all_gather",
    "reduce_scatter",
    "all_gather_variable",
    "gather_dp_execution_tokens",
]

_NZ_EXPORTS = frozenset(
    (
        "all_gather_nz_bytes",
        "empty_nz",
        "shard_nz",
    )
)


def __getattr__(name: str) -> Any:
    """Load NZ AllGather helpers only when requested (avoids importing torch_npu).

    Not listed in ``__all__`` so ``from xllm.python.distributed import *`` does
    not pull in torch_npu. Import explicitly, e.g.
    ``from xllm.python.distributed import all_gather_nz_bytes``.
    """
    if name not in _NZ_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module("xllm.python.distributed.all_gather_nz_bytes")
    return getattr(mod, name)
