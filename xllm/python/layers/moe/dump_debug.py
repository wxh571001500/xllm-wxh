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

"""Debug dump helpers for MoE cross-framework alignment.

Dumps the first ``XLLM_MOE_DUMP_STEPS`` MoE forward steps (prefill + decode)
of each rank into ``step0``, ``step1``, ... subdirectories. Delete the dump
directory before sending the real request to restart numbering from step0
(skipping warmup).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.distributed as dist

_DUMP_DIR = os.environ.get("XLLM_MOE_DUMP_DIR")
_MAX_STEPS = int(os.environ.get("XLLM_MOE_DUMP_STEPS", "300"))

_step_counter = {"n": 0}
_current_step = {"v": -1}
_CONTEXT_RANK = {"v": None}


def set_dump_rank(rank: int) -> None:
    """Record the current process rank (config.rank == node_rank)."""
    _CONTEXT_RANK["v"] = int(rank)


def _in_capture() -> bool:
    try:
        from xllm.python.model_executor.forward_context import get_forward_context

        return get_forward_context().acl_graph is not None
    except Exception:
        return False


def _global_rank() -> int:
    if _CONTEXT_RANK["v"] is not None:
        return _CONTEXT_RANK["v"]
    try:
        from xllm.python.distributed import parallel_group_rank

        return int(parallel_group_rank("world", torch.device("npu:0")))
    except Exception:
        pass
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _alloc_step(rank: int) -> int:
    marker = os.path.join(_DUMP_DIR, "step0", f"rank{rank}_moe_input.npy")
    if not os.path.exists(marker):
        _step_counter["n"] = 0
    step = _step_counter["n"]
    _step_counter["n"] += 1
    return step


def _save(step: int, rank: int, name: str, tensor: torch.Tensor) -> None:
    if tensor is None:
        return
    out_dir = os.path.join(_DUMP_DIR, f"step{step}")
    os.makedirs(out_dir, exist_ok=True)
    arr = tensor.detach().cpu()
    if name == "topk_ids":
        arr = arr.to(torch.int64).numpy()
    else:
        arr = arr.to(torch.float32).numpy()
    np.save(os.path.join(out_dir, f"rank{rank}_{name}.npy"), arr)


def dump_routing(
    moe_input: torch.Tensor,
    router_logits: torch.Tensor,
    prepared_router_logits: torch.Tensor | None,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    """Dump the routing-stage tensors of one MoE forward step."""
    if _DUMP_DIR is None:
        return
    rank = _global_rank()
    step = _alloc_step(rank)
    try:
        from xllm.python.model_executor.forward_context import get_forward_context

        m = get_forward_context().metadata
        ph = (
            "prefill"
            if (m is not None and (getattr(m, "is_prefill", False) or getattr(m, "is_chunked_prefill", False)))
            else "decode"
        )
    except Exception:
        ph = "?"
    import sys

    _ = sys.stderr.write(f"[MOE-DBG] rank={rank} step={step} phase={ph} moe_input={tuple(moe_input.shape)}\n")
    _current_step["v"] = step
    if step >= _MAX_STEPS or _in_capture():
        return
    _save(step, rank, "moe_input", moe_input)
    _save(step, rank, "router_logits", router_logits)
    _save(step, rank, "prepared_router_logits", prepared_router_logits)
    _save(step, rank, "topk_ids", topk_ids)
    _save(step, rank, "topk_weights", topk_weights)


def dump_moe_output(routed_output: torch.Tensor, output: torch.Tensor) -> None:
    """Dump the final MoE-layer output (routed + shared) of the current step."""
    if _DUMP_DIR is None:
        return
    step = _current_step["v"]
    if step < 0 or step >= _MAX_STEPS or _in_capture():
        return
    rank = _global_rank()
    _save(step, rank, "routed_output", routed_output)
    _save(step, rank, "moe_output", output)


def dump_input_ids(input_ids: torch.Tensor) -> None:
    """Dump the raw prompt token ids of the first prefill step (rank-shard-agnostic)."""
    if _DUMP_DIR is None or _in_capture():
        return
    rank = _global_rank()
    marker = os.path.join(_DUMP_DIR, "step0", f"rank{rank}_input_ids.npy")
    if os.path.exists(marker):
        return
    out_dir = os.path.join(_DUMP_DIR, "step0")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"rank{rank}_input_ids.npy"), input_ids.detach().cpu().to(torch.int64).numpy())
    try:
        import torch.distributed as _d

        _init = _d.is_available() and _d.is_initialized()
        _dr = "n/a"
        _dw = "n/a"
        if _init:
            try:
                _dr = str(_d.get_rank())
            except Exception as _e:
                _dr = f"err:{_e}"
            try:
                _dw = str(_d.get_world_size())
            except Exception as _e:
                _dw = f"err:{_e}"
        else:
            _dr = "not_initialized"
            _dw = "not_initialized"
        info = (
            f"dist_rank={_dr} dist_world={_dw} pid={os.getpid()} "
            f"ENV_RANK={os.environ.get('RANK')} ENV_LRANK={os.environ.get('LOCAL_RANK')} "
            f"ENV_WORLD_SIZE={os.environ.get('WORLD_SIZE')}"
        )
        with open(os.path.join(out_dir, f"rank{rank}_config_{os.getpid()}.txt"), "w") as _f:
            _f.write(info + "\n")
    except Exception as _e:
        pass


def dump_intermediate(name: str, tensor: torch.Tensor, layer: int = -1) -> None:
    """Dump a named intermediate tensor once (first prefill) for op-by-op alignment."""
    if _DUMP_DIR is None or _in_capture():
        return
    # Prefill runs with multiple tokens; decode runs with a single token.
    # The profiler warmup batch is padded to max_num_batched_tokens (8192), so
    # gate on a sane token range to capture the real prefill only.
    if not (1 < tensor.shape[0] < 1000):
        return
    rank = _global_rank()
    out_dir = os.path.join(_DUMP_DIR, "intermediate")
    os.makedirs(out_dir, exist_ok=True)
    key = f"rank{rank}_L{layer}_{name}" if layer >= 0 else f"rank{rank}_{name}"
    path = os.path.join(out_dir, key + ".npy")
    if os.path.exists(path):
        return
    np.save(path, tensor.detach().cpu().to(torch.float32).numpy())


__all__ = ["dump_routing", "dump_moe_output", "dump_input_ids", "dump_intermediate"]
