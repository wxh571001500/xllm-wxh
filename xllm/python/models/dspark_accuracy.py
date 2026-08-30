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

"""Opt-in tensor dumps for cross-engine Kimi K3 DSpark accuracy checks."""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch

_DUMP_LOCK = threading.Lock()
_DUMP_INDICES: defaultdict[str, int] = defaultdict(int)
_DUMP_REQUEST_KEY: tuple[str, ...] | None = None
_DUMP_WARMUP = False


def _dump_directory() -> Path | None:
    value = os.getenv("XLLM_DSPARK_ACCURACY_DUMP_DIR")
    return Path(value) if value else None


def _global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))


def snapshot_for_dump(tensor):
    if tensor is None or not is_dspark_accuracy_dump_enabled():
        return None
    if tensor.device.type in ("npu", "privateuseone"):
        torch.npu.synchronize()
    snapshot = (
        tensor.detach()
        .to(
            device="cpu",
            non_blocking=False,
        )
        .contiguous()
    )
    return snapshot.clone()


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    return bool(is_compiling is not None and is_compiling())


def _dump_triggered() -> bool:
    trigger_file = os.getenv("XLLM_DSPARK_ACCURACY_TRIGGER_FILE")
    return trigger_file is None or Path(trigger_file).is_file()


def is_dspark_accuracy_dump_enabled() -> bool:
    """Return whether this process should retain tensors for accuracy dumps."""
    if _is_compiling():
        return False
    return _dump_directory() is not None and _dump_triggered() and not _DUMP_WARMUP and _global_rank() == 0


def set_dspark_accuracy_context(
    is_warmup: bool,
    request_ids: Sequence[str],
) -> None:
    """Reset dump numbering for a new request and suppress warmup dumps."""
    global _DUMP_REQUEST_KEY, _DUMP_WARMUP
    request_key = tuple(str(request_id) for request_id in request_ids)
    with _DUMP_LOCK:
        _DUMP_WARMUP = bool(is_warmup)
        if _DUMP_WARMUP or not request_key or request_key == _DUMP_REQUEST_KEY:
            return
        _DUMP_REQUEST_KEY = request_key
        _DUMP_INDICES.clear()
        dump_directory = _dump_directory()
        if dump_directory is None:
            return
        for path in dump_directory.glob("*_call_*_rank_000.pt"):
            path.unlink(missing_ok=True)
        for path in dump_directory.glob("*_call_*_rank_000.inputs.json"):
            path.unlink(missing_ok=True)


def dump_dspark_tensors(
    component: str,
    tensors: dict[str, torch.Tensor | None],
) -> None:
    """Save one semantic DSpark boundary on rank zero when tracing is enabled."""
    if not is_dspark_accuracy_dump_enabled():
        return
    dump_directory = _dump_directory()
    assert dump_directory is not None

    max_calls = int(os.getenv("XLLM_DSPARK_ACCURACY_MAX_CALLS", "32"))
    with _DUMP_LOCK:
        call_index = _DUMP_INDICES[component]
        _DUMP_INDICES[component] += 1
    if call_index >= max_calls:
        return

    records: dict[str, dict[str, object]] = {}
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        cpu_tensor = snapshot_for_dump(tensor)
        if cpu_tensor.dtype != torch.float32:
            cpu_tensor = cpu_tensor.to(torch.float32)
        records[name] = {
            "shape": list(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype),
            "values": cpu_tensor,
        }

    dump_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "framework": "xllm",
        "component": component,
        "call_index": call_index,
        "rank": _global_rank(),
        "tensors": records,
    }
    path = dump_directory / f"{component}_call_{call_index:04d}_rank_000.pt"
    torch.save(payload, path)

    input_records = {
        name: {
            "shape": record["shape"],
            "values": record["values"].reshape(-1).tolist(),
        }
        for name, record in records.items()
        if name.endswith(".input_ids") or name.endswith(".positions")
    }
    if input_records:
        input_path = dump_directory / (f"{component}_call_{call_index:04d}_rank_000.inputs.json")
        input_path.write_text(
            json.dumps(
                {
                    "framework": "xllm",
                    "component": component,
                    "call_index": call_index,
                    "rank": _global_rank(),
                    "tensors": input_records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
