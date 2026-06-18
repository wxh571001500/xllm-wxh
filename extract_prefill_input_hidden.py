#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
import math
import zipfile
import warnings
from pathlib import Path
from typing import Any


def _load_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyTorch is required to read hidden.pt dumps") from exc
    return torch


def _extract_tensor_like(value):
    torch = _load_torch()
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _extract_tensor_like(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_tensor_like(item)
            if tensor is not None:
                return tensor
    return None


def _load_tensor(path: Path):
    torch = _load_torch()
    loaded = None
    if zipfile.is_zipfile(path):
        try:
            loaded = torch.jit.load(str(path), map_location="cpu")
        except Exception:  # noqa: BLE001
            loaded = None
    if loaded is None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*TorchScript archive.*",
                category=UserWarning,
            )
            try:
                loaded = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                loaded = torch.load(path, map_location="cpu")
    tensor = _extract_tensor_like(loaded)
    if tensor is None:
        raise RuntimeError(f"unable to extract tensor from {path}")
    return tensor.float().contiguous()


def _iter_meta(dump_root: Path):
    for path in sorted(dump_root.rglob("meta.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                yield path.parent, json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skip unreadable meta {path}: {exc}")


def _rank_key(rank: int, rank_mod: int | None) -> int:
    if rank_mod is not None and rank_mod > 0 and rank >= 0:
        return rank % rank_mod
    return rank


def _find_prefill_input_event(
    dump_root: Path,
    request_id: str,
    rank_mod: int | None,
    wanted_rank_key: int | None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for event_dir, meta in _iter_meta(dump_root):
        if meta.get("empty_dp_request"):
            continue
        if meta.get("model") != "target":
            continue
        if meta.get("stage") != "prefill":
            continue
        if meta.get("point") != "model_input_hidden":
            continue
        if int(meta.get("layer", -1)) != -1:
            continue
        if meta.get("tensor_kind") != "hidden":
            continue
        if not meta.get("hidden_file"):
            continue
        rank = int(meta.get("rank", -1))
        rank_key = _rank_key(rank, rank_mod)
        if wanted_rank_key is not None and rank_key != wanted_rank_key:
            continue
        for request in meta.get("requests", []):
            if request.get("request_id") == request_id:
                matches.append((event_dir, meta, request))
    if not matches:
        raise SystemExit(f"no target prefill input hidden event for {request_id}")
    matches.sort(key=lambda item: (int(item[1].get("event_index", -1)), str(item[0])))
    return matches[0]


def _request_hidden(event_dir: Path, meta: dict[str, Any], request: dict[str, Any]):
    hidden = _load_tensor(event_dir / str(meta["hidden_file"]))
    rows = [int(row) for row in request.get("token_rows", [])]
    if not rows:
        raise RuntimeError(f"request {request.get('request_id')} has empty token_rows")
    row_tensor = _load_torch().tensor(rows, dtype=_load_torch().long)
    return hidden.index_select(0, row_tensor).contiguous()


def _cosine_per_token(left, right) -> list[float | None]:
    count = min(left.shape[0], right.shape[0])
    values: list[float | None] = []
    for index in range(count):
        left_vec = left[index].reshape(-1)
        right_vec = right[index].reshape(-1)
        denom = float(left_vec.norm().item() * right_vec.norm().item())
        if denom <= 1e-12 or left_vec.numel() != right_vec.numel():
            values.append(None)
            continue
        cos = float(left_vec.dot(right_vec).item()) / denom
        values.append(cos if math.isfinite(cos) else None)
    return values


def _print_event(label: str, event_dir: Path, meta: dict[str, Any], request: dict[str, Any]) -> None:
    rows = [int(row) for row in request.get("token_rows", [])]
    print(
        f"{label}: rank={meta.get('rank')} event={meta.get('event_index')} "
        f"num_sequences={meta.get('num_sequences')} "
        f"request_index={request.get('request_index')} "
        f"q_seq_len={request.get('q_seq_len')} kv_seq_len={request.get('kv_seq_len')} "
        f"rows={rows[:6]}...{rows[-6:]} dir={event_dir}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract target prefill model_input_hidden for two request ids "
            "from xLLM spec feature dumps."
        )
    )
    parser.add_argument("--dump-root", required=True)
    parser.add_argument("left_request_id")
    parser.add_argument("right_request_id")
    parser.add_argument("--rank-mod", type=int, default=None)
    parser.add_argument("--rank-key", type=int, default=None)
    parser.add_argument("--save-prefix", default="")
    parser.add_argument("--print-tokens", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch = _load_torch()
    dump_root = Path(args.dump_root)
    left_event_dir, left_meta, left_request = _find_prefill_input_event(
        dump_root, args.left_request_id, args.rank_mod, args.rank_key
    )
    right_event_dir, right_meta, right_request = _find_prefill_input_event(
        dump_root, args.right_request_id, args.rank_mod, args.rank_key
    )
    _print_event("left", left_event_dir, left_meta, left_request)
    _print_event("right", right_event_dir, right_meta, right_request)

    left_hidden = _request_hidden(left_event_dir, left_meta, left_request)
    right_hidden = _request_hidden(right_event_dir, right_meta, right_request)
    print(f"left_hidden shape={list(left_hidden.shape)}")
    print(f"right_hidden shape={list(right_hidden.shape)}")

    cos_values = _cosine_per_token(left_hidden, right_hidden)
    valid = [value for value in cos_values if value is not None]
    if valid:
        min_index, min_value = min(
            ((index, value) for index, value in enumerate(cos_values) if value is not None),
            key=lambda item: item[1],
        )
        print(
            f"cos mean={sum(valid) / len(valid):.9g} "
            f"min={min_value:.9g} min_token={min_index}"
        )
    token_limit = len(cos_values) if args.print_tokens < 0 else args.print_tokens
    print("token_cos")
    for index, value in enumerate(cos_values[:token_limit]):
        text = "NA" if value is None else f"{value:.9g}"
        print(f"  {index}\t{text}")

    if args.save_prefix:
        prefix = Path(args.save_prefix)
        torch.save(left_hidden, f"{prefix}.left_hidden.pt")
        torch.save(right_hidden, f"{prefix}.right_hidden.pt")
        torch.save(torch.tensor([value if value is not None else float('nan') for value in cos_values]), f"{prefix}.cos.pt")
        print(f"saved {prefix}.left_hidden.pt")
        print(f"saved {prefix}.right_hidden.pt")
        print(f"saved {prefix}.cos.pt")


if __name__ == "__main__":
    main()
