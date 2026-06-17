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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HiddenEvent:
    request_id: str
    event_dir: Path
    rank: int
    rank_key: int
    event_index: int
    layer: int
    point: str
    request_index: int
    token_rows: list[int]
    tensor_path: Path


@dataclass(frozen=True)
class TensorCompare:
    same_shape: bool
    equal: bool
    mean_cos: float | None
    min_cos: float | None
    first_bad_index: int | None
    first_bad_cos: float | None
    max_abs: float | None
    l2_diff: float | None


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


def _rank_key(rank: int, rank_mod: int | None) -> int:
    if rank_mod is not None and rank_mod > 0 and rank >= 0:
        return rank % rank_mod
    return rank


def _iter_meta(dump_roots: list[Path]):
    for dump_root in dump_roots:
        for meta_path in sorted(dump_root.rglob("meta.json")):
            try:
                with meta_path.open("r", encoding="utf-8") as handle:
                    yield meta_path.parent, json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"skip unreadable meta {meta_path}: {exc}")


def _is_target_prefill_hidden(meta: dict[str, Any]) -> bool:
    return (
        not meta.get("empty_dp_request")
        and meta.get("model") == "target"
        and meta.get("stage") == "prefill"
        and meta.get("tensor_kind") == "hidden"
        and bool(meta.get("hidden_file"))
        and meta.get("point") in {"layer_input_hidden", "layer_output_hidden"}
    )


def _collect_events(
    dump_roots: list[Path],
    request_ids: set[str],
    rank_mod: int | None,
    rank_key: int,
    max_layer: int,
) -> dict[tuple[str, int, str], HiddenEvent]:
    candidates: dict[tuple[str, int, str], list[HiddenEvent]] = {}
    for event_dir, meta in _iter_meta(dump_roots):
        if not _is_target_prefill_hidden(meta):
            continue
        layer = int(meta.get("layer", -1))
        if layer < 0 or layer > max_layer:
            continue
        rank = int(meta.get("rank", -1))
        current_rank_key = _rank_key(rank, rank_mod)
        if current_rank_key != rank_key:
            continue
        point = str(meta.get("point", ""))
        for request in meta.get("requests", []):
            request_id = str(request.get("request_id", ""))
            if request_id not in request_ids:
                continue
            rows = [int(row) for row in request.get("token_rows", [])]
            if not rows:
                continue
            key = (request_id, layer, point)
            candidates.setdefault(key, []).append(
                HiddenEvent(
                    request_id=request_id,
                    event_dir=event_dir,
                    rank=rank,
                    rank_key=current_rank_key,
                    event_index=int(meta.get("event_index", -1)),
                    layer=layer,
                    point=point,
                    request_index=int(request.get("request_index", -1)),
                    token_rows=rows,
                    tensor_path=event_dir / str(meta["hidden_file"]),
                )
            )

    selected: dict[tuple[str, int, str], HiddenEvent] = {}
    for key, values in candidates.items():
        values.sort(key=lambda item: (item.event_index, item.rank, str(item.event_dir)))
        selected[key] = values[0]
    return selected


def _request_tensor(event: HiddenEvent):
    torch = _load_torch()
    tensor = _load_tensor(event.tensor_path)
    rows = torch.tensor(event.token_rows, dtype=torch.long)
    return tensor.index_select(0, rows).contiguous()


def _cosine_per_row(left, right) -> list[float | None]:
    count = min(left.shape[0], right.shape[0])
    values: list[float | None] = []
    for index in range(count):
        left_vec = left[index].reshape(-1)
        right_vec = right[index].reshape(-1)
        if left_vec.numel() != right_vec.numel():
            values.append(None)
            continue
        denom = float(left_vec.norm().item() * right_vec.norm().item())
        if denom <= 1e-12:
            values.append(None)
            continue
        cos = float(left_vec.dot(right_vec).item()) / denom
        values.append(cos if math.isfinite(cos) else None)
    return values


def _compare_tensors(left, right, threshold: float) -> TensorCompare:
    same_shape = tuple(left.shape) == tuple(right.shape)
    count = min(left.shape[0], right.shape[0])
    if count == 0:
        return TensorCompare(
            same_shape=same_shape,
            equal=False,
            mean_cos=None,
            min_cos=None,
            first_bad_index=None,
            first_bad_cos=None,
            max_abs=None,
            l2_diff=None,
        )
    left_part = left[:count]
    right_part = right[:count]
    cos_values = _cosine_per_row(left_part, right_part)
    valid = [(index, value) for index, value in enumerate(cos_values) if value is not None]
    first_bad_index = None
    first_bad_cos = None
    for index, value in valid:
        if value < threshold:
            first_bad_index = index
            first_bad_cos = value
            break
    diff = left_part - right_part
    return TensorCompare(
        same_shape=same_shape,
        equal=bool(same_shape and left.equal(right)),
        mean_cos=sum(value for _, value in valid) / len(valid) if valid else None,
        min_cos=min((value for _, value in valid), default=None),
        first_bad_index=first_bad_index,
        first_bad_cos=first_bad_cos,
        max_abs=float(diff.abs().max().item()) if diff.numel() else None,
        l2_diff=float(diff.norm().item()) if diff.numel() else None,
    )


def _format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.9g}"


def _print_event_map(events: dict[tuple[str, int, str], HiddenEvent], request_ids: list[str]) -> None:
    print("Selected rank_key events")
    for request_id in request_ids:
        print(f"  request_id={request_id}")
        for key, event in sorted(events.items(), key=lambda item: (item[0][1], item[0][2])):
            if key[0] != request_id:
                continue
            print(
                f"    layer={event.layer} point={event.point} "
                f"rank={event.rank} event={event.event_index} "
                f"request_index={event.request_index} rows={len(event.token_rows)}"
            )


def _print_chain_check(
    events: dict[tuple[str, int, str], HiddenEvent],
    request_id: str,
    max_layer: int,
    threshold: float,
) -> None:
    print(f"\nLayer chain check request_id={request_id}")
    print("  layer output_to_next_input equal mean_cos min_cos first_bad max_abs l2")
    for layer in range(max_layer):
        left_event = events.get((request_id, layer, "layer_output_hidden"))
        right_event = events.get((request_id, layer + 1, "layer_input_hidden"))
        if left_event is None or right_event is None:
            print(f"  {layer}->{layer + 1} missing")
            continue
        result = _compare_tensors(
            _request_tensor(left_event), _request_tensor(right_event), threshold
        )
        print(
            f"  {layer}->{layer + 1} {result.equal} "
            f"{_format_float(result.mean_cos)} {_format_float(result.min_cos)} "
            f"{result.first_bad_index} {_format_float(result.max_abs)} "
            f"{_format_float(result.l2_diff)}"
        )


def _print_request_compare(
    events: dict[tuple[str, int, str], HiddenEvent],
    left_request_id: str,
    right_request_id: str,
    max_layer: int,
    threshold: float,
) -> None:
    print(f"\nCross-request layer compare left={left_request_id} right={right_request_id}")
    print("  layer point same_shape equal mean_cos min_cos first_bad first_bad_cos max_abs l2")
    for layer in range(max_layer + 1):
        for point in ("layer_input_hidden", "layer_output_hidden"):
            left_event = events.get((left_request_id, layer, point))
            right_event = events.get((right_request_id, layer, point))
            if left_event is None or right_event is None:
                print(f"  {layer} {point} missing")
                continue
            result = _compare_tensors(
                _request_tensor(left_event), _request_tensor(right_event), threshold
            )
            print(
                f"  {layer} {point} {result.same_shape} {result.equal} "
                f"{_format_float(result.mean_cos)} {_format_float(result.min_cos)} "
                f"{result.first_bad_index} {_format_float(result.first_bad_cos)} "
                f"{_format_float(result.max_abs)} {_format_float(result.l2_diff)}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check target prefill layer chain consistency and compare two "
            "requests on one DP-domain local rank."
        )
    )
    parser.add_argument("--dump-root", action="append", required=True)
    parser.add_argument("--rank-mod", type=int, default=8)
    parser.add_argument("--rank-key", type=int, default=0)
    parser.add_argument("--max-layer", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--print-events", action="store_true")
    parser.add_argument("left_request_id")
    parser.add_argument("right_request_id")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    request_ids = [args.left_request_id, args.right_request_id]
    events = _collect_events(
        [Path(path) for path in args.dump_root],
        set(request_ids),
        args.rank_mod,
        args.rank_key,
        args.max_layer,
    )
    if args.print_events:
        _print_event_map(events, request_ids)
    _print_chain_check(events, args.left_request_id, args.max_layer, args.threshold)
    _print_chain_check(events, args.right_request_id, args.max_layer, args.threshold)
    _print_request_compare(
        events,
        args.left_request_id,
        args.right_request_id,
        args.max_layer,
        args.threshold,
    )


if __name__ == "__main__":
    main()
