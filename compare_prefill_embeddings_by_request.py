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


DEFAULT_IMAGE_TOKEN_IDS = "163605"


@dataclass(frozen=True)
class RequestEmbedding:
    request_id: str
    event_dir: Path
    rank: int
    rank_key: int
    event_index: int
    request_index: int
    q_seq_len: int
    token_ids: list[int]
    hidden: Any


@dataclass(frozen=True)
class PartCompareResult:
    name: str
    count: int
    same_shape: bool
    mean_cos: float | None
    min_cos: float | None
    min_index: int | None
    first_bad_index: int | None
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


def _is_prefill_input_meta(meta: dict[str, Any]) -> bool:
    return (
        not meta.get("empty_dp_request")
        and meta.get("model") == "target"
        and meta.get("stage") == "prefill"
        and meta.get("point") == "model_input_hidden"
        and int(meta.get("layer", -1)) == -1
        and meta.get("tensor_kind") == "hidden"
        and bool(meta.get("hidden_file"))
    )


def _load_request_embeddings(
    dump_roots: list[Path],
    request_ids: list[str],
    rank_mod: int | None,
    rank_key: int | None,
) -> dict[str, RequestEmbedding]:
    wanted = set(request_ids)
    matches: dict[str, list[tuple[Path, dict[str, Any], dict[str, Any]]]] = {
        request_id: [] for request_id in request_ids
    }
    for dump_root in dump_roots:
        for event_dir, meta in _iter_meta(dump_root):
            if not _is_prefill_input_meta(meta):
                continue
            rank = int(meta.get("rank", -1))
            current_rank_key = _rank_key(rank, rank_mod)
            if rank_key is not None and current_rank_key != rank_key:
                continue
            for request in meta.get("requests", []):
                request_id = request.get("request_id")
                if request_id in wanted:
                    matches[request_id].append((event_dir, meta, request))

    records: dict[str, RequestEmbedding] = {}
    torch = _load_torch()
    for request_id in request_ids:
        candidates = matches.get(request_id, [])
        if not candidates:
            print(f"missing request_id={request_id}")
            continue
        candidates.sort(
            key=lambda item: (
                int(item[1].get("event_index", -1)),
                int(item[1].get("rank", -1)),
                str(item[0]),
            )
        )
        event_dir, meta, request = candidates[0]
        hidden = _load_tensor(event_dir / str(meta["hidden_file"]))
        rows = [int(row) for row in request.get("token_rows", [])]
        token_ids = [int(token_id) for token_id in request.get("token_ids", [])]
        if not rows:
            print(f"skip request_id={request_id}: empty token_rows")
            continue
        if len(token_ids) != len(rows):
            print(
                f"skip request_id={request_id}: token_ids length {len(token_ids)} "
                f"does not match rows length {len(rows)}"
            )
            continue
        row_tensor = torch.tensor(rows, dtype=torch.long)
        request_hidden = hidden.index_select(0, row_tensor).contiguous()
        records[request_id] = RequestEmbedding(
            request_id=request_id,
            event_dir=event_dir,
            rank=int(meta.get("rank", -1)),
            rank_key=_rank_key(int(meta.get("rank", -1)), rank_mod),
            event_index=int(meta.get("event_index", -1)),
            request_index=int(request.get("request_index", -1)),
            q_seq_len=int(request.get("q_seq_len", 0)),
            token_ids=token_ids,
            hidden=request_hidden,
        )
    return records


def _select_rows(record: RequestEmbedding, image_token_ids: set[int], image: bool):
    torch = _load_torch()
    indices = [
        index
        for index, token_id in enumerate(record.token_ids)
        if (token_id in image_token_ids) == image
    ]
    if not indices:
        return record.hidden.new_empty((0, record.hidden.shape[-1]))
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return record.hidden.index_select(0, index_tensor).contiguous()


def _cosine_values(left, right) -> list[float | None]:
    values: list[float | None] = []
    count = min(left.shape[0], right.shape[0])
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
        value = float(left_vec.dot(right_vec).item()) / denom
        values.append(value if math.isfinite(value) else None)
    return values


def _compare_part(name: str, left, right, threshold: float) -> PartCompareResult:
    same_shape = tuple(left.shape) == tuple(right.shape)
    count = min(left.shape[0], right.shape[0])
    if count == 0:
        return PartCompareResult(
            name=name,
            count=0,
            same_shape=same_shape,
            mean_cos=None,
            min_cos=None,
            min_index=None,
            first_bad_index=None,
            max_abs=None,
            l2_diff=None,
        )

    left_part = left[:count]
    right_part = right[:count]
    cos_values = _cosine_values(left_part, right_part)
    valid = [(index, value) for index, value in enumerate(cos_values) if value is not None]
    mean_cos = None
    min_cos = None
    min_index = None
    first_bad_index = None
    if valid:
        mean_cos = sum(value for _, value in valid) / len(valid)
        min_index, min_cos = min(valid, key=lambda item: item[1])
        for index, value in valid:
            if value < threshold:
                first_bad_index = index
                break
    diff = left_part - right_part
    return PartCompareResult(
        name=name,
        count=count,
        same_shape=same_shape,
        mean_cos=mean_cos,
        min_cos=min_cos,
        min_index=min_index,
        first_bad_index=first_bad_index,
        max_abs=float(diff.abs().max().item()) if diff.numel() else None,
        l2_diff=float(diff.norm().item()) if diff.numel() else None,
    )


def _format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.9g}"


def _print_record(record: RequestEmbedding, image_token_ids: set[int]) -> None:
    image_count = sum(1 for token_id in record.token_ids if token_id in image_token_ids)
    word_count = len(record.token_ids) - image_count
    print(
        f"  {record.request_id} rank={record.rank} rank_key={record.rank_key} "
        f"event={record.event_index} request_index={record.request_index} "
        f"q_seq_len={record.q_seq_len} word_rows={word_count} "
        f"image_rows={image_count} dir={record.event_dir}"
    )


def _print_compare(
    base: RequestEmbedding,
    current: RequestEmbedding,
    image_token_ids: set[int],
    threshold: float,
) -> None:
    base_word = _select_rows(base, image_token_ids, image=False)
    current_word = _select_rows(current, image_token_ids, image=False)
    base_image = _select_rows(base, image_token_ids, image=True)
    current_image = _select_rows(current, image_token_ids, image=True)
    results = [
        _compare_part("word", base_word, current_word, threshold),
        _compare_part("image", base_image, current_image, threshold),
    ]
    print(f"\ncompare base={base.request_id} current={current.request_id}")
    for result in results:
        status = "OK"
        if not result.same_shape:
            status = "SHAPE_DIFF"
        elif result.min_cos is not None and result.min_cos < threshold:
            status = "DIFF"
        print(
            f"  {result.name}: {status} rows={result.count} "
            f"same_shape={result.same_shape} mean_cos={_format_float(result.mean_cos)} "
            f"min_cos={_format_float(result.min_cos)} min_index={result.min_index} "
            f"first_bad_index={result.first_bad_index} "
            f"max_abs={_format_float(result.max_abs)} l2_diff={_format_float(result.l2_diff)}"
        )


def _parse_request_ids(args: argparse.Namespace) -> list[str]:
    request_ids = list(args.request_ids)
    if args.request_id_file:
        with Path(args.request_id_file).open("r", encoding="utf-8") as handle:
            for line in handle:
                request_id = line.strip()
                if request_id:
                    request_ids.append(request_id)
    seen: set[str] = set()
    unique: list[str] = []
    for request_id in request_ids:
        if request_id in seen:
            continue
        seen.add(request_id)
        unique.append(request_id)
    return unique


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether target prefill word/image embeddings are identical "
            "for a set of request ids in xLLM spec feature dumps."
        )
    )
    parser.add_argument(
        "--dump-root",
        action="append",
        required=True,
        help="xllm-dump directory. Can be passed multiple times for multi-node dumps.",
    )
    parser.add_argument("--request-id-file", default="")
    parser.add_argument("--rank-mod", type=int, default=None)
    parser.add_argument("--rank-key", type=int, default=None)
    parser.add_argument("--base-request-id", default="")
    parser.add_argument("--threshold", type=float, default=0.999999)
    parser.add_argument("--image-token-ids", default=DEFAULT_IMAGE_TOKEN_IDS)
    parser.add_argument("request_ids", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    request_ids = _parse_request_ids(args)
    if len(request_ids) < 2:
        raise SystemExit("provide at least two request ids")
    image_token_ids = {
        int(value)
        for value in args.image_token_ids.replace(",", " ").split()
        if value
    }
    records = _load_request_embeddings(
        [Path(path) for path in args.dump_root],
        request_ids,
        args.rank_mod,
        args.rank_key,
    )
    if len(records) < 2:
        print("fewer than two request ids were found in dumps; nothing to compare")
        return

    print("Loaded requests")
    for request_id in request_ids:
        record = records.get(request_id)
        if record is not None:
            _print_record(record, image_token_ids)

    base_request_id = args.base_request_id or request_ids[0]
    if base_request_id not in records:
        first_found = next(
            request_id for request_id in request_ids if request_id in records
        )
        print(
            f"base request id not found: {base_request_id}; "
            f"use first found request as base: {first_found}"
        )
        base_request_id = first_found
    base = records[base_request_id]
    for request_id in request_ids:
        if request_id == base_request_id or request_id not in records:
            continue
        _print_compare(base, records[request_id], image_token_ids, args.threshold)


if __name__ == "__main__":
    main()
