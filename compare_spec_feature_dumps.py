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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.logger import logger


DEFAULT_DUMP_ROOT = "/export/home/weinan5/wangxiaohan/xllm-dump"


@dataclass(frozen=True)
class RecordKey:
    rank_key: int
    model: str
    stage: str
    layer: int
    point: str
    occurrence: int


@dataclass(frozen=True)
class RankPair:
    left_rank: int
    right_rank: int
    key: int


@dataclass
class DiffItem:
    key: RecordKey
    metric: str
    token_index: int
    left_value: float
    right_value: float
    abs_diff: float
    rel_diff: float
    left_record: dict[str, Any]
    right_record: dict[str, Any]


def _sanitize_path_component(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in {"_", "-", "."}:
            safe.append(ch)
        else:
            safe.append("_")
        if len(safe) >= 128:
            break
    return "".join(safe) or "empty"


def _find_request_dirs(dump_root: Path, request_id: str) -> list[Path]:
    prefix = f"request-{_sanitize_path_component(request_id)}-"
    matches = [path for path in dump_root.glob(prefix + "*") if path.is_dir()]
    verified = []
    for path in matches:
        request_id_file = path / "request_id.txt"
        if not request_id_file.exists():
            verified.append(path)
            continue
        try:
            if request_id_file.read_text().strip() == request_id:
                verified.append(path)
        except OSError:
            logger.warning(f"failed to read {request_id_file}")
    if verified:
        return sorted(verified)

    fallback = []
    for path in dump_root.glob("request-*"):
        if not path.is_dir():
            continue
        request_id_file = path / "request_id.txt"
        if not request_id_file.exists():
            continue
        try:
            if request_id_file.read_text().strip() == request_id:
                fallback.append(path)
        except OSError:
            logger.warning(f"failed to read {request_id_file}")
    return sorted(fallback)


def _load_request_records(dump_root: Path, request_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    request_dirs = _find_request_dirs(dump_root, request_id)
    if not request_dirs:
        raise FileNotFoundError(f"no dump directory found for request_id={request_id}")

    for request_dir in request_dirs:
        for jsonl_path in sorted(request_dir.glob("features_rank*_pid*.jsonl")):
            try:
                with jsonl_path.open("r", encoding="utf-8") as handle:
                    for line_no, line in enumerate(handle, start=1):
                        text = line.strip()
                        if not text:
                            continue
                        try:
                            record = json.loads(text)
                        except json.JSONDecodeError as exc:
                            logger.warning(
                                f"skip malformed JSONL {jsonl_path}:{line_no}: {exc}"
                            )
                            continue
                        record["_file"] = str(jsonl_path)
                        record["_line"] = line_no
                        records.append(record)
            except OSError:
                logger.exception(f"failed to read {jsonl_path}")
    return records


def _rank_value(record: dict[str, Any]) -> int:
    value = record.get("rank", -1)
    return int(value) if isinstance(value, int) else -1


def _rank_key(rank: int, rank_mod: int | None) -> int:
    if rank_mod is not None and rank_mod > 0 and rank >= 0:
        return rank % rank_mod
    return rank


def _select_rank_pairs(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    rank_mod: int | None,
    all_ranks: bool,
) -> list[RankPair]:
    left_by_key: dict[int, int] = {}
    right_by_key: dict[int, int] = {}
    for record in left_records:
        rank = _rank_value(record)
        left_by_key.setdefault(_rank_key(rank, rank_mod), rank)
    for record in right_records:
        rank = _rank_value(record)
        right_by_key.setdefault(_rank_key(rank, rank_mod), rank)

    common_keys = sorted(set(left_by_key) & set(right_by_key))
    if common_keys:
        keys = common_keys if all_ranks else common_keys[:1]
        return [
            RankPair(left_rank=left_by_key[key], right_rank=right_by_key[key], key=key)
            for key in keys
        ]

    left_ranks = sorted({_rank_value(record) for record in left_records})
    right_ranks = sorted({_rank_value(record) for record in right_records})
    count = min(len(left_ranks), len(right_ranks))
    if count == 0:
        return []
    indices = range(count) if all_ranks else range(1)
    return [
        RankPair(left_rank=left_ranks[index], right_rank=right_ranks[index], key=index)
        for index in indices
    ]


def _stage_order(stage: str) -> int:
    return {
        "prefill": 0,
        "mixed": 1,
        "decode": 2,
        "empty": 3,
    }.get(stage, 9)


def _phase_order(stage: str, model: str) -> int:
    if stage == "prefill" and model == "target":
        return 0
    if stage == "prefill" and model == "draft":
        return 1
    if stage == "mixed" and model == "target":
        return 2
    if stage == "mixed" and model == "draft":
        return 3
    if stage == "decode" and model == "draft":
        return 4
    if stage == "decode" and model == "target":
        return 5
    return 20 + _stage_order(stage)


def _point_order(point: str) -> int:
    return {
        "model_input_hidden": 0,
        "layer_input_hidden": 1,
        "layer_kv_before": 2,
        "layer_output_hidden": 3,
        "layer_kv_after": 4,
        "model_output_hidden": 5,
    }.get(point, 9)


def _record_sort_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(record.get("batch_id", 0)),
        int(record.get("event_index", 0)),
        int(record.get("_line", 0)),
    )


def _group_records(
    records: list[dict[str, Any]],
    rank_pair_rank: int,
    rank_pair_key: int,
    rank_mod: int | None,
) -> dict[RecordKey, dict[str, Any]]:
    filtered = [
        record
        for record in records
        if _rank_value(record) == rank_pair_rank
        or _rank_key(_rank_value(record), rank_mod) == rank_pair_key
    ]
    buckets: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for record in filtered:
        if bool(record.get("empty_dp_request", False)):
            continue
        key = (
            str(record.get("model", "")),
            str(record.get("stage", "")),
            int(record.get("layer", -1)),
            str(record.get("point", "")),
        )
        buckets.setdefault(key, []).append(record)

    grouped: dict[RecordKey, dict[str, Any]] = {}
    for base_key, bucket in buckets.items():
        bucket.sort(key=_record_sort_key)
        for occurrence, record in enumerate(bucket):
            grouped[
                RecordKey(
                    rank_key=rank_pair_key,
                    model=base_key[0],
                    stage=base_key[1],
                    layer=base_key[2],
                    point=base_key[3],
                    occurrence=occurrence,
                )
            ] = record
    return grouped


def _ordered_keys(keys: set[RecordKey]) -> list[RecordKey]:
    return sorted(
        keys,
        key=lambda key: (
            key.rank_key,
            _phase_order(key.stage, key.model),
            key.layer if key.layer >= 0 else -1,
            _point_order(key.point),
            key.occurrence,
        ),
    )


def _numeric_list(record: dict[str, Any], name: str) -> list[float | None]:
    value = record.get(name)
    if not isinstance(value, list):
        return []
    result: list[float | None] = []
    for item in value:
        if isinstance(item, (int, float)) and math.isfinite(float(item)):
            result.append(float(item))
        else:
            result.append(None)
    return result


def _metrics_for_record(record: dict[str, Any]) -> list[str]:
    if record.get("tensor_kind") == "kv":
        return ["k_l2", "v_l2"]
    return ["token_l2"]


def _relative_diff(left: float, right: float) -> float:
    denom = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denom


def _compare_metric(
    key: RecordKey,
    metric: str,
    left_record: dict[str, Any],
    right_record: dict[str, Any],
) -> DiffItem | None:
    left_values = _numeric_list(left_record, metric)
    right_values = _numeric_list(right_record, metric)
    count = min(len(left_values), len(right_values))
    best: DiffItem | None = None
    for index in range(count):
        left_value = left_values[index]
        right_value = right_values[index]
        if left_value is None or right_value is None:
            continue
        abs_diff = abs(left_value - right_value)
        rel_diff = _relative_diff(left_value, right_value)
        item = DiffItem(
            key=key,
            metric=metric,
            token_index=index,
            left_value=left_value,
            right_value=right_value,
            abs_diff=abs_diff,
            rel_diff=rel_diff,
            left_record=left_record,
            right_record=right_record,
        )
        if best is None or (item.rel_diff, item.abs_diff) > (
            best.rel_diff,
            best.abs_diff,
        ):
            best = item
    if len(left_values) != len(right_values):
        item = DiffItem(
            key=key,
            metric=f"{metric}.len",
            token_index=-1,
            left_value=float(len(left_values)),
            right_value=float(len(right_values)),
            abs_diff=float(abs(len(left_values) - len(right_values))),
            rel_diff=_relative_diff(float(len(left_values)), float(len(right_values))),
            left_record=left_record,
            right_record=right_record,
        )
        if best is None or (item.rel_diff, item.abs_diff) > (
            best.rel_diff,
            best.abs_diff,
        ):
            best = item
    return best


def _is_large_diff(item: DiffItem, abs_threshold: float, rel_threshold: float) -> bool:
    return item.abs_diff >= abs_threshold or item.rel_diff >= rel_threshold


def _compare_records(
    key: RecordKey,
    left_record: dict[str, Any],
    right_record: dict[str, Any],
) -> list[DiffItem]:
    metrics = sorted(set(_metrics_for_record(left_record)) | set(_metrics_for_record(right_record)))
    diffs = []
    for metric in metrics:
        item = _compare_metric(key, metric, left_record, right_record)
        if item is not None:
            diffs.append(item)
    diffs.sort(key=lambda item: (item.rel_diff, item.abs_diff), reverse=True)
    return diffs


def _format_key(key: RecordKey) -> str:
    layer = "model" if key.layer < 0 else str(key.layer)
    return (
        f"rank_key={key.rank_key} {key.model}/{key.stage} "
        f"layer={layer} point={key.point} occurrence={key.occurrence}"
    )


def _format_diff(item: DiffItem) -> str:
    return (
        f"{_format_key(item.key)} metric={item.metric} token={item.token_index} "
        f"good={item.left_value:.9g} bad={item.right_value:.9g} "
        f"abs={item.abs_diff:.9g} rel={item.rel_diff:.9g}"
    )


def _print_record_context(prefix: str, record: dict[str, Any]) -> None:
    print(
        f"{prefix}: q_seq_len={record.get('q_seq_len')} "
        f"kv_seq_len={record.get('kv_seq_len')} file={record.get('_file')}"
    )


def _list_requests(dump_root: Path) -> None:
    rows = []
    for request_dir in sorted(dump_root.glob("request-*")):
        if not request_dir.is_dir():
            continue
        request_id_file = request_dir / "request_id.txt"
        if not request_id_file.exists():
            continue
        try:
            request_id = request_id_file.read_text().strip()
        except OSError:
            continue
        feature_count = len(list(request_dir.glob("features_rank*_pid*.jsonl")))
        rows.append((request_id, feature_count, request_dir))
    for request_id, feature_count, request_dir in rows:
        print(f"{request_id}\tfeature_files={feature_count}\tdir={request_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare xLLM speculative feature L2 dumps for two request ids."
    )
    parser.add_argument("good_request_id", nargs="?")
    parser.add_argument("bad_request_id", nargs="?")
    parser.add_argument("--dump-root", default=DEFAULT_DUMP_ROOT)
    parser.add_argument(
        "--rank-mod",
        type=int,
        default=None,
        help="Compare ranks by rank %% N, useful when two DP groups use different global ranks.",
    )
    parser.add_argument(
        "--all-ranks",
        action="store_true",
        help="Compare all aligned ranks instead of only the first aligned rank.",
    )
    parser.add_argument("--abs-threshold", type=float, default=1e-5)
    parser.add_argument("--rel-threshold", type=float, default=1e-5)
    parser.add_argument("--max-summary", type=int, default=80)
    parser.add_argument("--max-details", type=int, default=20)
    parser.add_argument("--list-requests", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dump_root = Path(args.dump_root)
    if args.list_requests:
        _list_requests(dump_root)
        return
    if not args.good_request_id or not args.bad_request_id:
        raise SystemExit("good_request_id and bad_request_id are required")

    left_records = _load_request_records(dump_root, args.good_request_id)
    right_records = _load_request_records(dump_root, args.bad_request_id)
    rank_pairs = _select_rank_pairs(
        left_records, right_records, args.rank_mod, args.all_ranks
    )
    if not rank_pairs:
        raise SystemExit("no ranks found to compare")

    all_diffs: list[DiffItem] = []
    first_large: DiffItem | None = None
    missing = []

    print("Rank alignment")
    for pair in rank_pairs:
        print(f"  key={pair.key} good_rank={pair.left_rank} bad_rank={pair.right_rank}")

    for pair in rank_pairs:
        left_grouped = _group_records(left_records, pair.left_rank, pair.key, args.rank_mod)
        right_grouped = _group_records(right_records, pair.right_rank, pair.key, args.rank_mod)
        common_keys = set(left_grouped) & set(right_grouped)
        for key in _ordered_keys(set(left_grouped) | set(right_grouped)):
            if key not in common_keys:
                missing.append(key)
                continue
            diffs = _compare_records(key, left_grouped[key], right_grouped[key])
            if not diffs:
                continue
            best = diffs[0]
            all_diffs.append(best)
            if first_large is None and _is_large_diff(
                best, args.abs_threshold, args.rel_threshold
            ):
                first_large = best

    all_diffs.sort(
        key=lambda item: (
            item.key.rank_key,
            _phase_order(item.key.stage, item.key.model),
            item.key.layer if item.key.layer >= 0 else -1,
            _point_order(item.key.point),
            item.key.occurrence,
        )
    )

    print("\nSummary first large difference")
    if first_large is None:
        print(
            "  no diff exceeded thresholds "
            f"abs>={args.abs_threshold:g} or rel>={args.rel_threshold:g}"
        )
        if all_diffs:
            print("  max observed diff:")
            print(f"  {_format_diff(max(all_diffs, key=lambda item: (item.rel_diff, item.abs_diff)))}")
    else:
        print(f"  {_format_diff(first_large)}")
        _print_record_context("  good", first_large.left_record)
        _print_record_context("  bad", first_large.right_record)

    print("\nStage summary")
    for item in all_diffs[: max(args.max_summary, 0)]:
        print(f"  {_format_diff(item)}")
    if len(all_diffs) > args.max_summary:
        print(f"  ... {len(all_diffs) - args.max_summary} more summary rows")

    if first_large is not None:
        first_key = first_large.key
        print("\nDetails around first difference")
        detail_items = [
            item
            for item in all_diffs
            if item.key.model == first_key.model
            and item.key.stage == first_key.stage
            and item.key.layer == first_key.layer
            and item.key.occurrence == first_key.occurrence
        ]
        detail_items.sort(key=lambda item: (item.rel_diff, item.abs_diff), reverse=True)
        for item in detail_items[: max(args.max_details, 0)]:
            print(f"  {_format_diff(item)}")

    if missing:
        print("\nMissing comparable records")
        for key in _ordered_keys(set(missing))[:20]:
            print(f"  {_format_key(key)}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more missing keys")


if __name__ == "__main__":
    main()
