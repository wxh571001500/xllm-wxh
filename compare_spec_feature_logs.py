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
class FeatureKey:
    rank_key: int
    model: str
    stage: str
    layer: int
    point: str
    metric: str
    occurrence: int


@dataclass
class FeatureRecord:
    key: FeatureKey
    request_id: str
    rank: int
    event_index: int
    event_dir: Path
    values: list[float | None]
    positions: list[int | None]
    meta: dict[str, Any]


@dataclass
class DiffItem:
    key: FeatureKey
    index: int
    left_value: float
    right_value: float
    abs_diff: float
    rel_diff: float
    left_pos: int | None
    right_pos: int | None
    left_record: FeatureRecord
    right_record: FeatureRecord


@dataclass
class TensorEventRecord:
    key: FeatureKey
    request_id: str
    rank: int
    event_index: int
    event_dir: Path
    tensor_path: Path
    meta: dict[str, Any]
    request: dict[str, Any]


@dataclass
class CosineReport:
    key: FeatureKey
    count: int
    valid_count: int
    mean_cos: float | None
    min_cos: float | None
    first_bad_index: int | None
    first_bad_cos: float | None
    zero_pair_count: int
    shape_mismatch_count: int
    token_cos: list[tuple[int, float | None]]
    left_record: TensorEventRecord
    right_record: TensorEventRecord


def _rank_key(rank: int, rank_mod: int | None) -> int:
    if rank_mod is not None and rank_mod > 0 and rank >= 0:
        return rank % rank_mod
    return rank


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
    return 20


def _point_order(point: str) -> int:
    return {
        "model_input_hidden": 0,
        "layer_input_hidden": 1,
        "layer_kv_before": 2,
        "layer_output_hidden": 3,
        "layer_kv_after": 4,
        "model_output_hidden": 5,
    }.get(point, 9)


def _record_sort_key(record: FeatureRecord) -> tuple[int, int, int, str, str]:
    return (
        _phase_order(record.meta.get("stage", ""), record.meta.get("model", "")),
        int(record.meta.get("layer", -1)),
        _point_order(record.meta.get("point", "")),
        record.key.metric,
        str(record.event_dir),
    )


def _ordered_keys(keys: set[FeatureKey]) -> list[FeatureKey]:
    return sorted(
        keys,
        key=lambda key: (
            key.rank_key,
            _phase_order(key.stage, key.model),
            key.layer if key.layer >= 0 else -1,
            _point_order(key.point),
            key.metric,
            key.occurrence,
        ),
    )


def _relative_diff(left: float, right: float) -> float:
    denom = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denom


def _cosine_similarity(left, right) -> float | None:
    if left is None or right is None:
        return None
    if left.numel() == 0 or right.numel() == 0:
        return None
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    if left_flat.numel() != right_flat.numel():
        return None
    denom = float(left_flat.norm().item() * right_flat.norm().item())
    if denom <= 1e-12:
        return None
    cos = float(left_flat.dot(right_flat).item()) / denom
    if math.isfinite(cos):
        return cos
    return None


def _load_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required to analyze .pt dumps. Run this script in the "
            "same Python environment that has torch installed."
        ) from exc
    return torch


def _is_torchscript_archive(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        return False
    return any(name.startswith("code/") for name in names)


def _extract_tensor_like(value):
    torch = _load_torch()
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _extract_tensor_like(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_tensor_like(item)
            if tensor is not None:
                return tensor
        return None
    if hasattr(value, "forward"):
        try:
            output = value()
        except TypeError:
            try:
                output = value.forward()
            except Exception:  # noqa: BLE001
                output = None
        except Exception:  # noqa: BLE001
            output = None
        if output is not None:
            tensor = _extract_tensor_like(output)
            if tensor is not None:
                return tensor
    if hasattr(value, "state_dict"):
        try:
            state_dict = value.state_dict()
        except Exception:  # noqa: BLE001
            state_dict = None
        if state_dict is not None:
            for item in state_dict.values():
                tensor = _extract_tensor_like(item)
                if tensor is not None:
                    return tensor
    return None


def _as_float_tensor(path: Path):
    torch = _load_torch()
    try:
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
                    loaded = torch.load(
                        path, map_location="cpu", weights_only=False
                    )
                except TypeError:
                    loaded = torch.load(path, map_location="cpu")
        tensor = _extract_tensor_like(loaded)
        if tensor is None:
            raise TypeError(
                f"unable to extract tensor from {path} (type={type(loaded)!r})"
            )
        return tensor.float().contiguous()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to load tensor {path}: {exc}") from exc


def _row_l2_values(tensor, rows: list[int]) -> list[float | None]:
    if tensor is None or tensor.numel() == 0:
        return [None for _ in rows]
    if tensor.dim() == 1:
        tensor = tensor.reshape(1, -1)
    else:
        tensor = tensor.reshape(tensor.shape[0], -1)
    values: list[float | None] = []
    row_count = tensor.shape[0]
    for row in rows:
        if row is None or row < 0 or row >= row_count:
            values.append(None)
            continue
        value = float(tensor[row].pow(2).sum().sqrt().item())
        if math.isfinite(value):
            values.append(value)
        else:
            values.append(None)
    return values


def _block_index_map(selected_blocks: list[int]) -> dict[int, int]:
    return {block: index for index, block in enumerate(selected_blocks)}


def _kv_l2_values(
    tensor,
    token_slots: list[int],
    selected_blocks: list[int],
    block_size: int,
) -> list[float | None]:
    if tensor is None or tensor.numel() == 0 or block_size <= 0:
        return [None for _ in token_slots]
    block_to_row = _block_index_map(selected_blocks)
    values: list[float | None] = []
    for slot in token_slots:
        if slot is None or slot < 0:
            values.append(None)
            continue
        block = slot // block_size
        offset = slot % block_size
        row = block_to_row.get(block)
        if row is None or row < 0 or row >= tensor.shape[0]:
            values.append(None)
            continue
        if tensor.dim() < 2 or offset < 0 or offset >= tensor.shape[1]:
            values.append(None)
            continue
        token_tensor = tensor[row, offset].reshape(-1)
        value = float(token_tensor.pow(2).sum().sqrt().item())
        values.append(value if math.isfinite(value) else None)
    return values


def _hidden_token_tensors(tensor, rows: list[int]) -> list[Any]:
    if tensor is None or tensor.numel() == 0:
        return [None for _ in rows]
    if tensor.dim() == 1:
        tensor = tensor.reshape(1, -1)
    else:
        tensor = tensor.reshape(tensor.shape[0], -1)
    result: list[Any] = []
    row_count = tensor.shape[0]
    for row in rows:
        if row is None or row < 0 or row >= row_count:
            result.append(None)
            continue
        result.append(tensor[row])
    return result


def _kv_token_tensors(
    tensor,
    token_slots: list[int],
    selected_blocks: list[int],
    block_size: int,
) -> list[Any]:
    if tensor is None or tensor.numel() == 0 or block_size <= 0:
        return [None for _ in token_slots]
    block_to_row = _block_index_map(selected_blocks)
    result: list[Any] = []
    for slot in token_slots:
        if slot is None or slot < 0:
            result.append(None)
            continue
        block = slot // block_size
        offset = slot % block_size
        row = block_to_row.get(block)
        if row is None or row < 0 or row >= tensor.shape[0]:
            result.append(None)
            continue
        if tensor.dim() < 2 or offset < 0 or offset >= tensor.shape[1]:
            result.append(None)
            continue
        result.append(tensor[row, offset].reshape(-1))
    return result


def _iter_event_meta(dump_roots: list[str]) -> list[tuple[Path, dict[str, Any]]]:
    events: list[tuple[Path, dict[str, Any]]] = []
    for root_text in dump_roots:
        root = Path(root_text)
        if root.is_file() and root.name == "meta.json":
            paths = [root]
        else:
            paths = sorted(root.rglob("meta.json"))
        for path in paths:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    meta = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"skip unreadable meta {path}: {exc}")
                continue
            events.append((path.parent, meta))
    return events


def _request_ids_for_meta(meta: dict[str, Any]) -> list[str]:
    return [
        str(request.get("request_id", ""))
        for request in meta.get("requests", [])
        if request.get("request_id")
    ]


def _load_records(
    dump_roots: list[str],
    request_filter: set[str] | None,
    rank_mod: int | None,
) -> list[FeatureRecord]:
    records: list[FeatureRecord] = []
    for event_dir, meta in _iter_event_meta(dump_roots):
        if meta.get("empty_dp_request"):
            continue
        requests = meta.get("requests", [])
        if not requests:
            continue
        wanted_requests = [
            request
            for request in requests
            if request_filter is None
            or str(request.get("request_id", "")) in request_filter
        ]
        if not wanted_requests:
            continue

        rank = int(meta.get("rank", -1))
        key_prefix = FeatureKey(
            rank_key=_rank_key(rank, rank_mod),
            model=str(meta.get("model", "")),
            stage=str(meta.get("stage", "")),
            layer=int(meta.get("layer", -1)),
            point=str(meta.get("point", "")),
            metric="",
            occurrence=0,
        )
        tensor_kind = str(meta.get("tensor_kind", ""))
        event_index = int(meta.get("event_index", -1))
        if tensor_kind == "hidden":
            tensor = None
            hidden_file = str(meta.get("hidden_file", ""))
            if hidden_file:
                tensor = _as_float_tensor(event_dir / hidden_file)
            for request in wanted_requests:
                request_id = str(request.get("request_id", ""))
                rows = [int(row) for row in request.get("token_rows", [])]
                values = _row_l2_values(tensor, rows) if tensor is not None else []
                key = FeatureKey(
                    key_prefix.rank_key,
                    key_prefix.model,
                    key_prefix.stage,
                    key_prefix.layer,
                    key_prefix.point,
                    "hidden_l2",
                    0,
                )
                records.append(
                    FeatureRecord(
                        key=key,
                        request_id=request_id,
                        rank=rank,
                        event_index=event_index,
                        event_dir=event_dir,
                        values=values,
                        positions=list(range(len(values))),
                        meta=meta,
                    )
                )
        elif tensor_kind == "kv":
            block_size = int(meta.get("block_size", 0))
            selected_blocks = [
                int(block) for block in meta.get("selected_blocks", [])
            ]
            k_tensor = None
            v_tensor = None
            k_file = str(meta.get("k_file", ""))
            v_file = str(meta.get("v_file", ""))
            if k_file:
                k_tensor = _as_float_tensor(event_dir / k_file)
            if v_file:
                v_tensor = _as_float_tensor(event_dir / v_file)
            for request in wanted_requests:
                request_id = str(request.get("request_id", ""))
                token_slots = [
                    int(slot) for slot in request.get("token_slots", [])
                ]
                for metric, tensor in (("k_l2", k_tensor), ("v_l2", v_tensor)):
                    values = (
                        _kv_l2_values(
                            tensor, token_slots, selected_blocks, block_size
                        )
                        if tensor is not None
                        else []
                    )
                    key = FeatureKey(
                        key_prefix.rank_key,
                        key_prefix.model,
                        key_prefix.stage,
                        key_prefix.layer,
                        key_prefix.point,
                        metric,
                        0,
                    )
                    records.append(
                        FeatureRecord(
                            key=key,
                            request_id=request_id,
                            rank=rank,
                            event_index=event_index,
                            event_dir=event_dir,
                            values=values,
                            positions=list(range(len(values))),
                            meta=meta,
                        )
                    )
    return _assign_occurrences(records)


def _load_tensor_events(
    dump_roots: list[str],
    request_filter: set[str] | None,
    rank_mod: int | None,
) -> list[TensorEventRecord]:
    events: list[TensorEventRecord] = []
    for event_dir, meta in _iter_event_meta(dump_roots):
        if meta.get("empty_dp_request"):
            continue
        requests = meta.get("requests", [])
        if not requests:
            continue
        wanted_requests = [
            request
            for request in requests
            if request_filter is None
            or str(request.get("request_id", "")) in request_filter
        ]
        if not wanted_requests:
            continue
        rank = int(meta.get("rank", -1))
        key_prefix = FeatureKey(
            rank_key=_rank_key(rank, rank_mod),
            model=str(meta.get("model", "")),
            stage=str(meta.get("stage", "")),
            layer=int(meta.get("layer", -1)),
            point=str(meta.get("point", "")),
            metric="",
            occurrence=0,
        )
        tensor_kind = str(meta.get("tensor_kind", ""))
        event_index = int(meta.get("event_index", -1))
        if tensor_kind == "hidden":
            hidden_file = str(meta.get("hidden_file", ""))
            if not hidden_file:
                continue
            for request in wanted_requests:
                request_id = str(request.get("request_id", ""))
                events.append(
                    TensorEventRecord(
                        key=FeatureKey(
                            key_prefix.rank_key,
                            key_prefix.model,
                            key_prefix.stage,
                            key_prefix.layer,
                            key_prefix.point,
                            "hidden",
                            0,
                        ),
                        request_id=request_id,
                        rank=rank,
                        event_index=event_index,
                        event_dir=event_dir,
                        tensor_path=event_dir / hidden_file,
                        meta=meta,
                        request=request,
                    )
                )
        elif tensor_kind == "kv":
            k_file = str(meta.get("k_file", ""))
            v_file = str(meta.get("v_file", ""))
            if not k_file and not v_file:
                continue
            for request in wanted_requests:
                request_id = str(request.get("request_id", ""))
                for metric, tensor_file in (("k", k_file), ("v", v_file)):
                    if not tensor_file:
                        continue
                    events.append(
                        TensorEventRecord(
                            key=FeatureKey(
                                key_prefix.rank_key,
                                key_prefix.model,
                                key_prefix.stage,
                                key_prefix.layer,
                                key_prefix.point,
                                metric,
                                0,
                            ),
                            request_id=request_id,
                            rank=rank,
                            event_index=event_index,
                            event_dir=event_dir,
                            tensor_path=event_dir / tensor_file,
                            meta=meta,
                            request=request,
                        )
                    )
    return _assign_tensor_event_occurrences(events)


def _assign_tensor_event_occurrences(
    records: list[TensorEventRecord],
) -> list[TensorEventRecord]:
    grouped: dict[
        tuple[str, int, str, str, int, str, str], list[TensorEventRecord]
    ] = {}
    for record in records:
        base = (
            record.request_id,
            record.key.rank_key,
            record.key.model,
            record.key.stage,
            record.key.layer,
            record.key.point,
            record.key.metric,
        )
        grouped.setdefault(base, []).append(record)

    assigned: list[TensorEventRecord] = []
    for bucket in grouped.values():
        bucket.sort(
            key=lambda record: (
                _phase_order(
                    record.meta.get("stage", ""), record.meta.get("model", "")
                ),
                int(record.meta.get("layer", -1)),
                _point_order(record.meta.get("point", "")),
                record.key.metric,
                str(record.event_dir),
            )
        )
        for occurrence, record in enumerate(bucket):
            record.key = FeatureKey(
                record.key.rank_key,
                record.key.model,
                record.key.stage,
                record.key.layer,
                record.key.point,
                record.key.metric,
                occurrence,
            )
            assigned.append(record)
    return assigned


def _assign_occurrences(records: list[FeatureRecord]) -> list[FeatureRecord]:
    grouped: dict[tuple[str, int, str, str, int, str, str], list[FeatureRecord]] = {}
    for record in records:
        base = (
            record.request_id,
            record.key.rank_key,
            record.key.model,
            record.key.stage,
            record.key.layer,
            record.key.point,
            record.key.metric,
        )
        grouped.setdefault(base, []).append(record)

    assigned: list[FeatureRecord] = []
    for bucket in grouped.values():
        bucket.sort(key=_record_sort_key)
        for occurrence, record in enumerate(bucket):
            record.key = FeatureKey(
                record.key.rank_key,
                record.key.model,
                record.key.stage,
                record.key.layer,
                record.key.point,
                record.key.metric,
                occurrence,
            )
            assigned.append(record)
    return assigned


def _list_requests(dump_roots: list[str]) -> None:
    by_request: dict[str, set[int]] = {}
    counts: dict[str, int] = {}
    for _, meta in _iter_event_meta(dump_roots):
        rank = int(meta.get("rank", -1))
        for request_id in _request_ids_for_meta(meta):
            by_request.setdefault(request_id, set()).add(rank)
            counts[request_id] = counts.get(request_id, 0) + 1
    for request_id in sorted(by_request):
        print(
            f"{request_id}\tevents={counts[request_id]}"
            f"\tranks={sorted(by_request[request_id])}"
        )


def _select_rank_pairs(
    left_records: list[FeatureRecord],
    right_records: list[FeatureRecord],
    all_ranks: bool,
) -> list[tuple[int, int, int]]:
    left_by_key: dict[int, int] = {}
    right_by_key: dict[int, int] = {}
    for record in left_records:
        left_by_key.setdefault(record.key.rank_key, record.rank)
    for record in right_records:
        right_by_key.setdefault(record.key.rank_key, record.rank)

    common = sorted(set(left_by_key) & set(right_by_key))
    selected = common if all_ranks else common[:1]
    return [(left_by_key[key], right_by_key[key], key) for key in selected]


def _group_records(
    records: list[FeatureRecord],
    rank: int,
    rank_key: int,
) -> dict[FeatureKey, FeatureRecord]:
    result: dict[FeatureKey, FeatureRecord] = {}
    for record in records:
        if record.rank != rank and record.key.rank_key != rank_key:
            continue
        result[record.key] = record
    return result


def _group_tensor_events(
    records: list[TensorEventRecord],
    rank: int,
    rank_key: int,
) -> dict[FeatureKey, TensorEventRecord]:
    result: dict[FeatureKey, TensorEventRecord] = {}
    for record in records:
        if record.rank != rank and record.key.rank_key != rank_key:
            continue
        result[record.key] = record
    return result


def _cos_key_filter(key: FeatureKey, scope: str) -> bool:
    if scope == "prefill" and key.stage != "prefill":
        return False
    if key.model == "target":
        if key.layer < 0:
            return key.metric == "hidden" and key.point in {
                "model_input_hidden",
                "model_output_hidden",
            }
        return key.layer >= 0 and key.point in {
            "layer_input_hidden",
            "layer_output_hidden",
            "layer_kv_before",
            "layer_kv_after",
        }
    if key.model == "draft":
        if key.layer < 0:
            return key.metric == "hidden" and key.point in {
                "model_input_hidden",
                "model_output_hidden",
            }
        return key.layer >= 0 and key.point in {
            "layer_input_hidden",
            "layer_output_hidden",
            "layer_kv_before",
            "layer_kv_after",
        }
    return False


def _request_token_ids(request: dict[str, Any], count: int) -> list[int | None]:
    token_ids = [int(token_id) for token_id in request.get("token_ids", [])]
    if count <= 0:
        return token_ids
    if len(token_ids) >= count:
        return token_ids[:count]
    return token_ids + [None] * (count - len(token_ids))


def _token_tensors_for_event(record: TensorEventRecord) -> list[Any]:
    tensor = _as_float_tensor(record.tensor_path)
    if record.key.metric == "hidden":
        rows = [int(row) for row in record.request.get("token_rows", [])]
        return _hidden_token_tensors(tensor, rows)
    token_slots = [int(slot) for slot in record.request.get("token_slots", [])]
    selected_blocks = [int(block) for block in record.meta.get("selected_blocks", [])]
    block_size = int(record.meta.get("block_size", 0))
    return _kv_token_tensors(tensor, token_slots, selected_blocks, block_size)


def _compare_cosine_record(
    key: FeatureKey,
    left_record: TensorEventRecord,
    right_record: TensorEventRecord,
    max_tokens: int,
    bad_threshold: float,
) -> CosineReport:
    left_tensors = _token_tensors_for_event(left_record)
    right_tensors = _token_tensors_for_event(right_record)
    count = min(len(left_tensors), len(right_tensors))
    token_cos: list[tuple[int, float | None]] = []
    values: list[float] = []
    zero_pair_count = 0
    shape_mismatch_count = 0
    first_bad_index: int | None = None
    first_bad_cos: float | None = None
    for index in range(count):
        left_tensor = left_tensors[index]
        right_tensor = right_tensors[index]
        if left_tensor is None or right_tensor is None:
            cos = None
        elif left_tensor.numel() != right_tensor.numel():
            cos = None
            shape_mismatch_count += 1
        elif left_tensor.norm().item() <= 1e-12 or right_tensor.norm().item() <= 1e-12:
            cos = None
            zero_pair_count += 1
        else:
            cos = _cosine_similarity(left_tensor, right_tensor)
        if max_tokens < 0 or index < max_tokens:
            token_cos.append((index, cos))
        if cos is not None:
            values.append(cos)
            if first_bad_index is None and cos < bad_threshold:
                first_bad_index = index
                first_bad_cos = cos
    mean_cos = sum(values) / len(values) if values else None
    min_cos = min(values) if values else None
    return CosineReport(
        key=key,
        count=count,
        valid_count=len(values),
        mean_cos=mean_cos,
        min_cos=min_cos,
        first_bad_index=first_bad_index,
        first_bad_cos=first_bad_cos,
        zero_pair_count=zero_pair_count,
        shape_mismatch_count=shape_mismatch_count,
        token_cos=token_cos,
        left_record=left_record,
        right_record=right_record,
    )


def _compare_record(
    key: FeatureKey,
    left_record: FeatureRecord,
    right_record: FeatureRecord,
) -> DiffItem | None:
    count = min(len(left_record.values), len(right_record.values))
    best: DiffItem | None = None
    for index in range(count):
        left_value = left_record.values[index]
        right_value = right_record.values[index]
        if left_value is None or right_value is None:
            continue
        abs_diff = abs(left_value - right_value)
        rel_diff = _relative_diff(left_value, right_value)
        item = DiffItem(
            key=key,
            index=index,
            left_value=left_value,
            right_value=right_value,
            abs_diff=abs_diff,
            rel_diff=rel_diff,
            left_pos=left_record.positions[index],
            right_pos=right_record.positions[index],
            left_record=left_record,
            right_record=right_record,
        )
        if best is None or (item.rel_diff, item.abs_diff) > (
            best.rel_diff,
            best.abs_diff,
        ):
            best = item
    if len(left_record.values) != len(right_record.values):
        item = DiffItem(
            key=key,
            index=-1,
            left_value=float(len(left_record.values)),
            right_value=float(len(right_record.values)),
            abs_diff=float(abs(len(left_record.values) - len(right_record.values))),
            rel_diff=_relative_diff(
                float(len(left_record.values)), float(len(right_record.values))
            ),
            left_pos=None,
            right_pos=None,
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


def _format_key(key: FeatureKey) -> str:
    layer = "model" if key.layer < 0 else str(key.layer)
    return (
        f"rank_key={key.rank_key} {key.model}/{key.stage} "
        f"layer={layer} point={key.point} metric={key.metric} "
        f"occurrence={key.occurrence}"
    )


def _format_diff(item: DiffItem) -> str:
    return (
        f"{_format_key(item.key)} token_index={item.index} "
        f"good_pos={item.left_pos} bad_pos={item.right_pos} "
        f"good={item.left_value:.9g} bad={item.right_value:.9g} "
        f"abs={item.abs_diff:.9g} rel={item.rel_diff:.9g}"
    )


def _print_context(prefix: str, record: FeatureRecord) -> None:
    print(
        f"{prefix}: rank={record.rank} event_index={record.event_index} "
        f"dir={record.event_dir}"
    )


def _format_cos(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.9g}"


def _format_short_list(values: list[Any], limit: int = 8) -> str:
    if len(values) <= limit * 2:
        return str(values)
    head = ", ".join(str(value) for value in values[:limit])
    tail = ", ".join(str(value) for value in values[-limit:])
    return f"[{head}, ... {tail}]"


def _same_text(left: Any, right: Any) -> str:
    return "same" if left == right else "DIFF"


def _print_cosine_reports(
    reports: list[CosineReport],
    max_tokens: int,
    summary_only: bool,
    only_bad: bool,
) -> None:
    print("\nCosine summary")
    for report in reports:
        is_bad = (
            report.first_bad_index is not None
            or report.shape_mismatch_count > 0
            or report.valid_count != report.count
        )
        if only_bad and not is_bad:
            continue
        left_token_ids = _request_token_ids(
            report.left_record.request, len(report.token_cos)
        )
        print(
            f"  {_format_key(report.key)} tokens={report.count} "
            f"valid={report.valid_count} mean_cos={_format_cos(report.mean_cos)} "
            f"min_cos={_format_cos(report.min_cos)} "
            f"first_cos_below_threshold={report.first_bad_index} "
            f"first_bad_cos={_format_cos(report.first_bad_cos)} "
            f"zero_pairs={report.zero_pair_count} "
            f"shape_mismatches={report.shape_mismatch_count}"
        )
        if summary_only:
            continue
        token_parts = [
            f"{index}/{left_token_ids[index]}:{_format_cos(cos)}"
            if index < len(left_token_ids)
            else f"{index}/NA:{_format_cos(cos)}"
            for index, cos in report.token_cos
        ]
        label = "all_tokens" if max_tokens < 0 else f"first_{max_tokens}_tokens"
        print(f"    {label} {' '.join(token_parts)}")
        print(
            f"    good rank={report.left_record.rank} "
            f"event={report.left_record.event_index} "
            f"dir={report.left_record.event_dir}"
        )
        print(
            f"    bad  rank={report.right_record.rank} "
            f"event={report.right_record.event_index} "
            f"dir={report.right_record.event_dir}"
        )


def _cosine_report_event_order(report: CosineReport) -> tuple[int, int, int, int]:
    return (
        _phase_order(report.key.stage, report.key.model),
        report.left_record.event_index,
        report.right_record.event_index,
        _point_order(report.key.point),
    )


def _print_first_bad_cosine_report(reports: list[CosineReport]) -> None:
    bad_reports = [
        report for report in reports if report.first_bad_index is not None
    ]
    if not bad_reports:
        print("\nNo cosine value below threshold.")
        return

    report = sorted(bad_reports, key=_cosine_report_event_order)[0]
    token_ids = _request_token_ids(
        report.left_record.request, report.first_bad_index + 1
    )
    token_id = (
        token_ids[report.first_bad_index]
        if report.first_bad_index < len(token_ids)
        else None
    )
    print("\nFirst cosine below threshold")
    print(f"  {_format_key(report.key)}")
    print(
        f"  token_index={report.first_bad_index} token_id={token_id} "
        f"cos={_format_cos(report.first_bad_cos)}"
    )
    print(
        f"  summary tokens={report.count} valid={report.valid_count} "
        f"mean_cos={_format_cos(report.mean_cos)} "
        f"min_cos={_format_cos(report.min_cos)}"
    )
    print(
        f"  good rank={report.left_record.rank} "
        f"event={report.left_record.event_index} "
        f"dir={report.left_record.event_dir}"
    )
    print(
        f"  bad  rank={report.right_record.rank} "
        f"event={report.right_record.event_index} "
        f"dir={report.right_record.event_dir}"
    )


def _print_bad_cosine_table(reports: list[CosineReport]) -> None:
    bad_reports = [
        report
        for report in reports
        if report.first_bad_index is not None
        or report.shape_mismatch_count > 0
        or report.valid_count != report.count
    ]
    if not bad_reports:
        print("\nNo cosine differences below threshold.")
        return

    print("\nBad cosine table")
    print(
        "  rank_key model stage layer point metric occ tokens valid "
        "mean_cos min_cos first_bad first_bad_cos zero_pairs shape_mismatches "
        "good_event bad_event"
    )
    for report in sorted(bad_reports, key=_cosine_report_event_order):
        layer = "model" if report.key.layer < 0 else str(report.key.layer)
        print(
            f"  {report.key.rank_key} {report.key.model} {report.key.stage} "
            f"{layer} {report.key.point} {report.key.metric} "
            f"{report.key.occurrence} {report.count} {report.valid_count} "
            f"{_format_cos(report.mean_cos)} {_format_cos(report.min_cos)} "
            f"{report.first_bad_index} {_format_cos(report.first_bad_cos)} "
            f"{report.zero_pair_count} {report.shape_mismatch_count} "
            f"{report.left_record.event_index} {report.right_record.event_index}"
        )


def _print_input_meta_for_pair(
    left_record: TensorEventRecord,
    right_record: TensorEventRecord,
) -> None:
    left_meta = left_record.meta
    right_meta = right_record.meta
    left_request = left_record.request
    right_request = right_record.request
    left_rows = [int(row) for row in left_request.get("token_rows", [])]
    right_rows = [int(row) for row in right_request.get("token_rows", [])]
    left_slots = [int(slot) for slot in left_request.get("token_slots", [])]
    right_slots = [int(slot) for slot in right_request.get("token_slots", [])]
    print(
        f"  {_format_key(left_record.key)}\n"
        f"    good rank={left_record.rank} event={left_record.event_index} "
        f"dir={left_record.event_dir}\n"
        f"    bad  rank={right_record.rank} event={right_record.event_index} "
        f"dir={right_record.event_dir}"
    )
    fields = [
        ("num_sequences", left_meta.get("num_sequences"), right_meta.get("num_sequences")),
        ("q_seq_lens", left_meta.get("q_seq_lens"), right_meta.get("q_seq_lens")),
        ("kv_seq_lens", left_meta.get("kv_seq_lens"), right_meta.get("kv_seq_lens")),
        ("request_index", left_request.get("request_index"), right_request.get("request_index")),
        ("q_seq_len", left_request.get("q_seq_len"), right_request.get("q_seq_len")),
        ("kv_seq_len", left_request.get("kv_seq_len"), right_request.get("kv_seq_len")),
        ("token_rows_len", len(left_rows), len(right_rows)),
        ("token_slots_len", len(left_slots), len(right_slots)),
    ]
    for name, left_value, right_value in fields:
        print(
            f"    {name}: good={left_value} bad={right_value} "
            f"{_same_text(left_value, right_value)}"
        )
    print(
        f"    token_rows: good={_format_short_list(left_rows)} "
        f"bad={_format_short_list(right_rows)} {_same_text(left_rows, right_rows)}"
    )
    print(
        f"    token_ids: good={_format_short_list(left_request.get('token_ids', []))} "
        f"bad={_format_short_list(right_request.get('token_ids', []))} "
        f"{_same_text(left_request.get('token_ids', []), right_request.get('token_ids', []))}"
    )
    print(
        f"    token_slots: good={_format_short_list(left_slots)} "
        f"bad={_format_short_list(right_slots)} {_same_text(left_slots, right_slots)}"
    )


def _print_input_meta_reports(
    left_tensor_events: list[TensorEventRecord],
    right_tensor_events: list[TensorEventRecord],
    rank_pairs: list[tuple[int, int, int]],
) -> None:
    print("\nPrefill input metadata")
    for left_rank, right_rank, rank_key in rank_pairs:
        left_group = _group_tensor_events(left_tensor_events, left_rank, rank_key)
        right_group = _group_tensor_events(right_tensor_events, right_rank, rank_key)
        keys = {
            key
            for key in set(left_group) & set(right_group)
            if key.model == "target"
            and key.stage == "prefill"
            and key.layer < 0
            and key.point == "model_input_hidden"
            and key.metric == "hidden"
        }
        for key in _ordered_keys(keys):
            _print_input_meta_for_pair(left_group[key], right_group[key])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare speculative feature event dumps. The script restores KV "
            "token order from token_slots/selected_blocks and reports the "
            "first large L2 difference."
        )
    )
    parser.add_argument("good_request_id", nargs="?")
    parser.add_argument("bad_request_id", nargs="?")
    parser.add_argument(
        "--dump-root",
        "--log",
        action="append",
        default=[],
        help="Dump root directory containing event-*/meta.json. Repeatable.",
    )
    parser.add_argument("--rank-mod", type=int, default=None)
    parser.add_argument("--all-ranks", action="store_true")
    parser.add_argument("--abs-threshold", type=float, default=1e-5)
    parser.add_argument("--rel-threshold", type=float, default=1e-5)
    parser.add_argument("--max-summary", type=int, default=80)
    parser.add_argument("--list-requests", action="store_true")
    parser.add_argument(
        "--compare-input-meta",
        action="store_true",
        help=(
            "Print target prefill model_input_hidden batch/request metadata "
            "for two request ids without loading tensor data."
        ),
    )
    parser.add_argument(
        "--prefill-cos",
        action="store_true",
        help=(
            "Print cosine similarity for target/draft prefill hidden and KV "
            "features. Target layers: model input/output plus 0/30/60. "
            "Draft layers: model input/output plus layer 0."
        ),
    )
    parser.add_argument(
        "--cos-scope",
        choices=("prefill", "all"),
        default="prefill",
        help="Cosine comparison scope. Use all to include decode/mixed events.",
    )
    parser.add_argument(
        "--cos-tokens",
        type=int,
        default=-1,
        help="Number of token cosine values to print; negative means all tokens.",
    )
    parser.add_argument(
        "--cos-summary-only",
        action="store_true",
        help="Only print cosine event summaries, without per-token values or dirs.",
    )
    parser.add_argument(
        "--cos-only-bad",
        action="store_true",
        help="Only print cosine events below threshold or with invalid comparisons.",
    )
    parser.add_argument(
        "--cos-first-bad-only",
        action="store_true",
        help="Only print the first cosine event below threshold.",
    )
    parser.add_argument(
        "--cos-bad-table",
        action="store_true",
        help="Print all cosine events below threshold as a compact table.",
    )
    parser.add_argument("--cos-threshold", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dump_roots = args.dump_root or ["."]
    if args.list_requests:
        _list_requests(dump_roots)
        return
    if not args.good_request_id or not args.bad_request_id:
        raise SystemExit("good_request_id and bad_request_id are required")

    request_filter = {args.good_request_id, args.bad_request_id}
    tensor_events: list[TensorEventRecord] = []
    if args.prefill_cos or args.compare_input_meta:
        tensor_events = _load_tensor_events(dump_roots, request_filter, args.rank_mod)
        left_tensor_events = [
            record
            for record in tensor_events
            if record.request_id == args.good_request_id
        ]
        right_tensor_events = [
            record
            for record in tensor_events
            if record.request_id == args.bad_request_id
        ]
        if not left_tensor_events:
            raise SystemExit(f"no feature records for {args.good_request_id}")
        if not right_tensor_events:
            raise SystemExit(f"no feature records for {args.bad_request_id}")
        rank_pairs = _select_rank_pairs(
            [
                FeatureRecord(
                    key=record.key,
                    request_id=record.request_id,
                    rank=record.rank,
                    event_index=record.event_index,
                    event_dir=record.event_dir,
                    values=[],
                    positions=[],
                    meta=record.meta,
                )
                for record in left_tensor_events
            ],
            [
                FeatureRecord(
                    key=record.key,
                    request_id=record.request_id,
                    rank=record.rank,
                    event_index=record.event_index,
                    event_dir=record.event_dir,
                    values=[],
                    positions=[],
                    meta=record.meta,
                )
                for record in right_tensor_events
            ],
            args.all_ranks,
        )
    else:
        records = _load_records(dump_roots, request_filter, args.rank_mod)
        left_records = [
            record for record in records if record.request_id == args.good_request_id
        ]
        right_records = [
            record for record in records if record.request_id == args.bad_request_id
        ]
        if not left_records:
            raise SystemExit(f"no feature records for {args.good_request_id}")
        if not right_records:
            raise SystemExit(f"no feature records for {args.bad_request_id}")
        rank_pairs = _select_rank_pairs(left_records, right_records, args.all_ranks)
    if not rank_pairs:
        raise SystemExit("no common ranks to compare; try --rank-mod")

    print("Rank alignment")
    for left_rank, right_rank, key in rank_pairs:
        print(f"  key={key} good_rank={left_rank} bad_rank={right_rank}")

    if args.compare_input_meta:
        _print_input_meta_reports(
            left_tensor_events, right_tensor_events, rank_pairs
        )

    if args.prefill_cos:
        cosine_reports: list[CosineReport] = []
        for left_rank, right_rank, rank_key in rank_pairs:
            left_group = _group_tensor_events(
                left_tensor_events, left_rank, rank_key
            )
            right_group = _group_tensor_events(
                right_tensor_events, right_rank, rank_key
            )
            keys = {
                key
                for key in set(left_group) & set(right_group)
                if _cos_key_filter(key, args.cos_scope)
            }
            for key in _ordered_keys(keys):
                cosine_reports.append(
                    _compare_cosine_record(
                        key,
                        left_group[key],
                        right_group[key],
                        args.cos_tokens,
                        args.cos_threshold,
                    )
                )
        if args.cos_first_bad_only:
            _print_first_bad_cosine_report(cosine_reports)
            return
        if args.cos_bad_table:
            _print_bad_cosine_table(cosine_reports)
            return
        _print_cosine_reports(
            cosine_reports,
            args.cos_tokens,
            args.cos_summary_only,
            args.cos_only_bad,
        )

    if args.prefill_cos or args.compare_input_meta:
        return

    all_diffs: list[DiffItem] = []
    first_large: DiffItem | None = None
    missing: list[FeatureKey] = []
    for left_rank, right_rank, rank_key in rank_pairs:
        left_group = _group_records(left_records, left_rank, rank_key)
        right_group = _group_records(right_records, right_rank, rank_key)
        keys = set(left_group) | set(right_group)
        common = set(left_group) & set(right_group)
        for key in _ordered_keys(keys):
            if key not in common:
                missing.append(key)
                continue
            diff = _compare_record(key, left_group[key], right_group[key])
            if diff is None:
                continue
            all_diffs.append(diff)
            if first_large is None and _is_large_diff(
                diff, args.abs_threshold, args.rel_threshold
            ):
                first_large = diff

    all_diffs.sort(
        key=lambda item: (
            item.key.rank_key,
            _phase_order(item.key.stage, item.key.model),
            item.key.layer if item.key.layer >= 0 else -1,
            _point_order(item.key.point),
            item.key.metric,
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
            max_item = max(all_diffs, key=lambda item: (item.rel_diff, item.abs_diff))
            print("  max observed diff:")
            print(f"  {_format_diff(max_item)}")
    else:
        print(f"  {_format_diff(first_large)}")
        _print_context("  good", first_large.left_record)
        _print_context("  bad", first_large.right_record)

    print("\nStage summary")
    for item in all_diffs[: max(args.max_summary, 0)]:
        print(f"  {_format_diff(item)}")
    if len(all_diffs) > args.max_summary:
        print(f"  ... {len(all_diffs) - args.max_summary} more summary rows")

    if missing:
        print("\nMissing comparable records")
        for key in _ordered_keys(set(missing))[:20]:
            print(f"  {_format_key(key)}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more missing keys")


if __name__ == "__main__":
    main()
