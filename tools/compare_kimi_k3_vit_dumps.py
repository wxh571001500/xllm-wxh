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

"""Compare Kimi K3 ViT tensor dumps produced by xLLM and vLLM."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch


DEFAULT_XLLM_DIR = Path(
    "/export/home/wangxiaohan17/wangxiaohan/xllm-k3-dump"
)
DEFAULT_VLLM_DIR = Path(
    "/export/home/wangxiaohan17/wangxiaohan/vllm-k3-dump"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xllm-dir", type=Path, default=DEFAULT_XLLM_DIR)
    parser.add_argument("--vllm-dir", type=Path, default=DEFAULT_VLLM_DIR)
    parser.add_argument("--xllm-dump", type=Path)
    parser.add_argument("--vllm-dump", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--call-index",
        type=int,
        help="Use this call index in both dump directories; latest by default.",
    )
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def _select_dump(
    explicit_path: Path | None,
    dump_dir: Path,
    rank: int,
    call_index: int | None,
) -> Path:
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Dump does not exist: {explicit_path}")
        return explicit_path

    if call_index is not None:
        path = dump_dir / f"call_{call_index:04d}_rank_{rank:03d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Dump does not exist: {path}")
        return path

    candidates = sorted(dump_dir.glob(f"call_*_rank_{rank:03d}.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No Kimi K3 ViT dumps for rank {rank} under {dump_dir}"
        )
    return candidates[-1]


def _load_dump(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("tensors"), dict):
        raise TypeError(f"Invalid Kimi K3 ViT dump: {path}")
    return payload


def _name_sort_key(name: str) -> tuple[int, int, str]:
    if name.startswith("vit_input"):
        return (0, 0, name)
    if name.startswith("distribution") or name == "local_pixel_values":
        return (1, 0, name)
    if name == "patch_embed":
        return (2, 0, name)
    if name.startswith("encoder.block."):
        parts = name.split(".")
        layer_index = int(parts[2])
        if len(parts) > 4 and parts[3] == "weight":
            weight_order = {
                "norm0": 0,
                "qkv": 1,
                "wo": 2,
                "norm1": 3,
                "fc0": 4,
                "fc1": 5,
            }
            return (3, layer_index * 20 + weight_order.get(parts[4], 9), name)
        stage_order = {
            "norm0": 0,
            "qkv": 1,
            "attention_output": 2,
            "wo": 3,
            "norm1": 4,
            "fc0": 5,
            "activation": 6,
            "fc1": 7,
        }
        stage = parts[3] if len(parts) > 3 else ""
        return (3, layer_index * 20 + 10 + stage_order.get(stage, 9), name)
    if name == "encoder.final_layernorm":
        return (4, 0, name)
    if name == "local_tower_output":
        return (5, 0, name)
    if name == "gathered_tower_output":
        return (6, 0, name)
    if name == "vit_output":
        return (7, 0, name)
    return (8, 0, name)


def _format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6e}"


def _compare_tensor(
    name: str,
    xllm_record: dict[str, Any],
    vllm_record: dict[str, Any],
    atol: float,
    rtol: float,
) -> tuple[str, bool]:
    xllm_shape = list(xllm_record["shape"])
    vllm_shape = list(vllm_record["shape"])
    xllm_dtype = xllm_record.get("dtype", "unknown")
    vllm_dtype = vllm_record.get("dtype", "unknown")
    xllm_rows = xllm_record["row_indices"]
    vllm_rows = vllm_record["row_indices"]
    xllm_values = xllm_record["values"]
    vllm_values = vllm_record["values"]

    if xllm_shape != vllm_shape:
        return (
            f"{name:<38} SHAPE xllm={xllm_shape} vllm={vllm_shape}",
            False,
        )
    if xllm_dtype != vllm_dtype:
        return (
            f"{name:<38} DTYPE xllm={xllm_dtype} vllm={vllm_dtype}",
            False,
        )
    if not torch.equal(xllm_rows, vllm_rows):
        return (
            f"{name:<38} ROWS xllm={xllm_rows.tolist()} "
            f"vllm={vllm_rows.tolist()}",
            False,
        )
    if xllm_values.shape != vllm_values.shape:
        return (
            f"{name:<38} VALUE_SHAPE xllm={list(xllm_values.shape)} "
            f"vllm={list(vllm_values.shape)}",
            False,
        )
    if xllm_values.numel() == 0:
        return (
            f"{name:<38} EMPTY dtype={xllm_dtype} shape={xllm_shape}",
            True,
        )

    if not xllm_values.is_floating_point() and not vllm_values.is_floating_point():
        equal = torch.equal(xllm_values, vllm_values)
        return (
            f"{name:<38} EXACT={str(equal):<5} "
            f"dtype={xllm_dtype} shape={xllm_shape}",
            equal,
        )

    actual = xllm_values.float()
    expected = vllm_values.float()
    difference = actual - expected
    absolute = difference.abs()
    max_abs = absolute.max().item()
    mean_abs = absolute.mean().item()
    expected_norm = expected.norm().item()
    relative_l2 = difference.norm().item() / max(expected_norm, 1e-12)
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    norm_product = actual_flat.norm().item() * expected_flat.norm().item()
    cosine = (
        torch.dot(actual_flat, expected_flat).item() / norm_product
        if norm_product > 0
        else float("nan")
    )
    close = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    close_ratio = close.float().mean().item()
    all_close = bool(close.all().item())
    line = (
        f"{name:<38} max={_format_metric(max_abs)} "
        f"mean={_format_metric(mean_abs)} rel_l2={_format_metric(relative_l2)} "
        f"cos={_format_metric(cosine)} close={close_ratio:.6f} "
        f"allclose={all_close} dtype={xllm_dtype} shape={xllm_shape}"
    )
    return line, all_close


def main() -> int:
    args = _parse_args()
    xllm_path = _select_dump(
        args.xllm_dump,
        args.xllm_dir,
        args.rank,
        args.call_index,
    )
    vllm_path = _select_dump(
        args.vllm_dump,
        args.vllm_dir,
        args.rank,
        args.call_index,
    )
    xllm_dump = _load_dump(xllm_path)
    vllm_dump = _load_dump(vllm_path)

    print(f"xLLM: {xllm_path}")
    print(f"vLLM: {vllm_path}")
    print(f"tolerance: atol={args.atol} rtol={args.rtol}")

    xllm_tensors = xllm_dump["tensors"]
    vllm_tensors = vllm_dump["tensors"]
    common_names = sorted(
        set(xllm_tensors) & set(vllm_tensors),
        key=_name_sort_key,
    )
    only_xllm = sorted(set(xllm_tensors) - set(vllm_tensors), key=_name_sort_key)
    only_vllm = sorted(set(vllm_tensors) - set(xllm_tensors), key=_name_sort_key)
    if only_xllm:
        print(f"only in xLLM: {', '.join(only_xllm)}")
    if only_vllm:
        print(f"only in vLLM: {', '.join(only_vllm)}")

    all_match = bool(common_names)
    for name in common_names:
        line, matches = _compare_tensor(
            name,
            xllm_tensors[name],
            vllm_tensors[name],
            args.atol,
            args.rtol,
        )
        print(line)
        all_match = all_match and matches

    if args.fail_on_mismatch and not all_match:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
