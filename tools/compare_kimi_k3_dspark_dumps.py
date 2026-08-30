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

"""Compare Kimi K3 DSpark accuracy dumps from xLLM and vLLM Ascend."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xllm-dir", type=Path, required=True)
    parser.add_argument("--vllm-dir", type=Path, required=True)
    parser.add_argument(
        "--component",
        choices=(
            "target_forward",
            "target_logits",
            "context_projection",
            "draft_forward",
            "draft_logits",
            "draft_markov",
        ),
        default="draft_forward",
    )
    parser.add_argument("--xllm-call-index", type=int)
    parser.add_argument("--vllm-call-index", type=int)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available calls for the selected component and exit.",
    )
    return parser.parse_args()


def _dump_paths(directory: Path, component: str, rank: int) -> list[Path]:
    exact_paths = sorted(directory.rglob(f"{component}_call_*_rank_{rank:03d}.pt"))
    if exact_paths:
        return exact_paths

    # Older debug revisions did not always include the component/rank in the
    # filename. Inspect the payload as a fallback so the tool can diagnose
    # those dumps instead of reporting an empty directory.
    matched_paths: list[Path] = []
    for path in sorted(directory.rglob("*.pt")):
        try:
            payload = _load_dump(path)
        except (OSError, RuntimeError, TypeError, EOFError):
            continue
        if payload.get("component") != component:
            continue
        payload_rank = payload.get("rank", rank)
        if int(payload_rank) == rank:
            matched_paths.append(path)
    return matched_paths


def _describe_directory(directory: Path) -> str:
    paths = sorted(directory.rglob("*.pt"))
    if not paths:
        return "no .pt files"
    descriptions: list[str] = []
    for path in paths:
        try:
            payload = _load_dump(path)
            descriptions.append(
                f"{path.name}"
                f"(component={payload.get('component', '?')}, "
                f"rank={payload.get('rank', '?')})"
            )
        except (OSError, RuntimeError, TypeError, EOFError):
            descriptions.append(f"{path.name}(unreadable)")
    return ", ".join(descriptions)


def _select_dump(
    directory: Path,
    component: str,
    rank: int,
    call_index: int | None,
) -> Path:
    if call_index is not None:
        path = directory / (
            f"{component}_call_{call_index:04d}_rank_{rank:03d}.pt"
        )
        if path.is_file():
            return path
        for candidate in _dump_paths(directory, component, rank):
            payload = _load_dump(candidate)
            if int(payload.get("call_index", -1)) == call_index:
                return candidate
        raise FileNotFoundError(
            f"Dump does not exist for component={component}, rank={rank}, "
            f"call_index={call_index}. Files found: {_describe_directory(directory)}"
        )

    paths = _dump_paths(directory, component, rank)
    if not paths:
        raise FileNotFoundError(
            f"No {component} dumps for rank {rank} under {directory}. "
            f"Files found: {_describe_directory(directory)}"
        )
    return paths[-1]


def _load_dump(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("tensors"), dict):
        raise TypeError(f"Invalid DSpark accuracy dump: {path}")
    return payload


def _format_metric(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.6e}"


def _compare_tensor(
    name: str,
    xllm_record: dict[str, Any],
    vllm_record: dict[str, Any],
    atol: float,
    rtol: float,
) -> tuple[str, bool]:
    xllm_shape = list(xllm_record["shape"])
    vllm_shape = list(vllm_record["shape"])
    if xllm_shape != vllm_shape:
        detail = ""
        actual = xllm_record["values"]
        expected = vllm_record["values"]
        if not actual.is_floating_point() and not expected.is_floating_point():
            actual_flat = actual.reshape(-1)
            expected_flat = expected.reshape(-1)
            common_size = min(actual_flat.numel(), expected_flat.numel())
            mismatch = torch.nonzero(
                actual_flat[:common_size] != expected_flat[:common_size]
            ).reshape(-1)
            if mismatch.numel():
                index = int(mismatch[0].item())
                start = max(index - 3, 0)
                end = min(index + 4, common_size)
                detail = (
                    f" first_mismatch={index}"
                    f" xllm={actual_flat[start:end].tolist()}"
                    f" vllm={expected_flat[start:end].tolist()}"
                )
            elif actual_flat.numel() != expected_flat.numel():
                detail = (
                    f" common_prefix={common_size}"
                    f" xllm_next={actual_flat[common_size:common_size + 4].tolist()}"
                    f" vllm_next={expected_flat[common_size:common_size + 4].tolist()}"
                )
        return (
            f"{name:<42} SHAPE xllm={xllm_shape} vllm={vllm_shape}{detail}",
            False,
        )
    xllm_dtype = xllm_record["dtype"]
    vllm_dtype = vllm_record["dtype"]
    if xllm_dtype != vllm_dtype:
        return f"{name:<42} DTYPE xllm={xllm_dtype} vllm={vllm_dtype}", False

    actual = xllm_record["values"]
    expected = vllm_record["values"]
    if not actual.is_floating_point() and not expected.is_floating_point():
        equal = torch.equal(actual, expected)
        detail = ""
        if not equal and actual.numel() == expected.numel():
            actual_flat = actual.reshape(-1)
            expected_flat = expected.reshape(-1)
            mismatch = torch.nonzero(actual_flat != expected_flat).reshape(-1)
            if mismatch.numel():
                index = int(mismatch[0].item())
                start = max(index - 3, 0)
                end = min(index + 4, actual_flat.numel())
                detail = (
                    f" first_mismatch={index}"
                    f" xllm={actual_flat[start:end].tolist()}"
                    f" vllm={expected_flat[start:end].tolist()}"
                )
        elif not equal:
            detail = f" numel xllm={actual.numel()} vllm={expected.numel()}"
        return (
            f"{name:<42} exact={str(equal):<5} shape={xllm_shape}{detail}",
            equal,
        )

    actual_float = actual.float()
    expected_float = expected.float()
    difference = actual_float - expected_float
    absolute = difference.abs()
    max_abs = absolute.max().item() if absolute.numel() else 0.0
    mean_abs = absolute.mean().item() if absolute.numel() else 0.0
    expected_norm = expected_float.norm().item()
    relative_l2 = difference.norm().item() / max(expected_norm, 1e-12)
    actual_flat = actual_float.reshape(-1)
    expected_flat = expected_float.reshape(-1)
    norm_product = actual_flat.norm().item() * expected_flat.norm().item()
    cosine = (
        torch.dot(actual_flat, expected_flat).item() / norm_product
        if norm_product > 0
        else float("nan")
    )
    close = torch.isclose(actual_float, expected_float, atol=atol, rtol=rtol)
    close_ratio = close.float().mean().item() if close.numel() else 1.0
    all_close = bool(close.all().item()) if close.numel() else True
    argmax_detail = ""
    if "logits" in name and actual_float.ndim >= 1:
        argmax_match = (
            actual_float.argmax(dim=-1) == expected_float.argmax(dim=-1)
        ).float().mean().item()
        topk = min(5, actual_float.shape[-1])
        actual_topk = actual_float.topk(topk, dim=-1).indices
        expected_topk = expected_float.topk(topk, dim=-1).indices
        topk_overlap = (
            actual_topk.unsqueeze(-1) == expected_topk.unsqueeze(-2)
        ).any(dim=-1).float().mean().item()
        argmax_detail = (
            f" argmax={argmax_match:.6f} top{topk}_overlap={topk_overlap:.6f}"
        )
    return (
        f"{name:<42} max={_format_metric(max_abs)} "
        f"mean={_format_metric(mean_abs)} rel_l2={_format_metric(relative_l2)} "
        f"cos={_format_metric(cosine)} close={close_ratio:.6f} "
        f"allclose={all_close}{argmax_detail} shape={xllm_shape}",
        all_close,
    )


def _print_available(directory: Path, component: str, rank: int, label: str) -> None:
    paths = _dump_paths(directory, component, rank)
    print(f"{label} ({len(paths)} calls):")
    if not paths:
        print(f"  requested component={component}, rank={rank}")
        print(f"  files found: {_describe_directory(directory)}")
        return
    for path in paths:
        payload = _load_dump(path)
        shapes = {
            name: record["shape"]
            for name, record in payload["tensors"].items()
        }
        print(f"  {path.name}: {shapes}")
        for name, record in payload["tensors"].items():
            values = record["values"]
            if values.is_floating_point() or not values.numel():
                continue
            preview = values.reshape(-1)[:16].tolist()
            print(f"    {name}: {preview}")


def main() -> int:
    args = _parse_args()
    if args.list:
        _print_available(args.xllm_dir, args.component, args.rank, "xLLM")
        _print_available(args.vllm_dir, args.component, args.rank, "vLLM")
        return 0

    xllm_path = _select_dump(
        args.xllm_dir,
        args.component,
        args.rank,
        args.xllm_call_index,
    )
    vllm_path = _select_dump(
        args.vllm_dir,
        args.component,
        args.rank,
        args.vllm_call_index,
    )
    xllm_dump = _load_dump(xllm_path)
    vllm_dump = _load_dump(vllm_path)
    xllm_tensors = xllm_dump["tensors"]
    vllm_tensors = vllm_dump["tensors"]

    print(f"xLLM: {xllm_path}")
    print(f"vLLM: {vllm_path}")
    print(f"tolerance: atol={args.atol} rtol={args.rtol}")

    only_xllm = sorted(set(xllm_tensors) - set(vllm_tensors))
    only_vllm = sorted(set(vllm_tensors) - set(xllm_tensors))
    if only_xllm:
        print(f"only in xLLM: {', '.join(only_xllm)}")
    if only_vllm:
        print(f"only in vLLM: {', '.join(only_vllm)}")

    common_names = sorted(set(xllm_tensors) & set(vllm_tensors))
    all_match = bool(common_names) and not only_xllm and not only_vllm
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

    print(f"result: {'PASS' if all_match else 'MISMATCH'}")
    return 1 if args.fail_on_mismatch and not all_match else 0


if __name__ == "__main__":
    raise SystemExit(main())
