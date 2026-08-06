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

"""Replay a vLLM Ascend TP2 Kimi K3 Gated-MLA dump on one device."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "xllm/xllm/python/models/kimi_k3_gated_mla.py"
DEFAULT_DUMP_DIR = ROOT.parent / "vllm-workspace/outputs"
DEFAULT_STEM = "kimi_k3_layer3_mla"


def _load_module() -> Any:
    name = "kimi_k3_gated_mla_tp2_replay"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Gated-MLA module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config(module: Any) -> Any:
    return module.KimiK3GatedMLAConfig(
        hidden_size=7168,
        num_attention_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-5,
    )


def _path(dump_dir: Path, stem: str, rank: int, kind: str) -> Path:
    return dump_dir / f"{stem}_tp{rank}_{kind}.pt"


def _load_rank_weights(dump_dir: Path, stem: str, rank: int) -> dict[str, Any]:
    payload = torch.load(
        _path(dump_dir, stem, rank, "weights"),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("parameters"), dict):
        raise TypeError(f"rank {rank} weights must contain a parameters mapping")
    return payload


def _require_equal(name: str, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if not torch.equal(left, right):
        raise ValueError(f"replicated TP tensor differs across ranks: {name}")
    return left


def _join_output_shards(
    name: str,
    rank0: dict[str, torch.Tensor],
    rank1: dict[str, torch.Tensor],
) -> torch.Tensor:
    return torch.cat([rank0[name], rank1[name]], dim=0).contiguous()


def reconstruct_weights(
    rank0_payload: dict[str, Any], rank1_payload: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Restore PyTorch Linear-format global weights from vLLM TP2 tensors."""
    rank0 = rank0_payload["parameters"]
    rank1 = rank1_payload["parameters"]
    expected = {
        "fused_qkv_a_proj.weight": (7168, 2112),
        "q_b_proj.weight": (1536, 9216),
        "kv_b_proj.weight": (12288, 512),
        "g_proj.weight": (6144, 7168),
        "o_proj.weight": (7168, 6144),
    }
    for name, shape in expected.items():
        for rank, parameters in enumerate((rank0, rank1)):
            if tuple(parameters[name].shape) != shape:
                raise ValueError(
                    f"rank {rank} {name} expected {shape}, got {tuple(parameters[name].shape)}"
                )

    fused = _require_equal(
        "fused_qkv_a_proj.weight",
        rank0["fused_qkv_a_proj.weight"],
        rank1["fused_qkv_a_proj.weight"],
    )
    fused_scale = _require_equal(
        "fused_qkv_a_proj.weight_scale",
        rank0["fused_qkv_a_proj.weight_scale"],
        rank1["fused_qkv_a_proj.weight_scale"],
    )
    fused_offset = _require_equal(
        "fused_qkv_a_proj.weight_offset",
        rank0["fused_qkv_a_proj.weight_offset"],
        rank1["fused_qkv_a_proj.weight_offset"],
    )
    q_a, kv_a = fused.split([1536, 576], dim=1)
    q_a_scale, kv_a_scale = fused_scale.split([1536, 576])
    q_a_offset, kv_a_offset = fused_offset.split([1536, 576])

    q_b_runtime = torch.cat(
        [rank0["q_b_proj.weight"], rank1["q_b_proj.weight"]], dim=1
    )
    q_b_scale = torch.cat(
        [rank0["q_b_proj.weight_scale"], rank1["q_b_proj.weight_scale"]]
    )
    q_b_offset = torch.cat(
        [rank0["q_b_proj.weight_offset"], rank1["q_b_proj.weight_offset"]]
    )

    weights = {
        "q_a_proj.weight": q_a.t().contiguous(),
        "q_a_proj.weight_scale": q_a_scale.contiguous(),
        "q_a_proj.weight_offset": q_a_offset.contiguous(),
        "q_a_layernorm.weight": _require_equal(
            "q_a_layernorm.weight",
            rank0["q_a_layernorm.weight"],
            rank1["q_a_layernorm.weight"],
        ),
        "q_b_proj.weight": q_b_runtime.t().contiguous(),
        "q_b_proj.weight_scale": q_b_scale.contiguous(),
        "q_b_proj.weight_offset": q_b_offset.contiguous(),
        "kv_a_proj_with_mqa.weight": kv_a.t().contiguous(),
        "kv_a_proj_with_mqa.weight_scale": kv_a_scale.contiguous(),
        "kv_a_proj_with_mqa.weight_offset": kv_a_offset.contiguous(),
        "kv_a_layernorm.weight": _require_equal(
            "kv_a_layernorm.weight",
            rank0["kv_a_layernorm.weight"],
            rank1["kv_a_layernorm.weight"],
        ),
        "kv_b_proj.weight": _join_output_shards("kv_b_proj.weight", rank0, rank1),
        "g_proj.weight": _join_output_shards("g_proj.weight", rank0, rank1),
        "o_proj.weight": torch.cat(
            [rank0["o_proj.weight"], rank1["o_proj.weight"]], dim=1
        ).contiguous(),
    }
    return weights


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual_fp32 = actual.float().cpu()
    reference_fp32 = reference.float().cpu()
    difference = actual_fp32 - reference_fp32
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "finite": bool(torch.isfinite(actual_fp32).all()),
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "rmse": float(difference.square().mean().sqrt()),
        "cosine": float(F.cosine_similarity(
            actual_fp32.reshape(1, -1), reference_fp32.reshape(1, -1)
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--save-output", type=Path)
    parser.add_argument("--save-metrics", type=Path)
    args = parser.parse_args()

    if args.device == "npu":
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError("Ascend NPU is unavailable")

    rank_weights = [
        _load_rank_weights(args.dump_dir, args.stem, rank) for rank in (0, 1)
    ]
    hidden_states = [
        torch.load(
            _path(args.dump_dir, args.stem, rank, "input"),
            map_location="cpu",
            weights_only=True,
        )
        for rank in (0, 1)
    ]
    references = [
        torch.load(
            _path(args.dump_dir, args.stem, rank, "output"),
            map_location="cpu",
            weights_only=True,
        )
        for rank in (0, 1)
    ]
    _require_equal("input", hidden_states[0], hidden_states[1])
    _require_equal("reference output", references[0], references[1])

    module = _load_module()
    device = torch.device(args.device)
    model = module.KimiK3GatedMLA(_config(module), dtype=torch.bfloat16, device=device)
    model.load_checkpoint_weights(reconstruct_weights(*rank_weights))
    model.eval()
    with torch.inference_mode():
        output = model(hidden_states[0].to(device), sequence_lengths=[5])
    if args.device == "npu":
        torch.npu.synchronize()
    output = output.cpu()
    metrics = error_metrics(output, references[0])
    metrics.update({"device": args.device, "stem": args.stem, "tp_size": 2})

    if args.save_output is not None:
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, args.save_output)
    if args.save_metrics is not None:
        args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.save_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()