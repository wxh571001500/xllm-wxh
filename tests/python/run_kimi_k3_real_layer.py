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

"""Run one real Kimi K3 Gated-MLA checkpoint layer on CPU or Ascend NPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from safetensors import safe_open
import torch


ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "xllm/xllm/python/models/kimi_k3_gated_mla.py"
DEFAULT_MODEL_DIR = Path("/mnt/cfs/9n-das-admin/llm_models/kimi-k3")
DEFAULT_INPUT = ROOT / "inputs/kimi_k3_gated_mla_layer11_input.pt"
PREFIX_TEMPLATE = "language_model.model.layers.{layer}.self_attn."
WEIGHT_NAMES = (
    "q_a_proj.weight",
    "q_a_layernorm.weight",
    "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "g_proj.weight",
    "o_proj.weight",
)


def _load_module() -> Any:
    name = "kimi_k3_gated_mla_real_layer"
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


def create_fixed_input(path: Path, tokens: int = 2, seed: int = 20260804) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = torch.randn(tokens, 7168, generator=generator, dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden_states": hidden_states,
            "sequence_lengths": [tokens],
            "seed": seed,
            "layer_index": 11,
        },
        path,
    )
    metadata = {
        "dtype": "float32",
        "hidden_size": 7168,
        "layer_index": 11,
        "seed": seed,
        "sequence_lengths": [tokens],
        "shape": list(hidden_states.shape),
    }
    path.with_name(f"{path.stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def load_layer_weights(model_dir: Path, layer: int) -> dict[str, torch.Tensor]:
    index_path = model_dir / "quant_model_weights.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    prefix = PREFIX_TEMPLATE.format(layer=layer)
    tensor_names: list[str] = []
    for short_name in WEIGHT_NAMES:
        full_name = f"{prefix}{short_name}"
        if full_name not in weight_map:
            raise KeyError(f"checkpoint tensor not found: {full_name}")
        tensor_names.append(full_name)
        if not short_name.endswith("layernorm.weight"):
            for suffix in ("_scale", "_offset"):
                metadata_name = f"{full_name}{suffix}"
                if metadata_name in weight_map:
                    tensor_names.append(metadata_name)

    by_shard: dict[str, list[str]] = {}
    for name in tensor_names:
        by_shard.setdefault(weight_map[name], []).append(name)

    weights: dict[str, torch.Tensor] = {}
    for shard, names in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for name in names:
                weights[name] = handle.get_tensor(name)
    return weights


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.float().cpu()
    raw = values.contiguous().numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(values).all()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "sha256_fp32": hashlib.sha256(raw).hexdigest(),
    }


def configure_determinism(device: str, seed: int) -> None:
    """Configure the minimal deterministic settings used by this test."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if device == "npu":
        torch.npu.manual_seed_all(seed)


def validate_fixed_input(payload: dict[str, Any], layer: int) -> None:
    hidden_states = payload.get("hidden_states")
    lengths = payload.get("sequence_lengths")
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("fixed input must contain a hidden_states tensor")
    if hidden_states.shape != (2, 7168) or hidden_states.dtype != torch.float32:
        raise ValueError("fixed hidden_states must be FP32 with shape [2, 7168]")
    if lengths != [2]:
        raise ValueError("fixed sequence_lengths must equal [2]")
    if payload.get("layer_index") != layer:
        raise ValueError("fixed input layer index does not match requested layer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--create-input", action="store_true")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 2:
        raise ValueError("--repeat must be at least 2 for determinism verification")

    if args.device == "npu":
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError("Ascend NPU is unavailable")

    configure_determinism(args.device, args.seed)
    if args.create_input:
        create_fixed_input(args.input, seed=args.seed)
    if not args.input.exists():
        raise FileNotFoundError(
            f"fixed input does not exist: {args.input}; run once with --create-input"
        )
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    validate_fixed_input(payload, args.layer)
    if payload.get("seed") != args.seed:
        raise ValueError(
            f"fixed input seed {payload.get('seed')} does not match --seed {args.seed}"
        )

    module = _load_module()
    device = torch.device(args.device)
    model = module.KimiK3GatedMLA(_config(module), dtype=torch.bfloat16, device=device)
    weights = load_layer_weights(args.model_dir, args.layer)
    prefix = PREFIX_TEMPLATE.format(layer=args.layer)
    model.load_checkpoint_weights(weights, prefix=prefix)
    del weights

    hidden_states = payload["hidden_states"].to(device=device, dtype=torch.bfloat16)
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for _ in range(args.repeat):
            outputs.append(model(hidden_states, payload["sequence_lengths"]).cpu())
    if args.device == "npu":
        torch.npu.synchronize()
    output = outputs[0]
    if not all(torch.equal(output, repeated) for repeated in outputs[1:]):
        raise RuntimeError(
            f"non-deterministic output detected across {args.repeat} repeated forwards"
        )

    output_path = args.input.with_name(
        f"kimi_k3_gated_mla_layer{args.layer}_{args.device}_output.pt"
    )
    torch.save(output, output_path)
    summary = tensor_summary(output)
    summary["device"] = args.device
    summary["layer_index"] = args.layer
    summary["seed"] = args.seed
    summary["repeat"] = args.repeat
    summary["deterministic_repeat_equal"] = True
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()