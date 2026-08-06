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

"""Module-level alignment tests for standalone Kimi K3 Gated-MLA."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

_MODULE_NAME = "kimi_k3_gated_mla"
_MODULE_PATH = (
    Path(__file__).parents[2] / "xllm/python/models/kimi_k3_gated_mla.py"
)
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load standalone Gated-MLA module from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _MODULE
_SPEC.loader.exec_module(_MODULE)
KimiK3GatedMLA = _MODULE.KimiK3GatedMLA
KimiK3GatedMLAConfig = _MODULE.KimiK3GatedMLAConfig


def _tiny_config() -> KimiK3GatedMLAConfig:
    return KimiK3GatedMLAConfig(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=6,
        kv_lora_rank=5,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=3,
    )


def _rms_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_states.float() * torch.rsqrt(variance + eps)
    return normalized.to(hidden_states.dtype) * weight


def _reference_forward(
    hidden_states: torch.Tensor,
    weights: dict[str, torch.Tensor],
    config: KimiK3GatedMLAConfig,
    sequence_lengths: list[int],
) -> torch.Tensor:
    query_latent = F.linear(hidden_states, weights["q_a_proj.weight"])
    query_latent = _rms_norm(
        query_latent, weights["q_a_layernorm.weight"], config.rms_norm_eps
    )
    query = F.linear(query_latent, weights["q_b_proj.weight"])
    query = query.view(hidden_states.shape[0], config.num_attention_heads, -1)

    compressed_kv = F.linear(
        hidden_states, weights["kv_a_proj_with_mqa.weight"]
    )
    kv_latent, shared_key = compressed_kv.split(
        [config.kv_lora_rank, config.qk_rope_head_dim], dim=-1
    )
    kv_latent = _rms_norm(
        kv_latent, weights["kv_a_layernorm.weight"], config.rms_norm_eps
    )
    key_value = F.linear(kv_latent, weights["kv_b_proj.weight"])
    key_value = key_value.view(
        hidden_states.shape[0], config.num_attention_heads, -1
    )
    key_nope, value = key_value.split(
        [config.qk_nope_head_dim, config.v_head_dim], dim=-1
    )
    shared_key = shared_key.unsqueeze(1).expand(
        -1, config.num_attention_heads, -1
    )
    key = torch.cat([key_nope, shared_key], dim=-1)

    attended_sequences: list[torch.Tensor] = []
    start = 0
    scale = (config.qk_nope_head_dim + config.qk_rope_head_dim) ** -0.5
    for length in sequence_lengths:
        end = start + length
        query_sequence = query[start:end].transpose(0, 1).unsqueeze(0)
        key_sequence = key[start:end].transpose(0, 1).unsqueeze(0)
        value_sequence = value[start:end].transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            query_sequence,
            key_sequence,
            value_sequence,
            is_causal=True,
            scale=scale,
        )
        attended_sequences.append(attended.squeeze(0).transpose(0, 1))
        start = end

    attended = torch.cat(attended_sequences, dim=0).reshape(hidden_states.shape[0], -1)
    gate = torch.sigmoid(F.linear(hidden_states, weights["g_proj.weight"]))
    return F.linear(attended * gate, weights["o_proj.weight"])


def _random_weights(
    config: KimiK3GatedMLAConfig,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(23)
    shapes = {
        "q_a_proj.weight": (config.q_lora_rank, config.hidden_size),
        "q_a_layernorm.weight": (config.q_lora_rank,),
        "q_b_proj.weight": (
            config.num_attention_heads
            * (config.qk_nope_head_dim + config.qk_rope_head_dim),
            config.q_lora_rank,
        ),
        "kv_a_proj_with_mqa.weight": (
            config.kv_lora_rank + config.qk_rope_head_dim,
            config.hidden_size,
        ),
        "kv_a_layernorm.weight": (config.kv_lora_rank,),
        "kv_b_proj.weight": (
            config.num_attention_heads
            * (config.qk_nope_head_dim + config.v_head_dim),
            config.kv_lora_rank,
        ),
        "g_proj.weight": (
            config.num_attention_heads * config.v_head_dim,
            config.hidden_size,
        ),
        "o_proj.weight": (
            config.hidden_size,
            config.num_attention_heads * config.v_head_dim,
        ),
    }
    return {
        name: torch.randn(shape, generator=generator) / math.sqrt(shape[-1])
        for name, shape in shapes.items()
    }


def test_checkpoint_loading_and_reference_output_alignment() -> None:
    config = _tiny_config()
    model = KimiK3GatedMLA(config, dtype=torch.float32)
    weights = _random_weights(config)
    prefix = "model.layers.3.self_attn."
    model.load_checkpoint_weights(
        {f"{prefix}{name}": tensor for name, tensor in weights.items()}, prefix
    )
    hidden_states = torch.randn(7, config.hidden_size, generator=torch.Generator().manual_seed(11))

    actual = model(hidden_states, sequence_lengths=[3, 4])
    expected = _reference_forward(hidden_states, weights, config, [3, 4])

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_output_gate_is_applied_before_output_projection() -> None:
    config = _tiny_config()
    model = KimiK3GatedMLA(config, dtype=torch.float32)
    weights = _random_weights(config)
    weights["g_proj.weight"].zero_()
    model.load_checkpoint_weights(weights)
    hidden_states = torch.randn(4, config.hidden_size)

    query, key, value = model._project_qkv(hidden_states)
    ungated = model._causal_attention(query, key, value, [4]).reshape(4, -1)
    expected = F.linear(ungated * 0.5, weights["o_proj.weight"])

    torch.testing.assert_close(model(hidden_states), expected)


def test_checkpoint_loading_dequantizes_per_channel_int8_weight() -> None:
    config = _tiny_config()
    model = KimiK3GatedMLA(config, dtype=torch.float32)
    weights = _random_weights(config)
    quantized_name = "q_a_proj.weight"
    quantized = torch.arange(
        config.q_lora_rank * config.hidden_size, dtype=torch.int8
    ).reshape(config.q_lora_rank, config.hidden_size)
    scale = torch.linspace(0.001, 0.006, config.q_lora_rank).reshape(-1, 1)
    offset = torch.zeros_like(scale)
    weights[quantized_name] = quantized
    weights[f"{quantized_name}_scale"] = scale
    weights[f"{quantized_name}_offset"] = offset

    model.load_checkpoint_weights(weights)

    expected = quantized.float() * scale
    torch.testing.assert_close(model.q_a_proj.weight, expected)


def test_checkpoint_loading_rejects_missing_weight() -> None:
    config = _tiny_config()
    model = KimiK3GatedMLA(config)
    weights = _random_weights(config)
    del weights["g_proj.weight"]

    with pytest.raises(KeyError, match="g_proj.weight"):
        model.load_checkpoint_weights(weights)