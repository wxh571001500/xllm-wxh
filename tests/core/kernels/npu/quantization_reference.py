# Copyright 2026 The xLLM Authors.
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

"""Independent CPU checks for BF16/FP16 dynamic-quantization test inputs."""

from __future__ import annotations

import torch


def assert_input_quantization(
    hidden: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor
) -> dict[str, float | int]:
    """Validate nearest-code error, not a unique FP32 evaluation of a half tie.

    The small slack accounts only for this check's FP64 normalization. It does
    not allow arbitrary one-level differences: a non-nearest code still fails.
    """
    assert hidden.device.type == codes.device.type == scales.device.type == "cpu"
    assert hidden.ndim == 2 and hidden.numel() > 0
    assert hidden.dtype in (torch.bfloat16, torch.float16)
    assert codes.shape == hidden.shape and codes.dtype == torch.int8
    assert scales.shape == hidden.shape[:1] and scales.dtype == torch.float32
    assert bool(torch.isfinite(hidden).all()) and bool(torch.isfinite(scales).all())
    assert bool((scales >= 0).all())
    values = hidden.double()
    maximum = values.abs().amax(dim=-1)
    divisor = torch.where(maximum == 0, torch.ones_like(maximum), maximum)
    ideal = values * 127.0 / divisor[:, None]
    expected_scale = maximum / 127.0
    torch.testing.assert_close(scales.double(), expected_scale, rtol=4 * torch.finfo(torch.float32).eps, atol=0)
    distance = (codes.double() - ideal).abs()
    fp64_slack = 8 * torch.finfo(torch.float64).eps * 127
    assert bool((distance <= 0.5 + fp64_slack).all()), "dynamic quantization emitted a non-nearest INT8 code"
    legacy_scale = hidden.float().abs().amax(dim=-1) / 127.0
    legacy_divisor = torch.where(legacy_scale == 0, torch.ones_like(legacy_scale), legacy_scale)
    legacy_codes = torch.round(hidden.float() / legacy_divisor[:, None]).clamp(-128, 127).to(torch.int8)
    return {
        "elements": hidden.numel(),
        "max_distance_in_quantization_levels": float(distance.max()),
        "different_from_legacy_fp32_reference": int((codes != legacy_codes).sum()),
        "exact_half_tie_elements": int(((ideal - ideal.round()).abs() == 0.5).sum()),
    }


def make_exact_quantized_input(
    rows: int, hidden_size: int, seed: int, dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create independent known codes using exactly representable dyadic data."""
    if rows <= 0 or hidden_size < 2 or dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("exact quantization fixture requires positive rows, width >= 2 and BF16/FP16")
    codes = torch.randint(-96, 97, (rows, hidden_size), generator=torch.Generator().manual_seed(seed), dtype=torch.int8)
    codes[:, 0] = 127
    codes[:, 1] = -127
    scales = torch.full((rows,), 1.0 / 32, dtype=torch.float32)
    hidden = (codes.float() / 32).to(dtype)
    return hidden, codes, scales
