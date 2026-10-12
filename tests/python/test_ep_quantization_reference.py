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

"""Keep quantum-boundary allowances separate from expert-output tolerances."""

import runpy
from pathlib import Path

import pytest
import torch

_REFERENCE = runpy.run_path(str(Path(__file__).parents[1] / "core/kernels/npu/quantization_reference.py"))


@pytest.mark.parametrize("positive", [63, 64])
@pytest.mark.parametrize("negative", [-63, -64])
def test_half_ties_allow_either_nearest_code(positive: int, negative: int) -> None:
    hidden = torch.tensor([[13.0, 6.5, -6.5, 0.0]], dtype=torch.bfloat16)
    codes = torch.tensor([[127, positive, negative, 0]], dtype=torch.int8)
    scale = torch.tensor([13.0 / 127], dtype=torch.float32)
    report = _REFERENCE["assert_input_quantization"](hidden, codes, scale)
    assert report["max_distance_in_quantization_levels"] == 0.5


@pytest.mark.parametrize("bad_code", [62, 65])
def test_rejects_more_than_half_level_error(bad_code: int) -> None:
    hidden = torch.tensor([[13.0, 6.5]], dtype=torch.bfloat16)
    codes = torch.tensor([[127, bad_code]], dtype=torch.int8)
    with pytest.raises(AssertionError, match="non-nearest"):
        _REFERENCE["assert_input_quantization"](hidden, codes, torch.tensor([13.0 / 127]))


def test_one_level_difference_is_not_blanket_accepted() -> None:
    hidden = torch.tensor([[127.0, 63.0]], dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="non-nearest"):
        _REFERENCE["assert_input_quantization"](hidden, torch.tensor([[127, 64]], dtype=torch.int8), torch.ones(1))


def test_wrong_scale_is_rejected_even_when_codes_match() -> None:
    hidden = torch.tensor([[127.0, 63.0]], dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        _REFERENCE["assert_input_quantization"](
            hidden, torch.tensor([[127, 63]], dtype=torch.int8), torch.tensor([1.01])
        )


def test_zero_row_requires_zero_codes_and_scale() -> None:
    _REFERENCE["assert_input_quantization"](
        torch.zeros(1, 4, dtype=torch.bfloat16), torch.zeros(1, 4, dtype=torch.int8), torch.zeros(1)
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [128, 4096])
def test_known_input_codes_are_independent_and_exact(dtype: torch.dtype, width: int) -> None:
    hidden, codes, scales = _REFERENCE["make_exact_quantized_input"](4, width, 44170918, dtype)
    report = _REFERENCE["assert_input_quantization"](hidden, codes, scales)
    assert report["max_distance_in_quantization_levels"] == 0.0
    assert report["different_from_legacy_fp32_reference"] == 0
    assert report["exact_half_tie_elements"] == 0
