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

"""Check learned residual mixing against an FP64 reference without model loading."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
import torch
import torch.nn.functional as F


def _load_residual() -> Callable:
    source = Path(__file__).parents[2] / "xllm/python/models/kimi_k3_text.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_attention_residual"
    )
    namespace = {"torch": torch, "F": F, "ops": SimpleNamespace(rms_norm=_rms_norm)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.dtype == weight.dtype == torch.float32
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


@pytest.mark.parametrize("tokens,blocks", [(1, 0), (1, 1), (1, 8), (7, 4), (7, 13)])
@pytest.mark.parametrize("amplitude", [0.001, 1.0, 100.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_residual_matches_fp64(tokens: int, blocks: int, amplitude: float, dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(916)
    hidden = 128
    values = (torch.randn(tokens, blocks + 1, hidden, generator=generator) * amplitude).to(dtype)
    norm = SimpleNamespace(weight=(1 + torch.randn(hidden, generator=generator) * 0.1).to(dtype), eps=1e-5)
    projection = SimpleNamespace(weight=(torch.randn(1, hidden, generator=generator) / hidden**0.5).to(dtype))
    before = values.clone()
    result = _load_residual()(values[:, -1], values[:, :-1], projection, norm)
    x = values.double()
    normalized = x / torch.sqrt(x.square().mean(-1, keepdim=True) + norm.eps)
    scores = (normalized * norm.weight.double() * projection.weight.double()).sum(-1)
    reference = (scores.softmax(-1).unsqueeze(-1) * x).sum(1).to(dtype)
    torch.testing.assert_close(result, reference, rtol=2e-5, atol=amplitude * 2e-6)
    torch.testing.assert_close(values, before, rtol=0, atol=0)
    assert result.shape == (tokens, hidden)
    assert result.dtype == dtype
