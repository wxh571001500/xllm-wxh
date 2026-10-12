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

"""Real-device checks for MLA projections consuming TND inputs directly."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="MLA projection requires an Ascend NPU")


@pytest.fixture
def projection(monkeypatch: pytest.MonkeyPatch) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    monkeypatch.setattr(torch.ops.xllm_ops, "reshape_paged_cache", None, raising=False)
    monkeypatch.setattr(torch.ops.xllm_ops, "update_decode_graph_metadata", None, raising=False)
    module_path = Path(__file__).parents[2] / "xllm/python/kernels_npu/attention.py"
    spec = importlib.util.spec_from_file_location("_mla_projection_attention", module_path)
    assert spec is not None and spec.loader is not None
    attention = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attention)
    return attention.batch_matmul_transpose


def _inputs(layout: str, tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heads = 4
    if layout == "query":
        backing = torch.randn(tokens, heads, 256, dtype=torch.bfloat16, device="npu")
        x = backing.split((192, 64), dim=-1)[0]
        weight = torch.randn(heads, 192, 512, dtype=torch.bfloat16, device="npu")
    else:
        backing = torch.randn(tokens, heads * 2, 512, dtype=torch.bfloat16, device="npu")
        x = backing.narrow(1, heads, heads)
        weight = torch.randn(heads, 512, 256, dtype=torch.bfloat16, device="npu")
    return backing, x, weight


def _check_projection(x: torch.Tensor, weight: torch.Tensor, actual: torch.Tensor) -> None:
    reference = torch.einsum("thd,hdo->tho", x.cpu().float(), weight.cpu().float())
    assert actual.shape == reference.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.cpu().float(), reference, rtol=1e-2, atol=2e-2)


@pytest.mark.parametrize("layout", ("query", "value"))
@pytest.mark.parametrize("tokens", (1, 4, 32))
def test_mla_projection_noncontiguous_input(
    layout: str,
    tokens: int,
    projection: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    _, x, weight = _inputs(layout, tokens)
    _check_projection(x, weight, projection(x, weight))


@pytest.mark.parametrize("layout", ("query", "value"))
@pytest.mark.parametrize("tokens", (1, 4, 32))
def test_mla_projection_graph_replay_uses_current_values(
    layout: str,
    tokens: int,
    projection: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    backing, x, weight = _inputs(layout, tokens)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = projection(x, weight)
    for scale in (0.5, -1.0):
        backing.mul_(scale)
        graph.replay()
        torch.npu.synchronize()
        _check_projection(x, weight, actual)
