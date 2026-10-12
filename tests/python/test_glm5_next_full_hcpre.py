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

import pytest
import torch

from xllm.python.models import glm5_next


@pytest.fixture
def full_hc_pre(monkeypatch: pytest.MonkeyPatch) -> list[tuple[torch.Tensor, ...]]:
    calls = []

    def run(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        hc_mult: int,
        iterations: int,
        norm_eps: float,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append((weight, scale, base))
        assert weight.dtype == scale.dtype == base.dtype == torch.float32
        assert hc_mult == 4 and iterations == 20
        assert norm_eps == 1e-5 and hc_eps == 1e-6
        return hidden[:, :, 0], scale, base

    monkeypatch.setattr(glm5_next.kernels, "hc_pre_fused", run, raising=False)
    return calls


def _layer() -> glm5_next.Glm5NextHyperConnection:
    config = glm5_next.Glm5NextConfig(hidden_size=16)
    layer = glm5_next.Glm5NextHyperConnection(config, torch.bfloat16, torch.device("cpu")).to(torch.bfloat16)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(0.123456)
    return layer


def _forward(layer: glm5_next.Glm5NextHyperConnection) -> None:
    layer(torch.zeros(1, 4, 4, 16, dtype=layer.fn.dtype))


def test_full_hc_pre_requires_native_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(glm5_next.kernels, "hc_pre_fused", raising=False)
    with pytest.raises(RuntimeError, match="requires native hc_pre_fused support"):
        _layer()


@torch.inference_mode()
def test_full_hc_pre_requires_prepared_weights(full_hc_pre: list[tuple[torch.Tensor, ...]]) -> None:
    with pytest.raises(RuntimeError, match="prepared"):
        _forward(_layer())


@torch.inference_mode()
def test_full_hc_pre_caches_rounded_nonpersistent_weight(full_hc_pre: list[tuple[torch.Tensor, ...]]) -> None:
    layer = _layer()
    layer.process_weights_after_loading()
    _forward(layer)
    _forward(layer)
    weight = full_hc_pre[-1][0]
    assert weight is full_hc_pre[-2][0]
    assert weight.data_ptr() != layer.fn.data_ptr()
    assert set(layer.state_dict()) == {"fn", "scale", "base"}
    torch.testing.assert_close(weight, layer.fn.float(), rtol=0, atol=0)


@pytest.mark.parametrize("assign", [False, True])
@torch.inference_mode()
def test_full_hc_pre_reload_keeps_graph_addresses(full_hc_pre: list[tuple[torch.Tensor, ...]], assign: bool) -> None:
    layer = _layer()
    layer.process_weights_after_loading()
    _forward(layer)
    before = full_hc_pre[-1]
    state = {name: torch.full_like(value, 0.654321) for name, value in layer.state_dict().items()}
    layer.load_state_dict(state, assign=assign)
    _forward(layer)
    for name, previous, actual in zip(("fn", "scale", "base"), before, full_hc_pre[-1], strict=True):
        assert previous is actual
        torch.testing.assert_close(actual, getattr(layer, name).float(), rtol=0, atol=0)


@torch.inference_mode()
def test_full_hc_pre_dtype_move_refreshes_weight(full_hc_pre: list[tuple[torch.Tensor, ...]]) -> None:
    layer = _layer()
    layer.process_weights_after_loading()
    layer.to(torch.float32)
    layer.fn.fill_(0.7654321)
    layer.to(torch.bfloat16)
    _forward(layer)
    torch.testing.assert_close(full_hc_pre[-1][0], layer.fn.float(), rtol=0, atol=0)


@torch.inference_mode()
def test_full_hc_pre_forward_has_no_weight_cast(full_hc_pre: list[tuple[torch.Tensor, ...]]) -> None:
    layer = _layer()
    layer.process_weights_after_loading()
    hidden = torch.zeros(1, 4, 4, 16, dtype=torch.bfloat16)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        layer(hidden)
    assert not {event.key for event in trace.key_averages()} & {"aten::to", "aten::_to_copy", "aten::copy_"}
