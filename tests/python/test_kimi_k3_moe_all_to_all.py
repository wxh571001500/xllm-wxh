# Copyright 2026 The xLLM Authors.
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

"""Verify the Kimi-private eager all-to-all adapter."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from xllm.python.models import kimi_k3_moe as dispatcher

from .test_moe_all_to_all import _load_module


def _set_group(monkeypatch: pytest.MonkeyPatch, size: int) -> None:
    group = SimpleNamespace(size=lambda: size)
    monkeypatch.setattr(dispatcher.distributed, "moe_ep_group", lambda device: group, raising=False)


@pytest.mark.parametrize("rows", [0, 3])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_all_to_all_single_rank_returns_independent_storage(
    monkeypatch: pytest.MonkeyPatch, rows: int, asynchronous: bool
) -> None:
    _set_group(monkeypatch, 1)
    value = torch.arange(rows * 4).reshape(rows, 4)
    if asynchronous:
        actual, handle = dispatcher._async_all_to_all(value, [rows], [rows], "moe_ep")
        assert handle is None
    else:
        actual = dispatcher._all_to_all(value, [rows], [rows], "moe_ep")
    torch.testing.assert_close(actual, value)
    if rows:
        assert actual.data_ptr() != value.data_ptr()


@pytest.mark.parametrize("outputs,inputs", [([0, -1], [0, 0]), ([0, 0], [-1, 1]), ([0], [0, 0])])
def test_all_to_all_rejects_invalid_splits(
    monkeypatch: pytest.MonkeyPatch, outputs: list[int], inputs: list[int]
) -> None:
    _set_group(monkeypatch, 2)
    with pytest.raises(ValueError, match="splits"):
        dispatcher._all_to_all(torch.empty(0, 4), outputs, inputs, "moe_ep")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_all_to_all_single_rank_rejects_mismatched_splits(monkeypatch: pytest.MonkeyPatch, asynchronous: bool) -> None:
    _set_group(monkeypatch, 1)
    exchange = dispatcher._async_all_to_all if asynchronous else dispatcher._all_to_all
    with pytest.raises(ValueError, match="must match"):
        exchange(torch.ones(2, 4), [1], [2], "moe_ep")


def test_all_to_all_makes_strided_inputs_contiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_group(monkeypatch, 2)
    value = torch.arange(24).reshape(4, 6)[:, ::2]

    def exchange(output: torch.Tensor, source: torch.Tensor, **kwargs: object) -> None:
        assert source.is_contiguous()
        assert kwargs["input_split_sizes"] == [3, 1]
        assert kwargs["output_split_sizes"] == [1, 3]
        output.copy_(source.flip(0))

    monkeypatch.setattr(dist, "all_to_all_single", exchange)
    torch.testing.assert_close(dispatcher._all_to_all(value, [1, 3], [3, 1], "moe_ep"), value.flip(0))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_all_to_all_preserves_uninitialized_group_error(monkeypatch: pytest.MonkeyPatch, asynchronous: bool) -> None:
    # CPU conftest stubs the distributed package; load its real owner module
    # as the existing Gloo transport tests do, without changing that stub.
    collectives = _load_module(
        "_kimi_moe_collectives_contract", Path(__file__).parents[2] / "xllm/python/distributed/collectives.py"
    )

    monkeypatch.setattr(collectives, "_groups", {})
    monkeypatch.setattr(dispatcher.distributed, "moe_ep_group", collectives.moe_ep_group, raising=False)
    exchange = dispatcher._async_all_to_all if asynchronous else dispatcher._all_to_all
    with pytest.raises(RuntimeError, match="not initialized"):
        exchange(torch.ones(1, 4), [1], [1], "moe_ep")


def test_async_all_to_all_forwards_group_and_returns_work(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_group(monkeypatch, 2)
    value = torch.arange(24).reshape(4, 6)[:, ::2]
    handle = object()

    def exchange(output: torch.Tensor, source: torch.Tensor, **kwargs: object) -> object:
        assert source.is_contiguous()
        assert kwargs["input_split_sizes"] == [3, 1]
        assert kwargs["output_split_sizes"] == [1, 3]
        assert kwargs["group"].size() == 2
        assert kwargs["async_op"] is True
        output.copy_(source.flip(0))
        return handle

    monkeypatch.setattr(dist, "all_to_all_single", exchange)
    actual, work = dispatcher._async_all_to_all(value, [1, 3], [3, 1], "moe_ep")
    assert work is handle
    torch.testing.assert_close(actual, value.flip(0))
