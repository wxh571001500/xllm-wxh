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

"""CPU-mocked contracts for the NPU EP2 W8A8 MoE helper."""

from __future__ import annotations

import gc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

_REPO_ROOT = Path(__file__).parents[2]


def _load_npu_moe_module() -> ModuleType:
    gc.collect()
    path = _REPO_ROOT / "xllm/python/kernels_npu/moe.py"
    spec = importlib.util.spec_from_file_location("ep_npu_moe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # These are CPU call-contract tests, never real NPU operator tests. Avoid
    # importing the vendor runtime even when it happens to be installed.
    with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
        spec.loader.exec_module(module)
    return module


def _make_inputs() -> dict[str, Any]:
    hidden = torch.arange(48, dtype=torch.bfloat16).reshape(2, 3, 8).transpose(0, 1)
    topk_weights = (
        torch.tensor(
            [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
            ],
            dtype=torch.float32,
        )
        .reshape(2, 3, 2)
        .transpose(0, 1)
    )
    topk_ids = torch.arange(12, dtype=torch.int64).reshape(2, 3, 2).transpose(0, 1) % 8
    active_mask = torch.tensor(
        [True, False, False, True, True, False, False, True, True, False, False, True],
        dtype=torch.bool,
    )[::2]
    return {
        "hidden": hidden,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "w13": torch.empty(4, 8, 12, dtype=torch.int8),
        "w2": torch.empty(4, 6, 8, dtype=torch.int8),
        "w13_scale": torch.ones(4, 12, dtype=torch.float32),
        "w2_scale": torch.full((4, 8), 2.0, dtype=torch.float32),
        "group_ep": "ep_group",
        "ep_size": 2,
        "ep_rank": 1,
        "num_experts": 8,
        "global_bs": 0,
        "x_active_mask": active_mask,
        "swiglu_limit": 7.5,
    }


def _call_impl(moe: ModuleType, inputs: dict[str, Any]) -> torch.Tensor:
    return moe._ep_moe_w8a8_impl(**inputs)


@pytest.mark.parametrize("send,recv", [([2, 0], [0, 3]), ([0, 0], [0, 0]), ([1, 2], [2, 1])])
def test_alltoall_empty_edges_have_transport_rows_but_no_expert_rows(
    send: list[int], recv: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    moe = _load_npu_moe_module()
    value = torch.arange(sum(send) * 4).reshape(-1, 4)
    expected = torch.arange(sum(max(1, n) for n in recv) * 4).reshape(-1, 4)

    def exchange(out: torch.Tensor, src: torch.Tensor, **kwargs: Any) -> None:
        assert kwargs["input_split_sizes"] == [max(1, n) for n in send]
        assert kwargs["output_split_sizes"] == [max(1, n) for n in recv]
        offset = source_offset = 0
        for count in send:
            if count:
                torch.testing.assert_close(src[offset : offset + count], value[source_offset : source_offset + count])
            else:
                assert not torch.count_nonzero(src[offset])
            offset += max(1, count)
            source_offset += count
        out.copy_(expected)

    monkeypatch.setattr(torch.distributed, "all_to_all_single", exchange)
    actual = moe._alltoall_variable_rows(value, send, recv, object())
    pieces = []
    offset = 0
    for count in recv:
        pieces.append(expected[offset : offset + count])
        offset += max(1, count)
    torch.testing.assert_close(actual, torch.cat(pieces))


@pytest.mark.parametrize("has_active_source", (True, False))
def test_ep_moe_w8a8_preserves_dispatch_compute_combine_contract(
    monkeypatch: pytest.MonkeyPatch,
    has_active_source: bool,
) -> None:
    from xllm.python import kernels

    moe = _load_npu_moe_module()
    inputs = _make_inputs()
    hidden = inputs["hidden"]
    topk_weights = inputs["topk_weights"]
    topk_ids = inputs["topk_ids"]
    active_mask = inputs["x_active_mask"]
    assert isinstance(hidden, torch.Tensor)
    assert isinstance(topk_weights, torch.Tensor)
    assert isinstance(topk_ids, torch.Tensor)
    assert isinstance(active_mask, torch.Tensor)
    assert not hidden.is_contiguous()
    assert not topk_weights.is_contiguous()
    assert not topk_ids.is_contiguous()
    assert not active_mask.is_contiguous()
    if not has_active_source:
        active_mask.zero_()

    expand_x = torch.empty(9, 8, dtype=hidden.dtype)
    dynamic_scale = torch.arange(1, 10, dtype=torch.float32)
    assist_info = torch.arange(16, dtype=torch.int32)
    expert_token_nums = torch.tensor([2, 0, 4, 3], dtype=torch.int64)
    ep_recv_counts = torch.tensor([5, 4, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    tp_recv_counts = torch.empty(0, dtype=torch.int32)
    expand_scales = torch.arange(9, dtype=torch.float32)
    quantized_expand_x = torch.empty(9, 8, dtype=torch.int8)
    input_scale = torch.arange(21, 30, dtype=torch.float32)
    gemm1_out = torch.empty(9, 12, dtype=torch.int32)
    act_i8 = torch.empty(9, 6, dtype=torch.int8)
    act_scale = torch.arange(11, 20, dtype=torch.float32)
    expert_output = torch.empty(9, 8, dtype=torch.bfloat16)
    combine_output = torch.arange(48, dtype=torch.bfloat16).reshape(6, 8)
    if not has_active_source:
        combine_output.zero_()
    call_order: list[str] = []
    dispatch_calls: list[dict[str, Any]] = []
    gemm_calls: list[dict[str, Any]] = []
    dequant_calls: list[dict[str, Any]] = []
    combine_calls: list[dict[str, Any]] = []

    def dispatch(**kwargs: Any) -> tuple[torch.Tensor, ...]:
        call_order.append("dispatch")
        dispatch_calls.append(kwargs)
        return (
            expand_x,
            dynamic_scale,
            assist_info,
            expert_token_nums,
            ep_recv_counts,
            tp_recv_counts,
            expand_scales,
        )

    def dynamic_quant(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        call_order.append("dynamic_quant")
        assert value is expand_x
        return quantized_expand_x, input_scale

    def group_gemm(**kwargs: Any) -> torch.Tensor:
        call_order.append(f"gemm{len(gemm_calls) + 1}")
        gemm_calls.append(kwargs)
        return gemm1_out if len(gemm_calls) == 1 else expert_output

    def dequant_swiglu_quant(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        call_order.append("dequant_swiglu_quant")
        dequant_calls.append(kwargs)
        return act_i8, act_scale

    def combine(**kwargs: Any) -> torch.Tensor:
        call_order.append("combine")
        combine_calls.append(kwargs)
        return combine_output

    forbidden_routing = MagicMock(side_effect=AssertionError("local routing must not run in EP2 helper"))
    forbidden_unpermute = MagicMock(side_effect=AssertionError("local unpermute must not run in EP2 helper"))
    forbidden_all_reduce = MagicMock(side_effect=AssertionError("EP2 helper must not all-reduce"))
    monkeypatch.setattr(moe.torch_npu, "npu_moe_distribute_dispatch_v2", dispatch, raising=False)
    monkeypatch.setattr(moe.torch_npu, "npu_moe_distribute_combine_v2", combine, raising=False)
    monkeypatch.setattr(moe.torch_npu, "npu_moe_init_routing_v2", forbidden_routing, raising=False)
    monkeypatch.setattr(moe.torch_npu, "npu_moe_token_unpermute", forbidden_unpermute, raising=False)
    monkeypatch.setattr(moe, "_group_gemm", group_gemm)
    monkeypatch.setattr(kernels, "dynamic_quant", dynamic_quant, raising=False)
    monkeypatch.setattr(kernels, "dequant_swiglu_quant", dequant_swiglu_quant, raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", forbidden_all_reduce)

    result = _call_impl(moe, inputs)

    assert call_order == ["dispatch", "dynamic_quant", "gemm1", "dequant_swiglu_quant", "gemm2", "combine"]
    assert forbidden_routing.call_count == 0
    assert forbidden_unpermute.call_count == 0
    assert forbidden_all_reduce.call_count == 0
    assert result.shape == hidden.shape
    assert result.dtype == hidden.dtype
    torch.testing.assert_close(result.reshape(-1, 8), combine_output)

    assert len(dispatch_calls) == 1
    dispatch_kwargs = dispatch_calls[0]
    normalized_hidden = dispatch_kwargs["x"]
    normalized_ids = dispatch_kwargs["expert_ids"]
    assert dispatch_kwargs["expert_scales"] is None
    normalized_weights = combine_calls[0]["expert_scales"]
    normalized_mask = dispatch_kwargs["x_active_mask"]
    assert normalized_hidden.shape == (6, 8)
    assert normalized_hidden.dtype == torch.bfloat16
    assert normalized_hidden.is_contiguous()
    assert normalized_ids.shape == (6, 2)
    assert normalized_ids.dtype == torch.int32
    assert normalized_ids.is_contiguous()
    assert normalized_weights.shape == (6, 2)
    assert normalized_weights.dtype == torch.float32
    assert normalized_weights.is_contiguous()
    assert normalized_mask.dtype == torch.bool
    assert normalized_mask.is_contiguous()
    assert normalized_mask is not active_mask
    torch.testing.assert_close(normalized_hidden, hidden.reshape(-1, 8))
    torch.testing.assert_close(normalized_ids, topk_ids.reshape(-1, 2).to(torch.int32))
    expected_weights = topk_weights.reshape(-1, 2).to(hidden.dtype).float()
    torch.testing.assert_close(normalized_weights, expected_weights, rtol=0, atol=0)
    assert not torch.equal(normalized_weights, topk_weights.reshape(-1, 2).float())
    torch.testing.assert_close(normalized_mask, active_mask)
    assert dispatch_kwargs["scales"] is None
    assert dispatch_kwargs["group_ep"] == "ep_group"
    assert dispatch_kwargs["ep_world_size"] == 2
    assert dispatch_kwargs["ep_rank_id"] == 1
    assert dispatch_kwargs["moe_expert_num"] == 8
    assert dispatch_kwargs["tp_world_size"] == 0
    assert dispatch_kwargs["tp_rank_id"] == 0
    assert dispatch_kwargs["expert_shard_type"] == 0
    assert dispatch_kwargs["shared_expert_num"] == 1
    assert dispatch_kwargs["shared_expert_rank_num"] == 0
    assert dispatch_kwargs["quant_mode"] == 0
    assert dispatch_kwargs["expert_token_nums_type"] == 1
    assert dispatch_kwargs["global_bs"] == 0

    assert len(gemm_calls) == 2
    first_gemm, second_gemm = gemm_calls
    assert first_gemm["x"] is quantized_expand_x
    assert first_gemm["weight"] is inputs["w13"]
    assert first_gemm["scale"] is None
    assert first_gemm["per_token_scale"] is None
    assert first_gemm["output_dtype"] == torch.int32
    assert second_gemm["x"] is act_i8
    assert second_gemm["weight"] is inputs["w2"]
    assert second_gemm["per_token_scale"] is act_scale
    assert second_gemm["output_dtype"] == hidden.dtype
    assert all(call["group_list_type"] == 1 for call in gemm_calls)
    assert all(call["group_list"] is expert_token_nums for call in gemm_calls)
    torch.testing.assert_close(second_gemm["scale"], inputs["w2_scale"].to(hidden.dtype))

    assert len(dequant_calls) == 1
    dequant_kwargs = dequant_calls[0]
    assert dequant_kwargs["x"] is gemm1_out
    assert dequant_kwargs["weight_scale"] is inputs["w13_scale"]
    assert dequant_kwargs["activation_scale"] is input_scale
    assert dequant_kwargs["group_index"] is expert_token_nums
    assert dequant_kwargs["activate_left"] is True
    assert dequant_kwargs["quant_mode"] == 1
    assert dequant_kwargs["swiglu_mode"] == 1
    assert dequant_kwargs["clamp_limit"] == 7.5
    assert dequant_kwargs["glu_alpha"] == 1.0
    assert dequant_kwargs["glu_bias"] == 0.0

    assert len(combine_calls) == 1
    combine_kwargs = combine_calls[0]
    assert combine_kwargs["expand_x"] is expert_output
    assert combine_kwargs["expert_ids"] is normalized_ids
    assert combine_kwargs["expert_scales"] is normalized_weights
    assert combine_kwargs["assist_info_for_combine"] is assist_info
    assert combine_kwargs["ep_send_counts"] is ep_recv_counts
    assert combine_kwargs["tp_send_counts"] is tp_recv_counts
    assert combine_kwargs["expand_scales"] is None
    assert combine_kwargs["x_active_mask"] is normalized_mask
    assert combine_kwargs["shared_expert_x"] is None
    assert combine_kwargs["group_ep"] == dispatch_kwargs["group_ep"]
    assert combine_kwargs["ep_world_size"] == dispatch_kwargs["ep_world_size"]
    assert combine_kwargs["ep_rank_id"] == dispatch_kwargs["ep_rank_id"]
    assert combine_kwargs["moe_expert_num"] == dispatch_kwargs["moe_expert_num"]
    assert combine_kwargs["global_bs"] == dispatch_kwargs["global_bs"]
    assert combine_kwargs["comm_quant_mode"] == 0


def test_ep_moe_w8a8_runs_empty_local_expert_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xllm.python import kernels

    moe = _load_npu_moe_module()
    inputs = _make_inputs()
    inputs["hidden"] = torch.empty(2, 8, dtype=torch.float16)
    inputs["topk_weights"] = torch.ones(2, 2, dtype=torch.float16)
    inputs["topk_ids"] = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    inputs["x_active_mask"] = None
    expand_x = torch.empty(0, 8, dtype=torch.float16)
    expert_token_nums = torch.zeros(4, dtype=torch.int64)
    metadata = (
        torch.empty(0, dtype=torch.float32),
        torch.empty(0, dtype=torch.int32),
        expert_token_nums,
        torch.zeros(8, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.float32),
    )
    quantized = torch.empty(0, 8, dtype=torch.int8)
    input_scale = torch.empty(0, dtype=torch.float32)
    gemm1_out = torch.empty(0, 12, dtype=torch.int32)
    act_i8 = torch.empty(0, 6, dtype=torch.int8)
    act_scale = torch.empty(0, dtype=torch.float32)
    expert_output = torch.empty(0, 8, dtype=torch.float16)
    expected = torch.ones(2, 8, dtype=torch.float16)
    dispatch = MagicMock(return_value=(expand_x, *metadata))
    dynamic_quant = MagicMock(return_value=(quantized, input_scale))
    group_gemm = MagicMock(side_effect=(gemm1_out, expert_output))
    dequant = MagicMock(return_value=(act_i8, act_scale))
    combine = MagicMock(return_value=expected)
    monkeypatch.setattr(moe.torch_npu, "npu_moe_distribute_dispatch_v2", dispatch, raising=False)
    monkeypatch.setattr(moe.torch_npu, "npu_moe_distribute_combine_v2", combine, raising=False)
    monkeypatch.setattr(moe, "_group_gemm", group_gemm)
    monkeypatch.setattr(kernels, "dynamic_quant", dynamic_quant, raising=False)
    monkeypatch.setattr(kernels, "dequant_swiglu_quant", dequant, raising=False)

    result = _call_impl(moe, inputs)

    torch.testing.assert_close(result, expected)
    dispatch.assert_called_once()
    dynamic_quant.assert_not_called()
    group_gemm.assert_not_called()
    dequant.assert_not_called()
    combine.assert_called_once()
    assert combine.call_args.kwargs["expand_x"].shape == (0, 8)


def test_ep_moe_w8a8_propagates_clamped_activation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xllm.python import kernels

    moe = _load_npu_moe_module()
    inputs = _make_inputs()
    expanded = torch.empty(1, 8, dtype=torch.bfloat16)
    dispatch = MagicMock(
        return_value=(
            expanded,
            torch.ones(1),
            torch.empty(0, dtype=torch.int32),
            torch.tensor([1, 0, 0, 0], dtype=torch.int64),
            torch.zeros(8, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0),
        )
    )
    dequant = MagicMock(side_effect=RuntimeError("clamped activation failed"))
    combine = MagicMock()
    group_gemm = MagicMock(return_value=torch.empty(1, 12, dtype=torch.int32))
    monkeypatch.setattr(moe.torch_npu, "npu_moe_distribute_dispatch_v2", dispatch, raising=False)
    monkeypatch.setattr(moe.torch_npu, "npu_moe_distribute_combine_v2", combine, raising=False)
    monkeypatch.setattr(moe, "_group_gemm", group_gemm)
    monkeypatch.setattr(
        kernels,
        "dynamic_quant",
        MagicMock(return_value=(torch.empty(1, 8, dtype=torch.int8), torch.ones(1))),
        raising=False,
    )
    monkeypatch.setattr(kernels, "dequant_swiglu_quant", dequant, raising=False)

    with pytest.raises(RuntimeError, match="clamped activation failed"):
        _call_impl(moe, inputs)

    dispatch.assert_called_once()
    group_gemm.assert_called_once()
    dequant.assert_called_once()
    assert dequant.call_args.kwargs["clamp_limit"] == inputs["swiglu_limit"]
    combine.assert_not_called()


@pytest.mark.parametrize("has_active_source", (True, False))
def test_alltoall_uses_canonical_quantizer_but_transports_int8(
    monkeypatch: pytest.MonkeyPatch, has_active_source: bool
) -> None:
    from xllm.python import distributed, kernels

    moe = _load_npu_moe_module()
    inputs = _make_inputs()
    hidden = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    expanded = hidden[[0, 1, 0, 1]]
    codes = torch.arange(32, dtype=torch.int8).reshape(4, 8)
    scales = torch.tensor([0.125, 0.25, 0.5, 1.0])
    counts = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.int64)
    row_map = torch.arange(4, dtype=torch.int32)
    calls: list[str] = []
    group = MagicMock()
    group.size.return_value = 2
    group.rank.return_value = 0
    monkeypatch.setattr(distributed, "moe_ep_group", lambda device: group, raising=False)

    def route(value: torch.Tensor, ids: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        calls.append("route")
        torch.testing.assert_close(value, hidden)
        assert kwargs["quant_mode"] == -1
        assert kwargs["scale"] is None
        return expanded, row_map, counts, torch.empty(0)

    def quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append("quantize")
        assert value is expanded
        return codes, scales

    def exchange_counts(out: torch.Tensor, value: torch.Tensor, **kwargs: Any) -> None:
        assert kwargs["group"] is group
        out.copy_(value)

    def exchange_rows(value: torch.Tensor, send: list[int], recv: list[int], comm: Any) -> torch.Tensor:
        assert comm is group and send == recv
        if value.shape[1] == 12:
            calls.append("dispatch")
            assert value.dtype == torch.int8
            if has_active_source:
                torch.testing.assert_close(value[:, :8], codes)
                torch.testing.assert_close(value[:, 8:].contiguous().view(torch.float32).flatten(), scales)
            else:
                assert value.shape == (0, 12)
        else:
            calls.append("combine")
        return value

    def experts(q: torch.Tensor, scale: torch.Tensor, groups: torch.Tensor, *args: Any) -> torch.Tensor:
        calls.append("experts")
        order = torch.tensor([0, 2, 1, 3])
        if has_active_source:
            torch.testing.assert_close(q, codes[order])
            torch.testing.assert_close(scale, scales[order])
            torch.testing.assert_close(groups, torch.tensor([2, 2, 0, 0]))
        else:
            assert q.shape == (0, 8) and scale.numel() == 0
        return torch.zeros((q.shape[0], 8), dtype=hidden.dtype)

    def unpermute(**kwargs: Any) -> torch.Tensor:
        calls.append("unpermute")
        torch.testing.assert_close(kwargs["sorted_indices"], row_map)
        return torch.zeros_like(hidden)

    monkeypatch.setattr(moe.torch_npu, "npu_moe_init_routing_v2", route, raising=False)
    monkeypatch.setattr(moe.torch_npu, "npu_moe_token_unpermute", unpermute, raising=False)
    monkeypatch.setattr(kernels, "dynamic_quant", quantize, raising=False)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", exchange_counts)
    monkeypatch.setattr(moe, "_alltoall_variable_rows", exchange_rows)
    monkeypatch.setattr(moe, "_ep_grouped_w8a8", experts)
    for name in ("group_ep", "global_bs"):
        inputs.pop(name)
    inputs.update(
        hidden=hidden,
        topk_ids=torch.tensor([[0, 4], [1, 5]], dtype=torch.int32),
        topk_weights=torch.full((2, 2), 0.5),
        ep_rank=0,
        x_active_mask=torch.full((2,), has_active_source, dtype=torch.bool),
    )
    output = moe._ep_moe_w8a8_alltoall_impl(**inputs)
    torch.testing.assert_close(output, torch.zeros_like(hidden))
    assert calls == (
        ["route", "quantize", "dispatch", "experts", "combine", "unpermute"]
        if has_active_source
        else ["dispatch", "experts", "combine"]
    )


def test_ep_moe_w8a8_fake_preserves_hidden_metadata() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    moe = _load_npu_moe_module()
    with FakeTensorMode():
        hidden = torch.empty(3, 2, 8, dtype=torch.bfloat16)
        output = moe.ep_moe_w8a8(
            hidden,
            torch.empty(3, 2, 2, dtype=torch.float32),
            torch.empty(3, 2, 2, dtype=torch.int32),
            torch.empty(4, 8, 12, dtype=torch.int8),
            torch.empty(4, 6, 8, dtype=torch.int8),
            torch.empty(4, 12, dtype=torch.float32),
            torch.empty(4, 8, dtype=torch.float32),
            "ep_group",
            2,
            0,
            8,
        )

    assert output.shape == hidden.shape
    assert output.dtype == hidden.dtype
    assert output.device == hidden.device


def test_ep_moe_w8a8_validates_tensor_contracts() -> None:
    moe = _load_npu_moe_module()

    inputs = _make_inputs()
    inputs["hidden"] = inputs["hidden"].to(torch.float32)
    with pytest.raises(TypeError, match="bfloat16 or float16"):
        _call_impl(moe, inputs)

    inputs = _make_inputs()
    inputs["topk_ids"] = torch.zeros(5, 2, dtype=torch.int64)
    inputs["topk_weights"] = torch.ones(5, 2, dtype=torch.float32)
    with pytest.raises(ValueError, match="routing token count"):
        _call_impl(moe, inputs)

    inputs = _make_inputs()
    inputs["w13_scale"] = torch.ones(4, 12, dtype=torch.int64)
    with pytest.raises(TypeError, match="ordinary floating-point W8A8 scales"):
        _call_impl(moe, inputs)

    inputs = _make_inputs()
    inputs["x_active_mask"] = inputs["x_active_mask"].to(torch.int8)
    with pytest.raises(TypeError, match="x_active_mask must use bool"):
        _call_impl(moe, inputs)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"w13": torch.empty(4, 7, 12, dtype=torch.int8)}, "weight dimensions"),
        ({"w2": torch.empty(4, 7, 8, dtype=torch.int8)}, "weight dimensions"),
        ({"w13_scale": torch.ones(4, 11)}, "scale shapes"),
        ({"w2_scale": torch.ones(4, 7)}, "scale shapes"),
        ({"x_active_mask": torch.ones(5, dtype=torch.bool)}, "1D with 6 elements"),
        ({"topk_weights": torch.ones(6, 3)}, "shapes must match"),
    ),
)
def test_ep_moe_w8a8_rejects_incompatible_shapes(
    updates: dict[str, Any],
    message: str,
) -> None:
    moe = _load_npu_moe_module()
    inputs = _make_inputs()
    inputs.update(updates)
    with pytest.raises(ValueError, match=message):
        _call_impl(moe, inputs)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"group_ep": ""}, "group_ep must be non-empty"),
        ({"ep_size": 1}, "ep_size must be greater than one"),
        ({"ep_rank": 2}, "ep_rank 2 is outside"),
        ({"num_experts": 7}, "num_experts must be positive and divisible"),
        ({"global_bs": -1}, "global_bs=0"),
        ({"global_bs": 24}, "global_bs=0"),
    ),
)
def test_ep_moe_w8a8_validates_ep_metadata(updates: dict[str, Any], message: str) -> None:
    moe = _load_npu_moe_module()
    inputs = _make_inputs()
    inputs.update(updates)

    with pytest.raises(ValueError, match=message):
        _call_impl(moe, inputs)
