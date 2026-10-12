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

"""Exercise production routing, SiTU and token combination across EP owners."""

import pytest
import torch

from xllm.python.models.kimi_k3_moe import (
    FusedQuantizedSituAndMul,
    FusedW4A8RoutedExperts,
    GroupedTopKRouter,
    MoERouterConfig,
    MoETokenDispatchInput,
    NativeTokenDispatcher,
    SituAndMul,
)


def test_grouped_router_rejects_insufficient_candidates() -> None:
    with pytest.raises(ValueError, match="selected groups"):
        GroupedTopKRouter(MoERouterConfig(4, 2, "sigmoid", True, 1.0, True, 4, 1))


def test_grouped_router_candidate_limit_keeps_selected_experts() -> None:
    router = GroupedTopKRouter(MoERouterConfig(4, 2, "sigmoid", True, 1.0, True, 2, 1))
    result = router.select_experts(torch.zeros(1, 2), torch.tensor([[4.0, 3.0, -2.0, -3.0]]))
    assert set(result.topk_ids[0].tolist()) == {0, 1}


@pytest.mark.parametrize("ep_size", [1, 4, 64])
@pytest.mark.parametrize("tokens", [1, 7])
def test_partitioned_routing_and_situ_match_token_reference(ep_size: int, tokens: int) -> None:
    torch.manual_seed(20261009)
    num_experts, hidden_size, intermediate, top_k = 896, 8, 8, 4
    hidden = torch.randn(tokens, hidden_size)
    gate = torch.randn(num_experts, hidden_size)
    w13 = torch.randn(num_experts, 2 * intermediate, hidden_size) * 0.1
    w2 = torch.randn(num_experts, hidden_size, intermediate) * 0.1
    bias = torch.randn(num_experts) * 0.1
    config = MoERouterConfig(num_experts, top_k, "sigmoid", True, 2.5, True, 8, 2)
    router = GroupedTopKRouter(config)
    routing = router.select_experts(hidden, hidden @ gate.t(), bias)
    activation = SituAndMul(4.0, 25.0)

    # Independent token-major reference: bias affects selection only.
    scores = torch.sigmoid(hidden @ gate.t())
    torch.testing.assert_close(
        routing.topk_weights,
        scores.gather(1, routing.topk_ids.long())
        / scores.gather(1, routing.topk_ids.long()).sum(1, keepdim=True)
        * 2.5,
    )
    reference = torch.zeros_like(hidden)
    for token in range(tokens):
        for route in range(top_k):
            expert = int(routing.topk_ids[token, route])
            gate_up = hidden[token] @ w13[expert].t()
            g, u = gate_up.chunk(2)
            value = (4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)) * (25.0 * torch.tanh(u / 25.0))
            reference[token] += routing.topk_weights[token, route] * (value @ w2[expert].t())

    combined = torch.zeros_like(hidden)
    local_experts = num_experts // ep_size
    owners_without_tokens = 0
    routed_rows = 0
    for owner in range(ep_size):
        first = owner * local_experts
        dispatcher = NativeTokenDispatcher(num_experts, first, local_experts)
        dispatch = dispatcher.token_dispatch(MoETokenDispatchInput(hidden, routing))
        assert dispatch.group_list.shape == (local_experts,)
        assert int(dispatch.group_list.sum()) == dispatch.hidden_states.shape[0]
        owners_without_tokens += int(dispatch.hidden_states.shape[0] == 0)
        routed_rows += dispatch.hidden_states.shape[0]
        outputs = []
        start = 0
        for local, count in enumerate(dispatch.group_list.tolist()):
            expert = first + local
            rows = dispatch.hidden_states[start : start + count]
            if count:
                outputs.append(activation(rows @ w13[expert].t()) @ w2[expert].t())
            start += count
        partial = torch.cat(outputs) if outputs else hidden.new_empty((0, hidden_size))
        combined += dispatcher.token_combine(partial, dispatch.combine_metadata)
    assert routed_rows == tokens * top_k
    if ep_size == 64:
        assert owners_without_tokens >= ep_size - tokens * top_k
    torch.testing.assert_close(combined, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("group_list_type", [0, 1])
def test_unquantized_experts_use_grouped_matmul_weight_layout(
    monkeypatch: pytest.MonkeyPatch, group_list_type: int
) -> None:
    from xllm.python.models import kimi_k3_moe as experts_module
    from xllm.python.models.kimi_k3_moe import MoEExpertsConfig, MoETokenDispatchOutput

    torch.manual_seed(20261009)
    config = MoEExpertsConfig(3, 8, 16, 1, 0)
    expert = experts_module.UnquantizedRoutedExperts(config, SituAndMul(4.0, 25.0), torch.float32, torch.device("cpu"))
    hidden = torch.randn(3, 8)
    counts = torch.tensor([2, 0, 1])
    groups = counts.cumsum(0) if group_list_type == 0 else counts
    outputs = []
    start = 0
    for index, count in enumerate(counts):
        rows = hidden[start : start + count]
        gate, up = torch.nn.functional.linear(rows, expert.w13_weight[index]).chunk(2, -1)
        activated = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * (25.0 * torch.tanh(up / 25.0))
        outputs.append(torch.nn.functional.linear(activated, expert.w2_weight[index]))
        start += int(count)
    expected = torch.cat(outputs)

    def grouped_matmul(*, x: list[torch.Tensor], weight: list[torch.Tensor], **kwargs: object) -> list[torch.Tensor]:
        # Follow the NPU API's x[M,K] @ weight[E,K,N] contract. A square
        # weight would hide the checkpoint's [E,N,K] orientation error.
        assert x[0].shape[-1] == weight[0].shape[-2]
        ends = kwargs["group_list"]
        local_counts = torch.diff(ends, prepend=ends.new_zeros(1)) if kwargs["group_list_type"] == 0 else ends
        rows = x[0].split(local_counts.tolist())
        return [torch.cat([value @ weight[0][index] for index, value in enumerate(rows)])]

    monkeypatch.setattr(experts_module.torch_npu, "npu_grouped_matmul", grouped_matmul)
    actual = expert(MoETokenDispatchOutput(hidden, groups, group_list_type, None))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("capturing,expected_dim", [(False, 0), (True, -2)])
def test_kimi_row_gather_preserves_main_api_and_token_order(
    monkeypatch: pytest.MonkeyPatch, capturing: bool, expected_dim: int
) -> None:
    from xllm.python.models import kimi_k3_moe as prepare_finalize

    value = torch.arange(12).reshape(2, 6)[:, ::2]
    calls = []

    def gather(tensor: torch.Tensor, *, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        calls.append((dim, world_size, group_name))
        return torch.cat((tensor, tensor + 1), dim=dim)

    monkeypatch.setattr(prepare_finalize, "_in_acl_graph_capture", lambda: capturing)
    monkeypatch.setattr(prepare_finalize.distributed, "all_gather", gather, raising=False)
    actual = prepare_finalize._all_gather_rows(value, dim=0, world_size=2, group_name="moe_ep")
    torch.testing.assert_close(actual, torch.cat((value, value + 1), dim=0))
    assert calls == [(expected_dim, 2, "moe_ep")]


@pytest.mark.parametrize("beta", [0.0, -1.0])
def test_situ_activation_rejects_nonpositive_scales(beta: float) -> None:
    with pytest.raises(ValueError, match="beta must be positive"):
        SituAndMul(beta, 25.0)
    with pytest.raises(ValueError, match="beta must be positive"):
        FusedQuantizedSituAndMul(beta, 25.0)


@pytest.mark.parametrize("linear_beta", [0.0, -1.0])
def test_situ_activation_rejects_nonpositive_linear_scale(linear_beta: float) -> None:
    with pytest.raises(ValueError, match="linear_beta must be positive"):
        SituAndMul(4.0, linear_beta)
    with pytest.raises(ValueError, match="linear_beta must be positive"):
        FusedQuantizedSituAndMul(4.0, linear_beta)


@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_fused_situ_linear_beta_kernel_contract(monkeypatch: pytest.MonkeyPatch, linear_beta: float | None) -> None:
    def fused(*args: object) -> tuple[torch.Tensor, torch.Tensor]:
        assert args[8] == (0.0 if linear_beta is None else linear_beta)
        return torch.empty(2, 8, dtype=torch.int8), torch.empty(2)

    monkeypatch.setattr(torch.ops.xllm_ops, "dequant_situ_quant", fused, raising=False)
    result = FusedQuantizedSituAndMul(4.0, linear_beta)(torch.zeros(2, 16, dtype=torch.bfloat16))
    assert result[0].shape == (2, 8)


def test_situ_activation_none_disables_linear_scale() -> None:
    activation = SituAndMul(4.0, None)
    tensor = torch.tensor([[1.0, -2.0, 100.0, -100.0]])
    gate, up = tensor.chunk(2, dim=-1)
    expected = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * up
    torch.testing.assert_close(activation(tensor), expected)


def test_per_channel_scale_bits_are_zero_extended() -> None:
    scale = torch.tensor([[[0.0, -0.0], [-1.0, 1.0]]], dtype=torch.float32)
    expected = torch.tensor([[[0, 3212836864], [2147483648, 1065353216]]], dtype=torch.int64)
    actual = FusedW4A8RoutedExperts._encode_per_channel_scale(scale)
    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual, expected)
