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

"""Regression cases for the shared MoE capture and warmup query."""

from types import SimpleNamespace

import pytest
import torch

from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.models.kimi_k3_moe import _in_acl_graph_capture


def _graph_context(capture: bool, warmup: bool) -> ForwardContext:
    return ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=None,
        layer_caches=[],
        acl_graph=AclGraphCaptureContext(stream=object(), tasks=[]) if capture else None,
        execution_state=AclGraphExecutionState(persistent_buffers={}) if warmup else None,
    )


@pytest.mark.parametrize(
    ("capture", "warmup"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_capture_and_warmup_share_static_metadata_policy(capture: bool, warmup: bool) -> None:
    context = _graph_context(capture, warmup)
    assert not hasattr(context, "graph_warmup")
    with forward_context(context):
        assert _in_acl_graph_capture() is (capture or warmup)


def test_outside_forward_context_is_not_graph_execution() -> None:
    assert _in_acl_graph_capture() is False
    with forward_context(_graph_context(False, True)):
        assert _in_acl_graph_capture() is True
    assert _in_acl_graph_capture() is False


@pytest.mark.parametrize(("capture", "warmup"), [(True, False), (False, True)])
def test_variable_split_dispatch_rejects_graph_before_device_work(capture: bool, warmup: bool) -> None:
    from xllm.python.models import kimi_k3_moe as moe_module

    config = moe_module.MoEParallelConfig(tp_size=1, tp_rank=0, ep_size=2, ep_rank=0)
    dispatcher = moe_module.VllmAllToAllTokenDispatcher(config, num_experts=4, quantized=False)
    routing = moe_module.MoERoutingResult(torch.zeros(1, 1, dtype=torch.int32), torch.ones(1, 1))
    with forward_context(_graph_context(capture, warmup)), pytest.raises(RuntimeError, match="eager-only"):
        dispatcher.token_dispatch(moe_module.MoETokenDispatchInput(torch.ones(1, 2), routing))


@pytest.mark.parametrize(("capture", "warmup"), [(True, False), (False, True), (True, True)])
def test_metadata_free_decode_graph_has_no_global_prefill(capture: bool, warmup: bool) -> None:
    from xllm.python.models.kimi_k3_moe import _global_prefill_step

    with forward_context(_graph_context(capture, warmup)):
        assert _global_prefill_step(2) is False


@pytest.mark.parametrize(
    "metadata",
    [None, SimpleNamespace(), SimpleNamespace(dp_is_decode=[1, 2]), SimpleNamespace(dp_is_decode=())],
)
def test_eager_global_phase_requires_valid_metadata(metadata: object) -> None:
    from dataclasses import replace

    from xllm.python.models.kimi_k3_moe import _global_prefill_step

    with forward_context(replace(_graph_context(False, False), metadata=metadata)), pytest.raises(RuntimeError):
        _global_prefill_step(2)


@pytest.mark.parametrize(("phases", "expected"), [([1, 1], False), ([0, 1], True), ([1, 0], True)])
def test_eager_global_phase_uses_published_dp_vector(phases: list[int], expected: bool) -> None:
    from dataclasses import replace

    from xllm.python.models.kimi_k3_moe import _global_prefill_step

    metadata = SimpleNamespace(dp_is_decode=phases)
    with forward_context(replace(_graph_context(False, False), metadata=metadata)):
        assert _global_prefill_step(2) is expected


@pytest.mark.parametrize("metadata", [None, SimpleNamespace(), SimpleNamespace(dp_is_decode=())])
def test_single_dp_does_not_require_global_phase_metadata(metadata: object) -> None:
    from dataclasses import replace

    from xllm.python.models.kimi_k3_moe import _global_prefill_step

    with forward_context(replace(_graph_context(False, False), metadata=metadata)):
        assert _global_prefill_step(1) is False


@pytest.mark.parametrize(("ep_size", "world_size", "sequence_parallel"), [(1, 2, False), (2, 2, True), (2, 4, False)])
@pytest.mark.parametrize("publish_phase", [False, True])
def test_single_dp_tp_decode_does_not_require_global_phase(
    monkeypatch: pytest.MonkeyPatch, ep_size: int, world_size: int, sequence_parallel: bool, publish_phase: bool
) -> None:
    from dataclasses import replace

    from xllm.python.models import kimi_k3_moe as moe_module

    config = SimpleNamespace(
        num_experts=4,
        num_experts_per_token=1,
        hidden_size=2,
        routed_expert_hidden_size=2,
        moe_intermediate_size=4,
        moe_router_activation_func="sigmoid",
        moe_renormalize=True,
        routed_scaling_factor=1.0,
        ep_size=ep_size,
        dp_size=1,
        world_size=world_size,
        rank=0,
    )
    model = moe_module.KimiK3MoE(
        config, torch.float32, torch.device("cpu"), 2, 0, torch.nn.Identity(), torch.nn.Identity()
    )
    assert model.sequence_parallel_routed is sequence_parallel
    assert model.dp_size == 1
    hidden = torch.ones(1, 2)
    metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False)
    if publish_phase:
        metadata.dp_is_decode = ()
    # Exercise the real constructor and phase decision without a TP world.
    monkeypatch.setattr(moe_module.MoE, "forward", lambda self, value: value)
    with forward_context(replace(_graph_context(False, False), metadata=metadata)):
        assert model(hidden) is hidden


@pytest.mark.parametrize(
    ("capture", "warmup"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_adaptive_moe_warmup_and_capture_use_all_gather(capture: bool, warmup: bool) -> None:
    from xllm.python.models import kimi_k3_moe as moe_module

    hidden, logits = torch.ones(1, 2), torch.ones(1, 4)
    calls: list[str] = []

    def comm_method(name: str) -> SimpleNamespace:
        def prepare(value: torch.Tensor, scores: torch.Tensor) -> moe_module.MoEPrepareOutput:
            calls.append(name)
            return moe_module.MoEPrepareOutput(value, scores)

        return SimpleNamespace(prepare=prepare)

    method = moe_module.AdaptiveMoECommMethod.__new__(moe_module.AdaptiveMoECommMethod)
    method._config = moe_module.MoEParallelConfig(tp_size=1, tp_rank=0, ep_size=2, ep_rank=0)
    method._all_gather = comm_method("all_gather")
    method._all_to_all = comm_method("all_to_all")
    with forward_context(_graph_context(capture, warmup)):
        result = method.prepare(hidden, logits)
    assert calls == (["all_gather"] if capture or warmup else ["all_to_all"])
    torch.testing.assert_close(result.hidden_states, hidden)


@pytest.mark.parametrize("tokens", [0, 1, 512, 513])
def test_auto_eager_protocol_is_independent_of_local_token_count(tokens: int) -> None:
    from xllm.python.models import kimi_k3_moe as moe_module

    method = moe_module.AdaptiveMoECommMethod.__new__(moe_module.AdaptiveMoECommMethod)
    expected = object()
    method._all_to_all = SimpleNamespace(prepare=lambda hidden, logits: expected)
    method._all_gather = SimpleNamespace(prepare=lambda hidden, logits: pytest.fail("unexpected AllGather"))
    with forward_context(_graph_context(False, False)):
        assert method.prepare(torch.empty(tokens, 2), torch.empty(tokens, 4)) is expected


def test_tp_sharded_experts_keep_prefill_tp_reduction(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python.models import kimi_k3_moe as moe_module

    partials = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0], [7.0, 8.0]])]
    expected = partials[0] + partials[1]
    calls = []

    def reduce(value: torch.Tensor) -> None:
        calls.append(value.clone())
        value.copy_(expected)

    monkeypatch.setattr(moe_module.distributed, "all_reduce_", reduce, raising=False)
    for rank, partial in enumerate(partials):
        config = moe_module.MoEParallelConfig(tp_size=2, tp_rank=rank, ep_size=1, ep_rank=0, tp_group_name="tp")
        prepare = moe_module.AllGatherPrepareAndFinalize(config)
        with forward_context(_graph_context(False, False)):
            prepare.prepare(torch.ones(2, 2), torch.ones(2, 4))
            actual = prepare.finalize(partial.clone(), reduce_results=True)
        torch.testing.assert_close(actual, expected)
    assert len(calls) == 2


@pytest.mark.parametrize(("capture", "warmup"), [(True, False), (False, True)])
def test_mc2_graph_warmup_rejects_missing_static_bucket(
    monkeypatch: pytest.MonkeyPatch, capture: bool, warmup: bool
) -> None:
    from xllm.python.models import kimi_k3_moe as moe_module

    config = moe_module.MoEParallelConfig(tp_size=1, tp_rank=0, ep_size=2, ep_rank=0, input_tp_size=2, input_tp_rank=0)
    prepare = moe_module.MC2PrepareAndFinalize(config)
    monkeypatch.setattr(
        prepare, "_max_tokens_across_dp", lambda value: pytest.fail("graph execution must not gather dynamic counts")
    )
    with forward_context(_graph_context(capture, warmup)), pytest.raises(ValueError, match="static token bucket"):
        prepare.prepare(torch.ones(1, 2), torch.ones(1, 4))


@pytest.mark.parametrize(
    ("local_prefill", "global_prefill", "tokens", "sequence_tokens", "sequence_parallel"),
    [
        (False, True, 0, None, True),
        (False, True, 1, None, True),
        (False, True, 1, 1, True),
        (True, True, 2, 3, True),
        (True, True, 1, None, True),
        (True, False, 1, None, True),
        (False, False, 1, None, True),
        (False, False, 1, 1, True),
        (False, True, 1, None, False),
    ],
)
def test_global_prefill_protocol_does_not_require_local_sequence_metadata(
    monkeypatch: pytest.MonkeyPatch,
    local_prefill: bool,
    global_prefill: bool,
    tokens: int,
    sequence_tokens: int | None,
    sequence_parallel: bool,
) -> None:
    from xllm.python.models import kimi_k3_moe as moe_module

    hidden = torch.arange(tokens * 2, dtype=torch.float32).reshape(tokens, 2)
    calls: list[str] = []
    shared_inputs: list[torch.Tensor] = []

    class SharedExperts(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            shared_inputs.append(value.clone())
            return value.clone()

    def routed_forward(**kwargs: object) -> torch.Tensor:
        calls.append("prefill")
        assert kwargs["shared_experts"] is None
        return kwargs["hidden_states"] * 3

    def decode_forward(self: moe_module.MoE, value: torch.Tensor) -> torch.Tensor:
        calls.append("decode")
        return value * 7

    def gather(value: torch.Tensor, *, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert (dim, world_size, group_name) == (0, 2, "tp")
        calls.append("tp_gather")
        return torch.cat((value, value), dim=0)

    def reduce_scatter(value: torch.Tensor, *, world_size: int, group_name: str) -> torch.Tensor:
        assert (world_size, group_name) == (2, "tp")
        assert value.shape[0] % world_size == 0
        calls.append("tp_reduce_scatter")
        return value[:tokens] * 2

    def reduce(value: torch.Tensor, *, group_name: str) -> None:
        assert group_name == "tp"
        calls.append("tp_reduce")
        value.mul_(2)

    model = moe_module.KimiK3MoE.__new__(moe_module.KimiK3MoE)
    torch.nn.Module.__init__(model)
    model.hidden_size = 2
    model.tp_size = 2
    model.dp_size = 2
    model.sequence_parallel_routed = sequence_parallel
    model.gate = torch.nn.Linear(2, 4, bias=False)
    model.gate.e_score_correction_bias = torch.nn.Parameter(torch.zeros(4))
    model.experts = torch.nn.Identity()
    model.shared_experts = SharedExperts()
    model.routed_expert_down_proj = torch.nn.Identity()
    model._runner = SimpleNamespace(_unwrap_tensor=lambda value: value)
    model._prefill_runner = SimpleNamespace(forward=routed_forward)
    monkeypatch.setattr(moe_module, "_in_prefill", lambda: local_prefill)
    monkeypatch.setattr(moe_module, "_global_prefill_step", lambda dp_size: global_prefill)
    monkeypatch.setattr(moe_module.MoE, "forward", decode_forward)
    monkeypatch.setattr(moe_module.distributed, "all_gather", gather, raising=False)
    monkeypatch.setattr(moe_module.distributed, "reduce_scatter", reduce_scatter, raising=False)
    monkeypatch.setattr(moe_module.distributed, "all_reduce_", reduce, raising=False)

    actual = model(hidden, sequence_parallel_tokens=sequence_tokens)

    if not sequence_parallel or not (local_prefill or global_prefill):
        assert calls == ["decode"]
        assert shared_inputs == []
        torch.testing.assert_close(actual, hidden * 7)
    else:
        if local_prefill and sequence_tokens is not None:
            assert calls == ["prefill", "tp_gather", "tp_reduce_scatter"]
            torch.testing.assert_close(shared_inputs[0], torch.cat((hidden, hidden))[:sequence_tokens])
        else:
            assert calls == ["prefill", "tp_reduce"]
            torch.testing.assert_close(shared_inputs[0], hidden)
        assert actual.shape == hidden.shape
        torch.testing.assert_close(actual, hidden * 5)


@pytest.mark.parametrize("comm_type", ["mc2", "auto"])
@pytest.mark.parametrize("group_name", ["mc2", "custom_ep"])
def test_mc2_rejects_unsupported_ep_group_before_communicator_lookup(
    monkeypatch: pytest.MonkeyPatch, comm_type: str, group_name: str
) -> None:
    from xllm.python.models import kimi_k3_moe as moe_module

    config = moe_module.MoEParallelConfig(
        tp_size=1,
        tp_rank=0,
        ep_size=2,
        ep_rank=0,
        comm_type=moe_module.MoECommType(comm_type),
        ep_group_name=group_name,
    )

    def unexpected_lookup(device: torch.device) -> tuple[str, int, int]:
        pytest.fail("Unsupported group must be rejected before HCCL lookup")

    monkeypatch.setattr(moe_module.distributed, "moe_ep_hccl_info", unexpected_lookup, raising=False)
    with pytest.raises(ValueError, match="mainline moe_ep group"):
        moe_module.MC2TokenDispatcher(config, num_experts=4, quantized=False, device=torch.device("npu:0"))


@pytest.mark.parametrize(
    "counts",
    [(1, "decode"), (1,), (1, -1)],
)
def test_metadata_token_counts_reject_malformed_values(counts: tuple[object, ...]) -> None:
    from types import SimpleNamespace

    from xllm.python.models.kimi_k3_moe import AllGatherPrepareAndFinalize, MoEParallelConfig

    config = MoEParallelConfig(tp_size=1, tp_rank=0, ep_size=2, ep_rank=0, dp_size=2, dp_rank=0)
    prepare = AllGatherPrepareAndFinalize(config)
    context = SimpleNamespace(metadata=SimpleNamespace(dp_execution_token_counts=counts))
    from xllm.python.models import kimi_k3_moe as moe_module

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(moe_module, "get_forward_context", lambda: context)
        with pytest.raises(ValueError, match="dp_execution_token_counts"):
            prepare._max_tokens_from_metadata(torch.ones(1, 2))


def test_metadata_token_counts_are_used_without_collective() -> None:
    from types import SimpleNamespace

    from xllm.python.models import kimi_k3_moe as moe_module
    from xllm.python.models.kimi_k3_moe import AllGatherPrepareAndFinalize, MoEParallelConfig

    config = MoEParallelConfig(tp_size=1, tp_rank=0, ep_size=2, ep_rank=0, dp_size=2, dp_rank=0)
    prepare = AllGatherPrepareAndFinalize(config)
    context = SimpleNamespace(metadata=SimpleNamespace(dp_execution_token_counts=(1, 3)))
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(moe_module, "get_forward_context", lambda: context)
        monkeypatch.setattr(prepare, "_gather_token_counts", lambda value: pytest.fail("unexpected collective"))
        assert prepare._max_tokens_across_dp(torch.ones(1, 2)) == 3


@pytest.mark.parametrize("counts", [None, ()])
def test_missing_scheduler_counts_preserve_collective_fallback(
    monkeypatch: pytest.MonkeyPatch, counts: tuple[int, ...] | None
) -> None:
    from xllm.python.models import kimi_k3_moe as moe_module

    config = moe_module.MoEParallelConfig(tp_size=1, tp_rank=0, ep_size=2, ep_rank=0, dp_size=2, dp_rank=0)
    prepare = moe_module.AllGatherPrepareAndFinalize(config)
    context = SimpleNamespace(metadata=SimpleNamespace(dp_execution_token_counts=counts))
    monkeypatch.setattr(moe_module, "get_forward_context", lambda: context)
    calls: list[int] = []

    def gather(value: torch.Tensor) -> torch.Tensor:
        calls.append(value.shape[0])
        return torch.tensor([1, 3], dtype=torch.int64)

    monkeypatch.setattr(prepare, "_gather_token_counts", gather)
    assert prepare._max_tokens_across_dp(torch.ones(1, 2)) == 3
    assert calls == [1]


@pytest.mark.parametrize("phase", ["prefill", "decode", "warmup", "capture"])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("local_rows", [0, 2])
def test_fused_all_gather_uses_only_local_expert_counts(
    monkeypatch: pytest.MonkeyPatch, phase: str, quantized: bool, local_rows: int
) -> None:
    from unittest.mock import MagicMock

    from xllm.python.models import kimi_k3_moe as moe_module

    hidden = torch.ones(4, 2)
    ids = torch.tensor([[4, 0], [5, 8], [6, 9], [7, 10]], dtype=torch.int32)
    weights = torch.full((4, 2), 0.5)
    sorted_hidden = torch.arange(16).reshape(8, 2).to(torch.int8 if quantized else hidden.dtype)
    row_indices = torch.arange(8, dtype=torch.int32)
    # The active experts [4, 8) occupy the first four counts, not [4:8].
    # Unused entries must not affect local groups or the fixed routing buffer.
    local_counts = torch.tensor([local_rows, 0, 0, 0], dtype=torch.int32)
    global_counts = torch.cat((local_counts, torch.ones(12, dtype=torch.int32)))
    scales = torch.arange(8, dtype=torch.float32) + 1
    route = MagicMock(return_value=(sorted_hidden, row_indices, global_counts, scales if quantized else None))
    monkeypatch.setattr(moe_module.torch_npu, "npu_moe_init_routing_v2", route, raising=False)
    monkeypatch.setattr(moe_module, "_in_prefill", lambda: phase != "decode")
    dispatcher = moe_module.FusedAllGatherTokenDispatcher(
        num_experts=16, top_k=2, quantized=quantized, first_expert_id=4, num_local_experts=4
    )
    context = _graph_context(capture=phase == "capture", warmup=phase == "warmup")
    with forward_context(context):
        result = dispatcher.token_dispatch(
            moe_module.MoETokenDispatchInput(hidden, moe_module.MoERoutingResult(ids, weights))
        )

    route.assert_called_once()
    assert route.call_args.kwargs["active_expert_range"] == [4, 8]
    assert route.call_args.kwargs["expert_num"] == 16
    assert route.call_args.kwargs["active_num"] == 8
    assert route.call_args.kwargs["quant_mode"] == (1 if quantized else -1)
    torch.testing.assert_close(result.group_list, local_counts.to(torch.int64))
    assert result.group_list_type == 1
    expected_rows = 8
    torch.testing.assert_close(result.hidden_states, sorted_hidden[:expected_rows])
    if quantized:
        torch.testing.assert_close(result.dynamic_scale, scales[:expected_rows])
    else:
        assert result.dynamic_scale is None
    assert result.combine_metadata.expanded_row_indices is row_indices
    torch.testing.assert_close(result.combine_metadata.routing_weights, weights * ((ids >= 4) & (ids < 8)))
