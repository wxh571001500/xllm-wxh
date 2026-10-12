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

"""CPU contracts for GLM EPLv2 token ownership and routing."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xllm.python.layers.moe_parallel import Eplv2CommPolicy, TokenParallelLayout
from xllm.python.layers.npu.glm5_next_metadata import Glm5NextEplv2Metadata
from xllm.python.layers.qlinear import QLinearWeightLoader
from xllm.python.models import glm5_next


def _config(world: int = 2, rank: int = 0, **overrides: object) -> glm5_next.Glm5NextConfig:
    values = dict(
        hidden_size=8,
        moe_intermediate_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        tp_size=world,
        tp_rank=rank,
        ep_size=world,
        ep_rank=rank,
        moe_tp_size=1,
        moe_tp_rank=0,
        expert_parallel_degree=2,
    )
    values.update(overrides)
    return glm5_next.Glm5NextConfig(**values)


def _moe(config: glm5_next.Glm5NextConfig, dtype: torch.dtype = torch.float32) -> glm5_next.Glm5NextMoE:
    model = glm5_next.Glm5NextMoE(config, dtype, torch.device("cpu"))
    model.use_w8a8 = True
    model._comm_policy = Eplv2CommPolicy(512)
    model.experts_w13 = torch.empty(model.num_local_experts, 8, 32, dtype=torch.int8)
    model.experts_w2 = torch.empty(model.num_local_experts, 16, 8, dtype=torch.int8)
    model.experts_w13_scale = torch.ones(model.num_local_experts, 32)
    model.experts_w2_scale = torch.ones(model.num_local_experts, 8)
    model._ep_dispatch_group_info = ("test_ep", config.ep_rank, config.ep_size)
    return model


class _Shared(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 3


def _metadata(**overrides: object) -> SimpleNamespace:
    fields = dict(
        is_prefill=False,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        expanded_decode_metadata=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _routing(logits: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ones(logits.shape[0], 2), torch.zeros(logits.shape[0], 2, dtype=torch.int32)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("backend", ["mc2", "alltoall"])
def test_router_fp32_preserves_expert_input_dtype(
    dtype: torch.dtype, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _moe(_config(), dtype)
    assert model.gate.weight.dtype == torch.float32
    hidden = (torch.arange(16).reshape(2, 8) / 8 - 1).to(dtype)
    with torch.no_grad():
        model.gate.weight.copy_(torch.arange(64).reshape(8, 8) / 64 - 0.5)
    expected_logits = torch.nn.functional.linear(hidden.float(), model.gate.weight)

    def routing(logits: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        assert logits.dtype == torch.float32
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        return _routing(logits)

    route = Mock(side_effect=routing)
    dispatch = Mock(side_effect=lambda value, *args, **kwargs: value)
    monkeypatch.setattr(glm5_next.kernels, "moe_gate_routing", route, raising=False)
    kernel = "ep_moe_w8a8" if backend == "mc2" else "ep_moe_w8a8_alltoall"
    monkeypatch.setattr(glm5_next.kernels, kernel, dispatch, raising=False)
    layout = TokenParallelLayout(4, 2, 0, routed_backend=backend)
    actual = model._run_routed_sp(hidden, layout, torch.ones(2, dtype=torch.bool))

    route.assert_called_once()
    dispatch.assert_called_once()
    expert_input = dispatch.call_args.args[0]
    assert expert_input.dtype == dtype
    assert expert_input.data_ptr() == hidden.data_ptr()
    torch.testing.assert_close(actual, hidden, rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dp_size": 2},
        {"cp_size": 2},
        {"moe_tp_size": 2},
        {"ep_rank": 1},
        {"enable_mega_moe": True},
        {"enable_fused_mc2": True},
        {"enable_eplb": True},
        {"n_routed_experts": 7},
        {"tp_rank": -1},
        {"expert_parallel_degree": 3},
        {"swiglu_limit": 0.0},
        {"swiglu_limit": float("inf")},
        {"swiglu_limit": float("nan")},
    ],
)
def test_rejects_unsupported_parallel_modes(overrides: dict) -> None:
    with pytest.raises(ValueError):
        _config(**overrides)._validate_moe_parallelism()


def test_nested_config_preserves_native_parallel_overrides() -> None:
    config = glm5_next.Glm5NextConfig.from_dict(
        {
            "text_config": {"hidden_size": 8, "n_routed_experts": 8, "moe_intermediate_size": 16},
            "tp_size": 4,
            "tp_rank": 2,
            "ep_size": 4,
            "ep_rank": 2,
            "moe_tp_size": 1,
            "moe_tp_rank": 0,
            "expert_parallel_degree": 2,
        }
    )
    assert (config.tp_size, config.ep_size, config.moe_tp_size, config.ep_rank) == (4, 4, 1, 2)


@pytest.mark.parametrize("global_rank", range(8))
def test_eplv2_accepts_dp2_tp4_ep8_with_distinct_group_ranks(global_rank: int) -> None:
    cfg = _config(
        world=4,
        rank=global_rank % 4,
        dp_size=2,
        dp_rank=global_rank // 4,
        ep_size=8,
        ep_rank=global_rank,
    )
    assert cfg.ep_size * cfg.moe_tp_size == cfg.dp_size * cfg.tp_size == 8
    cfg._validate_moe_parallelism()
    cfg.ep_rank = (global_rank + 1) % 8
    with pytest.raises(ValueError, match="EP over all DP"):
        cfg._validate_moe_parallelism()


@pytest.mark.parametrize("dp,pcp,tp", [(1, 2, 4), (2, 2, 2), (1, 4, 2), (2, 4, 1)])
@pytest.mark.parametrize("global_rank", range(8))
def test_eplv2_pcp_layout_uses_global_ep_rank(dp: int, pcp: int, tp: int, global_rank: int) -> None:
    cfg = _config(
        world=tp,
        rank=global_rank % tp,
        dp_size=dp,
        dp_rank=global_rank // (pcp * tp),
        cp_size=pcp,
        cp_rank=(global_rank // tp) % pcp,
        ep_size=8,
        ep_rank=global_rank,
    )
    cfg._validate_moe_parallelism()
    assert cfg.ep_rank == (cfg.dp_rank * cfg.cp_size + cfg.cp_rank) * cfg.tp_size + cfg.tp_rank
    cfg.cp_rank = (cfg.cp_rank + 1) % pcp
    with pytest.raises(ValueError, match="EP over all DP/CP"):
        cfg._validate_moe_parallelism()


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("rows", [0, 1, 3, 7, 8, 9, 16, 31])
def test_token_ownership_and_restoration(world: int, rows: int, monkeypatch: pytest.MonkeyPatch) -> None:
    hidden = torch.arange(rows * 8).reshape(rows, 8)
    layouts = [TokenParallelLayout(rows, world, rank) for rank in range(world)]
    shards = [layout.shard(hidden) for layout in layouts]
    masks = [layout.shard(torch.ones(rows, dtype=torch.bool)) for layout in layouts]
    torch.testing.assert_close(torch.cat(shards)[torch.cat(masks)], hidden)
    assert sum(int(mask.sum()) for mask in masks) == rows
    monkeypatch.setattr(glm5_next.distributed, "tp_all_gather", lambda *args: torch.cat(shards), raising=False)
    for layout, shard in zip(layouts, shards):
        torch.testing.assert_close(layout.gather(shard), hidden)


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 3, 8, 9, 16])
def test_full_layout_reference_dispatches_each_active_row_once(
    world: int, rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    hidden = torch.arange(1, rows * 8 + 1, dtype=torch.float32).reshape(rows, 8)
    active = torch.ones(rows, dtype=torch.int8)
    active[-1] = 0
    seen = []
    monkeypatch.setattr(
        glm5_next,
        "get_forward_context_or_none",
        lambda: SimpleNamespace(
            metadata=_metadata(),
            execution_contexts={Glm5NextEplv2Metadata: Glm5NextEplv2Metadata(active)},
        ),
    )
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: True)
    monkeypatch.setattr(glm5_next.kernels, "moe_gate_routing", _routing, raising=False)
    forbidden_reduce = Mock(side_effect=AssertionError("dispatch output must not be EP-reduced again"))
    monkeypatch.setattr(glm5_next.distributed, "moe_ep_all_reduce", forbidden_reduce, raising=False)
    expected_routed = hidden * 2 * active[:, None]
    expected = (hidden * 8) * active[:, None]
    all_shards = [TokenParallelLayout(rows, world, rank).shard(expected) for rank in range(world)]
    monkeypatch.setattr(glm5_next.distributed, "tp_all_gather", lambda *args: torch.cat(all_shards), raising=False)

    def dispatch(local: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        assert kwargs["swiglu_limit"] == 10.0
        assert kwargs["group_ep"] == "test_ep"
        assert kwargs["ep_size"] == world
        mask = kwargs["x_active_mask"]
        assert mask.dtype == torch.bool
        seen.extend(local[mask, 0].tolist())
        return local * 2 * mask[:, None]

    monkeypatch.setattr(glm5_next.kernels, "ep_moe_w8a8", dispatch, raising=False)
    for rank in range(world):
        model = _moe(_config(world, rank, routed_scaling_factor=2.5))
        assert model.shared_experts.skip_tp_reduce is True
        assert model.shared_experts.tp_size == 1
        model.shared_experts = _Shared()
        torch.testing.assert_close(model(hidden), expected)
    assert sorted(seen) == hidden[active.bool(), 0].tolist()
    forbidden_reduce.assert_not_called()


@pytest.mark.parametrize("flag", ["is_prefill", "is_chunked_prefill", "is_mixed"])
def test_full_layout_prefill_uses_the_same_dispatch(flag: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _moe(_config(routed_scaling_factor=2.0))
    model.shared_experts = _Shared()
    hidden = torch.ones(3, 8)
    monkeypatch.setattr(
        glm5_next,
        "get_forward_context_or_none",
        lambda: SimpleNamespace(
            metadata=_metadata(**{flag: True}),
            execution_contexts={Glm5NextEplv2Metadata: Glm5NextEplv2Metadata(torch.ones(3, dtype=torch.bool))},
        ),
    )
    monkeypatch.setattr(glm5_next.kernels, "moe_gate_routing", _routing, raising=False)
    selected = Mock(side_effect=lambda h, *args, **kwargs: h * 5)
    monkeypatch.setattr(glm5_next.kernels, "ep_moe_w8a8", selected, raising=False)
    monkeypatch.setattr(glm5_next.distributed, "tp_all_gather", lambda *args: torch.ones(4, 8) * 13, raising=False)
    forbidden = Mock(side_effect=AssertionError("prefill must not use replicated grouped experts"))
    monkeypatch.setattr(glm5_next.kernels, "grouped_moe_with_selected_experts", forbidden, raising=False)
    torch.testing.assert_close(model(hidden), hidden * 13)
    selected.assert_called_once()
    assert selected.call_args.args[3] is model.experts_w13
    assert selected.call_args.kwargs["swiglu_limit"] == 10.0
    forbidden.assert_not_called()


@pytest.mark.parametrize("mask", [None, torch.ones(2, dtype=torch.float32), torch.ones(3, dtype=torch.int8)])
def test_graph_rejects_missing_or_malformed_mask(mask: torch.Tensor | None, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _moe(_config())
    monkeypatch.setattr(
        glm5_next,
        "get_forward_context_or_none",
        lambda: SimpleNamespace(
            metadata=_metadata(),
            execution_contexts={Glm5NextEplv2Metadata: Glm5NextEplv2Metadata(mask)} if mask is not None else {},
        ),
    )
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: True)
    with pytest.raises(ValueError, match="mask"):
        model(torch.ones(2, 8))


def test_speculative_rows_use_eplv2_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _moe(_config(world=8))
    model.shared_experts = _Shared()
    monkeypatch.setattr(
        glm5_next, "get_forward_context_or_none", lambda: SimpleNamespace(metadata=_metadata(is_spec_verify=True))
    )
    routed = Mock(return_value=torch.ones(2, 8))
    monkeypatch.setattr(model, "_run_routed_sp", routed)
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: False)
    layout = TokenParallelLayout(16, 8, 0, "mc2")
    actual = model(torch.ones(2, 8), token_layout=layout, token_mask=torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(actual, torch.full((2, 8), 5.5))
    routed.assert_called_once()


def test_pd_decode_dflash2_flags_accept_eplv2_without_fused_mc2() -> None:
    for rank in range(8):
        cfg = glm5_next.Glm5NextConfig.from_dict(
            {
                "hidden_size": 8,
                "moe_intermediate_size": 16,
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "tp_size": 8,
                "tp_rank": rank,
                "ep_size": 8,
                "ep_rank": rank,
                "moe_tp_size": 1,
                "moe_tp_rank": 0,
                "expert_parallel_degree": 2,
                "enable_fused_mc2": False,
                "num_speculative_tokens": 7,
                "layers_to_capture": (0, 3),
            }
        )
        assert (cfg.expert_parallel_degree, cfg.ep_size, cfg.num_speculative_tokens) == (2, 8, 7)


def test_tp_route_keeps_main_unified_w8a8_path(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(ep_size=1, ep_rank=0, moe_tp_size=2, expert_parallel_degree=0)
    model = _moe(config)
    assert model.shared_experts.skip_tp_reduce is True
    model.shared_experts = _Shared()
    hidden = torch.ones(3, 8)
    grouped = Mock(return_value=hidden * 5)
    monkeypatch.setattr(glm5_next.kernels, "moe_gate_routing", _routing, raising=False)
    monkeypatch.setattr(glm5_next.kernels, "grouped_moe_with_selected_experts", grouped, raising=False)
    old_tp = Mock(side_effect=AssertionError("Must not resurrect the old unclamped TP path"))
    monkeypatch.setattr(glm5_next.kernels, "grouped_moe", old_tp, raising=False)
    reduce = Mock()
    monkeypatch.setattr(glm5_next.distributed, "moe_tp_all_reduce", reduce, raising=False)
    expected = hidden * (5 * config.routed_scaling_factor + 3)
    torch.testing.assert_close(model(hidden), expected)
    grouped.assert_called_once()
    torch.testing.assert_close(reduce.call_args.args[0], expected)
    assert grouped.call_args.kwargs["swiglu_limit"] == config.swiglu_limit
    old_tp.assert_not_called()


@pytest.mark.parametrize("degree", [0, 1])
@pytest.mark.parametrize("dp,tp,ep", [(1, 8, 1), (1, 8, 8), (2, 4, 8), (2, 4, 4), (2, 4, 1)])
def test_ordinary_ep_does_not_inherit_eplv2_policy_or_shared_layout(
    monkeypatch: pytest.MonkeyPatch, degree: int, dp: int, tp: int, ep: int
) -> None:
    monkeypatch.setenv("XLLM_EPLV2_COMM_MODE", "invalid-for-eplv2")
    monkeypatch.setenv("XLLM_EPLV2_MC2_MAX_TOKENS", "not-an-integer")
    monkeypatch.setenv("HCCL_BUFFSIZE", "not-an-integer")
    cfg = glm5_next.Glm5NextConfig.from_dict(
        {
            "hidden_size": 8,
            "moe_intermediate_size": 16,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "tp_size": tp,
            "ep_size": ep,
            "dp_size": dp,
            "moe_tp_size": dp * tp // ep,
            "moe_tp_rank": 0,
            "expert_parallel_degree": degree,
        }
    )
    model = glm5_next.Glm5NextMoE(cfg, torch.float32, torch.device("cpu"))
    model.use_w8a8 = True
    assert model._comm_policy is None
    assert model._shared_stream is None
    assert model.shared_experts.tp_size == tp
    assert model.shared_experts.gate_up_proj.weight.shape == (2 * cfg.moe_intermediate_size // tp, cfg.hidden_size)
    assert model.shared_experts.skip_tp_reduce
    hidden = torch.ones(1, 8)
    expected = hidden * 7
    ordinary = Mock(return_value=expected)
    dispatch = Mock(side_effect=AssertionError("Ordinary EP must not enter EPLv2"))
    monkeypatch.setattr(model, "_forward_w8a8", ordinary)
    monkeypatch.setattr(model, "_forward_ep", dispatch)
    torch.testing.assert_close(model(hidden), expected)
    ordinary.assert_called_once_with(hidden)
    dispatch.assert_not_called()
    with pytest.raises(ValueError, match="EPLv2 SP requires"):
        model(
            hidden,
            token_layout=TokenParallelLayout(1, tp, 0),
            token_mask=torch.ones(1, dtype=torch.bool),
        )


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_shared_loader_override_is_exclusive_to_eplv2(monkeypatch: pytest.MonkeyPatch, degree: int) -> None:
    cfg = _config(world=4, expert_parallel_degree=degree, mlp_layer_types=["sparse"])
    model = glm5_next.Glm5NextForCausalLM.__new__(glm5_next.Glm5NextForCausalLM)
    torch.nn.Module.__init__(model)
    model.cfg = cfg
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.mlp = glm5_next.Glm5NextMoE(cfg, torch.float32, torch.device("cpu"))
    model.model.layers = torch.nn.ModuleList([layer])
    prefix = "model.layers.0.mlp."
    loader = SimpleNamespace(
        load_fp=Mock(),
        load_tensor=Mock(return_value=torch.zeros(8)),
        copy_in=Mock(),
        find=Mock(return_value=object()),
    )
    experts = Mock()
    shared = Mock()
    monkeypatch.setattr(model, "_load_experts_w8a8", experts)
    monkeypatch.setattr(model, "_load_mlp_fp_or_w8a8", shared)
    monkeypatch.setattr(glm5_next, "_call_process_weights_after_loading", lambda module: None)
    model._load_mlp(loader, prefix, 0)
    experts.assert_called_once_with(loader, prefix)
    if degree == 2:
        shared.assert_called_once_with(loader, prefix + "shared_experts.", world=1, rank=0)
        assert layer.mlp.shared_experts.tp_size == 1
    else:
        shared.assert_called_once_with(loader, prefix + "shared_experts.")
        assert layer.mlp.shared_experts.tp_size == cfg.tp_size


def test_eplv2_rejects_mismatched_pcp_context_before_selecting_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    model = glm5_next.Glm5NextModel.__new__(glm5_next.Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.cfg = _config(cp_size=2, ep_size=4)
    selector = Mock(side_effect=AssertionError("Reject CP before backend selection"))
    model._comm_policy = SimpleNamespace(select=selector)
    monkeypatch.setattr(
        glm5_next,
        "get_forward_context_or_none",
        lambda: SimpleNamespace(cp_context=SimpleNamespace(cp_size=2, cp_rank=1), metadata=None),
    )
    with pytest.raises(ValueError, match="matching prefill shard"):
        model._sp_layout_and_mask(torch.ones(1, 3, 8), torch.ones(1, 3, dtype=torch.bool))
    selector.assert_not_called()


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize(
    "failure,message",
    [
        ("mc2_capacity", "exceed capacity"),
        ("alltoall_graph", "requires eager"),
        ("graph_mask", "stable active-token mask"),
    ],
)
def test_model_rejects_unsupported_execution_before_first_layer(
    monkeypatch: pytest.MonkeyPatch, sequence_parallel: bool, failure: str, message: str
) -> None:
    model = glm5_next.Glm5NextModel.__new__(glm5_next.Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.cfg = _config(eplv2_sequence_parallel=sequence_parallel)
    model._inputs_embeds = torch.ones(1, 3, 8)
    model._comm_policy = Eplv2CommPolicy(
        1 if failure == "mc2_capacity" else 512,
        "alltoall" if failure == "alltoall_graph" else "mc2",
    )
    first_layer = torch.nn.Module()
    first_layer.forward = Mock(side_effect=AssertionError("Admission must precede Attention/KV updates"))
    model.layers = torch.nn.ModuleList([first_layer])
    context = SimpleNamespace(
        metadata=_metadata(),
        execution_contexts=(
            {}
            if failure == "graph_mask"
            else {Glm5NextEplv2Metadata: Glm5NextEplv2Metadata(torch.ones(3, dtype=torch.bool))}
        ),
    )
    monkeypatch.setattr(glm5_next, "get_forward_context_or_none", lambda: context)
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: failure in ("alltoall_graph", "graph_mask"))
    with pytest.raises(ValueError, match=message):
        model(torch.zeros(3, dtype=torch.long), torch.arange(3))
    first_layer.forward.assert_not_called()


def test_full_layout_mode_keeps_supported_execution_without_sp(monkeypatch: pytest.MonkeyPatch) -> None:
    model = glm5_next.Glm5NextModel.__new__(glm5_next.Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.cfg = _config(eplv2_sequence_parallel=False)
    model._comm_policy = Eplv2CommPolicy(512)
    monkeypatch.setattr(
        glm5_next,
        "get_forward_context_or_none",
        lambda: SimpleNamespace(
            metadata=_metadata(),
            execution_contexts={Glm5NextEplv2Metadata: Glm5NextEplv2Metadata(torch.ones(3, dtype=torch.bool))},
        ),
    )
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: False)
    assert model._sp_layout_and_mask(torch.ones(1, 3, 8), torch.ones(1, 3, dtype=torch.bool)) == (None, None)


def test_direct_moe_rejects_capacity_before_shared_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _moe(_config())
    model._comm_policy = Eplv2CommPolicy(1, "mc2")
    model.shared_experts = _Shared()
    shared = Mock(side_effect=AssertionError("Validate routed capacity before launching shared work"))
    monkeypatch.setattr(model.shared_experts, "forward", shared)
    monkeypatch.setattr(glm5_next, "get_forward_context_or_none", lambda: None)
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: False)
    with pytest.raises(ValueError, match="exceed capacity"):
        model(
            torch.ones(2, 8),
            token_layout=TokenParallelLayout(4, 2, 0),
            token_mask=torch.ones(2, dtype=torch.bool),
        )
    shared.assert_not_called()


def test_eplv2_full_layout_entry_rejects_unsharded_pcp_prefill_before_experts(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _moe(_config(eplv2_sequence_parallel=False))
    routed = Mock(side_effect=AssertionError("CP must be rejected before routed execution"))
    monkeypatch.setattr(model, "_run_routed_sp", routed)
    model.cfg.cp_size = 2
    monkeypatch.setattr(
        glm5_next,
        "get_forward_context_or_none",
        lambda: SimpleNamespace(cp_context=None, metadata=_metadata(is_prefill=True)),
    )
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: False)
    with pytest.raises(ValueError, match="prefill requires a matching PCP shard"):
        model(torch.ones(3, 8))
    routed.assert_not_called()


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors
        self.reads: list[str] = []

    def has(self, name: str) -> bool:
        return name in self.tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        self.reads.append(name)
        return self.tensors[name]


@pytest.mark.parametrize("rank", [0, 1])
def test_loader_reads_only_local_ep_experts_with_full_intermediate(rank: int) -> None:
    cfg = _config(2, rank)
    model = glm5_next.Glm5NextForCausalLM.__new__(glm5_next.Glm5NextForCausalLM)
    torch.nn.Module.__init__(model)
    model.cfg = cfg
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.mlp = glm5_next.Glm5NextMoE(cfg, torch.float32, torch.device("cpu"))
    model.model.layers = torch.nn.ModuleList([layer])
    prefix = "model.layers.0.mlp."
    tensors = {}
    for expert in range(8):
        for projection, shape in (("gate_proj", (16, 8)), ("up_proj", (16, 8)), ("down_proj", (8, 16))):
            key = f"{prefix}experts.{expert}.{projection}"
            tensors[key + ".weight"] = torch.full(shape, expert + 1, dtype=torch.int8)
            tensors[key + ".weight_scale"] = torch.ones(shape[0], 1)
            tensors[key + ".weight_offset"] = torch.zeros(shape[0], 1)
    state = _StateDict(tensors)
    loader = QLinearWeightLoader(model, [state], cfg.tp_size, cfg.tp_rank)
    model._load_experts_w8a8(loader, prefix)
    assert layer.mlp.experts_w13.shape == (4, 32, 8)
    assert layer.mlp.experts_w2.shape == (4, 8, 16)
    assert {int(name.split("experts.")[1].split(".")[0]) for name in state.reads} == set(range(rank * 4, rank * 4 + 4))
    for index in range(4):
        assert torch.all(layer.mlp.experts_w13[index] == rank * 4 + index + 1)
