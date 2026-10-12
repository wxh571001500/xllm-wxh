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

"""CPU layout contracts; these tests do not substitute for real NPU execution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from xllm.python.layers.moe_parallel import Eplv2CommPolicy, TokenParallelLayout
from xllm.python.layers.npu.glm5_next_metadata import Glm5NextEplv2Metadata
from xllm.python.layers.qlinear import QLinearWeightLoader
from xllm.python.models import glm5_next


@pytest.fixture(autouse=True)
def _cpu_hc_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide deterministic CPU substitutes for the NPU-only mHC kernels."""

    def hc_pre_fused(
        hidden: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        hc_mult: int,
        iterations: int,
        norm_eps: float,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del fn, scale, base, iterations, norm_eps, hc_eps
        post = hidden.new_ones((*hidden.shape[:-2], hc_mult))
        comb = torch.eye(hc_mult, dtype=hidden.dtype, device=hidden.device).expand(*hidden.shape[:-2], hc_mult, hc_mult)
        return hidden.mean(dim=-2), post, comb

    def hc_post(output: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        return post.to(output.dtype).unsqueeze(-1) * output.unsqueeze(-2) + torch.matmul(
            comb.to(output.dtype).transpose(-1, -2), residual
        )

    monkeypatch.setattr(glm5_next.kernels, "hc_pre_fused", hc_pre_fused, raising=False)
    monkeypatch.setattr(glm5_next.kernels, "hc_post", hc_post, raising=False)


def _config(world: int = 2, rank: int = 0, **kwargs: object) -> glm5_next.Glm5NextConfig:
    values = dict(
        hidden_size=8,
        intermediate_size=16,
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
        kda_num_heads=8,
        kda_head_dim=2,
        n_heads=8,
        vocab_size=32,
        n_layers=1,
        first_k_dense_replace=0,
    )
    values.update(kwargs)
    return glm5_next.Glm5NextConfig(**values)


def _metadata(**kwargs: object) -> SimpleNamespace:
    values = dict(
        is_prefill=False,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        expanded_decode_metadata=None,
    )
    values.update(kwargs)
    return SimpleNamespace(metadata=SimpleNamespace(**values))


def _moe(cfg: glm5_next.Glm5NextConfig) -> glm5_next.Glm5NextMoE:
    layer = glm5_next.Glm5NextMoE(cfg, torch.float32, torch.device("cpu"))
    layer.use_w8a8 = True
    layer._comm_policy = Eplv2CommPolicy(512)
    for name in ("experts_w13", "experts_w2", "experts_w13_scale", "experts_w2_scale"):
        setattr(layer, name, torch.empty(0))
    layer._ep_dispatch_group_info = ("sp_ep", cfg.ep_rank, cfg.ep_size)
    return layer


class _Shared(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[torch.Tensor] = []

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.rows.append(hidden.clone())
        return hidden * 3


def test_model_constructs_one_shared_eplv2_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock(wraps=glm5_next._eplv2_comm_policy)
    monkeypatch.setattr(glm5_next, "_eplv2_comm_policy", factory)
    cfg = _config(
        n_layers=3,
        layer_types=["linear_attention"] * 3,
        mlp_layer_types=["sparse", "dense", "sparse"],
    )
    model = glm5_next.Glm5NextModel(cfg, torch.float32, torch.device("cpu"))
    assert factory.call_count == 1
    for layer in model.layers:
        if isinstance(layer.mlp, glm5_next.Glm5NextMoE):
            assert layer.mlp._comm_policy is model._comm_policy

    factory.reset_mock()
    standalone = glm5_next.Glm5NextMoE(cfg, torch.float32, torch.device("cpu"))
    assert factory.call_count == 1
    assert standalone._comm_policy == model._comm_policy


@pytest.mark.parametrize("degree", [0, 1])
def test_ordinary_model_does_not_construct_eplv2_policy(degree: int, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock(side_effect=AssertionError("ordinary TP/EPLv1 must not construct an EPLv2 policy"))
    monkeypatch.setattr(glm5_next.Eplv2CommPolicy, "from_geometry", factory)
    cfg = _config(
        expert_parallel_degree=degree,
        ep_size=1 if degree == 0 else 2,
        moe_tp_size=2 if degree == 0 else 1,
        layer_types=["linear_attention"],
        mlp_layer_types=["sparse"],
    )
    model = glm5_next.Glm5NextModel(cfg, torch.float32, torch.device("cpu"))
    assert model._comm_policy is None
    assert model.layers[0].mlp._comm_policy is None
    factory.assert_not_called()


@pytest.mark.parametrize("capture,state", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("rows", [4, 5])
def test_shared_backend_selector_preserves_graph_admission(
    capture: bool, state: bool, rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: capture)
    context = SimpleNamespace(execution_state=object()) if state else None
    policy = Eplv2CommPolicy(4)
    if rows > 4 and (capture or state):
        with pytest.raises(ValueError, match="All-to-AllV requires eager"):
            glm5_next._select_eplv2_backend(policy, context, rows)
    else:
        assert glm5_next._select_eplv2_backend(policy, context, rows) == (
            "mc2" if rows <= 4 else "alltoall",
            capture or state,
        )


@pytest.mark.parametrize(
    "dtype,routed_value,shared_value,residual",
    [
        (torch.bfloat16, 1.0078125, -2.515625, 0.00390625),
        (torch.float16, 1.0009765625, -2.501953125, 0.00048828125),
    ],
)
def test_sp_scales_and_merges_in_fp32_before_final_cast(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    routed_value: float,
    shared_value: float,
    residual: float,
) -> None:
    layer = _moe(_config(routed_scaling_factor=2.5))
    routed = torch.full((2, 8), routed_value, dtype=dtype)
    shared = torch.full_like(routed, shared_value)
    layer.shared_experts = nn.Identity()
    monkeypatch.setattr(layer.shared_experts, "forward", lambda x: shared)
    monkeypatch.setattr(layer, "_run_routed_sp", lambda *args: routed)
    monkeypatch.setattr(glm5_next, "get_forward_context_or_none", lambda: None)
    mask = torch.tensor([True, False])
    actual = layer(
        torch.ones_like(routed),
        token_layout=TokenParallelLayout(4, 2, 0, "mc2"),
        token_mask=mask,
    )
    expected = (routed.double() * 2.5 + shared.double()).to(dtype).masked_fill(~mask[:, None], 0)
    assert expected[0, 0].item() == residual
    assert (routed * 2.5 + shared)[0, 0].item() == 0
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("rows", [0, 1, 3, 8, 9])
@pytest.mark.parametrize("group", ["tp", "moe_ep"])
def test_layout_reduces_partials_but_only_slices_replicas(
    world: int, rows: int, group: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = torch.arange(rows * 12, dtype=torch.float64).reshape(rows, 3, 4)
    total = full * (world * (world + 1) / 2)
    for rank in range(world):
        layout = TokenParallelLayout(rows, world, rank)
        partial = full * (rank + 1)

        def reduce(
            value: torch.Tensor,
            size: int,
            name: str,
            expected: torch.Tensor = partial,
            token_layout: TokenParallelLayout = layout,
        ) -> torch.Tensor:
            assert size == world and name == group
            torch.testing.assert_close(value[:rows], expected, rtol=0, atol=0)
            assert torch.count_nonzero(value[rows:]) == 0
            return token_layout.shard(total)

        monkeypatch.setattr(glm5_next.distributed, "reduce_scatter", reduce, raising=False)
        actual = layout.reduce_scatter(partial, group)
        torch.testing.assert_close(actual, layout.shard(total), rtol=0, atol=0)
        assert layout.valid_tokens == min(layout.shard_tokens, max(0, rows - rank * layout.shard_tokens))
        torch.testing.assert_close(
            layout.shard(full)[: layout.valid_tokens],
            full[rank * layout.shard_tokens :][: layout.valid_tokens],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 3, 9, 16])
def test_sp_decode_keeps_output_local_and_skips_empty_source_shared(
    world: int, rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = torch.arange(1, rows * 8 + 1, dtype=torch.float32).reshape(rows, 8)
    active = torch.ones(rows, dtype=torch.bool)
    active[-1] = False
    seen: list[float] = []
    monkeypatch.setattr(glm5_next, "get_forward_context_or_none", _metadata)
    forbidden = Mock(side_effect=AssertionError("SP decode must not gather or reduce its complete local result"))
    for name in ("all_reduce_", "moe_ep_all_reduce", "moe_tp_all_reduce", "tp_all_gather", "reduce_scatter"):
        monkeypatch.setattr(glm5_next.distributed, name, forbidden, raising=False)

    def dispatch(hidden: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        mask = kwargs["x_active_mask"]
        seen.extend(hidden[mask, 0].tolist())
        assert kwargs["swiglu_limit"] == 10 and kwargs["group_ep"] == "sp_ep"
        output = hidden * 2
        output[~mask] = float("nan")  # Invalid rows must be cleared, not multiplied by zero.
        return output

    monkeypatch.setattr(glm5_next.kernels, "ep_moe_w8a8", dispatch, raising=False)
    outputs = []
    for rank in range(world):
        cfg = _config(world, rank)
        layer = _moe(cfg)
        shared = _Shared()
        layer.shared_experts = shared
        route = Mock(side_effect=lambda h: (torch.ones(h.shape[0], 2), torch.zeros(h.shape[0], 2, dtype=torch.int32)))
        monkeypatch.setattr(layer, "_route_ep", route)
        layout = TokenParallelLayout(rows, world, rank)
        local, mask = layout.shard(full), layout.shard(active)
        result = layer(local, token_layout=layout, token_mask=mask)
        torch.testing.assert_close(result, local * 8 * mask[:, None], rtol=0, atol=0)
        assert len(shared.rows) == int(layout.valid_tokens > 0)
        assert route.call_count == int(layout.valid_tokens > 0)
        outputs.append(result)
    assert sorted(seen) == full[active, 0].tolist()
    torch.testing.assert_close(torch.cat(outputs)[:rows], full * 8 * active[:, None], rtol=0, atol=0)
    forbidden.assert_not_called()


@pytest.mark.parametrize("phase", ["is_prefill", "is_chunked_prefill", "is_mixed"])
@pytest.mark.parametrize("backend", ["mc2", "alltoall"])
def test_sp_backend_does_not_depend_on_phase(phase: str, backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    layer = _moe(_config(2, 1, routed_scaling_factor=2.0))
    layer.shared_experts = _Shared()
    layout = TokenParallelLayout(3, 2, 1, backend)
    full = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    local = layout.shard(full)
    mask = layout.shard(torch.ones(3, dtype=torch.bool))
    monkeypatch.setattr(glm5_next, "get_forward_context_or_none", lambda: _metadata(**{phase: True}))
    monkeypatch.setattr(layer, "_route_ep", lambda h: (torch.ones(h.shape[0], 2), torch.zeros(h.shape[0], 2)))
    forbidden = Mock(side_effect=AssertionError("SP routed must not gather full hidden or reduce a complete result"))
    for name in ("tp_all_gather", "reduce_scatter", "moe_ep_all_reduce"):
        monkeypatch.setattr(glm5_next.distributed, name, forbidden, raising=False)
    call = Mock(side_effect=lambda h, *args, **kwargs: h * 5)
    monkeypatch.setattr(glm5_next.kernels, "ep_moe_w8a8", call if backend == "mc2" else forbidden, raising=False)
    monkeypatch.setattr(
        glm5_next.kernels, "ep_moe_w8a8_alltoall", call if backend == "alltoall" else forbidden, raising=False
    )
    output = layer(local, token_layout=layout, token_mask=mask)
    torch.testing.assert_close(output, local * 13, rtol=0, atol=0)
    call.assert_called_once()
    forbidden.assert_not_called()


def test_shared_stream_fork_join_and_tensor_lifetime_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = []

    class Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_stream(self, other: Stream) -> None:
            trace.append((self.name, "wait", other.name))

    main, auxiliary = Stream("main"), Stream("shared")

    @contextmanager
    def stream_scope(stream: Stream) -> Iterator[None]:
        trace.append(("enter", stream.name))
        yield
        trace.append(("leave", stream.name))

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda device: main, stream=stream_scope), raising=False
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: trace.append(("lifetime", stream.name)))
    monkeypatch.setattr(glm5_next, "get_forward_context_or_none", _metadata)
    layer = _moe(_config())
    layer._shared_stream = auxiliary
    layer.shared_experts = _Shared()

    def shared(hidden: torch.Tensor) -> torch.Tensor:
        trace.append(("compute", "shared"))
        return hidden * 3

    def routed(*args: object) -> torch.Tensor:
        trace.append(("compute", "routed"))
        return args[0] * 2

    monkeypatch.setattr(layer.shared_experts, "forward", shared)
    monkeypatch.setattr(layer, "_run_routed_sp", routed)
    result = layer(
        torch.ones(2, 8), token_layout=TokenParallelLayout(4, 2, 0), token_mask=torch.ones(2, dtype=torch.bool)
    )
    torch.testing.assert_close(result, torch.full((2, 8), 8.0))
    assert trace == [
        ("shared", "wait", "main"),
        ("lifetime", "shared"),
        ("enter", "shared"),
        ("compute", "shared"),
        ("leave", "shared"),
        ("compute", "routed"),
        ("main", "wait", "shared"),
        ("lifetime", "main"),
    ]


@pytest.mark.parametrize("mask", [None, torch.ones(3, dtype=torch.bool), torch.ones(2, dtype=torch.int8)])
def test_decoder_rejects_bad_sp_mask_before_attention(
    mask: torch.Tensor | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(layer_types=["linear_attention"], mlp_layer_types=["sparse"])
    layer = glm5_next.Glm5NextDecoderLayer(cfg, 0, torch.float32, torch.device("cpu"))
    forbidden = Mock(side_effect=AssertionError("malformed layout must fail before communication"))
    monkeypatch.setattr(glm5_next.distributed, "tp_all_gather", forbidden, raising=False)
    with pytest.raises(ValueError, match="mask before attention"):
        layer(
            torch.zeros(1, 4, 4, 8),
            torch.arange(4),
            torch.ones(1, 4),
            output_layout=TokenParallelLayout(4, 2, 0),
            token_mask=mask,
        )
    forbidden.assert_not_called()


class _StateDict:
    def __init__(self, values: dict[str, torch.Tensor]) -> None:
        self.values = values

    def has(self, key: str) -> bool:
        return key in self.values

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.values[key]


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("quantized", [False, True])
def test_shared_weights_follow_moe_tp_not_attention_tp(world: int, quantized: bool) -> None:
    cfg = _config(world, world - 1)
    model = glm5_next.Glm5NextForCausalLM.__new__(glm5_next.Glm5NextForCausalLM)
    nn.Module.__init__(model)
    model.cfg = cfg
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = glm5_next.Glm5NextMoE(cfg, torch.float32, torch.device("cpu"))
    model.model.layers = nn.ModuleList([layer])
    prefix = "model.layers.0.mlp.shared_experts."
    values = {}
    for index, (projection, shape) in enumerate((("gate_proj", (16, 8)), ("up_proj", (16, 8)), ("down_proj", (8, 16)))):
        key = prefix + projection
        values[key + ".weight"] = (torch.arange(128).reshape(shape) % 7 + index).to(
            torch.int8 if quantized else torch.float32
        )
        if quantized:
            values[key + ".weight_scale"] = torch.ones(shape[0], 1) / 16
            values[key + ".weight_offset"] = torch.zeros(shape[0], 1)
    loader = QLinearWeightLoader(model, [_StateDict(values)], cfg.tp_size, cfg.tp_rank)
    model._load_mlp_fp_or_w8a8(loader, prefix, world=cfg.moe_tp_size, rank=cfg.moe_tp_rank)
    shared = layer.mlp.shared_experts
    assert shared.tp_size == 1 and shared.skip_tp_reduce
    gate = shared.gate_up_proj._w8a8 if quantized else shared.gate_up_proj
    down = shared.down_proj._w8a8 if quantized else shared.down_proj
    torch.testing.assert_close(
        gate.weight, torch.cat((values[prefix + "gate_proj.weight"], values[prefix + "up_proj.weight"])), rtol=0, atol=0
    )
    torch.testing.assert_close(down.weight, values[prefix + "down_proj.weight"], rtol=0, atol=0)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("rows", [1, 3, 9])
@pytest.mark.parametrize(
    "schedule", [("sparse", "sparse"), ("dense", "sparse", "sparse"), ("sparse", "dense", "sparse")]
)
@pytest.mark.parametrize("capture", [False, True])
def test_model_keeps_mhc_and_norm_local_until_attention(
    world: int, rows: int, schedule: tuple[str, ...], capture: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(5319)
    cfg = _config(
        world,
        n_layers=len(schedule),
        layer_types=["linear_attention"] * len(schedule),
        mlp_layer_types=list(schedule),
        layers_to_capture=(0, len(schedule) - 1) if capture else (),
    )
    model = glm5_next.Glm5NextModel(cfg, torch.float32, torch.device("cpu"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    for layer in model.layers:
        layer.attn_hc.process_weights_after_loading()
        layer.ffn_hc.process_weights_after_loading()
    monkeypatch.setattr(glm5_next, "get_forward_context_or_none", lambda: None)
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: False)
    incoming, normalized, layer_outputs, local_attention = {}, {}, {}, {}
    layer_events = Mock()
    monkeypatch.setattr(glm5_next, "record_layer_event", layer_events)
    norm_shapes: list[tuple[int, int]] = []

    class Attention(nn.Module):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

        def forward(
            self, hidden: torch.Tensor, *args: object, output_layout: TokenParallelLayout | None = None
        ) -> tuple[torch.Tensor, None]:
            if output_layout is None:
                normalized[self.index] = hidden.detach().reshape(rows, 8).clone()
                return hidden * 0.125, None
            torch.testing.assert_close(hidden.reshape(rows, 8), normalized[self.index], rtol=1e-5, atol=1e-6)
            output = output_layout.reduce_scatter(hidden.reshape(rows, 8) * (0.125 / world))
            local_attention[self.index] = output
            return output, None

    def pointwise(self: nn.Module, hidden: torch.Tensor, **kwargs: object) -> torch.Tensor:
        layout = kwargs.get("token_layout")
        if layout is not None and not layout.valid_tokens:
            assert hidden.data_ptr() == local_attention[self._test_index].data_ptr()
        return hidden * 0.25

    for index, layer in enumerate(model.layers):

        def capture_input(module: nn.Module, args: tuple, index: int = index) -> None:
            if not cfg.eplv2_sequence_parallel:
                incoming[index] = args[0].detach().reshape(rows, 4, 8).clone()

        layer.register_forward_pre_hook(capture_input)

        def capture_output(module: nn.Module, args: tuple, output: tuple, index: int = index) -> None:
            if not cfg.eplv2_sequence_parallel:
                layer_outputs[index] = output[0].detach().reshape(rows, 4, 8).clone()

        layer.register_forward_hook(capture_output)
        layer.self_attn = Attention(index)
        layer.mlp._test_index = index
        layer.mlp.forward = MethodType(pointwise, layer.mlp)
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            norm.register_forward_pre_hook(
                lambda module, args, index=index: norm_shapes.append((index, args[0].shape[1]))
            )
    hidden = torch.randn(1, rows, 8)
    ids = torch.zeros(rows, dtype=torch.long)
    positions = torch.arange(rows)
    cfg.eplv2_sequence_parallel = False
    model._inputs_embeds = hidden.clone()
    context = _metadata()
    context.execution_contexts = {Glm5NextEplv2Metadata: Glm5NextEplv2Metadata(torch.ones(rows, dtype=torch.bool))}
    monkeypatch.setattr(glm5_next, "get_forward_context_or_none", lambda: context)
    expected = model(ids, positions)
    if capture:
        expected_hidden, expected_aux = expected
    else:
        expected_hidden, expected_aux = expected, None
    expected_hidden = expected_hidden.detach()
    cfg.eplv2_sequence_parallel = True
    for rank in range(world):
        cfg.tp_rank = cfg.ep_rank = rank
        layout = TokenParallelLayout(rows, world, rank)
        queue = []
        previous_sp = False
        for index, kind in enumerate(schedule):
            if kind == "sparse":
                queue.append(normalized[index])
            elif previous_sp:
                queue.append(incoming[index])
            previous_sp = kind == "sparse"
            if capture and kind == "sparse" and index in cfg.layers_to_capture:
                queue.append(layer_outputs[index])
        if previous_sp:
            queue.append(expected_hidden)
        calls = []

        def gather(
            local: torch.Tensor,
            dim: int,
            size: int,
            expected_queue: list[torch.Tensor] = queue,
            recorded: list = calls,
            token_layout: TokenParallelLayout = layout,
        ) -> torch.Tensor:
            assert dim == 0 and size == world
            reference = expected_queue[len(recorded)]
            torch.testing.assert_close(local, token_layout.shard(reference), rtol=1e-5, atol=1e-6)
            recorded.append(local.shape)
            return torch.cat([TokenParallelLayout(rows, world, r).shard(reference) for r in range(world)])

        def reduce(partial: torch.Tensor, size: int, group: str, owner: int = rank) -> torch.Tensor:
            assert size == world and group == "tp"
            return (partial * world).chunk(world, dim=0)[owner].clone()

        monkeypatch.setattr(glm5_next.distributed, "tp_all_gather", gather, raising=False)
        monkeypatch.setattr(glm5_next.distributed, "reduce_scatter", reduce, raising=False)
        norm_shapes.clear()
        layer_events.reset_mock()
        model._inputs_embeds = hidden.clone()
        actual = model(ids, positions)
        if capture:
            actual_hidden, actual_aux = actual
            torch.testing.assert_close(actual_aux, expected_aux, rtol=1e-5, atol=1e-6)
        else:
            actual_hidden = actual
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=1e-5, atol=1e-6)
        assert len(calls) == len(queue)
        assert [call.args[0] for call in layer_events.call_args_list] == list(range(len(schedule)))
        assert norm_shapes == [
            (i, layout.shard_tokens if kind == "sparse" else rows)
            for i, kind in enumerate(schedule)
            if kind != "sparse" or layout.valid_tokens
            for _ in range(2)
        ]
