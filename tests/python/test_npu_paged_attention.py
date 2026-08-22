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

"""CPU contract tests for the NPU dense MLA prefill dispatch."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch_npu

_mock_ops = MagicMock()


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


_mock_ops.rms_norm.side_effect = _rms_norm
sys.modules.setdefault("xllm.python.ops", _mock_ops)
sys.modules.setdefault("xllm.python.ops.compute", _mock_ops)

from xllm.python.attention.backend import MlaUnabsorbedPrefill
from xllm.python.attention.npu_paged_attention import (
    NpuPagedAttentionBackend,
)
from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    ForwardContext,
    forward_context,
)


def test_cache_geometry_skips_leading_linear_attention_slots():
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    paged_cache = torch.empty(17, 128, 1, 64)
    backend._kv_caches = [
        (None, None, None),
        (None, None, None),
        (paged_cache, torch.empty_like(paged_cache), None),
    ]

    assert backend.num_kv_blocks == 17
    assert backend.page_size == 128


def test_cache_geometry_defaults_without_paged_attention_slots():
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kv_caches = [(None, None, None)]

    assert backend.num_kv_blocks == 0
    assert backend.page_size == 1


def test_mla_only_graph_prepare_skips_mha_workspace(monkeypatch):
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._has_mha_layers = False
    backend._graph_workspace = None
    backend._graph_outputs = {}
    backend._graph_lses = {}
    backend._current_graph_output = None
    backend._current_graph_lse = None
    backend._mla_actual_seq_q = None
    backend._mla_actual_seq_kv = None
    metadata = SimpleNamespace(
        q_cu_seq_lens=None,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([0, 1], dtype=torch.int32),
        kv_seq_lens=None,
    )

    workspace = MagicMock(side_effect=AssertionError("MHA workspace requested"))
    monkeypatch.setattr(
        torch_npu,
        "_npu_fused_infer_attention_score_get_max_workspace",
        workspace,
    )

    backend.prepare(metadata, graph_mode=True)

    workspace.assert_not_called()
    assert backend._graph_workspace is None


def test_execute_mla_uses_torch_npu_cache_writer(monkeypatch):
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._metadata = SimpleNamespace(
        slot_mapping=torch.arange(3, dtype=torch.int32),
        is_chunked_prefill=False,
        is_prefill=True,
    )
    nope_cache = torch.empty(2, 8, 1, 6)
    rope_cache = torch.empty(2, 8, 1, 2)
    backend._kv_caches = [(nope_cache, rope_cache, torch.empty(0))]
    captured = {}
    expected = torch.randn(3, 4, 2)

    monkeypatch.setattr(
        torch_npu,
        "_npu_reshape_and_cache",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        backend,
        "_mla_unabsorbed_prefill",
        lambda *args: expected,
    )
    monkeypatch.setattr(backend, "use_unabsorbed_mla_prefill", lambda: True)

    q_latent = torch.randn(3, 4, 6)
    q_pe = torch.randn(3, 4, 2)
    k_latent = torch.randn(3, 1, 6)
    k_pe = torch.randn(3, 1, 2)
    unabsorbed = MlaUnabsorbedPrefill(
        query_nope=torch.randn(3, 4, 2),
        key_nope=torch.randn(3, 4, 2),
        value=torch.randn(3, 4, 2),
    )
    output = backend.execute_mla(
        q_latent,
        q_pe,
        k_latent,
        k_pe,
        SimpleNamespace(layer_id=0),
        unabsorbed_prefill=unabsorbed,
    )

    assert output is expected
    assert captured == {
        "key": k_latent,
        "value": k_pe,
        "key_cache": nope_cache,
        "value_cache": rope_cache,
        "slot_indices": backend._metadata.slot_mapping,
    }


def test_dense_mla_prefill_uses_split_positional_inputs_and_bf16(monkeypatch):
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._actual_seq_lens = [2, 5]
    backend._causal_mask = torch.ones(8, 8, dtype=torch.int8)

    q_latent = torch.randn(5, 4, 6, dtype=torch.float32)
    q_pe = torch.randn(5, 4, 2, dtype=torch.float32)
    k_latent = torch.randn(5, 1, 6, dtype=torch.float32)
    k_pe = torch.randn(5, 1, 2, dtype=torch.float32)
    metadata = SimpleNamespace()
    layer = SimpleNamespace(num_heads=4, num_kv_heads=1, scale=0.125)
    captured = {}

    def fake_fia(query, key, value, **kwargs):
        captured.update(
            query=query,
            key=key,
            value=value,
            **kwargs,
        )
        return torch.zeros_like(query), torch.empty(0, dtype=query.dtype)

    monkeypatch.setattr(
        torch.ops.npu,
        "npu_fused_infer_attention_score",
        fake_fia,
    )

    output = backend._mla_dense_prefill(
        q_latent,
        q_pe,
        k_latent,
        k_pe,
        metadata,
        layer,
    )

    assert output.shape == (5, 4, 6)
    assert output.dtype == torch.float32
    assert captured["query"].dtype == torch.bfloat16
    assert captured["key"].dtype == torch.bfloat16
    assert captured["value"].dtype == torch.bfloat16
    assert captured["query_rope"].dtype == torch.bfloat16
    assert captured["key_rope"].dtype == torch.bfloat16
    assert torch.equal(captured["key"], captured["value"])
    assert torch.equal(captured["query_rope"], q_pe.to(torch.bfloat16))
    assert torch.equal(captured["key_rope"], k_pe.to(torch.bfloat16))
    assert captured["atten_mask"] is backend._causal_mask
    assert captured["actual_seq_lengths"] == [2, 5]
    assert captured["actual_seq_lengths_kv"] == [2, 5]
    assert captured["num_heads"] == 4
    assert captured["num_key_value_heads"] == 1
    assert captured["scale"] == 0.125
    assert captured["input_layout"] == "TND"
    assert captured["sparse_mode"] == 3


def test_dense_mla_prefill_keeps_bf16_without_conversion(monkeypatch):
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._actual_seq_lens = None
    backend._causal_mask = torch.ones(4, 4, dtype=torch.int8)
    query = torch.randn(3, 2, 4, dtype=torch.bfloat16)
    key = torch.randn(3, 1, 4, dtype=torch.bfloat16)
    rope = torch.randn(3, 1, 2, dtype=torch.bfloat16)

    monkeypatch.setattr(
        torch.ops.npu,
        "npu_fused_infer_attention_score",
        lambda q, k, v, **kwargs: (torch.zeros_like(q), torch.empty(0)),
    )
    output = backend._mla_dense_prefill(
        query, rope.expand(-1, 2, -1), key, rope, SimpleNamespace(),
        SimpleNamespace(num_heads=2, num_kv_heads=1, scale=0.5),
    )

    assert output.dtype == torch.bfloat16
    assert output.shape == (3, 2, 4)


def test_dense_mla_prefill_pads_unsupported_512_dim_head_count(monkeypatch):
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._actual_seq_lens = [3]
    backend._causal_mask = torch.ones(4, 4, dtype=torch.int8)
    query = torch.randn(3, 24, 512, dtype=torch.bfloat16)
    query_rope = torch.randn(3, 24, 64, dtype=torch.bfloat16)
    key = torch.randn(3, 1, 512, dtype=torch.bfloat16)
    key_rope = torch.randn(3, 1, 64, dtype=torch.bfloat16)
    captured = {}

    def fake_fia(q, k, v, **kwargs):
        captured.update(query=q, key=k, value=v, **kwargs)
        return torch.zeros_like(q), torch.empty(0, dtype=q.dtype)

    monkeypatch.setattr(
        torch.ops.npu,
        "npu_fused_infer_attention_score",
        fake_fia,
    )
    output = backend._mla_dense_prefill(
        query,
        query_rope,
        key,
        key_rope,
        SimpleNamespace(),
        SimpleNamespace(num_heads=24, num_kv_heads=1, scale=0.5),
    )

    assert captured["query"].shape == (3, 32, 512)
    assert captured["query_rope"].shape == (3, 32, 64)
    assert captured["num_heads"] == 32
    assert output.shape == (3, 24, 512)


def test_unabsorbed_mla_prefill_matches_kimi_fia_contract(monkeypatch):
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._actual_seq_lens = [2, 5]
    backend._causal_mask = torch.ones(8, 8, dtype=torch.int8)
    query_nope = torch.randn(5, 48, 128, dtype=torch.bfloat16)
    query_position = torch.randn(5, 48, 64, dtype=torch.bfloat16)
    key_nope = torch.randn(5, 48, 128, dtype=torch.bfloat16)
    key_position = torch.randn(5, 1, 64, dtype=torch.bfloat16)
    value = torch.randn(5, 48, 128, dtype=torch.bfloat16)
    layer = SimpleNamespace(num_heads=48, scale=192**-0.5)
    captured = {}

    def fake_fia(query, key, input_value, **kwargs):
        captured.update(
            query=query,
            key=key,
            value=input_value,
            **kwargs,
        )
        return torch.zeros_like(input_value), torch.empty(0, dtype=query.dtype)

    monkeypatch.setattr(
        torch_npu,
        "npu_fused_infer_attention_score",
        fake_fia,
    )
    output = backend._mla_unabsorbed_prefill(
        MlaUnabsorbedPrefill(
            query_nope=query_nope,
            key_nope=key_nope,
            value=value,
        ),
        query_position,
        key_position,
        SimpleNamespace(),
        layer,
    )

    assert output.shape == (5, 48, 128)
    torch.testing.assert_close(
        captured["query"],
        torch.cat((query_nope, query_position), dim=-1),
    )
    torch.testing.assert_close(
        captured["key"],
        torch.cat((key_nope, key_position.expand(-1, 48, -1)), dim=-1),
    )
    assert captured["value"] is value
    assert captured["num_heads"] == 48
    assert captured["num_key_value_heads"] == 48
    assert captured["input_layout"] == "TND"
    assert captured["atten_mask"] is backend._causal_mask
    assert captured["sparse_mode"] == 3
    assert captured["scale"] == layer.scale
    assert captured["antiquant_mode"] == 0
    assert captured["antiquant_scale"] is None
    assert captured["block_table"] is None
    assert captured["block_size"] == 0
    assert captured["softmax_lse_flag"]
    assert captured["actual_seq_lengths"] == [2, 5]
    assert captured["actual_seq_lengths_kv"] == [2, 5]


def test_dense_mla_decode_uses_paged_latent_and_positional_caches(monkeypatch):
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._block_table_i32 = torch.tensor([[2, 1], [0, 3]], dtype=torch.int32)
    backend._actual_seq_q = [1, 2]
    backend._actual_seq_kv = [7, 5]

    q_latent = torch.randn(2, 4, 6, dtype=torch.float32)
    q_pe = torch.randn(2, 4, 2, dtype=torch.float32)
    nope_cache = torch.randn(4, 8, 1, 6, dtype=torch.bfloat16)
    rope_cache = torch.randn(4, 8, 1, 2, dtype=torch.bfloat16)
    layer = SimpleNamespace(num_heads=4, num_kv_heads=1, scale=0.125)
    captured = {}

    def fake_fia(query, key, value, **kwargs):
        captured.update(query=query, key=key, value=value, **kwargs)
        return torch.zeros_like(query), torch.empty(0, dtype=query.dtype)

    monkeypatch.setattr(
        torch.ops.npu,
        "npu_fused_infer_attention_score",
        fake_fia,
    )

    output = backend._mla_dense_decode(
        q_latent, q_pe, nope_cache, rope_cache, layer
    )

    assert output.shape == (2, 4, 6)
    assert output.dtype == torch.float32
    assert captured["query"].dtype == torch.bfloat16
    assert captured["key"].shape == (4, 8, 6)
    assert captured["key_rope"].shape == (4, 8, 2)
    assert captured["key"].dtype == torch.bfloat16
    assert captured["key_rope"].dtype == torch.bfloat16
    assert torch.equal(captured["key"], captured["value"])
    assert torch.equal(captured["query_rope"], q_pe.to(torch.bfloat16))
    assert captured["actual_seq_lengths"] == [1, 2]
    assert captured["actual_seq_lengths_kv"] == [7, 5]
    assert torch.equal(captured["block_table"], backend._block_table_i32)
    assert captured["block_size"] == 8
    assert captured["num_heads"] == 4
    assert captured["num_key_value_heads"] == 1
    assert captured["scale"] == 0.125
    assert captured["input_layout"] == "TND"
    assert captured["sparse_mode"] == 0
    assert captured["atten_mask"] is None


def test_dense_mla_decode_v2_matches_kimi_fia_contract(monkeypatch):
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._block_table_i32 = torch.tensor([[2, 1], [0, 3]], dtype=torch.int32)
    backend._actual_seq_kv = [7, 5]
    q_latent = torch.randn(2, 48, 512, dtype=torch.bfloat16)
    q_position = torch.randn(2, 48, 64, dtype=torch.bfloat16)
    nope_cache = torch.randn(4, 8, 1, 512, dtype=torch.bfloat16)
    rope_cache = torch.randn(4, 8, 1, 64, dtype=torch.bfloat16)
    layer = SimpleNamespace(num_heads=48, num_kv_heads=1, scale=576**-0.5)
    captured = {}
    operator_output = torch.arange(
        64 * 2 * 512,
        dtype=torch.float32,
    ).reshape(64, 2, 1, 512).to(torch.bfloat16)

    def fake_fia_v2(query, key, value, **kwargs):
        captured.update(query=query, key=key, value=value, **kwargs)
        return operator_output, torch.empty(0, dtype=query.dtype)

    monkeypatch.setattr(
        torch_npu,
        "npu_fused_infer_attention_score_v2",
        fake_fia_v2,
    )
    context = ForwardContext(backend, torch.device("cpu"))
    with forward_context(context):
        output = backend._mla_dense_decode_v2(
            q_latent,
            q_position,
            nope_cache,
            rope_cache,
            layer,
        )

    assert captured["query"].shape == (2, 64, 1, 512)
    assert captured["query_rope"].shape == (2, 64, 1, 64)
    assert captured["key"].shape == (4, 1, 8, 512)
    assert captured["value"].shape == (4, 1, 8, 512)
    assert captured["key_rope"].shape == (4, 1, 8, 64)
    torch.testing.assert_close(captured["query"][:, :48, 0], q_latent)
    torch.testing.assert_close(captured["query_rope"][:, :48, 0], q_position)
    assert torch.count_nonzero(captured["query"][:, 48:]) == 0
    assert torch.count_nonzero(captured["query_rope"][:, 48:]) == 0
    assert captured["num_query_heads"] == 64
    assert captured["num_key_value_heads"] == 1
    assert captured["input_layout"] == "BNSD_NBSD"
    assert captured["atten_mask"] is None
    assert captured["sparse_mode"] == 0
    assert captured["softmax_scale"] == layer.scale
    assert torch.equal(
        captured["block_table"],
        backend._block_table_i32,
    )
    assert captured["block_size"] == 8
    assert captured["actual_seq_qlen"] is None
    assert captured["actual_seq_kvlen"] == [7, 5]
    assert not captured["return_softmax_lse"]
    expected = operator_output[:48].view(48, 2, 512).transpose(0, 1)
    torch.testing.assert_close(output, expected)


def test_dense_mla_graph_keeps_outputs_layer_local(monkeypatch):
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._block_table_i32 = torch.tensor([[0], [1]], dtype=torch.int32)
    backend._actual_seq_kv = [3, 5]
    backend._mla_v2_graph_workspaces = {}
    backend._mla_v2_graph_outputs = {}
    q_latent = torch.randn(2, 4, 6, dtype=torch.bfloat16)
    q_position = torch.randn(2, 4, 2, dtype=torch.bfloat16)
    nope_cache = torch.randn(2, 8, 1, 6, dtype=torch.bfloat16)
    rope_cache = torch.randn(2, 8, 1, 2, dtype=torch.bfloat16)
    workspace_calls = []
    operator_outputs = []

    monkeypatch.setattr(
        torch_npu,
        "_npu_fused_infer_attention_score_v2_get_max_workspace",
        lambda *args, **kwargs: workspace_calls.append((args, kwargs))
        or torch.empty(16, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        torch_npu.npu_fused_infer_attention_score_v2,
        "out",
        lambda *args, **kwargs: operator_outputs.append(kwargs["out"]),
    )

    class _Event:
        def wait(self, _stream):
            pass

        def reset(self, _stream):
            pass

    monkeypatch.setattr(torch.npu, "ExternalEvent", _Event)
    monkeypatch.setattr(torch.npu, "graph_task_group_begin", lambda _stream: None)
    monkeypatch.setattr(torch.npu, "graph_task_group_end", lambda _stream: object())

    capture = AclGraphCaptureContext(stream=object(), tasks=[])
    context = ForwardContext(
        backend,
        torch.device("cpu"),
        acl_graph=capture,
    )
    layer_3 = SimpleNamespace(
        layer_id=3,
        num_heads=4,
        num_kv_heads=1,
        scale=0.125,
    )
    layer_7 = SimpleNamespace(
        layer_id=7,
        num_heads=4,
        num_kv_heads=1,
        scale=0.125,
    )
    with forward_context(context):
        output_3 = backend._mla_dense_decode_v2(
            q_latent, q_position, nope_cache, rope_cache, layer_3
        )
        output_7 = backend._mla_dense_decode_v2(
            q_latent, q_position, nope_cache, rope_cache, layer_7
        )

    assert output_3.shape == output_7.shape == (2, 4, 6)
    assert output_3.data_ptr() != output_7.data_ptr()
    assert len(backend._mla_v2_graph_outputs) == 2
    assert len(backend._mla_v2_graph_workspaces) == 1
    assert len(workspace_calls) == 1
    assert len(capture.tasks) == 2

    # Task updates run after the Python method has reshaped its public result.
    # They must retain the original four-dimensional FIA output tensors.
    capture.tasks[0].update()
    assert len(operator_outputs) == 3
    assert operator_outputs[-1][0].shape == (4, 2, 1, 6)
    assert operator_outputs[-1][1].shape == (2,)


def test_dense_mla_chunked_prefill_uses_paged_caches_and_total_kv_lens(
    monkeypatch,
):
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._actual_seq_lens = [2, 5]
    backend._causal_mask = torch.ones(16, 16, dtype=torch.int8)
    backend._block_table_i32 = torch.tensor(
        [[2, 1, -1], [0, 3, 4]], dtype=torch.int32
    )

    q_latent = torch.randn(5, 4, 6, dtype=torch.float32)
    q_pe = torch.randn(5, 4, 2, dtype=torch.float32)
    nope_cache = torch.randn(5, 8, 1, 6, dtype=torch.bfloat16)
    rope_cache = torch.randn(5, 8, 1, 2, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        q_cu_seq_lens=torch.tensor([0, 2, 5], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([7, 11], dtype=torch.int32),
        block_table=backend._block_table_i32,
    )
    layer = SimpleNamespace(num_heads=4, num_kv_heads=1, scale=0.125)
    captured = {}

    def fake_fia(query, key, value, **kwargs):
        captured.update(query=query, key=key, value=value, **kwargs)
        return torch.zeros_like(query), torch.empty(0, dtype=query.dtype)

    monkeypatch.setattr(
        torch.ops.npu,
        "npu_fused_infer_attention_score",
        fake_fia,
    )

    output = backend._mla_dense_chunked_prefill(
        q_latent, q_pe, nope_cache, rope_cache, metadata, layer
    )

    assert output.shape == (5, 4, 6)
    assert output.dtype == torch.float32
    assert captured["query"].dtype == torch.bfloat16
    assert captured["key"].shape == (5, 8, 6)
    assert captured["key_rope"].shape == (5, 8, 2)
    assert captured["key"].dtype == torch.bfloat16
    assert captured["key_rope"].dtype == torch.bfloat16
    assert torch.equal(captured["key"], captured["value"])
    assert torch.equal(captured["query_rope"], q_pe.to(torch.bfloat16))
    assert captured["actual_seq_lengths"] == [2, 5]
    assert captured["actual_seq_lengths_kv"] == [7, 11]
    assert torch.equal(captured["block_table"], backend._block_table_i32)
    assert captured["block_size"] == 8
    assert captured["num_heads"] == 4
    assert captured["num_key_value_heads"] == 1
    assert captured["scale"] == 0.125
    assert captured["input_layout"] == "TND"
    assert captured["sparse_mode"] == 3
    assert captured["atten_mask"] is backend._causal_mask


def test_chunked_kv_seq_lens_accepts_cumulative_fallback():
    metadata = SimpleNamespace(
        q_cu_seq_lens=torch.tensor([0, 2, 5], dtype=torch.int32),
        kv_seq_lens_host=torch.tensor([0, 7, 18], dtype=torch.int32),
        block_table=torch.zeros(2, 2, dtype=torch.int32),
    )

    assert NpuPagedAttentionBackend._chunked_kv_seq_lens(metadata) == [7, 11]
