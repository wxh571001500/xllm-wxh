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

_mock_ops = MagicMock()
sys.modules.setdefault("xllm.python.ops", _mock_ops)
sys.modules.setdefault("xllm.python.ops.compute", _mock_ops)

from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend  # noqa: E402


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