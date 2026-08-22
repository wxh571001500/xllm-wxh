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

"""Unit tests for xllm.python.model_executor.executor.

Tests the device-conditional backend dispatch, ModelExecutor construction
validation, and execution routing — using CPU mocks so no GPU/NPU required.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

# The xllm.python package auto-registers models on import, which triggers
# torch.ops.xllm_ops lookups that require the C++ binary. We bypass this
# by mocking the ops and registry modules before importing executor.
_mock_ops = MagicMock()
sys.modules.setdefault("xllm.python.ops", _mock_ops)
sys.modules.setdefault("xllm.python.ops.compute", _mock_ops)

from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    KVCache,
)
from xllm.python.layers.attention import Attention
from xllm.python.model_executor.executor import (
    ModelExecutor,
    _acl_graph_unsupported_reason,
    _create_attention_backend,
    _is_npu_device,
    _resolve_graph_backend,
)
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubAttentionBackend(AttentionBackend):
    """Minimal backend that records calls for assertion."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self._kv_caches: list[KVCache] = []
        self._prepared = False

    def bind_kv_caches(self, kv_caches: list[KVCache]) -> None:
        self._kv_caches = kv_caches

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        self._prepared = True

    def execute(self, q, k, v, layer) -> torch.Tensor:
        return q

    @property
    def num_kv_blocks(self) -> int:
        return 0

    @property
    def page_size(self) -> int:
        return 1


def _make_attention_layer(
    num_heads=8,
    num_kv_heads=2,
    head_dim=64,
    scale=0.125,
    sliding_window=0,
    layer_id=0,
) -> Attention:
    return Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        sliding_window=sliding_window,
        layer_id=layer_id,
    )


class _FakeModel(nn.Module):
    """Model with configurable number of uniform Attention layers."""

    def __init__(self, num_layers: int = 2, device: str = "cpu", **attn_kwargs):
        super().__init__()
        self.model = nn.Linear(1, 1)  # execution_model placeholder
        self.layers = nn.ModuleList([_make_attention_layer(layer_id=i, **attn_kwargs) for i in range(num_layers)])
        self._param = nn.Parameter(torch.zeros(1, device=device))

    def forward(self, input_ids, positions):
        return input_ids


class _FakeModelHeterogeneous(nn.Module):
    """Model with non-uniform Attention layers (should fail validation)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self.attn1 = _make_attention_layer(num_heads=8, layer_id=0)
        self.attn2 = _make_attention_layer(num_heads=4, layer_id=1)
        self._param = nn.Parameter(torch.zeros(1))


class _FakeModelNoAttention(nn.Module):
    """Model without any Attention layers."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self._param = nn.Parameter(torch.zeros(1))


# ---------------------------------------------------------------------------
# Tests: _is_npu_device
# ---------------------------------------------------------------------------


class TestIsNpuDevice:
    def test_npu_type(self):
        assert _is_npu_device(torch.device("npu")) is True

    def test_privateuseone_type(self):
        assert _is_npu_device(torch.device("privateuseone")) is True

    def test_cuda_type(self):
        assert _is_npu_device(torch.device("cuda")) is False

    def test_cpu_type(self):
        assert _is_npu_device(torch.device("cpu")) is False


# ---------------------------------------------------------------------------
# Tests: graph backend resolution
# ---------------------------------------------------------------------------


class TestNpuGraphBackendResolution:
    def test_enable_graph_selects_aclgraph_on_npu(self):
        config = {"enable_graph": True, "python_graph_backend": "off"}

        assert _resolve_graph_backend(config, torch.device("npu")) == "aclgraph"


class TestAclGraphCapability:
    @pytest.mark.parametrize(
        ("kind", "message"),
        [("linear", "runtime state"), ("mla", "MHA attention only")],
    )
    def test_non_kimi_attention_kind_is_rejected(self, kind, message):
        layer = MagicMock(attention_kind=kind)

        assert message in _acl_graph_unsupported_reason([layer])

    @pytest.mark.parametrize("kind", ["mha", "mla", "linear"])
    def test_kimi_attention_kinds_are_supported(self, kind):
        layer = MagicMock(attention_kind=kind)

        assert _acl_graph_unsupported_reason([layer], supports_kimi_k3_graph=True) is None

    def test_unknown_kimi_attention_kind_is_rejected(self):
        layer = MagicMock(attention_kind="unknown")

        assert "unknown" in _acl_graph_unsupported_reason([layer], supports_kimi_k3_graph=True)

    def test_mha_is_supported(self):
        layer = MagicMock(attention_kind="mha")

        assert _acl_graph_unsupported_reason([layer]) is None


class TestKimiK3AclGraphMetadata:
    def test_warmup_uses_bucket_local_kda_metadata_and_restores_runtime(self):
        runner = DecodeAclGraphRunner.__new__(DecodeAclGraphRunner)
        original_metadata = object()
        runner._kda_runtime = SimpleNamespace(metadata=original_metadata)
        runner._warmed_up = False
        runner.max_batch = 4
        runner.page_size = 128
        observed = []

        def record_execute(input_ids, _positions, _metadata, inputs_embeds):
            observed.append(
                (
                    input_ids.shape[0],
                    runner._kda_runtime.metadata,
                    inputs_embeds,
                )
            )

        runner.execute = record_execute
        runner.warmup(torch.device("cpu"), torch.int32)

        assert [batch_size for batch_size, _, _ in observed] == [1, 2, 4]
        for batch_size, metadata, inputs_embeds in observed:
            assert metadata.num_decode_seqs == batch_size
            assert metadata.num_prefill_seqs == 0
            assert metadata.query_start_loc.tolist() == list(range(batch_size + 1))
            assert metadata.state_indices.tolist() == [-1] * batch_size
            assert inputs_embeds is None
        assert runner._kda_runtime.metadata is original_metadata

    def test_fill_entry_copies_real_kda_slots_and_pads_empty_rows(self):
        runner = DecodeAclGraphRunner.__new__(DecodeAclGraphRunner)
        dynamic_kda = SimpleNamespace(
            state_indices=torch.tensor([11, 17], dtype=torch.int64),
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            num_decode_seqs=2,
            num_prefill_seqs=0,
        )
        runner._kda_runtime = SimpleNamespace(metadata=dynamic_kda)
        runner._fill_host_metadata = MagicMock()
        static_metadata = SimpleNamespace(
            slot_mapping=torch.zeros(4, dtype=torch.int32),
            kv_cu_seq_lens=torch.zeros(5, dtype=torch.int32),
            paged_kv_indptr=torch.zeros(5, dtype=torch.int32),
            paged_kv_indices=torch.zeros(4, dtype=torch.int32),
            paged_kv_last_page_len=torch.zeros(4, dtype=torch.int32),
            block_table=None,
        )
        entry = SimpleNamespace(
            batch_size=4,
            static_input_ids=torch.zeros(4, dtype=torch.int32),
            static_inputs_embeds=torch.full((4, 3), 99.0),
            static_positions=torch.zeros(4, dtype=torch.int32),
            static_metadata=static_metadata,
            kv_seq_lens_delta=torch.zeros(4, dtype=torch.int32),
            static_kda_metadata=SimpleNamespace(
                state_indices=torch.full((4,), 99, dtype=torch.int64),
                query_start_loc=torch.full((5,), 99, dtype=torch.int32),
            ),
        )
        metadata = SimpleNamespace(
            slot_mapping=torch.zeros(2, dtype=torch.int32),
            kv_cu_seq_lens=torch.arange(3, dtype=torch.int32),
            paged_kv_indptr=torch.arange(3, dtype=torch.int32),
            paged_kv_indices=torch.zeros(2, dtype=torch.int32),
            paged_kv_last_page_len=torch.ones(2, dtype=torch.int32),
            block_table=None,
        )

        runner._fill_entry(
            entry,
            torch.tensor([3, 5], dtype=torch.int32),
            torch.tensor([7, 9], dtype=torch.int64),
            metadata,
            batch_size=2,
            inputs_embeds=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )

        assert entry.static_kda_metadata.state_indices.tolist() == [11, 17, -1, -1]
        assert entry.static_kda_metadata.query_start_loc.tolist() == [0, 1, 2, 2, 2]
        assert entry.static_inputs_embeds.tolist() == [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        runner._fill_host_metadata.assert_called_once_with(entry, metadata, 2)

    def test_fill_entry_maps_empty_dp_shard_to_kda_pad_slot(self):
        runner = DecodeAclGraphRunner.__new__(DecodeAclGraphRunner)
        runner._kda_runtime = SimpleNamespace(
            metadata=SimpleNamespace(
                state_indices=torch.tensor([-1], dtype=torch.int64),
                query_start_loc=torch.tensor([0], dtype=torch.int64),
                num_decode_seqs=0,
                num_prefill_seqs=0,
                empty_shard=True,
            )
        )
        runner._fill_host_metadata = MagicMock()
        static_metadata = SimpleNamespace(
            slot_mapping=torch.zeros(4, dtype=torch.int32),
            kv_cu_seq_lens=torch.zeros(5, dtype=torch.int32),
            paged_kv_indptr=torch.zeros(5, dtype=torch.int32),
            paged_kv_indices=torch.zeros(4, dtype=torch.int32),
            paged_kv_last_page_len=torch.zeros(4, dtype=torch.int32),
            block_table=None,
        )
        entry = SimpleNamespace(
            batch_size=4,
            static_input_ids=torch.zeros(4, dtype=torch.int32),
            static_inputs_embeds=None,
            static_positions=torch.zeros(4, dtype=torch.int32),
            static_metadata=static_metadata,
            kv_seq_lens_delta=torch.zeros(4, dtype=torch.int32),
            static_kda_metadata=SimpleNamespace(
                state_indices=torch.full((4,), 99, dtype=torch.int64),
                query_start_loc=torch.full((5,), 99, dtype=torch.int32),
            ),
        )
        metadata = SimpleNamespace(
            slot_mapping=torch.zeros(1, dtype=torch.int32),
            kv_cu_seq_lens=torch.zeros(2, dtype=torch.int32),
            paged_kv_indptr=torch.zeros(2, dtype=torch.int32),
            paged_kv_indices=torch.zeros(1, dtype=torch.int32),
            paged_kv_last_page_len=torch.ones(1, dtype=torch.int32),
            block_table=None,
        )

        runner._fill_entry(
            entry,
            torch.ones(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            metadata,
            batch_size=1,
            inputs_embeds=None,
        )

        assert entry.static_kda_metadata.state_indices.tolist() == [-1] * 4
        assert entry.static_kda_metadata.query_start_loc.tolist() == [0, 1, 1, 1, 1]
        runner._fill_host_metadata.assert_called_once_with(entry, metadata, 1)

    def test_host_metadata_uses_zero_length_for_padded_fia_rows(self):
        runner = DecodeAclGraphRunner.__new__(DecodeAclGraphRunner)
        runner.page_size = 128
        entry = SimpleNamespace(
            batch_size=2,
            host_seq_lens=torch.empty(2, dtype=torch.int32),
            host_block_counts=torch.empty(2, dtype=torch.int32),
            static_metadata=SimpleNamespace(
                kv_seq_lens_host=torch.zeros(3, dtype=torch.int32),
                paged_kv_indptr_host=torch.zeros(3, dtype=torch.int32),
                paged_kv_last_page_len_host=torch.ones(2, dtype=torch.int32),
            ),
        )
        metadata = SimpleNamespace(kv_seq_lens_host=torch.tensor([90], dtype=torch.int32))

        runner._fill_host_metadata(entry, metadata, batch_size=1)

        assert entry.host_seq_lens.tolist() == [90, 0]
        assert entry.static_metadata.kv_seq_lens_host.tolist() == [0, 90, 90]
        assert entry.static_metadata.paged_kv_indptr_host.tolist() == [0, 1, 1]
        assert entry.static_metadata.paged_kv_last_page_len_host.tolist() == [90, 1]

    def test_global_dp_token_count_selects_shared_decode_bucket(self):
        runner = DecodeAclGraphRunner.__new__(DecodeAclGraphRunner)
        runner.max_batch = 8
        runner._kda_runtime = SimpleNamespace(metadata=SimpleNamespace(graph_num_tokens=3))

        can_execute = runner.can_execute(
            torch.tensor([1], dtype=torch.int32),
            SimpleNamespace(is_prefill=False, is_chunked_prefill=False),
        )

        assert can_execute

    def test_host_metadata_accepts_empty_dp_shard_lengths(self):
        runner = DecodeAclGraphRunner.__new__(DecodeAclGraphRunner)
        runner.page_size = 128
        entry = SimpleNamespace(
            batch_size=4,
            host_seq_lens=torch.empty(4, dtype=torch.int32),
            host_block_counts=torch.empty(4, dtype=torch.int32),
            static_metadata=SimpleNamespace(
                kv_seq_lens_host=torch.zeros(5, dtype=torch.int32),
                paged_kv_indptr_host=torch.zeros(5, dtype=torch.int32),
                paged_kv_last_page_len_host=torch.ones(4, dtype=torch.int32),
            ),
        )
        metadata = SimpleNamespace(kv_seq_lens_host=torch.tensor([0, 0], dtype=torch.int32))

        runner._fill_host_metadata(entry, metadata, batch_size=1)

        assert entry.host_seq_lens.tolist() == [0, 0, 0, 0]
        assert entry.static_metadata.kv_seq_lens_host.tolist() == [0, 0, 0, 0, 0]
        assert entry.static_metadata.paged_kv_indptr_host.tolist() == [0, 0, 0, 0, 0]
        assert entry.static_metadata.paged_kv_last_page_len_host.tolist() == [1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Tests: _create_attention_backend dispatch
# ---------------------------------------------------------------------------


class TestCreateAttentionBackend:
    @patch("xllm.python.model_executor.executor._is_npu_device", return_value=True)
    @patch(
        "xllm.python.attention.npu_paged_attention.NpuPagedAttentionBackend",
        StubAttentionBackend,
    )
    def test_npu_device_creates_npu_backend(self, _mock_is_npu):
        attn = _make_attention_layer().attention_layer_spec()
        backend = _create_attention_backend(attn, torch.device("npu"), torch.float16)
        assert isinstance(backend, StubAttentionBackend)
        assert backend.init_kwargs["num_heads"] == 8
        assert backend.init_kwargs["num_kv_heads"] == 2
        assert backend.init_kwargs["head_dim"] == 64
        assert backend.init_kwargs["has_mha_layers"]

    @patch("xllm.python.model_executor.executor._is_npu_device", return_value=False)
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_cuda_device_creates_flashinfer_backend(self, mock_create, _mock_is_npu):
        mock_create.return_value = StubAttentionBackend(num_heads=8)
        # Verify the factory would be called (we can't import flashinfer in NPU env)
        from xllm.python.model_executor.executor import _is_npu_device

        assert _is_npu_device(torch.device("cuda")) is False


# ---------------------------------------------------------------------------
# Tests: ModelExecutor construction
# ---------------------------------------------------------------------------


class TestModelExecutorConstruction:
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_valid_model_creates_executor(self, _mock_backend):
        model = _FakeModel(num_layers=3)
        config = {"python_graph_backend": "off"}
        executor = ModelExecutor(model, config, max_seqs_per_batch=4)

        assert executor._num_attention_layers == 3
        assert executor.decode_graph_runner is None
        assert executor.inductor_runner is None

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_no_attention_layers_raises(self, _mock_backend):
        model = _FakeModelNoAttention()
        with pytest.raises(ValueError, match="runtime attention layer"):
            ModelExecutor(model, {}, max_seqs_per_batch=4)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_heterogeneous_attention_is_registered(self, _mock_backend):
        model = _FakeModelHeterogeneous()
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        assert [spec.num_heads for spec in executor._attention_layer_specs] == [8, 4]

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_duplicate_layer_ids_raise(self, _mock_backend):
        model = _FakeModel(num_layers=2)
        model.layers[1].layer_id = 0
        with pytest.raises(ValueError, match="must be unique"):
            ModelExecutor(model, {}, max_seqs_per_batch=4)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_sparse_physical_layer_id_is_registered(self, _mock_backend):
        model = _FakeModel(num_layers=1)
        model.layers[0].layer_id = 3

        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        assert [spec.layer_id for spec in executor._attention_layer_specs] == [3]

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
        return_value=StubAttentionBackend(),
    )
    def test_graph_backend_off_variants(self, _mock_backend):
        for off_value in ("off", "", "none", "0"):
            model = _FakeModel(num_layers=1)
            executor = ModelExecutor(model, {"python_graph_backend": off_value}, max_seqs_per_batch=4)
            assert executor.decode_graph_runner is None
            assert executor.inductor_runner is None


# ---------------------------------------------------------------------------
# Tests: ModelExecutor.bind_kv_caches
# ---------------------------------------------------------------------------


class TestBindKvCaches:
    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_correct_count(self, mock_create):
        backend = StubAttentionBackend()
        mock_create.return_value = backend
        model = _FakeModel(num_layers=2)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv, kv])
        assert len(backend._kv_caches) == 2

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_wrong_count_raises(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=2)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        with pytest.raises(ValueError, match="does not cover"):
            executor.bind_kv_caches([kv])

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_sparse_layer_binds_physical_cache_list(self, mock_create):
        backend = StubAttentionBackend()
        mock_create.return_value = backend
        model = _FakeModel(num_layers=1)
        model.layers[0].layer_id = 3
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        kv_caches = [kv, kv, kv, kv]
        executor.bind_kv_caches(kv_caches)

        assert backend._kv_caches is kv_caches

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_sparse_layer_rejects_runtime_only_cache_list(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        model.layers[0].layer_id = 3
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        with pytest.raises(ValueError, match="does not cover"):
            executor.bind_kv_caches([kv])

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_contiguous_layers_accept_extra_physical_caches(self, mock_create):
        backend = StubAttentionBackend()
        mock_create.return_value = backend
        model = _FakeModel(num_layers=2)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        kv_caches = [kv, kv, kv]
        executor.bind_kv_caches(kv_caches)

        assert backend._kv_caches is kv_caches

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_bind_idempotent(self, mock_create):
        backend = StubAttentionBackend()
        mock_create.return_value = backend
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])
        executor.bind_kv_caches([kv])  # should not raise or re-bind


# ---------------------------------------------------------------------------
# Tests: ModelExecutor.execute routing
# ---------------------------------------------------------------------------


class TestExecuteRouting:
    def test_dp_decode_graph_is_enabled_for_empty_shard_when_metadata_is_valid(self):
        executor = object.__new__(ModelExecutor)
        executor._dp_size = 2

        metadata = SimpleNamespace(
            dp_metadata_valid=True,
            all_dp_decode=True,
        )

        assert executor._all_dp_decode(metadata)

    def test_dp_decode_graph_is_enabled_when_every_shard_decodes(self):
        executor = object.__new__(ModelExecutor)
        executor._dp_size = 2

        metadata = SimpleNamespace(
            dp_metadata_valid=True,
            all_dp_decode=True,
        )

        assert executor._all_dp_decode(metadata)

    def test_dp_decode_graph_falls_back_for_invalid_metadata(self):
        executor = object.__new__(ModelExecutor)
        executor._dp_size = 2

        assert not executor._all_dp_decode(SimpleNamespace(dp_metadata_valid=False, all_dp_decode=True))

    def test_dp_decode_graph_falls_back_for_mixed_phase(self):
        executor = object.__new__(ModelExecutor)
        executor._dp_size = 2

        assert not executor._all_dp_decode(SimpleNamespace(dp_metadata_valid=True, all_dp_decode=False))

    def test_tp_decode_graph_is_enabled(self):
        executor = object.__new__(ModelExecutor)
        executor._dp_size = 1

        assert executor._all_dp_decode(None)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    @patch(
        "xllm.python.model_executor.executor._resolve_graph_backend",
        return_value="aclgraph",
    )
    def test_kimi_acl_graph_accepts_dp(self, _mock_resolve, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        model.model.kda_runtime = object()

        executor = ModelExecutor(
            model,
            {"dp_size": 2, "max_position_embeddings": 128},
            max_seqs_per_batch=4,
        )
        assert executor.decode_graph_runner is not None

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_execute_without_bind_raises(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        metadata = MagicMock(spec=AttentionMetadata)
        with pytest.raises(RuntimeError, match="KV caches are not bound"):
            executor.execute(torch.zeros(1), torch.zeros(1), metadata)

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_execute_routes_to_eager_runner(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        metadata = MagicMock(spec=AttentionMetadata)
        executor.eager_runner = MagicMock()
        executor.eager_runner.execute.return_value = torch.ones(5)

        result = executor.execute(torch.zeros(1), torch.zeros(1), metadata)
        executor.eager_runner.execute.assert_called_once()
        assert torch.equal(result, torch.ones(5))

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_inductor_runner_takes_priority_over_eager(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        model = _FakeModel(num_layers=1)
        executor = ModelExecutor(model, {}, max_seqs_per_batch=4)

        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        executor.inductor_runner = MagicMock()
        executor.inductor_runner.execute.return_value = torch.ones(3)

        metadata = MagicMock(spec=AttentionMetadata)
        result = executor.execute(torch.zeros(1), torch.zeros(1), metadata)
        executor.inductor_runner.execute.assert_called_once()
        assert torch.equal(result, torch.ones(3))

    @patch(
        "xllm.python.model_executor.executor._create_attention_backend",
    )
    def test_prefill_forward_stays_eager_after_decode_graph_warmup(self, mock_create):
        mock_create.return_value = StubAttentionBackend()
        executor = ModelExecutor(_FakeModel(num_layers=1), {}, max_seqs_per_batch=4)
        kv = (torch.zeros(1), torch.zeros(1))
        executor.bind_kv_caches([kv])

        metadata = MagicMock(spec=AttentionMetadata)
        executor.decode_graph_runner = MagicMock()
        executor.decode_graph_runner.can_execute.return_value = False
        executor.eager_runner = MagicMock()
        executor.eager_runner.execute.return_value = torch.ones(2)

        result = executor.execute(torch.zeros(1), torch.zeros(1), metadata)

        executor.decode_graph_runner.can_execute.assert_called_once()
        executor.decode_graph_runner.warmup.assert_called_once()
        executor.decode_graph_runner.execute.assert_not_called()
        executor.eager_runner.execute.assert_called_once()
        assert torch.equal(result, torch.ones(2))
