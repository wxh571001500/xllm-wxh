# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.modules.setdefault("torch_npu", types.ModuleType("torch_npu"))

from xllm.python.attention import npu_paged_attention as npu_paged_attention_module  # noqa: E402
from xllm.python.attention.backend import resolve_linear_state_io_indices  # noqa: E402
from xllm.python.attention.kv_shard_layout import (  # noqa: E402
    KVShardLayout,
    localize_pool_write_block_table,
    replicate_pool_write_block_table,
)
from xllm.python.attention.npu_paged_attention import (  # noqa: E402
    NpuPagedAttentionBackend,
    _build_stable_sfa_page_layout,
)
from xllm.python.model_executor.cp_utils import (  # noqa: E402
    build_cp_context,
    cp_index_write_mode,
)
from xllm.python.model_executor.runners.decode_acl_graph import (  # noqa: E402
    _StaticAttentionMetadata,
)


def test_decode_prepare_does_not_require_cp_metadata_fields() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._is_mla = True
    backend._kv_owner_representatives = torch.tensor([0])
    backend._materialized_block_table = torch.tensor([[0]], dtype=torch.int32)
    backend._sfa_page_layout = _build_stable_sfa_page_layout(
        torch.tensor([[0]], dtype=torch.int32),
    )

    backend._prepare_kv_shard_materialization(SimpleNamespace(is_prefill=False, is_chunked_prefill=False))

    assert backend._kv_owner_representatives is None
    assert backend._materialized_block_table is None
    assert backend._sfa_page_layout is None


def test_mla_index_context_accepts_decode_graph_static_metadata() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    slot_mapping = torch.arange(2, dtype=torch.int64)
    backend._metadata = _StaticAttentionMetadata(
        slot_mapping=slot_mapping,
        paged_kv_indptr=torch.arange(3, dtype=torch.int32),
        paged_kv_indices=torch.arange(2, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(2, dtype=torch.int32),
    )
    backend._block_table_i32 = torch.arange(2, dtype=torch.int32).view(2, 1)
    backend._mla_actual_seq_q = torch.arange(1, 3, dtype=torch.int32)
    backend._mla_actual_seq_kv = torch.arange(1, 3, dtype=torch.int32)
    backend._kv_caches = [
        SimpleNamespace(
            index=torch.zeros(2, 1, 1),
            index_scale=None,
            kpool_tail=None,
        )
    ]
    backend._kpool_cache_triton_compatible = (False,)

    with patch(
        "xllm.python.attention.npu_paged_attention.get_forward_context",
        return_value=SimpleNamespace(cp_context=None),
    ):
        context = backend.mla_index_context(SimpleNamespace(layer_id=0))

    assert context.slot_mapping.data_ptr() == slot_mapping.data_ptr()


def test_owner_local_index_write_redirects_non_owned_slots_to_padding() -> None:
    block_size = 2
    cache = torch.full((3, block_size, 1, 1), -1.0)
    slots = torch.tensor([2, -1, 3, -1, 4], dtype=torch.int64)
    values = torch.arange(5, dtype=torch.float32).view(-1, 1)

    NpuPagedAttentionBackend._update_mla_index_cache(
        cache,
        None,
        slots,
        values,
        None,
    )

    torch.testing.assert_close(cache.view(-1)[block_size:], torch.tensor([0.0, 2.0, 4.0, -1.0]))


def test_quantized_index_write_uses_shared_padding_slots() -> None:
    block_size = 2
    cache = torch.arange(12, dtype=torch.int8).view(3, block_size, 1, 2)
    scale_cache = torch.arange(6, dtype=torch.float16).view(3, block_size, 1)
    slots = torch.tensor([2, -1, 3, -1, 5], dtype=torch.int64)
    values = torch.arange(50, 60, dtype=torch.int8).view(-1, 2)
    scales = torch.arange(100, 105, dtype=torch.float16).view(-1, 1)
    expected_cache = cache.clone().view(-1, 2)
    expected_scale_cache = scale_cache.clone().view(-1, 1)
    expected_cache[2] = values[0]
    expected_cache[3] = values[2]
    expected_cache[5] = values[4]
    expected_scale_cache[2] = scales[0]
    expected_scale_cache[3] = scales[2]
    expected_scale_cache[5] = scales[4]

    NpuPagedAttentionBackend._update_mla_index_cache(
        cache,
        scale_cache,
        slots,
        values,
        scales,
    )

    torch.testing.assert_close(cache.view(-1, 2)[block_size:], expected_cache[block_size:])
    torch.testing.assert_close(
        scale_cache.view(-1, 1)[block_size:],
        expected_scale_cache[block_size:],
    )


@pytest.mark.parametrize(
    ("cache_dtype", "with_scales"),
    [
        pytest.param(torch.float32, False, id="float"),
        pytest.param(torch.int8, True, id="w8a8"),
    ],
)
def test_all_invalid_index_write_only_updates_padding_block(
    cache_dtype: torch.dtype,
    with_scales: bool,
) -> None:
    block_size = 2
    cache = torch.arange(12, dtype=cache_dtype).view(3, block_size, 1, 2)
    original_cache = cache.clone()
    scale_cache = torch.arange(6, dtype=torch.float16).view(3, block_size, 1) if with_scales else None
    original_scale_cache = scale_cache.clone() if scale_cache is not None else None
    slots = torch.full((4,), -1, dtype=torch.int64)
    values = torch.arange(40, 48, dtype=cache_dtype).view(4, 2)
    scales = torch.arange(50, 54, dtype=torch.float16).view(4, 1) if with_scales else None

    NpuPagedAttentionBackend._update_mla_index_cache(
        cache,
        scale_cache,
        slots,
        values,
        scales,
    )

    torch.testing.assert_close(
        cache.view(-1, 2)[block_size:],
        original_cache.view(-1, 2)[block_size:],
    )
    if scale_cache is not None and original_scale_cache is not None:
        torch.testing.assert_close(
            scale_cache.view(-1, 1)[block_size:],
            original_scale_cache.view(-1, 1)[block_size:],
        )


def test_proper_divisor_materialization_selects_one_replica_per_owner() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._is_mla = True
    backend.device = torch.device("cpu")
    backend._block_table_i32 = torch.tensor([[3, 7, -1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        has_kv_shard=True,
        kv_split_size=2,
        kv_split_rank=0,
    )
    cp_context = SimpleNamespace(cp_size=4)

    def all_gather(tensor: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0
        assert world_size == 4
        assert group_name == "cp"
        if tensor.shape == (1,):
            return torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        return torch.cat([tensor + rank * 100 for rank in range(4)], dim=0)

    cache = torch.arange(16, dtype=torch.float32).view(8, 2, 1)
    with (
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            return_value=4,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            side_effect=all_gather,
            create=True,
        ),
    ):
        backend._prepare_kv_shard_materialization(metadata)
        materialized, block_table = backend._materialize_cp_cache(cache, metadata, cp_context)

    assert block_table.tolist() == [[0, 1, 2, 3, -1, -1]]
    torch.testing.assert_close(materialized[0], cache[3])
    torch.testing.assert_close(materialized[1], cache[3] + 200)
    torch.testing.assert_close(materialized[2], cache[7])
    torch.testing.assert_close(materialized[3], cache[7] + 200)


def test_kv1_materialization_keeps_persistent_cache_view() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._is_mla = True
    backend.device = torch.device("cpu")
    backend._block_table_i32 = torch.tensor([[3, 1, -1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        has_kv_shard=True,
        kv_split_size=1,
        kv_split_rank=0,
    )
    cp_context = SimpleNamespace(cp_size=4)
    cache = torch.arange(8, dtype=torch.float32).view(4, 2, 1)

    with patch(
        "xllm.python.attention.npu_paged_attention.distributed.all_gather",
        create=True,
    ) as all_gather:
        backend._prepare_kv_shard_materialization(metadata)
        materialized, block_table = backend._materialize_cp_cache(cache, metadata, cp_context)

    all_gather.assert_not_called()
    assert materialized.data_ptr() == cache.data_ptr()
    assert block_table.data_ptr() == backend._block_table_i32.data_ptr()


def test_stable_sfa_layout_handles_multiple_sequences_and_invalid_tail() -> None:
    materialized_block_table = torch.tensor(
        [
            [4, 7, -1],
            [9, -1, -1],
            [3, 2, 8],
        ],
        dtype=torch.int32,
    )

    layout = _build_stable_sfa_page_layout(materialized_block_table)

    assert layout.source_page_ids.tolist() == [4, 7, 9, 3, 2, 8]
    assert layout.target_page_ids.tolist() == [1, 0, 3, 7, 6, 8]
    assert layout.block_table.tolist() == [
        [1, 0, -1],
        [3, -1, -1],
        [7, 6, 8],
    ]
    assert layout.page_count == 9


def test_build_cp_context_materializes_mla_segment_lengths_on_device() -> None:
    q_cu_seqlens = [2, 5]
    segment_kv_seq_lens = [4, 9]
    op_result = (
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([True, True]),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        q_cu_seqlens,
        [4, 9],
        torch.tensor([0, 1], dtype=torch.int64),
        segment_kv_seq_lens,
        2,
    )
    device = torch.device("cpu")

    with patch.object(
        torch.ops.xllm_ops,
        "build_cp_context",
        return_value=op_result,
        create=True,
    ):
        context = build_cp_context([2, 3], [4, 5], 2, 0, device)

    assert context.q_cu_seqlens is q_cu_seqlens
    assert context.segment_kv_seq_lens is segment_kv_seq_lens
    assert context.q_cu_seqlens_tensor.dtype == torch.int32
    assert context.q_cu_seqlens_tensor.device == device
    assert context.q_cu_seqlens_tensor.tolist() == q_cu_seqlens
    assert context.segment_kv_seq_lens_tensor.dtype == torch.int32
    assert context.segment_kv_seq_lens_tensor.device == device
    assert context.segment_kv_seq_lens_tensor.tolist() == segment_kv_seq_lens


def test_materialization_rejects_incomplete_owner_distribution() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._is_mla = True
    backend.device = torch.device("cpu")
    backend._block_table_i32 = torch.tensor([[0]], dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        has_kv_shard=True,
        kv_split_size=2,
        kv_split_rank=0,
    )

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            return_value=4,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            return_value=torch.tensor([0, 0, 0, 1], dtype=torch.int64),
            create=True,
        ),
        pytest.raises(RuntimeError, match="KV owner distribution"),
    ):
        backend._prepare_kv_shard_materialization(metadata)


def _nope_cp_partition_context() -> SimpleNamespace:
    """A cp_size=2 plan over 6 global tokens where this rank owns global rows
    ``[0, 2, 3, 4, 5]`` and local row 2 is padding, spread over three
    ``(sequence, half)`` segments.

    ``query_index`` selects local rows and ``shard_index`` maps each of them to
    its global row, so ``query_index`` is not the identity and neither is
    ``shard_index``; in particular ``shard_index[query_index]`` (the real
    global rows) differs from ``query_index`` itself -- local row 1 is global
    row 2, while global row 1 belongs to the peer rank. A bug that drops the
    real-row selection, or that indexes the global query with the local
    ``query_index``, changes which rows are attended and is observable in the
    assertions below.
    """
    return SimpleNamespace(
        query_index=torch.tensor([0, 1, 3, 4, 5], dtype=torch.int64),
        shard_index=torch.tensor([0, 2, -1, 3, 4, 5], dtype=torch.int64),
        segment_seq_indices=torch.tensor([0, 0, 1], dtype=torch.int64),
        q_cu_seqlens_tensor=torch.tensor([2, 3, 5], dtype=torch.int32),
        segment_kv_seq_lens_tensor=torch.tensor([10, 12, 7], dtype=torch.int32),
        total_local=6,
    )


def _nope_cp_partition_backend(nope_cache: torch.Tensor) -> NpuPagedAttentionBackend:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._metadata = SimpleNamespace(
        has_kv_shard=False,
        kv_split_size=1,
        slot_mapping=torch.arange(6, dtype=torch.int64),
    )
    backend._block_table_i32 = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    backend._kv_caches = [SimpleNamespace(key=nope_cache, value=None)]
    backend._mla_actual_seq_q = torch.tensor([3, 6], dtype=torch.int32)
    backend._mla_actual_seq_kv = torch.tensor([12, 7], dtype=torch.int32)
    return backend


def test_mla_nope_cp_partitions_query_and_skips_rope_cache() -> None:
    """Partitioned NoPE CP path: the model's **global** latent/query/topk reach
    the backend unchanged, the sparse kernel runs on this rank's real global
    rows only, and the result is scattered back into the global layout the
    model's ``cp_shard_rows`` then reslices.
    """
    nope_cache = torch.zeros(4, 8, 2)
    backend = _nope_cp_partition_backend(nope_cache)
    cp_context = _nope_cp_partition_context()
    real_rows = cp_context.shard_index.index_select(0, cp_context.query_index)
    q_latent = torch.arange(12, dtype=torch.float32).view(6, 1, 2)
    k_latent = torch.ones(6, 1, 2)
    topk = torch.arange(6, dtype=torch.int32).view(6, 1)
    layer = SimpleNamespace(layer_id=0, qk_rope_head_dim=0)

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context),
        ),
        patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True) as reshape,
        patch.object(backend, "_mla_sparse", side_effect=lambda query, *_args: torch.ones_like(query)) as sparse,
    ):
        output = backend.execute_mla(q_latent, None, k_latent, None, layer, topk)

    # The global latent is written straight through the global slot mapping.
    reshape.assert_called_once_with(
        backend._metadata.slot_mapping,
        k_latent,
        k_latent,
        nope_cache,
        nope_cache,
    )
    sparse_args = sparse.call_args.args
    assert sparse_args[0].shape == (5, 1, 2)  # only this rank's real global rows
    torch.testing.assert_close(sparse_args[0], q_latent.index_select(0, real_rows))
    assert sparse_args[1] is None  # q_pe
    assert sparse_args[3] is None  # rope cache
    torch.testing.assert_close(
        sparse_args[5],
        torch.tensor([[0, 1], [0, 1], [2, 3]], dtype=torch.int32),
    )
    assert sparse_args[6] is cp_context.q_cu_seqlens_tensor
    assert sparse_args[7] is cp_context.segment_kv_seq_lens_tensor
    # Global-layout return: computed rows filled, the peer-owned row zero.
    assert output.shape == q_latent.shape
    torch.testing.assert_close(output.index_select(0, real_rows), torch.ones(5, 1, 2))
    torch.testing.assert_close(output[1], torch.zeros(1, 2))


def test_mla_nope_cp_empty_rank_skips_sparse_and_keeps_cache_write() -> None:
    nope_cache = torch.zeros(4, 8, 2)
    backend = _nope_cp_partition_backend(nope_cache)
    cp_context = _nope_cp_partition_context()
    cp_context.query_index = torch.zeros(0, dtype=torch.int64)
    q_latent = torch.zeros(6, 1, 2)
    k_latent = torch.ones(6, 1, 2)
    topk = torch.arange(6, dtype=torch.int32).view(6, 1)
    layer = SimpleNamespace(layer_id=0, qk_rope_head_dim=0)

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context),
        ),
        patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True) as reshape,
        patch.object(backend, "_mla_sparse", create=True) as sparse,
    ):
        output = backend.execute_mla(q_latent, None, k_latent, None, layer, topk)

    sparse.assert_not_called()
    reshape.assert_called_once_with(
        backend._metadata.slot_mapping,
        k_latent,
        k_latent,
        nope_cache,
        nope_cache,
    )
    assert output.shape == q_latent.shape
    assert torch.equal(output, torch.zeros_like(q_latent))


def test_mla_nope_cp_partition_helper_returns_global_zeros_for_empty_rank() -> None:
    """An all-padding rank owns no real global row, so the helper itself must
    skip the sparse kernel and return a zero tensor in the global layout the
    model reslices."""
    nope_cache = torch.zeros(4, 8, 2)
    backend = _nope_cp_partition_backend(nope_cache)
    cp_context = _nope_cp_partition_context()
    cp_context.query_index = torch.zeros(0, dtype=torch.int64)
    q_latent = torch.arange(12, dtype=torch.float32).view(6, 1, 2)

    with patch.object(backend, "_mla_sparse", create=True) as sparse:
        output = backend._mla_cp_partitioned_query(
            q_latent,
            torch.zeros(6, 1, 1, dtype=torch.int32),
            nope_cache,
            backend._block_table_i32,
            cp_context,
            0,
        )

    sparse.assert_not_called()
    assert output.shape == q_latent.shape
    assert torch.equal(output, torch.zeros_like(q_latent))


@pytest.mark.parametrize("has_kv_shard", [False, True])
def test_mla_cp_uses_one_paged_sequence_per_zigzag_segment_and_reuses_lengths(
    has_kv_shard: bool,
) -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._metadata = SimpleNamespace(
        has_kv_shard=has_kv_shard,
        kv_split_size=1,
        local_slot_mapping=torch.arange(6, dtype=torch.int64),
        slot_mapping=torch.arange(6, dtype=torch.int64),
    )
    backend._block_table_i32 = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    nope_cache = torch.zeros(4, 8, 2)
    rope_cache = torch.zeros(4, 8, 1)
    backend._kv_caches = [
        SimpleNamespace(key=nope_cache, value=rope_cache),
        SimpleNamespace(key=nope_cache, value=rope_cache),
    ]
    backend._mla_actual_seq_q = torch.tensor([3, 6], dtype=torch.int32)
    backend._mla_actual_seq_kv = torch.tensor([12, 7], dtype=torch.int32)

    q_cu_seqlens_tensor = torch.tensor([2, 3, 5], dtype=torch.int32)
    segment_kv_seq_lens_tensor = torch.tensor([10, 12, 7], dtype=torch.int32)
    cp_context = SimpleNamespace(
        query_index=torch.tensor([0, 1, 3, 4, 5], dtype=torch.int64),
        segment_seq_indices=torch.tensor([0, 0, 1], dtype=torch.int64),
        q_cu_seqlens=[2, 3, 5],
        q_cu_seqlens_tensor=q_cu_seqlens_tensor,
        segment_kv_seq_lens=[10, 12, 7],
        segment_kv_seq_lens_tensor=segment_kv_seq_lens_tensor,
    )
    q_latent = torch.zeros(6, 1, 2)
    q_pe = torch.zeros(6, 1, 1)
    k_latent = torch.zeros(6, 1, 2)
    k_pe = torch.zeros(6, 1, 1)
    topk = torch.arange(6, dtype=torch.int32).view(6, 1)

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context),
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.cp_gather_kv",
            side_effect=lambda tensor, _context: tensor,
        ),
        patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True),
        patch.object(
            backend,
            "_mla_sparse",
            side_effect=lambda query, *_args: torch.ones_like(query),
        ) as sparse,
    ):
        outputs = [
            backend.execute_mla(
                q_latent,
                q_pe,
                k_latent,
                k_pe,
                SimpleNamespace(layer_id=layer_id),
                topk,
            )
            for layer_id in range(2)
        ]

    assert sparse.call_count == 2
    sparse_args = sparse.call_args_list[0].args
    torch.testing.assert_close(
        sparse_args[5],
        torch.tensor([[0, 1], [0, 1], [2, 3]], dtype=torch.int32),
    )
    for call in sparse.call_args_list:
        assert call.args[6] is q_cu_seqlens_tensor
        assert call.args[7] is segment_kv_seq_lens_tensor
        assert call.args[6].data_ptr() == q_cu_seqlens_tensor.data_ptr()
        assert call.args[7].data_ptr() == segment_kv_seq_lens_tensor.data_ptr()
    for output in outputs:
        torch.testing.assert_close(output[cp_context.query_index], torch.ones(5, 1, 2))
        torch.testing.assert_close(output[2], torch.zeros(1, 2))


# ---------------------------------------------------------------------------
# KV-split write-mode and owner-shard materialization suite (from the
# cp-kv-split branch): the XLLM_CP_INDEX_WRITE_MODE switch, the
# owner-sharded latent-cache write with materialized logical reads, the
# page-grouping geometry, and the cross-mode drift guards.
# ---------------------------------------------------------------------------

_SHARD_CP_SIZE = 4
_SHARD_KV_SPLIT = 2
_SHARD_RANK = 0
# cp_size=4 with kv_split_size=2: owners 0 and 1 are each held by two CP ranks,
# so the owner -> CP-rank mapping is not the identity and picking the wrong
# replica is observable.
_SHARD_OWNER_BY_CP_RANK = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
# page_size=2 and kv_split_size=2 => a logical block covers 4 tokens. Rank 0
# owns the first half of every logical block; the peer half is not written.
_SHARD_GLOBAL_SLOTS = torch.arange(12, dtype=torch.int64)
_SHARD_LOCAL_SLOTS = torch.tensor([0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1], dtype=torch.int64)


@pytest.fixture(autouse=True)
def _pin_cp_index_write_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the CP index-write mode switch instead of inheriting the ambient
    environment, so an exported ``XLLM_CP_INDEX_WRITE_MODE`` cannot turn the
    m7 shard-battery tests here into spurious failures: those tests pin the
    sharded geometry this branch landed with. The replicated-mode tests
    override this with ``"replicated"`` through the same ``monkeypatch``
    instance, and the switch-semantics test deletes the variable to probe the
    shipped default.
    """
    monkeypatch.setenv("XLLM_CP_INDEX_WRITE_MODE", "sharded")


def _prepared_shard_case():
    """The shared owner-sharded scenario: 2 logical blocks, 6 entries, 2 owners."""
    block_table = torch.tensor([[3, 7, -1], [5, -1, -1]], dtype=torch.int32)
    metadata = _shard_metadata(_SHARD_GLOBAL_SLOTS, block_table)
    # One distinguishable value per local page so the materialized interleave is
    # checkable element by element.
    nope_cache = (torch.arange(8, dtype=torch.float32) * 10 + 1).view(8, 1, 1)
    with (
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            return_value=_SHARD_CP_SIZE,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            side_effect=_shard_all_gather,
            create=True,
        ),
    ):
        backend = _shard_prepared_backend(nope_cache, metadata)
        output, reshape, sparse, cp_context = _run_shard_execute(backend)
    return backend, nope_cache, metadata, output, reshape, sparse, cp_context


def _replicated_index_context(
    backend: NpuPagedAttentionBackend,
) -> object:
    """Build the index context under a live CP forward context."""
    with patch(
        "xllm.python.attention.npu_paged_attention.get_forward_context",
        return_value=SimpleNamespace(cp_context=_shard_cp_context()),
    ):
        return backend.mla_index_context(SimpleNamespace(layer_id=0))


def _replicated_shard_backend(
    monkeypatch: pytest.MonkeyPatch,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    slots: torch.Tensor,
) -> tuple[NpuPagedAttentionBackend, SimpleNamespace]:
    """A prepared shard backend with the write mode pinned to replicated.

    ``prepare`` still gathers the owner vector over the CP group (the latent
    caches stay owner-sharded in both modes), so the same CP stubs the sharded
    tests use are installed here.
    """
    monkeypatch.setenv("XLLM_CP_INDEX_WRITE_MODE", "replicated")
    metadata = _shard_metadata(slots, block_table)
    metadata.q_cu_seq_lens = torch.tensor([0, 8, 12], dtype=torch.int32)
    metadata.kv_seq_lens = _REPLICATED_KV_SEQ_LENS
    metadata.local_slot_mapping = KVShardLayout(
        physical_block_size=2,
        dcp_size=_SHARD_KV_SPLIT,
        dcp_rank=_SHARD_RANK,
    ).localize_slots(slots)
    with (
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            return_value=_SHARD_CP_SIZE,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            side_effect=_shard_all_gather,
            create=True,
        ),
    ):
        backend = _shard_prepared_backend(index_cache, metadata)
    backend._kv_caches = [
        SimpleNamespace(
            key=index_cache,
            value=None,
            index=index_cache,
            index_scale=None,
            kpool_tail=None,
        )
    ]
    backend._kpool_cache_triton_compatible = (False,)
    return backend, metadata


def _run_shard_execute(backend: NpuPagedAttentionBackend):
    cp_context = _shard_cp_context()
    with (
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context),
        ),
        patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True) as reshape,
        patch.object(
            backend,
            "_mla_sparse",
            side_effect=lambda query, *_args: torch.ones_like(query),
        ) as sparse,
    ):
        output = backend.execute_mla(
            torch.zeros(6, 1, 2),
            None,
            torch.ones(6, 1, 2),
            None,
            SimpleNamespace(layer_id=0, qk_rope_head_dim=0),
            torch.arange(6, dtype=torch.int32).view(6, 1),
        )
    return output, reshape, sparse, cp_context


def _shard_all_gather(tensor: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
    """CP all-gather stub: the owner vector, or one page set per CP rank.

    Each rank's pages are offset by ``rank * 1000`` so a wrong owner
    representative selects visibly different pages.
    """
    assert dim == 0
    assert world_size == _SHARD_CP_SIZE
    assert group_name == "cp"
    if tensor.shape == (1,):
        return _SHARD_OWNER_BY_CP_RANK
    return torch.cat([tensor + rank * 1000 for rank in range(_SHARD_CP_SIZE)], dim=0)


def _shard_cp_context() -> SimpleNamespace:
    """The shared 6-token query plan, widened to the 4-rank CP group the owner
    shard tests run on (kv_split_size=2 => two holders per owner)."""
    context = _nope_cp_partition_context()
    context.cp_size = _SHARD_CP_SIZE
    return context


def _shard_metadata(slot_mapping: torch.Tensor, block_table: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        has_kv_shard=True,
        kv_split_size=_SHARD_KV_SPLIT,
        kv_split_rank=_SHARD_RANK,
        slot_mapping=slot_mapping,
        local_slot_mapping=_SHARD_LOCAL_SLOTS,
        block_table=block_table,
        q_cu_seq_lens=torch.tensor([0, 6, 12], dtype=torch.int32),
        kv_seq_lens=torch.tensor([12, 12], dtype=torch.int32),
    )


def _shard_prepared_backend(nope_cache: torch.Tensor, metadata: SimpleNamespace) -> NpuPagedAttentionBackend:
    """A backend driven through the real ``prepare`` -> ``execute_mla`` flow.

    Only the state ``__init__``/``bind_kv_caches`` would have set is filled in by
    hand; the shard materialization itself is *not* stubbed, so
    ``_kv_owner_representatives``, ``_materialized_block_table`` and the page
    layout all come from the production preparation path.
    """
    backend = object.__new__(NpuPagedAttentionBackend)
    backend.device = torch.device("cpu")
    backend._is_mla = True
    backend._uses_sparse_mla = True
    backend._kv_caches = [SimpleNamespace(key=nope_cache, value=None)]
    # prepare() reads these unconditionally (XFIA decode stays off, no KDA
    # speculative checkpointing, and the logical-block fallback needs a page
    # size).
    backend._use_xfia_decode = False
    backend._linear_state_checkpoint_stride = None
    backend._page_size = 2
    backend._block_table_i32 = None
    backend._kv_owner_representatives = None
    backend._materialized_block_table = None
    backend._sfa_page_layout = None
    backend._mla_quant_indexer_metadata = {}
    backend._mla_actual_seq_q = None
    backend._mla_actual_seq_kv = None
    backend._mla_actual_seq_q_host = None
    backend._mla_actual_seq_kv_host = None
    backend._mla_max_seqlen_q = 0
    backend._mla_max_seqlen_k = 0
    backend._actual_seq_q = []
    backend._actual_seq_kv = []
    backend._actual_seq_lens = None
    backend.prepare(metadata)
    return backend


def test_cp_index_write_mode_ships_replicated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CP indexer-cache write mode defaults to replicated and only an
    explicit, recognized value changes it.

    Replicated is the default because the NPU index cache is full-size per
    rank anyway (the sharded mode saved zero memory), the indexer input is the
    CP-merged global stream on every rank so every rank can persist every
    stripe bit-identically, and the replicated layout removes the per-layer
    cross-rank index gather while unlocking the equal-width D(kv_split)
    transfer topology. Unset, empty, or unrecognized values keep that default
    rather than silently flipping it -- an empty string is what an unset shell
    variable expands to.
    """
    monkeypatch.delenv("XLLM_CP_INDEX_WRITE_MODE", raising=False)
    assert cp_index_write_mode() == "replicated"
    for value in ("replicated", "REPLICATED", " replicated "):
        monkeypatch.setenv("XLLM_CP_INDEX_WRITE_MODE", value)
        assert cp_index_write_mode() == "replicated", value
    for value in ("sharded", "SHARDED", " sharded "):
        monkeypatch.setenv("XLLM_CP_INDEX_WRITE_MODE", value)
        assert cp_index_write_mode() == "sharded", value
    for value in ("", "   ", "bogus", "owner-sharded", "1", "default"):
        monkeypatch.setenv("XLLM_CP_INDEX_WRITE_MODE", value)
        assert cp_index_write_mode() == "replicated", value
    # The strip set is exactly the ASCII whitespace, mirroring the C++
    # reader's absl::StripAsciiWhitespace: a non-ASCII blank (full-width
    # space U+3000, common from CJK input methods) is NOT stripped on either
    # side, so the value stays unrecognized and keeps the default. Python
    # once used str.strip(), whose full Unicode set would read "sharded"
    # here while the C++ registration and transfer defenses read the
    # replicated default -- an owner-sharded instance declaring replicated.
    for value in ("\u3000SHARDED\u3000", "\xa0sharded"):
        monkeypatch.setenv("XLLM_CP_INDEX_WRITE_MODE", value)
        assert cp_index_write_mode() == "replicated", value.encode("unicode_escape").decode()


def test_gather_index_history_materializes_owner_shards() -> None:
    """The uncompressed kPool history walk must read the reconstructed logical
    index cache: the local cache holds one owner page per logical block at the
    first page of the block's index-page group (row ``block_table[b] *
    kv_split_size``), so walking it with the logical table would read a
    fraction of every block."""
    block_table = torch.tensor([[3, 7]], dtype=torch.int32)
    metadata = _shard_metadata(_SHARD_GLOBAL_SLOTS, block_table)
    # Two logical blocks of page_size * kv_split_size = 4 tokens each.
    metadata.kv_seq_lens = torch.tensor([8], dtype=torch.int32)
    # Sixteen pages: block 3's owner page is row 6 and block 7's is row 14.
    index_cache = torch.arange(32, dtype=torch.float32).view(16, 2, 1, 1)

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context_or_none",
            return_value=SimpleNamespace(cp_context=_shard_cp_context()),
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            return_value=_SHARD_CP_SIZE,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            side_effect=_shard_all_gather,
            create=True,
        ),
    ):
        backend = _shard_prepared_backend(index_cache, metadata)
        backend._kv_caches = [SimpleNamespace(index=index_cache)]
        history = backend.gather_index_history(SimpleNamespace(layer_id=0), batch_size=1)

    # Logical token order across the reconstructed pages: logical block 3 (owner
    # 0 -> CP rank 0 page 6, owner 1 -> CP rank 2 page 6) then logical block 7
    # (pages 14). Reading the local pages with the logical table instead (rows
    # 3 and 7, the pre-fix latent-cache rows) returns [6, 7, 14, 15, 0, 0, 0,
    # 0], so this pins both the gather row and the materialization.
    torch.testing.assert_close(
        history.reshape(-1),
        torch.tensor([12.0, 13.0, 2012.0, 2013.0, 28.0, 29.0, 2028.0, 2029.0]),
    )


def test_linear_state_slots_are_replicated_not_owner_localized() -> None:
    """R5: ``BlockType::LINEAR`` (KDA conv/ssm state and the kPool tail) is a
    per-sequence resource, not a kv-split block type.

    ``Glm5NextKdaAttention.forward`` merges to global rows with ``cp_merge_rows``
    before ``execute_linear``, so every CP rank runs the conv1d/delta-rule over
    the same global stream and writes the same values into its own replica of
    the same per-sequence slot. The state ids therefore must reach the state I/O
    unlocalized: sharding them would need a cross-owner reduction per step and a
    defined owner for the accumulator, and no such mechanism exists. This pins
    that contract by checking the ids are not the ones localization would
    produce.
    """
    slots = torch.tensor([2, 4, 6], dtype=torch.int64)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_spec_verify=False,
        linear_state_indices=slots,
        linear_state_read_indices=slots,
        linear_state_write_indices=slots,
    )

    read_indices, write_indices = resolve_linear_state_io_indices(metadata)

    assert read_indices is slots
    assert write_indices is slots
    # What routing them through the KV shard layout would have produced.
    localized = KVShardLayout(physical_block_size=2, dcp_size=2, dcp_rank=0).localize_slots(slots)
    torch.testing.assert_close(localized, torch.tensor([-1, 2, -1], dtype=torch.int64))
    assert not torch.equal(read_indices, localized)

    # The kPool tail rides on the same per-sequence ids and is exposed to the
    # indexer verbatim.
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._metadata = SimpleNamespace(
        has_kv_shard=True,
        kv_split_size=2,
        is_prefill=True,
        is_chunked_prefill=False,
        is_spec_verify=False,
        slot_mapping=_SHARD_GLOBAL_SLOTS,
        local_slot_mapping=_SHARD_LOCAL_SLOTS,
        linear_state_indices=slots,
        linear_state_read_indices=slots,
        linear_state_write_indices=slots,
    )
    backend._block_table_i32 = torch.tensor([[3]], dtype=torch.int32)
    backend._page_size = 2
    backend._mla_actual_seq_q = torch.tensor([6], dtype=torch.int32)
    backend._mla_actual_seq_kv = torch.tensor([12], dtype=torch.int32)
    backend._kv_caches = [
        SimpleNamespace(index=torch.zeros(8, 2, 1, 1), index_scale=None, kpool_tail=torch.zeros(4, 2, 4, 1))
    ]
    backend._kpool_cache_triton_compatible = (False,)
    backend._materialized_block_table = torch.zeros(1, 2, dtype=torch.int32)

    with patch(
        "xllm.python.attention.npu_paged_attention.get_forward_context",
        return_value=SimpleNamespace(cp_context=None),
    ):
        context = backend.mla_index_context(SimpleNamespace(layer_id=0))

    # The paged index write is owner-local AND grouped: latent page p expands
    # to index page p * kv_split_size (the first page of the block's group),
    # never the global mapping and never the raw latent rows. The state slots
    # are neither localized nor regrouped.
    assert context.slot_mapping is not _SHARD_GLOBAL_SLOTS
    assert context.slot_mapping is not _SHARD_LOCAL_SLOTS
    torch.testing.assert_close(
        context.slot_mapping,
        torch.tensor([0, 1, -1, -1, 4, 5, -1, -1, 8, 9, -1, -1], dtype=torch.int64),
    )
    assert context.kpool_tail_read_indices is slots
    assert context.kpool_tail_write_indices is slots


# ---------------------------------------------------------------------------
# Replicated index-write mode (XLLM_CP_INDEX_WRITE_MODE=replicated, default)
# ---------------------------------------------------------------------------
# 2 sequences of 8 and 4 tokens (logical blocks of page_size * kv_split = 4
# tokens): sequence 0 spans logical blocks at entries 3 and 7, sequence 1 the
# one at entry 5. The global slot mapping below is exactly what the engine
# derives from those tables, and it is IDENTICAL to the slot mapping a
# kv_split_size == 1 launch derives from the page-granular table
# [[6, 7, 14, 15], [10, 11, -1, -1]] -- the kv1 counterpart used by the
# roundtrip test.
_REPLICATED_BLOCK_TABLE = torch.tensor([[3, 7], [5, -1]], dtype=torch.int32)
_REPLICATED_KV1_BLOCK_TABLE = torch.tensor([[6, 7, 14, 15], [10, 11, -1, -1]], dtype=torch.int32)
_REPLICATED_GLOBAL_SLOTS = torch.tensor([12, 13, 14, 15, 28, 29, 30, 31, 20, 21, 22, 23], dtype=torch.int64)
_REPLICATED_KV_SEQ_LENS = torch.tensor([8, 4], dtype=torch.int32)


def test_localize_pool_write_block_table_marks_peer_pages_invalid() -> None:
    """A pool page holds block_size/index_kpool pools, so one logical block spans
    kv_split_size physical pool pages grouped under the block's resource. The
    write view points every owned column at the FIRST page of that group --
    ``entry * kv_split_size``, the page the PD transfer plan's page-level
    overlap moves -- and invalidates the peer columns, which is what the
    existing ``physical_blocks >= 0`` guard in both pool writers consumes."""
    block_table = torch.tensor([[3, 7, -1]], dtype=torch.int32)

    localized = localize_pool_write_block_table(block_table, dcp_size=2, dcp_rank=0)

    torch.testing.assert_close(
        localized,
        torch.tensor([[6, -1, 14, -1, -1, -1]], dtype=torch.int32),
    )
    peer = localize_pool_write_block_table(block_table, dcp_size=2, dcp_rank=1)
    torch.testing.assert_close(
        peer,
        torch.tensor([[-1, 6, -1, 14, -1, -1]], dtype=torch.int32),
    )
    # kv_split_size == 1 keeps the caller's table object, so the full-replica
    # launch passes exactly what it passed before.
    assert localize_pool_write_block_table(block_table, dcp_size=1, dcp_rank=0) is block_table


def test_mla_index_context_expands_shard_slots_to_index_page_grouping() -> None:
    """A paged index write under an owner shard must address the index cache's
    page grouping, not the latent cache's rows: the latent slot mapping stores
    one page per logical block while the index cache groups ``kv_split_size``
    pages under each block, and the PD transfer plan moves page 0 of every
    group. The context's slot mapping therefore expands latent page ``p`` to
    index page ``p * kv_split_size``; without a shard it stays the caller's
    mapping object."""
    block_table = torch.tensor([[3, 7, -1], [5, -1, -1]], dtype=torch.int32)
    metadata = _shard_metadata(_SHARD_GLOBAL_SLOTS, block_table)
    index_cache = torch.full((16, 2, 1, 1), -7.0)

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            return_value=_SHARD_CP_SIZE,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            side_effect=_shard_all_gather,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context",
            return_value=SimpleNamespace(cp_context=_shard_cp_context()),
        ),
    ):
        backend = _shard_prepared_backend(index_cache, metadata)
        backend._kv_caches = [
            SimpleNamespace(
                key=index_cache,
                value=None,
                index=index_cache,
                index_scale=None,
                kpool_tail=None,
            )
        ]
        backend._kpool_cache_triton_compatible = (False,)
        context = backend.mla_index_context(SimpleNamespace(layer_id=0))

    # Latent pages 0, 1, 2 expand to index pages 0, 2, 4 (the first page of
    # each block's group); peer-owned slots stay invalid, so the scatter's
    # padding redirect still targets the shared padding page 0.
    torch.testing.assert_close(
        context.slot_mapping,
        torch.tensor([0, 1, -1, -1, 4, 5, -1, -1, 8, 9, -1, -1], dtype=torch.int64),
    )

    # The write closure consumes the expanded mapping: owned rows land on the
    # block's group-first pages while every peer page keeps its padding value.
    values = torch.arange(12, dtype=torch.float32).view(12, 1)
    context.update_index_cache(values, None)
    flat = index_cache.view(-1)
    written = values.view(-1)
    torch.testing.assert_close(flat[1], written[1])
    torch.testing.assert_close(flat[4:6], written[4:6])
    torch.testing.assert_close(flat[8:10], written[8:10])
    torch.testing.assert_close(flat[2:4], torch.full((2,), -7.0))
    torch.testing.assert_close(flat[6:8], torch.full((2,), -7.0))
    torch.testing.assert_close(flat[10:16], torch.full((6,), -7.0))


def test_mla_index_context_replicated_write_uses_global_slots_and_writes_every_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replicated mode: the paged index write consumes the GLOBAL slot mapping
    (the natural row of global logical slot s is s itself) and persists every
    stripe's page of every logical block, so no peer page keeps a stale row --
    the exact opposite of the sharded mode, where only the owner's group-first
    page is written and the peer pages stay stale."""
    index_cache = torch.full((16, 2, 1, 1), -7.0)
    backend, _ = _replicated_shard_backend(
        monkeypatch,
        index_cache,
        _REPLICATED_BLOCK_TABLE,
        _REPLICATED_GLOBAL_SLOTS,
    )
    context = _replicated_index_context(backend)

    # The write mapping is the global one, object for object: no owner
    # localization, no page-grouping remap.
    assert context.slot_mapping is _REPLICATED_GLOBAL_SLOTS
    # The pool-write table is the natural-row expansion: stripe j of logical
    # block entry e addresses page e * kv_split + j, never the group-first
    # page the sharded mode lands on.
    torch.testing.assert_close(
        context.localize_pool_block_table(_REPLICATED_BLOCK_TABLE),
        torch.tensor([[6, 7, 14, 15], [10, 11, -1, -1]], dtype=torch.int32),
    )

    # The write closure scatters every token row at its natural row: pages 6,
    # 7 (block 3), 14, 15 (block 7) and 10, 11 (block 5) are all written. In
    # sharded mode rank 0 would write only pages 6, 14 and 10, leaving 7, 15
    # and 11 stale at -7.
    values = torch.arange(12, dtype=torch.float32).view(12, 1)
    context.update_index_cache(values, None)
    flat = index_cache.view(-1)
    torch.testing.assert_close(flat[12:16], values[0:4].view(-1))
    torch.testing.assert_close(flat[28:32], values[4:8].view(-1))
    torch.testing.assert_close(flat[20:24], values[8:12].view(-1))
    # No stale peer rows anywhere in the referenced groups.
    assert not bool((flat[[14, 15, 30, 31, 22, 23]] == -7.0).any())


def test_mla_nope_cp_kv1_flow_is_bit_identical_to_the_legacy_write() -> None:
    """``kv_split_size == 1`` must be untouched: the same prepared flow writes the
    global slot mapping, gathers nothing, and hands the sparse kernel the same
    cache and page-table objects the legacy path used."""
    block_table = torch.tensor([[3, 7, -1], [5, -1, -1]], dtype=torch.int32)
    global_slots = _SHARD_GLOBAL_SLOTS
    metadata = _shard_metadata(global_slots, block_table)
    metadata.has_kv_shard = False
    metadata.kv_split_size = 1
    metadata.local_slot_mapping = None
    nope_cache = torch.arange(8, dtype=torch.float32).view(8, 1, 1)

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            create=True,
        ) as all_gather,
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            create=True,
        ) as cp_world_size,
    ):
        backend = _shard_prepared_backend(nope_cache, metadata)
        output, reshape, sparse, _ = _run_shard_execute(backend)

    # No owner bookkeeping and no gather: the identity materialization path.
    all_gather.assert_not_called()
    cp_world_size.assert_not_called()
    assert backend._kv_owner_representatives is None
    assert backend._materialized_block_table is None
    assert backend._sfa_page_layout is None

    reshape.assert_called_once()
    assert reshape.call_args.args[0] is global_slots
    sparse_args = sparse.call_args.args
    assert sparse_args[2] is nope_cache
    # The same logical page table the legacy path sliced, column for column,
    # now cut to the partition plan's (sequence, half) segments.
    torch.testing.assert_close(
        sparse_args[5],
        backend._block_table_i32.index_select(0, torch.tensor([0, 0, 1])),
    )
    torch.testing.assert_close(
        sparse_args[5],
        torch.tensor([[3, 7, -1], [3, 7, -1], [5, -1, -1]], dtype=torch.int32),
    )
    assert output.shape == (6, 1, 2)


def test_mla_nope_cp_owner_shard_rejects_unprepared_materialization() -> None:
    """The sharded path must fail loudly when preparation never ran: silently
    attending the local pages would read one owner stripe as if it were the
    whole sequence."""
    nope_cache = torch.zeros(8, 1, 1)
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._metadata = _shard_metadata(_SHARD_GLOBAL_SLOTS, torch.tensor([[3]], dtype=torch.int32))
    backend._block_table_i32 = torch.tensor([[3]], dtype=torch.int32)
    backend._kv_caches = [SimpleNamespace(key=nope_cache, value=None)]
    backend._mla_actual_seq_q = torch.tensor([6], dtype=torch.int32)
    backend._mla_actual_seq_kv = torch.tensor([12], dtype=torch.int32)
    backend._kv_owner_representatives = None
    backend._materialized_block_table = None

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context",
            return_value=SimpleNamespace(cp_context=_shard_cp_context()),
        ),
        patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True),
        pytest.raises(RuntimeError, match="KV shard materialization was not prepared"),
    ):
        backend.execute_mla(
            torch.zeros(6, 1, 2),
            None,
            torch.ones(6, 1, 2),
            None,
            SimpleNamespace(layer_id=0, qk_rope_head_dim=0),
            torch.arange(6, dtype=torch.int32).view(6, 1),
        )


def test_mla_nope_cp_owner_shard_writes_local_and_reads_materialized_logical_cache() -> None:
    """Owner-sharded KV on the partitioned NoPE CP path.

    The real ``prepare`` -> ``execute_mla`` flow must (a) write the latent cache
    through the owner-local slot mapping -- the global one would address rows
    this rank does not own -- and (b) attend the logical cache reconstructed
    from every owner's stripe, so the sparse kernel sees exactly the stream the
    ``kv_split_size == 1`` shape produces.
    """
    backend, nope_cache, metadata, output, reshape, sparse, cp_context = _prepared_shard_case()

    # The owner mapping itself: owners 0 and 1 are held by CP ranks (0, 1) and
    # (2, 3), so the representatives are the first holder of each owner.
    assert torch.equal(backend._kv_owner_representatives, torch.tensor([0, 2]))

    # (a) the write goes through the owner-local mapping, slot for slot.
    reshape.assert_called_once()
    written = reshape.call_args.args[0]
    assert written is metadata.local_slot_mapping
    assert not torch.equal(written, metadata.slot_mapping)
    torch.testing.assert_close(written, _SHARD_LOCAL_SLOTS)
    # Rank 0 owns the first two tokens of every logical block and nothing else.
    assert int((written >= 0).sum().item()) == 6

    # (b) the sparse kernel attends the interleaved logical pages: page
    # entry*2 + owner, owner 0 taken from CP rank 0 and owner 1 from CP rank 2.
    sparse_args = sparse.call_args.args
    materialized = sparse_args[2]
    assert materialized.shape == (12, 1, 1)
    torch.testing.assert_close(
        materialized.reshape(-1),
        torch.tensor(
            [
                31.0,
                2031.0,  # logical block 3: rank0 page 3, rank2 page 3
                71.0,
                2071.0,  # logical block 7
                1.0,
                2001.0,  # padding column (block id -1 -> page 0)
                51.0,
                2051.0,  # logical block 5 (second request)
                1.0,
                2001.0,
                1.0,
                2001.0,
            ]
        ),
    )
    # ... and its matching page table, one column per logical block per owner,
    # sliced to the partition plan's (sequence, half) segments.
    assert backend._materialized_block_table is not None
    torch.testing.assert_close(
        sparse_args[5],
        backend._materialized_block_table.index_select(0, torch.tensor([0, 0, 1])),
    )
    torch.testing.assert_close(
        sparse_args[5],
        torch.tensor(
            [
                [0, 1, 2, 3, -1, -1],
                [0, 1, 2, 3, -1, -1],
                [6, 7, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
        ),
    )
    # The partitioned segment lengths still drive SFA, unchanged by the shard.
    assert sparse_args[6] is cp_context.q_cu_seqlens_tensor
    assert sparse_args[7] is cp_context.segment_kv_seq_lens_tensor
    assert output.shape == (6, 1, 2)


def test_replicated_index_materialization_is_identity_and_skips_the_cp_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replicated mode: the INDEX materialization must be the identity -- the
    full-replica layout is already physically local, so the per-DSA-layer
    cross-rank all_gather the sharded mode pays is pure overhead and must not
    run. The latent caches (pages_per_block == 1) keep gathering in both
    modes."""
    index_cache = torch.arange(32, dtype=torch.float32).view(16, 2, 1, 1)
    backend, metadata = _replicated_shard_backend(
        monkeypatch,
        index_cache,
        _REPLICATED_BLOCK_TABLE,
        _REPLICATED_GLOBAL_SLOTS,
    )
    assert backend._replicated_index_block_table is not None
    torch.testing.assert_close(
        backend._replicated_index_block_table,
        torch.tensor([[6, 7, 14, 15], [10, 11, -1, -1]], dtype=torch.int32),
    )

    # The INDEX materialize returns the local cache and the physically
    # expanded table without a single collective.
    with patch(
        "xllm.python.attention.npu_paged_attention.distributed.all_gather",
        create=True,
    ) as gather_spy:
        materialized, scale, table = backend._materialize_mla_index_cache(
            index_cache,
            None,
            metadata,
            _shard_cp_context(),
        )
    gather_spy.assert_not_called()
    assert materialized is index_cache
    assert scale is None
    assert table is backend._replicated_index_block_table

    # The context's read table is the same physical expansion, and the
    # context advertises the shard views (the read still needs the expanded
    # table even though the cache is a full replica).
    context = _replicated_index_context(backend)
    assert context.has_kv_shard is True
    assert context.materialized_block_table is backend._replicated_index_block_table
    with (
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context",
            return_value=SimpleNamespace(cp_context=_shard_cp_context()),
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context_or_none",
            return_value=SimpleNamespace(cp_context=_shard_cp_context()),
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            create=True,
        ) as gather_spy,
    ):
        materialized_context, _, context_table = context.materialize_index_cache()
        history = backend.gather_index_history(SimpleNamespace(layer_id=0), batch_size=2)
    gather_spy.assert_not_called()
    assert materialized_context is index_cache
    assert context_table is backend._replicated_index_block_table
    # The history walk reads the local pages through the expanded table:
    # sequence 0 walks pages 6, 7, 14, 15 and sequence 1 pages 10, 11 (padded
    # to the batch max KV length with zeros, as on the kv1 path).
    torch.testing.assert_close(
        history.reshape(-1),
        torch.tensor([12.0, 13.0, 14.0, 15.0, 28.0, 29.0, 30.0, 31.0, 20.0, 21.0, 22.0, 23.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_replicated_index_roundtrip_matches_kv1_full_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write-read roundtrip: with the SAME global slot mapping and equivalent
    block tables, a replicated kv_split_size == 2 launch writes and reads the
    index cache exactly like the kv_split_size == 1 full-replica launch --
    identical cache bytes and identical gathered history. This is the
    correctness contract that makes the replicated layout a drop-in for the
    full replica (and what lets any single PD transfer writer supply every
    valid page)."""
    values = torch.arange(12, dtype=torch.float32).view(12, 1) * 10 + 1

    replicated_cache = torch.full((16, 2, 1, 1), -7.0)
    backend, _ = _replicated_shard_backend(
        monkeypatch,
        replicated_cache,
        _REPLICATED_BLOCK_TABLE,
        _REPLICATED_GLOBAL_SLOTS,
    )
    context = _replicated_index_context(backend)
    assert context.slot_mapping is _REPLICATED_GLOBAL_SLOTS
    context.update_index_cache(values, None)
    with patch(
        "xllm.python.attention.npu_paged_attention.get_forward_context_or_none",
        return_value=SimpleNamespace(cp_context=_shard_cp_context()),
    ):
        replicated_history = backend.gather_index_history(SimpleNamespace(layer_id=0), batch_size=2)

    # The kv1 full-replica counterpart: page-granular block table, no shard
    # metadata, the same global slot mapping.
    kv1_cache = torch.full((16, 2, 1, 1), -7.0)
    kv1_metadata = _shard_metadata(_REPLICATED_GLOBAL_SLOTS, _REPLICATED_KV1_BLOCK_TABLE)
    kv1_metadata.has_kv_shard = False
    kv1_metadata.kv_split_size = 1
    kv1_metadata.local_slot_mapping = None
    kv1_metadata.q_cu_seq_lens = torch.tensor([0, 8, 12], dtype=torch.int32)
    kv1_metadata.kv_seq_lens = _REPLICATED_KV_SEQ_LENS
    kv1_backend = _shard_prepared_backend(kv1_cache, kv1_metadata)
    kv1_backend._kv_caches = [
        SimpleNamespace(
            key=kv1_cache,
            value=None,
            index=kv1_cache,
            index_scale=None,
            kpool_tail=None,
        )
    ]
    kv1_backend._kpool_cache_triton_compatible = (False,)
    with patch(
        "xllm.python.attention.npu_paged_attention.get_forward_context",
        return_value=SimpleNamespace(cp_context=None),
    ):
        kv1_context = kv1_backend.mla_index_context(SimpleNamespace(layer_id=0))
    assert kv1_context.slot_mapping is _REPLICATED_GLOBAL_SLOTS
    kv1_context.update_index_cache(values, None)
    kv1_history = kv1_backend.gather_index_history(SimpleNamespace(layer_id=0), batch_size=2)

    torch.testing.assert_close(replicated_cache, kv1_cache)
    torch.testing.assert_close(replicated_history, kv1_history)


def test_replicated_read_rejects_shard_prepared_under_sharded_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mode drift between ``prepare`` and the index context must fail loudly.

    If the shard materialization ran while the switch said ``sharded`` (no
    replicated table built) but the context is requested under ``replicated``,
    reading the owner-sharded cache through the full-replica view would
    silently drop every peer stripe; the context refuses instead."""
    index_cache = torch.full((16, 2, 1, 1), -7.0)
    monkeypatch.setenv("XLLM_CP_INDEX_WRITE_MODE", "sharded")
    # Prepare under the sharded mode: the same backend shape the shard tests
    # use, with only the owner bookkeeping and the gathered table built.
    metadata = _shard_metadata(_REPLICATED_GLOBAL_SLOTS, _REPLICATED_BLOCK_TABLE)
    metadata.q_cu_seq_lens = torch.tensor([0, 8, 12], dtype=torch.int32)
    metadata.kv_seq_lens = _REPLICATED_KV_SEQ_LENS
    with (
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            return_value=_SHARD_CP_SIZE,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            side_effect=_shard_all_gather,
            create=True,
        ),
    ):
        backend = _shard_prepared_backend(index_cache, metadata)
    backend._kv_caches = [
        SimpleNamespace(
            key=index_cache,
            value=None,
            index=index_cache,
            index_scale=None,
            kpool_tail=None,
        )
    ]
    backend._kpool_cache_triton_compatible = (False,)
    assert backend._materialized_block_table is not None
    assert backend._replicated_index_block_table is None

    monkeypatch.setenv("XLLM_CP_INDEX_WRITE_MODE", "replicated")
    with (
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context",
            return_value=SimpleNamespace(cp_context=_shard_cp_context()),
        ),
        pytest.raises(RuntimeError, match="replicated index views were not prepared"),
    ):
        backend.mla_index_context(SimpleNamespace(layer_id=0))


def test_sharded_write_mode_warns_once_about_equal_width_decode_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sharded writes leave every peer index page stale, so the equal-width
    D(kv_split SfaDcp) link -- which plans 1:1 and transfers EVERY page of
    each INDEX resource -- would hand the decode stale rows with no error
    anywhere. The link admission cannot see this python-side switch (the
    instance metadata carries no write mode), so the loudest available guard
    is the shard preparation's once-per-process warning; pin that it fires
    exactly once no matter how many prefill batches prepare."""
    monkeypatch.setenv("XLLM_CP_INDEX_WRITE_MODE", "sharded")
    npu_paged_attention_module._sharded_write_transfer_warning_emitted = False
    index_cache = torch.zeros(8, 1, 1)
    metadata = _shard_metadata(_SHARD_GLOBAL_SLOTS, torch.tensor([[3, 7, -1], [5, -1, -1]], dtype=torch.int32))
    with (
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            return_value=_SHARD_CP_SIZE,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            side_effect=_shard_all_gather,
            create=True,
        ),
        patch.object(npu_paged_attention_module, "logger") as logger_spy,
    ):
        _shard_prepared_backend(index_cache, metadata)
        _shard_prepared_backend(index_cache, metadata)
    logger_spy.warning.assert_called_once()
    assert npu_paged_attention_module._sharded_write_transfer_warning_emitted is True
