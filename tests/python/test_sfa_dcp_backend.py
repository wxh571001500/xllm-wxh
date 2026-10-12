# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""CPU tests for SFA DCP graph-prepare indexer paging."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("torch_npu", reason="SFA DCP backend tests import the NPU attention backend")

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.kv_shard_layout import replicate_pool_write_block_table
from xllm.python.attention.npu_paged_attention import write_mla_paged_cache
from xllm.python.attention.sfa_dcp_backend import SfaDcpAttentionBackend
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.models.glm5_next import _kpool_logical_rows, _kpool_logical_state_indices


class _FakeDcpGroup:
    def size(self) -> int:
        return 4

    def rank(self) -> int:
        return 0


def _cpu_context(execution_state: AclGraphExecutionState) -> ForwardContext:
    return ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[],
        execution_state=execution_state,
    )


def test_graph_prepare_keeps_valid_indexer_pages_for_padded_lanes() -> None:
    backend = SfaDcpAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        scale=0.1,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dcp_group=_FakeDcpGroup(),
        index_topk=2048,
        max_num_reqs=8,
    )
    page_size = 128
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.empty(16, page_size, 1, 512),
                value=torch.empty(16, page_size, 1, 64),
                index=torch.empty(64, page_size, 1, 128),
            )
        ]
    )

    block_table = torch.zeros((8, 2), dtype=torch.int32)
    block_table[:7] = torch.tensor([[1, 2]] * 7, dtype=torch.int32)
    slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5, 6, -1], dtype=torch.int32)
    kv_seq_lens = torch.tensor([1022, 1022, 1022, 1022, 1022, 1022, 1022, 1], dtype=torch.int32)
    metadata = SimpleNamespace(
        slot_mapping=slot_mapping,
        block_table=block_table,
        kv_seq_lens=kv_seq_lens,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
        q_cu_seq_lens=None,
        q_seq_lens=None,
        expanded_decode_metadata=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    with forward_context(_cpu_context(AclGraphExecutionState({}))):
        backend.prepare(metadata, graph_mode=True)

    expanded = backend._expanded_indexer_block_table
    assert expanded is not None
    assert (expanded[-1] >= 0).all()
    assert torch.equal(expanded[-1], torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32))
    assert torch.equal(expanded[0, :4], torch.tensor([4, 5, 6, 7], dtype=torch.int32))


def test_bind_kv_caches_accepts_missing_nope_rope_cache() -> None:
    backend = SfaDcpAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        scale=0.1,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dcp_group=_FakeDcpGroup(),
        index_topk=2048,
        max_num_reqs=8,
    )
    page_size = 128
    nope_cache = torch.empty(16, page_size, 1, 512)
    backend.bind_kv_caches(
        [
            LayerCache(
                key=nope_cache,
                value=None,
                index=torch.empty(64, page_size, 1, 128),
            )
        ]
    )
    bound = backend._kv_caches[0]
    assert bound.key is nope_cache
    assert bound.value is None
    assert backend.is_mla


def test_write_mla_paged_cache_nope_reuses_latent_cache() -> None:
    k_latent = torch.randn(2, 1, 16)
    nope_cache = torch.randn(4, 8, 1, 16)
    slots = torch.zeros(2, dtype=torch.int32)
    with patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True) as reshape:
        write_mla_paged_cache(slots, k_latent, None, nope_cache, None)
    assert reshape.call_args.args == (slots, k_latent, k_latent, nope_cache, nope_cache)


def test_write_mla_paged_cache_nope_ignores_zero_width_rope_cache() -> None:
    k_latent = torch.randn(2, 1, 16)
    nope_cache = torch.randn(4, 8, 1, 16)
    rope_cache = torch.empty(4, 8, 1, 0)
    slots = torch.zeros(2, dtype=torch.int32)
    with patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True) as reshape:
        write_mla_paged_cache(slots, k_latent, None, nope_cache, rope_cache)
    assert reshape.call_args.args[4] is nope_cache


def test_write_mla_paged_cache_keeps_glm52_rope() -> None:
    k_latent = torch.randn(2, 1, 16)
    k_pe = torch.randn(2, 1, 4)
    nope_cache = torch.randn(4, 8, 1, 16)
    rope_cache = torch.randn(4, 8, 1, 4)
    slots = torch.zeros(2, dtype=torch.int32)
    with patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True) as reshape:
        write_mla_paged_cache(slots, k_latent, k_pe, nope_cache, rope_cache)
    assert reshape.call_args.args == (slots, k_latent, k_pe, nope_cache, rope_cache)


def test_execute_mla_nope_accepts_none_q_pe_and_k_pe() -> None:
    backend = SfaDcpAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        scale=0.1,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dcp_group=_FakeDcpGroup(),
        index_topk=2048,
        max_num_reqs=8,
    )
    page_size = 128
    nope_cache = torch.empty(16, page_size, 1, 512)
    rope_cache = torch.empty(16, page_size, 1, 0)
    layer_cache = LayerCache(
        key=nope_cache,
        value=rope_cache,
        index=torch.empty(64, page_size, 1, 128),
    )
    backend.bind_kv_caches([layer_cache])

    block_table = torch.zeros((1, 2), dtype=torch.int32)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0], dtype=torch.int32),
        block_table=block_table,
        kv_seq_lens=torch.tensor([8], dtype=torch.int32),
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
        q_cu_seq_lens=None,
        q_seq_lens=None,
        expanded_decode_metadata=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    q_latent = torch.randn(1, 8, 512)
    k_latent = torch.randn(1, 1, 512)
    topk = torch.zeros(1, 1, 2048, dtype=torch.int32)
    layer = SimpleNamespace(layer_id=0)
    attn_out = torch.randn(1, 8, 512)
    context = ForwardContext(
        attention_backend=backend,
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[layer_cache],
        execution_state=None,
    )
    assert backend._impl is not None
    with (
        forward_context(context),
        patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True) as reshape_paged_cache,
        patch.object(backend._impl, "_store_parallel_kv", return_value=(None, k_latent, None)),
        patch.object(backend._impl, "_record_query_gather_context") as record_query,
        patch.object(
            backend._impl,
            "_execute_sparse_flash_attention_process",
            return_value=attn_out,
        ) as execute,
    ):
        backend.prepare(metadata, graph_mode=False)
        out = backend.execute_mla(q_latent, None, k_latent, None, layer, topk=topk)
    assert out is attn_out
    reshape_args = reshape_paged_cache.call_args.args
    assert reshape_args[1] is k_latent
    assert reshape_args[2] is k_latent
    assert reshape_args[3] is nope_cache
    assert reshape_args[4] is nope_cache
    record_query.assert_called_once()
    assert record_query.call_args.args[1] is None
    execute.assert_called_once()
    assert execute.call_args.args[1] is None
    assert execute.call_args.args[2][1] is rope_cache


def test_gather_index_history_dcp_covers_tokens_in_one_logical_page() -> None:
    """Prefill warmup can sit in one logical DCP page.

    With dcp=4 and page_size=128 a logical page covers 512 tokens, so a 256-token
    sequence is one engine block-table column. The indexer cache is still paged
    at page_size=128; gather must walk the expanded table (4 physical pages) or
    it writes 128 rows into a 256-row target.
    """
    backend = SfaDcpAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        scale=0.1,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dcp_group=_FakeDcpGroup(),
        index_topk=2048,
        max_num_reqs=8,
    )
    page_size = 128
    width = 257
    n_phys = 16
    index = torch.zeros(n_phys, page_size, 1, width)
    index[:, :, 0, -1] = torch.arange(n_phys, dtype=index.dtype).unsqueeze(1)
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.empty(n_phys, page_size, 1, 512),
                value=torch.empty(n_phys, page_size, 1, 0),
                index=index,
            )
        ]
    )

    kv_len = 256
    logical_block_table = torch.tensor([[0]], dtype=torch.int32)
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(kv_len, dtype=torch.int32),
        block_table=logical_block_table,
        kv_seq_lens=torch.tensor([kv_len], dtype=torch.int32),
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
        q_cu_seq_lens=None,
        q_seq_lens=None,
        expanded_decode_metadata=None,
        is_prefill=True,
        is_chunked_prefill=False,
        has_kv_shard=False,
        kv_split_size=4,
    )
    backend.prepare(metadata, graph_mode=False)

    expanded = backend.indexer_block_table()
    assert torch.equal(expanded[0], torch.tensor([0, 1, 2, 3], dtype=torch.int32))

    packed = backend.gather_index_history(SimpleNamespace(layer_id=0), batch_size=kv_len)
    assert packed.shape == (1, kv_len, width)
    assert torch.equal(packed[0, :page_size, -1], torch.zeros(page_size))
    assert torch.equal(packed[0, page_size:kv_len, -1], torch.ones(page_size))


# ---------------------------------------------------------------------------
# M11.3 Route B: decode-step invariant pins. The C++ KVShardBatchMetadata
# builder attaches shard metadata to prefill-ish MLA batches under an active
# CP group only (py_executor_impl.cpp), so a cp_size == 1 + kv_split > 1
# decode batch reaches this backend with has_kv_shard=False and
# kv_split_size=1 -- the values the tests below pin into the metadata stubs.
# Correctness on decode is SfaDcp's SELF-CONTAINMENT plus these invariants,
# not C++-metadata symmetry.
# ---------------------------------------------------------------------------


class _FakeDcpGroupOf:
    """DCP process-group stub with a configurable rank (size fixed at 4)."""

    def __init__(self, rank: int) -> None:
        self._rank = rank

    def size(self) -> int:
        return 4

    def rank(self) -> int:
        return self._rank


# Shared decode scenario: page_size=128, dcp_size=4, logical block 512. The
# four requests write their next token inside logical block 2 (global slots
# [1024, 1536)), one per DCP stripe, so rank 0 owns exactly the first.
_DECODE_BLOCK_TABLE = torch.tensor([[2, 9], [2, -1], [2, -1], [2, 4]], dtype=torch.int32)
_DECODE_KV_SEQ_LENS = torch.tensor([1035, 1155, 1285, 1415], dtype=torch.int32)
_DECODE_SLOT_MAPPING = torch.tensor([1034, 1154, 1284, 1414], dtype=torch.int32)
# Slot 1034: block 2, offset 10, stripe 0 -> rank 0 local page 2, offset 10.
# Slots 1154/1284/1414 sit on stripes 1/2/3 -> INVALID on rank 0.
_DECODE_LOCAL_SLOTS = torch.tensor([266, -1, -1, -1], dtype=torch.int32)
# 1035 = 2 * 512 + 11 -> 2 local pages + 11 owned remainder tokens; the
# others' remainders exceed one full physical page (128 owned tokens).
_DECODE_LOCAL_SEQ_LENS = torch.tensor([267, 384, 384, 384], dtype=torch.int32)
_DECODE_EXPANDED_INDEXER_TABLE = torch.tensor(
    [
        [8, 9, 10, 11, 36, 37, 38, 39],
        [8, 9, 10, 11, -1, -1, -1, -1],
        [8, 9, 10, 11, -1, -1, -1, -1],
        [8, 9, 10, 11, 16, 17, 18, 19],
    ],
    dtype=torch.int32,
)


def _decode_prepared_backend(rank: int = 0) -> SfaDcpAttentionBackend:
    """A SfaDcp backend bound to caches and prepared on a plain decode batch.

    The metadata stub carries the production decode values the C++ view
    exposes for an MLA instance (``has_kv_shard=False``, ``kv_split_size=1``):
    PyExecutorImpl::run builds KVShardBatchMetadata for prefill-ish MLA
    batches under an active CP group and for dense non-MLA DCP steps only, so
    these are the fields a cp1 + kv_split > 1 decode step really sees.
    """
    backend = SfaDcpAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        scale=0.1,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dcp_group=_FakeDcpGroupOf(rank),
        index_topk=2048,
        max_num_reqs=8,
    )
    page_size = 128
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.empty(64, page_size, 1, 512),
                value=torch.empty(64, page_size, 1, 0),
                index=torch.empty(256, page_size, 1, 128),
            )
        ]
    )
    metadata = SimpleNamespace(
        slot_mapping=_DECODE_SLOT_MAPPING,
        block_table=_DECODE_BLOCK_TABLE,
        kv_seq_lens=_DECODE_KV_SEQ_LENS,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
        q_cu_seq_lens=None,
        q_seq_lens=None,
        expanded_decode_metadata=None,
        is_prefill=False,
        is_chunked_prefill=False,
        has_kv_shard=False,
        kv_split_size=1,
        kv_split_rank=0,
    )
    backend.prepare(metadata, graph_mode=False)
    return backend


def test_decode_prepare_is_self_contained_slots_table_and_seq_lens() -> None:
    """M11.3 Route B core pin: SfaDcp derives its whole decode metadata from
    the global batch fields -- localized latent slots, the page-expanded
    indexer table and the per-rank sequence lengths -- and never reads the C++
    shard fields, which stay inert (has_kv_shard=False / kv_split_size=1) on
    decode. Content and shape are asserted, not just object identity."""
    backend = _decode_prepared_backend(rank=0)

    # Latent write mapping: rank 0 owns stripe 0 of every logical block; the
    # peer stripes stay INVALID so the paged write skips their rows.
    assert backend._local_slot_mapping is not None
    assert tuple(backend._local_slot_mapping.shape) == (4,)
    torch.testing.assert_close(backend._local_slot_mapping, _DECODE_LOCAL_SLOTS)

    # Indexer table: every logical-block column expands to its dcp_size
    # physical pages at their NATURAL rows (entry e -> pages [4e, 4e+4)), and
    # an invalid column stays invalid page by page.
    assert backend._expanded_indexer_block_table is not None
    assert tuple(backend._expanded_indexer_block_table.shape) == (4, 8)
    torch.testing.assert_close(
        backend._expanded_indexer_block_table,
        _DECODE_EXPANDED_INDEXER_TABLE,
    )

    # Local sequence lengths from the KVShardLayout: full logical blocks
    # contribute one physical page each, the remainder only the tokens this
    # rank owns. The builder hands them to the kernel as dcp_context fields.
    assert backend._sfa_metadata is not None
    torch.testing.assert_close(
        backend._sfa_metadata.dcp_context.seq_lens,
        _DECODE_LOCAL_SEQ_LENS,
    )
    assert backend._sfa_metadata.dcp_context.slot_mapping.data_ptr() == backend._local_slot_mapping.data_ptr()
    assert torch.equal(
        backend._sfa_metadata.dcp_context.block_table,
        backend._block_table_i32[:4],
    )
    assert backend._sfa_metadata.num_prefills == 0


@pytest.mark.parametrize("write_mode", ["replicated", "sharded"])
def test_decode_index_context_stays_global_and_expanded_in_both_write_modes(
    monkeypatch: pytest.MonkeyPatch,
    write_mode: str,
) -> None:
    """M11.3 Route B pin: on a decode batch the paged INDEX write consumes the
    GLOBAL slot mapping and the pool write table is the expanded natural-row
    table, regardless of XLLM_CP_INDEX_WRITE_MODE.

    The mode switch only reroutes batches that carry ``has_kv_shard=True``
    with ``kv_split_size > 1`` (the prefill PCP shape); a decode batch has
    neither, so both modes must agree: the natural row of global logical slot
    s is s itself, and the pool write table expands every logical block to
    its dcp_size physical pages at their natural rows -- rank-independent
    geometry, so every rank persists identical pool bytes at identical rows.
    """
    monkeypatch.setenv("XLLM_CP_INDEX_WRITE_MODE", write_mode)
    backend = _decode_prepared_backend(rank=0)

    with forward_context(_cpu_context(None)):
        context = backend.mla_index_context(SimpleNamespace(layer_id=0))

    # The paged index write uses the GLOBAL mapping, object for object: no
    # owner localization and no page-grouping remap in either mode.
    assert context.slot_mapping is _DECODE_SLOT_MAPPING
    # The decode context stays shard-free: no materialized views, no gather.
    assert context.has_kv_shard is False
    assert context.materialized_block_table is None
    # SfaDcp's override swaps in the page-expanded table and attaches it to
    # the index materialization, so a kPool read/write addresses the natural
    # physical pages directly.
    assert context.block_table is backend._expanded_indexer_block_table
    with forward_context(_cpu_context(None)):
        cache, scale, table = context.materialize_index_cache()
    assert cache is backend._kv_caches[0].index
    assert scale is None
    assert table is backend._expanded_indexer_block_table

    # The pool write table is the identity over the expanded table on decode
    # (the metadata view reports kv_split_size=1), and that table is exactly
    # the natural-row replicated expansion of the engine's logical table --
    # the same geometry replicate_pool_write_block_table produces, with no
    # dcp_rank term anywhere.
    assert context.localize_pool_block_table(_DECODE_EXPANDED_INDEXER_TABLE) is _DECODE_EXPANDED_INDEXER_TABLE
    torch.testing.assert_close(
        _DECODE_EXPANDED_INDEXER_TABLE,
        replicate_pool_write_block_table(_DECODE_BLOCK_TABLE, 4),
    )


def test_decode_compressed_tail_collapses_expanded_table_to_identical_pools() -> None:
    """M11.3 Route B combination pin: compressed-tail kPool update on a
    spec-verify decode step (token-expanded rows) collapses the expanded
    table to one row per request and derives a pool write table that is
    IDENTICAL on every rank -- the invariant the write correctness silently
    depends on (glm5_next.py's kPool update path).

    The latent slots are owner-local (rank 1 owns a different stripe), but
    the pool path has no rank term: the collapse uses only the query lens and
    the table, and the expansion is natural-row arithmetic. Decode rows are
    TP-replicated, so identical write geometry means identical pool bytes.
    """
    # Two requests x MTP depth 1 -> two token rows each. Both requests run
    # inside logical block 1 (global slots [512, 1024)); request 0's rows sit
    # on stripe 0 (rank 0), request 1's on stripe 1 (rank 1).
    block_table = torch.tensor([[1], [1], [1], [1]], dtype=torch.int32)
    slot_mapping = torch.tensor([522, 523, 645, 646], dtype=torch.int32)
    kv_seq_lens = torch.tensor([522, 523, 645, 646], dtype=torch.int32)
    kpool_query_lens = [2, 2]

    def _prepared(rank: int) -> SfaDcpAttentionBackend:
        backend = SfaDcpAttentionBackend(
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            scale=0.1,
            sliding_window=0,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            dcp_group=_FakeDcpGroupOf(rank),
            index_topk=2048,
            max_num_reqs=8,
        )
        page_size = 128
        backend.bind_kv_caches(
            [
                LayerCache(
                    key=torch.empty(64, page_size, 1, 512),
                    value=torch.empty(64, page_size, 1, 0),
                    index=torch.empty(256, page_size, 1, 128),
                    kpool_tail=torch.empty(8, 4, 2, 2),
                )
            ]
        )
        metadata = SimpleNamespace(
            slot_mapping=slot_mapping,
            block_table=block_table,
            kv_seq_lens=kv_seq_lens,
            kv_seq_lens_host=None,
            kv_seq_lens_host_values=None,
            q_cu_seq_lens=None,
            q_seq_lens=None,
            expanded_decode_metadata=None,
            is_prefill=False,
            is_chunked_prefill=False,
            has_kv_shard=False,
            kv_split_size=1,
            kv_split_rank=0,
            kpool_query_lens=kpool_query_lens,
            kpool_query_lens_device=None,
            # One per-sequence linear-state id per token row: the kPool tail
            # rides on per-sequence slots, replicated (never owner-localized).
            linear_state_indices=torch.tensor([7, 7, 9, 9], dtype=torch.int64),
        )
        backend.prepare(metadata, graph_mode=False)
        return backend

    pool_tables = {}
    tail_ids = {}
    for rank in (0, 1):
        backend = _prepared(rank)
        with forward_context(_cpu_context(None)):
            context = backend.mla_index_context(SimpleNamespace(layer_id=0))

        # glm5_next.py's kPool update derivation, verbatim: the expanded
        # table collapses to one row per request, then the pool write table
        # applies the (identity on decode) localization wrapper.
        kpool_block_table = context.block_table
        assert kpool_block_table is not None
        kpool_kv_lens = context.actual_seq_kv.reshape(-1).to(torch.int64)
        kpool_block_table, kpool_kv_lens = _kpool_logical_rows(
            kpool_block_table,
            kpool_kv_lens,
            kpool_query_lens,
            None,
        )
        pool_write_block_table = context.localize_pool_block_table(kpool_block_table)
        # The tail state ids collapse to one id per request, unlocalized.
        tail_read, _tail_write = _kpool_logical_state_indices(
            context.kpool_tail_read_indices,
            context.kpool_tail_write_indices,
            kpool_query_lens,
            None,
        )
        pool_tables[rank] = pool_write_block_table
        tail_ids[rank] = tail_read

        # The latent mapping IS owner-local: rank 0 owns stripe 0 (slots 522,
        # 523 -> local page 1, offsets 10/11), rank 1 owns stripe 1 (slots
        # 645, 646 -> local page 1, offsets 5/6).
        if rank == 0:
            torch.testing.assert_close(
                backend._local_slot_mapping,
                torch.tensor([138, 139, -1, -1], dtype=torch.int32),
            )
        else:
            torch.testing.assert_close(
                backend._local_slot_mapping,
                torch.tensor([-1, -1, 133, 134], dtype=torch.int32),
            )

    # The collapse: rows 0 and 2 (each request's first token row) with the
    # per-request kv length taken from the LAST row of the span.
    torch.testing.assert_close(pool_tables[0], torch.tensor([[4, 5, 6, 7], [4, 5, 6, 7]], dtype=torch.int32))
    # ... which is exactly the natural-row replicated expansion of the
    # logical table -- the geometry every rank computes identically.
    torch.testing.assert_close(
        pool_tables[0],
        replicate_pool_write_block_table(torch.tensor([[1], [1]], dtype=torch.int32), 4),
    )
    # The rank-independence pin: identical pool write table and identical
    # tail ids on both ranks, while their latent slots differ above.
    torch.testing.assert_close(pool_tables[1], pool_tables[0])
    assert tail_ids[0].tolist() == [7, 9]
    torch.testing.assert_close(tail_ids[1], tail_ids[0])
