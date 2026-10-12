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

"""Unit tests for DCP paged-KV slot localization."""

from __future__ import annotations

import pytest
import torch

from xllm.python.attention.kv_shard_layout import (
    KVShardLayout,
    localize_index_write_slots,
    localize_pool_write_block_table,
    replicate_pool_write_block_table,
)


def test_localize_slots_does_not_rewrite_worker_logical_slots() -> None:
    layout = KVShardLayout(
        physical_block_size=4,
        dcp_size=2,
        dcp_rank=1,
    )
    logical_slots = torch.tensor([-1, 0, 3, 4, 7, 8, 12], dtype=torch.int32)
    original = logical_slots.clone()

    local_slots = layout.localize_slots(logical_slots)

    assert torch.equal(logical_slots, original)
    assert torch.equal(local_slots, torch.tensor([-1, -1, -1, 0, 3, -1, 4], dtype=torch.int32))


def test_local_seq_lens_are_derived_from_global_kv_seq_lens() -> None:
    layout = KVShardLayout(
        physical_block_size=4,
        dcp_size=2,
        dcp_rank=1,
    )
    global_seq_lens = torch.tensor([0, 4, 6, 8], dtype=torch.int32)

    assert torch.equal(
        layout.local_seq_lens(global_seq_lens),
        torch.tensor([0, 0, 2, 4], dtype=torch.int32),
    )


def test_local_token_count_matches_tensor_layout() -> None:
    layout = KVShardLayout(
        physical_block_size=4,
        dcp_size=2,
        dcp_rank=1,
    )
    global_seq_lens = torch.arange(-2, 18, dtype=torch.int32)

    assert [layout.local_token_count(int(length)) for length in global_seq_lens] == layout.local_seq_lens(
        global_seq_lens
    ).tolist()


def test_indexer_reads_expanded_logical_block_table() -> None:
    layout = KVShardLayout(
        physical_block_size=4,
        dcp_size=2,
        dcp_rank=0,
    )
    logical_blocks = torch.tensor([[3, 7, -1], [0, 2, 4]], dtype=torch.int32)

    assert torch.equal(
        layout.expand_indexer_block_table(logical_blocks),
        torch.tensor([[6, 7, 14, 15, -1, -1], [0, 1, 4, 5, 8, 9]], dtype=torch.int32),
    )


def test_pack_owned_slots_drops_foreign_and_invalid_slots() -> None:
    layout = KVShardLayout(
        physical_block_size=4,
        dcp_size=2,
        dcp_rank=1,
    )
    logical_slots = torch.tensor([-1, 0, 3, 4, 7, 8, 12], dtype=torch.int32)

    packed = layout.pack_owned_slots(logical_slots)

    # Owned slots keep their relative order; foreign and invalid ones end up behind them.
    assert torch.equal(
        packed,
        torch.tensor([0, 3, 4, -1, -1, -1, -1], dtype=torch.int32),
    )


def test_pack_owned_slots_keeps_kpool_tail_inside_the_valid_run() -> None:
    """page=128/DCP=4, top-k prefix 0..2047, kPool tail 2048..2050 (GLM-5.3).

    Rank 0 owns 128 of every 512 logical slots, so 512 prefix slots are local.
    The packed row has to carry all 515 owned slots ahead of the first ``-1``:
    SFA stops scanning at the first ``-1``, so a tail parked behind the padding
    would be dropped.
    """
    layout = KVShardLayout(
        physical_block_size=128,
        dcp_size=4,
        dcp_rank=0,
    )
    logical_slots = torch.cat(
        [
            torch.arange(2048, dtype=torch.int32),
            torch.arange(2048, 2051, dtype=torch.int32),
        ]
    )

    packed = layout.pack_owned_slots(logical_slots)

    assert torch.equal(packed[:512], torch.arange(512, dtype=torch.int32))
    assert torch.equal(packed[512:515], torch.tensor([512, 513, 514], dtype=torch.int32))
    assert bool((packed[515:] < 0).all())
    assert int((packed >= 0).sum()) == 515


def test_graph_padded_zero_block_expands_to_valid_indexer_pages() -> None:
    layout = KVShardLayout(
        physical_block_size=128,
        dcp_size=4,
        dcp_rank=0,
    )
    padded_row = torch.zeros((1, 2), dtype=torch.int32)

    expanded = layout.expand_indexer_block_table(padded_row)

    assert torch.equal(
        expanded,
        torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.int32),
    )


@pytest.mark.parametrize("split", [1, 2, 4])
def test_indexer_page_bounds(split: int) -> None:
    layout = KVShardLayout(physical_block_size=4, dcp_size=split, dcp_rank=0)
    logical_blocks = torch.arange(16, dtype=torch.int32).reshape(1, -1)
    pages = layout.expand_indexer_block_table(logical_blocks)
    assert torch.equal(pages, torch.arange(16 * split, dtype=torch.int32).reshape(1, -1))
    assert pages[0, -1].item() == 16 * split - 1
    # Read every mapped page from a full-history cache, including the last page.
    cache = torch.arange(16 * split * 4).reshape(16 * split, 4)
    assert cache[pages.long()].shape == (1, 16 * split, 4)
    if split == 4:
        assert pages[0, 16:20].tolist() == [16, 17, 18, 19]
        with pytest.raises(IndexError):
            _ = cache[:16][pages.long()]

    padded = layout.expand_indexer_block_table(torch.tensor([[15, -1]], dtype=torch.int32))
    assert padded[0, :split].tolist() == list(range(15 * split, 16 * split))
    assert padded[0, split:].tolist() == [-1] * split


def test_localize_index_write_slots_expands_latent_pages_to_group_first_pages() -> None:
    """The NPU index cache groups ``dcp_size`` pages under one logical block,
    so a paged index write must move latent page ``p`` to index page
    ``p * dcp_size`` -- the first page of the block's group and the page the
    PD transfer plan's page-level overlap moves."""
    local_slots = torch.tensor([0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1], dtype=torch.int32)

    expanded = localize_index_write_slots(local_slots, page_size=2, dcp_size=2)

    # Latent pages 0, 1, 2 become index pages 0, 2, 4; peer slots stay invalid.
    assert torch.equal(expanded, torch.tensor([0, 1, -1, -1, 4, 5, -1, -1, 8, 9, -1, -1], dtype=torch.int32))
    assert localize_index_write_slots(local_slots, page_size=2, dcp_size=4).tolist() == [
        0,
        1,
        -1,
        -1,
        8,
        9,
        -1,
        -1,
        16,
        17,
        -1,
        -1,
    ]
    # kv_split_size == 1 keeps the caller's mapping object.
    assert localize_index_write_slots(local_slots, page_size=2, dcp_size=1) is local_slots


def test_replicate_pool_write_block_table_addresses_natural_rows() -> None:
    """A replicated pool write (``XLLM_CP_INDEX_WRITE_MODE=replicated``, the
    default) must persist every stripe's page at its NATURAL row: output
    column ``i`` (stripe ``i % dcp_size`` of logical block
    ``block_table[i // dcp_size]``) addresses page
    ``block_table[i // dcp_size] * dcp_size + i % dcp_size``, which is the
    group layout platform.h defines (logical block B owns index rows
    ``[B * split, (B + 1) * split)``). No column is filtered, so every rank
    writes every page of every block and the local cache becomes a physical
    full replica."""
    block_table = torch.tensor([[3, 7, -1]], dtype=torch.int32)

    replicated = replicate_pool_write_block_table(block_table, dcp_size=2)

    # Stripes 0 and 1 of logical block 3 land on pages 6 and 7; logical block
    # 7 on pages 14 and 15; the invalid tail column stays invalid for both
    # stripes. Contrast with the sharded view, which points only the owner's
    # column at the group-FIRST page (6 / 14) and invalidates the peer.
    assert torch.equal(
        replicated,
        torch.tensor([[6, 7, 14, 15, -1, -1]], dtype=torch.int32),
    )
    sharded_owner0 = localize_pool_write_block_table(block_table, dcp_size=2, dcp_rank=0)
    sharded_owner1 = localize_pool_write_block_table(block_table, dcp_size=2, dcp_rank=1)
    assert not torch.equal(replicated, sharded_owner0)
    assert not torch.equal(replicated, sharded_owner1)
    # Neither owner's sharded view alone covers the natural rows: the union of
    # the two owners' valid pages is {6, 14} (each written at the group-first
    # row), never {6, 7, 14, 15}.
    assert sorted(sharded_owner0[sharded_owner0 >= 0].tolist()) == [6, 14]
    assert sorted(sharded_owner1[sharded_owner1 >= 0].tolist()) == [6, 14]

    # A four-way split interleave: logical block 2 spans pages 8..11.
    assert torch.equal(
        replicate_pool_write_block_table(torch.tensor([[2]], dtype=torch.int32), dcp_size=4),
        torch.tensor([[8, 9, 10, 11]], dtype=torch.int32),
    )
    # kv_split_size == 1 keeps the caller's table object, so the full-replica
    # launch passes exactly what it passed before.
    assert replicate_pool_write_block_table(block_table, dcp_size=1) is block_table
