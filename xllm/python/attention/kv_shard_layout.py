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

"""Logical-to-physical paged-KV mapping for DCP."""

from __future__ import annotations

import torch


def has_rope_dim(tensor: torch.Tensor | None) -> bool:
    """Whether an MLA rope operand carries a rope dimension.

    NoPE models (``qk_rope_head_dim == 0``) surface the rope operand as ``None``
    or as a zero-width tensor; both mean "skip the rope path".
    """
    return tensor is not None and tensor.shape[-1] > 0


class KVShardLayout:
    """Maps a logical paged-KV coordinate onto one rank's physical cache."""

    INVALID_SLOT = -1

    def __init__(
        self,
        physical_block_size: int,
        dcp_size: int,
        dcp_rank: int,
    ) -> None:
        if physical_block_size <= 0:
            raise ValueError(f"physical_block_size must be positive, got {physical_block_size}")
        if dcp_size <= 0:
            raise ValueError(f"dcp_size must be positive, got {dcp_size}")
        if dcp_rank < 0 or dcp_rank >= dcp_size:
            raise ValueError(
                f"dcp_rank must satisfy 0 <= dcp_rank < dcp_size, got dcp_rank={dcp_rank}, dcp_size={dcp_size}"
            )
        self.physical_block_size = physical_block_size
        self.dcp_size = dcp_size
        self.dcp_rank = dcp_rank

    @property
    def logical_block_size(self) -> int:
        return self.physical_block_size * self.dcp_size

    def local_token_count(self, global_token_count: int) -> int:
        """Return the tokens owned by this rank in one logical sequence."""
        token_count = max(int(global_token_count), 0)
        full_blocks, remainder = divmod(token_count, self.logical_block_size)
        rank_start = self.dcp_rank * self.physical_block_size
        owned_remainder = min(
            max(remainder - rank_start, 0),
            self.physical_block_size,
        )
        return full_blocks * self.physical_block_size + owned_remainder

    def local_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        seq_lens = seq_lens.clamp_min(0)
        logical = self.logical_block_size
        physical = self.physical_block_size
        full_blocks = torch.div(seq_lens, logical, rounding_mode="floor")
        remainder = torch.remainder(seq_lens, logical)
        rank_start = self.dcp_rank * physical
        owned_in_remainder = torch.clamp(remainder - rank_start, 0, physical)
        return full_blocks * physical + owned_in_remainder

    def localize_slots(self, logical_slots: torch.Tensor) -> torch.Tensor:
        valid_slots = logical_slots >= 0
        safe_slots = logical_slots.clamp_min(0)
        logical_offsets = torch.remainder(safe_slots, self.logical_block_size)
        owner_ranks = torch.div(
            logical_offsets,
            self.physical_block_size,
            rounding_mode="floor",
        )
        owned_slots = valid_slots & (owner_ranks == self.dcp_rank)
        logical_block_ids = torch.div(
            safe_slots,
            self.logical_block_size,
            rounding_mode="floor",
        )
        local_offsets = torch.remainder(logical_offsets, self.physical_block_size)
        local_slots = logical_block_ids * self.physical_block_size + local_offsets
        return torch.where(
            owned_slots,
            local_slots,
            torch.full_like(local_slots, self.INVALID_SLOT),
        )

    def pack_owned_slots(self, logical_slots: torch.Tensor) -> torch.Tensor:
        """Map logical slots onto this rank and pack every owned slot to the front.

        SFA walks ``sparse_indices`` until the first ``INVALID_SLOT``, so all slots
        this rank owns must form one leading run. A kPool indexer appends tail
        columns next to the top-k ones; packing the whole row keeps that tail in
        the same run instead of leaving it behind the ``-1`` padding.
        """
        localized = self.localize_slots(logical_slots)
        width = int(localized.shape[-1])
        original_order = torch.arange(
            width,
            dtype=torch.float32,
            device=localized.device,
        ).expand_as(localized)
        pack_keys = original_order + (localized < 0).to(torch.float32) * width
        _, pack_order = torch.sort(pack_keys, dim=-1)
        return torch.gather(localized, dim=-1, index=pack_order.to(torch.int32))

    def expand_indexer_block_table(
        self,
        logical_block_table: torch.Tensor,
    ) -> torch.Tensor:
        if logical_block_table.dim() != 2:
            raise ValueError("indexer block table must be two-dimensional")
        shard_offsets = torch.arange(
            self.dcp_size,
            dtype=logical_block_table.dtype,
            device=logical_block_table.device,
        )
        expanded = logical_block_table.unsqueeze(-1) * self.dcp_size + shard_offsets
        expanded = torch.where(
            logical_block_table.unsqueeze(-1) >= 0,
            expanded,
            torch.full_like(expanded, -1),
        )
        return expanded.flatten(start_dim=1).contiguous()


def localize_pool_write_block_table(
    block_table: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
) -> torch.Tensor:
    """Local page table for an owner-sharded (``XLLM_CP_INDEX_WRITE_MODE=sharded``)
    compressed-kPool write.

    A pool page holds ``block_size / index_kpool`` pools, so one logical block
    of ``block_size * dcp_size`` tokens spans ``dcp_size`` physical pool pages.
    Both pool writers (``update_compressed_kpool`` and the compact Triton
    update) index a pool table by the **global physical page**
    ``pool_id // pools_per_page`` and skip a page whose id is negative, while
    the local cache stores only this rank's own stripe. The NPU groups those
    ``dcp_size`` pages under the logical block's resource (platform.h:
    logical block B owns index rows ``[B * dcp_size, (B + 1) * dcp_size)``),
    and the PD transfer plan's page-level overlap reads PAGE 0 of every source
    resource, so this rank's stripe must live at the first page of the block's
    group. This view therefore maps every owned column onto
    ``block_table[column // dcp_size] * dcp_size`` and marks the peer-owned
    columns invalid, which is what lets both writers stay ownership-agnostic
    while the transferred page is exactly the page written.

    Returns ``block_table`` unchanged when ``dcp_size <= 1``, so a full-replica
    (``kv_split_size == 1``) launch keeps its existing table object.
    """
    if dcp_size <= 1:
        return block_table
    if block_table.dim() != 2:
        raise ValueError("pool block table must be two-dimensional")
    repeated = block_table.repeat_interleave(dcp_size, dim=1)
    owners = torch.arange(repeated.shape[1], device=block_table.device) % dcp_size
    owned = (owners == dcp_rank).expand_as(repeated)
    grouped = repeated * dcp_size
    return torch.where(
        owned & (repeated >= 0),
        grouped,
        torch.full_like(grouped, KVShardLayout.INVALID_SLOT),
    )


def replicate_pool_write_block_table(
    block_table: torch.Tensor,
    dcp_size: int,
) -> torch.Tensor:
    """Page table for a replicated (``XLLM_CP_INDEX_WRITE_MODE=replicated``,
    the default) compressed-kPool write.

    Same column contract as :func:`localize_pool_write_block_table` -- one
    input column per logical block, one output column per global physical
    page, consumed by both pool writers through their negative-id skip -- but
    with the full-table arithmetic and no owner filter: output column ``i``
    (stripe ``i % dcp_size`` of logical block ``block_table[i // dcp_size]``)
    addresses that stripe's NATURAL page
    ``block_table[i // dcp_size] * dcp_size + i % dcp_size``. Every rank
    computes identical pools from the CP-merged global stream, so every rank
    persists every stripe of every block and the local index cache becomes a
    physical full replica: the read needs no cross-rank gather, and any single
    PD transfer writer supplies every valid page of every resource.

    Returns ``block_table`` unchanged when ``dcp_size <= 1``, so a full-replica
    (``kv_split_size == 1``) launch keeps its existing table object.
    """
    if dcp_size <= 1:
        return block_table
    if block_table.dim() != 2:
        raise ValueError("pool block table must be two-dimensional")
    # The natural-row expansion is exactly KVShardLayout's indexer-block
    # table view: column i addresses block_table[i // dcp_size] * dcp_size
    # + i % dcp_size and a negative group stays invalid (INVALID_SLOT ==
    # -1 matches expand_indexer_block_table's fill). Reuse the authority
    # implementation that _materialized_block_table already consumes
    # instead of keeping a second copy of the group-layout invariant.
    return KVShardLayout(physical_block_size=1, dcp_size=dcp_size, dcp_rank=0).expand_indexer_block_table(block_table)


def localize_index_write_slots(
    slot_mapping: torch.Tensor,
    page_size: int,
    dcp_size: int,
) -> torch.Tensor:
    """Remap owner-local latent-cache slots onto the index cache's page group.

    This is the ``XLLM_CP_INDEX_WRITE_MODE=sharded`` write geometry. The NPU
    index cache groups ``dcp_size`` physical pages under one logical block
    (platform.h), while the owner-local slot mapping addresses the latent
    cache, which stores a single page per logical block. A paged index write
    must therefore land on page ``page * dcp_size`` -- the first page of the
    logical block's group, exactly the page the PD transfer plan's page-level
    overlap moves against a ``kv_split_size == 1`` destination. Peer-owned
    slots stay invalid, so the scatter's padding redirect is unchanged.

    The replicated mode (the default) needs no remapping at all: a global
    logical slot ``s`` (block ``s // (page_size * dcp_size)``, stripe
    ``(s % (page_size * dcp_size)) // page_size``) already addresses its
    natural index row ``(s // (page_size * dcp_size)) * page_size * dcp_size +
    s % (page_size * dcp_size) == s``, so the backend hands the writers the
    global slot mapping unchanged and every stripe's page is written.

    Returns ``slot_mapping`` unchanged when ``dcp_size <= 1``, so a
    full-replica (``kv_split_size == 1``) launch keeps its mapping object.
    """
    if dcp_size <= 1:
        return slot_mapping
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    valid_slots = slot_mapping >= 0
    safe_slots = slot_mapping.clamp_min(0)
    pages = torch.div(safe_slots, page_size, rounding_mode="floor")
    grouped = safe_slots + pages * page_size * (dcp_size - 1)
    return torch.where(
        valid_slots,
        grouped,
        torch.full_like(grouped, KVShardLayout.INVALID_SLOT),
    )
