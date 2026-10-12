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

"""Context-Parallel (CP) sequence sharding for the Python model executor.

CP splits a single request's tokens along the sequence dimension across the CP
group so each rank runs attention over its own shard. This module owns the pure
index math: it turns per-sequence lengths into the shard/restore indices plus
the packed query/KV gather indices that one FIA call consumes.

Sharding is *zigzag* (head-tail balanced): each sequence is padded up to a
multiple of ``2 * cp_size`` and cut into ``2 * cp_size`` equal chunks; rank ``r``
owns chunk ``r`` (an early, short-prefix segment) paired with chunk
``2 * cp_size - 1 - r`` (a late, long-prefix segment). Under causal attention a
late token attends far more KV than an early one, so pairing a low chunk with
the mirrored high chunk equalizes per-rank attention work — the load imbalance
of a plain contiguous split (rank ``r`` attends ``(r+1)/cp_size`` of the
sequence) is gone.

Attention is computed against the *full* sequence: ``cp_merge_rows`` /
``cp_gather_kv`` all-gather every rank's shard, so each rank reconstructs the
complete global-order KV and then attends its two owned segments over their
exact causal prefixes. Both segments are contiguous real ranges, so a single
FIA call with per-segment ``actual_seq_lengths`` and ``sparse_mode=3``
(right-aligned causal) masks every row exactly, with no custom mask.

The zigzag index math (shard/restore/query/KV gather indices) is built by the
``xllm_ops::build_cp_context`` C++ op and validated by its gtest
(``cp_context_builder_test``). The shard/merge functions here stay plain torch
index ops so ``merge(all_gather(shard(x))) == x`` holds by construction.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from xllm.python import distributed

# Explicit runtime switch for the CP indexer-cache (kPool/INDEX) write mode.
# Defaults to replicated.
#
# Under CP + kv_split the NPU index cache is FULL-SIZE on every rank
# (kv_cache_shape.cpp init_index_cache_shape: n_blocks * kv_split pages;
# platform.h requires_dsa_indexer_cache_replication), and the indexer input is
# the CP-merged global stream on every rank (Glm5NextMlaAttention.forward
# merges hidden states before select_qli), so every rank can persist ALL
# kv_split pages of every block bit-identically. The m7 owner-sharded mode
# saved zero memory -- the tensor is full-size anyway -- and left the peer
# pages stale, which forced a per-DSA-layer cross-rank all_gather on every
# index read and made the equal-width D(kv_split) transfer topology
# inexpressible (only page 0 of each source resource was ever valid).
# Replicated writes make the local cache a physical full replica: the index
# read becomes the identity, and any single transfer writer supplies every
# valid page, which is what the PR370/M0 planner model already assumes.
_CP_INDEX_WRITE_MODE_ENV = "XLLM_CP_INDEX_WRITE_MODE"
# Only an explicit, recognized value selects a mode; anything else leaves the
# default (replicated) in place rather than silently flipping it. An empty
# value is deliberately NOT a mode: it commonly expands from an unset shell
# variable, so it keeps the default instead of being read as a typo.
_CP_INDEX_WRITE_MODES = frozenset({"replicated", "sharded"})
_CP_INDEX_WRITE_MODE_DEFAULT = "replicated"


def cp_index_write_mode() -> str:
    """Which page layout CP indexer-cache writes persist under ``kv_split``.

    ``"replicated"`` (default) drives every rank to persist ALL
    ``kv_split_size`` pages of every logical block at their natural rows
    (page ``block_table[b] * kv_split_size + stripe``), so the local index
    cache is a physical full replica: index reads skip the cross-rank gather
    and the PD transfer plan can take any single writer per resource.

    ``"sharded"`` keeps the m7 owner-local behavior: each rank persists only
    its own stripe, placed at the FIRST page of each block's index-page group
    (the page the transfer plan's page-level overlap moves against a
    ``kv_split_size == 1`` destination), and every index read reconstructs the
    logical cache through a per-layer all_gather over the CP group. PD
    transfer under this mode therefore supports ONLY ``kv_split_size == 1``
    decode destinations: an equal-width ``kv_split_size > 1`` decode (the
    D(kv_split SfaDcp) topology) plans 1:1 and moves every page of each INDEX
    resource, but only page 0 of each resource was ever written on any rank,
    so the decode would receive stale peer pages -- silent corruption. The
    instance metadata carries this mode (``InstanceInfo.cp_index_write_mode``,
    the proto field the registration publishes), and the transfer plan
    (``plan_kv_split_widths``) refuses an equal-width ``kv_split_size > 1``
    decode destination while this mode is ``sharded`` on either side: link
    such pairs only under ``replicated`` (the default).

    Read at call time (not import time) so a run can be flipped without a
    rebuild. Unset, empty, or unrecognized values keep the default.
    """
    value = os.environ.get(_CP_INDEX_WRITE_MODE_ENV)
    if value is None:
        return _CP_INDEX_WRITE_MODE_DEFAULT
    # Strip exactly the ASCII whitespace set the C++ parser
    # (absl::StripAsciiWhitespace) strips, NOT str.strip()'s full Unicode
    # set: the write geometry decided HERE and the declared mode the C++
    # registration/transfer defenses read must normalize identically, or a
    # value like "\u3000SHARDED\u3000" (full-width space, common from CJK
    # input methods) would write sharded while every C++ side reads the
    # replicated default -- an owner-sharded instance declaring replicated.
    normalized = value.strip(" \t\n\v\f\r").lower()
    if normalized in _CP_INDEX_WRITE_MODES:
        return normalized
    return _CP_INDEX_WRITE_MODE_DEFAULT


def cp_index_write_replicated() -> bool:
    """Whether CP indexer-cache writes replicate every stripe (the default)."""
    return cp_index_write_mode() == _CP_INDEX_WRITE_MODE_DEFAULT


@dataclass(frozen=True)
class CpContext:
    """Per-forward CP sharding plan (zigzag head-tail split).

    All index tensors live on the target device and use int64 (torch gather /
    index_select require int64 indices). Sequence-length metadata retains host
    int lists for FIA interfaces that require them and prebuilds the device
    int32 forms consumed by sparse MLA. Cumulative lengths omit a leading 0.
    """

    cp_size: int
    cp_rank: int
    # Number of (padded) rows this rank holds; identical across ranks so
    # all_gather is well-formed. Layout per sequence: [first-half chunk_len
    # rows, second-half chunk_len rows].
    total_local: int
    # [total_local] global row for each local row, or -1 for a padding row.
    shard_index: torch.Tensor
    # [total_local] shard_index with -1 replaced by 0 so it is a valid gather
    # index; pair with shard_valid_mask to zero padding rows afterwards.
    shard_gather_index: torch.Tensor
    # [total_local] True for real tokens, False for padding rows.
    shard_valid_mask: torch.Tensor
    # [total_global] index into the rank-major all-gather output that restores
    # the original global token order.
    restore_index: torch.Tensor
    # [total_real_local] local rows carrying a real token (this rank's queries),
    # packed per (sequence, half) in local order. Selects the FIA query rows
    # from the local packed hidden and, reused as a scatter index, writes the
    # FIA output back into the [total_local] layout (padding rows stay zero).
    query_index: torch.Tensor
    # FIA host actual_seq_lengths: real query count of each non-empty
    # (sequence, half) segment, cumulative.
    q_cu_seqlens: list[int]
    # Device form consumed by sparse MLA in every layer.
    q_cu_seqlens_tensor: torch.Tensor
    # [sum(kv_cu_seqlens)] index into the global-order KV selecting each
    # segment's causal prefix [0, prefix_len), packed in the same segment order
    # as query_index.
    kv_gather_index: torch.Tensor
    # FIA actual_seq_lengths_kv: causal-prefix length of each segment,
    # cumulative. prefix_len == segment_start + query_count, so with
    # sparse_mode=3 query row i attends KV [0, segment_start + i] exactly.
    kv_cu_seqlens: list[int]
    # Original block-table row for each non-empty (sequence, half) segment.
    segment_seq_indices: torch.Tensor
    # Host absolute KV prefix length for each non-empty (sequence, half)
    # segment.
    segment_kv_seq_lens: list[int]
    # Device form consumed by sparse MLA in every layer.
    segment_kv_seq_lens_tensor: torch.Tensor
    # True when this forward extends an existing paged-cache prefix.
    has_prefix: bool


def build_cp_context(
    q_seq_lens: Sequence[int],
    kv_seq_lens: Sequence[int],
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> CpContext:
    """Build a zigzag CP context from per-sequence lengths.

    ``q_seq_lens`` and ``kv_seq_lens`` are per-request lengths in packed batch
    order. Their difference is the already-cached prefix for chunked prefill.
    The index math runs in the ``xllm_ops::build_cp_context`` C++ op (host
    scalar loops are far cheaper there than in Python on the prefill critical
    path); this wrapper just packs the returned tensors into a ``CpContext``.
    Returns index tensors on ``device``.
    """
    (
        shard_index,
        shard_gather_index,
        shard_valid_mask,
        restore_index,
        query_index,
        kv_gather_index,
        q_cu_seqlens,
        kv_cu_seqlens,
        segment_seq_indices,
        segment_kv_seq_lens,
        total_local,
    ) = torch.ops.xllm_ops.build_cp_context(
        [int(length) for length in q_seq_lens],
        [int(length) for length in kv_seq_lens],
        cp_size,
        cp_rank,
        device,
    )

    return CpContext(
        cp_size=cp_size,
        cp_rank=cp_rank,
        total_local=total_local,
        shard_index=shard_index,
        shard_gather_index=shard_gather_index,
        shard_valid_mask=shard_valid_mask,
        restore_index=restore_index,
        query_index=query_index,
        q_cu_seqlens=q_cu_seqlens,
        q_cu_seqlens_tensor=torch.tensor(q_cu_seqlens, dtype=torch.int32, device=device),
        kv_gather_index=kv_gather_index,
        kv_cu_seqlens=kv_cu_seqlens,
        segment_seq_indices=segment_seq_indices,
        segment_kv_seq_lens=segment_kv_seq_lens,
        segment_kv_seq_lens_tensor=torch.tensor(
            segment_kv_seq_lens,
            dtype=torch.int32,
            device=device,
        ),
        has_prefix=any(kv != q for q, kv in zip(q_seq_lens, kv_seq_lens, strict=True)),
    )


def cp_shard_rows(x: torch.Tensor, ctx: CpContext) -> torch.Tensor:
    """Select this rank's rows from a global packed tensor ``[T_global, ...]``.

    Padding rows are zeroed. Returns ``[total_local, ...]``.
    """
    local = x.index_select(0, ctx.shard_gather_index)
    # Always apply the mask (matching cp_shard_positions). Guarding on
    # ``bool(mask.all())`` would force a device->host sync on the prefill
    # critical path; the elementwise multiply is cheap and graph-safe.
    mask_shape = [ctx.shard_valid_mask.shape[0]] + [1] * (x.dim() - 1)
    return local * ctx.shard_valid_mask.view(mask_shape).to(local.dtype)


def cp_shard_positions(positions: torch.Tensor, ctx: CpContext) -> torch.Tensor:
    """Shard 1-D position ids; padding positions become 0."""
    local = positions.index_select(0, ctx.shard_gather_index)
    return local * ctx.shard_valid_mask.to(local.dtype)


def cp_merge_rows(local: torch.Tensor, ctx: CpContext) -> torch.Tensor:
    """Reassemble the global packed tensor from this rank's local shard.

    All-gathers the rank-major shards over the CP group then restores the
    original global token order. Returns ``[T_global, ...]``.
    """
    gathered = distributed.all_gather(local, 0, ctx.cp_size, "cp")
    return gathered.index_select(0, ctx.restore_index)


def cp_gather_kv(local_kv: torch.Tensor, ctx: CpContext) -> torch.Tensor:
    """All-gather this rank's KV shard back to full global token order.

    ``local_kv`` is ``[total_local, ...]`` (this rank's padded segments of every
    sequence). Returns ``kv_global`` ``[T_global, ...]``: the complete sequence
    in original token order. The caller uses this request-local intermediate to
    write only owner-local persistent cache slots and to construct the current
    attention view; it does not make the persistent cache replicated.

    Identical all-gather + restore as :func:`cp_merge_rows`; kept as a named
    alias to document the KV-gather intent at the attention call site.
    """
    return cp_merge_rows(local_kv, ctx)
