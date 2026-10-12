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

"""DeepSeek-V4 prefill context-parallel plan.

Faithful Python port of ``xllm::layer::v4_cp`` in
``xllm/core/layers/npu_torch/deepseek_v4_cp_context.h``.  The framework's
existing :mod:`xllm.python.model_executor.cp_utils` implements a zigzag split
that is correct for FIA/Qwen3; DeepSeek-V4's DSA metadata is one row per
sequence and hides 3D, so it needs this contiguous split instead.

The key invariant is the same as C++: the KV / compressor / index-cache write
path runs on the full global token set (every CP rank holds a full KV replica),
while only the query axis is localized to this rank's contiguous rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from xllm.python import distributed
from xllm.python.models.deepseek_v4 import _expand_half_rope_cos_sin


@dataclass
class DeepseekV4CpContext:
    """Per-forward prefill CP plan for DeepSeek-V4.

    Unlike the framework zigzag ``CpContext``, this keeps the query axis in one
    contiguous segment per sequence and never renumbers positions. Local rows
    retain their true global position ids, which keeps RoPE and slot mapping
    correct without a separate position table.
    """

    cp_size: int = 1
    cp_rank: int = 0
    global_token_count: int = 0
    local_token_count: int = 0

    # Global row indices this rank owns, ascending in global order.
    local_row_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    # Maps global row -> its position in rank-major gathered output.
    restore_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    tokens_per_rank: list[int] = field(default_factory=list)
    local_positions: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    global_positions: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))

    global_q_seq_lens: list[int] = field(default_factory=list)
    global_kv_seq_lens: list[int] = field(default_factory=list)
    local_q_seq_lens: list[int] = field(default_factory=list)
    local_kv_seq_lens: list[int] = field(default_factory=list)
    local_kv_cu_seq_lens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))
    global_q_cu_seq_lens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))
    # Full-position RoPE caches keyed by compression ratio. The C++ context
    # keeps ``global_rope_by_ratio`` as built cos/sin tensors; Python keeps the
    # full cache plus ``global_positions`` and builds the same request-shaped
    # tensors lazily via :meth:`global_rope`, which is what the mock round-trip
    # lets a test verify without owning a rotary cache implementation.
    global_rope_caches: dict[int, torch.Tensor] = field(default_factory=dict)
    # Request-shaped built cos/sin tensors keyed by compression ratio, mirroring
    # the C++ ``global_rope_by_ratio`` map. ``set_global_rope_pair`` expands
    # compressor-format half-width tables into the full-width interleave format
    # consumed by the KV query path. The compressor still reads its separate
    # half-width tables from the DSA metadata and expands them internally.
    global_rope_by_ratio: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    def enabled(self) -> bool:
        """Whether this plan owns a live CP split for the current forward.

        Empty CP ranks are legal and must stay enabled: they participate in the
        uneven ``gather_restore`` collective even when their local row set is
        empty, otherwise the rank-major restore indices on the other ranks
        would be wrong.
        """
        return self.cp_size > 1 and distributed is not None

    def set_global_rope_cache(self, ratio: int, cos_sin_cache: torch.Tensor) -> None:
        """Register a full-position RoPE cache for one compression ratio."""
        self.global_rope_caches[int(ratio)] = cos_sin_cache

    def global_rope(self, ratio: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return global-position KV cos/sin for ``ratio``.

        Return full-width interleave-format cos/sin for the KV rotary kernel.
        Prefer the requested ratio, fall back to ratio 1, and return undefined
        tensors when CP is off or no table was registered. Built pairs are
        normalized to full width by :meth:`set_global_rope_pair`; lazy cache
        indexing uses the same helper.
        """
        built = self.global_rope_by_ratio.get(int(ratio))
        if built is None:
            built = self.global_rope_by_ratio.get(1)
        if built is not None:
            return built
        cache = self.global_rope_caches.get(int(ratio))
        if cache is None:
            cache = self.global_rope_caches.get(1)
        if cache is None or self.global_positions is None or self.global_positions.numel() == 0:
            return None, None
        cos_sin = cache.index_select(0, self.global_positions.long())
        return _expand_half_rope_cos_sin(cos_sin)

    def set_global_rope_pair(self, ratio: int, cos_sin_pair: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Register a global cos/sin pair for one compression ratio.

        ``build_dsa_rope_metadata`` stores compressor-format half-width tables.
        The model-facing KV path needs the full-width interleave format, so
        normalize here without mutating the compressor-owned inputs.
        """
        if cos_sin_pair is None or cos_sin_pair[0] is None or cos_sin_pair[1] is None:
            return
        cos, sin = cos_sin_pair
        rope_dim = cos.size(-1) * 2
        if sin.size(-1) * 2 != rope_dim:
            raise RuntimeError(f"DeepSeek-V4 CP RoPE cos/sin widths differ: cos={cos.shape}, sin={sin.shape}")
        full_cos, full_sin = _expand_half_rope_cos_sin(torch.cat((cos, sin), dim=-1))
        self.global_rope_by_ratio[int(ratio)] = (full_cos, full_sin)

    def shard_rows(self, global_tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled():
            return global_tensor
        indices = self.local_row_indices.to(global_tensor.device)
        return global_tensor.index_select(0, indices.to(torch.int64))

    def gather_restore(self, local_tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled():
            return local_tensor
        if self.cp_rank < 0 or self.cp_rank >= len(self.tokens_per_rank):
            raise RuntimeError("DeepSeek-V4 CP rank is outside the token-count table")
        # C++ ``DeepseekV4CpContext::gather_restore`` calls
        # ``parallel_state::gather(input, cp_group, tokens_per_rank)``, which is
        # the uneven path when segments differ. ``all_gather_variable`` pads to
        # the group max on the fly and concatenates the valid chunks in rank
        # order, exactly the layout ``restore_indices`` was built for.
        gathered = distributed.all_gather_variable(
            local_tensor.contiguous(),
            self.tokens_per_rank,
            self.cp_rank,
            "cp",
        )
        restore = self.restore_indices.to(gathered.device)
        return gathered.index_select(0, restore)


def _compute_cp_rows_by_rank(
    cp_size: int,
    global_q_seq_lens: Sequence[int],
) -> list[list[int]]:
    """Return global row indices owned by each CP rank (rank-major)."""
    if not global_q_seq_lens:
        return [[] for _ in range(cp_size)]
    rows_by_rank: list[list[int]] = [[] for _ in range(cp_size)]
    seq_base = 0
    for q_len in global_q_seq_lens:
        segment = (int(q_len) + cp_size - 1) // cp_size
        for rank in range(cp_size):
            start = min(rank * segment, int(q_len))
            end = min(start + segment, int(q_len))
            rows_by_rank[rank].extend(range(seq_base + start, seq_base + end))
        seq_base += int(q_len)
    return rows_by_rank


def _compute_cp_local_seq_lens(
    cp_size: int,
    cp_rank: int,
    global_q_seq_lens: Sequence[int],
    global_kv_seq_lens: Sequence[int],
) -> tuple[list[int], list[int]]:
    local_q: list[int] = []
    local_kv: list[int] = []
    for q_len, kv_len in zip(global_q_seq_lens, global_kv_seq_lens):
        segment = (q_len + cp_size - 1) // cp_size
        start = min(cp_rank * segment, q_len)
        end = min(start + segment, q_len)
        local_q.append(end - start)
        prefix = kv_len - q_len
        local_kv.append(prefix + end)
    return local_q, local_kv


def build_deepseek_v4_cp_context(
    cp_size: int,
    cp_rank: int,
    global_q_seq_lens: Sequence[int],
    global_kv_seq_lens: Sequence[int],
    global_positions: torch.Tensor | None = None,
) -> DeepseekV4CpContext:
    """Build a contiguous DeepSeek-V4 CP plan for one prefill forward."""
    context = DeepseekV4CpContext(cp_size=cp_size, cp_rank=cp_rank)
    context.global_q_seq_lens = list(global_q_seq_lens)
    context.global_kv_seq_lens = list(global_kv_seq_lens)
    if cp_size <= 1:
        return context
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"DeepSeek-V4 CP rank {cp_rank} is outside [0, {cp_size})")
    if len(context.global_q_seq_lens) != len(context.global_kv_seq_lens):
        raise ValueError(
            "DeepSeek-V4 CP expects one kv length per q length: "
            f"{len(context.global_q_seq_lens)} q vs {len(context.global_kv_seq_lens)} kv"
        )

    context.global_token_count = sum(context.global_q_seq_lens)
    rows_by_rank = _compute_cp_rows_by_rank(cp_size, context.global_q_seq_lens)
    context.tokens_per_rank = [len(rows) for rows in rows_by_rank]
    context.local_token_count = context.tokens_per_rank[cp_rank]
    context.local_row_indices = torch.tensor(rows_by_rank[cp_rank], dtype=torch.int64)

    global_rows = torch.cat(
        [torch.tensor(rows, dtype=torch.int64) for rows in rows_by_rank],
        dim=0,
    )
    if global_rows.numel() != context.global_token_count:
        raise RuntimeError("DeepSeek-V4 CP segments must cover every global row exactly once")
    restore = torch.empty(context.global_token_count, dtype=torch.int64)
    if global_rows.numel() > 0:
        restore[global_rows] = torch.arange(global_rows.numel(), dtype=torch.int64)
    context.restore_indices = restore

    context.local_q_seq_lens, context.local_kv_seq_lens = _compute_cp_local_seq_lens(
        cp_size, cp_rank, context.global_q_seq_lens, context.global_kv_seq_lens
    )

    q_lens = torch.tensor(context.global_q_seq_lens, dtype=torch.int32)
    context.global_q_cu_seq_lens = torch.cat((torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, dim=0)))
    kv_lens = torch.tensor(context.local_kv_seq_lens, dtype=torch.int32)
    context.local_kv_cu_seq_lens = torch.cat((torch.zeros(1, dtype=torch.int32), torch.cumsum(kv_lens, dim=0)))

    device = (
        global_positions.device
        if global_positions is not None and global_positions.device.type != "cpu"
        else torch.device("cpu")
    )
    context.local_row_indices = context.local_row_indices.to(device)

    if global_positions is not None and global_positions.numel() > 0:
        context.local_positions = global_positions.index_select(0, context.local_row_indices)
        context.global_positions = global_positions.contiguous()

    context.restore_indices = context.restore_indices.to(device)
    context.local_positions = context.local_positions.to(device)
    context.global_positions = context.global_positions.to(device)
    context.global_q_cu_seq_lens = context.global_q_cu_seq_lens.to(device)
    context.local_kv_cu_seq_lens = context.local_kv_cu_seq_lens.to(device)
    return context
