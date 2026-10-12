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

"""DeepSeek-V4.1 Flash CSA2 attention backend (pure-torch reference path).

Implements the official reference forward semantics (the DeepSeek-V4.1 Flash
reference ``inference/model.py``) on top of the
runtime :class:`DsaMetadata` contract built by :mod:`dsa_metadata`:

  * SWA ring writes/reads through the SWA-group slot mappings;
  * the ratio-2 softmax-gated compressor with per-token state parking in the
    SWA ring slots (contract section 2), ratio-1 plain projection;
  * the CSA2 indexer (FP4-QDQ query/key, relu-weighted scoring, hierarchical
    candidate pool, top-k with the window-row offset baked in);
  * the sparse attention gather over ``[window rows ; top-k compressed rows]``
    with the attention-sink normalizer term.

Everything is plain torch, so the forward runs eagerly on the NPU until the
NPU kernel milestone replaces the internals without touching the model-facing
contract; there is no CPU/CUDA fallback (DeepSeek-V4.1 is NPU-only). The model
owns all weights; the backend only orchestrates caches and math through the
weight helpers on the V4.1 attention layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from scripts.logger import logger
from xllm.python.attention.csa_attention import (
    DsaAttentionBackend,
    _CompressedAttentionCacheMapping,
    _get_layer_cache_tensor,
    _scatter_by_slot,
)
from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    DSA_CACHE_TOKEN,
    READ_TABLE_LAYOUT_RING,
    DsaMetadata,
    DsaMetadataBuilder,
    build_cache_specs_v41,
)
from xllm.python.models.deepseek_v41 import (
    V41_QUANT_ASCEND,
    V41_QUANT_NONE,
    fp4_act_qdq_e4m3_scale,
    fp4_act_qdq_e8m0,
)

# Softmax-scale exponent base of the sparse kernel's running-max floor.
_NEG_INF_FLOOR = -1e30
# Query rows per attention tile (the reference kernel processes 64 gathered
# rows per tile; a few hundred rows keep the [s, H, n] score tile bounded).
_ATTENTION_QUERY_TILE = 256
# Query rows per indexer scoring tile. The fp32 score intermediate is
# [tile, index_heads, capacity]; a whole-prompt chunk against a long-context
# committed capacity reaches multiple GiB per layer (30k ctx x 2k chunk x 32
# heads x 4B ~= 4 GiB), which overflows the activation headroom of the
# W8A8+engram full model. Top-k selection is per query row, so tiling over
# the query axis is exact; decode/graph batches (q_len <= tile) take the
# single-tile path unchanged.
_INDEXER_QUERY_TILE = 256


@dataclass
class _SwaGroupMetadata:
    """SWA-group tensors shared by every layer (one ring layout per request)."""

    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    block_size: int
    cache: torch.Tensor


@dataclass
class Csa2LayerContext:
    """Per-layer activations the model hands to the backend.

    ``x`` is the layer's attention input (the post-``attn_norm`` collapsed
    stream that also feeds its Q / SWA-KV / compressor projections), ``qr``
    the shared q-lora latent, and ``cos_sin_cache`` the layer's own RoPE table
    (the compress-theta table for every ratio > 0 layer).
    """

    x: torch.Tensor
    qr: torch.Tensor
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor


@dataclass
class CSA2SharedRuntime:
    """Cross-layer hand-off state for one model forward.

    Mirrors the reference ``SharedAttentionRuntime`` (model.py:1166-1180):
    kv-source layers publish their caches, index-source layers publish the
    top-k, and the candidate-source layer publishes the candidate pool. The
    candidate pool is per sequence (the reference keeps a ``[b, s, t]`` mask;
    each sequence's pool only covers its own committed rows). Layers run in
    order, so one rolling slot per field is enough.
    """

    key_cache: torch.Tensor | None = None
    key_block_table: torch.Tensor | None = None
    index_cache: torch.Tensor | None = None
    index_block_table: torch.Tensor | None = None
    compress_ratio: int = 0
    topk_idxs: torch.Tensor | None = None
    topk_layer: int = -1
    candidates: list[torch.Tensor | None] | None = None
    candidates_layer: int = -1


def apply_interleaved_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Interleaved-pair complex RoPE on the last ``2 * half`` dims (model.py:392-406).

    Adjacent element pairs ``(x[2i], x[2i+1])`` form one complex number that
    is multiplied by ``exp(i * theta_p)``; ``inverse`` conjugates the rotation,
    which de-rotates the attention output by the query's own rotation. The
    per-position table broadcasts over any leading head/batch axes of ``x``
    (the position axis is always ``x``'s first axis). The math runs in fp32
    and the result is cast back to ``x``'s dtype.
    """
    half = cos_sin_cache.size(-1) // 2
    rows = cos_sin_cache.index_select(0, positions.reshape(-1).to(torch.int64))
    cos = rows[..., :half].to(torch.float32)
    sin = rows[..., half:].to(torch.float32)
    if inverse:
        sin = -sin
    for _ in range(x.dim() - 2):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    dim = x.size(-1)
    tail = x[..., dim - 2 * half :].to(torch.float32)
    even = tail[..., 0::2]
    odd = tail[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    rotated = rotated.flatten(-2).to(x.dtype)
    if 2 * half == dim:
        return rotated
    return torch.cat((x[..., : dim - 2 * half], rotated), dim=-1)


def _ensure_seq_lens_tensor(
    ctx_lens: torch.Tensor | Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Coerce sequence lengths to a tensor (graph safety: never a host read).

    The eager path may still hand over plain lists; the graph path always
    receives the persistent ``dsa.seq_lens`` tensor whose values are refreshed
    per replay, so every consumer below must treat it as device data.
    """
    if torch.is_tensor(ctx_lens):
        return ctx_lens
    return torch.tensor(list(ctx_lens), dtype=torch.int64, device=device)


def _indexer_score(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Indexer scores ``score[s, t] = sum_h w[s, h] * relu(q[s, h] . k[t])``.

    Mirrors model.py:556-557: the per-head dot products are rectified BEFORE
    the learned per-head weights are applied and summed.
    """
    score = torch.einsum("shd,td->sht", q.float(), index_k.float())
    score = score.relu_() * weights.float().unsqueeze(-1)
    return score.sum(dim=1)


def _select_candidate_blocks(
    score: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k (model.py:583-610).

    ``score`` is ``[s, t]`` with unreachable positions already ``-inf``.
    ``compress_lens`` holds the per-query committed counts. Returns a bool
    mask shaped like ``score``.
    """
    width = score.size(-1)
    padded = torch.nn.functional.pad(score, (0, -width % block_size), value=float("-inf"))
    blocks = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = blocks.size(-1)
    # Pin the block holding the query's newest committed position: it is only
    # partly filled and could otherwise be outscored by an older, full block.
    last = torch.div(compress_lens - 1, block_size, rounding_mode="floor")
    blocks = blocks.masked_fill(
        torch.arange(num_blocks, device=score.device) == last.unsqueeze(-1),
        float("inf"),
    )
    top = blocks.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _indexer_topk(
    score: torch.Tensor,
    compress_lens: torch.Tensor,
    index_topk: int,
    offset: int,
) -> torch.Tensor:
    """Top-k selection re-sorted by position; unreachable picks become -1 (model.py:577-580).

    ``compress_lens`` is the per-query visibility count ``(p + 1) // ratio``;
    valid picks are shifted by ``offset`` (the window-row count of the
    concatenated gather tensor). A pick is recoverable only when its score is
    finite *and* its position is visible: the Reindex candidate pool masks
    out-of-pool rows with ``-inf`` while those rows still fall inside the
    visible range, so a pool narrower than ``index_topk`` would otherwise leak
    them into the picks -- the finite check drops them onto the -1 sentinel.
    Ties are broken toward lower positions via a
    stable descending sort: ``torch.topk`` leaves tie order
    implementation-defined, which would make a whole-prompt prefill and its
    token-by-token replay disagree on the zero-score ties the relu-weighted
    scoring produces.
    """
    topk = min(index_topk, score.size(-1))
    order = torch.argsort(score, dim=-1, descending=True, stable=True)
    idxs = order[..., :topk]
    idxs = idxs.sort(dim=-1).values
    finite = torch.isfinite(score.gather(-1, idxs))
    visible = compress_lens.unsqueeze(-1)
    return torch.where((idxs < visible) & finite, idxs + offset, -1).to(torch.int32)


def _picked_row_mask(idxs: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Row mask of each query's index picks, with static output shapes.

    Equivalent to the nonzero formulation
    ``row_mask[q, idxs[q, j]] = idxs[q, j] >= 0`` but ACL-graph-safe:
    ``torch.nonzero`` and boolean-mask indexing produce data-dependent
    output shapes (the pick count varies with window visibility and top-k
    results across replays), which a captured graph cannot re-shape.
    Instead the pick bits scatter into a mask one column wider and invalid
    picks (-1, or past the gathered row count when the capacity bound is
    narrower than a replay's context) route to the never-read dummy column;
    duplicate picks write the same True bit.
    """
    picked = idxs >= 0
    valid = picked & (idxs < num_rows)
    safe_idxs = torch.where(valid, idxs, torch.full_like(idxs, num_rows))
    wide_mask = torch.zeros(idxs.shape[0], num_rows + 1, dtype=torch.bool, device=idxs.device)
    wide_mask.scatter_(1, safe_idxs, valid)
    return wide_mask[:, :num_rows]


def _window_visible_idxs(
    positions_seq: torch.Tensor,
    col_pos: torch.Tensor,
    cols: torch.Tensor,
    win: int,
) -> torch.Tensor:
    """Per-query SWA window index matrix over the fixed-capacity window columns.

    The window columns are ``[win ring candidates ; current chunk]``: a ring
    candidate's absolute position may be negative (before the sequence start)
    and the chunk columns follow it. Query ``i`` sits at absolute position
    ``positions_seq[i]`` and must see exactly the columns whose absolute
    position lies in ``[max(p - win + 1, 0), p]`` -- a *per-query* lower bound
    (only the chunk's first token shares the chunk-level
    ``max(prev - win + 1, 0)``). Masked columns become the ``-1`` sentinel,
    which :func:`_picked_row_mask` drops.
    """
    window_lo = torch.clamp(positions_seq - win + 1, min=0)
    valid = (col_pos.unsqueeze(0) >= window_lo.unsqueeze(-1)) & (col_pos.unsqueeze(0) <= positions_seq.unsqueeze(-1))
    return torch.where(valid, cols.unsqueeze(0).expand(positions_seq.numel(), col_pos.numel()), -1)


def _token_cache_slots(
    block_table: torch.Tensor | None,
    cache: torch.Tensor | None,
    rows: torch.Tensor,
    seqs: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Physical slots of compressed rows in a paged TOKEN-group cache.

    The compact ``slot_mappings`` layout (one entry per newly committed row)
    packs sequences in commit order and therefore cannot be consumed inside a
    captured graph; the slots are recomputed from the persistent block table
    instead, with ``mask`` folding the commit decision into -1 (scatter skips)
    so no host branch remains. ``rows``/``seqs`` are flat per-row tensors; the
    -1 slots of uncommitted or padded rows are never written.
    """
    if block_table is None or cache is None:
        return torch.full_like(rows, -1)
    if block_table.numel() == 0:
        return torch.full_like(rows, -1)
    block_size = cache.size(1)
    cols = torch.div(rows, block_size, rounding_mode="floor") % block_table.size(1)
    blocks = block_table[seqs, cols].to(torch.int64).clamp_min(0)
    slots = blocks * block_size + rows % block_size
    if mask is not None:
        slots = torch.where(mask, slots, -1)
    return slots


# Serving-context cap applied before deriving the default graph token-capacity
# granularity. The model's ``max_position_embeddings`` (1048576 for V4.1) is the
# theoretical RoPE horizon, not any served context, and deriving from it yields
# an impractical 65536-token bucket whose indexer gather touches 65536 rows per
# layer; the dual-node launcher historically pinned 2048 to keep that gather
# bounded. Capping the derivation source at 32768 reproduces exactly that 2048
# floor (32768 -> ceil(32768 / 16) = 2048) while still scaling with shorter
# served contexts. An explicit, non-empty
# ``XLLM_V41_GRAPH_CAPACITY_GRANULARITY`` still wins over this derivation.
_GRAPH_TOKEN_CAPACITY_SERVING_CAP = 32768


def _derive_graph_token_capacity_granularity(max_model_len: int) -> int:
    """Default ACL-graph token-capacity granularity from the served context.

    The granularity is the token width of one captured capacity bucket
    (``_committed_capacity`` gathers ``granularity // ratio`` compressed rows
    per bucket). The derivation source is the served context length capped at
    :data:`_GRAPH_TOKEN_CAPACITY_SERVING_CAP` so a large
    ``max_position_embeddings`` cannot blow up the per-layer gather. Aligning
    ``ceil(serving_capped / 16)`` up to a 1024 multiple keeps the captured
    bucket count at roughly 16 across context lengths, while the 1024 floor
    avoids over-fragmenting short contexts. Returns 0 when the context length
    is unknown, keeping the fail-closed capture guard.
    """
    if max_model_len <= 0:
        return 0
    serving_capped = min(max_model_len, _GRAPH_TOKEN_CAPACITY_SERVING_CAP)
    scaled = (serving_capped + 15) // 16
    return max(1024, ((scaled + 1023) // 1024) * 1024)


class Csa2AttentionBackend(DsaAttentionBackend):
    """Pure-torch CSA2 backend for DeepSeek-V4.1 Flash.

    Consumes the V4.1 cache layout of contract section 2 through the standard
    ``LayerCache`` slots (key=0, index=2, swa=5, compress_kv_state=6,
    compress_score_state=7) and the ``DsaMetadataBuilder`` block tables / slot
    mappings. The AICPU precomputed-metadata step of the V4 backend is not
    needed (no NPU kernels are dispatched here).

    ACL graph integration: every data-dependent branch in the forward is
    tensor arithmetic (see :meth:`execute_csa2_layer`), so one captured graph
    stays correct across parity and length changes. The one remaining dynamic
    shape -- the committed-row gather of the indexer / sparse attention -- is
    pinned to a constant row capacity instead of a per-replay host read:
    ``graph_token_capacity_granularity`` is the largest decode context length
    the serving run can reach, and the captured graph gathers at
    ``granularity // ratio`` rows for every replay.

    The granularity is resolved in :meth:`__init__`: an explicit, non-empty
    ``XLLM_V41_GRAPH_CAPACITY_GRANULARITY`` wins; otherwise it is derived from
    ``max_model_len`` capped at :data:`_GRAPH_TOKEN_CAPACITY_SERVING_CAP` (0
    keeps the eager exact-max behaviour).

    MoE graph capability: only the Ascend W8A8 export routes the experts
    through the grouped kernel chain (MC2) an ACL graph can replay. The
    ``fp8``/``none`` checkpoint modes serve ``DeepseekV41MoE``, whose forward
    syncs to the host and loops over data-dependent expert segments, so no
    captured graph can replay it. :attr:`moe_graph_capturable` records that
    verdict (derived from the shared model config) for the decode-graph
    admission; :attr:`acl_graph_enabled` gates the startup capacity check.
    """

    def __init__(
        self,
        compress_ratios: list[int],
        window_size: int,
        n_layers: int,
        num_heads: int,
        attn_head_dim: int,
        index_topk: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        kv_source_layer_ids: list[int],
        index_source_layer_ids: list[int],
        candidate_source_layer_id: int,
        candidate_topk_blocks: int,
        candidate_block_size: int,
        max_model_len: int = 0,
        quant_mode: str = V41_QUANT_NONE,
        acl_graph_enabled: bool = False,
    ) -> None:
        self.caches_info, self.group_infos = build_cache_specs_v41(
            compress_ratios,
            kv_source_layer_ids,
            index_source_layer_ids,
            window_size,
            n_layers,
        )
        self._builder = DsaMetadataBuilder(self.caches_info, self.group_infos, read_table_layout=READ_TABLE_LAYOUT_RING)
        self.window_size = window_size
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.num_heads = num_heads
        self.head_dim = attn_head_dim
        self.device = device
        self.dtype = dtype
        self.scale = attn_head_dim**-0.5

        self.index_source_layer_ids = sorted({int(source) for source in index_source_layer_ids})
        self.candidate_source_layer_id = int(candidate_source_layer_id)
        self.candidate_topk_blocks = int(candidate_topk_blocks)
        self.candidate_block_size = int(candidate_block_size)
        if self.candidate_source_layer_id >= 0 and self.candidate_source_layer_id not in self.index_source_layer_ids:
            raise ValueError(
                "DeepSeek-V4.1 candidate source layer must also be an index source layer: "
                f"{self.candidate_source_layer_id}"
            )

        # MoE graph capability marker (see the class docstring): the fp8/none
        # torch MoE cannot be captured. Derived from the shared model config, so
        # every rank of a DP group computes the same value and the decode-graph
        # admission stays group-consistent.
        self.quant_mode = quant_mode
        self.moe_graph_capturable = quant_mode == V41_QUANT_ASCEND

        raw_granularity = os.environ.get("XLLM_V41_GRAPH_CAPACITY_GRANULARITY")
        if raw_granularity is not None and raw_granularity.strip() != "":
            self.graph_token_capacity_granularity = int(raw_granularity)
            granularity_source = "XLLM_V41_GRAPH_CAPACITY_GRANULARITY"
        else:
            # An unset OR empty env value means "not configured": the serving
            # entry is expected to derive the default from max_model_len (the
            # launcher exports the variable empty on purpose).
            self.graph_token_capacity_granularity = _derive_graph_token_capacity_granularity(max_model_len)
            granularity_source = f"derived from max_model_len={max_model_len}"
        logger.info(
            f"DeepSeek-V4.1 CSA2 graph_token_capacity_granularity="
            f"{self.graph_token_capacity_granularity} ({granularity_source})"
        )

        # Startup checks (51361671 C). The graph-vs-eager decision and the
        # committed-row capacity are both static for the run, so fail (or warn)
        # here at construction instead of at the first captured forward, where
        # the root cause is far from the symptom.
        if acl_graph_enabled and not self.moe_graph_capturable:
            logger.warning(
                f"DeepSeek-V4.1 quant_mode={quant_mode} serves the torch MoE, which no ACL graph "
                "can capture; decode-graph admission is disabled and decode stays eager"
            )
        if acl_graph_enabled and self.moe_graph_capturable and self.graph_token_capacity_granularity <= 0:
            raise RuntimeError(
                "DeepSeek-V4.1 ACL-graph serving needs a static committed-row capacity, but "
                "XLLM_V41_GRAPH_CAPACITY_GRANULARITY is unset/empty and max_model_len was not "
                "provided to derive one. Set the env var or provide max_position_embeddings, or "
                "disable graph execution."
            )

        self._kv_caches: list[Any] = []
        self._metadata: Any | None = None
        self._v41_shared: CSA2SharedRuntime | None = None

    # -- V4 backend overrides ----------------------------------------------

    def reset_forward(self, metadata: Any | None = None) -> None:
        super().reset_forward(metadata)
        self._v41_shared = None

    def _build_precomputed_metadata(
        self,
        compressed_metadata: DsaMetadata,
        metadata: Any,
        cu_seqlens_ori_kv_override: torch.Tensor | None = None,
        max_query_len_override: int | None = None,
        max_seq_len_override: int | None = None,
    ) -> None:
        """The pure-torch path dispatches no AICPU kernels; nothing to build."""
        del (
            compressed_metadata,
            metadata,
            cu_seqlens_ori_kv_override,
            max_query_len_override,
            max_seq_len_override,
        )

    def _resolve_cache_mapping(self, layer_id: int, compress_ratio: int) -> _CompressedAttentionCacheMapping:
        """V4.1 mapping: TOKEN entries exist for ratio-1 sources too."""
        mapping = _CompressedAttentionCacheMapping()
        if layer_id < 0 or layer_id >= len(self.caches_info):
            return mapping
        token_indices: list[int] = []
        swa_indices: list[int] = []
        for cache_idx, ci in enumerate(self.caches_info[layer_id]):
            if ci.cache_type == DSA_CACHE_TOKEN:
                token_indices.append(cache_idx)
            elif ci.cache_type == DSA_CACHE_SLIDING_WINDOW:
                swa_indices.append(cache_idx)
        if token_indices and compress_ratio > 0:
            mapping.cmp_cache_idx = token_indices[0]
        if len(token_indices) > 1:
            mapping.index_cache_idx = token_indices[1]
        if swa_indices:
            mapping.ori_cache_idx = swa_indices[0]
        if len(swa_indices) > 1:
            mapping.kv_state_cache_idx = swa_indices[1]
        if len(swa_indices) > 2:
            mapping.score_state_cache_idx = swa_indices[2]
        return mapping

    def _layer_compress_ratio(self, layer_id: int) -> int:
        caches = self.caches_info[layer_id] if 0 <= layer_id < len(self.caches_info) else []
        for ci in caches:
            if ci.cache_type == DSA_CACHE_TOKEN:
                return ci.ratio
        return 0

    def swa_group_metadata(self, dsa: DsaMetadata | None) -> _SwaGroupMetadata | None:
        """The SWA-group slot mapping / block table / cache shared by all layers.

        Every layer's first cache entry addresses the SWA group and the group
        shares one block-table/slot layout (contract section 2), so layer 0's
        view serves any consumer that needs position-stable ring slots (the
        engram hash cache).
        """
        if dsa is None or not self._kv_caches:
            return None
        mapping = self._resolve_cache_mapping(0, 0)
        layer_cache = self._kv_caches[0]
        if layer_cache.swa is None or mapping.ori_cache_idx < 0:
            return None
        slot_mapping = _get_layer_cache_tensor(dsa.slot_mappings, 0, mapping.ori_cache_idx)
        block_table = _get_layer_cache_tensor(dsa.block_tables, 0, mapping.ori_cache_idx)
        if slot_mapping is None or block_table is None:
            return None
        return _SwaGroupMetadata(
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=layer_cache.swa.size(1),
            cache=layer_cache.swa,
        )

    # -- model-facing entry point -------------------------------------------

    def execute_csa2_layer(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: Any,
    ) -> torch.Tensor:
        """Run one V4.1 layer (SWA-only or CSA2) end to end.

        ``q`` is ``[T, H, D]`` (RoPE applied at token positions) and ``kv`` is
        ``[T, D]`` (RoPE applied, FP8-QDQ'd) from the model's projections; the
        layer carries its CSA2 mode and the :class:`Csa2LayerContext`.

        ACL-graph safety: ``seq_lens_q`` is read as a Python list because its
        value is constant per captured graph (one row per sequence on the
        decode path), while ``seq_lens`` changes every step and is therefore
        consumed strictly as a device tensor (the persistent buffer refreshed
        by ``refresh_dsa_metadata_for_graph_replay``). Every data-dependent
        branch below is expressed as tensor arithmetic so one captured graph
        stays correct across parity / length changes.
        """
        metadata = self._current_forward_metadata()
        dsa = getattr(metadata, "dsa_metadata", None)
        if dsa is None:
            raise RuntimeError("DeepSeek-V4.1 requires prepared DSA metadata before attention")
        mode = layer.mode
        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        mapping = self._resolve_cache_mapping(layer_id, mode.compress_ratio)
        ctx = layer._csa2_ctx
        if ctx is None:
            raise RuntimeError(f"DeepSeek-V4.1 layer {layer_id} ran without its CSA2 context")
        if dsa.is_acl_graph:
            # Decode-graph contract: one token row per sequence. The host list
            # is a capture-stable constant, so never sync-read the device
            # tensor inside the captured stream (aclrtMemcpy on a captured
            # stream fails with 107030).
            q_lens = [1] * dsa.seq_lens_q.numel()
        else:
            q_lens = [int(value) for value in dsa.seq_lens_q.tolist()]
        ctx_lens = _ensure_seq_lens_tensor(dsa.seq_lens, ctx.x.device)

        # 1) SWA ring write: one slot per query token (ring by position).
        swa_slot = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, mapping.ori_cache_idx)
        if layer_cache.swa is not None and swa_slot is not None:
            _scatter_by_slot(layer_cache.swa, swa_slot, kv)

        shared = self._shared_runtime()
        # 2) Compressor: only kv-source (Full) layers produce the shared main KV.
        if mode.is_full:
            self._run_compressor(layer, ctx, dsa, layer_cache, mapping, q_lens, ctx_lens, shared)
        # 3) Indexer: Full/Reindex layers compute a fresh top-k; Reuse layers
        #    consume the tensor object their group source published.
        if mode.is_full or mode.is_reindex:
            topk = self._run_indexer(layer, ctx, dsa, q_lens, ctx_lens, shared)
        elif mode.is_reuse:
            topk = shared.topk_idxs
            if topk is None and dsa.block_tables:
                # Real batches always publish the group source's top-k;
                # profiling batches (empty block tables) degrade to the
                # plain window path.
                raise RuntimeError(
                    f"DeepSeek-V4.1 reuse layer {layer_id} has no shared top-k; its group source must run first"
                )
        else:
            topk = None
        layer._csa2_used_topk = topk
        # 4) Sparse attention over [window rows ; top-k compressed rows].
        return self._csa2_attention(q, kv, layer, dsa, q_lens, ctx_lens, topk, shared)

    # -- compressor ---------------------------------------------------------

    def _run_compressor(
        self,
        layer: Any,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata,
        layer_cache: Any,
        mapping: _CompressedAttentionCacheMapping,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        shared: CSA2SharedRuntime,
    ) -> None:
        """Compress the layer's own attention input into the shared main KV.

        Ratio 1 commits one row per token (plain ``RMSNorm(wkv(x))``); ratio 2
        pools each non-overlapping group of 2 with an fp32 softmax gate,
        parking partial-group tokens in the state ring (reference model.py:458-485,
        chunked-prefill generalization per the semantics doc section 4).

        ACL-graph safety: parking runs unconditionally (a completed group's
        members are never re-read from the ring, so the extra writes are
        harmless) and the commit uses a fixed ``q_len // ratio + 1`` row
        capacity per sequence with a tensor ``commit_mask``; member values are
        selected between the current chunk and the parked ring with
        ``torch.where`` instead of host branches, and every cache slot is
        computed from the persistent block tables as device arithmetic.
        """
        ratio = layer.mode.compress_ratio
        layer_id = layer.layer_id
        ctx_lens = _ensure_seq_lens_tensor(ctx_lens, ctx.x.device)
        swa_slot = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, mapping.ori_cache_idx)
        kv_state = layer_cache.compress_kv_state
        score_state = layer_cache.compress_score_state
        kv_state_flat = kv_state.view(-1, layer_cache.swa.size(-1)) if kv_state is not None else None
        score_state_flat = score_state.view(-1, layer_cache.swa.size(-1)) if score_state is not None else None

        latent_rows: list[torch.Tensor] = []
        group_starts: list[torch.Tensor] = []
        slot_rows: list[torch.Tensor] = []
        slot_seqs: list[torch.Tensor] = []
        slot_masks: list[torch.Tensor] = []
        token_offset = 0
        for seq, q_len in enumerate(q_lens):
            ctx_len = ctx_lens[seq]
            prev_ctx = ctx_len - q_len
            # Padded graph rows (ctx <= 0) stay fully inert: no parking, no
            # commit, so a garbage block-table entry can never be written.
            seq_valid = ctx_len > 0
            x_seq = ctx.x[token_offset : token_offset + q_len]
            rows = torch.arange(q_len, device=ctx.x.device)
            # Fixed commit capacity: a chunk of q_len tokens completes at most
            # q_len // ratio + 1 groups (the group straddling each boundary).
            capacity = q_len // ratio + 1
            g = torch.arange(capacity, device=ctx.x.device)
            group_idx = torch.div(prev_ctx, ratio, rounding_mode="floor") + g
            commit_mask = g < (
                torch.div(ctx_len, ratio, rounding_mode="floor") - torch.div(prev_ctx, ratio, rounding_mode="floor")
            )
            commit_mask = commit_mask & seq_valid
            if ratio == 1:
                # One token per group: nothing to pool, no gate, no state. The
                # commit count is exactly q_len (ctx - prev == q_len), so the
                # slot segment stays q_len rows and aligns 1:1 with the latent
                # rows in the concatenated scatter.
                kv_seq, _ = layer.compress_kv_score(x_seq)
                latent_rows.append(layer.compress_rmsnorm(kv_seq))
                group_starts.append(ctx.positions[token_offset : token_offset + q_len].to(torch.int64))
                rows_segment = group_idx[:q_len]
                mask_segment = commit_mask[:q_len]
            else:
                kv_seq, score_seq = layer.compress_kv_score(x_seq)
                # Park every token whose group does not complete this forward.
                # The parity branch is folded into the slot mask (-1 skips).
                pos = prev_ctx + rows
                park_mask = ((pos + 1) % ratio != 0) & seq_valid
                if swa_slot is not None and kv_state_flat is not None and score_state_flat is not None:
                    slot_seq = swa_slot[token_offset : token_offset + q_len]
                    park_slot = torch.where(park_mask, slot_seq, -1)
                    _scatter_by_slot(kv_state_flat, park_slot, kv_seq)
                    _scatter_by_slot(score_state_flat, park_slot, score_seq)
                # Pool the groups committed by this forward; a member inside
                # the chunk reads the fresh projection, one before the chunk
                # start reads its parked copy from the state ring.
                member_pos = group_idx.unsqueeze(-1) * ratio + torch.arange(ratio, device=ctx.x.device)
                # Trailing singleton so the [cap, ratio] mask right-aligns
                # with the [cap, ratio, dim] member tensors in torch.where.
                in_chunk = (member_pos >= prev_ctx).unsqueeze(-1)
                member_row = (member_pos - prev_ctx).clamp(0, q_len - 1)
                parked_slot = self._swa_position_slots(
                    dsa, layer_id, mapping.ori_cache_idx, seq, member_pos.clamp_min(0)
                )
                if kv_state_flat is not None and score_state_flat is not None:
                    member_kv = torch.where(in_chunk, kv_seq[member_row], kv_state_flat[parked_slot])
                    member_score = torch.where(in_chunk, score_seq[member_row], score_state_flat[parked_slot])
                else:
                    member_kv = kv_seq[member_row]
                    member_score = score_seq[member_row]
                # Softmax-gated pooling over the group members, one gate per
                # channel (wgate projects to the full latent width;
                # model.py:480-485).
                gate = member_score.softmax(dim=1)
                pooled = (member_kv * gate).sum(dim=1)
                # RMSNorm AFTER pooling, BEFORE RoPE (model.py:485).
                latent_rows.append(layer.compress_rmsnorm(pooled.to(ctx.x.dtype)))
                # Garbage rows (beyond this sequence's commit count) keep a
                # valid non-negative rope position; the slot mask keeps them
                # unwritten. The lower clamp also catches fully padded rows
                # (ctx_len <= 0 -> group_idx = -1), which stay inert via the
                # mask but still flow through apply_interleaved_rope.
                group_starts.append((group_idx * ratio).clamp(0, ctx.cos_sin_cache.size(0) - 1))
                rows_segment = group_idx
                mask_segment = commit_mask
            slot_rows.append(rows_segment)
            slot_seqs.append(torch.full_like(rows_segment, seq))
            slot_masks.append(mask_segment)
            token_offset += q_len

        latent = torch.cat(latent_rows, dim=0) if latent_rows else None
        all_starts = torch.cat(group_starts, dim=0) if group_starts else None
        all_slot_rows = torch.cat(slot_rows, dim=0) if slot_rows else None
        all_slot_seqs = torch.cat(slot_seqs, dim=0) if slot_seqs else None
        all_slot_mask = torch.cat(slot_masks, dim=0) if slot_masks else None
        index_block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.index_cache_idx)
        key_block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.cmp_cache_idx)
        # The indexer consumes the PRE-RoPE latent before the cache write.
        if latent is not None and layer.indexer is not None and layer.indexer.owns_k:
            index_k = layer.indexer_key(latent)
            index_k = apply_interleaved_rope(index_k, all_starts, ctx.cos_sin_cache)
            index_k = fp4_act_qdq_e8m0(index_k, 32)
            if layer_cache.index is not None:
                index_slots = _token_cache_slots(
                    index_block_table,
                    layer_cache.index,
                    all_slot_rows,
                    all_slot_seqs,
                    all_slot_mask,
                )
                _scatter_by_slot(layer_cache.index, index_slots, index_k)
        if latent is not None:
            rotated = apply_interleaved_rope(latent, all_starts, ctx.cos_sin_cache)
            rotated = fp4_act_qdq_e4m3_scale(rotated, 16)
            if layer_cache.key is not None:
                key_slots = _token_cache_slots(
                    key_block_table,
                    layer_cache.key,
                    all_slot_rows,
                    all_slot_seqs,
                    all_slot_mask,
                )
                _scatter_by_slot(layer_cache.key, key_slots, rotated)
        # Publish the shared caches for this group's Reindex/Reuse layers.
        shared.key_cache = layer_cache.key
        shared.key_block_table = key_block_table
        shared.index_cache = layer_cache.index
        shared.index_block_table = index_block_table
        shared.compress_ratio = ratio

    # -- indexer ------------------------------------------------------------

    def _run_indexer(
        self,
        layer: Any,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        shared: CSA2SharedRuntime,
    ) -> torch.Tensor:
        """Score the shared index K and select the top-k compressed rows.

        Full layers own their K (written by :meth:`_run_compressor`); Reindex
        layers re-score the group's shared K restricted to the candidate pool
        (reference model.py:527-580).

        ACL-graph safety: the score matrix is gathered at the batch-level
        committed-row capacity (the graph's static capacity bound under
        capture, the exact batch maximum eagerly) so every tensor shape is
        graph-stable; rows beyond a sequence's committed count are masked to
        ``-inf`` and can never enter a top-k pick.
        """
        layer_id = layer.layer_id
        ratio = shared.compress_ratio
        ctx_lens = _ensure_seq_lens_tensor(ctx_lens, ctx.x.device)
        if ratio <= 0 or shared.index_cache is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 indexer for layer {layer_id} has no shared index cache; "
                "its group kv source must run first"
            )
        if shared.index_block_table is None:
            # Profiling / warmup batches carry no block tables (see the
            # zeros ring rows in _csa2_attention); skip the scoring pass so
            # the sparse attention runs the plain window path. Reuse layers
            # degrade the same way via the empty dsa.block_tables check.
            return None
        index_cache = shared.index_cache
        block_table = shared.index_block_table
        block_size = index_cache.size(1)
        capacity = self._committed_capacity(dsa, ratio, block_table, block_size, ctx_lens)

        q_idx = layer.indexer_query(ctx.qr)
        q_idx = apply_interleaved_rope(q_idx, ctx.positions, ctx.cos_sin_cache)
        q_idx = fp4_act_qdq_e8m0(q_idx, 32)
        weights = layer.indexer_weights(ctx.x)

        candidates_per_seq: list[torch.Tensor | None] | None = None
        if layer_id == self.candidate_source_layer_id:
            candidates_per_seq = [None] * len(q_lens)
        rows_per_seq: list[torch.Tensor] = []
        token_offset = 0
        for seq, q_len in enumerate(q_lens):
            q_seq = q_idx[token_offset : token_offset + q_len]
            w_seq = weights[token_offset : token_offset + q_len]
            positions_seq = ctx.positions[token_offset : token_offset + q_len].to(torch.int64)
            ctx_len = ctx_lens[seq]
            committed = torch.div(ctx_len, ratio, rounding_mode="floor")
            if capacity <= 0:
                rows_per_seq.append(torch.full((q_len, 0), -1, dtype=torch.int32, device=q_idx.device))
                token_offset += q_len
                continue
            index_k = self._gather_cache_rows(index_cache, block_table, seq, capacity, block_size)
            # A compressed group is visible to query p once p has passed its
            # last token: visible(p) = (p + 1) // ratio (model.py:562-567);
            # rows past this sequence's committed count (or past the graph's
            # capacity bound) are padding and can never be picked.
            visible = torch.div(positions_seq + 1, ratio, rounding_mode="floor")
            t_range = torch.arange(capacity, device=q_idx.device)
            # The compressed picks address the concatenated
            # [window_kv ; compress_kv] tensor. The window is a fixed-capacity
            # [ring candidates ; chunk] block, so the compressed region starts
            # at a constant column and no ctx-length term remains.
            offset = self.window_size + q_len
            # Score in query tiles: the [tile, heads, capacity] fp32
            # intermediate is GiB-scale for whole-prompt chunks against long
            # contexts (see _INDEXER_QUERY_TILE). Top-k and the candidate
            # pool are per query row, so tiling is exact.
            row_tiles: list[torch.Tensor] = []
            pool_tiles: list[torch.Tensor] = []
            for tile_start in range(0, q_len, _INDEXER_QUERY_TILE):
                tile_end = min(tile_start + _INDEXER_QUERY_TILE, q_len)
                score = _indexer_score(q_seq[tile_start:tile_end], index_k, w_seq[tile_start:tile_end])
                tile_visible = visible[tile_start:tile_end]
                score.masked_fill_(t_range >= tile_visible.unsqueeze(-1), float("-inf"))
                score.masked_fill_(t_range >= committed.unsqueeze(-1), float("-inf"))
                if layer_id == self.candidate_source_layer_id:
                    assert candidates_per_seq is not None
                    pool_tiles.append(
                        _select_candidate_blocks(
                            score,
                            tile_visible,
                            self.candidate_topk_blocks,
                            self.candidate_block_size,
                        )
                    )
                elif self._uses_candidates(layer_id):
                    if shared.candidates is None or seq >= len(shared.candidates):
                        raise RuntimeError(
                            f"DeepSeek-V4.1 reindex layer {layer_id} requires the candidate pool "
                            f"from layer {self.candidate_source_layer_id}"
                        )
                    pool = shared.candidates[seq]
                    if pool is None:
                        raise RuntimeError(
                            f"DeepSeek-V4.1 reindex layer {layer_id} has no candidate pool for sequence {seq}"
                        )
                    score.masked_fill_(~pool[tile_start:tile_end], float("-inf"))
                row_tiles.append(_indexer_topk(score, tile_visible, self.index_topk, offset))
            if pool_tiles and candidates_per_seq is not None:
                candidates_per_seq[seq] = torch.cat(pool_tiles, dim=0)
            if row_tiles:
                rows_per_seq.append(row_tiles[0] if len(row_tiles) == 1 else torch.cat(row_tiles, dim=0))
            else:
                rows_per_seq.append(torch.full((q_len, 0), -1, dtype=torch.int32, device=q_idx.device))
            token_offset += q_len

        # One rectangular tensor (padded with -1) keeps the shared hand-off a
        # single object; padding columns are invalid and contribute nothing.
        seq_count = len(q_lens)
        s_max = max(q_lens) if q_lens else 0
        k_width = min(self.index_topk, capacity) if capacity > 0 else 0
        topk = torch.full((seq_count, s_max, k_width), -1, dtype=torch.int32, device=q_idx.device)
        for seq, rows in enumerate(rows_per_seq):
            if rows.numel() > 0:
                topk[seq, : rows.size(0), : rows.size(1)] = rows
        shared.topk_idxs = topk
        shared.topk_layer = layer_id
        if candidates_per_seq is not None:
            shared.candidates = candidates_per_seq
            shared.candidates_layer = layer_id
        return topk

    def _uses_candidates(self, layer_id: int) -> bool:
        return 0 <= self.candidate_source_layer_id < layer_id

    # -- sparse attention ----------------------------------------------------

    def _csa2_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: Any,
        dsa: DsaMetadata,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        topk: torch.Tensor | None,
        shared: CSA2SharedRuntime,
    ) -> torch.Tensor:
        """Gather ``[window rows ; top-k compressed rows]`` and run the sink softmax.

        Per (batch, query) the window covers positions ``max(0, p - win + 1)..p``
        at a fixed ``win + q_len`` row capacity (ring candidates masked per
        query to ``pos >= max(p - win + 1, 0)``, the current chunk appended
        after), the compressed rows
        come from the group's shared key cache at the committed capacity, and
        the attention sink contributes only to the softmax normalizer
        (kernel.py:310-389). All index math is tensor arithmetic, so the whole
        method replays correctly inside a captured ACL graph.
        """
        mode = layer.mode
        layer_id = layer.layer_id
        ratio = mode.compress_ratio
        win = self.window_size
        head_dim = q.size(-1)
        sinks = layer.attn_sink.to(torch.float32)
        ctx_lens = _ensure_seq_lens_tensor(ctx_lens, q.device)
        layer_cache = self._kv_caches[layer_id]
        mapping = self._resolve_cache_mapping(layer_id, ratio)
        swa_flat = layer_cache.swa.view(-1, head_dim) if layer_cache.swa is not None else None
        swa_bt = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.ori_cache_idx)
        swa_bs = layer_cache.swa.size(1) if layer_cache.swa is not None else win

        out = torch.empty_like(q)
        token_offset = 0
        for seq, q_len in enumerate(q_lens):
            ctx_len = ctx_lens[seq]
            prev_ctx = ctx_len - q_len
            q_seq = q[token_offset : token_offset + q_len]
            kv_seq = kv[token_offset : token_offset + q_len]
            # Window rows at fixed capacity: [win ring candidates ; chunk].
            # Ring candidate c sits at absolute position prev - win + c; which
            # candidates a given query may read is decided per query below.
            ring_pos = (prev_ctx - win) + torch.arange(win, device=q.device)
            if swa_flat is not None and swa_bt is not None:
                ring_rows = self._gather_swa_positions(swa_flat, swa_bt, seq, ring_pos.clamp_min(0), swa_bs)
            else:
                # Profiling / warmup batches carry no block tables: keep the
                # traced shapes intact with zeros; the mask below drops them.
                ring_rows = q.new_zeros((win, kv_seq.size(-1)))
            window_kv = torch.cat((ring_rows, kv_seq), dim=0)
            n_w = window_kv.size(0)
            # Window index matrix: query p (position prev+i) sees exactly the
            # window rows whose absolute position lies in
            # [max(p - win + 1, 0), p]; the lower bound tightens with the query
            # (only the chunk's first token shares the chunk-level bound).
            positions_seq = prev_ctx + torch.arange(q_len, device=q.device)
            col_pos = torch.cat((ring_pos, positions_seq), dim=0)
            cols = torch.arange(n_w, device=q.device)
            window_idxs = _window_visible_idxs(positions_seq, col_pos, cols, win)

            # Compressed rows for the top-k picks (shared group cache),
            # gathered at the committed capacity; padding rows are unreachable
            # because every top-k pick past `committed` is -1.
            capacity = 0
            if ratio > 0 and topk is not None and shared.key_cache is not None:
                capacity = self._committed_capacity(
                    dsa,
                    ratio,
                    shared.key_block_table,
                    shared.key_cache.size(1),
                    ctx_lens,
                )
            compress_kv = None
            if capacity > 0:
                shared_key_cache = shared.key_cache
                shared_key_table = shared.key_block_table
                assert shared_key_cache is not None and shared_key_table is not None
                compress_kv = self._gather_cache_rows(
                    shared_key_cache,
                    shared_key_table,
                    seq,
                    capacity,
                    shared_key_cache.size(1),
                )
            if compress_kv is not None:
                assert topk is not None
                kv_cat = torch.cat((window_kv, compress_kv), dim=0)
                idxs = torch.cat((window_idxs, topk[seq][:q_len]), dim=-1)
            else:
                kv_cat = window_kv
                idxs = window_idxs
            kv_cat_f = kv_cat.to(torch.float32)
            # Each query attends exactly the rows its index list selects
            # (window picks + compressed top-k picks), while the gathered
            # tensor holds every committed row: expand the per-query index
            # list into a row mask with static shapes (see _picked_row_mask)
            # so the block replays inside a captured ACL graph.
            row_mask = _picked_row_mask(idxs, kv_cat.size(0))

            for tile_start in range(0, q_len, _ATTENTION_QUERY_TILE):
                tile_end = min(tile_start + _ATTENTION_QUERY_TILE, q_len)
                scores = torch.einsum("shd,nd->shn", q_seq[tile_start:tile_end].to(torch.float32), kv_cat_f)
                scores = scores * self.scale
                scores = scores.masked_fill(~row_mask[tile_start:tile_end].unsqueeze(1), float("-inf"))
                row_max = scores.amax(dim=-1).clamp_min(_NEG_INF_FLOOR)
                probs = torch.exp(scores - row_max.unsqueeze(-1))
                # The sink adds exp(sink - row_max) to the normalizer only.
                sum_exp = probs.sum(dim=-1) + torch.exp(sinks - row_max)
                tile_out = torch.einsum("shn,nd->shd", probs, kv_cat_f)
                tile_out = tile_out / sum_exp.unsqueeze(-1)
                out[token_offset + tile_start : token_offset + tile_end] = tile_out.to(q.dtype)
            token_offset += q_len
        return out

    # -- cache gather helpers -------------------------------------------------

    def _committed_capacity(
        self,
        dsa: DsaMetadata | None,
        ratio: int,
        block_table: torch.Tensor | None,
        block_size: int,
        ctx_lens: torch.Tensor,
    ) -> int:
        """Compressed-row gather capacity (rows) for one forward.

        Under ACL graph capture this is a Python constant baked into the
        graph: the runner's per-bucket bound ``graph_token_capacity`` when the
        captured entry carries one (the decode-graph runner sets it to
        ``bucket * granularity``), otherwise ``graph_token_capacity_granularity
        // ratio`` -- so no replay ever sync-reads the device sequence lengths.
        Eagerly it is the exact batch maximum, read on the host where the sync
        is free.
        """
        if ratio <= 0:
            return 0
        if bool(getattr(dsa, "is_acl_graph", False)):
            bucketed_capacity = int(getattr(dsa, "graph_token_capacity", 0) or 0)
            if bucketed_capacity > 0:
                return max(bucketed_capacity // ratio, 1)
            if self.graph_token_capacity_granularity > 0:
                return max(self.graph_token_capacity_granularity // ratio, 1)
            raise RuntimeError(
                "DeepSeek-V4.1 ACL graph capture needs a static committed-row capacity, but the "
                "serving entry did not derive one (XLLM_V41_GRAPH_CAPACITY_GRANULARITY is unset and "
                "no max_model_len was provided). The engine should have derived a default at "
                "construction; a missing value means the deployment entry point passed no "
                "max_model_len."
            )
        if ctx_lens.numel() == 0:
            return 0
        return max(int(torch.div(ctx_lens.max(), ratio, rounding_mode="floor").item()), 0)

    def _gather_cache_rows(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        seq: int,
        count: int,
        block_size: int,
    ) -> torch.Tensor:
        """Gather logical rows ``0..count-1`` of one sequence from a paged cache.

        ``count`` may exceed the sequence's committed rows (capacity gather):
        the unassigned tail blocks clamp to block 0 and produce garbage rows
        the caller masks out, keeping the gather shape graph-stable.
        """
        rows = torch.arange(count, device=cache.device)
        cols = torch.div(rows, block_size, rounding_mode="floor") % block_table.size(1)
        blocks = block_table[seq, cols].to(torch.int64).clamp_min(0)
        slots = blocks * block_size + (rows % block_size)
        return cache.view(-1, cache.size(-1)).index_select(0, slots)

    def _gather_swa_positions(
        self,
        swa_flat: torch.Tensor,
        block_table: torch.Tensor,
        seq: int,
        positions: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Read ring rows by absolute position (SWA-group slot semantics).

        Unassigned blocks clamp to block 0: the resulting garbage rows are
        produced only for out-of-window / padded candidates whose mask keeps
        them out of every score, but the clamped gather never indexes below
        the cache (which would crash a captured graph).
        """
        cols = torch.div(positions, block_size, rounding_mode="floor") % block_table.size(1)
        blocks = block_table[seq, cols].to(torch.int64).clamp_min(0)
        slots = blocks * block_size + (positions % block_size)
        return swa_flat.index_select(0, slots)

    def _swa_position_slots(
        self,
        dsa: DsaMetadata,
        layer_id: int,
        swa_cache_idx: int,
        seq: int,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """SWA-group physical slots of absolute positions (contract section 2).

        Vectorized, graph-safe counterpart of the scalar lookup: unassigned
        blocks clamp to block 0 and yield garbage rows that the caller's
        commit mask keeps unwritten. Valid for the V4.1 layout where the SWA
        group's block size equals the window (one ring block per sequence),
        so every read-side block-table column resolves to the sequence's ring
        block.
        """
        block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, swa_cache_idx)
        swa = self._kv_caches[layer_id].swa
        block_size = swa.size(1) if swa is not None else self.window_size
        if block_table is None:
            # Profiling / warmup batches carry no block tables (see the zeros
            # ring rows in _csa2_attention); resolve every position to ring
            # block 0 so the traced shapes stay intact. Real batches always
            # ship the SWA block table.
            return positions % block_size
        cols = torch.div(positions, block_size, rounding_mode="floor") % block_table.size(1)
        blocks = block_table[seq, cols].to(torch.int64).clamp_min(0)
        return blocks * block_size + positions % block_size

    def _shared_runtime(self) -> CSA2SharedRuntime:
        if self._v41_shared is None:
            self._v41_shared = CSA2SharedRuntime()
        return self._v41_shared
