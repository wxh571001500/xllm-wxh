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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from xllm.python.attention.expanded_decode_metadata import (
    ExpandedDecodeMetadataLike,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention
    from xllm.python.model_executor.cp_utils import CpContext


@dataclass(frozen=True, slots=True)
class LayerCache:
    """Every cache a layer may own, named rather than positional.

    A layer holds a subset: full attention uses ``key``/``value``, the MLA
    sparse indexer adds ``index`` and optionally ``kpool_tail``, and a
    linear-attention layer uses ``conv``/``ssm`` instead of K/V. An absent
    slot is ``None`` rather than an empty tensor, so a layer that reads the
    wrong slot fails loudly.
    """

    key: torch.Tensor | None
    value: torch.Tensor | None
    index: torch.Tensor | None = None
    conv: torch.Tensor | None = None
    ssm: torch.Tensor | None = None
    # DeepSeek-V4 compressed-attention cache slots. Generic models leave these
    # as None; the tuple order is shared with PyExecutorImpl::bind_kv_caches.
    swa: torch.Tensor | None = None
    compress_kv_state: torch.Tensor | None = None
    compress_score_state: torch.Tensor | None = None
    compress_index_kv_state: torch.Tensor | None = None
    compress_index_score_state: torch.Tensor | None = None
    indexer_scale: torch.Tensor | None = None
    kpool_tail: torch.Tensor | None = None

    @property
    def index_scale(self) -> torch.Tensor | None:
        """Legacy MLA alias for the shared indexer scale cache."""
        return self.indexer_scale


#: Field order of the tuple form, which is what the C++ executor hands over.
_LAYER_CACHE_SLOTS = (
    "key",
    "value",
    "index",
    "conv",
    "ssm",
    "swa",
    "compress_kv_state",
    "compress_score_state",
    "compress_index_kv_state",
    "compress_index_score_state",
    "indexer_scale",
    "kpool_tail",
)

LayerCacheInput = LayerCache | tuple[torch.Tensor | None, ...]


def normalize_layer_caches(caches: Sequence[LayerCacheInput]) -> list[LayerCache]:
    """Accept the tuple form and return named caches with empty slots dropped."""
    normalized: list[LayerCache] = []
    for cache in caches:
        if isinstance(cache, LayerCache):
            normalized.append(cache)
            continue
        if not 2 <= len(cache) <= len(_LAYER_CACHE_SLOTS):
            raise ValueError(f"layer cache must hold between K/V and {'/'.join(_LAYER_CACHE_SLOTS)} tensors")
        slots = [None if tensor is None or not tensor.numel() else tensor for tensor in cache]
        slots.extend([None] * (len(_LAYER_CACHE_SLOTS) - len(slots)))
        normalized.append(LayerCache(*slots))
    return normalized


class AttentionMetadata(Protocol):
    slot_mapping: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    qo_indptr: torch.Tensor | None
    q_cu_seq_lens: torch.Tensor | None
    q_cu_seq_lens_host_values: list[int] | None
    kv_cu_seq_lens: torch.Tensor | None
    kv_seq_lens_host: torch.Tensor | None
    kv_seq_lens_host_values: list[int] | None
    new_cache_slots_host_values: list[int] | None
    q_seq_lens_host: torch.Tensor | None
    paged_kv_indptr_host: torch.Tensor | None
    paged_kv_last_page_len_host: torch.Tensor | None
    block_table: torch.Tensor | None
    kv_seq_lens: torch.Tensor | None
    max_query_len: int
    max_seq_len: int
    multi_block_tables: Sequence[torch.Tensor | None]
    # Legacy C++/pybind DSAMetadata field names retained for ABI compatibility.
    dsa_metadata: object | None
    dsa_positions: torch.Tensor | None
    dsa_cos_sin: torch.Tensor | None
    dsa_c4_cos_sin: torch.Tensor | None
    dsa_c128_cos_sin: torch.Tensor | None
    dsa_graph_block_table_cols: int
    dsa_graph_mode: bool
    linear_state_indices: torch.Tensor | None
    linear_state_read_indices: torch.Tensor | None
    linear_state_write_indices: torch.Tensor | None
    kpool_query_lens: Sequence[int]
    has_initial_state: torch.Tensor | None
    num_accepted_tokens: torch.Tensor | None
    dp_execution_token_counts: Sequence[int]
    raw_dp_execution_token_counts: Sequence[int]
    # Host-planned logical KV lengths for all DP ranks, including this query.
    # Empty DP ranks retain zero; this is not reserved block-table capacity.
    dp_global_kv_max_seq_lens: Sequence[int]
    mega_moe_token_mask: torch.Tensor | None
    dp_is_decode: Sequence[int]
    q_seq_lens: torch.Tensor | None
    expanded_decode_metadata: ExpandedDecodeMetadataLike
    is_prefill: bool
    is_chunked_prefill: bool
    is_mixed: bool
    is_spec_verify: bool
    is_dflash_proposal: bool
    is_dummy: bool
    local_slot_mapping: torch.Tensor | None
    kv_split_size: int
    kv_split_rank: int
    has_kv_shard: bool


def linear_state_checkpoint_stride(
    conv_cache: torch.Tensor,
    ssm_cache: torch.Tensor,
) -> int:
    """Return the number of SSM checkpoints owned by each linear slot."""
    if conv_cache.shape[0] <= 0 or ssm_cache.shape[0] % conv_cache.shape[0] != 0:
        raise RuntimeError("linear SSM cache does not align with state slots")
    return ssm_cache.shape[0] // conv_cache.shape[0]


def build_speculative_ssm_state_indices(
    state_indices: torch.Tensor,
    checkpoint_stride: int,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Map slots to checkpoints using dtype for offsets and the result."""
    checkpoint_offsets = torch.arange(
        checkpoint_stride,
        dtype=dtype,
        device=state_indices.device,
    )
    return state_indices.to(dtype).contiguous().view(-1, 1) * checkpoint_stride + checkpoint_offsets.view(1, -1)


def resolve_linear_state_io_indices(
    metadata: AttentionMetadata,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve linear-state read and live/write slots for one forward.

    These ids are deliberately **not** owner-localized when KV is owner-sharded
    (``kv_split_size > 1``): ``BlockType::LINEAR`` -- the KDA conv/ssm recurrent
    state and the kPool tail it also indexes -- is not a kv-split block type
    (``is_kv_split_cache_block_type``), so the state is replicated to every
    owner as one slot per sequence. Replication is exact rather than merely
    close, because the KDA layer merges to global rows with ``cp_merge_rows``
    before ``execute_linear``: every CP rank then runs the conv1d/delta-rule
    over the same global stream, with the same weights, in the same order, and
    writes identical values into its own replica of the same slot. Sharding
    would instead need a cross-owner reduction per step and a defined owner for
    the accumulator, neither of which exists.
    """
    write_indices = getattr(metadata, "linear_state_write_indices", None)
    if write_indices is None:
        write_indices = getattr(metadata, "linear_state_indices", None)
    read_indices = getattr(metadata, "linear_state_read_indices", None)
    if read_indices is None:
        return write_indices, write_indices
    if write_indices is None:
        raise RuntimeError("linear-state read indices require write indices")
    if read_indices.shape != write_indices.shape:
        raise RuntimeError(
            "linear-state read/write indices must have the same shape: "
            f"read={tuple(read_indices.shape)}, write={tuple(write_indices.shape)}"
        )
    is_prefill = metadata.is_prefill or metadata.is_chunked_prefill
    if not is_prefill or getattr(metadata, "is_spec_verify", False):
        if not torch.equal(read_indices, write_indices):
            raise RuntimeError("linear-state read/write separation is only supported for non-speculative prefill")
        # Canonicalize equal metadata to one object. Hot-path state users can
        # then skip an otherwise redundant snapshot and self-copy.
        return write_indices, write_indices
    return read_indices, write_indices


def _identity_block_table(block_table: torch.Tensor) -> torch.Tensor:
    """Full-replica pool page view: the local table already addresses the cache."""
    return block_table


@dataclass(frozen=True)
class MlaIndexContext:
    """Public contract handed to an optional LightningIndexer.

    Replaces direct model access to ``backend._metadata`` / ``backend._kv_caches``
    for MLA layers. The backend owns the paged index cache (``LayerCache.index``)
    and prepares the paging / sequence-length metadata once per step; the indexer
    receives this view and produces ``topk``. ``materialize_index_cache`` returns
    the cache, optional scale, and their single matching block table together.

    ``has_kv_shard`` marks a shard-aware index cache (``kv_split_size > 1``
    under CP): the latent cache is owner-sharded either way, while the INDEX
    cache's layout follows ``XLLM_CP_INDEX_WRITE_MODE``. In the default
    ``replicated`` mode every rank persists ALL ``kv_split_size`` pages of
    every logical block at their natural rows (``block_table[b] *
    kv_split_size + stripe``; the index tensor is full-size per rank on NPU),
    so ``materialize_index_cache`` is the identity returning the local cache
    with its physically expanded table and the pool write goes through
    ``localize_pool_block_table`` (the natural-row expansion). In ``sharded``
    mode each rank holds only its own KV owner stripe, stored at the first
    page of each logical block's index-page group (row ``block_table[b] *
    kv_split_size``) so the PD transfer plan's page-level overlap moves
    exactly the page this rank wrote; the pool write goes through
    ``localize_pool_block_table`` (owner-filtered), ``slot_mapping`` is remapped
    onto the same page grouping for paged index writes, and a pool read goes
    through ``materialize_index_cache``/``materialized_block_table`` (the
    cross-rank reconstruction). All of these default to the full-replica
    behaviour, which is every ``kv_split_size == 1`` launch.
    """

    index_cache: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor | None
    actual_seq_q: torch.Tensor
    actual_seq_kv: torch.Tensor
    index_cache_scale: torch.Tensor | None
    get_quant_indexer_metadata: Callable[[int, int, int, int], torch.Tensor]
    update_index_cache: Callable[[torch.Tensor, torch.Tensor | None], None]
    materialize_index_cache: Callable[
        [],
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor],
    ]
    # Optional model-side CP plan for packed local query segments.
    cp_context: CpContext | None = None
    # Request-owned raw K/gate rows used to complete pools across forwards.
    kpool_tail: torch.Tensor | None = None
    kpool_tail_read_indices: torch.Tensor | None = None
    kpool_tail_write_indices: torch.Tensor | None = None
    kpool_query_lens: Sequence[int] = ()
    kpool_query_lens_device: torch.Tensor | None = None
    kpool_cache_triton_compatible: bool = False
    # True only when the index cache really needs shard-aware views (an
    # owner-sharded cache in sharded mode, a physically full replica that
    # still reads through page-expanded tables in replicated mode); the
    # latent cache is owner-sharded in both modes.
    has_kv_shard: bool = False
    # Page table the index READ goes through: the gathered logical pool cache
    # (one page per logical block per owner) in sharded mode, the physically
    # expanded local table in replicated mode; None while the cache is a full
    # replica.
    materialized_block_table: torch.Tensor | None = None
    # Maps a pool block table onto this rank's pool write pages: the natural
    # rows (``entry * kv_split + stripe``) in replicated mode, the owner's
    # page at the first page of each logical block's index-page group with
    # peer columns invalid in sharded mode. The identity while the cache is a
    # full replica.
    localize_pool_block_table: Callable[[torch.Tensor], torch.Tensor] = _identity_block_table


@dataclass(frozen=True)
class MlaPreprocessContext:
    """Decode cache tensors consumed by fused MLA preprocessing."""

    kv_cache: torch.Tensor
    rope_cache: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass(frozen=True)
class CsaIndexContext:
    """Per-forward cache and metadata view consumed by the DSV4 indexer."""

    index_cache: torch.Tensor
    indexer_scale: torch.Tensor | None
    slot_mapping: torch.Tensor
    block_table: torch.Tensor | None
    cmp_block_table: torch.Tensor | None
    kv_state: torch.Tensor | None
    score_state: torch.Tensor | None
    kv_block_table: torch.Tensor | None
    score_block_table: torch.Tensor | None
    actual_seq_q: torch.Tensor
    actual_seq_kv: torch.Tensor
    start_pos: torch.Tensor | None
    qli_metadata: torch.Tensor | None


class AttentionBackend(ABC):
    def reset_forward(self, metadata: AttentionMetadata | None = None) -> None:
        """Reset request-owned state before a model attaches current inputs."""
        del metadata

    @property
    def is_mla(self) -> bool:
        """Whether this backend implements MLA execution."""
        return False

    @abstractmethod
    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        pass

    @abstractmethod
    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
    ) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def num_kv_blocks(self) -> int:
        pass

    @property
    @abstractmethod
    def page_size(self) -> int:
        pass

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent: torch.Tensor | None,
        k_pe: torch.Tensor | None,
        layer: Attention,
        topk: torch.Tensor | None = None,
        cache_is_preprocessed: bool = False,
    ) -> torch.Tensor:
        """Absorbed-MLA attention over paged latent (nope) + rope caches.

        Returns ``[T, H, kv_lora]``; caller bmm's ``W_UV``. When ``topk`` is
        provided, dispatches to the sparse SFA path driven by an optional
        LightningIndexer; otherwise a dense MLA path is requested. Backends
        that do not implement MLA raise.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support MLA")

    def mla_preprocess_context(
        self,
        layer: Attention,
    ) -> MlaPreprocessContext | None:
        """Return decode cache tensors for a fused preprocessing region."""
        del layer
        return None

    def execute_linear(
        self,
        mixed_qkv: torch.Tensor,
        beta_raw: torch.Tensor,
        layer: Attention,
        raw_gate_proj: torch.Tensor,
    ) -> torch.Tensor:
        """KDA linear attention (conv1d + delta-rule) over framework state.

        ``mixed_qkv`` is ``[B, 3*qkv_dim, S]`` (q/k/v concatenated, the local
        head-subset already sharded). ``raw_gate_proj`` is the pre-gate forget
        projection ``f_b(f_a(x))`` shaped ``[B, S, num_heads_local, head_dim]``;
        the plain decode/prefill kernels fuse the safe-gate from it in-kernel,
        while unsupported safe-gate configurations materialize the gate on
        demand. ``beta_raw`` is the pre-sigmoid
        ``[B, S, num_heads_local]`` projection. Returns the core attention
        output ``[B, S, num_heads_local, head_dim]`` for the caller to gate +
        project.

        The backend owns the per-layer conv/ssm state (the conv/ssm slots of
        ``LayerCache``) and reads/advances/writes it via the
        ``linear_state_indices`` / ``has_initial_state`` metadata view.
        Backends that do not implement linear attention raise.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support linear attention (KDA)")

    def mla_index_context(self, layer: Attention) -> MlaIndexContext:
        """Public hook for an optional LightningIndexer.

        Hands out the paged index cache view (``LayerCache.index``) plus the
        paging / sequence-length metadata the indexer needs, so the model never
        touches ``backend._metadata`` / ``backend._kv_caches`` directly.
        Backends that do not support the sparse MLA indexer raise.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support MLA indexer")
