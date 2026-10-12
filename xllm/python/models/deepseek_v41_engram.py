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

"""DeepSeek-V4.1 Engram: n-gram hash lookups gated into the mHC residual stream.

Port of the reference ``inference/engram.py`` (vLLM
``vllm/models/deepseek_v4_1/common/engram.py`` is the upstream mirror).
Engram modules live on the backbone layers listed in ``engram_layer_ids``
only; the gate is a normalized dot product of the residual stream against a
per-hc-copy key, signed-sqrt'ed before the sigmoid:

    out[t, j] = h[t, j] + sigmoid(signed_sqrt(|dot|)) * value[t]

Three pieces of derived state are NOT in the checkpoint and are rebuilt at
init (deterministically, exactly as the reference does):

* ``token_map``: token id -> compressed vocab id (normalization-folded).
  Loaded from the ``engram_hash_state.bin`` sidecar
  (:func:`build_engram_hash_state` builds it) or injected for tests.
* ``primes`` / ``offsets``: per-(layer, n-gram size, head) prime-sized
  bucket ranges, drawn in order from ``engram_vocab_size - 1`` upward and
  never reused across layers.
* ``multipliers``: one odd int64 per (layer, lookback shift) from the
  per-layer RNG seeded ``10007 * layer_id`` (numpy PCG64, matching the
  reference bit-for-bit).

The hash computation keeps a slot-keyed rolling store (``hash_cache``) of
compressed ids, one entry per SWA KV slot of the first engram layer: slots
are stable per (request, position), prefix-cache hits reuse both the
physical blocks and the identical token ids, and spec-decode rollbacks
rewrite the same slots, so lookbacks read back exactly what the owning
request wrote. The runner may pass ``lookback_token_ids`` (ids just before
each request's chunk start), which take precedence over the slot cache.

Everything is plain torch with static shapes and no host synchronization,
so the whole module is ACL-graph safe.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import weakref
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Cache value for tokens that take no part in an n-gram (image spans).
ENGRAM_DEAD_ID = -1
# Image sentinel / padding token ids that break an n-gram (mm_preprocess.py).
ENGRAM_IMAGE_SENTINEL_BASE_ID = 129264
ENGRAM_IMAGE_PAD_ID = 129265
# Signed-sqrt clamp of the gate input (reference kernel clamp_value).
_ENGRAM_GATE_CLAMP = 1e-6

_WHITESPACE_RE = re.compile(r"[ \t\r\n]+")
# Private-use char so a token that is exactly one space survives Strip().
_SENTINEL = "\ue000"


def image_sentinel_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """True where the token is an image sentinel and must break every n-gram."""
    ids = input_ids.reshape(-1)
    return (ids == ENGRAM_IMAGE_SENTINEL_BASE_ID) | (ids == ENGRAM_IMAGE_PAD_ID)


# ---------------------------------------------------------------------------
# Prime bucket layout
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2**32 (same witnesses as vLLM)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above ``start`` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


class EngramLayout:
    """Bucket layout of the per-layer n-gram hash tables.

    A position is hashed as ``max_ngram_size - 1`` n-grams (2-gram .. max),
    each split over ``n_heads`` heads. Every (n-gram size, head) pair owns a
    prime-sized bucket range in the layer's table; the primes are drawn in
    order and never reused, keeping the ranges disjoint.
    """

    def __init__(
        self,
        layer_ids: Sequence[int],
        num_embeddings: Sequence[int],
        max_ngram_size: int,
        n_heads: int,
        head_dim: int,
        vocab_size: int,
        compressed_vocab_size: int,
        pad_token_id: int,
    ) -> None:
        self.layer_ids: tuple[int, ...] = tuple(int(i) for i in layer_ids)
        self.num_embeddings: tuple[int, ...] = tuple(int(n) for n in num_embeddings)
        self.max_ngram_size = int(max_ngram_size)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.vocab_size = int(vocab_size)
        self.compressed_vocab_size = int(compressed_vocab_size)
        self.pad_token_id = int(pad_token_id)
        if len(self.layer_ids) != len(self.num_embeddings):
            raise ValueError(
                f"engram_layer_ids {self.layer_ids} and engram_num_embeddings "
                f"{self.num_embeddings} must have the same length"
            )
        if self.max_ngram_size < 2:
            raise ValueError(f"engram_max_ngram_size must be >= 2, got {max_ngram_size}")

        primes: list[list[tuple[int, ...]]] = []
        seen: set[int] = set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes = []
                current = self.vocab_size - 1
                for _ in range(self.n_heads):
                    current = _find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(per_ngram)
        self.primes: tuple[tuple[tuple[int, ...], ...], ...] = tuple(tuple(p) for p in primes)
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads
        for layer, layer_primes in enumerate(self.primes):
            flat = [size for per_ngram in layer_primes for size in per_ngram]
            total = sum(flat)
            if total > self.num_embeddings[layer]:
                raise ValueError(
                    f"engram layer {self.layer_ids[layer]} prime buckets sum to {total} rows "
                    f"but engram_num_embeddings only provides {self.num_embeddings[layer]}"
                )
        offsets = []
        for layer_primes in primes:
            flat = [size for per_ngram in layer_primes for size in per_ngram]
            prefix = [0]
            for size in flat[:-1]:
                prefix.append(prefix[-1] + size)
            offsets.append(prefix)
        self.offsets = torch.tensor(offsets, dtype=torch.int64)

    def head_sizes(self, layer_hash_index: int) -> tuple[int, ...]:
        return tuple(size for per_ngram in self.primes[layer_hash_index] for size in per_ngram)


def compute_hash_multipliers(
    layer_ids: Sequence[int],
    max_ngram_size: int,
    compressed_vocab_size: int,
) -> torch.Tensor:
    """One odd int64 multiplier per (layer, lookback shift).

    Must match the reference bit-for-bit: the multipliers were drawn with
    ``numpy.random.default_rng(10007 * layer_id).integers(0, bound, size=(n,))``
    at training time, so the exact PCG64 stream is part of the checkpoint
    contract. Values are doubled-plus-one (odd) and bounded so that
    ``compressed_id * multiplier`` cannot overflow int64.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Building DeepSeek-V4.1 engram hash multipliers requires numpy (the "
            "reference RNG is numpy PCG64 and must be reproduced bit-exactly). "
            "Install numpy, or pre-generate an 'engram_hash_state.bin' sidecar "
            "with build_engram_hash_state() and place it next to "
            "the checkpoint."
        ) from exc
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1, dtype=torch.int64))
    return torch.stack(rows)


# ---------------------------------------------------------------------------
# Token map (tokenizer-derived, sidecar-loaded)
# ---------------------------------------------------------------------------


def normalize_token_text(text: str) -> str:
    """The reference normalization chain (engram.py ``build_compressed_token_map``).

    NFKC -> NFD -> strip accents -> lowercase -> fold whitespace runs to one
    space -> protect a lone space with a sentinel -> strip -> restore the
    space. Tokens that normalize alike collapse onto one compressed id.
    """
    s = unicodedata.normalize("NFKC", text)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = _WHITESPACE_RE.sub(" ", s)
    if s == " ":
        s = _SENTINEL
    s = s.strip()
    return s.replace(_SENTINEL, " ")


def build_compressed_token_map_from_texts(
    token_texts: Sequence[str],
    id_to_token: Callable[[int], str] | None = None,
) -> tuple[list[int], int]:
    """Fold token texts onto compressed ids (first-seen order).

    ``token_texts[i]`` is the decoded text of token id ``i``. Tokens whose
    decoded text contains U+FFFD (partial UTF-8 byte tokens) are keyed by
    their raw vocabulary form via ``id_to_token``.
    """
    key_to_new: dict[str, int] = {}
    lookup: list[int] = []
    for token_id, text in enumerate(token_texts):
        if "\ufffd" in text:
            key = id_to_token(token_id) if id_to_token is not None else text
        else:
            normalized = normalize_token_text(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup.append(new_id)
    return lookup, len(key_to_new)


def _load_engram_hash_state_sidecar(model_path: str) -> dict[str, torch.Tensor] | None:
    """Load the optional ``engram_hash_state.bin`` sidecar.

    Holds the tokenizer-derived state (``token_map`` int32 [vocab],
    ``multipliers`` int64 [n_layers, max_ngram]) so serving environments
    without numpy/tokenizers stay self-contained.
    """
    import os

    sidecar = os.path.join(model_path, "engram_hash_state.bin")
    if not os.path.isfile(sidecar):
        return None
    try:
        return torch.load(sidecar, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"failed to load engram hash-state sidecar {sidecar}: {exc}") from exc


def build_engram_hash_state(
    model_path: str,
    tokenizer_path: str | None = None,
    output_path: str | None = None,
) -> str:
    """Generate the ``engram_hash_state.bin`` sidecar next to a checkpoint.

    Writes the tokenizer-derived state :class:`EngramHashState` needs but that
    is not stored in the checkpoint: the compressed token map (every token id
    decoded and folded through :func:`normalize_token_text`) and the numpy
    PCG64 hash multipliers (seed ``10007 * layer_id``). Both are rebuildable
    deterministically from ``config.json`` + ``tokenizer.json``, but the
    multipliers reproduce a specific RNG stream, so they are baked once here
    instead of rebuilt on every model load (the serving path only needs
    ``torch``, not ``numpy``/``tokenizers``).

    Requires the optional ``tokenizers`` and ``numpy`` packages. Refuses to
    write a sidecar whose compressed count diverges from the config's
    ``engram_compressed_vocab_size``: every hash multiplier derives from that
    count, so a mismatch would silently rehash the engram tables.

    Returns the path written.
    """
    model_dir = os.path.abspath(model_path)
    if not os.path.isdir(model_dir):
        model_dir = os.path.dirname(model_dir)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        top = json.load(handle)
    config = {**top, **top.get("text_config", {})}

    layer_ids = config.get("engram_layer_ids") or []
    if not layer_ids:
        raise ValueError("config.json has no engram_layer_ids; this model does not use engram")
    vocab_size = int(config["vocab_size"])
    max_ngram_size = int(config["engram_max_ngram_size"])
    compressed_vocab = int(config["engram_compressed_vocab_size"])
    pad_token_id = int(config["engram_pad_token_id"])
    if not 0 <= pad_token_id < vocab_size:
        raise ValueError(f"engram_pad_token_id {pad_token_id} is out of vocab range {vocab_size}")

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "building the engram token map requires the 'tokenizers' package (pip install tokenizers)"
        ) from exc
    tokenizer_file = tokenizer_path or os.path.join(model_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_file):
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_file}")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if tokenizer_vocab != vocab_size:
        raise ValueError(f"tokenizer vocab size {tokenizer_vocab} does not match config vocab_size {vocab_size}")

    token_texts = tokenizer.decode_batch([[token_id] for token_id in range(vocab_size)])
    lookup, compressed = build_compressed_token_map_from_texts(token_texts, id_to_token=tokenizer.id_to_token)
    if compressed != compressed_vocab:
        raise ValueError(
            f"the tokenizer folds to {compressed} compressed ids but the config declares "
            f"engram_compressed_vocab_size={compressed_vocab}; every hash multiplier derives "
            "from this count, so the engram tables would be silently rehashed -- refusing to "
            "write the sidecar"
        )

    multipliers = compute_hash_multipliers(layer_ids, max_ngram_size, compressed_vocab)
    output = output_path or os.path.join(model_dir, "engram_hash_state.bin")
    torch.save(
        {
            "token_map": torch.tensor(lookup, dtype=torch.int32),
            "multipliers": multipliers,
        },
        output,
    )
    return output


# ---------------------------------------------------------------------------
# Hash state (per model)
# ---------------------------------------------------------------------------


class EngramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there.

    Stateless pieces (``token_map`` / ``primes`` / ``offsets`` /
    ``multipliers``) are built once at init; the slot-keyed ``hash_cache``
    is sized lazily from the bound SWA KV cache and is discarded on cache
    rebind (graph memory profiling swaps in a temporary, smaller cache).
    """

    def __init__(
        self,
        layout: EngramLayout,
        token_map: torch.Tensor,
        multipliers: torch.Tensor,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.lookback_depth = layout.max_ngram_size - 1
        self.pad_id = int(token_map[layout.pad_token_id].item())
        self.register_buffer("token_map", token_map.to(torch.int64), persistent=False)
        self.register_buffer(
            "primes",
            torch.tensor(layout.primes, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("offsets", layout.offsets.to(torch.int64), persistent=False)
        self.register_buffer("multipliers", multipliers.to(torch.int64), persistent=False)
        self._cache: torch.Tensor | None = None
        self._kv_cache_ref: weakref.ReferenceType[torch.Tensor] | None = None

    @classmethod
    def from_config(cls, layout: EngramLayout, config: dict) -> EngramHashState:
        """Build from the model config dict, resolving the token map source.

        Order: explicit ``engram_token_map`` injection (tests) -> the
        ``engram_hash_state.bin`` sidecar next to the checkpoint -> a hard
        error telling the operator to generate the sidecar.
        """
        injected = config.get("engram_token_map")
        if injected is not None:
            token_map = torch.as_tensor(injected, dtype=torch.int64)
            multipliers = compute_hash_multipliers(
                layout.layer_ids, layout.max_ngram_size, layout.compressed_vocab_size
            )
        else:
            sidecar = _load_engram_hash_state_sidecar(str(config.get("model_path") or ""))
            if sidecar is None:
                raise RuntimeError(
                    "DeepSeek-V4.1 engram requires the tokenizer-derived hash state. "
                    "Call build_engram_hash_state(model_path) against the "
                    "checkpoint's tokenizer to write 'engram_hash_state.bin' (or pass "
                    "'engram_token_map' in the config for tests)."
                )
            token_map = sidecar["token_map"]
            multipliers = sidecar["multipliers"]
        # layout.vocab_size is the engram hash-bucket space (engram_vocab_size,
        # e.g. 16M), NOT the tokenizer vocab; validate the map against the
        # model's vocab_size when the caller provides it.
        expected_vocab = int(config.get("vocab_size") or token_map.numel())
        if token_map.numel() != expected_vocab:
            raise ValueError(
                f"engram token map covers {token_map.numel()} ids but model vocab_size is {expected_vocab}"
            )
        compressed = int(token_map.max().item()) + 1
        if compressed != layout.compressed_vocab_size:
            raise ValueError(
                f"engram token map compresses to {compressed} ids but config expects "
                f"{layout.compressed_vocab_size}; every hash multiplier derives from "
                "it, so the engram tables would be silently rehashed"
            )
        return cls(layout, token_map, multipliers)

    def ensure_cache(self, swa_cache: torch.Tensor, block_size: int) -> bool:
        """Lazily size the slot cache from the bound SWA KV cache.

        Returns False while the KV cache is unbound (profile run); the
        caller skips engram hashing then.
        """
        if swa_cache is None or swa_cache.numel() == 0:
            self._cache = None
            self._kv_cache_ref = None
            return False
        if self._kv_cache_ref is not None and self._kv_cache_ref() is swa_cache:
            return True
        flat_rows = swa_cache.numel() // swa_cache.size(-1) if swa_cache.dim() > 1 else swa_cache.numel()
        self._cache = torch.zeros(flat_rows, dtype=torch.int32, device=swa_cache.device)
        self._kv_cache_ref = weakref.ref(swa_cache)
        del block_size
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
        swa_block_size: int,
    ) -> torch.Tensor:
        """Compute ``[T, n_engram_layers, n_hash_cols]`` int32 n-gram hashes.

        History resolution per (token, shift): the current chunk first, then
        ``lookback_token_ids`` (runner-provided, -1 = unknown), then the
        slot cache. Once a lookback is out of range or dead, every further
        shift pools ``pad_id`` into the rolling hash (the pad rows are
        trained, not skipped).
        """
        num_tokens = input_ids.shape[0]
        num_layers = len(self.layout.layer_ids)
        n_heads = self.layout.n_heads
        max_ngram = self.layout.max_ngram_size
        device = input_ids.device
        output = torch.zeros(num_tokens, num_layers, self.layout.n_hash_cols, dtype=torch.int32, device=device)
        if num_tokens == 0:
            return output
        if slot_mapping is not None and self._cache is not None:
            self._write_hash_cache(input_ids, dead_mask, slot_mapping)

        pos = positions.reshape(-1).to(torch.int64)
        ids = input_ids.reshape(-1).to(torch.int64)
        req = self._sequence_index(pos, query_start_loc)
        chunk_start = pos[query_start_loc.to(torch.int64)[req].clamp(0, num_tokens - 1)]

        comp = self.token_map[ids.clamp(0, self.token_map.numel() - 1)]
        comp = torch.where(dead_mask.reshape(-1), ENGRAM_DEAD_ID, comp).to(torch.int64)

        rows = torch.arange(num_tokens, device=device)
        rolling = torch.zeros(num_tokens, num_layers, dtype=torch.int64, device=device)
        blocked = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        cache = self._cache
        for shift in range(max_ngram):
            lookback = pos - shift
            in_batch = lookback >= chunk_start
            source = self._chunk_source(comp, rows, shift, in_batch, num_tokens)
            # ``known`` marks positions whose explicit history resolved to a
            # real token id; the slot cache must not overwrite them. None means
            # no explicit history was passed, so the cache stays the only
            # history source (unchanged behavior).
            known = None
            if lookback_token_ids is not None:
                source, known = self._lookback_source(
                    source, comp, req, lookback, in_batch, chunk_start, lookback_token_ids
                )
            if cache is not None and block_table is not None:
                source = self._cache_source(source, req, lookback, in_batch, known, block_table, swa_block_size)
            blocked = blocked | (lookback < 0) | (source == ENGRAM_DEAD_ID)
            value = torch.where(blocked, self.pad_id, source)
            rolling ^= value.unsqueeze(1) * self.multipliers[:, shift].unsqueeze(0)
            if shift > 0:
                lo = (shift - 1) * n_heads
                prime = self.primes[:, shift - 1, :]  # [L, n_heads]
                offset = self.offsets[:, lo : lo + n_heads]  # [L, n_heads]
                hashed = rolling.unsqueeze(-1) % prime.unsqueeze(0) + offset.unsqueeze(0)
                output[:, :, lo : lo + n_heads] = hashed.to(torch.int32)
        return output

    def _write_hash_cache(
        self,
        input_ids: torch.Tensor,
        dead_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        assert self._cache is not None
        comp = self.token_map[input_ids.reshape(-1).to(torch.int64)]
        comp = torch.where(dead_mask.reshape(-1), ENGRAM_DEAD_ID, comp).to(torch.int32)
        slots = slot_mapping.reshape(-1).to(torch.int64)
        valid = slots >= 0
        safe = torch.where(valid, slots, torch.zeros_like(slots))
        keep = self._cache.index_select(0, safe)
        self._cache.scatter_(0, safe, torch.where(valid, comp, keep))

    def _sequence_index(
        self,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token request index via the cumulative chunk boundaries."""
        num_tokens = positions.shape[0]
        qsl = query_start_loc.to(torch.int64).reshape(-1)
        ends = qsl[1:].contiguous()
        req = torch.searchsorted(ends, torch.arange(num_tokens, device=positions.device), right=True)
        return req.clamp(0, max(qsl.numel() - 2, 0))

    def _chunk_source(
        self,
        comp: torch.Tensor,
        rows: torch.Tensor,
        shift: int,
        in_batch: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Compressed id of ``rows - shift`` when it stays inside the chunk."""
        source_row = (rows - shift).clamp_min(0).clamp_max(num_tokens - 1)
        source = comp.index_select(0, source_row)
        return torch.where(in_batch, source, torch.zeros_like(source))

    def _lookback_source(
        self,
        source: torch.Tensor,
        comp: torch.Tensor,
        req: torch.Tensor,
        lookback: torch.Tensor,
        in_batch: torch.Tensor,
        chunk_start: torch.Tensor,
        lookback_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit-history id where the runner provided one, plus its mask.

        Returns ``(source, known)``: ``known`` is True exactly where the
        position is outside the current chunk and the runner supplied a real
        token id (``>= 0``). Callers use ``known`` to keep explicit history
        ahead of the slot cache.
        """
        depth = lookback_token_ids.shape[-1]
        col = chunk_start - 1 - lookback
        in_window = (~in_batch) & (col >= 0) & (col < depth)
        col_safe = col.clamp(0, depth - 1)
        req_safe = req.clamp(0, lookback_token_ids.shape[0] - 1)
        token = lookback_token_ids.to(torch.int64)[req_safe, col_safe]
        known = in_window & (token >= 0)
        token_safe = token.clamp_min(0)
        mapped = self.token_map[token_safe.clamp(0, self.token_map.numel() - 1)]
        dead = (token_safe == ENGRAM_IMAGE_SENTINEL_BASE_ID) | (token_safe == ENGRAM_IMAGE_PAD_ID)
        mapped = torch.where(dead, ENGRAM_DEAD_ID, mapped)
        return torch.where(known, mapped, source), known

    def _cache_source(
        self,
        source: torch.Tensor,
        req: torch.Tensor,
        lookback: torch.Tensor,
        in_batch: torch.Tensor,
        known: torch.Tensor | None,
        block_table: torch.Tensor,
        swa_block_size: int,
    ) -> torch.Tensor:
        """Slot-cache id, applied only where neither the chunk nor explicit history resolved.

        ``known`` is the ``_lookback_source`` mask (None when no explicit
        history was passed) and takes precedence over the cache, matching the
        documented resolution order: chunk -> explicit lookback -> slot cache.
        """
        assert self._cache is not None
        table_cols = block_table.shape[1]
        clamped = lookback.clamp(0, table_cols * swa_block_size - 1)
        col = torch.div(clamped, swa_block_size, rounding_mode="floor") % table_cols
        req_safe = req.clamp(0, block_table.shape[0] - 1)
        block = block_table.to(torch.int64)[req_safe, col]
        slot = (block * swa_block_size + clamped % swa_block_size).clamp(0, self._cache.numel() - 1)
        fallback = self._cache.index_select(0, slot).to(torch.int64)
        resolved = in_batch if known is None else (in_batch | known)
        return torch.where(resolved, source, fallback)


# ---------------------------------------------------------------------------
# Hash table embedding
# ---------------------------------------------------------------------------


# Table storage layouts: the official FP8 checkpoint keeps FP8-E4M3 payloads
# plus ue8m0 per-32 scales as raw bytes; the Eco-Tech ascend (W8A8) export
# requantizes the tables to symmetric int8 with float32 group-32 scales
# (quant_model_description.json optional.embedding_storage, format int8_sym).
ENGRAM_STORAGE_FP8 = "fp8"
ENGRAM_STORAGE_INT8 = "int8"


def load_quarot_gate_unrotate(model_path: str, device: torch.device) -> torch.Tensor | None:
    """The QuaRot global rotation transpose used to un-rotate the gate input.

    The Eco-Tech W8A8 export rotates the residual stream (h_rot = h @ Q) and
    folds the rotation into every trunk weight, so the stream reaches the
    engram gate in the rotated basis while q/k/wkv-K stay in the original
    basis. The gate dot therefore needs ``h_rot @ Q.T``; the V half of ``wkv``
    is pre-folded (Q.T @ Wv) and must NOT be rotated again. Returns ``Q.T``
    ([hidden, hidden] fp32) or None when the checkpoint is not rotated. The
    safetensors payload may be saved as F64/F32/F16/BF16; the dtype is read
    from the file header and the result is converted to fp32 on return.
    """
    import json
    import os
    import struct

    config_path = os.path.join(str(model_path), "config.json")
    try:
        with open(config_path, encoding="utf-8") as handle:
            rotation = json.load(handle).get("engram_rotation_config") or {}
    except OSError:
        return None
    if not rotation.get("value_projection_rotated"):
        return None
    quarot_path = os.path.join(str(model_path), "optional", "quarot.safetensors")
    try:
        from safetensors.torch import load_file

        q = load_file(quarot_path, device="cpu")["global_rotation"]
    except ImportError:
        # Manual header parse when the safetensors package is unavailable;
        # real load failures from load_file must propagate.
        dtype_map = {
            "F64": (torch.float64, 8),
            "F32": (torch.float32, 4),
            "F16": (torch.float16, 2),
            "BF16": (torch.bfloat16, 2),
        }
        with open(quarot_path, "rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        meta = header["global_rotation"]
        if meta["dtype"] not in dtype_map:
            raise ValueError(f"unsupported quarot global_rotation dtype: {meta['dtype']!r}")
        q_dtype, itemsize = dtype_map[meta["dtype"]]
        offset = 8 + header_len + meta["data_offsets"][0]
        numel = 1
        for dim in meta["shape"]:
            numel *= dim
        with open(quarot_path, "rb") as handle:
            handle.seek(offset)
            raw = handle.read(numel * itemsize)
        q = torch.frombuffer(bytearray(raw), dtype=q_dtype).reshape(meta["shape"])
    return q.t().contiguous().to(device=device, dtype=torch.float32)


class EngramEmbedding(nn.Module):
    """The n-gram hash table, sharded by complete hash heads over TP ranks.

    Rows stay quantized in memory and are dequantized to bf16 on lookup (a
    3.8e8-row table must not be materialized in bf16). TP ranks own
    contiguous head-bucket row ranges. World sizes that do not divide the
    hash-column count (or exceed it) are supported: short head slices are
    padded and ranks owning no heads contribute all-zero blocks, so the
    all-gather still reassembles every real column.

    ``storage`` selects the on-device layout: ``fp8`` (official FP8 release:
    FP8-E4M3 payload bytes + ue8m0 scale bytes) or ``int8`` (Eco-Tech W8A8
    export: symmetric int8 payload + float32 per-group-32 scales).
    """

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: Sequence[int],
        tp_size: int,
        tp_rank: int,
        quant_block: int = 32,
        storage: str = ENGRAM_STORAGE_FP8,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if not head_sizes or any(size <= 0 for size in head_sizes):
            raise ValueError(f"engram head sizes must be positive, got {head_sizes}")
        if sum(head_sizes) > num_embeddings:
            raise ValueError(f"engram head sizes sum to {sum(head_sizes)} rows but the table only has {num_embeddings}")
        if dim % quant_block != 0:
            raise ValueError(f"engram head dim {dim} is not divisible by {quant_block}")
        self.num_embeddings = int(num_embeddings)
        self.dim = int(dim)
        self.quant_block = int(quant_block)
        self.storage = storage
        self.n_hash_cols = len(head_sizes)
        self.part_n_hash_cols = (self.n_hash_cols + tp_size - 1) // tp_size
        self.head_start = tp_rank * self.part_n_hash_cols
        head_end = min(self.head_start + self.part_n_hash_cols, self.n_hash_cols)
        self.vocab_start = sum(head_sizes[: self.head_start])
        self.vocab_end = sum(head_sizes[:head_end])
        self.part_num_embeddings = self.vocab_end - self.vocab_start
        self.tp_size = int(tp_size)
        if storage == ENGRAM_STORAGE_INT8:
            # Symmetric int8 payloads with one float32 scale per 32 columns
            # (Eco-Tech W8A8 embedding_storage format int8_sym).
            self.weight = nn.Parameter(
                torch.empty(self.part_num_embeddings, self.dim, dtype=torch.int8, device=device),
                requires_grad=False,
            )
            self.weight_scale = nn.Parameter(
                torch.empty(
                    self.part_num_embeddings,
                    self.dim // self.quant_block,
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
        else:
            # FP8-E4M3 payloads and ue8m0 scales are kept as raw uint8 bytes:
            # index_select stays on a universally supported dtype and the
            # bit-level decoders in deepseek_v41 do the rest.
            self.weight = nn.Parameter(
                torch.empty(self.part_num_embeddings, self.dim, dtype=torch.uint8, device=device),
                requires_grad=False,
            )
            self.weight_scale = nn.Parameter(
                torch.empty(
                    self.part_num_embeddings,
                    self.dim // self.quant_block,
                    dtype=torch.uint8,
                    device=device,
                ),
                requires_grad=False,
            )

    def _local_block(self, indices: torch.Tensor) -> torch.Tensor:
        """This rank's ``[T, part_n_hash_cols, dim]`` block of the lookup.

        Real head columns are dequantized from the quantized table; columns
        this rank does not own (hash ids outside its row range, slice padding
        when ``tp_size`` does not divide ``n_hash_cols``, or a rank entirely
        past the last head) contribute zeros.
        """
        # Deferred: deepseek_v41 imports this module at load time, so the
        # inlined quant decoders are resolved on first use instead.
        from xllm.python.models.deepseek_v41 import _e4m3_to_float, ue8m0_to_scale

        num_tokens = indices.shape[0]
        if self.part_num_embeddings == 0:
            # tp_size > n_hash_cols: this rank's head slice starts past the
            # last head, so it owns no table rows; the empty table cannot
            # serve index_select at all -- contribute an all-zero block.
            return torch.zeros(
                num_tokens,
                self.part_n_hash_cols,
                self.dim,
                dtype=torch.bfloat16,
                device=indices.device,
            )
        local_cols = indices[:, self.head_start : self.head_start + self.part_n_hash_cols]
        if local_cols.size(1) < self.part_n_hash_cols:
            # tp_size does not divide n_hash_cols: the last head-owning
            # rank's slice runs short of the uniform block width. Pad with
            # column 0 (id 0 predates every non-first rank's row range, so
            # the padding is never owned and is zeroed below) to keep the
            # gathered blocks shape-uniform.
            local_cols = F.pad(local_cols, (0, self.part_n_hash_cols - local_cols.size(1)))
        owned = (local_cols >= self.vocab_start) & (local_cols < self.vocab_end)
        local = (local_cols - self.vocab_start).clamp(0, self.part_num_embeddings - 1)
        flat = local.reshape(-1).to(torch.int64)
        payload = self.weight.index_select(0, flat)
        scale = self.weight_scale.index_select(0, flat)
        if self.storage == ENGRAM_STORAGE_INT8:
            values = payload.to(torch.float32)
            scale = scale.repeat_interleave(self.quant_block, dim=-1)
        else:
            values = _e4m3_to_float(payload)
            scale = ue8m0_to_scale(scale).repeat_interleave(self.quant_block, dim=-1)
        rows = (values * scale).to(torch.bfloat16)
        rows = rows.view(num_tokens, self.part_n_hash_cols, self.dim)
        return torch.where(owned.unsqueeze(-1), rows, torch.zeros_like(rows))

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """``[T, n_hash_cols]`` global row ids -> ``[T, n_hash_cols, dim]`` bf16.

        Only this rank's head columns are read; foreign rows contribute zeros
        and the TP all-gather reassembles the full head set. Short or empty
        head slices (``tp_size`` not dividing ``n_hash_cols``) contribute
        zero columns; the post-gather trim restores exactly the
        ``n_hash_cols`` real columns.
        """
        rows = self._local_block(indices)
        if self.tp_size > 1:
            from xllm.python import distributed as _distributed

            gathered = _distributed.tp_all_gather(rows, dim=1, world_size=self.tp_size)
            rows = gathered[:, : self.n_hash_cols]
        return rows


# ---------------------------------------------------------------------------
# Per-layer Engram module
# ---------------------------------------------------------------------------


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by match quality.

    The hash ids fetch ``n_hash_cols`` rows; ``wkv`` turns them into one key
    per hc copy plus a shared value. The gate is a normalized dot product of
    the stream against the key, signed-sqrt'ed before the sigmoid (matching
    the reference kernel bit-for-bit, including the ``dot == 0`` edge that
    keeps the positive clamp value).
    """

    def __init__(
        self,
        layout: EngramLayout,
        layer_hash_index: int,
        hidden_size: int,
        hc_mult: int,
        rms_norm_eps: float,
        tp_size: int,
        tp_rank: int,
        dtype: torch.dtype,
        device: torch.device,
        storage: str = ENGRAM_STORAGE_FP8,
        gate_unrotate: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.layer_hash_index = layer_hash_index
        self.dim = hidden_size
        self.hc_mult = hc_mult
        self.eps = rms_norm_eps
        self.embed_tokens = EngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            layout.head_sizes(layer_hash_index),
            tp_size,
            tp_rank,
            storage=storage,
            device=device,
        )
        n_hash_cols = layout.n_hash_cols
        # Replicated (not TP-sharded), matching the reference ReplicatedLinear.
        self.wkv = nn.Linear(
            n_hash_cols * layout.head_dim,
            hidden_size * (hc_mult + 1),
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.q_weight = nn.Parameter(
            torch.empty(hc_mult, hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.k_weight = nn.Parameter(
            torch.empty(hc_mult, hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        # QuaRot gate un-rotation (Eco-Tech W8A8 export): the stream arrives
        # rotated (h_rot) while q/k/wkv-K are in the original basis, so the
        # gate dot uses h_rot @ Q.T. None on unrotated checkpoints.
        self.register_buffer("gate_unrotate", gate_unrotate, persistent=False)
        self._staged_rows: torch.Tensor | None = None

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Gather this layer's rows before the decoder layer loop starts."""
        self._staged_rows = self.embed_tokens.lookup(hash_ids)

    def embed(self, hash_ids: torch.Tensor) -> torch.Tensor:
        if self._staged_rows is None or self._staged_rows.shape[0] < hash_ids.shape[0]:
            self._staged_rows = self.embed_tokens.lookup(hash_ids)
        return self._staged_rows[: hash_ids.shape[0]]

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gated residual injection into the full mHC stream.

        Args:
            hidden_states: ``[T, hc_mult, dim]`` residual stream entering this
                block (between the previous sublayer's post and this block's
                pre, so the mix coefficients see the injected stream).
            hash_ids: ``[T, n_hash_cols]`` row ids of this layer.
            token_mask: ``[T]``; False shuts the gate so those positions pass
                through untouched (image spans).
        """
        rows = self.embed(hash_ids).to(self.wkv.weight.dtype)
        kv = self.wkv(rows.flatten(-2))
        num_tokens, hc_mult, dim = hidden_states.shape
        keys = kv[:, : hc_mult * dim].view(num_tokens, hc_mult, dim).float()
        value = kv[:, hc_mult * dim :].float()
        hidden = hidden_states.float()
        # QuaRot: q/k/keys are original-basis, the stream is rotated; the
        # gate dot must run on the un-rotated stream (h_rot @ Q.T). The value
        # half of wkv is pre-folded to output the rotated basis, so the
        # residual update below must NOT rotate the increment again.
        gate_hidden = hidden
        if self.gate_unrotate is not None:
            gate_hidden = hidden @ self.gate_unrotate
        hidden_rms = torch.rsqrt(gate_hidden.square().mean(-1) + self.eps)
        key_rms = torch.rsqrt(keys.square().mean(-1) + self.eps)
        dot = (gate_hidden * self.q_weight.float() * self.k_weight.float() * keys).sum(-1)
        dot = dot * hidden_rms * key_rms * float(dim) ** -0.5
        gate_input = dot.abs().clamp_min(_ENGRAM_GATE_CLAMP).sqrt()
        gate_input = torch.where(dot < 0.0, -gate_input, gate_input)
        gate = torch.sigmoid(gate_input)
        if token_mask is not None:
            gate = torch.where(token_mask.reshape(-1, 1), gate, torch.zeros_like(gate))
        # The value is shared across hc copies ([T, dim]); the gate is
        # per-copy ([T, hc]).
        output = hidden + gate.unsqueeze(-1) * value.unsqueeze(1)
        return output.to(hidden_states.dtype)
