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

"""Load a draft model's optional and target-derived vocabulary weights.

A block-diffusion draft may ship its own embedding / lm_head in its weights,
share the target's, or (for a QuaRot target) reuse the target weights rotated
back into the draft's original hidden basis. ``load_draft_*_if_present`` load
whatever the draft's own weights contain, leaving missing modules unset;
``load_missing_draft_vocab_from_quarot_target`` then fills whatever is still
unset. Layer imports stay lazy until the C++ runtime bootstrap is ready.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn as nn

from scripts.logger import logger

from .parallel_load_context import ParallelLoadContext
from .scoped_weight_loader import ScopedWeightLoader
from .sharding import shard_tensor

if TYPE_CHECKING:
    from xllm.python.layers import ColumnParallelLinear, HiddenParallelEmbedding


class _DraftModelWithConfig(Protocol):
    cfg: Any  # duck-typed: reads .hidden_size / .vocab_size below
    dtype: torch.dtype
    device: torch.device


class _DraftCausalLM(_DraftModelWithConfig, Protocol):
    model: Any  # backbone exposing an optional embed_tokens module
    lm_head: nn.Module | None


def _create_embedding_shard(
    cfg: Any,
    tp_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> HiddenParallelEmbedding:
    """Allocate an embedding weight shard along the hidden dimension."""
    from xllm.python.layers import HiddenParallelEmbedding

    return HiddenParallelEmbedding(
        cfg.vocab_size,
        cfg.hidden_size // tp_size,
        tp_size,
        dtype=dtype,
        device=device,
    )


def _create_lm_head_shard(
    cfg: Any,
    tp_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> ColumnParallelLinear:
    """Allocate an output-head weight shard along the vocabulary dimension."""
    from xllm.python.layers import ColumnParallelLinear

    return ColumnParallelLinear(
        cfg.hidden_size,
        (getattr(cfg, "draft_vocab_size", 0) or cfg.vocab_size) // tp_size,
        tp_size,
        gather_output=True,
        dtype=dtype,
        device=device,
    )


def _load_draft_module_if_present(
    model: object,
    weights: ScopedWeightLoader,
    weight_key: str,
    attr_name: str,
    create_module: Callable[[], nn.Module],
    *,
    context: ParallelLoadContext,
    shard_dim: int,
) -> None:
    """Create and load a module only when the draft weights contain it."""
    if not weights.has(weight_key):
        return
    setattr(model, attr_name, create_module())
    weights.load_tensor(
        getattr(model, attr_name).weight,
        weight_key,
        dim=shard_dim,
        rank=context.tp_rank,
        world_size=context.tp_size,
    )


def load_draft_embedding_if_present(
    model: _DraftModelWithConfig,
    weights: ScopedWeightLoader,
    *,
    context: ParallelLoadContext,
) -> None:
    """Load the draft weights' optional embedding, preserving its trained token rows."""
    _load_draft_module_if_present(
        model,
        weights,
        "embed_tokens.weight",
        "embed_tokens",
        lambda: _create_embedding_shard(model.cfg, context.tp_size, model.dtype, model.device),
        context=context,
        shard_dim=1,
    )


def load_draft_lm_head_if_present(
    model: _DraftModelWithConfig,
    weights: ScopedWeightLoader,
    *,
    context: ParallelLoadContext,
) -> None:
    """Load the draft weights' optional output head with vocabulary-parallel sharding."""
    _load_draft_module_if_present(
        model,
        weights,
        "lm_head.weight",
        "lm_head",
        lambda: _create_lm_head_shard(model.cfg, context.tp_size, model.dtype, model.device),
        context=context,
        shard_dim=0,
    )


def _target_ties_word_embeddings(target_model_path: str | Path) -> bool:
    """Read the target ``config.json``'s ``tie_word_embeddings`` flag.

    The QuaRot rotation is applied before ``lm_head`` shares the embedding
    weight in tied targets, so a tied target may omit ``lm_head.weight``
    entirely. Returns ``False`` when the config is missing or unreadable —
    callers fall back to their existing lm_head-name lookup, which surfaces
    the underlying ``KeyError`` for untied targets that genuinely lack the
    weight.
    """
    config_path = Path(target_model_path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        # Corrupt / unreadable config surfaces later as a KeyError from the
        # lm_head name lookup; the warning here points at the real root cause.
        logger.warning(f"target config.json unreadable, treating as untied: {config_path}")
        return False
    # Multimodal targets keep the language model's flag in text_config. An
    # explicit text flag takes precedence over the enclosing model's default.
    text_config = config.get("text_config") or {}
    return bool(text_config.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))


def _find_target_weight(
    model_path: Path,
    names: tuple[str, ...],
) -> tuple[Path, str]:
    """Locate a target tensor without loading unrelated weight shards."""
    from safetensors import safe_open

    for index_path in sorted(model_path.glob("*.safetensors.index.json")):
        with index_path.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        for name in names:
            if name in weight_map:
                return model_path / weight_map[name], name
    for shard_path in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            keys = set(shard.keys())
        for name in names:
            if name in keys:
                return shard_path, name
    raise KeyError(f"target weights have none of {names}: {model_path}")


@torch.no_grad()
def _load_target_weight_in_draft_basis(
    weight: torch.Tensor,
    target_model_path: str,
    names: tuple[str, ...],
    rotation: torch.Tensor,
    shard_dim: int,
    tp_rank: int,
    tp_size: int,
) -> None:
    """Undo the target's hidden rotation into one draft TP weight shard."""
    from safetensors import safe_open

    shard_path, name = _find_target_weight(Path(target_model_path), names)
    expected_shape = list(weight.shape)
    expected_shape[shard_dim] *= tp_size
    if rotation.shape != (expected_shape[1], expected_shape[1]):
        raise ValueError("target rotation must match the draft hidden size")
    row_offset = 0
    if shard_dim == 0:
        row_offset = tp_rank * weight.size(0)
    else:
        rotation = shard_tensor(rotation, 0, tp_rank, tp_size, name="global_rotation", contiguous=False)
    rotation_t = rotation.to(device=weight.device, dtype=torch.float32).T

    with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
        target_slice = shard.get_slice(name)
        if target_slice.get_shape() != expected_shape:
            raise ValueError(f"target {name} has shape {target_slice.get_shape()}, expected {expected_shape}")
        # Chunk rows to bound temporary device memory independently of vocab
        # size; each rank reads only its own slice, so no collective is needed.
        rows_per_chunk = 1024
        for start in range(0, weight.size(0), rows_per_chunk):
            end = min(start + rows_per_chunk, weight.size(0))
            block = target_slice[row_offset + start : row_offset + end]
            if not block.is_floating_point():
                raise ValueError(f"target {name} must be a floating-point weight")
            aligned = block.to(device=weight.device, dtype=torch.float32) @ rotation_t
            weight[start:end].copy_(aligned)


def load_missing_draft_vocab_from_quarot_target(
    draft: _DraftCausalLM,
    target_model_path: str,
    rotation: torch.Tensor,
) -> None:
    """Fill missing embedding/head modules in the draft's original hidden basis.

    This runs after ``load_draft_*_if_present`` and before target sharing.
    Existing modules are preserved; only missing modules need target weights.
    """
    cfg = draft.cfg
    embed_names = (
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    )

    def _load_vocab_shard(weight: torch.Tensor, names: tuple[str, ...], shard_dim: int) -> None:
        _load_target_weight_in_draft_basis(
            weight,
            target_model_path,
            names,
            rotation,
            shard_dim=shard_dim,
            tp_rank=cfg.tp_rank,
            tp_size=cfg.tp_size,
        )

    if draft.model.embed_tokens is None:
        embedding = _create_embedding_shard(cfg, cfg.tp_size, draft.dtype, draft.device)
        _load_vocab_shard(embedding.weight, embed_names, shard_dim=1)
        draft.model.embed_tokens = embedding
    if draft.lm_head is None:
        head = _create_lm_head_shard(cfg, cfg.tp_size, draft.dtype, draft.device)
        # A tied target may save only embed_tokens.weight. The output head uses
        # the same full tensor, but shards its vocabulary dimension.
        head_names = (
            embed_names
            if _target_ties_word_embeddings(target_model_path)
            else ("lm_head.weight", "language_model.lm_head.weight", "model.lm_head.weight")
        )
        _load_vocab_shard(head.weight, head_names, shard_dim=0)
        draft.lm_head = head
