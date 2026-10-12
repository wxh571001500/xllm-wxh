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

"""DeepSeek-V4.1 Flash Python model (CED + CSA2, TORCH backend).

Faithful port of the official reference implementation
(``/weights/DeepSeek/DeepSeek-V4.1-Flash/inference/model.py``, which is the
semantic source of truth). The forward is pure
torch and NPU-only (there is no CPU/CUDA attention backend); the reference QDQ
numerics are preserved by the inlined quantization helpers at the bottom of
this file (device-agnostic, unit-tested on CPU; ue8m0 decode, FP8 block
dequant, MXFP4 dequant, activation QDQ):

  * ``CSA2LayerPlan`` -- static Full/Reindex/Reuse/SWA assignment. Every
    kv-source layer compresses its OWN attention input (there is no
    cross-layer CED hidden capture in the reference); the group's Reindex /
    Reuse layers read that source's caches (``main_kv_source_layer``).
  * ``DeepseekV41Attention`` -- latent Q/KV projections + interleaved-pair
    RoPE, the ratio-1/ratio-2 compressor weights, the CSA2 indexer weights
    (``wk`` acts on the 512-d latent, not the hidden), and the grouped
    ``wo_a``/``wo_b`` output projection with output de-rotation.
  * ``Csa2AttentionBackend`` (``xllm/python/attention/csa2_attention.py``)
    owns the cache orchestration: SWA ring, state parking/pooling, index
    writes, scoring, candidate pool, top-k and the sink softmax.
  * ``DeepseekV41HyperConnection`` -- single-pass Mega-mHC with the
    Sinkhorn split; attention collapses with the previous block's
    ``ffn_pre``, the FFN with this block's ``attn_pre``, and the final
    merge is the learned ``hc_pre(h, last_ffn_pre)`` weighted sum.
  * ``DeepseekV41MoE`` -- fp32 sqrtsoftplus routing with bias-steered
    selection (``gate.bias`` / per-token ``gate.bias_vl``), clamped SwiGLU
    experts and one shared expert.
  * ``DeepseekV41ForCausalLM`` -- checkpoint-mode-aware construction and
    loading: ``fp8`` (official FP8-block/MXFP4 release, dequantized to bf16
    at load), ``ascend`` (W8A8 export, existing kernel path) and ``none``.

Vision and the DSpark draft model are out of scope here; their checkpoint
tensors are simply not consumed. Engram IS implemented
(:mod:`deepseek_v41_engram`): the layers listed in ``engram_layer_ids``
carry the gated n-gram lookup injected into the mHC residual stream, with
the tokenizer-derived hash state loaded from the
``engram_hash_state.bin`` sidecar.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.logger import logger
from xllm.python.layers.attention import Attention
from xllm.python.layers.embedding import HiddenParallelEmbedding
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.model_executor.forward_context import (
    get_forward_context,
    record_layer_event,
)
from xllm.python.model_loader.module_loaders import load_w8a8_dynamic_projection
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    DeepseekV4MoE,
    DeepseekV4RotaryEmbedding,
    _find_checkpoint_prefix,
    _pick,
)
from xllm.python.models.deepseek_v32 import DeepseekV3MLP, W8A8DynamicLinear
from xllm.python.models.deepseek_v41_engram import (
    ENGRAM_STORAGE_FP8,
    ENGRAM_STORAGE_INT8,
    Engram,
    EngramHashState,
    EngramLayout,
    image_sentinel_mask,
    load_quarot_gate_unrotate,
)
from xllm.python.models.weight_utils import W8A8WeightLoader

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict

    from xllm.python.attention.csa2_attention import Csa2LayerContext

# ---------------------------------------------------------------------------
# CSA2 layer plan (Full / Reindex / Reuse / SWA)
# ---------------------------------------------------------------------------

CSA2_MODE_SWA = "swa"
CSA2_MODE_FULL = "full"
CSA2_MODE_REINDEX = "reindex"
CSA2_MODE_REUSE = "reuse"


@dataclass(frozen=True)
class CSA2LayerMode:
    """Static CSA2 assignment for one layer."""

    layer_id: int
    mode: str
    # Main-KV compression ratio: 2 for encoder CSA2 layers, 1 for decoder
    # main-KV layers, 0 for pure SWA layers.
    compress_ratio: int
    # Layer whose main-KV / index caches this layer reads (-1 for SWA layers,
    # self for Full layers, the group's kv source for Reindex/Reuse layers).
    main_kv_source_layer: int

    @property
    def is_csa2(self) -> bool:
        return self.mode != CSA2_MODE_SWA

    @property
    def is_full(self) -> bool:
        return self.mode == CSA2_MODE_FULL

    @property
    def is_reindex(self) -> bool:
        return self.mode == CSA2_MODE_REINDEX

    @property
    def is_reuse(self) -> bool:
        return self.mode == CSA2_MODE_REUSE


class CSA2LayerPlan:
    """Static per-layer CSA2 assignment, built once at model init.

    Derivation (reference model.py:653-661): ``kv_source_layer_ids`` are the
    Full layers that own a Compressor and publish the group's compressed KV
    and index K; ``index_source_layer_ids`` additionally lists the Reindex
    layers that re-score the shared K with their own indexer Q; every other
    layer after the nearest preceding kv source reuses the shared main KV and
    top-k. Layers before the first kv source (or with compress ratio 0) are
    pure SWA. For the reference config (``kv=[2,8,14,20]``,
    ``index=[2,8,14,20,24,28,32,36]``) this reproduces the documented
    assignment: layers 0-1 SWA, 2-19 encoder groups (Full at 2/8/14, Reuse
    otherwise), 20-39 decoder group (Full at 20, Reindex at 24/28/32/36,
    Reuse otherwise).

    There is NO cross-layer hidden capture: every kv-source layer compresses
    its own attention input (semantics doc section 3).
    """

    def __init__(self, modes: list[CSA2LayerMode]) -> None:
        self._modes = modes

    def mode(self, layer_id: int) -> CSA2LayerMode:
        if layer_id < 0 or layer_id >= len(self._modes):
            raise IndexError(f"layer {layer_id} outside the CSA2 plan's {len(self._modes)} layers")
        return self._modes[layer_id]

    def __len__(self) -> int:
        return len(self._modes)

    @classmethod
    def build(
        cls,
        n_layers: int,
        compress_ratios: Sequence[int],
        kv_source_layer_ids: Sequence[int],
        index_source_layer_ids: Sequence[int],
    ) -> CSA2LayerPlan:
        """Validate the config-derived inputs and derive the per-layer modes."""
        kv_sources = sorted({int(source) for source in kv_source_layer_ids})
        index_sources = sorted({int(source) for source in index_source_layer_ids})
        kv_source_set = set(kv_sources)
        index_source_set = set(index_sources)
        for source in kv_sources:
            if not 0 <= source < n_layers:
                raise ValueError(f"kv source layer {source} is outside the model's {n_layers} layers")
        for source in index_sources:
            if not 0 <= source < n_layers:
                raise ValueError(f"index source layer {source} is outside the model's {n_layers} layers")
        non_indexed = [source for source in kv_sources if source not in index_source_set]
        if non_indexed:
            raise ValueError(f"kv source layers must also be index sources: {non_indexed}")
        for source in index_sources:
            if not any(kv_source <= source for kv_source in kv_sources):
                raise ValueError(f"index source layer {source} has no preceding kv source layer")
        ratios = [int(ratio) for ratio in compress_ratios]
        for source in kv_sources:
            if source >= len(ratios) or ratios[source] <= 0:
                raise ValueError(f"kv source layer {source} must have a positive compress ratio")

        modes: list[CSA2LayerMode] = []
        for layer_id in range(n_layers):
            ratio = ratios[layer_id] if layer_id < len(ratios) else 0
            kv_source = max((source for source in kv_sources if source <= layer_id), default=-1)
            if ratio <= 0 or kv_source < 0:
                mode = CSA2_MODE_SWA
                main_kv_source = -1
            elif layer_id in kv_source_set:
                mode = CSA2_MODE_FULL
                main_kv_source = layer_id
            elif layer_id in index_source_set:
                mode = CSA2_MODE_REINDEX
                main_kv_source = kv_source
            else:
                mode = CSA2_MODE_REUSE
                main_kv_source = kv_source
            modes.append(
                CSA2LayerMode(
                    layer_id=layer_id,
                    mode=mode,
                    compress_ratio=max(ratio, 0),
                    main_kv_source_layer=main_kv_source,
                )
            )
        return cls(modes)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _int_list(d: dict, key: str, default: list[int] | None = None) -> list[int]:
    value = d.get(key)
    if value is None:
        return list(default or [])
    return [int(item) for item in value]


def _parse_v41_compress_ratios(raw: Sequence[int], n_layers: int) -> list[int]:
    """Parse V4.1 ``compress_ratios`` preserving ratio 0 / 1 / 2.

    V4 normalized every ``ratio <= 1`` to 1, which erases V4.1's distinction
    between pure-SWA layers (ratio 0) and decoder main-KV layers (ratio 1).
    The checkpoint config carries ``n_layers + num_nextn_predict_layers``
    entries (trailing zeros for the DSpark draft blocks); the main model
    consumes the first ``n_layers``. Missing trailing entries pad with 0
    (pure SWA) so no cache group is fabricated.
    """
    ratios = [int(ratio) for ratio in raw]
    for ratio in ratios:
        if ratio not in (0, 1, 2, 4):
            raise ValueError(f"unsupported DeepSeek-V4.1 compression ratio: {ratio}")
    if len(ratios) < n_layers:
        ratios.extend([0] * (n_layers - len(ratios)))
    return ratios[:n_layers]


# Defaults for config fields whose V4 values differ; injected only when the
# config omits them so an explicit value always wins.
_V41_FIELD_DEFAULTS: dict[str, Any] = {
    "model_type": "deepseek_v41",
    "hidden_size": 5120,
    "q_lora_rank": 1280,
    "n_routed_experts": 384,
    "moe_intermediate_size": 2304,
    "index_n_heads": 32,
    "rms_norm_eps": 1e-20,
}


@dataclass
class DeepseekV41Config(DeepseekV4Config):
    """DeepSeek-V4.1 Flash model config (V4 fields + CSA2/CED/Engram/DSpark)."""

    model_type: str = "deepseek_v41"
    hidden_size: int = 5120
    n_layers: int = 40
    q_lora_rank: int = 1280
    rms_norm_eps: float = 1e-20
    n_routed_experts: int = 384
    moe_intermediate_size: int = 2304
    index_n_heads: int = 32
    # -- CSA2 / CED --
    # candidate_source_layer_id / image_token_id default to the C++ "disabled"
    # sentinels (LOAD_ARG_OR in xllm/models/llm/deepseek_v41.h) so a config that
    # omits them does not silently activate a candidate pool or fabricate image
    # token matches. The released config always carries the real values.
    kv_source_layer_ids: list[int] = field(default_factory=list)
    index_source_layer_ids: list[int] = field(default_factory=list)
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    # -- Engram (gated n-gram lookup injected on engram_layer_ids) --
    engram_layer_ids: list[int] = field(default_factory=list)
    engram_num_embeddings: list[int] = field(default_factory=list)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_compressed_vocab_size: int = 99092
    # -1 is the "unset" sentinel so DeepseekV41Model.__init__ can reject an
    # enabled engram with no explicit pad id (guard: `engram_pad_token_id < 0`).
    # C++ defaults to the same -1 (deepseek_v41.h / model_args.h), so the
    # sentinel is not overwritten on the service path.
    engram_pad_token_id: int = -1
    # -- DSpark draft geometry (consumed by the draft model, Phase 3) --
    num_nextn_predict_layers: int = 3
    dspark_target_layer_ids: list[int] = field(default_factory=lambda: [37, 38, 39])
    dspark_block_size: int = 5
    dspark_markov_rank: int = 256
    dspark_noise_token_id: int = 128799
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3
    # -- Vision (Phase 4); image_token_id also drives gate.bias_vl --
    # -1 = no image tokens (text-only), matching the C++ loader default.
    image_token_id: int = -1

    @classmethod
    def from_dict(cls, d: dict) -> DeepseekV41Config:
        merged = dict(d)
        for key, value in _V41_FIELD_DEFAULTS.items():
            if merged.get(key) is None:
                merged[key] = value
        # The layer count has two aliases; only inject when both are absent
        # so an explicit value of either alias always wins.
        if merged.get("num_hidden_layers") is None and merged.get("n_layers") is None:
            merged["num_hidden_layers"] = 40
        cfg = super().from_dict(merged)
        return replace(
            cfg,
            # Re-parse: V4's from_dict normalized "ratio <= 1 -> 1".
            compress_ratios=_parse_v41_compress_ratios(list(merged.get("compress_ratios") or []), cfg.n_layers),
            # V4.1 removed hash routing (gate.tid2eid); the field stays for
            # the inherited V4 machinery.
            n_hash_layers=0,
            kv_source_layer_ids=_int_list(merged, "kv_source_layer_ids"),
            index_source_layer_ids=_int_list(merged, "index_source_layer_ids"),
            candidate_source_layer_id=int(_pick(merged, "candidate_source_layer_id", default=-1)),
            candidate_topk_blocks=int(_pick(merged, "candidate_topk_blocks", default=2048)),
            candidate_block_size=int(_pick(merged, "candidate_block_size", default=8)),
            engram_layer_ids=_int_list(merged, "engram_layer_ids"),
            engram_num_embeddings=_int_list(merged, "engram_num_embeddings"),
            engram_max_ngram_size=int(_pick(merged, "engram_max_ngram_size", default=4)),
            engram_vocab_size=int(_pick(merged, "engram_vocab_size", default=16000000)),
            engram_n_heads=int(_pick(merged, "engram_n_heads", default=8)),
            engram_head_dim=int(_pick(merged, "engram_head_dim", default=256)),
            engram_compressed_vocab_size=int(_pick(merged, "engram_compressed_vocab_size", default=99092)),
            engram_pad_token_id=int(_pick(merged, "engram_pad_token_id", default=-1)),
            num_nextn_predict_layers=int(_pick(merged, "num_nextn_predict_layers", default=3)),
            dspark_target_layer_ids=_int_list(merged, "dspark_target_layer_ids", default=[37, 38, 39]),
            dspark_block_size=int(_pick(merged, "dspark_block_size", default=5)),
            dspark_markov_rank=int(_pick(merged, "dspark_markov_rank", default=256)),
            dspark_noise_token_id=int(_pick(merged, "dspark_noise_token_id", default=128799)),
            dspark_n_routed_experts=int(_pick(merged, "dspark_n_routed_experts", default=128)),
            dspark_num_experts_per_tok=int(_pick(merged, "dspark_num_experts_per_tok", default=3)),
            image_token_id=int(_pick(merged, "image_token_id", default=-1)),
        )


# ---------------------------------------------------------------------------
# Checkpoint quantization modes (adaptation contract section 4)
# ---------------------------------------------------------------------------

V41_QUANT_FP8 = "fp8"
V41_QUANT_ASCEND = "ascend"
V41_QUANT_NONE = "none"


def _detect_v41_quant_mode(config: dict) -> str:
    """Read ``{model_path}/config.json`` and detect the checkpoint mode.

    ``quantization_config.quant_method == "fp8"`` selects the official
    FP8-block/MXFP4 release (dequantized to bf16 at load); a
    ``quant_model_description.json`` or ``quant_method == "ascend"`` selects
    the W8A8 Eco-Tech export; anything else is bf16.
    """
    model_path = config.get("model_path")
    if not model_path:
        logger.warning(
            "DeepSeek-V4.1 quant mode detection: model_path missing; defaulting to bf16 (V41_QUANT_NONE)"
        )
        return V41_QUANT_NONE
    if os.path.isfile(os.path.join(model_path, "quant_model_description.json")):
        return V41_QUANT_ASCEND
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return V41_QUANT_NONE
    with open(config_path, encoding="utf-8") as handle:
        checkpoint_config = json.load(handle)
    quant_config = checkpoint_config.get("quantization_config") or {}
    method = quant_config.get("quant_method")
    if method == "fp8":
        return V41_QUANT_FP8
    if method == "ascend":
        return V41_QUANT_ASCEND
    return V41_QUANT_NONE


def _make_v41_linear(
    in_features: int,
    out_features: int,
    quant_mode: str,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """Construct one attention/dense linear per checkpoint mode."""
    if quant_mode == V41_QUANT_ASCEND:
        return W8A8DynamicLinear(
            in_features,
            out_features,
            device,
            transpose_weight_after_loading=False,
        )
    return nn.Linear(in_features, out_features, bias=False, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Pure-torch RMSNorm / RoPE tables / mHC
# ---------------------------------------------------------------------------


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm (model.py:281-293): fp32 statistics, cast back."""
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (weight * xf).to(dtype)


class DeepseekV41RMSNorm(nn.Module):
    """Pure-torch RMSNorm (the shared layer RMSNorm calls NPU kernels)."""

    def __init__(
        self,
        dim: int,
        eps: float,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None:
            x = x + residual
        return _rms_norm(x, self.weight, self.eps)


def _build_plain_cos_sin_cache(
    rotary_dim: int,
    max_positions: int,
    theta: float,
    device: torch.device,
) -> torch.Tensor:
    """Plain RoPE table (theta 10000, YaRN off) for the ratio-0 layers.

    Mirrors ``precompute_freqs_cis`` with ``original_seq_len == 0``
    (model.py:368-389): one cos/sin pair per frequency per position, stored
    as ``[max_positions, rotary_dim]`` (cos | sin halves).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    positions = torch.arange(max_positions, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return cache.to(device).contiguous()


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the hc mixes into (pre, post, comb) coefficients.

    Faithful port of the reference ``hc_split_sinkhorn`` (kernel.py:406-474):
    ``pre = sigmoid(mix * scale[0] + base) + eps``,
    ``post = 2 * sigmoid(mix * scale[1] + base)``, and ``comb`` from a row
    softmax (+eps) followed by one column normalization and
    ``sinkhorn_iters - 1`` row+column normalization rounds.
    """
    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = mixes[:, 2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult))
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    return pre, post, comb


class DeepseekV41HyperConnection(nn.Module):
    """Single-pass Mega-mHC hyper-connections (semantics doc section 7).

    Pure-torch counterpart of the reference ``Block`` mHC pieces
    (model.py:948-966): ``hc_mixes`` projects the flattened hc*dim stream
    (normalized AFTER projection), ``hc_pre`` collapses the hc copies with a
    per-token coefficient vector, and ``hc_post`` expands a sublayer output
    back mixing the residual through the doubly-stochastic ``comb``.
    """

    def __init__(self, cfg: DeepseekV41Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_eps = cfg.hc_eps
        self.norm_eps = cfg.rms_norm_eps
        self.sinkhorn_iters = cfg.hc_sinkhorn_iters
        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_dim = cfg.hc_mult * cfg.hidden_size
        # hc_fn/scale/base per sub-block, fp32 as in the checkpoint.
        for part in ("attn", "ffn"):
            self.register_parameter(
                f"hc_{part}_fn",
                nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32, device=device)),
            )
            self.register_parameter(
                f"hc_{part}_base",
                nn.Parameter(torch.empty(mix_hc, dtype=torch.float32, device=device)),
            )
            self.register_parameter(
                f"hc_{part}_scale",
                nn.Parameter(torch.empty(3, dtype=torch.float32, device=device)),
            )

    def hc_mixes(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x ``[T, hc, d]`` -> (pre, post, comb) per token."""
        flattened = x.flatten(-2).float()
        rsqrt = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flattened, hc_fn) * rsqrt
        return _hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Collapse the hc copies: ``sum_j pre[j] * x[j]`` (model.py:957-960)."""
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=-2)
        return y.to(x.dtype)

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Expand back: ``out[j] = post[j] * x + sum_k comb[j, k] * res[k]``."""
        y = post.unsqueeze(-1) * x.float().unsqueeze(-2)
        y = y + torch.einsum("tjk,tkd->tjd", comb.float(), residual.float())
        return y.to(x.dtype)

    @staticmethod
    def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
        """Initial one-hot mix on copy 0 (model.py:1159-1163)."""
        pre_mix = torch.zeros(x.size(0), hc_mult, dtype=torch.float32, device=x.device)
        pre_mix[:, 0] = 1.0
        return pre_mix


# ---------------------------------------------------------------------------
# Indexer weights
# ---------------------------------------------------------------------------


class DeepseekV41Indexer(nn.Module):
    """DeepSeek-V4.1 CSA2 indexer weights (semantics doc section 5).

    ``wq_b`` projects the shared q-lora latent; ``weights_proj`` produces the
    per-head score weights; kv-source layers additionally own the key path
    ``wk`` + ``k_norm`` -- ``wk`` acts on the 512-d compressor latent, NOT on
    hidden states (checkpoint ``layers.N.attn.indexer.*``). There is no
    Hadamard transform and no INT8 quant in V4.1: Q/K are FP4-E2M1 QDQ and
    the scoring runs through einsum in the backend.
    """

    def __init__(
        self,
        cfg: DeepseekV41Config,
        dtype: torch.dtype,
        device: torch.device,
        quant_mode: str,
        owns_k: bool,
    ) -> None:
        super().__init__()
        self.owns_k = owns_k
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.wq_b = _make_v41_linear(
            cfg.q_lora_rank,
            self.n_heads * self.head_dim,
            quant_mode,
            dtype,
            device,
        )
        self.weights_proj = nn.Linear(cfg.hidden_size, self.n_heads, bias=False, dtype=dtype, device=device)
        if owns_k:
            # wk: head_dim (512) -> index_head_dim (128); one shared key per
            # compressed position (model.py:518).
            self.wk = nn.Linear(cfg.head_dim, self.head_dim, bias=False, dtype=dtype, device=device)
            self.k_norm = DeepseekV41RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype, device)

    def process_weights_after_loading(self) -> None:
        if isinstance(self.wq_b, W8A8DynamicLinear):
            self.wq_b.process_weights_after_loading()


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class DeepseekV41Attention(Attention):
    """DeepSeek-V4.1 CSA2 attention (semantics doc sections 4-6).

    Owns the reference weight set: latent q/kv projections with
    interleaved-pair RoPE on the last ``qk_rope_head_dim`` dims, the
    attention sink, the compressor weights (Full layers) and the indexer
    weights (Full/Reindex layers), plus the grouped ``wo_a``/``wo_b`` output
    projection. Cache orchestration and the sink softmax live in
    :class:`Csa2AttentionBackend`.
    """

    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        mode: CSA2LayerMode,
        quant_mode: str = V41_QUANT_NONE,
    ) -> None:
        tp = cfg.tp_size
        num_heads = cfg.n_heads // tp
        head_dim = cfg.head_dim
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=1,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            sliding_window=cfg.window_size,
            layer_id=layer_id,
        )
        self.cfg = cfg
        self.layer_id = layer_id
        self.mode = mode
        self.quant_mode = quant_mode
        self.num_heads_local = num_heads
        self.head_dim = head_dim
        # Attention sink (learnable logit bias per head); contributes to the
        # softmax normalizer only.
        self.register_parameter(
            "attn_sink",
            nn.Parameter(torch.zeros(num_heads, dtype=torch.float32, device=device)),
        )
        self.attn_sink_loaded = False
        # Q/KV projections.
        self.q_a_proj = _make_v41_linear(cfg.hidden_size, cfg.q_lora_rank, quant_mode, dtype, device)
        self.q_a_layernorm = DeepseekV41RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps, dtype, device)
        self.q_b_proj = _make_v41_linear(cfg.q_lora_rank, num_heads * head_dim, quant_mode, dtype, device)
        self.kv_proj = _make_v41_linear(cfg.hidden_size, head_dim, quant_mode, dtype, device)
        self.kv_a_layernorm = DeepseekV41RMSNorm(head_dim, cfg.rms_norm_eps, dtype, device)
        # Grouped output projection: o_a (block-diagonal over groups, bf16
        # after the official conversion) -> o_b.
        assert cfg.o_groups % tp == 0
        self.n_local_groups = cfg.o_groups // tp
        self.o_lora_rank = cfg.o_lora_rank
        o_a_in = (cfg.n_heads * head_dim) // cfg.o_groups
        if quant_mode == V41_QUANT_ASCEND:
            self.o_a_proj = ColumnParallelLinear(
                o_a_in,
                (cfg.o_groups * cfg.o_lora_rank) // tp,
                tp,
                dtype=dtype,
                device=device,
            )
            self.o_b_proj = RowParallelLinear(
                (cfg.o_groups * cfg.o_lora_rank) // tp,
                cfg.hidden_size,
                tp,
                dtype=dtype,
                device=device,
                use_checkpoint_layout=True,
            )
        else:
            # TP-sharded like the ascend branch: each rank owns the head
            # groups matching its q head slice; the o_b output is partial and
            # all-reduced across the TP group in forward.
            self.o_a_proj = nn.Linear(
                o_a_in,
                self.n_local_groups * cfg.o_lora_rank,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.o_b_proj = nn.Linear(
                self.n_local_groups * cfg.o_lora_rank,
                cfg.hidden_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
        # Main-KV compressor (Full layers only). Ratio 2 runs the gated
        # pooling in fp32, so its wkv/wgate are fp32; ratio 1 is a plain bf16
        # projection with no gate and no state (model.py:437-456).
        if mode.is_full:
            ratio = mode.compress_ratio
            cmp_dtype = torch.float32 if ratio > 1 else dtype
            self.cmp_wkv = nn.Linear(cfg.hidden_size, head_dim, bias=False, dtype=cmp_dtype, device=device)
            if ratio > 1:
                self.cmp_wgate = nn.Linear(cfg.hidden_size, head_dim, bias=False, dtype=torch.float32, device=device)
            self.cmp_norm = DeepseekV41RMSNorm(head_dim, cfg.rms_norm_eps, dtype, device)
        # Indexer: Full layers write the shared K and select; Reindex layers
        # re-select against it; Reuse/SWA layers own no indexer weights.
        self.indexer: DeepseekV41Indexer | None = (
            DeepseekV41Indexer(cfg, dtype, device, quant_mode, owns_k=mode.is_full)
            if mode.is_full or mode.is_reindex
            else None
        )
        # Per-forward hand-off to the backend.
        self._csa2_ctx: Csa2LayerContext | None = None
        self._csa2_used_topk: torch.Tensor | None = None

    def process_weights_after_loading(self) -> None:
        if self.quant_mode == V41_QUANT_ASCEND:
            for module in (self.q_a_proj, self.q_b_proj, self.kv_proj):
                module.process_weights_after_loading()
            if hasattr(self.o_a_proj, "process_weights_after_loading"):
                self.o_a_proj.process_weights_after_loading()
            if self.indexer is not None:
                self.indexer.process_weights_after_loading()

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        # The CSA2 layer context / RoPE helpers live in the attention backend
        # module (csa2_attention); the import is deferred so the model file
        # itself has no backend dependency at import time.
        from xllm.python.attention.csa2_attention import (
            Csa2LayerContext,
            apply_interleaved_rope,
        )

        backend = get_forward_context().attention_backend
        execute_csa2 = getattr(backend, "execute_csa2_layer", None)
        if not callable(execute_csa2):
            raise RuntimeError("DeepSeek-V4.1 requires the CSA2 attention backend")
        num_tokens = hidden.size(0)
        # Q path: qr = q_norm(wq_a(x)); q = wq_b(qr) (no norm after wq_b);
        # RoPE on the last rope dims at token positions (model.py:770-772).
        qr = self.q_a_layernorm(self.q_a_proj(hidden))
        q = self.q_b_proj(qr).view(num_tokens, self.num_heads_local, self.head_dim)
        q = apply_interleaved_rope(q, positions, cos_sin_cache)
        # SWA KV path: kv_norm -> RoPE -> FP8 QDQ over the whole post-RoPE
        # vector (model.py:700-720).
        kv = self.kv_a_layernorm(self.kv_proj(hidden))
        kv = apply_interleaved_rope(kv, positions, cos_sin_cache)
        kv = fp8_act_qdq(kv, 32)
        self._csa2_ctx = Csa2LayerContext(
            x=hidden,
            qr=qr,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
        )
        attn_out = execute_csa2(q, kv, self)
        # De-rotate the rope dims by the query's own rotation (conjugate)
        # BEFORE the o path (model.py:781).
        attn_out = apply_interleaved_rope(attn_out, positions, cos_sin_cache, inverse=True)
        # O path: grouped LoRA projection (model.py:783-789).
        out = attn_out.view(num_tokens, self.n_local_groups, -1)
        wo_a = self.o_a_proj.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o_low = torch.einsum("tgd,grd->tgr", out, wo_a)
        o = self.o_b_proj(o_low.reshape(num_tokens, -1))
        if self.cfg.tp_size > 1 and self.quant_mode != V41_QUANT_ASCEND:
            # Row-parallel partial: sum the per-rank head-group contributions.
            from xllm.python import distributed as _distributed

            _distributed.tp_all_reduce(o)
        return o

    # -- weight helpers used by the backend -------------------------------------

    def compress_kv_score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compressor projections: fp32 (kv, score) for ratio > 1, (kv, None) for ratio 1."""
        if self.mode.compress_ratio > 1:
            xf = x.float()
            return self.cmp_wkv(xf), self.cmp_wgate(xf)
        return self.cmp_wkv(x), None

    def compress_rmsnorm(self, v: torch.Tensor) -> torch.Tensor:
        """RMSNorm AFTER pooling (ratio > 1) / after the projection (ratio 1)."""
        return _rms_norm(v, self.cmp_norm.weight, self.cfg.rms_norm_eps)

    def indexer_key(self, latent: torch.Tensor) -> torch.Tensor:
        """Indexer K from the pre-RoPE compressor latent: k_norm(wk(latent))."""
        k = self.indexer.wk(latent)
        return _rms_norm(k, self.indexer.k_norm.weight, self.cfg.rms_norm_eps)

    def indexer_query(self, qr: torch.Tensor) -> torch.Tensor:
        """Indexer Q from the shared q-lora latent: wq_b(qr)."""
        return self.indexer.wq_b(qr).view(-1, self.indexer.n_heads, self.indexer.head_dim)

    def indexer_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-head scoring weights: weights_proj(x) * (ihd^-0.5 * nh^-0.5)."""
        scale = self.indexer.head_dim**-0.5 * self.indexer.n_heads**-0.5
        return self.indexer.weights_proj(x) * scale


# ---------------------------------------------------------------------------
# MoE (torch) + dense MLP + W8A8 MoE (ascend)
# ---------------------------------------------------------------------------


class DeepseekV41TorchMLP(nn.Module):
    """Dense SwiGLU FFN with the reference clamps (fp8/none checkpoint modes)."""

    def __init__(
        self,
        cfg: DeepseekV41Config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.swiglu_limit = cfg.swiglu_limit
        inter = cfg.moe_intermediate_size
        self.w1 = nn.Linear(cfg.hidden_size, inter, bias=False, dtype=dtype, device=device)
        self.w3 = nn.Linear(cfg.hidden_size, inter, bias=False, dtype=dtype, device=device)
        self.w2 = nn.Linear(inter, cfg.hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _swiglu_clamped(x, self.w1(x), self.w3(x), self.w2, self.swiglu_limit, None)


def _swiglu_clamped(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    w2: nn.Module,
    limit: float,
    route_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Reference Expert.forward (model.py:841-851).

    The up branch clamps both sides, the gate branch only from above; the
    routing weight (when given) multiplies the hidden before w2.
    """
    gate = gate_proj.float()
    up = up_proj.float()
    if 0.0 < limit < 1_000_000.0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    hidden = F.silu(gate) * up
    if route_weight is not None:
        hidden = route_weight.unsqueeze(-1) * hidden
    return w2(hidden.to(x.dtype))


class DeepseekV41MoE(nn.Module):
    """DeepSeek-V4.1 MoE for the fp8/none checkpoint modes (pure torch).

    Routing (model.py:809-827): fp32 ``softplus(x @ W).sqrt()`` scores, top-k
    of ``scores + bias`` where image-span tokens use ``gate.bias_vl``
    (selection only), weights gathered from the UNBIASED scores, normalized
    (``sum + 1e-20``) and scaled by ``routed_scaling_factor``. Experts are
    the clamped SwiGLU with fp32 routed accumulation plus one shared expert.
    """

    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.topk = cfg.n_activated_experts
        self.n_routed_experts = cfg.n_routed_experts
        self.routed_scaling = cfg.routed_scaling_factor
        self.norm_topk_prob = cfg.norm_topk_prob
        self.swiglu_limit = cfg.swiglu_limit
        self.image_token_id = cfg.image_token_id
        self.gate = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False, dtype=dtype, device=device)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(cfg.n_routed_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.gate_bias_vl = nn.Parameter(
            torch.zeros(cfg.n_routed_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.gate_bias_vl_loaded = False
        inter = cfg.moe_intermediate_size
        hidden = cfg.hidden_size
        # EP: every rank holds only its slice of the routed experts
        # (mirrors C++ FusedMoEImpl and DeepseekV4MoE). The forward
        # accumulates only the local expert slice and all-reduces the routed
        # partial across the EP group, so the summed output is exact.
        ep_size = cfg.ep_size if cfg.ep_size > 0 else 1
        ep_rank = cfg.ep_rank if cfg.ep_size > 0 else 0
        if cfg.n_routed_experts % ep_size != 0:
            raise ValueError(
                f"deepseek v4.1 torch MoE requires n_routed_experts divisible by ep_size, "
                f"got {cfg.n_routed_experts} experts and ep_size={ep_size}"
            )
        self.ep_size = ep_size
        self.num_experts_per_rank = cfg.n_routed_experts // ep_size
        self.start_expert_id = ep_rank * self.num_experts_per_rank
        # Stacked dequantized experts (fp8 mode dequantizes at load).
        self.experts_w1 = nn.Parameter(
            torch.empty(self.num_experts_per_rank, inter, hidden, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.experts_w2 = nn.Parameter(
            torch.empty(self.num_experts_per_rank, hidden, inter, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.experts_w3 = nn.Parameter(
            torch.empty(self.num_experts_per_rank, inter, hidden, dtype=dtype, device=device),
            requires_grad=False,
        )
        # One shared expert every token goes through (no routing weight).
        self.shared_w1 = nn.Linear(hidden, inter, bias=False, dtype=dtype, device=device)
        self.shared_w2 = nn.Linear(inter, hidden, bias=False, dtype=dtype, device=device)
        self.shared_w3 = nn.Linear(hidden, inter, bias=False, dtype=dtype, device=device)

    def _selection_logits(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 sqrtsoftplus scores and the bias-steered selection logits.

        The selection bias picks experts but never scales them (noaux_tc);
        tokens inside image spans use ``gate.bias_vl`` instead of
        ``gate.bias`` -- a per-token modality switch (model.py:809-823).
        """
        scores = F.linear(x.float(), self.gate.weight.float())
        scores = F.softplus(scores).sqrt()
        bias = self.e_score_correction_bias
        if input_ids is not None and self.gate_bias_vl_loaded:
            image_mask = input_ids.reshape(-1) == self.image_token_id
            bias = torch.where(image_mask.unsqueeze(-1), self.gate_bias_vl, self.e_score_correction_bias)
        return scores + bias, scores

    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        shape = hidden.shape
        x = hidden.reshape(-1, self.cfg.hidden_size)
        logits, scores = self._selection_logits(x, input_ids)
        # The bias picks experts but does not scale them.
        indices = logits.topk(self.topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling
        # This is the fp8/none serving path, not just a CPU reference.
        # Sort local routes once instead of scanning all routes for each expert.
        # Only active experts launch GEMMs; keep ascending expert order for the
        # fp32 accumulation, and stable token order within each expert.
        flat_experts = indices.reshape(-1)
        local_routes = torch.where(
            (flat_experts >= self.start_expert_id) & (flat_experts < self.start_expert_id + self.num_experts_per_rank)
        )[0]
        sorted_experts, permutation = flat_experts[local_routes].sort(stable=True)
        sorted_routes = local_routes[permutation]
        sorted_rows = sorted_routes // self.topk
        sorted_weights = weights.reshape(-1)[sorted_routes]
        active_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        # Interim eager implementation: copy segment metadata to the host once,
        # rather than synchronizing once per expert. Active experts still run
        # serially; this is not graph-capturable or a final fused MoE path.
        # A grouped GEMM replacement must preserve fp32 asymmetric SwiGLU and
        # apply routing weights BEFORE the down projection (not at combine).
        segments = torch.stack((active_experts - self.start_expert_id, counts), dim=1).cpu().tolist()
        y = torch.zeros_like(x, dtype=torch.float32)
        offset = 0
        for local_slot, count in segments:
            rows = sorted_rows[offset : offset + count]
            expert_input = x[rows]
            expert_out = _swiglu_clamped(
                expert_input,
                F.linear(expert_input, self.experts_w1[local_slot]),
                F.linear(expert_input, self.experts_w3[local_slot]),
                _StackedLinear(self.experts_w2, local_slot),
                self.swiglu_limit,
                sorted_weights[offset : offset + count],
            )
            y.index_add_(0, rows, expert_out.float())
            offset += count
        if self.ep_size > 1:
            from xllm.python import distributed as _distributed

            _distributed.moe_ep_all_reduce(y)
        y += self._shared_forward(x).float()
        return y.type_as(x).view(shape)

    def _shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        return _swiglu_clamped(x, self.shared_w1(x), self.shared_w3(x), self.shared_w2, self.swiglu_limit, None)


class _StackedLinear(nn.Module):
    """Adapter so a stacked expert weight row can be used as the w2 module."""

    def __init__(self, stacked: nn.Parameter, index: int) -> None:
        super().__init__()
        self.weight = stacked[index]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class DeepseekV41W8A8MoE(DeepseekV4MoE):
    """Ascend W8A8-export MoE (existing V4 kernel path, NPU-only).

    Keeps the bias-based ``moe_gating_top_k_hash`` routing and the grouped
    W8A8 expert GEMMs; the visual-token ``bias_vl`` selection bias of the
    official checkpoint is loaded but only applied by the torch MoE (the
    multimodal path lands with the vision phase).
    """

    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if cfg.n_hash_layers != 0:
            raise ValueError("DeepSeek-V4.1 removed hash routing; n_hash_layers must be 0")
        super().__init__(cfg, layer_id, dtype, device)
        # Deferred until the vision phase: the W8A8 expert path never reads
        # image_token_id or gate_bias_vl; both are staged for the torch-MoE
        # multimodal path only (see the class docstring).
        self.image_token_id = cfg.image_token_id
        self.gate_bias_vl = nn.Parameter(
            torch.empty(cfg.n_routed_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.gate_bias_vl_loaded = False


# ---------------------------------------------------------------------------
# Decoder layer + model
# ---------------------------------------------------------------------------


class DeepseekV41DecoderLayer(nn.Module):
    """DeepSeek-V4.1 decoder layer with the single-pass mHC shift.

    Attention collapses with the PREVIOUS block's ``ffn_pre`` and the FFN
    with THIS block's ``attn_pre``; the block returns its ``ffn_pre`` for the
    next block's attention (model.py:968-994).
    """

    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        mode: CSA2LayerMode,
        quant_mode: str = V41_QUANT_NONE,
        engram_layout: EngramLayout | None = None,
        engram_storage: str = ENGRAM_STORAGE_FP8,
        engram_gate_unrotate: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.mode = mode
        self.hc = DeepseekV41HyperConnection(cfg, dtype, device)
        self.input_layernorm = DeepseekV41RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.self_attn = DeepseekV41Attention(cfg, layer_id, dtype, device, mode, quant_mode)
        self.post_attention_layernorm = DeepseekV41RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        # Engram: gated n-gram lookup on the configured backbone layers only.
        self.engram: Engram | None = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            self.engram = Engram(
                engram_layout,
                engram_layout.layer_ids.index(layer_id),
                cfg.hidden_size,
                cfg.hc_mult,
                cfg.rms_norm_eps,
                cfg.tp_size,
                cfg.tp_rank,
                dtype,
                device,
                storage=engram_storage,
                gate_unrotate=engram_gate_unrotate,
            )
        # Dense vs MoE by first_k_dense_replace / moe_layer_freq.
        is_dense = (layer_id < cfg.first_k_dense_replace) or (
            cfg.moe_layer_freq > 1 and layer_id % cfg.moe_layer_freq != 0
        )
        if is_dense:
            if quant_mode == V41_QUANT_ASCEND:
                self.mlp: nn.Module = DeepseekV3MLP(
                    cfg,
                    cfg.moe_intermediate_size,
                    dtype,
                    device,
                    swiglu_limit=cfg.swiglu_limit,
                )
            else:
                self.mlp = DeepseekV41TorchMLP(cfg, dtype, device)
        elif quant_mode == V41_QUANT_ASCEND:
            self.mlp = DeepseekV41W8A8MoE(cfg, layer_id, dtype, device)
        else:
            self.mlp = DeepseekV41MoE(cfg, layer_id, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        pre_mix: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pre_mix is None:
            pre_mix = DeepseekV41HyperConnection.make_identity_pre_mix(hidden, self.hc.hc_mult)
        # Engram injection happens between the previous sublayer's post and
        # this block's pre, on the full hc stream, so the mix coefficients
        # see the injected stream (reference model.py:336-360).
        if self.engram is not None and engram_hashes is not None:
            hidden = self.engram(
                hidden,
                engram_hashes[:, self.engram.layer_hash_index],
                engram_mask,
            )
        residual = hidden
        attn_pre, attn_post, attn_comb = self.hc.hc_mixes(
            hidden, self.hc.hc_attn_fn, self.hc.hc_attn_scale, self.hc.hc_attn_base
        )
        x = self.hc.hc_pre(hidden, pre_mix)
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions, cos_sin_cache)
        hidden = self.hc.hc_post(x, residual, attn_post, attn_comb)

        residual = hidden
        ffn_pre, ffn_post, ffn_comb = self.hc.hc_mixes(
            hidden, self.hc.hc_ffn_fn, self.hc.hc_ffn_scale, self.hc.hc_ffn_base
        )
        x = self.hc.hc_pre(hidden, attn_pre)
        x = self.post_attention_layernorm(x)
        if isinstance(self.mlp, (DeepseekV41MoE, DeepseekV41W8A8MoE)):
            x = self.mlp(x, input_ids)
        else:
            x = self.mlp(x)
        hidden = self.hc.hc_post(x, residual, ffn_post, ffn_comb)
        return hidden, ffn_pre


class DeepseekV41Model(nn.Module):
    """DeepSeek-V4.1 transformer body.

    The layer loop threads the mHC ``pre_mix``: the residual stream starts
    as ``hc_mult`` identical copies of the embedding with a one-hot mix on
    copy 0, and after the last block the final merge is the learned
    ``hc_pre(h, last_ffn_pre)`` weighted sum (NOT a mean; model.py:1268-1269).
    Each layer selects its RoPE table by its CSA2 mode: ratio > 0 layers use
    the compress table (theta 160000 + YaRN), ratio-0 layers the plain
    theta-10000 table with YaRN off.
    """

    def __init__(
        self,
        cfg: DeepseekV41Config,
        dtype: torch.dtype,
        device: torch.device,
        quant_mode: str = V41_QUANT_NONE,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.quant_mode = quant_mode
        self.csa2_plan = CSA2LayerPlan.build(
            cfg.n_layers,
            cfg.compress_ratios,
            cfg.kv_source_layer_ids,
            cfg.index_source_layer_ids,
        )
        # Engram layout (None when the checkpoint has no engram layers).
        self.engram_layout: EngramLayout | None = None
        self.engram_hash: EngramHashState | None = None
        if cfg.engram_layer_ids:
            if cfg.engram_pad_token_id < 0:
                raise ValueError(
                    "DeepSeek-V4.1 engram is enabled (engram_layer_ids="
                    f"{cfg.engram_layer_ids}) but engram_pad_token_id is unset"
                )
            self.engram_layout = EngramLayout(
                cfg.engram_layer_ids,
                cfg.engram_num_embeddings,
                cfg.engram_max_ngram_size,
                cfg.engram_n_heads,
                cfg.engram_head_dim,
                cfg.engram_vocab_size,
                cfg.engram_compressed_vocab_size,
                cfg.engram_pad_token_id,
            )
            # The sidecar-derived buffers (token_map / primes / offsets /
            # multipliers) load on CPU; move them onto the model device.
            self.engram_hash = EngramHashState.from_config(
                self.engram_layout,
                {**(config or {}), "vocab_size": cfg.vocab_size},
            ).to(device)
        # Engram runtime format per checkpoint family: the ascend (W8A8)
        # export stores the tables as symmetric int8 + fp32 group scales and
        # rotates the residual stream (QuaRot), so the gate dot needs the
        # un-rotated stream; the official FP8/bf16 releases need neither.
        self.engram_storage = ENGRAM_STORAGE_INT8 if quant_mode == V41_QUANT_ASCEND else ENGRAM_STORAGE_FP8
        self.engram_gate_unrotate = None
        if cfg.engram_layer_ids and quant_mode == V41_QUANT_ASCEND:
            self.engram_gate_unrotate = load_quarot_gate_unrotate(str((config or {}).get("model_path") or ""), device)
        tp = cfg.tp_size
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // tp, tp, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(
                    cfg,
                    i,
                    dtype,
                    device,
                    self.csa2_plan.mode(i),
                    quant_mode,
                    self.engram_layout,
                    engram_storage=self.engram_storage,
                    engram_gate_unrotate=self.engram_gate_unrotate,
                )
                for i in range(cfg.n_layers)
            ]
        )
        self.norm = DeepseekV41RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        # RoPE tables: ratio-0 layers use theta 10000 with YaRN OFF; ratio>0
        # layers use the compress theta 160000 + YaRN (factor 16, original
        # 65536, beta 32/1, no temperature correction). Both are fp32,
        # matching the reference's complex64 tables (model.py:680-698).
        self.rotary = _PlainRotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            device,
        )
        self.compress_rotary = DeepseekV4RotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.compress_rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.original_max_position_embeddings,
            dtype=torch.float32,
            device=device,
        )

    def _layer_rope_cache(self, mode: CSA2LayerMode) -> torch.Tensor:
        """Per-layer RoPE table: compress theta for CSA2 layers, plain otherwise."""
        return self.compress_rotary.cos_sin_cache if mode.is_csa2 else self.rotary.cos_sin_cache

    def _compute_engram_hashes(
        self,
        backend: Any,
        metadata: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Hash the token stream once for every engram layer.

        Returns ``(hashes, gate_mask)`` or ``(None, None)`` while the SWA cache
        is unbound (graph memory profiling); the caller then skips engram
        entirely, matching the reference ``ensure_cache`` protocol.
        """
        assert self.engram_hash is not None
        dsa = getattr(metadata, "dsa_metadata", None)
        swa_group = backend.swa_group_metadata(dsa)
        if swa_group is None or not self.engram_hash.ensure_cache(swa_group.cache, swa_group.block_size):
            return None, None
        seq_lens_q = dsa.seq_lens_q.reshape(-1).to(torch.int64)
        q_cu = torch.cat((torch.zeros(1, dtype=torch.int64, device=seq_lens_q.device), seq_lens_q.cumsum(0)))
        dead = image_sentinel_mask(input_ids)
        hashes = self.engram_hash(
            input_ids,
            positions,
            q_cu,
            dead,
            None,
            swa_group.slot_mapping,
            swa_group.block_table,
            swa_group.block_size,
        )
        return hashes, ~dead

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.cfg.cp_size > 1:
            raise NotImplementedError(
                "DeepSeek-V4.1 context parallelism is scheduled with the Phase-6 distributed work"
            )
        hidden = self.embed_tokens(input_ids)
        positions = positions.to(torch.int64).contiguous()
        context = get_forward_context()
        backend = context.attention_backend
        metadata = context.metadata
        graph_mode = bool(getattr(metadata, "dsa_graph_mode", False))
        if graph_mode and getattr(metadata, "dsa_metadata", None) is not None:
            # ACL graph replay reuses the metadata captured during warmup.
            pass
        else:
            backend.reset_forward(metadata)
            metadata.dsa_graph_mode = graph_mode
            metadata.dsa_positions = positions
            prepare_dsa = getattr(backend, "prepare_dsa_metadata_for_forward", None)
            if prepare_dsa is None:
                raise RuntimeError("DeepSeek-V4.1 requires prepare_dsa_metadata_for_forward")
            prepare_dsa(metadata)
            if graph_mode:
                metadata.dsa_graph_mode = True

        # Engram: compute the n-gram hashes once, on the full token stream,
        # before the layer loop; each engram layer then pre-gathers its rows.
        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if self.engram_hash is not None:
            engram_hashes, engram_mask = self._compute_engram_hashes(backend, metadata, input_ids, positions)
            if engram_hashes is not None:
                for layer in self.layers:
                    engram = getattr(layer, "engram", None)
                    if engram is not None:
                        engram.prepare_embeddings(engram_hashes[:, engram.layer_hash_index].to(torch.int64))

        # Expand to hc_mult identical copies; the first pre_mix is one-hot on
        # copy 0 (model.py:1258-1260).
        hidden = hidden.unsqueeze(1).expand(-1, self.cfg.hc_mult, -1).contiguous()
        pre_mix = DeepseekV41HyperConnection.make_identity_pre_mix(hidden, self.cfg.hc_mult)
        for layer_id, layer in enumerate(self.layers):
            layer_cos_sin_cache = self._layer_rope_cache(self.csa2_plan.mode(layer_id))
            hidden, pre_mix = layer(
                hidden,
                positions,
                layer_cos_sin_cache,
                input_ids,
                pre_mix,
                engram_hashes,
                engram_mask,
            )
            record_layer_event(layer_id)
        # Final merge: the learned weighted sum with the LAST block's ffn_pre.
        merged = self.layers[-1].hc.hc_pre(hidden, pre_mix)
        return self.norm(merged)


class _PlainRotaryEmbedding(nn.Module):
    """Plain (no YaRN) RoPE table for the ratio-0 layers."""

    def __init__(
        self,
        rotary_dim: int,
        max_positions: int,
        theta: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "cos_sin_cache",
            _build_plain_cos_sin_cache(rotary_dim, max_positions, theta, device),
            persistent=False,
        )


# ---------------------------------------------------------------------------
# Causal LM: mode-aware construction + weight loading
# ---------------------------------------------------------------------------


class DeepseekV41ForCausalLM(PyModelBase):
    """DeepSeek-V4.1 causal LM driven by the C++ PyCausalLM bridge."""

    # ``_load_dsv4_moe`` is borrowed from ``DeepseekV4ForCausalLM`` by calling
    # it unbound with this instance (V4.1 does not inherit the V4 LM), and it
    # now dispatches its routing half through ``self._load_dsv4_moe_routing``.
    # Re-expose that static helper so the borrowed loader resolves it.
    _load_dsv4_moe_routing = staticmethod(DeepseekV4ForCausalLM._load_dsv4_moe_routing)

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = DeepseekV41Config.from_dict(config)
        self.quant_mode = _detect_v41_quant_mode(config)
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0"))
        self.model = DeepseekV41Model(self.cfg, dtype, device, self.quant_mode, config)
        tp = self.cfg.tp_size
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    # -- weight loading ---------------------------------------------------------

    def load_weights(self, state_dicts: list[StateDict], tp_rank: int, tp_size: int) -> None:
        if self.quant_mode == V41_QUANT_ASCEND:
            self._load_weights_ascend(state_dicts, tp_rank, tp_size)
        else:
            self._load_weights_torch(state_dicts, tp_rank, tp_size)

    def _load_weights_torch(self, state_dicts: list[StateDict], tp_rank: int, tp_size: int) -> None:
        """fp8-block / bf16 loading: dequantized bf16 plain-linear modules."""
        cfg = self.cfg
        loader = W8A8WeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)
        is_fp8 = self.quant_mode == V41_QUANT_FP8

        def _has(name: str) -> bool:
            return loader.has(name)

        def _linear(ckpt_prefix: str, param_prefix: str, shard_dim: int | None = None) -> None:
            if is_fp8:
                weight = loader.load_tensor(ckpt_prefix + ".weight")
                scale = loader.load_tensor(ckpt_prefix + ".scale")
                weight = dequant_fp8_block(weight, scale)
            else:
                weight = loader.load_tensor(ckpt_prefix + ".weight")
            if shard_dim is not None:
                weight = loader.shard(weight, dim=shard_dim)
            loader.copy_in(param_prefix + ".weight", weight)

        loader.copy_in(
            "model.embed_tokens.weight",
            loader.shard(loader.load_tensor("embed.weight"), dim=1),
        )
        for i in range(cfg.n_layers):
            ck = f"layers.{i}."
            pm = f"model.layers.{i}."
            attn = self.model.layers[i].self_attn
            # The five attention projections are constructed unconditionally by
            # DeepseekV41Attention, so a missing checkpoint tensor is a broken
            # export (not an optional module): fail instead of running with
            # uninitialized weights.
            for ckpt_name, param_name, shard_dim in (
                ("attn.wq_a", "self_attn.q_a_proj", None),
                ("attn.wq_b", "self_attn.q_b_proj", 0),
                ("attn.wkv", "self_attn.kv_proj", None),
                ("attn.wo_a", "self_attn.o_a_proj", 0),
                ("attn.wo_b", "self_attn.o_b_proj", 1),
            ):
                weight_key = ck + ckpt_name + ".weight"
                if not _has(weight_key):
                    raise KeyError(
                        f"required DeepSeek-V4.1 attention weight '{weight_key}' "
                        f"is missing from the checkpoint (decoder layer {i})"
                    )
                _linear(ck + ckpt_name, pm + param_name, shard_dim)
            loader.copy_in(
                pm + "self_attn.q_a_layernorm.weight",
                loader.load_tensor(ck + "attn.q_norm.weight"),
            )
            loader.copy_in(
                pm + "self_attn.kv_a_layernorm.weight",
                loader.load_tensor(ck + "attn.kv_norm.weight"),
            )
            sink_key = ck + "attn.attn_sink"
            if not _has(sink_key):
                sink_key = ck + "attn.attn_sink.weight"
            if _has(sink_key):
                sink = loader.load_tensor(sink_key)
                if cfg.tp_size > 1:
                    # Per-head sink sliced to this rank's q head range.
                    sink = loader.shard(sink, dim=0)
                loader.copy_in(pm + "self_attn.attn_sink", sink)
                attn.attn_sink_loaded = True
            self._load_common_layer_norms(loader, ck, pm)
            # Main-KV compressor (Full layers; no gate at ratio 1).
            if hasattr(attn, "cmp_wkv") and _has(ck + "attn.compressor.wkv.weight"):
                loader.copy_in(
                    pm + "self_attn.cmp_wkv.weight",
                    loader.load_tensor(ck + "attn.compressor.wkv.weight"),
                )
                if hasattr(attn, "cmp_wgate") and _has(ck + "attn.compressor.wgate.weight"):
                    loader.copy_in(
                        pm + "self_attn.cmp_wgate.weight",
                        loader.load_tensor(ck + "attn.compressor.wgate.weight"),
                    )
                loader.copy_in(
                    pm + "self_attn.cmp_norm.weight",
                    loader.load_tensor(ck + "attn.compressor.norm.weight"),
                )
            # Indexer (Full + Reindex layers; wk/k_norm only on kv sources).
            if attn.indexer is not None and _has(ck + "attn.indexer.wq_b.weight"):
                _linear(ck + "attn.indexer.wq_b", pm + "self_attn.indexer.wq_b")
                loader.copy_in(
                    pm + "self_attn.indexer.weights_proj.weight",
                    loader.load_tensor(ck + "attn.indexer.weights_proj.weight"),
                )
                if attn.indexer.owns_k and _has(ck + "attn.indexer.wk.weight"):
                    loader.copy_in(
                        pm + "self_attn.indexer.wk.weight",
                        loader.load_tensor(ck + "attn.indexer.wk.weight"),
                    )
                    loader.copy_in(
                        pm + "self_attn.indexer.k_norm.weight",
                        loader.load_tensor(ck + "attn.indexer.k_norm.weight"),
                    )
            mlp = self.model.layers[i].mlp
            # Dispatch on the constructed module type, not on checkpoint
            # presence: a MoE layer always requires its gate + experts +
            # shared expert, so a missing key must surface from
            # _load_torch_moe instead of silently leaving the MLP at
            # uninitialized values.
            if isinstance(mlp, DeepseekV41MoE):
                self._load_torch_moe(loader, ck, pm, i, is_fp8)
            elif isinstance(mlp, DeepseekV41TorchMLP):
                for name in ("w1", "w2", "w3"):
                    _linear(ck + f"ffn.{name}", pm + f"mlp.{name}")
            # Engram (only the configured engram layers carry the module).
            if self.model.layers[i].engram is not None:
                self._load_engram_torch(loader, ck, pm, i, is_fp8)
        loader.copy_in("model.norm.weight", loader.load_tensor("norm.weight"))
        self._load_lm_head(loader)

    def _load_torch_moe(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        layer_id: int,
        is_fp8: bool,
    ) -> None:
        """Load gate + bias + stacked experts + shared expert of one torch MoE layer."""
        mlp = self.model.layers[layer_id].mlp
        loader.copy_in(pm + "mlp.gate.weight", loader.load_tensor(ck + "ffn.gate.weight"))
        loader.copy_in(
            pm + "mlp.e_score_correction_bias",
            loader.load_tensor(ck + "ffn.gate.bias"),
        )
        if loader.has(ck + "ffn.gate.bias_vl"):
            loader.copy_in(pm + "mlp.gate_bias_vl", loader.load_tensor(ck + "ffn.gate.bias_vl"))
            mlp.gate_bias_vl_loaded = True
        for local_slot in range(mlp.num_experts_per_rank):
            expert = mlp.start_expert_id + local_slot
            for name, param in (
                ("w1", mlp.experts_w1),
                ("w2", mlp.experts_w2),
                ("w3", mlp.experts_w3),
            ):
                prefix = ck + f"ffn.experts.{expert}.{name}"
                if is_fp8:
                    weight = dequant_fp4_group(
                        loader.load_tensor(prefix + ".weight"),
                        loader.load_tensor(prefix + ".scale"),
                    )
                else:
                    weight = loader.load_tensor(prefix + ".weight")
                param.data[local_slot].copy_(weight.to(param.dtype))
        shared_prefix = ck + "ffn.shared_experts."
        for name, module_name in (
            ("w1", "shared_w1"),
            ("w2", "shared_w2"),
            ("w3", "shared_w3"),
        ):
            prefix = shared_prefix + name
            if is_fp8:
                weight = dequant_fp8_block(
                    loader.load_tensor(prefix + ".weight"),
                    loader.load_tensor(prefix + ".scale"),
                )
            else:
                weight = loader.load_tensor(prefix + ".weight")
            loader.copy_in(pm + f"mlp.{module_name}.weight", weight)

    def _load_engram_torch(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        layer_id: int,
        is_fp8: bool,
    ) -> None:
        """Load one engram layer (official FP8 / bf16 checkpoint layout).

        The hash table stays quantized in memory: FP8-E4M3 payloads plus
        ue8m0 per-32 scales are narrowed to this rank's head-bucket row range
        (a 3.8e8-row table must not be materialized in bf16), while the small
        ``wkv`` projection is dequantized at load like every other linear.
        """
        eng = self.model.layers[layer_id].engram
        assert eng is not None
        table = loader.load_tensor(ck + "engram.embed.weight")
        table_u8 = table.view(torch.uint8) if table.element_size() == 1 else table
        scale = loader.load_tensor(ck + "engram.embed.scale")
        scale_u8 = scale.view(torch.uint8) if scale.element_size() == 1 else scale
        start = eng.embed_tokens.vocab_start
        rows = eng.embed_tokens.part_num_embeddings
        if table_u8.size(0) < eng.embed_tokens.num_embeddings:
            raise ValueError(
                f"engram table for layer {layer_id} has {table_u8.size(0)} rows, "
                f"config declares {eng.embed_tokens.num_embeddings}"
            )
        if rows > 0:
            # Direct H2D copy_ from the mmap view (no device temporary).
            eng.embed_tokens.weight.data.copy_(table_u8.narrow(0, start, rows))
            eng.embed_tokens.weight_scale.data.copy_(scale_u8.narrow(0, start, rows))
        if is_fp8:
            wkv = dequant_fp8_block(
                loader.load_tensor(ck + "engram.wkv.weight"),
                loader.load_tensor(ck + "engram.wkv.scale"),
            )
        else:
            wkv = loader.load_tensor(ck + "engram.wkv.weight")
        loader.copy_in(pm + "engram.wkv.weight", wkv)
        loader.copy_in(pm + "engram.q_weight", loader.load_tensor(ck + "engram.q_weight"))
        loader.copy_in(pm + "engram.k_weight", loader.load_tensor(ck + "engram.k_weight"))

    def _load_weights_ascend(self, state_dicts: list[StateDict], tp_rank: int, tp_size: int) -> None:
        """W8A8 export loading (existing V4 W8A8DynamicLinear path).

        The Eco-Tech W8A8 export ships the engram tables as symmetric int8 +
        fp32 group-32 scales and keeps engram wkv/q/k in bf16; the QuaRot
        trunk rotation is fully folded into the weights, so only the engram
        gate needs the runtime un-rotation (see Engram.forward).
        """
        cfg = self.cfg
        loader = W8A8WeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)

        def _has(name: str) -> bool:
            return loader.has(name)

        def _w8a8(
            ckpt_prefix: str,
            param_prefix: str,
            shard_dims: dict | None = None,
            required: bool = False,
        ) -> None:
            """Load one W8A8-dynamic projection (weight + scale + offset).

            Delegates to :func:`load_w8a8_dynamic_projection` so a missing
            ``weight_offset`` is explicitly zero-filled: the symmetric int8
            export omits it, and ``W8A8DynamicLinear.weight_offset`` otherwise
            stays at its ``torch.empty`` (garbage) initialization, failing the
            all-zero check in ``process_weights_after_loading``. A present
            non-zero offset is copied verbatim and still rejected there.
            ``required`` marks the projections
            :class:`DeepseekV41Attention` constructs unconditionally, so a
            missing ``weight`` / ``weight_scale`` raises.
            """
            load_w8a8_dynamic_projection(
                loader,
                ckpt_prefix,
                param_prefix,
                shard_dims,
                require_weight_and_scale=required,
                fill_missing_offset=True,
                description=f"DeepSeek-V4.1 W8A8 {ckpt_prefix}",
            )

        loader.copy_in(
            "model.embed_tokens.weight",
            loader.shard(loader.load_tensor("embed.weight"), dim=1),
        )
        for i in range(cfg.n_layers):
            ck = f"layers.{i}."
            pm = f"model.layers.{i}."
            attn = self.model.layers[i].self_attn
            _w8a8(ck + "attn.wq_a", pm + "self_attn.q_a_proj", required=True)
            _w8a8(
                ck + "attn.wq_b",
                pm + "self_attn.q_b_proj",
                {"weight": 0, "weight_scale": 0, "weight_offset": 0},
                required=True,
            )
            _w8a8(ck + "attn.wkv", pm + "self_attn.kv_proj", required=True)
            loader.copy_in(
                pm + "self_attn.o_a_proj.weight",
                loader.shard(loader.load_tensor(ck + "attn.wo_a.weight"), dim=0),
            )
            loader.copy_in(
                pm + "self_attn.o_b_proj.weight",
                loader.shard(loader.load_tensor(ck + "attn.wo_b.weight"), dim=1),
            )
            loader.copy_in(
                pm + "self_attn.q_a_layernorm.weight",
                loader.load_tensor(ck + "attn.q_norm.weight"),
            )
            loader.copy_in(
                pm + "self_attn.kv_a_layernorm.weight",
                loader.load_tensor(ck + "attn.kv_norm.weight"),
            )
            sink_key = ck + "attn.attn_sink"
            if not _has(sink_key):
                sink_key = ck + "attn.attn_sink.weight"
            if _has(sink_key):
                sink = loader.load_tensor(sink_key)
                if sink.dim() == 1 and sink.size(0) == cfg.n_heads and cfg.tp_size > 1:
                    shard_size = cfg.n_heads // cfg.tp_size
                    sink = sink.narrow(0, cfg.tp_rank * shard_size, shard_size)
                loader.copy_in(pm + "self_attn.attn_sink", sink)
                attn.attn_sink_loaded = True
            self._load_common_layer_norms(loader, ck, pm)
            if hasattr(attn, "cmp_wkv") and _has(ck + "attn.compressor.wkv.weight"):
                loader.copy_in(
                    pm + "self_attn.cmp_wkv.weight",
                    loader.load_tensor(ck + "attn.compressor.wkv.weight"),
                )
                if hasattr(attn, "cmp_wgate") and _has(ck + "attn.compressor.wgate.weight"):
                    loader.copy_in(
                        pm + "self_attn.cmp_wgate.weight",
                        loader.load_tensor(ck + "attn.compressor.wgate.weight"),
                    )
                loader.copy_in(
                    pm + "self_attn.cmp_norm.weight",
                    loader.load_tensor(ck + "attn.compressor.norm.weight"),
                )
            if attn.indexer is not None and _has(ck + "attn.indexer.wq_b.weight"):
                _w8a8(ck + "attn.indexer.wq_b", pm + "self_attn.indexer.wq_b")
                loader.copy_in(
                    pm + "self_attn.indexer.weights_proj.weight",
                    loader.load_tensor(ck + "attn.indexer.weights_proj.weight"),
                )
                if attn.indexer.owns_k and _has(ck + "attn.indexer.wk.weight"):
                    loader.copy_in(
                        pm + "self_attn.indexer.wk.weight",
                        loader.load_tensor(ck + "attn.indexer.wk.weight"),
                    )
                    loader.copy_in(
                        pm + "self_attn.indexer.k_norm.weight",
                        loader.load_tensor(ck + "attn.indexer.k_norm.weight"),
                    )
            attn.process_weights_after_loading()
            mlp = self.model.layers[i].mlp
            # Dispatch on the constructed module type, not on checkpoint
            # presence: the experts are mandatory for a W8A8 MoE layer, so a
            # missing expert tensor must surface from _load_dsv4_moe instead of
            # silently skipping the whole MLP.
            if isinstance(mlp, DeepseekV41W8A8MoE):
                self._load_ascend_moe(loader, ck, pm, i)
                mlp.process_weights_after_loading()
            elif isinstance(mlp, DeepseekV3MLP):
                DeepseekV4ForCausalLM._load_dsv4_dense_mlp(loader, ck, pm, mlp)
            # Engram (Eco-Tech export: int8 table + bf16 wkv/q/k).
            if self.model.layers[i].engram is not None:
                self._load_engram_ascend(loader, ck, pm, i)
        loader.copy_in("model.norm.weight", loader.load_tensor("norm.weight"))
        self._load_lm_head(loader)

    def _load_engram_ascend(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        layer_id: int,
    ) -> None:
        """Load one engram layer from the Eco-Tech W8A8 export.

        The hash table stays int8 in memory (payload [rows, 256] int8 + one
        fp32 scale per 32-column group) and is narrowed to this rank's
        head-bucket row range; ``wkv`` / ``q_weight`` / ``k_weight`` are bf16
        and copied verbatim (the K half and gate stay in the original basis,
        the V half is pre-folded with the QuaRot rotation).
        """
        eng = self.model.layers[layer_id].engram
        assert eng is not None
        table = loader.load_tensor(ck + "engram.embed.weight")
        if table.dtype == torch.uint8:
            table = table.view(torch.int8)
        if table.dtype != torch.int8:
            raise NotImplementedError(
                f"engram table for layer {layer_id} has dtype {table.dtype}; the "
                "ascend loader implements the int8_sym export only (fp32 group "
                "scales, see quant_model_description.json optional.embedding_storage)"
            )
        scale = loader.load_tensor(ck + "engram.embed.scale")
        if scale.dtype != torch.float32:
            raise NotImplementedError(
                f"engram table scale for layer {layer_id} has dtype {scale.dtype}; "
                "expected float32 group scales of the int8_sym export"
            )
        start = eng.embed_tokens.vocab_start
        rows = eng.embed_tokens.part_num_embeddings
        if table.size(0) < eng.embed_tokens.num_embeddings:
            raise ValueError(
                f"engram table for layer {layer_id} has {table.size(0)} rows, "
                f"config declares {eng.embed_tokens.num_embeddings}"
            )
        if rows > 0:
            # Direct H2D copy_ from the mmap view; a prior .to(device) would
            # materialize a 12+GB device temporary on top of the parameter.
            eng.embed_tokens.weight.data.copy_(table.narrow(0, start, rows))
            eng.embed_tokens.weight_scale.data.copy_(scale.narrow(0, start, rows))
        loader.copy_in(pm + "engram.wkv.weight", loader.load_tensor(ck + "engram.wkv.weight"))
        loader.copy_in(pm + "engram.q_weight", loader.load_tensor(ck + "engram.q_weight"))
        loader.copy_in(pm + "engram.k_weight", loader.load_tensor(ck + "engram.k_weight"))

    def _load_ascend_moe(self, loader: W8A8WeightLoader, ck: str, pm: str, layer_id: int) -> None:
        """Ascend MoE: the V4 W8A8 staging plus the V4.1 ``gate.bias_vl``."""
        # The W8A8 export stores the MoE block under either the native
        # ``ffn.`` or the legacy ``mlp.`` prefix; resolve it once here and
        # share it with the shared V4 loader and the V4.1-only ``bias_vl``.
        source_prefix = _find_checkpoint_prefix(
            loader,
            (ck + "ffn.", ck + "mlp."),
            ("experts.0.w1.weight",),
        )
        if source_prefix is None:
            raise KeyError(f"DeepSeek-V4.1 layer {layer_id} MoE expert weights not found")
        DeepseekV4ForCausalLM._load_dsv4_moe(self, loader, ck, pm, layer_id, source_prefix)
        mlp = self.model.layers[layer_id].mlp
        bias_vl_key = source_prefix + "gate.bias_vl"
        if loader.has(bias_vl_key):
            loader.copy_in(pm + "mlp.gate_bias_vl", loader.load_tensor(bias_vl_key))
            # Deferred until the vision phase; loaded only for checkpoint
            # compatibility. The W8A8 expert path never reads gate_bias_vl.
            mlp.gate_bias_vl_loaded = True

    def _load_common_layer_norms(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
    ) -> None:
        """Layer norms + mHC weights shared by every checkpoint mode."""
        loader.copy_in(
            pm + "input_layernorm.weight",
            loader.load_tensor(ck + "attn_norm.weight"),
        )
        loader.copy_in(
            pm + "post_attention_layernorm.weight",
            loader.load_tensor(ck + "ffn_norm.weight"),
        )
        for part in ("attn", "ffn"):
            for suffix in ("fn", "scale", "base"):
                name = f"hc_{part}_{suffix}"
                loader.copy_in(pm + "hc." + name, loader.load_tensor(ck + name))

    def _load_lm_head(self, loader: W8A8WeightLoader) -> None:
        """Load the output head under whichever alias the export uses."""
        lm_head_key = next(
            (
                name
                for name in ("lm_head.weight", "model.lm_head.weight", "model.head.weight", "head.weight")
                if loader.has(name)
            ),
            None,
        )
        assert lm_head_key is not None, "checkpoint output-head weight not found"
        loader.copy_in(
            "lm_head.weight",
            loader.shard(loader.load_tensor(lm_head_key), dim=0),
        )


# ---------------------------------------------------------------------------
# Pure-torch quantization helpers for the official checkpoint
#
# Reference recipes from the official DeepSeek-V4.1 Flash inference/kernel.py:
# ue8m0 scale decoding, 32x32-block FP8 E4M3 weight dequant, MXFP4
# (E2M1 + per-32 ue8m0 group scale) expert dequant, and the dynamic FP8/FP4
# activation QDQ variants of the reference inference/kernel.py. Leaf section:
# depends only on torch and the stdlib (imported once at the module head),
# device-agnostic (CPU and NPU).
# ---------------------------------------------------------------------------

_FP8_E4M3_MAX = 448.0
_FP4_E2M1_MAX = 6.0
_FP8_AMAX_FLOOR = 1e-4
_FP4_E8M0_AMAX_FLOOR = 6.0 * (2.0**-126)
_FP4_E4M3_AMAX_FLOOR = 6.0 * (2.0**-9)

# IEEE 754 bits of the float32 subnormal 2**-127 (ue8m0 byte 0x00).
_UE8M0_ZERO_BITS = 0x00400000

_FP4_NIBBLE_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_E2M1_POSITIVE_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Round-half-to-even index on the positive grid for each exact midpoint
# (0.75 -> 1.0, 1.75 -> 2.0, 3.5 -> 4.0, ... : ties go to the even mantissa).
_E2M1_MIDPOINT_EVEN_INDEX = (0, 2, 2, 4, 4, 6, 6)


def ue8m0_to_scale(scale_u8: torch.Tensor) -> torch.Tensor:
    """Decodes ue8m0 scale bytes into float32 powers of two.

    Byte ``b`` decodes to ``2 ** (b - 127)``. Byte ``0x00`` is the subnormal
    ``2 ** -127`` (the format has no zero encoding) and byte ``0xFF`` (NaN)
    is invalid.

    Args:
        scale_u8: tensor of raw ue8m0 bytes; uint8, or any 1-byte dtype
            (int8 / float8_e8m0fnu / float4_e2m1fn_x2) reinterpreted as uint8.

    Returns:
        Float32 tensor of the same shape holding ``2 ** (b - 127)``.
    """
    if scale_u8.dtype == torch.uint8:
        raw = scale_u8
    elif scale_u8.element_size() == 1:
        raw = scale_u8.view(torch.uint8)
    else:
        raise ValueError(f"expected 1-byte ue8m0 scale storage, got {scale_u8.dtype}")
    raw_i32 = raw.to(torch.int32)
    if bool(torch.any(raw_i32 == 0xFF)):
        raise ValueError("ue8m0 scale byte 0xFF (NaN) is invalid")
    bits = torch.where(
        raw_i32 == 0,
        torch.full_like(raw_i32, _UE8M0_ZERO_BITS),
        raw_i32 << 23,
    )
    return bits.contiguous().view(torch.float32)


def dequant_fp8_block(
    weight_e4m3: torch.Tensor,
    scale_u8: torch.Tensor,
    block: tuple[int, int] = (32, 32),
) -> torch.Tensor:
    """Dequantizes block-wise FP8 E4M3 weights to bf16.

    ``W[n, k] = fp8_value(weight[n, k]) * scale[n // block_out, k // block_in]``
    with one ue8m0 power-of-two scale per output-by-input block (the official
    checkpoint uses 32x32 blocks and ``[ceil(out/32), ceil(in/32)]`` scales).

    Args:
        weight_e4m3: ``[out, in]`` float8_e4m3fn weight tensor.
        scale_u8: ``[ceil(out / block_out), ceil(in / block_in)]`` ue8m0 bytes.
        block: ``(block_out, block_in)`` block shape.

    Returns:
        Bfloat16 ``[out, in]`` dequantized weight.
    """
    if weight_e4m3.ndim != 2:
        raise ValueError(f"weight must be 2-D [out, in], got ndim={weight_e4m3.ndim}")
    if len(block) != 2 or block[0] <= 0 or block[1] <= 0:
        raise ValueError(f"block must be a pair of positive ints, got {block}")
    block_out, block_in = int(block[0]), int(block[1])
    out_dim, in_dim = weight_e4m3.shape
    expected_shape = (
        (out_dim + block_out - 1) // block_out,
        (in_dim + block_in - 1) // block_in,
    )
    if tuple(scale_u8.shape) != expected_shape:
        raise ValueError(
            f"scale shape {tuple(scale_u8.shape)} does not match expected "
            f"{expected_shape} for weight [{out_dim}, {in_dim}] with block {block}"
        )
    scale = ue8m0_to_scale(scale_u8)
    scale = scale.repeat_interleave(block_out, dim=0)
    scale = scale.repeat_interleave(block_in, dim=1)[:out_dim, :in_dim]
    return (_e4m3_to_float(weight_e4m3) * scale).to(torch.bfloat16)


def fp4_nibble_table(device: Optional[Union[torch.device, str]] = None) -> torch.Tensor:
    """Returns the 16-value E2M1 (MXFP4) decode table indexed by nibble.

    Nibble bit 3 is the sign; both ``0x0`` and ``0x8`` decode to (negative)
    zero. There is no Inf or NaN in the finite-only E2M1 format.
    """
    return torch.tensor(_FP4_NIBBLE_VALUES, dtype=torch.float32, device=device)


def unpack_fp4(
    weight_i8: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Unpacks packed MXFP4 nibbles into ``[N, K]`` values.

    The ``[..., K / 2]`` 1-byte storage (int8 viewed as uint8) holds two E2M1
    nibbles per byte along the last dim: the LOW nibble is element ``2j`` and
    the HIGH nibble is element ``2j + 1`` (matches reference ``convert.py``
    and the torch ``float4_e2m1fn_x2`` layout).

    Args:
        weight_i8: ``[..., K / 2]`` packed fp4 bytes.
        dtype: output dtype (default bf16).

    Returns:
        ``[..., K]`` tensor of decoded E2M1 values.
    """
    if weight_i8.dtype == torch.uint8:
        packed = weight_i8
    elif weight_i8.element_size() == 1:
        packed = weight_i8.view(torch.uint8)
    else:
        raise ValueError(f"expected 1-byte packed fp4 storage, got {weight_i8.dtype}")
    table = fp4_nibble_table(device=packed.device)
    low = table[(packed & 0x0F).to(torch.int64)]
    high = table[(packed >> 4).to(torch.int64)]
    unpacked = torch.stack((low, high), dim=-1).flatten(-2)
    return unpacked.to(dtype)


def dequant_fp4_group(
    weight_i8: torch.Tensor,
    scale_u8: torch.Tensor,
    group: int = 32,
) -> torch.Tensor:
    """Dequantizes MXFP4 expert weights to bf16.

    ``W[n, k] = e2m1(nibble) * 2 ** (S[n, k // group] - 127)`` with one ue8m0
    scale per row per ``group`` consecutive input elements (the official
    checkpoint uses ``[out, in / 32]`` scales).

    Args:
        weight_i8: ``[N, K / 2]`` packed fp4 bytes.
        scale_u8: ``[N, ceil(K / group)]`` ue8m0 scale bytes.
        group: number of input elements sharing one scale.

    Returns:
        Bfloat16 ``[N, K]`` dequantized weight.
    """
    if group <= 0:
        raise ValueError(f"group must be positive, got {group}")
    unpacked = unpack_fp4(weight_i8, dtype=torch.float32)
    in_dim = unpacked.size(-1)
    expected_groups = (in_dim + group - 1) // group
    if scale_u8.size(-1) != expected_groups:
        raise ValueError(
            f"scale has {scale_u8.size(-1)} groups, expected {expected_groups} for K={in_dim} with group={group}"
        )
    scale = ue8m0_to_scale(scale_u8)
    scale = scale.repeat_interleave(group, dim=-1)[..., :in_dim]
    return (unpacked * scale).to(torch.bfloat16)


def fp8_act_qdq(x: torch.Tensor, group: int = 32) -> torch.Tensor:
    """Dynamic per-(row, group) FP8-E4M3 QDQ with ue8m0 power-of-2 scales.

    Matches the reference ``act_quant`` with ``scale_fmt="ue8m0"``:
    ``amax = max(|x|, 1e-4)`` per group,
    ``s = 2 ** ceil(log2(amax * (1 / 448)))`` (exact power of two), then
    ``y = fp8_e4m3(clamp(x / s, -448, 448)) * s``. Every step is exact, so
    outputs are FP8-representable values scaled by ``s``.

    Args:
        x: tensor whose last dim is divisible by ``group``.
        group: number of elements sharing one scale.

    Returns:
        Tensor of the original dtype.
    """
    orig_dtype = x.dtype
    grouped = _grouped(x.float().contiguous(), group)
    amax = grouped.abs().amax(dim=-1).clamp_min(_FP8_AMAX_FLOOR)
    scale = _pow2(_log2_ceil(amax * (1.0 / _FP8_E4M3_MAX)))
    normalized = torch.clamp(grouped / scale.unsqueeze(-1), -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    y = _round_e4m3(normalized) * scale.unsqueeze(-1)
    return y.flatten(-2).to(orig_dtype)


def fp4_act_qdq_e8m0(x: torch.Tensor, group: int = 32) -> torch.Tensor:
    """Dynamic per-(row, group) MXFP4 QDQ with ue8m0 power-of-2 scales.

    Matches the reference ``fp4_act_quant`` with e8m0 scales (indexer Q/K):
    ``amax = max(|x|, 6 * 2**-126)`` per group,
    ``s = 2 ** ceil(log2(amax * (1 / 6)))``, then
    ``y = e2m1(clamp(x / s, -6, 6)) * s`` with round-to-nearest-even on the
    E2M1 grid.

    Args:
        x: tensor whose last dim is divisible by ``group``.
        group: number of elements sharing one scale.

    Returns:
        Tensor of the original dtype.
    """
    orig_dtype = x.dtype
    grouped = _grouped(x.float().contiguous(), group)
    amax = grouped.abs().amax(dim=-1).clamp_min(_FP4_E8M0_AMAX_FLOOR)
    scale = _pow2(_log2_ceil(amax * (1.0 / _FP4_E2M1_MAX)))
    normalized = torch.clamp(grouped / scale.unsqueeze(-1), -_FP4_E2M1_MAX, _FP4_E2M1_MAX)
    y = _round_e2m1(normalized) * scale.unsqueeze(-1)
    return y.flatten(-2).to(orig_dtype)


def fp4_act_qdq_e4m3_scale(x: torch.Tensor, group: int = 16) -> torch.Tensor:
    """Dynamic per-(row, group) MXFP4 QDQ with e4m3 scales (compressed KV).

    Matches the reference ``fp4_act_quant`` with e4m3 scales:
    ``amax = max(|x|, 6 * 2**-9)`` per group,
    ``s = float8_e4m3fn(amax / 6)`` (round-to-nearest, NOT a power of two),
    then ``y = e2m1(clamp(x / s, -6, 6)) * s``.

    Args:
        x: tensor whose last dim is divisible by ``group``.
        group: number of elements sharing one scale.

    Returns:
        Tensor of the original dtype.
    """
    orig_dtype = x.dtype
    grouped = _grouped(x.float().contiguous(), group)
    amax = grouped.abs().amax(dim=-1).clamp_min(_FP4_E4M3_AMAX_FLOOR)
    scale = _round_e4m3(amax / _FP4_E2M1_MAX)
    normalized = torch.clamp(grouped / scale.unsqueeze(-1), -_FP4_E2M1_MAX, _FP4_E2M1_MAX)
    y = _round_e2m1(normalized) * scale.unsqueeze(-1)
    return y.flatten(-2).to(orig_dtype)


def _log2_ceil(x: torch.Tensor) -> torch.Tensor:
    """Computes ceil(log2(x)) for positive normal float32 via bit inspection."""
    bits = x.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    has_mantissa = (bits & 0x7FFFFF) != 0
    return exponent - 127 + has_mantissa.to(torch.int32)


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    """Computes ``2 ** exponent`` for int32 exponents >= -126 via bit construction."""
    bits = (exponent + 127) << 23
    return bits.contiguous().view(torch.float32)


def _grouped(x: torch.Tensor, group: int) -> torch.Tensor:
    """Views the last dim as (..., groups, group); validates divisibility."""
    if group <= 0:
        raise ValueError(f"group must be positive, got {group}")
    if x.size(-1) % group != 0:
        raise ValueError(f"last dim {x.size(-1)} is not divisible by group {group}")
    return x.unflatten(-1, (-1, group))


_E2M1_TABLE_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _e2m1_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Device-cached E2M1 rounding tables.

    ``torch.tensor(..., device=npu)`` is a synchronous H2D copy; building the
    tables per call is both an eager-mode stall and a hard error inside ACL
    graph capture (rtMemcpy 107030 on a captured stream), so cache one set
    per device. The eager first call performs the copy legally; captured
    forwards reuse the cached device tensors.
    """
    cached = _E2M1_TABLE_CACHE.get(device)
    if cached is None:
        cached = (
            torch.tensor(_E2M1_POSITIVE_VALUES, dtype=torch.float32, device=device),
            torch.tensor(_E2M1_MIDPOINTS, dtype=torch.float32, device=device),
            torch.tensor(_E2M1_MIDPOINT_EVEN_INDEX, dtype=torch.int64, device=device),
        )
        _E2M1_TABLE_CACHE[device] = cached
    return cached


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Rounds float32 values to the E2M1 grid, nearest with ties-to-even."""
    positive, midpoints, even_index = _e2m1_tables(x.device)
    magnitude = x.abs().clamp(max=_FP4_E2M1_MAX)
    index = torch.searchsorted(midpoints, magnitude, right=False)
    clipped = index.clamp(max=len(_E2M1_MIDPOINTS) - 1)
    is_tie = magnitude == midpoints[clipped]
    rounded_index = torch.where(is_tie, even_index[clipped], index)
    rounded = positive[rounded_index]
    return torch.where(x < 0, -rounded, rounded)


_E4M3_MAX = 448.0
_E4M3_SUBNORMAL_THRESHOLD = 2.0**-6
_E4M3_SUBNORMAL_STEP = 2.0**-9


def _round_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Rounds float32 values to the E4M3 grid, RTNE with saturation.

    Equivalent to ``x.to(torch.float8_e4m3fn).to(torch.float32)`` for
    in-range values but free of native float8 casts: torch_npu's
    device-side e4m3 cast hits an aclnnInplaceCopy failure (CANN 561103),
    so the grid rounding is done arithmetically. ``torch.round`` provides
    the ties-to-even semantics and every intermediate (integer q times a
    power-of-two step) is exact in fp32. Overflow saturates to +/-448,
    unlike the native e4m3fn cast whose overflow semantics yield NaN.
    """
    magnitude = x.abs().clamp(max=_E4M3_MAX)
    # frexp semantics (magnitude = m * 2^e, e = E + 1) are recovered from the
    # fp32 bit pattern instead of torch.frexp: NPU has no frexp kernel and
    # the CPU fallback does a synchronous D2H/H2D round-trip, which is both
    # an eager-mode stall and a hard error inside ACL graph capture
    # (rtMemcpy 107030 on a captured stream). bits>>23 & 0xFF is the biased
    # exponent; frexp's e = exp_field - 126 for normals, and frexp(+-0)
    # returns e = 0. The where maps the zero/subnormal bit pattern (exp_field
    # == 0) to e = 0: a zero magnitude rounds to 0 for any step, and fp32
    # subnormals (< 2^-126) sit far below the E4M3 subnormal threshold, so
    # the normal branch's value is irrelevant for them (0 here is also a
    # strict improvement over the frexp form, whose 2^-130 step underflows
    # to 0 and yields NaN from the 0/0 division).
    bits = magnitude.view(torch.int32)
    exp_field = (bits >> 23) & 0xFF
    exponent = torch.where(exp_field > 0, exp_field - 126, 0).to(magnitude.dtype)
    # Normal binade: magnitude in [2^E, 2^(E+1)) -> grid step 2^(E-3).
    # step = 2^(e - 4) and magnitude / step lies in [8, 16); q == 16
    # overflows exactly onto the next binade's first grid point.
    step = torch.exp2(exponent - 4.0)
    q = torch.round(magnitude / step)
    normal_val = q * step
    # Subnormal region: magnitude < 2^-6 -> grid step 2^-9; q reaches 8 at
    # the boundary and lands exactly on 2^-6, the first normal value. The
    # step/threshold stay Python scalars: torch.tensor(scalar, device=npu)
    # is a synchronous H2D copy, illegal inside ACL graph capture, and both
    # are exact powers of two so the scalar kernels are bitwise identical.
    q_sub = torch.round(magnitude / _E4M3_SUBNORMAL_STEP)
    sub_val = q_sub * _E4M3_SUBNORMAL_STEP
    val = torch.where(magnitude < _E4M3_SUBNORMAL_THRESHOLD, sub_val, normal_val)
    # torch.copysign also lacks an NPU kernel (same CPU-fallback hazard);
    # the where form differs from copysign only on signed zeros, which round
    # to zero on the E4M3 grid either way.
    return torch.where(x < 0, -val, val)


def _e4m3_to_float(x_e4m3: torch.Tensor) -> torch.Tensor:
    """Decodes an E4M3 tensor to float32 without native float8 casts.

    Bit-level decode (sign | 4-bit biased exponent, bias 7 | 3-bit mantissa;
    exp 15 + mantissa 7 is NaN, subnormals scale by 2^-9) so weight dequant
    works identically on CPU and NPU.
    """
    bits = x_e4m3.view(torch.uint8).to(torch.int32)
    sign = torch.where(bits >= 128, -1.0, 1.0)
    magnitude_bits = bits & 0x7F
    exponent = (magnitude_bits >> 3) & 0xF
    mantissa = magnitude_bits & 0x7
    mantissa_f = mantissa.to(torch.float32)
    normal = (1.0 + mantissa_f / 8.0) * torch.exp2(exponent.to(torch.float32) - 7.0)
    subnormal = mantissa_f * _E4M3_SUBNORMAL_STEP
    value = torch.where(exponent > 0, normal, subnormal)
    value = torch.where(magnitude_bits == 0x7F, torch.full_like(value, float("nan")), value)
    return value * sign
