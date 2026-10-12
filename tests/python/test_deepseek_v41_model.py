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

"""DeepSeek-V4.1 Python model: config parsing, CSA2 plan, structure, weight
loading, forward path, and the V4.1 quant kernels inlined in deepseek_v41
(pure torch; the quant helpers are device-agnostic).

The forward tests drive :class:`DeepseekV41ForCausalLM` through
:class:`ModelExecutor` with the CSA2 attention backend and synthetic cache /
metadata structures. V4.1 is NPU-only (the backend has no CPU/CUDA path), so
those tests are marked ``npu_only`` and skip elsewhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa2_attention import (
    Csa2AttentionBackend,
    Csa2LayerContext,
    CSA2SharedRuntime,
    _derive_graph_token_capacity_granularity,
    _indexer_score,
    _indexer_topk,
    _select_candidate_blocks,
    _window_visible_idxs,
    apply_interleaved_rope,
)
from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    DSA_CACHE_TOKEN,
    build_cache_specs_v41,
)
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.models import deepseek_v41 as dq
from xllm.python.models.deepseek_v41 import (
    CSA2_MODE_FULL,
    CSA2_MODE_REINDEX,
    CSA2_MODE_REUSE,
    CSA2_MODE_SWA,
    V41_QUANT_ASCEND,
    V41_QUANT_FP8,
    V41_QUANT_NONE,
    CSA2LayerPlan,
    DeepseekV41Config,
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
    DeepseekV41MoE,
    fp4_act_qdq_e4m3_scale,
    fp4_act_qdq_e8m0,
)
from xllm.python.models.deepseek_v41_engram import (
    ENGRAM_IMAGE_SENTINEL_BASE_ID,
    ENGRAM_STORAGE_INT8,
    Engram,
    EngramEmbedding,
    EngramHashState,
    EngramLayout,
    _load_engram_hash_state_sidecar,
    build_compressed_token_map_from_texts,
    build_engram_hash_state,
    load_quarot_gate_unrotate,
    normalize_token_text,
)
from xllm.python.registry import get_model_class


# DeepSeek-V4.1 is NPU-only: ModelExecutor binds the CSA2 attention backend,
# which has no CPU/CUDA implementation. Tests that drive the executor skip on
# hosts without an NPU instead of monkeypatching a backend in.
def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


npu_only = pytest.mark.skipif(
    not _npu_available(),
    reason="DeepSeek-V4.1 requires an NPU attention backend",
)


def _active_device() -> torch.device:
    """NPU when present (the only supported V4.1 platform); CPU for skip-guarded runs."""
    return torch.device("npu:0") if _npu_available() else torch.device("cpu")


# Reference V4.1 Flash config fields (adaptation report 1.4), with the full
# 43-entry compress_ratios layout: [0,0] + [2]*18 + [1]*20 + draft tail [0,0,0].
_DSV41_CONFIG = {
    "model_type": "deepseek_v41",
    "architectures": ["DeepseekV41ForCausalLM"],
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "num_attention_heads": 64,
    "head_dim": 512,
    "vocab_size": 129280,
    "rms_norm_eps": 1e-20,
    "rope_theta": 10000.0,
    "max_position_embeddings": 1048576,
    "rope_scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 16, "type": "yarn"},
    "q_lora_rank": 1280,
    "qk_rope_head_dim": 64,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    "compress_rope_theta": 160000.0,
    "window_size": 128,
    "sliding_window": 128,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_topk": 512,
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "engram_layer_ids": [1, 14],
    "engram_num_embeddings": [384006168, 384016682],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_compressed_vocab_size": 99092,
    "engram_pad_token_id": 2,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [37, 38, 39],
    "dspark_markov_rank": 256,
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3,
    "n_activated_experts": 6,
    "n_routed_experts": 384,
    "moe_intermediate_size": 2304,
    "n_shared_experts": 1,
    "scoring_func": "sqrtsoftplus",
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "swiglu_limit": 10.0,
    "image_token_id": 129264,
    "first_k_dense_replace": 0,
    "tie_word_embeddings": False,
}


# Small geometry for construction / weight-loading tests (6 layers: 0-1 SWA,
# 2-3 encoder CSA2 with Full at 2, 4-5 decoder with Full at 4).
#
# Engram is shrunk to a testable bucket layout: with engram_vocab_size 64 the
# layout draws its primes from 63 upward (67, 71, 73, 79 -> 290 rows), the
# injected token_map covers the 64-id vocabulary and compresses onto 16 ids,
# and layer 1 alone carries the module.
def _engram_test_token_map() -> list[int]:
    return (torch.arange(64) % 16).tolist()


_ENGRAM_TINY_OVERRIDE = {
    "engram_layer_ids": [1],
    "engram_num_embeddings": [290],
    "engram_max_ngram_size": 3,
    "engram_vocab_size": 64,
    "engram_n_heads": 2,
    "engram_head_dim": 32,
    "engram_compressed_vocab_size": 16,
    "engram_pad_token_id": 0,
    "engram_token_map": _engram_test_token_map(),
}

_DSV41_TINY_CONFIG = {
    **_DSV41_CONFIG,
    **_ENGRAM_TINY_OVERRIDE,
    "num_hidden_layers": 6,
    "hidden_size": 128,
    "num_attention_heads": 4,
    "head_dim": 32,
    "q_lora_rank": 32,
    "o_lora_rank": 16,
    "o_groups": 2,
    "qk_rope_head_dim": 8,
    "vocab_size": 64,
    "n_routed_experts": 4,
    "n_activated_experts": 2,
    "moe_intermediate_size": 32,
    "index_n_heads": 2,
    "index_head_dim": 16,
    "index_topk": 4,
    "compress_ratios": [0, 0, 2, 2, 1, 1],
    "kv_source_layer_ids": [2, 4],
    "index_source_layer_ids": [2, 4],
}

# Forward-path geometry (contract section 5): head_dim 64 and index_head_dim 32
# satisfy the QDQ group divisibility; index sources [2, 4, 5] make layer 5 a
# Reindex layer consuming layer 4's candidate pool.
_DSV41_FORWARD_CONFIG = {
    **_DSV41_TINY_CONFIG,
    "num_attention_heads": 2,
    "head_dim": 64,
    "qk_rope_head_dim": 16,
    "index_head_dim": 32,
    "max_position_embeddings": 64,
    "window_size": 8,
    "sliding_window": 8,
    "index_source_layer_ids": [2, 4, 5],
    "candidate_source_layer_id": 4,
    "candidate_topk_blocks": 2,
    "candidate_block_size": 4,
}


def test_config_from_dict_and_defaults() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_CONFIG)
    assert cfg.model_type == "deepseek_v41"
    assert cfg.n_layers == 40
    assert cfg.q_lora_rank == 1280
    assert cfg.n_routed_experts == 384
    assert cfg.kv_source_layer_ids == [2, 8, 14, 20]
    assert cfg.index_source_layer_ids == [2, 8, 14, 20, 24, 28, 32, 36]
    assert cfg.candidate_source_layer_id == 20
    assert cfg.engram_layer_ids == [1, 14]
    assert cfg.engram_pad_token_id == 2
    assert cfg.image_token_id == 129264
    # Sparse dict: V4.1 defaults, the n_layers alias, no fabricated cache
    # groups and no hash routing.
    sparse = DeepseekV41Config.from_dict({"model_type": "deepseek_v41", "n_layers": 6})
    assert sparse.n_layers == 6
    assert sparse.rms_norm_eps == 1e-20
    assert sparse.compress_ratios == [0] * 6
    assert sparse.n_hash_layers == 0


def test_config_preserves_v41_compress_ratios() -> None:
    """Ratio 0 (SWA), 1 (decoder) and 2 (encoder CSA2) stay distinct.

    V4's ``<= 1 -> 1`` normalization would erase the SWA/decoder distinction;
    the trailing draft entries (3 zeros) are dropped for the 40-layer model.
    """
    cfg = DeepseekV41Config.from_dict(_DSV41_CONFIG)
    assert cfg.compress_ratios == [0, 0] + [2] * 18 + [1] * 20
    assert len(cfg.compress_ratios) == cfg.n_layers


@pytest.mark.parametrize("ratio", [-1, 3, 8, 128])
def test_config_rejects_unsupported_compress_ratios(ratio: int) -> None:
    with pytest.raises(ValueError, match="unsupported DeepSeek-V4.1 compression ratio"):
        DeepseekV41Config.from_dict({**_DSV41_CONFIG, "compress_ratios": [ratio] * 40})


# ---------------------------------------------------------------------------
# CSA2 layer plan (semantics doc section 3)
# ---------------------------------------------------------------------------


def test_csa2_plan_matches_reference_layer_assignment() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_CONFIG)
    plan = CSA2LayerPlan.build(cfg.n_layers, cfg.compress_ratios, cfg.kv_source_layer_ids, cfg.index_source_layer_ids)
    # No cross-layer CED hidden capture exists in the reference: every
    # kv-source layer compresses its OWN attention input.
    full_layers = {2, 8, 14, 20}
    reindex_layers = {24, 28, 32, 36}
    encoder_group_source = {
        **{layer: 2 for layer in range(2, 8)},
        **{layer: 8 for layer in range(8, 14)},
        **{layer: 14 for layer in range(14, 20)},
        **{layer: 20 for layer in range(20, 40)},
    }
    for layer_id in range(cfg.n_layers):
        mode = plan.mode(layer_id)
        if layer_id in (0, 1):
            assert mode.mode == CSA2_MODE_SWA
            assert mode.compress_ratio == 0
        elif layer_id in full_layers:
            assert mode.mode == CSA2_MODE_FULL
            assert mode.main_kv_source_layer == layer_id
        elif layer_id in reindex_layers:
            assert mode.mode == CSA2_MODE_REINDEX
            assert mode.main_kv_source_layer == 20
        else:
            assert mode.mode == CSA2_MODE_REUSE
            assert mode.main_kv_source_layer == encoder_group_source[layer_id]
        if layer_id >= 2:
            assert mode.compress_ratio == (2 if layer_id < 20 else 1)


@pytest.mark.parametrize(
    ("n_layers", "ratios", "kv_sources", "index_sources"),
    [
        (40, [1] * 40, [45], [45, 2]),  # kv source out of range
        (40, [1] * 40, [2], [8]),  # kv source not an index source
        (40, [1] * 40, [8], [2, 8]),  # index source before any kv source
        (40, [0] * 40, [2], [2, 8]),  # zero ratio on a kv source
    ],
)
def test_csa2_plan_rejects_invalid_source_layouts(
    n_layers: int,
    ratios: list[int],
    kv_sources: list[int],
    index_sources: list[int],
) -> None:
    with pytest.raises(ValueError):
        CSA2LayerPlan.build(n_layers, ratios, kv_sources, index_sources)


# ---------------------------------------------------------------------------
# Module layout / structure (semantics doc sections 2.6, 5, 6)
# ---------------------------------------------------------------------------


def _tiny_model() -> DeepseekV41Model:
    """Tiny model with the injected engram token map (construction tests)."""
    return DeepseekV41Model(
        DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG),
        torch.float32,
        torch.device("cpu"),
        config=_DSV41_TINY_CONFIG,
    )


def test_module_layout_matches_csa2_modes() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    model = _tiny_model()
    swa = model.layers[0].self_attn
    full_encoder = model.layers[2].self_attn
    reuse_encoder = model.layers[3].self_attn
    full_decoder = model.layers[4].self_attn
    reuse_decoder = model.layers[5].self_attn

    # Full layers own the compressor (no ape in V4.1) + indexer and are their
    # own group's main-KV source; Reuse/SWA layers own neither.
    assert full_encoder.indexer is not None
    assert full_encoder.cmp_wkv.out_features == cfg.head_dim
    assert not hasattr(full_encoder, "cmp_ape")
    assert full_encoder.mode.main_kv_source_layer == 2
    assert full_decoder.mode.main_kv_source_layer == 4
    for attn in (reuse_encoder, reuse_decoder, swa):
        assert attn.indexer is None
        assert not hasattr(attn, "cmp_wkv")
    # V4.1 removed the top-level hc_head_* weights (single-pass mHC).
    assert not any(name.startswith("hc_head") for name, _ in model.named_parameters())


def test_v41_indexer_uses_independent_key_path() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    model = _tiny_model()
    indexer = model.layers[2].self_attn.indexer
    assert indexer is not None
    assert indexer.wq_b.weight.shape == (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)
    # wk acts on the compressor latent (NOT the hidden states) and produces ONE
    # shared key per compressed position: head_dim -> index_head_dim.
    assert indexer.wk.weight.shape == (cfg.index_head_dim, cfg.head_dim)
    assert indexer.k_norm.weight.shape == (cfg.index_head_dim,)
    assert indexer.weights_proj.weight.shape == (cfg.index_n_heads, cfg.hidden_size)


def test_moe_bias_vl_shifts_only_visual_tokens() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    model = _tiny_model()
    moe = model.layers[2].mlp
    # gate_bias_vl exists but is unloaded until the checkpoint provides it;
    # V4.1 removed hash routing (no tid2eid).
    assert moe.gate_bias_vl.shape == (cfg.n_routed_experts,)
    assert moe.gate_bias_vl_loaded is False
    assert not hasattr(moe, "tid2eid")
    assert hasattr(moe, "e_score_correction_bias")

    moe.gate_bias_vl_loaded = True
    with torch.no_grad():
        moe.gate_bias_vl.fill_(1.5)
    gate_input = torch.zeros(2, cfg.hidden_size)
    input_ids = torch.tensor([1, cfg.image_token_id])
    logits, scores = moe._selection_logits(gate_input, input_ids)
    # Only the image token's SELECTION logits shift by bias_vl; the unbiased
    # scores (which carry the routing weights) are modality-independent.
    assert torch.allclose(logits[1], logits[0] + 1.5)
    assert torch.allclose(scores[1], scores[0])
    # Text-only tokens route on the unshifted scores.
    text_logits, _ = moe._selection_logits(gate_input, None)
    assert torch.allclose(text_logits, scores.expand(2, -1))


@pytest.mark.parametrize("tokens", [0, 1, 6])
@pytest.mark.parametrize("topk,normalize,limit", [(1, True, 2.0), (2, True, 2.0), (2, False, 0.0)])
@pytest.mark.parametrize("ep_size,ep_rank,selected", [(1, 0, 193), (2, 1, 193), (2, 1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_moe_active_segments_match_all_expert_reference(
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
    topk: int,
    normalize: bool,
    limit: float,
    ep_size: int,
    ep_rank: int,
    selected: int,
    dtype: torch.dtype,
) -> None:
    from xllm.python import distributed

    torch.manual_seed(41)
    cfg = DeepseekV41Config.from_dict(
        {
            **_DSV41_TINY_CONFIG,
            "hidden_size": 8,
            "moe_intermediate_size": 6,
            "n_routed_experts": 384,
            "n_activated_experts": topk,
            "norm_topk_prob": normalize,
            "swiglu_limit": limit,
        }
    )
    cfg.ep_size, cfg.ep_rank = ep_size, ep_rank
    moe = DeepseekV41MoE(cfg, 2, dtype, torch.device("cpu"))
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.normal_(std=0.7)
        moe.e_score_correction_bias.zero_()
        moe.e_score_correction_bias[selected : selected + 2] = 100.0
        moe.gate_bias_vl.zero_()
        moe.gate_bias_vl[selected + 2 : selected + 4] = 100.0
    moe.gate_bias_vl_loaded = True
    hidden = torch.randn(1, tokens, cfg.hidden_size, dtype=dtype)
    input_ids = torch.tensor([cfg.image_token_id if i % 2 else 1 for i in range(tokens)])
    x = hidden.reshape(-1, cfg.hidden_size)
    logits, scores = moe._selection_logits(x, input_ids)
    indices = logits.topk(topk, dim=-1).indices
    weights = scores.gather(1, indices)
    if normalize and topk > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * moe.routed_scaling
    expected = torch.zeros_like(x, dtype=torch.float32)
    active = 0
    for local_slot in range(moe.num_experts_per_rank):
        rows, slots = torch.where(indices == moe.start_expert_id + local_slot)
        if rows.numel() == 0:
            continue
        active += 1
        gate = torch.nn.functional.linear(x[rows], moe.experts_w1[local_slot]).float()
        up = torch.nn.functional.linear(x[rows], moe.experts_w3[local_slot]).float()
        if 0.0 < limit < 1_000_000.0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        intermediate = weights[rows, slots, None] * (torch.nn.functional.silu(gate) * up)
        output = torch.nn.functional.linear(intermediate.to(dtype), moe.experts_w2[local_slot])
        expected.index_add_(0, rows, output.float())

    reduced: list[torch.Tensor] = []

    def fake_all_reduce(partial: torch.Tensor) -> None:
        reduced.append(partial.clone())
        partial.mul_(2)

    monkeypatch.setattr(distributed, "moe_ep_all_reduce", fake_all_reduce)
    original_where = torch.where
    where_calls = 0

    def counted_where(*args: torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
        nonlocal where_calls
        where_calls += 1
        return original_where(*args)

    original_linear = torch.nn.functional.linear
    linear_calls = 0

    def counted_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        nonlocal linear_calls
        linear_calls += 1
        return original_linear(input, weight, bias)

    monkeypatch.setattr(torch, "where", counted_where)
    monkeypatch.setattr(torch.nn.functional, "linear", counted_linear)
    actual = moe(hidden, input_ids)
    # One visual-bias selection and one local-route filter, independent of E.
    assert where_calls == 2
    # Three projections per active expert, plus routing and shared projections.
    assert linear_calls == 3 * active + 4
    assert active <= min(tokens * topk, 4)
    if ep_size > 1:
        assert len(reduced) == 1
        torch.testing.assert_close(reduced[0], expected)
        expected *= 2
    else:
        assert not reduced
    expected += moe._shared_forward(x).float()
    torch.testing.assert_close(actual, expected.to(dtype).view_as(hidden))


def test_registry_resolves_deepseek_v41() -> None:
    """V4.1 is registered, and NPU-gated: non-NPU platforms fail fast."""
    names = ("deepseek_v41", "DeepseekV41ForCausalLM")
    if _npu_available():
        for name in names:
            assert get_model_class(name).__name__ == "DeepseekV41ForCausalLM"
        return
    for name in names:
        with pytest.raises(NotImplementedError, match="supported platforms: \\['npu'\\]"):
            get_model_class(name)


def test_attention_backend_rejects_v41_off_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The executor rejects V4.1 before the CUDA branch can return a backend.

    Simulating the CUDA platform is the point: without the entry guard the
    CUDA ``FlashInferBackend`` branch would return a backend for V4.1. Only
    platform detection is patched; no attention backend is faked.
    """
    from xllm.python.model_executor import executor as executor_module

    monkeypatch.setattr(executor_module.current_platform, "is_npu", lambda: False)
    monkeypatch.setattr(executor_module.current_platform, "is_cuda", lambda: True)
    with pytest.raises(NotImplementedError, match="runs on NPU only"):
        executor_module._create_attention_backend(
            first_attention=object(),
            device=torch.device("cuda"),
            dtype=torch.float16,
            config={"model_type": "deepseek_v41"},
        )


# ---------------------------------------------------------------------------
# Weight loading (torch path, "converted" reference layout)
# ---------------------------------------------------------------------------


class _FakeStateDict:
    """Minimal StateDict stand-in for the weight-loading mapping test."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def _build_tiny_checkpoint(cfg: DeepseekV41Config) -> dict[str, torch.Tensor]:
    """Reference (converted) checkpoint layout for the torch load path.

    Plain float tensors with the reference names; no ``.scale`` companions.
    The compressor gate and the indexer key path exist only on the kv-source
    layers, with ``wgate`` absent at ratio 1 and ``wk`` shaped
    ``[index_head_dim, head_dim]``.
    """
    hidden = cfg.hidden_size
    q_lora = cfg.q_lora_rank
    head_dim = cfg.head_dim
    n_heads = cfg.n_heads
    inter = cfg.moe_intermediate_size
    n_experts = cfg.n_routed_experts
    o_lora = cfg.o_lora_rank

    def w(out: int, inp: int) -> torch.Tensor:
        return torch.randn(out, inp)

    tensors: dict[str, torch.Tensor] = {
        "embed.weight": torch.randn(cfg.vocab_size, hidden),
        "norm.weight": torch.ones(hidden),
        "head.weight": torch.randn(cfg.vocab_size, hidden),
    }
    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        tensors[p + "attn_norm.weight"] = torch.ones(hidden)
        tensors[p + "ffn_norm.weight"] = torch.ones(hidden)
        for part in ("attn", "ffn"):
            for suffix, shape in (
                ("fn", ((2 + cfg.hc_mult) * cfg.hc_mult, cfg.hc_mult * hidden)),
                ("scale", (3,)),
                ("base", ((2 + cfg.hc_mult) * cfg.hc_mult,)),
            ):
                tensors[p + f"hc_{part}_{suffix}"] = torch.randn(*shape)
        tensors[p + "attn.wq_a.weight"] = w(q_lora, hidden)
        tensors[p + "attn.wq_b.weight"] = w(n_heads * head_dim, q_lora)
        tensors[p + "attn.wkv.weight"] = w(head_dim, hidden)
        tensors[p + "attn.wo_a.weight"] = w(cfg.o_groups * o_lora, (n_heads * head_dim) // cfg.o_groups)
        tensors[p + "attn.wo_b.weight"] = w(hidden, cfg.o_groups * o_lora)
        tensors[p + "attn.q_norm.weight"] = torch.ones(q_lora)
        tensors[p + "attn.kv_norm.weight"] = torch.ones(head_dim)
        tensors[p + "attn.attn_sink"] = torch.zeros(n_heads)
        if i in (2, 4):  # Full layers: compressor (gate only at ratio 2) + indexer.
            tensors[p + "attn.compressor.wkv.weight"] = w(head_dim, hidden)
            if cfg.compress_ratios[i] > 1:
                tensors[p + "attn.compressor.wgate.weight"] = w(head_dim, hidden)
            tensors[p + "attn.compressor.norm.weight"] = torch.ones(head_dim)
            tensors[p + "attn.indexer.wq_b.weight"] = w(cfg.index_n_heads * cfg.index_head_dim, q_lora)
            tensors[p + "attn.indexer.wk.weight"] = w(cfg.index_head_dim, head_dim)
            tensors[p + "attn.indexer.k_norm.weight"] = torch.ones(cfg.index_head_dim)
            tensors[p + "attn.indexer.weights_proj.weight"] = w(cfg.index_n_heads, hidden)
        # MoE: gate + bias + bias_vl, routed + shared experts.
        tensors[p + "ffn.gate.weight"] = torch.randn(n_experts, hidden)
        tensors[p + "ffn.gate.bias"] = torch.randn(n_experts)
        tensors[p + "ffn.gate.bias_vl"] = torch.randn(n_experts)
        for expert in range(n_experts):
            for name, out, inp in (("w1", inter, hidden), ("w2", hidden, inter), ("w3", inter, hidden)):
                tensors[p + f"ffn.experts.{expert}.{name}.weight"] = w(out, inp)
        for name, out, inp in (("w1", inter, hidden), ("w2", hidden, inter), ("w3", inter, hidden)):
            tensors[p + f"ffn.shared_experts.{name}.weight"] = w(out, inp)
        # Engram layer: FP8-E4M3 payload bytes + ue8m0 scales stay raw (uint8),
        # wkv / q / k arrive as plain floats.
        if i in cfg.engram_layer_ids:
            n_hash_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
            head_dim = cfg.engram_head_dim
            rows = cfg.engram_num_embeddings[cfg.engram_layer_ids.index(i)]
            tensors[p + "engram.embed.weight"] = torch.randint(0, 256, (rows, head_dim), dtype=torch.uint8)
            tensors[p + "engram.embed.scale"] = torch.randint(126, 130, (rows, head_dim // 32), dtype=torch.uint8)
            tensors[p + "engram.wkv.weight"] = w(hidden * (cfg.hc_mult + 1), n_hash_cols * head_dim)
            tensors[p + "engram.q_weight"] = w(cfg.hc_mult, hidden)
            tensors[p + "engram.k_weight"] = w(cfg.hc_mult, hidden)
    return tensors


def test_load_weights_maps_v41_checkpoint_layout() -> None:
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    lm = DeepseekV41ForCausalLM({**_DSV41_TINY_CONFIG, "device": "cpu", "dtype": "float32"})
    tensors = _build_tiny_checkpoint(cfg)
    lm.load_weights([_FakeStateDict(tensors)], 0, 1)

    # Attention projections keep the checkpoint layout; the ratio-1 source has
    # no wgate; the indexer key path maps wk + k_norm.
    assert torch.equal(lm.model.layers[0].self_attn.q_a_proj.weight, tensors["layers.0.attn.wq_a.weight"])
    assert lm.model.layers[3].self_attn.attn_sink_loaded
    assert torch.equal(lm.model.layers[2].self_attn.cmp_wkv.weight, tensors["layers.2.attn.compressor.wkv.weight"])
    assert not hasattr(lm.model.layers[4].self_attn, "cmp_wgate")
    assert torch.equal(lm.model.layers[2].self_attn.indexer.wk.weight, tensors["layers.2.attn.indexer.wk.weight"])
    # MoE: gate bias + bias_vl + experts.
    assert torch.equal(lm.model.layers[5].mlp.e_score_correction_bias, tensors["layers.5.ffn.gate.bias"])
    assert lm.model.layers[5].mlp.gate_bias_vl_loaded
    assert torch.equal(lm.model.layers[1].mlp.experts_w3[3], tensors["layers.1.ffn.experts.3.w3.weight"])
    assert torch.equal(lm.model.layers[0].mlp.shared_w1.weight, tensors["layers.0.ffn.shared_experts.w1.weight"])
    # Engram layer 1: the table stays quantized (raw bytes) and is not
    # silently dropped; every other layer carries no module.
    eng = lm.model.layers[1].engram
    assert eng is not None
    assert torch.equal(eng.embed_tokens.weight, tensors["layers.1.engram.embed.weight"])
    assert torch.equal(eng.embed_tokens.weight_scale, tensors["layers.1.engram.embed.scale"])
    assert torch.equal(eng.wkv.weight, tensors["layers.1.engram.wkv.weight"])
    assert torch.equal(eng.q_weight, tensors["layers.1.engram.q_weight"])
    assert torch.equal(eng.k_weight, tensors["layers.1.engram.k_weight"])
    for layer_id in (0, 2, 3, 4, 5):
        assert lm.model.layers[layer_id].engram is None
    # Final norm + head (no hc_head).
    assert torch.equal(lm.model.norm.weight, tensors["norm.weight"])
    assert torch.equal(lm.lm_head.weight, tensors["head.weight"])


def test_load_weights_requires_attention_projections() -> None:
    """A missing mandatory attention projection fails the load.

    :class:`DeepseekV41Attention` constructs q_a/q_b/kv/o_a/o_b
    unconditionally, so an export that omits one must raise instead of leaving
    the parameter at uninitialized values.
    """
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    tensors = _build_tiny_checkpoint(cfg)
    del tensors["layers.0.attn.wq_a.weight"]
    lm = DeepseekV41ForCausalLM({**_DSV41_TINY_CONFIG, "device": "cpu", "dtype": "float32"})
    with pytest.raises(KeyError, match=r"layers\.0\.attn\.wq_a\.weight"):
        lm.load_weights([_FakeStateDict(tensors)], 0, 1)


def test_load_weights_requires_moe_gate() -> None:
    """A MoE layer dispatches on module type, so a missing gate must raise.

    Layer 5 is MoE in the tiny config. Dropping its ``ffn.gate.weight`` must
    surface from ``_load_torch_moe`` instead of silently skipping the MLP
    (the old ``_has(ffn.gate.weight)`` dispatch left the layer uninitialized).
    """
    cfg = DeepseekV41Config.from_dict(_DSV41_TINY_CONFIG)
    tensors = _build_tiny_checkpoint(cfg)
    del tensors["layers.5.ffn.gate.weight"]
    lm = DeepseekV41ForCausalLM({**_DSV41_TINY_CONFIG, "device": "cpu", "dtype": "float32"})
    with pytest.raises(KeyError):
        lm.load_weights([_FakeStateDict(tensors)], 0, 1)


# ---------------------------------------------------------------------------
# W8A8 (ascend export) loading: a missing weight_offset must be zero-filled
# ---------------------------------------------------------------------------


# Minimal ascend geometry: one SWA (no compressor/indexer) dense layer, so the
# loader stays small; the W8A8DynamicLinear buffers are the point of the test.
_DSV41_ASCEND_CONFIG = {
    **_DSV41_CONFIG,
    "num_hidden_layers": 1,
    "hidden_size": 32,
    "num_attention_heads": 2,
    "head_dim": 16,
    "q_lora_rank": 8,
    "o_lora_rank": 4,
    "o_groups": 2,
    "qk_rope_head_dim": 4,
    "vocab_size": 32,
    "moe_intermediate_size": 16,
    "first_k_dense_replace": 1,
    "compress_ratios": [0],
    "kv_source_layer_ids": [],
    "index_source_layer_ids": [],
    "candidate_source_layer_id": -1,
    "candidate_topk_blocks": 0,
    "candidate_block_size": 0,
    "engram_layer_ids": [],
    "max_position_embeddings": 64,
    "window_size": 8,
    "sliding_window": 8,
}


def _make_ascend_lm(tmp_path, monkeypatch: pytest.MonkeyPatch) -> DeepseekV41ForCausalLM:
    """Ascend-mode model; ``quant_model_description.json`` selects the export.

    ``prepare_quant_weight`` is the only kernel hit by
    ``process_weights_after_loading`` on these non-transposed projections, so
    stubbing it keeps the weight-loading test free of NPU kernels.
    """
    from xllm.python import kernels

    monkeypatch.setattr(kernels, "prepare_quant_weight", lambda weight: weight, raising=False)
    (tmp_path / "quant_model_description.json").write_text("{}")
    lm = DeepseekV41ForCausalLM(
        {**_DSV41_ASCEND_CONFIG, "device": "cpu", "dtype": "float32", "model_path": str(tmp_path)}
    )
    assert lm.quant_mode == "ascend"
    return lm


def _ascend_tensors(omit_attn_offsets: bool = False) -> dict[str, torch.Tensor]:
    """W8A8 export layout for the minimal ascend config (dense, one SWA layer)."""
    cfg = _DSV41_ASCEND_CONFIG
    hidden = cfg["hidden_size"]
    q_lora = cfg["q_lora_rank"]
    head_dim = cfg["head_dim"]
    n_heads = cfg["num_attention_heads"]
    o_groups = cfg["o_groups"]
    o_lora = cfg["o_lora_rank"]
    inter = cfg["moe_intermediate_size"]
    p = "layers.0."
    tensors: dict[str, torch.Tensor] = {
        "embed.weight": torch.randn(cfg["vocab_size"], hidden),
        "norm.weight": torch.ones(hidden),
        "head.weight": torch.randn(cfg["vocab_size"], hidden),
    }
    tensors[p + "attn_norm.weight"] = torch.ones(hidden)
    tensors[p + "ffn_norm.weight"] = torch.ones(hidden)
    hc_mult = cfg["hc_mult"]
    for part in ("attn", "ffn"):
        tensors[p + f"hc_{part}_fn"] = torch.randn((2 + hc_mult) * hc_mult, hc_mult * hidden)
        tensors[p + f"hc_{part}_scale"] = torch.randn(3)
        tensors[p + f"hc_{part}_base"] = torch.randn((2 + hc_mult) * hc_mult)

    def w8a8(prefix: str, out: int, inp: int) -> None:
        tensors[prefix + ".weight"] = torch.zeros(out, inp, dtype=torch.int8)
        tensors[prefix + ".weight_scale"] = torch.ones(out, 1)
        if not omit_attn_offsets:
            tensors[prefix + ".weight_offset"] = torch.zeros(out, 1)

    w8a8(p + "attn.wq_a", q_lora, hidden)
    w8a8(p + "attn.wq_b", n_heads * head_dim, q_lora)
    w8a8(p + "attn.wkv", head_dim, hidden)
    tensors[p + "attn.wo_a.weight"] = torch.randn(o_groups * o_lora, (n_heads * head_dim) // o_groups)
    tensors[p + "attn.wo_b.weight"] = torch.randn(hidden, o_groups * o_lora)
    tensors[p + "attn.q_norm.weight"] = torch.ones(q_lora)
    tensors[p + "attn.kv_norm.weight"] = torch.ones(head_dim)
    for name, rows, cols in (
        ("gate_proj", inter, hidden),
        ("up_proj", inter, hidden),
        ("down_proj", hidden, inter),
    ):
        tensors[p + f"ffn.{name}.weight"] = torch.zeros(rows, cols, dtype=torch.int8)
        tensors[p + f"ffn.{name}.weight_scale"] = torch.ones(rows, 1)
        tensors[p + f"ffn.{name}.weight_offset"] = torch.zeros(rows, 1)
    return tensors


def _poison_w8a8_offsets(lm: DeepseekV41ForCausalLM) -> None:
    """Simulate the ``torch.empty`` buffer contents: fill every offset with NaN."""
    for module in lm.modules():
        if hasattr(module, "weight_offset"):
            module.weight_offset.data.fill_(float("nan"))


def _ascend_attn_offsets(lm: DeepseekV41ForCausalLM) -> list[torch.Tensor]:
    attn = lm.model.layers[0].self_attn
    return [getattr(attn, name).weight_offset for name in ("q_a_proj", "q_b_proj", "kv_proj")]


def test_ascend_load_zero_fills_missing_weight_offset(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing ``weight_offset`` is copied in as zeros, not left uninitialized.

    ``W8A8DynamicLinear.weight_offset`` starts as ``torch.empty`` and
    ``process_weights_after_loading`` asserts it is all-zero, so skipping the
    tensor would surface as a spurious ValueError. Pre-filling the buffers with
    NaN makes the uninitialized case deterministic.
    """
    lm = _make_ascend_lm(tmp_path, monkeypatch)
    _poison_w8a8_offsets(lm)
    lm.load_weights([_FakeStateDict(_ascend_tensors(omit_attn_offsets=True))], 0, 1)
    offsets = _ascend_attn_offsets(lm)
    assert all(torch.all(offset == 0) for offset in offsets)
    assert all(offset.dtype == torch.float32 for offset in offsets)
    assert [offset.numel() for offset in offsets] == [8, 32, 16]


def test_ascend_load_rejects_nonzero_weight_offset(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint carrying a non-zero (asymmetric) offset is still rejected."""
    lm = _make_ascend_lm(tmp_path, monkeypatch)
    _poison_w8a8_offsets(lm)
    tensors = _ascend_tensors()
    tensors["layers.0.attn.wq_a.weight_offset"] = torch.ones(8, 1)
    with pytest.raises(ValueError, match="zero weight_offset"):
        lm.load_weights([_FakeStateDict(tensors)], 0, 1)


def test_ascend_load_rejects_missing_weight_and_scale(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The unconditionally-constructed projections require ``weight`` / ``weight_scale``."""
    for missing in ("layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.weight_scale"):
        lm = _make_ascend_lm(tmp_path, monkeypatch)
        tensors = _ascend_tensors()
        del tensors[missing]
        with pytest.raises(KeyError):
            lm.load_weights([_FakeStateDict(tensors)], 0, 1)


# V4.1 quant kernels, inlined in deepseek_v41 (semantics doc sections 2.1-2.5)
# ---------------------------------------------------------------------------

_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_e2m1_slow(value: float) -> float:
    """Independent slow E2M1 rounding: nearest value, ties to even mantissa."""
    magnitude = abs(value)
    below = max((g for g in _E2M1_GRID if g <= magnitude), default=0.0)
    above = min((g for g in _E2M1_GRID if g >= magnitude), default=6.0)
    if magnitude - below < above - magnitude:
        nearest = below
    elif magnitude - below > above - magnitude:
        nearest = above
    else:
        nearest = below if _E2M1_GRID.index(below) % 2 == 0 else above
    return -nearest if value < 0 else nearest


def _expected_fp8_act_qdq(x: torch.Tensor, group: int) -> torch.Tensor:
    shape = x.shape
    grouped = x.float().reshape(-1, group)
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-4)
    t = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    scale = torch.exp2(torch.ceil(torch.log2(t.double()))).float()
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    return (normalized.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)).reshape(shape)


def _expected_fp4_e8m0(x: torch.Tensor, group: int) -> torch.Tensor:
    shape = x.shape
    grouped = x.float().reshape(-1, group)
    amax = grouped.abs().amax(dim=-1).clamp_min(6.0 * 2.0**-126)
    t = amax * torch.tensor(1.0 / 6.0, dtype=torch.float32)
    scale = torch.exp2(torch.ceil(torch.log2(t.double()))).float()
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    rounded = torch.tensor(
        [_round_e2m1_slow(v) for v in normalized.flatten().tolist()],
        dtype=torch.float32,
    ).reshape(normalized.shape)
    return (rounded * scale.unsqueeze(-1)).reshape(shape)


def _expected_fp4_e4m3(x: torch.Tensor, group: int) -> torch.Tensor:
    shape = x.shape
    grouped = x.float().reshape(-1, group)
    amax = grouped.abs().amax(dim=-1).clamp_min(6.0 * 2.0**-9)
    scale = (amax / 6.0).to(torch.float8_e4m3fn).float()
    normalized = (grouped / scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    rounded = torch.tensor(
        [_round_e2m1_slow(v) for v in normalized.flatten().tolist()],
        dtype=torch.float32,
    ).reshape(normalized.shape)
    return (rounded * scale.unsqueeze(-1)).reshape(shape)


def _make_block_quantized(out_dim: int, in_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds a synthetic 32x32-block FP8 weight that round-trips exactly."""
    blocks_out = -(-out_dim // 32)
    blocks_in = -(-in_dim // 32)
    exponents = torch.randint(-8, 9, (blocks_out, blocks_in))
    scale_bytes = (exponents + 127).to(torch.uint8)
    scales = (
        torch.exp2(exponents.double())
        .repeat_interleave(32, dim=0)
        .repeat_interleave(32, dim=1)[:out_dim, :in_dim]
        .float()
    )
    base = (torch.randn(out_dim, in_dim) * 100).clamp(-448, 448)
    base = base.to(torch.float8_e4m3fn).float()
    weight = (base * scales).to(torch.bfloat16)
    quantized = (weight.float() / scales).to(torch.float8_e4m3fn)
    return quantized, scale_bytes, weight


def test_ue8m0_to_scale() -> None:
    scale_bytes = torch.tensor([0x01, 0x7F, 0x80, 0x00, 0xFE], dtype=torch.uint8)
    expected = torch.tensor([2.0**-126, 1.0, 2.0, 2.0**-127, 2.0**127])
    assert torch.equal(dq.ue8m0_to_scale(scale_bytes), expected)
    # int8 view of the same bytes; the NaN exponent byte 0xFF is rejected.
    raw = torch.tensor([0x80, 0x7F], dtype=torch.uint8).view(torch.int8)
    assert torch.equal(dq.ue8m0_to_scale(raw), torch.tensor([2.0, 1.0]))
    with pytest.raises(ValueError):
        dq.ue8m0_to_scale(torch.tensor([0x01, 0xFF, 0x02], dtype=torch.uint8))


def test_dequant_fp8_block_roundtrip_exact() -> None:
    torch.manual_seed(0)
    quantized, scale_bytes, weight = _make_block_quantized(96, 160)
    out = dq.dequant_fp8_block(quantized, scale_bytes)
    assert out.dtype == torch.bfloat16
    assert out.shape == weight.shape
    assert torch.equal(out, weight)
    # Non-multiple-of-32 boundaries clip the last block correctly.
    torch.manual_seed(1)
    quantized, scale_bytes, weight = _make_block_quantized(70, 100)
    out = dq.dequant_fp8_block(quantized, scale_bytes)
    assert out.shape == (70, 100)
    assert torch.equal(out, weight)
    with pytest.raises(ValueError):
        dq.dequant_fp8_block(quantized, scale_bytes[:-1])


def test_fp4_nibble_table_and_unpack_order() -> None:
    table = dq.fp4_nibble_table()
    expected = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    assert table.numel() == 16
    assert torch.equal(table, torch.tensor(expected, dtype=torch.float32))
    assert table.view(torch.int32)[8].item() & 0xFFFFFFFF == 0x80000000  # -0.0 bit pattern
    # LOW nibble is element 2j, HIGH nibble is element 2j+1 (convert.py:31-34).
    packed = torch.tensor([[0x12, 0x21, 0x98, 0x9A, 0x00, 0x77, 0xF0]], dtype=torch.uint8)
    out = dq.unpack_fp4(packed)
    expected_row = [1.0, 0.5, 0.5, 1.0, -0.0, -0.5, -1.0, -0.5, 0.0, 0.0, 6.0, 6.0, 0.0, -6.0]
    assert out.dtype == torch.bfloat16 and out.shape == (1, 14)
    assert torch.equal(out.float(), torch.tensor([expected_row]))


def test_dequant_fp4_group_known_scales() -> None:
    torch.manual_seed(3)
    n, k = 8, 96
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.int8)
    # Group scales 2^0, 2^1, 2^-2 for column blocks [0:32), [32:64), [64:96).
    scale_bytes = torch.tensor([[0x7F, 0x80, 0x7D]] * n, dtype=torch.uint8)
    out = dq.dequant_fp4_group(packed, scale_bytes)
    unpacked = dq.unpack_fp4(packed, dtype=torch.float32)
    scales = torch.tensor([1.0, 2.0, 0.25]).repeat_interleave(32)
    assert out.dtype == torch.bfloat16 and out.shape == (n, k)
    assert torch.equal(out, (unpacked * scales).to(torch.bfloat16))
    # Zero scale bytes decode to 2**-127 (16 packed bytes = 32 elements = one group).
    zero_out = dq.dequant_fp4_group(packed[:, :16], torch.zeros(n, 1, dtype=torch.uint8))
    assert torch.equal(zero_out, (unpacked[:, :32] * 2.0**-127).to(torch.bfloat16))


def test_fp8_act_qdq_known_and_reference() -> None:
    # amax = 448 gives scale exactly 1: values round straight through the e4m3
    # grid (100 -> 96, 300 -> 288, round-to-nearest-even; 449 saturates).
    x = torch.tensor([[448.0, 224.0, 1.0, 0.0, -448.0, 100.0, 0.5, 300.0]])
    expected = torch.tensor([[448.0, 224.0, 1.0, 0.0, -448.0, 96.0, 0.5, 288.0]])
    assert torch.equal(dq.fp8_act_qdq(x, group=8), expected)
    assert torch.all(dq.fp8_act_qdq(torch.full((1, 32), 449.0), group=32) == 448.0)
    torch.manual_seed(6)
    xr = (torch.randn(4, 128) * 7).to(torch.bfloat16)
    xr[:, :5] = 0
    out = dq.fp8_act_qdq(xr, group=32)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.float(), _expected_fp8_act_qdq(xr, 32))
    assert torch.all(out[:, :5] == 0)


def test_fp4_act_qdq_e8m0_known_and_reference() -> None:
    # amax = 12 -> s = 2; includes every tie case, which must round to the
    # even E2M1 neighbor (0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2 after /s).
    x = torch.tensor(
        [[6.0, 3.0, 1.5, 0.75, 0.25, 0.0, -6.0, -0.75, 12.0, 5.0, 2.5, 3.5, 0.5, 1.0, 4.0, 0.26]],
        dtype=torch.bfloat16,
    )
    expected = torch.tensor([[6, 3, 2, 1, 0, 0, -6, -1, 12, 4, 2, 4, 0, 1, 4, 0]], dtype=torch.bfloat16)
    assert torch.equal(dq.fp4_act_qdq_e8m0(x, group=16), expected)
    torch.manual_seed(8)
    xr = (torch.randn(4, 128) * 3).to(torch.bfloat16)
    xr[:, :5] = 0
    out = dq.fp4_act_qdq_e8m0(xr, group=32)
    assert torch.equal(out.float(), _expected_fp4_e8m0(xr, 32))
    # All-zero group stays zero.
    assert torch.all(dq.fp4_act_qdq_e8m0(torch.zeros(2, 32, dtype=torch.bfloat16), group=32) == 0)


def test_fp4_act_qdq_e4m3_scale_known_and_reference() -> None:
    # s = e4m3(amax / 6) is NOT a power of two (e.g. 1 -> 0.171875).
    cases = [
        (6.0, 6.0),  # s = e4m3(1) = 1
        (1.0, 1.03125),  # s = e4m3(1/6) = 0.171875
        (0.0234375, 0.0234375),  # 6 * 2**-8: s = 2**-8 exactly
        (5.0, 4.875),  # s = e4m3(5/6) = 0.8125; 5/0.8125 rounds to 6
        (0.0, 0.0),  # all-zero group: amax floor 6 * 2**-9
    ]
    for value, expected in cases:
        out = dq.fp4_act_qdq_e4m3_scale(torch.tensor([value]), group=1)
        assert out.item() == expected, (value, out.item(), expected)
    torch.manual_seed(9)
    xr = (torch.randn(4, 128) * 2).to(torch.bfloat16)
    xr[:, :3] = 0
    out = dq.fp4_act_qdq_e4m3_scale(xr, group=16)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.float(), _expected_fp4_e4m3(xr, 16))


def test_qdq_rejects_indivisible_group() -> None:
    x = torch.randn(2, 40)
    with pytest.raises(ValueError):
        dq.fp8_act_qdq(x, group=32)
    with pytest.raises(ValueError):
        dq.fp4_act_qdq_e8m0(x, group=32)
    with pytest.raises(ValueError):
        dq.fp4_act_qdq_e4m3_scale(x, group=16)


# ---------------------------------------------------------------------------
# Cache layout (adaptation contract section 2)
# ---------------------------------------------------------------------------


def test_build_cache_specs_v41() -> None:
    caches_info, group_infos = build_cache_specs_v41(
        compress_ratios=[0, 0, 2, 2, 1, 1],
        kv_source_layer_ids=[2, 4],
        index_source_layer_ids=[2, 4, 5],
        window_size=128,
        n_layers=6,
    )
    # Group registration order is exact: [SWA, TOKEN(2), TOKEN(1)] -- also when
    # the kv-source input arrives unsorted (groups register by layer id).
    expected_groups = [
        (DSA_CACHE_SLIDING_WINDOW, 1, 128),
        (DSA_CACHE_TOKEN, 2, 128),
        (DSA_CACHE_TOKEN, 1, 128),
    ]
    assert [(g.cache_type, g.ratio, g.block_size) for g in group_infos] == expected_groups
    _, unsorted = build_cache_specs_v41(
        compress_ratios=[1, 1, 2, 2, 1, 1],
        kv_source_layer_ids=[4, 2],
        index_source_layer_ids=[2, 4, 5],
        window_size=128,
        n_layers=6,
    )
    assert [(g.cache_type, g.ratio, g.block_size) for g in unsorted] == expected_groups

    # A plan whose lowest-id kv-source carries ratio 1 would register
    # [SWA, TOKEN(1), TOKEN(2)], diverging from the worker multi_block_tables
    # export order (kMultiBlockExportOrder -> [SWA, TOKEN(2), TOKEN(1)]). The
    # builder pairs tables and groups by index, so this must fail loud rather
    # than mis-pair them.
    with pytest.raises(ValueError, match="export order"):
        build_cache_specs_v41(
            compress_ratios=[0, 0, 1, 1, 2, 2],
            kv_source_layer_ids=[2, 4],
            index_source_layer_ids=[2, 4, 5],
            window_size=128,
            n_layers=6,
        )

    def entries(layer_id: int) -> list[tuple[int, int, int]]:
        return [(c.cache_type, c.ratio, c.group_id) for c in caches_info[layer_id]]

    swa = (DSA_CACHE_SLIDING_WINDOW, 1, 0)
    # kv-source ratio-2 layer: key + index + [swa, kv_state, score_state];
    # kv-source ratio-1: key + index + swa (no states); others only swa.
    assert entries(0) == [swa]
    assert entries(3) == [swa]
    assert entries(2) == [(DSA_CACHE_TOKEN, 2, 1), (DSA_CACHE_TOKEN, 2, 1), swa, swa, swa]
    assert entries(4) == [(DSA_CACHE_TOKEN, 1, 2), (DSA_CACHE_TOKEN, 1, 2), swa]

    # The backend resolves the LayerCache slot layout from the same specs.
    backend = _make_csa2_backend(DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG))
    full_r2 = backend._resolve_cache_mapping(2, 2)
    assert (
        full_r2.cmp_cache_idx,
        full_r2.index_cache_idx,
        full_r2.ori_cache_idx,
        full_r2.kv_state_cache_idx,
        full_r2.score_state_cache_idx,
    ) == (0, 1, 2, 3, 4)
    full_r1 = backend._resolve_cache_mapping(4, 1)
    assert (full_r1.cmp_cache_idx, full_r1.index_cache_idx, full_r1.ori_cache_idx) == (0, 1, 2)
    assert (full_r1.kv_state_cache_idx, full_r1.score_state_cache_idx) == (-1, -1)
    reindex = backend._resolve_cache_mapping(5, 1)
    assert (reindex.cmp_cache_idx, reindex.ori_cache_idx) == (-1, 0)


# ---------------------------------------------------------------------------
# Forward-path harness (ModelExecutor + pure-torch CSA2 backend)
# ---------------------------------------------------------------------------

# TOKEN cache block size of the V4.1 layout (contract section 2).
_TOKEN_BLOCK_SIZE = 128
# SWA ring blocks per sequence (contract: the ring is at least 2 blocks).
_SWA_RING_BLOCKS = 2
# The host-side metadata fields (kv_seq_lens_host / q_seq_lens_host) are CPU
# tensors by production contract: PyAttentionMetadataView::make_host_int32_view
# builds them with torch::kCPU (py_attention_metadata.cpp:481-493). Keep them on
# the host even when the rest of the harness runs on NPU.
_HOST_DEVICE = torch.device("cpu")


def _make_csa2_backend(
    cfg: DeepseekV41Config,
    device: torch.device | None = None,
    *,
    quant_mode: str = V41_QUANT_NONE,
    acl_graph_enabled: bool = False,
    max_model_len: int = 0,
) -> Csa2AttentionBackend:
    return Csa2AttentionBackend(
        compress_ratios=list(cfg.compress_ratios),
        window_size=cfg.window_size,
        n_layers=cfg.n_layers,
        num_heads=cfg.n_heads,
        attn_head_dim=cfg.head_dim,
        index_topk=cfg.index_topk,
        index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim,
        rope_head_dim=cfg.qk_rope_head_dim,
        device=device or torch.device("cpu"),
        dtype=torch.float32,
        kv_source_layer_ids=list(cfg.kv_source_layer_ids),
        index_source_layer_ids=list(cfg.index_source_layer_ids),
        candidate_source_layer_id=cfg.candidate_source_layer_id,
        candidate_topk_blocks=cfg.candidate_topk_blocks,
        candidate_block_size=cfg.candidate_block_size,
        quant_mode=quant_mode,
        acl_graph_enabled=acl_graph_enabled,
        max_model_len=max_model_len,
    )


# ---------------------------------------------------------------------------
# ACL-graph quant admission + startup capacity check (51361671 A/C)
# ---------------------------------------------------------------------------


def _graph_admission_cfg() -> DeepseekV41Config:
    return DeepseekV41Config.from_dict(_DSV41_FORWARD_CONFIG)


@pytest.mark.parametrize(
    "quant_mode,expected",
    [(V41_QUANT_ASCEND, True), (V41_QUANT_FP8, False), (V41_QUANT_NONE, False)],
)
def test_csa2_marks_moe_graph_capability_from_quant_mode(quant_mode: str, expected: bool) -> None:
    """Only the W8A8 (ascend) export has an ACL-graph-capturable MoE chain."""
    backend = _make_csa2_backend(_graph_admission_cfg(), torch.device("cpu"), quant_mode=quant_mode)
    assert backend.quant_mode == quant_mode
    assert backend.moe_graph_capturable is expected


def test_csa2_acl_graph_capacity_is_checked_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """C: fail at startup, not at the first captured forward.

    A derivable ``max_model_len`` or an explicit env value satisfies the check;
    a non-capturable quant mode never reaches it (graph is disabled anyway).
    """
    monkeypatch.delenv("XLLM_V41_GRAPH_CAPACITY_GRANULARITY", raising=False)
    cfg = _graph_admission_cfg()

    derived = _make_csa2_backend(
        cfg,
        torch.device("cpu"),
        quant_mode=V41_QUANT_ASCEND,
        acl_graph_enabled=True,
        max_model_len=32768,
    )
    assert derived.graph_token_capacity_granularity > 0

    with pytest.raises(RuntimeError, match="static committed-row capacity"):
        _make_csa2_backend(
            cfg,
            torch.device("cpu"),
            quant_mode=V41_QUANT_ASCEND,
            acl_graph_enabled=True,
            max_model_len=0,
        )

    # fp8/none already forces eager: the startup capacity check must not fire.
    _make_csa2_backend(
        cfg,
        torch.device("cpu"),
        quant_mode=V41_QUANT_NONE,
        acl_graph_enabled=True,
        max_model_len=0,
    )

    # An explicit env value overrides the derivation and satisfies the check.
    monkeypatch.setenv("XLLM_V41_GRAPH_CAPACITY_GRANULARITY", "8192")
    explicit = _make_csa2_backend(
        cfg,
        torch.device("cpu"),
        quant_mode=V41_QUANT_ASCEND,
        acl_graph_enabled=True,
        max_model_len=0,
    )
    assert explicit.graph_token_capacity_granularity == 8192


def test_graph_token_capacity_granularity_caps_the_serving_bound() -> None:
    """The derivation caps ``max_model_len`` at 32768 (-> 2048 buckets).

    ``max_position_embeddings`` (1048576 for V4.1) is the RoPE horizon, not a
    served context, and must not drive the per-layer gather width; the cap
    reproduces the launcher's historical 2048 default while still shortening
    buckets for smaller served contexts. ``0`` / unknown keeps the fail-closed
    eager guard.
    """
    assert _derive_graph_token_capacity_granularity(1048576) == 2048
    assert _derive_graph_token_capacity_granularity(32768) == 2048
    assert _derive_graph_token_capacity_granularity(8192) == 1024
    assert _derive_graph_token_capacity_granularity(0) == 0
    assert _derive_graph_token_capacity_granularity(-1) == 0


def _random_init_v41(lm: DeepseekV41ForCausalLM) -> None:
    """Deterministic random weights: RMSNorm ones, biases/sinks zero."""
    with torch.no_grad():
        for name, param in lm.named_parameters():
            if param.dtype == torch.uint8:
                # Engram table bytes: fill E4M3 1.0 payloads and 2^0 scales
                # (uninitialized bytes can decode to NaN).
                fill = 127 if name.endswith("weight_scale") else 0x38
                param.copy_(torch.full_like(param, fill))
                continue
            if name == "model.norm.weight" or name.endswith(("layernorm.weight", "cmp_norm.weight", "k_norm.weight")):
                param.copy_(torch.ones_like(param))
            elif name.endswith(("attn_sink", "e_score_correction_bias", "gate_bias_vl")):
                param.copy_(torch.zeros_like(param))
            else:
                param.normal_(mean=0.0, std=0.05)
            param.requires_grad_(False)


def _allocate_v41_layer_caches(
    cfg: DeepseekV41Config,
    n_seqs: int,
    dtype: torch.dtype,
    device: torch.device | None = None,
    *,
    reserve_token_sink: bool = False,
) -> list[LayerCache]:
    """LayerCache tensors for the contract-section-2 layout.

    With ``reserve_token_sink`` the TOKEN pools carry one extra leading block
    (block 0) that is never written, so real per-sequence blocks are addressed
    1-based. This mirrors the production invariant that block 0 is a reserved
    zero padding sink, which the padded (``-1``) rows clamp to.
    """
    device = device or torch.device("cpu")

    def swa(dtype_: torch.dtype = dtype) -> torch.Tensor:
        return torch.zeros(_SWA_RING_BLOCKS * n_seqs, cfg.window_size, cfg.head_dim, dtype=dtype_, device=device)

    token_blocks = n_seqs + 1 if reserve_token_sink else n_seqs
    caches: list[LayerCache] = []
    for layer_id in range(cfg.n_layers):
        if layer_id not in cfg.kv_source_layer_ids:
            caches.append(LayerCache(key=None, value=None, swa=swa()))
            continue
        ratio = cfg.compress_ratios[layer_id]
        states = ratio > 1
        caches.append(
            LayerCache(
                key=torch.zeros(token_blocks, _TOKEN_BLOCK_SIZE, cfg.head_dim, dtype=dtype, device=device),
                value=None,
                index=torch.zeros(token_blocks, _TOKEN_BLOCK_SIZE, cfg.index_head_dim, dtype=dtype, device=device),
                swa=swa(),
                compress_kv_state=swa(torch.float32) if states else None,
                compress_score_state=swa(torch.float32) if states else None,
            )
        )
    return caches


class _V41ForwardHarness:
    """Drives one model through ModelExecutor with synthetic paging metadata.

    The block tables follow the contract-section-2 groups: the SWA ring (two
    blocks per sequence), the TOKEN(2) pool of layer 2 and the TOKEN(1) pool
    of layer 4, one block per sequence each.
    """

    def __init__(
        self,
        lm: DeepseekV41ForCausalLM,
        config: dict,
        n_seqs: int,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> None:
        self.lm = lm
        self.cfg = lm.cfg
        self.n_seqs = n_seqs
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.swa_bt = torch.arange(_SWA_RING_BLOCKS * n_seqs, dtype=torch.int32, device=self.device).view(n_seqs, -1)
        # Block 0 of the TOKEN pools is a reserved zero padding sink, so real
        # per-sequence blocks are addressed 1-based (matching production).
        self.tok2_bt = (torch.arange(n_seqs, dtype=torch.int32, device=self.device) + 1).view(n_seqs, 1)
        self.tok1_bt = (torch.arange(n_seqs, dtype=torch.int32, device=self.device) + 1).view(n_seqs, 1)
        self.layer_caches = _allocate_v41_layer_caches(
            self.cfg, n_seqs, dtype, device=self.device, reserve_token_sink=True
        )
        self.executor = ModelExecutor(lm, dict(config), max_seqs_per_batch=n_seqs)
        self.executor.bind_kv_caches(self.layer_caches)
        self.backend = self.executor.attention_backend

    def step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_lens: list[int],
        q_lens: list[int],
        *,
        is_prefill: bool,
        is_chunked_prefill: bool = False,
    ) -> torch.Tensor:
        metadata = SimpleNamespace(
            is_prefill=is_prefill,
            is_chunked_prefill=is_chunked_prefill,
            is_mixed=False,
            is_spec_verify=False,
            multi_block_tables=[self.swa_bt, self.tok2_bt, self.tok1_bt],
            kv_seq_lens_host=torch.tensor(kv_lens, dtype=torch.int32, device=_HOST_DEVICE),
            kv_seq_lens_host_values=list(kv_lens),
            q_seq_lens_host=torch.tensor(q_lens, dtype=torch.int32, device=_HOST_DEVICE),
        )
        return self.executor.execute(input_ids, positions, metadata)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.lm.compute_logits(hidden, None).float()


def _make_forward_lm(
    dtype: torch.dtype,
    seed: int,
    device: torch.device | None = None,
) -> DeepseekV41ForCausalLM:
    torch.manual_seed(seed)
    lm = DeepseekV41ForCausalLM({**_DSV41_FORWARD_CONFIG, "device": str(device or "cpu"), "dtype": dtype})
    _random_init_v41(lm)
    return lm


# ---------------------------------------------------------------------------
# Compressor math (semantics doc section 4)
# ---------------------------------------------------------------------------


def test_compressor_math() -> None:
    lm = _make_forward_lm(torch.float32, seed=11)
    cfg = lm.cfg
    model = lm.model
    backend = _make_csa2_backend(cfg)
    layer_caches = _allocate_v41_layer_caches(cfg, n_seqs=1, dtype=torch.float32)
    backend.bind_kv_caches(layer_caches)
    layer = model.layers[2].self_attn
    mapping = backend._resolve_cache_mapping(2, layer.mode.compress_ratio)
    cos_sin = model.compress_rotary.cos_sin_cache

    def build_dsa(kv_len: int, q_len: int, start: int) -> SimpleNamespace:
        return backend._builder.build(
            multi_block_tables=[
                torch.tensor([[0, 1]], dtype=torch.int32),
                torch.tensor([[0]], dtype=torch.int32),
                torch.tensor([[0]], dtype=torch.int32),
            ],
            kv_seq_lens=[kv_len],
            q_seq_lens=[q_len],
            positions=torch.arange(start, start + q_len, dtype=torch.int64),
            dsa_cos_sin=None,
            is_prefill=False,
            is_chunked_prefill=True,
        )

    def expected_key_row(latent: torch.Tensor, position: int) -> torch.Tensor:
        rotated = apply_interleaved_rope(latent.unsqueeze(0), torch.tensor([position]), cos_sin)
        return fp4_act_qdq_e4m3_scale(rotated, 16)[0]

    def expected_index_rows(latent: torch.Tensor, starts: list[int]) -> torch.Tensor:
        index_k = layer.indexer_key(latent)
        return fp4_act_qdq_e8m0(apply_interleaved_rope(index_k, torch.tensor(starts), cos_sin), 32)

    # Step 1: 3 tokens (positions 0-2). Group (0, 1) commits; position 2 is
    # parked in the state ring.
    x1 = torch.randn(3, cfg.hidden_size)
    ctx1 = Csa2LayerContext(
        x=x1,
        qr=torch.randn(3, cfg.q_lora_rank),
        positions=torch.arange(3),
        cos_sin_cache=cos_sin,
    )
    shared = CSA2SharedRuntime()
    backend._run_compressor(layer, ctx1, build_dsa(3, 3, 0), layer_caches[2], mapping, [3], [3], shared)

    kv1, score1 = layer.compress_kv_score(x1)
    gate0 = score1[:2].softmax(dim=0)
    pooled0 = (kv1[:2] * gate0).sum(dim=0)
    latent0 = layer.compress_rmsnorm(pooled0.unsqueeze(0))[0]
    key_cache = layer_caches[2].key
    assert torch.allclose(key_cache[0, 0], expected_key_row(latent0, 0), atol=1e-6)
    assert torch.allclose(layer_caches[2].index[0, 0], expected_index_rows(latent0.unsqueeze(0), [0])[0], atol=1e-6)
    # One committed row after 3 tokens (ctx // ratio); the partial group's
    # members are parked in the state ring at their SWA slots.
    assert (key_cache[0, : 3 // 2].abs().sum(dim=-1) > 0).all()
    assert (key_cache[0, 3 // 2 :] == 0).all()
    state_kv = layer_caches[2].compress_kv_state.view(-1, cfg.head_dim)
    state_score = layer_caches[2].compress_score_state.view(-1, cfg.head_dim)
    assert torch.allclose(state_kv[2], kv1[2], atol=1e-6)
    assert torch.allclose(state_score[2], score1[2], atol=1e-6)

    # Step 2: 3 more tokens (positions 3-5). Group (2, 3) completes across the
    # chunk boundary: member 2 comes from the parked state, member 3 is fresh.
    x2 = torch.randn(3, cfg.hidden_size)
    ctx2 = Csa2LayerContext(
        x=x2,
        qr=torch.randn(3, cfg.q_lora_rank),
        positions=torch.arange(3, 6),
        cos_sin_cache=cos_sin,
    )
    backend._run_compressor(layer, ctx2, build_dsa(6, 3, 3), layer_caches[2], mapping, [3], [6], shared)

    kv2, score2 = layer.compress_kv_score(x2)
    members = torch.stack((state_kv[2].clone(), kv2[0]))
    gates = torch.stack((state_score[2].clone(), score2[0])).softmax(dim=0)
    pooled1 = (members * gates).sum(dim=0)
    latent1 = layer.compress_rmsnorm(pooled1.unsqueeze(0))[0]
    gate2 = score2[1:3].softmax(dim=0)
    pooled2 = (kv2[1:3] * gate2).sum(dim=0)
    latent2 = layer.compress_rmsnorm(pooled2.unsqueeze(0))[0]
    # RMSNorm sits AFTER pooling; RoPE uses the group-start position.
    assert torch.allclose(key_cache[0, 1], expected_key_row(latent1, 2), atol=1e-6)
    assert torch.allclose(key_cache[0, 2], expected_key_row(latent2, 4), atol=1e-6)
    expected_index = expected_index_rows(torch.stack((latent1, latent2)), [2, 4])
    assert torch.allclose(layer_caches[2].index[0, 1:3], expected_index, atol=1e-6)
    assert (key_cache[0, : 6 // 2].abs().sum(dim=-1) > 0).all()
    assert (key_cache[0, 6 // 2 :] == 0).all()

    # Ratio 1 (layer 4): a plain projection, one row per token, no gate.
    layer4 = model.layers[4].self_attn
    mapping4 = backend._resolve_cache_mapping(4, layer4.mode.compress_ratio)
    x = torch.randn(5, cfg.hidden_size)
    ctx = Csa2LayerContext(
        x=x,
        qr=torch.randn(5, cfg.q_lora_rank),
        positions=torch.arange(5),
        cos_sin_cache=cos_sin,
    )
    shared4 = CSA2SharedRuntime()
    backend._run_compressor(layer4, ctx, build_dsa(5, 5, 0), layer_caches[4], mapping4, [5], [5], shared4)
    kv, _ = layer4.compress_kv_score(x)
    latent = layer4.compress_rmsnorm(kv)
    expected = fp4_act_qdq_e4m3_scale(apply_interleaved_rope(latent, torch.arange(5), cos_sin), 16)
    assert torch.allclose(layer_caches[4].key[0, :5], expected, atol=1e-6)
    assert (layer_caches[4].key[0, 5:] == 0).all()
    # Ratio 1 keeps no pooling states and no gate.
    assert not hasattr(layer4, "cmp_wgate")
    assert layer_caches[4].compress_kv_state is None
    assert layer_caches[4].compress_score_state is None


def test_compressor_padded_row_uses_nonnegative_rope_position() -> None:
    """A fully padded row (q_len=1, ctx_len=0) on a ratio-2 layer must not
    drive the compressor RoPE position negative.

    With ``ctx_len == 0`` the row is inert (``commit_mask`` all-False, every
    slot mask ``-1``), but the ratio-2 branch still feeds
    ``group_starts = (group_idx * ratio)`` to ``apply_interleaved_rope``, whose
    ``cos_sin_cache.index_select(0, positions)`` rejects the negative
    ``group_idx = floor(-1 / 2) = -1``. Only an upper clamp used to guard this,
    so the row crashed before the mask could make it a no-op; the fix clamps
    the group-start element to a valid non-negative position. The placeholder
    latent is invalid, so it must stay out of the caches.
    """
    lm = _make_forward_lm(torch.float32, seed=11)
    cfg = lm.cfg
    model = lm.model
    backend = _make_csa2_backend(cfg)
    layer_caches = _allocate_v41_layer_caches(cfg, n_seqs=1, dtype=torch.float32)
    backend.bind_kv_caches(layer_caches)
    layer = model.layers[2].self_attn
    assert layer.mode.compress_ratio == 2
    mapping = backend._resolve_cache_mapping(2, layer.mode.compress_ratio)
    cos_sin = model.compress_rotary.cos_sin_cache

    dsa = backend._builder.build(
        multi_block_tables=[
            torch.tensor([[0, 1]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
        ],
        kv_seq_lens=[0],
        q_seq_lens=[1],
        positions=torch.arange(0, 1, dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=True,
    )
    ctx = Csa2LayerContext(
        x=torch.randn(1, cfg.hidden_size),
        qr=torch.randn(1, cfg.q_lora_rank),
        positions=torch.arange(0, 1),
        cos_sin_cache=cos_sin,
    )
    key_before = layer_caches[2].key.clone()
    index_before = layer_caches[2].index.clone()

    # Pre-fix this raised IndexError inside apply_interleaved_rope (:758).
    backend._run_compressor(layer, ctx, dsa, layer_caches[2], mapping, [1], [0], CSA2SharedRuntime())

    # The padded row is unwritten: the key/index caches are byte-identical.
    assert torch.equal(layer_caches[2].key, key_before)
    assert torch.equal(layer_caches[2].index, index_before)


# ---------------------------------------------------------------------------
# Indexer scoring (semantics doc section 5)
# ---------------------------------------------------------------------------


def test_indexer_scoring() -> None:
    # relu-weighted einsum: score[s, t] = sum_h w[s, h] * relu(q[s, h] . k[t]).
    torch.manual_seed(21)
    q = torch.randn(4, 3, 32)
    index_k = torch.randn(6, 32)
    weights = torch.randn(4, 3)
    score = _indexer_score(q, index_k, weights)
    manual = torch.einsum("shd,td->sht", q, index_k).relu_() * weights.unsqueeze(-1)
    assert torch.allclose(score, manual.sum(dim=1), atol=1e-5)

    # The per-head weights carry the (index_head_dim * index_n_heads)^-0.5 scale.
    lm = _make_forward_lm(torch.float32, seed=22)
    layer = lm.model.layers[2].self_attn
    x = torch.randn(4, lm.cfg.hidden_size)
    scale = layer.indexer.head_dim**-0.5 * layer.indexer.n_heads**-0.5
    assert torch.allclose(layer.indexer_weights(x), layer.indexer.weights_proj(x) * scale)

    # Top-k: picks are re-sorted by position (valid picks ascending, invalid
    # tail last), unreachable picks become -1, valid picks shift by the
    # window-row offset. Positions 3-4 are beyond the visibility count.
    score_row = torch.tensor([[3.0, 1.0, 2.0, float("-inf"), float("-inf")]])
    idxs = _indexer_topk(score_row, torch.tensor([3]), index_topk=4, offset=7)
    assert idxs.tolist() == [[7, 8, 9, -1]]

    # Pool-outside rows hold finite positions inside the visible range (the
    # Reindex pool mask), so a pool smaller than index_topk must drop its
    # -inf-scored picks onto the -1 sentinel instead of letting the
    # visibility check pass them (MR 439 review note 3455352).
    score_pool = torch.tensor([[3.0, float("-inf"), 1.0, 2.0, float("-inf")]])
    idxs_pool = _indexer_topk(score_pool, torch.tensor([5]), index_topk=4, offset=7)
    assert idxs_pool.tolist() == [[7, -1, 9, 10]]

    # Candidate pool: block score = max over its positions, the newest partial
    # block is pinned, unreachable picks are dropped.
    score_pool = torch.tensor(
        [[1.0, 2.0, float("-inf"), float("-inf"), 3.0, 4.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")]]
    )
    pinned = _select_candidate_blocks(score_pool, torch.tensor([6]), topk_blocks=1, block_size=4)
    assert pinned.tolist() == [[False] * 4 + [True] * 4 + [False] * 2]
    both = _select_candidate_blocks(score_pool, torch.tensor([6]), topk_blocks=2, block_size=4)
    assert both.tolist() == [[True] * 8 + [False] * 2]


def test_indexer_causal_visibility_and_candidate_pool() -> None:
    lm = _make_forward_lm(torch.float32, seed=23)
    cfg = lm.cfg
    backend = _make_csa2_backend(cfg)
    cos_sin = lm.model.compress_rotary.cos_sin_cache

    # Full layer 2 (ratio 2): fresh top-k over the shared index cache. Every
    # valid pick must address a compressed group the query has fully passed
    # (t < (p + 1) // ratio) and land inside the window-row offset.
    layer2 = lm.model.layers[2].self_attn
    index_cache = torch.randn(1, _TOKEN_BLOCK_SIZE, cfg.index_head_dim) * 0.5
    shared = CSA2SharedRuntime(
        index_cache=index_cache,
        index_block_table=torch.tensor([[0]], dtype=torch.int32),
        compress_ratio=2,
    )
    q_len, ctx_len = 8, 8
    ctx = Csa2LayerContext(
        x=torch.randn(q_len, cfg.hidden_size),
        qr=torch.randn(q_len, cfg.q_lora_rank),
        positions=torch.arange(q_len),
        cos_sin_cache=cos_sin,
    )
    topk = backend._run_indexer(layer2, ctx, None, [q_len], [ctx_len], shared)
    # The fixed-capacity window is [ring candidates ; chunk], so the
    # compressed region starts after window_size + q_len rows.
    offset = cfg.window_size + q_len
    for query in range(q_len):
        visible = (query + 1) // 2
        picks = topk[0, query]
        valid = picks[picks >= 0] - offset
        assert ((valid >= 0) & (valid < visible)).all()
        assert valid.numel() == min(cfg.index_topk, visible)
    # A Full layer before the candidate source does not build the pool.
    assert shared.candidates is None

    # Candidate-source layer 4 publishes the per-sequence pool.
    layer4 = lm.model.layers[4].self_attn
    shared4 = CSA2SharedRuntime(
        index_cache=torch.randn(1, _TOKEN_BLOCK_SIZE, cfg.index_head_dim) * 0.5,
        index_block_table=torch.tensor([[0]], dtype=torch.int32),
        compress_ratio=1,
    )
    backend._run_indexer(layer4, ctx, None, [q_len], [ctx_len], shared4)
    assert shared4.candidates_layer == cfg.candidate_source_layer_id
    assert len(shared4.candidates) == 1

    # Reindex layer 5 re-scores the shared K restricted to the pool. With the
    # query past enough context that visibility no longer truncates the picks,
    # every pick the visibility filter keeps comes from the pool.
    layer5 = lm.model.layers[5].self_attn
    pool = torch.zeros(1, ctx_len, dtype=torch.bool)
    pool[0, [1, 3, 4, 6]] = True
    shared5 = CSA2SharedRuntime(
        index_cache=shared4.index_cache,
        index_block_table=shared4.index_block_table,
        compress_ratio=1,
        candidates=[pool],
    )
    ctx_late = Csa2LayerContext(
        x=torch.randn(1, cfg.hidden_size),
        qr=torch.randn(1, cfg.q_lora_rank),
        positions=torch.tensor([ctx_len - 1]),
        cos_sin_cache=cos_sin,
    )
    topk5 = backend._run_indexer(layer5, ctx_late, None, [1], [ctx_len], shared5)
    picks = topk5[0, 0]
    assert (picks[picks >= 0] - (cfg.window_size + 1)).tolist() == [1, 3, 4, 6]

    # Regression (MR 439 review note 3455352): a candidate pool smaller than
    # index_topk must not leak pool-outside rows into the picks. The pool
    # holds fewer rows than index_topk while every non-pool row still sits
    # inside the visible range, so the visibility filter alone cannot exclude
    # them -- only the -inf pool mask can, and the picks it cuts must land on
    # the -1 sentinel (which _picked_row_mask drops, so attention reads pool
    # rows + window rows only).
    backend.index_topk = 512
    topk_wide = backend._run_indexer(layer5, ctx_late, None, [1], [ctx_len], shared5)
    picks_wide = topk_wide[0, 0]
    assert picks_wide.numel() == ctx_len
    assert (picks_wide[picks_wide >= 0] - (cfg.window_size + 1)).tolist() == [1, 3, 4, 6]
    assert (picks_wide == -1).sum().item() == ctx_len - 4


def test_swa_window_lower_bound_is_per_query() -> None:
    """The SWA window a query reads is bounded per query, not per chunk.

    Regression guard for :func:`_window_visible_idxs` (the ``_csa2_attention``
    window mask): query ``p`` must see exactly the window columns at absolute
    positions ``[max(p - win + 1, 0), p]``. A chunk-level bound shared by the
    whole chunk (only query index 0 sits at the chunk start) would let every
    later query read positions older than its window.
    """
    cases = [
        # (win, prev_ctx, q_len): the first three over-admit under a chunk-level
        # bound, the last is the no-over-masking boundary at the chunk edge.
        (8, 0, 9),
        (2, 5, 3),
        (128, 200, 2),
        (8, 0, 8),
    ]
    for win, prev_ctx, q_len in cases:
        ring_pos = (prev_ctx - win) + torch.arange(win, dtype=torch.int64)
        positions_seq = prev_ctx + torch.arange(q_len, dtype=torch.int64)
        col_pos = torch.cat((ring_pos, positions_seq), dim=0)
        cols = torch.arange(col_pos.numel(), dtype=torch.int64)
        idxs = _window_visible_idxs(positions_seq, col_pos, cols, win)
        assert idxs.shape == (q_len, win + q_len)
        for i in range(q_len):
            p = prev_ctx + i
            picked = idxs[i][idxs[i] >= 0]
            got = col_pos[picked].tolist()
            expected = list(range(max(p - win + 1, 0), p + 1))
            assert got == expected, f"win={win} prev_ctx={prev_ctx} q_len={q_len} query={i}: {got} != {expected}"


# ---------------------------------------------------------------------------
# mHC final merge (semantics doc section 7)
# ---------------------------------------------------------------------------


@npu_only
def test_mhc_final_merge_is_learned_weighted_sum() -> None:
    device = _active_device()
    assert device.type == "npu"
    lm = _make_forward_lm(torch.float32, seed=31, device=device)
    assert lm.model.norm.weight.device.type == "npu"
    # Boost the hc bases so the per-copy coefficients are clearly non-uniform
    # (a near-uniform sigmoid would make the merge indistinguishable from a mean).
    with torch.no_grad():
        for layer in lm.model.layers:
            layer.hc.hc_attn_base.normal_(mean=0.0, std=1.5)
            layer.hc.hc_ffn_base.normal_(mean=0.0, std=1.5)
            layer.hc.hc_attn_scale.fill_(1.0)
            layer.hc.hc_ffn_scale.fill_(1.0)
    harness = _V41ForwardHarness(lm, _DSV41_FORWARD_CONFIG, n_seqs=1, dtype=torch.float32, device=device)
    assert harness.layer_caches[0].swa.device.type == "npu"

    captured: dict[int, dict[str, torch.Tensor]] = {}
    for layer_id in (4, 5):
        decoder_layer = lm.model.layers[layer_id]
        original = decoder_layer.forward

        def make_wrapper(orig, store):
            def wrapper(
                hidden, positions, cos_sin_cache, input_ids=None, pre_mix=None, engram_hashes=None, engram_mask=None
            ):
                out, ffn_pre = orig(hidden, positions, cos_sin_cache, input_ids, pre_mix, engram_hashes, engram_mask)
                store["hidden"] = out.clone()
                store["ffn_pre"] = ffn_pre.clone()
                return out, ffn_pre

            return wrapper

        decoder_layer.forward = make_wrapper(original, captured.setdefault(layer_id, {}))

    input_ids = torch.randint(0, lm.cfg.vocab_size, (5,)).to(device)
    with torch.inference_mode():
        hidden = harness.step(input_ids, torch.arange(5, device=device), [5], [5], is_prefill=True)

    last = captured[5]
    earlier = captured[4]
    # The learned coefficients are per-token and non-uniform.
    assert last["ffn_pre"].shape == (5, lm.cfg.hc_mult)
    assert (last["ffn_pre"].std(dim=-1) > 1e-3).all()

    with torch.inference_mode():
        hc = lm.model.layers[5].hc
        expected = lm.model.norm(hc.hc_pre(last["hidden"], last["ffn_pre"]))
        mean_merge = lm.model.norm(last["hidden"].float().mean(dim=-2))
        earlier_merge = lm.model.norm(hc.hc_pre(last["hidden"], earlier["ffn_pre"]))
    # The final collapse is the learned weighted sum with the LAST block's
    # ffn_pre -- not a mean, not an earlier block's coefficients.
    assert torch.allclose(hidden, expected, atol=1e-5)
    assert not torch.allclose(hidden, mean_merge, atol=1e-3)
    assert not torch.allclose(hidden, earlier_merge, atol=1e-3)


# ---------------------------------------------------------------------------
# Forward smoke + prefill-vs-decode consistency (milestone gate)
# ---------------------------------------------------------------------------


@npu_only
def test_forward_smoke_and_prefill_decode_consistency() -> None:
    device = _active_device()
    assert device.type == "npu"
    lm = _make_forward_lm(torch.bfloat16, seed=41, device=device)
    assert lm.model.norm.weight.device.type == "npu"
    cfg = lm.cfg
    # A prompt one token past the window keeps every query of the prefill chunk
    # inside the window (prompt_len == window_size is the boundary that a
    # chunk-level window bound would still mask identically).
    n_seqs, prompt_len, decode_steps = 2, cfg.window_size + 1, 4
    harness = _V41ForwardHarness(lm, _DSV41_FORWARD_CONFIG, n_seqs, torch.bfloat16, device=device)
    assert harness.layer_caches[0].swa.device.type == "npu"
    input_ids = torch.randint(0, cfg.vocab_size, (n_seqs, prompt_len)).to(device)

    # -- Prefill: the whole 18-token prompt in one batch (2 seqs x 9 tokens).
    with torch.inference_mode():
        hidden = harness.step(
            input_ids.reshape(-1),
            torch.arange(prompt_len, device=device).repeat(n_seqs),
            [prompt_len] * n_seqs,
            [prompt_len] * n_seqs,
            is_prefill=True,
        )
    assert hidden.shape == (n_seqs * prompt_len, cfg.hidden_size)
    assert torch.isfinite(hidden.float()).all()
    prefill_logits = harness.logits(hidden)
    assert prefill_logits.shape == (n_seqs * prompt_len, cfg.vocab_size)
    assert torch.isfinite(prefill_logits).all()

    # SWA ring writes: positions 0-7 fill the first ring block of each
    # sequence; position 8 opens the second ring block.
    swa = harness.layer_caches[0].swa
    assert (swa[0].abs().sum(dim=-1) > 0).all()
    assert (swa[2].abs().sum(dim=-1) > 0).all()
    assert swa[1, 0].abs().sum(dim=-1) > 0
    assert (swa[1, 1:] == 0).all()
    assert swa[3, 0].abs().sum(dim=-1) > 0
    assert (swa[3, 1:] == 0).all()

    # Committed compressed rows = ctx // ratio for both kv sources. Block 0 is
    # the reserved zero padding sink, so sequence 0's real block is index 1.
    key2 = harness.layer_caches[2].key
    key4 = harness.layer_caches[4].key
    assert (key2[1, : prompt_len // 2].abs().sum(dim=-1) > 0).all()
    assert (key2[1, prompt_len // 2 :] == 0).all()
    assert (key4[1, :prompt_len].abs().sum(dim=-1) > 0).all()
    assert (key4[1, prompt_len:] == 0).all()

    # Reuse layer 3 consumes the group source's top-k object as-is.
    assert lm.model.layers[3].self_attn._csa2_used_topk is lm.model.layers[2].self_attn._csa2_used_topk
    # The candidate pool is published by layer 4 only, and the reindex layer's
    # picks stay inside the pool.
    shared = harness.backend._v41_shared
    assert shared.candidates_layer == cfg.candidate_source_layer_id
    topk5 = lm.model.layers[5].self_attn._csa2_used_topk
    # Layer 5 (ratio 1) ran with q_len = prompt_len, so its compressed picks
    # start after window_size + prompt_len window rows.
    compress_offset = cfg.window_size + prompt_len
    for seq in range(n_seqs):
        pool = shared.candidates[seq]
        for query in range(prompt_len):
            valid = topk5[seq, query] - compress_offset
            for value in valid[valid >= 0].tolist():
                assert pool[query, value]

    # -- Decode: 4 single-token steps per sequence continue the same caches.
    decode_ids = torch.randint(0, cfg.vocab_size, (n_seqs, decode_steps)).to(device)
    for step in range(decode_steps):
        position = prompt_len + step
        with torch.inference_mode():
            step_hidden = harness.step(
                decode_ids[:, step],
                torch.full((n_seqs,), position, device=device),
                [position + 1] * n_seqs,
                [1] * n_seqs,
                is_prefill=False,
            )
        assert step_hidden.shape == (n_seqs, cfg.hidden_size)
        assert torch.isfinite(step_hidden.float()).all()
    # Ring writes for positions 8-12 fill the second ring block rows 0-4
    # (position 8 from the prefill, 9-12 from the decode steps).
    second_block_rows = prompt_len - cfg.window_size + decode_steps
    assert (swa[1, :second_block_rows].abs().sum(dim=-1) > 0).all()
    assert (swa[1, second_block_rows:] == 0).all()
    assert (swa[3, :second_block_rows].abs().sum(dim=-1) > 0).all()
    # Committed rows follow ctx // ratio at ctx = prompt_len + decode_steps.
    # Block 0 is the reserved zero padding sink; sequence 0's block is index 1.
    ctx_len = prompt_len + decode_steps
    assert (key2[1, : ctx_len // 2].abs().sum(dim=-1) > 0).all()
    assert (key2[1, ctx_len // 2 :] == 0).all()
    assert (key4[1, :ctx_len].abs().sum(dim=-1) > 0).all()
    assert (key4[1, ctx_len:] == 0).all()

    # -- Consistency: the same prompt re-run token by token as decode steps
    # with fresh caches reproduces the prefill logits.
    replay = _V41ForwardHarness(lm, _DSV41_FORWARD_CONFIG, n_seqs, torch.bfloat16, device=device)
    replay_logits = []
    for position in range(prompt_len):
        with torch.inference_mode():
            step_hidden = replay.step(
                input_ids[:, position],
                torch.full((n_seqs,), position, device=device),
                [position + 1] * n_seqs,
                [1] * n_seqs,
                is_prefill=False,
            )
        replay_logits.append(replay.logits(step_hidden))
    for position in range(prompt_len):
        for seq in range(n_seqs):
            expected = prefill_logits[seq * prompt_len + position]
            diff = (replay_logits[position][seq] - expected).abs().max().item()
            assert diff < 2e-2, f"decode replay diverged at position {position} of sequence {seq}: {diff}"


# ---------------------------------------------------------------------------
# Engram (gated n-gram lookup, reference engram.py semantics)
# ---------------------------------------------------------------------------


def test_engram_layout_prime_buckets() -> None:
    layout = EngramLayout(
        layer_ids=[1, 14],
        num_embeddings=[384006168, 384016682],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    # Primes are drawn from 63 upward, never reused across layers/heads.
    assert layout.primes[0][0] == (67, 71)
    assert layout.primes[0][1] == (73, 79)
    # Layer 14 continues the shared search: 83, 89, 97, 101.
    assert layout.primes[1][0] == (83, 89)
    assert layout.primes[1][1] == (97, 101)
    assert layout.n_hash_cols == 4
    # Offsets are exclusive prefix sums of the flattened head sizes.
    assert torch.equal(layout.offsets, torch.tensor([[0, 67, 138, 211], [0, 83, 172, 269]]))


def test_engram_token_map_normalization_chain() -> None:
    # Case / accent / whitespace folding collapses onto one compressed id.
    lookup, size = build_compressed_token_map_from_texts(["The", "the", "THE", " thé ", "x", "y"])
    assert size == 3
    assert lookup[0] == lookup[1] == lookup[2] == lookup[3]
    assert lookup[4] != lookup[5]
    assert normalize_token_text("  A\tB\nC ") == "a b c"
    # A lone space survives Strip via the private-use sentinel.
    assert normalize_token_text(" ") == " "


def _tiny_engram_hash_state() -> EngramHashState:
    from xllm.python.models.deepseek_v41_engram import compute_hash_multipliers

    layout = EngramLayout(
        layer_ids=[1],
        num_embeddings=[290],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, 16)
    return EngramHashState(layout, torch.tensor(_engram_test_token_map(), dtype=torch.int64), multipliers)


def test_engram_hash_rolling_xor_and_dead_tokens() -> None:
    state = _tiny_engram_hash_state()
    pad_id = state.pad_id
    # Two sequences of 3 tokens; sequence 0's token at position 1 is dead.
    ids = torch.tensor([5, 6, 7, 10, 11, 12])
    positions = torch.tensor([0, 1, 2, 0, 1, 2])
    q_cu = torch.tensor([0, 3, 6])
    dead = torch.tensor([False, True, False, False, False, False])
    hashes = state(ids, positions, q_cu, dead, None, None, None, 0)
    assert hashes.shape == (6, 1, 4)

    comp = torch.tensor(_engram_test_token_map(), dtype=torch.int64)
    m = state.multipliers[0]
    # Column layout: shift 1 -> cols 0/1 (primes 67/71), shift 2 -> cols 2/3
    # (primes 73/79). `blocked` is sticky: once a lookback is out of range or
    # dead, every further shift pools pad_id into the rolling hash.
    r_pos0_2gram = (comp[5] * m[0]) ^ (pad_id * m[1])
    r_pos0_3gram = r_pos0_2gram ^ (pad_id * m[2])
    assert torch.equal(hashes[0, 0, 0], (r_pos0_2gram % 67).to(torch.int32))
    assert torch.equal(hashes[0, 0, 2], ((r_pos0_3gram % 73) + 138).to(torch.int32))
    # Position 2: the dead token at position 1 blocks shifts 1 AND 2 (sticky),
    # even though position 0 itself is a healthy chunk member.
    r_pos2_2gram = (comp[7] * m[0]) ^ (pad_id * m[1])
    r_pos2_3gram = r_pos2_2gram ^ (pad_id * m[2])
    assert torch.equal(hashes[2, 0, 0], (r_pos2_2gram % 67).to(torch.int32))
    assert torch.equal(hashes[2, 0, 2], ((r_pos2_3gram % 73) + 138).to(torch.int32))
    # Sequence 1 (no dead tokens): the 2-gram at position 1 pools position 0.
    r = (comp[11] * m[0]) ^ (comp[10] * m[1])
    assert torch.equal(hashes[4, 0, 0], (r % 67).to(torch.int32))


def test_engram_hash_slot_cache_roundtrip() -> None:
    """Generated tokens read their lookback back from the slot cache."""
    state = _tiny_engram_hash_state()
    state._cache = torch.zeros(64, dtype=torch.int32)
    # Step 1: a 2-token chunk parks its compressed ids at slots 0/1.
    ids1 = torch.tensor([3, 9])
    bt = torch.tensor([[0, 0]], dtype=torch.int32)
    slot1 = torch.tensor([0, 1], dtype=torch.int32)
    q_cu = torch.tensor([0, 2])
    state(
        ids1,
        torch.tensor([0, 1]),
        q_cu,
        torch.zeros(2, dtype=torch.bool),
        None,
        slot1,
        bt,
        32,
    )
    # Step 2: one more token; positions 0/1 precede the chunk start, so their
    # compressed ids must come from the slot cache (slots 0/1).
    comp = torch.tensor(_engram_test_token_map(), dtype=torch.int64)
    m = state.multipliers[0]
    ids2 = torch.tensor([20])
    q_cu2 = torch.tensor([0, 1])
    slot2 = torch.tensor([2], dtype=torch.int32)
    hashes = state(
        ids2,
        torch.tensor([2]),
        q_cu2,
        torch.zeros(1, dtype=torch.bool),
        None,
        slot2,
        bt,
        32,
    )
    r = (comp[20] * m[0]) ^ (comp[9] * m[1]) ^ (comp[3] * m[2])
    assert torch.equal(hashes[0, 0, 2], ((r % 73) + 138).to(torch.int32))


def test_engram_explicit_lookback_beats_slot_cache() -> None:
    """Resolution order is chunk -> explicit lookback -> slot cache.

    A real ``lookback_token_ids`` id must win even when the cache holds a
    different value for the same position; an unknown (``-1``) id must fall
    through to the cache; an image-dead id must block the n-gram instead of
    letting the cache resurrect a live lookback.
    """
    state = _tiny_engram_hash_state()
    state._cache = torch.zeros(64, dtype=torch.int32)
    comp = torch.tensor(_engram_test_token_map(), dtype=torch.int64)
    m = state.multipliers[0]
    ids = torch.tensor([5])
    positions = torch.tensor([1])
    q_cu = torch.tensor([0, 1])
    dead = torch.zeros(1, dtype=torch.bool)
    bt = torch.tensor([[0, 0]], dtype=torch.int32)
    # Chunk start is position 1, so shift 1 (position 0) is out of chunk and
    # reads slot 0: seed it with a live value distinct from any lookback id.
    state._cache[0] = int(comp[9])

    def _hash_with_lookback(token_id: int) -> torch.Tensor:
        return state(
            ids,
            positions,
            q_cu,
            dead,
            torch.tensor([[token_id]], dtype=torch.int64),
            None,
            bt,
            32,
        )

    chunk_roll = comp[5] * m[0]
    # A supplied id (7) outranks the cache's 9.
    explicit = _hash_with_lookback(7)
    explicit_roll = chunk_roll ^ (comp[7] * m[1])
    assert torch.equal(explicit[0, 0, 0], (explicit_roll % 67).to(torch.int32))
    assert torch.equal(explicit[0, 0, 1], ((explicit_roll % 71) + 67).to(torch.int32))
    # -1 means unknown, so the slot cache supplies the lookback (9).
    unknown = _hash_with_lookback(-1)
    cache_roll = chunk_roll ^ (comp[9] * m[1])
    assert torch.equal(unknown[0, 0, 0], (cache_roll % 67).to(torch.int32))
    assert torch.equal(unknown[0, 0, 1], ((cache_roll % 71) + 67).to(torch.int32))
    # An image-dead id blocks the n-gram (pads) rather than reading the cache.
    dead_lookback = _hash_with_lookback(ENGRAM_IMAGE_SENTINEL_BASE_ID)
    assert torch.equal(dead_lookback[0, 0, 0], (chunk_roll % 67).to(torch.int32))
    assert not torch.equal(dead_lookback[0, 0, 0], (cache_roll % 67).to(torch.int32))


def test_engram_hash_state_builder_roundtrip(tmp_path) -> None:
    """build_engram_hash_state() writes a sidecar from_config can load.

    Builds a tiny tokenizer (Hello/hello fold onto one compressed id), runs
    the in-module builder end to end, and loads the result through the
    serving path. Guards the builder the from_config error message points at.
    """
    pytest.importorskip("tokenizers")
    pytest.importorskip("numpy")
    import json

    from tokenizers import Tokenizer, models

    tok = Tokenizer(
        models.WordLevel(
            vocab={"[PAD]": 0, "Hello": 1, "hello": 2, "WORLD": 3, "x": 4},
            unk_token="[PAD]",
        )
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tok.save(str(tokenizer_path))
    config = {
        "model_type": "deepseek_v41",
        "vocab_size": 5,
        "engram_layer_ids": [1],
        "engram_num_embeddings": [1000],
        "engram_max_ngram_size": 3,
        "engram_compressed_vocab_size": 4,  # [pad], hello, world, x
        "engram_pad_token_id": 0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    written = build_engram_hash_state(str(tmp_path))
    assert written == str(tmp_path / "engram_hash_state.bin")

    # First-seen compressed ids: [PAD]->0, Hello->1, hello->1, WORLD->2, x->3.
    sidecar = _load_engram_hash_state_sidecar(str(tmp_path))
    assert sidecar is not None
    assert sidecar["token_map"].tolist() == [0, 1, 1, 2, 3]
    assert sidecar["multipliers"].shape == (1, 3)
    # The serving path accepts the generated sidecar.
    layout = EngramLayout(
        layer_ids=[1],
        num_embeddings=[1000],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=5,
        compressed_vocab_size=4,
        pad_token_id=0,
    )
    state = EngramHashState.from_config(layout, {"model_path": str(tmp_path)})
    assert state.token_map.tolist() == [0, 1, 1, 2, 3]
    assert torch.equal(state.multipliers, sidecar["multipliers"])


def test_engram_embedding_tp_sharding_nondivisible() -> None:
    """Head-column TP shards reassemble exactly when tp_size does not divide.

    World sizes leaving a short slice on the last head-owning rank or ranks
    past the last head (tp_size > n_hash_cols) must still produce uniform
    blocks; concatenated and trimmed back to n_hash_cols they equal the
    unsharded lookup. Regression: the uniform block view raised a shape
    error for such world sizes (e.g. tp_size=16 on the 24-column default).
    """
    head_sizes = [7, 11, 13, 5, 3]
    dim = 32
    num_rows = sum(head_sizes)
    g = torch.Generator().manual_seed(1234)
    payload = torch.randint(0, 256, (num_rows, dim), dtype=torch.uint8, generator=g)
    # E4M3FN 0x7F/0xFF decode to NaN; NaN != NaN breaks the exact-equality
    # checks below, so remap those byte patterns to finite neighbours.
    payload = torch.where((payload == 0x7F) | (payload == 0xFF), payload - 1, payload)
    scales = torch.randint(126, 130, (num_rows, dim // 32), dtype=torch.uint8, generator=g)

    def make(tp_size: int, tp_rank: int) -> EngramEmbedding:
        emb = EngramEmbedding(num_rows, dim, head_sizes, tp_size, tp_rank)
        with torch.no_grad():
            emb.weight.copy_(payload[emb.vocab_start : emb.vocab_end])
            emb.weight_scale.copy_(scales[emb.vocab_start : emb.vocab_end])
        return emb

    ref = make(1, 0)
    t = 6
    ids = torch.zeros(t, len(head_sizes), dtype=torch.int64)
    for j, size in enumerate(head_sizes):
        ids[:, j] = torch.randint(0, size, (t,), generator=g) + sum(head_sizes[:j])

    for tp_size in (2, 3, 5, 8):
        part = (len(head_sizes) + tp_size - 1) // tp_size
        blocks = [make(tp_size, rank)._local_block(ids) for rank in range(tp_size)]
        assert all(block.shape == (t, part, dim) for block in blocks)
        # Ranks past the last head own no table rows and contribute zeros.
        for rank in range(tp_size):
            if rank * part >= len(head_sizes):
                assert not blocks[rank].any()
        gathered = torch.cat(blocks, dim=1)[:, : ref.n_hash_cols]
        assert torch.equal(gathered, ref.lookup(ids))


def test_engram_embedding_int8_storage_dequant() -> None:
    """int8_sym storage (Eco-Tech W8A8 export) dequantizes payload * fp32 group scales."""
    head_sizes = [7, 11, 13, 5, 3]
    dim = 32
    num_rows = sum(head_sizes)
    g = torch.Generator().manual_seed(4321)
    payload = torch.randint(-127, 128, (num_rows, dim), dtype=torch.int8, generator=g)
    scales = torch.rand(num_rows, dim // 32, generator=g).float() * 0.05 + 0.01

    emb = EngramEmbedding(num_rows, dim, head_sizes, 1, 0, storage=ENGRAM_STORAGE_INT8)
    assert emb.weight.dtype == torch.int8
    assert emb.weight_scale.dtype == torch.float32
    with torch.no_grad():
        emb.weight.copy_(payload)
        emb.weight_scale.copy_(scales)

    t = 9
    ids = torch.zeros(t, len(head_sizes), dtype=torch.int64)
    for j, size in enumerate(head_sizes):
        ids[:, j] = torch.randint(0, size, (t,), generator=g) + sum(head_sizes[:j])
    out = emb.lookup(ids)
    expected = (payload[ids].float() * scales[ids].repeat_interleave(32, dim=-1)).to(torch.bfloat16)
    assert out.shape == (t, len(head_sizes), dim)
    assert torch.equal(out, expected)


def _quarot_gate_reference(
    eng: Engram,
    hidden: torch.Tensor,
    staged_rows: torch.Tensor,
    unrotate: torch.Tensor | None,
) -> torch.Tensor:
    """Manual Engram.forward math (gate dot optionally on the un-rotated stream)."""
    rows = staged_rows.to(eng.wkv.weight.dtype)
    kv = eng.wkv(rows.flatten(-2))
    num_tokens, hc_mult, dim = hidden.shape
    keys = kv[:, : hc_mult * dim].view(num_tokens, hc_mult, dim).float()
    value = kv[:, hc_mult * dim :].float()
    h = hidden.float()
    gate_hidden = h @ unrotate.to(torch.float32) if unrotate is not None else h
    hidden_rms = torch.rsqrt(gate_hidden.square().mean(-1) + eng.eps)
    key_rms = torch.rsqrt(keys.square().mean(-1) + eng.eps)
    dot = (gate_hidden * eng.q_weight.float() * eng.k_weight.float() * keys).sum(-1)
    dot = dot * hidden_rms * key_rms * float(dim) ** -0.5
    gate_input = dot.abs().clamp_min(1e-6).sqrt()
    gate_input = torch.where(dot < 0.0, -gate_input, gate_input)
    gate = torch.sigmoid(gate_input)
    return (h + gate.unsqueeze(-1) * value.unsqueeze(1)).to(hidden.dtype)


def test_engram_gate_quarot_unrotation() -> None:
    """With gate_unrotate the gate dot runs on h_rot @ Q.T; V stays rotated-basis."""
    layout = EngramLayout(
        layer_ids=[1],
        num_embeddings=[290],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )
    dim, hc_mult = 48, 2
    g = torch.Generator().manual_seed(99)
    q_orth = torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]

    def make(gate_unrotate: torch.Tensor | None) -> Engram:
        eng = Engram(
            layout,
            0,
            dim,
            hc_mult,
            rms_norm_eps=1e-6,
            tp_size=1,
            tp_rank=0,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            storage=ENGRAM_STORAGE_INT8,
            gate_unrotate=gate_unrotate,
        )
        with torch.no_grad():
            eng.wkv.weight.copy_(torch.randn(eng.wkv.weight.shape, generator=g).bfloat16())
            eng.q_weight.copy_(torch.randn(hc_mult, dim, generator=g).bfloat16())
            eng.k_weight.copy_(torch.randn(hc_mult, dim, generator=g).bfloat16())
        return eng

    t, n_hash_cols = 7, layout.n_hash_cols
    staged = torch.randn(t, n_hash_cols, layout.head_dim, generator=g).bfloat16()
    hidden = torch.randn(t, hc_mult, dim, generator=g).bfloat16()

    plain = make(None)
    plain._staged_rows = staged
    out_plain = plain(hidden, torch.zeros(t, n_hash_cols, dtype=torch.int64))
    assert torch.equal(out_plain, _quarot_gate_reference(plain, hidden, staged, None))

    rotated = make(q_orth.t().contiguous())
    rotated.wkv.load_state_dict(plain.wkv.state_dict())
    with torch.no_grad():
        rotated.q_weight.copy_(plain.q_weight)
        rotated.k_weight.copy_(plain.k_weight)
    rotated._staged_rows = staged
    out_rot = rotated(hidden, torch.zeros(t, n_hash_cols, dtype=torch.int64))
    assert torch.equal(out_rot, _quarot_gate_reference(rotated, hidden, staged, q_orth.t()))
    # The un-rotation must actually change the gated output (non-identity Q).
    assert not torch.equal(out_plain, out_rot)


def test_load_quarot_gate_unrotate(tmp_path, monkeypatch) -> None:
    """Q.T loads from optional/quarot.safetensors only for rotated checkpoints."""
    import json as _json
    import sys

    import safetensors.torch as _st
    from safetensors import SafetensorError

    model_path = tmp_path / "eco"
    (model_path / "optional").mkdir(parents=True)
    g = torch.Generator().manual_seed(7)
    q = torch.linalg.qr(torch.randn(64, 64, generator=g))[0]

    def write_config(rotated: bool) -> None:
        (model_path / "config.json").write_text(
            _json.dumps({"engram_rotation_config": {"value_projection_rotated": rotated}})
        )

    # Unrotated config -> None (no optional file needed).
    write_config(False)
    assert load_quarot_gate_unrotate(str(model_path), torch.device("cpu")) is None

    # Rotated config without the quarot file stays None only for missing
    # config; with the file present the transpose must load exactly.
    write_config(True)
    _st.save_file({"global_rotation": q.contiguous()}, str(model_path / "optional" / "quarot.safetensors"))
    loaded = load_quarot_gate_unrotate(str(model_path), torch.device("cpu"))
    assert loaded is not None
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, q.t().contiguous().to(torch.float32))

    # A bf16-saved rotation matrix loads with the header-derived dtype and is
    # still returned as fp32.
    q_bf = q.to(torch.bfloat16)
    _st.save_file({"global_rotation": q_bf.contiguous()}, str(model_path / "optional" / "quarot.safetensors"))
    loaded_bf = load_quarot_gate_unrotate(str(model_path), torch.device("cpu"))
    assert loaded_bf is not None
    assert loaded_bf.dtype == torch.float32
    assert torch.equal(loaded_bf, q_bf.t().contiguous().to(torch.float32))

    # Manual header-parse branch (safetensors import unavailable): both fp32
    # and bf16 payloads decode exactly.
    for q_saved in (q.contiguous(), q_bf.contiguous()):
        _st.save_file({"global_rotation": q_saved}, str(model_path / "optional" / "quarot.safetensors"))
        monkeypatch.setitem(sys.modules, "safetensors.torch", None)
        loaded_manual = load_quarot_gate_unrotate(str(model_path), torch.device("cpu"))
        monkeypatch.undo()
        assert loaded_manual is not None
        assert loaded_manual.dtype == torch.float32
        assert torch.equal(loaded_manual, q_saved.t().contiguous().to(torch.float32))

    # Corrupt quarot payload with safetensors available must raise instead of
    # silently falling back to the manual parser.
    (model_path / "optional" / "quarot.safetensors").write_bytes(b"garbage-bytes")
    with pytest.raises(SafetensorError):
        load_quarot_gate_unrotate(str(model_path), torch.device("cpu"))


def test_engram_gate_math_and_mask_passthrough() -> None:
    lm = _make_forward_lm(torch.float32, seed=43)
    cfg = lm.cfg
    eng = lm.model.layers[1].engram
    assert eng is not None
    with torch.no_grad():
        eng.q_weight.normal_(0.0, 0.05)
        eng.k_weight.normal_(0.0, 0.05)
        eng.wkv.weight.normal_(0.0, 0.05)
    # A deterministic table row: ue8m0 scale 2^0 dequantizes e4m3 payloads.
    rows, dim = eng.embed_tokens.weight.shape
    eng.embed_tokens.weight.zero_()
    payload = torch.tensor([0x38, 0xC0], dtype=torch.uint8)  # 1.0 and -1.0
    eng.embed_tokens.weight[:, 0] = payload[0]
    eng.embed_tokens.weight[:, 1] = payload[1]
    eng.embed_tokens.weight_scale.fill_(127)  # 2^0

    t, hc, hidden = 2, cfg.hc_mult, cfg.hidden_size
    hash_ids = torch.zeros(t, eng.embed_tokens.n_hash_cols, dtype=torch.int64)
    hidden_states = torch.randn(t, hc, hidden)

    def reference() -> torch.Tensor:
        rows_gathered = eng.embed_tokens.lookup(hash_ids).to(eng.wkv.weight.dtype)
        kv = eng.wkv(rows_gathered.flatten(-2))
        keys = kv[:, : hc * hidden].view(t, hc, hidden).float()
        value = kv[:, hc * hidden :].float()
        h = hidden_states.float()
        q = eng.q_weight.float()
        k = eng.k_weight.float()
        eps = cfg.rms_norm_eps
        h_rms = torch.rsqrt(h.square().mean(-1) + eps)
        k_rms = torch.rsqrt(keys.square().mean(-1) + eps)
        dot = (h * q * k * keys).sum(-1) * h_rms * k_rms * hidden**-0.5
        gate_input = dot.abs().clamp_min(1e-6).sqrt()
        gate_input = torch.where(dot < 0.0, -gate_input, gate_input)
        gate = torch.sigmoid(gate_input)
        return h + gate.unsqueeze(-1) * value.unsqueeze(1)

    out = eng(hidden_states, hash_ids)
    assert torch.allclose(out, reference(), atol=1e-4)

    # token_mask False shuts the gate: those rows pass through untouched.
    mask = torch.tensor([True, False])
    out_masked = eng(hidden_states, hash_ids, mask)
    assert torch.allclose(out_masked[0], out[0], atol=1e-6)
    assert torch.allclose(out_masked[1], hidden_states[1], atol=1e-6)

    # Image sentinels break n-grams and close the gate in the model mask.
    assert cfg.image_token_id == ENGRAM_IMAGE_SENTINEL_BASE_ID


@npu_only
def test_engram_injection_shifts_residual_stream() -> None:
    """The model wires hashing, staging and the gated injection end to end."""
    device = _active_device()
    assert device.type == "npu"
    lm = _make_forward_lm(torch.float32, seed=45, device=device)
    assert lm.model.norm.weight.device.type == "npu"
    cfg = lm.cfg
    n_seqs, prompt_len = 1, 6
    harness = _V41ForwardHarness(lm, _DSV41_FORWARD_CONFIG, n_seqs, torch.float32, device=device)
    assert harness.layer_caches[0].swa.device.type == "npu"
    input_ids = torch.randint(0, cfg.vocab_size, (n_seqs, prompt_len)).to(device)
    with torch.inference_mode():
        hidden = harness.step(
            input_ids.reshape(-1),
            torch.arange(prompt_len, device=device),
            [prompt_len],
            [prompt_len],
            is_prefill=True,
        )
    assert torch.isfinite(hidden).all()
    # The engram hash state is wired through the model and its slot cache
    # was sized from the bound SWA cache.
    assert lm.model.engram_hash is not None
    assert lm.model.engram_hash._cache is not None
    assert lm.model.engram_hash._cache.numel() == harness.layer_caches[0].swa.numel() // cfg.head_dim
    # The slot cache holds the chunk's compressed ids (dead tokens excluded).
    state = lm.model.engram_hash
    comp = state.token_map[input_ids.reshape(-1).to(torch.int64)]
    flat_cache = state._cache
    for pos, value in enumerate(comp.tolist()):
        slot = int(harness.swa_bt[0, pos // cfg.window_size]) * cfg.window_size + pos % cfg.window_size
        assert int(flat_cache[slot]) == value
    # Every engram layer staged its rows before the decoder loop, and the
    # staged rows are the dequantized table lookups of the hash ids.
    eng = lm.model.layers[1].engram
    assert eng is not None and eng._staged_rows is not None
    n_hash_cols = cfg.engram_n_heads * (cfg.engram_max_ngram_size - 1)
    assert eng._staged_rows.shape == (prompt_len, n_hash_cols, cfg.engram_head_dim)
    assert torch.isfinite(eng._staged_rows.float()).all()
    # Hash ids stay inside the layer's prime bucket ranges.
    hashes = state(
        input_ids.reshape(-1),
        torch.arange(prompt_len, device=device),
        torch.tensor([0, prompt_len], device=device),
        torch.zeros(prompt_len, dtype=torch.bool, device=device),
        None,
        None,
        None,
        0,
    )
    assert (hashes >= 0).all() and (hashes < cfg.engram_num_embeddings[0]).all()
