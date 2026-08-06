# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the Kimi K3 Python text-model scaffold and weight loader."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F


_mock_ops = MagicMock()


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


_mock_ops.rms_norm.side_effect = _rms_norm
sys.modules.setdefault("xllm.python.ops", _mock_ops)
sys.modules.setdefault("xllm.python.ops.compute", _mock_ops)

from xllm.python.model_executor.forward_context import (  # noqa: E402
    ForwardContext,
    forward_context,
)
from xllm.python.models.kimi_k3 import (  # noqa: E402
    KimiK3ForConditionalGeneration,
)
from xllm.python.models.kimi_k3_gated_mla import KimiK3GatedMLA  # noqa: E402
from xllm.python.models.kimi_k3_text import (  # noqa: E402
    KimiK3AttentionPlaceholder,
    KimiK3ForCausalLM,
    KimiK3MLAAttention,
    KimiK3TextConfig,
)
from xllm.python.models.kimi_k3_vit import KimiK3VisionModel  # noqa: E402


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]

    def get_sharded_tensor(
        self,
        name: str,
        dim: int,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        tensor = self.get_tensor(name)
        if world_size == 1:
            return tensor
        shard_size = tensor.shape[dim] // world_size
        return tensor.narrow(dim, rank * shard_size, shard_size).contiguous()

    def get_dict_with_prefix(self, prefix: str) -> "_StateDict":
        return _StateDict(
            {
                name[len(prefix) :]: tensor
                for name, tensor in self._tensors.items()
                if name.startswith(prefix)
            }
        )

    def get_dict_with_prefixes(self, prefixes: list[str]) -> "_StateDict":
        for prefix in prefixes:
            state_dict = self.get_dict_with_prefix(prefix)
            if state_dict.size() > 0:
                return state_dict
        return _StateDict({})

    def size(self) -> int:
        return len(self._tensors)

    def keys(self) -> list[str]:
        return list(self._tensors)


def _tiny_config() -> dict:
    return {
        "device": "cpu",
        "dtype": "float32",
        "text_config": {
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": None,
            "intermediate_size": 16,
            "vocab_size": 16,
            "rms_norm_eps": 1e-5,
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "attn_res_block_size": 2,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_experts": 2,
            "num_experts_per_token": 1,
            "num_shared_experts": 1,
            "moe_intermediate_size": 4,
            "routed_expert_hidden_size": 4,
            "latent_moe_use_norm": True,
            "moe_renormalize": True,
            "moe_router_activation_func": "sigmoid",
            "routed_scaling_factor": 1.0,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "mla_use_nope": True,
            "mla_use_rope": False,
            "mla_use_output_gate": True,
            "linear_attn_config": {
                "head_dim": 4,
                "kda_layers": [1],
                "full_attn_layers": [2],
            },
        },
    }


def _multimodal_tiny_config() -> dict:
    config = _tiny_config()
    config["vision_config"] = {
        "patch_size": 2,
        "in_chans": 3,
        "vt_hidden_size": 4,
        "vt_intermediate_size": 8,
        "vt_num_attention_heads": 2,
        "vt_num_hidden_layers": 1,
        "qkv_hidden_size": 8,
        "init_pos_emb_height": 2,
        "init_pos_emb_width": 2,
        "init_pos_emb_time": 2,
        "merge_kernel_size": [2, 2],
        "mm_hidden_size": 4,
        "text_hidden_size": 8,
    }
    return config


def _weight(shape: tuple[int, ...], value: float) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float32)


def _int8_weight(shape: tuple[int, ...], value: int) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.int8)


def _checkpoint() -> dict[str, torch.Tensor]:
    weights = {
        "language_model.model.embed_tokens.weight": _weight((16, 8), 1),
        "language_model.model.output_attn_res_norm.weight": _weight((8,), 2),
        "language_model.model.output_attn_res_proj.weight": _weight((1, 8), 3),
        "language_model.model.norm.weight": _weight((8,), 4),
        "language_model.lm_head.weight": _weight((16, 8), 5),
    }
    for layer_id in range(2):
        prefix = f"language_model.model.layers.{layer_id}."
        weights.update(
            {
                prefix + "input_layernorm.weight": _weight((8,), 10 + layer_id),
                prefix + "post_attention_layernorm.weight": _weight((8,), 12 + layer_id),
                prefix + "self_attention_res_norm.weight": _weight((8,), 14 + layer_id),
                prefix + "mlp_res_norm.weight": _weight((8,), 16 + layer_id),
                prefix + "self_attention_res_proj.weight": _weight((1, 8), 18 + layer_id),
                prefix + "mlp_res_proj.weight": _weight((1, 8), 20 + layer_id),
            }
        )
    dense_prefix = "language_model.model.layers.0.mlp."
    weights.update(
        {
            dense_prefix + "gate_proj.weight": _weight((16, 8), 30),
            dense_prefix + "up_proj.weight": _weight((16, 8), 31),
            dense_prefix + "down_proj.weight": _weight((8, 16), 32),
        }
    )
    moe_prefix = "language_model.model.layers.1.block_sparse_moe."
    weights.update(
        {
            moe_prefix + "gate.weight": _weight((2, 8), 40),
            moe_prefix + "gate.e_score_correction_bias": _weight((2,), 41),
            moe_prefix + "routed_expert_down_proj.weight": _weight((4, 8), 42),
            moe_prefix + "routed_expert_up_proj.weight": _weight((8, 4), 43),
            moe_prefix + "routed_expert_norm.weight": _weight((4,), 44),
            moe_prefix + "shared_experts.gate_proj.weight": _weight((4, 8), 45),
            moe_prefix + "shared_experts.up_proj.weight": _weight((4, 8), 46),
            moe_prefix + "shared_experts.down_proj.weight": _weight((8, 4), 47),
        }
    )
    for expert_id in range(2):
        expert_prefix = moe_prefix + f"experts.{expert_id}."
        weights.update(
            {
                expert_prefix + "w1.weight": _weight((4, 4), 50 + expert_id),
                expert_prefix + "w3.weight": _weight((4, 4), 52 + expert_id),
                expert_prefix + "w2.weight": _weight((4, 4), 54 + expert_id),
            }
        )
    mla_prefix = "language_model.model.layers.1.self_attn."
    weights.update(
        {
            mla_prefix + "q_a_proj.weight": _weight((4, 8), 60),
            mla_prefix + "q_a_layernorm.weight": _weight((4,), 61),
            mla_prefix + "q_b_proj.weight": _weight((8, 4), 62),
            mla_prefix + "kv_a_proj_with_mqa.weight": _weight((6, 8), 63),
            mla_prefix + "kv_a_layernorm.weight": _weight((4,), 64),
            mla_prefix + "kv_b_proj.weight": torch.arange(
                8 * 4, dtype=torch.float32
            ).reshape(8, 4),
            mla_prefix + "g_proj.weight": _weight((4, 8), 66),
            mla_prefix + "o_proj.weight": _weight((8, 4), 67),
        }
    )
    return weights


def _quantized_tiny_config() -> dict:
    config = _tiny_config()
    config.update(
        {
            "quantize_type": "w4a8_dynamic",
            "quant_method": "ascend_int4",
            "quant_version": "1.0.0",
            "quant_group_size": 0,
        }
    )
    return config


def _quantized_checkpoint() -> dict[str, torch.Tensor]:
    weights = _checkpoint()
    dense_prefix = "language_model.model.layers.0.mlp."
    moe_prefix = "language_model.model.layers.1.block_sparse_moe."

    for projection in ("gate_proj", "up_proj", "down_proj"):
        weights.pop(dense_prefix + projection + ".weight")
    weights.update(
        {
            dense_prefix + "gate_proj.weight": _int8_weight((16, 8), 1),
            dense_prefix + "gate_proj.weight_scale": _weight((16, 1), 2),
            dense_prefix + "gate_proj.weight_offset": _weight((16, 1), 0),
            dense_prefix + "up_proj.weight": _int8_weight((16, 8), 3),
            dense_prefix + "up_proj.weight_scale": _weight((16, 1), 4),
            dense_prefix + "up_proj.weight_offset": _weight((16, 1), 0),
            dense_prefix + "down_proj.weight": _int8_weight((8, 16), 5),
            dense_prefix + "down_proj.weight_scale": _weight((8, 1), 6),
            dense_prefix + "down_proj.weight_offset": _weight((8, 1), 0),
        }
    )

    quantized_linears = {
        "routed_expert_down_proj": ((4, 8), 10),
        "routed_expert_up_proj": ((8, 4), 11),
        "shared_experts.gate_proj": ((4, 8), 12),
        "shared_experts.up_proj": ((4, 8), 13),
        "shared_experts.down_proj": ((8, 4), 14),
    }
    for projection, (shape, value) in quantized_linears.items():
        weights[moe_prefix + projection + ".weight"] = _int8_weight(shape, value)
        weights[moe_prefix + projection + ".weight_scale"] = _weight(
            (shape[0], 1),
            value + 20,
        )
        weights[moe_prefix + projection + ".weight_offset"] = _weight(
            (shape[0], 1),
            0,
        )

    for expert_id in range(2):
        expert_prefix = moe_prefix + f"experts.{expert_id}."
        for projection in ("w1", "w3", "w2"):
            weights.pop(expert_prefix + projection + ".weight")
        for projection, value in (("w1", 20), ("w3", 30), ("w2", 40)):
            weights[expert_prefix + projection + ".weight"] = _int8_weight(
                (2, 4),
                value + expert_id,
            )
            weights[expert_prefix + projection + ".weight_scale"] = _weight(
                (4, 1),
                value + expert_id + 1,
            )
            weights[expert_prefix + projection + ".weight_offset"] = _weight(
                (4, 1),
                0,
            )
            scale_bias_shape = (4, 16) if projection == "w2" else (4, 1)
            weights[expert_prefix + projection + ".scale_bias"] = _weight(
                scale_bias_shape,
                value + expert_id + 2,
            )
    mla_prefix = "language_model.model.layers.1.self_attn."
    for projection in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa"):
        weights.pop(mla_prefix + projection + ".weight")
    for projection, shape, weight_value, scale_value, offset_value in (
        ("q_a_proj", (4, 8), 7, 2, 1),
        ("q_b_proj", (8, 4), 9, 3, 2),
        ("kv_a_proj_with_mqa", (6, 8), 11, 4, 3),
    ):
        weights[mla_prefix + projection + ".weight"] = _int8_weight(
            shape, weight_value
        )
        weights[mla_prefix + projection + ".weight_scale"] = _weight(
            (shape[0], 1), scale_value
        )
        weights[mla_prefix + projection + ".weight_offset"] = _weight(
            (shape[0], 1), offset_value
        )
    return weights


def test_config_reads_head_dim_from_linear_attention() -> None:
    config = KimiK3TextConfig.from_dict(_tiny_config())

    assert config.n_layers == 2
    assert config.head_dim == 4
    assert config.num_experts == 2
    assert config.q_lora_rank == 4
    assert config.kv_lora_rank == 4
    assert config.qk_nope_head_dim == 2
    assert config.qk_rope_head_dim == 2
    assert config.v_head_dim == 2
    assert config.is_kda_layer(0)
    assert not config.is_mla_layer(0)
    assert config.is_mla_layer(1)


def test_decoder_registers_moe_under_checkpoint_name() -> None:
    model = KimiK3ForCausalLM(_tiny_config())

    assert hasattr(model.model.layers[1], "block_sparse_moe")
    assert not hasattr(model.model.layers[1], "mlp")
    assert "block_sparse_moe.experts.w13_weight" in dict(
        model.model.layers[1].named_parameters()
    )
    assert "block_sparse_moe.gate.e_score_correction_bias" in dict(
        model.model.layers[1].named_parameters()
    )


def test_multimodal_model_owns_real_text_and_vision_models() -> None:
    model = KimiK3ForConditionalGeneration(_multimodal_tiny_config())

    assert isinstance(model.language_model, KimiK3ForCausalLM)
    assert isinstance(model.vision_model, KimiK3VisionModel)
    assert model.model is model.language_model.model


def test_multimodal_model_merges_image_and_token_embeddings() -> None:
    model = KimiK3ForConditionalGeneration(_multimodal_tiny_config())
    embedding_weight = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    model.language_model.model.embed_tokens.weight.data.copy_(embedding_weight)
    input_ids = torch.tensor([1, 2, 3])
    multimodal_embeds = torch.tensor(
        [[-1.0] * 8, [-2.0] * 8],
        dtype=torch.float32,
    )
    multimodal_mask = torch.tensor([True, False, True])

    inputs_embeds = model.get_input_embeddings(
        input_ids,
        multimodal_embeds,
        multimodal_mask,
    )

    torch.testing.assert_close(inputs_embeds[0], multimodal_embeds[0])
    torch.testing.assert_close(inputs_embeds[1], embedding_weight[2])
    torch.testing.assert_close(inputs_embeds[2], multimodal_embeds[1])


def test_multimodal_model_dispatches_weights_to_both_submodels() -> None:
    model = KimiK3ForConditionalGeneration(_multimodal_tiny_config())
    model.vision_model.load_weights = MagicMock(return_value={"vision.weight"})
    model.language_model.load_weights = MagicMock(return_value={"text.weight"})
    state_dicts = [object()]

    loaded = model.load_weights(state_dicts, tp_rank=0, tp_size=1)

    assert loaded == {"vision.weight", "text.weight"}
    model.vision_model.load_weights.assert_called_once_with(state_dicts, 0, 1)
    model.language_model.load_weights.assert_called_once_with(state_dicts, 0, 1)


def test_text_model_uses_precomputed_input_embeddings() -> None:
    model = KimiK3ForCausalLM(_tiny_config())
    text_model = model.model
    text_model.layers = torch.nn.ModuleList()
    text_model.embed_tokens.forward = MagicMock(
        side_effect=AssertionError("token embedding should be bypassed")
    )
    text_model.output_attn_res_proj.weight.data.zero_()
    text_model.output_attn_res_norm.weight.data.fill_(1)
    text_model.norm.weight.data.fill_(1)
    input_ids = torch.tensor([1, 2, 3])
    positions = torch.tensor([0, 1, 2])
    inputs_embeds = torch.randn(3, 8)

    hidden_states = text_model(input_ids, positions, inputs_embeds)

    torch.testing.assert_close(hidden_states, text_model.norm(inputs_embeds))
    text_model.embed_tokens.forward.assert_not_called()


def test_model_dispatches_language_model_weights_to_owners() -> None:
    model = KimiK3ForCausalLM(_tiny_config())

    loaded = model.load_weights([_StateDict(_checkpoint())], tp_rank=0, tp_size=1)

    dense_mlp = model.model.layers[0].mlp
    mla = model.model.layers[1].self_attn
    routed_experts = model.model.layers[1].block_sparse_moe.experts
    assert isinstance(model.model.layers[0].self_attn, KimiK3AttentionPlaceholder)
    assert isinstance(mla, KimiK3MLAAttention)
    assert isinstance(mla, KimiK3GatedMLA)
    torch.testing.assert_close(dense_mlp.gate_up_proj.weight[:16], _weight((16, 8), 30))
    torch.testing.assert_close(dense_mlp.gate_up_proj.weight[16:], _weight((16, 8), 31))
    torch.testing.assert_close(routed_experts.w13_weight[0, :4], _weight((4, 4), 50))
    torch.testing.assert_close(routed_experts.w13_weight[0, 4:], _weight((4, 4), 52))
    torch.testing.assert_close(model.lm_head.weight, _weight((16, 8), 5))
    kv_b = torch.arange(8 * 4, dtype=torch.float32).reshape(2, 4, 4)
    torch.testing.assert_close(mla.W_UK, kv_b[:, :2])
    torch.testing.assert_close(mla.W_UV, kv_b[:, 2:].transpose(1, 2))
    assert "model.layers.1.self_attn.kv_b_proj.weight" in loaded
    assert "model.layers.1.block_sparse_moe.experts.1.w2.weight" in loaded


def test_text_mla_adapter_matches_shared_eager_math() -> None:
    model = KimiK3ForCausalLM(_tiny_config())
    checkpoint = _checkpoint()
    model.load_weights([_StateDict(checkpoint)], tp_rank=0, tp_size=1)
    mla = model.model.layers[1].self_attn
    assert isinstance(mla, KimiK3MLAAttention)

    prefix = "language_model.model.layers.1.self_attn."
    eager = KimiK3GatedMLA(mla.config, dtype=torch.float32)
    eager.load_checkpoint_weights(
        {
            name[len(prefix) :]: tensor
            for name, tensor in checkpoint.items()
            if name.startswith(prefix)
        }
    )

    class _DenseMlaBackend:
        def __init__(self) -> None:
            self.topk = object()

        def execute_mla(
            self,
            q_latent: torch.Tensor,
            q_pe: torch.Tensor,
            k_latent: torch.Tensor,
            k_pe: torch.Tensor,
            layer: KimiK3MLAAttention,
            topk: int | None,
        ) -> torch.Tensor:
            self.topk = topk
            num_heads = q_latent.shape[1]
            key_latent = k_latent.expand(-1, num_heads, -1)
            key_pe = k_pe.expand(-1, num_heads, -1)
            query = torch.cat((q_latent, q_pe), dim=-1).transpose(0, 1).unsqueeze(0)
            key = torch.cat((key_latent, key_pe), dim=-1).transpose(0, 1).unsqueeze(0)
            value = key_latent.transpose(0, 1).unsqueeze(0)
            return F.scaled_dot_product_attention(
                query, key, value, is_causal=True, scale=layer.scale
            ).squeeze(0).transpose(0, 1)

    backend = _DenseMlaBackend()
    hidden_states = torch.randn(
        5, model.cfg.hidden_size, generator=torch.Generator().manual_seed(17)
    )
    positions = torch.arange(hidden_states.shape[0])
    with forward_context(ForwardContext(backend, torch.device("cpu"))):
        actual = mla(hidden_states, positions)
    expected = eager(hidden_states, sequence_lengths=[hidden_states.shape[0]])

    assert backend.topk is None
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_quantized_model_loads_w8_and_packed_w4_tensors() -> None:
    model = KimiK3ForCausalLM(_quantized_tiny_config())

    loaded = model.load_weights(
        [_StateDict(_quantized_checkpoint())],
        tp_rank=0,
        tp_size=1,
    )

    dense_mlp = model.model.layers[0].mlp
    routed_moe = model.model.layers[1].block_sparse_moe
    experts = routed_moe.experts
    assert dense_mlp.gate_up_proj.weight.dtype == torch.int8
    assert dense_mlp.gate_up_proj.weight.shape == (8, 32)
    torch.testing.assert_close(
        dense_mlp.gate_up_proj.weight[:, :16],
        _int8_weight((16, 8), 1).transpose(0, 1),
    )
    torch.testing.assert_close(
        dense_mlp.gate_up_proj.weight_scale[:16],
        _weight((16,), 2),
    )
    assert routed_moe.routed_expert_down_proj.weight.shape == (8, 4)
    assert experts.w13_weight.shape == (2, 4, 4)
    assert experts.w2_weight.shape == (2, 2, 4)
    torch.testing.assert_close(
        experts.w13_weight[0, :2],
        _int8_weight((2, 4), 20),
    )
    torch.testing.assert_close(
        experts.w13_weight_scale[0, :4],
        _weight((4, 1), 21),
    )
    torch.testing.assert_close(
        experts.w2_scale_bias[1],
        _weight((4, 16), 43),
    )
    assert (
        "model.layers.1.block_sparse_moe.experts.1.w2.scale_bias"
        in loaded
    )
    mla = model.model.layers[1].self_attn
    assert isinstance(mla, KimiK3MLAAttention)
    torch.testing.assert_close(mla.q_a_proj.weight, _weight((4, 8), 12))
    torch.testing.assert_close(mla.q_b_proj.weight, _weight((8, 4), 21))
    torch.testing.assert_close(
        mla.kv_a_proj_with_mqa.weight,
        _weight((6, 8), 32),
    )
    assert "model.layers.1.self_attn.q_b_proj.weight_scale" in loaded


def test_quantized_weight_loading_shards_tp_dimensions() -> None:
    config = _quantized_tiny_config()
    config.update({"tp_size": 2, "tp_rank": 1})
    model = KimiK3ForCausalLM(config)

    model.load_weights(
        [_StateDict(_quantized_checkpoint())],
        tp_rank=1,
        tp_size=2,
    )

    dense_mlp = model.model.layers[0].mlp
    experts = model.model.layers[1].block_sparse_moe.experts
    assert dense_mlp.gate_up_proj.weight.shape == (8, 16)
    assert dense_mlp.down_proj.weight.shape == (8, 8)
    assert experts.w13_weight.shape == (2, 2, 4)
    assert experts.w2_weight.shape == (2, 2, 2)
    assert experts.w2_scale_bias.shape == (2, 4, 8)
    mla = model.model.layers[1].self_attn
    assert isinstance(mla, KimiK3MLAAttention)
    assert isinstance(mla, KimiK3GatedMLA)
    assert mla.q_b_proj.weight.shape == (4, 4)
    assert mla.kv_b_proj.weight.shape == (4, 4)
    assert mla.g_proj.weight.shape == (2, 8)
    assert mla.o_proj.weight.shape == (8, 2)
    expected_kv_b = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)[4:]
    expected_kv_b = expected_kv_b.reshape(1, 4, 4)
    torch.testing.assert_close(mla.W_UK, expected_kv_b[:, :2])
    torch.testing.assert_close(mla.W_UV, expected_kv_b[:, 2:].transpose(1, 2))
    torch.testing.assert_close(mla.q_a_proj.weight, _weight((4, 8), 12))
    torch.testing.assert_close(mla.q_b_proj.weight, _weight((4, 4), 21))
    torch.testing.assert_close(
        mla.kv_a_proj_with_mqa.weight,
        _weight((6, 8), 32),
    )


@pytest.mark.parametrize("companion", ("weight_scale", "weight_offset"))
def test_quantized_mla_weight_requires_scale_and_offset(companion: str) -> None:
    checkpoint = _quantized_checkpoint()
    mla_prefix = "language_model.model.layers.1.self_attn."
    checkpoint.pop(mla_prefix + "q_a_proj." + companion)
    model = KimiK3ForCausalLM(_quantized_tiny_config())

    with pytest.raises(KeyError, match=f"q_a_proj.{companion}"):
        model.load_weights([_StateDict(checkpoint)], tp_rank=0, tp_size=1)


def test_weight_loading_accumulates_across_state_dict_shards() -> None:
    checkpoint = _checkpoint()
    keys = list(checkpoint)
    state_dicts = [
        _StateDict({key: checkpoint[key] for key in keys[offset::3]})
        for offset in range(3)
    ]
    model = KimiK3ForCausalLM(_tiny_config())

    model.load_weights(state_dicts, tp_rank=0, tp_size=1)

    torch.testing.assert_close(
        model.model.layers[0].mlp.down_proj.weight,
        _weight((8, 16), 32),
    )
    torch.testing.assert_close(
        model.model.layers[1].block_sparse_moe.experts.w2_weight[1],
        _weight((4, 4), 55),
    )
