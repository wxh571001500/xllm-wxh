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
from unittest.mock import MagicMock, patch

import torch


_mock_ops = MagicMock()
sys.modules.setdefault("xllm.python.ops", _mock_ops)
sys.modules.setdefault("xllm.python.ops.compute", _mock_ops)

from xllm.python.layers.moe import KimiK3MoE, KimiK3RoutedExperts  # noqa: E402
from xllm.python.models.kimi_k3 import (  # noqa: E402
    KimiK3ForConditionalGeneration,
)
from xllm.python.models.kimi_k3_text import (  # noqa: E402
    KimiK3ForCausalLM,
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


class _FixedOutput(torch.nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, *args: object) -> torch.Tensor:
        return self.output.clone()


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
            "linear_attn_config": {
                "head_dim": 4,
                "kda_layers": [],
                "full_attn_layers": [1, 2],
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
    return weights


def test_config_reads_head_dim_from_linear_attention() -> None:
    config = KimiK3TextConfig.from_dict(_tiny_config())

    assert config.n_layers == 2
    assert config.head_dim == 4
    assert config.num_experts == 2


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
    routed_experts = model.model.layers[1].block_sparse_moe.experts
    torch.testing.assert_close(dense_mlp.gate_up_proj.weight[:16], _weight((16, 8), 30))
    torch.testing.assert_close(dense_mlp.gate_up_proj.weight[16:], _weight((16, 8), 31))
    torch.testing.assert_close(routed_experts.w13_weight[0, :4], _weight((4, 4), 50))
    torch.testing.assert_close(routed_experts.w13_weight[0, 4:], _weight((4, 4), 52))
    torch.testing.assert_close(model.lm_head.weight, _weight((16, 8), 5))
    assert "model.layers.1.block_sparse_moe.experts.1.w2.weight" in loaded


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


def test_quantized_experts_prepare_runtime_layout() -> None:
    experts = KimiK3RoutedExperts(
        num_experts=2,
        hidden_size=8,
        intermediate_size=8,
        tp_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        quantized=True,
    )
    experts.w13_weight.data.fill_(1)
    experts.w2_weight.data.fill_(2)
    experts.w13_weight_scale.data.fill_(1.0)
    experts.w2_weight_scale.data.fill_(2.0)
    experts.w13_scale_bias.data.fill_(3.0)
    experts.w2_scale_bias.data.fill_(4.0)

    experts._process_quantized_weights()

    assert experts.w13_weight.shape == (2, 8, 2)
    assert experts.w13_weight.dtype == torch.int32
    assert experts.w2_weight.shape == (2, 8, 1)
    assert experts.w2_weight.dtype == torch.int32
    assert experts.w13_weight_scale.shape == (2, 16)
    assert experts.w13_weight_scale.dtype == torch.int64
    assert experts.w2_weight_scale.shape == (2, 1, 8)
    assert experts.w2_weight_scale.dtype == torch.int64
    assert experts.w13_scale_bias.shape == (2, 16)
    assert experts.w2_scale_bias.shape == (2, 8)
    torch.testing.assert_close(
        experts.w2_scale_bias,
        torch.full((2, 8), 64.0),
    )


def test_quantized_experts_execute_situ_pipeline() -> None:
    experts = KimiK3RoutedExperts(
        num_experts=2,
        hidden_size=8,
        intermediate_size=8,
        tp_size=1,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        quantized=True,
    )
    experts._runtime_weights_ready = True
    hidden_states = torch.randn(2, 8, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int64)
    topk_weights = torch.ones(2, 1, dtype=torch.bfloat16)
    sorted_hidden_states = torch.ones(2, 8, dtype=torch.int8)
    expanded_row_indices = torch.tensor([0, 1], dtype=torch.int32)
    expert_tokens = torch.tensor([1, 1], dtype=torch.int32)
    input_scale = torch.ones(2, dtype=torch.float32)
    gate_up = torch.randn(2, 16, dtype=torch.bfloat16)
    activated = torch.ones(2, 8, dtype=torch.int8)
    activated_scale = torch.ones(2, dtype=torch.float32)
    expert_output = torch.randn(2, 8, dtype=torch.bfloat16)
    expected = torch.randn(2, 8, dtype=torch.bfloat16)

    with (
        patch(
            "xllm.python.layers.moe.torch_npu.npu_moe_init_routing_v2",
            return_value=(
                sorted_hidden_states,
                expanded_row_indices,
                expert_tokens,
                input_scale,
            ),
        ) as mock_routing,
        patch(
            "xllm.python.layers.moe.torch_npu.npu_grouped_matmul",
            side_effect=([gate_up], [expert_output]),
        ) as mock_grouped_matmul,
        patch(
            "xllm.python.layers.moe._dequant_situ_quant",
            return_value=(activated, activated_scale),
        ) as mock_situ,
        patch(
            "xllm.python.layers.moe.torch_npu.npu_moe_token_unpermute",
            return_value=expected,
        ) as mock_unpermute,
    ):
        output = experts(
            hidden_states,
            topk_ids,
            topk_weights,
            beta=4.0,
            linear_beta=25.0,
        )

    assert output is expected
    assert mock_grouped_matmul.call_count == 2
    assert mock_grouped_matmul.call_args_list[0].kwargs["group_list_type"] == 1
    assert mock_grouped_matmul.call_args_list[1].kwargs["group_list_type"] == 1
    mock_routing.assert_called_once()
    mock_situ.assert_called_once_with(gate_up, 4.0, 25.0)
    mock_unpermute.assert_called_once()


def test_moe_uses_ascend_fused_topk_contract() -> None:
    config = KimiK3TextConfig.from_dict(_tiny_config())
    moe = KimiK3MoE(
        config,
        torch.float32,
        torch.device("cpu"),
        tp_size=1,
        tp_rank=0,
        routed_expert_down_proj=torch.nn.Identity(),
        routed_expert_up_proj=torch.nn.Identity(),
    )
    router_logits = torch.randn(2, 2)
    expected_weights = torch.tensor([[0.25], [0.75]])
    expected_ids = torch.tensor([[1], [0]], dtype=torch.int64)

    with patch.object(
        torch.ops._C_ascend,
        "moe_gating_top_k",
        return_value=(expected_weights, expected_ids, torch.empty(0)),
        create=True,
    ) as mock_topk:
        topk_ids, topk_weights = moe._ascend_topk(router_logits)

    torch.testing.assert_close(topk_weights, expected_weights)
    torch.testing.assert_close(topk_ids, expected_ids.to(torch.int32))
    assert mock_topk.call_args.kwargs["renorm"] == 1
    assert mock_topk.call_args.kwargs["norm_type"] == 1
    assert mock_topk.call_args.kwargs["group_count"] == 1


def test_moe_reduces_shared_expert_output_for_tp() -> None:
    config = KimiK3TextConfig.from_dict(_tiny_config())
    moe = KimiK3MoE(
        config,
        torch.float32,
        torch.device("cpu"),
        tp_size=2,
        tp_rank=0,
        routed_expert_down_proj=torch.nn.Identity(),
        routed_expert_up_proj=torch.nn.Identity(),
        shared_experts=_FixedOutput(torch.full((2, 8), 2.0)),
    )
    moe.experts = _FixedOutput(torch.ones(2, 8))
    moe.routed_expert_norm = None
    hidden_states = torch.zeros(2, 8)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)
    topk_weights = torch.ones(2, 1)

    with (
        patch.object(
            moe,
            "_topk",
            return_value=(topk_ids, topk_weights),
        ),
        patch(
            "xllm.python.layers.moe.ops.all_reduce_",
            side_effect=lambda tensor: tensor.mul_(2),
        ) as mock_all_reduce,
    ):
        output = moe(hidden_states)

    torch.testing.assert_close(output, torch.full((2, 8), 5.0))
    mock_all_reduce.assert_called_once()


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
