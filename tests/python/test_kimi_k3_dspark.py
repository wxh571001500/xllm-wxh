# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""CPU contract tests for the Kimi-K3 DSpark PyTorch graph."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch_npu")

from xllm.python.models import kimi_k3_dspark


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def keys(self) -> list[str]:
        return list(self._tensors)

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
        return tensor.chunk(world_size, dim=dim)[rank].contiguous()


class _RecordingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_keys: list[str] = []

    def load_weights(
        self,
        state_dict: object,
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        del tp_rank, tp_size
        self.seen_keys = state_dict.keys()
        return set(self.seen_keys)


class _DraftBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = None
        self.context_proj = nn.Linear(24, 8, bias=False)
        self.context_norm = nn.LayerNorm(8)
        self.final_norm = nn.LayerNorm(8)
        self.markov_w1 = nn.Embedding(16, 4)
        self.markov_w2 = nn.Linear(4, 16, bias=False)
        self.layers = nn.ModuleList([_RecordingLayer(), _RecordingLayer()])


def _config(n_layers: int = 2) -> dict[str, object]:
    return {
        "device": "cpu",
        "dtype": "float32",
        "hidden_size": 8,
        "intermediate_size": 16,
        "n_heads": 2,
        "n_layers": n_layers,
        "dspark_num_target_layers": 3,
        "dspark_target_hidden_size": 8,
        "q_lora_rank": 4,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "vocab_size": 16,
        "markov_rank": 4,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 32,
        "rope_theta": 10000.0,
        "rope_scaling_factor": 2.0,
        "rope_scaling_original_max_position_embeddings": 16,
        "rope_scaling_beta_fast": 32,
        "rope_scaling_beta_slow": 1,
        "rope_scaling_mscale": 1.0,
        "rope_scaling_mscale_all_dim": 1.0,
        "tp_size": 1,
        "tp_rank": 0,
    }


def test_config_parses_flattened_model_args() -> None:
    config = kimi_k3_dspark.KimiK3DSparkConfig.from_dict(_config())
    config.validate()

    assert config.n_layers == 2
    assert config.num_target_layers == 3
    assert config.target_hidden_size == 8
    assert config.rope_scaling_factor == 2.0
    assert config.rope_original_max_position_embeddings == 16


def test_config_uses_kimi_k3_mla_geometry_by_default() -> None:
    config = kimi_k3_dspark.KimiK3DSparkConfig.from_dict(
        {
            "hidden_size": 7168,
            "intermediate_size": 14336,
            "num_hidden_layers": 5,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "target_hidden_size": 7168,
            "num_target_layers": 5,
            "vocab_size": 163840,
            "markov_rank": 256,
        }
    )

    assert config.n_heads == 64
    assert config.rope_scaling_factor == 32.0
    assert config.rms_norm_eps == 1e-5


def test_context_kv_uses_single_head_mla_geometry() -> None:
    config = kimi_k3_dspark.KimiK3DSparkConfig.from_dict(_config(n_layers=1))
    model = kimi_k3_dspark.KimiK3DSparkModel(
        config,
        torch.float32,
        torch.device("cpu"),
    )
    target_hidden = torch.randn(3, 24)
    positions = torch.arange(3)
    slots = torch.arange(3)
    latent_cache = torch.empty(1, 3, 1, 4)
    rope_cache = torch.empty(1, 3, 1, 2)
    index_cache = torch.empty(0)

    with patch.object(kimi_k3_dspark.kernels, "write_mla_kv_cache") as write:
        hidden = model.write_context_kv(
            target_hidden,
            positions,
            slots,
            [(latent_cache, rope_cache, index_cache)],
        )

    assert hidden.shape == (3, 8)
    raw_kv = write.call_args.args[0]
    assert raw_kv.shape == (3, 6)
    assert write.call_args.args[4] is slots
    assert write.call_args.args[5] is latent_cache
    assert write.call_args.args[6] is rope_cache
    assert write.call_args.args[7:9] == (4, 2)


def test_fused_qkv_a_loader_accepts_separate_checkpoint_weights() -> None:
    config = kimi_k3_dspark.KimiK3DSparkConfig.from_dict(_config())
    projection = kimi_k3_dspark.K3DSparkFusedQKVAProjection(
        config,
        torch.float32,
        torch.device("cpu"),
    )
    query = torch.randn(4, 8)
    kv = torch.randn(6, 8)

    loaded = projection.load_weights(
        _StateDict(
            {
                "q_a_proj.weight": query,
                "kv_a_proj_with_mqa.weight": kv,
            }
        )
    )

    assert loaded == {"q_a_proj.weight", "kv_a_proj_with_mqa.weight"}
    torch.testing.assert_close(projection.projection.weight, torch.cat((query, kv)))


def test_fused_qkv_a_loader_accepts_mapped_checkpoint_weight() -> None:
    config = kimi_k3_dspark.KimiK3DSparkConfig.from_dict(_config())
    projection = kimi_k3_dspark.K3DSparkFusedQKVAProjection(
        config,
        torch.float32,
        torch.device("cpu"),
    )
    fused = torch.randn(10, 8)

    loaded = projection.load_weights(
        _StateDict({"fused_qkv_a_proj.weight": fused})
    )

    assert loaded == {"fused_qkv_a_proj.weight"}
    torch.testing.assert_close(projection.projection.weight, fused)


def test_fused_gate_up_loader_shards_each_projection_independently() -> None:
    fused = torch.arange(16, dtype=torch.float32).view(8, 2)

    shard = kimi_k3_dspark._shard_fused_gate_up(
        fused,
        tp_rank=1,
        tp_size=2,
    )

    expected = torch.cat((fused[2:4], fused[6:8]), dim=0)
    torch.testing.assert_close(shard, expected)


def test_constructor_defers_shared_target_modules() -> None:
    body = nn.Module()
    body.embed_tokens = None

    with patch.object(
        kimi_k3_dspark,
        "KimiK3DSparkModel",
        return_value=body,
    ):
        draft = kimi_k3_dspark.KimiK3DSparkForCausalLM(_config())

    assert draft.lm_head is None
    assert draft.model.embed_tokens is None

    target_lm_head = nn.Linear(8, 16, bias=False)
    target_embedding = nn.Embedding(16, 8)
    draft.lm_head = target_lm_head
    draft.model.embed_tokens = target_embedding

    assert draft.lm_head is target_lm_head
    assert draft.model.embed_tokens is target_embedding


def test_weight_loader_maps_offset_checkpoint_layers_in_order() -> None:
    body = _DraftBody()
    with patch.object(
        kimi_k3_dspark,
        "KimiK3DSparkModel",
        return_value=body,
    ):
        draft = kimi_k3_dspark.KimiK3DSparkForCausalLM(_config())

    tensors = {
        "context_proj.weight": torch.ones_like(body.context_proj.weight),
        "context_norm.weight": torch.ones_like(body.context_norm.weight),
        "final_norm.weight": torch.ones_like(body.final_norm.weight),
        "markov_head.markov_w1.weight": torch.ones_like(body.markov_w1.weight),
        "markov_head.markov_w2.weight": torch.ones_like(body.markov_w2.weight),
        "layers.93.sentinel": torch.tensor(93),
        "layers.97.sentinel": torch.tensor(97),
    }

    loaded = draft.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    assert body.layers[0].seen_keys == ["sentinel"]
    assert body.layers[1].seen_keys == ["sentinel"]
    assert "layers.93.sentinel" in loaded
    assert "layers.97.sentinel" in loaded
