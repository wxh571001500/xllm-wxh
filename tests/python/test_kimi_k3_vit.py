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

"""CPU behavior tests for the Kimi K3 Python vision tower."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F


def _encoder_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sequence_lengths: Sequence[int],
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    start = 0
    for length in sequence_lengths:
        end = start + length
        q = query[start:end].transpose(0, 1).unsqueeze(0)
        k = key[start:end].transpose(0, 1).unsqueeze(0)
        v = value[start:end].transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(q, k, v)
        outputs.append(attended.squeeze(0).transpose(0, 1))
        start = end
    return torch.cat(outputs, dim=0)


_mock_ops = MagicMock()
_mock_ops.encoder_attention = _encoder_attention
sys.modules.setdefault("xllm.python.ops", _mock_ops)
sys.modules.setdefault("xllm.python.ops.compute", _mock_ops)
sys.modules.setdefault("torch_npu", MagicMock())

from xllm.python.models.kimi_k3_vit import (  # noqa: E402
    KimiK3VisionConfig,
    KimiK3VisionModel,
    _shard_vision_weight,
    tpool_patch_merger,
)


def _tiny_config() -> dict:
    return {
        "device": "cpu",
        "dtype": "float32",
        "vision_config": {
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
        },
    }


def test_config_reads_nested_k3_vision_fields() -> None:
    config = KimiK3VisionConfig.from_dict(_tiny_config())

    assert config.hidden_size == 4
    assert config.qkv_hidden_size == 8
    assert config.merge_kernel_size == (2, 2)
    config.validate()


def test_tpool_merger_pools_time_before_spatial_reorder() -> None:
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(16, 1)

    outputs = tpool_patch_merger(hidden_states, [[2, 2, 4]], (2, 2))

    assert len(outputs) == 1
    assert outputs[0].shape == (2, 4, 1)
    expected = torch.tensor(
        [[4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
    )
    torch.testing.assert_close(outputs[0].squeeze(-1), expected)


def test_tiny_vision_model_projects_each_media_item() -> None:
    torch.manual_seed(7)
    model = KimiK3VisionModel(_tiny_config())
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
    pixel_values = torch.randn(8, 3, 2, 2)
    grid_thws = torch.tensor([[2, 2, 2]], dtype=torch.int64)

    outputs = model(pixel_values, grid_thws)

    assert len(outputs) == 1
    assert outputs[0].shape == (1, 8)
    assert torch.isfinite(outputs[0]).all()


def test_qkv_weight_sharding_preserves_qkv_order() -> None:
    qkv = torch.arange(3 * 8 * 4, dtype=torch.float32).reshape(3 * 8, 4)

    shard = _shard_vision_weight(
        "vision_tower.encoder.blocks.0.wqkv.weight",
        qkv,
        rank=1,
        world_size=2,
    )

    query, key, value = qkv.chunk(3, dim=0)
    expected = torch.cat([query[4:], key[4:], value[4:]], dim=0)
    torch.testing.assert_close(shard, expected)
