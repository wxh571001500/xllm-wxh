# Copyright 2026 The xLLM Authors.
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

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from xllm.python.kernels_npu.vision import encoder_attention
from xllm.python.models import kimi_k3_vit as _vit_module
from xllm.python.models.kimi_k3_vit import (
    KimiK3VisionConfig,
    KimiK3VisionModel,
    _get_load_balance_assignment,
    _run_data_parallel_vision_tower,
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


def test_config_replicates_vision_weights_across_runtime_tp_ranks() -> None:
    raw_config = _tiny_config()
    raw_config["tp_size"] = 4
    raw_config["tp_rank"] = 2

    config = KimiK3VisionConfig.from_dict(raw_config)

    assert config.tp_size == 1
    assert config.tp_rank == 0
    assert config.encoder_dp_size == 4
    assert config.encoder_dp_rank == 2
    config.validate()


def test_tpool_merger_pools_time_before_spatial_reorder() -> None:
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(16, 1)

    outputs = tpool_patch_merger(hidden_states, [[2, 2, 4]], (2, 2))

    assert len(outputs) == 1
    assert outputs[0].shape == (2, 4, 1)
    expected = torch.tensor([[4.0, 5.0, 8.0, 9.0], [6.0, 7.0, 10.0, 11.0]])
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
    assert outputs[0].shape == (2, 8)
    assert torch.isfinite(outputs[0]).all()


def test_encoder_data_mode_uses_vllm_load_balanced_media_order(monkeypatch) -> None:
    captured_boundaries: list[list[int]] = []

    def capture_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: Sequence[int],
    ) -> torch.Tensor:
        captured_boundaries.append(list(cu_seqlens))
        return encoder_attention(query, key, value, cu_seqlens)

    monkeypatch.setattr(_vit_module, "encoder_attention", capture_attention)
    config = _tiny_config()
    config["vision_config"]["merge_kernel_size"] = [1, 1]
    model = KimiK3VisionModel(config)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
    pixel_values = torch.randn(3, 3, 2, 2)
    grid_thws = torch.tensor([[1, 1, 1], [1, 1, 2]], dtype=torch.int64)

    outputs = model(pixel_values, grid_thws)

    assert captured_boundaries == [[0, 2, 3]]
    assert [output.shape[0] for output in outputs] == [1, 2]


def test_encoder_data_mode_matches_vllm_greedy_assignment() -> None:
    assignment, image_counts, rank_loads = _get_load_balance_assignment(
        [1000, 100, 200, 50],
        num_ranks=2,
    )

    assert assignment == [0, 2, 1, 3]
    assert image_counts == [1, 3]
    assert rank_loads == [1000, 350]


def test_encoder_data_parallel_reuses_mainline_tp_group(monkeypatch) -> None:
    class _VisionTower:
        merge_kernel_size = (1, 1)
        config = MagicMock(hidden_size=2)

        def __call__(self, pixel_values, _grids):
            return [pixel_values]

    gathered = torch.tensor([[10.0, 11.0], [20.0, 21.0]], dtype=torch.float32).unsqueeze(1)

    def all_gather(tensor, *, dim, world_size):
        assert tensor.shape == (1, 1, 2)
        assert dim == 0
        assert world_size == 2
        return gathered

    monkeypatch.setattr(_vit_module.distributed, "tp_all_gather", all_gather)
    outputs = _run_data_parallel_vision_tower(
        _VisionTower(),
        torch.tensor([[10.0, 11.0], [20.0, 21.0]]).unsqueeze(1),
        [[1, 1, 1], [1, 1, 1]],
        rank=0,
        world_size=2,
    )

    assert [output.squeeze(0).tolist() for output in outputs] == [
        [[10.0, 11.0]],
        [[20.0, 21.0]],
    ]


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


def test_packed_attention_keeps_media_items_independent() -> None:
    torch.manual_seed(11)
    query, key, value = [torch.randn(5, 2, 4) for _ in range(3)]
    packed = encoder_attention(query, key, value, [0, 2, 5])
    expected = []
    for start, end in [(0, 2), (2, 5)]:
        attended = F.scaled_dot_product_attention(
            query[start:end].transpose(0, 1),
            key[start:end].transpose(0, 1),
            value[start:end].transpose(0, 1),
            is_causal=False,
        )
        expected.append(attended.transpose(0, 1))
    torch.testing.assert_close(packed, torch.cat(expected))


@pytest.mark.parametrize("boundaries", [[], [1, 5], [0, 4], [0, 3, 2, 5], [0, 2, 2, 5]])
def test_packed_attention_rejects_invalid_boundaries(boundaries: list[int]) -> None:
    tensor = torch.randn(5, 2, 4)
    with pytest.raises(ValueError, match="cu_seqlens"):
        encoder_attention(tensor, tensor, tensor, boundaries)


def test_packed_attention_supports_empty_encoder_rank() -> None:
    tensor = torch.empty(0, 2, 4)
    output = encoder_attention(tensor, tensor, tensor, [0])
    assert output.shape == tensor.shape
    assert output.dtype == tensor.dtype


def test_multiple_media_match_separate_vision_forward() -> None:
    torch.manual_seed(17)
    config = _tiny_config()
    config["vision_config"]["merge_kernel_size"] = [1, 1]
    model = KimiK3VisionModel(config)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
    pixels = torch.randn(3, 3, 2, 2)
    packed = model(pixels, [[1, 1, 1], [1, 1, 2]])
    separate = [model(pixels[:1], [[1, 1, 1]])[0], model(pixels[1:], [[1, 1, 2]])[0]]
    for actual, expected in zip(packed, separate):
        torch.testing.assert_close(actual, expected)


def test_vision_weight_loading_accepts_checkpoint_projector_aliases() -> None:
    model = KimiK3VisionModel(_tiny_config())
    checkpoint = {}
    expected = {}
    for name, parameter in model.named_parameters():
        expected[name] = torch.full_like(parameter, 0.125)
        checkpoint_name = name.replace("mm_projector.linear_1.", "mm_projector.proj.0.").replace(
            "mm_projector.linear_2.", "mm_projector.proj.2."
        )
        checkpoint[f"model.{checkpoint_name}"] = expected[name]
    state_dict = MagicMock()
    state_dict.has.side_effect = checkpoint.__contains__
    state_dict.get_tensor.side_effect = checkpoint.__getitem__
    loaded = model.load_weights([state_dict], tp_rank=0, tp_size=1)
    assert loaded == set(expected)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected[name])
