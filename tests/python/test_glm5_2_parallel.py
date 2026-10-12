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

"""Parallel-layout tests for the GLM-5.2 Python NPU model."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models import deepseek_v32, glm5_2
from xllm.python.models.glm5_2 import Glm52Config, Glm52ForCausalLM, Glm52MoE
from xllm.python.models.weight_utils import W8A8WeightLoader


def _config(**overrides) -> dict:
    values = {
        "model_type": "glm_moe_dsa",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 32,
        "vocab_size": 32,
        "max_position_embeddings": 16,
        "q_lora_rank": 8,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 4,
        "first_k_dense_replace": 0,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "tp_size": 2,
        "tp_rank": 0,
        "dp_size": 2,
        "dp_rank": 0,
        "cp_size": 1,
        "cp_rank": 0,
        "world_size": 4,
        "moe_tp_size": 1,
        "moe_tp_rank": 0,
        "ep_size": 4,
        "ep_rank": 0,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


def test_full_world_ep_partitions_glm_experts() -> None:
    cfg = Glm52Config.from_dict(_config(ep_rank=3))
    cfg.validate()

    model = Glm52ForCausalLM(_config(ep_rank=3))
    moe = model.model.layers[0].mlp

    assert moe.local_expert_start == 6
    assert moe.local_expert_end == 8
    assert moe.num_local_experts == 2

    moe.allocate_experts_w13_for_loading()
    moe.allocate_experts_w2_for_loading()
    assert moe.experts_w13.shape == (2, 16, 16)
    assert moe.experts_w2.shape == (2, 16, 8)


def test_glm_parallel_world_size_defaults_to_tp_dp_product() -> None:
    values = _config()
    values.pop("world_size")

    cfg = Glm52Config.from_dict(values)

    assert cfg.world_size == cfg.tp_size * cfg.dp_size * cfg.cp_size == 4


def test_glm_parallel_world_size_includes_context_parallel() -> None:
    cfg = Glm52Config.from_dict(_config(cp_size=2, cp_rank=1, world_size=8, ep_size=8))

    cfg.validate()

    assert cfg.world_size == cfg.tp_size * cfg.dp_size * cfg.cp_size == 8
    assert cfg.cp_rank == 1


def test_glm_layerwise_split_rank_is_validated() -> None:
    cfg = Glm52Config.from_dict(_config(layerwise_split_size=2, layerwise_split_rank=1))
    cfg.validate()
    assert cfg.layerwise_split_rank == 1

    invalid = Glm52Config.from_dict(_config(layerwise_split_size=2, layerwise_split_rank=2))
    with pytest.raises(ValueError, match="layerwise_split_rank"):
        invalid.validate()


def test_glm_layerwise_split_cannot_overlap_context_parallel() -> None:
    cfg = Glm52Config.from_dict(
        _config(
            cp_size=2,
            cp_rank=0,
            world_size=8,
            ep_size=8,
            layerwise_split_size=2,
        )
    )
    with pytest.raises(ValueError, match="CP and layerwise"):
        cfg.validate()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ep_size": 2}, "ep_size must be 1 or world_size"),
        ({"world_size": 8}, r"world_size must equal tp_size \* dp_size \* cp_size"),
        ({"cp_rank": 2}, "cp_rank must be in"),
        ({"n_routed_experts": 10}, "n_routed_experts must be divisible by ep_size"),
        ({"moe_tp_size": 2}, r"moe_tp_size \* ep_size"),
        ({"ep_rank": 4}, "ep_rank must be in"),
    ],
)
def test_invalid_glm_parallel_topology_is_rejected(overrides: dict, message: str) -> None:
    cfg = Glm52Config.from_dict(_config(**overrides))

    with pytest.raises(ValueError, match=message):
        cfg.validate()


class _RecordingLoader(W8A8WeightLoader):
    latest: _RecordingLoader | None = None

    def __init__(self, model, state_dicts, tp_size: int, tp_rank: int) -> None:
        super().__init__(model, state_dicts, tp_size, tp_rank)
        self.loaded: list[str] = []
        self.shared_shards: list[tuple[str, int, int]] = []
        type(self).latest = self

    def load_tensor(self, name: str) -> torch.Tensor:
        self.loaded.append(name)
        if ".mlp.experts." not in name:
            return torch.zeros(32, 32)
        if name.endswith(("gate_proj.weight", "up_proj.weight")):
            return torch.zeros(8, 16, dtype=torch.int8)
        if name.endswith(("gate_proj.weight_scale", "up_proj.weight_scale")):
            return torch.zeros(8, 1)
        if name.endswith(("gate_proj.weight_offset", "up_proj.weight_offset")):
            return torch.zeros(8, 1)
        if name.endswith("down_proj.weight"):
            return torch.zeros(16, 8, dtype=torch.int8)
        if name.endswith(("down_proj.weight_scale", "down_proj.weight_offset")):
            return torch.zeros(16, 1)
        raise AssertionError(f"unexpected expert tensor: {name}")

    def copy_in(self, name: str, tensor: torch.Tensor) -> None:
        self.loaded.append(name)
        assert tensor.is_contiguous()

    def load_w8a8_projection(self, prefix: str, proj: str, _shard_dims: dict | None = None) -> None:
        self.loaded.append(prefix + proj)

    def load_w8a8_mlp(
        self,
        prefix: str,
        world: int | None = None,
        rank: int | None = None,
    ) -> None:
        self.loaded.append(prefix)
        if ".shared_experts." in prefix:
            self.shared_shards.append((prefix, world, rank))


def test_glm_weight_loader_reads_only_local_ep_experts(monkeypatch) -> None:
    model = Glm52ForCausalLM(_config(ep_rank=2))
    model.model.layers[0].self_attn.process_weights_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w13_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w2_after_loading = MagicMock()
    model.model.layers[0].mlp.shared_experts.process_weights_after_loading = MagicMock()
    monkeypatch.setattr(glm5_2, "W8A8WeightLoader", _RecordingLoader)

    model.load_weights([], tp_rank=0, tp_size=2)

    loader = _RecordingLoader.latest
    assert loader is not None
    expert_names = [name for name in loader.loaded if ".mlp.experts." in name]
    assert expert_names
    assert all(".experts.4." in name or ".experts.5." in name for name in expert_names)
    assert loader.tp_size == 2
    assert loader.tp_rank == 0
    assert loader.shared_shards == [("model.layers.0.mlp.shared_experts.", 1, 0)]


def _npu_device() -> object:
    return SimpleNamespace(type="npu")


def _forward_ctx(*, is_prefill: bool, is_chunked_prefill: bool = False, **metadata_fields):
    metadata = SimpleNamespace(
        is_prefill=is_prefill,
        is_chunked_prefill=is_chunked_prefill,
        **metadata_fields,
    )
    return ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[],
    )


def _expanded_decode_metadata(num_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        kv_seq_lens=torch.ones(num_tokens, dtype=torch.int32),
        block_table=torch.zeros(num_tokens, 1, dtype=torch.int32),
        paged_kv_indptr=torch.arange(num_tokens + 1, dtype=torch.int32),
        paged_kv_indices=torch.zeros(num_tokens, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(num_tokens, dtype=torch.int32),
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=[1] * num_tokens,
    )


def _make_mega_moe_layer() -> Glm52MoE:
    cfg = Glm52Config.from_dict(_config())
    moe = Glm52MoE(cfg, layer_id=0, dtype=torch.float32, device=torch.device("cpu"))
    moe._mega_moe_enabled = True
    moe._mega_moe_ccl_buffer_size = 8192
    moe._mega_moe_num_max_tokens_per_rank = 32
    moe.mega_moe_context = torch.zeros(4, dtype=torch.int32)
    moe.dp_size = 1
    return moe


def test_glm_mega_moe_config_fields_are_parsed() -> None:
    context = torch.zeros(4, dtype=torch.int32)
    cfg = Glm52Config.from_dict(
        _config(
            enable_mega_moe=True,
            expert_parallel_degree=2,
            mega_moe_context=context,
            mega_moe_ccl_buffer_size=8192,
            mega_moe_num_max_tokens_per_rank=32,
        )
    )
    assert cfg.enable_mega_moe
    assert cfg.expert_parallel_degree == 2
    assert cfg.mega_moe_ccl_buffer_size == 8192
    assert cfg.mega_moe_num_max_tokens_per_rank == 32
    assert torch.equal(cfg.mega_moe_context, context)


@pytest.mark.parametrize(
    ("model_type", "overrides", "device_type", "expected"),
    [
        ("glm_moe_dsa", {"enable_mega_moe": True, "ep_size": 4, "expert_parallel_degree": 2}, "npu", True),
        ("deepseek_v32", {"enable_mega_moe": True, "ep_size": 4, "expert_parallel_degree": 2}, "npu", False),
        ("glm_moe_dsa", {"enable_mega_moe": True, "ep_size": 4, "expert_parallel_degree": 2}, "cpu", False),
        ("glm_moe_dsa", {"enable_mega_moe": True, "ep_size": 1, "expert_parallel_degree": 2}, "npu", False),
        ("glm_moe_dsa", {"enable_mega_moe": False, "ep_size": 4, "expert_parallel_degree": 2}, "npu", False),
        ("glm_moe_dsa", {"enable_mega_moe": True, "ep_size": 4, "expert_parallel_degree": 1}, "npu", False),
        (
            "glm_moe_dsa",
            {"enable_mega_moe": True, "ep_size": 4, "expert_parallel_degree": 2, "enable_eplb": True},
            "npu",
            False,
        ),
        (
            "deepseek_v32_mtp",
            {"enable_mega_moe": True, "ep_size": 4, "expert_parallel_degree": 2},
            "npu",
            False,
        ),
        (
            "glm_moe_dsa_mtp",
            {"enable_mega_moe": True, "ep_size": 4, "expert_parallel_degree": 2},
            "npu",
            False,
        ),
        (
            "qwen3_moe",
            {"enable_mega_moe": True, "ep_size": 4, "expert_parallel_degree": 2},
            "npu",
            False,
        ),
    ],
)
def test_shared_moe_mega_moe_enable_conditions(
    model_type: str, overrides: dict, device_type: str, expected: bool
) -> None:
    values = _config(model_type=model_type, **overrides)
    cfg = Glm52Config.from_dict(values)
    device = _npu_device() if device_type == "npu" else torch.device("cpu")
    assert Glm52MoE._can_enable_mega_moe(cfg, device) is expected


def test_shared_moe_mega_moe_ignores_speculative_token_count() -> None:
    cfg = Glm52Config.from_dict(_config(enable_mega_moe=True, ep_size=4, expert_parallel_degree=2))
    cfg.num_speculative_tokens = 4
    assert Glm52MoE._can_enable_mega_moe(cfg, _npu_device()) is True


def test_prefill_keeps_grouped_moe_and_skips_mega_moe(monkeypatch) -> None:
    moe = _make_mega_moe_layer()
    grouped = MagicMock(return_value=torch.zeros(2, moe.hidden))
    mega = MagicMock()
    monkeypatch.setattr(deepseek_v32.kernels, "grouped_moe", grouped, raising=False)
    monkeypatch.setattr(deepseek_v32.kernels, "mega_moe", mega, raising=False)
    monkeypatch.setattr(deepseek_v32.distributed, "all_reduce_", MagicMock(), raising=False)
    hidden = torch.zeros(2, moe.hidden)
    with forward_context(
        _forward_ctx(
            is_prefill=True,
            expanded_decode_metadata=SimpleNamespace(enabled=False),
        )
    ):
        use_mega_moe = moe._should_use_mega_moe()
        routed = moe._run_routed_experts(hidden, use_mega_moe)
        moe._combine_expert_outputs(routed, torch.zeros_like(routed), use_mega_moe)
    grouped.assert_called_once()
    mega.assert_not_called()


@pytest.mark.parametrize(
    ("is_prefill", "is_chunked_prefill"),
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_expanded_speculative_decode_calls_mega_moe(monkeypatch, is_prefill: bool, is_chunked_prefill: bool) -> None:
    moe = _make_mega_moe_layer()
    grouped = MagicMock()
    mega = MagicMock(return_value=torch.zeros(2, moe.hidden))
    gate = MagicMock(return_value=(torch.ones(2, moe.topk), torch.zeros(2, moe.topk, dtype=torch.int32)))
    monkeypatch.setattr(deepseek_v32.kernels, "grouped_moe", grouped, raising=False)
    monkeypatch.setattr(deepseek_v32.kernels, "mega_moe", mega, raising=False)
    monkeypatch.setattr(deepseek_v32.kernels, "moe_gate_routing", gate, raising=False)
    monkeypatch.setattr(deepseek_v32.distributed, "all_reduce_", MagicMock(), raising=False)
    hidden = torch.zeros(2, moe.hidden)
    with forward_context(
        _forward_ctx(
            is_prefill=is_prefill,
            is_chunked_prefill=is_chunked_prefill,
            slot_mapping=torch.arange(2, dtype=torch.int32),
            expanded_decode_metadata=_expanded_decode_metadata(2),
        )
    ):
        moe._run_routed_experts(hidden, moe._should_use_mega_moe())
    grouped.assert_not_called()
    mega.assert_called_once()


def test_decode_calls_mega_moe_with_shared_weights(monkeypatch) -> None:
    moe = _make_mega_moe_layer()
    grouped = MagicMock()
    mega = MagicMock(return_value=torch.zeros(2, moe.hidden))
    gate = MagicMock(return_value=(torch.ones(2, moe.topk), torch.zeros(2, moe.topk, dtype=torch.int32)))
    ep_reduce = MagicMock()
    monkeypatch.setattr(deepseek_v32.kernels, "grouped_moe", grouped, raising=False)
    monkeypatch.setattr(deepseek_v32.kernels, "mega_moe", mega, raising=False)
    monkeypatch.setattr(deepseek_v32.kernels, "moe_gate_routing", gate, raising=False)
    monkeypatch.setattr(deepseek_v32.distributed, "all_reduce_", ep_reduce, raising=False)
    hidden = torch.zeros(2, moe.hidden)
    with forward_context(_forward_ctx(is_prefill=False, is_chunked_prefill=False)):
        use_mega_moe = moe._should_use_mega_moe()
        routed = moe._run_routed_experts(hidden, use_mega_moe)
        moe._combine_expert_outputs(routed, torch.zeros_like(routed), use_mega_moe)
    grouped.assert_not_called()
    mega.assert_called_once()
    args = mega.call_args.args
    assert args[4] is moe.experts_w13
    assert args[5] is moe.experts_w2
    assert args[6] is moe.mega_moe_w13_scale
    assert args[7] is moe.mega_moe_w2_scale
    assert args[6].dtype == torch.int64
    assert args[7].dtype == torch.int64
    ep_calls = [call.args for call in ep_reduce.call_args_list]
    assert all(len(call) < 2 or call[1] != "moe_ep" for call in ep_calls)


def test_load_experts_encodes_scales_without_cloning_nz_weights(monkeypatch) -> None:
    moe = _make_mega_moe_layer()
    loader = _RecordingLoader(moe, [], tp_size=1, tp_rank=0)
    encode = MagicMock(side_effect=lambda scale, offset: torch.zeros(scale.shape, dtype=torch.int64))
    monkeypatch.setattr(deepseek_v32.kernels, "encode_mega_moe_scale", encode, raising=False)
    monkeypatch.setattr(deepseek_v32.kernels, "format_cast_nz", lambda tensor: tensor, raising=False)
    moe.load_experts(loader, "model.layers.0.mlp.experts.", world=1, rank=0)
    assert not hasattr(moe, "mega_moe_w13")
    assert not hasattr(moe, "mega_moe_w2")
    assert moe.mega_moe_w13_scale.dtype == torch.int64
    assert moe.mega_moe_w2_scale.dtype == torch.int64
    assert moe.experts_w13_scale.dtype == torch.float32
    assert moe.experts_w2_scale_compute.dtype == torch.bfloat16
    assert encode.call_count == 2
    for scale, offset in (call.args for call in encode.call_args_list):
        assert scale.dtype == torch.float32
        assert torch.count_nonzero(offset) == 0
    offset_names = [name for name in loader.loaded if name.endswith("weight_offset")]
    assert len(offset_names) == 3 * moe.num_local_experts
    weight_names = [name for name, _ in moe.named_parameters() if "experts_w" in name]
    assert weight_names == ["experts_w13", "experts_w2"]
    named = dict(moe.named_buffers())
    assert "mega_moe_w13" not in named
    assert "mega_moe_w2" not in named


def test_mega_moe_token_mask_reuses_acl_graph_buffer() -> None:
    moe = _make_mega_moe_layer()
    hidden = torch.zeros(8, moe.hidden)
    graph_mask = torch.tensor([1, 1, 1, 0, 1, 0, 0, 0], dtype=torch.int8)
    with forward_context(
        _forward_ctx(
            is_prefill=False,
            dp_execution_token_counts=(4, 4),
            mega_moe_token_mask=graph_mask,
        )
    ):
        mask = moe._mega_moe_token_mask(hidden)
    assert mask is graph_mask


def test_mega_moe_compact_dp_gather_does_not_need_mask() -> None:
    moe = _make_mega_moe_layer()
    moe.dp_size = 2
    hidden = torch.zeros(4, moe.hidden)
    with forward_context(_forward_ctx(is_prefill=False, dp_execution_token_counts=(3, 1))):
        mask = moe._mega_moe_token_mask(hidden)
    assert mask is None


def test_mega_moe_raises_when_tokens_exceed_comm_cap(monkeypatch) -> None:
    moe = _make_mega_moe_layer()
    moe.dp_size = 1
    moe._mega_moe_num_max_tokens_per_rank = 2
    grouped = MagicMock(return_value=torch.zeros(4, moe.hidden))
    mega = MagicMock()
    monkeypatch.setattr(deepseek_v32.kernels, "grouped_moe", grouped, raising=False)
    monkeypatch.setattr(deepseek_v32.kernels, "mega_moe", mega, raising=False)
    hidden = torch.zeros(4, moe.hidden)
    with (
        forward_context(_forward_ctx(is_prefill=False)),
        pytest.raises(RuntimeError, match="exceeds the communication cap"),
    ):
        moe._run_routed_experts(hidden, moe._should_use_mega_moe())
    grouped.assert_not_called()
    mega.assert_not_called()


def test_mixed_dp_decode_keeps_grouped_moe_and_ep_reduce(monkeypatch) -> None:
    moe = _make_mega_moe_layer()
    moe.dp_size = 2
    grouped = MagicMock(return_value=torch.zeros(2, moe.hidden))
    mega = MagicMock()
    ep_reduce = MagicMock()
    monkeypatch.setattr(deepseek_v32.kernels, "grouped_moe", grouped, raising=False)
    monkeypatch.setattr(deepseek_v32.kernels, "mega_moe", mega, raising=False)
    monkeypatch.setattr(deepseek_v32.distributed, "all_reduce_", ep_reduce, raising=False)
    hidden = torch.zeros(2, moe.hidden)
    with forward_context(
        _forward_ctx(
            is_prefill=False,
            dp_is_decode=(1, 0),
            dp_execution_token_counts=(2, 1),
        )
    ):
        use_mega_moe = moe._should_use_mega_moe()
        routed = moe._run_routed_experts(hidden, use_mega_moe)
        moe._combine_expert_outputs(routed, torch.zeros_like(routed), use_mega_moe)
    grouped.assert_called_once()
    mega.assert_not_called()
    assert any(len(call.args) >= 2 and call.args[1] == "moe_ep" for call in ep_reduce.call_args_list)


def test_all_decode_dp_uses_mega_moe_without_ep_reduce(monkeypatch) -> None:
    moe = _make_mega_moe_layer()
    moe.dp_size = 2
    grouped = MagicMock()
    mega = MagicMock(return_value=torch.zeros(2, moe.hidden))
    gate = MagicMock(return_value=(torch.ones(2, moe.topk), torch.zeros(2, moe.topk, dtype=torch.int32)))
    ep_reduce = MagicMock()
    monkeypatch.setattr(deepseek_v32.kernels, "grouped_moe", grouped, raising=False)
    monkeypatch.setattr(deepseek_v32.kernels, "mega_moe", mega, raising=False)
    monkeypatch.setattr(deepseek_v32.kernels, "moe_gate_routing", gate, raising=False)
    monkeypatch.setattr(deepseek_v32.distributed, "all_reduce_", ep_reduce, raising=False)
    hidden = torch.zeros(2, moe.hidden)
    with forward_context(
        _forward_ctx(
            is_prefill=False,
            dp_is_decode=(1, 1),
            dp_execution_token_counts=(1, 1),
        )
    ):
        use_mega_moe = moe._should_use_mega_moe()
        routed = moe._run_routed_experts(hidden, use_mega_moe)
        moe._combine_expert_outputs(routed, torch.zeros_like(routed), use_mega_moe)
    grouped.assert_not_called()
    mega.assert_called_once()
    assert all(len(call.args) < 2 or call.args[1] != "moe_ep" for call in ep_reduce.call_args_list)
