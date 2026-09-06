# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Kimi-K3 MoE configuration types."""

import importlib.util
import sys
from pathlib import Path

# Direct import to avoid triggering xllm.python.layers runtime initialization
moe_types_file = Path(__file__).parent.parent.parent.parent / "xllm" / "python" / "layers" / "moe" / "types.py"
spec = importlib.util.spec_from_file_location("moe_types_module", moe_types_file)
moe_types = importlib.util.module_from_spec(spec)
sys.modules["moe_types_module"] = moe_types
spec.loader.exec_module(moe_types)

MoECommType = moe_types.MoECommType
MoEExpertsConfig = moe_types.MoEExpertsConfig
MoEParallelConfig = moe_types.MoEParallelConfig
MoERouterConfig = moe_types.MoERouterConfig


class TestMoECommType:
    """Test MoE communication type enum."""

    def test_comm_type_values(self):
        """Verify MoECommType enum values."""
        assert MoECommType.ALL_GATHER == "all_gather"
        assert MoECommType.ALL_TO_ALL == "all_to_all"
        assert MoECommType.MC2 == "mc2"
        assert MoECommType.AUTO == "auto"

    def test_comm_type_from_value(self):
        """Test MoECommType.from_value() normalization."""
        assert MoECommType.from_value("all_gather") == MoECommType.ALL_GATHER
        assert MoECommType.from_value("ALL_GATHER") == MoECommType.ALL_GATHER
        assert MoECommType.from_value("allgather") == MoECommType.ALL_GATHER
        assert MoECommType.from_value("all-gather") == MoECommType.ALL_GATHER
        assert MoECommType.from_value(MoECommType.ALL_GATHER) == MoECommType.ALL_GATHER

    def test_comm_type_aliases(self):
        """Test MoECommType aliases."""
        assert MoECommType.from_value("alltoall") == MoECommType.ALL_TO_ALL
        assert MoECommType.from_value("all2all") == MoECommType.ALL_TO_ALL


class TestMoEParallelConfig:
    """Test MoE parallel configuration."""

    def test_parallel_config_creation(self):
        """Test creating MoEParallelConfig with required fields."""
        config = MoEParallelConfig(
            tp_size=4,
            tp_rank=0,
            ep_size=2,
            ep_rank=0,
        )
        assert config.tp_size == 4
        assert config.tp_rank == 0
        assert config.ep_size == 2
        assert config.ep_rank == 0
        assert config.dp_size == 1
        assert config.dp_rank == 0
        assert config.comm_type == MoECommType.ALL_GATHER

    def test_parallel_config_with_comm_type(self):
        """Test MoEParallelConfig with explicit comm_type."""
        config = MoEParallelConfig(
            tp_size=2,
            tp_rank=1,
            ep_size=4,
            ep_rank=2,
            comm_type=MoECommType.ALL_TO_ALL,
        )
        assert config.comm_type == MoECommType.ALL_TO_ALL

    def test_parallel_config_mc2_settings(self):
        """Test MoEParallelConfig with MC2 communication."""
        config = MoEParallelConfig(
            tp_size=1,
            tp_rank=0,
            ep_size=8,
            ep_rank=3,
            comm_type=MoECommType.MC2,
            mc2_tokens_capacity=1024,
        )
        assert config.comm_type == MoECommType.MC2
        assert config.mc2_tokens_capacity == 1024

    def test_parallel_config_input_tp(self):
        """Test MoEParallelConfig with distinct input TP."""
        config = MoEParallelConfig(
            tp_size=1,
            tp_rank=0,
            ep_size=4,
            ep_rank=1,
            input_tp_size=4,
            input_tp_rank=1,
        )
        assert config.input_tp_size == 4
        assert config.input_tp_rank == 1
        assert config.partitions_replicated_input is True


class TestMoEExpertsConfig:
    """Test MoE experts configuration."""

    def test_experts_config_creation(self):
        """Test creating MoEExpertsConfig."""
        config = MoEExpertsConfig(
            num_experts=8,
            hidden_size=4096,
            intermediate_size=14336,
            tp_size=2,
            tp_rank=0,
            ep_size=2,
            ep_rank=0,
        )
        assert config.num_experts == 8
        assert config.hidden_size == 4096
        assert config.intermediate_size == 14336
        assert config.tp_size == 2
        assert config.ep_size == 2
        assert config.num_local_experts == 4

    def test_experts_config_single_ep(self):
        """Test MoEExpertsConfig with single EP rank."""
        config = MoEExpertsConfig(
            num_experts=16,
            hidden_size=2048,
            intermediate_size=8192,
            tp_size=4,
            tp_rank=2,
        )
        assert config.ep_size == 1
        assert config.num_local_experts == 16
        assert config.first_expert_id == 0

    def test_experts_config_multi_ep(self):
        """Test MoEExpertsConfig with multiple EP ranks."""
        config = MoEExpertsConfig(
            num_experts=64,
            hidden_size=8192,
            intermediate_size=28672,
            tp_size=2,
            tp_rank=0,
            ep_size=8,
            ep_rank=3,
        )
        assert config.num_local_experts == 8
        assert config.first_expert_id == 24


class TestMoERouterConfig:
    """Test MoE router configuration."""

    def test_router_config_creation(self):
        """Test creating MoERouterConfig."""
        config = MoERouterConfig(
            num_experts=8,
            top_k=2,
            scoring_func="softmax",
            renormalize=True,
            routed_scaling_factor=1.0,
        )
        assert config.num_experts == 8
        assert config.top_k == 2
        assert config.scoring_func == "softmax"
        assert config.renormalize is True
        assert config.routed_scaling_factor == 1.0

    def test_router_config_grouped_topk(self):
        """Test MoERouterConfig with grouped top-k."""
        config = MoERouterConfig(
            num_experts=16,
            top_k=4,
            scoring_func="softmax",
            renormalize=False,
            routed_scaling_factor=0.5,
            use_grouped_topk=True,
            num_expert_group=4,
            topk_group=2,
        )
        assert config.use_grouped_topk is True
        assert config.num_expert_group == 4
        assert config.topk_group == 2

    def test_router_config_validation(self):
        """Test MoERouterConfig with valid parameters."""
        config = MoERouterConfig(
            num_experts=32,
            top_k=2,
            scoring_func="softmax",
            renormalize=True,
            routed_scaling_factor=1.0,
        )
        assert config.top_k <= config.num_experts
        assert config.routed_scaling_factor > 0
