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

"""Unit tests for Kimi-K3 MoE configuration validation and edge cases."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

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
MoERoutingResult = moe_types.MoERoutingResult


class TestMoEParallelConfigValidation:
    """Test MoE parallel configuration validation."""

    def test_parallel_config_invalid_tp_size(self):
        """Test that invalid TP size raises ValueError."""
        with pytest.raises(ValueError, match="MoE TP rank and size are invalid"):
            MoEParallelConfig(
                tp_size=0,  # Invalid
                tp_rank=0,
                ep_size=2,
                ep_rank=0,
            )

    def test_parallel_config_invalid_tp_rank(self):
        """Test that TP rank out of bounds raises ValueError."""
        with pytest.raises(ValueError, match="MoE TP rank and size are invalid"):
            MoEParallelConfig(
                tp_size=4,
                tp_rank=4,  # Out of bounds
                ep_size=2,
                ep_rank=0,
            )

    def test_parallel_config_invalid_ep_size(self):
        """Test that invalid EP size raises ValueError."""
        with pytest.raises(ValueError, match="MoE EP rank and size are invalid"):
            MoEParallelConfig(
                tp_size=2,
                tp_rank=0,
                ep_size=-1,  # Invalid
                ep_rank=0,
            )

    def test_parallel_config_invalid_ep_rank(self):
        """Test that EP rank out of bounds raises ValueError."""
        with pytest.raises(ValueError, match="MoE EP rank and size are invalid"):
            MoEParallelConfig(
                tp_size=2,
                tp_rank=0,
                ep_size=4,
                ep_rank=4,  # Out of bounds
            )

    def test_parallel_config_invalid_mc2_capacity(self):
        """Test that negative MC2 capacity raises ValueError."""
        with pytest.raises(ValueError, match="MoE MC2 token capacity must be positive"):
            MoEParallelConfig(
                tp_size=1,
                tp_rank=0,
                ep_size=2,
                ep_rank=0,
                mc2_tokens_capacity=0,  # Invalid
            )

    def test_parallel_config_input_tp_defaults(self):
        """Test that input_tp defaults are set correctly."""
        config = MoEParallelConfig(
            tp_size=4,
            tp_rank=2,
            ep_size=2,
            ep_rank=1,
        )

        # input_tp_size should default to tp_size
        assert config.input_tp_size == 4
        # input_tp_rank should default to tp_rank
        assert config.input_tp_rank == 2

    def test_parallel_config_replicated_input(self):
        """Test partitions_replicated_input property."""
        # When input_tp_size == tp_size, not replicated
        config1 = MoEParallelConfig(
            tp_size=4,
            tp_rank=0,
            ep_size=4,
            ep_rank=0,
        )
        assert config1.partitions_replicated_input is False

        # When input_tp_size != tp_size, replicated
        config2 = MoEParallelConfig(
            tp_size=1,
            tp_rank=0,
            ep_size=4,
            ep_rank=0,
            input_tp_size=4,
        )
        assert config2.partitions_replicated_input is True


class TestMoEExpertsConfigValidation:
    """Test MoE experts configuration validation."""

    def test_experts_config_invalid_ep_size(self):
        """Test that invalid EP size raises ValueError."""
        with pytest.raises(ValueError, match="MoE expert EP rank and size are invalid"):
            MoEExpertsConfig(
                num_experts=8,
                hidden_size=4096,
                intermediate_size=14336,
                tp_size=2,
                tp_rank=0,
                ep_size=0,  # Invalid
                ep_rank=0,
            )

    def test_experts_config_invalid_ep_rank(self):
        """Test that EP rank out of bounds raises ValueError."""
        with pytest.raises(ValueError, match="MoE expert EP rank and size are invalid"):
            MoEExpertsConfig(
                num_experts=8,
                hidden_size=4096,
                intermediate_size=14336,
                tp_size=2,
                tp_rank=0,
                ep_size=2,
                ep_rank=2,  # Out of bounds
            )

    def test_experts_config_uneven_expert_split(self):
        """Test that experts must divide evenly across EP ranks."""
        with pytest.raises(ValueError, match="MoE experts must divide evenly across EP ranks"):
            MoEExpertsConfig(
                num_experts=7,  # Cannot divide evenly by ep_size=2
                hidden_size=4096,
                intermediate_size=14336,
                tp_size=2,
                tp_rank=0,
                ep_size=2,
                ep_rank=0,
            )

    def test_experts_config_negative_experts(self):
        """Test that negative number of experts raises ValueError."""
        with pytest.raises(ValueError, match="MoE experts must divide evenly"):
            MoEExpertsConfig(
                num_experts=-1,  # Invalid
                hidden_size=4096,
                intermediate_size=14336,
                tp_size=1,
                tp_rank=0,
            )

    def test_experts_config_local_expert_calculation(self):
        """Test num_local_experts calculation."""
        config = MoEExpertsConfig(
            num_experts=64,
            hidden_size=4096,
            intermediate_size=14336,
            tp_size=2,
            tp_rank=0,
            ep_size=8,
            ep_rank=3,
        )

        assert config.num_local_experts == 8  # 64 / 8
        assert config.first_expert_id == 24  # 3 * 8

    def test_experts_config_first_expert_id_all_ranks(self):
        """Test first_expert_id for all EP ranks."""
        num_experts = 16
        ep_size = 4

        for ep_rank in range(ep_size):
            config = MoEExpertsConfig(
                num_experts=num_experts,
                hidden_size=4096,
                intermediate_size=14336,
                tp_size=1,
                tp_rank=0,
                ep_size=ep_size,
                ep_rank=ep_rank,
            )

            expected_first_id = ep_rank * (num_experts // ep_size)
            assert config.first_expert_id == expected_first_id


class TestMoERoutingResult:
    """Test MoE routing result dataclass."""

    def test_routing_result_creation(self):
        """Test creating MoE routing result."""
        batch_size = 4
        top_k = 2
        topk_ids = torch.tensor([[0, 3], [1, 5], [2, 7], [0, 4]], dtype=torch.int64)
        topk_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.55, 0.45], [0.8, 0.2]], dtype=torch.float32)

        result = MoERoutingResult(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        assert result.topk_ids.shape == (batch_size, top_k)
        assert result.topk_weights.shape == (batch_size, top_k)

    def test_routing_result_weights_normalized(self):
        """Test that routing weights can be normalized."""
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        topk_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3]], dtype=torch.float32)

        result = MoERoutingResult(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        # Check weights sum to 1 per token
        weight_sums = result.topk_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones(2), atol=1e-6)


class TestMoECommTypeNormalization:
    """Test MoE communication type normalization edge cases."""

    def test_comm_type_case_insensitive(self):
        """Test that comm type parsing is case insensitive."""
        variations = ["ALL_GATHER", "all_gather", "All_Gather", "ALL_GATHER"]
        for var in variations:
            result = MoECommType.from_value(var)
            assert result == MoECommType.ALL_GATHER

    def test_comm_type_with_hyphens(self):
        """Test comm type with hyphens vs underscores."""
        assert MoECommType.from_value("all-gather") == MoECommType.ALL_GATHER
        assert MoECommType.from_value("all_gather") == MoECommType.ALL_GATHER
        assert MoECommType.from_value("all-to-all") == MoECommType.ALL_TO_ALL

    def test_comm_type_idempotent(self):
        """Test that from_value is idempotent."""
        comm_type = MoECommType.ALL_TO_ALL
        result = MoECommType.from_value(comm_type)
        assert result == comm_type
        assert result is comm_type


class TestMoEConfigRealisticScenarios:
    """Test realistic MoE configuration scenarios."""

    def test_kimi_k3_typical_config(self):
        """Test typical Kimi-K3 MoE configuration."""
        # Typical Kimi-K3: 64 experts, top-2, TP=4, EP=2
        parallel_config = MoEParallelConfig(
            tp_size=4,
            tp_rank=0,
            ep_size=2,
            ep_rank=0,
        )

        experts_config = MoEExpertsConfig(
            num_experts=64,
            hidden_size=6144,
            intermediate_size=16384,
            tp_size=4,
            tp_rank=0,
            ep_size=2,
            ep_rank=0,
        )

        router_config = MoERouterConfig(
            num_experts=64,
            top_k=6,
            scoring_func="softmax",
            renormalize=True,
            routed_scaling_factor=1.0,
        )

        # Verify configurations are consistent
        assert experts_config.num_experts == router_config.num_experts
        assert experts_config.ep_size == parallel_config.ep_size
        assert experts_config.num_local_experts == 32  # 64 / 2

    def test_single_node_all_experts(self):
        """Test single-node configuration with all experts."""
        config = MoEExpertsConfig(
            num_experts=8,
            hidden_size=4096,
            intermediate_size=11008,
            tp_size=8,
            tp_rank=0,
            ep_size=1,  # All experts on one node
            ep_rank=0,
        )

        assert config.num_local_experts == 8
        assert config.first_expert_id == 0

    def test_large_scale_expert_parallelism(self):
        """Test large-scale EP configuration."""
        num_experts = 128
        ep_size = 16

        config = MoEExpertsConfig(
            num_experts=num_experts,
            hidden_size=8192,
            intermediate_size=28672,
            tp_size=1,
            tp_rank=0,
            ep_size=ep_size,
            ep_rank=8,
        )

        assert config.num_local_experts == 8  # 128 / 16
        assert config.first_expert_id == 64  # 8 * 8
