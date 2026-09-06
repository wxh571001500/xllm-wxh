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

"""Unit tests for MoE module."""

import pytest
import torch
import torch.nn as nn

from xllm.python.layers.moe import (
    GroupedTopKRouter,
    MoE,
    MoECommType,
    MoEExpertsConfig,
    MoEParallelConfig,
    MoERouterConfig,
    SituAndMul,
    UnquantizedRoutedExperts,
)


class TestMoEBasicFunctionality:
    """Test basic MoE functionality without distributed environment."""

    def test_situ_and_mul_forward(self):
        """Test SituAndMul activation function."""
        batch_size = 4
        hidden_size = 256

        activation = SituAndMul()
        x = torch.randn(batch_size, hidden_size * 2)

        output = activation(x)

        assert output.shape == (batch_size, hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_grouped_topk_router(self):
        """Test GroupedTopKRouter routing logic."""
        num_experts = 8
        hidden_size = 256
        top_k = 2
        batch_size = 16

        config = MoERouterConfig(
            num_experts=num_experts,
            top_k=top_k,
            normalize_topk_prob=True,
        )

        router = GroupedTopKRouter(
            hidden_size=hidden_size,
            config=config,
        )

        hidden_states = torch.randn(batch_size, hidden_size)
        routing_result = router(hidden_states)

        # Check routing result structure
        assert routing_result.selected_experts.shape == (batch_size, top_k)
        assert routing_result.routing_weights.shape == (batch_size, top_k)

        # Check expert indices are in valid range
        assert routing_result.selected_experts.min() >= 0
        assert routing_result.selected_experts.max() < num_experts

        # Check routing weights are normalized
        if config.normalize_topk_prob:
            weights_sum = routing_result.routing_weights.sum(dim=1)
            assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)

    def test_unquantized_routed_experts(self):
        """Test UnquantizedRoutedExperts forward pass."""
        num_experts = 4
        hidden_size = 128
        intermediate_size = 512
        num_tokens = 8

        experts_config = MoEExpertsConfig(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        experts = UnquantizedRoutedExperts(experts_config)

        # Create input
        hidden_states = torch.randn(num_tokens, hidden_size)
        expert_indices = torch.randint(0, num_experts, (num_tokens,))

        # Forward pass
        output = experts(hidden_states, expert_indices)

        assert output.shape == (num_tokens, hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestMoEConfigurations:
    """Test different MoE configurations."""

    def test_moe_config_creation(self):
        """Test MoE configuration objects creation."""
        router_config = MoERouterConfig(
            num_experts=8,
            top_k=2,
            normalize_topk_prob=True,
        )

        experts_config = MoEExpertsConfig(
            num_experts=8,
            hidden_size=256,
            intermediate_size=1024,
        )

        parallel_config = MoEParallelConfig(
            tp_size=1,
            ep_size=1,
            comm_type=MoECommType.TENSOR_PARALLEL,
        )

        assert router_config.num_experts == 8
        assert router_config.top_k == 2
        assert experts_config.hidden_size == 256
        assert parallel_config.tp_size == 1

    def test_moe_comm_types(self):
        """Test MoE communication type enum."""
        assert MoECommType.TENSOR_PARALLEL.value == "tensor_parallel"
        assert MoECommType.ALL_TO_ALL.value == "all_to_all"
        assert MoECommType.ALL_GATHER.value == "all_gather"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
class TestMoENPU:
    """Test MoE module on NPU device."""

    def test_router_npu(self):
        """Test router on NPU."""
        device = torch.device("npu")
        num_experts = 8
        hidden_size = 256
        top_k = 2
        batch_size = 16

        config = MoERouterConfig(
            num_experts=num_experts,
            top_k=top_k,
        )

        router = GroupedTopKRouter(
            hidden_size=hidden_size,
            config=config,
        ).to(device)

        hidden_states = torch.randn(batch_size, hidden_size, device=device)
        routing_result = router(hidden_states)

        assert routing_result.selected_experts.device.type == "npu"
        assert routing_result.routing_weights.device.type == "npu"

    def test_experts_npu(self):
        """Test experts on NPU."""
        device = torch.device("npu")
        num_experts = 4
        hidden_size = 128
        intermediate_size = 512
        num_tokens = 8

        experts_config = MoEExpertsConfig(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        experts = UnquantizedRoutedExperts(experts_config).to(device)

        hidden_states = torch.randn(num_tokens, hidden_size, device=device)
        expert_indices = torch.randint(0, num_experts, (num_tokens,), device=device)

        output = experts(hidden_states, expert_indices)

        assert output.device.type == "npu"
        assert output.shape == (num_tokens, hidden_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
