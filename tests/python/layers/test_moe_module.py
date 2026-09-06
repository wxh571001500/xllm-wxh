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

"""Unit tests for MoE module.

Note: These tests verify the structure and configuration of MoE components.
Full integration tests with forward passes require xllm.python runtime
initialization which is done by the C++ framework in production.
"""

import pytest
import torch

from xllm.python.layers.moe.activation import SituAndMul
from xllm.python.layers.moe.types import (
    MoECommType,
    MoEExpertsConfig,
    MoEParallelConfig,
    MoERouterConfig,
)


class TestMoEBasicFunctionality:
    """Test basic MoE functionality."""

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


class TestMoEConfigurations:
    """Test different MoE configurations."""

    def test_router_config_creation(self):
        """Test MoE router configuration creation."""
        config = MoERouterConfig(
            num_experts=8,
            top_k=2,
            normalize_topk_prob=True,
        )

        assert config.num_experts == 8
        assert config.top_k == 2
        assert config.normalize_topk_prob is True

    def test_experts_config_creation(self):
        """Test MoE experts configuration creation."""
        config = MoEExpertsConfig(
            num_experts=8,
            hidden_size=256,
            intermediate_size=1024,
        )

        assert config.num_experts == 8
        assert config.hidden_size == 256
        assert config.intermediate_size == 1024

    def test_parallel_config_creation(self):
        """Test MoE parallel configuration creation."""
        config = MoEParallelConfig(
            tp_size=1,
            ep_size=1,
            comm_type=MoECommType.TENSOR_PARALLEL,
        )

        assert config.tp_size == 1
        assert config.ep_size == 1
        assert config.comm_type == MoECommType.TENSOR_PARALLEL

    def test_moe_comm_types(self):
        """Test MoE communication type enum."""
        assert MoECommType.TENSOR_PARALLEL.value == "tensor_parallel"
        assert MoECommType.ALL_TO_ALL.value == "all_to_all"
        assert MoECommType.ALL_GATHER.value == "all_gather"
        assert MoECommType.MC2.value == "mc2"

    def test_parallel_config_with_different_comm_types(self):
        """Test parallel config with different communication types."""
        comm_types = [
            MoECommType.TENSOR_PARALLEL,
            MoECommType.ALL_TO_ALL,
            MoECommType.ALL_GATHER,
            MoECommType.MC2,
        ]

        for comm_type in comm_types:
            config = MoEParallelConfig(
                tp_size=4,
                ep_size=2,
                comm_type=comm_type,
            )
            assert config.comm_type == comm_type
            assert config.tp_size == 4
            assert config.ep_size == 2


class TestMoEActivation:
    """Test MoE activation functions."""

    def test_situ_and_mul_shape(self):
        """Test SituAndMul output shape."""
        activation = SituAndMul()

        # Test different batch sizes
        for batch_size in [1, 4, 16, 32]:
            for hidden_size in [128, 256, 512]:
                x = torch.randn(batch_size, hidden_size * 2)
                output = activation(x)
                assert output.shape == (batch_size, hidden_size)

    def test_situ_and_mul_numerical_stability(self):
        """Test SituAndMul numerical stability with edge cases."""
        activation = SituAndMul()

        # Test with zeros
        x = torch.zeros(4, 512)
        output = activation(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Test with large values
        x = torch.ones(4, 512) * 10.0
        output = activation(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Test with negative values
        x = torch.ones(4, 512) * -10.0
        output = activation(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
class TestMoENPU:
    """Test MoE components on NPU device."""

    def test_situ_and_mul_npu(self):
        """Test SituAndMul on NPU."""
        device = torch.device("npu")
        activation = SituAndMul().to(device)

        x = torch.randn(16, 512, device=device)
        output = activation(x)

        assert output.device.type == "npu"
        assert output.shape == (16, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
