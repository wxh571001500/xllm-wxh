# Copyright 2025-2026 The xLLM Authors.
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

"""Reusable and backend-owned layers for Python model execution.

Layers depend only on the op dispatch layer (:mod:`python.ops`); they never
touch the kernel backends directly. The dependency direction is
``models -> layers -> ops -> kernels``.
Simple layers use the active :mod:`python.kernels` package directly. Complex
models whose native fusion boundaries differ by backend (e.g. Ascend vs CUDA)
select a kernel set at construction time.
"""

from xllm.python.layers.attention import Attention
from xllm.python.layers.embedding import HiddenParallelEmbedding
from xllm.python.layers.fused_moe import FusedMoE
from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.layers.layernorm import GemmaRMSNorm, RMSNorm
from xllm.python.layers.linear import (
    ColumnParallelLinear,
    LinearMethod,
    RowParallelLinear,
    W8A8DynamicLinearMethod,
)
from xllm.python.layers.rotary_embedding import RotaryEmbedding

# MoE classes are imported at module level but only exported when available.
# This avoids __getattr__ race conditions while keeping distributed dependencies
# optional for non-K3 models.
try:
    from xllm.python.layers.moe import (
        GroupedTopKRouter,
        KimiK3MoE,
        MoE,
        MoERunner,
        RoutedExperts,
        TensorParallelCommMethod,
    )

    _MOE_AVAILABLE = True
except ImportError:
    _MOE_AVAILABLE = False

__all__ = [
    "Attention",
    "FusedMoE",
    "GatedMLP",
    "GemmaRMSNorm",
    "RMSNorm",
    "RotaryEmbedding",
    "ColumnParallelLinear",
    "LinearMethod",
    "RowParallelLinear",
    "W8A8DynamicLinearMethod",
    "HiddenParallelEmbedding",
]

# Add MoE classes to __all__ only if successfully imported
if _MOE_AVAILABLE:
    __all__.extend(
        [
            "GroupedTopKRouter",
            "KimiK3MoE",
            "MoE",
            "MoERunner",
            "RoutedExperts",
            "TensorParallelCommMethod",
        ]
    )
