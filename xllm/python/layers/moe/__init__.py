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

"""Reusable MoE layers and model-specific compositions."""

from xllm.python.layers.moe.activation import SituAndMul
from xllm.python.layers.moe.communication import (
    MoECommMethod,
    TensorParallelCommMethod,
)
from xllm.python.layers.moe.experts import (
    FusedQuantizedSituAndMul,
    FusedW4A8RoutedExperts,
    QuantizedExpertsActivation,
    RoutedExperts,
    UnquantizedRoutedExperts,
    _dequant_situ_quant,
    _ensure_fused_w4a8_custom_op,
)
from xllm.python.layers.moe.moe import (
    KimiK3MoE,
    KimiK3MoERunner,
    MoE,
)
from xllm.python.layers.moe.prepare_finalize import (
    PrepareAndFinalize,
    TensorParallelPrepareAndFinalize,
)
from xllm.python.layers.moe.router import GroupedTopKRouter, MoERouter
from xllm.python.layers.moe.runner import MoERunner
from xllm.python.layers.moe.token_dispatcher import (
    FusedAllGatherTokenDispatcher,
    MoETokenDispatcher,
    NativeTokenDispatcher,
)
from xllm.python.layers.moe.types import (
    MoEExpertsConfig,
    MoEFusedExpertsResult,
    MoEPrepareOutput,
    MoERouterConfig,
    MoERoutingResult,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
)

# Compatibility for the Kimi accuracy tool created before the generic rename.
_ensure_kimi_k3_w4a8_custom_op = _ensure_fused_w4a8_custom_op

__all__ = [
    "FusedAllGatherTokenDispatcher",
    "FusedQuantizedSituAndMul",
    "FusedW4A8RoutedExperts",
    "GroupedTopKRouter",
    "KimiK3MoE",
    "KimiK3MoERunner",
    "MoE",
    "MoECommMethod",
    "MoEExpertsConfig",
    "MoEFusedExpertsResult",
    "MoEPrepareOutput",
    "MoERouter",
    "MoERouterConfig",
    "MoERoutingResult",
    "MoERunner",
    "MoETokenDispatchInput",
    "MoETokenDispatchOutput",
    "MoETokenDispatcher",
    "NativeTokenDispatcher",
    "PrepareAndFinalize",
    "QuantizedExpertsActivation",
    "RoutedExperts",
    "SituAndMul",
    "TensorParallelCommMethod",
    "TensorParallelPrepareAndFinalize",
    "UnquantizedRoutedExperts",
]
