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

"""Tensor-parallel linear layers.

At ``tp_size==1`` these hold full-size weights and skip all collectives, so they
are numerically identical to plain ``nn.Linear`` and preserve the single-card
byte parity. At ``tp_size>1`` each rank holds a per-partition shard and inserts
the same all-reduce / all-gather the native C++ parallel layers use (via
:mod:`python.distributed`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import distributed, kernels


class ColumnParallelLinear(nn.Module):
    """Linear sharded on the output dim (dim 0): each rank owns
    ``[out_per_partition, in]`` and computes its slice of the output. No
    communication unless ``gather_output`` (then an all-gather along the last
    dim reconstructs the full output — used by lm_head). An optional bias is
    sharded on the output dim like the weight and applied per partition (before
    any gather). Mirrors native ColumnParallelLinear / QKVParallelLinear (which
    set gather_output=False so the following RowParallel all-reduce combines the
    partial outputs).
    """

    def __init__(
        self,
        in_features: int,
        out_features_per_partition: int,
        tp_size: int,
        gather_output: bool = False,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.gather_output = gather_output
        self.weight = nn.Parameter(
            torch.empty(
                out_features_per_partition,
                in_features,
                dtype=dtype,
                device=device,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features_per_partition, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.gather_output and self.tp_size > 1:
            out = distributed.tp_all_gather(out, dim=-1, world_size=self.tp_size)
        return out


class RowParallelLinear(nn.Module):
    """Linear sharded on the input dim (dim 1): each rank owns
    ``[out, in_per_partition]`` and consumes its slice of an already-partitioned
    input, producing a partial output that is SUM all-reduced across the TP
    group. An optional bias is replicated (full ``out``) and added once AFTER
    the all-reduce, so it is not summed ``tp_size`` times. Mirrors native
    RowParallelLinear (o_proj / down_proj with enable_result_reduction=true).
    """

    def __init__(
        self,
        in_features_per_partition: int,
        out_features: int,
        tp_size: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self._weight_is_transposed = False
        self.reduce_results = reduce_results
        if bias and not reduce_results:
            # The bias is replicated and must be added exactly once, which is
            # only possible here when this layer owns the reduction.
            raise ValueError("a deferred reduction cannot be combined with a replicated bias")
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features_per_partition,
                dtype=dtype,
                device=device,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def process_weights_after_loading(self) -> None:
        """Prepare the weight layout selected by the active device backend."""
        if self._weight_is_transposed:
            return
        prepared, is_transposed = kernels.prepare_row_parallel_weight(self.weight.data)
        self.weight.data = prepared
        self._weight_is_transposed = is_transposed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_is_transposed:
            out = torch.matmul(x, self.weight)
        else:
            out = torch.nn.functional.linear(x, self.weight)
        if self.tp_size > 1 and self.reduce_results:
            distributed.tp_all_reduce(out)
        if self.bias is not None:
            out = out + self.bias
        return out


def _copy_parameter(parameter: torch.Tensor, tensor: torch.Tensor) -> None:
    if parameter.shape != tensor.shape:
        raise ValueError(
            f"Kimi K3 parameter expects {parameter.shape}, got {tensor.shape}"
        )
    parameter.data.copy_(tensor.to(dtype=parameter.dtype, device=parameter.device))


class KimiK3W8A8DynamicLinear(nn.Module):
    """Kimi dynamic W8A8 linear (int8 weight + int8 dynamic-quant activation).

    The Kimi-K3 ``w8a8_dynamic`` checkpoint stores the weight as int8
    ``[out, in]`` with a per-output-channel float32 ``weight_scale`` /
    ``weight_offset``. At runtime the activation is dynamically quantized per
    token to int8 (``npu_dynamic_quant``) and the matmul is an int8 x int8
    ``quant_matmul`` dequantized back to the activation dtype. This mirrors
    vllm-ascend's ``AscendW8A8DynamicLinearMethod`` and backs the dense MLP, the
    routed latent projections, and the KDA q/k/v projections.

    Callers hand ``load_weight`` already-sharded tensors and must invoke
    ``finish_weight_loading`` before the first forward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device,
        tp_size: int = 1,
        reduce_results: bool = False,
        gather_output: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.reduce_results = reduce_results
        self.gather_output = gather_output
        self._processed = False
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=torch.int8,
                device=device,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(out_features, 1, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "weight_offset",
            torch.empty(out_features, 1, dtype=torch.float32, device=device),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._processed:
            raise RuntimeError("Kimi K3 W8A8 weights have not finished loading")
        quantized, per_token_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
        output = ops.quant_matmul(
            quantized,
            self.weight,
            False,
            self.weight_scale,
            None,
            per_token_scale,
            None,
            hidden_states.dtype,
        )
        if self.reduce_results and self.tp_size > 1:
            ops.all_reduce_(output)
        if self.gather_output and self.tp_size > 1:
            output = ops.all_gather(output, dim=-1, world_size=self.tp_size)
        return output

    def load_weight(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> bool:
        targets = {
            "weight": self.weight,
            "weight_scale": self.weight_scale,
            "weight_offset": self.weight_offset,
        }
        target = targets.get(name)
        if target is None:
            return False
        _copy_parameter(target, tensor)
        return True

    def finish_weight_loading(self) -> None:
        if self._processed:
            return
        self.weight.data = self.weight.data.transpose(0, 1).contiguous()
        self.weight_scale.data = self.weight_scale.data.flatten().contiguous()
        self.weight_offset.data = self.weight_offset.data.flatten().contiguous()
        self._processed = True
