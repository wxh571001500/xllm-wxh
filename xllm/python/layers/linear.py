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
the same all-reduce / all-gather the native C++ parallel layers use (via the op
dispatch layer :mod:`python.ops`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from xllm.python import ops


def _copy_parameter(parameter: torch.Tensor, tensor: torch.Tensor) -> None:
    if parameter.shape != tensor.shape:
        raise ValueError(f"Linear parameter expects {parameter.shape}, got {tensor.shape}")
    parameter.data.copy_(tensor.to(dtype=parameter.dtype, device=parameter.device))


class LinearMethod(ABC):
    """Parameter and compute policy owned by reusable linear layers."""

    @abstractmethod
    def create_weights(
        self,
        layer: nn.Module,
        in_features: int,
        out_features: int,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def load_weight(
        self,
        layer: nn.Module,
        name: str,
        tensor: torch.Tensor,
    ) -> bool:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        del layer


class W8A8DynamicLinearMethod(LinearMethod):
    """Dynamic per-token W8A8 compute with per-channel weight metadata."""

    _MAX_OUTPUT_DIM = 65535

    def create_weights(
        self,
        layer: nn.Module,
        in_features: int,
        out_features: int,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
    ) -> None:
        del dtype
        layer.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=torch.int8,
                device=device,
            ),
            requires_grad=False,
        )
        layer.register_buffer(
            "weight_scale",
            torch.empty(
                out_features,
                1,
                dtype=torch.float32,
                device=device,
            ),
        )
        layer.register_buffer(
            "weight_offset",
            torch.empty(
                out_features,
                1,
                dtype=torch.float32,
                device=device,
            ),
        )

    def apply(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if not layer._weights_processed:
            raise RuntimeError("W8A8 dynamic weights have not finished loading")
        quantized, per_token_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
        return ops.quant_matmul(
            quantized,
            layer.weight,
            False,
            layer.weight_scale,
            None,
            per_token_scale,
            bias,
            hidden_states.dtype,
        )

    def load_weight(
        self,
        layer: nn.Module,
        name: str,
        tensor: torch.Tensor,
    ) -> bool:
        targets = {
            "weight": layer.weight,
            "weight_scale": layer.weight_scale,
            "weight_offset": layer.weight_offset,
        }
        target = targets.get(name)
        if target is None:
            return False
        _copy_parameter(target, tensor)
        return True

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if layer._weights_processed:
            return
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten().contiguous()
        layer.weight_offset.data = layer.weight_offset.data.flatten().contiguous()
        # aclnnQuantMatmulV4 currently limits a single output dimension to
        # 65535. Model-specific splitting should live in the owning module,
        # rather than changing the behavior of every shared W8A8 linear.
        if layer.weight.shape[1] > self._MAX_OUTPUT_DIM:
            raise ValueError(
                "W8A8 output dimension exceeds the Ascend quant_matmul "
                f"limit: {layer.weight.shape[1]} > {self._MAX_OUTPUT_DIM}"
            )
        layer._weights_processed = True


class _LinearBase(nn.Module):
    """Shared parameter ownership for local and tensor-parallel linears."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
        quant_method: LinearMethod | None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_method = quant_method
        self._weights_processed = quant_method is None
        if quant_method is None:
            self.weight = nn.Parameter(
                torch.empty(
                    out_features,
                    in_features,
                    dtype=dtype,
                    device=device,
                )
            )
        else:
            quant_method.create_weights(
                self,
                in_features,
                out_features,
                dtype,
                device,
            )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def _apply_linear(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._apply_linear_with_bias(hidden_states, self.bias)

    def _apply_linear_with_bias(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.quant_method is None:
            return torch.nn.functional.linear(
                hidden_states,
                self.weight,
                bias,
            )
        return self.quant_method.apply(self, hidden_states, bias)

    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        if name == "bias":
            if self.bias is None:
                return False
            _copy_parameter(self.bias, tensor)
            return True
        if self.quant_method is not None:
            return self.quant_method.load_weight(self, name, tensor)
        target = self.weight if name == "weight" else None
        if target is None:
            return False
        _copy_parameter(target, tensor)
        return True

    def finish_weight_loading(self) -> None:
        if self.quant_method is not None:
            self.quant_method.process_weights_after_loading(self)


class ColumnParallelLinear(_LinearBase):
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
        quant_method: LinearMethod | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features_per_partition,
            bias,
            dtype,
            device,
            quant_method,
        )
        self.tp_size = tp_size
        self.gather_output = gather_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._apply_linear(x)
        if self.gather_output and self.tp_size > 1:
            out = ops.all_gather(out, dim=-1, world_size=self.tp_size)
        return out


class RowParallelLinear(_LinearBase):
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
        quant_method: LinearMethod | None = None,
    ) -> None:
        super().__init__(
            in_features_per_partition,
            out_features,
            bias,
            dtype,
            device,
            quant_method,
        )
        self.tp_size = tp_size
        self.reduce_results = reduce_results
        self._weight_is_transposed = False

    def process_weights_after_loading(self) -> None:
        """Prepare the weight layout selected by the active device backend.

        Compatibility shim for existing model code (qwen3, deepseek_v32/v4, glm5_2).
        Calls both quantization finalization and NPU format conversion.
        """
        self.finish_weight_loading()
        self.format_npu_weight_()

    def format_npu_weight_(self) -> None:
        """Store the weight as ``[K, N]`` FRACTAL_NZ for non-transposed matmul."""
        if self.quant_method is not None:
            return
        if self.weight.device.type not in ("npu", "privateuseone"):
            return
        if self._weight_is_transposed:
            return
        import torch_npu

        transposed = self.weight.data.transpose(0, 1).contiguous()
        self.weight.data = torch_npu.npu_format_cast(transposed, 29)
        self._weight_is_transposed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_is_transposed:
            out = torch.matmul(x, self.weight)
        else:
            out = self._apply_linear_with_bias(x, None)
        if self.reduce_results and self.tp_size > 1:
            ops.all_reduce_(out)
        if self.bias is not None:
            out = out + self.bias
        return out


__all__ = [
    "ColumnParallelLinear",
    "LinearMethod",
    "RowParallelLinear",
    "W8A8DynamicLinearMethod",
]
