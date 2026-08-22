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

"""Standalone eager PyTorch implementation of Kimi K3 Gated-MLA."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KimiK3GatedMLAConfig:
    """Tensor dimensions required by the standalone Gated-MLA module."""

    hidden_size: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float = 1e-6

    def validate(self) -> None:
        dimensions = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
        }
        invalid = [name for name, value in dimensions.items() if value <= 0]
        if invalid:
            raise ValueError(f"Kimi K3 Gated-MLA dimensions must be positive: {invalid}")
        if self.rms_norm_eps <= 0:
            raise ValueError("Kimi K3 Gated-MLA RMSNorm epsilon must be positive")


class _RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return normalized.to(hidden_states.dtype) * self.weight


class KimiK3GatedMLA(nn.Module):
    """Dense causal Kimi K3 Gated-MLA for module-level alignment.

    Kimi K3 does not apply RoPE in MLA. The checkpoint's RoPE-named query
    and key slices are retained as position-independent attention features.
    Output gating is ``attention_output * sigmoid(g_proj(hidden_states))``
    and is applied before ``o_proj``.
    """

    _WEIGHT_NAMES = (
        "q_a_proj.weight",
        "q_a_layernorm.weight",
        "q_b_proj.weight",
        "kv_a_proj_with_mqa.weight",
        "kv_a_layernorm.weight",
        "kv_b_proj.weight",
        "g_proj.weight",
        "o_proj.weight",
    )

    def __init__(
        self,
        config: KimiK3GatedMLAConfig,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.scale = (config.qk_nope_head_dim + config.qk_rope_head_dim) ** -0.5
        query_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        kv_head_dim = config.qk_nope_head_dim + config.v_head_dim
        output_size = config.num_attention_heads * config.v_head_dim

        self.q_a_proj = nn.Linear(
            config.hidden_size, config.q_lora_rank, bias=False, dtype=dtype, device=device
        )
        self.q_a_layernorm = _RMSNorm(
            config.q_lora_rank, config.rms_norm_eps, dtype, device
        )
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            config.num_attention_heads * query_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.kv_a_layernorm = _RMSNorm(
            config.kv_lora_rank, config.rms_norm_eps, dtype, device
        )
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.num_attention_heads * kv_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.g_proj = nn.Linear(
            config.hidden_size, output_size, bias=False, dtype=dtype, device=device
        )
        self.o_proj = nn.Linear(
            output_size, config.hidden_size, bias=False, dtype=dtype, device=device
        )

    def load_checkpoint_weights(
        self,
        weights: Mapping[str, torch.Tensor],
        prefix: str = "",
    ) -> None:
        """Load one attention layer, including per-output-channel W8 weights."""
        parameters = dict(self.named_parameters())
        missing: list[str] = []
        for name in self._WEIGHT_NAMES:
            checkpoint_name = f"{prefix}{name}"
            tensor = weights.get(checkpoint_name)
            if tensor is None:
                missing.append(checkpoint_name)
                continue
            parameter = parameters[name]
            if tensor.shape != parameter.shape:
                raise ValueError(
                    f"shape mismatch for {checkpoint_name}: expected "
                    f"{tuple(parameter.shape)}, got {tuple(tensor.shape)}"
                )
            if tensor.is_floating_point():
                loaded = tensor
            else:
                scale_name = f"{checkpoint_name}_scale"
                offset_name = f"{checkpoint_name}_offset"
                scale = weights.get(scale_name)
                offset = weights.get(offset_name)
                if scale is None or offset is None:
                    raise KeyError(
                        "quantized Kimi K3 weight requires scale and offset: "
                        f"{scale_name}, {offset_name}"
                    )
                if scale.shape not in ((tensor.shape[0],), (tensor.shape[0], 1)):
                    raise ValueError(
                        f"unsupported scale shape for {checkpoint_name}: "
                        f"{tuple(scale.shape)}"
                    )
                if offset.shape != scale.shape:
                    raise ValueError(
                        f"offset shape for {checkpoint_name} must match scale shape"
                    )
                # Kimi K3 W8A8_DYNAMIC stores one scale and zero point per
                # output channel. The current checkpoint has all-zero offsets.
                if scale.ndim == 1:
                    scale = scale.unsqueeze(1)
                    offset = offset.unsqueeze(1)
                loaded = (tensor.float() - offset.float()) * scale.float()
            parameter.data.copy_(loaded.to(device=parameter.device, dtype=parameter.dtype))
        if missing:
            raise KeyError(f"missing Kimi K3 Gated-MLA weights: {missing}")

    def _project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self.config
        num_tokens = hidden_states.shape[0]
        query = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query = query.view(num_tokens, config.num_attention_heads, -1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, key_position_independent = compressed_kv.split(
            [config.kv_lora_rank, config.qk_rope_head_dim], dim=-1
        )
        key_value = self.kv_b_proj(self.kv_a_layernorm(kv_latent))
        key_value = key_value.view(num_tokens, config.num_attention_heads, -1)
        key_nope, value = key_value.split(
            [config.qk_nope_head_dim, config.v_head_dim], dim=-1
        )
        key_position_independent = key_position_independent.unsqueeze(1).expand(
            -1, config.num_attention_heads, -1
        )
        key = torch.cat([key_nope, key_position_independent], dim=-1)
        return query, key, value

    def _causal_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sequence_lengths: Sequence[int],
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        start = 0
        for length in sequence_lengths:
            end = start + length
            sequence_query = query[start:end].transpose(0, 1)
            sequence_key = key[start:end].transpose(0, 1)
            sequence_value = value[start:end].transpose(0, 1)
            scores = torch.matmul(sequence_query, sequence_key.transpose(-1, -2))
            scores = scores * self.scale
            mask = torch.ones(
                length, length, dtype=torch.bool, device=query.device
            ).triu(diagonal=1)
            scores = scores.masked_fill(mask, -math.inf)
            probabilities = F.softmax(scores.float(), dim=-1).to(value.dtype)
            outputs.append(torch.matmul(probabilities, sequence_value).transpose(0, 1))
            start = end
        return torch.cat(outputs, dim=0)

    def apply_output_gate(
        self,
        attention_output: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Kimi K3's output gate before the output projection."""
        gate = torch.sigmoid(self.g_proj(hidden_states))
        return self.o_proj(attention_output * gate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "Kimi K3 Gated-MLA hidden states must have shape "
                f"[tokens, {self.config.hidden_size}]"
            )
        num_tokens = hidden_states.shape[0]
        lengths = [num_tokens] if sequence_lengths is None else list(sequence_lengths)
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("Kimi K3 Gated-MLA sequence lengths must be positive")
        if sum(lengths) != num_tokens:
            raise ValueError("Kimi K3 Gated-MLA sequence lengths must sum to token count")

        query, key, value = self._project_qkv(hidden_states)
        attended = self._causal_attention(query, key, value, lengths)
        attended = attended.reshape(num_tokens, -1)
        return self.apply_output_gate(attended, hidden_states)