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

"""NPU-native Qwen3.5 gated delta network composition."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.layers.npu.qwen3_5.gdn_metadata import (
    MEGA_GDN_MTP_MAX_VERIFY_TOKENS,
    MEGA_GDN_MTP_MIN_VERIFY_TOKENS,
    GdnDecodeMetadata,
    GdnMetadata,
    GdnPrefillMetadata,
    GdnSpecVerifyMetadata,
)
from xllm.python.layers.qwen3_5.common import Qwen3_5GatedDeltaNetConfig
from xllm.python.layers.qwen3_5.gated_delta_net import shard_qkv_rows
from xllm.python.model_executor.forward_context import get_execution_context
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader

_MEGA_GDN_MAX_DECODE_BATCH_SIZE = 32
_MEGA_GDN_FIXED_RMS_NORM_EPS = 1e-6
_SUPPORTED_MEGA_PREFILL_HEAD_COUNTS = frozenset((1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64))


def _validate_mega_gdn_head_geometry(
    num_key_heads: int,
    num_value_heads: int,
) -> None:
    """Reject model geometry unsupported by either MegaGdn phase."""
    if num_value_heads not in _SUPPORTED_MEGA_PREFILL_HEAD_COUNTS:
        raise NotImplementedError(f"Qwen3.5 MegaGdnPrefill does not support {num_value_heads} local value heads")
    if num_key_heads <= 0 or num_key_heads > 16:
        raise NotImplementedError("Qwen3.5 MegaGdnDecode supports 1 to 16 local key heads")
    if num_key_heads & (num_key_heads - 1):
        raise NotImplementedError("Qwen3.5 MegaGdnDecode requires a power-of-two local key-head count")
    if num_value_heads % num_key_heads != 0:
        raise NotImplementedError("Qwen3.5 MegaGdn requires Nv divisible by Nk")
    value_heads_per_key = num_value_heads // num_key_heads
    if value_heads_per_key > 4:
        raise NotImplementedError("Qwen3.5 MegaGdnDecode supports at most four value heads per key head")


class NpuQwen3_5GatedDeltaNet(nn.Module):
    """NPU graph whose nodes match the native fused operator boundaries."""

    def __init__(
        self,
        cfg: Qwen3_5GatedDeltaNetConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if not math.isclose(
            cfg.rms_norm_eps,
            _MEGA_GDN_FIXED_RMS_NORM_EPS,
            rel_tol=1e-6,
            abs_tol=0.0,
        ):
            raise NotImplementedError(
                "Qwen3.5 MegaGdn fuses RMSNorm with a fixed epsilon; "
                f"cfg.rms_norm_eps={cfg.rms_norm_eps} is unsupported"
            )
        self.cfg = cfg
        self.layer_id = layer_id
        self.num_k_heads = cfg.linear_num_key_heads // cfg.tp_size
        self.num_v_heads = cfg.linear_num_value_heads // cfg.tp_size
        _validate_mega_gdn_head_geometry(self.num_k_heads, self.num_v_heads)
        self.key_head_dim = cfg.linear_key_head_dim
        self.value_head_dim = cfg.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.key_head_dim
        self.value_dim = self.num_v_heads * self.value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim

        self.A_log = nn.Parameter(
            torch.empty(
                self.num_v_heads,
                dtype=torch.float32,
                device=device,
            )
        )
        # MegaGdn consumes dt_bias as FP32. Keep it in the required dtype once
        # instead of allocating a converted tensor on every forward.
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.num_v_heads,
                dtype=torch.float32,
                device=device,
            )
        )
        self.norm_weight = nn.Parameter(
            torch.ones(
                self.value_head_dim,
                dtype=dtype,
                device=device,
            )
        )
        self.out_proj = RowParallelLinear(
            self.value_dim,
            cfg.hidden_size,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        # Keep one rank-local storage for each pair of projections. Prefill
        # consumes contiguous row views, while Decode runs two packed
        # projections and splits their outputs for the existing Mega operator.
        self.in_proj_qkvz = ColumnParallelLinear(
            cfg.hidden_size,
            self.conv_dim + self.value_dim,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        self.in_proj_ba = ColumnParallelLinear(
            cfg.hidden_size,
            2 * self.num_v_heads,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        self.conv1d_weight = nn.Parameter(
            torch.empty(
                self.conv_kernel_size,
                self.conv_dim,
                dtype=dtype,
                device=device,
            )
        )
        self.register_buffer(
            "conv1d_bias",
            torch.zeros(self.conv_dim, dtype=dtype, device=device),
            persistent=False,
        )

    def _input_projection_weight_views(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_weight = self.in_proj_qkvz.weight[: self.conv_dim]
        z_weight = self.in_proj_qkvz.weight[self.conv_dim :]
        b_weight = self.in_proj_ba.weight[: self.num_v_heads]
        a_weight = self.in_proj_ba.weight[self.num_v_heads :]
        return qkv_weight, z_weight, b_weight, a_weight

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: ParallelLoadContext,
    ) -> None:
        qkv_weight, z_weight, b_weight, a_weight = self._input_projection_weight_views()
        state.copy(
            qkv_weight,
            shard_qkv_rows(
                state,
                state.get_tensor("in_proj_qkv.weight"),
                "in_proj_qkv.weight",
                self.cfg,
                context,
            ),
            "in_proj_qkv.weight",
        )
        for target, name in (
            (z_weight, "in_proj_z"),
            (b_weight, "in_proj_b"),
            (a_weight, "in_proj_a"),
        ):
            state.load_tensor(
                target,
                f"{name}.weight",
                dim=0,
                rank=context.tp_rank,
                world_size=context.tp_size,
            )

        local_conv = shard_qkv_rows(
            state,
            state.get_tensor("conv1d.weight").squeeze(1),
            "conv1d.weight",
            self.cfg,
            context,
        )
        state.copy(
            self.conv1d_weight,
            local_conv.transpose(0, 1).contiguous(),
            "conv1d.weight",
        )
        for name in ("A_log", "dt_bias"):
            state.load_tensor(
                getattr(self, name),
                name,
                dim=0,
                rank=context.tp_rank,
                world_size=context.tp_size,
            )
        state.load_tensor(self.norm_weight, "norm.weight")
        state.load_tensor(
            self.out_proj.weight,
            "out_proj.weight",
            dim=1,
            rank=context.tp_rank,
            world_size=context.tp_size,
        )
        self.out_proj.process_weights_after_loading()

    def _project_prefill_inputs(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_weight, z_weight, b_weight, a_weight = self._input_projection_weight_views()
        mixed_qkv = torch.nn.functional.linear(hidden, qkv_weight)
        z = torch.nn.functional.linear(hidden, z_weight).view(
            -1,
            self.num_v_heads,
            self.value_head_dim,
        )
        b = torch.nn.functional.linear(hidden, b_weight)
        a = torch.nn.functional.linear(hidden, a_weight)
        return mixed_qkv, a, b, z

    def _decode(
        self,
        hidden: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        read_state_indices: torch.Tensor,
        write_state_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden.shape[0]
        if batch_size <= 0 or read_state_indices.shape != (batch_size,) or write_state_indices.shape != (batch_size,):
            raise ValueError("Qwen3.5 MegaGdnDecode requires one state slot per token")

        mixed_qkvz = self.in_proj_qkvz(hidden)
        mixed_ba = self.in_proj_ba(hidden)
        if mixed_qkvz.dtype != torch.bfloat16 or mixed_ba.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode projections must be BF16")
        if mixed_qkvz.shape != (batch_size, self.conv_dim + self.value_dim):
            raise ValueError("Qwen3.5 MegaGdnDecode received an invalid packed QKVZ projection")
        if mixed_ba.shape != (batch_size, 2 * self.num_v_heads):
            raise ValueError("Qwen3.5 MegaGdnDecode received an invalid packed B/A projection")
        mixed_qkv, z = mixed_qkvz.split((self.conv_dim, self.value_dim), dim=-1)
        b, a = mixed_ba.split((self.num_v_heads, self.num_v_heads), dim=-1)
        z = z.view(batch_size, self.num_v_heads, self.value_head_dim)
        if self.conv_kernel_size != 4:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode requires convolution width 4")
        if self.key_head_dim != 128 or self.value_head_dim != 128:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode requires K/V head dimension 128")
        if self.num_k_heads <= 0 or self.num_k_heads > 16:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode supports 1 to 16 local key heads")
        if self.num_k_heads & (self.num_k_heads - 1):
            raise NotImplementedError("Qwen3.5 MegaGdnDecode requires a power-of-two local key-head count")
        if self.num_v_heads % self.num_k_heads != 0:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode requires Nv divisible by Nk")
        value_heads_per_key = self.num_v_heads // self.num_k_heads
        if value_heads_per_key < 1 or value_heads_per_key > 4:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode supports at most four value heads per key head")
        if self.conv1d_weight.shape != (self.conv_kernel_size, self.conv_dim):
            raise ValueError("Qwen3.5 MegaGdnDecode received an invalid Conv weight")
        if self.conv1d_weight.dtype != torch.bfloat16 or conv_state.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode Conv tensors must be BF16")
        if conv_state.dim() != 3 or conv_state.shape[1:] != (3, self.conv_dim):
            raise ValueError("Qwen3.5 MegaGdnDecode requires a three-token Conv cache")
        if ssm_state.dim() != 4 or ssm_state.shape != (
            conv_state.shape[0],
            self.num_v_heads,
            self.key_head_dim,
            self.value_head_dim,
        ):
            raise ValueError("Qwen3.5 MegaGdnDecode requires one SSM row per Conv slot")
        if ssm_state.dtype != torch.float32:
            raise NotImplementedError("Qwen3.5 MegaGdnDecode SSM cache must be FP32")
        if self.A_log.dtype != torch.float32 or self.dt_bias.dtype != torch.float32:
            raise ValueError("Qwen3.5 MegaGdnDecode A_log and dt_bias must be FP32")
        if self.norm_weight.dtype != torch.bfloat16:
            raise ValueError("Qwen3.5 MegaGdnDecode norm weight must be BF16")

        read_state_indices = read_state_indices.contiguous()
        write_state_indices = write_state_indices.contiguous()
        chunk_outputs = []
        for start in range(0, batch_size, _MEGA_GDN_MAX_DECODE_BATCH_SIZE):
            end = min(start + _MEGA_GDN_MAX_DECODE_BATCH_SIZE, batch_size)
            chunk_outputs.append(
                kernels.mega_gdn_decode(
                    mixed_qkv[start:end].contiguous(),
                    z[start:end].contiguous(),
                    b[start:end].contiguous(),
                    a[start:end].contiguous(),
                    self.conv1d_weight,
                    conv_state,
                    self.A_log,
                    self.dt_bias,
                    ssm_state,
                    read_state_indices[start:end].contiguous(),
                    write_state_indices[start:end].contiguous(),
                    self.norm_weight,
                    True,
                )
            )
        return chunk_outputs[0] if len(chunk_outputs) == 1 else torch.cat(chunk_outputs, dim=0)

    def _spec_verify(
        self,
        hidden: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        read_state_indices: torch.Tensor,
        write_state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Verify a speculative token block via the fused MegaGdn MTP decode.

        The kernel consumes dense per-sequence blocks ``[B, S, ...]``: the
        speculative path packs exactly ``S`` tokens per sequence, where ``S`` is
        the checkpoint stride carried by the (conv, ssm) cache shapes. State
        read/write indices and accepted-token counts are sequence-scoped;
        ``num_accepted_tokens`` lets the kernel commit recurrent state only up to
        each sequence's accepted prefix.
        """
        num_accepted_tokens = num_accepted_tokens.to(device=hidden.device, dtype=torch.int32)

        batch_size = read_state_indices.numel()
        if batch_size <= 0 or write_state_indices.numel() != batch_size:
            raise ValueError("Qwen3.5 MegaGdnMtpDecode requires one state slot per sequence")
        if num_accepted_tokens.numel() != batch_size:
            raise ValueError("Qwen3.5 MegaGdnMtpDecode accepted-token counts must be sequence-scoped")
        total_tokens = hidden.shape[0]
        if total_tokens % batch_size:
            raise ValueError("Qwen3.5 MegaGdnMtpDecode packs one equal-width block per sequence")
        sequence_length = total_tokens // batch_size
        if not MEGA_GDN_MTP_MIN_VERIFY_TOKENS <= sequence_length <= MEGA_GDN_MTP_MAX_VERIFY_TOKENS:
            raise NotImplementedError(
                f"Qwen3.5 MegaGdnMtpDecode supports {MEGA_GDN_MTP_MIN_VERIFY_TOKENS} "
                f"to {MEGA_GDN_MTP_MAX_VERIFY_TOKENS} verify tokens"
            )

        if conv_state.dim() != 3 or conv_state.shape[2] != self.conv_dim:
            raise ValueError("Qwen3.5 MegaGdnMtpDecode received an invalid Conv cache")
        if conv_state.shape[1] != sequence_length + 2:
            raise ValueError("Qwen3.5 MegaGdnMtpDecode requires an S+2 Conv history")
        if ssm_state.dim() != 4 or ssm_state.shape[1:] != (
            self.num_v_heads,
            self.key_head_dim,
            self.value_head_dim,
        ):
            raise ValueError("Qwen3.5 MegaGdnMtpDecode received an invalid SSM cache")
        if ssm_state.shape[0] != conv_state.shape[0] * sequence_length:
            raise ValueError("Qwen3.5 MegaGdnMtpDecode requires one SSM checkpoint per verify token")
        if self.conv1d_weight.dtype != torch.bfloat16 or conv_state.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.5 MegaGdnMtpDecode Conv tensors must be BF16")
        if ssm_state.dtype != torch.float32:
            raise NotImplementedError("Qwen3.5 MegaGdnMtpDecode SSM cache must be FP32")

        mixed_qkvz = self.in_proj_qkvz(hidden)
        mixed_ba = self.in_proj_ba(hidden)
        if mixed_qkvz.dtype != torch.bfloat16 or mixed_ba.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.5 MegaGdnMtpDecode projections must be BF16")
        mixed_qkv, z = mixed_qkvz.split((self.conv_dim, self.value_dim), dim=-1)
        b, a = mixed_ba.split((self.num_v_heads, self.num_v_heads), dim=-1)
        mixed_qkv = mixed_qkv.view(batch_size, sequence_length, self.conv_dim)
        z = z.view(batch_size, sequence_length, self.num_v_heads, self.value_head_dim)
        b = b.view(batch_size, sequence_length, self.num_v_heads)
        a = a.view(batch_size, sequence_length, self.num_v_heads)

        read_state_indices = read_state_indices.contiguous()
        write_state_indices = write_state_indices.contiguous()
        chunk_outputs = []
        for start in range(0, batch_size, _MEGA_GDN_MAX_DECODE_BATCH_SIZE):
            end = min(start + _MEGA_GDN_MAX_DECODE_BATCH_SIZE, batch_size)
            chunk_outputs.append(
                kernels.mega_gdn_mtp_decode(
                    mixed_qkv[start:end].contiguous(),
                    z[start:end].contiguous(),
                    b[start:end].contiguous(),
                    a[start:end].contiguous(),
                    self.conv1d_weight,
                    conv_state,
                    self.A_log,
                    self.dt_bias,
                    ssm_state,
                    read_state_indices[start:end].contiguous(),
                    write_state_indices[start:end].contiguous(),
                    num_accepted_tokens[start:end].contiguous(),
                    self.norm_weight,
                    True,
                )
            )
        return chunk_outputs[0] if len(chunk_outputs) == 1 else torch.cat(chunk_outputs, dim=0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        metadata = get_execution_context(GdnMetadata)
        if metadata is None:
            raise RuntimeError("Qwen3.5 GDN metadata is unavailable")
        state_cache = metadata.state_caches.get(self.layer_id)
        if state_cache is None:
            raise RuntimeError(f"Qwen3.5 GDN state cache is missing for layer {self.layer_id}")

        if isinstance(metadata, GdnPrefillMetadata):
            mixed_qkv, a, b, z = self._project_prefill_inputs(hidden)
            output = kernels.mega_gdn_prefill(
                mixed_qkv.contiguous(),
                b.contiguous(),
                a.contiguous(),
                z.contiguous(),
                self.conv1d_weight,
                state_cache.conv_state,
                self.A_log,
                self.dt_bias,
                metadata.conv_read_indices,
                metadata.conv_write_indices,
                metadata.ssm_read_indices,
                metadata.ssm_write_indices,
                state_cache.ssm_state,
                metadata.cu_seqlens.contiguous(),
                self.norm_weight,
                metadata.num_matrices,
            )
        elif isinstance(metadata, GdnSpecVerifyMetadata):
            tail_padding = metadata.num_tail_padding_tokens
            verify_hidden = hidden[:-tail_padding] if tail_padding else hidden
            output = self._spec_verify(
                verify_hidden,
                state_cache.conv_state,
                state_cache.ssm_state,
                metadata.read_state_indices,
                metadata.write_state_indices,
                metadata.num_accepted_tokens,
            )
            if tail_padding:
                # Keep incomplete padding out of the recurrent chain, then
                # restore the bucket shape before the TP output projection.
                output = torch.nn.functional.pad(output.reshape(-1, self.value_dim), (0, 0, 0, tail_padding))
        elif isinstance(metadata, GdnDecodeMetadata):
            output = self._decode(
                hidden,
                state_cache.conv_state,
                state_cache.ssm_state,
                metadata.read_state_indices,
                metadata.write_state_indices,
            )
        else:
            raise TypeError(f"Qwen3.5 GDN received unsupported metadata: {type(metadata).__name__}")
        return self.out_proj(output.reshape(-1, self.value_dim))
