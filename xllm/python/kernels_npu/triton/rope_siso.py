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

"""NPU Triton in-place partial RoPE for a single tensor."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import get_vectorcore_num


@triton.jit
def _triton_rope_siso(
    qk_ptr,
    qk_row_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    num_tokens,
    n_h: tl.constexpr,
    hd: tl.constexpr,
    rope_dim: tl.constexpr,
    pad_n_h: tl.constexpr,
    pad_rope_dim: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_block_size = tl.num_programs(0)

    for row_idx in tl.range(pid, num_tokens, row_block_size):
        qk_start_ptr = qk_ptr + row_idx * qk_row_stride
        cos_start_ptr = cos_ptr + row_idx * cos_row_stride
        sin_start_ptr = sin_ptr + row_idx * sin_row_stride
        head_offsets = tl.arange(0, pad_n_h)[:, None] * hd

        if IS_NEOX_STYLE:
            cos_offsets = tl.arange(0, pad_rope_dim // 2)
            cos_mask = cos_offsets < (rope_dim // 2)
            cos_row = tl.load(cos_start_ptr + cos_offsets, mask=cos_mask, other=0)
            sin_row = tl.load(sin_start_ptr + cos_offsets, mask=cos_mask, other=0)
            rotary_offsets = tl.arange(0, pad_rope_dim // 2)[None, :]
            first_half_offsets = head_offsets + rotary_offsets
            first_mask = (tl.arange(0, pad_n_h)[:, None] < n_h) & (rotary_offsets < (rope_dim // 2))
            qk_tile_1 = tl.load(qk_start_ptr + first_half_offsets, mask=first_mask, other=0)
            second_half_offsets = first_half_offsets + (rope_dim // 2)
            qk_tile_2 = tl.load(qk_start_ptr + second_half_offsets, mask=first_mask, other=0)

            new_qk_tile_1 = qk_tile_1 * cos_row - qk_tile_2 * sin_row
            new_qk_tile_2 = qk_tile_2 * cos_row + qk_tile_1 * sin_row
            tl.store(qk_start_ptr + first_half_offsets, new_qk_tile_1, mask=first_mask)
            tl.store(qk_start_ptr + second_half_offsets, new_qk_tile_2, mask=first_mask)
        else:
            # A single contiguous store avoids two masked BF16 stores to
            # alternating channels when heads or RoPE channels are padded.
            rotary_offsets = tl.arange(0, pad_rope_dim)[None, :]
            pair_offsets = rotary_offsets // 2
            pair_mask = pair_offsets < (rope_dim // 2)
            cos_row = tl.load(cos_start_ptr + pair_offsets, mask=pair_mask, other=0)
            sin_row = tl.load(sin_start_ptr + pair_offsets, mask=pair_mask, other=0)
            row_mask = (tl.arange(0, pad_n_h)[:, None] < n_h) & (rotary_offsets < rope_dim)
            qk_offsets = head_offsets + rotary_offsets
            qk_tile = tl.load(qk_start_ptr + qk_offsets, mask=row_mask, other=0)
            paired_tile = tl.load(qk_start_ptr + head_offsets + (rotary_offsets ^ 1), mask=row_mask, other=0)
            rotated_first = qk_tile * cos_row - paired_tile * sin_row
            rotated_second = qk_tile * cos_row + paired_tile * sin_row
            rotated = tl.where((rotary_offsets % 2) == 0, rotated_first, rotated_second)
            tl.store(qk_start_ptr + qk_offsets, rotated, mask=row_mask)


def rope_forward_triton_siso(
    qk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    is_neox_style: bool = True,
) -> torch.Tensor:
    """Rotate the leading RoPE channels of a contiguous 3D tensor in place."""
    if qk.dim() != 3:
        raise ValueError(f"qk must be 3D, got {qk.dim()} dimensions")
    if not qk.is_contiguous():
        raise ValueError("qk must be contiguous for in-place RoPE")

    num_tokens, n_head, head_dim = qk.shape
    if cos.dim() != 2 or sin.dim() != 2 or cos.shape[0] != num_tokens or sin.shape[0] != num_tokens:
        raise ValueError("cos and sin must be 2D with one row per token")
    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have identical shapes")
    if rope_dim <= 0 or rope_dim > head_dim or rope_dim % 2 != 0:
        raise ValueError("rope_dim must be positive, even, and no larger than head_dim")
    if cos.shape[-1] != rope_dim // 2:
        raise ValueError("cos and sin must contain half-width RoPE tables")
    if cos.dtype != qk.dtype or sin.dtype != qk.dtype:
        raise ValueError("cos and sin must have the same dtype as qk")
    if cos.device != qk.device or sin.device != qk.device:
        raise ValueError("cos and sin must be on the same device as qk")
    if cos.stride(-1) != 1 or sin.stride(-1) != 1:
        raise ValueError("cos and sin must have contiguous RoPE channels")
    if num_tokens == 0:
        return qk

    pad_rope_dim = triton.next_power_of_2(rope_dim)
    pad_n_head = triton.next_power_of_2(n_head)
    if not is_neox_style:
        # Singleton 2D tiles can read zero from paired BF16 channels on NPU.
        # Pad to two rows; the extra row remains masked by n_h.
        pad_n_head = max(2, pad_n_head)
    num_vectorcore = get_vectorcore_num()
    if num_vectorcore <= 0:
        raise RuntimeError(f"invalid NPU vector core count reported by Triton: {num_vectorcore}")
    n_row = min(num_tokens, num_vectorcore)
    _triton_rope_siso[(n_row,)](
        qk,
        qk.stride(0),
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        num_tokens,
        n_head,
        head_dim,
        rope_dim,
        pad_n_head,
        pad_rope_dim,
        IS_NEOX_STYLE=is_neox_style,
    )
    return qk


__all__ = ["rope_forward_triton_siso"]
