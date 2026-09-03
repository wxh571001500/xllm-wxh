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

"""Vision encoder operators selected through PyTorch device dispatch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def encoder_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Sequence[int],
) -> torch.Tensor:
    """Run vLLM-compatible packed attention for independent media items."""
    if query.ndim != 3 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("vision attention expects equal [tokens, heads, dim] tensors")
    if not cu_seqlens or cu_seqlens[0] != 0:
        raise ValueError("vision cu_seqlens must start with zero")
    if cu_seqlens[-1] != query.shape[0]:
        raise ValueError(f"vision cu_seqlens cover {cu_seqlens[-1]} tokens, got {query.shape[0]}")

    actual_seq_lengths = list(cu_seqlens[1:])
    if query.device.type in ("npu", "privateuseone"):
        import torch_npu

        return torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key.contiguous(),
            value=value.contiguous(),
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths,
            num_heads=query.shape[1],
            num_key_value_heads=key.shape[1],
            scale=query.shape[2] ** -0.5,
            input_layout="TND",
            block_size=128,
            sparse_mode=0,
            pre_tokens=2147483647,
            next_tokens=2147483647,
        )[0]

    outputs: list[torch.Tensor] = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        if end <= start:
            raise ValueError(f"vision cu_seqlens must increase, got boundary {start}, {end}")
        q = query[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        k = key[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        v = value[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
        outputs.append(attended.squeeze(0).transpose(0, 1))
    if not outputs:
        return torch.empty_like(query)
    return torch.cat(outputs, dim=0)
