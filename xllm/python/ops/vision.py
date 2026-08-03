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

"""Vision encoder operators selected through PyTorch device dispatch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def encoder_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sequence_lengths: Sequence[int],
) -> torch.Tensor:
    """Run non-causal packed vision attention for independent media items."""
    if query.ndim != 3 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("vision attention expects equal [tokens, heads, dim] tensors")

    outputs: list[torch.Tensor] = []
    start = 0
    for length in sequence_lengths:
        if length <= 0:
            raise ValueError(f"vision sequence length must be positive, got {length}")
        end = start + length
        q = query[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        k = key[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        v = value[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        if query.device.type in ("npu", "privateuseone"):
            import torch_npu

            attended = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num=query.shape[1],
                input_layout="BNSD",
                scale=query.shape[2] ** -0.5,
                keep_prob=1.0,
                pre_tockens=65535,
                next_tockens=65535,
            )[0]
        else:
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
            )
        outputs.append(attended.squeeze(0).transpose(0, 1))
        start = end

    if start != query.shape[0]:
        raise ValueError(
            f"vision sequence lengths cover {start} tokens, got {query.shape[0]}"
        )
    if not outputs:
        return torch.empty_like(query)
    return torch.cat(outputs, dim=0)
