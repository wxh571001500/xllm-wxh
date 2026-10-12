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


# Preserve the separate gate upper bound and up two-sided bound.
# The output remains gate/up concatenated for the existing activation/quantizer.
import triton
import triton.language as tl


@triton.jit
def fused_moe_clamp(
    X: tl.tensor,
    Y: tl.tensor,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    LIMIT: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
) -> None:
    nc = tl.cdiv(WIDTH, BN)
    jobs = tl.cdiv(ROWS, BM) * nc
    limit = tl.full((), LIMIT, tl.float32).to(X.dtype.element_ty).to(tl.float32)
    for job in range(tl.program_id(0), jobs, tl.num_programs(0)):
        row = (job // nc) * BM
        col = (job % nc) * BN
        inp = tl.make_block_ptr(
            base=X,
            shape=(ROWS, WIDTH),
            strides=(STRIDE, 1),
            offsets=(row, col),
            block_shape=(BM, BN),
            order=(1, 0),
        )
        out = tl.make_block_ptr(
            base=Y,
            shape=(ROWS, WIDTH),
            strides=(WIDTH, 1),
            offsets=(row, col),
            block_shape=(BM, BN),
            order=(1, 0),
        )
        if ROWS % BM == 0 and WIDTH % BN == 0:
            value = tl.load(inp).to(tl.float32)
        else:
            value = tl.load(inp, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        column = col + tl.arange(0, BN)
        value = tl.where(value > limit, limit, value)
        value = tl.where((column[None, :] >= WIDTH // 2) & (value < -limit), -limit, value)
        if ROWS % BM == 0 and WIDTH % BN == 0:
            tl.store(out, value.to(Y.dtype.element_ty))
        else:
            tl.store(out, value.to(Y.dtype.element_ty), boundary_check=(0, 1))
