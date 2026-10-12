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

"""KDA PCP affine-state summary and merge kernels for GLM5-Next prefill."""

import triton
import triton.language as tl


@triton.jit
def kda_pcp_summary_kernel(
    inverse: tl.tensor,
    w: tl.tensor,
    u: tl.tensor,
    kg: tl.tensor,
    gate_cumsum: tl.tensor,
    cu_seqlens: tl.tensor,
    summary: tl.tensor,
    H: tl.constexpr,
    BT: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
) -> None:
    job = tl.program_id(0)
    row_blocks: tl.constexpr = triton.cdiv(D, BV)
    row_block = job % row_blocks
    head = (job // row_blocks) % H
    sequence = job // (row_blocks * H)
    rows = row_block * BV + tl.arange(0, BV)
    keys = tl.arange(0, D)
    chunk_rows = tl.arange(0, BT)
    ext = tl.full((BV, D), 0.0, tl.float32)
    transition = tl.where(rows[:, None] == keys[None, :], 1.0, 0.0).to(tl.float32)

    first_chunk = 0
    for previous in range(sequence):
        previous_begin = tl.load(cu_seqlens + previous)
        previous_end = tl.load(cu_seqlens + previous + 1)
        first_chunk += tl.cdiv(previous_end - previous_begin, BT)
    sequence_begin = tl.load(cu_seqlens + sequence)
    sequence_end = tl.load(cu_seqlens + sequence + 1)
    last_chunk = first_chunk + tl.cdiv(sequence_end - sequence_begin, BT)

    for chunk in range(first_chunk, last_chunk):
        local_chunk = chunk - first_chunk
        valid_rows = tl.minimum(BT, sequence_end - sequence_begin - local_chunk * BT)
        row_mask = chunk_rows < valid_rows
        factor_base = (chunk * H + head) * BT * D
        w_offsets = factor_base + chunk_rows[:, None] * D + keys[None, :]
        w_values = tl.load(w + w_offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
        u_offsets = factor_base + rows[:, None] * BT + chunk_rows[None, :]
        u_values = tl.load(u + u_offsets, mask=row_mask[None, :], other=0.0).to(tl.float32)
        ext_values = u_values - tl.dot(ext, tl.trans(w_values), allow_tf32=False)
        transition_values = -tl.dot(transition, tl.trans(w_values), allow_tf32=False)

        inverse_base = (chunk * H + head) * BT * BT
        inverse_offsets = inverse_base + chunk_rows[:, None] * BT + chunk_rows[None, :]
        inverse_values = tl.load(inverse + inverse_offsets).to(tl.float32)
        inverse_values = tl.where(chunk_rows[:, None] >= chunk_rows[None, :], inverse_values, 0.0)
        kg_values = tl.load(kg + w_offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
        decay = tl.extra.mlu.libdevice.fast_expf(tl.load(gate_cumsum + (chunk * H + head) * D + keys).to(tl.float32))
        if BT == 16:
            resolved_key = tl.dot(tl.trans(inverse_values), kg_values, allow_tf32=False)
            ext = ext * decay[None, :] + tl.dot(ext_values, resolved_key, allow_tf32=False)
            transition = transition * decay[None, :] + tl.dot(transition_values, resolved_key, allow_tf32=False)
        else:
            ext_values = tl.dot(ext_values, tl.trans(inverse_values), allow_tf32=False)
            transition_values = tl.dot(transition_values, tl.trans(inverse_values), allow_tf32=False)
            ext = ext * decay[None, :] + tl.dot(ext_values, kg_values, allow_tf32=False)
            transition = transition * decay[None, :] + tl.dot(transition_values, kg_values, allow_tf32=False)

    output_base = ((sequence * H + head) * 2 * D + rows[:, None]) * D + keys[None, :]
    tl.store(summary + output_base, ext)
    tl.store(summary + output_base + D * D, transition)


@triton.jit
def kda_pcp_merge_kernel(
    gathered_summary: tl.tensor,
    prefix_state: tl.tensor,
    local_initial_state: tl.tensor,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    RANK: tl.constexpr,
    BV: tl.constexpr,
) -> None:
    job = tl.program_id(0)
    row_blocks: tl.constexpr = triton.cdiv(D, BV)
    row_block = job % row_blocks
    head = (job // row_blocks) % H
    sequence = job // (row_blocks * H)
    rows = row_block * BV + tl.arange(0, BV)
    keys = tl.arange(0, D)
    state_offsets = ((sequence * H + head) * D + rows[:, None]) * D + keys[None, :]
    state = tl.load(prefix_state + state_offsets).to(tl.float32)

    for previous in range(0, RANK):
        summary_base = ((previous * N + sequence) * H + head) * 2 * D * D
        matrix_offsets = summary_base + (D + keys[:, None]) * D + keys[None, :]
        matrix = tl.load(gathered_summary + matrix_offsets).to(tl.float32)
        contribution_offsets = summary_base + rows[:, None] * D + keys[None, :]
        contribution = tl.load(gathered_summary + contribution_offsets).to(tl.float32)
        state = tl.dot(state, matrix, allow_tf32=False) + contribution

    tl.store(local_initial_state + state_offsets, state)
