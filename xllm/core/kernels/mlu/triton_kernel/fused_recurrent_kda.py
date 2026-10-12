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

# Local implementation of sequence/head/value parallel KDA recurrence. The
# independent-tile scheduling follows vLLM's GLM5Next KDA implementation at
# commit 58ad1f3b8973b23943107b51230d594050b42ec3, with MLU value tiles.
# The arithmetic follows xLLM's fused_sigmoid_gating_delta_rule_update.py,
# derived from vLLM and flash-linear-attention (Songlin Yang, Yu Zhang).

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n"])
def fused_recurrent_kda_kernel(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    v_ptr: tl.tensor,
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    a_log_ptr: tl.tensor,
    dt_bias_ptr: tl.tensor,
    initial_state_ptr: tl.tensor,
    final_state_ptr: tl.tensor,
    output_ptr: tl.tensor,
    cu_seqlens_ptr: tl.tensor,
    state_indices_ptr: tl.tensor,
    accepted_tokens_ptr: tl.tensor,
    n: tl.int32,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    STRIDE_INDICES_SEQ: tl.constexpr,
    STRIDE_INDICES_TOK: tl.constexpr,
    SCALE: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    SPEC: tl.constexpr,
    INPLACE: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
) -> None:
    nv: tl.constexpr = triton.cdiv(DV, BV)
    tile = tl.program_id(0)
    seq = tile // (HV * nv)
    head = tile // nv % HV
    value_tile = tile % nv
    kh = head // (HV // H)
    kk = tl.arange(0, BK)
    vv = value_tile * BV + tl.arange(0, BV)
    mask = (vv[:, None] < DV) & (kk[None, :] < DK)
    bos = tl.load(cu_seqlens_ptr + seq)
    eos = tl.load(cu_seqlens_ptr + seq + 1)
    if bos == eos:
        return
    accepted = 0
    if SPEC:
        accepted = tl.load(accepted_tokens_ptr + seq) - 1
    slot = tl.load(state_indices_ptr + seq * STRIDE_INDICES_SEQ + accepted * STRIDE_INDICES_TOK).to(tl.int64)
    if slot <= 0:
        return
    state_offset = head * DV * DK + vv[:, None] * DK + kk[None, :]
    state = tl.load(initial_state_ptr + slot * HV * DV * DK + state_offset, mask=mask, other=0).to(tl.float32)
    bias = tl.load(dt_bias_ptr + head * DK + kk, mask=kk < DK, other=0).to(tl.float32)
    a_scale = tl.exp(tl.load(a_log_ptr + head).to(tl.float32))
    for token in range(bos, eos):
        q = tl.load(q_ptr + (token * H + kh) * DK + kk, mask=kk < DK, other=0).to(tl.float32)
        k = tl.load(k_ptr + (token * H + kh) * DK + kk, mask=kk < DK, other=0).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6) * SCALE
        k = k * tl.rsqrt(tl.sum(k * k, 0) + 1.0e-6)
        a = tl.load(a_ptr + (token * HV + head) * DK + kk, mask=kk < DK, other=0).to(tl.float32)
        gate = tl.exp(LOWER_BOUND * tl.sigmoid(a_scale * (a + bias)))
        beta = tl.sigmoid(tl.load(b_ptr + token * HV + head).to(tl.float32))
        value = tl.load(v_ptr + (token * HV + head) * DV + vv, mask=vv < DV, other=0).to(tl.float32)
        state = state * gate[None, :]
        if BV == 128:
            # Group K before reduction to shrink the MLU reduction transposes.
            projection = tl.sum(tl.sum((state * k[None, :]).reshape((BV, 4, BK // 4)), 1), 1)
            delta = (value - projection) * beta
            state = (
                state.reshape((BV, 4, BK // 4)) + delta[:, None, None] * k.reshape((4, BK // 4))[None, :, :]
            ).reshape((BV, BK))
            output = tl.sum(tl.sum((state * q[None, :]).reshape((BV, 4, BK // 4)), 1), 1)
        else:
            # Small value tiles favor the original reduction and store order.
            delta = (value - tl.sum(state * k[None, :], 1)) * beta
            state = state + delta[:, None] * k[None, :]
            output = tl.sum(state * q[None, :], 1)
            tl.store(output_ptr + (token * HV + head) * DV + vv, output, mask=vv < DV)
        if INPLACE:
            final_slot = tl.load(state_indices_ptr + seq * STRIDE_INDICES_SEQ + (token - bos) * STRIDE_INDICES_TOK).to(
                tl.int64
            )
            if final_slot > 0:
                tl.store(final_state_ptr + final_slot * HV * DV * DK + state_offset, state, mask=mask)
        else:
            tl.store(final_state_ptr + token * HV * DV * DK + state_offset, state, mask=mask)
        if BV == 128:
            # Issue the large checkpoint write before the small output write.
            tl.store(output_ptr + (token * HV + head) * DV + vv, output, mask=vv < DV)


# Four-token KDA verification adapted from vllm_mlu kda_decode_chunk.py
# at commit 0029935c161e38d3979a67593bc5cacc1fffd939.
# Copyright (C) 2025-2026 Cambricon.


@triton.jit
def _token_decay(
    raw_gate: tl.tensor,
    gate_bias: tl.tensor,
    gate_scale: tl.tensor,
    vg_base: tl.tensor,
    rk: tl.tensor,
    tok: tl.constexpr,
    length: tl.int32,
    HV: tl.constexpr,
    K: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
) -> tl.tensor:
    g_t = tl.load(raw_gate + vg_base + tok * HV * K + rk, mask=tok < length, other=0).to(tl.float32)
    return tl.where(
        tok < length,
        tl.exp(LOWER_BOUND * tl.sigmoid(gate_scale * (g_t + gate_bias))),
        1.0,
    )


@triton.jit
def _snapshot_store(snap: tl.tensor, sidx: tl.tensor, snap_bp: tl.tensor, active: tl.tensor) -> None:
    if (sidx > 0) & (active != 0):
        tl.store(snap_bp, snap)


@triton.jit(do_not_specialize=["N", "TOTAL_TOKENS"])
def fused_chunk_kda_decode_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    raw_gate: tl.tensor,
    raw_beta: tl.tensor,
    output: tl.tensor,
    state: tl.tensor,
    cu_seqlens: tl.tensor,
    state_indices: tl.tensor,
    accepted_tokens: tl.tensor,
    a_log: tl.tensor,
    gate_bias: tl.tensor,
    scale: tl.float32,
    N: tl.int32,
    TOTAL_TOKENS: tl.int32,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    workers = tl.num_programs(0)
    value_tiles = tl.cdiv(V, BV)
    _rt4 = tl.arange(0, 4)
    eye = tl.where(_rt4[:, None] == _rt4[None, :], 1.0, 0.0)
    head_tile = pid % (HV * value_tiles)
    first_n = pid // (HV * value_tiles)
    seq_step = workers // (HV * value_tiles)
    i_v = head_tile % value_tiles
    i_hv = head_tile // value_tiles
    i_h = i_hv // (HV // H)
    rk = tl.arange(0, K)
    rt = tl.arange(0, 4)
    rv = i_v * BV + tl.arange(0, BV)
    gate_scale = tl.exp(tl.load(a_log + i_h, cache_modifier=".ca").to(tl.float32))
    g_bias = tl.load(gate_bias + i_h * K + rk, cache_modifier=".ca").to(tl.float32)
    bos = tl.load(cu_seqlens + first_n)
    eos = tl.load(cu_seqlens + first_n + 1)
    first_column = tl.load(accepted_tokens + first_n) - 1
    state_idx = tl.load(
        state_indices + first_n * stride_indices_seq + tl.minimum(tl.maximum(first_column, 0), 3) * stride_indices_tok
    )
    state_idx = tl.where((first_column >= 0) & (first_column < 4), state_idx, 0)
    for i_n in range(first_n, N, seq_step):
        next_n = tl.minimum(i_n + seq_step, N - 1)
        bos_next = tl.load(cu_seqlens + next_n)
        eos_next = tl.load(cu_seqlens + next_n + 1)
        next_column = tl.load(accepted_tokens + next_n) - 1
        state_idx_next = tl.load(
            state_indices + next_n * stride_indices_seq + tl.minimum(tl.maximum(next_column, 0), 3) * stride_indices_tok
        )
        state_idx_next = tl.where((next_column >= 0) & (next_column < 4), state_idx_next, 0)
        length = eos - bos
        active = ((state_idx > 0) & (length > 0)).to(tl.int32)
        state_slot = state_idx * active
        qk_base = (bos * H + i_h) * K
        vg_base = (bos * HV + i_hv) * V
        q_kt = tl.load(
            q + qk_base + rk[:, None] + rt[None, :] * (H * K),
            mask=rt[None, :] < length,
            other=0,
        ).to(tl.float32)
        k_kt = tl.load(
            k + qk_base + rk[:, None] + rt[None, :] * (H * K),
            mask=rt[None, :] < length,
            other=0,
        ).to(tl.float32)
        v_vt = tl.load(
            v + vg_base + rv[:, None] + rt[None, :] * (HV * V),
            mask=rt[None, :] < length,
            other=0,
        ).to(tl.float32)
        beta_t = tl.load(raw_beta + bos * HV + i_hv + rt * HV, mask=rt < length, other=0).to(tl.float32)
        q_inv = 1.0 / tl.sqrt(tl.sum(q_kt * q_kt, axis=0) + 1e-06)
        k_inv = 1.0 / tl.sqrt(tl.sum(k_kt * k_kt, axis=0) + 1e-06)
        qn = q_kt * (q_inv * scale)[None, :]
        kn = k_kt * k_inv[None, :]
        dec0 = _token_decay(raw_gate, g_bias, gate_scale, vg_base, rk, 0, length, HV, K, LOWER_BOUND)
        dec1 = _token_decay(raw_gate, g_bias, gate_scale, vg_base, rk, 1, length, HV, K, LOWER_BOUND)
        dec2 = _token_decay(raw_gate, g_bias, gate_scale, vg_base, rk, 2, length, HV, K, LOWER_BOUND)
        dec3 = _token_decay(raw_gate, g_bias, gate_scale, vg_base, rk, 3, length, HV, K, LOWER_BOUND)
        lam0 = dec0
        lam1 = lam0 * dec1
        lam2 = lam1 * dec2
        lam3 = lam2 * dec3
        lam = tl.where(
            rt[None, :] == 0,
            lam0[:, None],
            tl.where(
                rt[None, :] == 1,
                lam1[:, None],
                tl.where(rt[None, :] == 2, lam2[:, None], lam3[:, None]),
            ),
        )
        qs = lam * qn
        kb = lam * kn
        km = kn / lam
        state_load_bp = tl.make_block_ptr(
            base=state + state_slot * stride_state_slot + i_hv * V * K,
            shape=(V, K),
            strides=(K, 1),
            offsets=(i_v * BV, 0),
            block_shape=(BV, K),
            order=(1, 0),
        )
        recurrent = tl.load(state_load_bp).to(tl.float32)
        km4 = tl.trans(km)
        sidx_vec = tl.load(state_indices + i_n * stride_indices_seq + rt * stride_indices_tok)
        sid0 = tl.sum(tl.where(rt == 0, sidx_vec, 0), axis=0)
        sid1 = tl.sum(tl.where(rt == 1, sidx_vec, 0), axis=0)
        sid2 = tl.sum(tl.where(rt == 2, sidx_vec, 0), axis=0)
        sid3 = tl.sum(tl.where(rt == 3, sidx_vec, 0), axis=0)
        bp0 = tl.make_block_ptr(
            base=state + tl.where(sid0 > 0, sid0, 0) * stride_state_slot + i_hv * V * K,
            shape=(V, K),
            strides=(K, 1),
            offsets=(i_v * BV, 0),
            block_shape=(BV, K),
            order=(1, 0),
        )
        bp1 = tl.make_block_ptr(
            base=state + tl.where(sid1 > 0, sid1, 0) * stride_state_slot + i_hv * V * K,
            shape=(V, K),
            strides=(K, 1),
            offsets=(i_v * BV, 0),
            block_shape=(BV, K),
            order=(1, 0),
        )
        bp2 = tl.make_block_ptr(
            base=state + tl.where(sid2 > 0, sid2, 0) * stride_state_slot + i_hv * V * K,
            shape=(V, K),
            strides=(K, 1),
            offsets=(i_v * BV, 0),
            block_shape=(BV, K),
            order=(1, 0),
        )
        bp3 = tl.make_block_ptr(
            base=state + tl.where(sid3 > 0, sid3, 0) * stride_state_slot + i_hv * V * K,
            shape=(V, K),
            strides=(K, 1),
            offsets=(i_v * BV, 0),
            block_shape=(BV, K),
            order=(1, 0),
        )
        ut_qk_t = tl.dot(km4, qs, input_precision="tf32")
        ut_qk_t = tl.where(rt[None, :] >= rt[:, None], ut_qk_t, 0.0)
        ut_kk = tl.dot(tl.trans(kb), km, input_precision="ieee")
        beta_s = tl.where(rt < length, tl.sigmoid(beta_t), 0.0)
        lower = tl.where(rt[None, :] < rt[:, None], beta_s[:, None] * ut_kk, 0.0)
        lower2 = tl.dot(lower, lower, input_precision="ieee")
        lower3 = tl.dot(lower2, lower, input_precision="ieee")
        tri_inv = eye - lower + lower2 - lower3
        tri_inv_t = tl.trans(tri_inv)
        s0_kb = tl.dot(recurrent, kb, input_precision="ieee")
        s0_qs = tl.dot(recurrent, qs, input_precision="tf32")
        rhs = (v_vt - s0_kb) * beta_s[None, :]
        delta = tl.dot(rhs, tri_inv_t, input_precision="ieee")
        out_vt = s0_qs + tl.dot(delta, ut_qk_t, input_precision="tf32")
        # Guard the store uniformly: the MLU backend can drop a scalar
        # activity predicate broadcast only along the token mask dimension.
        if active != 0:
            tl.store(
                output + (bos * HV + i_hv) * V + rv[:, None] + rt[None, :] * (HV * V),
                out_vt.to(output.dtype.element_ty),
                mask=rt[None, :] < length,
            )
        d0 = tl.where(rt[None, :] <= 0, delta, 0.0)
        a0 = tl.dot(d0, km4, acc=recurrent, input_precision="ieee")
        _snapshot_store(lam0[None, :] * a0, sid0, bp0, active & (length > 0))
        d1 = tl.where(rt[None, :] <= 1, delta, 0.0)
        a1 = tl.dot(d1, km4, acc=recurrent, input_precision="ieee")
        _snapshot_store(lam1[None, :] * a1, sid1, bp1, active & (length > 1))
        d2 = tl.where(rt[None, :] <= 2, delta, 0.0)
        a2 = tl.dot(d2, km4, acc=recurrent, input_precision="ieee")
        _snapshot_store(lam2[None, :] * a2, sid2, bp2, active & (length > 2))
        a3 = tl.dot(delta, km4, acc=recurrent, input_precision="ieee")
        _snapshot_store(lam3[None, :] * a3, sid3, bp3, active & (length > 3))
        bos = bos_next
        eos = eos_next
        state_idx = state_idx_next
