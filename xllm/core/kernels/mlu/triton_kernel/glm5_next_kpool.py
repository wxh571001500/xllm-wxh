# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Request-local fused decode update and bounded prefill gather/write helpers.
import triton
import triton.language as tl


@triton.jit
def _compress(
    k: tl.tensor,
    gate: tl.tensor,
    ape: tl.tensor,
    hadamard: tl.tensor,
    tail: tl.tensor,
    positions: tl.tensor,
    starts: tl.tensor,
    tail_ids: tl.tensor,
    rows: tl.tensor,
    endpoints: tl.tensor,
    valid: tl.tensor,
    k_stride: tl.tensor,
    gate_stride: tl.tensor,
    P: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BP: tl.constexpr,
    BD: tl.constexpr,
) -> tl.tensor:
    pos = tl.load(positions + endpoints, valid, -1)
    start = tl.load(starts + rows, valid, 0)
    state = tl.load(tail_ids + rows, valid, 0)
    m = tl.arange(0, BP)
    d = tl.arange(0, BD)
    source = endpoints[:, None] - P + 1 + m[None, :]
    source_pos = tl.load(
        positions + source,
        valid[:, None] & (m[None, :] < P) & (source >= start[:, None]),
        -1,
    )
    current = (source >= start[:, None]) & (source_pos >= 0)
    mask = valid[:, None, None] & (m[None, :, None] < P) & (d[None, None, :] < D)
    raw = tl.load(
        k + source[:, :, None] * k_stride + d[None, None, :],
        mask & current[:, :, None],
        0,
    ).to(tl.float32)
    g = tl.load(
        gate + source[:, :, None] * gate_stride + d[None, None, :],
        mask & current[:, :, None],
        0,
    ).to(tl.float32)
    ring = (pos[:, None] - P + 1 + m[None, :]) % T
    base = (state[:, None] * 2 * T + ring) * D
    old = tl.load(tail + base[:, :, None] + d[None, None, :], mask & ~current[:, :, None], 0).to(tl.float32)
    old_g = tl.load(
        tail + base[:, :, None] + T * D + d[None, None, :],
        mask & ~current[:, :, None],
        0,
    ).to(tl.float32)
    a = tl.load(ape + m[:, None] * D + d[None, :], (m[:, None] < P) & (d[None, :] < D), 0)
    logits = tl.where(
        m[None, :, None] < P,
        tl.where(current[:, :, None], g, old_g) + a[None, :, :],
        -float("inf"),
    )
    prob = tl.exp2((logits - tl.max(logits, 1)[:, None, :]) * 1.4426950408889634)
    prob /= tl.sum(prob, 1)[:, None, :]
    pooled = tl.sum(tl.where(current[:, :, None], raw, old) * prob, 1)
    pooled = pooled.to(tl.bfloat16).to(tl.float32)
    h = tl.load(hadamard + d[:, None] + d[None, :] * D, (d[:, None] < D) & (d[None, :] < D), 0).to(tl.float32)
    return tl.sum(h * pooled.reshape(BD, 1), 0).reshape(1, BD)


@triton.jit
def _compress_decode_four(
    k: tl.tensor,
    gate: tl.tensor,
    ape: tl.tensor,
    hadamard: tl.tensor,
    tail: tl.tensor,
    positions: tl.tensor,
    starts: tl.tensor,
    tail_ids: tl.tensor,
    rows: tl.tensor,
    endpoints: tl.tensor,
    valid: tl.tensor,
    k_stride: tl.tensor,
    gate_stride: tl.tensor,
    P: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BP: tl.constexpr,
    BD: tl.constexpr,
) -> tl.tensor:
    """Pool four members without a cross-member vector reduction."""
    pos = tl.load(positions + endpoints, valid, -1)
    start = tl.load(starts + rows, valid, 0)
    state = tl.load(tail_ids + rows, valid, 0)
    d = tl.arange(0, BD)
    maximum = tl.full((1, BD), -float("inf"), tl.float32)

    for slot in tl.static_range(0, P):
        source = endpoints - P + 1 + slot
        source_pos = tl.load(positions + source, valid & (source >= start), -1)
        current = (source >= start) & (source_pos >= 0)
        mask = valid[:, None] & (d[None, :] < D)
        ring = (pos - P + 1 + slot) % T
        base = (state * 2 * T + ring) * D
        new_gate = tl.load(
            gate + source[:, None] * gate_stride + d[None, :],
            mask & current[:, None],
            0,
        ).to(tl.float32)
        old_gate = tl.load(
            tail + base[:, None] + T * D + d[None, :],
            mask & ~current[:, None],
            0,
        ).to(tl.float32)
        bias = tl.load(ape + slot * D + d, d < D, 0)
        logit = tl.where(current[:, None], new_gate, old_gate) + bias[None, :]
        maximum = tl.maximum(maximum, logit)

    numerator = tl.full((1, BD), 0, tl.float32)
    denominator = tl.full((1, BD), 0, tl.float32)
    for slot in tl.static_range(0, P):
        source = endpoints - P + 1 + slot
        source_pos = tl.load(positions + source, valid & (source >= start), -1)
        current = (source >= start) & (source_pos >= 0)
        mask = valid[:, None] & (d[None, :] < D)
        ring = (pos - P + 1 + slot) % T
        base = (state * 2 * T + ring) * D
        new_gate = tl.load(
            gate + source[:, None] * gate_stride + d[None, :],
            mask & current[:, None],
            0,
        ).to(tl.float32)
        old_gate = tl.load(
            tail + base[:, None] + T * D + d[None, :],
            mask & ~current[:, None],
            0,
        ).to(tl.float32)
        bias = tl.load(ape + slot * D + d, d < D, 0)
        probability = tl.exp2(
            (tl.where(current[:, None], new_gate, old_gate) + bias[None, :] - maximum) * 1.4426950408889634
        )
        new_key = tl.load(
            k + source[:, None] * k_stride + d[None, :],
            mask & current[:, None],
            0,
        ).to(tl.float32)
        old_key = tl.load(
            tail + base[:, None] + d[None, :],
            mask & ~current[:, None],
            0,
        ).to(tl.float32)
        numerator += tl.where(current[:, None], new_key, old_key) * probability
        denominator += probability

    pooled = (numerator / denominator).to(tl.bfloat16).to(tl.float32)
    h = tl.load(
        hadamard + d[:, None] + d[None, :] * D,
        (d[:, None] < D) & (d[None, :] < D),
        0,
    ).to(tl.float32)
    return tl.sum(h * pooled.reshape(BD, 1), 0).reshape(1, BD)


@triton.jit
def gather_members(
    k: tl.tensor,
    gate: tl.tensor,
    tail: tl.tensor,
    tail_ids: tl.tensor,
    table: tl.tensor,
    positions: tl.tensor,
    rows: tl.tensor,
    starts: tl.tensor,
    keys: tl.tensor,
    gates: tl.tensor,
    slots: tl.tensor,
    offset: tl.int64,
    tokens: tl.int64,
    table_width: tl.int64,
    cache_blocks: tl.int64,
    states: tl.int64,
    P: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BP: tl.constexpr,
    BD: tl.constexpr,
    PB: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    endpoint = offset + row
    # Runtime token positions are int32; keep scalar sentinel predicates signed.
    pos = tl.load(positions + endpoint).to(tl.int32)
    req = tl.load(rows + endpoint).to(tl.int32)
    state = tl.load(tail_ids + req).to(tl.int32)
    start = tl.load(starts + req).to(tl.int32)
    pool = pos // P
    # State slot zero is reserved for graph-padding requests in xLLM.
    valid = (pos >= 0) & (pos % P == P - 1) & (state > 0) & (state < states)
    valid &= pool // PB < table_width
    page = tl.load(
        table + req * table_width + tl.maximum(0, tl.minimum(pool // PB, table_width - 1)),
        valid,
        -1,
    )
    valid &= (page >= 0) & (page < cache_blocks)
    if P <= 16:
        m = tl.arange(0, BP)
    else:
        m = tl.program_id(1) * BP + tl.arange(0, BP)
    d = tl.arange(0, BD)
    source = endpoint - P + 1 + m
    safe_source = tl.maximum(0, tl.minimum(source, tokens - 1))
    source_pos = tl.load(positions + safe_source, valid & (m < P) & (source >= start), -1)
    current = (source >= start) & (source_pos >= 0)
    mask = valid & (m[:, None] < P) & (d[None, :] < D)
    ring = (pos - P + 1 + m) % T
    base = (tl.maximum(0, tl.minimum(state, states - 1)) * 2 * T + tl.maximum(ring[:, None], 0)) * D + d[None, :]
    old_k = tl.load(tail + base, mask & ~current[:, None], 0)
    old_g = tl.load(tail + base + T * D, mask & ~current[:, None], 0)
    new_k = tl.load(k + safe_source[:, None] * D + d[None, :], mask & current[:, None], 0)
    new_g = tl.load(gate + safe_source[:, None] * D + d[None, :], mask & current[:, None], 0)
    out = (row * P + m[:, None]) * D + d[None, :]
    tl.store(
        keys + out,
        tl.where(current[:, None], new_k, old_k),
        (m[:, None] < P) & (d[None, :] < D),
    )
    tl.store(
        gates + out,
        tl.where(current[:, None], new_g, old_g),
        (m[:, None] < P) & (d[None, :] < D),
    )
    if P <= 16 or tl.program_id(1) == 0:
        tl.store(slots + row, tl.where(valid, page * PB + pool % PB, -1))


@triton.jit
def write_pools(
    values: tl.tensor,
    slots: tl.tensor,
    cache: tl.tensor,
    D: tl.constexpr,
    BD: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    slot = tl.load(slots + row)
    d = tl.arange(0, BD)
    value = tl.load(values + row * D + d, d < D, 0)
    tl.store(cache + slot * D + d, value, (slot >= 0) & (d < D))


@triton.jit
def kpool_decode_update(
    k: tl.tensor,
    gate: tl.tensor,
    ape: tl.tensor,
    hadamard: tl.tensor,
    cache: tl.tensor,
    tail: tl.tensor,
    tail_ids: tl.tensor,
    table: tl.tensor,
    positions: tl.tensor,
    starts: tl.tensor,
    k_stride: tl.int64,
    gate_stride: tl.int64,
    table_stride: tl.int64,
    P: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BP: tl.constexpr,
    BD: tl.constexpr,
    PB: tl.constexpr,
) -> None:
    req = tl.program_id(0)
    start = tl.load(starts + req)
    end = tl.load(starts + req + 1)
    state = tl.load(tail_ids + req)
    d = tl.arange(0, BD)
    for i in range(start, end):
        pos = tl.load(positions + i)
        if (pos >= 0) & (state > 0):
            if pos % P == P - 1:
                if P == 4 and D == 128:
                    values = _compress_decode_four(
                        k,
                        gate,
                        ape,
                        hadamard,
                        tail,
                        positions,
                        starts,
                        tail_ids,
                        req[None],
                        i[None],
                        tl.full((1,), True, tl.int1),
                        k_stride,
                        gate_stride,
                        P,
                        T,
                        D,
                        BP,
                        BD,
                    )
                else:
                    values = _compress(
                        k,
                        gate,
                        ape,
                        hadamard,
                        tail,
                        positions,
                        starts,
                        tail_ids,
                        req[None],
                        i[None],
                        tl.full((1,), True, tl.int1),
                        k_stride,
                        gate_stride,
                        P,
                        T,
                        D,
                        BP,
                        BD,
                    )
                pool = pos // P
                block = tl.load(table + req * table_stride + pool // PB)
                slot = block * PB + pool % PB
                tl.store(cache + slot * D + d, values.reshape(BD), (d < D) & (block >= 0))
            # Complete from old ring/raw batch before overwriting any ring slot.
            kv = tl.load(k + i * k_stride + d, d < D, 0)
            gv = tl.load(gate + i * gate_stride + d, d < D, 0)
            base = (state * 2 * T + pos % T) * D
            tl.store(tail + base + d, kv, d < D)
            tl.store(tail + base + T * D + d, gv, d < D)


@triton.jit
def kpool_stash(
    k: tl.tensor,
    gate: tl.tensor,
    tail: tl.tensor,
    tail_ids: tl.tensor,
    positions: tl.tensor,
    starts: tl.tensor,
    k_stride: tl.int64,
    gate_stride: tl.int64,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Only the last writer of each circular slot survives a batch update."""
    batch = tl.program_id(0)
    start = tl.load(starts + batch)
    end = tl.load(starts + batch + 1)
    # Draft placeholders and ragged padding may occupy the final rows.
    last = tl.load(positions + end - 1, end > start, -1).to(tl.int64)
    while (end > start) & (last < 0):
        end -= 1
        last = tl.load(positions + end - 1, end > start, -1).to(tl.int64)
    if end > start:
        tail_id = tl.load(tail_ids + batch)
        if (last >= 0) & (tail_id > 0):
            d = tl.arange(0, BLOCK_D)
            r = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
            source = end - 1 - (last % T - r + T) % T
            source_pos = tl.load(positions + source, (r < T) & (source >= start), -1)
            mask = (r[:, None] < T) & (d[None, :] < D) & (source_pos[:, None] >= 0)
            kv = tl.load(k + source[:, None] * k_stride + d[None, :], mask, 0)
            gv = tl.load(gate + source[:, None] * gate_stride + d[None, :], mask, 0)
            tl.store(tail + (tail_id * 2 * T + r[:, None]) * D + d[None, :], kv, mask)
            tl.store(tail + (tail_id * 2 * T + T + r[:, None]) * D + d[None, :], gv, mask)


@triton.jit
def kpool_prefill_complete(
    k: tl.tensor,
    gate: tl.tensor,
    ape: tl.tensor,
    hadamard: tl.tensor,
    cache: tl.tensor,
    tail: tl.tensor,
    tail_ids: tl.tensor,
    table: tl.tensor,
    positions: tl.tensor,
    row_batch: tl.tensor,
    starts: tl.tensor,
    k_stride: tl.int64,
    gate_stride: tl.int64,
    table_stride: tl.int64,
    P: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    POOL_BLOCK: tl.constexpr,
) -> None:
    """Complete prefill pools before any request tail is overwritten."""
    i = tl.program_id(0)
    pos = tl.load(positions + i)
    batch = tl.load(row_batch + i)
    tail_id = tl.load(tail_ids + batch)
    if (tail_id > 0) & (pos >= 0) & (pos % P == P - 1):
        start = tl.load(starts + batch)
        r = tl.arange(0, BLOCK_P)
        d = tl.arange(0, BLOCK_D)
        source = i - P + 1 + r
        current = source >= start
        source_pos = tl.load(positions + source, (r < P) & current, -1)
        current &= source_pos >= 0
        raw_k = tl.load(
            k + source[:, None] * k_stride + d[None, :],
            (r[:, None] < P) & (d[None, :] < D) & current[:, None],
            0,
        ).to(tl.float32)
        raw_g = tl.load(
            gate + source[:, None] * gate_stride + d[None, :],
            (r[:, None] < P) & (d[None, :] < D) & current[:, None],
            0,
        ).to(tl.float32)
        tail_r = (pos - P + 1 + r) % T
        old_k = tl.load(
            tail + (tail_id * 2 * T + tail_r[:, None]) * D + d[None, :],
            (r[:, None] < P) & (d[None, :] < D) & ~current[:, None],
            0,
        ).to(tl.float32)
        old_g = tl.load(
            tail + (tail_id * 2 * T + T + tail_r[:, None]) * D + d[None, :],
            (r[:, None] < P) & (d[None, :] < D) & ~current[:, None],
            0,
        ).to(tl.float32)
        a = tl.load(
            ape + r[:, None] * D + d[None, :],
            (r[:, None] < P) & (d[None, :] < D),
            0,
            cache_modifier=".ca",
        )
        logits = tl.where(
            r[:, None] < P,
            tl.where(current[:, None], raw_g, old_g) + a,
            -float("inf"),
        )
        probability = tl.exp(logits - tl.max(logits, 0)[None, :])
        probability /= tl.sum(probability, 0)[None, :]
        pooled = tl.sum(tl.where(current[:, None], raw_k, old_k) * probability, 0)
        pooled = pooled.to(tl.bfloat16).to(tl.float32)
        matrix = tl.load(
            hadamard + d[:, None] * D + d[None, :],
            (d[:, None] < D) & (d[None, :] < D),
            0,
            cache_modifier=".ca",
        )
        rotated = tl.sum(matrix * pooled[None, :], 1)
        pool = pos // P
        block = tl.load(table + batch * table_stride + pool // POOL_BLOCK)
        slot = block * POOL_BLOCK + pool % POOL_BLOCK
        tl.store(cache + slot * D + d, rotated, (block >= 0) & (d < D))


@triton.jit
def kpool_prefill_stash(
    k: tl.tensor,
    gate: tl.tensor,
    tail: tl.tensor,
    tail_ids: tl.tensor,
    positions: tl.tensor,
    starts: tl.tensor,
    k_stride: tl.int64,
    gate_stride: tl.int64,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Stash the final valid prefill tail for each request."""
    batch = tl.program_id(0)
    start = tl.load(starts + batch)
    end = tl.load(starts + batch + 1)
    last = tl.load(positions + end - 1, end > start, -1).to(tl.int64)
    while (end > start) & (last < 0):
        end -= 1
        last = tl.load(positions + end - 1, end > start, -1).to(tl.int64)
    if end > start:
        tail_id = tl.load(tail_ids + batch)
        if (last >= 0) & (tail_id > 0):
            d = tl.arange(0, BLOCK_D)
            r = tl.arange(0, BLOCK_P)
            source = end - 1 - (last % T - r + T) % T
            source_pos = tl.load(positions + source, (r < T) & (source >= start), -1)
            mask = (r[:, None] < T) & (d[None, :] < D) & (source_pos[:, None] >= 0)
            kv = tl.load(k + source[:, None] * k_stride + d[None, :], mask, 0)
            gv = tl.load(gate + source[:, None] * gate_stride + d[None, :], mask, 0)
            tl.store(tail + (tail_id * 2 * T + r[:, None]) * D + d[None, :], kv, mask)
            tl.store(tail + (tail_id * 2 * T + T + r[:, None]) * D + d[None, :], gv, mask)


@triton.jit
def kpool_rows(
    starts: tl.tensor,
    rows: tl.tensor,
    tokens: tl.int64,
    N: tl.constexpr,
    BN: tl.constexpr,
    BT: tl.constexpr,
) -> None:
    token = tl.program_id(0) * BT + tl.arange(0, BT)
    req = tl.arange(0, BN)
    begin = tl.load(starts + req, req < N, 0x7FFFFFFFFFFFFFFF)
    row = tl.sum((token[:, None] >= begin[None, :]).to(tl.int32), 1) - 1
    tl.store(rows + token, row, token < tokens)


# Four-query KPool verification adapted from vLLM-MLU kpool_logits.py
# at commit 0029935c161e38d3979a67593bc5cacc1fffd939.


@triton.jit(do_not_specialize=["scale", "table_width"])
def score_verify_pools(
    query: tl.tensor,
    weights: tl.tensor,
    cache: tl.tensor,
    table: tl.tensor,
    positions: tl.tensor,
    rows: tl.tensor,
    query_starts: tl.tensor,
    output: tl.tensor,
    tokens: tl.int64,
    capacity: tl.int64,
    cache_blocks: tl.int64,
    table_width: tl.int64,
    output_stride: tl.int64,
    scale: tl.float32,
    BLOCK_POOLS: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
) -> None:
    """Score a request's at-most-four live queries against each shared key tile.

    The caller selects this path for T4 verify batches; replay may shorten or
    mask individual requests. Q/cache are contiguous BF16, weights/output FP32,
    with H=32, D=128 and four compressed pools per physical cache page.
    """
    request_id = tl.program_id(0)
    begin = tl.load(query_starts + request_id).to(tl.int64)
    end = tl.load(query_starts + request_id + 1).to(tl.int64)
    t = tl.arange(0, 4)
    h = tl.arange(0, 32)
    d = tl.arange(0, 128)
    query_ids = begin + t
    query_valid = (query_ids < end) & (query_ids < tokens)
    request = tl.load(rows + begin, (begin < end) & (begin < tokens), 0).to(tl.int64)
    request_valid = (request >= 0) & (request < tl.num_programs(0))
    safe_request = tl.where(request_valid, request, 0)
    lengths = tl.load(positions + query_ids, query_valid, -1).to(tl.int64) + 1
    live_pools = tl.minimum(tl.maximum(lengths, 0) // 4, tl.minimum(capacity, table_width * 4))
    query_values = tl.load(
        query + query_ids[None, :, None] * 4096 + h[:, None, None] * 128 + d[None, None, :],
        query_valid[None, :, None],
        0.0,
    ).reshape(128, 128)
    head_weights = tl.load(
        weights + query_ids[None, :] * 32 + h[:, None],
        query_valid[None, :],
        0.0,
    ).to(tl.float32)
    first_tile = tl.program_id(1) * TILES_PER_PROGRAM
    for tile in range(first_tile, first_tile + TILES_PER_PROGRAM):
        pools = tile * BLOCK_POOLS + tl.arange(0, BLOCK_POOLS)
        logical_pages = tile * (BLOCK_POOLS // 4) + tl.arange(0, BLOCK_POOLS // 4)
        page_live = (logical_pages < table_width) & request_valid
        pages = tl.load(
            table + safe_request * table_width + logical_pages,
            page_live,
            -1,
        ).to(tl.int64)
        page_live &= (pages >= 0) & (pages < cache_blocks)
        safe_pages = tl.where(page_live, pages, 0)
        # Invalid pages read the safe page and are discarded below. A scalar
        # mask preserves contiguous 512-element page loads; a per-page mask
        # lowers to individual element gathers with large NRAM index buffers.
        keys = tl.load(
            cache + safe_pages[:, None] * 512 + tl.arange(0, 512)[None, :],
            cache_blocks > 0,
            0.0,
            cache_modifier=".cg",
        ).reshape(BLOCK_POOLS, 128)
        # Head-major query rows produce [H,T,P] directly, avoiding a full
        # score transpose before the FP32 head reduction. Scaling remains
        # before ReLU, and weights are never narrowed or pre-scaled.
        dots = tl.dot(query_values, tl.trans(keys), allow_tf32=False).to(tl.float32)
        dots = dots.reshape(32, 4, BLOCK_POOLS)
        weighted = tl.maximum(dots * scale, 0.0) * head_weights[:, :, None]
        scores = tl.sum(weighted, 0)
        pool_valid = tl.broadcast_to(page_live[:, None], (BLOCK_POOLS // 4, 4)).reshape(BLOCK_POOLS)
        valid = pool_valid[None, :] & (pools[None, :] < live_pools[:, None])
        tl.store(
            output + query_ids[:, None] * output_stride + pools[None, :],
            tl.where(valid, scores, -float("inf")),
            query_valid[:, None] & (pools[None, :] < capacity),
        )
        # Graph padding beyond the last live offset must not retain logits
        # from the previous replay. The last request owns this disjoint suffix.
        if request_id == tl.num_programs(0) - 1:
            for padding_row in range(end, tokens):
                tl.store(
                    output + padding_row * output_stride + pools,
                    -float("inf"),
                    pools < capacity,
                )
