# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

import triton
import triton.language as tl
import triton.language.extra.mlu as mlu


@triton.jit(do_not_specialize=["softmax_scale", "block_table_width"])
def score_pools(
    query: tl.tensor,
    head_weights: tl.tensor,
    compressed_pool_cache: tl.tensor,
    block_table: tl.tensor,
    positions: tl.tensor,
    rows: tl.tensor,
    pool_scores: tl.tensor,
    query_row_stride: tl.int64,
    query_head_stride: tl.int64,
    query_dim_stride: tl.int64,
    weight_row_stride: tl.int64,
    weight_head_stride: tl.int64,
    cache_block_stride: tl.int64,
    cache_head_stride: tl.int64,
    cache_pool_stride: tl.int64,
    cache_dim_stride: tl.int64,
    block_table_row_stride: tl.int64,
    block_table_column_stride: tl.int64,
    score_row_stride: tl.int64,
    score_column_stride: tl.int64,
    softmax_scale: tl.float32,
    capacity: tl.int64,
    cache_blocks: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    POOL_BLOCK_SIZE: tl.constexpr,
    block_table_width: tl.int64,
    INDEX_KPOOL: tl.constexpr,
    BLOCK_POOLS: tl.constexpr,
    NUM_HEAD_PAD: tl.constexpr,
    DENSE_BLOCK: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    CHUNK: tl.constexpr,
) -> None:
    """Score pool tiles directly from the paged compressed cache.

    Each program owns TILES_PER_PROG consecutive pool tiles of one query.
    With DENSE_BLOCK set (cache layout [blocks, 1, POOL_BLOCK_SIZE, HEAD_DIM]
    whose last two dims are contiguous and POOL_BLOCK_SIZE * HEAD_DIM is a
    power of two), one pool block is loaded as a single contiguous row and
    CHUNK tiles are fused into one tensor-engine dot per iteration, so the
    paged-cache read lowers to a row gather. The fallback path keeps the
    original per-tile, fully strided addressing.
    """
    query_id = tl.program_id(0)
    pool_capacity = capacity
    request = tl.load(rows + query_id).to(tl.int64)

    kv_seq_len = tl.load(positions + query_id).to(tl.int64) + 1
    live_pool_count = tl.minimum(
        tl.maximum(kv_seq_len, 0) // INDEX_KPOOL,
        tl.minimum(pool_capacity, block_table_width * POOL_BLOCK_SIZE),
    )

    dim_offsets = tl.arange(0, HEAD_DIM)
    head_offsets = tl.arange(0, NUM_HEAD_PAD)
    head_mask = head_offsets < NUM_HEADS
    query_offsets = (
        query_id * query_row_stride
        + head_offsets[:, None] * query_head_stride
        + dim_offsets[None, :] * query_dim_stride
    )
    query_values = tl.load(
        query + query_offsets,
        mask=head_mask[:, None],
        other=0.0,
    )
    weights = tl.load(
        head_weights + query_id * weight_row_stride + head_offsets * weight_head_stride,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    query_t = tl.trans(query_values)

    if DENSE_BLOCK:
        blocks_per_tile: tl.constexpr = BLOCK_POOLS // POOL_BLOCK_SIZE
        pools_per_iter: tl.constexpr = BLOCK_POOLS * CHUNK
        blocks_per_iter: tl.constexpr = blocks_per_tile * CHUNK
        flat_offsets = tl.arange(0, POOL_BLOCK_SIZE * HEAD_DIM)
        for it in range(0, TILES_PER_PROG // CHUNK):
            first_tile = tl.program_id(1) * TILES_PER_PROG + it * CHUNK
            pool_ids = first_tile * BLOCK_POOLS + tl.arange(0, pools_per_iter)
            pool_in_capacity = pool_ids < pool_capacity
            pool_is_live = pool_in_capacity & (pool_ids < live_pool_count)
            scores = tl.full((pools_per_iter,), float("-inf"), dtype=tl.float32)
            if first_tile * BLOCK_POOLS < live_pool_count:
                block_ids = first_tile * blocks_per_tile + tl.arange(0, blocks_per_iter)
                block_live = block_ids < ((live_pool_count + POOL_BLOCK_SIZE - 1) // POOL_BLOCK_SIZE)
                block_live = block_live & (block_ids < block_table_width)
                physical_blocks = tl.load(
                    block_table + request * block_table_row_stride + block_ids * block_table_column_stride,
                    mask=block_live,
                    other=-1,
                ).to(tl.int64)
                block_live &= (physical_blocks >= 0) & (physical_blocks < cache_blocks)
                pool_is_live &= tl.broadcast_to(block_live[:, None], (blocks_per_iter, POOL_BLOCK_SIZE)).reshape(
                    pools_per_iter
                )
                cache_offsets = physical_blocks[:, None] * cache_block_stride + flat_offsets[None, :]
                keys_flat = tl.load(
                    compressed_pool_cache + cache_offsets,
                    mask=block_live[:, None],
                    other=0.0,
                )
                pooled_keys = tl.reshape(
                    keys_flat,
                    (pools_per_iter, HEAD_DIM),
                )
                per_pool = tl.dot(
                    pooled_keys,
                    query_t,
                    allow_tf32=False,
                ).to(tl.float32)
                per_pool = tl.maximum(per_pool * softmax_scale, 0.0)
                live_scores = tl.sum(per_pool * weights[None, :], axis=1)
                scores = tl.where(pool_is_live, live_scores, float("-inf"))

            tl.store(
                pool_scores + query_id * score_row_stride + pool_ids * score_column_stride,
                scores,
                mask=pool_in_capacity,
            )
    else:
        for tile_id in range(0, TILES_PER_PROG):
            tile = tl.program_id(1) * TILES_PER_PROG + tile_id
            pool_ids = tile * BLOCK_POOLS + tl.arange(0, BLOCK_POOLS)
            pool_in_capacity = pool_ids < pool_capacity
            pool_is_live = pool_in_capacity & (pool_ids < live_pool_count)
            scores = tl.full((BLOCK_POOLS,), float("-inf"), dtype=tl.float32)
            if tile * BLOCK_POOLS < live_pool_count:
                logical_blocks = pool_ids // POOL_BLOCK_SIZE
                physical_blocks = tl.load(
                    block_table + request * block_table_row_stride + logical_blocks * block_table_column_stride,
                    mask=pool_is_live,
                    other=-1,
                ).to(tl.int64)
                pool_is_live &= (physical_blocks >= 0) & (physical_blocks < cache_blocks)
                pool_offsets = pool_ids % POOL_BLOCK_SIZE
                cache_offsets = (
                    physical_blocks[:, None] * cache_block_stride
                    + 0 * cache_head_stride
                    + pool_offsets[:, None] * cache_pool_stride
                    + dim_offsets[None, :] * cache_dim_stride
                )
                pooled_keys = tl.load(
                    compressed_pool_cache + cache_offsets,
                    mask=pool_is_live[:, None],
                    other=0.0,
                )
                per_pool = tl.dot(
                    pooled_keys,
                    query_t,
                    allow_tf32=False,
                ).to(tl.float32)
                per_pool = tl.maximum(per_pool * softmax_scale, 0.0)
                live_scores = tl.sum(per_pool * weights[None, :], axis=1)
                scores = tl.where(pool_is_live, live_scores, float("-inf"))

            tl.store(
                pool_scores + query_id * score_row_stride + pool_ids * score_column_stride,
                scores,
                mask=pool_in_capacity,
            )
