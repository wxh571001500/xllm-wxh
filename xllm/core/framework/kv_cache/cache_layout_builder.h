/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "platform/platform.h"
#include "util/utils.h"

namespace xllm {

// Records each cache pool's physical token-axis span from the tensors that
// carry a token axis, so later describe and manifest steps validate a tensor
// against its own pool geometry instead of the uniform scheduler block size.
void record_group_block_capacities(const std::vector<KVCache>& kv_caches,
                                   CacheTensorLayoutContext* tensor_layout);

// Returns the physical token-axis span a pool's resources cover: the pool's
// recorded span when present, block_token_capacity otherwise.
int64_t effective_block_token_capacity(const CacheTensorLayoutContext& context,
                                       int32_t group_id);

// Adds an explicit logical descriptor to a cache tensor produced by KVCache.
// Returns false with an actionable error when the tensor cannot be represented
// without guessing an unsupported layout.
bool describe_cache_tensor(const CacheTensorLayoutContext& context,
                           KVCacheTensor* cache_tensor,
                           std::string* error);

// Physical index pages owned by one logical cache block. The NPU keeps the
// whole DSA indexer cache on every rank, so a logical block B maps to index
// pages [B * split, (B + 1) * split) (see platform.h). Every other backend
// allocates a single index page per block. This is the allocation predicate of
// KVCacheShape::init_index_cache_shape() and the transfer geometry of
// describe_cache_tensor(), so both consume this one definition. Defined inline
// here on purpose: kv_cache_estimation.cpp lives in :kv_cache_estimation, which
// :kv_cache already depends on, so a .cpp definition in this library would make
// the dependency a cycle. Both callees below are header-only.
inline int64_t indexer_pages_per_block() {
  if (!Platform::requires_dsa_indexer_cache_replication()) {
    return 1;
  }
  return std::max<int32_t>(util::kv_split_size_effective(), 1);
}

// Physical tensor rows one logical cache block spans for `role`, the row-count
// companion of the logical descriptor describe_cache_tensor() writes for the
// same role: SSM packs its checkpoint stride into one block, the NPU indexer
// cache covers every kv_split page, and a replicated pool (a standalone
// drafter) covers the whole logical block. Every other role stores one row per
// logical block.
int64_t physical_rows_per_resource(KVCacheTensorRole role,
                                   int64_t replicated_block_pages,
                                   int64_t ssm_checkpoint_stride);

}  // namespace xllm
