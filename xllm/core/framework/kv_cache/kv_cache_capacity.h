/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

#include <cstdint>

#include "common/macros.h"
#include "framework/kv_cache/cache_layout_types.h"

namespace xllm {

class KVCacheCapacity final {
 public:
  KVCacheCapacity& index_cache_sizes(int64_t slot_size, int64_t block_size) {
    CHECK_GE(slot_size, 0) << "Index cache slot size must be non-negative.";
    CHECK_GE(block_size, 0) << "Index cache block size must be non-negative.";
    CHECK_EQ(slot_size == 0, block_size == 0)
        << "Index cache slot and block sizes must both be zero or positive.";
    index_slot_size_ = slot_size;
    index_block_size_ = block_size;
    return *this;
  }

  [[nodiscard]] int64_t index_slot_size() const noexcept {
    return index_slot_size_;
  }

  [[nodiscard]] int64_t index_block_size() const noexcept {
    return index_block_size_;
  }

  PROPERTY(int64_t, n_blocks) = 0;
  PROPERTY(int64_t, cache_size_in_bytes) = 0;
  PROPERTY(int64_t, block_size) = 0;
  PROPERTY(int64_t, slot_size) = 0;

  // Physical pages one logical block owns in this pool. 1 means the pool
  // follows the target's kv_split sharding (each rank owns one page per logical
  // block); kv_split means the pool is replicated, so every rank holds the
  // whole block and addresses it by the logical slot index. A standalone
  // drafter (DFlash/DFlash2) sets this to kv_split.
  PROPERTY(int64_t, replicated_block_pages) = 1;

  PROPERTY(KPoolCacheLayout, kpool_layout) = default_kpool_layout();
  PROPERTY(int64_t, kpool_tail_len) = 0;
  PROPERTY(int64_t, kpool_tail_slot_size) = 0;

  // for index cache
  // Rounded-up per-token diagnostic size; allocation uses exact block bytes.
  PROPERTY(int64_t, num_indexer_layers) = 0;
  PROPERTY(bool, enable_indexer_cache_quant) = false;

  // MLA SFA C8: KV cache is packed (int8 nope + bf16 rope + fp32 scale) in a
  // single byte tensor. Consumed by KVCacheShape to switch to the 656B layout.
  PROPERTY(bool, enable_mla_kv_cache_quant) = false;

  // for kv cache quantization scale cache
  PROPERTY(int64_t, scale_slot_size) = 0;

  // for linear attention
  PROPERTY(int64_t, linear_slot_size) = 0;
  PROPERTY(int64_t, linear_cache_size_in_bytes) = 0;
  PROPERTY(int64_t, linear_conv_state_len) = 0;
  PROPERTY(int64_t, linear_ssm_checkpoint_stride) = 1;
  PROPERTY(int64_t, n_layers) = 0;
  PROPERTY(int64_t, num_linear_state_blocks) = 0;
  PROPERTY(int64_t, num_full_attention_layers) = 0;
  PROPERTY(int64_t, num_linear_attention_layers) = 0;

  // DeepSeek V4 uses separate block pools for sliding-window and compressed
  // caches. These fields are only meaningful for deepseek_v4.
  PROPERTY(int64_t, swa_count) = 0;
  PROPERTY(int64_t, c4_count) = 0;
  PROPERTY(int64_t, c128_count) = 0;

  // DeepSeek V4.1 TOKEN-group pool counts (kv-source layers, compress ratios
  // 2 and 1). Only meaningful for deepseek_v41, whose positional pool vector
  // is [swa_count, c2_count, c1_count].
  PROPERTY(int64_t, c2_count) = 0;
  PROPERTY(int64_t, c1_count) = 0;

 private:
  int64_t index_slot_size_ = 0;
  int64_t index_block_size_ = 0;
};

}  // namespace xllm
