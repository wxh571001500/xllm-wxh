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

#include <torch/torch.h>

#include <cstdint>
#include <limits>

namespace xllm::kernel {

struct DcpScorePolicy {
  int64_t row_cap;
  bool row_owned;
  bool small_rows;
  bool preload_counts;
  bool prefetch_rows;
  bool program_proof;
  bool narrow;
};

// Keep policy construction shared by the production launcher and kernel tests.
// The caller handles empty inputs before constructing the policy.
inline DcpScorePolicy dcp_score_policy(const torch::Tensor& query,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& cache,
                                       const torch::Tensor& slots,
                                       const torch::Tensor& counts,
                                       int64_t cores,
                                       int64_t block_n,
                                       int64_t slot_cap) {
  const int64_t rows = slots.size(0);
  const int64_t width = slots.size(1);
  const int64_t cache_slots = cache.size(0);
  int64_t row_cap = 0;
  if (rows <= 16384) {
    row_cap = 1;
    while (row_cap < rows) {
      row_cap *= 2;
    }
  }
  const int64_t bound = row_cap == 0 ? rows : row_cap;
  // Widen before multiplication: valid int64 strides can overflow int64 byte
  // bounds even when no corresponding large allocation is made (zero strides).
  const auto wide = [](int64_t value) { return static_cast<__int128>(value); };
  constexpr int64_t kLimit = std::numeric_limits<int32_t>::max();
  const bool narrow =
      cache_slots <= kLimit &&
      (wide(bound - 1) * query.stride(0) + wide(31) * query.stride(1) +
       wide(127) * query.stride(2)) *
              4 <
          kLimit &&
      (wide(bound - 1) * weights.stride(0) + wide(31) * weights.stride(1)) * 4 <
          kLimit &&
      (wide(cache_slots - 1) * cache.stride(0) + wide(127) * cache.stride(1)) *
              4 <
          kLimit &&
      (wide(bound - 1) * slots.stride(0) +
       (wide(slot_cap) + block_n - 1) * slots.stride(1)) *
              4 <
          kLimit &&
      wide(bound - 1) * counts.stride(0) * 4 < kLimit &&
      (wide(bound) * width + block_n) * 4 < kLimit;
  const bool row_owned = rows >= cores;
  return {row_cap,
          row_owned,
          rows <= 128,
          row_owned && rows > 128 && rows <= 16384,
          row_owned && rows > cores && rows <= 128,
          rows >= 2 * cores && rows <= 128 && width <= 4096,
          narrow};
}

}  // namespace xllm::kernel
