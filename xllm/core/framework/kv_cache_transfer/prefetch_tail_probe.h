/* Copyright 2026 The xLLM Authors.

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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "framework/kv_cache_transfer/prefetch_result.h"

namespace xllm {

// Gated blocks beyond a worker's contiguous hit prefix, checked for existence
// after the stream closes. This includes units read in the last batch but not
// usable as a prefix after a gated miss. Presence is per rank, not a global
// measure of reusable KV or recomputed tokens.
class PrefetchTailProbe final {
 public:
  // First unit of the tail after a batch starting at unit_begin: the first
  // gated miss, or the unit after the batch when every unit hit. Only the
  // first unit_count entries of gated_hits are real units.
  static size_t tail_begin(size_t unit_begin,
                           const std::vector<uint8_t>& gated_hits,
                           size_t unit_count) {
    CHECK_LE(unit_count, gated_hits.size());
    const auto end =
        gated_hits.begin() + static_cast<std::ptrdiff_t>(unit_count);
    const auto miss =
        std::find(gated_hits.begin(), end, static_cast<uint8_t>(0));
    return unit_begin + static_cast<size_t>(miss - gated_hits.begin());
  }

  // Units in [begin_unit, request.unit_count()) form the tail; a begin_unit
  // past the end yields an empty tail.
  PrefetchTailProbe(const StoragePrefetchRequest& request, size_t begin_unit) {
    const size_t unit_total = request.unit_count();
    const size_t begin = std::min(begin_unit, unit_total);
    size_t block_count = 0;
    for (size_t unit = begin; unit < unit_total; ++unit) {
      block_count += request.units[unit].gated_blocks.size();
    }
    transfers_.reserve(block_count);
    offsets_.reserve(unit_total - begin + 1);
    for (size_t unit = begin; unit < unit_total; ++unit) {
      offsets_.emplace_back(transfers_.size());
      const std::vector<BlockTransferInfo>& gated =
          request.units[unit].gated_blocks;
      transfers_.insert(transfers_.end(), gated.begin(), gated.end());
    }
    offsets_.emplace_back(transfers_.size());
  }

  size_t unit_count() const { return offsets_.size() - 1; }

  // Flattened gated blocks of every tail unit, in unit order.
  std::vector<BlockTransferInfo>& transfers() { return transfers_; }

  // present holds one entry per transfers() element.
  std::optional<size_t> count_present_units(
      const std::vector<uint8_t>& present) const {
    if (present.size() != transfers_.size()) {
      LOG(ERROR) << "Invalid prefetch probe result size: " << present.size()
                 << ", expected: " << transfers_.size();
      return std::nullopt;
    }
    size_t present_units = 0;
    for (size_t unit = 0; unit < unit_count(); ++unit) {
      const auto begin =
          present.begin() + static_cast<std::ptrdiff_t>(offsets_[unit]);
      const auto end =
          present.begin() + static_cast<std::ptrdiff_t>(offsets_[unit + 1]);
      present_units +=
          std::all_of(begin, end, [](uint8_t hit) { return hit != 0; }) ? 1 : 0;
    }
    return present_units;
  }

 private:
  std::vector<BlockTransferInfo> transfers_;
  // offsets_[unit] is the first transfers_ index of a tail unit; the last
  // entry is transfers_.size().
  std::vector<size_t> offsets_;
};

}  // namespace xllm
