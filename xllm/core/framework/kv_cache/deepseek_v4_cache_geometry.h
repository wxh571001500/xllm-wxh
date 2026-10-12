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

#include <cstdint>

namespace xllm {

constexpr int32_t kDsv4C4CompressRatio = 4;
constexpr int32_t kDsv4C128CompressRatio = 128;
#if defined(USE_MLU)
constexpr int64_t kDsv4CompressedBlockTokenSize = 128;
#else
constexpr int64_t kDsv4CompressedBlockTokenSize = 2048;
#endif

class Dsv4CacheGeometry final {
 public:
  [[nodiscard]] constexpr int64_t compressed_block_token_size() const {
    return kDsv4CompressedBlockTokenSize;
  }

  [[nodiscard]] constexpr int64_t compressed_physical_dim(
      int32_t compress_ratio) const {
    if ((compress_ratio != kDsv4C4CompressRatio &&
         compress_ratio != kDsv4C128CompressRatio)) {
      return 0;
    }
    return kDsv4CompressedBlockTokenSize / compress_ratio;
  }

  [[nodiscard]] constexpr int64_t c4_physical_dim() const {
    return compressed_physical_dim(kDsv4C4CompressRatio);
  }

  [[nodiscard]] constexpr int64_t c128_physical_dim() const {
    return compressed_physical_dim(kDsv4C128CompressRatio);
  }
};

}  // namespace xllm
