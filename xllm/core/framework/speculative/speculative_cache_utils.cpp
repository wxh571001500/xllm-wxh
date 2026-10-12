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

#include "core/framework/speculative/speculative_cache_utils.h"

#include <glog/logging.h>

namespace xllm::speculative {

torch::Tensor map_positions_to_cache_slots(const torch::Tensor& block_tables,
                                           const torch::Tensor& positions,
                                           int64_t block_size) {
  CHECK_EQ(positions.dim(), 2);
  CHECK(block_tables.defined());
  CHECK_EQ(block_tables.dim(), 2);
  CHECK_GE(block_tables.size(0), positions.size(0));
  CHECK_GT(block_size, 0);
  const int64_t batch_size = positions.size(0);
  torch::Tensor position_long =
      positions.to(torch::dtype(torch::kLong).device(positions.device()));
  torch::Tensor block_indices =
      torch::floor_divide(position_long, block_size)
          .to(torch::dtype(torch::kLong).device(position_long.device()));
  torch::Tensor block_ids = block_tables.slice(/*dim=*/0, 0, batch_size)
                                .gather(/*dim=*/1, block_indices)
                                .to(torch::kLong);
  return (block_ids * block_size + position_long.remainder(block_size))
      .to(torch::kInt)
      .flatten();
}

}  // namespace xllm::speculative
