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

#include "layers/mlu/glm5_next/glm5_next_pcp_context.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace xllm::layer::glm5_next_pcp {
namespace {

TEST(Glm5NextPcpContextTest, ConvertsCumulativeRequestLengths) {
  EXPECT_EQ(query_lengths_from_cumulative({0, 9, 9, 17}, 17),
            std::vector<int32_t>({9, 0, 8}));
}

TEST(Glm5NextPcpContextTest, SplitsPackedRequestsOnChunkBoundaries) {
  const Geometry geometry = build_geometry({10, 3},
                                           /*cp_size=*/2,
                                           /*chunk_size=*/4);

  EXPECT_EQ(geometry.rows_by_rank[0], std::vector<int64_t>({0, 1, 2, 3}));
  EXPECT_EQ(geometry.rows_by_rank[1],
            std::vector<int64_t>({4, 5, 6, 7, 8, 9, 10, 11, 12}));
  EXPECT_EQ(geometry.lengths_by_rank[0], std::vector<int32_t>({4, 0}));
  EXPECT_EQ(geometry.lengths_by_rank[1], std::vector<int32_t>({6, 3}));
  EXPECT_EQ(geometry.tokens_per_rank, std::vector<int32_t>({4, 9}));
  EXPECT_EQ(geometry.restore_indices,
            std::vector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TEST(Glm5NextPcpContextTest, EmptyRanksPreserveCollectiveOrder) {
  const Geometry geometry = build_geometry({1, 0, 7},
                                           /*cp_size=*/4,
                                           /*chunk_size=*/4);

  ASSERT_EQ(geometry.tokens_per_rank.size(), 4u);
  EXPECT_EQ(geometry.tokens_per_rank, std::vector<int32_t>({0, 4, 0, 4}));
  EXPECT_EQ(geometry.lengths_by_rank[0], std::vector<int32_t>({0, 0, 0}));
  EXPECT_EQ(geometry.lengths_by_rank[3], std::vector<int32_t>({1, 0, 3}));
}

TEST(Glm5NextPcpContextTest, RestoresEveryTokenOnceForUnevenSegments) {
  const Geometry geometry = build_geometry({9, 5, 1},
                                           /*cp_size=*/3,
                                           /*chunk_size=*/4);

  std::vector<int64_t> restored(geometry.total_tokens, -1);
  int64_t gathered_index = 0;
  for (const std::vector<int64_t>& rows : geometry.rows_by_rank) {
    for (const int64_t row : rows) {
      restored[static_cast<size_t>(row)] = gathered_index++;
    }
  }
  EXPECT_EQ(restored, geometry.restore_indices);
  std::sort(restored.begin(), restored.end());
  for (int32_t token = 0; token < geometry.total_tokens; ++token) {
    EXPECT_EQ(restored[static_cast<size_t>(token)], token);
  }
}

TEST(Glm5NextPcpContextTest, DeferredGatherRestoresPackedRowsAndDropsPadding) {
  Context context;
  context.geometry = build_geometry({9, 5, 1}, /*cp_size=*/3, /*chunk_size=*/4);
  context.local_query_lengths = context.geometry.lengths_by_rank[0];
  const auto options = torch::TensorOptions().dtype(torch::kInt64);
  context.restore_indices =
      torch::tensor(context.geometry.restore_indices, options);
  parallel_state::GatherAsyncCtx pending;
  pending.token_num_list = context.geometry.tokens_per_rank;
  const int32_t max_tokens = *std::max_element(pending.token_num_list.begin(),
                                               pending.token_num_list.end());
  pending.stacked = torch::full({3, max_tokens, 1}, -99, options);
  for (int32_t rank = 0; rank < 3; ++rank) {
    const auto& rows = context.geometry.rows_by_rank[rank];
    pending.stacked[rank]
        .narrow(0, 0, static_cast<int64_t>(rows.size()))
        .copy_(torch::tensor(rows, options).unsqueeze(1));
  }
  const torch::Tensor actual =
      finish_gather_restore(std::move(pending), context);
  EXPECT_TRUE(torch::equal(actual, torch::arange(15, options).unsqueeze(1)));
}

TEST(Glm5NextPcpContextTest, DeferredGatherPreservesSingleRequestOrder) {
  Context context;
  context.local_query_lengths = {3};
  const auto options = torch::TensorOptions().dtype(torch::kInt64);
  parallel_state::GatherAsyncCtx pending;
  pending.token_num_list = {3, 3};
  pending.stacked = torch::arange(6, options).view({2, 3, 1});
  const torch::Tensor actual =
      finish_gather_restore(std::move(pending), context);
  EXPECT_TRUE(torch::equal(actual, torch::arange(6, options).unsqueeze(1)));
}

}  // namespace
}  // namespace xllm::layer::glm5_next_pcp
