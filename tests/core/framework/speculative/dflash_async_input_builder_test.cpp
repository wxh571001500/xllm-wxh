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

#include "core/framework/speculative/dflash_async_input_builder.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm::dflash_async {
namespace {

TEST(DFlashAsyncInputBuilderTest, AllAcceptanceLengthsAcrossPageBoundaries) {
  constexpr int64_t kWidth = 8;
  constexpr int64_t kBlockSize = 128;
  torch::Tensor accepted = torch::full({kWidth, kWidth}, -1, torch::kLong);
  torch::Tensor base =
      torch::tensor({120, 121, 126, 127, 128, 2297, 2303, 2304}, torch::kLong);
  torch::Tensor table =
      torch::arange(8 * 32, torch::kInt32).reshape({8, 32}).flip({1});
  for (int64_t seq = 0; seq < kWidth; ++seq) {
    for (int64_t token = 0; token <= seq; ++token) {
      accepted[seq][token] = 100 + seq * kWidth + token;
    }
  }
  NextDraftInputs result = prepare_next_draft(
      accepted, base, table, /*mask_token_id=*/999, kBlockSize);
  for (int64_t seq = 0; seq < kWidth; ++seq) {
    const int64_t prefix = seq + 1;
    const int64_t old_position = base[seq].item<int64_t>();
    EXPECT_EQ(result.accepted_counts[seq].item<int64_t>(), prefix);
    EXPECT_EQ(result.kv_lengths[seq].item<int64_t>(),
              old_position + prefix + kWidth);
    for (int64_t token = 0; token < kWidth; ++token) {
      const int64_t index = seq * kWidth + token;
      const int64_t query_position = old_position + prefix + token;
      const int64_t page =
          table[seq][query_position / kBlockSize].item<int64_t>();
      EXPECT_EQ(result.query_positions[index].item<int64_t>(), query_position);
      EXPECT_EQ(result.query_slots[index].item<int64_t>(),
                page * kBlockSize + query_position % kBlockSize);
      EXPECT_EQ(result.query_tokens[index].item<int64_t>(),
                token == 0 ? accepted[seq][prefix - 1].item<int64_t>() : 999);
      const int64_t context_position = old_position + token;
      const int64_t context_slot =
          token < prefix
              ? table[seq][context_position / kBlockSize].item<int64_t>() *
                        kBlockSize +
                    context_position % kBlockSize
              : -1;
      EXPECT_EQ(result.context_positions[index].item<int64_t>(),
                context_position);
      EXPECT_EQ(result.context_slots[index].item<int64_t>(), context_slot);
    }
  }
}

TEST(DFlashAsyncInputBuilderTest,
     OnlyTheContiguousAcceptedPrefixWritesContext) {
  torch::Tensor accepted =
      torch::tensor({{31, 32, -1, 77, -1, -1, -1, -1}}, torch::kLong);
  NextDraftInputs result =
      prepare_next_draft(accepted,
                         torch::tensor({127}, torch::kLong),
                         torch::tensor({{8, 3, 9}}, torch::kInt32),
                         /*mask_token_id=*/999,
                         /*block_size=*/128);
  EXPECT_EQ(result.accepted_counts.item<int64_t>(), 2);
  EXPECT_EQ(result.anchor_tokens.item<int64_t>(), 32);
  EXPECT_TRUE(torch::equal(
      result.context_slots,
      torch::tensor({1151, 384, -1, -1, -1, -1, -1, -1}, torch::kInt32)));
  EXPECT_EQ(result.query_positions[0].item<int64_t>(), 129);
  EXPECT_EQ(result.query_slots[0].item<int64_t>(), 385);
}

}  // namespace
}  // namespace xllm::dflash_async
