/* Copyright 2025-2026 The xLLM Authors.

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

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <limits>

#include "kernels/mlu/kpool.h"
#include "platform/platform.h"

namespace xllm::kernel::mlu {
namespace {

TEST(KPoolKernelTest, ScoresRespectHeadsCausalityAndPages) {
  torch::manual_seed(203);
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor q = torch::randn({7, 32, 128}, opts);
  const torch::Tensor w = torch::randn({7, 32});
  const torch::Tensor cache = torch::randn({4, 1, 4, 128}, opts);
  const torch::Tensor table = torch::tensor({{3, 0}, {2, -1}}, torch::kInt32);
  const torch::Tensor positions =
      torch::tensor({-1, 2, 3, 6, 7, 17, 22}, torch::kInt32);
  const torch::Tensor rows =
      torch::tensor({0, 0, 0, 0, 0, 1, 1}, torch::kInt64);
  torch::Tensor expected =
      torch::full({7, 8}, -std::numeric_limits<float>::infinity());
  for (int64_t i = 0; i < 7; ++i) {
    for (int64_t j = 0; j < 8; ++j) {
      const int64_t page =
          table[rows[i].item<int64_t>()][j / 4].item<int32_t>();
      if (page < 0 || (j + 1) * 4 > positions[i].item<int32_t>() + 1) {
        continue;
      }
      const torch::Tensor dots =
          (q[i].to(torch::kFloat32) * cache[page][0][j % 4].to(torch::kFloat32))
              .sum(1);
      expected[i][j] = (torch::relu(dots * 0.125) * w[i]).sum();
    }
  }
  {
    torch::Tensor storage =
        torch::full({7, 10}, 99.0, expected.options().device(device));
    torch::Tensor scores = storage.narrow(1, 1, 8);
    score_kpool(q.to(device),
                w.to(device),
                cache.to(device),
                table.to(device),
                positions.to(device),
                rows.to(device),
                scores,
                16,
                4,
                0.125);
    const torch::Tensor actual = scores.cpu();
    EXPECT_TRUE(
        torch::equal(torch::isneginf(actual), torch::isneginf(expected)));
    const torch::Tensor finite = torch::isfinite(expected);
    EXPECT_TRUE(torch::allclose(actual.masked_select(finite),
                                expected.masked_select(finite),
                                1e-4,
                                1e-4));
    EXPECT_TRUE(storage.select(1, 0).eq(99).all().item<bool>());
    EXPECT_TRUE(storage.select(1, 9).eq(99).all().item<bool>());
  }
}

TEST(KPoolKernelTest, NativeTopkHandlesTiesNonfiniteAndEmpty) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  for (const int64_t columns : {0, 17, 513}) {
    torch::Tensor scores = torch::zeros({3, columns}, opts);
    if (columns > 0) {
      scores[1].fill_(-std::numeric_limits<float>::infinity());
      scores[2][0] = std::numeric_limits<float>::quiet_NaN();
      scores[2][1] = std::numeric_limits<float>::infinity();
    }
    const torch::Tensor actual = select_kpool(scores, 512);
    EXPECT_EQ(actual.scalar_type(), torch::kInt64);
    EXPECT_EQ(actual.size(1), 512);
    const int64_t k = std::min<int64_t>(512, columns);
    if (k > 0) {
      const auto [v, ids] = scores.topk(k, -1);
      EXPECT_TRUE(torch::equal(actual.narrow(1, 0, k),
                               torch::where(torch::isfinite(v), ids, -1)));
    }
    EXPECT_TRUE(actual.narrow(1, k, 512 - k).eq(-1).all().item<bool>());
  }
  EXPECT_EQ(select_kpool(torch::empty({0, 5}, opts), 3).numel(), 0);
  EXPECT_EQ(select_kpool(torch::empty({2, 5}, opts), 0).numel(), 0);
}

void check_verify_scores(int64_t requests, int64_t capacity) {
  torch::manual_seed(19023);
  const torch::Device device(Platform::type_torch(), 0);
  const auto bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto fp32 = bf16.dtype(torch::kFloat32);
  const auto ints = bf16.dtype(torch::kInt32);
  const int64_t tokens = requests * 4;
  const int64_t pages_per_request = (capacity + 3) / 4;
  const int64_t pages = requests * pages_per_request;
  const torch::Tensor query = torch::randn({tokens, 32, 128}, bf16);
  const torch::Tensor weights = torch::randn({tokens, 32}, fp32);
  const torch::Tensor cache = torch::randn({pages, 1, 4, 128}, bf16);
  const torch::Tensor query_before = query.clone();
  const torch::Tensor weights_before = weights.clone();
  const torch::Tensor cache_before = cache.clone();
  torch::Tensor table =
      torch::arange(pages, ints).flip({0}).view({requests, pages_per_request});
  table.index_put_({0, 0}, -1);
  table.index_put_({1, 1}, pages + 3);
  torch::Tensor positions =
      (torch::arange(4, ints) + capacity * 4 - 4).repeat({requests});
  positions.index_put_({0}, -1);
  const torch::Tensor rows = torch::arange(requests, ints).repeat_interleave(4);
  const torch::Tensor starts = torch::arange(requests + 1, ints) * 4;
  torch::Tensor expected = torch::empty({tokens, capacity}, fp32);
  torch::Tensor storage = torch::full({tokens, capacity + 2}, 99.0, fp32);
  torch::Tensor actual = storage.narrow(/*dim=*/1, /*start=*/1, capacity);
  score_kpool(query,
              weights,
              cache,
              table,
              positions,
              rows,
              expected,
              /*block_size=*/16,
              /*pool_size=*/4,
              /*scale=*/0.015625);
  score_kpool(query,
              weights,
              cache,
              table,
              positions,
              rows,
              actual,
              /*block_size=*/16,
              /*pool_size=*/4,
              /*scale=*/0.015625,
              starts);
  EXPECT_TRUE(torch::equal(torch::isneginf(actual), torch::isneginf(expected)));
  const torch::Tensor finite = torch::isfinite(expected);
  EXPECT_TRUE(torch::allclose(actual.masked_select(finite),
                              expected.masked_select(finite),
                              /*rtol=*/1e-4,
                              /*atol=*/1e-4));
  EXPECT_TRUE(torch::equal(select_kpool(actual, /*count=*/512),
                           select_kpool(expected, /*count=*/512)));
  EXPECT_TRUE(storage.select(1, 0).eq(99.0).all().item<bool>());
  EXPECT_TRUE(storage.select(1, capacity + 1).eq(99.0).all().item<bool>());
  EXPECT_TRUE(torch::equal(query, query_before));
  EXPECT_TRUE(torch::equal(weights, weights_before));
  EXPECT_TRUE(torch::equal(cache, cache_before));
}

TEST(KPoolKernelTest, VerifyScoresPreserveFp32WeightsPagesAndCausality) {
  check_verify_scores(/*requests=*/3, /*capacity=*/137);
}

TEST(KPoolKernelTest, VerifyScoresMatchAtBatch16Context64K) {
  check_verify_scores(/*requests=*/16, /*capacity=*/16384);
}

TEST(KPoolKernelTest, VerifyGraphReadsLiveOffsetsAndClearsUnassignedPadding) {
  torch::NoGradGuard no_grad;
  torch::manual_seed(8221);
  const torch::Device device(Platform::type_torch(), 0);
  const auto bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto fp32 = bf16.dtype(torch::kFloat32);
  const auto ints = bf16.dtype(torch::kInt64);
  const torch::Tensor query = torch::randn({12, 32, 128}, bf16);
  const torch::Tensor weights = torch::randn({12, 32}, fp32);
  const torch::Tensor cache = torch::randn({12, 1, 4, 128}, bf16);
  torch::Tensor table = torch::arange(12, ints).view({3, 4});
  torch::Tensor positions = torch::full({12}, 63, ints);
  torch::Tensor rows = torch::arange(3, ints).repeat_interleave(4);
  torch::Tensor starts = torch::tensor({0, 4, 8, 12}, ints);
  torch::Tensor storage = torch::full({12, 18}, 99.0, fp32);
  torch::Tensor actual = storage.narrow(/*dim=*/1, /*start=*/1, /*length=*/16);
  torch::Tensor expected = torch::empty({12, 16}, fp32);
  const auto score_verify = [&]() {
    score_kpool(query,
                weights,
                cache,
                table,
                positions,
                rows,
                actual,
                /*block_size=*/16,
                /*pool_size=*/4,
                /*scale=*/0.015625,
                starts);
  };
  score_verify();
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  {
    torch_mlu::mlu::MLUStreamGuard guard(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false,
                                     /*device_index=*/0));
    graph.capture_begin();
    score_verify();
    graph.capture_end();
  }
  // The middle request becomes empty, the last request moves to row two,
  // and the fixed graph bucket retains six completely unassigned rows.
  starts.copy_(torch::tensor({0, 2, 2, 6}, ints));
  rows.copy_(torch::tensor({0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0}, ints));
  positions.copy_(
      torch::tensor({3, 4, 7, 8, 11, 12, -1, -1, -1, -1, -1, -1}, ints));
  table.copy_(table.flip({1}));
  graph.replay();
  torch_mlu::synchronize();
  score_kpool(query,
              weights,
              cache,
              table,
              positions,
              rows,
              expected,
              /*block_size=*/16,
              /*pool_size=*/4,
              /*scale=*/0.015625);
  EXPECT_TRUE(torch::equal(torch::isneginf(actual), torch::isneginf(expected)));
  const torch::Tensor finite = torch::isfinite(expected);
  EXPECT_TRUE(torch::allclose(actual.masked_select(finite),
                              expected.masked_select(finite),
                              /*rtol=*/1e-4,
                              /*atol=*/1e-4));
  EXPECT_TRUE(storage.select(1, 0).eq(99.0).all().item<bool>());
  EXPECT_TRUE(storage.select(1, 17).eq(99.0).all().item<bool>());
}
}  // namespace
}  // namespace xllm::kernel::mlu
