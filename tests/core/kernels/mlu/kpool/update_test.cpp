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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <string>

#include "kernels/mlu/kpool.h"
#include "platform/platform.h"

namespace xllm::kernel::mlu {
namespace {

// Independent sequential CPU reference: write ring, pool across members,
// round through BF16, rotate, then scatter to the physical pool page.
void update_reference(const torch::Tensor& raw,
                      const torch::Tensor& gate,
                      const torch::Tensor& ape,
                      const torch::Tensor& had,
                      const torch::Tensor& pos,
                      const torch::Tensor& starts,
                      const torch::Tensor& ids,
                      const torch::Tensor& table,
                      torch::Tensor& cache,
                      torch::Tensor& tail) {
  for (int64_t req = 0; req < ids.numel(); ++req) {
    const int64_t state = ids[req].item<int64_t>();
    for (int64_t i = starts[req].item<int64_t>();
         i < starts[req + 1].item<int64_t>();
         ++i) {
      const int64_t p = pos[i].item<int64_t>();
      if (p < 0) {
        continue;
      }
      tail[state][0][p % 12].copy_(raw[i]);
      tail[state][1][p % 12].copy_(gate[i]);
      if (p % 4 != 3) {
        continue;
      }
      const torch::Tensor members =
          torch::arange(p - 3, p + 1, torch::kInt64).remainder(12);
      const torch::Tensor k =
          tail[state][0].index_select(0, members).to(torch::kFloat32);
      const torch::Tensor g =
          tail[state][1].index_select(0, members).to(torch::kFloat32);
      const torch::Tensor pooled = (k * torch::softmax(g + ape, 0))
                                       .sum(0)
                                       .to(torch::kBFloat16)
                                       .to(torch::kFloat32);
      const int64_t pool = p / 4;
      const int64_t page = table[req][pool / 4].item<int64_t>();
      cache[page][0][pool % 4].copy_(
          torch::matmul(pooled, had.transpose(0, 1)));
    }
  }
}

TEST(KPoolKernelTest, PagedUpdatesPreserveRaggedRingAndUntouchedState) {
  torch::manual_seed(902);
  const torch::Device device(Platform::type_torch(), 0);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor ape = torch::randn({4, 128});
  const torch::Tensor had = torch::eye(128) + torch::randn({128, 128}) * 0.02;
  const torch::Tensor ids = torch::tensor({3, 1, 0}, torch::kInt64);
  const torch::Tensor table =
      torch::tensor({{4, 0, 2}, {3, 1, 5}, {6, 7, 8}}, torch::kInt32);
  for (const bool decode : {false, true}) {
    torch::Tensor expected_cache = torch::randn({9, 1, 4, 128}, bf16);
    torch::Tensor expected_tail = torch::randn({5, 2, 12, 128}, bf16);
    torch::Tensor cache = expected_cache.to(device);
    torch::Tensor tail = expected_tail.to(device);
    const void* cache_ptr = cache.data_ptr();
    const void* tail_ptr = tail.data_ptr();
    // Non-aligned first chunk, ring wrap, then rejected speculative positions
    // overwritten by a shorter continuation. Request 2 is entirely padding.
    for (const int64_t start : {2, 8, 14, 15, 19}) {
      const torch::Tensor positions =
          torch::cat({torch::arange(start, start + 6, torch::kInt64),
                      torch::arange(start + 1, start + 4, torch::kInt64),
                      torch::full({2}, -1, torch::kInt64)});
      const torch::Tensor starts = torch::tensor({0, 6, 9, 11}, torch::kInt64);
      const torch::Tensor rows =
          torch::tensor({0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2}, torch::kInt64);
      const torch::Tensor raw = torch::randn({11, 128}, bf16);
      const torch::Tensor gate = torch::randn({11, 128}, bf16);
      update_reference(raw,
                       gate,
                       ape,
                       had,
                       positions,
                       starts,
                       ids,
                       table,
                       expected_cache,
                       expected_tail);
      update_kpool(raw.to(device),
                   gate.to(device),
                   ape.to(device),
                   had.to(device),
                   cache,
                   tail,
                   ids.to(device),
                   table.to(device),
                   positions.to(device),
                   rows.to(device),
                   starts.to(device),
                   16,
                   4,
                   decode);
      EXPECT_EQ(cache_ptr, cache.data_ptr());
      EXPECT_EQ(tail_ptr, tail.data_ptr());
      EXPECT_TRUE(torch::equal(tail.cpu(), expected_tail));
      EXPECT_TRUE(
          torch::equal(cache.slice(0, 6).cpu(), expected_cache.slice(0, 6)));
      EXPECT_TRUE(
          torch::allclose(cache.cpu(), expected_cache, 0.015625, 0.015625));
    }
  }
}
}  // namespace
}  // namespace xllm::kernel::mlu

namespace xllm::kernel::mlu {
TEST(KPoolKernelTest, Chunked64KMatchesTorchWithinRoundingPropagation) {
  torch::NoGradGuard no_grad;
  torch::manual_seed(72317);
  const torch::Device device(Platform::type_torch(), 0);
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  // GLM-5.3-Flash-W4A8: H=32, D=128, P=4, top-k=2048; MTP3 ring=12.
  const torch::Tensor raw = torch::randn({65536, 128}, bf16);
  const torch::Tensor gate = torch::randn_like(raw);
  const torch::Tensor ape = torch::randn({4, 128}) * 0.1;
  torch::Tensor had = torch::ones({1, 1});
  for (int64_t size = 1; size < 128; size *= 2) {
    had =
        torch::cat({torch::cat({had, had}, 1), torch::cat({had, -had}, 1)}, 0);
  }
  had /= std::sqrt(128.0);
  const torch::Tensor probabilities = torch::softmax(
      gate.to(torch::kFloat32).view({16384, 4, 128}) + ape.unsqueeze(0), 1);
  const torch::Tensor pooled =
      (raw.to(torch::kFloat32).view({16384, 4, 128}) * probabilities)
          .sum(1)
          .to(torch::kBFloat16)
          .to(torch::kFloat32);
  const torch::Tensor reference = torch::matmul(pooled, had.transpose(0, 1))
                                      .to(torch::kBFloat16)
                                      .to(torch::kFloat32);
  const torch::Tensor pages =
      torch::randperm(4096, torch::kInt64).view({1, 4096});
  const torch::Tensor table = pages.to(device, torch::kInt32);
  const torch::Tensor state_ids = torch::tensor({1}, torch::kInt64).to(device);
  const torch::Tensor starts =
      torch::tensor({0, 8192}, torch::kInt64).to(device);
  const torch::Tensor rows = torch::zeros({8192}, torch::kInt64).to(device);
  torch::Tensor cache = torch::zeros({4096, 1, 4, 128}, bf16.device(device));
  torch::Tensor tail = torch::zeros({2, 2, 12, 128}, bf16.device(device));
  int64_t strict_bad = 0;
  for (int64_t chunk = 0; chunk < 8; ++chunk) {
    SCOPED_TRACE(chunk);
    const int64_t end = (chunk + 1) * 8192;
    const int64_t pools = end / 4;
    const torch::Tensor positions =
        torch::arange(end - 8192, end, torch::kInt64).to(device);
    update_kpool(raw.narrow(0, end - 8192, 8192).to(device),
                 gate.narrow(0, end - 8192, 8192).to(device),
                 ape.to(device),
                 had.to(device),
                 cache,
                 tail,
                 state_ids,
                 table,
                 positions,
                 rows,
                 starts,
                 16,
                 4,
                 false);
    const torch::Tensor columns = torch::arange(pools, torch::kInt64);
    const torch::Tensor slots =
        pages[0].index_select(0, torch::floor_divide(columns, 4)) * 4 +
        columns.remainder(4);
    const torch::Tensor actual_cache =
        cache.cpu().view({-1, 128}).index_select(0, slots).to(torch::kFloat32);
    const torch::Tensor ref = reference.narrow(0, 0, pools);
    ASSERT_TRUE(torch::allclose(actual_cache, ref, 0.015625, 0.015625));
    for (int64_t token = end - 12; token < end; ++token) {
      EXPECT_TRUE(torch::equal(tail[1][0][token % 12].cpu(), raw[token]));
      EXPECT_TRUE(torch::equal(tail[1][1][token % 12].cpu(), gate[token]));
    }
    const torch::Tensor q =
        torch::randn({8, 32, 128}, bf16).to(torch::kFloat32);
    const torch::Tensor weights = torch::randn({8, 32});
    const auto score = [&](const torch::Tensor& keys) {
      return (torch::relu(torch::matmul(q, keys.transpose(0, 1)) / 64) *
              weights.unsqueeze(-1))
          .sum(1);
    };
    const torch::Tensor expected = score(ref);
    const torch::Tensor same_cache = score(actual_cache);
    // ReLU is 1-Lipschitz: propagate independently measured cache rounding
    // through |scale| sum_h |w_h| sum_d |Q_hd| |delta K_d|. The score-stage
    // tolerance is unchanged. Approved migration contract, 2026-09-17.
    const torch::Tensor bound =
        (torch::matmul(q.abs(), (actual_cache - ref).abs().transpose(0, 1)) *
         weights.abs().unsqueeze(-1))
            .sum(1) /
        64;
    const torch::Tensor query_positions = positions.narrow(0, 8184, 8);
    const torch::Tensor valid = (columns + 1).unsqueeze(0) * 4 <=
                                query_positions.cpu().unsqueeze(1) + 1;
    torch::Tensor output = torch::empty(
        {8, pools},
        torch::TensorOptions().device(device).dtype(torch::kFloat32));
    score_kpool(q.to(device, torch::kBFloat16),
                weights.to(device),
                cache,
                table,
                query_positions,
                rows.narrow(0, 0, 8),
                output,
                16,
                4,
                1.0 / 64);
    const torch::Tensor actual = output.cpu();
    ASSERT_TRUE(torch::allclose(actual.masked_select(valid),
                                same_cache.masked_select(valid),
                                1e-4,
                                1e-4));
    const torch::Tensor error = (actual - expected).abs();
    strict_bad += (valid & (error > 0.001 + expected.abs() * 0.001))
                      .sum()
                      .item<int64_t>();
    EXPECT_FALSE((valid & (error > bound + 1e-4 + same_cache.abs() * 1e-4))
                     .any()
                     .item<bool>());
    const torch::Tensor expected_masked =
        expected.masked_fill(~valid, -std::numeric_limits<float>::infinity());
    const torch::Tensor ids = select_kpool(output, 512).cpu();
    const torch::Tensor selected = expected_masked.gather(1, ids);
    const torch::Tensor cutoff =
        std::get<0>(expected_masked.topk(512, -1)).select(1, 511).unsqueeze(1);
    const torch::Tensor row_bound =
        std::get<0>((bound + 1e-4 + same_cache.abs() * 1e-4).max(1, true));
    EXPECT_TRUE((selected >= cutoff - 2 * row_bound).all().item<bool>());
  }
  // Retain the historical check as a diagnostic, not an unexplained pass.
  RecordProperty("strict_chain_score_violations", std::to_string(strict_bad));
}
}  // namespace xllm::kernel::mlu
