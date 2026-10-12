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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "core/layers/common/dflash2_grouped_conv.h"
#include "core/layers/mlu/attention.h"
#include "core/layers/mlu/dflash2_context_kv.h"
#include "core/platform/platform.h"

namespace xllm::layer {
namespace {

ModelArgs make_args() {
  ModelArgs args;
  args.n_layers(2);
  args.hidden_size(16);
  args.head_dim(128);
  args.n_heads(32);
  args.n_kv_heads(8);
  args.max_position_embeddings(32);
  args.rope_theta(10000);
  args.rms_norm_eps(1e-5);
  return args;
}

torch::Tensor make_weight(int64_t offset) {
  return ((torch::arange(8 * 128 * 16, torch::kFloat32) + offset)
              .remainder(29) -
          14)
             .reshape({8 * 128, 16}) /
         32;
}

class DFlash2ContextKVTest
    : public ::testing::TestWithParam<std::pair<int32_t, int32_t>> {};

TEST_P(DFlash2ContextKVTest, WritesOnlySelectedSlotsWithReferenceNormAndRope) {
  const auto [tp_size, rank] = GetParam();
  const ModelArgs args = make_args();
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(torch::Device(Platform::type_torch(), 0));
  ModelContext context(
      ParallelArgs(rank, tp_size, nullptr), args, QuantArgs(), options);
  DFlash2ContextKV writer(context);
  const torch::Tensor norm =
      torch::linspace(0.5, 1.5, 128).to(torch::kBFloat16);
  for (int32_t layer = 0; layer < 2; ++layer) {
    const std::string prefix =
        "layers." + std::to_string(layer) + ".self_attn.";
    // Deliberately split K, V and norm across independent checkpoint shards.
    writer.load_state_dict(StateDict(
        {{prefix + "k_proj.weight", make_weight(layer).to(torch::kBFloat16)}}));
    writer.load_state_dict(
        StateDict({{prefix + "v_proj.weight",
                    make_weight(layer + 7).to(torch::kBFloat16)}}));
    writer.load_state_dict(StateDict({{prefix + "k_norm.weight", norm}}));
  }
  writer.finalize_loaded_weights();

  const int64_t heads = std::max<int64_t>(8 / tp_size, 1);
  const int64_t first_head = tp_size <= 8 ? rank * heads : rank / (tp_size / 8);
  std::vector<KVCache> caches;
  caches.reserve(2);
  for (int32_t layer = 0; layer < 2; ++layer) {
    caches.emplace_back(
        KVCacheTensors{torch::zeros({3, heads, 16, 128}, options),
                       torch::zeros({3, heads, 16, 128}, options)});
  }
  const torch::Tensor hidden =
      (torch::arange(48, torch::kFloat32).reshape({3, 16}).remainder(13) / 16)
          .to(torch::kBFloat16);
  const torch::Tensor positions = torch::tensor({2, 7, 4}, torch::kInt32);
  const torch::Tensor slots = torch::tensor({17, 34, 21}, torch::kInt32);
  ASSERT_TRUE(writer.write(hidden.to(options),
                           positions.to(options.device()),
                           slots.to(options.device()),
                           caches,
                           ModelInputParams()));

  const torch::Tensor inv_freq = torch::exp(
      -std::log(10000.0) * torch::arange(0, 128, 2, torch::kFloat32) / 128);
  const torch::Tensor angles =
      positions.to(torch::kFloat32).view({3, 1}) * inv_freq;
  const torch::Tensor cosine = torch::cat({angles.cos(), angles.cos()}, -1)
                                   .to(torch::kBFloat16)
                                   .to(torch::kFloat32)
                                   .unsqueeze(1);
  const torch::Tensor sine = torch::cat({angles.sin(), angles.sin()}, -1)
                                 .to(torch::kBFloat16)
                                 .to(torch::kFloat32)
                                 .unsqueeze(1);
  for (int32_t layer = 0; layer < 2; ++layer) {
    const auto project = [&](int64_t offset) {
      const torch::Tensor weight =
          make_weight(offset)
              .to(torch::kBFloat16)
              .slice(0, first_head * 128, (first_head + heads) * 128);
      return torch::matmul(hidden.to(torch::kFloat32),
                           weight.to(torch::kFloat32).t())
          .to(torch::kBFloat16)
          .to(torch::kFloat32)
          .view({3, heads, 128});
    };
    torch::Tensor key = project(layer);
    key = (key * torch::rsqrt(key.pow(2).mean(-1, true) + 1e-5) *
           norm.to(torch::kFloat32))
              .to(torch::kBFloat16)
              .to(torch::kFloat32);
    const torch::Tensor rotated =
        torch::cat({-key.slice(-1, 64, 128), key.slice(-1, 0, 64)}, -1);
    key = (key * cosine + rotated * sine)
              .to(torch::kBFloat16)
              .to(torch::kFloat32);
    const torch::Tensor value = project(layer + 7);
    torch::Tensor expected_k = torch::zeros({3, heads, 16, 128});
    torch::Tensor expected_v = torch::zeros_like(expected_k);
    for (int64_t row = 0; row < 3; ++row) {
      const int32_t slot = slots[row].item<int32_t>();
      expected_k.select(0, slot / 16).select(1, slot % 16).copy_(key[row]);
      expected_v.select(0, slot / 16).select(1, slot % 16).copy_(value[row]);
    }
    EXPECT_TRUE(
        torch::allclose(caches[layer].get_k_cache().cpu().to(torch::kFloat32),
                        expected_k,
                        0.02,
                        0.02));
    EXPECT_TRUE(
        torch::allclose(caches[layer].get_v_cache().cpu().to(torch::kFloat32),
                        expected_v,
                        0.02,
                        0.02));
  }
}

INSTANTIATE_TEST_SUITE_P(ShardedAndReplicatedHeads,
                         DFlash2ContextKVTest,
                         ::testing::Values(std::pair<int32_t, int32_t>{1, 0},
                                           std::pair<int32_t, int32_t>{2, 0},
                                           std::pair<int32_t, int32_t>{2, 1},
                                           std::pair<int32_t, int32_t>{16, 0},
                                           std::pair<int32_t, int32_t>{16, 1},
                                           std::pair<int32_t, int32_t>{16, 14},
                                           std::pair<int32_t, int32_t>{16,
                                                                       15}));

TEST(DFlash2ContextKVDeathTest, RejectsInvalidHeadTopologyBeforeAllocation) {
  ModelArgs args = make_args();
  args.n_heads(30);
  ModelContext context(
      ParallelArgs(0, 16, nullptr), args, QuantArgs(), torch::TensorOptions());
  EXPECT_DEATH(
      { DFlash2ContextKV writer(context); }, "args.n_heads\\(\\) % tp_size");
}

class DFlash2AttentionTest : public ::testing::TestWithParam<bool> {};

TEST_P(DFlash2AttentionTest,
       NonCausalWindowIsOptInAndRespectsSequenceBoundaries) {
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(torch::Device(Platform::type_torch(), 0));
  Attention attention(/*num_heads=*/1,
                      /*head_size=*/128,
                      /*scale=*/1.0f,
                      /*num_kv_heads=*/1,
                      /*sliding_window=*/3);
  AttentionMetadata metadata{};
  metadata.is_dummy = false;
  metadata.is_prefill = !GetParam();
  metadata.is_chunked_prefill = GetParam();
  metadata.block_table =
      torch::tensor({1, 2}, options.dtype(torch::kInt32)).reshape({2, 1});
  metadata.is_causal = false;  // Legacy callers did not control MLU causality.
  metadata.max_query_len = 4;
  metadata.max_seq_len = 4;
  metadata.q_cu_seq_lens =
      torch::tensor({0, 4, 8}, options.dtype(torch::kInt32));
  metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
  metadata.slot_mapping = torch::tensor({16, 17, 18, 19, 32, 33, 34, 35},
                                        options.dtype(torch::kInt32));
  KVCache cache(KVCacheTensors{torch::zeros({3, 1, 16, 128}, options),
                               torch::zeros({3, 1, 16, 128}, options)});
  torch::Tensor query = torch::zeros({8, 128}, options);
  torch::Tensor key = torch::zeros_like(query);
  torch::Tensor value =
      torch::tensor({1., 2., 3., 4., 10., 20., 30., 40.}, options)
          .view({8, 1})
          .expand({8, 128})
          .contiguous();
  const torch::Tensor input_value = value;
  const torch::Tensor causal =
      std::get<0>(attention->forward(metadata, query, key, value, cache));
  const torch::Tensor expected_causal =
      torch::tensor({1., 1.5, 2., 3., 10., 15., 20., 30.});
  EXPECT_TRUE(torch::allclose(causal.cpu().to(torch::kFloat32).select(1, 0),
                              expected_causal,
                              0.01,
                              0.01));

  // Paged prefill rebinds key/value to the cache tensors. Each invocation
  // must receive fresh query-block inputs, as the decoder layer supplies.
  key = torch::zeros({8, 128}, options);
  value = input_value;
  metadata.non_causal_window_right = 3;
  const torch::Tensor bidirectional =
      std::get<0>(attention->forward(metadata, query, key, value, cache));
  const torch::Tensor expected_bidirectional =
      torch::tensor({2.5, 2.5, 2.5, 3., 25., 25., 25., 30.});
  EXPECT_TRUE(
      torch::allclose(bidirectional.cpu().to(torch::kFloat32).select(1, 0),
                      expected_bidirectional,
                      0.01,
                      0.01));
}

INSTANTIATE_TEST_SUITE_P(DenseAndPaged,
                         DFlash2AttentionTest,
                         ::testing::Bool());

TEST_P(DFlash2AttentionTest, ShorterBlocksDoNotReadPastTheirLogicalKVLength) {
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(torch::Device(Platform::type_torch(), 0));
  Attention attention(/*num_heads=*/1,
                      /*head_size=*/128,
                      /*scale=*/1.0f,
                      /*num_kv_heads=*/1,
                      /*sliding_window=*/2048);
  AttentionMetadata metadata{};
  metadata.is_dummy = false;
  metadata.is_prefill = !GetParam();
  metadata.is_chunked_prefill = GetParam();
  metadata.non_causal_window_right = 5;
  metadata.max_query_len = 6;  // One anchor and five candidate positions.
  metadata.max_seq_len = 6;
  metadata.q_cu_seq_lens =
      torch::tensor({0, 6, 12}, options.dtype(torch::kInt32));
  metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
  metadata.block_table =
      torch::tensor({1, 2}, options.dtype(torch::kInt32)).reshape({2, 1});
  metadata.slot_mapping =
      torch::tensor({16, 17, 18, 19, 20, 21, 32, 33, 34, 35, 36, 37},
                    options.dtype(torch::kInt32));
  // Poison unused cache rows: a window extending past the six valid tokens
  // must not attend to stale speculative tokens or the adjacent sequence.
  KVCache cache(KVCacheTensors{torch::zeros({3, 1, 16, 128}, options),
                               torch::full({3, 1, 16, 128}, 1000, options)});
  torch::Tensor query = torch::zeros({12, 128}, options);
  torch::Tensor key = torch::zeros_like(query);
  torch::Tensor value =
      torch::tensor({1., 2., 3., 4., 5., 6., 10., 20., 30., 40., 50., 60.},
                    options)
          .view({12, 1})
          .expand({12, 128})
          .contiguous();
  const torch::Tensor output =
      std::get<0>(attention->forward(metadata, query, key, value, cache));
  const torch::Tensor expected = torch::tensor(
      {3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 35., 35., 35., 35., 35., 35.});
  EXPECT_TRUE(torch::allclose(
      output.cpu().to(torch::kFloat32).select(1, 0), expected, 0.01, 0.01));
}

TEST(DFlash2GroupedConvTest, FiveCandidateBlocksResetAtEachAnchor) {
  // Four six-row sequences: the total row count is also divisible by eight,
  // so using the trained block size would silently mix neighboring requests.
  const torch::Tensor hidden = torch::arange(1, 25, torch::kFloat32)
                                   .view({24, 1})
                                   .expand({24, 4})
                                   .contiguous();
  const torch::Tensor output = dflash2_grouped_conv(hidden,
                                                    torch::zeros({24, 2, 2}),
                                                    torch::ones({2, 4}),
                                                    /*block_size=*/6,
                                                    /*num_groups=*/2,
                                                    /*group_size=*/2,
                                                    /*taps=*/2);
  const torch::Tensor expected =
      torch::tensor({1.,  3.,  5.,  7.,  9.,  11., 7.,  15.,
                     17., 19., 21., 23., 13., 27., 29., 31.,
                     33., 35., 19., 39., 41., 43., 45., 47.})
          .view({24, 1})
          .expand({24, 4});
  EXPECT_TRUE(torch::equal(output, expected));
}

}  // namespace
}  // namespace xllm::layer
