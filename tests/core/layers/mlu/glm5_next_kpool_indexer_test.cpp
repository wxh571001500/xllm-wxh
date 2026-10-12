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

#include "layers/mlu/glm5_next/glm5_next_kpool_indexer.h"

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "framework/config/kv_cache_config.h"
#include "framework/speculative/spec_input_builder.h"
#include "kernels/mlu/kpool.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "layers/common/attention_metadata_builder.h"
#include "layers/mlu/attention.h"
#include "models/llm/mlu/glm5_next_graph.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "runtime/forward_params.h"
#include "util/linalg.h"

namespace xllm::layer {
namespace {
// GLM-5.3-Flash-W4A8 text_config (2026-09-17), SHA256
// 07c7491191283d15537bb54351c98570e94e72207a25445bbb39b019acff3bb5.
constexpr int64_t kHidden = 4096;
constexpr int64_t kRank = 1536;
constexpr int64_t kHeads = 32;
constexpr int64_t kTopk = 2048;
constexpr int64_t kSelected = 512;

struct PoolHistory {
  torch::Tensor keys;
  torch::Tensor valid;
};

torch::Tensor reference_compression(const torch::Tensor& raw_k,
                                    const torch::Tensor& gate_score,
                                    const torch::Tensor& ape,
                                    const torch::Tensor& hadamard,
                                    int64_t index_kpool) {
  const int64_t num_pools = raw_k.size(0) / index_kpool;
  if (num_pools == 0) {
    return torch::empty({0, raw_k.size(1)},
                        raw_k.options().dtype(torch::kBFloat16));
  }

  const torch::Tensor keys = raw_k.view({num_pools, index_kpool, raw_k.size(1)})
                                 .to(torch::kCPU, torch::kFloat32);
  const torch::Tensor logits =
      gate_score.view({num_pools, index_kpool, raw_k.size(1)})
          .to(torch::kCPU, torch::kFloat32) +
      ape.to(torch::kCPU, torch::kFloat32).unsqueeze(/*dim=*/0);
  const torch::Tensor probabilities = torch::softmax(logits, /*dim=*/1);
  const torch::Tensor pooled = (keys * probabilities).sum(/*dim=*/1);
  return torch::matmul(
             pooled.to(torch::kBFloat16).to(torch::kCPU, torch::kFloat32),
             hadamard.to(torch::kCPU, torch::kFloat32).transpose(0, 1))
      .to(raw_k.device(), torch::kBFloat16);
}

PoolHistory read_pool_history(const torch::Tensor& compressed_pool_cache,
                              const torch::Tensor& block_table,
                              const torch::Tensor& kv_seq_lens,
                              int64_t max_kv_seq_len,
                              int64_t block_size,
                              int64_t index_kpool) {
  const int64_t count = (max_kv_seq_len + index_kpool - 1) / index_kpool;
  const int64_t pb = block_size / index_kpool;
  const torch::Tensor pages = block_table.cpu();
  const torch::Tensor lengths = kv_seq_lens.cpu();
  const torch::Tensor cache = compressed_pool_cache.cpu();
  torch::Tensor keys =
      torch::zeros({pages.size(0), count, cache.size(3)}, cache.options());
  torch::Tensor valid = torch::zeros({pages.size(0), count}, torch::kBool);
  for (int64_t req = 0; req < pages.size(0); ++req) {
    for (int64_t pool = 0; pool < count; ++pool) {
      if (pool / pb >= pages.size(1) ||
          (pool + 1) * index_kpool > lengths[req].item<int64_t>()) {
        continue;
      }
      const int64_t page = pages[req][pool / pb].item<int64_t>();
      if (page < 0 || page >= cache.size(0)) {
        continue;
      }
      keys[req][pool].copy_(cache[page][0][pool % pb]);
      valid[req][pool] = true;
    }
  }
  return {keys.to(compressed_pool_cache.device()),
          valid.to(compressed_pool_cache.device())};
}

// Exercise paged scoring with independently supplied contiguous keys.
torch::Tensor score_keys(const torch::Tensor& query,
                         const torch::Tensor& weights,
                         const torch::Tensor& keys,
                         double scale) {
  const int64_t pages = (keys.size(0) + 3) / 4;
  torch::Tensor cache = torch::zeros({pages, 1, 4, 128}, keys.options());
  cache.view({-1, 128}).narrow(0, 0, keys.size(0)).copy_(keys);
  const auto ints = query.options().dtype(torch::kInt64);
  torch::Tensor scores = torch::empty({query.size(0), keys.size(0)},
                                      query.options().dtype(torch::kFloat32));
  kernel::mlu::score_kpool(
      query,
      weights,
      cache,
      torch::arange(pages, ints).view({1, pages}),
      torch::full({query.size(0)}, keys.size(0) * 4 - 1, ints),
      torch::zeros({query.size(0)}, ints),
      scores,
      16,
      4,
      scale);
  return scores;
}

torch::Tensor reference_selection(const torch::Tensor& query,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& positions,
                                  const torch::Tensor& rows,
                                  const torch::Tensor& cache,
                                  const torch::Tensor& table,
                                  int64_t pool_block_size,
                                  int64_t pool_size,
                                  int64_t top_k,
                                  double scale) {
  const torch::Tensor cpu_positions = positions.to(torch::kCPU);
  const torch::Tensor cpu_rows = rows.to(torch::kCPU);
  torch::Tensor result = torch::full(
      {query.size(0), top_k}, -1, query.options().dtype(torch::kInt64));
  for (int64_t row = 0; row < query.size(0); ++row) {
    const int64_t count = (cpu_positions[row].item<int64_t>() + 1) / pool_size;
    if (count == 0) {
      continue;
    }
    const torch::Tensor ids = torch::arange(count, result.options());
    const torch::Tensor slots =
        table[cpu_rows[row].item<int64_t>()].index_select(
            0, torch::floor_divide(ids, pool_block_size)) *
            pool_block_size +
        torch::remainder(ids, pool_block_size);
    const torch::Tensor keys =
        cache.view({-1, query.size(2)}).index_select(0, slots);
    const torch::Tensor scores =
        (torch::relu(torch::matmul(query[row].to(torch::kFloat32),
                                   keys.to(torch::kFloat32).transpose(0, 1)) *
                     scale) *
         weights[row].unsqueeze(1))
            .sum(0);
    const auto [values, indices] = scores.topk(std::min(count, top_k));
    result[row]
        .narrow(0, 0, indices.numel())
        .copy_(torch::where(torch::isfinite(values), indices, -1));
  }
  return result;
}

kernel::mlu::KPoolSelection reference_expansion(const torch::Tensor& pool_ids,
                                                const torch::Tensor& positions,
                                                const torch::Tensor& rows,
                                                const torch::Tensor& table,
                                                int64_t block_size,
                                                int64_t index_topk,
                                                int64_t pool_size,
                                                bool select_tail) {
  const torch::Tensor ids =
      pool_ids.to(torch::kCPU, torch::kInt64).contiguous();
  const torch::Tensor pos =
      positions.to(torch::kCPU, torch::kInt64).contiguous();
  const torch::Tensor batch = rows.to(torch::kCPU, torch::kInt64).contiguous();
  const torch::Tensor blocks =
      table.to(torch::kCPU, torch::kInt64).contiguous();
  torch::Tensor slots =
      torch::full({ids.size(0), index_topk + pool_size - 1}, -1, torch::kInt32);
  torch::Tensor lengths = torch::zeros({ids.size(0)}, torch::kInt32);
  const auto input = ids.accessor<int64_t, 2>();
  const auto query_positions = pos.accessor<int64_t, 1>();
  const auto row_batch = batch.accessor<int64_t, 1>();
  const auto block_table = blocks.accessor<int64_t, 2>();
  auto output = slots.accessor<int32_t, 2>();
  auto counts = lengths.accessor<int32_t, 1>();
  for (int64_t row = 0; row < ids.size(0); ++row) {
    const auto physical = [&](int64_t logical) {
      const int64_t column = logical / block_size;
      if (logical < 0 || column >= blocks.size(1) ||
          block_table[row_batch[row]][column] < 0) {
        return int32_t{-1};
      }
      return static_cast<int32_t>(block_table[row_batch[row]][column] *
                                      block_size +
                                  logical % block_size);
    };
    int32_t count = 0;
    for (int64_t col = 0; col < ids.size(1); ++col) {
      if (input[row][col] < 0) {
        continue;
      }
      for (int64_t member = 0; member < pool_size; ++member) {
        const int64_t token = input[row][col] * pool_size + member;
        if (token <= query_positions[row] && physical(token) >= 0) {
          output[row][count++] = physical(token);
        }
      }
    }
    const int64_t tail_begin =
        (query_positions[row] + 1) / pool_size * pool_size;
    const int64_t tail_count =
        select_tail ? (query_positions[row] + 1) % pool_size : 0;
    for (int64_t member = 0; member < tail_count; ++member) {
      const int32_t slot = physical(tail_begin + member);
      if (slot >= 0) {
        output[row][count++] = slot;
      }
    }
    counts[row] = count;
  }
  return {.physical_slots = std::move(slots),
          .context_lens = std::move(lengths)};
}

void check_forward_after_short_prefill(bool graph_decode,
                                       bool prepare_metadata = true,
                                       int64_t num_heads = kHeads,
                                       int64_t head_dim = 128) {
  torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const int64_t saved_block_size = KVCacheConfig::get_instance().block_size();
  KVCacheConfig::get_instance().block_size(16);
  ModelArgs args;
  args.model_type("glm5_next");
  args.hidden_size(4096);
  args.q_lora_rank(kRank);
  args.index_n_heads(num_heads);
  args.index_head_dim(head_dim);
  args.qk_rope_head_dim(0);
  args.index_topk(kTopk);
  args.index_kpool(4);
  args.index_kpool_compress(true);
  args.index_kpool_always_select_tail(true);
  args.max_position_embeddings(1048576);
  ParallelArgs parallel(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  Glm5NextKPoolIndexer full_indexer(
      args, QuantArgs(), parallel, nullptr, options);
  Glm5NextKPoolIndexer decode_indexer(
      args, QuantArgs(), parallel, nullptr, options);
  torch::manual_seed(929);
  for (auto& parameter : full_indexer->named_parameters()) {
    parameter.value().normal_(/*mean=*/0.0, /*std=*/0.1);
  }
  auto decode_parameters = decode_indexer->named_parameters();
  for (const auto& parameter : full_indexer->named_parameters()) {
    decode_parameters[parameter.key()].copy_(parameter.value());
  }
  const torch::Tensor hidden = torch::randn({41, 4096}, options);
  const torch::Tensor q_norm = torch::randn({41, kRank}, options);
  const torch::Tensor positions =
      torch::arange(41, options.dtype(torch::kInt32));
  AttentionMetadata meta{};
  meta.block_table =
      torch::tensor({{2, 0, 1, 3}}, options.dtype(torch::kInt32));
  meta.q_seq_lens = torch::tensor({41}, options.dtype(torch::kInt32));
  meta.kv_seq_lens = meta.q_seq_lens.clone();
  meta.q_seq_lens_vec = {41};
  meta.kv_seq_lens_vec = {41};
  meta.linear_state_indices = torch::tensor({1}, options.dtype(torch::kInt32));
  meta.is_prefill = true;
  if (prepare_metadata) {
    prepare_glm5_next_kpool_metadata(meta, device);
  }
  torch::Tensor full_cache = torch::zeros({4, 1, 4, head_dim}, options);
  torch::Tensor decode_cache = full_cache.clone();
  torch::Tensor full_tail = torch::zeros({2, 2, 4, head_dim}, options);
  torch::Tensor decode_tail = full_tail.clone();
  const auto [full_slots, full_lens] = full_indexer->forward(
      hidden, q_norm, positions, full_cache, full_tail, meta);
  // Decode crosses pool boundaries using the short prefill's caches.
  meta.q_seq_lens_vec = {7};
  meta.kv_seq_lens_vec = {7};
  meta.q_seq_lens.fill_(7);
  meta.kv_seq_lens.fill_(7);
  if (prepare_metadata) {
    prepare_glm5_next_kpool_metadata(meta, device);
  }
  decode_indexer->forward(hidden.narrow(0, 0, 7),
                          q_norm.narrow(0, 0, 7),
                          positions.narrow(0, 0, 7),
                          decode_cache,
                          decode_tail,
                          meta);
  meta.is_prefill = false;
  meta.q_seq_lens_vec = {1};
  meta.q_seq_lens.fill_(1);
  meta.enable_cuda_graph = graph_decode;
  for (int64_t i = 7; i < 41; ++i) {
    meta.kv_seq_lens_vec = {static_cast<int32_t>(i + 1)};
    meta.kv_seq_lens.fill_(i + 1);
    if (prepare_metadata) {
      prepare_glm5_next_kpool_metadata(meta, device);
    }
    const auto [slots, lens] =
        decode_indexer->forward(hidden.narrow(0, i, 1),
                                q_norm.narrow(0, i, 1),
                                positions.narrow(0, i, 1),
                                decode_cache,
                                decode_tail,
                                meta);
    EXPECT_TRUE(torch::equal(lens, full_lens.narrow(0, i, 1))) << i;
    EXPECT_TRUE(torch::equal(std::get<0>(slots.sort(-1)),
                             std::get<0>(full_slots.narrow(0, i, 1).sort(-1))))
        << i;
  }
  EXPECT_TRUE(torch::allclose(full_cache, decode_cache, 0.015625, 0.015625));
  KVCacheConfig::get_instance().block_size(saved_block_size);
}

TEST(Glm5NextKPoolIndexerTest, PrefillMatchesPagedPathAcrossRaggedTiles) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const int64_t saved_block_size = KVCacheConfig::get_instance().block_size();
  KVCacheConfig::get_instance().block_size(16);
  torch::manual_seed(72317);
  for (const int64_t length : {32, 33, 129, 512}) {
    for (const int64_t budget : {kTopk}) {
      for (const bool tail : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << length << "/" << budget << "/" << tail);
        ModelArgs args;
        args.model_type("glm5_next");
        args.hidden_size(kHidden);
        args.q_lora_rank(kRank);
        args.index_n_heads(kHeads);
        args.index_head_dim(128);
        args.qk_rope_head_dim(0);
        args.index_topk(budget);
        args.index_kpool(4);
        args.index_kpool_compress(true);
        args.index_kpool_always_select_tail(tail);
        args.max_position_embeddings(1048576);
        ParallelArgs parallel(
            /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
        Glm5NextKPoolIndexer indexer(
            args, QuantArgs(), parallel, nullptr, options);
        for (auto& parameter : indexer->named_parameters()) {
          parameter.value().normal_(/*mean=*/0.0, /*std=*/0.1);
        }
        const torch::Tensor hidden =
            torch::randn({length + 3, kHidden}, options);
        const torch::Tensor query = torch::randn({length + 3, kRank}, options);
        torch::Tensor positions = torch::cat(
            {torch::arange(7, length + 7, options.dtype(torch::kInt32)),
             torch::arange(3, options.dtype(torch::kInt32))});
        positions[length - 1] = -1;
        AttentionMetadata meta{};
        meta.q_seq_lens_vec = {static_cast<int32_t>(length), 3};
        meta.kv_seq_lens_vec = {static_cast<int32_t>(length + 7), 3};
        meta.q_seq_lens =
            torch::tensor(meta.q_seq_lens_vec, options.dtype(torch::kInt32));
        meta.kv_seq_lens =
            torch::tensor(meta.kv_seq_lens_vec, options.dtype(torch::kInt32));
        const int64_t pages = (length + 22) / 16;
        meta.block_table =
            torch::randperm(pages * 2, options.dtype(torch::kInt32))
                .view({2, pages});
        meta.block_table[0][1] = -1;
        meta.linear_state_indices =
            torch::tensor({2, 1}, options.dtype(torch::kInt32));
        prepare_glm5_next_kpool_metadata(meta, device);
        const torch::Tensor initial_cache =
            torch::randn({pages * 2, 1, 4, 128}, options);
        const torch::Tensor initial_tail =
            torch::randn({3, 2, 4, 128}, options);
        torch::Tensor baseline_cache = initial_cache.clone();
        torch::Tensor baseline_tail = initial_tail.clone();
        const auto [expected_slots, expected_lens] = indexer->forward(
            hidden, query, positions, baseline_cache, baseline_tail, meta);
        // Chunked-prefill alone must dispatch correctly, including padding and
        // an independently mapped request with no complete pools.
        meta.is_chunked_prefill = true;
        torch::Tensor cache = initial_cache.clone();
        torch::Tensor state = initial_tail.clone();
        const auto [slots, lens] =
            indexer->forward(hidden, query, positions, cache, state, meta);
        // Torch prefill pooling and the retained fused decode update have
        // different FP32 reduction orders; use the cache precision contract.
        EXPECT_TRUE(torch::allclose(cache, baseline_cache, 0.015625, 0.015625));
        EXPECT_TRUE(torch::equal(state, baseline_tail));
        EXPECT_TRUE(torch::equal(lens, expected_lens));
        EXPECT_TRUE(torch::equal(std::get<0>(slots.sort(-1)),
                                 std::get<0>(expected_slots.sort(-1))));
        EXPECT_TRUE(slots[length - 1].eq(-1).all().item<bool>());
        EXPECT_EQ(lens[length - 1].item<int32_t>(), 0);
      }
    }
  }
  KVCacheConfig::get_instance().block_size(saved_block_size);
}

TEST(Glm5NextKPoolIndexerTest, ScorePreservesInputsAndFp32HeadOrder) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  torch::manual_seed(72318);
  const torch::Tensor query =
      torch::randn({129, 32, 128}, options.dtype(torch::kBFloat16));
  const torch::Tensor weights = torch::randn({129, 32}, options);
  const torch::Tensor keys =
      torch::randn({137, 128}, options.dtype(torch::kBFloat16));
  const torch::Tensor query_before = query.clone();
  const torch::Tensor weights_before = weights.clone();
  const torch::Tensor keys_before = keys.clone();
  const torch::Tensor expected =
      (torch::relu(
           torch::matmul(query.cpu().to(torch::kFloat32),
                         keys.cpu().to(torch::kFloat32).transpose(0, 1)) *
           0.125) *
       weights.cpu().unsqueeze(-1))
          .sum(1);
  const torch::Tensor actual = score_keys(query, weights, keys, 0.125);
  EXPECT_TRUE(torch::allclose(actual.cpu(), expected, 0.001, 0.001));
  EXPECT_TRUE(torch::equal(query, query_before));
  EXPECT_TRUE(torch::equal(weights, weights_before));
  EXPECT_TRUE(torch::equal(keys, keys_before));
}

TEST(Glm5NextKPoolIndexerTest, ForwardMatchesTokenDecodeAfterShortPrefill) {
  check_forward_after_short_prefill(/*graph_decode=*/false);
}

TEST(Glm5NextKPoolIndexerTest, ForwardMatchesGraphDecodeAfterShortPrefill) {
  check_forward_after_short_prefill(/*graph_decode=*/true,
                                    /*prepare_metadata=*/false);
}

TEST(Glm5NextKPoolIndexerTest,
     BatchMetadataRebuildsRaggedAndCumulativeLengths) {
  const torch::Device device(torch::kCPU);
  AttentionMetadata meta{};
  meta.q_seq_lens = torch::tensor({2, 0, 3}, torch::kInt32);
  meta.kv_seq_lens = torch::tensor({5, 7, 9}, torch::kInt32);
  meta.q_seq_lens_vec = {0, 2, 2, 5};
  meta.kv_seq_lens_vec = {0, 5, 12, 21};
  prepare_glm5_next_kpool_metadata(meta, device);
  const auto first = meta.kpool_batch_metadata;
  EXPECT_EQ(first->q_seq_lens, (std::vector<int32_t>{2, 0, 3}));
  EXPECT_EQ(first->max_kv_len, 9);
  EXPECT_TRUE(torch::equal(first->row_batch,
                           torch::tensor({0, 0, 2, 2, 2}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(first->query_starts,
                           torch::tensor({0, 2, 2, 5}, torch::kInt64)));
  meta.q_seq_lens_vec = {1, 1, 1};
  meta.kv_seq_lens_vec = {6, 8, 10};
  prepare_glm5_next_kpool_metadata(meta, device);
  EXPECT_NE(first.get(), meta.kpool_batch_metadata.get());
  EXPECT_EQ(meta.kpool_batch_metadata->max_kv_len, 10);
  EXPECT_TRUE(
      torch::equal(meta.kpool_batch_metadata->row_batch, torch::arange(3)));
  EXPECT_EQ(first->max_kv_len, 9);
}

TEST(Glm5NextKPoolIndexerTest,
     PagedSelectionPreservesRaggedCausalityAcrossChunks) {
  torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::manual_seed(827);
  const torch::Tensor cache = torch::randn({6, 1, 4, 128}, options);
  const torch::Tensor table =
      torch::tensor({{3, 0, 4}, {2, 5, 1}}, options.dtype(torch::kInt32));
  const torch::Tensor q = torch::randn({5, kHeads, 128}, options);
  const torch::Tensor weights =
      torch::randn({5, kHeads}, options.dtype(torch::kFloat32));
  const torch::Tensor positions =
      torch::tensor({0, 3, 8, 39, 43}, options.dtype(torch::kInt32));
  const torch::Tensor rows =
      torch::tensor({0, 1, 0, 1, 0}, options.dtype(torch::kInt64));
  const double scale = 1.0 / std::sqrt(32.0 * 128.0);
  const torch::Tensor actual =
      glm5_next_kpool_select(q,
                             weights,
                             positions,
                             rows,
                             cache,
                             table,
                             /*max_kv_seq_len=*/44,
                             /*block_size=*/16,
                             /*index_kpool=*/4,
                             /*index_topk=*/kTopk,
                             scale,
                             /*workspace_bytes=*/2 * 11 * sizeof(float));
  const PoolHistory history =
      read_pool_history(cache,
                        table,
                        torch::tensor({44, 44}, options.dtype(torch::kInt32)),
                        /*max_kv_seq_len=*/44,
                        /*block_size=*/16,
                        /*index_kpool=*/4);
  torch::Tensor expected =
      torch::full({5, kSelected}, -1, options.dtype(torch::kInt64));
  for (int64_t i = 0; i < 5; ++i) {
    const int64_t completed = (positions[i].item<int32_t>() + 1) / 4;
    if (completed == 0) {
      continue;
    }
    const torch::Tensor keys =
        history.keys[rows[i].item<int64_t>()].narrow(0, 0, completed);
    const torch::Tensor scores =
        (torch::relu(torch::matmul(q.narrow(0, i, 1).to(torch::kFloat32),
                                   keys.to(torch::kFloat32).transpose(0, 1)) *
                     scale) *
         weights.narrow(0, i, 1).unsqueeze(-1))
            .sum(1);
    const int64_t count = std::min<int64_t>(kSelected, completed);
    expected[i]
        .narrow(0, 0, count)
        .copy_(std::get<1>(scores.topk(count, -1))[0]);
  }
  Device(device).synchronize_default_stream();
  EXPECT_TRUE(torch::equal(std::get<0>(actual.sort(-1)),
                           std::get<0>(expected.sort(-1))));
  const torch::Tensor unchunked = glm5_next_kpool_select(
      q, weights, positions, rows, cache, table, 44, 16, 4, kTopk, scale);
  EXPECT_TRUE(torch::equal(actual, unchunked));
}

TEST(Glm5NextKPoolIndexerTest,
     GeneralSelectionSupportsHeadTilesAndDenseTailPage) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::manual_seed(973);
  for (const int64_t heads : {kHeads}) {
    constexpr int64_t kDim = 128;
    const torch::Tensor q =
        torch::randn({40, heads, kDim * 2}, options).slice(2, 0, kDim * 2, 2);
    const torch::Tensor weights =
        torch::randn({40, heads * 2}, options.dtype(torch::kFloat32))
            .slice(1, 0, heads * 2, 2);
    const torch::Tensor cache = torch::randn({66, 1, 16, kDim}, options);
    const torch::Tensor cache_before = cache.clone();
    const torch::Tensor table =
        torch::randperm(66, options.dtype(torch::kInt32)).view({2, 33});
    const torch::Tensor positions =
        torch::cat({torch::arange(2019, 2052, options.dtype(torch::kInt32)),
                    torch::arange(5, 12, options.dtype(torch::kInt32))});
    const torch::Tensor rows =
        torch::cat({torch::zeros({33}, options.dtype(torch::kInt64)),
                    torch::ones({7}, options.dtype(torch::kInt64))});
    KPoolBatchMetadata batch;
    batch.q_seq_lens = {33, 7};
    batch.kv_seq_lens = {2052, 12};
    const double scale = 1.0 / std::sqrt(static_cast<double>(heads * kDim));
    const torch::Tensor expected = reference_selection(
        q, weights, positions, rows, cache, table, 16, 4, kSelected, scale);
    for (const int64_t chunk_rows : {1, 2}) {
      const torch::Tensor actual = glm5_next_kpool_select(
          q,
          weights,
          positions,
          rows,
          cache,
          table,
          2052,
          64,
          4,
          kTopk,
          scale,
          /*workspace_bytes=*/chunk_rows * 513 * sizeof(float));
      EXPECT_TRUE(torch::equal(std::get<0>(actual.sort(-1)),
                               std::get<0>(expected.sort(-1))))
          << heads;
    }
    EXPECT_TRUE(torch::equal(cache, cache_before));
  }
}

TEST(Glm5NextKPoolIndexerTest, GeneralSelectionHandlesEmptyTiesAndLongHistory) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  for (const int64_t pools : {0, 17, 65537}) {
    const int64_t pages = (pools + 15) / 16;
    const torch::Tensor cache = torch::ones({pages, 1, 16, 128}, options);
    const torch::Tensor table =
        torch::arange(pages, options.dtype(torch::kInt32)).view({1, pages});
    const torch::Tensor q = torch::ones({1, kHeads, 128}, options);
    const torch::Tensor weights =
        torch::ones({1, kHeads}, options.dtype(torch::kFloat32));
    const torch::Tensor positions =
        torch::full({1}, pools * 4 - 1, options.dtype(torch::kInt32));
    const torch::Tensor rows = torch::zeros({1}, options.dtype(torch::kInt64));
    const torch::Tensor actual = glm5_next_kpool_select(q,
                                                        weights,
                                                        positions,
                                                        rows,
                                                        cache,
                                                        table,
                                                        pools * 4,
                                                        64,
                                                        4,
                                                        kTopk,
                                                        1.0);
    torch::Tensor expected =
        torch::full({1, kSelected}, -1, options.dtype(torch::kInt64));
    const int64_t count = std::min<int64_t>(pools, kSelected);
    EXPECT_EQ(actual.ge(0).sum().item<int64_t>(), count);
    const torch::Tensor valid = actual.masked_select(actual >= 0);
    EXPECT_TRUE(valid.lt(pools).all().item<bool>());
    if (count > 1) {
      const torch::Tensor sorted = std::get<0>(valid.sort());
      EXPECT_TRUE(
          sorted.slice(0, 1).ne(sorted.slice(0, 0, -1)).all().item<bool>());
    }
    const torch::Tensor empty =
        glm5_next_kpool_select(q.narrow(0, 0, 0),
                               weights.narrow(0, 0, 0),
                               positions.narrow(0, 0, 0),
                               rows.narrow(0, 0, 0),
                               cache,
                               table,
                               pools * 4,
                               64,
                               4,
                               kTopk,
                               1.0);
    EXPECT_EQ(empty.size(0), 0);
    EXPECT_EQ(empty.size(1), kSelected);
  }
}

TEST(Glm5NextKPoolIndexerTest, PagedGraphReplayUsesUpdatedSequenceLengths) {
  torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::Tensor cache = torch::ones({1, 1, 16, 128}, options);
  const torch::Tensor table =
      torch::zeros({1, 1}, options.dtype(torch::kInt32));
  const torch::Tensor q = torch::ones({1, kHeads, 128}, options);
  const torch::Tensor weights =
      torch::ones({1, kHeads}, options.dtype(torch::kFloat32));
  torch::Tensor positions = torch::tensor({0}, options.dtype(torch::kInt32));
  const torch::Tensor rows = torch::zeros({1}, options.dtype(torch::kInt64));
  const auto select = [&]() {
    return glm5_next_kpool_select(q,
                                  weights,
                                  positions,
                                  rows,
                                  cache,
                                  table,
                                  64,
                                  64,
                                  4,
                                  kTopk,
                                  /*softmax_scale=*/1.0);
  };
  select();
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  torch::Tensor captured;
  {
    torch_mlu::mlu::MLUStreamGuard guard(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, device.index()));
    graph.capture_begin();
    captured = select();
    graph.capture_end();
  }
  positions.fill_(7);
  graph.replay();
  torch_mlu::synchronize();
  EXPECT_EQ(captured.ge(0).sum().item<int64_t>(), 2);
  EXPECT_TRUE(
      torch::equal(std::get<0>(captured.masked_select(captured >= 0).sort()),
                   torch::tensor({0, 1}, options.dtype(torch::kInt64))));
  positions.fill_(0);
  graph.replay();
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(
      captured, torch::full({1, kSelected}, -1, options.dtype(torch::kInt64))));
}

TEST(Glm5NextKPoolIndexerTest, ChunkBoundariesPreserveCompressedAndTailCaches) {
  torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  constexpr int64_t kDim = 128;
  for (const int64_t kPool : {4}) {
    SCOPED_TRACE("D=" + std::to_string(kDim) + " P=" + std::to_string(kPool));
    const int64_t kBlock = 4 * kPool;
    const int64_t tokens = 5 * kPool + 3;
    torch::manual_seed(719);
    const torch::Tensor keys = torch::randn({tokens, kDim}, options);
    const torch::Tensor gates = torch::randn({tokens, kDim}, options);
    const torch::Tensor ape =
        torch::randn({kPool, kDim}, options.dtype(torch::kFloat32));
    const torch::Tensor hadamard = util::create_hadamard_matrix(
        kDim, torch::kFloat32, device, /*normalize=*/true);
    const torch::Tensor table =
        torch::tensor({{1, 0, 2}}, options.dtype(torch::kInt32));
    const torch::Tensor tail_ids =
        torch::tensor({1}, options.dtype(torch::kInt32));
    const torch::Tensor positions =
        torch::arange(tokens, options.dtype(torch::kInt32));
    const torch::Tensor rows =
        torch::zeros({tokens}, options.dtype(torch::kInt64));
    torch::Tensor full_cache = torch::zeros({3, 1, 4, kDim}, options);
    torch::Tensor full_tail = torch::zeros({2, 2, kPool, kDim}, options);
    torch::Tensor split_cache = full_cache.clone();
    torch::Tensor split_tail = full_tail.clone();
    kernel::mlu::update_kpool(
        keys,
        gates,
        ape,
        hadamard,
        full_cache,
        full_tail,
        tail_ids,
        table,
        positions,
        rows,
        torch::tensor({int64_t{0}, tokens}, options.dtype(torch::kInt64)),
        kBlock,
        kPool);
    int64_t offset = 0;
    for (const int64_t length :
         {int64_t{1}, kPool, 2 * kPool + 1, 2 * kPool + 1}) {
      kernel::mlu::update_kpool(
          keys.narrow(0, offset, length),
          gates.narrow(0, offset, length),
          ape,
          hadamard,
          split_cache,
          split_tail,
          tail_ids,
          table,
          positions.narrow(0, offset, length),
          rows.narrow(0, offset, length),
          torch::tensor({int64_t{0}, length}, options.dtype(torch::kInt64)),
          kBlock,
          kPool);
      offset += length;
    }
    Device(device).synchronize_default_stream();
    EXPECT_TRUE(torch::equal(full_tail, split_tail));
    EXPECT_TRUE(torch::equal(full_cache, split_cache));
    const torch::Tensor expected_pools =
        reference_compression(keys.narrow(0, 0, tokens / kPool * kPool),
                              gates.narrow(0, 0, tokens / kPool * kPool),
                              ape,
                              hadamard,
                              kPool);
    const PoolHistory history =
        read_pool_history(split_cache,
                          table,
                          torch::tensor({tokens}, options.dtype(torch::kInt32)),
                          tokens,
                          kBlock,
                          kPool);
    EXPECT_TRUE(torch::allclose(history.keys[0].narrow(0, 0, tokens / kPool),
                                expected_pools,
                                /*rtol=*/0.015625,
                                /*atol=*/0.015625));
  }
}

TEST(Glm5NextKPoolIndexerTest, PaddedMlaRowsSkipCacheWrites) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const auto ints = options.dtype(torch::kInt32);
  auto kv = torch::randn({3, 1, 1, 576}, options);
  auto sin = torch::zeros({16, 64}, options);
  auto cos = torch::ones_like(sin);
  auto gamma = torch::ones({512}, options);
  auto positions = torch::zeros({3}, ints);
  auto cache = torch::full({2, 1, 16, 576}, 7, options);
  auto slots = torch::tensor({16, -1, -1}, ints).view({3, 1});
  kernel::mlu::fused_mla_kv(kv,
                            sin,
                            cos,
                            positions,
                            gamma,
                            cache,
                            std::nullopt,
                            slots,
                            std::nullopt,
                            std::nullopt,
                            /*quant_mode=*/"none",
                            /*is_paged_cache=*/true,
                            /*eps=*/1e-5,
                            /*interleaved=*/true);
  EXPECT_TRUE(torch::all(cache[0] == 7).item<bool>());
  EXPECT_TRUE(torch::all(cache[1].narrow(1, 1, 15) == 7).item<bool>());
  EXPECT_FALSE(torch::all(cache[1].select(1, 0) == 7).item<bool>());

  // Sparse attention must overwrite even an uninitialized output row when
  // KPool reports zero context for a virtual request.
  Attention attention(/*num_heads=*/16,
                      /*head_size=*/576,
                      /*num_kv_heads=*/1,
                      /*v_head_dim=*/512,
                      /*sliding_window=*/-1,
                      /*scale=*/0.04f,
                      /*use_fused_mla_qkv=*/true,
                      /*enable_lighting_indexer=*/true,
                      /*enable_mla=*/true);
  AttentionMetadata metadata{};
  metadata.block_table = torch::tensor({16, 0, 0}, ints).view({3, 1});
  metadata.kv_seq_lens = torch::tensor({1, 0, 0}, ints);
  metadata.max_seq_len = 1;
  metadata.compute_dtype = "half";
  auto query = torch::zeros({3, 16 * 576}, options);
  auto output = torch::full(
      {3, 16 * 512}, std::numeric_limits<float>::quiet_NaN(), options);
  std::optional<torch::Tensor> lse;
  attention->decoder_forward(query,
                             output,
                             lse,
                             cache,
                             std::nullopt,
                             metadata,
                             std::nullopt,
                             std::nullopt,
                             /*return_lse=*/false);
  EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
  EXPECT_TRUE(torch::equal(output.narrow(0, 1, 2),
                           torch::zeros_like(output.narrow(0, 1, 2))));
}

TEST(Glm5NextKPoolIndexerTest, GraphCacheUpdateMatchesReference) {
  torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto ints = options.dtype(torch::kInt32);
  constexpr int64_t kDim = 128;
  constexpr int64_t kPool = 4;
  constexpr int64_t kBlock = 16;
  torch::manual_seed(731);
  torch::Tensor keys = torch::randn({4, kDim}, options);
  torch::Tensor gates = torch::randn({4, kDim}, options);
  const torch::Tensor ape =
      torch::randn({kPool, kDim}, options.dtype(torch::kFloat32));
  const torch::Tensor hadamard = util::create_hadamard_matrix(
      kDim, torch::kFloat32, device, /*normalize=*/true);
  const torch::Tensor table =
      torch::tensor({{1, 0}, {3, 2}, {0, 0}, {0, 0}}, ints);
  const torch::Tensor tail_ids = torch::tensor({2, 1, 0, 0}, ints);
  const torch::Tensor rows =
      torch::tensor({0, 1, 2, 3}, options.dtype(torch::kInt64));
  const torch::Tensor starts =
      torch::tensor({0, 1, 2, 3, 4}, options.dtype(torch::kInt64));
  torch::Tensor positions = torch::zeros({4}, ints);
  torch::Tensor eager_cache =
      torch::zeros({4, 1, kBlock / kPool, kDim}, options);
  torch::Tensor graph_cache = eager_cache.clone();
  torch::Tensor eager_tail = torch::zeros({3, 2, kPool, kDim}, options);
  torch::Tensor graph_tail = eager_tail.clone();
  const auto update = [&](torch::Tensor& cache, torch::Tensor& tail) {
    kernel::mlu::update_kpool(keys,
                              gates,
                              ape,
                              hadamard,
                              cache,
                              tail,
                              tail_ids,
                              table,
                              positions,
                              rows,
                              starts,
                              kBlock,
                              kPool);
  };
  update(graph_cache, graph_tail);
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  {
    torch_mlu::mlu::MLUStreamGuard guard(
        torch_mlu::getStreamFromPool(false, 0));
    graph.capture_begin();
    update(graph_cache, graph_tail);
    graph.capture_end();
  }
  graph_cache.zero_();
  graph_tail.zero_();
  for (int64_t position = 0; position < 8 * kPool; ++position) {
    positions.fill_(position);
    keys.normal_();
    gates.normal_();
    for (int64_t request = 0; request < 2; ++request) {
      const int64_t tail_id = request == 0 ? 2 : 1;
      eager_tail[tail_id][0][position % kPool].copy_(keys[request]);
      eager_tail[tail_id][1][position % kPool].copy_(gates[request]);
      if (position % kPool == kPool - 1) {
        const int64_t page = position < kBlock ? 2 * request + 1 : 2 * request;
        eager_cache[page][0][position / kPool % (kBlock / kPool)].copy_(
            reference_compression(eager_tail[tail_id][0],
                                  eager_tail[tail_id][1],
                                  ape,
                                  hadamard,
                                  kPool)[0]);
      }
    }
    torch_mlu::synchronize();
    graph.replay();
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::equal(eager_tail, graph_tail)) << position;
    EXPECT_TRUE(torch::allclose(eager_cache, graph_cache, 0.015625, 0.015625))
        << position;
  }
}

TEST(Glm5NextKPoolIndexerTest,
     NormalizesBFloat16ProjectionWithFloat32LayerNorm) {
  constexpr int64_t kHeadDim = 128;
  constexpr double kEps = 1e-6;
  torch::Device device(Platform::type_torch(), 0);
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  torch::Tensor key = torch::randn({2, kHeadDim}, bf16_options).contiguous();
  RMSNorm key_norm(kHeadDim, kEps, fp32_options);
  key_norm->set_layernorm_mode();
  auto parameters = key_norm->named_parameters();
  parameters["weight"].fill_(1.0f);
  parameters["bias"].zero_();

  const torch::Tensor actual = glm5_next_kpool_normalize_key(key, key_norm);
  Device(device).synchronize_default_stream();

  EXPECT_EQ(actual.scalar_type(), torch::kBFloat16);
  EXPECT_EQ(actual.sizes(), key.sizes());
  EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
}

TEST(Glm5NextKPoolIndexerTest, AppliesReluBeforeReducingQueryHeads) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  torch::Tensor query = torch::zeros({1, kHeads, 128}, opts);
  query[0][0][0] = 1;
  query[0][1][0] = -1;
  const torch::Tensor head_weights =
      torch::ones({1, kHeads}, opts.dtype(torch::kFloat32));
  torch::Tensor pooled_key = torch::zeros({1, 128}, opts);
  pooled_key[0][0] = 1;

  const torch::Tensor actual =
      score_keys(query, head_weights, pooled_key, /*softmax_scale=*/1.0);

  ASSERT_EQ(actual.sizes(), (torch::IntArrayRef{1, 1}));
  EXPECT_FLOAT_EQ(actual.item<float>(), 1.0f);
}

TEST(Glm5NextKPoolIndexerTest,
     ExpansionSupportsLargeStridedShapesAndCrossBlockPools) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
  torch::manual_seed(3147);
  for (const auto& shape : std::vector<std::vector<int64_t>>{
           {65, 512, 4, 16}, {257, 512, 4, 64}, {2, 512, 4, 16}}) {
    const int64_t queries = shape[0];
    const int64_t pools = shape[1];
    const int64_t pool_size = shape[2];
    const int64_t block_size = shape[3];
    const int64_t history_pools = std::max<int64_t>(64, pools * 2);
    const int64_t pages =
        (history_pools * pool_size + block_size - 1) / block_size;
    torch::Tensor ids =
        torch::randint(history_pools, {queries, pools * 2}, options)
            .slice(1, 0, pools * 2, 2);
    ids.slice(1, 1, pools, 3).fill_(-1);
    ids.select(1, 2).copy_(ids.select(1, 0));
    const torch::Tensor before = ids.clone();
    const torch::Tensor positions =
        torch::randint(history_pools * pool_size, {queries * 2}, options)
            .slice(0, 0, queries * 2, 2);
    const torch::Tensor rows = torch::arange(queries, options).remainder(2);
    const torch::Tensor table =
        torch::randperm(4 * pages, options.dtype(torch::kInt32))
            .view({2, pages * 2})
            .slice(1, 0, pages * 2, 2);
    for (const bool tail : {false, true}) {
      const auto expected = reference_expansion(ids,
                                                positions,
                                                rows,
                                                table,
                                                block_size,
                                                pools * pool_size,
                                                pool_size,
                                                tail);
      const auto actual = kernel::mlu::expand_kpool(ids,
                                                    positions,
                                                    rows,
                                                    table,
                                                    block_size,
                                                    pools * pool_size,
                                                    pool_size,
                                                    tail);
      EXPECT_TRUE(
          torch::equal(actual.physical_slots.cpu(), expected.physical_slots));
      EXPECT_TRUE(
          torch::equal(actual.context_lens.cpu(), expected.context_lens));
    }
    EXPECT_TRUE(torch::equal(ids, before));
  }
}

TEST(Glm5NextKPoolIndexerTest, ExpansionHandlesEmptySelections) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  for (const int64_t queries : {0, 3}) {
    const torch::Tensor ids = torch::full({queries, kSelected}, -1, options);
    const torch::Tensor positions = torch::zeros({queries}, options);
    const torch::Tensor rows = torch::zeros({queries}, options);
    const torch::Tensor table = torch::tensor({{3}}, options);
    for (const int64_t pool_size : {4}) {
      const auto actual =
          kernel::mlu::expand_kpool(ids,
                                    positions,
                                    rows,
                                    table,
                                    /*block_size=*/16,
                                    /*index_topk=*/kTopk,
                                    pool_size,
                                    /*always_select_tail=*/true);
      const auto expected = reference_expansion(ids,
                                                positions,
                                                rows,
                                                table,
                                                /*block_size=*/16,
                                                /*index_topk=*/kTopk,
                                                pool_size,
                                                /*select_tail=*/true);
      EXPECT_TRUE(
          torch::equal(actual.physical_slots.cpu(), expected.physical_slots));
      EXPECT_TRUE(
          torch::equal(actual.context_lens.cpu(), expected.context_lens));
    }
  }
}

TEST(Glm5NextKPoolIndexerTest,
     ExpansionGraphReplayUsesLivePoolsPositionsAndTables) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
  torch::Tensor ids = torch::full({2, kSelected}, -1, options);
  ids.narrow(1, 0, 17).copy_(torch::arange(17, options));
  torch::Tensor positions = torch::tensor({70, 65}, options);
  torch::Tensor rows = torch::tensor({0, 1}, options);
  torch::Tensor table =
      torch::tensor({{3, 1}, {2, 0}}, options.dtype(torch::kInt32));
  const auto expand = [&]() {
    return kernel::mlu::expand_kpool(ids,
                                     positions,
                                     rows,
                                     table,
                                     /*block_size=*/64,
                                     /*index_topk=*/kTopk,
                                     /*index_kpool=*/4,
                                     /*always_select_tail=*/true);
  };
  expand();
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  kernel::mlu::KPoolSelection captured;
  {
    torch_mlu::mlu::MLUStreamGuard guard(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, device.index()));
    graph.capture_begin();
    captured = expand();
    graph.capture_end();
  }
  ids.slice(1, 0, 17, 2).fill_(-1);
  positions.copy_(torch::tensor({68, 66}, options));
  rows.copy_(torch::tensor({1, 0}, options));
  table.copy_(torch::tensor({{0, 2}, {1, 3}}, table.options()));
  graph.replay();
  torch_mlu::synchronize();
  const auto expected = reference_expansion(ids,
                                            positions,
                                            rows,
                                            table,
                                            /*block_size=*/64,
                                            /*index_topk=*/kTopk,
                                            /*pool_size=*/4,
                                            /*select_tail=*/true);
  EXPECT_TRUE(
      torch::equal(captured.physical_slots.cpu(), expected.physical_slots));
  EXPECT_TRUE(torch::equal(captured.context_lens.cpu(), expected.context_lens));
}

}  // namespace

TEST(Glm5NextKPoolIndexerTest, DecodeReadsPdStateRestoredAfterGraphCapture) {
  torch::NoGradGuard no_grad;
  torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto ints = options.dtype(torch::kInt32);
  const int64_t old_block_size = KVCacheConfig::get_instance().block_size();
  KVCacheConfig::get_instance().block_size(16);
  ModelArgs args;
  args.model_type("glm5_next")
      .hidden_size(kHidden)
      .q_lora_rank(kRank)
      .index_n_heads(kHeads)
      .index_head_dim(128)
      .qk_rope_head_dim(0)
      .index_topk(kTopk)
      .index_kpool(4)
      .index_kpool_compress(true)
      .index_kpool_always_select_tail(true)
      .max_position_embeddings(1048576);
  ParallelArgs parallel(0, 1, nullptr);
  Glm5NextKPoolIndexer prefill(args, QuantArgs(), parallel, nullptr, options);
  Glm5NextKPoolIndexer decode(args, QuantArgs(), parallel, nullptr, options);
  torch::manual_seed(20260916);
  auto decode_parameters = decode->named_parameters();
  for (auto& parameter : prefill->named_parameters()) {
    parameter.value().normal_(0.0, 0.1);
    decode_parameters[parameter.key()].copy_(parameter.value());
  }
  const torch::Tensor hidden = torch::randn({48, kHidden}, options);
  const torch::Tensor queries = torch::randn({48, kRank}, options);
  const torch::Tensor positions = torch::arange(48, ints);
  torch::Tensor prefill_cache = torch::zeros({4, 1, 4, 128}, options);
  torch::Tensor prefill_tail = torch::zeros({3, 2, 8, 128}, options);
  torch::Tensor decode_cache = torch::zeros({6, 1, 4, 128}, options);
  torch::Tensor decode_tail = torch::zeros_like(prefill_tail);
  // Exercise the model metadata path with a one-token initial context. Replay
  // below crosses multiple pool/page boundaries without recapturing.
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 1;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 1;
  params.enable_graph = true;
  params.attention.device.block_tables = torch::tensor({{4, 1, 5, 3}}, ints);
  params.embedding.linear_state_ids = {2};
  params.embedding.linear_state_indices = torch::tensor({2}, ints);
  params.attention.device.q_seq_lens = torch::tensor({0, 1}, ints);
  params.attention.device.kv_seq_lens = torch::tensor({0, 1}, ints);
  params.attention.host.q_seq_lens = {0, 1};
  params.attention.host.kv_seq_lens = {0, 1};
  mlu::model::Glm5NextGraphMetadataState graph_metadata;
  mlu::model::Glm5NextGraphMetadata::prepare(
      &graph_metadata, positions.narrow(0, 0, 1), params);
  AttentionMetadata& meta = *params.attn_metadata;
  torch::Tensor live_hidden = hidden.narrow(0, 0, 1).clone();
  torch::Tensor live_queries = queries.narrow(0, 0, 1).clone();
  torch::Tensor live_positions = positions.narrow(0, 0, 1).clone();
  decode->forward(live_hidden,
                  live_queries,
                  live_positions,
                  decode_cache,
                  decode_tail,
                  meta);
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  std::tuple<torch::Tensor, torch::Tensor> output;
  {
    torch_mlu::mlu::MLUStreamGuard guard(
        torch_mlu::getStreamFromPool(false, 0));
    graph.capture_begin();
    output = decode->forward(live_hidden,
                             live_queries,
                             live_positions,
                             decode_cache,
                             decode_tail,
                             meta);
    graph.capture_end();
  }
  // Reuse the captured D buffers for requests ending inside a pool, exactly
  // on a page boundary, and after a page boundary. P has different page/slot
  // IDs.
  for (int64_t prompt_length : {1, 7, 16, 19, 33}) {
    SCOPED_TRACE("prompt_length=" + std::to_string(prompt_length));
    AttentionMetadata prefill_meta{};
    prefill_meta.block_table = torch::tensor({{2, 0, 1, 3}}, ints);
    prefill_meta.linear_state_indices = torch::tensor({1}, ints);
    prefill_meta.q_seq_lens = torch::tensor({prompt_length}, ints);
    prefill_meta.kv_seq_lens = prefill_meta.q_seq_lens.clone();
    prefill_meta.q_seq_lens_vec = {static_cast<int32_t>(prompt_length)};
    prefill_meta.kv_seq_lens_vec = prefill_meta.q_seq_lens_vec;
    prefill_meta.is_prefill = true;
    prefill_cache.zero_();
    prefill_tail.zero_();
    prefill->forward(hidden.narrow(0, 0, prompt_length),
                     queries.narrow(0, 0, prompt_length),
                     positions.narrow(0, 0, prompt_length),
                     prefill_cache,
                     prefill_tail,
                     prefill_meta);

    // PD restores only allocated KV pages and the complete request tail, not
    // layer-local state. Poison D first to detect stale state between requests.
    decode_cache.fill_(17);
    decode_tail.fill_(17);
    decode_cache[4].copy_(prefill_cache[2]);
    if (prompt_length > 16) {
      decode_cache[1].copy_(prefill_cache[0]);
    }
    decode_tail[2].copy_(prefill_tail[1]);

    const int64_t total_length = prompt_length + 5;
    torch::Tensor reference_cache = torch::zeros_like(decode_cache);
    torch::Tensor reference_tail = torch::zeros_like(decode_tail);
    prefill_meta.block_table = meta.block_table;
    prefill_meta.linear_state_indices = meta.linear_state_indices;
    prefill_meta.q_seq_lens.fill_(total_length);
    prefill_meta.kv_seq_lens.fill_(total_length);
    prefill_meta.q_seq_lens_vec = {static_cast<int32_t>(total_length)};
    prefill_meta.kv_seq_lens_vec = prefill_meta.q_seq_lens_vec;
    const auto expected = prefill->forward(hidden.narrow(0, 0, total_length),
                                           queries.narrow(0, 0, total_length),
                                           positions.narrow(0, 0, total_length),
                                           reference_cache,
                                           reference_tail,
                                           prefill_meta);
    for (int64_t token = prompt_length; token < total_length; ++token) {
      live_hidden.copy_(hidden.narrow(0, token, 1));
      live_queries.copy_(queries.narrow(0, token, 1));
      live_positions.copy_(positions.narrow(0, token, 1));
      meta.kv_seq_lens.fill_(token + 1);
      torch_mlu::synchronize();
      graph.replay();
      torch_mlu::synchronize();
      EXPECT_TRUE(torch::equal(std::get<1>(output),
                               std::get<1>(expected).narrow(0, token, 1)));
      EXPECT_TRUE(torch::equal(
          std::get<0>(std::get<0>(output).sort(-1)),
          std::get<0>(std::get<0>(expected).narrow(0, token, 1).sort(-1))))
          << "token=" << token;
    }
  }
  KVCacheConfig::get_instance().block_size(old_block_size);
}

TEST(Glm5NextKPoolIndexerTest, VerifyGraphRejectsAndResumesFromRestoredTail) {
  torch::NoGradGuard no_grad;
  torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto ints = options.dtype(torch::kInt32);
  const int64_t old_block_size = KVCacheConfig::get_instance().block_size();
  KVCacheConfig::get_instance().block_size(16);
  ModelArgs args;
  args.model_type("glm5_next")
      .hidden_size(kHidden)
      .q_lora_rank(kRank)
      .index_n_heads(kHeads)
      .index_head_dim(128)
      .qk_rope_head_dim(0)
      .index_topk(kTopk)
      .index_kpool(4)
      .index_kpool_compress(true)
      .index_kpool_always_select_tail(true)
      .max_position_embeddings(1048576);
  ParallelArgs parallel(0, 1, nullptr);
  Glm5NextKPoolIndexer indexer(args, QuantArgs(), parallel, nullptr, options);
  torch::manual_seed(1203);
  for (auto& parameter : indexer->named_parameters()) {
    parameter.value().normal_(0.0, 0.1);
  }
  torch::Tensor history = torch::randn({64, kHidden}, options);
  torch::Tensor queries = torch::randn({64, kRank}, options);
  torch::Tensor positions = torch::arange(64, ints);
  torch::Tensor cache = torch::zeros({4, 1, 4, 128}, options);
  torch::Tensor tail = torch::zeros({4, 2, 16, 128}, options);
  AttentionMetadata meta{};
  meta.block_table = torch::tensor({{2, 0, 1, 3}}, ints);
  meta.linear_state_indices = torch::tensor({1}, ints);
  meta.q_seq_lens = torch::tensor({6}, ints);
  meta.kv_seq_lens = torch::tensor({6}, ints);
  meta.q_seq_lens_vec = {6};
  meta.kv_seq_lens_vec = {6};
  meta.is_prefill = true;
  indexer->forward(history.narrow(0, 0, 6),
                   queries.narrow(0, 0, 6),
                   positions.narrow(0, 0, 6),
                   cache,
                   tail,
                   meta);

  torch::Tensor live_hidden = history.narrow(0, 6, 6).clone();
  torch::Tensor live_queries = queries.narrow(0, 6, 6).clone();
  torch::Tensor live_positions = positions.narrow(0, 6, 6).clone();
  meta.is_prefill = false;
  meta.is_spec_verify = true;
  meta.kv_seq_lens.fill_(12);
  meta.kv_seq_lens_vec = {12};
  prepare_glm5_next_kpool_metadata(meta, device);
  meta.enable_cuda_graph = true;
  const torch::Tensor saved_cache = cache.clone();
  const torch::Tensor saved_tail = tail.clone();
  indexer->forward(
      live_hidden, live_queries, live_positions, cache, tail, meta);
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  std::tuple<torch::Tensor, torch::Tensor> output;
  {
    torch_mlu::mlu::MLUStreamGuard guard(
        torch_mlu::getStreamFromPool(false, 0));
    graph.capture_begin();
    output = indexer->forward(
        live_hidden, live_queries, live_positions, cache, tail, meta);
    graph.capture_end();
  }
  cache.copy_(saved_cache);
  tail.copy_(saved_tail);
  int64_t committed = 6;
  // One committed base token means all five draft tokens were rejected.
  for (const int64_t accepted : {1, 1, 3, 6, 2}) {
    SCOPED_TRACE("committed=" + std::to_string(committed));
    history.narrow(0, committed, 6).normal_();
    queries.narrow(0, committed, 6).normal_();
    live_hidden.copy_(history.narrow(0, committed, 6));
    live_queries.copy_(queries.narrow(0, committed, 6));
    live_positions.copy_(positions.narrow(0, committed, 6));
    meta.kv_seq_lens.fill_(committed + 6);

    // A fresh full-prefix execution knows nothing about rejected history.
    torch::Tensor reference_cache = torch::zeros_like(cache);
    torch::Tensor reference_tail = torch::zeros_like(tail);
    AttentionMetadata reference{};
    reference.block_table = meta.block_table;
    reference.linear_state_indices = meta.linear_state_indices;
    reference.q_seq_lens = torch::tensor({committed + 6}, ints);
    reference.kv_seq_lens = reference.q_seq_lens.clone();
    reference.q_seq_lens_vec = {static_cast<int32_t>(committed + 6)};
    reference.kv_seq_lens_vec = reference.q_seq_lens_vec;
    reference.is_prefill = true;
    const auto expected =
        indexer->forward(history.narrow(0, 0, committed + 6),
                         queries.narrow(0, 0, committed + 6),
                         positions.narrow(0, 0, committed + 6),
                         reference_cache,
                         reference_tail,
                         reference);
    torch_mlu::synchronize();
    graph.replay();
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::equal(std::get<0>(output),
                             std::get<0>(expected).narrow(0, committed, 6)));
    EXPECT_TRUE(torch::equal(std::get<1>(output),
                             std::get<1>(expected).narrow(0, committed, 6)));
    committed += accepted;
    if (committed == 7) {
      // Restore the whole speculative ring at a different request slot and
      // relocate pages. Visibility remains governed by the committed length.
      const torch::Tensor host_tail = tail[1].cpu();
      const torch::Tensor host_cache = cache.cpu();
      tail.zero_();
      tail[2].copy_(host_tail);
      cache[1].copy_(host_cache[2]);
      cache[3].copy_(host_cache[0]);
      cache[0].copy_(host_cache[1]);
      cache[2].copy_(host_cache[3]);
      meta.linear_state_indices.fill_(2);
      meta.block_table.copy_(torch::tensor({{1, 3, 0, 2}}, ints));
    }
  }
  KVCacheConfig::get_instance().block_size(old_block_size);
}

TEST(Glm5NextKPoolIndexerTest,
     SpeculativeInputRowsRetainLogicalRequestOwnership) {
  torch::Device device(Platform::type_torch(), 0);
  ForwardInput input;
  input.token_ids_host = torch::tensor({11, 22}, torch::kInt32);
  input.positions_host = torch::tensor({6, 9}, torch::kInt32);
  auto& params = input.input_params;
  params.meta.num_sequences = 2;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.attention.host.kv_seq_lens = {0, 7, 17};
  params.attention.host.block_tables =
      torch::tensor({{2, 1}, {3, 0}}, torch::kInt32);
  params.embedding.linear_state_indices = torch::tensor({1, 2}, torch::kInt32);
  params.linear_state_validity_mask = {1, 1};
  const auto context = specBuilder::make_decode_row_context(input);
  specBuilder::DecodeBuildBuffers buffers;
  const std::vector<int32_t> query_lens{6, 3};
  for (int32_t seq = 0; seq < 2; ++seq) {
    for (int32_t offset = 0; offset < query_lens[seq]; ++offset) {
      specBuilder::RowSpec row;
      row.seq_id = seq;
      row.position_offset = offset;
      row.append_q_len_one = true;
      row.append_block_table = true;
      specBuilder::append_decode_row(context, row, 16, buffers);
    }
  }
  params.meta.num_sequences = 9;
  specBuilder::update_input_params(params,
                                   buffers,
                                   1,
                                   std::move(buffers.out_q_seq_lens),
                                   std::move(buffers.out_q_cu_seq_lens),
                                   buffers.meta.kv_max_seq_len,
                                   std::move(buffers.out_kv_seq_lens),
                                   true);
  params.attention.host.kpool_query_lens = query_lens;
  params.enable_graph = false;
  params.attention.rebuild_device_buffer(device);
  ModelInputParams ready = params.to(device);
  AttentionMetadata metadata =
      AttentionMetadataBuilder::build(ready, true, {}, device);
  prepare_glm5_next_kpool_metadata(metadata, device);
  const auto& batch = *metadata.kpool_batch_metadata;
  EXPECT_EQ(batch.q_seq_lens, query_lens);
  EXPECT_EQ(batch.kv_seq_lens, (std::vector<int32_t>{12, 12}));
  EXPECT_TRUE(torch::equal(batch.query_starts.cpu(),
                           torch::tensor({0, 6, 9}, torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(batch.row_batch.cpu(),
                   torch::tensor({0, 0, 0, 0, 0, 0, 1, 1, 1}, torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(batch.block_table.cpu(),
                   input.input_params.attention.host.block_tables.index_select(
                       0, torch::tensor({0, 6}, torch::kInt64))));
  EXPECT_TRUE(torch::equal(batch.tail_indices.cpu(),
                           torch::tensor({1, 2}, torch::kInt32)));
}

TEST(Glm5NextKPoolIndexerTest, GraphReplaysRaggedRequestsAndMasksPaddingState) {
  torch::NoGradGuard no_grad;
  torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto ints = options.dtype(torch::kInt32);
  const int64_t old_block_size = KVCacheConfig::get_instance().block_size();
  KVCacheConfig::get_instance().block_size(16);
  ModelArgs args;
  args.model_type("glm5_next")
      .hidden_size(kHidden)
      .q_lora_rank(kRank)
      .index_n_heads(kHeads)
      .index_head_dim(128)
      .qk_rope_head_dim(0)
      .index_topk(kTopk)
      .index_kpool(4)
      .index_kpool_compress(true)
      .index_kpool_always_select_tail(true)
      .max_position_embeddings(1048576);
  ParallelArgs parallel(0, 1, nullptr);
  Glm5NextKPoolIndexer indexer(args, QuantArgs(), parallel, nullptr, options);
  torch::manual_seed(1207);
  for (auto& parameter : indexer->named_parameters()) {
    parameter.value().normal_(0.0, 0.1);
  }
  torch::Tensor hidden = torch::randn({6, kHidden}, options);
  torch::Tensor queries = torch::randn({6, kRank}, options);
  torch::Tensor positions = torch::tensor({0, 1, 2, 0, 1, 2}, ints);
  torch::Tensor cache = torch::zeros({8, 1, 4, 128}, options);
  torch::Tensor tail = torch::zeros({4, 2, 16, 128}, options);
  torch::Tensor eager_cache = cache.clone();
  torch::Tensor eager_tail = tail.clone();
  AttentionMetadata meta{};
  meta.block_table = torch::tensor({{2, 0, 4, 6}, {1, 3, 5, 7}}, ints);
  meta.linear_state_indices = torch::tensor({1, 2}, ints);
  meta.q_seq_lens = torch::tensor({3, 3}, ints);
  meta.kv_seq_lens = torch::tensor({3, 3}, ints);
  meta.q_seq_lens_vec = {3, 3};
  meta.kv_seq_lens_vec = {3, 3};
  prepare_glm5_next_kpool_metadata(meta, device);
  const torch::Tensor starts = meta.kpool_batch_metadata->query_starts;
  meta.enable_cuda_graph = true;
  indexer->forward(hidden, queries, positions, cache, tail, meta);
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  std::tuple<torch::Tensor, torch::Tensor> output;
  {
    torch_mlu::mlu::MLUStreamGuard guard(
        torch_mlu::getStreamFromPool(false, 0));
    graph.capture_begin();
    output = indexer->forward(hidden, queries, positions, cache, tail, meta);
    graph.capture_end();
  }
  cache.zero_();
  tail.zero_();
  const std::vector<std::vector<int32_t>> widths{
      {3, 3}, {1, 5}, {0, 6}, {2, 4}};
  const std::vector<std::vector<int32_t>> token_positions{{0, 1, 2, 0, 1, 2},
                                                          {3, 3, 4, 5, 6, 7},
                                                          {0, 1, 2, 3, 4, 5},
                                                          {3, 7, 6, 7, 8, 9}};
  const std::vector<std::vector<int32_t>> lengths{
      {3, 3}, {4, 8}, {0, 6}, {0, 10}};
  for (size_t step = 0; step < widths.size(); ++step) {
    SCOPED_TRACE(step);
    hidden.normal_();
    queries.normal_();
    positions.copy_(torch::tensor(token_positions[step], ints));
    starts.copy_(torch::tensor({0, widths[step][0], 6}, starts.options()));
    meta.kv_seq_lens.copy_(torch::tensor(lengths[step], ints));
    if (step >= 2) {
      meta.linear_state_indices.copy_(torch::tensor({0, 3}, ints));
    }
    AttentionMetadata eager = meta;
    eager.enable_cuda_graph = false;
    eager.kpool_batch_metadata.reset();
    eager.q_seq_lens_vec = widths[step];
    eager.kv_seq_lens_vec = lengths[step];
    const auto expected = indexer->forward(
        hidden, queries, positions, eager_cache, eager_tail, eager);
    torch_mlu::synchronize();
    graph.replay();
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::equal(std::get<0>(output), std::get<0>(expected)));
    EXPECT_TRUE(torch::equal(std::get<1>(output), std::get<1>(expected)));
    EXPECT_TRUE(torch::equal(tail, eager_tail));
    EXPECT_TRUE(torch::allclose(cache, eager_cache, 0.015625, 0.015625));
    EXPECT_EQ(tail[0].count_nonzero().item<int64_t>(), 0);
  }
  EXPECT_EQ(std::get<1>(output).narrow(0, 0, 2).sum().item<int64_t>(), 0);
  KVCacheConfig::get_instance().block_size(old_block_size);
}

TEST(Glm5NextKPoolIndexerTest, DraftPlaceholderDoesNotOverwriteRetainedTail) {
  torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto ints = options.dtype(torch::kInt32);
  torch::manual_seed(1211);
  torch::Tensor keys = torch::randn({8, 128}, options);
  torch::Tensor gates = torch::randn({8, 128}, options);
  const torch::Tensor ape =
      torch::randn({4, 128}, options.dtype(torch::kFloat32));
  const torch::Tensor hadamard =
      util::create_hadamard_matrix(128, torch::kFloat32, device, true);
  torch::Tensor cache = torch::zeros({1, 1, 4, 128}, options);
  torch::Tensor tail = torch::zeros({2, 2, 16, 128}, options);
  const torch::Tensor ids = torch::tensor({1}, ints);
  const torch::Tensor table = torch::tensor({{0}}, ints);
  kernel::mlu::update_kpool(keys.narrow(0, 0, 6),
                            gates.narrow(0, 0, 6),
                            ape,
                            hadamard,
                            cache,
                            tail,
                            ids,
                            table,
                            torch::arange(6, ints),
                            torch::zeros({6}, options.dtype(torch::kInt64)),
                            torch::tensor({0, 6}, options.dtype(torch::kInt64)),
                            16,
                            4);
  // The fake previous token has unrelated projections but must retain the
  // real position 5 already cached by prefill when positions 6 and 7 arrive.
  torch::Tensor extend_keys = keys.narrow(0, 5, 3).clone();
  torch::Tensor extend_gates = gates.narrow(0, 5, 3).clone();
  extend_keys[0].fill_(100);
  extend_gates[0].fill_(100);
  kernel::mlu::update_kpool(extend_keys,
                            extend_gates,
                            ape,
                            hadamard,
                            cache,
                            tail,
                            ids,
                            table,
                            torch::tensor({-1, 6, 7}, ints),
                            torch::zeros({3}, options.dtype(torch::kInt64)),
                            torch::tensor({0, 3}, options.dtype(torch::kInt64)),
                            16,
                            4);
  const torch::Tensor expected =
      reference_compression(keys, gates, ape, hadamard, 4);
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(
      cache[0][0].narrow(0, 0, 2), expected, 0.015625, 0.015625));
  EXPECT_TRUE(torch::equal(tail[1][0][5], keys[5]));
}

}  // namespace xllm::layer

namespace xllm::layer {
TEST(Glm5NextKPoolIndexerTest, VerifyMasksHighScoringFuturePoolBeforeTopk) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::Tensor cache = torch::ones({2, 1, 4, 128}, opts);
  cache[1][0][1].fill_(100);
  const torch::Tensor q = torch::ones({4, kHeads, 128}, opts);
  const torch::Tensor weights =
      torch::ones({4, kHeads}, opts.dtype(torch::kFloat32));
  const torch::Tensor positions =
      torch::tensor({6, 7, 8, 9}, opts.dtype(torch::kInt32));
  const torch::Tensor rows =
      torch::tensor({0, 0, 0, 1}, opts.dtype(torch::kInt64));
  const torch::Tensor table =
      torch::tensor({{1, 0}, {1, 0}}, opts.dtype(torch::kInt32));
  KPoolBatchMetadata batch;
  batch.q_seq_lens = {3, 1};
  batch.query_starts = torch::tensor({0, 3, 4}, opts.dtype(torch::kInt64));
  for (const int64_t bytes :
       {int64_t{3 * sizeof(float)}, int64_t{12 * sizeof(float)}}) {
    const torch::Tensor ids = glm5_next_kpool_select(q,
                                                     weights,
                                                     positions,
                                                     rows,
                                                     cache,
                                                     table,
                                                     10,
                                                     16,
                                                     4,
                                                     kTopk,
                                                     1.0,
                                                     bytes);
    EXPECT_TRUE(torch::equal(ids.select(1, 0).cpu(),
                             torch::tensor({0, 1, 1, 1}, torch::kInt64)));
    EXPECT_TRUE(ids.slice(1, 2).eq(-1).all().item<bool>());
    const auto output = kernel::mlu::expand_kpool(
        ids, positions, rows, table, 16, kTopk, 4, true);
    EXPECT_TRUE(torch::equal(output.context_lens.cpu(),
                             torch::tensor({7, 8, 9, 10}, torch::kInt32)));
    EXPECT_EQ(output.physical_slots[0][0].item<int32_t>(), 16);
    EXPECT_EQ(output.physical_slots[1][0].item<int32_t>(), 20);
  }
}

TEST(Glm5NextKPoolIndexerTest,
     FourQuerySelectionKeepsCausalityWithSmallWorkspace) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto ints = opts.dtype(torch::kInt64);
  torch::Tensor cache = torch::arange(1, 17, opts)
                            .view({4, 1, 4, 1})
                            .expand({4, 1, 4, 128})
                            .contiguous();
  cache[1][0][1].fill_(10000);
  const torch::Tensor q = torch::ones({8, kHeads, 128}, opts);
  const torch::Tensor weights =
      torch::ones({8, kHeads}, opts.dtype(torch::kFloat32));
  const torch::Tensor positions =
      torch::tensor({6, 7, 8, 9, 14, 15, 16, -1}, ints);
  const torch::Tensor rows = torch::tensor({0, 0, 0, 0, 1, 1, 1, 1}, ints);
  const torch::Tensor table = torch::tensor({{1, 0}, {3, 2}}, ints);
  const torch::Tensor starts = torch::tensor({0, 4, 8}, ints);
  const torch::Tensor expected = glm5_next_kpool_select(q,
                                                        weights,
                                                        positions,
                                                        rows,
                                                        cache,
                                                        table,
                                                        /*max_kv_seq_len=*/32,
                                                        /*block_size=*/16,
                                                        /*index_kpool=*/4,
                                                        kTopk,
                                                        /*softmax_scale=*/1.0);
  for (const int64_t bytes : {int64_t{64}, int64_t{256}}) {
    const torch::Tensor actual = glm5_next_kpool_select(q,
                                                        weights,
                                                        positions,
                                                        rows,
                                                        cache,
                                                        table,
                                                        /*max_kv_seq_len=*/32,
                                                        /*block_size=*/16,
                                                        /*index_kpool=*/4,
                                                        kTopk,
                                                        /*softmax_scale=*/1.0,
                                                        bytes,
                                                        starts);
    EXPECT_TRUE(torch::equal(actual, expected));
    EXPECT_EQ(actual[0][0].item<int64_t>(), 0);
    EXPECT_EQ(actual[1][0].item<int64_t>(), 1);
    EXPECT_TRUE(actual[7].eq(-1).all().item<bool>());
  }
}
}  // namespace xllm::layer

namespace xllm::layer {
TEST(Glm5NextKPoolIndexerTest,
     GraphSelectionCrosses64KWithLivePagesAndPadding) {
  torch::NoGradGuard no_grad;
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const auto ints = opts.dtype(torch::kInt64);
  torch::Tensor cache = torch::zeros({4097, 1, 4, 128}, opts);
  const torch::Tensor columns = torch::arange(4097 * 4, ints);
  cache.view({-1, 128}).select(1, 0).copy_(columns.remainder(128));
  cache.view({-1, 128}).select(1, 1).copy_(torch::floor_divide(columns, 128));
  torch::Tensor q = torch::zeros({4, kHeads, 128}, opts);
  q.select(1, 0).select(1, 0).fill_(1);
  q.select(1, 0).select(1, 1).fill_(128);
  const torch::Tensor weights =
      torch::ones({4, kHeads}, opts.dtype(torch::kFloat32));
  torch::Tensor positions = torch::tensor({32767, 65535, 65539, -1}, ints);
  const torch::Tensor rows = torch::zeros({4}, ints);
  torch::Tensor table = torch::arange(4097, ints).view({1, 4097});
  const auto select = [&]() {
    return glm5_next_kpool_select(
        q, weights, positions, rows, cache, table, 65552, 16, 4, kTopk, 1.0);
  };
  select();
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  torch::Tensor captured;
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(false, 0));
    graph.capture_begin();
    captured = select();
    graph.capture_end();
  }
  for (const bool relocate : {false, true}) {
    if (relocate) {
      table.copy_(table.flip({1}));
      positions.copy_(torch::tensor({65539, -1, 65535, 32768}, ints));
    }
    graph.replay();
    torch_mlu::synchronize();
    const auto expected = reference_selection(
        q, weights, positions, rows, cache, table, 4, 4, kSelected, 1.0);
    EXPECT_TRUE(torch::equal(captured, expected));
  }
}
}  // namespace xllm::layer

namespace xllm::layer {
TEST(Glm5NextKPoolIndexerTest, PrefillChainMatchesIndependentTorchState) {
  torch::NoGradGuard no_grad;
  torch::manual_seed(6159);
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const int64_t saved = KVCacheConfig::get_instance().block_size();
  KVCacheConfig::get_instance().block_size(16);
  ModelArgs args;
  args.model_type("glm5_next")
      .hidden_size(kHidden)
      .q_lora_rank(kRank)
      .index_n_heads(kHeads)
      .index_head_dim(128)
      .qk_rope_head_dim(0)
      .index_topk(kTopk)
      .index_kpool(4)
      .index_kpool_compress(true)
      .index_kpool_always_select_tail(true)
      .max_position_embeddings(1048576);
  ParallelArgs parallel(0, 1, nullptr);
  Glm5NextKPoolIndexer indexer(args, QuantArgs(), parallel, nullptr, opts);
  auto parameters = indexer->named_parameters();
  for (auto& p : parameters) {
    p.value().zero_();
  }
  // Sparse identity projections make the reference independent of GEMM tiling.
  const auto identity = torch::eye(128, opts);
  parameters["wk.weight"].narrow(1, 0, 128).copy_(identity);
  parameters["wq_b.weight"].narrow(1, 0, 128).copy_(
      identity.repeat({kHeads, 1}));
  parameters["k_norm.weight"].fill_(1);
  parameters["weights_proj.weight"].select(1, 128).fill_(1.0 / 32);
  const int64_t tokens = 2083;
  torch::Tensor hidden = torch::randn({tokens, kHidden}, opts);
  hidden.select(1, 128).fill_(1);
  const torch::Tensor query_input = torch::randn({tokens, kRank}, opts);
  const torch::Tensor cpu_hidden = hidden.cpu().to(torch::kFloat32);
  const torch::Tensor raw = cpu_hidden.narrow(1, 0, 128);
  const torch::Tensor centered = raw - raw.mean(1, true);
  const torch::Tensor keys =
      (centered * torch::rsqrt(centered.square().mean(1, true) + 1e-6))
          .to(torch::kBFloat16);
  torch::Tensor hadamard = torch::ones({1, 1});
  for (int64_t size = 1; size < 128; size *= 2) {
    hadamard = torch::cat({torch::cat({hadamard, hadamard}, 1),
                           torch::cat({hadamard, -hadamard}, 1)},
                          0);
  }
  hadamard /= std::sqrt(128.0);
  const torch::Tensor compressed =
      reference_compression(keys.narrow(0, 0, 2080),
                            torch::zeros({2080, 128}, torch::kBFloat16),
                            torch::zeros({4, 128}),
                            hadamard,
                            4);
  const auto ints = opts.dtype(torch::kInt32);
  AttentionMetadata meta{};
  meta.is_prefill = true;
  meta.q_seq_lens_vec = {static_cast<int32_t>(tokens)};
  meta.kv_seq_lens_vec = meta.q_seq_lens_vec;
  meta.q_seq_lens = torch::tensor(meta.q_seq_lens_vec, ints);
  meta.kv_seq_lens = meta.q_seq_lens.clone();
  meta.linear_state_indices = torch::tensor({1}, ints);
  meta.block_table = torch::arange(131, ints).flip({0}).view({1, 131});
  const torch::Tensor positions = torch::arange(tokens, ints);
  torch::Tensor cache = torch::zeros({131, 1, 4, 128}, opts);
  torch::Tensor tail = torch::zeros({2, 2, 12, 128}, opts);
  const auto [slots, lengths] =
      indexer->forward(hidden, query_input, positions, cache, tail, meta);
  const PoolHistory actual = read_pool_history(
      cache, meta.block_table, meta.kv_seq_lens, tokens, 16, 4);
  EXPECT_TRUE(torch::allclose(
      actual.keys[0].narrow(0, 0, 520).cpu(), compressed, 0.015625, 0.015625));
  for (int64_t token = tokens - 12; token < tokens; ++token) {
    EXPECT_TRUE(torch::equal(tail[1][0][token % 12].cpu(), keys[token]));
  }
  for (const int64_t row : {0, 3, 31, 2046, 2047}) {
    const int64_t count = (row + 1) / 4;
    if (count == 0) {
      EXPECT_EQ(lengths[row].item<int32_t>(), row + 1);
      continue;
    }
    const torch::Tensor q =
        torch::matmul(
            query_input[row].cpu().narrow(0, 0, 128).to(torch::kFloat32),
            hadamard.transpose(0, 1))
            .to(torch::kBFloat16)
            .to(torch::kFloat32);
    const torch::Tensor scores =
        torch::relu(torch::matmul(
            compressed.narrow(0, 0, count).to(torch::kFloat32), q)) /
        std::sqrt(4096.0);
    const torch::Tensor ids =
        std::get<1>(scores.topk(std::min<int64_t>(512, count)));
    const torch::Tensor expected_ids = torch::full({1, 512}, -1, torch::kInt64);
    expected_ids[0].narrow(0, 0, ids.numel()).copy_(ids);
    const auto expected = reference_expansion(expected_ids,
                                              torch::tensor({row}),
                                              torch::zeros({1}, torch::kInt64),
                                              meta.block_table,
                                              16,
                                              2048,
                                              4,
                                              true);
    EXPECT_EQ(lengths[row].item<int32_t>(),
              expected.context_lens[0].item<int32_t>());
    // Equal zero scores may permute; stable membership must still agree.
    EXPECT_TRUE(torch::equal(std::get<0>(slots[row].cpu().sort()),
                             std::get<0>(expected.physical_slots[0].sort())));
  }
  KVCacheConfig::get_instance().block_size(saved);
}

// Long prefills score straight from the paged pool cache.  Rows that see more
// complete pools than the selection budget must keep the highest-scoring ones.
TEST(Glm5NextKPoolIndexerTest, LongPrefillKeepsTopScoringPoolsBeyondBudget) {
  torch::NoGradGuard no_grad;
  torch::manual_seed(44207);
  const torch::Device device(Platform::type_torch(), 0);
  const auto opts =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const int64_t saved = KVCacheConfig::get_instance().block_size();
  KVCacheConfig::get_instance().block_size(16);
  ModelArgs args;
  args.model_type("glm5_next")
      .hidden_size(kHidden)
      .q_lora_rank(kRank)
      .index_n_heads(kHeads)
      .index_head_dim(128)
      .qk_rope_head_dim(0)
      .index_topk(kTopk)
      .index_kpool(4)
      .index_kpool_compress(true)
      .index_kpool_always_select_tail(false)
      .max_position_embeddings(1048576);
  ParallelArgs parallel(0, 1, nullptr);
  Glm5NextKPoolIndexer indexer(args, QuantArgs(), parallel, nullptr, opts);
  auto parameters = indexer->named_parameters();
  for (auto& p : parameters) {
    p.value().zero_();
  }
  // Sparse identity projections make the reference independent of GEMM tiling.
  const auto identity = torch::eye(128, opts);
  parameters["wk.weight"].narrow(1, 0, 128).copy_(identity);
  parameters["wq_b.weight"].narrow(1, 0, 128).copy_(
      identity.repeat({kHeads, 1}));
  parameters["k_norm.weight"].fill_(1);
  parameters["weights_proj.weight"].select(1, 128).fill_(1.0 / 32);
  constexpr int64_t kTokens = 4800;
  constexpr int64_t kPages = kTokens / 16;
  torch::Tensor hidden = torch::randn({kTokens, kHidden}, opts);
  hidden.select(1, 128).fill_(1);
  const torch::Tensor query_input = torch::randn({kTokens, kRank}, opts);
  torch::Tensor hadamard = torch::ones({1, 1});
  for (int64_t size = 1; size < 128; size *= 2) {
    hadamard = torch::cat({torch::cat({hadamard, hadamard}, 1),
                           torch::cat({hadamard, -hadamard}, 1)},
                          0);
  }
  hadamard /= std::sqrt(128.0);
  const auto ints = opts.dtype(torch::kInt32);
  AttentionMetadata meta{};
  meta.is_prefill = true;
  meta.q_seq_lens_vec = {static_cast<int32_t>(kTokens)};
  meta.kv_seq_lens_vec = meta.q_seq_lens_vec;
  meta.q_seq_lens = torch::tensor(meta.q_seq_lens_vec, ints);
  meta.kv_seq_lens = meta.q_seq_lens.clone();
  meta.linear_state_indices = torch::tensor({1}, ints);
  meta.block_table = torch::arange(kPages, ints).flip({0}).view({1, kPages});
  const torch::Tensor positions = torch::arange(kTokens, ints);
  torch::Tensor cache = torch::zeros({kPages, 1, 4, 128}, opts);
  torch::Tensor tail = torch::zeros({2, 2, 12, 128}, opts);
  const auto [slots, lengths] =
      indexer->forward(hidden, query_input, positions, cache, tail, meta);
  // Selection is defined on the keys the prefill actually cached.
  const torch::Tensor keys =
      read_pool_history(
          cache, meta.block_table, meta.kv_seq_lens, kTokens, 16, 4)
          .keys[0]
          .cpu()
          .to(torch::kFloat32);
  // The device rounds the rotated query to BF16 on its own; scores within this
  // margin of the budget boundary may fall on either side.
  constexpr double kScoreMargin = 1e-3;
  for (const int64_t row : {2051, 3999, 4799}) {
    SCOPED_TRACE(row);
    const int64_t count = (row + 1) / 4;
    ASSERT_GT(count, kSelected);
    const torch::Tensor q =
        torch::matmul(
            query_input[row].cpu().narrow(0, 0, 128).to(torch::kFloat32),
            hadamard.transpose(0, 1))
            .to(torch::kBFloat16)
            .to(torch::kFloat32);
    const torch::Tensor scores =
        torch::relu(torch::matmul(keys.narrow(0, 0, count), q)) /
        std::sqrt(4096.0);
    const double boundary =
        std::get<0>(scores.topk(kSelected))[kSelected - 1].item<double>();
    const torch::Tensor must_select = scores > boundary + kScoreMargin;
    const torch::Tensor may_select = scores >= boundary - kScoreMargin;

    const torch::Tensor physical = slots[row].cpu().to(torch::kInt64);
    const torch::Tensor live = physical.masked_select(physical >= 0);
    EXPECT_EQ(lengths[row].item<int32_t>(), kSelected * 4);
    EXPECT_EQ(live.numel(), kSelected * 4);
    // block_table maps logical page p to physical page kPages - 1 - p.
    const torch::Tensor logical =
        (kPages - 1 - torch::floor_divide(live, 16)) * 16 +
        torch::remainder(live, 16);
    const torch::Tensor pools = torch::floor_divide(logical, 4);
    ASSERT_TRUE(pools.lt(count).all().item<bool>());
    torch::Tensor selected = torch::zeros({count}, torch::kBool);
    selected.index_fill_(/*dim=*/0, pools, true);
    EXPECT_EQ(selected.sum().item<int64_t>(), kSelected);
    EXPECT_FALSE((must_select & ~selected).any().item<bool>());
    EXPECT_FALSE((selected & ~may_select).any().item<bool>());
  }
  // The longest rows have more positive scores than the budget, so the boundary
  // separates real scores rather than ReLU ties.
  const torch::Tensor last_q =
      torch::matmul(
          query_input[kTokens - 1].cpu().narrow(0, 0, 128).to(torch::kFloat32),
          hadamard.transpose(0, 1))
          .to(torch::kBFloat16)
          .to(torch::kFloat32);
  EXPECT_GT(
      torch::matmul(keys, last_q).gt(64 * kScoreMargin).sum().item<int64_t>(),
      kSelected);
  KVCacheConfig::get_instance().block_size(saved);
}
}  // namespace xllm::layer
