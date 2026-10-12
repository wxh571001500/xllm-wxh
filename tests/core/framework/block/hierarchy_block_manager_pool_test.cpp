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

#include "hierarchy_block_manager_pool.h"

#include <folly/futures/Future.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "block_manager_impl.h"
#include "common/global_flags.h"
#include "framework/block/block_manager_pool.h"
#include "framework/block/block_utils.h"
#include "framework/block/composite_block_manager.h"
#include "framework/block/linear_state_block_manager.h"
#include "framework/block/sliding_window_block_manager.h"
#include "framework/config/scheduler_config.h"
#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

// Peer that reaches inside HierarchyBlockManagerPool for verification of host
// leaf construction. The pool is heavy to spin up; we exercise the plumbing
// with an Engine stub set to nullptr because the constructor only touches the
// engine during allocate / transfer paths, not at build time.
class HierarchyPoolTestPeer final {
 public:
  static const std::vector<std::unique_ptr<CompositeBlockManager>>&
  host_block_managers(const HierarchyBlockManagerPool& pool) {
    return pool.host_block_managers_;
  }

  static const CompositeBlockManager::LeafMap& host_leaves(
      const HierarchyBlockManagerPool& pool,
      int32_t dp_rank = 0) {
    return pool.host_block_managers_[dp_rank]->leaf_entries();
  }

  static std::vector<BlockTransferInfo> pending_load_infos(
      const HierarchyBlockManagerPool& pool) {
    std::vector<BlockTransferInfo> infos;
    for (const auto& per_dp : pool.load_block_transfer_infos_) {
      infos.insert(infos.end(), per_dp.begin(), per_dp.end());
    }
    return infos;
  }

  static const std::vector<BlockTransferInfo>& pending_load_infos(
      const HierarchyBlockManagerPool& pool,
      int32_t dp_rank) {
    return pool.load_block_transfer_infos_.at(dp_rank);
  }

  static CompositeBlockManager* device_composite(
      HierarchyBlockManagerPool& pool) {
    return static_cast<CompositeBlockManager*>(
        pool.block_managers_.front().get());
  }

  static void dispatch_pending_h2d(HierarchyBlockManagerPool& pool) {
    for (auto& per_dp : pool.load_block_transfer_infos_) {
      per_dp.clear();
    }
  }

  static void collect_offload_pairs(HierarchyBlockManagerPool& pool,
                                    Sequence* sequence) {
    pool.collect_offload_pairs(sequence);
  }

  static size_t pending_offload_pair_count(
      const HierarchyBlockManagerPool& pool) {
    size_t count = 0;
    for (const auto& queue : pool.offload_block_pair_queues_) {
      count += queue.size_approx();
    }
    return count;
  }

  static size_t pending_offload_pair_count(
      const HierarchyBlockManagerPool& pool,
      int32_t dp_rank) {
    return pool.offload_block_pair_queues_.at(dp_rank).size_approx();
  }

  static void release_pending_offload_pairs(HierarchyBlockManagerPool& pool) {
    for (auto& queue : pool.offload_block_pair_queues_) {
      std::shared_ptr<OffloadBlockPair> pair;
      while (queue.try_dequeue(pair)) {
        pair.reset();
      }
    }
  }

  static std::shared_ptr<OffloadBlockPair> take_pending_offload_pair(
      HierarchyBlockManagerPool& pool,
      int32_t dp_rank = 0) {
    std::shared_ptr<OffloadBlockPair> pair;
    pool.offload_block_pair_queues_.at(dp_rank).try_dequeue(pair);
    return pair;
  }

  static std::set<std::pair<BlockType, std::string>> complete_offload_pairs(
      HierarchyBlockManagerPool& pool) {
    std::set<std::pair<BlockType, std::string>> stored_keys;
    std::map<BlockType, std::vector<Block>> host_blocks;
    std::vector<Block> device_blocks;
    const size_t pair_count = pending_offload_pair_count(pool);
    device_blocks.reserve(pair_count);
    std::shared_ptr<OffloadBlockPair> pair;
    while (pool.offload_block_pair_queues_.front().try_dequeue(pair)) {
      stored_keys.emplace(pair->block_type,
                          std::string(reinterpret_cast<const char*>(
                                          pair->dst.get_immutable_hash_value()),
                                      XXH3_128BITS_HASH_VALUE_LEN));
      auto [blocks, inserted] = host_blocks.try_emplace(pair->block_type);
      if (inserted) {
        blocks->second.reserve(pair_count);
      }
      blocks->second.emplace_back(std::move(pair->dst));
      device_blocks.emplace_back(std::move(pair->src));
      pair.reset();
    }
    pool.block_managers_.front()->deallocate(device_blocks);
    for (auto& [type, blocks] : host_blocks) {
      auto* leaf =
          pool.host_block_managers_.front()->leaf_entries().at(type).leaf.get();
      leaf->cache(blocks);
      leaf->deallocate(blocks);
    }
    return stored_keys;
  }
};

namespace {

class FakeOffloadEngine final : public Engine {
 public:
  ForwardOutput step(std::vector<Batch>&) override { return {}; }

  void update_last_step_result(std::vector<Batch>&) override {}

  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }

  uint32_t host_transfer_worker_count() const override { return 2; }

  std::vector<folly::SemiFuture<uint32_t>> transfer_kv_blocks(
      uint32_t,
      const std::vector<BlockTransferInfo>& transfer_infos) override {
    transfer_count_ = static_cast<uint32_t>(transfer_infos.size());
    promises_.resize(2);
    std::vector<folly::SemiFuture<uint32_t>> results;
    results.reserve(promises_.size());
    for (auto& promise : promises_) {
      results.emplace_back(promise.getSemiFuture());
    }
    return results;
  }

  void finish_worker(size_t index, bool success) {
    promises_.at(index).setValue(success ? transfer_count_ : 0u);
  }

 private:
  uint32_t transfer_count_ = 0;
  std::vector<folly::Promise<uint32_t>> promises_;
};

class FakePrefetchEngine final : public Engine {
 public:
  explicit FakePrefetchEngine(size_t worker_count,
                              int64_t timeout_ms = -1,
                              size_t batch_size = 2)
      : worker_count_(worker_count),
        timeout_ms_(timeout_ms),
        batch_size_(batch_size) {}

  ForwardOutput step(std::vector<Batch>& /*batch*/) override { return {}; }

  void update_last_step_result(std::vector<Batch>& /*batch*/) override {}

  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }

  void prefetch_from_storage(
      uint32_t dp_rank,
      std::shared_ptr<const StoragePrefetchRequest> request,
      PrefetchResult::StopPredicate stop_requested,
      PrefetchResult::DoneCallback done) override {
    CHECK(request != nullptr);
    dp_rank_ = dp_rank;
    ++request_count_;
    request_ = std::move(request);
    result_ = std::make_shared<PrefetchResult>(worker_count_,
                                               *request_,
                                               batch_size_,
                                               timeout_ms_,
                                               std::move(stop_requested),
                                               std::move(done));
  }

  uint32_t dp_rank() const { return dp_rank_; }

  const StoragePrefetchRequest& request() const { return *request_; }

  const std::shared_ptr<PrefetchResult>& result() const { return result_; }

  size_t request_count() const { return request_count_; }

  void finish_worker(size_t worker_index,
                     size_t hit_units,
                     bool worker_ok = true) {
    CHECK(result_ != nullptr);
    size_t remaining_hits = hit_units;
    for (size_t batch_index = 0;
         batch_index < request_->batch_count(batch_size_);
         ++batch_index) {
      const size_t batch_units =
          request_->batch_unit_count(batch_index, batch_size_);
      const uint8_t batch_hits =
          static_cast<uint8_t>(std::min(remaining_hits, batch_units));
      const std::optional<PrefetchControl> control =
          result_->record_batch_result(worker_index, batch_hits);
      CHECK(control.has_value());
      remaining_hits -= batch_hits;
      if (*control == PrefetchControl::STOP) {
        break;
      }
    }
    CHECK_EQ(remaining_hits, 0u);
    result_->mark_worker_ended(worker_index, worker_ok);
  }

  void finish_worker_with_logical_hits(size_t worker_index,
                                       const std::vector<uint8_t>& logical_hits,
                                       bool worker_ok = true) {
    CHECK(result_ != nullptr);
    size_t logical_index = 0;
    for (size_t batch_index = 0;
         batch_index < request_->batch_count(batch_size_);
         ++batch_index) {
      const size_t unit_begin =
          request_->batch_unit_begin(batch_index, batch_size_);
      const size_t unit_count =
          request_->batch_unit_count(batch_index, batch_size_);
      std::vector<uint8_t> gated(batch_size_, 0);
      std::vector<uint8_t> non_gated(batch_size_, 0);
      for (size_t local = 0; local < unit_count; ++local) {
        const PrefetchUnit& unit = request_->units[unit_begin + local];
        bool gated_hit = true;
        for (size_t i = 0; i < unit.gated_blocks.size(); ++i) {
          CHECK_LT(logical_index, logical_hits.size());
          gated_hit = gated_hit && logical_hits[logical_index++] != 0;
        }
        bool non_gated_hit = !unit.has_non_gated;
        if (unit.has_non_gated) {
          non_gated_hit = !unit.non_gated_blocks.empty();
          for (size_t i = 0; i < unit.non_gated_blocks.size(); ++i) {
            CHECK_LT(logical_index, logical_hits.size());
            non_gated_hit = non_gated_hit && logical_hits[logical_index++] != 0;
          }
        }
        gated[local] = gated_hit ? 1 : 0;
        non_gated[local] = non_gated_hit ? 1 : 0;
      }
      const std::optional<PrefetchControl> control =
          result_->record_batch_result(worker_index, gated, non_gated);
      CHECK(control.has_value());
      if (*control == PrefetchControl::STOP) {
        break;
      }
    }
    CHECK_EQ(logical_index, logical_hits.size());
    result_->mark_worker_ended(worker_index, worker_ok);
  }

 private:
  size_t worker_count_ = 0;
  int64_t timeout_ms_ = -1;
  size_t batch_size_ = 2;
  uint32_t dp_rank_ = 0;
  size_t request_count_ = 0;
  std::shared_ptr<const StoragePrefetchRequest> request_;
  std::shared_ptr<PrefetchResult> result_;
};

class ControlledOffloadEngine final : public Engine {
 public:
  explicit ControlledOffloadEngine(size_t result_count)
      : promises_(result_count) {}

  ForwardOutput step(std::vector<Batch>& /*batch*/) override { return {}; }
  void update_last_step_result(std::vector<Batch>& /*batch*/) override {}
  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }
  uint32_t host_transfer_worker_count() const override { return 8; }

  std::vector<folly::SemiFuture<uint32_t>> transfer_kv_blocks(
      uint32_t dp_rank,
      const std::vector<BlockTransferInfo>& infos) override {
    EXPECT_EQ(dp_rank, 0U);
    transfer_infos_ = infos;
    std::vector<folly::SemiFuture<uint32_t>> futures;
    futures.reserve(promises_.size());
    for (auto& promise : promises_) {
      futures.emplace_back(promise.getSemiFuture());
    }
    return futures;
  }

  std::vector<folly::Promise<uint32_t>> promises_;
  std::vector<BlockTransferInfo> transfer_infos_;
};

BlockManagerPool::Options make_flat_kv_options() {
  BlockManagerPool::Options opts;
  opts.num_blocks(64)
      .host_num_blocks(128)
      .block_size(128)
      .enable_prefix_cache(true)
      .enable_host_offload(true);
  return opts;
}

BlockManagerPool::Options make_typed_cache_options() {
  constexpr uint32_t kBaseBlockSize = 128;
  constexpr uint32_t kWindow = 128;
  const uint32_t swa_blocks_per_seq =
      static_cast<uint32_t>(get_swa_blocks_per_seq(kWindow, kBaseBlockSize));

  BlockManagerPool::Options opts;
  opts.num_blocks(4096)
      .block_size(kBaseBlockSize)
      .enable_prefix_cache(true)
      .enable_host_offload(true)
      .sliding_window_size(kWindow)
      .swa_blocks_per_seq(swa_blocks_per_seq)
      .swa_num_blocks(266)
      .c4_num_blocks(256)
      .c128_num_blocks(256)
      .max_tokens_per_batch(32768)
      .max_seqs_per_batch(4)
      // SlidingWindow + BlockManagerImpl (C4) + BlockManagerImpl (C128).
      // The 0/4/128 compress_ratios drive the sub-manager block sizes.
      .manager_types({1u, 0u, 0u})
      .compress_ratios({0u, 4u, 128u})
      .host_num_blocks_by_type({{BlockType::SWA, 512},
                                {BlockType::C4, 512},
                                {BlockType::C128, 512}});
  return opts;
}

Sequence make_test_sequence(size_t index,
                            const std::vector<int32_t>& prompt_token_ids) {
  torch::Device device(Platform::type_torch(), 0);
  // Sequence borrows these defaults for its entire lifetime.
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker(/*max_generated_tokens=*/16,
                                          /*max_context_len=*/0,
                                          /*eos_token=*/-1,
                                          /*ignore_eos=*/false,
                                          /*stop_tokens=*/{},
                                          /*stop_sequences=*/{});
  SequenceParams seq_params;
  seq_params.seq_capacity =
      std::max<size_t>(32768, prompt_token_ids.size() + 16);
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = false;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  return Sequence(index,
                  prompt_token_ids,
                  input_embedding,
                  mm_data,
                  std::move(decoder),
                  seq_params);
}

std::shared_ptr<Request> make_test_request(
    const std::vector<int32_t>& prompt_token_ids,
    size_t best_of = 1) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(16);
  stopping_checker.set_max_context_len(prompt_token_ids.size() + 16);
  stopping_checker.set_ignore_eos(true);

  RequestState request_state("test",
                             prompt_token_ids,
                             sampling_param,
                             scheduler_param,
                             stopping_checker,
                             prompt_token_ids.size() + 16,
                             /*n=*/best_of,
                             /*best_of=*/best_of,
                             /*logprobs=*/false,
                             /*stream=*/false,
                             /*echo=*/false,
                             /*skip_special_tokens=*/true,
                             /*enable_schedule_overlap=*/false,
                             /*output_func=*/nullptr,
                             /*outputs_func=*/nullptr);
  return std::make_shared<Request>("request",
                                   "x-request",
                                   "time",
                                   std::move(request_state),
                                   "service-request");
}

void seed_host_prefix(BlockManager* leaf, const std::vector<int32_t>& tokens) {
  ASSERT_NE(leaf, nullptr);
  const size_t block_count = tokens.size() / leaf->block_size();
  std::vector<Block> blocks = leaf->allocate(block_count);
  ASSERT_EQ(blocks.size(), block_count);
  leaf->cache(tokens, blocks);
  leaf->deallocate(blocks);
  blocks.clear();
}

size_t count_valid_blocks(const Slice<Block>& blocks) {
  return static_cast<size_t>(
      std::count_if(blocks.begin(), blocks.end(), [](const Block& block) {
        return block.is_valid();
      }));
}

bool allocate_with_host_cache_budget(HierarchyBlockManagerPool* pool,
                                     Sequence* sequence,
                                     size_t num_tokens,
                                     size_t max_copy_units) {
  pool->allocate_shared(sequence);
  const HostCacheRestorePoint selected =
      pool->select_host_cache_restore(sequence, max_copy_units);
  pool->trim_host_cache(sequence, selected);
  return pool->allocate(sequence, num_tokens);
}

}  // namespace

// A flat KV layout still creates a single KV host leaf. num_total_blocks
// reports one less than the raw count because block id 0 is reserved as a
// sentinel by BlockManagerImpl.
TEST(HierarchyBlockManagerPoolTest, FlatKvHasSingleHostKvLeaf) {
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  const auto& per_dp = HierarchyPoolTestPeer::host_block_managers(pool);
  ASSERT_EQ(per_dp.size(), 1u);
  const auto& per_type = per_dp.front()->leaf_entries();
  ASSERT_EQ(per_type.size(), 1u);
  ASSERT_TRUE(per_type.count(BlockType::KV) == 1);
  EXPECT_EQ(per_type.at(BlockType::KV).leaf->block_size(), 128);
  EXPECT_EQ(per_type.at(BlockType::KV).leaf->num_total_blocks(), 127u);
}

// A typed SWA/C4/C128 layout creates matching Host leaves with the expected
// block_size and count. num_total_blocks() reports one less than the raw
// count because block id 0 is reserved as a sentinel by BlockManagerImpl and
// SlidingWindowBlockManager (a subclass).
TEST(HierarchyBlockManagerPoolTest, TypedLayoutHasSwaC4C128HostLeaves) {
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  const auto& per_dp = HierarchyPoolTestPeer::host_block_managers(pool);
  ASSERT_EQ(per_dp.size(), 1u);
  const auto& per_type = per_dp.front()->leaf_entries();
  ASSERT_EQ(per_type.size(), 3u);

  // SWA: gap-tolerant leaf with the base block size.
  ASSERT_TRUE(per_type.count(BlockType::SWA) == 1);
  EXPECT_EQ(per_type.at(BlockType::SWA).leaf->block_size(), 128);
  EXPECT_EQ(per_type.at(BlockType::SWA).leaf->num_total_blocks(), 511u);

  // C4: one logical block covers 2048 original tokens.
  ASSERT_TRUE(per_type.count(BlockType::C4) == 1);
  EXPECT_EQ(per_type.at(BlockType::C4).leaf->block_size(), 2048u);
  EXPECT_EQ(per_type.at(BlockType::C4).leaf->num_total_blocks(), 511u);

  // C128 has the same logical span and block count as C4.
  ASSERT_TRUE(per_type.count(BlockType::C128) == 1);
  EXPECT_EQ(per_type.at(BlockType::C128).leaf->block_size(), 2048u);
  EXPECT_EQ(per_type.at(BlockType::C128).leaf->num_total_blocks(), 511u);
}

// Multi-DP: each DP rank owns its own host leaf triplet with fresh block-id
// spaces.
TEST(HierarchyBlockManagerPoolTest, TypedLayoutHasPerDpRankLeaves) {
  constexpr int32_t kDpSize = 2;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/kDpSize);
  const auto& per_dp = HierarchyPoolTestPeer::host_block_managers(pool);
  ASSERT_EQ(per_dp.size(), static_cast<size_t>(kDpSize));
  for (const auto& composite : per_dp) {
    const auto& per_type = composite->leaf_entries();
    EXPECT_EQ(per_type.size(), 3u);
    EXPECT_TRUE(per_type.count(BlockType::SWA) == 1);
    EXPECT_TRUE(per_type.count(BlockType::C4) == 1);
    EXPECT_TRUE(per_type.count(BlockType::C128) == 1);
  }
}

TEST(HierarchyBlockManagerPoolTest,
     DecodeTypedLayoutKeepsSwaHostLeafForOffloadOnly) {
  BlockManagerPool::Options options = make_typed_cache_options();
  options.instance_is_decode(true).enable_disagg_pd(true);
  HierarchyBlockManagerPool pool(options,
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);

  const auto& per_type = HierarchyPoolTestPeer::host_leaves(pool);
  ASSERT_EQ(per_type.size(), 3u);
  ASSERT_TRUE(per_type.count(BlockType::SWA) == 1);
  ASSERT_TRUE(per_type.count(BlockType::C4) == 1);
  ASSERT_TRUE(per_type.count(BlockType::C128) == 1);

  // Decode never probes or restores Host SWA, but it still needs an SWA Host
  // destination so completed decode blocks can be offloaded. Compressed leaves
  // continue to participate in prefix matching.
  EXPECT_FALSE(per_type.at(BlockType::SWA).supports_prefix_cache);
  EXPECT_TRUE(per_type.at(BlockType::C4).supports_prefix_cache);
  EXPECT_TRUE(per_type.at(BlockType::C128).supports_prefix_cache);
}

TEST(HierarchyBlockManagerPoolTest, DecodeTypedLayoutProbesOnlyDeviceC4C128) {
  constexpr size_t kPromptTokens = 20001;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.instance_is_decode(true).enable_disagg_pd(true);
  HierarchyBlockManagerPool pool(options,
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 83);

  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(device->leaf_entries().at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(device->leaf_entries().at(BlockType::C128).leaf.get(),
                   tokens);
  seed_host_prefix(host.at(BlockType::SWA).leaf.get(), tokens);
  seed_host_prefix(host.at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(host.at(BlockType::C128).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  sequence.kv_state().set_kv_cache_tokens_num(kPromptTokens);
  ASSERT_EQ(sequence.stage(), SequenceStage::DECODE);
  EXPECT_TRUE(pool.needs_shared_reprobe(&sequence));
  pool.allocate_shared(&sequence);

  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::SWA), 0u);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::C4), 9u);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::C128), 9u);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 18432u);
  EXPECT_FALSE(sequence.host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence.has_host_cache_match());
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
}

TEST(HierarchyBlockManagerPoolTest,
     DecodeTypedLayoutOffloadsNonPrefixSwaAndCompressedLeaves) {
  constexpr size_t kPromptTokens = 20001;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.instance_is_decode(true).enable_disagg_pd(true);
  HierarchyBlockManagerPool pool(options,
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 89);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);

  sequence.kv_state().set_kv_cache_tokens_num(kPromptTokens);
  ASSERT_TRUE(pool.allocate(&sequence, kPromptTokens));
  sequence.kv_state().set_kv_cache_tokens_num(kPromptTokens);
  const auto& host = HierarchyPoolTestPeer::host_block_managers(pool).front();
  const size_t unit_tokens =
      host->leaf_entries().at(BlockType::C128).leaf->block_size();
  const size_t completed_units = kPromptTokens / unit_tokens;
  const size_t checkpoint_block =
      completed_units * unit_tokens / options.block_size() - 1;
  const bool has_checkpoint =
      sequence.kv_state().blocks(BlockType::SWA)[checkpoint_block].is_valid();
  pool.deallocate(&sequence);

  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool),
            2 * completed_units + static_cast<size_t>(has_checkpoint));
}

TEST(HierarchyBlockManagerPoolTest, AllocateSharedMountsMatchesWithoutH2d) {
  constexpr size_t kPromptTokens = 20001;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 5);
  auto& host_leaves = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host_leaves.at(BlockType::SWA).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C128).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);

  EXPECT_EQ(sequence.kv_cache_tokens_num(), 18432u);
  EXPECT_EQ(sequence.host_cache_copy_units(), 9u);
  EXPECT_FALSE(sequence.kv_state().has_any_blocks());
  EXPECT_TRUE(sequence.host_kv_state().has_any_blocks());
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
}

TEST(HierarchyBlockManagerPoolTest,
     AllocateSharedAdvertisesC128AlignedCopyUnits) {
  constexpr size_t kPromptTokens = 65537;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.host_num_blocks_by_type(
      {{BlockType::SWA, 640}, {BlockType::C4, 64}, {BlockType::C128, 64}});
  HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 6);
  auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host.at(BlockType::SWA).leaf.get(), tokens);
  seed_host_prefix(host.at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(host.at(BlockType::C128).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);

  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_FALSE(sequence.kv_state().has_any_blocks());
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 65536u);
  EXPECT_EQ(sequence.host_kv_state().shared_blocks_num(BlockType::SWA), 512u);
  EXPECT_EQ(sequence.host_kv_state().shared_blocks_num(BlockType::C4), 32u);
  EXPECT_EQ(sequence.host_kv_state().shared_blocks_num(BlockType::C128), 32u);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 65536u);
  EXPECT_EQ(sequence.host_cache_copy_units(), 32u);
  const HostCacheRestorePoint selected =
      pool.select_host_cache_restore(&sequence, /*max_copy_units=*/3);
  EXPECT_EQ(selected.restore_target_tokens, 6144u);
  EXPECT_EQ(selected.copy_units, 3u);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());

  pool.trim_host_cache(&sequence, selected);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 6144u);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 6144u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::SWA), 512u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::C4), 32u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::C128), 32u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::SWA), 48u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::C4), 3u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::C128), 3u);

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     SuccessfulAllocateBuildsTypedC128AlignedH2dPlan) {
  constexpr size_t kPromptTokens = 20001;
  const size_t unit_tokens = kDsv4CompressedBlockTokenSize;
  const size_t safe_hit_tokens =
      (kPromptTokens - 1) / unit_tokens * unit_tokens;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 7);

  auto& host_leaves = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host_leaves.at(BlockType::SWA).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C128).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool,
      &sequence,
      /*num_tokens=*/kPromptTokens,
      /*max_copy_units=*/std::numeric_limits<size_t>::max()));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), safe_hit_tokens);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), safe_hit_tokens);
  // Dense seeding mounts every SWA block in the matched prefix; real offload
  // retains only the last min(unit/base, swa_blocks_per_seq) = 1 per unit.
  EXPECT_EQ(sequence.host_kv_state().shared_blocks_num(BlockType::SWA),
            safe_hit_tokens / 128);
  EXPECT_EQ(sequence.host_kv_state().shared_blocks_num(BlockType::C4),
            safe_hit_tokens / unit_tokens);
  EXPECT_EQ(sequence.host_kv_state().shared_blocks_num(BlockType::C128),
            safe_hit_tokens / unit_tokens);

  const std::vector<BlockTransferInfo> infos =
      HierarchyPoolTestPeer::pending_load_infos(pool);
  size_t swa_count = 0;
  size_t c4_count = 0;
  size_t c128_count = 0;
  for (const BlockTransferInfo& info : infos) {
    EXPECT_EQ(info.transfer_type, TransferType::H2D);
    const Slice<Block> host_blocks =
        sequence.host_kv_state().blocks(info.block_type);
    const Slice<Block> hbm_blocks = sequence.kv_state().blocks(info.block_type);
    const auto host_it = std::find_if(
        host_blocks.begin(), host_blocks.end(), [&](const Block& b) {
          return b.id() == info.src_block_id;
        });
    const auto hbm_it =
        std::find_if(hbm_blocks.begin(), hbm_blocks.end(), [&](const Block& b) {
          return b.id() == info.dst_block_id;
        });
    ASSERT_NE(host_it, host_blocks.end());
    ASSERT_NE(hbm_it, hbm_blocks.end());
    const XXH3Key hbm_hash(hbm_it->get_immutable_hash_value());
    EXPECT_TRUE(XXH3Key(host_it->get_immutable_hash_value()) == hbm_hash);
    EXPECT_TRUE(XXH3Key(info.hash_key) == hbm_hash);
    switch (info.block_type) {
      case BlockType::SWA:
        ++swa_count;
        break;
      case BlockType::C4:
        ++c4_count;
        break;
      case BlockType::C128:
        ++c128_count;
        break;
      default:
        ADD_FAILURE() << "Unexpected H2D block type: "
                      << static_cast<int32_t>(info.block_type);
    }
  }
  EXPECT_EQ(swa_count, 1u);
  EXPECT_EQ(c4_count, safe_hit_tokens / unit_tokens);
  EXPECT_EQ(c128_count, safe_hit_tokens / unit_tokens);
}

TEST(HierarchyBlockManagerPoolTest,
     Dsv4WideWindowPrefetchRetainsFullRestoreTail) {
  constexpr size_t kWindowTokens = 2048;
  const size_t unit_tokens = kDsv4CompressedBlockTokenSize;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.sliding_window_size(kWindowTokens)
      .swa_blocks_per_seq(kWindowTokens / options.block_size())
      .enable_kvcache_store(true)
      .prefetch_batch_size(64);
  FakePrefetchEngine engine(/*worker_count=*/1,
                            /*timeout_ms=*/-1,
                            /*batch_size=*/64);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::shared_ptr<Request> request = make_test_request(
      std::vector<int32_t>(kWindowTokens + unit_tokens + 1, 23));
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;
  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_NE(engine.result(), nullptr);
  const size_t window_blocks = kWindowTokens / options.block_size();
  const size_t unit_blocks = unit_tokens / options.block_size();
  ASSERT_EQ(engine.request().units.size(),
            (kWindowTokens + unit_tokens) / unit_tokens);
  const PrefetchUnit& last_unit = engine.request().units.back();
  EXPECT_EQ(last_unit.non_gated_blocks.size(), window_blocks);
  const Slice<Block> swa_blocks =
      sequence->host_kv_state().blocks(BlockType::SWA);
  const size_t last_unit_end =
      kWindowTokens / options.block_size() + unit_blocks;
  for (size_t index = last_unit_end - window_blocks; index < last_unit_end;
       ++index) {
    EXPECT_TRUE(swa_blocks[index].is_valid()) << index;
  }
  engine.finish_worker_with_logical_hits(
      /*worker_index=*/0,
      std::vector<uint8_t>(engine.request().batch_transfer_count(
                               0, engine.request().units.size()),
                           1));
  EXPECT_TRUE(completed);
  EXPECT_EQ(sequence->host_kv_state().kv_cache_tokens_num(),
            kWindowTokens + unit_tokens);
  EXPECT_EQ(
      count_valid_blocks(sequence->host_kv_state().blocks(BlockType::SWA)),
      window_blocks);
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     IncompleteTiersDoNotCombineIntoHostRestore) {
  constexpr size_t kPromptTokens = 20001;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 17);

  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(device->leaf_entries().at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(device->leaf_entries().at(BlockType::C128).leaf.get(),
                   tokens);
  seed_host_prefix(host.at(BlockType::SWA).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence.host_cache_copy_units(), 0u);

  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool,
      &sequence,
      /*num_tokens=*/kPromptTokens,
      /*max_copy_units=*/std::numeric_limits<size_t>::max()));

  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_GE(sequence.kv_state().current_max_tokens_capacity(), kPromptTokens);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_GE(sequence.host_kv_state().current_max_tokens_capacity(),
            kPromptTokens);
  const std::vector<BlockTransferInfo> infos =
      HierarchyPoolTestPeer::pending_load_infos(pool);
  EXPECT_TRUE(infos.empty());
  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     PartialDsv4PrefillDoesNotReprobeAnUnmatchedHostTier) {
  constexpr size_t kPromptTokens = 20001;
  constexpr size_t kSharedTokens = 2048;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 41);
  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);

  // Model an HBM prefix retained after a failed allocation, while the released
  // Host tier still needs a fresh probe. Every typed leaf must cover the same
  // C128-aligned prefix for the shared-token cursor to be meaningful.
  auto& device_leaves = device->leaf_entries();
  sequence.kv_state().mount_composite_shared(
      BlockType::SWA,
      device_leaves.at(BlockType::SWA).leaf->allocate(kSharedTokens / 128));
  sequence.kv_state().mount_composite_shared(
      BlockType::C4,
      device_leaves.at(BlockType::C4).leaf->allocate(kSharedTokens / 2048));
  sequence.kv_state().mount_composite_shared(
      BlockType::C128,
      device_leaves.at(BlockType::C128).leaf->allocate(kSharedTokens / 2048));
  sequence.kv_state().set_kv_cache_tokens_num(kSharedTokens);
  sequence.kv_state().set_prefix_cache_matched();
  ASSERT_EQ(sequence.kv_state().kv_cache_tokens_num(), kSharedTokens);
  ASSERT_EQ(sequence.kv_state().shared_tokens_num(), kSharedTokens);

  sequence.host_kv_state().set_prefix_cache_matched();
  EXPECT_FALSE(pool.needs_shared_reprobe(&sequence));
  sequence.host_kv_state().set_prefix_cache_matched(false);
  EXPECT_TRUE(pool.needs_shared_reprobe(&sequence));

  sequence.kv_state().set_kv_cache_tokens_num(kSharedTokens + 1);
  EXPECT_FALSE(pool.needs_shared_reprobe(&sequence));

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest, FlatKvRestoreBudgetRemainsBlockLinear) {
  constexpr size_t kPromptTokens = 1025;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 23);
  auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 1024u);
  EXPECT_EQ(sequence.host_cache_copy_units(), 8u);

  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool, &sequence, /*num_tokens=*/kPromptTokens, /*max_copy_units=*/4));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 512u);
  const std::vector<BlockTransferInfo> infos =
      HierarchyPoolTestPeer::pending_load_infos(pool);
  ASSERT_EQ(infos.size(), 4u);
  for (const BlockTransferInfo& info : infos) {
    EXPECT_EQ(info.transfer_type, TransferType::H2D);
    EXPECT_EQ(info.block_type, BlockType::KV);
  }

  HierarchyPoolTestPeer::dispatch_pending_h2d(pool);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/kPromptTokens));
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     ZeroCopyRestoreTargetDoesNotReprobeHostCache) {
  constexpr size_t kPromptTokens = 1025;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 67);
  auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  const HostCacheRestorePoint selected =
      pool.select_host_cache_restore(&sequence, /*max_copy_units=*/0);
  ASSERT_EQ(selected.restore_target_tokens, 0u);
  ASSERT_EQ(selected.copy_units, 0u);

  pool.trim_host_cache(&sequence, selected);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/1));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
  EXPECT_FALSE(sequence.has_host_cache_match());

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest, FailedHostGrowthRejectsAllocation) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.host_num_blocks(3);
  HierarchyBlockManagerPool pool(options, nullptr, 1);
  std::vector<int32_t> tokens(257, 23);
  BlockManager* host_leaf =
      HierarchyPoolTestPeer::host_leaves(pool).at(BlockType::KV).leaf.get();
  seed_host_prefix(host_leaf, tokens);
  Sequence sequence = make_test_sequence(0, tokens);

  EXPECT_FALSE(allocate_with_host_cache_budget(&pool, &sequence, 257, 2));
  EXPECT_FALSE(sequence.kv_state().has_any_blocks());
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);
  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest, EmptyHostConfigurationRejectsAllocation) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.host_num_blocks(0).host_num_blocks_by_type({});
  HierarchyBlockManagerPool pool(options, nullptr, 1);
  EXPECT_EQ(HierarchyPoolTestPeer::host_block_managers(pool).front(), nullptr);
  Sequence sequence = make_test_sequence(0, std::vector<int32_t>(257, 23));
  EXPECT_DEATH(pool.allocate(&sequence, 257), "host_manager");
}

TEST(HierarchyBlockManagerPoolTest, FailedHbmAllocationDoesNotQueueH2d) {
  constexpr size_t kPromptTokens = 1025;
  BlockManagerPool::Options options = make_flat_kv_options();
  options.num_blocks(8);
  HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 29);
  auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
  EXPECT_TRUE(sequence.host_kv_state().has_any_blocks());
  ASSERT_TRUE(sequence.has_host_cache_match());
  const std::vector<size_t> used_before = pool.num_used_blocks();

  EXPECT_FALSE(allocate_with_host_cache_budget(
      &pool, &sequence, /*num_tokens=*/kPromptTokens, /*max_copy_units=*/4));
  EXPECT_FALSE(sequence.kv_state().has_any_blocks());
  EXPECT_FALSE(sequence.host_kv_state().has_any_blocks());
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
  EXPECT_EQ(pool.num_used_blocks(), used_before);
}

TEST(HierarchyBlockManagerPoolTest, ExistingBlocksSkipHostCacheRematch) {
  constexpr size_t kPromptTokens = 4097;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 31);
  auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool, &sequence, /*num_tokens=*/1024, /*max_copy_units=*/4));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 512u);
  sequence.kv_state().incr_kv_cache_tokens_num_up_to(/*new_target=*/1024);
  HierarchyPoolTestPeer::dispatch_pending_h2d(pool);
  pool.cache(&sequence);

  pool.allocate_shared(&sequence);
  const HostCacheRestorePoint selected =
      pool.select_host_cache_restore(&sequence, /*max_copy_units=*/8);
  ASSERT_EQ(selected.restore_target_tokens, 1024u);
  ASSERT_EQ(selected.copy_units, 0u);

  pool.trim_host_cache(&sequence, selected);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/2304));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 1024u);
  EXPECT_GE(sequence.kv_state().current_max_tokens_capacity(), 2304u);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest, ExistingHostBlocksSkipPrefixRematch) {
  constexpr size_t kPromptTokens = 1025;
  constexpr size_t kInitialCachedTokens = 128;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 43);
  auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host.at(BlockType::KV).leaf.get(),
                   std::vector<int32_t>(tokens.begin(),
                                        tokens.begin() + kInitialCachedTokens));

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  ASSERT_FALSE(sequence.kv_state().has_any_blocks());
  ASSERT_TRUE(sequence.host_kv_state().has_any_blocks());
  ASSERT_EQ(sequence.kv_cache_tokens_num(), kInitialCachedTokens);

  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);
  pool.allocate_shared(&sequence);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), kInitialCachedTokens);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::KV), 1u);

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest, ExistingBlocksSkipHbmPrefixRematch) {
  constexpr size_t kPromptTokens = 2305;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 47);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);

  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/1024));
  sequence.kv_state().incr_kv_cache_tokens_num_up_to(/*new_target=*/768);

  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  seed_host_prefix(device->leaf_entries().at(BlockType::KV).leaf.get(),
                   std::vector<int32_t>(tokens.begin(), tokens.begin() + 1280));

  pool.allocate_shared(&sequence);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 768u);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/1536));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 768u);
  EXPECT_GE(sequence.kv_state().current_max_tokens_capacity(), 1536u);

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest, HbmAllocationFailureDoesNotQueueH2d) {
  constexpr size_t kPromptTokens = 20001;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 11);
  auto& host_leaves = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host_leaves.at(BlockType::SWA).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C128).leaf.get(), tokens);

  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  std::vector<Block> exhausted_c128 =
      device->allocate_blocks(BlockType::C128, 247);
  ASSERT_EQ(exhausted_c128.size(), 247u);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  EXPECT_FALSE(allocate_with_host_cache_budget(
      &pool,
      &sequence,
      /*num_tokens=*/kPromptTokens,
      /*max_copy_units=*/std::numeric_limits<size_t>::max()));
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_FALSE(sequence.kv_state().has_any_blocks());
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());

  device->deallocate(exhausted_c128);
  exhausted_c128.clear();
}

TEST(HierarchyBlockManagerPoolTest, HostAllocationFailureRejectsHbmAllocation) {
  BlockManagerPool::Options options = make_flat_kv_options();
  // BlockManagerImpl reserves block id 0, leaving only one usable Host block.
  // HBM has enough capacity for both blocks requested below.
  options.host_num_blocks(2);
  HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);

  std::vector<int32_t> tokens(257, 71);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  EXPECT_FALSE(pool.allocate(&sequence, /*num_tokens=*/256));
  EXPECT_FALSE(sequence.kv_state().has_any_blocks());
  EXPECT_FALSE(sequence.host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence.has_host_cache_match());
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);

  pool.deallocate(&sequence);
}

namespace {

struct OffloadOutcome {
  size_t result_count_;
  int32_t failed_worker_;
  bool exception_;
  bool publish_;
};

class HostOffloadCompletionTest
    : public ::testing::TestWithParam<OffloadOutcome> {};

TEST_P(HostOffloadCompletionTest,
       PublishesOnlyCompleteCopiesAfterAllWorkersEnd) {
  const auto& outcome = GetParam();
  ControlledOffloadEngine engine(outcome.result_count_);
  HierarchyBlockManagerPool pool(
      make_flat_kv_options(), &engine, /*dp_size=*/1);
  const auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  BlockManager* host_leaf = host.at(BlockType::KV).leaf.get();
  const size_t initial_free = host_leaf->num_free_blocks();
  const size_t initial_device_free = pool.num_free_blocks().front();
  const std::vector<int32_t> tokens(257, 73);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/128));
  sequence.kv_state().set_kv_cache_tokens_num(128);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/256));
  pool.transfer_blocks();
  EXPECT_EQ(engine.transfer_infos_.size(), 1U);
  pool.deallocate(&sequence);
  pool.reset_prefix_cache();

  for (size_t i = 0; i < engine.promises_.size(); ++i) {
    // Even if an earlier worker failed, neither publish nor release the Host
    // reservation while another worker can still be writing it.
    EXPECT_TRUE(pool.has_pending_async_block_release());
    pool.reset_prefix_cache();
    EXPECT_EQ(pool.num_free_blocks().front(), initial_device_free - 1);
    EXPECT_EQ(host_leaf->num_blocks_in_prefix_cache(), 0U);
    EXPECT_EQ(host_leaf->num_free_blocks(), initial_free - 1);
    if (static_cast<int32_t>(i) != outcome.failed_worker_) {
      engine.promises_[i].setValue(1U);
    } else if (outcome.exception_) {
      engine.promises_[i].setException(
          folly::make_exception_wrapper<std::runtime_error>("offload failed"));
    } else {
      engine.promises_[i].setValue(0U);
    }
  }
  EXPECT_FALSE(pool.has_pending_async_block_release());
  pool.reset_prefix_cache();
  EXPECT_EQ(pool.num_free_blocks().front(), initial_device_free);
  EXPECT_EQ(host_leaf->num_blocks_in_prefix_cache(),
            outcome.publish_ ? 1U : 0U);
  EXPECT_EQ(host_leaf->num_free_blocks(),
            initial_free - (outcome.publish_ ? 1U : 0U));
  std::vector<Block> matched = host_leaf->allocate_shared(tokens);
  EXPECT_EQ(matched.size(), outcome.publish_ ? 1U : 0U);
  host_leaf->deallocate(matched);
}

INSTANTIATE_TEST_SUITE_P(
    Workers,
    HostOffloadCompletionTest,
    ::testing::Values(OffloadOutcome{8, -1, false, true},
                      OffloadOutcome{8, 2, false, false},
                      OffloadOutcome{8, 2, true, false},
                      OffloadOutcome{2, -1, false, false},
                      OffloadOutcome{0, -1, false, false},
                      OffloadOutcome{9, -1, false, false}));

class TypedHostOffloadTest : public ::testing::TestWithParam<bool> {};

TEST_P(TypedHostOffloadTest, PreservesTypedReservationsUntilAllCopiesEnd) {
  ControlledOffloadEngine engine(/*result_count=*/8);
  HierarchyBlockManagerPool pool(
      make_typed_cache_options(), &engine, /*dp_size=*/1);
  const auto& leaves = HierarchyPoolTestPeer::host_leaves(pool);
  const std::map<BlockType, size_t> copy_counts = {
      {BlockType::SWA, 8}, {BlockType::C4, 8}, {BlockType::C128, 8}};
  std::map<BlockType, size_t> initial_free;
  for (const auto& [type, count] : copy_counts) {
    initial_free.emplace(type, leaves.at(type).leaf->num_free_blocks());
  }
  Sequence sequence =
      make_test_sequence(/*index=*/0, std::vector<int32_t>(20001, 73));
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/16384));
  sequence.kv_state().set_kv_cache_tokens_num(16384);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/20001));
  pool.transfer_blocks();
  EXPECT_EQ(engine.transfer_infos_.size(), 24U);
  pool.deallocate(&sequence);
  for (size_t i = 0; i < engine.promises_.size(); ++i) {
    EXPECT_TRUE(pool.has_pending_async_block_release());
    for (const auto& [type, count] : copy_counts) {
      EXPECT_EQ(leaves.at(type).leaf->num_blocks_in_prefix_cache(), 0U);
      EXPECT_EQ(leaves.at(type).leaf->num_free_blocks(),
                initial_free.at(type) - count);
    }
    engine.promises_[i].setValue(!GetParam() && i == 2 ? 23U : 24U);
  }
  EXPECT_FALSE(pool.has_pending_async_block_release());
  for (const auto& [type, count] : copy_counts) {
    const size_t published = GetParam() ? count : 0;
    EXPECT_EQ(leaves.at(type).leaf->num_blocks_in_prefix_cache(), published);
    EXPECT_EQ(leaves.at(type).leaf->num_free_blocks(),
              initial_free.at(type) - published);
  }
}

INSTANTIATE_TEST_SUITE_P(Completion, TypedHostOffloadTest, ::testing::Bool());

}  // namespace

TEST(HierarchyBlockManagerPoolTest,
     FlatKvChunkGrowthOffloadsCompletedBlocksIncrementally) {
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(385, 73);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);

  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/128));
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);

  sequence.kv_state().set_kv_cache_tokens_num(128);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/256));
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 1u);
  EXPECT_TRUE(sequence.kv_state().blocks(BlockType::KV)[0].is_valid());
  EXPECT_FALSE(sequence.host_kv_state().blocks(BlockType::KV)[0].is_valid());

  sequence.kv_state().set_kv_cache_tokens_num(256);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/384));
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 2u);
  EXPECT_FALSE(sequence.host_kv_state().blocks(BlockType::KV)[1].is_valid());

  pool.deallocate(&sequence);
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 2u);
}

TEST(HierarchyBlockManagerPoolTest, D2hUsesHbmProgressInsteadOfRestoreTarget) {
  constexpr size_t kBlockSize = 16;
  constexpr size_t kRestoreTokens = 64;
  for (const size_t completed_tokens : {0u, 15u, 16u, 17u, 48u, 63u, 64u}) {
    SCOPED_TRACE(completed_tokens);
    auto options = make_flat_kv_options();
    options.block_size(kBlockSize).hasher_type(BlockHasherType::MTP_TEXT);
    HierarchyBlockManagerPool pool(options, nullptr, 1);
    const std::vector<int32_t> tokens(kRestoreTokens + 1, 83);
    auto request = make_test_request(tokens);
    Sequence* sequence = request->sequences().front().get();
    ASSERT_TRUE(pool.allocate(sequence, tokens.size()));
    sequence->kv_state().set_kv_cache_tokens_num(completed_tokens);
    sequence->set_host_cache_match(kRestoreTokens, kRestoreTokens / kBlockSize);
    ASSERT_EQ(sequence->kv_cache_tokens_num(), kRestoreTokens);
    ASSERT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);

    pool.deallocate(sequence);

    EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool),
              completed_tokens / kBlockSize);
    EXPECT_FALSE(sequence->has_any_blocks());
    EXPECT_FALSE(sequence->has_host_cache_match());
  }
}

TEST(HierarchyBlockManagerPoolTest, D2hUsesSequenceDpRank) {
  auto options = make_flat_kv_options();
  options.block_size(16).hasher_type(BlockHasherType::MTP_TEXT);
  HierarchyBlockManagerPool pool(options, nullptr, 2);
  const std::vector<int32_t> tokens(49, 89);
  auto request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  sequence->set_dp_rank(1);
  ASSERT_TRUE(pool.allocate(sequence, 32));
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);

  sequence->kv_state().set_kv_cache_tokens_num(17);
  ASSERT_TRUE(pool.allocate(sequence, tokens.size()));
  EXPECT_EQ(sequence->dp_rank(), 1);
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool, 0), 0u);
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool, 1), 1u);
  EXPECT_FALSE(sequence->host_kv_state().blocks(BlockType::KV)[0].is_valid());
  EXPECT_TRUE(sequence->host_kv_state().blocks(BlockType::KV)[1].is_valid());
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());

  sequence->kv_state().set_kv_cache_tokens_num(32);
  pool.deallocate(sequence);
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool, 0), 0u);
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool, 1), 2u);
  EXPECT_FALSE(sequence->has_any_blocks());
}

TEST(HierarchyBlockManagerPoolTest,
     Dsv4ChunkGrowthOffloadsAllCompletedCacheGroups) {
  constexpr size_t kFirstChunkTokens = 2048;
  constexpr size_t kPromptTokens = 4097;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 79);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);

  ASSERT_TRUE(pool.allocate(&sequence, kFirstChunkTokens));
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);

  sequence.kv_state().set_kv_cache_tokens_num(kFirstChunkTokens);
  ASSERT_TRUE(pool.allocate(&sequence, kPromptTokens));
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 3u);

  const size_t swa_checkpoint = kFirstChunkTokens / 128 - 1;
  EXPECT_FALSE(sequence.host_kv_state()
                   .blocks(BlockType::SWA)[swa_checkpoint]
                   .is_valid());
  EXPECT_FALSE(sequence.host_kv_state().blocks(BlockType::C4)[0].is_valid());
  EXPECT_TRUE(sequence.host_kv_state().blocks(BlockType::C4)[1].is_valid());
  EXPECT_FALSE(sequence.host_kv_state().blocks(BlockType::C128)[0].is_valid());
  EXPECT_TRUE(sequence.host_kv_state().blocks(BlockType::C128)[1].is_valid());
}

TEST(HierarchyBlockManagerPoolTest, Dsv4OffloadWaitsForCompleteC128Unit) {
  for (const bool decode : {false, true}) {
    SCOPED_TRACE(decode);
    auto options = make_typed_cache_options();
    options.instance_is_decode(decode);
    HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);
    const auto& host = HierarchyPoolTestPeer::host_block_managers(pool).front();
    const size_t unit_tokens =
        host->leaf_entries().at(BlockType::C128).leaf->block_size();
    std::vector<int32_t> tokens(decode ? unit_tokens - 1 : unit_tokens + 1,
                                101);
    Sequence sequence = make_test_sequence(/*index=*/0, tokens);
    if (decode) {
      sequence.kv_state().set_kv_cache_tokens_num(unit_tokens - 1);
    }
    ASSERT_TRUE(pool.allocate(&sequence, unit_tokens));
    sequence.kv_state().set_kv_cache_tokens_num(unit_tokens - 1);
    HierarchyPoolTestPeer::device_composite(pool)
        ->cache_full_blocks_for_sequence(&sequence);
    HierarchyPoolTestPeer::collect_offload_pairs(pool, &sequence);
    EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);
    if (decode) {
      sequence.append_token(Token(101));
    }
    sequence.kv_state().set_kv_cache_tokens_num(unit_tokens);
    pool.deallocate(&sequence);
    EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 3u);
  }
}

TEST(HierarchyBlockManagerPoolTest, Dsv4WideWindowOffloadsEveryTailBlock) {
  constexpr size_t kWindowTokens = 2048;
  auto options = make_typed_cache_options();
  options.sliding_window_size(kWindowTokens)
      .swa_blocks_per_seq(kWindowTokens / options.block_size());
  HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);
  const size_t unit_tokens = kDsv4CompressedBlockTokenSize;
  Sequence sequence = make_test_sequence(
      /*index=*/0, std::vector<int32_t>(unit_tokens + 1, 61));
  ASSERT_TRUE(pool.allocate(&sequence, unit_tokens));
  sequence.kv_state().set_kv_cache_tokens_num(unit_tokens);
  pool.deallocate(&sequence);
  const size_t swa_tail_blocks = std::min<size_t>(
      unit_tokens / options.block_size(), options.swa_blocks_per_seq());
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool),
            swa_tail_blocks + 2);
}

TEST(HierarchyBlockManagerPoolTest, Dsv4LargeChunksPersistEveryC128Checkpoint) {
  for (const size_t unit_count : {2u, 4u}) {
    SCOPED_TRACE(unit_count);
    auto options = make_typed_cache_options();
    options.enable_kvcache_store(true).prefetch_batch_size(8);
    HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);
    const auto& host = HierarchyPoolTestPeer::host_block_managers(pool).front();
    const size_t unit_tokens =
        host->leaf_entries().at(BlockType::C128).leaf->block_size();
    const size_t completed_tokens = unit_count * unit_tokens;
    std::vector<int32_t> tokens(completed_tokens + unit_tokens + 1, 97);
    Sequence sequence = make_test_sequence(/*index=*/0, tokens);
    ASSERT_TRUE(pool.allocate(&sequence, completed_tokens));
    sequence.kv_state().set_kv_cache_tokens_num(completed_tokens);
    ASSERT_TRUE(pool.allocate(&sequence, completed_tokens + unit_tokens));
    EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool),
              unit_count * 3);

    const auto stored_keys =
        HierarchyPoolTestPeer::complete_offload_pairs(pool);
    EXPECT_EQ(stored_keys.size(), unit_count * 3);
    Sequence host_reader = make_test_sequence(/*index=*/1, tokens);
    pool.allocate_shared(&host_reader);
    EXPECT_EQ(host_reader.host_kv_state().kv_cache_tokens_num(),
              completed_tokens);
    const auto host_restore =
        pool.select_host_cache_restore(&host_reader, unit_count);
    EXPECT_EQ(host_restore.restore_target_tokens, completed_tokens);

    FakePrefetchEngine engine(/*worker_count=*/1);
    HierarchyBlockManagerPool replay_pool(options, &engine, /*dp_size=*/1);
    auto request =
        make_test_request(std::vector<int32_t>(completed_tokens + 1, 97));
    bool done = false;
    replay_pool.prefetch_from_storage(
        request, [&](std::shared_ptr<Request>) { done = true; });
    const auto& prefetch = engine.request();
    std::vector<uint8_t> hits;
    ASSERT_EQ(prefetch.units.size(), unit_count);
    for (const PrefetchUnit& unit : prefetch.units) {
      EXPECT_EQ(unit.non_gated_blocks.size(), 1u);
      for (const auto& transfers :
           {std::cref(unit.gated_blocks), std::cref(unit.non_gated_blocks)}) {
        for (const BlockTransferInfo& info : transfers.get()) {
          const std::string key(reinterpret_cast<const char*>(info.hash_key),
                                XXH3_128BITS_HASH_VALUE_LEN);
          hits.emplace_back(stored_keys.count({info.block_type, key}) != 0);
        }
      }
    }
    EXPECT_EQ(std::count(hits.begin(), hits.end(), 1), hits.size());
    engine.finish_worker_with_logical_hits(/*worker_index=*/0, hits);
    EXPECT_TRUE(done);
    EXPECT_EQ(
        request->sequences().front()->host_kv_state().kv_cache_tokens_num(),
        completed_tokens);
    replay_pool.deallocate(request->sequences().front().get());
    pool.deallocate(&host_reader);
    pool.deallocate(&sequence);
  }
}

TEST(HierarchyBlockManagerPoolTest,
     Dsv4OneMillionTokensReuseHostBlocksWithSmallChunks) {
  constexpr size_t kChunkTokens = 2048;
  constexpr size_t kUnitTokens = 2048;
  constexpr size_t kContextTokens = 1024 * 1024;
  constexpr size_t kFinalTokens = kContextTokens + kChunkTokens;

  BlockManagerPool::Options options = make_typed_cache_options();
  options.num_blocks(16384)
      .swa_num_blocks(36)
      .c4_num_blocks(1024)
      .c128_num_blocks(1024)
      .max_tokens_per_batch(kChunkTokens)
      .max_seqs_per_batch(1)
      .host_num_blocks_by_type(
          {{BlockType::SWA, 144}, {BlockType::C4, 4}, {BlockType::C128, 4}});
  HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);
  std::vector<int32_t> tokens(kFinalTokens + 1, 83);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  size_t offloaded_units = 0;

  for (size_t target_tokens = kChunkTokens; target_tokens <= kFinalTokens;
       target_tokens += kChunkTokens) {
    ASSERT_TRUE(pool.allocate(&sequence, target_tokens)) << target_tokens;
    const size_t completed_tokens = sequence.kv_state().kv_cache_tokens_num();
    if (completed_tokens > 0) {
      EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 3u);
      HierarchyPoolTestPeer::release_pending_offload_pairs(pool);
      offloaded_units += kChunkTokens / kUnitTokens;
    } else {
      EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);
    }
    sequence.kv_state().set_kv_cache_tokens_num(target_tokens);
  }

  EXPECT_EQ(count_valid_blocks(sequence.host_kv_state().blocks(BlockType::SWA)),
            kChunkTokens / 128);
  EXPECT_EQ(count_valid_blocks(sequence.host_kv_state().blocks(BlockType::C4)),
            kChunkTokens / kUnitTokens);
  EXPECT_EQ(
      count_valid_blocks(sequence.host_kv_state().blocks(BlockType::C128)),
      kChunkTokens / kUnitTokens);
  EXPECT_EQ(offloaded_units, kContextTokens / kUnitTokens);
  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     H2dRestoreIsPublishedToDevicePrefixCacheDuringAllocation) {
  constexpr size_t kPromptTokens = 20001;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 13);
  auto& host_leaves = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host_leaves.at(BlockType::SWA).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C128).leaf.get(), tokens);

  Sequence restored = make_test_sequence(/*index=*/0, tokens);
  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool,
      &restored,
      /*num_tokens=*/kPromptTokens,
      /*max_copy_units=*/std::numeric_limits<size_t>::max()));
  ASSERT_FALSE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
  HierarchyPoolTestPeer::dispatch_pending_h2d(pool);

  Sequence replay = make_test_sequence(/*index=*/1, tokens);
  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool,
      &replay,
      /*num_tokens=*/kPromptTokens,
      /*max_copy_units=*/std::numeric_limits<size_t>::max()));
  EXPECT_EQ(replay.kv_state().kv_cache_tokens_num(), 18432u);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
}

TEST(HierarchyBlockManagerPoolTest,
     Dsv4HostRestoreAllocatesOnlySwaTailWindowAndCurrentChunk) {
  constexpr size_t kRestoreTokens = 8192;
  constexpr size_t kChunkTokens = 2048;
  constexpr size_t kTargetTokens = kRestoreTokens + kChunkTokens;
  constexpr size_t kPromptTokens = kTargetTokens + 1;
  constexpr size_t kSwaPhysicalBlocks = 132;

  BlockManagerPool::Options options = make_typed_cache_options();
  options.swa_num_blocks(kSwaPhysicalBlocks)
      .max_tokens_per_batch(kChunkTokens)
      .max_seqs_per_batch(1)
      .host_num_blocks_by_type(
          {{BlockType::SWA, 1024}, {BlockType::C4, 64}, {BlockType::C128, 64}});
  HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 47);
  auto& host_leaves = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host_leaves.at(BlockType::SWA).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C4).leaf.get(), tokens);
  seed_host_prefix(host_leaves.at(BlockType::C128).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  const HostCacheRestorePoint selected =
      pool.select_host_cache_restore(&sequence, /*max_copy_units=*/4);
  ASSERT_EQ(selected.restore_target_tokens, kRestoreTokens);
  pool.trim_host_cache(&sequence, selected);

  ASSERT_TRUE(pool.allocate(&sequence, kTargetTokens));

  const auto* device = HierarchyPoolTestPeer::device_composite(pool);
  const BlockManager* swa_leaf =
      device->leaf_entries().at(BlockType::SWA).leaf.get();
  const size_t blocks_per_window = swa_leaf->options().swa_blocks_per_seq();
  const size_t chunk_blocks = kChunkTokens / swa_leaf->block_size();
  EXPECT_EQ(count_valid_blocks(sequence.kv_state().blocks(BlockType::SWA)),
            blocks_per_window + chunk_blocks);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::SWA),
            kTargetTokens / swa_leaf->block_size());
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::C4), 5u);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::C128), 5u);
  EXPECT_GE(sequence.kv_state().current_max_tokens_capacity(), kTargetTokens);

  size_t swa_h2d_blocks = 0;
  size_t c4_h2d_blocks = 0;
  size_t c128_h2d_blocks = 0;
  for (const BlockTransferInfo& info :
       HierarchyPoolTestPeer::pending_load_infos(pool)) {
    switch (info.block_type) {
      case BlockType::SWA:
        ++swa_h2d_blocks;
        break;
      case BlockType::C4:
        ++c4_h2d_blocks;
        break;
      case BlockType::C128:
        ++c128_h2d_blocks;
        break;
      default:
        break;
    }
  }
  EXPECT_EQ(swa_h2d_blocks, blocks_per_window);
  EXPECT_EQ(c4_h2d_blocks, 4u);
  EXPECT_EQ(c128_h2d_blocks, 4u);

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     H2dRestoreDescriptionsRemainOwnedByPoolUntilDispatch) {
  constexpr size_t kPromptTokens = 1025;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 37);
  auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);
  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool, &sequence, /*num_tokens=*/kPromptTokens, /*max_copy_units=*/4));
  ASSERT_EQ(HierarchyPoolTestPeer::pending_load_infos(pool).size(), 4u);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 4u);

  HierarchyPoolTestPeer::dispatch_pending_h2d(pool);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 4u);
}

TEST(HierarchyBlockManagerPoolTest, H2dUsesCachedTokensInsteadOfPublishCursor) {
  constexpr size_t kBlockSize = 16;
  constexpr size_t kHostCachedTokens = 48;
  constexpr size_t kPromptTokens = 65;
  for (size_t initial_hbm_tokens : {0u, 16u, 17u, 32u, 48u, 64u}) {
    SCOPED_TRACE(initial_hbm_tokens);
    auto options = make_flat_kv_options();
    options.block_size(kBlockSize).hasher_type(BlockHasherType::MTP_TEXT);
    HierarchyBlockManagerPool pool(options, nullptr, 1);
    const std::vector<int32_t> tokens(kPromptTokens, 41);
    const auto& host_leaves = HierarchyPoolTestPeer::host_leaves(pool);
    seed_host_prefix(
        host_leaves.at(BlockType::KV).leaf.get(),
        std::vector<int32_t>(tokens.begin(),
                             tokens.begin() + kHostCachedTokens + 1));

    auto request = make_test_request(tokens);
    Sequence* sequence = request->sequences().front().get();
    sequence->set_dp_rank(0);
    CompositeBlockManager* device =
        HierarchyPoolTestPeer::device_composite(pool);
    ASSERT_TRUE(device->allocate_sequence(sequence, initial_hbm_tokens));
    KVCacheState& hbm_state = sequence->kv_state();
    hbm_state.set_kv_cache_tokens_num(initial_hbm_tokens);
    hbm_state.set_prefix_cache_matched();
    ASSERT_EQ(hbm_state.num_cached_blocks(BlockType::KV), 0u);

    KVCacheState& host_state = sequence->host_kv_state();
    CompositeBlockManager* host_manager =
        HierarchyPoolTestPeer::host_block_managers(pool).front().get();
    host_manager->allocate_shared_for_sequence(sequence, host_state);
    ASSERT_EQ(host_state.kv_cache_tokens_num(), kHostCachedTokens);
    ASSERT_TRUE(pool.allocate(sequence, kPromptTokens));

    const size_t host_blocks = kHostCachedTokens / kBlockSize;
    const size_t begin_block =
        std::min(initial_hbm_tokens / kBlockSize, host_blocks);
    const auto transfers = HierarchyPoolTestPeer::pending_load_infos(pool);
    ASSERT_EQ(transfers.size(), host_blocks - begin_block);
    for (size_t block_index = begin_block; block_index < host_blocks;
         ++block_index) {
      const BlockTransferInfo& transfer = transfers[block_index - begin_block];
      EXPECT_EQ(transfer.transfer_type, TransferType::H2D);
      EXPECT_EQ(transfer.block_type, BlockType::KV);
      EXPECT_EQ(transfer.src_block_id,
                host_state.blocks(BlockType::KV)[block_index].id());
      EXPECT_EQ(transfer.dst_block_id,
                hbm_state.blocks(BlockType::KV)[block_index].id());
    }
    const size_t restore_tokens =
        std::max(initial_hbm_tokens, kHostCachedTokens);
    EXPECT_EQ(hbm_state.kv_cache_tokens_num(), restore_tokens);
    EXPECT_EQ(hbm_state.num_cached_blocks(BlockType::KV),
              restore_tokens / kBlockSize);
    EXPECT_EQ(host_state.num_blocks(BlockType::KV),
              (kPromptTokens + kBlockSize - 1) / kBlockSize);
    EXPECT_EQ(host_state.kv_cache_tokens_num(),
              std::max(kHostCachedTokens, initial_hbm_tokens));
    ASSERT_TRUE(pool.allocate(sequence, kPromptTokens));
    EXPECT_EQ(HierarchyPoolTestPeer::pending_load_infos(pool).size(),
              transfers.size());
    pool.deallocate(sequence);
  }
}

TEST(HierarchyBlockManagerPoolTest, H2dUsesSequenceDpRank) {
  constexpr size_t kPromptTokens = 65;
  constexpr size_t kPrefixBlocks = 4;
  auto options = make_flat_kv_options();
  options.block_size(16).hasher_type(BlockHasherType::MTP_TEXT);
  HierarchyBlockManagerPool pool(options, nullptr, 2);
  const std::vector<int32_t> tokens(kPromptTokens, 43);
  const auto& host_leaves = HierarchyPoolTestPeer::host_leaves(pool, 1);
  seed_host_prefix(host_leaves.at(BlockType::KV).leaf.get(), tokens);

  auto request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  sequence->set_dp_rank(1);
  ASSERT_TRUE(pool.allocate(sequence, kPromptTokens));
  EXPECT_EQ(sequence->dp_rank(), 1);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool, 0).empty());
  const auto& transfers = HierarchyPoolTestPeer::pending_load_infos(pool, 1);
  ASSERT_EQ(transfers.size(), kPrefixBlocks);
  for (size_t block_index = 0; block_index < kPrefixBlocks; ++block_index) {
    const BlockTransferInfo& transfer = transfers[block_index];
    EXPECT_EQ(transfer.transfer_type, TransferType::H2D);
    EXPECT_EQ(transfer.block_type, BlockType::KV);
    EXPECT_EQ(
        transfer.src_block_id,
        sequence->host_kv_state().blocks(BlockType::KV)[block_index].id());
    EXPECT_EQ(transfer.dst_block_id,
              sequence->kv_state().blocks(BlockType::KV)[block_index].id());
  }
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     SharedHostPrefixRestoresAndGrowsEveryInBatchSequence) {
  constexpr size_t kPromptTokens = 20017;
  constexpr size_t kStepTargetTokens = 18432;
  constexpr size_t kCopyUnits = 9;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);

  std::vector<int32_t> first_tokens(kPromptTokens, 41);
  std::vector<int32_t> second_tokens = first_tokens;
  second_tokens.back() = 43;
  auto& host_leaves = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host_leaves.at(BlockType::SWA).leaf.get(), first_tokens);
  seed_host_prefix(host_leaves.at(BlockType::C4).leaf.get(), first_tokens);
  seed_host_prefix(host_leaves.at(BlockType::C128).leaf.get(), first_tokens);

  Sequence first = make_test_sequence(/*index=*/0, first_tokens);
  Sequence second = make_test_sequence(/*index=*/1, second_tokens);
  pool.allocate_shared(&first);
  pool.allocate_shared(&second);

  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool, &first, kStepTargetTokens, /*max_copy_units=*/kCopyUnits));
  pool.cache(&first, kStepTargetTokens);
  ASSERT_TRUE(allocate_with_host_cache_budget(
      &pool, &second, kStepTargetTokens, /*max_copy_units=*/kCopyUnits));

  EXPECT_GE(first.kv_state().current_max_tokens_capacity(), kStepTargetTokens);
  EXPECT_GE(second.kv_state().current_max_tokens_capacity(), kStepTargetTokens);

  pool.deallocate(&first);
  pool.deallocate(&second);
}

TEST(HierarchyBlockManagerPoolTest,
     SharedHostPrefixCopiesEveryPrivateDestination) {
  constexpr size_t kPromptTokens = 65;
  constexpr size_t kPrefixBlocks = 4;
  for (size_t hbm_prefix_tokens : {0u, 32u}) {
    SCOPED_TRACE(hbm_prefix_tokens);
    auto options = make_flat_kv_options();
    options.block_size(16).hasher_type(BlockHasherType::MTP_TEXT);
    HierarchyBlockManagerPool pool(options, /*engine=*/nullptr, /*dp_size=*/1);

    std::vector<int32_t> tokens(kPromptTokens, 41);
    auto& host = HierarchyPoolTestPeer::host_leaves(pool);
    seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);
    if (hbm_prefix_tokens > 0) {
      CompositeBlockManager* device =
          HierarchyPoolTestPeer::device_composite(pool);
      // MTP needs the next token to prove the second 16-token block.
      seed_host_prefix(
          device->leaf_entries().at(BlockType::KV).leaf.get(),
          std::vector<int32_t>(tokens.begin(), tokens.begin() + 33));
    }

    Sequence first = make_test_sequence(/*index=*/0, tokens);
    Sequence second = make_test_sequence(/*index=*/1, tokens);
    // Admission may pin the same Host prefix for multiple requests before
    // either request receives its own HBM allocation.
    pool.allocate_shared(&first);
    pool.allocate_shared(&second);
    ASSERT_EQ(first.kv_cache_tokens_num(), 64u);
    ASSERT_EQ(second.kv_cache_tokens_num(), 64u);
    ASSERT_EQ(first.kv_state().kv_cache_tokens_num(), hbm_prefix_tokens);
    ASSERT_EQ(second.kv_state().kv_cache_tokens_num(), hbm_prefix_tokens);
    ASSERT_TRUE(pool.allocate(&first, kPromptTokens));
    ASSERT_TRUE(pool.allocate(&second, kPromptTokens));

    const auto transfers = HierarchyPoolTestPeer::pending_load_infos(pool);
    const size_t expected_transfers = hbm_prefix_tokens == 0 ? 8 : 4;
    EXPECT_EQ(transfers.size(), expected_transfers);
    for (Sequence* sequence : {&first, &second}) {
      const auto host_blocks = sequence->host_kv_state().blocks(BlockType::KV);
      const auto device_blocks = sequence->kv_state().blocks(BlockType::KV);
      for (size_t i = 0; i < kPrefixBlocks; ++i) {
        const size_t expected_copies = i < hbm_prefix_tokens / 16 ? 0 : 1;
        EXPECT_EQ(
            std::count_if(transfers.begin(),
                          transfers.end(),
                          [&](const BlockTransferInfo& info) {
                            return info.transfer_type == TransferType::H2D &&
                                   info.block_type == BlockType::KV &&
                                   info.src_block_id == host_blocks[i].id() &&
                                   info.dst_block_id == device_blocks[i].id();
                          }),
            expected_copies)
            << "Unexpected Host restore count for sequence "
            << sequence->seq_id() << ", logical block " << i;
      }
    }

    for (size_t i = hbm_prefix_tokens / 16; i < kPrefixBlocks; ++i) {
      EXPECT_EQ(first.host_kv_state().blocks(BlockType::KV)[i].id(),
                second.host_kv_state().blocks(BlockType::KV)[i].id());
      EXPECT_NE(first.kv_state().blocks(BlockType::KV)[i].id(),
                second.kv_state().blocks(BlockType::KV)[i].id());
    }

    ASSERT_TRUE(pool.allocate(&first, kPromptTokens));
    ASSERT_TRUE(pool.allocate(&second, kPromptTokens));
    EXPECT_EQ(HierarchyPoolTestPeer::pending_load_infos(pool).size(),
              expected_transfers);

    pool.deallocate(&first);
    pool.deallocate(&second);
  }
}

TEST(HierarchyBlockManagerPoolTest,
     CompleteC4C128CombinationCountsOneCopyUnit) {
  constexpr size_t kPromptTokens = 4097;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 59);
  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  auto& host = HierarchyPoolTestPeer::host_leaves(pool);

  seed_host_prefix(device->leaf_entries().at(BlockType::SWA).leaf.get(),
                   std::vector<int32_t>(tokens.begin(), tokens.begin() + 2048));
  seed_host_prefix(device->leaf_entries().at(BlockType::C4).leaf.get(),
                   std::vector<int32_t>(tokens.begin(), tokens.begin() + 2048));
  seed_host_prefix(device->leaf_entries().at(BlockType::C128).leaf.get(),
                   std::vector<int32_t>(tokens.begin(), tokens.begin() + 2048));
  seed_host_prefix(host.at(BlockType::SWA).leaf.get(),
                   std::vector<int32_t>(tokens.begin(), tokens.begin() + 4096));
  seed_host_prefix(host.at(BlockType::C4).leaf.get(),
                   std::vector<int32_t>(tokens.begin(), tokens.begin() + 4096));
  seed_host_prefix(host.at(BlockType::C128).leaf.get(),
                   std::vector<int32_t>(tokens.begin(), tokens.begin() + 4096));

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  pool.allocate_shared(&sequence);

  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 2048u);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 4096u);
  // Cache publication cursors retain each state/type's own probe reach. The
  // longer Host matches must not be folded into the HBM cursors, and the
  // common restore boundary must not flatten the three block sizes.
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::SWA), 16u);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::C4), 1u);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::C128), 1u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::SWA), 32u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::C4), 2u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::C128), 2u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::C4), 2u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::C128), 2u);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 4096u);
  EXPECT_EQ(sequence.host_cache_copy_units(), 1u);
  const HostCacheRestorePoint zero_budget =
      pool.select_host_cache_restore(&sequence, /*max_copy_units=*/0);
  EXPECT_EQ(zero_budget.restore_target_tokens, 2048u);
  EXPECT_EQ(zero_budget.copy_units, 0u);
  EXPECT_TRUE(HierarchyPoolTestPeer::pending_load_infos(pool).empty());
}

TEST(HierarchyBlockManagerPoolTest,
     CopyBudgetTrimFallsBackPastMissingSwaWindow) {
  constexpr size_t kRestoreTokens = 8192;
  constexpr size_t kInitialBudgetBoundary = 6144;
  constexpr size_t kExpectedBoundary = 4096;
  HierarchyBlockManagerPool pool(make_typed_cache_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kRestoreTokens + 1, 61);
  Sequence sequence = make_test_sequence(/*index=*/0, tokens);

  auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  BlockManager* swa_leaf = host.at(BlockType::SWA).leaf.get();
  const size_t block_size = swa_leaf->block_size();
  const size_t tail_blocks = swa_leaf->options().swa_blocks_per_seq();
  ASSERT_GT(tail_blocks, 0u);
  ASSERT_EQ(kRestoreTokens % block_size, 0u);
  ASSERT_EQ(kInitialBudgetBoundary % block_size, 0u);
  ASSERT_EQ(kExpectedBoundary % block_size, 0u);

  std::vector<Block> allocated = swa_leaf->allocate(tail_blocks * 2);
  ASSERT_EQ(allocated.size(), tail_blocks * 2);
  std::vector<Block> sparse_swa(kRestoreTokens / block_size);
  const size_t expected_end = kExpectedBoundary / block_size;
  const size_t restore_end = kRestoreTokens / block_size;
  for (size_t i = 0; i < tail_blocks; ++i) {
    sparse_swa[expected_end - tail_blocks + i] = std::move(allocated[i]);
    sparse_swa[restore_end - tail_blocks + i] =
        std::move(allocated[tail_blocks + i]);
  }
  sequence.host_kv_state().replace_composite_blocks(
      BlockType::SWA,
      std::move(sparse_swa),
      /*num_shared_blocks=*/restore_end,
      /*num_cached_blocks=*/restore_end);
  sequence.set_host_cache_match(kRestoreTokens, /*copy_units=*/4);

  BlockManager* device_swa = HierarchyPoolTestPeer::device_composite(pool)
                                 ->leaf_entries()
                                 .at(BlockType::SWA)
                                 .leaf.get();
  std::vector<Block> device_window = device_swa->allocate(tail_blocks);
  ASSERT_EQ(device_window.size(), tail_blocks);
  const size_t incomplete_boundary = kInitialBudgetBoundary / block_size;
  std::vector<Block> device_blocks(incomplete_boundary);
  for (size_t index = 0; index < tail_blocks; ++index) {
    device_blocks[incomplete_boundary - tail_blocks + index] =
        std::move(device_window[index]);
  }
  sequence.kv_state().replace_composite_blocks(
      BlockType::SWA, std::move(device_blocks), 0, 0);

  const HostCacheRestorePoint selected =
      pool.select_host_cache_restore(&sequence, /*max_copy_units=*/3);
  EXPECT_EQ(selected.restore_target_tokens, kExpectedBoundary);
  EXPECT_EQ(selected.copy_units, 2u);

  pool.deallocate(&sequence);
}

TEST(HierarchyBlockManagerPoolTest, ExistingHostPrefixSkipsDuplicateD2h) {
  constexpr size_t kCachedTokens = 1024;
  HierarchyBlockManagerPool pool(make_flat_kv_options(),
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  std::vector<int32_t> tokens(kCachedTokens + 1, 19);
  auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  seed_host_prefix(host.at(BlockType::KV).leaf.get(), tokens);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  sequence.set_dp_rank(0);
  CompositeBlockManager* device = HierarchyPoolTestPeer::device_composite(pool);
  ASSERT_TRUE(device->allocate_sequence(&sequence, kCachedTokens));
  sequence.kv_state().set_kv_cache_tokens_num(kCachedTokens);
  device->cache_for_sequence(&sequence);

  HierarchyPoolTestPeer::collect_offload_pairs(pool, &sequence);
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);

  device->deallocate_for_sequence(&sequence);
  sequence.reset();
}

TEST(HierarchyBlockManagerPoolTest,
     OffloadCompletionOwnsBlocksAndPublishesOnlyAfterAllWorkersSucceed) {
  for (const bool typed : {false, true}) {
    for (const bool copy_ok : {false, true}) {
      SCOPED_TRACE(typed);
      SCOPED_TRACE(copy_ok);
      FakeOffloadEngine engine;
      const auto options =
          typed ? make_typed_cache_options() : make_flat_kv_options();
      HierarchyBlockManagerPool pool(options, &engine, 1);
      const size_t completed_tokens = typed ? 16384 : 256;
      const std::vector<int32_t> tokens(completed_tokens + 1, 19);
      const auto& host_leaves = HierarchyPoolTestPeer::host_leaves(pool);
      const auto& device_leaves =
          HierarchyPoolTestPeer::device_composite(pool)->leaf_entries();
      {
        Sequence sequence = make_test_sequence(0, tokens);
        ASSERT_TRUE(pool.allocate(&sequence, completed_tokens));
        sequence.kv_state().set_kv_cache_tokens_num(completed_tokens);
        pool.deallocate(&sequence);
        EXPECT_FALSE(sequence.has_any_blocks());
      }
      const size_t transfer_count =
          HierarchyPoolTestPeer::pending_offload_pair_count(pool);
      EXPECT_EQ(transfer_count, typed ? 24u : 2u);
      for (const auto& [type, entry] : host_leaves) {
        EXPECT_EQ(entry.leaf->num_blocks_in_prefix_cache(), 0u);
        EXPECT_GT(entry.leaf->num_used_blocks(), 0u);
      }

      pool.transfer_blocks();
      EXPECT_TRUE(pool.has_pending_async_block_release());
      engine.finish_worker(0, copy_ok);
      EXPECT_TRUE(pool.has_pending_async_block_release());
      for (const auto& [type, entry] : host_leaves) {
        EXPECT_EQ(entry.leaf->num_blocks_in_prefix_cache(), 0u);
      }
      engine.finish_worker(1, true);
      EXPECT_FALSE(pool.has_pending_async_block_release());
      for (const auto& [type, entry] : host_leaves) {
        EXPECT_EQ(entry.leaf->num_used_blocks(), 0u);
        EXPECT_EQ(entry.leaf->num_blocks_in_prefix_cache() > 0, copy_ok);
      }
      for (const auto& [type, entry] : device_leaves) {
        EXPECT_EQ(entry.leaf->num_used_blocks(), 0u);
      }

      Sequence hit = make_test_sequence(1, tokens);
      pool.allocate_shared(&hit);
      EXPECT_EQ(hit.host_kv_state().kv_cache_tokens_num(),
                copy_ok ? completed_tokens : 0u);
      pool.deallocate(&hit);
    }
  }
}

TEST(PrefetchResultTest, UsesMinimumContiguousUnitPrefixAcrossWorkers) {
  size_t common_hit_units = std::numeric_limits<size_t>::max();
  StoragePrefetchRequest request;
  for (size_t unit_index = 0; unit_index < 5; ++unit_index) {
    PrefetchUnit unit;
    BlockTransferInfo info(/*src_block_id=*/-1,
                           /*dst_block_id=*/static_cast<int32_t>(unit_index));
    info.transfer_type = TransferType::G2H;
    info.block_type = BlockType::KV;
    unit.gated_blocks.emplace_back(info);
    request.units.emplace_back(std::move(unit));
  }
  PrefetchResult result(
      /*worker_count=*/2,
      request,
      /*batch_size=*/2,
      /*timeout_ms=*/-1,
      [] { return false; },
      [&common_hit_units](PrefetchSummary summary) {
        common_hit_units = 0;
        while (common_hit_units < summary.gated_hits.size() &&
               summary.gated_hits[common_hit_units] != 0) {
          ++common_hit_units;
        }
      });

  std::optional<PrefetchControl> control =
      result.record_batch_result(/*worker_index=*/0, /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::CONTINUE);
  control =
      result.record_batch_result(/*worker_index=*/1, /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::CONTINUE);

  control =
      result.record_batch_result(/*worker_index=*/0, /*prefix_hit_units=*/1);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
  result.mark_worker_ended(/*worker_index=*/0, /*worker_ok=*/true);
  EXPECT_EQ(common_hit_units, std::numeric_limits<size_t>::max());

  control =
      result.record_batch_result(/*worker_index=*/1, /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  // Worker 0 has already observed the shared gate miss, so worker 1 closes
  // after its current batch even though that batch itself is a hit.
  EXPECT_EQ(*control, PrefetchControl::STOP);
  result.mark_worker_ended(/*worker_index=*/1, /*worker_ok=*/true);

  EXPECT_EQ(common_hit_units, 3u);
}

TEST(PrefetchResultTest, WorkersAdvanceIndependently) {
  StoragePrefetchRequest request;
  for (size_t unit_index = 0; unit_index < 4; ++unit_index) {
    PrefetchUnit unit;
    BlockTransferInfo info(/*src_block_id=*/-1,
                           /*dst_block_id=*/static_cast<int32_t>(unit_index));
    info.transfer_type = TransferType::G2H;
    info.block_type = BlockType::KV;
    unit.gated_blocks.emplace_back(info);
    request.units.emplace_back(std::move(unit));
  }
  PrefetchResult result(
      /*worker_count=*/2,
      request,
      /*batch_size=*/2,
      /*timeout_ms=*/-1,
      [] { return false; },
      [](PrefetchSummary /*summary*/) {});
  const std::vector<uint8_t> hits = {1, 1};

  std::optional<PrefetchControl> control =
      result.record_batch_result(/*worker_index=*/0, hits, hits);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::CONTINUE);
  control = result.record_batch_result(/*worker_index=*/0, hits, hits);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);

  control = result.record_batch_result(/*worker_index=*/1, hits, hits);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::CONTINUE);
  control = result.record_batch_result(/*worker_index=*/1, hits, hits);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
  result.mark_worker_ended(/*worker_index=*/0, /*worker_ok=*/true);
  result.mark_worker_ended(/*worker_index=*/1, /*worker_ok=*/true);
}

TEST(PrefetchResultTest, OptionalMissDoesNotStopGatePrefix) {
  StoragePrefetchRequest request;
  for (size_t unit_index = 0; unit_index < 2; ++unit_index) {
    PrefetchUnit unit;
    BlockTransferInfo gated(/*src_block_id=*/-1,
                            /*dst_block_id=*/static_cast<int32_t>(unit_index));
    gated.transfer_type = TransferType::G2H;
    gated.block_type = BlockType::KV;
    unit.gated_blocks.emplace_back(gated);
    BlockTransferInfo non_gated(
        /*src_block_id=*/-1,
        /*dst_block_id=*/static_cast<int32_t>(unit_index + 2));
    non_gated.transfer_type = TransferType::G2H;
    non_gated.block_type = BlockType::LINEAR;
    unit.non_gated_blocks.emplace_back(non_gated);
    unit.has_non_gated = true;
    request.units.emplace_back(std::move(unit));
  }

  PrefetchSummary summary;
  PrefetchResult result(
      /*worker_count=*/1,
      request,
      /*batch_size=*/2,
      /*timeout_ms=*/-1,
      [] { return false; },
      [&summary](PrefetchSummary value) { summary = std::move(value); });
  const std::optional<PrefetchControl> control = result.record_batch_result(
      /*worker_index=*/0,
      /*gated_hits=*/std::vector<uint8_t>{1, 1},
      /*non_gated_hits=*/std::vector<uint8_t>{0, 0});
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
  result.mark_worker_ended(/*worker_index=*/0, /*worker_ok=*/true);

  EXPECT_EQ(summary.gated_hits, (std::vector<uint8_t>{1, 1}));
  EXPECT_EQ(summary.non_gated_hits, (std::vector<uint8_t>{0, 0}));
}

TEST(PrefetchResultTest, LogicalHitsAreAndedAcrossWorkers) {
  StoragePrefetchRequest request;
  for (size_t unit_index = 0; unit_index < 2; ++unit_index) {
    PrefetchUnit unit;
    BlockTransferInfo gated(/*src_block_id=*/-1,
                            /*dst_block_id=*/static_cast<int32_t>(unit_index));
    gated.transfer_type = TransferType::G2H;
    gated.block_type = BlockType::KV;
    unit.gated_blocks.emplace_back(gated);
    BlockTransferInfo non_gated(
        /*src_block_id=*/-1,
        /*dst_block_id=*/static_cast<int32_t>(unit_index + 2));
    non_gated.transfer_type = TransferType::G2H;
    non_gated.block_type = BlockType::LINEAR;
    unit.non_gated_blocks.emplace_back(non_gated);
    unit.has_non_gated = true;
    request.units.emplace_back(std::move(unit));
  }

  PrefetchSummary summary;
  PrefetchResult result(
      /*worker_count=*/2,
      request,
      /*batch_size=*/2,
      /*timeout_ms=*/-1,
      [] { return false; },
      [&summary](PrefetchSummary value) { summary = std::move(value); });
  const std::vector<uint8_t> worker_zero_gated = {1, 1};
  const std::vector<uint8_t> worker_zero_non_gated = {0, 1};
  const std::vector<uint8_t> worker_one_gated = {1, 1};
  const std::vector<uint8_t> worker_one_non_gated = {1, 0};
  ASSERT_TRUE(
      result.record_batch_result(0, worker_zero_gated, worker_zero_non_gated)
          .has_value());
  result.mark_worker_ended(0, /*worker_ok=*/true);
  ASSERT_TRUE(
      result.record_batch_result(1, worker_one_gated, worker_one_non_gated)
          .has_value());
  result.mark_worker_ended(1, /*worker_ok=*/true);

  EXPECT_EQ(summary.gated_hits, (std::vector<uint8_t>{1, 1}));
  EXPECT_EQ(summary.non_gated_hits, (std::vector<uint8_t>{0, 0}));
}

TEST(PrefetchResultTest, WorkerFailureMarksSummaryUnsuccessful) {
  StoragePrefetchRequest request;
  BlockTransferInfo info(/*src_block_id=*/-1, /*dst_block_id=*/0);
  info.transfer_type = TransferType::G2H;
  info.block_type = BlockType::KV;
  PrefetchUnit unit;
  unit.gated_blocks.emplace_back(info);
  request.units.emplace_back(std::move(unit));

  PrefetchSummary summary;
  PrefetchResult result(
      /*worker_count=*/1,
      request,
      /*batch_size=*/1,
      /*timeout_ms=*/-1,
      [] { return false; },
      [&summary](PrefetchSummary value) { summary = std::move(value); });
  ASSERT_TRUE(result
                  .record_batch_result(
                      0, std::vector<uint8_t>{1}, std::vector<uint8_t>{1})
                  .has_value());
  result.mark_worker_ended(0, /*worker_ok=*/false);

  EXPECT_TRUE(summary.gated_hits.empty());
}

TEST(StoragePrefetchRequestTest, SwaMissPreservesCompressedGatePrefix) {
  const std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> hash = {};
  StoragePrefetchRequest request;
  for (size_t unit_index = 0; unit_index < 2; ++unit_index) {
    PrefetchUnit unit;
    for (BlockType block_type :
         {BlockType::SWA, BlockType::C4, BlockType::C128}) {
      BlockTransferInfo info(-1,
                             static_cast<int32_t>(unit_index),
                             hash.data(),
                             TransferType::G2H,
                             block_type);
      if (block_type == BlockType::SWA) {
        unit.non_gated_blocks.emplace_back(std::move(info));
        unit.has_non_gated = true;
      } else {
        unit.gated_blocks.emplace_back(std::move(info));
      }
    }
    request.units.emplace_back(std::move(unit));
  }
  ASSERT_TRUE(request.valid());
  PrefetchSummary summary;
  PrefetchResult result(
      1,
      request,
      2,
      -1,
      [] { return false; },
      [&summary](PrefetchSummary value) { summary = std::move(value); });
  ASSERT_TRUE(result
                  .record_batch_result(
                      0, std::vector<uint8_t>{1, 1}, std::vector<uint8_t>{0, 0})
                  .has_value());
  result.mark_worker_ended(0, true);
  EXPECT_EQ(summary.gated_hits, (std::vector<uint8_t>{1, 1}));
  EXPECT_EQ(summary.non_gated_hits, (std::vector<uint8_t>{0, 0}));
}

TEST(PrefetchResultTest, StopsAfterCancellationOrTimeout) {
  bool cancelled = false;
  size_t common_hit_units = 0;
  StoragePrefetchRequest request;
  for (size_t unit_index = 0; unit_index < 4; ++unit_index) {
    PrefetchUnit unit;
    BlockTransferInfo info(/*src_block_id=*/-1,
                           /*dst_block_id=*/static_cast<int32_t>(unit_index));
    info.transfer_type = TransferType::G2H;
    info.block_type = BlockType::KV;
    unit.gated_blocks.emplace_back(info);
    request.units.emplace_back(std::move(unit));
  }
  PrefetchResult cancelled_result(
      /*worker_count=*/1,
      request,
      /*batch_size=*/2,
      /*timeout_ms=*/-1,
      [&cancelled] { return cancelled; },
      [&common_hit_units](PrefetchSummary summary) {
        common_hit_units = 0;
        while (common_hit_units < summary.gated_hits.size() &&
               summary.gated_hits[common_hit_units] != 0) {
          ++common_hit_units;
        }
      });
  cancelled = true;
  std::optional<PrefetchControl> control = cancelled_result.record_batch_result(
      /*worker_index=*/0, /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
  EXPECT_FALSE(cancelled_result
                   .record_batch_result(/*worker_index=*/0,
                                        /*prefix_hit_units=*/2)
                   .has_value());
  cancelled_result.mark_worker_ended(/*worker_index=*/0, /*worker_ok=*/true);
  EXPECT_EQ(common_hit_units, 2u);

  PrefetchResult timed_out_result(
      /*worker_count=*/1,
      request,
      /*batch_size=*/2,
      /*timeout_ms=*/1,
      [] { return false; },
      [](PrefetchSummary /*summary*/) {});
  EXPECT_EQ(timed_out_result.stream_idle_timeout_ms(), 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  control = timed_out_result.record_batch_result(
      /*worker_index=*/0, /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
}

TEST(HierarchyBlockManagerPoolTest,
     StoragePrefetchTimeoutWaitsForWorkersToEnd) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_kvcache_store(true).prefetch_batch_size(2);
  FakePrefetchEngine engine(/*worker_count=*/2, /*timeout_ms=*/1);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::vector<int32_t> tokens(1025, 83);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;

  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_NE(engine.result(), nullptr);
  EXPECT_EQ(engine.request().units.size(), 8u);
  EXPECT_EQ(engine.request().batch_count(2), 4u);
  EXPECT_TRUE(sequence->host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence->host_kv_state().prefix_cache_matched());

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  std::optional<PrefetchControl> control =
      engine.result()->record_batch_result(/*worker_index=*/0,
                                           /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
  engine.result()->mark_worker_ended(/*worker_index=*/0, /*worker_ok=*/true);
  EXPECT_FALSE(completed);
  control = engine.result()->record_batch_result(/*worker_index=*/1,
                                                 /*prefix_hit_units=*/2);
  ASSERT_TRUE(control.has_value());
  EXPECT_EQ(*control, PrefetchControl::STOP);
  engine.result()->mark_worker_ended(/*worker_index=*/1, /*worker_ok=*/true);

  EXPECT_TRUE(completed);
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::KV), 2u);
  EXPECT_EQ(sequence->kv_cache_tokens_num(), 256u);
  EXPECT_TRUE(sequence->host_kv_state().prefix_cache_matched());
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     StoragePrefetchPublishesOnlyAfterAllTpWorkersEnd) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_kvcache_store(true).prefetch_batch_size(2);
  FakePrefetchEngine engine(/*worker_count=*/2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::vector<int32_t> tokens(1025, 73);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;

  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_EQ(engine.dp_rank(), 0u);
  ASSERT_EQ(engine.request().units.size(), 8u);
  ASSERT_NE(engine.result(), nullptr);
  EXPECT_FALSE(sequence->kv_state().has_any_blocks());
  EXPECT_TRUE(sequence->host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence->host_kv_state().prefix_cache_matched());

  engine.finish_worker(/*worker_index=*/0, /*hit_units=*/8);
  EXPECT_FALSE(completed);
  EXPECT_FALSE(sequence->host_kv_state().prefix_cache_matched());
  engine.finish_worker(/*worker_index=*/1, /*hit_units=*/8);

  EXPECT_TRUE(completed);
  EXPECT_FALSE(sequence->kv_state().has_any_blocks());
  EXPECT_TRUE(sequence->host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence->kv_state().prefix_cache_matched());
  EXPECT_TRUE(sequence->host_kv_state().prefix_cache_matched());
  pool.allocate_shared(sequence);
  EXPECT_FALSE(sequence->kv_state().has_any_blocks());
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::KV), 8u);
  EXPECT_EQ(sequence->kv_cache_tokens_num(), 1024u);
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     CancelledStoragePrefetchReleasesReservedBlocks) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_kvcache_store(true).prefetch_batch_size(2);
  FakePrefetchEngine engine(/*worker_count=*/2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::vector<int32_t> tokens(1025);
  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = static_cast<int32_t>(i / 128);
  }
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  BlockManager* host_leaf =
      HierarchyPoolTestPeer::host_leaves(pool).at(BlockType::KV).leaf.get();
  const std::vector<int32_t> cached_tokens(tokens.begin(),
                                           tokens.begin() + 512);
  seed_host_prefix(host_leaf, cached_tokens);
  const size_t free_blocks_before = host_leaf->num_free_blocks();
  const size_t cached_blocks_before = host_leaf->num_blocks_in_prefix_cache();
  bool completed = false;

  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_NE(engine.result(), nullptr);
  ASSERT_LT(host_leaf->num_free_blocks(), free_blocks_before);
  request->set_cancel();

  engine.finish_worker(/*worker_index=*/0, /*hit_units=*/2);
  EXPECT_FALSE(completed);
  engine.finish_worker(/*worker_index=*/1, /*hit_units=*/2);

  EXPECT_TRUE(completed);
  EXPECT_EQ(host_leaf->num_free_blocks(), free_blocks_before);
  EXPECT_EQ(host_leaf->num_blocks_in_prefix_cache(), cached_blocks_before);
  EXPECT_EQ(host_leaf->num_used_blocks(), 0u);
  EXPECT_FALSE(sequence->host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence->host_kv_state().prefix_cache_matched());
}

TEST(HierarchyBlockManagerPoolTest,
     FailedStoragePrefetchDoesNotPublishHostPrefix) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_kvcache_store(true).prefetch_batch_size(2);
  FakePrefetchEngine engine(/*worker_count=*/1);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::vector<int32_t> tokens(1025, 41);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  BlockManager* host_leaf =
      HierarchyPoolTestPeer::host_leaves(pool).at(BlockType::KV).leaf.get();
  bool completed = false;

  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_NE(engine.result(), nullptr);
  engine.finish_worker(/*worker_index=*/0,
                       /*hit_units=*/0,
                       /*worker_ok=*/false);

  EXPECT_TRUE(completed);
  EXPECT_EQ(host_leaf->num_used_blocks(), 0u);
  EXPECT_EQ(host_leaf->num_blocks_in_prefix_cache(), 0u);
  EXPECT_FALSE(sequence->host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence->host_kv_state().prefix_cache_matched());
}

TEST(HierarchyBlockManagerPoolTest,
     MultiSequenceStoragePrefetchUsesOnlyFirstSequence) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_kvcache_store(true).prefetch_batch_size(2);
  FakePrefetchEngine engine(/*worker_count=*/1);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::vector<int32_t> tokens(1025, 43);
  std::shared_ptr<Request> request = make_test_request(tokens, /*best_of=*/2);
  ASSERT_TRUE(request->expand_sequences(/*share_prefix=*/false));
  ASSERT_EQ(request->sequences().size(), 2u);
  bool completed = false;

  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_EQ(engine.request_count(), 1u);
  ASSERT_EQ(engine.request().units.size(), 8u);

  engine.finish_worker(/*worker_index=*/0,
                       /*hit_units=*/8);
  EXPECT_TRUE(completed);
  EXPECT_TRUE(request->sequences()[0]->host_kv_state().has_any_blocks());
  EXPECT_TRUE(request->sequences()[0]->host_kv_state().prefix_cache_matched());
  EXPECT_FALSE(request->sequences()[1]->host_kv_state().has_any_blocks());
  EXPECT_FALSE(request->sequences()[1]->host_kv_state().prefix_cache_matched());
  BlockManager* host_leaf =
      HierarchyPoolTestPeer::host_leaves(pool).at(BlockType::KV).leaf.get();
  EXPECT_EQ(host_leaf->num_blocks_in_prefix_cache(), 8u);
  pool.deallocate(request->sequences()[0].get());
  pool.deallocate(request->sequences()[1].get());
}

TEST(HierarchyBlockManagerPoolTest,
     TypedStoragePrefetchRetainsLongestPrefixWithinHostCapacity) {
  constexpr size_t kPromptTokens = 4097;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.enable_kvcache_store(true)
      .prefetch_batch_size(2)
      .host_num_blocks_by_type(
          {{BlockType::SWA, 129}, {BlockType::C4, 2}, {BlockType::C128, 2}});
  FakePrefetchEngine engine(/*worker_count=*/2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 79);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;

  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_NE(engine.result(), nullptr);
  EXPECT_EQ(engine.request().units.size(), 1u);
  EXPECT_EQ(engine.request().batch_transfer_count(0, 2), 3u);

  engine.finish_worker(/*worker_index=*/0, /*hit_units=*/1);
  EXPECT_FALSE(completed);
  engine.finish_worker(/*worker_index=*/1, /*hit_units=*/1);
  ASSERT_TRUE(completed);

  pool.allocate_shared(sequence);
  const Slice<Block> swa_blocks =
      sequence->host_kv_state().blocks(BlockType::SWA);
  ASSERT_EQ(swa_blocks.size(), 16u);
  for (size_t i = 0; i + 1 < swa_blocks.size(); ++i) {
    EXPECT_FALSE(swa_blocks[i].is_valid());
  }
  EXPECT_TRUE(swa_blocks.back().is_valid());
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::C4), 1u);
  EXPECT_EQ(sequence->host_kv_state().num_blocks(BlockType::C128), 1u);
  EXPECT_EQ(sequence->kv_cache_tokens_num(), 2048u);
  pool.deallocate(sequence);
}

TEST(HierarchyBlockManagerPoolTest,
     TypedStoragePrefetchStopsAtFirstIncompleteUnit) {
  constexpr size_t kPromptTokens = 4097;
  BlockManagerPool::Options options = make_typed_cache_options();
  options.enable_kvcache_store(true).prefetch_batch_size(2);
  FakePrefetchEngine engine(/*worker_count=*/2);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::vector<int32_t> tokens(kPromptTokens, 79);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;

  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_NE(engine.result(), nullptr);
  ASSERT_EQ(engine.request().units.size(), 2u);
  EXPECT_EQ(engine.request().batch_transfer_count(0, 2), 6u);

  engine.finish_worker(/*worker_index=*/0, /*hit_units=*/0);
  EXPECT_FALSE(completed);
  engine.finish_worker(/*worker_index=*/1, /*hit_units=*/2);

  EXPECT_TRUE(completed);
  EXPECT_FALSE(sequence->host_kv_state().has_any_blocks());
  EXPECT_TRUE(sequence->host_kv_state().prefix_cache_matched());
  EXPECT_EQ(sequence->kv_cache_tokens_num(), 0u);
}

TEST(HierarchyBlockManagerPoolTest, TypedLayoutSupportsStoragePrefetch) {
  BlockManagerPool::Options options = make_typed_cache_options();
  options.enable_kvcache_store(true);
  HierarchyBlockManagerPool pool(options,
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  EXPECT_EQ(HierarchyPoolTestPeer::host_leaves(pool).size(), 3u);
}

TEST(HierarchyBlockManagerPoolTest, SupportsLinearCacheLayout) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_linear_state(true)
      .linear_state_num_slots(64)
      .host_num_blocks_by_type({{BlockType::KV, 128}, {BlockType::LINEAR, 64}});

  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 128;
  HierarchyBlockManagerPool pool(options,
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  const auto* device = HierarchyPoolTestPeer::device_composite(pool);
  EXPECT_EQ(device->leaf_combination(),
            CompositeBlockManager::LeafCombination::FLAT_KV_LINEAR);
  const auto& host = HierarchyPoolTestPeer::host_leaves(pool);
  EXPECT_TRUE(host.contains(BlockType::KV));
  EXPECT_TRUE(host.contains(BlockType::LINEAR));
  EXPECT_EQ(host.at(BlockType::LINEAR).leaf->block_size(), 128);
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      original_chunk_stride;
}

TEST(HierarchyBlockManagerPoolTest,
     DecodeLinearLayoutOffloadsKvWithoutPrefixPrefetch) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_linear_state(true)
      .linear_state_num_slots(64)
      .host_num_blocks_by_type({{BlockType::KV, 128}, {BlockType::LINEAR, 64}})
      .enable_prefix_cache(false)
      .enable_disagg_pd(true)
      .instance_is_decode(true)
      .enable_kvcache_store(true)
      .prefetch_batch_size(4);

  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 2048;

  FakePrefetchEngine engine(/*worker_count=*/1,
                            /*timeout_ms=*/-1,
                            /*batch_size=*/4);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  const auto* device = HierarchyPoolTestPeer::device_composite(pool);
  EXPECT_EQ(device->leaf_combination(),
            CompositeBlockManager::LeafCombination::UNSUPPORTED);

  std::vector<int32_t> tokens(4097, 103);
  std::shared_ptr<Request> request = make_test_request(tokens);
  bool completed = false;
  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  EXPECT_TRUE(completed);
  EXPECT_EQ(engine.request_count(), 0u);

  Sequence sequence = make_test_sequence(/*index=*/0, tokens);
  ASSERT_TRUE(pool.allocate(&sequence, tokens.size()));
  sequence.kv_state().set_kv_cache_tokens_num(4096);
  pool.deallocate(&sequence);

  // Decode's LINEAR leaf is receiver/live state, not a Prefill checkpoint.
  // The 32 completed KV blocks are still queued for Store writeback.
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 32u);
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      original_chunk_stride;
}

TEST(HierarchyBlockManagerPoolTest,
     DecodeLinearOffloadUsesCanonicalMtpCheckpointHash) {
  constexpr int32_t kCheckpointStride = 8;
  constexpr size_t kAcceptedLength = 3;
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_linear_state(true)
      .linear_state_num_slots(64)
      .host_num_blocks_by_type({{BlockType::KV, 128}, {BlockType::LINEAR, 64}})
      .enable_prefix_cache(false)
      .enable_disagg_pd(true)
      .instance_is_decode(true)
      .enable_kvcache_store(true)
      .num_embedding_blocks(64)
      .num_speculative_tokens(kAcceptedLength);

  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      kCheckpointStride;

  HierarchyBlockManagerPool pool(options,
                                 /*engine=*/nullptr,
                                 /*dp_size=*/1);
  Sequence sequence =
      make_test_sequence(/*index=*/0, std::vector<int32_t>(9, 103));
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/9));
  sequence.kv_state().set_kv_cache_tokens_num(9);
  sequence.set_last_confirmed_cached_tokens_num(6);
  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/12));
  EXPECT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 0u);

  const Slice<Block> hbm_before = sequence.kv_state().blocks(BlockType::LINEAR);
  const Slice<Block> host_before =
      sequence.host_kv_state().blocks(BlockType::LINEAR);
  ASSERT_EQ(hbm_before.size(), 2u);
  ASSERT_EQ(host_before.size(), 2u);
  const int32_t hbm_checkpoint_id = hbm_before.front().id();
  const int32_t hbm_live_id = hbm_before.back().id();
  const int32_t host_checkpoint_id = host_before.front().id();
  const int32_t host_live_id = host_before.back().id();

  sequence.append_token(104);
  sequence.append_token(105);
  sequence.append_token(106);
  ASSERT_EQ(sequence.last_confirmed_cached_tokens_num(), 11u);
  sequence.set_accepted_length(kAcceptedLength);
  ASSERT_TRUE(pool.allocate(&sequence, sequence.num_tokens()));

  const Slice<XXH3Key> checkpoint_hashes = sequence.linear_state_hashes();
  ASSERT_EQ(checkpoint_hashes.size(), 1u);
  const XXH3Key expected_hash = checkpoint_hashes.front();

  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
            hbm_live_id);
  ASSERT_EQ(sequence.host_kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(sequence.host_kv_state().blocks(BlockType::LINEAR).front().id(),
            host_live_id);
  ASSERT_EQ(HierarchyPoolTestPeer::pending_offload_pair_count(pool), 1u);

  const std::shared_ptr<OffloadBlockPair> pair =
      HierarchyPoolTestPeer::take_pending_offload_pair(pool);
  ASSERT_NE(pair, nullptr);
  EXPECT_EQ(pair->src.id(), hbm_checkpoint_id);
  EXPECT_EQ(pair->dst.id(), host_checkpoint_id);
  EXPECT_EQ(pair->block_type, BlockType::LINEAR);
  EXPECT_EQ(pair->checkpoint_row, 0u);
  EXPECT_TRUE(XXH3Key(pair->src.get_immutable_hash_value()) == expected_hash);
  EXPECT_TRUE(XXH3Key(pair->dst.get_immutable_hash_value()) == expected_hash);

  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      original_chunk_stride;
}

TEST(HierarchyBlockManagerPoolTest,
     DecodeLinearCheckpointRowTracksBoundaryOvershoot) {
  constexpr size_t kCheckpointStride = 8;
  constexpr size_t kAcceptedLength = 3;
  Sequence sequence =
      make_test_sequence(/*index=*/0, std::vector<int32_t>(16, 103));
  sequence.set_accepted_length(kAcceptedLength);

  for (size_t confirmed_tokens = 8; confirmed_tokens <= 11;
       ++confirmed_tokens) {
    sequence.set_last_confirmed_cached_tokens_num(confirmed_tokens);
    const std::optional<size_t> checkpoint_row =
        last_decode_step_checkpoint_row(sequence, kCheckpointStride);
    ASSERT_TRUE(checkpoint_row.has_value());
    EXPECT_EQ(checkpoint_row.value(), 11 - confirmed_tokens);
    EXPECT_TRUE(
        last_decode_step_crosses_checkpoint(sequence, kCheckpointStride));
  }

  sequence.set_last_confirmed_cached_tokens_num(12);
  EXPECT_FALSE(
      last_decode_step_checkpoint_row(sequence, kCheckpointStride).has_value());
  EXPECT_FALSE(
      last_decode_step_crosses_checkpoint(sequence, kCheckpointStride));

  sequence.set_last_confirmed_cached_tokens_num(2049);
  EXPECT_EQ(
      last_decode_step_checkpoint_row(sequence, /*checkpoint_stride=*/512), 2u);
}

TEST(HierarchyBlockManagerPoolTest,
     LinearOptionalStoreMissLeavesSparseHostCheckpoint) {
  BlockManagerPool::Options options = make_flat_kv_options();
  options.enable_linear_state(true)
      .linear_state_num_slots(64)
      .host_num_blocks_by_type({{BlockType::KV, 128}, {BlockType::LINEAR, 64}})
      .enable_kvcache_store(true)
      .prefetch_batch_size(4);

  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 128;

  FakePrefetchEngine engine(/*worker_count=*/1,
                            /*timeout_ms=*/-1,
                            /*batch_size=*/4);
  HierarchyBlockManagerPool pool(options, &engine, /*dp_size=*/1);
  std::vector<int32_t> tokens(513, 97);
  std::shared_ptr<Request> request = make_test_request(tokens);
  Sequence* sequence = request->sequences().front().get();
  bool completed = false;

  pool.prefetch_from_storage(
      request,
      [&completed](std::shared_ptr<Request> /*request*/) { completed = true; });
  ASSERT_NE(engine.result(), nullptr);
  ASSERT_EQ(engine.request().units.size(), 4u);

  std::vector<uint8_t> logical_hits;
  logical_hits.reserve(engine.request().batch_transfer_count(0, 4));
  for (size_t unit = 0; unit < 4; ++unit) {
    logical_hits.emplace_back(uint8_t{1});
    logical_hits.emplace_back(unit == 2 ? uint8_t{0} : uint8_t{1});
  }
  engine.finish_worker_with_logical_hits(/*worker_index=*/0, logical_hits);

  EXPECT_TRUE(completed);
  EXPECT_TRUE(sequence->host_kv_state().prefix_cache_matched());
  EXPECT_EQ(sequence->kv_cache_tokens_num(), 512u);
  const Slice<Block> linear_blocks =
      sequence->host_kv_state().blocks(BlockType::LINEAR);
  ASSERT_EQ(linear_blocks.size(), 1u);
  EXPECT_TRUE(linear_blocks[0].is_valid());
  EXPECT_EQ(sequence->host_kv_state().num_cached_blocks(BlockType::LINEAR), 1u);

  const HostCacheRestorePoint restore =
      pool.select_host_cache_restore(sequence, /*max_copy_units=*/4);
  EXPECT_EQ(restore.restore_target_tokens, 512u);
  EXPECT_EQ(restore.copy_units, 4u);
  pool.deallocate(sequence);

  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      original_chunk_stride;
}

}  // namespace xllm
