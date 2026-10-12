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

#include "composite_block_manager.h"

#include <gtest/gtest.h>

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "core/framework/batch/batch.h"
#include "core/framework/block/block_manager_impl.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/block/concurrent_block_manager_impl.h"
#include "core/framework/block/embedding_block_manager.h"
#include "framework/block/block_manager_pool_test_peer.h"
#include "framework/block/block_utils.h"
#include "framework/block/linear_state_block_manager.h"
#include "framework/config/scheduler_config.h"
#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

namespace {

constexpr uint32_t kManagerTypeBlockManagerImpl = 0;
constexpr uint32_t kManagerTypeSlidingWindowBlockManager = 1;
constexpr uint32_t kMaxTokensPerBatch = 1280;
constexpr uint32_t kCompressedBlockTokenSize = kDsv4CompressedBlockTokenSize;

// C4 and C128 cover the same number of original tokens per logical block, so
// a base token capacity maps to equal typed block counts.
BlockManager::Options MakeCompositeOptions(uint32_t base_num_blocks,
                                           uint32_t block_size,
                                           uint32_t window_size,
                                           uint32_t max_seqs_per_batch) {
  const uint32_t swa_blocks_per_seq =
      static_cast<uint32_t>(get_swa_blocks_per_seq(window_size, block_size));
  const uint32_t burst_blocks =
      (kMaxTokensPerBatch + block_size - 1) / block_size;
  const uint32_t swa_num_blocks = swa_blocks_per_seq * max_seqs_per_batch +
                                  burst_blocks + max_seqs_per_batch + 2;
  const uint32_t compressed_num_blocks =
      base_num_blocks * block_size / kCompressedBlockTokenSize;
  BlockManager::Options opts;
  opts.num_blocks(base_num_blocks)
      .block_size(block_size)
      .sliding_window_size(window_size)
      .swa_blocks_per_seq(swa_blocks_per_seq)
      .swa_num_blocks(swa_num_blocks)
      .c4_num_blocks(compressed_num_blocks)
      .c128_num_blocks(compressed_num_blocks)
      .max_tokens_per_batch(kMaxTokensPerBatch)
      .max_seqs_per_batch(max_seqs_per_batch)
      .manager_types({kManagerTypeSlidingWindowBlockManager,
                      kManagerTypeBlockManagerImpl,
                      kManagerTypeBlockManagerImpl})
      .compress_ratios({0, 4, 128});
  return opts;
}

void set_swa_capacity_for_token_budget(BlockManager::Options* options,
                                       uint32_t max_tokens_per_batch) {
  ASSERT_NE(options, nullptr);
  const uint32_t block_size = options->block_size();
  ASSERT_GT(block_size, 0u);
  const uint32_t burst_blocks =
      (max_tokens_per_batch + block_size - 1) / block_size;
  const uint32_t max_seqs = std::max(options->max_seqs_per_batch(), 1u);
  const uint32_t swa_num_blocks =
      options->swa_blocks_per_seq() * max_seqs + burst_blocks + max_seqs + 2;
  options->max_tokens_per_batch(max_tokens_per_batch)
      .swa_num_blocks(swa_num_blocks);
}

constexpr uint32_t kBaseBlockSize = 128;
constexpr uint32_t kCompressedUnitTokens = kCompressedBlockTokenSize;

inline size_t CeilBlocks(size_t num_tokens, size_t block_size) {
  return (num_tokens + block_size - 1) / block_size;
}

inline size_t ExpectedSwaLogicalBlocks(size_t num_tokens) {
  return CeilBlocks(num_tokens, kBaseBlockSize);
}

// Creates a minimal Sequence for testing (same pattern as batch_test.cpp).
Sequence MakeTestSequence(size_t index,
                          const std::vector<int32_t>& prompt_token_ids) {
  torch::Device device(Platform::type_torch(), 0);
  // Sequence keeps non-owning pointers to these parameters after this helper
  // returns, so they must outlive every sequence created here.
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker = [] {
    StoppingChecker checker;
    checker.set_max_generated_tokens(256);
    return checker;
  }();
  SequenceParams seq_params;
  // Large enough to hold DSV4-scale prompts (>= a full compressed block = 2048
  // tokens). Individual tests can still use short prompts; sequence capacity
  // just has to bound `num_tokens + max_generated_tokens`.
  seq_params.seq_capacity = 65536;
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

// Blocks are keyed by BlockType in the sequence KVCacheState, not by
// sub-manager index. These helpers read the three DSV4 groups by type so the
// tests stay independent of compress_ratios ordering.
std::vector<Block> SwaBlocks(Sequence& seq) {
  const Slice<Block> s = seq.kv_state().blocks(BlockType::SWA);
  return std::vector<Block>(s.begin(), s.end());
}
std::vector<Block> C4Blocks(Sequence& seq) {
  const Slice<Block> s = seq.kv_state().blocks(BlockType::C4);
  return std::vector<Block>(s.begin(), s.end());
}
std::vector<Block> C128Blocks(Sequence& seq) {
  const Slice<Block> s = seq.kv_state().blocks(BlockType::C128);
  return std::vector<Block>(s.begin(), s.end());
}

}  // namespace

TEST(CompositeBlockManagerTest, ExplicitHostAllocationNeverPublishesProgress) {
  BlockManager::Options options;
  options.num_blocks(16).block_size(4).enable_host_offload(true);
  CompositeBlockManager device(build_composite_leaves(options), options);
  auto host_leaves = build_composite_leaves(options);
  host_leaves.at(BlockType::KV).participates_in_admission = false;
  CompositeBlockManager host(
      std::move(host_leaves),
      options,
      CompositeBlockManager::PrefixCachePublishMode::EXPLICIT);
  Sequence sequence = MakeTestSequence(0, std::vector<int32_t>(9, 7));
  ASSERT_TRUE(device.allocate_sequence(&sequence, 9));
  sequence.kv_state().set_kv_cache_tokens_num(8);
  const int32_t device_id = sequence.kv_state().blocks(BlockType::KV)[0].id();

  ASSERT_TRUE(host.allocate_sequence(&sequence, sequence.host_kv_state(), 9));
  EXPECT_EQ(host.num_total_blocks(), 0u);
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::KV), 3u);
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::KV), 0u);
  for (const Block& block : sequence.host_kv_state().blocks(BlockType::KV)) {
    EXPECT_EQ(block.ref_count(), 1u);
  }
  host.cache_for_sequence(&sequence);
  host.cache_for_sequence(&sequence, 8);
  host.cache_full_blocks_for_sequence(&sequence);
  EXPECT_EQ(host.num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(
      host.leaf_entries().at(BlockType::KV).leaf->num_blocks_in_prefix_cache(),
      0u);
  EXPECT_EQ(device.num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::KV), 0u);
  EXPECT_EQ(sequence.host_kv_state().num_cached_blocks(BlockType::KV), 0u);
  for (const Block& block : sequence.kv_state().blocks(BlockType::KV)) {
    EXPECT_EQ(block.ref_count(), 1u);
  }
  for (const Block& block : sequence.host_kv_state().blocks(BlockType::KV)) {
    EXPECT_EQ(block.ref_count(), 1u);
  }

  host.deallocate_for_sequence(&sequence, sequence.host_kv_state());
  EXPECT_EQ(sequence.host_kv_state().num_blocks(BlockType::KV), 3u);
  sequence.host_kv_state().reset();
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::KV)[0].id(), device_id);
  EXPECT_EQ(sequence.kv_state().kv_cache_tokens_num(), 8u);
  EXPECT_EQ(host.num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(host.leaf_entries().at(BlockType::KV).leaf->num_used_blocks(), 0u);
  device.deallocate_for_sequence(&sequence);
  sequence.reset();
  EXPECT_EQ(device.num_blocks_in_prefix_cache(), 2u);
}

TEST(CompositeBlockManagerTest, ExplicitPublicationEnablesHostOnlySharedMatch) {
  BlockManager::Options options;
  options.num_blocks(16).block_size(4).enable_host_offload(true);
  CompositeBlockManager host(
      build_composite_leaves(options),
      options,
      CompositeBlockManager::PrefixCachePublishMode::EXPLICIT);
  Sequence source = MakeTestSequence(0, std::vector<int32_t>(9, 7));
  ASSERT_TRUE(host.allocate_sequence(&source, source.host_kv_state(), 8));
  source.update_block_hashes(4, BlockHasherType::TEXT);
  auto& blocks = *source.host_kv_state().mutable_blocks(BlockType::KV);
  for (size_t index = 0; index < blocks.size(); ++index) {
    blocks[index].set_hash_value(source.block_hashes()[index].data);
  }
  host.cache_blocks(BlockType::KV, blocks);
  EXPECT_EQ(host.num_blocks_in_prefix_cache(), 2u);
  EXPECT_EQ(source.host_kv_state().num_cached_blocks(BlockType::KV), 0u);
  EXPECT_EQ(source.host_kv_state().kv_cache_tokens_num(), 0u);
  for (const Block& block : blocks) {
    EXPECT_EQ(block.ref_count(), 2u);
  }

  Sequence hit = MakeTestSequence(1, std::vector<int32_t>(9, 7));
  host.allocate_shared_for_sequence(&hit, hit.host_kv_state());
  EXPECT_EQ(hit.host_kv_state().kv_cache_tokens_num(), 8u);
  EXPECT_EQ(hit.host_kv_state().num_cached_blocks(BlockType::KV), 2u);
  EXPECT_TRUE(hit.host_kv_state().prefix_cache_matched());
  EXPECT_FALSE(hit.kv_state().has_any_blocks());
  EXPECT_FALSE(hit.kv_state().prefix_cache_matched());
  host.deallocate_for_sequence(&hit, hit.host_kv_state());
  hit.host_kv_state().reset();

  Sequence exact = MakeTestSequence(2, std::vector<int32_t>(8, 7));
  host.allocate_shared_for_sequence(&exact, exact.host_kv_state());
  EXPECT_EQ(exact.host_kv_state().kv_cache_tokens_num(), 4u);
  EXPECT_EQ(exact.host_kv_state().num_cached_blocks(BlockType::KV), 1u);
  host.deallocate_for_sequence(&exact, exact.host_kv_state());
  exact.host_kv_state().reset();
  host.deallocate_for_sequence(&source, source.host_kv_state());
  source.host_kv_state().reset();
  EXPECT_EQ(host.leaf_entries().at(BlockType::KV).leaf->num_used_blocks(), 0u);
}

TEST(CompositeBlockManagerTest, FailedHostGrowthRollsBackAllStagedLeaves) {
  BlockManager::Options options = MakeCompositeOptions(4096, 128, 128, 4);
  options.enable_prefix_cache(true).enable_host_offload(true);
  CompositeBlockManager device(build_composite_leaves(options), options);
  CompositeBlockManager host(
      build_composite_leaves(options),
      options,
      CompositeBlockManager::PrefixCachePublishMode::EXPLICIT);
  Sequence sequence = MakeTestSequence(0, std::vector<int32_t>(129, 7));
  ASSERT_TRUE(device.allocate_sequence(&sequence, 128));
  const int32_t device_id = sequence.kv_state().blocks(BlockType::C4)[0].id();
  BlockManager* c128 = host.leaf_entries().at(BlockType::C128).leaf.get();
  auto held_blocks = c128->allocate(c128->num_free_blocks());
  std::map<BlockType, size_t> used_before;
  for (const auto& [type, entry] : host.leaf_entries()) {
    used_before.emplace(type, entry.leaf->num_used_blocks());
  }

  EXPECT_FALSE(
      host.allocate_sequence(&sequence, sequence.host_kv_state(), 128));
  EXPECT_FALSE(sequence.host_kv_state().has_any_blocks());
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::C4)[0].id(), device_id);
  for (const auto& [type, entry] : host.leaf_entries()) {
    EXPECT_EQ(entry.leaf->num_used_blocks(), used_before.at(type));
  }
  host.deallocate(held_blocks);
  held_blocks.clear();
  ASSERT_TRUE(host.allocate_sequence(&sequence, sequence.host_kv_state(), 128));
  for (const auto& [type, entry] : host.leaf_entries()) {
    EXPECT_EQ(sequence.host_kv_state().blocks(type)[0].ref_count(), 1u);
  }
  host.deallocate_for_sequence(&sequence, sequence.host_kv_state());
  sequence.host_kv_state().reset();
  device.deallocate_for_sequence(&sequence);
  sequence.reset();
}

TEST(CompositeBlockManagerTest, EmptyHostGrowthCannotPassCapacityValidation) {
  class EmptyHostGrowthBlockManager final : public BlockManagerImpl {
   public:
    explicit EmptyHostGrowthBlockManager(const BlockManager::Options& options)
        : BlockManagerImpl(options) {}

    std::optional<std::vector<Block>> allocate_for_sequence(Sequence*,
                                                            KVCacheState&,
                                                            size_t) override {
      return std::vector<Block>{};
    }
  };

  BlockManager::Options options;
  options.num_blocks(16).block_size(4);
  auto leaves = build_composite_leaves(options);
  leaves.at(BlockType::KV).leaf =
      std::make_unique<EmptyHostGrowthBlockManager>(options);
  CompositeBlockManager host(
      std::move(leaves),
      options,
      CompositeBlockManager::PrefixCachePublishMode::EXPLICIT);
  Sequence sequence = MakeTestSequence(0, std::vector<int32_t>(9, 7));
  EXPECT_FALSE(host.allocate_sequence(&sequence, sequence.host_kv_state(), 9));
  EXPECT_FALSE(sequence.host_kv_state().has_any_blocks());
  EXPECT_FALSE(sequence.kv_state().has_any_blocks());
}

TEST(CompositeBlockManagerTest, HostWindowReleaseUsesSequenceProgress) {
  BlockManager::Options options = MakeCompositeOptions(4096, 128, 128, 4);
  options.enable_prefix_cache(true);
  CompositeBlockManager device(build_composite_leaves(options), options);
  CompositeBlockManager host(
      build_composite_leaves(options),
      options,
      CompositeBlockManager::PrefixCachePublishMode::EXPLICIT);
  Sequence sequence = MakeTestSequence(0, std::vector<int32_t>(1025, 7));
  ASSERT_TRUE(device.allocate_sequence(&sequence, 1024));
  ASSERT_TRUE(
      host.allocate_sequence(&sequence, sequence.host_kv_state(), 1024));
  sequence.kv_state().set_kv_cache_tokens_num(1024);
  host.release_out_of_window_for_sequence(&sequence, sequence.host_kv_state());
  const Slice<Block> host_swa = sequence.host_kv_state().blocks(BlockType::SWA);
  ASSERT_EQ(host_swa.size(), 8u);
  for (size_t index = 0; index + 1 < host_swa.size(); ++index) {
    EXPECT_FALSE(host_swa[index].is_valid());
    EXPECT_TRUE(sequence.kv_state().blocks(BlockType::SWA)[index].is_valid());
  }
  EXPECT_TRUE(host_swa.back().is_valid());
  EXPECT_EQ(sequence.host_kv_state().kv_cache_tokens_num(), 0u);
  EXPECT_EQ(host.num_blocks_in_prefix_cache(), 0u);
  host.deallocate_for_sequence(&sequence, sequence.host_kv_state());
  sequence.host_kv_state().reset();
  device.deallocate_for_sequence(&sequence);
  sequence.reset();
}

TEST(CompositeBlockManagerTest, ExplicitDeviceStatePreservesEmbeddingDispatch) {
  BlockManager::Options options;
  options.num_blocks(16).block_size(128);
  auto leaves = build_composite_leaves(options);
  leaves.emplace(
      BlockType::EMBEDDING,
      CompositeBlockManager::LeafEntry{
          std::make_unique<EmbeddingBlockManager>(4, "test"), false, false});
  CompositeBlockManager manager(std::move(leaves), options);
  Sequence sequence = MakeTestSequence(0, std::vector<int32_t>(1025, 7));
  ASSERT_TRUE(manager.allocate_sequence(&sequence, sequence.kv_state(), 128));
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::EMBEDDING), 1u);
  const int32_t embedding_id =
      sequence.kv_state().blocks(BlockType::EMBEDDING)[0].id();
  ASSERT_TRUE(manager.allocate_sequence(&sequence, sequence.kv_state(), 1024));
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::EMBEDDING), 1u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::EMBEDDING)[0].id(),
            embedding_id);
  manager.deallocate_for_sequence(&sequence);
  sequence.reset();
}

TEST(CompositeBlockManagerTest, EmbeddingLeafUsesExplicitState) {
  for (const bool concurrent : {false, true}) {
    SCOPED_TRACE(concurrent);
    std::unique_ptr<BlockManager> manager =
        std::make_unique<EmbeddingBlockManager>(3, "test");
    if (concurrent) {
      manager =
          std::make_unique<ConcurrentBlockManagerImpl>(std::move(manager));
    }
    Sequence sequence = MakeTestSequence(0, std::vector<int32_t>(65, 7));
    sequence.kv_state().add_blocks(BlockType::EMBEDDING, manager->allocate(1));
    const int32_t sequence_block_id =
        sequence.kv_state().blocks(BlockType::EMBEDDING)[0].id();
    KVCacheState kv_state;

    auto allocated = manager->allocate_for_sequence(&sequence, kv_state, 64);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    EXPECT_NE(allocated->front().id(), sequence_block_id);
    kv_state.add_blocks(BlockType::EMBEDDING, *allocated);
    allocated->clear();

    allocated = manager->allocate_for_sequence(&sequence, kv_state, 128);
    ASSERT_TRUE(allocated.has_value());
    EXPECT_TRUE(allocated->empty());
    EXPECT_EQ(kv_state.num_blocks(BlockType::EMBEDDING), 1u);
    EXPECT_EQ(sequence.kv_state().blocks(BlockType::EMBEDDING)[0].id(),
              sequence_block_id);
    manager->deallocate(kv_state.blocks(BlockType::EMBEDDING));
    kv_state.reset();
    manager->deallocate(sequence.kv_state().blocks(BlockType::EMBEDDING));
    sequence.reset();
    EXPECT_EQ(manager->num_used_blocks(), 0u);
    EXPECT_EQ(manager->num_free_blocks(), manager->num_total_blocks());
  }
}

TEST(CompositeBlockManagerTest, SlidingWindowLeafAcceptsNullSequence) {
  for (const bool concurrent : {false, true}) {
    SCOPED_TRACE(concurrent);
    BlockManager::Options options = MakeCompositeOptions(512, 128, 256, 1);
    options.enable_disagg_pd(concurrent).enable_prefix_cache(false);
    CompositeBlockManager manager(build_composite_leaves(options), options);
    BlockManager* leaf = manager.leaf_entries().at(BlockType::SWA).leaf.get();
    KVCacheState kv_state;

    EXPECT_FALSE(
        leaf->allocate_for_sequence(nullptr, kv_state, 128).has_value());
    leaf->release_out_of_window(nullptr, kv_state);
    EXPECT_EQ(kv_state.num_blocks(BlockType::SWA), 0u);
    EXPECT_EQ(leaf->num_used_blocks(), 0u);
  }
}

TEST(CompositeBlockManagerTest, AllocateForSequence_SingleSeq) {
  const uint32_t base_block_size = kBaseBlockSize;
  const uint32_t base_num_blocks = 4096;  // C4/C128: 256 blocks each.
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, base_block_size, window_size, max_seqs_per_batch);

  CompositeBlockManager manager(build_composite_leaves(opts), opts);
  EXPECT_TRUE(manager.is_composite());
  EXPECT_EQ(manager.num_sub_managers(), 3u);

  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(1024, 1));
  const size_t num_tokens = 1024;

  EXPECT_TRUE(manager.allocate_sequence(&seq, num_tokens));

  const std::vector<Block> swa = SwaBlocks(seq);
  const std::vector<Block> c4 = C4Blocks(seq);
  const std::vector<Block> c128 = C128Blocks(seq);

  // SlidingWindow group: logical block count follows sequence length.
  EXPECT_EQ(swa.size(), ExpectedSwaLogicalBlocks(num_tokens));
  for (const auto& b : swa) {
    EXPECT_TRUE(b.is_valid());
    EXPECT_EQ(b.size(), base_block_size);
  }

  // C4 logical block size is 2048 original tokens.
  const size_t expected_blocks_1 =
      CeilBlocks(num_tokens, kCompressedUnitTokens);
  EXPECT_EQ(c4.size(), expected_blocks_1);
  for (const auto& b : c4) {
    EXPECT_TRUE(b.is_valid());
    EXPECT_EQ(b.size(), kCompressedUnitTokens);
  }

  // C128 logical block size is also 2048 original tokens.
  const size_t expected_blocks_2 =
      CeilBlocks(num_tokens, kCompressedUnitTokens);
  EXPECT_EQ(c128.size(), expected_blocks_2);
  for (const auto& b : c128) {
    EXPECT_TRUE(b.is_valid());
    EXPECT_EQ(b.size(), kCompressedUnitTokens);
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, AllocateForSequence_DifferentBatchSeqs) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;
  const uint32_t sliding_window_blocks_per_sequence = static_cast<uint32_t>(
      get_swa_blocks_per_seq(window_size, kBaseBlockSize));

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  // Seq1: 1024 tokens. Both compressed groups need one logical block.
  Sequence seq1 = MakeTestSequence(0, std::vector<int32_t>(1024, 1));
  EXPECT_TRUE(manager.allocate_sequence(&seq1, 1024));
  const std::vector<Block> s1_swa = SwaBlocks(seq1);
  const std::vector<Block> s1_c4 = C4Blocks(seq1);
  const std::vector<Block> s1_c128 = C128Blocks(seq1);
  EXPECT_EQ(s1_swa.size(), ExpectedSwaLogicalBlocks(1024));
  EXPECT_EQ(s1_c4.size(), CeilBlocks(1024, kCompressedUnitTokens));
  EXPECT_EQ(s1_c128.size(), CeilBlocks(1024, kCompressedUnitTokens));

  // Seq2: 1400 tokens. Both compressed groups need one 2048-token logical
  // block. Keep total SWA logical blocks within the dynamic pool budget
  // derived from max_tokens_per_batch.
  Sequence seq2 = MakeTestSequence(1, std::vector<int32_t>(1400, 1));
  EXPECT_TRUE(manager.allocate_sequence(&seq2, 1400));
  const std::vector<Block> s2_swa = SwaBlocks(seq2);
  const std::vector<Block> s2_c4 = C4Blocks(seq2);
  const std::vector<Block> s2_c128 = C128Blocks(seq2);
  EXPECT_EQ(s2_swa.size(), ExpectedSwaLogicalBlocks(1400));
  EXPECT_EQ(s2_c4.size(), CeilBlocks(1400, kCompressedUnitTokens));
  EXPECT_EQ(s2_c128.size(), CeilBlocks(1400, kCompressedUnitTokens));

  // Blocks allocated to different seqs must not overlap (distinct block ids).
  std::set<int32_t> ids1_0, ids1_1, ids1_2, ids2_0, ids2_1, ids2_2;
  for (const auto& b : s1_swa) ids1_0.insert(b.id());
  for (const auto& b : s1_c4) ids1_1.insert(b.id());
  for (const auto& b : s1_c128) ids1_2.insert(b.id());
  for (const auto& b : s2_swa) ids2_0.insert(b.id());
  for (const auto& b : s2_c4) ids2_1.insert(b.id());
  for (const auto& b : s2_c128) ids2_2.insert(b.id());
  for (int32_t id : ids1_0) EXPECT_EQ(ids2_0.count(id), 0u);
  for (int32_t id : ids1_1) EXPECT_EQ(ids2_1.count(id), 0u);
  for (int32_t id : ids1_2) EXPECT_EQ(ids2_2.count(id), 0u);
  const int32_t max_swa_block_id =
      sliding_window_blocks_per_sequence * max_seqs_per_batch +
      CeilBlocks(kMaxTokensPerBatch, kBaseBlockSize) + max_seqs_per_batch + 1;
  for (int32_t id : ids1_0) EXPECT_LE(id, max_swa_block_id);
  for (int32_t id : ids2_0) EXPECT_LE(id, max_swa_block_id);

  manager.deallocate_for_sequence(&seq1);
  manager.deallocate_for_sequence(&seq2);
}

TEST(CompositeBlockManagerTest, AllocateForSequence_GrowSameSeq) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1, 2, 3});
  // Both compressed leaves use one common 2048-token logical unit.
  EXPECT_TRUE(manager.allocate_sequence(&seq, 600));
  EXPECT_EQ(SwaBlocks(seq).size(), ExpectedSwaLogicalBlocks(600));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(600, kCompressedUnitTokens));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(600, kCompressedUnitTokens));

  // Growing within the same compressed unit must not allocate another block.
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1200));
  EXPECT_EQ(SwaBlocks(seq).size(), ExpectedSwaLogicalBlocks(1200));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(1200, kCompressedUnitTokens));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(1200, kCompressedUnitTokens));

  // No growth: still 1200 tokens, block counts unchanged.
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1200));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(1200, kCompressedUnitTokens));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(1200, kCompressedUnitTokens));

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, AllocateContinuesAfterSatisfiedTokenManager) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  opts.compress_ratios({0, 128, 4});
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1024));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(1024, kCompressedUnitTokens));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(1024, kCompressedUnitTokens));

  EXPECT_TRUE(manager.allocate_sequence(&seq, 1500));
  EXPECT_EQ(C128Blocks(seq).size(), CeilBlocks(1500, kCompressedUnitTokens));
  EXPECT_EQ(C4Blocks(seq).size(), CeilBlocks(1500, kCompressedUnitTokens));

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, AllocateForSequence_NullSeqReturnsFalse) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);
  EXPECT_FALSE(manager.allocate_sequence(nullptr, 10));
}

TEST(CompositeBlockManagerTest, FailedGrowthRollsBackNewBlocks) {
  BlockManager::Options opts = MakeCompositeOptions(/*base_num_blocks=*/256,
                                                    kBaseBlockSize,
                                                    /*window_size=*/12,
                                                    /*max_seqs_per_batch=*/4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  ASSERT_TRUE(manager.allocate_sequence(&seq, 1024));
  const size_t used_before = manager.num_used_blocks();
  const std::vector<Block> before_swa = SwaBlocks(seq);
  const size_t swa_blocks_before = before_swa.size();
  const size_t c4_blocks_before = C4Blocks(seq).size();
  const size_t c128_blocks_before = C128Blocks(seq).size();
  std::vector<int32_t> swa_ids_before;
  swa_ids_before.reserve(before_swa.size());
  for (const auto& block : before_swa) {
    ASSERT_TRUE(block.is_valid());
    swa_ids_before.push_back(block.id());
  }
  seq.kv_state().incr_kv_cache_tokens_num(1024);

  EXPECT_FALSE(manager.allocate_sequence(&seq, 4096));
  EXPECT_EQ(manager.num_used_blocks(), used_before);

  const std::vector<Block> after_swa = SwaBlocks(seq);
  EXPECT_EQ(after_swa.size(), swa_blocks_before);
  EXPECT_EQ(C4Blocks(seq).size(), c4_blocks_before);
  EXPECT_EQ(C128Blocks(seq).size(), c128_blocks_before);
  ASSERT_EQ(after_swa.size(), swa_ids_before.size());
  for (size_t i = 0; i < after_swa.size(); ++i) {
    EXPECT_TRUE(after_swa[i].is_valid());
    EXPECT_EQ(after_swa[i].id(), swa_ids_before[i]);
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, DeallocateToleratesRolledBackEmptySequence) {
  BlockManager::Options opts = MakeCompositeOptions(/*base_num_blocks=*/128,
                                                    kBaseBlockSize,
                                                    /*window_size=*/12,
                                                    /*max_seqs_per_batch=*/4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});

  EXPECT_FALSE(manager.allocate_sequence(&seq, 4096));
  EXPECT_NO_FATAL_FAILURE(manager.deallocate_for_sequence(&seq));
  EXPECT_FALSE(seq.kv_state().has_multi_block_export());
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

// Verifies that when seq token count increases, CompositeBlockManager correctly
// adds blocks (only appends new blocks; existing block ids are preserved).
TEST(CompositeBlockManagerTest, TokenIncrease_AddsBlocksIncrementally) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  std::vector<size_t> token_steps = {100, 600, 1200, 2000, 2400};

  std::vector<std::set<int32_t>> prev_ids_1;
  std::vector<std::set<int32_t>> prev_ids_2;

  for (size_t num_tokens : token_steps) {
    EXPECT_TRUE(manager.allocate_sequence(&seq, num_tokens));

    const std::vector<Block> c4 = C4Blocks(seq);
    const std::vector<Block> c128 = C128Blocks(seq);

    const size_t expect_1 = CeilBlocks(num_tokens, kCompressedUnitTokens);
    const size_t expect_2 = CeilBlocks(num_tokens, kCompressedUnitTokens);
    EXPECT_EQ(c4.size(), expect_1)
        << "num_tokens=" << num_tokens << " C4 block count";
    EXPECT_EQ(c128.size(), expect_2)
        << "num_tokens=" << num_tokens << " C128 block count";

    // Check that previously allocated block ids are still present (only
    // append).
    std::set<int32_t> ids_1, ids_2;
    for (const auto& b : c4) ids_1.insert(b.id());
    for (const auto& b : c128) ids_2.insert(b.id());
    for (const auto& prev : prev_ids_1) {
      for (int32_t id : prev) EXPECT_GT(ids_1.count(id), 0u);
    }
    for (const auto& prev : prev_ids_2) {
      for (int32_t id : prev) EXPECT_GT(ids_2.count(id), 0u);
    }
    prev_ids_1.push_back(ids_1);
    prev_ids_2.push_back(ids_2);
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, Dsv4CompressedAllocationUsesPlatformUnits) {
  BlockManager::Options opts = MakeCompositeOptions(/*base_num_blocks=*/4096,
                                                    kBaseBlockSize,
                                                    /*window_size=*/128,
                                                    /*max_seqs_per_batch=*/4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  for (const auto& [num_tokens, expected_blocks] :
       std::vector<std::pair<size_t, size_t>>{{kCompressedUnitTokens - 1, 1},
                                              {kCompressedUnitTokens, 1},
                                              {kCompressedUnitTokens + 1, 2}}) {
    SCOPED_TRACE(num_tokens);
    ASSERT_TRUE(manager.allocate_sequence(&seq, num_tokens));
    EXPECT_EQ(C4Blocks(seq).size(), expected_blocks);
    EXPECT_EQ(C128Blocks(seq).size(), expected_blocks);
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, SlidingWindowReleasesSkippedPhysicalBlocks) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t sliding_window_blocks_per_sequence = 3;
  const uint32_t window_size =
      sliding_window_blocks_per_sequence * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  const size_t window_tokens =
      static_cast<size_t>(sliding_window_blocks_per_sequence) * kBaseBlockSize;

  EXPECT_TRUE(manager.allocate_sequence(&seq, window_tokens));
  const std::vector<Block> initial = SwaBlocks(seq);
  ASSERT_EQ(initial.size(), sliding_window_blocks_per_sequence);
  seq.kv_state().incr_kv_cache_tokens_num(window_tokens);

  std::vector<int32_t> initial_ids;
  initial_ids.reserve(initial.size());
  for (const auto& block : initial) {
    initial_ids.push_back(block.id());
  }

  EXPECT_TRUE(
      manager.allocate_sequence(&seq, window_tokens + 2 * kBaseBlockSize));
  const std::vector<Block> boundary = SwaBlocks(seq);
  ASSERT_EQ(boundary.size(), sliding_window_blocks_per_sequence + 2);
  for (size_t i = 0; i < initial_ids.size(); ++i) {
    EXPECT_EQ(boundary[i].id(), initial_ids[i]);
  }
  seq.kv_state().incr_kv_cache_tokens_num(2 * kBaseBlockSize);

  EXPECT_TRUE(
      manager.allocate_sequence(&seq, window_tokens + 3 * kBaseBlockSize));
  const std::vector<Block> exceeded = SwaBlocks(seq);
  ASSERT_EQ(exceeded.size(), sliding_window_blocks_per_sequence + 3);
  EXPECT_FALSE(exceeded[0].is_valid());
  EXPECT_FALSE(exceeded[1].is_valid());
  EXPECT_EQ(exceeded[0].id(), -1);
  EXPECT_EQ(exceeded[1].id(), -1);
  for (size_t i = 2; i < initial_ids.size(); ++i) {
    EXPECT_EQ(exceeded[i].id(), initial_ids[i]);
  }
  for (size_t i = initial_ids.size(); i < exceeded.size(); ++i) {
    EXPECT_TRUE(exceeded[i].is_valid());
  }

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, DeallocateSliceDispatchesToOwnerManagers) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  opts.enable_prefix_cache(false);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1500));
  EXPECT_GT(manager.num_used_blocks(), 0u);

  std::vector<Block> flat_blocks;
  for (const BlockType type :
       {BlockType::SWA, BlockType::C4, BlockType::C128}) {
    const Slice<Block> manager_blocks = seq.kv_state().blocks(type);
    flat_blocks.insert(
        flat_blocks.end(), manager_blocks.begin(), manager_blocks.end());
  }

  manager.deallocate(flat_blocks);
  EXPECT_EQ(manager.num_used_blocks(), 0u);

  seq.reset();
}

TEST(CompositeBlockManagerTest,
     DeallocateSliceDispatchesWithoutInflatingRefCount) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  Sequence seq = MakeTestSequence(0, {1});
  EXPECT_TRUE(manager.allocate_sequence(&seq, 1500));
  EXPECT_GT(manager.num_used_blocks(), 0u);

  std::vector<const Block*> flat_blocks;
  for (const BlockType type :
       {BlockType::SWA, BlockType::C4, BlockType::C128}) {
    const Slice<Block> manager_blocks = seq.kv_state().blocks(type);
    for (const auto& block : manager_blocks) {
      flat_blocks.push_back(&block);
    }
  }

  for (const Block* block : flat_blocks) {
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(block->ref_count(), 1u);
  }

  for (const Block* block : flat_blocks) {
    manager.deallocate(Slice<Block>(block, 1));
  }
  EXPECT_EQ(manager.num_used_blocks(), 0u);

  for (const Block* block : flat_blocks) {
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(block->ref_count(), 1u);
  }

  seq.reset();
}

TEST(CompositeBlockManagerTest, CapacityStatsUseBaseBlockUnits) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t total = manager.num_total_blocks();
  const size_t base_blocks_per_compressed_unit =
      kCompressedUnitTokens / kBaseBlockSize;
  EXPECT_EQ(total,
            (opts.c4_num_blocks() - 1u) * base_blocks_per_compressed_unit);

  // Free (no sequence yet) equals total; used is 0.
  EXPECT_EQ(manager.num_free_blocks(), total);
  EXPECT_EQ(manager.num_used_blocks(), 0u);

  // After allocating one sequence, used remains normalized to base blocks.
  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(1024, 1));
  ASSERT_TRUE(manager.allocate_sequence(&seq, 1024));
  const size_t c4_used = C4Blocks(seq).size();  // capacity leaf's used count
  EXPECT_EQ(manager.num_used_blocks(),
            c4_used * base_blocks_per_compressed_unit);
  EXPECT_EQ(manager.num_free_blocks(),
            total - c4_used * base_blocks_per_compressed_unit);

  manager.deallocate_for_sequence(&seq);
}

// DSV4 prefix cache: a fresh sequence with the same prompt as a previously
// released sequence should mount shared blocks from all three prefix-cache
// leaves (SWA / C4 / C128). The composite takes the cross-leaf min hit length
// clamped to a common compressed block, so the prompt has to span at least
// 2048 tokens for the hit to be non-trivial. Uses a two-unit
// prompt so we get a meaningful hit even after the exact-repeat pop.
TEST(CompositeBlockManagerTest, Dsv4PrefixCacheHitOnRepeatedPrefix) {
  const uint32_t base_num_blocks = 4096;
  // Sliding window covers a couple compressed units worth of tokens so the SWA
  // tail-continuity check has room to succeed even after the exact-repeat pop.
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  // max_tokens_per_batch has to accommodate the prompt so allocate_sequence
  // does not exceed the SWA burst budget.
  set_swa_capacity_for_token_budget(&opts, 3 * kCompressedUnitTokens);
  ASSERT_TRUE(opts.enable_prefix_cache());
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  // First sequence: a 2*C128 prompt (32768 tokens) so both C4 and C128 have
  // multiple full blocks worth of cacheable prefix. Mark all tokens as
  // forwarded so the pre-grow hook (fired during the NEXT allocate) has data
  // to insert -- and we drive one extra allocate to trigger it explicitly.
  const size_t num_tokens = 2 * kCompressedUnitTokens;
  const std::vector<int32_t> prompt(num_tokens, 7);
  Sequence seq_first = MakeTestSequence(0, prompt);
  ASSERT_TRUE(manager.allocate_sequence(&seq_first, num_tokens));
  seq_first.kv_state().incr_kv_cache_tokens_num(num_tokens);
  // deallocate_for_sequence runs the pre-grow hook once more (via
  // cache_for_sequence's fan-out), flushing the residual full blocks.
  manager.deallocate_for_sequence(&seq_first);
  seq_first.reset();

  // The prefix cache under SWA / C4 / C128 should now hold the prompt's full
  // blocks. On a second sequence with the same prompt, admission mounts from
  // all three leaves and pins kv_cache_tokens_num to the safe min hit.
  Sequence seq_hit = MakeTestSequence(1, prompt);
  manager.allocate_shared_for_sequence(&seq_hit);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::SWA), 0u);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::C4), 0u);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::C128), 0u);
  const size_t kv_tokens = seq_hit.kv_state().kv_cache_tokens_num();
  EXPECT_GT(kv_tokens, 0u);
  EXPECT_LT(kv_tokens, num_tokens);  // exact-repeat pop kept at least one c128
  EXPECT_EQ(kv_tokens % kCompressedUnitTokens, 0u)
      << "safe_hit_tokens must be a multiple of the compressed unit";
  // SWA vector length matches safe_hit / base and the tail carries valid
  // blocks.
  const std::vector<Block> hit_swa = SwaBlocks(seq_hit);
  EXPECT_EQ(hit_swa.size(), kv_tokens / kBaseBlockSize);
  EXPECT_TRUE(hit_swa.back().is_valid());

  manager.deallocate_for_sequence(&seq_hit);
}

TEST(CompositeBlockManagerTest,
     Dsv4PrefixCacheMatchesAtCompressedUnitBoundary) {
  const size_t unit_tokens = kCompressedUnitTokens;
  const std::vector<std::pair<size_t, size_t>> cases = {
      {unit_tokens - 1, 0},
      {unit_tokens, 0},
      {unit_tokens + 1, unit_tokens},
      {2 * unit_tokens, unit_tokens},
      {2 * unit_tokens + 1, 2 * unit_tokens},
  };
  for (const auto& [prompt_tokens, expected_hit_tokens] : cases) {
    SCOPED_TRACE(prompt_tokens);
    const uint32_t base_num_blocks = 4096;
    const uint32_t window_size = kBaseBlockSize;
    const uint32_t max_seqs_per_batch = 4;
    BlockManager::Options options = MakeCompositeOptions(
        base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
    set_swa_capacity_for_token_budget(&options, 3 * kCompressedUnitTokens);
    CompositeBlockManager manager(build_composite_leaves(options), options);

    const std::vector<int32_t> cold_prompt(prompt_tokens + 1, 7);
    Sequence cold = MakeTestSequence(0, cold_prompt);
    ASSERT_TRUE(manager.allocate_sequence(&cold, cold_prompt.size()));
    cold.kv_state().incr_kv_cache_tokens_num(cold_prompt.size());
    manager.deallocate_for_sequence(&cold);
    cold.reset();

    const size_t published_units = prompt_tokens / unit_tokens;
    EXPECT_EQ(manager.leaf_entries()
                  .at(BlockType::C4)
                  .leaf->num_blocks_in_prefix_cache(),
              published_units);
    EXPECT_EQ(manager.leaf_entries()
                  .at(BlockType::C128)
                  .leaf->num_blocks_in_prefix_cache(),
              published_units);

    Sequence hot = MakeTestSequence(1, std::vector<int32_t>(prompt_tokens, 7));
    manager.allocate_shared_for_sequence(&hot);
    EXPECT_EQ(hot.kv_state().kv_cache_tokens_num(), expected_hit_tokens);
    EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C4),
              expected_hit_tokens / unit_tokens);
    EXPECT_EQ(hot.kv_state().shared_blocks_num(BlockType::C128),
              expected_hit_tokens / unit_tokens);
    manager.deallocate_for_sequence(&hot);
  }
}

// DSV4 prefix cache: a fresh sequence with a completely different prompt should
// miss cleanly (no shared mount, kv_cache_tokens_num unchanged).
TEST(CompositeBlockManagerTest, Dsv4PrefixCacheMissCleanly) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  set_swa_capacity_for_token_budget(&opts, 3 * kCompressedUnitTokens);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t num_tokens = 2 * kCompressedUnitTokens;
  const std::vector<int32_t> prompt_a(num_tokens, 7);
  Sequence seq_a = MakeTestSequence(0, prompt_a);
  ASSERT_TRUE(manager.allocate_sequence(&seq_a, num_tokens));
  seq_a.kv_state().incr_kv_cache_tokens_num(num_tokens);
  manager.deallocate_for_sequence(&seq_a);
  seq_a.reset();

  // A sequence with a totally different prompt should not share any block.
  std::vector<int32_t> prompt_b(num_tokens, 0);
  for (size_t i = 0; i < prompt_b.size(); ++i) {
    prompt_b[i] = static_cast<int32_t>(i + 100);
  }
  Sequence seq_b = MakeTestSequence(1, prompt_b);
  manager.allocate_shared_for_sequence(&seq_b);
  EXPECT_EQ(seq_b.kv_state().shared_blocks_num(BlockType::SWA), 0u);
  EXPECT_EQ(seq_b.kv_state().shared_blocks_num(BlockType::C4), 0u);
  EXPECT_EQ(seq_b.kv_state().shared_blocks_num(BlockType::C128), 0u);
  EXPECT_EQ(seq_b.kv_state().kv_cache_tokens_num(), 0u);

  manager.deallocate_for_sequence(&seq_b);
}

TEST(CompositeBlockManagerTest, Dsv4PrefixCacheEvictsAtC128Capacity) {
  // C128 has four physical blocks and each prompt consumes two. The third
  // distinct prompt must evict the first prompt from both compressed leaves.
  const uint32_t base_num_blocks =
      4 * kCompressedBlockTokenSize / kBaseBlockSize;
  const uint32_t window_size = 4 * kBaseBlockSize;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, /*max_seqs_per_batch=*/1);
  set_swa_capacity_for_token_budget(&opts, 2 * kCompressedUnitTokens);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t num_tokens = 2 * kCompressedUnitTokens;
  for (int32_t value : {11, 22, 33}) {
    Sequence seq =
        MakeTestSequence(value, std::vector<int32_t>(num_tokens, value));
    ASSERT_TRUE(manager.allocate_sequence(&seq, num_tokens));
    seq.kv_state().incr_kv_cache_tokens_num(num_tokens);
    manager.deallocate_for_sequence(&seq);
    seq.reset();
  }

  Sequence first = MakeTestSequence(99, std::vector<int32_t>(num_tokens, 11));
  manager.allocate_shared_for_sequence(&first);
  EXPECT_EQ(first.kv_state().shared_blocks_num(BlockType::C4), 0u);
  EXPECT_EQ(first.kv_state().shared_blocks_num(BlockType::C128), 0u);
  EXPECT_EQ(first.kv_state().kv_cache_tokens_num(), 0u);
  manager.deallocate_for_sequence(&first);
}

// SWA slid-out blocks should enter the prefix cache via the pre-grow hook (v2b:
// hook fires at every allocate_sequence, insertion is incremental and cached
// blocks survive slid-out release through the ref<=2u path). A follow-up
// sequence with the same prompt should hit those cached blocks.
TEST(CompositeBlockManagerTest, SlidingWindowSlidOutBlocksEnterPrefixCache) {
  const uint32_t base_num_blocks = 4096;
  // Window covers a small number of base blocks so slide-out happens quickly.
  const uint32_t sliding_window_blocks_per_sequence = 3;
  const uint32_t window_size =
      sliding_window_blocks_per_sequence * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  set_swa_capacity_for_token_budget(&opts, 3 * kCompressedUnitTokens);
  ASSERT_TRUE(opts.enable_prefix_cache());
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  // Prompt spans two full compressed units so the min-across-leaves gate can
  // survive AND the exact-repeat pop (one compressed unit) still leaves shared
  // blocks behind. A single-c128 prompt would end up entirely popped.
  const size_t num_tokens = 2 * kCompressedUnitTokens;
  const std::vector<int32_t> prompt(num_tokens, 7);
  Sequence seq_first = MakeTestSequence(0, prompt);
  ASSERT_TRUE(manager.allocate_sequence(&seq_first, num_tokens));
  seq_first.kv_state().incr_kv_cache_tokens_num(num_tokens);
  // The SWA leaf should have released everything but the last window-sized
  // slab back to the pool (as invalid placeholders), and those blocks landed
  // in the prefix cache via the pre-grow hook before release.
  manager.deallocate_for_sequence(&seq_first);
  seq_first.reset();

  // A second sequence with the same prompt should hit the cached SWA tail.
  Sequence seq_hit = MakeTestSequence(1, prompt);
  manager.allocate_shared_for_sequence(&seq_hit);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::SWA), 0u);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::C4), 0u);
  EXPECT_GT(seq_hit.kv_state().shared_blocks_num(BlockType::C128), 0u);

  manager.deallocate_for_sequence(&seq_hit);
}

TEST(CompositeBlockManagerTest,
     SlidingWindowReclaimsUncachedBlockBeforeFullCacheUnit) {
#if defined(USE_MLU)
  GTEST_SKIP() << "Requires a compressed unit larger than the base block";
#endif
  const uint32_t window_size = 2 * kBaseBlockSize;
  BlockManager::Options opts = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      kBaseBlockSize,
      window_size,
      /*max_seqs_per_batch=*/1);
  opts.enable_prefix_cache(true).swa_num_blocks(
      /*three live blocks plus padding=*/4);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t first_chunk_tokens = 3 * kBaseBlockSize;
  const size_t second_chunk_tokens = 4 * kBaseBlockSize;
  Sequence seq =
      MakeTestSequence(0, std::vector<int32_t>(second_chunk_tokens, 7));

  ASSERT_TRUE(manager.allocate_sequence(&seq, first_chunk_tokens));
  BlockManager* swa_leaf = manager.leaf_entries().at(BlockType::SWA).leaf.get();
  ASSERT_NE(swa_leaf, nullptr);
  EXPECT_EQ(swa_leaf->num_free_blocks(), 0u);

  seq.kv_state().incr_kv_cache_tokens_num(first_chunk_tokens);

  // The first block is outside the two-block window. No compressed-cache unit
  // is complete yet, so the next growth releases it directly and reuses its
  // physical id without publishing a partial DSV4 prefix.
  ASSERT_TRUE(manager.allocate_sequence(&seq, second_chunk_tokens));
  const std::vector<Block> swa_blocks = SwaBlocks(seq);
  ASSERT_EQ(swa_blocks.size(), 4u);
  EXPECT_FALSE(swa_blocks.front().is_valid());
  EXPECT_TRUE(swa_blocks.back().is_valid());
  EXPECT_EQ(swa_leaf->num_free_blocks(), 0u);
  EXPECT_EQ(swa_leaf->num_blocks_in_prefix_cache(), 0u);

  manager.deallocate_for_sequence(&seq);
}

// The post-grow hook advances KVCacheState::num_cached_blocks incrementally.
// Newly allocated blocks are present by then, but the token cursor limits the
// published range to blocks completed by the preceding forward.
TEST(CompositeBlockManagerTest, Dsv4PrefixCachePostGrowCursorAdvances) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  set_swa_capacity_for_token_budget(&opts, 4 * kCompressedUnitTokens);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  // Two-chunk prompt. Chunk 1 is one compressed unit wide.
  const size_t chunk = kCompressedUnitTokens;
  const size_t total = 2 * chunk;
  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(total, 7));

  // Chunk 1: allocate + forward.
  ASSERT_TRUE(manager.allocate_sequence(&seq, chunk));
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::SWA), 0u)
      << "post-grow hook has nothing to cache on the first allocate: kv=0";
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C4), 0u);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C128), 0u);
  seq.kv_state().incr_kv_cache_tokens_num(chunk);

  // Chunk 2 allocation triggers the post-grow hook, which now sees kv=chunk
  // worth of forwarded tokens and inserts them.
  ASSERT_TRUE(manager.allocate_sequence(&seq, total));
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::SWA),
            chunk / kBaseBlockSize);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C4),
            chunk / kCompressedUnitTokens);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C128),
            chunk / kCompressedUnitTokens);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::SWA)
                .leaf->num_blocks_in_prefix_cache(),
            4u);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::C4)
                .leaf->num_blocks_in_prefix_cache(),
            1u);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::C128)
                .leaf->num_blocks_in_prefix_cache(),
            1u);

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest, Dsv4PrefixCacheSkipsPartialCacheUnitTail) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  set_swa_capacity_for_token_budget(&opts, 2 * kCompressedUnitTokens);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t completed_tokens = kCompressedUnitTokens + kBaseBlockSize;
  Sequence seq =
      MakeTestSequence(0, std::vector<int32_t>(completed_tokens, 17));
  ASSERT_TRUE(manager.allocate_sequence(&seq, completed_tokens));
  seq.kv_state().incr_kv_cache_tokens_num(completed_tokens);
  manager.cache_for_sequence(&seq);

  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::SWA)
                .leaf->num_blocks_in_prefix_cache(),
            1u);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::C4)
                .leaf->num_blocks_in_prefix_cache(),
            1u);
  EXPECT_EQ(manager.leaf_entries()
                .at(BlockType::C128)
                .leaf->num_blocks_in_prefix_cache(),
            1u);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::SWA),
            kCompressedUnitTokens / kBaseBlockSize);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C4),
            kCompressedUnitTokens / kCompressedUnitTokens);
  EXPECT_EQ(seq.kv_state().num_cached_blocks(BlockType::C128), 1u);

  manager.deallocate_for_sequence(&seq);
}

// v2b: exact-repeat safe-hit hits the pop path -- when a fresh sequence has a
// prompt whose entire length is cached, safe_hit_tokens is popped by one
// c128 block worth of tokens so the forward has data to process. The test
// verifies kv_cache_tokens_num after mount is strictly less than the prompt.
TEST(CompositeBlockManagerTest, Dsv4PrefixCacheExactRepeatPopsOneC128) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  set_swa_capacity_for_token_budget(&opts, 4 * kCompressedUnitTokens);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t num_tokens = 3 * kCompressedUnitTokens;
  const std::vector<int32_t> prompt(num_tokens, 42);
  Sequence seq_first = MakeTestSequence(0, prompt);
  ASSERT_TRUE(manager.allocate_sequence(&seq_first, num_tokens));
  seq_first.kv_state().incr_kv_cache_tokens_num(num_tokens);
  manager.deallocate_for_sequence(&seq_first);
  seq_first.reset();

  Sequence seq_hit = MakeTestSequence(1, prompt);
  manager.allocate_shared_for_sequence(&seq_hit);
  // Full-length hit → pop one c128 block. Expect kv = num_tokens - c128.
  EXPECT_EQ(seq_hit.kv_state().kv_cache_tokens_num(),
            num_tokens - kCompressedUnitTokens);

  manager.deallocate_for_sequence(&seq_hit);
}

TEST(CompositeBlockManagerTest,
     Dsv4ExactRepeatRejectsEvictedPreviousSwaWindow) {
  const uint32_t window_size = 4 * kBaseBlockSize;
  BlockManager::Options opts = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      kBaseBlockSize,
      window_size,
      /*max_seqs_per_batch=*/1);
  set_swa_capacity_for_token_budget(&opts, 2 * kCompressedUnitTokens);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t prompt_tokens = 2 * kCompressedUnitTokens;
  const std::vector<int32_t> prompt_a(prompt_tokens, 42);
  Sequence source = MakeTestSequence(/*index=*/0, prompt_a);
  ASSERT_TRUE(manager.allocate_sequence(&source, prompt_tokens));
  source.kv_state().incr_kv_cache_tokens_num(prompt_tokens);
  manager.deallocate_for_sequence(&source);
  source.reset();

  // Filling the SWA pool with a different active sequence evicts the oldest
  // cached SWA blocks from A, including A's window ending at 16384, while its
  // newest window ending at 32768 remains. The larger C4/C128 pools retain A's
  // complete compressed prefix.
  const std::vector<int32_t> prompt_b(prompt_tokens, 43);
  Sequence pressure = MakeTestSequence(/*index=*/1, prompt_b);
  ASSERT_TRUE(manager.allocate_sequence(&pressure, prompt_tokens));

  Sequence probe = MakeTestSequence(/*index=*/2, prompt_a);
  manager.allocate_shared_for_sequence(&probe);
  EXPECT_EQ(probe.kv_state().kv_cache_tokens_num(), 0u);

  manager.deallocate_for_sequence(&probe);
  manager.deallocate_for_sequence(&pressure);
}

// DSV4 D-side (instance_is_decode=true) should skip the SWA leaf's prefix
// cache entirely: no shared blocks on SWA even when the same prompt was
// previously seeded. C4 / C128 continue to hit because the role predicate
// only masks SWA / LINEAR / EMBEDDING on the DECODE side. This is the mirror of
// Dsv4PrefixCacheHitOnRepeatedPrefix under the DECODE role.
TEST(CompositeBlockManagerTest, DecodeRoleSkipsSwaPrefixCache) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 4 * kBaseBlockSize;
  const uint32_t max_seqs_per_batch = 4;
  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  set_swa_capacity_for_token_budget(&opts, 3 * kCompressedUnitTokens);
  // First seed the cache under a PREFILL role so all three leaves publish
  // their blocks -- then swap in a DECODE-role composite that shares the
  // hash space via the same leaf construction path.
  ASSERT_TRUE(opts.enable_prefix_cache());
  {
    CompositeBlockManager prefill_manager(build_composite_leaves(opts), opts);
    const size_t num_tokens = 2 * kCompressedUnitTokens;
    const std::vector<int32_t> prompt(num_tokens, 7);
    Sequence seq_seed = MakeTestSequence(0, prompt);
    ASSERT_TRUE(prefill_manager.allocate_sequence(&seq_seed, num_tokens));
    seq_seed.kv_state().incr_kv_cache_tokens_num(num_tokens);
    prefill_manager.deallocate_for_sequence(&seq_seed);
    seq_seed.reset();

    // Under the PREFILL role, SWA/C4/C128 all hit.
    Sequence seq_p_hit = MakeTestSequence(1, prompt);
    prefill_manager.allocate_shared_for_sequence(&seq_p_hit);
    EXPECT_GT(seq_p_hit.kv_state().shared_blocks_num(BlockType::SWA), 0u);
    EXPECT_GT(seq_p_hit.kv_state().shared_blocks_num(BlockType::C4), 0u);
    EXPECT_GT(seq_p_hit.kv_state().shared_blocks_num(BlockType::C128), 0u);
    prefill_manager.deallocate_for_sequence(&seq_p_hit);
  }

  // Under DECODE role: separate manager (its own SWA leaf ⇒ own cache), so a
  // fresh prompt lookup must return zero SWA / zero C4 / zero C128. The
  // point of this test is that CONSTRUCTION does not FATAL and the SWA leaf
  // is skipped at classify + probe time; the composite still classifies as
  // SWA_COMPRESSED (C4+C128 remain prefix-cache-on).
  BlockManager::Options decode_opts = opts;
  decode_opts.instance_is_decode(true);
  CompositeBlockManager decode_manager(build_composite_leaves(decode_opts),
                                       decode_opts);
  const size_t num_tokens = 2 * kCompressedUnitTokens;
  const std::vector<int32_t> prompt(num_tokens, 7);
  Sequence seq_d = MakeTestSequence(2, prompt);
  decode_manager.allocate_shared_for_sequence(&seq_d);
  // SWA prefix cache is off on DECODE ⇒ never hits.
  EXPECT_EQ(seq_d.kv_state().shared_blocks_num(BlockType::SWA), 0u);
  // C4/C128 caches are fresh (per-manager) so also zero here, but the intent
  // is: the participate predicate keeps them enabled so future hits work.
  EXPECT_EQ(seq_d.kv_state().shared_blocks_num(BlockType::C4), 0u);
  EXPECT_EQ(seq_d.kv_state().shared_blocks_num(BlockType::C128), 0u);
  // No safe_hit_tokens change either (nothing mounted).
  EXPECT_EQ(seq_d.kv_state().kv_cache_tokens_num(), 0u);

  decode_manager.deallocate_for_sequence(&seq_d);
}

TEST(CompositeBlockManagerTest, DecodeInitialSwaAllocationKeepsOnlyWindowTail) {
  const uint32_t window_size = 2 * kBaseBlockSize;
  BlockManager::Options opts = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      kBaseBlockSize,
      window_size,
      /*max_seqs_per_batch=*/1);
  opts.instance_is_decode(true).enable_prefix_cache(false).swa_num_blocks(
      /*two windows plus padding=*/5);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t logical_blocks = 10;
  const size_t num_tokens = logical_blocks * kBaseBlockSize;
  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(num_tokens, 7));

  ASSERT_TRUE(manager.allocate_sequence(&seq, num_tokens));
  const std::vector<Block> swa = SwaBlocks(seq);
  ASSERT_EQ(swa.size(), logical_blocks);
  for (size_t i = 0; i < logical_blocks - 2; ++i) {
    EXPECT_FALSE(swa[i].is_valid());
  }
  EXPECT_TRUE(swa[logical_blocks - 2].is_valid());
  EXPECT_TRUE(swa[logical_blocks - 1].is_valid());

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest,
     DecodeInitialSwaAllocationCoversNonAlignedWindow) {
  const uint32_t window_size = kBaseBlockSize;
  BlockManager::Options opts = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      kBaseBlockSize,
      window_size,
      /*max_seqs_per_batch=*/1);
  opts.instance_is_decode(true)
      .enable_prefix_cache(false)
      .num_speculative_tokens(5)
      .swa_num_blocks(/*two windows plus padding=*/5);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t num_tokens = 10000;
  const size_t logical_blocks =
      (num_tokens + kBaseBlockSize - 1) / kBaseBlockSize;
  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(num_tokens, 7));

  ASSERT_TRUE(manager.allocate_sequence(&seq, num_tokens));
  const std::vector<Block> swa = SwaBlocks(seq);
  ASSERT_EQ(swa.size(), logical_blocks);
  for (size_t i = 0; i < logical_blocks - 2; ++i) {
    EXPECT_FALSE(swa[i].is_valid());
  }
  EXPECT_TRUE(swa[logical_blocks - 2].is_valid());
  EXPECT_TRUE(swa[logical_blocks - 1].is_valid());

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest,
     DecodeInitialSwaAllocationKeepsBlockContainingWindowStart) {
  const uint32_t window_size = kBaseBlockSize;
  BlockManager::Options opts = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      kBaseBlockSize,
      window_size,
      /*max_seqs_per_batch=*/1);
  opts.instance_is_decode(true)
      .enable_prefix_cache(false)
      .num_speculative_tokens(0)
      .swa_num_blocks(/*two windows plus padding=*/5);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  const size_t num_tokens = 2 * kBaseBlockSize - 1;
  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(num_tokens, 7));

  ASSERT_TRUE(manager.allocate_sequence(&seq, num_tokens));
  const std::vector<Block> swa = SwaBlocks(seq);
  ASSERT_EQ(swa.size(), 2u);
  EXPECT_TRUE(swa[0].is_valid());
  EXPECT_TRUE(swa[1].is_valid());

  manager.deallocate_for_sequence(&seq);
}

TEST(CompositeBlockManagerTest,
     DecodeLinearWithoutPrefixCacheKeepsReceivedState) {
  // LinearStateBlockManager pulls its chunk stride from the global scheduler
  // config, so pin a valid stride for this test and restore it after.
  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 128;

  const uint32_t block_size = 128;
  const uint32_t num_blocks = 128;
  BlockManager::Options opts;
  opts.num_blocks(num_blocks)
      .block_size(block_size)
      .enable_linear_state(true)
      .linear_state_num_slots(64)
      .enable_prefix_cache(false)
      .instance_is_decode(true);
  // No manager_types → flat KV shape. LINEAR is added on top by
  // build_composite_leaves.
  CompositeBlockManager manager(build_composite_leaves(opts), opts);

  EXPECT_FALSE(manager.leaf_entries().at(BlockType::KV).supports_prefix_cache);
  EXPECT_FALSE(
      manager.leaf_entries().at(BlockType::LINEAR).supports_prefix_cache);

  const std::vector<int32_t> prompt(4 * block_size, 5);
  Sequence seq = MakeTestSequence(0, prompt);
  ASSERT_TRUE(manager.allocate_sequence(&seq, prompt.size()));
  const int32_t received_id = seq.get_linear_state_slot_id();
  ASSERT_TRUE(manager.allocate_sequence(&seq, prompt.size()));
  EXPECT_EQ(seq.get_linear_state_slot_id(), received_id);
  seq.kv_state().incr_kv_cache_tokens_num(prompt.size());

  EXPECT_EQ(seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
  seq.append_token(8);
  ASSERT_TRUE(manager.allocate_sequence(&seq, seq.num_tokens()));
  ASSERT_EQ(seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(seq.get_linear_state_slot_id(), received_id);
  seq.kv_state().incr_kv_cache_tokens_num(1);
  manager.deallocate_for_sequence(&seq);

  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      original_chunk_stride;
}

TEST(CompositeBlockManagerTest, DecodeRoleLinearDefaultStrideReusesSlot) {
  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  // Emulate --enable_chunked_prefill=false: stride stays at the -1 default.
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = -1;

  const uint32_t block_size = 128;
  const uint32_t num_blocks = 128;
  BlockManager::Options opts;
  opts.num_blocks(num_blocks)
      .block_size(block_size)
      .enable_linear_state(true)
      .linear_state_num_slots(64)
      .enable_prefix_cache(false)
      .instance_is_decode(true);
  CompositeBlockManager manager(build_composite_leaves(opts), opts);
  const std::vector<int32_t> prompt(4 * block_size, 5);
  Sequence seq = MakeTestSequence(0, prompt);
  ASSERT_TRUE(manager.allocate_sequence(&seq, prompt.size()));
  const int32_t received_id = seq.get_linear_state_slot_id();
  seq.kv_state().set_kv_cache_tokens_num(prompt.size());
  seq.append_token(8);
  ASSERT_TRUE(manager.allocate_sequence(&seq, seq.num_tokens()));
  EXPECT_EQ(seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(seq.get_linear_state_slot_id(), received_id);
  manager.deallocate_for_sequence(&seq);

  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      original_chunk_stride;
}

namespace {

class LinearStateWindowTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    original_stride_ =
        SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 4;
    stopping_checker_.set_max_generated_tokens(256);
  }

  void TearDown() override {
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
        original_stride_;
  }

  std::unique_ptr<CompositeBlockManager> make_manager(
      uint32_t slots = 16,
      uint32_t kv_blocks = 32,
      uint32_t speculative_tokens = 0,
      bool concurrent = false) {
    BlockManager::Options options;
    options.num_blocks(kv_blocks)
        .block_size(2)
        .enable_linear_state(true)
        .linear_state_num_slots(slots)
        .num_speculative_tokens(speculative_tokens)
        .enable_disagg_pd(concurrent)
        .enable_prefix_cache(GetParam());
    return std::make_unique<CompositeBlockManager>(
        build_composite_leaves(options), options);
  }

  Sequence make_sequence(size_t index,
                         const std::vector<int32_t>& prompt_token_ids,
                         bool enable_schedule_overlap = false) {
    SequenceParams sequence_params;
    sequence_params.seq_capacity = 8192;
    sequence_params.stopping_checker = &stopping_checker_;
    sequence_params.sampling_param = &sampling_param_;
    sequence_params.skip_special_tokens = true;
    sequence_params.echo = false;
    sequence_params.logprobs = false;
    sequence_params.enable_schedule_overlap = enable_schedule_overlap;
    IncrementalDecoder decoder("", 1, false, false);
    return Sequence(index,
                    prompt_token_ids,
                    torch::Tensor(),
                    MMData(),
                    std::move(decoder),
                    sequence_params);
  }

  void finish_input(Sequence& sequence, size_t cached_tokens) {
    sequence.kv_state().set_kv_cache_tokens_num(cached_tokens);
  }

 private:
  int32_t original_stride_ = 0;
  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
};

}  // namespace

TEST_P(LinearStateWindowTest, PrefillLeafUsesExplicitState) {
  for (const bool concurrent : {false, true}) {
    SCOPED_TRACE(concurrent);
    auto manager = make_manager(8, 32, 0, concurrent);
    Sequence sequence = make_sequence(0, std::vector<int32_t>(12, 7));
    ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
    finish_input(sequence, 4);
    const int32_t sequence_source_id = sequence.get_linear_state_slot_id();
    BlockManager* leaf =
        manager->leaf_entries().at(BlockType::LINEAR).leaf.get();
    KVCacheState kv_state;
    kv_state.add_blocks(BlockType::LINEAR, leaf->allocate(2));
    const int32_t retained_source_id =
        kv_state.blocks(BlockType::LINEAR).back().id();

    auto allocated = leaf->allocate_for_sequence(&sequence, kv_state, 8);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 2u);
    EXPECT_TRUE(kv_state.blocks(BlockType::LINEAR).empty());
    EXPECT_EQ(kv_state.num_cached_blocks(BlockType::LINEAR), 0u);
    EXPECT_NE(allocated->front().id(), retained_source_id);
    EXPECT_NE(allocated->back().id(), retained_source_id);
    EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::LINEAR), 0u);
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), sequence_source_id);
    allocated.reset();
    leaf->deallocate(kv_state.blocks(BlockType::LINEAR));
    kv_state.reset();
    manager->deallocate_for_sequence(&sequence);
    sequence.reset();
    EXPECT_EQ(leaf->num_used_blocks(), 0u);
  }
}

TEST_P(LinearStateWindowTest, DecodeReceiverUsesExplicitState) {
  for (const bool concurrent : {false, true}) {
    SCOPED_TRACE(concurrent);
    std::unique_ptr<BlockManager> manager =
        std::make_unique<LinearStateBlockManager>(4, 4, GetParam(), true);
    if (concurrent) {
      manager =
          std::make_unique<ConcurrentBlockManagerImpl>(std::move(manager));
    }
    Sequence sequence = make_sequence(0, std::vector<int32_t>(12, 7));
    sequence.kv_state().add_blocks(BlockType::LINEAR, manager->allocate(1));
    const int32_t sequence_source_id = sequence.get_linear_state_slot_id();
    KVCacheState kv_state;

    auto allocated = manager->allocate_for_sequence(&sequence, kv_state, 12);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    kv_state.add_blocks(BlockType::LINEAR, *allocated);
    allocated->clear();
    kv_state.add_blocks(BlockType::LINEAR, manager->allocate(1));
    const int32_t retained_source_id =
        kv_state.blocks(BlockType::LINEAR).back().id();

    allocated = manager->allocate_for_sequence(&sequence, kv_state, 13);
    ASSERT_TRUE(allocated.has_value());
    EXPECT_TRUE(allocated->empty());
    ASSERT_EQ(kv_state.num_blocks(BlockType::LINEAR), 1u);
    EXPECT_EQ(kv_state.blocks(BlockType::LINEAR).front().id(),
              retained_source_id);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), sequence_source_id);
    EXPECT_EQ(manager->num_used_blocks(), 2u);
    manager->deallocate(kv_state.blocks(BlockType::LINEAR));
    kv_state.reset();
    manager->deallocate(sequence.kv_state().blocks(BlockType::LINEAR));
    sequence.reset();
    EXPECT_EQ(manager->num_used_blocks(), 0u);
    EXPECT_EQ(manager->num_free_blocks(), manager->num_total_blocks());
  }
}

TEST_P(LinearStateWindowTest,
       HostDecodeCheckpointDoesNotAdvanceSequenceHashes) {
  LinearStateBlockManager host_manager(
      /*num_slots=*/4,
      /*chunk_stride=*/4,
      GetParam(),
      /*instance_is_decode=*/true,
      /*num_speculative_tokens=*/0,
      /*preserve_decode_checkpoint=*/true);
  Sequence sequence = make_sequence(0, {7, 7, 7});
  sequence.kv_state().set_kv_cache_tokens_num(3);
  sequence.append_token(8);
  sequence.set_last_confirmed_cached_tokens_num(4);
  sequence.host_kv_state().add_blocks(BlockType::LINEAR,
                                      host_manager.allocate(2));
  const int32_t checkpoint_id =
      sequence.host_kv_state().blocks(BlockType::LINEAR).front().id();
  const int32_t live_id =
      sequence.host_kv_state().blocks(BlockType::LINEAR).back().id();

  std::optional<std::vector<Block>> allocated =
      host_manager.allocate_for_sequence(
          &sequence, sequence.host_kv_state(), sequence.num_tokens());

  ASSERT_TRUE(allocated.has_value());
  EXPECT_TRUE(allocated->empty());
  EXPECT_TRUE(sequence.linear_state_hashes().empty());
  ASSERT_EQ(sequence.host_kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.host_kv_state().blocks(BlockType::LINEAR).front().id(),
            checkpoint_id);
  EXPECT_EQ(sequence.host_kv_state().blocks(BlockType::LINEAR).back().id(),
            live_id);

  host_manager.deallocate(sequence.host_kv_state().blocks(BlockType::LINEAR));
  sequence.host_kv_state().reset();
  EXPECT_EQ(host_manager.num_used_blocks(), 0u);
}

TEST_P(LinearStateWindowTest, DecodeReceiverAllocationIsOwnedByLeaf) {
  for (const size_t prompt_length : {12u, 13u}) {
    LinearStateBlockManager manager(3, 4, GetParam(), true);
    Sequence sequence =
        make_sequence(0, std::vector<int32_t>(prompt_length, 7));
    auto allocated = manager.allocate_for_sequence(
        &sequence, sequence.kv_state(), prompt_length);
    ASSERT_TRUE(allocated.has_value());
    ASSERT_EQ(allocated->size(), 1u);
    EXPECT_TRUE(allocated->back().is_valid());
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 0u);
    allocated.reset();
    EXPECT_EQ(manager.num_used_blocks(), 0u);

    allocated = manager.allocate_for_sequence(
        &sequence, sequence.kv_state(), prompt_length);
    ASSERT_TRUE(allocated.has_value());
    sequence.kv_state().add_blocks(BlockType::LINEAR, *allocated);
    allocated->clear();
    const int32_t received_id = sequence.get_linear_state_slot_id();

    finish_input(sequence, prompt_length);
    sequence.append_token(8);
    sequence.set_last_confirmed_cached_tokens_num(prompt_length);
    allocated = manager.allocate_for_sequence(
        &sequence, sequence.kv_state(), prompt_length + 1);
    ASSERT_TRUE(allocated.has_value());
    EXPECT_TRUE(allocated->empty());
    EXPECT_EQ(sequence.last_confirmed_cached_tokens_num(), prompt_length);
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
    EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
              received_id);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), received_id);
    EXPECT_EQ(manager.num_used_blocks(), 1u);
    manager.deallocate(sequence.kv_state().blocks(BlockType::LINEAR));
  }
}

TEST_P(LinearStateWindowTest, DecodeReceiverNeedsOnlyOneAvailableSlot) {
  BlockManager::Options options;
  options.num_blocks(32)
      .block_size(2)
      .enable_linear_state(true)
      .linear_state_num_slots(3)
      .enable_prefix_cache(false)
      .instance_is_decode(true);
  CompositeBlockManager manager(build_composite_leaves(options), options);
  BlockManager* linear_leaf =
      manager.leaf_entries().at(BlockType::LINEAR).leaf.get();
  auto held_slots = linear_leaf->allocate(2);
  ASSERT_EQ(held_slots.size(), 2u);
  Sequence sequence = make_sequence(0, std::vector<int32_t>(12, 7));
  EXPECT_FALSE(manager.allocate_sequence(&sequence, 12));
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::KV), 0u);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 0u);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 0u);

  held_slots.pop_back();
  ASSERT_TRUE(manager.allocate_sequence(&sequence, 12));
  const int32_t received_id = sequence.get_linear_state_slot_id();
  EXPECT_EQ(linear_leaf->num_free_blocks(), 0u);
  finish_input(sequence, 12);
  sequence.append_token(8);
  ASSERT_TRUE(manager.allocate_sequence(&sequence, 13));
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(sequence.get_linear_state_slot_id(), received_id);
  EXPECT_EQ(sequence.kv_cache_tokens_num(), 12u);
  manager.deallocate_for_sequence(&sequence);
  EXPECT_EQ(linear_leaf->num_used_blocks(), 1u);
}

TEST_P(LinearStateWindowTest,
       PrefillUsesTwoBlocksAndPublishesLogicalCheckpoints) {
  auto manager = make_manager();
  Sequence sequence = make_sequence(0, std::vector<int32_t>(13, 7));
  int32_t previous_id = -1;
  for (size_t output_index = 0; output_index < 4; ++output_index) {
    const size_t target = std::min((output_index + 1) * 4, size_t{13});
    ASSERT_TRUE(manager->allocate_sequence(&sequence, target));
    const Slice<Block> blocks = sequence.kv_state().blocks(BlockType::LINEAR);
    ASSERT_EQ(blocks.size(), output_index == 0 ? 1u : 2u);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), blocks.back().id());
    EXPECT_NE(blocks.back().id(), previous_id);
    if (output_index > 0) {
      EXPECT_EQ(blocks.front().id(), previous_id);
      if (GetParam()) {
        EXPECT_EQ(XXH3Key(blocks.front().get_immutable_hash_value()),
                  sequence.linear_state_hashes()[output_index - 1]);
      }
    }
    const int32_t output_id = blocks.back().id();
    EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::LINEAR),
              GetParam() ? output_index : 0u);
    finish_input(sequence, target);
    manager->cache_for_sequence(&sequence);
    EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::LINEAR),
              GetParam() ? output_index : 0u);
    previous_id = output_id;
  }
  sequence.append_token(8);
  ASSERT_TRUE(manager->allocate_sequence(&sequence, sequence.num_tokens()));
  const Slice<Block> blocks = sequence.kv_state().blocks(BlockType::LINEAR);
  ASSERT_EQ(blocks.size(), 1u);
  EXPECT_EQ(blocks.back().id(), previous_id);
  for (size_t index = 0; index + 1 < blocks.size(); ++index) {
    EXPECT_FALSE(blocks[index].is_valid());
  }
  manager->deallocate_for_sequence(&sequence);
  EXPECT_EQ(
      manager->leaf_entries().at(BlockType::LINEAR).leaf->num_used_blocks(),
      0u);
}

TEST_P(LinearStateWindowTest, ExhaustionPreservesSourceAndRollsBackKvGrowth) {
  auto manager = make_manager(3);
  Sequence sequence = make_sequence(0, std::vector<int32_t>(13, 7));
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
  finish_input(sequence, 4);
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 8));
  finish_input(sequence, 8);
  const int32_t source_id = sequence.get_linear_state_slot_id();
  Block retained_old_read =
      sequence.kv_state().blocks(BlockType::LINEAR).front();
  const size_t kv_used =
      manager->leaf_entries().at(BlockType::KV).leaf->num_used_blocks();
  for (size_t retry = 0; retry < 2; ++retry) {
    EXPECT_FALSE(manager->allocate_sequence(&sequence, 12));
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), source_id);
    EXPECT_EQ(sequence.kv_cache_tokens_num(), 8u);
    EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::LINEAR),
              GetParam() ? 2u : 0u);
    EXPECT_EQ(manager->leaf_entries().at(BlockType::KV).leaf->num_used_blocks(),
              kv_used);
  }
  retained_old_read = Block();
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 12));
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
            source_id);
  manager->deallocate_for_sequence(&sequence);
  EXPECT_EQ(
      manager->leaf_entries().at(BlockType::LINEAR).leaf->num_used_blocks(),
      0u);
}

TEST_P(LinearStateWindowTest, DecodeRollingExhaustionFallsBackToCurrentSource) {
  auto manager = make_manager(3, 32);
  Sequence sequence = make_sequence(0, {7, 7, 7});
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 3));
  finish_input(sequence, 3);
  sequence.append_token(8);
  sequence.set_last_confirmed_cached_tokens_num(3);
  const int32_t source_id = sequence.get_linear_state_slot_id();
  BlockManager* linear_leaf =
      manager->leaf_entries().at(BlockType::LINEAR).leaf.get();
  std::vector<Block> held = linear_leaf->allocate(1);
  ASSERT_EQ(held.size(), 1u);

  for (size_t retry = 0; retry < 2; ++retry) {
    EXPECT_TRUE(manager->allocate_sequence(&sequence, 4));
    ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), source_id);
  }

  held.clear();
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
            source_id);
  manager->deallocate_for_sequence(&sequence);
}

TEST_P(LinearStateWindowTest, KvFailureDoesNotChangeLinearWindow) {
  auto manager = make_manager(16, 3);
  Sequence sequence = make_sequence(0, std::vector<int32_t>(9, 7));
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
  finish_input(sequence, 4);
  const int32_t source_id = sequence.get_linear_state_slot_id();
  EXPECT_FALSE(manager->allocate_sequence(&sequence, 8));
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(sequence.get_linear_state_slot_id(), source_id);
  EXPECT_EQ(
      manager->leaf_entries().at(BlockType::LINEAR).leaf->num_used_blocks(),
      1u);
  manager->deallocate_for_sequence(&sequence);
}

TEST_P(LinearStateWindowTest, IncompleteKvGrowthRollsBackLinearAllocation) {
  class EmptyGrowthBlockManager final : public BlockManagerImpl {
   public:
    explicit EmptyGrowthBlockManager(const BlockManager::Options& options)
        : BlockManagerImpl(options) {}

    std::optional<std::vector<Block>> allocate_for_sequence(Sequence*,
                                                            KVCacheState&,
                                                            size_t) override {
      return std::vector<Block>{};
    }
  };

  BlockManager::Options options;
  options.num_blocks(32)
      .block_size(2)
      .enable_linear_state(true)
      .linear_state_num_slots(3)
      .enable_prefix_cache(GetParam());
  auto leaves = build_composite_leaves(options);
  leaves.at(BlockType::KV).leaf =
      std::make_unique<EmptyGrowthBlockManager>(options);
  CompositeBlockManager manager(std::move(leaves), options);
  Sequence sequence = make_sequence(0, std::vector<int32_t>(3, 7));
  sequence.add_blocks(
      BlockType::KV,
      manager.leaf_entries().at(BlockType::KV).leaf->allocate(2));
  ASSERT_TRUE(manager.allocate_sequence(&sequence, 3));
  finish_input(sequence, 3);
  sequence.append_token(8);
  sequence.set_last_confirmed_cached_tokens_num(3);
  ASSERT_TRUE(manager.allocate_sequence(&sequence, 4));
  const int32_t source_id = sequence.get_linear_state_slot_id();

  EXPECT_FALSE(manager.allocate_sequence(&sequence, 5));
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(sequence.get_linear_state_slot_id(), source_id);
  EXPECT_EQ(
      manager.leaf_entries().at(BlockType::LINEAR).leaf->num_used_blocks(), 1u);
  sequence.add_blocks(
      BlockType::KV,
      manager.leaf_entries().at(BlockType::KV).leaf->allocate(1));
  ASSERT_TRUE(manager.allocate_sequence(&sequence, 5));
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
            source_id);
  manager.deallocate_for_sequence(&sequence);
}

TEST_P(LinearStateWindowTest, PrefillCapacityIsIndependentOfCheckpointStride) {
  auto manager = make_manager();
  Sequence sequence = make_sequence(0, std::vector<int32_t>(13, 7));
  for (const size_t target : {3u, 5u, 8u, 13u}) {
    ASSERT_TRUE(manager->allocate_sequence(&sequence, target));
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR),
              target == 3 ? 1u : 2u);
    EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::LINEAR),
              GetParam() && target == 13 ? 2u : 0u);
    finish_input(sequence, target);
  }
  manager->deallocate_for_sequence(&sequence);

  Sequence full_prefill = make_sequence(1, std::vector<int32_t>(13, 8));
  ASSERT_TRUE(manager->allocate_sequence(&full_prefill, 16));
  EXPECT_EQ(full_prefill.kv_state().num_blocks(BlockType::LINEAR), 1u);
  manager->deallocate_for_sequence(&full_prefill);
}

TEST_P(LinearStateWindowTest, PrefillReusesTwoSlotsAcrossArbitraryChunks) {
  auto manager = make_manager(3);
  Sequence sequence = make_sequence(0, std::vector<int32_t>(13, 7));
  int32_t previous_write_id = -1;
  for (const size_t target : {3u, 5u, 8u, 12u, 13u}) {
    ASSERT_TRUE(manager->allocate_sequence(&sequence, target));
    const Slice<Block> blocks = sequence.kv_state().blocks(BlockType::LINEAR);
    ASSERT_EQ(blocks.size(), target == 3 ? 1u : 2u);
    if (previous_write_id >= 0) {
      EXPECT_NE(blocks.front().id(), blocks.back().id());
      EXPECT_EQ(blocks.front().id(), previous_write_id);
    }
    previous_write_id = blocks.back().id();
    finish_input(sequence, target);
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR),
              target == 3 ? 1u : 2u);
  }
  manager->deallocate_for_sequence(&sequence);
}

TEST_P(LinearStateWindowTest, ColdPrefillNeedsOnlyOneSlot) {
  auto manager = make_manager(2);
  Sequence sequence = make_sequence(0, std::vector<int32_t>(9, 7));
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
  const int32_t source_id = sequence.get_linear_state_slot_id();
  finish_input(sequence, 4);
  EXPECT_FALSE(manager->allocate_sequence(&sequence, 8));
  EXPECT_EQ(sequence.get_linear_state_slot_id(), source_id);
  EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(sequence.kv_state().num_cached_blocks(BlockType::LINEAR),
            GetParam() ? 1u : 0u);
  Sequence consumer = make_sequence(1, std::vector<int32_t>(9, 7));
  manager->allocate_shared_for_sequence(&consumer);
  EXPECT_EQ(consumer.kv_cache_tokens_num(), GetParam() ? 4u : 0u);
  if (GetParam()) {
    EXPECT_EQ(consumer.get_linear_state_slot_id(), source_id);
  }
  manager->deallocate_for_sequence(&consumer);
  manager->deallocate_for_sequence(&sequence);
}

TEST_P(LinearStateWindowTest, NonOverlapDecodePredictsAndRetiresCheckpoint) {
  for (const bool concurrent : {false, true}) {
    SCOPED_TRACE(concurrent);
    auto manager = make_manager(3, 256, 0, concurrent);
    Sequence sequence = make_sequence(0, {7, 7, 7});
    ASSERT_TRUE(manager->allocate_sequence(&sequence, 3));
    const int32_t initial_id = sequence.get_linear_state_slot_id();
    finish_input(sequence, 3);
    sequence.append_token(8);
    sequence.set_last_confirmed_cached_tokens_num(3);
    EXPECT_EQ(sequence.last_confirmed_cached_tokens_num(), 3u);

    ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
    ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
    EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
              initial_id);
    const int32_t rolled_id = sequence.get_linear_state_slot_id();
    EXPECT_NE(rolled_id, initial_id);

    sequence.append_token(9);
    sequence.set_last_confirmed_cached_tokens_num(4);
    EXPECT_EQ(sequence.last_confirmed_cached_tokens_num(), 4u);
    ASSERT_TRUE(manager->allocate_sequence(&sequence, 5));
    ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
    EXPECT_EQ(sequence.get_linear_state_slot_id(), rolled_id);

    BlockManager* leaf =
        manager->leaf_entries().at(BlockType::LINEAR).leaf.get();
    EXPECT_EQ(leaf->num_blocks_in_prefix_cache(), 0u);
    manager->deallocate_for_sequence(&sequence);
    EXPECT_EQ(leaf->num_used_blocks(), 0u);
  }
}

TEST_P(LinearStateWindowTest, OverlapDecodePredictsAndResolvesCheckpoint) {
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 8;
  for (const bool checkpoint_hit : {false, true}) {
    SCOPED_TRACE(checkpoint_hit);
    auto manager = make_manager(4, 256, 3);
    Sequence sequence = make_sequence(0, std::vector<int32_t>(6, 7), true);
    ASSERT_TRUE(manager->allocate_sequence(&sequence, 6));
    finish_input(sequence, 6);
    sequence.set_last_confirmed_cached_tokens_num(6);

    ASSERT_TRUE(manager->allocate_sequence(&sequence, 10));
    EXPECT_EQ(sequence.last_confirmed_cached_tokens_num(), 6u);
    ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
    const int32_t initial_id = sequence.get_linear_state_slot_id();
    const int32_t historical_id =
        sequence.kv_state().blocks(BlockType::LINEAR).front().id();
    EXPECT_NE(initial_id, historical_id);

    sequence.set_last_confirmed_cached_tokens_num(checkpoint_hit ? 8 : 7);
    sequence.kv_state().set_kv_cache_tokens_num(checkpoint_hit ? 9 : 8);
    ASSERT_TRUE(manager->allocate_sequence(&sequence, 12));
    if (checkpoint_hit) {
      ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 1u);
      EXPECT_EQ(sequence.get_linear_state_slot_id(), initial_id);
    } else {
      ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
      EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
                initial_id);
      EXPECT_EQ(sequence.get_linear_state_slot_id(), historical_id);
      EXPECT_EQ(
          manager->leaf_entries().at(BlockType::LINEAR).leaf->num_used_blocks(),
          2u);
    }
    manager->deallocate_for_sequence(&sequence);
  }
}

TEST_P(LinearStateWindowTest, NullSequenceDoesNotAllocate) {
  LinearStateBlockManager manager(2, 2, GetParam());
  KVCacheState kv_state;
  EXPECT_FALSE(manager.allocate_for_sequence(nullptr, kv_state, 8).has_value());
  EXPECT_EQ(manager.num_used_blocks(), 0u);
}

TEST_P(LinearStateWindowTest, AllowsSpeculativeSpanAboveHalfStride) {
  LinearStateBlockManager manager(/*num_slots=*/3,
                                  /*chunk_stride=*/4,
                                  /*enable_prefix_cache=*/GetParam(),
                                  /*instance_is_decode=*/false,
                                  /*num_speculative_tokens=*/2);
  EXPECT_EQ(manager.options().num_speculative_tokens(), 2u);
}

TEST_P(LinearStateWindowTest, FlatPrefixMountRestoresCachedTokens) {
  BlockManager::Options options;
  options.num_blocks(32).block_size(2).enable_prefix_cache(GetParam());
  CompositeBlockManager manager(build_composite_leaves(options), options);
  Sequence producer = make_sequence(0, std::vector<int32_t>(8, 7));
  ASSERT_TRUE(manager.allocate_sequence(&producer, 8));
  finish_input(producer, 8);
  manager.deallocate_for_sequence(&producer);
  Sequence consumer = make_sequence(1, std::vector<int32_t>(8, 7));
  manager.allocate_shared_for_sequence(&consumer);
  EXPECT_EQ(consumer.kv_cache_tokens_num(), GetParam() ? 6u : 0u);
  manager.deallocate_for_sequence(&consumer);
}

TEST_P(LinearStateWindowTest, PrefixMountIsSparseAndMatchedOnlyOnce) {
  auto manager = make_manager();
  Sequence producer = make_sequence(0, std::vector<int32_t>(13, 7));
  for (const size_t target : {4, 8, 12, 13}) {
    ASSERT_TRUE(manager->allocate_sequence(&producer, target));
    finish_input(producer, target);
  }
  manager->deallocate_for_sequence(&producer);
  Sequence consumer = make_sequence(1, std::vector<int32_t>(13, 7));
  manager->allocate_shared_for_sequence(&consumer);
  EXPECT_EQ(consumer.kv_cache_tokens_num(), GetParam() ? 12u : 0u);
  EXPECT_EQ(consumer.last_confirmed_cached_tokens_num(), 0u);
  EXPECT_EQ(consumer.kv_state().num_blocks(BlockType::LINEAR),
            GetParam() ? 3u : 0u);
  const int32_t source_id = consumer.get_linear_state_slot_id();
  if (GetParam()) {
    const Slice<Block> blocks = consumer.kv_state().blocks(BlockType::LINEAR);
    EXPECT_FALSE(blocks[0].is_valid());
    EXPECT_FALSE(blocks[1].is_valid());
  }
  manager->allocate_shared_for_sequence(&consumer);
  EXPECT_EQ(consumer.get_linear_state_slot_id(), source_id);
  ASSERT_TRUE(manager->allocate_sequence(&consumer, GetParam() ? 13 : 4));
  manager->allocate_shared_for_sequence(&consumer);
  EXPECT_EQ(consumer.kv_state().num_blocks(BlockType::LINEAR),
            GetParam() ? 2u : 1u);
  if (GetParam()) {
    EXPECT_EQ(consumer.kv_state().blocks(BlockType::LINEAR).front().id(),
              source_id);
  }
  finish_input(consumer, GetParam() ? 13 : 4);
  EXPECT_EQ(consumer.kv_state().num_blocks(BlockType::LINEAR),
            GetParam() ? 2u : 1u);
  manager->deallocate_for_sequence(&consumer);
  consumer.reset();
  manager->allocate_shared_for_sequence(&consumer);
  EXPECT_EQ(consumer.kv_cache_tokens_num(), GetParam() ? 12u : 0u);
  EXPECT_EQ(consumer.last_confirmed_cached_tokens_num(), 0u);
  manager->deallocate_for_sequence(&consumer);
}

TEST_P(LinearStateWindowTest, DuplicateHashesStillAllocateDistinctOutputs) {
  auto manager = make_manager();
  Sequence first = make_sequence(0, std::vector<int32_t>(9, 7));
  Sequence second = make_sequence(1, std::vector<int32_t>(9, 7));
  manager->allocate_shared_for_sequence(&first);
  manager->allocate_shared_for_sequence(&second);
  ASSERT_TRUE(manager->allocate_sequence(&first, 4));
  ASSERT_TRUE(manager->allocate_sequence(&second, 4));
  const int32_t second_source_id = second.get_linear_state_slot_id();
  finish_input(first, 4);
  finish_input(second, 4);
  ASSERT_TRUE(manager->allocate_sequence(&first, 8));
  ASSERT_TRUE(manager->allocate_sequence(&second, 8));
  EXPECT_NE(second.get_linear_state_slot_id(), second_source_id);
  EXPECT_EQ(second.kv_state().blocks(BlockType::LINEAR)[0].id(),
            second_source_id);
  EXPECT_EQ(manager->leaf_entries()
                .at(BlockType::LINEAR)
                .leaf->num_blocks_in_prefix_cache(),
            GetParam() ? 1u : 0u);
  manager->deallocate_for_sequence(&first);
  manager->deallocate_for_sequence(&second);
}

TEST_P(LinearStateWindowTest, PrefixMissStaysMatchedUntilReset) {
  auto manager = make_manager();
  Sequence waiting = make_sequence(0, std::vector<int32_t>(9, 7));
  manager->allocate_shared_for_sequence(&waiting);
  EXPECT_EQ(waiting.kv_cache_tokens_num(), 0u);
  Sequence producer = make_sequence(1, std::vector<int32_t>(9, 7));
  for (const size_t target : {4, 8, 9}) {
    ASSERT_TRUE(manager->allocate_sequence(&producer, target));
    finish_input(producer, target);
  }
  manager->deallocate_for_sequence(&producer);
  manager->allocate_shared_for_sequence(&waiting);
  EXPECT_EQ(waiting.kv_cache_tokens_num(), 0u);
  ASSERT_TRUE(manager->allocate_sequence(&waiting, 4));
  EXPECT_EQ(waiting.kv_state().num_blocks(BlockType::LINEAR), 1u);
  manager->deallocate_for_sequence(&waiting);
  waiting.reset();
  manager->allocate_shared_for_sequence(&waiting);
  EXPECT_EQ(waiting.kv_cache_tokens_num(), GetParam() ? 8u : 0u);
  manager->deallocate_for_sequence(&waiting);
}

TEST_P(LinearStateWindowTest,
       OverlapPrefillKeepsTwoBlocksWithoutCompletionHooks) {
  auto manager = make_manager();
  Sequence sequence = make_sequence(0, std::vector<int32_t>(13, 7));
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 4));
  finish_input(sequence, 4);
  const int32_t first_id = sequence.get_linear_state_slot_id();
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 8));
  const int32_t second_id = sequence.get_linear_state_slot_id();
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
            first_id);
  Batch second_batch(&sequence);
  sequence.kv_state().set_kv_cache_tokens_num(8);
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.get_linear_state_slot_id(), second_id);
  ASSERT_TRUE(manager->allocate_sequence(&sequence, 12));
  const int32_t third_id = sequence.get_linear_state_slot_id();
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR).front().id(),
            second_id);
  EXPECT_NE(second_id, third_id);
  Batch third_batch(&sequence);
  sequence.kv_state().set_kv_cache_tokens_num(12);
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.get_linear_state_slot_id(), third_id);
  if (!GetParam()) {
    for (const Block& block : sequence.kv_state().blocks(BlockType::LINEAR)) {
      EXPECT_EQ(block.ref_count(), 1u);
    }
  }
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.get_linear_state_slot_id(), third_id);
  manager->deallocate_for_sequence(&sequence);
}

TEST_P(LinearStateWindowTest, PoolReleasesCancelledSequenceBeforeResults) {
  for (const bool discard : {false, true}) {
    BlockManagerPool::Options options;
    options.num_blocks(32)
        .block_size(2)
        .enable_linear_state(true)
        .linear_state_num_slots(8)
        .enable_prefix_cache(GetParam());
    BlockManagerPool pool(options, 1);
    Sequence sequence = make_sequence(0, std::vector<int32_t>(13, 7));
    ASSERT_TRUE(pool.allocate(&sequence, 4));
    std::vector<Batch> first_batch(1);
    first_batch.front().add(&sequence, 4);
    sequence.kv_state().set_kv_cache_tokens_num(4);
    ASSERT_TRUE(pool.allocate(&sequence, 8));
    std::vector<Batch> second_batch(1);
    second_batch.front().add(&sequence, 4);
    sequence.kv_state().set_kv_cache_tokens_num(8);
    sequence.set_cancel();
    if (discard) {
      pool.deallocate_without_cache(&sequence);
    } else {
      pool.deallocate(&sequence);
    }
    EXPECT_FALSE(sequence.has_any_blocks());
    EXPECT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 0u);
    EXPECT_EQ(sequence.kv_cache_tokens_num(), 0u);
    Sequence replacement = make_sequence(1, {9, 9, 9, 9});
    EXPECT_TRUE(pool.allocate(&replacement, 4));
    pool.deallocate_without_cache(&replacement);
  }
}

INSTANTIATE_TEST_SUITE_P(PrefixCacheModes,
                         LinearStateWindowTest,
                         ::testing::Bool());

// V4.1 default configuration: enable_prefix_cache defaults to true while the
// C2/C1 TOKEN leaves never participate in prefix cache, so the builder must
// demote the composite to prefix-cache-off instead of aborting in
// classify_leaf_combination ("SWA prefix cache is on but neither C4 nor C128
// is present" FATAL). Regression test for the default-config startup crash.
TEST(CompositeBlockManagerTest, V41DefaultPrefixCacheDemotesInsteadOfAbort) {
  const uint32_t window_size = 4 * kBaseBlockSize;
  BlockManager::Options opts = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      /*block_size=*/kBaseBlockSize,
      /*window_size=*/window_size,
      /*max_seqs_per_batch=*/4);
  // V4.1 token shape: [SWA, TOKEN(2), TOKEN(1)]. The demotion falls out of the
  // per-group leaf_participates_in_prefix_cache property (C1/C2 never
  // participate), not of the model type.
  opts.model_type("deepseek_v41");
  opts.manager_types({kManagerTypeSlidingWindowBlockManager,
                      kManagerTypeBlockManagerImpl,
                      kManagerTypeBlockManagerImpl});
  opts.compress_ratios({0, 2, 1});
  ASSERT_TRUE(opts.enable_prefix_cache());

  // Cover both PREFILL/MIX and DECODE explicitly; keep the cache flag default.
  for (const bool is_decode : {false, true}) {
    SCOPED_TRACE(is_decode);
    opts.instance_is_decode(is_decode);
    CompositeBlockManager manager(build_composite_leaves(opts), opts);
    for (const BlockType type :
         {BlockType::SWA, BlockType::C2, BlockType::C1}) {
      const BlockManager* leaf =
          BlockManagerPoolTestPeer::leaf_of(manager, type);
      ASSERT_NE(leaf, nullptr);
      EXPECT_FALSE(leaf->options().enable_prefix_cache());
    }
    EXPECT_EQ(manager.leaf_combination(),
              CompositeBlockManager::LeafCombination::UNSUPPORTED);
  }
}

TEST(CompositeBlockManagerTest, V4DefaultPrefixCacheKeepsRolePolicy) {
  BlockManager::Options opts = MakeCompositeOptions(
      /*base_num_blocks=*/4096,
      /*block_size=*/kBaseBlockSize,
      /*window_size=*/4 * kBaseBlockSize,
      /*max_seqs_per_batch=*/4);
  ASSERT_TRUE(opts.enable_prefix_cache());
  for (const bool is_decode : {false, true}) {
    SCOPED_TRACE(is_decode);
    opts.instance_is_decode(is_decode);
    CompositeBlockManager manager(build_composite_leaves(opts), opts);
    const BlockManager* swa =
        BlockManagerPoolTestPeer::leaf_of(manager, BlockType::SWA);
    ASSERT_NE(swa, nullptr);
    EXPECT_EQ(swa->options().enable_prefix_cache(), !is_decode);
    for (const BlockType type : {BlockType::C4, BlockType::C128}) {
      const BlockManager* leaf =
          BlockManagerPoolTestPeer::leaf_of(manager, type);
      ASSERT_NE(leaf, nullptr);
      EXPECT_TRUE(leaf->options().enable_prefix_cache());
    }
  }
}

}  // namespace xllm
