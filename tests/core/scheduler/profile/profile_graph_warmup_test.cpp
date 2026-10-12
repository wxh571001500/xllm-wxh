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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/block/block_manager_pool.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/mtp_utils.h"
#include "core/framework/request/incremental_decoder.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "platform/platform.h"
#include "runtime/decode_graph_bucket.h"
#include "scheduler/profile/decode_graph_warmup_plan.h"
#include "scheduler/profile/graph_warmup.h"
#include "scheduler/profile/profile_manager.h"

namespace xllm {
namespace {

TEST(StepTimeProfilePlanTest, NeverExceedsConfiguredSequenceCapacity) {
  const std::vector<int32_t> batch_sizes =
      build_step_time_profile_batch_sizes(/*max_seqs_per_batch=*/16);
  ASSERT_FALSE(batch_sizes.empty());
  EXPECT_EQ(batch_sizes.back(), 16);
  EXPECT_TRUE(std::all_of(batch_sizes.begin(),
                          batch_sizes.end(),
                          [](int32_t x) { return x >= 1 && x <= 16; }));
}

TEST(StepTimeProfilePlanTest, BatchSizesCoverBoundaries) {
  // Minimal input yields the single smallest bucket.
  EXPECT_EQ(build_step_time_profile_batch_sizes(/*max_seqs_per_batch=*/1),
            (std::vector<int32_t>{1}));
  // Even tail buckets are padded up to the configured max.
  EXPECT_EQ(build_step_time_profile_batch_sizes(/*max_seqs_per_batch=*/2),
            (std::vector<int32_t>{1, 2}));
  // The legacy upper bound keeps odd buckets only.
  EXPECT_EQ(build_step_time_profile_batch_sizes(/*max_seqs_per_batch=*/23),
            (std::vector<int32_t>{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23}));
  // Inputs beyond the legacy bound are capped by it.
  EXPECT_EQ(build_step_time_profile_batch_sizes(/*max_seqs_per_batch=*/100),
            build_step_time_profile_batch_sizes(/*max_seqs_per_batch=*/23));
}

TEST(StepTimeProfilePlanTest, WarmupDecodeSeqLenFollowsFiaAndContextBounds) {
  // Without FIA the legacy 16-sequence bound applies, capped by the context.
  const bool without_fia = false;
  const bool with_fia = true;
  EXPECT_EQ(warmup_decode_seq_len(without_fia, /*max_context_len=*/8), 8);
  EXPECT_EQ(warmup_decode_seq_len(without_fia, /*max_context_len=*/16), 16);
  EXPECT_EQ(warmup_decode_seq_len(without_fia, /*max_context_len=*/4096), 16);
  // With FIA the bound rises to 2048, still capped by the context length.
  EXPECT_EQ(warmup_decode_seq_len(with_fia, /*max_context_len=*/16), 16);
  EXPECT_EQ(warmup_decode_seq_len(with_fia, /*max_context_len=*/2048), 2048);
  EXPECT_EQ(warmup_decode_seq_len(with_fia, /*max_context_len=*/4096), 2048);
}

Sequence make_sequence(size_t index, const std::vector<int32_t>& tokens) {
  RequestSamplingParam sampling_param;
  sampling_param.beam_width = 0;
  sampling_param.is_embeddings = false;

  StoppingChecker stopping_checker;

  SequenceParams params;
  params.seq_capacity = tokens.size() + 8;
  params.echo = false;
  params.skip_special_tokens = true;
  params.streaming = false;
  params.enable_schedule_overlap = false;
  params.rec_type = RecType::kNone;
  params.bos_token_id = 0;
  params.request_id = "profile_graph_warmup_test";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;

  IncrementalDecoder decoder(
      /*prompt=*/"prompt",
      /*num_prompt_tokens=*/tokens.size(),
      /*echo=*/params.echo,
      /*skip_special_tokens=*/params.skip_special_tokens);

  return Sequence(index,
                  tokens,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  params);
}

runtime::DecodeGraphWarmupConfig make_decode_graph_warmup_config(
    int64_t num_decoding_tokens,
    int32_t num_speculative_tokens,
    bool enable_no_padding,
    int32_t max_graph_batch_size = 0) {
  runtime::DecodeGraphWarmupConfig warmup_config;
  warmup_config.num_decoding_tokens = num_decoding_tokens;
  warmup_config.num_speculative_tokens = num_speculative_tokens;
  warmup_config.enable_graph_mode_decode_no_padding = enable_no_padding;
  warmup_config.max_graph_batch_size = max_graph_batch_size;
  return warmup_config;
}

class RecordingProfileEngine final : public Engine {
 public:
  explicit RecordingProfileEngine(bool linear_attention = false,
                                  InstanceRole role = InstanceRole::DEFAULT,
                                  int32_t num_speculative_tokens = 0,
                                  bool enable_prefix_cache = false)
      : linear_attention_(linear_attention),
        num_speculative_tokens_(num_speculative_tokens) {
    BlockManagerPool::Options options;
    options.num_blocks(/*num_blocks=*/64)
        .block_size(/*block_size=*/4)
        .enable_prefix_cache(enable_prefix_cache)
        .max_seqs_per_batch(/*max_seqs_per_batch=*/8)
        .enable_linear_state(linear_attention)
        .linear_state_num_slots(16)
        .num_embedding_blocks(16)
        .num_speculative_tokens(num_speculative_tokens)
        .instance_is_decode(role == InstanceRole::DECODE);
    block_manager_ = std::make_unique<BlockManagerPool>(options, /*dp_size=*/1);
    model_args_.vocab_size(128)
        .eos_token_id(2)
        .max_position_embeddings(16)
        .hidden_size(8);
    if (linear_attention) {
      model_args_.model_type("qwen3_5_moe_text")
          .layer_types({"linear_attention"})
          .max_position_embeddings(64);
    }
  }

  ForwardOutput step(std::vector<Batch>& batches) override {
    for (Batch& batch : batches) {
      std::vector<int32_t> prefill_lengths;
      prefill_lengths.reserve(batch.get_sequences().size());
      for (Sequence* sequence : batch.get_sequences()) {
        all_requests_marked_ =
            all_requests_marked_ && sequence->is_graph_warmup();
        if (linear_attention_) {
          EXPECT_GE(sequence->get_linear_state_slot_id(), 0);
        }
        if (sequence->is_prefill_stage()) {
          prefill_lengths.emplace_back(
              static_cast<int32_t>(sequence->num_tokens()));
        } else {
          ++decode_sequences_;
        }
      }
      if (!prefill_lengths.empty()) {
        prefill_batches_.emplace_back(std::move(prefill_lengths));
      }
    }
    return ForwardOutput();
  }

  void update_last_step_result(std::vector<Batch>& batches) override {
    (void)batches;
  }

  BlockManagerPool* block_manager_pool() const override {
    return block_manager_.get();
  }

  const ModelArgs& model_args() const override { return model_args_; }

  runtime::DecodeGraphWarmupConfig decode_graph_warmup_config() const override {
    return make_decode_graph_warmup_config(num_speculative_tokens_ + 1,
                                           num_speculative_tokens_,
                                           /*enable_no_padding=*/false);
  }

  std::vector<int64_t> get_active_activation_memory() const override {
    return {};
  }

  void reset_profile_markers() { all_requests_marked_ = true; }

  bool all_requests_marked() const { return all_requests_marked_; }

  const std::vector<std::vector<int32_t>>& prefill_batches() const {
    return prefill_batches_;
  }

  int32_t decode_sequences() const { return decode_sequences_; }

 private:
  std::unique_ptr<BlockManagerPool> block_manager_;
  ModelArgs model_args_;
  bool all_requests_marked_ = true;
  bool linear_attention_ = false;
  int32_t num_speculative_tokens_ = 0;
  std::vector<std::vector<int32_t>> prefill_batches_;
  int32_t decode_sequences_ = 0;
};

#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
class LinearProfileGraphWarmupTest
    : public ::testing::TestWithParam<std::tuple<bool, int32_t>> {
 protected:
  void SetUp() override {
    original_scheduler_ = SchedulerConfig::get_instance();
    original_execution_ = ExecutionConfig::get_instance();
    SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 4;
    SchedulerConfig::get_instance().enable_chunked_prefill() = true;
    SchedulerConfig::get_instance().enable_dp_fair_token_budget() = false;
    ExecutionConfig::get_instance().enable_graph() = true;
    ExecutionConfig::get_instance().disable_graph_warmup() = false;
  }

  void TearDown() override {
    SchedulerConfig::get_instance() = original_scheduler_;
    ExecutionConfig::get_instance() = original_execution_;
  }

  ProfileManager::Options options(InstanceRole role,
                                  int32_t token_budget = 10,
                                  int32_t sequence_budget = 4) const {
    ProfileManager::Options options;
    options.instance_role(role)
        .max_tokens_per_batch(token_budget)
        .max_seqs_per_batch(sequence_budget)
        .enable_schedule_overlap(std::get<0>(GetParam()));
    return options;
  }

 private:
  SchedulerConfig original_scheduler_;
  ExecutionConfig original_execution_;
};

TEST_P(LinearProfileGraphWarmupTest, PrefillWarmupDoesNotSplitIntoChunks) {
  RecordingProfileEngine engine(
      true, InstanceRole::PREFILL, std::get<1>(GetParam()));
  ProfileManager profile_manager(&engine, options(InstanceRole::PREFILL));

  EXPECT_EQ(engine.prefill_batches(),
            (std::vector<std::vector<int32_t>>{{10}}));
  EXPECT_EQ(engine.decode_sequences(), 0);
  EXPECT_TRUE(engine.all_requests_marked());
  EXPECT_EQ(engine.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0}));
}

TEST_P(LinearProfileGraphWarmupTest, PrefillRespectsSequenceLimit) {
  RecordingProfileEngine engine(
      true, InstanceRole::PREFILL, std::get<1>(GetParam()));
  ProfileManager profile_manager(&engine,
                                 options(InstanceRole::PREFILL, 10, 2));

  EXPECT_EQ(engine.prefill_batches(),
            (std::vector<std::vector<int32_t>>{{10}}));
}

TEST_P(LinearProfileGraphWarmupTest, EagerWarmupRespectsBackendSupport) {
  ExecutionConfig::get_instance().enable_graph() = false;
  RecordingProfileEngine engine(
      true, InstanceRole::PREFILL, std::get<1>(GetParam()));
  ProfileManager profile_manager(&engine, options(InstanceRole::PREFILL));
#if defined(USE_NPU)
  // NPU eager warmup accepts a prefill spanning multiple checkpoints.
  EXPECT_EQ(engine.prefill_batches(),
            (std::vector<std::vector<int32_t>>{{10}}));
#else
  EXPECT_TRUE(engine.prefill_batches().empty());
#endif
  EXPECT_EQ(engine.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0}));
}

TEST_P(LinearProfileGraphWarmupTest,
       ProfilingAcceptsUnalignedPrefixAndLongQuery) {
  ExecutionConfig::get_instance().disable_graph_warmup() = true;
  RecordingProfileEngine engine(
      true, InstanceRole::PREFILL, std::get<1>(GetParam()));
  ProfileManager profile_manager(&engine, options(InstanceRole::PREFILL));
  profile_manager.run_request(10, 3);
  profile_manager.run_request(8, 0);
  EXPECT_EQ(engine.prefill_batches(),
            (std::vector<std::vector<int32_t>>{{10}, {8}}));
  EXPECT_EQ(engine.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0}));
}

TEST_P(LinearProfileGraphWarmupTest, UnifiedWarmupSeedsDecodeSlots) {
  RecordingProfileEngine engine(
      true, InstanceRole::DEFAULT, std::get<1>(GetParam()), true);
  ProfileManager profile_manager(&engine, options(InstanceRole::DEFAULT));

  EXPECT_EQ(engine.prefill_batches(),
            (std::vector<std::vector<int32_t>>{{10}}));
  EXPECT_GT(engine.decode_sequences(), 0);
  EXPECT_TRUE(engine.all_requests_marked());
  EXPECT_EQ(engine.block_manager_pool()->num_blocks_in_prefix_cache(),
            (std::vector<size_t>{0}));
  EXPECT_EQ(engine.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0}));
}

TEST_P(LinearProfileGraphWarmupTest, DecodeOnlyWarmupAllocatesWorkingSlots) {
  RecordingProfileEngine engine(
      true, InstanceRole::DECODE, std::get<1>(GetParam()));
  ProfileManager profile_manager(&engine, options(InstanceRole::DECODE));

  EXPECT_TRUE(engine.prefill_batches().empty());
  EXPECT_GT(engine.decode_sequences(), 0);
  EXPECT_TRUE(engine.all_requests_marked());
  EXPECT_EQ(engine.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0}));
}

INSTANTIATE_TEST_SUITE_P(OverlapAndSpeculation,
                         LinearProfileGraphWarmupTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Values(0, 2)));

#endif

TEST(GraphWarmupTest, BuildsCanonicalBuckets) {
  const DecodeGraphWarmupPlan plan = get_compatibility_decode_graph_warmup_plan(
      /*max_global_batch_size=*/64, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes,
            (std::vector<int32_t>{1, 2, 4, 8, 16, 32, 48, 64}));
}

TEST(GraphWarmupTest, IncludesNonCanonicalMaxBucket) {
  const DecodeGraphWarmupPlan plan = get_compatibility_decode_graph_warmup_plan(
      /*max_global_batch_size=*/40, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{1, 2, 4, 8, 16, 32, 40}));
}

TEST(GraphWarmupTest, SkipsBucketsBelowDpSize) {
  const DecodeGraphWarmupPlan plan = get_compatibility_decode_graph_warmup_plan(
      /*max_global_batch_size=*/16, /*dp_size=*/4);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{4, 8, 16}));
}

TEST(GraphWarmupTest, AllowsAllBucketsSkipped) {
  const DecodeGraphWarmupPlan plan = get_compatibility_decode_graph_warmup_plan(
      /*max_global_batch_size=*/2, /*dp_size=*/4);

  EXPECT_TRUE(plan.batch_sizes.empty());
}

TEST(DecodeGraphWarmupPlanTest, CoversEveryPaddedMtpTokenBucket) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const runtime::DecodeGraphWarmupConfig warmup_config =
      make_decode_graph_warmup_config(
          /*num_decoding_tokens=*/4,
          /*num_speculative_tokens=*/3,
          /*enable_no_padding=*/false);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/64, /*dp_size=*/1);

  EXPECT_EQ(
      plan.batch_sizes,
      (std::vector<int32_t>{
          1, 2, 3, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61}));
}

TEST(DecodeGraphBucketTest, UsesSharedWidthForActiveDpShards) {
  EXPECT_EQ(runtime::get_decode_graph_token_bucket(
                /*num_tokens=*/4, /*enable_no_padding=*/false),
            4);
  EXPECT_EQ(runtime::get_decode_graph_token_bucket(
                /*num_tokens=*/20, /*enable_no_padding=*/false),
            32);
  EXPECT_EQ(
      runtime::get_decode_graph_dp_token_counts({4, 0, 0, 0, 0, 0, 0, 0}, 4),
      (std::vector<int32_t>{4, 1, 1, 1, 1, 1, 1, 1}));
  EXPECT_EQ(runtime::get_decode_graph_dp_token_counts(
                {16, 16, 20, 12, 16, 16, 12, 20}, 32),
            (std::vector<int32_t>{32, 32, 32, 32, 32, 32, 32, 32}));
  EXPECT_EQ(runtime::get_decode_graph_dp_layout_token_count(
                /*dp_size=*/8, /*graph_token_count=*/32),
            256);
  EXPECT_EQ(runtime::get_decode_graph_dp_layout_token_count(
                /*dp_size=*/8, /*graph_token_count=*/4),
            32);
}

TEST(DecodeGraphWarmupPlanTest, UsesLocalDpBatchesAndKeepsPartialBatch) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const runtime::DecodeGraphWarmupConfig warmup_config =
      make_decode_graph_warmup_config(
          /*num_decoding_tokens=*/4,
          /*num_speculative_tokens=*/0,
          /*enable_no_padding=*/false);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/66, /*dp_size=*/4);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{4, 8, 12, 20, 36, 52, 66}));
}

TEST(DecodeGraphWarmupPlanTest, Dp32C64UsesLocalGraphBuckets) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const runtime::DecodeGraphWarmupConfig warmup_config =
      make_decode_graph_warmup_config(
          /*num_decoding_tokens=*/4,
          /*num_speculative_tokens=*/3,
          /*enable_no_padding=*/false);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/64, /*dp_size=*/32);

  // The scheduler batch is global, while graph shapes are keyed by the
  // largest local DP batch. C64 therefore needs the local-batch-2 bucket.
  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{32, 64}));
}

TEST(DecodeGraphWarmupPlanTest,
     GraphLimitKeepsCppBatchSemanticsAndCoversMtpBuckets) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const runtime::DecodeGraphWarmupConfig warmup_config =
      make_decode_graph_warmup_config(
          /*num_decoding_tokens=*/4,
          /*num_speculative_tokens=*/3,
          /*enable_no_padding=*/false,
          /*max_graph_batch_size=*/16);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/64, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes,
            (std::vector<int32_t>{1, 2, 4, 8, 12, 16, 32, 48, 64}));
}

TEST(DecodeGraphWarmupPlanTest,
     Mtp1GraphLimitPreservesCompatibilityBatchSizes) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const runtime::DecodeGraphWarmupConfig warmup_config =
      make_decode_graph_warmup_config(
          /*num_decoding_tokens=*/2,
          /*num_speculative_tokens=*/1,
          /*enable_no_padding=*/false,
          /*max_graph_batch_size=*/16);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/64, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes,
            (std::vector<int32_t>{1, 2, 4, 8, 16, 32, 48, 64}));
}

// The graph batch limit caps the DP-local decode batch, not the
// scheduler-global one, so with dp_size > 1 the sweep must still reach every
// graph-eligible local batch's token bucket.
TEST(DecodeGraphWarmupPlanTest, GraphLimitIsLocalAcrossDpGroups) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const runtime::DecodeGraphWarmupConfig warmup_config =
      make_decode_graph_warmup_config(
          /*num_decoding_tokens=*/6,
          /*num_speculative_tokens=*/5,
          /*enable_no_padding=*/false,
          /*max_graph_batch_size=*/16);
  const int32_t dp_size = 4;
  const int32_t max_global_batch_size = 64;
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, max_global_batch_size, dp_size);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{4, 8, 20, 32, 40, 52, 64}));

  std::vector<int64_t> warmed_buckets;
  for (const int32_t batch_size : plan.batch_sizes) {
    const int64_t num_tokens = static_cast<int64_t>(batch_size / dp_size) *
                               warmup_config.num_decoding_tokens;
    warmed_buckets.push_back(runtime::get_decode_graph_token_bucket(
        num_tokens, /*enable_no_padding=*/false));
  }
  for (int32_t local_batch_size = 1;
       local_batch_size <= std::min(max_global_batch_size / dp_size,
                                    warmup_config.max_graph_batch_size);
       ++local_batch_size) {
    const int64_t num_tokens = static_cast<int64_t>(local_batch_size) *
                               warmup_config.num_decoding_tokens;
    const int64_t bucket = runtime::get_decode_graph_token_bucket(
        num_tokens, /*enable_no_padding=*/false);
    EXPECT_NE(std::find(warmed_buckets.begin(), warmed_buckets.end(), bucket),
              warmed_buckets.end())
        << "local batch " << local_batch_size
        << " maps to unwarmed token bucket " << bucket;
  }
}

TEST(DecodeGraphWarmupPlanTest, NoPaddingKeepsCompatibilityBatches) {
  const runtime::DecodeGraphWarmupConfig warmup_config =
      make_decode_graph_warmup_config(
          /*num_decoding_tokens=*/4,
          /*num_speculative_tokens=*/3,
          /*enable_no_padding=*/true);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/64, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes,
            (std::vector<int32_t>{1, 2, 4, 8, 16, 32, 48, 64}));
}

TEST(DecodeGraphWarmupPlanTest,
     NoPaddingGraphLimitAddsIntermediateMtpTokenBuckets) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const runtime::DecodeGraphWarmupConfig warmup_config =
      make_decode_graph_warmup_config(
          /*num_decoding_tokens=*/4,
          /*num_speculative_tokens=*/3,
          /*enable_no_padding=*/true,
          /*max_graph_batch_size=*/16);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/16, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{1, 2, 4, 8, 12, 16}));
}

TEST(DecodeGraphWarmupPlanTest, NoPaddingMtpKeepsNonDivisibleDpTailBuckets) {
  if (!Platform::supports_mtp_decode_graph_warmup()) {
    GTEST_SKIP() << "MTP decode graph warmup is not supported.";
  }

  const runtime::DecodeGraphWarmupConfig warmup_config =
      make_decode_graph_warmup_config(
          /*num_decoding_tokens=*/4,
          /*num_speculative_tokens=*/3,
          /*enable_no_padding=*/true,
          /*max_graph_batch_size=*/16);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/7, /*dp_size=*/2);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{2, 4, 6, 7}));
}

TEST(DecodeGraphWarmupPlanTest, PreservesSuppliedWarmupConfig) {
  const runtime::DecodeGraphWarmupConfig warmup_config =
      make_decode_graph_warmup_config(
          /*num_decoding_tokens=*/4,
          /*num_speculative_tokens=*/3,
          /*enable_no_padding=*/false);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/16, /*dp_size=*/1);

  EXPECT_EQ(plan.warmup_config.num_decoding_tokens, 4);
  EXPECT_EQ(plan.warmup_config.num_speculative_tokens, 3);
  EXPECT_FALSE(plan.warmup_config.enable_graph_mode_decode_no_padding);
  EXPECT_EQ(plan.warmup_config.max_graph_batch_size, 0);
  if (Platform::supports_mtp_decode_graph_warmup()) {
    EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{1, 2, 3, 5, 9, 13}));
  } else {
    EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{1, 2, 4, 8, 16}));
  }
}

TEST(DecodeGraphWarmupPlanTest, RespectsReducedWarmupCapacity) {
  const runtime::DecodeGraphWarmupConfig warmup_config =
      make_decode_graph_warmup_config(
          /*num_decoding_tokens=*/1,
          /*num_speculative_tokens=*/0,
          /*enable_no_padding=*/false);
  const DecodeGraphWarmupPlan plan = build_decode_graph_warmup_plan(
      warmup_config, /*max_global_batch_size=*/15, /*dp_size=*/1);

  EXPECT_EQ(plan.batch_sizes, (std::vector<int32_t>{1, 2, 4, 8, 15}));
}

TEST(GraphWarmupTest, PrefillRoleUsesPrefillOnlyPlan) {
  EXPECT_EQ(graph_warmup_plan(InstanceRole::PREFILL),
            GraphWarmupPlan::PREFILL_ONLY);
}

TEST(GraphWarmupTest, NonPrefillRolesUseUnifiedPlan) {
  EXPECT_EQ(graph_warmup_plan(InstanceRole::DEFAULT), GraphWarmupPlan::UNIFIED);
  EXPECT_EQ(graph_warmup_plan(InstanceRole::MIX), GraphWarmupPlan::UNIFIED);
  EXPECT_EQ(graph_warmup_plan(InstanceRole::INVALID), GraphWarmupPlan::UNIFIED);
}

TEST(GraphWarmupTest, DecodeRoleUsesDecodeOnlyPlan) {
  EXPECT_EQ(graph_warmup_plan(InstanceRole::DECODE),
            GraphWarmupPlan::DECODE_ONLY);
}

TEST(GraphWarmupTest, FormatsWarmupProgress) {
  const std::string progress = graph_warmup_progress(
      /*completed=*/3, /*total=*/8, /*token_bucket=*/8, /*latency_ms=*/12.5);

  EXPECT_EQ(progress,
            "Graph warmup progress: [########------------] 3/8 37.5%, "
            "token_bucket=8, latency=12.50 ms");
}

TEST(GraphWarmupTest, FormatsCompletedWarmupProgress) {
  const std::string progress = graph_warmup_progress(
      /*completed=*/8,
      /*total=*/8,
      /*token_bucket=*/64,
      /*latency_ms=*/100.0);

  EXPECT_EQ(progress,
            "Graph warmup progress: [####################] 8/8 100.0%, "
            "token_bucket=64, latency=100.00 ms");
}

TEST(GraphWarmupTest, InjectsBootstrapEmbeddingWhenSpeculativeEnabled) {
  Sequence sequence = make_sequence(/*index=*/0, /*tokens=*/{1, 2, 3});

  prepare_warmup_decode_sequence(&sequence,
                                 /*embedding_width=*/128,
                                 /*num_speculative_tokens=*/3);

  const torch::Tensor embedding = sequence.get_mtp_bootstrap_embedding();
  ASSERT_TRUE(embedding.defined());
  EXPECT_EQ(embedding.dim(), 2);
  EXPECT_EQ(embedding.size(0), 1);
  EXPECT_EQ(embedding.size(1), 128);
}

TEST(GraphWarmupTest, DeepseekV4MtpUsesFlattenedHyperConnectionWidth) {
  ModelArgs model_args;
  model_args.model_type() = "deepseek_v4_mtp";
  model_args.hidden_size() = 128;
  model_args.hc_mult() = 4;

  EXPECT_EQ(mtp_hidden_state_width(model_args), 512);
}

TEST(GraphWarmupTest, OtherMtpModelsUseDenseHiddenWidth) {
  ModelArgs model_args;
  model_args.model_type() = "deepseek_v32_mtp";
  model_args.hidden_size() = 128;
  model_args.hc_mult() = 4;

  EXPECT_EQ(mtp_hidden_state_width(model_args), 128);
}

TEST(GraphWarmupTest, SkipsBootstrapEmbeddingWhenSpeculativeDisabled) {
  Sequence sequence = make_sequence(/*index=*/0, /*tokens=*/{1, 2, 3});

  prepare_warmup_decode_sequence(&sequence,
                                 /*embedding_width=*/128,
                                 /*num_speculative_tokens=*/0);

  EXPECT_FALSE(sequence.get_mtp_bootstrap_embedding().defined());
}

TEST(GraphWarmupTest, ProducesUniqueWarmupRequestIds) {
  const std::string first = next_warmup_request_id();
  const std::string second = next_warmup_request_id();

  EXPECT_FALSE(first.empty());
  EXPECT_FALSE(second.empty());
  EXPECT_NE(first, second);
}

TEST(GraphWarmupTest, PresetDpRankControlsBlockAllocation) {
  BlockManagerPool::Options options;
  options.num_blocks(/*num_blocks=*/8)
      .block_size(/*block_size=*/2)
      .enable_prefix_cache(/*enable_prefix_cache=*/false)
      .max_seqs_per_batch(/*max_seqs_per_batch=*/4);
  BlockManagerPool pool(options, /*dp_size=*/2);
  Sequence sequence = make_sequence(/*index=*/0, /*tokens=*/{1, 2, 3});
  sequence.set_dp_rank(/*dp_rank=*/1);
  const std::vector<size_t> used_before = pool.num_used_blocks();

  ASSERT_TRUE(pool.allocate(&sequence, /*num_tokens=*/3));

  EXPECT_EQ(sequence.dp_rank(), 1);
  EXPECT_EQ(pool.num_used_blocks()[0], used_before[0]);
  EXPECT_GT(pool.num_used_blocks()[1], used_before[1]);
}

TEST(GraphWarmupTest, MarksOrdinaryProfileRequestsAsSyntheticLoad) {
  RecordingProfileEngine engine;
  ProfileManager::Options options;
  options.max_tokens_per_batch(4).max_seqs_per_batch(1).dp_size(1);
  ProfileManager profile_manager(&engine, options);
  engine.reset_profile_markers();

  profile_manager.run_request(/*token_length=*/4, /*prefix_length=*/0);

  EXPECT_TRUE(engine.all_requests_marked());
}

}  // namespace
}  // namespace xllm
