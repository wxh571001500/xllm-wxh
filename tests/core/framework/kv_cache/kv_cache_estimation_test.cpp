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

#include "framework/kv_cache/kv_cache_estimation.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "core/framework/config/parallel_config.h"
#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "framework/kv_cache/deepseek_v4_cache_policy.h"
#include "framework/model/model_args.h"

namespace xllm {
namespace {

class DcpIndexerEstimationTest : public ::testing::TestWithParam<int32_t> {
 protected:
  void SetUp() override {
    ParallelConfig::get_instance().kv_split_size(GetParam()).cp_size(4);
  }

  void TearDown() override { ParallelConfig::get_instance() = saved_config_; }

  int64_t expected_factor() const {
#if defined(USE_NPU)
    return GetParam() == 0 ? 4 : GetParam();
#else
    return 1;
#endif
  }

 private:
  ParallelConfig saved_config_ = ParallelConfig::get_instance();
};

ModelArgs make_standard_args() {
  ModelArgs model_args;
  model_args.n_layers(4).head_dim(16);
  return model_args;
}

KVCacheEstimateOptions make_estimate_options() {
  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat16;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 1024 * 1024;
  options.block_size = 16;
  options.world_size = 1;
  options.n_local_kv_heads = 2;
  options.max_seqs_per_batch = 8;
  return options;
}

}  // namespace

TEST(KVCacheEstimationTest, EstimatesStandardAttentionBlocks) {
  ModelArgs model_args = make_standard_args();
  KVCacheEstimateOptions options = make_estimate_options();

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.cache_size_in_bytes(), 1024 * 1024);
  EXPECT_EQ(capacity.block_size(), 16);
  EXPECT_EQ(capacity.slot_size(), 128);
  EXPECT_EQ(capacity.n_layers(), 4);
  EXPECT_EQ(capacity.num_full_attention_layers(), 4);
  EXPECT_EQ(capacity.num_linear_attention_layers(), 0);
  EXPECT_EQ(capacity.n_blocks(), 128);
}

TEST(KVCacheEstimationTest, IgnoresLinearStateSlotsWithoutLinearAttention) {
  ModelArgs model_args = make_standard_args();
  KVCacheEstimateOptions options = make_estimate_options();
  options.max_linear_state_cache_slots = 32;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_linear_attention_layers(), 0);
  EXPECT_EQ(capacity.num_linear_state_blocks(), 2);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 0);
}

TEST(KVCacheEstimationTest, IndexCacheSizesMustBeConfiguredTogether) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        KVCacheCapacity capacity;
        capacity.index_cache_sizes(/*slot_size=*/8, /*block_size=*/0);
      },
      "slot and block sizes must both be zero or positive");
}

TEST(KVCacheEstimationTest, UserIndexerCacheDtypeDirectlyControlsQuantization) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("unsupported_model")
      .index_n_heads(1)
      .index_head_dim(16);
  KVCacheEstimateOptions options = make_estimate_options();

  options.indexer_cache_dtype = "auto";
  KVCacheCapacity auto_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(auto_capacity.index_slot_size(), 32);
  EXPECT_FALSE(auto_capacity.enable_indexer_cache_quant());

  options.indexer_cache_dtype = "int8";
  KVCacheCapacity int8_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(int8_capacity.index_slot_size(), 20);
  EXPECT_TRUE(int8_capacity.enable_indexer_cache_quant());

#if defined(USE_NPU)
  ModelArgs kpool_args = make_standard_args();
  kpool_args.model_type("glm5_next")
      .index_n_heads(32)
      .index_head_dim(128)
      .index_kpool(4)
      .index_kpool_compress(true);
  KVCacheEstimateOptions kpool_options = make_estimate_options();
  kpool_options.block_size = 128;
  kpool_options.dtype = torch::kBFloat16;
  kpool_options.kpool_layout = KPoolCacheLayout::COMPRESSED_WITH_TAIL;

  const KVCacheCapacity packed_capacity =
      estimate_kv_cache_capacity(kpool_args, kpool_options);
  EXPECT_EQ(packed_capacity.kpool_layout(), KPoolCacheLayout::PACKED);
  EXPECT_EQ(packed_capacity.index_block_size(),
            packed_capacity.index_slot_size() * kpool_options.block_size);

  kpool_args.index_kpool_always_select_tail(true);
  kpool_options.num_speculative_tokens = 3;
  kpool_options.kpool_layout = KPoolCacheLayout::PACKED;
  const KVCacheCapacity compressed_capacity =
      estimate_kv_cache_capacity(kpool_args, kpool_options);
  EXPECT_EQ(compressed_capacity.kpool_layout(),
            KPoolCacheLayout::COMPRESSED_WITH_TAIL);
  EXPECT_EQ(compressed_capacity.kpool_tail_len(), 7);

  kpool_args.model_type("deepseek_v32").index_n_heads(1).index_head_dim(16);
  const KVCacheCapacity non_target_capacity =
      estimate_kv_cache_capacity(kpool_args, kpool_options);
  EXPECT_EQ(non_target_capacity.kpool_layout(), KPoolCacheLayout::PACKED);

  kpool_args.model_type("glm5_next")
      .index_n_heads(32)
      .index_head_dim(128)
      .index_kpool(4)
      .index_kpool_compress(true)
      .index_kpool_always_select_tail(true);
  kpool_options.indexer_cache_dtype = "int8";

  EXPECT_DEATH((void)estimate_kv_cache_capacity(kpool_args, kpool_options),
               "KPool cache layouts do not support.*int8");
#endif
}

TEST(KVCacheEstimationTest, IndexerScaleUsesLogicalCacheCapacity) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("deepseek_v32").index_n_heads(1).index_head_dim(16);
  KVCacheEstimateOptions options = make_estimate_options();
  options.indexer_cache_dtype = "int8";
  options.cache_size_in_bytes = 10 * 1024 * 1024;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.n_blocks(), 1107);
}

#if defined(USE_MLU) || defined(USE_NPU)
TEST(KVCacheEstimationTest, SharedDsaLayersDoNotConsumeIndexerCacheBudget) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("glm_moe_dsa")
      .index_n_heads(1)
      .index_head_dim(16)
      .index_topk(8)
      .index_topk_pattern("FSFS");
  KVCacheEstimateOptions options = make_estimate_options();

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_full_attention_layers(), 4);
  EXPECT_EQ(capacity.num_indexer_layers(), 2);
  EXPECT_EQ(capacity.n_blocks(), 113);
}

TEST(KVCacheEstimationTest, GlmSharedIndexerFreqAllocatesOnlyFullLayers) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("glm_moe_dsa")
      .n_layers(78)
      .index_n_heads(1)
      .index_head_dim(16)
      .index_topk(8)
      .index_topk_freq(4)
      .index_skip_topk_offset(3);

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, make_estimate_options());

  EXPECT_EQ(capacity.num_full_attention_layers(), 78);
  EXPECT_EQ(capacity.num_indexer_layers(), 21);
}

#endif

namespace {

ModelArgs make_linear_attention_args(int64_t head_dim = 16) {
  ModelArgs model_args = make_standard_args();
  model_args.head_dim(head_dim)
      .full_attention_interval(2)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  return model_args;
}

KVCacheEstimateOptions make_linear_attention_options() {
  KVCacheEstimateOptions options = make_estimate_options();
  options.n_local_linear_k_heads = 2;
  options.n_local_linear_v_heads = 2;
  return options;
}

}  // namespace

TEST(KVCacheEstimationTest, ReservesLinearAttentionState) {
  ModelArgs model_args = make_linear_attention_args();
  KVCacheEstimateOptions options = make_linear_attention_options();

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_full_attention_layers(), 2);
  EXPECT_EQ(capacity.num_linear_attention_layers(), 2);
  EXPECT_EQ(capacity.num_linear_state_blocks(), 229);
  EXPECT_EQ(capacity.linear_slot_size(), 256);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 117248);
  EXPECT_EQ(capacity.n_blocks(), 227);
}

TEST(KVCacheEstimationTest, LinearStateCapacityDoesNotDependOnPrefixCache) {
  const ModelArgs model_args = make_linear_attention_args(1);
  for (const int64_t explicit_slots : {0, 32}) {
    KVCacheEstimateOptions options = make_linear_attention_options();
    options.block_size = 1;
    options.n_local_kv_heads = 1;
    options.max_concurrent_requests = 1;
    options.max_linear_state_cache_slots = explicit_slots;
    options.enable_prefix_cache = false;
    const KVCacheCapacity disabled =
        estimate_kv_cache_capacity(model_args, options);
    options.enable_prefix_cache = true;
    const KVCacheCapacity enabled =
        estimate_kv_cache_capacity(model_args, options);
    EXPECT_EQ(disabled.num_linear_state_blocks(),
              enabled.num_linear_state_blocks());
    EXPECT_EQ(disabled.n_blocks(), enabled.n_blocks());
    EXPECT_EQ(disabled.linear_cache_size_in_bytes(),
              enabled.linear_cache_size_in_bytes());
    EXPECT_EQ(disabled.num_linear_state_blocks(),
              explicit_slots > 0 ? 34 : 970);
    EXPECT_GT(disabled.n_blocks(), 0);
    EXPECT_LE(disabled.linear_cache_size_in_bytes(),
              options.cache_size_in_bytes);
  }
}

#if defined(USE_MLU)
TEST(KVCacheEstimationTest, Glm5NextSpecVerifyExpandsHybridState) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("glm5_next")
      .enable_mla(true)
      .kv_lora_rank(8)
      .qk_rope_head_dim(0)
      .index_n_heads(1)
      .index_head_dim(16)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(4)
      .linear_conv_kernel_dim(4)
      .layer_types({"linear_attention",
                    "deepseek_sparse_attention",
                    "linear_attention",
                    "deepseek_sparse_attention"});
  KVCacheEstimateOptions options = make_estimate_options();
  options.n_local_linear_k_heads = 2;
  options.n_local_linear_v_heads = 2;
  options.num_speculative_tokens = 2;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.linear_conv_state_len(), 5);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 3);
}

TEST(KVCacheEstimationTest, KPoolReservesSpeculativeTailWithRequestState) {
  ModelArgs args = make_linear_attention_args();
  args.index_n_heads(1).index_head_dim(16).index_kpool(4).index_kpool_compress(
      true);
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.dtype = torch::kBFloat16;
  options.num_speculative_tokens = 5;
  // Fix the request-state capacity; automatic sizing follows the memory budget.
  options.max_linear_state_cache_slots = 8;
  const KVCacheCapacity capacity = estimate_kv_cache_capacity(args, options);
  // Two recurrent layers and two indexer layers. Each tail keeps K + W
  // BF16 key/gate rows; W includes the verify base token and prelaunch room.
  EXPECT_EQ(capacity.num_linear_state_blocks(), 10);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 25600);
  EXPECT_EQ(
      capacity.num_linear_attention_layers() * capacity.linear_slot_size(),
      512);
  EXPECT_EQ(capacity.num_indexer_layers() * capacity.kpool_tail_slot_size(),
            2048);
  EXPECT_EQ(capacity.kpool_tail_len(), 16);
  EXPECT_LE(capacity.n_blocks() * 16 * (2 * 128 + 2 * 8) +
                capacity.linear_cache_size_in_bytes(),
            options.cache_size_in_bytes);
  args.num_nextn_predict_layers(2);
  const KVCacheCapacity speculative = estimate_kv_cache_capacity(args, options);
  EXPECT_EQ(speculative.num_full_attention_layers(), 2);
  EXPECT_EQ(speculative.num_indexer_layers(), 2);
  EXPECT_EQ(speculative.linear_cache_size_in_bytes(),
            capacity.linear_cache_size_in_bytes());
  EXPECT_EQ(speculative.n_blocks(), capacity.n_blocks());
}
#endif

TEST(KVCacheEstimationTest, LinearStateCapacityVariants) {
  struct TestCase {
    const char* name;
    int64_t head_dim;
    int64_t cache_size_in_bytes;
    int64_t block_size;
    int64_t n_local_kv_heads;
    int64_t max_seqs_per_batch;
    bool enable_prefix_cache;
    int64_t max_linear_state_cache_slots;
    int64_t expected_num_linear_state_blocks;
    int64_t min_num_linear_state_blocks;
  };

  const std::vector<TestCase> test_cases = {
      {"PrefixCacheGrowsLinearStateCheckpointPool",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/64LL << 30,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/200,
       /*enable_prefix_cache=*/true,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/-1,
       /*min_num_linear_state_blocks=*/202},
      {"PrefixCacheUsesLinearStateMemoryRatio",
       /*head_dim=*/1,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/1,
       /*n_local_kv_heads=*/1,
       /*max_seqs_per_batch=*/8,
       /*enable_prefix_cache=*/true,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/970,
       /*min_num_linear_state_blocks=*/-1},
      {"NoPrefixCacheCapsLinearStateBlocksByBudget",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/100000,
       /*enable_prefix_cache=*/false,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/229,
       /*min_num_linear_state_blocks=*/-1},
      {"UnlimitedConcurrencyUsesLinearStateMemoryRatio",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/1024 * 1024,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/0,
       /*enable_prefix_cache=*/false,
       /*max_linear_state_cache_slots=*/0,
       /*expected_num_linear_state_blocks=*/229,
       /*min_num_linear_state_blocks=*/-1},
      {"ExplicitLinearStateSlotsOverrideAutoSizing",
       /*head_dim=*/16,
       /*cache_size_in_bytes=*/64LL << 30,
       /*block_size=*/16,
       /*n_local_kv_heads=*/2,
       /*max_seqs_per_batch=*/200,
       /*enable_prefix_cache=*/true,
       /*max_linear_state_cache_slots=*/32,
       /*expected_num_linear_state_blocks=*/34,
       /*min_num_linear_state_blocks=*/-1},
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    ModelArgs model_args = make_linear_attention_args(test_case.head_dim);
    KVCacheEstimateOptions options = make_linear_attention_options();
    options.cache_size_in_bytes = test_case.cache_size_in_bytes;
    options.block_size = test_case.block_size;
    options.n_local_kv_heads = test_case.n_local_kv_heads;
    options.max_seqs_per_batch = test_case.max_seqs_per_batch;
    options.enable_prefix_cache = test_case.enable_prefix_cache;
    options.max_linear_state_cache_slots =
        test_case.max_linear_state_cache_slots;

    KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

    if (test_case.expected_num_linear_state_blocks >= 0) {
      EXPECT_EQ(capacity.num_linear_state_blocks(),
                test_case.expected_num_linear_state_blocks);
    }
    if (test_case.min_num_linear_state_blocks >= 0) {
      EXPECT_GT(capacity.num_linear_state_blocks(),
                test_case.min_num_linear_state_blocks);
    }
  }
}

#if defined(USE_NPU)
TEST(KVCacheEstimationTest, CapsQwen35PhysicalLinearStateSlotsForMegaGdn) {
  ModelArgs model_args = make_linear_attention_args(/*head_dim=*/1);
  model_args.model_type("qwen3_5");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.cache_size_in_bytes = 64LL << 30;
  options.enable_prefix_cache = true;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.num_linear_state_blocks(), 1024);
}

TEST(KVCacheEstimationTest, RejectsQwen35ExplicitSlotsBeyondMegaGdnLimit) {
  ModelArgs model_args = make_linear_attention_args(/*head_dim=*/1);
  model_args.model_type("qwen3_5");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.cache_size_in_bytes = 64LL << 30;
  options.enable_prefix_cache = true;
  options.max_linear_state_cache_slots = 1023;

  EXPECT_DEATH((void)estimate_kv_cache_capacity(model_args, options),
               "MegaGdn supports at most 1024 physical linear-state slots");
}
#endif

TEST(KVCacheEstimationTest, Qwen35MtpExpandsConvStateLen) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("qwen3_5")
      .full_attention_interval(2)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  KVCacheEstimateOptions options = make_estimate_options();
  options.n_local_linear_k_heads = 2;
  options.n_local_linear_v_heads = 2;
  options.num_speculative_tokens = 1;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.linear_conv_state_len(), 3);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 2);
  EXPECT_EQ(capacity.linear_slot_size(), 448);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 189056);
}

TEST(KVCacheEstimationTest, Qwen35TextMtpUsesSsmCheckpointStride) {
  ModelArgs model_args = make_standard_args();
  model_args.model_type("qwen3_5_text")
      .full_attention_interval(2)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  KVCacheEstimateOptions options = make_estimate_options();
  options.n_local_linear_k_heads = 2;
  options.n_local_linear_v_heads = 2;
  options.num_speculative_tokens = 1;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.linear_conv_state_len(), 3);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 2);
  EXPECT_EQ(capacity.linear_slot_size(), 448);
  EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 189056);
}

#if defined(USE_NPU)
TEST(KVCacheEstimationTest, Glm5MtpPreallocatesSpeculativeCheckpoints) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.model_type("glm5_next");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.num_speculative_tokens = 3;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.linear_conv_state_len(), 5);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 4);
  EXPECT_EQ(capacity.linear_slot_size(), 832);
}

TEST(KVCacheEstimationTest, Glm5PrefillDoesNotReserveMtpCheckpoints) {
  ModelArgs model_args = make_linear_attention_args();
  model_args.model_type("glm5_next");
  KVCacheEstimateOptions options = make_linear_attention_options();
  options.num_speculative_tokens = 3;
  options.instance_role = InstanceRole::PREFILL;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.linear_conv_state_len(), 2);
  EXPECT_EQ(capacity.linear_ssm_checkpoint_stride(), 1);
  EXPECT_LT(capacity.linear_slot_size(), 832);
}
#endif

TEST(KVCacheEstimationTest, EstimatesDeepSeekV4Pools) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes =
      2818048 + /*sixteen_additional_swa_blocks=*/16 * 90112;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 2176;

  KVCacheCapacity capacity = estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 35);
  const Dsv4CacheGeometry geometry;
  const int64_t manager_blocks_per_unit =
      geometry.compressed_block_token_size() / options.block_size;
  constexpr int64_t kSwaBytesPerBlock =
      128 * 16 * 4 + 128 * (16 * 4 + 2 * 16 * 4 * 2 + 2 * 8 * 4 * 2) +
      128 * (16 * 4 + 16 * 4 * 2);
#if defined(USE_MLU)
  constexpr int64_t kIndexBytes = 4;
  constexpr int64_t kScaleBytes = 0;
  constexpr int64_t kExpectedUnits = 352;
#else
  constexpr int64_t kIndexBytes = 1;
  constexpr int64_t kScaleBytes = 2;
  constexpr int64_t kExpectedUnits = 28;
#endif
  const int64_t token_unit_bytes =
      geometry.c4_physical_dim() * (16 * 4 + 8 * kIndexBytes + kScaleBytes) +
      geometry.c128_physical_dim() * 16 * 4;
  const int64_t expected_units =
      (options.cache_size_in_bytes - 35 * kSwaBytesPerBlock) / token_unit_bytes;
  EXPECT_EQ(expected_units, kExpectedUnits);
  EXPECT_EQ(capacity.c4_count(), expected_units);
  EXPECT_EQ(capacity.c128_count(), expected_units);
  EXPECT_EQ(capacity.n_blocks(), expected_units * manager_blocks_per_unit);
}

TEST(KVCacheEstimationTest, DeepSeekV4RejectsBudgetWithoutCompressedCacheUnit) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(128)
      .compress_ratios({1, 4, 128});

  constexpr int64_t kSwaCount = 7;
  constexpr int64_t kSwaBytesPerBlock =
      /*c1=*/128 * 16 * 4 +
      /*c4=*/128 * (16 * 4 + 2 * 16 * 4 * 2 + 2 * 8 * 4 * 2) +
      /*c128=*/128 * (16 * 4 + 16 * 4 * 2);
  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes =
      kSwaCount * kSwaBytesPerBlock + /*remaining_bytes=*/1;
  options.block_size = 128;
  options.max_seqs_per_batch = 1;
  options.max_tokens_per_batch = 384;
  options.max_tokens_per_chunk_for_prefill = 128;

  EXPECT_DEATH(
      estimate_kv_cache_capacity(model_args, options),
      "minimum DSV4 SWA cache leaves insufficient memory for one compressed "
      "cache unit");
}

TEST(KVCacheEstimationTest, DeepSeekV4PdPrefillUsesBatchTokenCapacity) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.max_tokens_per_chunk_for_prefill = 385;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // Four sequences retain three window blocks each, plus four burst blocks,
  // four per-sequence tail blocks, and two guard blocks.
  EXPECT_EQ(capacity.swa_count(), 22);
}

TEST(KVCacheEstimationTest, DeepSeekV4MixUsesBatchTokenCapacity) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 16384;
  options.max_tokens_per_chunk_for_prefill = 385;
  options.instance_role = InstanceRole::MIX;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 146);
}

TEST(KVCacheEstimationTest,
     DeepSeekV4PdPrefillWithoutChunkingUsesBatchCapacity) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.max_tokens_per_chunk_for_prefill = 129;
  options.enable_chunked_prefill = false;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // Chunking does not affect SWA sizing.
  EXPECT_EQ(capacity.swa_count(), 22);
}

// The SWA ring backs both the live prefill burst and the rows of a published
// prefix-cache checkpoint. The fair per-DP token budget only bounds the
// former, so a ring sized from the per-DP share alone recycles checkpoint rows
// before the next request can probe them and turns every cross-request prefix
// hit into a full re-prefill. The ring must keep one publish unit of rows.
TEST(KVCacheEstimationTest,
     DeepSeekV4PdPrefillFairBudgetKeepsPrefixCachePublishUnit) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 64 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 16384;
  options.max_tokens_per_chunk_for_prefill = 4096;
  options.enable_prefix_cache = true;
  options.enable_dp_fair_token_budget = true;
  options.dp_size = 4;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // Four sequences retain three window blocks each, one C128 publish unit of
  // checkpoint rows (128 blocks), four per-sequence tail blocks, and two
  // guard blocks. The per-DP share alone would only fund 32 burst blocks.
  EXPECT_EQ(capacity.swa_count(), 146);

  // Without a compressed publish unit there is nothing to retain, so the
  // per-DP share still sizes the ring.
  KVCacheEstimateOptions no_prefix_options = options;
  no_prefix_options.enable_prefix_cache = false;
  const KVCacheCapacity no_prefix_capacity =
      estimate_kv_cache_capacity(model_args, no_prefix_options);
  EXPECT_EQ(no_prefix_capacity.swa_count(), 50);
}

// Same guard, replayed with the shipped DeepSeek-V4-Flash-0731-w8a8 shape and
// the exact serving parameters of
// deepseek_v4_xllm_perf_prefix_i30000_o1024_bs1_dp4_ep16, so this test predicts
// the startup log line the CICD run prints:
//   Initializing DSV4 kv cache with shape: [swa_count=162, ...]   (fixed)
//   Initializing DSV4 kv cache with shape: [swa_count=66, ...]    (regression)
// swa_count is a function of the window, max_seqs_per_batch, the per-step
// token bound and the compress ratios only -- it does not depend on the memory
// budget -- so the two budget cases below must agree.
TEST(KVCacheEstimationTest, DeepSeekV4FlashPrefillSwaRingMatchesShippedShape) {
  ModelArgs model_args;
  // compress_ratios from DeepSeek-V4-Flash-0731-w8a8/config.json (46 entries;
  // only the 43 real layers are read).
  model_args.model_type("deepseek_v4")
      .n_layers(43)
      .head_dim(512)
      .index_head_dim(128)
      .index_n_heads(64)
      .window_size(128)
      .compress_ratios({0, 0,   4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                        4, 128, 4, 128, 4, 128, 4, 0,   0, 0});

  KVCacheEstimateOptions options;
  options.dtype = torch::kBFloat16;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 64LL * 1024 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 16;
  options.max_tokens_per_batch = 16384;
  options.max_tokens_per_chunk_for_prefill = 4096;
  options.enable_chunked_prefill = true;
  options.enable_prefix_cache = true;
  options.enable_dp_fair_token_budget = true;
  options.dp_size = 4;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  // window 128 / block 128 -> one window block per sequence, 16 sequences;
  // 128 burst blocks for max_tokens_per_batch = 16384; 16 tail + 2 guard.
  EXPECT_EQ(capacity.swa_count(), 162);
  // C4 and C128 now use the same 2048-token logical publish unit, so the
  // scheduler owns the same number of typed blocks for both cache tensors.
  EXPECT_EQ(capacity.c4_count(), capacity.c128_count());

  // The regression produced 66 = 16 + ceil((16384 / 4) / 128) + 16 + 2.
  EXPECT_NE(capacity.swa_count(), 66);
}

TEST(KVCacheEstimationTest, DeepSeekV4PdDecodeUsesBatchTokenCapacity) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.instance_role = InstanceRole::DECODE;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 22);
}

TEST(KVCacheEstimationTest,
     DeepSeekV4PdDecodeSwaCapacityIgnoresScheduleOverlap) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(128)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 1;
  options.max_tokens_per_batch = 385;
  options.num_speculative_tokens = 65;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::DECODE;

  options.enable_schedule_overlap = false;
  const KVCacheCapacity non_overlap_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(non_overlap_capacity.swa_count(), 8);

  options.enable_schedule_overlap = true;
  const KVCacheCapacity overlap_capacity =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(overlap_capacity.swa_count(), 8);
}

TEST(KVCacheEstimationTest, DeepSeekV4PrefixCacheKeepsOperationalSwaPool) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 128 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.max_tokens_per_chunk_for_prefill = 385;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;
  options.enable_prefix_cache = true;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  ASSERT_GT(capacity.c128_count(), 0);
  EXPECT_EQ(capacity.c4_count(), capacity.c128_count());
  EXPECT_EQ(capacity.swa_count(), 22);
}

TEST(KVCacheEstimationTest,
     DeepSeekV4RealisticMixBudgetRetainsCompressedPools) {
  std::vector<int32_t> compress_ratios{0, 0};
  compress_ratios.reserve(43);
  for (int32_t layer_id = 2; layer_id < 43; ++layer_id) {
    compress_ratios.emplace_back(layer_id % 2 == 0 ? 4 : 128);
  }

  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(43)
      .head_dim(512)
      .index_head_dim(128)
      .window_size(128)
      .max_seq_len(1048576)
      .compress_ratios(std::move(compress_ratios));

  KVCacheEstimateOptions options;
  options.dtype = torch::kBFloat16;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = int64_t{16} * 1024 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 10;
  options.max_tokens_per_batch = 10240;
  options.max_tokens_per_chunk_for_prefill = 2048;
  options.enable_prefix_cache = true;
  options.instance_role = InstanceRole::MIX;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 102);
  EXPECT_GT(capacity.c4_count(), 0);
  EXPECT_GT(capacity.c128_count(), 0);
  EXPECT_EQ(capacity.c4_count(), capacity.c128_count());
}

TEST(KVCacheEstimationTest, DeepSeekV4DecodeKeepsOperationalSwaPool) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 128 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::DECODE;
  options.enable_prefix_cache = true;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 22);
  EXPECT_GT(capacity.c128_count(), 0);
}

TEST(KVCacheEstimationTest, DeepSeekV4SwaOnlyPrefixKeepsOperationalSwaPool) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4_dspark")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 1, 1});

  constexpr int64_t kSwaBytesPerBlock = 3 * 128 * 16 * 4;
  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 43 * kSwaBytesPerBlock;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 385;
  options.max_tokens_per_chunk_for_prefill = 385;
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::PREFILL;
  options.enable_prefix_cache = true;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 22);
  EXPECT_EQ(capacity.c4_count(), 0);
  EXPECT_EQ(capacity.c128_count(), 0);
}

TEST(KVCacheEstimationTest, EstimatesDeepSeekV4DSparkSwaPool) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v4_dspark")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 1, 1});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 2818048;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 2176;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 35);
  EXPECT_EQ(capacity.c4_count(), 0);
  EXPECT_EQ(capacity.c128_count(), 0);
  EXPECT_EQ(capacity.n_blocks(), 1);
}

TEST(KVCacheEstimationTest,
     DeepSeekV4SpeculativeBudgetIncludesDSparkDraftSwaPool) {
  ModelArgs target_args;
  target_args.model_type("deepseek_v4")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 4, 128});
  ModelArgs draft_args;
  draft_args.model_type("deepseek_v4_dspark")
      .n_layers(3)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(257)
      .compress_ratios({1, 1, 1});

  KVCacheEstimateOptions target_options;
  target_options.dtype = torch::kFloat32;
  target_options.kv_cache_dtype = "auto";
  const DeepSeekV4CachePolicy cache_policy =
      get_dsv4_cache_policy(target_options.dtype);
  const int64_t scale_bytes =
      cache_policy.has_indexer_cache_scale ? cache_policy.scale_dtype_size : 0;
  const Dsv4CacheGeometry geometry;
  const int64_t c4_physical_dim = geometry.c4_physical_dim();
  const int64_t c128_physical_dim = geometry.c128_physical_dim();
  const int64_t c4_block_bytes =
      c4_physical_dim *
      (16 * 4 + 8 * cache_policy.index_dtype_size + scale_bytes);
  const int64_t c128_block_bytes = c128_physical_dim * 16 * 4;
  const int64_t compressed_unit_bytes = c4_block_bytes + c128_block_bytes;
  constexpr int64_t kTargetSwaBytes = 35 * 90112;
  constexpr int64_t kDraftSwaBytes =
      /*layers=*/3 * /*swa_count=*/35 * /*block_size=*/128 *
      /*head_dim=*/16 * /*float32_bytes=*/4;
  // Reserve both SWA pools and exactly two compressed units for either
  // the model-dtype indexer cache or the quantized indexer cache with scales.
  target_options.cache_size_in_bytes =
      kTargetSwaBytes + kDraftSwaBytes + 2 * compressed_unit_bytes;
  target_options.block_size = 128;
  target_options.max_seqs_per_batch = 4;
  target_options.max_tokens_per_batch = 2176;
  KVCacheEstimateOptions draft_options = target_options;
  draft_options.is_draft_engine = true;

  const KVCacheCapacity target_only_capacity =
      estimate_kv_cache_capacity(target_args, target_options);

  target_options.draft_model_args = &draft_args;
  target_options.draft_options = &draft_options;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(target_args, target_options);

  EXPECT_LE(capacity.cache_size_in_bytes() + kDraftSwaBytes,
            target_options.cache_size_in_bytes);
  EXPECT_EQ(capacity.swa_count(), 35);
  EXPECT_EQ(capacity.c4_count(), 2);
  EXPECT_EQ(capacity.c128_count(), 2);
  EXPECT_EQ(capacity.swa_count(), target_only_capacity.swa_count());
  EXPECT_GT(target_only_capacity.c4_count(), capacity.c4_count());
  EXPECT_EQ(target_only_capacity.c4_count(), target_only_capacity.c128_count());
  EXPECT_LT(capacity.n_blocks(), target_only_capacity.n_blocks());
}

TEST_P(DcpIndexerEstimationTest, IncludesAllIndexerBytes) {
  ModelArgs model_args = make_standard_args();
  model_args.index_n_heads(1).index_head_dim(16);
  KVCacheEstimateOptions options = make_estimate_options();
  options.indexer_cache_dtype = "auto";
  const KVCacheCapacity ordinary =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(ordinary.index_slot_size(), 32 * expected_factor());

  options.indexer_cache_dtype = "int8";
  const KVCacheCapacity quantized =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(quantized.index_slot_size(), 20 * expected_factor());
  EXPECT_EQ(quantized.slot_size(), ordinary.slot_size());

  model_args.index_n_heads(0);
  const KVCacheCapacity absent =
      estimate_kv_cache_capacity(model_args, options);
  EXPECT_EQ(absent.index_slot_size(), 0);
}

INSTANTIATE_TEST_SUITE_P(KvSplits,
                         DcpIndexerEstimationTest,
                         ::testing::Values(1, 2, 4, 0));

TEST(KVCacheEstimationTest, EstimatesDeepSeekV41Pools) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v41")
      .n_layers(6)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(128)
      .compress_ratios({0, 0, 2, 2, 1, 1})
      .kv_source_layer_ids({2, 4});

  // Cost model (see estimate_dsv41_kv_cache_cost): every layer pays
  // 128 * 16 * 4 = 8192 B of SWA window per block and the single ratio-2
  // kv source adds 2 * 128 * 16 * 4 = 16384 B of fp32 kv/score state, so
  // swa_bytes_per_block = 5 * 8192 + (8192 + 16384) = 65536 B. One TOKEN
  // block is 128 * (16 + 8) * 4 = 12288 B; a 2-engine-block unit holds one
  // c2 block plus two c1 blocks = 36864 B.
  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = /*swa=*/27 * 65536 + /*ten units=*/10 * 36864;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 2176;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_EQ(capacity.swa_count(), 27);
  EXPECT_EQ(capacity.c2_count(), 10);
  EXPECT_EQ(capacity.c1_count(), 20);
  EXPECT_EQ(capacity.c4_count(), 0);
  EXPECT_EQ(capacity.c128_count(), 0);
  EXPECT_EQ(capacity.n_blocks(), 20);
}

TEST(KVCacheEstimationTest, DeepSeekV41CountsOnlyKvSourceLayersForTokenGroups) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v41")
      .n_layers(6)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(128)
      .compress_ratios({0, 0, 2, 2, 1, 1})
      // Layer 3 (ratio 2) and layer 5 (ratio 1) are reuse layers: they must
      // not grow the TOKEN-group budget.
      .kv_source_layer_ids({2});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 16 * 1024 * 1024;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 2176;

  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(model_args, options);

  EXPECT_GT(capacity.c2_count(), 0);
  EXPECT_EQ(capacity.c1_count(), 0);
  EXPECT_EQ(capacity.c4_count(), 0);
  EXPECT_EQ(capacity.c128_count(), 0);
  EXPECT_EQ(capacity.n_blocks(), 2 * capacity.c2_count());
}

TEST(KVCacheEstimationTest, DeepSeekV41RejectsBudgetWithoutTokenCacheUnit) {
  ModelArgs model_args;
  model_args.model_type("deepseek_v41")
      .n_layers(6)
      .head_dim(16)
      .index_head_dim(8)
      .window_size(128)
      .compress_ratios({0, 0, 2, 2, 1, 1})
      .kv_source_layer_ids({2, 4});

  KVCacheEstimateOptions options;
  options.dtype = torch::kFloat32;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = /*swa=*/27 * 65536 + /*remaining_bytes=*/1;
  options.block_size = 128;
  options.max_seqs_per_batch = 4;
  options.max_tokens_per_batch = 2176;

  EXPECT_DEATH(
      estimate_kv_cache_capacity(model_args, options),
      "minimum DSV4 SWA cache leaves insufficient memory for one compressed "
      "cache unit");
}

}  // namespace xllm

namespace xllm {
TEST(KVCacheEstimationTest, CompressedPoolsUseExactBlockBytes) {
  // NPU selects this layout from model configuration, not the option alone.
  ModelArgs args;
  args.model_type("glm5_next")
      .n_layers(1)
      .head_dim(128)
      .index_n_heads(32)
      .index_head_dim(128)
      .index_kpool(3)
      .index_kpool_compress(true)
      .index_kpool_always_select_tail(true);
  KVCacheEstimateOptions options;
  options.dtype = torch::kBFloat16;
  options.kv_cache_dtype = "auto";
  options.cache_size_in_bytes = 1024 * 1024;
  options.block_size = 24;
  options.world_size = 1;
  options.n_local_kv_heads = 1;
  options.max_seqs_per_batch = 1;
  options.kpool_layout = KPoolCacheLayout::COMPRESSED_WITH_TAIL;
  const auto capacity = estimate_kv_cache_capacity(args, options);
  ASSERT_EQ(capacity.kpool_layout(), KPoolCacheLayout::COMPRESSED_WITH_TAIL);
  const auto actual = torch::empty({1, 1, 8, 128}, torch::kBFloat16);
  EXPECT_EQ(capacity.index_block_size(), actual.nbytes());
  const int64_t block_bytes = 24 * capacity.slot_size() + actual.nbytes();
  const int64_t available =
      capacity.cache_size_in_bytes() - capacity.linear_cache_size_in_bytes();
  EXPECT_EQ(capacity.n_blocks(), available / block_bytes);
}
}  // namespace xllm
