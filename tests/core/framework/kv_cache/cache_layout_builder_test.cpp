/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "framework/kv_cache/cache_layout_builder.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "framework/kv_cache/kv_shard_layout.h"
#include "layers/common/kv_shard_batch_metadata.h"

namespace xllm {

namespace {

TEST(CacheLayoutBuilderTest, DescribesTokenMajorGqaHeads) {
  CacheTensorLayoutContext context;
  context.tp_rank = 1;
  context.tp_size = 3;
  context.block_token_capacity = 3;
  context.kv_head_count = 6;
  KVCacheTensor tensor{KVCacheTensorRole::KEY,
                       torch::zeros({2, 3, 2, 4}),
                       cache_group_id(BlockType::KV)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::SHARDED);
  EXPECT_EQ(descriptor.resource_scope, CacheResourceScope::BLOCK);
  ASSERT_EQ(descriptor.spans.size(), 2U);
  EXPECT_EQ(descriptor.spans[0].logical_offset_bytes, 2U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[1].logical_offset_bytes, 3U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 3U);
  EXPECT_EQ(descriptor.spans[0].logical_stride_bytes, 6U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].physical_stride_bytes, 2U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 1);
}

TEST(CacheLayoutBuilderTest, SelectsStableOwnerForMqaReplicas) {
  CacheTensorLayoutContext context;
  context.tp_rank = 3;
  context.tp_size = 4;
  context.block_token_capacity = 2;
  context.kv_head_count = 1;
  KVCacheTensor tensor{KVCacheTensorRole::VALUE,
                       torch::zeros({2, 2, 1, 8}),
                       cache_group_id(BlockType::KV)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
}

TEST(CacheLayoutBuilderTest, SelectsContiguousReplicaGroupForGqaHeads) {
  CacheTensorLayoutContext context;
  context.tp_rank = 2;
  context.tp_size = 4;
  context.block_token_capacity = 2;
  context.kv_head_count = 2;
  KVCacheTensor tensor{KVCacheTensorRole::VALUE,
                       torch::zeros({2, 2, 1, 8}),
                       cache_group_id(BlockType::KV)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].logical_offset_bytes, 8U * 4U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 2);
}

TEST(CacheLayoutBuilderTest, DescribesMlaAsWholeResourceReplica) {
  CacheTensorLayoutContext context;
  context.tp_rank = 2;
  context.tp_size = 4;
  context.enable_mla = true;
  KVCacheTensor tensor{KVCacheTensorRole::KEY,
                       torch::zeros({2, 3, 16}),
                       cache_group_id(BlockType::KV)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region,
            static_cast<uint64_t>(tensor.tensor.nbytes() / 2));
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
}

TEST(CacheLayoutBuilderTest, DescribesSharedIndexerKeyAsReplica) {
  CacheTensorLayoutContext context;
  context.tp_rank = 7;
  context.tp_size = 8;
  context.block_token_capacity = 3;
  context.index_head_count = 64;
  KVCacheTensor tensor{KVCacheTensorRole::INDEX,
                       torch::zeros({2, 3, 1, 4}),
                       cache_group_id(BlockType::C4)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 3U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
}

TEST(CacheLayoutBuilderTest, DescribesSharedIndexerScaleAsReplica) {
  CacheTensorLayoutContext context;
  context.tp_rank = 3;
  context.tp_size = 4;
  context.block_token_capacity = 3;
  context.index_head_count = 64;
  KVCacheTensor tensor{KVCacheTensorRole::INDEX_SCALE,
                       torch::zeros({2, 3, 1}),
                       cache_group_id(BlockType::C4)};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 4U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 3U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
}

TEST(CacheLayoutBuilderTest, DescribesSequenceScopedSsmHeads) {
  CacheTensorLayoutContext context;
  context.tp_rank = 1;
  context.tp_size = 2;
#if defined(USE_NPU)
  context.enable_mla = true;
#endif
  context.linear_value_head_count = 4;
  KVCacheTensor tensor{KVCacheTensorRole::SSM,
                       torch::zeros({2, 2, 3, 4}),
                       cache_group_id(BlockType::LINEAR),
                       /*sequence_scoped=*/true};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::SHARDED);
  EXPECT_EQ(descriptor.resource_scope, CacheResourceScope::SEQUENCE);
  ASSERT_EQ(descriptor.spans.size(), 2U);
  EXPECT_EQ(descriptor.spans[0].logical_offset_bytes, 2U * 3U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 3U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 1U);
}

TEST(CacheLayoutBuilderTest, DescribesCheckpointedSsmRowsPerLogicalSlot) {
  CacheTensorLayoutContext context;
  context.tp_rank = 1;
  context.tp_size = 2;
  context.linear_value_head_count = 4;
  context.linear_ssm_checkpoint_stride = 3;
  KVCacheTensor tensor{KVCacheTensorRole::SSM,
                       torch::zeros({6, 2, 3, 4}),
                       cache_group_id(BlockType::LINEAR),
                       /*sequence_scoped=*/true};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  ASSERT_EQ(descriptor.spans.size(), 2U);
  EXPECT_EQ(descriptor.spans[0].repeat_count, 3U);
  EXPECT_EQ(descriptor.spans[0].logical_stride_bytes, 4U * 3U * 4U * 4U);
  EXPECT_EQ(descriptor.spans[0].physical_stride_bytes, 2U * 3U * 4U * 4U);
}

TEST(CacheLayoutBuilderTest, DescribesCompositeConvState) {
  CacheTensorLayoutContext context;
  context.tp_rank = 1;
  context.tp_size = 2;
#if defined(USE_NPU)
  context.enable_mla = true;
#endif
  context.linear_key_head_count = 4;
  context.linear_value_head_count = 2;
  context.linear_key_head_dim = 3;
  KVCacheTensor tensor{KVCacheTensorRole::CONV,
                       torch::zeros({2, 5, 15}),
                       cache_group_id(BlockType::LINEAR),
                       /*sequence_scoped=*/true};
  std::string error;

  ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
  ASSERT_TRUE(tensor.shard_descriptor.has_value());
  const LogicalShardDescriptor& descriptor = *tensor.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::COMPOSITE);
  EXPECT_EQ(descriptor.resource_scope, CacheResourceScope::SEQUENCE);
  ASSERT_EQ(descriptor.spans.size(), 5U);
  EXPECT_EQ(descriptor.spans[0].logical_tensor, "conv_key_a");
  EXPECT_EQ(descriptor.spans[2].logical_tensor, "conv_key_b");
  EXPECT_EQ(descriptor.spans[4].logical_tensor, "conv_value");
  EXPECT_EQ(descriptor.spans[0].repeat_count, 5U);
}

TEST(CacheLayoutBuilderTest, HybridMlaKeepsKPoolTailSequenceScopedReplica) {
  CacheTensorLayoutContext context;
  context.enable_mla = true;
  context.tp_rank = 1;
  context.tp_size = 2;
  context.linear_value_head_count = 4;
  KVCacheTensor tail{KVCacheTensorRole::KPOOL_TAIL,
                     torch::zeros({3, 2, 8, 4}, torch::kBFloat16),
                     cache_group_id(BlockType::LINEAR),
                     /*sequence_scoped=*/true};
  std::string error;
  ASSERT_TRUE(describe_cache_tensor(context, &tail, &error)) << error;
  const auto& descriptor = *tail.shard_descriptor;
  EXPECT_EQ(descriptor.kind, LogicalShardKind::REPLICATED);
  EXPECT_EQ(descriptor.resource_scope, CacheResourceScope::SEQUENCE);
  ASSERT_EQ(descriptor.spans.size(), 1U);
  EXPECT_EQ(descriptor.spans[0].owner_tp_rank, 0);
  EXPECT_EQ(descriptor.spans[0].bytes_per_region, 2U * 8U * 4U * 2U);
}

TEST(KVShardLayoutTest, MapsTokensAndSlots) {
  const int64_t dcp4_lengths[] = {128, 128, 1, 0};
  for (int32_t rank = 0; rank < 4; ++rank) {
    EXPECT_EQ(KVShardLayout(128, 4, rank).local_token_count(257),
              dcp4_lengths[rank]);
  }
  EXPECT_EQ(KVShardLayout(128, 2, 0).local_token_count(300), 172);
  EXPECT_EQ(KVShardLayout(128, 2, 1).local_token_count(300), 128);

  const torch::Tensor slots =
      torch::tensor({-1, 0, 127, 128, 255, 256}, torch::kInt32);
  EXPECT_TRUE(torch::equal(
      layer::localize_kv_shard_slots(slots, KVShardLayout(128, 2, 0)),
      torch::tensor({-1, 0, 127, -1, -1, 128}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      layer::localize_kv_shard_slots(slots, KVShardLayout(128, 2, 1)),
      torch::tensor({-1, -1, -1, 0, 127, -1}, torch::kInt32)));
  EXPECT_EQ(layer::localize_kv_shard_slots(torch::empty({0}),
                                           KVShardLayout(128, 2, 0))
                .numel(),
            0);
  EXPECT_FALSE(
      layer::localize_kv_shard_slots(torch::Tensor(), KVShardLayout(128, 2, 0))
          .defined());
}

TEST(CacheLayoutBuilderTest, RecordsPerPoolSpansFromTensors) {
  const Dsv4CacheGeometry geometry;
  CacheTensorLayoutContext context;
  context.tp_rank = 0;
  context.tp_size = 1;
  context.block_token_capacity = 128;
  context.kv_head_count = 1;

  std::vector<KVCache> caches;
  DeepSeekV4KVCacheTensors swa_tensors;
  swa_tensors.swa_cache = torch::zeros({2, 128, 1, 8}, torch::kBFloat16);
  caches.emplace_back(swa_tensors);
  DeepSeekV4KVCacheTensors c4_tensors;
  c4_tensors.compressed_block_type = BlockType::C4;
  c4_tensors.key_cache =
      torch::zeros({4, geometry.c4_physical_dim(), 1, 8}, torch::kBFloat16);
  // INDEX gets a narrower token axis on purpose: compressed KPool layouts
  // shape it differently from the pool's K/V tensors, so recording it into
  // the per-pool span map would abort with a span conflict.
  c4_tensors.index_cache = torch::zeros({4, 128, 1, 8}, torch::kInt8);
  c4_tensors.swa_cache = torch::zeros({2, 128, 1, 8}, torch::kBFloat16);
  caches.emplace_back(c4_tensors);
  DeepSeekV4KVCacheTensors c128_tensors;
  c128_tensors.compressed_block_type = BlockType::C128;
  c128_tensors.key_cache =
      torch::zeros({8, geometry.c128_physical_dim(), 1, 8}, torch::kBFloat16);
  c128_tensors.swa_cache = torch::zeros({2, 128, 1, 8}, torch::kBFloat16);
  caches.emplace_back(c128_tensors);

  record_group_block_capacities(caches, &context);

  // Only the compressed K/V pools record a span: WINDOW, INDEX and the state
  // tensors do not define their pool's token axis, so the SWA group keeps the
  // global block_token_capacity fallback.
  ASSERT_EQ(context.group_block_capacities.size(), 2U);
  EXPECT_EQ(context.group_block_capacities.at(cache_group_id(BlockType::C4)),
            geometry.c4_physical_dim());
  EXPECT_EQ(context.group_block_capacities.at(cache_group_id(BlockType::C128)),
            geometry.c128_physical_dim());
  EXPECT_EQ(
      context.group_block_capacities.count(cache_group_id(BlockType::SWA)), 0U);
}

TEST(CacheLayoutBuilderTest, DescribesTypedPoolAgainstRecordedSpan) {
  const Dsv4CacheGeometry geometry;
  CacheTensorLayoutContext context;
  context.tp_rank = 0;
  context.tp_size = 1;
  context.block_token_capacity = 128;
  context.kv_head_count = 1;

  std::vector<KVCache> caches;
  DeepSeekV4KVCacheTensors c4_tensors;
  c4_tensors.compressed_block_type = BlockType::C4;
  c4_tensors.key_cache =
      torch::zeros({4, geometry.c4_physical_dim(), 1, 8}, torch::kBFloat16);
  DeepSeekV4KVCacheTensors c128_tensors;
  c128_tensors.compressed_block_type = BlockType::C128;
  c128_tensors.key_cache =
      torch::zeros({8, geometry.c128_physical_dim(), 1, 8}, torch::kBFloat16);
  caches.emplace_back(c4_tensors);
  caches.emplace_back(c128_tensors);
  record_group_block_capacities(caches, &context);

  // Each typed pool validates against its own recorded span instead of the
  // uniform scheduler block size.
  KVCacheTensor c4_key{
      KVCacheTensorRole::KEY,
      torch::zeros({4, geometry.c4_physical_dim(), 1, 8}, torch::kBFloat16),
      cache_group_id(BlockType::C4)};
  std::string error;
  ASSERT_TRUE(describe_cache_tensor(context, &c4_key, &error)) << error;

  KVCacheTensor c128_key{
      KVCacheTensorRole::KEY,
      torch::zeros({4, geometry.c128_physical_dim(), 1, 8}, torch::kBFloat16),
      cache_group_id(BlockType::C128)};
  error.clear();
  ASSERT_TRUE(describe_cache_tensor(context, &c128_key, &error)) << error;

  // Pools without a recorded span keep the global capacity.
  KVCacheTensor uniform_key{KVCacheTensorRole::KEY,
                            torch::zeros({4, 128, 1, 8}, torch::kBFloat16),
                            cache_group_id(BlockType::KV)};
  error.clear();
  EXPECT_TRUE(describe_cache_tensor(context, &uniform_key, &error)) << error;

  // A C4 tensor with the uniform capacity is rejected: the pool's recorded
  // span is the authority.
  KVCacheTensor mismatched_key{KVCacheTensorRole::KEY,
                               torch::zeros({4, 128, 1, 8}, torch::kBFloat16),
                               cache_group_id(BlockType::C4)};
  error.clear();
  EXPECT_FALSE(describe_cache_tensor(context, &mismatched_key, &error));
  EXPECT_NE(error.find("token capacity differs"), std::string::npos);
}

TEST(CacheLayoutBuilderTest, FailsWhenPoolSpansConflict) {
  CacheTensorLayoutContext context;
  context.tp_rank = 0;
  context.tp_size = 1;
  context.block_token_capacity = 16;
  context.kv_head_count = 1;

  // Both spans deviate from the uniform capacity: pools running at the
  // uniform capacity record nothing, so a conflict needs two distinct
  // deviating spans inside one pool.
  KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({4, 32, 1, 8}, torch::kBFloat16);
  tensors.value_cache = torch::zeros({4, 64, 1, 8}, torch::kBFloat16);
  std::vector<KVCache> caches;
  caches.emplace_back(tensors);

  EXPECT_DEATH(record_group_block_capacities(caches, &context),
               "Cache tensors of one pool must share the token-axis span");
}

TEST(CacheLayoutBuilderTest, SkipsSpanRecordingForMlaLayouts) {
  CacheTensorLayoutContext context;
  context.tp_rank = 0;
  context.tp_size = 1;
  context.block_token_capacity = 16;
  context.kv_head_count = 1;
  context.enable_mla = true;

  // MLA pools legitimately mix K/V token-axis spans: the NPU NZ layout
  // shapes the KEY axis after kv_lora_rank and the VALUE axis after
  // qk_rope_head_dim, and describe routes MLA tensors through the
  // replicated-tensor contract, which never reads a pool span.
  KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({4, 32, 16, 16}, torch::kBFloat16);
  tensors.value_cache = torch::zeros({4, 4, 16, 16}, torch::kBFloat16);
  std::vector<KVCache> caches;
  caches.emplace_back(tensors);

  record_group_block_capacities(caches, &context);
  EXPECT_EQ(context.group_block_capacities.size(), 0U);
}

TEST(CacheLayoutBuilderTest, SkipsSpanRecordingForUniformPools) {
  CacheTensorLayoutContext context;
  context.tp_rank = 0;
  context.tp_size = 1;
  context.block_token_capacity = 16;
  context.kv_head_count = 1;

  // Uniform pools run at the declared block capacity: recording their span
  // would make describe validate the tensor against its own size and mute
  // the producer-capacity check. They stay on the global fallback.
  KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({4, 16, 1, 8}, torch::kBFloat16);
  tensors.value_cache = torch::zeros({4, 16, 1, 8}, torch::kBFloat16);
  std::vector<KVCache> caches;
  caches.emplace_back(tensors);

  record_group_block_capacities(caches, &context);
  EXPECT_EQ(context.group_block_capacities.size(), 0U);

  // The producer-declared capacity stays the authority: a uniform-pool
  // tensor whose token axis diverges from it is rejected at describe
  // instead of silently redefining the pool's published span.
  KVCacheTensor diverging_key{KVCacheTensorRole::KEY,
                              torch::zeros({4, 32, 1, 8}, torch::kBFloat16),
                              cache_group_id(BlockType::KV)};
  std::string error;
  EXPECT_FALSE(describe_cache_tensor(context, &diverging_key, &error));
  EXPECT_NE(error.find("token capacity differs"), std::string::npos);
}

TEST(CacheLayoutBuilderTest, SkipsSpanRecordingForTensorsWithoutTokenAxis) {
  CacheTensorLayoutContext context;
  context.tp_rank = 0;
  context.tp_size = 1;
  context.block_token_capacity = 16;
  context.kv_head_count = 1;

  // A tensor without the token axis defines no pool span: recording it
  // would read a nonexistent axis. It is skipped here and left to
  // describe_cache_tensor, which rejects it with a precise error. The
  // value cache runs at a deviating span so recording still happens for
  // the well-formed tensor.
  KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({4}, torch::kBFloat16);
  tensors.value_cache = torch::zeros({4, 32, 1, 8}, torch::kBFloat16);
  std::vector<KVCache> caches;
  caches.emplace_back(tensors);

  record_group_block_capacities(caches, &context);

  ASSERT_EQ(context.group_block_capacities.size(), 1U);
  EXPECT_EQ(context.group_block_capacities.at(cache_group_id(BlockType::KV)),
            32);
}

}  // namespace

}  // namespace xllm
