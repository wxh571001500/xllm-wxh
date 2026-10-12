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

#include "framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/config/scheduler_config.h"
#include "framework/kv_cache/kv_cache_capacity.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/model/model_args.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace {

TEST(HierarchyKVCacheTransferTest,
     RoundTripPublishesStrategyLayerReadyGranularity) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "An accelerator device is required for hierarchy KV transfer.";
  }

  constexpr int64_t kBlockCount = 2;
  constexpr int64_t kBlockSize = 4;
  constexpr int64_t kLayerCount = 4;
  constexpr int64_t kSourceBlockId = 0;
  constexpr int64_t kDestinationBlockId = 1;
  constexpr uint64_t kBatchId = 7;

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount).block_size(kBlockSize);
  ModelArgs model_args;
  model_args.model_type("test_model")
      .n_layers(kLayerCount)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(8);
  const KVCacheShape cache_shape(capacity, model_args, /*world_size=*/1);

  KVCacheCreateOptions create_options;
  create_options.device(device.unwrap())
      .dtype(torch::kFloat32)
      .num_layers(kLayerCount)
      .model_type("test_model");
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, cache_shape, create_options);
  ASSERT_EQ(caches.size(), static_cast<size_t>(kLayerCount));

  for (size_t layer_index = 0; layer_index < caches.size(); ++layer_index) {
    const double layer_value = static_cast<double>(layer_index);
    caches[layer_index].get_k_cache()[kSourceBlockId].fill_(3.0 + layer_value);
    caches[layer_index].get_v_cache()[kSourceBlockId].fill_(7.0 + layer_value);
    caches[layer_index].get_k_cache()[kDestinationBlockId].zero_();
    caches[layer_index].get_v_cache()[kDestinationBlockId].zero_();
  }
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.layers(kLayerCount)
      .host_blocks_factor(2.0)
      .layers_wise_copy_batchs(2);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  HierarchyKVCacheTransfer transfer(transfer_options,
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &caches,
                                    cache_shape,
                                    create_options);

  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ModelInputParams missing;
  missing.meta.batch_id = kBatchId;
  missing.meta.requires_host_restore = true;
  EXPECT_DEATH(transfer.set_layer_synchronizer(missing),
               "Missing Host restore handle");

  BlockTransferInfo offload_info(kSourceBlockId, /*dst_block_id=*/0);
  offload_info.block_type = BlockType::KV;
  offload_info.transfer_type = TransferType::D2H2G;
  EXPECT_EQ(transfer.transfer_kv_blocks(kBatchId, {offload_info}), 1U);

  BlockTransferInfo load_info(/*src_block_id=*/0, kDestinationBlockId);
  load_info.block_type = BlockType::KV;
  load_info.transfer_type = TransferType::H2D;
  EXPECT_EQ(transfer.transfer_kv_blocks(kBatchId, {load_info}), 1U);

  missing.meta.batch_id = kBatchId + 1;
  EXPECT_DEATH(transfer.set_layer_synchronizer(missing),
               "Missing Host restore handle");

  ModelInputParams input_params;
  input_params.meta.batch_id = kBatchId;
  input_params.meta.requires_host_restore = true;
  transfer.set_layer_synchronizer(input_params);
  ASSERT_NE(input_params.parallel.layer_wise_load_synchronizer, nullptr);
  EXPECT_FALSE(input_params.meta.requires_host_restore);
  const auto synchronizer = input_params.parallel.layer_wise_load_synchronizer;
  transfer.set_layer_synchronizer(input_params);
  EXPECT_EQ(input_params.parallel.layer_wise_load_synchronizer, synchronizer);
  EXPECT_FALSE(input_params.meta.requires_host_restore);
  missing.meta.batch_id = kBatchId;
  EXPECT_DEATH(transfer.set_layer_synchronizer(missing),
               "Missing Host restore handle");
  ModelInputParams next;
  next.meta.batch_id = kBatchId + 1;
  transfer.set_layer_synchronizer(next);
  EXPECT_EQ(next.parallel.layer_wise_load_synchronizer, nullptr);
  EXPECT_FALSE(next.meta.requires_host_restore);
  EXPECT_EQ(input_params.parallel.layer_wise_load_synchronizer->size(), 2U);
  EXPECT_EQ(input_params.parallel.layers_per_event, 2U);
  for (uint32_t layer_index = 0; layer_index < kLayerCount; ++layer_index) {
    ASSERT_TRUE(input_params.synchronize_layer(layer_index));
  }

  for (KVCache& cache : caches) {
    EXPECT_TRUE(torch::equal(cache.get_k_cache()[kSourceBlockId],
                             cache.get_k_cache()[kDestinationBlockId]));
    EXPECT_TRUE(torch::equal(cache.get_v_cache()[kSourceBlockId],
                             cache.get_v_cache()[kDestinationBlockId]));
  }
}

void verify_committed_linear_round_trip(const char* model_type) {
  constexpr int64_t kSlotCount = 3;
  constexpr int64_t kSourceSlot = 1;
  constexpr int64_t kDestinationSlot = 2;
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  KVCacheCapacity capacity;
  capacity.n_blocks(2)
      .block_size(4)
      .num_linear_state_blocks(kSlotCount)
      .linear_conv_state_len(5)
      .linear_ssm_checkpoint_stride(3);
  ModelArgs model_args;
  model_args.model_type(model_type)
      .n_layers(2)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(4)
      .full_attention_interval(2)
      .linear_conv_kernel_dim(4)
      .linear_num_key_heads(1)
      .linear_num_value_heads(1)
      .linear_key_head_dim(2)
      .linear_value_head_dim(2);
  const KVCacheShape cache_shape(capacity, model_args, /*world_size=*/1);
  KVCacheCreateOptions create_options;
  create_options.device(device.unwrap())
      .dtype(torch::kFloat32)
      .ssm_dtype(torch::kFloat32)
      .block_size(4)
      .num_layers(2)
      .full_attention_interval(2)
      .enable_linear_attention(true)
      .model_type(model_type);
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, cache_shape, create_options);
  torch::Tensor conv = caches[0].get_conv_cache();
  torch::Tensor ssm = caches[0].get_ssm_cache();
  conv[kSourceSlot].copy_(
      torch::arange(conv[kSourceSlot].numel(), conv.options())
          .view_as(conv[kSourceSlot]));
  ssm.narrow(0, kSourceSlot * 3, 3)
      .copy_(torch::arange(ssm[0].numel() * 3, ssm.options())
                 .view_as(ssm.narrow(0, kSourceSlot * 3, 3)));
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.layers(2).host_blocks_factor(2.0).layers_wise_copy_batchs(1);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  const int32_t old_chunk_size =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(8);
  HierarchyKVCacheTransfer transfer(transfer_options,
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &caches,
                                    cache_shape,
                                    create_options);
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(
      old_chunk_size);
  BlockTransferInfo offload(kSourceSlot, /*dst_block_id=*/0);
  offload.block_type = BlockType::LINEAR;
  offload.transfer_type = TransferType::D2H2G;
  ASSERT_EQ(transfer.transfer_kv_blocks(/*batch_id=*/9, {offload}), 1U);

  conv[kDestinationSlot].fill_(-1.0);
  ssm.narrow(0, kDestinationSlot * 3, 3).fill_(-1.0);
  ASSERT_EQ(device.synchronize_default_stream(), 0);
  BlockTransferInfo load(/*src_block_id=*/0, kDestinationSlot);
  load.block_type = BlockType::LINEAR;
  load.transfer_type = TransferType::H2D;
  ASSERT_EQ(transfer.transfer_kv_blocks(/*batch_id=*/9, {load}), 1U);
  ModelInputParams input_params;
  input_params.meta.batch_id = 9;
  input_params.meta.requires_host_restore = true;
  transfer.set_layer_synchronizer(input_params);
  ASSERT_TRUE(input_params.synchronize_all_layers());
  EXPECT_FALSE(transfer.take_load_handle(/*batch_id=*/9).has_value());

#if defined(USE_MUSA)
  constexpr int64_t kConvHistoryAxis = 1;
#else
  constexpr int64_t kConvHistoryAxis = 0;
#endif
  const torch::Tensor destination_conv = conv[kDestinationSlot];
  const torch::Tensor source_conv = conv[kSourceSlot];
  EXPECT_TRUE(torch::equal(destination_conv.narrow(kConvHistoryAxis, 0, 3),
                           source_conv.narrow(kConvHistoryAxis, 0, 3)));
  EXPECT_TRUE(torch::all(destination_conv.narrow(kConvHistoryAxis, 3, 2) == -1)
                  .item<bool>());
  EXPECT_TRUE(torch::equal(ssm[kDestinationSlot * 3], ssm[kSourceSlot * 3]));
  EXPECT_TRUE(torch::all(ssm.narrow(0, kDestinationSlot * 3 + 1, 2) == -1)
                  .item<bool>());
}

TEST(HierarchyKVCacheTransferTest,
     LinearRoundTripCopiesOnlyCommittedCheckpoint) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "An accelerator device is required for hierarchy KV transfer.";
  }

  const int32_t original_chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() = 4;
  for (const char* model_type : {"qwen3_5_text", "glm5_next"}) {
    SCOPED_TRACE(model_type);
    verify_committed_linear_round_trip(model_type);
  }
  SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill() =
      original_chunk_stride;
}

TEST(HierarchyKVCacheTransferTest, RejectsMixedOffloadBatchBeforeSubmission) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "An accelerator device is required for hierarchy KV transfer.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  KVCacheCapacity capacity;
  capacity.n_blocks(2).block_size(4);
  ModelArgs model_args;
  model_args.model_type("test_model")
      .n_layers(1)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(8);
  const KVCacheShape cache_shape(capacity, model_args, /*world_size=*/1);
  KVCacheCreateOptions create_options;
  create_options.device(device.unwrap())
      .dtype(torch::kFloat32)
      .num_layers(1)
      .model_type("test_model");
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, cache_shape, create_options);

  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.layers(1).host_blocks_factor(2.0).layers_wise_copy_batchs(1);
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  HierarchyKVCacheTransfer transfer(transfer_options,
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &caches,
                                    cache_shape,
                                    create_options);

  BlockTransferInfo d2h_info(/*src_block_id=*/0, /*dst_block_id=*/0);
  d2h_info.block_type = BlockType::KV;
  d2h_info.transfer_type = TransferType::D2H2G;
  BlockTransferInfo h2d_info(/*src_block_id=*/0, /*dst_block_id=*/1);
  h2d_info.block_type = BlockType::KV;
  h2d_info.transfer_type = TransferType::H2D;

  EXPECT_DEATH(
      {
        (void)transfer.transfer_kv_blocks(/*batch_id=*/7, {d2h_info, h2d_info});
      },
      "mixed transfer types");
}

}  // namespace
}  // namespace xllm
