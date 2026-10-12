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

#include "framework/kv_cache_transfer/host_transfer/basic_transfer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache_transfer/host_transfer/compact_transfer.h"
#include "framework/kv_cache_transfer/host_transfer/layout.h"
#include "framework/kv_cache_transfer/host_transfer/transfer.h"
#include "platform/device.h"
#include "platform/layer_synchronizer.h"
#include "platform/platform.h"

namespace xllm {
namespace {

HostKVLayout make_layout(const Device& device, int64_t num_layers) {
  const torch::TensorOptions host_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  const torch::TensorOptions device_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap());
  HostKVGroupLayout group;
  group.group_id = 9;
  group.host_roles.emplace(KVCacheTensorRole::KEY,
                           torch::zeros({1, num_layers, 1}, host_options));
  group.layers.reserve(num_layers);
  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    group.layers.emplace_back(HostKVLayerLayout{
        layer_id,
        layer_id,
        {{KVCacheTensorRole::KEY, torch::zeros({1, 1}, device_options)}}});
  }
  return HostKVLayout(num_layers, {std::move(group)}, device.unwrap());
}

TEST(BasicHostKVTransferTest, KeepsLayerBatchingEdgeSemantics) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator device is required for Host KV transfer.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  std::unique_ptr<Stream> compute_stream = device.current_stream();

  BasicHostKVTransfer single_window(make_layout(device, /*num_layers=*/4),
                                    device,
                                    *compute_stream,
                                    /*layer_copy_batches=*/0);
  HostKVLoadHandle single_handle = single_window.prepare_load();
  EXPECT_EQ(single_handle.synchronizer->size(), 1U);
  EXPECT_EQ(single_handle.layers_per_event, 4U);
  single_window.drain();

  BasicHostKVTransfer per_layer(make_layout(device, /*num_layers=*/4),
                                device,
                                *compute_stream,
                                /*layer_copy_batches=*/8);
  HostKVLoadHandle per_layer_handle = per_layer.prepare_load();
  EXPECT_EQ(per_layer_handle.synchronizer->size(), 4U);
  EXPECT_EQ(per_layer_handle.layers_per_event, 1U);
  per_layer.drain();

  BasicHostKVTransfer composite(make_layout(device, /*num_layers=*/4),
                                device,
                                *compute_stream,
                                /*layer_copy_batches=*/4);
  HostKVLoadHandle composite_handle = composite.prepare_load(/*draft=*/true);
  EXPECT_EQ(composite_handle.synchronizer->size(), 5U);
  composite.drain();
}

TEST(HostKVTransferFactoryTest, SelectsConfiguredStrategy) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator device is required for Host KV transfer.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  std::unique_ptr<Stream> compute_stream = device.current_stream();

  HostKVTransferConfig config;
  std::unique_ptr<HostKVTransfer> automatic = create_host_kv_transfer(
      make_layout(device, /*num_layers=*/1), device, *compute_stream, config);
  if (Platform::supports_compact_host_kv_transfer()) {
    EXPECT_NE(dynamic_cast<CompactHostKVTransfer*>(automatic.get()), nullptr);
  } else {
    EXPECT_NE(dynamic_cast<BasicHostKVTransfer*>(automatic.get()), nullptr);
  }
  automatic->drain();

  config.mode = HostKVTransferMode::BASIC;
  std::unique_ptr<HostKVTransfer> basic = create_host_kv_transfer(
      make_layout(device, /*num_layers=*/1), device, *compute_stream, config);
  EXPECT_NE(dynamic_cast<BasicHostKVTransfer*>(basic.get()), nullptr);
  basic->drain();
}

TEST(BasicHostKVTransferTest, RoundTripUsesConfiguredLayerEventGroups) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator device is required for Host KV transfer.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const torch::TensorOptions host_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  const torch::TensorOptions device_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap());
  torch::Tensor host_key = torch::zeros({2, 4, 2}, host_options);

  HostKVGroupLayout group;
  group.group_id = 5;
  group.host_roles.emplace(KVCacheTensorRole::KEY, host_key);
  group.layers.reserve(4);
  std::vector<torch::Tensor> device_layers;
  device_layers.reserve(4);
  for (int64_t layer_index = 0; layer_index < 4; ++layer_index) {
    torch::Tensor blocks = torch::zeros({2, 2}, device_options);
    blocks[0].fill_(10.0 + static_cast<double>(layer_index));
    device_layers.emplace_back(blocks);
    group.layers.emplace_back(
        HostKVLayerLayout{layer_index,
                          layer_index,
                          {{KVCacheTensorRole::KEY, std::move(blocks)}}});
  }
  HostKVLayout layout(/*num_layers=*/4, {std::move(group)}, device.unwrap());
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  BasicHostKVTransfer transfer(std::move(layout),
                               device,
                               *compute_stream,
                               /*layer_copy_batches=*/2);
  ASSERT_EQ(compute_stream->synchronize(), 0);

  const HostKVRequest offload_request{{HostKVMapping{5, 0, 0}}};
  ASSERT_TRUE(transfer.offload(offload_request));
  const HostKVRequest load_request{{HostKVMapping{5, 0, 1}}};
  HostKVLoadHandle handle = transfer.prepare_load();
  ASSERT_NE(handle.synchronizer, nullptr);
  EXPECT_EQ(handle.synchronizer->size(), 2U);
  EXPECT_EQ(handle.layers_per_event, 2U);
  ASSERT_TRUE(transfer.load(load_request, handle));
  ASSERT_TRUE(handle.synchronizer->synchronize_layer(/*layer_index=*/0));
  ASSERT_TRUE(handle.synchronizer->synchronize_layer(/*layer_index=*/1));

  for (const torch::Tensor& blocks : device_layers) {
    EXPECT_TRUE(torch::equal(blocks[0], blocks[1]));
  }
  transfer.drain();
  transfer.drain();
}

TEST(BasicHostKVTransferTest, OffloadAndRestoreLinearCheckpointRows) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator device is required for Host KV transfer.";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const torch::TensorOptions host_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  const torch::TensorOptions device_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap());
  constexpr int64_t kBlockCount = 4;
  constexpr int64_t kCheckpointStride = 4;
  constexpr int64_t kConvHistoryRows = 6;
  torch::Tensor conv_cache =
      torch::empty({kBlockCount, kConvHistoryRows, 2}, device_options);
  torch::Tensor ssm_cache =
      torch::empty({kBlockCount * kCheckpointStride, 2}, device_options);
  for (int64_t block_id = 0; block_id < kBlockCount; ++block_id) {
    for (int64_t row = 0; row < kConvHistoryRows; ++row) {
      conv_cache[block_id][row].fill_(
          static_cast<double>(block_id * 100 + row * 10));
    }
    for (int64_t checkpoint = 0; checkpoint < kCheckpointStride; ++checkpoint) {
      ssm_cache[block_id * kCheckpointStride + checkpoint].fill_(
          static_cast<double>(block_id * 1000 + checkpoint * 100));
    }
  }
  KVCache device_cache(LinearAttentionKVCacheTensors{conv_cache, ssm_cache});

  torch::Tensor host_conv = torch::zeros({kBlockCount, 1, 3, 2}, host_options);
  torch::Tensor host_ssm = torch::zeros({kBlockCount, 1, 1, 2}, host_options);
  HostKVGroupLayout group;
  group.group_id = 11;
  group.host_roles.emplace(KVCacheTensorRole::CONV, host_conv);
  group.host_roles.emplace(KVCacheTensorRole::SSM, host_ssm);
  HostKVLayerLayout layer;
  layer.absolute_layer_id = 0;
  layer.group_layer_slot = 0;
  layer.device_roles = device_cache.get_block_type_tensors(
      BlockType::LINEAR, /*checkpoint_row=*/0);
  layer.device_cache = &device_cache;
  layer.block_type = BlockType::LINEAR;
  group.layers.emplace_back(std::move(layer));

  HostKVLayout layout(/*num_layers=*/1, {std::move(group)}, device.unwrap());
  std::unique_ptr<Stream> compute_stream = device.current_stream();
  BasicHostKVTransfer transfer(std::move(layout),
                               device,
                               *compute_stream,
                               /*layer_copy_batches=*/1);
  const HostKVRequest request{{HostKVMapping{11, 0, 0, 0},
                               HostKVMapping{11, 1, 1, 1},
                               HostKVMapping{11, 2, 2, 2},
                               HostKVMapping{11, 3, 3, 3}}};
  ASSERT_TRUE(transfer.offload(request));

  EXPECT_TRUE(torch::equal(host_conv[0][0],
                           conv_cache[0]
                               .narrow(/*dim=*/0,
                                       /*start=*/0,
                                       /*length=*/3)
                               .to(torch::kCPU)));
  EXPECT_TRUE(torch::equal(host_conv[1][0],
                           conv_cache[1]
                               .narrow(/*dim=*/0,
                                       /*start=*/1,
                                       /*length=*/3)
                               .to(torch::kCPU)));
  EXPECT_TRUE(torch::equal(host_conv[2][0],
                           conv_cache[2]
                               .narrow(/*dim=*/0,
                                       /*start=*/2,
                                       /*length=*/3)
                               .to(torch::kCPU)));
  EXPECT_TRUE(torch::equal(host_conv[3][0],
                           conv_cache[3]
                               .narrow(/*dim=*/0,
                                       /*start=*/3,
                                       /*length=*/3)
                               .to(torch::kCPU)));
  EXPECT_TRUE(torch::equal(host_ssm[0][0][0], ssm_cache[0].to(torch::kCPU)));
  EXPECT_TRUE(torch::equal(host_ssm[1][0][0],
                           ssm_cache[kCheckpointStride + 1].to(torch::kCPU)));
  EXPECT_TRUE(torch::equal(
      host_ssm[2][0][0], ssm_cache[2 * kCheckpointStride + 2].to(torch::kCPU)));
  EXPECT_TRUE(torch::equal(
      host_ssm[3][0][0], ssm_cache[3 * kCheckpointStride + 3].to(torch::kCPU)));

  conv_cache[0].narrow(/*dim=*/0, /*start=*/0, /*length=*/3).zero_();
  ssm_cache[0].zero_();
  ASSERT_EQ(compute_stream->synchronize(), 0);
  const HostKVRequest restore_request{{HostKVMapping{11, 2, 0, 0}}};
  HostKVLoadHandle restore_handle = transfer.prepare_load();
  ASSERT_NE(restore_handle.synchronizer, nullptr);
  ASSERT_TRUE(transfer.load(restore_request, restore_handle));
  ASSERT_TRUE(
      restore_handle.synchronizer->synchronize_layer(/*layer_index=*/0));
  EXPECT_TRUE(torch::equal(conv_cache[0]
                               .narrow(/*dim=*/0, /*start=*/0, /*length=*/3)
                               .to(torch::kCPU),
                           host_conv[2][0]));
  EXPECT_TRUE(torch::equal(ssm_cache[0].to(torch::kCPU), host_ssm[2][0][0]));
  transfer.drain();
}

}  // namespace
}  // namespace xllm
