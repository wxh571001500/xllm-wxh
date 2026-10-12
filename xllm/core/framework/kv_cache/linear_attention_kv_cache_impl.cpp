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

#include "framework/kv_cache/linear_attention_kv_cache_impl.h"

#include "core/framework/config/scheduler_config.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "util/tensor_helper.h"

namespace xllm {

LinearAttentionKVCacheImpl::LinearAttentionKVCacheImpl(
    const LinearAttentionKVCacheTensors& tensors)
    : conv_cache_(tensors.conv_cache),
      ssm_cache_(tensors.ssm_cache),
      conv_cache_shape_(get_tensor_shape(tensors.conv_cache)),
      ssm_cache_shape_(get_tensor_shape(tensors.ssm_cache)) {}

LinearAttentionKVCacheImpl::LinearAttentionKVCacheImpl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : LinearAttentionKVCacheImpl(
          create_linear_attention_kv_cache_tensors(kv_cache_shape,
                                                   create_options)) {
  conv_cache_shape_ = kv_cache_shape.conv_cache_shape();
  ssm_cache_shape_ = kv_cache_shape.ssm_cache_shape();
}

LinearAttentionKVCacheImpl::LinearAttentionKVCacheImpl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    BlockType type,
    int64_t layer_count)
    : KVCacheImpl() {
  CHECK(type == BlockType::LINEAR)
      << "LinearAttentionKVCacheImpl host cache only supports "
         "BlockType::LINEAR.";
  CHECK(kv_cache_shape.has_conv_cache_shape() &&
        kv_cache_shape.has_ssm_cache_shape())
      << "Host LINEAR cache requires both Conv and SSM tensors.";
  const std::vector<int64_t>& device_conv_shape =
      kv_cache_shape.conv_cache_shape();
  const std::vector<int64_t>& device_ssm_shape =
      kv_cache_shape.ssm_cache_shape();
  CHECK(kv_cache_shape.has_key_cache_shape())
      << "Host LINEAR cache capacity requires the KV cache shape.";
  const int64_t host_kv_block_count = scale_host_block_count(
      kv_cache_shape.key_cache_shape()[0], create_options.host_blocks_factor());
  const int64_t host_slot_count = linear_state_block_count(
      host_kv_block_count,
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      create_options.block_size());

  host_page_aligned_regions_.reserve(2);
  std::vector<int64_t> host_conv_shape = device_conv_shape;
  host_conv_shape[0] = host_slot_count;
#if defined(USE_MUSA)
  constexpr int64_t kConvHistoryAxis = 2;
#else
  constexpr int64_t kConvHistoryAxis = 1;
#endif
  constexpr int64_t kCommittedConvRows = 3;
  CHECK_GT(host_conv_shape.size(), kConvHistoryAxis);
  CHECK_GE(host_conv_shape[kConvHistoryAxis], kCommittedConvRows);
  host_conv_shape[kConvHistoryAxis] = kCommittedConvRows;
  host_conv_shape.insert(host_conv_shape.begin() + 1, layer_count);
  create_host_tensor(host_conv_shape,
                     create_options.dtype(),
                     &conv_cache_,
                     &conv_cache_shape_);

  std::vector<int64_t> host_ssm_shape = device_ssm_shape;
  host_ssm_shape[0] = 1;
  host_ssm_shape.insert(host_ssm_shape.begin(), layer_count);
  host_ssm_shape.insert(host_ssm_shape.begin(), host_slot_count);
  create_host_tensor(host_ssm_shape,
                     create_options.ssm_dtype(),
                     &ssm_cache_,
                     &ssm_cache_shape_);
}

torch::Tensor LinearAttentionKVCacheImpl::get_conv_cache() const {
  return conv_cache_;
}

torch::Tensor LinearAttentionKVCacheImpl::get_ssm_cache() const {
  return ssm_cache_;
}

BlockTypeTensorMap LinearAttentionKVCacheImpl::get_block_type_tensors(
    BlockType type) const {
  BlockTypeTensorMap tensor_map;
  if (type != BlockType::LINEAR) {
    return tensor_map;
  }
  if (conv_cache_.defined() && conv_cache_.numel() > 0) {
    tensor_map.emplace(KVCacheTensorRole::CONV, conv_cache_);
  }
  if (ssm_cache_.defined() && ssm_cache_.numel() > 0) {
    tensor_map.emplace(KVCacheTensorRole::SSM, ssm_cache_);
  }
  return tensor_map;
}

BlockTypeTensorMap LinearAttentionKVCacheImpl::get_block_type_tensors(
    BlockType type,
    size_t checkpoint_row) const {
  if (type != BlockType::LINEAR) {
    return get_block_type_tensors(type);
  }

  BlockTypeTensorMap tensor_map = get_block_type_tensors(type);
  auto conv_it = tensor_map.find(KVCacheTensorRole::CONV);
  auto ssm_it = tensor_map.find(KVCacheTensorRole::SSM);
  if (conv_it == tensor_map.end() || ssm_it == tensor_map.end()) {
    return tensor_map;
  }

  constexpr int64_t kCommittedConvRows = 3;
#if defined(USE_MUSA)
  constexpr int64_t kConvHistoryAxis = 2;
#else
  constexpr int64_t kConvHistoryAxis = 1;
#endif
  CHECK_GT(conv_it->second.dim(), kConvHistoryAxis);
  const int64_t conv_history_rows = conv_it->second.size(kConvHistoryAxis);
  CHECK_GE(conv_history_rows, kCommittedConvRows);
  CHECK_LE(checkpoint_row,
           static_cast<size_t>(conv_history_rows - kCommittedConvRows))
      << "LINEAR checkpoint row selects rows outside the device cache: "
      << "checkpoint_row=" << checkpoint_row;
  conv_it->second = conv_it->second.narrow(kConvHistoryAxis,
                                           static_cast<int64_t>(checkpoint_row),
                                           kCommittedConvRows);

  CHECK_GT(ssm_it->second.size(0), 0);
  CHECK_EQ(ssm_it->second.size(0) % conv_cache_.size(0), 0)
      << "SSM rows must be divisible by logical LINEAR slots.";
  const int64_t checkpoint_stride =
      ssm_it->second.size(0) / conv_cache_.size(0);
  CHECK_LT(checkpoint_row, static_cast<size_t>(checkpoint_stride))
      << "LINEAR checkpoint row exceeds the checkpoint stride: "
      << "checkpoint_row=" << checkpoint_row
      << ", checkpoint_stride=" << checkpoint_stride;
  std::vector<int64_t> logical_ssm_shape = ssm_it->second.sizes().vec();
  logical_ssm_shape[0] = checkpoint_stride;
  logical_ssm_shape.insert(logical_ssm_shape.begin(), conv_cache_.size(0));
  ssm_it->second = ssm_it->second.view(std::move(logical_ssm_shape))
                       .narrow(/*dim=*/1,
                               static_cast<int64_t>(checkpoint_row),
                               /*length=*/1);
  return tensor_map;
}

bool LinearAttentionKVCacheImpl::empty() const {
  return !conv_cache_.defined() || !ssm_cache_.defined();
}

std::vector<std::vector<int64_t>> LinearAttentionKVCacheImpl::get_shapes()
    const {
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(2);
  shapes.emplace_back(conv_cache_shape_);
  shapes.emplace_back(ssm_cache_shape_);
  return shapes;
}

}  // namespace xllm
