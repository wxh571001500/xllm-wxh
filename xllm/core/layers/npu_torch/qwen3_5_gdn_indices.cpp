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

#include "core/layers/npu_torch/qwen3_5_gdn_indices.h"

#include <glog/logging.h>

#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>

#include "core/common/constants.h"
#include "core/layers/common/attention_metadata.h"

namespace xllm::layer::qwen3_5_gdn_internal {

void check_live_slots(const std::vector<int32_t>& live_slots,
                      int64_t batch_size,
                      int64_t num_slots) {
  CHECK_EQ(static_cast<int64_t>(live_slots.size()), batch_size)
      << "linear_state_ids must be sequence-scoped.";
  CHECK_GT(num_slots, 0) << "GDN cache must contain at least one slot.";
  std::unordered_set<int32_t> unique_slots;
  unique_slots.reserve(live_slots.size());
  for (const int32_t slot : live_slots) {
    CHECK_GE(slot, 0) << "GDN live slot must be non-negative.";
    CHECK_LT(static_cast<int64_t>(slot), num_slots)
        << "GDN live slot exceeds cache capacity.";
    if (slot != kPaddingLinearStateId) {
      CHECK(unique_slots.emplace(slot).second)
          << "GDN write slots must be unique within a batch, duplicate="
          << slot;
    }
  }
}

torch::Tensor slots_to_device(const std::vector<int32_t>& slots,
                              const torch::Device& device) {
  return torch::tensor(slots, torch::TensorOptions().dtype(torch::kInt))
      .to(device);
}

const MegaGdnPrefillIndicesCache& get_or_build_prefill_indices(
    const AttentionMetadata& attn_metadata,
    const std::vector<int32_t>& live_slots,
    const std::vector<int64_t>& validity_mask,
    const std::vector<int32_t>& read_slots,
    int64_t checkpoint_stride,
    int64_t num_slots,
    const torch::Device& device) {
  const int64_t batch_size = static_cast<int64_t>(live_slots.size());
  check_live_slots(live_slots, batch_size, num_slots);
  CHECK_EQ(static_cast<int64_t>(validity_mask.size()), batch_size)
      << "linear_state_validity_mask must be sequence-scoped.";
  CHECK(read_slots.empty() ||
        static_cast<int64_t>(read_slots.size()) == batch_size)
      << "linear_state_read_ids must be empty or sequence-scoped.";
  const auto& read_ids = read_slots.empty() ? live_slots : read_slots;
  for (size_t row = 0; row < read_ids.size(); ++row) {
    CHECK_GE(read_ids[row], kPaddingLinearStateId);
    CHECK_LT(static_cast<int64_t>(read_ids[row]), num_slots)
        << "linear-state source exceeds cache capacity.";
    CHECK((read_ids[row] == kPaddingLinearStateId) ==
          (live_slots[row] == kPaddingLinearStateId))
        << "padding must not be used as a real linear-state source or target";
    CHECK(read_ids[row] == live_slots[row] || validity_mask[row] == 1)
        << "linear-state direct-read row must be warm.";
  }
  CHECK_GT(checkpoint_stride, 0) << "checkpoint stride must be positive.";
  CHECK_LE(checkpoint_stride,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "checkpoint stride does not fit the operator int32 ABI.";
  CHECK_LE(num_slots - 1,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()) /
               checkpoint_stride)
      << "SSM state index does not fit the operator int32 ABI.";
  for (const int64_t validity : validity_mask) {
    CHECK(validity == 0 || validity == 1)
        << "linear state validity must be 0 or 1.";
  }

  std::optional<MegaGdnPrefillIndicesCache>& cache =
      attn_metadata.mega_gdn_prefill_indices;
  if (cache.has_value()) {
    const MegaGdnPrefillIndicesKey& key = cache->key;
    CHECK_EQ(key.device, device)
        << "MegaGdn Prefill indices cache device changed within one forward.";
    CHECK_EQ(key.batch_size, batch_size)
        << "MegaGdn Prefill indices cache batch size changed within one "
           "forward.";
    CHECK_EQ(key.num_slots, num_slots)
        << "MegaGdn Prefill cache slot geometry changed within one forward.";
    CHECK_EQ(key.checkpoint_stride, checkpoint_stride)
        << "MegaGdn Prefill checkpoint stride changed within one forward.";
    CHECK(key.linear_state_ids == live_slots)
        << "MegaGdn Prefill linear state ids changed within one forward.";
    CHECK(key.linear_state_validity_mask == validity_mask)
        << "MegaGdn Prefill validity changed within one forward.";
    CHECK(key.linear_state_read_ids == read_ids)
        << "MegaGdn Prefill read state ids changed within one forward.";
  }

  if (cache.has_value()) {
    return cache.value();
  }

  std::vector<int32_t> conv_read;
  std::vector<int32_t> conv_write;
  std::vector<int32_t> ssm_read;
  std::vector<int32_t> ssm_write;
  conv_read.reserve(batch_size);
  conv_write.reserve(batch_size);
  ssm_read.reserve(batch_size);
  ssm_write.reserve(batch_size);
  const int32_t stride = static_cast<int32_t>(checkpoint_stride);
  for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    const int32_t live_slot = live_slots[batch_index];
    const int32_t read_slot =
        validity_mask[batch_index] == 0 ? -1 : read_ids[batch_index];
    conv_read.emplace_back(read_slot);
    conv_write.emplace_back(live_slot);
    ssm_read.emplace_back(read_slot < 0 ? -1 : read_slot * stride);
    ssm_write.emplace_back(live_slot * stride);
  }

  int64_t device_tensor_materializations = 0;
  const auto materialize = [&](const std::vector<int32_t>& slots) {
    torch::Tensor tensor = slots_to_device(slots, device);
    ++device_tensor_materializations;
    return tensor;
  };
  MegaGdnPrefillIndicesCache built{
      .conv_read = materialize(conv_read),
      .conv_write = materialize(conv_write),
      .ssm_read = materialize(ssm_read),
      .ssm_write = materialize(ssm_write),
      .key = {.device = device,
              .batch_size = batch_size,
              .num_slots = num_slots,
              .checkpoint_stride = checkpoint_stride,
              .linear_state_ids = live_slots,
              .linear_state_validity_mask = validity_mask,
              .linear_state_read_ids = read_ids},
      .device_tensor_materializations = device_tensor_materializations};
  cache.emplace(std::move(built));
  return cache.value();
}

}  // namespace xllm::layer::qwen3_5_gdn_internal
