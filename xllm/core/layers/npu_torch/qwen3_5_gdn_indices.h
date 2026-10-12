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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace xllm::layer {

struct AttentionMetadata;
struct MegaGdnPrefillIndicesCache;

namespace qwen3_5_gdn_internal {

void check_live_slots(const std::vector<int32_t>& live_slots,
                      int64_t batch_size,
                      int64_t num_slots);

torch::Tensor slots_to_device(const std::vector<int32_t>& slots,
                              const torch::Device& device);

const MegaGdnPrefillIndicesCache& get_or_build_prefill_indices(
    const AttentionMetadata& attn_metadata,
    const std::vector<int32_t>& live_slots,
    const std::vector<int64_t>& validity_mask,
    const std::vector<int32_t>& read_slots,
    int64_t checkpoint_stride,
    int64_t num_slots,
    const torch::Device& device);

}  // namespace qwen3_5_gdn_internal
}  // namespace xllm::layer
