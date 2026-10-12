/* Copyright 2026 The xLLM Authors.

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

#include "processors/qwen3_omni_image_processor.h"

#include <cstdint>

#include "core/framework/config/model_config.h"
#include "processors/qwen2_vl_image_processor.h"

namespace xllm {

namespace {

constexpr int32_t kQwen3OmniMinimumImageTokens = 4;

}  // namespace

Qwen3OmniImageProcessor::Qwen3OmniImageProcessor(const ModelArgs& args)
    : Qwen2VLImageProcessor(args) {
  const int32_t max_tokens =
      ::xllm::ModelConfig::get_instance().image_max_tokens_num();
  if (max_tokens > 0) {
    CHECK_GE(max_tokens, kQwen3OmniMinimumImageTokens)
        << "image_max_tokens_num must be at least "
        << kQwen3OmniMinimumImageTokens;
    const int32_t factor = patch_size_ * merge_size_;
    min_pixels_ = kQwen3OmniMinimumImageTokens * factor * factor;
    max_pixels_ = max_tokens * factor * factor;
    LOG(INFO) << "Qwen3-Omni image preprocessing uses max_tokens=" << max_tokens
              << ", max_pixels=" << max_pixels_;
  }
}

}  // namespace xllm
