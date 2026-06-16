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

#pragma once

#include <cstdint>
#include <string>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"

namespace xllm::spec_feature_dump {

struct FeatureMetadata {
  std::string model;
  std::string point;
  int64_t rank = -1;
  int32_t layer = -1;
};

bool enabled();

bool should_dump_layer(const std::string& model, int32_t layer);

void dump_hidden(const FeatureMetadata& metadata,
                 const torch::Tensor& hidden_states,
                 const ModelInputParams& input_params);

void dump_kv(const FeatureMetadata& metadata,
             const KVCache& kv_cache,
             const ModelInputParams& input_params);

}  // namespace xllm::spec_feature_dump
