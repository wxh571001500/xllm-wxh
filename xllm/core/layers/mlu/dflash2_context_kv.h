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

#include <memory>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/state_dict/state_dict.h"

namespace xllm::layer {

// Owns the context projection weights and their backend-specific cache layout.
// Callers supply normalized context rows and logical positions; TP replication,
// K normalization, RoPE and paged-cache scattering remain internal.
class DFlash2ContextKV final {
 public:
  explicit DFlash2ContextKV(const ModelContext& context);
  ~DFlash2ContextKV();

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;
  void finalize_loaded_weights();
  bool write(const torch::Tensor& hidden,
             const torch::Tensor& positions,
             const torch::Tensor& cache_slots,
             std::vector<KVCache>& kv_caches,
             const ModelInputParams& input_params) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xllm::layer
