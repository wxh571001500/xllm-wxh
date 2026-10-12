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

#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <vector>

namespace xllm {

struct ModelInputParams;

void register_input_batch_metadata_view(pybind11::module_& module);

class PyInputBatchMetadataView final {
 public:
  explicit PyInputBatchMetadataView(const ModelInputParams& params);

  int32_t num_reqs() const;
  int64_t num_tokens() const;
  const std::vector<int32_t>& num_scheduled_tokens() const;
  const std::vector<int32_t>& num_computed_tokens() const;
  const std::vector<int32_t>& query_start_loc() const;
  const std::vector<uint8_t>& is_prefilling() const;

 private:
  int32_t num_reqs_ = 0;
  int64_t num_tokens_ = 0;
  std::vector<int32_t> num_scheduled_tokens_;
  std::vector<int32_t> num_computed_tokens_;
  std::vector<int32_t> query_start_loc_;
  std::vector<uint8_t> is_prefilling_;
};

}  // namespace xllm
