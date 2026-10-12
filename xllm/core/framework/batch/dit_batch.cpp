/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "dit_batch.h"

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "core/framework/config/dit_config.h"

namespace {

bool check_tensors_valid(const std::vector<torch::Tensor>& vec) {
  CHECK(!vec.empty());

  torch::Tensor ref_tensor = vec[0];
  if (!ref_tensor.defined()) return false;

  if (vec.size() == 1) return true;

  const auto ref_shape = ref_tensor.sizes();
  for (size_t i = 1; i < vec.size(); ++i) {
    if (!vec[i].defined()) return false;

    if (vec[i].sizes() != ref_shape) {
      return false;
    }
  }

  return true;
}

torch::Tensor batch_tensors(const std::vector<torch::Tensor>& tensors) {
  CHECK(check_tensors_valid(tensors));
  if (tensors.size() == 1) {
    return tensors.front().unsqueeze(0);
  }
  return torch::stack(tensors);
}

}  // namespace

namespace xllm {

DiTForwardInput DiTBatch::prepare_forward_input() {
  CHECK(!request_vec_.empty());
  if (::xllm::DiTConfig::get_instance().dit_debug_print()) {
    LOG(INFO) << "DiT batch_size=" << request_vec_.size();
  }
  if (request_vec_[0]->state().request_kind() == DiTRequestKind::kText) {
    CHECK_EQ(request_vec_.size(), 1U)
        << "Cola-DLM text generation supports batch_size=1 only.";
  }

  DiTForwardInput input;
  input.batch_size = request_vec_.size();
  input.generation_params = request_vec_[0]->state().generation_params();

  const size_t batch_size = request_vec_.size();
  for (const auto& request : request_vec_) {
    const auto& generation_params = request->state().generation_params();
    CHECK(input.generation_params == generation_params)
        << "DiT generation params must be equal in the same batch";

    const auto& input_params = request->state().input_params();
    if (!input_params.prompt.empty())
      input.prompts.emplace_back(input_params.prompt);

    if (!input_params.prompt_2.empty())
      input.prompts_2.emplace_back(input_params.prompt_2);

    if (!input_params.negative_prompt.empty())
      input.negative_prompts.emplace_back(input_params.negative_prompt);

    if (!input_params.negative_prompt_2.empty())
      input.negative_prompts_2.emplace_back(input_params.negative_prompt_2);

    if (!input_params.audio_prompt_text.empty() &&
        input.audio_prompt_text.empty()) {
      input.audio_prompt_text = input_params.audio_prompt_text;
    }
  }

  if (input.prompts.size() != request_vec_.size()) {
    input.prompts.clear();
  }

  if (input.prompts_2.size() != request_vec_.size()) {
    input.prompts_2.clear();
  }

  const bool has_full_negative_prompts =
      input.negative_prompts.size() == request_vec_.size();
  if (!has_full_negative_prompts) {
    input.negative_prompts.clear();
  }

  if (input.negative_prompts_2.size() != request_vec_.size()) {
    input.negative_prompts_2.clear();
  }

  const DiTMediaSources& first_media_sources =
      request_vec_[0]->state().input_params().media_sources;
  for (size_t index = 0; index < first_media_sources.size(); ++index) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(batch_size);
    for (const auto& request : request_vec_) {
      tensors.emplace_back(
          request->state().input_params().media_sources.at(index).tensor);
    }
    const MediaNamedTensor& source = first_media_sources.at(index);
    input.media_sources.add(source.name,
                            source.modality,
                            batch_tensors(tensors),
                            source.parameters);
  }

  const DiTTensorSources& first_tensor_sources =
      request_vec_[0]->state().input_params().tensor_sources;
  for (const NamedTensor& tensor_input : first_tensor_sources.entries()) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(batch_size);
    for (const auto& request : request_vec_) {
      std::optional<torch::Tensor> tensor =
          request->state().input_params().tensor_sources.get(tensor_input.name);
      CHECK(tensor.has_value());
      tensors.emplace_back(*tensor);
    }
    input.tensor_sources.add(
        tensor_input.name, batch_tensors(tensors), tensor_input.parameters);
  }

  return input;
}

void DiTBatch::process_forward_output(const DiTForwardOutput& output) {
  // Text diffusion models produce text output directly.
  if (!output.text_output.empty()) {
    CHECK(request_vec_.size() == output.text_output.size());
    for (int32_t idx = 0; idx < static_cast<int32_t>(request_vec_.size());
         ++idx) {
      auto& request = request_vec_[idx];
      request->handle_forward_text_output(output.text_output[idx]);
    }
    return;
  }
  CHECK(request_vec_.size() == output.tensors.size());
  CHECK(output.audio_tensors.empty() ||
        output.audio_tensors.size() == output.tensors.size());
  for (int32_t idx = 0; idx < static_cast<int32_t>(request_vec_.size());
       ++idx) {
    auto& request = request_vec_[idx];
    torch::Tensor audio_output = output.audio_tensors.empty()
                                     ? torch::Tensor()
                                     : output.audio_tensors[idx];
    request->handle_forward_output(output.tensors[idx], audio_output);
  }
}

}  // namespace xllm
