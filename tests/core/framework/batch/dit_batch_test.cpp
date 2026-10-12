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

#include "core/framework/batch/dit_batch.h"

#include <gtest/gtest.h>

#include <memory>

namespace xllm {
namespace {

TEST(DiTBatchTest, SingleRequestSourcesUseBatchViews) {
  DiTInputParams input_params;
  torch::Tensor image =
      torch::arange(12, torch::dtype(torch::kUInt8)).reshape({3, 2, 2});
  torch::Tensor prompt_embed = torch::randn({4, 8});
  torch::Tensor prompt_audio = torch::randn({1, 32});
  input_params.media_sources.add("image", "image", image);
  input_params.tensor_sources.add("prompt_embed", prompt_embed);
  input_params.tensor_sources.add("prompt_audio", prompt_audio);

  DiTGenerationParams generation_params;
  DiTOutputFunc output_func;
  DiTOutputsFunc outputs_func;
  DiTRequestState state(input_params,
                        generation_params,
                        output_func,
                        outputs_func,
                        DiTRequestKind::kImage);
  auto request = std::make_shared<DiTRequest>("request", "rid", "rtime", state);

  DiTBatch batch;
  batch.add(request);
  DiTForwardInput forward_input = batch.prepare_forward_input();

  const torch::Tensor& batched_image = forward_input.media_sources.at(0).tensor;
  const torch::Tensor batched_embed =
      *forward_input.tensor_sources.get("prompt_embed");
  const torch::Tensor batched_audio =
      *forward_input.tensor_sources.get("prompt_audio");

  EXPECT_EQ(batched_image.sizes(), torch::IntArrayRef({1, 3, 2, 2}));
  EXPECT_EQ(batched_embed.sizes(), torch::IntArrayRef({1, 4, 8}));
  EXPECT_EQ(batched_audio.sizes(), torch::IntArrayRef({1, 1, 32}));
  EXPECT_EQ(batched_image.data_ptr(), image.data_ptr());
  EXPECT_EQ(batched_embed.data_ptr(), prompt_embed.data_ptr());
  EXPECT_EQ(batched_audio.data_ptr(), prompt_audio.data_ptr());
}

}  // namespace
}  // namespace xllm
