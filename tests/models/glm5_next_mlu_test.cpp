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

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "models/llm/mlu/glm5_next.h"

namespace xllm::mlu::model {
namespace {

TEST(Glm5NextMluDeathTest, RejectsNonSpanVerificationBeforeUsingInputs) {
  const torch::Device device(torch::kCPU);
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, device);
  ParallelArgs parallel_args(/*rank=*/0, /*world_size=*/1, &group);
  parallel_args.tp_group_ = &group;
  ModelArgs args;
  args.model_type("glm5_next");
  args.n_layers(0);
  args.hidden_size(4);
  args.vocab_size(8);
  args.hc_mult(1);
  args.linear_num_key_heads(1);
  const ModelContext context(
      parallel_args,
      args,
      QuantArgs(),
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  Glm5NextModelImpl model(context);
  std::vector<KVCache> caches;

  // Undefined inputs must never reach embedding, metadata building, or layers.
  const torch::Tensor tokens;
  const torch::Tensor positions;
  ModelInputParams params;
  params.is_spec_verify = true;
  EXPECT_DEATH(model.forward(tokens, positions, caches, params),
               "requires chunked-prefill");

  params.is_spec_verify = false;
  params.attn_metadata = std::make_shared<layer::AttentionMetadata>();
  params.attn_metadata->is_spec_verify = true;
  EXPECT_DEATH(model.forward(tokens, positions, caches, params),
               "requires chunked-prefill");
}

}  // namespace
}  // namespace xllm::mlu::model
