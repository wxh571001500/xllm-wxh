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

#include <array>
#include <cstdint>
#include <vector>

#include "core/framework/speculative/embedding_cache.h"
#include "core/framework/speculative/spec_input_builder.h"

namespace xllm {

struct ForwardInput;

namespace mtp_async {

struct DraftReplayInputPlan final {
  specBuilder::DecodeBuildBuffers rows;
  std::vector<torch::Tensor> embeddings;
  std::vector<int32_t> selected_rows;
  std::vector<int32_t> source_sequences;
  std::vector<int32_t> valid_rows;
  std::vector<std::array<int32_t, 3>> mrope_positions;
  std::vector<int32_t> kpool_query_lens;
};

DraftReplayInputPlan build_draft_replay_input_plan(
    const ForwardInput& base_input,
    const std::vector<EmbeddingCache::DecodeState>& states,
    const torch::Tensor& embedding_placeholder,
    int32_t logical_block_size,
    int32_t uniform_width,
    bool graph_warmup,
    int32_t pad_cache_slot = 0);

}  // namespace mtp_async
}  // namespace xllm
