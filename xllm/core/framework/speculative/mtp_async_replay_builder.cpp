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

#include "core/framework/speculative/mtp_async_replay_builder.h"

#include <utility>

#include "runtime/forward_params.h"

namespace xllm::mtp_async {

DraftReplayInputPlan build_draft_replay_input_plan(
    const ForwardInput& base_input,
    const std::vector<EmbeddingCache::DecodeState>& states,
    const torch::Tensor& embedding_placeholder,
    int32_t logical_block_size,
    int32_t uniform_width,
    bool graph_warmup,
    int32_t pad_cache_slot) {
  const auto row_ctx = specBuilder::make_decode_row_context(base_input);
  auto replay = specBuilder::build_mtp_replay_inputs(row_ctx,
                                                     states,
                                                     embedding_placeholder,
                                                     logical_block_size,
                                                     uniform_width,
                                                     graph_warmup,
                                                     pad_cache_slot);

  DraftReplayInputPlan plan;
  plan.rows = std::move(replay.rows);
  plan.embeddings = std::move(replay.embeddings);
  plan.selected_rows = std::move(replay.selected_rows);
  plan.source_sequences = std::move(replay.source_sequences);
  plan.valid_rows = std::move(replay.valid_rows);
  plan.mrope_positions = std::move(replay.mrope_positions);
  plan.kpool_query_lens.assign(row_ctx.num_sequences, uniform_width);
  return plan;
}

}  // namespace xllm::mtp_async
