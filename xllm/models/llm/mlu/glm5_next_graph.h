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

#include "core/framework/model/causal_lm.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/mlu/glm5_next/glm5_next_kpool_indexer.h"

namespace xllm::mlu::model {

struct Glm5NextGraphMetadataState final : ModelGraphMetadataState {
  std::shared_ptr<layer::AttentionMetadata> metadata;
};

// Shared by target and MTP models; consumes graph-owned padded input storage.
class Glm5NextGraphMetadata final {
 public:
  static void prepare(ModelGraphMetadataState* state,
                      const torch::Tensor& positions,
                      ModelInputParams& params) {
    auto* graph_state = dynamic_cast<Glm5NextGraphMetadataState*>(state);
    CHECK(graph_state != nullptr) << "GLM5 requires per-graph metadata state";
    const auto& attention = params.attention.device;
    if (!graph_state->metadata) {
      // Materialize host-derived tensors outside device graph capture.
      graph_state->metadata = std::make_shared<layer::AttentionMetadata>(
          layer::AttentionMetadataBuilder::build(params,
                                                 /*enable_mla=*/true,
                                                 /*compute_dtype=*/"half",
                                                 /*attn_mask=*/std::nullopt,
                                                 positions.device()));
      // A captured score workspace must cover future context growth. Zero
      // selects KPool's static capacity from the persistent block-table width,
      // rather than freezing capacity at the first request's current length.
      graph_state->metadata->max_seq_len = 0;
      params.attn_metadata = graph_state->metadata;
      layer::prepare_glm5_next_kpool_metadata(*params.attn_metadata,
                                              positions.device());
      return;
    }
    params.attn_metadata = graph_state->metadata;
    auto& metadata = *params.attn_metadata;
    metadata.kv_seq_lens.copy_(torch::diff(attention.kv_seq_lens));
    metadata.q_seq_lens.copy_(torch::diff(attention.q_seq_lens));
    if (metadata.has_initial_states.defined()) {
      // The executor refreshes the graph-owned device mask, preferring an
      // explicit device input over the host source just like the builder.
      if (params.linear_state_validity_mask_tensor.defined()) {
        if (metadata.has_initial_states.data_ptr() !=
            params.linear_state_validity_mask_tensor.data_ptr()) {
          metadata.has_initial_states.copy_(
              params.linear_state_validity_mask_tensor);
        }
      } else {
        metadata.has_initial_states.copy_(
            torch::tensor(params.linear_state_validity_mask,
                          metadata.has_initial_states.options()));
      }
    }
    auto& batch = *metadata.kpool_batch_metadata;
    // Ordinary decode aliases graph-owned storage, refreshed by the executor.
    // Expanded attention rows materialize a request-level table instead. Copy
    // its current page IDs on every replay without replacing captured storage.
    if (batch.block_table.data_ptr() != attention.block_tables.data_ptr()) {
      batch.block_table.copy_(attention.block_tables.index_select(
          /*dim=*/0,
          batch.query_starts.narrow(
              /*dim=*/0, /*start=*/0, batch.q_seq_lens.size())));
    }
  }
};

}  // namespace xllm::mlu::model
