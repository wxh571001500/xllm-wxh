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

#pragma once

#include <torch/torch.h>

#include <optional>

#include "core/framework/model/mtp_topk_state.h"

namespace xllm {
struct ModelOutput {
  // [num_tokens, hidden_size]
  torch::Tensor hidden_states;
  // [num_tokens, hidden_size]
  torch::Tensor residual;
  // [num_tokens, ...]
  torch::Tensor aux_hidden_states;
  // Backend-neutral state emitted for the next MTP draft step.
  MtpTopkStatePtr mtp_topk_state;
  // Eagle3 GRAPH path only: the per-layer aux hidden states, UN-combined (one
  // [num_tokens, hidden] tensor per captured layer). Kept separate so the
  // hidden-dim concat is NOT done inside the ACL Graph capture region: a
  // combined [num_tokens, 3*hidden] tensor (~2.75MB at bucket 64) crosses the
  // CANN large-copy/SDMA threshold and dispatches to a side stream that is not
  // joined to the capture stream, failing capture_end(). Each element here is
  // ~917KB (same as the working set_hidden_states copy). The executor copies
  // them per-layer into a persistent [nc, max_tokens, hidden] buffer and the
  // getter concatenates AFTER capture_end (graph-external). Mirrors vLLM, which
  // returns aux as a list and cat's outside the graph.
  std::vector<torch::Tensor> aux_hidden_states_list;

  ModelOutput() = default;

  // Constructor with only hidden_states (for backward compatibility)
  explicit ModelOutput(const torch::Tensor& hidden_states)
      : hidden_states(hidden_states) {}

  explicit ModelOutput(const torch::Tensor& hidden_states,
                       const torch::Tensor& residual)
      : hidden_states(hidden_states), residual(residual) {}

  // Constructor with optional residual for multi-device compatibility
  explicit ModelOutput(const torch::Tensor& hidden_states,
                       const std::optional<torch::Tensor>& residual)
      : hidden_states(hidden_states) {
    if (residual.has_value()) {
      this->residual = residual.value();
    }
  }

  explicit ModelOutput(
      std::pair<torch::Tensor, torch::Tensor> hidden_states_and_residual)
      : hidden_states(hidden_states_and_residual.first),
        residual(hidden_states_and_residual.second) {}

  ModelOutput(torch::Tensor hidden_states,
              torch::Tensor residual,
              torch::Tensor aux_hidden_states)
      : hidden_states(hidden_states),
        residual(residual),
        aux_hidden_states(aux_hidden_states) {}
};
}  // namespace xllm
