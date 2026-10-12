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

#include <cstdint>
#include <vector>

#include "framework/parallel_state/parallel_state.h"
#include "layers/common/attention_metadata.h"

namespace xllm::layer::glm5_next_pcp {

struct Geometry final {
  std::vector<std::vector<int64_t>> rows_by_rank;
  std::vector<std::vector<int32_t>> lengths_by_rank;
  std::vector<int32_t> tokens_per_rank;
  std::vector<int64_t> restore_indices;
  int32_t total_tokens = 0;
};

struct Context final {
  Geometry geometry;
  std::vector<int32_t> local_query_lengths;
  AttentionMetadata local_metadata;
  torch::Tensor global_positions;
  torch::Tensor local_row_indices;
  torch::Tensor restore_indices;
  torch::Tensor local_cu_seqlens;
  torch::Tensor local_chunk_indices;
  int32_t local_max_query_len = 0;
  int32_t cp_rank = 0;
  ProcessGroup* cp_group = nullptr;
};

std::vector<int32_t> query_lengths_from_cumulative(
    const std::vector<int32_t>& cumulative_lengths,
    int64_t total_tokens);

Geometry build_geometry(const std::vector<int32_t>& query_lengths,
                        int32_t cp_size,
                        int32_t chunk_size);

Context build_context(const std::vector<int32_t>& query_lengths,
                      int32_t cp_rank,
                      ProcessGroup* cp_group,
                      int32_t chunk_size,
                      const torch::Device& device);

// The last CP rank holds the tail of every request, so it owns the final
// recurrent states that prefill publishes to the other ranks.
int32_t last_rank(const Context& context);

torch::Tensor shard_rows(const torch::Tensor& global, const Context& context);

torch::Tensor finish_gather_restore(parallel_state::GatherAsyncCtx pending,
                                    const Context& context);

torch::Tensor gather_restore(const torch::Tensor& local,
                             const Context& context);

}  // namespace xllm::layer::glm5_next_pcp
