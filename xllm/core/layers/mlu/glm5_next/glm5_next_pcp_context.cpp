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

#include "layers/mlu/glm5_next/glm5_next_pcp_context.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "framework/parallel_state/parallel_state.h"

namespace xllm::layer::glm5_next_pcp {

std::vector<int32_t> query_lengths_from_cumulative(
    const std::vector<int32_t>& cumulative_lengths,
    int64_t total_tokens) {
  CHECK_GE(cumulative_lengths.size(), 2u);
  CHECK_EQ(cumulative_lengths.front(), 0);
  CHECK_EQ(cumulative_lengths.back(), total_tokens);
  std::vector<int32_t> lengths;
  lengths.reserve(cumulative_lengths.size() - 1);
  for (size_t index = 1; index < cumulative_lengths.size(); ++index) {
    const int32_t length =
        cumulative_lengths[index] - cumulative_lengths[index - 1];
    CHECK_GE(length, 0);
    lengths.emplace_back(length);
  }
  return lengths;
}

Geometry build_geometry(const std::vector<int32_t>& query_lengths,
                        int32_t cp_size,
                        int32_t chunk_size) {
  CHECK_GT(cp_size, 0);
  CHECK_GT(chunk_size, 0);

  Geometry geometry;
  geometry.rows_by_rank.resize(static_cast<size_t>(cp_size));
  geometry.lengths_by_rank.resize(static_cast<size_t>(cp_size));
  geometry.tokens_per_rank.resize(static_cast<size_t>(cp_size), 0);

  int64_t request_offset = 0;
  for (const int32_t query_length : query_lengths) {
    CHECK_GE(query_length, 0);
    const int64_t chunk_count =
        (static_cast<int64_t>(query_length) + chunk_size - 1) / chunk_size;
    for (int32_t rank = 0; rank < cp_size; ++rank) {
      const int32_t begin = static_cast<int32_t>(std::min<int64_t>(
          query_length, chunk_size * (chunk_count * rank / cp_size)));
      const int32_t end = static_cast<int32_t>(std::min<int64_t>(
          query_length, chunk_size * (chunk_count * (rank + 1) / cp_size)));
      const size_t rank_index = static_cast<size_t>(rank);
      geometry.lengths_by_rank[rank_index].emplace_back(end - begin);
      std::vector<int64_t>& rows = geometry.rows_by_rank[rank_index];
      for (int32_t token = begin; token < end; ++token) {
        rows.emplace_back(request_offset + token);
      }
    }
    request_offset += query_length;
  }
  CHECK_LE(request_offset, std::numeric_limits<int32_t>::max());
  geometry.total_tokens = static_cast<int32_t>(request_offset);

  geometry.restore_indices.resize(static_cast<size_t>(geometry.total_tokens));
  int64_t gathered_index = 0;
  for (int32_t rank = 0; rank < cp_size; ++rank) {
    const size_t rank_index = static_cast<size_t>(rank);
    const std::vector<int64_t>& rows = geometry.rows_by_rank[rank_index];
    geometry.tokens_per_rank[rank_index] = static_cast<int32_t>(rows.size());
    for (const int64_t row : rows) {
      geometry.restore_indices[static_cast<size_t>(row)] = gathered_index++;
    }
  }
  return geometry;
}

Context build_context(const std::vector<int32_t>& query_lengths,
                      int32_t cp_rank,
                      ProcessGroup* cp_group,
                      int32_t chunk_size,
                      const torch::Device& device) {
  CHECK(cp_group != nullptr);
  const int32_t cp_size = cp_group->world_size();
  CHECK_GT(cp_size, 1);
  CHECK_GE(cp_rank, 0);
  CHECK_LT(cp_rank, cp_size);
  Context context;
  context.geometry = build_geometry(query_lengths, cp_size, chunk_size);
  context.cp_rank = cp_rank;
  context.cp_group = cp_group;

  const torch::TensorOptions int64_options =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
  const torch::TensorOptions int32_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const size_t rank_index = static_cast<size_t>(cp_rank);
  context.local_query_lengths = context.geometry.lengths_by_rank[rank_index];
  if (!context.local_query_lengths.empty()) {
    context.local_max_query_len = *std::max_element(
        context.local_query_lengths.begin(), context.local_query_lengths.end());
  }
  context.local_row_indices =
      torch::tensor(context.geometry.rows_by_rank[rank_index], int64_options);
  context.restore_indices =
      torch::tensor(context.geometry.restore_indices, int64_options);

  std::vector<int32_t> cumulative = {0};
  std::vector<int32_t> indices;
  for (size_t sequence = 0;
       sequence < context.geometry.lengths_by_rank[rank_index].size();
       ++sequence) {
    const int32_t length =
        context.geometry.lengths_by_rank[rank_index][sequence];
    cumulative.emplace_back(cumulative.back() + length);
    const int32_t chunks = (length + chunk_size - 1) / chunk_size;
    for (int32_t chunk = 0; chunk < chunks; ++chunk) {
      indices.emplace_back(static_cast<int32_t>(sequence));
      indices.emplace_back(chunk);
    }
  }
  context.local_cu_seqlens = torch::tensor(cumulative, int32_options);
  context.local_chunk_indices =
      torch::tensor(indices, int32_options)
          .view({static_cast<int64_t>(indices.size() / 2), 2});
  return context;
}

int32_t last_rank(const Context& context) {
  return context.cp_group->world_size() - 1;
}

torch::Tensor shard_rows(const torch::Tensor& global, const Context& context) {
  return global.index_select(/*dim=*/0, context.local_row_indices);
}

torch::Tensor gather_restore(const torch::Tensor& local,
                             const Context& context) {
  return finish_gather_restore(
      parallel_state::launch_gather(
          local, context.cp_group, context.geometry.tokens_per_rank),
      context);
}

torch::Tensor finish_gather_restore(parallel_state::GatherAsyncCtx pending,
                                    const Context& context) {
  torch::Tensor gathered = parallel_state::finish_gather(std::move(pending));
  if (context.local_query_lengths.size() == 1) {
    return gathered;
  }
  return gathered.index_select(/*dim=*/0, context.restore_indices);
}

}  // namespace xllm::layer::glm5_next_pcp
