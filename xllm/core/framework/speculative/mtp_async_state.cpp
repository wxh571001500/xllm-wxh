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

#include "core/framework/speculative/mtp_async_state.h"

#include <glog/logging.h>

#include <algorithm>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"

namespace xllm::mtp_async {
namespace {

torch::Tensor gather_sequence_rows(const torch::Tensor& values,
                                   const torch::Tensor& indices) {
  CHECK_GE(values.dim(), 2);
  CHECK_EQ(values.size(0), indices.numel());
  torch::Tensor gather_index =
      indices.to(torch::dtype(torch::kLong).device(indices.device()))
          .view({-1, 1});
  for (int64_t dim = 2; dim < values.dim(); ++dim) {
    gather_index = gather_index.unsqueeze(-1);
  }
  std::vector<int64_t> expanded_shape = values.sizes().vec();
  expanded_shape[1] = 1;
  gather_index = gather_index.expand(expanded_shape);
  return values.gather(/*dim=*/1, gather_index).squeeze(/*dim=*/1);
}

}  // namespace

TargetSpecVerifyMode classify_target_spec_verify_mode(
    std::string_view model_type,
    bool is_python_model) {
  if (is_qwen3_5_target_model_type(model_type)) {
    return TargetSpecVerifyMode::EXPANDED_VERIFY;
  }
  if (is_glm5_next_target_model_type(model_type) ||
      model_type == "glm5_next_text") {
    return is_python_model ? TargetSpecVerifyMode::EXPANDED_VERIFY
                           : TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL;
  }
  if (is_python_model &&
      (model_type == "deepseek_v32" || model_type == "deepseek_v4" ||
       model_type == "deepseek_v4_dspark")) {
    return TargetSpecVerifyMode::EXPANDED_VERIFY;
  }
  if (model_type == "mimo") {
    return TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL;
  }
  return TargetSpecVerifyMode::GENERIC;
}

bool requires_uniform_spec_verify(std::string_view model_type) {
  return is_qwen3_5_target_model_type(model_type) ||
         is_glm5_next_target_model_type(model_type) ||
         model_type == "glm5_next_text";
}

bool supports_native_spec_verify_replay_update(TargetSpecVerifyMode mode,
                                               bool is_python_model) {
  return !is_python_model && mode == TargetSpecVerifyMode::EXPANDED_VERIFY;
}

int64_t speculative_verify_block_table_capacity(int64_t max_position_embeddings,
                                                int64_t block_size) {
  CHECK_GT(max_position_embeddings, 0);
  CHECK_GT(block_size, 0);
  return (max_position_embeddings + block_size - 1) / block_size + 1;
}

bool has_speculative_verify_block_table_layout(
    const torch::Tensor& block_tables,
    const std::vector<torch::Tensor>& multi_block_tables,
    int64_t num_sequences) {
  if (num_sequences <= 0) {
    return false;
  }
  const auto is_compatible = [num_sequences](const torch::Tensor& table) {
    return table.defined() && table.dim() == 2 &&
           table.size(0) == num_sequences && table.size(1) > 0 &&
           table.scalar_type() == torch::kInt32;
  };
  if (block_tables.defined()) {
    return is_compatible(block_tables);
  }
  return !multi_block_tables.empty() && std::all_of(multi_block_tables.begin(),
                                                    multi_block_tables.end(),
                                                    is_compatible);
}

torch::Tensor make_speculative_verify_control_block_table(
    int64_t num_sequences,
    int64_t block_table_capacity) {
  CHECK_GT(num_sequences, 0);
  CHECK_GT(block_table_capacity, 0);
  return torch::zeros({num_sequences, block_table_capacity},
                      torch::TensorOptions()
                          .dtype(torch::kInt32)
                          .device(torch::kCPU)
                          .pinned_memory(true));
}

CombinedDraftExecutionPath classify_combined_draft_execution_path(
    std::string_view model_type) {
  if (model_type == "qwen3_5_mtp" || model_type == "qwen3_5_moe_mtp") {
    return CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION;
  }
  if (model_type == "glm_moe_dsa_mtp") {
    return CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION;
  }
  return CombinedDraftExecutionPath::UNSUPPORTED;
}

bool supports_combined_draft_configuration(
    CombinedDraftExecutionPath execution_path,
    std::string_view npu_backend,
    int32_t dp_size) {
  switch (execution_path) {
    case CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION:
      return npu_backend == "TORCH" && dp_size <= 1;
    case CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION:
      return npu_backend == "ATB";
    case CombinedDraftExecutionPath::UNSUPPORTED:
      return false;
  }
  return false;
}

bool supports_python_dsv32_continuous_draft_configuration(
    std::string_view target_model_type,
    std::string_view draft_model_type,
    std::string_view npu_backend,
    int32_t dp_size,
    bool is_python_target,
    bool is_python_draft,
    bool has_model_managed_block_tables,
    int32_t target_index_topk,
    int32_t draft_index_topk) {
  return target_model_type == "deepseek_v32" &&
         draft_model_type == "deepseek_v32_mtp" && npu_backend == "TORCH" &&
         dp_size == 1 && is_python_target && is_python_draft &&
         !has_model_managed_block_tables && target_index_topk > 0 &&
         draft_index_topk > 0;
}

bool json_object_allows_draft_prelaunch(
    int32_t dp_size,
    const std::vector<int32_t>& dp_global_json_object_active,
    bool has_local_json_object_states) {
  if (dp_size == 1) {
    // A single rank owns the whole batch, so its local state decides.
    return !has_local_json_object_states;
  }
  if (dp_size <= 0 ||
      dp_global_json_object_active.size() != static_cast<size_t>(dp_size)) {
    return false;
  }
  // Only a complete all-zero vector proves the whole DP batch is plain. This
  // also rejects flags outside {0, 1} without a rank-local fallback.
  return std::all_of(dp_global_json_object_active.begin(),
                     dp_global_json_object_active.end(),
                     [](int32_t active) { return active == 0; });
}

torch::Tensor materialize_speculative_verify_tokens(
    const torch::Tensor& verify_tokens,
    const std::vector<torch::Tensor>& draft_token_sources) {
  if (draft_token_sources.empty()) {
    return verify_tokens;
  }
  CHECK(verify_tokens.defined());
  CHECK_EQ(verify_tokens.dim(), 1);
  const int64_t verify_width =
      static_cast<int64_t>(draft_token_sources.size()) + 1;
  CHECK_EQ(verify_tokens.numel() % verify_width, 0);
  const int64_t batch_size = verify_tokens.numel() / verify_width;
  torch::Tensor verify_rows = verify_tokens.view({batch_size, verify_width});
  for (size_t step = 0; step < draft_token_sources.size(); ++step) {
    const torch::Tensor& source = draft_token_sources[step];
    CHECK(source.defined());
    CHECK_EQ(source.numel(), batch_size);
    verify_rows.select(/*dim=*/1, static_cast<int64_t>(step) + 1)
        .copy_(source.flatten(), /*non_blocking=*/true);
  }
  return verify_tokens;
}

torch::Tensor materialize_graph_speculative_verify_tokens(
    const torch::Tensor& tokens,
    const GraphInput& graph_input) {
  const torch::Tensor& verify_tokens =
      graph_input.input_tokens_override.defined()
          ? graph_input.input_tokens_override
          : tokens;
  return materialize_speculative_verify_tokens(
      verify_tokens, graph_input.spec_verify_draft_token_sources);
}

torch::Tensor extract_target_base_kv_seq_lens(
    const torch::Tensor& validate_kv_seq_lens,
    int64_t batch_size,
    int64_t num_validate_tokens,
    bool use_chunked_prefill) {
  CHECK(validate_kv_seq_lens.defined());
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_validate_tokens, 0);
  torch::Tensor flattened = validate_kv_seq_lens.flatten();
  if (use_chunked_prefill) {
    CHECK_GE(flattened.numel(), batch_size);
    return flattened.slice(/*dim=*/0, /*start=*/0, /*end=*/batch_size) -
           (num_validate_tokens - 1);
  }

  const int64_t expanded_rows = batch_size * num_validate_tokens;
  CHECK_GE(flattened.numel(), expanded_rows);
  return flattened.slice(/*dim=*/0, /*start=*/0, /*end=*/expanded_rows)
      .view({batch_size, num_validate_tokens})
      .select(/*dim=*/1, /*index=*/0)
      .contiguous();
}

AcceptedState build_accepted_state(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& accepted_embeddings,
                                   const torch::Tensor& embedding_placeholder,
                                   const torch::Tensor& base_positions,
                                   const torch::Tensor& base_kv_seq_lens) {
  AcceptedTokenMetadata token_metadata = build_accepted_token_metadata(
      accepted_tokens, base_positions, base_kv_seq_lens);
  CHECK_EQ(accepted_embeddings.dim(), 3);
  const int64_t batch_size = accepted_tokens.size(0);
  CHECK_EQ(accepted_embeddings.size(0), batch_size);

  AcceptedState state;
  state.accepted_lengths = token_metadata.accepted_lengths;
  state.all_draft_accepted =
      state.accepted_lengths.eq(accepted_tokens.size(/*dim=*/1));
  state.last_tokens = token_metadata.last_tokens;
  state.base_positions = token_metadata.base_positions;
  state.base_kv_seq_lens = token_metadata.base_kv_seq_lens;

  torch::Tensor previous_indices = (state.accepted_lengths - 2).clamp_min(0);
  torch::Tensor gathered_previous_tokens =
      gather_sequence_rows(accepted_tokens, previous_indices);
  torch::Tensor has_previous = state.accepted_lengths.gt(1);
  state.previous_tokens =
      torch::where(has_previous, gathered_previous_tokens, state.last_tokens);
  torch::Tensor last_indices = (state.accepted_lengths - 1).clamp_min(0);
  state.last_embeddings =
      gather_sequence_rows(accepted_embeddings, last_indices);
  torch::Tensor gathered_previous_embeddings =
      gather_sequence_rows(accepted_embeddings, previous_indices);
  torch::Tensor placeholder = embedding_placeholder;
  if (placeholder.dim() == 1) {
    placeholder = placeholder.unsqueeze(0);
  }
  placeholder = placeholder.expand_as(gathered_previous_embeddings);
  state.previous_embeddings = torch::where(has_previous.view({batch_size, 1}),
                                           gathered_previous_embeddings,
                                           placeholder);

  return state;
}

AcceptedTokenMetadata build_accepted_token_metadata(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens) {
  CHECK_EQ(accepted_tokens.dim(), 2);
  const int64_t batch_size = accepted_tokens.size(0);
  CHECK_GT(accepted_tokens.size(1), 0);
  CHECK_GE(base_positions.numel(), batch_size);
  CHECK_GE(base_kv_seq_lens.numel(), batch_size);

  AcceptedTokenMetadata metadata;
  metadata.accepted_lengths =
      accepted_tokens.ge(0).sum(/*dim=*/1).to(torch::kLong);
  torch::Tensor last_indices = (metadata.accepted_lengths - 1).clamp_min(0);
  metadata.last_tokens = gather_sequence_rows(accepted_tokens, last_indices);
  metadata.base_positions =
      base_positions.flatten().slice(0, 0, batch_size).to(torch::kLong) +
      metadata.accepted_lengths;
  metadata.base_kv_seq_lens =
      base_kv_seq_lens.flatten().slice(0, 0, batch_size).to(torch::kLong) +
      metadata.accepted_lengths;
  return metadata;
}

torch::Tensor make_row_positions(const AcceptedState& state,
                                 const torch::Tensor& offsets) {
  return state.base_positions.unsqueeze(1) +
         offsets.to(state.base_positions.options()).unsqueeze(0);
}

torch::Tensor make_kv_seq_lens(const AcceptedState& state,
                               const torch::Tensor& offsets,
                               bool use_chunked_prefill) {
  if (use_chunked_prefill) {
    return state.base_kv_seq_lens;
  }
  return (state.base_kv_seq_lens.unsqueeze(1) +
          offsets.to(state.base_kv_seq_lens.options()).unsqueeze(0))
      .flatten();
}

torch::Tensor make_repair_cache_positions(const AcceptedState& state) {
  return torch::where(state.all_draft_accepted,
                      state.base_positions - 1,
                      state.base_positions + 1);
}

}  // namespace xllm::mtp_async
