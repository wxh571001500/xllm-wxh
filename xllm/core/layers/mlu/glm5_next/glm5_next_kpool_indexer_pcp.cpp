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

#include "common/constants.h"
#include "kernels/mlu/kpool.h"
#include "layers/mlu/glm5_next/glm5_next_kpool_indexer.h"

namespace xllm::layer {
namespace {

// Rows that belong to padded linear states or unmapped slots must not update
// the KPool cache; mark their positions as -1.
torch::Tensor mask_inactive_positions(const torch::Tensor& positions,
                                      const AttentionMetadata& metadata,
                                      const KPoolBatchMetadata& batch) {
  torch::Tensor active =
      batch.tail_indices.index_select(/*dim=*/0, batch.row_batch) >
      kPaddingLinearStateId;
  if (metadata.slot_mapping.defined()) {
    active = active & (metadata.slot_mapping.reshape({-1}) > 0);
  }
  return torch::where(active, positions, torch::full_like(positions, -1));
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> Glm5NextKPoolIndexerImpl::forward_pcp(
    const torch::Tensor& local_hidden_states,
    const torch::Tensor& local_q_norm,
    const torch::Tensor& local_positions,
    const torch::Tensor& global_positions,
    torch::Tensor& index_cache,
    torch::Tensor& tail_cache,
    const AttentionMetadata& global_metadata,
    const AttentionMetadata& local_metadata,
    const glm5_next_pcp::Context& context) {
  // PCP is admitted only when every CP rank owns tokens.
  CHECK_GT(local_hidden_states.size(0), 0);
  const torch::Tensor local_raw_k =
      project_raw_k(local_hidden_states, local_positions, local_metadata);
  const torch::Tensor local_gate =
      torch::nn::functional::linear(local_hidden_states,
                                    index_kpool_compress_gate_)
          .to(torch::kBFloat16);
  auto pending_index = parallel_state::launch_gather(
      torch::cat({local_raw_k, local_gate}, /*dim=*/1),
      context.cp_group,
      context.geometry.tokens_per_rank);
  const Execution global_execution =
      prepare_execution(global_metadata, global_positions);
  const torch::Tensor cache_positions = mask_inactive_positions(
      global_positions, global_metadata, *global_execution.batch);
  const Execution local_execution =
      prepare_execution(local_metadata, local_positions);
  const torch::Tensor local_cache_positions = mask_inactive_positions(
      local_positions, local_metadata, *local_execution.batch);
  const torch::Tensor query = project_query(
      local_q_norm, torch::clamp_min(local_cache_positions, 0), local_metadata);
  const torch::Tensor weights =
      weights_proj_->forward(local_hidden_states.to(torch::kFloat32));
  const torch::Tensor gathered =
      glm5_next_pcp::finish_gather_restore(std::move(pending_index), context);
  const torch::Tensor global_raw_k =
      gathered.narrow(/*dim=*/1, /*start=*/0, head_dim_);
  const torch::Tensor global_gate =
      gathered.narrow(/*dim=*/1, head_dim_, head_dim_);
  update_cache(global_raw_k,
               global_gate,
               cache_positions,
               index_cache,
               tail_cache,
               global_execution.batch->tail_indices,
               global_execution.batch->block_table,
               global_execution);

  const torch::Tensor pools = select_projected_pools(query,
                                                     weights,
                                                     local_cache_positions,
                                                     index_cache,
                                                     local_metadata,
                                                     local_execution);
  kernel::mlu::KPoolSelection selection =
      kernel::mlu::expand_kpool(pools,
                                local_cache_positions,
                                local_execution.batch->row_batch,
                                local_execution.batch->block_table,
                                block_size_,
                                index_topk_,
                                index_kpool_,
                                always_select_tail_);
  return {std::move(selection.physical_slots),
          std::move(selection.context_lens)};
}

}  // namespace xllm::layer
