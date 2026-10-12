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

#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

#include "kernels/mlu/chunk_kda_pcp.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "layers/mlu/glm5_next/glm5_next_kda.h"
#include "layers/mlu/qwen3_5/qwen3_5_gated_delta_net.h"

namespace xllm::layer {
namespace {

torch::Tensor prepare_conv_tails(const torch::Tensor& local_qkv,
                                 int64_t history_size,
                                 const glm5_next_pcp::Context& context) {
  const int64_t channels = local_qkv.size(1);
  const int64_t requests =
      static_cast<int64_t>(context.local_query_lengths.size());
  std::vector<torch::Tensor> tails;
  tails.reserve(static_cast<size_t>(requests));
  int64_t local_offset = 0;
  for (const int32_t length : context.local_query_lengths) {
    torch::Tensor tail =
        torch::zeros({history_size, channels}, local_qkv.options());
    const int64_t count = std::min<int64_t>(length, history_size);
    if (count > 0) {
      tail.narrow(/*dim=*/0, history_size - count, count)
          .copy_(local_qkv.narrow(
              /*dim=*/0, local_offset + length - count, count));
    }
    tails.emplace_back(std::move(tail));
    local_offset += length;
  }
  return torch::stack(tails);
}

std::tuple<torch::Tensor, torch::Tensor> local_causal_conv(
    const torch::Tensor& local_qkv,
    const torch::Tensor& gathered_tails,
    const torch::Tensor& weight,
    const torch::Tensor& conv_cache,
    const torch::Tensor& state_indices,
    const AttentionMetadata& global_metadata,
    const glm5_next_pcp::Context& context) {
  const int64_t history_size = weight.size(1) - 1;
  const int64_t requests =
      static_cast<int64_t>(context.local_query_lengths.size());
  torch::Tensor initial_history =
      conv_cache.index_select(/*dim=*/0, state_indices)
          .transpose(/*dim0=*/1, /*dim1=*/2)
          .contiguous();
  initial_history =
      torch::where(global_metadata.has_initial_states.view({requests, 1, 1}),
                   initial_history,
                   torch::zeros_like(initial_history));
  std::vector<torch::Tensor> local_history;
  local_history.reserve(static_cast<size_t>(requests));
  for (int64_t request = 0; request < requests; ++request) {
    torch::Tensor history = initial_history.select(/*dim=*/0, request);
    for (int32_t rank = 0; rank < context.cp_rank; ++rank) {
      const int64_t length =
          context.geometry.lengths_by_rank[static_cast<size_t>(rank)]
                                          [static_cast<size_t>(request)];
      const int64_t count = std::min(length, history_size);
      if (count > 0) {
        const torch::Tensor earlier =
            gathered_tails.select(/*dim=*/0, rank)
                .select(/*dim=*/0, request)
                .narrow(/*dim=*/0, history_size - count, count);
        history = torch::cat({history, earlier}, /*dim=*/0)
                      .narrow(/*dim=*/0, count, history_size);
      }
    }
    local_history.emplace_back(std::move(history));
  }
  torch::Tensor temporary_state = torch::stack(local_history)
                                      .transpose(/*dim0=*/1, /*dim1=*/2)
                                      .contiguous();
  temporary_state = torch::cat({torch::zeros_like(temporary_state.narrow(
                                    /*dim=*/0, /*start=*/0, /*length=*/1)),
                                temporary_state},
                               /*dim=*/0);
  const torch::Tensor compact_state_indices = torch::arange(
      1, requests + 1, state_indices.options().dtype(torch::kInt32));
  torch::Tensor convolved = kernel::mlu::causal_conv1d_fn(
      local_qkv.transpose(/*dim0=*/0, /*dim1=*/1),
      weight,
      temporary_state,
      context.local_metadata.q_cu_seq_lens,
      context.local_metadata.batch,
      context.local_metadata.token_block_offset,
      context.local_metadata.tot,
      /*bias=*/std::nullopt,
      compact_state_indices,
      torch::ones_like(global_metadata.has_initial_states),
      /*initial_state_idx=*/std::nullopt,
      /*num_accepted_tokens=*/std::nullopt,
      /*inplace_final_state=*/true);
  torch::Tensor final_state =
      context.cp_rank == glm5_next_pcp::last_rank(context)
          ? temporary_state.narrow(/*dim=*/0, 1, requests).contiguous()
          : torch::empty_like(temporary_state.narrow(/*dim=*/0, 1, requests));
  return {convolved.transpose(/*dim0=*/0, /*dim1=*/1), final_state};
}

}  // namespace

torch::Tensor Glm5NextKDAImpl::forward_pcp(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const glm5_next_pcp::Context& context) {
  const int64_t num_tokens = hidden_states.size(0);
  torch::Tensor projected = in_proj_qkvbfg_a_->forward(hidden_states);
  const auto projections = projected.split_with_sizes(
      {3 * local_projection_size_, local_num_heads_, head_dim_, head_dim_},
      /*dim=*/-1);
  torch::Tensor mixed_qkv = projections[0].contiguous();
  torch::Tensor beta = projections[1].contiguous();

  // Prefill consumes only committed history; speculative checkpoints occupy
  // the remaining columns of the same request's convolution state.
  torch::Tensor conv_cache = kv_cache.get_conv_cache().transpose(-1, -2).narrow(
      /*dim=*/2,
      /*start=*/0,
      /*length=*/conv_kernel_size_ - 1);
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  const int64_t checkpoint_stride = get_checkpoint_stride(kv_cache);
  torch::Tensor conv_weight = torch::cat(
      {q_conv1d_->weight(), k_conv1d_->weight(), v_conv1d_->weight()},
      /*dim=*/0);
  torch::Tensor logical_state_indices =
      get_linear_state_indices(input_params, mixed_qkv.device());
  torch::Tensor state_base_indices =
      build_linear_state_base_indices(logical_state_indices, checkpoint_stride);

  // Halo depends only on QKV; gate projections can run while it travels.
  torch::Tensor local_tails =
      prepare_conv_tails(mixed_qkv, conv_weight.size(1) - 1, context);
  auto tail_shape = local_tails.sizes().vec();
  tail_shape.insert(tail_shape.begin(), context.cp_group->world_size());
  torch::Tensor gathered_tails =
      torch::empty(tail_shape, local_tails.options());
  auto halo_work =
      context.cp_group->allgather_base_async(local_tails, gathered_tails);
  torch::Tensor raw_gate = f_b_proj_->forward(projections[2].contiguous())
                               .view({num_tokens, local_num_heads_, head_dim_})
                               .contiguous();
  torch::Tensor prefix_state = ssm_cache.index({state_base_indices});
  prefix_state.index_put_(
      {~attn_metadata.has_initial_states, torch::indexing::Ellipsis}, 0.0f);
  halo_work->wait();
  torch::Tensor conv_final_state;
  std::tie(mixed_qkv, conv_final_state) =
      local_causal_conv(mixed_qkv,
                        gathered_tails,
                        conv_weight,
                        conv_cache,
                        logical_state_indices,
                        attn_metadata,
                        context);
  auto conv_work = context.cp_group->broadcast_async(
      conv_final_state, glm5_next_pcp::last_rank(context));
  torch::Tensor q;
  torch::Tensor k;
  torch::Tensor v;
  std::tie(q, k, v) = split_mixed_qkv(mixed_qkv);

  const torch::Tensor local_q = q;
  const torch::Tensor local_k = k;
  const torch::Tensor local_v = v;
  const torch::Tensor local_gate = raw_gate;
  const torch::Tensor local_beta = beta;
  const int64_t chunk_size = kernel::mlu::kda_prefill_chunk_size(
      local_num_heads_, /*use_qk_l2norm=*/true);
  kernel::mlu::ChunkKDAPcpPrepared prepared =
      kernel::mlu::chunk_kda_pcp_prepare(local_q,
                                         local_k,
                                         local_v,
                                         local_gate,
                                         A_log_,
                                         dt_bias_,
                                         gate_lower_bound_,
                                         local_beta,
                                         context.local_cu_seqlens,
                                         context.local_chunk_indices,
                                         chunk_size);
  auto summary_shape = prepared.summary.sizes().vec();
  summary_shape.insert(summary_shape.begin(), context.cp_group->world_size());
  torch::Tensor gathered_summary =
      torch::empty(summary_shape, prepared.summary.options());
  auto summary_work = context.cp_group->allgather_base_async(prepared.summary,
                                                             gathered_summary);
  torch::Tensor output_gate =
      g_b_proj_->forward(projections[3].contiguous())
          .view({num_tokens, local_num_heads_, head_dim_})
          .contiguous();
  torch::Tensor local_output;
  torch::Tensor local_final_state;
  if (context.cp_rank == 0) {
    // Rank zero's prefix is independent of every remote summary.
    std::tie(local_output, local_final_state) =
        kernel::mlu::chunk_kda_pcp_replay(prepared, prefix_state);
    summary_work->wait();
  } else {
    summary_work->wait();
    torch::Tensor local_initial = kernel::mlu::chunk_kda_pcp_merge(
        gathered_summary, prefix_state, context.cp_rank, local_q.size(1));
    std::tie(local_output, local_final_state) =
        kernel::mlu::chunk_kda_pcp_replay(prepared, local_initial);
  }

  torch::Tensor final_state =
      context.cp_rank == glm5_next_pcp::last_rank(context)
          ? local_final_state
          : torch::empty_like(prefix_state);
  auto state_work = context.cp_group->broadcast_async(
      final_state, glm5_next_pcp::last_rank(context));

  torch::Tensor core_output = local_output.squeeze(/*dim=*/0);
  core_output = core_output.view({-1, head_dim_});
  output_gate = output_gate.view({-1, head_dim_});
  torch::Tensor normalized = o_norm_->forward(core_output);
  normalized = glm5_next_kda_apply_output_gate(normalized, output_gate);
  normalized = normalized.view({num_tokens, local_projection_size_});
  torch::Tensor output = o_proj_->forward(normalized);
  // Publish caches on the caller's stream before returning to the decoder.
  conv_work->wait();
  conv_cache.index_put_({logical_state_indices}, conv_final_state);
  state_work->wait();
  ssm_cache.index_put_({state_base_indices},
                       final_state.to(ssm_cache.scalar_type()));
  return output;
}

}  // namespace xllm::layer
