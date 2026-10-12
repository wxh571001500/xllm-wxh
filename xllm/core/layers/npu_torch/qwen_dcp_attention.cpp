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

#include "layers/npu_torch/qwen_dcp_attention.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

#include "framework/kv_cache/kv_shard_layout.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/parallel_state/process_group.h"
#include "kernels/npu/npu_ops_api.h"
#include "layers/common/attention_metadata.h"
#include "layers/npu_torch/fused_infer_attention_utils.h"
#include "platform/npu/acl_graph_task_update_context.h"

namespace xllm::layer {
namespace detail {

DcpAttentionResult merge_dcp_attention_shards(
    const torch::Tensor& partial_outputs,
    const torch::Tensor& partial_lse) {
  CHECK_EQ(partial_outputs.dim(), partial_lse.dim());
  CHECK_EQ(partial_lse.size(-1), 1);
  for (int64_t dim = 0; dim + 1 < partial_lse.dim(); ++dim) {
    CHECK_EQ(partial_outputs.size(dim), partial_lse.size(dim));
  }

  const torch::Tensor lse = partial_lse.to(torch::kFloat32);
  const torch::Tensor finite_lse = torch::isfinite(lse);
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  const torch::Tensor sanitized_lse =
      torch::where(finite_lse, lse, torch::full_like(lse, negative_infinity));
  const torch::Tensor max_lse =
      std::get<0>(sanitized_lse.max(/*dim=*/0, /*keepdim=*/true));
  const torch::Tensor safe_max_lse = torch::where(
      torch::isfinite(max_lse), max_lse, torch::zeros_like(max_lse));
  const torch::Tensor weights = torch::exp(sanitized_lse - safe_max_lse);
  const torch::Tensor weight_sum = weights.sum(/*dim=*/0);
  const torch::Tensor safe_weight_sum =
      torch::where(weight_sum > 0, weight_sum, torch::ones_like(weight_sum));
  const torch::Tensor safe_outputs =
      torch::where(finite_lse.expand_as(partial_outputs),
                   partial_outputs,
                   torch::zeros_like(partial_outputs));
  const torch::Tensor weighted_output =
      (safe_outputs.to(torch::kFloat32) * weights).sum(/*dim=*/0);
  const torch::Tensor output =
      torch::where((weight_sum > 0).expand_as(weighted_output),
                   weighted_output / safe_weight_sum,
                   torch::zeros_like(weighted_output));
  const torch::Tensor merged_lse = torch::where(
      weight_sum > 0,
      safe_max_lse.squeeze(/*dim=*/0) + torch::log(safe_weight_sum),
      torch::full_like(weight_sum, negative_infinity));
  return {output.to(partial_outputs.scalar_type()),
          merged_lse.squeeze(/*dim=*/-1)};
}

}  // namespace detail

namespace {

using detail::DcpAttentionResult;
using detail::make_decode_actual_seq_lengths;
using detail::merge_dcp_attention_shards;
using detail::resolve_fia_sparse_mode;

void mask_empty_shard_lse(torch::Tensor& partial_lse,
                          const std::vector<int64_t>& local_lengths,
                          const std::vector<int64_t>& query_end_offsets) {
  int64_t previous_query_end = 0;
  for (size_t request_index = 0; request_index < local_lengths.size();
       ++request_index) {
    const int64_t query_end = query_end_offsets[request_index];
    if (local_lengths[request_index] == 0) {
      const int64_t query_length = query_end - previous_query_end;
      partial_lse.narrow(/*dim=*/0, previous_query_end, query_length)
          .fill_(-std::numeric_limits<float>::infinity());
    }
    previous_query_end = query_end;
  }
}

void copy_rank_local_heads(const torch::Tensor& gathered_output,
                           torch::Tensor& output,
                           int32_t dcp_rank) {
  const int64_t local_num_heads = output.size(1);
  const int64_t head_begin = static_cast<int64_t>(dcp_rank) * local_num_heads;
  output.copy_(gathered_output.slice(
      /*dim=*/1, head_begin, head_begin + local_num_heads));
}

DcpAttentionResult run_local_dcp_cache_attention(
    const torch::Tensor& gathered_query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const AttentionMetadata& attention_metadata,
    const std::vector<int64_t>& query_end_offsets,
    const std::vector<int64_t>& global_kv_lengths,
    int64_t num_kv_heads,
    double scale,
    ProcessGroup& dcp_group) {
  const int32_t dcp_size = dcp_group.world_size();
  const int32_t dcp_rank = dcp_group.rank();
  const int64_t block_size = key_cache.size(1);
  const KVShardLayout layout(
      static_cast<int32_t>(block_size), dcp_size, dcp_rank);
  std::vector<int64_t> local_lengths = global_kv_lengths;
  for (int64_t& length : local_lengths) {
    length = layout.local_token_count(length);
  }
  const torch::Tensor key_view =
      key_cache.view({key_cache.size(0), block_size, -1});
  const torch::Tensor value_view =
      value_cache.view({value_cache.size(0), block_size, -1});

  torch::Tensor partial_output;
  torch::Tensor partial_lse;
  if (attention_metadata.paged_attention_tiling_data.defined()) {
    partial_output = torch::empty_like(gathered_query);
    partial_lse = detail::run_fused_infer_attention_graph(
        attention_metadata.acl_graph_task_update_context,
        gathered_query,
        key_view,
        value_view,
        attention_metadata.block_table,
        query_end_offsets,
        local_lengths,
        gathered_query.size(1),
        num_kv_heads,
        scale,
        dcp_size,
        dcp_rank,
        npu::FusedInferAttentionGraphBranch::kDecode,
        partial_output);
  } else {
    std::tie(partial_output, partial_lse) =
        kernel::npu::npu_fused_infer_attention(
            gathered_query,
            key_view,
            value_view,
            /*atten_mask=*/std::nullopt,
            std::make_optional(attention_metadata.block_table),
            query_end_offsets,
            local_lengths,
            gathered_query.size(1),
            num_kv_heads,
            scale,
            block_size,
            /*sparse_mode=*/0,
            "TND",
            /*softmax_lse_flag=*/true);
  }
  if (attention_metadata.paged_attention_tiling_data.defined()) {
    const torch::Tensor empty_shards =
        (attention_metadata.kv_seq_lens <=
         static_cast<int64_t>(dcp_rank) * block_size)
            .view({-1, 1, 1});
    partial_lse.copy_(torch::where(
        empty_shards,
        torch::full_like(partial_lse, -std::numeric_limits<float>::infinity()),
        partial_lse));
  } else {
    mask_empty_shard_lse(partial_lse, local_lengths, query_end_offsets);
  }
  return {partial_output, partial_lse};
}

}  // namespace

void qwen_dcp_decode(const torch::Tensor& query,
                     torch::Tensor& output,
                     const torch::Tensor& key_cache,
                     const torch::Tensor& value_cache,
                     const AttentionMetadata& attention_metadata,
                     int64_t num_kv_heads,
                     double scale,
                     ProcessGroup& dcp_group) {
  const std::vector<int64_t>& global_kv_lengths =
      attention_metadata.kv_seq_lens_host_vec;
  const std::vector<int64_t> query_end_offsets =
      make_decode_actual_seq_lengths(query.size(0));
  const torch::Tensor gathered_query =
      parallel_state::gather(query, &dcp_group, /*dim=*/1);
  const DcpAttentionResult local =
      run_local_dcp_cache_attention(gathered_query,
                                    key_cache,
                                    value_cache,
                                    attention_metadata,
                                    query_end_offsets,
                                    global_kv_lengths,
                                    num_kv_heads,
                                    scale,
                                    dcp_group);
  const DcpAttentionResult merged = merge_dcp_attention_shards(
      dcp_group.allgather_base_sync(local.output.contiguous()),
      dcp_group.allgather_base_sync(local.lse.contiguous()));
  copy_rank_local_heads(merged.output, output, dcp_group.rank());
}

void qwen_dcp_chunked_prefill(const torch::Tensor& query,
                              const torch::Tensor& key,
                              const torch::Tensor& value,
                              torch::Tensor& output,
                              const torch::Tensor& key_cache,
                              const torch::Tensor& value_cache,
                              const AttentionMetadata& attention_metadata,
                              int64_t num_kv_heads,
                              double scale,
                              ProcessGroup& dcp_group) {
  const int64_t token_count = query.size(0);
  const std::vector<int64_t>& global_kv_lengths =
      attention_metadata.kv_seq_lens_host_vec;
  const std::vector<int64_t>& query_end_offsets =
      attention_metadata.q_cu_seq_lens_host_vec;
  std::vector<int64_t> global_context_lengths = global_kv_lengths;
  int64_t previous_query_end = 0;
  for (size_t index = 0; index < query_end_offsets.size(); ++index) {
    const int64_t query_end = query_end_offsets[index];
    global_context_lengths[index] -= query_end - previous_query_end;
    previous_query_end = query_end;
  }
  const torch::Tensor gathered_query =
      parallel_state::gather(query, &dcp_group, /*dim=*/1);
  const int64_t head_size = query.size(2);
  const auto [current_output_raw, current_lse_raw] =
      kernel::npu::npu_fused_infer_attention(
          gathered_query,
          key.view({token_count, num_kv_heads, head_size}),
          value.view({token_count, num_kv_heads, head_size}),
          attention_metadata.fia_attn_mask.defined()
              ? std::make_optional(attention_metadata.fia_attn_mask)
              : std::nullopt,
          /*block_table=*/std::nullopt,
          query_end_offsets,
          query_end_offsets,
          gathered_query.size(1),
          num_kv_heads,
          scale,
          /*block_size=*/0,
          resolve_fia_sparse_mode(attention_metadata),
          "TND",
          /*softmax_lse_flag=*/true,
          attention_metadata.is_causal,
          attention_metadata.fia_pre_tokens,
          attention_metadata.fia_next_tokens);
  if (std::all_of(global_context_lengths.begin(),
                  global_context_lengths.end(),
                  [](int64_t length) { return length == 0; })) {
    copy_rank_local_heads(current_output_raw, output, dcp_group.rank());
    return;
  }

  const DcpAttentionResult context =
      run_local_dcp_cache_attention(gathered_query,
                                    key_cache,
                                    value_cache,
                                    attention_metadata,
                                    query_end_offsets,
                                    global_context_lengths,
                                    num_kv_heads,
                                    scale,
                                    dcp_group);
  const DcpAttentionResult merged = merge_dcp_attention_shards(
      torch::cat({dcp_group.allgather_base_sync(context.output.contiguous()),
                  current_output_raw.unsqueeze(/*dim=*/0)},
                 /*dim=*/0),
      torch::cat({dcp_group.allgather_base_sync(context.lse.contiguous()),
                  current_lse_raw.unsqueeze(/*dim=*/0)},
                 /*dim=*/0));
  copy_rank_local_heads(merged.output, output, dcp_group.rank());
}

}  // namespace xllm::layer
