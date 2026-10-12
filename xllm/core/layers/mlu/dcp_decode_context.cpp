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

#include "layers/mlu/dcp_decode_context.h"

#include <glog/logging.h>

#include <cstdint>
#include <utility>

#include "framework/parallel_state/process_group.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "layers/common/kv_shard_batch_metadata.h"

namespace xllm::layer {

namespace {

torch::Tensor merge_dcp_attention(torch::Tensor local_output,
                                  const torch::Tensor& local_lse,
                                  const torch::Tensor& slot_mapping,
                                  ProcessGroup& dcp_group,
                                  bool head_sharded) {
  const int64_t world_size = dcp_group.world_size();
  if (world_size <= 1) {
    return local_output;
  }

  // Only the cheap per-rank LSE is gathered; the [B, H, V] output stays local
  // until the single cross-rank reduction below.
  torch::Tensor lses =
      dcp_group.allgather_base_sync(local_lse.contiguous());  // [N, B, H, 1]
  lses = lses.squeeze(-1).contiguous();                       // [N, B, H]

  torch::Tensor out = local_output.squeeze(1).contiguous();  // [B, H, V]
  const int64_t batch_size = out.size(0);
  const int64_t heads = out.size(1);
  const int64_t head_dim = out.size(2);
  CHECK(out.numel() > 0);
  CHECK_EQ(slot_mapping.numel(), batch_size)
      << "DCP attention slot mappings must match output rows";

  if (head_sharded) {
    // DCP spans the TP ranks, so heads are distributed: reduce-scatter over the
    // head dim (dim 0 of the fused output) and keep this rank's own H/dcp
    // heads, avoiding the full all-gather of `out`.
    CHECK_EQ(heads % world_size, 0)
        << "DCP head-sharded merge requires heads divisible by dcp size";
    torch::Tensor head_major =
        torch::empty({heads, batch_size, head_dim}, out.options());
    kernel::mlu::dcp_correct_attn_transpose(
        out, lses, slot_mapping, dcp_group.rank(), head_major);
    const int64_t chunk_heads = head_major.size(0) / world_size;
    torch::Tensor reduced =
        torch::empty({chunk_heads, batch_size, head_dim}, out.options());
    dcp_group.reduce_scatter(head_major, reduced);
    return reduced.movedim(0, 1).contiguous().unsqueeze(/*dim=*/1);
  }
  // Heads are replicated across the DCP ranks: sum the weighted partials.
  kernel::mlu::dcp_correct_attn_out(out, lses, slot_mapping, dcp_group.rank());
  dcp_group.allreduce(out);
  return out.unsqueeze(/*dim=*/1);
}

DsaTopkState merge_indexer_candidates(const torch::Tensor& scores,
                                      const torch::Tensor& global_slots,
                                      int64_t topk,
                                      const torch::Tensor& slot_mapping) {
  const int64_t query_count = scores.size(1);
  torch::Tensor output_slots =
      torch::empty({query_count, topk}, global_slots.options());
  torch::Tensor context_lens =
      torch::empty({query_count}, global_slots.options());
  kernel::mlu::dcp_merge_topk(
      scores, global_slots, slot_mapping, topk, output_slots, context_lens);
  return DsaTopkState(std::move(output_slots), std::move(context_lens));
}

}  // namespace

DcpDecodeContext::DcpDecodeContext(KVShardLayout layout,
                                   ProcessGroup* dcp_group)
    : layout_(std::move(layout)), dcp_group_(dcp_group) {}

int32_t DcpDecodeContext::world_size() const {
  return dcp_group_ == nullptr ? 1 : dcp_group_->world_size();
}

torch::Tensor DcpDecodeContext::localize_slots(
    const torch::Tensor& global_slots) const {
  return localize_kv_shard_slots(global_slots, layout_);
}

// Fused local top-k localization (Triton): de-interleaves the gathered global
// top-k slots into rank-local slots, filters ownership, and compacts owned
// entries to the row front with 0 padding — one kernel instead of a torch
// cumsum/scatter chain.
DsaTopkState DcpDecodeContext::localize_topk(
    const DsaTopkState& global_state) const {
  const torch::Tensor& global_table = global_state.block_tables();
  const torch::Tensor& global_context_lens = global_state.context_lens();
  CHECK_EQ(global_table.dim(), 2) << "Flatten MTP TopK before localization";
  const int64_t rows = global_table.size(0);
  const int64_t width = global_table.size(1);
  torch::Tensor local_table =
      torch::empty({rows, width}, global_table.options());
  torch::Tensor local_lens =
      torch::empty({rows}, global_context_lens.options());
  kernel::mlu::dcp_localize_topk(
      global_table,
      global_context_lens,
      static_cast<int32_t>(layout_.dcp_rank()),
      static_cast<int32_t>(layout_.dcp_size()),
      static_cast<int32_t>(layout_.physical_block_size()),
      local_table,
      local_lens);

  return DsaTopkState(std::move(local_table), std::move(local_lens));
}

DcpIndexerGatherAsyncCtx DcpDecodeContext::launch_indexer_candidate_gather(
    const torch::Tensor& local_scores,
    const torch::Tensor& local_global_slots) const {
  return launch_dcp_indexer_candidate_gather(
      local_scores, local_global_slots, dcp_group_);
}

DsaTopkState DcpDecodeContext::finish_indexer_candidate_merge(
    DcpIndexerGatherAsyncCtx&& gather,
    int64_t topk,
    const torch::Tensor& slot_mapping) const {
  const DcpIndexerGatheredCandidates gathered =
      finish_dcp_indexer_candidate_gather(std::move(gather));
  return merge_indexer_candidates(
      gathered.scores, gathered.global_slots, topk, slot_mapping);
}

torch::Tensor DcpDecodeContext::merge(const torch::Tensor& local_output,
                                      const torch::Tensor& local_lse,
                                      const torch::Tensor& slot_mapping,
                                      bool head_sharded) const {
  if (dcp_group_ == nullptr || dcp_group_->world_size() == 1) {
    return local_output;
  }
  return merge_dcp_attention(
      local_output, local_lse, slot_mapping, *dcp_group_, head_sharded);
}

}  // namespace xllm::layer
