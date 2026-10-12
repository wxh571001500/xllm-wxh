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

#include "layers/common/kv_shard_batch_metadata.h"

#include <glog/logging.h>

#include "layers/common/attention_metadata.h"

namespace xllm::layer {

torch::Tensor localize_kv_shard_slots(const torch::Tensor& logical_slots,
                                      const KVShardLayout& layout) {
  if (!logical_slots.defined() || logical_slots.numel() == 0) {
    return logical_slots;
  }
  CHECK(logical_slots.scalar_type() == torch::kInt32 ||
        logical_slots.scalar_type() == torch::kInt64)
      << "cache-shard slot mapping must use int32 or int64";
  torch::Tensor valid_slots = logical_slots >= 0;
  torch::Tensor safe_slots = torch::clamp_min(logical_slots, 0);
  torch::Tensor logical_offsets =
      torch::remainder(safe_slots, layout.logical_block_size());
  torch::Tensor owner_ranks =
      torch::floor_divide(logical_offsets, layout.physical_block_size());
  torch::Tensor owned_slots =
      torch::logical_and(valid_slots, owner_ranks == layout.dcp_rank());
  torch::Tensor logical_block_ids =
      torch::floor_divide(safe_slots, layout.logical_block_size());
  torch::Tensor local_offsets =
      torch::remainder(logical_offsets, layout.physical_block_size());
  torch::Tensor local_slots =
      logical_block_ids * layout.physical_block_size() + local_offsets;
  return torch::where(
      owned_slots,
      local_slots,
      torch::full_like(local_slots, KVShardLayout::kInvalidSlot));
}

torch::Tensor localize_kv_shard_context_lens(
    const torch::Tensor& global_context_lens,
    const KVShardLayout& layout) {
  CHECK(global_context_lens.scalar_type() == torch::kInt32 ||
        global_context_lens.scalar_type() == torch::kInt64)
      << "cache-shard context lengths must use int32 or int64";
  torch::Tensor nonnegative_lens = torch::clamp_min(global_context_lens, 0);
  const int64_t logical_block_size = layout.logical_block_size();
  const int64_t physical_block_size = layout.physical_block_size();
  const int64_t rank_start =
      static_cast<int64_t>(layout.dcp_rank()) * physical_block_size;
  torch::Tensor full_logical_blocks =
      torch::floor_divide(nonnegative_lens, logical_block_size);
  torch::Tensor logical_block_remainder =
      torch::remainder(nonnegative_lens, logical_block_size);
  torch::Tensor owned_remainder =
      torch::clamp(logical_block_remainder - rank_start,
                   /*min=*/0,
                   /*max=*/physical_block_size);
  return full_logical_blocks * physical_block_size + owned_remainder;
}

torch::Tensor expand_replicated_block_table(
    const torch::Tensor& logical_block_table,
    int64_t replicated_block_pages) {
  if (!logical_block_table.defined() || logical_block_table.numel() == 0 ||
      replicated_block_pages <= 1) {
    return logical_block_table;
  }
  CHECK_EQ(logical_block_table.dim(), 2)
      << "block table must be [sequences, logical_blocks]";
  CHECK(logical_block_table.scalar_type() == torch::kInt32 ||
        logical_block_table.scalar_type() == torch::kInt64)
      << "block table must use int32 or int64";
  const int64_t rows = logical_block_table.size(0);
  const int64_t columns = logical_block_table.size(1);
  torch::Tensor page_offsets =
      torch::arange(replicated_block_pages, logical_block_table.options());
  torch::Tensor expanded =
      logical_block_table.unsqueeze(-1) * replicated_block_pages + page_offsets;
  expanded =
      torch::where(logical_block_table.unsqueeze(-1) >= 0,
                   expanded,
                   torch::full_like(expanded, KVShardLayout::kInvalidSlot));
  return expanded.reshape({rows, columns * replicated_block_pages})
      .contiguous();
}

std::shared_ptr<const KVShardBatchMetadata> build_kv_shard_batch_metadata(
    const AttentionMetadata& attention_metadata,
    const KVShardLayout& layout) {
  CHECK(attention_metadata.slot_mapping.defined())
      << "cache-shard batch metadata requires slot mapping";
  auto metadata = std::make_shared<KVShardBatchMetadata>();
  metadata->kv_split_size = layout.dcp_size();
  metadata->kv_split_rank = layout.dcp_rank();
  metadata->local_slot_mapping =
      localize_kv_shard_slots(attention_metadata.slot_mapping, layout);
  return metadata;
}

}  // namespace xllm::layer
