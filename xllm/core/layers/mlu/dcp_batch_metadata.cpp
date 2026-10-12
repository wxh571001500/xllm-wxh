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

#include "layers/mlu/dcp_batch_metadata.h"

#include <glog/logging.h>

#include <tuple>
#include <utility>

#include "layers/common/attention_metadata.h"

namespace xllm::layer {

std::pair<torch::Tensor, torch::Tensor> build_prefill_slot_order(
    const torch::Tensor& source_slots) {
  CHECK_EQ(source_slots.dim(), 1);
  CHECK(source_slots.scalar_type() == torch::kInt32 ||
        source_slots.scalar_type() == torch::kInt64);
  auto [slots, rows] = torch::sort(source_slots);
  const int64_t token_count = slots.numel();
  if (token_count > 1) {
    torch::Tensor repeated =
        (slots.slice(0, 1) == slots.slice(0, 0, -1)) & (slots.slice(0, 1) >= 0);
    repeated = torch::cat({torch::zeros({1}, repeated.options()), repeated});
    torch::Tensor indices = torch::arange(token_count, rows.options());
    // index_select checks bounds on MLU without copying a predicate to CPU.
    // A duplicate source has no unique K row and must fail, not pick a winner.
    rows = rows.index_select(0, torch::where(repeated, token_count, indices));
  }
  return {std::move(slots), std::move(rows)};
}

KVShardCausalSelectorMetadata build_kv_shard_causal_selector_metadata(
    const AttentionMetadata& attention_metadata,
    const KVShardLayout& layout) {
  CHECK(attention_metadata.q_cu_seq_lens.defined())
      << "cache-shard causal selector requires query cumulative lengths";
  CHECK(attention_metadata.kv_cu_seq_lens.defined())
      << "cache-shard causal selector requires KV cumulative lengths";
  CHECK(attention_metadata.block_table.defined())
      << "cache-shard causal selector requires a block table";
  CHECK(attention_metadata.slot_mapping.defined())
      << "cache-shard causal selector requires slot mapping";
  CHECK_EQ(attention_metadata.q_cu_seq_lens.dim(), 1)
      << "cache-shard causal selector query lengths must be one-dimensional";
  CHECK_EQ(attention_metadata.kv_cu_seq_lens.dim(), 1)
      << "cache-shard causal selector KV lengths must be one-dimensional";
  CHECK_EQ(attention_metadata.q_cu_seq_lens.numel(),
           attention_metadata.kv_cu_seq_lens.numel())
      << "cache-shard causal selector query and KV batches must match";
  CHECK_EQ(attention_metadata.block_table.size(0),
           attention_metadata.q_cu_seq_lens.numel() - 1)
      << "cache-shard causal selector block-table batch must match lengths";
  CHECK_EQ(attention_metadata.q_cu_seq_lens.scalar_type(), torch::kInt32)
      << "cache-shard causal selector query lengths must be int32";
  CHECK_EQ(attention_metadata.kv_cu_seq_lens.scalar_type(), torch::kInt32)
      << "cache-shard causal selector KV lengths must be int32";

  torch::Tensor query_lens = torch::diff(attention_metadata.q_cu_seq_lens);
  torch::Tensor kv_lens = torch::diff(attention_metadata.kv_cu_seq_lens);
  torch::Tensor prefix_lens = kv_lens - query_lens;
  const int64_t token_count = attention_metadata.slot_mapping.numel();

  torch::Tensor token_prefix_lens =
      torch::repeat_interleave(prefix_lens, query_lens, /*dim=*/0);
  torch::Tensor token_query_starts = torch::repeat_interleave(
      attention_metadata.q_cu_seq_lens.slice(/*dim=*/0,
                                             /*start=*/0,
                                             /*end=*/-1),
      query_lens,
      /*dim=*/0);
  torch::Tensor token_offsets =
      torch::arange(token_count, attention_metadata.q_cu_seq_lens.options());
  torch::Tensor global_context_lens =
      token_prefix_lens + token_offsets - token_query_starts + 1;
  torch::Tensor query_block_table = attention_metadata.block_table
                                        .repeat_interleave(query_lens,
                                                           /*dim=*/0)
                                        .contiguous();
  torch::Tensor selector_q_cu_seq_lens = torch::arange(
      token_count + 1, attention_metadata.q_cu_seq_lens.options());
  return KVShardCausalSelectorMetadata{
      std::move(query_block_table),
      localize_kv_shard_context_lens(global_context_lens, layout),
      std::move(selector_q_cu_seq_lens)};
}

std::shared_ptr<const KVShardBatchMetadata> build_mlu_shard_metadata(
    const AttentionMetadata& attention_metadata,
    const KVShardLayout& layout) {
  auto metadata = std::make_shared<KVShardBatchMetadata>(
      *build_kv_shard_batch_metadata(attention_metadata, layout));
  if (attention_metadata.is_dummy) {
    return metadata;
  }
  if (attention_metadata.kv_seq_lens.defined()) {
    metadata->local_indexer_context_lens =
        localize_kv_shard_context_lens(attention_metadata.kv_seq_lens, layout);
  }
  if (attention_metadata.is_prefill && !attention_metadata.is_dummy) {
    std::tie(metadata->prefill_sorted_slots, metadata->prefill_sorted_rows) =
        build_prefill_slot_order(attention_metadata.slot_mapping);
    metadata->prefill_sorted_slots =
        metadata->prefill_sorted_slots.to(torch::kInt64);
  }
  // The causal selector is consumed only by DCP prefill paths. Decode graph
  // buckets pad sequence metadata independently of token rows, so building it
  // there is both unnecessary and can produce a non-token-shaped selector.
  if ((attention_metadata.is_prefill ||
       attention_metadata.is_chunked_prefill) &&
      attention_metadata.q_cu_seq_lens.defined() &&
      attention_metadata.kv_cu_seq_lens.defined() &&
      attention_metadata.block_table.defined() &&
      attention_metadata.slot_mapping.defined()) {
    metadata->causal_selector =
        build_kv_shard_causal_selector_metadata(attention_metadata, layout);
  }
  return metadata;
}

}  // namespace xllm::layer
