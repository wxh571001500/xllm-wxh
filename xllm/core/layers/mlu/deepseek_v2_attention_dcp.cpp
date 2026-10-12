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

#include <glog/logging.h>

#include <tuple>
#include <utility>
#include <vector>

#include "core/layers/mlu/dcp_batch_metadata.h"
#include "core/layers/mlu/deepseek_v2_attention.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace {

inline bool needs_local_topk_view(bool enable_dcp_decode,
                                  bool enable_dcp_chunked_prefill) {
  return enable_dcp_chunked_prefill || enable_dcp_decode;
}

inline bool needs_tp_query_gather(bool dcp_spans_tp,
                                  bool enable_dcp_decode,
                                  bool enable_dcp_chunked_prefill) {
  return dcp_spans_tp && (enable_dcp_chunked_prefill || enable_dcp_decode);
}

parallel_state::GatherAsyncCtx launch_tp_query_gather(
    const torch::Tensor& q_input,
    ProcessGroup* tp_group) {
  parallel_state::GatherAsyncCtx ctx;
  ctx.input = q_input;
  std::vector<int64_t> stacked_shape = q_input.sizes().vec();
  stacked_shape.insert(stacked_shape.begin(), tp_group->world_size());
  ctx.stacked = torch::empty(stacked_shape, q_input.options());
  ctx.work = tp_group->allgather_base_async(q_input, ctx.stacked);
  return ctx;
}

torch::Tensor finish_tp_query_gather(parallel_state::GatherAsyncCtx ctx) {
  if (ctx.work.defined()) {
    ctx.work->wait();
  }
  return torch::cat(ctx.stacked.unbind(0), /*dim=*/1).contiguous();
}

}  // namespace

torch::Tensor DeepseekV2AttentionImpl::remap_prefill_blocks(
    const torch::Tensor& sorted_slots,
    const torch::Tensor& sorted_rows,
    const AttentionMetadata& metadata) const {
  const int64_t token_count = sorted_slots.numel();
  CHECK_GT(token_count, 0) << "Empty prefill must bypass attention";
  CHECK_EQ(sorted_rows.numel(), token_count);
  CHECK_EQ(sorted_slots.scalar_type(), torch::kInt64);
  CHECK_EQ(sorted_rows.scalar_type(), torch::kInt64);
  CHECK_EQ(metadata.block_table.dim(), 2);
  CHECK_EQ(metadata.block_table.size(0), metadata.kv_seq_lens.numel());

  const torch::Tensor& topk = metadata.block_table;
  torch::Tensor columns = torch::arange(topk.size(1), topk.options());
  torch::Tensor valid =
      (columns < metadata.kv_seq_lens.unsqueeze(1)) & (topk >= 0);
  torch::Tensor search_slots = topk.to(torch::kInt64).contiguous();
  torch::Tensor positions = torch::searchsorted(sorted_slots, search_slots);
  torch::Tensor safe_positions = positions.clamp_max(token_count - 1);
  torch::Tensor matches =
      (positions < token_count) &
      (sorted_slots.index_select(0, safe_positions.flatten()).view_as(topk) ==
       search_slots);
  // Missing effective slots deliberately reach index_select's device bounds
  // check. Clamping them to row zero would silently discard historical K.
  torch::Tensor checked_positions =
      torch::where(valid, torch::where(matches, positions, token_count), 0);
  torch::Tensor scratch_rows =
      sorted_rows.index_select(0, checked_positions.flatten()).view_as(topk);
  return torch::where(valid, scratch_rows, torch::where(topk < 0, -1, 0))
      .to(topk.scalar_type());
}

std::pair<KVCache, AttentionMetadata>
DeepseekV2AttentionImpl::build_prefill_cache(
    const torch::Tensor& k_input,
    const torch::Tensor& k_cache,
    const torch::Tensor& sorted_slots,
    const torch::Tensor& sorted_rows,
    const AttentionMetadata& metadata,
    const torch::Tensor& mapped_blocks) const {
  const int64_t token_count = k_input.size(0);
  CHECK_GT(token_count, 0) << "Empty prefill must bypass attention";
  CHECK_EQ(k_input.dim(), 2);
  CHECK_EQ(k_cache.dim(), 4);
  CHECK_EQ(k_cache.size(1), 1);
  CHECK_EQ(k_cache.size(3), k_input.size(1));
  CHECK_EQ(sorted_slots.numel(), token_count);
  CHECK_EQ(sorted_rows.numel(), token_count);
  const int64_t block_size = k_cache.size(2);
  CHECK_GT(block_size, 0);

  AttentionMetadata compact_metadata = metadata;
  compact_metadata.block_table =
      mapped_blocks.defined()
          ? mapped_blocks
          : remap_prefill_blocks(sorted_slots, sorted_rows, metadata);

  torch::Tensor scratch =
      torch::zeros({(token_count + block_size - 1) / block_size,
                    1,
                    block_size,
                    k_input.size(1)},
                   k_input.options());
  xllm::kernel::ReshapePagedCacheParams write_params;
  write_params.key = k_input.unsqueeze(1);
  write_params.k_cache = scratch;
  write_params.slot_mapping =
      torch::arange(token_count, metadata.slot_mapping.options());
  xllm::kernel::reshape_paged_cache(write_params);
  return {KVCache(KVCacheTensors{std::move(scratch), torch::Tensor()}),
          std::move(compact_metadata)};
}

DsaTopkState DeepseekV2AttentionImpl::resolve_dcp_local_topk(
    const DsaTopkState& global_topk,
    const DsaTopkTransfer* topk_transfer) const {
  if (topk_transfer != nullptr) {
    const DsaTopkState* relayed_view = topk_transfer->localized_input();
    if (relayed_view != nullptr) {
      return *relayed_view;
    }
  }
  CHECK(dcp_decode_context_ != nullptr);
  return dcp_decode_context_->localize_topk(global_topk);
}

torch::Tensor DeepseekV2AttentionImpl::run_dcp_paged_attention_local(
    const torch::Tensor& q_input,
    const DsaTopkState& local_topk,
    std::optional<parallel_state::GatherAsyncCtx>& query_gather,
    KVCache& kv_cache,
    const AttentionMetadata& base_metadata) {
  CHECK(dcp_decode_context_ != nullptr);
  const HeadInfo& heads = active_heads();
  const int64_t token_count = q_input.size(0);
  CHECK_EQ(local_topk.block_tables().size(0), token_count);

  AttentionMetadata local_metadata =
      build_mla_attention_metadata(base_metadata, local_topk);
  local_metadata.is_prefill = false;
  local_metadata.is_chunked_prefill = false;

  torch::Tensor query = q_input;
  const bool gathers_tp_heads = dcp_spans_tp_ && !use_replicated_attn_weights();
  if (gathers_tp_heads) {
    CHECK(query_gather.has_value())
        << "DCP local decode attention requires the early-launched TP query "
           "gather";
    query = finish_tp_query_gather(std::move(query_gather.value()));
    query_gather.reset();
  }
  Attention& local_attention = gathers_tp_heads ? dcp_full_head_attn_ : attn_;
  const int64_t attention_heads =
      gathers_tp_heads ? full_heads().attn : heads.attn;

  torch::Tensor output = torch::empty(
      {token_count, attention_heads * kv_lora_rank_}, q_input.options());
  std::optional<torch::Tensor> output_lse = std::nullopt;
  local_attention->decoder_forward(query,
                                   output,
                                   output_lse,
                                   kv_cache.get_k_cache(),
                                   std::nullopt,
                                   local_metadata,
                                   kv_cache.get_k_cache_scale(),
                                   std::nullopt,
                                   /*return_lse=*/true);
  CHECK(output_lse.has_value())
      << "DCP local decode attention requires local sparse LSE";
  CHECK_EQ(output_lse->dim(), 3)
      << "DCP local decode LSE must be [query, heads, 1]";
  CHECK_EQ(output_lse->size(0), token_count);
  CHECK_EQ(output_lse->size(1), attention_heads);
  CHECK_EQ(output_lse->size(2), 1);

  // Head reduce-scatter is only valid when the DCP group is exactly the TP
  // group (no PCP replication); otherwise heads are merged with an all-reduce
  // and redistributed by tp_rank_ below.
  const bool head_sharded =
      gathers_tp_heads &&
      tp_group_->world_size() == dcp_decode_context_->world_size();
  torch::Tensor merged_output = dcp_decode_context_->merge(
      output.view({token_count, 1, attention_heads, kv_lora_rank_}),
      output_lse.value(),
      base_metadata.slot_mapping,
      /*head_sharded=*/head_sharded);
  if (gathers_tp_heads && !head_sharded) {
    const int64_t first_head = tp_rank_ * heads.attn;
    const int64_t last_head = first_head + heads.attn;
    merged_output =
        merged_output.slice(/*dim=*/2, first_head, last_head).contiguous();
  }
  return merged_output;
}

torch::Tensor DeepseekV2AttentionImpl::run_dcp_paged_attention(
    const torch::Tensor& q_input,
    const std::optional<DsaTopkState>& local_topk,
    std::optional<parallel_state::GatherAsyncCtx>& query_gather,
    KVCache& kv_cache,
    const AttentionMetadata& base_metadata) {
  CHECK(local_topk.has_value())
      << "DCP local decode attention requires the rank-local top-k view";
  return run_dcp_paged_attention_local(
      q_input, local_topk.value(), query_gather, kv_cache, base_metadata);
}

torch::Tensor DeepseekV2AttentionImpl::run_dcp_chunked_prefill_attention(
    const torch::Tensor& q_input,
    const DsaTopkState& local_topk,
    std::optional<parallel_state::GatherAsyncCtx>& query_gather,
    KVCache& kv_cache,
    const AttentionMetadata& local_cache_metadata,
    const torch::Tensor& global_slot_mapping) {
  CHECK(dcp_decode_context_ != nullptr);
  const HeadInfo& heads = active_heads();
  AttentionMetadata local_metadata =
      build_mla_attention_metadata(local_cache_metadata, local_topk);
  torch::Tensor query = q_input;
  const bool gathers_tp_heads = dcp_spans_tp_ && !use_replicated_attn_weights();
  if (gathers_tp_heads) {
    CHECK(query_gather.has_value())
        << "DCP chunked prefill requires the early-launched TP query gather";
    query = finish_tp_query_gather(std::move(query_gather.value()));
    query_gather.reset();
  }
  torch::Tensor unused_key;
  torch::Tensor unused_value;
  Attention& local_attention = gathers_tp_heads ? dcp_full_head_attn_ : attn_;
  auto [local_output, local_lse] = local_attention(local_metadata,
                                                   query,
                                                   unused_key,
                                                   unused_value,
                                                   kv_cache,
                                                   /*return_lse=*/true);
  const int64_t attention_heads =
      gathers_tp_heads ? full_heads().attn : heads.attn;
  const bool head_sharded =
      gathers_tp_heads &&
      tp_group_->world_size() == dcp_decode_context_->world_size();
  // Only the global mapping marks graph-padding rows. The localized mapping
  // contains -1 for valid queries whose KV slots belong to another DCP rank.
  torch::Tensor merged_output = dcp_decode_context_->merge(
      local_output.view({q_input.size(0), 1, attention_heads, kv_lora_rank_}),
      local_lse.value(),
      global_slot_mapping,
      /*head_sharded=*/head_sharded);
  if (!gathers_tp_heads) {
    return merged_output;
  }
  if (!head_sharded) {
    const int64_t first_head = tp_rank_ * heads.attn;
    const int64_t last_head = first_head + heads.attn;
    return merged_output.slice(/*dim=*/2, first_head, last_head).contiguous();
  }
  return merged_output;
}

torch::Tensor DeepseekV2AttentionImpl::forward_dcp(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    bool is_prefill_or_chunked_prefill,
    DsaTopkTransfer* topk_transfer) {
  CHECK(enable_mla_cache_sharding_);
  CHECK(!attn_metadata.is_dummy);
  CHECK(dcp_decode_context_ != nullptr);

  if (attn_metadata.is_prefill && hidden_states.size(0) == 0) {
    CHECK_EQ(attn_metadata.slot_mapping.numel(), 0)
        << "Empty prefill cannot supply K slots";
    return torch::empty_like(hidden_states);
  }

  const HeadInfo& heads = active_heads();
  const bool enable_dcp_decode = !is_prefill_or_chunked_prefill;
  const bool enable_dcp_chunked_prefill = attn_metadata.is_chunked_prefill;
  const bool enable_dcp_paged_attention =
      enable_dcp_decode || enable_dcp_chunked_prefill;
  const bool enable_dcp_prefill_cache_write = attn_metadata.is_prefill;
  AttentionMetadata cache_metadata = attn_metadata;
  const std::shared_ptr<const KVShardBatchMetadata>& shard_metadata =
      attn_metadata.kv_shard_batch_metadata;
  if (shard_metadata != nullptr) {
    CHECK(shard_metadata->local_slot_mapping.defined())
        << "cache-shard batch metadata requires localized slot mapping";
    cache_metadata.slot_mapping = shard_metadata->local_slot_mapping;
  } else {
    cache_metadata.slot_mapping =
        dcp_decode_context_->localize_slots(attn_metadata.slot_mapping);
  }

  torch::Tensor q;
  torch::Tensor q_norm;
  torch::Tensor q_input = torch::empty(
      {hidden_states.size(0), heads.attn, kv_lora_rank_ + qk_rope_head_dim_},
      hidden_states.options());
  torch::Tensor latent_cache;
  torch::Tensor k_cache = kv_cache.get_k_cache();
  std::optional<torch::Tensor> k_cache_scale = kv_cache.get_k_cache_scale();
  const bool enable_fused_qkv =
      use_fused_mla_qkv_ && !is_prefill_or_chunked_prefill;
  const bool use_prompt_rope = attn_metadata.is_prefill;
  prepare_mla_query_side(q,
                         q_norm,
                         q_input,
                         hidden_states,
                         positions,
                         cache_metadata,
                         enable_fused_qkv,
                         use_prompt_rope);
  q_input = q_input.view({q_input.size(0), -1});

  std::optional<parallel_state::GatherAsyncCtx> query_gather;
  if (needs_tp_query_gather(dcp_spans_tp_ && !use_replicated_attn_weights(),
                            enable_dcp_decode,
                            enable_dcp_chunked_prefill)) {
    query_gather = launch_tp_query_gather(q_input, tp_group_);
  }

  const DsaTopkState* external_topk =
      topk_transfer != nullptr ? topk_transfer->input() : nullptr;

  std::optional<DcpIndexerGatherAsyncCtx> candidate_gather;
  int64_t candidate_topk = 0;
  std::optional<DsaTopkState> topk_state;
  if (external_topk != nullptr) {
    topk_state = *external_topk;
  } else if (has_indexer_) {
    AttentionMetadata local_indexer_metadata = attn_metadata;
    local_indexer_metadata.slot_mapping = cache_metadata.slot_mapping;
    local_indexer_metadata.block_table = attn_metadata.block_table;
    torch::Tensor index_cache = kv_cache.get_index_cache();
    DcpIndexerLocalCandidates candidates;
    if (enable_dcp_decode) {
      if (shard_metadata != nullptr &&
          shard_metadata->local_indexer_context_lens.defined()) {
        local_indexer_metadata.kv_seq_lens =
            shard_metadata->local_indexer_context_lens;
      } else {
        local_indexer_metadata.kv_seq_lens = localize_kv_shard_context_lens(
            attn_metadata.kv_seq_lens, dcp_decode_context_->layout());
      }
      candidates = indexer_->forward_dcp_local(hidden_states,
                                               q_norm,
                                               positions,
                                               index_cache,
                                               local_indexer_metadata);
    } else {
      KVShardCausalSelectorMetadata causal_selector;
      if (shard_metadata != nullptr &&
          shard_metadata->causal_selector.block_table.defined()) {
        causal_selector = shard_metadata->causal_selector;
      } else {
        causal_selector = build_kv_shard_causal_selector_metadata(
            attn_metadata, dcp_decode_context_->layout());
      }
      AttentionMetadata selector_metadata = local_indexer_metadata;
      selector_metadata.q_cu_seq_lens = causal_selector.q_cu_seq_lens;
      selector_metadata.kv_cu_seq_lens = causal_selector.q_cu_seq_lens;
      selector_metadata.kv_seq_lens = causal_selector.local_context_lens;
      selector_metadata.block_table = causal_selector.block_table;
      selector_metadata.max_query_len = 1;
      selector_metadata.max_seq_len = index_topk_;
      // The decode-view selector does not consume total_kv_len. Keeping this
      // host metadata scalar avoids a device-to-host synchronization here.
      selector_metadata.total_kv_len = 0;
      selector_metadata.is_prefill = false;
      selector_metadata.is_chunked_prefill = false;
      candidates = indexer_->forward_dcp_local_prefill(hidden_states,
                                                       q_norm,
                                                       positions,
                                                       index_cache,
                                                       local_indexer_metadata,
                                                       selector_metadata);
    }
    candidate_gather = dcp_decode_context_->launch_indexer_candidate_gather(
        candidates.scores, candidates.global_slots);
    candidate_topk = candidates.global_slots.size(1);
  } else {
    CHECK(!enable_lighting_indexer_)
        << "Shared DSA layer requires externally supplied top-k metadata.";
  }
  prepare_mla_latent_side(latent_cache,
                          k_cache,
                          k_cache_scale,
                          hidden_states,
                          positions,
                          cache_metadata,
                          enable_fused_qkv,
                          use_prompt_rope);
  torch::Tensor v_input = latent_cache.slice(-1, 0, kv_lora_rank_);
  torch::Tensor k_input = latent_cache;
  k_input = k_input.view({k_input.size(0), -1});
  v_input = v_input.view({v_input.size(0), -1});
  if (!enable_dcp_prefill_cache_write) {
    update_mla_k_cache(k_input,
                       attn_metadata,
                       kv_cache,
                       k_cache_scale,
                       is_prefill_or_chunked_prefill ||
                           (enable_dcp_paged_attention && !enable_fused_qkv),
                       cache_metadata.slot_mapping);
  }
  if (candidate_gather.has_value()) {
    topk_state = dcp_decode_context_->finish_indexer_candidate_merge(
        std::move(candidate_gather.value()),
        candidate_topk,
        attn_metadata.slot_mapping);
  }

  std::optional<DsaTopkState> local_topk;
  if (topk_state.has_value() &&
      needs_local_topk_view(enable_dcp_decode, enable_dcp_chunked_prefill)) {
    local_topk = resolve_dcp_local_topk(topk_state.value(), topk_transfer);
  }
  if (topk_transfer != nullptr) {
    topk_transfer->complete(topk_state, local_topk);
  }
  AttentionMetadata kernel_metadata =
      build_mla_attention_metadata(attn_metadata, topk_state);
  if (enable_dcp_paged_attention) {
    CHECK(topk_state.has_value())
        << "GLM-5.2 DCP paged attention requires global DSA top-k metadata";
  }

  torch::Tensor attn_output;
  if (enable_dcp_decode) {
    attn_output =
        run_dcp_paged_attention(
            q_input, local_topk, query_gather, kv_cache, attn_metadata)
            .view({q_input.size(0), heads.attn * kv_lora_rank_});
  } else if (enable_dcp_chunked_prefill) {
    attn_output = run_dcp_chunked_prefill_attention(q_input,
                                                    local_topk.value(),
                                                    query_gather,
                                                    kv_cache,
                                                    cache_metadata,
                                                    attn_metadata.slot_mapping)
                      .view({q_input.size(0), heads.attn * kv_lora_rank_});
  } else {
    torch::Tensor sorted_slots;
    torch::Tensor sorted_rows;
    if (shard_metadata != nullptr &&
        shard_metadata->prefill_sorted_slots.defined()) {
      sorted_slots = shard_metadata->prefill_sorted_slots;
      sorted_rows = shard_metadata->prefill_sorted_rows;
    } else {
      // Standalone attention callers can omit the model's batch metadata.
      std::tie(sorted_slots, sorted_rows) =
          build_prefill_slot_order(attn_metadata.slot_mapping);
      sorted_slots = sorted_slots.to(torch::kInt64);
    }
    auto [attention_cache, compact_metadata] = build_prefill_cache(
        k_input, k_cache, sorted_slots, sorted_rows, kernel_metadata);
    std::tie(attn_output, std::ignore) =
        attn_(compact_metadata, q_input, k_input, v_input, attention_cache);
  }

  if (enable_dcp_prefill_cache_write) {
    update_mla_k_cache(k_input,
                       attn_metadata,
                       kv_cache,
                       k_cache_scale,
                       /*is_prefill_phase=*/true,
                       cache_metadata.slot_mapping);
  }
  return project_output(attn_output, heads);
}

}  // namespace layer
}  // namespace xllm
