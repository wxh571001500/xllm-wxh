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

#include "framework/parallel_state/parallel_state.h"
#include "layers/mlu/deepseek_v2_attention.h"

namespace xllm::layer {

torch::Tensor DeepseekV2AttentionImpl::forward_glm5_next_pcp(
    const torch::Tensor& local_positions,
    const torch::Tensor& local_hidden_states,
    const AttentionMetadata& global_metadata,
    const AttentionMetadata& local_metadata,
    KVCache& kv_cache,
    const glm5_next_pcp::Context& context) {
  // PCP is admitted only when every CP rank owns tokens.
  CHECK_GT(local_hidden_states.size(0), 0);
  torch::Tensor local_k = kv_a_proj_with_mqa_(local_hidden_states);
  decode_kv_pre_base(
      local_k, local_positions, local_metadata, /*use_prompt_rope=*/false);
  auto pending_k = parallel_state::launch_gather(
      local_k, context.cp_group, context.geometry.tokens_per_rank);
  QueryPrep query = prep_query(local_hidden_states, active_heads());
  torch::Tensor q_input = torch::empty({local_hidden_states.size(0),
                                        active_heads().attn,
                                        kv_lora_rank_ + qk_rope_head_dim_},
                                       local_hidden_states.options());
  fill_q_input(q_input,
               query.q,
               local_positions,
               local_metadata,
               /*use_prompt_rope=*/false);
  q_input = q_input.flatten(1);

  torch::Tensor index_cache = kv_cache.get_index_cache();
  torch::Tensor tail_cache = kv_cache.get_kpool_tail();
  auto [block_tables, context_lens] =
      glm5_next_kpool_indexer_->forward_pcp(local_hidden_states,
                                            query.q_norm,
                                            local_positions,
                                            context.global_positions,
                                            index_cache,
                                            tail_cache,
                                            global_metadata,
                                            local_metadata,
                                            context);
  torch::Tensor global_k =
      glm5_next_pcp::finish_gather_restore(std::move(pending_k), context);
  update_mla_k_cache(global_k,
                     global_metadata,
                     kv_cache,
                     kv_cache.get_k_cache_scale(),
                     /*is_prefill_phase=*/true);
  DsaTopkState topk_state(block_tables, context_lens);
  AttentionMetadata kernel_metadata =
      build_mla_attention_metadata(local_metadata, topk_state);
  torch::Tensor attention_output;
  torch::Tensor global_v = global_k.slice(-1, 0, kv_lora_rank_);
  std::tie(attention_output, std::ignore) =
      attn_(kernel_metadata, q_input, global_k, global_v, kv_cache);
  torch::Tensor output = project_output(attention_output, active_heads());
  if (!use_replicated_attn_weights()) {
    output = parallel_state::reduce(output, tp_group_);
  }
  return output;
}

}  // namespace xllm::layer
