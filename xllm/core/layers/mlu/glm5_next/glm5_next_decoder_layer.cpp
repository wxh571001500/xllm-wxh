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

#include "layers/mlu/glm5_next/glm5_next_decoder_layer.h"

#include <glog/logging.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "framework/parallel_state/parallel_state.h"

namespace xllm {
namespace layer {
namespace {

std::string layer_type_at(const std::vector<std::string>& types,
                          int32_t layer_id,
                          const std::string& fallback) {
  CHECK_GE(layer_id, 0);
  if (layer_id < static_cast<int32_t>(types.size())) {
    return types[static_cast<size_t>(layer_id)];
  }
  return fallback;
}

}  // namespace

Glm5NextLayerRole resolve_glm5_next_layer_role(const ModelArgs& args,
                                               int32_t layer_id) {
  const std::string attention_type =
      layer_type_at(args.layer_types(), layer_id, "deepseek_sparse_attention");
  CHECK(attention_type == "linear_attention" ||
        attention_type == "deepseek_sparse_attention" ||
        attention_type == "attention" || attention_type == "full_attention")
      << "Unsupported GLM5-Next attention layer type at layer " << layer_id
      << ": " << attention_type;

  const std::string mlp_type =
      layer_type_at(args.mlp_layer_types(), layer_id, "sparse");
  CHECK(mlp_type == "dense" || mlp_type == "sparse")
      << "Unsupported GLM5-Next MLP layer type at layer " << layer_id << ": "
      << mlp_type;

  return {
      .attention = attention_type == "linear_attention"
                       ? Glm5NextAttentionRole::KDA
                       : Glm5NextAttentionRole::DSA,
      .mlp = mlp_type == "dense" ? Glm5NextMlpRole::DENSE
                                 : Glm5NextMlpRole::SPARSE,
  };
}

Glm5NextDecoderLayerImpl::Glm5NextDecoderLayerImpl(const ModelContext& context,
                                                   int32_t layer_id)
    : layer_id_(layer_id),
      role_(resolve_glm5_next_layer_role(context.get_model_args(), layer_id)) {
  const ModelArgs& args = context.get_model_args();
  const QuantArgs& quant_args = context.get_quant_args();
  const ParallelArgs& parallel_args = context.get_parallel_args();
  const torch::TensorOptions& options = context.get_tensor_options();
  tp_group_ = parallel_args.tp_group_;

  const int64_t hidden_size = args.hidden_size();
  const double norm_eps = static_cast<double>(args.rms_norm_eps());
  attn_hc_pre_ = register_module("attn_hc_pre",
                                 MHCPre(args.hc_mult(),
                                        hidden_size,
                                        args.hc_sinkhorn_iters(),
                                        static_cast<double>(args.hc_eps()),
                                        norm_eps,
                                        options));
  ffn_hc_pre_ = register_module("ffn_hc_pre",
                                MHCPre(args.hc_mult(),
                                       hidden_size,
                                       args.hc_sinkhorn_iters(),
                                       static_cast<double>(args.hc_eps()),
                                       norm_eps,
                                       options));
  hc_post_ = register_module("hc_post", MHCPost(norm_eps));
  input_norm_ = register_module(
      "input_layernorm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  post_norm_ =
      register_module("post_attention_layernorm",
                      RMSNorm(hidden_size, args.rms_norm_eps(), options));

  if (role_.attention == Glm5NextAttentionRole::KDA) {
    kda_ = register_module("self_attn", Glm5NextKDA(context));
  } else {
    dsa_ = register_module("self_attn", DeepseekV2Attention(context));
  }

  if (role_.mlp == Glm5NextMlpRole::DENSE) {
    dense_mlp_ =
        register_module("mlp",
                        DenseMLP(hidden_size,
                                 args.intermediate_size(),
                                 /*is_gated=*/true,
                                 /*has_bias=*/false,
                                 args.hidden_act(),
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 parallel_args.tp_group_,
                                 options,
                                 /*module_prefix=*/"",
                                 static_cast<double>(args.swiglu_limit())));
  } else {
    const std::shared_ptr<ModelStreamRegistry>& streams =
        context.stream_registry();
    sparse_moe_ = register_module(
        "mlp",
        DeepseekV4SparseMoEBlock(
            args,
            quant_args,
            parallel_args,
            options,
            /*use_hash=*/false,
            streams->get(ExecutionStreamRole::COMMUNICATION),
            streams->get(ExecutionStreamRole::AUXILIARY_COMPUTE),
            "model.layers." + std::to_string(layer_id) + ".mlp"));
  }
}

void Glm5NextDecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  const StateDict attention_state = state_dict.get_dict_with_prefix(
      std::vector<std::string>{"self_attn.", "attn."});
  if (kda_) {
    kda_->load_state_dict(attention_state);
  } else {
    dsa_->load_state_dict(attention_state);
  }
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  if (dense_mlp_) {
    dense_mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  } else {
    sparse_moe_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  }
  attn_hc_pre_->load_state_dict(
      get_hc_state(state_dict, "attn_hc_pre.", "hc_attn_"));
  ffn_hc_pre_->load_state_dict(
      get_hc_state(state_dict, "ffn_hc_pre.", "hc_ffn_"));
}

void Glm5NextDecoderLayerImpl::verify_loaded_weights() const {
  if (dsa_) {
    dsa_->verify_loaded_weights();
  }
}

torch::Tensor Glm5NextDecoderLayerImpl::forward(
    torch::Tensor& hidden_states,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    std::optional<PendingMHC>* pending_mhc,
    bool is_last_layer,
    bool materialize_output,
    const glm5_next_pcp::Context* pcp_context) {
  if (pcp_context != nullptr) {
    return forward_pcp(hidden_states,
                       residual,
                       positions,
                       attn_metadata,
                       pcp_context->local_metadata,
                       kv_cache,
                       input_params,
                       pending_mhc,
                       is_last_layer,
                       materialize_output,
                       *pcp_context);
  }
  residual = std::nullopt;

  const bool has_pending_storage = pending_mhc != nullptr;
  const bool has_pending = has_pending_storage && pending_mhc->has_value();
  const MHCFusionPlan mhc_plan = resolve_mhc_fusion({
      .supports_fused_mhc = attn_hc_pre_->supports_fused_mhc() &&
                            ffn_hc_pre_->supports_fused_mhc(),
      .has_pending_storage = has_pending_storage,
      .has_pending = has_pending,
      .is_last_layer = is_last_layer,
  });
  CHECK(!has_pending || mhc_plan.consume_pending)
      << "GLM5-Next pending mHC state cannot enter an unfused layer.";

  torch::Tensor residual_attention;
  MHCPreOutput attention_hc;
  torch::Tensor attention_input;
  if (mhc_plan.consume_pending) {
    PendingMHC& pending = pending_mhc->value();
    std::tie(attention_input,
             residual_attention,
             attention_hc.post,
             attention_hc.comb) =
        attn_hc_pre_->fused_post_pre_norm(pending.x,
                                          pending.residual,
                                          pending.post,
                                          pending.comb,
                                          input_norm_->weight());
    pending_mhc->reset();
  } else {
    residual_attention = hidden_states;
    attention_hc = attn_hc_pre_->forward(hidden_states);
    attention_input = std::get<0>(input_norm_->forward(attention_hc.output));
  }

  torch::Tensor attention_output;
  if (kda_) {
    attention_output =
        kda_->forward(attention_input, attn_metadata, kv_cache, input_params);
  } else {
    attention_output = dsa_->forward(positions,
                                     attention_input,
                                     attn_metadata,
                                     kv_cache,
                                     /*sp_ctx=*/nullptr,
                                     /*topk_transfer=*/nullptr)
                           .output;
    if (tp_group_ != nullptr && tp_group_->world_size() > 1) {
      attention_output = parallel_state::reduce(attention_output, tp_group_);
    }
  }

  torch::Tensor residual_ffn;
  MHCPreOutput ffn_hc;
  torch::Tensor ffn_input;
  if (mhc_plan.use_fused_mhc) {
    std::tie(ffn_input, residual_ffn, ffn_hc.post, ffn_hc.comb) =
        ffn_hc_pre_->fused_post_pre_norm(attention_output,
                                         residual_attention,
                                         attention_hc.post,
                                         attention_hc.comb,
                                         post_norm_->weight());
  } else {
    std::tie(hidden_states, std::ignore) = hc_post_->forward(attention_output,
                                                             residual_attention,
                                                             attention_hc.post,
                                                             attention_hc.comb);
    residual_ffn = hidden_states;
    ffn_hc = ffn_hc_pre_->forward(hidden_states);
    ffn_input = std::get<0>(post_norm_->forward(ffn_hc.output));
  }
  torch::Tensor ffn_output;
  if (dense_mlp_) {
    ffn_output = dense_mlp_->forward(ffn_input);
  } else {
    FusedMoEImpl::RouteInfo route = sparse_moe_->prep_route(ffn_input);
    ffn_output = sparse_moe_->forward_selected(
        ffn_input, route.reduce_weight, route.expert_id, input_params);
  }
  // Auxiliary hidden capture needs the post-mHC residual streams at this
  // layer boundary. Keep fusion within the layer and consume any pending input.
  if (mhc_plan.defer_post && !materialize_output) {
    pending_mhc->emplace(
        PendingMHC{ffn_output, residual_ffn, ffn_hc.post, ffn_hc.comb});
    hidden_states = ffn_output;
    return hidden_states;
  }
  std::tie(hidden_states, std::ignore) =
      hc_post_->forward(ffn_output, residual_ffn, ffn_hc.post, ffn_hc.comb);
  return hidden_states;
}

torch::Tensor Glm5NextDecoderLayerImpl::forward_pcp(
    torch::Tensor& hidden_states,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& global_metadata,
    const AttentionMetadata& local_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    std::optional<PendingMHC>* pending_mhc,
    bool is_last_layer,
    bool materialize_output,
    const glm5_next_pcp::Context& context) {
  residual = std::nullopt;
  const bool has_pending_storage = pending_mhc != nullptr;
  const bool has_pending = has_pending_storage && pending_mhc->has_value();
  const MHCFusionPlan mhc_plan = resolve_mhc_fusion({
      .supports_fused_mhc = attn_hc_pre_->supports_fused_mhc() &&
                            ffn_hc_pre_->supports_fused_mhc(),
      .has_pending_storage = has_pending_storage,
      .has_pending = has_pending,
      .is_last_layer = is_last_layer,
  });
  CHECK(!has_pending || mhc_plan.consume_pending);

  torch::Tensor residual_attention;
  MHCPreOutput attention_hc;
  torch::Tensor attention_input;
  if (mhc_plan.consume_pending) {
    PendingMHC& pending = pending_mhc->value();
    std::tie(attention_input,
             residual_attention,
             attention_hc.post,
             attention_hc.comb) =
        attn_hc_pre_->fused_post_pre_norm(pending.x,
                                          pending.residual,
                                          pending.post,
                                          pending.comb,
                                          input_norm_->weight());
    pending_mhc->reset();
  } else {
    residual_attention = hidden_states;
    attention_hc = attn_hc_pre_->forward(hidden_states);
    attention_input = std::get<0>(input_norm_->forward(attention_hc.output));
  }

  torch::Tensor attention_output;
  if (kda_) {
    attention_output = kda_->forward_pcp(
        attention_input, global_metadata, kv_cache, input_params, context);
  } else {
    attention_output = dsa_->forward_glm5_next_pcp(positions,
                                                   attention_input,
                                                   global_metadata,
                                                   local_metadata,
                                                   kv_cache,
                                                   context);
  }

  torch::Tensor residual_ffn;
  MHCPreOutput ffn_hc;
  torch::Tensor ffn_input;
  if (mhc_plan.use_fused_mhc) {
    std::tie(ffn_input, residual_ffn, ffn_hc.post, ffn_hc.comb) =
        ffn_hc_pre_->fused_post_pre_norm(attention_output,
                                         residual_attention,
                                         attention_hc.post,
                                         attention_hc.comb,
                                         post_norm_->weight());
  } else {
    std::tie(hidden_states, std::ignore) = hc_post_->forward(attention_output,
                                                             residual_attention,
                                                             attention_hc.post,
                                                             attention_hc.comb);
    residual_ffn = hidden_states;
    ffn_hc = ffn_hc_pre_->forward(hidden_states);
    ffn_input = std::get<0>(post_norm_->forward(ffn_hc.output));
  }
  torch::Tensor ffn_output;
  if (dense_mlp_) {
    ffn_output = dense_mlp_->forward(ffn_input);
  } else {
    ffn_output = sparse_moe_->forward_cp(ffn_input,
                                         std::nullopt,
                                         context.geometry.tokens_per_rank,
                                         context.cp_group);
  }
  if (mhc_plan.defer_post && !materialize_output) {
    pending_mhc->emplace(
        PendingMHC{ffn_output, residual_ffn, ffn_hc.post, ffn_hc.comb});
    hidden_states = ffn_output;
    return hidden_states;
  }
  std::tie(hidden_states, std::ignore) =
      hc_post_->forward(ffn_output, residual_ffn, ffn_hc.post, ffn_hc.comb);
  return hidden_states;
}

}  // namespace layer
}  // namespace xllm
