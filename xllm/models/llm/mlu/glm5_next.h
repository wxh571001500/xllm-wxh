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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/aux_hidden_capture.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/kernels/mlu/chunk_kda.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/mlu/glm5_next/glm5_next_decoder_layer.h"
#include "core/layers/mlu/glm5_next/glm5_next_pcp_context.h"
#include "models/llm/llm_model_base.h"
#include "models/llm/mlu/glm5_next_graph.h"
#include "models/model_registry.h"

namespace xllm {
namespace mlu {
namespace model {

namespace glm5_next_pcp {

inline std::optional<layer::glm5_next_pcp::Context> prepare_context(
    const layer::AttentionMetadata& metadata,
    const std::vector<int32_t>& host_query_starts,
    torch::Tensor& hidden_states,
    torch::Tensor& positions,
    int32_t cp_size,
    int32_t cp_rank,
    ProcessGroup* cp_group,
    int64_t local_linear_heads,
    const torch::Device& device) {
  if (cp_size <= 1 || metadata.is_dummy || metadata.is_mixed ||
      metadata.is_spec_verify ||
      (!metadata.is_prefill && !metadata.is_chunked_prefill)) {
    return std::nullopt;
  }
  const std::vector<int32_t> query_lengths =
      layer::glm5_next_pcp::query_lengths_from_cumulative(host_query_starts,
                                                          positions.size(0));
  const int32_t chunk_size = static_cast<int32_t>(
      kernel::mlu::kda_prefill_chunk_size(local_linear_heads,
                                          /*use_qk_l2norm=*/true));
  layer::glm5_next_pcp::Context context = layer::glm5_next_pcp::build_context(
      query_lengths, cp_rank, cp_group, chunk_size, device);
  const bool all_ranks_have_tokens =
      std::all_of(context.geometry.tokens_per_rank.begin(),
                  context.geometry.tokens_per_rank.end(),
                  [](int32_t count) { return count > 0; });
  if (!all_ranks_have_tokens) {
    // A rank owns whole KDA chunks, so a batch with fewer chunks than CP ranks
    // leaves some rank empty.  This is common for short prompts and for the
    // last short chunk of a chunked prefill; every rank then runs the full
    // sequence.
    LOG_EVERY_N(WARNING, 100)
        << "GLM5-Next PCP skipped: a CP rank would receive no tokens (tokens="
        << context.geometry.total_tokens
        << ", requests=" << query_lengths.size()
        << ", chunk_size=" << chunk_size << ", cp_size=" << cp_size << ").";
    return std::nullopt;
  }

  context.global_positions = positions;
  context.local_metadata = metadata;
  context.local_metadata.q_cu_seq_lens = context.local_cu_seqlens;
  context.local_metadata.q_seq_lens_vec = context.local_query_lengths;
  context.local_metadata.kpool_query_lens = context.local_query_lengths;
  context.local_metadata.kpool_batch_metadata.reset();
  context.local_metadata.q_seq_lens =
      torch::tensor(context.local_query_lengths,
                    torch::TensorOptions().dtype(torch::kInt32).device(device));
  context.local_metadata.max_query_len = context.local_max_query_len;
  context.local_metadata.slot_mapping =
      layer::glm5_next_pcp::shard_rows(metadata.slot_mapping, context);
  layer::AttentionMetadataBuilder::build_linear_prefill(context.local_metadata,
                                                        chunk_size);
  hidden_states = layer::glm5_next_pcp::shard_rows(hidden_states, context);
  positions = layer::glm5_next_pcp::shard_rows(positions, context);
  return context;
}

inline void restore_output(
    torch::Tensor& normalized,
    std::optional<torch::Tensor>& residual,
    const std::optional<layer::glm5_next_pcp::Context>& context) {
  if (!context.has_value()) {
    return;
  }
  normalized = layer::glm5_next_pcp::gather_restore(normalized, *context);
  if (residual.has_value()) {
    residual = layer::glm5_next_pcp::gather_restore(residual.value(), *context);
  }
}

}  // namespace glm5_next_pcp

class Glm5NextModelImpl final
    : public LlmModelImplBase<layer::Glm5NextDecoderLayer> {
 public:
  explicit Glm5NextModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::Glm5NextDecoderLayer>("glm5_next",
                                                      context.get_model_args()),
        device_(context.get_tensor_options().device()),
        hc_mult_(context.get_model_args().hc_mult()),
        aux_capture_(context.get_model_args(),
                     context.get_tensor_options(),
                     SchedulerConfig::get_instance().max_tokens_per_batch()) {
    const ModelArgs& args = context.get_model_args();
    const ParallelArgs& parallel_args = context.get_parallel_args();
    cp_size_ = parallel_args.cp_size();
    cp_rank_ = parallel_args.cp_rank();
    cp_group_ = parallel_args.cp_group_;
    CHECK(cp_size_ == 1 || cp_group_ != nullptr)
        << "GLM5-Next PCP requires a CP process group.";
    CHECK_GT(hc_mult_, 0) << "GLM5-Next requires hc_mult > 0.";

    const int64_t tp_size = std::max<int64_t>(parallel_args.tp_size(), 1);
    CHECK_EQ(args.linear_num_key_heads() % tp_size, 0)
        << "GLM5-Next KDA head count must be divisible by TP size.";
    local_linear_heads_ = args.linear_num_key_heads() / tp_size;

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(static_cast<size_t>(args.n_layers()));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
    norm_ = register_module("norm", layer::RMSNorm(context));
    for (int32_t layer_id = 0; layer_id < args.n_layers(); ++layer_id) {
      layer::Glm5NextDecoderLayer decoder_layer(context, layer_id);
      layers_.emplace_back(decoder_layer);
      blocks_->push_back(decoder_layer);
    }
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    torch::NoGradGuard no_grad;
    if (input_params.is_spec_verify ||
        (input_params.attn_metadata &&
         input_params.attn_metadata->is_spec_verify)) {
      CHECK(input_params.meta.batch_forward_type.is_chunked_prefill())
          << "GLM5-Next speculative verification requires chunked-prefill "
             "Dense Validate Span.";
    }
    CHECK_EQ(kv_caches.size(), layers_.size())
        << "GLM5-Next requires one layer-specific cache object per layer.";

    if (tokens.numel() == 0) {
      tokens = torch::ones(
          {1}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
      positions = torch::zeros(
          {1}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    }

    torch::Tensor hidden_states =
        input_params.embedding.input_embedding.defined()
            ? input_params.embedding.input_embedding
            : embed_tokens_(tokens);
    if (hidden_states.dim() == 2) {
      hidden_states =
          hidden_states.unsqueeze(1).repeat({1, hc_mult_, 1}).contiguous();
    }

    ModelInputParams modified_input_params = input_params;
    std::vector<int32_t>& dp_token_nums =
        modified_input_params.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(modified_input_params,
                                                     model_args_.enable_mla(),
                                                     /*compute_dtype=*/"half",
                                                     /*attn_mask=*/std::nullopt,
                                                     /*device=*/device_));
    }
    if (!modified_input_params.attn_metadata->is_spec_verify &&
        (modified_input_params.attn_metadata->is_prefill ||
         modified_input_params.attn_metadata->is_chunked_prefill)) {
      layer::AttentionMetadataBuilder::build_linear_prefill(
          *modified_input_params.attn_metadata,
          kernel::mlu::kda_prefill_chunk_size(local_linear_heads_,
                                              /*use_qk_l2norm=*/true));
    }
    const layer::AttentionMetadata& attn_metadata =
        *modified_input_params.attn_metadata;

    std::optional<layer::glm5_next_pcp::Context> pcp_context =
        glm5_next_pcp::prepare_context(
            attn_metadata,
            modified_input_params.attention.host.q_seq_lens,
            hidden_states,
            positions,
            cp_size_,
            cp_rank_,
            cp_group_,
            local_linear_heads_,
            device_);

    std::optional<torch::Tensor> residual;
    std::optional<layer::PendingMHC> pending_mhc;
    for (size_t layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      if (!modified_input_params.synchronize_layer(
              static_cast<uint32_t>(layer_id))) {
        return ModelOutput();
      }
      const bool capture_hidden =
          aux_capture_.should_capture(static_cast<int32_t>(layer_id));
      hidden_states = layers_[layer_id]->forward(
          hidden_states,
          residual,
          positions,
          attn_metadata,
          kv_caches[layer_id],
          modified_input_params,
          &pending_mhc,
          /*is_last_layer=*/layer_id + 1 == layers_.size(),
          /*materialize_output=*/capture_hidden,
          pcp_context.has_value() ? &pcp_context.value() : nullptr);
      if (capture_hidden) {
        CHECK(!pending_mhc.has_value())
            << "Captured GLM5-Next layers must materialize mHC output.";
        torch::Tensor captured_hidden = hidden_states.mean(/*dim=*/-2);
        // Captured layers carry no residual; this only satisfies the
        // restore_output signature.
        std::optional<torch::Tensor> captured_residual;
        glm5_next_pcp::restore_output(
            captured_hidden, captured_residual, pcp_context);
        aux_capture_.capture_layer(
            static_cast<int32_t>(layer_id), captured_hidden, std::nullopt);
      }
      if (!modified_input_params.record_layer(static_cast<uint32_t>(layer_id),
                                              hidden_states.device())) {
        return ModelOutput();
      }
    }

    CHECK(!pending_mhc.has_value())
        << "GLM5-Next final layer must materialize pending mHC state.";
    CHECK_EQ(hidden_states.dim(), 3)
        << "GLM5-Next mHC output must retain the residual-stream dimension.";
    CHECK_EQ(hidden_states.size(-2), hc_mult_)
        << "GLM5-Next mHC residual-stream count changed unexpectedly.";
    hidden_states = hidden_states.mean(/*dim=*/-2);
    auto [normalized, residual_out] = norm_(hidden_states, std::nullopt);
    glm5_next_pcp::restore_output(normalized, residual_out, pcp_context);
    return aux_capture_.finalize(normalized, residual_out);
  }

  void load_state_dict(const StateDict& state_dict) override {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    for (size_t layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layers_[layer_id]->load_state_dict(state_dict.get_dict_with_prefix(
          "layers." + std::to_string(layer_id) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights() const {
    for (const auto& layer : layers_) {
      layer->verify_loaded_weights();
    }
  }

 private:
  torch::Device device_;
  int64_t hc_mult_ = 1;
  AuxHiddenCapture aux_capture_;
  int64_t local_linear_heads_ = 1;
  int32_t cp_size_ = 1;
  int32_t cp_rank_ = 0;
  ProcessGroup* cp_group_ = nullptr;
  torch::nn::ModuleList blocks_{nullptr};
};
TORCH_MODULE(Glm5NextModel);

class Glm5NextForCausalLMImpl final
    : public LlmForCausalLMImplBase<Glm5NextModel> {
 public:
  explicit Glm5NextForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Glm5NextModel>(context) {}

  bool requires_graph_forward_metadata() { return true; }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return std::make_unique<Glm5NextGraphMetadataState>();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& params) {
    Glm5NextGraphMetadata::prepare(state, positions, params);
  }

  bool is_hybrid_linear_attention() { return true; }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    LlmForCausalLMImplBase<Glm5NextModel>::load_model(std::move(loader),
                                                      std::move(prefix));
    model_->verify_loaded_weights();
  }
};
TORCH_MODULE(Glm5NextForCausalLM);

REGISTER_CAUSAL_MODEL(glm5_next, Glm5NextForCausalLM);

}  // namespace model
}  // namespace mlu
}  // namespace xllm
