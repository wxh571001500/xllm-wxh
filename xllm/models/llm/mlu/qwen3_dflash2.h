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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/layers/common/add_matmul.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/mlu/dflash2_context_kv.h"
#include "framework/model_loader.h"
#include "models/llm/qwen3.h"
#include "models/model_registry.h"

namespace xllm::mlu::model {

class DFlash2CandidateSelector final {
 public:
  DFlash2CandidateSelector(const ModelArgs& args,
                           const torch::TensorOptions& options)
      : options_(options),
        hidden_size_(args.hidden_size()),
        vocab_size_(args.vocab_size()),
        rank_(args.dflash2_selector_rank()),
        top_k_(args.dflash2_selector_top_k()) {
    CHECK_GT(rank_, 0) << "DFlash2 selector_rank must be positive.";
    CHECK_GT(top_k_, 0) << "DFlash2 selector_top_k must be positive.";
    CHECK_LE(top_k_, vocab_size_)
        << "DFlash2 selector_top_k exceeds vocabulary.";
  }

  void load_state_dict(const StateDict& state_dict) {
    torch::Tensor hidden_projection =
        state_dict.get_tensor("hidden_projection.weight");
    torch::Tensor predecessor = state_dict.get_tensor("predecessor_codebook");
    torch::Tensor successor = state_dict.get_tensor("successor_codebook");
    if (hidden_projection.defined()) {
      hidden_projection_ = hidden_projection.to(options_);
    }
    if (predecessor.defined()) {
      predecessor_codebook_ = predecessor.to(options_);
    }
    if (successor.defined()) {
      successor_codebook_ = successor.to(options_);
    }
  }

  void verify_loaded_weights() const {
    CHECK(hidden_projection_.defined())
        << "Missing DFlash2 candidate_selector.hidden_projection.weight.";
    CHECK(predecessor_codebook_.defined())
        << "Missing DFlash2 candidate_selector.predecessor_codebook.";
    CHECK(successor_codebook_.defined())
        << "Missing DFlash2 candidate_selector.successor_codebook.";
    CHECK_EQ(hidden_projection_.sizes(),
             torch::IntArrayRef({rank_, hidden_size_}));
    CHECK_EQ(predecessor_codebook_.sizes(),
             torch::IntArrayRef({vocab_size_, rank_}));
    CHECK_EQ(successor_codebook_.sizes(),
             torch::IntArrayRef({vocab_size_, rank_}));
  }

  DFlash2CandidateOutput forward(const torch::Tensor& hidden_states,
                                 const torch::Tensor& unary_logits,
                                 const torch::Tensor& anchor_token_ids) const {
    CHECK_EQ(hidden_states.dim(), 3);
    CHECK_EQ(unary_logits.dim(), 3);
    CHECK_EQ(hidden_states.size(0), unary_logits.size(0));
    CHECK_EQ(hidden_states.size(1), unary_logits.size(1));
    CHECK_EQ(unary_logits.size(2), vocab_size_);
    CHECK_EQ(anchor_token_ids.dim(), 1);
    CHECK_EQ(anchor_token_ids.size(0), hidden_states.size(0));

    auto topk = torch::topk(unary_logits, top_k_, /*dim=*/-1);
    torch::Tensor values = std::get<0>(topk).to(torch::kFloat32);
    torch::Tensor candidate_ids = std::get<1>(topk).to(torch::kLong);

    namespace F = torch::nn::functional;
    torch::Tensor hidden = F::linear(hidden_states, hidden_projection_);
    torch::Tensor successors = F::embedding(candidate_ids, successor_codebook_);
    torch::Tensor anchor =
        anchor_token_ids.view({-1, 1, 1}).expand({-1, 1, top_k_});
    torch::Tensor predecessor_ids =
        torch::cat({anchor,
                    candidate_ids.slice(/*dim=*/1,
                                        /*start=*/0,
                                        candidate_ids.size(1) - 1)},
                   /*dim=*/1);
    torch::Tensor predecessors =
        F::embedding(predecessor_ids, predecessor_codebook_);
    torch::Tensor pair_scores =
        torch::einsum("blpr,blcr->blpc",
                      {predecessors * hidden.unsqueeze(/*dim=*/2), successors});

    DFlash2CandidateOutput output;
    output.candidate_ids = candidate_ids;
    output.edge_logits = values.unsqueeze(/*dim=*/2) + pair_scores;
    return output;
  }

 private:
  torch::Tensor hidden_projection_;
  torch::Tensor predecessor_codebook_;
  torch::Tensor successor_codebook_;
  torch::TensorOptions options_;
  int64_t hidden_size_ = 0;
  int64_t vocab_size_ = 0;
  int64_t rank_ = 0;
  int64_t top_k_ = 0;
};

class DFlash2Qwen3ModelImpl final : public ::xllm::QWen3ModelImpl {
 public:
  explicit DFlash2Qwen3ModelImpl(const ModelContext& context)
      : ::xllm::QWen3ModelImpl(validate_context(context)),
        selector_(context.get_model_args(), context.get_tensor_options()),
        context_kv_(context) {
    const ModelArgs& args = context.get_model_args();
    const auto& tensor_options = context.get_tensor_options();
    block_size_ = args.dflash2_block_size();

    fc_ = register_module("fc",
                          layer::AddMatmul(args.hidden_size() * args.n_layers(),
                                           args.hidden_size(),
                                           /*with_bias=*/false,
                                           tensor_options));
    hidden_norm_ = register_module(
        "hidden_norm",
        layer::RMSNorm(
            args.hidden_size(), args.rms_norm_eps(), tensor_options));
  }

  void load_state_dict(const StateDict& state_dict) override {
    fc_->load_state_dict(state_dict.get_dict_with_prefix("fc."));
    hidden_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("hidden_norm."));
    selector_.load_state_dict(
        state_dict.get_dict_with_prefix("candidate_selector."));
    context_kv_.load_state_dict(state_dict);
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights() const {
    fc_->verify_loaded_weights("fc.");
    hidden_norm_->verify_loaded_weights("hidden_norm.");
    norm_->verify_loaded_weights("norm.");
    selector_.verify_loaded_weights();
    context_kv_.verify_loaded_weights();
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      layers_[i]->verify_loaded_weights("layers." + std::to_string(i) + ".");
    }
  }

  void finalize_loaded_weights() { context_kv_.finalize_loaded_weights(); }

  DFlash2CandidateOutput candidates(
      const torch::Tensor& hidden_states,
      const torch::Tensor& unary_logits,
      const torch::Tensor& anchor_token_ids) const {
    return selector_.forward(hidden_states, unary_logits, anchor_token_ids);
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& input_params) {
    CHECK_EQ(target_hidden.dim(), 2);
    torch::Tensor hidden = fc_->forward(target_hidden);
    hidden = std::get<0>(hidden_norm_->forward(hidden));
    if (!context_kv_.write(
            hidden, positions, device_cache_slots, kv_caches, input_params)) {
      return ModelOutput();
    }
    return ModelOutput(hidden);
  }

 protected:
  layer::AttentionMetadata get_attention_metadata(
      const ModelInputParams& params,
      const torch::Tensor& h) override {
    layer::AttentionMetadata metadata =
        QWen3ModelImpl::get_attention_metadata(params, h);
    // DFlash2 jointly denoises a complete proposal block within the
    // checkpoint's sliding window.
    metadata.non_causal_window_right = block_size_ - 1;
    return metadata;
  }

 private:
  static const ModelContext& validate_context(const ModelContext& context) {
    const ModelArgs& args = context.get_model_args();
    CHECK_GT(args.n_layers(), 0);
    CHECK_GT(args.hidden_size(), 0);
    CHECK_GT(args.sliding_window(), 0);
    CHECK_GT(args.dflash2_block_size(), 1);
    CHECK_GT(args.dflash2_conv_group_size(), 0);
    CHECK_EQ(args.hidden_size() % args.dflash2_conv_group_size(), 0);
    CHECK_GT(args.dflash2_conv_kernel_size(), 0);
    CHECK_LE(args.dflash2_conv_kernel_size(), args.dflash2_block_size());
    CHECK_GT(args.dflash2_selector_rank(), 0);
    CHECK_GT(args.dflash2_selector_top_k(), 0);
    CHECK_LE(args.dflash2_selector_top_k(), args.vocab_size());
    return context;
  }

  layer::AddMatmul fc_{nullptr};
  layer::RMSNorm hidden_norm_{nullptr};
  DFlash2CandidateSelector selector_;
  layer::DFlash2ContextKV context_kv_;
  int32_t block_size_ = 0;
};
TORCH_MODULE(DFlash2Qwen3Model);

class DFlash2Qwen3ForCausalLMImpl final
    : public ::xllm::LlmForCausalLMImplBase<DFlash2Qwen3Model> {
 public:
  using Base = ::xllm::LlmForCausalLMImplBase<DFlash2Qwen3Model>;

  explicit DFlash2Qwen3ForCausalLMImpl(const ModelContext& context)
      : Base(context) {}

  using Base::logits;

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes,
                       torch::Tensor& out_hidden) override {
    out_hidden = selected_idxes.defined()
                     ? hidden_states.index_select(
                           /*dim=*/0, selected_idxes.to(torch::kLong))
                     : hidden_states;
    return lm_head_(out_hidden);
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "") override {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
    }
    model_->verify_loaded_weights();
    model_->finalize_loaded_weights();
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& input_params) {
    return model_->write_context_kv(
        target_hidden, positions, device_cache_slots, kv_caches, input_params);
  }

  DFlash2CandidateOutput dflash2_candidates(
      const torch::Tensor& hidden_states,
      const torch::Tensor& unary_logits,
      const torch::Tensor& anchor_token_ids) {
    return model_->candidates(hidden_states, unary_logits, anchor_token_ids);
  }
};
TORCH_MODULE(DFlash2Qwen3ForCausalLM);

REGISTER_CAUSAL_MODEL_WITH_VARNAME(dflash2_draft_model,
                                   DFlash2DraftModel,
                                   DFlash2Qwen3ForCausalLM);

}  // namespace xllm::mlu::model
