/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/dit_model_context.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/linear.h"
#include "models/model_registry.h"

namespace xllm {
namespace minimax_h3 {

// RMSNorm module owning its scale weight. It also exposes the fp32 RMSNorm
// math for callers that need the same operation without owning a module.
class H3RMSNormImpl final : public torch::nn::Module {
 public:
  H3RMSNormImpl(int64_t dim, double eps) : eps_(eps) {
    weight_ = register_parameter("weight", torch::ones({dim}));
  }

  static torch::Tensor rms_norm(const torch::Tensor& input,
                                const torch::Tensor& weight,
                                double eps) {
    torch::Tensor x = input.to(torch::kFloat32);
    torch::Tensor variance = x.pow(2).mean(-1, true);
    torch::Tensor output = x * torch::rsqrt(variance + eps);
    if (weight.defined()) {
      output = output * weight.to(output.device(), output.scalar_type());
    }
    return output.to(input.scalar_type());
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    return rms_norm(input, weight_, eps_);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", weight_, weight_is_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
  }

 private:
  double eps_;
  torch::Tensor weight_;
  bool weight_is_loaded_ = false;
};
TORCH_MODULE(H3RMSNorm);

class H3SwiGLUFFNImpl final : public torch::nn::Module {
 public:
  H3SwiGLUFFNImpl(int64_t hidden_size,
                  int64_t ffn_dim,
                  bool bias,
                  const ModelContext& context)
      : options_(context.get_tensor_options()),
        tp_group_(context.get_parallel_args().dit_tp_group_) {
    CHECK(!bias) << "MiniMax-H3 SwiGLU checkpoint does not use bias";
    const QuantArgs& quant_args = context.get_quant_args();
    proj_in_ =
        register_module("proj_in",
                        layer::ColumnParallelLinear(hidden_size,
                                                    2 * ffn_dim,
                                                    bias,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    tp_group_,
                                                    options_));
    proj_out_ = register_module(
        "proj_out",
        layer::RowParallelLinear(ffn_dim,
                                 hidden_size,
                                 bias,
                                 /*input_is_parallelized=*/true,
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 tp_group_,
                                 options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor x = proj_in_->forward(hidden_states);
    std::vector<torch::Tensor> chunks = x.chunk(2, -1);
    torch::Tensor activation = chunks[0] * torch::silu(chunks[1]);
    return proj_out_->forward(activation);
  }

  void load_state_dict(const StateDict& state_dict) {
    const StateDict proj_in_state =
        state_dict.get_dict_with_prefix("net.0.proj.");
    const int64_t tp_size = tp_group_ == nullptr ? 1 : tp_group_->world_size();
    if (tp_size == 1) {
      proj_in_->load_state_dict(proj_in_state);
    } else {
      const int64_t tp_rank = tp_group_->rank();
      torch::Tensor weight = proj_in_state.get_tensor("weight");
      if (weight.defined()) {
        CHECK_EQ(weight.size(0) % (2 * tp_size), 0)
            << "MiniMax-H3 SwiGLU dimension must be divisible by TP size";
        const int64_t ffn_dim = weight.size(0) / 2;
        std::vector<torch::Tensor> up_shards =
            weight.slice(0, 0, ffn_dim).chunk(tp_size, 0);
        std::vector<torch::Tensor> gate_shards =
            weight.slice(0, ffn_dim).chunk(tp_size, 0);
        torch::Tensor local_weight =
            torch::cat({up_shards[tp_rank], gate_shards[tp_rank]}, 0)
                .to(options_);
        for (auto& parameter : proj_in_->named_parameters()) {
          if (parameter.key() == "weight") {
            CHECK_EQ(parameter.value().sizes(), local_weight.sizes());
            parameter.value().data().copy_(local_weight);
            proj_in_weight_is_loaded_ = true;
            break;
          }
        }
      }
    }
    proj_out_->load_state_dict(state_dict.get_dict_with_prefix("net.2."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(proj_in_weight_is_loaded_ || proj_in_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "net.0.proj.weight";
    CHECK(proj_out_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "net.2.weight";
  }

 private:
  torch::TensorOptions options_;
  ProcessGroup* tp_group_ = nullptr;
  layer::ColumnParallelLinear proj_in_{nullptr};
  layer::RowParallelLinear proj_out_{nullptr};
  bool proj_in_weight_is_loaded_ = false;
};
TORCH_MODULE(H3SwiGLUFFN);

inline torch::Tensor h3_timestep_embedding(const torch::Tensor& timesteps,
                                           int64_t embedding_dim,
                                           bool flip_sin_to_cos = true,
                                           double downscale_freq_shift = 0.0,
                                           double scale = 1.0,
                                           int64_t max_period = 10000) {
  int64_t half_dim = embedding_dim / 2;
  torch::Tensor exponent = -std::log(static_cast<double>(max_period)) *
                           torch::arange(0,
                                         half_dim,
                                         torch::TensorOptions()
                                             .dtype(torch::kFloat32)
                                             .device(timesteps.device()));
  exponent = exponent / (half_dim - downscale_freq_shift);
  torch::Tensor emb = torch::exp(exponent);
  emb = timesteps.unsqueeze(1).to(torch::kFloat32) * emb.unsqueeze(0);
  emb = scale * emb;
  emb = torch::cat({torch::sin(emb), torch::cos(emb)}, -1);
  if (flip_sin_to_cos) {
    emb = torch::cat({emb.slice(-1, half_dim), emb.slice(-1, 0, half_dim)}, -1);
  }
  if (embedding_dim % 2 == 1) {
    emb = torch::nn::functional::pad(
        emb, torch::nn::functional::PadFuncOptions({0, 1, 0, 0}));
  }
  return emb;
}

class H3TimestepEmbeddingImpl final : public torch::nn::Module {
 public:
  H3TimestepEmbeddingImpl(int64_t in_dim, int64_t hidden_dim, int64_t out_dim) {
    linear_1_ = register_module(
        "linear_1",
        torch::nn::Linear(
            torch::nn::LinearOptions(in_dim, hidden_dim).bias(true)));
    linear_2_ = register_module(
        "linear_2",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_dim, out_dim).bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& sample) {
    return linear_2_->forward(torch::silu(linear_1_->forward(sample)));
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict,
                        "linear_1.weight",
                        linear_1_->weight,
                        linear_1_weight_is_loaded_);
    weight::load_weight(
        state_dict, "linear_1.bias", linear_1_->bias, linear_1_bias_is_loaded_);
    weight::load_weight(state_dict,
                        "linear_2.weight",
                        linear_2_->weight,
                        linear_2_weight_is_loaded_);
    weight::load_weight(
        state_dict, "linear_2.bias", linear_2_->bias, linear_2_bias_is_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(linear_1_weight_is_loaded_)
        << "weight is not loaded for " << prefix + "linear_1.weight";
    CHECK(linear_1_bias_is_loaded_)
        << "bias is not loaded for " << prefix + "linear_1.bias";
    CHECK(linear_2_weight_is_loaded_)
        << "weight is not loaded for " << prefix + "linear_2.weight";
    CHECK(linear_2_bias_is_loaded_)
        << "bias is not loaded for " << prefix + "linear_2.bias";
  }

 private:
  torch::nn::Linear linear_1_{nullptr};
  torch::nn::Linear linear_2_{nullptr};
  bool linear_1_weight_is_loaded_ = false;
  bool linear_1_bias_is_loaded_ = false;
  bool linear_2_weight_is_loaded_ = false;
  bool linear_2_bias_is_loaded_ = false;
};
TORCH_MODULE(H3TimestepEmbedding);

class H3RotaryPosEmbedImpl final : public torch::nn::Module {
 public:
  explicit H3RotaryPosEmbedImpl(int64_t rope_freq_dim = 16,
                                double rope_theta = 10000.0)
      : rope_freq_dim_(rope_freq_dim), rope_theta_(rope_theta) {}

  std::pair<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& position_ids) const {
    torch::Tensor pos = position_ids.to(torch::kFloat32);
    torch::Tensor arange = torch::arange(
        0,
        2 * rope_freq_dim_,
        2,
        torch::TensorOptions().dtype(torch::kFloat32).device(pos.device()));
    torch::Tensor inv_freq =
        1.0 / torch::pow(rope_theta_,
                         arange / static_cast<double>(2 * rope_freq_dim_));
    torch::Tensor freqs = pos.unsqueeze(-1) * inv_freq.view({1, 1, -1});
    torch::Tensor freqs_t = freqs.select(1, 0);
    torch::Tensor freqs_h = freqs.select(1, 1);
    torch::Tensor freqs_w = freqs.select(1, 2);
    torch::Tensor cat = torch::cat({freqs_t, freqs_h, freqs_w}, -1);
    cat = torch::cat({cat, cat}, -1);
    return {torch::cos(cat), torch::sin(cat)};
  }

 private:
  int64_t rope_freq_dim_;
  double rope_theta_;
};
TORCH_MODULE(H3RotaryPosEmbed);

inline torch::Tensor h3_apply_rotary(const torch::Tensor& hidden_states,
                                     const torch::Tensor& cos,
                                     const torch::Tensor& sin) {
  int64_t rotary_dim = cos.size(-1);
  torch::Tensor rotary = hidden_states.slice(-1, 0, rotary_dim);
  torch::Tensor pass = hidden_states.slice(-1, rotary_dim);
  auto chunks = rotary.chunk(2, -1);
  torch::Tensor rotated = torch::cat({-chunks[1], chunks[0]}, -1);
  torch::Tensor cos_b =
      cos.to(hidden_states.scalar_type()).unsqueeze(0).unsqueeze(2);
  torch::Tensor sin_b =
      sin.to(hidden_states.scalar_type()).unsqueeze(0).unsqueeze(2);
  rotary = rotary * cos_b + rotated * sin_b;
  return torch::cat({rotary, pass}, -1).contiguous();
}

class H3AttentionImpl final : public torch::nn::Module {
 public:
  H3AttentionImpl(int64_t hidden_size,
                  int64_t heads,
                  int64_t dim_head,
                  double qk_norm_eps,
                  const ModelContext& context)
      : dim_head_(dim_head) {
    norm_q_ = register_module("norm_q", H3RMSNorm(dim_head, qk_norm_eps));
    norm_k_ = register_module("norm_k", H3RMSNorm(dim_head, qk_norm_eps));
    ProcessGroup* tp_group = context.get_parallel_args().dit_tp_group_;
    const int64_t tp_size = tp_group == nullptr ? 1 : tp_group->world_size();
    CHECK_EQ(heads % tp_size, 0)
        << "MiniMax-H3 attention heads must be divisible by TP size";
    local_heads_ = heads / tp_size;
    const int64_t inner_dim = heads * dim_head;
    const QuantArgs& quant_args = context.get_quant_args();
    const torch::TensorOptions& options = context.get_tensor_options();
    to_q_ = register_module("to_q",
                            layer::ColumnParallelLinear(hidden_size,
                                                        inner_dim,
                                                        /*bias=*/false,
                                                        /*gather_output=*/false,
                                                        quant_args,
                                                        tp_group,
                                                        options));
    to_k_ = register_module("to_k",
                            layer::ColumnParallelLinear(hidden_size,
                                                        inner_dim,
                                                        /*bias=*/false,
                                                        /*gather_output=*/false,
                                                        quant_args,
                                                        tp_group,
                                                        options));
    to_v_ = register_module("to_v",
                            layer::ColumnParallelLinear(hidden_size,
                                                        inner_dim,
                                                        /*bias=*/false,
                                                        /*gather_output=*/false,
                                                        quant_args,
                                                        tp_group,
                                                        options));
    to_out_ = register_module(
        "to_out",
        layer::RowParallelLinear(inner_dim,
                                 hidden_size,
                                 /*bias=*/false,
                                 /*input_is_parallelized=*/true,
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 tp_group,
                                 options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& cos = torch::Tensor(),
                        const torch::Tensor& sin = torch::Tensor()) {
    int64_t b = hidden_states.size(0);
    int64_t s = hidden_states.size(1);
    torch::Tensor q_projection = to_q_->forward(hidden_states);
    torch::Tensor k_projection = to_k_->forward(hidden_states);
    torch::Tensor v_projection = to_v_->forward(hidden_states);
    torch::Tensor q = q_projection.view({b, s, local_heads_, dim_head_});
    torch::Tensor k = k_projection.view({b, s, local_heads_, dim_head_});
    torch::Tensor v = v_projection.view({b, s, local_heads_, dim_head_});
    q = norm_q_->forward(q);
    k = norm_k_->forward(k);
    if (cos.defined() && sin.defined()) {
      q = h3_apply_rotary(q, cos, sin);
      k = h3_apply_rotary(k, cos, sin);
    }
    q = q.permute({0, 2, 1, 3}).contiguous();
    k = k.permute({0, 2, 1, 3}).contiguous();
    v = v.permute({0, 2, 1, 3}).contiguous();
    torch::Tensor out = torch::scaled_dot_product_attention(
        q, k, v, torch::nullopt, /*dropout_p=*/0.0, /*is_causal=*/false);
    out = out.permute({0, 2, 1, 3}).contiguous();
    out = out.view({b, s, local_heads_ * dim_head_});
    return to_out_->forward(out);
  }

  void load_state_dict(const StateDict& state_dict) {
    to_q_->load_state_dict(state_dict.get_dict_with_prefix("to_q."));
    to_k_->load_state_dict(state_dict.get_dict_with_prefix("to_k."));
    to_v_->load_state_dict(state_dict.get_dict_with_prefix("to_v."));
    to_out_->load_state_dict(state_dict.get_dict_with_prefix("to_out.0."));
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(to_q_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "to_q.weight";
    CHECK(to_k_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "to_k.weight";
    CHECK(to_v_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "to_v.weight";
    CHECK(to_out_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "to_out.0.weight";
    norm_q_->verify_loaded_weights(prefix + "norm_q.");
    norm_k_->verify_loaded_weights(prefix + "norm_k.");
  }

 private:
  int64_t local_heads_;
  int64_t dim_head_;
  layer::ColumnParallelLinear to_q_{nullptr};
  layer::ColumnParallelLinear to_k_{nullptr};
  layer::ColumnParallelLinear to_v_{nullptr};
  layer::RowParallelLinear to_out_{nullptr};
  H3RMSNorm norm_q_{nullptr};
  H3RMSNorm norm_k_{nullptr};
};
TORCH_MODULE(H3Attention);

class H3TokenRefinerBlockImpl final : public torch::nn::Module {
 public:
  H3TokenRefinerBlockImpl(int64_t hidden_size,
                          int64_t heads,
                          int64_t dim_head,
                          int64_t ffn_dim,
                          double norm_eps,
                          double qk_norm_eps,
                          const ModelContext& context) {
    norm1_ = register_module("norm1", H3RMSNorm(hidden_size, norm_eps));
    attn_ = register_module(
        "attn",
        H3Attention(hidden_size, heads, dim_head, qk_norm_eps, context));
    norm2_ = register_module("norm2", H3RMSNorm(hidden_size, norm_eps));
    ff_ = register_module("ff",
                          H3SwiGLUFFN(hidden_size, ffn_dim, false, context));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor norm1 = norm1_->forward(hidden_states);
    torch::Tensor attention = attn_->forward(norm1);
    torch::Tensor h = hidden_states + attention;
    torch::Tensor norm2 = norm2_->forward(h);
    h = h + ff_->forward(norm2);
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    ff_->load_state_dict(state_dict.get_dict_with_prefix("ff."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    norm1_->verify_loaded_weights(prefix + "norm1.");
    attn_->verify_loaded_weights(prefix + "attn.");
    norm2_->verify_loaded_weights(prefix + "norm2.");
    ff_->verify_loaded_weights(prefix + "ff.");
  }

 private:
  H3RMSNorm norm1_{nullptr};
  H3Attention attn_{nullptr};
  H3RMSNorm norm2_{nullptr};
  H3SwiGLUFFN ff_{nullptr};
};
TORCH_MODULE(H3TokenRefinerBlock);

class H3TokenRefinerImpl final : public torch::nn::Module {
 public:
  H3TokenRefinerImpl(int64_t hidden_size,
                     int64_t heads,
                     int64_t dim_head,
                     int64_t ffn_dim,
                     int64_t layers,
                     double norm_eps,
                     double qk_norm_eps,
                     double final_norm_eps,
                     const ModelContext& context) {
    blocks_ = register_module("refiner_blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < layers; ++i) {
      blocks_->push_back(H3TokenRefinerBlock(hidden_size,
                                             heads,
                                             dim_head,
                                             ffn_dim,
                                             norm_eps,
                                             qk_norm_eps,
                                             context));
    }
    final_norm_ =
        register_module("final_norm", H3RMSNorm(hidden_size, final_norm_eps));
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    for (auto& block : *blocks_) {
      hidden_states = block->as<H3TokenRefinerBlock>()->forward(hidden_states);
    }
    hidden_states = final_norm_->forward(hidden_states);
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < blocks_->size(); ++i) {
      blocks_[i]->as<H3TokenRefinerBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("refiner_blocks." +
                                          std::to_string(i) + "."));
    }
    final_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("final_norm."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t i = 0; i < blocks_->size(); ++i) {
      blocks_[i]->as<H3TokenRefinerBlock>()->verify_loaded_weights(
          prefix + "refiner_blocks." + std::to_string(i) + ".");
    }
    final_norm_->verify_loaded_weights(prefix + "final_norm.");
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  H3RMSNorm final_norm_{nullptr};
};
TORCH_MODULE(H3TokenRefiner);

class H3AdaLayerNormModulationImpl final : public torch::nn::Module {
 public:
  H3AdaLayerNormModulationImpl(int64_t time_embed_dim,
                               int64_t hidden_size,
                               const ModelContext& context)
      : hidden_size_(hidden_size) {
    linear_ = register_module(
        "linear",
        layer::ColumnParallelLinear(time_embed_dim,
                                    6 * hidden_size * 3,
                                    /*bias=*/true,
                                    /*gather_output=*/true,
                                    context.get_quant_args(),
                                    context.get_parallel_args().dit_tp_group_,
                                    context.get_tensor_options()));
  }

  std::vector<torch::Tensor> forward(const torch::Tensor& temb) {
    torch::Tensor x =
        linear_->forward(torch::silu(temb).to(linear_->weight().scalar_type()));
    x = x.view({-1, 6 * hidden_size_});
    return x.chunk(6, -1);
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(linear_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "linear.weight";
  }

 private:
  int64_t hidden_size_;
  layer::ColumnParallelLinear linear_{nullptr};
};
TORCH_MODULE(H3AdaLayerNormModulation);

class H3TransformerBlockImpl final : public torch::nn::Module {
 public:
  H3TransformerBlockImpl(int64_t hidden_size,
                         int64_t heads,
                         int64_t dim_head,
                         int64_t ffn_dim,
                         int64_t time_embed_dim,
                         double norm_eps,
                         double qk_norm_eps,
                         const ModelContext& context) {
    norm1_ = register_module("norm1", H3RMSNorm(hidden_size, norm_eps));
    attn_ = register_module(
        "attn",
        H3Attention(hidden_size, heads, dim_head, qk_norm_eps, context));
    norm2_ = register_module("norm2", H3RMSNorm(hidden_size, norm_eps));
    ff_ = register_module("ff",
                          H3SwiGLUFFN(hidden_size, ffn_dim, false, context));
    adaln_ = register_module(
        "adaln_proj",
        H3AdaLayerNormModulation(time_embed_dim, hidden_size, context));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb,
                        const torch::Tensor& adaln_indices,
                        const torch::Tensor& cos,
                        const torch::Tensor& sin) {
    auto chunks = adaln_->forward(temb);
    torch::Tensor shift_msa =
        chunks[0].index_select(0, adaln_indices).unsqueeze(0);
    torch::Tensor scale_msa =
        chunks[1].index_select(0, adaln_indices).unsqueeze(0);
    torch::Tensor gate_msa =
        chunks[2].index_select(0, adaln_indices).unsqueeze(0);
    torch::Tensor shift_mlp =
        chunks[3].index_select(0, adaln_indices).unsqueeze(0);
    torch::Tensor scale_mlp =
        chunks[4].index_select(0, adaln_indices).unsqueeze(0);
    torch::Tensor gate_mlp =
        chunks[5].index_select(0, adaln_indices).unsqueeze(0);

    torch::Tensor norm_h = norm1_->forward(hidden_states);
    norm_h = norm_h * (1.0 + scale_msa) + shift_msa;
    torch::Tensor h =
        hidden_states + gate_msa * attn_->forward(norm_h, cos, sin);
    norm_h = norm2_->forward(h);
    norm_h = norm_h * (1.0 + scale_mlp) + shift_mlp;
    h = h + gate_mlp * ff_->forward(norm_h);
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    ff_->load_state_dict(state_dict.get_dict_with_prefix("ff."));
    adaln_->load_state_dict(state_dict.get_dict_with_prefix("adaln_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    norm1_->verify_loaded_weights(prefix + "norm1.");
    attn_->verify_loaded_weights(prefix + "attn.");
    norm2_->verify_loaded_weights(prefix + "norm2.");
    ff_->verify_loaded_weights(prefix + "ff.");
    adaln_->verify_loaded_weights(prefix + "adaln_proj.");
  }

 private:
  H3RMSNorm norm1_{nullptr};
  H3Attention attn_{nullptr};
  H3RMSNorm norm2_{nullptr};
  H3SwiGLUFFN ff_{nullptr};
  H3AdaLayerNormModulation adaln_{nullptr};
};
TORCH_MODULE(H3TransformerBlock);

class H3AdaLayerNormOutImpl final : public torch::nn::Module {
 public:
  H3AdaLayerNormOutImpl(int64_t hidden_size,
                        int64_t time_embed_dim,
                        double eps) {
    norm_ = register_module("norm", H3RMSNorm(hidden_size, eps));
    linear_ =
        register_module("linear",
                        torch::nn::Linear(torch::nn::LinearOptions(
                                              time_embed_dim, 2 * hidden_size)
                                              .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb,
                        const torch::Tensor& timestep_indices) {
    torch::Tensor x =
        linear_->forward(torch::silu(temb).to(linear_->weight.scalar_type()));
    auto chunks = x.chunk(2, -1);
    torch::Tensor shift =
        chunks[0].index_select(0, timestep_indices).unsqueeze(0);
    torch::Tensor scale =
        chunks[1].index_select(0, timestep_indices).unsqueeze(0);
    torch::Tensor h = norm_->forward(hidden_states);
    return h * (1.0 + scale) + shift;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    weight::load_weight(
        state_dict, "linear.weight", linear_->weight, linear_weight_is_loaded_);
    weight::load_weight(
        state_dict, "linear.bias", linear_->bias, linear_bias_is_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    norm_->verify_loaded_weights(prefix + "norm.");
    CHECK(linear_weight_is_loaded_)
        << "weight is not loaded for " << prefix + "linear.weight";
    CHECK(linear_bias_is_loaded_)
        << "bias is not loaded for " << prefix + "linear.bias";
  }

 private:
  H3RMSNorm norm_{nullptr};
  torch::nn::Linear linear_{nullptr};
  bool linear_weight_is_loaded_ = false;
  bool linear_bias_is_loaded_ = false;
};
TORCH_MODULE(H3AdaLayerNormOut);

}  // namespace minimax_h3

struct MiniMaxH3TransformerOutput {
  torch::Tensor sample;
  torch::Tensor audio_sample;
};

class MiniMaxH3Transformer3DModelImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3Transformer3DModelImpl(const DiTModelContext& context)
      : options_(context.get_tensor_options()) {
    using namespace minimax_h3;
    const ParallelArgs& parallel_args = context.get_parallel_args();
    CHECK(parallel_args.tp_size() == 1 ||
          parallel_args.dit_tp_group_ != nullptr)
        << "MiniMax-H3 TP requires a DiT TP process group";
    ModelContext transformer_context(parallel_args,
                                     context.get_model_args("transformer"),
                                     context.get_quant_args("transformer"),
                                     options_);
    const ModelArgs& model_args = transformer_context.get_model_args();
    heads_ = model_args.n_heads();
    head_dim_ = model_args.head_dim();
    hidden_size_ = model_args.hidden_size();
    num_layers_ = model_args.num_layers();
    num_refiner_layers_ = model_args.num_refiner_layers();
    ffn_dim_ = model_args.ffn_dim();
    in_channels_ = model_args.in_channels();
    audio_in_channels_ = model_args.audio_in_channels();
    patch_size_ = model_args.wan_patch_size();
    text_dim_ = model_args.text_dim();
    freq_dim_ = model_args.time_freq_dim();
    time_embed_hidden_dim_ = model_args.time_embed_hidden_dim();
    time_embed_dim_ = model_args.time_embed_dim();
    rope_freq_dim_ = model_args.rope_freq_dim();
    rope_theta_ = model_args.rope_theta();
    norm_eps_ = model_args.norm_eps();
    qk_norm_eps_ = model_args.qk_norm_eps();
    final_norm_eps_ = model_args.final_norm_eps();
    CHECK_EQ(patch_size_.size(), static_cast<size_t>(3))
        << "MiniMax-H3 transformer patch_size must have 3 dimensions";
    CHECK_GT(heads_, 0);
    CHECK_GT(head_dim_, 0);
    CHECK_GT(hidden_size_, 0);
    CHECK_GT(num_layers_, 0);
    CHECK_GT(num_refiner_layers_, 0);
    CHECK_GT(ffn_dim_, 0);
    CHECK_GT(in_channels_, 0);
    CHECK_GT(audio_in_channels_, 0);
    CHECK_GT(text_dim_, 0);
    CHECK_GT(freq_dim_, 0);
    CHECK_GT(time_embed_hidden_dim_, 0);
    CHECK_GT(time_embed_dim_, 0);
    CHECK_GT(rope_freq_dim_, 0);
    const int64_t video_patch_dim =
        in_channels_ * patch_size_[0] * patch_size_[1] * patch_size_[2];
    proj_in_ =
        register_module("proj_in",
                        torch::nn::Linear(torch::nn::LinearOptions(
                                              video_patch_dim, hidden_size_)
                                              .bias(true)));
    audio_proj_in_ =
        register_module("audio_proj_in",
                        torch::nn::Linear(torch::nn::LinearOptions(
                                              audio_in_channels_, hidden_size_)
                                              .bias(true)));
    context_embedder_ = register_module(
        "context_embedder",
        torch::nn::Linear(
            torch::nn::LinearOptions(text_dim_, hidden_size_).bias(true)));
    time_embedder_ = register_module(
        "time_embedder",
        H3TimestepEmbedding(
            freq_dim_, time_embed_hidden_dim_, time_embed_dim_));
    rope_ =
        register_module("rope", H3RotaryPosEmbed(rope_freq_dim_, rope_theta_));
    token_refiner_ = register_module("token_refiner",
                                     H3TokenRefiner(hidden_size_,
                                                    heads_,
                                                    head_dim_,
                                                    ffn_dim_,
                                                    num_refiner_layers_,
                                                    norm_eps_,
                                                    qk_norm_eps_,
                                                    final_norm_eps_,
                                                    transformer_context));
    blocks_ = register_module("transformer_blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < num_layers_; ++i) {
      blocks_->push_back(H3TransformerBlock(hidden_size_,
                                            heads_,
                                            head_dim_,
                                            ffn_dim_,
                                            time_embed_dim_,
                                            norm_eps_,
                                            qk_norm_eps_,
                                            transformer_context));
    }
    norm_out_ = register_module(
        "norm_out",
        H3AdaLayerNormOut(hidden_size_, time_embed_dim_, final_norm_eps_));
    proj_out_ =
        register_module("proj_out",
                        torch::nn::Linear(torch::nn::LinearOptions(
                                              hidden_size_, video_patch_dim)
                                              .bias(true)));
    audio_proj_out_ =
        register_module("audio_proj_out",
                        torch::nn::Linear(torch::nn::LinearOptions(
                                              hidden_size_, audio_in_channels_)
                                              .bias(true)));
  }

  void apply_reference_precision(const torch::Device& device) {
    proj_in_->to(device, torch::kFloat32);
    audio_proj_in_->to(device, torch::kFloat32);
    time_embedder_->to(device, torch::kFloat32);
    rope_->to(device, torch::kFloat32);
    proj_out_->to(device, torch::kFloat32);
    audio_proj_out_->to(device, torch::kFloat32);

    context_embedder_->to(device, torch::kBFloat16);
    token_refiner_->to(device, torch::kBFloat16);
    blocks_->to(device, torch::kBFloat16);
    norm_out_->to(device, torch::kBFloat16);

    validate_module_dtype("proj_in", *proj_in_, torch::kFloat32);
    validate_module_dtype("audio_proj_in", *audio_proj_in_, torch::kFloat32);
    validate_module_dtype("time_embedder", *time_embedder_, torch::kFloat32);
    validate_module_dtype("rope", *rope_, torch::kFloat32);
    validate_module_dtype("proj_out", *proj_out_, torch::kFloat32);
    validate_module_dtype("audio_proj_out", *audio_proj_out_, torch::kFloat32);
    validate_module_dtype(
        "context_embedder", *context_embedder_, torch::kBFloat16);
    validate_module_dtype("token_refiner", *token_refiner_, torch::kBFloat16);
    validate_module_dtype("transformer_blocks", *blocks_, torch::kBFloat16);
    validate_module_dtype("norm_out", *norm_out_, torch::kBFloat16);
  }

  MiniMaxH3TransformerOutput forward(const torch::Tensor& hidden_states,
                                     const torch::Tensor& audio_hidden_states,
                                     const torch::Tensor& encoder_hidden_states,
                                     const torch::Tensor& timestep,
                                     const torch::Tensor& timestep_indices,
                                     const torch::Tensor& position_ids,
                                     const torch::Tensor& token_tags,
                                     const torch::Tensor& video_indices,
                                     const torch::Tensor& audio_indices,
                                     const torch::Tensor& text_indices) {
    torch::Tensor video_embeds =
        proj_in_->forward(hidden_states.to(proj_in_->weight.scalar_type()));
    torch::Tensor audio_embeds = audio_proj_in_->forward(
        audio_hidden_states.to(audio_proj_in_->weight.scalar_type()));
    torch::Tensor text_embeds = context_embedder_->forward(
        encoder_hidden_states.to(context_embedder_->weight.scalar_type()));
    text_embeds = token_refiner_->forward(text_embeds);

    const int64_t batch = text_embeds.size(0);
    const int64_t sequence_length = position_ids.size(0);
    torch::Tensor packed = torch::zeros({batch, sequence_length, hidden_size_},
                                        text_embeds.options());
    packed.index_copy_(1, text_indices, text_embeds);
    packed.index_copy_(
        1, video_indices, video_embeds.to(text_embeds.scalar_type()));
    packed.index_copy_(
        1, audio_indices, audio_embeds.to(text_embeds.scalar_type()));
    torch::Tensor timestep_proj =
        minimax_h3::h3_timestep_embedding(timestep.flatten(),
                                          freq_dim_,
                                          /*flip_sin_to_cos=*/true,
                                          /*downscale_freq_shift=*/0.0);
    torch::Tensor temb =
        time_embedder_->forward(timestep_proj.to(torch::kFloat32));
    auto rotary = rope_->forward(position_ids);
    torch::Tensor adaln_indices =
        timestep_indices.to(torch::kLong) * 3 + token_tags.to(torch::kLong);
    for (auto& block : *blocks_) {
      packed = block->as<minimax_h3::H3TransformerBlock>()->forward(
          packed, temb, adaln_indices, rotary.first, rotary.second);
    }

    packed =
        norm_out_->forward(packed, temb, timestep_indices.to(torch::kLong));
    torch::Tensor video_head =
        proj_out_->forward(packed.to(proj_out_->weight.scalar_type()));
    torch::Tensor audio_head = audio_proj_out_->forward(
        packed.to(audio_proj_out_->weight.scalar_type()));
    torch::Tensor video_output = video_head.index_select(1, video_indices);
    torch::Tensor audio_output = audio_head.index_select(1, audio_indices);
    return {video_output, audio_output};
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(
        state_dict, "proj_in.weight", proj_in_->weight, proj_in_weight_loaded_);
    weight::load_weight(
        state_dict, "proj_in.bias", proj_in_->bias, proj_in_bias_loaded_);
    weight::load_weight(state_dict,
                        "audio_proj_in.weight",
                        audio_proj_in_->weight,
                        audio_proj_in_weight_loaded_);
    weight::load_weight(state_dict,
                        "audio_proj_in.bias",
                        audio_proj_in_->bias,
                        audio_proj_in_bias_loaded_);
    weight::load_weight(state_dict,
                        "context_embedder.weight",
                        context_embedder_->weight,
                        context_embedder_weight_loaded_);
    weight::load_weight(state_dict,
                        "context_embedder.bias",
                        context_embedder_->bias,
                        context_embedder_bias_loaded_);
    time_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("time_embedder."));
    token_refiner_->load_state_dict(
        state_dict.get_dict_with_prefix("token_refiner."));
    for (size_t i = 0; i < blocks_->size(); ++i) {
      blocks_[i]->as<minimax_h3::H3TransformerBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("transformer_blocks." +
                                          std::to_string(i) + "."));
    }
    norm_out_->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    weight::load_weight(state_dict,
                        "proj_out.weight",
                        proj_out_->weight,
                        proj_out_weight_loaded_);
    weight::load_weight(
        state_dict, "proj_out.bias", proj_out_->bias, proj_out_bias_loaded_);
    weight::load_weight(state_dict,
                        "audio_proj_out.weight",
                        audio_proj_out_->weight,
                        audio_proj_out_weight_loaded_);
    weight::load_weight(state_dict,
                        "audio_proj_out.bias",
                        audio_proj_out_->bias,
                        audio_proj_out_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(proj_in_weight_loaded_)
        << "weight is not loaded for " << prefix + "proj_in.weight";
    CHECK(proj_in_bias_loaded_)
        << "bias is not loaded for " << prefix + "proj_in.bias";
    CHECK(audio_proj_in_weight_loaded_)
        << "weight is not loaded for " << prefix + "audio_proj_in.weight";
    CHECK(audio_proj_in_bias_loaded_)
        << "bias is not loaded for " << prefix + "audio_proj_in.bias";
    CHECK(context_embedder_weight_loaded_)
        << "weight is not loaded for " << prefix + "context_embedder.weight";
    CHECK(context_embedder_bias_loaded_)
        << "bias is not loaded for " << prefix + "context_embedder.bias";
    time_embedder_->verify_loaded_weights(prefix + "time_embedder.");
    token_refiner_->verify_loaded_weights(prefix + "token_refiner.");
    for (size_t i = 0; i < blocks_->size(); ++i) {
      blocks_[i]->as<minimax_h3::H3TransformerBlock>()->verify_loaded_weights(
          prefix + "transformer_blocks." + std::to_string(i) + ".");
    }
    norm_out_->verify_loaded_weights(prefix + "norm_out.");
    CHECK(proj_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "proj_out.weight";
    CHECK(proj_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "proj_out.bias";
    CHECK(audio_proj_out_weight_loaded_)
        << "weight is not loaded for " << prefix + "audio_proj_out.weight";
    CHECK(audio_proj_out_bias_loaded_)
        << "bias is not loaded for " << prefix + "audio_proj_out.bias";
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    CHECK(loader != nullptr)
        << "MiniMax-H3 transformer loader must not be null";
    for (const auto& state_dict_ptr : loader->get_state_dicts()) {
      load_state_dict(*state_dict_ptr);
    }
    verify_loaded_weights("");
  }

 private:
  static void validate_module_dtype(const std::string& module_name,
                                    const torch::nn::Module& module,
                                    torch::ScalarType expected_dtype) {
    for (const auto& parameter : module.named_parameters(/*recurse=*/true)) {
      CHECK_EQ(parameter.value().scalar_type(), expected_dtype)
          << "MiniMax-H3 parameter dtype mismatch: " << module_name << '.'
          << parameter.key();
    }
    for (const auto& buffer : module.named_buffers(/*recurse=*/true)) {
      if (!buffer.value().is_floating_point()) {
        continue;
      }
      CHECK_EQ(buffer.value().scalar_type(), expected_dtype)
          << "MiniMax-H3 buffer dtype mismatch: " << module_name << '.'
          << buffer.key();
    }
  }

  torch::TensorOptions options_;
  int64_t heads_;
  int64_t head_dim_;
  int64_t hidden_size_;
  int64_t num_layers_;
  int64_t num_refiner_layers_;
  int64_t ffn_dim_;
  int64_t in_channels_;
  int64_t audio_in_channels_;
  std::vector<int64_t> patch_size_;
  int64_t text_dim_;
  int64_t freq_dim_;
  int64_t time_embed_hidden_dim_;
  int64_t time_embed_dim_;
  int64_t rope_freq_dim_;
  double rope_theta_;
  double norm_eps_;
  double qk_norm_eps_;
  double final_norm_eps_;
  torch::nn::Linear proj_in_{nullptr};
  torch::nn::Linear audio_proj_in_{nullptr};
  torch::nn::Linear context_embedder_{nullptr};
  minimax_h3::H3TimestepEmbedding time_embedder_{nullptr};
  minimax_h3::H3RotaryPosEmbed rope_{nullptr};
  minimax_h3::H3TokenRefiner token_refiner_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  minimax_h3::H3AdaLayerNormOut norm_out_{nullptr};
  torch::nn::Linear proj_out_{nullptr};
  torch::nn::Linear audio_proj_out_{nullptr};
  bool proj_in_weight_loaded_ = false;
  bool proj_in_bias_loaded_ = false;
  bool audio_proj_in_weight_loaded_ = false;
  bool audio_proj_in_bias_loaded_ = false;
  bool context_embedder_weight_loaded_ = false;
  bool context_embedder_bias_loaded_ = false;
  bool proj_out_weight_loaded_ = false;
  bool proj_out_bias_loaded_ = false;
  bool audio_proj_out_weight_loaded_ = false;
  bool audio_proj_out_bias_loaded_ = false;
};
TORCH_MODULE(MiniMaxH3Transformer3DModel);

REGISTER_MODEL_ARGS(MiniMaxH3Transformer3DModel, [&] {
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(n_heads, "num_attention_heads", 56);
  LOAD_ARG_OR(head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(hidden_size, "hidden_size", 5376);
  LOAD_ARG_OR(num_layers, "num_layers", 50);
  LOAD_ARG_OR(num_refiner_layers, "num_refiner_layers", 2);
  LOAD_ARG_OR(ffn_dim, "ffn_dim", 14336);
  LOAD_ARG_OR(in_channels, "in_channels", 24);
  LOAD_ARG_OR(audio_in_channels, "audio_in_channels", 32);
  LOAD_ARG_OR(wan_patch_size, "patch_size", (std::vector<int64_t>{1, 2, 2}));
  LOAD_ARG_OR(text_dim, "text_dim", 5120);
  LOAD_ARG_OR(time_freq_dim, "freq_dim", 256);
  LOAD_ARG_OR(time_embed_hidden_dim, "time_embed_hidden_dim", 5376);
  LOAD_ARG_OR(time_embed_dim, "time_embed_dim", 2688);
  LOAD_ARG_OR(rope_freq_dim, "rope_freq_dim", 16);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(norm_eps, "norm_eps", 1e-5f);
  LOAD_ARG_OR(qk_norm_eps, "qk_norm_eps", 1e-5f);
  LOAD_ARG_OR(final_norm_eps, "final_norm_eps", 1e-5f);
});

}  // namespace xllm
