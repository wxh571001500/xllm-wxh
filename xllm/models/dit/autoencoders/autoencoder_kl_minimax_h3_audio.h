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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "models/model_registry.h"

namespace xllm {
namespace minimax_h3_audio {

class Conv1dImpl final : public torch::nn::Module {
 public:
  Conv1dImpl(int64_t in_channels,
             int64_t out_channels,
             int64_t kernel_size,
             int64_t stride,
             int64_t padding,
             int64_t dilation,
             bool bias)
      : stride_(stride), padding_(padding), dilation_(dilation) {
    weight_ = register_parameter(
        "weight", torch::empty({out_channels, in_channels, kernel_size}));
    if (bias) {
      bias_ = register_parameter("bias", torch::empty({out_channels}));
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) const {
    return torch::conv1d(hidden_states,
                         weight_,
                         bias_,
                         /*stride=*/{stride_},
                         /*padding=*/{padding_},
                         /*dilation=*/{dilation_},
                         /*groups=*/1);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", weight_, is_weight_loaded_);
    if (bias_.defined()) {
      weight::load_weight(state_dict, "bias", bias_, is_bias_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_weight_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    if (bias_.defined()) {
      CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix + "bias";
    }
  }

 private:
  int64_t stride_;
  int64_t padding_;
  int64_t dilation_;
  torch::Tensor weight_;
  torch::Tensor bias_;
  bool is_weight_loaded_ = false;
  bool is_bias_loaded_ = false;
};
TORCH_MODULE(Conv1d);

class WeightNormConv1dImpl final : public torch::nn::Module {
 public:
  WeightNormConv1dImpl(int64_t in_channels,
                       int64_t out_channels,
                       int64_t kernel_size,
                       int64_t stride,
                       int64_t padding,
                       int64_t dilation,
                       bool bias)
      : stride_(stride), padding_(padding), dilation_(dilation) {
    weight_g_ =
        register_parameter("weight_g", torch::empty({out_channels, 1, 1}));
    weight_v_ = register_parameter(
        "weight_v", torch::empty({out_channels, in_channels, kernel_size}));
    if (bias) {
      bias_ = register_parameter("bias", torch::empty({out_channels}));
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) const {
    torch::Tensor norm = weight_v_.pow(2).sum({1, 2}, true).sqrt();
    torch::Tensor weight = weight_v_ * weight_g_ / norm;
    return torch::conv1d(hidden_states,
                         weight,
                         bias_,
                         /*stride=*/{stride_},
                         /*padding=*/{padding_},
                         /*dilation=*/{dilation_},
                         /*groups=*/1);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight_g", weight_g_, is_weight_g_loaded_);
    weight::load_weight(state_dict, "weight_v", weight_v_, is_weight_v_loaded_);
    if (bias_.defined()) {
      weight::load_weight(state_dict, "bias", bias_, is_bias_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_weight_g_loaded_)
        << "weight is not loaded for " << prefix + "weight_g";
    CHECK(is_weight_v_loaded_)
        << "weight is not loaded for " << prefix + "weight_v";
    if (bias_.defined()) {
      CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix + "bias";
    }
  }

 private:
  int64_t stride_;
  int64_t padding_;
  int64_t dilation_;
  torch::Tensor weight_g_;
  torch::Tensor weight_v_;
  torch::Tensor bias_;
  bool is_weight_g_loaded_ = false;
  bool is_weight_v_loaded_ = false;
  bool is_bias_loaded_ = false;
};
TORCH_MODULE(WeightNormConv1d);

class Snake1dImpl final : public torch::nn::Module {
 public:
  explicit Snake1dImpl(int64_t channels) {
    alpha_ = register_parameter("alpha", torch::ones({1, channels, 1}));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) const {
    return hidden_states + torch::reciprocal(alpha_ + 1e-9) *
                               torch::sin(alpha_ * hidden_states).pow(2);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "alpha", alpha_, is_alpha_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_alpha_loaded_) << "alpha is not loaded for " << prefix + "alpha";
  }

 private:
  torch::Tensor alpha_;
  bool is_alpha_loaded_ = false;
};
TORCH_MODULE(Snake1d);

class EncoderResidualUnitImpl final : public torch::nn::Module {
 public:
  EncoderResidualUnitImpl(int64_t channels, int64_t dilation) {
    snake_1_ = register_module("snake_1", Snake1d(channels));
    conv_1_ = register_module("conv_1",
                              WeightNormConv1d(channels,
                                               channels,
                                               /*kernel_size=*/7,
                                               /*stride=*/1,
                                               /*padding=*/3 * dilation,
                                               dilation,
                                               /*bias=*/true));
    snake_2_ = register_module("snake_2", Snake1d(channels));
    conv_2_ = register_module("conv_2",
                              WeightNormConv1d(channels,
                                               channels,
                                               /*kernel_size=*/1,
                                               /*stride=*/1,
                                               /*padding=*/0,
                                               /*dilation=*/1,
                                               /*bias=*/true));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) const {
    torch::Tensor residual = snake_1_->forward(hidden_states);
    residual = conv_1_->forward(residual);
    residual = snake_2_->forward(residual);
    residual = conv_2_->forward(residual);
    const int64_t crop = (hidden_states.size(-1) - residual.size(-1)) / 2;
    torch::Tensor shortcut = hidden_states;
    if (crop > 0) {
      shortcut = shortcut.slice(-1, crop, shortcut.size(-1) - crop);
    }
    return shortcut + residual;
  }

  void load_state_dict(const StateDict& state_dict) {
    snake_1_->load_state_dict(state_dict.get_dict_with_prefix("block.0."));
    conv_1_->load_state_dict(state_dict.get_dict_with_prefix("block.1."));
    snake_2_->load_state_dict(state_dict.get_dict_with_prefix("block.2."));
    conv_2_->load_state_dict(state_dict.get_dict_with_prefix("block.3."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    snake_1_->verify_loaded_weights(prefix + "block.0.");
    conv_1_->verify_loaded_weights(prefix + "block.1.");
    snake_2_->verify_loaded_weights(prefix + "block.2.");
    conv_2_->verify_loaded_weights(prefix + "block.3.");
  }

 private:
  Snake1d snake_1_{nullptr};
  WeightNormConv1d conv_1_{nullptr};
  Snake1d snake_2_{nullptr};
  WeightNormConv1d conv_2_{nullptr};
};
TORCH_MODULE(EncoderResidualUnit);

class EncoderBlockImpl final : public torch::nn::Module {
 public:
  EncoderBlockImpl(int64_t output_channels, int64_t stride) {
    const int64_t input_channels = output_channels / 2;
    residual_units_ =
        register_module("residual_units", torch::nn::ModuleList());
    for (int64_t dilation : {1, 3, 9}) {
      residual_units_->push_back(EncoderResidualUnit(input_channels, dilation));
    }
    snake_ = register_module("snake", Snake1d(input_channels));
    downsample_ = register_module("downsample",
                                  WeightNormConv1d(input_channels,
                                                   output_channels,
                                                   /*kernel_size=*/2 * stride,
                                                   stride,
                                                   /*padding=*/(stride + 1) / 2,
                                                   /*dilation=*/1,
                                                   /*bias=*/true));
  }

  torch::Tensor forward(torch::Tensor hidden_states) const {
    for (const auto& unit : *residual_units_) {
      hidden_states = unit->as<EncoderResidualUnit>()->forward(hidden_states);
    }
    return downsample_->forward(snake_->forward(hidden_states));
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t index = 0; index < residual_units_->size(); ++index) {
      residual_units_[index]->as<EncoderResidualUnit>()->load_state_dict(
          state_dict.get_dict_with_prefix("block." + std::to_string(index) +
                                          "."));
    }
    snake_->load_state_dict(state_dict.get_dict_with_prefix("block.3."));
    downsample_->load_state_dict(state_dict.get_dict_with_prefix("block.4."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t index = 0; index < residual_units_->size(); ++index) {
      residual_units_[index]->as<EncoderResidualUnit>()->verify_loaded_weights(
          prefix + "block." + std::to_string(index) + ".");
    }
    snake_->verify_loaded_weights(prefix + "block.3.");
    downsample_->verify_loaded_weights(prefix + "block.4.");
  }

 private:
  torch::nn::ModuleList residual_units_{nullptr};
  Snake1d snake_{nullptr};
  WeightNormConv1d downsample_{nullptr};
};
TORCH_MODULE(EncoderBlock);

class AudioEncoderImpl final : public torch::nn::Module {
 public:
  AudioEncoderImpl(int64_t encoder_dim,
                   int64_t latent_dim,
                   const std::vector<int64_t>& encoder_rates) {
    CHECK_GT(encoder_dim, 0);
    CHECK_GT(latent_dim, 0);
    CHECK(!encoder_rates.empty())
        << "MiniMax-H3 audio VAE encoder_rates must not be empty";
    conv_pre_ = register_module("conv_pre",
                                WeightNormConv1d(/*in_channels=*/1,
                                                 encoder_dim,
                                                 /*kernel_size=*/7,
                                                 /*stride=*/1,
                                                 /*padding=*/3,
                                                 /*dilation=*/1,
                                                 /*bias=*/true));
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    int64_t channels = encoder_dim;
    for (int64_t stride : encoder_rates) {
      channels *= 2;
      blocks_->push_back(EncoderBlock(channels, stride));
    }
    snake_post_ = register_module("snake_post", Snake1d(channels));
    conv_post_ = register_module("conv_post",
                                 WeightNormConv1d(channels,
                                                  latent_dim,
                                                  /*kernel_size=*/3,
                                                  /*stride=*/1,
                                                  /*padding=*/1,
                                                  /*dilation=*/1,
                                                  /*bias=*/true));
  }

  torch::Tensor forward(torch::Tensor hidden_states) const {
    hidden_states = conv_pre_->forward(hidden_states);
    for (const auto& block : *blocks_) {
      hidden_states = block->as<EncoderBlock>()->forward(hidden_states);
    }
    return conv_post_->forward(snake_post_->forward(hidden_states));
  }

  void load_state_dict(const StateDict& state_dict) {
    conv_pre_->load_state_dict(state_dict.get_dict_with_prefix("block.0."));
    for (size_t index = 0; index < blocks_->size(); ++index) {
      blocks_[index]->as<EncoderBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("block." + std::to_string(index + 1) +
                                          "."));
    }
    snake_post_->load_state_dict(state_dict.get_dict_with_prefix(
        "block." + std::to_string(blocks_->size() + 1) + "."));
    conv_post_->load_state_dict(state_dict.get_dict_with_prefix(
        "block." + std::to_string(blocks_->size() + 2) + "."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    conv_pre_->verify_loaded_weights(prefix + "block.0.");
    for (size_t index = 0; index < blocks_->size(); ++index) {
      blocks_[index]->as<EncoderBlock>()->verify_loaded_weights(
          prefix + "block." + std::to_string(index + 1) + ".");
    }
    snake_post_->verify_loaded_weights(
        prefix + "block." + std::to_string(blocks_->size() + 1) + ".");
    conv_post_->verify_loaded_weights(
        prefix + "block." + std::to_string(blocks_->size() + 2) + ".");
  }

 private:
  WeightNormConv1d conv_pre_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  Snake1d snake_post_{nullptr};
  WeightNormConv1d conv_post_{nullptr};
};
TORCH_MODULE(AudioEncoder);

class GeGluMlpImpl final : public torch::nn::Module {
 public:
  GeGluMlpImpl(int64_t input_features, int64_t hidden_features) {
    norm_ = register_module("norm",
                            torch::nn::LayerNorm(torch::nn::LayerNormOptions(
                                std::vector<int64_t>{input_features})));
    w0_ = register_module("w0",
                          torch::nn::Linear(input_features, hidden_features));
    w1_ = register_module("w1",
                          torch::nn::Linear(input_features, hidden_features));
    w2_ = register_module("w2",
                          torch::nn::Linear(hidden_features, input_features));
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    hidden_states = norm_->forward(hidden_states);
    return w2_->forward(torch::gelu(w0_->forward(hidden_states), "tanh") *
                        w1_->forward(hidden_states));
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(
        state_dict, "norm.weight", norm_->weight, is_norm_weight_loaded_);
    weight::load_weight(
        state_dict, "norm.bias", norm_->bias, is_norm_bias_loaded_);
    weight::load_weight(
        state_dict, "w0.weight", w0_->weight, is_w0_weight_loaded_);
    weight::load_weight(state_dict, "w0.bias", w0_->bias, is_w0_bias_loaded_);
    weight::load_weight(
        state_dict, "w1.weight", w1_->weight, is_w1_weight_loaded_);
    weight::load_weight(state_dict, "w1.bias", w1_->bias, is_w1_bias_loaded_);
    weight::load_weight(
        state_dict, "w2.weight", w2_->weight, is_w2_weight_loaded_);
    weight::load_weight(state_dict, "w2.bias", w2_->bias, is_w2_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_norm_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm.weight";
    CHECK(is_norm_bias_loaded_)
        << "bias is not loaded for " << prefix + "norm.bias";
    CHECK(is_w0_weight_loaded_)
        << "weight is not loaded for " << prefix + "w0.weight";
    CHECK(is_w0_bias_loaded_)
        << "bias is not loaded for " << prefix + "w0.bias";
    CHECK(is_w1_weight_loaded_)
        << "weight is not loaded for " << prefix + "w1.weight";
    CHECK(is_w1_bias_loaded_)
        << "bias is not loaded for " << prefix + "w1.bias";
    CHECK(is_w2_weight_loaded_)
        << "weight is not loaded for " << prefix + "w2.weight";
    CHECK(is_w2_bias_loaded_)
        << "bias is not loaded for " << prefix + "w2.bias";
  }

 private:
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Linear w0_{nullptr};
  torch::nn::Linear w1_{nullptr};
  torch::nn::Linear w2_{nullptr};
  bool is_norm_weight_loaded_ = false;
  bool is_norm_bias_loaded_ = false;
  bool is_w0_weight_loaded_ = false;
  bool is_w0_bias_loaded_ = false;
  bool is_w1_weight_loaded_ = false;
  bool is_w1_bias_loaded_ = false;
  bool is_w2_weight_loaded_ = false;
  bool is_w2_bias_loaded_ = false;
};
TORCH_MODULE(GeGluMlp);

class CausalAttentionImpl final : public torch::nn::Module {
 public:
  CausalAttentionImpl(int64_t input_dim, int64_t output_dim, int64_t num_heads)
      : output_dim_(output_dim),
        num_heads_(num_heads),
        head_dim_(input_dim / num_heads) {
    qkv_ = register_module(
        "qkv",
        torch::nn::Linear(
            torch::nn::LinearOptions(input_dim, input_dim * 3).bias(false)));
    q_bias_ = register_parameter("q_bias", torch::zeros({input_dim}));
    v_bias_ = register_parameter("v_bias", torch::zeros({input_dim}));
    proj_ = register_module("proj", torch::nn::Linear(output_dim, output_dim));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    const int64_t batch_size = hidden_states.size(0);
    const int64_t sequence_length = hidden_states.size(1);
    torch::Tensor zero_key_bias = torch::zeros_like(q_bias_);
    torch::Tensor qkv =
        qkv_->forward(hidden_states) +
        torch::cat({q_bias_, zero_key_bias, v_bias_}, 0).view({1, 1, -1});
    std::vector<torch::Tensor> chunks =
        qkv.reshape({batch_size, sequence_length, 3, num_heads_, head_dim_})
            .unbind(2);
    torch::Tensor query = chunks[0].permute({0, 2, 1, 3});
    torch::Tensor key = chunks[1].permute({0, 2, 1, 3});
    torch::Tensor value = chunks[2].permute({0, 2, 1, 3});
    torch::Tensor scores = torch::matmul(query, key.transpose(-2, -1)) /
                           std::sqrt(static_cast<double>(head_dim_));
    torch::Tensor causal_mask = torch::ones({sequence_length, sequence_length},
                                            torch::TensorOptions()
                                                .dtype(torch::kBool)
                                                .device(hidden_states.device()))
                                    .triu(1);
    scores = scores.masked_fill(causal_mask,
                                -std::numeric_limits<float>::infinity());
    torch::Tensor attended = torch::matmul(torch::softmax(scores, -1), value)
                                 .permute({0, 2, 1, 3})
                                 .mean(2);
    attended = torch::adaptive_avg_pool1d(attended, {output_dim_});
    return proj_->forward(attended);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(
        state_dict, "qkv.weight", qkv_->weight, is_qkv_weight_loaded_);
    weight::load_weight(state_dict, "q_bias", q_bias_, is_q_bias_loaded_);
    weight::load_weight(state_dict, "v_bias", v_bias_, is_v_bias_loaded_);
    weight::load_weight(
        state_dict, "proj.weight", proj_->weight, is_proj_weight_loaded_);
    weight::load_weight(
        state_dict, "proj.bias", proj_->bias, is_proj_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_qkv_weight_loaded_)
        << "weight is not loaded for " << prefix + "qkv.weight";
    CHECK(is_q_bias_loaded_) << "bias is not loaded for " << prefix + "q_bias";
    CHECK(is_v_bias_loaded_) << "bias is not loaded for " << prefix + "v_bias";
    CHECK(is_proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "proj.weight";
    CHECK(is_proj_bias_loaded_)
        << "bias is not loaded for " << prefix + "proj.bias";
  }

 private:
  int64_t output_dim_;
  int64_t num_heads_;
  int64_t head_dim_;
  torch::nn::Linear qkv_{nullptr};
  torch::Tensor q_bias_;
  torch::Tensor v_bias_;
  torch::nn::Linear proj_{nullptr};
  bool is_qkv_weight_loaded_ = false;
  bool is_q_bias_loaded_ = false;
  bool is_v_bias_loaded_ = false;
  bool is_proj_weight_loaded_ = false;
  bool is_proj_bias_loaded_ = false;
};
TORCH_MODULE(CausalAttention);

class AttentionProjectionImpl final : public torch::nn::Module {
 public:
  AttentionProjectionImpl(int64_t input_dim,
                          int64_t output_dim,
                          int64_t num_heads) {
    norm_1_ = register_module("norm_1",
                              torch::nn::LayerNorm(torch::nn::LayerNormOptions(
                                  std::vector<int64_t>{input_dim})));
    attention_ = register_module(
        "attention", CausalAttention(input_dim, output_dim, num_heads));
    projection_ =
        register_module("projection", torch::nn::Linear(input_dim, output_dim));
    norm_3_ = register_module("norm_3",
                              torch::nn::LayerNorm(torch::nn::LayerNormOptions(
                                  std::vector<int64_t>{input_dim})));
    norm_2_ = register_module("norm_2",
                              torch::nn::LayerNorm(torch::nn::LayerNormOptions(
                                  std::vector<int64_t>{output_dim})));
    mlp_ = register_module("mlp", GeGluMlp(output_dim, output_dim * 2));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor projected =
        projection_->forward(norm_3_->forward(hidden_states)) +
        attention_->forward(norm_1_->forward(hidden_states));
    return projected + mlp_->forward(norm_2_->forward(projected));
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict.get_dict_with_prefix("norm1."),
                        "weight",
                        norm_1_->weight,
                        is_norm_1_weight_loaded_);
    weight::load_weight(state_dict.get_dict_with_prefix("norm1."),
                        "bias",
                        norm_1_->bias,
                        is_norm_1_bias_loaded_);
    attention_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    weight::load_weight(state_dict.get_dict_with_prefix("proj."),
                        "weight",
                        projection_->weight,
                        is_projection_weight_loaded_);
    weight::load_weight(state_dict.get_dict_with_prefix("proj."),
                        "bias",
                        projection_->bias,
                        is_projection_bias_loaded_);
    weight::load_weight(state_dict.get_dict_with_prefix("norm3."),
                        "weight",
                        norm_3_->weight,
                        is_norm_3_weight_loaded_);
    weight::load_weight(state_dict.get_dict_with_prefix("norm3."),
                        "bias",
                        norm_3_->bias,
                        is_norm_3_bias_loaded_);
    weight::load_weight(state_dict.get_dict_with_prefix("norm2."),
                        "weight",
                        norm_2_->weight,
                        is_norm_2_weight_loaded_);
    weight::load_weight(state_dict.get_dict_with_prefix("norm2."),
                        "bias",
                        norm_2_->bias,
                        is_norm_2_bias_loaded_);
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_norm_1_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm1.weight";
    CHECK(is_norm_1_bias_loaded_)
        << "bias is not loaded for " << prefix + "norm1.bias";
    attention_->verify_loaded_weights(prefix + "attn.");
    CHECK(is_projection_weight_loaded_)
        << "weight is not loaded for " << prefix + "proj.weight";
    CHECK(is_projection_bias_loaded_)
        << "bias is not loaded for " << prefix + "proj.bias";
    CHECK(is_norm_3_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm3.weight";
    CHECK(is_norm_3_bias_loaded_)
        << "bias is not loaded for " << prefix + "norm3.bias";
    CHECK(is_norm_2_weight_loaded_)
        << "weight is not loaded for " << prefix + "norm2.weight";
    CHECK(is_norm_2_bias_loaded_)
        << "bias is not loaded for " << prefix + "norm2.bias";
    mlp_->verify_loaded_weights(prefix + "mlp.");
  }

 private:
  torch::nn::LayerNorm norm_1_{nullptr};
  CausalAttention attention_{nullptr};
  torch::nn::Linear projection_{nullptr};
  torch::nn::LayerNorm norm_3_{nullptr};
  torch::nn::LayerNorm norm_2_{nullptr};
  GeGluMlp mlp_{nullptr};
  bool is_norm_1_weight_loaded_ = false;
  bool is_norm_1_bias_loaded_ = false;
  bool is_projection_weight_loaded_ = false;
  bool is_projection_bias_loaded_ = false;
  bool is_norm_3_weight_loaded_ = false;
  bool is_norm_3_bias_loaded_ = false;
  bool is_norm_2_weight_loaded_ = false;
  bool is_norm_2_bias_loaded_ = false;
};
TORCH_MODULE(AttentionProjection);

class WeightNormConvTranspose1dImpl final : public torch::nn::Module {
 public:
  WeightNormConvTranspose1dImpl(int64_t in_channels,
                                int64_t out_channels,
                                int64_t kernel_size,
                                int64_t stride,
                                int64_t padding)
      : stride_(stride), padding_(padding) {
    weight_g_ =
        register_parameter("weight_g", torch::empty({in_channels, 1, 1}));
    weight_v_ = register_parameter(
        "weight_v", torch::empty({in_channels, out_channels, kernel_size}));
    bias_ = register_parameter("bias", torch::empty({out_channels}));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) const {
    torch::Tensor norm = weight_v_.pow(2).sum({1, 2}, true).sqrt();
    torch::Tensor weight = weight_v_ * weight_g_ / norm;
    return torch::conv_transpose1d(hidden_states,
                                   weight,
                                   bias_,
                                   /*stride=*/{stride_},
                                   /*padding=*/{padding_},
                                   /*output_padding=*/{0},
                                   /*groups=*/1,
                                   /*dilation=*/{1});
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight_g", weight_g_, is_weight_g_loaded_);
    weight::load_weight(state_dict, "weight_v", weight_v_, is_weight_v_loaded_);
    weight::load_weight(state_dict, "bias", bias_, is_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_weight_g_loaded_)
        << "weight is not loaded for " << prefix + "weight_g";
    CHECK(is_weight_v_loaded_)
        << "weight is not loaded for " << prefix + "weight_v";
    CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix + "bias";
  }

 private:
  int64_t stride_;
  int64_t padding_;
  torch::Tensor weight_g_;
  torch::Tensor weight_v_;
  torch::Tensor bias_;
  bool is_weight_g_loaded_ = false;
  bool is_weight_v_loaded_ = false;
  bool is_bias_loaded_ = false;
};
TORCH_MODULE(WeightNormConvTranspose1d);

class SnakeBetaImpl final : public torch::nn::Module {
 public:
  explicit SnakeBetaImpl(int64_t channels) {
    alpha_ = register_parameter("alpha", torch::zeros({channels}));
    beta_ = register_parameter("beta", torch::zeros({channels}));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) const {
    torch::Tensor alpha = torch::exp(alpha_.view({1, -1, 1}));
    torch::Tensor beta = torch::exp(beta_.view({1, -1, 1}));
    return hidden_states + torch::reciprocal(beta + 1e-9) *
                               torch::sin(alpha * hidden_states).pow(2);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "alpha", alpha_, is_alpha_loaded_);
    weight::load_weight(state_dict, "beta", beta_, is_beta_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_alpha_loaded_) << "alpha is not loaded for " << prefix + "alpha";
    CHECK(is_beta_loaded_) << "beta is not loaded for " << prefix + "beta";
  }

 private:
  torch::Tensor alpha_;
  torch::Tensor beta_;
  bool is_alpha_loaded_ = false;
  bool is_beta_loaded_ = false;
};
TORCH_MODULE(SnakeBeta);

class LowPassFilter1dImpl final : public torch::nn::Module {
 public:
  LowPassFilter1dImpl(int64_t stride, int64_t kernel_size)
      : stride_(stride),
        pad_left_(kernel_size / 2 - static_cast<int64_t>(kernel_size % 2 == 0)),
        pad_right_(kernel_size / 2) {
    filter_ = register_buffer("filter", torch::zeros({1, 1, kernel_size}));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) const {
    torch::Tensor padded = torch::nn::functional::pad(
        hidden_states,
        torch::nn::functional::PadFuncOptions({pad_left_, pad_right_})
            .mode(torch::kReplicate));
    const int64_t channels = hidden_states.size(1);
    const std::array<int64_t, 1> stride = {stride_};
    const std::array<int64_t, 1> padding = {0};
    const std::array<int64_t, 1> dilation = {1};
    return torch::conv1d(padded,
                         filter_.expand({channels, -1, -1}),
                         /*bias=*/std::nullopt,
                         stride,
                         padding,
                         dilation,
                         /*groups=*/channels);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "filter", filter_, is_filter_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_filter_loaded_)
        << "filter is not loaded for " << prefix + "filter";
  }

 private:
  int64_t stride_;
  int64_t pad_left_;
  int64_t pad_right_;
  torch::Tensor filter_;
  bool is_filter_loaded_ = false;
};
TORCH_MODULE(LowPassFilter1d);

class DownSample1dImpl final : public torch::nn::Module {
 public:
  DownSample1dImpl(int64_t ratio, int64_t kernel_size) {
    lowpass_ = register_module("lowpass", LowPassFilter1d(ratio, kernel_size));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) const {
    return lowpass_->forward(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    lowpass_->load_state_dict(state_dict.get_dict_with_prefix("lowpass."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    lowpass_->verify_loaded_weights(prefix + "lowpass.");
  }

 private:
  LowPassFilter1d lowpass_{nullptr};
};
TORCH_MODULE(DownSample1d);

class UpSample1dImpl final : public torch::nn::Module {
 public:
  UpSample1dImpl(int64_t ratio, int64_t kernel_size)
      : ratio_(ratio),
        pad_(kernel_size / ratio - 1),
        pad_left_(pad_ * ratio + (kernel_size - ratio) / 2),
        pad_right_(pad_ * ratio + (kernel_size - ratio + 1) / 2) {
    filter_ = register_buffer("filter", torch::zeros({1, 1, kernel_size}));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) const {
    torch::Tensor padded = torch::nn::functional::pad(
        hidden_states,
        torch::nn::functional::PadFuncOptions({pad_, pad_})
            .mode(torch::kReplicate));
    const int64_t channels = hidden_states.size(1);
    torch::Tensor output =
        ratio_ * torch::conv_transpose1d(padded,
                                         filter_.expand({channels, -1, -1}),
                                         torch::Tensor(),
                                         /*stride=*/{ratio_},
                                         /*padding=*/{0},
                                         /*output_padding=*/{0},
                                         /*groups=*/channels,
                                         /*dilation=*/{1});
    return output.slice(-1, pad_left_, output.size(-1) - pad_right_);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "filter", filter_, is_filter_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(is_filter_loaded_)
        << "filter is not loaded for " << prefix + "filter";
  }

 private:
  int64_t ratio_;
  int64_t pad_;
  int64_t pad_left_;
  int64_t pad_right_;
  torch::Tensor filter_;
  bool is_filter_loaded_ = false;
};
TORCH_MODULE(UpSample1d);

class Activation1dImpl final : public torch::nn::Module {
 public:
  explicit Activation1dImpl(int64_t channels) {
    activation_ = register_module("act", SnakeBeta(channels));
    upsample_ = register_module("upsample",
                                UpSample1d(/*ratio=*/2, /*kernel_size=*/12));
    downsample_ = register_module(
        "downsample", DownSample1d(/*ratio=*/2, /*kernel_size=*/12));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    return downsample_->forward(
        activation_->forward(upsample_->forward(hidden_states)));
  }

  void load_state_dict(const StateDict& state_dict) {
    activation_->load_state_dict(state_dict.get_dict_with_prefix("act."));
    upsample_->load_state_dict(state_dict.get_dict_with_prefix("upsample."));
    downsample_->load_state_dict(
        state_dict.get_dict_with_prefix("downsample."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    activation_->verify_loaded_weights(prefix + "act.");
    upsample_->verify_loaded_weights(prefix + "upsample.");
    downsample_->verify_loaded_weights(prefix + "downsample.");
  }

 private:
  SnakeBeta activation_{nullptr};
  UpSample1d upsample_{nullptr};
  DownSample1d downsample_{nullptr};
};
TORCH_MODULE(Activation1d);

class AMPBlockImpl final : public torch::nn::Module {
 public:
  AMPBlockImpl(int64_t channels,
               int64_t kernel_size,
               const std::vector<int64_t>& dilations) {
    convs1_ = register_module("convs1", torch::nn::ModuleList());
    convs2_ = register_module("convs2", torch::nn::ModuleList());
    activations_ = register_module("activations", torch::nn::ModuleList());
    dilations_ = dilations;
    CHECK(!dilations_.empty())
        << "MiniMax-H3 audio VAE AMPBlock dilations must not be empty";
    for (int64_t dilation : dilations_) {
      convs1_->push_back(
          WeightNormConv1d(channels,
                           channels,
                           kernel_size,
                           /*stride=*/1,
                           (kernel_size * dilation - dilation) / 2,
                           dilation,
                           /*bias=*/true));
      convs2_->push_back(WeightNormConv1d(channels,
                                          channels,
                                          kernel_size,
                                          /*stride=*/1,
                                          (kernel_size - 1) / 2,
                                          /*dilation=*/1,
                                          /*bias=*/true));
      activations_->push_back(Activation1d(channels));
      activations_->push_back(Activation1d(channels));
    }
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    for (size_t index = 0; index < dilations_.size(); ++index) {
      torch::Tensor residual = convs1_[index]->as<WeightNormConv1d>()->forward(
          activations_[2 * index]->as<Activation1d>()->forward(hidden_states));
      residual = convs2_[index]->as<WeightNormConv1d>()->forward(
          activations_[2 * index + 1]->as<Activation1d>()->forward(residual));
      hidden_states = hidden_states + residual;
    }
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t index = 0; index < dilations_.size(); ++index) {
      convs1_[index]->as<WeightNormConv1d>()->load_state_dict(
          state_dict.get_dict_with_prefix("convs1." + std::to_string(index) +
                                          "."));
      convs2_[index]->as<WeightNormConv1d>()->load_state_dict(
          state_dict.get_dict_with_prefix("convs2." + std::to_string(index) +
                                          "."));
      activations_[2 * index]->as<Activation1d>()->load_state_dict(
          state_dict.get_dict_with_prefix("activations." +
                                          std::to_string(2 * index) + "."));
      activations_[2 * index + 1]->as<Activation1d>()->load_state_dict(
          state_dict.get_dict_with_prefix("activations." +
                                          std::to_string(2 * index + 1) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    for (size_t index = 0; index < dilations_.size(); ++index) {
      convs1_[index]->as<WeightNormConv1d>()->verify_loaded_weights(
          prefix + "convs1." + std::to_string(index) + ".");
      convs2_[index]->as<WeightNormConv1d>()->verify_loaded_weights(
          prefix + "convs2." + std::to_string(index) + ".");
      activations_[2 * index]->as<Activation1d>()->verify_loaded_weights(
          prefix + "activations." + std::to_string(2 * index) + ".");
      activations_[2 * index + 1]->as<Activation1d>()->verify_loaded_weights(
          prefix + "activations." + std::to_string(2 * index + 1) + ".");
    }
  }

 private:
  std::vector<int64_t> dilations_;
  torch::nn::ModuleList convs1_{nullptr};
  torch::nn::ModuleList convs2_{nullptr};
  torch::nn::ModuleList activations_{nullptr};
};
TORCH_MODULE(AMPBlock);

class UpsampleStageImpl final : public torch::nn::Module {
 public:
  UpsampleStageImpl(int64_t in_channels,
                    int64_t out_channels,
                    int64_t kernel_size,
                    int64_t stride,
                    int64_t padding) {
    convolution_ = register_module(
        "0",
        WeightNormConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    return convolution_->forward(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    convolution_->load_state_dict(state_dict.get_dict_with_prefix("0."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    convolution_->verify_loaded_weights(prefix + "0.");
  }

 private:
  WeightNormConvTranspose1d convolution_{nullptr};
};
TORCH_MODULE(UpsampleStage);

class BigVGANDecoderImpl final : public torch::nn::Module {
 public:
  BigVGANDecoderImpl(int64_t latent_dim,
                     int64_t decoder_dim,
                     const std::vector<int64_t>& upsample_rates,
                     const std::vector<int64_t>& upsample_kernels,
                     const std::vector<int64_t>& residual_kernels,
                     const std::vector<int64_t>& residual_dilations) {
    CHECK_GT(latent_dim, 0);
    CHECK_GT(decoder_dim, 0);
    CHECK(!upsample_rates.empty())
        << "MiniMax-H3 audio VAE decoder_rates must not be empty";
    CHECK_EQ(upsample_rates.size(), upsample_kernels.size());
    CHECK(!residual_kernels.empty())
        << "MiniMax-H3 audio VAE resblock_kernel_sizes must not be empty";
    upsample_rates_ = upsample_rates;
    upsample_kernels_ = upsample_kernels;
    residual_kernels_ = residual_kernels;
    residual_dilations_ = residual_dilations;
    conv_pre_ = register_module("conv_pre",
                                WeightNormConv1d(latent_dim,
                                                 decoder_dim,
                                                 /*kernel_size=*/7,
                                                 /*stride=*/1,
                                                 /*padding=*/3,
                                                 /*dilation=*/1,
                                                 /*bias=*/true));
    upsamplers_ = register_module("ups", torch::nn::ModuleList());
    residual_blocks_ = register_module("resblocks", torch::nn::ModuleList());
    int64_t channels = decoder_dim;
    for (size_t stage = 0; stage < upsample_rates_.size(); ++stage) {
      const int64_t in_channels = channels;
      const int64_t out_channels = channels / 2;
      CHECK_GT(out_channels, 0)
          << "MiniMax-H3 audio VAE decoder channel count must be positive";
      channels = out_channels;
      upsamplers_->push_back(UpsampleStage(
          in_channels,
          out_channels,
          upsample_kernels_[stage],
          upsample_rates_[stage],
          (upsample_kernels_[stage] - upsample_rates_[stage]) / 2));
      for (int64_t kernel_size : residual_kernels_) {
        residual_blocks_->push_back(
            AMPBlock(out_channels, kernel_size, residual_dilations_));
      }
    }
    activation_post_ =
        register_module("activation_post", Activation1d(channels));
    conv_post_ = register_module("conv_post",
                                 WeightNormConv1d(channels,
                                                  /*out_channels=*/1,
                                                  /*kernel_size=*/7,
                                                  /*stride=*/1,
                                                  /*padding=*/3,
                                                  /*dilation=*/1,
                                                  /*bias=*/false));
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    hidden_states = conv_pre_->forward(hidden_states);
    for (size_t stage = 0; stage < upsample_rates_.size(); ++stage) {
      hidden_states =
          upsamplers_[stage]->as<UpsampleStage>()->forward(hidden_states);
      torch::Tensor residual;
      for (size_t kernel = 0; kernel < residual_kernels_.size(); ++kernel) {
        torch::Tensor block =
            residual_blocks_[stage * residual_kernels_.size() + kernel]
                ->as<AMPBlock>()
                ->forward(hidden_states);
        residual = residual.defined() ? residual + block : block;
      }
      hidden_states = residual / static_cast<double>(residual_kernels_.size());
    }
    hidden_states = activation_post_->forward(hidden_states);
    return torch::clamp(conv_post_->forward(hidden_states), -1.0, 1.0);
  }

  void load_state_dict(const StateDict& state_dict) {
    conv_pre_->load_state_dict(state_dict.get_dict_with_prefix("conv_pre."));
    for (size_t stage = 0; stage < upsample_rates_.size(); ++stage) {
      upsamplers_[stage]->as<UpsampleStage>()->load_state_dict(
          state_dict.get_dict_with_prefix("ups." + std::to_string(stage) +
                                          "."));
    }
    for (size_t block = 0;
         block < upsample_rates_.size() * residual_kernels_.size();
         ++block) {
      residual_blocks_[block]->as<AMPBlock>()->load_state_dict(
          state_dict.get_dict_with_prefix("resblocks." + std::to_string(block) +
                                          "."));
    }
    activation_post_->load_state_dict(
        state_dict.get_dict_with_prefix("activation_post."));
    conv_post_->load_state_dict(state_dict.get_dict_with_prefix("conv_post."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    conv_pre_->verify_loaded_weights(prefix + "conv_pre.");
    for (size_t stage = 0; stage < upsample_rates_.size(); ++stage) {
      upsamplers_[stage]->as<UpsampleStage>()->verify_loaded_weights(
          prefix + "ups." + std::to_string(stage) + ".");
    }
    for (size_t block = 0;
         block < upsample_rates_.size() * residual_kernels_.size();
         ++block) {
      residual_blocks_[block]->as<AMPBlock>()->verify_loaded_weights(
          prefix + "resblocks." + std::to_string(block) + ".");
    }
    activation_post_->verify_loaded_weights(prefix + "activation_post.");
    conv_post_->verify_loaded_weights(prefix + "conv_post.");
  }

 private:
  std::vector<int64_t> upsample_rates_;
  std::vector<int64_t> upsample_kernels_;
  std::vector<int64_t> residual_kernels_;
  std::vector<int64_t> residual_dilations_;
  WeightNormConv1d conv_pre_{nullptr};
  torch::nn::ModuleList upsamplers_{nullptr};
  torch::nn::ModuleList residual_blocks_{nullptr};
  Activation1d activation_post_{nullptr};
  WeightNormConv1d conv_post_{nullptr};
};
TORCH_MODULE(BigVGANDecoder);

}  // namespace minimax_h3_audio

class AutoencoderKLMiniMaxH3AudioImpl final : public torch::nn::Module {
 public:
  explicit AutoencoderKLMiniMaxH3AudioImpl(const ModelContext& context) {
    const ModelArgs& args = context.get_model_args();
    encoder_dim_ = args.encoder_dim();
    latent_dim_ = args.latent_dim();
    latent_channels_ = args.latent_channels();
    decoder_dim_ = args.decoder_dim();
    sampling_rate_ = static_cast<int32_t>(args.sampling_rate());
    latents_mean_ = args.latents_mean();
    latents_std_ = args.latents_std();
    CHECK_GT(encoder_dim_, 0);
    CHECK_GT(latent_dim_, 0);
    CHECK_GT(latent_channels_, 0);
    CHECK_GT(decoder_dim_, 0);
    CHECK_GT(args.num_attention_heads(), 0);
    CHECK_GT(sampling_rate_, 0);
    CHECK_EQ(latent_dim_ % latent_channels_, 0)
        << "MiniMax-H3 audio VAE latent_dim must be a multiple of "
           "latent_channels";
    const auto rate_product = [](const std::vector<int64_t>& rates) {
      CHECK(!rates.empty()) << "MiniMax-H3 audio VAE rates must not be empty";
      int64_t product = 1;
      for (int64_t rate : rates) {
        CHECK_GT(rate, 0);
        CHECK_LE(product, std::numeric_limits<int64_t>::max() / rate)
            << "MiniMax-H3 audio VAE rate product overflows";
        product *= rate;
      }
      return product;
    };
    hop_length_ = rate_product(args.encoder_rates());
    CHECK_EQ(rate_product(args.decoder_rates()), hop_length_)
        << "MiniMax-H3 audio VAE encoder and decoder rate products must match";
    CHECK_EQ(latents_mean_.size(), static_cast<size_t>(latent_channels_));
    CHECK_EQ(latents_std_.size(), static_cast<size_t>(latent_channels_));
    encoder_ =
        register_module("encoder",
                        minimax_h3_audio::AudioEncoder(
                            encoder_dim_, latent_dim_, args.encoder_rates()));
    pre_block_ = register_module(
        "pre_block",
        minimax_h3_audio::AttentionProjection(
            latent_dim_, latent_channels_, args.num_attention_heads()));
    mean_proj_ = register_module("mean_proj",
                                 minimax_h3_audio::Conv1d(latent_channels_,
                                                          latent_channels_,
                                                          /*kernel_size=*/1,
                                                          /*stride=*/1,
                                                          /*padding=*/0,
                                                          /*dilation=*/1,
                                                          /*bias=*/true));
    logs_proj_ = register_module("logs_proj",
                                 minimax_h3_audio::Conv1d(latent_channels_,
                                                          latent_channels_,
                                                          /*kernel_size=*/1,
                                                          /*stride=*/1,
                                                          /*padding=*/0,
                                                          /*dilation=*/1,
                                                          /*bias=*/true));
    dec_in_proj_ = register_module("dec_in_proj",
                                   minimax_h3_audio::Conv1d(latent_channels_,
                                                            latent_dim_,
                                                            /*kernel_size=*/1,
                                                            /*stride=*/1,
                                                            /*padding=*/0,
                                                            /*dilation=*/1,
                                                            /*bias=*/true));
    decoder_ = register_module(
        "decoder",
        minimax_h3_audio::BigVGANDecoder(latent_dim_,
                                         decoder_dim_,
                                         args.decoder_rates(),
                                         args.decoder_kernel_sizes(),
                                         args.resblock_kernel_sizes(),
                                         default_residual_dilations()));
  }

  torch::Tensor encode_mode(torch::Tensor sample) {
    CHECK_EQ(sample.dim(), 3)
        << "MiniMax-H3 audio VAE encoder expects [B,1,S] waveform";
    CHECK_EQ(sample.size(1), 1);
    const int64_t right_pad =
        (hop_length() - sample.size(2) % hop_length()) % hop_length();
    if (right_pad > 0) {
      sample = torch::pad(sample, {0, right_pad}, "constant", 0.0);
    }
    torch::Tensor hidden_states = encoder_->forward(sample.to(torch::kFloat32));
    hidden_states =
        pre_block_->forward(hidden_states.transpose(1, 2)).transpose(1, 2);
    return mean_proj_->forward(hidden_states).to(torch::kFloat32);
  }

  torch::Tensor decode(const torch::Tensor& latent) {
    CHECK_EQ(latent.dim(), 3) << "MiniMax-H3 audio VAE expects [B,C,T] latents";
    CHECK_EQ(latent.size(1), latent_channels_)
        << "MiniMax-H3 audio VAE latent channel mismatch";
    return decoder_->forward(dec_in_proj_->forward(latent));
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    CHECK(loader != nullptr) << "MiniMax-H3 audio VAE loader must not be null";
    load_config(loader->model_weights_path() + "/config.json");
    for (const auto& state_dict : loader->get_state_dicts()) {
      encoder_->load_state_dict(state_dict->get_dict_with_prefix("encoder."));
      pre_block_->load_state_dict(
          state_dict->get_dict_with_prefix("pre_block."));
      mean_proj_->load_state_dict(
          state_dict->get_dict_with_prefix("mean_proj."));
      logs_proj_->load_state_dict(
          state_dict->get_dict_with_prefix("logs_proj."));
      dec_in_proj_->load_state_dict(
          state_dict->get_dict_with_prefix("dec_in_proj."));
      decoder_->load_state_dict(state_dict->get_dict_with_prefix("decoder."));
    }
    verify_loaded_weights("");
  }

  void verify_loaded_weights(const std::string& prefix) {
    encoder_->verify_loaded_weights(prefix + "encoder.");
    pre_block_->verify_loaded_weights(prefix + "pre_block.");
    mean_proj_->verify_loaded_weights(prefix + "mean_proj.");
    logs_proj_->verify_loaded_weights(prefix + "logs_proj.");
    dec_in_proj_->verify_loaded_weights(prefix + "dec_in_proj.");
    decoder_->verify_loaded_weights(prefix + "decoder.");
  }

  const std::vector<double>& latents_mean() const { return latents_mean_; }
  const std::vector<double>& latents_std() const { return latents_std_; }
  int32_t sampling_rate() const { return sampling_rate_; }
  int64_t latent_channels() const { return latent_channels_; }
  int64_t hop_length() const { return hop_length_; }

 private:
  void load_config(const std::string& path) {
    std::ifstream input(path);
    CHECK(input.good()) << "MiniMax-H3 audio VAE config not found: " << path;
    nlohmann::json config;
    input >> config;
    CHECK_EQ(config.at("encoder_dim").get<int64_t>(), encoder_dim_);
    CHECK_EQ(config.at("latent_dim").get<int64_t>(), latent_dim_);
    CHECK_EQ(config.at("latent_channels").get<int64_t>(), latent_channels_);
    CHECK_EQ(config.at("decoder_dim").get<int64_t>(), decoder_dim_);
    CHECK_EQ(config.at("sampling_rate").get<int32_t>(), sampling_rate_);
    if (config.contains("resblock_dilation_sizes")) {
      const std::vector<int64_t> expected_dilations =
          default_residual_dilations();
      for (const auto& dilation_row : config.at("resblock_dilation_sizes")) {
        const std::vector<int64_t> dilation_values =
            dilation_row.get<std::vector<int64_t>>();
        CHECK(dilation_values == expected_dilations)
            << "MiniMax-H3 audio VAE uses one shared AMPBlock dilation layout";
      }
    }
    CHECK_EQ(latents_mean_.size(), static_cast<size_t>(latent_channels_));
    CHECK_EQ(latents_std_.size(), static_cast<size_t>(latent_channels_));
  }

  static std::vector<int64_t> default_residual_dilations() { return {1, 3, 5}; }

  int64_t hop_length_ = 0;
  int64_t encoder_dim_ = 0;
  int64_t latent_dim_ = 0;
  int64_t latent_channels_ = 0;
  int64_t decoder_dim_ = 0;
  minimax_h3_audio::AudioEncoder encoder_{nullptr};
  minimax_h3_audio::AttentionProjection pre_block_{nullptr};
  minimax_h3_audio::Conv1d mean_proj_{nullptr};
  minimax_h3_audio::Conv1d logs_proj_{nullptr};
  minimax_h3_audio::Conv1d dec_in_proj_{nullptr};
  minimax_h3_audio::BigVGANDecoder decoder_{nullptr};
  std::vector<double> latents_mean_;
  std::vector<double> latents_std_;
  int32_t sampling_rate_ = 32000;
};
TORCH_MODULE(AutoencoderKLMiniMaxH3Audio);

REGISTER_MODEL_ARGS(AutoencoderKLMiniMaxH3Audio, [&] {
  LOAD_ARG_OR(encoder_dim, "encoder_dim", 64);
  LOAD_ARG_OR(
      encoder_rates, "encoder_rates", (std::vector<int64_t>{2, 4, 4, 5, 5}));
  LOAD_ARG_OR(latent_dim, "latent_dim", 2048);
  LOAD_ARG_OR(latent_channels, "latent_channels", 32);
  LOAD_ARG_OR(decoder_dim, "decoder_dim", 1024);
  LOAD_ARG_OR(decoder_rates,
              "decoder_rates",
              (std::vector<int64_t>{5, 5, 2, 2, 2, 2, 2}));
  LOAD_ARG_OR(decoder_kernel_sizes,
              "decoder_kernel_sizes",
              (std::vector<int64_t>{9, 9, 4, 4, 4, 4, 4}));
  LOAD_ARG_OR(num_attention_heads, "num_attention_heads", 8);
  LOAD_ARG_OR(resblock_kernel_sizes,
              "resblock_kernel_sizes",
              (std::vector<int64_t>{3, 7, 11}));
  LOAD_ARG_OR(sampling_rate, "sampling_rate", 32000);
  LOAD_ARG_OR(latents_mean, "latents_mean", (std::vector<double>(32, 0.0)));
  LOAD_ARG_OR(latents_std, "latents_std", (std::vector<double>(32, 1.0)));
});

}  // namespace xllm
