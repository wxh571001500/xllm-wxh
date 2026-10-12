/* Copyright 2026 The xLLM Authors.

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

// JoyOV2 audio VAE (AutoencoderKLXDAC). Decoder-only port of SGLang/diffusers
// XDAC. Checkpoint uses torch.nn.utils.parametrizations.weight_norm keys:
//   *.parametrizations.weight.original0  (= weight_g)
//   *.parametrizations.weight.original1  (= weight_v)

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "models/model_registry.h"

namespace xllm {
namespace joyo_xdac {

// Avoid /0 when weight_v is all zeros (PyTorch WeightNorm has the same edge).
constexpr double kWeightNormEps = 1e-8;

inline void load_wn_weight(const StateDict& state_dict,
                           torch::Tensor& weight_g,
                           torch::Tensor& weight_v,
                           bool& g_loaded,
                           bool& v_loaded) {
  // Prefer modern parametrizations keys; fall back to weight_g / weight_v.
  auto g = state_dict.get_tensor("parametrizations.weight.original0");
  auto v = state_dict.get_tensor("parametrizations.weight.original1");
  if (!g.defined()) {
    g = state_dict.get_tensor("weight_g");
  }
  if (!v.defined()) {
    v = state_dict.get_tensor("weight_v");
  }
  if (g.defined()) {
    CHECK_EQ(g.sizes(), weight_g.sizes());
    weight_g.data().copy_(g);
    g_loaded = true;
  }
  if (v.defined()) {
    CHECK_EQ(v.sizes(), weight_v.sizes());
    weight_v.data().copy_(v);
    v_loaded = true;
  }
}

class WeightNormConv1dImpl : public torch::nn::Module {
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

  torch::Tensor forward(const torch::Tensor& x) const {
    auto norm =
        weight_v_.pow(2).sum({1, 2}, /*keepdim=*/true).sqrt() + kWeightNormEps;
    auto weight = weight_v_ * weight_g_ / norm;
    return torch::conv1d(x,
                         weight,
                         bias_,
                         /*stride=*/{stride_},
                         /*padding=*/{padding_},
                         /*dilation=*/{dilation_},
                         /*groups=*/1);
  }

  void load_state_dict(const StateDict& state_dict) {
    load_wn_weight(state_dict,
                   weight_g_,
                   weight_v_,
                   is_weight_g_loaded_,
                   is_weight_v_loaded_);
    if (bias_.defined()) {
      weight::load_weight(state_dict, "bias", bias_, is_bias_loaded_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_weight_g_loaded_)
        << "weight_g is not loaded for " << prefix << "weight_g";
    CHECK(is_weight_v_loaded_)
        << "weight_v is not loaded for " << prefix << "weight_v";
    if (bias_.defined()) {
      CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix << "bias";
    }
  }

 private:
  int64_t stride_ = 1;
  int64_t padding_ = 0;
  int64_t dilation_ = 1;
  torch::Tensor weight_g_;
  torch::Tensor weight_v_;
  torch::Tensor bias_;
  bool is_weight_g_loaded_ = false;
  bool is_weight_v_loaded_ = false;
  bool is_bias_loaded_ = false;
};
TORCH_MODULE(WeightNormConv1d);

class WeightNormConvTranspose1dImpl : public torch::nn::Module {
 public:
  WeightNormConvTranspose1dImpl(int64_t in_channels,
                                int64_t out_channels,
                                int64_t kernel_size,
                                int64_t stride,
                                int64_t padding,
                                int64_t output_padding)
      : stride_(stride), padding_(padding), output_padding_(output_padding) {
    weight_g_ =
        register_parameter("weight_g", torch::empty({in_channels, 1, 1}));
    weight_v_ = register_parameter(
        "weight_v", torch::empty({in_channels, out_channels, kernel_size}));
    bias_ = register_parameter("bias", torch::empty({out_channels}));
  }

  torch::Tensor forward(const torch::Tensor& x) const {
    auto norm =
        weight_v_.pow(2).sum({1, 2}, /*keepdim=*/true).sqrt() + kWeightNormEps;
    auto weight = weight_v_ * weight_g_ / norm;
    return torch::conv_transpose1d(x,
                                   weight,
                                   bias_,
                                   /*stride=*/{stride_},
                                   /*padding=*/{padding_},
                                   /*output_padding=*/{output_padding_},
                                   /*groups=*/1,
                                   /*dilation=*/{1});
  }

  void load_state_dict(const StateDict& state_dict) {
    load_wn_weight(state_dict,
                   weight_g_,
                   weight_v_,
                   is_weight_g_loaded_,
                   is_weight_v_loaded_);
    weight::load_weight(state_dict, "bias", bias_, is_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_weight_g_loaded_)
        << "weight_g is not loaded for " << prefix << "weight_g";
    CHECK(is_weight_v_loaded_)
        << "weight_v is not loaded for " << prefix << "weight_v";
    CHECK(is_bias_loaded_) << "bias is not loaded for " << prefix << "bias";
  }

 private:
  int64_t stride_ = 1;
  int64_t padding_ = 0;
  int64_t output_padding_ = 0;
  torch::Tensor weight_g_;
  torch::Tensor weight_v_;
  torch::Tensor bias_;
  bool is_weight_g_loaded_ = false;
  bool is_weight_v_loaded_ = false;
  bool is_bias_loaded_ = false;
};
TORCH_MODULE(WeightNormConvTranspose1d);

class Snake1dImpl : public torch::nn::Module {
 public:
  explicit Snake1dImpl(int64_t channels) {
    alpha_ = register_parameter("alpha", torch::ones({1, channels, 1}));
  }

  torch::Tensor forward(const torch::Tensor& x) const {
    return x + torch::reciprocal(alpha_ + 1e-9) * torch::sin(alpha_ * x).pow(2);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "alpha", alpha_, is_alpha_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_alpha_loaded_) << "alpha is not loaded for " << prefix << "alpha";
  }

 private:
  torch::Tensor alpha_;
  bool is_alpha_loaded_ = false;
};
TORCH_MODULE(Snake1d);

class ResidualUnitImpl : public torch::nn::Module {
 public:
  ResidualUnitImpl(int64_t dim, int64_t dilation) {
    const int64_t pad = ((7 - 1) * dilation) / 2;
    snake1_ = register_module("snake1", Snake1d(dim));
    conv1_ = register_module("conv1",
                             WeightNormConv1d(dim,
                                              dim,
                                              /*kernel_size=*/7,
                                              /*stride=*/1,
                                              pad,
                                              dilation,
                                              /*bias=*/true));
    snake2_ = register_module("snake2", Snake1d(dim));
    conv2_ = register_module("conv2",
                             WeightNormConv1d(dim,
                                              dim,
                                              /*kernel_size=*/1,
                                              /*stride=*/1,
                                              /*padding=*/0,
                                              /*dilation=*/1,
                                              /*bias=*/true));
  }

  torch::Tensor forward(const torch::Tensor& x) const {
    auto y =
        conv2_->forward(snake2_->forward(conv1_->forward(snake1_->forward(x))));
    const int64_t pad = (x.size(-1) - y.size(-1)) / 2;
    auto x_trim = pad > 0 ? x.slice(/*dim=*/-1, pad, x.size(-1) - pad) : x;
    return x_trim + y;
  }

  // Checkpoint: block.0/1/2/3 = snake, wnconv, snake, wnconv
  void load_state_dict(const StateDict& state_dict) {
    snake1_->load_state_dict(state_dict.get_dict_with_prefix("block.0."));
    conv1_->load_state_dict(state_dict.get_dict_with_prefix("block.1."));
    snake2_->load_state_dict(state_dict.get_dict_with_prefix("block.2."));
    conv2_->load_state_dict(state_dict.get_dict_with_prefix("block.3."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    snake1_->verify_loaded_weights(prefix + "block.0.");
    conv1_->verify_loaded_weights(prefix + "block.1.");
    snake2_->verify_loaded_weights(prefix + "block.2.");
    conv2_->verify_loaded_weights(prefix + "block.3.");
  }

 private:
  Snake1d snake1_{nullptr};
  WeightNormConv1d conv1_{nullptr};
  Snake1d snake2_{nullptr};
  WeightNormConv1d conv2_{nullptr};
};
TORCH_MODULE(ResidualUnit);

class DecoderBlockImpl : public torch::nn::Module {
 public:
  DecoderBlockImpl(int64_t input_dim, int64_t output_dim, int64_t stride) {
    const int64_t pad = static_cast<int64_t>(std::ceil(stride / 2.0));
    const int64_t out_pad = stride % 2;
    snake_ = register_module("snake", Snake1d(input_dim));
    upsample_ =
        register_module("upsample",
                        WeightNormConvTranspose1d(input_dim,
                                                  output_dim,
                                                  /*kernel_size=*/2 * stride,
                                                  stride,
                                                  pad,
                                                  out_pad));
    res1_ = register_module("res1", ResidualUnit(output_dim, /*dilation=*/1));
    res3_ = register_module("res3", ResidualUnit(output_dim, /*dilation=*/3));
    res9_ = register_module("res9", ResidualUnit(output_dim, /*dilation=*/9));
  }

  torch::Tensor forward(const torch::Tensor& x) const {
    auto h = upsample_->forward(snake_->forward(x));
    h = res1_->forward(h);
    h = res3_->forward(h);
    return res9_->forward(h);
  }

  // Checkpoint: block.0=snake, block.1=transpose, block.2/3/4=residual units
  void load_state_dict(const StateDict& state_dict) {
    snake_->load_state_dict(state_dict.get_dict_with_prefix("block.0."));
    upsample_->load_state_dict(state_dict.get_dict_with_prefix("block.1."));
    res1_->load_state_dict(state_dict.get_dict_with_prefix("block.2."));
    res3_->load_state_dict(state_dict.get_dict_with_prefix("block.3."));
    res9_->load_state_dict(state_dict.get_dict_with_prefix("block.4."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    snake_->verify_loaded_weights(prefix + "block.0.");
    upsample_->verify_loaded_weights(prefix + "block.1.");
    res1_->verify_loaded_weights(prefix + "block.2.");
    res3_->verify_loaded_weights(prefix + "block.3.");
    res9_->verify_loaded_weights(prefix + "block.4.");
  }

 private:
  Snake1d snake_{nullptr};
  WeightNormConvTranspose1d upsample_{nullptr};
  ResidualUnit res1_{nullptr};
  ResidualUnit res3_{nullptr};
  ResidualUnit res9_{nullptr};
};
TORCH_MODULE(DecoderBlock);

class DecoderImpl : public torch::nn::Module {
 public:
  DecoderImpl(int64_t latent_channels,
              int64_t channels,
              const std::vector<int64_t>& rates,
              int64_t out_channels) {
    first_conv_ = register_module("first_conv",
                                  WeightNormConv1d(latent_channels,
                                                   channels,
                                                   /*kernel_size=*/7,
                                                   /*stride=*/1,
                                                   /*padding=*/3,
                                                   /*dilation=*/1,
                                                   /*bias=*/true));
    int64_t out_dim = channels;
    for (size_t i = 0; i < rates.size(); ++i) {
      const int64_t in_dim = channels / (1LL << static_cast<int64_t>(i));
      out_dim = channels / (1LL << static_cast<int64_t>(i + 1));
      layers_.push_back(
          register_module("layer_" + std::to_string(i),
                          DecoderBlock(in_dim, out_dim, rates[i])));
    }
    out_snake1_ = register_module("out_snake1", Snake1d(out_dim));
    out_conv1_ = register_module("out_conv1",
                                 WeightNormConv1d(out_dim,
                                                  out_dim,
                                                  /*kernel_size=*/7,
                                                  /*stride=*/1,
                                                  /*padding=*/3,
                                                  /*dilation=*/1,
                                                  /*bias=*/true));
    out_snake2_ = register_module("out_snake2", Snake1d(out_dim));
    out_conv2_ = register_module("out_conv2",
                                 WeightNormConv1d(out_dim,
                                                  out_channels,
                                                  /*kernel_size=*/7,
                                                  /*stride=*/1,
                                                  /*padding=*/3,
                                                  /*dilation=*/1,
                                                  /*bias=*/true));
  }

  torch::Tensor forward(const torch::Tensor& z) const {
    auto x = first_conv_->forward(z);
    for (const auto& layer : layers_) {
      x = layer->forward(x);
    }
    x = out_conv1_->forward(out_snake1_->forward(x));
    x = out_conv2_->forward(out_snake2_->forward(x));
    return torch::tanh(x);
  }

  void load_state_dict(const StateDict& state_dict) {
    first_conv_->load_state_dict(
        state_dict.get_dict_with_prefix("first_conv."));
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    out_snake1_->load_state_dict(state_dict.get_dict_with_prefix("out.0."));
    out_conv1_->load_state_dict(state_dict.get_dict_with_prefix("out.1."));
    out_snake2_->load_state_dict(state_dict.get_dict_with_prefix("out.2."));
    out_conv2_->load_state_dict(state_dict.get_dict_with_prefix("out.3."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    first_conv_->verify_loaded_weights(prefix + "first_conv.");
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    out_snake1_->verify_loaded_weights(prefix + "out.0.");
    out_conv1_->verify_loaded_weights(prefix + "out.1.");
    out_snake2_->verify_loaded_weights(prefix + "out.2.");
    out_conv2_->verify_loaded_weights(prefix + "out.3.");
  }

 private:
  WeightNormConv1d first_conv_{nullptr};
  std::vector<DecoderBlock> layers_;
  Snake1d out_snake1_{nullptr};
  WeightNormConv1d out_conv1_{nullptr};
  Snake1d out_snake2_{nullptr};
  WeightNormConv1d out_conv2_{nullptr};
};
TORCH_MODULE(Decoder);

}  // namespace joyo_xdac

class AutoencoderKLXDACImpl : public torch::nn::Module {
 public:
  explicit AutoencoderKLXDACImpl(const ModelContext& context)
      // Audio VAE weights are fp32; ignore DiT bf16 root options_.
      : options_(context.get_tensor_options().dtype(torch::kFloat)) {
    const ModelArgs& args = context.get_model_args();
    latent_channels_ =
        args.latent_channels() > 0 ? args.latent_channels() : 128;
    out_channels_ = args.out_channels() > 0 ? args.out_channels() : 2;
    sampling_rate_ = args.sampling_rate() > 0 ? args.sampling_rate() : 44100;
    decoder_dim_ = args.decoder_dim() > 0 ? args.decoder_dim() : 2048;
    encoder_rates_ = args.encoder_rates();
    decoder_rates_ = args.decoder_rates();
    if (encoder_rates_.empty()) {
      encoder_rates_ = {2, 4, 4, 8, 8};
    }
    if (decoder_rates_.empty()) {
      decoder_rates_ = {8, 8, 4, 4, 2};
    }
    latents_mean_ = args.latents_mean();
    latents_std_ = args.latents_std();

    post_quant_conv_ = register_module(
        "post_quant_conv",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(latent_channels_,
                                                   latent_channels_,
                                                   /*kernel_size=*/1)));
    decoder_ = register_module(
        "decoder",
        joyo_xdac::Decoder(
            latent_channels_, decoder_dim_, decoder_rates_, out_channels_));

    LOG(INFO) << "AutoencoderKLXDAC: latent_channels=" << latent_channels_
              << " sampling_rate=" << sampling_rate_
              << " hop_length=" << hop_length()
              << " decoder_dim=" << decoder_dim_;
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    CHECK(loader != nullptr);
    for (auto& sd : loader->get_state_dicts()) {
      load_state_dict(*sd);
    }
    verify_loaded_weights("");
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict,
                        "post_quant_conv.weight",
                        post_quant_conv_->weight,
                        is_post_quant_conv_weight_loaded_);
    weight::load_weight(state_dict,
                        "post_quant_conv.bias",
                        post_quant_conv_->bias,
                        is_post_quant_conv_bias_loaded_);
    decoder_->load_state_dict(state_dict.get_dict_with_prefix("decoder."));
    // Prefer checkpoint buffers when present.
    auto mean = state_dict.get_tensor("latents_mean_buf");
    auto inv_std = state_dict.get_tensor("latents_inv_std_buf");
    if (mean.defined()) {
      latents_mean_buf_ = mean.to(options_.device()).to(torch::kFloat);
    }
    if (inv_std.defined()) {
      latents_inv_std_buf_ = inv_std.to(options_.device()).to(torch::kFloat);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_post_quant_conv_weight_loaded_)
        << "weight is not loaded for " << prefix << "post_quant_conv.weight";
    CHECK(is_post_quant_conv_bias_loaded_)
        << "bias is not loaded for " << prefix << "post_quant_conv.bias";
    decoder_->verify_loaded_weights(prefix + "decoder.");
  }

  torch::Tensor denormalize_latents(const torch::Tensor& latents) const {
    if (latents_mean_buf_.defined() && latents_inv_std_buf_.defined()) {
      return latents.to(torch::kFloat) /
                 latents_inv_std_buf_.to(latents.device()) +
             latents_mean_buf_.to(latents.device());
    }
    if (latents_mean_.empty() || latents_std_.empty()) {
      return latents;
    }
    CHECK_EQ(static_cast<int64_t>(latents_mean_.size()), latents.size(1));
    auto opts =
        torch::TensorOptions().dtype(torch::kFloat).device(latents.device());
    auto mean = torch::tensor(latents_mean_, opts).view({1, -1, 1});
    auto std = torch::tensor(latents_std_, opts).view({1, -1, 1});
    return latents.to(torch::kFloat) * std + mean;
  }

  // latent: [B, C, T] (denormalized) -> waveform [B, channels, samples]
  torch::Tensor decode(const torch::Tensor& latent) {
    // Match Conv1d weight/bias dtype (fp32); do not cast to DiT bf16.
    auto z =
        post_quant_conv_->forward(latent.to(options_.device(), torch::kFloat));
    return decoder_->forward(z);
  }

  int64_t latent_channels() const { return latent_channels_; }
  int64_t sampling_rate() const { return sampling_rate_; }
  int64_t hop_length() const {
    int64_t hop = 1;
    for (int64_t rate : encoder_rates_) {
      hop *= rate;
    }
    return hop > 0 ? hop : 2048;
  }

 private:
  torch::TensorOptions options_;
  int64_t latent_channels_ = 128;
  int64_t out_channels_ = 2;
  int64_t sampling_rate_ = 44100;
  int64_t decoder_dim_ = 2048;
  std::vector<int64_t> encoder_rates_;
  std::vector<int64_t> decoder_rates_;
  std::vector<double> latents_mean_;
  std::vector<double> latents_std_;
  torch::Tensor latents_mean_buf_;
  torch::Tensor latents_inv_std_buf_;
  torch::nn::Conv1d post_quant_conv_{nullptr};
  joyo_xdac::Decoder decoder_{nullptr};
  bool is_post_quant_conv_weight_loaded_ = false;
  bool is_post_quant_conv_bias_loaded_ = false;
};
TORCH_MODULE(AutoencoderKLXDAC);

REGISTER_MODEL_ARGS(AutoencoderKLXDAC, [&] {
  LOAD_ARG_OR(model_type, "_class_name", "AutoencoderKLXDAC");
  LOAD_ARG_OR(dtype, "dtype", "float32");
  LOAD_ARG_OR(out_channels, "out_channels", 2);
  LOAD_ARG_OR(latent_channels, "latent_channels", 128);
  LOAD_ARG_OR(sampling_rate, "sampling_rate", 44100);
  LOAD_ARG_OR(decoder_dim, "decoder_dim", 2048);
  LOAD_ARG_OR(
      encoder_rates, "encoder_rates", (std::vector<int64_t>{2, 4, 4, 8, 8}));
  LOAD_ARG_OR(
      decoder_rates, "decoder_rates", (std::vector<int64_t>{8, 8, 4, 4, 2}));
  LOAD_ARG_OR(latents_mean, "latents_mean", (std::vector<double>{}));
  LOAD_ARG_OR(latents_std, "latents_std", (std::vector<double>{}));
});

}  // namespace xllm
