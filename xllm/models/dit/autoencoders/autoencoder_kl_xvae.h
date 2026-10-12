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

// JoyOV2 video VAE (AutoencoderKLXVAE). Decoder-only port of SGLang / diffusers
// XVAE. Spatial: patch_size(2) × 2^4 = 32x; temporal = 4x; z_dim = 128.
// Full-sequence decode only (no feat_cache / chunked causal path).

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
#include "models/dit/utils/dit_parallel_mixin.h"
#include "models/model_registry.h"

namespace xllm {
namespace joyo_xvae {

inline torch::Tensor swish(const torch::Tensor& x) {
  return x * torch::sigmoid(x);
}

// Causal 3D conv: manual pad (T left = 2*pad_t, H/W symmetric), Conv3d pad=0.
class CausalConv3dImpl : public torch::nn::Module,
                         public dit::VaeParallelMixin {
 public:
  CausalConv3dImpl(const ModelContext& context,
                   int64_t in_channels,
                   int64_t out_channels,
                   std::vector<int64_t> kernel_size,
                   std::vector<int64_t> stride,
                   std::vector<int64_t> padding)
      : dit::VaeParallelMixin(context) {
    CHECK_EQ(kernel_size.size(), 3u);
    CHECK_EQ(stride.size(), 3u);
    CHECK_EQ(padding.size(), 3u);
    padding_thw_ = padding;
    // F::pad layout: {W_l, W_r, H_l, H_r, T_front, T_back}
    causal_padding_ = {
        padding[2], padding[2], padding[1], padding[1], 2 * padding[0], 0};
    // When VAE-parallel, spatial pad is applied by Conv3d (Wan-style) so halo
    // exchange can supply neighbor columns; otherwise keep JoyO F::pad path.
    const bool spatial_in_conv =
        vae_parallel_enabled() && padding[1] == 1 && padding[2] == 1;
    conv_ = register_module(
        "conv",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(spatial_in_conv
                             ? std::vector<int64_t>{0, padding[1], padding[2]}
                             : std::vector<int64_t>{0, 0, 0})
                .bias(true)));
    spatial_in_conv_ = spatial_in_conv;
  }

  torch::Tensor forward(const torch::Tensor& x) {
    torch::Tensor input = x;
    const bool use_halo =
        vae_parallel_enabled() && padding_thw_[1] == 1 && padding_thw_[2] == 1;
    if (use_halo) {
      input = vae_parallel_exchange(input, /*pad=*/true);
    }
    if (spatial_in_conv_) {
      input = torch::nn::functional::pad(
          input,
          torch::nn::functional::PadFuncOptions(
              {0, 0, 0, 0, causal_padding_[4], causal_padding_[5]}));
    } else {
      input = torch::nn::functional::pad(
          input, torch::nn::functional::PadFuncOptions(causal_padding_));
    }
    auto out = conv_->forward(input);
    if (use_halo) {
      out = out.slice(/*dim=*/-1, 1, out.size(-1) - 1);
    }
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", conv_->weight, is_weight_loaded_);
    weight::load_weight(state_dict, "bias", conv_->bias, is_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_weight_loaded_) << "weight not loaded for " << prefix << "weight";
    CHECK(is_bias_loaded_) << "bias not loaded for " << prefix << "bias";
  }

 private:
  torch::nn::Conv3d conv_{nullptr};
  std::vector<int64_t> causal_padding_;
  std::vector<int64_t> padding_thw_;
  bool spatial_in_conv_ = false;
  bool is_weight_loaded_ = false;
  bool is_bias_loaded_ = false;
};
TORCH_MODULE(CausalConv3d);

class XVAERMSNormImpl : public torch::nn::Module {
 public:
  explicit XVAERMSNormImpl(int64_t dim)
      : scale_(std::sqrt(static_cast<double>(dim))) {
    // gamma shape [C, 1, 1, 1] for channel-first video tensors.
    gamma_ = register_parameter("gamma", torch::ones({dim, 1, 1, 1}));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    const bool needs_fp32 =
        x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16;
    auto norm_input = needs_fp32 ? x.to(torch::kFloat32) : x;
    auto normed =
        torch::nn::functional::normalize(
            norm_input,
            torch::nn::functional::NormalizeFuncOptions().dim(1).eps(1e-12))
            .to(x.dtype());
    return normed * scale_ * gamma_;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "gamma", gamma_, is_gamma_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_gamma_loaded_) << "gamma not loaded for " << prefix << "gamma";
  }

 private:
  double scale_ = 1.0;
  torch::Tensor gamma_;
  bool is_gamma_loaded_ = false;
};
TORCH_MODULE(XVAERMSNorm);

class FP32UpsampleImpl : public torch::nn::Module {
 public:
  torch::Tensor forward(const torch::Tensor& x) {
    auto result = torch::nn::functional::interpolate(
        x.to(torch::kFloat32),
        torch::nn::functional::InterpolateFuncOptions()
            .scale_factor(std::vector<double>{1.0, 2.0, 2.0})
            .mode(torch::kNearestExact));
    return result.to(x.dtype());
  }
};
TORCH_MODULE(FP32Upsample);

class ResidualBlockImpl : public torch::nn::Module {
 public:
  ResidualBlockImpl(const ModelContext& context,
                    int64_t in_channels,
                    int64_t out_channels)
      : in_channels_(in_channels), out_channels_(out_channels) {
    norm1_ = register_module("norm1", XVAERMSNorm(in_channels));
    conv1_ = register_module(
        "conv1",
        CausalConv3d(context,
                     in_channels,
                     out_channels,
                     /*kernel_size=*/std::vector<int64_t>{3, 3, 3},
                     /*stride=*/std::vector<int64_t>{1, 1, 1},
                     /*padding=*/std::vector<int64_t>{1, 1, 1}));
    norm2_ = register_module("norm2", XVAERMSNorm(out_channels));
    conv2_ = register_module(
        "conv2",
        CausalConv3d(context,
                     out_channels,
                     out_channels,
                     /*kernel_size=*/std::vector<int64_t>{3, 3, 3},
                     /*stride=*/std::vector<int64_t>{1, 1, 1},
                     /*padding=*/std::vector<int64_t>{1, 1, 1}));
    if (in_channels != out_channels) {
      nin_shortcut_ = register_module(
          "nin_shortcut",
          CausalConv3d(context,
                       in_channels,
                       out_channels,
                       /*kernel_size=*/std::vector<int64_t>{1, 1, 1},
                       /*stride=*/std::vector<int64_t>{1, 1, 1},
                       /*padding=*/std::vector<int64_t>{0, 0, 0}));
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    torch::Tensor shortcut = x;
    auto h = conv1_->forward(swish(norm1_->forward(x)));
    h = conv2_->forward(swish(norm2_->forward(h)));
    if (nin_shortcut_) {
      shortcut = nin_shortcut_->forward(shortcut);
    }
    return h + shortcut;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    conv1_->load_state_dict(state_dict.get_dict_with_prefix("conv1."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    conv2_->load_state_dict(state_dict.get_dict_with_prefix("conv2."));
    if (nin_shortcut_) {
      nin_shortcut_->load_state_dict(
          state_dict.get_dict_with_prefix("nin_shortcut."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    norm1_->verify_loaded_weights(prefix + "norm1.");
    conv1_->verify_loaded_weights(prefix + "conv1.");
    norm2_->verify_loaded_weights(prefix + "norm2.");
    conv2_->verify_loaded_weights(prefix + "conv2.");
    if (nin_shortcut_) {
      nin_shortcut_->verify_loaded_weights(prefix + "nin_shortcut.");
    }
  }

 private:
  int64_t in_channels_ = 0;
  int64_t out_channels_ = 0;
  XVAERMSNorm norm1_{nullptr};
  CausalConv3d conv1_{nullptr};
  XVAERMSNorm norm2_{nullptr};
  CausalConv3d conv2_{nullptr};
  CausalConv3d nin_shortcut_{nullptr};
};
TORCH_MODULE(ResidualBlock);

class AttnBlockImpl : public torch::nn::Module, public dit::VaeParallelMixin {
 public:
  AttnBlockImpl(const ModelContext& context, int64_t in_channels)
      : dit::VaeParallelMixin(context), in_channels_(in_channels) {
    norm_ = register_module("norm", XVAERMSNorm(in_channels));
    q_ = register_module(
        "q",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, in_channels, 1).bias(true)));
    k_ = register_module(
        "k",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, in_channels, 1).bias(true)));
    v_ = register_module(
        "v",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, in_channels, 1).bias(true)));
    proj_out_ = register_module(
        "proj_out",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels, in_channels, 1).bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    const int64_t b = x.size(0);
    const int64_t c = x.size(1);
    const int64_t t = x.size(2);
    const int64_t h = x.size(3);
    const int64_t w = x.size(4);
    CHECK_EQ(c, in_channels_);

    torch::Tensor residual = x;
    auto h_norm = norm_->forward(x);

    auto to_bt_hw_c = [&](const torch::Tensor& y, int64_t width) {
      // b c t h w -> (b t) 1 (h w) c
      return y.permute({0, 2, 3, 4, 1})
          .contiguous()
          .view({b * t, h * width, c})
          .unsqueeze(1);
    };

    auto q = to_bt_hw_c(q_->forward(h_norm), w);
    auto k_sp = k_->forward(h_norm);
    auto v_sp = v_->forward(h_norm);
    // Match Wan/QwenImage VAE attention: gather K/V over global W so each
    // local Q can attend across rank boundaries; output stays Q-local width.
    if (vae_parallel_enabled()) {
      k_sp = vae_parallel_merge(k_sp);
      v_sp = vae_parallel_merge(v_sp);
    }
    const int64_t w_k = k_sp.size(-1);
    auto k = to_bt_hw_c(k_sp, w_k);
    auto v = to_bt_hw_c(v_sp, w_k);

    auto attn = torch::scaled_dot_product_attention(
        q, k, v, std::nullopt, /*dropout_p=*/0.0, /*is_causal=*/false);
    // (b t) 1 (h w_local) c -> b c t h w_local
    attn = attn.squeeze(1)
               .view({b, t, h, w, c})
               .permute({0, 4, 1, 2, 3})
               .contiguous();
    attn = proj_out_->forward(attn);
    return residual + attn;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    weight::load_weight(
        state_dict, "q.weight", q_->weight, is_q_weight_loaded_);
    weight::load_weight(state_dict, "q.bias", q_->bias, is_q_bias_loaded_);
    weight::load_weight(
        state_dict, "k.weight", k_->weight, is_k_weight_loaded_);
    weight::load_weight(state_dict, "k.bias", k_->bias, is_k_bias_loaded_);
    weight::load_weight(
        state_dict, "v.weight", v_->weight, is_v_weight_loaded_);
    weight::load_weight(state_dict, "v.bias", v_->bias, is_v_bias_loaded_);
    weight::load_weight(state_dict,
                        "proj_out.weight",
                        proj_out_->weight,
                        is_proj_weight_loaded_);
    weight::load_weight(
        state_dict, "proj_out.bias", proj_out_->bias, is_proj_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    norm_->verify_loaded_weights(prefix + "norm.");
    CHECK(is_q_weight_loaded_) << "q.weight not loaded for " << prefix;
    CHECK(is_q_bias_loaded_) << "q.bias not loaded for " << prefix;
    CHECK(is_k_weight_loaded_) << "k.weight not loaded for " << prefix;
    CHECK(is_k_bias_loaded_) << "k.bias not loaded for " << prefix;
    CHECK(is_v_weight_loaded_) << "v.weight not loaded for " << prefix;
    CHECK(is_v_bias_loaded_) << "v.bias not loaded for " << prefix;
    CHECK(is_proj_weight_loaded_)
        << "proj_out.weight not loaded for " << prefix;
    CHECK(is_proj_bias_loaded_) << "proj_out.bias not loaded for " << prefix;
  }

 private:
  int64_t in_channels_ = 0;
  XVAERMSNorm norm_{nullptr};
  torch::nn::Conv3d q_{nullptr};
  torch::nn::Conv3d k_{nullptr};
  torch::nn::Conv3d v_{nullptr};
  torch::nn::Conv3d proj_out_{nullptr};
  bool is_q_weight_loaded_ = false;
  bool is_q_bias_loaded_ = false;
  bool is_k_weight_loaded_ = false;
  bool is_k_bias_loaded_ = false;
  bool is_v_weight_loaded_ = false;
  bool is_v_bias_loaded_ = false;
  bool is_proj_weight_loaded_ = false;
  bool is_proj_bias_loaded_ = false;
};
TORCH_MODULE(AttnBlock);

class UpsampleBlockImpl : public torch::nn::Module,
                          public dit::VaeParallelMixin {
 public:
  UpsampleBlockImpl(const ModelContext& context,
                    int64_t in_channels,
                    int64_t out_channels,
                    bool temporal_upsample)
      : dit::VaeParallelMixin(context),
        in_channels_(in_channels),
        out_channels_(out_channels),
        temporal_upsample_(temporal_upsample) {
    if (temporal_upsample_) {
      temporal_ = register_module(
          "temporal",
          CausalConv3d(context,
                       in_channels,
                       in_channels * 2,
                       /*kernel_size=*/std::vector<int64_t>{3, 1, 1},
                       /*stride=*/std::vector<int64_t>{1, 1, 1},
                       /*padding=*/std::vector<int64_t>{1, 0, 0}));
    }
    spatial_ = register_module(
        "spatial",
        torch::nn::Sequential(
            FP32Upsample(),
            torch::nn::Conv3d(
                torch::nn::Conv3dOptions(in_channels, out_channels, {1, 3, 3})
                    .padding({0, 1, 1})
                    .bias(true))));
    // (8 if temporal else 4) * out / in
    const int64_t factor = temporal_upsample_ ? 8 : 4;
    CHECK_EQ((factor * out_channels) % in_channels, 0)
        << "invalid upsample channel ratio in=" << in_channels
        << " out=" << out_channels;
    repeats_ = (factor * out_channels) / in_channels;
  }

  torch::Tensor shortcut(const torch::Tensor& x, bool first_chunk) const {
    const int64_t r1 = temporal_upsample_ ? 2 : 1;
    const int64_t skip = (temporal_upsample_ && first_chunk) ? 1 : 0;
    const int64_t b = x.size(0);
    const int64_t f = x.size(2);
    const int64_t h = x.size(3);
    const int64_t w = x.size(4);

    auto y = x.repeat_interleave(/*repeats=*/repeats_, /*dim=*/1);
    // JoyO XVAE shortcut layout is factor-major (same family as unpatchify):
    // b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3).
    // Do NOT use Wan DupUp3D's channel-major (c, r1, r2, r3) view — that
    // scrambles decode into scanline/grid garbage (seen on SP case2 rebuild).
    const int64_t c = y.size(1) / (r1 * 2 * 2);
    y = y.view({b, r1, 2, 2, c, f, h, w});
    y = y.permute({0, 4, 5, 1, 6, 2, 7, 3}).contiguous();
    y = y.view({b, c, f * r1, h * 2, w * 2});
    if (skip > 0) {
      CHECK_GT(y.size(2), skip)
          << "XVAE UpsampleBlock shortcut: temporal dim " << y.size(2)
          << " cannot drop first " << skip << " frame(s)";
      y = y.slice(/*dim=*/2, skip, y.size(2));
    }
    return y;
  }

  torch::Tensor forward(const torch::Tensor& x) {
    // Full-sequence path: feat_cache is always absent → shortcut
    // first_chunk=true.
    // Shortcut upsamples local W → 2*W_local. Spatial path does Wan-style
    // halo exchange before 2× upsample + Conv3d(k=3,pad=1), then trims
    // 2 cols/side so W matches shortcut without exchanging the shortcut.
    auto sc = shortcut(x, /*first_chunk=*/true);

    torch::Tensor hidden = x;
    if (temporal_) {
      const int64_t b = hidden.size(0);
      const int64_t c = hidden.size(1);
      const int64_t t = hidden.size(2);
      const int64_t height = hidden.size(3);
      const int64_t width = hidden.size(4);
      auto x0 = hidden.slice(/*dim=*/2, 0, 1);
      if (t > 1) {
        auto x_t = temporal_->forward(hidden).slice(/*dim=*/2, 1, t);
        // b (r c) t h w -> b c (t r) h w
        const int64_t t2 = x_t.size(2);
        x_t = x_t.view({b, 2, c, t2, height, width});
        x_t = x_t.permute({0, 2, 3, 1, 4, 5}).contiguous();
        x_t = x_t.view({b, c, t2 * 2, height, width});
        hidden = torch::cat({x0, x_t}, /*dim=*/2);
      } else {
        auto dummy = temporal_->forward(hidden);
        const int64_t t_dummy = dummy.size(2);
        dummy = dummy.view({b, 2, c, t_dummy, height, width});
        dummy = dummy.permute({0, 2, 3, 1, 4, 5}).contiguous();
        dummy = dummy.view({b, c, t_dummy * 2, height, width});
        dummy = dummy.slice(/*dim=*/2, 0, 1);
        hidden = x0 + 0.0 * dummy;
      }
    }

    hidden = vae_parallel_exchange(hidden, /*pad=*/false);
    hidden = spatial_->forward(hidden);
    hidden = vae_parallel_trim_halo(hidden, /*per_side=*/2);
    return hidden + sc;
  }

  void load_state_dict(const StateDict& state_dict) {
    if (temporal_) {
      temporal_->load_state_dict(state_dict.get_dict_with_prefix("temporal."));
    }
    auto params = spatial_->named_parameters();
    for (auto& param : params) {
      if (param.key() == "1.weight") {
        weight::load_weight(state_dict,
                            "spatial.1.weight",
                            param.value(),
                            is_spatial_weight_loaded_);
      } else if (param.key() == "1.bias") {
        weight::load_weight(state_dict,
                            "spatial.1.bias",
                            param.value(),
                            is_spatial_bias_loaded_);
      }
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    if (temporal_) {
      temporal_->verify_loaded_weights(prefix + "temporal.");
    }
    CHECK(is_spatial_weight_loaded_)
        << "spatial.1.weight not loaded for " << prefix;
    CHECK(is_spatial_bias_loaded_)
        << "spatial.1.bias not loaded for " << prefix;
  }

 private:
  int64_t in_channels_ = 0;
  int64_t out_channels_ = 0;
  bool temporal_upsample_ = false;
  int64_t repeats_ = 0;
  CausalConv3d temporal_{nullptr};
  torch::nn::Sequential spatial_{nullptr};
  bool is_spatial_weight_loaded_ = false;
  bool is_spatial_bias_loaded_ = false;
};
TORCH_MODULE(UpsampleBlock);

class XVAEDecoderImpl : public torch::nn::Module {
 public:
  XVAEDecoderImpl(const ModelContext& context,
                  int64_t z_channels,
                  int64_t out_channels,
                  int64_t num_res_blocks,
                  const std::vector<int64_t>& block_in_channels,
                  const std::vector<bool>& temporal_upsample,
                  bool channel_doubling)
      : num_res_blocks_(num_res_blocks),
        block_in_channels_(block_in_channels),
        temporal_upsample_(temporal_upsample),
        channel_doubling_(channel_doubling) {
    CHECK(!block_in_channels_.empty());
    CHECK_EQ(temporal_upsample_.size(), block_in_channels_.size());

    const int64_t block_in0 = block_in_channels_[0];
    conv_in_ = register_module(
        "conv_in",
        CausalConv3d(context,
                     z_channels,
                     block_in0,
                     /*kernel_size=*/std::vector<int64_t>{3, 3, 3},
                     /*stride=*/std::vector<int64_t>{1, 1, 1},
                     /*padding=*/std::vector<int64_t>{1, 1, 1}));

    mid_blocks_ = register_module("mid_blocks", torch::nn::ModuleList());
    mid_blocks_->push_back(ResidualBlock(context, block_in0, block_in0));
    mid_blocks_->push_back(AttnBlock(context, block_in0));
    mid_blocks_->push_back(ResidualBlock(context, block_in0, block_in0));
    mid_is_attn_ = {false, true, false};

    up_blocks_ = register_module("up_blocks", torch::nn::ModuleList());
    for (size_t i_level = 0; i_level < block_in_channels_.size(); ++i_level) {
      const int64_t block_in = block_in_channels_[i_level];
      for (int64_t i = 0; i < num_res_blocks_ + 1; ++i) {
        up_blocks_->push_back(ResidualBlock(context, block_in, block_in));
        up_is_upsample_.push_back(false);
      }
      if (i_level + 1 < block_in_channels_.size()) {
        const int64_t out_ch = channel_doubling_
                                   ? (block_in / 2)
                                   : block_in_channels_[i_level + 1];
        up_blocks_->push_back(UpsampleBlock(
            context, block_in, out_ch, temporal_upsample_[i_level]));
        up_is_upsample_.push_back(true);
      }
    }
    CHECK_EQ(up_is_upsample_.size(), up_blocks_->size());

    const int64_t block_in_last = block_in_channels_.back();
    norm_out_ = register_module("norm_out", XVAERMSNorm(block_in_last));
    conv_out_ = register_module(
        "conv_out",
        CausalConv3d(context,
                     block_in_last,
                     out_channels,
                     /*kernel_size=*/std::vector<int64_t>{3, 3, 3},
                     /*stride=*/std::vector<int64_t>{1, 1, 1},
                     /*padding=*/std::vector<int64_t>{1, 1, 1}));
  }

  torch::Tensor forward(const torch::Tensor& z) {
    auto x = conv_in_->forward(z);

    for (size_t i = 0; i < mid_blocks_->size(); ++i) {
      if (mid_is_attn_[i]) {
        x = mid_blocks_[i]->as<AttnBlock>()->forward(x);
      } else {
        x = mid_blocks_[i]->as<ResidualBlock>()->forward(x);
      }
    }

    for (size_t i = 0; i < up_blocks_->size(); ++i) {
      if (up_is_upsample_[i]) {
        x = up_blocks_[i]->as<UpsampleBlock>()->forward(x);
      } else {
        x = up_blocks_[i]->as<ResidualBlock>()->forward(x);
      }
    }

    x = swish(norm_out_->forward(x));
    return conv_out_->forward(x);
  }

  void load_state_dict(const StateDict& state_dict) {
    conv_in_->load_state_dict(state_dict.get_dict_with_prefix("conv_in."));
    for (size_t i = 0; i < mid_blocks_->size(); ++i) {
      const std::string prefix = "mid_blocks." + std::to_string(i) + ".";
      if (mid_is_attn_[i]) {
        mid_blocks_[i]->as<AttnBlock>()->load_state_dict(
            state_dict.get_dict_with_prefix(prefix));
      } else {
        mid_blocks_[i]->as<ResidualBlock>()->load_state_dict(
            state_dict.get_dict_with_prefix(prefix));
      }
    }
    for (size_t i = 0; i < up_blocks_->size(); ++i) {
      const std::string prefix = "up_blocks." + std::to_string(i) + ".";
      if (up_is_upsample_[i]) {
        up_blocks_[i]->as<UpsampleBlock>()->load_state_dict(
            state_dict.get_dict_with_prefix(prefix));
      } else {
        up_blocks_[i]->as<ResidualBlock>()->load_state_dict(
            state_dict.get_dict_with_prefix(prefix));
      }
    }
    norm_out_->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    conv_out_->load_state_dict(state_dict.get_dict_with_prefix("conv_out."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    conv_in_->verify_loaded_weights(prefix + "conv_in.");
    for (size_t i = 0; i < mid_blocks_->size(); ++i) {
      const std::string p = prefix + "mid_blocks." + std::to_string(i) + ".";
      if (mid_is_attn_[i]) {
        mid_blocks_[i]->as<AttnBlock>()->verify_loaded_weights(p);
      } else {
        mid_blocks_[i]->as<ResidualBlock>()->verify_loaded_weights(p);
      }
    }
    for (size_t i = 0; i < up_blocks_->size(); ++i) {
      const std::string p = prefix + "up_blocks." + std::to_string(i) + ".";
      if (up_is_upsample_[i]) {
        up_blocks_[i]->as<UpsampleBlock>()->verify_loaded_weights(p);
      } else {
        up_blocks_[i]->as<ResidualBlock>()->verify_loaded_weights(p);
      }
    }
    norm_out_->verify_loaded_weights(prefix + "norm_out.");
    conv_out_->verify_loaded_weights(prefix + "conv_out.");
  }

  int64_t num_up_blocks() const {
    return static_cast<int64_t>(up_blocks_->size());
  }

 private:
  int64_t num_res_blocks_ = 2;
  std::vector<int64_t> block_in_channels_;
  std::vector<bool> temporal_upsample_;
  bool channel_doubling_ = false;

  CausalConv3d conv_in_{nullptr};
  torch::nn::ModuleList mid_blocks_{nullptr};
  std::vector<bool> mid_is_attn_;
  torch::nn::ModuleList up_blocks_{nullptr};
  std::vector<bool> up_is_upsample_;
  XVAERMSNorm norm_out_{nullptr};
  CausalConv3d conv_out_{nullptr};
};
TORCH_MODULE(XVAEDecoder);

inline torch::Tensor unpatchify(const torch::Tensor& x, int64_t patch_size) {
  if (patch_size == 1) {
    return x;
  }
  const int64_t b = x.size(0);
  const int64_t c_in = x.size(1);
  const int64_t t = x.size(2);
  const int64_t h = x.size(3);
  const int64_t w = x.size(4);
  const int64_t r1 = patch_size;
  const int64_t r2 = patch_size;
  CHECK_EQ(c_in % (r1 * r2), 0);
  const int64_t c = c_in / (r1 * r2);
  // b (r1 r2 c) t h w -> b c t (h r1) (w r2)
  auto y = x.view({b, r1, r2, c, t, h, w});
  y = y.permute({0, 3, 4, 5, 1, 6, 2}).contiguous();
  return y.view({b, c, t, h * r1, w * r2});
}

}  // namespace joyo_xvae

class AutoencoderKLXVAEImpl : public torch::nn::Module,
                              public dit::VaeParallelMixin {
 public:
  explicit AutoencoderKLXVAEImpl(const ModelContext& context)
      // VAE checkpoint + nn::Conv3d weights are fp32. Pipeline root options_
      // may be bf16 (DiT); keep activations in weight dtype for decode.
      : dit::VaeParallelMixin(context),
        options_(context.get_tensor_options().dtype(torch::kFloat)) {
    const ModelArgs& args = context.get_model_args();
    latent_channels_ =
        args.latent_channels() > 0 ? args.latent_channels() : 128;
    out_channels_ = args.out_channels() > 0 ? args.out_channels() : 3;
    scale_factor_spatial_ = args.scale_factor_spatial() > 0
                                ? args.scale_factor_spatial()
                                : (args.vae_scale_factor_spatial() > 0
                                       ? args.vae_scale_factor_spatial()
                                       : 32);
    scale_factor_temporal_ = args.scale_factor_temporal() > 0
                                 ? args.scale_factor_temporal()
                                 : (args.vae_scale_factor_temporal() > 0
                                        ? args.vae_scale_factor_temporal()
                                        : 4);
    patch_size_ = args.patch_size() > 0
                      ? args.patch_size()
                      : (args.vae_patch_size() > 0 ? args.vae_patch_size() : 2);
    num_res_blocks_ = args.num_res_blocks() > 0 ? args.num_res_blocks() : 2;
    block_in_channels_ = args.block_in_channels();
    // Checkpoint stores null → heal to SGLang/diffusers defaults.
    if (block_in_channels_.empty()) {
      block_in_channels_ = {160, 320, 640, 1280, 1280};
    }
    temporal_upsample_ = args.temporal_downsample();
    if (temporal_upsample_.empty()) {
      temporal_upsample_ = {false, true, true, false, false};
    }
    latents_mean_ = args.latents_mean();
    latents_std_ = args.latents_std();
    CHECK_GT(latent_channels_, 0) << "XVAE latent_channels must be > 0";
    CHECK_EQ(temporal_upsample_.size(), block_in_channels_.size());

    // Decoder uses reversed block channels (encoder order in config).
    std::vector<int64_t> decoder_block_in_channels(block_in_channels_.rbegin(),
                                                   block_in_channels_.rend());
    // Mirror temporal_downsample as temporal_upsample (same boolean sequence).
    std::vector<bool> decoder_temporal_upsample = temporal_upsample_;

    const int64_t decoder_out_channels =
        out_channels_ * patch_size_ * patch_size_;
    decoder_ =
        register_module("decoder",
                        joyo_xvae::XVAEDecoder(context,
                                               latent_channels_,
                                               decoder_out_channels,
                                               num_res_blocks_,
                                               decoder_block_in_channels,
                                               decoder_temporal_upsample,
                                               /*channel_doubling=*/false));

    LOG(INFO) << "AutoencoderKLXVAE: latent_channels=" << latent_channels_
              << " spatial=" << scale_factor_spatial_
              << " temporal=" << scale_factor_temporal_
              << " patch=" << patch_size_
              << " blocks=" << block_in_channels_.size()
              << " up_blocks=" << decoder_->num_up_blocks()
              << " mean_dim=" << latents_mean_.size()
              << " vae_parallel=" << vae_parallel_enabled();
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    CHECK(loader != nullptr);
    for (auto& sd : loader->get_state_dicts()) {
      load_state_dict(*sd);
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    decoder_->load_state_dict(state_dict.get_dict_with_prefix("decoder."));
    auto mean = state_dict.get_tensor("latents_mean_buf");
    auto inv_std = state_dict.get_tensor("latents_inv_std_buf");
    if (mean.defined()) {
      latents_mean_buf_ = mean.to(options_.device()).to(torch::kFloat);
    }
    if (inv_std.defined()) {
      latents_inv_std_buf_ = inv_std.to(options_.device()).to(torch::kFloat);
    }
  }

  // z_normed → z in native VAE space: z / inv_std + mean == z * std + mean
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
    CHECK_EQ(static_cast<int64_t>(latents_std_.size()), latents.size(1));
    auto opts =
        torch::TensorOptions().dtype(torch::kFloat).device(latents.device());
    auto mean = torch::tensor(latents_mean_, opts).view({1, -1, 1, 1, 1});
    auto std = torch::tensor(latents_std_, opts).view({1, -1, 1, 1, 1});
    return latents.to(torch::kFloat) * std + mean;
  }

  // latent: [B, C, T, H, W] -> pixels [B, 3, T', H', W']
  torch::Tensor decode(const torch::Tensor& latent) {
    CHECK_EQ(latent.dim(), 5) << "XVAE decode expects [B,C,T,H,W]";
    CHECK_EQ(latent.size(1), latent_channels_);
    // Match Conv3d weight/bias dtype (fp32). Casting to DiT bf16 mismatches
    // bias and can fall back to CPU aten.
    auto z = latent.to(options_.device(), torch::kFloat);
    z = vae_parallel_split(z);
    auto decoded = decoder_->forward(z);
    decoded = vae_parallel_merge(decoded);
    return joyo_xvae::unpatchify(decoded, patch_size_);
  }

  int64_t latent_channels() const { return latent_channels_; }
  int64_t scale_factor_spatial() const { return scale_factor_spatial_; }
  int64_t scale_factor_temporal() const { return scale_factor_temporal_; }

 private:
  torch::TensorOptions options_;
  int64_t latent_channels_ = 128;
  int64_t out_channels_ = 3;
  int64_t scale_factor_spatial_ = 32;
  int64_t scale_factor_temporal_ = 4;
  int64_t patch_size_ = 2;
  int64_t num_res_blocks_ = 2;
  std::vector<int64_t> block_in_channels_;
  std::vector<bool> temporal_upsample_;
  std::vector<double> latents_mean_;
  std::vector<double> latents_std_;
  torch::Tensor latents_mean_buf_;
  torch::Tensor latents_inv_std_buf_;
  joyo_xvae::XVAEDecoder decoder_{nullptr};
};
TORCH_MODULE(AutoencoderKLXVAE);

REGISTER_MODEL_ARGS(AutoencoderKLXVAE, [&] {
  LOAD_ARG_OR(model_type, "_class_name", "AutoencoderKLXVAE");
  LOAD_ARG_OR(dtype, "dtype", "float32");
  LOAD_ARG_OR(out_channels, "out_channels", 3);
  LOAD_ARG_OR(latent_channels, "latent_channels", 128);
  LOAD_ARG_OR(scale_factor_spatial, "scale_factor_spatial", 32);
  LOAD_ARG_OR(scale_factor_temporal, "scale_factor_temporal", 4);
  LOAD_ARG_OR(vae_scale_factor_spatial, "scale_factor_spatial", 32);
  LOAD_ARG_OR(vae_scale_factor_temporal, "scale_factor_temporal", 4);
  LOAD_ARG_OR(latents_mean, "latents_mean", (std::vector<double>{}));
  LOAD_ARG_OR(latents_std, "latents_std", (std::vector<double>{}));
  LOAD_ARG_OR(block_in_channels,
              "block_in_channels",
              (std::vector<int64_t>{160, 320, 640, 1280, 1280}));
  LOAD_ARG_OR(temporal_downsample,
              "temporal_downsample",
              (std::vector<bool>{false, true, true, false, false}));
  LOAD_ARG_OR(num_res_blocks, "num_res_blocks", 2);
  // Checkpoint stores scalar patch_size (int), not a vector.
  LOAD_ARG_OR(patch_size, "patch_size", 2);
  LOAD_ARG_OR(vae_patch_size, "patch_size", 2);
});

}  // namespace xllm
