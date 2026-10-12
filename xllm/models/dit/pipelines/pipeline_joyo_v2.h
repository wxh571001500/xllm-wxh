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

// JoyOV2Pipeline — Text-to-Video+Audio (T2VA) DiT pipeline for xLLM.
// Reference: tools/SglangXvideo-f3a6983/.../pipelines/joyo_v2_pipeline.py
//
// Text path (default = MiniMax-H3, no in-process encoder):
//   - Requests MUST supply prompt_embed (proto VideoInput.prompt_embed).
//   - CFG seqlens: negative_prompt_embed sizes, else
//     prompt_embed.parameters["joyo_text_seqlens"].
//   - --dit_enable_joyo_text_encoder: load Qwen3-VL with
//     dit_text_encoder_tp_group_ (requires explicit --text_encoder_tp_size).
//     Resident, no rolling. prompt_embed still overrides in-process encode
//     when present.
//
// Memory:
//   - DiT blocks: DitRollingLoad (--enable_rolling_load) or whole-model
//     resident on device (TP/EP>1, no rolling).
//   - DiT embedders: on device for denoise.
//   - VAE / audio VAE: always device-resident.
//   - Text encoder (if enabled): resident, TP-sharded.

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/dit_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/dit_model_context.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/request/dit_input_sources.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/runtime/dit_forward_params.h"
#include "models/dit/autoencoders/autoencoder_kl_xdac.h"
#include "models/dit/autoencoders/autoencoder_kl_xvae.h"
#include "models/dit/transformers/transformer_joyo_v2.h"
#include "models/dit/utils/util.h"
#include "models/model_registry.h"
#include "models/vlm/qwen3_vl.h"
#if defined(USE_NPU)
#include "core/layers/npu/loader/rolling_weight_buffer.h"
#include "models/dit/utils/dit_block_weight_manager.h"
#endif

namespace xllm {

namespace joyo_v2 {

enum class PipelineStage : int8_t {
  kIdle = 0,
  kDenoise = 1,
  kDecode = 2,
};

inline int64_t module_npu_tensor_count(torch::nn::Module& module) {
  int64_t count = 0;
  auto count_one = [&](const torch::Tensor& tensor) {
    if (tensor.defined() && !tensor.device().is_cpu()) {
      count += 1;
    }
  };
  for (const auto& kv : module.named_parameters(/*recurse=*/true)) {
    count_one(kv.value());
  }
  for (const auto& kv : module.named_buffers(/*recurse=*/true)) {
    count_one(kv.value());
  }
  return count;
}

// nn::Module::to(cpu) often leaves NPU CachingAllocator blocks alive because
// Parameter still holds the old Device storage. Copy to host then set_data so
// the NPU TensorImpl is dropped, then empty_cache.
inline void force_module_storage_to_cpu(torch::nn::Module& module) {
  torch::NoGradGuard no_grad;
  auto move_one = [](torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.device().is_cpu()) {
      return;
    }
    torch::Tensor host = tensor.to(torch::kCPU).contiguous();
    tensor.set_data(host);
  };
  for (auto& kv : module.named_parameters(/*recurse=*/true)) {
    move_one(kv.value());
  }
  for (auto& kv : module.named_buffers(/*recurse=*/true)) {
    move_one(kv.value());
  }
}

template <typename ModuleHolder>
void module_to_device(ModuleHolder& module, const torch::Device& device) {
  if (module.is_empty()) {
    return;
  }
  module->to(device);
}

template <typename ModuleHolder>
void module_to_cpu(ModuleHolder& module) {
  if (module.is_empty()) {
    return;
  }
  // First the regular to(); then rewrite Parameter storage so NPU blocks
  // are not kept alive by the caching allocator.
  module->to(torch::Device(torch::kCPU));
  force_module_storage_to_cpu(*module);
}

// SD3-style time-shift: sigma(t) = 1 - (shift * t) / (1 + (shift - 1) * t)
inline torch::Tensor sd3_timeshift(const torch::Tensor& t, double shift) {
  return 1.0 - (shift * t) / (1.0 + (shift - 1.0) * t);
}

inline int64_t compute_audio_latent_length(int64_t sampling_rate,
                                           int64_t hop_length,
                                           int64_t num_frames,
                                           double fps) {
  const double audio_num_samples = static_cast<double>(sampling_rate) *
                                   static_cast<double>(num_frames) / fps;
  return static_cast<int64_t>(
      std::ceil(audio_num_samples / static_cast<double>(hop_length)));
}

inline torch::Tensor build_position_ids(
    int64_t t,
    int64_t h,
    int64_t w,
    const torch::Device& device,
    int64_t pre_len = 0,
    torch::ScalarType dtype = torch::kInt32) {
  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  auto t_ids = torch::arange(pre_len, pre_len + t, opts)
                   .unsqueeze(1)
                   .unsqueeze(2)
                   .expand({t, h, w});
  auto h_ids =
      torch::arange(h, opts).unsqueeze(0).unsqueeze(2).expand({t, h, w});
  auto w_ids =
      torch::arange(w, opts).unsqueeze(0).unsqueeze(1).expand({t, h, w});
  return torch::stack({t_ids.flatten(), h_ids.flatten(), w_ids.flatten()},
                      /*dim=*/0);
}

inline JoyOV2RopeMeta make_rope_meta_from_seqlens(
    const torch::Tensor& seqlens,      // [n] int32
    const torch::Tensor& position_id,  // [3, S]
    const torch::Device& device) {
  JoyOV2RopeMeta meta;
  meta.seqlens = seqlens.to(torch::kInt32).to(device);
  const int64_t n = meta.seqlens.size(0);
  meta.cu_seqlens = torch::zeros(
      {n + 1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  meta.cu_seqlens.slice(0, 1, n + 1) = torch::cumsum(meta.seqlens, /*dim=*/0);
  meta.position_id = position_id.to(device);
  meta.max_seq_len =
      meta.seqlens.numel() > 0 ? meta.seqlens.max().item<int64_t>() : 0;
  return meta;
}

// Build rope_meta for T2VA packed CFG (pos+neg). Matches SGLang build_rope_meta
// for the no-cond-frames case.
inline JoyOV2PackedRopeMeta build_rope_meta(
    const torch::Tensor& text_seqlens,  // [n_samples]
    int64_t latent_t,
    int64_t latent_h,
    int64_t latent_w,
    const torch::Device& device,
    const std::optional<torch::Tensor>& audio_seqlens = std::nullopt,
    std::optional<int64_t> num_real_samples = std::nullopt) {
  const int64_t n_samples = text_seqlens.size(0);
  const int64_t n_real =
      num_real_samples.has_value() ? *num_real_samples : n_samples;
  const int64_t pixel_seqlen = latent_t * latent_h * latent_w;
  const bool has_audio = audio_seqlens.has_value();

  std::vector<torch::Tensor> text_pos_list;
  text_pos_list.reserve(static_cast<size_t>(n_samples));
  for (int64_t i = 0; i < n_samples; ++i) {
    const int64_t t_len = text_seqlens[i].item<int64_t>();
    auto t_pos =
        torch::arange(
            t_len, torch::TensorOptions().dtype(torch::kInt32).device(device))
            .unsqueeze(0)
            .expand({3, t_len});
    text_pos_list.push_back(t_pos);
  }
  auto text_position_ids = torch::cat(text_pos_list, /*dim=*/1);

  // pixel.position_id has NO text-offset (Joytron PreCalRopeMeta / SGLang).
  // Only the pixel slice inside mix.position_id is shifted by text_len.
  std::vector<torch::Tensor> pixel_pos_list;
  pixel_pos_list.reserve(static_cast<size_t>(n_real));
  std::vector<int32_t> pixel_lens(static_cast<size_t>(n_real),
                                  static_cast<int32_t>(pixel_seqlen));
  for (int64_t i = 0; i < n_real; ++i) {
    pixel_pos_list.push_back(build_position_ids(
        latent_t, latent_h, latent_w, device, /*pre_len=*/0));
  }
  auto pixel_position_ids = torch::cat(pixel_pos_list, /*dim=*/1);
  auto pixel_seqlens = torch::tensor(
      pixel_lens, torch::TensorOptions().dtype(torch::kInt32).device(device));

  JoyOV2PackedRopeMeta packed;
  packed.text =
      make_rope_meta_from_seqlens(text_seqlens, text_position_ids, device);
  packed.pixel =
      make_rope_meta_from_seqlens(pixel_seqlens, pixel_position_ids, device);

  if (has_audio) {
    // Standalone audio rope starts at 0 (SGLang), not text+pixel offset.
    std::vector<torch::Tensor> audio_pos_list;
    audio_pos_list.reserve(static_cast<size_t>(n_real));
    for (int64_t i = 0; i < n_real; ++i) {
      const int64_t a_len = (*audio_seqlens)[i].item<int64_t>();
      auto a_pos =
          torch::arange(
              a_len, torch::TensorOptions().dtype(torch::kInt32).device(device))
              .unsqueeze(0)
              .expand({3, a_len});
      audio_pos_list.push_back(a_pos);
    }
    auto audio_position_ids = torch::cat(audio_pos_list, /*dim=*/1);
    packed.audio = make_rope_meta_from_seqlens(
        (*audio_seqlens).to(torch::kInt32).to(device),
        audio_position_ids,
        device);
  }

  // Mix: per real sample [text | pixel | audio]
  std::vector<torch::Tensor> mix_pos_list;
  std::vector<int32_t> mix_lens;
  mix_pos_list.reserve(static_cast<size_t>(n_real));
  for (int64_t i = 0; i < n_real; ++i) {
    const int64_t t_len = text_seqlens[i].item<int64_t>();
    auto t_pos =
        torch::arange(
            t_len, torch::TensorOptions().dtype(torch::kInt32).device(device))
            .unsqueeze(0)
            .expand({3, t_len});
    auto p_pos =
        build_position_ids(latent_t, latent_h, latent_w, device, t_len);
    if (has_audio) {
      // Mix audio uses fractional positions overlapping pixel T-axis
      // (SGLang / Joytron audio-fractional-rope), not after pixel_seqlen.
      const int64_t a_len = (*audio_seqlens)[i].item<int64_t>();
      auto a_pos =
          torch::linspace(
              static_cast<double>(t_len),
              static_cast<double>(t_len + latent_t - 1),
              a_len,
              torch::TensorOptions().dtype(torch::kFloat).device(device))
              .unsqueeze(0)
              .expand({3, a_len});
      mix_pos_list.push_back(
          torch::cat({t_pos.to(torch::kFloat), p_pos.to(torch::kFloat), a_pos},
                     /*dim=*/1));
      mix_lens.push_back(static_cast<int32_t>(t_len + pixel_seqlen + a_len));
    } else {
      mix_pos_list.push_back(torch::cat({t_pos, p_pos}, /*dim=*/1));
      mix_lens.push_back(static_cast<int32_t>(t_len + pixel_seqlen));
    }
  }
  auto mix_position_ids = torch::cat(mix_pos_list, /*dim=*/1);
  auto mix_seqlens = torch::tensor(
      mix_lens, torch::TensorOptions().dtype(torch::kInt32).device(device));
  packed.mix =
      make_rope_meta_from_seqlens(mix_seqlens, mix_position_ids, device);
  return packed;
}

// Joytron GaussianDenoiser.denoise_update — CFG Zero-Star + Euler.
inline torch::Tensor denoise_update(torch::Tensor hidden_states,
                                    torch::Tensor noise_pred,
                                    const torch::Tensor& sigmas,
                                    int64_t step_idx,
                                    double guidance_scale,
                                    int64_t zero_cfg_star_step,
                                    int64_t zero_init_steps,
                                    bool do_cfg) {
  const auto dt = sigmas[step_idx + 1] - sigmas[step_idx];
  hidden_states = hidden_states.to(torch::kFloat);
  noise_pred = noise_pred.to(torch::kFloat);

  if (do_cfg) {
    auto hs = hidden_states.chunk(2, /*dim=*/0);
    auto np = noise_pred.chunk(2, /*dim=*/0);
    auto& hs_pos = hs[0];
    auto& np_pos = np[0];
    auto& np_neg = np[1];

    torch::Tensor st_star;
    if (zero_cfg_star_step >= 0) {
      const auto dot = torch::sum(np_neg * np_pos);
      const auto sq = torch::sum(np_neg * np_neg) + 1e-8;
      st_star = dot / sq;
    } else {
      st_star = torch::tensor(1.0, np_pos.options());
    }

    auto combined =
        np_neg * st_star + guidance_scale * (np_pos - np_neg * st_star);
    if (step_idx >= zero_init_steps) {
      hs_pos = hs_pos + combined * dt;
    }
    hidden_states = torch::cat({hs_pos, hs_pos}, /*dim=*/0);
  } else if (step_idx >= zero_init_steps) {
    hidden_states = hidden_states + noise_pred * dt;
  }
  return hidden_states.to(torch::kBFloat16);
}

inline std::pair<int64_t, int64_t>
latent_hw_from_pixels(int64_t height, int64_t width, int64_t scale_spatial) {
  return {height / scale_spatial, width / scale_spatial};
}

inline int64_t latent_t_from_frames(int64_t num_frames,
                                    int64_t scale_temporal) {
  // XVAE: (num_frames - 1) / temporal + 1  (common causal VAE convention)
  return (num_frames - 1) / scale_temporal + 1;
}
}  // namespace joyo_v2

using JoyOV2TextEncoder = Qwen3_VLForConditionalGeneration;

class JoyOV2PipelineImpl : public torch::nn::Module {
 public:
  explicit JoyOV2PipelineImpl(const DiTModelContext& context)
      : options_(joyo_v2::resolve_component_tensor_options(
            context.get_tensor_options(),
            context.get_model_args("transformer").dtype())),
        parallel_args_(context.get_parallel_args()) {
    LOG(INFO) << "Initializing JoyOV2Pipeline (T2VA)...";

    transformer_ = register_module(
        "transformer",
        JoyOV2Transformer3DModel(context.get_model_context("transformer")));

    CHECK(context.has_component("vae"))
        << "JoyOV2Pipeline requires vae component";
    vae_ = register_module("vae",
                           AutoencoderKLXVAE(context.get_model_context("vae")));

    generate_audio_ = context.has_component("audio_vae");
    if (generate_audio_) {
      audio_vae_ = register_module(
          "audio_vae",
          AutoencoderKLXDAC(context.get_model_context("audio_vae")));
    }

    enable_joyo_text_encoder_ =
        DiTConfig::get_instance().dit_enable_joyo_text_encoder();
    if (enable_joyo_text_encoder_) {
      CHECK(context.has_component("text_encoder"))
          << "JoyOV2 --dit_enable_joyo_text_encoder requires a text_encoder "
             "component";
#if defined(USE_NPU)
      CHECK_EQ(KernelConfig::get_instance().npu_kernel_backend(), "TORCH")
          << "JoyOV2 in-process Qwen3-VL requires --npu_kernel_backend=TORCH";
#endif
      ProcessGroup* tp_group = parallel_args_.dit_text_encoder_tp_group_;
      CHECK(tp_group != nullptr)
          << "JoyOV2 in-process encoder needs dit_text_encoder_tp_group_ "
             "(pass --text_encoder_tp_size explicitly, typically = --tp_size)";
      ParallelArgs vlm_parallel_args(
          tp_group->rank(), tp_group->world_size(), tp_group);
      vlm_parallel_args.tp_size(tp_group->world_size());
      vlm_parallel_args.tp_group_ = tp_group;
      text_encoder_model_args_ = context.get_model_args("text_encoder");
      text_encoder_empty_kv_caches_.resize(
          static_cast<size_t>(text_encoder_model_args_.n_layers()));
      const ModelContext source_vlm_context =
          context.get_model_context("text_encoder");
      // Joy-only: seed from this component's config.json dtype, not the
      // worker-wide get_torch_dtype() first-hit (often float32 from VAE).
      const torch::TensorOptions te_options =
          joyo_v2::resolve_component_tensor_options(
              source_vlm_context.get_tensor_options(),
              source_vlm_context.get_model_args().dtype());
      ModelContext vlm_context(vlm_parallel_args,
                               source_vlm_context.get_model_args(),
                               source_vlm_context.get_quant_args(),
                               te_options);
      text_encoder_ =
          register_module("text_encoder", JoyOV2TextEncoder(vlm_context));
      CHECK(!text_encoder_.is_empty())
          << "Failed to create JoyOV2 Qwen3-VL text encoder";
      LOG(INFO) << "JoyOV2Pipeline: in-process Qwen3-VL encoder "
                   "te_tp="
                << tp_group->world_size() << " local_rank=" << tp_group->rank()
                << " (resident, no rolling)";
    } else if (context.has_component("text_encoder")) {
      LOG(INFO) << "JoyOV2Pipeline: model tree has text_encoder but it is not "
                   "loaded (use --dit_enable_joyo_text_encoder, or "
                   "external prompt_embed)";
    }

    latent_channels_ = transformer_->latent_channels();
    text_dim_ = transformer_->text_dim();
    CHECK_EQ(vae_->latent_channels(), latent_channels_)
        << "JoyOV2 XVAE latent_channels (" << vae_->latent_channels()
        << ") must match transformer (" << latent_channels_ << ")";
    if (generate_audio_) {
      CHECK_EQ(audio_vae_->latent_channels(), latent_channels_)
          << "JoyOV2 audio VAE latent_channels ("
          << audio_vae_->latent_channels() << ") must match transformer ("
          << latent_channels_ << ")";
    }

    LOG(INFO) << "JoyOV2Pipeline ready: generate_audio=" << generate_audio_
              << " rolling="
              << JoyOV2Transformer3DModelImpl::dit_rolling_load_enabled()
              << " in_process_te=" << enable_joyo_text_encoder_
              << " latent_channels=" << latent_channels_
              << " text_dim=" << text_dim_;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    CHECK(loader != nullptr);
    LOG(INFO) << "JoyOV2Pipeline loading from " << loader->model_root_path();

    auto transformer_loader = loader->take_component_loader("transformer");
    transformer_->load_model(std::move(transformer_loader));
#if defined(USE_NPU)
    use_rolling_load_ =
        JoyOV2Transformer3DModelImpl::dit_rolling_load_enabled();
    if (use_rolling_load_) {
      transformer_->prepare_for_rolling_load(options_.device());
      init_rolling_transformer();
      LOG(INFO) << "JoyOV2Pipeline: DitRollingLoad ready (slots="
                << LoadConfig::get_instance().rolling_load_num_rolling_slots()
                << ")";
    } else {
      transformer_->prepare_resident_device(options_.device());
      LOG(INFO) << "JoyOV2Pipeline: whole DiT resident on device (no rolling)";
    }
#else
    // Rolling unavailable off NPU; keep whole DiT resident like no-rolling NPU.
    transformer_->prepare_resident_device(options_.device());
    LOG(INFO) << "JoyOV2Pipeline: whole DiT resident on device "
                 "(non-NPU build; rolling unavailable)";
#endif

    auto vae_loader = loader->take_component_loader("vae");
    vae_->load_model(std::move(vae_loader));
    joyo_v2::module_to_device(vae_, options_.device());

    if (generate_audio_) {
      auto audio_vae_loader = loader->take_component_loader("audio_vae");
      audio_vae_->load_model(std::move(audio_vae_loader));
      joyo_v2::module_to_device(audio_vae_, options_.device());
    }

    if (enable_joyo_text_encoder_) {
      auto text_encoder_loader = loader->take_component_loader("text_encoder");
      CHECK(text_encoder_loader != nullptr);
      std::unique_ptr<Tokenizer> tokenizer;
      if (loader->has_component("tokenizer")) {
        auto tokenizer_loader = loader->take_component_loader("tokenizer");
        tokenizer = tokenizer_loader->tokenizer();
      } else {
        tokenizer = text_encoder_loader->tokenizer();
      }
      CHECK(tokenizer != nullptr) << "JoyOV2 failed to load Qwen3-VL tokenizer";
      tokenizer_ = std::shared_ptr<Tokenizer>(std::move(tokenizer));
      text_encoder_->load_model(std::move(text_encoder_loader));
      set_text_encoder_on_device(true);
    }

    activate_stage(joyo_v2::PipelineStage::kDenoise);
    LOG(INFO) << "JoyOV2Pipeline load done; active_stage=denoise"
              << " rolling=" << use_rolling_load_
              << " in_process_te=" << enable_joyo_text_encoder_;
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    torch::NoGradGuard no_grad;

    const DiTGenerationParams& params = input.generation_params;
    const int64_t height = params.height > 0 ? params.height : 288;
    const int64_t width = params.width > 0 ? params.width : 512;
    const int64_t num_frames = params.num_frames > 0 ? params.num_frames : 161;
    const int64_t num_steps =
        params.num_inference_steps > 0 ? params.num_inference_steps : 50;
    const double guidance =
        params.guidance_scale > 0 ? params.guidance_scale : 4.0;
    const double timeshift =
        params.flow_shift > 0 ? static_cast<double>(params.flow_shift) : 8.0;
    const double fps = params.video_fps > 0 ? params.video_fps : 16.0;
    const bool do_cfg = guidance > 1.0;
    const int64_t zero_cfg_star_step = 0;
    const int64_t zero_init_steps = 0;
    const int64_t batch_size = 1;
    const int64_t n_cfg = do_cfg ? 2 : 1;

    const int64_t scale_s =
        vae_->scale_factor_spatial() > 0 ? vae_->scale_factor_spatial() : 32;
    const int64_t scale_t =
        vae_->scale_factor_temporal() > 0 ? vae_->scale_factor_temporal() : 4;
    const auto [latent_h, latent_w] =
        joyo_v2::latent_hw_from_pixels(height, width, scale_s);
    const int64_t latent_t = joyo_v2::latent_t_from_frames(num_frames, scale_t);

    LOG(INFO) << "JoyOV2Pipeline::forward H=" << height << " W=" << width
              << " frames=" << num_frames << " steps=" << num_steps;

    const auto device = options_.device();
    const auto dtype = options_.dtype().toScalarType();

    // --- Text embeds ---
    torch::Tensor text_embeddings;
    torch::Tensor text_seqlens;
    resolve_prompt_embeds(
        input, do_cfg, n_cfg, device, text_embeddings, text_seqlens);

    // --- Stage: denoise ---
    activate_stage(joyo_v2::PipelineStage::kDenoise);
    const auto t_denoise0 = std::chrono::steady_clock::now();

    // Private CPU Generator → .to(device); do not touch global / NPU RNG.
    auto latents = xllm::dit::randn_tensor(
        {batch_size, latent_channels_, latent_t, latent_h, latent_w},
        params.seed,
        options_,
        torch::kFloat32);
    if (do_cfg) {
      latents = torch::cat({latents, latents}, /*dim=*/0);
    }
    const int64_t num_samples = latents.size(0);
    auto hidden_states = latents.permute({0, 2, 3, 4, 1})
                             .reshape({-1, latent_channels_})
                             .to(dtype);

    torch::Tensor audio_hidden_states;
    torch::Tensor audio_seqlens;
    torch::Tensor audio_sigmas;
    int64_t audio_latent_t = 0;
    if (generate_audio_) {
      CHECK_EQ(params.audio_sampling_rate, audio_vae_->sampling_rate())
          << "JoyOV2 sampling_rate must match the audio VAE config";
      const int64_t sr = audio_vae_->sampling_rate();
      const int64_t hop = audio_vae_->hop_length();
      audio_latent_t =
          joyo_v2::compute_audio_latent_length(sr, hop, num_frames, fps);
      // seed+1 keeps audio stream independent of video noise.
      auto audio_noise = xllm::dit::randn_tensor(
          {batch_size, audio_latent_t, latent_channels_},
          params.seed + 1,
          options_,
          torch::kFloat32);
      if (do_cfg) {
        audio_noise = torch::cat({audio_noise, audio_noise}, /*dim=*/0);
      }
      audio_hidden_states =
          audio_noise.reshape({-1, latent_channels_}).to(dtype);
      audio_seqlens = torch::full(
          {num_samples},
          audio_latent_t,
          torch::TensorOptions().dtype(torch::kInt32).device(device));
    }

    auto rope_meta = joyo_v2::build_rope_meta(
        text_seqlens,
        latent_t,
        latent_h,
        latent_w,
        device,
        audio_hidden_states.defined()
            ? std::optional<torch::Tensor>(audio_seqlens)
            : std::nullopt,
        /*num_real_samples=*/num_samples);

    auto t_lin = torch::linspace(1, 0, num_steps + 1, device);
    auto sigmas = joyo_v2::sd3_timeshift(t_lin, timeshift);
    if (audio_hidden_states.defined()) {
      audio_sigmas = joyo_v2::sd3_timeshift(t_lin, timeshift);
    }

    for (int64_t step_idx = 0; step_idx < num_steps; ++step_idx) {
      auto timestep = sigmas[step_idx].reshape({1}).expand({num_samples});
      std::optional<torch::Tensor> audio_in =
          audio_hidden_states.defined()
              ? std::optional<torch::Tensor>(audio_hidden_states)
              : std::nullopt;
      std::pair<torch::Tensor, torch::Tensor> preds;
#if defined(USE_NPU)
      if (use_rolling_load_) {
        preds = transformer_->forward(
            hidden_states,
            text_embeddings,
            timestep,
            rope_meta,
            audio_in,
            [this](int32_t i) { rolling_transformer_.wait_h2d(i); },
            [this](int32_t i) { rolling_transformer_.schedule_next_h2d(i); });
      } else {
        preds = transformer_->forward(
            hidden_states, text_embeddings, timestep, rope_meta, audio_in);
      }
#else
      preds = transformer_->forward(
          hidden_states, text_embeddings, timestep, rope_meta, audio_in);
#endif
      auto& noise_pred = preds.first;
      auto& audio_pred = preds.second;

      hidden_states = joyo_v2::denoise_update(hidden_states,
                                              noise_pred,
                                              sigmas,
                                              step_idx,
                                              guidance,
                                              zero_cfg_star_step,
                                              zero_init_steps,
                                              do_cfg);
      if (audio_hidden_states.defined()) {
        audio_hidden_states = joyo_v2::denoise_update(audio_hidden_states,
                                                      audio_pred,
                                                      audio_sigmas,
                                                      step_idx,
                                                      guidance,
                                                      zero_cfg_star_step,
                                                      zero_init_steps,
                                                      do_cfg);
      }
    }

    if (do_cfg) {
      const int64_t s_per = latent_t * latent_h * latent_w;
      hidden_states = hidden_states.slice(/*dim=*/0, 0, batch_size * s_per);
      if (audio_hidden_states.defined()) {
        audio_hidden_states = audio_hidden_states.slice(
            /*dim=*/0, 0, batch_size * audio_latent_t);
      }
    }

    const auto t_denoise1 = std::chrono::steady_clock::now();
    const double denoise_s =
        std::chrono::duration<double>(t_denoise1 - t_denoise0).count();

    // --- Stage: decode ---
    activate_stage(joyo_v2::PipelineStage::kDecode);
    const auto t_decode0 = std::chrono::steady_clock::now();

    auto video_latents =
        hidden_states.to(torch::kFloat)
            .view({batch_size, latent_t, latent_h, latent_w, latent_channels_})
            .permute({0, 4, 1, 2, 3})
            .contiguous();
    video_latents = vae_->denormalize_latents(video_latents);
    auto video = vae_->decode(video_latents).to(torch::kFloat);
    video = (video / 2.0 + 0.5).clamp(/*min=*/0.0, /*max=*/1.0);
    video = video.permute({0, 2, 1, 3, 4}).contiguous();

    DiTForwardOutput out;
    out.tensors.push_back(video.cpu());
    if (audio_hidden_states.defined()) {
      auto audio_latents =
          audio_hidden_states.to(torch::kFloat)
              .view({batch_size, audio_latent_t, latent_channels_})
              .permute({0, 2, 1})
              .contiguous();
      audio_latents = audio_vae_->denormalize_latents(audio_latents);
      auto audio = audio_vae_->decode(audio_latents).to(torch::kFloat);
      out.audio_tensors.push_back(audio.cpu());
    }
    const auto t_decode1 = std::chrono::steady_clock::now();
    const double decode_s =
        std::chrono::duration<double>(t_decode1 - t_decode0).count();
    LOG(INFO) << "JoyOV2Pipeline: decoded video " << video.sizes()
              << (generate_audio_ ? " + audio" : "")
              << " timing_s denoise=" << denoise_s << " decode=" << decode_s
              << " sum=" << (denoise_s + decode_s);

    // Stay in decode residency; next request activate_stage(denoise) remats
    // DiT embedders.
    return out;
  }

 private:
  static torch::Tensor normalize_text_emb(torch::Tensor t, int64_t text_dim) {
    // H3 uses [1, T, D]; JoyO DiT consumes packed [T, D].
    // Wire: client sends [T,D]; DiTBatch unsqueezes → [1,T,D] (same as H3).
    if (t.dim() == 3 && t.size(0) == 1) {
      t = t.squeeze(0);
    }
    CHECK_EQ(t.dim(), 2) << "JoyOV2 prompt_embed must be [T,D] or [1,T,D], got "
                         << t.sizes();
    CHECK_EQ(t.size(1), text_dim)
        << "JoyOV2 prompt_embed last dim must be text_dim=" << text_dim;
    CHECK(t.is_floating_point())
        << "JoyOV2 prompt_embed must be floating point";
    CHECK(torch::isfinite(t).all().item<bool>())
        << "JoyOV2 prompt_embed must not contain NaN or Inf";
    return t;
  }

  ModelInputParams build_text_encoder_input(const torch::Tensor& tokens) const {
    CHECK_LE(tokens.numel(), std::numeric_limits<int32_t>::max())
        << "JoyOV2 Qwen3-VL prompt is too long";
    const int32_t sequence_length = static_cast<int32_t>(tokens.numel());
    CHECK_GT(sequence_length, 0) << "JoyOV2 Qwen3-VL prompt must not be empty";

    ModelInputParams params;
    params.meta.num_sequences = 1;
    params.meta.actual_num_sequences = 1;
    params.meta.q_max_seq_len = sequence_length;
    params.meta.kv_max_seq_len = sequence_length;
    params.meta.batch_forward_type = BatchForwardType::PREFILL;
    params.prefill_without_cache = true;
    params.attention.host.q_seq_lens = {sequence_length};
    params.attention.host.kv_seq_lens = {sequence_length};
#if defined(USE_NPU)
    params.attention.host.q_cu_seq_lens = {sequence_length};
#else
    params.attention.host.q_cu_seq_lens = {0, sequence_length};
#endif
    params.attention.device.q_seq_lens =
        torch::tensor({sequence_length}, torch::kInt).to(tokens.device());
    params.attention.device.kv_seq_lens =
        torch::tensor({sequence_length}, torch::kInt).to(tokens.device());
#if defined(USE_NPU)
    params.attention.device.q_cu_seq_lens =
        torch::tensor({sequence_length}, torch::kInt).to(tokens.device());
#else
    params.attention.device.q_cu_seq_lens =
        torch::tensor({0, sequence_length}, torch::kInt).to(tokens.device());
#endif
    return params;
  }

  torch::Tensor encode_one_prompt(const std::string& prompt) {
    CHECK(!text_encoder_.is_empty()) << "Qwen3-VL text encoder is not loaded";
    CHECK(tokenizer_ != nullptr) << "Qwen3-VL tokenizer is not loaded";

    const std::string wrapped =
        std::string(kJoyoChatPrefix) + prompt + kJoyoChatSuffix;
    std::vector<int32_t> token_ids;
    CHECK(tokenizer_->encode(wrapped, &token_ids, /*add_special_tokens=*/false))
        << "JoyOV2 Qwen3-VL tokenizer encode failed";
    CHECK(!token_ids.empty()) << "JoyOV2 Qwen3-VL encoded empty prompt";

    const auto device = options_.device();
    torch::Tensor tokens =
        torch::tensor(token_ids, torch::TensorOptions().dtype(torch::kInt32))
            .to(device);
    torch::Tensor positions =
        torch::arange(tokens.numel(),
                      torch::TensorOptions().dtype(torch::kInt).device(device));
    ModelInputParams input_params = build_text_encoder_input(tokens);
    input_params.embedding.input_embedding =
        text_encoder_->get_input_embeddings(tokens, input_params);
    ModelOutput model_output = text_encoder_->forward(
        tokens, positions, text_encoder_empty_kv_caches_, input_params);
    CHECK(model_output.residual.defined())
        << "JoyOV2 requires Qwen3-VL pre-norm residual (TORCH backend)";
    torch::Tensor residual = model_output.residual;
    if (residual.dim() == 3 && residual.size(0) == 1) {
      residual = residual.squeeze(0);
    }
    CHECK_EQ(residual.dim(), 2)
        << "JoyOV2 encoder residual must be [T,D], got " << residual.sizes();
    // JoyOV2 DiT caption path has no text projection: Qwen3-VL hidden size
    // must equal transformer text_dim_ (typically 5120). A mismatch means
    // the wrong encoder checkpoint / DiT config pair, not a missing Linear.
    CHECK_EQ(residual.size(1), text_dim_)
        << "JoyOV2 encoder residual last dim must be text_dim=" << text_dim_
        << " got " << residual.size(1);
    return normalize_text_emb(residual.to(options_), text_dim_);
  }

  void set_text_encoder_on_device(bool on_device) {
    if (!enable_joyo_text_encoder_ || text_encoder_.is_empty()) {
      return;
    }
    if (text_encoder_on_device_ == on_device) {
      return;
    }
    const torch::Device device = options_.device();
    if (on_device) {
      joyo_v2::module_to_device(text_encoder_, device);
    } else {
      joyo_v2::module_to_cpu(text_encoder_);
      CHECK_EQ(joyo_v2::module_npu_tensor_count(*text_encoder_), 0)
          << "JoyOV2 text encoder still has tensors on NPU after D2H";
    }
    joyo_v2::empty_device_cache(device);
    text_encoder_on_device_ = on_device;
  }

  void resolve_prompt_embeds(const DiTForwardInput& input,
                             bool do_cfg,
                             int64_t n_cfg,
                             const torch::Device& device,
                             torch::Tensor& text_embeddings,
                             torch::Tensor& text_seqlens) {
    const std::optional<NamedTensorConstRef> prompt_embed =
        input.tensor_sources.get_named_tensor("prompt_embed");
    if (prompt_embed.has_value()) {
      resolve_external_prompt_embeds(
          input, do_cfg, n_cfg, device, text_embeddings, text_seqlens);
      return;
    }

    CHECK(enable_joyo_text_encoder_)
        << "JoyOV2 T2VA requires prompt_embed (external encoder) or "
           "--dit_enable_joyo_text_encoder";
    CHECK(!text_encoder_.is_empty())
        << "JoyOV2 in-process encoder is enabled but not constructed";
    CHECK(!input.prompts.empty())
        << "JoyOV2 in-process encode requires input.prompts";

    set_text_encoder_on_device(true);
    const auto t_enc0 = std::chrono::steady_clock::now();
    text_embeddings = encode_one_prompt(input.prompts.front());
    if (do_cfg) {
      CHECK(!input.negative_prompts.empty())
          << "JoyOV2 CFG with in-process encoder requires negative_prompt";
      torch::Tensor neg_emb = encode_one_prompt(input.negative_prompts.front());
      const int64_t s0 = text_embeddings.size(0);
      const int64_t s1 = neg_emb.size(0);
      text_seqlens = torch::tensor(
          {s0, s1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
      text_embeddings = torch::cat({text_embeddings, neg_emb}, /*dim=*/0);
    } else {
      text_seqlens = torch::tensor(
          {text_embeddings.size(0)},
          torch::TensorOptions().dtype(torch::kInt32).device(device));
    }
    const auto t_enc1 = std::chrono::steady_clock::now();
    const double encode_s =
        std::chrono::duration<double>(t_enc1 - t_enc0).count();
    LOG(INFO) << "JoyOV2Pipeline: in-process encode " << text_embeddings.sizes()
              << " seqlens=" << text_seqlens << " encode_s=" << encode_s;
    // Text encoder is unused after encode; drop it before denoise/decode
    // Conv3d.
    set_text_encoder_on_device(false);
  }

  void resolve_external_prompt_embeds(const DiTForwardInput& input,
                                      bool do_cfg,
                                      int64_t n_cfg,
                                      const torch::Device& device,
                                      torch::Tensor& text_embeddings,
                                      torch::Tensor& text_seqlens) const {
    const std::optional<NamedTensorConstRef> prompt_embed =
        input.tensor_sources.get_named_tensor("prompt_embed");
    CHECK(prompt_embed.has_value());

    text_embeddings =
        normalize_text_emb(prompt_embed->get().tensor.to(options_), text_dim_);

    if (do_cfg && input.tensor_sources.contains("negative_prompt_embed")) {
      torch::Tensor neg_emb =
          normalize_text_emb(input.tensor_sources.get("negative_prompt_embed")
                                 .value()
                                 .to(options_),
                             text_dim_);
      const int64_t s0 = text_embeddings.size(0);
      const int64_t s1 = neg_emb.size(0);
      text_seqlens = torch::tensor(
          {s0, s1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
      text_embeddings = torch::cat({text_embeddings, neg_emb}, /*dim=*/0);
    } else {
      // H3-style: side-channel on Tensor.parameters (see prompt_token_tags).
      const std::vector<int64_t>* seq_param =
          get_tensor_parameter<std::vector<int64_t>>(
              prompt_embed->get().parameters, "joyo_text_seqlens");
      if (seq_param != nullptr && !seq_param->empty()) {
        CHECK_EQ(static_cast<int64_t>(seq_param->size()), n_cfg)
            << "joyo_text_seqlens size must equal n_cfg=" << n_cfg;
        int64_t sum = 0;
        for (int64_t s : *seq_param) {
          CHECK_GT(s, 0);
          sum += s;
        }
        CHECK_EQ(sum, text_embeddings.size(0))
            << "joyo_text_seqlens sum must match prompt_embed T";
        text_seqlens = torch::tensor(
            *seq_param,
            torch::TensorOptions().dtype(torch::kInt32).device(device));
      } else {
        CHECK_EQ(text_embeddings.size(0) % n_cfg, 0)
            << "packed prompt_embed T must divide n_cfg without "
               "joyo_text_seqlens / negative_prompt_embed";
        const int64_t per = text_embeddings.size(0) / n_cfg;
        text_seqlens = torch::full(
            {n_cfg},
            per,
            torch::TensorOptions().dtype(torch::kInt32).device(device));
      }
    }

    LOG(INFO) << "JoyOV2Pipeline: prompt_embed " << text_embeddings.sizes()
              << " seqlens=" << text_seqlens;
  }

  void activate_stage(joyo_v2::PipelineStage stage) {
    if (active_stage_ == stage) {
      return;
    }
    const auto device = options_.device();
    switch (stage) {
      case joyo_v2::PipelineStage::kDenoise: {
#if defined(USE_NPU)
        if (use_rolling_load_) {
          transformer_->prepare_rolling_denoise(device);
        } else {
          transformer_->prepare_resident_device(device);
        }
#else
        // prepare_rolling_denoise only H2Ds non-block modules; blocks stay on
        // CPU unless prepare_resident_device (rolling is NPU-only).
        transformer_->prepare_resident_device(device);
#endif
        break;
      }
      case joyo_v2::PipelineStage::kDecode:
        // Text encoder unused in VAE decode; D2H so Conv3d / GE compile has
        // HBM.
        set_text_encoder_on_device(false);
        break;
      case joyo_v2::PipelineStage::kIdle:
      default:
        break;
    }
    active_stage_ = stage;
  }

#if defined(USE_NPU)
  void init_rolling_transformer() {
    torch::DeviceGuard guard(options_.device());
    auto loaders = transformer_->get_block_weight_loaders();
    CHECK(!loaders.empty()) << "JoyOV2 rolling: no block weight loaders";

    size_t max_storage = 0;
    for (auto* loader : loaders) {
      max_storage = std::max(max_storage, loader->storage_size());
    }

    auto& load_config = LoadConfig::get_instance();
    int32_t num_slots =
        std::max(load_config.rolling_load_num_rolling_slots(), 2);
    auto buffer =
        std::make_shared<layer::RollingWeightBuffer>(num_slots, max_storage);
    rolling_transformer_.init(std::move(loaders), std::move(buffer), num_slots);
    rolling_transformer_.preload();
  }
#endif

  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
  joyo_v2::PipelineStage active_stage_ = joyo_v2::PipelineStage::kIdle;
  bool generate_audio_ = false;
  bool use_rolling_load_ = false;
  bool enable_joyo_text_encoder_ = false;
  bool text_encoder_on_device_ = false;
  int64_t latent_channels_ = 128;
  int64_t text_dim_ = 5120;

  static constexpr const char* kJoyoChatPrefix = "<|im_start|>user\n";
  static constexpr const char* kJoyoChatSuffix =
      "<|im_end|>\n<|im_start|>assistant\n";

  JoyOV2Transformer3DModel transformer_{nullptr};
  AutoencoderKLXVAE vae_{nullptr};
  AutoencoderKLXDAC audio_vae_{nullptr};
  JoyOV2TextEncoder text_encoder_{nullptr};
  std::shared_ptr<Tokenizer> tokenizer_;
  ModelArgs text_encoder_model_args_;
  std::vector<KVCache> text_encoder_empty_kv_caches_;
#if defined(USE_NPU)
  dit::DitRollingLoadManager rolling_transformer_;
#endif
};
TORCH_MODULE(JoyOV2Pipeline);

REGISTER_DIT_MODEL(JoyOV2Pipeline, JoyOV2Pipeline);

}  // namespace xllm
