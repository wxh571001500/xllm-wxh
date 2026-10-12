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
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/config/dit_config.h"
#include "core/framework/dit_model_context.h"
#include "core/framework/dit_model_loader.h"
#include "core/runtime/dit_forward_params.h"
#include "models/dit/autoencoders/autoencoder_kl_minimax_h3_audio.h"
#include "models/dit/autoencoders/autoencoder_kl_minimax_h3_video.h"
#include "models/dit/processors/minimax_h3_audio_processor.h"
#include "models/dit/processors/minimax_h3_visual_processor.h"
#include "models/dit/schedulers/minimax_h3_scheduler.h"
#include "models/dit/transformers/transformer_minimax_h3.h"
#include "models/model_registry.h"

namespace xllm {

struct MiniMaxH3TimestepLayout {
  torch::Tensor timesteps;
  torch::Tensor indices;
};

class MiniMaxH3PipelineImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3PipelineImpl(const DiTModelContext& context)
      : options_(context.get_tensor_options()),
        tp_group_(context.get_parallel_args().dit_tp_group_),
        is_tp_driver_(tp_group_ == nullptr || tp_group_->rank() == 0),
        task_type_(DiTConfig::get_instance().dit_h3_task_type()) {
    CHECK(task_type_ == "fl2va" || task_type_ == "ref2va")
        << "MiniMax-H3 dit_h3_task_type must be fl2va or ref2va";
    const ModelArgs& transformer_args = context.get_model_args("transformer");
    const ModelArgs& vae_args = context.get_model_args("vae");
    const std::vector<int64_t>& patch_size = transformer_args.wan_patch_size();
    CHECK_EQ(patch_size.size(), static_cast<size_t>(3))
        << "MiniMax-H3 transformer patch_size must have 3 dimensions";
    patch_h_ = patch_size[1];
    patch_w_ = patch_size[2];
    video_latent_channels_ = transformer_args.in_channels();
    audio_latent_channels_ = transformer_args.audio_in_channels();
    text_hidden_size_ = transformer_args.text_dim();
    vae_spatial_ratio_ = product_or_one(vae_args.spatial_downsample_factors());
    vae_temporal_ratio_ =
        product_or_one(vae_args.temporal_downsample_factors());
    video_encode_clip_length_ = vae_args.clip_length();
    video_encode_token_drop_ = vae_args.token_drop();
    video_encode_latents_per_clip_ =
        (video_encode_clip_length_ - 1) / vae_temporal_ratio_ + 1;
    CHECK_GT(patch_h_, 0);
    CHECK_GT(patch_w_, 0);
    CHECK_GT(video_latent_channels_, 0);
    CHECK_GT(audio_latent_channels_, 0);
    CHECK_GT(text_hidden_size_, 0);
    CHECK_GT(vae_spatial_ratio_, 0);
    CHECK_GT(vae_temporal_ratio_, 0);
    CHECK_GT(video_encode_clip_length_, 0);
    CHECK_GE(video_encode_token_drop_, 0);
    CHECK_GT(video_encode_latents_per_clip_, 0);
    CHECK_LT(video_encode_latents_per_clip_, video_encode_clip_length_);
    CHECK_LT(video_encode_token_drop_, video_encode_latents_per_clip_);
    LOG(INFO) << "Initializing MiniMax-H3 " << task_type_ << " native pipeline";
    visual_processor_ = register_module(
        "visual_processor",
        MiniMaxH3VisualProcessor(
            context.get_model_context("vae"),
            MiniMaxH3VisualProcessorConfig{
                /*canvas_multiple=*/vae_spatial_ratio_ * patch_h_}));
    audio_processor_ = register_module(
        "audio_processor",
        MiniMaxH3AudioProcessor(context.get_model_context("audio_vae"),
                                /*channel_count=*/audio_channels_));
    transformer_ =
        register_module("transformer", MiniMaxH3Transformer3DModel(context));
    if (is_tp_driver_) {
      vae_ = register_module(
          "vae", AutoencoderKLMiniMaxH3(context.get_model_context("vae")));
      audio_vae_ = register_module(
          "audio_vae",
          AutoencoderKLMiniMaxH3Audio(context.get_model_context("audio_vae")));
    }
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    torch::NoGradGuard no_grad;
    // 1. Prepare Parameters
    CHECK_EQ(input.batch_size, 1)
        << "MiniMax-H3 native T2VA currently supports batch_size=1";
    CHECK_EQ(input.generation_params.num_videos_per_prompt, 1)
        << "MiniMax-H3 currently supports num_videos_per_prompt=1";
    const std::optional<NamedTensorConstRef> prompt_embed =
        input.tensor_sources.get_named_tensor("prompt_embed");
    CHECK(prompt_embed.has_value())
        << "MiniMax-H3 T2VA requires prompt_embed from the encoder service";
    const std::vector<int64_t>* prompt_token_tags =
        get_tensor_parameter<std::vector<int64_t>>(
            prompt_embed->get().parameters, "prompt_token_tags");
    CHECK(prompt_token_tags != nullptr)
        << "MiniMax-H3 T2VA requires prompt_token_tags from the encoder "
           "service";
    const torch::Tensor& prompt_embeds = prompt_embed->get().tensor;
    CHECK(prompt_embeds.is_floating_point())
        << "MiniMax-H3 prompt_embed must be floating point";
    CHECK(torch::isfinite(prompt_embeds).all().item<bool>())
        << "MiniMax-H3 prompt_embed must not contain NaN or Inf";
    CHECK_EQ(prompt_embeds.dim(), 3)
        << "MiniMax-H3 prompt_embed must have shape [1, T, 5120]";
    CHECK_EQ(prompt_embeds.size(0), 1);
    CHECK_EQ(prompt_embeds.size(2), text_hidden_size_);
    torch::Tensor text_token_tags = torch::tensor(
        *prompt_token_tags,
        torch::TensorOptions().dtype(torch::kInt64).device(options_.device()));
    CHECK_EQ(text_token_tags.size(0), prompt_embeds.size(1));
    CHECK(torch::all((text_token_tags == video_tag_) |
                     (text_token_tags == text_tag_))
              .item<bool>())
        << "MiniMax-H3 prompt_token_tags may contain only video/text tags";
    const DiTGenerationParams& generation_params = input.generation_params;
    const double video_fps = generation_params.video_fps;
    CHECK(std::isfinite(video_fps) && video_fps > 0.0)
        << "MiniMax-H3 fps must be finite and positive";
    CHECK_EQ(generation_params.audio_sampling_rate,
             audio_processor_->sampling_rate())
        << "MiniMax-H3 sampling_rate must match the audio VAE config";
    validate_media_sources(input.media_sources);
    const int64_t requested_height = generation_params.height;
    const int64_t requested_width = generation_params.width;
    const int64_t height_multiple = vae_spatial_ratio_ * patch_h_;
    const int64_t width_multiple = vae_spatial_ratio_ * patch_w_;
    CHECK_GT(requested_height, 0);
    CHECK_GT(requested_width, 0);
    const int64_t height =
        ((requested_height - 1) / height_multiple + 1) * height_multiple;
    const int64_t width =
        ((requested_width - 1) / width_multiple + 1) * width_multiple;
    if (requested_height != height || requested_width != width) {
      LOG(WARNING) << "MiniMax-H3 canvas rounded from " << requested_width
                   << "x" << requested_height << " to " << width << "x"
                   << height;
    }
    const int64_t requested_frames = generation_params.num_frames;
    CHECK_GT(requested_frames, 0);
    const int64_t num_frames = align_num_frames(requested_frames);
    const double duration = static_cast<double>(num_frames) / video_fps;
    CHECK_GE(duration, min_duration_seconds_)
        << "MiniMax-H3 duration must be at least 5 seconds";
    CHECK_LE(duration, max_duration_seconds_)
        << "MiniMax-H3 duration must be at most 15 seconds";
    if (requested_frames != num_frames) {
      LOG(WARNING) << "MiniMax-H3 num_frames rounded from " << requested_frames
                   << " to " << num_frames;
    }

    const int64_t latent_frames =
        (num_frames - video_encode_latents_per_clip_) /
            video_encode_clip_length_ * video_encode_latents_per_clip_ +
        (video_encode_latents_per_clip_ - video_encode_token_drop_);
    const int64_t latent_height = height / vae_spatial_ratio_;
    const int64_t latent_width = width / vae_spatial_ratio_;
    const int64_t audio_latents =
        audio_processor_->audio_latents_for_video(num_frames, video_fps);
    const int64_t seed =
        generation_params.seed_is_set ? generation_params.seed : 42;

    // 2. Check for FL2VA or Ref2VA
    const bool is_ref2va = task_type_ == "ref2va";
    std::vector<std::string> keyframe_anchors;
    size_t image_count = 0;
    for (const MediaNamedTensor& source : input.media_sources.entries()) {
      if (source.name == "image") {
        ++image_count;
      }
    }
    const bool has_first_keyframe = !is_ref2va && image_count > 0;
    const bool has_last_keyframe = !is_ref2va && image_count == 2;
    CHECK(is_ref2va || image_count <= 2)
        << "MiniMax-H3 FL2VA accepts at most two images";
    if (has_first_keyframe) {
      keyframe_anchors.emplace_back("first");
    }
    if (has_last_keyframe) {
      keyframe_anchors.emplace_back("last");
    }
    const bool has_keyframes = !keyframe_anchors.empty();
    const bool has_video_tags =
        torch::any(text_token_tags == video_tag_).item<bool>();
    if (!is_ref2va) {
      CHECK_EQ(has_keyframes, has_video_tags)
          << "MiniMax-H3 FL2VA requires image-aware encoder output and "
             "matching images inputs";
    }

    // 3. Prepare Layout for FL2VA or Ref2VA
    torch::Generator generator =
        torch::make_generator<torch::CPUGeneratorImpl>();
    generator.set_current_seed(seed);
    ReferenceData reference_data;
    PackedLayout layout;
    torch::Tensor condition_rows;
    torch::Tensor audio_condition_rows;
    if (is_ref2va) {
      reference_data =
          prepare_reference_data(input, num_frames, video_fps, generator);
      condition_rows = reference_data.video_rows;
      audio_condition_rows = reference_data.audio_rows;
      layout = build_ref_layout(text_token_tags,
                                reference_data.blocks,
                                latent_frames,
                                latent_height,
                                latent_width,
                                audio_latents);
    } else {
      layout = build_layout(text_token_tags,
                            latent_frames,
                            latent_height,
                            latent_width,
                            audio_latents,
                            keyframe_anchors);
      if (has_keyframes) {
        // Prepare FL2VA frame conditions.
        condition_rows = prepare_fl_frame_data(
            input, height, width, generator, keyframe_anchors);
        if (tp_group_ != nullptr && tp_group_->world_size() > 1) {
          tp_group_->broadcast(condition_rows, /*root_rank=*/0);
        }
        CHECK_EQ(condition_rows.dim(), 3)
            << "MiniMax-H3 FL2VA condition rows must be [B,S,D]";
        CHECK_EQ(condition_rows.size(0), 1)
            << "MiniMax-H3 FL2VA condition batch size mismatch";
        CHECK_EQ(condition_rows.size(1), layout.num_condition_video_rows)
            << "MiniMax-H3 FL2VA condition row count mismatch";
      }
    }

    // 4. Prepare noise and Patchify
    std::pair<torch::Tensor, torch::Tensor> noise = prepare_noise(
        latent_frames, latent_height, latent_width, audio_latents, generator);
    torch::Tensor video_rows = patchify_video(noise.first);
    if (condition_rows.defined()) {
      video_rows = torch::cat({condition_rows, video_rows}, 1);
    }
    torch::Tensor audio_rows = patchify_audio(noise.second);
    if (audio_condition_rows.defined() && audio_condition_rows.size(1) > 0) {
      audio_rows = torch::cat({audio_condition_rows, audio_rows}, 1);
    }

    // 5. Denoise
    const int64_t num_inference_steps = generation_params.num_inference_steps;
    scheduler_.set_timesteps(num_inference_steps, options_.device());
    audio_scheduler_.set_timesteps(num_inference_steps, options_.device());
    CHECK_EQ(scheduler_.timesteps().numel(),
             audio_scheduler_.timesteps().numel());
    for (int64_t step = 0; step < scheduler_.timesteps().numel(); ++step) {
      torch::Tensor video_timestep = scheduler_.timesteps()[step];
      torch::Tensor audio_timestep = audio_scheduler_.timesteps()[step];
      MiniMaxH3TimestepLayout timestep_layout =
          build_minimax_h3_timestep_layout(
              layout, video_timestep, audio_timestep);
      MiniMaxH3TransformerOutput prediction =
          transformer_->forward(video_rows,
                                audio_rows,
                                prompt_embeds,
                                timestep_layout.timesteps,
                                timestep_layout.indices,
                                layout.position_ids,
                                layout.token_tags,
                                layout.video_indices,
                                layout.audio_indices,
                                layout.text_indices);
      if (layout.num_condition_video_rows > 0) {
        torch::Tensor generated_rows =
            video_rows.slice(1, layout.num_condition_video_rows);
        torch::Tensor generated_prediction =
            prediction.sample.slice(1, layout.num_condition_video_rows);
        generated_rows = scheduler_.step(
            generated_prediction, video_timestep, generated_rows);
        video_rows =
            torch::cat({video_rows.slice(1, 0, layout.num_condition_video_rows),
                        generated_rows},
                       1);
      } else {
        video_rows =
            scheduler_.step(prediction.sample, video_timestep, video_rows);
      }
      if (layout.num_condition_audio_rows > 0) {
        torch::Tensor generated_audio_rows =
            audio_rows.slice(1, layout.num_condition_audio_rows);
        torch::Tensor generated_audio_prediction =
            prediction.audio_sample.slice(1, layout.num_condition_audio_rows);
        generated_audio_rows = audio_scheduler_.step(
            generated_audio_prediction, audio_timestep, generated_audio_rows);
        audio_rows =
            torch::cat({audio_rows.slice(1, 0, layout.num_condition_audio_rows),
                        generated_audio_rows},
                       1);
      } else {
        audio_rows = audio_scheduler_.step(
            prediction.audio_sample, audio_timestep, audio_rows);
      }
    }

    if (!is_tp_driver_) {
      synchronize_tp_ranks();
      return DiTForwardOutput();
    }

    // 6. Decode
    torch::Tensor generated_video_rows =
        layout.num_condition_video_rows > 0
            ? video_rows.slice(1, layout.num_condition_video_rows)
            : video_rows;
    torch::Tensor video_latent = unpatchify_video(
        generated_video_rows, latent_frames, latent_height, latent_width);
    torch::Tensor generated_audio_rows =
        layout.num_condition_audio_rows > 0
            ? audio_rows.slice(1, layout.num_condition_audio_rows)
            : audio_rows;
    torch::Tensor audio_latent =
        unpatchify_audio(generated_audio_rows, audio_latents);

    // Denormalize video latents.
    torch::Tensor video_latent_mean =
        torch::tensor(vae_->latents_mean(),
                      torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(video_latent.device()))
            .view({1, video_latent_channels_, 1, 1, 1});
    torch::Tensor video_latent_std =
        torch::tensor(vae_->latents_std(),
                      torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(video_latent.device()))
            .view({1, video_latent_channels_, 1, 1, 1});
    video_latent =
        video_latent.to(torch::kFloat32) * video_latent_std + video_latent_mean;

    // Denormalize audio latents.
    torch::Tensor audio_latent_mean =
        torch::tensor(audio_vae_->latents_mean(),
                      torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(audio_latent.device()))
            .view({1, audio_latent_channels_, 1});
    torch::Tensor audio_latent_std =
        torch::tensor(audio_vae_->latents_std(),
                      torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(audio_latent.device()))
            .view({1, audio_latent_channels_, 1});
    audio_latent =
        audio_latent.to(torch::kFloat32) * audio_latent_std + audio_latent_mean;

    torch::Tensor video = vae_->decode(video_latent);
    // Convert decoded video to RGB pixels.
    torch::Tensor pixel_mean =
        torch::tensor(std::vector<float>{0.485f, 0.456f, 0.406f},
                      torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(video.device()))
            .view({1, 3, 1, 1, 1});
    torch::Tensor pixel_std =
        torch::tensor(std::vector<float>{0.229f, 0.224f, 0.225f},
                      torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(video.device()))
            .view({1, 3, 1, 1, 1});
    video = torch::clamp(
                video.to(torch::kFloat32) * pixel_std + pixel_mean, 0.0, 1.0)
                .permute({0, 2, 1, 3, 4})
                .contiguous();
    torch::Tensor audio = audio_vae_->decode(audio_latent.to(torch::kFloat32));
    audio = audio.permute({1, 0, 2}).contiguous();
    const int64_t requested_samples = static_cast<int64_t>(
        std::llround(static_cast<double>(num_frames) / video_fps *
                     audio_vae_->sampling_rate()));
    audio = audio.slice(2, 0, requested_samples);

    DiTForwardOutput output;
    output.tensors = torch::chunk(video, /*chunks=*/1, /*dim=*/0);
    output.audio_tensors = torch::chunk(audio, /*chunks=*/1, /*dim=*/0);
    synchronize_tp_ranks();
    return output;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    CHECK(loader != nullptr) << "MiniMax-H3 loader must not be null";
    LOG(INFO) << "MiniMax-H3 " << task_type_ << " loading model from "
              << loader->model_root_path();
    const std::string transformer_component =
        task_type_ == "ref2va" ? "transformer_ref" : "transformer";
    CHECK(loader->has_component(transformer_component));
    CHECK(loader->has_component("vae"));
    CHECK(loader->has_component("audio_vae"));

    transformer_->load_model(
        loader->take_component_loader(transformer_component));
    if (is_tp_driver_) {
      vae_->load_model(loader->take_component_loader("vae"));
      audio_vae_->load_model(loader->take_component_loader("audio_vae"));
    }
    if (loader->has_component("scheduler")) {
      scheduler_.load_model(loader->take_component_loader("scheduler"));
    }
    if (loader->has_component("audio_scheduler")) {
      audio_scheduler_.load_model(
          loader->take_component_loader("audio_scheduler"));
    }

    transformer_->apply_reference_precision(options_.device());
    if (is_tp_driver_) {
      vae_->to(options_.device(), torch::kFloat32);
      audio_vae_->to(options_.device(), torch::kFloat32);
    }
    LOG(INFO) << "MiniMax-H3 " << task_type_ << " native components loaded";
  }

 private:
  void validate_media_sources(const DiTMediaSources& media_sources) const {
    size_t image_count = 0;
    size_t video_count = 0;
    size_t audio_count = 0;
    size_t video_audio_count = 0;
    const std::vector<MediaNamedTensor>& sources = media_sources.entries();
    for (size_t index = 0; index < sources.size(); ++index) {
      const std::string& name = sources[index].name;
      if (name == "image") {
        ++image_count;
      } else if (name == "prompt_video") {
        ++video_count;
      } else if (name == "prompt_audio") {
        ++audio_count;
      } else if (name == "prompt_audio_in_video") {
        CHECK_GT(index, static_cast<size_t>(0));
        CHECK_EQ(sources[index - 1].name, "prompt_video")
            << "MiniMax-H3 embedded audio must follow its prompt_video";
        ++video_audio_count;
      } else {
        LOG(FATAL) << "MiniMax-H3 unsupported media source name: " << name;
      }
    }

    if (task_type_ == "fl2va") {
      CHECK_LE(image_count, static_cast<size_t>(2));
      CHECK_EQ(video_count, static_cast<size_t>(0));
      CHECK_EQ(audio_count, static_cast<size_t>(0));
      CHECK_EQ(video_audio_count, static_cast<size_t>(0));
      return;
    }
    CHECK_LE(image_count, static_cast<size_t>(9));
    CHECK_LE(video_count, static_cast<size_t>(1));
    CHECK_LE(audio_count, static_cast<size_t>(1));
    CHECK_LE(video_audio_count, video_count);
  }

  struct PackedLayout {
    torch::Tensor position_ids;
    torch::Tensor token_tags;
    torch::Tensor video_indices;
    torch::Tensor audio_indices;
    torch::Tensor text_indices;
    int64_t sequence_length;
    int64_t audio_start;
    int64_t video_start;
    int64_t num_condition_video_rows;
    int64_t num_condition_audio_rows = 0;
  };

  enum class ReferenceKind { IMAGE, VIDEO, AUDIO };

  struct ReferenceBlock {
    ReferenceKind kind;
    bool has_audio = false;
    int64_t latent_frames = 0;
    int64_t latent_height = 0;
    int64_t latent_width = 0;
    int64_t audio_latents = 0;
  };

  struct ReferenceData {
    std::vector<ReferenceBlock> blocks;
    torch::Tensor video_rows;
    torch::Tensor audio_rows;
  };

  MiniMaxH3TimestepLayout build_minimax_h3_timestep_layout(
      const PackedLayout& layout,
      const torch::Tensor& video_timestep,
      const torch::Tensor& audio_timestep) const {
    CHECK_EQ(video_timestep.numel(), 1);
    CHECK_EQ(audio_timestep.numel(), 1);
    CHECK_EQ(video_timestep.device(), audio_timestep.device());
    CHECK_EQ(video_timestep.scalar_type(), audio_timestep.scalar_type());
    CHECK_GE(layout.num_condition_video_rows, 0);
    CHECK_GE(layout.num_condition_audio_rows, 0);
    CHECK_LE(layout.num_condition_video_rows, layout.video_indices.numel());
    CHECK_LE(layout.num_condition_audio_rows, layout.audio_indices.numel());

    const bool has_condition = layout.num_condition_video_rows > 0 ||
                               layout.num_condition_audio_rows > 0;
    if (!has_condition) {
      CHECK_GE(layout.audio_start, 0);
      CHECK_GE(layout.video_start, layout.audio_start);
      CHECK_LE(layout.video_start, layout.sequence_length);
      const double video_value = video_timestep.item<double>();
      const double audio_value = audio_timestep.item<double>();
      torch::TensorOptions index_options = torch::TensorOptions()
                                               .dtype(torch::kLong)
                                               .device(video_timestep.device());
      if (video_value == audio_value) {
        return {video_timestep.reshape({1}),
                torch::zeros({layout.sequence_length}, index_options)};
      }

      const bool video_first = video_value < audio_value;
      torch::Tensor timesteps =
          video_first ? torch::stack({video_timestep, audio_timestep})
                      : torch::stack({audio_timestep, video_timestep});
      const int64_t video_index = video_first ? 0 : 1;
      const int64_t audio_index = video_first ? 1 : 0;
      torch::Tensor indices =
          torch::full({layout.sequence_length}, video_index, index_options);
      indices.slice(0, layout.audio_start, layout.video_start)
          .fill_(audio_index);
      return {timesteps, indices};
    }

    const int64_t sequence_length = layout.video_indices.numel() +
                                    layout.audio_indices.numel() +
                                    layout.text_indices.numel();
    CHECK_EQ(sequence_length, layout.sequence_length);
    torch::Tensor row_timesteps =
        torch::full({sequence_length},
                    video_timestep.item<float>(),
                    torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(video_timestep.device()));
    if (layout.num_condition_video_rows > 0) {
      row_timesteps.index_put_(
          {layout.video_indices.slice(0, 0, layout.num_condition_video_rows)},
          std::max(video_timestep.item<double>(), keyframe_noise_aug_));
    }
    row_timesteps.index_put_(
        {layout.audio_indices.slice(0, layout.num_condition_audio_rows)},
        audio_timestep.item<float>());
    if (layout.num_condition_audio_rows > 0) {
      row_timesteps.index_put_(
          {layout.audio_indices.slice(0, 0, layout.num_condition_audio_rows)},
          /*condition_audio_timestep=*/1.0);
    }
    std::tuple<torch::Tensor, torch::Tensor> unique =
        torch::_unique(row_timesteps,
                       /*sorted=*/true,
                       /*return_inverse=*/true);
    return {std::get<0>(unique), std::get<1>(unique).to(torch::kLong)};
  }

  void synchronize_tp_ranks() const {
    if (tp_group_ == nullptr || tp_group_->world_size() == 1) {
      return;
    }
    torch::Tensor barrier = torch::zeros(
        {1}, options_.dtype(torch::kFloat32).device(options_.device()));
    tp_group_->allreduce(barrier);
  }

  int64_t align_num_frames(int64_t num_frames) const {
    while (num_frames % video_encode_clip_length_ !=
           video_encode_latents_per_clip_) {
      ++num_frames;
    }
    return num_frames;
  }

  PackedLayout build_layout(
      const torch::Tensor& text_tags,
      int64_t latent_frames,
      int64_t latent_height,
      int64_t latent_width,
      int64_t audio_latents,
      const std::vector<std::string>& keyframe_anchors) const {
    const int64_t text_tokens = text_tags.numel();
    torch::Tensor cpu_text_tags = text_tags.to(torch::kCPU, torch::kLong);
    const int64_t rows_per_frame =
        latent_height / patch_h_ * (latent_width / patch_w_);
    const int64_t condition_rows =
        static_cast<int64_t>(keyframe_anchors.size()) * rows_per_frame;
    const int64_t audio_rows = audio_channels_ * audio_latents;
    const int64_t video_rows = latent_frames * rows_per_frame;
    const int64_t condition_start = text_tokens;
    const int64_t audio_start = condition_start + condition_rows;
    const int64_t video_start = audio_start + audio_rows;
    const int64_t sequence_length = video_start + video_rows;
    torch::TensorOptions double_options =
        torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor positions =
        torch::zeros({sequence_length, 3}, double_options);
    positions.slice(0, 0, text_tokens)
        .select(1, 0)
        .copy_(torch::arange(text_tokens, double_options));

    const double sqrt_area =
        std::sqrt(static_cast<double>(latent_height * latent_width));
    torch::Tensor height_grid =
        spatial_grid(latent_height, patch_h_, sqrt_area);
    torch::Tensor width_grid = spatial_grid(latent_width, patch_w_, sqrt_area);
    const int64_t grid_height = latent_height / patch_h_;
    const int64_t grid_width = latent_width / patch_w_;
    torch::Tensor frame_grid =
        torch::stack({height_grid.view({grid_height, 1})
                          .expand({grid_height, grid_width})
                          .reshape({-1}),
                      width_grid.view({1, grid_width})
                          .expand({grid_height, grid_width})
                          .reshape({-1})},
                     1);

    for (size_t index = 0; index < keyframe_anchors.size(); ++index) {
      double anchor_time = static_cast<double>(text_tokens);
      if (keyframe_anchors[index] == "last") {
        torch::Tensor spans =
            torch::ones({latent_frames}, double_options) * rope_frame_scale_;
        for (int64_t offset = 0; offset < video_encode_latents_per_clip_;
             ++offset) {
          const double frame_span =
              offset == 0 ? 1.0 : static_cast<double>(vae_temporal_ratio_);
          spans.slice(
              0, offset, latent_frames, video_encode_latents_per_clip_) *=
              frame_span;
        }
        anchor_time += spans.sum().item<double>() - rope_frame_scale_;
      } else {
        CHECK_EQ(keyframe_anchors[index], "first")
            << "MiniMax-H3 keyframe anchor must be first or last";
      }
      const int64_t row_start =
          condition_start + static_cast<int64_t>(index) * rows_per_frame;
      const int64_t row_end = row_start + rows_per_frame;
      positions.slice(0, row_start, row_end).select(1, 0).fill_(anchor_time);
      positions.slice(0, row_start, row_end).slice(1, 1).copy_(frame_grid);
    }

    positions.slice(0, audio_start, video_start)
        .select(1, 0)
        .copy_((static_cast<double>(text_tokens) +
                torch::arange(audio_latents, double_options))
                   .repeat({audio_channels_}));
    torch::Tensor audio_width = torch::cat(
        {torch::full(
             {audio_latents}, width_grid[0].item<double>(), double_options),
         torch::full({audio_latents},
                     width_grid[width_grid.size(0) - 1].item<double>(),
                     double_options)});
    positions.slice(0, audio_start, video_start)
        .select(1, 2)
        .copy_(audio_width);

    torch::Tensor video_positions =
        torch::empty({latent_frames, rows_per_frame, 3}, double_options);
    video_positions.select(2, 0).copy_(
        temporal_grid(latent_frames, static_cast<double>(text_tokens))
            .view({latent_frames, 1})
            .expand({latent_frames, rows_per_frame}));
    video_positions.slice(2, 1).copy_(
        frame_grid.view({1, rows_per_frame, 2})
            .expand({latent_frames, rows_per_frame, 2}));
    positions.slice(0, video_start)
        .copy_(video_positions.reshape({video_rows, 3}));

    torch::TensorOptions long_options =
        torch::TensorOptions().dtype(torch::kLong);
    torch::Tensor text_indices = torch::arange(text_tokens, long_options);
    torch::Tensor audio_indices =
        torch::arange(audio_start, video_start, long_options);
    torch::Tensor video_indices =
        condition_rows > 0
            ? torch::cat(
                  {torch::arange(condition_start, audio_start, long_options),
                   torch::arange(video_start, sequence_length, long_options)})
            : torch::arange(video_start, sequence_length, long_options);
    torch::Tensor tags = torch::empty({sequence_length}, long_options);
    tags.slice(0, 0, text_tokens).copy_(cpu_text_tags);
    tags.slice(0, audio_start, video_start).fill_(audio_tag_);
    if (condition_rows > 0) {
      tags.slice(0, condition_start, audio_start).fill_(video_tag_);
    }
    tags.slice(0, video_start).fill_(video_tag_);

    return {positions.to(options_.device()),
            tags.to(options_.device()),
            video_indices.to(options_.device()),
            audio_indices.to(options_.device()),
            text_indices.to(options_.device()),
            sequence_length,
            audio_start,
            video_start,
            condition_rows};
  }

  PackedLayout build_ref_layout(const torch::Tensor& text_tags,
                                const std::vector<ReferenceBlock>& references,
                                int64_t latent_frames,
                                int64_t latent_height,
                                int64_t latent_width,
                                int64_t audio_latents) const {
    const int64_t text_tokens = text_tags.numel();
    torch::Tensor cpu_text_tags = text_tags.to(torch::kCPU, torch::kLong);
    const int64_t target_rows_per_frame =
        latent_height / patch_h_ * (latent_width / patch_w_);
    const int64_t target_video_rows = latent_frames * target_rows_per_frame;
    const int64_t target_audio_rows = audio_latents * audio_channels_;
    int64_t reference_video_rows = 0;
    int64_t reference_audio_rows = 0;
    for (const ReferenceBlock& reference : references) {
      if (reference.kind != ReferenceKind::AUDIO) {
        reference_video_rows += reference.latent_frames *
                                (reference.latent_height / patch_h_) *
                                (reference.latent_width / patch_w_);
      }
      reference_audio_rows += reference.audio_latents * audio_channels_;
    }
    const int64_t sequence_length = text_tokens + reference_video_rows +
                                    reference_audio_rows + target_audio_rows +
                                    target_video_rows;
    torch::TensorOptions double_options =
        torch::TensorOptions().dtype(torch::kFloat64);
    torch::TensorOptions long_options =
        torch::TensorOptions().dtype(torch::kLong);
    torch::Tensor positions =
        torch::zeros({sequence_length, 3}, double_options);
    positions.slice(0, 0, text_tokens)
        .select(1, 0)
        .copy_(torch::arange(text_tokens, double_options));

    auto frame_grid = [&](int64_t height, int64_t width) {
      const double sqrt_area = std::sqrt(static_cast<double>(height * width));
      torch::Tensor height_grid = spatial_grid(height, patch_h_, sqrt_area);
      torch::Tensor width_grid = spatial_grid(width, patch_w_, sqrt_area);
      const int64_t grid_height = height / patch_h_;
      const int64_t grid_width = width / patch_w_;
      torch::Tensor grid = torch::stack({height_grid.view({grid_height, 1})
                                             .expand({grid_height, grid_width})
                                             .reshape({-1}),
                                         width_grid.view({1, grid_width})
                                             .expand({grid_height, grid_width})
                                             .reshape({-1})},
                                        1);
      return std::make_pair(grid, width_grid);
    };
    auto fill_audio_positions = [&](int64_t start,
                                    int64_t count,
                                    int64_t num_latents,
                                    double rotary_time,
                                    const torch::Tensor& width_grid) {
      if (count == 0) {
        return;
      }
      positions.slice(0, start, start + count)
          .select(1, 0)
          .copy_((rotary_time + torch::arange(num_latents, double_options))
                     .repeat({audio_channels_}));
      positions.slice(0, start, start + count)
          .select(1, 2)
          .copy_(torch::cat(
              {torch::full(
                   {num_latents}, width_grid[0].item<double>(), double_options),
               torch::full({num_latents},
                           width_grid[width_grid.size(0) - 1].item<double>(),
                           double_options)}));
    };

    std::pair<torch::Tensor, torch::Tensor> target_grid =
        frame_grid(latent_height, latent_width);
    std::vector<torch::Tensor> video_index_blocks;
    std::vector<torch::Tensor> audio_index_blocks;
    int64_t cursor = text_tokens;
    double rotary_time = static_cast<double>(text_tokens);
    for (const ReferenceBlock& reference : references) {
      if (reference.kind == ReferenceKind::IMAGE) {
        std::pair<torch::Tensor, torch::Tensor> grid =
            frame_grid(reference.latent_height, reference.latent_width);
        const int64_t rows = reference.latent_frames * grid.first.size(0);
        video_index_blocks.emplace_back(
            torch::arange(cursor, cursor + rows, long_options));
        positions.slice(0, cursor, cursor + rows)
            .select(1, 0)
            .fill_(rotary_time);
        positions.slice(0, cursor, cursor + rows)
            .slice(1, 1)
            .copy_(grid.first.repeat({reference.latent_frames, 1}));
        cursor += rows;
        rotary_time += 1.0;
      } else if (reference.kind == ReferenceKind::AUDIO) {
        const int64_t rows = reference.audio_latents * audio_channels_;
        audio_index_blocks.emplace_back(
            torch::arange(cursor, cursor + rows, long_options));
        fill_audio_positions(cursor,
                             rows,
                             reference.audio_latents,
                             rotary_time,
                             target_grid.second);
        cursor += rows;
        rotary_time += static_cast<double>(reference.audio_latents);
      } else {
        std::pair<torch::Tensor, torch::Tensor> grid =
            frame_grid(reference.latent_height, reference.latent_width);
        const int64_t audio_rows = reference.audio_latents * audio_channels_;
        const int64_t video_rows = reference.latent_frames * grid.first.size(0);
        if (audio_rows > 0) {
          audio_index_blocks.emplace_back(
              torch::arange(cursor, cursor + audio_rows, long_options));
        }
        video_index_blocks.emplace_back(
            torch::arange(cursor + audio_rows,
                          cursor + audio_rows + video_rows,
                          long_options));
        fill_audio_positions(cursor,
                             audio_rows,
                             reference.audio_latents,
                             rotary_time,
                             grid.second);
        const int64_t video_start = cursor + audio_rows;
        positions.slice(0, video_start, video_start + video_rows)
            .select(1, 0)
            .copy_(torch::repeat_interleave(
                temporal_grid(reference.latent_frames, rotary_time),
                grid.first.size(0)));
        positions.slice(0, video_start, video_start + video_rows)
            .slice(1, 1)
            .copy_(grid.first.repeat({reference.latent_frames, 1}));
        cursor += audio_rows + video_rows;
        double video_span = 0.0;
        for (int64_t frame = 0; frame < reference.latent_frames; ++frame) {
          const double frame_span =
              frame % video_encode_latents_per_clip_ == 0
                  ? 1.0
                  : static_cast<double>(vae_temporal_ratio_);
          video_span += rope_frame_scale_ * frame_span;
        }
        rotary_time +=
            std::max(static_cast<double>(reference.audio_latents), video_span);
      }
    }

    const int64_t audio_start = cursor;
    const int64_t video_start = audio_start + target_audio_rows;
    fill_audio_positions(audio_start,
                         target_audio_rows,
                         audio_latents,
                         rotary_time,
                         target_grid.second);
    positions.slice(0, video_start)
        .select(1, 0)
        .copy_(
            torch::repeat_interleave(temporal_grid(latent_frames, rotary_time),
                                     target_grid.first.size(0)));
    positions.slice(0, video_start)
        .slice(1, 1)
        .copy_(target_grid.first.repeat({latent_frames, 1}));
    video_index_blocks.emplace_back(
        torch::arange(video_start, sequence_length, long_options));
    audio_index_blocks.emplace_back(
        torch::arange(audio_start, video_start, long_options));
    torch::Tensor video_indices = torch::cat(video_index_blocks);
    torch::Tensor audio_indices = torch::cat(audio_index_blocks);
    torch::Tensor text_indices = torch::arange(text_tokens, long_options);
    torch::Tensor tags = torch::empty({sequence_length}, long_options);
    tags.index_put_({text_indices}, cpu_text_tags);
    tags.index_put_({audio_indices}, audio_tag_);
    tags.index_put_({video_indices}, video_tag_);
    return {positions.to(options_.device()),
            tags.to(options_.device()),
            video_indices.to(options_.device()),
            audio_indices.to(options_.device()),
            text_indices.to(options_.device()),
            sequence_length,
            audio_start,
            video_start,
            reference_video_rows,
            reference_audio_rows};
  }

  torch::Tensor spatial_grid(int64_t dimension,
                             int64_t patch,
                             double sqrt_area) const {
    const int64_t count = dimension / patch;
    const double ratio = static_cast<double>(dimension) / sqrt_area;
    const double left = (1.0 - ratio) / 2.0;
    torch::TensorOptions options =
        torch::TensorOptions().dtype(torch::kFloat64);
    return (left + torch::arange(count, options) *
                       (ratio / static_cast<double>(count))) *
           static_cast<double>(vae_spatial_ratio_ * patch);
  }

  torch::Tensor temporal_grid(int64_t latent_frames, double origin) const {
    std::vector<double> values;
    values.reserve(static_cast<size_t>(latent_frames));
    double position = origin;
    for (int64_t frame = 0; frame < latent_frames; ++frame) {
      values.emplace_back(position);
      const double frame_span = frame % video_encode_latents_per_clip_ == 0
                                    ? 1.0
                                    : static_cast<double>(vae_temporal_ratio_);
      position += rope_frame_scale_ * frame_span;
    }
    return torch::tensor(values, torch::dtype(torch::kFloat64));
  }

  std::pair<torch::Tensor, torch::Tensor> prepare_noise(
      int64_t latent_frames,
      int64_t latent_height,
      int64_t latent_width,
      int64_t audio_latents,
      torch::Generator& generator) const {
    const int64_t video_values =
        video_latent_channels_ * latent_frames * latent_height * latent_width;
    const int64_t audio_values =
        audio_channels_ * audio_latent_channels_ * audio_latents;
    torch::Tensor noise = torch::randn(
        {video_values + audio_values},
        generator,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor video = noise.slice(0, 0, video_values)
                              .view({1,
                                     video_latent_channels_,
                                     latent_frames,
                                     latent_height,
                                     latent_width})
                              .to(options_.device());
    torch::Tensor audio =
        noise.slice(0, video_values)
            .view({audio_channels_, audio_latent_channels_, audio_latents})
            .to(options_.device());
    return {video, audio};
  }

  ReferenceData prepare_reference_data(const DiTForwardInput& input,
                                       int64_t num_frames,
                                       double video_fps,
                                       torch::Generator& generator) {
    ReferenceData result;
    std::vector<torch::Tensor> visual_latents;
    std::vector<torch::Tensor> audio_rows;
    std::vector<torch::Tensor> reference_audio_inputs;
    reference_audio_inputs.reserve(input.media_sources.size());
    const int64_t max_audio_samples =
        audio_processor_->max_reference_samples(num_frames, video_fps);

    const std::vector<MediaNamedTensor>& sources =
        input.media_sources.entries();
    for (size_t index = 0; index < sources.size(); ++index) {
      const MediaNamedTensor& source = sources[index];
      if (source.name == "prompt_audio_in_video") {
        continue;
      }
      if (source.name == "image") {
        const torch::Tensor& batched_image = source.tensor;
        CHECK_EQ(batched_image.dim(), 4);
        CHECK_EQ(batched_image.size(0), 1);
        const std::pair<int64_t, int64_t> size =
            visual_processor_->resolve_reference_image_size(
                batched_image.size(2), batched_image.size(3));
        result.blocks.push_back({ReferenceKind::IMAGE,
                                 false,
                                 1,
                                 size.first / vae_spatial_ratio_,
                                 size.second / vae_spatial_ratio_,
                                 0});
        if (is_tp_driver_) {
          torch::Tensor image = visual_processor_->prepare_reference_image(
              batched_image.select(0, 0), size.first, size.second);
          visual_latents.emplace_back(vae_->encode_reference_condition(
              image.to(options_.device(), torch::kUInt8)
                  .unsqueeze(0)
                  .unsqueeze(2)));
        }
        continue;
      }
      if (source.name == "prompt_video") {
        CHECK_EQ(source.tensor.dim(), 5);
        CHECK_EQ(source.tensor.size(0), 1);
        const double* prompt_video_fps =
            get_tensor_parameter<double>(source.parameters, "prompt_video_fps");
        CHECK(prompt_video_fps != nullptr)
            << "MiniMax-H3 reference video requires prompt_video_fps";
        CHECK(std::isfinite(*prompt_video_fps) && *prompt_video_fps > 0.0)
            << "MiniMax-H3 prompt_video_fps must be finite and positive";
        torch::Tensor video_input = source.tensor.select(0, 0);
        const int64_t normalized_frames =
            visual_processor_->normalized_video_frame_count(
                video_input.size(0), *prompt_video_fps, video_fps, num_frames);
        const int64_t encode_frames =
            visual_processor_->reference_video_encode_frames(normalized_frames);
        const std::pair<int64_t, int64_t> size =
            visual_processor_->resolve_reference_canvas(video_input.size(2),
                                                        video_input.size(3));
        const bool has_audio =
            index + 1 < sources.size() &&
            sources[index + 1].name == "prompt_audio_in_video";
        torch::Tensor soundtrack;
        int64_t reference_audio_latents = 0;
        if (has_audio) {
          const torch::Tensor& batched_soundtrack = sources[index + 1].tensor;
          CHECK_EQ(batched_soundtrack.dim(), 3);
          CHECK_EQ(batched_soundtrack.size(0), 1);
          soundtrack = batched_soundtrack.select(0, 0);
          const int64_t samples =
              std::min(soundtrack.size(1), max_audio_samples);
          reference_audio_latents =
              audio_processor_->reference_audio_latents(samples);
        }
        const int64_t reference_video_latent_frames =
            encode_frames == 1
                ? 1
                : (encode_frames + video_encode_clip_length_ - 1) /
                          video_encode_clip_length_ *
                          video_encode_latents_per_clip_ -
                      video_encode_token_drop_;
        result.blocks.push_back({ReferenceKind::VIDEO,
                                 has_audio,
                                 reference_video_latent_frames,
                                 size.first / vae_spatial_ratio_,
                                 size.second / vae_spatial_ratio_,
                                 reference_audio_latents});
        if (is_tp_driver_) {
          torch::Tensor video =
              visual_processor_->prepare_reference_video(video_input,
                                                         *prompt_video_fps,
                                                         video_fps,
                                                         num_frames,
                                                         size.first,
                                                         size.second);
          visual_latents.emplace_back(vae_->encode_reference_condition(
              video.permute({1, 0, 2, 3})
                  .unsqueeze(0)
                  .to(options_.device(), torch::kUInt8)));
          if (has_audio) {
            reference_audio_inputs.emplace_back(std::move(soundtrack));
          }
        }
        continue;
      }
      CHECK_EQ(source.name, "prompt_audio");
      CHECK_EQ(source.tensor.dim(), 3);
      CHECK_EQ(source.tensor.size(0), 1);
      torch::Tensor audio_input = source.tensor.select(0, 0);
      const int64_t samples = std::min(audio_input.size(1), max_audio_samples);
      const int64_t reference_audio_latents =
          audio_processor_->reference_audio_latents(samples);
      result.blocks.push_back(
          {ReferenceKind::AUDIO, true, 0, 0, 0, reference_audio_latents});
      if (is_tp_driver_) {
        reference_audio_inputs.emplace_back(std::move(audio_input));
      }
    }

    // Encode reference audio conditions.
    if (is_tp_driver_ && !reference_audio_inputs.empty()) {
      torch::Tensor mean =
          torch::tensor(
              audio_vae_->latents_mean(),
              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
              .view({1, 1, audio_latent_channels_});
      torch::Tensor std =
          torch::tensor(
              audio_vae_->latents_std(),
              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
              .view({1, 1, audio_latent_channels_});
      audio_rows.reserve(reference_audio_inputs.size());
      for (torch::Tensor waveform : reference_audio_inputs) {
        waveform =
            audio_processor_->trim_reference_audio(waveform, max_audio_samples)
                .to(options_.device(), torch::kFloat32);
        torch::Tensor latent = audio_vae_->encode_mode(waveform.unsqueeze(1))
                                   .to(torch::kCPU, torch::kFloat32)
                                   .transpose(1, 2);
        audio_rows.emplace_back(((latent.to(torch::kFloat32) - mean) / std)
                                    .reshape({1, -1, audio_latent_channels_})
                                    .contiguous()
                                    .to(options_.device()));
      }
    }

    std::vector<torch::Tensor> packed_visual_rows;
    size_t visual_index = 0;
    for (const ReferenceBlock& block : result.blocks) {
      if (block.kind == ReferenceKind::AUDIO) {
        continue;
      }
      const std::vector<int64_t> shape = {1,
                                          video_latent_channels_,
                                          block.latent_frames,
                                          block.latent_height,
                                          block.latent_width};
      torch::Tensor noise =
          torch::randn(
              shape,
              generator,
              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
              .to(options_.device());
      if (is_tp_driver_) {
        CHECK_EQ(visual_latents[visual_index].dim(), 5);
        for (int64_t dimension = 0; dimension < 5; ++dimension) {
          CHECK_EQ(visual_latents[visual_index].size(dimension),
                   shape[static_cast<size_t>(dimension)]);
        }
        torch::Tensor timestep = torch::full({1},
                                             keyframe_noise_aug_,
                                             torch::TensorOptions()
                                                 .dtype(torch::kFloat32)
                                                 .device(options_.device()));
        packed_visual_rows.emplace_back(patchify_video(scheduler_.scale_noise(
            visual_latents[visual_index], timestep, noise)));
      }
      ++visual_index;
    }
    const int64_t video_row_count = [&]() {
      int64_t rows = 0;
      for (const ReferenceBlock& block : result.blocks) {
        if (block.kind != ReferenceKind::AUDIO) {
          rows += block.latent_frames * (block.latent_height / patch_h_) *
                  (block.latent_width / patch_w_);
        }
      }
      return rows;
    }();
    const int64_t audio_row_count = [&]() {
      int64_t rows = 0;
      for (const ReferenceBlock& block : result.blocks) {
        rows += block.audio_latents * audio_channels_;
      }
      return rows;
    }();
    if (is_tp_driver_ && !packed_visual_rows.empty()) {
      result.video_rows = torch::cat(packed_visual_rows, 1).contiguous();
    } else {
      result.video_rows = torch::empty(
          {1, video_row_count, video_latent_channels_ * patch_h_ * patch_w_},
          torch::TensorOptions()
              .dtype(torch::kFloat32)
              .device(options_.device()));
    }
    if (is_tp_driver_ && !audio_rows.empty()) {
      result.audio_rows = torch::cat(audio_rows, 1).contiguous();
    } else {
      result.audio_rows =
          torch::empty({1, audio_row_count, audio_latent_channels_},
                       torch::TensorOptions()
                           .dtype(torch::kFloat32)
                           .device(options_.device()));
    }
    if (tp_group_ != nullptr && tp_group_->world_size() > 1) {
      if (video_row_count > 0) {
        tp_group_->broadcast(result.video_rows, /*root_rank=*/0);
      }
      if (audio_row_count > 0) {
        tp_group_->broadcast(result.audio_rows, /*root_rank=*/0);
      }
    }
    CHECK_EQ(result.video_rows.size(1), video_row_count);
    CHECK_EQ(result.audio_rows.size(1), audio_row_count);
    return result;
  }

  torch::Tensor prepare_fl_frame_data(
      const DiTForwardInput& input,
      int64_t height,
      int64_t width,
      torch::Generator& generator,
      const std::vector<std::string>& keyframe_anchors) {
    const int64_t latent_height = height / vae_spatial_ratio_;
    const int64_t latent_width = width / vae_spatial_ratio_;
    const int64_t rows_per_frame =
        latent_height / patch_h_ * (latent_width / patch_w_);
    if (!is_tp_driver_) {
      for (size_t index = 0; index < keyframe_anchors.size(); ++index) {
        (void)torch::randn(
            {1, video_latent_channels_, 1, latent_height, latent_width},
            generator,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      }
      return torch::empty(
          {1,
           static_cast<int64_t>(keyframe_anchors.size()) * rows_per_frame,
           video_latent_channels_ * patch_h_ * patch_w_},
          options_.dtype(torch::kFloat32).device(options_.device()));
    }

    std::vector<torch::Tensor> keyframes;
    keyframes.reserve(keyframe_anchors.size());
    std::vector<torch::Tensor> image_sources;
    image_sources.reserve(/*new_cap=*/2);
    for (const MediaNamedTensor& source : input.media_sources.entries()) {
      if (source.name == "image") {
        image_sources.emplace_back(source.tensor);
      }
    }
    if (!image_sources.empty()) {
      keyframes.emplace_back(visual_processor_->prepare_keyframe_image(
          image_sources.front().select(0, 0),
          height,
          width,
          /*is_follower=*/false));
    }
    if (image_sources.size() == 2) {
      keyframes.emplace_back(visual_processor_->prepare_keyframe_image(
          image_sources.back().select(0, 0),
          height,
          width,
          /*is_follower=*/true));
    }
    CHECK_EQ(keyframes.size(), keyframe_anchors.size());

    std::vector<torch::Tensor> rows;
    rows.reserve(keyframes.size());
    for (const torch::Tensor& keyframe : keyframes) {
      torch::Tensor pixels = keyframe.to(options_.device(), torch::kUInt8)
                                 .unsqueeze(0)
                                 .unsqueeze(2);
      torch::Tensor condition = vae_->encode_keyframe_condition(pixels);
      torch::Tensor noise =
          torch::randn(
              condition.sizes().vec(),
              generator,
              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
              .to(options_.device());
      torch::Tensor timestep = torch::full({1},
                                           keyframe_noise_aug_,
                                           torch::TensorOptions()
                                               .dtype(torch::kFloat32)
                                               .device(options_.device()));
      torch::Tensor noised = scheduler_.scale_noise(condition, timestep, noise);
      rows.emplace_back(patchify_video(noised));
    }
    return torch::cat(rows, 1).contiguous();
  }

  torch::Tensor patchify_video(const torch::Tensor& latent) const {
    const int64_t batch_size = latent.size(0);
    const int64_t channels = latent.size(1);
    const int64_t frames = latent.size(2);
    const int64_t height = latent.size(3);
    const int64_t width = latent.size(4);
    return latent
        .reshape({batch_size,
                  channels,
                  frames,
                  height / patch_h_,
                  patch_h_,
                  width / patch_w_,
                  patch_w_})
        .permute({0, 2, 3, 5, 1, 4, 6})
        .reshape({batch_size,
                  frames * height / patch_h_ * width / patch_w_,
                  channels * patch_h_ * patch_w_})
        .contiguous();
  }

  torch::Tensor patchify_audio(const torch::Tensor& latent) const {
    return latent.permute({0, 2, 1})
        .reshape({1, -1, audio_latent_channels_})
        .contiguous();
  }

  torch::Tensor unpatchify_video(const torch::Tensor& rows,
                                 int64_t frames,
                                 int64_t height,
                                 int64_t width) const {
    return rows
        .reshape({1,
                  frames,
                  height / patch_h_,
                  width / patch_w_,
                  video_latent_channels_,
                  patch_h_,
                  patch_w_})
        .permute({0, 4, 1, 2, 5, 3, 6})
        .reshape({1, video_latent_channels_, frames, height, width})
        .contiguous();
  }

  torch::Tensor unpatchify_audio(const torch::Tensor& rows,
                                 int64_t audio_latents) const {
    return rows
        .reshape({audio_channels_, audio_latents, audio_latent_channels_})
        .permute({0, 2, 1})
        .contiguous();
  }

  int64_t audio_channels_ = 2;
  int64_t video_tag_ = 0;
  int64_t text_tag_ = 1;
  int64_t audio_tag_ = 2;
  double min_duration_seconds_ = 5.0;
  double max_duration_seconds_ = 15.0;
  double keyframe_noise_aug_ = 0.999;
  double rope_frame_scale_ = 5.0 / 3.0;

  static int64_t product_or_one(const std::vector<int64_t>& values) {
    int64_t product = 1;
    for (int64_t value : values) {
      product *= value;
    }
    return product;
  }

  torch::TensorOptions options_;
  ProcessGroup* tp_group_ = nullptr;
  bool is_tp_driver_ = true;
  std::string task_type_;
  int64_t patch_h_;
  int64_t patch_w_;
  int64_t video_latent_channels_;
  int64_t audio_latent_channels_;
  int64_t text_hidden_size_;
  int64_t vae_spatial_ratio_;
  int64_t vae_temporal_ratio_;
  int64_t video_encode_clip_length_;
  int64_t video_encode_latents_per_clip_;
  int64_t video_encode_token_drop_;
  MiniMaxH3VisualProcessor visual_processor_{nullptr};
  MiniMaxH3AudioProcessor audio_processor_{nullptr};
  MiniMaxH3Transformer3DModel transformer_{nullptr};
  AutoencoderKLMiniMaxH3 vae_{nullptr};
  AutoencoderKLMiniMaxH3Audio audio_vae_{nullptr};
  MiniMaxH3Scheduler scheduler_;
  MiniMaxH3Scheduler audio_scheduler_{/*shift=*/3.0};
};
TORCH_MODULE(MiniMaxH3Pipeline);

REGISTER_DIT_MODEL(MiniMaxH3ModularPipeline, MiniMaxH3Pipeline);

}  // namespace xllm
