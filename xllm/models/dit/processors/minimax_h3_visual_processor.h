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
#include <utility>
#include <vector>

#include "framework/model_context.h"
#include "models/dit/processors/vae_image_processor.h"

namespace xllm {

struct MiniMaxH3VisualProcessorConfig {
  int64_t canvas_multiple;
};

class MiniMaxH3VisualProcessorImpl final : public torch::nn::Module {
 public:
  MiniMaxH3VisualProcessorImpl(const ModelContext& context,
                               MiniMaxH3VisualProcessorConfig config)
      : image_processor_(
            register_module("image_processor",
                            VAEImageProcessor(context,
                                              /*do_resize=*/true,
                                              /*do_normalize=*/false))),
        config_(config) {
    CHECK_GT(config_.canvas_multiple, 0);
    const ModelArgs& model_args = context.get_model_args();
    clip_length_ = model_args.clip_length();
    int64_t temporal_ratio = 1;
    for (int64_t factor : model_args.temporal_downsample_factors()) {
      CHECK_GT(factor, 0);
      temporal_ratio *= factor;
    }
    CHECK_GT(clip_length_, 0);
    latents_per_clip_ = (clip_length_ - 1) / temporal_ratio + 1;
    CHECK_GT(latents_per_clip_, 0);
    CHECK_LT(latents_per_clip_, clip_length_);
  }

  std::pair<int64_t, int64_t> resolve_reference_canvas(
      int64_t source_height,
      int64_t source_width) const {
    CHECK_GT(source_height, 0);
    CHECK_GT(source_width, 0);
    const double ratio =
        static_cast<double>(source_width) / static_cast<double>(source_height);
    CHECK_GE(ratio, 0.25) << "MiniMax-H3 references must be within 1:4 and 4:1";
    CHECK_LE(ratio, 4.0) << "MiniMax-H3 references must be within 1:4 and 4:1";

    double width =
        ratio >= 1.0 ? static_cast<double>(reference_canvas_short_edge_) * ratio
                     : static_cast<double>(reference_canvas_short_edge_);
    double height =
        ratio >= 1.0
            ? static_cast<double>(reference_canvas_short_edge_)
            : static_cast<double>(reference_canvas_short_edge_) / ratio;
    const double area = width * height;
    if (area > reference_canvas_max_pixels_) {
      const double scale = std::sqrt(reference_canvas_max_pixels_ / area);
      width *= scale;
      height *= scale;
    }
    return {round_to_multiple(height, config_.canvas_multiple),
            round_to_multiple(width, config_.canvas_multiple)};
  }

  std::pair<int64_t, int64_t> resolve_reference_image_size(
      int64_t source_height,
      int64_t source_width) const {
    CHECK_GT(source_height, 0);
    CHECK_GT(source_width, 0);
    CHECK_LE(source_width, 4 * source_height)
        << "MiniMax-H3 reference images must be within 1:4 and 4:1";
    CHECK_LE(source_height, 4 * source_width)
        << "MiniMax-H3 reference images must be within 1:4 and 4:1";
    const double scale =
        static_cast<double>(reference_image_short_edge_) /
        static_cast<double>(std::min(source_height, source_width));
    return {round_to_multiple(source_height * scale, config_.canvas_multiple),
            round_to_multiple(source_width * scale, config_.canvas_multiple)};
  }

  int64_t normalized_video_frame_count(int64_t source_frames,
                                       double source_fps,
                                       double target_fps,
                                       int64_t target_frames) const {
    CHECK_GT(source_frames, 0);
    CHECK_GT(source_fps, 0.0);
    CHECK_GT(target_fps, 0.0);
    CHECK_GT(target_frames, 0);
    int64_t frames = source_frames;
    if (source_fps != target_fps) {
      frames = static_cast<int64_t>(
          std::floor(source_frames * target_fps / source_fps + 0.5));
    }
    return std::min(frames, target_frames);
  }

  int64_t reference_video_encode_frames(int64_t normalized_frames) const {
    CHECK_GT(normalized_frames, 0);
    const int64_t snapped =
        std::max<int64_t>(
            1, (normalized_frames - latents_per_clip_) / clip_length_) *
            clip_length_ +
        latents_per_clip_;
    return std::min(normalized_frames, snapped);
  }

  torch::Tensor prepare_reference_image(torch::Tensor image,
                                        int64_t target_height,
                                        int64_t target_width) {
    CHECK_EQ(image.dim(), 3) << "MiniMax-H3 reference image must be [C,H,W]";
    CHECK_EQ(image.size(0), 3);
    CHECK_GT(target_height, 0);
    CHECK_GT(target_width, 0);
    image = image.to(torch::kCPU, torch::kUInt8).contiguous();
    if (image.size(1) == target_height && image.size(2) == target_width) {
      return image;
    }
    return image_processor_
        ->resize(image, target_height, target_width, "lanczos")
        .to(torch::kCPU, torch::kUInt8)
        .contiguous();
  }

  torch::Tensor prepare_reference_video(torch::Tensor video,
                                        double source_fps,
                                        double target_fps,
                                        int64_t target_frames,
                                        int64_t target_height,
                                        int64_t target_width) {
    CHECK_EQ(video.dim(), 4) << "MiniMax-H3 reference video must be [T,C,H,W]";
    CHECK_EQ(video.size(1), 3);
    CHECK_GT(video.size(0), 0);
    CHECK_GT(source_fps, 0.0);
    CHECK_GT(target_fps, 0.0);
    CHECK_GT(target_frames, 0);
    CHECK_GT(target_height, 0);
    CHECK_GT(target_width, 0);
    video = video.to(torch::kCPU, torch::kUInt8);
    if (source_fps != target_fps) {
      const double scale = target_fps / source_fps;
      const int64_t final_slot = static_cast<int64_t>(
          std::floor(static_cast<double>(video.size(0)) * scale + 0.5));
      std::vector<int64_t> indices;
      indices.reserve(static_cast<size_t>(final_slot));
      for (int64_t frame = 0; frame < video.size(0); ++frame) {
        const int64_t start = static_cast<int64_t>(
            std::floor(static_cast<double>(frame) * scale + 0.5));
        const int64_t end =
            frame + 1 < video.size(0)
                ? static_cast<int64_t>(
                      std::floor(static_cast<double>(frame + 1) * scale + 0.5))
                : final_slot;
        for (int64_t slot = start; slot < end; ++slot) {
          indices.emplace_back(frame);
        }
      }
      CHECK(!indices.empty())
          << "MiniMax-H3 reference video is empty after FPS resampling";
      video = video.index_select(
          0, torch::tensor(indices, torch::dtype(torch::kLong)));
    }

    video = video.slice(0, 0, std::min(video.size(0), target_frames));
    const int64_t encode_frames = reference_video_encode_frames(video.size(0));
    video = video.slice(0, 0, encode_frames);
    if (video.size(2) == target_height && video.size(3) == target_width) {
      return video.contiguous();
    }

    std::vector<torch::Tensor> frames;
    frames.reserve(static_cast<size_t>(video.size(0)));
    for (int64_t frame = 0; frame < video.size(0); ++frame) {
      frames.emplace_back(
          prepare_reference_image(video[frame], target_height, target_width));
    }
    return torch::stack(frames).contiguous();
  }

  torch::Tensor prepare_keyframe_image(torch::Tensor image,
                                       int64_t height,
                                       int64_t width,
                                       bool is_follower) {
    CHECK_EQ(image.dim(), 3) << "MiniMax-H3 keyframe image must be [C,H,W]";
    CHECK_EQ(image.size(0), 3);
    image = image.unsqueeze(0).to(torch::kFloat32);
    const int64_t source_height = image.size(2);
    const int64_t source_width = image.size(3);
    if (source_height == height && source_width == width) {
      return image.squeeze(0).to(torch::kUInt8).contiguous();
    }
    if (!is_follower) {
      return image_processor_->resize(image, height, width, "bilinear")
          .squeeze(0)
          .to(torch::kUInt8)
          .contiguous();
    }

    const double scale = std::max(
        static_cast<double>(width) / static_cast<double>(source_width),
        static_cast<double>(height) / static_cast<double>(source_height));
    const int64_t resized_width =
        std::max(width,
                 static_cast<int64_t>(
                     std::llround(static_cast<double>(source_width) * scale)));
    const int64_t resized_height =
        std::max(height,
                 static_cast<int64_t>(
                     std::llround(static_cast<double>(source_height) * scale)));
    torch::Tensor resized = image_processor_->resize(
        image, resized_height, resized_width, "bilinear");
    const int64_t left = std::max<int64_t>(0, (resized_width - width) / 2);
    const int64_t top = std::max<int64_t>(0, (resized_height - height) / 2);
    return resized.slice(2, top, top + height)
        .slice(3, left, left + width)
        .squeeze(0)
        .to(torch::kUInt8)
        .contiguous();
  }

 private:
  int64_t reference_image_short_edge_ = 2048;
  int64_t reference_canvas_short_edge_ = 768;
  double reference_canvas_max_pixels_ = 768.0 * 1344.0;

  static int64_t round_to_multiple(double value, int64_t multiple) {
    return std::max<int64_t>(
        multiple,
        static_cast<int64_t>(std::nearbyint(value / multiple)) * multiple);
  }

  VAEImageProcessor image_processor_{nullptr};
  MiniMaxH3VisualProcessorConfig config_;
  int64_t clip_length_ = 0;
  int64_t latents_per_clip_ = 0;
};
TORCH_MODULE(MiniMaxH3VisualProcessor);

}  // namespace xllm
