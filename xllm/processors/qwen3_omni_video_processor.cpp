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

#include "processors/qwen3_omni_video_processor.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>

#include "core/framework/config/model_config.h"
#include "processors/transforms.h"

namespace xllm {

namespace {

using Size = std::pair<int32_t, int32_t>;

std::optional<Size> smart_resize(int32_t height,
                                 int32_t width,
                                 int32_t factor,
                                 int64_t min_pixels,
                                 int64_t max_pixels) {
  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200.0) {
    LOG(ERROR) << "Absolute aspect ratio must be smaller than 200, height: "
               << height << ", width: " << width;
    return std::nullopt;
  }

  auto round_by = [](int32_t n, int32_t f) {
    return static_cast<int32_t>(std::rint(n / static_cast<double>(f))) * f;
  };
  auto floor_by = [](double n, int32_t f) {
    return static_cast<int32_t>(std::floor(n / f)) * f;
  };
  auto ceil_by = [](double n, int32_t f) {
    return static_cast<int32_t>(std::ceil(n / f)) * f;
  };

  int32_t h_bar = std::max(factor, round_by(height, factor));
  int32_t w_bar = std::max(factor, round_by(width, factor));
  const int64_t pixels = static_cast<int64_t>(h_bar) * w_bar;

  if (pixels > max_pixels) {
    const double beta =
        std::sqrt((static_cast<double>(height) * width) / max_pixels);
    h_bar = floor_by(height / beta, factor);
    w_bar = floor_by(width / beta, factor);
  } else if (pixels < min_pixels) {
    const double beta =
        std::sqrt(min_pixels / (static_cast<double>(height) * width));
    h_bar = ceil_by(height * beta, factor);
    w_bar = ceil_by(width * beta, factor);
  }

  return std::make_pair(h_bar, w_bar);
}

}  // namespace

Qwen3OmniVideoProcessor::Qwen3OmniVideoProcessor(const ModelArgs& args) {
  image_mean_ = torch::tensor(args.mm_image_normalize_mean(),
                              torch::dtype(torch::kFloat32));
  image_std_ = torch::tensor(args.mm_image_normalize_std(),
                             torch::dtype(torch::kFloat32));
  patch_size_ = args.mm_image_patch_size();
  temporal_patch_size_ = args.mm_image_temporal_patch_size();
  merge_size_ = args.mm_image_merge_size();
  video_max_pixels_ = args.mm_video_max_pixels();

  const ModelConfig& model_config = ModelConfig::get_instance();
  fps_min_frames_ = model_config.fps_min_frames();
  fps_max_frames_ = model_config.fps_max_frames();
  if (model_config.video_max_token_num() > 0) {
    video_max_token_num_ = model_config.video_max_token_num();
    video_max_pixels_ = 0;
  }
}

torch::Tensor Qwen3OmniVideoProcessor::sample_frames(
    const VideoMetadata& metadata) const {
  const int32_t total_frames = metadata.total_num_frames;
  CHECK_GT(total_frames, 0) << "video metadata carries no total_num_frames";
  CHECK_GT(metadata.fps, 0.0)
      << "Asked to sample `fps` frames per second but no video metadata was "
         "provided which is required when sampling with `fps`.";

  // Mirrors HF qwen_omni_utils.smart_nframes: clamp the fps-driven frame count
  // into [min_frames, max_frames, total_frames], then floor to frame_factor_.
  auto ceil_by = [](double n, int32_t f) {
    return static_cast<int32_t>(std::ceil(n / f)) * f;
  };
  auto floor_by = [](double n, int32_t f) {
    return static_cast<int32_t>(std::floor(n / f)) * f;
  };
  const int32_t min_frames = ceil_by(fps_min_frames_, frame_factor_);
  const int32_t max_frames =
      floor_by(std::min(fps_max_frames_, total_frames), frame_factor_);
  double nframes = total_frames / metadata.fps * fps_;
  nframes =
      std::min(std::min(std::max(nframes, static_cast<double>(min_frames)),
                        static_cast<double>(max_frames)),
               static_cast<double>(total_frames));
  const int32_t result = floor_by(nframes, frame_factor_);
  CHECK(result >= frame_factor_ && result <= total_frames)
      << "nframes should be in interval [" << frame_factor_ << ", "
      << total_frames << "], but got " << result << ".";

  auto lin = torch::linspace(0.0,
                             total_frames - 1,
                             result,
                             torch::TensorOptions().dtype(torch::kFloat32));
  auto idx = torch::round(lin).to(torch::kLong);
  idx = torch::clamp(idx, 0, total_frames - 1);
  return idx;
}

bool Qwen3OmniVideoProcessor::process(const torch::Tensor& origin_video,
                                      const VideoMetadata& metadata,
                                      MMDataItem& output_item) const {
  torch::Tensor pixel_values;
  torch::Tensor thw;
  VideoMetadata output_metadata = metadata;
  if (!process_video(origin_video, output_metadata, pixel_values, thw)) {
    return false;
  }

  double fps = output_metadata.sampled_fps > 0.0 ? output_metadata.sampled_fps
                                                 : output_metadata.fps;
  double seconds_per_grid = static_cast<double>(temporal_patch_size_) / fps;
  torch::Tensor second_per_grid_ts = torch::tensor(
      {seconds_per_grid}, torch::TensorOptions().dtype(torch::kFloat32));
  output_item = MMDataItem(MMType::VIDEO,
                           MMDict{{"pixel_values_videos", pixel_values},
                                  {"video_grid_thw", thw},
                                  {"second_per_grid_ts", second_per_grid_ts}},
                           output_metadata);
  return true;
}

bool Qwen3OmniVideoProcessor::process_video(const torch::Tensor& origin_video,
                                            VideoMetadata& metadata,
                                            torch::Tensor& pixel_values,
                                            torch::Tensor& thw) const {
  if (origin_video.dim() != 4) {
    LOG(FATAL) << "video must be TCHW";
  }

  torch::Tensor indices = sample_frames(metadata);
  auto video = origin_video.index_select(/*dim=*/0, indices);
  const int64_t sampled_total_frames = video.size(0);

  metadata.frame_indices = indices;
  metadata.timestamps.clear();
  metadata.timestamps.reserve(static_cast<size_t>(sampled_total_frames));
  double fps_for_ts = (metadata.fps > 0.0) ? metadata.fps : 24.0;
  for (int64_t i = 0; i < sampled_total_frames; ++i) {
    int64_t frame_idx = metadata.frame_indices[i].item<int64_t>();
    metadata.timestamps.push_back(static_cast<double>(frame_idx) / fps_for_ts);
  }

  if (metadata.total_num_frames > 0 && metadata.fps > 0.0) {
    metadata.sampled_fps = double(sampled_total_frames) /
                           double(metadata.total_num_frames) * metadata.fps;
  } else {
    metadata.sampled_fps = fps_for_ts;
  }

  auto shape = video.sizes();
  auto channel = shape[1];
  auto resized_height = shape[2];
  auto resized_width = shape[3];

  const int32_t factor = patch_size_ * merge_size_;
  if (do_resize_) {
    const int64_t min_pixels =
        static_cast<int64_t>(video_min_token_num_) * factor * factor;
    const int64_t frame_max_pixels =
        video_max_pixels_ > 0
            ? video_max_pixels_
            : static_cast<int64_t>(video_max_token_num_) * factor * factor;
    const double total_pixels =
        static_cast<double>(model_seq_len_) * factor * factor * 0.9;
    int64_t max_pixels = static_cast<int64_t>(
        std::max(std::min(static_cast<double>(frame_max_pixels),
                          total_pixels / sampled_total_frames * frame_factor_),
                 static_cast<double>(static_cast<int64_t>(min_pixels * 1.05))));
    auto size = smart_resize(static_cast<int32_t>(resized_height),
                             static_cast<int32_t>(resized_width),
                             factor,
                             min_pixels,
                             max_pixels);
    if (!size) {
      return false;
    }
    std::tie(resized_height, resized_width) = *size;
  }

  // qwen_omni_utils first resizes the uint8 frames.
  auto out_video = video;
  if (do_resize_) {
    out_video = transforms::resize(video.to(torch::kFloat32),
                                   {resized_height, resized_width},
                                   resample_,
                                   true)
                    .round_()
                    .clamp_(0, 255)
                    .to(torch::kUInt8);
  }

  // Transformers applies a second resize with its fixed video token budget.
  if (do_resize_) {
    const int64_t stage2_min_pixels =
        static_cast<int64_t>(video_min_token_num_) * factor * factor;
    const int64_t stage2_max_pixels =
        static_cast<int64_t>(hf_video_max_token_num_) * factor * factor;
    auto size2 = smart_resize(static_cast<int32_t>(resized_height),
                              static_cast<int32_t>(resized_width),
                              factor,
                              stage2_min_pixels,
                              stage2_max_pixels);
    if (!size2) {
      return false;
    }
    if (size2->first != resized_height || size2->second != resized_width) {
      resized_height = size2->first;
      resized_width = size2->second;
      out_video = transforms::resize(out_video.to(torch::kFloat32),
                                     {resized_height, resized_width},
                                     resample_,
                                     true)
                      .round_()
                      .clamp_(0, 255)
                      .to(torch::kUInt8);
    }
  }

  out_video = out_video.to(torch::kFloat32);
  if (do_rescale_) {
    out_video = transforms::rescale(out_video, rescale_factor_);
  }
  if (do_normalize_) {
    out_video = transforms::normalize(out_video, image_mean_, image_std_);
  }

  auto pad_t =
      (temporal_patch_size_ - (out_video.size(0) % temporal_patch_size_)) %
      temporal_patch_size_;
  if (pad_t != 0) {
    auto last = out_video.index({out_video.size(0) - 1})
                    .unsqueeze(0)
                    .repeat({pad_t, 1, 1, 1});
    out_video = torch::cat({out_video, last}, 0);
  }

  shape = out_video.sizes();
  auto grid_h = resized_height / patch_size_;
  auto grid_w = resized_width / patch_size_;
  auto grid_t = shape[0] / temporal_patch_size_;

  out_video = out_video.contiguous();

  auto patches = out_video.view({grid_t,
                                 temporal_patch_size_,
                                 channel,
                                 grid_h / merge_size_,
                                 merge_size_,
                                 patch_size_,
                                 grid_w / merge_size_,
                                 merge_size_,
                                 patch_size_});

  patches = patches.permute({0, 3, 6, 4, 7, 2, 1, 5, 8});
  patches = patches.reshape(
      {grid_t * grid_h * grid_w,
       channel * temporal_patch_size_ * patch_size_ * patch_size_});

  pixel_values = patches;
  thw = torch::tensor({grid_t, grid_h, grid_w}).clone().reshape({-1, 3});

  return true;
}

}  // namespace xllm
