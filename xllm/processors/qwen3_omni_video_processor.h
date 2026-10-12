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

#pragma once

#include <torch/torch.h>

#include <cstdint>

#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_input.h"
#include "processors/video_processor.h"

namespace xllm {

class Qwen3OmniVideoProcessor final : public VideoProcessor {
 public:
  explicit Qwen3OmniVideoProcessor(const ModelArgs& args);

  bool process(const torch::Tensor& origin_video,
               const VideoMetadata& metadata,
               MMDataItem& output_item) const override;

 private:
  torch::Tensor sample_frames(const VideoMetadata& metadata) const;

  bool process_video(const torch::Tensor& origin_video,
                     VideoMetadata& metadata,
                     torch::Tensor& pixel_values,
                     torch::Tensor& thw) const;

 private:
  bool do_normalize_ = true;
  bool do_rescale_ = true;
  bool do_resize_ = true;

  torch::Tensor image_mean_;
  torch::Tensor image_std_;

  int32_t merge_size_ = 2;
  int32_t patch_size_ = 16;

  int32_t resample_ = 3;
  double rescale_factor_ = 0.00392156862745098;
  int32_t temporal_patch_size_ = 2;

  double fps_ = 2.0;
  int32_t frame_factor_ = 2;
  int32_t fps_min_frames_ = 4;
  int32_t fps_max_frames_ = 768;
  int32_t video_min_token_num_ = 128;
  int32_t video_max_token_num_ = 768;
  int64_t model_seq_len_ = 128000;
  int32_t hf_video_max_token_num_ = 768;
  int64_t video_max_pixels_ = 0;
};

}  // namespace xllm
