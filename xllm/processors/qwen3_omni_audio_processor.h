/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <cstdint>
#include <string>

#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"
#include "processors/audio_processor.h"

namespace xllm {

// Whisper-style log-mel audio frontend for the Qwen3-Omni-Thinker.
class Qwen3OmniAudioProcessor final : public AudioProcessor {
 public:
  explicit Qwen3OmniAudioProcessor(const ModelArgs& args);

  bool process(const torch::Tensor& origin_audio,
               const AudioMetadata& metadata,
               MMDataItem& output_item) const override;

 private:
  // STFT -> log-mel for a [1, T] waveform; returns
  // [1, feature_size_, frames] with the last STFT frame already dropped (HF
  // whisper semantics).
  torch::Tensor extract_log_mel_features(const torch::Tensor& waveform) const;

  int64_t feature_size_;        // mel bins, mm_audio_feature_size
  int64_t sampling_rate_;       // mm_audio_sampling_rate
  int64_t n_fft_;               // mm_audio_n_fft
  int64_t hop_length_;          // mm_audio_hop_length
  int64_t n_samples_;           // chunk_length * sampling_rate (pad target)
  int64_t max_length_;          // mm_audio_max_length, <= 0 means unset
  int64_t pad_to_multiple_of_;  // mm_audio_pad_to_multiple_of, <= 0: disabled
  // Raw mm_audio_padding_strategy value: 0 = do not pad, 1 = longest (no-op
  // for single clips), 2 = pad to max_length_ (default n_samples_).
  int64_t padding_strategy_;
  double padding_value_;        // mm_audio_padding_value
  double dither_;               // mm_audio_dither
  bool truncation_;             // mm_audio_truncation
  bool do_normalize_;           // mm_audio_do_normalize
  bool return_attention_mask_;  // mm_audio_return_attention_mask
  std::string padding_side_;    // mm_audio_padding_side

  torch::Tensor window_;          // [n_fft_] periodic Hann window
  torch::Tensor mel_filters_;     // [1 + n_fft_ / 2, feature_size_] slaney
  torch::TensorOptions options_;  // float32 CPU
};

}  // namespace xllm
