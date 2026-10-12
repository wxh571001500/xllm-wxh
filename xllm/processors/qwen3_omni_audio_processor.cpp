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

#include "processors/qwen3_omni_audio_processor.h"

#include <glog/logging.h>

#include <optional>

#include "processors/audio_utils.h"

namespace xllm {
namespace {

// Raw values of the mm_audio_padding_strategy config (HF PaddingStrategy).
constexpr int64_t kPaddingMaxLength = 2;

// Round `max_length` up to a multiple of `pad_to_multiple_of` (HF
// FeatureExtractor._truncate / _pad behavior); no-op when either is unset.
int64_t round_up_to_multiple(int64_t max_length, int64_t pad_to_multiple_of) {
  if (pad_to_multiple_of <= 0 || max_length % pad_to_multiple_of == 0) {
    return max_length;
  }
  return (max_length / pad_to_multiple_of + 1) * pad_to_multiple_of;
}

}  // namespace

Qwen3OmniAudioProcessor::Qwen3OmniAudioProcessor(const ModelArgs& args)
    : feature_size_(args.mm_audio_feature_size()),
      sampling_rate_(args.mm_audio_sampling_rate()),
      n_fft_(args.mm_audio_n_fft()),
      hop_length_(args.mm_audio_hop_length()),
      n_samples_(args.mm_audio_chunk_length() * args.mm_audio_sampling_rate()),
      max_length_(args.mm_audio_max_length()),
      pad_to_multiple_of_(args.mm_audio_pad_to_multiple_of()),
      padding_strategy_(args.mm_audio_padding_strategy()),
      padding_value_(args.mm_audio_padding_value()),
      dither_(args.mm_audio_dither()),
      truncation_(args.mm_audio_truncation()),
      do_normalize_(args.mm_audio_do_normalize()),
      return_attention_mask_(args.mm_audio_return_attention_mask()),
      padding_side_(args.mm_audio_padding_side()),
      options_(
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)) {
  CHECK_GT(feature_size_, 0);
  CHECK_GT(n_fft_, 0);
  CHECK_GT(hop_length_, 0);
  window_ = torch::hann_window(n_fft_, /*periodic=*/true, options_);
  // Same filterbank as HF whisper: slaney mel scale + slaney norm, 8kHz cap.
  mel_filters_ = audio_utils::mel_filter_bank(
      /*num_frequency_bins=*/1 + n_fft_ / 2,
      /*num_mel_filters=*/feature_size_,
      /*min_frequency=*/0.0,
      /*max_frequency=*/8000.0,
      /*sampling_rate=*/sampling_rate_,
      /*norm=*/"slaney",
      /*mel_scale=*/"slaney");
  LOG(INFO) << "Qwen3OmniAudioProcessor: feature_size=" << feature_size_
            << " n_fft=" << n_fft_ << " hop_length=" << hop_length_
            << " sampling_rate=" << sampling_rate_;
}

torch::Tensor Qwen3OmniAudioProcessor::extract_log_mel_features(
    const torch::Tensor& waveform) const {
  torch::Tensor wave = waveform.to(options_.device(), torch::kFloat32);
  if (dither_ != 0.0) {
    wave = wave + dither_ * torch::randn(wave.sizes(), wave.options());
  }
  torch::Tensor stft = torch::stft(wave,
                                   /*n_fft=*/n_fft_,
                                   /*hop_length=*/hop_length_,
                                   /*win_length=*/n_fft_,
                                   /*window=*/window_,
                                   /*center=*/true,
                                   /*pad_mode=*/"reflect",
                                   /*normalized=*/false,
                                   /*onesided=*/std::nullopt,
                                   /*return_complex=*/true);
  // HF whisper drops the last STFT frame before the power spectrum.
  torch::Tensor magnitudes = stft.slice(-1, 0, -1).abs().pow(2);
  torch::Tensor mel_spec =
      torch::matmul(mel_filters_.to(wave.options()).t(), magnitudes);
  torch::Tensor log_spec = torch::clamp(mel_spec, 1e-10).log10();
  // waveform is [1, T], so log_spec is [1, feature_size_, frames]: the
  // dynamic-range clamp takes the max over the whole clip (HF batched path).
  torch::Tensor max_val =
      log_spec.amax({2}, /*keepdim=*/true).amax({1}, /*keepdim=*/true);
  log_spec = torch::maximum(log_spec, max_val - 8.0);
  return (log_spec + 4.0) / 4.0;
}

bool Qwen3OmniAudioProcessor::process(const torch::Tensor& origin_audio,
                                      const AudioMetadata& /*metadata*/,
                                      MMDataItem& output_item) const {
  // mm_codec decodes audio as [channels, samples]; squeeze the mono case to a
  // plain 1D waveform so the hop-aligned trim below applies.
  torch::Tensor wave = origin_audio;
  if (wave.dim() == 2 && wave.size(0) == 1) {
    wave = wave.squeeze(0);
  }
  if (wave.dim() != 1) {
    LOG(ERROR) << "Only mono-channel audio is supported";
    return false;
  }

  // Align with transformers/qwen_omni_utils trim_to_hop_length: drop tail
  // samples that do not fill a complete hop, so the STFT tail frames (and the
  // audio encoder's last output tokens) match the reference.
  int64_t valid_length = wave.size(0) / hop_length_ * hop_length_;
  if (valid_length == 0) {
    LOG(ERROR) << "Audio is shorter than one hop (" << hop_length_
               << " samples)";
    return false;
  }
  wave = wave.slice(0, 0, valid_length);

  // HF pads a single clip to max_length (default: one chunk of n_samples_);
  // LONGEST padding is a no-op for a single clip.
  const int64_t max_length = round_up_to_multiple(
      max_length_ > 0 ? max_length_ : n_samples_, pad_to_multiple_of_);

  if (truncation_ && valid_length > max_length) {
    wave = wave.slice(0, 0, max_length);
    valid_length = max_length;
  }

  torch::Tensor padded = wave;
  int64_t padded_length = valid_length;
  bool pad_right = true;
  if (padding_strategy_ == kPaddingMaxLength && valid_length < max_length) {
    if (padding_side_ == "right" || padding_side_ == "left") {
      pad_right = (padding_side_ == "right");
      padded = torch::full({max_length}, padding_value_, wave.options());
      const int64_t offset = pad_right ? 0 : max_length - valid_length;
      padded.narrow(0, offset, valid_length).copy_(wave);
      padded_length = max_length;
    } else {
      LOG(ERROR) << "Invalid padding side: " << padding_side_;
      return false;
    }
  }

  if (do_normalize_) {
    // Zero-mean/unit-var over the valid region; the padding region is reset
    // to padding_value_ (HF WhisperFeatureExtractor zero_mean_unit_var_norm).
    // For left padding the valid region sits at the tail.
    const int64_t data_offset = pad_right ? 0 : padded_length - valid_length;
    torch::Tensor valid = padded.narrow(0, data_offset, valid_length);
    torch::Tensor mean_val = valid.mean();
    torch::Tensor var_val = valid.var(/*unbiased=*/false);
    padded = (padded - mean_val) / torch::sqrt(var_val + 1e-7f);
    if (valid_length < padded_length) {
      const int64_t pad_offset = pad_right ? valid_length : 0;
      padded.narrow(0, pad_offset, padded_length - valid_length) =
          padding_value_;
    }
  }

  // [1, feature_size_, frames] -> [frames, feature_size_].
  torch::Tensor frame_features =
      extract_log_mel_features(padded.unsqueeze(0))[0].permute({1, 0});

  torch::Tensor input_features;
  torch::Tensor feat_origin_lens;
  if (return_attention_mask_) {
    // 0/1 sample mask, resampled to one entry per hop == one per STFT frame.
    torch::Tensor mask = torch::zeros({padded_length}, torch::kInt32);
    const int64_t ones_offset = pad_right ? 0 : padded_length - valid_length;
    mask.narrow(0, ones_offset, valid_length).fill_(1);
    torch::Tensor rescaled_mask = mask.index(
        {torch::indexing::Slice(0, torch::indexing::None, hop_length_)});
    if (padded_length % hop_length_ != 0) {
      rescaled_mask = rescaled_mask.slice(0, 0, rescaled_mask.size(0) - 1);
    }
    feat_origin_lens =
        torch::tensor({rescaled_mask.sum().item<int64_t>()}, torch::kLong);
    input_features = frame_features.index({rescaled_mask.to(torch::kBool)});
  } else {
    feat_origin_lens = torch::tensor({frame_features.size(0)}, torch::kLong);
    input_features = frame_features;
  }

  torch::Tensor feat_length =
      audio_utils::get_feat_extract_output_lengths(feat_origin_lens);

  output_item.add<torch::Tensor>("input_features", input_features);
  output_item.add<torch::Tensor>("feat_length", feat_length);
  output_item.add<torch::Tensor>("feat_origin_lens", feat_origin_lens);
  return true;
}

}  // namespace xllm
