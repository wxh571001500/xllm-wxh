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

#include "processors/qwen3_omni_prompt_processor.h"

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

namespace {

// Tracks, for one modality, the pad token string, the index of the next
// pending input, and the per-input sizes used to compute the expansion
// length.
struct ModalityIndexRef {
  uint32_t modality_index = 0;
  const std::string modality_token;
  // audio : [feat_length]
  // image or video : [grid_thw]
  const torch::Tensor* modality_size_ptr = nullptr;
  uint32_t modality_nums_ = 0;

  torch::Tensor get_modality_size() {
    CHECK(modality_index < modality_nums_)
        << "The index of " << modality_token
        << " modality is out of range, have " << modality_nums_
        << " modality inputs "
        << "but try to access index " << modality_index;
    return (*modality_size_ptr)[modality_index++];
  }

  const std::string& get_modality_token() { return modality_token; }

  ModalityIndexRef() = default;

  ModalityIndexRef(const std::string& modality_token,
                   const torch::Tensor* modality_size_ptr)
      : modality_token(modality_token), modality_size_ptr(modality_size_ptr) {
    if (modality_size_ptr->defined()) {
      modality_nums_ = modality_size_ptr->size(0);
    }
  }
};

}  // namespace

Qwen3OmniPromptProcessor::Qwen3OmniPromptProcessor(const ModelArgs& args) {
  merge_size_ = args.mm_image_merge_size();
  fps_ = args.mm_fps();
  temporal_patch_size_ = args.mm_temporal_patch_size();
  use_audio_in_video_ = args.mm_use_audio_in_video();
  position_id_per_seconds_ = args.mm_position_id_per_seconds();
  video_second_per_grid_ = temporal_patch_size_ / fps_;
  vision_start_token_id_ = args.vision_start_token_id();
  vision_end_token_id_ = args.vision_end_token_id();
  image_token_id_ = args.image_token_id();
  video_token_id_ = args.video_token_id();
  audio_token_id_ = args.audio_token_id();
  audio_start_token_id_ = args.audio_start_token_id();
  audio_end_token_id_ = args.audio_end_token_id();
}

bool Qwen3OmniPromptProcessor::process(std::string& prompt,
                                       const MMData& mm_data) {
  torch::Tensor image_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("image_grid_thw")) {
    image_grid_thw = res.value();
  }

  torch::Tensor video_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("video_grid_thw")) {
    video_grid_thw = res.value();
  }

  torch::Tensor feat_length;
  if (auto res = mm_data.get<torch::Tensor>("feat_length")) {
    feat_length = res.value();
  }

  if (!image_grid_thw.defined() && !video_grid_thw.defined() &&
      !feat_length.defined()) {
    return true;
  }

  auto merge_length = merge_size_ * merge_size_;

  uint32_t total_audio_token = 0;
  if (feat_length.defined()) {
    auto count = feat_length.sizes()[0];
    for (size_t idx = 0; idx < count; ++idx)
      total_audio_token += feat_length[idx].item<int>();
  }

  uint32_t total_image_token = 0;
  if (image_grid_thw.defined()) {
    auto count = image_grid_thw.sizes()[0];
    for (size_t idx = 0; idx < count; ++idx)
      total_image_token +=
          image_grid_thw[idx].prod().item<int>() / merge_length;
  }

  uint32_t total_video_token = 0;
  if (video_grid_thw.defined()) {
    auto count = video_grid_thw.sizes()[0];
    for (size_t idx = 0; idx < count; ++idx) {
      total_video_token +=
          video_grid_thw[idx].prod().item<int>() / merge_length;
    }
  }

  uint32_t total_token_len = total_image_token * image_token_.size() +
                             total_video_token * video_token_.size() +
                             total_audio_token * audio_token_.size();
  std::string data;
  data.reserve(prompt.size() + total_token_len);

  std::unordered_map<TokenType, ModalityIndexRef> modality_index_map;
  modality_index_map.emplace(TokenType::AUDIO,
                             ModalityIndexRef(audio_token_, &feat_length));
  modality_index_map.emplace(TokenType::IMAGE,
                             ModalityIndexRef(image_token_, &image_grid_thw));
  modality_index_map.emplace(TokenType::VIDEO,
                             ModalityIndexRef(video_token_, &video_grid_thw));

  size_t begin = 0;
  auto pair = find_special_token(prompt, begin);
  while (pair.second != std::string::npos) {
    data.append(prompt, begin, pair.second - begin);

    auto& cur_modality = modality_index_map[pair.first];
    auto modality_size_ref = cur_modality.get_modality_size();
    auto modality_token_ref = cur_modality.get_modality_token();
    if (pair.first == TokenType::AUDIO) {
      // for audio
      auto token_num = modality_size_ref.item<int32_t>();
      while (token_num--) data.append(modality_token_ref);
    } else if (pair.first == TokenType::VIDEO && use_audio_in_video_) {
      // for audio in video
      auto& audio_modality = modality_index_map[TokenType::AUDIO];
      auto audio_size_ref = audio_modality.get_modality_size();
      auto audio_token_ref = audio_modality.get_modality_token();
      auto audio_token_indices =
          torch::arange(audio_size_ref.item<int32_t>()).to(torch::kInt32);

      int32_t T = modality_size_ref[0].item<int32_t>();
      int32_t H = modality_size_ref[1].item<int32_t>();
      int32_t W = modality_size_ref[2].item<int32_t>();

      int32_t height = H / merge_size_;
      int32_t width = W / merge_size_;

      auto video_token_indices_1d = torch::arange(T);
      auto video_token_indices = video_token_indices_1d.view({T, 1, 1});

      video_token_indices = video_token_indices.expand({T, height, width});

      video_token_indices = video_token_indices.reshape({-1});
      video_token_indices = video_token_indices * video_second_per_grid_ *
                            position_id_per_seconds_;
      auto video_indices_vec = video_token_indices.accessor<float, 1>();
      auto audio_indices_vec = audio_token_indices.accessor<int32_t, 1>();

      std::string placeholder_string = audio_bos_token_;

      size_t video_data_index = 0;
      size_t audio_data_index = 0;
      size_t video_len = video_indices_vec.size(0);
      size_t audio_len = audio_indices_vec.size(0);
      while (video_data_index < video_len && audio_data_index < audio_len) {
        if (video_indices_vec[video_data_index] <=
            audio_indices_vec[audio_data_index]) {
          placeholder_string.append(modality_token_ref);
          video_data_index++;
        } else {
          placeholder_string.append(audio_token_ref);
          audio_data_index++;
        }
      }

      if (video_data_index < video_len) {
        size_t remaining_video = video_len - video_data_index;
        for (size_t i = 0; i < remaining_video; ++i) {
          placeholder_string.append(modality_token_ref);
        }
      }

      if (audio_data_index < audio_len) {
        size_t remaining_audio = audio_len - audio_data_index;
        for (size_t i = 0; i < remaining_audio; ++i) {
          placeholder_string.append(audio_token_ref);
        }
      }

      placeholder_string.append(audio_eos_token_);
      data.append(placeholder_string);
    } else {
      // for image and video
      auto token_num = modality_size_ref.prod().item<int32_t>() / merge_length;
      while (token_num--) data.append(modality_token_ref);
    }
    begin = pair.second + modality_token_ref.size();
    pair = find_special_token(prompt, begin);
  }

  if (begin < prompt.size()) {
    data.append(prompt, begin, std::string::npos);
  }

  prompt = std::move(data);
  return true;
}

bool Qwen3OmniPromptProcessor::find_mm_spans(const std::vector<int32_t>& prompt,
                                             MMData& mm_data) {
  auto start = prompt.begin();
  uint32_t global_mm_index = 0;
  uint32_t offset = 0;
  uint32_t length = 0;
  auto& mm_items = mm_data.items<MMItemVec>();
  while (true) {
    auto vision_start_it =
        std::find(start, prompt.end(), vision_start_token_id_);
    auto vision_end_it = std::find(start, prompt.end(), vision_end_token_id_);
    auto audio_start_it = std::find(start, prompt.end(), audio_start_token_id_);
    auto audio_end_it = std::find(start, prompt.end(), audio_end_token_id_);
    // vision_start_it == audio_start_it when reach the end
    if (vision_start_it == audio_start_it) {
      break;
    }

    auto min_start_it = std::min(vision_start_it, audio_start_it);
    auto min_end_it = std::min(vision_end_it, audio_end_it);
    auto max_end_it = std::max(vision_end_it, audio_end_it);
    offset = std::distance(prompt.begin(), min_start_it);
    length = std::distance(min_start_it + 1, min_end_it);
    auto& item = mm_items[global_mm_index];

    // use_audio_in_video case, offset subtract the audio_start_token,
    // length subtract the audio_start_token and audio_end_token
    if (*min_start_it == vision_start_token_id_ &&
        (*(min_start_it + 1) == audio_start_token_id_)) {
      item.mutable_state().mutable_token_pos() = {
          static_cast<int32_t>(offset + 2), static_cast<int32_t>(length - 1)};
      std::vector<int32_t> audio_in_video_propmt(
          prompt.begin() + offset + 2,
          prompt.begin() + offset + 2 + length - 1);
      torch::Tensor audio_in_video_mask =
          torch::tensor(audio_in_video_propmt, torch::kInt32);
      item.add("audio_in_video_mask", audio_in_video_mask);
      // Every position in an audio-in-video span is a real multimodal token
      // (interleaved video/audio pads); the merged audio_in_video_embedding
      // fills the whole span, so the merge mask is all-true.
      item.mutable_state().mutable_mm_token_mask() = torch::ones(
          {static_cast<int64_t>(length - 1)},
          torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
      // audio_in_video case, vision end is always greater than audio_end
      min_end_it = max_end_it;
    } else {
      item.mutable_state().mutable_token_pos() = {
          static_cast<int32_t>(offset + 1), static_cast<int32_t>(length)};
      // For plain image / video / audio spans every position is a real
      // multimodal token of this item's modality.
      item.mutable_state().mutable_mm_token_mask() = torch::ones(
          {static_cast<int64_t>(length)},
          torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    }
    global_mm_index++;
    start = std::next(min_end_it);
  }
  return true;
}

std::pair<Qwen3OmniPromptProcessor::TokenType, size_t>
Qwen3OmniPromptProcessor::find_special_token(const std::string& prompt,
                                             size_t begin) {
  struct TokenInfo {
    const std::string& token;
    TokenType type;
    size_t pos;
  };

  std::vector<TokenInfo> tokens = {
      {image_token_, TokenType::IMAGE, std::string::npos},
      {video_token_, TokenType::VIDEO, std::string::npos},
      {audio_token_, TokenType::AUDIO, std::string::npos}};

  for (auto& token_info : tokens) {
    token_info.pos = prompt.find(token_info.token, begin);
  }

  auto earliest = std::min_element(
      tokens.begin(), tokens.end(), [](const TokenInfo& a, const TokenInfo& b) {
        if (a.pos == std::string::npos) {
          return false;
        }
        if (b.pos == std::string::npos) {
          return true;
        }
        return a.pos < b.pos;
      });

  if (earliest == tokens.end() || earliest->pos == std::string::npos) {
    return {TokenType::INVALID, std::string::npos};
  }

  return {earliest->type, earliest->pos};
}

}  // namespace xllm
