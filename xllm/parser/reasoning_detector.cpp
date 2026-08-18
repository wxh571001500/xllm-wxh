/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "reasoning_detector.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"

namespace xllm {
namespace {
std::string absl_trim(absl::string_view str) {
  absl::string_view trimmed = absl::StripAsciiWhitespace(str);
  return std::string(trimmed);
}

size_t partial_token_overlap(absl::string_view text,
                             absl::string_view token) {
  const size_t maximum = std::min(text.size(), token.size() - 1);
  for (size_t overlap = maximum; overlap > 0; --overlap) {
    if (text.substr(text.size() - overlap) == token.substr(0, overlap)) {
      return overlap;
    }
  }
  return 0;
}
}  // namespace

ReasoningDetector::ReasoningDetector(const std::string& think_start_token,
                                     const std::string& think_end_token,
                                     bool force_reasoning,
                                     bool stream_reasoning,
                                     bool trim_output)
    : think_start_token_(think_start_token),
      think_end_token_(think_end_token),
      in_reasoning_(force_reasoning),
      stream_reasoning_(stream_reasoning),
      trim_output_(trim_output) {}

ReasoningResult ReasoningDetector::detect_and_parse(std::string& text) {
  bool in_reasoning =
      in_reasoning_ || absl::StrContains(text, think_start_token_);

  if (!in_reasoning) {
    return ReasoningResult(text, std::nullopt);
  }

  std::string processed_text =
      absl::StrReplaceAll(text, {{think_start_token_, ""}});
  if (trim_output_) {
    processed_text = absl_trim(processed_text);
  }

  if (!absl::StrContains(processed_text, think_end_token_)) {
    return ReasoningResult(std::nullopt, processed_text);
  }

  std::vector<absl::string_view> parts =
      absl::StrSplit(processed_text, absl::MaxSplits(think_end_token_, 1));

  std::string reasoning_text = std::string(parts[0]);
  std::string normal_text = parts.size() > 1 ? std::string(parts[1]) : "";
  if (trim_output_) {
    normal_text = absl_trim(normal_text);
  }

  return ReasoningResult(normal_text, reasoning_text);
}

ReasoningResult ReasoningDetector::parse_streaming_increment(
    std::string& new_text) {
  buffer_.append(new_text.data(), new_text.size());
  std::string current_text = buffer_;

  // If the current text is a prefix of the think token, keep buffering
  bool is_start_prefix = absl::StartsWith(think_start_token_, current_text) &&
                         (think_start_token_ != current_text);
  bool is_end_prefix = absl::StartsWith(think_end_token_, current_text) &&
                       (think_end_token_ != current_text);

  if (is_start_prefix || is_end_prefix) {
    return ReasoningResult();
  }

  // Strip `<think>` token if present
  if (!stripped_think_start_ &&
      absl::StrContains(current_text, think_start_token_)) {
    absl::StrReplaceAll(
        {{absl::string_view(think_start_token_), absl::string_view()}},
        &current_text);
    stripped_think_start_ = true;
    in_reasoning_ = true;
  }

  // Handle end of reasoning block
  if (in_reasoning_ && absl::StrContains(current_text, think_end_token_)) {
    std::vector<absl::string_view> parts =
        absl::StrSplit(current_text, absl::MaxSplits(think_end_token_, 1));

    std::string reasoning_text = std::string(parts[0]);
    std::string normal_text = parts.size() > 1 ? std::string(parts[1]) : "";
    if (trim_output_) {
      normal_text = absl_trim(normal_text);
    }

    buffer_.clear();
    in_reasoning_ = false;

    return ReasoningResult(normal_text, reasoning_text);
  }

  // Continue with reasoning content
  if (in_reasoning_) {
    if (stream_reasoning_) {
      const size_t overlap =
          partial_token_overlap(current_text, think_end_token_);
      const size_t safe_size = current_text.size() - overlap;
      std::string reasoning_text = current_text.substr(0, safe_size);
      buffer_ = current_text.substr(safe_size);
      if (reasoning_text.empty()) {
        return ReasoningResult();
      }
      return ReasoningResult(std::nullopt, std::move(reasoning_text));
    } else {
      return ReasoningResult();
    }
  }

  // If we're not in a reasoning block return as normal text
  if (!in_reasoning_) {
    buffer_.clear();
    return ReasoningResult(std::string(current_text), std::nullopt);
  }

  return ReasoningResult();
}
}  // namespace xllm
