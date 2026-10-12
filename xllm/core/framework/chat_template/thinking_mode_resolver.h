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

#include <nlohmann/json.hpp>
#include <string>

#include "core/framework/chat_template/deepseek_v4_thinking_mode.h"

namespace xllm::thinking_mode {

inline constexpr char kDeepseekV4ReasoningParser[] = "deepseek-v4";

// Keep response parsing and the string-prompt JSON grammar fallback aligned.
// For rendered chat messages, the template's generation mode is authoritative.
inline bool is_enabled(const nlohmann::json& kwargs,
                       const std::string& reasoning_parser_format) {
  if (reasoning_parser_format == kDeepseekV4ReasoningParser) {
    return deepseek_v4::resolve_thinking_mode(kwargs) ==
           deepseek_v4::kThinkingModeThinking;
  }

  // Other parsers retain the legacy OR semantics for explicit bool flags.
  const bool default_value = !reasoning_parser_format.empty();
  if (!kwargs.contains("enable_thinking") && !kwargs.contains("thinking")) {
    return default_value;
  }
  const auto is_true = [&kwargs](const char* key) {
    const auto it = kwargs.find(key);
    return it != kwargs.end() && it->is_boolean() && it->get<bool>();
  };
  return is_true("enable_thinking") || is_true("thinking");
}

}  // namespace xllm::thinking_mode
