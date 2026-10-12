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

namespace xllm::deepseek_v4 {

inline constexpr char kThinkingModeThinking[] = "thinking";
inline constexpr char kThinkingModeChat[] = "chat";
inline constexpr char kReasoningEffortNone[] = "none";

template <typename Json>
std::string resolve_thinking_mode(const Json& kwargs) {
  const auto mode = kwargs.find("thinking_mode");
  if (mode != kwargs.end() && mode->is_string()) {
    return mode->template get<std::string>();
  }

  const auto effort = kwargs.find("reasoning_effort");
  if (effort != kwargs.end() && effort->is_string() &&
      effort->template get<std::string>() == kReasoningEffortNone) {
    return kThinkingModeChat;
  }

  const auto thinking = kwargs.find("thinking");
  if (thinking != kwargs.end() && thinking->is_boolean()) {
    return thinking->template get<bool>() ? kThinkingModeThinking
                                          : kThinkingModeChat;
  }
  const auto enable_thinking = kwargs.find("enable_thinking");
  if (enable_thinking != kwargs.end() && enable_thinking->is_boolean()) {
    return enable_thinking->template get<bool>() ? kThinkingModeThinking
                                                 : kThinkingModeChat;
  }
  return kThinkingModeThinking;
}

}  // namespace xllm::deepseek_v4
