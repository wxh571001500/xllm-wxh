/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"
#include "core/framework/chat_template/chat_template.h"

namespace xllm {

class KimiK3ChatTemplate final : public ChatTemplate {
 public:
  std::optional<std::string> apply(
      const ChatMessages& messages) const override;

  std::optional<std::string> apply(
      const ChatMessages& messages,
      const std::vector<xllm::JsonTool>& json_tools,
      const nlohmann::ordered_json& chat_template_kwargs) const override;
};

}  // namespace xllm
