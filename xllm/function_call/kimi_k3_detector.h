/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstddef>
#include <string>
#include <vector>

#include "base_format_detector.h"

namespace xllm {
namespace function_call {

class KimiK3Detector final : public BaseFormatDetector {
 public:
  KimiK3Detector() = default;
  ~KimiK3Detector() override = default;

  StreamingParseResult detect_and_parse(
      const std::string& text,
      const std::vector<JsonTool>& tools) override;

  bool has_tool_call(const std::string& text) override;

  StreamingParseResult parse_streaming_increment(
      const std::string& new_text,
      const std::vector<JsonTool>& tools) override;

 private:
  size_t streamed_content_size_ = 0;
  size_t streamed_call_count_ = 0;
};

}  // namespace function_call
}  // namespace xllm
