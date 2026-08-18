/* Copyright 2025-2026 The xLLM Authors.

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

#include "xllm/parser/reasoning_parser.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(KimiK3ReasoningParserTest, ParsesReasoningAndResponseBoundary) {
  ReasoningParser parser("kimi_k3", true, true);
  std::string text =
      "plan<|close|>think<|sep|><|open|>response<|sep|>answer";
  const auto result = parser.parse_non_stream(text);
  EXPECT_EQ(result.reasoning_text, "plan");
  EXPECT_EQ(result.normal_text, "<|open|>response<|sep|>answer");
}

TEST(KimiK3ReasoningParserTest, StreamsPartialThinkEndMarker) {
  ReasoningParser parser("kimi_k3", true, true);
  EXPECT_EQ(parser.parse_stream_chunk("plan<|close|>think<|se").reasoning_text,
            "plan");
  const auto result = parser.parse_stream_chunk(
      "p|><|open|>response<|sep|>answer");
  EXPECT_EQ(result.reasoning_text, "");
  EXPECT_EQ(result.normal_text, "<|open|>response<|sep|>answer");
}

TEST(KimiK3ReasoningParserTest, AutoRegistryRecognizesKimiK3) {
  EXPECT_EQ(ReasoningParser::get_parser_auto("auto", "kimi_k3"), "kimi_k3");
}

}  // namespace
}  // namespace xllm
