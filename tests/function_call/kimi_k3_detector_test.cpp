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

#include "xllm/function_call/kimi_k3_detector.h"
#include "xllm/function_call/function_call_parser.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

namespace xllm {
namespace function_call {
namespace {

JsonTool make_tool(const std::string& name) {
  return JsonTool("function",
                  JsonFunction(name,
                               "test tool",
                               nlohmann::json{{"type", "object"}}));
}

constexpr const char* kResponseEnd = "<|close|>response<|sep|>";
constexpr const char* kToolsStart = "<|open|>tools<|sep|>";
constexpr const char* kToolsEnd = "<|close|>tools<|sep|>";
constexpr const char* kMessageEnd = "<|close|>message<|sep|>";
constexpr const char* kEndOfMessage = "<|end_of_msg|>";

TEST(KimiK3DetectorTest, ParsesJsonAndTypedArguments) {
  KimiK3Detector detector;
  const std::vector<JsonTool> tools = {make_tool("get_weather"),
                                       make_tool("calculate")};
  const std::string text =
      "The answer" + std::string(kResponseEnd) + kToolsStart +
      "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
      "<|open|>json type=\"object\"<|sep|>{\"city\": \"Beijing\"}"
      "<|close|>json<|sep|><|close|>call<|sep|>"
      "<|open|>call tool=\"calculate\" index=\"2\"<|sep|>"
      "<|open|>argument key=\"value\" type=\"number\"<|sep|>3"
      "<|close|>argument<|sep|><|close|>call<|sep|>" +
      std::string(kToolsEnd) + kMessageEnd + kEndOfMessage;

  const auto result = detector.detect_and_parse(text, tools);
  ASSERT_EQ(result.normal_text, "The answer");
  ASSERT_EQ(result.calls.size(), 2);
  EXPECT_EQ(result.calls[0].tool_index, 0);
  EXPECT_EQ(result.calls[0].name, "get_weather");
  EXPECT_EQ(nlohmann::json::parse(result.calls[0].parameters)["city"],
            "Beijing");
  EXPECT_EQ(result.calls[1].tool_index, 1);
  EXPECT_EQ(result.calls[1].name, "calculate");
  EXPECT_EQ(nlohmann::json::parse(result.calls[1].parameters)["value"], 3);
}

TEST(KimiK3DetectorTest, ParsesGeneratedResponseWithoutResponseStart) {
  KimiK3Detector detector;
  const auto result = detector.detect_and_parse(
      "plain response" + std::string(kResponseEnd) + kMessageEnd,
      {});
  EXPECT_EQ(result.normal_text, "plain response");
  EXPECT_TRUE(result.calls.empty());
}

TEST(KimiK3DetectorTest, FunctionParserCleansContentWithoutToolCalls) {
  FunctionCallParser parser({}, "kimi_k3");
  const auto [content, calls] = parser.parse_non_stream(
      "plain response" + std::string(kResponseEnd) + kMessageEnd);

  EXPECT_EQ(content, "plain response");
  EXPECT_TRUE(calls.empty());
}

TEST(KimiK3DetectorTest, EmitsStreamingCallsOnlyAfterMessageEnd) {
  KimiK3Detector detector;
  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  const std::string prefix = "hello" + std::string(kResponseEnd) + kToolsStart +
                             "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
                             "<|open|>json type=\"object\"<|sep|>{}"
                             "<|close|>json<|sep|><|close|>call<|sep|>" +
                             std::string(kToolsEnd);
  EXPECT_EQ(detector.parse_streaming_increment(prefix, tools).calls.size(), 0);
  const auto result = detector.parse_streaming_increment(
      std::string(kMessageEnd) + kEndOfMessage, tools);
  ASSERT_EQ(result.calls.size(), 1);
  EXPECT_EQ(result.calls[0].name, "get_weather");
  EXPECT_EQ(result.calls[0].parameters, "{}");
}

TEST(KimiK3DetectorTest, RejectsUnknownToolAndNonSequentialIndex) {
  KimiK3Detector detector;
  const std::vector<JsonTool> tools = {make_tool("get_weather")};
  const std::string unknown =
      std::string(kResponseEnd) + kToolsStart +
      "<|open|>call tool=\"unknown\" index=\"1\"<|sep|><|close|>call<|sep|>" +
      std::string(kToolsEnd) + kMessageEnd;
  EXPECT_TRUE(detector.detect_and_parse(unknown, tools).calls.empty());

  const std::string bad_index =
      std::string(kResponseEnd) + kToolsStart +
      "<|open|>call tool=\"get_weather\" index=\"2\"<|sep|><|close|>call<|sep|>" +
      std::string(kToolsEnd) + kMessageEnd;
  EXPECT_TRUE(detector.detect_and_parse(bad_index, tools).calls.empty());
}

}  // namespace
}  // namespace function_call
}  // namespace xllm
