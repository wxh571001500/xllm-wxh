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

#include "framework/chat_template/kimi_k3_chat_template.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(KimiK3ChatTemplateTest, AddsRequiredToolChoiceInstruction) {
  KimiK3ChatTemplate chat_template;
  ChatMessages messages = {Message("user", "weather?")};
  nlohmann::ordered_json kwargs = {
      {"thinking", false}, {"tool_choice", "required"}};

  const auto prompt = chat_template.apply(messages, {}, kwargs);

  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("<|open|>message role=\"system\" "
                         "type=\"tool-choice\"<|sep|>"),
            std::string::npos);
  EXPECT_NE(prompt->find("You MUST call tools in the next message."),
            std::string::npos);
  EXPECT_TRUE(prompt->ends_with(
      "<|open|>message role=\"assistant\"<|sep|>"
      "<|open|>response<|sep|>"));
}

TEST(KimiK3ChatTemplateTest, UsesMappedThinkingEffort) {
  KimiK3ChatTemplate chat_template;
  ChatMessages messages = {Message("user", "hello")};
  nlohmann::ordered_json kwargs = {
      {"thinking", true}, {"thinking_effort", "high"}};

  const auto prompt = chat_template.apply(messages, {}, kwargs);

  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("thinking_effort=high"), std::string::npos);
  EXPECT_TRUE(prompt->ends_with(
      "<|open|>message role=\"assistant\"<|sep|>"
      "<|open|>think<|sep|>"));
}

TEST(KimiK3ChatTemplateTest, RejectsUnsupportedThinkingEffort) {
  KimiK3ChatTemplate chat_template;
  ChatMessages messages = {Message("user", "hello")};
  nlohmann::ordered_json kwargs = {
      {"thinking", true}, {"thinking_effort", "medium"}};

  EXPECT_FALSE(chat_template.apply(messages, {}, kwargs).has_value());
}

TEST(KimiK3ChatTemplateTest, ResolvesToolResultNameFromToolCallId) {
  KimiK3ChatTemplate chat_template;
  Message assistant("assistant", "");
  Message::ToolCall tool_call;
  tool_call.id = "call_123";
  tool_call.type = "function";
  tool_call.function.name = "get_weather";
  tool_call.function.arguments = R"({"city":"Beijing"})";
  assistant.tool_calls = Message::ToolCallVec{tool_call};
  Message tool_result("tool", R"({"temperature":20})");
  tool_result.tool_call_id = "call_123";
  ChatMessages messages = {
      Message("user", "weather?"), assistant, tool_result};
  nlohmann::ordered_json kwargs = {{"thinking", false}};

  const auto prompt = chat_template.apply(messages, {}, kwargs);

  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("<|open|>message role=\"tool\" "
                         "tool=\"get_weather\" index=\"1\"<|sep|>"),
            std::string::npos);
  EXPECT_EQ(prompt->find("tool=\"call_123\""), std::string::npos);
}

}  // namespace
}  // namespace xllm
