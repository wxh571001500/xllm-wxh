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

#include "core/framework/chat_template/kimi_k3_chat_template.h"

#include <glog/logging.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

namespace xllm {
namespace {

constexpr const char* kOpenToken = "<|open|>";
constexpr const char* kCloseToken = "<|close|>";
constexpr const char* kSepToken = "<|sep|>";
constexpr const char* kEndOfMessageToken = "<|end_of_msg|>";
constexpr const char* kImagePrompt =
    "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>";

constexpr const char* kRoleSystem = "system";
constexpr const char* kRoleUser = "user";
constexpr const char* kRoleAssistant = "assistant";
constexpr const char* kRoleTool = "tool";

std::string escape_attribute(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    if (character == '&') {
      escaped.append("&amp;");
    } else if (character == '"') {
      escaped.append("&quot;");
    } else {
      escaped.push_back(character);
    }
  }
  return escaped;
}

void append_open_tag(
    std::ostringstream& output,
    const std::string& tag,
    const std::vector<std::pair<std::string, std::string>>& attributes = {}) {
  output << kOpenToken << tag;
  for (const auto& [key, value] : attributes) {
    output << ' ' << key << "=\"" << escape_attribute(value) << '"';
  }
  output << kSepToken;
}

void append_close_tag(std::ostringstream& output, const std::string& tag) {
  output << kCloseToken << tag << kSepToken;
}

void append_message_end(std::ostringstream& output) {
  append_close_tag(output, "message");
  output << kEndOfMessageToken;
}

void append_content(std::ostringstream& output,
                    const Message::Content& content) {
  if (std::holds_alternative<std::string>(content)) {
    output << std::get<std::string>(content);
    return;
  }

  for (const MMContent& item : std::get<MMContentVec>(content)) {
    if (item.type == "text") {
      output << item.text;
    } else if (item.type == "image" || item.type == "image_url" ||
               item.type == "image_embedding") {
      output << kImagePrompt;
    }
  }
}

bool get_thinking(const nlohmann::ordered_json& kwargs) {
  auto thinking = kwargs.find("thinking");
  if (thinking != kwargs.end() && thinking->is_boolean()) {
    return thinking->get<bool>();
  }
  thinking = kwargs.find("enable_thinking");
  if (thinking != kwargs.end() && thinking->is_boolean()) {
    return thinking->get<bool>();
  }
  return true;
}

std::optional<std::string> get_thinking_effort(
    const nlohmann::ordered_json& kwargs) {
  auto thinking_effort = kwargs.find("thinking_effort");
  if (thinking_effort == kwargs.end() || thinking_effort->is_null()) {
    return std::string("max");
  }
  if (!thinking_effort->is_string()) {
    return std::nullopt;
  }

  const std::string value = thinking_effort->get<std::string>();
  static const std::unordered_set<std::string> kValidValues = {
      "low", "high", "max"};
  return kValidValues.contains(value) ? std::optional<std::string>(value)
                                      : std::nullopt;
}

void append_internal_system_message(std::ostringstream& output,
                                    const std::string& message_type,
                                    const std::string& body) {
  append_open_tag(output,
                  "message",
                  {{"role", kRoleSystem}, {"type", message_type}});
  output << body;
  append_message_end(output);
}

void append_tools(std::ostringstream& output,
                  const std::vector<xllm::JsonTool>& json_tools) {
  if (json_tools.empty()) {
    return;
  }

  nlohmann::ordered_json tools = nlohmann::json::array();
  for (const JsonTool& tool : json_tools) {
    tools.emplace_back(nlohmann::ordered_json{
        {"type", tool.type},
        {"function",
         nlohmann::ordered_json{{"name", tool.function.name},
                                {"description", tool.function.description},
                                {"parameters", tool.function.parameters}}}});
  }

  append_internal_system_message(
      output,
      "tool-declare",
      "# Tools\nHere are the available tools, described in JSONSchema.\n\n"
      "```json\n" +
          tools.dump() + "\n```");
}

void append_assistant_message(std::ostringstream& output,
                              const Message& message,
                              bool thinking) {
  append_open_tag(output, "message", {{"role", kRoleAssistant}});
  if (thinking) {
    append_open_tag(output, "think");
    if (message.reasoning_content.has_value()) {
      output << *message.reasoning_content;
    }
    append_close_tag(output, "think");
  }

  append_open_tag(output, "response");
  append_content(output, message.content);
  append_close_tag(output, "response");

  if (message.tool_calls.has_value()) {
    append_open_tag(output, "tools");
    int32_t index = 1;
    for (const Message::ToolCall& tool_call : *message.tool_calls) {
      append_open_tag(output,
                      "call",
                      {{"tool", tool_call.function.name},
                       {"index", std::to_string(index)}});
      if (!tool_call.function.arguments.empty()) {
        append_open_tag(output, "json", {{"type", "object"}});
        output << tool_call.function.arguments;
        append_close_tag(output, "json");
      }
      append_close_tag(output, "call");
      ++index;
    }
    append_close_tag(output, "tools");
  }
  append_message_end(output);
}

}  // namespace

std::optional<std::string> KimiK3ChatTemplate::apply(
    const ChatMessages& messages) const {
  const std::vector<xllm::JsonTool> empty_tools;
  const nlohmann::ordered_json kwargs = nlohmann::json::object();
  return apply(messages, empty_tools, kwargs);
}

std::optional<std::string> KimiK3ChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  const bool thinking = get_thinking(chat_template_kwargs);
  const std::optional<std::string> thinking_effort =
      get_thinking_effort(chat_template_kwargs);
  if (thinking && !thinking_effort.has_value()) {
    LOG(ERROR) << "Kimi K3 thinking_effort must be low, high, or max.";
    return std::nullopt;
  }

  std::ostringstream output;
  append_tools(output, json_tools);
  if (thinking) {
    append_internal_system_message(
        output,
        "thinking-effort",
        "`thinking_effort` guides on how much to think in your thinking "
        "channel (not including the response channel), supported values "
        "include `low`, `medium`, `high`, and `max`.\nNow the system is "
        "invoked with `thinking_effort=" +
            *thinking_effort + "`.");
  }

  int32_t tool_index = 0;
  std::unordered_map<std::string, std::pair<int32_t, std::string>>
      pending_tool_calls;
  for (const Message& message : messages) {
    if (message.role == kRoleAssistant) {
      append_assistant_message(output, message, thinking);
      tool_index = 0;
      pending_tool_calls.clear();
      if (message.tool_calls.has_value()) {
        int32_t index = 1;
        for (const Message::ToolCall& tool_call : *message.tool_calls) {
          if (!tool_call.id.empty()) {
            pending_tool_calls.emplace(
                tool_call.id,
                std::make_pair(index, tool_call.function.name));
          }
          ++index;
        }
      }
      continue;
    }

    std::vector<std::pair<std::string, std::string>> attributes = {
        {"role", message.role}};
    if (message.role == kRoleTool) {
      ++tool_index;
      std::string tool_name = "tool";
      int32_t resolved_index = tool_index;
      if (message.tool_call_id.has_value()) {
        const auto call_it = pending_tool_calls.find(*message.tool_call_id);
        if (call_it != pending_tool_calls.end()) {
          resolved_index = call_it->second.first;
          tool_name = call_it->second.second;
        }
      }
      attributes.emplace_back("tool", tool_name);
      attributes.emplace_back("index", std::to_string(resolved_index));
    } else if (message.role != kRoleUser && message.role != kRoleSystem) {
      continue;
    }

    append_open_tag(output, "message", attributes);
    append_content(output, message.content);
    append_message_end(output);
  }

  const auto tool_choice = chat_template_kwargs.find("tool_choice");
  if (tool_choice != chat_template_kwargs.end() &&
      tool_choice->is_string()) {
    const std::string choice = tool_choice->get<std::string>();
    if (choice == "required") {
      append_internal_system_message(
          output,
          "tool-choice",
          "The system is invoked with `tool_choice=required`.\n"
          "You MUST call tools in the next message.");
    } else if (choice == "none") {
      append_internal_system_message(
          output,
          "tool-choice",
          "The system is invoked with `tool_choice=none`.\n"
          "You MUST NOT call any tools in the next message.");
    }
  }

  append_open_tag(output, "message", {{"role", kRoleAssistant}});
  append_open_tag(output, thinking ? "think" : "response");
  return output.str();
}

}  // namespace xllm
