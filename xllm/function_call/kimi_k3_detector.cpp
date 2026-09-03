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

#include "kimi_k3_detector.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace xllm {
namespace function_call {
namespace {

constexpr std::string_view kOpenToken = "<|open|>";
constexpr std::string_view kSepToken = "<|sep|>";
constexpr std::string_view kEndOfMessageToken = "<|end_of_msg|>";
constexpr std::string_view kThinkStart = "<|open|>think<|sep|>";
constexpr std::string_view kThinkEnd = "<|close|>think<|sep|>";
constexpr std::string_view kResponseStart = "<|open|>response<|sep|>";
constexpr std::string_view kResponseEnd = "<|close|>response<|sep|>";
constexpr std::string_view kToolsStart = "<|open|>tools<|sep|>";
constexpr std::string_view kToolsEnd = "<|close|>tools<|sep|>";
constexpr std::string_view kCallEnd = "<|close|>call<|sep|>";
constexpr std::string_view kArgumentEnd = "<|close|>argument<|sep|>";
constexpr std::string_view kJsonEnd = "<|close|>json<|sep|>";
constexpr std::string_view kMessageEnd = "<|close|>message<|sep|>";

struct OpenTag {
  std::unordered_map<std::string, std::string> attributes;
  size_t body_start = 0;
};

struct ParseSnapshot {
  std::string content;
  std::vector<ToolCallItem> calls;
  bool protocol_complete = false;
};

bool is_attribute_key_char(char character) {
  return std::isalnum(static_cast<unsigned char>(character)) != 0 ||
         character == '_' || character == '.' || character == '-';
}

std::string decode_attribute(std::string_view value) {
  std::string decoded;
  decoded.reserve(value.size());
  for (size_t index = 0; index < value.size();) {
    if (value.substr(index).starts_with("&quot;")) {
      decoded.push_back('"');
      index += 6;
    } else if (value.substr(index).starts_with("&amp;")) {
      decoded.push_back('&');
      index += 5;
    } else if (value[index] == '&') {
      throw std::runtime_error("Unsupported XTML attribute entity.");
    } else {
      decoded.push_back(value[index]);
      ++index;
    }
  }
  return decoded;
}

std::unordered_map<std::string, std::string> parse_attributes(
    std::string_view attributes) {
  std::unordered_map<std::string, std::string> result;
  size_t cursor = 0;
  while (cursor < attributes.size()) {
    while (cursor < attributes.size() &&
           std::isspace(static_cast<unsigned char>(attributes[cursor])) != 0) {
      ++cursor;
    }
    if (cursor == attributes.size()) {
      break;
    }

    const size_t key_start = cursor;
    while (cursor < attributes.size() &&
           is_attribute_key_char(attributes[cursor])) {
      ++cursor;
    }
    if (key_start == cursor || cursor + 1 >= attributes.size() ||
        attributes[cursor] != '=' || attributes[cursor + 1] != '"') {
      throw std::runtime_error("Malformed XTML attributes.");
    }
    const std::string key(attributes.substr(key_start, cursor - key_start));
    cursor += 2;
    const size_t value_start = cursor;
    const size_t quote = attributes.find('"', cursor);
    if (quote == std::string_view::npos) {
      throw std::runtime_error("Malformed XTML attribute value.");
    }
    if (result.contains(key)) {
      throw std::runtime_error("Duplicate XTML attribute.");
    }
    result.emplace(
        key,
        decode_attribute(attributes.substr(value_start, quote - value_start)));
    cursor = quote + 1;
  }
  return result;
}

OpenTag parse_open_tag(const std::string& text,
                       size_t cursor,
                       std::string_view tag) {
  const std::string prefix = std::string(kOpenToken) + std::string(tag);
  if (text.compare(cursor, prefix.size(), prefix) != 0) {
    throw std::runtime_error("Unexpected XTML tag.");
  }
  const size_t separator = text.find(kSepToken, cursor + prefix.size());
  if (separator == std::string::npos) {
    throw std::runtime_error("Unclosed XTML start tag.");
  }

  OpenTag result;
  const size_t attributes_start = cursor + prefix.size();
  result.attributes = parse_attributes(std::string_view(text).substr(
      attributes_start, separator - attributes_start));
  result.body_start = separator + kSepToken.size();
  return result;
}

void require_attributes(
    const std::unordered_map<std::string, std::string>& attributes,
    std::initializer_list<std::string_view> required) {
  if (attributes.size() != required.size()) {
    throw std::runtime_error("Unexpected XTML attributes.");
  }
  for (const std::string_view key : required) {
    if (!attributes.contains(std::string(key))) {
      throw std::runtime_error("Missing required XTML attribute.");
    }
  }
}

size_t partial_marker_overlap(std::string_view text, std::string_view marker) {
  const size_t maximum = std::min(text.size(), marker.size() - 1);
  for (size_t overlap = maximum; overlap > 0; --overlap) {
    if (text.substr(text.size() - overlap) == marker.substr(0, overlap)) {
      return overlap;
    }
  }
  return 0;
}

std::string safe_prefix(std::string_view text, std::string_view marker) {
  const size_t overlap = partial_marker_overlap(text, marker);
  return std::string(text.substr(0, text.size() - overlap));
}

void skip_whitespace(std::string_view text, size_t* cursor) {
  while (*cursor < text.size() &&
         std::isspace(static_cast<unsigned char>(text[*cursor])) != 0) {
    ++*cursor;
  }
}

nlohmann::json parse_json_value(std::string_view value) {
  std::unordered_map<int, std::unordered_set<std::string>> keys_by_depth;
  auto reject_duplicate_keys = [&keys_by_depth](
                                   int depth,
                                   nlohmann::json::parse_event_t event,
                                   nlohmann::json& parsed_value) {
    if (event == nlohmann::json::parse_event_t::object_start) {
      keys_by_depth[depth].clear();
    } else if (event == nlohmann::json::parse_event_t::key) {
      const std::string key = parsed_value.get<std::string>();
      if (!keys_by_depth[depth].insert(key).second) {
        throw std::runtime_error("Duplicate JSON key in K3 arguments.");
      }
    } else if (event == nlohmann::json::parse_event_t::object_end) {
      keys_by_depth.erase(depth);
    }
    return true;
  };
  const nlohmann::json parsed = nlohmann::json::parse(
      value.begin(), value.end(), reject_duplicate_keys, false);
  if (parsed.is_discarded()) {
    throw std::runtime_error("Invalid JSON in K3 tool arguments.");
  }
  return parsed;
}

nlohmann::json parse_typed_argument(std::string_view value,
                                    const std::string& type) {
  if (type == "string") {
    return std::string(value);
  }

  nlohmann::json parsed = parse_json_value(value);
  const bool valid = (type == "number" && parsed.is_number()) ||
                     (type == "boolean" && parsed.is_boolean()) ||
                     (type == "null" && parsed.is_null()) ||
                     (type == "object" && parsed.is_object()) ||
                     (type == "array" && parsed.is_array());
  if (!valid) {
    throw std::runtime_error("K3 typed argument has an invalid value.");
  }
  return parsed;
}

std::string parse_call_arguments(std::string_view body) {
  size_t cursor = 0;
  skip_whitespace(body, &cursor);
  if (cursor == body.size()) {
    return "{}";
  }

  const std::string json_prefix = std::string(kOpenToken) + "json";
  if (body.substr(cursor).starts_with(json_prefix)) {
    const OpenTag json_tag = parse_open_tag(std::string(body), cursor, "json");
    require_attributes(json_tag.attributes, {"type"});
    if (json_tag.attributes.at("type") != "object") {
      throw std::runtime_error("K3 json arguments must have object type.");
    }
    const size_t json_end = body.find(kJsonEnd, json_tag.body_start);
    if (json_end == std::string_view::npos) {
      throw std::runtime_error("K3 json argument block is not closed.");
    }
    const nlohmann::json value = parse_json_value(
        body.substr(json_tag.body_start, json_end - json_tag.body_start));
    if (!value.is_object()) {
      throw std::runtime_error("K3 tool arguments must be a JSON object.");
    }
    size_t tail = json_end + kJsonEnd.size();
    skip_whitespace(body, &tail);
    if (tail != body.size()) {
      throw std::runtime_error("K3 call mixes argument encodings.");
    }
    return value.dump();
  }

  nlohmann::ordered_json arguments = nlohmann::ordered_json::object();
  while (cursor < body.size()) {
    const OpenTag argument_tag =
        parse_open_tag(std::string(body), cursor, "argument");
    require_attributes(argument_tag.attributes, {"key", "type"});
    const std::string& key = argument_tag.attributes.at("key");
    if (key.empty() || arguments.contains(key)) {
      throw std::runtime_error("K3 argument key is empty or duplicated.");
    }
    const size_t argument_end =
        body.find(kArgumentEnd, argument_tag.body_start);
    if (argument_end == std::string_view::npos) {
      throw std::runtime_error("K3 typed argument is not closed.");
    }
    arguments[key] = parse_typed_argument(
        body.substr(argument_tag.body_start,
                    argument_end - argument_tag.body_start),
        argument_tag.attributes.at("type"));
    cursor = argument_end + kArgumentEnd.size();
    skip_whitespace(body, &cursor);
  }
  return arguments.dump();
}

std::vector<ToolCallItem> parse_calls(
    std::string_view body,
    const std::unordered_set<std::string>& allowed_tool_names,
    bool emit_calls) {
  std::vector<ToolCallItem> calls;
  size_t cursor = 0;
  while (cursor < body.size()) {
    skip_whitespace(body, &cursor);
    if (cursor == body.size()) {
      break;
    }

    const OpenTag call_tag = parse_open_tag(std::string(body), cursor, "call");
    require_attributes(call_tag.attributes, {"tool", "index"});
    const std::string& name = call_tag.attributes.at("tool");
    const std::string& index_text = call_tag.attributes.at("index");
    if (name.empty() || index_text.empty() || index_text.front() == '0') {
      throw std::runtime_error("Invalid K3 call name or index.");
    }
    size_t parsed_characters = 0;
    const int64_t index = std::stoll(index_text, &parsed_characters);
    if (parsed_characters != index_text.size() || index <= 0 ||
        index != static_cast<int64_t>(calls.size() + 1)) {
      throw std::runtime_error(
          "K3 call indices must be sequential and start at one.");
    }
    if (emit_calls && !allowed_tool_names.contains(name)) {
      throw std::runtime_error("K3 attempted to call an unknown tool.");
    }

    const size_t call_end = body.find(kCallEnd, call_tag.body_start);
    if (call_end == std::string_view::npos) {
      throw std::runtime_error("K3 call block is not closed.");
    }
    const std::string arguments = parse_call_arguments(
        body.substr(call_tag.body_start, call_end - call_tag.body_start));
    if (emit_calls) {
      calls.emplace_back(static_cast<int32_t>(index - 1), name, arguments);
    } else {
      calls.emplace_back();
    }
    cursor = call_end + kCallEnd.size();
  }
  if (!emit_calls) {
    calls.clear();
  }
  return calls;
}

ParseSnapshot parse_snapshot(const std::string& text,
                             const std::vector<JsonTool>& tools,
                             bool final) {
  ParseSnapshot snapshot;
  size_t cursor = 0;

  if (text.starts_with(kThinkStart)) {
    const size_t think_end = text.find(kThinkEnd, kThinkStart.size());
    if (think_end == std::string::npos) {
      if (final) {
        throw std::runtime_error("K3 think block is not closed.");
      }
      return snapshot;
    }
    cursor = think_end + kThinkEnd.size();
  }

  const std::string_view response_tail(text.data() + cursor,
                                       text.size() - cursor);
  if (response_tail.starts_with(kResponseStart)) {
    cursor += kResponseStart.size();
  } else if (!final && kResponseStart.starts_with(response_tail)) {
    return snapshot;
  }

  // The generation prompt already opens the response block when thinking is
  // disabled, so generated text legitimately starts with response content.

  const size_t response_end = text.find(kResponseEnd, cursor);
  if (response_end == std::string::npos) {
    const std::string_view response_body(text.data() + cursor,
                                         text.size() - cursor);
    if (final) {
      throw std::runtime_error("K3 response block is not closed.");
    }
    snapshot.content = safe_prefix(response_body, kResponseEnd);
    return snapshot;
  }
  snapshot.content = text.substr(cursor, response_end - cursor);
  cursor = response_end + kResponseEnd.size();

  const std::string_view tail(text.data() + cursor, text.size() - cursor);
  if (tail.starts_with(kToolsStart)) {
    const size_t tools_body_start = cursor + kToolsStart.size();
    const size_t tools_end = text.find(kToolsEnd, tools_body_start);
    if (tools_end == std::string::npos) {
      if (final) {
        throw std::runtime_error("K3 tools block is not closed.");
      }
      return snapshot;
    }
    std::unordered_set<std::string> allowed_tool_names;
    for (const JsonTool& tool : tools) {
      allowed_tool_names.insert(tool.function.name);
    }
    snapshot.calls =
        parse_calls(std::string_view(text).substr(tools_body_start,
                                                  tools_end - tools_body_start),
                    allowed_tool_names,
                    !tools.empty());
    cursor = tools_end + kToolsEnd.size();
  } else if (!final && kToolsStart.starts_with(tail)) {
    return snapshot;
  }

  const std::string_view message_tail(text.data() + cursor,
                                      text.size() - cursor);
  if (!message_tail.starts_with(kMessageEnd)) {
    if (!final && kMessageEnd.starts_with(message_tail)) {
      return snapshot;
    }
    throw std::runtime_error("K3 output is missing its message end marker.");
  }
  cursor += kMessageEnd.size();
  if (std::string_view(text).substr(cursor).starts_with(kEndOfMessageToken)) {
    cursor += kEndOfMessageToken.size();
  }
  const std::string_view remainder(text.data() + cursor, text.size() - cursor);
  if (std::any_of(remainder.begin(), remainder.end(), [](char character) {
        return std::isspace(static_cast<unsigned char>(character)) == 0;
      })) {
    throw std::runtime_error("Unexpected content after the K3 message.");
  }
  snapshot.protocol_complete = true;
  return snapshot;
}

}  // namespace

StreamingParseResult KimiK3Detector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  try {
    ParseSnapshot snapshot = parse_snapshot(text, tools, true);
    return StreamingParseResult(std::move(snapshot.content),
                                std::move(snapshot.calls));
  } catch (const std::exception& error) {
    LOG(ERROR) << "Invalid Kimi K3 XTML output: " << error.what();
    return StreamingParseResult();
  }
}

bool KimiK3Detector::has_tool_call(const std::string& text) {
  return text.find(kToolsStart) != std::string::npos;
}

StreamingParseResult KimiK3Detector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  buffer_.append(new_text);
  try {
    ParseSnapshot snapshot = parse_snapshot(buffer_, tools, false);
    std::string content_delta;
    if (snapshot.content.size() > streamed_content_size_) {
      content_delta = snapshot.content.substr(streamed_content_size_);
      streamed_content_size_ = snapshot.content.size();
    }

    std::vector<ToolCallItem> call_deltas;
    if (snapshot.protocol_complete &&
        snapshot.calls.size() > streamed_call_count_) {
      for (size_t index = streamed_call_count_; index < snapshot.calls.size();
           ++index) {
        call_deltas.push_back(snapshot.calls[index]);
        prev_tool_call_arr_.push_back(
            {{"name", snapshot.calls[index].name.value_or("")},
             {"arguments", snapshot.calls[index].parameters}});
        streamed_args_for_tool_.push_back(snapshot.calls[index].parameters);
      }
      streamed_call_count_ = snapshot.calls.size();
    }
    return StreamingParseResult(std::move(content_delta),
                                std::move(call_deltas));
  } catch (const std::exception& error) {
    LOG(ERROR) << "Invalid streaming Kimi K3 XTML output: " << error.what();
    return StreamingParseResult();
  }
}

}  // namespace function_call
}  // namespace xllm
