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

#pragma once
#include <absl/strings/str_split.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace xllm {

// an thin wrapper around nlohmann/json to read json files.
// it supports read keys with dot notation from json.
// for example: value_or("a.b.c", 0) will return 100 for following json:
// {
//   "a": {
//     "b": {
//       "c": 100
//     }
//   }
// }
//
class JsonReader {
 public:
  // parse the json file, return true if success
  bool parse(const std::string& json_file_path);

  // parse json content from a string, return true if success
  bool parse_text(const std::string& json_text);

  // check if the json contains the key, key can be nested with dot notation
  bool contains(const std::string& key) const;

  template <typename T, typename T2>
  T value_or(const std::vector<std::string>& keys, T2 default_value) const {
    for (const auto& key : keys) {
      if (auto data = value<T>(key)) {
        return data.value();
      }
    }
    // may introduce implicit conversion from T2 to T
    return default_value;
  }

  template <typename T, typename T2>
  T value_or(const std::string& key, T2 default_value) const {
    if (auto data = value<T>(key)) {
      return data.value();
    }
    // may introduce implicit conversion from T2 to T
    return default_value;
  }

  template <typename T>
  std::optional<T> value(const std::string& key) const {
    if (auto data = resolve(key)) {
      if (data->is_null() || data->is_object()) {
        // cannot convert null or object data to T
        return std::nullopt;
      }
      return data->get<T>();
    }
    return std::nullopt;
  }

  // Read an integer field that HF configs may encode as either a scalar or an
  // array (e.g. eos_token_id) into a vector. A scalar yields a single-element
  // vector; a non-empty array is read as-is. Returns nullopt when the key is
  // absent, null, an object, OR an empty array (so callers get their default
  // and downstream code never has to guard against ``vec.front()`` on empty).
  // Unlike value<std::vector<int32_t>>, a scalar node does not throw.
  std::optional<std::vector<int32_t>> value_int_or_array(
      const std::string& key) const {
    const nlohmann::json* data = resolve(key);
    if (data == nullptr || data->is_null() || data->is_object()) {
      return std::nullopt;
    }
    if (data->is_array()) {
      auto vec = data->get<std::vector<int32_t>>();
      if (vec.empty()) {
        return std::nullopt;
      }
      return vec;
    }
    return std::vector<int32_t>{data->get<int32_t>()};
  }

  // Resolve a dot-separated key path against a json object; returns nullptr if
  // any segment is missing.
  static const nlohmann::json* resolve_path(const nlohmann::json& root,
                                            const std::string& key) {
    const std::vector<std::string> keys = absl::StrSplit(key, '.');
    const nlohmann::json* data = &root;
    for (const auto& k : keys) {
      if (data->contains(k)) {
        data = &(*data)[k];
      } else {
        return nullptr;
      }
    }
    return data;
  }

  // Resolve a dot-separated key path against the top-level document. Loaders
  // that need to read from a "text_config" subtree must spell the path
  // explicitly (e.g. "text_config.num_hidden_layers"); the reader does not
  // perform an implicit fallback that would silently redirect vision-side keys
  // (e.g. "head_dim") to the text subtree.
  const nlohmann::json* resolve(const std::string& key) const {
    return resolve_path(data_, key);
  }

  nlohmann::json data() const { return data_; }

  // Mutable access to the parsed document, for in-place rewrites that avoid a
  // dump()/parse_text() round-trip. The reader keeps no derived state, so
  // resolve()/value() see mutations immediately.
  nlohmann::json& mutable_data() { return data_; }

 private:
  nlohmann::json data_;
};

}  // namespace xllm
