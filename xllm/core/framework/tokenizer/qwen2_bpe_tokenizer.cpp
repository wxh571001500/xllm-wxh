/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "qwen2_bpe_tokenizer.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/string_view.h>
#include <glog/logging.h>

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xllm {

namespace {

std::string resolve_path(const std::string_view& dir_path,
                         const std::string& file_name) {
  if (file_name.empty()) {
    return "";
  }
  if (dir_path.empty() || std::filesystem::path(file_name).is_absolute()) {
    return file_name;
  }
  return absl::StrCat(dir_path, "/", file_name);
}

bool add_special_token_id(const std::string& token,
                          std::optional<int32_t> token_id,
                          std::vector<int32_t>* ids,
                          bool prepend) {
  if (token.empty() || !token_id.has_value()) {
    if (!token.empty() && !token_id.has_value()) {
      LOG(WARNING) << "Failed to find token ID for token: " << token;
    }
    return false;
  }

  const int32_t id = token_id.value();
  if (prepend) {
    if (!ids->empty() && ids->front() == id) {
      return false;
    }
    ids->insert(ids->begin(), id);
  } else {
    if (!ids->empty() && ids->back() == id) {
      return false;
    }
    ids->push_back(id);
  }
  return true;
}

}  // namespace

Qwen2BPETokenizer::Qwen2BPETokenizer(const std::string_view& dir_path,
                                     const TokenizerArgs& args)
    : dir_path_(dir_path), args_(args) {
  const std::string vocab_file_path =
      resolve_path(dir_path_, args.vocab_file());
  const std::string merges_file_path =
      resolve_path(dir_path_, args.merges_file());

  handle_ = tokenizers_new_qwen2_from_files(vocab_file_path.c_str(),
                                            merges_file_path.c_str(),
                                            args_.add_prefix_space());
  CHECK(handle_ != nullptr) << "Failed to create Qwen2 tokenizer from "
                            << vocab_file_path << " and " << merges_file_path;

  if (!args.special_tokens().empty()) {
    load_special_tokens(args.special_tokens());
  }

  if (!args.prefix_tokens().empty()) {
    for (const auto& token : args.prefix_tokens()) {
      if (token.empty()) {
        continue;
      }
      const auto token_id = token_to_id(token);
      if (token_id.has_value()) {
        prefix_token_ids_.push_back(token_id.value());
        LOG(INFO) << "Prefix token: " << token << ", id: " << token_id.value();
      } else {
        LOG(ERROR) << "Failed to find prefix token: " << token;
      }
    }
  }
}

Qwen2BPETokenizer::~Qwen2BPETokenizer() {
  if (handle_ != nullptr) {
    tokenizers_free(handle_);
    handle_ = nullptr;
  }
}

void Qwen2BPETokenizer::load_special_tokens(
    const std::vector<SpecialToken>& special_tokens) {
  for (const auto& [token, id] : special_tokens) {
    if (token.empty()) {
      continue;
    }

    if (!special_token_encoder_.try_emplace(token, id).second) {
      LOG(WARNING) << "Duplicate special token: " << token << ", id: " << id;
    }
    if (!special_token_decoder_.try_emplace(id, token).second) {
      LOG(WARNING) << "Duplicate special token: " << token << ", id: " << id;
    }
  }

  std::vector<std::string> escaped_tokens;
  escaped_tokens.reserve(special_tokens.size());
  for (const auto& [token, id] : special_tokens) {
    UNUSED_PARAMETER(id);
    if (token.empty()) {
      continue;
    }
    escaped_tokens.push_back(re2::RE2::QuoteMeta(token));
  }

  std::sort(escaped_tokens.begin(),
            escaped_tokens.end(),
            [](const std::string& lhs, const std::string& rhs) {
              return lhs.size() > rhs.size();
            });
  if (!escaped_tokens.empty()) {
    special_token_regex_ = std::make_unique<re2::RE2>(
        absl::StrCat("(", absl::StrJoin(escaped_tokens, "|"), ")"));
    CHECK_EQ(special_token_regex_->error_code(), 0)
        << "Failed to compile special token regex: "
        << special_token_regex_->error();
  }
}

bool Qwen2BPETokenizer::encode_internal(const std::string_view& text,
                                        std::vector<int32_t>* ids) const {
  if (text.empty()) {
    return true;
  }

  TokenizerEncodeResult result;
  tokenizers_encode(
      handle_, text.data(), text.size(), /*add_special_token=*/0, &result);

  ids->insert(ids->end(), result.token_ids, result.token_ids + result.len);
  if (result.token_ids != nullptr && result.len > 0) {
    tokenizers_free_encode_results(&result, 1);
  }
  return true;
}

bool Qwen2BPETokenizer::encode(const std::string_view& text,
                               std::vector<int32_t>* ids,
                               bool add_special_tokens) const {
  if (!prefix_token_ids_.empty()) {
    ids->insert(
        ids->begin(), prefix_token_ids_.begin(), prefix_token_ids_.end());
  }

  if (special_token_regex_ == nullptr) {
    if (!encode_internal(text, ids)) {
      return false;
    }
  } else {
    absl::string_view input{text.data(), text.size()};
    absl::string_view special;
    while (true) {
      const auto* start = input.begin();
      if (!re2::RE2::FindAndConsume(&input, *special_token_regex_, &special)) {
        break;
      }

      const std::string_view sub_input(start,
                                       input.begin() - start - special.size());
      if (!encode_internal(sub_input, ids)) {
        return false;
      }

      const auto it = special_token_encoder_.find(special);
      if (it != special_token_encoder_.end()) {
        ids->push_back(it->second);
      }
    }

    if (!encode_internal({input.data(), input.size()}, ids)) {
      return false;
    }
  }

  if (add_special_tokens && args_.add_bos_token() &&
      !args_.bos_token().empty()) {
    add_special_token_id(args_.bos_token(),
                         token_to_id(args_.bos_token()),
                         ids,
                         /*prepend=*/true);
  }
  if (add_special_tokens && args_.add_eos_token() &&
      !args_.eos_token().empty()) {
    add_special_token_id(args_.eos_token(),
                         token_to_id(args_.eos_token()),
                         ids,
                         /*prepend=*/false);
  }

  return true;
}

std::string Qwen2BPETokenizer::decode_internal(const Slice<int32_t>& ids,
                                               size_t start,
                                               size_t end) const {
  if (start >= end) {
    return "";
  }

  const char* data = nullptr;
  size_t len = 0;
  tokenizers_decode(handle_,
                    reinterpret_cast<const uint32_t*>(ids.data() + start),
                    end - start,
                    /*skip_special_tokens=*/0,
                    &data,
                    &len);
  return {data, len};
}

std::string Qwen2BPETokenizer::decode(const Slice<int32_t>& ids,
                                      bool skip_special_tokens) const {
  std::string decoded;
  size_t start = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    const auto it = special_token_decoder_.find(ids[i]);
    if (it == special_token_decoder_.end()) {
      continue;
    }

    decoded.append(decode_internal(ids, start, i));
    if (!skip_special_tokens) {
      decoded.append(it->second);
    }
    start = i + 1;
  }
  decoded.append(decode_internal(ids, start, ids.size()));
  return decoded;
}

std::optional<int32_t> Qwen2BPETokenizer::token_to_id(
    const std::string_view& token) const {
  const absl::string_view token_view{token.data(), token.size()};
  if (const auto it = special_token_encoder_.find(token_view);
      it != special_token_encoder_.end()) {
    return it->second;
  }

  int32_t id = -1;
  tokenizers_token_to_id(handle_, token.data(), token.size(), &id);
  return id >= 0 ? std::optional<int32_t>(id) : std::nullopt;
}

std::string Qwen2BPETokenizer::id_to_token(int32_t id) const {
  if (const auto it = special_token_decoder_.find(id);
      it != special_token_decoder_.end()) {
    return it->second;
  }

  const char* data = nullptr;
  size_t len = 0;
  tokenizers_id_to_token(handle_, id, &data, &len);
  return {data, len};
}

size_t Qwen2BPETokenizer::vocab_size() const {
  size_t size = 0;
  tokenizers_get_vocab_size(handle_, &size);
  for (const auto& [id, token] : special_token_decoder_) {
    UNUSED_PARAMETER(token);
    const char* data = nullptr;
    size_t len = 0;
    tokenizers_id_to_token(handle_, id, &data, &len);
    const std::string base_token =
        (data == nullptr) ? std::string() : std::string(data, len);
    if (base_token.empty()) {
      ++size;
    }
  }
  return size;
}

std::unique_ptr<Tokenizer> Qwen2BPETokenizer::clone() const {
  return std::make_unique<Qwen2BPETokenizer>(dir_path_, args_);
}

}  // namespace xllm
