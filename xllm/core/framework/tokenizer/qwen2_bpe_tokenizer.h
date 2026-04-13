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

#pragma once

#include <absl/container/flat_hash_map.h>
#include <re2/re2.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tokenizer.h"
#include "tokenizer_args.h"
#include "tokenizers/tokenizers.h"

namespace xllm {

// Qwen3-TTS text tokenizer is a Qwen2-style byte-level BPE tokenizer backed by
// vocab.json + merges.txt.
class Qwen2BPETokenizer : public Tokenizer {
 public:
  Qwen2BPETokenizer(const std::string_view& dir_path,
                    const TokenizerArgs& args);
  ~Qwen2BPETokenizer() override;

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const override;

  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override;

  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override;

  std::string id_to_token(int32_t id) const override;

  size_t vocab_size() const override;

  std::unique_ptr<Tokenizer> clone() const override;

 private:
  void load_special_tokens(const std::vector<SpecialToken>& special_tokens);
  bool encode_internal(const std::string_view& text,
                       std::vector<int32_t>* ids) const;
  std::string decode_internal(const Slice<int32_t>& ids,
                              size_t start,
                              size_t end) const;

  std::string dir_path_;
  TokenizerArgs args_;

  absl::flat_hash_map<std::string, int32_t> special_token_encoder_;
  absl::flat_hash_map<int32_t, std::string> special_token_decoder_;
  std::unique_ptr<re2::RE2> special_token_regex_;
  std::vector<int32_t> prefix_token_ids_;
  TokenizerHandle handle_ = nullptr;
};

}  // namespace xllm
