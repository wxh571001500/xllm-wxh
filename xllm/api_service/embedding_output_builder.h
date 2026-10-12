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

#include "core/common/message.h"
#include "core/common/types.h"
#include "core/framework//request/request_output.h"
#include "core/util/utils.h"
#include "embedding.pb.h"
#include "tensor.pb.h"

namespace xllm {
class EmbeddingOutputBuilder {
 public:
  EmbeddingOutputBuilder(bool embedding_use_binary_encoding,
                         bool metadata_use_binary_encoding);
  ~EmbeddingOutputBuilder();
  bool build_repeated_embedding_output(
      const std::vector<EmbeddingOutput>& in_embeddings,
      google::protobuf::RepeatedPtrField<xllm::proto::Embedding>&
          out_embeddings,
      std::string& binary_payload);
  bool build_embedding_output(const EmbeddingOutput& in_embedding,
                              xllm::proto::Embedding& out_embedding,
                              std::string& binary_payload);

 private:
  bool embedding_use_binary_encoding_;
  bool metadata_use_binary_encoding_;
};

}  // namespace xllm
