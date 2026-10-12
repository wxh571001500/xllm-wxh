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

#include "embedding_output_builder.h"

namespace xllm {
namespace {

bool serialize_tensor(const torch::Tensor& tensor,
                      bool use_binary_encoding,
                      proto::Tensor* output,
                      std::string& binary_payload) {
  if (use_binary_encoding) {
    return util::torch_to_proto(tensor, output, binary_payload);
  }
  return util::torch_to_proto(tensor, output);
}

}  // namespace

EmbeddingOutputBuilder::EmbeddingOutputBuilder(
    bool embedding_use_binary_encoding,
    bool metadata_use_binary_encoding)
    : embedding_use_binary_encoding_(embedding_use_binary_encoding),
      metadata_use_binary_encoding_(metadata_use_binary_encoding) {};

EmbeddingOutputBuilder::~EmbeddingOutputBuilder() {};

bool EmbeddingOutputBuilder::build_repeated_embedding_output(
    const std::vector<EmbeddingOutput>& in_embeddings,
    google::protobuf::RepeatedPtrField<xllm::proto::Embedding>& out_embeddings,
    std::string& binary_payload) {
  for (const auto& in_embedding : in_embeddings) {
    xllm::proto::Embedding* out_embedding = out_embeddings.Add();
    if (!build_embedding_output(in_embedding, *out_embedding, binary_payload)) {
      return false;
    }
  }
  return true;
}

bool EmbeddingOutputBuilder::build_embedding_output(
    const EmbeddingOutput& in_embedding,
    xllm::proto::Embedding& out_embedding,
    std::string& binary_payload) {
  if (!serialize_tensor(in_embedding.embedding,
                        embedding_use_binary_encoding_,
                        out_embedding.mutable_embedding(),
                        binary_payload)) {
    return false;
  }

  out_embedding.set_hash_key(in_embedding.hash_key);
  auto* meta_map = out_embedding.mutable_metadata();
  for (const auto& [key, tensor] : in_embedding.metadata) {
    xllm::proto::Tensor metadata_tensor;
    if (!serialize_tensor(tensor,
                          metadata_use_binary_encoding_,
                          &metadata_tensor,
                          binary_payload)) {
      return false;
    }
    (*meta_map)[key] = std::move(metadata_tensor);
  }
  return true;
};

};  // namespace xllm
