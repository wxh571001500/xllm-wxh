/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

torch::Tensor x_flash_attention_decode_out(const torch::Tensor& query,
                                           const torch::Tensor& key,
                                           const torch::Tensor& value,
                                           const torch::Tensor& block_table,
                                           const torch::Tensor& query_ends,
                                           const torch::Tensor& kv_lengths,
                                           const torch::Tensor& kv_starts,
                                           double scale,
                                           torch::Tensor& output) {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(query.size(2), 128) << "XFIA decode currently supports head_dim=128";
  CHECK(query.scalar_type() == torch::kBFloat16 ||
        query.scalar_type() == torch::kFloat16);
  CHECK_EQ(key.dim(), 4);
  CHECK_EQ(key.size(1), 128) << "XFIA decode currently supports block_size=128";
  CHECK_EQ(key.size(3), query.size(2));
  CHECK_EQ(query.size(1) % key.size(2), 0);
  CHECK(value.sizes() == key.sizes());
  CHECK(output.sizes() == query.sizes());
  CHECK_EQ(block_table.dim(), 2);
  // Decode expands each proposal token into an independent query row.
  CHECK_EQ(block_table.size(0), query.size(0));
  CHECK_EQ(query_ends.dim(), 1);
  CHECK_EQ(kv_lengths.dim(), 1);
  CHECK_EQ(query_ends.numel(), query.size(0));
  CHECK_EQ(kv_lengths.numel(), query.size(0));
  // One logical KV-window start per query row.
  CHECK_EQ(kv_starts.numel(), query.size(0));
  for (const torch::Tensor& tensor : {query,
                                      key,
                                      value,
                                      block_table,
                                      query_ends,
                                      kv_lengths,
                                      kv_starts,
                                      output}) {
    CHECK(tensor.is_contiguous());
    CHECK(tensor.device() == query.device());
  }
  for (const torch::Tensor& tensor : {key, value, output}) {
    CHECK(tensor.scalar_type() == query.scalar_type());
  }
  for (const torch::Tensor& tensor :
       {block_table, query_ends, kv_lengths, kv_starts}) {
    CHECK(tensor.scalar_type() == torch::kInt32);
  }
  const std::optional<torch::Tensor> mask = std::nullopt;
  // The original ABI requires extra_tiling, but TND no-mask never reads it.
  // An empty tensor satisfies that contract without allocating device storage.
  const torch::Tensor extra_tiling = torch::empty({0}, query_ends.options());
  char layout[] = "TND";
  int64_t query_heads = query.size(1);
  int64_t kv_heads = key.size(2);
  EXEC_NPU_CMD(aclnnXFlashAttentionInfer,
               query,
               key,
               value,
               mask,
               block_table,
               query_ends,
               kv_lengths,
               extra_tiling,
               kv_starts,
               layout,
               query_heads,
               kv_heads,
               scale,
               output);
  return output;
}

}  // namespace xllm::kernel::npu
