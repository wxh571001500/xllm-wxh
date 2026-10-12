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
#include <torch/torch.h>

#include <cstdint>

namespace xllm::kernel::mlu {

struct KPoolSelection final {
  torch::Tensor physical_slots;
  torch::Tensor context_lens;
};

// Inputs are request-major, with ascending consecutive active positions.
// Cache storage is framework-owned; padding positions never mutate state.
void update_kpool(const torch::Tensor& k,
                  const torch::Tensor& gate,
                  const torch::Tensor& ape,
                  const torch::Tensor& hadamard,
                  torch::Tensor& cache,
                  torch::Tensor& tail,
                  const torch::Tensor& tail_ids,
                  const torch::Tensor& table,
                  const torch::Tensor& positions,
                  const torch::Tensor& rows,
                  const torch::Tensor& starts,
                  int64_t block_size,
                  int64_t pool_size,
                  bool decode = false);

// Writes caller-owned FP32 workspace. Positions are causal, zero-based token
// positions; -1 masks the whole row. Rows map queries to request page tables.
// Optional device query_starts describes request-major verify spans bounded by
// four queries, including shorter live spans during graph replay. Without it,
// scoring retains the independent-query path. Rows must name the same
// page-table request throughout each span; the model's batch metadata
// guarantees this.
void score_kpool(const torch::Tensor& query,
                 const torch::Tensor& weights,
                 const torch::Tensor& cache,
                 const torch::Tensor& table,
                 const torch::Tensor& positions,
                 const torch::Tensor& rows,
                 torch::Tensor& scores,
                 int64_t block_size,
                 int64_t pool_size,
                 double scale,
                 const torch::Tensor& query_starts = torch::Tensor());

// Native torch.topk tie ordering; nonfinite selections and missing columns -1.
torch::Tensor select_kpool(const torch::Tensor& scores, int64_t count);

KPoolSelection expand_kpool(const torch::Tensor& ids,
                            const torch::Tensor& positions,
                            const torch::Tensor& rows,
                            const torch::Tensor& table,
                            int64_t block_size,
                            int64_t token_budget,
                            int64_t pool_size,
                            bool always_tail);
}  // namespace xllm::kernel::mlu
