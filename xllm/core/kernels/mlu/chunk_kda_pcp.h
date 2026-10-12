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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace xllm::kernel::mlu {

struct ChunkKDAPcpPrepared final {
  torch::Tensor summary;
  torch::Tensor v;
  torch::Tensor lower_inverse;
  torch::Tensor w;
  torch::Tensor u;
  torch::Tensor qg;
  torch::Tensor kg;
  torch::Tensor aq;
  torch::Tensor gate_cumsum;
  torch::Tensor cu_seqlens;
  torch::Tensor chunk_indices;
  int64_t chunks = 0;
  int64_t heads = 0;
  int64_t sequences = 0;
  int32_t chunk_size = 0;
};

ChunkKDAPcpPrepared chunk_kda_pcp_prepare(const torch::Tensor& q,
                                          const torch::Tensor& k,
                                          const torch::Tensor& v,
                                          const torch::Tensor& raw_gate,
                                          const torch::Tensor& a_log,
                                          const torch::Tensor& dt_bias,
                                          float gate_lower_bound,
                                          const torch::Tensor& raw_beta,
                                          const torch::Tensor& cu_seqlens,
                                          const torch::Tensor& chunk_indices,
                                          int64_t chunk_size);

std::tuple<torch::Tensor, torch::Tensor> chunk_kda_pcp_replay(
    const ChunkKDAPcpPrepared& prepared,
    const torch::Tensor& initial_state);

// Computes one [E; M] affine summary per packed sequence and local head.
// Q/K/V are the post-convolution BF16 tensors used by the ordinary KDA path.
// The caller owns communication and must keep the packed sequence order stable.
torch::Tensor chunk_kda_pcp_summary(const torch::Tensor& q,
                                    const torch::Tensor& k,
                                    const torch::Tensor& v,
                                    const torch::Tensor& raw_gate,
                                    const torch::Tensor& a_log,
                                    const torch::Tensor& dt_bias,
                                    float gate_lower_bound,
                                    const torch::Tensor& raw_beta,
                                    const torch::Tensor& cu_seqlens,
                                    const torch::Tensor& chunk_indices,
                                    int64_t chunk_size);

// gathered_summary is CP-rank-major [P,N,H,2D,D].
torch::Tensor chunk_kda_pcp_merge(const torch::Tensor& gathered_summary,
                                  const torch::Tensor& prefix_state,
                                  int32_t cp_rank,
                                  int64_t local_token_count);

}  // namespace xllm::kernel::mlu
