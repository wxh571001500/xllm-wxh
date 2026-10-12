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

#include "kernels/mlu/chunk_kda_pcp.h"

#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <cstdint>

#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {
namespace {

constexpr int32_t kDim = 128;
constexpr int32_t kInverseBlock = 16;
constexpr char kPreparePath[] =
    "xllm.core.kernels.mlu.triton_kernel.chunk_kda_fwd";
constexpr char kPcpPath[] = "xllm.core.kernels.mlu.triton_kernel.chunk_kda_pcp";

}  // namespace

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
                                          int64_t chunk_size) {
  CHECK(chunk_size == 16 || chunk_size == 64);
  CHECK_EQ(q.dim(), 4);
  CHECK_EQ(q.size(0), 1);
  CHECK_EQ(q.size(3), kDim);
  CHECK_EQ(k.sizes(), q.sizes());
  CHECK_EQ(v.sizes(), q.sizes());
  CHECK_EQ(raw_gate.numel(), q.numel());
  CHECK_EQ(raw_beta.numel(), q.size(1) * q.size(2));
  CHECK_EQ(cu_seqlens.dim(), 1);
  CHECK_EQ(chunk_indices.size(1), 2);

  const int64_t chunks = chunk_indices.size(0);
  const int64_t heads = q.size(2);
  const int64_t sequences = cu_seqlens.size(0) - 1;
  const int32_t block = static_cast<int32_t>(chunk_size);
  const torch::TensorOptions fp32 = q.options().dtype(torch::kFloat32);
  ChunkKDAPcpPrepared prepared;
  prepared.v = v;
  prepared.chunks = chunks;
  prepared.heads = heads;
  prepared.sequences = sequences;
  prepared.chunk_size = block;
  prepared.summary = torch::empty({sequences, heads, 2 * kDim, kDim}, fp32);
  if (chunks == 0) {
    prepared.summary.zero_();
    prepared.summary.slice(/*dim=*/2, kDim, 2 * kDim)
        .copy_(torch::eye(kDim, fp32));
    return prepared;
  }

  torch::Tensor lower_inverse =
      torch::empty({chunks, heads, block, block}, fp32);
  torch::Tensor aq = torch::empty_like(lower_inverse);
  torch::Tensor w = torch::empty({chunks, heads, block, kDim}, fp32);
  torch::Tensor u = torch::empty({chunks, heads, kDim, block}, fp32);
  torch::Tensor qg = torch::empty_like(w);
  torch::Tensor kg = torch::empty_like(w);
  torch::Tensor normalized_q = torch::empty_like(w);
  torch::Tensor normalized_k = torch::empty_like(w);
  torch::Tensor cumulative_gate = torch::empty_like(w);
  torch::Tensor gate_cumsum = torch::empty({chunks, heads, kDim}, fp32);
  const torch::Tensor cu = cu_seqlens.contiguous().to(torch::kInt32);
  const torch::Tensor indices = chunk_indices.contiguous().to(torch::kInt32);
  const torch::Tensor q_work = q.contiguous();
  const torch::Tensor k_work = k.contiguous();
  const torch::Tensor v_work = v.contiguous();
  const torch::Tensor beta_work =
      raw_beta.contiguous().view({q.size(1), heads});
  const torch::Tensor gate_work =
      raw_gate.contiguous().view({q.size(1), heads, kDim});
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  const uint32_t prepare_jobs = static_cast<uint32_t>(chunks * heads);
  using xllm::triton_jit::JITKernel;
  JITKernel::get(kPreparePath, "tmo_chunk_kda_optimized_raw_gate_kernel")
      .launch(static_cast<void*>(queue),
              {prepare_jobs, 1, 1},
              {/*num_warps=*/1, /*num_stages=*/4},
              v_work,
              beta_work,
              w,
              u,
              qg,
              kg,
              gate_work,
              a_log,
              dt_bias,
              gate_lower_bound,
              gate_cumsum,
              q_work,
              k_work,
              normalized_q,
              normalized_k,
              cumulative_gate,
              cu,
              indices,
              /*chunk_base=*/int64_t{0},
              chunks,
              /*H=*/static_cast<int32_t>(heads),
              /*BT=*/block,
              /*D=*/kDim,
              /*BK=*/kDim);

  JITKernel::get(kPreparePath, "tmo_chunk_kda_kkt_kernel")
      .launch(static_cast<void*>(queue),
              {prepare_jobs, 1, 1},
              {/*num_warps=*/1, /*num_stages=*/5},
              normalized_q,
              normalized_k,
              cumulative_gate,
              beta_work,
              lower_inverse,
              aq,
              cu,
              indices,
              /*chunk_base=*/int64_t{0},
              chunks,
              /*H=*/static_cast<int32_t>(heads),
              /*BT=*/block,
              /*D=*/kDim,
              /*BC=*/kInverseBlock,
              /*BN=*/block,
              /*BK=*/kDim,
              /*RAW_BETA=*/1);

  JITKernel::get(kPreparePath, "tmo_chunk_kda_inverse_kernel")
      .launch(static_cast<void*>(queue),
              {prepare_jobs, 1, 1},
              {/*num_warps=*/1, /*num_stages=*/4},
              lower_inverse,
              chunks,
              /*H=*/static_cast<int32_t>(heads),
              /*BT=*/block,
              /*B0=*/kInverseBlock);

  const int32_t value_block = q.size(1) <= 128 ? 16 : (block == 16 ? 128 : 64);
  const uint32_t summary_jobs =
      static_cast<uint32_t>(sequences * heads * (kDim / value_block));
  JITKernel::get(kPcpPath, "kda_pcp_summary_kernel")
      .launch(static_cast<void*>(queue),
              {summary_jobs, 1, 1},
              {/*num_warps=*/1, /*num_stages=*/4},
              lower_inverse,
              w,
              u,
              kg,
              gate_cumsum,
              cu,
              prepared.summary,
              /*H=*/static_cast<int32_t>(heads),
              /*BT=*/block,
              /*D=*/kDim,
              /*BV=*/value_block);
  prepared.lower_inverse = lower_inverse;
  prepared.w = w;
  prepared.u = u;
  prepared.qg = qg;
  prepared.kg = kg;
  prepared.aq = aq;
  prepared.gate_cumsum = gate_cumsum;
  prepared.cu_seqlens = cu;
  prepared.chunk_indices = indices;
  return prepared;
}

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
                                    int64_t chunk_size) {
  return chunk_kda_pcp_prepare(q,
                               k,
                               v,
                               raw_gate,
                               a_log,
                               dt_bias,
                               gate_lower_bound,
                               raw_beta,
                               cu_seqlens,
                               chunk_indices,
                               chunk_size)
      .summary;
}

std::tuple<torch::Tensor, torch::Tensor> chunk_kda_pcp_replay(
    const ChunkKDAPcpPrepared& prepared,
    const torch::Tensor& initial_state) {
  if (prepared.chunks == 0) {
    return {torch::empty_like(prepared.v), initial_state};
  }
  torch::Tensor output = torch::empty_like(prepared.v);
  torch::Tensor final_state = torch::empty_like(initial_state);
  constexpr int32_t kValueBlock = 32;
  const uint32_t jobs =
      static_cast<uint32_t>(prepared.heads * (kDim / kValueBlock));
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  xllm::triton_jit::JITKernel::get(kPreparePath,
                                   "tmo_chunk_kda_optimized_state_kernel")
      .launch(static_cast<void*>(queue),
              {jobs, 1, 1},
              {/*num_warps=*/1, /*num_stages=*/99},
              prepared.lower_inverse,
              prepared.w,
              prepared.u,
              prepared.qg,
              prepared.kg,
              prepared.aq,
              prepared.gate_cumsum,
              initial_state,
              final_state,
              output,
              prepared.cu_seqlens,
              prepared.chunk_indices,
              /*chunk_base=*/int64_t{0},
              prepared.chunks,
              prepared.sequences,
              /*H=*/static_cast<int32_t>(prepared.heads),
              /*BT=*/prepared.chunk_size,
              /*D=*/kDim,
              /*BK=*/kDim,
              /*BV=*/kValueBlock);
  return {output, final_state};
}

torch::Tensor chunk_kda_pcp_merge(const torch::Tensor& gathered_summary,
                                  const torch::Tensor& prefix_state,
                                  int32_t cp_rank,
                                  int64_t local_token_count) {
  CHECK_EQ(gathered_summary.dim(), 5);
  CHECK_EQ(prefix_state.dim(), 4);
  CHECK_EQ(gathered_summary.size(1), prefix_state.size(0));
  CHECK_EQ(gathered_summary.size(2), prefix_state.size(1));
  CHECK_EQ(gathered_summary.size(3), 2 * kDim);
  CHECK_EQ(gathered_summary.size(4), kDim);
  CHECK_EQ(prefix_state.size(2), kDim);
  CHECK_EQ(prefix_state.size(3), kDim);
  CHECK_GE(cp_rank, 0);
  CHECK_LT(cp_rank, gathered_summary.size(0));
  CHECK_GE(local_token_count, 0);

  const int32_t value_block = local_token_count <= 128
                                  ? 32
                                  : (gathered_summary.size(0) >= 4 ? 128 : 64);
  const uint32_t jobs = static_cast<uint32_t>(
      prefix_state.size(0) * prefix_state.size(1) * (kDim / value_block));
  torch::Tensor local_initial = torch::empty_like(prefix_state);
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  xllm::triton_jit::JITKernel::get(kPcpPath, "kda_pcp_merge_kernel")
      .launch(static_cast<void*>(queue),
              {jobs, 1, 1},
              {/*num_warps=*/1, /*num_stages=*/4},
              gathered_summary,
              prefix_state,
              local_initial,
              /*H=*/static_cast<int32_t>(prefix_state.size(1)),
              /*N=*/static_cast<int32_t>(prefix_state.size(0)),
              /*D=*/kDim,
              /*P=*/static_cast<int32_t>(gathered_summary.size(0)),
              /*RANK=*/cp_rank,
              /*BV=*/value_block);
  return local_initial;
}

}  // namespace xllm::kernel::mlu
