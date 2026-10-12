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

#include <framework/core/device.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "kernels/mlu/chunk_kda.h"

namespace xllm::kernel::mlu {
namespace {

struct SummaryCase final {
  int64_t chunk_size = 0;
  std::vector<int64_t> lengths;
};

struct MergeCase final {
  int64_t world_size = 0;
  int32_t cp_rank = 0;
  int64_t local_token_count = 0;
};

torch::Tensor make_cu_seqlens(const std::vector<int64_t>& lengths,
                              const torch::TensorOptions& int32) {
  std::vector<int32_t> cu_seqlens;
  cu_seqlens.reserve(lengths.size() + 1);
  cu_seqlens.emplace_back(0);
  for (const int64_t length : lengths) {
    cu_seqlens.emplace_back(cu_seqlens.back() + static_cast<int32_t>(length));
  }
  return torch::tensor(cu_seqlens, int32);
}

// Sequence-major {sequence, local chunk} rows, matching the packed layout.
torch::Tensor make_chunk_indices(const std::vector<int64_t>& lengths,
                                 int64_t chunk_size,
                                 const torch::TensorOptions& int32) {
  std::vector<int32_t> chunk_indices;
  for (size_t sequence = 0; sequence < lengths.size(); ++sequence) {
    const int64_t chunks = (lengths[sequence] + chunk_size - 1) / chunk_size;
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
      chunk_indices.emplace_back(static_cast<int32_t>(sequence));
      chunk_indices.emplace_back(static_cast<int32_t>(chunk));
    }
  }
  return torch::tensor(chunk_indices, int32).view({-1, 2});
}

// Chunk-by-chunk FP32 recurrence of the [E; M] affine summary on CPU, built
// from the prepared factors instead of the summary kernel.
torch::Tensor reference_summary(const ChunkKDAPcpPrepared& prepared,
                                const std::vector<int64_t>& lengths) {
  const torch::Tensor inverse = prepared.lower_inverse.cpu();
  const torch::Tensor w = prepared.w.cpu();
  const torch::Tensor u = prepared.u.cpu();
  const torch::Tensor kg = prepared.kg.cpu();
  const torch::Tensor gate = prepared.gate_cumsum.cpu();
  const int64_t chunk_size = prepared.chunk_size;
  const int64_t dim = w.size(-1);
  std::vector<torch::Tensor> summaries;
  summaries.reserve(lengths.size());
  int64_t chunk = 0;
  for (const int64_t length : lengths) {
    torch::Tensor ext = torch::zeros({prepared.heads, dim, dim});
    torch::Tensor transition =
        torch::eye(dim).expand({prepared.heads, dim, dim});
    for (int64_t offset = 0; offset < length; offset += chunk_size, ++chunk) {
      const int64_t valid = std::min(chunk_size, length - offset);
      const torch::Tensor w_transposed =
          w[chunk].narrow(/*dim=*/1, /*start=*/0, valid).transpose(1, 2);
      const torch::Tensor kg_chunk =
          kg[chunk].narrow(/*dim=*/1, /*start=*/0, valid);
      const torch::Tensor inverse_transposed =
          torch::tril(inverse[chunk]
                          .narrow(/*dim=*/1, /*start=*/0, valid)
                          .narrow(/*dim=*/2, /*start=*/0, valid))
              .transpose(1, 2);
      const torch::Tensor u_chunk =
          u[chunk].narrow(/*dim=*/2, /*start=*/0, valid);
      const torch::Tensor decay = gate[chunk].exp().unsqueeze(/*dim=*/1);
      ext = ext * decay +
            torch::matmul(
                torch::matmul(u_chunk - torch::matmul(ext, w_transposed),
                              inverse_transposed),
                kg_chunk);
      transition =
          transition * decay -
          torch::matmul(torch::matmul(torch::matmul(transition, w_transposed),
                                      inverse_transposed),
                        kg_chunk);
    }
    summaries.emplace_back(torch::cat({ext, transition}, /*dim=*/1));
  }
  return torch::stack(summaries);
}

// Applies S <- S * M_p + E_p for every CP rank before cp_rank on CPU.
torch::Tensor reference_merge(const torch::Tensor& gathered,
                              const torch::Tensor& prefix,
                              int32_t cp_rank) {
  const int64_t dim = prefix.size(-1);
  torch::Tensor state = prefix;
  for (int32_t previous = 0; previous < cp_rank; ++previous) {
    state = torch::matmul(state,
                          gathered[previous].slice(/*dim=*/2, dim, 2 * dim)) +
            gathered[previous].slice(/*dim=*/2, /*start=*/0, dim);
  }
  return state;
}

TEST(ChunkKDAPcpTest, AffineSummaryReproducesNonzeroInitialState) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20260923);
  constexpr int64_t kHeads = 8;
  constexpr int64_t kDim = 128;
  const int64_t chunk_size = kda_prefill_chunk_size(kHeads, true);
  const int64_t tokens = chunk_size + 1;
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions fp32 = options.dtype(torch::kFloat32);
  const torch::Tensor q = (torch::randn({1, tokens, kHeads, kDim}, fp32) * 0.1f)
                              .to(torch::kBFloat16);
  const torch::Tensor k = (torch::randn({1, tokens, kHeads, kDim}, fp32) * 0.1f)
                              .to(torch::kBFloat16);
  const torch::Tensor v = (torch::randn({1, tokens, kHeads, kDim}, fp32) * 0.1f)
                              .to(torch::kBFloat16);
  const torch::Tensor raw_gate =
      (torch::randn({tokens, kHeads, kDim}, fp32) - 2.0f).to(torch::kBFloat16);
  const torch::Tensor raw_beta = torch::randn({tokens, kHeads}, options);
  const torch::Tensor a_log = torch::zeros({kHeads}, fp32);
  const torch::Tensor dt_bias = torch::zeros({kHeads * kDim}, fp32);
  const torch::Tensor initial =
      torch::randn({1, kHeads, kDim, kDim}, fp32) * 0.01f;
  const torch::Tensor cu = torch::tensor({0, static_cast<int32_t>(tokens)},
                                         options.dtype(torch::kInt32));
  const torch::Tensor indices =
      torch::tensor({{0, 0}, {0, 1}}, options.dtype(torch::kInt32));

  const ChunkKDAPcpPrepared prepared = chunk_kda_pcp_prepare(q,
                                                             k,
                                                             v,
                                                             raw_gate,
                                                             a_log,
                                                             dt_bias,
                                                             -5.0f,
                                                             raw_beta,
                                                             cu,
                                                             indices,
                                                             chunk_size);
  const torch::Tensor summary = prepared.summary;
  const torch::Tensor gathered = summary.unsqueeze(0).repeat({2, 1, 1, 1, 1});
  const torch::Tensor actual =
      chunk_kda_pcp_merge(gathered, initial, 1, tokens);
  const torch::Tensor expected =
      torch::matmul(initial, summary.slice(2, kDim, 2 * kDim)) +
      summary.slice(2, 0, kDim);
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(actual, expected, 1e-3, 2e-5));

  ChunkKDA ordinary(kHeads);
  auto [output, final_state] = ordinary->forward_raw_gate(q,
                                                          k,
                                                          v,
                                                          raw_gate,
                                                          a_log,
                                                          dt_bias,
                                                          -5.0f,
                                                          raw_beta,
                                                          initial,
                                                          cu,
                                                          indices,
                                                          true,
                                                          true);
  auto [replayed_output, replayed_state] =
      chunk_kda_pcp_replay(prepared, initial);
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(expected, final_state, 1e-3, 2e-5));
  EXPECT_TRUE(torch::allclose(replayed_output, output, 5e-3, 2e-5));
  EXPECT_TRUE(torch::allclose(replayed_state, final_state, 1e-3, 2e-5));
}

TEST(ChunkKDAPcpTest, EmptyRankKeepsPrefixState) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  constexpr int64_t kHeads = 8;
  constexpr int64_t kDim = 128;
  const torch::TensorOptions bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions fp32 = bf16.dtype(torch::kFloat32);
  const torch::TensorOptions int32 = bf16.dtype(torch::kInt32);
  const torch::Tensor empty_qkv = torch::empty({1, 0, kHeads, kDim}, bf16);
  const torch::Tensor empty_beta = torch::empty({0, kHeads}, bf16);
  const torch::Tensor cu = torch::tensor({0, 0, 0}, int32);
  const torch::Tensor indices = torch::empty({0, 2}, int32);
  const torch::Tensor summary =
      chunk_kda_pcp_summary(empty_qkv,
                            empty_qkv,
                            empty_qkv,
                            empty_qkv,
                            torch::zeros({kHeads}, fp32),
                            torch::zeros({kHeads * kDim}, fp32),
                            -5.0f,
                            empty_beta,
                            cu,
                            indices,
                            /*chunk_size=*/64);
  const torch::Tensor prefix = torch::randn({2, kHeads, kDim, kDim}, fp32);
  const torch::Tensor merged =
      chunk_kda_pcp_merge(summary.unsqueeze(0).repeat({2, 1, 1, 1, 1}),
                          prefix,
                          /*cp_rank=*/1,
                          /*local_token_count=*/0);
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(merged, prefix, 1e-5, 1e-5));
}

TEST(ChunkKDAPcpTest, PackedSequenceSummaryMatchesFp32Reference) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(2026);
  constexpr int64_t kHeads = 2;
  constexpr int64_t kDim = 128;
  const torch::TensorOptions fp32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const torch::TensorOptions bf16 = fp32.dtype(torch::kBFloat16);
  const torch::TensorOptions int32 = fp32.dtype(torch::kInt32);
  const torch::Tensor a_log = torch::zeros({kHeads}, fp32);
  const torch::Tensor dt_bias = torch::zeros({kHeads * kDim}, fp32);
  // Every case ends a sequence on a partial chunk. The first two stay within
  // the short-shard value block and the last two take the long-shard one.
  const std::vector<SummaryCase> cases = {
      {/*chunk_size=*/16, /*lengths=*/{17, 31}},
      {/*chunk_size=*/64, /*lengths=*/{65, 1}},
      {/*chunk_size=*/16, /*lengths=*/{130, 47}},
      {/*chunk_size=*/64, /*lengths=*/{65, 130}}};

  for (const SummaryCase& test_case : cases) {
    const torch::Tensor cu = make_cu_seqlens(test_case.lengths, int32);
    const int64_t tokens = cu[-1].item<int64_t>();
    const torch::Tensor q =
        (torch::randn({1, tokens, kHeads, kDim}, fp32) * 0.1f).to(bf16);
    const torch::Tensor k =
        (torch::randn({1, tokens, kHeads, kDim}, fp32) * 0.1f).to(bf16);
    const torch::Tensor v =
        (torch::randn({1, tokens, kHeads, kDim}, fp32) * 0.1f).to(bf16);
    const torch::Tensor gate =
        (torch::randn({tokens, kHeads, kDim}, fp32) - 2.0f).to(bf16);
    const torch::Tensor beta = torch::randn({tokens, kHeads}, bf16);
    const ChunkKDAPcpPrepared prepared = chunk_kda_pcp_prepare(
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        /*gate_lower_bound=*/-5.0f,
        beta,
        cu,
        make_chunk_indices(test_case.lengths, test_case.chunk_size, int32),
        test_case.chunk_size);
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::allclose(prepared.summary.cpu(),
                                reference_summary(prepared, test_case.lengths),
                                /*rtol=*/1e-3,
                                /*atol=*/2e-5))
        << "chunk_size=" << test_case.chunk_size << " tokens=" << tokens;
  }
}

TEST(ChunkKDAPcpTest, MergeComposesEveryPrecedingRankSummary) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(2026);
  constexpr int64_t kSequences = 2;
  constexpr int64_t kHeads = 2;
  constexpr int64_t kDim = 128;
  // local_token_count and world_size select the merge value block.
  const std::vector<MergeCase> cases = {
      {/*world_size=*/3, /*cp_rank=*/2, /*local_token_count=*/17},
      {/*world_size=*/3, /*cp_rank=*/2, /*local_token_count=*/4096},
      {/*world_size=*/4, /*cp_rank=*/3, /*local_token_count=*/4096}};

  for (const MergeCase& test_case : cases) {
    torch::Tensor gathered =
        torch::randn(
            {test_case.world_size, kSequences, kHeads, 2 * kDim, kDim}) *
        0.01f;
    gathered.slice(/*dim=*/3, kDim, 2 * kDim) += torch::eye(kDim) * 0.9f;
    // Summaries of the local and later ranks must not reach the result.
    gathered.slice(/*dim=*/0, test_case.cp_rank, test_case.world_size)
        .fill_(std::numeric_limits<float>::quiet_NaN());
    const torch::Tensor prefix =
        torch::randn({kSequences, kHeads, kDim, kDim}) * 0.01f;
    const torch::Tensor actual =
        chunk_kda_pcp_merge(gathered.to(device),
                            prefix.to(device),
                            test_case.cp_rank,
                            test_case.local_token_count);
    torch_mlu::synchronize();
    EXPECT_TRUE(
        torch::allclose(actual.cpu(),
                        reference_merge(gathered, prefix, test_case.cp_rank),
                        /*rtol=*/1e-3,
                        /*atol=*/2e-5))
        << "world_size=" << test_case.world_size
        << " cp_rank=" << test_case.cp_rank
        << " local_token_count=" << test_case.local_token_count;
  }
}

TEST(ChunkKDAPcpTest, LongShardsMatchUnshardedKDA) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20260923);
  constexpr int64_t kHeads = 8;
  constexpr int64_t kDim = 128;
  const int64_t chunk_size = kda_prefill_chunk_size(kHeads, true);
  const torch::TensorOptions fp32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const torch::TensorOptions bf16 = fp32.dtype(torch::kBFloat16);
  const torch::TensorOptions int32 = fp32.dtype(torch::kInt32);
  const torch::Tensor a_log = torch::zeros({kHeads}, fp32);
  const torch::Tensor dt_bias = torch::zeros({kHeads * kDim}, fp32);
  ChunkKDA ordinary(kHeads);

  for (const int64_t tokens : {int64_t{8192}, int64_t{16384}}) {
    const torch::Tensor q =
        (torch::randn({1, tokens, kHeads, kDim}, fp32) * 0.1f).to(bf16);
    const torch::Tensor k =
        (torch::randn({1, tokens, kHeads, kDim}, fp32) * 0.1f).to(bf16);
    const torch::Tensor v =
        (torch::randn({1, tokens, kHeads, kDim}, fp32) * 0.1f).to(bf16);
    const torch::Tensor gate =
        (torch::randn({tokens, kHeads, kDim}, fp32) - 2.0f).to(bf16);
    const torch::Tensor beta = torch::randn({tokens, kHeads}, bf16);
    const torch::Tensor initial =
        torch::randn({1, kHeads, kDim, kDim}, fp32) * 0.01f;
    const torch::Tensor cu =
        torch::tensor({0, static_cast<int32_t>(tokens)}, int32);
    const torch::Tensor indices =
        torch::stack({torch::zeros({tokens / chunk_size}, int32),
                      torch::arange(tokens / chunk_size, int32)},
                     /*dim=*/1);
    auto [expected_output, expected_state] = ordinary->forward_raw_gate(q,
                                                                        k,
                                                                        v,
                                                                        gate,
                                                                        a_log,
                                                                        dt_bias,
                                                                        -5.0f,
                                                                        beta,
                                                                        initial,
                                                                        cu,
                                                                        indices,
                                                                        true,
                                                                        true);
    const int64_t shard_tokens = tokens / 2;
    const torch::Tensor shard_cu =
        torch::tensor({0, static_cast<int32_t>(shard_tokens)}, int32);
    const torch::Tensor shard_indices =
        torch::stack({torch::zeros({shard_tokens / chunk_size}, int32),
                      torch::arange(shard_tokens / chunk_size, int32)},
                     /*dim=*/1);
    std::vector<ChunkKDAPcpPrepared> shards;
    for (int64_t rank = 0; rank < 2; ++rank) {
      const int64_t offset = rank * shard_tokens;
      shards.emplace_back(
          chunk_kda_pcp_prepare(q.narrow(1, offset, shard_tokens),
                                k.narrow(1, offset, shard_tokens),
                                v.narrow(1, offset, shard_tokens),
                                gate.narrow(0, offset, shard_tokens),
                                a_log,
                                dt_bias,
                                -5.0f,
                                beta.narrow(0, offset, shard_tokens),
                                shard_cu,
                                shard_indices,
                                chunk_size));
    }
    const torch::Tensor gathered =
        torch::stack({shards[0].summary, shards[1].summary});
    const torch::Tensor second_initial =
        chunk_kda_pcp_merge(gathered, initial, 1, shard_tokens);
    auto [first_output, first_state] = chunk_kda_pcp_replay(shards[0], initial);
    auto [second_output, second_state] =
        chunk_kda_pcp_replay(shards[1], second_initial);
    const torch::Tensor actual_output =
        torch::cat({first_output, second_output}, /*dim=*/1);
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::allclose(actual_output, expected_output, 5e-3, 2e-5))
        << "tokens=" << tokens;
    EXPECT_TRUE(torch::allclose(second_state, expected_state, 1e-3, 2e-5))
        << "tokens=" << tokens;
  }
}

}  // namespace
}  // namespace xllm::kernel::mlu
