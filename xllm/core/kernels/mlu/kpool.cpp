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

#include "kernels/mlu/kpool.h"

#include <glog/logging.h>

#include <algorithm>

#include "framework/core/MLUStream.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {
namespace {
constexpr char kUpdate[] =
    "xllm.core.kernels.mlu.triton_kernel.glm5_next_kpool";
constexpr char kScore[] =
    "xllm.core.kernels.mlu.triton_kernel.glm5_next_kpool_select";
constexpr char kExpand[] =
    "xllm.core.kernels.mlu.triton_kernel.glm5_next_kpool_expand";
int32_t tile_size(int64_t n) {
  int32_t result = 1;
  while (result < n) {
    result *= 2;
  }
  return result;
}
// Preserve the old integer addressing fast path without restricting layouts.
torch::Tensor divide(const torch::Tensor& input, int64_t divisor) {
  if ((divisor & (divisor - 1)) != 0) {
    return torch::floor_divide(input, divisor);
  }
  int64_t shift = 0;
  while (divisor > 1) {
    divisor >>= 1;
    ++shift;
  }
  return torch::bitwise_right_shift(input, shift);
}

torch::Tensor remainder(const torch::Tensor& input, int64_t divisor) {
  return (divisor & (divisor - 1)) == 0 ? torch::bitwise_and(input, divisor - 1)
                                        : torch::remainder(input, divisor);
}
}  // namespace

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
                  bool decode) {
  CHECK_GT(pool_size, 0);
  CHECK_EQ(block_size % pool_size, 0);
  CHECK_EQ(k.dim(), 2);
  CHECK_EQ(k.scalar_type(), torch::kBFloat16);
  CHECK_EQ(gate.scalar_type(), torch::kBFloat16);
  CHECK_EQ(cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(tail.scalar_type(), torch::kBFloat16);
  CHECK_EQ(ape.dim(), 2);
  CHECK_EQ(ape.size(0), pool_size);
  CHECK_EQ(ape.size(1), k.size(1));
  CHECK_EQ(hadamard.dim(), 2);
  CHECK_EQ(hadamard.size(0), k.size(1));
  CHECK_EQ(hadamard.size(1), k.size(1));
  CHECK_EQ(table.dim(), 2);
  CHECK_EQ(table.size(0), tail_ids.numel());
  CHECK(k.sizes() == gate.sizes());
  CHECK_EQ(tail.dim(), 4);
  CHECK_EQ(tail.size(1), 2);
  CHECK_GE(tail.size(2), pool_size);
  CHECK_EQ(tail.size(3), k.size(1));
  CHECK_EQ(starts.numel(), tail_ids.numel() + 1);
  CHECK_EQ(positions.numel(), k.size(0));
  CHECK_EQ(rows.numel(), k.size(0));
  CHECK(cache.is_contiguous());
  CHECK(tail.is_contiguous());
  if (k.size(0) == 0) {
    return;
  }
  const torch::Tensor raw = k.contiguous();
  const torch::Tensor gates = gate.contiguous();
  const torch::Tensor a = ape.to(k.device(), torch::kFloat32).contiguous();
  const torch::Tensor h = hadamard.contiguous();
  const torch::Tensor ids = tail_ids.contiguous();
  const torch::Tensor pages = table.contiguous();
  const torch::Tensor pos = positions.contiguous();
  const torch::Tensor batch = rows.contiguous();
  const torch::Tensor offsets = starts.contiguous();
  const int32_t p = static_cast<int32_t>(pool_size);
  const int32_t t = static_cast<int32_t>(tail.size(2));
  const int32_t d = static_cast<int32_t>(k.size(1));
  const int32_t pb = static_cast<int32_t>(block_size / pool_size);
  void* queue = static_cast<void*>(torch_mlu::getCurMLUStream());
  // Keep the fused update in its original NRAM range. Larger pools use the
  // same ring protocol through bounded gathers and Torch pooling.
  if (decode && pool_size * d <= 8192) {
    triton_jit::JITKernel::get(kUpdate, "kpool_decode_update")
        .launch(queue,
                {static_cast<uint32_t>(ids.numel()), 1, 1},
                {1, 1},
                raw,
                gates,
                a,
                h,
                cache,
                tail,
                ids,
                pages,
                pos,
                offsets,
                raw.stride(0),
                gates.stride(0),
                pages.stride(0),
                p,
                t,
                d,
                tile_size(p),
                tile_size(d),
                pb);
    return;
  }
  // Bound gather, FP32 softmax and projection scratch independently of the
  // input length. Slots stay static in shape, including incomplete pools.
  constexpr int64_t kPoolBytes = 16 * 1024 * 1024;
  const int64_t tile = std::max<int64_t>(
      1, std::min<int64_t>(1024, kPoolBytes / (pool_size * d * 24)));
  for (int64_t offset = 0; offset < raw.size(0); offset += tile) {
    const int64_t count = std::min(tile, raw.size(0) - offset);
    torch::Tensor keys = torch::empty({count, p, d}, raw.options());
    torch::Tensor logits = torch::empty_like(keys);
    torch::Tensor slots =
        torch::empty({count}, ids.options().dtype(torch::kInt64));
    triton_jit::JITKernel::get(kUpdate, "gather_members")
        .launch(queue,
                {static_cast<uint32_t>(count),
                 static_cast<uint32_t>((p + 15) / 16),
                 1},
                {1, 1},
                raw,
                gates,
                tail,
                ids,
                pages,
                pos,
                batch,
                offsets,
                keys,
                logits,
                slots,
                offset,
                raw.size(0),
                pages.size(1),
                cache.size(0),
                tail.size(0),
                p,
                t,
                d,
                tile_size(std::min(p, 16)),
                tile_size(d),
                pb);
    const torch::Tensor probabilities =
        torch::softmax(logits.to(torch::kFloat32) + a.unsqueeze(0), /*dim=*/1);
    const torch::Tensor pooled = (keys.to(torch::kFloat32) * probabilities)
                                     .sum(/*dim=*/1)
                                     .to(torch::kBFloat16)
                                     .to(torch::kFloat32);
    const torch::Tensor values =
        torch::matmul(pooled, h.transpose(0, 1)).to(torch::kBFloat16);
    triton_jit::JITKernel::get(kUpdate, "write_pools")
        .launch(queue,
                {static_cast<uint32_t>(count), 1, 1},
                {1, 1},
                values,
                slots,
                cache,
                d,
                tile_size(d));
  }
  // All old-tail reads complete before the ring is overwritten.
  triton_jit::JITKernel::get(kUpdate, "kpool_stash")
      .launch(queue,
              {static_cast<uint32_t>(ids.numel()),
               static_cast<uint32_t>((t + 15) / 16),
               1},
              {1, 1},
              raw,
              gates,
              tail,
              ids,
              pos,
              offsets,
              raw.stride(0),
              gates.stride(0),
              t,
              d,
              /*BLOCK_P=*/16,
              tile_size(d));
}

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
                 const torch::Tensor& query_starts) {
  CHECK_GT(pool_size, 0);
  CHECK_EQ(block_size % pool_size, 0);
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(weights.dim(), 2);
  CHECK_EQ(positions.numel(), query.size(0));
  CHECK_EQ(rows.numel(), query.size(0));
  CHECK_EQ(cache.size(-1), query.size(2));
  CHECK(cache.is_contiguous());
  CHECK_EQ(table.dim(), 2);
  CHECK_EQ(weights.size(0), query.size(0));
  CHECK_EQ(weights.size(1), query.size(1));
  CHECK_EQ(scores.size(0), query.size(0));
  CHECK_EQ(scores.scalar_type(), torch::kFloat32);
  CHECK_EQ(scores.stride(1), 1);
  if (scores.numel() == 0) {
    return;
  }
  const torch::Tensor q = query.contiguous();
  const torch::Tensor w = weights.contiguous();
  const torch::Tensor pages = table.contiguous();
  const torch::Tensor pos = positions.contiguous();
  const torch::Tensor batch = rows.contiguous();
  const bool four_query_verify =
      query_starts.defined() && query_starts.dim() == 1 &&
      query_starts.numel() == table.size(0) + 1 && table.size(0) > 0 &&
      (query_starts.scalar_type() == torch::kInt32 ||
       query_starts.scalar_type() == torch::kInt64) &&
      query_starts.device() == q.device() && q.size(0) == table.size(0) * 4 &&
      cache.dim() == 4 && cache.size(1) == 1 && cache.size(2) == 4 &&
      cache.size(3) == 128 && q.size(1) == 32 && q.size(2) == 128 &&
      block_size == 16 && pool_size == 4 &&
      q.scalar_type() == torch::kBFloat16 &&
      cache.scalar_type() == torch::kBFloat16 &&
      w.scalar_type() == torch::kFloat32;
  if (four_query_verify) {
    constexpr int32_t kPoolsPerTile = 128;
    constexpr int32_t kTilesPerProgram = 4;
    const torch::Tensor starts = query_starts.contiguous();
    triton_jit::JITKernel::get(
        "xllm.core.kernels.mlu.triton_kernel.glm5_next_kpool",
        "score_verify_pools")
        .launch(static_cast<void*>(torch_mlu::getCurMLUStream()),
                {static_cast<uint32_t>(table.size(0)),
                 static_cast<uint32_t>(
                     (scores.size(1) + kPoolsPerTile * kTilesPerProgram - 1) /
                     (kPoolsPerTile * kTilesPerProgram)),
                 1},
                {1, 1},
                q,
                w,
                cache,
                pages,
                pos,
                batch,
                starts,
                scores,
                q.size(0),
                scores.size(1),
                cache.size(0),
                pages.size(1),
                scores.stride(0),
                static_cast<float>(scale),
                kPoolsPerTile,
                kTilesPerProgram);
    return;
  }
  constexpr int32_t kTile = 128;
  constexpr int32_t kChunk = 4;
  const int64_t tiles = (scores.size(1) + kTile - 1) / kTile;
  int32_t tiles_per_prog = kChunk;
  for (const int32_t candidate : {16, 8}) {
    if (q.size(0) * ((tiles + candidate - 1) / candidate) >= 32) {
      tiles_per_prog = candidate;
      break;
    }
  }
  const int64_t pb = block_size / pool_size;
  const bool dense = pb <= kTile && (pb & (pb - 1)) == 0;
  triton_jit::JITKernel::get(kScore, "score_pools")
      .launch(
          static_cast<void*>(torch_mlu::getCurMLUStream()),
          {static_cast<uint32_t>(q.size(0)),
           static_cast<uint32_t>((tiles + tiles_per_prog - 1) / tiles_per_prog),
           1},
          {1, 1},
          q,
          w,
          cache,
          pages,
          pos,
          batch,
          scores,
          q.stride(0),
          q.stride(1),
          q.stride(2),
          w.stride(0),
          w.stride(1),
          cache.stride(0),
          cache.stride(1),
          cache.stride(2),
          cache.stride(3),
          pages.stride(0),
          pages.stride(1),
          scores.stride(0),
          scores.stride(1),
          static_cast<float>(scale),
          scores.size(1),
          cache.size(0),
          static_cast<int32_t>(q.size(1)),
          static_cast<int32_t>(q.size(2)),
          static_cast<int32_t>(pb),
          pages.size(1),
          static_cast<int32_t>(pool_size),
          kTile,
          std::max(16, tile_size(q.size(1))),
          dense ? 1 : 0,
          tiles_per_prog,
          kChunk);
}

torch::Tensor select_kpool(const torch::Tensor& scores, int64_t count) {
  CHECK_GE(count, 0);
  torch::Tensor result = torch::full(
      {scores.size(0), count}, -1, scores.options().dtype(torch::kInt64));
  const int64_t k = std::min(count, scores.size(1));
  if (scores.size(0) == 0 || k == 0) {
    return result;
  }
  const auto [values, ids] = torch::topk(scores, k, /*dim=*/-1);
  result.narrow(1, 0, k).copy_(torch::where(torch::isfinite(values), ids, -1));
  return result;
}

KPoolSelection expand_kpool(const torch::Tensor& ids,
                            const torch::Tensor& positions,
                            const torch::Tensor& rows,
                            const torch::Tensor& table,
                            int64_t block_size,
                            int64_t token_budget,
                            int64_t pool_size,
                            bool always_tail) {
  CHECK_GT(pool_size, 0);
  CHECK_GT(block_size, 0);
  CHECK_GE(token_budget, 0);
  CHECK_LE(ids.size(1), token_budget / pool_size);
  CHECK_EQ(ids.size(0), positions.numel());
  CHECK_EQ(ids.size(0), rows.numel());
  const int64_t width = token_budget + pool_size - 1;
  const auto options = ids.options().dtype(torch::kInt64);
  KPoolSelection result{
      torch::full({ids.size(0), width}, -1, options.dtype(torch::kInt32)),
      torch::zeros({ids.size(0)}, options.dtype(torch::kInt32))};
  if (ids.size(0) == 0 || width == 0 || table.size(1) == 0) {
    return result;
  }
  // The old fused shape is an optimization, not a production restriction.
  if (ids.size(1) == 512 && token_budget == 2048 && pool_size == 4 &&
      block_size == 16 && ids.size(0) <= 256 && table.size(1) <= 8192) {
    triton_jit::JITKernel::get(kExpand, "expand_compact")
        .launch(static_cast<void*>(torch_mlu::getCurMLUStream()),
                {static_cast<uint32_t>(ids.size(0)), 1, 1},
                {1, 3},
                ids,
                positions,
                rows,
                table,
                result.physical_slots,
                result.context_lens,
                ids.stride(0),
                ids.stride(1),
                positions.stride(0),
                rows.stride(0),
                table.stride(0),
                table.stride(1),
                table.size(1),
                tile_size(table.size(1)),
                always_tail ? 1 : 0);
    return result;
  }
  const torch::Tensor members = torch::arange(pool_size, options);
  const torch::Tensor tail_members = torch::arange(pool_size - 1, options);
  const int64_t candidates = ids.size(1) * pool_size + pool_size - 1;
  const torch::Tensor columns = torch::arange(candidates, options).unsqueeze(0);
  constexpr int64_t kExpandBytes = 64 * 1024 * 1024;
  const int64_t tile = std::min<int64_t>(
      128,
      std::max<int64_t>(1,
                        kExpandBytes / (std::max<int64_t>(candidates, 1) * 8 *
                                        sizeof(int64_t))));
  for (int64_t start = 0; start < ids.size(0); start += tile) {
    const int64_t length = std::min(tile, ids.size(0) - start);
    const torch::Tensor pools = ids.narrow(0, start, length).to(torch::kInt64);
    const torch::Tensor pos =
        positions.narrow(0, start, length).to(torch::kInt64).unsqueeze(1);
    torch::Tensor tokens = (pools.unsqueeze(-1) * pool_size + members)
                               .reshape({length, ids.size(1) * pool_size});
    torch::Tensor tail = divide(pos + 1, pool_size) * pool_size + tail_members;
    if (!always_tail) {
      tail.fill_(-1);
    }
    tokens = torch::cat({tokens, tail}, /*dim=*/1);
    const torch::Tensor logical = divide(tokens.clamp_min(0), block_size);
    const torch::Tensor pages =
        table.index_select(0, rows.narrow(0, start, length).to(torch::kInt64));
    const torch::Tensor physical =
        pages.gather(1, logical.clamp_max(table.size(1) - 1)).to(torch::kInt64);
    const torch::Tensor valid = (tokens >= 0) & (tokens <= pos) & (pos >= 0) &
                                (logical < table.size(1)) & (physical >= 0);
    // Unique valid-column keys preserve input order without relying on stable
    // sorting of the invalid entries.
    const torch::Tensor order = std::get<1>(torch::sort(
        torch::where(valid, columns.expand_as(tokens), candidates), /*dim=*/1));
    const torch::Tensor slots =
        physical * block_size + remainder(tokens.clamp_min(0), block_size);
    result.physical_slots.narrow(0, start, length)
        .narrow(1, 0, candidates)
        .copy_(torch::where(valid, slots, -1).gather(1, order));
    result.context_lens.narrow(0, start, length).copy_(valid.sum(1));
  }
  return result;
}

}  // namespace xllm::kernel::mlu
