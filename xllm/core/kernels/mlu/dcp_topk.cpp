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

#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <tuple>

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {
namespace {

using triton_jit::JITKernel;
using triton_jit::LaunchCfg;

void check_tensor(const torch::Tensor& tensor,
                  torch::ScalarType dtype,
                  int64_t dims,
                  const torch::Device& device) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.dim(), dims);
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK(tensor.device() == device);
  for (int64_t axis = 0; axis < dims; ++axis) {
    CHECK(tensor.size(axis) <= 1 || tensor.stride(axis) > 0)
        << "Expanded and negative strides are unsupported";
  }
}

void check_outputs(const torch::Tensor& output,
                   const torch::Tensor& lengths,
                   int64_t rows,
                   int64_t width,
                   const torch::Device& device,
                   std::initializer_list<torch::Tensor> inputs) {
  check_tensor(output, torch::kInt32, 2, device);
  check_tensor(lengths, torch::kInt32, 1, device);
  CHECK(output.sizes() == torch::IntArrayRef({rows, width}));
  CHECK_EQ(lengths.size(0), rows);
  CHECK_LE(rows, std::numeric_limits<uint32_t>::max());
  if (rows == 0) {
    return;
  }
  // Accept dense, transposed and sliced tables; reject overlapping writes.
  const int64_t small = output.stride(0) <= output.stride(1) ? 0 : 1;
  const int64_t large = 1 - small;
  CHECK(output.size(large) <= 1 ||
        output.stride(large) > (output.size(small) - 1) * output.stride(small));
  CHECK(!output.is_alias_of(lengths));
  for (const torch::Tensor& input : inputs) {
    CHECK(!output.is_alias_of(input));
    CHECK(!lengths.is_alias_of(input));
  }
}

}  // namespace

void dcp_merge_topk(const torch::Tensor& scores,
                    const torch::Tensor& slots,
                    const torch::Tensor& mapping,
                    int64_t topk,
                    torch::Tensor& output,
                    torch::Tensor& lengths) {
  CHECK(scores.defined());
  CHECK(scores.device().is_privateuseone());
  const torch::Device device = scores.device();
  check_tensor(scores, torch::kFloat32, 3, device);
  check_tensor(slots, torch::kInt32, 3, device);
  check_tensor(mapping, torch::kInt32, 1, device);
  CHECK(scores.sizes() == slots.sizes());
  const int64_t d = scores.size(0);
  const int64_t q = scores.size(1);
  const int64_t c = scores.size(2);
  CHECK_GT(d, 0);
  CHECK_GT(c, 0);
  CHECK_LE(d, std::numeric_limits<int32_t>::max() / c);
  CHECK_GT(topk, 0);
  CHECK_LE(topk, d * c);
  CHECK_EQ(mapping.size(0), q);
  check_outputs(output, lengths, q, topk, device, {scores, slots, mapping});
  if (q == 0) {
    return;
  }

  const torch::DeviceGuard guard(device);
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  const int64_t width = d * c;
  // Ascending negated scores put canonical NaN padding after valid -inf.
  torch::Tensor sortable_scores = torch::empty({q, width}, scores.options());
  torch::Tensor prepared_slots = torch::empty({q, width}, slots.options());
  int64_t candidate_block = 1;
  while (candidate_block < c) {
    candidate_block *= 2;
  }
  JITKernel::get("xllm.core.kernels.mlu.triton_kernel.dcp.dcp_indexer_merge",
                 "tmo_dcp_prepare_indexer_merge_kernel")
      .launch(static_cast<void*>(queue),
              {static_cast<uint32_t>(d), static_cast<uint32_t>(q), 1},
              {/*num_warps=*/1, /*num_stages=*/1},
              scores,
              slots,
              sortable_scores,
              prepared_slots,
              scores.stride(0),
              scores.stride(1),
              scores.stride(2),
              slots.stride(0),
              slots.stride(1),
              slots.stride(2),
              width,
              c,
              candidate_block);

  torch::Tensor score_order = std::get<1>(torch::topk(
      sortable_scores, topk, /*dim=*/1, /*largest=*/false, /*sorted=*/true));
  int64_t topk_block = 1;
  while (topk_block < topk) {
    topk_block *= 2;
  }
  JITKernel::get("xllm.core.kernels.mlu.triton_kernel.dcp.dcp_indexer_merge",
                 "tmo_dcp_finalize_indexer_merge_kernel")
      .launch(static_cast<void*>(queue),
              {static_cast<uint32_t>(q), 1, 1},
              {/*num_warps=*/1, /*num_stages=*/1},
              prepared_slots,
              score_order,
              mapping,
              output,
              lengths,
              width,
              score_order.stride(0),
              mapping.stride(0),
              output.stride(0),
              output.stride(1),
              lengths.stride(0),
              topk,
              topk_block);
}

void dcp_localize_topk(const torch::Tensor& slots,
                       const torch::Tensor& lengths,
                       int32_t rank,
                       int32_t size,
                       int32_t interleave,
                       torch::Tensor& output,
                       torch::Tensor& output_lengths) {
  CHECK(slots.defined());
  CHECK(slots.device().is_privateuseone());
  const torch::Device device = slots.device();
  check_tensor(slots, torch::kInt32, 2, device);
  check_tensor(lengths, torch::kInt32, 1, device);
  const int64_t q = slots.size(0);
  const int64_t width = slots.size(1);
  CHECK_GT(width, 0);
  CHECK_LE(width, std::numeric_limits<int32_t>::max());
  CHECK_EQ(lengths.size(0), q);
  CHECK(lengths.is_contiguous());
  CHECK_GT(size, 0);
  CHECK_GE(rank, 0);
  CHECK_LT(rank, size);
  CHECK_GT(interleave, 0);
  check_outputs(output, output_lengths, q, width, device, {slots, lengths});
  CHECK(output_lengths.is_contiguous());
  if (q == 0) {
    return;
  }

  int64_t block = q <= 128 ? 1024 : 512;
  int32_t stages = q < 1024 ? 3 : 4;
  if (width != 2048 || interleave != 16 || (size != 4 && size != 8)) {
    stages = 3;
  }
  if (slots.stride(1) != 1 || output.stride(1) != 1) {
    block = 256;
    stages = 3;
  }
  LaunchCfg cfg;
  cfg.num_warps = 1;
  cfg.num_stages = stages;
  cfg.force_use_shared_memory = true;
  const torch::DeviceGuard guard(device);
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  JITKernel::get("xllm.core.kernels.mlu.triton_kernel.dcp.dcp_localize_topk",
                 "tmo_dcp_localize_topk_kernel")
      .launch(static_cast<void*>(queue),
              {static_cast<uint32_t>(q), 1, 1},
              cfg,
              slots,
              lengths,
              output,
              output_lengths,
              rank,
              size,
              interleave,
              static_cast<int32_t>(width),
              slots.stride(0),
              slots.stride(1),
              output.stride(0),
              output.stride(1),
              block);
}

}  // namespace xllm::kernel::mlu
