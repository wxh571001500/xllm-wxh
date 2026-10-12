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
#include <framework/core/device.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"
#include "util/utils.h"

namespace xllm::kernel::mlu {
namespace {

constexpr char kModule[] =
    "xllm.core.kernels.mlu.triton_kernel.dcp.dcp_correct_attn_out";

bool supported_dtype(torch::ScalarType dtype) {
  return dtype == torch::kBFloat16 || dtype == torch::kFloat16 ||
         dtype == torch::kFloat32;
}

void launch_correction(const torch::Tensor& src,
                       const torch::Tensor& lse,
                       const torch::Tensor& slots,
                       int64_t rank,
                       torch::Tensor& dst,
                       bool transpose,
                       bool base_e) {
  CHECK_EQ(src.dim(), 3);
  CHECK_EQ(lse.dim(), 3);
  CHECK_EQ(slots.dim(), 1);
  CHECK(src.device().is_privateuseone());
  CHECK(src.device() == lse.device());
  CHECK(src.device() == slots.device());
  CHECK(src.device() == dst.device());
  CHECK(supported_dtype(src.scalar_type()));
  CHECK(supported_dtype(lse.scalar_type()));
  CHECK_EQ(src.scalar_type(), dst.scalar_type());
  CHECK_EQ(slots.scalar_type(), torch::kInt32);
  CHECK(slots.is_contiguous());
  const int64_t b = src.size(0);
  const int64_t h = src.size(1);
  const int64_t v = src.size(2);
  const int64_t n = lse.size(0);
  CHECK_GT(h, 0);
  CHECK_GT(v, 0);
  CHECK_GT(n, 0);
  CHECK_GE(rank, 0);
  CHECK_LT(rank, n);
  CHECK_EQ(lse.size(1), b);
  CHECK_EQ(lse.size(2), h);
  CHECK_EQ(slots.numel(), b);
  if (transpose) {
    CHECK(dst.sizes() == torch::IntArrayRef({h, b, v}));
    CHECK(dst.is_contiguous());
    CHECK(!dst.is_alias_of(src));
    CHECK(!dst.is_alias_of(lse));
    CHECK(!dst.is_alias_of(slots));
  }
  if (b == 0) {
    return;
  }
  const torch::DeviceGuard guard(src.device());
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  const bool fast = base_e && n == 4 && h == 64 && v == 512 &&
                    src.scalar_type() == torch::kBFloat16 &&
                    lse.scalar_type() == torch::kFloat32 &&
                    src.is_contiguous() && lse.is_contiguous();
  using xllm::triton_jit::JITKernel;
  if (fast && (b <= 128 || (!transpose && b >= 256))) {
    const torch_mlu::DeviceProp* prop =
        torch_mlu::getDeviceProperties(torch_mlu::current_device());
    CHECK(prop != nullptr);
    const int64_t cores = prop->cluster_count * prop->core_num_per_cluster;
    if (b <= 128) {
      JITKernel::get(kModule, "tmo_dcp_decode_kernel")
          .launch(static_cast<void*>(queue),
                  {static_cast<uint32_t>(std::min(b, cores)), 1, 1},
                  {/*num_warps=*/1, /*num_stages=*/4},
                  src,
                  dst,
                  lse,
                  slots,
                  rank,
                  b,
                  transpose,
                  /*G=*/int64_t{64});
    } else {
      JITKernel::get(kModule, "tmo_dcp_prefill_kernel")
          .launch(static_cast<void*>(queue),
                  {static_cast<uint32_t>(cores), 1, 1},
                  {/*num_warps=*/1, /*num_stages=*/4},
                  dst,
                  lse,
                  slots,
                  b,
                  h,
                  v,
                  rank,
                  base_e,
                  /*G=*/int64_t{64});
    }
    return;
  }
  const int64_t vn = util::ceil_pow2(static_cast<int32_t>(v));
  const int64_t nn = util::ceil_pow2(static_cast<int32_t>(n));
  JITKernel::get(kModule, "tmo_dcp_correct_attn_out_kernel")
      .launch(static_cast<void*>(queue),
              {static_cast<uint32_t>(b), static_cast<uint32_t>(h), 1},
              {/*num_warps=*/1, /*num_stages=*/1},
              src,
              dst,
              lse,
              slots,
              rank,
              n,
              v,
              src.stride(0),
              src.stride(1),
              src.stride(2),
              dst.stride(transpose ? 1 : 0),
              dst.stride(transpose ? 0 : 1),
              dst.stride(2),
              lse.stride(0),
              lse.stride(1),
              lse.stride(2),
              vn,
              nn,
              base_e);
}

}  // namespace

void dcp_correct_attn_out(torch::Tensor& output,
                          const torch::Tensor& lse,
                          const torch::Tensor& slots,
                          int64_t rank,
                          bool base_e) {
  launch_correction(output,
                    lse,
                    slots,
                    rank,
                    output,
                    /*transpose=*/false,
                    base_e);
}

void dcp_correct_attn_transpose(const torch::Tensor& output,
                                const torch::Tensor& lse,
                                const torch::Tensor& slots,
                                int64_t rank,
                                torch::Tensor& dst,
                                bool base_e) {
  launch_correction(output,
                    lse,
                    slots,
                    rank,
                    dst,
                    /*transpose=*/true,
                    base_e);
}

}  // namespace xllm::kernel::mlu
