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
#include <torch/torch.h>

#include <cmath>
#include <cstdint>

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {

torch::Tensor fused_moe_clamp(const torch::Tensor& input, float limit) {
  CHECK(input.device().is_privateuseone());
  CHECK(input.is_contiguous());
  CHECK_GE(input.dim(), 2);
  CHECK(input.scalar_type() == torch::kBFloat16 ||
        input.scalar_type() == torch::kFloat16 ||
        input.scalar_type() == torch::kFloat32);
  CHECK(std::isfinite(limit));
  CHECK_GE(limit, 0.0f);
  const int64_t width = input.size(-1);
  CHECK_GT(width, 0);
  CHECK_EQ(width % 2, 0);
  torch::Tensor output = torch::empty_like(input);
  if (input.numel() == 0) {
    return output;
  }
  const int64_t rows = input.numel() / width;
  torch_mlu::DeviceProp* prop =
      torch_mlu::getDeviceProperties(torch_mlu::current_device());
  CHECK(prop != nullptr);
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  triton_jit::JITKernel& kernel = triton_jit::JITKernel::get(
      "xllm.core.kernels.mlu.triton_kernel.fused_moe_clamp", "fused_moe_clamp");
  kernel.launch(static_cast<void*>(queue),
                {static_cast<uint32_t>(prop->cluster_count), 1, 1},
                {/*num_warps=*/4, /*num_stages=*/4},
                input,
                output,
                rows,
                width,
                /*row_stride=*/width,
                limit,
                /*block_columns=*/int64_t{4096},
                /*block_rows=*/int64_t{8});
  return output;
}

}  // namespace xllm::kernel::mlu
