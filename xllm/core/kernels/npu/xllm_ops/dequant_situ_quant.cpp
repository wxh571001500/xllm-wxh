/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>

#include <string>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

std::tuple<torch::Tensor, torch::Tensor> dequant_situ_quant(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& weight_scale,
    const std::optional<torch::Tensor>& activation_scale,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& quant_scale,
    const std::optional<torch::Tensor>& quant_offset,
    const std::optional<torch::Tensor>& group_index,
    double beta,
    double linear_beta,
    bool activate_left,
    const std::string& quant_mode) {
  CHECK_EQ(x.dim(), 2)
      << "dequant_situ_quant expects a 2D [rows, width] tensor";
  CHECK_GT(x.size(1), 0) << "dequant_situ_quant input width must be positive";
  // The A3 kernel splits [rows, 2H] into FP32 gate/up buffers. H must
  // be a multiple of 8 so each half starts at a 32-byte vector boundary.
  // Match the INT32/BF16 width constraint in the operator's host tiling.
  CHECK_EQ(x.size(1) % 16, 0)
      << "dequant_situ_quant input width must be a multiple of 16 "
         "for 32-byte FP32 gate/up alignment";
  CHECK(x.scalar_type() == torch::kInt32 || x.scalar_type() == torch::kBFloat16)
      << "dequant_situ_quant supports int32 or bfloat16 input";

  auto y =
      torch::empty({x.size(0), x.size(1) / 2}, x.options().dtype(torch::kInt8));
  auto scale = torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));
  if (x.size(0) == 0) {
    return {y, scale};
  }

  // EXEC_NPU_CMD passes attributes to GetWorkspaceSize synchronously.
  // Keep the string alive through that call; dispatch then uses the executor.
  std::string mode = quant_mode;
  char* quant_mode_ptr = const_cast<char*>(mode.c_str());
  EXEC_NPU_CMD(aclnnDequantSituQuant,
               x,
               weight_scale,
               activation_scale,
               bias,
               quant_scale,
               quant_offset,
               group_index,
               beta,
               linear_beta,
               activate_left,
               quant_mode_ptr,
               y,
               scale);
  return {y, scale};
}

}  // namespace xllm::kernel::npu
