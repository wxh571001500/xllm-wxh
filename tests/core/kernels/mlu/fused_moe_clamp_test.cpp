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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <optional>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

TEST(FusedMoeClampTest, MatchesOriginalClampsForDtypesAndTailTiles) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch::DeviceGuard guard(device);
  for (const torch::ScalarType dtype :
       {torch::kBFloat16, torch::kFloat16, torch::kFloat32}) {
    for (const int64_t width : {14, 258, 4096}) {
      torch::Tensor input = torch::randn({9, width}).mul(12).to(dtype);
      input[0][0] = -std::numeric_limits<float>::infinity();
      input[0][1] = std::numeric_limits<float>::infinity();
      input[0][2] = std::numeric_limits<float>::quiet_NaN();
      input[0][width / 2] = -std::numeric_limits<float>::infinity();
      input[0][width / 2 + 1] = std::numeric_limits<float>::infinity();
      input[0][width / 2 + 2] = std::numeric_limits<float>::quiet_NaN();
      const auto parts = input.chunk(2, -1);
      const torch::Tensor expected =
          torch::cat({parts[0].clamp_max(10), parts[1].clamp(-10, 10)}, -1);
      const torch::Tensor device_input = input.to(device);
      for (int32_t repeat = 0; repeat < 2; ++repeat) {
        const torch::Tensor output =
            kernel::mlu::fused_moe_clamp(device_input, 10);
        EXPECT_TRUE(torch::allclose(output.cpu(), expected, 0, 0, true));
        EXPECT_TRUE(torch::allclose(device_input.cpu(), input, 0, 0, true));
        EXPECT_NE(output.data_ptr(), device_input.data_ptr());
      }
    }
  }
}

TEST(FusedMoeClampTest, AlignedThreeDimensionalInputAndEmptyBatch) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch::DeviceGuard guard(device);
  const torch::Tensor input =
      torch::randn({2, 8, 4096}).mul(12).to(torch::kBFloat16);
  const auto parts = input.chunk(2, -1);
  const torch::Tensor expected =
      torch::cat({parts[0].clamp_max(10), parts[1].clamp(-10, 10)}, -1);
  const auto output = kernel::mlu::fused_moe_clamp(input.to(device), 10);
  EXPECT_EQ(output.sizes(), input.sizes());
  EXPECT_TRUE(torch::equal(output.cpu(), expected));
  const auto empty = torch::empty({0, 4096}, input.options().device(device));
  EXPECT_EQ(kernel::mlu::fused_moe_clamp(empty, 10).sizes(), empty.sizes());
}

TEST(FusedMoeClampTest, MatchesOriginalActivationQuantizationChain) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch::DeviceGuard guard(device);
  const torch::Tensor input =
      torch::randn({16, 4096}).mul(12).to(torch::kBFloat16).to(device);
  const auto parts = input.chunk(2, -1);
  const torch::Tensor original =
      torch::cat({parts[0].clamp_max(10), parts[1].clamp(-10, 10)}, -1);
  const torch::Tensor fused = kernel::mlu::fused_moe_clamp(input, 10);
  const torch::Tensor smooth =
      torch::ones({2048}, input.options().dtype(torch::kFloat32));
  auto quantize = [&smooth](const torch::Tensor& value) {
    return kernel::mlu::scaled_quantize(value,
                                        smooth,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        "silu",
                                        /*active_coef=*/1.0,
                                        /*is_gated=*/true);
  };
  const auto [expected, expected_scale] = quantize(original);
  const auto [actual, actual_scale] = quantize(fused);
  EXPECT_TRUE(torch::equal(actual.cpu(), expected.cpu()));
  EXPECT_TRUE(torch::equal(actual_scale.cpu(), expected_scale.cpu()));
}

}  // namespace
}  // namespace xllm
