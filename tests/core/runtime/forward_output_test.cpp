/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "runtime/forward_params.h"

namespace xllm {
namespace {

TEST(ForwardOutputTest, NonDriverFailureIsReturnedWhenEplbIsDisabled) {
  std::optional<ForwardOutput> output = make_non_driver_forward_output(
      ForwardOutput{}, {"request-a"}, /*eplb_enabled=*/false);

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->failed_request_ids,
            (std::vector<std::string>{"request-a"}));
}

TEST(ForwardOutputTest, NonDriverSuccessRemainsEmptyWhenEplbIsDisabled) {
  std::optional<ForwardOutput> output = make_non_driver_forward_output(
      ForwardOutput{}, {}, /*eplb_enabled=*/false);

  EXPECT_FALSE(output.has_value());
}

TEST(ForwardOutputTest, NonDriverEplbOutputIsRetainedWithoutFailures) {
  std::optional<ForwardOutput> output = make_non_driver_forward_output(
      ForwardOutput{}, {}, /*eplb_enabled=*/true);

  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(output->failed_request_ids.empty());
}

}  // namespace
}  // namespace xllm
