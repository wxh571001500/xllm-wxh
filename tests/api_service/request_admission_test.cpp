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

#include "api_service/request_admission.h"

#include <absl/container/flat_hash_set.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "core/framework/config/service_config.h"

namespace xllm {
namespace {

TEST(RequestAdmissionTest, UnsupportedModelDoesNotConsumeConcurrencySlot) {
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;
  const absl::flat_hash_set<std::string> models = {"supported-model"};

  for (int32_t attempt = 0; attempt < 2; ++attempt) {
    const Status status = api_service_internal::admit_rpc_request(
        "unsupported-model", models, &rate_limiter);

    EXPECT_EQ(status.code(), StatusCode::UNKNOWN);
    EXPECT_EQ(status.message(), "Model not supported");
    EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);
  }

  const Status status = api_service_internal::admit_rpc_request(
      "supported-model", models, &rate_limiter);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);
  rate_limiter.decrease_one_request();
}

}  // namespace
}  // namespace xllm
