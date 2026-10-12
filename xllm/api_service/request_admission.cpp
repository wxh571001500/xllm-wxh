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

#include <glog/logging.h>

namespace xllm::api_service_internal {

Status admit_rpc_request(const std::string& model,
                         const absl::flat_hash_set<std::string>& models,
                         RateLimiter* rate_limiter) {
  if (!models.contains(model)) {
    return {StatusCode::UNKNOWN, "Model not supported"};
  }

  CHECK(rate_limiter != nullptr);
  if (rate_limiter->is_limited()) {
    return {StatusCode::RESOURCE_EXHAUSTED,
            "The number of concurrent requests has reached the limit."};
  }
  return {};
}

}  // namespace xllm::api_service_internal
