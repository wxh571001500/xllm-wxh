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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xllm {

enum class KVTransferErrorCode : uint8_t {
  NONE = 0,
  FAILED = 1,
};

struct KVTransferTaskResult {
  std::vector<std::string> request_ids;
  KVTransferErrorCode error_code = KVTransferErrorCode::NONE;
};

// Store Get volume split by the replica tier that served each object. An
// object whose tier could not be resolved only contributes to read_bytes.
struct StoreGetStats final {
  uint64_t read_bytes = 0;
  uint64_t memory_objects = 0;
  uint64_t memory_bytes = 0;
  uint64_t disk_objects = 0;
  uint64_t disk_bytes = 0;
  // Time spent resolving replica tiers before the Get, which is not part of
  // the data read itself.
  uint64_t tier_query_us = 0;
};

}  // namespace xllm
