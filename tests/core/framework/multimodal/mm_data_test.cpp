/* Copyright 2026 The xLLM Authors.

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

#include "core/framework/multimodal/mm_data.h"

#include <gtest/gtest.h>

#include <cstdint>

#include "core/framework/multimodal/mm_batch_data.h"

namespace xllm {
namespace {

struct TypeMaskCase {
  uint32_t stored_type;
  uint32_t query_type;
  bool expected;
};

class MMDataTypeMaskTest : public ::testing::TestWithParam<TypeMaskCase> {};

TEST_P(MMDataTypeMaskTest, ChecksDataTypes) {
  const TypeMaskCase& test_case = GetParam();
  MMData data;
  data.set(test_case.stored_type, MMDict{});

  EXPECT_EQ(data.has(test_case.query_type), test_case.expected);
  EXPECT_EQ(data.has(MMType{static_cast<MMType::Value>(test_case.query_type)}),
            test_case.expected);
}

TEST_P(MMDataTypeMaskTest, ChecksBatchTypes) {
  const TypeMaskCase& test_case = GetParam();
  const MMBatchData data(test_case.stored_type, MMDict{});

  EXPECT_EQ(data.has(test_case.query_type), test_case.expected);
}

INSTANTIATE_TEST_SUITE_P(
    TypeMasks,
    MMDataTypeMaskTest,
    ::testing::Values(
        TypeMaskCase{MMType::NONE, MMType::NONE, false},
        TypeMaskCase{MMType::NONE, MMType::IMAGE, false},
        TypeMaskCase{MMType::NONE, MMType::VIDEO, false},
        TypeMaskCase{MMType::NONE, MMType::AUDIO, false},
        TypeMaskCase{MMType::NONE, MMType::EMBEDDING, false},
        TypeMaskCase{MMType::IMAGE, MMType::NONE, false},
        TypeMaskCase{MMType::IMAGE, MMType::IMAGE, true},
        TypeMaskCase{MMType::IMAGE, MMType::VIDEO, false},
        TypeMaskCase{MMType::VIDEO, MMType::VIDEO, true},
        TypeMaskCase{MMType::VIDEO, MMType::IMAGE, false},
        TypeMaskCase{MMType::AUDIO, MMType::AUDIO, true},
        TypeMaskCase{MMType::AUDIO, MMType::IMAGE, false},
        TypeMaskCase{MMType::EMBEDDING, MMType::EMBEDDING, true},
        TypeMaskCase{MMType::EMBEDDING, MMType::IMAGE, false},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING, MMType::NONE, false},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING, MMType::VIDEO, true},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING,
                     MMType::EMBEDDING,
                     true},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING, MMType::IMAGE, false},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING, MMType::AUDIO, false},
        TypeMaskCase{MMType::VIDEO, MMType::VIDEO | MMType::AUDIO, true},
        TypeMaskCase{MMType::VIDEO, MMType::IMAGE | MMType::AUDIO, false},
        TypeMaskCase{MMType::IMAGE | MMType::AUDIO,
                     MMType::VIDEO | MMType::AUDIO,
                     true}));

}  // namespace
}  // namespace xllm
