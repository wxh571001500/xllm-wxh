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

#include "models/model_registry.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include "core/util/json_reader.h"
#include "core/util/model_config_utils.h"

namespace xllm {
namespace {

TEST(ModelRegistryTest, DeepseekV4DSparkUsesTorchBackend) {
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("deepseek_v4_dspark",
                                         "AUTO",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v4_dspark");
  EXPECT_TRUE(error_message.empty());

  EXPECT_TRUE(resolve_model_registration("deepseek_v4_dspark",
                                         "TORCH",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "deepseek_v4_dspark");

  EXPECT_FALSE(resolve_model_registration("deepseek_v4_dspark",
                                          "ATB",
                                          &effective_backend,
                                          &resolved_name,
                                          &error_message));
  EXPECT_EQ(error_message,
            "Model type deepseek_v4_dspark only supports "
            "--npu_kernel_backend=TORCH.");
}

TEST(ModelRegistryTest, KimiK3DSparkUsesTorchBackend) {
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  EXPECT_TRUE(resolve_model_registration("k3_dspark",
                                         "AUTO",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "k3_dspark");
  EXPECT_TRUE(error_message.empty());

  EXPECT_FALSE(resolve_model_registration("k3_dspark",
                                          "ATB",
                                          &effective_backend,
                                          &resolved_name,
                                          &error_message));
  EXPECT_EQ(error_message,
            "Model type k3_dspark only supports "
            "--npu_kernel_backend=TORCH.");
}

TEST(ModelRegistryTest, KimiK3KeepsVlmTargetAndLlmDraftBackends) {
  EXPECT_EQ(ModelRegistry::get_model_backend("kimi_k3"), "vlm");
  EXPECT_EQ(ModelRegistry::get_model_backend("k3_dspark"), "llm");
}

TEST(ModelRegistryTest, LegacyDSparkArchitectureKeepsQwen3ModelType) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3",
      "architectures": ["DSparkDraftModel"]
    }
  )json"));

  EXPECT_EQ(util::get_model_type(reader, std::filesystem::path("draft")),
            "qwen3");
}

TEST(ModelRegistryTest, GenericDSparkArchitectureKeepsK3ModelType) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "k3_dspark",
      "architectures": ["DSparkDraftModel"]
    }
  )json"));

  EXPECT_EQ(util::get_model_type(reader, std::filesystem::path("draft")),
            "k3_dspark");
}

TEST(ModelRegistryTest, KimiK3DSparkArchitectureSelectsK3ModelType) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3",
      "architectures": ["K3DSparkModel"]
    }
  )json"));

  EXPECT_EQ(util::get_model_type(reader, std::filesystem::path("draft")),
            "k3_dspark");
}

}  // namespace
}  // namespace xllm
