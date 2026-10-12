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

#include "core/runtime/draft_model_config.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/runtime/options.h"

namespace xllm {
namespace {

#if defined(USE_MLU)
TEST(DraftModelConfigTest, ShortensDFlash2BlockBeforeModelConstruction) {
  ModelArgs args;
  args.model_type("qwen3")
      .dflash2_block_size(8)
      .dflash2_conv_kernel_size(4)
      .layers_to_capture({1, 3});
  runtime::Options options;
  options.is_draft_engine(true)
      .speculative_algorithm("DFlash2")
      .draft_graph_query_width(6);

  configure_block_diffusion_model(args, options);

  EXPECT_EQ(args.model_type(), "DFlash2DraftModel");
  EXPECT_EQ(args.dflash2_block_size(), 6);
  EXPECT_EQ(args.num_speculative_tokens(), 5);
  EXPECT_EQ(args.dummy_token_count(), 6);
  EXPECT_EQ(args.dflash2_conv_kernel_size(), 4);
  EXPECT_TRUE(args.layers_to_capture().empty());
  EXPECT_TRUE(args.requires_eager_execution());
}

TEST(DraftModelConfigTest, AcceptsFullBlockAndConvolutionBoundary) {
  for (const int32_t width : {8, 4}) {
    ModelArgs args;
    args.model_type("qwen3").dflash2_block_size(8).dflash2_conv_kernel_size(4);
    runtime::Options options;
    options.is_draft_engine(true)
        .speculative_algorithm("DFlash2")
        .draft_graph_query_width(width);
    configure_block_diffusion_model(args, options);
    EXPECT_EQ(args.dflash2_block_size(), width);
    EXPECT_EQ(args.dummy_token_count(), width);
    EXPECT_EQ(args.num_speculative_tokens(), width == 8 ? 7 : 3);
  }
}

TEST(DraftModelConfigDeathTest, RejectsInvalidRuntimeGeometry) {
  struct Geometry {
    int32_t trained;
    int32_t runtime;
    const char* error;
  };
  const std::vector<Geometry> geometries = {
      {8, 1, "requires candidate tokens"},
      {8, 9, "must not exceed"},
      {8, 3, "must cover the convolution kernel"},
      {0, 4, "trained_block_size > 0"},
      {-1, 4, "trained_block_size > 0"}};
  for (const Geometry& geometry : geometries) {
    SCOPED_TRACE(geometry.trained);
    SCOPED_TRACE(geometry.runtime);
    ModelArgs args;
    args.model_type("qwen3")
        .dflash2_block_size(geometry.trained)
        .dflash2_conv_kernel_size(4);
    runtime::Options options;
    options.is_draft_engine(true)
        .speculative_algorithm("DFlash2")
        .draft_graph_query_width(geometry.runtime);
    EXPECT_DEATH(configure_block_diffusion_model(args, options),
                 geometry.error);
  }
}

TEST(DraftModelConfigDeathTest, RejectsUnsupportedDraftAlgorithmsOnMlu) {
  for (const std::string& algorithm : {"DFlash", "DSpark"}) {
    ModelArgs args;
    args.model_type("qwen3");
    runtime::Options options;
    options.is_draft_engine(true).speculative_algorithm(algorithm);
    EXPECT_DEATH(configure_block_diffusion_model(args, options),
                 "not supported on mlu");
  }
}
#endif

class DraftCaptureConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    char path[] = "/tmp/xllm-draft-config-XXXXXX";
    ASSERT_NE(mkdtemp(path), nullptr);
    model_path_ = path;
  }

  void TearDown() override { std::filesystem::remove_all(model_path_); }

  void write_config(const std::string& json) {
    std::ofstream file(model_path_ + "/config.json");
    ASSERT_TRUE(file.is_open());
    file << json;
    ASSERT_TRUE(file.good());
  }

  std::string model_path_;
};

TEST_F(DraftCaptureConfigTest, ConfiguresTargetWithoutChangingModelGeometry) {
  write_config(R"({"dflash_config":{"target_layer_ids":[1,3]}})");
  ModelArgs args;
  args.model_type("glm_moe_dsa")
      .n_layers(8)
      .dflash2_block_size(8)
      .num_speculative_tokens(5);
  runtime::Options options;
  options.is_draft_engine(false)
      .speculative_algorithm("DFlash2")
      .draft_model_path(model_path_);
  configure_block_diffusion_model(args, options);
  EXPECT_EQ(args.layers_to_capture(), (std::vector<int32_t>{1, 3}));
  EXPECT_EQ(args.model_type(), "glm_moe_dsa");
  EXPECT_EQ(args.dflash2_block_size(), 8);
  EXPECT_EQ(args.num_speculative_tokens(), 5);
}

TEST_F(DraftCaptureConfigTest, ConvertsBoundaryIndicesForSharedDraftCapture) {
  write_config(R"({"aux_hidden_state_layer_ids":[2,4]})");
  EXPECT_EQ(read_capture_layer_ids(model_path_), (std::vector<int32_t>{1, 3}));
}

TEST_F(DraftCaptureConfigTest, PrefersLegacyPostLayerIndices) {
  write_config(R"({"dspark_target_layer_ids":[1,3],
                    "aux_hidden_state_layer_ids":[3,5]})");
  EXPECT_EQ(read_capture_layer_ids(model_path_), (std::vector<int32_t>{1, 3}));
}

TEST_F(DraftCaptureConfigTest, AllowsEagleCaptureDefaultsForMissingConfig) {
  EXPECT_TRUE(read_capture_layer_ids(model_path_, /*required=*/false).empty());
  write_config("{}");
  EXPECT_TRUE(read_capture_layer_ids(model_path_, /*required=*/false).empty());
}

TEST_F(DraftCaptureConfigTest, RejectsMissingRequiredCaptureConfig) {
  EXPECT_DEATH(read_capture_layer_ids(model_path_), "Failed to parse");
  write_config("{}");
  EXPECT_DEATH(read_capture_layer_ids(model_path_),
               "requires dspark_target_layer_ids");
}

TEST_F(DraftCaptureConfigTest, RejectsEmbeddingCaptureBoundary) {
  write_config(R"({"aux_hidden_state_layer_ids":[0,4]})");
  EXPECT_DEATH(read_capture_layer_ids(model_path_), "Embedding capture");
}

TEST_F(DraftCaptureConfigTest, RejectsInvalidTargetCaptureLayers) {
  for (const std::string& json : {R"({"target_layer_ids":[-1,3]})",
                                  R"({"target_layer_ids":[1,8]})",
                                  R"({"target_layer_ids":[1,1]})"}) {
    write_config(json);
    ModelArgs args;
    args.n_layers(8);
    runtime::Options options;
    options.is_draft_engine(false).draft_model_path(model_path_);
    EXPECT_DEATH(configure_block_diffusion_model(args, options),
                 "Invalid target capture layer|out of range|Duplicate target "
                 "capture layer");
  }
}

TEST(DraftModelConfigDeathTest, RequiresDraftPathForTargetCapture) {
  ModelArgs args;
  runtime::Options options;
  options.is_draft_engine(false);
  EXPECT_DEATH(configure_block_diffusion_model(args, options),
               "requires --draft_model");
}

#if defined(USE_NPU)
TEST(DraftModelConfigTest, PreservesNpuCheckpointBlockGeometry) {
  ModelArgs args;
  args.model_type("qwen3")
      .dflash2_block_size(8)
      .dflash2_conv_kernel_size(4)
      .num_speculative_tokens(7);
  runtime::Options options;
  options.is_draft_engine(true)
      .speculative_algorithm("DFlash2")
      .draft_graph_query_width(6);
  configure_block_diffusion_model(args, options);
  EXPECT_EQ(args.dflash2_block_size(), 8);
  EXPECT_EQ(args.num_speculative_tokens(), 7);
  EXPECT_EQ(args.dummy_token_count(), 8);
  EXPECT_EQ(args.model_type(), "DFlash2DraftModel");
}
#endif

}  // namespace
}  // namespace xllm
