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

#include "core/distributed_runtime/master.h"
#include "core/runtime/dflash_worker_impl.h"

namespace xllm::dflash_detail {
namespace {

class MasterRouteProbe final : public Master {
 public:
  using Master::should_use_vlm_speculative_engine;

  void run() override {}

 private:
  MasterRouteProbe() = delete;
};

TEST(DFlashBackendTest, PreservesVlmTargetBackend) {
  runtime::Options options;
  options.backend("vlm")
      .enable_schedule_overlap(true)
      .is_draft_engine(true)
      .enable_graph_aux_hidden_states(false);

  const runtime::Options target = target_options(options);

  EXPECT_EQ(target.backend(), "vlm");
  EXPECT_FALSE(target.enable_schedule_overlap());
  EXPECT_FALSE(target.is_draft_engine());
  EXPECT_TRUE(target.enable_graph_aux_hidden_states());
}

TEST(DFlashBackendTest, UsesLlmBackendForTextTargets) {
  runtime::Options options;
  options.backend("llm");

  EXPECT_EQ(target_options(options).backend(), "llm");

  options.backend("");
  EXPECT_EQ(target_options(options).backend(), "llm");
}

TEST(DFlashBackendTest, RoutesOnlyVlmDSparkToSpeculativeEngine) {
  Options options;
  options.speculative_algorithm("DSpark").draft_model_path("draft-model");

  EXPECT_TRUE(MasterRouteProbe::should_use_vlm_speculative_engine(options));

  options.speculative_algorithm("DFlash");
  EXPECT_FALSE(MasterRouteProbe::should_use_vlm_speculative_engine(options));

  options.speculative_algorithm("DSpark").draft_model_path("");
  EXPECT_FALSE(MasterRouteProbe::should_use_vlm_speculative_engine(options));
}

}  // namespace
}  // namespace xllm::dflash_detail
