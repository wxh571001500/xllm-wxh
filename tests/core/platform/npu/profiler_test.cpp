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

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>

#include "core/platform/npu/npu_profiler.h"
#include "tests/npu_test_environment.h"

namespace {

namespace fs = std::filesystem;

class NpuProfilerTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    xllm::testing::init_npu_test_runtime();
    google::InitGoogleLogging("npu_profiler_test");
    google::SetStderrLogging(google::INFO);

    int ret = aclrtSetDevice(0);
    if (ret != 0) {
      LOG(ERROR) << "ACL set device id: 0 failed, ret:" << ret;
    }
    torch_npu::init_npu("npu:0");
  }

  void TearDown() override {
    torch_npu::finalize_npu();
    aclrtResetDevice(0);
    aclFinalize();
    google::ShutdownGoogleLogging();
    xllm::testing::finalize_npu_test_runtime();
  }
};

::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new NpuProfilerTestEnvironment);

fs::path make_temp_dir(const std::string& tag) {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto ts =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
  fs::path dir = fs::temp_directory_path() /
                 ("xllm_npu_profiler_test_" + tag + "_" +
                  std::to_string(::getpid()) + "_" + std::to_string(ts));
  fs::create_directories(dir);
  return dir;
}

// Count *_ascend_pt/ directories that torch_npu.profiler's
// tensorboard_trace_handler drops under `dir`. Anything else (log dirs, stray
// files) is ignored so the assertion focuses on the profiler artifact.
size_t count_ascend_pt_dirs(const fs::path& dir) {
  size_t count = 0;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.is_directory() && entry.path().filename().string().rfind(
                                    "_ascend_pt") != std::string::npos) {
      ++count;
    }
  }
  return count;
}

}  // namespace

TEST(NpuProfilerTest, StopWithoutStartIsNoop) {
  auto& profiler = xllm::NpuProfiler::get_instance();
  EXPECT_FALSE(profiler.is_running());
  // stop() without a prior start() must not crash and must not flip state.
  EXPECT_TRUE(profiler.stop());
  EXPECT_FALSE(profiler.is_running());
}

// Cover idempotent start() and real *_ascend_pt/ artifact production.
TEST(NpuProfilerTest, StartStopWritesAscendPtDir) {
  auto& profiler = xllm::NpuProfiler::get_instance();
  ASSERT_FALSE(profiler.is_running());

  fs::path dir = make_temp_dir("write");
  constexpr int32_t kRank = 7;
  ASSERT_TRUE(profiler.start(dir.string(), kRank));
  EXPECT_TRUE(profiler.is_running());
  // Second start() while already running must be a no-op success.
  EXPECT_TRUE(profiler.start(dir.string(), kRank));
  EXPECT_TRUE(profiler.is_running());

  // Give the CANN AclProfiler something to record so the trace has at least
  // one host-side op and one NPU-side kernel event.
  {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device("npu:0");
    auto a = torch::ones({16}, opts);
    auto b = a + 1.0f;
    (void)b.cpu();
  }

  ASSERT_TRUE(profiler.stop());
  EXPECT_FALSE(profiler.is_running());

  // tensorboard_trace_handler drops one *_ascend_pt/ directory per invocation.
  // The "xllm_rank7_" prefix comes from the worker_name we pass to start().
  size_t ascend_dirs = count_ascend_pt_dirs(dir);
  EXPECT_GE(ascend_dirs, 1u)
      << "no *_ascend_pt/ directory produced under " << dir;

  bool has_rank_prefix = false;
  for (const auto& entry : fs::directory_iterator(dir)) {
    const auto name = entry.path().filename().string();
    LOG(INFO) << "npu profiler produced: " << name;
    if (name.rfind("xllm_rank7_", 0) == 0 &&
        name.find("_ascend_pt") != std::string::npos) {
      has_rank_prefix = true;
    }
  }
  EXPECT_TRUE(has_rank_prefix)
      << "no xllm_rank7_*_ascend_pt/ directory under " << dir;

  fs::remove_all(dir);
}
