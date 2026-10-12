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
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>

#include "core/kernels/xllm_torch_ops.h"

namespace py = pybind11;

namespace xllm {
namespace {

TEST(Qwen35TargetGraphTest, DSparkPythonTargetAclGraphMatchesEager) {
  ensure_xllm_torch_ops_registered();
  if (c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
          ->deviceCount() == 0) {
    GTEST_SKIP() << "NPU is unavailable.";
  }
  // cc_test initializes the real Python/NPU runtime before this test.
  ASSERT_TRUE(Py_IsInitialized());
  py::gil_scoped_acquire gil;
  try {
    std::filesystem::path root(__FILE__);
    for (int32_t depth = 0; depth < 5; ++depth) {
      root = root.parent_path();
    }
    py::module_::import("sys").attr("path").attr("insert")(0, root.string());
    py::module_::import("xllm.python").attr("initialize_runtime")();
    const std::filesystem::path path =
        std::filesystem::path(__FILE__).parent_path() /
        "qwen3_5_target_graph_cases.py";
    py::dict test = py::module_::import("runpy")
                        .attr("run_path")(path.string())
                        .cast<py::dict>();
    test["check_dspark_target_aclgraph"]();
  } catch (const py::error_already_set& error) {
    // Keep the GIL while reporting and destroying Python assertion errors.
    FAIL() << error.what();
  }
}

}  // namespace
}  // namespace xllm

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int32_t result = RUN_ALL_TESTS();
  if (!Py_IsInitialized()) {
    return result;
  }
  // npu_test_environment.cpp initialized ACL and restored the main-thread GIL.
  // Release GE resources while torch_npu's allocators are still alive, rather
  // than relying on the order of shared-library destructors at process exit.
  PyThreadState* state = PyEval_SaveThread();
  const aclError status = aclFinalize();
  PyEval_RestoreThread(state);
  if (status != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "ACL test finalization failed: %d\n",
                 static_cast<int32_t>(status));
    return 1;
  }
  return result;
}
