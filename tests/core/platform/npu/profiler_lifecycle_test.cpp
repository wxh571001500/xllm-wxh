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

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "platform/npu/npu_profiler.h"

namespace xllm {
namespace {

namespace py = pybind11;

constexpr char kFakeProfiler[] = R"PY(
import functools
import pathlib
import sys
import threading
import types
from typing import Callable

package = types.ModuleType("torch_npu")
package.__path__ = []
profiler = types.ModuleType("torch_npu.profiler")
native = types.ModuleType("torch_npu._C._profiler")
profiler.events = []
profiler.failure = ""
profiler.generation = 0
profiler.children = set()

def record(action: str) -> None:
    profiler.events.append((action, threading.get_ident()))
    if profiler.failure == action:
        raise RuntimeError("injected " + action + " failure")

def swallow(function: Callable) -> Callable:
    @functools.wraps(function)
    def wrapped(*args: object, **kwargs: object) -> object:
        try:
            return function(*args, **kwargs)
        except Exception:
            return None
    return wrapped

class Interface:
    def init_trace(self) -> None:
        record("init")
        profiler.generation += 1
        self.prof_path = str(pathlib.Path(profiler.output_dir) /
                             (profiler.worker_name + "_" + str(profiler.generation) + "_ascend_pt"))
        pathlib.Path(self.prof_path).mkdir(parents=True)

    def start_trace(self) -> None:
        record("start")
        profiler.owner = threading.get_ident()

    def stop_trace(self) -> None:
        record("stop")
        assert profiler.owner == threading.get_ident(), "wrong stop thread"
        assert not profiler.children, "child callbacks still attached"

    def finalize_trace(self) -> None:
        record("finalize")

class Profile:
    def __init__(self, **kwargs: object) -> None:
        self.prof_if = Interface()
        self.on_trace_ready = kwargs["on_trace_ready"]

    @swallow
    def start(self) -> None:
        self.prof_if.init_trace()
        self.prof_if.start_trace()

    @swallow
    def stop(self) -> None:
        self.prof_if.stop_trace()
        self.prof_if.finalize_trace()
        self.on_trace_ready(self)

def trace_handler(dir_name: str, worker_name: str, async_mode: bool = False) -> Callable:
    profiler.output_dir = dir_name
    profiler.worker_name = worker_name
    def handler(instance: Profile) -> None:
        if profiler.failure == "analyse":
            return
        output = pathlib.Path(instance.prof_if.prof_path) / "ASCEND_PROFILER_OUTPUT"
        output.mkdir()
        content = "" if profiler.failure == "empty_trace" else '{"traceEvents": []}'
        (output / "trace_view.json").write_text(content)
    return handler

def enable_child(config: object) -> None:
    record("enable_child")
    profiler.children.add(threading.get_ident())

def disable_child() -> None:
    record("disable_child")
    profiler.children.remove(threading.get_ident())

profiler.profile = Profile
profiler.tensorboard_trace_handler = trace_handler
profiler._ExperimentalConfig = lambda **kwargs: None
profiler.ExportType = types.SimpleNamespace(Text="text")
profiler.ProfilerLevel = types.SimpleNamespace(Level1=1)
profiler.AiCMetrics = types.SimpleNamespace(AiCoreNone=0)
profiler.ProfilerActivity = types.SimpleNamespace(CPU=0, NPU=1)
native.NpuProfilerConfig = lambda *args: None
native._ExperimentalConfig = lambda: None
native._enable_profiler_in_child_thread = enable_child
native._disable_profiler_in_child_thread = disable_child
package.npu = types.SimpleNamespace(synchronize=lambda: None)
package.profiler = profiler
sys.modules["torch_npu"] = package
sys.modules["torch_npu.profiler"] = profiler
sys.modules["torch_npu._C._profiler"] = native
)PY";

}  // namespace

class NpuProfilerLifecycleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    py::initialize_interpreter();
    py::exec(kFakeProfiler);
    profiler_.reset(new NpuProfiler);
    thread_state_ = PyEval_SaveThread();
    const int64_t timestamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    directory_ = std::filesystem::temp_directory_path() /
                 ("xllm_profiler_lifecycle_" + std::to_string(::getpid()) +
                  "_" + std::to_string(timestamp));
  }

  void TearDown() override {
    profiler_.reset();
    finalize_python();
    std::filesystem::remove_all(directory_);
  }

  void finalize_python() {
    if (thread_state_ != nullptr) {
      PyEval_RestoreThread(thread_state_);
      thread_state_ = nullptr;
      py::finalize_interpreter();
    }
  }

  void set_failure(const std::string& operation) {
    py::gil_scoped_acquire gil;
    py::module_::import("torch_npu.profiler").attr("failure") = operation;
  }

  std::vector<std::string> events() {
    py::gil_scoped_acquire gil;
    std::vector<std::string> result;
    const py::list recorded =
        py::module_::import("torch_npu.profiler").attr("events");
    result.reserve(recorded.size());
    for (const py::handle event : recorded) {
      result.emplace_back(py::cast<std::string>(event[py::int_(0)]));
    }
    return result;
  }

  std::unique_ptr<NpuProfiler, void (*)(NpuProfiler*)> profiler_{
      nullptr,
      [](NpuProfiler* profiler) { delete profiler; }};
  PyThreadState* thread_state_ = nullptr;
  std::filesystem::path directory_;
};

TEST_F(NpuProfilerLifecycleTest, StopWithoutStartIsIdempotent) {
  EXPECT_TRUE(profiler_->stop());
  EXPECT_FALSE(profiler_->is_running());
  EXPECT_TRUE(events().empty());
}

TEST_F(NpuProfilerLifecycleTest, PropagatesInitFailure) {
  set_failure("init");
  EXPECT_FALSE(profiler_->start(directory_.string(), 0));
  EXPECT_FALSE(profiler_->is_running());
}

TEST_F(NpuProfilerLifecycleTest, PropagatesStartFailureAndRejectsReuse) {
  set_failure("start");
  EXPECT_FALSE(profiler_->start(directory_.string(), 0));
  EXPECT_FALSE(profiler_->is_running());
  set_failure("");
  EXPECT_FALSE(profiler_->start(directory_.string(), 0));
}

TEST_F(NpuProfilerLifecycleTest, PropagatesStopFailure) {
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  set_failure("stop");
  EXPECT_FALSE(profiler_->stop());
  EXPECT_FALSE(profiler_->start(directory_.string(), 0));
}

TEST_F(NpuProfilerLifecycleTest, PropagatesFinalizeFailure) {
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  set_failure("finalize");
  EXPECT_FALSE(profiler_->stop());
  EXPECT_FALSE(profiler_->is_running());
  EXPECT_FALSE(profiler_->start(directory_.string(), 0));
}

TEST_F(NpuProfilerLifecycleTest, RejectsMissingTraceAndAllowsNewCapture) {
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  set_failure("analyse");
  EXPECT_FALSE(profiler_->stop());
  EXPECT_FALSE(profiler_->is_running());
  EXPECT_FALSE(profiler_->stop());
  set_failure("");
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  EXPECT_TRUE(profiler_->stop());
}

TEST_F(NpuProfilerLifecycleTest, RejectsEmptyTrace) {
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  set_failure("empty_trace");
  EXPECT_FALSE(profiler_->stop());
}

TEST_F(NpuProfilerLifecycleTest, RepeatedCallsAndNewOutputDirectory) {
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  EXPECT_TRUE(profiler_->start(directory_.string(), 0));
  EXPECT_TRUE(profiler_->stop());
  EXPECT_TRUE(profiler_->stop());
  const std::filesystem::path second = directory_ / "second";
  ASSERT_TRUE(profiler_->start(second.string(), 7));
  EXPECT_TRUE(profiler_->stop());
  EXPECT_TRUE(std::filesystem::exists(
      second /
      "xllm_rank7_2_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json"));
}

TEST_F(NpuProfilerLifecycleTest, RejectsStopFromUnregisteredThread) {
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  auto foreign_stop =
      std::async(std::launch::async, [this] { return profiler_->stop(); });
  EXPECT_FALSE(foreign_stop.get());
  EXPECT_TRUE(profiler_->is_running());
  EXPECT_TRUE(profiler_->stop());
}

TEST_F(NpuProfilerLifecycleTest, ChildStopDoesNotStopOwner) {
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  std::promise<void> child_started;
  auto child = std::async(std::launch::async, [this, &child_started] {
    EXPECT_TRUE(profiler_->start(directory_.string(), 1));
    EXPECT_TRUE(profiler_->start(directory_.string(), 1));
    child_started.set_value();
    EXPECT_TRUE(profiler_->stop());
    EXPECT_TRUE(profiler_->stop());
  });
  child_started.get_future().get();
  EXPECT_EQ(child.wait_for(std::chrono::milliseconds(25)),
            std::future_status::timeout);
  EXPECT_TRUE(profiler_->is_running());
  EXPECT_TRUE(profiler_->stop());
  child.get();
  EXPECT_EQ(events(),
            (std::vector<std::string>{"init",
                                      "start",
                                      "enable_child",
                                      "disable_child",
                                      "stop",
                                      "finalize"}));
}

TEST_F(NpuProfilerLifecycleTest, OwnerWaitsForChildCallbacksBeforeStopping) {
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  std::promise<void> child_started;
  std::promise<void> stop_child;
  auto stop_requested = stop_child.get_future();
  auto child =
      std::async(std::launch::async, [this, &child_started, &stop_requested] {
        EXPECT_TRUE(profiler_->start(directory_.string(), 1));
        child_started.set_value();
        stop_requested.get();
        EXPECT_TRUE(profiler_->stop());
      });
  child_started.get_future().get();
  auto release_child = std::async(std::launch::async, [&stop_child] {
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    stop_child.set_value();
  });
  EXPECT_TRUE(profiler_->stop());
  child.get();
  release_child.get();
  EXPECT_EQ(events(),
            (std::vector<std::string>{"init",
                                      "start",
                                      "enable_child",
                                      "disable_child",
                                      "stop",
                                      "finalize"}));
}

TEST_F(NpuProfilerLifecycleTest, PropagatesChildEnableFailure) {
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  set_failure("enable_child");
  auto child = std::async(std::launch::async, [this] {
    return profiler_->start(directory_.string(), 1);
  });
  EXPECT_FALSE(child.get());
}

TEST_F(NpuProfilerLifecycleTest, ConcurrentWorkersDetachBeforeGlobalStop) {
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  constexpr int32_t kChildCount = 3;
  std::vector<std::promise<void>> started(kChildCount);
  std::promise<void> stop_children;
  const std::shared_future<void> stop_requested =
      stop_children.get_future().share();
  std::vector<std::future<bool>> children;
  children.reserve(kChildCount);
  for (int32_t child_rank = 0; child_rank < kChildCount; ++child_rank) {
    children.emplace_back(std::async(
        std::launch::async, [this, child_rank, &started, stop_requested] {
          EXPECT_TRUE(profiler_->start(directory_.string(), child_rank + 1));
          started[child_rank].set_value();
          stop_requested.get();
          return profiler_->stop();
        }));
  }
  for (auto& start : started) {
    start.get_future().get();
  }
  stop_children.set_value();
  EXPECT_TRUE(profiler_->stop());
  for (auto& child : children) {
    EXPECT_TRUE(child.get());
  }
  const std::vector<std::string> actions = events();
  EXPECT_EQ(std::count(actions.begin(), actions.end(), "enable_child"),
            kChildCount);
  EXPECT_EQ(std::count(actions.begin(), actions.end(), "disable_child"),
            kChildCount);
  ASSERT_GE(actions.size(), 2u);
  EXPECT_EQ(actions[actions.size() - 2], "stop");
  EXPECT_EQ(actions.back(), "finalize");
}

TEST_F(NpuProfilerLifecycleTest, PropagatesChildDisableFailure) {
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  auto child = std::async(std::launch::async, [this] {
    EXPECT_TRUE(profiler_->start(directory_.string(), 1));
    set_failure("disable_child");
    return profiler_->stop();
  });
  EXPECT_FALSE(child.get());
  EXPECT_FALSE(profiler_->stop());
}

TEST_F(NpuProfilerLifecycleTest, RejectsStartAfterPythonFinalization) {
  finalize_python();
  EXPECT_FALSE(profiler_->start(directory_.string(), 0));
}

TEST_F(NpuProfilerLifecycleTest, OwnerFailureIsReportedToChild) {
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  set_failure("finalize");
  std::promise<void> child_started;
  auto child = std::async(std::launch::async, [this, &child_started] {
    EXPECT_TRUE(profiler_->start(directory_.string(), 1));
    child_started.set_value();
    return profiler_->stop();
  });
  child_started.get_future().get();
  EXPECT_FALSE(profiler_->stop());
  EXPECT_FALSE(child.get());
}

TEST_F(NpuProfilerLifecycleTest, DestroysCachedObjectAfterPythonFinalization) {
  ASSERT_TRUE(profiler_->start(directory_.string(), 0));
  EXPECT_TRUE(profiler_->stop());
  finalize_python();
}

}  // namespace xllm
