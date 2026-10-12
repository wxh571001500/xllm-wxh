/* Copyright 2025-2026 The xLLM Authors.

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

#include "platform/npu/npu_profiler.h"

#include <glog/logging.h>
#include <pybind11/embed.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <string>
#include <thread>
#include <unordered_map>

#include "util/pybind_helper.h"

namespace xllm {

namespace py = pybind11;

// pybind11 types are declared with hidden visibility; matching Impl's
// visibility avoids GCC's -Wattributes warning under -Werror when the outer
// NpuProfiler class has default visibility.
struct __attribute__((visibility("hidden"))) NpuProfiler::Impl {
  py::object profiler;
  std::thread::id owner_thread;
  std::unordered_map<std::thread::id, bool> workers;
  std::condition_variable workers_stopped;
  bool stopping = false;
  bool failed = false;
  bool stop_succeeded = true;
};

NpuProfiler& NpuProfiler::get_instance() {
  static NpuProfiler instance;
  return instance;
}

NpuProfiler::NpuProfiler() : impl_(std::make_unique<Impl>()) {}

NpuProfiler::~NpuProfiler() {
  if (impl_) {
    clear_python_object(impl_->profiler);
  }
}

bool NpuProfiler::is_running() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return running_;
}

bool NpuProfiler::start(const std::string& profile_dir, int32_t rank) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!Py_IsInitialized() || impl_->failed || impl_->stopping) {
    LOG(ERROR) << "NpuProfiler cannot start: Python is unavailable or the "
                  "profiling session is stopping/failed.";
    return false;
  }

  const std::thread::id current_thread = std::this_thread::get_id();
  if (running_) {
    if (impl_->workers.count(current_thread) != 0) {
      return true;
    }
    try {
      py::gil_scoped_acquire gil;
      py::module_ native = py::module_::import("torch_npu._C._profiler");
      py::object config = native.attr("NpuProfilerConfig")(
          /*path=*/"",
          /*record_shapes=*/false,
          /*profile_memory=*/false,
          /*with_stack=*/false,
          /*with_flops=*/false,
          /*with_modules=*/false,
          native.attr("_ExperimentalConfig")());
      impl_->workers.emplace(current_thread, true);
      native.attr("_enable_profiler_in_child_thread")(config);
      return true;
    } catch (const std::exception& error) {
      impl_->failed = true;
      LOG(ERROR) << "NpuProfiler child registration failed: " << error.what()
                 << "; further captures require a process restart.";
      return false;
    }
  }

  std::error_code error_code;
  const std::string resolved_dir =
      profile_dir.empty() ? std::filesystem::current_path(error_code).string()
                          : profile_dir;
  if (!error_code) {
    std::filesystem::create_directories(resolved_dir, error_code);
  }
  if (error_code) {
    LOG(ERROR) << "NpuProfiler: failed to create profile dir " << resolved_dir
               << ": " << error_code.message();
    return false;
  }

  const std::string worker_name = "xllm_rank" + std::to_string(rank);

  try {
    py::gil_scoped_acquire gil;
    py::module_ tp = py::module_::import("torch_npu.profiler");

    // Level1 + AiCoreNone is the cheapest CANN AclProfiler setting that
    // still yields op_summary.csv and kernel_details.csv. The remaining
    // fields are all pinned to their conservative defaults so the trace
    // envelope stays stable across runs.
    py::object experimental = tp.attr("_ExperimentalConfig")(
        py::arg("export_type") = tp.attr("ExportType").attr("Text"),
        py::arg("profiler_level") = tp.attr("ProfilerLevel").attr("Level1"),
        py::arg("msprof_tx") = false,
        py::arg("aic_metrics") = tp.attr("AiCMetrics").attr("AiCoreNone"),
        py::arg("l2_cache") = false,
        py::arg("op_attr") = false,
        py::arg("data_simplification") = true,
        py::arg("record_op_args") = false,
        py::arg("gc_detect_threshold") = py::none());

    py::list activities;
    activities.append(tp.attr("ProfilerActivity").attr("CPU"));
    activities.append(tp.attr("ProfilerActivity").attr("NPU"));

    py::object trace_handler = tp.attr("tensorboard_trace_handler")(
        py::str(resolved_dir),
        py::arg("worker_name") = py::str(worker_name),
        py::arg("async_mode") = false);
    if (!PyCallable_Check(trace_handler.ptr())) {
      LOG(ERROR) << "NpuProfiler trace handler initialization failed.";
      return false;
    }

    impl_->profiler =
        tp.attr("profile")(py::arg("activities") = activities,
                           py::arg("with_stack") = false,
                           py::arg("profile_memory") = false,
                           py::arg("with_modules") = false,
                           py::arg("experimental_config") = experimental,
                           py::arg("on_trace_ready") = trace_handler);

    impl_->owner_thread = current_thread;
    impl_->workers.clear();
    impl_->workers.emplace(current_thread, true);
    py::object profiler_interface = impl_->profiler.attr("prof_if");
    profiler_interface.attr("init_trace")();
    profiler_interface.attr("start_trace")();
  } catch (const std::exception& error) {
    impl_->failed = true;
    LOG(ERROR) << "NpuProfiler::start failed: " << error.what()
               << "; further captures require a process restart.";
    return false;
  }

  running_ = true;
  impl_->stop_succeeded = true;
  LOG(INFO) << "NpuProfiler started: dir=" << resolved_dir
            << ", worker_name=" << worker_name;
  return true;
}

bool NpuProfiler::stop() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!running_) {
    return !impl_->failed && impl_->stop_succeeded;
  }
  if (!Py_IsInitialized()) {
    LOG(ERROR) << "NpuProfiler cannot stop after Python finalization.";
    return false;
  }

  const std::thread::id current_thread = std::this_thread::get_id();
  auto worker = impl_->workers.find(current_thread);
  if (worker == impl_->workers.end()) {
    LOG(ERROR) << "NpuProfiler::stop must run on a participating worker.";
    return false;
  }
  impl_->stopping = true;
  constexpr std::chrono::seconds kStopTimeout{30};

  if (current_thread != impl_->owner_thread) {
    try {
      if (worker->second) {
        py::gil_scoped_acquire gil;
        py::module_::import("torch_npu").attr("npu").attr("synchronize")();
        py::module_::import("torch_npu._C._profiler")
            .attr("_disable_profiler_in_child_thread")();
        worker->second = false;
      }
    } catch (const std::exception& error) {
      impl_->failed = true;
      LOG(ERROR) << "NpuProfiler child removal failed: " << error.what()
                 << "; further captures require a process restart.";
    }
    impl_->workers_stopped.notify_all();
    impl_->workers_stopped.wait_for(
        lock, kStopTimeout, [this] { return !running_ || impl_->failed; });
    return !running_ && !impl_->failed && impl_->stop_succeeded;
  }

  const auto children_stopped = [this] {
    return std::none_of(impl_->workers.begin(),
                        impl_->workers.end(),
                        [this](const auto& entry) {
                          return entry.first != impl_->owner_thread &&
                                 entry.second;
                        });
  };
  impl_->workers_stopped.wait_for(
      lock, kStopTimeout, [&] { return impl_->failed || children_stopped(); });
  if (!children_stopped()) {
    LOG(ERROR) << "NpuProfiler cannot finalize while child callbacks remain; "
                  "ensure stop is broadcast to all participating workers.";
    return false;
  }

  std::filesystem::path trace_path;
  try {
    py::gil_scoped_acquire gil;
    py::object profiler_interface = impl_->profiler.attr("prof_if");
    profiler_interface.attr("stop_trace")();
    running_ = false;
    impl_->stop_succeeded = false;
    worker->second = false;
    profiler_interface.attr("finalize_trace")();
    impl_->profiler.attr("on_trace_ready")(impl_->profiler);
    trace_path = std::filesystem::path(
                     profiler_interface.attr("prof_path").cast<std::string>()) /
                 "ASCEND_PROFILER_OUTPUT" / "trace_view.json";
  } catch (const std::exception& error) {
    impl_->failed = true;
    impl_->workers_stopped.notify_all();
    LOG(ERROR) << "NpuProfiler::stop failed: " << error.what()
               << "; further captures require a process restart.";
    return false;
  }

  impl_->stopping = false;
  std::error_code error_code;
  const uintmax_t trace_size =
      std::filesystem::file_size(trace_path, error_code);
  if (error_code || trace_size == 0) {
    impl_->workers_stopped.notify_all();
    LOG(ERROR) << "NpuProfiler did not produce a non-empty trace at "
               << trace_path << ": "
               << (error_code ? error_code.message() : "empty file");
    return false;
  }
  LOG(INFO) << "NpuProfiler stopped: trace=" << trace_path;
  impl_->stop_succeeded = !impl_->failed;
  impl_->workers_stopped.notify_all();
  return impl_->stop_succeeded;
}

}  // namespace xllm
