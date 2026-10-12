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

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace xllm {

// NpuProfiler drives online timeline collection through the unwrapped
// torch_npu profiler lifecycle. Unlike the in-process Kineto path
// (TorchProfiler), this backend routes directly to CANN AclProfiler and
// therefore produces artifacts under a *_ascend_pt/ directory containing
// ASCEND_PROFILER_OUTPUT/*.csv, which is what torch_npu.profiler.profiler
// .analyse() consumes and what MindStudio Insight expects.
//
// Each capture has one owner thread. Other worker threads register child
// callbacks against that process-wide capture. Each new capture binds its
// output directory and trace name to the owner that starts it.
//
// IMPORTANT: NPU CPU-op capture uses thread-local RecordFunction callbacks,
// so start() and stop() must be invoked on the compute thread that runs the
// forward pass — WorkerImpl::threadpool_ (single-thread) is that thread.
class NpuProfiler final {
 public:
  static NpuProfiler& get_instance();

  // Open the CANN collection window or attach this worker to the active one.
  // Repeated calls on an attached worker are idempotent. Native lifecycle
  // failures reject further starts until process restart.
  bool start(const std::string& profile_dir, int32_t rank);

  // Detach this worker. The owner waits for children before closing the window
  // and exporting the trace. Children wait for the owner's result, so callers
  // must broadcast stop to all workers.
  // Returns false on lifecycle/export failure or if children fail to stop
  // within the bounded wait. An idle, healthy session is an idempotent success.
  bool stop();

  bool is_running() const;

 private:
  friend class NpuProfilerLifecycleTest;

  NpuProfiler();
  ~NpuProfiler();
  NpuProfiler(const NpuProfiler&) = delete;
  NpuProfiler& operator=(const NpuProfiler&) = delete;

  // Impl holds the cached pybind11::object; kept out of the header so callers
  // do not need pybind11 include paths and so the class visibility does not
  // conflict with pybind11's hidden-visibility types under -Wattributes.
  struct Impl;
  std::unique_ptr<Impl> impl_;

  mutable std::mutex mutex_;
  bool running_ = false;
};

}  // namespace xllm
