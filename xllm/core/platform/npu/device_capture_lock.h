/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <c10/core/Device.h>

#include <memory>
#include <mutex>
#include <unordered_map>

#include "common/macros.h"

namespace xllm {
namespace npu {

// Device-level mutex manager for protecting ACL Graph capture operations
// Prevents prepare_work_before_execute and capture from executing
// simultaneously to avoid synchronization conflicts during capture
class DeviceCaptureLock {
 public:
  // Get singleton instance
  static DeviceCaptureLock& get_instance() {
    static DeviceCaptureLock instance;
    return instance;
  }

  // Get mutex for a specific device
  // Creates a new mutex if one doesn't exist for the device
  std::mutex& get_lock(c10::DeviceIndex device_index) {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    auto it = locks_.find(device_index);
    if (it == locks_.end()) {
      locks_[device_index] = std::make_unique<std::mutex>();
      return *locks_[device_index];
    }
    return *it->second;
  }

  void begin_capture(c10::DeviceIndex device_index) {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    ++capture_counts_[device_index];
  }

  void end_capture(c10::DeviceIndex device_index) {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    auto it = capture_counts_.find(device_index);
    if (it == capture_counts_.end() || it->second == 0) {
      return;
    }
    --it->second;
  }

  bool is_capture_active(c10::DeviceIndex device_index) {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    auto it = capture_counts_.find(device_index);
    return it != capture_counts_.end() && it->second > 0;
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(DeviceCaptureLock);
  DeviceCaptureLock() = default;
  ~DeviceCaptureLock() = default;

  // Map from device index to mutex
  std::unordered_map<c10::DeviceIndex, std::unique_ptr<std::mutex>> locks_;
  std::unordered_map<c10::DeviceIndex, int32_t> capture_counts_;
  // Mutex to protect the map itself
  std::mutex map_mutex_;
};

}  // namespace npu
}  // namespace xllm
