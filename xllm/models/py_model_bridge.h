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

#pragma once

#include <pybind11/pybind11.h>

namespace xllm {

// Common accessors used by the Python executor for both causal LMs and VLMs.
class __attribute__((visibility("hidden"))) PyModelBridge {
 public:
  virtual ~PyModelBridge() = default;
  virtual pybind11::object& python_model() = 0;
  virtual const pybind11::object& config_dict() const = 0;
};

}  // namespace xllm
