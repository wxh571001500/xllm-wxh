/* Copyright 2025-2026 The xLLM Authors.

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

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/common/property_reflect.h"
#include "core/framework/state_dict/state_dict.h"

namespace xllm {

struct ParallelArgs;

// Initializes the embedded CPython interpreter (idempotent, process-wide).
void ensure_python_interpreter();

// Convert torch dtype to the string form used by Python model config.
std::string dtype_to_string(const torch::TensorOptions& options);

// Create the embedded Python process groups described by ParallelArgs.
void init_python_process_groups(const ParallelArgs& parallel_args,
                                const torch::Device& device);

// PropertyVisitor that writes each field into a pybind11 dict.
class __attribute__((visibility("hidden"))) PyDictVisitor final
    : public PropertyVisitor {
 public:
  explicit PyDictVisitor(pybind11::dict& dict) : dict_(dict) {}

  void visit(const std::string& name, bool value) override { set(name, value); }
  void visit(const std::string& name, int32_t value) override {
    set(name, value);
  }
  void visit(const std::string& name, int64_t value) override {
    set(name, value);
  }
  void visit(const std::string& name, float value) override {
    set(name, value);
  }
  void visit(const std::string& name, double value) override {
    set(name, value);
  }
  void visit(const std::string& name, const std::string& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<int32_t>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<int64_t>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<float>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<double>& value) override {
    set(name, value);
  }
  void visit(const std::string& name, const std::vector<bool>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<std::string>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::unordered_set<int32_t>& value) override {
    set(name, value);
  }
  void visit_absent(const std::string& name) override {
    dict_[pybind11::str(name)] = pybind11::none();
  }

 private:
  template <typename T>
  void set(const std::string& name, const T& value) {
    dict_[pybind11::str(name)] = value;
  }

  pybind11::dict& dict_;
};

// pybind11-visible wrapper around StateDict for Python weight loading.
class PyStateDict final {
 public:
  explicit PyStateDict(const StateDict* sd) : sd_(sd) {}
  PyStateDict(PyStateDict&& other) noexcept
      : owned_sd_(std::move(other.owned_sd_)),
        sd_(owned_sd_ != nullptr ? owned_sd_.get() : other.sd_) {
    other.sd_ = nullptr;
  }
  PyStateDict& operator=(PyStateDict&& other) noexcept {
    if (this != &other) {
      owned_sd_ = std::move(other.owned_sd_);
      sd_ = owned_sd_ != nullptr ? owned_sd_.get() : other.sd_;
      other.sd_ = nullptr;
    }
    return *this;
  }
  PyStateDict(const PyStateDict&) = delete;
  PyStateDict& operator=(const PyStateDict&) = delete;

  torch::Tensor get_tensor(const std::string& name) const;
  torch::Tensor get_sharded_tensor(const std::string& name,
                                   int64_t dim,
                                   int32_t rank,
                                   int32_t world_size) const;
  bool has(const std::string& name) const;
  pybind11::list keys() const;
  PyStateDict get_dict_with_prefix(const std::string& prefix) const;
  PyStateDict get_dict_with_prefixes(
      const std::vector<std::string>& prefixes) const;
  size_t size() const;

 private:
  explicit PyStateDict(std::unique_ptr<StateDict> sd)
      : owned_sd_(std::move(sd)), sd_(owned_sd_.get()) {}

  std::unique_ptr<StateDict> owned_sd_;
  const StateDict* sd_ = nullptr;
};

}  // namespace xllm
