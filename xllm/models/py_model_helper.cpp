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

// Infrastructure for the embedded Python model executor:
// - Interpreter lifecycle (ensure_python_interpreter)
// - Weight loading (PyStateDict + PYBIND11_EMBEDDED_MODULE)
// - Config serialization (dtype_to_string, PyDictVisitor)

#include "models/py_model_helper.h"

#include <Python.h>
#include <c10/util/Exception.h>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdlib>
#include <mutex>
#include <string>

#include "core/framework/config/model_config.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/kernels/xllm_torch_ops.h"

namespace py = pybind11;

namespace xllm {

namespace {

void prepend_sys_path(const std::string& dir) {
  if (dir.empty()) {
    return;
  }
  py::module_ sys = py::module_::import("sys");
  py::list path = py::reinterpret_borrow<py::list>(sys.attr("path"));
  for (auto item : path) {
    if (py::isinstance<py::str>(item) && item.cast<std::string>() == dir) {
      return;
    }
  }
  path.attr("insert")(0, py::str(dir));
}

}  // namespace

// ---------------------------------------------------------------------------
// dtype_to_string
// ---------------------------------------------------------------------------

std::string dtype_to_string(const torch::TensorOptions& options) {
  switch (c10::typeMetaToScalarType(options.dtype())) {
    case torch::kBFloat16:
      return "bfloat16";
    case torch::kFloat16:
      return "float16";
    case torch::kFloat32:
      return "float32";
    case torch::kFloat64:
      return "float64";
    default:
      return "bfloat16";
  }
}

void init_python_process_groups(const ParallelArgs& parallel_args,
                                const torch::Device& device) {
  CHECK(!parallel_args.python_rendezvous_host_.empty());
  CHECK_GT(parallel_args.python_rendezvous_port_, 0);
  CHECK(!parallel_args.python_process_group_specs_.empty());

  py::list group_specs;
  for (const PythonProcessGroupSpec& spec :
       parallel_args.python_process_group_specs_) {
    py::dict group_spec;
    group_spec["name"] = spec.name;
    group_spec["ranks"] = spec.ranks;
    group_spec["local_rank"] = spec.local_rank;
    group_spec["group_id"] = spec.group_id;
    if (spec.alias_of.empty()) {
      group_spec["alias_of"] = py::none();
    } else {
      group_spec["alias_of"] = spec.alias_of;
    }
    group_specs.append(std::move(group_spec));
  }

  py::module_::import("xllm.python.distributed")
      .attr("init_parallel_groups")(parallel_args.python_rendezvous_host_,
                                    parallel_args.python_rendezvous_port_,
                                    parallel_args.rank(),
                                    parallel_args.world_size(),
                                    c10::str(device),
                                    group_specs);
}

// ---------------------------------------------------------------------------
// ensure_python_interpreter
// ---------------------------------------------------------------------------

void ensure_python_interpreter() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    ensure_xllm_torch_ops_registered();

    const bool we_initialized = !Py_IsInitialized();
    if (we_initialized) {
#if defined(USE_NPU)
      setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 0);
#endif
      py::initialize_interpreter(/*init_signal_handlers=*/false);
    }

    {
      py::gil_scoped_acquire gil;
      std::string model_path = ModelConfig::get_instance().python_model_path();
      if (model_path.empty()) {
        const char* env = std::getenv("XLLM_PYTHON_MODEL_PATH");
        if (env != nullptr) {
          model_path = env;
        }
      }
      prepend_sys_path(model_path);
#if defined(USE_NPU)
      if (we_initialized) {
        py::module_::import("xllm.python._npu_bootstrap");
      }
#endif
      try {
        py::module_::import("xllm.python");
      } catch (const py::error_already_set& e) {
        LOG(FATAL) << "Failed to import the 'xllm.python' model package for "
                      "the Python model executor. Set --python_model_path (or "
                      "XLLM_PYTHON_MODEL_PATH) to the directory containing the "
                      "'xllm' package. Error: "
                   << e.what();
      }
    }

    if (we_initialized) {
      PyEval_SaveThread();
    }
  });
}

// ---------------------------------------------------------------------------
// PyStateDict
// ---------------------------------------------------------------------------

torch::Tensor PyStateDict::get_tensor(const std::string& name) const {
  CHECK(sd_ != nullptr) << "PyStateDict: access after release";
  return sd_->get_tensor(name);
}

torch::Tensor PyStateDict::get_sharded_tensor(const std::string& name,
                                              int64_t dim,
                                              int32_t rank,
                                              int32_t world_size) const {
  CHECK(sd_ != nullptr) << "PyStateDict: access after release";
  return sd_->get_sharded_tensor(name, dim, rank, world_size);
}

bool PyStateDict::has(const std::string& name) const {
  CHECK(sd_ != nullptr) << "PyStateDict: access after release";
  return sd_->has(name);
}

py::list PyStateDict::keys() const {
  CHECK(sd_ != nullptr) << "PyStateDict: access after release";
  py::list result;
  for (const auto& [key, _] : *sd_) {
    result.append(py::str(key));
  }
  return result;
}

PyStateDict PyStateDict::get_dict_with_prefix(const std::string& prefix) const {
  CHECK(sd_ != nullptr) << "PyStateDict: access after release";
  return PyStateDict(
      std::make_unique<StateDict>(sd_->get_dict_with_prefix(prefix)));
}

PyStateDict PyStateDict::get_dict_with_prefixes(
    const std::vector<std::string>& prefixes) const {
  CHECK(sd_ != nullptr) << "PyStateDict: access after release";
  return PyStateDict(
      std::make_unique<StateDict>(sd_->get_dict_with_prefix(prefixes)));
}

size_t PyStateDict::size() const {
  CHECK(sd_ != nullptr) << "PyStateDict: access after release";
  return sd_->size();
}

PYBIND11_EMBEDDED_MODULE(xllm_weight_loader, m) {
  py::class_<PyStateDict>(m, "StateDict")
      .def("get_tensor", &PyStateDict::get_tensor, py::arg("name"))
      .def("get_sharded_tensor",
           &PyStateDict::get_sharded_tensor,
           py::arg("name"),
           py::arg("dim"),
           py::arg("rank"),
           py::arg("world_size"))
      .def("has", &PyStateDict::has, py::arg("name"))
      .def("keys", &PyStateDict::keys)
      .def("get_dict_with_prefix",
           &PyStateDict::get_dict_with_prefix,
           py::arg("prefix"))
      .def("get_dict_with_prefixes",
           &PyStateDict::get_dict_with_prefixes,
           py::arg("prefixes"))
      .def("size", &PyStateDict::size);
}

}  // namespace xllm
