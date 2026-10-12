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

#include "core/runtime/py_input_batch.h"

#include <pybind11/stl.h>

#include "core/framework/model/model_input_params.h"

namespace py = pybind11;

namespace xllm {

void register_input_batch_metadata_view(py::module_& module) {
  py::class_<PyInputBatchMetadataView>(module, "InputBatchMetadataView")
      .def_property_readonly("num_reqs", &PyInputBatchMetadataView::num_reqs)
      .def_property_readonly("num_tokens",
                             &PyInputBatchMetadataView::num_tokens)
      .def_property_readonly("num_scheduled_tokens",
                             &PyInputBatchMetadataView::num_scheduled_tokens)
      .def_property_readonly("num_computed_tokens",
                             &PyInputBatchMetadataView::num_computed_tokens)
      .def_property_readonly("query_start_loc",
                             &PyInputBatchMetadataView::query_start_loc)
      .def_property_readonly("is_prefilling",
                             &PyInputBatchMetadataView::is_prefilling);
}

PyInputBatchMetadataView::PyInputBatchMetadataView(
    const ModelInputParams& params)
    : num_reqs_(params.execution_batch.num_reqs),
      num_tokens_(params.execution_batch.num_tokens),
      num_scheduled_tokens_(params.execution_batch.num_scheduled_tokens),
      num_computed_tokens_(params.execution_batch.num_computed_tokens),
      query_start_loc_(params.execution_batch.query_start_loc),
      is_prefilling_(params.execution_batch.is_prefilling) {}

int32_t PyInputBatchMetadataView::num_reqs() const { return num_reqs_; }

int64_t PyInputBatchMetadataView::num_tokens() const { return num_tokens_; }

const std::vector<int32_t>& PyInputBatchMetadataView::num_scheduled_tokens()
    const {
  return num_scheduled_tokens_;
}

const std::vector<int32_t>& PyInputBatchMetadataView::num_computed_tokens()
    const {
  return num_computed_tokens_;
}

const std::vector<int32_t>& PyInputBatchMetadataView::query_start_loc() const {
  return query_start_loc_;
}

const std::vector<uint8_t>& PyInputBatchMetadataView::is_prefilling() const {
  return is_prefilling_;
}

}  // namespace xllm
