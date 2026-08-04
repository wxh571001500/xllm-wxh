/* Copyright 2026 The xLLM Authors.

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

#include "models/vlm/py_causal_vlm.h"

#include <glog/logging.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_loader.h"
#include "models/py_model_helper.h"

namespace py = pybind11;

namespace xllm {

PyCausalVLM::PyCausalVLM(const ModelContext& context)
    : model_args_(context.get_model_args()),
      options_(context.get_tensor_options()),
      device_(context.get_tensor_options().device()) {
  ensure_python_interpreter();

  const ParallelArgs& parallel_args = context.get_parallel_args();
  tp_group_ = parallel_args.tp_group_;
  tp_size_ = (tp_group_ != nullptr) ? tp_group_->world_size() : 1;
  tp_rank_ = (tp_group_ != nullptr) ? tp_group_->rank() : 0;

  py::gil_scoped_acquire gil;
  if (tp_size_ > 1) {
    CHECK(!parallel_args.python_tp_rendezvous_host_.empty());
    CHECK_GT(parallel_args.python_tp_rendezvous_port_, 0);
    py::module_::import("xllm.python.ops")
        .attr("init_tp_group")(parallel_args.python_tp_rendezvous_host_,
                               parallel_args.python_tp_rendezvous_port_,
                               tp_rank_,
                               tp_size_,
                               c10::str(device_));
  }

  py::module_ registry = py::module_::import("xllm.python.registry");
  py::object model_cls = registry.attr("get_model_class")(
      py::str(context.get_model_args().model_type()));
  config_dict_ = build_config_dict(parallel_args);
  py_model_ = model_cls(config_dict_);
  py_model_.attr("eval")();
}

PyCausalVLM::~PyCausalVLM() {
  py::gil_scoped_acquire gil;
  py_model_ = py::object();
  config_dict_ = py::object();
}

py::dict PyCausalVLM::build_config_dict(
    const ParallelArgs& parallel_args) const {
  py::dict config;
  py::module_ json = py::module_::import("json");
  py::module_ builtins = py::module_::import("builtins");
  const std::string config_path =
      ModelConfig::get_instance().model() + "/config.json";
  py::object config_file = builtins.attr("open")(config_path, "r");
  config = json.attr("load")(config_file).cast<py::dict>();
  config_file.attr("close")();

  PyDictVisitor visitor(config);
  visit_properties(model_args_, visitor);
  visit_properties(parallel_args, visitor);
  config["dtype"] = dtype_to_string(options_);
  config["device"] = c10::str(device_);
  config["tp_size"] = tp_size_;
  config["tp_rank"] = tp_rank_;
  config["enable_graph"] = ExecutionConfig::get_instance().enable_graph();
  config["python_graph_backend"] =
      ExecutionConfig::get_instance().python_graph_backend();
  return config;
}

MMDict PyCausalVLM::encode(const ModelInputParams& parameters) {
  const auto& mm_data = parameters.multimodal.mm_data;
  const auto pixel_values = mm_data.get<torch::Tensor>("pixel_values");
  const auto grid_thws = mm_data.get<torch::Tensor>("image_grid_thw");
  if (!pixel_values.has_value() || !grid_thws.has_value()) {
    return {};
  }

  py::gil_scoped_acquire gil;
  py::object result = py_model_.attr("encode_multimodal")(
      pixel_values.value(), grid_thws.value());
  MMDict output;
  output["image|embedding"] =
      result.cast<std::vector<torch::Tensor>>();
  return output;
}

torch::Tensor PyCausalVLM::get_input_embeddings(
    const torch::Tensor& input_ids,
    const ModelInputParams& input_params) {
  const auto& mm_data = input_params.multimodal.mm_data;
  const auto multimodal_embeds =
      mm_data.get<torch::Tensor>("image|embedding");
  const auto multimodal_mask = mm_data.get<torch::Tensor>("image|mask");
  py::gil_scoped_acquire gil;
  py::object embeddings = multimodal_embeds.has_value()
                              ? py::object(py::cast(multimodal_embeds.value()))
                              : py::object(py::none());
  py::object mask = multimodal_mask.has_value()
                        ? py::object(py::cast(multimodal_mask.value()))
                        : py::object(py::none());
  py::object output = py_model_.attr("get_input_embeddings")(
      input_ids, embeddings, mask);
  return output.cast<torch::Tensor>();
}

ModelOutput PyCausalVLM::forward(const torch::Tensor&,
                                 const torch::Tensor&,
                                 std::vector<KVCache>&,
                                 const ModelInputParams&) {
  LOG(FATAL) << "PyCausalVLM::forward() must not be called directly. "
             << "Python model forward goes through PyExecutorImpl.";
  return ModelOutput(torch::Tensor());
}

torch::Tensor PyCausalVLM::logits(const torch::Tensor& hidden_states,
                                  const torch::Tensor& seleted_idxes) {
  torch::NoGradGuard no_grad;
  py::gil_scoped_acquire gil;
  py::object selected = seleted_idxes.defined()
                            ? py::object(py::cast(seleted_idxes))
                            : py::object(py::none());
  py::object output =
      py_model_.attr("compute_logits")(hidden_states, selected);
  return output.cast<torch::Tensor>();
}

void PyCausalVLM::load_model(std::unique_ptr<ModelLoader> loader) {
  py::gil_scoped_acquire gil;
  py::module_::import("xllm_weight_loader");
  py::list state_dicts;
  for (const auto& state_dict : loader->get_state_dicts()) {
    state_dicts.append(
        py::cast(PyStateDict(state_dict.get()), py::return_value_policy::move));
  }
  py_model_.attr("load_weights")(state_dicts,
                                 static_cast<int32_t>(tp_rank_),
                                 static_cast<int32_t>(tp_size_));
}

}  // namespace xllm
