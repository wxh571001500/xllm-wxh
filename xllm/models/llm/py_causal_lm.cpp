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

#include "models/llm/py_causal_lm.h"

#include <glog/logging.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <memory>
#include <string>
#include <utility>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_loader.h"
#include "core/framework/state_dict/state_dict.h"
#include "models/py_model_helper.h"

namespace py = pybind11;

namespace xllm::detail {
void share_python_model_weights(py::object& draft_model,
                                const py::object& target_model) {
  draft_model.attr("lm_head") = target_model.attr("lm_head");
  py::object draft_body = draft_model.attr("model");
  py::object target_body = target_model.attr("model");
  draft_body.attr("embed_tokens") = target_body.attr("embed_tokens");
}
}  // namespace xllm::detail

namespace xllm {

PyCausalLM::PyCausalLM(const ModelContext& context)
    : model_args_(context.get_model_args()),
      options_(context.get_tensor_options()),
      device_(context.get_tensor_options().device()),
      enable_mla_(context.get_model_args().enable_mla()) {
  ensure_python_interpreter();

  const ParallelArgs& parallel_args = context.get_parallel_args();
  tp_group_ = parallel_args.tp_group_;
  tp_size_ = (tp_group_ != nullptr) ? tp_group_->world_size() : 1;
  tp_rank_ = (tp_group_ != nullptr) ? tp_group_->rank() : 0;

  py::gil_scoped_acquire gil;
  init_python_process_groups(parallel_args, device_);
  const std::string module_name = context.get_model_args().model_type().empty()
                                      ? std::string("Qwen3ForCausalLM")
                                      : context.get_model_args().model_type();

  py::module_ registry = py::module_::import("xllm.python.registry");
  py::object model_cls = registry.attr("get_model_class")(py::str(module_name));
  config_dict_ = build_config_dict(parallel_args, context.get_quant_args());
  py_model_ = model_cls(config_dict_);
  py_model_.attr("eval")();
}

PyCausalLM::~PyCausalLM() {
  py::gil_scoped_acquire gil;
  py_model_ = py::object();
  config_dict_ = py::object();
}

py::dict PyCausalLM::build_config_dict(const ParallelArgs& parallel_args,
                                       const QuantArgs& quant_args) const {
  py::dict d;
  if (model_args_.model_type() == "kimi_k3") {
    py::module_ json = py::module_::import("json");
    py::module_ builtins = py::module_::import("builtins");
    const std::string config_path =
        ModelConfig::get_instance().model() + "/config.json";
    py::object config_file = builtins.attr("open")(config_path, "r");
    d = json.attr("load")(config_file).cast<py::dict>();
    config_file.attr("close")();
    d["quantize_type"] = quant_args.quantize_type();
    d["quant_method"] = quant_args.quant_method();
    d["quant_group_size"] = quant_args.group_size();
    d["quant_version"] = quant_args.quant_version();
  }
  PyDictVisitor visitor(d);
  visit_properties(model_args_, visitor);
  visit_properties(parallel_args, visitor);
  d["dtype"] = dtype_to_string(options_);
  d["device"] = c10::str(device_);
  d["tp_size"] = tp_size_;
  d["tp_rank"] = tp_rank_;
  d["enable_graph"] = ExecutionConfig::get_instance().enable_graph();
  d["python_graph_backend"] =
      ExecutionConfig::get_instance().python_graph_backend();
  return d;
}

void PyCausalLM::load_model(std::unique_ptr<ModelLoader> loader) {
  py::gil_scoped_acquire gil;
  auto& state_dicts = loader->get_state_dicts();
  py::module_::import("xllm_weight_loader");

  py::list py_state_dicts;
  for (const auto& sd : state_dicts) {
    py_state_dicts.append(
        py::cast(PyStateDict(sd.get()), py::return_value_policy::move));
  }

  py_model_.attr("load_weights")(py_state_dicts,
                                 static_cast<int32_t>(tp_rank_),
                                 static_cast<int32_t>(tp_size_));
}

ModelOutput PyCausalLM::forward(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& parameters) {
  LOG(FATAL) << "PyCausalLM::forward() must not be called directly. "
             << "Python model forward goes through PyExecutorImpl.";
  return ModelOutput(torch::Tensor());
}

torch::Tensor PyCausalLM::logits(const torch::Tensor& hidden_states,
                                 const torch::Tensor& seleted_idxes) {
  torch::NoGradGuard no_grad;
  py::gil_scoped_acquire gil;
  py::object selected = seleted_idxes.defined()
                            ? py::object(py::cast(seleted_idxes))
                            : py::object(py::none());
  py::object out = py_model_.attr("compute_logits")(hidden_states, selected);
  return out.cast<torch::Tensor>();
}

}  // namespace xllm
