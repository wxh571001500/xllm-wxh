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

#include "executor.h"

#include <string>

#include "executor_impl_factory.h"
#include "platform/device.h"

namespace xllm {

namespace {

bool is_eagle3_model_type(const std::string& model_type) {
  return model_type == "qwen3_eagle3" || model_type == "kimi_k25_eagle3";
}

bool is_kimi_k25_eagle3_speculative_target(const ModelArgs& args,
                                           const runtime::Options& options) {
  return args.model_type() == "kimi_k25" &&
         options.enable_speculative_decode() &&
         options.speculative_algorithm() == "Eagle3";
}

}  // namespace

Executor::Executor(CausalLM* model,
                   const ModelArgs& args,
                   const torch::Device& device,
                   const runtime::Options& options) {
  const bool enable_model_graph =
      options.enable_graph() && !is_eagle3_model_type(args.model_type()) &&
      !is_kimi_k25_eagle3_speculative_target(args, options);
  std::string backend = options.backend() != "vlm" && enable_model_graph
                            ? Device::type_str()
                            : options.backend();
  impl_ = ExecutorImplFactory::get_instance().create_executor_impl(
      model, args, device, options, backend);
}

ForwardInput Executor::prepare_inputs(Batch& batch) {
  return impl_->prepare_inputs(batch);
}

ModelOutput Executor::forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& params) {
  return impl_->run(tokens, positions, kv_caches, params);
}

}  // namespace xllm
