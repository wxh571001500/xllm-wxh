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

#pragma once

#include <glog/logging.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/model/causal_vlm.h"
#include "core/framework/model_context.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "models/model_registry.h"
#include "models/py_model_helper.h"
#include "processors/kimi25_image_processor.h"
#include "processors/multimodal_processor.h"

namespace xllm {

namespace py = pybind11;

class __attribute__((visibility("hidden")))
    KimiK3ForConditionalGenerationImpl final : public torch::nn::Module {
 public:
  explicit KimiK3ForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    ensure_python_interpreter();

    const ParallelArgs& parallel_args = context.get_parallel_args();
    tp_size_ = parallel_args.tp_group_ != nullptr
                   ? parallel_args.tp_group_->world_size()
                   : 1;
    tp_rank_ = parallel_args.tp_group_ != nullptr
                   ? parallel_args.tp_group_->rank()
                   : 0;

    py::gil_scoped_acquire gil;
    if (tp_size_ > 1) {
      CHECK(!parallel_args.python_tp_rendezvous_host_.empty());
      CHECK_GT(parallel_args.python_tp_rendezvous_port_, 0);
      py::module_::import("xllm.python.ops")
          .attr("init_tp_group")(parallel_args.python_tp_rendezvous_host_,
                                  parallel_args.python_tp_rendezvous_port_,
                                  tp_rank_,
                                  tp_size_,
                                  c10::str(options_.device()));
    }

    py::module_ json = py::module_::import("json");
    py::module_ builtins = py::module_::import("builtins");
    const std::string config_path =
        ModelConfig::get_instance().model() + "/config.json";
    py::object config_file = builtins.attr("open")(config_path, "r");
    config_ = json.attr("load")(config_file).cast<py::dict>();
    config_file.attr("close")();
    config_[py::str("dtype")] = dtype_to_string(options_);
    config_[py::str("device")] = c10::str(options_.device());
    config_[py::str("tp_size")] = tp_size_;
    config_[py::str("tp_rank")] = tp_rank_;

    py::module_ vision_module =
        py::module_::import("xllm.python.models.kimi_k3_vit");
    py::object vision_class = vision_module.attr("KimiK3VisionModel");
    vision_model_ = vision_class(config_);
    vision_model_.attr("eval")();
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& parameters) {
    torch::Tensor pixel_values;
    if (auto value = parameters.multimodal.mm_data.get<torch::Tensor>(
            "pixel_values")) {
      pixel_values = value.value();
    }

    torch::Tensor grid_thws;
    if (auto value = parameters.multimodal.mm_data.get<torch::Tensor>(
            "image_grid_thw")) {
      grid_thws = value.value();
    }

    CHECK(pixel_values.defined()) << "Kimi K3 image pixels are missing";
    CHECK(grid_thws.defined()) << "Kimi K3 image grid_thw is missing";

    py::gil_scoped_acquire gil;
    py::object result = vision_model_.attr("forward")(pixel_values, grid_thws);
    auto embeddings = result.cast<std::vector<torch::Tensor>>();
    MMDict output;
    output["image|embedding"] = std::move(embeddings);
    return output;
  }

  torch::Tensor get_input_embeddings(const torch::Tensor& input_ids,
                                     const ModelInputParams& input_params) {
    auto inputs_embeds = torch::zeros(
        {input_ids.size(0), model_args_.hidden_size()}, options_);
    const auto& mm_data = input_params.multimodal.mm_data;
    if (!mm_data.valid()) {
      return inputs_embeds;
    }

    const auto multimodal_embeds =
        mm_data.get<torch::Tensor>("image|embedding");
    const auto multimodal_mask = mm_data.get<torch::Tensor>("image|mask");
    if (!multimodal_embeds.has_value() || !multimodal_mask.has_value()) {
      return inputs_embeds;
    }

    CHECK_EQ(multimodal_embeds->size(0), multimodal_mask->sum().item<int64_t>())
        << "Kimi K3 image embedding and mask sizes do not match";
    inputs_embeds.index_put_({multimodal_mask.value()},
                             multimodal_embeds.value());
    return inputs_embeds;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInputParams& parameters) {
    if (parameters.embedding.input_embedding.defined()) {
      return ModelOutput(parameters.embedding.input_embedding);
    }
    return ModelOutput(torch::zeros(
        {tokens.size(0), model_args_.hidden_size()}, options_));
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_indexes) {
    int64_t output_tokens = hidden_states.size(0);
    if (selected_indexes.defined()) {
      output_tokens = selected_indexes.size(0);
    }
    return torch::zeros({output_tokens, model_args_.vocab_size()}, options_);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    py::gil_scoped_acquire gil;
    py::module_::import("xllm_weight_loader");
    py::list state_dicts;
    for (const auto& state_dict : loader->get_state_dicts()) {
      state_dicts.append(
          py::cast(PyStateDict(state_dict.get()), py::return_value_policy::move));
    }
    vision_model_.attr("load_weights")(state_dicts, tp_rank_, tp_size_);
  }

  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) {}
  void update_expert_weight(int32_t) {}

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  py::dict config_;
  py::object vision_model_;
  int32_t tp_rank_ = 0;
  int32_t tp_size_ = 1;
};

class __attribute__((visibility("hidden"))) KimiK3ForConditionalGeneration
    : public torch::nn::ModuleHolder<KimiK3ForConditionalGenerationImpl> {
 public:
  using torch::nn::ModuleHolder<KimiK3ForConditionalGenerationImpl>::
      ModuleHolder;
};

using KimiK3MultimodalProcessor =
    MultimodalProcessor<KimiK25PromptProcessor, KimiK25ImageProcessor>;

REGISTER_MULTIMODAL_PROCESSOR(kimi_k3, KimiK3MultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(kimi_k3, KimiK3ForConditionalGeneration);

REGISTER_MODEL_ARGS(kimi_k3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "kimi_k3");
  LOAD_ARG_OR_FUNC(dtype, "dtype", [&] {
    return json.value_or<std::string>("torch_dtype", "bfloat16");
  });

  // The text decoder is a temporary shell. The vision tower and projector
  // use the real K3 dimensions, while this keeps KV allocation bounded until
  // the K3 language model is implemented.
  SET_ARG(n_layers, 1);
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 7168);
  SET_ARG(head_dim, 1);
  SET_ARG(n_heads, 1);
  SET_ARG(n_kv_heads, std::optional<int64_t>(1));
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 163840);
  LOAD_ARG_OR(max_position_embeddings,
              "text_config.max_position_embeddings",
              4096);
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 163585);
  LOAD_ARG_OR(pad_token_id, "text_config.pad_token_id", 163839);

  LOAD_ARG_OR(mm_num_channels, "vision_config.in_chans", 3);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_hidden_size, "vision_config.vt_hidden_size", 1024);
  LOAD_ARG_OR(mm_intermediate_size,
              "vision_config.vt_intermediate_size",
              4096);
  LOAD_ARG_OR(mm_num_attention_heads,
              "vision_config.vt_num_attention_heads",
              12);
  LOAD_ARG_OR(mm_num_hidden_layers,
              "vision_config.vt_num_hidden_layers",
              27);
  LOAD_ARG_OR_FUNC(mm_projection_dim, "vision_config.text_hidden_size", [&] {
    return json.value_or<int64_t>("text_config.hidden_size", 7168);
  });
  LOAD_ARG_OR(mm_projector_type,
              "vision_config.mm_projector_type",
              "patchmergerv2");
  LOAD_ARG_OR(mm_projector_hidden_act,
              "vision_config.projector_hidden_act",
              "gelu");
  LOAD_ARG_OR(mm_layer_norm_eps,
              "vision_config.projector_ln_eps",
              1e-5f);
  LOAD_ARG_OR_FUNC(mm_spatial_merge_size, "vision_config.merge_kernel_size", [&] {
    auto merge_kernel_size =
        json.value<std::vector<int64_t>>("vision_config.merge_kernel_size");
    return merge_kernel_size.has_value() && !merge_kernel_size->empty()
               ? (*merge_kernel_size)[0]
               : int64_t(2);
  });
  SET_ARG(mm_image_merge_size, args->mm_spatial_merge_size());
  LOAD_ARG_OR(mm_init_pos_emb_time,
              "vision_config.init_pos_emb_time",
              4);
  LOAD_ARG_OR(mm_init_pos_emb_width,
              "vision_config.init_pos_emb_width",
              64);
  LOAD_ARG_OR(mm_init_pos_emb_height,
              "vision_config.init_pos_emb_height",
              64);

  SET_ARG(vision_start_token_id, 163602);
  SET_ARG(vision_token_id, 163603);
  SET_ARG(vision_end_token_id, 163604);
  SET_ARG(image_token_id, 163605);
  SET_ARG(video_token_id, 163605);
  SET_ARG(mm_km_patch_size, 14);
  SET_ARG(mm_km_merge_kernel_size, 2);
  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>({0, 163585, 163586}));
});

REGISTER_TOKENIZER_ARGS(kimi_k3, [&] {
  SET_ARG(tokenizer_type, "tiktoken");
  SET_ARG(vocab_file, "tiktoken.model");
  SET_ARG(special_tokens,
          std::vector<SpecialToken>({{"[BOS]", 163584},
                                     {"[EOS]", 163585},
                                     {"<|im_end|>", 163586},
                                     {"<|im_user|>", 163587},
                                     {"<|im_assistant|>", 163588},
                                     {"<|media_begin|>", 163602},
                                     {"<|media_content|>", 163603},
                                     {"<|media_end|>", 163604},
                                     {"<|media_pad|>", 163605},
                                     {"[UNK]", 163838},
                                     {"[PAD]", 163839}}));
});

}  // namespace xllm
