/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/util/model_config_utils.h"
#include "models/model_registry.h"

namespace xllm {

REGISTER_MODEL_BACKEND(k3_dspark, "llm");

const bool kKimiK3DSparkTraitsRegistered = []() {
  ModelRegistry::register_npu_torch_only_model("k3_dspark");
  ModelRegistry::register_mla_model("k3_dspark");
  // DSparkDraftModel denotes the legacy Qwen3/GQA drafter. K3 MLA drafts
  // identify themselves through model_type or a K3-specific architecture.
  util::register_model_architecture_alias("K3DSparkModel", "k3_dspark");
  util::register_model_architecture_alias("K3DSparkForCausalLM",
                                          "k3_dspark");
  return true;
}();

REGISTER_MODEL_ARGS(k3_dspark, [&] {
  SET_ARG(model_type, "k3_dspark");
  SET_ARG(enable_mla, true);
  LOAD_ARG_OR_FUNC(dtype, "dtype", [&] {
    return json.value_or<std::string>("torch_dtype", "bfloat16");
  });
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 5);
  LOAD_ARG_OR(dspark_num_layers, "num_hidden_layers", 5);
  SET_ARG(dspark_num_target_layers, [&] {
    auto target_layer_ids = json.value<std::vector<int32_t>>(
        "dflash_config.target_layer_ids");
    if (!target_layer_ids.has_value()) {
      target_layer_ids =
          json.value<std::vector<int32_t>>("target_layer_ids");
    }
    if (target_layer_ids.has_value() && !target_layer_ids->empty()) {
      return static_cast<int32_t>(target_layer_ids->size());
    }
    const int32_t configured = json.value_or<int32_t>("num_target_layers", 5);
    return configured > 32 ? 5 : configured;
  }());
  LOAD_ARG_OR(hidden_size, "hidden_size", 7168);
  LOAD_ARG_OR(dspark_target_hidden_size, "target_hidden_size", 7168);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 14336);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
  SET_ARG(n_kv_heads, 1);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 128);
  SET_ARG(head_dim,
          args->qk_nope_head_dim() + args->qk_rope_head_dim());
  SET_ARG(rotary_dim, args->qk_rope_head_dim());
  LOAD_ARG_OR(vocab_size, "vocab_size", 163840);
  LOAD_ARG_OR(markov_rank, "markov_rank", 256);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5f);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR_FUNC(rope_theta, "rope_parameters.rope_theta", [&] {
    return json.value_or<float>("rope_theta", 1000000.0f);
  });
  LOAD_ARG_OR(rope_scaling_factor, "rope_parameters.factor", 32.0f);
  LOAD_ARG_OR(rope_scaling_beta_fast, "rope_parameters.beta_fast", 32);
  LOAD_ARG_OR(rope_scaling_beta_slow, "rope_parameters.beta_slow", 1);
  LOAD_ARG_OR(rope_scaling_mscale, "rope_parameters.mscale", 1.0f);
  LOAD_ARG_OR(
      rope_scaling_mscale_all_dim, "rope_parameters.mscale_all_dim", 0.0f);
  LOAD_ARG_OR_FUNC(rope_scaling_original_max_position_embeddings,
                   "rope_parameters.original_max_position_embeddings",
                   [&] { return args->max_position_embeddings(); });
  LOAD_ARG_OR(dspark_block_size, "block_size", 7);
  SET_ARG(hidden_act, "silu");
  SET_ARG(layer_types,
          std::vector<std::string>(
              static_cast<size_t>(args->n_layers()), "full_attention"));
});

}  // namespace xllm
