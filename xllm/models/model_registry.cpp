/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "model_registry.h"

#include <glog/logging.h>

#include <iostream>
#include <mutex>
#include <optional>
#include <unordered_set>
#include <vector>

#include "core/framework/config/kernel_config.h"
#include "core/framework/config/model_config.h"
#include "core/util/dit_model_discovery.h"
#include "llm/py_causal_lm.h"
#include "models.h"
#include "models/vlm/py_causal_vlm.h"
#include "processors/kimi25_image_processor.h"
#include "processors/multimodal_processor.h"

namespace {

// Safe logging macro to avoid crashes during static initialization
#define SAFE_LOG_WARNING(message)                       \
  do {                                                  \
    if (google::IsGoogleLoggingInitialized()) {         \
      LOG(WARNING) << message;                          \
    } else {                                            \
      std::cerr << "WARNING: " << message << std::endl; \
    }                                                   \
  } while (0)

#define SAFE_LOG_ERROR(message)                       \
  do {                                                \
    if (google::IsGoogleLoggingInitialized()) {       \
      LOG(ERROR) << message;                          \
    } else {                                          \
      std::cerr << "ERROR: " << message << std::endl; \
    }                                                 \
  } while (0)

#define SAFE_LOG_INFO(message)                       \
  do {                                               \
    if (google::IsGoogleLoggingInitialized()) {      \
      LOG(INFO) << message;                          \
    } else {                                         \
      std::cerr << "INFO: " << message << std::endl; \
    }                                                \
  } while (0)

}  // anonymous namespace

namespace xllm {

namespace {

using KimiK3MultimodalProcessor =
    MultimodalProcessor<KimiK25PromptProcessor, KimiK25ImageProcessor>;

REGISTER_MULTIMODAL_PROCESSOR(kimi_k3, KimiK3MultimodalProcessor);
const bool kimi_k3_python_vlm_registered = []() {
  ModelRegistry::register_model_backend("kimi_k3", "vlm");
  return true;
}();

REGISTER_MODEL_ARGS(kimi_k3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "kimi_k3");
  LOAD_ARG_OR_FUNC(dtype, "dtype", [&] {
    return json.value_or<std::string>("torch_dtype", "bfloat16");
  });

  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 93);
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 7168);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 96);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 96);
  LOAD_ARG_OR(qk_nope_head_dim, "text_config.qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "text_config.qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "text_config.v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "text_config.q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "text_config.kv_lora_rank", 512);
  SET_ARG(head_dim, args->qk_nope_head_dim() + args->qk_rope_head_dim());
  SET_ARG(rotary_dim, args->qk_rope_head_dim());
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 163840);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 1048576);
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 163586);
  LOAD_ARG_OR(pad_token_id, "text_config.pad_token_id", 163839);

  // Kimi-K3 Delta Attention (KDA / linear attention). These flat linear_*
  // fields drive the shared C++ linear-attention cache path
  // (has_linear_attention_layers -> KVCacheShape conv/ssm allocation). Kimi-K3
  // stores its KDA config nested under text_config.linear_attn_config, and it
  // shares one head_dim / num_heads across q/k/v (unlike Qwen GDN), so key and
  // value dims/heads map to the same source field.
  LOAD_ARG_OR(linear_conv_kernel_dim,
              "text_config.linear_attn_config.short_conv_kernel_size",
              4);
  LOAD_ARG_OR(
      linear_key_head_dim, "text_config.linear_attn_config.head_dim", 128);
  LOAD_ARG_OR(
      linear_value_head_dim, "text_config.linear_attn_config.head_dim", 128);
  LOAD_ARG_OR(
      linear_num_key_heads, "text_config.linear_attn_config.num_heads", 96);
  LOAD_ARG_OR(
      linear_num_value_heads, "text_config.linear_attn_config.num_heads", 96);
  // The recurrent (ssm) state must stay fp32 to match the KDA layer contract
  // (kda.py state_dtypes); the KVCache create-option default is bf16.
  SET_ARG(mamba_ssm_dtype, "float32");
  // Build per-layer types from the 1-based kda_layers list so both
  // has_linear_attention_layers and the per-layer cache dispatch classify
  // Kimi-K3's irregular KDA layout correctly.
  [&] {
    int64_t n_layers = args->n_layers();
    auto kda_layers = json.value<std::vector<int64_t>>(
        "text_config.linear_attn_config.kda_layers");
    if (!kda_layers.has_value() || kda_layers->empty() || n_layers <= 0) {
      return;
    }
    std::unordered_set<int64_t> kda_layer_set(kda_layers->begin(),
                                              kda_layers->end());
    std::vector<std::string> layer_types;
    layer_types.reserve(n_layers);
    for (int64_t layer_id = 0; layer_id < n_layers; ++layer_id) {
      // kda_layers is 1-based; a 0-based layer_id maps to layer_id + 1.
      const bool is_kda = kda_layer_set.count(layer_id + 1) > 0;
      layer_types.emplace_back(is_kda ? "linear_attention" : "full_attention");
    }
    args->layer_types() = std::move(layer_types);
  }();

  LOAD_ARG_OR(mm_num_channels, "vision_config.in_chans", 3);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  LOAD_ARG_OR(mm_hidden_size, "vision_config.vt_hidden_size", 1024);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.vt_intermediate_size", 4096);
  LOAD_ARG_OR(
      mm_num_attention_heads, "vision_config.vt_num_attention_heads", 12);
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.vt_num_hidden_layers", 27);
  LOAD_ARG_OR_FUNC(mm_projection_dim, "vision_config.text_hidden_size", [&] {
    return json.value_or<int64_t>("text_config.hidden_size", 7168);
  });
  LOAD_ARG_OR(
      mm_projector_type, "vision_config.mm_projector_type", "patchmergerv2");
  LOAD_ARG_OR(
      mm_projector_hidden_act, "vision_config.projector_hidden_act", "gelu");
  LOAD_ARG_OR(mm_layer_norm_eps, "vision_config.projector_ln_eps", 1e-5f);
  [&] {
    auto merge_kernel_size =
        json.value<std::vector<int64_t>>("vision_config.merge_kernel_size");
    args->mm_spatial_merge_size() =
        merge_kernel_size.has_value() && !merge_kernel_size->empty()
            ? (*merge_kernel_size)[0]
            : int64_t(2);
  }();
  SET_ARG(mm_image_merge_size, args->mm_spatial_merge_size());
  LOAD_ARG_OR(mm_init_pos_emb_time, "vision_config.init_pos_emb_time", 4);
  LOAD_ARG_OR(mm_init_pos_emb_width, "vision_config.init_pos_emb_width", 64);
  LOAD_ARG_OR(mm_init_pos_emb_height, "vision_config.init_pos_emb_height", 64);

  SET_ARG(vision_start_token_id, 163602);
  SET_ARG(vision_token_id, 163603);
  SET_ARG(vision_end_token_id, 163604);
  SET_ARG(image_token_id, 163605);
  SET_ARG(video_token_id, 163605);
  SET_ARG(mm_km_patch_size, 14);
  SET_ARG(mm_km_merge_kernel_size, 2);
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({0, 163585, 163586}));
});

REGISTER_TOKENIZER_ARGS(kimi_k3, [&] {
  SET_ARG(tokenizer_type, "tiktoken");
  SET_ARG(vocab_file, "tiktoken.model");
  SET_ARG(special_tokens,
          std::vector<SpecialToken>({{"[BOS]", 163584},
                                     {"[EOS]", 163585},
                                     {"<|end_of_msg|>", 163586},
                                     {"<|open|>", 163587},
                                     {"<|close|>", 163588},
                                     {"<|sep|>", 163589},
                                     {"[start_header_id]", 163590},
                                     {"[end_header_id]", 163591},
                                     {"[EOT]", 163593},
                                     {"<|media_begin|>", 163602},
                                     {"<|media_content|>", 163603},
                                     {"<|media_end|>", 163604},
                                     {"<|media_pad|>", 163605},
                                     {"[UNK]", 163838},
                                     {"[PAD]", 163839}}));
  SET_ARG(visible_special_tokens,
          std::vector<std::string>(
              {"<|end_of_msg|>", "<|open|>", "<|close|>", "<|sep|>"}));
  SET_ARG(
      pattern,
      R"([\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+[^\s]|\s+)");
});

#if defined(USE_NPU)
constexpr char kAutoBackend[] = "AUTO";
constexpr char kAtbBackend[] = "ATB";
constexpr char kTorchBackend[] = "TORCH";

bool is_torch_only_model_type(const std::string& model_type) {
  static const std::unordered_set<std::string> kTorchOnlyModelTypes = {
      "deepseek_v4",
      "deepseek_v4_mtp",
      "qwen3_5",
      "qwen3_5_text",
      "qwen3_5_moe",
      "qwen3_5_moe_text",
      "qwen3_5_mtp",
      "qwen3_5_moe_mtp",
      "qwen3_next",
      "minimax_m2",
      "kimi_k3"};
  return kTorchOnlyModelTypes.count(model_type) != 0;
}
#endif

}  // namespace

bool resolve_model_registration(const std::string& model_type,
                                const std::string& requested_npu_kernel_backend,
                                std::string* effective_npu_kernel_backend,
                                std::string* resolved_name,
                                std::string* error_message) {
  if (resolved_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "resolved_name must not be null";
    }
    return false;
  }

#if defined(USE_NPU)
  const std::string backend = requested_npu_kernel_backend.empty()
                                  ? kAutoBackend
                                  : requested_npu_kernel_backend;
  if (backend != kAutoBackend && backend != kAtbBackend &&
      backend != kTorchBackend) {
    if (error_message != nullptr) {
      *error_message = "Unsupported --npu_kernel_backend=" + backend +
                       ". Supported values: AUTO, ATB, TORCH.";
    }
    return false;
  }

  std::string effective_backend = backend;
  if (backend == kAutoBackend) {
    effective_backend =
        is_torch_only_model_type(model_type) ? kTorchBackend : kAtbBackend;
  } else if (model_type == "qwen3" || model_type == "qwen3_moe" ||
             model_type == "deepseek_v32") {
    // qwen3/qwen3_moe/deepseek_v32 support both backends.
  } else if (is_torch_only_model_type(model_type)) {
    if (backend != kTorchBackend) {
      if (error_message != nullptr) {
        *error_message = "Model type " + model_type +
                         " only supports --npu_kernel_backend=TORCH.";
      }
      return false;
    }
  } else if (backend != kAtbBackend) {
    if (error_message != nullptr) {
      *error_message = "Model type " + model_type +
                       " only supports --npu_kernel_backend=ATB.";
    }
    return false;
  }

  if (effective_npu_kernel_backend != nullptr) {
    *effective_npu_kernel_backend = effective_backend;
  }
  if (model_type == "qwen3" && effective_backend == kAtbBackend) {
    *resolved_name = "qwen3_atb";
  } else if (model_type == "qwen3_moe" && effective_backend == kAtbBackend) {
    *resolved_name = "qwen3_moe_atb";
  } else {
    *resolved_name = model_type;
  }
  return true;
#else
  *resolved_name = model_type;
  return true;
#endif
}

bool resolve_model_registration_name(const std::string& model_type,
                                     std::string* resolved_name,
                                     std::string* error_message) {
#if defined(USE_NPU)
  return resolve_model_registration(
      model_type,
      ::xllm::KernelConfig::get_instance().npu_kernel_backend(),
      nullptr,
      resolved_name,
      error_message);
#else
  return resolve_model_registration(
      model_type, "", nullptr, resolved_name, error_message);
#endif
}

bool is_npu_model_cp_capable(const std::string& resolved_name) {
  static const std::unordered_set<std::string> kCpCapableModels = {
      "deepseek_v32",
      "deepseek_v32_mtp",
      "glm_moe_dsa",
      "glm_moe_dsa_mtp",
  };
  static std::once_flag once;
  std::call_once(once, []() {
    for (const std::string& name : kCpCapableModels) {
      ModelRegistry::register_cp_sharding_mode(name, CpShardingMode::NPU_MODEL);
    }
  });
  return ModelRegistry::get_cp_sharding_mode(resolved_name) ==
         CpShardingMode::NPU_MODEL;
}

ModelRegistry* ModelRegistry::get_instance() {
  static ModelRegistry registry;

  return &registry;
}

void ModelRegistry::register_causallm_factory(const std::string& name,
                                              CausalLMFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].causal_lm_factory != nullptr) {
    SAFE_LOG_WARNING("causal lm factory for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].causal_lm_factory = factory;
    instance->model_backend_[name] = "llm";
  }
}

void ModelRegistry::register_rec_model_factory(const std::string& name,
                                               RecModelFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].rec_model_factory != nullptr) {
    SAFE_LOG_WARNING("rec model factory for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].rec_model_factory = factory;
    instance->model_backend_[name] = "rec";
  }
}

void ModelRegistry::register_causalvlm_factory(const std::string& name,
                                               CausalVLMFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].causal_vlm_factory != nullptr) {
    SAFE_LOG_WARNING("causal vlm factory for " << name
                                               << " already registered.");
  } else {
    instance->model_registry_[name].causal_vlm_factory = factory;
    instance->model_backend_[name] = "vlm";
  }
}

void ModelRegistry::register_model_backend(const std::string& name,
                                           const std::string& backend) {
  ModelRegistry* instance = get_instance();
  instance->model_backend_[name] = backend;
}

void ModelRegistry::register_dit_model_factory(const std::string& name,
                                               DiTModelFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].dit_model_factory != nullptr) {
    SAFE_LOG_WARNING("DiT model factory for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].dit_model_factory = factory;
    instance->model_backend_[name] = "dit";
  }
}

void ModelRegistry::register_multimodal_processor_factory(
    const std::string& name,
    MultimodalProcessorFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].multimodal_processor_factory != nullptr) {
    SAFE_LOG_WARNING("multimodal processor factory for "
                     << name << " already registered.");
  } else {
    instance->model_registry_[name].multimodal_processor_factory =
        std::move(factory);
  }
}

void ModelRegistry::register_model_args_loader(const std::string& name,
                                               ModelArgsLoader loader) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].model_args_loader != nullptr) {
    SAFE_LOG_WARNING("model args loader for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].model_args_loader = loader;
  }
}

void ModelRegistry::register_quant_args_loader(const std::string& name,
                                               QuantArgsLoader loader) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].quant_args_loader != nullptr) {
    SAFE_LOG_WARNING("quant args loader for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].quant_args_loader = loader;
  }
}

void ModelRegistry::register_tokenizer_args_loader(const std::string& name,
                                                   TokenizerArgsLoader loader) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].tokenizer_args_loader != nullptr) {
    SAFE_LOG_WARNING("tokenizer args loader for " << name
                                                  << " already registered.");
  } else {
    instance->model_registry_[name].tokenizer_args_loader = loader;
  }
}

void ModelRegistry::register_cp_sharding_mode(const std::string& name,
                                              CpShardingMode mode) {
  ModelRegistry* instance = get_instance();
  instance->model_registry_[name].cp_sharding_mode = mode;
}

CpShardingMode ModelRegistry::get_cp_sharding_mode(const std::string& name) {
  ModelRegistry* instance = get_instance();
  const auto it = instance->model_registry_.find(name);
  if (it == instance->model_registry_.end()) {
    return CpShardingMode::NONE;
  }
  return it->second.cp_sharding_mode;
}

CausalLMFactory ModelRegistry::get_causallm_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].causal_lm_factory;
}

RecModelFactory ModelRegistry::get_rec_model_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].rec_model_factory;
}

CausalVLMFactory ModelRegistry::get_causalvlm_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].causal_vlm_factory;
}

DiTModelFactory ModelRegistry::get_dit_model_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_registry_[name].dit_model_factory;
}

MultimodalProcessorFactory ModelRegistry::get_multimodal_processor_factory(
    const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_registry_[name].multimodal_processor_factory;
}

ModelArgsLoader ModelRegistry::get_model_args_loader(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].model_args_loader;
}

QuantArgsLoader ModelRegistry::get_quant_args_loader(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].quant_args_loader;
}

TokenizerArgsLoader ModelRegistry::get_tokenizer_args_loader(
    const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].tokenizer_args_loader;
}

bool ModelRegistry::has_dit_model_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();
  const auto it = instance->model_registry_.find(name);
  if (it == instance->model_registry_.end()) {
    return false;
  }
  return it->second.dit_model_factory != nullptr;
}

namespace util {

namespace {

std::string try_resolve_from_component_key(const std::string& key) {
  if (key.empty()) {
    return {};
  }
  if (ModelRegistry::has_dit_model_factory(key)) {
    return key;
  }

  auto try_prefix = [](const std::string& prefix) -> std::string {
    if (ModelRegistry::has_dit_model_factory(prefix)) {
      return prefix;
    }
    for (const char* suffix : {"_dlm", "_dit", "_diffusion", "_model"}) {
      const std::string candidate = prefix + suffix;
      if (ModelRegistry::has_dit_model_factory(candidate)) {
        return candidate;
      }
    }
    return {};
  };

  if (key.size() > 4 && key.substr(key.size() - 4) == "_dit") {
    if (std::string resolved = try_prefix(key.substr(0, key.size() - 4));
        !resolved.empty()) {
      return resolved;
    }
  }
  if (key.size() > 4 && key.substr(key.size() - 4) == "_vae") {
    if (std::string resolved = try_prefix(key.substr(0, key.size() - 4));
        !resolved.empty()) {
      return resolved;
    }
  }
  return {};
}

}  // namespace

std::string resolve_dit_pipeline_type(
    const std::vector<DitDiscoveredComponent>& components) {
  if (components.empty()) {
    return {};
  }

  for (const auto& component : components) {
    if (std::string resolved =
            try_resolve_from_component_key(component.component_type);
        !resolved.empty()) {
      return resolved;
    }
    if (component.name != component.component_type) {
      if (std::string resolved = try_resolve_from_component_key(component.name);
          !resolved.empty()) {
        return resolved;
      }
    }
  }

  std::string component_summary;
  for (const auto& component : components) {
    if (!component_summary.empty()) {
      component_summary += "; ";
    }
    component_summary +=
        component.name + " (model_type=" + component.component_type + ")";
  }
  LOG(FATAL) << "Unable to resolve a registered DiT pipeline type from "
                "discovered components: "
             << component_summary;
  return {};
}

}  // namespace util

std::string ModelRegistry::get_model_backend(const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_backend_[name];
}

std::unique_ptr<CausalLM> create_llm_model(const ModelContext& context) {
  // Python model executor: build the graph via the embedded interpreter instead
  // of resolving a C++ model class from the registry.
  const auto& model_impl = context.get_model_impl();
#if defined(USE_CUDA) || defined(USE_NPU)
  if (ModelConfig::is_python_model_impl(model_impl)) {
    return std::make_unique<PyCausalLM>(context);
  }
#else
  if (ModelConfig::is_python_model_impl(model_impl)) {
    LOG(ERROR) << "--model_impl=python is only supported on CUDA/NPU builds.";
    return nullptr;
  }
#endif

  std::string resolved_name;
  std::string error_message;
  if (!resolve_model_registration_name(context.get_model_args().model_type(),
                                       &resolved_name,
                                       &error_message)) {
    LOG(ERROR) << error_message;
    return nullptr;
  }

  auto factory = ModelRegistry::get_causallm_factory(resolved_name);
  if (factory) {
    return factory(context);
  }

  LOG(ERROR) << "Unsupported model type: "
             << context.get_model_args().model_type();

  return nullptr;
}

std::unique_ptr<CausalLM> create_rec_model(const ModelContext& context) {
  std::string resolved_name;
  std::string error_message;
  if (!resolve_model_registration_name(context.get_model_args().model_type(),
                                       &resolved_name,
                                       &error_message)) {
    LOG(ERROR) << error_message;
    return nullptr;
  }

  auto factory = ModelRegistry::get_rec_model_factory(resolved_name);
  if (factory) {
    return factory(context);
  }

  LOG(ERROR) << "Unsupported rec model type: "
             << context.get_model_args().model_type();

  return nullptr;
}

std::unique_ptr<CausalVLM> create_vlm_model(const ModelContext& context) {
  const auto& model_impl = context.get_model_impl();
#if defined(USE_CUDA) || defined(USE_NPU)
  if (ModelConfig::is_python_model_impl(model_impl)) {
    return std::make_unique<PyCausalVLM>(context);
  }
#else
  if (ModelConfig::is_python_model_impl(model_impl)) {
    LOG(ERROR) << "--model_impl=python is only supported on CUDA/NPU builds.";
    return nullptr;
  }
#endif

  std::string resolved_name;
  std::string error_message;
  if (!resolve_model_registration_name(context.get_model_args().model_type(),
                                       &resolved_name,
                                       &error_message)) {
    LOG(ERROR) << error_message;
    return nullptr;
  }

  auto factory = ModelRegistry::get_causalvlm_factory(resolved_name);
  if (factory) {
    return factory(context);
  }

  LOG(ERROR) << "Unsupported model type: "
             << context.get_model_args().model_type();

  return nullptr;
}

std::unique_ptr<DiTModel> create_dit_model(const DiTModelContext& context) {
  // get the factory function for the model type from model registry
  auto factory = ModelRegistry::get_dit_model_factory(context.model_type());
  if (factory) {
    return factory(context);
  }
  LOG(INFO) << "DiT Model type: " << context.model_type();
  LOG(ERROR) << "Unsupported model type: " << context.model_type();

  return nullptr;
}

}  // namespace xllm
