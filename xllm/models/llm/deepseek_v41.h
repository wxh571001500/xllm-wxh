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

#pragma once

#include <absl/strings/str_join.h>
#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/util/json_reader.h"
#include "models/model_registry.h"

namespace xllm {

// DeepSeek V4.1 Flash config.json -> ModelArgs loader plus its args policy.
//
// deepseek_v41 is a TORCH-only, python-impl model (like glm5_next): this
// header registers ONLY the args loader and the "llm" model backend; the
// CausalLM is built by PyCausalLM via xllm.python.models.deepseek_v41.
//
// The V4.1 HF config keeps model_type / dtype / token ids / image_token_id at
// the root and nests every text-side field under "text_config" (verified
// against /weights/DeepSeek/DeepSeek-V4.1-Flash/config.json), so unlike the
// V4 loader every text field is read with an explicit "text_config." prefix.
struct DeepseekV41ArgsPolicy {
  std::unordered_set<int32_t> supported_compress_ratios;
  std::unordered_set<std::string> supported_score_funcs;
};

inline DeepseekV41ArgsPolicy build_deepseek_v41_args_policy() {
  DeepseekV41ArgsPolicy policy;
  // V4.1 compress_ratios are NOT normalized (unlike V4): 0 (pure SWA), 1
  // (CSA2(1) decoder) and 2 (CSA2(2) encoder) select different cache groups
  // and attention paths, so they must stay distinct. 4 is accepted for
  // forward compatibility with the V4 family.
  policy.supported_compress_ratios = {0, 1, 2, 4};
  policy.supported_score_funcs = {"softmax", "sigmoid", "sqrtsoftplus"};
  return policy;
}

inline void process_deepseek_v41_args(ModelArgs* args) {
  // Keep alias fields consistent after loading.
  SET_ARG(n_activated_experts, args->num_experts_per_tok());

  // V4.1 removed hash routing (gate.tid2eid); the field stays 0 for the
  // inherited V4 machinery (python does the same override).
  SET_ARG(n_hash_layers, 0);

  // Build stop token set from eos for runtime usage.
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));

  // NOTE: no compress_ratios normalization and no tail-padding here. The
  // 43-entry list (40 backbone + 3 DSpark) is kept exactly as configured;
  // validate_deepseek_v41_args only checks the per-layer entries exist and
  // are supported.
}

inline void validate_deepseek_v41_args(const ModelArgs& args,
                                       const DeepseekV41ArgsPolicy& policy) {
  CHECK(!policy.supported_compress_ratios.empty())
      << "deepseek_v41 internal supported_compress_ratios must not be empty";
  CHECK_GT(args.n_layers(), 0)
      << "deepseek_v41 config num_hidden_layers must be > 0, got "
      << args.n_layers();
  CHECK_GE(static_cast<int64_t>(args.compress_ratios().size()), args.n_layers())
      << "deepseek_v41 config compress_ratios must have one entry per layer "
         "(backbone + DSpark tail), got "
      << args.compress_ratios().size() << " vs " << args.n_layers();
  for (size_t i = 0; i < args.compress_ratios().size(); ++i) {
    const int32_t ratio = args.compress_ratios()[i];
    CHECK(policy.supported_compress_ratios.count(ratio) > 0)
        << "deepseek_v41 config compress_ratios[" << i
        << "] must be in supported_compress_ratios, got " << ratio;
  }
  // A kv-source layer is consumed by three places (llm_engine.cpp,
  // kv_cache_estimation.cpp, deepseek_v4_kv_cache_impl.cpp), all of which only
  // give it a TOKEN manager when its compress ratio is 1 or 2. An out-of-range
  // id or a ratio-0/4 source is silently skipped there, so validate the
  // contract here and fail fast at load time instead of diverging quietly.
  for (const int32_t layer_id : args.kv_source_layer_ids()) {
    CHECK_GE(layer_id, 0)
        << "deepseek_v41 config kv_source_layer_ids entry must be >= 0, got "
        << layer_id;
    CHECK_LT(layer_id, static_cast<int32_t>(args.compress_ratios().size()))
        << "deepseek_v41 config kv_source_layer_ids entry must index into "
           "compress_ratios, got "
        << layer_id << " vs " << args.compress_ratios().size();
    const int32_t ratio = args.compress_ratios()[static_cast<size_t>(layer_id)];
    CHECK(ratio == 1 || ratio == 2)
        << "deepseek_v41 config kv_source_layer_ids[" << layer_id
        << "] must reference a ratio 1/2 layer (only those own a V4.1 TOKEN "
           "manager), got ratio "
        << ratio;
  }
  CHECK_GT(args.window_size(), 0)
      << "deepseek_v41 config window_size/sliding_window must be > 0, got "
      << args.window_size();
  // SWA group chunk width is taken from window_size while the device block
  // width is taken from block_size; anchoring the two (equality) is pending
  // confirmation of the online config's window_size, so only block_size is
  // asserted here.
  CHECK_EQ(KVCacheConfig::get_instance().block_size(), 128)
      << "DeepSeek V4.1 currently only supports block_size=128, got "
      << KVCacheConfig::get_instance().block_size();
  CHECK_GT(args.n_routed_experts(), 0)
      << "deepseek_v41 config n_routed_experts must be > 0, got "
      << args.n_routed_experts();
  CHECK_GT(args.n_activated_experts(), 0)
      << "deepseek_v41 config num_experts_per_tok must be > 0, got "
      << args.n_activated_experts();
  CHECK_LE(args.n_activated_experts(), args.n_routed_experts())
      << "deepseek_v41 config num_experts_per_tok must be <= n_routed_experts"
      << ", got " << args.n_activated_experts() << " vs "
      << args.n_routed_experts();
  CHECK_EQ(args.n_hash_layers(), 0)
      << "deepseek_v41 removed hash routing (gate.tid2eid); "
         "num_hash_layers must stay 0, got "
      << args.n_hash_layers();
  CHECK_GT(args.routed_scaling_factor(), 0.0f)
      << "deepseek_v41 config routed_scaling_factor must be > 0, got "
      << args.routed_scaling_factor();
  CHECK_GT(args.swiglu_limit(), 0.0f)
      << "deepseek_v41 config swiglu_limit must be > 0, got "
      << args.swiglu_limit();
  CHECK(!args.scoring_func().empty())
      << "deepseek_v41 config scoring_func must not be empty";
  {
    std::string score_func = args.scoring_func();
    std::transform(
        score_func.begin(),
        score_func.end(),
        score_func.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    CHECK(policy.supported_score_funcs.count(score_func) > 0)
        << "deepseek_v41 config scoring_func must be in "
        << absl::StrJoin(policy.supported_score_funcs, ", ") << ", got "
        << args.scoring_func();
  }
  CHECK_GT(args.index_head_dim(), 0)
      << "deepseek_v41 config index_head_dim must be > 0, got "
      << args.index_head_dim();
  CHECK_GT(args.index_n_heads(), 0)
      << "deepseek_v41 config index_n_heads must be > 0, got "
      << args.index_n_heads();
  CHECK_GT(args.index_topk(), 0)
      << "deepseek_v41 config index_topk must be > 0, got "
      << args.index_topk();
  CHECK_GT(args.hc_mult(), 0)
      << "deepseek_v41 config hc_mult must be > 0, got " << args.hc_mult();
  CHECK_GE(args.hc_sinkhorn_iters(), 0)
      << "deepseek_v41 config hc_sinkhorn_iters must be >= 0, got "
      << args.hc_sinkhorn_iters();
  CHECK_GT(args.hc_eps(), 0.0f)
      << "deepseek_v41 config hc_eps must be > 0, got " << args.hc_eps();
  CHECK_GT(args.factor(), 0.0f)
      << "deepseek_v41 requires positive rope_scaling.factor, got "
      << args.factor();
  CHECK_GT(args.rope_scaling_attn_factor(), 0.0f)
      << "deepseek_v41 requires positive rope_scaling.attn_factor, got "
      << args.rope_scaling_attn_factor();
  CHECK_GT(args.rope_theta(), 0.0f)
      << "deepseek_v41 requires positive rope_theta, got " << args.rope_theta();
  CHECK_GT(args.compress_rope_theta(), 0.0f)
      << "deepseek_v41 requires positive compress_rope_theta, got "
      << args.compress_rope_theta();
}

inline bool load_deepseek_v41_model_args(const JsonReader& json,
                                         ModelArgs* args) {
  // --------------------------------------------------------------------------
  // Root-level fields (model identity, dtype, token ids, image token id).
  // --------------------------------------------------------------------------
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v41");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");

  // --------------------------------------------------------------------------
  // Base args (shared/common HF CausalLM args), all nested under text_config.
  // Defaults match the released DeepSeek-V4.1-Flash config.
  // --------------------------------------------------------------------------
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 5120);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 40);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 64);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 1);
  // V4.1 head_dim (512) is NOT hidden_size / n_heads; never derive it.
  LOAD_ARG_OR(head_dim, "text_config.head_dim", 512);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");

  // LoRA / groups
  LOAD_ARG_OR(q_lora_rank, "text_config.q_lora_rank", 1280);
  LOAD_ARG_OR(o_lora_rank, "text_config.o_lora_rank", 1024);
  LOAD_ARG_OR(o_groups, "text_config.o_groups", 8);
  LOAD_ARG_OR(qk_rope_head_dim, "text_config.qk_rope_head_dim", 64);
  LOAD_ARG_OR(rope_head_dim, "text_config.qk_rope_head_dim", 64);

  // MoE (loaded before intermediate_size so its fallback can read it)
  LOAD_ARG_OR(n_routed_experts, "text_config.n_routed_experts", 384);
  LOAD_ARG_OR(n_activated_experts, "text_config.n_activated_experts", 6);
  LOAD_ARG_OR(num_experts_per_tok,
              "text_config.num_experts_per_tok",
              args->n_activated_experts());
  LOAD_ARG_OR(n_shared_experts, "text_config.n_shared_experts", 1);
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 2304);
  LOAD_ARG_OR(routed_scaling_factor, "text_config.routed_scaling_factor", 1.5f);
  LOAD_ARG_OR(scoring_func, "text_config.scoring_func", "sqrtsoftplus");
  LOAD_ARG_OR(swiglu_limit, "text_config.swiglu_limit", 10.0f);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(topk_method, "text_config.topk_method", "noaux_tc");

  LOAD_ARG_OR_FUNC(intermediate_size, "text_config.intermediate_size", [&] {
    if (args->intermediate_size() > 0) {
      return args->intermediate_size();
    }
    if (args->moe_intermediate_size() > 0) {
      return static_cast<int64_t>(args->moe_intermediate_size());
    }
    if (args->hidden_size() > 0) {
      return args->hidden_size() * 4;
    }
    return int64_t{0};
  });

  // Norm / RoPE
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-20f);
  LOAD_ARG_OR(rope_theta, "text_config.rope_theta", 10000.0f);
  LOAD_ARG_OR(factor, "text_config.rope_scaling.factor", 16.0f);
  LOAD_ARG_OR(rope_scaling_factor, "text_config.rope_scaling.factor", 16.0f);
  LOAD_ARG_OR(beta_fast, "text_config.rope_scaling.beta_fast", 32.0f);
  LOAD_ARG_OR(beta_slow, "text_config.rope_scaling.beta_slow", 1.0f);
  LOAD_ARG_OR(
      rope_scaling_attn_factor, "text_config.rope_scaling.attn_factor", 1.0f);
  LOAD_ARG_OR(
      rope_scaling_rope_type, "text_config.rope_scaling.rope_type", "yarn");
  LOAD_ARG_OR(rope_scaling_original_max_position_embeddings,
              "text_config.rope_scaling.original_max_position_embeddings",
              65536);

  // --------------------------------------------------------------------------
  // DeepSeek V4.1 args.
  // --------------------------------------------------------------------------
  // KV compression / windowing. The HF config spells the window as
  // sliding_window; the reference inference config uses window_size.
  LOAD_ARG(compress_ratios, "text_config.compress_ratios");
  LOAD_ARG_OR(
      compress_rope_theta, "text_config.compress_rope_theta", 160000.0f);
  LOAD_ARG_OR_FUNC(window_size, "text_config.window_size", [&] {
    return json.value_or<int32_t>("text_config.sliding_window", 128);
  });

  // Indexer
  LOAD_ARG_OR(index_head_dim, "text_config.index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "text_config.index_n_heads", 32);
  LOAD_ARG_OR(index_topk, "text_config.index_topk", 512);

  // HC / DSA helpers
  LOAD_ARG_OR(hc_mult, "text_config.hc_mult", 4);
  LOAD_ARG_OR(hc_sinkhorn_iters, "text_config.hc_sinkhorn_iters", 20);
  LOAD_ARG_OR(hc_eps, "text_config.hc_eps", 1e-6f);
  LOAD_ARG_OR(scale_fmt, "scale_fmt", "ue8m0");
  LOAD_ARG_OR(scale_fmt, "quantization_config.scale_fmt", args->scale_fmt());

  // CSA2/CED shared-cache layer plan and candidate pool. Defaults are the
  // disabled sentinels from the V4.1 contract; real configs always set them.
  LOAD_ARG(kv_source_layer_ids, "text_config.kv_source_layer_ids");
  LOAD_ARG(index_source_layer_ids, "text_config.index_source_layer_ids");
  LOAD_ARG_OR(
      candidate_source_layer_id, "text_config.candidate_source_layer_id", -1);
  LOAD_ARG_OR(candidate_topk_blocks, "text_config.candidate_topk_blocks", 0);
  LOAD_ARG_OR(candidate_block_size, "text_config.candidate_block_size", 0);

  // Engram sparse-recall tables.
  LOAD_ARG(engram_layer_ids, "text_config.engram_layer_ids");
  LOAD_ARG(engram_num_embeddings, "text_config.engram_num_embeddings");
  LOAD_ARG_OR(engram_max_ngram_size, "text_config.engram_max_ngram_size", 0);
  LOAD_ARG_OR(engram_vocab_size, "text_config.engram_vocab_size", 0);
  LOAD_ARG_OR(engram_n_heads, "text_config.engram_n_heads", 0);
  LOAD_ARG_OR(engram_head_dim, "text_config.engram_head_dim", 0);
  // -1 is the "unset" sentinel, matching DeepseekV41Config.engram_pad_token_id
  // in xllm/python/models/deepseek_v41.py. This args loader only feeds the
  // Python impl via PyCausalLM::build_config_dict; C++ never indexes a tensor
  // with this value. Keeping 2 here would overwrite Python's sentinel in the
  // service path and defeat its "engram enabled but pad id unset" guard.
  LOAD_ARG_OR(engram_pad_token_id, "text_config.engram_pad_token_id", -1);
  LOAD_ARG_OR(engram_compressed_vocab_size,
              "text_config.engram_compressed_vocab_size",
              0);

  // DSpark draft geometry. The three draft layers (40-42) are part of the
  // model itself; dspark_target_layer_ids only feeds dspark_num_layers here.
  LOAD_ARG_OR(
      num_nextn_predict_layers, "text_config.num_nextn_predict_layers", 3);
  LOAD_ARG_OR(markov_rank, "text_config.dspark_markov_rank", 0);
  args->dspark_num_layers() = static_cast<int32_t>(
      json.value_or<std::vector<int32_t>>("text_config.dspark_target_layer_ids",
                                          std::vector<int32_t>{})
          .size());
  // Don't arm dspark_block_size on the target (mirrors deepseek_v4: it would
  // enable non-causal DSpark attention; the draft worker sets it from the
  // checkpoint).
  LOAD_ARG_OR(
      dspark_n_routed_experts, "text_config.dspark_n_routed_experts", 0);
  LOAD_ARG_OR(
      dspark_num_experts_per_tok, "text_config.dspark_num_experts_per_tok", 0);

  // Runtime sizing hints
  LOAD_ARG_OR_FUNC(
      max_batch_size, "max_batch_size", [&] { return args->max_batch_size(); });
  LOAD_ARG_OR_FUNC(
      max_seq_len, "max_seq_len", [&] { return args->max_seq_len(); });

  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 129280);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 1048576);

  // Token ids (root level in the released config; text_config kept as
  // fallback for exports that nest them).
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", args->bos_token_id());
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", args->eos_token_id());
  LOAD_ARG_OR(pad_token_id, "pad_token_id", 2);
  LOAD_ARG_OR(pad_token_id, "text_config.pad_token_id", args->pad_token_id());
  // -1 = no image tokens (text-only run); the released config sets 129264.
  LOAD_ARG_OR(image_token_id, "image_token_id", -1);

  process_deepseek_v41_args(args);
  validate_deepseek_v41_args(*args, build_deepseek_v41_args_policy());
  return true;
}

// Register the deepseek_v41 model backend ("llm"); no C++ CausalLM class is
// registered because the model runs exclusively through the python executor
// (--model_impl=python -> PyCausalLM -> xllm.python.models.deepseek_v41).
REGISTER_MODEL_BACKEND(deepseek_v41, "llm");

// register the model args
REGISTER_MODEL_ARGS_LOADER(deepseek_v41, &load_deepseek_v41_model_args);

}  // namespace xllm
