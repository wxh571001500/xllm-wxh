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

// Model-level regression test for GLM-5.3-Flash prefill context parallelism.
//
// Eight ranks (cp_size=8, kv_split_size=1, TP=1, EP=8) run the first eight
// decoder layers through consecutive chunked-prefill segments, decode steps
// and a second turn that reuses the caches, and compare the outputs, the
// captured intermediate hidden states and the caches with a frozen baseline.
//
// Kept at production values: hidden size, mHC, the KDA/DSA and dense/MoE layer
// pattern of layers 0-7, every KDA, MLA and KPool indexer dimension, weight
// dtypes, the compressed-tensors W8A8/W4A8 formats with their scales, and the
// MTP cache layout (num_speculative_tokens=3).
//
// Reduced to bound the cost, and therefore not covered:
//   - intermediate_size 12288 -> 512 and moe_intermediate_size 2048 -> 256;
//   - n_routed_experts 288 -> 16, so each EP rank owns 2 experts instead of 36;
//   - vocab_size 154880 -> 512 and 45 layers -> 8;
//   - sequences shorter than index_topk, so sparse selection keeps every block;
//   - no speculative verification, MTP draft, graph replay, lm_head or
//     scheduler; ranks share one host.
// The packed INT4 expert bytes are arbitrary fixed bytes rather than a
// quantization of meaningful weights.
//
// Environment:
//   XLLM_GLM5_NEXT_PCP_BASELINE_DIR   directory holding the frozen baseline.
//   XLLM_GLM5_NEXT_PCP_BASELINE_MODE  "verify" (default) or "record". Record
//                                     creates a baseline and never overwrites.
//   XLLM_GLM5_NEXT_PCP_STRICT_BITS    "1" requires bit-identical tensors.

#include <fcntl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "framework/config/eplb_config.h"
#include "framework/config/kv_cache_config.h"
#include "framework/config/scheduler_config.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model/model_output.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/process_group.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "models/llm/mlu/glm5_next.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"
#include "util/tensor_helper.h"

namespace xllm::mlu::model {
namespace {

constexpr int32_t kWorldSize = 8;
constexpr int32_t kNumLayers = 8;
constexpr int32_t kNumSpeculativeTokens = 3;
constexpr int64_t kBlockSize = 16;
constexpr int64_t kMaxTokensPerBatch = 16384;
constexpr uint64_t kWeightSeed = 0x474c4d3533464c41ULL;

constexpr int64_t kHiddenSize = 4096;
constexpr int64_t kHcMult = 4;
constexpr int64_t kLinearHeads = 64;
constexpr int64_t kLinearHeadDim = 128;
constexpr int64_t kConvKernel = 4;
constexpr int64_t kAttentionHeads = 64;
constexpr int64_t kQLoraRank = 1536;
constexpr int64_t kKvLoraRank = 512;
constexpr int64_t kQkNopeHeadDim = 256;
constexpr int64_t kVHeadDim = 256;
constexpr int64_t kIndexHeads = 32;
constexpr int64_t kIndexHeadDim = 128;
constexpr int64_t kIndexTopk = 2048;
constexpr int64_t kIndexKpool = 4;
constexpr int64_t kW4GroupSize = 128;

// Reduced dimensions; see the file comment.
constexpr int64_t kVocabSize = 512;
constexpr int64_t kIntermediateSize = 512;
constexpr int64_t kMoeIntermediateSize = 256;
constexpr int64_t kRoutedExperts = 16;

// Linear-state slot 0 and KV block 0 are reserved for padding.
constexpr int64_t kLinearStateBlocks = 3;
constexpr int64_t kKvBlocks = 192;
constexpr float kConvCheckpointSentinel = 0.375f;
constexpr float kSsmCheckpointSentinel = -1.25f;

constexpr double kTolerance = 1e-3;

constexpr int32_t kExitSkip = 77;
constexpr int32_t kExitMismatch = 1;
constexpr int32_t kExitMissingBaseline = 3;
constexpr uint32_t kChildTimeoutSeconds = 1500;

constexpr char kBaselineDirEnv[] = "XLLM_GLM5_NEXT_PCP_BASELINE_DIR";
constexpr char kBaselineModeEnv[] = "XLLM_GLM5_NEXT_PCP_BASELINE_MODE";
constexpr char kStrictBitsEnv[] = "XLLM_GLM5_NEXT_PCP_STRICT_BITS";
constexpr char kManifestName[] = "MANIFEST";
constexpr char kTensorMagic[8] = {'X', 'L', 'T', 'E', 'N', 'S', '0', '1'};

using Weights = std::unordered_map<std::string, torch::Tensor>;

enum class BaselineMode : int8_t { VERIFY = 0, RECORD = 1 };

struct BaselineOptions {
  std::string directory;
  BaselineMode mode = BaselineMode::VERIFY;
  bool strict_bits = false;
};

uint64_t fnv1a64(const std::string& text) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  for (const char character : text) {
    hash ^= static_cast<uint8_t>(character);
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

uint64_t splitmix64(uint64_t& state) {
  state += 0x9e3779b97f4a7c15ULL;
  uint64_t value = state;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

// Every tensor is a pure function of its key, so each rank derives identical
// full weights on the CPU regardless of creation order or thread count.
torch::Tensor uniform_tensor(const std::string& key,
                             std::vector<int64_t> shape,
                             float low,
                             float high,
                             torch::ScalarType dtype) {
  torch::Tensor values = torch::empty(shape, torch::kFloat32);
  float* data = values.data_ptr<float>();
  const int64_t count = values.numel();
  uint64_t state = fnv1a64(key) ^ kWeightSeed;
  const float span = high - low;
  for (int64_t index = 0; index < count; ++index) {
    const float unit =
        static_cast<float>(splitmix64(state) >> 40) * (1.0f / 16777216.0f);
    data[index] = low + span * unit;
  }
  return values.to(dtype);
}

// symmetric=true yields INT8 weights in [-127, 127]; false yields raw bytes
// for the packed INT4 expert weights.
torch::Tensor int8_tensor(const std::string& key,
                          std::vector<int64_t> shape,
                          bool symmetric) {
  torch::Tensor values = torch::empty(shape, torch::kInt8);
  int8_t* data = values.data_ptr<int8_t>();
  const int64_t count = values.numel();
  uint64_t state = fnv1a64(key) ^ kWeightSeed;
  for (int64_t index = 0; index < count; ++index) {
    const uint64_t random = splitmix64(state) >> 32;
    data[index] =
        symmetric
            ? static_cast<int8_t>(static_cast<int32_t>(random % 255) - 127)
            : static_cast<int8_t>(random & 0xff);
  }
  return values;
}

class WeightBuilder final {
 public:
  explicit WeightBuilder(const torch::Device& device) : device_(device) {}

  void add_uniform(const std::string& key,
                   std::vector<int64_t> shape,
                   float low,
                   float high,
                   torch::ScalarType dtype = torch::kBFloat16) {
    add(key, uniform_tensor(key, std::move(shape), low, high, dtype));
  }

  // Unquantized BF16 projection with weights of magnitude ~0.01.
  void add_float_linear(const std::string& prefix,
                        int64_t out_features,
                        int64_t in_features) {
    add_uniform(prefix + ".weight",
                {out_features, in_features},
                /*low=*/-0.02f,
                /*high=*/0.02f);
  }

  // compressed-tensors int-quantized W8A8: INT8 weight, per-channel BF16
  // scale and an input smooth vector.
  void add_w8a8_linear(const std::string& prefix,
                       int64_t out_features,
                       int64_t in_features) {
    add_w8a8_linear(prefix, out_features, in_features, prefix + ".smooth");
  }

  // Projections that read the same input share one smooth vector, selected
  // by smooth_key.
  void add_w8a8_linear(const std::string& prefix,
                       int64_t out_features,
                       int64_t in_features,
                       const std::string& smooth_key) {
    add(prefix + ".weight",
        int8_tensor(prefix + ".weight",
                    {out_features, in_features},
                    /*symmetric=*/true));
    add_uniform(prefix + ".weight_scale",
                {out_features, 1},
                /*low=*/1.0e-4f,
                /*high=*/2.0e-4f);
    add(prefix + ".smooth",
        uniform_tensor(smooth_key,
                       {in_features},
                       /*low=*/0.5f,
                       /*high=*/1.5f,
                       torch::kBFloat16));
  }

  // SwiGLU MLP in W8A8: gate_proj and up_proj share their input smooth.
  void add_w8a8_swiglu(const std::string& prefix, int64_t intermediate_size) {
    const std::string input_smooth = prefix + "input.smooth";
    add_w8a8_linear(
        prefix + "gate_proj", intermediate_size, kHiddenSize, input_smooth);
    add_w8a8_linear(
        prefix + "up_proj", intermediate_size, kHiddenSize, input_smooth);
    add_w8a8_linear(prefix + "down_proj", kHiddenSize, intermediate_size);
  }

  // compressed-tensors int4-le-pack-quantized W4A8: two INT4 values per
  // byte, one BF16 scale per group of 128 input channels.
  void add_w4a8_linear(const std::string& prefix,
                       int64_t out_features,
                       int64_t in_features,
                       const torch::Tensor& smooth) {
    add(prefix + ".weight",
        int8_tensor(prefix + ".weight",
                    {out_features, in_features / 2},
                    /*symmetric=*/false));
    add_uniform(prefix + ".weight_scale",
                {out_features, in_features / kW4GroupSize},
                /*low=*/1.5e-3f,
                /*high=*/3.0e-3f);
    add(prefix + ".smooth", smooth);
  }

  void add(const std::string& key, const torch::Tensor& cpu_tensor) {
    const bool inserted = weights_.emplace(key, cpu_tensor.to(device_)).second;
    CHECK(inserted) << "Duplicate weight key: " << key;
  }

  Weights release() { return std::move(weights_); }

 private:
  torch::Device device_;
  Weights weights_;
};

bool is_kda_layer(int32_t layer_id) { return layer_id % 4 != 3; }

bool is_dense_layer(int32_t layer_id) { return layer_id < 3; }

void add_kda_weights(WeightBuilder& builder, const std::string& prefix) {
  const int64_t projection = kLinearHeads * kLinearHeadDim;
  for (const std::string name : {"q_proj", "k_proj", "v_proj"}) {
    builder.add_float_linear(prefix + name, projection, kHiddenSize);
  }
  builder.add_float_linear(prefix + "b_proj", kLinearHeads, kHiddenSize);
  for (const std::string name : {"f_a_proj", "g_a_proj"}) {
    builder.add_float_linear(prefix + name, kLinearHeadDim, kHiddenSize);
  }
  for (const std::string name : {"f_b_proj", "g_b_proj"}) {
    builder.add_float_linear(prefix + name, projection, kLinearHeadDim);
  }
  for (const std::string name : {"q_conv1d", "k_conv1d", "v_conv1d"}) {
    builder.add_uniform(prefix + name + ".weight",
                        {projection, 1, kConvKernel},
                        /*low=*/-0.2f,
                        /*high=*/0.2f,
                        torch::kFloat32);
  }
  builder.add_uniform(prefix + "A_log",
                      {kLinearHeads},
                      /*low=*/-0.2f,
                      /*high=*/0.2f,
                      torch::kFloat32);
  builder.add_uniform(prefix + "dt_bias",
                      {projection},
                      /*low=*/-0.2f,
                      /*high=*/0.2f,
                      torch::kFloat32);
  builder.add_uniform(
      prefix + "o_norm.weight", {kLinearHeadDim}, /*low=*/0.9f, /*high=*/1.1f);
  builder.add_float_linear(prefix + "o_proj", kHiddenSize, projection);
}

void add_dsa_weights(WeightBuilder& builder, const std::string& prefix) {
  builder.add_w8a8_linear(prefix + "q_a_proj", kQLoraRank, kHiddenSize);
  builder.add_uniform(prefix + "q_a_layernorm.weight",
                      {kQLoraRank},
                      /*low=*/0.9f,
                      /*high=*/1.1f);
  builder.add_w8a8_linear(
      prefix + "q_b_proj", kAttentionHeads * kQkNopeHeadDim, kQLoraRank);
  builder.add_w8a8_linear(
      prefix + "kv_a_proj_with_mqa", kKvLoraRank, kHiddenSize);
  builder.add_uniform(prefix + "kv_a_layernorm.weight",
                      {kKvLoraRank},
                      /*low=*/0.9f,
                      /*high=*/1.1f);
  builder.add_float_linear(prefix + "kv_b_proj",
                           kAttentionHeads * (kQkNopeHeadDim + kVHeadDim),
                           kKvLoraRank);
  builder.add_w8a8_linear(
      prefix + "o_proj", kHiddenSize, kAttentionHeads * kVHeadDim);

  const std::string indexer = prefix + "indexer.";
  builder.add_float_linear(
      indexer + "wq_b", kIndexHeads * kIndexHeadDim, kQLoraRank);
  builder.add_float_linear(indexer + "wk", kIndexHeadDim, kHiddenSize);
  builder.add_float_linear(indexer + "weights_proj", kIndexHeads, kHiddenSize);
  builder.add_uniform(indexer + "k_norm.weight",
                      {kIndexHeadDim},
                      /*low=*/0.9f,
                      /*high=*/1.1f);
  builder.add_uniform(indexer + "k_norm.bias",
                      {kIndexHeadDim},
                      /*low=*/-0.05f,
                      /*high=*/0.05f);
  builder.add_uniform(indexer + "index_kpool_compress_gate",
                      {kIndexHeadDim, kHiddenSize},
                      /*low=*/-0.02f,
                      /*high=*/0.02f);
  builder.add_uniform(indexer + "index_kpool_compress_ape",
                      {kIndexKpool, kIndexHeadDim},
                      /*low=*/-0.05f,
                      /*high=*/0.05f);
}

void add_sparse_moe_weights(WeightBuilder& builder, const std::string& prefix) {
  builder.add_float_linear(prefix + "gate", kRoutedExperts, kHiddenSize);
  builder.add_uniform(prefix + "gate.e_score_correction_bias",
                      {kRoutedExperts},
                      /*low=*/-0.1f,
                      /*high=*/0.1f,
                      torch::kFloat32);
  for (int64_t expert = 0; expert < kRoutedExperts; ++expert) {
    const std::string expert_prefix =
        prefix + "experts." + std::to_string(expert) + ".";
    // gate_proj and up_proj share one input smooth vector per expert.
    const torch::Tensor input_smooth = uniform_tensor(expert_prefix + "smooth",
                                                      {kHiddenSize},
                                                      /*low=*/0.5f,
                                                      /*high=*/1.5f,
                                                      torch::kBFloat16);
    builder.add_w4a8_linear(expert_prefix + "gate_proj",
                            kMoeIntermediateSize,
                            kHiddenSize,
                            input_smooth);
    builder.add_w4a8_linear(expert_prefix + "up_proj",
                            kMoeIntermediateSize,
                            kHiddenSize,
                            input_smooth);
    builder.add_w4a8_linear(expert_prefix + "down_proj",
                            kHiddenSize,
                            kMoeIntermediateSize,
                            uniform_tensor(expert_prefix + "down_proj.smooth",
                                           {kMoeIntermediateSize},
                                           /*low=*/0.5f,
                                           /*high=*/1.5f,
                                           torch::kBFloat16));
  }
  builder.add_w8a8_swiglu(prefix + "shared_experts.", kMoeIntermediateSize);
}

// Full checkpoint-format weights. Each rank builds all of them; the
// production loader keeps the EP shard [rank * E / 8, (rank + 1) * E / 8) of
// the routed experts and replicates everything else.
Weights make_weights(const torch::Device& device) {
  WeightBuilder builder(device);
  builder.add_uniform("embed_tokens.weight",
                      {kVocabSize, kHiddenSize},
                      /*low=*/-1.5f,
                      /*high=*/1.5f);
  builder.add_uniform(
      "norm.weight", {kHiddenSize}, /*low=*/0.9f, /*high=*/1.1f);
  const int64_t hc_rows = (2 + kHcMult) * kHcMult;
  for (int32_t layer_id = 0; layer_id < kNumLayers; ++layer_id) {
    const std::string layer = "layers." + std::to_string(layer_id) + ".";
    for (const std::string name :
         {"input_layernorm", "post_attention_layernorm"}) {
      builder.add_uniform(layer + name + ".weight",
                          {kHiddenSize},
                          /*low=*/0.9f,
                          /*high=*/1.1f);
    }
    for (const std::string name : {"hc_attn_", "hc_ffn_"}) {
      builder.add_uniform(layer + name + "fn",
                          {hc_rows, kHcMult * kHiddenSize},
                          /*low=*/-0.02f,
                          /*high=*/0.02f);
      builder.add_uniform(
          layer + name + "base", {hc_rows}, /*low=*/-0.1f, /*high=*/0.1f);
      builder.add_uniform(
          layer + name + "scale", {3}, /*low=*/0.05f, /*high=*/0.15f);
    }
    if (is_kda_layer(layer_id)) {
      add_kda_weights(builder, layer + "self_attn.");
    } else {
      add_dsa_weights(builder, layer + "self_attn.");
    }
    if (is_dense_layer(layer_id)) {
      builder.add_w8a8_swiglu(layer + "mlp.", kIntermediateSize);
    } else {
      add_sparse_moe_weights(builder, layer + "mlp.");
    }
  }
  return builder.release();
}

ModelArgs make_model_args(const std::vector<int32_t>& layers_to_capture) {
  ModelArgs args;
  args.model_type() = "glm5_next";
  args.dtype() = "bfloat16";
  args.vocab_size() = kVocabSize;
  args.hidden_size() = kHiddenSize;
  args.n_layers() = kNumLayers;
  args.n_heads() = kAttentionHeads;
  args.n_kv_heads() = kAttentionHeads;
  args.intermediate_size() = kIntermediateSize;
  args.max_position_embeddings() = 1048576;
  args.rms_norm_eps() = 1e-5f;
  args.tie_word_embeddings() = false;
  args.first_k_dense_replace() = 3;
  args.hidden_act() = "silu";
  args.n_routed_experts() = kRoutedExperts;
  args.n_shared_experts() = 1;
  args.num_experts_per_tok() = 8;
  args.moe_intermediate_size() = kMoeIntermediateSize;
  args.routed_scaling_factor() = 2.5f;
  args.norm_topk_prob() = true;
  args.n_group() = 1;
  args.topk_group() = 1;
  args.scoring_func() = "sigmoid";
  args.topk_method() = "noaux_tc";
  args.swiglu_limit() = 10.0f;
  args.hc_mult() = kHcMult;
  args.hc_sinkhorn_iters() = 20;
  args.hc_eps() = 1e-6f;
  args.qk_nope_head_dim() = kQkNopeHeadDim;
  args.qk_rope_head_dim() = 0;
  args.v_head_dim() = kVHeadDim;
  args.head_dim() = kQkNopeHeadDim;
  args.q_lora_rank() = kQLoraRank;
  args.kv_lora_rank() = kKvLoraRank;
  args.enable_mla() = true;
  args.rope_scaling_rope_type() = "default";
  args.index_head_dim() = kIndexHeadDim;
  args.index_n_heads() = kIndexHeads;
  args.index_topk() = kIndexTopk;
  args.index_kpool() = kIndexKpool;
  args.index_kpool_compress() = true;
  args.index_kpool_always_select_tail() = true;
  args.index_share_for_mtp_iteration() = true;
  args.linear_num_key_heads() = kLinearHeads;
  args.linear_num_value_heads() = kLinearHeads;
  args.linear_key_head_dim() = kLinearHeadDim;
  args.linear_value_head_dim() = kLinearHeadDim;
  args.linear_conv_kernel_dim() = kConvKernel;
  args.linear_lower_bound() = -5.0f;
  args.mamba_ssm_dtype() = "float32";
  std::vector<std::string> layer_types;
  std::vector<std::string> mlp_layer_types;
  layer_types.reserve(kNumLayers);
  mlp_layer_types.reserve(kNumLayers);
  for (int32_t layer_id = 0; layer_id < kNumLayers; ++layer_id) {
    layer_types.emplace_back(is_kda_layer(layer_id)
                                 ? "linear_attention"
                                 : "deepseek_sparse_attention");
    mlp_layer_types.emplace_back(is_dense_layer(layer_id) ? "dense" : "sparse");
  }
  args.layer_types() = std::move(layer_types);
  args.mlp_layer_types() = std::move(mlp_layer_types);
  args.layers_to_capture() = layers_to_capture;
  return args;
}

// Mirrors load_ct_quant_config for the production quantization_config; the
// target and ignore patterns are the checkpoint's own.
QuantArgs make_quant_args() {
  const std::string layers =
      "(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|"
      "26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45)";
  QuantArgs args;
  args.quant_method() = "compressed-tensors";
  args.bits() = 8;
  args.is_sym() = true;
  args.activation_dynamic() = true;
  args.is_compressed_tensors_w8a8_dynamic() = true;
  args.ignored_modules() = {"re:.*lm_head",
                            "re:.*embed_tokens",
                            R"(re:model\.visual)",
                            R"(re:.*\.norm$)",
                            R"(re:.*mlp\.gate$)",
                            "re:.*router$",
                            "re:.*attn_mha$",
                            "re:.*attn_mqa$",
                            "re:.*hyper_connection",
                            "re:.*mapping_proj$",
                            R"(re:.*shared_head\.norm$)",
                            "re:.*eh_proj$",
                            "re:.*downsample$"};
  args.compressed_groups() = {
      CompressedQuantGroup{
          .targets =
              {R"(re:.*mlp\.(gate_proj|up_proj|down_proj)$)",
               R"(re:.*layers\.)" + layers +
                   R"(\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)$)",
               R"(re:.*layers\.(3|7|11|15|19|23|27|31|35|39|43|45)\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)$)"},
          .bits = 8,
          .group_size = 0,
          .preserve_smooth = true},
      CompressedQuantGroup{
          .targets =
              {R"(re:.*layers\.)" + layers +
               R"(\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$)"},
          .bits = 4,
          .group_size = kW4GroupSize,
          .preserve_smooth = true}};
  return args;
}

struct LayerCacheTensors {
  torch::Tensor conv;
  torch::Tensor ssm;
  torch::Tensor key;
  torch::Tensor index;
  torch::Tensor kpool_tail;
};

// Cache tensors laid out as KVCacheShape lays them out for a GLM5-Next MTP
// target with num_speculative_tokens=3: the convolution state keeps
// kernel-1 committed columns followed by one column per speculative token,
// and every linear-state slot owns num_speculative_tokens+1 SSM rows whose
// first row is the committed state.
std::vector<LayerCacheTensors> make_cache_tensors(const torch::Device& device) {
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const int64_t conv_state_len = kConvKernel - 1 + kNumSpeculativeTokens;
  const int64_t checkpoint_stride = kNumSpeculativeTokens + 1;
  const int64_t kpool_tail_len = kIndexKpool + 2 * (kNumSpeculativeTokens + 1);
  std::vector<LayerCacheTensors> tensors;
  tensors.reserve(kNumLayers);
  for (int32_t layer_id = 0; layer_id < kNumLayers; ++layer_id) {
    LayerCacheTensors layer;
    if (is_kda_layer(layer_id)) {
      layer.conv = torch::zeros({kLinearStateBlocks,
                                 conv_state_len,
                                 3 * kLinearHeads * kLinearHeadDim},
                                options);
      layer.ssm = torch::zeros(
          {kLinearStateBlocks * checkpoint_stride,
           kLinearHeads,
           kLinearHeadDim,
           kLinearHeadDim},
          options.dtype(resolve_ssm_dtype(
              /*mamba_ssm_dtype_str=*/"float32", torch::kBFloat16)));
      layer.conv.narrow(/*dim=*/1, kConvKernel - 1, kNumSpeculativeTokens)
          .fill_(kConvCheckpointSentinel);
      layer.ssm.view({kLinearStateBlocks, checkpoint_stride, -1})
          .narrow(/*dim=*/1, /*start=*/1, kNumSpeculativeTokens)
          .fill_(kSsmCheckpointSentinel);
    } else {
      layer.key =
          torch::zeros({kKvBlocks, 1, kBlockSize, kKvLoraRank}, options);
      layer.index = torch::zeros(
          {kKvBlocks, 1, kBlockSize / kIndexKpool, kIndexHeadDim}, options);
      layer.kpool_tail = torch::zeros(
          {kLinearStateBlocks, 2, kpool_tail_len, kIndexHeadDim}, options);
    }
    tensors.emplace_back(std::move(layer));
  }
  return tensors;
}

std::vector<KVCache> make_caches(
    const std::vector<LayerCacheTensors>& tensors) {
  std::vector<KVCache> caches;
  caches.reserve(tensors.size());
  for (const LayerCacheTensors& layer : tensors) {
    if (layer.conv.defined()) {
      caches.emplace_back(
          KVCache(LinearAttentionKVCacheTensors{layer.conv, layer.ssm}));
      continue;
    }
    IndexedKVCacheTensors indexed;
    indexed.kv_cache_tensors.key_cache = layer.key;
    indexed.index_cache = layer.index;
    indexed.kpool_tail = layer.kpool_tail;
    caches.emplace_back(KVCache(indexed));
  }
  return caches;
}

// Speculative checkpoints are owned by speculative verification; ordinary
// prefill and decode must leave them untouched.
bool checkpoints_untouched(const std::vector<LayerCacheTensors>& tensors) {
  bool untouched = true;
  for (size_t layer_id = 0; layer_id < tensors.size(); ++layer_id) {
    const LayerCacheTensors& layer = tensors[layer_id];
    if (!layer.conv.defined()) {
      continue;
    }
    const bool conv_untouched =
        layer.conv.narrow(/*dim=*/1, kConvKernel - 1, kNumSpeculativeTokens)
            .eq(kConvCheckpointSentinel)
            .all()
            .item<bool>();
    const bool ssm_untouched =
        layer.ssm.view({kLinearStateBlocks, kNumSpeculativeTokens + 1, -1})
            .narrow(/*dim=*/1, /*start=*/1, kNumSpeculativeTokens)
            .eq(kSsmCheckpointSentinel)
            .all()
            .item<bool>();
    LOG_IF(ERROR, !conv_untouched)
        << "Layer " << layer_id << " rewrote speculative conv checkpoints";
    LOG_IF(ERROR, !ssm_untouched)
        << "Layer " << layer_id << " rewrote speculative SSM checkpoints";
    untouched = untouched && conv_untouched && ssm_untouched;
  }
  return untouched;
}

struct SequenceState {
  int32_t linear_state_id = 0;
  int32_t first_block = 0;
  int32_t num_blocks = 0;
  int32_t length = 0;
};

struct Step {
  std::string name;
  BatchForwardType::Value type = BatchForwardType::PREFILL;
  std::vector<int32_t> new_tokens;
  bool snapshot_caches = false;
};

// Every prefill segment gives sequence 0 at least 449 tokens, i.e. eight
// 64-token KDA chunks, so all eight ranks own tokens and PCP engages.
std::vector<Step> make_steps() {
  return {
      {"prefill", BatchForwardType::PREFILL, {512, 200}, false},
      {"chunk1", BatchForwardType::CHUNKED_PREFILL, {576, 96}, false},
      {"chunk2", BatchForwardType::CHUNKED_PREFILL, {449, 1}, true},
      {"decode0", BatchForwardType::DECODE, {1, 1}, false},
      {"decode1", BatchForwardType::DECODE, {1, 1}, false},
      {"decode2", BatchForwardType::DECODE, {1, 1}, true},
      {"turn2_chunk", BatchForwardType::CHUNKED_PREFILL, {512, 130}, false},
      {"turn2_decode", BatchForwardType::DECODE, {1, 1}, true},
  };
}

int32_t token_id_at(size_t sequence, int32_t position) {
  uint64_t state = kWeightSeed ^ (static_cast<uint64_t>(sequence) << 32) ^
                   static_cast<uint64_t>(position);
  return 1 + static_cast<int32_t>(splitmix64(state) % (kVocabSize - 1));
}

struct StepInput {
  torch::Tensor tokens;
  torch::Tensor positions;
  ModelInputParams params;
};

// Fills the fields BatchInputBuilder::state_to_forward_input and
// prepare_input_params_for_linear_attention fill on MLU, where sequence
// lengths are cumulative with a leading zero.
StepInput make_step_input(const Step& step,
                          const std::vector<SequenceState>& sequences,
                          const torch::Device& device) {
  const auto ints = torch::TensorOptions().dtype(torch::kInt32).device(device);
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  std::vector<int32_t> slots;
  std::vector<int32_t> q_cumulative = {0};
  std::vector<int32_t> kv_cumulative = {0};
  std::vector<int32_t> cached_tokens;
  std::vector<int32_t> linear_state_ids;
  std::vector<int32_t> block_table;
  cached_tokens.reserve(sequences.size());
  linear_state_ids.reserve(sequences.size());
  int32_t table_width = 0;
  for (const SequenceState& state : sequences) {
    table_width = std::max(table_width, state.num_blocks);
  }
  block_table.reserve(sequences.size() * static_cast<size_t>(table_width));
  StepInput input;
  int32_t max_query = 0;
  int32_t max_kv = 0;
  for (size_t sequence = 0; sequence < sequences.size(); ++sequence) {
    const SequenceState& state = sequences[sequence];
    const int32_t count = step.new_tokens[sequence];
    for (int32_t offset = 0; offset < count; ++offset) {
      const int32_t position = state.length + offset;
      tokens.emplace_back(token_id_at(sequence, position));
      positions.emplace_back(position);
      slots.emplace_back(
          (state.first_block + position / static_cast<int32_t>(kBlockSize)) *
              static_cast<int32_t>(kBlockSize) +
          position % static_cast<int32_t>(kBlockSize));
    }
    // Unused entries point at the reserved padding block 0.
    for (int32_t block = 0; block < table_width; ++block) {
      block_table.emplace_back(
          block < state.num_blocks ? state.first_block + block : 0);
    }
    q_cumulative.emplace_back(q_cumulative.back() + count);
    kv_cumulative.emplace_back(kv_cumulative.back() + state.length + count);
    cached_tokens.emplace_back(state.length);
    linear_state_ids.emplace_back(state.linear_state_id);
    input.params.linear_state_validity_mask.emplace_back(state.length > 0 ? 1
                                                                          : 0);
    max_query = std::max(max_query, count);
    max_kv = std::max(max_kv, state.length + count);
  }

  input.tokens = torch::tensor(tokens, ints);
  input.positions = torch::tensor(positions, ints);
  ModelInputParams& params = input.params;
  params.meta.batch_forward_type = BatchForwardType(step.type);
  params.meta.num_sequences = static_cast<int32_t>(sequences.size());
  params.meta.actual_num_sequences = params.meta.num_sequences;
  params.meta.q_max_seq_len = max_query;
  params.meta.kv_max_seq_len = max_kv;
  params.attention.device.q_seq_lens = torch::tensor(q_cumulative, ints);
  params.attention.device.q_cu_seq_lens = torch::tensor(q_cumulative, ints);
  params.attention.device.kv_seq_lens = torch::tensor(kv_cumulative, ints);
  params.attention.device.kv_cache_tokens_nums =
      torch::tensor(cached_tokens, ints);
  params.attention.device.new_cache_slots = torch::tensor(slots, ints);
  params.attention.device.block_tables =
      torch::tensor(block_table, ints)
          .view({static_cast<int64_t>(sequences.size()), table_width});
  params.attention.host.q_seq_lens = q_cumulative;
  params.attention.host.q_cu_seq_lens = std::move(q_cumulative);
  params.attention.host.kv_seq_lens = std::move(kv_cumulative);
  params.attention.host.kv_cache_tokens_nums = std::move(cached_tokens);
  params.attention.host.block_tables =
      params.attention.device.block_tables.cpu();
  params.embedding.linear_state_indices = torch::tensor(linear_state_ids, ints);
  params.embedding.linear_state_ids = std::move(linear_state_ids);
  params.parallel.dp_global_token_nums = {static_cast<int32_t>(tokens.size())};
  return input;
}

std::string tensor_path(const std::string& directory, const std::string& name) {
  return directory + "/" + name + ".tensor";
}

// Stores one tensor in its original dtype: magic, dtype, rank, shape, bytes.
// O_EXCL keeps record mode from ever replacing an existing baseline file.
bool write_tensor_exclusive(const std::string& path,
                            const torch::Tensor& tensor) {
  const torch::Tensor cpu = tensor.cpu().contiguous();
  const int32_t descriptor =
      ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, /*mode=*/0644);
  if (descriptor < 0) {
    LOG(ERROR) << "Cannot create baseline file " << path << ": "
               << std::strerror(errno);
    return false;
  }
  std::string header(kTensorMagic, sizeof(kTensorMagic));
  const int32_t dtype = static_cast<int32_t>(cpu.scalar_type());
  const int32_t rank = static_cast<int32_t>(cpu.dim());
  header.append(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
  header.append(reinterpret_cast<const char*>(&rank), sizeof(rank));
  for (const int64_t size : cpu.sizes()) {
    header.append(reinterpret_cast<const char*>(&size), sizeof(size));
  }
  const auto write_all = [descriptor](const char* data, size_t bytes) {
    while (bytes > 0) {
      const ssize_t written = ::write(descriptor, data, bytes);
      if (written <= 0) {
        return false;
      }
      data += written;
      bytes -= static_cast<size_t>(written);
    }
    return true;
  };
  const bool written =
      write_all(header.data(), header.size()) &&
      write_all(static_cast<const char*>(cpu.data_ptr()), cpu.nbytes());
  const bool closed = ::close(descriptor) == 0;
  LOG_IF(ERROR, !(written && closed)) << "Failed writing " << path;
  return written && closed;
}

std::optional<torch::Tensor> read_tensor(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return std::nullopt;
  }
  char magic[sizeof(kTensorMagic)];
  int32_t dtype = 0;
  int32_t rank = 0;
  file.read(magic, sizeof(magic));
  file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
  file.read(reinterpret_cast<char*>(&rank), sizeof(rank));
  if (!file || std::memcmp(magic, kTensorMagic, sizeof(magic)) != 0 ||
      rank < 0 || rank > 8) {
    return std::nullopt;
  }
  std::vector<int64_t> shape(static_cast<size_t>(rank));
  file.read(reinterpret_cast<char*>(shape.data()),
            static_cast<std::streamsize>(shape.size() * sizeof(int64_t)));
  torch::Tensor tensor =
      torch::empty(shape, static_cast<torch::ScalarType>(dtype));
  file.read(static_cast<char*>(tensor.data_ptr()),
            static_cast<std::streamsize>(tensor.nbytes()));
  if (!file) {
    return std::nullopt;
  }
  return tensor;
}

// Records tensors into a new baseline, or compares them with a frozen one.
// A verification never derives its expectation from the current run.
class BaselineSession final {
 public:
  BaselineSession(BaselineOptions options, int32_t rank)
      : options_(std::move(options)), rank_(rank) {}

  // Returns false when a required baseline is absent.
  bool open() {
    if (options_.mode == BaselineMode::RECORD) {
      return true;
    }
    std::ifstream manifest(options_.directory + "/" + kManifestName);
    if (!manifest) {
      return false;
    }
    std::string name;
    while (std::getline(manifest, name)) {
      if (!name.empty()) {
        expected_.emplace(std::move(name));
      }
    }
    return !expected_.empty();
  }

  void check(const std::string& name, const torch::Tensor& tensor) {
    const torch::Tensor actual = tensor.cpu().contiguous();
    names_.emplace_back(name);
    if (torch::isFloatingType(actual.scalar_type()) &&
        !torch::isfinite(actual.to(torch::kFloat32)).all().item<bool>()) {
      fail(name, "contains non-finite values");
      return;
    }
    if (options_.mode == BaselineMode::RECORD) {
      // Rank 0 owns the files; the other ranks only run the model.
      if (rank_ == 0 && !write_tensor_exclusive(
                            tensor_path(options_.directory, name), actual)) {
        fail(name, "could not be recorded");
      }
      return;
    }
    if (!expected_.contains(name)) {
      fail(name, "is missing from the baseline manifest");
      return;
    }
    const std::optional<torch::Tensor> expected =
        read_tensor(tensor_path(options_.directory, name));
    if (!expected.has_value()) {
      fail(name, "baseline file is missing or unreadable");
      return;
    }
    if (expected->scalar_type() != actual.scalar_type() ||
        expected->sizes() != actual.sizes()) {
      fail(name, "dtype or shape differs from the baseline");
      return;
    }
    const bool bit_identical =
        std::memcmp(expected->data_ptr(), actual.data_ptr(), actual.nbytes()) ==
        0;
    const torch::Tensor actual_wide = actual.to(torch::kFloat64);
    const torch::Tensor expected_wide = expected->to(torch::kFloat64);
    const double max_error =
        actual.numel() == 0
            ? 0.0
            : (actual_wide - expected_wide).abs().max().item<double>();
    const bool close = torch::isFloatingType(actual.scalar_type())
                           ? torch::allclose(actual_wide,
                                             expected_wide,
                                             /*rtol=*/kTolerance,
                                             /*atol=*/kTolerance)
                           : bit_identical;
    ++compared_;
    bit_identical_ += bit_identical ? 1 : 0;
    max_error_ = std::max(max_error_, max_error);
    LOG(INFO) << "[baseline] rank=" << rank_ << " " << name
              << " bit_identical=" << bit_identical
              << " max_abs_error=" << max_error
              << " within_tolerance=" << close;
    if (!close) {
      fail(name, "exceeds atol=rtol=1e-3");
    } else if (options_.strict_bits && !bit_identical) {
      fail(name, "is not bit-identical in strict mode");
    }
  }

  void fail(const std::string& name, const std::string& reason) {
    LOG(ERROR) << "[baseline] rank=" << rank_ << " " << name << " " << reason;
    ++failures_;
  }

  // Returns true when the whole session succeeded.
  bool finish() {
    if (options_.mode == BaselineMode::RECORD) {
      if (rank_ == 0 && failures_ == 0) {
        std::string manifest;
        for (const std::string& name : names_) {
          manifest += name + "\n";
        }
        const std::string path = options_.directory + "/" + kManifestName;
        const int32_t descriptor =
            ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, /*mode=*/0644);
        const bool written =
            descriptor >= 0 &&
            ::write(descriptor, manifest.data(), manifest.size()) ==
                static_cast<ssize_t>(manifest.size()) &&
            ::close(descriptor) == 0;
        if (!written) {
          fail(kManifestName, "could not be recorded");
        }
        LOG(INFO) << "[baseline] recorded " << names_.size() << " tensors in "
                  << options_.directory;
      }
      return failures_ == 0;
    }
    const std::unordered_set<std::string> produced(names_.begin(),
                                                   names_.end());
    for (const std::string& name : expected_) {
      if (!produced.contains(name)) {
        fail(name, "is in the baseline but was not produced");
      }
    }
    LOG(INFO) << "[baseline] rank=" << rank_ << " compared=" << compared_
              << " bit_identical=" << bit_identical_
              << " max_abs_error=" << max_error_ << " failures=" << failures_
              << " strict_bits=" << options_.strict_bits;
    return failures_ == 0;
  }

 private:
  BaselineOptions options_;
  int32_t rank_ = 0;
  std::unordered_set<std::string> expected_;
  std::vector<std::string> names_;
  int64_t compared_ = 0;
  int64_t bit_identical_ = 0;
  int64_t failures_ = 0;
  double max_error_ = 0.0;
};

// Saves every cache in its own dtype. The SSM snapshot keeps the committed
// row of each real sequence; its speculative rows are covered by
// checkpoints_untouched.
void check_caches(const std::string& prefix,
                  const std::vector<LayerCacheTensors>& tensors,
                  BaselineSession& session) {
  for (size_t layer_id = 0; layer_id < tensors.size(); ++layer_id) {
    const LayerCacheTensors& layer = tensors[layer_id];
    const std::string name = prefix + ".layer" + std::to_string(layer_id);
    if (layer.conv.defined()) {
      session.check(name + ".conv", layer.conv);
      session.check(name + ".ssm",
                    layer.ssm
                        .view({kLinearStateBlocks,
                               kNumSpeculativeTokens + 1,
                               kLinearHeads,
                               kLinearHeadDim,
                               kLinearHeadDim})
                        .narrow(/*dim=*/0, /*start=*/1, kLinearStateBlocks - 1)
                        .select(/*dim=*/1, /*index=*/0));
      continue;
    }
    session.check(name + ".key", layer.key);
    session.check(name + ".index", layer.index);
    session.check(name + ".kpool_tail", layer.kpool_tail);
  }
}

// One pass over the scenario. The production pass captures nothing, as the
// MTP deployment does; the capture pass materializes the hidden states after
// the last dense KDA layer, the first DSA+MoE layer and a KDA+MoE layer.
bool run_scenario(const std::string& pass,
                  const std::vector<int32_t>& layers_to_capture,
                  const ParallelArgs& parallel_args,
                  const Weights& weights,
                  Device& device,
                  BaselineSession& session) {
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  const ModelArgs model_args = make_model_args(layers_to_capture);
  const QuantArgs quant_args = make_quant_args();
  const ModelContext context(parallel_args, model_args, quant_args, options);
  Glm5NextModel model(context);
  model->load_state_dict(StateDict(weights));
  model->verify_loaded_weights();

  const std::vector<Step> steps = make_steps();
  const std::vector<LayerCacheTensors> cache_tensors =
      make_cache_tensors(device.unwrap());
  std::vector<KVCache> caches = make_caches(cache_tensors);
  // Sequence 0 reaches 2053 tokens and sequence 1 reaches 431.
  std::vector<SequenceState> sequences = {
      {.linear_state_id = 1, .first_block = 1, .num_blocks = 144, .length = 0},
      {.linear_state_id = 2,
       .first_block = 145,
       .num_blocks = 40,
       .length = 0}};
  CHECK_LE(sequences.back().first_block + sequences.back().num_blocks,
           kKvBlocks);

  bool checkpoints_ok = true;
  for (const Step& step : steps) {
    if (step.type != BatchForwardType::DECODE) {
      // Guards the scenario itself: a rank without tokens would silently
      // route this segment through the non-PCP path.
      const layer::glm5_next_pcp::Geometry geometry =
          layer::glm5_next_pcp::build_geometry(
              step.new_tokens,
              kWorldSize,
              static_cast<int32_t>(kernel::mlu::kda_prefill_chunk_size(
                  kLinearHeads, /*use_qk_l2norm=*/true)));
      CHECK(std::all_of(geometry.tokens_per_rank.begin(),
                        geometry.tokens_per_rank.end(),
                        [](int32_t count) { return count > 0; }))
          << step.name << " does not give every CP rank tokens";
    }
    StepInput input = make_step_input(step, sequences, device.unwrap());
    const ModelOutput output =
        model->forward(input.tokens, input.positions, caches, input.params);
    device.synchronize_default_stream();
    CHECK(output.hidden_states.defined())
        << pass << "." << step.name << " produced no hidden states";
    for (size_t sequence = 0; sequence < sequences.size(); ++sequence) {
      sequences[sequence].length += step.new_tokens[sequence];
      CHECK_LE(
          sequences[sequence].length,
          sequences[sequence].num_blocks * static_cast<int32_t>(kBlockSize));
    }
    const std::string name = pass + "." + step.name;
    session.check(name + ".hidden", output.hidden_states);
    if (!layers_to_capture.empty()) {
      session.check(name + ".aux_hidden", output.aux_hidden_states);
    }
    if (step.snapshot_caches && layers_to_capture.empty()) {
      check_caches(name, cache_tensors, session);
    }
    const bool untouched = checkpoints_untouched(cache_tensors);
    LOG_IF(ERROR, !untouched)
        << name << " rewrote uncommitted checkpoints on rank "
        << parallel_args.rank();
    checkpoints_ok = checkpoints_ok && untouched;
  }
  return checkpoints_ok;
}

std::unique_ptr<ProcessGroup> create_rank_group(int32_t rank,
                                                int32_t port,
                                                bool trans,
                                                const std::string& name,
                                                const torch::Device& device) {
  std::unique_ptr<ProcessGroup> group = create_process_group(
      rank, kWorldSize, kWorldSize, port, trans, "127.0.0.1", name, device);
  // TCPStore construction does not wait for every peer; one collective keeps
  // the server alive until all ranks have joined.
  torch::Tensor rendezvous = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  group->allreduce(rendezvous);
  CHECK_EQ(rendezvous.item<float>(), static_cast<float>(kWorldSize))
      << "Process group " << name << " did not reach every rank";
  return group;
}

int32_t run_rank(int32_t rank,
                 int32_t cp_port,
                 int32_t ep_port,
                 const BaselineOptions& baseline) {
  // Bounds a stuck collective, including a peer that died during setup.
  ::alarm(kChildTimeoutSeconds);
  if (Platform::device_count() < kWorldSize) {
    return kExitSkip;
  }
  BaselineSession session(baseline, rank);
  if (!session.open()) {
    return kExitMissingBaseline;
  }

  torch::InferenceMode inference_mode;
  Device device(rank);
  device.set_device();
  KVCacheConfig::get_instance().block_size(kBlockSize);
  SchedulerConfig::get_instance().max_tokens_per_batch(kMaxTokensPerBatch);
  EPLBConfig::get_instance().expert_parallel_degree(1);

  // Production wiring for dp=1, cp=8, tp=1, ep=8: separate eight-rank CP and
  // MoE EP groups, single-rank TP groups, and kv_split_size=1 so every rank
  // keeps the full KV cache.
  std::unique_ptr<ProcessGroup> cp_group =
      create_rank_group(rank,
                        cp_port,
                        /*trans=*/false,
                        "glm5_next_pcp_cp_group",
                        device.unwrap());
  std::unique_ptr<ProcessGroup> ep_group = create_rank_group(
      rank, ep_port, /*trans=*/true, "glm5_next_pcp_ep_group", device.unwrap());
  ProcessGroup single_rank(/*rank=*/0, /*world_size=*/1, device.unwrap());
  ParallelArgs parallel_args(rank,
                             kWorldSize,
                             /*dp_size=*/1,
                             /*cp_size=*/kWorldSize,
                             cp_group.get(),
                             /*ep_size=*/kWorldSize);
  parallel_args.kv_split_size() = 1;
  parallel_args.tp_size() = 1;
  parallel_args.cp_group_ = cp_group.get();
  parallel_args.moe_ep_group_ = ep_group.get();
  parallel_args.tp_group_ = &single_rank;
  parallel_args.single_rank_group_ = &single_rank;
  parallel_args.moe_tp_group_ = &single_rank;
  parallel_args.dcp_group_ = &single_rank;
  CHECK_EQ(parallel_args.cp_rank(), rank);
  CHECK_EQ(parallel_args.kv_split_size_effective(), 1);

  const Weights weights = make_weights(device.unwrap());
  const bool production_ok = run_scenario("production",
                                          /*layers_to_capture=*/{},
                                          parallel_args,
                                          weights,
                                          device,
                                          session);
  const bool capture_ok = run_scenario("capture",
                                       /*layers_to_capture=*/{2, 3, 6},
                                       parallel_args,
                                       weights,
                                       device,
                                       session);
  const bool baseline_ok = session.finish();
  return production_ok && capture_ok && baseline_ok ? 0 : kExitMismatch;
}

std::optional<BaselineOptions> read_baseline_options(std::string& error) {
  BaselineOptions options;
  const char* directory = std::getenv(kBaselineDirEnv);
  if (directory == nullptr || directory[0] == '\0') {
    error = std::string(kBaselineDirEnv) + " is not set";
    return std::nullopt;
  }
  options.directory = directory;
  const char* mode = std::getenv(kBaselineModeEnv);
  const std::string mode_name = mode == nullptr ? "verify" : mode;
  if (mode_name == "record") {
    options.mode = BaselineMode::RECORD;
  } else if (mode_name != "verify") {
    error = std::string(kBaselineModeEnv) + " must be verify or record";
    return std::nullopt;
  }
  const char* strict = std::getenv(kStrictBitsEnv);
  options.strict_bits = strict != nullptr && std::string(strict) == "1";
  return options;
}

TEST(Glm5NextMluPcpRegressionTest, EightRanksMatchFrozenBaseline) {
  // The frozen baseline lives outside the repository, so a default ctest run
  // has nothing to compare against.
  const char* baseline_directory = std::getenv(kBaselineDirEnv);
  if (baseline_directory == nullptr || baseline_directory[0] == '\0') {
    GTEST_SKIP() << "Set " << kBaselineDirEnv
                 << " to a frozen baseline, or create one explicitly with "
                 << kBaselineModeEnv << "=record.";
  }
  std::string error;
  const std::optional<BaselineOptions> baseline = read_baseline_options(error);
  ASSERT_TRUE(baseline.has_value())
      << error << ". Point " << kBaselineDirEnv
      << " at a frozen baseline, or create one explicitly with "
      << kBaselineModeEnv << "=record.";
  const std::string manifest = baseline->directory + "/" + kManifestName;
  if (baseline->mode == BaselineMode::RECORD) {
    std::error_code status;
    std::filesystem::create_directories(baseline->directory, status);
    ASSERT_FALSE(status) << "Cannot create " << baseline->directory;
    ASSERT_TRUE(std::filesystem::is_empty(baseline->directory, status))
        << "Record mode never overwrites: " << baseline->directory
        << " already holds files.";
  }

  const int32_t cp_port = net::get_local_free_port();
  const int32_t ep_port = net::get_local_free_port();
  ASSERT_GT(cp_port, 0);
  ASSERT_GT(ep_port, 0);
  ASSERT_NE(cp_port, ep_port);

  // Fork before any MLU initialization in this process.
  std::vector<pid_t> children;
  children.reserve(kWorldSize);
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    const pid_t child = ::fork();
    ASSERT_GE(child, 0) << "fork failed: " << std::strerror(errno);
    if (child == 0) {
      ::_exit(run_rank(rank, cp_port, ep_port, *baseline));
    }
    children.emplace_back(child);
  }

  int32_t skipped = 0;
  int32_t missing_baseline = 0;
  int32_t failed = 0;
  for (size_t rank = 0; rank < children.size(); ++rank) {
    int32_t status = 0;
    pid_t waited = -1;
    do {
      waited = ::waitpid(children[rank], &status, /*options=*/0);
    } while (waited < 0 && errno == EINTR);
    ASSERT_EQ(waited, children[rank]);
    const int32_t exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    skipped += exit_code == kExitSkip ? 1 : 0;
    missing_baseline += exit_code == kExitMissingBaseline ? 1 : 0;
    failed += exit_code != 0 && exit_code != kExitSkip &&
                      exit_code != kExitMissingBaseline
                  ? 1
                  : 0;
    if (!WIFEXITED(status)) {
      ADD_FAILURE() << "Rank " << rank << " terminated by signal "
                    << (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    } else if (exit_code != 0 && exit_code != kExitSkip &&
               exit_code != kExitMissingBaseline) {
      ADD_FAILURE() << "Rank " << rank << " exited with code " << exit_code
                    << "; see its [baseline] log lines.";
    }
  }
  if (skipped == kWorldSize) {
    GTEST_SKIP() << "Requires " << kWorldSize << " MLU devices.";
  }
  ASSERT_EQ(missing_baseline, 0)
      << "No baseline manifest at " << manifest << ". Record one with "
      << kBaselineModeEnv << "=record; verification never creates it.";
  ASSERT_EQ(skipped, 0);
  ASSERT_EQ(failed, 0);
}

}  // namespace
}  // namespace xllm::mlu::model
