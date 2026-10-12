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

#include "hf_model_loader.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <string_view>
#include <type_traits>

#include "core/framework/kv_cache/kv_cache_estimation.h"
#include "core/framework/kv_cache/kv_cache_shape.h"
#include "core/framework/model/aux_hidden_capture.h"
#include "core/framework/model/mtp_utils.h"
#include "core/framework/model/rec_causal_lm.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/runtime/options.h"
#include "core/util/model_config_utils.h"
#include "models/llm/glm5_next_mtp_args.h"
#if defined(USE_MLU)
#include "models/llm/mlu/glm5_next_mtp.h"
#include "models/llm/mtp_model_base.h"
#endif
#include "models/model_registry.h"

namespace xllm {

namespace {

using ExpectedRecModelFactory =
    std::function<std::unique_ptr<RecCausalLM>(const ModelContext& context)>;

static_assert(std::is_same_v<RecModelFactory, ExpectedRecModelFactory>,
              "RecModelFactory must return std::unique_ptr<RecCausalLM>.");
static_assert(std::is_base_of_v<CausalLM, RecCausalLM>,
              "RecCausalLM must derive from CausalLM.");

class DummyRecCausalLM final : public RecCausalLM {
 public:
  explicit DummyRecCausalLM(const torch::TensorOptions& options)
      : options_(options) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& parameters) override {
    UNUSED_PARAMETER(tokens);
    UNUSED_PARAMETER(positions);
    UNUSED_PARAMETER(kv_caches);
    UNUSED_PARAMETER(parameters);
    return ModelOutput();
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    UNUSED_PARAMETER(hidden_states);
    UNUSED_PARAMETER(seleted_idxes);
    return torch::Tensor();
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    UNUSED_PARAMETER(loader);
  }

  torch::Device device() const override { return options_.device(); }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {
    UNUSED_PARAMETER(layer_id);
    UNUSED_PARAMETER(expert_ids);
  }

  void update_expert_weight(int32_t layer_id) override {
    UNUSED_PARAMETER(layer_id);
  }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  torch::TensorOptions options_;
};

#if defined(USE_MLU)
// Keep real MTP checkpoint slicing; replace only the large decoder body with
// a scalar weight so this test needs no full checkpoint.
class CheckpointDecoderImpl final : public torch::nn::Module {
 public:
  CheckpointDecoderImpl(const ModelContext& /*context*/, int32_t layer_idx) {
    EXPECT_EQ(layer_idx, 45);  // Quantization prefixes must keep this index.
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      weight_ = weight;
    }
  }

  torch::Tensor forward(torch::Tensor hidden_states,
                        std::optional<torch::Tensor>& /*residual*/,
                        torch::Tensor& /*positions*/,
                        const layer::AttentionMetadata& /*metadata*/,
                        KVCache& /*cache*/,
                        ModelInputParams& /*params*/,
                        const std::optional<torch::Tensor>& /*input_ids*/) {
    return hidden_states;
  }

  void verify_loaded_weights() const {
    ASSERT_TRUE(weight_.defined());
    EXPECT_EQ(weight_.item<float>(), 7.0f);
  }

 private:
  torch::Tensor weight_;
};
TORCH_MODULE(CheckpointDecoder);

using CheckpointMtpModelImpl = MtpModelImplBase<CheckpointDecoder>;
TORCH_MODULE(CheckpointMtpModel);
using GlmCheckpointModel =
    mlu::model::Glm5NextMtpCheckpointImplBase<CheckpointMtpModel>;

class GlmCheckpointLoader final : public ModelLoader {
 public:
  GlmCheckpointLoader(const std::string& prefix,
                      bool include_embedding,
                      bool include_head) {
    states_.reserve(3);
    const std::string layer_prefix = prefix + "layers.45.";
    states_.emplace_back(std::make_unique<StateDict>(
        std::unordered_map<std::string, torch::Tensor>{
            {layer_prefix + "weight", torch::tensor(7.0f)},
            {layer_prefix + "shared_head.norm.weight", torch::full({4}, 11.0f)},
            {prefix + "embed_tokens.weight", torch::full({8, 4}, -1.0f)},
            {"lm_head.weight", torch::full({8, 4}, -2.0f)}}));
    if (include_embedding) {
      states_.emplace_back(std::make_unique<StateDict>(
          std::unordered_map<std::string, torch::Tensor>{
              {layer_prefix + "embed_tokens.weight",
               torch::arange(32, torch::kFloat32).view({8, 4})}}));
    }
    if (include_head) {
      states_.emplace_back(std::make_unique<StateDict>(
          std::unordered_map<std::string, torch::Tensor>{
              {layer_prefix + "shared_head.head.weight",
               torch::arange(32, torch::kFloat32).view({8, 4}) + 3}}));
    }
  }

  std::unique_ptr<Tokenizer> tokenizer() const override { return nullptr; }
  std::vector<std::unique_ptr<StateDict>>& get_state_dicts() override {
    return states_;
  }
  std::string model_weights_path() const override { return ""; }

 private:
  std::vector<std::unique_ptr<StateDict>> states_;
};
#endif

}  // namespace

TEST(HFModelLoaderTest, Qwen35DenseBackendAwareModelTypeSelection) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "architectures": ["Qwen3_5ForConditionalGeneration"],
      "image_token_id": 248056,
      "model_type": "qwen3_5",
      "text_config": {
        "model_type": "qwen3_5_text"
      },
      "video_token_id": 248057,
      "vision_config": {
        "model_type": "qwen3_5"
      }
    }
  )json"));

  const std::filesystem::path fake_model_path("/tmp/Qwen3.5-9B");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path), "qwen3_5_text");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path, "llm"),
            "qwen3_5_text");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path, "vlm"), "qwen3_5");
  EXPECT_EQ(ModelRegistry::get_model_backend("qwen3_5_text"), "llm");
}

TEST(HFModelLoaderTest, Qwen35MoeBackendAwareModelTypeSelection) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "architectures": ["Qwen3_5MoeForConditionalGeneration"],
      "image_token_id": 248056,
      "model_type": "qwen3_5_moe",
      "text_config": {
        "model_type": "qwen3_5_moe_text",
        "num_experts": 256,
        "num_experts_per_tok": 8
      },
      "video_token_id": 248057,
      "vision_config": {
        "model_type": "qwen3_5_moe"
      }
    }
  )json"));

  const std::filesystem::path fake_model_path("/tmp/Qwen3.5-MoE");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path), "qwen3_5_moe_text");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path, "llm"),
            "qwen3_5_moe_text");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path, "vlm"),
            "qwen3_5_moe");
  EXPECT_EQ(ModelRegistry::get_model_backend("qwen3_5_moe_text"), "llm");
}

#if defined(USE_MLU)
TEST(HFModelLoaderTest, Glm5NextRootArgsLoadNativeMluFields) {
  ModelArgsLoader loader = ModelRegistry::get_model_args_loader("glm5_next");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "glm5_next",
      "text_config": {
        "num_hidden_layers": 4,
        "first_k_dense_replace": 3,
        "layer_types": [
          "linear_attention",
          "linear_attention",
          "linear_attention",
          "deepseek_sparse_attention"
        ],
        "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "swiglu_limit": 10.0,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 0.000001,
        "index_kpool": 4,
        "index_kpool_compress": true,
        "index_kpool_always_select_tail": true,
        "linear_attn_config": {
          "gate_lower_bound": -5.0,
          "head_dim": 128,
          "num_heads": 64,
          "short_conv_kernel_size": 4
        }
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "glm5_next");
  EXPECT_EQ(args.scoring_func(), "sigmoid");
  EXPECT_EQ(args.topk_method(), "noaux_tc");
  EXPECT_FLOAT_EQ(args.swiglu_limit(), 10.0F);
  EXPECT_EQ(args.hc_mult(), 4);
  EXPECT_EQ(args.hc_sinkhorn_iters(), 20);
  EXPECT_FLOAT_EQ(args.hc_eps(), 0.000001F);
  EXPECT_EQ(args.index_kpool(), 4);
  EXPECT_TRUE(args.index_kpool_compress());
  EXPECT_TRUE(args.index_kpool_always_select_tail());
  EXPECT_FLOAT_EQ(args.linear_lower_bound(), -5.0F);
  EXPECT_EQ(args.mlp_layer_types(),
            (std::vector<std::string>{"dense", "dense", "dense", "sparse"}));
}

TEST(HFModelLoaderTest, Glm5NextRootTypeHasNativeMluRegistration) {
  EXPECT_EQ(ModelRegistry::get_model_backend("glm5_next"), "llm");
  CausalLMFactory factory = ModelRegistry::get_causallm_factory("glm5_next");
  EXPECT_TRUE(static_cast<bool>(factory));
}

TEST(HFModelLoaderTest, Qwen35MoeRootTypeHasCausalModelFactory) {
  CausalLMFactory factory = ModelRegistry::get_causallm_factory("qwen3_5_moe");
  EXPECT_TRUE(static_cast<bool>(factory));
}
#endif

TEST(HFModelLoaderTest, LoadCompressedTensorsFp8StaticConfig) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "quantization_config": {
        "config_groups": {
          "group_0": {
            "input_activations": {
              "dynamic": false,
              "num_bits": 8,
              "type": "float"
            },
            "weights": {
              "num_bits": 8,
              "type": "float"
            }
          }
        },
        "ignore": [
          "lm_head",
          "model.layers.1.mlp.down_proj"
        ],
        "quant_method": "compressed-tensors"
      }
    }
  )json"));

  QuantArgs quant_args;
  ASSERT_TRUE(load_quant_cfg(reader, quant_args));
  EXPECT_EQ(quant_args.quant_method(), kQuantMethodFp8);
  EXPECT_EQ(quant_args.bits(), 8);
  EXPECT_EQ(quant_args.moe_weight_bits(), 8);
  EXPECT_FALSE(quant_args.activation_dynamic());
  ASSERT_EQ(quant_args.ignored_modules().size(), 2);
  EXPECT_EQ(quant_args.ignored_modules()[0], "lm_head");
  EXPECT_EQ(quant_args.ignored_modules()[1], "model.layers.1.mlp.down_proj");
}

TEST(HFModelLoaderTest, KeepLegacyFp8ConfigUnchanged) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "quantization_config": {
        "activation_scheme": "static",
        "quant_method": "fp8"
      }
    }
  )json"));

  QuantArgs quant_args;
  ASSERT_TRUE(load_quant_cfg(reader, quant_args));
  EXPECT_EQ(quant_args.quant_method(), kQuantMethodFp8);
  EXPECT_FALSE(quant_args.activation_dynamic());
}

TEST(HFModelLoaderTest, RegisterRecFactoryAcceptsRecCausalLmReturnType) {
  const std::string factory_name = "rec_causallm_factory_contract_test";
  RecModelFactory unregistered_factory =
      ModelRegistry::get_rec_model_factory(factory_name);
  EXPECT_FALSE(static_cast<bool>(unregistered_factory));

  ModelRegistry::register_rec_model_factory(
      factory_name,
      [](const ModelContext& context) -> std::unique_ptr<RecCausalLM> {
        UNUSED_PARAMETER(context);
        return nullptr;
      });

  RecModelFactory factory = ModelRegistry::get_rec_model_factory(factory_name);
  EXPECT_TRUE(static_cast<bool>(factory));
}

TEST(HFModelLoaderTest, RecFactoryCreatesRecCausalLmInstance) {
  const std::string kFactoryName = "rec_causallm_instance_contract_test";
  ModelRegistry::register_rec_model_factory(
      kFactoryName,
      [](const ModelContext& context) -> std::unique_ptr<RecCausalLM> {
        UNUSED_PARAMETER(context);
        const torch::TensorOptions options =
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        return std::make_unique<DummyRecCausalLM>(options);
      });

  RecModelFactory factory = ModelRegistry::get_rec_model_factory(kFactoryName);
  ASSERT_TRUE(static_cast<bool>(factory));

  ModelContext context;
  std::unique_ptr<RecCausalLM> rec_model = factory(context);
  ASSERT_NE(rec_model, nullptr);

  CausalLM* causal_model = dynamic_cast<CausalLM*>(rec_model.get());
  EXPECT_NE(causal_model, nullptr);
  EXPECT_EQ(rec_model->device(), torch::Device(torch::kCPU));
}

#if defined(USE_NPU) || defined(USE_MLU)
TEST(HFModelLoaderTest, Glm5NextDefaultsMissingMtpLayerCountToZero) {
  auto loader = ModelRegistry::get_model_args_loader("glm5_next");
  ASSERT_NE(loader, nullptr);
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json({
    "model_type": "glm5_next",
    "text_config": {"num_hidden_layers": 45}
  })json"));
  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.num_nextn_predict_layers(), 0);
}

TEST(HFModelLoaderTest, RegisteredMtpAdapterKeepsTargetAndFixesDraftCache) {
  auto loader = ModelRegistry::get_model_args_loader("glm5_next");
  ASSERT_NE(loader, nullptr);
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json({
    "model_type": "glm5_next",
    "text_config": {
      "num_hidden_layers": 45,
      "num_nextn_predict_layers": 1,
      "index_kpool_compress": true
    }
  })json"));
  ModelArgs target_args;
  ASSERT_TRUE(loader(reader, &target_args));
  target_args.enable_mla(true);
  ModelRegistry::configure_mtp_args(target_args,
                                    "MTP",
                                    /*is_draft_engine=*/false,
                                    /*is_python_model=*/false);
  EXPECT_EQ(target_args.model_type(), "glm5_next");
  EXPECT_EQ(target_args.n_layers(), 45);
  EXPECT_TRUE(has_linear_attention_layers(target_args));
  ModelArgs other_args = target_args;
  ModelRegistry::configure_mtp_args(other_args,
                                    "Eagle3",
                                    /*is_draft_engine=*/true,
                                    /*is_python_model=*/false);
  EXPECT_EQ(other_args.model_type(), "glm5_next");
  EXPECT_EQ(other_args.layer_types(), target_args.layer_types());
  ModelArgs draft_args = target_args;
  ModelRegistry::configure_mtp_args(draft_args,
                                    "mTp",
                                    /*is_draft_engine=*/true,
                                    /*is_python_model=*/false);
  ModelRegistry::configure_mtp_args(draft_args,
                                    "MTP",
                                    /*is_draft_engine=*/true,
                                    /*is_python_model=*/false);
  EXPECT_EQ(draft_args.model_type(), "glm5_next_mtp");
  EXPECT_FALSE(has_linear_attention_layers(draft_args));
  EXPECT_EQ(draft_args.n_layers(), 1);
  EXPECT_EQ(draft_args.mtp_start_layer_idx(), 45);
  EXPECT_EQ(draft_args.layer_types(),
            std::vector<std::string>({"deepseek_sparse_attention"}));
  KVCacheEstimateOptions options;
  options.cache_size_in_bytes = 1024 * 1024 * 1024;
  options.block_size = 16;
  options.world_size = 8;
  options.n_local_kv_heads = 8;
  options.max_seqs_per_batch = 32;
  options.is_draft_engine = true;
  const KVCacheCapacity capacity =
      estimate_kv_cache_capacity(draft_args, options);
  EXPECT_EQ(capacity.n_layers(), 1);
  EXPECT_EQ(capacity.num_linear_attention_layers(), 0);
  EXPECT_EQ(capacity.num_full_attention_layers(), 1);
  EXPECT_EQ(capacity.num_indexer_layers(), 1);
}

#if defined(USE_MLU)
TEST(HFModelLoaderTest, GlmMtpLoadsBodyAndSharesTargetVocabularyWeights) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, device);
  ParallelArgs parallel_args(/*rank=*/0, /*world_size=*/1, &group);
  parallel_args.tp_group_ = &group;
  for (const std::string prefix : {"model.", "model.language_model."}) {
    for (const bool tied : {false, true}) {
      SCOPED_TRACE(prefix);
      SCOPED_TRACE(tied);
      ModelArgs args;
      args.model_type("glm5_next_mtp")
          .n_layers(1)
          .mtp_start_layer_idx(45)
          .num_nextn_predict_layers(1)
          .hidden_size(4)
          .vocab_size(8)
          .tie_word_embeddings(tied);
      ModelContext context(parallel_args, args, QuantArgs(), options);
      GlmCheckpointModel model(context);
      model.load_model(std::make_unique<GlmCheckpointLoader>(
          prefix, /*include_embedding=*/true, /*include_head=*/!tied));
      EXPECT_FALSE(model.has_loaded_vocab_weights());
      const auto expected_embedding =
          torch::arange(32, torch::kFloat32).view({8, 4});
      const auto expected_head =
          tied ? expected_embedding : expected_embedding + 3;
      layer::WordEmbedding target_embedding(context);
      target_embedding->load_state_dict(
          StateDict(std::unordered_map<std::string, torch::Tensor>{
              {"weight", expected_embedding}}));
      layer::LmHead target_head(context);
      target_head->load_state_dict(
          StateDict(std::unordered_map<std::string, torch::Tensor>{
              {"weight", expected_head}}));
      model.set_word_embedding(target_embedding);
      model.set_lm_head(target_head);
      EXPECT_TRUE(model.has_loaded_vocab_weights());
      const auto ids = torch::tensor({0, 3, 7}, torch::kInt64);
      EXPECT_TRUE(torch::equal(
          model.get_input_embeddings(ids.to(device)).cpu().to(torch::kFloat32),
          expected_embedding.index_select(0, ids)));
      EXPECT_TRUE(
          torch::equal(model.logits(torch::eye(4, options), torch::Tensor())
                           .cpu()
                           .to(torch::kFloat32),
                       expected_head.transpose(0, 1)));
      EXPECT_TRUE(
          torch::equal(model.named_parameters()["model.norm.weight"].cpu().to(
                           torch::kFloat32),
                       torch::full({4}, 11.0f)));
    }
  }
}

TEST(HFModelLoaderTest, GlmMtpAllowsMissingOwnVocabularyWeights) {
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  ParallelArgs parallel_args(/*rank=*/0, /*world_size=*/1, &group);
  parallel_args.tp_group_ = &group;
  ModelArgs args;
  args.model_type("glm5_next_mtp")
      .n_layers(1)
      .mtp_start_layer_idx(45)
      .num_nextn_predict_layers(1)
      .hidden_size(4)
      .vocab_size(8);
  ModelContext context(
      parallel_args,
      args,
      QuantArgs(),
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  for (const bool include_embedding : {false, true}) {
    for (const bool include_head : {false, true}) {
      GlmCheckpointModel model(context);
      model.load_model(std::make_unique<GlmCheckpointLoader>(
          "model.", include_embedding, include_head));
      EXPECT_FALSE(model.has_loaded_vocab_weights());
      EXPECT_TRUE(
          torch::equal(model.named_parameters()["model.norm.weight"].cpu(),
                       torch::full({4}, 11.0f)));
    }
  }
}

TEST(HFModelLoaderTest, GlmMtpNativeAdapterRejectsExportedLayout) {
  ModelArgs args;
  args.model_type("glm5_next_mtp").n_layers(1).num_nextn_predict_layers(1);
  EXPECT_DEATH(ModelRegistry::configure_mtp_args(args,
                                                 "MTP",
                                                 /*is_draft_engine=*/true,
                                                 /*is_python_model=*/false),
               "exported draft checkpoints are not supported");
}

TEST(HFModelLoaderTest, MtpModelLoadsExplicitAndLegacyCheckpointOffsets) {
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  ParallelArgs parallel_args(/*rank=*/0, /*world_size=*/1, &group);
  parallel_args.tp_group_ = &group;
  for (const bool normalized : {false, true}) {
    SCOPED_TRACE(normalized);
    ModelArgs args;
    args.model_type("deepseek_v3_mtp")
        .n_layers(45)
        .num_nextn_predict_layers(1)
        .hidden_size(4)
        .vocab_size(8);
    if (normalized) {
      args.model_type("glm5_next");
      ModelRegistry::configure_mtp_args(args,
                                        "MTP",
                                        /*is_draft_engine=*/true,
                                        /*is_python_model=*/false);
    }
    ModelContext context(
        parallel_args,
        args,
        QuantArgs(),
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    MtpModelImplBase<CheckpointDecoder> model(context);
    StateDict weights({
        {"layers.45.weight", torch::tensor(7.0f)},
        {"layers.45.shared_head.norm.weight", torch::full({4}, 11.0f)},
        {"layers.1.weight", torch::tensor(-1.0f)},
        {"layers.1.shared_head.norm.weight", torch::full({4}, -1.0f)},
    });
    model.load_state_dict(weights);
    model.verify_loaded_weights();
    EXPECT_TRUE(torch::equal(model.named_parameters()["norm.weight"],
                             torch::full({4}, 11.0f)));
  }
}
#endif

TEST(HFModelLoaderTest, GlmMtpAdapterLimitsOnlyNativeMluReuse) {
  for (const bool is_python_model : {false, true}) {
    ModelArgs args;
    args.model_type("glm5_next")
        .n_layers(45)
        .num_nextn_predict_layers(1)
        .index_share_for_mtp_iteration(true);
    ModelRegistry::configure_mtp_args(
        args, "MTP", /*is_draft_engine=*/true, is_python_model);
    EXPECT_EQ(args.n_layers(), 1);
    EXPECT_EQ(args.mtp_start_layer_idx(), 45);
    if (Platform::is_mlu() && !is_python_model) {
      EXPECT_FALSE(args.index_share_for_mtp_iteration());
      EXPECT_TRUE(args.index_topk_pattern().empty());
    } else {
      EXPECT_TRUE(args.index_share_for_mtp_iteration());
      EXPECT_EQ(args.index_topk_pattern(), "S");
    }
    ModelRegistry::configure_mtp_args(
        args, "MTP", /*is_draft_engine=*/true, is_python_model);
    EXPECT_EQ(args.n_layers(), 1);
    EXPECT_EQ(args.mtp_start_layer_idx(), 45);
  }
}

TEST(HFModelLoaderTest, GlmMtpAdapterPreservesExportedLayout) {
  ModelArgs args;
  args.model_type("glm5_next_mtp")
      .n_layers(1)
      .num_nextn_predict_layers(1)
      .layer_types({"deepseek_sparse_attention"});
  ModelRegistry::configure_mtp_args(args,
                                    "MTP",
                                    /*is_draft_engine=*/true,
                                    /*is_python_model=*/true);
  EXPECT_EQ(args.n_layers(), 1);
  EXPECT_EQ(args.mtp_start_layer_idx(), -1);
  EXPECT_EQ(args.layer_types(),
            std::vector<std::string>({"deepseek_sparse_attention"}));
}

#if defined(USE_NPU)
TEST(HFModelLoaderTest, Glm5NextNativeMtpUsesAppendedLayer) {
  auto loader = ModelRegistry::get_model_args_loader("glm5_next");
  ASSERT_NE(loader, nullptr);
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json({
    "model_type": "glm5_next",
    "text_config": {
      "num_hidden_layers": 45,
      "num_nextn_predict_layers": 1,
      "first_k_dense_replace": 3,
      "index_share_for_mtp_iteration": true,
      "index_topk_freq": 1
    }
  })json"));
  ModelArgs target_args;
  ASSERT_TRUE(loader(reader, &target_args));
  EXPECT_EQ(target_args.num_nextn_predict_layers(), 1);
  EXPECT_EQ(target_args.mtp_start_layer_idx(), -1);
  EXPECT_FALSE(configure_glm5_next_mtp_args(target_args,
                                            /*speculative_algorithm=*/"MTP",
                                            /*is_draft_engine=*/false));
  ModelArgs draft_args = target_args;
  ASSERT_TRUE(configure_glm5_next_mtp_args(draft_args,
                                           /*speculative_algorithm=*/"MTP",
                                           /*is_draft_engine=*/true));
  EXPECT_EQ(target_args.model_type(), "glm5_next");
  EXPECT_EQ(target_args.n_layers(), 45);
  EXPECT_EQ(draft_args.model_type(), "glm5_next_mtp");
  EXPECT_EQ(draft_args.n_layers(), 1);
  EXPECT_EQ(draft_args.mtp_start_layer_idx(), 45);
  EXPECT_EQ(draft_args.first_k_dense_replace(), 0);
  EXPECT_EQ(draft_args.layer_types(),
            std::vector<std::string>({"deepseek_sparse_attention"}));
  EXPECT_EQ(draft_args.full_attn_layers(), std::vector<int32_t>({0}));
  EXPECT_EQ(draft_args.mlp_layer_types(), std::vector<std::string>({"sparse"}));
  EXPECT_EQ(draft_args.indexer_types(), std::vector<std::string>({"full"}));
  EXPECT_EQ(draft_args.index_topk_freq(), 2);
  EXPECT_EQ(draft_args.index_skip_topk_offset(), 0);
  EXPECT_EQ(draft_args.index_topk_pattern(), "S");
}

TEST(HFModelLoaderTest, Glm5NextExportedMtpKeepsNativeKeys) {
  auto loader = ModelRegistry::get_model_args_loader("glm5_next_mtp");
  ASSERT_NE(loader, nullptr);
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json({
    "model_type": "glm5_next_mtp",
    "text_config": {
      "num_hidden_layers": 1,
      "num_nextn_predict_layers": 1,
      "first_k_dense_replace": 0,
      "layer_types": ["deepseek_sparse_attention"],
      "mlp_layer_types": ["sparse"],
      "indexer_types": ["full"]
    }
  })json"));
  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_FALSE(configure_glm5_next_mtp_args(args,
                                            /*speculative_algorithm=*/"MTP",
                                            /*is_draft_engine=*/true));
  EXPECT_EQ(args.model_type(), "glm5_next_mtp");
  EXPECT_EQ(args.n_layers(), 1);
  EXPECT_EQ(args.mtp_start_layer_idx(), -1);
}

TEST(HFModelLoaderTest, Glm5NextMtpNormalizationIsIdempotent) {
  ModelArgs args;
  args.model_type("glm5_next").n_layers(45).num_nextn_predict_layers(1);
  ASSERT_TRUE(configure_glm5_next_mtp_args(args,
                                           /*speculative_algorithm=*/"MTP",
                                           /*is_draft_engine=*/true));
  EXPECT_FALSE(configure_glm5_next_mtp_args(args,
                                            /*speculative_algorithm=*/"MTP",
                                            /*is_draft_engine=*/true));
  EXPECT_EQ(args.model_type(), "glm5_next_mtp");
  EXPECT_EQ(args.n_layers(), 1);
  EXPECT_EQ(args.mtp_start_layer_idx(), 45);
}

TEST(HFModelLoaderTest, Glm5NextMtpNormalizationIgnoresOtherAlgorithms) {
  for (const std::string_view algorithm : {"", "Eagle3", "DFlash", "DSpark"}) {
    SCOPED_TRACE(algorithm);
    ModelArgs args;
    args.model_type("glm5_next").n_layers(45).num_nextn_predict_layers(1);
    EXPECT_FALSE(configure_glm5_next_mtp_args(
        args, algorithm, /*is_draft_engine=*/true));
    EXPECT_EQ(args.model_type(), "glm5_next");
    EXPECT_EQ(args.n_layers(), 45);
    EXPECT_EQ(args.mtp_start_layer_idx(), -1);
  }
}

TEST(HFModelLoaderTest, Glm5NextMtpNormalizationIgnoresOtherModels) {
  for (const std::string model_type : {"deepseek_v3", "qwen3_5_text"}) {
    SCOPED_TRACE(model_type);
    ModelArgs args;
    args.model_type(model_type).n_layers(45).num_nextn_predict_layers(1);
    EXPECT_FALSE(configure_glm5_next_mtp_args(args,
                                              /*speculative_algorithm=*/"MTP",
                                              /*is_draft_engine=*/true));
    EXPECT_EQ(args.model_type(), model_type);
    EXPECT_EQ(args.n_layers(), 45);
    EXPECT_EQ(args.mtp_start_layer_idx(), -1);
  }
}

TEST(HFModelLoaderTest, Glm5NextMtpEngineAndWorkerMatchExportedCacheLayout) {
  auto loader = ModelRegistry::get_model_args_loader("glm5_next");
  ASSERT_NE(loader, nullptr);
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json({
    "model_type": "glm5_next",
    "text_config": {
      "num_hidden_layers": 45,
      "num_nextn_predict_layers": 1,
      "index_share_for_mtp_iteration": true,
      "index_kpool_compress": true,
      "linear_attn_config": {
        "num_heads": 64,
        "head_dim": 128,
        "short_conv_kernel_size": 4
      }
    }
  })json"));
  ModelArgs engine_args;
  ASSERT_TRUE(loader(reader, &engine_args));
  ASSERT_TRUE(has_linear_attention_layers(engine_args));
  engine_args.enable_mla(true);
  ModelArgs worker_args = engine_args;
  runtime::Options engine_options;
  engine_options.is_draft_engine(true)
      .speculative_algorithm("mTp")
      .enable_speculative_decode(false);
  runtime::Options worker_options = engine_options;
  worker_options.enable_speculative_decode(true);
  ASSERT_TRUE(
      configure_glm5_next_mtp_args(engine_args,
                                   engine_options.speculative_algorithm(),
                                   engine_options.is_draft_engine()));
  ASSERT_TRUE(
      configure_glm5_next_mtp_args(worker_args,
                                   worker_options.speculative_algorithm(),
                                   worker_options.is_draft_engine()));

  JsonReader exported_reader;
  ASSERT_TRUE(exported_reader.parse_text(R"json({
    "model_type": "glm5_next_mtp",
    "text_config": {
      "num_hidden_layers": 1,
      "num_nextn_predict_layers": 1,
      "first_k_dense_replace": 0,
      "layer_types": ["deepseek_sparse_attention"],
      "mlp_layer_types": ["sparse"],
      "indexer_types": ["full"],
      "index_kpool_compress": true,
      "linear_attn_config": {
        "num_heads": 64,
        "head_dim": 128,
        "short_conv_kernel_size": 4
      }
    }
  })json"));
  ModelArgs exported_args;
  ASSERT_TRUE(loader(exported_reader, &exported_args));
  exported_args.enable_mla(true);
  EXPECT_FALSE(
      configure_glm5_next_mtp_args(exported_args,
                                   engine_options.speculative_algorithm(),
                                   engine_options.is_draft_engine()));

  KVCacheEstimateOptions cache_options;
  cache_options.cache_size_in_bytes = 1024 * 1024;
  cache_options.block_size = 128;
  cache_options.world_size = 8;
  cache_options.n_local_kv_heads = 8;
  cache_options.max_seqs_per_batch = 16;
  cache_options.is_draft_engine = true;
  const KVCacheCapacity exported_capacity =
      estimate_kv_cache_capacity(exported_args, cache_options);
  const KVCacheShape exported_shape(
      exported_capacity, exported_args, cache_options.world_size);
  for (const ModelArgs* model_args : {&engine_args, &worker_args}) {
    EXPECT_FALSE(has_linear_attention_layers(*model_args));
    EXPECT_EQ(model_args->mtp_start_layer_idx(), 45);
    const KVCacheCapacity capacity =
        estimate_kv_cache_capacity(*model_args, cache_options);
    EXPECT_EQ(capacity.n_layers(), 1);
    EXPECT_EQ(capacity.num_full_attention_layers(), 1);
    EXPECT_EQ(capacity.num_linear_attention_layers(), 0);
    EXPECT_EQ(capacity.num_indexer_layers(), 1);
    EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 0);
    EXPECT_EQ(capacity.n_blocks(), exported_capacity.n_blocks());
    EXPECT_EQ(capacity.slot_size(), exported_capacity.slot_size());
    EXPECT_EQ(capacity.index_slot_size(), exported_capacity.index_slot_size());
    const KVCacheShape shape(capacity, *model_args, cache_options.world_size);
    EXPECT_EQ(shape.key_cache_shape(), exported_shape.key_cache_shape());
    EXPECT_EQ(shape.value_cache_shape(), exported_shape.value_cache_shape());
    ASSERT_TRUE(shape.has_index_cache_shape());
    EXPECT_EQ(shape.index_cache_shape(), exported_shape.index_cache_shape());
    EXPECT_FALSE(shape.has_conv_cache_shape());
    EXPECT_FALSE(shape.has_ssm_cache_shape());
  }
}

TEST(HFModelLoaderTest, Glm5NextNativeMtpRejectsUnsupportedLayerCounts) {
  ModelArgs args;
  args.model_type("glm5_next").n_layers(45);
  EXPECT_DEATH(configure_glm5_next_mtp_args(args,
                                            /*speculative_algorithm=*/"MTP",
                                            /*is_draft_engine=*/true),
               "exactly one");
  args.num_nextn_predict_layers(2);
  EXPECT_DEATH(configure_glm5_next_mtp_args(args,
                                            /*speculative_algorithm=*/"MTP",
                                            /*is_draft_engine=*/true),
               "exactly one");
  args.num_nextn_predict_layers(1).n_layers(0);
  EXPECT_DEATH(configure_glm5_next_mtp_args(args,
                                            /*speculative_algorithm=*/"MTP",
                                            /*is_draft_engine=*/true),
               "n_layers");
}

TEST(HFModelLoaderTest, Qwen3DSparkFieldsFromTorchConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3",
      "markov_rank": 256,
      "enable_confidence_head": true,
      "confidence_head_with_markov": true
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.markov_rank(), 256);
  EXPECT_TRUE(args.enable_confidence_head());
  EXPECT_TRUE(args.confidence_head_with_markov());
}

TEST(HFModelLoaderTest, DeepseekV4DSparkModelArgsFrom0731Config) {
  auto loader = ModelRegistry::get_model_args_loader("deepseek_v4");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "deepseek_v4",
      "hidden_size": 4096,
      "num_hidden_layers": 43,
      "num_attention_heads": 32,
      "num_key_value_heads": 1,
      "compress_ratios": [1],
      "dspark_block_size": 5,
      "dspark_markov_rank": 256,
      "dspark_noise_token_id": 128799,
      "dspark_target_layer_ids": [40, 41, 42]
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "deepseek_v4");
  EXPECT_EQ(args.dspark_num_layers(), 3);
  // Base loader leaves this 0; applied later in
  // configure_deepseek_v4_dspark_args.
  EXPECT_EQ(args.dspark_block_size(), 0);
  EXPECT_EQ(args.markov_rank(), 256);
  ASSERT_EQ(args.compress_ratios().size(), 43);
}

TEST(HFModelLoaderTest, DeepseekV41ModelArgsFromConfig) {
  auto loader = ModelRegistry::get_model_args_loader("deepseek_v41");
  ASSERT_NE(loader, nullptr);

  // Mirrors the released DeepSeek-V4.1-Flash config; fields that equal the
  // loader defaults in load_deepseek_v41_model_args are omitted.
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "deepseek_v41",
      "image_token_id": 129264,
      "quantization_config": {
        "scale_fmt": "ue8m0"
      },
      "text_config": {
        "sliding_window": 128,
        "compress_ratios": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "engram_layer_ids": [1, 14],
        "engram_pad_token_id": 2,
        "engram_num_embeddings": [384006168, 384016682],
        "engram_max_ngram_size": 4,
        "engram_vocab_size": 16000000,
        "engram_n_heads": 8,
        "engram_head_dim": 256,
        "engram_compressed_vocab_size": 99092,
        "dspark_markov_rank": 256,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "deepseek_v41");
  EXPECT_EQ(args.head_dim(), 512);
  EXPECT_EQ(args.window_size(), 128);
  EXPECT_EQ(args.n_hash_layers(), 0);
  EXPECT_EQ(args.image_token_id(), 129264);
  EXPECT_EQ(args.scale_fmt(), "ue8m0");
  EXPECT_EQ(args.stop_token_ids(), std::unordered_set<int32_t>({1}));
  EXPECT_EQ(args.num_nextn_predict_layers(), 3);
  EXPECT_EQ(args.dspark_num_layers(), 3);

  // Ratios are preserved as configured (no V4-style normalization);
  // 43 entries = 40 backbone + 3 DSpark tail.
  EXPECT_EQ(args.compress_ratios(),
            std::vector<int32_t>({0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                  2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}));

  EXPECT_EQ(args.kv_source_layer_ids(), std::vector<int32_t>({2, 8, 14, 20}));
  EXPECT_EQ(args.index_source_layer_ids(),
            std::vector<int32_t>({2, 8, 14, 20, 24, 28, 32, 36}));
  EXPECT_EQ(args.candidate_source_layer_id(), 20);
  EXPECT_EQ(args.candidate_topk_blocks(), 2048);
  EXPECT_EQ(args.candidate_block_size(), 8);

  EXPECT_EQ(args.engram_layer_ids(), std::vector<int32_t>({1, 14}));
  EXPECT_EQ(args.engram_num_embeddings(),
            std::vector<int64_t>({384006168, 384016682}));
  EXPECT_EQ(args.engram_max_ngram_size(), 4);
  EXPECT_EQ(args.engram_vocab_size(), 16000000);
  EXPECT_EQ(args.engram_n_heads(), 8);
  EXPECT_EQ(args.engram_head_dim(), 256);
  EXPECT_EQ(args.engram_compressed_vocab_size(), 99092);
  EXPECT_EQ(args.engram_pad_token_id(), 2);

  EXPECT_EQ(args.markov_rank(), 256);
  EXPECT_EQ(args.dspark_n_routed_experts(), 128);
  EXPECT_EQ(args.dspark_num_experts_per_tok(), 3);
}
#endif

TEST(HFModelLoaderTest, Qwen35MtpModelArgsFromDenseConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5_mtp");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_5",
      "text_config": {
        "mtp_num_hidden_layers": 1,
        "layer_types": ["linear_attention"]
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "qwen3_5_mtp");
  EXPECT_EQ(args.num_nextn_predict_layers(), 1);
  EXPECT_EQ(args.n_layers(), 1);
  ASSERT_EQ(args.layer_types().size(), 1);
  EXPECT_EQ(args.layer_types()[0], "full_attention");
}

TEST(HFModelLoaderTest, Qwen35MtpModelArgsFromMoeConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5_moe_mtp");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_5_moe",
      "text_config": {
        "mtp_num_hidden_layers": 2,
        "layer_types": ["linear_attention", "linear_attention"]
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "qwen3_5_moe_mtp");
  EXPECT_EQ(args.num_nextn_predict_layers(), 2);
  EXPECT_EQ(args.n_layers(), 2);
  ASSERT_EQ(args.layer_types().size(), 2);
  EXPECT_EQ(args.layer_types()[0], "full_attention");
  EXPECT_EQ(args.layer_types()[1], "full_attention");
}
#endif

#if defined(USE_CUDA)
TEST(HFModelLoaderTest, MiMoMtpModelArgsFromExportedConfig) {
  // export_mtp.py sets num_hidden_layers = mtp_layer_count (1) in the exported
  // config, matching mtp_layers.0.* weight layout loaded by MiMoMtpModelImpl.
  auto loader = ModelRegistry::get_model_args_loader("mimo_mtp");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "mimo_mtp",
      "num_hidden_layers": 1,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "hidden_size": 4096,
      "intermediate_size": 11008,
      "rope_theta": 640000.0,
      "num_nextn_predict_layers": 1
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "mimo_mtp");
  EXPECT_EQ(args.n_layers(), 1);
  EXPECT_EQ(args.n_heads(), 32);
  EXPECT_EQ(args.n_kv_heads(), std::optional<int64_t>(8));
  EXPECT_EQ(args.hidden_size(), 4096);
  EXPECT_EQ(args.intermediate_size(), 11008);
  EXPECT_FLOAT_EQ(args.rope_theta(), 640000.0f);
  EXPECT_EQ(args.num_nextn_predict_layers(), 1);
  // head_dim is derived from hidden_size / n_heads when absent from config
  EXPECT_EQ(args.head_dim(), 128);
  EXPECT_EQ(args.stop_token_ids(),
            std::unordered_set<int32_t>({args.eos_token_id()}));
}

TEST(HFModelLoaderTest, MiMoMtpModelArgsDefaults) {
  // Verify REGISTER_MODEL_ARGS defaults match MiMo-7B-Base architecture.
  auto loader = ModelRegistry::get_model_args_loader("mimo_mtp");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json({ "model_type": "mimo_mtp" })json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "mimo_mtp");
  EXPECT_EQ(args.n_layers(), 36);
  EXPECT_EQ(args.n_heads(), 32);
  EXPECT_EQ(args.hidden_size(), 4096);
  EXPECT_EQ(args.vocab_size(), 151680);
  EXPECT_FLOAT_EQ(args.rope_theta(), 640000.0f);
  EXPECT_EQ(args.num_nextn_predict_layers(), 1);
  EXPECT_EQ(args.head_dim(), 128);
  EXPECT_EQ(args.eos_token_id(), 151643);
  EXPECT_EQ(args.stop_token_ids(), std::unordered_set<int32_t>({151643}));
}

TEST(HFModelLoaderTest, Rwkv7ModelArgsFromConvertedConfig) {
  auto loader = ModelRegistry::get_model_args_loader("rwkv7");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "rwkv7",
      "torch_dtype": "float16",
      "vocab_size": 65536,
      "hidden_size": 768,
      "num_hidden_layers": 12,
      "head_size": 64,
      "intermediate_size": 3072,
      "num_attention_heads": 12,
      "layer_norm_eps": 1e-5,
      "max_position_embeddings": 4096,
      "bos_token_id": 0,
      "eos_token_id": 0
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "rwkv7");
  EXPECT_EQ(args.vocab_size(), 65536);
  EXPECT_EQ(args.hidden_size(), 768);
  EXPECT_EQ(args.n_layers(), 12);
  EXPECT_EQ(args.head_dim(), 64);
  EXPECT_EQ(args.intermediate_size(), 3072);
  EXPECT_EQ(args.n_heads(), 12);
  EXPECT_EQ(args.max_position_embeddings(), 4096);
  EXPECT_FLOAT_EQ(args.layer_norm_eps(), 1e-5f);
  EXPECT_EQ(args.linear_conv_kernel_dim(), 2);
  EXPECT_EQ(args.full_attention_interval(), 13);
  EXPECT_TRUE(has_linear_attention_layers(args));
  ASSERT_EQ(args.layer_types().size(), 12U);
  EXPECT_EQ(args.layer_types().front(), "rwkv7");
  EXPECT_EQ(args.stop_token_ids(), std::unordered_set<int32_t>({0}));
}
#endif  // USE_CUDA

TEST(HFModelLoaderTest, LoadCompressedTensorsInt8Scheme) {
  struct TestCase {
    const char* name;
    const char* config;
    bool is_w8a8_dynamic;
  };
  const TestCase test_cases[] = {
      {"dynamic_activation",
       R"json(
         {
           "quantization_config": {
             "config_groups": {
               "group_0": {
                 "input_activations": {
                   "dynamic": true,
                   "num_bits": 8,
                   "type": "int"
                 },
                 "weights": {
                   "dynamic": false,
                   "num_bits": 8,
                   "type": "int"
                 }
               }
             },
             "ignore": [
               "lm_head",
               "model.layers.0.self_attn.o_proj"
             ],
             "quant_method": "compressed-tensors"
           }
         }
       )json",
       true},
      {"static_activation",
       R"json(
         {
           "quantization_config": {
             "config_groups": {
               "group_0": {
                 "input_activations": {
                   "dynamic": false,
                   "num_bits": 8,
                   "type": "int"
                 },
                 "weights": {
                   "dynamic": false,
                   "num_bits": 8,
                   "type": "int"
                 }
               }
             },
             "quant_method": "compressed-tensors"
           }
         }
       )json",
       false},
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    JsonReader reader;
    ASSERT_TRUE(reader.parse_text(test_case.config));

    QuantArgs quant_args;
    EXPECT_EQ(load_quant_cfg(reader, quant_args), test_case.is_w8a8_dynamic);
    EXPECT_EQ(quant_args.quant_method(), "compressed-tensors");
    EXPECT_EQ(quant_args.is_compressed_tensors_w8a8_dynamic(),
              test_case.is_w8a8_dynamic);
    if (test_case.is_w8a8_dynamic) {
      EXPECT_EQ(quant_args.bits(), 8);
      EXPECT_EQ(quant_args.moe_weight_bits(), 8);
      EXPECT_TRUE(quant_args.activation_dynamic());
      ASSERT_EQ(quant_args.ignored_modules().size(), 2);
      EXPECT_EQ(quant_args.ignored_modules()[0], "lm_head");
      EXPECT_EQ(quant_args.ignored_modules()[1],
                "model.layers.0.self_attn.o_proj");
    }
  }
}

TEST(HFModelLoaderTest, LoadCompressedTensorsWithoutSmooth) {
  JsonReader reader;
  const std::string config = R"json({"quantization_config": {
    "quant_method": "compressed-tensors", "format": "int-quantized",
    "ignore": ["lm_head", "re:.*mlp\\.gate$"],
    "config_groups": {"group_0": {
      "targets": ["Linear"],
      "weights": {"type": "int", "num_bits": 8, "dynamic": false,
                  "strategy": "channel", "symmetric": true,
                  "preserve_smooth": false},
      "input_activations": {"type": "int", "num_bits": 8, "dynamic": true,
                            "strategy": "token", "symmetric": true}
    }}
  }})json";
  std::string text = config;
  ASSERT_TRUE(reader.parse_text(text));
  QuantArgs args;
  ASSERT_TRUE(load_quant_cfg(reader, args));
  EXPECT_EQ(args.quant_method(), "compressed-tensors");
  EXPECT_TRUE(args.is_compressed_tensors_w8a8_dynamic());
  EXPECT_EQ(args.bits(), 8);
  EXPECT_EQ(args.moe_weight_bits(), 8);
  EXPECT_TRUE(args.is_sym());
  EXPECT_TRUE(args.should_ignore_module("model.layers.0.mlp.gate"));
  text.replace(text.find("\"channel\""), 9, "\"group\"");
  ASSERT_TRUE(reader.parse_text(text));
  EXPECT_FALSE(load_quant_cfg(reader, args));
}

TEST(HFModelLoaderTest, LoadCompressedTensorsPreservedSmooth) {
  const auto base = nlohmann::json::parse(R"json({
    "quant_method": "compressed-tensors",
    "config_groups": {"group_0": {
      "weights": {"type": "int", "num_bits": 8},
      "input_activations": {"type": "int", "num_bits": 8, "dynamic": true}
    }}
  })json");
  for (const std::string& key : {"quantization_config", "compression_config"}) {
    for (const std::string& target : {"weights", "input_activations"}) {
      auto config = base;
      config["config_groups"]["group_0"][target]["preserve_smooth"] = true;
      SCOPED_TRACE(key + ": " + config.dump());
      JsonReader reader;
      ASSERT_TRUE(reader.parse_text(nlohmann::json({{key, config}}).dump()));
      QuantArgs args;
      EXPECT_EQ(load_quant_cfg(reader, args), target == "weights");
      if (target == "weights") {
        EXPECT_TRUE(args.for_module("model.layers.0.self_attn.o_proj")
                        .preserve_smooth());
      }
    }
  }
}

TEST(HFModelLoaderTest, LoadCompressedTensorsConfigAlias) {
  for (const std::string& key : {"quantization_config", "compression_config"}) {
    JsonReader reader;
    ASSERT_TRUE(reader.parse_text("{\"" + key + R"json(": {
      "quant_method": "compressed-tensors",
      "ignore": ["lm_head"],
      "config_groups": {"group_0": {
        "weights": {"type": "int", "num_bits": 8},
        "input_activations": {"type": "int", "num_bits": 8, "dynamic": true}
      }}
    }})json"));
    QuantArgs args;
    ASSERT_TRUE(load_quant_cfg(reader, args));
    EXPECT_EQ(args.quant_method(), "compressed-tensors");
    EXPECT_TRUE(args.is_compressed_tensors_w8a8_dynamic());
    EXPECT_TRUE(args.is_sym());
    EXPECT_TRUE(args.should_ignore_module("lm_head"));
  }
}

TEST(HFModelLoaderTest, RejectUnsupportedCompressedTensorsScheme) {
  const auto base = nlohmann::json::parse(R"json({
    "quant_method": "compressed-tensors",
    "config_groups": {"group_0": {
      "weights": {"type": "int", "num_bits": 8},
      "input_activations": {"type": "int", "num_bits": 8, "dynamic": true}
    }}
  })json");
  const std::vector<nlohmann::json> overrides = {
      {{"format", "pack-quantized"}},
      {{"kv_cache_scheme", {{"num_bits", 8}}}},
      {{"transform_config", {{"rotation", true}}}},
      {{"config_groups", nullptr}},
      {{"config_groups", {{"group_0", {{"weights", {{"symmetric", false}}}}}}}},
      {{"config_groups",
        {{"group_0", {{"input_activations", {{"strategy", "tensor"}}}}}}}},
      {{"config_groups",
        {{"group_0", {{"targets", nlohmann::json::array()}}}}}}};
  for (const auto& patch : overrides) {
    auto config = base;
    config.merge_patch(patch);
    SCOPED_TRACE(config.dump());
    JsonReader reader;
    ASSERT_TRUE(reader.parse_text(
        nlohmann::json({{"quantization_config", config}}).dump()));
    QuantArgs args;
    EXPECT_FALSE(load_quant_cfg(reader, args));
  }
}

namespace {

nlohmann::json speculators_eagle3_config() {
  return nlohmann::json{
      {"model_type", "speculators"},
      {"speculators_model_type", "eagle3"},
      {"draft_vocab_size", 32000},
      {"dtype", "bfloat16"},
      {"eagle_aux_hidden_state_layer_ids", nlohmann::json{2, 32, 61}},
      {"transformer_layer_config",
       nlohmann::json{
           {"model_type", "qwen3"},
           {"hidden_size", 5120},
           {"num_attention_heads", 64},
           {"num_key_value_heads", 8},
           {"head_dim", 128},
           {"vocab_size", 151936},
       }},
  };
}

}  // namespace

// A null discriminator must behave like a missing key (JsonReader contract),
// not throw nlohmann type_error.302 from json::value().
TEST(HFModelLoaderTest, NormalizeSpeculatorsNullDiscriminatorIsNoOp) {
  auto config = nlohmann::json{{"model_type", "qwen3"},
                               {"speculators_model_type", nullptr}};
  normalize_speculators_config(&config);
  EXPECT_EQ(config.at("model_type"), "qwen3");
  EXPECT_FALSE(config.contains("use_qk_norm"));
}

TEST(HFModelLoaderTest, NormalizeSpeculatorsMissingDiscriminatorIsNoOp) {
  auto config = speculators_eagle3_config();
  config.erase("speculators_model_type");
  normalize_speculators_config(&config);
  EXPECT_EQ(config.at("model_type"), "speculators");
  EXPECT_FALSE(config.contains("use_qk_norm"));
}

TEST(HFModelLoaderTest, NormalizeSpeculatorsNonEagle3IsNoOp) {
  auto config = speculators_eagle3_config();
  config["speculators_model_type"] = "dspark";
  normalize_speculators_config(&config);
  EXPECT_EQ(config.at("model_type"), "speculators");
  EXPECT_FALSE(config.contains("use_qk_norm"));
}

TEST(HFModelLoaderTest, NormalizeSpeculatorsRewritesConfig) {
  auto config = speculators_eagle3_config();
  normalize_speculators_config(&config);

  EXPECT_EQ(config.at("model_type"), "qwen3_eagle3");
  // QK norm is inferred from the Qwen3 layer template.
  EXPECT_EQ(config.at("use_qk_norm"), true);

  // An explicit use_qk_norm wins over the layer-template inference.
  auto explicit_flag = speculators_eagle3_config();
  explicit_flag["transformer_layer_config"]["use_qk_norm"] = false;
  normalize_speculators_config(&explicit_flag);
  EXPECT_EQ(explicit_flag.at("use_qk_norm"), false);

  // A llama-style layer template never carries QK norm.
  auto llama = speculators_eagle3_config();
  llama["transformer_layer_config"]["model_type"] = "llama";
  normalize_speculators_config(&llama);
  EXPECT_EQ(llama.at("use_qk_norm"), false);

  // An eagle3 config without transformer_layer_config is fatal.
  auto broken = nlohmann::json{{"model_type", "speculators"},
                               {"speculators_model_type", "eagle3"}};
  EXPECT_DEATH(normalize_speculators_config(&broken), "transformer_layer");
}

#if defined(USE_NPU)
TEST(HFModelLoaderTest, Qwen3Eagle3LayersToCaptureFromAuxList) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_eagle3");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_eagle3",
      "eagle_aux_hidden_state_layer_ids": [2, 32, 61]
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  // The draft sizes num_aux_layers off this list's element count; only the
  // count is used, so the speculators boundary indices are stored as-is.
  EXPECT_EQ(args.layers_to_capture(), (std::vector<int32_t>{2, 32, 61}));
}

TEST(HFModelLoaderTest, Qwen3Eagle3LayersToCaptureDefaultsEmpty) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_eagle3");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_eagle3"
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  // Absent list leaves layers_to_capture empty; the draft falls back to the
  // EAGLE-3 low/mid/high default count via num_aux_layers.
  EXPECT_TRUE(args.layers_to_capture().empty());
}
#endif

TEST(HFModelLoaderTest, BoundaryToPostLayerIdsShiftsAndRejectsEmbedding) {
  // Speculators boundary indices (0=embedding, v=output of layer v-1) shift
  // to 0-based post-layer capture indices.
  EXPECT_EQ(AuxHiddenCapture::boundary_to_post_layer_ids({2, 32, 61}),
            (std::vector<int32_t>{1, 31, 60}));
  EXPECT_TRUE(AuxHiddenCapture::boundary_to_post_layer_ids({}).empty());

  // Boundary 0 (the embedding output) is legal in speculators but xLLM only
  // captures decoder-layer outputs; it must be rejected loudly instead of
  // shifted to -1, where it would never match and silently feed the draft
  // uninitialized capture slots.
  EXPECT_DEATH(AuxHiddenCapture::boundary_to_post_layer_ids({0, 32, 61}),
               "Embedding capture");
}

TEST(HFModelLoaderTest, NumAuxLayersFallsBackToThreeWhenListOmitted) {
  ModelArgs args;
  // Omitted capture list: EAGLE-3 low/mid/high default count, matching the
  // speculators None->3 fallback (fc_norm drafts without an aux list must
  // still build three per-chunk norms).
  EXPECT_EQ(AuxHiddenCapture::num_aux_layers(args), 3);

  args.layers_to_capture({2, 32, 61});
  EXPECT_EQ(AuxHiddenCapture::num_aux_layers(args), 3);

  args.layers_to_capture({1, 2, 3, 4});
  EXPECT_EQ(AuxHiddenCapture::num_aux_layers(args), 4);
}

namespace {
nlohmann::json mixed_quant_config() {
  return nlohmann::json::parse(R"json(
    {
      "quantization_config": {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "ignore": ["re:.*mlp\\.gate$", "re:.*norm$"],
        "config_groups": {
          "dense_w8a8": {
            "format": "int-quantized",
            "targets": ["re:.*self_attn\\.(q_b_proj|o_proj)$",
                        "re:.*mlp\\.shared_experts\\..*proj$"],
            "weights": {
              "type": "int", "num_bits": 8, "dynamic": false,
              "strategy": "channel", "symmetric": true,
              "preserve_smooth": true
            },
            "input_activations": {
              "type": "int", "num_bits": 8, "dynamic": true,
              "strategy": "token", "symmetric": true,
              "preserve_smooth": false
            }
          },
          "experts_w4a8": {
            "format": "int4-le-pack-quantized",
            "targets": ["re:.*layers\\.[0-9]+\\.mlp\\.experts\\.[0-9]+\\.(gate_proj|up_proj|down_proj)$"],
            "weights": {
              "type": "int", "num_bits": 4, "dynamic": false,
              "strategy": "group", "group_size": 128, "symmetric": true,
              "preserve_smooth": true
            },
            "input_activations": {
              "type": "int", "num_bits": 8, "dynamic": true,
              "strategy": "token", "symmetric": true,
              "preserve_smooth": false
            }
          }
        }
      }
    }
  )json");
}
}  // namespace

TEST(HFModelLoaderTest, LoadCompressedTensorsMixedPrecision) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(mixed_quant_config().dump()));

  QuantArgs args;
  ASSERT_TRUE(load_quant_cfg(reader, args));
  EXPECT_EQ(args.quant_method(), "compressed-tensors");
  EXPECT_EQ(args.compressed_groups().size(), 2);
  EXPECT_TRUE(args.is_compressed_tensors_w8a8_dynamic());
  EXPECT_EQ(args.bits(), 8);
  const auto expert_args =
      args.for_module("model.layers.3.mlp.experts.7.gate_proj");
  EXPECT_EQ(expert_args.bits(), 4);
  EXPECT_EQ(expert_args.moe_weight_bits(), 4);
  EXPECT_EQ(expert_args.group_size(), 128);
  EXPECT_TRUE(args.activation_dynamic());
  EXPECT_TRUE(args.should_ignore_module("model.layers.0.mlp.gate"));
  EXPECT_EQ(args.module_quant_method("model.layers.3.mlp.experts.7.gate_proj"),
            std::optional<std::string>("w4a8_dynamic"));
  EXPECT_EQ(args.module_quant_method("model.layers.3.self_attn.o_proj"),
            std::optional<std::string>("w8a8_dynamic"));
  EXPECT_EQ(
      args.module_quant_method("model.layers.3.mlp.shared_experts.up_proj"),
      std::optional<std::string>("w8a8_dynamic"));
  EXPECT_EQ(args.module_quant_method("model.layers.3.mlp.gate"), std::nullopt);
  EXPECT_EQ(args.module_quant_method("model.layers.3.self_attn.kv_b_proj"),
            std::nullopt);
  EXPECT_TRUE(
      args.for_module("model.layers.3.self_attn.o_proj").preserve_smooth());
  EXPECT_FALSE(
      args.for_module("model.layers.3.self_attn.kv_b_proj").preserve_smooth());
  EXPECT_FALSE(args.for_module("model.layers.3.mlp.gate").preserve_smooth());
}

TEST(HFModelLoaderTest, ValidateMixedPrecisionModuleSelection) {
  const std::vector<std::string> targets = {
      R"(re:.*layers\.[0-9]+\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$)",
      R"(re:.*layers\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45)\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$)",
      R"(re:.*layers\.(3|7)\.mlp\.experts\.0\.(up_proj|gate_proj)$)",
      "model.layers.3.mlp.experts.0.gate_proj"};
  for (const auto& target : targets) {
    SCOPED_TRACE(target);
    auto config = mixed_quant_config();
    config["quantization_config"]["config_groups"]["experts_w4a8"]["targets"] =
        {target};
    JsonReader reader;
    ASSERT_TRUE(reader.parse_text(config.dump()));
    QuantArgs args;
    ASSERT_TRUE(load_quant_cfg(reader, args));
    EXPECT_EQ(
        args.module_quant_method("model.layers.3.mlp.experts.0.gate_proj"),
        std::optional<std::string>("w4a8_dynamic"));
    EXPECT_EQ(
        args.module_quant_method("model.layers.46.mlp.experts.0.gate_proj")
            .has_value(),
        target == targets.front());
  }
}

TEST(HFModelLoaderTest, ResolveGroupsIndependentlyOfNamesAndSmooth) {
  for (const std::string format : {"mixed-precision", "int-quantized"}) {
    for (bool dense_smooth : {false, true}) {
      for (bool expert_smooth : {false, true}) {
        auto config = mixed_quant_config();
        auto& quant = config["quantization_config"];
        quant["format"] = format;
        auto& groups = quant["config_groups"];
        groups["attention"] = groups["dense_w8a8"];
        groups["routed"] = groups["experts_w4a8"];
        groups.erase("dense_w8a8");
        groups.erase("experts_w4a8");
        groups["attention"]["weights"]["preserve_smooth"] = dense_smooth;
        groups["routed"]["weights"]["preserve_smooth"] = expert_smooth;
        JsonReader reader;
        ASSERT_TRUE(reader.parse_text(config.dump()));
        QuantArgs args;
        ASSERT_TRUE(load_quant_cfg(reader, args));
        const auto dense = args.for_module("model.layers.3.self_attn.o_proj");
        const auto expert =
            args.for_module("model.layers.3.mlp.experts.0.up_proj");
        EXPECT_EQ(dense.bits(), 8);
        EXPECT_EQ(dense.preserve_smooth(), dense_smooth);
        EXPECT_EQ(expert.bits(), 4);
        EXPECT_EQ(expert.preserve_smooth(), expert_smooth);
      }
    }
  }
}

TEST(HFModelLoaderTest, SingleGroupDefaultsWithoutSmooth) {
  auto config = mixed_quant_config();
  auto& groups = config["quantization_config"]["config_groups"];
  groups.erase("experts_w4a8");
  groups["dense_w8a8"]["weights"].erase("preserve_smooth");
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(config.dump()));
  QuantArgs args;
  ASSERT_TRUE(load_quant_cfg(reader, args));
  const auto local = args.for_module("model.layers.3.self_attn.o_proj");
  EXPECT_EQ(local.bits(), 8);
  EXPECT_FALSE(local.preserve_smooth());
  EXPECT_TRUE(args.for_module("model.layers.3.mlp.experts.0.up_proj")
                  .quant_method()
                  .empty());
}

TEST(HFModelLoaderTest, SameBitwidthHasLayerSpecificSmooth) {
  auto config = mixed_quant_config();
  auto& groups = config["quantization_config"]["config_groups"];
  groups.erase("experts_w4a8");
  groups["layer_zero"] = groups["dense_w8a8"];
  groups["layer_zero"]["targets"] = {"model.layers.0.self_attn.o_proj"};
  groups["layer_zero"]["weights"]["preserve_smooth"] = false;
  groups["dense_w8a8"]["targets"] = {"model.layers.1.self_attn.o_proj"};
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(config.dump()));
  QuantArgs args;
  ASSERT_TRUE(load_quant_cfg(reader, args));
  EXPECT_FALSE(
      args.for_module("model.layers.0.self_attn.o_proj").preserve_smooth());
  EXPECT_TRUE(
      args.for_module("model.layers.1.self_attn.o_proj").preserve_smooth());
}

TEST(HFModelLoaderDeathTest, ConflictingGroupsFailUnlessIgnored) {
  auto config = mixed_quant_config();
  auto& groups = config["quantization_config"]["config_groups"];
  groups["duplicate"] = groups["dense_w8a8"];
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(config.dump()));
  QuantArgs args;
  ASSERT_TRUE(load_quant_cfg(reader, args));
  EXPECT_TRUE(
      args.for_module("model.layers.3.self_attn.o_proj").preserve_smooth());
  groups["duplicate"]["weights"]["preserve_smooth"] = false;
  ASSERT_TRUE(reader.parse_text(config.dump()));
  ASSERT_TRUE(load_quant_cfg(reader, args));
  EXPECT_DEATH(args.for_module("model.layers.3.self_attn.o_proj"),
               "Conflicting");
  args.ignored_modules() = {"model.layers.3.self_attn.o_proj"};
  EXPECT_TRUE(args.for_module("model.layers.3.self_attn.o_proj")
                  .quant_method()
                  .empty());
}

TEST(HFModelLoaderTest, RejectMixedPrecisionMalformedRegex) {
  for (const std::string group : {"experts_w4a8", "dense_w8a8"}) {
    auto config = mixed_quant_config();
    config["quantization_config"]["config_groups"][group]["targets"] = {
        "re:*("};
    JsonReader reader;
    ASSERT_TRUE(reader.parse_text(config.dump()));
    QuantArgs args;
    EXPECT_FALSE(load_quant_cfg(reader, args));
  }
}

}  // namespace xllm
