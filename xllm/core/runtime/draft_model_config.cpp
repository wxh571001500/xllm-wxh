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

#include "core/runtime/draft_model_config.h"

#include <glog/logging.h>

#include <string_view>
#include <unordered_set>
#include <utility>

#include "core/framework/config/speculative_config.h"
#include "core/framework/model/aux_hidden_capture.h"
#include "core/framework/model/model_args.h"
#include "core/platform/platform.h"
#include "core/runtime/options.h"
#include "core/util/json_reader.h"
#include "core/util/utils.h"
#if defined(USE_NPU)
#include "core/framework/config/kernel_config.h"
#endif

namespace xllm {
namespace {

// The caller checks backend support before applying the runtime geometry.
// Call before constructing layers so attention, convolution and dummy inputs
// share the same block size, which includes the anchor token.
void apply_dflash2_runtime_block_size(ModelArgs& args,
                                      int32_t runtime_block_size) {
  const int32_t trained_block_size = args.dflash2_block_size();
  CHECK_GT(trained_block_size, 0);
  CHECK_GT(runtime_block_size, 1) << "DFlash2 requires candidate tokens.";
  CHECK_LE(runtime_block_size, trained_block_size)
      << "DFlash2 runtime block must not exceed the checkpoint's "
         "dflash_config.block_size (which includes the anchor).";
  CHECK_LE(args.dflash2_conv_kernel_size(), runtime_block_size)
      << "DFlash2 runtime block must cover the convolution kernel.";
  args.dflash2_block_size(runtime_block_size);
  args.num_speculative_tokens(runtime_block_size - 1);
  LOG(INFO) << "DFlash2 trained block size: " << trained_block_size
            << ", runtime block size: " << runtime_block_size
            << ", candidate tokens: " << runtime_block_size - 1;
}

#if defined(USE_NPU)
int32_t read_block_size(const std::string& model_weights_path) {
  JsonReader reader;
  const std::string config_path = model_weights_path + "/config.json";
  CHECK(reader.parse(config_path))
      << "Failed to parse DSpark draft config: " << config_path;
  return reader.value_or<int32_t>("dspark_block_size", 0);
}

void configure_deepseek_v4_dspark_args(ModelArgs& args,
                                       const runtime::Options& options) {
  CHECK_GT(args.dspark_num_layers(), 0)
      << "DeepSeek-V4 DSpark requires at least one draft layer.";
  args.n_layers(args.dspark_num_layers());
  args.n_hash_layers(0);

  // Default to the checkpoint's block_size; --num_speculative_tokens overrides.
  const int32_t ckpt_block_size = read_block_size(options.model_path());
  const int32_t user_num_spec = options.num_speculative_tokens();
  if (user_num_spec > 0 && user_num_spec != ckpt_block_size) {
    LOG(WARNING) << "--num_speculative_tokens=" << user_num_spec
                 << " overrides DSpark checkpoint dspark_block_size="
                 << ckpt_block_size << ".";
  }
  args.dspark_block_size(user_num_spec > 0 ? user_num_spec : ckpt_block_size);

  // DSpark stages are all standard SWA layers. Their stage ids are not target
  // model layer ids, so target compress_ratios[0..N) must not be reused.
  args.compress_ratios(
      std::vector<int32_t>(static_cast<size_t>(args.dspark_num_layers()),
                           /*value=*/1));

  args.dspark_use_native_sas(
      KernelConfig::get_instance().enable_dspark_native_sas());
  args.enable_confidence_head(options.enable_adaptive_speculative_decode());
  args.confidence_head_with_markov(true);
}
#endif

// The caller has already classified the algorithm as block diffusion.
bool supports_block_diffusion_draft(std::string_view algorithm) {
  if (Platform::is_npu()) {
    return true;
  }
  if (Platform::is_mlu()) {
    return SpeculativeConfig::is_dflash2_algorithm(algorithm);
  }
  return false;
}

}  // namespace

// Reads a draft config's target-side aux hidden capture layers as 0-based
// post-layer indices: the legacy target-layer keys are already post-layer, the
// speculators boundary-index keys are shifted. With `required` an empty result
// is fatal; otherwise it is returned for the caller to default.
std::vector<int32_t> read_capture_layer_ids(
    const std::string& model_weights_path,
    bool required) {
  JsonReader reader;
  const std::string config_path = model_weights_path + "/config.json";
  if (!reader.parse(config_path)) {
    CHECK(!required) << "Failed to parse draft config: " << config_path;
    return {};
  }

  // Legacy xLLM/vLLM draft configs already use 0-based post-layer output
  // indices, which match ModelArgs::layers_to_capture directly.
  std::vector<int32_t> capture_layer_ids =
      reader.value_or<std::vector<int32_t>>(
          std::vector<std::string>{"dspark_target_layer_ids",
                                   "target_layer_ids",
                                   "dflash_config.target_layer_ids"},
          std::vector<int32_t>{});
  if (!capture_layer_ids.empty()) {
    return capture_layer_ids;
  }

  // Speculators-format keys are hidden-state boundary indices (0=embedding
  // output, v=output of layer v-1); shift them to the post-layer contract.
  capture_layer_ids = reader.value_or<std::vector<int32_t>>(
      std::vector<std::string>{"aux_hidden_state_layer_ids",
                               "eagle_aux_hidden_state_layer_ids"},
      std::vector<int32_t>{});
  capture_layer_ids =
      AuxHiddenCapture::boundary_to_post_layer_ids(capture_layer_ids);
  CHECK(!required || !capture_layer_ids.empty())
      << "Block-diffusion draft config requires dspark_target_layer_ids, "
         "target_layer_ids, dflash_config.target_layer_ids, or "
         "aux_hidden_state_layer_ids: "
      << config_path;
  return capture_layer_ids;
}

// Configure both target capture and draft identity in one place. Backend
// capability checks precede changes to model arguments.
void configure_block_diffusion_model(ModelArgs& args,
                                     const runtime::Options& options) {
  if (!options.is_draft_engine()) {
    CHECK(options.draft_model_path().has_value())
        << "block-diffusion speculative decoding requires --draft_model.";
    std::vector<int32_t> capture_ids =
        read_capture_layer_ids(*options.draft_model_path());
    std::unordered_set<int32_t> unique_ids;
    for (const int32_t layer_id : capture_ids) {
      CHECK_GE(layer_id, 0) << "Invalid target capture layer.";
      CHECK_LT(layer_id, args.n_layers())
          << "Target capture layer is out of range.";
      CHECK(unique_ids.insert(layer_id).second)
          << "Duplicate target capture layer.";
    }
    args.layers_to_capture(std::move(capture_ids));
    return;
  }

  const std::string& algorithm = options.speculative_algorithm();
  CHECK(supports_block_diffusion_draft(algorithm))
      << algorithm << " block-diffusion draft is not supported on "
      << Platform::type_str() << ".";
  if (SpeculativeConfig::is_dflash2_algorithm(algorithm) &&
      Platform::supports_dflash2_runtime_block_size()) {
    apply_dflash2_runtime_block_size(args, options.draft_graph_query_width());
  }
  const bool is_dspark = algorithm == "DSpark";
  const bool is_deepseek_v4_dspark =
      is_dspark && util::is_deepseek_v4_model_type(args.model_type());
  std::string draft_model_type = std::string(kDFlashDraftModelType);
  if (is_dspark) {
    draft_model_type = std::string(kDSparkDraftModelType);
  } else if (SpeculativeConfig::is_dflash2_algorithm(algorithm)) {
    draft_model_type = std::string(kDFlash2DraftModelType);
    CHECK_GT(args.dflash2_block_size(), 0);
    args.dummy_token_count(args.dflash2_block_size());
  }
  if (is_deepseek_v4_dspark) {
    draft_model_type = std::string(util::kDeepseekV4DSparkModelType);
  }
  LOG(INFO) << "Overriding draft model_type from " << args.model_type()
            << " to " << draft_model_type
            << " for block-diffusion speculative decoding";
  args.layers_to_capture({});
  args.model_type(draft_model_type);
  args.requires_eager_execution(!is_deepseek_v4_dspark);
#if defined(USE_NPU)
  if (is_deepseek_v4_dspark) {
    configure_deepseek_v4_dspark_args(args, options);
  }
#endif
}

}  // namespace xllm
