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

#include <torch/torch.h>

#include <cstdint>
#include <optional>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/dense_mlp.h"
#include "layers/common/rms_norm.h"
#include "layers/mlu/deepseek_v2_attention.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_sparse_moe_block.h"
#include "layers/mlu/glm5_next/glm5_next_kda.h"
#include "layers/mlu/glm5_next/glm5_next_pcp_context.h"
#include "layers/mlu/hyper_connection.h"

namespace xllm {
namespace layer {

enum class Glm5NextAttentionRole : int8_t {
  KDA = 0,
  DSA = 1,
};

enum class Glm5NextMlpRole : int8_t {
  DENSE = 0,
  SPARSE = 1,
};

struct Glm5NextLayerRole final {
  Glm5NextAttentionRole attention = Glm5NextAttentionRole::DSA;
  Glm5NextMlpRole mlp = Glm5NextMlpRole::SPARSE;
};

Glm5NextLayerRole resolve_glm5_next_layer_role(const ModelArgs& args,
                                               int32_t layer_id);

class Glm5NextDecoderLayerImpl final : public torch::nn::Module {
 public:
  Glm5NextDecoderLayerImpl(const ModelContext& context, int32_t layer_id);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;

  torch::Tensor forward(torch::Tensor& hidden_states,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        std::optional<PendingMHC>* pending_mhc = nullptr,
                        bool is_last_layer = true,
                        bool materialize_output = false,
                        const glm5_next_pcp::Context* pcp_context = nullptr);

  torch::Tensor forward_pcp(torch::Tensor& hidden_states,
                            std::optional<torch::Tensor>& residual,
                            torch::Tensor& positions,
                            const AttentionMetadata& global_metadata,
                            const AttentionMetadata& local_metadata,
                            KVCache& kv_cache,
                            const ModelInputParams& input_params,
                            std::optional<PendingMHC>* pending_mhc,
                            bool is_last_layer,
                            bool materialize_output,
                            const glm5_next_pcp::Context& context);

 private:
  int32_t layer_id_ = 0;
  Glm5NextLayerRole role_;
  ProcessGroup* tp_group_ = nullptr;

  MHCPre attn_hc_pre_{nullptr};
  MHCPre ffn_hc_pre_{nullptr};
  MHCPost hc_post_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};

  Glm5NextKDA kda_{nullptr};
  DeepseekV2Attention dsa_{nullptr};
  DenseMLP dense_mlp_{nullptr};
  DeepseekV4SparseMoEBlock sparse_moe_{nullptr};
};

TORCH_MODULE(Glm5NextDecoderLayer);

}  // namespace layer
}  // namespace xllm
