/* Copyright 2026 The xLLM Authors.

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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/runtime/options.h"
#include "core/util/utils.h"

namespace xllm::npu {

inline bool is_kimi_k25_eagle3_target(const ModelArgs& args,
                                      const runtime::Options& options) {
  return args.model_type() == "kimi_k25" &&
         options.enable_speculative_decode() && !options.is_draft_engine() &&
         options.speculative_algorithm() == "Eagle3";
}

inline uint32_t kimi_eagle3_bucket_num_tokens(uint32_t num_tokens) {
  if (::xllm::ExecutionConfig::get_instance()
          .enable_graph_mode_decode_no_padding()) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  }
  if (num_tokens <= 2) {
    return 2;
  }
  if (num_tokens <= 4) {
    return 4;
  }
  if (num_tokens <= 8) {
    return 8;
  }
  return ((num_tokens + 15) / 16) * 16;
}

// Returns the uniform DP graph bucket, or zero when this step remains eager.
inline uint32_t kimi_eagle3_canonicalize_bucket(
    const ModelArgs& args,
    const runtime::Options& options,
    const ModelInputParams& params) {
  if (!::xllm::ExecutionConfig::get_instance().enable_graph() ||
      !is_kimi_k25_eagle3_target(args, options) || args.n_layers() <= 1 ||
      !params.meta.batch_forward_type.is_decode()) {
    return 0;
  }

  const std::vector<int32_t>& dp_tokens = params.parallel.dp_global_token_nums;
  const std::vector<int32_t>& dp_is_decode = params.parallel.dp_is_decode;
  if (dp_tokens.size() <= 1 || dp_is_decode.size() != dp_tokens.size() ||
      std::find(dp_is_decode.begin(), dp_is_decode.end(), 0) !=
          dp_is_decode.end()) {
    return 0;
  }

  const uint32_t num_decoding_tokens = static_cast<uint32_t>(
      std::max<int64_t>(options.num_decoding_tokens(), 1));
  const uint32_t graph_num_tokens = static_cast<uint32_t>(util::max(dp_tokens));
  const uint32_t global_batch_size = graph_num_tokens / num_decoding_tokens;
  const uint32_t batch_size_limit = static_cast<uint32_t>(
      std::max<int32_t>(1,
                        ::xllm::ExecutionConfig::get_instance()
                            .acl_graph_decode_batch_size_limit()));
  if (global_batch_size > batch_size_limit) {
    return 0;
  }

  if (!params.parallel.dp_global_kv_max_seq_lens.empty() &&
      util::max(params.parallel.dp_global_kv_max_seq_lens) >
          args.max_position_embeddings()) {
    return 0;
  }
  return kimi_eagle3_bucket_num_tokens(graph_num_tokens);
}

}  // namespace xllm::npu
