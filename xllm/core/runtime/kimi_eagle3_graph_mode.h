/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "core/framework/config/execution_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/runtime/options.h"
#include "core/util/utils.h"

// Shared helpers for the Kimi K25 Eagle3 target MLA ACL-graph decode path.
// Used by both the ACL graph executor (which decides capture/replay/eager and
// builds the graph key) and the worker (which builds DpEpPadding and pads local
// rows). Keeping the bucket ladder and the graph-vs-eager predicate in one
// place guarantees the executor and worker make identical, all-DP-rank-
// consistent decisions, which is required to keep HCCL collectives in sync.
//
// Graph keying strategy: CANONICALIZE. Every DP rank is padded to a uniform
// bucket B and DpEpPadding is built from a uniform [B,B,B,B] layout, so a
// single graph per bucket replays for any real DP token layout. Standard graph
// behavior; padding rows cost full MoE compute (accepted tradeoff).
namespace xllm::npu {

// Is this engine/model the Kimi K25 Eagle3 target (non-draft) graph target?
inline bool is_kimi_k25_eagle3_target(const ModelArgs& args,
                                      const runtime::Options& options) {
  return args.model_type() == "kimi_k25" &&
         options.enable_speculative_decode() && !options.is_draft_engine() &&
         options.speculative_algorithm() == "Eagle3";
}

// Decode bucket ladder shared by executor (bucket_num_tokens) and worker
// (uniform DpEpPadding + local row padding). MUST stay identical in both call
// sites, otherwise ranks would pad to different B and desync HCCL.
inline uint32_t kimi_eagle3_bucket_num_tokens(uint32_t num_tokens) {
  if (::xllm::ExecutionConfig::get_instance()
          .enable_graph_mode_decode_no_padding()) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  } else if (num_tokens <= 2) {
    return 2;
  } else if (num_tokens <= 4) {
    return 4;
  } else if (num_tokens <= 8) {
    return 8;
  }
  return ((num_tokens + 15) / 16) * 16;
}

// Decide, using ONLY globally-broadcast DP metadata + static config, whether a
// Kimi K25 Eagle3 decode step will run the canonicalized graph (worker must
// then build uniform DpEpPadding) vs pure eager (worker keeps real counts).
//
// Every input here is identical across all DP ranks (dp_global_token_nums,
// dp_is_decode, dp_global_kv_max_seq_lens are all-gathered and broadcast; the
// rest is static config), so all ranks reach the same answer and stay in sync
// on HCCL collectives. This mirrors the executor's graph-vs-eager gate exactly.
//
// Returns the uniform bucket B (>0) when canonicalize applies; 0 otherwise.
inline uint32_t kimi_eagle3_canonicalize_bucket(
    const ModelArgs& args,
    const runtime::Options& options,
    const ModelInputParams& params) {
  if (!::xllm::ExecutionConfig::get_instance().enable_graph()) return 0;
  if (!is_kimi_k25_eagle3_target(args, options)) return 0;
  if (args.n_layers() <= 1) return 0;
  // Decode-only, and ALL dp ranks must be decode (else bucket sizing is
  // unsafe).
  if (!params.meta.batch_forward_type.is_decode()) return 0;
  const auto& dp_tokens = params.parallel.dp_global_token_nums;
  const auto& dp_is_decode = params.parallel.dp_is_decode;
  if (dp_tokens.size() <= 1) return 0;
  if (dp_is_decode.size() != dp_tokens.size()) return 0;
  if (std::find(dp_is_decode.begin(), dp_is_decode.end(), 0) !=
      dp_is_decode.end())
    return 0;
  const uint32_t num_decoding_tokens = static_cast<uint32_t>(
      std::max<int64_t>(options.num_decoding_tokens(), 1));
  const uint32_t graph_num_tokens = static_cast<uint32_t>(util::max(dp_tokens));
  const uint32_t global_batch_size = graph_num_tokens / num_decoding_tokens;
  const uint32_t limit = static_cast<uint32_t>(
      std::max<int32_t>(1,
                        ::xllm::ExecutionConfig::get_instance()
                            .acl_graph_decode_batch_size_limit()));
  if (global_batch_size > limit) return 0;
  // DP-consistent seq-len gate: use the global max so all ranks agree (a
  // per-rank local check could make ranks diverge graph/eager -> HCCL hang).
  if (!params.parallel.dp_global_kv_max_seq_lens.empty()) {
    const int64_t global_kv_max =
        util::max(params.parallel.dp_global_kv_max_seq_lens);
    if (global_kv_max > args.max_position_embeddings()) return 0;
  }
  return kimi_eagle3_bucket_num_tokens(graph_num_tokens);
}

}  // namespace xllm::npu
