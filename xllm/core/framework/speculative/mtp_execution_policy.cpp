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

#include "core/framework/speculative/mtp_execution_policy.h"

#include <glog/logging.h>

namespace xllm::mtp_async {

bool replay_compatible(TargetSpecVerifyMode target_mode,
                       DraftContextUpdate update,
                       bool target_capable,
                       bool draft_capable,
                       bool is_python_target,
                       bool is_python_draft,
                       bool uses_embedded_eagle3) {
  if (update == DraftContextUpdate::TAIL_EXTEND) {
    return true;
  }
  return update == DraftContextUpdate::ACCEPTED_SPAN_REPLAY && target_capable &&
         draft_capable && !is_python_target && !is_python_draft &&
         !uses_embedded_eagle3 &&
         (target_mode == TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL ||
          target_mode == TargetSpecVerifyMode::EXPANDED_VERIFY);
}

DraftContextReplaySemantics draft_context_replay_semantics(
    DraftContextUpdate update,
    int32_t num_speculative_tokens) {
  CHECK_GT(num_speculative_tokens, 0);
  switch (update) {
    case DraftContextUpdate::TAIL_EXTEND:
      return {/*full_target_replay=*/false,
              /*target_expansion_width=*/2,
              /*draft_position_offset=*/0};
    case DraftContextUpdate::ACCEPTED_SPAN_REPLAY:
      return {/*full_target_replay=*/true,
              /*target_expansion_width=*/num_speculative_tokens + 1,
              /*draft_position_offset=*/-1};
  }
  LOG(FATAL) << "Unknown draft context update policy: "
             << static_cast<int32_t>(update);
}

}  // namespace xllm::mtp_async
