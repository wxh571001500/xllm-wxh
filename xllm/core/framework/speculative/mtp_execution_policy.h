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

#include <cstdint>

#include "core/framework/model/causal_lm.h"
#include "core/framework/speculative/mtp_async_state.h"

namespace xllm::mtp_async {

// Semantic knobs shared by empty-batch expansion, target replay construction,
// and recursive draft input construction. Keeping these together prevents a
// change to the replay protocol from updating only one execution path.
struct DraftContextReplaySemantics final {
  bool full_target_replay;
  int32_t target_expansion_width;
  int32_t draft_position_offset;
};

DraftContextReplaySemantics draft_context_replay_semantics(
    DraftContextUpdate update,
    int32_t num_speculative_tokens);

// Match a model-declared native replay contract to its active verify mode.
bool replay_compatible(TargetSpecVerifyMode target_mode,
                       DraftContextUpdate update,
                       bool target_capable,
                       bool draft_capable,
                       bool is_python_target,
                       bool is_python_draft,
                       bool uses_embedded_eagle3);

}  // namespace xllm::mtp_async
