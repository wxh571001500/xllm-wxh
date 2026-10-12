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
#include <optional>

#include "runtime/forward_params.h"

namespace xllm {
class ProcessGroup;
class Stream;
class WorkerImpl;
struct ParallelArgs;

namespace speculative_worker {

ProcessGroup* sampling_process_group(const ParallelArgs& parallel_args);

// The caller selects the group(s) and whether consensus is required.
void broadcast_tokens_in_group(torch::Tensor& tokens,
                               ProcessGroup* process_group,
                               int32_t root_rank = 0);

void record_metadata_ready_event(Stream& stream, ForwardInput& input);
void wait_metadata_ready_event(const ForwardInput& input, Stream& stream);
void clear_selected_embeddings(ForwardOutput& output);
void clear_all_output_embeddings(ForwardOutput& output);

struct WorkerStreamOptions {
  bool record_input_ready_event = true;
  bool record_output_ready_event = true;
};

// Prepare and execute on the supplied streams without a host wait after
// execution. MTP can defer the leaf LLM output event until its final sampling
// work; DFlash retains the default input and output events. Other worker
// types keep their own output-event behavior.
std::optional<ForwardOutput> run_worker_no_sync(
    WorkerImpl& worker,
    const ForwardInput& input,
    Stream& prepare_stream,
    Stream& compute_stream,
    ForwardInput* processed_output = nullptr,
    WorkerStreamOptions options = {});

}  // namespace speculative_worker
}  // namespace xllm
