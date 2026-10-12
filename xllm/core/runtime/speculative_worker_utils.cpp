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

#include "runtime/speculative_worker_utils.h"

#include <glog/logging.h>

#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/process_group.h"
#include "runtime/llm_worker_impl.h"

namespace xllm::speculative_worker {

ProcessGroup* sampling_process_group(const ParallelArgs& parallel_args) {
  return parallel_args.tp_group_ != nullptr ? parallel_args.tp_group_
                                            : parallel_args.process_group_;
}

void broadcast_tokens_in_group(torch::Tensor& tokens,
                               ProcessGroup* process_group,
                               int32_t root_rank) {
  if (process_group == nullptr || process_group->world_size() <= 1 ||
      !tokens.defined()) {
    return;
  }
  tokens = tokens.contiguous();
  process_group->broadcast(tokens, root_rank);
}

void record_metadata_ready_event(Stream& stream, ForwardInput& input) {
  input.metadata_ready_event = stream.record_event_or_sync();
}

void wait_metadata_ready_event(const ForwardInput& input, Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait speculative metadata ready event";
}

void clear_selected_embeddings(ForwardOutput& output) {
  output.sample_output.selected_embeddings = torch::Tensor();
}

void clear_all_output_embeddings(ForwardOutput& output) {
  output.sample_output.embeddings = torch::Tensor();
  clear_selected_embeddings(output);
}

std::optional<ForwardOutput> run_worker_no_sync(WorkerImpl& worker,
                                                const ForwardInput& input,
                                                Stream& prepare_stream,
                                                Stream& compute_stream,
                                                ForwardInput* processed_output,
                                                WorkerStreamOptions options) {
  if (processed_output == nullptr) {
    ForwardInput processed_input;
    return run_worker_no_sync(worker,
                              input,
                              prepare_stream,
                              compute_stream,
                              &processed_input,
                              options);
  }
  ForwardInput& processed_input = *processed_output;
  worker.prepare_work_before_execute_on_stream(
      input, processed_input, prepare_stream, options.record_input_ready_event);
  if (!options.record_output_ready_event) {
    if (auto* llm_worker = dynamic_cast<LLMWorkerImpl*>(&worker);
        llm_worker != nullptr) {
      return llm_worker->execute_no_sync_on_stream(
          processed_input, compute_stream, /*record_ready_event=*/false);
    }
  }
  return worker.execute_no_sync_on_stream(processed_input, compute_stream);
}

}  // namespace xllm::speculative_worker
