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

#include <unordered_map>

#include "runtime/dflash_worker_impl.h"

namespace xllm {

class ProcessGroup;

class DFlash2WorkerImpl final : public DFlashWorkerImpl {
 public:
  DFlash2WorkerImpl(const ParallelArgs& parallel_args,
                    const torch::Device& device,
                    const runtime::Options& options);

  ~DFlash2WorkerImpl() override;

 protected:
  bool can_prepare_without_compute_stream_wait(
      const ForwardInput& input) const override;
  DraftBlock run_decode_draft(const ForwardInput& input,
                              ForwardInput& validate_input) override;

  bool prepare_draft_prelaunch(const ForwardInput& input) override;
  void launch_draft_prelaunch(const ForwardInput& input,
                              const SampleOutput& output,
                              const torch::Tensor& accepted_host) override;
  void discard_draft_prelaunch() override;
  void finish_draft_prelaunch(const ForwardInput& input) override;

 private:
  friend class DFlash2WorkerImplTestPeer;

  void prepare_next_target(const ForwardInput& input);
  bool can_stage_prelaunch_input(const ForwardInput& input) const;
  DraftBlock execute_draft_query(const ForwardInput& input,
                                 ForwardInput query_input,
                                 const torch::Tensor& anchor_token_ids,
                                 const torch::Tensor& gumbel_noise,
                                 bool prepared,
                                 Stream* execution_stream = nullptr);
  bool pending_draft_matches(const ForwardInput& input) const;
  DraftBlock execute_prelaunch_body(const ForwardInput& input,
                                    ForwardInput& query,
                                    const torch::Tensor& accepted_tokens,
                                    const torch::Tensor& embeddings,
                                    const torch::Tensor& base_positions,
                                    const torch::Tensor& noise,
                                    Stream& stream);
  struct DraftGraph;
  DraftBlock replay_prelaunch_graph(const ForwardInput& input,
                                    ForwardInput query,
                                    const SampleOutput& output,
                                    const torch::Tensor& noise,
                                    const torch::Tensor& base_positions);
  struct DraftGraphResources;
  std::unique_ptr<DraftGraphResources> draft_graph_resources_;
  std::unordered_map<uint64_t, std::unique_ptr<DraftGraph>> draft_graphs_;
  SamplingParameters prepared_sampling_;
  bool prelaunch_graph_enabled_ = false;

  struct DraftTemplate {
    ForwardInput query;
    SamplingParameters sampling;
    torch::Tensor noise;
    torch::Tensor base_positions;
  };
  struct PendingDraft {
    ForwardInput identity;
    torch::Tensor accepted_host;
    torch::Tensor retained_embeddings;
    DraftBlock block;
    StreamEventPtr ready;
    std::optional<ForwardInput> target_input;
    std::optional<DraftTemplate> next_template;
  };
  bool prelaunch_enabled_ = false;
  std::optional<ForwardInput> prepared_prelaunch_;
  torch::Tensor prepared_noise_;
  torch::Tensor prepared_base_positions_;
  std::optional<PendingDraft> pending_draft_;

  struct BlockSampleOutput {
    torch::Tensor token_ids;
    torch::Tensor dense_probs;
  };

  BlockSampleOutput sample_path(const DFlash2CandidateOutput& candidates,
                                const SamplingParameters& sampling_params,
                                const torch::Tensor& gumbel_noise,
                                int64_t vocab_size) const;

  ProcessGroup* sampling_process_group_ = nullptr;
};

}  // namespace xllm
