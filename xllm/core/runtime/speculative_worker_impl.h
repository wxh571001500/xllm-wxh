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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/macros.h"
#include "framework/sampling/rejection_sampler.h"
#include "runtime/options.h"
#include "runtime/worker_impl.h"

namespace xllm {

// Returns whether this rank may execute the multi-step speculative decode
// plan for the current global DP batch.
bool should_run_speculative_decode(const ModelInputParams& params);

// Keep padded and raw DP token-count views in the same speculative layout.
void scale_speculative_parallel_token_counts(ModelInputParams& params,
                                             int32_t multiplier);

struct SpeculativeOutputStats {
  std::vector<int64_t> accepted_per_position;
  std::vector<SpeculativeTokenStats> sequence_stats;
  int64_t committed_tokens = 0;
};

SpeculativeOutputStats calculate_speculative_output_stats(
    const torch::Tensor& tokens,
    int64_t num_speculative_tokens);

std::vector<SpeculativeTokenStats> calculate_mtp_speculative_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens);

std::vector<SpeculativeTokenStats> calculate_block_speculative_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens);

// Base class for all speculative decoding workers.
// Provides common logic: target model management, step dispatch, and
// sampling parameter updates. Subclasses implement algorithm-specific
// draft generation and validation (MTP, Eagle3, Suffix, DFlash, etc.).
class SpeculativeWorkerImpl : public WorkerImpl {
 public:
  ~SpeculativeWorkerImpl() override;

 protected:
  // `options` is passed to WorkerImpl (preserves enable_schedule_overlap etc.),
  // `target_options` is used to create impl_ (target model worker).
  // Each algorithm subclass decides its own target_options.
  SpeculativeWorkerImpl(const ParallelArgs& parallel_args,
                        const torch::Device& device,
                        const runtime::Options& options,
                        const runtime::Options& target_options,
                        WorkerType worker_type = WorkerType::LLM);

 public:
  // initialize model, cache manager. blocking call
  bool init_model(ModelContext& context) override {
    // do nothing
    return true;
  };

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<uint16_t>& ports) override {
    return impl_->link_cluster(cluster_ids, addrs, ports);
  };

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<uint16_t>& ports) override {
    return impl_->unlink_cluster(cluster_ids, addrs, ports);
  };

  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override {
    return impl_->estimate_kv_cache_capacity();
  };

  // allocate kv cache. blocking call
  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

#if defined(USE_NPU)
  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;
#endif

  void get_cache_info(uint64_t& cluster_id,
                      std::string& addr,
                      uint16_t& port) override {
    impl_->get_cache_info(cluster_id, addr, port);
  };

  // prepare input for execution
  ForwardInput prepare_inputs(Batch& batch) override {
    return impl_->prepare_inputs(batch);
  };

  void restore_json_object_states(ForwardInput& input) override;

  // Common step dispatch: prefill / decode / empty
  std::optional<ForwardOutput> step(const ForwardInput& input) override;

  ForwardInput update_input_by_last_step_output(ForwardInput& inputs) override;

  folly::SemiFuture<bool> pull_kv_blocks_async(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<KVTransferMapping>& mappings) override {
    return impl_->pull_kv_blocks_async(src_cluster_id, src_addr, mappings);
  };

 protected:
  // Algorithm-specific virtual methods for subclasses to implement
  virtual std::optional<ForwardOutput> step_prefill(
      const ForwardInput& input) = 0;
  virtual std::optional<ForwardOutput> step_decode(
      const ForwardInput& inputs) = 0;
  virtual std::optional<ForwardOutput> step_empty(
      const ForwardInput& inputs) = 0;

  // Common helper: update sampling params for validation
  void update_sampling_params(SamplingParameters& sampling_params,
                              const int32_t num_val_tokens,
                              const int32_t total_num_val_tokens,
                              bool repeat_unique_history = true);
  void update_sampling_params(SamplingParameters& sampling_params,
                              const std::vector<int32_t>& per_seq_val_tokens,
                              const int32_t total_num_val_tokens);

  // prepare inputs for target model at Decode phase (validation).
  // repeat_unique_history=false skips repeating the unique-token history
  // tensors when the caller rebuilds them itself (the VLM suffix path
  // extends them per verification position in extend_validation_history).
  void prepare_validate_inputs(const ForwardInput& inputs,
                               ForwardInput& validate_inputs,
                               bool repeat_unique_history = true);
  // Per-seq variant used by adaptive-speculative pruning: each sequence's
  // validate row width equals per_seq_val_tokens[i] (must be in [1, N+1]).
  // The dense meta/token/position/kv-slot buffers are rebuilt as varlen with
  // total_tokens = Σ per_seq_val_tokens.
  void prepare_validate_inputs(const ForwardInput& inputs,
                               ForwardInput& validate_inputs,
                               const std::vector<int32_t>& per_seq_val_tokens);

  // Tokens covered by one column of the target's block table. The table holds
  // logical blocks, so when the KV cache is split across ranks a column spans
  // block_size * kv_split_size tokens; algorithms that carry that geometry
  // (MTP, DFlash) override this, the rest keep the physical block size.
  virtual int32_t logical_block_size() const {
    return static_cast<int32_t>(options_.block_size());
  }

 protected:
  // Target model worker
  std::unique_ptr<WorkerImpl> impl_;

  bool enable_fused_kernel_ = false;
};
}  // namespace xllm
