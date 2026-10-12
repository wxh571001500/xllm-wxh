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

#include <glog/logging.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>

#include "framework/sampling/draft_sampling_mode.h"
#include "runtime/speculative_worker_impl.h"

namespace xllm {

class AdaptiveSpeculativeController;
class EmbeddingCache;
class LLMWorkerImpl;
class ModelArgs;

// Builds the draft KV cache geometry, reusing the target's grouped pool counts
// and propagating its packed-C8 layout when kv_cache_dtype selects it.
// `replicated_block_pages` > 1 makes the draft pool a full replica: one logical
// block owns that many physical pages on every rank, so the draft addresses its
// cache by the logical slot index instead of the target's kv_split shard.
KVCacheShape build_speculative_draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape,
    const ModelArgs& draft_model_args,
    int64_t block_size,
    int64_t draft_world_size,
    const std::string& kv_cache_dtype,
    int64_t replicated_block_pages = 1);

// Base for draft-model speculative workers (MTP, DFlash/DSpark); Suffix has no
// draft model and derives from SpeculativeWorkerImpl directly.
class DraftModelSpecWorkerImpl : public SpeculativeWorkerImpl {
 public:
  ~DraftModelSpecWorkerImpl() override;

  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

#if defined(USE_NPU) || defined(USE_MLU)
  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;
#endif

  // Draft-model spec workers build their own draft inputs in the derived step_*
  // paths, so SpeculativeWorkerImpl's decode-input prep is a no-op here.
  ForwardInput update_input_by_last_step_output(ForwardInput& inputs) override;

 protected:
  DraftModelSpecWorkerImpl(
      const ParallelArgs& parallel_args,
      const torch::Device& device,
      const runtime::Options& options,
      const runtime::Options& target_options,
      WorkerType worker_type,
      const std::function<std::unique_ptr<LLMWorkerImpl>()>& draft_factory);

  virtual KVCacheShape draft_kv_cache_shape(
      const KVCacheShape& target_kv_cache_shape) const;
  // Reuse grouped target pool counts; otherwise build the draft shape from the
  // draft's own ModelArgs (heads/dims), borrowing only the target's block count
  // so it never inherits an MLA target's compressed [1, kv_lora_rank] slot
  // shape. draft_world_size <= 0 -> draft's DP-local TP.
  KVCacheShape build_draft_kv_cache_shape(
      const KVCacheShape& target_kv_cache_shape,
      int64_t draft_world_size = -1) const;

  // Per-row embedding placeholder width reserved in embedding_cache_ during
  // allocation. 0 (default) reserves none; MTP-family drafts override to stash
  // their pre-head hidden state.
  virtual int64_t get_embedding_placeholder_size() const { return 0; }

  // Some drafts replay every verified KV position, so retain the full
  // accepted span when allocating their embedding cache.
  virtual bool requires_full_target_replay() const { return false; }

  // Physical pages one logical block owns in the draft pool. 1 (default) keeps
  // the draft on the target's kv_split sharding; > 1 replicates the whole
  // context on every rank, which is what a standalone drafter needs.
  virtual int64_t draft_replicated_block_pages() const { return 1; }

  // Tokens covered by one column of the TARGET's block table. The table holds
  // logical blocks, so under kv_split one column spans block_size *
  // kv_split_size tokens. Every conversion between a KV length and a column
  // count against that table has to use this value; the draft's own table is
  // expanded to physical pages instead (DFlashWorkerImpl::
  // expand_draft_block_tables). SuffixWorkerImpl has no draft and derives from
  // SpeculativeWorkerImpl directly, so it keeps the physical block size.
  int32_t logical_block_size() const override {
    return static_cast<int32_t>(options_.block_size()) *
           parallel_args_.kv_split_size_effective();
  }

  void prepare_hierarchy_kv_cache_transfers();
  void finalize_hierarchy_kv_cache_transfers();
  void init_embedding_cache(int64_t num_blocks);
  // Allocate one pool when its worker is LOADED; assert READY (already cached)
  // otherwise. `make_shape` is evaluated only on the LOADED path, so the draft
  // shape is not built when the draft is already cached. `allocate` runs the
  // worker-specific allocation call.
  template <typename MakeShape, typename Allocate>
  bool allocate_pool_if_loaded(WorkerImpl& worker,
                               MakeShape&& make_shape,
                               Allocate&& allocate) {
    const WorkerImpl::Status status = worker.get_status();
    if (status == WorkerImpl::Status::LOADED) {
      return allocate(worker, make_shape());
    }
    CHECK_EQ(status, WorkerImpl::Status::READY);
    return true;
  }
  using AllocateFn = std::function<bool(WorkerImpl&, const KVCacheShape&)>;
  // Allocates both KV pools via `allocate`, then unconditionally (re)builds the
  // draft embedding cache. The hierarchy transfer is finalized only when both
  // pools succeed.
  bool allocate_pools(const KVCacheShape& kv_cache_shape,
                      const AllocateFn& allocate);

  // Target-side cache budget after reserving storage for the colocated draft.
  // DeepSeek-V4's fixed SWA pools require both geometries to participate.
  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity_with_draft(
      const runtime::Options& target_options,
      const runtime::Options& draft_options);

  static void force_greedy_draft_sampling(SamplingParameters& sampling_params);

  // Overwrite dp_global_token_nums / raw_dp_global_token_nums with the true
  // post-pruning validate token count of every DP peer, gathered over the DP
  // group. Adaptive pruning makes each rank's validate token count
  // data-dependent, so the engine-supplied global vector (which assumes a
  // uniform per-seq width) no longer matches; DpEpPadding needs the real
  // per-rank counts to compute matching MoE all-to-all pads. No-op when the DP
  // group spans a single rank. MUST be called on every DP rank each validate
  // step (both the pruned and the unpruned branch) so the collective stays in
  // lockstep and does not deadlock.
  void sync_dp_global_token_nums_after_prune(ModelInputParams& input_params,
                                             int32_t local_total_val_tokens);

  // Idle-rank counterpart: a DP rank whose shard is empty still runs the target
  // validate forward (fake input) while busy peers run the pruned forward.
  // Both must join the same DP allgather. This variant contributes the idle
  // rank's own current dp_global_token_nums entry (already scaled to the
  // uniform validate width) so it stays symmetric with the busy peers.
  void sync_dp_global_token_nums_for_idle_rank(ModelInputParams& input_params);

  std::unique_ptr<LLMWorkerImpl> draft_impl_;
  std::shared_ptr<EmbeddingCache> embedding_cache_;
  std::unique_ptr<AdaptiveSpeculativeController> adaptive_spec_controller_;

  DraftSamplingMode draft_sampling_mode_ = DraftSamplingMode::GREEDY;
};

}  // namespace xllm
