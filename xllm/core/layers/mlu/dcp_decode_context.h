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

#include "framework/kv_cache/kv_shard_layout.h"
#include "layers/mlu/dcp_indexer_candidate.h"
#include "layers/mlu/dsa_topk_state.h"

namespace xllm {

class ProcessGroup;

namespace layer {

class DcpDecodeContext final {
 public:
  DcpDecodeContext(KVShardLayout layout, ProcessGroup* dcp_group);

  torch::Tensor localize_slots(const torch::Tensor& global_slots) const;
  // Projects the gathered global top-k onto this rank's local KV slots with
  // the fused Triton localizer: owned entries are de-interleaved to local
  // slots, filtered by ownership, and compacted to the row front in column
  // order; context lens are rewritten to the per-row owned counts.
  DsaTopkState localize_topk(const DsaTopkState& global_state) const;
  // Launch/finish pair around the indexer candidate merge: the launch
  // submits the cross-rank candidate all-gather without waiting, so the
  // caller can enqueue independent compute (e.g. the deferred MLA latent
  // projections) before the finish to overlap it with the communication.
  DcpIndexerGatherAsyncCtx launch_indexer_candidate_gather(
      const torch::Tensor& local_scores,
      const torch::Tensor& local_global_slots) const;
  DsaTopkState finish_indexer_candidate_merge(
      DcpIndexerGatherAsyncCtx&& gather,
      int64_t topk,
      const torch::Tensor& slot_mapping) const;
  torch::Tensor merge(const torch::Tensor& local_output,
                      const torch::Tensor& local_lse,
                      const torch::Tensor& slot_mapping,
                      bool head_sharded) const;

  // DCP group size (1 when no DCP group is configured).
  int32_t world_size() const;
  const KVShardLayout& layout() const { return layout_; }

 private:
  KVShardLayout layout_;
  ProcessGroup* dcp_group_ = nullptr;
};

}  // namespace layer
}  // namespace xllm
