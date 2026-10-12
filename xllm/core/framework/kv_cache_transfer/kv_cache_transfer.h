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

#include <folly/futures/Future.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/types.h"
#include "core/framework/kv_cache_transfer/kv_transfer_types.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#if defined(USE_NPU)
#include "platform/npu/npu_layer_synchronizer.h"
#endif
#if defined(USE_MLU)
#include "platform/mlu/mlu_layer_synchronizer.h"
#endif
#if defined(USE_DCU)
#include "platform/dcu/dcu_layer_synchronizer.h"
#endif
#include "framework/parallel_state/parallel_args.h"
#include "platform/device.h"
#include "util/threadpool.h"

namespace xllm {

#if defined(USE_NPU)
using KVPushSynchronizerImpl = NPULayerSynchronizerImpl;
#elif defined(USE_MLU)
using KVPushSynchronizerImpl = MLULayerSynchronizerImpl;
#elif defined(USE_DCU)
using KVPushSynchronizerImpl = DCULayerSynchronizerImpl;
#endif

// How a PUSH may map source logical blocks onto the destination's declared
// block table, given the two sides' DECLARED KV split widths (the source's
// kv_split_size_effective() and the destination's registered
// InstanceInfo.kv_split_size).
//
// Only two shapes are expressible; anything else must be refused before any
// bytes are mapped, because the source cannot derive the destination's block
// geometry from a width the destination did not declare:
//   - RANK_LOCAL (src == dst): both sides shard KV identically, so each source
//     rank's block k maps one-to-one onto the destination's block k.
//   - STRIDED (dst == 1 < src): the destination is unsplit and holds src times
//     as many blocks as one source logical block covers, so the source has to
//     interleave (remote_stride = src) and filter_kv_split_infos hands each
//     rank its own slice.
//   - UNRECONCILABLE: no supported shape -- `reason` names both widths when it
//     is not nullptr. A width below 1 means that side never declared one (0 is
//     proto3's absent value on the registration record): that is refused rather
//     than read as 1, because an undeclared peer is the one case where the
//     mapping cannot be validated at all. A source that declares the SHARDED
//     CP index write mode is equally refused against an equal-width
//     kv_split_size > 1 destination: only page 0 of each INDEX resource is
//     valid on any rank, so the 1:1 page moves of a RANK_LOCAL plan would
//     deliver stale peer pages.
enum class KvSplitWidthPlan : int8_t {
  RANK_LOCAL = 0,
  STRIDED = 1,
  UNRECONCILABLE = 2,
};

// The DECLARED KV split widths and CP index write modes of both sides are the
// plan's inputs; see declared_cp_index_write_mode() for how the local mode is
// resolved and InstanceInfo.cp_index_write_mode for the peer's. Both mode
// arguments use the kCpIndexWriteMode* values (common/types.h): only
// kCpIndexWriteModeSharded changes admission -- on EITHER side of an
// equal-width kv_split_size > 1 pair -- and kCpIndexWriteModeUnspecified (an
// instance that predates the field) keeps the legacy behavior.
KvSplitWidthPlan plan_kv_split_widths(int32_t src_kv_split_size,
                                      int32_t dst_kv_split_size,
                                      int32_t src_cp_index_write_mode,
                                      int32_t dst_cp_index_write_mode,
                                      std::string* reason);

// This process's declared CP index write mode, read from
// XLLM_CP_INDEX_WRITE_MODE with the same semantics as the python-side
// cp_index_write_mode() (xllm/python/model_executor/cp_utils.py): only an
// explicit "sharded" (after trim + lowercase) selects the sharded mode;
// unset, empty, or unrecognized values keep the default replicated mode.
// The registration record (register_instance_info) and the push-side width
// validation both call this, so the process declares and enforces one mode.
int32_t declared_cp_index_write_mode();

// Stable operator-facing name of a kCpIndexWriteMode* value ("unspecified",
// "replicated", "sharded"); used by registration logs and rejection reasons.
const char* cp_index_write_mode_label(int32_t cp_index_write_mode);

// In KV-split mode, filters and remaps each block-scoped cache mapping's
// remote_ids so that every KV-split rank sees only the destination blocks
// assigned to it. This includes ordinary KV and grouped SWA/C4/C128 caches.
// When `kv_split_size == 1` the caller should skip this entirely (every rank
// holds the full KV replica and remote_ids is 1:1 with local_ids).
//
// Note: prior to the KV-split / CP decoupling refactor this was named
// filter_cp_kv_infos and gated on cp_size>1. The behavior is identical when
// kv_split_size == cp_size (the legacy default), so callers that pass cp_rank
// / cp_size keep working byte-for-byte.
std::vector<TransferKVInfo> filter_kv_split_infos(
    int32_t kv_split_rank,
    int32_t kv_split_size,
    const std::vector<TransferKVInfo>& kv_infos);

inline void append_unique_request_id(
    std::unordered_map<std::string, std::unordered_set<std::string>>& seen,
    const std::string& key,
    std::vector<std::string>& request_ids,
    const std::string& request_id) {
  auto& seen_ids = seen[key];
  if (seen_ids.empty() && !request_ids.empty()) {
    seen_ids.insert(request_ids.begin(), request_ids.end());
  }
  if (seen_ids.emplace(request_id).second) {
    request_ids.emplace_back(request_id);
  }
}

class KVCacheTransfer {
 public:
  struct KVCacheInfo {
    uint64_t dst_cluster_id;
    std::string dst_addr;
    std::vector<KVTransferMapping> mappings;
    std::vector<std::string> request_ids;

    // XTensor mode: destination offsets from D-node (per-layer)
    // dst_xtensor_layer_offsets[layer_id] = {k_offsets, v_offsets}
    std::vector<XTensorLayerOffsets> dst_xtensor_layer_offsets;
  };

  static std::vector<std::string> rotate_dst_rank(
      const std::vector<std::string>& keys,
      int32_t kv_split_rank);

  KVCacheTransfer() = default;
  virtual ~KVCacheTransfer() = default;

  virtual void initialize(int32_t device_id) {};

  virtual void finalize() {};

  virtual void free_kv_cache() {};

  virtual void configure_cache_layout(const ParallelArgs& parallel_args,
                                      const ModelArgs& model_args,
                                      int32_t block_token_capacity,
                                      bool is_spec_draft) {}

  virtual void register_kv_cache(std::vector<xllm::KVCache>& kv_caches,
                                 const KVCacheShape& kv_cache_shape,
                                 const torch::ScalarType dtype) {};

  virtual void register_kv_cache_spec(std::vector<xllm::KVCache>& kv_caches,
                                      const KVCacheShape& kv_cache_shape,
                                      const torch::ScalarType dtype) {
    NOT_IMPLEMENTED();
  };

  virtual void get_cache_info(uint64_t& cluster_id, std::string& addr) = 0;

  virtual bool link_clusters(const std::vector<uint64_t>& cluster_ids,
                             const std::vector<std::string>& remote_addrs,
                             const std::vector<uint16_t>& ports) = 0;

  virtual bool unlink_cluster(const uint64_t& cluster_id,
                              const std::string& remote_addr,
                              const uint16_t port,
                              bool force_flag = true) = 0;

  virtual bool pull_kv_blocks(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<KVTransferMapping>& mappings) = 0;

  virtual folly::SemiFuture<bool> pull_kv_blocks_async(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<KVTransferMapping>& mappings);

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  virtual folly::SemiFuture<std::vector<KVTransferTaskResult>>
  push_kv_blocks_async(
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      const ParallelArgs& parallel_args,
      std::shared_ptr<KVPushSynchronizerImpl> layer_synchronizer,
      bool is_spec_draft);
#endif

  virtual void merge_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      const ParallelArgs& parallel_args);

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  virtual std::vector<KVTransferTaskResult> push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft,
      int32_t kv_split_rank,
      int32_t kv_split_size) = 0;
#endif

 protected:
  static bool validate_transfer_mappings(
      const std::vector<KVTransferMapping>& mappings,
      const std::string& request_id,
      int32_t kv_split_size,
      bool rank_local_mapping = false);

  static bool validate_transfer_mappings(
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      int32_t kv_split_size);

  // Refuses a push whose mapping decision contradicts the two sides' DECLARED
  // KV split widths: `rank_local_mapping` asserts that both sides shard KV
  // identically, so an info that claims it while the destination declared a
  // different width must not map bytes.
  //
  // `src_kv_split_size` must be the SOURCE's effective width. This is separate
  // from validate_transfer_mappings() on purpose: the post-filter call there
  // passes the literal 1 to mean "each rank's mapping is 1:1 now", which is not
  // a statement about the source's declared width, and must never be read as
  // one.
  static bool validate_kv_split_width_plan(
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      int32_t src_kv_split_size);

  // working thread
  ThreadPool threadpool_{/*num_threads=*/1,
                         /*cpu_binding=*/false,
                         /*pool_name=*/"KVCacheTransfer.async"};
};

class KVCacheTransferFactory {
 public:
  static std::shared_ptr<KVCacheTransfer> create(
      uint16_t transfer_listen_port,
      const Device& device,
      const std::string& model_type = "",
      const std::string& model_id = "");
};

}  // namespace xllm
