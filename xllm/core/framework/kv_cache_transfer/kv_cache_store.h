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
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/types.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache_transfer/kv_transfer_types.h"
#include "framework/kv_cache_transfer/mooncake_store_backend.h"
#include "framework/model/model_input_params.h"
#include "util/slice.h"

namespace xllm {

struct HostCacheStoreEntry {
  uint32_t cache_handle = 0;
  std::string key_component;
  KVCache* cache = nullptr;
};

enum class StoreTpLayout : uint8_t {
  TP_REPLICATED,
  TP_SHARDED,
};

using HostCacheStoreIndex =
    std::map<BlockType, std::vector<HostCacheStoreEntry>>;

struct KVCacheStoreInitConfig {
  std::string localhost_name = "127.0.0.1";
  std::string protocol = "tcp";
  std::string rdma_devices;
  std::string metadata_server;
  std::string master_server_address;
  std::string model_id;
  int32_t replica_num = 1;
  uint32_t tp_rank = 0;
  uint32_t tp_size = 1;
  int32_t cp_rank = 0;
  int32_t cp_size = 1;
  int32_t kv_split_full_domain_size = 1;
  int32_t kv_split_size = 1;
  int32_t kv_split_rank = 0;
  // The process's declared CP index write mode (XLLM_CP_INDEX_WRITE_MODE,
  // resolved by declared_cp_index_write_mode()). Part of the object-key
  // schema: INDEX objects written under different modes are not
  // interchangeable even though their tensor shapes match. Defaults to the
  // switch's own default so a construction site that forgets to wire it
  // keys the same space as the majority (replicated) writers.
  int32_t cp_index_write_mode = kCpIndexWriteModeReplicated;
  bool enable_mla = false;
};

class KVCacheStore final {
 public:
  KVCacheStore() = default;
  ~KVCacheStore();

  bool init(const KVCacheStoreInitConfig& config,
            HostCacheStoreIndex store_index);

  uint32_t batch_put(
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    return batch_put(slice);
  }

  uint32_t batch_get(
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    return batch_get(slice);
  }

  uint32_t batch_put(Slice<BlockTransferInfo>& block_transfer_info);
  uint32_t batch_get(Slice<BlockTransferInfo>& block_transfer_info);
  // When stats is non-null, the replica tier of every object is queried
  // synchronously before the Get and successful reads are accumulated into
  // stats. The query is an extra Mooncake master RPC on the prefetch path, so
  // callers measuring end-to-end latency include it; its cost is reported
  // separately in stats->tier_query_us.
  std::vector<uint8_t> batch_get_with_status(
      Slice<BlockTransferInfo>& block_transfer_info,
      StoreGetStats* stats = nullptr);
  // Metadata-only check whether every object of each logical block is present.
  std::vector<uint8_t> batch_exist(
      Slice<BlockTransferInfo>& block_transfer_info);

 private:
  friend class KVCacheStoreTestPeer;

  struct StoreEntry {
    uint32_t cache_handle = 0;
    std::string key_component;
    std::string schema_hash;
    KVCache* cache = nullptr;
    BlockType block_type = BlockType::KV;
    std::string key_prefix;
    std::vector<torch::Tensor> block_tensors;
    bool is_put_owner = true;
  };

  struct PhysicalRequest {
    size_t logical_index = 0;
    const StoreEntry* entry = nullptr;
    std::string key;
  };

  struct RequestGroup {
    std::string key;
    std::vector<size_t> request_indices;
  };

  struct GroupedRequests {
    std::vector<PhysicalRequest> requests;
    std::vector<RequestGroup> groups;
  };

  KVCacheStore(const KVCacheStore&) = delete;
  KVCacheStore& operator=(const KVCacheStore&) = delete;

  void initialize_store_index(HostCacheStoreIndex store_index);
  StoreTpLayout resolve_tp_layout(BlockType block_type) const;
  bool is_put_owner(StoreTpLayout tp_layout) const;
  std::string build_schema_hash(BlockType block_type,
                                const KVCache& cache,
                                StoreTpLayout tp_layout) const;
  std::string build_key_prefix(const std::string& key_component,
                               BlockType block_type,
                               const std::string& schema_hash,
                               StoreTpLayout tp_layout) const;
  std::string build_key(const StoreEntry& entry,
                        const BlockTransferInfo& block_info) const;
  std::vector<PhysicalRequest> build_requests(
      Slice<BlockTransferInfo>& block_transfer_info) const;
  static std::vector<RequestGroup> group_requests(
      const std::vector<PhysicalRequest>& requests);
  GroupedRequests build_grouped_requests(
      Slice<BlockTransferInfo>& block_transfer_info) const;
  static std::vector<uint8_t> aggregate_results(
      size_t logical_count,
      const std::vector<PhysicalRequest>& requests,
      const std::vector<uint8_t>& physical_results);
  static std::optional<std::string> get_store_device_names(
      const KVCacheStoreInitConfig& config);
  std::optional<MooncakeMultiBuffer> build_multi_buffer(const StoreEntry& entry,
                                                        int32_t block_id) const;
  std::optional<std::vector<MooncakeRegisteredRange>> collect_ranges() const;
  static void record_get(MooncakeReplicaTier tier,
                         uint64_t bytes,
                         StoreGetStats* stats);
  // 1 for every object held by at least one complete replica.
  static std::vector<uint8_t> present_objects(
      const std::vector<MooncakeReplicaTier>& tiers);

 private:
  bool is_initialized_ = false;
  KVCacheStoreInitConfig config_;
  std::map<BlockType, std::vector<StoreEntry>> store_index_;
  size_t max_entries_per_type_ = 0;
  std::unique_ptr<MooncakeStoreBackend> backend_;
};

}  // namespace xllm
