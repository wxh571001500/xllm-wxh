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

#include "framework/kv_cache_transfer/kv_cache_store.h"

#include <Mooncake/mooncake-store/include/utils.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "util/hash_util.h"
#include "util/timer.h"

namespace xllm {
namespace {

void append_key_field(std::string& key, const std::string& value) {
  key.append(std::to_string(value.size()));
  key.push_back(':');
  key.append(value);
  key.push_back(':');
}

}  // namespace

bool KVCacheStore::init(const KVCacheStoreInitConfig& config,
                        HostCacheStoreIndex store_index) {
  CHECK(!is_initialized_) << "KVCacheStore is already initialized.";
  CHECK(!config.model_id.empty())
      << "KVCacheStore requires a target model identity.";
  CHECK_GT(config.tp_size, 0U);
  CHECK_LT(config.tp_rank, config.tp_size);
  CHECK_GT(config.cp_size, 0);
  CHECK_GE(config.cp_rank, 0);
  CHECK_LT(config.cp_rank, config.cp_size);
  CHECK_GE(config.kv_split_full_domain_size, config.cp_size);
  CHECK_EQ(config.kv_split_full_domain_size % config.cp_size, 0);
  const uint32_t workers_per_cp =
      static_cast<uint32_t>(config.kv_split_full_domain_size / config.cp_size);
  CHECK(config.tp_size == workers_per_cp ||
        config.tp_size ==
            static_cast<uint32_t>(config.kv_split_full_domain_size))
      << "Store TP ranks must either repeat within each CP rank or span the "
         "full DP-local domain.";
  CHECK_GE(config.kv_split_size, 1);
  CHECK_GE(config.kv_split_rank, 0);
  CHECK_LT(config.kv_split_rank, config.kv_split_size);
  CHECK(config.cp_size % config.kv_split_size == 0 ||
        config.kv_split_size == config.kv_split_full_domain_size)
      << "kv_split_size must divide cp_size or equal the full DCP domain.";
  config_ = config;
  initialize_store_index(std::move(store_index));

  const std::optional<std::string> device_names =
      get_store_device_names(config_);
  if (config_.protocol == "rdma") {
    LOG(INFO) << "Mooncake RDMA device_names: "
              << device_names.value_or("auto-discover");
  } else if (!config_.rdma_devices.empty()) {
    LOG(WARNING) << "Ignoring store_rdma_devices for Store protocol "
                 << config_.protocol << ".";
  }

  const std::optional<std::vector<MooncakeRegisteredRange>> ranges =
      collect_ranges();
  if (!ranges.has_value()) {
    LOG(ERROR) << "Failed to collect Mooncake Host tensor ranges.";
    return false;
  }
  MooncakeStoreBackendConfig backend_config;
  backend_config.localhost_name = config_.localhost_name;
  backend_config.protocol = config_.protocol;
  backend_config.rdma_devices = device_names.value_or("");
  backend_config.metadata_server = config_.metadata_server;
  backend_config.master_server_address = config_.master_server_address;
  backend_config.replica_num = config_.replica_num;
  backend_ = std::make_unique<MooncakeStoreBackend>();
  if (!backend_->init(backend_config, *ranges)) {
    backend_.reset();
    return false;
  }

  is_initialized_ = true;
  return true;
}

std::optional<std::string> KVCacheStore::get_store_device_names(
    const KVCacheStoreInitConfig& config) {
  if (config.protocol != "rdma" || config.rdma_devices.empty()) {
    return std::nullopt;
  }
  return config.rdma_devices;
}

KVCacheStore::~KVCacheStore() { backend_.reset(); }

void KVCacheStore::initialize_store_index(HostCacheStoreIndex store_index) {
  CHECK(store_index_.empty()) << "KVCacheStore index is already initialized.";
  CHECK(!store_index.empty()) << "KVCacheStore requires Host caches.";
  for (auto& [block_type, entries] : store_index) {
    CHECK(!entries.empty()) << "KVCacheStore block type has no entries.";
    std::unordered_set<uint32_t> cache_handles;
    std::unordered_set<std::string> key_components;
    std::vector<StoreEntry>& store_entries = store_index_[block_type];
    store_entries.reserve(entries.size());
    max_entries_per_type_ = std::max(max_entries_per_type_, entries.size());
    for (HostCacheStoreEntry& entry : entries) {
      CHECK(entry.cache != nullptr) << "KVCacheStore cache must not be null.";
      CHECK(!entry.key_component.empty())
          << "KVCacheStore key component must not be empty.";
      CHECK(cache_handles.emplace(entry.cache_handle).second)
          << "Duplicate KVCacheStore cache handle for BlockType "
          << static_cast<int32_t>(block_type) << ".";
      CHECK(key_components.emplace(entry.key_component).second)
          << "Duplicate KVCacheStore key component for BlockType "
          << static_cast<int32_t>(block_type) << ": " << entry.key_component;
      const StoreTpLayout tp_layout = resolve_tp_layout(block_type);
      std::string schema_hash =
          build_schema_hash(block_type, *entry.cache, tp_layout);
      std::string key_prefix = build_key_prefix(
          entry.key_component, block_type, schema_hash, tp_layout);
      const BlockTypeTensorMap tensors =
          entry.cache->get_block_type_tensors(block_type);
      std::vector<torch::Tensor> block_tensors;
      block_tensors.reserve(tensors.size());
      for (const auto& tensor_entry : tensors) {
        block_tensors.emplace_back(tensor_entry.second);
      }
      store_entries.emplace_back(StoreEntry{entry.cache_handle,
                                            std::move(entry.key_component),
                                            std::move(schema_hash),
                                            entry.cache,
                                            block_type,
                                            std::move(key_prefix),
                                            std::move(block_tensors),
                                            is_put_owner(tp_layout)});
    }
  }
}

StoreTpLayout KVCacheStore::resolve_tp_layout(BlockType block_type) const {
  if (config_.enable_mla && block_type == BlockType::KV) {
    return StoreTpLayout::TP_REPLICATED;
  }
  return StoreTpLayout::TP_SHARDED;
}

bool KVCacheStore::is_put_owner(StoreTpLayout tp_layout) const {
  // A full-domain DCP rank uniquely identifies every DP-local worker, even
  // when that domain happens to have the same size as CP.
  if (config_.kv_split_size == config_.kv_split_full_domain_size) {
    return true;
  }

  const bool tp_rank_spans_cp =
      config_.tp_size ==
      static_cast<uint32_t>(config_.kv_split_full_domain_size);
  // MLU Store TP ranks span the whole DP-local domain. TP-sharded keys are
  // therefore already unique across CP ranks and must all be published.
  if (tp_layout == StoreTpLayout::TP_SHARDED && tp_rank_spans_cp) {
    return true;
  }

  // When DCP partitions PCP, multiple CP ranks own replicas of the same split.
  const int32_t cp_replicas_per_split = config_.cp_size / config_.kv_split_size;
  const bool is_cp_replica_owner = config_.cp_rank % cp_replicas_per_split == 0;
  const uint32_t workers_per_cp = static_cast<uint32_t>(
      config_.kv_split_full_domain_size / config_.cp_size);
  // TP-sharded Store keys include tp_rank, so every shard is a distinct writer.
  // TP-replicated keys omit tp_rank; select one representative worker from each
  // owning CP split to avoid concurrent writes to the same Store key.
  const bool is_tp_replica_owner = tp_layout == StoreTpLayout::TP_SHARDED ||
                                   config_.tp_rank % workers_per_cp == 0;
  return is_cp_replica_owner && is_tp_replica_owner;
}

std::string KVCacheStore::build_schema_hash(BlockType block_type,
                                            const KVCache& cache,
                                            StoreTpLayout tp_layout) const {
  const BlockTypeTensorMap tensors = cache.get_block_type_tensors(block_type);
  CHECK(!tensors.empty()) << "Host cache has no tensors for BlockType "
                          << static_cast<int32_t>(block_type);

  std::string cache_schema = config_.enable_mla
                                 ? "parallel=mla"
                                 : "tp=" + std::to_string(config_.tp_size);
  cache_schema.append("|type=");
  cache_schema.append(std::to_string(static_cast<int32_t>(block_type)));
  // The CP index write mode decides which physical pages of an INDEX
  // resource hold valid data (replicated persists every natural row;
  // sharded keeps only page 0 of each page group), so objects written
  // under different modes are not interchangeable even though the INDEX
  // tensors' shapes are identical. Key the modes apart, or an instance
  // reading replicated would score a sharded writer's stale peer pages as
  // valid with no error signal. The constructor wires the process's
  // declared mode (see hierarchy_kv_cache_transfer) into this config.
  cache_schema.append("|cp_index_write_mode=");
  cache_schema.append(std::to_string(config_.cp_index_write_mode));
  int64_t host_blocks = -1;
  for (const auto& [role, tensor] : tensors) {
    CHECK(tensor.defined() && tensor.dim() > 0 && tensor.is_contiguous());
    if (host_blocks < 0) {
      host_blocks = tensor.size(0);
    } else {
      CHECK_EQ(host_blocks, tensor.size(0));
    }
    cache_schema.append(",role=");
    cache_schema.append(std::to_string(static_cast<int32_t>(role)));
    cache_schema.append(",dtype=");
    cache_schema.append(
        std::to_string(static_cast<int32_t>(tensor.scalar_type())));
    cache_schema.append(",shape=");
    for (int64_t dim = 1; dim < tensor.dim(); ++dim) {
      cache_schema.append(std::to_string(tensor.size(dim)));
      cache_schema.push_back('x');
    }
  }
  const XXH3Key schema_hash = hash_string(cache_schema);
  return std::string(reinterpret_cast<const char*>(schema_hash.data),
                     sizeof(schema_hash.data));
}

std::string KVCacheStore::build_key_prefix(const std::string& key_component,
                                           BlockType block_type,
                                           const std::string& schema_hash,
                                           StoreTpLayout tp_layout) const {
  std::string prefix = "xllm-kv-v3:";
  append_key_field(prefix, config_.model_id);
  append_key_field(prefix, key_component);
  if (tp_layout == StoreTpLayout::TP_REPLICATED) {
    prefix.append("mla:");
  } else {
    prefix.append(std::to_string(config_.tp_size));
    prefix.push_back(':');
    prefix.append(std::to_string(config_.tp_rank));
    prefix.push_back(':');
  }
  prefix.append(std::to_string(config_.kv_split_size));
  prefix.push_back(':');
  prefix.append(std::to_string(config_.kv_split_rank));
  prefix.push_back(':');
  prefix.append(std::to_string(static_cast<int32_t>(block_type)));
  prefix.push_back(':');
  prefix.append(schema_hash);
  return prefix;
}

std::string KVCacheStore::build_key(const StoreEntry& entry,
                                    const BlockTransferInfo& block_info) const {
  CHECK(entry.block_type == block_info.block_type);
  std::string key = entry.key_prefix;
  key.append(reinterpret_cast<const char*>(block_info.hash_key),
             XXH3_128BITS_HASH_VALUE_LEN);
  return key;
}

std::vector<KVCacheStore::PhysicalRequest> KVCacheStore::build_requests(
    Slice<BlockTransferInfo>& block_transfer_info) const {
  std::vector<PhysicalRequest> requests;
  requests.reserve(block_transfer_info.size() * max_entries_per_type_);
  for (size_t logical_index = 0; logical_index < block_transfer_info.size();
       ++logical_index) {
    const BlockTransferInfo& block_info = block_transfer_info[logical_index];
    const auto entries_it = store_index_.find(block_info.block_type);
    if (entries_it == store_index_.end()) {
      LOG(ERROR) << "KVCacheStore has no entry for BlockType "
                 << static_cast<int32_t>(block_info.block_type) << ".";
      continue;
    }
    for (const StoreEntry& entry : entries_it->second) {
      requests.emplace_back(
          PhysicalRequest{logical_index, &entry, build_key(entry, block_info)});
    }
  }
  return requests;
}

std::vector<KVCacheStore::RequestGroup> KVCacheStore::group_requests(
    const std::vector<PhysicalRequest>& requests) {
  std::vector<RequestGroup> groups;
  groups.reserve(requests.size());
  std::unordered_map<std::string, size_t> group_indices;
  group_indices.reserve(requests.size());
  for (size_t request_index = 0; request_index < requests.size();
       ++request_index) {
    const PhysicalRequest& request = requests[request_index];
    const auto [group_it, inserted] =
        group_indices.emplace(request.key, groups.size());
    if (inserted) {
      groups.emplace_back(RequestGroup{request.key, {}});
    }
    groups[group_it->second].request_indices.emplace_back(request_index);
  }
  return groups;
}

KVCacheStore::GroupedRequests KVCacheStore::build_grouped_requests(
    Slice<BlockTransferInfo>& block_transfer_info) const {
  GroupedRequests grouped;
  grouped.requests = build_requests(block_transfer_info);
  grouped.groups = group_requests(grouped.requests);
  return grouped;
}

std::vector<uint8_t> KVCacheStore::aggregate_results(
    size_t logical_count,
    const std::vector<PhysicalRequest>& requests,
    const std::vector<uint8_t>& physical_results) {
  std::vector<uint32_t> required_counts(logical_count, 0);
  std::vector<uint32_t> success_counts(logical_count, 0);
  for (size_t request_index = 0; request_index < requests.size();
       ++request_index) {
    const size_t logical_index = requests[request_index].logical_index;
    CHECK_LT(logical_index, logical_count);
    ++required_counts[logical_index];
    if (request_index < physical_results.size() &&
        physical_results[request_index] != 0) {
      ++success_counts[logical_index];
    }
  }

  std::vector<uint8_t> logical_results(logical_count, /*value=*/0);
  for (size_t logical_index = 0; logical_index < logical_count;
       ++logical_index) {
    if (required_counts[logical_index] > 0 &&
        required_counts[logical_index] == success_counts[logical_index]) {
      logical_results[logical_index] = 1;
    }
  }
  return logical_results;
}

uint32_t KVCacheStore::batch_put(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (!is_initialized_ || block_transfer_info.empty()) {
    return 0;
  }
  const GroupedRequests grouped = build_grouped_requests(block_transfer_info);
  const std::vector<PhysicalRequest>& requests = grouped.requests;
  const std::vector<RequestGroup>& groups = grouped.groups;
  if (groups.empty()) {
    return 0;
  }

  std::vector<uint8_t> physical_results(requests.size(), /*value=*/0);
  std::vector<std::string> put_keys;
  std::vector<MooncakeMultiBuffer> put_buffers;
  std::vector<size_t> put_group_indices;
  put_keys.reserve(groups.size());
  put_buffers.reserve(groups.size());
  put_group_indices.reserve(groups.size());
  for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
    const RequestGroup& group = groups[group_index];
    const PhysicalRequest& request = requests[group.request_indices.front()];
    if (!request.entry->is_put_owner) {
      for (size_t request_index : group.request_indices) {
        physical_results[request_index] = 1;
      }
      continue;
    }
    const std::optional<MooncakeMultiBuffer> buffer = build_multi_buffer(
        *request.entry,
        block_transfer_info[request.logical_index].dst_block_id);
    if (!buffer.has_value()) {
      continue;
    }
    put_keys.emplace_back(group.key);
    put_buffers.emplace_back(*buffer);
    put_group_indices.emplace_back(group_index);
  }

  if (!put_keys.empty() && backend_ != nullptr) {
    const std::vector<uint8_t> results =
        backend_->batch_put(put_keys, put_buffers);
    for (size_t result_index = 0; result_index < put_group_indices.size() &&
                                  result_index < results.size();
         ++result_index) {
      if (results[result_index] == 0) {
        continue;
      }
      for (size_t request_index :
           groups[put_group_indices[result_index]].request_indices) {
        physical_results[request_index] = 1;
      }
    }
  }

  const std::vector<uint8_t> logical_results =
      aggregate_results(block_transfer_info.size(), requests, physical_results);
  return static_cast<uint32_t>(std::count(
      logical_results.begin(), logical_results.end(), static_cast<uint8_t>(1)));
}

uint32_t KVCacheStore::batch_get(
    Slice<BlockTransferInfo>& block_transfer_info) {
  const std::vector<uint8_t> statuses =
      batch_get_with_status(block_transfer_info);
  return static_cast<uint32_t>(
      std::count(statuses.begin(), statuses.end(), static_cast<uint8_t>(1)));
}

std::vector<uint8_t> KVCacheStore::batch_get_with_status(
    Slice<BlockTransferInfo>& block_transfer_info,
    StoreGetStats* stats) {
  std::vector<uint8_t> statuses(block_transfer_info.size(), /*value=*/0);
  if (!is_initialized_ || block_transfer_info.empty()) {
    return statuses;
  }

  const std::vector<PhysicalRequest> requests =
      build_requests(block_transfer_info);
  if (requests.empty()) {
    return statuses;
  }

  std::unordered_set<std::string> unique_keys;
  unique_keys.reserve(requests.size());
  for (const PhysicalRequest& request : requests) {
    if (!unique_keys.emplace(request.key).second) {
      LOG(ERROR) << "Duplicate KVCacheStore BatchGet key in one request.";
      return statuses;
    }
  }

  std::vector<MooncakeReplicaTier> tiers;
  if (stats != nullptr && backend_ != nullptr) {
    std::vector<std::string> keys;
    keys.reserve(requests.size());
    for (const PhysicalRequest& request : requests) {
      keys.emplace_back(request.key);
    }
    const Timer tier_timer;
    tiers = backend_->batch_query_tiers(keys);
    stats->tier_query_us +=
        static_cast<uint64_t>(tier_timer.elapsed_microseconds());
  }

  std::vector<uint8_t> physical_results(requests.size(), /*value=*/0);
  for (size_t request_index = 0; request_index < requests.size();
       ++request_index) {
    const PhysicalRequest& request = requests[request_index];
    const std::optional<MooncakeMultiBuffer> buffer = build_multi_buffer(
        *request.entry,
        block_transfer_info[request.logical_index].dst_block_id);
    if (!buffer.has_value() || backend_ == nullptr) {
      continue;
    }
    physical_results[request_index] =
        backend_->get(request.key, *buffer) ? 1 : 0;
    if (stats == nullptr || physical_results[request_index] == 0) {
      continue;
    }
    const uint64_t bytes = std::accumulate(
        buffer->sizes.begin(), buffer->sizes.end(), static_cast<uint64_t>(0));
    const MooncakeReplicaTier tier = request_index < tiers.size()
                                         ? tiers[request_index]
                                         : MooncakeReplicaTier::MISSING;
    record_get(tier, bytes, stats);
  }
  return aggregate_results(
      block_transfer_info.size(), requests, physical_results);
}

std::vector<uint8_t> KVCacheStore::batch_exist(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (!is_initialized_ || block_transfer_info.empty() || backend_ == nullptr) {
    return std::vector<uint8_t>(block_transfer_info.size(), /*value=*/0);
  }
  const std::vector<PhysicalRequest> requests =
      build_requests(block_transfer_info);
  std::vector<std::string> keys;
  keys.reserve(requests.size());
  for (const PhysicalRequest& request : requests) {
    keys.emplace_back(request.key);
  }
  const std::vector<MooncakeReplicaTier> tiers =
      backend_->batch_query_tiers(keys);
  CHECK_EQ(tiers.size(), requests.size());
  return aggregate_results(
      block_transfer_info.size(), requests, present_objects(tiers));
}

std::vector<uint8_t> KVCacheStore::present_objects(
    const std::vector<MooncakeReplicaTier>& tiers) {
  std::vector<uint8_t> present;
  present.reserve(tiers.size());
  for (const MooncakeReplicaTier tier : tiers) {
    present.emplace_back(tier != MooncakeReplicaTier::MISSING ? 1 : 0);
  }
  return present;
}

void KVCacheStore::record_get(MooncakeReplicaTier tier,
                              uint64_t bytes,
                              StoreGetStats* stats) {
  CHECK(stats != nullptr);
  stats->read_bytes += bytes;
  switch (tier) {
    case MooncakeReplicaTier::MEMORY:
      ++stats->memory_objects;
      stats->memory_bytes += bytes;
      break;
    case MooncakeReplicaTier::DISK:
      ++stats->disk_objects;
      stats->disk_bytes += bytes;
      break;
    case MooncakeReplicaTier::MISSING:
      break;
  }
}

std::optional<MooncakeMultiBuffer> KVCacheStore::build_multi_buffer(
    const StoreEntry& entry,
    int32_t block_id) const {
  if (entry.block_tensors.empty()) {
    LOG(ERROR) << "Missing Host cache for BlockType "
               << static_cast<int32_t>(entry.block_type);
    return std::nullopt;
  }

  MooncakeMultiBuffer buffer;
  buffer.addresses.reserve(entry.block_tensors.size());
  buffer.sizes.reserve(entry.block_tensors.size());
  for (const torch::Tensor& tensor : entry.block_tensors) {
    if (block_id < 0 || block_id >= tensor.size(0)) {
      LOG(ERROR) << "Invalid Host cache block id=" << block_id;
      return std::nullopt;
    }
    torch::Tensor block = tensor[block_id];
    if (!block.is_contiguous() || block.numel() <= 0 ||
        static_cast<uint64_t>(block.numel()) >
            std::numeric_limits<size_t>::max() /
                static_cast<size_t>(block.element_size())) {
      LOG(ERROR) << "Invalid Host cache tensor for BlockType "
                 << static_cast<int32_t>(entry.block_type);
      return std::nullopt;
    }
    buffer.addresses.emplace_back(block.data_ptr());
    buffer.sizes.emplace_back(static_cast<size_t>(block.numel()) *
                              static_cast<size_t>(block.element_size()));
  }
  return buffer;
}

std::optional<std::vector<MooncakeRegisteredRange>>
KVCacheStore::collect_ranges() const {
  std::vector<MooncakeRegisteredRange> ranges;
  for (const auto& [block_type, entries] : store_index_) {
    for (const StoreEntry& entry : entries) {
      const BlockTypeTensorMap tensors =
          entry.cache->get_block_type_tensors(block_type);
      for (const auto& tensor_entry : tensors) {
        const torch::Tensor& tensor = tensor_entry.second;
        if (!tensor.defined() || tensor.numel() <= 0 ||
            static_cast<uint64_t>(tensor.numel()) >
                std::numeric_limits<size_t>::max() /
                    static_cast<size_t>(tensor.element_size())) {
          LOG(ERROR) << "Invalid Host tensor registration range.";
          return std::nullopt;
        }
        ranges.emplace_back(MooncakeRegisteredRange{
            tensor.data_ptr(),
            static_cast<size_t>(tensor.numel()) *
                static_cast<size_t>(tensor.element_size())});
      }
    }
  }
  return ranges;
}

}  // namespace xllm
