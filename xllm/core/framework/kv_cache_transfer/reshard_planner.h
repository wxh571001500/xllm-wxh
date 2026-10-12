/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/types.h"
#include "framework/kv_cache_transfer/cache_layout.h"

namespace xllm {

struct ByteRegion {
  uint64_t local_buffer_id = 0;
  uint64_t local_offset = 0;
  uint64_t remote_buffer_id = 0;
  uint64_t remote_offset = 0;
  uint64_t length = 0;
};

struct StridedRegionTemplate {
  CacheNamespace cache_namespace = CacheNamespace::MAIN;
  int64_t layer_id = 0;
  int32_t role = 0;
  int32_t group_id = 0;
  std::string logical_tensor;
  uint64_t local_buffer_id = 0;
  uint64_t remote_buffer_id = 0;
  uint64_t local_offset_in_resource = 0;
  uint64_t remote_offset_in_resource = 0;
  uint64_t bytes_per_region = 0;
  uint64_t repeat_count = 1;
  uint64_t local_stride = 0;
  uint64_t remote_stride = 0;
  uint64_t local_resource_count = 0;
  uint64_t remote_resource_count = 0;
  uint64_t local_resource_stride = 0;
  uint64_t remote_resource_stride = 0;
  uint64_t local_buffer_bytes = 0;
  uint64_t remote_buffer_bytes = 0;
  bool local_explicit_resource_offsets = false;
  bool remote_explicit_resource_offsets = false;
};

// Runtime resource binding for page-mapped buffers such as GlobalXTensor.
// IDs preserve the logical cache-resource identity used for manifest bounds;
// offsets are byte bases within the registered MoonCake buffer.
struct ExplicitResourceMapping {
  int32_t group_id = 0;
  int32_t role = 0;
  std::vector<uint64_t> local_ids;
  std::vector<uint64_t> remote_ids;
  std::vector<uint64_t> local_offsets;
  std::vector<uint64_t> remote_offsets;
};

struct ReshardPlanTemplate {
  std::string source_incarnation;
  uint64_t source_layout_generation = 0;
  std::string destination_incarnation;
  uint64_t destination_layout_generation = 0;
  ParallelCoordinates source_coordinates;
  ParallelCoordinates destination_coordinates;
  std::string destination_addr;
  std::vector<StridedRegionTemplate> regions;
};

// Whether `source` and `destination` describe the same cache instance
// layout: schema_version, fingerprint, backend and layout_family must all
// agree. validate_compatibility() gates every planner-built plan on this
// identity, and the cp1 receiver-link admission in
// mooncake_transfer_engine.cpp re-checks it because that path never
// consults the planner. When the layouts differ, `mismatched_fields` (if
// not null) receives the names of the disagreeing fields for error
// messages.
bool same_instance_layout(
    const WorkerCacheLayoutManifest& source,
    const WorkerCacheLayoutManifest& destination,
    std::vector<std::string>* mismatched_fields = nullptr);

class ReshardPlanner final {
 public:
  Status select_sources(const std::vector<WorkerCacheLayoutManifest>& sources,
                        const WorkerCacheLayoutManifest& destination,
                        std::vector<size_t>* selected_indices) const;

  Status validate_destination_coverage(
      const std::vector<WorkerCacheLayoutManifest>& sources,
      const WorkerCacheLayoutManifest& destination) const;

  // include_replicas: the link calls this per selected peer, so the source owns
  // everything it holds inside that pair, replica copies included (see
  // reshard_planner.cpp).
  //
  // source_cp_index_write_mode (a kCpIndexWriteMode* value, common/types.h)
  // selects which physical page of a source INDEX resource the plan reads
  // under a width-collapsing (kv_split_size > 1 -> 1) transfer: the replicated
  // mode (and legacy `unspecified`, which behaves as replicated) persists
  // every stripe's page at its natural row, so the plan must read page
  // `source.coordinates.kv_split_rank` -- the stripe this rank ships for the
  // strided mapping (remote block r + k * kv_split needs source page
  // k * kv_split + r, and both owners shipping page 0 would deliver the
  // first stripe's bytes twice). The sharded mode keeps each rank's stripe
  // at page 0, so the natural overlap already selects the right bytes.
  Status build_outgoing_plan(
      const WorkerCacheLayoutManifest& source,
      const WorkerCacheLayoutManifest& destination,
      ReshardPlanTemplate* plan,
      bool include_replicas = false,
      int32_t source_cp_index_write_mode = kCpIndexWriteModeReplicated) const;
};

class RequestRegionBinder final {
 public:
  Status bind(const ReshardPlanTemplate& plan,
              const std::vector<KVTransferMapping>& mappings,
              CacheNamespace cache_namespace,
              int64_t layer_id,
              std::vector<ByteRegion>* regions) const;

  Status bind_explicit(const ReshardPlanTemplate& plan,
                       const std::vector<ExplicitResourceMapping>& mappings,
                       CacheNamespace cache_namespace,
                       int64_t layer_id,
                       std::vector<ByteRegion>* regions) const;
};

}  // namespace xllm
