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

#include <glog/logging.h>

#include <algorithm>
#include <cstring>
#include <string>

#include "core/common/types.h"
#include "core/framework/kv_cache_transfer/kv_transfer_completion.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/platform/device.h"

namespace xllm {
namespace {

uint64_t hash_transfer_request_ids(
    const std::vector<std::string>& request_ids) {
  constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  uint64_t hash = kFnvOffsetBasis;
  for (const std::string& request_id : request_ids) {
    for (const unsigned char character : request_id) {
      hash ^= character;
      hash *= kFnvPrime;
    }
    hash ^= 0xff;
    hash *= kFnvPrime;
  }
  return hash;
}

template <typename ParseRankFn>
void for_each_reduction_rank(ProcessGroup* (&reduction_groups)[2],
                             const torch::Tensor& payload,
                             const char* shape_error,
                             ParseRankFn&& parse_rank) {
  ProcessGroup* previous_group = nullptr;
  for (ProcessGroup* group : reduction_groups) {
    if (group == nullptr || group == previous_group ||
        group->world_size() <= 1) {
      continue;
    }
    torch::Tensor gathered =
        group->allgather_base_sync(payload).to(torch::kCPU);
    CHECK_EQ(gathered.numel(),
             static_cast<int64_t>(group->world_size() * payload.numel()))
        << shape_error;
    const int64_t* gathered_data = gathered.data_ptr<int64_t>();
    for (int32_t rank = 0; rank < group->world_size(); ++rank) {
      const int64_t rank_offset = rank * payload.numel();
      parse_rank(gathered_data + rank_offset);
    }
    previous_group = group;
  }
}

}  // namespace

std::vector<std::string> canonical_transfer_request_ids(
    const std::vector<TransferKVInfo>& transfer_kv_infos) {
  std::vector<std::string> request_ids =
      unique_transfer_request_ids(transfer_kv_infos);
  std::sort(request_ids.begin(), request_ids.end());
  return request_ids;
}

std::vector<std::string> reduce_failed_request_ids(
    const std::unordered_set<std::string>& local_failed_request_ids,
    const std::vector<std::string>& canonical_request_ids,
    const ParallelArgs& parallel_args,
    const Device& device) {
  // Request batches are replicated across TP/CP ranks, but differ between DP
  // replicas. Reducing on the DP group would therefore combine unrelated
  // request-ID bitmaps and can deadlock when their shapes differ. The
  // fixed-size metadata collective below also verifies that TP/CP ranks agree
  // on empty transfer sets before any rank skips the bitmap collective.
  ProcessGroup* reduction_groups[] = {parallel_args.tp_group_,
                                      parallel_args.cp_group_};
  bool needs_collective = false;
  for (ProcessGroup* group : reduction_groups) {
    if (group != nullptr && group->world_size() > 1) {
      needs_collective = true;
      break;
    }
  }
  if (!needs_collective) {
    std::vector<std::string> failed_request_ids;
    for (const std::string& request_id : canonical_request_ids) {
      if (local_failed_request_ids.find(request_id) !=
          local_failed_request_ids.end()) {
        failed_request_ids.emplace_back(request_id);
      }
    }
    return failed_request_ids;
  }

  constexpr size_t kMetadataPayloadSize = 2;
  std::vector<int64_t> metadata_values(kMetadataPayloadSize, 0);
  metadata_values[0] = static_cast<int64_t>(canonical_request_ids.size());
  const uint64_t request_ids_hash =
      hash_transfer_request_ids(canonical_request_ids);
  std::memcpy(&metadata_values[1], &request_ids_hash, sizeof(request_ids_hash));
  torch::Tensor metadata_payload =
      torch::from_blob(
          metadata_values.data(),
          {static_cast<int64_t>(metadata_values.size())},
          torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
          .clone()
          .to(device);

  // Agree on the request set before any rank branches around an empty set or
  // enters the variable-length failure-bitmap collective.
  for_each_reduction_rank(
      reduction_groups,
      metadata_payload,
      "unexpected KV transfer request metadata shape",
      [&](const int64_t* rank_data) {
        CHECK_EQ(rank_data[0], metadata_values[0])
            << "KV transfer request count differs across reduction ranks";
        CHECK_EQ(rank_data[1], metadata_values[1])
            << "KV transfer request IDs differ across reduction ranks";
      });
  if (canonical_request_ids.empty()) {
    return {};
  }

  std::vector<int64_t> payload_values(canonical_request_ids.size(), 0);
  for (size_t index = 0; index < canonical_request_ids.size(); ++index) {
    if (local_failed_request_ids.find(canonical_request_ids[index]) !=
        local_failed_request_ids.end()) {
      payload_values[index] = 1;
    }
  }
  torch::Tensor payload =
      torch::from_blob(
          payload_values.data(),
          {static_cast<int64_t>(payload_values.size())},
          torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
          .clone()
          .to(device);
  std::vector<int32_t> reduced_bitmap(canonical_request_ids.size(), 0);
  for_each_reduction_rank(
      reduction_groups,
      payload,
      "unexpected KV transfer request payload shape",
      [&](const int64_t* rank_data) {
        for (size_t index = 0; index < canonical_request_ids.size(); ++index) {
          if (rank_data[index] > 0) {
            reduced_bitmap[index] = 1;
          }
        }
      });
  std::vector<std::string> failed_request_ids;
  for (size_t index = 0; index < canonical_request_ids.size(); ++index) {
    if (reduced_bitmap[index] > 0) {
      failed_request_ids.emplace_back(canonical_request_ids[index]);
    }
  }
  return failed_request_ids;
}

std::vector<std::string> finalize_kv_push_failures(
    KVTransferCompletion& kv_transfers,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const std::string& kv_cache_transfer_mode,
    InstanceRole instance_role,
    const ParallelArgs& parallel_args,
    const Device& device) {
  const std::unordered_set<std::string> local_failed_request_ids =
      kv_transfers.wait();
  const std::vector<std::string> canonical_request_ids =
      canonical_transfer_request_ids(transfer_kv_infos);
  if (kv_cache_transfer_mode != "PUSH") {
    return {};
  }
  // These roles never send PD KV. In particular, decode's speculative body
  // may be captured in a graph, where the metadata collective's synchronous
  // host/device copies are forbidden. Use rank-uniform configuration here;
  // PUSH senders must still validate even an empty local request set.
  if (instance_role == InstanceRole::DEFAULT ||
      instance_role == InstanceRole::DECODE) {
    CHECK(canonical_request_ids.empty())
        << "non-sending instance has outgoing KV transfer requests";
    CHECK(local_failed_request_ids.empty());
    return {};
  }
  return reduce_failed_request_ids(
      local_failed_request_ids, canonical_request_ids, parallel_args, device);
}

}  // namespace xllm
