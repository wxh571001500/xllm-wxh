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

#include <brpc/channel.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "disagg_pd.pb.h"
#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#include "framework/request/request.h"
#include "framework/tokenizer/tokenizer.h"
#include "runtime/xservice_client.h"
#include "scheduler/continuous_scheduler.h"
#include "server/xllm_server_registry.h"
#include "util/blockingconcurrentqueue.h"
#include "util/threadpool.h"

namespace xllm {

inline constexpr int32_t kDecodeAddNewPromptTooLongStatusCode = 413;

bool is_permanent_rejection(int32_t status_code);

// Flat KV managers reserve block 0 for padding, so only num_blocks - 1 blocks
// can belong to a request. Returns true only when the prompt can never fit,
// independent of current cache pressure.
bool exceeds_decode_capacity(size_t num_prompt_tokens,
                             size_t block_size,
                             size_t num_blocks);

bool has_rank_preserving_kv_groups(const proto::DisaggResponse& response);

// The KV-split admission decision one dispatched request resolves to: the
// width plan, the TransferKVInfo mapping claim it licenses, and the rejection
// reason when no mapping is expressible.
struct KvSplitDispatchPlan {
  KvSplitWidthPlan plan = KvSplitWidthPlan::UNRECONCILABLE;
  // Whether a TransferKVInfo built from the decode instance's allocation
  // response may claim the rank-local 1:1 mapping. Exact only when a response
  // was passed to resolve_kv_split_plan().
  bool rank_local_mapping = false;
  // Human-readable rejection reason; empty unless the plan is UNRECONCILABLE.
  std::string reason;
};

// Shared KV-split width resolution for every dispatch override
// (DisaggPDScheduler::dispatch_requests and PDOOCScheduler::dispatch_requests,
// online and offline): determines how a PUSH toward `remote_info` may map
// source logical blocks onto the destination's declared block table, and
// assembles the TransferKVInfo rank-local mapping decision from the decode
// instance's allocation `response`.
//
// `src_kv_split_size` / `src_cp_index_write_mode` are the SOURCE's declared
// width and write mode (instance_info_); `remote_info` is the destination's
// registered record. `response` is nullptr until the AddNewRequests RPC has
// returned: the plan and rejection reason are then still exact, while
// rank_local_mapping stays false because the destination's group layout is
// not yet known.
//
// Both overrides MUST resolve the plan through this helper. The OOC path
// historically skipped it, so an equal-width pair (decode declaring
// kv_split_size > 1) silently fell back to strided remapping and was
// mis-addressed by kv_split_rank + k * kv_split_size, and unreconcilable
// pairs (including a sharded-write source against an equal-width
// destination) were never rejected.
KvSplitDispatchPlan resolve_kv_split_plan(
    int32_t src_kv_split_size,
    int32_t src_cp_index_write_mode,
    const InstanceInfo& remote_info,
    const proto::DisaggResponse* response);

class DisaggPDScheduler : public ContinuousScheduler {
 public:
  DisaggPDScheduler(Engine* engine, const Options& options);

  ~DisaggPDScheduler() override;

  uint32_t get_waiting_requests_num() const override {
    return prefill_queue_->size() +
           prefetching_requests_.load(std::memory_order_relaxed);
  };

  void step(const absl::Duration& timeout) override;

  // prefill-1: for prefill send new request to decode
  virtual void dispatch_requests();
  // prefill-2: for prefill send first token to decode
  virtual void prefill_send_first_generation();

  // decode-1: for decode recveive new request from prefill
  virtual bool decode_schedule(std::shared_ptr<Request>& request,
                               const std::string& prefill_instance_name);
  // decode-2: for decode receive first token from prefill
  virtual bool decode_recv_first_generation(
      const std::string& req_id,
      int64_t token_id,
      bool has_logprob,
      float logprob,
      double time_to_first_token_latency_seconds,
      double upstream_elapsed_seconds,
      std::vector<int64_t> top_tokens,
      std::vector<float> top_logprobs,
      const std::string& kv_cache_transfer_mode,
      std::vector<uint64_t> src_cluster_ids,
      std::vector<std::string> src_addrs,
      std::vector<KVTransferMapping> source_mappings,
      int32_t src_dp_size,
      int32_t src_dp_rank,
      torch::Tensor mtp_bootstrap_embedding = torch::Tensor(),
      int32_t num_cached_tokens = 0);

  // Only consumes a matching waiting reservation, never a running request.
  bool release_reservation(const std::string& req_id,
                           const std::string& reservation_id);

  // decode allocate blocks with prefix cache.
  bool try_allocate(Sequence* sequence);

  // Classifies a failed allocation as permanently oversized.
  // DSV4 multi-manager and XTensor layouts conservatively return false because
  // their effective token capacity cannot be derived from the flat KV count.
  bool exceeds_decode_capacity(Sequence* sequence) const;

  bool enable_schedule_overlap() { return options_.enable_schedule_overlap(); };

  void get_latency_metrics(std::vector<int64_t>& ttft,
                           std::vector<int64_t>& tbt) override;

  bool link_instance(const std::string& instance_name,
                     const std::vector<uint64_t>& cluster_ids,
                     const std::vector<std::string>& addrs,
                     const std::vector<uint16_t>& ports,
                     const int32_t dp_size,
                     const int32_t src_kv_split_size);

  bool unlink_instance(const std::string& instance_name,
                       const std::vector<uint64_t>& cluster_ids,
                       const std::vector<std::string>& addrs,
                       const std::vector<uint16_t>& ports,
                       const int32_t dp_size,
                       const int32_t src_kv_split_size);

 protected:
  void release_failed_request(const std::shared_ptr<Request>& request) override;

  // Fail a request whose kv_split pair resolve_kv_split_plan marked
  // UNRECONCILABLE. Shared by every dispatch path (the online dispatch
  // here, PDOOCScheduler's online and offline dispatches) so the width
  // gate's failure handling cannot drift between the non-OOC and OOC
  // paths -- the OOC path historically missed the width gate for exactly
  // that reason. release_prefill_blocks additionally frees the
  // prefill-side blocks the offline path has already allocated.
  void reject_unreconcilable_kv_split(const std::shared_ptr<Request>& request,
                                      const std::string& instance_name,
                                      const std::string& reason,
                                      bool release_prefill_blocks);

  // Caller-safe snapshot of a destination's registered record, taken under
  // instance_channel_map_mutex_: check_remote_instance_info inserts into
  // remote_instances_info_ on the RPC thread while the OOC scheduler runs
  // its online and offline dispatches on two concurrent threads, so an
  // unlocked operator[] read races the map's rehash -- and for a missing
  // key operator[] would INSERT a default record (kv_split_size = 1,
  // empty name), silently flipping the width gate's failure direction to
  // "admit". Dispatch paths take one snapshot per request before the
  // width gate and carry it through the response handling, so the push
  // defense sees exactly the record the gate validated. std::nullopt when
  // the instance has no record (unlinked, or the cached channel predates
  // registration) -- callers must fail closed on that.
  std::optional<InstanceInfo> locked_remote_instance_info(
      const std::string& instance_name);

  // The full KV-split admission sequence every dispatch path runs before
  // touching a destination: snapshot the peer's registered record under the
  // registry lock, fail closed when the record is missing (the request is
  // already failed through process_failed_request), resolve the width plan
  // through resolve_kv_split_plan, and reject through
  // reject_unreconcilable_kv_split when the pair is UNRECONCILABLE. Returns
  // the snapshot on success. The online dispatch here, PDOOCScheduler's
  // online dispatch, and its offline dispatch MUST route through this helper
  // -- the sequence was previously copied verbatim per path, and the next
  // check added to one copy would silently miss the other two.
  // release_prefill_blocks frees the prefill-side blocks the offline path
  // has already allocated when it rejects.
  std::optional<InstanceInfo> admit_kv_split_request(
      const std::shared_ptr<Request>& request,
      const std::string& instance_name,
      bool release_prefill_blocks);

  // Caller holds received_request_map_mutex_. Deallocation stays outside it.
  std::shared_ptr<Request> take_waiting_request(const std::string& req_id);

  void do_permanent_rejection(const std::shared_ptr<Request>& request);

  void enqueue_ready_request(std::shared_ptr<Request> request) override;

  // Pre-execute prefill requests of different lengths at startup and obtain the
  // corresponding TTFT for calculating the estimated TTFT of requests.
  void profile_ttft();

  void profile_tpot();

  void cache_prefill_blocks(Request* request);

  void restore_disaggregated_latency(Request* request,
                                     double time_to_first_token_latency_seconds,
                                     double upstream_elapsed_seconds);

  // check remote instance info, if not exist, get from master service
  bool check_remote_instance_info(const std::string& instance_name);

  // create rpc channel to remote instance,
  // we can get remote instance info from master service.
  proto::DisaggPDService_Stub* create_rpc_channel(
      const std::string& instance_name);

  virtual void start_rpc_server();

  // Initialize RPC server and xservice client
  // This method waits for the RPC server to be initialized and sets up the
  // xservice client connection.
  void initialize_rpc_server(const std::string& server_name);

  // Register instance information including name, RPC address, type, and cache
  // info
  void register_instance_info(const std::string& server_name, Engine* engine);

  void update_token_latency_metrics(std::vector<Sequence*>& sequences) override;

  // remote instance name(ID) -> instance info
  std::unordered_map<std::string, InstanceInfo> remote_instances_info_;

  // rpc server for prefill/decode instance
  std::unique_ptr<std::thread> rpc_server_thread_;

  // request_id -> brpc channel
  // brpc channel is connected to remote instance rpc server
  std::unordered_map<std::string, proto::DisaggPDService_Stub*>
      req_to_channel_map_;
  std::unordered_map<std::string, proto::DisaggPDService_Stub*>
      instance_channel_map_;
  std::mutex req_to_channel_map_mutex_;
  std::mutex instance_channel_map_mutex_;

  // for prefill, dispatch request to Decode instance
  std::unique_ptr<std::thread> dispatch_thread_;

  moodycamel::BlockingConcurrentQueue<std::shared_ptr<Request>>
      prefill_request_queue_;
  moodycamel::BlockingConcurrentQueue<std::shared_ptr<Request>>
      prefill_request_queue_offline_;

  // use threadpool to handle prefill-completed request
  ThreadPool prefill_threadpool_{/*num_threads=*/1,
                                 /*cpu_binding=*/false,
                                 /*pool_name=*/"DisaggPDScheduler.prefill"};

  // Release retries must not delay FirstGeneration or local KV reclamation.
  ThreadPool reservation_release_threadpool_{
      /*num_threads=*/1,
      /*cpu_binding=*/false,
      /*pool_name=*/"DisaggPDScheduler.reservation_release"};

  // related decode instance name(ID) list (used by PDOOCScheduler override)
  std::vector<std::string> decode_inst_names_;
  // TODO later
  // std::vector<std::string> updated_decode_inst_names;
  int current_decode_idx_ = 0;

  // for decode
  // request_id -> Request object
  std::unordered_map<std::string, std::shared_ptr<Request>>
      received_request_map_;
  // prefill_instance_name -> set of request_ids.
  // Used for bulk cleanup when a prefill instance is unlinked.
  std::unordered_map<std::string, std::unordered_set<std::string>>
      instance_to_received_requests_map_;
  // request_id -> prefill_instance_name.
  // Used to efficiently remove a request from
  // instance_to_received_requests_map_ when the request is processed.
  std::unordered_map<std::string, std::string> request_to_instance_map_;
  std::mutex received_request_map_mutex_;

  // Lock for multi-threaded read-write latency metrics
  std::vector<int64_t> recent_ttft_;
  std::vector<int64_t> recent_tbt_;
  std::mutex latency_metrics_mutex_;

  // Lock for multi-threaded read-write linked instances
  std::mutex linked_instances_mutex_;
  std::unordered_set<std::string> linked_instance_;

  std::string server_name_;
};

}  // namespace xllm
