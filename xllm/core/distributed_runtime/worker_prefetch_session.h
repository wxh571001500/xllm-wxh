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

#include <brpc/stream.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "framework/kv_cache_transfer/kv_transfer_types.h"
#include "framework/kv_cache_transfer/prefetch_result.h"
#include "util/slice.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {

struct PrefetchWorkerStats final {
  uint64_t executed_units = 0;
  uint64_t hit_units = 0;
  uint64_t requested_blocks = 0;
  uint64_t hit_blocks = 0;
  StoreGetStats get;
  uint64_t batches = 0;
  uint64_t queue_wait_us = 0;
  // Excludes get.tier_query_us; total_us includes the tier queries.
  uint64_t get_us = 0;
  uint64_t control_wait_us = 0;
  uint64_t total_us = 0;
  // Tail after this rank's contiguous gated-hit prefix, including hits read
  // after the first miss in the last batch but not usable as a prefix.
  size_t probe_begin_unit = 0;
};

struct PrefetchSessionReport final {
  PrefetchWorkerStats stats;
  bool failed = false;
  size_t probed_units = 0;
  // Empty when probing was skipped on failure or returned an invalid size.
  std::optional<size_t> probed_present_units;
};

// External Store, stream and diagnostic boundaries. Empty stream/report
// callbacks select the brpc stream operations and the worker log sink.
struct PrefetchSessionCallbacks final {
  std::function<std::vector<uint8_t>(Slice<BlockTransferInfo>&, StoreGetStats*)>
      get;
  std::function<std::vector<uint8_t>(Slice<BlockTransferInfo>&)> probe;
  std::function<int32_t(brpc::StreamId, const butil::IOBuf&)> write;
  std::function<void(brpc::StreamId)> close;
  std::function<void(const PrefetchSessionReport&)> report;
};

class WorkerPrefetchSession final
    : public brpc::StreamInputHandler,
      public std::enable_shared_from_this<WorkerPrefetchSession> {
 public:
  WorkerPrefetchSession(ThreadPool* copy_pool,
                        ThreadPool* stats_pool,
                        StoragePrefetchRequest request,
                        size_t batch_size,
                        int64_t worker_rank,
                        PrefetchSessionCallbacks callbacks);

  void retain();
  void release();
  void start(brpc::StreamId stream_id);
  // Stops further batches and reports failure once the current Get returns.
  // The owner must drain copy_pool, then stats_pool, before releasing Store.
  void shutdown();

  int on_received_messages(brpc::StreamId id,
                           butil::IOBuf* const messages[],
                           size_t size) override;
  void on_idle_timeout(brpc::StreamId id) override;
  void on_failed(brpc::StreamId id,
                 int error_code,
                 const std::string& error_text) override;
  void on_closed(brpc::StreamId id) override;

 private:
  enum class State : uint8_t {
    CREATED = 0,
    RUNNING_BATCH = 1,
    WAITING_DECISION = 2,
    COMPLETED = 3,
    FAILED = 4,
    CLOSED = 5,
  };

  static uint64_t elapsed_us(std::chrono::steady_clock::time_point since);
  void schedule_batch();
  void run_batch();
  void finish(brpc::StreamId id, bool failed);
  void report_ready();  // Requires mutex_.
  void report_stats() const;
  void log_stats(const PrefetchSessionReport& report) const;

  ThreadPool* copy_pool_ = nullptr;
  ThreadPool* stats_pool_ = nullptr;
  StoragePrefetchRequest request_;
  size_t batch_size_ = 0;
  int64_t worker_rank_ = 0;
  PrefetchSessionCallbacks callbacks_;
  Timer timer_;
  mutable std::mutex mutex_;
  std::condition_variable closed_;
  PrefetchWorkerStats stats_;
  std::chrono::steady_clock::time_point batch_scheduled_at_;
  std::chrono::steady_clock::time_point result_written_at_;
  brpc::StreamId stream_id_ = brpc::INVALID_STREAM_ID;
  size_t batch_index_ = 0;
  bool last_batch_gate_complete_ = false;
  bool stop_after_batch_ = false;
  bool batch_running_ = false;
  bool finished_ = false;
  bool failed_ = false;
  bool stream_closed_ = false;
  bool report_scheduled_ = false;
  State state_ = State::CREATED;
  std::shared_ptr<WorkerPrefetchSession> keepalive_;
};

}  // namespace xllm
