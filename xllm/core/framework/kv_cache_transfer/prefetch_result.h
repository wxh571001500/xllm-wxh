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

#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "framework/model/model_input_params.h"
#include "util/timer.h"

namespace xllm {

enum class PrefetchControl : uint8_t {
  CONTINUE = 0,
  STOP = 1,
};

class PrefetchUnit final {
 public:
  std::vector<BlockTransferInfo> gated_blocks;
  std::vector<BlockTransferInfo> non_gated_blocks;
  // A unit may have an optional cache whose allocation failed. Keeping this
  // bit separate from the vector lets the worker report an optional miss
  // instead of treating an empty vector as "not applicable".
  bool has_non_gated = false;

  bool valid() const {
    const auto valid_transfer = [](const BlockTransferInfo& info) {
      return info.transfer_type == TransferType::G2H;
    };
    return !gated_blocks.empty() &&
           std::all_of(
               gated_blocks.begin(), gated_blocks.end(), valid_transfer) &&
           std::all_of(non_gated_blocks.begin(),
                       non_gated_blocks.end(),
                       valid_transfer);
  }
};

class StoragePrefetchRequest final {
 public:
  // Identifies the owning request in worker-side prefetch logs.
  std::string request_id;
  std::vector<PrefetchUnit> units;

  bool valid() const {
    return !units.empty() &&
           std::all_of(units.begin(),
                       units.end(),
                       [](const PrefetchUnit& unit) { return unit.valid(); });
  }

  size_t unit_count() const { return units.size(); }

  size_t batch_count(size_t batch_size) const {
    CHECK_GT(batch_size, 0u);
    return (units.size() + batch_size - 1) / batch_size;
  }

  size_t batch_unit_begin(size_t batch_index, size_t batch_size) const {
    CHECK_LT(batch_index, batch_count(batch_size));
    return batch_index * batch_size;
  }

  // This is the number of real units in the batch. The worker still returns
  // batch_size entries for both vectors and pads the remainder with zeroes.
  size_t batch_unit_count(size_t batch_index, size_t batch_size) const {
    const size_t begin = batch_unit_begin(batch_index, batch_size);
    return std::min(batch_size, units.size() - begin);
  }

  size_t batch_transfer_count(size_t batch_index, size_t batch_size) const {
    const size_t begin = batch_unit_begin(batch_index, batch_size);
    const size_t count = batch_unit_count(batch_index, batch_size);
    size_t result = 0;
    for (size_t index = begin; index < begin + count; ++index) {
      result += units[index].gated_blocks.size();
      result += units[index].non_gated_blocks.size();
    }
    return result;
  }
};

enum class PrefetchStopReason : uint8_t {
  COMPLETE = 0,
  // A gated unit missed in at least one worker.
  MISS = 1,
  TIMEOUT = 2,
  // The owning request finished or was cancelled.
  CANCELLED = 3,
  WORKER_FAILED = 4,
};

inline const char* prefetch_stop_reason_name(PrefetchStopReason reason) {
  switch (reason) {
    case PrefetchStopReason::COMPLETE:
      return "complete";
    case PrefetchStopReason::MISS:
      return "miss";
    case PrefetchStopReason::TIMEOUT:
      return "timeout";
    case PrefetchStopReason::CANCELLED:
      return "cancelled";
    case PrefetchStopReason::WORKER_FAILED:
      return "worker_failed";
  }
  return "unknown";
}

struct PrefetchSummary final {
  // Per-unit AND across all workers. gated_hits is used for the contiguous
  // full-cache prefix; non_gated_hits selects the deepest optional checkpoint
  // inside that prefix.
  std::vector<uint8_t> gated_hits;
  std::vector<uint8_t> non_gated_hits;

  // Measurement fields, valid even when a worker failed.
  PrefetchStopReason stop_reason = PrefetchStopReason::COMPLETE;
  double latency_ms = 0.0;
};

class PrefetchResult final {
 public:
  using StopPredicate = std::function<bool()>;
  using DoneCallback = std::function<void(PrefetchSummary)>;

  PrefetchResult(size_t worker_count,
                 const StoragePrefetchRequest& request,
                 size_t batch_size,
                 int64_t timeout_ms,
                 StopPredicate stop_requested,
                 DoneCallback done)
      : workers_(worker_count),
        remaining_workers_(worker_count),
        request_(request),
        batch_size_(batch_size),
        timeout_ms_(timeout_ms),
        stop_requested_(std::move(stop_requested)),
        done_(std::move(done)) {
    CHECK_GT(worker_count, 0u);
    CHECK_GT(batch_size_, 0u);
    CHECK(request_.valid());
    CHECK(timeout_ms_ == -1 || timeout_ms_ > 0);
    CHECK(stop_requested_ != nullptr);
    CHECK(done_ != nullptr);
    for (WorkerProgress& worker : workers_) {
      worker.gated_hits.assign(request_.unit_count(), 0);
      worker.non_gated_hits.assign(request_.unit_count(), 0);
    }
  }

  size_t worker_count() const { return workers_.size(); }
  int64_t stream_idle_timeout_ms() const { return timeout_ms_; }
  size_t batch_wire_size() const { return batch_size_; }
  size_t batch_count() const { return request_.batch_count(batch_size_); }
  void mark_worker_failed() {
    std::lock_guard<std::mutex> lock(mutex_);
    failed_ = true;
    continue_prefetch_->store(false, std::memory_order_release);
  }

  size_t batch_unit_count(size_t worker_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_LT(worker_index, workers_.size());
    return request_.batch_unit_count(workers_[worker_index].batch_index,
                                     batch_size_);
  }

  // The wire response always has 2 * batch_size bytes, including zero padding
  // for a short final batch. Only the first real unit_count bytes from each
  // vector are committed to the aggregate summary.
  std::optional<PrefetchControl> record_batch_result(
      size_t worker_index,
      const std::vector<uint8_t>& gated_hits,
      const std::vector<uint8_t>& non_gated_hits) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_index >= workers_.size()) {
      continue_prefetch_->store(false, std::memory_order_release);
      return std::nullopt;
    }
    WorkerProgress& worker = workers_[worker_index];
    if (worker.state != WorkerState::WAITING_RESULT ||
        worker.batch_index >= batch_count() ||
        gated_hits.size() != batch_size_ ||
        non_gated_hits.size() != batch_size_) {
      continue_prefetch_->store(false, std::memory_order_release);
      return std::nullopt;
    }
    const size_t begin =
        request_.batch_unit_begin(worker.batch_index, batch_size_);
    const size_t count =
        request_.batch_unit_count(worker.batch_index, batch_size_);
    std::copy(gated_hits.begin(),
              gated_hits.begin() + static_cast<std::ptrdiff_t>(count),
              worker.gated_hits.begin() + static_cast<std::ptrdiff_t>(begin));
    std::copy(
        non_gated_hits.begin(),
        non_gated_hits.begin() + static_cast<std::ptrdiff_t>(count),
        worker.non_gated_hits.begin() + static_cast<std::ptrdiff_t>(begin));

    bool gated_complete = true;
    for (size_t index = 0; index < count; ++index) {
      gated_complete = gated_complete && gated_hits[index] != 0;
    }
    const bool has_next_batch = worker.batch_index + 1 < batch_count();
    if (!should_continue_prefetch(gated_complete, has_next_batch)) {
      worker.state = WorkerState::WAITING_CLOSE;
      return PrefetchControl::STOP;
    }
    ++worker.batch_index;
    return PrefetchControl::CONTINUE;
  }

  // Compact legacy response: synthesize unit bytes from a contiguous prefix.
  std::optional<PrefetchControl> record_batch_result(size_t worker_index,
                                                     uint8_t prefix_hit_units) {
    const size_t count = batch_unit_count(worker_index);
    std::vector<uint8_t> gated(batch_size_, 0);
    std::vector<uint8_t> optional(batch_size_, 1);
    std::fill(gated.begin(),
              gated.begin() + std::min<size_t>(prefix_hit_units, count),
              1);
    return record_batch_result(worker_index, gated, optional);
  }

  void mark_worker_ended(size_t worker_index) {
    DoneCallback done;
    PrefetchSummary summary;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      CHECK_LT(worker_index, workers_.size());
      WorkerProgress& worker = workers_[worker_index];
      if (worker.state == WorkerState::ENDED) {
        return;
      }
      if (worker.state != WorkerState::WAITING_CLOSE) {
        failed_ = true;
        continue_prefetch_->store(false, std::memory_order_release);
      }
      worker.state = WorkerState::ENDED;
      CHECK_GT(remaining_workers_, 0u);
      --remaining_workers_;
      if (remaining_workers_ == 0) {
        summary.gated_hits.assign(request_.unit_count(), 1);
        summary.non_gated_hits.assign(request_.unit_count(), 1);
        for (const WorkerProgress& progress : workers_) {
          for (size_t index = 0; index < request_.unit_count(); ++index) {
            summary.gated_hits[index] =
                summary.gated_hits[index] && progress.gated_hits[index];
            summary.non_gated_hits[index] =
                summary.non_gated_hits[index] && progress.non_gated_hits[index];
          }
        }
        fill_measurement(&summary);
        if (failed_) {
          summary.gated_hits.clear();
          summary.non_gated_hits.clear();
        }
        done = std::move(done_);
      }
    }
    if (done != nullptr) {
      done(std::move(summary));
    }
  }

  void mark_worker_ended(size_t worker_index, bool worker_ok) {
    if (!worker_ok) {
      mark_worker_failed();
    }
    mark_worker_ended(worker_index);
  }

 private:
  enum class WorkerState : uint8_t {
    WAITING_RESULT = 0,
    WAITING_CLOSE = 1,
    ENDED = 2,
  };

  struct WorkerProgress {
    size_t batch_index = 0;
    WorkerState state = WorkerState::WAITING_RESULT;
    std::vector<uint8_t> gated_hits;
    std::vector<uint8_t> non_gated_hits;
  };

  bool should_continue_prefetch(bool gated_complete, bool has_next_batch) {
    const bool timed_out =
        timeout_ms_ > 0 && timer_.elapsed_milliseconds() >= timeout_ms_;
    const bool stop_requested = stop_requested_();
    if (!gated_complete || timed_out || stop_requested) {
      continue_prefetch_->store(false, std::memory_order_release);
    }
    // Only a stop that leaves units unfetched is recorded; the first recorded
    // reason wins across workers.
    if (stop_reason_ == PrefetchStopReason::COMPLETE) {
      if (!gated_complete) {
        stop_reason_ = PrefetchStopReason::MISS;
      } else if (has_next_batch && timed_out) {
        stop_reason_ = PrefetchStopReason::TIMEOUT;
      } else if (has_next_batch && stop_requested) {
        stop_reason_ = PrefetchStopReason::CANCELLED;
      }
    }
    return gated_complete && has_next_batch &&
           continue_prefetch_->load(std::memory_order_acquire);
  }

  // Requires mutex_.
  void fill_measurement(PrefetchSummary* summary) const {
    summary->stop_reason =
        failed_ ? PrefetchStopReason::WORKER_FAILED : stop_reason_;
    summary->latency_ms = timer_.elapsed_milliseconds();
  }

  mutable std::mutex mutex_;
  std::vector<WorkerProgress> workers_;
  size_t remaining_workers_ = 0;
  StoragePrefetchRequest request_;
  size_t batch_size_ = 0;
  int64_t timeout_ms_ = -1;
  StopPredicate stop_requested_;
  DoneCallback done_;
  Timer timer_;
  bool failed_ = false;
  PrefetchStopReason stop_reason_ = PrefetchStopReason::COMPLETE;
  std::shared_ptr<std::atomic_bool> continue_prefetch_ =
      std::make_shared<std::atomic_bool>(true);
};

}  // namespace xllm
