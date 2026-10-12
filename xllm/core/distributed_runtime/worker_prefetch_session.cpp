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

#include "core/distributed_runtime/worker_prefetch_session.h"

#include <glog/logging.h>

#include <algorithm>
#include <iomanip>
#include <utility>

#include "framework/kv_cache_transfer/prefetch_tail_probe.h"

namespace xllm {
namespace {
double bytes_to_gib(uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}
}  // namespace

WorkerPrefetchSession::WorkerPrefetchSession(ThreadPool* copy_pool,
                                             ThreadPool* stats_pool,
                                             StoragePrefetchRequest request,
                                             size_t batch_size,
                                             int64_t worker_rank,
                                             PrefetchSessionCallbacks callbacks)
    : copy_pool_(copy_pool),
      stats_pool_(stats_pool),
      request_(std::move(request)),
      batch_size_(batch_size),
      worker_rank_(worker_rank),
      callbacks_(std::move(callbacks)) {
  CHECK(copy_pool_ != nullptr);
  CHECK(request_.valid());
  CHECK_GT(batch_size_, 0u);
  CHECK(callbacks_.get != nullptr);
  CHECK(stats_pool_ == nullptr || callbacks_.probe != nullptr);
  if (callbacks_.write == nullptr) {
    callbacks_.write = [](brpc::StreamId id, const butil::IOBuf& result) {
      return brpc::StreamWrite(id, result);
    };
  }
  if (callbacks_.close == nullptr) {
    callbacks_.close = [](brpc::StreamId id) { brpc::StreamClose(id); };
  }
}

void WorkerPrefetchSession::retain() {
  std::lock_guard<std::mutex> lock(mutex_);
  keepalive_ = shared_from_this();
}

void WorkerPrefetchSession::release() {
  std::lock_guard<std::mutex> lock(mutex_);
  keepalive_.reset();
}

void WorkerPrefetchSession::start(brpc::StreamId stream_id) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (finished_ || state_ != State::CREATED) {
      return;
    }
    stream_id_ = stream_id;
    state_ = State::RUNNING_BATCH;
  }
  schedule_batch();
}

void WorkerPrefetchSession::shutdown() {
  brpc::StreamId id;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    id = stream_id_;
  }
  finish(id, /*failed=*/true);
  // A brpc callback may already be between marking the session finished and
  // closing the stream. Wait until it can no longer initiate a new report.
  std::unique_lock<std::mutex> lock(mutex_);
  closed_.wait(lock, [this]() { return stream_closed_; });
}

int WorkerPrefetchSession::on_received_messages(brpc::StreamId id,
                                                butil::IOBuf* const messages[],
                                                size_t size) {
  if (size != 1 || messages[0]->length() != 1) {
    finish(id, /*failed=*/true);
    return -1;
  }
  uint8_t control_byte = 0;
  messages[0]->copy_to(&control_byte, sizeof(control_byte));
  const PrefetchControl control = static_cast<PrefetchControl>(control_byte);
  bool run_next = false;
  bool complete = false;
  bool failed = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (finished_) {
      return 0;
    }
    if (stats_pool_ != nullptr && state_ == State::WAITING_DECISION) {
      stats_.control_wait_us += elapsed_us(result_written_at_);
    }
    if (state_ == State::RUNNING_BATCH && control == PrefetchControl::STOP) {
      stop_after_batch_ = true;
    } else if (state_ != State::WAITING_DECISION) {
      failed = true;
    } else if (control == PrefetchControl::STOP) {
      complete = true;
    } else if (control == PrefetchControl::CONTINUE &&
               last_batch_gate_complete_ &&
               batch_index_ + 1 < request_.batch_count(batch_size_)) {
      ++batch_index_;
      state_ = State::RUNNING_BATCH;
      run_next = true;
    } else {
      failed = true;
    }
  }
  if (run_next) {
    schedule_batch();
  } else if (complete || failed) {
    finish(id, failed);
  }
  return failed && control != PrefetchControl::STOP ? -1 : 0;
}

void WorkerPrefetchSession::on_idle_timeout(brpc::StreamId id) {
  finish(id, /*failed=*/true);
}

void WorkerPrefetchSession::on_failed(brpc::StreamId id,
                                      int /*error_code*/,
                                      const std::string& /*error_text*/) {
  finish(id, /*failed=*/true);
}

void WorkerPrefetchSession::on_closed(brpc::StreamId /*id*/) {
  std::shared_ptr<WorkerPrefetchSession> self = shared_from_this();
  std::lock_guard<std::mutex> lock(mutex_);
  if (!finished_) {
    finished_ = true;
    failed_ = true;
    stats_.total_us = static_cast<uint64_t>(timer_.elapsed_microseconds());
  }
  state_ = State::CLOSED;
  stream_closed_ = true;
  report_ready();
  keepalive_.reset();
  closed_.notify_all();
}

uint64_t WorkerPrefetchSession::elapsed_us(
    std::chrono::steady_clock::time_point since) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - since)
          .count());
}

void WorkerPrefetchSession::schedule_batch() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (finished_ || state_ != State::RUNNING_BATCH) {
    return;
  }
  batch_scheduled_at_ = std::chrono::steady_clock::now();
  std::shared_ptr<WorkerPrefetchSession> self = shared_from_this();
  copy_pool_->schedule([self = std::move(self)]() { self->run_batch(); });
}

void WorkerPrefetchSession::run_batch() {
  size_t batch_index = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (finished_ || state_ != State::RUNNING_BATCH) {
      return;
    }
    batch_running_ = true;
    batch_index = batch_index_;
    if (stats_pool_ != nullptr) {
      stats_.queue_wait_us += elapsed_us(batch_scheduled_at_);
    }
  }
  const size_t unit_begin = request_.batch_unit_begin(batch_index, batch_size_);
  const size_t unit_count = request_.batch_unit_count(batch_index, batch_size_);
  std::vector<BlockTransferInfo> batch_transfers;
  batch_transfers.reserve(
      request_.batch_transfer_count(batch_index, batch_size_));
  std::vector<size_t> gated_offsets;
  gated_offsets.reserve(unit_count);
  std::vector<size_t> non_gated_offsets;
  non_gated_offsets.reserve(unit_count);
  for (size_t index = unit_begin; index < unit_begin + unit_count; ++index) {
    const PrefetchUnit& unit = request_.units[index];
    gated_offsets.emplace_back(batch_transfers.size());
    batch_transfers.insert(batch_transfers.end(),
                           unit.gated_blocks.begin(),
                           unit.gated_blocks.end());
    non_gated_offsets.emplace_back(batch_transfers.size());
    batch_transfers.insert(batch_transfers.end(),
                           unit.non_gated_blocks.begin(),
                           unit.non_gated_blocks.end());
  }
  Slice<BlockTransferInfo> batch_slice(batch_transfers);
  StoreGetStats get_stats;
  const std::chrono::steady_clock::time_point get_begin =
      std::chrono::steady_clock::now();
  const std::vector<uint8_t> transfer_hits = callbacks_.get(
      batch_slice, stats_pool_ != nullptr ? &get_stats : nullptr);
  const uint64_t call_us = elapsed_us(get_begin);
  const uint64_t get_us = call_us - std::min(call_us, get_stats.tier_query_us);
  const bool valid_result = transfer_hits.size() == batch_transfers.size();
  std::vector<uint8_t> gated_hits(batch_size_, 0);
  std::vector<uint8_t> non_gated_hits(batch_size_, 0);
  bool gate_complete = valid_result;
  if (valid_result) {
    for (size_t local = 0; local < unit_count; ++local) {
      const PrefetchUnit& unit = request_.units[unit_begin + local];
      bool gated_hit = true;
      for (size_t offset = 0; offset < unit.gated_blocks.size(); ++offset) {
        gated_hit =
            gated_hit && transfer_hits[gated_offsets[local] + offset] != 0;
      }
      bool non_gated_hit = !unit.has_non_gated;
      if (unit.has_non_gated) {
        non_gated_hit = !unit.non_gated_blocks.empty();
        for (size_t offset = 0; offset < unit.non_gated_blocks.size();
             ++offset) {
          non_gated_hit = non_gated_hit &&
                          transfer_hits[non_gated_offsets[local] + offset] != 0;
        }
      }
      gated_hits[local] = gated_hit ? 1 : 0;
      non_gated_hits[local] = non_gated_hit ? 1 : 0;
      gate_complete = gate_complete && gated_hit;
    }
  }
  bool close_after_result = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stats_pool_ != nullptr) {
      ++stats_.batches;
      stats_.executed_units += unit_count;
      stats_.hit_units += static_cast<uint64_t>(std::count(
          gated_hits.begin(),
          gated_hits.begin() + static_cast<std::ptrdiff_t>(unit_count),
          static_cast<uint8_t>(1)));
      stats_.requested_blocks += batch_transfers.size();
      if (valid_result) {
        stats_.hit_blocks +=
            static_cast<uint64_t>(std::count(transfer_hits.begin(),
                                             transfer_hits.end(),
                                             static_cast<uint8_t>(1)));
      }
      stats_.get.read_bytes += get_stats.read_bytes;
      stats_.get.memory_objects += get_stats.memory_objects;
      stats_.get.memory_bytes += get_stats.memory_bytes;
      stats_.get.disk_objects += get_stats.disk_objects;
      stats_.get.disk_bytes += get_stats.disk_bytes;
      stats_.get.tier_query_us += get_stats.tier_query_us;
      stats_.get_us += get_us;
      stats_.probe_begin_unit =
          PrefetchTailProbe::tail_begin(unit_begin, gated_hits, unit_count);
      result_written_at_ = std::chrono::steady_clock::now();
    }
    batch_running_ = false;
    if (finished_) {
      report_ready();
      return;
    }
    last_batch_gate_complete_ = gate_complete;
    close_after_result = stop_after_batch_;
    state_ = close_after_result ? State::COMPLETED : State::WAITING_DECISION;
  }
  if (!valid_result) {
    finish(stream_id_, /*failed=*/true);
    return;
  }
  butil::IOBuf result;
  result.append(gated_hits.data(), gated_hits.size());
  result.append(non_gated_hits.data(), non_gated_hits.size());
  if (callbacks_.write(stream_id_, result) != 0) {
    finish(stream_id_, /*failed=*/true);
  } else if (close_after_result) {
    finish(stream_id_, /*failed=*/false);
  }
}

void WorkerPrefetchSession::finish(brpc::StreamId id, bool failed) {
  std::shared_ptr<WorkerPrefetchSession> self = shared_from_this();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (finished_) {
      return;
    }
    finished_ = true;
    failed_ = failed;
    state_ = failed ? State::FAILED : State::COMPLETED;
    stats_.total_us = static_cast<uint64_t>(timer_.elapsed_microseconds());
  }
  callbacks_.close(id);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stream_closed_ = true;
    report_ready();
    closed_.notify_all();
  }
}

void WorkerPrefetchSession::report_ready() {
  if (stats_pool_ == nullptr || !stream_closed_ || batch_running_ ||
      report_scheduled_) {
    return;
  }
  report_scheduled_ = true;
  std::shared_ptr<WorkerPrefetchSession> self = shared_from_this();
  stats_pool_->schedule([self = std::move(self)]() { self->report_stats(); });
}

void WorkerPrefetchSession::report_stats() const {
  PrefetchSessionReport report;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    report.stats = stats_;
    report.failed = failed_;
  }
  if (!report.failed) {
    PrefetchTailProbe probe(request_, report.stats.probe_begin_unit);
    report.probed_units = probe.unit_count();
    if (probe.transfers().empty()) {
      report.probed_present_units = 0;
    } else {
      Slice<BlockTransferInfo> probe_slice(probe.transfers());
      report.probed_present_units =
          probe.count_present_units(callbacks_.probe(probe_slice));
    }
  }
  if (callbacks_.report != nullptr) {
    callbacks_.report(report);
  } else {
    log_stats(report);
  }
}

void WorkerPrefetchSession::log_stats(
    const PrefetchSessionReport& report) const {
  const PrefetchWorkerStats& stats = report.stats;
  const double get_seconds = static_cast<double>(stats.get_us) / 1e6;
  const double get_bw =
      get_seconds > 0 ? bytes_to_gib(stats.get.read_bytes) / get_seconds : 0;
  const char* probe_status =
      report.failed
          ? "skipped"
          : (report.probed_present_units.has_value() ? "complete" : "invalid");
  LOG(INFO) << std::fixed << std::setprecision(2)
            << "[StorePrefetch][Worker] request_id: " << request_.request_id
            << ", worker_rank: " << worker_rank_
            << ", units: " << stats.hit_units << "/" << stats.executed_units
            << "/" << request_.unit_count() << ", blocks: " << stats.hit_blocks
            << "/" << stats.requested_blocks
            << ", read_bytes: " << bytes_to_gib(stats.get.read_bytes)
            << "GiB, memory_objects: " << stats.get.memory_objects
            << ", memory_bytes: " << bytes_to_gib(stats.get.memory_bytes)
            << "GiB, disk_objects: " << stats.get.disk_objects
            << ", disk_bytes: " << bytes_to_gib(stats.get.disk_bytes)
            << "GiB, batches: " << stats.batches << std::setprecision(1)
            << ", queue_wait: " << stats.queue_wait_us / 1e3
            << "ms, get_time: " << stats.get_us / 1e3
            << "ms, tier_query_time: " << stats.get.tier_query_us / 1e3
            << "ms, control_wait: " << stats.control_wait_us / 1e3
            << "ms, total: " << stats.total_us / 1e3 << "ms"
            << std::setprecision(2) << ", get_bw: " << get_bw
            << "GiB/s, status: " << (report.failed ? "failed" : "completed")
            << ", probe_status: " << probe_status
            << ", probed_units: " << report.probed_units
            << ", probed_present_units: "
            << (report.probed_present_units.has_value()
                    ? std::to_string(*report.probed_present_units)
                    : "unknown");
}

}  // namespace xllm
