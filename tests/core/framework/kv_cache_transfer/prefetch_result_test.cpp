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

#include "framework/kv_cache_transfer/prefetch_result.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace xllm {
namespace {

StoragePrefetchRequest make_request(size_t unit_count) {
  StoragePrefetchRequest request;
  request.request_id = "prefetch-result-test";
  request.units.reserve(unit_count);
  for (size_t unit = 0; unit < unit_count; ++unit) {
    PrefetchUnit prefetch_unit;
    prefetch_unit.gated_blocks.emplace_back(
        /*src_block_id=*/-1, /*dst_block_id=*/static_cast<int32_t>(unit));
    request.units.emplace_back(std::move(prefetch_unit));
  }
  return request;
}

class PrefetchResultHarness final {
 public:
  PrefetchResultHarness(size_t worker_count,
                        size_t unit_count,
                        size_t batch_size,
                        int64_t timeout_ms = -1,
                        bool stop_requested = false)
      : request_(make_request(unit_count)),
        result_(std::make_shared<PrefetchResult>(
            worker_count,
            request_,
            batch_size,
            timeout_ms,
            [stop_requested]() { return stop_requested; },
            [this](PrefetchSummary summary) {
              summary_ = std::move(summary);
            })) {}

  PrefetchResult& result() { return *result_; }
  const std::optional<PrefetchSummary>& summary() const { return summary_; }

 private:
  StoragePrefetchRequest request_;
  std::optional<PrefetchSummary> summary_;
  std::shared_ptr<PrefetchResult> result_;
};

TEST(PrefetchResultTest, CompleteRunReportsComplete) {
  PrefetchResultHarness harness(
      /*worker_count=*/2, /*unit_count=*/3, /*batch_size=*/2);
  PrefetchResult& result = harness.result();
  for (size_t worker = 0; worker < 2; ++worker) {
    EXPECT_EQ(result.record_batch_result(worker, {1, 1}, {1, 1}),
              PrefetchControl::CONTINUE);
    EXPECT_EQ(result.record_batch_result(worker, {1, 0}, {1, 0}),
              PrefetchControl::STOP);
    result.mark_worker_ended(worker);
  }

  ASSERT_TRUE(harness.summary().has_value());
  const PrefetchSummary& summary = *harness.summary();
  EXPECT_EQ(summary.stop_reason, PrefetchStopReason::COMPLETE);
  EXPECT_EQ(summary.gated_hits, std::vector<uint8_t>({1, 1, 1}));
  EXPECT_GE(summary.latency_ms, 0.0);
}

TEST(PrefetchResultTest, GatedMissIsReported) {
  PrefetchResultHarness harness(
      /*worker_count=*/1, /*unit_count=*/4, /*batch_size=*/2);
  PrefetchResult& result = harness.result();
  EXPECT_EQ(result.record_batch_result(/*worker_index=*/0, {1, 0}, {1, 1}),
            PrefetchControl::STOP);
  result.mark_worker_ended(/*worker_index=*/0);

  ASSERT_TRUE(harness.summary().has_value());
  EXPECT_EQ(harness.summary()->stop_reason, PrefetchStopReason::MISS);
  EXPECT_EQ(harness.summary()->gated_hits, std::vector<uint8_t>({1, 0, 0, 0}));
}

TEST(PrefetchResultTest, TimeoutOnLastBatchIsComplete) {
  PrefetchResultHarness harness(/*worker_count=*/1,
                                /*unit_count=*/2,
                                /*batch_size=*/2,
                                /*timeout_ms=*/1);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  PrefetchResult& result = harness.result();
  EXPECT_EQ(result.record_batch_result(0, {1, 1}, {1, 1}),
            PrefetchControl::STOP);
  result.mark_worker_ended(0);

  ASSERT_TRUE(harness.summary().has_value());
  EXPECT_EQ(harness.summary()->stop_reason, PrefetchStopReason::COMPLETE);
}

TEST(PrefetchResultTest, TimeoutWithRemainingBatchesIsReported) {
  PrefetchResultHarness harness(/*worker_count=*/1,
                                /*unit_count=*/4,
                                /*batch_size=*/2,
                                /*timeout_ms=*/1);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  PrefetchResult& result = harness.result();
  EXPECT_EQ(result.record_batch_result(0, {1, 1}, {1, 1}),
            PrefetchControl::STOP);
  result.mark_worker_ended(0);

  ASSERT_TRUE(harness.summary().has_value());
  EXPECT_EQ(harness.summary()->stop_reason, PrefetchStopReason::TIMEOUT);
}

TEST(PrefetchResultTest, CancelledRequestWithRemainingBatchesIsReported) {
  PrefetchResultHarness harness(/*worker_count=*/1,
                                /*unit_count=*/4,
                                /*batch_size=*/2,
                                /*timeout_ms=*/-1,
                                /*stop_requested=*/true);
  PrefetchResult& result = harness.result();
  EXPECT_EQ(result.record_batch_result(0, {1, 1}, {1, 1}),
            PrefetchControl::STOP);
  result.mark_worker_ended(0);

  ASSERT_TRUE(harness.summary().has_value());
  EXPECT_EQ(harness.summary()->stop_reason, PrefetchStopReason::CANCELLED);
}

TEST(PrefetchResultTest, FailedWorkerDiscardsHitsButKeepsReason) {
  PrefetchResultHarness harness(
      /*worker_count=*/1, /*unit_count=*/2, /*batch_size=*/2);
  PrefetchResult& result = harness.result();
  result.mark_worker_ended(/*worker_index=*/0, /*worker_ok=*/false);

  ASSERT_TRUE(harness.summary().has_value());
  EXPECT_EQ(harness.summary()->stop_reason, PrefetchStopReason::WORKER_FAILED);
  EXPECT_TRUE(harness.summary()->gated_hits.empty());
}

}  // namespace
}  // namespace xllm
