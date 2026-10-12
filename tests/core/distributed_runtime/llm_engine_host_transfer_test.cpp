/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/distributed_runtime/llm_engine.h"
#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/request.h"
#include "framework/request/stopping_checker.h"

namespace xllm {
namespace {

class RecordingWorker final : public WorkerClient {
 public:
  void prefetch_from_storage(
      const std::shared_ptr<const StoragePrefetchRequest>& request,
      std::shared_ptr<PrefetchResult> result,
      size_t worker_index) override {
    prefetch_request_ = request;
    prefetch_result_ = std::move(result);
    prefetch_index_ = worker_index;
  }

  folly::SemiFuture<uint32_t> transfer_kv_blocks(
      const std::vector<BlockTransferInfo>& infos) override {
    ++offload_count_;
    offload_infos_ = infos;
    return folly::makeSemiFuture(static_cast<uint32_t>(infos.size()));
  }

  void transfer_kv_blocks(
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& infos) override {
    ++load_count_;
    load_batch_id_ = batch_id;
    load_infos_ = infos;
  }

  uint32_t offload_count_ = 0;
  uint32_t load_count_ = 0;
  uint64_t load_batch_id_ = 0;
  std::vector<BlockTransferInfo> offload_infos_;
  std::vector<BlockTransferInfo> load_infos_;
  std::shared_ptr<const StoragePrefetchRequest> prefetch_request_;
  std::shared_ptr<PrefetchResult> prefetch_result_;
  size_t prefetch_index_ = 0;
};

class ClientEngine final : public LLMEngine {
 public:
  ClientEngine(runtime::Options options,
               std::vector<std::shared_ptr<WorkerClient>> clients)
      : LLMEngine(std::move(options), std::move(clients)) {}
};

struct TransferTopology {
  uint32_t workers_;
  uint32_t dp_;
  uint32_t cp_;
  uint32_t rank_;
  std::vector<uint32_t> expected_;
};

class EngineHostTransferTest
    : public ::testing::TestWithParam<TransferTopology> {};

TEST_P(EngineHostTransferTest, TransfersEveryShardInOnlyTheRequestedDpGroup) {
  const auto& topology = GetParam();
  std::vector<std::shared_ptr<WorkerClient>> clients;
  clients.reserve(topology.workers_);
  std::vector<std::shared_ptr<RecordingWorker>> workers;
  workers.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    auto worker = std::make_shared<RecordingWorker>();
    clients.emplace_back(worker);
    workers.emplace_back(std::move(worker));
  }
  runtime::Options options;
  options.world_size(static_cast<int32_t>(topology.workers_))
      .dp_size(static_cast<int32_t>(topology.dp_))
      .cp_size(static_cast<int32_t>(topology.cp_));
  ClientEngine engine(std::move(options), std::move(clients));
  EXPECT_EQ(engine.host_transfer_worker_count(), topology.expected_.size());
  BlockTransferInfo info(/*src_block_id=*/3, /*dst_block_id=*/7);
  info.transfer_type = TransferType::D2H2G;
  info.block_type = BlockType::C4;
  std::fill(std::begin(info.hash_key), std::end(info.hash_key), 42);
  auto futures = engine.transfer_kv_blocks(topology.rank_, {info});
  EXPECT_EQ(futures.size(), topology.expected_.size());
  for (auto& future : futures) {
    EXPECT_EQ(std::move(future).get(), 1U);
  }
  info.transfer_type = TransferType::H2D;
  engine.transfer_kv_blocks(topology.rank_, /*batch_id=*/123, {info});
  std::vector<uint32_t> offloaded;
  std::vector<uint32_t> loaded;
  offloaded.reserve(topology.workers_);
  loaded.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    const auto& worker = *workers[i];
    EXPECT_LE(worker.offload_count_, 1U);
    EXPECT_LE(worker.load_count_, 1U);
    if (worker.offload_count_ != 0) {
      offloaded.emplace_back(i);
      ASSERT_EQ(worker.offload_infos_.size(), 1U);
      EXPECT_EQ(worker.offload_infos_[0].src_block_id, 3);
      EXPECT_EQ(worker.offload_infos_[0].dst_block_id, 7);
      EXPECT_EQ(worker.offload_infos_[0].transfer_type, TransferType::D2H2G);
    }
    if (worker.load_count_ == 0) {
      continue;
    }
    loaded.emplace_back(i);
    EXPECT_EQ(worker.load_batch_id_, 123U);
    ASSERT_EQ(worker.load_infos_.size(), 1U);
    EXPECT_EQ(worker.load_infos_[0].src_block_id, 3);
    EXPECT_EQ(worker.load_infos_[0].dst_block_id, 7);
    EXPECT_EQ(worker.load_infos_[0].block_type, BlockType::C4);
    EXPECT_EQ(worker.load_infos_[0].transfer_type, TransferType::H2D);
    EXPECT_TRUE(std::equal(std::begin(info.hash_key),
                           std::end(info.hash_key),
                           std::begin(worker.load_infos_[0].hash_key)));
  }
  EXPECT_EQ(offloaded, topology.expected_);
  EXPECT_EQ(loaded, topology.expected_);
}

TEST_P(EngineHostTransferTest, PrefetchWaitsForEveryShardAndStopsAtFirstMiss) {
  const auto& topology = GetParam();
  std::vector<std::shared_ptr<WorkerClient>> clients;
  std::vector<std::shared_ptr<RecordingWorker>> workers;
  clients.reserve(topology.workers_);
  workers.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    auto worker = std::make_shared<RecordingWorker>();
    clients.emplace_back(worker);
    workers.emplace_back(std::move(worker));
  }
  runtime::Options options;
  options.world_size(static_cast<int32_t>(topology.workers_))
      .dp_size(static_cast<int32_t>(topology.dp_))
      .cp_size(static_cast<int32_t>(topology.cp_))
      .prefetch_timeout(0)
      .prefetch_batch_size(5);
  ClientEngine engine(std::move(options), std::move(clients));
  auto request = std::make_shared<StoragePrefetchRequest>();
  for (int32_t i = 0; i < 5; ++i) {
    BlockTransferInfo info(/*src_block_id=*/i, /*dst_block_id=*/i);
    info.transfer_type = TransferType::G2H;
    PrefetchUnit unit;
    unit.gated_blocks.emplace_back(info);
    request->units.emplace_back(std::move(unit));
  }
  size_t callbacks = 0;
  size_t common_prefix = 0;
  engine.prefetch_from_storage(
      topology.rank_,
      request,
      [] { return false; },
      [&](PrefetchSummary summary) {
        ++callbacks;
        common_prefix = 0;
        while (common_prefix < summary.gated_hits.size() &&
               summary.gated_hits[common_prefix] != 0) {
          ++common_prefix;
        }
      });
  std::vector<uint32_t> prefetched;
  prefetched.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    if (workers[i]->prefetch_request_ != nullptr) {
      prefetched.emplace_back(i);
    }
  }
  ASSERT_EQ(prefetched, topology.expected_);
  for (size_t i = 0; i < topology.expected_.size(); ++i) {
    const auto& worker = *workers[topology.expected_[i]];
    EXPECT_EQ(worker.prefetch_request_, request);
    EXPECT_EQ(worker.prefetch_index_, i);
    EXPECT_EQ(worker.prefetch_result_->worker_count(),
              topology.expected_.size());
    EXPECT_EQ(callbacks, 0U);
    const bool last = i + 1 == topology.expected_.size();
    const std::vector<uint8_t> gated_hits =
        last ? std::vector<uint8_t>{1, 1, 1, 0, 1}
             : std::vector<uint8_t>{1, 1, 1, 1, 1};
    const std::vector<uint8_t> non_gated_hits(5, 1);
    EXPECT_EQ(worker.prefetch_result_->record_batch_result(
                  i, gated_hits, non_gated_hits),
              PrefetchControl::STOP);
    EXPECT_EQ(callbacks, 0U);
    worker.prefetch_result_->mark_worker_ended(i, /*worker_ok=*/true);
  }
  EXPECT_EQ(callbacks, 1U);
  EXPECT_EQ(common_prefix, 3U);
}

INSTANTIATE_TEST_SUITE_P(
    DpCp,
    EngineHostTransferTest,
    ::testing::Values(
        TransferTopology{8, 1, 1, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
        TransferTopology{8, 1, 4, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
        TransferTopology{16, 2, 1, 1, {8, 9, 10, 11, 12, 13, 14, 15}},
        TransferTopology{16, 2, 4, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
        TransferTopology{16, 2, 4, 1, {8, 9, 10, 11, 12, 13, 14, 15}}));

// ===========================================================================
// M11.5 scheduler-overlap prework: retain_cache_blocks coverage for a
// non-DFlash2 CP batch (llm_engine.cpp LLMEngine::step).
//
// The engine only retains a dispatched batch's cache blocks when
// enable_schedule_overlap && is_dflash2_algorithm. That guard is sufficient
// for a non-DFlash2 CP prefill because the scheduler-side block free of a
// request finished in the deferred output processing (collect_finished ->
// BlockManagerPool::deallocate, scheduler_policy.cpp) cannot corrupt the
// in-flight overlap step: every device access to KV/LINEAR cache tensors is
// enqueued on the per-worker compute stream in step-task order (the
// single-threaded WorkerImpl task pool, worker_impl.h, keeps two step tasks
// FIFO-serial; LLMWorkerImpl::step_for_schedule_overlap chains the deferred
// linear-state restore and the step on compute_stream_), so a block freed at
// the next schedule_request and reallocated into the following batch is only
// written by device work ordered after the in-flight step. DFlash2 needs the
// extra batch-held Block aliases because its prelaunch enqueues KV writes one
// additional step ahead, outside that FIFO window (see
// dflash2_prelaunch_lifecycle_test).
// ===========================================================================

class StepCaptureWorker final : public WorkerClient {
 public:
  folly::SemiFuture<std::optional<RawForwardOutput>> step_remote_async(
      const ForwardInput& inputs) override {
    ++step_count_;
    captured_token_ids_ = inputs.token_ids.clone();
    RawForwardOutput output;
    RawSampleOutput sample;
    RawToken token;
    token.id = 5;
    sample.tokens.emplace_back(std::move(token));
    output.outputs.emplace_back(std::move(sample));
    return folly::makeSemiFuture(
        std::optional<RawForwardOutput>(std::move(output)));
  }

  uint32_t step_count() const { return step_count_; }
  const torch::Tensor& captured_token_ids() const {
    return captured_token_ids_;
  }

 private:
  uint32_t step_count_ = 0;
  torch::Tensor captured_token_ids_;
};

std::shared_ptr<Request> make_overlap_step_request() {
  const std::vector<int32_t> prompt{1, 2, 3, 4};
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(32);
  stopping_checker.set_max_context_len(30000);
  stopping_checker.set_ignore_eos(true);
  RequestState req_state("x",
                         prompt,
                         sampling_param,
                         scheduler_param,
                         stopping_checker,
                         prompt.size() + 30000,
                         /*n=*/1,
                         /*best_of=*/1,
                         /*logprobs=*/false,
                         /*stream=*/false,
                         /*echo=*/false,
                         /*skip_special_tokens=*/false,
                         /*enable_schedule_overlap=*/false,
                         /*output_func=*/nullptr,
                         /*outputs_func=*/nullptr);
  return std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
}

TEST(LLMEngineOverlapRetainTest,
     NonDFlash2OverlapStepDoesNotRetainCpPrefillBlocks) {
  // Two workers form one DP group with cp_size=2: the engine must dispatch
  // the same forward_inputs[dp_rank] to both CP ranks (llm_engine.cpp: "Engine
  // sends full global tokens; model-side CP shards inside the worker").
  std::vector<std::shared_ptr<StepCaptureWorker>> workers;
  std::vector<std::shared_ptr<WorkerClient>> clients;
  workers.reserve(2);
  clients.reserve(2);
  for (int32_t worker_rank = 0; worker_rank < 2; ++worker_rank) {
    auto worker = std::make_shared<StepCaptureWorker>();
    workers.emplace_back(worker);
    clients.emplace_back(std::move(worker));
  }
  runtime::Options options;
  options.world_size(2).dp_size(1).cp_size(2).enable_schedule_overlap(true);
  // Any non-DFlash2 algorithm leaves the retain guard off.
  options.speculative_algorithm("mtp");
  ClientEngine engine(std::move(options), std::move(clients));

  BlockManagerPool::Options pool_options;
  pool_options.num_blocks_ = 32;
  pool_options.block_size_ = 4;
  pool_options.max_seqs_per_batch_ = 1024;
  BlockManagerPool pool(pool_options, /*dp_size=*/1);
  auto request = make_overlap_step_request();
  Sequence* sequence = request->sequences().front().get();
  ASSERT_TRUE(pool.try_allocate(sequence));
  ASSERT_GT(sequence->kv_state().num_blocks(BlockType::KV), 0u);
  // A prefill batch processes every prompt token: no KV tokens yet.
  sequence->kv_state().set_kv_cache_tokens_num(0);

  std::vector<Batch> batches;
  batches.emplace_back(sequence);
  engine.step(batches);

  // Both CP ranks ran the same batch with byte-identical token inputs.
  for (const auto& worker : workers) {
    ASSERT_EQ(worker->step_count(), 1u);
  }
  EXPECT_TRUE(torch::equal(workers[0]->captured_token_ids(),
                           workers[1]->captured_token_ids()));

  // The DFlash2-only guard means a non-DFlash2 overlap step does NOT pin the
  // batch's blocks: the sequence's own alias is the only reference. Safety
  // for the freed-block window comes from compute-stream FIFO ordering, not
  // from scheduler-side retention (see the comment above this test).
  const auto kv_blocks = sequence->kv_state().blocks(BlockType::KV);
  ASSERT_FALSE(kv_blocks.empty());
  EXPECT_EQ(kv_blocks.front().ref_count(), 1u);
}

TEST(LLMEngineOverlapRetainTest, DFlash2OverlapStepRetainsCpPrefillBlocks) {
  std::vector<std::shared_ptr<WorkerClient>> clients;
  clients.reserve(2);
  for (int32_t worker_rank = 0; worker_rank < 2; ++worker_rank) {
    clients.emplace_back(std::make_shared<StepCaptureWorker>());
  }
  runtime::Options options;
  options.world_size(2).dp_size(1).cp_size(2).enable_schedule_overlap(true);
  options.speculative_algorithm("DFlash2");
  ClientEngine engine(std::move(options), std::move(clients));

  BlockManagerPool::Options pool_options;
  pool_options.num_blocks_ = 32;
  pool_options.block_size_ = 4;
  pool_options.max_seqs_per_batch_ = 1024;
  BlockManagerPool pool(pool_options, /*dp_size=*/1);
  auto request = make_overlap_step_request();
  Sequence* sequence = request->sequences().front().get();
  ASSERT_TRUE(pool.try_allocate(sequence));
  ASSERT_GT(sequence->kv_state().num_blocks(BlockType::KV), 0u);
  sequence->kv_state().set_kv_cache_tokens_num(0);

  std::vector<Batch> batches;
  batches.emplace_back(sequence);
  engine.step(batches);

  // With DFlash2 the dispatched batch holds Block aliases until the deferred
  // output processing retires it, so the sequence's blocks are shared.
  const auto kv_blocks = sequence->kv_state().blocks(BlockType::KV);
  ASSERT_FALSE(kv_blocks.empty());
  EXPECT_EQ(kv_blocks.front().ref_count(), 2u);
  // Dropping the batch releases the retain.
  batches.clear();
  EXPECT_EQ(sequence->kv_state().blocks(BlockType::KV).front().ref_count(), 1u);
}

}  // namespace
}  // namespace xllm
