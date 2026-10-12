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

#include "fixed_steps_scheduler.h"

#include <absl/time/time.h>
#include <gtest/gtest.h>

#include <algorithm>

#include "common/metrics.h"
#include "continuous_scheduler.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "distributed_runtime/engine.h"
#include "framework/request/rec_type.h"

namespace xllm {

namespace {

class FakeTokenizer : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const override {
    (void)text;
    (void)ids;
    (void)add_special_tokens;
    return false;
  }
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override {
    (void)ids;
    (void)skip_special_tokens;
    return "";
  }
  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override {
    (void)token;
    return std::nullopt;
  }
  std::string id_to_token(int32_t id) const override {
    (void)id;
    return "";
  }
  size_t vocab_size() const override { return 0; }
  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

class FakeEngine : public Engine {
 public:
  FakeEngine(int32_t num_blocks,
             int32_t block_size,
             bool linear_state = false,
             int32_t dp_size = 1,
             bool enable_prefix_cache = false) {
    BlockManagerPool::Options opt;
    opt.num_blocks_ = num_blocks;
    opt.block_size_ = block_size;
    opt.enable_prefix_cache_ = enable_prefix_cache;
    opt.enable_linear_state_ = linear_state;
    opt.linear_state_num_slots_ = linear_state ? 8 : 0;
    fake_tokenizer_ = std::make_unique<FakeTokenizer>();
    fake_block_manager_ = std::make_unique<BlockManagerPool>(opt, dp_size);
  }
  ForwardOutput step(std::vector<Batch>& batch) override {
    for (Batch& item : batch) {
      for (Sequence* sequence : item.get_sequences()) {
        if (sequence->has_linear_state_slot()) {
          EXPECT_EQ(sequence->kv_state().num_blocks(BlockType::LINEAR), 1u);
        }
      }
    }
    return ForwardOutput();
  }
  void update_last_step_result(std::vector<Batch>& batch) override {
    (void)batch;
  }
  const Tokenizer* tokenizer() const override { return fake_tokenizer_.get(); }
  BlockManagerPool* block_manager_pool() const override {
    return fake_block_manager_.get();
  }
  const ModelArgs& model_args() const override {
    static ModelArgs args;
    return args;
  }
  const TokenizerArgs& tokenizer_args() const override {
    static TokenizerArgs args;
    return args;
  }
  std::vector<int64_t> get_active_activation_memory() const override {
    return {};
  }
  bool init() override { return true; }

 private:
  std::unique_ptr<Tokenizer> fake_tokenizer_;
  std::unique_ptr<BlockManagerPool> fake_block_manager_;
};

template <typename T>
class ScopedConfigValue final {
 public:
  ScopedConfigValue(T& value, T new_value) : value_(value), old_(value) {
    value_ = new_value;
  }

  ~ScopedConfigValue() { value_ = old_; }

 private:
  T& value_;
  T old_;
};

ContinuousScheduler::Options CreateOptions(
    int32_t max_tokens_per_batch = 10000,
    int32_t max_seqs_per_batch = 256,
    int32_t dp_size = 1,
    bool enable_schedule_overlap = false,
    int32_t rec_worker_max_concurrency = 1) {
  ContinuousScheduler::Options opt;
  opt.max_tokens_per_batch_ = max_tokens_per_batch;
  opt.max_seqs_per_batch_ = max_seqs_per_batch;
  opt.dp_size_ = dp_size;
  opt.enable_schedule_overlap_ = enable_schedule_overlap;
  opt.rec_worker_max_concurrency_ = rec_worker_max_concurrency;
  opt.max_tokens_per_chunk_for_prefill_ = 1024;
  opt.num_speculative_tokens_ = 0;
  return opt;
}

std::vector<std::shared_ptr<Request>> GenRequests(
    const std::vector<int32_t>& prompt_lens,
    const std::vector<int32_t>& max_tokens,
    RecType rec_type,
    int32_t max_context_len = 30000) {
  std::vector<std::shared_ptr<Request>> requests;
  EXPECT_EQ(prompt_lens.size(), max_tokens.size());
  for (size_t i = 0; i < prompt_lens.size(); ++i) {
    std::vector<int32_t> prompt_token_ids(prompt_lens[i], 0);
    RequestSamplingParam sampling_param;
    SchedulerParam scheduler_param;
    scheduler_param.offline = false;
    scheduler_param.priority = RequestPriority::NORMAL;
    StoppingChecker stopping_checker;
    stopping_checker.set_max_generated_tokens(max_tokens[i]);
    stopping_checker.set_max_context_len(max_context_len);
    stopping_checker.set_ignore_eos(true);
    RequestState req_state("x",
                           prompt_token_ids,
                           sampling_param,
                           scheduler_param,
                           stopping_checker,
                           static_cast<size_t>(prompt_lens[i]) + 30000,
                           1,
                           1,
                           false,
                           false,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr);
    req_state.rec_type = rec_type;
    auto request =
        std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
    requests.emplace_back(request);
  }
  return requests;
}

}  // namespace

TEST(FixedStepsSchedulerTest, AddRequestSuccess) {
  auto engine = std::make_unique<FakeEngine>(32, 32);
  auto opt = CreateOptions();
  FixedStepsScheduler scheduler(engine.get(), opt);
  auto requests = GenRequests({64}, {10}, RecType::kOneRec);
  std::shared_ptr<Request> req = requests[0];
  EXPECT_TRUE(scheduler.add_request(req));
}

TEST(FixedStepsSchedulerTest, PrepareBatchEmptyWhenNoRequests) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  auto engine = std::make_unique<FakeEngine>(32, 32);
  auto opt = CreateOptions();
  FixedStepsScheduler scheduler(engine.get(), opt);
  ContinuousScheduler* base = &scheduler;
  std::vector<Batch> batches = base->prepare_batch_test();
  EXPECT_FALSE(batches.empty());
  EXPECT_TRUE(batches[0].empty());
}

TEST(FixedStepsSchedulerTest, ReportsBlockCountsInsteadOfRankCount) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  auto engine = std::make_unique<FakeEngine>(32, 32);
  int32_t dp_rank = -1;
  auto held_blocks =
      engine->block_manager_pool()->allocate(/*num_tokens=*/64, dp_rank);
  ASSERT_EQ(dp_rank, 0);
  ASSERT_EQ(held_blocks.size(), 2u);
  auto opt = CreateOptions();
  FixedStepsScheduler scheduler(engine.get(), opt);
  ContinuousScheduler* base = &scheduler;
  base->prepare_batch_test();

  const std::vector<size_t> free_blocks =
      engine->block_manager_pool()->num_free_blocks();
  const std::vector<size_t> used_blocks =
      engine->block_manager_pool()->num_used_blocks();
  const std::vector<size_t> prefix_blocks =
      engine->block_manager_pool()->num_blocks_in_prefix_cache();
  ASSERT_EQ(free_blocks.size(), 1u);
  ASSERT_EQ(used_blocks.size(), 1u);
  ASSERT_EQ(prefix_blocks.size(), 1u);
  EXPECT_GT(free_blocks[0], 1u);
  EXPECT_GT(used_blocks[0], 1u);
  EXPECT_DOUBLE_EQ(GAUGE_num_free_blocks.get_value(),
                   static_cast<double>(free_blocks[0]));
  EXPECT_DOUBLE_EQ(GAUGE_num_used_blocks.get_value(),
                   static_cast<double>(used_blocks[0]));
  EXPECT_DOUBLE_EQ(GAUGE_num_blocks_in_prefix_cache.get_value(),
                   static_cast<double>(prefix_blocks[0]));
}

TEST(FixedStepsSchedulerTest, ReportsBlockCountsAcrossMultipleRanks) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  auto engine = std::make_unique<FakeEngine>(/*num_blocks=*/32,
                                             /*block_size=*/32,
                                             /*linear_state=*/false,
                                             /*dp_size=*/2,
                                             /*enable_prefix_cache=*/true);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();
  int32_t lightly_loaded_rank = -1;
  auto lightly_held_blocks =
      block_manager_pool->allocate(/*num_tokens=*/64, lightly_loaded_rank);
  ASSERT_EQ(lightly_held_blocks.size(), 2u);
  int32_t heavily_loaded_rank = -1;
  auto heavily_held_blocks =
      block_manager_pool->allocate(/*num_tokens=*/192, heavily_loaded_rank);
  ASSERT_EQ(heavily_held_blocks.size(), 6u);
  ASSERT_NE(lightly_loaded_rank, heavily_loaded_rank);

  auto cached_requests = GenRequests({64}, {10}, RecType::kNone);
  Sequence* cached_sequence = cached_requests[0]->sequences()[0].get();
  ASSERT_EQ(cached_sequence->num_tokens(), 64u);
  cached_sequence->set_dp_rank(heavily_loaded_rank);
  ASSERT_TRUE(block_manager_pool->allocate(cached_sequence,
                                           cached_sequence->num_tokens()));
  cached_sequence->kv_state().set_kv_cache_tokens_num(
      cached_sequence->num_tokens());
  block_manager_pool->deallocate(cached_sequence);

  auto opt = CreateOptions(/*max_tokens_per_batch=*/10000,
                           /*max_seqs_per_batch=*/256,
                           /*dp_size=*/2);
  FixedStepsScheduler scheduler(engine.get(), opt);
  ContinuousScheduler* base = &scheduler;
  base->prepare_batch_test();

  const std::vector<size_t> free_blocks = block_manager_pool->num_free_blocks();
  const std::vector<size_t> used_blocks = block_manager_pool->num_used_blocks();
  const std::vector<size_t> prefix_blocks =
      block_manager_pool->num_blocks_in_prefix_cache();
  ASSERT_EQ(free_blocks.size(), 2u);
  ASSERT_EQ(used_blocks.size(), 2u);
  ASSERT_EQ(prefix_blocks.size(), 2u);
  EXPECT_GT(free_blocks[lightly_loaded_rank], free_blocks[heavily_loaded_rank]);
  EXPECT_LT(used_blocks[lightly_loaded_rank], used_blocks[heavily_loaded_rank]);
  EXPECT_EQ(prefix_blocks[lightly_loaded_rank], 0u);
  EXPECT_EQ(prefix_blocks[heavily_loaded_rank], 2u);
  EXPECT_DOUBLE_EQ(GAUGE_num_free_blocks.get_value(),
                   static_cast<double>(free_blocks[lightly_loaded_rank]));
  EXPECT_DOUBLE_EQ(GAUGE_num_used_blocks.get_value(),
                   static_cast<double>(used_blocks[lightly_loaded_rank]));
  EXPECT_DOUBLE_EQ(GAUGE_num_blocks_in_prefix_cache.get_value(),
                   static_cast<double>(prefix_blocks[lightly_loaded_rank]));

  block_manager_pool->reset_prefix_cache();
}

TEST(FixedStepsSchedulerTest, PrepareBatchOneRecSchedulesRequest) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  auto opt = CreateOptions(10000, 256);
  FixedStepsScheduler scheduler(engine.get(), opt);
  auto requests = GenRequests({64, 64}, {10, 10}, RecType::kOneRec);
  for (auto& req : requests) {
    scheduler.add_request(req);
  }
  ContinuousScheduler* base = &scheduler;
  std::vector<Batch> batches = base->prepare_batch_test();
  EXPECT_FALSE(batches.empty());
  bool has_non_empty = false;
  for (const auto& b : batches) {
    if (!b.empty()) {
      has_non_empty = true;
      break;
    }
  }
  EXPECT_TRUE(has_non_empty);
  EXPECT_EQ(base->get_running_requests().size(), 2u);
}

TEST(FixedStepsSchedulerTest, PrepareBatchRespectsTokenBudget) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  auto opt = CreateOptions(50, 1);
  FixedStepsScheduler scheduler(engine.get(), opt);
  auto requests = GenRequests({40, 40}, {10, 10}, RecType::kOneRec);
  for (auto& req : requests) {
    scheduler.add_request(req);
  }
  ContinuousScheduler* base = &scheduler;
  base->prepare_batch_test();
  EXPECT_LE(base->get_running_requests().size(), 1u);
}

TEST(FixedStepsSchedulerTest, StepCompletesWithRequest) {
  ScopedConfigValue<bool> prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), false);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      1.0);
  for (const RecType rec_type : {RecType::kOneRec, RecType::kLlmRec}) {
    const bool linear_state = rec_type == RecType::kLlmRec;
    auto engine = std::make_unique<FakeEngine>(64, 32, linear_state);
    auto opt = CreateOptions(10000, 256);
    FixedStepsScheduler scheduler(engine.get(), opt);
    auto requests = GenRequests({32}, {10}, rec_type);
    scheduler.add_request(requests[0]);
    EXPECT_NO_THROW(scheduler.step(absl::Milliseconds(500)));
    EXPECT_EQ(requests[0]->sequences()[0]->has_linear_state_slot(),
              linear_state);
  }
}

}  // namespace xllm
