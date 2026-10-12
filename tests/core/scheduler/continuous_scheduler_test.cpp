#include "continuous_scheduler.h"

#include <absl/time/clock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "distributed_runtime/engine.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "scheduler_factory.h"
#include "util/utils.h"

namespace xllm {

namespace {
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

class ControllablePrefetchBlockManagerPool final : public BlockManagerPool {
 public:
  explicit ControllablePrefetchBlockManagerPool(const Options& options)
      : BlockManagerPool(options, /*dp_size=*/1) {}

  void prefetch_from_storage(std::shared_ptr<Request> request,
                             PrefetchDoneCallback done) override {
    ++prefetch_calls_;
    if (prefetch_ready_) {
      done(std::move(request));
      return;
    }
    pending_.emplace_back(std::move(request), std::move(done));
  }

  void set_prefetch_ready(bool ready) {
    prefetch_ready_ = ready;
    if (!prefetch_ready_) {
      return;
    }

    auto pending = std::move(pending_);
    pending_.clear();
    for (auto& [request, done] : pending) {
      done(std::move(request));
    }
  }

  size_t prefetch_calls() const { return prefetch_calls_; }

 private:
  bool prefetch_ready_ = true;
  size_t prefetch_calls_ = 0;
  std::vector<std::pair<std::shared_ptr<Request>, PrefetchDoneCallback>>
      pending_;
};

class FakeTokenizer : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const {
    NOT_IMPLEMENTED();
  }
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const {
    return "";
  }
  std::optional<int32_t> token_to_id(const std::string_view& token) const {
    NOT_IMPLEMENTED();
  }
  std::string id_to_token(int32_t id) const { NOT_IMPLEMENTED(); }
  size_t vocab_size() const { NOT_IMPLEMENTED(); }
  std::unique_ptr<Tokenizer> clone() const {
    return std::make_unique<FakeTokenizer>();
  }
};

class FakeEngine : public Engine {
 public:
  FakeEngine(int32_t num_blocks,
             int32_t block_size,
             bool enable_prefix_cache = false,
             bool enable_linear_attention = false,
             InstanceRole instance_role = InstanceRole::DEFAULT)
      : prefix_cache_config_(
            KVCacheConfig::get_instance().enable_prefix_cache(),
            enable_prefix_cache) {
    BlockManagerPool::Options opt;
    opt.num_blocks_ = num_blocks;
    opt.block_size_ = block_size;
    opt.max_seqs_per_batch_ = 1024;
    opt.enable_linear_state_ = enable_linear_attention;
    opt.instance_is_decode(instance_role == InstanceRole::DECODE);
    if (enable_linear_attention) {
      model_args_.layer_types({"linear_attention"});
      // The unified linear-state slot pool needs a positive physical capacity;
      // size it generously so tests never hit slot pressure.
      opt.linear_state_num_slots_ = num_blocks + 2;
    }
    runtime::Options options;
    options.enable_prefix_cache(enable_prefix_cache)
        .instance_role(instance_role);
    configure_prefix_cache(options);
    opt.enable_prefix_cache(options.enable_prefix_cache());
    fake_tokenizer_ = std::make_unique<FakeTokenizer>();
    fake_block_manager_ =
        std::make_unique<ControllablePrefetchBlockManagerPool>(opt);
  }
  ForwardOutput step(std::vector<Batch>& batch) { return {}; }
  void update_last_step_result(std::vector<Batch>& batch) { NOT_IMPLEMENTED(); }
  const Tokenizer* tokenizer() const { return fake_tokenizer_.get(); }
  BlockManagerPool* block_manager_pool() const {
    return fake_block_manager_.get();
  }
  const ModelArgs& model_args() const { return model_args_; }
  const TokenizerArgs& tokenizer_args() const { NOT_IMPLEMENTED(); }
  std::vector<int64_t> get_active_activation_memory() const { return {0}; }
  bool init() override { return true; }

  using Engine::configure_prefix_cache;

  void set_model_args(ModelArgs model_args) {
    model_args_ = std::move(model_args);
  }

  void set_prefetch_ready(bool ready) {
    fake_block_manager_->set_prefetch_ready(ready);
  }

  size_t prefetch_calls() const {
    return fake_block_manager_->prefetch_calls();
  }

 private:
  ScopedConfigValue<bool> prefix_cache_config_;
  std::unique_ptr<Tokenizer> fake_tokenizer_;
  std::unique_ptr<ControllablePrefetchBlockManagerPool> fake_block_manager_;
  ModelArgs model_args_;
};

class TestContinuousScheduler final : public ContinuousScheduler {
 public:
  TestContinuousScheduler(Engine* engine, const Options& options)
      : ContinuousScheduler(engine, options) {}

  void reject_stream(const std::shared_ptr<Request>& request) {
    response_processor_->process_stream_request(request);
    response_processor_->wait_completion();
  }

  void reject_streams(const std::vector<std::shared_ptr<Request>>& requests) {
    std::vector<std::shared_ptr<Request>> request_copy = requests;
    response_processor_->batch_process_stream_requests(request_copy);
    response_processor_->wait_completion();
  }

  size_t scheduler_queue_size() { return request_queue_.size(); }

  bool enqueue_direct(std::shared_ptr<Request> request) {
    return request_queue_.write(std::move(request));
  }

  bool dequeue_direct(std::shared_ptr<Request>& request) {
    return request_queue_.read(request);
  }

  uint64_t queue_write_count() const { return request_queue_.writeCount(); }

  void wait_for_responses() { response_processor_->wait_completion(); }

  bool in_batch_prefix_cache_enabled() const {
    return enable_in_batch_prefix_cache_;
  }
};

ContinuousScheduler::Options create_scheduler_options(
    int32_t max_tokens_per_batch,
    int32_t max_seqs_per_batch,
    int32_t num_speculative_tokens,
    int32_t max_tokens_per_chunk_for_prefill,
    int32_t dp_size,
    const std::string& priority_strategy = "fcfs",
    bool enable_profile_kv_blocks = true,
    bool enable_latency_aware_schedule = false,
    int32_t max_global_ttft_ms = std::numeric_limits<int32_t>::max(),
    int32_t max_global_tpot_ms = std::numeric_limits<int32_t>::max()) {
  ContinuousScheduler::Options opt;
  opt.num_speculative_tokens_ = num_speculative_tokens;
  opt.max_tokens_per_chunk_for_prefill_ = max_tokens_per_chunk_for_prefill;
  opt.max_tokens_per_batch_ = max_tokens_per_batch;
  opt.max_seqs_per_batch_ = max_seqs_per_batch;
  opt.dp_size_ = dp_size;
  opt.priority_strategy_ = priority_strategy;
  opt.enable_profile_kv_blocks_ = enable_profile_kv_blocks;
  opt.enable_latency_aware_schedule_ = enable_latency_aware_schedule;
  opt.max_global_ttft_ms_ = max_global_ttft_ms;
  opt.max_global_tpot_ms_ = max_global_tpot_ms;
  return opt;
}

TEST(ContinuousSchedulerBatchModeTest, DcpControlsMixedBatches) {
  ScopedConfigValue<bool> mix_batch(
      SchedulerConfig::get_instance().enable_mix_batch(), true);
  ScopedConfigValue<int32_t> kv_split_size(
      ParallelConfig::get_instance().kv_split_size(), 2);
  ContinuousScheduler::Options options = create_scheduler_options(
      10000, 256, /*num_speculative_tokens=*/0, 1024, /*dp_size=*/1);
  options.enable_chunked_prefill(true);

  const BatchMode mode = resolve_batch_mode(options);
  EXPECT_FALSE(mode.enable_mix_batch);
}

std::vector<std::shared_ptr<Request>> generate_request(
    const std::vector<int32_t>& prompt_lens,
    const std::vector<int32_t>& max_tokens,
    std::optional<std::vector<bool>> offlines,
    std::optional<std::vector<int32_t>> priorities,
    std::optional<std::vector<int32_t>> ns,
    std::optional<std::vector<int32_t>> beam_widths,
    int32_t max_context_len) {
  std::vector<std::shared_ptr<Request>> requests;
  EXPECT_TRUE(prompt_lens.size() == max_tokens.size());

  size_t batch_size = prompt_lens.size();
  std::vector<bool> offline_vec;
  std::vector<int32_t> priority_vec;
  if (offlines.has_value()) {
    offline_vec = *offlines;
  } else {
    offline_vec = std::vector<bool>(batch_size, false);
  }

  if (priorities.has_value()) {
    priority_vec = priorities.value();
  } else {
    priority_vec = std::vector<int32_t>(
        batch_size, static_cast<int32_t>(RequestPriority::NORMAL));
  }

  std::vector<int32_t> n_vec;
  std::vector<int32_t> beam_width_vec;
  if (ns.has_value()) {
    n_vec = *ns;
  } else {
    n_vec = std::vector<int32_t>(batch_size, 1);
  }
  if (beam_widths.has_value()) {
    beam_width_vec = *beam_widths;
  } else {
    beam_width_vec = std::vector<int32_t>(batch_size, 0);
  }

  for (size_t i = 0; i < batch_size; ++i) {
    std::vector<int32_t> prompt_token_ids;
    prompt_token_ids.resize(prompt_lens[i]);
    RequestSamplingParam sampling_param;
    sampling_param.beam_width = beam_width_vec[i];
    SchedulerParam scheduler_param;
    scheduler_param.offline = offline_vec[i];
    scheduler_param.priority = static_cast<RequestPriority>(priority_vec[i]);

    StoppingChecker stopping_checker;
    stopping_checker.set_max_generated_tokens(max_tokens[i]);
    stopping_checker.set_max_context_len(max_context_len);
    stopping_checker.set_ignore_eos(true);

    RequestState req_state("x",
                           prompt_token_ids,
                           sampling_param,
                           scheduler_param,
                           stopping_checker,
                           prompt_lens[i] + 30000,
                           n_vec[i],
                           1,
                           false,
                           false,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr);

    auto request =
        std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
    requests.emplace_back(request);
  }

  return requests;
}

std::shared_ptr<Request> generate_request_with_prompt_tokens(
    const std::vector<int32_t>& prompt_token_ids,
    int32_t max_tokens,
    int32_t max_context_len) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(max_tokens);
  stopping_checker.set_max_context_len(max_context_len);
  stopping_checker.set_ignore_eos(true);

  RequestState req_state("x",
                         prompt_token_ids,
                         sampling_param,
                         scheduler_param,
                         stopping_checker,
                         prompt_token_ids.size() + 30000,
                         1,
                         1,
                         false,
                         false,
                         false,
                         false,
                         false,
                         nullptr,
                         nullptr);

  return std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
}

std::shared_ptr<Request> generate_request_with_best_of(
    const std::vector<int32_t>& prompt_token_ids,
    int32_t max_tokens,
    int32_t max_context_len,
    size_t n,
    size_t best_of) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(max_tokens);
  stopping_checker.set_max_context_len(max_context_len);
  stopping_checker.set_ignore_eos(true);

  RequestState req_state("x",
                         prompt_token_ids,
                         sampling_param,
                         scheduler_param,
                         stopping_checker,
                         prompt_token_ids.size() + 30000,
                         n,
                         best_of,
                         false,
                         false,
                         false,
                         false,
                         false,
                         nullptr,
                         nullptr);

  return std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
}

// dont not consider speculative decoding.
void update_requests(std::vector<std::shared_ptr<Request>> requests) {
  for (auto req : requests) {
    for (auto& seq : req->sequences()) {
      if (seq->kv_state().kv_cache_tokens_num() == 0) {
        seq->kv_state().incr_kv_cache_tokens_num(seq->num_prompt_tokens());
      } else {
        seq->kv_state().incr_kv_cache_tokens_num(1);
      }
      Token token(1);
      seq->append_token(token);
    }
  }
}

void make_request_decode_ready(const std::shared_ptr<Request>& request) {
  for (auto& seq : request->sequences()) {
    seq->kv_state().set_kv_cache_tokens_num(seq->num_prompt_tokens());
    Token token(1);
    seq->append_token(token);
  }
}

void set_chunk_kv(const std::shared_ptr<Request>& request, size_t kv_tokens) {
  for (auto& seq : request->sequences()) {
    seq->kv_state().set_kv_cache_tokens_num(kv_tokens);
  }
}

class OverlapDrainEngine final : public FakeEngine {
 public:
  OverlapDrainEngine() : FakeEngine(32, 4) {}

  void update_last_step_result(std::vector<Batch>& batch) override {
    ++completed_batches_;
  }

  size_t completed_batches() const { return completed_batches_; }

 private:
  size_t completed_batches_ = 0;
};

class CancelledOverlapEngine final : public FakeEngine {
 public:
  explicit CancelledOverlapEngine(bool linear_state)
      : FakeEngine(32, 4, false, linear_state) {}

  ForwardOutput step(std::vector<Batch>& batches) override {
    for (Batch& batch : batches) {
      if (batch.empty()) {
        continue;
      }
      batch.prepare_forward_input(1, 0, model_args());
      RawForwardOutput output;
      for (Sequence* sequence : batch.get_sequences()) {
        RawSampleOutput sample;
        RawToken token;
        token.id = -1;
        sample.tokens.emplace_back(std::move(token));
        output.outputs.emplace_back(std::move(sample));
        EXPECT_FALSE(sequence->cancelled());
      }
      batch.process_sample_output(output, false);
    }
    return {};
  }

  void update_last_step_result(std::vector<Batch>& batches) override {
    ++completed_batches_;
    for (Batch& batch : batches) {
      RawForwardOutput output;
      for (Sequence* sequence : batch.get_sequences()) {
        EXPECT_TRUE(sequence->cancelled());
        EXPECT_FALSE(sequence->has_any_blocks());
        EXPECT_FALSE(sequence->has_linear_state_slot());
        EXPECT_EQ(sequence->kv_cache_tokens_num(), 0u);
        RawSampleOutput sample;
        RawToken token;
        token.id = 17;
        sample.tokens.emplace_back(std::move(token));
        output.outputs.emplace_back(std::move(sample));
      }
      batch.process_sample_output(output, true);
    }
  }

  size_t completed_batches() const { return completed_batches_; }

 private:
  size_t completed_batches_ = 0;
};

class BlockedOverlapScheduler final : public ContinuousScheduler {
 public:
  BlockedOverlapScheduler(Engine* engine,
                          const Options& options,
                          Sequence* sequence)
      : ContinuousScheduler(engine, options) {
    last_batch_.emplace_back(sequence);
    is_first_step_ = false;
  }

  size_t prepare_calls() const { return prepare_calls_; }

 protected:
  std::vector<Batch> prepare_batch() override {
    ++prepare_calls_;
    return std::vector<Batch>(1);
  }

  bool if_queue_not_empty() override { return prepare_calls_ < 3; }

 private:
  size_t prepare_calls_ = 0;
};

}  // namespace

TEST(EnginePrefixCacheTest, DecodeLinearKeepsKvPrefixesAndReceiving) {
  ScopedConfigValue<bool> global_prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  ScopedConfigValue<bool> xtensor(
      KVCacheConfig::get_instance().enable_xtensor(), false);
  FakeEngine engine(32, 4, true, true, InstanceRole::DECODE);
  BlockManagerPool* pool = engine.block_manager_pool();
  EXPECT_TRUE(pool->options().enable_prefix_cache());
  EXPECT_TRUE(KVCacheConfig::get_instance().enable_prefix_cache());
  const std::vector<size_t> initial_free_blocks = pool->num_free_blocks();

  const std::vector<int32_t> prompt{1, 2, 3, 4, 5, 6, 7, 8};
  auto first_request = generate_request_with_prompt_tokens(prompt, 4, 32);
  Sequence* first_sequence = first_request->sequences().front().get();
  ASSERT_TRUE(pool->try_allocate(first_sequence));
  EXPECT_EQ(first_sequence->kv_cache_tokens_num(), prompt.size());
  ASSERT_GT(first_sequence->kv_state().num_blocks(BlockType::KV), 0u);
  ASSERT_EQ(first_sequence->kv_state().num_blocks(BlockType::LINEAR), 1u);
  const int32_t received_state = first_sequence->get_linear_state_slot_id();
  ASSERT_GE(received_state, 0);
  pool->cache(first_sequence, prompt.size());
  pool->cache(first_sequence);
  EXPECT_GT(pool->num_blocks_in_prefix_cache().front(), 0u);

  auto second_request = generate_request_with_prompt_tokens(prompt, 4, 32);
  Sequence* second_sequence = second_request->sequences().front().get();
  pool->allocate_shared(second_sequence);
  EXPECT_TRUE(second_sequence->has_any_blocks());
  EXPECT_GT(second_sequence->kv_cache_tokens_num(), 0u);
  ASSERT_TRUE(pool->try_allocate(second_sequence));
  EXPECT_GT(second_sequence->kv_state().shared_tokens_num(), 0u);
  EXPECT_EQ(second_sequence->kv_state().blocks(BlockType::KV).front().id(),
            first_sequence->kv_state().blocks(BlockType::KV).front().id());
  EXPECT_NE(second_sequence->get_linear_state_slot_id(), received_state);

  pool->deallocate(first_sequence);
  pool->deallocate(second_sequence);
  EXPECT_EQ(pool->num_free_blocks(),
            std::vector<size_t>{initial_free_blocks.front() - 2});
  EXPECT_GT(pool->num_blocks_in_prefix_cache().front(), 0u);
  EXPECT_TRUE(first_sequence->is_prefill_stage());
  pool->allocate_shared(first_sequence);
  EXPECT_TRUE(first_sequence->has_any_blocks());
  EXPECT_GT(first_sequence->kv_cache_tokens_num(), 0u);
  ASSERT_TRUE(pool->allocate(first_sequence, 4));
  EXPECT_GT(first_sequence->kv_state().shared_tokens_num(), 0u);
  pool->deallocate(first_sequence);
  EXPECT_EQ(pool->num_free_blocks(),
            std::vector<size_t>{initial_free_blocks.front() - 2});
}

TEST(EnginePrefixCacheTest, LinearPrefillAndMixedRolesKeepRequestedSetting) {
  ScopedConfigValue<bool> xtensor(
      KVCacheConfig::get_instance().enable_xtensor(), false);
  FakeEngine engine(32, 4, false, true);
  for (const InstanceRole role :
       {InstanceRole::DEFAULT, InstanceRole::PREFILL, InstanceRole::MIX}) {
    SCOPED_TRACE(role.to_string());
    for (const bool requested : {false, true}) {
      runtime::Options options;
      options.enable_prefix_cache(requested).instance_role(role);
      engine.configure_prefix_cache(options);
      EXPECT_EQ(options.enable_prefix_cache(), requested);
      EXPECT_EQ(KVCacheConfig::get_instance().enable_prefix_cache(), requested);
    }
  }
}

TEST(EnginePrefixCacheTest, ImplicitLinearLayersKeepDecodePrefixCache) {
  ScopedConfigValue<bool> xtensor(
      KVCacheConfig::get_instance().enable_xtensor(), false);
  FakeEngine engine(32, 4);
  ModelArgs model_args;
  model_args.full_attention_interval(4);
  engine.set_model_args(std::move(model_args));
  runtime::Options options;
  options.enable_prefix_cache(true).instance_role(InstanceRole::DECODE);
  engine.configure_prefix_cache(options);
  EXPECT_TRUE(options.enable_prefix_cache());
  EXPECT_TRUE(KVCacheConfig::get_instance().enable_prefix_cache());
}

TEST(EnginePrefixCacheTest, FullAttentionDecodeKeepsRequestedSetting) {
  ScopedConfigValue<bool> xtensor(
      KVCacheConfig::get_instance().enable_xtensor(), false);
  FakeEngine engine(32, 4);
  for (const bool requested : {false, true}) {
    runtime::Options options;
    options.enable_prefix_cache(requested).instance_role(InstanceRole::DECODE);
    engine.configure_prefix_cache(options);
    EXPECT_EQ(options.enable_prefix_cache(), requested);
    EXPECT_EQ(KVCacheConfig::get_instance().enable_prefix_cache(), requested);
  }
}

TEST(EnginePrefixCacheTest, MainEngineSynchronizesGlobalPrefixCache) {
  ScopedConfigValue<bool> xtensor(
      KVCacheConfig::get_instance().enable_xtensor(), false);
  FakeEngine engine(32, 4);
  for (const bool requested : {false, true}) {
    for (const bool global_enabled : {false, true}) {
      ScopedConfigValue<bool> global_prefix_cache(
          KVCacheConfig::get_instance().enable_prefix_cache(), global_enabled);
      runtime::Options options;
      options.enable_prefix_cache(requested);
      engine.configure_prefix_cache(options);
      EXPECT_EQ(options.enable_prefix_cache(), requested);
      EXPECT_EQ(KVCacheConfig::get_instance().enable_prefix_cache(), requested);
    }
  }
}

TEST(EnginePrefixCacheTest, DraftPrefixCacheDoesNotChangeGlobalConfig) {
  ScopedConfigValue<bool> xtensor(
      KVCacheConfig::get_instance().enable_xtensor(), false);
  FakeEngine engine(32, 4);
  ModelArgs model_args;
  model_args.full_attention_interval(4);
  engine.set_model_args(std::move(model_args));
  for (const bool global_enabled : {false, true}) {
    ScopedConfigValue<bool> global_prefix_cache(
        KVCacheConfig::get_instance().enable_prefix_cache(), global_enabled);
    for (const InstanceRole role :
         {InstanceRole::DEFAULT, InstanceRole::DECODE}) {
      runtime::Options options;
      options.enable_prefix_cache(true).is_draft_engine(true).instance_role(
          role);
      engine.configure_prefix_cache(options);
      EXPECT_TRUE(options.enable_prefix_cache());
      EXPECT_EQ(KVCacheConfig::get_instance().enable_prefix_cache(),
                global_enabled);
    }
  }
}

TEST(EnginePrefixCacheTest, FullAttentionDecodeStillReusesKvPrefixes) {
  ScopedConfigValue<bool> xtensor(
      KVCacheConfig::get_instance().enable_xtensor(), false);
  FakeEngine engine(32, 4, true, false, InstanceRole::DECODE);
  BlockManagerPool* pool = engine.block_manager_pool();
  const std::vector<int32_t> prompt{1, 2, 3, 4, 5, 6, 7, 8};
  auto first_request = generate_request_with_prompt_tokens(prompt, 4, 32);
  Sequence* first_sequence = first_request->sequences().front().get();
  ASSERT_TRUE(pool->try_allocate(first_sequence));
  pool->cache(first_sequence);
  EXPECT_GT(pool->num_blocks_in_prefix_cache().front(), 0u);

  auto second_request = generate_request_with_prompt_tokens(prompt, 4, 32);
  Sequence* second_sequence = second_request->sequences().front().get();
  ASSERT_TRUE(pool->try_allocate(second_sequence));
  EXPECT_GT(second_sequence->kv_state().shared_tokens_num(), 0u);
  EXPECT_EQ(second_sequence->kv_state().blocks(BlockType::KV).front().id(),
            first_sequence->kv_state().blocks(BlockType::KV).front().id());
  pool->deallocate(first_sequence);
  pool->deallocate(second_sequence);
}

TEST(EnginePrefixCacheTest, KPoolDoesNotDisableDecodePrefixCache) {
  ScopedConfigValue<bool> xtensor(
      KVCacheConfig::get_instance().enable_xtensor(), false);
  FakeEngine engine(32, 4);
  ModelArgs model_args;
  model_args.model_type("deepseek_v4")
      .layer_types({"deepseek_sparse_attention"})
      .compress_ratios({0, 4, 128});
  engine.set_model_args(std::move(model_args));
  runtime::Options options;
  options.enable_prefix_cache(true).instance_role(InstanceRole::DECODE);
  engine.configure_prefix_cache(options);
  EXPECT_TRUE(options.enable_prefix_cache());
}

TEST(EnginePrefixCacheTest, XTensorStillDisablesPrefixCache) {
  ScopedConfigValue<bool> xtensor(
      KVCacheConfig::get_instance().enable_xtensor(), true);
  FakeEngine engine(32, 4, true);
  runtime::Options options;
  options.enable_prefix_cache(true);
  engine.configure_prefix_cache(options);
  EXPECT_FALSE(options.enable_prefix_cache());
  EXPECT_FALSE(KVCacheConfig::get_instance().enable_prefix_cache());
}

TEST(EnginePrefixCacheTest, MtpDraftKeepsSharedCacheLayoutConfiguration) {
  ScopedConfigValue<bool> xtensor(
      KVCacheConfig::get_instance().enable_xtensor(), false);
  FakeEngine engine(32, 4);
  runtime::Options draft_options;
  draft_options.enable_prefix_cache(true)
      .is_draft_engine(true)
      .speculative_algorithm("mTp");
  engine.configure_prefix_cache(draft_options);
  EXPECT_TRUE(draft_options.enable_prefix_cache());
  runtime::Options target_options;
  target_options.enable_prefix_cache(true).speculative_algorithm("MTP");
  engine.configure_prefix_cache(target_options);
  EXPECT_TRUE(target_options.enable_prefix_cache());
}

TEST(EnginePrefixCacheTest, DecodeLinearSupportsHostPrefixCacheAndStore) {
  ScopedConfigValue<bool> xtensor(
      KVCacheConfig::get_instance().enable_xtensor(), false);
  FakeEngine engine(32, 4, true, true, InstanceRole::DECODE);
  HostCacheValidationOptions options;
  options.host_blocks_factor = 2.0;
  options.device_block_count = 32;
  options.supports_host_kv_offload = true;
  options.enable_prefix_cache =
      engine.block_manager_pool()->options().enable_prefix_cache();
  options.enable_disagg_pd = true;
  options.instance_role = InstanceRole::DECODE;
  options.has_conv_cache_shape = true;
  options.has_ssm_cache_shape = true;
  for (const bool enable_store : {false, true}) {
    options.enable_kvcache_store = enable_store;
    const std::optional<std::string> error =
        validate_host_cache_options(options);
    EXPECT_FALSE(error.has_value()) << error.value_or("");
  }
}

TEST(ContinuousSchedulerTest, InBatchPrefixCacheUsesGlobalConfig) {
  ScopedConfigValue<bool> xtensor(
      KVCacheConfig::get_instance().enable_xtensor(), false);
  for (const bool global_enabled : {false, true}) {
    FakeEngine engine(32, 4, global_enabled);
    EXPECT_EQ(KVCacheConfig::get_instance().enable_prefix_cache(),
              global_enabled);
    for (const bool in_batch_enabled : {false, true}) {
      ScopedConfigValue<bool> in_batch_prefix_cache(
          KVCacheConfig::get_instance().enable_in_batch_prefix_cache(),
          in_batch_enabled);
      ContinuousScheduler::Options options =
          create_scheduler_options(16, 4, 0, 4, 1);
      TestContinuousScheduler scheduler(&engine, options);
      EXPECT_EQ(scheduler.in_batch_prefix_cache_enabled(),
                global_enabled && in_batch_enabled);
    }
  }
}

TEST(ContinuousSchedulerTest, DecodeLinearEnablesInBatchPrefixReuse) {
  ScopedConfigValue<bool> global_prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  ScopedConfigValue<bool> in_batch_prefix_cache(
      KVCacheConfig::get_instance().enable_in_batch_prefix_cache(), true);
  ScopedConfigValue<bool> xtensor(
      KVCacheConfig::get_instance().enable_xtensor(), false);
  FakeEngine engine(32, 4, true, true, InstanceRole::DECODE);
  ContinuousScheduler::Options options =
      create_scheduler_options(16, 4, 0, 4, 1);
  options.instance_role(InstanceRole::DECODE).enable_disagg_pd(true);
  TestContinuousScheduler scheduler(&engine, options);
  EXPECT_TRUE(scheduler.in_batch_prefix_cache_enabled());

  const std::vector<int32_t> prompt{1, 2, 3, 4, 5, 6, 7, 8};
  auto first_request = generate_request_with_prompt_tokens(prompt, 4, 32);
  auto second_request = generate_request_with_prompt_tokens(prompt, 4, 32);
  ASSERT_TRUE(scheduler.add_request(first_request));
  ASSERT_TRUE(scheduler.add_request(second_request));
  auto batches = scheduler.prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches.front().size(), 2u);
  Sequence* first_sequence = first_request->sequences().front().get();
  Sequence* second_sequence = second_request->sequences().front().get();
  EXPECT_EQ(first_sequence->kv_state().shared_tokens_num(), 0u);
  EXPECT_GT(second_sequence->kv_state().shared_tokens_num(), 0u);
  EXPECT_EQ(second_sequence->kv_state().blocks(BlockType::KV).front().id(),
            first_sequence->kv_state().blocks(BlockType::KV).front().id());
  EXPECT_EQ(engine.block_manager_pool()->num_blocks_in_prefix_cache(),
            std::vector<size_t>{2});
}

TEST(ContinuousSchedulerTest, DrainsInflightBatchBeforeRetryingBlockedQueue) {
  auto request = generate_request_with_prompt_tokens({1, 2, 3, 4}, 4, 30000);
  OverlapDrainEngine engine;
  ContinuousScheduler::Options options =
      create_scheduler_options(1024, 16, 0, 1024, 1);
  options.enable_schedule_overlap(true);
  BlockedOverlapScheduler scheduler(
      &engine, options, request->sequences().front().get());

  scheduler.step(absl::ZeroDuration());

  EXPECT_EQ(scheduler.prepare_calls(), 1u);
  EXPECT_EQ(engine.completed_batches(), 1u);
}

TEST(ContinuousSchedulerTest, CancelledOverlapRequestReleasesBeforeResult) {
  ScopedConfigValue<bool> overlap(
      SchedulerConfig::get_instance().enable_schedule_overlap(), true);
  for (const bool linear_state : {false, true}) {
    SCOPED_TRACE(linear_state);
    CancelledOverlapEngine engine(linear_state);
    ContinuousScheduler::Options options =
        create_scheduler_options(16, 4, 0, 4, 1);
    options.enable_schedule_overlap(true);
    TestContinuousScheduler scheduler(&engine, options);
    auto request = generate_request_with_prompt_tokens({1, 2, 3, 4}, 8, 32);
    request->state().enable_schedule_overlap = true;
    request->state().output_func = [](const RequestOutput&) { return true; };
    const size_t initial_free_blocks =
        engine.block_manager_pool()->num_free_blocks().front();
    scheduler.add_request(request);
    scheduler.step(absl::ZeroDuration());
    Sequence* sequence = request->sequences().front().get();
    ASSERT_TRUE(sequence->has_any_blocks());
    ASSERT_EQ(engine.completed_batches(), 0u);
    const std::vector<int32_t> tokens(sequence->tokens());
    request->set_cancel();
    scheduler.step(absl::ZeroDuration());
    EXPECT_EQ(engine.completed_batches(), 1u);
    EXPECT_EQ(std::vector<int32_t>(sequence->tokens()), tokens);
    EXPECT_EQ(sequence->kv_cache_tokens_num(), 0u);
    EXPECT_EQ(engine.block_manager_pool()->num_free_blocks().front(),
              initial_free_blocks);
    scheduler.wait_for_responses();
  }
}

// ===========================================================================
// M11.5 scheduler-overlap prework: block-lifecycle ordering when a request
// finishes inside the deferred output processing of an overlapped step.
//
// The timeline this pins (step_with_schedule_overlap, one step scheduled in
// advance):
//   step N    : batch B_N dispatched while B_{N-1} runs on the device; the
//               deferred processing of B_{N-1} commits the request's last
//               real token and the request finishes.
//   step N+1 A: collect_finished deallocates the finished request's blocks
//               BEFORE the engine collects B_N's result (C).
//   step N+1 C: update_last_step_result(B_N) still dereferences the request's
//               sequences -- the scheduler's last_running_requests_ snapshot
//               (copied at step N's tail) keeps the request object alive
//               until this collection completes.
//
// The early block free is safe for a non-DFlash2 overlap step because every
// device access to KV/LINEAR cache tensors is enqueued on the per-worker
// compute stream in step-task order (single-threaded WorkerImpl task pool,
// worker_impl.h; LLMWorkerImpl::step_for_schedule_overlap chains the deferred
// linear-state restore and the step on compute_stream_), so a freed block
// reallocated into the next batch is only written by device work ordered
// after the in-flight step. That is why LLMEngine::step's
// retain_cache_blocks guard is DFlash2-only (see
// llm_engine_host_transfer_test for the engine-side pin): DFlash2's
// prelaunch enqueues KV writes one additional step ahead, outside this FIFO
// window. A CP prefill lifting the glm5_next overlap ban (M11.6) relies on
// exactly this ordering, since the engine broadcasts one batch to every CP
// rank of the DP group.
// ===========================================================================

class OverlapFinishEngine final : public FakeEngine {
 public:
  OverlapFinishEngine() : FakeEngine(32, 4) {}

  ForwardOutput step(std::vector<Batch>& batches) override {
    ++step_calls_;
    for (Batch& batch : batches) {
      if (batch.empty()) {
        continue;
      }
      batch.prepare_forward_input(1, 0, model_args());
      RawForwardOutput output;
      for (Sequence* sequence : batch.get_sequences()) {
        // The batch being dispatched still references live blocks.
        EXPECT_TRUE(sequence->has_any_blocks());
        RawSampleOutput sample;
        RawToken token;
        // Stage-1 append under overlap: the driver's fake output is a
        // negative placeholder (worker_service.cpp builds arange(-1, ...)),
        // so the sequence stores a fake token for the next step's input.
        token.id = -1;
        sample.tokens.emplace_back(std::move(token));
        output.outputs.emplace_back(std::move(sample));
      }
      batch.process_sample_output(output, false);
    }
    return {};
  }

  void update_last_step_result(std::vector<Batch>& batches) override {
    ++completed_batches_;
    for (Batch& batch : batches) {
      if (batch.empty()) {
        continue;
      }
      RawForwardOutput output;
      for (Sequence* sequence : batch.get_sequences()) {
        if (completed_batches_ == 1u) {
          // First deferred collection: the request has not been collected as
          // finished yet, so its blocks are still allocated.
          EXPECT_TRUE(sequence->has_any_blocks());
        } else {
          // Later deferred collections: collect_finished of THIS step
          // already deallocated the finished request's blocks before the
          // in-flight batch's result is collected here. The sequence object
          // itself must still be alive -- last_running_requests_ keeps the
          // request alive until this collection completes.
          EXPECT_EQ(sequence->kv_cache_tokens_num(), 0u);
          EXPECT_FALSE(sequence->has_any_blocks());
        }
        RawSampleOutput sample;
        RawToken token;
        token.id = 5;
        sample.tokens.emplace_back(std::move(token));
        output.outputs.emplace_back(std::move(sample));
      }
      batch.process_sample_output(output, true);
    }
  }

  size_t step_calls() const { return step_calls_; }
  size_t completed_batches() const { return completed_batches_; }

 private:
  size_t step_calls_ = 0;
  size_t completed_batches_ = 0;
};

TEST(ContinuousSchedulerTest, OverlapFinishedRequestOutlivesInflightResult) {
  ScopedConfigValue<bool> overlap(
      SchedulerConfig::get_instance().enable_schedule_overlap(), true);
  OverlapFinishEngine engine;
  ContinuousScheduler::Options options =
      create_scheduler_options(16, 4, 0, 4, 1);
  options.enable_schedule_overlap(true);
  TestContinuousScheduler scheduler(&engine, options);
  // max_tokens=1: the request finishes as soon as the deferred processing
  // commits its first real generated token. Sequence params copy
  // enable_schedule_overlap at Request construction, so the flag must be in
  // the RequestState (update_last_step_token CHECKs it).
  const std::vector<int32_t> prompt{1, 2, 3, 4};
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1);
  stopping_checker.set_max_context_len(32);
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
                         /*enable_schedule_overlap=*/true,
                         /*output_func=*/nullptr,
                         /*outputs_func=*/nullptr);
  auto request =
      std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
  request->state().output_func = [](const RequestOutput&) { return true; };
  const std::weak_ptr<Request> watch = request;
  const size_t initial_free_blocks =
      engine.block_manager_pool()->num_free_blocks().front();
  scheduler.add_request(request);
  request.reset();

  // Step 1: prefill dispatched; its (fake) output appends the placeholder.
  scheduler.step(absl::ZeroDuration());
  // Step 2: decode dispatched one step ahead; the prefill's deferred result
  // commits the real token and finishes the request.
  scheduler.step(absl::ZeroDuration());
  // Step 3: collect_finished frees the finished request's blocks, THEN the
  // engine collects the in-flight decode batch's result (assertions inside
  // OverlapFinishEngine::update_last_step_result pin both halves).
  scheduler.step(absl::ZeroDuration());

  EXPECT_EQ(engine.step_calls(), 2u);
  EXPECT_EQ(engine.completed_batches(), 2u);
  scheduler.wait_for_responses();
  // The scheduler no longer holds the request: the deferred pipeline drained
  // and the last_running_requests_ snapshot was replaced.
  EXPECT_TRUE(watch.expired());
  EXPECT_EQ(engine.block_manager_pool()->num_free_blocks().front(),
            initial_free_blocks);
}

TEST(ContinuousSchedulerFactoryTest,
     ChunkedPrefillWithoutSPCreatesContinuousScheduler) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  opt.enable_chunked_prefill() = true;

  auto engine = std::make_unique<FakeEngine>(32, 32);
  auto scheduler = create_continuous_scheduler(engine.get(), opt);

  // All non-PD paths now create ContinuousScheduler with BatchMode routing.
  EXPECT_NE(dynamic_cast<ContinuousScheduler*>(scheduler.get()), nullptr);
}

TEST(ContinuousSchedulerTest,
     AdmissionRejectsMultipleCandidatesForLinearModels) {
  const std::vector<std::pair<int32_t, size_t>> sampling_modes = {
      {0, 1}, {1, 1}, {2, 1}, {0, 2}};
  for (const bool linear_state : {false, true}) {
    for (const auto& [beam_width, best_of] : sampling_modes) {
      SCOPED_TRACE(::testing::Message() << "linear_state=" << linear_state
                                        << ", beam_width=" << beam_width);
      FakeEngine engine(32, 4, false, linear_state);
      ContinuousScheduler::Options options =
          create_scheduler_options(16, 4, 0, 4, 1);
      TestContinuousScheduler scheduler(&engine, options);
      auto request = generate_request_with_prompt_tokens({1, 2, 3, 4}, 8, 32);
      request->state().sampling_param.beam_width = beam_width;
      request->state().best_of = best_of;
      int32_t callback_count = 0;
      std::optional<Status> callback_status;
      request->state().output_func = [&](const RequestOutput& output) {
        ++callback_count;
        callback_status = output.status;
        return true;
      };
      const size_t free_blocks =
          engine.block_manager_pool()->num_free_blocks().front();
      EXPECT_TRUE(scheduler.add_request(request));
      scheduler.wait_for_responses();
      if (linear_state && (beam_width > 1 || best_of > 1)) {
        EXPECT_EQ(callback_count, 1);
        ASSERT_TRUE(callback_status.has_value());
        EXPECT_EQ(callback_status->code(), StatusCode::INVALID_ARGUMENT);
        EXPECT_EQ(callback_status->message(),
                  "Beam search and best_of > 1 are not supported with LINEAR "
                  "attention.");
        EXPECT_EQ(scheduler.scheduler_queue_size(), 0u);
        EXPECT_EQ(engine.prefetch_calls(), 0u);
      } else {
        EXPECT_EQ(callback_count, 0);
        EXPECT_EQ(scheduler.scheduler_queue_size(), 1u);
        EXPECT_EQ(engine.prefetch_calls(), 1u);
      }
      EXPECT_EQ(engine.block_manager_pool()->num_free_blocks().front(),
                free_blocks);
      EXPECT_FALSE(request->sequences().front()->has_any_blocks());
      EXPECT_FALSE(request->sequences().front()->has_linear_state_slot());
    }
  }
}

TEST(ContinuousSchedulerTest, PrefetchCompletesBeforeSchedulerQueueAdmission) {
  ContinuousScheduler::Options options =
      create_scheduler_options(64, 4, 0, 64, 1);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  engine->set_prefetch_ready(false);
  auto scheduler =
      std::make_unique<TestContinuousScheduler>(engine.get(), options);
  std::shared_ptr<Request> request =
      generate_request({8},
                       {4},
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/30000)[0];

  ASSERT_TRUE(scheduler->add_request(request));
  EXPECT_EQ(engine->prefetch_calls(), 1u);
  EXPECT_EQ(scheduler->scheduler_queue_size(), 0u);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 1u);

  std::vector<Batch> batches = scheduler->prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  EXPECT_TRUE(batches.front().empty());
  EXPECT_EQ(scheduler->scheduler_queue_size(), 0u);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 1u);

  engine->set_prefetch_ready(true);
  batches = scheduler->prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches.front().size(), 1u);
  EXPECT_EQ(batches.front()[0], request->sequences().front().get());
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 0u);
  EXPECT_EQ(scheduler->scheduler_queue_size(), 0u);
}

TEST(ContinuousSchedulerTest,
     CancelledPendingPrefetchReleasesAdmissionWithoutEnqueue) {
  ContinuousScheduler::Options options =
      create_scheduler_options(64, 4, 0, 64, 1);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  engine->set_prefetch_ready(false);
  auto scheduler =
      std::make_unique<TestContinuousScheduler>(engine.get(), options);
  std::shared_ptr<Request> request =
      generate_request({8},
                       {4},
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/30000)[0];

  ASSERT_TRUE(scheduler->add_request(request));
  ASSERT_EQ(scheduler->get_waiting_requests_num(), 1u);
  request->set_cancel();

  std::vector<Batch> batches = scheduler->prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  EXPECT_TRUE(batches.front().empty());
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 1u);
  EXPECT_EQ(scheduler->scheduler_queue_size(), 0u);

  engine->set_prefetch_ready(true);
  batches = scheduler->prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  EXPECT_TRUE(batches.front().empty());
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 0u);
  EXPECT_EQ(scheduler->scheduler_queue_size(), 0u);
}

TEST(ContinuousSchedulerTest, QueueCapacityRejectsBeforePrefetchStarts) {
  ScopedConfigValue<int32_t> request_queue_size(
      RecConfig::get_instance().request_queue_size(), 1);
  ContinuousScheduler::Options options =
      create_scheduler_options(64, 4, 0, 64, 1);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  engine->set_prefetch_ready(false);
  auto scheduler =
      std::make_unique<TestContinuousScheduler>(engine.get(), options);
  std::vector<std::shared_ptr<Request>> requests =
      generate_request({8, 8},
                       {4, 4},
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/30000);

  ASSERT_TRUE(scheduler->add_request(requests[0]));
  EXPECT_EQ(engine->prefetch_calls(), 1u);
  EXPECT_EQ(scheduler->scheduler_queue_size(), 0u);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 1u);

  EXPECT_FALSE(scheduler->add_request(requests[1]));
  EXPECT_EQ(engine->prefetch_calls(), 1u);
  engine->set_prefetch_ready(true);
  EXPECT_EQ(scheduler->scheduler_queue_size(), 1u);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 0u);
}

TEST(ContinuousSchedulerTest,
     CompletedPrefetchWaitsWhenAnotherProducerFillsQueue) {
  ScopedConfigValue<int32_t> request_queue_size(
      RecConfig::get_instance().request_queue_size(), 1);
  ContinuousScheduler::Options options =
      create_scheduler_options(64, 4, 0, 64, 1);
  auto engine = std::make_unique<FakeEngine>(64, 32);
  engine->set_prefetch_ready(false);
  auto scheduler =
      std::make_unique<TestContinuousScheduler>(engine.get(), options);
  std::vector<std::shared_ptr<Request>> requests =
      generate_request({8, 8},
                       {4, 4},
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                       /*max_context_len=*/30000);

  ASSERT_TRUE(scheduler->add_request(requests[0]));
  ASSERT_TRUE(scheduler->enqueue_direct(requests[1]));

  auto completion = std::async(std::launch::async,
                               [&engine] { engine->set_prefetch_ready(true); });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (scheduler->queue_write_count() < 2 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  EXPECT_EQ(scheduler->queue_write_count(), 2u);
  EXPECT_EQ(completion.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler->dequeue_direct(queued));
  EXPECT_EQ(queued, requests[1]);
  ASSERT_EQ(completion.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  completion.get();

  ASSERT_TRUE(scheduler->dequeue_direct(queued));
  EXPECT_EQ(queued, requests[0]);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 0u);
}

TEST(ContinuousSchedulerFactoryTest,
     ChunkedPrefillWithSPCreatesContinuousScheduler) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  opt.enable_chunked_prefill() = true;

  auto engine = std::make_unique<FakeEngine>(32, 32);
  auto scheduler = create_continuous_scheduler(engine.get(), opt);

  // All non-PD paths now create ContinuousScheduler with BatchMode routing.
  EXPECT_NE(dynamic_cast<ContinuousScheduler*>(scheduler.get()), nullptr);
}

TEST(ContinuousSchedulerFactoryTest,
     ChunkedPrefillWithSPAndSpeculativeCreatesContinuousScheduler) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 4, 1024, 1);
  opt.enable_chunked_prefill() = true;

  auto engine = std::make_unique<FakeEngine>(32, 32);
  auto scheduler = create_continuous_scheduler(engine.get(), opt);

  // All non-PD paths now create ContinuousScheduler with BatchMode routing.
  EXPECT_NE(dynamic_cast<ContinuousScheduler*>(scheduler.get()), nullptr);
}

TEST(ContinuousSchedulerFactoryTest,
     ChunkedPrefillWithSPDoesNotBuildMixedBatch) {
  ContinuousScheduler::Options opt = create_scheduler_options(8, 8, 0, 4, 1);
  opt.enable_chunked_prefill() = true;
  opt.cp_size() = 2;  // CP > 1 forces exclusive batch (no mix)

  auto engine = std::make_unique<FakeEngine>(32, 32);
  auto scheduler = create_continuous_scheduler(engine.get(), opt);
  ASSERT_NE(scheduler.get(), nullptr);

  auto requests = generate_request({2, 10},
                                   {8, 8},
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   30000);
  for (auto& req : requests) {
    scheduler->add_request(req);
  }

  auto batches = scheduler->prepare_batch_test();
  ASSERT_EQ(batches.size(), 1);
  ASSERT_EQ(batches[0].size(), 2);
  const auto& allowed_max_tokens = batches[0].get_allowed_max_tokens();
  ASSERT_EQ(allowed_max_tokens.size(), 2);

  make_request_decode_ready(requests[0]);
  set_chunk_kv(requests[1], allowed_max_tokens[1]);

  batches = scheduler->prepare_batch_test();
  ASSERT_EQ(batches.size(), 1);
  ASSERT_EQ(batches[0].size(), 1);

  const auto forward_input =
      batches[0].prepare_forward_input(1, 0, ModelArgs());
  EXPECT_TRUE(
      forward_input.input_params.meta.batch_forward_type.is_chunked_prefill());
  EXPECT_FALSE(forward_input.input_params.meta.batch_forward_type.is_mixed());
  EXPECT_EQ(forward_input.input_params.meta.num_sequences, 1);
  EXPECT_EQ(batches[0].get_allowed_max_tokens()[0],
            opt.max_tokens_per_chunk_for_prefill());
}

TEST(SchedulerFactoryTest, DisaggPDChunkedPrefillUsesDisaggPD) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 2, 1024, 1);
  opt.enable_disagg_pd() = true;
  opt.enable_pd_ooc() = false;
  opt.enable_chunked_prefill() = true;

  EXPECT_EQ(select_scheduler_kind(opt), SchedulerKind::DISAGG_PD);
}

TEST(SchedulerFactoryTest, DisaggPDOOCKeepsPDOOCKind) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  opt.enable_disagg_pd() = true;
  opt.enable_pd_ooc() = true;
  opt.enable_chunked_prefill() = true;

  EXPECT_EQ(select_scheduler_kind(opt), SchedulerKind::PD_OOC);
}

TEST(ContinuousSchedulerTest, BeamStrictNoPartialScheduling) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(2, 8, 0, 1024, 1, "fcfs");
  auto engine = std::make_unique<FakeEngine>(64, 32);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  auto req1 = generate_request({64},
                               {10},
                               std::nullopt,
                               std::nullopt,
                               std::vector<int32_t>{1},
                               std::vector<int32_t>{0},
                               30000)[0];
  make_request_decode_ready(req1);

  auto beam_req = generate_request({64},
                                   {10},
                                   std::nullopt,
                                   std::nullopt,
                                   std::vector<int32_t>{1},
                                   std::vector<int32_t>{2},
                                   30000)[0];
  // Manually duplicate beam sequence to simulate a decoded beam group.
  beam_req->sequences().emplace_back(
      std::make_unique<Sequence>(*beam_req->sequences()[0]));
  ASSERT_EQ(beam_req->sequences().size(), 2u);
  make_request_decode_ready(beam_req);

  scheduler->add_request(req1);
  scheduler->add_request(beam_req);
  auto batch = scheduler->prepare_batch_test();
  ASSERT_EQ(batch.size(), 1u);
  EXPECT_EQ(batch[0].size(), 1u);
  EXPECT_EQ(batch[0][0], req1->sequences()[0].get());
  EXPECT_EQ(scheduler->get_running_requests().size(), 1u);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 0u);
  EXPECT_NE(batch[0][0], beam_req->sequences()[0].get());
  EXPECT_NE(batch[0][0], beam_req->sequences()[1].get());
}

TEST(BlockManagerPoolTest, AllocateFailureRollsBackSharedPrefixBlocks) {
  auto engine = std::make_unique<FakeEngine>(3, 4, true);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();

  auto cached_request =
      generate_request_with_prompt_tokens({1, 2, 3, 4, 5, 6, 7, 8}, 1, 30000);
  auto failed_request = generate_request_with_prompt_tokens(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 1, 30000);
  auto later_request =
      generate_request_with_prompt_tokens({20, 21, 22, 23}, 1, 30000);

  auto* cached_sequence = cached_request->sequences()[0].get();
  ASSERT_TRUE(block_manager_pool->allocate(cached_sequence,
                                           cached_sequence->num_tokens()));
  cached_sequence->kv_state().set_kv_cache_tokens_num(
      cached_sequence->num_tokens());
  block_manager_pool->deallocate(cached_sequence);

  const size_t free_blocks_before_failure =
      util::max(block_manager_pool->num_free_blocks());
  const size_t used_blocks_before_failure =
      util::min(block_manager_pool->num_used_blocks());
  EXPECT_EQ(free_blocks_before_failure, 0);

  auto* failed_sequence = failed_request->sequences()[0].get();
  EXPECT_FALSE(block_manager_pool->allocate(failed_sequence,
                                            failed_sequence->num_tokens()));
  EXPECT_EQ(failed_sequence->kv_state().num_blocks(BlockType::KV), 0);
  EXPECT_EQ(failed_sequence->kv_state().shared_blocks_num(BlockType::KV), 0);
  EXPECT_EQ(util::max(block_manager_pool->num_free_blocks()),
            free_blocks_before_failure);
  EXPECT_EQ(util::min(block_manager_pool->num_used_blocks()),
            used_blocks_before_failure);

  auto* later_sequence = later_request->sequences()[0].get();
  EXPECT_TRUE(block_manager_pool->allocate(later_sequence,
                                           later_sequence->num_tokens()));
  EXPECT_EQ(later_sequence->kv_state().num_blocks(BlockType::KV), 1);

  (void)engine.release();
}

TEST(ContinuousSchedulerTest,
     BlockExhaustionDefersRequestAfterCurrentBatchHasWork) {
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      2.0);
  auto engine = std::make_unique<FakeEngine>(
      /*num_blocks=*/4, /*block_size=*/4);
  ContinuousScheduler::Options options =
      create_scheduler_options(/*max_tokens_per_batch=*/16,
                               /*max_seqs_per_batch=*/2,
                               /*num_speculative_tokens=*/0,
                               /*max_tokens_per_chunk_for_prefill=*/8,
                               /*dp_size=*/1);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), options);

  auto scheduled_request =
      generate_request_with_prompt_tokens({1, 2, 3, 4, 5, 6, 7, 8}, 1, 30000);
  auto deferred_request = generate_request_with_prompt_tokens(
      {9, 10, 11, 12, 13, 14, 15, 16}, 1, 30000);
  ASSERT_TRUE(scheduler->add_request(scheduled_request));
  ASSERT_TRUE(scheduler->add_request(deferred_request));

  std::vector<Batch> batches = scheduler->prepare_batch_test();
  ASSERT_EQ(batches.size(), 1u);
  ASSERT_EQ(batches.front().size(), 1u);
  EXPECT_EQ(batches.front()[0], scheduled_request->sequences()[0].get());
  EXPECT_EQ(scheduler->get_running_requests().size(), 1u);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 1u);
  EXPECT_FALSE(deferred_request->finished());
  EXPECT_EQ(
      deferred_request->sequences()[0]->kv_state().num_blocks(BlockType::KV),
      0u);

  (void)engine.release();
}

TEST(ContinuousSchedulerTest,
     PDDecodeBestOfNExpandsAndSharesPromptViaPrefixCache) {
  // Disagg PD decode instance flow:
  //   1. request arrives from the prefill instance with kv_cache_tokens_num
  //      already advanced (try_allocate + append_token(first_token)).
  //   2. ContinuousScheduler::prepare_batch must short-circuit the waiting
  //      queue and push directly into running_requests_.
  //   3. handle_running_requests must trigger expand_sequences(true) and
  //      cache(seq[0]) so the expanded seq[1..best_of-1] can hit the
  //      shared prompt blocks via prefix cache.
  ContinuousScheduler::Options opt =
      create_scheduler_options(1024, 16, 0, 1024, 1);
  auto engine =
      std::make_unique<FakeEngine>(32, 4, /*enable_prefix_cache=*/true);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();

  constexpr size_t kBestOf = 4;
  constexpr size_t kN = 2;
  auto request = generate_request_with_best_of({1, 2, 3, 4, 5, 6, 7, 8},
                                               /*max_tokens=*/4,
                                               /*max_context_len=*/30000,
                                               kN,
                                               kBestOf);

  Sequence* seq0 = request->sequences()[0].get();
  ASSERT_TRUE(block_manager_pool->try_allocate(seq0));
  ASSERT_EQ(seq0->kv_state().kv_cache_tokens_num(), seq0->num_prompt_tokens());
  Token first(42);
  seq0->append_token(first);

  scheduler->add_request(request);

  auto batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(request->sequences().size(), kBestOf);
  EXPECT_EQ(batch[0].size(), kBestOf);

  for (size_t i = 1; i < kBestOf; ++i) {
    EXPECT_GT(
        request->sequences()[i]->kv_state().shared_blocks_num(BlockType::KV),
        0u)
        << "expanded seq " << i
        << " should reuse seq[0] prompt blocks via prefix cache";
  }

  scheduler.reset();
  (void)engine.release();
}

TEST(ContinuousSchedulerTest, PDDecodeBestOfOneSkipsExpansionAndShares) {
  // Sanity check: best_of==n==1 should not trigger any expansion, and the
  // request should still flow through the PD-decode short-circuit.
  auto request = generate_request_with_best_of({1, 2, 3, 4, 5, 6, 7, 8},
                                               /*max_tokens=*/4,
                                               /*max_context_len=*/30000,
                                               /*n=*/1,
                                               /*best_of=*/1);
  Sequence* seq0 = request->sequences()[0].get();

  ContinuousScheduler::Options opt =
      create_scheduler_options(1024, 16, 0, 1024, 1);
  // Prefix cache is not under test here; disabling it avoids teardown putting
  // blocks into the prefix-cache table instead of the free list.
  auto engine =
      std::make_unique<FakeEngine>(32, 4, /*enable_prefix_cache=*/false);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();

  ASSERT_TRUE(block_manager_pool->try_allocate(seq0));
  ASSERT_EQ(seq0->kv_state().kv_cache_tokens_num(), seq0->num_prompt_tokens());
  Token first(42);
  seq0->append_token(first);

  scheduler->add_request(request);

  auto batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(request->sequences().size(), 1u);
  EXPECT_EQ(batch[0].size(), 1u);

  // prepare_batch may allocate extra decode blocks; release them before engine
  // teardown (BlockManagerImpl checks all blocks are on the free list).
  block_manager_pool->deallocate_without_cache(seq0);
  (void)engine.release();
}

TEST(ContinuousSchedulerTest, RejectedStreamCancelsAtSchedulingBoundary) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(1024, 16, 0, 1024, 1);
  opt.enable_schedule_overlap() = false;
  auto engine = std::make_unique<FakeEngine>(32, 4);
  auto scheduler = std::make_unique<TestContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();
  const size_t initial_free_blocks =
      util::max(block_manager_pool->num_free_blocks());

  auto request = generate_request_with_prompt_tokens({1, 2, 3, 4}, 4, 30000);
  request->state().stream = true;
  request->state().output_func = [](const RequestOutput&) { return false; };
  make_request_decode_ready(request);
  scheduler->add_request(request);

  std::vector<Batch> batch = scheduler->prepare_batch_test();
  ASSERT_EQ(batch.size(), 1u);
  ASSERT_EQ(batch[0].size(), 1u);
  EXPECT_LT(util::max(block_manager_pool->num_free_blocks()),
            initial_free_blocks);

  scheduler->reject_stream(request);

  EXPECT_FALSE(request->cancelled());

  scheduler->step(absl::ZeroDuration());

  EXPECT_TRUE(request->cancelled());
  EXPECT_TRUE(scheduler->get_running_requests().empty());
  EXPECT_EQ(util::max(block_manager_pool->num_free_blocks()),
            initial_free_blocks);

  scheduler->reject_stream(request);
  scheduler->step(absl::ZeroDuration());

  EXPECT_TRUE(request->cancelled());
  EXPECT_TRUE(scheduler->get_running_requests().empty());
  EXPECT_EQ(util::max(block_manager_pool->num_free_blocks()),
            initial_free_blocks);
}

TEST(ContinuousSchedulerTest, FailedStreamReturnsStatusExactlyOnce) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(1024, 16, 0, 1024, 1);
  opt.enable_schedule_overlap() = false;
  auto engine = std::make_unique<FakeEngine>(32, 4);
  auto scheduler = std::make_unique<TestContinuousScheduler>(engine.get(), opt);
  auto request = generate_request_with_prompt_tokens({1, 2, 3, 4}, 4, 30000);
  request->state().stream = true;
  make_request_decode_ready(request);
  scheduler->add_request(request);

  int32_t callback_count = 0;
  std::optional<Status> callback_status;
  request->state().output_func = [&](const RequestOutput& output) {
    ++callback_count;
    callback_status = output.status;
    return true;
  };
  std::vector<Batch> batch = scheduler->prepare_batch_test();
  ASSERT_EQ(batch.size(), 1u);
  ASSERT_EQ(batch[0].size(), 1u);

  request->sequences()[0]->fail(
      Status(StatusCode::UNKNOWN, "json_object row mismatch"));
  scheduler->process_batch_output_test(/*enable_schedule_overlap=*/false);
  scheduler->wait_for_responses();
  EXPECT_EQ(callback_count, 0);

  (void)scheduler->prepare_batch_test();
  scheduler->wait_for_responses();

  EXPECT_EQ(callback_count, 1);
  ASSERT_TRUE(callback_status.has_value());
  EXPECT_EQ(callback_status->code(), StatusCode::UNKNOWN);
  EXPECT_EQ(callback_status->message(), "json_object row mismatch");

  (void)scheduler->prepare_batch_test();
  scheduler->wait_for_responses();
  EXPECT_EQ(callback_count, 1);
}

TEST(ContinuousSchedulerTest, BatchRejectedStreamsCancelAtSchedulingBoundary) {
  ContinuousScheduler::Options opt =
      create_scheduler_options(1024, 16, 0, 1024, 1);
  opt.enable_schedule_overlap() = false;
  auto engine = std::make_unique<FakeEngine>(32, 4);
  auto scheduler = std::make_unique<TestContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();

  std::vector<std::shared_ptr<Request>> requests =
      generate_request({4, 4},
                       {4, 4},
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                       30000);
  for (std::shared_ptr<Request>& request : requests) {
    request->state().stream = true;
    request->state().output_func = [](const RequestOutput&) { return true; };
    make_request_decode_ready(request);
    scheduler->add_request(request);
  }
  requests[0]->state().outputs_func = [](const std::vector<RequestOutput>&) {
    return std::vector<bool>{false, true};
  };

  std::vector<Batch> batch = scheduler->prepare_batch_test();
  ASSERT_EQ(batch.size(), 1u);
  ASSERT_EQ(batch[0].size(), 2u);
  const size_t free_blocks_before_cancel =
      util::max(block_manager_pool->num_free_blocks());

  scheduler->reject_streams(requests);
  scheduler->reject_streams(requests);

  EXPECT_FALSE(requests[0]->cancelled());
  EXPECT_FALSE(requests[1]->cancelled());

  scheduler->step(absl::ZeroDuration());

  EXPECT_TRUE(requests[0]->cancelled());
  EXPECT_FALSE(requests[1]->cancelled());
  ASSERT_EQ(scheduler->get_running_requests().size(), 1u);
  EXPECT_EQ(scheduler->get_running_requests()[0], requests[1]);
  EXPECT_GT(util::max(block_manager_pool->num_free_blocks()),
            free_blocks_before_cancel);

  requests[1]->state().outputs_func = [](const std::vector<RequestOutput>&) {
    return std::vector<bool>{false};
  };
  scheduler->reject_streams({requests[1]});
  EXPECT_FALSE(requests[1]->cancelled());
  scheduler->step(absl::ZeroDuration());
  EXPECT_TRUE(requests[1]->cancelled());
  EXPECT_TRUE(scheduler->get_running_requests().empty());
}
// ============== Async RL training: Pause/Resume tests ==============

// TEST: pause()/resume() state transitions are correct and idempotent.
TEST(ContinuousSchedulerTest, PauseResumeStateTransition) {
  int block_num = 9;
  int block_size = 32;
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);

  EXPECT_FALSE(scheduler->is_paused());
  scheduler->pause();
  EXPECT_TRUE(scheduler->is_paused());
  scheduler->pause();  // idempotent
  EXPECT_TRUE(scheduler->is_paused());
  scheduler->resume();
  EXPECT_FALSE(scheduler->is_paused());
  scheduler->resume();  // idempotent
  EXPECT_FALSE(scheduler->is_paused());

  (void)engine.release();
}

// TEST: pause preempts all running requests, frees KV cache, moves them back
// to the waiting queue (vLLM-compatible semantics for RL).
TEST(ContinuousSchedulerTest, PausePreemptsRunningAndFreesKVCache) {
  int block_num = 33;
  int block_size = 32;
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();
  ASSERT_TRUE(scheduler != nullptr);

  auto requests = generate_request({127, 127},
                                   {10, 10},
                                   std::vector<bool>{false, false},
                                   std::vector<int32_t>{2, 2},
                                   std::nullopt,
                                   std::nullopt,
                                   30000);
  std::vector<std::shared_ptr<Request>> running_requests = requests;
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(batch[0].size(), 2);
  update_requests(running_requests);

  EXPECT_EQ(scheduler->get_running_requests().size(), 2);
  int free_blocks_before = util::max(block_manager_pool->num_free_blocks());

  scheduler->pause();
  scheduler->preempt_all_running_requests_test();

  int free_blocks_after = util::max(block_manager_pool->num_free_blocks());
  EXPECT_GT(free_blocks_after, free_blocks_before);
  EXPECT_EQ(scheduler->get_running_requests().size(), 0);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 2);

  (void)engine.release();
}

// TEST: after resume, preempted requests in the waiting queue get re-scheduled.
TEST(ContinuousSchedulerTest, ResumeReschedulesPreemptedRequests) {
  int block_num = 33;
  int block_size = 32;
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  ASSERT_TRUE(scheduler != nullptr);

  auto requests = generate_request({127, 127},
                                   {10, 10},
                                   std::vector<bool>{false, false},
                                   std::vector<int32_t>{2, 2},
                                   std::nullopt,
                                   std::nullopt,
                                   30000);
  std::vector<std::shared_ptr<Request>> running_requests = requests;
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  (void)scheduler->prepare_batch_test();
  update_requests(running_requests);
  EXPECT_EQ(scheduler->get_running_requests().size(), 2);

  scheduler->pause();
  scheduler->preempt_all_running_requests_test();
  EXPECT_EQ(scheduler->get_running_requests().size(), 0);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 2);

  scheduler->resume();
  EXPECT_FALSE(scheduler->is_paused());

  auto batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(batch[0].size(), 2);
  EXPECT_EQ(scheduler->get_running_requests().size(), 2);
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 0);

  (void)engine.release();
}

// TEST: ABORT mode cancels running requests, frees KV cache, and does NOT
// push them back to the waiting queue (clients must retry).
TEST(ContinuousSchedulerTest, PauseAbortCancelsRunningRequests) {
  int block_num = 33;
  int block_size = 32;
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000, 256, 0, 1024, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();
  ASSERT_TRUE(scheduler != nullptr);

  auto requests = generate_request({127, 127},
                                   {10, 10},
                                   std::vector<bool>{false, false},
                                   std::vector<int32_t>{2, 2},
                                   std::nullopt,
                                   std::nullopt,
                                   30000);
  std::vector<std::shared_ptr<Request>> running_requests = requests;
  for (auto req : requests) {
    // process_failed_request invokes the requests output callback; give it a
    // no-op so ABORT can notify completion without a real client.
    req->state().output_func = [](const RequestOutput&) { return true; };
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(batch[0].size(), 2);
  update_requests(running_requests);

  EXPECT_EQ(scheduler->get_running_requests().size(), 2);
  int free_blocks_before = util::max(block_manager_pool->num_free_blocks());

  scheduler->abort_all_running_requests_test();

  int free_blocks_after = util::max(block_manager_pool->num_free_blocks());
  EXPECT_GT(free_blocks_after, free_blocks_before);        // KV cache freed
  EXPECT_EQ(scheduler->get_running_requests().size(), 0);  // running cleared
  EXPECT_EQ(scheduler->get_waiting_requests_num(), 0);     // NOT requeued

  (void)engine.release();
}

// With in-batch prefix cache enabled, a request admitted earlier in the same
// scheduling step publishes its full prompt blocks, so a later request with the
// same prefix reuses them (shared_kv_blocks_num > 0). When disabled, the later
// request shares nothing within the same batch.
TEST(ContinuousSchedulerTest,
     InBatchCachePrefillBlocksIncreaseSharedBlocksForLaterRequests) {
  const auto run_with_in_batch_prefix_cache =
      [](bool enable_in_batch_prefix_cache) -> size_t {
    KVCacheConfig& kv_config = KVCacheConfig::get_instance();
    const bool saved_prefix_cache = kv_config.enable_prefix_cache();
    const bool saved_in_batch = kv_config.enable_in_batch_prefix_cache();
    kv_config.enable_prefix_cache(true);
    kv_config.enable_in_batch_prefix_cache(enable_in_batch_prefix_cache);

    ContinuousScheduler::Options opt =
        create_scheduler_options(1024, 16, 0, 1024, 1);
    auto engine =
        std::make_unique<FakeEngine>(32, 4, /*enable_prefix_cache=*/true);
    auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);

    auto first_request =
        generate_request_with_prompt_tokens({1, 2, 3, 4, 5, 6, 7, 8}, 1, 30000);
    auto second_request =
        generate_request_with_prompt_tokens({1, 2, 3, 4, 5, 6, 7, 8}, 1, 30000);
    scheduler->add_request(first_request);
    scheduler->add_request(second_request);

    auto batch = scheduler->prepare_batch_test();
    EXPECT_EQ(batch.size(), 1);
    EXPECT_EQ(batch[0].size(), 2);
    EXPECT_EQ(first_request->sequences()[0]->kv_state().shared_blocks_num(
                  BlockType::KV),
              0u);
    const size_t second_shared =
        second_request->sequences()[0]->kv_state().shared_blocks_num(
            BlockType::KV);

    scheduler.reset();
    // Leak the engine: cached blocks live in the prefix-cache table at
    // teardown.
    (void)engine.release();

    kv_config.enable_prefix_cache(saved_prefix_cache);
    kv_config.enable_in_batch_prefix_cache(saved_in_batch);
    return second_shared;
  };

  const size_t shared_when_enabled = run_with_in_batch_prefix_cache(true);
  const size_t shared_when_disabled = run_with_in_batch_prefix_cache(false);

  EXPECT_GT(shared_when_enabled, shared_when_disabled);
  EXPECT_GT(shared_when_enabled, 0u);
  EXPECT_EQ(shared_when_disabled, 0u);
}

// End-to-end check through the real scheduler path (add_request ->
// prepare_batch): two requests that share only the first 2 of 3 prompt blocks
// are admitted in the same step. With in-batch prefix cache on, the first
// request publishes its full blocks, so the second request reuses EXACTLY the 2
// shared blocks (the 3rd differs and must be a miss). The prefix-cache table
// also grows by the published blocks. With it off, nothing is shared in-batch.
TEST(ContinuousSchedulerTest, InBatchCacheReusesPartialPrefixWithinSameBatch) {
  // block_size = 4. 12 tokens => 3 full blocks per prompt.
  const std::vector<int32_t> prompt_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  // Shares blocks [0,1] (tokens 0..7) with prompt_a, differs in block 2.
  const std::vector<int32_t> prompt_b = {
      1, 2, 3, 4, 5, 6, 7, 8, 99, 100, 101, 102};

  struct Result {
    size_t first_shared = 0;
    size_t second_shared = 0;
    size_t batch_count = 0;
    size_t batch0_size = 0;
    size_t prefix_cache_blocks = 0;
  };

  const auto run = [&](bool enable_in_batch_prefix_cache) -> Result {
    KVCacheConfig& kv_config = KVCacheConfig::get_instance();
    const bool saved_prefix_cache = kv_config.enable_prefix_cache();
    const bool saved_in_batch = kv_config.enable_in_batch_prefix_cache();
    kv_config.enable_prefix_cache(true);
    kv_config.enable_in_batch_prefix_cache(enable_in_batch_prefix_cache);

    ContinuousScheduler::Options opt =
        create_scheduler_options(1024, 16, 0, 1024, 1);
    auto engine =
        std::make_unique<FakeEngine>(64, 4, /*enable_prefix_cache=*/true);
    auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);

    auto first_request =
        generate_request_with_prompt_tokens(prompt_a, 1, 30000);
    auto second_request =
        generate_request_with_prompt_tokens(prompt_b, 1, 30000);
    scheduler->add_request(first_request);
    scheduler->add_request(second_request);

    auto batch = scheduler->prepare_batch_test();

    Result r;
    r.batch_count = batch.size();
    r.batch0_size = batch.empty() ? 0 : batch[0].size();
    r.first_shared =
        first_request->sequences()[0]->kv_state().shared_blocks_num(
            BlockType::KV);
    r.second_shared =
        second_request->sequences()[0]->kv_state().shared_blocks_num(
            BlockType::KV);
    r.prefix_cache_blocks =
        engine->block_manager_pool()->num_blocks_in_prefix_cache()[0];

    scheduler.reset();
    // Leak the engine: published blocks live in the prefix-cache table at
    // teardown, which would otherwise trip the free-list check in dtor.
    (void)engine.release();

    kv_config.enable_prefix_cache(saved_prefix_cache);
    kv_config.enable_in_batch_prefix_cache(saved_in_batch);
    return r;
  };

  const Result enabled = run(true);
  const Result disabled = run(false);

  // Both requests must land in the same batch for in-batch reuse to apply.
  EXPECT_EQ(enabled.batch_count, 1u);
  EXPECT_EQ(enabled.batch0_size, 2u);

  // The first request can never share within the same step (nothing published
  // yet when it is admitted).
  EXPECT_EQ(enabled.first_shared, 0u);

  // The second request reuses exactly the 2 identical leading blocks.
  EXPECT_EQ(enabled.second_shared, 2u);

  // Prefix cache holds first request's 3 blocks plus the second request's one
  // distinct trailing block => 4 unique blocks.
  EXPECT_EQ(enabled.prefix_cache_blocks, 4u);

  // With the feature disabled, no in-batch sharing happens.
  EXPECT_EQ(disabled.first_shared, 0u);
  EXPECT_EQ(disabled.second_shared, 0u);
  EXPECT_GT(enabled.second_shared, disabled.second_shared);
}

// A newly scheduled prefill sequence that shares a long cached prefix must end
// up with enough kv capacity to cover its matched prefix plus the tokens it is
// asked to compute this step, even when the per-step token budget is clamped to
// fewer tokens than the prefix length. This is the invariant enforced by
// batch_input_builder.cpp (current_max_tokens_capacity >= kv + q_seq_len).
TEST(ContinuousSchedulerTest,
     PrefixHitUnderClampedBudgetKeepsCapacitySufficient) {
  constexpr int32_t kBlockSize = 8;
  constexpr int32_t kNumBlocks = 128;
  // 12 full blocks worth of shared prefix (96 tokens).
  constexpr int32_t kPrefixLen = 96;
  // Full prompt shares the whole prefix and appends a small unique tail.
  constexpr int32_t kPromptLen = 120;
  // Per-step token budget deliberately clamped below the prefix length.
  constexpr int32_t kMaxTokensPerBatch = 40;

  ScopedConfigValue<bool> enable_prefix_cache(
      KVCacheConfig::get_instance().enable_prefix_cache(), true);
  // Keep the prefill memory-usage guard from blocking the single sequence.
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      2.0);

  auto engine = std::make_unique<FakeEngine>(
      kNumBlocks, kBlockSize, /*enable_prefix_cache=*/true);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();

  // PrefillOnly behavior: enable_chunked_prefill=true, num_speculative_tokens>0
  // gives enable_mix_batch=false + enable_chunked_prefill=true.
  ContinuousScheduler::Options opt = create_scheduler_options(
      kMaxTokensPerBatch, 256, /*num_speculative_tokens=*/5, 1024, 1);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  ASSERT_TRUE(scheduler != nullptr);

  std::vector<int32_t> prefix_token_ids;
  prefix_token_ids.reserve(kPrefixLen);
  for (int32_t i = 0; i < kPrefixLen; ++i) {
    prefix_token_ids.emplace_back(i + 1);
  }
  // Seed the prefix cache.
  auto cached_request = generate_request_with_prompt_tokens(
      prefix_token_ids, /*max_tokens=*/1, /*max_context_len=*/30000);
  Sequence* cached_sequence = cached_request->sequences()[0].get();
  ASSERT_TRUE(block_manager_pool->allocate(cached_sequence,
                                           cached_sequence->num_tokens()));
  cached_sequence->kv_state().set_kv_cache_tokens_num(
      cached_sequence->num_tokens());
  block_manager_pool->deallocate(cached_sequence);

  std::vector<int32_t> prompt_token_ids = prefix_token_ids;
  prompt_token_ids.reserve(kPromptLen);
  for (int32_t i = kPrefixLen; i < kPromptLen; ++i) {
    prompt_token_ids.emplace_back(i + 1);
  }
  auto request = generate_request_with_prompt_tokens(
      prompt_token_ids, /*max_tokens=*/10, /*max_context_len=*/30000);
  scheduler->add_request(request);

  auto batch = scheduler->prepare_batch_test();
  ASSERT_EQ(batch.size(), 1);
  ASSERT_EQ(batch[0].size(), 1);

  auto running_requests = scheduler->get_running_requests();
  auto budgets = scheduler->get_running_sequences_budgets();
  ASSERT_EQ(running_requests.size(), 1);
  ASSERT_EQ(budgets.size(), 1);

  Sequence* sequence = running_requests[0]->sequences()[0].get();
  const size_t kv_cache_tokens = sequence->kv_state().kv_cache_tokens_num();
  // Precondition of the regression: the cached prefix was matched and it is
  // longer than the clamped per-step budget.
  ASSERT_EQ(kv_cache_tokens, static_cast<size_t>(kPrefixLen));
  ASSERT_LT(static_cast<size_t>(kMaxTokensPerBatch), kv_cache_tokens);

  const size_t q_seq_len =
      std::min(sequence->num_tokens() - kv_cache_tokens, budgets[0]);
  EXPECT_GE(sequence->kv_state().current_max_tokens_capacity(),
            kv_cache_tokens + q_seq_len);

  // Cached prefix blocks stay resident in the prefix-cache table, so leak the
  // engine to skip the block manager's "all blocks freed" teardown check.
  scheduler.reset();
  (void)engine.release();
}

}  // namespace xllm
