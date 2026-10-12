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
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/kv_cache/kv_cache_capacity.h"
#include "core/framework/kv_cache/kv_cache_shape.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/runtime/dflash_worker_impl.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/runtime/mtp_worker_impl.h"
#include "core/runtime/options.h"
#include "core/runtime/speculative_worker_impl.h"
#include "core/util/slice.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

constexpr int64_t kBlockCount = 2;
constexpr int64_t kBlockSize = 4;

ModelArgs make_model_args(const std::string& model_type,
                          int64_t layer_count,
                          int64_t head_dim) {
  ModelArgs model_args;
  model_args.model_type(model_type)
      .n_layers(layer_count)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(head_dim);
  return model_args;
}

KVCacheShape make_cache_shape(const ModelArgs& model_args) {
  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount).block_size(kBlockSize);
  return KVCacheShape(capacity, model_args, /*world_size=*/1);
}

KVCacheShape make_dsv4_cache_shape(const ModelArgs& model_args,
                                   int64_t c4_count,
                                   int64_t c128_count) {
  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount)
      .block_size(kBlockSize)
      .swa_count(kBlockCount)
      .c4_count(c4_count)
      .c128_count(c128_count);
  return KVCacheShape(capacity, model_args, /*world_size=*/1);
}

KVCacheCreateOptions make_create_options(const torch::Device& device,
                                         const ModelArgs& model_args) {
  KVCacheCreateOptions create_options;
  create_options.device(device)
      .dtype(torch::kFloat32)
      .num_layers(model_args.n_layers())
      .model_type(model_args.model_type())
      .block_size(kBlockSize)
      .head_dim(model_args.head_dim())
      .index_head_dim(std::max<int64_t>(model_args.index_head_dim(), 1))
      .window_size(std::max<int64_t>(model_args.window_size(), 1))
      .compress_ratios(model_args.compress_ratios());
  return create_options;
}

runtime::Options make_runtime_options(double host_blocks_factor) {
  runtime::Options options;
  options.model_id("test_target")
      .block_size(kBlockSize)
      .num_speculative_tokens(1)
      .host_blocks_factor(host_blocks_factor)
      .layers_wise_copy_batchs(1)
      .world_size(1)
      .dp_size(1)
      .cp_size(1);
  return options;
}

// B5 (implement.md, design.md §3.1 Gate B4): a scratch model directory whose
// config.json declares model_type so DFlashWorkerImpl's allow_cp check (which
// calls util::get_model_type on options.model_path()) can resolve a real
// model_type without loading actual weights.
// Writes a one-file model_type config into a UNIQUE temp directory and
// removes the directory on scope exit. The pre-fix helper returned a bare
// path from a fixed per-model_type name: two of these tests running in
// parallel (or one crashing) collided on or leaked the directory.
class ScopedModelTypeConfig {
 public:
  explicit ScopedModelTypeConfig(const std::string& model_type) {
    static std::atomic<uint64_t> counter{0};
    dir_ = std::filesystem::temp_directory_path() /
           ("xllm_dflash_allow_cp_test_" + model_type + "_" +
            std::to_string(::getpid()) + "_" +
            std::to_string(counter.fetch_add(1)));
    std::filesystem::create_directories(dir_);
    std::ofstream config_file(dir_ / "config.json");
    config_file << "{\"model_type\": \"" << model_type << "\"}";
  }

  ~ScopedModelTypeConfig() { std::filesystem::remove_all(dir_); }

  ScopedModelTypeConfig(const ScopedModelTypeConfig&) = delete;
  ScopedModelTypeConfig& operator=(const ScopedModelTypeConfig&) = delete;

  const std::filesystem::path& path() const { return dir_; }

 private:
  std::filesystem::path dir_;
};

class RecordingTransferWorker final : public LLMWorkerImpl {
 public:
  RecordingTransferWorker(const ParallelArgs& parallel_args,
                          const torch::Device& device,
                          const runtime::Options& options,
                          uint32_t transfer_result)
      : LLMWorkerImpl(parallel_args, device, options),
        transfer_result_(transfer_result) {}

  uint32_t transfer_kv_blocks(
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) override {
    last_batch_id_ = batch_id;
    last_transfer_size_ = block_transfer_info.size();
    ++vector_transfer_count_;
    return transfer_result_;
  }

  uint32_t transfer_kv_blocks(
      uint64_t batch_id,
      Slice<BlockTransferInfo>& block_transfer_info) override {
    last_batch_id_ = batch_id;
    last_transfer_size_ = block_transfer_info.size();
    ++slice_transfer_count_;
    return transfer_result_;
  }

  std::vector<uint8_t> prefetch_kv_blocks(
      Slice<BlockTransferInfo>& block_transfer_info,
      StoreGetStats* /*stats*/) override {
    ++prefetch_count_;
    return std::vector<uint8_t>(block_transfer_info.size(), /*value=*/1);
  }

  uint32_t vector_transfer_count() const { return vector_transfer_count_; }
  uint32_t slice_transfer_count() const { return slice_transfer_count_; }
  uint32_t prefetch_count() const { return prefetch_count_; }
  uint64_t last_batch_id() const { return last_batch_id_; }
  size_t last_transfer_size() const { return last_transfer_size_; }

 private:
  uint32_t transfer_result_ = 0;
  uint32_t vector_transfer_count_ = 0;
  uint32_t slice_transfer_count_ = 0;
  uint32_t prefetch_count_ = 0;
  uint64_t last_batch_id_ = 0;
  size_t last_transfer_size_ = 0;
};

class HierarchyTransferTestWorker : public LLMWorkerImpl {
 public:
  HierarchyTransferTestWorker(const ParallelArgs& parallel_args,
                              const torch::Device& device,
                              const runtime::Options& options,
                              const ModelArgs& model_args)
      : LLMWorkerImpl(parallel_args, device, options) {
    dtype_ = torch::kFloat32;
    context_ =
        ModelContext(parallel_args,
                     model_args,
                     QuantArgs(),
                     torch::TensorOptions().dtype(dtype_).device(device));
  }

  void initialize_hierarchy_cache(const KVCacheShape& cache_shape) {
    const ModelArgs& model_args = context_.get_model_args();
    const KVCacheCreateOptions create_options =
        make_create_options(device_.unwrap(), model_args);
    allocate_kv_caches(kv_caches_, cache_shape, create_options);
    init_hierarchy_kv_cache_transfer(cache_shape, create_options);
  }

  void mark_loaded() { status_ = WorkerImpl::Status::LOADED; }

  bool allocate_kv_cache(const KVCacheShape& cache_shape) override {
    initialize_hierarchy_cache(cache_shape);
    status_ = WorkerImpl::Status::READY;
    return true;
  }

  void fill_block(int64_t block_id, double value) {
    for (size_t layer_index = 0; layer_index < kv_caches_.size();
         ++layer_index) {
      const double layer_value = value + static_cast<double>(layer_index);
      kv_caches_[layer_index].get_k_cache()[block_id].fill_(layer_value);
      kv_caches_[layer_index].get_v_cache()[block_id].fill_(layer_value + 0.5);
    }
  }

  void zero_block(int64_t block_id) {
    for (KVCache& cache : kv_caches_) {
      cache.get_k_cache()[block_id].zero_();
      cache.get_v_cache()[block_id].zero_();
    }
  }

  bool blocks_equal(int64_t source_block_id,
                    int64_t destination_block_id) const {
    for (const KVCache& cache : kv_caches_) {
      if (!torch::equal(cache.get_k_cache()[source_block_id],
                        cache.get_k_cache()[destination_block_id]) ||
          !torch::equal(cache.get_v_cache()[source_block_id],
                        cache.get_v_cache()[destination_block_id])) {
        return false;
      }
    }
    return true;
  }

  void fill_block(BlockType block_type, int64_t block_id, double value) {
    for (KVCache& cache : kv_caches_) {
      BlockTypeTensorMap tensors = cache.get_block_type_tensors(block_type);
      for (auto& [role, tensor] : tensors) {
        tensor[block_id].fill_(value + static_cast<double>(role));
      }
    }
  }

  void zero_block(BlockType block_type, int64_t block_id) {
    for (KVCache& cache : kv_caches_) {
      BlockTypeTensorMap tensors = cache.get_block_type_tensors(block_type);
      for (auto& [role, tensor] : tensors) {
        tensor[block_id].zero_();
      }
    }
  }

  bool blocks_equal(BlockType block_type,
                    int64_t source_block_id,
                    int64_t destination_block_id) const {
    for (const KVCache& cache : kv_caches_) {
      const BlockTypeTensorMap tensors =
          cache.get_block_type_tensors(block_type);
      for (const auto& [role, tensor] : tensors) {
        if (!torch::equal(tensor[source_block_id],
                          tensor[destination_block_id])) {
          return false;
        }
      }
    }
    return true;
  }
};

class ImmediateHierarchyTransferTestWorker final
    : public HierarchyTransferTestWorker {
 public:
  ImmediateHierarchyTransferTestWorker(const ParallelArgs& parallel_args,
                                       const torch::Device& device,
                                       const runtime::Options& options,
                                       const ModelArgs& model_args)
      : HierarchyTransferTestWorker(parallel_args,
                                    device,
                                    options,
                                    model_args) {}

  uint32_t transfer_kv_blocks(
      uint64_t /*batch_id*/,
      const std::vector<BlockTransferInfo>& block_transfer_info) override {
    ++vector_transfer_count_;
    return static_cast<uint32_t>(block_transfer_info.size());
  }

  uint32_t vector_transfer_count() const { return vector_transfer_count_; }

 private:
  uint32_t vector_transfer_count_ = 0;
};

class TestMTPWorker final : public MTPWorkerImpl {
 public:
  TestMTPWorker(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options)
      : MTPWorkerImpl(parallel_args, device, options, WorkerType::LLM) {}

  std::shared_ptr<EmbeddingCache> initialize_replay_cache(bool retain_span) {
    retain_replay_span_ = retain_span;
    init_embedding_cache(kBlockCount);
    return embedding_cache_;
  }

  void replace_transfer_workers(std::unique_ptr<LLMWorkerImpl> target,
                                std::unique_ptr<LLMWorkerImpl> draft) {
    impl_ = std::move(target);
    draft_impl_ = std::move(draft);
  }

  void prepare_hierarchy_transfers() { prepare_hierarchy_kv_cache_transfers(); }

  void finalize_hierarchy_transfers() {
    finalize_hierarchy_kv_cache_transfers();
  }

  std::shared_ptr<HierarchyKVCacheTransfer> target_transfer_owner() const {
    return get_hierarchy_kv_cache_transfer();
  }

  std::shared_ptr<HierarchyKVCacheTransfer> target_worker_transfer() const {
    return impl_->get_hierarchy_kv_cache_transfer();
  }

  std::shared_ptr<HierarchyKVCacheTransfer> draft_worker_transfer() const {
    return draft_impl_->get_hierarchy_kv_cache_transfer();
  }

  void block_outer_executor(std::promise<void>* started,
                            std::shared_future<void> release) {
    CHECK(started != nullptr);
    threadpool_.schedule([started, release = std::move(release)]() mutable {
      started->set_value();
      release.wait();
    });
  }

  static int32_t preserve_checkpoint_width(
      int32_t effective_speculative_tokens,
      const std::vector<int32_t>& accepted_prefix_lengths,
      int32_t num_speculative_tokens) {
    return preserve_checkpoint_validate_width(effective_speculative_tokens,
                                              accepted_prefix_lengths,
                                              num_speculative_tokens);
  }

 private:
  bool requires_full_target_replay() const override {
    return retain_replay_span_;
  }
  bool retain_replay_span_ = false;
};

class TestDFlashWorker final : public DFlashWorkerImpl {
 public:
  TestDFlashWorker(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options)
      : DFlashWorkerImpl(parallel_args, device, options) {}

  void replace_transfer_workers(std::unique_ptr<LLMWorkerImpl> target,
                                std::unique_ptr<LLMWorkerImpl> draft) {
    impl_ = std::move(target);
    draft_impl_ = std::move(draft);
  }

  std::shared_ptr<HierarchyKVCacheTransfer> transfer_owner() const {
    return get_hierarchy_kv_cache_transfer();
  }

  std::shared_ptr<HierarchyKVCacheTransfer> target_worker_transfer() const {
    return impl_->get_hierarchy_kv_cache_transfer();
  }

  std::shared_ptr<HierarchyKVCacheTransfer> draft_worker_transfer() const {
    return draft_impl_->get_hierarchy_kv_cache_transfer();
  }
};

class MTPHostOffloadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "An accelerator is required for MTP host offload tests.";
    }
  }
};

TEST(MTPAdaptiveValidateWidthTest, PreservesWidthForPreviousAcceptedTokens) {
  EXPECT_EQ(TestMTPWorker::preserve_checkpoint_width(
                /*effective_speculative_tokens=*/1,
                /*accepted_prefix_lengths=*/{4, 1},
                /*num_speculative_tokens=*/3),
            3);
}

// Replace model construction and weight I/O, keeping the real worker loading
// lifecycle, policy validation, KV allocation and embedding cache.
class ContextPolicyModel final : public CausalLM {
 public:
  ContextPolicyModel(DraftContextUpdate update, torch::TensorOptions options)
      : update_(update), options_(std::move(options)) {}

  bool share_weights_from(CausalLM& /*source*/) override { return true; }

  ModelOutput forward(const torch::Tensor& /*tokens*/,
                      const torch::Tensor& /*positions*/,
                      std::vector<KVCache>& /*caches*/,
                      const ModelInputParams& /*params*/) override {
    LOG(FATAL) << "Model execution is outside the context-policy test.";
    return {};
  }
  torch::Tensor logits(const torch::Tensor& /*hidden*/,
                       const torch::Tensor& /*indices*/) override {
    LOG(FATAL) << "Logits are outside the context-policy test.";
    return {};
  }
  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {
    LOG(FATAL) << "Expert loading is outside the context-policy test.";
  }
  void update_expert_weight(int32_t /*layer_id*/) override {
    LOG(FATAL) << "Expert updates are outside the context-policy test.";
  }

 private:
  DraftContextUpdate update_;
  torch::TensorOptions options_;
};

class ContextLoadingWorker final : public HierarchyTransferTestWorker {
 public:
  ContextLoadingWorker(const ParallelArgs& parallel_args,
                       const torch::Device& device,
                       const runtime::Options& options,
                       const ModelArgs& args,
                       DraftContextUpdate update,
                       std::string draft_model_type = "glm5_next_mtp",
                       std::string draft_model_impl = "native")
      : HierarchyTransferTestWorker(parallel_args, device, options, args),
        update_(update),
        draft_model_type_(std::move(draft_model_type)),
        draft_model_impl_(std::move(draft_model_impl)) {}

  bool init_model(ModelContext& context) override {
    if (options_.is_draft_engine()) {
      // Supply the fake draft's identity while retaining the real loading path.
      ModelArgs args = context.get_model_args();
      args.model_type(draft_model_type_);
      context = ModelContext(context.get_parallel_args(),
                             args,
                             context.get_quant_args(),
                             context.get_tensor_options());
      context.set_model_impl(draft_model_impl_);
    }
    model_ = std::make_unique<ContextPolicyModel>(update_,
                                                  context.get_tensor_options());
    return true;
  }
  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}

  void load_target(const std::string& model_impl) {
    context_.set_model_impl(model_impl);
    CHECK(init_model(context_));
    mark_loaded();
  }

  void add_request_state_cache() {
    const torch::Tensor state =
        torch::zeros({1}, context_.get_tensor_options());
    kv_caches_.emplace_back(LinearAttentionKVCacheTensors{state, state});
  }

 private:
  DraftContextUpdate update_;
  std::string draft_model_type_;
  std::string draft_model_impl_;
};

class ContextPolicyMTPWorker final : public MTPWorkerImpl {
 public:
  ContextPolicyMTPWorker(const ParallelArgs& parallel_args,
                         const torch::Device& device,
                         const runtime::Options& options,
                         std::unique_ptr<LLMWorkerImpl> target,
                         std::unique_ptr<LLMWorkerImpl> draft)
      : MTPWorkerImpl(parallel_args, device, options, WorkerType::LLM) {
    impl_ = std::move(target);
    draft_impl_ = std::move(draft);
    dtype_ = impl_->dtype();
  }

  std::shared_ptr<EmbeddingCache> embedding_cache() const {
    return embedding_cache_;
  }

  void add_draft_request_state_cache() {
    static_cast<ContextLoadingWorker*>(draft_impl_.get())
        ->add_request_state_cache();
  }

  ForwardInput prepare_next_draft(const ForwardInput& input) {
    ForwardInput draft_input;
    prepare_draft_inputs(input, draft_input, /*position_offset=*/0);
    return draft_input;
  }
};

class MTPContextPolicyTest : public MTPHostOffloadTest {
 protected:
  void SetUp() override {
#if defined(USE_NPU)
    GTEST_SKIP() << "GLM5 Next MTP replay capabilities are registered only "
                    "for MLU models.";
#else
    MTPHostOffloadTest::SetUp();
    if (IsSkipped()) {
      return;
    }
    char path[] = "/tmp/xllm_context_policy_XXXXXX";
    ASSERT_NE(mkdtemp(path), nullptr);
    model_path_ = path;
    std::ofstream(model_path_ / "config.json") << R"json({
      "model_type": "qwen3", "torch_dtype": "float32",
      "hidden_size": 2, "num_hidden_layers": 1,
      "num_attention_heads": 1, "num_key_value_heads": 1,
      "head_dim": 2, "vocab_size": 8
    })json";
    // A valid empty safetensors checkpoint; the test worker skips weight I/O.
    std::ofstream(model_path_ / "model.safetensors", std::ios::binary)
        .write("\x08\0\0\0\0\0\0\0{}      ", 16);
#endif
  }

  void TearDown() override {
    if (!model_path_.empty()) {
      std::filesystem::remove_all(model_path_);
    }
  }

  std::unique_ptr<ContextPolicyMTPWorker> make_worker(
      DraftContextUpdate update,
      const std::string& target_type = "glm5_next",
      const std::string& target_impl = "native",
      bool embedded_eagle3 = false,
      bool target_ready = false,
      const std::string& draft_type = "glm5_next_mtp",
      const std::string& draft_impl = "native") {
    const torch::Device device(Platform::type_torch(), /*index=*/0);
    const ParallelArgs parallel_args(
        /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
    runtime::Options options = make_runtime_options(/*host_blocks_factor=*/0);
    options.speculative_algorithm(embedded_eagle3 ? "Eagle3" : "MTP");
    ModelArgs args =
        make_model_args(target_type, /*layer_count=*/1, /*head_dim=*/2);
    args.hidden_size(2).enable_embedded_eagle3_draft(embedded_eagle3);
    auto target = std::make_unique<ContextLoadingWorker>(
        parallel_args, device, options, args, DraftContextUpdate::TAIL_EXTEND);
    target->load_target(target_impl);
    if (target_ready) {
      CHECK(target->allocate_kv_cache(cache_shape()));
    }
    runtime::Options draft_options = options;
    draft_options.is_draft_engine(true).num_speculative_tokens(0);
    std::string resolved_draft_type = draft_type;
    if (update == DraftContextUpdate::TAIL_EXTEND &&
        resolved_draft_type == "glm5_next_mtp") {
      resolved_draft_type = "mimo_mtp";
    }
    auto draft = std::make_unique<ContextLoadingWorker>(parallel_args,
                                                        device,
                                                        draft_options,
                                                        args,
                                                        update,
                                                        resolved_draft_type,
                                                        draft_impl);
    return std::make_unique<ContextPolicyMTPWorker>(
        parallel_args, device, options, std::move(target), std::move(draft));
  }

  KVCacheShape cache_shape() const {
    return make_cache_shape(
        make_model_args("qwen3", /*layer_count=*/1, /*head_dim=*/2));
  }

  ForwardInput make_decode_input() const {
    const torch::Device device(Platform::type_torch(), /*index=*/0);
    ForwardInput input;
    input.token_ids_host = torch::tensor({7}, torch::kInt);
    input.token_ids = input.token_ids_host.to(device);
    input.positions_host = torch::tensor({2}, torch::kInt);
    input.positions = input.positions_host.to(device);
    ModelInputParams& params = input.input_params;
    params.meta.batch_forward_type = BatchForwardType::DECODE;
    params.meta.num_sequences = 1;
    params.meta.q_max_seq_len = 1;
    params.meta.kv_max_seq_len = 3;
#if defined(USE_NPU)
    params.attention.host.q_seq_lens = {1};
    params.attention.host.kv_seq_lens = {3};
#else
    params.attention.host.q_seq_lens = {0, 1};
    params.attention.host.kv_seq_lens = {0, 3};
#endif
    params.attention.host.q_cu_seq_lens = {0, 1};
    params.attention.host.block_tables = torch::tensor({{0}}, torch::kInt);
    params.embedding.linear_state_ids = {5};
    params.embedding.linear_state_indices =
        torch::tensor({5}, torch::kInt).to(device);
    params.embedding.linear_state_read_ids = {6};
    params.embedding.linear_state_read_indices =
        torch::tensor({6}, torch::kInt).to(device);
    params.linear_state_validity_mask = {1};
    params.linear_state_validity_mask_tensor =
        torch::tensor({1}, torch::kInt).to(device);
    params.num_accepted_tokens = torch::tensor({2}, torch::kInt).to(device);
    params.num_accepted_tokens_host = {2};
    return input;
  }

  std::filesystem::path model_path_;
};

TEST_F(MTPContextPolicyTest, HybridTargetKeepsTailDraftContextCompact) {
  auto worker = make_worker(DraftContextUpdate::TAIL_EXTEND);
  ASSERT_TRUE(worker->init_model(
      model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP));
  ASSERT_TRUE(worker->allocate_kv_cache(cache_shape()));
  auto cache = worker->embedding_cache();
  const torch::Tensor hidden = torch::tensor({1.0f, 2.0f});
  cache->write_mtp_bootstrap_context(
      /*embedding_id=*/0, "request", /*token_id=*/7, hidden);
  const auto states = cache->read_decode_states({0}, {"request"});
  ASSERT_EQ(states.size(), 1);
  EXPECT_TRUE(states.front().replay_token_ids.empty());
  EXPECT_FALSE(states.front().replay_embeddings.defined());
  EXPECT_TRUE(torch::equal(states.front().embedding, hidden));
}

TEST_F(MTPContextPolicyTest, FullAttentionDraftDropsTargetState) {
  auto worker = make_worker(DraftContextUpdate::TAIL_EXTEND);
  ASSERT_TRUE(worker->init_model(
      model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP));
  const ForwardInput input = make_decode_input();

  const ForwardInput draft = worker->prepare_next_draft(input);
  const ModelInputParams& params = draft.input_params;
  EXPECT_TRUE(params.embedding.linear_state_ids.empty());
  EXPECT_FALSE(params.embedding.linear_state_indices.defined());
  EXPECT_TRUE(params.embedding.linear_state_read_ids.empty());
  EXPECT_FALSE(params.embedding.linear_state_read_indices.defined());
  EXPECT_TRUE(params.linear_state_validity_mask.empty());
  EXPECT_FALSE(params.linear_state_validity_mask_tensor.defined());
  EXPECT_FALSE(params.num_accepted_tokens.defined());
  EXPECT_TRUE(params.num_accepted_tokens_host.empty());
  EXPECT_EQ(input.input_params.embedding.linear_state_ids,
            (std::vector<int32_t>{5}));
}

TEST_F(MTPContextPolicyTest, StatefulDraftRetainsRequestState) {
  auto worker = make_worker(DraftContextUpdate::TAIL_EXTEND);
  ASSERT_TRUE(worker->init_model(
      model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP));
  worker->add_draft_request_state_cache();
  const ForwardInput input = make_decode_input();

  const ForwardInput draft = worker->prepare_next_draft(input);
  const ModelInputParams& params = draft.input_params;
  EXPECT_EQ(params.embedding.linear_state_ids, (std::vector<int32_t>{5}));
  EXPECT_EQ(params.embedding.linear_state_read_ids, (std::vector<int32_t>{5}));
  EXPECT_TRUE(torch::equal(params.embedding.linear_state_indices,
                           input.input_params.embedding.linear_state_indices));
  EXPECT_TRUE(torch::equal(params.embedding.linear_state_read_indices,
                           input.input_params.embedding.linear_state_indices));
  EXPECT_EQ(params.linear_state_validity_mask, (std::vector<int64_t>{1}));
  EXPECT_TRUE(
      torch::equal(params.linear_state_validity_mask_tensor,
                   input.input_params.linear_state_validity_mask_tensor));
  EXPECT_TRUE(torch::equal(params.num_accepted_tokens,
                           input.input_params.num_accepted_tokens));
  EXPECT_EQ(params.num_accepted_tokens_host, (std::vector<int64_t>{2}));
  EXPECT_EQ(params.attention.host.kpool_query_lens, (std::vector<int32_t>{1}));
}

TEST_F(MTPContextPolicyTest, QwenNativePairRetainsAcceptedSpan) {
  auto worker = make_worker(DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                            /*target_type=*/"qwen3_5",
                            /*target_impl=*/"native",
                            /*embedded_eagle3=*/false,
                            /*target_ready=*/false,
                            /*draft_type=*/"qwen3_5_mtp");
  ASSERT_TRUE(worker->init_model(
      model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP));
  ASSERT_TRUE(worker->allocate_kv_cache(cache_shape()));
  auto cache = worker->embedding_cache();
  cache->write_target_context({0},
                              {"request"},
                              torch::tensor({{7, 8}}, torch::kInt),
                              torch::tensor({{{1.0f, 2.0f}, {3.0f, 4.0f}}}),
                              /*num_speculative_tokens=*/1);
  const auto states = cache->read_decode_states({0}, {"request"});
  EXPECT_EQ(states.front().replay_token_ids, (std::vector<int32_t>{7, 8}));
}

TEST_F(MTPContextPolicyTest, QwenPythonTargetKeepsTailContext) {
  auto worker = make_worker(DraftContextUpdate::TAIL_EXTEND,
                            /*target_type=*/"qwen3_5",
                            /*target_impl=*/"python",
                            /*embedded_eagle3=*/false,
                            /*target_ready=*/false,
                            /*draft_type=*/"qwen3_5_mtp");
  ASSERT_TRUE(worker->init_model(
      model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP));
  ASSERT_TRUE(worker->allocate_kv_cache(cache_shape()));
  auto cache = worker->embedding_cache();
  cache->write_mtp_bootstrap_context(
      /*embedding_id=*/0,
      "request",
      /*token_id=*/7,
      torch::tensor({1.0f, 2.0f}));
  const auto states = cache->read_decode_states({0}, {"request"});
  EXPECT_TRUE(states.front().replay_token_ids.empty());
}

TEST_F(MTPContextPolicyTest, RejectsCacheAllocationBeforeDraftInitialization) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        auto worker = make_worker(DraftContextUpdate::TAIL_EXTEND);
        worker->allocate_kv_cache(cache_shape());
      },
      "Draft context replay policy is not initialized");
}

TEST_F(MTPContextPolicyTest,
       ReplayDraftRetainsAcceptedSpanAcrossCacheAllocation) {
  // The target may already own its cache when the draft finishes loading.
  auto worker = make_worker(DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                            /*target_type=*/"glm5_next",
                            /*target_impl=*/"native",
                            /*embedded_eagle3=*/false,
                            /*target_ready=*/true);
  ASSERT_TRUE(worker->init_model(
      model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP));
  for (int32_t allocation = 0; allocation < 2; ++allocation) {
    ASSERT_TRUE(worker->allocate_kv_cache(cache_shape()));
    auto cache = worker->embedding_cache();
    torch::Tensor tokens = torch::tensor({{7, 8}}, torch::kInt);
    torch::Tensor hidden = torch::tensor({{{1.0f, 2.0f}, {3.0f, 4.0f}}});
    cache->write_target_context(
        {0}, {"request"}, tokens, hidden, /*num_speculative_tokens=*/1);
    hidden.fill_(99);
    tokens.fill_(-1);
    const auto states = cache->read_decode_states({0}, {"request"});
    ASSERT_EQ(states.size(), 1);
    EXPECT_EQ(states.front().replay_token_ids, (std::vector<int32_t>{7, 8}));
    EXPECT_TRUE(torch::equal(states.front().replay_embeddings,
                             torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}})));
    EXPECT_TRUE(
        torch::equal(states.front().embedding, torch::tensor({3.0f, 4.0f})));
  }
}

TEST_F(MTPContextPolicyTest, RejectsReplayForGenericTarget) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        auto worker = make_worker(DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                                  /*target_type=*/"qwen3");
        worker->init_model(
            model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP);
      },
      "Incompatible draft context update.*target=qwen3");
}

TEST_F(MTPContextPolicyTest, RejectsMixedNativeReplayFamilies) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        auto worker = make_worker(DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                                  /*target_type=*/"glm5_next",
                                  /*target_impl=*/"native",
                                  /*embedded_eagle3=*/false,
                                  /*target_ready=*/false,
                                  /*draft_type=*/"qwen3_5_mtp");
        worker->init_model(
            model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP);
      },
      "Incompatible draft context update");
}

TEST_F(MTPContextPolicyTest, RejectsReplayWithoutTargetFamily) {
  ASSERT_TRUE(ModelRegistry::get_mtp_capabilities("qwen3_5")
                  .supports_expanded_replay_target);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        MtpModelCapabilities target =
            ModelRegistry::get_mtp_capabilities("qwen3_5");
        target.replay_family = MtpReplayFamily::NONE;
        ModelRegistry::register_mtp_capabilities("qwen3_5", target);
        auto worker = make_worker(DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                                  /*target_type=*/"qwen3_5",
                                  /*target_impl=*/"native",
                                  /*embedded_eagle3=*/false,
                                  /*target_ready=*/false,
                                  /*draft_type=*/"qwen3_5_mtp");
        worker->init_model(
            model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP);
      },
      "Incompatible draft context update");
}

TEST_F(MTPContextPolicyTest, RejectsReplayForOtherChunkedPrefillTargets) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        auto worker = make_worker(DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                                  /*target_type=*/"mimo");
        worker->init_model(
            model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP);
      },
      "Incompatible draft context update.*target=mimo");
}

TEST_F(MTPContextPolicyTest, RejectsReplayForPythonDraft) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        auto worker = make_worker(DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                                  /*target_type=*/"glm5_next",
                                  /*target_impl=*/"native",
                                  /*embedded_eagle3=*/false,
                                  /*target_ready=*/false,
                                  /*draft_type=*/"glm5_next_mtp",
                                  /*draft_impl=*/"python");
        worker->init_model(
            model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP);
      },
      "Incompatible draft context update");
}

TEST_F(MTPContextPolicyTest, PythonTargetKeepsTailUpdatesAndRejectsReplay) {
  auto worker = make_worker(DraftContextUpdate::TAIL_EXTEND,
                            /*target_type=*/"glm5_next",
                            /*target_impl=*/"python");
  ASSERT_TRUE(worker->init_model(
      model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP));
  ASSERT_TRUE(worker->allocate_kv_cache(cache_shape()));
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        auto replay_worker =
            make_worker(DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                        /*target_type=*/"glm5_next",
                        /*target_impl=*/"python");
        replay_worker->init_model(
            model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP);
      },
      "Incompatible draft context update.*target=glm5_next");
}

TEST_F(MTPContextPolicyTest, EmbeddedEagleKeepsTailUpdatesAndRejectsReplay) {
  auto worker = make_worker(DraftContextUpdate::TAIL_EXTEND,
                            /*target_type=*/"glm5_next",
                            /*target_impl=*/"native",
                            /*embedded_eagle3=*/true);
  ASSERT_TRUE(worker->init_model(
      model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP));
  ASSERT_TRUE(worker->allocate_kv_cache(cache_shape()));
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        auto replay_worker =
            make_worker(DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                        /*target_type=*/"glm5_next",
                        /*target_impl=*/"native",
                        /*embedded_eagle3=*/true);
        replay_worker->init_model(
            model_path_.string(), /*random_seed=*/0, MasterStatus::WAKEUP);
      },
      "Incompatible draft context update");
}

TEST(SpeculativeDraftKVCacheShapeTest, ReusesGroupedTargetPoolCounts) {
  ModelArgs target_model_args =
      make_model_args("deepseek_v4", /*layer_count=*/3, /*head_dim=*/8);
  target_model_args.compress_ratios({1, 4, 128});
  ModelArgs draft_model_args =
      make_model_args("deepseek_v4_mtp", /*layer_count=*/3, /*head_dim=*/8);
  draft_model_args.compress_ratios({1, 4, 128});

  KVCacheCapacity target_capacity;
  target_capacity.block_size(kBlockSize).swa_count(2).c4_count(3).c128_count(5);
  const KVCacheShape target_shape(
      target_capacity, target_model_args, /*world_size=*/8);

  const KVCacheShape draft_shape =
      build_speculative_draft_kv_cache_shape(target_shape,
                                             draft_model_args,
                                             kBlockSize,
                                             /*draft_world_size=*/1,
                                             /*kv_cache_dtype=*/"auto");

  EXPECT_TRUE(draft_shape.has_grouped_cache_layout());
  EXPECT_EQ(draft_shape.key_cache_shape(), (std::vector<int64_t>{2, 3, 5}));
}

TEST_F(MTPHostOffloadTest, SharedCacheAllocationHonorsReplaySpanPolicy) {
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const runtime::Options options =
      make_runtime_options(/*host_blocks_factor=*/0.0);
  TestMTPWorker worker(parallel_args, device, options);
  const auto embedding = torch::tensor({1.0f, 2.0f});
  for (bool retain_span : {false, true}) {
    auto cache = worker.initialize_replay_cache(retain_span);
    cache->write_mtp_bootstrap_context(
        /*embedding_id=*/0, "request", /*token_id=*/7, embedding);
    const auto states = cache->read_decode_states({0}, {"request"});
    ASSERT_EQ(states.size(), 1);
    EXPECT_EQ(states.front().replay_embeddings.defined(), retain_span);
    EXPECT_EQ(states.front().replay_token_ids,
              retain_span ? std::vector<int32_t>{7} : std::vector<int32_t>{});
    EXPECT_TRUE(torch::equal(states.front().embedding, embedding));
  }
}

TEST_F(MTPHostOffloadTest, VectorTransferWithoutHierarchyIsNoop) {
  constexpr uint64_t kBatchId = 42;
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const runtime::Options options = make_runtime_options(0.0);
  TestMTPWorker worker(parallel_args, device, options);

  const std::vector<BlockTransferInfo> transfer_info = {
      BlockTransferInfo(/*src_block_id=*/1, /*dst_block_id=*/2),
      BlockTransferInfo(/*src_block_id=*/3, /*dst_block_id=*/4)};
  auto target = std::make_unique<RecordingTransferWorker>(
      parallel_args,
      device,
      options,
      static_cast<uint32_t>(transfer_info.size()));
  auto draft = std::make_unique<RecordingTransferWorker>(
      parallel_args,
      device,
      options,
      static_cast<uint32_t>(transfer_info.size()));
  RecordingTransferWorker* target_ptr = target.get();
  RecordingTransferWorker* draft_ptr = draft.get();
  worker.replace_transfer_workers(std::move(target), std::move(draft));

  const uint32_t transferred =
      worker.transfer_kv_blocks(kBatchId, transfer_info);

  EXPECT_EQ(transferred, 0);
  EXPECT_EQ(target_ptr->vector_transfer_count(), 0);
  EXPECT_EQ(draft_ptr->vector_transfer_count(), 0);
}

TEST_F(MTPHostOffloadTest, SliceTransferWithoutHierarchyIsNoop) {
  constexpr uint64_t kBatchId = 73;
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const runtime::Options options = make_runtime_options(0.0);
  TestMTPWorker worker(parallel_args, device, options);

  const std::vector<BlockTransferInfo> transfer_info = {
      BlockTransferInfo(/*src_block_id=*/5, /*dst_block_id=*/6)};
  auto target = std::make_unique<RecordingTransferWorker>(
      parallel_args, device, options, /*transfer_result=*/1);
  auto draft = std::make_unique<RecordingTransferWorker>(
      parallel_args, device, options, /*transfer_result=*/0);
  RecordingTransferWorker* target_ptr = target.get();
  RecordingTransferWorker* draft_ptr = draft.get();
  worker.replace_transfer_workers(std::move(target), std::move(draft));
  Slice<BlockTransferInfo> transfer_slice(transfer_info);

  const uint32_t transferred =
      worker.transfer_kv_blocks(kBatchId, transfer_slice);

  EXPECT_EQ(transferred, 0);
  EXPECT_EQ(target_ptr->slice_transfer_count(), 0);
  EXPECT_EQ(draft_ptr->slice_transfer_count(), 0);
}

TEST_F(MTPHostOffloadTest, StorePrefetchWithoutHierarchyMisses) {
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const runtime::Options options = make_runtime_options(0.0);
  TestMTPWorker worker(parallel_args, device, options);

  const std::vector<BlockTransferInfo> transfer_info = {
      BlockTransferInfo(/*src_block_id=*/5, /*dst_block_id=*/6)};
  auto target = std::make_unique<RecordingTransferWorker>(
      parallel_args, device, options, /*transfer_result=*/1);
  auto draft = std::make_unique<RecordingTransferWorker>(
      parallel_args, device, options, /*transfer_result=*/1);
  RecordingTransferWorker* target_ptr = target.get();
  RecordingTransferWorker* draft_ptr = draft.get();
  worker.replace_transfer_workers(std::move(target), std::move(draft));
  Slice<BlockTransferInfo> transfer_slice(transfer_info);

  const std::vector<uint8_t> hits =
      worker.prefetch_kv_blocks(transfer_slice, /*stats=*/nullptr);

  EXPECT_EQ(hits, std::vector<uint8_t>({0}));
  EXPECT_EQ(target_ptr->prefetch_count(), 0U);
  EXPECT_EQ(draft_ptr->prefetch_count(), 0U);
}

TEST_F(MTPHostOffloadTest, BindsAndReleasesUnifiedTransferOwner) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const runtime::Options options = make_runtime_options(2.0);
  const ModelArgs target_model_args =
      make_model_args("test_target", /*layer_count=*/2, /*head_dim=*/8);
  const ModelArgs draft_model_args =
      make_model_args("test_draft", /*layer_count=*/1, /*head_dim=*/4);
  const KVCacheShape target_shape = make_cache_shape(target_model_args);
  const KVCacheShape draft_shape = make_cache_shape(draft_model_args);
  std::weak_ptr<HierarchyKVCacheTransfer> transfer_lifetime;

  {
    auto worker = std::make_unique<TestMTPWorker>(
        parallel_args, device.unwrap(), options);
    auto target = std::make_unique<HierarchyTransferTestWorker>(
        parallel_args, device.unwrap(), options, target_model_args);
    auto draft = std::make_unique<HierarchyTransferTestWorker>(
        parallel_args, device.unwrap(), options, draft_model_args);
    HierarchyTransferTestWorker* target_ptr = target.get();
    HierarchyTransferTestWorker* draft_ptr = draft.get();

    worker->replace_transfer_workers(std::move(target), std::move(draft));
    worker->prepare_hierarchy_transfers();
    worker->prepare_hierarchy_transfers();

    ASSERT_NE(target_ptr->get_hierarchy_kv_cache_transfer(), nullptr);
    ASSERT_NE(draft_ptr->get_hierarchy_kv_cache_transfer(), nullptr);
    EXPECT_FALSE(target_ptr->get_hierarchy_kv_cache_transfer()
                     ->registration_finalized());
    EXPECT_FALSE(
        draft_ptr->get_hierarchy_kv_cache_transfer()->registration_finalized());

    target_ptr->initialize_hierarchy_cache(target_shape);
    draft_ptr->initialize_hierarchy_cache(draft_shape);
    EXPECT_FALSE(target_ptr->get_hierarchy_kv_cache_transfer()
                     ->registration_finalized());
    EXPECT_FALSE(
        draft_ptr->get_hierarchy_kv_cache_transfer()->registration_finalized());
    worker->finalize_hierarchy_transfers();
    worker->finalize_hierarchy_transfers();

    std::shared_ptr<HierarchyKVCacheTransfer> target_owner =
        worker->target_transfer_owner();
    std::shared_ptr<HierarchyKVCacheTransfer> target_worker_transfer =
        worker->target_worker_transfer();
    std::shared_ptr<HierarchyKVCacheTransfer> draft_worker_transfer =
        worker->draft_worker_transfer();
    ASSERT_NE(target_owner, nullptr);
    EXPECT_EQ(target_owner.get(), target_worker_transfer.get());
    EXPECT_EQ(target_owner.get(), draft_worker_transfer.get());
    EXPECT_TRUE(target_owner->registration_finalized());
    EXPECT_TRUE(target_owner->supports_block_type(
        HierarchyKVCacheTransfer::CacheRole::TARGET, BlockType::KV));
    EXPECT_TRUE(target_owner->supports_block_type(
        HierarchyKVCacheTransfer::CacheRole::DRAFT, BlockType::KV));
    transfer_lifetime = target_owner;
  }

  EXPECT_TRUE(transfer_lifetime.expired());
}

TEST_F(MTPHostOffloadTest, DFlashUsesSpeculativeHierarchyTransferLifecycle) {
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options = make_runtime_options(2.0);
  options.speculative_algorithm("DFlash");
  const ModelArgs model_args =
      make_model_args("test_dflash", /*layer_count=*/2, /*head_dim=*/8);
  const KVCacheShape cache_shape = make_cache_shape(model_args);
  std::weak_ptr<HierarchyKVCacheTransfer> transfer_lifetime;

  {
    auto worker = std::make_unique<TestDFlashWorker>(
        parallel_args, device.unwrap(), options);
    auto target = std::make_unique<HierarchyTransferTestWorker>(
        parallel_args, device.unwrap(), options, model_args);
    auto draft = std::make_unique<HierarchyTransferTestWorker>(
        parallel_args, device.unwrap(), options, model_args);
    target->mark_loaded();
    draft->mark_loaded();
    worker->replace_transfer_workers(std::move(target), std::move(draft));

    ASSERT_TRUE(worker->allocate_kv_cache(cache_shape));
    std::shared_ptr<HierarchyKVCacheTransfer> transfer =
        worker->transfer_owner();
    ASSERT_NE(transfer, nullptr);
    EXPECT_EQ(transfer.get(), worker->target_worker_transfer().get());
    EXPECT_EQ(transfer.get(), worker->draft_worker_transfer().get());
    EXPECT_TRUE(transfer->registration_finalized());
    EXPECT_TRUE(transfer->supports_block_type(
        HierarchyKVCacheTransfer::CacheRole::TARGET, BlockType::KV));
    EXPECT_TRUE(transfer->supports_block_type(
        HierarchyKVCacheTransfer::CacheRole::DRAFT, BlockType::KV));
    transfer_lifetime = transfer;
  }

  EXPECT_TRUE(transfer_lifetime.expired());
}

// B5 (implement.md, design.md §3.1 Gate B4): glm5_next + cp_size>1 +
// DFlash2 must no longer CHECK-fail in DFlashWorkerImpl's allow_cp gate.
// Model-level correctness of the aux-hidden-buffer CP restore this relies on
// is proven separately in
// tests/python/test_glm5_next_cp.py::test_layer_capture_restores_global_rows_with_context_parallelism.
TEST_F(MTPHostOffloadTest, DFlash2AllowsGlm5NextContextParallelism) {
  if (!Platform::is_npu()) {
    GTEST_SKIP() << "allow_cp's glm5_next exemption is NPU-only.";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  // cp_size=2, dp_size=1, ep_size=1: exercises the parallel_args.cp_size() > 1
  // branch of DFlashWorkerImpl's allow_cp lambda.
  const ParallelArgs parallel_args(
      /*rank=*/0,
      /*world_size=*/2,
      /*dp_size=*/1,
      /*cp_size=*/2,
      /*process_group=*/nullptr,
      /*ep_size=*/1);
  runtime::Options options = make_runtime_options(2.0);
  options.speculative_algorithm("DFlash2");
  const ScopedModelTypeConfig model_config("glm5_next");
  options.model_path(model_config.path().string());

  // Construction must not CHECK-fail: allow_cp resolves true because
  // model_type == "glm5_next" && is_dflash2_algorithm("DFlash2").
  auto worker = std::make_unique<TestDFlashWorker>(
      parallel_args, device.unwrap(), options);
  EXPECT_NE(worker, nullptr);
}

// Regression: the pre-existing DeepSeek-V4 + cp_size>1 exemption (allow_cp's
// original condition) must remain unaffected by B5's glm5_next addition.
TEST_F(MTPHostOffloadTest, DeepseekV4StillAllowsContextParallelism) {
  if (!Platform::is_npu()) {
    GTEST_SKIP() << "allow_cp's deepseek_v4 exemption is NPU-only.";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0,
      /*world_size=*/2,
      /*dp_size=*/1,
      /*cp_size=*/2,
      /*process_group=*/nullptr,
      /*ep_size=*/1);
  runtime::Options options = make_runtime_options(2.0);
  options.speculative_algorithm("MTP");
  const ScopedModelTypeConfig model_config("deepseek_v4");
  options.model_path(model_config.path().string());

  auto worker = std::make_unique<TestDFlashWorker>(
      parallel_args, device.unwrap(), options);
  EXPECT_NE(worker, nullptr);
}

TEST_F(MTPHostOffloadTest, UnifiedTransferRoundTripUsesSharedSynchronizer) {
  constexpr uint64_t kBatchId = 91;
  constexpr int64_t kSourceBlockId = 0;
  constexpr int64_t kDestinationBlockId = 1;
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const runtime::Options options = make_runtime_options(2.0);
  const ModelArgs target_model_args =
      make_model_args("test_target", /*layer_count=*/2, /*head_dim=*/8);
  const ModelArgs draft_model_args =
      make_model_args("test_draft", /*layer_count=*/1, /*head_dim=*/4);
  const KVCacheShape target_shape = make_cache_shape(target_model_args);
  const KVCacheShape draft_shape = make_cache_shape(draft_model_args);
  TestMTPWorker worker(parallel_args, device.unwrap(), options);
  auto target = std::make_unique<HierarchyTransferTestWorker>(
      parallel_args, device.unwrap(), options, target_model_args);
  auto draft = std::make_unique<HierarchyTransferTestWorker>(
      parallel_args, device.unwrap(), options, draft_model_args);
  HierarchyTransferTestWorker* target_ptr = target.get();
  HierarchyTransferTestWorker* draft_ptr = draft.get();
  worker.replace_transfer_workers(std::move(target), std::move(draft));
  worker.prepare_hierarchy_transfers();
  target_ptr->initialize_hierarchy_cache(target_shape);
  draft_ptr->initialize_hierarchy_cache(draft_shape);
  worker.finalize_hierarchy_transfers();
  std::shared_ptr<HierarchyKVCacheTransfer> unified_transfer =
      worker.target_transfer_owner();
  ASSERT_NE(unified_transfer, nullptr);

  target_ptr->fill_block(kSourceBlockId, /*value=*/3.0);
  draft_ptr->fill_block(kSourceBlockId, /*value=*/13.0);
  target_ptr->zero_block(kDestinationBlockId);
  draft_ptr->zero_block(kDestinationBlockId);
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  BlockTransferInfo offload_info(kSourceBlockId, /*dst_block_id=*/0);
  offload_info.block_type = BlockType::KV;
  offload_info.transfer_type = TransferType::D2H2G;
  EXPECT_EQ(worker.transfer_kv_blocks(kBatchId, {offload_info}), 1U);

  BlockTransferInfo load_info(/*src_block_id=*/0, kDestinationBlockId);
  load_info.block_type = BlockType::KV;
  load_info.transfer_type = TransferType::H2D;
  EXPECT_EQ(worker.transfer_kv_blocks(kBatchId, {load_info}), 1U);

  ModelInputParams target_input_params;
  target_input_params.meta.batch_id = kBatchId;
  target_input_params.meta.requires_host_restore = true;
  worker.set_hierarchy_layer_synchronizer(target_input_params);
  EXPECT_FALSE(target_input_params.meta.requires_host_restore);
  ASSERT_NE(target_input_params.parallel.layer_wise_load_synchronizer, nullptr);
  EXPECT_FALSE(unified_transfer->take_load_handle(kBatchId).has_value());

  ModelInputParams draft_input_params = target_input_params;
  draft_ptr->set_hierarchy_layer_synchronizer(draft_input_params);
  EXPECT_FALSE(draft_input_params.meta.requires_host_restore);
  ASSERT_NE(draft_input_params.parallel.layer_wise_load_synchronizer, nullptr);
  EXPECT_EQ(target_input_params.parallel.layer_wise_load_synchronizer.get(),
            draft_input_params.parallel.layer_wise_load_synchronizer.get());
  EXPECT_EQ(target_input_params.parallel.layers_per_event, 2U);
  EXPECT_EQ(draft_input_params.parallel.layers_per_event, 2U);

  for (uint32_t layer_index = 0; layer_index < 2; ++layer_index) {
    ASSERT_TRUE(target_input_params.synchronize_layer(layer_index));
  }
  ASSERT_TRUE(draft_input_params.synchronize_draft_layer());
  EXPECT_TRUE(target_ptr->blocks_equal(kSourceBlockId, kDestinationBlockId));
  EXPECT_TRUE(draft_ptr->blocks_equal(kSourceBlockId, kDestinationBlockId));
}

TEST_F(MTPHostOffloadTest, D2HUsesSpeculativeWorkerExecutor) {
  constexpr uint64_t kBatchId = 84;
  constexpr int64_t kSourceBlockId = 0;
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  runtime::Options options = make_runtime_options(2.0);
  options.enable_schedule_overlap(true);
  const ModelArgs target_model_args =
      make_model_args("test_target", /*layer_count=*/2, /*head_dim=*/8);
  const ModelArgs draft_model_args =
      make_model_args("test_draft", /*layer_count=*/1, /*head_dim=*/4);
  const KVCacheShape target_shape = make_cache_shape(target_model_args);
  const KVCacheShape draft_shape = make_cache_shape(draft_model_args);
  TestMTPWorker worker(parallel_args, device.unwrap(), options);
  auto target = std::make_unique<ImmediateHierarchyTransferTestWorker>(
      parallel_args, device.unwrap(), options, target_model_args);
  auto draft = std::make_unique<HierarchyTransferTestWorker>(
      parallel_args, device.unwrap(), options, draft_model_args);
  ImmediateHierarchyTransferTestWorker* target_ptr = target.get();
  HierarchyTransferTestWorker* draft_ptr = draft.get();
  worker.replace_transfer_workers(std::move(target), std::move(draft));
  worker.prepare_hierarchy_transfers();
  target_ptr->initialize_hierarchy_cache(target_shape);
  draft_ptr->initialize_hierarchy_cache(draft_shape);
  worker.finalize_hierarchy_transfers();

  std::promise<void> blocker_started;
  std::future<void> blocker_started_future = blocker_started.get_future();
  std::promise<void> release_blocker;
  worker.block_outer_executor(&blocker_started,
                              release_blocker.get_future().share());
  blocker_started_future.get();

  BlockTransferInfo offload_info(kSourceBlockId, /*dst_block_id=*/0);
  offload_info.block_type = BlockType::KV;
  offload_info.transfer_type = TransferType::D2H2G;
  std::future<uint32_t> transfer_future =
      std::async(std::launch::async, [&worker, offload_info]() {
        return worker.transfer_kv_blocks(kBatchId, {offload_info});
      });

  EXPECT_EQ(transfer_future.wait_for(std::chrono::milliseconds(100)),
            std::future_status::timeout);
  release_blocker.set_value();
  EXPECT_EQ(transfer_future.get(), 1U);
  EXPECT_EQ(target_ptr->vector_transfer_count(), 0U);
}

TEST_F(MTPHostOffloadTest, Dsv4DraftSkipsUnsupportedCompressedBlockTypes) {
  constexpr uint64_t kBatchId = 92;
  constexpr int64_t kSourceBlockId = 0;
  constexpr int64_t kDestinationBlockId = 1;
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const runtime::Options options = make_runtime_options(2.0);

  ModelArgs target_model_args =
      make_model_args("deepseek_v4", /*layer_count=*/3, /*head_dim=*/8);
  target_model_args.index_head_dim(4)
      .window_size(kBlockSize)
      .compress_ratios({1, 4, 128});
  ModelArgs draft_model_args =
      make_model_args("deepseek_v4_mtp", /*layer_count=*/1, /*head_dim=*/8);
  draft_model_args.index_head_dim(4)
      .window_size(kBlockSize)
      .compress_ratios({1});
  const KVCacheShape target_shape = make_dsv4_cache_shape(
      target_model_args, /*c4_count=*/kBlockCount, /*c128_count=*/kBlockCount);
  const KVCacheShape draft_shape =
      make_dsv4_cache_shape(draft_model_args, /*c4_count=*/0, /*c128_count=*/0);

  TestMTPWorker worker(parallel_args, device.unwrap(), options);
  auto target = std::make_unique<HierarchyTransferTestWorker>(
      parallel_args, device.unwrap(), options, target_model_args);
  auto draft = std::make_unique<HierarchyTransferTestWorker>(
      parallel_args, device.unwrap(), options, draft_model_args);
  HierarchyTransferTestWorker* target_ptr = target.get();
  HierarchyTransferTestWorker* draft_ptr = draft.get();
  worker.replace_transfer_workers(std::move(target), std::move(draft));
  worker.prepare_hierarchy_transfers();
  target_ptr->initialize_hierarchy_cache(target_shape);
  draft_ptr->initialize_hierarchy_cache(draft_shape);
  worker.finalize_hierarchy_transfers();

  std::shared_ptr<HierarchyKVCacheTransfer> unified_transfer =
      worker.target_transfer_owner();
  ASSERT_NE(unified_transfer, nullptr);
  ASSERT_TRUE(unified_transfer->supports_block_type(
      HierarchyKVCacheTransfer::CacheRole::DRAFT, BlockType::SWA));
  ASSERT_FALSE(unified_transfer->supports_block_type(
      HierarchyKVCacheTransfer::CacheRole::DRAFT, BlockType::C4));
  ASSERT_FALSE(unified_transfer->supports_block_type(
      HierarchyKVCacheTransfer::CacheRole::DRAFT, BlockType::C128));

  const std::vector<BlockType> block_types = {
      BlockType::SWA, BlockType::C4, BlockType::C128};
  std::vector<BlockTransferInfo> offload_info;
  std::vector<BlockTransferInfo> load_info;
  offload_info.reserve(block_types.size());
  load_info.reserve(block_types.size());
  for (BlockType block_type : block_types) {
    target_ptr->fill_block(block_type, kSourceBlockId, /*value=*/3.0);
    target_ptr->zero_block(block_type, kDestinationBlockId);
    if (block_type == BlockType::SWA) {
      draft_ptr->fill_block(block_type, kSourceBlockId, /*value=*/13.0);
      draft_ptr->zero_block(block_type, kDestinationBlockId);
    }

    offload_info.emplace_back(kSourceBlockId, /*dst_block_id=*/0);
    offload_info.back().block_type = block_type;
    offload_info.back().transfer_type = TransferType::D2H2G;
    load_info.emplace_back(/*src_block_id=*/0, kDestinationBlockId);
    load_info.back().block_type = block_type;
    load_info.back().transfer_type = TransferType::H2D;
  }
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  EXPECT_EQ(worker.transfer_kv_blocks(kBatchId, offload_info),
            block_types.size());
  EXPECT_EQ(worker.transfer_kv_blocks(kBatchId, load_info), block_types.size());

  ModelInputParams target_input_params;
  target_input_params.meta.batch_id = kBatchId;
  target_input_params.meta.requires_host_restore = true;
  worker.set_hierarchy_layer_synchronizer(target_input_params);
  EXPECT_FALSE(target_input_params.meta.requires_host_restore);
  for (uint32_t layer_index = 0; layer_index < 3; ++layer_index) {
    ASSERT_TRUE(target_input_params.synchronize_layer(layer_index));
  }
  ModelInputParams draft_input_params = target_input_params;
  draft_ptr->set_hierarchy_layer_synchronizer(draft_input_params);
  EXPECT_FALSE(draft_input_params.meta.requires_host_restore);
  ASSERT_TRUE(draft_input_params.synchronize_draft_layer());

  for (BlockType block_type : block_types) {
    EXPECT_TRUE(target_ptr->blocks_equal(
        block_type, kSourceBlockId, kDestinationBlockId));
  }
  EXPECT_TRUE(draft_ptr->blocks_equal(
      BlockType::SWA, kSourceBlockId, kDestinationBlockId));
}

#if defined(USE_NPU)
// The SFA C8 packed KV cache only exists on NPU builds
// (kv_cache_shape.cpp gates the packed shape under `#if defined(USE_NPU)`),
// so guard the check on the same build. The predicate itself is host code
// and does not touch the device.
constexpr int64_t kMtpTestBlockCount = 8;
constexpr int64_t kMtpTestBlockSize = 16;
constexpr int64_t kMtpTestKvLoraRank = 512;
constexpr int64_t kMtpTestQkRopeHeadDim = 64;
constexpr int64_t kMtpTestIndexHeadDim = 128;
// Packed C8 layout: kv_lora + 2 * rope_dim + 4 * (kv_lora / tile=128).
constexpr int64_t kMtpTestExpectedPackedHeadDim =
    kMtpTestKvLoraRank + 2 * kMtpTestQkRopeHeadDim +
    4 * (kMtpTestKvLoraRank / 128);

ModelArgs make_glm_dsa_target_args() {
  ModelArgs args;
  args.model_type("glm_moe_dsa")
      .enable_mla(true)
      .n_heads(64)
      .n_kv_heads(1)
      .head_dim(kMtpTestKvLoraRank + kMtpTestQkRopeHeadDim)
      .kv_lora_rank(kMtpTestKvLoraRank)
      .qk_rope_head_dim(kMtpTestQkRopeHeadDim)
      .index_n_heads(1)
      .index_head_dim(kMtpTestIndexHeadDim);
  return args;
}

KVCacheCapacity make_glm_dsa_target_capacity(bool enable_packed_c8) {
  KVCacheCapacity capacity;
  capacity.n_blocks(kMtpTestBlockCount)
      .block_size(kMtpTestBlockSize)
      .enable_mla_kv_cache_quant(enable_packed_c8);
  return capacity;
}

TEST(SpeculativeDraftKVCacheShapeTest,
     PropagatesPackedC8LayoutOnGlmMoeDsaMtpDraft) {
  const ModelArgs target_args = make_glm_dsa_target_args();
  const KVCacheCapacity target_capacity =
      make_glm_dsa_target_capacity(/*enable_packed_c8=*/true);
  const KVCacheShape target_shape(
      target_capacity, target_args, /*world_size=*/1);
  ASSERT_TRUE(target_shape.has_key_cache_shape());
  ASSERT_EQ(target_shape.key_cache_shape().back(),
            kMtpTestExpectedPackedHeadDim);

  ModelArgs draft_args = target_args;
  draft_args.model_type("glm_moe_dsa_mtp");

  const KVCacheShape draft_shape =
      build_speculative_draft_kv_cache_shape(target_shape,
                                             draft_args,
                                             kMtpTestBlockSize,
                                             /*draft_world_size=*/1,
                                             /*kv_cache_dtype=*/"int8");
  ASSERT_TRUE(draft_shape.has_key_cache_shape());
  EXPECT_EQ(draft_shape.key_cache_shape(), target_shape.key_cache_shape());
  ASSERT_TRUE(draft_shape.has_value_cache_shape());
  EXPECT_EQ(draft_shape.value_cache_shape(), target_shape.value_cache_shape());
  EXPECT_EQ(draft_shape.key_cache_shape().back(),
            kMtpTestExpectedPackedHeadDim);
}

TEST(SpeculativeDraftKVCacheShapeTest, KeepsBf16LayoutWhenKvCacheDtypeIsAuto) {
  const ModelArgs target_args = make_glm_dsa_target_args();
  // enable_mla_kv_cache_quant defaults to false; kv_cache_dtype auto keeps
  // the legacy BF16 [nope][rope] split (trailing dim is kv_lora, not the
  // 656B packed row).
  const KVCacheCapacity target_capacity =
      make_glm_dsa_target_capacity(/*enable_packed_c8=*/false);
  const KVCacheShape target_shape(
      target_capacity, target_args, /*world_size=*/1);
  ASSERT_TRUE(target_shape.has_key_cache_shape());
  EXPECT_EQ(target_shape.key_cache_shape().back(), kMtpTestKvLoraRank);

  ModelArgs draft_args = target_args;
  draft_args.model_type("glm_moe_dsa_mtp");

  const KVCacheShape draft_shape =
      build_speculative_draft_kv_cache_shape(target_shape,
                                             draft_args,
                                             kMtpTestBlockSize,
                                             /*draft_world_size=*/1,
                                             /*kv_cache_dtype=*/"auto");
  ASSERT_TRUE(draft_shape.has_key_cache_shape());
  EXPECT_EQ(draft_shape.key_cache_shape(), target_shape.key_cache_shape());
  EXPECT_EQ(draft_shape.key_cache_shape().back(), kMtpTestKvLoraRank);
}
#endif  // USE_NPU

}  // namespace
}  // namespace xllm
