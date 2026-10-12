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

#include <framework/core/device.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "base_executor_impl.h"
#include "core/common/constants.h"
#include "core/distributed_runtime/engine.h"
#include "core/framework/batch/batch.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_output.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/word_embedding.h"
#include "mlu_graph_executor_impl.h"
#include "models/llm/mlu/glm5_next_graph.h"
#include "models/llm/mlu/mtp_topk_state.h"
#include "models/model_registry.h"
#include "platform/device.h"
#include "runtime/decode_graph_bucket.h"
#include "runtime/options.h"
#include "tests/core/layers/mlu/tests_utils.h"
#include "util/env_var.h"

namespace xllm {
namespace {

class ScopedConfigSnapshot final {
 public:
  ScopedConfigSnapshot()
      : max_tokens_for_graph_mode_(
            ExecutionConfig::get_instance().max_tokens_for_graph_mode()),
        enable_graph_mode_decode_no_padding_(
            ExecutionConfig::get_instance()
                .enable_graph_mode_decode_no_padding()) {}

  ~ScopedConfigSnapshot() {
    ExecutionConfig::get_instance().max_tokens_for_graph_mode(
        max_tokens_for_graph_mode_);
    ExecutionConfig::get_instance().enable_graph_mode_decode_no_padding(
        enable_graph_mode_decode_no_padding_);
  }

 private:
  int32_t max_tokens_for_graph_mode_;
  bool enable_graph_mode_decode_no_padding_;
};

class ScopedEnvVar final {
 public:
  ScopedEnvVar(std::string name, std::string value) : name_(std::move(name)) {
    const char* old_value = std::getenv(name_.c_str());
    if (old_value != nullptr) {
      old_value_ = old_value;
    }
    CHECK_EQ(setenv(name_.c_str(), value.c_str(), /*overwrite=*/1), 0);
  }

  ~ScopedEnvVar() {
    if (old_value_.has_value()) {
      CHECK_EQ(setenv(name_.c_str(), old_value_->c_str(), /*overwrite=*/1), 0);
      return;
    }
    CHECK_EQ(unsetenv(name_.c_str()), 0);
  }

 private:
  std::string name_;
  std::optional<std::string> old_value_;
};

class CompatibilityShapeEngine final : public Engine {
 public:
  ForwardOutput step(std::vector<Batch>& /*batch*/) override { return {}; }

  void update_last_step_result(std::vector<Batch>& /*batch*/) override {}

  std::vector<int64_t> get_active_activation_memory() const override {
    return {};
  }
};

TEST(DecodeGraphBucketTest, MapsTokenBucketsWithAndWithoutPadding) {
  EXPECT_EQ(runtime::get_decode_graph_token_bucket(
                /*num_tokens=*/1, /*enable_no_padding=*/false),
            1);
  EXPECT_EQ(runtime::get_decode_graph_token_bucket(
                /*num_tokens=*/2, /*enable_no_padding=*/false),
            2);
  EXPECT_EQ(runtime::get_decode_graph_token_bucket(
                /*num_tokens=*/3, /*enable_no_padding=*/false),
            4);
  EXPECT_EQ(runtime::get_decode_graph_token_bucket(
                /*num_tokens=*/8, /*enable_no_padding=*/false),
            8);
  EXPECT_EQ(runtime::get_decode_graph_token_bucket(
                /*num_tokens=*/9, /*enable_no_padding=*/false),
            16);
  EXPECT_EQ(runtime::get_decode_graph_token_bucket(
                /*num_tokens=*/17, /*enable_no_padding=*/false),
            32);
  EXPECT_EQ(runtime::get_decode_graph_token_bucket(
                /*num_tokens=*/17, /*enable_no_padding=*/true),
            17);
}

TEST(DecodeGraphWarmupConfigTest, DefaultEngineUsesSingleTokenDecodeConfig) {
  CompatibilityShapeEngine engine;

  const runtime::DecodeGraphWarmupConfig warmup_config =
      engine.decode_graph_warmup_config();

  EXPECT_EQ(warmup_config.num_decoding_tokens, 1);
  EXPECT_EQ(warmup_config.num_speculative_tokens, 0);
  EXPECT_FALSE(warmup_config.enable_graph_mode_decode_no_padding);
}

}  // namespace

class MockCausalLM : public CausalLM {
 public:
  MockCausalLM(const torch::TensorOptions& options) : options_(options) {
    auto weight = torch::randn({1024, 1024}, options_) * 0.02;
    weight_ = register_parameter("weight", weight, false);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params) override {
    (void)tokens;
    (void)positions;
    (void)kv_caches;
    ++forward_cnt_;
    last_tokens_size_ = tokens.size(0);
    last_dp_token_nums_ = params.parallel.dp_global_token_nums;
    last_kv_max_seq_len_ = params.meta.kv_max_seq_len;
    last_block_table_width_ = params.attention.device.block_tables.defined()
                                  ? params.attention.device.block_tables.size(1)
                                  : 0;
    auto hidden_states = params.embedding.input_embedding.matmul(weight_);
    ModelOutput output(hidden_states);
    if (return_aux_hidden_states_) {
      output.aux_hidden_states = hidden_states + 1;
    }
    output.mtp_topk_state = mtp_topk_state_;
    return output;
  }
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    (void)seleted_idxes;
    return hidden_states;
  }
  int32_t forward_cnt() const { return forward_cnt_; }
  int64_t last_tokens_size() const { return last_tokens_size_; }
  const std::vector<int32_t>& last_dp_token_nums() const {
    return last_dp_token_nums_;
  }
  int64_t last_kv_max_seq_len() const { return last_kv_max_seq_len_; }
  int64_t last_block_table_width() const { return last_block_table_width_; }
  void return_aux_hidden_states(bool value) {
    return_aux_hidden_states_ = value;
  }
  void set_mtp_topk_state(MtpTopkStatePtr state) {
    mtp_topk_state_ = std::move(state);
  }
  void load_model(std::unique_ptr<ModelLoader> loader) override {}
  torch::Device device() const override { return options_.device(); }
  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {}
  void update_expert_weight(int32_t layer_id) override {}
  const torch::TensorOptions& options() const override { return options_; }

 private:
  torch::Tensor input_;
  torch::Tensor weight_;
  MtpTopkStatePtr mtp_topk_state_;
  torch::TensorOptions options_;
  bool return_aux_hidden_states_ = false;
  int32_t forward_cnt_ = 0;
  int64_t last_tokens_size_ = 0;
  int64_t last_kv_max_seq_len_ = 0;
  int64_t last_block_table_width_ = 0;
  std::vector<int32_t> last_dp_token_nums_;
};

// Make optional-input branches and block-table refresh observable in replay.
class GraphInputsModel final : public MockCausalLM {
 public:
  explicit GraphInputsModel(const torch::TensorOptions& options)
      : MockCausalLM(options) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& caches,
                      const ModelInputParams& params) override {
    ModelOutput output =
        MockCausalLM::forward(tokens, positions, caches, params);
    if (params.linear_state_validity_mask_tensor.defined()) {
      output.hidden_states +=
          params.linear_state_validity_mask_tensor.unsqueeze(1);
    }
    for (const auto& table : params.multi_block_tables) {
      output.hidden_states += table.sum(1, /*keepdim=*/true);
    }
    return output;
  }
};

// A replay must refresh request metadata and clear unused table rows/columns.
class ReplayInputModel final : public MockCausalLM {
 public:
  using MockCausalLM::MockCausalLM;

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& caches,
                      const ModelInputParams& params) override {
    ModelOutput output =
        MockCausalLM::forward(tokens, positions, caches, params);
    const auto ints = tokens.options().dtype(torch::kInt32);
    const torch::Tensor state_ids =
        params.embedding.linear_state_indices.defined()
            ? params.embedding.linear_state_indices
            : torch::tensor(params.embedding.linear_state_ids, ints);
    const torch::Tensor mask =
        params.linear_state_validity_mask_tensor.defined()
            ? params.linear_state_validity_mask_tensor
            : torch::tensor(params.linear_state_validity_mask, ints);
    torch::Tensor values = output.hidden_states.to(torch::kFloat32);
    values += state_ids.to(torch::kFloat32).unsqueeze(1);
    values += mask.to(torch::kFloat32).unsqueeze(1);
    values += params.num_accepted_tokens.to(torch::kFloat32).unsqueeze(1);
    values += params.attention.device.block_tables.sum().to(torch::kFloat32);
    for (const torch::Tensor& table : params.multi_block_tables) {
      values += table.sum().to(torch::kFloat32);
    }
    output.hidden_states = values;
    return output;
  }
};

// Make mode, input format, and history changes observable without depending
// on a particular layer's weights or kernel numerical tolerance.
class GraphContractModel final : public MockCausalLM {
 public:
  explicit GraphContractModel(const torch::TensorOptions& options,
                              bool dual_input = false)
      : MockCausalLM(options),
        group_(
            std::make_unique<layer::test::MockProcessGroup>(options.device())),
        embedding_(make_embedding(options)),
        dual_input_(dual_input) {}

  layer::WordEmbedding get_word_embedding() override { return embedding_; }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& /*caches*/,
                      const ModelInputParams& params) override {
    ++calls_;
    torch::Tensor token_embedding;
    if (dual_input_ || !params.embedding.input_embedding.defined()) {
      token_embedding = embedding_(tokens);
    }
    torch::Tensor embedding = params.embedding.input_embedding.defined()
                                  ? params.embedding.input_embedding
                                  : token_embedding;
    torch::Tensor values = embedding.sum(1).to(torch::kFloat32);
    if (dual_input_) {
      values += token_embedding.sum(1).to(torch::kFloat32);
    }
    values = values + (positions.dim() == 2 ? positions.sum(0) : positions);
    if (params.is_spec_verify) {
      values = values + params.num_accepted_tokens;
    }
    if (params.meta.batch_forward_type.is_chunked_prefill()) {
      values = values + 100;
    }
    values = values + params.meta.q_max_seq_len;
    return ModelOutput(values.unsqueeze(1));
  }

  int32_t calls() const { return calls_; }

  void set_token_weights() {
    for (int32_t token = 1; token <= 16; ++token) {
      embedding_->weight().select(0, token).fill_(token * 0.01f);
    }
  }

 private:
  layer::WordEmbedding make_embedding(const torch::TensorOptions& options) {
    ParallelArgs args(0, 1, group_.get());
    args.tp_group_ = group_.get();
    layer::WordEmbedding embedding(1024, 1024, args, options);
    embedding->weight().fill_(0.01);
    return embedding;
  }

  std::unique_ptr<layer::test::MockProcessGroup> group_;
  layer::WordEmbedding embedding_{nullptr};
  bool dual_input_;
  int32_t calls_ = 0;
};

class PositionEchoModel final : public MockCausalLM {
 public:
  explicit PositionEchoModel(const torch::TensorOptions& options)
      : MockCausalLM(options) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& /*caches*/,
                      const ModelInputParams& /*params*/) override {
    ++calls_;
    torch::Tensor values = tokens.to(torch::kFloat32) * 1000 +
                           positions.select(0, 0).to(torch::kFloat32) +
                           positions.select(0, 1).to(torch::kFloat32) * 10 +
                           positions.select(0, 2).to(torch::kFloat32) * 100;
    return ModelOutput(values.unsqueeze(1));
  }

  int32_t calls() const { return calls_; }

 private:
  int32_t calls_ = 0;
};

namespace {

// Exercise the production GLM metadata through real executor capture/replay.
// Reading a page payload through KPool's table makes stale page IDs observable.
class GlmGraphMetadataModel final : public MockCausalLM {
 public:
  explicit GlmGraphMetadataModel(const torch::TensorOptions& options)
      : MockCausalLM(options),
        pages_(torch::arange(256, options.dtype(torch::kInt32)) * 7) {}

  bool requires_graph_forward_metadata() override { return true; }

  std::unique_ptr<ModelGraphMetadataState> create_graph_forward_metadata_state()
      override {
    return std::make_unique<mlu::model::Glm5NextGraphMetadataState>();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& params) override {
    mlu::model::Glm5NextGraphMetadata::prepare(state, positions, params);
  }

  ModelOutput forward(const torch::Tensor& /*tokens*/,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelInputParams& params) override {
    ++capture_count_;
    const auto& metadata = *params.attn_metadata;
    const auto& batch = *metadata.kpool_batch_metadata;
    CHECK_EQ(batch.row_batch.numel(), positions.numel());
    CHECK_EQ(batch.tail_indices.numel(), positions.numel());
    auto block_ids = batch.block_table.select(1, 0).to(torch::kInt64);
    auto values = pages_.index_select(0, block_ids) + metadata.kv_seq_lens +
                  batch.tail_indices;
    return ModelOutput(values.unsqueeze(1).expand({positions.numel(), 1024}));
  }

  int32_t capture_count() const { return capture_count_; }

 private:
  torch::Tensor pages_;
  int32_t capture_count_ = 0;
};

// No production model name is used: graph dispatch follows the input contract.
class GroupedCausalLM : public MockCausalLM {
 public:
  explicit GroupedCausalLM(const torch::TensorOptions& options)
      : MockCausalLM(options) {}

  bool requires_graph_forward_metadata() override { return true; }

  std::unique_ptr<ModelGraphMetadataState> create_graph_forward_metadata_state()
      override {
    return std::make_unique<mlu::model::Glm5NextGraphMetadataState>();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& params) override {
    mlu::model::Glm5NextGraphMetadata::prepare(state, positions, params);
  }
};

class GroupedBlockTableModel final : public GroupedCausalLM {
 public:
  using GroupedCausalLM::GroupedCausalLM;

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params) override {
    MockCausalLM::forward(tokens, positions, kv_caches, params);
    const auto& batch = *params.attn_metadata->kpool_batch_metadata;
    return ModelOutput(batch.block_table.index_select(0, batch.row_batch)
                           .select(1, 0)
                           .reshape({-1, 1}));
  }
};

class RequestSlotsModel final : public GroupedCausalLM {
 public:
  using GroupedCausalLM::GroupedCausalLM;
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& caches,
                      const ModelInputParams& params) override {
    MockCausalLM::forward(tokens, positions, caches, params);
    const auto& batch = *params.attn_metadata->kpool_batch_metadata;
    return ModelOutput(
        params.embedding.linear_state_indices.index_select(0, batch.row_batch)
            .reshape({-1, 1}));
  }
};

class StatefulVerifyModel final : public GroupedCausalLM {
 public:
  explicit StatefulVerifyModel(const torch::TensorOptions& options)
      : GroupedCausalLM(options),
        state_(torch::zeros({64}, options.dtype(torch::kFloat32))) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelInputParams& params) override {
    ++calls_;
    const int64_t batch = params.meta.num_sequences;
    const auto ids = params.embedding.linear_state_indices.to(torch::kInt64);
    auto value = tokens.to(torch::kFloat32).view({batch, -1}) +
                 positions.to(torch::kFloat32).view({batch, -1}) +
                 params.attention.device.new_cache_slots.view({batch, -1});
    const auto metadata = params.attn_metadata
                              ? params.attn_metadata
                              : std::make_shared<layer::AttentionMetadata>(
                                    layer::AttentionMetadataBuilder::build(
                                        params,
                                        /*enable_mla=*/true,
                                        /*compute_dtype=*/"half",
                                        std::nullopt,
                                        tokens.device()));
    auto history = metadata->kv_seq_lens;
    if (metadata->has_initial_states.defined()) {
      history = history + metadata->has_initial_states;
    }
    auto next = state_.index_select(0, ids) + params.num_accepted_tokens +
                history + params.attention.device.block_tables.select(1, 0);
    state_.index_copy_(
        0, ids, torch::where(ids > 0, next + 1, state_.index_select(0, ids)));
    return ModelOutput((value + next.unsqueeze(1)).reshape({-1, 1}));
  }

  int32_t calls() const { return calls_; }
  const torch::Tensor& state() const { return state_; }

 private:
  torch::Tensor state_;
  int32_t calls_ = 0;
};

TEST(MluSpecVerifyGraphTest, ReplaysUpdatedInputsAndAdvancesStateExactlyOnce) {
  ScopedEnvVar graph_kv_cap("XLLM_GRAPH_INDEX_HISTORY_MAX_KV", "4096");
  const torch::NoGradGuard no_grad;
  const torch::Device device("mlu:0");
  const auto floats =
      torch::TensorOptions().device(device).dtype(torch::kFloat32);
  const auto ints = floats.dtype(torch::kInt32);
  StatefulVerifyModel graph_model(floats);
  StatefulVerifyModel eager_model(floats);
  ModelArgs args;
  args.index_kpool(4).index_kpool_compress(true);
  args.model_type("glm5_next")
      .dtype("float32")
      .hidden_size(1)
      .max_position_embeddings(4096);
  runtime::Options options;
  options.block_size(16)
      .num_decoding_tokens(4)
      .dp_size(1)
      .cp_size(1)
      .enable_speculative_decode(true)
      .speculative_algorithm("MTP")
      .is_draft_engine(false);
  mlu::MluGraphExecutorImpl executor(&graph_model, args, device, options);
  std::vector<KVCache> caches;
  caches.emplace_back(
      LinearAttentionKVCacheTensors{torch::Tensor(), graph_model.state()});
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.is_spec_verify = true;
  for (int32_t width : {2, 3, 4, 5}) {
    params.meta.q_max_seq_len = width;
    for (int32_t round = 0; round < 4; ++round) {
      const int32_t requests = round == 1 ? 7 : 5;
      params.meta.num_sequences = requests;
      params.meta.kv_max_seq_len = round == 2 ? 100 : 32;
      params.attention.host.q_seq_lens.resize(requests + 1);
      for (int32_t i = 0; i <= requests; ++i) {
        params.attention.host.q_seq_lens[i] = i * width;
      }
      params.attention.device.q_seq_lens =
          torch::tensor(params.attention.host.q_seq_lens, ints);
      auto tokens = torch::arange(requests * width, ints) + round;
      auto positions = tokens + 12;
      params.embedding.linear_state_ids.resize(requests);
      std::iota(params.embedding.linear_state_ids.begin(),
                params.embedding.linear_state_ids.end(),
                round + 1);
      params.embedding.linear_state_indices =
          torch::tensor(params.embedding.linear_state_ids, ints);
      params.linear_state_validity_mask.assign(requests, round % 2);
      params.num_accepted_tokens =
          torch::full({requests}, round % width + 1, ints);
      params.attention.device.kv_seq_lens =
          torch::arange(requests + 1, ints) * (16 + round);
      params.attention.device.new_cache_slots = tokens + 32 * round;
      params.attention.device.block_tables =
          torch::full({requests, round == 2 ? 8 : 3}, round + 1, ints);
      // Both sources are legal. When supplied, the device mask must win even
      // when its values differ from the host mask used for metadata shape.
      params.linear_state_validity_mask_tensor =
          round % 2 == 0 ? torch::full({requests}, 1, ints.dtype(torch::kBool))
                         : torch::Tensor();
      auto expected = eager_model.forward(tokens, positions, caches, params);
      if (round % 2 == 1) {
        params.embedding.linear_state_indices = torch::Tensor();
      }
      auto actual = executor.run(tokens, positions, caches, params);
      Device(device).synchronize_default_stream();
      EXPECT_TRUE(torch::equal(actual.hidden_states, expected.hidden_states));
      EXPECT_TRUE(torch::equal(graph_model.state(), eager_model.state()));
    }
  }
  // One fixed-capacity graph per width, each with warmup and capture only.
  EXPECT_EQ(graph_model.calls(), 8);
}

TEST(MluSpecVerifyGraphTest, OrdinaryStatefulDecodeAdvancesOncePerRun) {
  const torch::NoGradGuard no_grad;
  const torch::Device device("mlu:0");
  const auto floats =
      torch::TensorOptions().device(device).dtype(torch::kFloat32);
  const auto ints = floats.dtype(torch::kInt32);
  StatefulVerifyModel graph_model(floats);
  StatefulVerifyModel eager_model(floats);
  ModelArgs args;
  args.dtype("float32").hidden_size(1).max_position_embeddings(128);
  runtime::Options options;
  options.block_size(16).dp_size(1).cp_size(1);
  mlu::MluGraphExecutorImpl executor(&graph_model, args, device, options);
  std::vector<KVCache> caches;
  caches.emplace_back(
      LinearAttentionKVCacheTensors{torch::Tensor(), graph_model.state()});
  for (int32_t requests : {3, 4, 3, 5, 3}) {
    ModelInputParams params;
    params.meta.batch_forward_type = BatchForwardType::DECODE;
    params.meta.num_sequences = requests;
    params.meta.q_max_seq_len = 1;
    params.meta.kv_max_seq_len = 16;
    params.attention.host.q_seq_lens.resize(requests + 1);
    std::iota(params.attention.host.q_seq_lens.begin(),
              params.attention.host.q_seq_lens.end(),
              0);
    params.attention.device.q_seq_lens = torch::arange(requests + 1, ints);
    params.attention.device.kv_seq_lens =
        torch::arange(requests + 1, ints) * 16;
    params.attention.device.new_cache_slots = torch::arange(requests, ints);
    params.attention.device.block_tables = torch::ones({requests, 1}, ints);
    params.embedding.linear_state_ids.resize(requests);
    std::iota(params.embedding.linear_state_ids.begin(),
              params.embedding.linear_state_ids.end(),
              1);
    params.num_accepted_tokens = torch::ones({requests}, ints);
    params.linear_state_validity_mask.assign(requests, 1);
    // The eager reference consumes the same public device metadata contract.
    params.embedding.linear_state_indices =
        torch::tensor(params.embedding.linear_state_ids, ints);
    auto tokens = torch::arange(requests, ints);
    auto expected = eager_model.forward(tokens, tokens, caches, params);
    auto actual = executor.run(tokens, tokens, caches, params);
    EXPECT_TRUE(torch::equal(actual.hidden_states, expected.hidden_states));
    EXPECT_TRUE(torch::equal(graph_model.state(), eager_model.state()));
  }
  EXPECT_EQ(graph_model.calls(), 4);
}

}  // namespace

class MluGraphExecutorTest : public ::testing::Test {
 protected:
  MluGraphExecutorTest() = default;

  void SetUp() override {
    torch::Device device("mlu:0");
    tensor_options_ = torch::TensorOptions(torch::kBFloat16).device(device);

    model_args_.model_type("test_model");
    model_args_.dtype("bfloat16");
    model_args_.hidden_size(1024);
    model_args_.max_position_embeddings(2048);

    const uint32_t block_size = 16;
    options_.num_decoding_tokens(1);
    options_.block_size(block_size);

    model_ = std::make_unique<MockCausalLM>(tensor_options_);
    rebuild_impl();
  }

  ForwardInput prepare_inputs(int32_t batch_size, uint64_t seed) {
    Device device(tensor_options_.device());
    device.set_seed(seed);
    const int64_t max_seq_len = model_args_.max_position_embeddings();
    const uint32_t block_size = options_.block_size();
    const int64_t num_blocks_per_req =
        (max_seq_len + block_size - 1) / block_size + 1;
    auto int_tensor_options = tensor_options_.dtype(torch::kInt32);
    auto token_ids = torch::full({batch_size}, 1, int_tensor_options);
    auto positions = torch::full({batch_size}, 1, int_tensor_options);
    auto new_cache_slots =
        torch::randint(0, 10, {batch_size}, int_tensor_options);
    auto block_table = torch::randint(
        0, 10, {batch_size, num_blocks_per_req}, int_tensor_options);
    std::vector<int32_t> q_seq_lens_vec(batch_size + 1, 0);
    std::vector<int32_t> kv_seq_lens_vec(batch_size + 1, 0);
    for (int32_t i = 0; i < batch_size; ++i) {
      q_seq_lens_vec[i + 1] = q_seq_lens_vec[i] + 1;
      kv_seq_lens_vec[i + 1] = kv_seq_lens_vec[i] + 1;
    }
    auto q_seq_lens = torch::tensor(q_seq_lens_vec, int_tensor_options);
    auto kv_seq_lens = torch::tensor(kv_seq_lens_vec, int_tensor_options);
    auto input_embedding =
        torch::randn({batch_size, model_args_.hidden_size()}, tensor_options_) *
        0.1;
    ModelInputParams input_params;
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    input_params.meta.num_sequences = batch_size;
    input_params.meta.kv_max_seq_len = 1;
    input_params.meta.q_max_seq_len = 1;
    input_params.parallel.dp_global_token_nums = {1};
    input_params.parallel.dp_is_decode = {1};
    input_params.attention.device.new_cache_slots = new_cache_slots;
    input_params.attention.device.block_tables = block_table;
    input_params.attention.device.q_seq_lens = q_seq_lens;
    input_params.attention.device.kv_seq_lens = kv_seq_lens;
    input_params.attention.host.q_seq_lens = q_seq_lens_vec;
    input_params.attention.host.kv_seq_lens = kv_seq_lens_vec;
    input_params.embedding.input_embedding = input_embedding;

    kv_caches_.resize(batch_size);
    if (model_args_.index_kpool_compress()) {
      kv_caches_[0] = KVCache(LinearAttentionKVCacheTensors{
          torch::Tensor(), torch::zeros({1}, tensor_options_)});
    }
    ForwardInput input;
    input.token_ids = token_ids;
    input.positions = positions;
    input.input_params = input_params;
    return input;
  }

  void rebuild_impl() {
    if (model_->requires_graph_forward_metadata() &&
        model_args_.index_kpool() > 0) {
      MtpModelCapabilities capabilities;
      capabilities.supports_grouped_mtp_graph = true;
      capabilities.graph_history = MtpGraphHistoryPolicy::KPOOL_FIXED;
      ModelRegistry::register_mtp_capabilities("test_grouped_model",
                                               capabilities);
      ModelRegistry::register_graph_history_capacity(
          "test_grouped_model", [](const ModelArgs& args) {
            return std::min<int64_t>(
                util::get_int_env("XLLM_GRAPH_INDEX_HISTORY_MAX_KV", 32768),
                args.max_position_embeddings());
          });
      model_args_.model_type("test_grouped_model");
    }
    const torch::Device device("mlu:0");
    impl_ = std::make_unique<::xllm::mlu::MluGraphExecutorImpl>(
        model_.get(), model_args_, device, options_);
    base_impl_ = std::make_unique<BaseExecutorImpl>(
        model_.get(), model_args_, device, options_);
  }

  ModelArgs model_args_;
  torch::TensorOptions tensor_options_;
  runtime::Options options_;
  std::unique_ptr<MockCausalLM> model_;
  std::vector<KVCache> kv_caches_;
  std::unique_ptr<::xllm::mlu::MluGraphExecutorImpl> impl_;
  std::unique_ptr<BaseExecutorImpl> base_impl_;
};

TEST_F(MluGraphExecutorTest, GlmMetadataReplaysPageRemapsAndPaddedBuckets) {
  options_.enable_graph_mode_decode_no_padding(false);
  auto model = std::make_unique<GlmGraphMetadataModel>(tensor_options_);
  mlu::MluGraphExecutorImpl executor(
      model.get(), model_args_, tensor_options_.device(), options_);

  // Revisit the four-token graph after capturing a different bucket, then
  // change request count within that bucket and change only physical page IDs.
  const std::vector<int32_t> batch_sizes = {3, 3, 2, 4, 3, 3};
  for (size_t step = 0; step < batch_sizes.size(); ++step) {
    const int32_t rows = batch_sizes[step];
    auto input = prepare_inputs(rows, /*seed=*/97);
    const int32_t page_id = static_cast<int32_t>(step) + 1;
    const int32_t state_id = page_id + 10;
    auto& params = input.input_params;
    params.attention.device.block_tables.fill_(page_id);
    params.embedding.linear_state_ids.assign(rows, state_id);
    params.embedding.linear_state_indices =
        torch::full({rows}, state_id, tensor_options_.dtype(torch::kInt32));
    for (int32_t row = 0; row <= rows; ++row) {
      params.attention.host.kv_seq_lens[row] = row * page_id;
    }
    params.attention.device.kv_seq_lens =
        torch::tensor(params.attention.host.kv_seq_lens,
                      tensor_options_.dtype(torch::kInt32));
    params.meta.kv_max_seq_len = page_id;

    auto output =
        executor.run(input.token_ids, input.positions, kv_caches_, params);
    torch_mlu::synchronize();
    auto expected =
        torch::full({rows, 1024}, page_id * 8 + state_id, tensor_options_);
    EXPECT_TRUE(torch::equal(output.hidden_states, expected)) << step;
  }
  EXPECT_EQ(model->capture_count(), 2);
}

// Test graph creation and execution with different batch sizes
TEST_F(MluGraphExecutorTest, ReusesGraphAcrossSourceStridesAndRows) {
  model_ = std::make_unique<GraphInputsModel>(tensor_options_);
  rebuild_impl();
  for (int32_t round = 0; round < 4; ++round) {
    const int32_t rows = round % 2 == 0 ? 5 : 7;
    auto input = prepare_inputs(rows, /*seed=*/101 + round);
    auto& params = input.input_params;
    params.linear_state_validity_mask_tensor =
        torch::ones({rows}, tensor_options_.dtype(torch::kBool));
    params.multi_block_tables = {torch::full(
        {rows, 3}, round + 1, tensor_options_.dtype(torch::kInt32))};
    const auto strided = [](const torch::Tensor& tensor) {
      auto shape = tensor.sizes().vec();
      shape.back() *= 2;
      torch::Tensor storage = torch::zeros(shape, tensor.options());
      torch::Tensor view = storage.slice(
          tensor.dim() - 1, /*start=*/0, shape.back(), /*step=*/2);
      view.copy_(tensor);
      return view;
    };
    // Capture from non-contiguous inputs, then alternate with contiguous ones.
    if (round % 2 == 0) {
      input.token_ids = strided(input.token_ids);
      input.positions = strided(input.positions);
      params.embedding.input_embedding =
          strided(params.embedding.input_embedding);
      params.attention.device.q_seq_lens =
          strided(params.attention.device.q_seq_lens);
      params.attention.device.kv_seq_lens =
          strided(params.attention.device.kv_seq_lens);
      params.attention.device.new_cache_slots =
          strided(params.attention.device.new_cache_slots);
      params.linear_state_validity_mask_tensor =
          strided(params.linear_state_validity_mask_tensor);
      params.multi_block_tables[0] = strided(params.multi_block_tables[0]);
    }
    auto expected =
        model_->forward(input.token_ids, input.positions, kv_caches_, params);
    auto actual =
        impl_->run(input.token_ids, input.positions, kv_caches_, params);
    EXPECT_TRUE(torch::equal(expected.hidden_states, actual.hidden_states));
    // Stateless capture calls forward once; replay does not call it again.
    EXPECT_EQ(model_->forward_cnt(), round + 2);
  }
}

TEST_F(MluGraphExecutorTest, SeparatesBlockTableLayoutsAndRefreshesMasks) {
  model_ = std::make_unique<GraphInputsModel>(tensor_options_);
  rebuild_impl();
  for (int32_t round = 0; round < 8; ++round) {
    auto input = prepare_inputs(/*batch_size=*/2, /*seed=*/105 + round);
    auto& params = input.input_params;
    const int32_t variant = round % 4;
    // A deployed model has fixed mask requirements. Values change on replay;
    // manager count, column count and order change the captured table layout.
    params.linear_state_validity_mask_tensor =
        torch::full({2}, round / 4, tensor_options_.dtype(torch::kBool));
    const int32_t columns = variant == 2 ? 4 : 3;
    params.multi_block_tables = {torch::full(
        {2, columns}, round + 1, tensor_options_.dtype(torch::kInt32))};
    if (variant == 1 || variant == 3) {
      params.multi_block_tables.emplace_back(
          torch::full({2, 1}, round + 2, tensor_options_.dtype(torch::kInt32)));
    }
    if (variant == 1) {
      std::reverse(params.multi_block_tables.begin(),
                   params.multi_block_tables.end());
    }
    auto expected =
        model_->forward(input.token_ids, input.positions, kv_caches_, params);
    auto actual =
        impl_->run(input.token_ids, input.positions, kv_caches_, params);
    EXPECT_TRUE(torch::equal(expected.hidden_states, actual.hidden_states));
    EXPECT_EQ(model_->forward_cnt(), round + 1 + std::min(round + 1, 4));
  }
}

TEST_F(MluGraphExecutorTest, ReplaysShrunkRowsAndClearsTableTails) {
  model_ = std::make_unique<ReplayInputModel>(tensor_options_);
  rebuild_impl();
  const auto ints = tensor_options_.dtype(torch::kInt32);
  const std::vector<int32_t> rows_by_step = {7, 5, 7, 6};
  const std::vector<int32_t> columns_by_step = {6, 2, 4, 1};
  std::vector<torch::Tensor> saved_outputs;
  saved_outputs.reserve(rows_by_step.size());

  for (std::size_t step = 0; step < rows_by_step.size(); ++step) {
    const int32_t rows = rows_by_step[step];
    auto input = prepare_inputs(rows, /*seed=*/180 + step);
    auto& params = input.input_params;
    const int32_t value = static_cast<int32_t>(step) + 1;
    params.attention.device.block_tables =
        torch::full({rows, columns_by_step[step]}, value * 2, ints);
    params.multi_block_tables = {torch::full({rows, 2}, value * 3, ints),
                                 torch::full({rows, 3}, value * 5, ints)};
    params.embedding.linear_state_ids.resize(rows);
    std::iota(params.embedding.linear_state_ids.begin(),
              params.embedding.linear_state_ids.end(),
              value);
    params.linear_state_validity_mask.assign(rows, value % 2);
    params.num_accepted_tokens = torch::full({rows}, value + 1, ints);
    if (step % 2 == 0) {
      params.embedding.linear_state_indices =
          torch::tensor(params.embedding.linear_state_ids, ints);
      params.linear_state_validity_mask_tensor =
          torch::full({rows}, value % 2, ints.dtype(torch::kBool));
    } else {
      params.embedding.linear_state_indices = torch::Tensor();
      params.linear_state_validity_mask_tensor = torch::Tensor();
    }

    const torch::Tensor expected =
        model_->forward(input.token_ids, input.positions, kv_caches_, params)
            .hidden_states.clone();
    const int32_t calls_before_run = model_->forward_cnt();
    const torch::Tensor actual =
        impl_->run(input.token_ids, input.positions, kv_caches_, params)
            .hidden_states.clone();
    torch_mlu::synchronize();
    EXPECT_EQ(actual.size(0), rows);
    EXPECT_TRUE(torch::equal(actual, expected)) << step;
    EXPECT_EQ(model_->forward_cnt(), calls_before_run + (step == 0 ? 1 : 0));
    saved_outputs.emplace_back(actual);
  }
  EXPECT_FALSE(torch::equal(saved_outputs[0], saved_outputs[2]));
}

TEST_F(MluGraphExecutorTest, RejectsInputContractChangesBeforeBufferUpdates) {
  auto input = prepare_inputs(/*batch_size=*/2, /*seed=*/113);
  auto& params = input.input_params;
  mlu::GraphLayout layout;
  layout.num_reqs = 2;
  layout.padded_num_reqs = 2;
  layout.padded_num_tokens = 2;
  mlu::GraphPersistentParam buffers(input.token_ids,
                                    input.positions,
                                    params,
                                    layout,
                                    /*graph_max_kv_seq_len=*/2048,
                                    options_.block_size());
  ModelInputParams missing = params;
  missing.embedding.input_embedding = torch::Tensor();
  torch_mlu::synchronize();
  EXPECT_DEATH(buffers.update_input_buffer(
                   input.token_ids, input.positions, missing, layout),
               "input_embedding graph input presence changed");
  // Device conversions must happen before the death-test fork. The child
  // exercises only host-side validation and must not initialize an MLU context.
  auto wide_tokens = input.token_ids.to(torch::kInt64);
  auto wide_positions = input.positions.to(torch::kInt64);
  auto ranked_positions = input.positions.unsqueeze(0);
  torch_mlu::synchronize();
  EXPECT_DEATH(
      buffers.update_input_buffer(wide_tokens, input.positions, params, layout),
      "tokens graph input dtype changed");
  EXPECT_DEATH(buffers.update_input_buffer(
                   input.token_ids, wide_positions, params, layout),
               "positions graph input dtype changed");
  EXPECT_DEATH(buffers.update_input_buffer(
                   input.token_ids, ranked_positions, params, layout),
               "positions graph input rank changed");
  torch::Tensor embedding = params.embedding.input_embedding;
  params.embedding.input_embedding = embedding.to(torch::kFloat32);
  EXPECT_DEATH(buffers.update_input_buffer(
                   input.token_ids, input.positions, params, layout),
               "input_embedding graph input dtype changed");
  params.embedding.input_embedding = embedding.unsqueeze(1);
  EXPECT_DEATH(buffers.update_input_buffer(
                   input.token_ids, input.positions, params, layout),
               "input_embedding graph input rank changed");
  params.embedding.input_embedding = embedding.slice(1, 0, 512);
  EXPECT_DEATH(buffers.update_input_buffer(
                   input.token_ids, input.positions, params, layout),
               "input_embedding graph input feature dimension changed");
  params.embedding.input_embedding = embedding.slice(0, 0, 1);
  EXPECT_DEATH(buffers.update_input_buffer(
                   input.token_ids, input.positions, params, layout),
               "input_embedding graph input row count changed");
  params.embedding.input_embedding =
      torch::zeros({2, 1024}, torch::TensorOptions().dtype(embedding.dtype()));
  EXPECT_DEATH(buffers.update_input_buffer(
                   input.token_ids, input.positions, params, layout),
               "input_embedding graph input device changed");
}

TEST_F(MluGraphExecutorTest, CaptureValidatesExternalEmbeddingLayout) {
  auto input = prepare_inputs(/*batch_size=*/2, /*seed=*/272);
  mlu::GraphLayout layout;
  layout.num_reqs = 2;
  layout.padded_num_reqs = 2;
  layout.padded_num_tokens = 2;
  ModelInputParams params = input.input_params;
  params.embedding.input_embedding =
      torch::ones({2, 64}, tensor_options_.dtype(torch::kFloat32));
  mlu::GraphPersistentParam valid(input.token_ids,
                                  input.positions,
                                  params,
                                  layout,
                                  /*graph_max_kv_seq_len=*/2048,
                                  options_.block_size());
  EXPECT_TRUE(valid.params_.embedding.input_embedding.defined());
  EXPECT_EQ(valid.params_.embedding.input_embedding.size(1), 64);
  EXPECT_EQ(valid.params_.embedding.input_embedding.scalar_type(),
            torch::kFloat32);

  params.embedding.input_embedding = torch::ones({2, 4, 4}, tensor_options_);
  torch_mlu::synchronize();
  EXPECT_DEATH(mlu::GraphPersistentParam(input.token_ids,
                                         input.positions,
                                         params,
                                         layout,
                                         /*graph_max_kv_seq_len=*/2048,
                                         options_.block_size()),
               "input_embedding graph input rank changed");
  params.embedding.input_embedding = torch::ones({1, 64}, tensor_options_);
  EXPECT_DEATH(mlu::GraphPersistentParam(input.token_ids,
                                         input.positions,
                                         params,
                                         layout,
                                         /*graph_max_kv_seq_len=*/2048,
                                         options_.block_size()),
               "input_embedding graph input row count changed");
  params.embedding.input_embedding = torch::ones({2, 64});
  EXPECT_DEATH(mlu::GraphPersistentParam(input.token_ids,
                                         input.positions,
                                         params,
                                         layout,
                                         /*graph_max_kv_seq_len=*/2048,
                                         options_.block_size()),
               "input_embedding graph input device changed");
  EXPECT_DEATH(impl_->run(input.token_ids, input.positions, kv_caches_, params),
               "input_embedding graph input device changed");
}

TEST_F(MluGraphExecutorTest, PersistentInputsClearShrunkTailsAndDefaults) {
  auto full = prepare_inputs(/*batch_size=*/4, /*seed=*/114);
  const auto ints = tensor_options_.dtype(torch::kInt32);
  full.positions =
      torch::tensor({{1, 2, 3, 4}, {11, 12, 13, 14}, {21, 22, 23, 24}}, ints);
  auto& full_params = full.input_params;
  full_params.attention.device.new_cache_slots = torch::full({4}, 8, ints);
  full_params.attention.device.block_tables = torch::full({4, 5}, 9, ints);
  full_params.embedding.input_embedding =
      torch::full({4, 1024}, 7, tensor_options_);
  full_params.embedding.linear_state_ids = {1, 2, 3, 4};
  full_params.embedding.linear_state_indices =
      torch::tensor({1, 2, 3, 4}, ints);
  full_params.num_accepted_tokens = torch::full({4}, 8, ints);
  full_params.linear_state_validity_mask = {1, 1, 1, 1};
  full_params.linear_state_validity_mask_tensor =
      torch::ones({4}, ints.dtype(torch::kBool));
  full_params.multi_block_tables = {torch::full({4, 2}, 8, ints),
                                    torch::full({4, 3}, 9, ints)};
  mlu::GraphLayout layout;
  layout.num_reqs = 4;
  layout.padded_num_reqs = 4;
  layout.padded_num_tokens = 4;
  layout.cache_pad_slot = -7;
  mlu::GraphPersistentParam buffers(full.token_ids,
                                    full.positions,
                                    full_params,
                                    layout,
                                    /*graph_max_kv_seq_len=*/2048,
                                    /*main_block_table_columns=*/5);
  buffers.update_input_buffer(
      full.token_ids, full.positions, full_params, layout);

  auto short_input = prepare_inputs(/*batch_size=*/2, /*seed=*/115);
  short_input.token_ids = torch::tensor({21, 22}, ints);
  short_input.positions = torch::tensor({{5, 6}, {15, 16}, {25, 26}}, ints);
  auto& short_params = short_input.input_params;
  short_params.attention.host.kv_seq_lens = {0, 4, 8};
  short_params.attention.device.kv_seq_lens = torch::tensor({0, 2, 4}, ints);
  short_params.attention.device.new_cache_slots = torch::tensor({31, 32}, ints);
  short_params.attention.device.block_tables = torch::full({2, 2}, 3, ints);
  short_params.embedding.linear_state_ids = {11, 12};
  short_params.embedding.linear_state_indices = torch::Tensor();
  short_params.num_accepted_tokens = torch::tensor({4, 5}, ints);
  short_params.linear_state_validity_mask = {1, 0};
  short_params.linear_state_validity_mask_tensor = torch::Tensor();
  short_params.multi_block_tables = {torch::full({2, 2}, 4, ints),
                                     torch::full({2, 3}, 5, ints)};
  short_params.embedding.input_embedding =
      torch::full({2, 1024}, 2, tensor_options_);
  layout.num_reqs = 2;
  buffers.update_input_buffer(
      short_input.token_ids, short_input.positions, short_params, layout);
  torch_mlu::synchronize();

  EXPECT_TRUE(
      torch::equal(buffers.tokens_, torch::tensor({21, 22, 0, 0}, ints)));
  EXPECT_TRUE(torch::equal(
      buffers.positions_,
      torch::tensor({{5, 6, 0, 0}, {15, 16, 0, 0}, {25, 26, 0, 0}}, ints)));
  EXPECT_TRUE(torch::equal(buffers.params_.attention.device.new_cache_slots,
                           torch::tensor({31, 32, -7, -7}, ints)));
  EXPECT_TRUE(torch::equal(buffers.params_.attention.device.q_seq_lens,
                           torch::tensor({0, 1, 2, 3, 4}, ints)));
  EXPECT_TRUE(torch::equal(buffers.params_.attention.device.kv_seq_lens,
                           torch::tensor({0, 2, 4, 9, 10}, ints)));
  EXPECT_TRUE(torch::equal(
      buffers.params_.attention.device.block_tables,
      torch::tensor(
          {{3, 3, 0, 0, 0}, {3, 3, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
          ints)));
  EXPECT_TRUE(torch::equal(buffers.params_.embedding.linear_state_indices,
                           torch::tensor({11, 12, 0, 0}, ints)));
  EXPECT_TRUE(torch::equal(buffers.params_.num_accepted_tokens,
                           torch::tensor({4, 5, 1, 1}, ints)));
  EXPECT_TRUE(torch::equal(
      buffers.params_.linear_state_validity_mask_tensor,
      torch::tensor({true, false, false, false}, ints.dtype(torch::kBool))));
  EXPECT_TRUE(
      torch::equal(buffers.params_.embedding.input_embedding,
                   torch::cat({short_params.embedding.input_embedding,
                               torch::zeros({2, 1024}, tensor_options_)})));
  EXPECT_TRUE(torch::equal(buffers.params_.multi_block_tables[0],
                           torch::cat({short_params.multi_block_tables[0],
                                       torch::zeros({2, 2}, ints)})));
  EXPECT_TRUE(torch::equal(buffers.params_.multi_block_tables[1],
                           torch::cat({short_params.multi_block_tables[1],
                                       torch::zeros({2, 3}, ints)})));

  short_params.embedding.linear_state_indices = torch::tensor({13, 14}, ints);
  short_params.num_accepted_tokens = torch::Tensor();
  short_params.num_accepted_tokens_host = {9, 9};
  short_params.linear_state_validity_mask_tensor =
      torch::tensor({false, true}, ints.dtype(torch::kBool));
  buffers.update_input_buffer(
      short_input.token_ids, short_input.positions, short_params, layout);
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(buffers.params_.embedding.linear_state_indices,
                           torch::tensor({13, 14, 0, 0}, ints)));
  EXPECT_TRUE(torch::equal(buffers.params_.num_accepted_tokens,
                           torch::ones({4}, ints)));
  EXPECT_TRUE(torch::equal(
      buffers.params_.linear_state_validity_mask_tensor,
      torch::tensor({false, true, false, false}, ints.dtype(torch::kBool))));
  EXPECT_TRUE(
      torch::equal(buffers.params_.embedding.input_embedding,
                   torch::cat({short_params.embedding.input_embedding,
                               torch::zeros({2, 1024}, tensor_options_)})));

  short_params.attention.device.block_tables = torch::full({2, 7}, 6, ints);
  short_params.embedding.linear_state_indices = torch::Tensor();
  short_params.embedding.linear_state_ids.clear();
  short_params.linear_state_validity_mask_tensor = torch::Tensor();
  short_params.linear_state_validity_mask.clear();
  buffers.update_input_buffer(
      short_input.token_ids, short_input.positions, short_params, layout);
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(
      buffers.params_.attention.device.block_tables,
      torch::tensor(
          {{6, 6, 6, 6, 6}, {6, 6, 6, 6, 6}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}},
          ints)));
  EXPECT_TRUE(torch::equal(buffers.params_.embedding.linear_state_indices,
                           torch::zeros({4}, ints)));
  EXPECT_TRUE(torch::equal(buffers.params_.linear_state_validity_mask_tensor,
                           torch::zeros({4}, ints.dtype(torch::kBool))));
}

TEST_F(MluGraphExecutorTest, InvalidLayoutDoesNotPoisonGraphCache) {
  for (int32_t invalid = 0; invalid < 4; ++invalid) {
    auto input = prepare_inputs(/*batch_size=*/4, /*seed=*/114);
    auto& params = input.input_params;
    if (invalid == 0) {
      params.meta.q_max_seq_len = 2;
    } else if (invalid == 1) {
      params.attention.host.q_seq_lens = {0, 1, 3, 4};
    } else if (invalid == 2) {
      params.attention.host.kpool_query_lens = {2, 1};
    } else {
      params.attention.host.q_seq_lens = {1, 2, 3, 4, 5};
    }
    const int32_t before = model_->forward_cnt();
    impl_->run(input.token_ids, input.positions, kv_caches_, params);
    impl_->run(input.token_ids, input.positions, kv_caches_, params);
    EXPECT_EQ(model_->forward_cnt(), before + 2);
  }
  auto input = prepare_inputs(/*batch_size=*/4, /*seed=*/115);
  const int32_t before = model_->forward_cnt();
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  EXPECT_EQ(model_->forward_cnt(), before + 1);
}

TEST_F(MluGraphExecutorTest, DifferentBatchSizes) {
  // Test with different batch sizes to ensure graph creation works
  const std::vector<uint32_t> batch_sizes = {1, 3, 13, 21, 65};
  for (auto batch_size : batch_sizes) {
    auto forward_input = prepare_inputs(batch_size, 1);
    auto eager_model_output = base_impl_->run({forward_input.token_ids},
                                              {forward_input.positions},
                                              kv_caches_,
                                              {forward_input.input_params});
    auto eager_output = eager_model_output.hidden_states;
    const int32_t calls_after_eager = model_->forward_cnt();

    auto graph_model_output = impl_->run({forward_input.token_ids},
                                         {forward_input.positions},
                                         kv_caches_,
                                         {forward_input.input_params});
    auto graph_output = graph_model_output.hidden_states;
    EXPECT_EQ(model_->forward_cnt(), calls_after_eager + 1);

    auto replay_model_output = impl_->run({forward_input.token_ids},
                                          {forward_input.positions},
                                          kv_caches_,
                                          {forward_input.input_params});
    auto replay_output = replay_model_output.hidden_states;
    EXPECT_EQ(model_->forward_cnt(), calls_after_eager + 1);

    CHECK_EQ(eager_output.sizes(), graph_output.sizes());
    CHECK_EQ(eager_output.sizes(), replay_output.sizes());
    // Compare outputs - should be identical
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::allclose(eager_output, graph_output, 1e-5, 1e-6));
    EXPECT_TRUE(torch::allclose(eager_output, replay_output, 1e-5, 1e-6));
  }
}

// Test multiple runs to verify consistency across different execution modes
TEST_F(MluGraphExecutorTest, MluGraphExecutorVsBaseExecutorImplMultipleRuns) {
  int32_t batch_size = 5;
  int32_t seed = 42;
  auto forward_input = prepare_inputs(batch_size, seed);
  auto eager_model_output = base_impl_->run({forward_input.token_ids},
                                            {forward_input.positions},
                                            kv_caches_,
                                            {forward_input.input_params});
  auto eager_output = eager_model_output.hidden_states;

  auto graph_model_output = impl_->run({forward_input.token_ids},
                                       {forward_input.positions},
                                       kv_caches_,
                                       {forward_input.input_params});
  auto graph_output = graph_model_output.hidden_states;

  CHECK_EQ(eager_output.sizes(), graph_output.sizes());
  // Compare outputs - should be identical
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(eager_output, graph_output, 1e-5, 1e-6));

  // Run multiple times and compare results
  const int num_runs = 5;
  auto base_forward_input = prepare_inputs(batch_size + 1, seed);
  auto replay_forward_input = prepare_inputs(batch_size + 1, seed);
  EXPECT_TRUE(torch::allclose(
      base_forward_input.input_params.embedding.input_embedding,
      replay_forward_input.input_params.embedding.input_embedding,
      1e-5,
      1e-6));

  for (int i = 0; i < num_runs; ++i) {
    auto base_model_output = base_impl_->run({base_forward_input.token_ids},
                                             {base_forward_input.positions},
                                             kv_caches_,
                                             {base_forward_input.input_params});
    auto base_output = base_model_output.hidden_states;

    auto replay_model_output = impl_->run({replay_forward_input.token_ids},
                                          {replay_forward_input.positions},
                                          kv_caches_,
                                          {replay_forward_input.input_params});
    auto replay_output = replay_model_output.hidden_states;
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::allclose(base_output, replay_output, 1e-5, 1e-6));
    base_forward_input.input_params.embedding.input_embedding = base_output;
    replay_forward_input.input_params.embedding.input_embedding = replay_output;
    CHECK_EQ(base_output.sizes(), replay_output.sizes());
  }

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(
      base_forward_input.input_params.embedding.input_embedding,
      replay_forward_input.input_params.embedding.input_embedding,
      1e-5,
      1e-6));
}

TEST_F(MluGraphExecutorTest, DraftDecodeCapturesThenReplays) {
  model_args_.model_type("qwen3_5_moe_mtp");
  options_.is_draft_engine(true);
  rebuild_impl();

  const int32_t batch_size = 5;
  const uint64_t seed = 7;
  auto forward_input = prepare_inputs(batch_size, seed);

  auto eager_model_output = base_impl_->run({forward_input.token_ids},
                                            {forward_input.positions},
                                            kv_caches_,
                                            {forward_input.input_params});
  auto eager_output = eager_model_output.hidden_states;

  auto first_impl_output = impl_
                               ->run({forward_input.token_ids},
                                     {forward_input.positions},
                                     kv_caches_,
                                     {forward_input.input_params})
                               .hidden_states;
  auto second_impl_output = impl_
                                ->run({forward_input.token_ids},
                                      {forward_input.positions},
                                      kv_caches_,
                                      {forward_input.input_params})
                                .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(eager_output, first_impl_output, 1e-5, 1e-6));
  EXPECT_TRUE(
      torch::allclose(first_impl_output, second_impl_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), 2);
}

TEST_F(MluGraphExecutorTest, DraftEagerDoesNotExposeAuxWhenDisabled) {
  model_->return_aux_hidden_states(true);
  options_.is_draft_engine(true);
  options_.enable_graph_aux_hidden_states(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  const uint64_t seed = 17;
  auto forward_input = prepare_inputs(batch_size, seed);

  ModelOutput output = impl_->run({forward_input.token_ids},
                                  {forward_input.positions},
                                  kv_caches_,
                                  {forward_input.input_params});

  EXPECT_FALSE(output.aux_hidden_states.defined());
  EXPECT_EQ(model_->forward_cnt(), 1);
}

TEST_F(MluGraphExecutorTest, DraftEagerPreservesTypedTopkWhenAuxIsDisabled) {
  mlu::model::MluMtpTopkState::LayerStates expected_layers;
  expected_layers.emplace_back(layer::DsaTopkState(
      torch::tensor({{1, 2}, {3, 4}}, tensor_options_.dtype(torch::kInt32)),
      torch::tensor({5, 6}, tensor_options_.dtype(torch::kInt32))));
  const MtpTopkStatePtr expected_state =
      std::make_shared<mlu::model::MluMtpTopkState>(std::move(expected_layers));
  model_->return_aux_hidden_states(true);
  model_->set_mtp_topk_state(expected_state);
  model_args_.model_type("test_mtp");
  model_args_.index_share_for_mtp_iteration(true);
  model_args_.index_n_heads(1);
  model_args_.index_head_dim(1);
  model_args_.index_topk(1);
  options_.is_draft_engine(true);
  options_.enable_graph_aux_hidden_states(false);
  rebuild_impl();

  const int32_t batch_size = 2;
  const uint64_t seed = 23;
  auto forward_input = prepare_inputs(batch_size, seed);

  ModelOutput output = impl_->run({forward_input.token_ids},
                                  {forward_input.positions},
                                  kv_caches_,
                                  {forward_input.input_params});

  EXPECT_EQ(output.mtp_topk_state.get(), expected_state.get());
  EXPECT_FALSE(output.aux_hidden_states.defined());
  EXPECT_EQ(model_->forward_cnt(), 1);
}

TEST_F(MluGraphExecutorTest, TargetDecodeCapturesThenReplays) {
  options_.is_draft_engine(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  const uint64_t seed = 11;
  auto forward_input = prepare_inputs(batch_size, seed);

  auto first_impl_output = impl_
                               ->run({forward_input.token_ids},
                                     {forward_input.positions},
                                     kv_caches_,
                                     {forward_input.input_params})
                               .hidden_states;
  auto second_impl_output = impl_
                                ->run({forward_input.token_ids},
                                      {forward_input.positions},
                                      kv_caches_,
                                      {forward_input.input_params})
                                .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(
      torch::allclose(first_impl_output, second_impl_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), 1);
}

TEST_F(MluGraphExecutorTest, Glm52DecodeReusesGraphAcrossBlockTableWidths) {
  model_args_.model_type("glm_moe_dsa");
  options_.is_draft_engine(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  auto short_history = prepare_inputs(batch_size, /*seed=*/31);
  short_history.input_params.attention.device.block_tables =
      short_history.input_params.attention.device.block_tables.narrow(
          /*dim=*/1, /*start=*/0, /*length=*/2);
  short_history.input_params.meta.kv_max_seq_len = 32;

  auto long_history = prepare_inputs(batch_size, /*seed=*/37);
  long_history.input_params.attention.device.block_tables =
      long_history.input_params.attention.device.block_tables.narrow(
          /*dim=*/1, /*start=*/0, /*length=*/17);
  long_history.input_params.meta.kv_max_seq_len = 272;

  impl_->run({short_history.token_ids},
             {short_history.positions},
             kv_caches_,
             {short_history.input_params});
  impl_->run({long_history.token_ids},
             {long_history.positions},
             kv_caches_,
             {long_history.input_params});

  EXPECT_EQ(model_->forward_cnt(), 1);
}

TEST(ModelRegistryGraphHistoryTest, UsesDefaultCapacityWithoutAdapter) {
  const ModelArgs args{};
  EXPECT_EQ(ModelRegistry::get_graph_history_capacity(
                "unregistered_graph_history_capacity", args, 128),
            128);
}

TEST_F(MluGraphExecutorTest,
       KPoolDecodeUsesFixedHistoryCapacityAndFallsBackAboveIt) {
  ScopedEnvVar graph_kv_cap("XLLM_GRAPH_INDEX_HISTORY_MAX_KV", "64");
  model_args_.index_kpool(4).index_kpool_compress(true);
  model_ = std::make_unique<GroupedCausalLM>(tensor_options_);
  options_.is_draft_engine(false);
  rebuild_impl();

  const auto set_request_metadata = [&](ForwardInput& input) {
    input.input_params.embedding.linear_state_ids = {1, 2, 3, 4, 5};
    input.input_params.embedding.linear_state_indices =
        torch::tensor(input.input_params.embedding.linear_state_ids,
                      tensor_options_.dtype(torch::kInt32));
  };
  const int32_t batch_size = 5;
  auto short_history = prepare_inputs(batch_size, /*seed=*/41);
  set_request_metadata(short_history);
  short_history.input_params.attention.device.block_tables =
      short_history.input_params.attention.device.block_tables.narrow(
          /*dim=*/1, /*start=*/0, /*length=*/2);
  short_history.input_params.meta.kv_max_seq_len = 32;

  auto at_cap = prepare_inputs(batch_size, /*seed=*/43);
  set_request_metadata(at_cap);
  at_cap.input_params.attention.device.block_tables =
      at_cap.input_params.attention.device.block_tables.narrow(
          /*dim=*/1, /*start=*/0, /*length=*/4);
  at_cap.input_params.meta.kv_max_seq_len = 64;

  auto over_cap = prepare_inputs(batch_size, /*seed=*/47);
  set_request_metadata(over_cap);
  over_cap.input_params.attention.device.block_tables =
      over_cap.input_params.attention.device.block_tables.narrow(
          /*dim=*/1, /*start=*/0, /*length=*/5);
  over_cap.input_params.meta.kv_max_seq_len = 65;

  impl_->run({short_history.token_ids},
             {short_history.positions},
             kv_caches_,
             {short_history.input_params});
  EXPECT_EQ(model_->forward_cnt(), 2);
  EXPECT_EQ(model_->last_kv_max_seq_len(), 64);
  EXPECT_EQ(model_->last_block_table_width(), 4);

  impl_->run({at_cap.token_ids},
             {at_cap.positions},
             kv_caches_,
             {at_cap.input_params});
  EXPECT_EQ(model_->forward_cnt(), 2);

  impl_->run({over_cap.token_ids},
             {over_cap.positions},
             kv_caches_,
             {over_cap.input_params});
  impl_->run({over_cap.token_ids},
             {over_cap.positions},
             kv_caches_,
             {over_cap.input_params});
  EXPECT_EQ(model_->forward_cnt(), 4);
}

TEST_F(MluGraphExecutorTest, UnsupportedVerifyUsesEagerAndDecodeCaptures) {
  options_.is_draft_engine(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  auto spec_input = prepare_inputs(batch_size, /*seed=*/13);
  spec_input.input_params.is_spec_verify = true;
  spec_input.input_params.num_accepted_tokens =
      torch::ones({batch_size}, tensor_options_.dtype(torch::kInt32));

  const int32_t start_cnt = model_->forward_cnt();
  auto first_spec_output = impl_
                               ->run({spec_input.token_ids},
                                     {spec_input.positions},
                                     kv_caches_,
                                     {spec_input.input_params})
                               .hidden_states;
  auto second_spec_output = impl_
                                ->run({spec_input.token_ids},
                                      {spec_input.positions},
                                      kv_caches_,
                                      {spec_input.input_params})
                                .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(
      torch::allclose(first_spec_output, second_spec_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 2);

  auto decode_input = prepare_inputs(batch_size, /*seed=*/14);
  auto first_decode_output = impl_
                                 ->run({decode_input.token_ids},
                                       {decode_input.positions},
                                       kv_caches_,
                                       {decode_input.input_params})
                                 .hidden_states;
  auto second_decode_output = impl_
                                  ->run({decode_input.token_ids},
                                        {decode_input.positions},
                                        kv_caches_,
                                        {decode_input.input_params})
                                  .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(
      torch::allclose(first_decode_output, second_decode_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 3);
}

TEST_F(MluGraphExecutorTest, OrdinaryTargetOnlyReplaysDecodeAtSameShape) {
  model_args_.model_type("qwen3_5_text");
  auto model = std::make_unique<GraphContractModel>(tensor_options_);
  GraphContractModel* contract = model.get();
  model_ = std::move(model);
  rebuild_impl();
  for (int32_t step = 0; step < 4; ++step) {
    auto input = prepare_inputs(/*batch_size=*/3, /*seed=*/210 + step);
    auto& params = input.input_params;
    params.embedding.input_embedding = torch::Tensor();
    params.is_spec_verify = step == 1 || step == 2;
    if (params.is_spec_verify) {
      params.num_accepted_tokens =
          torch::ones({3}, tensor_options_.dtype(torch::kInt32));
    }
    ModelOutput expected =
        contract->forward(input.token_ids, input.positions, kv_caches_, params);
    const int32_t before = contract->calls();
    ModelOutput actual =
        impl_->run(input.token_ids, input.positions, kv_caches_, params);
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::equal(actual.hidden_states, expected.hidden_states));
    EXPECT_EQ(contract->calls(), before + (step != 3 ? 1 : 0));
  }
}

TEST_F(MluGraphExecutorTest, MtpTargetOnlyReplaysVerifyPrefillAtSameShape) {
  model_args_.model_type("qwen3_5_text");
  options_.enable_speculative_decode(true)
      .speculative_algorithm("MTP")
      .is_draft_engine(false);
  auto model = std::make_unique<GraphContractModel>(tensor_options_);
  GraphContractModel* contract = model.get();
  model_ = std::move(model);
  rebuild_impl();

  for (int32_t step = 0; step < 6; ++step) {
    auto input = prepare_inputs(/*batch_size=*/3, /*seed=*/220 + step);
    auto& params = input.input_params;
    params.is_spec_verify = step < 3 || step == 5;
    params.meta.batch_forward_type = step == 0 || step == 5
                                         ? BatchForwardType::CHUNKED_PREFILL
                                         : BatchForwardType::DECODE;
    if (params.is_spec_verify) {
      params.num_accepted_tokens =
          torch::ones({3}, tensor_options_.dtype(torch::kInt32));
    }
    ModelOutput expected =
        contract->forward(input.token_ids, input.positions, kv_caches_, params);
    const int32_t before = contract->calls();
    ModelOutput actual =
        impl_->run(input.token_ids, input.positions, kv_caches_, params);
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::equal(actual.hidden_states, expected.hidden_states));
    EXPECT_EQ(contract->calls(), before + (step != 5 ? 1 : 0));
  }
}

TEST_F(MluGraphExecutorTest, MtpDraftOnlyReplaysDecodeAtSameShape) {
  model_args_.model_type("qwen3_5_mtp");
  options_.enable_speculative_decode(true)
      .speculative_algorithm("MTP")
      .is_draft_engine(true);
  auto model = std::make_unique<GraphContractModel>(tensor_options_);
  GraphContractModel* contract = model.get();
  model_ = std::move(model);
  rebuild_impl();

  for (int32_t step = 0; step < 4; ++step) {
    auto input = prepare_inputs(/*batch_size=*/3, /*seed=*/230 + step);
    auto& params = input.input_params;
    params.meta.batch_forward_type = step == 1 || step == 2
                                         ? BatchForwardType::CHUNKED_PREFILL
                                         : BatchForwardType::DECODE;
    ModelOutput expected =
        contract->forward(input.token_ids, input.positions, kv_caches_, params);
    const int32_t before = contract->calls();
    ModelOutput actual =
        impl_->run(input.token_ids, input.positions, kv_caches_, params);
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::equal(actual.hidden_states, expected.hidden_states));
    EXPECT_EQ(contract->calls(), before + (step != 3 ? 1 : 0));
  }
}

TEST_F(MluGraphExecutorTest, TargetGraphCachesBothEmbeddingShapes) {
  auto model = std::make_unique<GraphContractModel>(tensor_options_);
  GraphContractModel* contract = model.get();
  model_ = std::move(model);
  rebuild_impl();
  for (int32_t round = 0; round < 3; ++round) {
    auto input = prepare_inputs(/*batch_size=*/3, 240 + round);
    if (round == 1) {
      input.input_params.embedding.input_embedding = torch::Tensor();
    }
    ModelOutput expected = contract->forward(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    const int32_t before = contract->calls();
    ModelOutput actual = impl_->run(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    EXPECT_TRUE(torch::equal(actual.hidden_states, expected.hidden_states));
    EXPECT_EQ(contract->calls(), before + (round < 2 ? 1 : 0));
  }
}

TEST_F(MluGraphExecutorTest, QwenMtpGraphKeepsTokenAndHiddenInputsSeparate) {
  auto model = std::make_unique<GraphContractModel>(tensor_options_,
                                                    /*dual_input=*/true);
  GraphContractModel* contract = model.get();
  model_ = std::move(model);
  rebuild_impl();
  for (int32_t round = 0; round < 4; ++round) {
    auto input = prepare_inputs(/*batch_size=*/3, 250 + round);
    if (round % 2 == 0) {
      input.input_params.embedding.input_embedding = torch::Tensor();
    }
    ModelOutput expected = contract->forward(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    const int32_t before = contract->calls();
    ModelOutput actual = impl_->run(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    EXPECT_TRUE(torch::equal(actual.hidden_states, expected.hidden_states));
    EXPECT_EQ(contract->calls(), before + (round < 2 ? 1 : 0));
  }
}

TEST_F(MluGraphExecutorTest, DualEmbeddingsAndMropeSurviveBucketRevisit) {
  auto model = std::make_unique<GraphContractModel>(tensor_options_,
                                                    /*dual_input=*/true);
  model->set_token_weights();
  GraphContractModel* contract = model.get();
  model_ = std::move(model);
  rebuild_impl();
  const auto ints = tensor_options_.dtype(torch::kInt32);
  const std::vector<int32_t> rows_by_step = {7, 5, 7, 6};
  std::vector<torch::Tensor> saved_outputs;
  saved_outputs.reserve(rows_by_step.size());

  for (std::size_t step = 0; step < rows_by_step.size(); ++step) {
    const int32_t rows = rows_by_step[step];
    const int32_t value = static_cast<int32_t>(step) + 1;
    auto input = prepare_inputs(rows, /*seed=*/190 + step);
    input.token_ids = torch::arange(1, rows + 1, ints) + value;
    input.positions = torch::stack({torch::arange(rows, ints) + value,
                                    torch::arange(rows, ints) + value + 10,
                                    torch::arange(rows, ints) + value + 20});
    input.input_params.embedding.input_embedding =
        torch::full({rows, model_args_.hidden_size()},
                    static_cast<float>(step + 1) * 0.02f,
                    tensor_options_);

    const torch::Tensor expected = contract
                                       ->forward(input.token_ids,
                                                 input.positions,
                                                 kv_caches_,
                                                 input.input_params)
                                       .hidden_states.clone();
    const int32_t calls_before_run = contract->calls();
    const torch::Tensor actual = impl_
                                     ->run(input.token_ids,
                                           input.positions,
                                           kv_caches_,
                                           input.input_params)
                                     .hidden_states.clone();
    torch_mlu::synchronize();
    EXPECT_EQ(actual.size(0), rows);
    EXPECT_TRUE(torch::equal(actual, expected)) << step;
    EXPECT_EQ(contract->calls(), calls_before_run + (step == 0 ? 1 : 0));
    saved_outputs.emplace_back(actual);
  }
  EXPECT_FALSE(torch::equal(saved_outputs[0], saved_outputs[2]));
}

TEST_F(MluGraphExecutorTest, ExternalEmbeddingCanCaptureAfterTokenInput) {
  auto model = std::make_unique<GraphContractModel>(tensor_options_,
                                                    /*dual_input=*/false);
  GraphContractModel* contract = model.get();
  model_ = std::move(model);
  rebuild_impl();
  for (int32_t round = 0; round < 3; ++round) {
    auto input = prepare_inputs(/*batch_size=*/3, 260 + round);
    if (round != 1) {
      input.input_params.embedding.input_embedding = torch::Tensor();
    }
    ModelOutput expected = contract->forward(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    const int32_t before = contract->calls();
    ModelOutput actual = impl_->run(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    EXPECT_TRUE(torch::equal(actual.hidden_states, expected.hidden_states));
    EXPECT_EQ(contract->calls(), before + (round < 2 ? 1 : 0));
  }
}

// Input formats are fixed within a deployment. Different executors may use
// different formats; they do not need entries in a shared graph cache.
TEST_F(MluGraphExecutorTest, FixedInputFormatsCaptureOncePerDeployment) {
  for (const char* model_type : {"test_model", "qwen3_5_text"}) {
    for (int32_t format = 0; format < 4; ++format) {
      model_args_.model_type(model_type);
      auto model = std::make_unique<GraphContractModel>(tensor_options_);
      GraphContractModel* contract = model.get();
      model_ = std::move(model);
      rebuild_impl();
      for (int32_t round = 0; round < 3; ++round) {
        auto input = prepare_inputs(/*batch_size=*/3, 220 + round);
        if (format == 1) {
          input.token_ids = input.token_ids.to(torch::kInt64);
        } else if (format == 2) {
          input.positions = input.positions.to(torch::kInt64);
        } else if (format == 3) {
          input.positions = input.positions.unsqueeze(0).repeat({3, 1});
        }
        input.positions.add_(round);
        auto expected = contract->forward(
            input.token_ids, input.positions, kv_caches_, input.input_params);
        const int32_t before = contract->calls();
        auto actual = impl_->run(
            input.token_ids, input.positions, kv_caches_, input.input_params);
        EXPECT_TRUE(torch::equal(actual.hidden_states, expected.hidden_states));
        EXPECT_EQ(contract->calls(), before + (round == 0 ? 1 : 0));
      }
    }
  }
}

TEST_F(MluGraphExecutorTest, QwenHistoryWithinContextReusesOneGraph) {
  model_args_.model_type("qwen3_5_text").max_position_embeddings(8192);
  rebuild_impl();
  const std::vector<int32_t> histories = {
      1024, 1025, 2048, 4096, 4097, 8192, 1024};
  for (size_t i = 0; i < histories.size(); ++i) {
    auto input = prepare_inputs(/*batch_size=*/3, 230 + i);
    input.input_params.meta.kv_max_seq_len = histories[i];
    input.input_params.attention.device.block_tables =
        torch::zeros({3, 513}, tensor_options_.dtype(torch::kInt32));
    input.input_params.attention.device.kv_seq_lens =
        torch::arange(4, tensor_options_.dtype(torch::kInt32)) * histories[i];
    auto actual = impl_
                      ->run(input.token_ids,
                            input.positions,
                            kv_caches_,
                            input.input_params)
                      .hidden_states.clone();
    EXPECT_EQ(model_->forward_cnt(), 1 + static_cast<int32_t>(i));
    if (i == 0) {
      EXPECT_EQ(model_->last_kv_max_seq_len(), 8192);
      EXPECT_EQ(model_->last_block_table_width(), 513);
    }
    auto expected = model_->forward(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    EXPECT_TRUE(torch::equal(actual, expected.hidden_states));
  }
}

TEST_F(MluGraphExecutorTest, QwenRequiresPositiveDeclaredContext) {
  model_args_.model_type("qwen3_5_text").max_position_embeddings(0);
  EXPECT_DEATH(rebuild_impl(), "positive max_position_embeddings");
}

TEST_F(MluGraphExecutorTest, QwenHistoryBeyondContextFallsBackThenReplays) {
  model_args_.model_type("qwen3_5_text").max_position_embeddings(8192);
  rebuild_impl();
  auto input = prepare_inputs(/*batch_size=*/3, 239);
  input.input_params.attention.device.block_tables =
      torch::zeros({3, 514}, tensor_options_.dtype(torch::kInt32));
  input.input_params.meta.kv_max_seq_len = 8192;
  auto first =
      impl_
          ->run(
              input.token_ids, input.positions, kv_caches_, input.input_params)
          .hidden_states.clone();
  EXPECT_EQ(model_->forward_cnt(), 1);

  input.input_params.meta.kv_max_seq_len = 8193;
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  EXPECT_EQ(model_->forward_cnt(), 2);

  input.input_params.meta.kv_max_seq_len = 8192;
  auto again =
      impl_
          ->run(
              input.token_ids, input.positions, kv_caches_, input.input_params)
          .hidden_states.clone();
  EXPECT_EQ(model_->forward_cnt(), 2);
  EXPECT_TRUE(torch::equal(first, again));
}

class GraphPeerGroup final : public ProcessGroup {
 public:
  enum class PeerState : int8_t {
    UNSUPPORTED,
    CACHE_MISS,
    INCOMPATIBLE,
    SUPPORTED_PEER,
    MATCHED
  };

  explicit GraphPeerGroup(const torch::Device& device)
      : ProcessGroup(/*rank=*/0, /*world_size=*/2, device) {}

  void peer_state(PeerState state) { state_ = state; }
  int32_t calls() const { return calls_; }

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    ++calls_;
    torch::Tensor local = input.to(torch::kCPU);
    torch::Tensor peer = local.clone();
    if (local.data_ptr<int64_t>()[0] != 0) {
      supported_packet_ = local.clone();
    }
    if (state_ == PeerState::SUPPORTED_PEER) {
      CHECK(supported_packet_.defined());
      peer = supported_packet_.clone();
    }
    int64_t* values = peer.data_ptr<int64_t>();
    if (state_ == PeerState::UNSUPPORTED) {
      values[0] = 0;
    } else if (state_ == PeerState::CACHE_MISS) {
      values[1] = 0;
    } else if (state_ == PeerState::INCOMPATIBLE) {
      values[5] += 1;
    }
    return torch::stack({local, peer});
  }

 private:
  PeerState state_ = PeerState::MATCHED;
  int32_t calls_ = 0;
  torch::Tensor supported_packet_;
};

TEST_F(MluGraphExecutorTest, DpRanksAgreeOnEagerCaptureAndReplay) {
  model_args_.model_type("qwen3_5_text").max_position_embeddings(8192);
  options_.dp_size(2).world_size(2);
  rebuild_impl();
  GraphPeerGroup group(torch::Device("mlu:0"));
  impl_->set_dp_process_group(&group);
  auto input = prepare_inputs(/*batch_size=*/3, 240);
  input.input_params.parallel.dp_global_token_nums = {3, 3};
  input.input_params.parallel.dp_is_decode = {1, 1};
  input.input_params.attention.device.block_tables =
      torch::zeros({3, 513}, tensor_options_.dtype(torch::kInt32));

  group.peer_state(GraphPeerGroup::PeerState::UNSUPPORTED);
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  EXPECT_EQ(model_->forward_cnt(), 1);

  group.peer_state(GraphPeerGroup::PeerState::CACHE_MISS);
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  EXPECT_EQ(model_->forward_cnt(), 2);
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  EXPECT_EQ(model_->forward_cnt(), 3);

  group.peer_state(GraphPeerGroup::PeerState::MATCHED);
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  EXPECT_EQ(model_->forward_cnt(), 3);

  group.peer_state(GraphPeerGroup::PeerState::INCOMPATIBLE);
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  EXPECT_EQ(model_->forward_cnt(), 4);

  group.peer_state(GraphPeerGroup::PeerState::SUPPORTED_PEER);
  input.input_params.meta.kv_max_seq_len = 8193;
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  EXPECT_EQ(model_->forward_cnt(), 5);
  input.input_params.meta.kv_max_seq_len = 1;
  group.peer_state(GraphPeerGroup::PeerState::MATCHED);
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  EXPECT_EQ(model_->forward_cnt(), 5);
  EXPECT_EQ(group.calls(), 7);
}

TEST_F(MluGraphExecutorTest, QwenVerifyPadsWholeRequestsAndReplays) {
  model_args_.model_type("qwen3_5_moe");
  options_.enable_speculative_decode(true)
      .speculative_algorithm("MTP")
      .is_draft_engine(false);
  rebuild_impl();

  auto input = prepare_inputs(/*batch_size=*/5, /*seed=*/113);
  input.input_params.is_spec_verify = true;
  input.input_params.meta.batch_forward_type =
      BatchForwardType::CHUNKED_PREFILL;
  input.input_params.num_accepted_tokens =
      torch::ones({5}, tensor_options_.dtype(torch::kInt32));
  const int32_t start_cnt = model_->forward_cnt();

  torch::Tensor first = impl_
                            ->run({input.token_ids},
                                  {input.positions},
                                  kv_caches_,
                                  {input.input_params})
                            .hidden_states;
  torch::Tensor second = impl_
                             ->run({input.token_ids},
                                   {input.positions},
                                   kv_caches_,
                                   {input.input_params})
                             .hidden_states;
  torch_mlu::synchronize();

  EXPECT_TRUE(torch::allclose(first, second, 1e-5, 1e-6));
  EXPECT_EQ(first.size(0), 5);
  EXPECT_EQ(model_->last_tokens_size(), 8);
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 1);
}

TEST_F(MluGraphExecutorTest, QwenVerifyKeepsThreeTokenRequestsIntact) {
  model_args_.model_type("qwen3_5_moe");
  options_.enable_speculative_decode(true)
      .speculative_algorithm("MTP")
      .is_draft_engine(false);
  rebuild_impl();

  auto input = prepare_inputs(/*batch_size=*/9, /*seed=*/115);
  input.input_params.is_spec_verify = true;
  input.input_params.meta.batch_forward_type =
      BatchForwardType::CHUNKED_PREFILL;
  input.input_params.meta.num_sequences = 3;
  input.input_params.meta.q_max_seq_len = 3;
  input.input_params.attention.host.q_seq_lens = {0, 3, 6, 9};
  input.input_params.attention.host.kv_seq_lens = {0, 3, 6, 9};
  input.input_params.attention.device.q_seq_lens =
      torch::tensor({0, 3, 6, 9}, tensor_options_.dtype(torch::kInt32));
  input.input_params.attention.device.kv_seq_lens =
      input.input_params.attention.device.q_seq_lens.clone();
  input.input_params.attention.device.block_tables =
      input.input_params.attention.device.block_tables.narrow(0, 0, 3);
  input.input_params.num_accepted_tokens =
      torch::tensor({1, 2, 3}, tensor_options_.dtype(torch::kInt32));

  torch::Tensor output = impl_
                             ->run({input.token_ids},
                                   {input.positions},
                                   kv_caches_,
                                   {input.input_params})
                             .hidden_states;
  torch_mlu::synchronize();

  EXPECT_EQ(output.size(0), 9);
  EXPECT_EQ(model_->last_tokens_size(), 12);
  EXPECT_EQ(model_->last_block_table_width(), 129);
}

TEST_F(MluGraphExecutorTest, QwenDraftReusesRequestBucketAcrossRowCounts) {
  model_args_.model_type("qwen3_5_mtp");
  options_.is_draft_engine(true);
  rebuild_impl();

  auto short_input = prepare_inputs(/*batch_size=*/9, /*seed=*/117);
  short_input.input_params.attention.host.kpool_query_lens = {3, 3, 3};
  auto long_input = prepare_inputs(/*batch_size=*/12, /*seed=*/119);
  long_input.input_params.attention.host.kpool_query_lens = {3, 3, 3, 3};
  const int32_t start_cnt = model_->forward_cnt();

  torch::Tensor first = impl_
                            ->run({short_input.token_ids},
                                  {short_input.positions},
                                  kv_caches_,
                                  {short_input.input_params})
                            .hidden_states;
  torch_mlu::synchronize();
  first = first.clone();
  torch::Tensor second = impl_
                             ->run({long_input.token_ids},
                                   {long_input.positions},
                                   kv_caches_,
                                   {long_input.input_params})
                             .hidden_states;
  torch_mlu::synchronize();

  EXPECT_EQ(first.size(0), 9);
  EXPECT_EQ(second.size(0), 12);
  EXPECT_EQ(model_->last_tokens_size(), 12);
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 1);
  EXPECT_FALSE(torch::allclose(first, second.narrow(0, 0, 9)));
}

TEST_F(MluGraphExecutorTest, QwenMropeGraphRefreshesAllAxes) {
  model_args_.model_type("qwen3_5_text");
  auto model = std::make_unique<PositionEchoModel>(tensor_options_);
  PositionEchoModel* echo = model.get();
  model_ = std::move(model);
  options_.is_draft_engine(false);
  rebuild_impl();

  auto input = prepare_inputs(/*batch_size=*/5, /*seed=*/121);
  const auto ints = tensor_options_.dtype(torch::kInt32);
  input.positions = torch::tensor(
      {{1, 2, 3, 4, 5}, {11, 12, 13, 14, 15}, {21, 22, 23, 24, 25}}, ints);
  torch::Tensor first = impl_
                            ->run({input.token_ids},
                                  {input.positions},
                                  kv_caches_,
                                  {input.input_params})
                            .hidden_states.clone();

  torch::Tensor second_positions = torch::tensor(
      {{2, 3, 4, 5, 6}, {12, 13, 14, 15, 16}, {22, 23, 24, 25, 26}}, ints);
  torch::Tensor second = impl_
                             ->run({input.token_ids},
                                   {second_positions},
                                   kv_caches_,
                                   {input.input_params})
                             .hidden_states.clone();
  torch::Tensor again = impl_
                            ->run({input.token_ids},
                                  {input.positions},
                                  kv_caches_,
                                  {input.input_params})
                            .hidden_states;
  torch_mlu::synchronize();

  EXPECT_TRUE(torch::equal(
      first,
      torch::tensor({{3211.0f}, {3322.0f}, {3433.0f}, {3544.0f}, {3655.0f}},
                    tensor_options_.dtype(torch::kFloat32))));
  EXPECT_TRUE(torch::equal(
      second,
      torch::tensor({{3322.0f}, {3433.0f}, {3544.0f}, {3655.0f}, {3766.0f}},
                    tensor_options_.dtype(torch::kFloat32))));
  EXPECT_TRUE(torch::equal(first, again));
  EXPECT_EQ(echo->calls(), 1);
}

TEST(MluMtpContractTest, QwenAdapterKeepsOnlyDraftLayers) {
  ModelArgs args;
  args.model_type("qwen3_5_text");
  args.n_layers(24);
  args.num_nextn_predict_layers(2);
  args.full_attention_interval(4);
  ModelRegistry::configure_mtp_args(
      args, "MTP", /*is_draft_engine=*/true, /*is_python_model=*/false);

  EXPECT_EQ(args.n_layers(), 2);
  EXPECT_EQ(args.layer_types(),
            std::vector<std::string>({"full_attention", "full_attention"}));
  EXPECT_EQ(args.full_attention_interval(), 1);
  EXPECT_TRUE(ModelRegistry::get_mtp_capabilities("qwen3_5_text")
                  .supports_expanded_replay_target);
  EXPECT_TRUE(ModelRegistry::get_mtp_capabilities("qwen3_5_mtp")
                  .supports_accepted_span_replay);
}

TEST_F(MluGraphExecutorTest, LargeDecodeBucketCapturesThenReplays) {
  ScopedConfigSnapshot config_snapshot;
  ExecutionConfig::get_instance().max_tokens_for_graph_mode(128);
  options_.is_draft_engine(false);
  options_.max_seqs_per_batch(128);
  rebuild_impl();

  const int32_t batch_size = 65;
  const uint64_t seed = 19;
  auto forward_input = prepare_inputs(batch_size, seed);
  const int32_t start_cnt = model_->forward_cnt();

  auto first_output = impl_
                          ->run({forward_input.token_ids},
                                {forward_input.positions},
                                kv_caches_,
                                {forward_input.input_params})
                          .hidden_states;
  auto second_output = impl_
                           ->run({forward_input.token_ids},
                                 {forward_input.positions},
                                 kv_caches_,
                                 {forward_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_output, second_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 1);
  EXPECT_EQ(model_->last_tokens_size(), 80);
}

TEST_F(MluGraphExecutorTest, OverConfiguredTokenLimitFallsBackToEager) {
  ScopedConfigSnapshot config_snapshot;
  ExecutionConfig::get_instance().max_tokens_for_graph_mode(33);
  options_.is_draft_engine(false);
  options_.max_seqs_per_batch(33);
  rebuild_impl();

  const int32_t batch_size = 33;
  const uint64_t seed = 79;
  auto forward_input = prepare_inputs(batch_size, seed);
  const int32_t start_cnt = model_->forward_cnt();

  auto first_output = impl_
                          ->run({forward_input.token_ids},
                                {forward_input.positions},
                                kv_caches_,
                                {forward_input.input_params})
                          .hidden_states;
  auto second_output = impl_
                           ->run({forward_input.token_ids},
                                 {forward_input.positions},
                                 kv_caches_,
                                 {forward_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_output, second_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 2);
  EXPECT_EQ(model_->last_tokens_size(), batch_size);
}

TEST_F(MluGraphExecutorTest, PersistentTensorBytesIncludeLazyAux) {
  auto input = prepare_inputs(/*batch_size=*/4, /*seed=*/83);
  mlu::GraphLayout layout;
  layout.num_reqs = 4;
  layout.padded_num_reqs = 4;
  layout.padded_num_tokens = 4;
  mlu::GraphPersistentParam param(input.token_ids,
                                  input.positions,
                                  input.input_params,
                                  layout,
                                  /*block_capacity=*/256,
                                  options_.block_size());
  const std::size_t base_bytes = param.get_persistent_tensor_bytes();
  param.aux_hidden_states_ = torch::zeros({4, 1024}, tensor_options_);
  const std::size_t aux_bytes = param.aux_hidden_states_.numel() *
                                param.aux_hidden_states_.element_size();
  EXPECT_EQ(param.get_persistent_tensor_bytes(), base_bytes + aux_bytes);
}

TEST_F(MluGraphExecutorTest, PrefillThenDecodeCapturesAndReplays) {
  options_.is_draft_engine(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  const uint64_t prefill_seed = 23;
  auto prefill_input = prepare_inputs(batch_size, prefill_seed);
  prefill_input.input_params.meta.batch_forward_type =
      BatchForwardType::PREFILL;

  ModelOutput prefill_output = impl_->run({prefill_input.token_ids},
                                          {prefill_input.positions},
                                          kv_caches_,
                                          {prefill_input.input_params});

  const uint64_t decode_seed = 29;
  auto decode_input = prepare_inputs(batch_size, decode_seed);
  auto first_decode_output = impl_
                                 ->run({decode_input.token_ids},
                                       {decode_input.positions},
                                       kv_caches_,
                                       {decode_input.input_params})
                                 .hidden_states;
  auto second_decode_output = impl_
                                  ->run({decode_input.token_ids},
                                        {decode_input.positions},
                                        kv_caches_,
                                        {decode_input.input_params})
                                  .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(prefill_output.hidden_states.defined());
  EXPECT_TRUE(
      torch::allclose(first_decode_output, second_decode_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), 2);
}

TEST_F(MluGraphExecutorTest, DpDecodePadsEqualAndUnevenCountsToTpGraphSize) {
  options_.is_draft_engine(false);
  options_.world_size(8);
  options_.dp_size(2);
  rebuild_impl();

  for (const std::vector<int32_t>& token_nums :
       {std::vector<int32_t>{2, 2}, std::vector<int32_t>{1, 2}}) {
    auto input = prepare_inputs(/*batch_size=*/2, /*seed=*/61);
    input.input_params.parallel.dp_global_token_nums = token_nums;
    input.input_params.parallel.dp_is_decode = {1, 1};
    const torch::Tensor first = impl_
                                    ->run(input.token_ids,
                                          input.positions,
                                          kv_caches_,
                                          input.input_params)
                                    .hidden_states.clone();
    const torch::Tensor second = impl_
                                     ->run(input.token_ids,
                                           input.positions,
                                           kv_caches_,
                                           input.input_params)
                                     .hidden_states;
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::equal(first, second));
    EXPECT_EQ(model_->forward_cnt(), 1);
    EXPECT_EQ(model_->last_tokens_size(), 4);
    EXPECT_EQ(model_->last_dp_token_nums(), std::vector<int32_t>({4, 4}));
  }
}

TEST_F(MluGraphExecutorTest, MtpSeqLensCapacityUsesSpecFactor) {
  options_.is_draft_engine(false);
  options_.num_speculative_tokens(1);
  options_.num_decoding_tokens(options_.num_speculative_tokens() + 1);
  options_.enable_speculative_decode(true);
  options_.max_seqs_per_batch(2);
  rebuild_impl();

  auto forward_input = prepare_inputs(/*batch_size=*/4, /*seed=*/71);
  forward_input.input_params.parallel.dp_global_token_nums = {4, 4};
  forward_input.input_params.parallel.dp_is_decode = {1, 1};

  auto first_output = impl_
                          ->run({forward_input.token_ids},
                                {forward_input.positions},
                                kv_caches_,
                                {forward_input.input_params})
                          .hidden_states;
  auto second_output = impl_
                           ->run({forward_input.token_ids},
                                 {forward_input.positions},
                                 kv_caches_,
                                 {forward_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_output, second_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), 1);
  EXPECT_EQ(model_->last_tokens_size(), 4);
}

TEST_F(MluGraphExecutorTest, MtpSeqLensCapacityIncludesGraphPadding) {
  options_.is_draft_engine(false);
  options_.num_speculative_tokens(2);
  options_.num_decoding_tokens(options_.num_speculative_tokens() + 1);
  options_.enable_speculative_decode(true);
  options_.max_seqs_per_batch(8);
  rebuild_impl();

  // Eight MTP validation requests produce 24 token rows. The MLU graph
  // executor pads that input to the 32-token bucket, so cumulative sequence
  // lengths need 33 entries.
  auto forward_input = prepare_inputs(/*batch_size=*/24, /*seed=*/73);

  EXPECT_NO_THROW({
    ModelOutput first_output = impl_->run({forward_input.token_ids},
                                          {forward_input.positions},
                                          kv_caches_,
                                          {forward_input.input_params});
    ModelOutput second_output = impl_->run({forward_input.token_ids},
                                           {forward_input.positions},
                                           kv_caches_,
                                           {forward_input.input_params});

    torch_mlu::synchronize();
    EXPECT_TRUE(torch::allclose(
        first_output.hidden_states, second_output.hidden_states, 1e-5, 1e-6));
  });
  EXPECT_EQ(model_->forward_cnt(), 1);
  EXPECT_EQ(model_->last_tokens_size(), 32);
}

TEST_F(MluGraphExecutorTest, DpDummyFallbackKeepsGraphCacheUsable) {
  options_.is_draft_engine(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  const uint64_t seed = 31;
  auto forward_input = prepare_inputs(batch_size, seed);
  forward_input.input_params.parallel.dp_global_token_nums = {batch_size, 0};
  forward_input.input_params.parallel.dp_is_decode = {1, 0};

  const int32_t start_cnt = model_->forward_cnt();
  auto first_output = impl_
                          ->run({forward_input.token_ids},
                                {forward_input.positions},
                                kv_caches_,
                                {forward_input.input_params})
                          .hidden_states;
  auto second_output = impl_
                           ->run({forward_input.token_ids},
                                 {forward_input.positions},
                                 kv_caches_,
                                 {forward_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_output, second_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 2);

  auto decode_input = prepare_inputs(batch_size, /*seed=*/41);
  decode_input.input_params.parallel.dp_global_token_nums = {batch_size,
                                                             batch_size};
  decode_input.input_params.parallel.dp_is_decode = {1, 1};
  const torch::Tensor captured = impl_
                                     ->run(decode_input.token_ids,
                                           decode_input.positions,
                                           kv_caches_,
                                           decode_input.input_params)
                                     .hidden_states.clone();
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 3);
  const torch::Tensor replayed = impl_
                                     ->run(decode_input.token_ids,
                                           decode_input.positions,
                                           kv_caches_,
                                           decode_input.input_params)
                                     .hidden_states;
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(captured, replayed));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 3);
}

TEST_F(MluGraphExecutorTest, DpUnevenFallbackKeepsGraphCacheUsable) {
  options_.is_draft_engine(false);
  rebuild_impl();

  const int32_t batch_size = 5;
  auto forward_input = prepare_inputs(batch_size, 43);
  forward_input.input_params.parallel.dp_global_token_nums = {batch_size,
                                                              batch_size - 1};
  forward_input.input_params.parallel.dp_is_decode = {1, 1};
  forward_input.input_params.meta.q_max_seq_len = 2;

  const int32_t start_cnt = model_->forward_cnt();
  auto first_output = impl_
                          ->run({forward_input.token_ids},
                                {forward_input.positions},
                                kv_caches_,
                                {forward_input.input_params})
                          .hidden_states;
  auto second_output = impl_
                           ->run({forward_input.token_ids},
                                 {forward_input.positions},
                                 kv_caches_,
                                 {forward_input.input_params})
                           .hidden_states;

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(first_output, second_output, 1e-5, 1e-6));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 2);

  auto decode_input = prepare_inputs(batch_size, /*seed=*/53);
  decode_input.input_params.parallel.dp_global_token_nums = {batch_size,
                                                             batch_size};
  decode_input.input_params.parallel.dp_is_decode = {1, 1};
  const torch::Tensor captured = impl_
                                     ->run(decode_input.token_ids,
                                           decode_input.positions,
                                           kv_caches_,
                                           decode_input.input_params)
                                     .hidden_states.clone();
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 3);
  const torch::Tensor replayed = impl_
                                     ->run(decode_input.token_ids,
                                           decode_input.positions,
                                           kv_caches_,
                                           decode_input.input_params)
                                     .hidden_states;
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::equal(captured, replayed));
  EXPECT_EQ(model_->forward_cnt(), start_cnt + 3);
}

TEST_F(MluGraphExecutorTest, PaddedDraftGraphReplaysFreshEmbeddings) {
  options_.is_draft_engine(true);
  model_args_.index_kpool(4).index_kpool_compress(true);
  model_ = std::make_unique<GroupedCausalLM>(tensor_options_);
  rebuild_impl();
  for (int32_t round = 0; round < 3; ++round) {
    auto input = prepare_inputs(/*batch_size=*/4, /*seed=*/31 + round);
    input.input_params.embedding.linear_state_ids = {1, 2};
    input.input_params.embedding.linear_state_indices = torch::tensor(
        {1 + round, 2 + round}, tensor_options_.dtype(torch::kInt32));
    input.input_params.attention.host.kpool_query_lens = {2, 2};
    auto expected = base_impl_->run(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    auto actual = impl_->run(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::equal(expected.hidden_states, actual.hidden_states));
    EXPECT_EQ(model_->forward_cnt(), round + 3);
  }
}

TEST_F(MluGraphExecutorTest, PaddedDecodeGraphReusesShapeAcrossRequestChanges) {
  options_.is_draft_engine(false);
  model_args_.index_kpool(4).index_kpool_compress(true);
  model_ = std::make_unique<RequestSlotsModel>(tensor_options_);
  rebuild_impl();
  const auto ints = tensor_options_.dtype(torch::kInt32);
  for (int32_t round = 0; round < 3; ++round) {
    auto input = prepare_inputs(/*batch_size=*/2, /*seed=*/51 + round);
    input.input_params.embedding.linear_state_ids = {1, 2};
    input.input_params.embedding.linear_state_indices =
        torch::tensor({1 + round, 2 + round}, ints);
    const torch::Tensor expected =
        input.input_params.embedding.linear_state_indices.reshape({2, 1});
    auto actual = impl_->run(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::equal(expected, actual.hidden_states));
    EXPECT_EQ(model_->forward_cnt(), 2);
  }
}

TEST_F(MluGraphExecutorTest, PaddedGraphKeepsRequestGroupsAndRefreshesBlocks) {
  model_args_.index_kpool(4).index_kpool_compress(true);
  model_ = std::make_unique<GroupedBlockTableModel>(tensor_options_);
  options_.is_draft_engine(true);
  rebuild_impl();
  auto input = prepare_inputs(/*batch_size=*/4, /*seed=*/77);
  input.input_params.embedding.linear_state_ids = {1, 2, 3, 4};
  const auto ints = tensor_options_.dtype(torch::kInt32);
  // Keep tensor shapes, token count and state slots fixed. Only the model's
  // request grouping changes; returning to an earlier group must reuse its
  // graph and refresh the block table materialized by the metadata hook.
  // The final two rounds also remap pages consecutively at the same shape.
  for (int32_t round = 0; round < 4; ++round) {
    const bool grouped = round != 1;
    input.input_params.embedding.linear_state_ids =
        grouped ? std::vector<int32_t>{1, 2} : std::vector<int32_t>{1, 2, 3, 4};
    input.input_params.attention.host.kpool_query_lens =
        grouped ? std::vector<int32_t>{2, 2} : std::vector<int32_t>{1, 1, 1, 1};
    input.input_params.attention.device.block_tables =
        (torch::arange(4, ints) + round * 10).reshape({4, 1});
    auto expected =
        (grouped ? torch::tensor({0, 0, 2, 2}, ints) : torch::arange(4, ints)) +
        round * 10;
    auto actual = impl_->run(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::equal(expected.reshape({4, 1}), actual.hidden_states));
  }
  EXPECT_EQ(model_->forward_cnt(), 4);
}

TEST_F(MluGraphExecutorTest, PaddedGraphReturnsFreshAuxiliaryHiddenStates) {
  model_args_.index_kpool(4).index_kpool_compress(true);
  model_ = std::make_unique<GroupedCausalLM>(tensor_options_);
  model_->return_aux_hidden_states(true);
  options_.enable_graph_aux_hidden_states(true);
  rebuild_impl();
  const void* hidden_buffer = nullptr;
  const void* aux_buffer = nullptr;
  for (int32_t round = 0; round < 3; ++round) {
    auto input = prepare_inputs(/*batch_size=*/3, /*seed=*/83 + round);
    input.input_params.embedding.linear_state_ids = {1, 2, 3};
    auto expected = base_impl_->run(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    auto actual = impl_->run(
        input.token_ids, input.positions, kv_caches_, input.input_params);
    torch_mlu::synchronize();
    ASSERT_TRUE(actual.aux_hidden_states.defined());
    EXPECT_EQ(actual.hidden_states.size(0), 3);
    EXPECT_EQ(actual.aux_hidden_states.size(0), 3);
    if (round == 0) {
      hidden_buffer = actual.hidden_states.data_ptr();
      aux_buffer = actual.aux_hidden_states.data_ptr();
    } else {
      EXPECT_EQ(actual.hidden_states.data_ptr(), hidden_buffer);
      EXPECT_EQ(actual.aux_hidden_states.data_ptr(), aux_buffer);
    }
    EXPECT_TRUE(torch::equal(expected.hidden_states, actual.hidden_states));
    EXPECT_TRUE(
        torch::equal(expected.aux_hidden_states, actual.aux_hidden_states));
    EXPECT_EQ(model_->forward_cnt(), round + 3);
  }
}

TEST_F(MluGraphExecutorTest, PaddedGraphOmitsDisabledAuxiliaryHiddenStates) {
  model_args_.index_kpool(4).index_kpool_compress(true);
  model_ = std::make_unique<GroupedCausalLM>(tensor_options_);
  model_->return_aux_hidden_states(true);
  options_.enable_graph_aux_hidden_states(false);
  rebuild_impl();
  auto input = prepare_inputs(/*batch_size=*/3, /*seed=*/87);
  input.input_params.embedding.linear_state_ids = {1, 2, 3};
  const ModelOutput expected = base_impl_->run(
      input.token_ids, input.positions, kv_caches_, input.input_params);

  const ModelOutput first = impl_->run(
      input.token_ids, input.positions, kv_caches_, input.input_params);
  const torch::Tensor first_hidden = first.hidden_states.clone();
  const void* first_buffer = first.hidden_states.data_ptr();
  const int32_t calls_after_capture = model_->forward_cnt();
  const ModelOutput replay = impl_->run(
      input.token_ids, input.positions, kv_caches_, input.input_params);
  torch_mlu::synchronize();

  EXPECT_EQ(first.hidden_states.size(0), 3);
  EXPECT_EQ(replay.hidden_states.size(0), 3);
  EXPECT_TRUE(torch::equal(first_hidden, expected.hidden_states));
  EXPECT_TRUE(torch::equal(replay.hidden_states, expected.hidden_states));
  EXPECT_EQ(first_buffer, replay.hidden_states.data_ptr());
  EXPECT_FALSE(first.aux_hidden_states.defined());
  EXPECT_FALSE(replay.aux_hidden_states.defined());
  EXPECT_EQ(model_->forward_cnt(), calls_after_capture);
}

TEST_F(MluGraphExecutorTest, PaddedDraftKeepsWholeRequestsAcrossStages) {
  model_args_.index_kpool(4).index_kpool_compress(true);
  model_ = std::make_unique<RequestSlotsModel>(tensor_options_);
  options_.is_draft_engine(true);
  rebuild_impl();
  const auto ints = tensor_options_.dtype(torch::kInt32);
  for (int32_t width : {2, 3, 4, 5}) {
    for (int32_t requests : {5, 7, 5}) {
      auto input = prepare_inputs(requests * width, /*seed=*/93);
      auto& params = input.input_params;
      params.attention.host.kpool_query_lens.assign(requests, width);
      params.embedding.linear_state_ids.resize(requests);
      std::iota(params.embedding.linear_state_ids.begin(),
                params.embedding.linear_state_ids.end(),
                requests);
      params.embedding.linear_state_indices =
          torch::tensor(params.embedding.linear_state_ids, ints);
      const auto result =
          impl_->run(input.token_ids, input.positions, kv_caches_, params);
      auto expected = torch::tensor(params.embedding.linear_state_ids, ints)
                          .unsqueeze(1)
                          .expand({requests, width})
                          .reshape({-1, 1});
      EXPECT_TRUE(torch::equal(result.hidden_states, expected));
      EXPECT_EQ(model_->last_tokens_size(), 8 * width);
    }
  }
}

TEST_F(MluGraphExecutorTest, PaddedRequestLimitHonorsNoPaddingOption) {
  ScopedConfigSnapshot config_snapshot;
  ExecutionConfig::get_instance().max_tokens_for_graph_mode(16);
  model_args_.index_kpool(4).index_kpool_compress(true);
  options_.is_draft_engine(true).max_seqs_per_batch(5);
  for (bool no_padding : {false, true}) {
    options_.enable_graph_mode_decode_no_padding(no_padding);
    model_ = std::make_unique<GroupedCausalLM>(tensor_options_);
    rebuild_impl();
    auto input = prepare_inputs(/*batch_size=*/15, /*seed=*/95);
    input.input_params.attention.host.kpool_query_lens.assign(5, 3);
    input.input_params.embedding.linear_state_ids = {1, 2, 3, 4, 5};
    for (int32_t round = 0; round < 3; ++round) {
      auto actual = impl_->run(
          input.token_ids, input.positions, kv_caches_, input.input_params);
      EXPECT_EQ(actual.hidden_states.size(0), 15);
    }
    EXPECT_EQ(model_->forward_cnt(), no_padding ? 2 : 3);
    EXPECT_EQ(model_->last_tokens_size(), 15);
  }
}

TEST_F(MluGraphExecutorTest, RequestPaddingAlignsTokensToTpSize) {
  model_args_.index_kpool(4).index_kpool_compress(true);
  model_ = std::make_unique<RequestSlotsModel>(tensor_options_);
  options_.is_draft_engine(true).dp_size(2).world_size(12);
  rebuild_impl();
  auto input = prepare_inputs(/*batch_size=*/12, /*seed=*/94);
  auto& params = input.input_params;
  params.parallel.dp_global_token_nums = {12, 12};
  params.parallel.dp_is_decode = {1, 1};
  params.attention.host.kpool_query_lens = {4, 4, 4};
  params.embedding.linear_state_ids = {1, 2, 3};
  const auto ints = tensor_options_.dtype(torch::kInt32);
  auto expected = torch::tensor({1, 2, 3}, ints)
                      .unsqueeze(1)
                      .expand({3, 4})
                      .reshape({12, 1});
  for (int32_t round = 0; round < 2; ++round) {
    auto actual =
        impl_->run(input.token_ids, input.positions, kv_caches_, params);
    EXPECT_TRUE(torch::equal(actual.hidden_states, expected));
    EXPECT_EQ(model_->last_tokens_size(), 24);
  }
  EXPECT_EQ(model_->forward_cnt(), 2);
}

TEST_F(MluGraphExecutorTest, PaddedDraftSeparatesEqualTokenCounts) {
  model_args_.index_kpool(4).index_kpool_compress(true);
  model_ = std::make_unique<RequestSlotsModel>(tensor_options_);
  options_.is_draft_engine(true);
  rebuild_impl();
  const auto ints = tensor_options_.dtype(torch::kInt32);
  const std::vector<int32_t> widths = {4, 1, 4, 1};
  for (std::size_t step = 0; step < widths.size(); ++step) {
    const int32_t width = widths[step];
    auto input = prepare_inputs(/*batch_size=*/16, /*seed=*/94);
    const int32_t requests = 16 / width;
    auto& params = input.input_params;
    params.attention.host.kpool_query_lens.assign(requests, width);
    params.embedding.linear_state_ids.resize(requests);
    std::iota(params.embedding.linear_state_ids.begin(),
              params.embedding.linear_state_ids.end(),
              1);
    auto result =
        impl_->run(input.token_ids, input.positions, kv_caches_, params);
    auto expected = torch::tensor(params.embedding.linear_state_ids, ints)
                        .unsqueeze(1)
                        .expand({requests, width})
                        .reshape({-1, 1});
    EXPECT_TRUE(torch::equal(result.hidden_states, expected));
    EXPECT_EQ(model_->forward_cnt(), step < 2 ? (step + 1) * 2 : 4);
  }
}

TEST_F(MluGraphExecutorTest,
       PaddedGraphRejectsUnsupportedBatchesBeforeCapture) {
  model_args_.index_kpool(4).index_kpool_compress(true);
  model_ = std::make_unique<GroupedCausalLM>(tensor_options_);
  options_.dp_size(2).world_size(2);
  rebuild_impl();
  auto input = prepare_inputs(/*batch_size=*/2, /*seed=*/91);
  input.input_params.embedding.linear_state_ids = {1, 2};
  input.input_params.parallel.dp_is_decode = {1, 1};
  input.input_params.parallel.dp_global_token_nums = {2, 1};
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  EXPECT_EQ(model_->forward_cnt(), 1);
  input.input_params.parallel.dp_global_token_nums = {2, 2};
  input.input_params.parallel.dp_is_decode = {1, 0};
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  EXPECT_EQ(model_->forward_cnt(), 2);
  input.input_params.parallel.dp_is_decode = {1, 1};
  input.input_params.is_spec_verify = true;
  // The model requires acceptance metadata for verification.
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  EXPECT_EQ(model_->forward_cnt(), 3);
  input.input_params.is_spec_verify = false;
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  EXPECT_EQ(model_->forward_cnt(), 5);
  impl_->run(input.token_ids, input.positions, kv_caches_, input.input_params);
  torch_mlu::synchronize();
  EXPECT_EQ(model_->forward_cnt(), 5);
}

}  // namespace xllm
