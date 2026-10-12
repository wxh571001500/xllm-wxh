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

#include <framework/core/MLUStream.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model_context.h"
#include "framework/parallel_state/process_group.h"
#include "framework/state_dict/state_dict.h"
#include "models/vlm/qwen3_5.h"
#include "platform/device.h"

namespace xllm {
namespace {

constexpr int64_t kHiddenSize = 128;
constexpr double kNormEps = 1e-6;

bool has_safe_visible_device() {
  const char* visible = std::getenv("MLU_VISIBLE_DEVICES");
  if (visible == nullptr) {
    return false;
  }
  const std::string devices(visible);
  if (devices.find(',') != std::string::npos) {
    return false;
  }
  int32_t physical = 0;
  const auto [end, error] = std::from_chars(
      devices.data(), devices.data() + devices.size(), physical);
  return error == std::errc{} && end == devices.data() + devices.size() &&
         physical > 0;
}

struct ModelFixture {
  explicit ModelFixture(int32_t dp_size)
      : device("mlu:0"),
        options(torch::TensorOptions().dtype(torch::kBFloat16).device(device)),
        tp_group(std::make_unique<ProcessGroup>(
            /*rank=*/0,
            /*world_size=*/1,
            device)) {
    ParallelArgs parallel_args(/*rank=*/0,
                               /*world_size=*/dp_size,
                               dp_size,
                               tp_group.get());
    parallel_args.tp_group_ = tp_group.get();

    ModelArgs args;
    args.model_type("qwen3_5")
        .n_layers(0)
        .hidden_size(kHiddenSize)
        .vocab_size(16)
        .head_dim(64)
        .partial_rotary_factor(0.25f)
        .rope_scaling_mrope_section({3, 3, 2})
        .max_position_embeddings(32)
        .rope_theta(10000.0f)
        .rms_norm_eps(kNormEps);
    ModelContext context(parallel_args, args, QuantArgs(), options);
    model = std::make_unique<Qwen3_5ModelImpl>(context);
    auto parameters = model->named_parameters();
    parameters["norm.weight"].fill_(0);
    parameters["embed_tokens.weight"].fill_(0.25);
  }

  torch::Device device;
  torch::TensorOptions options;
  std::unique_ptr<ProcessGroup> tp_group;
  std::unique_ptr<Qwen3_5ModelImpl> model;
};

ModelInputParams make_params(const torch::Device& device,
                             const std::vector<int32_t>& dp_counts,
                             bool dummy) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = dummy ? 0 : 2;
  params.meta.q_max_seq_len = dummy ? 0 : 1;
  params.meta.kv_max_seq_len = dummy ? 0 : 1;
  params.parallel.dp_global_token_nums = dp_counts;
  params.parallel.dp_is_decode.assign(dp_counts.size(), 1);
  if (!dummy) {
    const auto ints =
        torch::TensorOptions().dtype(torch::kInt32).device(device);
    params.attention.host.q_seq_lens = {0, 1, 2};
    params.attention.host.kv_seq_lens = {0, 1, 2};
    params.attention.device.q_seq_lens = torch::tensor({0, 1, 2}, ints);
    params.attention.device.kv_seq_lens = torch::tensor({0, 1, 2}, ints);
    params.attention.device.new_cache_slots = torch::tensor({0, 1}, ints);
    params.attention.device.block_tables = torch::zeros({2, 1}, ints);
    const torch::Tensor values = torch::arange(2 * kHiddenSize, torch::kFloat32)
                                     .reshape({2, kHiddenSize});
    params.embedding.input_embedding =
        ((values.remainder(17) - 8.0f) / 8.0f).to(device, torch::kBFloat16);
  }
  return params;
}

ModelOutput run_forward(ModelFixture& fixture,
                        const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        const ModelInputParams& params) {
  std::vector<KVCache> caches;
  return fixture.model->forward(tokens, positions, caches, params);
}

void expect_norm_output(const torch::Tensor& output,
                        const torch::Tensor& input) {
  const torch::Tensor values = input.to(torch::kCPU).to(torch::kFloat32);
  const torch::Tensor variance =
      values.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
  const torch::Tensor expected = values * torch::rsqrt(variance + kNormEps);
  EXPECT_EQ(output.sizes(), input.sizes());
  EXPECT_TRUE(torch::allclose(output.to(torch::kCPU).to(torch::kFloat32),
                              expected,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2));
}

void expect_selected_params(const ModelInputParams& params,
                            int32_t dp_size,
                            const std::vector<int32_t>& expected,
                            bool expect_copy) {
  std::optional<ModelInputParams> patched_params;
  const ModelInputParams& selected =
      detail::select_qwen_dp_params(params, dp_size, patched_params);
  EXPECT_EQ(selected.parallel.dp_global_token_nums, expected);
  if (expect_copy) {
    ASSERT_TRUE(patched_params.has_value());
    EXPECT_EQ(&selected, &*patched_params);
    EXPECT_NE(&selected, &params);
  } else {
    EXPECT_FALSE(patched_params.has_value());
    EXPECT_EQ(&selected, &params);
  }
}

struct SyncGatherObserved {};
struct AsyncGatherObserved {};

class GatherProbeGroup final : public ProcessGroup {
 public:
  explicit GatherProbeGroup(const torch::Device& device)
      : ProcessGroup(/*rank=*/0, /*world_size=*/2, device) {}

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    ++sync_calls_;
    input_rows_ = input.size(0);
    torch_mlu::getCurrentMLUStream(input.device().index()).synchronize();
    throw SyncGatherObserved{};
  }

  c10::intrusive_ptr<c10d::Work> allgather_base_async(
      const torch::Tensor& /*input*/,
      torch::Tensor& /*output*/) override {
    ++async_calls_;
    throw AsyncGatherObserved{};
  }

  int32_t sync_calls() const { return sync_calls_; }
  int32_t async_calls() const { return async_calls_; }
  int64_t input_rows() const { return input_rows_; }

 private:
  int32_t sync_calls_ = 0;
  int32_t async_calls_ = 0;
  int64_t input_rows_ = 0;
};

TEST(Qwen35MluForwardTest, MoeForwardUsesPatchedDpCounts) {
  if (!has_safe_visible_device()) {
    GTEST_SKIP() << "Set MLU_VISIBLE_DEVICES to a healthy physical card ID "
                    "excluding 0";
  }
  const torch::Device device("mlu:0");
  Device xllm_device(device);
  xllm_device.set_device();
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto ints = options.dtype(torch::kInt32);

  ProcessGroup tp_group(/*rank=*/0, /*world_size=*/1, device);
  GatherProbeGroup dp_group(device);
  ParallelArgs parallel_args(/*rank=*/0,
                             /*world_size=*/2,
                             /*dp_size=*/2,
                             /*cp_size=*/1,
                             /*process_group=*/nullptr,
                             /*ep_size=*/2);
  parallel_args.tp_group_ = &tp_group;
  parallel_args.moe_tp_group_ = &tp_group;
  parallel_args.dp_local_process_group_ = &dp_group;
  parallel_args.moe_ep_group_ = &dp_group;

  ModelArgs args;
  args.model_type("qwen3_5")
      .n_layers(1)
      .layer_types({"full_attention"})
      .hidden_size(kHiddenSize)
      .vocab_size(16)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(64)
      .partial_rotary_factor(0.25f)
      .rope_scaling_mrope_section({3, 3, 2})
      .max_position_embeddings(32)
      .rope_theta(10000.0f)
      .rms_norm_eps(kNormEps)
      .n_routed_experts(2)
      .num_experts_per_tok(1)
      .moe_intermediate_size(64)
      .n_shared_experts(0)
      .decoder_sparse_step(1)
      .n_group(1)
      .topk_group(1)
      .routed_scaling_factor(1.0f)
      .scoring_func("softmax")
      .hidden_act("silu");
  ModelContext context(parallel_args, args, QuantArgs(), options);
  Qwen3_5ModelImpl model(context);

  // The probe stops in MoE gather, before the gate or expert weights are read.
  // Load finite weights for every preceding embedding, norm and attention op.
  std::unordered_map<std::string, torch::Tensor> weights;
  const auto add_weight =
      [&](const std::string& name, torch::IntArrayRef shape, float value) {
        weights.emplace(name, torch::full(shape, value, options));
      };
  add_weight("embed_tokens.weight", {16, kHiddenSize}, 0.25f);
  add_weight("norm.weight", {kHiddenSize}, 1.0f);
  add_weight("layers.0.input_layernorm.weight", {kHiddenSize}, 1.0f);
  add_weight("layers.0.post_attention_layernorm.weight", {kHiddenSize}, 1.0f);
  add_weight(
      "layers.0.self_attn.q_proj.weight", {kHiddenSize, kHiddenSize}, 0.01f);
  add_weight("layers.0.self_attn.k_proj.weight", {64, kHiddenSize}, 0.01f);
  add_weight("layers.0.self_attn.v_proj.weight", {64, kHiddenSize}, 0.01f);
  add_weight(
      "layers.0.self_attn.o_proj.weight", {kHiddenSize, kHiddenSize}, 0.01f);
  add_weight("layers.0.self_attn.q_norm.weight", {64}, 1.0f);
  add_weight("layers.0.self_attn.k_norm.weight", {64}, 1.0f);
  model.load_state_dict(StateDict(std::move(weights)));

  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 1;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 1;
  params.parallel.dp_global_token_nums = {1, 0};
  params.parallel.dp_is_decode = {1, 1};
  params.attention.host.q_seq_lens = {0, 1};
  params.attention.host.kv_seq_lens = {0, 1};
  params.attention.device.q_seq_lens = torch::tensor({0, 1}, ints);
  params.attention.device.kv_seq_lens = torch::tensor({0, 1}, ints);
  params.attention.device.new_cache_slots = torch::tensor({0}, ints);
  params.attention.device.block_tables = torch::zeros({1, 1}, ints);
  params.embedding.input_embedding =
      torch::full({1, kHiddenSize}, 0.25f, options);
  const void* embedding_storage = params.embedding.input_embedding.data_ptr();
  const torch::Tensor tokens = torch::tensor({1}, ints);
  const torch::Tensor positions = torch::tensor({0}, ints);
  std::vector<KVCache> caches;
  caches.emplace_back(KVCacheTensors{torch::zeros({1, 1, 16, 64}, options),
                                     torch::zeros({1, 1, 16, 64}, options)});

  // The real decoder reaches gather_dp_tokens. Patching {1, 0} to {1, 1}
  // selects its equal-count sync gather; passing the original vector selects
  // the uneven async path instead. The probe stops before expert computation.
  EXPECT_THROW(model.forward(tokens, positions, caches, params),
               SyncGatherObserved);
  EXPECT_EQ(dp_group.sync_calls(), 1);
  EXPECT_EQ(dp_group.async_calls(), 0);
  EXPECT_EQ(dp_group.input_rows(), 1);
  EXPECT_EQ(params.parallel.dp_global_token_nums, std::vector<int32_t>({1, 0}));
  EXPECT_EQ(params.embedding.input_embedding.data_ptr(), embedding_storage);
}

TEST(Qwen35MluForwardTest, SelectsOriginalOrPatchedDpParams) {
  ModelInputParams params;
  params.parallel.dp_global_token_nums = {2, 0};
  expect_selected_params(params,
                         /*dp_size=*/1,
                         /*expected=*/{2, 0},
                         /*expect_copy=*/false);
  EXPECT_EQ(params.parallel.dp_global_token_nums, std::vector<int32_t>({2, 0}));

  params.parallel.dp_global_token_nums = {2, 2};
  expect_selected_params(params,
                         /*dp_size=*/2,
                         /*expected=*/{2, 2},
                         /*expect_copy=*/false);
  params.parallel.dp_global_token_nums = {2, 0};
  expect_selected_params(params,
                         /*dp_size=*/2,
                         /*expected=*/{2, 1},
                         /*expect_copy=*/true);
  EXPECT_EQ(params.parallel.dp_global_token_nums, std::vector<int32_t>({2, 0}));

  params.parallel.dp_global_token_nums = {0, 2};
  expect_selected_params(params,
                         /*dp_size=*/2,
                         /*expected=*/{1, 2},
                         /*expect_copy=*/true);
  EXPECT_EQ(params.parallel.dp_global_token_nums, std::vector<int32_t>({0, 2}));
}

TEST(Qwen35MluForwardTest, DpOnePreservesCallerAndOutput) {
  if (!has_safe_visible_device()) {
    GTEST_SKIP() << "Set MLU_VISIBLE_DEVICES to healthy physical card IDs "
                    "excluding 0";
  }
  ModelFixture fixture(/*dp_size=*/1);
  const auto ints =
      torch::TensorOptions().dtype(torch::kInt32).device(fixture.device);
  const torch::Tensor tokens = torch::tensor({1, 2}, ints);
  const torch::Tensor positions = torch::tensor({0, 1}, ints);
  ModelInputParams params = make_params(fixture.device, {2}, /*dummy=*/false);
  const torch::Tensor input_before = params.embedding.input_embedding.clone();
  expect_selected_params(params,
                         /*dp_size=*/1,
                         /*expected=*/{2},
                         /*expect_copy=*/false);

  const ModelOutput output = run_forward(fixture, tokens, positions, params);

  ASSERT_TRUE(output.hidden_states.defined());
  expect_norm_output(output.hidden_states, input_before);
  EXPECT_EQ(params.parallel.dp_global_token_nums, std::vector<int32_t>({2}));
  EXPECT_TRUE(torch::equal(params.embedding.input_embedding, input_before));
}

TEST(Qwen35MluForwardTest, DpTwoZeroTransitionsPreserveCaller) {
  if (!has_safe_visible_device()) {
    GTEST_SKIP() << "Set MLU_VISIBLE_DEVICES to healthy physical card IDs "
                    "excluding 0";
  }
  ModelFixture fixture(/*dp_size=*/2);
  const auto ints =
      torch::TensorOptions().dtype(torch::kInt32).device(fixture.device);
  const torch::Tensor tokens = torch::tensor({1, 2}, ints);
  const torch::Tensor positions = torch::tensor({0, 1}, ints);
  ModelInputParams params =
      make_params(fixture.device, {2, 2}, /*dummy=*/false);
  const torch::Tensor input_before = params.embedding.input_embedding.clone();

  for (const std::vector<int32_t>& counts :
       {std::vector<int32_t>{2, 2}, {2, 0}, {2, 2}, {2, 0}}) {
    params.parallel.dp_global_token_nums = counts;
    const std::vector<int32_t> expected =
        counts[1] == 0 ? std::vector<int32_t>{2, 1} : counts;
    expect_selected_params(params,
                           /*dp_size=*/2,
                           expected,
                           /*expect_copy=*/counts[1] == 0);
    const ModelOutput output = run_forward(fixture, tokens, positions, params);
    ASSERT_TRUE(output.hidden_states.defined());
    expect_norm_output(output.hidden_states, input_before);
    EXPECT_EQ(params.parallel.dp_global_token_nums, counts);
    EXPECT_TRUE(torch::equal(params.embedding.input_embedding, input_before));
  }
}

TEST(Qwen35MluForwardTest, EmptyLocalTokensUseDummyWithAndWithoutZero) {
  if (!has_safe_visible_device()) {
    GTEST_SKIP() << "Set MLU_VISIBLE_DEVICES to healthy physical card IDs "
                    "excluding 0";
  }
  ModelFixture fixture(/*dp_size=*/2);
  const auto ints =
      torch::TensorOptions().dtype(torch::kInt32).device(fixture.device);
  const torch::Tensor empty = torch::empty({0}, ints);
  const torch::Tensor dummy = torch::tensor({1}, ints);
  for (const std::vector<int32_t>& counts :
       {std::vector<int32_t>{0, 2}, {1, 2}, {1, 0}}) {
    ModelInputParams params =
        make_params(fixture.device, counts, /*dummy=*/true);
    std::vector<int32_t> expected = counts;
    for (int32_t& count : expected) {
      if (count == 0) {
        count = 1;
      }
    }
    expect_selected_params(params,
                           /*dp_size=*/2,
                           expected,
                           /*expect_copy=*/expected != counts);
    const ModelOutput from_empty = run_forward(fixture, empty, empty, params);
    const ModelOutput from_dummy = run_forward(fixture, dummy, dummy, params);

    ASSERT_TRUE(from_empty.hidden_states.defined());
    ASSERT_TRUE(from_dummy.hidden_states.defined());
    EXPECT_EQ(from_empty.hidden_states.sizes(),
              torch::IntArrayRef({1, kHiddenSize}));
    EXPECT_TRUE(
        torch::equal(from_empty.hidden_states, from_dummy.hidden_states));
    EXPECT_EQ(empty.numel(), 0);
    EXPECT_EQ(params.parallel.dp_global_token_nums, counts);
    EXPECT_FALSE(params.embedding.input_embedding.defined());
  }
}

}  // namespace
}  // namespace xllm
