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

#include "layers/mlu/glm5_next/glm5_next_decoder_layer.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/model/aux_hidden_capture.h"
#include "framework/model/model_args.h"
#include "kernels/mlu/chunk_kda.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"

namespace xllm::layer {
namespace {

enum class LayerTestMode {
  PREFILL,
  CHUNKED_PREFILL,
  DECODE_CAPTURE,
  VERIFY_CAPTURE,
};

void verify_capture_boundary(Glm5NextDecoderLayer& layer,
                             const ModelArgs& args,
                             const torch::TensorOptions& options,
                             bool spec_verify) {
  const int32_t width = spec_verify ? 3 : 1;
  const int64_t heads = args.linear_num_key_heads();
  const int64_t head_dim = args.linear_key_head_dim();
  const auto ints = options.dtype(torch::kInt32);
  const auto make_cache = [&]() {
    return KVCache(LinearAttentionKVCacheTensors{
        torch::zeros({2,
                      width + args.linear_conv_kernel_dim() - 2,
                      3 * heads * head_dim},
                     options),
        torch::zeros({2 * width, heads, head_dim, head_dim},
                     options.dtype(torch::kFloat32))});
  };
  ModelInputParams input;
  input.is_spec_verify = spec_verify;
  input.embedding.linear_state_ids = {1};
  input.embedding.linear_state_indices = torch::tensor({1}, ints);
  input.num_accepted_tokens = torch::tensor({1}, ints);
  AttentionMetadata metadata{};
  metadata.is_chunked_prefill = spec_verify;
  metadata.is_spec_verify = spec_verify;
  metadata.max_query_len = width;
  metadata.q_cu_seq_lens = torch::tensor({0, width}, ints);
  torch::Tensor positions = torch::arange(width, options.dtype(torch::kInt64));
  torch::Tensor actual =
      torch::randn({width, args.hc_mult(), args.hidden_size()}, options);
  torch::Tensor expected = actual.clone();
  std::optional<torch::Tensor> actual_residual;
  std::optional<torch::Tensor> expected_residual;
  std::optional<PendingMHC> pending;
  ModelArgs capture_args = args;
  capture_args.layers_to_capture({1});
  AuxHiddenCapture capture(capture_args, options, width);

  // Capture in the middle of a pending chain, then resume deferral. The
  // reference completes every layer using the unfused mHC operations.
  for (int32_t layer_id = 0; layer_id < 4; ++layer_id) {
    SCOPED_TRACE(layer_id);
    KVCache actual_cache = make_cache();
    KVCache expected_cache = make_cache();
    const bool capture_hidden = capture.should_capture(layer_id);
    actual = layer->forward(actual,
                            actual_residual,
                            positions,
                            metadata,
                            actual_cache,
                            input,
                            &pending,
                            /*is_last_layer=*/layer_id == 3,
                            /*materialize_output=*/capture_hidden);
    expected = layer->forward(expected,
                              expected_residual,
                              positions,
                              metadata,
                              expected_cache,
                              input);
    if (layer_id == 0 || layer_id == 2) {
      ASSERT_TRUE(pending.has_value());
      continue;
    }
    ASSERT_FALSE(pending.has_value());
    ASSERT_EQ(actual.sizes(), expected.sizes());
    EXPECT_TRUE(torch::allclose(actual, expected, 0.02, 0.02));
    if (capture_hidden) {
      capture.capture_layer(layer_id, actual.mean(-2), std::nullopt);
      const ModelOutput captured = capture.finalize(actual.mean(-2));
      EXPECT_TRUE(torch::allclose(
          captured.aux_hidden_states, expected.mean(-2), 0.02, 0.02));
    }
  }
}

void run_layer_chain(LayerTestMode mode) {
  const bool chunked_prefill = mode == LayerTestMode::CHUNKED_PREFILL;
  torch::InferenceMode guard;
  const torch::Device device(torch::kPrivateUse1, /*index=*/0);
  Device mlu_device(device);
  mlu_device.set_seed(/*seed=*/20260916);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto fp32_options = options.dtype(torch::kFloat32);
  const auto int_options = options.dtype(torch::kInt32);
  std::unique_ptr<ProcessGroup> process_group;
  const ParallelArgs parallel_args =
      test::create_default_parallel_args(process_group);
  constexpr int64_t kHiddenSize = 4096;
  constexpr int64_t kHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kProjectionSize = kHeads * kHeadDim;
  constexpr int64_t kConvWidth = 4;
  constexpr int64_t kIntermediateSize = 256;
  ModelArgs args;
  args.model_type() = "glm5_next";
  args.hidden_size() = kHiddenSize;
  args.intermediate_size() = kIntermediateSize;
  args.hidden_act() = "silu";
  args.rms_norm_eps() = 1e-5f;
  args.hc_mult() = 4;
  args.hc_sinkhorn_iters() = 20;
  args.hc_eps() = 1e-6f;
  args.layer_types() = {"linear_attention", "linear_attention"};
  args.mlp_layer_types() = {"dense", "dense"};
  args.linear_num_key_heads() = kHeads;
  args.linear_num_value_heads() = kHeads;
  args.linear_key_head_dim() = kHeadDim;
  args.linear_value_head_dim() = kHeadDim;
  args.linear_conv_kernel_dim() = kConvWidth;
  args.linear_lower_bound() = -5.0f;
  const QuantArgs quant_args;
  const ModelContext context(parallel_args, args, quant_args, options);
  Glm5NextDecoderLayer first_layer(context, /*layer_id=*/0);
  Glm5NextDecoderLayer second_layer(context, /*layer_id=*/1);

  std::unordered_map<std::string, torch::Tensor> weights;
  for (const std::string name : {"q_proj", "k_proj", "v_proj"}) {
    weights["self_attn." + name + ".weight"] =
        torch::randn({kProjectionSize, kHiddenSize}, options) * 0.01f;
  }
  weights["self_attn.b_proj.weight"] =
      torch::randn({kHeads, kHiddenSize}, options) * 0.01f;
  for (const std::string name : {"f_a_proj", "g_a_proj"}) {
    weights["self_attn." + name + ".weight"] =
        torch::randn({kHeadDim, kHiddenSize}, options) * 0.01f;
  }
  for (const std::string name : {"f_b_proj", "g_b_proj"}) {
    weights["self_attn." + name + ".weight"] =
        torch::randn({kProjectionSize, kHeadDim}, options) * 0.01f;
  }
  for (const std::string name : {"q_conv1d", "k_conv1d", "v_conv1d"}) {
    weights["self_attn." + name + ".weight"] =
        torch::randn({kProjectionSize, 1, kConvWidth}, options) * 0.1f;
  }
  weights["self_attn.A_log"] = torch::zeros({kHeads}, fp32_options);
  weights["self_attn.dt_bias"] = torch::zeros({kProjectionSize}, fp32_options);
  weights["self_attn.o_norm.weight"] = torch::ones({kHeadDim}, options);
  weights["self_attn.o_proj.weight"] =
      torch::randn({kHiddenSize, kProjectionSize}, options) * 0.01f;
  for (const std::string name :
       {"input_layernorm", "post_attention_layernorm"}) {
    weights[name + ".weight"] = torch::ones({kHiddenSize}, options);
  }
  for (const std::string name : {"hc_attn_", "hc_ffn_"}) {
    weights[name + "fn"] =
        torch::randn({24, 4 * kHiddenSize}, fp32_options) * 0.01f;
    weights[name + "base"] = torch::zeros({24}, fp32_options);
    weights[name + "scale"] = torch::full({3}, 0.1f, fp32_options);
  }
  for (const std::string name : {"gate_proj", "up_proj"}) {
    weights["mlp." + name + ".weight"] =
        torch::randn({kIntermediateSize, kHiddenSize}, options) * 0.01f;
  }
  weights["mlp.down_proj.weight"] =
      torch::randn({kHiddenSize, kIntermediateSize}, options) * 0.01f;
  const StateDict state_dict(weights);
  first_layer->load_state_dict(state_dict);
  second_layer->load_state_dict(state_dict);
  if (mode == LayerTestMode::DECODE_CAPTURE ||
      mode == LayerTestMode::VERIFY_CAPTURE) {
    verify_capture_boundary(
        first_layer,
        args,
        options,
        /*spec_verify=*/mode == LayerTestMode::VERIFY_CAPTURE);
    return;
  }
  const auto make_cache = [&]() {
    return KVCache(LinearAttentionKVCacheTensors{
        torch::zeros({2, kConvWidth - 1, 3 * kProjectionSize}, options),
        torch::zeros({2, kHeads, kHeadDim, kHeadDim}, fp32_options)});
  };
  std::array<KVCache, 2> caches = {make_cache(), make_cache()};
  ModelInputParams input_params;
  input_params.embedding.linear_state_ids = {1};
  input_params.embedding.linear_state_indices =
      torch::tensor(std::vector<int32_t>{1}, int_options);

  const std::vector<int64_t> segment_lengths =
      chunked_prefill ? std::vector<int64_t>{17, 31} : std::vector<int64_t>{33};
  int64_t start_position = 0;
  for (const int64_t tokens : segment_lengths) {
    SCOPED_TRACE(::testing::Message()
                 << "start=" << start_position << " tokens=" << tokens);
    AttentionMetadata metadata;
    metadata.is_prefill = !chunked_prefill;
    metadata.is_chunked_prefill = chunked_prefill;
    metadata.max_query_len = tokens;
    metadata.max_seq_len = start_position + tokens;
    metadata.total_kv_len = start_position + tokens;
    metadata.q_cu_seq_lens = torch::tensor(
        std::vector<int32_t>{0, static_cast<int32_t>(tokens)}, int_options);
    metadata.has_initial_states =
        torch::full({1}, start_position > 0, options.dtype(torch::kBool));
    // All segments fit in one causal-convolution block of 64 tokens.
    metadata.batch = torch::full({2048}, -1, int_options);
    metadata.token_block_offset = torch::full({2048}, -1, int_options);
    metadata.batch[0] = 0;
    metadata.token_block_offset[0] = 0;
    metadata.tot = 1;
    const int64_t chunk_size =
        kernel::mlu::kda_prefill_chunk_size(kHeads, /*use_qk_l2norm=*/true);
    const int64_t num_chunks = (tokens + chunk_size - 1) / chunk_size;
    metadata.chunk_indices =
        torch::stack({torch::zeros({num_chunks}, int_options),
                      torch::arange(num_chunks, int_options)},
                     /*dim=*/1);
    torch::Tensor hidden = torch::randn({tokens, 4, kHiddenSize}, options);
    torch::Tensor positions = torch::arange(
        start_position, start_position + tokens, options.dtype(torch::kInt64));
    std::optional<torch::Tensor> residual;
    std::optional<PendingMHC> pending;
    torch::Tensor output = first_layer->forward(hidden,
                                                residual,
                                                positions,
                                                metadata,
                                                caches[0],
                                                input_params,
                                                &pending,
                                                /*is_last_layer=*/false);
    ASSERT_TRUE(pending.has_value());
    ASSERT_EQ(output.sizes(), (torch::IntArrayRef{tokens, kHiddenSize}));
    output = second_layer->forward(hidden,
                                   residual,
                                   positions,
                                   metadata,
                                   caches[1],
                                   input_params,
                                   &pending,
                                   /*is_last_layer=*/true);
    mlu_device.synchronize_default_stream();
    ASSERT_FALSE(pending.has_value());
    ASSERT_EQ(output.sizes(), (torch::IntArrayRef{tokens, 4, kHiddenSize}));
    ASSERT_TRUE(torch::isfinite(output).all().item<bool>());
    ASSERT_GT(output.abs().sum().item<float>(), 0.0f);
    for (const KVCache& cache : caches) {
      ASSERT_TRUE(torch::isfinite(cache.get_ssm_cache()).all().item<bool>());
      ASSERT_GT(cache.get_ssm_cache()[1].abs().sum().item<float>(), 0.0f);
      ASSERT_GT(cache.get_conv_cache()[1].abs().sum().item<float>(), 0.0f);
    }
    start_position += tokens;
  }
}

TEST(Glm5NextDecoderLayerTest, PrefillDefersMHCThroughTheLayerChain) {
  run_layer_chain(LayerTestMode::PREFILL);
}

TEST(Glm5NextDecoderLayerTest, ChunkedPrefillDefersMHCThroughTheLayerChain) {
  run_layer_chain(LayerTestMode::CHUNKED_PREFILL);
}

TEST(Glm5NextDecoderLayerTest, DecodeCaptureMaterializesMHCAndResumesFusion) {
  run_layer_chain(LayerTestMode::DECODE_CAPTURE);
}

TEST(Glm5NextDecoderLayerTest, VerifyCaptureMaterializesMHCAndResumesFusion) {
  run_layer_chain(LayerTestMode::VERIFY_CAPTURE);
}

TEST(Glm5NextDecoderLayerTest, ResolvesAttentionAndMlpRolesIndependently) {
  ModelArgs args;
  args.layer_types(
      {"linear_attention", "deepseek_sparse_attention", "linear_attention"});
  args.mlp_layer_types({"dense", "sparse", "sparse"});

  const Glm5NextLayerRole layer0 = resolve_glm5_next_layer_role(args, 0);
  EXPECT_EQ(layer0.attention, Glm5NextAttentionRole::KDA);
  EXPECT_EQ(layer0.mlp, Glm5NextMlpRole::DENSE);

  const Glm5NextLayerRole layer1 = resolve_glm5_next_layer_role(args, 1);
  EXPECT_EQ(layer1.attention, Glm5NextAttentionRole::DSA);
  EXPECT_EQ(layer1.mlp, Glm5NextMlpRole::SPARSE);

  const Glm5NextLayerRole layer2 = resolve_glm5_next_layer_role(args, 2);
  EXPECT_EQ(layer2.attention, Glm5NextAttentionRole::KDA);
  EXPECT_EQ(layer2.mlp, Glm5NextMlpRole::SPARSE);
}

}  // namespace
}  // namespace xllm::layer
