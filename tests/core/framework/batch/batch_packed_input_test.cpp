/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/batch/batch_input_builder.h"
#include "core/framework/block/block_manager_impl.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/params_utils.h"

namespace xllm {

TEST(BatchPackedInputTest, ModelInputConversionPreservesValidityMask) {
  ModelInputParams input_params;
  input_params.linear_state_validity_mask = {0, 1, 1, 0};
  ModelInputParams converted = input_params.to(torch::Device(torch::kCPU));
  EXPECT_EQ(converted.linear_state_validity_mask,
            std::vector<int64_t>({0, 1, 1, 0}));
}

template <typename T>
bool tensor_equals_vector(const torch::Tensor& tensor,
                          const std::vector<T>& values) {
  auto flat = tensor.flatten();
  if (flat.size(0) != values.size()) {
    return false;
  }
  for (int64_t i = 0; i < flat.size(0); ++i) {
    if (flat[i].item<T>() != values[static_cast<size_t>(i)]) {
      return false;
    }
  }
  return true;
}

TEST(BatchPackedInputTest, PackedProtoLazyUnpackPreservesLinearReadWriteIds) {
  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);

  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = true;
  seq_params.enable_schedule_overlap = true;

  torch::Tensor input_embedding;
  MMData mm_data;
  BlockManager::Options options;
  options.num_blocks(3).block_size(4);
  BlockManagerImpl manager(options);

  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 2, 3, 4},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);

  seq.add_blocks(BlockType::KV, manager.allocate(1));

  Sequence second_sequence(1,
                           {5, 6, 7, 8},
                           input_embedding,
                           mm_data,
                           IncrementalDecoder("", 1, false, false),
                           seq_params);
  second_sequence.add_blocks(BlockType::KV, manager.allocate(1));

  std::vector<Sequence*> sequences = {&seq, &second_sequence};
  std::vector<uint32_t> budgets = {4, 4};
  BatchInputBuilder builder(sequences,
                            budgets,
                            {},
                            {},
                            nullptr,
                            /*batch_id=*/1,
                            nullptr,
                            BatchForwardType::DECODE);

  ForwardInput input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);
  ASSERT_EQ(input.input_params.meta.num_sequences, 2);
  input.input_params.embedding.linear_state_ids = {7, 8};
  input.input_params.embedding.linear_state_indices =
      torch::tensor({7, 8}, torch::kInt32);
  input.input_params.embedding.linear_state_read_ids = {3, 4};
  input.input_params.embedding.linear_state_read_indices =
      torch::tensor({3, 4}, torch::kInt32);

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(
      packed_input, lazy_input, torch::Device(torch::kCPU), nullptr);
  EXPECT_TRUE(lazy_input.input_params.embedding.linear_state_read_ids.empty());
  EXPECT_TRUE(lazy_input.input_host_buffer_has_layout);

  ForwardInput unpacked_input;
  unpacked_input.input_params.dit_forward_input.emplace().batch_size = 1;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(lazy_input,
                                                    torch::Device(torch::kCPU),
                                                    torch::kFloat32,
                                                    unpacked_input,
                                                    false));
  EXPECT_FALSE(unpacked_input.input_params.dit_forward_input.has_value());
  EXPECT_EQ(unpacked_input.input_params.embedding.linear_state_ids,
            std::vector<int32_t>({7, 8}));
  EXPECT_EQ(unpacked_input.input_params.embedding.linear_state_read_ids,
            std::vector<int32_t>({3, 4}));
  EXPECT_FALSE(unpacked_input.input_params.embedding.linear_state_read_indices
                   .defined());
  ModelInputParams device_params =
      unpacked_input.input_params.to(torch::Device(torch::kCPU));
  EXPECT_TRUE(torch::equal(device_params.embedding.linear_state_read_indices,
                           torch::tensor({3, 4}, torch::kInt32)));
}

TEST(BatchPackedInputTest,
     LinearReadIndicesPreserveHostIdsAndClearWithTargetState) {
  ModelInputParams params;
  params.embedding.linear_state_ids = {7, 8};
  params.embedding.linear_state_read_ids = {3, 4};
  ModelInputParams device_params = params.to(torch::Device(torch::kCPU));
  EXPECT_EQ(device_params.embedding.linear_state_read_ids,
            std::vector<int32_t>({3, 4}));
  EXPECT_TRUE(torch::equal(device_params.embedding.linear_state_read_indices,
                           torch::tensor({3, 4}, torch::kInt32)));
  device_params.clear_linear_attention_state();
  EXPECT_TRUE(device_params.embedding.linear_state_ids.empty());
  EXPECT_TRUE(device_params.embedding.linear_state_read_ids.empty());
  EXPECT_FALSE(device_params.embedding.linear_state_indices.defined());
  EXPECT_FALSE(device_params.embedding.linear_state_read_indices.defined());
}

TEST(BatchPackedInputTest, PackedProtoLazyUnpackRestoresSampleIdxes) {
  RequestSamplingParam sampling_param;
  sampling_param.logprobs = true;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);

  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = true;
  seq_params.enable_schedule_overlap = true;

  torch::Tensor input_embedding;
  MMData mm_data;
  BlockManager::Options options;
  options.num_blocks(2).block_size(4);
  BlockManagerImpl manager(options);

  IncrementalDecoder decoder("", 1, false, false);
  Sequence seq(/*index=*/0,
               /*token_ids=*/{1, 2, 3, 4},
               input_embedding,
               mm_data,
               std::move(decoder),
               seq_params);

  seq.add_blocks(BlockType::KV, manager.allocate(1));

  std::vector<Sequence*> sequences = {&seq};
  std::vector<uint32_t> budgets = {4};
  BatchInputBuilder builder(sequences,
                            budgets,
                            {},
                            {},
                            nullptr,
                            /*batch_id=*/1,
                            nullptr,
                            BatchForwardType::DECODE);

  ForwardInput input =
      builder.build_forward_input(/*num_decoding_tokens=*/1,
                                  /*min_decoding_batch_size=*/0);
  ASSERT_TRUE(input.sampling_params.sample_idxes.defined());
  input.sampling_params.filter_bitmask =
      torch::tensor({{static_cast<int32_t>(0x5)}},
                    torch::TensorOptions().dtype(torch::kInt32));

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(
      packed_input, lazy_input, torch::Device(torch::kCPU), nullptr);
  EXPECT_FALSE(lazy_input.sampling_params.sample_idxes.defined());
  EXPECT_TRUE(lazy_input.input_host_buffer_has_layout);

  ForwardInput unpacked_input;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(lazy_input,
                                                    torch::Device(torch::kCPU),
                                                    torch::kFloat32,
                                                    unpacked_input,
                                                    false));
  ASSERT_TRUE(unpacked_input.sampling_params.sample_idxes.defined());
  EXPECT_TRUE(tensor_equals_vector<int32_t>(
      unpacked_input.sampling_params.sample_idxes, {0}));
  ASSERT_TRUE(unpacked_input.sampling_params.filter_bitmask.defined());
  EXPECT_TRUE(torch::equal(unpacked_input.sampling_params.filter_bitmask,
                           input.sampling_params.filter_bitmask));
}

TEST(BatchPackedInputTest, PackedProtoPreservesDiTGenerationParams) {
  ForwardInput input;
  DiTForwardInput& dit_input = input.input_params.dit_forward_input.emplace();
  dit_input.batch_size = 1;
  dit_input.media_sources.add(
      "prompt_video",
      "video",
      torch::tensor({1, 2}, torch::dtype(torch::kUInt8)),
      {{std::string("prompt_video_fps"), 24.0}});
  dit_input.media_sources.add(
      "image", "image", torch::tensor({3, 4}, torch::dtype(torch::kUInt8)));
  dit_input.media_sources.add(
      "mask_image",
      "image",
      torch::tensor({5, 6}, torch::dtype(torch::kUInt8)));
  dit_input.tensor_sources.add(
      "prompt_embed",
      torch::tensor({1.5f, 2.5f}),
      {{std::string("prompt_token_tags"), std::vector<int64_t>{0, 1}}});
  dit_input.tensor_sources.add("latent", torch::tensor({3.5f, 4.5f}));
  DiTGenerationParams& params = dit_input.generation_params;
  params.width = 640;
  params.height = 480;
  params.num_inference_steps = 31;
  params.true_cfg_scale = 2.5f;
  params.guidance_scale = 6.5f;
  params.num_images_per_prompt = 2;
  params.num_videos_per_prompt = 3;
  params.seed = 1234567;
  params.seed_is_set = true;
  params.max_sequence_length = 768;
  params.strength = 0.75f;
  params.enable_cfg_renorm = false;
  params.cfg_renorm_min = 0.25f;
  params.audio_duration_frames = 321;
  params.audio_steps = 24;
  params.audio_guidance_method = "apg";
  params.audio_sampling_rate = 48000;
  params.num_frames = 49;
  params.video_fps = 12.5;
  params.guidance_scale_2 = 4.25f;
  params.seconds = 7;
  params.boundary_ratio = 0.8f;
  params.flow_shift = 1.25f;
  params.max_new_tokens = 96;
  params.diffusion_steps = 12;
  params.temperature = 0.6f;
  params.top_k = 32;
  params.top_p = 0.85f;
  params.repetition_penalty = 1.2f;

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(
      packed_input, lazy_input, torch::Device(torch::kCPU), nullptr);
  ForwardInput unpacked_input;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(lazy_input,
                                                    torch::Device(torch::kCPU),
                                                    torch::kFloat32,
                                                    unpacked_input,
                                                    false));

  ASSERT_TRUE(unpacked_input.input_params.dit_forward_input.has_value());
  const DiTForwardInput& unpacked_dit_input =
      *unpacked_input.input_params.dit_forward_input;
  EXPECT_EQ(unpacked_dit_input.generation_params, params);
  ASSERT_EQ(unpacked_dit_input.media_sources.size(), 3u);
  EXPECT_EQ(unpacked_dit_input.media_sources.at(0).name, "prompt_video");
  EXPECT_EQ(unpacked_dit_input.media_sources.at(0).modality, "video");
  EXPECT_EQ(unpacked_dit_input.media_sources.at(1).name, "image");
  EXPECT_EQ(unpacked_dit_input.media_sources.at(2).name, "mask_image");
  EXPECT_TRUE(torch::equal(unpacked_dit_input.media_sources.at(0).tensor,
                           dit_input.media_sources.at(0).tensor));
  EXPECT_EQ(unpacked_dit_input.media_sources.at(0).parameters,
            dit_input.media_sources.at(0).parameters);
  ASSERT_EQ(unpacked_dit_input.tensor_sources.size(), 2u);
  EXPECT_TRUE(
      torch::equal(*unpacked_dit_input.tensor_sources.get("prompt_embed"),
                   *dit_input.tensor_sources.get("prompt_embed")));
  EXPECT_EQ(unpacked_dit_input.tensor_sources.get_named_tensor("prompt_embed")
                ->get()
                .parameters,
            dit_input.tensor_sources.get_named_tensor("prompt_embed")
                ->get()
                .parameters);
  EXPECT_TRUE(torch::equal(*unpacked_dit_input.tensor_sources.get("latent"),
                           *dit_input.tensor_sources.get("latent")));

  proto::DiTForwardInput proto_input;
  ASSERT_TRUE(dit_forward_input_to_proto(dit_input, &proto_input));
  DiTForwardInput proto_input_round_trip;
  ASSERT_TRUE(proto_to_dit_forward_input(proto_input, proto_input_round_trip));
  ASSERT_EQ(proto_input_round_trip.media_sources.size(), 3u);
  EXPECT_EQ(proto_input_round_trip.media_sources.at(2).name, "mask_image");
  EXPECT_EQ(proto_input_round_trip.media_sources.at(2).modality, "image");
  EXPECT_TRUE(torch::equal(proto_input_round_trip.media_sources.at(1).tensor,
                           dit_input.media_sources.at(1).tensor));
  EXPECT_EQ(proto_input_round_trip.media_sources.at(0).parameters,
            dit_input.media_sources.at(0).parameters);
  EXPECT_TRUE(
      torch::equal(*proto_input_round_trip.tensor_sources.get("prompt_embed"),
                   *dit_input.tensor_sources.get("prompt_embed")));
  EXPECT_EQ(
      proto_input_round_trip.tensor_sources.get_named_tensor("prompt_embed")
          ->get()
          .parameters,
      dit_input.tensor_sources.get_named_tensor("prompt_embed")
          ->get()
          .parameters);
  EXPECT_TRUE(torch::equal(*proto_input_round_trip.tensor_sources.get("latent"),
                           *dit_input.tensor_sources.get("latent")));

  proto::DiTGenerationParams proto_params;
  ASSERT_TRUE(generation_params_to_proto(params, &proto_params));
  DiTGenerationParams proto_round_trip;
  ASSERT_TRUE(proto_to_generation_params(proto_params, proto_round_trip));
  EXPECT_EQ(proto_round_trip, params);
}

TEST(BatchPackedInputTest, PackedProtoLazyToPreservesJsonMetadata) {
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);

  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.enable_schedule_overlap = true;

  MMData mm_data;
  BlockManager::Options options;
  options.num_blocks(2).block_size(4);
  BlockManagerImpl manager(options);

  IncrementalDecoder decoder("", 1, false, false);
  Sequence sequence(/*index=*/0,
                    /*token_ids=*/{1, 2, 3, 4},
                    torch::Tensor(),
                    mm_data,
                    std::move(decoder),
                    seq_params);
  sequence.add_blocks(BlockType::KV, manager.allocate(1));

  std::vector<Sequence*> sequences = {&sequence};
  std::vector<uint32_t> allowed_max_tokens = {4};
  BatchInputBuilder builder(sequences,
                            allowed_max_tokens,
                            {},
                            {},
                            nullptr,
                            /*batch_id=*/2,
                            nullptr,
                            BatchForwardType::DECODE);
  ForwardInput input = builder.build_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0);

  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  JsonObjectGrammarState state = grammar.initial_state();
  ASSERT_TRUE(state.accept_token(/*open_object=*/0));
  input.json_object_state_snapshots = {state.snapshot()};
  input.sample_sequence_ids = {"req-json#0"};
  input.sample_prior_output_rows = {-1};

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(
      packed_input, lazy_input, torch::Device(torch::kCPU), nullptr);
  EXPECT_TRUE(lazy_input.input_host_buffer_has_layout);
  EXPECT_TRUE(lazy_input.json_object_state_snapshots.empty());

  const ForwardInput materialized_input =
      lazy_input.to(torch::Device(torch::kCPU), torch::kFloat32);
  ASSERT_EQ(materialized_input.json_object_state_snapshots.size(), 1u);
  EXPECT_EQ(materialized_input.json_object_state_snapshots[0].token_ids,
            std::vector<int32_t>({0}));
  EXPECT_FALSE(
      materialized_input.json_object_state_snapshots[0].reasoning_enabled);
  EXPECT_EQ(materialized_input.sample_sequence_ids,
            std::vector<std::string>({"req-json#0"}));
  EXPECT_EQ(materialized_input.sample_prior_output_rows,
            std::vector<int32_t>({-1}));
}

TEST(BatchPackedInputTest, PackedProtoRejectsIncompatibleLayoutHeader) {
  // The assertions below inspect the CHECK message. Under the default fast
  // style gtest forks this already multi-threaded test process, and the child
  // can die before glog flushes, leaving no message to match. The threadsafe
  // style re-executes the binary in a clean child instead.
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  FLAGS_logtostderr = true;

  // The payload header is consumed as [format_tag][descriptor_bytes]
  // [tensor_arena_offset][tensor_arena_bytes] with no length self-description,
  // so a foreign or corrupted header must be rejected before any field is
  // parsed instead of shifting every later field.
  constexpr size_t kRawLayoutHeaderBytes = 4 * sizeof(uint64_t);

  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);

  SequenceParams seq_params;
  seq_params.seq_capacity = 32;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.enable_schedule_overlap = true;

  MMData mm_data;
  BlockManager::Options options;
  options.num_blocks(2).block_size(4);
  BlockManagerImpl manager(options);

  IncrementalDecoder decoder("", 1, false, false);
  Sequence sequence(/*index=*/0,
                    /*token_ids=*/{1, 2, 3, 4},
                    torch::Tensor(),
                    mm_data,
                    std::move(decoder),
                    seq_params);
  sequence.add_blocks(BlockType::KV, manager.allocate(1));

  std::vector<Sequence*> sequences = {&sequence};
  std::vector<uint32_t> allowed_max_tokens = {4};
  BatchInputBuilder builder(sequences,
                            allowed_max_tokens,
                            {},
                            {},
                            nullptr,
                            /*batch_id=*/2,
                            nullptr,
                            BatchForwardType::DECODE);
  ForwardInput input = builder.build_forward_input(
      /*num_decoding_tokens=*/1, /*min_decoding_batch_size=*/0);

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));
  const std::string payload = packed_input.payload();
  ASSERT_GE(payload.size(), kRawLayoutHeaderBytes);

  // An engine build without the format tag leads with the old descriptor
  // length instead, which must not be parsed as a matching tag.
  std::string legacy_payload = payload;
  std::memcpy(legacy_payload.data(),
              payload.data() + sizeof(uint64_t),
              sizeof(uint64_t));
  proto::PackedForwardInput legacy_input;
  legacy_input.mutable_payload()->swap(legacy_payload);
  ForwardInput legacy_lazy;
  packed_proto_to_forward_input(
      legacy_input, legacy_lazy, torch::Device(torch::kCPU), nullptr);
  ForwardInput legacy_unpacked;
  EXPECT_DEATH(detail::unpack_from_input_host_buffer(
                   legacy_lazy,
                   torch::Device(torch::kCPU),
                   torch::kFloat32,
                   legacy_unpacked,
                   /*materialize_device_buffer=*/false),
               "incompatible raw input format");

  // An untrusted arena length must not wrap the layout bounds check.
  std::string corrupt_payload = payload;
  const uint64_t huge_arena_bytes = std::numeric_limits<uint64_t>::max();
  std::memcpy(corrupt_payload.data() + 3 * sizeof(uint64_t),
              &huge_arena_bytes,
              sizeof(huge_arena_bytes));
  proto::PackedForwardInput corrupt_input;
  corrupt_input.mutable_payload()->swap(corrupt_payload);
  ForwardInput corrupt_lazy;
  packed_proto_to_forward_input(
      corrupt_input, corrupt_lazy, torch::Device(torch::kCPU), nullptr);
  ForwardInput corrupt_unpacked;
  EXPECT_DEATH(detail::unpack_from_input_host_buffer(
                   corrupt_lazy,
                   torch::Device(torch::kCPU),
                   torch::kFloat32,
                   corrupt_unpacked,
                   /*materialize_device_buffer=*/false),
               "raw input layout overflow");
}

}  // namespace xllm
