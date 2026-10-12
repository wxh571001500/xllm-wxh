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

#include "core/framework/speculative/mtp_async_state.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/speculative/mtp_execution_policy.h"
#include "core/framework/speculative/speculative_cache_utils.h"

namespace xllm::mtp_async {
namespace {

TEST(MtpAsyncStateTest, ClassifiesClosedTargetSpecVerifyPolicy) {
  struct TestCase {
    std::string_view model_type;
    TargetSpecVerifyMode native_mode;
    TargetSpecVerifyMode python_mode;
  };
  using Mode = TargetSpecVerifyMode;
  const TestCase test_cases[] = {
      {"qwen3_5", Mode::EXPANDED_VERIFY, Mode::EXPANDED_VERIFY},
      {"qwen3_5_moe", Mode::EXPANDED_VERIFY, Mode::EXPANDED_VERIFY},
      {"qwen3_5_text", Mode::EXPANDED_VERIFY, Mode::EXPANDED_VERIFY},
      {"qwen3_5_moe_text", Mode::EXPANDED_VERIFY, Mode::EXPANDED_VERIFY},
      {"deepseek_v32", Mode::GENERIC, Mode::EXPANDED_VERIFY},
      {"deepseek_v4", Mode::GENERIC, Mode::EXPANDED_VERIFY},
      {"deepseek_v4_dspark", Mode::GENERIC, Mode::EXPANDED_VERIFY},
      {"mimo", Mode::CAUSAL_CHUNKED_PREFILL, Mode::CAUSAL_CHUNKED_PREFILL},
      {"glm5_next", Mode::CAUSAL_CHUNKED_PREFILL, Mode::EXPANDED_VERIFY},
      {"glm5_3_flash", Mode::CAUSAL_CHUNKED_PREFILL, Mode::EXPANDED_VERIFY},
      {"glm5_next_text", Mode::CAUSAL_CHUNKED_PREFILL, Mode::EXPANDED_VERIFY},
      {"glm5_next_mtp", Mode::GENERIC, Mode::GENERIC},
      {"qwen3_next", Mode::GENERIC, Mode::GENERIC},
      {"qwen3_5_mtp", Mode::GENERIC, Mode::GENERIC},
      {"qwen3_5_moe_mtp", Mode::GENERIC, Mode::GENERIC},
      {"glm_moe_dsa", Mode::GENERIC, Mode::GENERIC},
      {"mimo_mtp", Mode::GENERIC, Mode::GENERIC},
      {"unknown_model", Mode::GENERIC, Mode::GENERIC},
  };
  for (const auto& test_case : test_cases) {
    SCOPED_TRACE(test_case.model_type);
    EXPECT_EQ(classify_target_spec_verify_mode(test_case.model_type,
                                               /*is_python_model=*/false),
              test_case.native_mode);
    EXPECT_EQ(classify_target_spec_verify_mode(test_case.model_type,
                                               /*is_python_model=*/true),
              test_case.python_mode);
  }
}

TEST(MtpAsyncStateTest, RequiresUniformVerifyWidthsForRecurrentTargets) {
  for (std::string_view model_type : {"qwen3_5",
                                      "qwen3_5_moe",
                                      "qwen3_5_text",
                                      "qwen3_5_moe_text",
                                      "glm5_next",
                                      "glm5_3_flash",
                                      "glm5_next_text"}) {
    EXPECT_TRUE(requires_uniform_spec_verify(model_type)) << model_type;
  }
  for (std::string_view model_type : {"deepseek_v32",
                                      "deepseek_v4",
                                      "mimo",
                                      "glm5_next_mtp",
                                      "qwen3_5_mtp",
                                      "unknown_model"}) {
    EXPECT_FALSE(requires_uniform_spec_verify(model_type)) << model_type;
  }
}

TEST(MtpAsyncStateTest, RestrictsFusedVerifyTokenUpdateToNativeExecutors) {
  for (TargetSpecVerifyMode mode :
       {TargetSpecVerifyMode::GENERIC,
        TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL,
        TargetSpecVerifyMode::EXPANDED_VERIFY}) {
    EXPECT_FALSE(supports_native_spec_verify_replay_update(mode, true));
    EXPECT_EQ(supports_native_spec_verify_replay_update(mode, false),
              mode == TargetSpecVerifyMode::EXPANDED_VERIFY);
  }
}

TEST(MtpAsyncStateTest, ClassifiesSupportedCombinedDraftExecutionPaths) {
  EXPECT_EQ(classify_combined_draft_execution_path("qwen3_5_mtp"),
            CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION);
  EXPECT_EQ(classify_combined_draft_execution_path("qwen3_5_moe_mtp"),
            CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION);
  EXPECT_EQ(classify_combined_draft_execution_path("glm_moe_dsa_mtp"),
            CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION);
  EXPECT_EQ(classify_combined_draft_execution_path("deepseek_v32_mtp"),
            CombinedDraftExecutionPath::UNSUPPORTED);
  EXPECT_EQ(classify_combined_draft_execution_path("deepseek_v32"),
            CombinedDraftExecutionPath::UNSUPPORTED);
  EXPECT_EQ(classify_combined_draft_execution_path("qwen3_next_mtp"),
            CombinedDraftExecutionPath::UNSUPPORTED);
  EXPECT_EQ(classify_combined_draft_execution_path("mimo_mtp"),
            CombinedDraftExecutionPath::UNSUPPORTED);
}

TEST(MtpAsyncStateTest, RestrictsCombinedDraftToValidatedConfigurations) {
  EXPECT_TRUE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION,
      "TORCH",
      /*dp_size=*/1));
  EXPECT_FALSE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION,
      "ATB",
      /*dp_size=*/1));
  EXPECT_FALSE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION,
      "TORCH",
      /*dp_size=*/2));
  EXPECT_TRUE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION,
      "ATB",
      /*dp_size=*/1));
  EXPECT_TRUE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION,
      "ATB",
      /*dp_size=*/2));
  EXPECT_FALSE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION,
      "TORCH",
      /*dp_size=*/1));
  EXPECT_FALSE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::UNSUPPORTED,
      "ATB",
      /*dp_size=*/1));
}

TEST(MtpAsyncStateTest,
     AllowsDeepseekV32ContinuousDraftOnlyForSupportedConfig) {
  struct TestCase {
    std::string_view target_model_type;
    std::string_view draft_model_type;
    std::string_view npu_backend;
    int32_t dp_size;
    bool is_python_target;
    bool is_python_draft;
    bool has_model_managed_block_tables;
    bool allowed;
    int32_t target_index_topk = 1;
    int32_t draft_index_topk = 1;
  };
  const TestCase cases[] = {
      {"deepseek_v32", "deepseek_v32_mtp", "TORCH", 1, true, true, false, true},
      {"deepseek_v4", "deepseek_v32_mtp", "TORCH", 1, true, true, false, false},
      {"deepseek_v32", "mimo_mtp", "TORCH", 1, true, true, false, false},
      {"deepseek_v32", "deepseek_v32_mtp", "ATB", 1, true, true, false, false},
      {"deepseek_v32",
       "deepseek_v32_mtp",
       "TORCH",
       2,
       true,
       true,
       false,
       false},
      {"deepseek_v32",
       "deepseek_v32_mtp",
       "TORCH",
       1,
       false,
       true,
       false,
       false},
      {"deepseek_v32",
       "deepseek_v32_mtp",
       "TORCH",
       1,
       true,
       false,
       false,
       false},
      {"deepseek_v32", "deepseek_v32_mtp", "TORCH", 1, true, true, true, false},
      // The same gate controls prelaunch, so zero topk cannot enter either
      // the prelaunch or the continuous expanded-verify path.
      {"deepseek_v32",
       "deepseek_v32_mtp",
       "TORCH",
       1,
       true,
       true,
       false,
       false,
       0,
       1},
      {"deepseek_v32",
       "deepseek_v32_mtp",
       "TORCH",
       1,
       true,
       true,
       false,
       false,
       1,
       0},
  };
  for (const TestCase& test_case : cases) {
    SCOPED_TRACE(test_case.npu_backend);
    SCOPED_TRACE(test_case.target_index_topk);
    SCOPED_TRACE(test_case.draft_index_topk);
    EXPECT_EQ(supports_python_dsv32_continuous_draft_configuration(
                  test_case.target_model_type,
                  test_case.draft_model_type,
                  test_case.npu_backend,
                  test_case.dp_size,
                  test_case.is_python_target,
                  test_case.is_python_draft,
                  test_case.has_model_managed_block_tables,
                  test_case.target_index_topk,
                  test_case.draft_index_topk),
              test_case.allowed);
  }
}

TEST(MtpAsyncStateTest, AllowsDraftPrelaunchOnlyForPlainBatches) {
  const std::vector<int32_t> no_dp_metadata;
  const std::vector<int32_t> plain_dp_batch = {0, 0};

  // A single DP rank owns the whole batch, so its local grammar state decides.
  EXPECT_TRUE(json_object_allows_draft_prelaunch(
      /*dp_size=*/1, no_dp_metadata, /*has_local_json_object_states=*/false));
  EXPECT_FALSE(json_object_allows_draft_prelaunch(
      /*dp_size=*/1, no_dp_metadata, /*has_local_json_object_states=*/true));

  // In DP only a complete all-zero vector permits the prelaunch.
  EXPECT_TRUE(json_object_allows_draft_prelaunch(
      /*dp_size=*/2, plain_dp_batch, /*has_local_json_object_states=*/false));
  EXPECT_FALSE(json_object_allows_draft_prelaunch(
      /*dp_size=*/2, {0, 1}, /*has_local_json_object_states=*/false));
  EXPECT_FALSE(json_object_allows_draft_prelaunch(
      /*dp_size=*/2, {1, 0}, /*has_local_json_object_states=*/false));
  EXPECT_FALSE(json_object_allows_draft_prelaunch(
      /*dp_size=*/2, {1, 1}, /*has_local_json_object_states=*/false));
}

TEST(MtpAsyncStateTest, RejectsDraftPrelaunchOnMalformedDpMetadata) {
  // Missing, short, and long metadata is ineligible instead of falling back to
  // rank-local state, which would desynchronise the collective order.
  EXPECT_FALSE(json_object_allows_draft_prelaunch(
      /*dp_size=*/2, {}, /*has_local_json_object_states=*/false));
  EXPECT_FALSE(json_object_allows_draft_prelaunch(
      /*dp_size=*/2, {0}, /*has_local_json_object_states=*/false));
  EXPECT_FALSE(json_object_allows_draft_prelaunch(
      /*dp_size=*/3, {0, 0}, /*has_local_json_object_states=*/false));
  EXPECT_FALSE(json_object_allows_draft_prelaunch(
      /*dp_size=*/0, {}, /*has_local_json_object_states=*/false));
  EXPECT_FALSE(json_object_allows_draft_prelaunch(
      /*dp_size=*/-1, {}, /*has_local_json_object_states=*/false));
  // A flag outside {0, 1} is malformed, not constrained.
  EXPECT_FALSE(json_object_allows_draft_prelaunch(
      /*dp_size=*/2, {0, 2}, /*has_local_json_object_states=*/false));

  // The replicated vector is authoritative in DP: a contradictory rank-local
  // flag must not flip the decision on its own.
  EXPECT_TRUE(json_object_allows_draft_prelaunch(
      /*dp_size=*/2, {0, 0}, /*has_local_json_object_states=*/true));
}

TEST(MtpAsyncStateTest, ExtractsTargetBaseKvLengthsFromVerifyLayouts) {
  const torch::Tensor chunked_kv_seq_lens =
      torch::tensor({104, 204}, torch::kInt);
  const torch::Tensor decode_kv_seq_lens =
      torch::tensor({101, 102, 103, 104, 201, 202, 203, 204}, torch::kInt);

  EXPECT_TRUE(torch::equal(
      extract_target_base_kv_seq_lens(chunked_kv_seq_lens,
                                      /*batch_size=*/2,
                                      /*num_validate_tokens=*/4,
                                      /*use_chunked_prefill=*/true),
      torch::tensor({101, 201}, torch::kInt)));
  EXPECT_TRUE(torch::equal(
      extract_target_base_kv_seq_lens(decode_kv_seq_lens,
                                      /*batch_size=*/2,
                                      /*num_validate_tokens=*/4,
                                      /*use_chunked_prefill=*/false),
      torch::tensor({101, 201}, torch::kInt)));
}

TEST(MtpAsyncStateTest, ComputesSharedSpecVerifyBlockTableCapacity) {
  EXPECT_EQ(speculative_verify_block_table_capacity(262144, 16), 16385);
  EXPECT_EQ(speculative_verify_block_table_capacity(262144, 32), 8193);
  EXPECT_EQ(speculative_verify_block_table_capacity(262144, 64), 4097);
  EXPECT_EQ(speculative_verify_block_table_capacity(262144, 128), 2049);
  EXPECT_EQ(speculative_verify_block_table_capacity(300000, 128), 2345);
}

TEST(MtpAsyncStateTest, AcceptsPrimaryOrModelManagedBlockTableLayouts) {
  const torch::Tensor primary = torch::zeros({2, 4}, torch::kInt32);
  const std::vector<torch::Tensor> model_managed = {
      torch::zeros({2, 3}, torch::kInt32),
      torch::zeros({2, 2}, torch::kInt32),
  };

  EXPECT_TRUE(has_speculative_verify_block_table_layout(
      primary, /*multi_block_tables=*/{}, /*num_sequences=*/2));
  EXPECT_TRUE(has_speculative_verify_block_table_layout(
      torch::Tensor(), model_managed, /*num_sequences=*/2));
  EXPECT_FALSE(has_speculative_verify_block_table_layout(
      torch::Tensor(), /*multi_block_tables=*/{}, /*num_sequences=*/2));
  EXPECT_FALSE(has_speculative_verify_block_table_layout(
      torch::Tensor(),
      {torch::zeros({1, 3}, torch::kInt32)},
      /*num_sequences=*/2));
  EXPECT_FALSE(has_speculative_verify_block_table_layout(
      torch::zeros({2, 4}, torch::kInt64),
      model_managed,
      /*num_sequences=*/2));
}

#if defined(USE_NPU)
TEST(MtpAsyncStateTest, BuildsPinnedZeroFilledSpecVerifyControlBlockTable) {
  const torch::Tensor control =
      make_speculative_verify_control_block_table(/*num_sequences=*/3,
                                                  /*block_table_capacity=*/17);

  EXPECT_TRUE(control.device().is_cpu());
  EXPECT_TRUE(control.is_pinned());
  EXPECT_EQ(control.scalar_type(), torch::kInt32);
  EXPECT_EQ(control.dim(), 2);
  EXPECT_EQ(control.size(0), 3);
  EXPECT_EQ(control.size(1), 17);
  EXPECT_EQ(control.count_nonzero().item<int64_t>(), 0);
}
#endif

TEST(MtpAsyncStateTest, MaterializesDraftColumnsForEagerFallback) {
  torch::Tensor verify_tokens =
      torch::tensor({10, -1, -1, 20, -1, -1}, torch::kInt);
  const std::vector<torch::Tensor> draft_sources = {
      torch::tensor({11, 21}, torch::kLong),
      torch::tensor({12, 22}, torch::kLong)};

  torch::Tensor materialized =
      materialize_speculative_verify_tokens(verify_tokens, draft_sources);

  EXPECT_EQ(materialized.data_ptr(), verify_tokens.data_ptr());
  EXPECT_TRUE(torch::equal(
      materialized, torch::tensor({10, 11, 12, 20, 21, 22}, torch::kInt)));
}

TEST(MtpAsyncStateTest, LeavesOrdinaryEagerTokensUnchanged) {
  const torch::Tensor verify_tokens = torch::tensor({10, 11}, torch::kInt);
  torch::Tensor materialized =
      materialize_speculative_verify_tokens(verify_tokens, {});

  EXPECT_EQ(materialized.data_ptr(), verify_tokens.data_ptr());
  EXPECT_TRUE(torch::equal(materialized, verify_tokens));
}

TEST(MtpAsyncStateTest, SelectsGraphVerifyTokenOverrideForEagerExecution) {
  const torch::Tensor tokens = torch::tensor({1, 2}, torch::kInt);
  GraphInput graph_input;
  graph_input.input_tokens_override =
      torch::tensor({10, -1, -1, 20, -1, -1}, torch::kInt);
  graph_input.spec_verify_draft_token_sources = {
      torch::tensor({11, 21}, torch::kLong),
      torch::tensor({12, 22}, torch::kLong)};

  torch::Tensor materialized =
      materialize_graph_speculative_verify_tokens(tokens, graph_input);

  EXPECT_EQ(materialized.data_ptr(),
            graph_input.input_tokens_override.data_ptr());
  EXPECT_TRUE(torch::equal(
      materialized, torch::tensor({10, 11, 12, 20, 21, 22}, torch::kInt)));
}

TEST(MtpAsyncStateTest, BuildsMixedAcceptanceStateWithoutHostRoundTrip) {
  const torch::Tensor accepted_tokens = torch::tensor(
      {{10, 11, 12, 13}, {20, 21, -1, -1}, {30, -1, -1, -1}}, torch::kLong);
  const torch::Tensor accepted_embeddings =
      torch::arange(24, torch::kFloat).reshape({3, 4, 2});
  const torch::Tensor placeholder = torch::tensor({-100.0, -101.0});
  const torch::Tensor base_positions = torch::tensor({100, 200, 300});
  const torch::Tensor base_kv_seq_lens = torch::tensor({101, 201, 301});

  const AcceptedState state = build_accepted_state(accepted_tokens,
                                                   accepted_embeddings,
                                                   placeholder,
                                                   base_positions,
                                                   base_kv_seq_lens);

  EXPECT_TRUE(torch::equal(state.accepted_lengths,
                           torch::tensor({4, 2, 1}, torch::kLong)));
  EXPECT_TRUE(torch::equal(state.all_draft_accepted,
                           torch::tensor({true, false, false})));
  EXPECT_TRUE(torch::equal(state.last_tokens, torch::tensor({13, 21, 30})));
  EXPECT_TRUE(torch::equal(state.previous_tokens, torch::tensor({12, 20, 30})));
  EXPECT_TRUE(torch::equal(state.last_embeddings,
                           torch::stack({accepted_embeddings[0][3],
                                         accepted_embeddings[1][1],
                                         accepted_embeddings[2][0]})));
  EXPECT_TRUE(torch::equal(state.previous_embeddings,
                           torch::stack({accepted_embeddings[0][2],
                                         accepted_embeddings[1][0],
                                         placeholder})));
  EXPECT_TRUE(torch::equal(state.base_positions,
                           torch::tensor({104, 202, 301}, torch::kLong)));
  EXPECT_TRUE(torch::equal(state.base_kv_seq_lens,
                           torch::tensor({105, 203, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, BuildsTargetMetadataWithoutEmbeddingGather) {
  const torch::Tensor accepted_tokens = torch::tensor(
      {{10, 11, 12, 13}, {20, 21, -1, -1}, {30, -1, -1, -1}}, torch::kLong);
  const torch::Tensor base_positions = torch::tensor({100, 200, 300});
  const torch::Tensor base_kv_seq_lens = torch::tensor({101, 201, 301});

  const AcceptedTokenMetadata metadata = build_accepted_token_metadata(
      accepted_tokens, base_positions, base_kv_seq_lens);

  EXPECT_TRUE(torch::equal(metadata.accepted_lengths,
                           torch::tensor({4, 2, 1}, torch::kLong)));
  EXPECT_TRUE(torch::equal(metadata.last_tokens, torch::tensor({13, 21, 30})));
  EXPECT_TRUE(torch::equal(metadata.base_positions,
                           torch::tensor({104, 202, 301}, torch::kLong)));
  EXPECT_TRUE(torch::equal(metadata.base_kv_seq_lens,
                           torch::tensor({105, 203, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, BuildsRowMetadataForChunkedAndDecodeLayouts) {
  AcceptedState state;
  state.base_positions = torch::tensor({104, 202, 301}, torch::kLong);
  state.base_kv_seq_lens = torch::tensor({105, 203, 302}, torch::kLong);
  const torch::Tensor offsets = torch::tensor({-1, 0}, torch::kLong);

  EXPECT_TRUE(torch::equal(
      make_row_positions(state, offsets),
      torch::tensor({{103, 104}, {201, 202}, {300, 301}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(
      make_kv_seq_lens(state, offsets, /*use_chunked_prefill=*/true),
      state.base_kv_seq_lens));
  EXPECT_TRUE(torch::equal(
      make_kv_seq_lens(state, offsets, /*use_chunked_prefill=*/false),
      torch::tensor({104, 105, 202, 203, 301, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, RedirectsUnusedRepairRowsToScratchPositions) {
  AcceptedState state;
  state.base_positions = torch::tensor({104, 202, 301}, torch::kLong);
  state.all_draft_accepted = torch::tensor({true, false, false});

  EXPECT_TRUE(torch::equal(make_repair_cache_positions(state),
                           torch::tensor({103, 203, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, MapsPositionsAcrossCacheBlockBoundaries) {
  const torch::Tensor block_tables =
      torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kInt);
  const torch::Tensor positions = torch::tensor({{3, 4}, {7, 8}}, torch::kLong);

  EXPECT_TRUE(torch::equal(speculative::map_positions_to_cache_slots(
                               block_tables, positions, /*block_size=*/4),
                           torch::tensor({43, 44, 87, 88}, torch::kInt)));
}

TEST(MtpAsyncStateTest, BuildsLaterDraftMetadataFromAcceptedDeviceBase) {
  AcceptedState state;
  state.base_positions = torch::tensor({3, 7}, torch::kLong);
  state.base_kv_seq_lens = torch::tensor({4, 8}, torch::kInt);
  const torch::Tensor offsets = torch::tensor({2}, torch::kLong);
  const torch::Tensor positions = make_row_positions(state, offsets);
  const torch::Tensor block_tables =
      torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kInt);

  EXPECT_TRUE(torch::equal(positions, torch::tensor({{5}, {9}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(
      make_kv_seq_lens(state, offsets, /*use_chunked_prefill=*/false),
      torch::tensor({6, 10}, torch::kLong)));
  EXPECT_TRUE(torch::equal(speculative::map_positions_to_cache_slots(
                               block_tables, positions, /*block_size=*/4),
                           torch::tensor({45, 89}, torch::kInt)));
}

TEST(MtpAsyncStateTest, KeepsReplaySemanticsTogether) {
  const DraftContextReplaySemantics tail =
      draft_context_replay_semantics(DraftContextUpdate::TAIL_EXTEND, 4);
  EXPECT_FALSE(tail.full_target_replay);
  EXPECT_EQ(tail.target_expansion_width, 2);
  EXPECT_EQ(tail.draft_position_offset, 0);

  const DraftContextReplaySemantics replay = draft_context_replay_semantics(
      DraftContextUpdate::ACCEPTED_SPAN_REPLAY, 4);
  EXPECT_TRUE(replay.full_target_replay);
  EXPECT_EQ(replay.target_expansion_width, 5);
  EXPECT_EQ(replay.draft_position_offset, -1);
}

TEST(MtpAsyncStateTest, MatchesNativeReplayContractsByVerifyMode) {
  EXPECT_TRUE(replay_compatible(TargetSpecVerifyMode::EXPANDED_VERIFY,
                                DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                                /*target_capable=*/true,
                                /*draft_capable=*/true,
                                /*is_python_target=*/false,
                                /*is_python_draft=*/false,
                                /*uses_embedded_eagle3=*/false));
  EXPECT_TRUE(replay_compatible(TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL,
                                DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                                /*target_capable=*/true,
                                /*draft_capable=*/true,
                                /*is_python_target=*/false,
                                /*is_python_draft=*/false,
                                /*uses_embedded_eagle3=*/false));
  EXPECT_FALSE(replay_compatible(TargetSpecVerifyMode::EXPANDED_VERIFY,
                                 DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                                 /*target_capable=*/false,
                                 /*draft_capable=*/true,
                                 /*is_python_target=*/false,
                                 /*is_python_draft=*/false,
                                 /*uses_embedded_eagle3=*/false));
  EXPECT_FALSE(replay_compatible(TargetSpecVerifyMode::EXPANDED_VERIFY,
                                 DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                                 /*target_capable=*/true,
                                 /*draft_capable=*/false,
                                 /*is_python_target=*/false,
                                 /*is_python_draft=*/false,
                                 /*uses_embedded_eagle3=*/false));
  EXPECT_FALSE(replay_compatible(TargetSpecVerifyMode::EXPANDED_VERIFY,
                                 DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                                 /*target_capable=*/true,
                                 /*draft_capable=*/true,
                                 /*is_python_target=*/true,
                                 /*is_python_draft=*/false,
                                 /*uses_embedded_eagle3=*/false));
  EXPECT_FALSE(replay_compatible(TargetSpecVerifyMode::EXPANDED_VERIFY,
                                 DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                                 /*target_capable=*/true,
                                 /*draft_capable=*/true,
                                 /*is_python_target=*/false,
                                 /*is_python_draft=*/false,
                                 /*uses_embedded_eagle3=*/true));
  EXPECT_FALSE(replay_compatible(TargetSpecVerifyMode::EXPANDED_VERIFY,
                                 DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                                 /*target_capable=*/true,
                                 /*draft_capable=*/true,
                                 /*is_python_target=*/false,
                                 /*is_python_draft=*/true,
                                 /*uses_embedded_eagle3=*/false));
  for (TargetSpecVerifyMode mode :
       {TargetSpecVerifyMode::GENERIC, static_cast<TargetSpecVerifyMode>(-1)}) {
    EXPECT_FALSE(replay_compatible(mode,
                                   DraftContextUpdate::ACCEPTED_SPAN_REPLAY,
                                   /*target_capable=*/true,
                                   /*draft_capable=*/true,
                                   /*is_python_target=*/false,
                                   /*is_python_draft=*/false,
                                   /*uses_embedded_eagle3=*/false));
  }
}

TEST(MtpAsyncStateTest, KeepsTailExtensionAvailableWithoutReplayCapabilities) {
  for (TargetSpecVerifyMode mode :
       {TargetSpecVerifyMode::GENERIC,
        TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL,
        TargetSpecVerifyMode::EXPANDED_VERIFY}) {
    EXPECT_TRUE(replay_compatible(mode,
                                  DraftContextUpdate::TAIL_EXTEND,
                                  /*target_capable=*/false,
                                  /*draft_capable=*/false,
                                  /*is_python_target=*/true,
                                  /*is_python_draft=*/true,
                                  /*uses_embedded_eagle3=*/true));
  }
}

}  // namespace
}  // namespace xllm::mtp_async
