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

#include "core/runtime/mlu_graph_planner.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <variant>
#include <vector>

#include "core/framework/model/mtp_topk_state.h"

namespace xllm::mlu {
namespace {

GraphPlannerConfig qwen_config() {
  GraphPlannerConfig config;
  config.options.block_size(16).num_decoding_tokens(1);
  config.capabilities.graph_history =
      MtpGraphHistoryPolicy::QWEN_FULL_ATTENTION;
  config.history_capacity = 32;
  config.max_tokens = 64;
  return config;
}

ModelInputParams decode_input(int64_t history, int64_t rows = 3) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = rows;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = history;
  params.parallel.dp_global_token_nums = {static_cast<int32_t>(rows)};
  params.parallel.dp_is_decode = {1};
  for (int64_t row = 0; row <= rows; ++row) {
    params.attention.host.q_seq_lens.push_back(static_cast<int32_t>(row));
  }
  params.attention.device.block_tables = torch::zeros({rows, 2}, torch::kInt32);
  return params;
}

ModelInputParams grouped_input(int64_t rows = 3) {
  ModelInputParams params = decode_input(/*history=*/16, rows);
  params.attention.device.q_seq_lens = torch::zeros({rows + 1}, torch::kInt32);
  params.attention.device.kv_seq_lens = torch::zeros({rows + 1}, torch::kInt32);
  params.attention.device.new_cache_slots = torch::zeros({rows}, torch::kInt32);
  return params;
}

void expect_eager(const MluGraphPlanner& planner,
                  int64_t tokens,
                  const ModelInputParams& params,
                  EagerReason reason) {
  const GraphDecision decision = planner.plan({tokens, params});
  ASSERT_TRUE(std::holds_alternative<EagerReason>(decision));
  EXPECT_EQ(std::get<EagerReason>(decision), reason);
}

class PlannerTopkState final : public MtpTopkState {
 public:
  int64_t num_rows() const override { return 1; }
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  MtpTopkStatePtr to(const torch::Device& /*device*/) const override {
    return nullptr;
  }
  MtpTopkStatePtr index_select_rows(
      const torch::Tensor& /*index*/) const override {
    return nullptr;
  }
};

TEST(MluGraphPlannerTest, OrdinaryStagesKeepReasonPriority) {
  GraphPlannerConfig config = qwen_config();
  ModelInputParams params = decode_input(16);
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(
      MluGraphPlanner(config).plan({3, params})));

  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  expect_eager(MluGraphPlanner(config), 3, params, EagerReason::NON_DECODE);
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.mtp_topk_state = std::make_shared<PlannerTopkState>();
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.mtp_topk_state.reset();
  params.embedding.input_embedding = torch::ones({3, 2});
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(
      MluGraphPlanner(config).plan({3, params})));
  params.embedding.input_embedding = torch::Tensor();
  params.is_spec_verify = true;
  expect_eager(MluGraphPlanner(config), 3, params, EagerReason::NON_DECODE);

  params.is_spec_verify = false;
  config.options.is_draft_engine(true);
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, OrdinaryDpAdmissionChecksCountsAndDecodeFlags) {
  GraphPlannerConfig config = qwen_config();
  config.options.dp_size(2).world_size(8);
  const MluGraphPlanner planner(config);
  ModelInputParams params = decode_input(16);
  params.parallel.dp_global_token_nums = {3, 3};
  params.parallel.dp_is_decode = {1, 1};
  const GraphDecision equal = planner.plan({3, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(equal));
  EXPECT_EQ(std::get<GraphPlan>(equal).layout.padded_num_tokens, 4);

  params.parallel.dp_global_token_nums = {2, 3};
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, params})));
  params.meta.q_max_seq_len = 2;
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.meta.q_max_seq_len = 1;
  params.parallel.dp_global_token_nums = {3, 0};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_global_token_nums = {3, 3};
  params.parallel.dp_is_decode = {1, 0};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_is_decode = {1};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_is_decode = {1, 1};
  params.meta.q_max_seq_len = 0;
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, OrdinaryDpRejectsCountsOutsideConfiguredGroup) {
  GraphPlannerConfig config = qwen_config();
  config.options.dp_size(2).world_size(2);
  const MluGraphPlanner planner(config);
  ModelInputParams params = decode_input(16);
  // The former RunMode skipped DP validation for a singleton count and never
  // compared a longer count vector with dp_size. Both are malformed for DP=2.
  params.parallel.dp_global_token_nums = {3};
  params.parallel.dp_is_decode = {1};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_global_token_nums = {3, 3, 3};
  params.parallel.dp_is_decode = {1, 1, 1};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
}

TEST(MluGraphPlannerTest, GroupedStagesAndRestrictionsStayDistinct) {
  GraphPlannerConfig config = qwen_config();
  config.options.is_draft_engine(true);
  config.capabilities.supports_grouped_mtp_graph = true;
  ModelInputParams params = grouped_input();
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(
      MluGraphPlanner(config).plan({3, params})));
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  expect_eager(MluGraphPlanner(config), 3, params, EagerReason::NON_DECODE);
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  config.options.cp_size(2);
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  config.options.cp_size(1);
  params.prefill_without_cache = true;
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.prefill_without_cache = false;
  params.mtp_topk_state = std::make_shared<PlannerTopkState>();
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.mtp_topk_state.reset();
  params.attention.device.new_cache_slots = torch::Tensor();
  expect_eager(
      MluGraphPlanner(config), 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.attention.device.new_cache_slots = torch::zeros({3}, torch::kInt32);
  params.embedding.linear_state_ids = {1, 2};
  expect_eager(MluGraphPlanner(config), 3, params, EagerReason::INVALID_LAYOUT);
}

TEST(MluGraphPlannerTest, GroupedVerifyChecksStageAndDpMetadata) {
  GraphPlannerConfig config = qwen_config();
  config.options.enable_speculative_decode(true).speculative_algorithm("MTP");
  config.options.dp_size(2).world_size(2);
  config.capabilities.supports_grouped_mtp_graph = true;
  const MluGraphPlanner planner(config);
  ModelInputParams params = grouped_input();
  params.is_spec_verify = true;
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.num_accepted_tokens = torch::ones({3}, torch::kInt32);
  params.parallel.dp_global_token_nums = {3, 3};
  params.parallel.dp_is_decode = {1, 1};
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, params})));
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  expect_eager(planner, 3, params, EagerReason::NON_DECODE);
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.parallel.dp_global_token_nums = {3, 2};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_global_token_nums = {3};
  params.parallel.dp_is_decode = {1};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_global_token_nums = {3, 3};
  params.parallel.dp_is_decode = {1, 0};
  expect_eager(planner, 3, params, EagerReason::UNSUPPORTED_INPUT);
  params.parallel.dp_is_decode = {1, 1};
  params.num_accepted_tokens = torch::ones({2}, torch::kInt32);
  expect_eager(planner, 3, params, EagerReason::INVALID_LAYOUT);
}

TEST(MluGraphPlannerTest, GroupedPaddingKeepsRequestsAndTpAlignment) {
  GraphPlannerConfig config = qwen_config();
  config.options.is_draft_engine(true).dp_size(2).world_size(12);
  config.capabilities.supports_grouped_mtp_graph = true;
  const MluGraphPlanner planner(config);
  ModelInputParams params = grouped_input(/*rows=*/12);
  params.meta.num_sequences = 3;
  params.attention.host.kpool_query_lens = {4, 4, 4};
  params.parallel.dp_global_token_nums = {12, 12};
  params.parallel.dp_is_decode = {1, 1};
  const GraphDecision decision = planner.plan({12, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(decision));
  const GraphPlan& plan = std::get<GraphPlan>(decision);
  EXPECT_EQ(plan.layout.tokens_per_request, 4);
  EXPECT_EQ(plan.layout.num_reqs, 3);
  EXPECT_EQ(plan.layout.padded_num_reqs, 6);
  EXPECT_EQ(plan.layout.padded_num_tokens, 24);

  config.options.enable_graph_mode_decode_no_padding(true);
  const GraphDecision no_padding = MluGraphPlanner(config).plan({12, params});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(no_padding));
  EXPECT_EQ(std::get<GraphPlan>(no_padding).layout.padded_num_reqs, 3);
  EXPECT_EQ(std::get<GraphPlan>(no_padding).layout.padded_num_tokens, 12);
}

TEST(MluGraphPlannerTest, HistoryAtCapacityReusesLayoutAcrossSteps) {
  MluGraphPlanner planner(qwen_config());
  ModelInputParams first = decode_input(16);
  ModelInputParams boundary = decode_input(32);
  GraphDecision first_decision = planner.plan({3, first});
  GraphDecision boundary_decision = planner.plan({3, boundary});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(first_decision));
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(boundary_decision));
  EXPECT_EQ(std::get<GraphPlan>(first_decision).history_capacity, 32);
  EXPECT_EQ(std::get<GraphPlan>(first_decision).main_block_table_columns, 3);
  EXPECT_TRUE(std::get<GraphPlan>(first_decision).key ==
              std::get<GraphPlan>(boundary_decision).key);
  EXPECT_EQ(std::get<GraphPlan>(first_decision).layout.padded_num_tokens, 4);
}

TEST(MluGraphPlannerTest, ExternalEmbeddingGetsSeparateGraphKey) {
  MluGraphPlanner planner(qwen_config());
  ModelInputParams tokens_only = decode_input(/*history=*/16);
  ModelInputParams external = decode_input(/*history=*/16);
  external.embedding.input_embedding = torch::ones({3, 2});

  const GraphDecision tokens_decision = planner.plan({3, tokens_only});
  const GraphDecision external_decision = planner.plan({3, external});
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(tokens_decision));
  ASSERT_TRUE(std::holds_alternative<GraphPlan>(external_decision));
  const GraphKey& token_key = std::get<GraphPlan>(tokens_decision).key;
  const GraphKey& external_key = std::get<GraphPlan>(external_decision).key;
  EXPECT_FALSE(token_key.has_input_embedding);
  EXPECT_TRUE(external_key.has_input_embedding);
  EXPECT_FALSE(token_key == external_key);
  EXPECT_NE(GraphKeyHash{}(token_key), GraphKeyHash{}(external_key));
  EXPECT_TRUE(external_key ==
              std::get<GraphPlan>(planner.plan({3, external})).key);
}

TEST(MluGraphPlannerTest, HistoryAboveCapacityFallsBackAndCanReturn) {
  MluGraphPlanner planner(qwen_config());
  ModelInputParams over = decode_input(33);
  ModelInputParams back = decode_input(16);
  EXPECT_EQ(std::get<EagerReason>(planner.plan({3, over})),
            EagerReason::HISTORY_LIMIT);
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(planner.plan({3, back})));
}

TEST(MluGraphPlannerTest, InvalidOffsetsAndDistinctMultiTablesStaySeparate) {
  MluGraphPlanner planner(qwen_config());
  ModelInputParams first = decode_input(16);
  ModelInputParams second = decode_input(16);
  first.multi_block_tables.push_back(torch::zeros({3, 4}, torch::kInt32));
  second.multi_block_tables.push_back(torch::zeros({3, 8}, torch::kInt32));
  EXPECT_FALSE(std::get<GraphPlan>(planner.plan({3, first})).key ==
               std::get<GraphPlan>(planner.plan({3, second})).key);
  second.attention.host.q_seq_lens = {1, 2, 3, 4};
  EXPECT_EQ(std::get<EagerReason>(planner.plan({3, second})),
            EagerReason::INVALID_LAYOUT);
}

TEST(MluGraphPlannerTest, NonDecodeAndTokenLimitReturnReasons) {
  GraphPlannerConfig config = qwen_config();
  config.max_tokens = 3;
  MluGraphPlanner planner(config);
  ModelInputParams params = decode_input(16);
  EXPECT_EQ(std::get<EagerReason>(planner.plan({3, params})),
            EagerReason::TOKEN_LIMIT);
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  EXPECT_EQ(std::get<EagerReason>(planner.plan({3, params})),
            EagerReason::NON_DECODE);
}

TEST(MluGraphPlannerTest, DraftAndVerifyUseCurrentStageMetadata) {
  GraphPlannerConfig draft_config = qwen_config();
  draft_config.options.is_draft_engine(true);
  draft_config.capabilities.supports_grouped_mtp_graph = true;
  MluGraphPlanner draft(draft_config);
  ModelInputParams params = decode_input(32);
  params.attention.device.q_seq_lens = torch::zeros({4}, torch::kInt32);
  params.attention.device.kv_seq_lens = torch::zeros({4}, torch::kInt32);
  params.attention.device.new_cache_slots = torch::zeros({3}, torch::kInt32);
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(draft.plan({3, params})));
  params.meta.kv_max_seq_len = 33;
  EXPECT_EQ(std::get<EagerReason>(draft.plan({3, params})),
            EagerReason::HISTORY_LIMIT);

  GraphPlannerConfig verify_config = qwen_config();
  verify_config.options.enable_speculative_decode(true).speculative_algorithm(
      "MTP");
  verify_config.capabilities.supports_grouped_mtp_graph = true;
  MluGraphPlanner verify(verify_config);
  params.meta.kv_max_seq_len = 32;
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.is_spec_verify = true;
  params.num_accepted_tokens = torch::ones({3}, torch::kInt32);
  EXPECT_TRUE(std::holds_alternative<GraphPlan>(verify.plan({3, params})));
}

}  // namespace
}  // namespace xllm::mlu
