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

#include "core/framework/speculative/embedding_cache.h"

#include <gtest/gtest.h>

#include <array>

#include "core/framework/speculative/mtp_async_replay_builder.h"
#include "core/framework/speculative/spec_input_builder.h"
#include "core/runtime/forward_params.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

namespace {

bool tensor_equal(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  return lhs.defined() && rhs.defined() && torch::equal(lhs, rhs);
}

}  // namespace

TEST(EmbeddingCacheTest, WritePrefillTargetContextAndClear) {
  torch::Device device(Platform::type_torch(), 0);
  EmbeddingCache cache(/*total_nums=*/4);

  std::vector<int32_t> ids = {3, 2};
  std::vector<std::string> request_ids = {"req_0", "req_1"};
  torch::Tensor target_tokens = torch::tensor({31, 41}, torch::kInt);
  torch::Tensor target_embeddings = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});

  cache.write_prefill_target_context(
      ids, request_ids, target_tokens, target_embeddings);

  std::vector<EmbeddingCache::DecodeState> states =
      cache.read_decode_states(ids, request_ids);
  ASSERT_EQ(states.size(), ids.size());
  EXPECT_TRUE(states[0].valid);
  EXPECT_EQ(states[0].request_id, "req_0");
  EXPECT_EQ(states[0].token_id, 31);
  EXPECT_EQ(states[0].position_offset, 0);
  EXPECT_FALSE(states[0].all_draft_accepted);
  EXPECT_EQ(states[0].prev_token_id, -1);
  EXPECT_TRUE(tensor_equal(states[0].embedding, target_embeddings[0]));
  EXPECT_TRUE(states[1].valid);
  EXPECT_EQ(states[1].token_id, 41);
  EXPECT_EQ(states[1].position_offset, 0);
  EXPECT_TRUE(tensor_equal(states[1].embedding, target_embeddings[1]));

  cache.clear(ids);
  states = cache.read_decode_states(ids, request_ids);
  EXPECT_FALSE(states[0].valid);
  EXPECT_EQ(states[0].token_id, 0);
  EXPECT_EQ(states[0].position_offset, 0);
  EXPECT_FALSE(states[0].embedding.defined());
  EXPECT_FALSE(states[1].valid);
  EXPECT_EQ(states[1].token_id, 0);
  EXPECT_EQ(states[1].position_offset, 0);
  EXPECT_FALSE(states[1].embedding.defined());
}

TEST(EmbeddingCacheTest, WritePrefillTargetContextSelectsEmbeddings) {
  EmbeddingCache cache(/*total_nums=*/4);

  std::vector<int32_t> ids = {1, 2};
  std::vector<std::string> request_ids = {"req_0", "req_1"};
  torch::Tensor target_tokens = torch::tensor({51, 61}, torch::kInt);
  torch::Tensor full_embeddings =
      torch::tensor({{1.0f, 1.1f}, {2.0f, 2.1f}, {3.0f, 3.1f}});
  torch::Tensor selected_idxes = torch::tensor({2, 0}, torch::kInt);

  cache.write_prefill_target_context(
      ids, request_ids, target_tokens, full_embeddings, selected_idxes);

  std::vector<EmbeddingCache::DecodeState> states =
      cache.read_decode_states(ids, request_ids);
  ASSERT_EQ(states.size(), ids.size());
  EXPECT_EQ(states[0].token_id, 51);
  EXPECT_TRUE(tensor_equal(states[0].embedding, full_embeddings[2]));
  EXPECT_EQ(states[1].token_id, 61);
  EXPECT_TRUE(tensor_equal(states[1].embedding, full_embeddings[0]));
}

TEST(EmbeddingCacheTest, WriteValidateTargetContext) {
  torch::Device device(Platform::type_torch(), 0);
  EmbeddingCache cache(/*total_nums=*/2);
  std::vector<int32_t> ids = {0, 1};
  std::vector<std::string> request_ids = {"req_0", "req_1"};
  torch::Tensor accepted_tokens =
      torch::tensor({{11, 12, 13}, {21, -1, -1}}, torch::kInt);
  torch::Tensor accepted_embeddings =
      torch::tensor({{{1.0f, 1.1f}, {1.2f, 1.3f}, {1.4f, 1.5f}},
                     {{2.0f, 2.1f}, {2.2f, 2.3f}, {2.4f, 2.5f}}});

  cache.write_target_context(ids,
                             request_ids,
                             accepted_tokens,
                             accepted_embeddings,
                             /*num_speculative_tokens=*/2);

  std::vector<EmbeddingCache::DecodeState> states =
      cache.read_decode_states(ids, request_ids);
  EXPECT_EQ(states[0].token_id, 13);
  EXPECT_EQ(states[0].position_offset, 2);
  EXPECT_TRUE(states[0].all_draft_accepted);
  EXPECT_EQ(states[0].prev_token_id, 12);
  EXPECT_TRUE(
      tensor_equal(states[0].prev_embedding, accepted_embeddings[0][1]));
  EXPECT_TRUE(tensor_equal(states[0].embedding, accepted_embeddings[0][2]));

  EXPECT_EQ(states[1].token_id, 21);
  EXPECT_EQ(states[1].position_offset, 0);
  EXPECT_FALSE(states[1].all_draft_accepted);
  EXPECT_EQ(states[1].prev_token_id, -1);
  EXPECT_TRUE(tensor_equal(states[1].embedding, accepted_embeddings[1][0]));
}

TEST(EmbeddingCacheTest, RequestMismatchMaterializesMissingState) {
  EmbeddingCache cache(/*total_nums=*/2);
  std::vector<int32_t> ids = {0};
  std::vector<std::string> request_ids = {"old_req"};
  torch::Tensor target_tokens = torch::tensor({31}, torch::kInt);
  torch::Tensor target_embeddings = torch::tensor({{1.0f, 2.0f}});

  cache.write_prefill_target_context(
      ids, request_ids, target_tokens, target_embeddings);

  std::vector<EmbeddingCache::DecodeState> states =
      cache.read_decode_states(ids, {"new_req"});
  ASSERT_EQ(states.size(), ids.size());
  EXPECT_FALSE(states[0].valid);
  EXPECT_EQ(states[0].token_id, 0);
  EXPECT_FALSE(states[0].embedding.defined());
}

TEST(EmbeddingCacheTest, ReadAcceptedPrefixLengthsValidatesRequestAndCapacity) {
  EmbeddingCache cache(/*total_nums=*/2);
  std::vector<int32_t> ids = {0, 1};
  std::vector<std::string> request_ids = {"req_0", "req_1"};
  torch::Tensor accepted_tokens =
      torch::tensor({{11, 12, 13}, {21, 22, -1}}, torch::kInt);
  torch::Tensor accepted_embeddings =
      torch::tensor({{{1.0f, 1.1f}, {1.2f, 1.3f}, {1.4f, 1.5f}},
                     {{2.0f, 2.1f}, {2.2f, 2.3f}, {2.4f, 2.5f}}});

  cache.write_target_context(ids,
                             request_ids,
                             accepted_tokens,
                             accepted_embeddings,
                             /*num_speculative_tokens=*/2);

  // Matching request_id returns correction_position_offset + 1.
  std::vector<int32_t> lengths =
      cache.read_accepted_prefix_lengths(ids,
                                         request_ids,
                                         /*max_accepted_tokens=*/3);
  ASSERT_EQ(lengths.size(), ids.size());
  EXPECT_EQ(lengths[0], 3);
  EXPECT_EQ(lengths[1], 2);

  // A preempted-and-reused embedding_id now owned by a fresh request must not
  // leak the previous request's correction offset.
  std::vector<int32_t> reused =
      cache.read_accepted_prefix_lengths({0},
                                         {"new_req"},
                                         /*max_accepted_tokens=*/3);
  ASSERT_EQ(reused.size(), 1u);
  EXPECT_EQ(reused[0], 1);

  EXPECT_DEATH(
      (void)cache.read_accepted_prefix_lengths({0},
                                               {"req_0"},
                                               /*max_accepted_tokens=*/2),
      "accepted prefix length exceeds speculative checkpoint capacity");
}

TEST(EmbeddingCacheTest, BootstrapPreservesExistingPrefillContext) {
  EmbeddingCache cache(2);
  torch::Tensor embedding = torch::tensor({{1.0f, 2.0f}});
  cache.write_prefill_target_context(
      {1}, {"request"}, torch::tensor({17}, torch::kInt), embedding);

  cache.write_mtp_bootstrap_context(
      1, "request", 99, torch::tensor({9.0f, 9.0f}));

  const auto states = cache.read_decode_states({1}, {"request"});
  ASSERT_EQ(states.size(), 1);
  EXPECT_EQ(states[0].token_id, 17);
  EXPECT_TRUE(tensor_equal(states[0].embedding, embedding[0]));
}

class BootstrapAfterValidateTest : public ::testing::TestWithParam<int32_t> {};

TEST_P(BootstrapAfterValidateTest, PreservesAcceptedStateAfterOverlapPause) {
  EmbeddingCache cache(2);
  const int32_t accepted_count = GetParam();
  torch::Tensor accepted_tokens = torch::tensor({{11, 12, 13}}, torch::kInt64);
  accepted_tokens.slice(1, accepted_count).fill_(-1);
  torch::Tensor accepted_embeddings =
      torch::tensor({{{1.0f, 1.1f}, {2.0f, 2.1f}, {3.0f, 3.1f}}});
  cache.write_target_context(
      {1}, {"request"}, accepted_tokens, accepted_embeddings, 2);

  cache.write_mtp_bootstrap_context(
      1, "request", 10 + accepted_count, torch::tensor({9.0f, 9.0f}));

  const auto states = cache.read_decode_states({1}, {"request"});
  ASSERT_EQ(states.size(), 1);
  EXPECT_TRUE(states[0].valid);
  EXPECT_EQ(states[0].request_id, "request");
  EXPECT_EQ(states[0].token_id, 10 + accepted_count);
  EXPECT_EQ(states[0].position_offset, accepted_count - 1);
  EXPECT_EQ(states[0].correction_token_id, 10 + accepted_count);
  EXPECT_EQ(states[0].correction_position_offset, accepted_count - 1);
  EXPECT_EQ(states[0].all_draft_accepted, accepted_count == 3);
  EXPECT_TRUE(tensor_equal(states[0].embedding,
                           accepted_embeddings[0][accepted_count - 1]));
  EXPECT_EQ(cache.read_accepted_prefix_lengths({1},
                                               {"request"},
                                               /*max_accepted_tokens=*/3),
            std::vector<int32_t>({accepted_count}));
  if (accepted_count > 1) {
    EXPECT_EQ(states[0].prev_token_id, 9 + accepted_count);
    EXPECT_TRUE(tensor_equal(states[0].prev_embedding,
                             accepted_embeddings[0][accepted_count - 2]));
  }
}

INSTANTIATE_TEST_SUITE_P(AcceptedWidths,
                         BootstrapAfterValidateTest,
                         ::testing::Values(1, 2, 3));

TEST(EmbeddingCacheTest, BootstrapInitializesRecycledSlotForNewRequest) {
  EmbeddingCache cache(2);
  cache.write_mtp_bootstrap_context(1, "old", 17, torch::tensor({1.0f, 2.0f}));
  torch::Tensor embedding = torch::tensor({3.0f, 4.0f});
  cache.write_mtp_bootstrap_context(1, "new", 27, embedding);

  const auto states = cache.read_decode_states({1}, {"new"});
  ASSERT_EQ(states.size(), 1);
  EXPECT_TRUE(states[0].valid);
  EXPECT_EQ(states[0].token_id, 27);
  EXPECT_TRUE(tensor_equal(states[0].embedding, embedding));
}

TEST(EmbeddingCacheTest, BootstrapInitializesClearedSlotForSameRequest) {
  EmbeddingCache cache(2);
  cache.write_mtp_bootstrap_context(
      1, "request", 17, torch::tensor({1.0f, 2.0f}));
  cache.clear({1});
  torch::Tensor embedding = torch::tensor({3.0f, 4.0f});
  cache.write_mtp_bootstrap_context(1, "request", 27, embedding);

  const auto states = cache.read_decode_states({1}, {"request"});
  ASSERT_EQ(states.size(), 1);
  EXPECT_TRUE(states[0].valid);
  EXPECT_EQ(states[0].token_id, 27);
  EXPECT_TRUE(tensor_equal(states[0].embedding, embedding));
}

TEST(EmbeddingCacheTest,
     BootstrapWithoutRequestIdentityRetainsOverwriteBehavior) {
  EmbeddingCache cache(2);
  cache.write_mtp_bootstrap_context(1, "", 17, torch::tensor({1.0f, 2.0f}));
  cache.write_mtp_bootstrap_context(1, "", 27, torch::tensor({3.0f, 4.0f}));

  const auto states = cache.read_decode_states({1}, {});
  ASSERT_EQ(states.size(), 1);
  EXPECT_EQ(states[0].token_id, 27);
}

TEST(EmbeddingCacheTest, WriteMtpBootstrapContextStoresExactDecodeState) {
  EmbeddingCache cache(/*total_nums=*/2);
  torch::Tensor embedding = torch::tensor({1.0f, 2.0f, 3.0f});

  cache.write_mtp_bootstrap_context(
      /*embedding_id=*/1, "req_bootstrap", /*token_id=*/17, embedding);

  std::vector<EmbeddingCache::DecodeState> states =
      cache.read_decode_states({1}, {"req_bootstrap"});
  ASSERT_EQ(states.size(), 1u);
  EXPECT_TRUE(states[0].valid);
  EXPECT_EQ(states[0].request_id, "req_bootstrap");
  EXPECT_EQ(states[0].token_id, 17);
  EXPECT_EQ(states[0].position_offset, 0);
  EXPECT_FALSE(states[0].all_draft_accepted);
  EXPECT_TRUE(tensor_equal(states[0].embedding, embedding));

  cache.clear({1});
  states = cache.read_decode_states({1}, {"req_bootstrap"});
  EXPECT_FALSE(states[0].valid);
  EXPECT_FALSE(states[0].embedding.defined());
}

namespace {

ForwardInput make_replay_input(const std::vector<int32_t>& tokens,
                               const std::vector<int32_t>& positions) {
  ForwardInput input;
  input.token_ids_host = torch::tensor(tokens, torch::kInt);
  input.positions_host = torch::tensor(positions, torch::kInt);
  input.input_params.meta.num_sequences = static_cast<int32_t>(tokens.size());
  input.input_params.attention.host.block_tables =
      torch::tensor({{2, 5, 9, 10}}, torch::kInt)
          .repeat({static_cast<int64_t>(tokens.size()), 1});
  for (int32_t position : positions) {
    specBuilder::append_seq_len_by_layout(
        input.input_params.attention.host.kv_seq_lens, position + 1);
  }
  return input;
}

}  // namespace

TEST(EmbeddingCacheTest,
     ReplayRetainsEveryAcceptedTargetHiddenAfterProducerReuse) {
  EmbeddingCache cache(/*total_nums=*/2, /*retain_replay_span=*/true);
  for (int32_t span = 1; span <= 4; ++span) {
    auto tokens = torch::tensor({{41, 42, 43, 44}}, torch::kInt);
    tokens.narrow(1, span, 4 - span).fill_(-1);
    auto hidden = torch::arange(8, torch::kFloat).reshape({1, 4, 2});
    const auto expected_hidden = hidden[0].narrow(0, 0, span).clone();
    cache.write_target_context({1},
                               {"req"},
                               tokens,
                               hidden,
                               /*num_speculative_tokens=*/3);
    hidden.fill_(999);
    tokens.fill_(-1);
    const auto states = cache.read_decode_states({1}, {"req"});
    ASSERT_EQ(states[0].replay_token_ids.size(), span);
    EXPECT_TRUE(torch::equal(states[0].replay_embeddings, expected_hidden));

    // The last emitted token is at position 4 + span. Its target hidden,
    // and hence the last draft KV entry, belongs at the preceding position.
    auto input = make_replay_input({40 + span}, {4 + span});
    const auto replay = specBuilder::build_mtp_replay_inputs(
        specBuilder::make_decode_row_context(input),
        states,
        torch::zeros({2}),
        /*block_size=*/4);
    const std::vector<int32_t> all_tokens = {41, 42, 43, 44};
    const std::vector<int32_t> all_positions = {4, 5, 6, 7};
    const std::vector<int32_t> all_slots = {20, 21, 22, 23};
    EXPECT_EQ(
        replay.rows.out_token_ids,
        std::vector<int32_t>(all_tokens.begin(), all_tokens.begin() + span));
    EXPECT_EQ(replay.rows.out_positions,
              std::vector<int32_t>(all_positions.begin(),
                                   all_positions.begin() + span));
    EXPECT_EQ(
        replay.rows.out_new_cache_slots,
        std::vector<int32_t>(all_slots.begin(), all_slots.begin() + span));
    EXPECT_EQ(replay.selected_rows, std::vector<int32_t>({span - 1}));
    EXPECT_TRUE(torch::equal(torch::stack(replay.embeddings), expected_hidden));
    std::vector<int32_t> expected_kv_lens;
    for (int32_t len = 5; len < 5 + span; ++len) {
      specBuilder::append_seq_len_by_layout(expected_kv_lens, len);
    }
    EXPECT_EQ(replay.rows.out_kv_seq_lens, expected_kv_lens);
  }
}

TEST(EmbeddingCacheTest, BuildsDraftReplayInputPlanWithoutRuntimeState) {
  EmbeddingCache cache(/*total_nums=*/1, /*retain_replay_span=*/true);
  cache.write_target_context({0},
                             {"request"},
                             torch::tensor({{41, 42, -1}}, torch::kInt),
                             torch::arange(6, torch::kFloat).reshape({1, 3, 2}),
                             /*num_speculative_tokens=*/2);
  const auto states = cache.read_decode_states({0}, {"request"});
  const auto input = make_replay_input({42}, {5});

  const auto plan =
      mtp_async::build_draft_replay_input_plan(input,
                                               states,
                                               torch::zeros({2}),
                                               /*logical_block_size=*/4,
                                               /*uniform_width=*/3,
                                               /*graph_warmup=*/false,
                                               /*pad_cache_slot=*/-1);

  EXPECT_EQ(plan.rows.out_token_ids, std::vector<int32_t>({0, 41, 42}));
  EXPECT_EQ(plan.rows.out_positions, std::vector<int32_t>({0, 3, 4}));
  EXPECT_EQ(plan.rows.out_new_cache_slots, std::vector<int32_t>({-1, 11, 20}));
  EXPECT_EQ(plan.selected_rows, std::vector<int32_t>({2}));
  EXPECT_EQ(plan.source_sequences, std::vector<int32_t>({0, 0, 0}));
  EXPECT_EQ(plan.valid_rows, std::vector<int32_t>({0, 1, 1}));
  EXPECT_EQ(plan.kpool_query_lens, std::vector<int32_t>({3}));
}

TEST(EmbeddingCacheTest,
     PrefillAndBootstrapReplayOverwriteTailWithoutAppending) {
  EmbeddingCache cache(/*total_nums=*/2, /*retain_replay_span=*/true);
  auto hidden = torch::arange(8, torch::kFloat).reshape({4, 2});
  cache.write_prefill_target_context({0},
                                     {"prefill"},
                                     torch::tensor({40}, torch::kInt),
                                     hidden,
                                     torch::tensor({3}, torch::kInt));
  cache.write_mtp_bootstrap_context(1, "bootstrap", 50, hidden[3]);
  hidden.fill_(999);
  auto states = cache.read_decode_states({0, 1}, {"prefill", "bootstrap"});
  auto input = make_replay_input({40, 50}, {4, 4});
  const auto replay = specBuilder::build_mtp_replay_inputs(
      specBuilder::make_decode_row_context(input),
      states,
      torch::zeros({2}),
      /*block_size=*/4);
  EXPECT_EQ(replay.rows.out_positions, std::vector<int32_t>({3, 3}));
  EXPECT_EQ(replay.rows.out_new_cache_slots, std::vector<int32_t>({11, 11}));
  EXPECT_EQ(replay.rows.out_token_ids, std::vector<int32_t>({40, 50}));
  EXPECT_TRUE(torch::equal(torch::stack(replay.embeddings),
                           torch::tensor({{6.0f, 7.0f}, {6.0f, 7.0f}})));
}

TEST(EmbeddingCacheTest, PrefillReplayKeepsDistinctRotaryAxes) {
  EmbeddingCache cache(/*total_nums=*/1, /*retain_replay_span=*/true);
  torch::Tensor positions =
      torch::tensor({{5, 6}, {10, 11}, {20, 21}}, torch::kInt);
  cache.write_prefill_target_context(
      {0},
      {"vlm"},
      torch::tensor({42}, torch::kInt),
      torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}),
      torch::tensor({1}, torch::kInt),
      positions);
  positions.fill_(999);
  const auto states = cache.read_decode_states({0}, {"vlm"});
  auto input = make_replay_input({42}, {7});
  input.positions_host = torch::tensor({{7}, {7}, {7}}, torch::kInt);
  const auto replay = specBuilder::build_mtp_replay_inputs(
      specBuilder::make_decode_row_context(input),
      states,
      torch::zeros({2}),
      /*block_size=*/4,
      /*uniform_width=*/1);

  EXPECT_EQ(replay.rows.out_positions, std::vector<int32_t>({6}));
  const std::vector<std::array<int32_t, 3>> expected_rotary = {{6, 11, 21}};
  EXPECT_EQ(replay.mrope_positions, expected_rotary);
  EXPECT_TRUE(torch::equal(states[0].replay_mrope_positions,
                           torch::tensor({{6}, {11}, {21}}, torch::kInt)));
}

TEST(EmbeddingCacheTest, ReplayFollowsRequestOrderAndPadsOnlyReservedSlots) {
  EmbeddingCache cache(/*total_nums=*/2, /*retain_replay_span=*/true);
  cache.write_target_context(
      {0, 1},
      {"short", "long"},
      torch::tensor({{99, -1, -1, -1}, {41, 42, 43, -1}}, torch::kInt),
      torch::arange(16, torch::kFloat).reshape({2, 4, 2}),
      /*num_speculative_tokens=*/3);
  const auto states = cache.read_decode_states({1, 0}, {"long", "short"});
  auto input = make_replay_input({43, 99}, {7, 9});
  const auto ctx = specBuilder::make_decode_row_context(input);
  const auto compact = specBuilder::build_mtp_replay_inputs(
      ctx, states, torch::zeros({2}), /*block_size=*/4);
  EXPECT_EQ(compact.rows.out_token_ids, std::vector<int32_t>({41, 42, 43, 99}));
  EXPECT_EQ(compact.rows.out_positions, std::vector<int32_t>({4, 5, 6, 8}));
  EXPECT_EQ(compact.rows.out_new_cache_slots,
            std::vector<int32_t>({20, 21, 22, 36}));
  EXPECT_EQ(compact.selected_rows, std::vector<int32_t>({2, 3}));
  const auto padded = specBuilder::build_mtp_replay_inputs(
      ctx, states, torch::zeros({2}), /*block_size=*/4, /*uniform_width=*/4);
  EXPECT_EQ(padded.rows.out_positions,
            std::vector<int32_t>({0, 4, 5, 6, 0, 0, 0, 8}));
  EXPECT_EQ(padded.rows.out_new_cache_slots,
            std::vector<int32_t>({0, 20, 21, 22, 0, 0, 0, 36}));
  EXPECT_EQ(padded.selected_rows, std::vector<int32_t>({3, 7}));
  EXPECT_EQ(padded.valid_rows, std::vector<int32_t>({0, 1, 1, 1, 0, 0, 0, 1}));
  EXPECT_TRUE(torch::equal(padded.embeddings[7], compact.embeddings[3]));
  const auto stale = cache.read_decode_states({1}, {"replacement"});
  EXPECT_FALSE(stale[0].valid);
  EXPECT_TRUE(stale[0].replay_token_ids.empty());
  EXPECT_FALSE(stale[0].replay_embeddings.defined());
}

TEST(EmbeddingCacheTest, ReplayRejectsMissingContextOutsideGraphWarmup) {
  auto input = make_replay_input({0}, {0});
  const auto ctx = specBuilder::make_decode_row_context(input);
  std::vector<EmbeddingCache::DecodeState> states(1);
  EXPECT_DEATH(specBuilder::build_mtp_replay_inputs(
                   ctx, states, torch::zeros({2}), /*block_size=*/4),
               "requires target context");
  const auto warmup =
      specBuilder::build_mtp_replay_inputs(ctx,
                                           states,
                                           torch::zeros({2}),
                                           /*block_size=*/4,
                                           /*uniform_width=*/0,
                                           /*is_graph_warmup=*/true);
  EXPECT_EQ(warmup.rows.out_positions, std::vector<int32_t>({0}));
  EXPECT_EQ(warmup.rows.out_new_cache_slots, std::vector<int32_t>({0}));
  EXPECT_EQ(warmup.selected_rows, std::vector<int32_t>({0}));
}

}  // namespace xllm
