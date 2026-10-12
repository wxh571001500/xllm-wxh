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

#include "core/framework/speculative/mtp_async_input_builder.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/kernels/xllm_torch_ops.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/py_attention_metadata.h"
#include "models/llm/py_causal_lm.h"

namespace py = pybind11;

namespace xllm::mtp_async {
namespace {

constexpr int32_t kBlockSize = 4;

ForwardInput make_draft_input(int64_t batch_size, int64_t hidden_size) {
  ForwardInput input;
  input.token_ids = torch::zeros({batch_size * 2}, torch::kInt);
  input.positions = torch::zeros({batch_size * 2}, torch::kInt);
  input.input_params.embedding.input_embedding =
      torch::zeros({batch_size * 2, hidden_size}, torch::kFloat32);
  input.input_params.attention.device.kv_seq_lens =
      torch::zeros({batch_size * 2}, torch::kInt);
  return input;
}

ForwardInput make_block_table_source(const torch::Tensor& block_tables,
                                     std::vector<int32_t> kv_seq_lens) {
  ForwardInput input;
  input.input_params.attention.device.block_tables = block_tables;
  input.input_params.attention.host.block_tables = block_tables;
  input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens);
  input.input_params.multi_block_tables.emplace_back(torch::zeros({1}));
  return input;
}

void prepare_single_sequence(ForwardInput& draft_input,
                             const ForwardInput& block_table_source,
                             int32_t base_kv_seq_len,
                             bool rebuild_expanded_decode_metadata = true) {
  const torch::Tensor accepted_tokens = torch::tensor({{42, -1}}, torch::kLong);
  const torch::Tensor accepted_embeddings =
      torch::tensor({{{1.0F, 2.0F}, {3.0F, 4.0F}}});
  const torch::Tensor embedding_placeholder = torch::zeros({2});
  const torch::Tensor base_positions =
      torch::tensor({base_kv_seq_len - 2}, torch::kInt);
  const torch::Tensor base_kv_seq_lens =
      torch::tensor({base_kv_seq_len - 1}, torch::kInt);

  prepare_next_draft_from_accepted_state(draft_input,
                                         block_table_source,
                                         accepted_tokens,
                                         accepted_embeddings,
                                         embedding_placeholder,
                                         base_positions,
                                         base_kv_seq_lens,
                                         /*use_chunked_prefill=*/false,
                                         rebuild_expanded_decode_metadata,
                                         kBlockSize);
}

void prepend_python_model_path() {
  std::filesystem::path repo_root(__FILE__);
  for (int32_t depth = 0; depth < 5; ++depth) {
    repo_root = repo_root.parent_path();
  }
  py::list sys_path = py::module_::import("sys").attr("path");
  sys_path.attr("insert")(0, repo_root.string());
}

TEST(MtpAsyncInputBuilderTest, BuildsExpandedMetadataAcrossBlockBoundary) {
  ForwardInput draft_input = make_draft_input(/*batch_size=*/1,
                                              /*hidden_size=*/2);
  const torch::Tensor block_tables = torch::tensor({{10, 11}}, torch::kInt);
  ForwardInput block_table_source = make_block_table_source(block_tables, {5});

  prepare_single_sequence(
      draft_input, block_table_source, /*base_kv_seq_len=*/5);

  const auto& attention = draft_input.input_params.attention.device;
  EXPECT_TRUE(torch::equal(draft_input.input_params.graph.expanded_kv_seq_lens,
                           torch::tensor({4, 5}, torch::kInt)));
  EXPECT_EQ(draft_input.input_params.graph.expanded_kv_seq_lens_vec,
            (std::vector<int32_t>{4, 5}));
  EXPECT_TRUE(torch::equal(attention.paged_kv_indptr,
                           torch::tensor({0, 1, 3}, torch::kInt)));
  EXPECT_TRUE(torch::equal(attention.paged_kv_indices,
                           torch::tensor({10, 10, 11}, torch::kInt)));
  EXPECT_TRUE(torch::equal(attention.paged_kv_last_page_len,
                           torch::tensor({4, 1}, torch::kInt)));
}

TEST(MtpAsyncInputBuilderTest, CanSkipExpandedMetadataRebuild) {
  ForwardInput draft_input = make_draft_input(/*batch_size=*/1,
                                              /*hidden_size=*/2);
  const torch::Tensor template_block_tables =
      torch::tensor({{90, 91}, {90, 91}}, torch::kInt);
  draft_input.input_params.attention.device.block_tables =
      template_block_tables;
  const torch::Tensor block_tables = torch::tensor({{10, 11}}, torch::kInt);
  ForwardInput block_table_source = make_block_table_source(block_tables, {5});

  prepare_single_sequence(draft_input,
                          block_table_source,
                          /*base_kv_seq_len=*/5,
                          /*rebuild_expanded_decode_metadata=*/false);

  EXPECT_TRUE(
      torch::equal(draft_input.input_params.attention.device.block_tables,
                   template_block_tables));
  EXPECT_FALSE(draft_input.input_params.graph.expanded_kv_seq_lens.defined());
}

TEST(MtpAsyncInputBuilderTest,
     CorrectsExpandedVerifyForPartialAndFullAcceptanceAcrossBlocks) {
  const torch::Tensor block_tables =
      torch::tensor({{10, 11, 12, 13}, {20, 21, 22, 23}}, torch::kInt);
  ForwardInput validate_input;
  validate_input.token_ids =
      torch::tensor({11, -1, -2, 22, -1, -2}, torch::kInt);
  validate_input.positions = torch::tensor({3, 4, 5, 7, 8, 9}, torch::kInt);
  auto& attention = validate_input.input_params.attention.device;
  attention.block_tables = block_tables;
  attention.kv_seq_lens = torch::tensor({4, 8}, torch::kInt);
  validate_input.input_params.attention.host.kv_seq_lens = {9, 13};
  auto& graph = validate_input.input_params.graph;
  graph.use_expanded_decode_for_spec_verify_attention = true;
  graph.expanded_kv_seq_lens = torch::zeros({6}, torch::kInt);
  graph.expanded_kv_seq_lens_vec = {7, 8, 9, 11, 12, 13};
  validate_input.input_params.meta.num_sequences = 2;
  validate_input.input_params.meta.kv_max_seq_len = 13;
  validate_input.input_params.attention.host.q_seq_lens = {3, 3};
  graph.expanded_block_tables = torch::tensor({{10, 11, 12, 13},
                                               {10, 11, 12, 13},
                                               {10, 11, 12, 13},
                                               {20, 21, 22, 23},
                                               {20, 21, 22, 23},
                                               {20, 21, 22, 23}},
                                              torch::kInt);
  layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
      validate_input.input_params,
      graph.expanded_kv_seq_lens,
      graph.expanded_block_tables,
      graph.expanded_kv_seq_lens_vec,
      kBlockSize);
  const torch::Tensor expected_expanded_block_tables =
      graph.expanded_block_tables.clone();

  prepare_expanded_target_verify_from_accepted_state(
      validate_input,
      torch::tensor({{42, -1, -1}, {73, 74, 75}}, torch::kInt),
      torch::tensor({3, 7}, torch::kInt),
      torch::tensor({4, 8}, torch::kInt),
      kBlockSize);

  EXPECT_TRUE(
      torch::equal(validate_input.token_ids,
                   torch::tensor({42, -1, -2, 75, -1, -2}, torch::kInt)));
  EXPECT_TRUE(torch::equal(validate_input.positions,
                           torch::tensor({4, 5, 6, 10, 11, 12}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(attention.new_cache_slots,
                   torch::tensor({44, 45, 46, 90, 91, 92}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(attention.kv_seq_lens, torch::tensor({7, 13}, torch::kInt)));
  EXPECT_TRUE(torch::equal(graph.expanded_kv_seq_lens,
                           torch::tensor({5, 6, 7, 11, 12, 13}, torch::kInt)));
  EXPECT_TRUE(torch::equal(graph.expanded_block_tables,
                           expected_expanded_block_tables));
  EXPECT_TRUE(validate_input.device_tensors_ready);

  // The host template assumed a different accepted count for the first
  // sequence. Use the host state from the existing target-context flush to
  // update the attention plan without reading the device lengths back.
  refresh_expanded_target_verify_metadata(validate_input, {5, 11}, kBlockSize);
  EXPECT_EQ(validate_input.input_params.attention.host.kv_seq_lens,
            (std::vector<int32_t>{7, 13}));
  EXPECT_EQ(validate_input.input_params.meta.kv_max_seq_len, 13);
  EXPECT_EQ(graph.expanded_kv_seq_lens_vec,
            (std::vector<int32_t>{5, 6, 7, 11, 12, 13}));
  EXPECT_TRUE(torch::equal(graph.expanded_kv_seq_lens,
                           torch::tensor({5, 6, 7, 11, 12, 13}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(graph.expanded_paged_kv_indptr,
                   torch::tensor({0, 2, 4, 6, 9, 12, 16}, torch::kInt)));
  EXPECT_TRUE(torch::equal(
      graph.expanded_paged_kv_indices,
      torch::tensor(
          {10, 11, 10, 11, 10, 11, 20, 21, 22, 20, 21, 22, 20, 21, 22, 23},
          torch::kInt)));
  EXPECT_TRUE(torch::equal(graph.expanded_paged_kv_last_page_len,
                           torch::tensor({1, 2, 3, 3, 4, 1}, torch::kInt)));

  attention.kv_seq_lens = torch::zeros({6}, torch::kInt);
  EXPECT_DEATH(prepare_expanded_target_verify_from_accepted_state(
                   validate_input,
                   torch::tensor({{42, -1, -1}, {73, 74, 75}}, torch::kInt),
                   torch::tensor({3, 7}, torch::kInt),
                   torch::tensor({4, 8}, torch::kInt),
                   kBlockSize),
               "verify KV lengths must be sequence-scoped");
}

TEST(MtpAsyncInputBuilderTest, RejectsEmptyExpandedVerifyQuery) {
  ForwardInput validate_input;
  auto& params = validate_input.input_params;
  params.meta.num_sequences = 1;
  params.attention.host.q_seq_lens = {0};
  params.graph.use_expanded_decode_for_spec_verify_attention = true;
  params.graph.expanded_kv_seq_lens = torch::zeros({1}, torch::kInt);
  params.graph.expanded_block_tables = torch::zeros({1, 1}, torch::kInt);

  EXPECT_DEATH(
      refresh_expanded_target_verify_metadata(validate_input, {5}, kBlockSize),
      "verify query length must be positive");
}

TEST(MtpAsyncInputBuilderTest,
     RefreshesExpandedVerifyHostPlanAfterPartialAcceptance) {
  ForwardInput validate_input;
  auto& params = validate_input.input_params;
  params.meta.num_sequences = 2;
  params.meta.kv_max_seq_len = 10;
  params.attention.host.q_seq_lens = {3, 3};
  params.attention.host.kv_seq_lens = {10, 6};
  params.graph.use_expanded_decode_for_spec_verify_attention = true;
  params.graph.expanded_kv_seq_lens_vec = {8, 9, 10, 4, 5, 6};
  params.graph.expanded_kv_seq_lens =
      torch::tensor({6, 7, 8, 4, 5, 6}, torch::kInt);
  params.graph.expanded_block_tables = torch::tensor({{10, 11, 12},
                                                      {10, 11, 12},
                                                      {10, 11, 12},
                                                      {20, 21, 22},
                                                      {20, 21, 22},
                                                      {20, 21, 22}},
                                                     torch::kInt);
  layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
      params,
      params.graph.expanded_kv_seq_lens,
      params.graph.expanded_block_tables,
      params.graph.expanded_kv_seq_lens_vec,
      kBlockSize);
  EXPECT_EQ(params.graph.expanded_paged_kv_indices.numel(), 13);
  const torch::Tensor corrected_device_kv_lens =
      params.graph.expanded_kv_seq_lens.clone();

  refresh_expanded_target_verify_metadata(validate_input, {6, 4}, kBlockSize);

  EXPECT_EQ(params.attention.host.kv_seq_lens, (std::vector<int32_t>{8, 6}));
  EXPECT_EQ(params.meta.kv_max_seq_len, 8);
  EXPECT_EQ(params.graph.expanded_kv_seq_lens_vec,
            (std::vector<int32_t>{6, 7, 8, 4, 5, 6}));
  EXPECT_TRUE(torch::equal(params.graph.expanded_kv_seq_lens,
                           corrected_device_kv_lens));
  EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_indptr,
                           torch::tensor({0, 2, 4, 6, 7, 9, 11}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(params.graph.expanded_paged_kv_indices,
                   torch::tensor({10, 11, 10, 11, 10, 11, 20, 20, 21, 20, 21},
                                 torch::kInt)));
  EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_last_page_len,
                           torch::tensor({2, 3, 4, 4, 1, 2}, torch::kInt)));
}

TEST(MtpAsyncInputBuilderTest, AdvancesLaterDraftFromAcceptedDeviceBase) {
  ForwardInput draft_input;
  draft_input.positions = torch::zeros({2}, torch::kInt);
  draft_input.input_params.attention.device.kv_seq_lens =
      torch::zeros({2}, torch::kInt);
  ForwardInput block_table_source;
  block_table_source.input_params.attention.device.block_tables =
      torch::tensor({{10, 11, 12, 13}, {20, 21, 22, 23}}, torch::kInt);
  const torch::Tensor base_positions = torch::tensor({4, 10}, torch::kInt);
  const torch::Tensor base_kv_seq_lens = torch::tensor({5, 11}, torch::kInt);

  prepare_later_draft_from_device_base(draft_input,
                                       block_table_source,
                                       base_positions,
                                       base_kv_seq_lens,
                                       /*position_offset=*/1,
                                       kBlockSize);
  EXPECT_TRUE(
      torch::equal(draft_input.positions, torch::tensor({5, 11}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(draft_input.input_params.attention.device.new_cache_slots,
                   torch::tensor({45, 91}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(draft_input.input_params.attention.device.kv_seq_lens,
                   torch::tensor({6, 12}, torch::kInt)));

  prepare_later_draft_from_device_base(draft_input,
                                       block_table_source,
                                       base_positions,
                                       base_kv_seq_lens,
                                       /*position_offset=*/2,
                                       kBlockSize);
  EXPECT_TRUE(
      torch::equal(draft_input.positions, torch::tensor({6, 12}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(draft_input.input_params.attention.device.new_cache_slots,
                   torch::tensor({46, 92}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(draft_input.input_params.attention.device.kv_seq_lens,
                   torch::tensor({7, 13}, torch::kInt)));
}

TEST(MtpAsyncInputBuilderTest, BuildsTokenwiseSpecVerifyKvLengths) {
  EXPECT_EQ(layer::ExpandedDecodeMetadataBuilder::build_tokenwise_kv_seq_lens(
                /*q_seq_lens=*/{2, 1}, /*kv_seq_lens=*/{4, 3}),
            (std::vector<int32_t>{3, 4, 3}));
}

TEST(MtpAsyncInputBuilderTest, KeepsGenericPagedMetadataSeparate) {
  ModelInputParams params;
  params.attention.device.paged_kv_indptr = torch::tensor({0, 1}, torch::kInt);
  params.attention.device.paged_kv_indices = torch::tensor({99}, torch::kInt);
  params.attention.device.paged_kv_last_page_len =
      torch::tensor({1}, torch::kInt);

  layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
      params,
      torch::tensor({3, 4}, torch::kInt),
      torch::tensor({{10}, {10}}, torch::kInt),
      /*expanded_host_kv_seq_lens=*/{3, 4},
      kBlockSize);

  EXPECT_TRUE(torch::equal(params.attention.device.paged_kv_indptr,
                           torch::tensor({0, 1}, torch::kInt)));
  EXPECT_TRUE(torch::equal(params.attention.device.paged_kv_indices,
                           torch::tensor({99}, torch::kInt)));
  EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_indptr,
                           torch::tensor({0, 1, 2}, torch::kInt)));
  EXPECT_TRUE(torch::equal(params.graph.expanded_paged_kv_indices,
                           torch::tensor({10, 10}, torch::kInt)));
}

TEST(MtpAsyncInputBuilderTest, SupportsMaximumBlockTableWidth) {
  ForwardInput draft_input = make_draft_input(/*batch_size=*/1,
                                              /*hidden_size=*/2);
  const torch::Tensor block_tables = torch::tensor({{10, 11}}, torch::kInt);
  ForwardInput block_table_source = make_block_table_source(block_tables, {8});

  prepare_single_sequence(
      draft_input, block_table_source, /*base_kv_seq_len=*/8);

  const auto& attention = draft_input.input_params.attention.device;
  EXPECT_TRUE(torch::equal(attention.paged_kv_indptr,
                           torch::tensor({0, 2, 4}, torch::kInt)));
  EXPECT_TRUE(torch::equal(attention.paged_kv_indices,
                           torch::tensor({10, 11, 10, 11}, torch::kInt)));
  EXPECT_TRUE(torch::equal(attention.paged_kv_last_page_len,
                           torch::tensor({3, 4}, torch::kInt)));
}

TEST(MtpAsyncInputBuilderTest, RejectsPageCountBeyondBlockTableWidth) {
  ForwardInput draft_input = make_draft_input(/*batch_size=*/1,
                                              /*hidden_size=*/2);
  const torch::Tensor block_tables = torch::tensor({{10, 11}}, torch::kInt);
  ForwardInput block_table_source = make_block_table_source(block_tables, {9});

  EXPECT_DEATH(prepare_single_sequence(
                   draft_input, block_table_source, /*base_kv_seq_len=*/9),
               "Expanded KV length exceeds block-table capacity");
}

TEST(MtpAsyncInputBuilderTest, PybindViewOwnsGlobalDpKvMaxSequenceLengths) {
  if (!Py_IsInitialized()) {
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", /*overwrite=*/1);
    Py_InitializeEx(/*initsigs=*/0);
  }
  py::gil_scoped_acquire gil;
  py::module_ main_module = py::module_::import("__main__");
  if (!py::hasattr(main_module, "AttentionMetadataView")) {
    register_attention_metadata_views(main_module);
  }

  const std::vector<std::vector<int32_t>> histories = {{32769, 0}, {0, 32769}};
  for (const std::vector<int32_t>& expected : histories) {
    py::object py_metadata;
    {
      auto metadata = std::make_shared<layer::AttentionMetadata>();
      metadata->kv_seq_lens_vec = {1};
      metadata->max_seq_len = 1;
      metadata->is_dummy = true;

      ModelInputParams params;
      params.parallel.dp_global_token_nums = {expected[0] == 0 ? 0 : 1,
                                              expected[1] == 0 ? 0 : 1};
      params.parallel.dp_global_kv_max_seq_lens = expected;
      PyAttentionMetadataView view(std::move(metadata), params);
      EXPECT_EQ(view.dp_global_kv_max_seq_lens(), expected);
      EXPECT_EQ(view.dp_execution_token_counts(), (std::vector<int32_t>{1, 1}));
      py_metadata = py::cast(std::move(view));

      params.parallel.dp_global_kv_max_seq_lens = {7, 9};
      EXPECT_EQ(py_metadata.attr("dp_global_kv_max_seq_lens")
                    .cast<std::vector<int32_t>>(),
                expected);
    }

    EXPECT_EQ(py_metadata.attr("dp_global_kv_max_seq_lens")
                  .cast<std::vector<int32_t>>(),
              expected);
    py::list lengths = py_metadata.attr("dp_global_kv_max_seq_lens");
    lengths[0] = py::int_(/*value=*/123);
    EXPECT_EQ(py_metadata.attr("dp_global_kv_max_seq_lens")
                  .cast<std::vector<int32_t>>(),
              expected);
    EXPECT_THROW(
        py::setattr(py_metadata, "dp_global_kv_max_seq_lens", py::none()),
        py::error_already_set);
  }
}

TEST(MtpAsyncInputBuilderTest, PybindViewDefaultsGlobalDpKvMaxSequenceLengths) {
  if (!Py_IsInitialized()) {
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", /*overwrite=*/1);
    Py_InitializeEx(/*initsigs=*/0);
  }
  py::gil_scoped_acquire gil;
  py::module_ main_module = py::module_::import("__main__");
  if (!py::hasattr(main_module, "AttentionMetadataView")) {
    register_attention_metadata_views(main_module);
  }

  auto metadata = std::make_shared<layer::AttentionMetadata>();
  py::object py_metadata = py::cast(PyAttentionMetadataView(metadata));
  EXPECT_TRUE(py_metadata.attr("dp_global_kv_max_seq_lens")
                  .cast<std::vector<int32_t>>()
                  .empty());

  ModelInputParams params;
  py::object py_params_metadata =
      py::cast(PyAttentionMetadataView(std::move(metadata), params));
  EXPECT_TRUE(py_params_metadata.attr("dp_global_kv_max_seq_lens")
                  .cast<std::vector<int32_t>>()
                  .empty());
}

TEST(MtpAsyncInputBuilderTest, PybindViewExposesLinearStateReadAndWriteSlots) {
  ensure_xllm_torch_ops_registered();
  if (!Py_IsInitialized()) {
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);
    Py_InitializeEx(0);
  }
  py::gil_scoped_acquire gil;
  prepend_python_model_path();
  py::module_::import("xllm.python._npu_bootstrap");
  py::module_::import("xllm.python").attr("initialize_runtime")();
  py::module_ main_module = py::module_::import("__main__");
  if (!py::hasattr(main_module, "AttentionMetadataView")) {
    register_attention_metadata_views(main_module);
  }

  auto metadata = std::make_shared<layer::AttentionMetadata>();
  metadata->is_prefill = true;
  metadata->is_dummy = true;

  ModelInputParams params;
  params.embedding.linear_state_ids = {3, 7};
  params.embedding.linear_state_indices = torch::tensor({3, 7}, torch::kInt);
  params.embedding.linear_state_read_ids = {2, 6};
  params.meta.batch_forward_type = BatchForwardType::PREFILL;

  py::object py_metadata = py::cast(PyAttentionMetadataView(metadata, params));
  EXPECT_TRUE(torch::equal(
      py_metadata.attr("linear_state_indices").cast<torch::Tensor>(),
      torch::tensor({3, 7}, torch::kInt)));
  EXPECT_TRUE(torch::equal(
      py_metadata.attr("linear_state_write_indices").cast<torch::Tensor>(),
      torch::tensor({3, 7}, torch::kInt)));
  EXPECT_TRUE(torch::equal(
      py_metadata.attr("linear_state_read_indices").cast<torch::Tensor>(),
      torch::tensor({2, 6}, torch::kInt)));
  EXPECT_TRUE(py_metadata.attr("is_dummy").cast<bool>());
  EXPECT_TRUE(py_metadata.attr("num_accepted_tokens").is_none());

  params.num_accepted_tokens = torch::tensor({4, 2}, torch::kInt);
  py::object accepted_metadata =
      py::cast(PyAttentionMetadataView(metadata, params));
  torch::Tensor accepted_counts =
      accepted_metadata.attr("num_accepted_tokens").cast<torch::Tensor>();
  EXPECT_EQ(accepted_counts.data_ptr(), params.num_accepted_tokens.data_ptr());
  EXPECT_TRUE(torch::equal(accepted_counts, params.num_accepted_tokens));
  params.num_accepted_tokens.fill_(1);
  EXPECT_TRUE(torch::equal(accepted_counts, torch::ones({2}, torch::kInt)));
}

TEST(MtpAsyncInputBuilderTest, PybindViewSelectsExpandedGraphMetadata) {
  ensure_xllm_torch_ops_registered();
  if (!Py_IsInitialized()) {
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);
    Py_InitializeEx(0);
  }
  py::gil_scoped_acquire gil;
  prepend_python_model_path();
  py::module_::import("xllm.python._npu_bootstrap");
  py::module_::import("xllm.python").attr("initialize_runtime")();
  py::module_ main_module = py::module_::import("__main__");
  if (!py::hasattr(main_module, "AttentionMetadataView")) {
    register_attention_metadata_views(main_module);
  }

  auto metadata = std::make_shared<layer::AttentionMetadata>();
  metadata->slot_mapping = torch::arange(4, torch::kInt);
  metadata->expanded_decode.enabled = true;
  metadata->expanded_decode.kv_seq_lens =
      torch::tensor({3, 4, 7, 8}, torch::kInt);
  metadata->expanded_decode.block_table =
      torch::tensor({{10, 11}, {10, 11}, {20, 21}, {20, 21}}, torch::kInt);
  metadata->expanded_decode.paged_kv_indptr =
      torch::tensor({0, 1, 2, 4, 6}, torch::kInt);
  metadata->expanded_decode.paged_kv_indices =
      torch::tensor({10, 10, 20, 21, 20, 21}, torch::kInt);
  metadata->expanded_decode.paged_kv_last_page_len =
      torch::tensor({3, 4, 3, 4}, torch::kInt);
  metadata->expanded_decode.kv_seq_lens_host_vec = {3, 4, 7, 8};
  metadata->expanded_decode.kv_seq_lens_host =
      torch::tensor({3, 4, 7, 8}, torch::kInt);

  py::module_ runner_module = py::module_::import(
      "xllm.python.model_executor.runners.decode_acl_graph");
  py::object runner_class = runner_module.attr("DecodeAclGraphRunner");
  py::object runner = runner_class.attr("__new__")(runner_class);
  py::module_ types = py::module_::import("types");
  runner.attr("attention_backend") = types.attr("SimpleNamespace")(
      py::arg("page_size") = kBlockSize, py::arg("is_mla") = false);

  ModelInputParams params;
  params.num_accepted_tokens = torch::tensor({1, 2}, torch::kInt);
  py::object py_metadata = py::cast(PyAttentionMetadataView(metadata, params));
  EXPECT_TRUE(torch::equal(
      py_metadata.attr("num_accepted_tokens").cast<torch::Tensor>(),
      params.num_accepted_tokens));
  py::tuple selected = runner.attr("_decode_metadata")(py_metadata);

  EXPECT_TRUE(torch::equal(selected[0].cast<torch::Tensor>(),
                           metadata->expanded_decode.block_table));
  EXPECT_TRUE(torch::equal(selected[1].cast<torch::Tensor>(),
                           metadata->expanded_decode.kv_seq_lens));
  EXPECT_EQ(selected[2].cast<std::vector<int32_t>>(),
            metadata->expanded_decode.kv_seq_lens_host_vec);
  EXPECT_TRUE(torch::equal(selected[3].cast<torch::Tensor>(),
                           metadata->expanded_decode.paged_kv_indptr));
}

TEST(MtpAsyncInputBuilderTest, PybindViewDecodeStepStaysKvShardFree) {
  // M11.3 Route B pin: an MLA decode step never carries KV-shard metadata.
  // PyExecutorImpl::run attaches KVShardBatchMetadata only for prefill-ish
  // MLA batches under an active CP group (cp_size > 1 && kv_split_size > 1)
  // or for dense non-MLA DCP steps, so the cp_size == 1 + kv_split_size > 1
  // decode shape reaches Python with has_kv_shard=False and
  // kv_split_size=1 -- the C++ shard fields stay inert on decode by
  // construction, and SfaDcpAttentionBackend (the only backend selected for
  // that shape) localizes its own slots in prepare() instead of reading
  // them. A regression that arms decode-side shard metadata would flip
  // these accessors and break m11's replicated index-write contract.
  if (!Py_IsInitialized()) {
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);
    Py_InitializeEx(0);
  }
  py::gil_scoped_acquire gil;

  auto decode_metadata = std::make_shared<layer::AttentionMetadata>();
  decode_metadata->is_prefill = false;
  decode_metadata->is_chunked_prefill = false;
  decode_metadata->kv_seq_lens_vec = {4};
  decode_metadata->max_seq_len = 4;
  PyAttentionMetadataView decode_view(decode_metadata);

  EXPECT_FALSE(decode_view.has_kv_shard());
  EXPECT_EQ(decode_view.kv_split_size(), 1);
  EXPECT_EQ(decode_view.kv_split_rank(), 0);
  EXPECT_TRUE(decode_view.local_slot_mapping().is_none());

  // Contrast: the prefill PCP shape the builder does serve. The same view
  // must expose the shard verbatim, so the decode result above is the
  // builder's phase gate at work, not a view that ignores shard metadata.
  auto prefill_metadata = std::make_shared<layer::AttentionMetadata>();
  prefill_metadata->is_prefill = true;
  prefill_metadata->is_chunked_prefill = false;
  auto shard = std::make_shared<layer::KVShardBatchMetadata>();
  shard->kv_split_size = 2;
  shard->kv_split_rank = 1;
  shard->local_slot_mapping = torch::tensor({0, -1}, torch::kLong);
  prefill_metadata->kv_shard_batch_metadata = shard;
  PyAttentionMetadataView prefill_view(prefill_metadata);

  EXPECT_TRUE(prefill_view.has_kv_shard());
  EXPECT_EQ(prefill_view.kv_split_size(), 2);
  EXPECT_EQ(prefill_view.kv_split_rank(), 1);
  EXPECT_TRUE(
      torch::equal(prefill_view.local_slot_mapping().cast<torch::Tensor>(),
                   shard->local_slot_mapping));
}

TEST(MtpAsyncInputBuilderTest, SharedModulesPointToTargetModel) {
  py::gil_scoped_acquire gil;
  py::module_ types = py::module_::import("types");
  py::object target_lm_head = py::module_::import("builtins").attr("object")();
  py::object target_embedding =
      py::module_::import("builtins").attr("object")();
  py::object target_body =
      types.attr("SimpleNamespace")(py::arg("embed_tokens") = target_embedding);
  py::object target_model = types.attr("SimpleNamespace")(
      py::arg("lm_head") = target_lm_head, py::arg("model") = target_body);
  py::object draft_body =
      types.attr("SimpleNamespace")(py::arg("embed_tokens") = py::none());
  py::object draft_model = types.attr("SimpleNamespace")(
      py::arg("lm_head") = py::none(), py::arg("model") = draft_body);

  ::xllm::detail::share_python_model_weights(draft_model, target_model);

  py::object draft_lm_head = draft_model.attr("lm_head");
  py::object draft_embedding = draft_model.attr("model").attr("embed_tokens");
  EXPECT_TRUE(draft_lm_head.is(target_lm_head));
  EXPECT_TRUE(draft_embedding.is(target_embedding));
}

}  // namespace
}  // namespace xllm::mtp_async
