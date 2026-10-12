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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "framework/config/model_config.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/options.h"

namespace xllm {
namespace {

constexpr int64_t kNumSlots = 7;
constexpr int64_t kNumLayers = 40;

struct LayerState {
  torch::Tensor conv;
  torch::Tensor ssm;
};

using StateSnapshot = std::vector<LayerState>;

StateSnapshot make_states(const torch::Device& device, int64_t stride) {
  StateSnapshot states;
  states.reserve(kNumLayers);
  for (int64_t layer = 0; layer < kNumLayers; ++layer) {
    if (layer % 4 == 3) {
      continue;
    }
    const auto conv_options =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    const auto ssm_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    states.push_back({torch::full({kNumSlots, 3, 8}, -layer - 1, conv_options),
                      torch::arange(kNumSlots * stride * 8, ssm_options)
                              .reshape({kNumSlots * stride, 2, 2, 2}) +
                          layer * 1000});
  }
  return states;
}

StateSnapshot clone_states(const StateSnapshot& states) {
  StateSnapshot snapshot;
  snapshot.reserve(states.size());
  for (const LayerState& state : states) {
    snapshot.push_back({state.conv.clone(), state.ssm.clone()});
  }
  return snapshot;
}

void apply_reference(StateSnapshot& states,
                     const ModelInputParams& params,
                     int64_t stride,
                     bool reads_distinct_state) {
  if (reads_distinct_state) {
    return;
  }
  for (size_t row = 0; row < params.embedding.linear_state_ids.size(); ++row) {
    const int32_t write_id = params.embedding.linear_state_ids[row];
    const int32_t read_id = params.embedding.linear_state_read_ids.empty()
                                ? write_id
                                : params.embedding.linear_state_read_ids[row];
    if (write_id == 0 || read_id == write_id) {
      continue;
    }
    for (LayerState& state : states) {
      state.conv[write_id].copy_(state.conv[read_id]);
      for (int64_t checkpoint = 0; checkpoint < stride; ++checkpoint) {
        state.ssm[write_id * stride + checkpoint].copy_(
            state.ssm[read_id * stride + checkpoint]);
      }
    }
  }
}

void advance_state(StateSnapshot& states,
                   int64_t slot,
                   int64_t read_slot,
                   int64_t stride) {
  for (LayerState& state : states) {
    if (slot != read_slot) {
      state.conv[slot].copy_(state.conv[read_slot]);
      state.ssm.narrow(0, slot * stride, stride)
          .copy_(state.ssm.narrow(0, read_slot * stride, stride));
    }
    state.conv[slot].add_(2);
    for (int64_t checkpoint = 0; checkpoint < stride; ++checkpoint) {
      state.ssm[slot * stride + checkpoint].add_(checkpoint + 1);
    }
  }
}

void expect_bitwise_equal(const StateSnapshot& actual,
                          const StateSnapshot& expected) {
  ASSERT_EQ(actual.size(), 30);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t layer = 0; layer < actual.size(); ++layer) {
    SCOPED_TRACE(layer);
    EXPECT_TRUE(torch::equal(actual[layer].conv.cpu().view(torch::kUInt8),
                             expected[layer].conv.view(torch::kUInt8)));
    EXPECT_TRUE(torch::equal(actual[layer].ssm.cpu().view(torch::kUInt8),
                             expected[layer].ssm.view(torch::kUInt8)));
  }
}

// The three branches of the reads_distinct_state predicate computed inside
// WorkerImpl::prepare_linear_state_cache (worker_impl.cpp): native qwen3.5
// targets read a distinct checkpoint slot for every batch, legacy native
// models restore in place (slot copy), and python glm5_next reads a distinct
// state only for non-spec-verify prefill batches. The family picks the
// worker's model_type; the per-batch predicate below mirrors the worker's.
enum class RestoreFamily {
  kLegacyInPlace,
  kNativeQwenDirectRead,
  kPythonGlm5Next,
};

bool reads_distinct_state_for(RestoreFamily family,
                              const ModelInputParams& params) {
  switch (family) {
    case RestoreFamily::kLegacyInPlace:
      return false;
    case RestoreFamily::kNativeQwenDirectRead:
      return true;
    case RestoreFamily::kPythonGlm5Next:
      // Mirrors the python-model branch of the worker predicate
      // (worker_impl.cpp, prepare_linear_state_cache):
      //   is_python_model && is_glm5_next_target_model(args) &&
      //   params.meta.batch_forward_type.no_decode() && !params.is_spec_verify
      // (the python model impl is pinned by the fixture's ModelConfig).
      return params.meta.batch_forward_type.no_decode() &&
             !params.is_spec_verify;
  }
  return false;
}

class LifecycleWorker final : public LLMWorkerImpl {
 public:
  LifecycleWorker(const ParallelArgs& parallel_args,
                  const torch::Device& device,
                  const runtime::Options& options,
                  int64_t stride,
                  RestoreFamily family)
      : LLMWorkerImpl(parallel_args, device, options),
        stride_(stride),
        family_(family) {
    dtype_ = torch::kBFloat16;
    ModelArgs model_args;
    switch (family_) {
      case RestoreFamily::kLegacyInPlace:
        model_args.model_type("legacy_linear_test");
        break;
      case RestoreFamily::kNativeQwenDirectRead:
        model_args.model_type("qwen3_5");
        break;
      case RestoreFamily::kPythonGlm5Next:
        model_args.model_type("glm5_next");
        break;
    }
    std::vector<std::string> layer_types;
    layer_types.reserve(kNumLayers);
    for (int64_t layer = 0; layer < kNumLayers; ++layer) {
      layer_types.push_back(layer % 4 == 3 ? "full_attention"
                                           : "linear_attention");
    }
    model_args.layer_types(std::move(layer_types));
    model_args.n_layers(kNumLayers);
    context_ =
        ModelContext(parallel_args,
                     model_args,
                     QuantArgs(),
                     torch::TensorOptions().dtype(dtype_).device(device));
    auto stream_guard = compute_stream_->set_stream_guard();
    states_ = make_states(device, stride);
    kv_caches_.reserve(kNumLayers);
    size_t linear_layer = 0;
    for (int64_t layer = 0; layer < kNumLayers; ++layer) {
      if (layer % 4 == 3) {
        kv_caches_.emplace_back();
        continue;
      }
      const LayerState& state = states_[linear_layer++];
      kv_caches_.emplace_back(
          LinearAttentionKVCacheTensors{state.conv, state.ssm});
    }
    snapshots_.reserve(10);
    CHECK_EQ(compute_stream_->synchronize(), 0);
  }

  void prepare_input(const ForwardInput& input, ForwardInput& processed) {
    if (enable_schedule_overlap()) {
      prepare_work_before_execute(input, processed);
    } else {
      prepare_work_before_execute_on_stream(
          input, processed, *compute_stream_, /*record_ready_event=*/false);
    }
  }

  void run(ForwardInput& input) {
    if (enable_schedule_overlap()) {
      step_for_schedule_overlap(input);
    } else {
      execute_no_sync_on_stream(input, *compute_stream_);
    }
  }

  void prepare_state(ModelInputParams& params) {
    auto stream_guard = compute_stream_->set_stream_guard();
    prepare_linear_state_cache(params);
  }

  std::optional<ForwardOutput> execute_no_sync_on_stream(
      const ForwardInput& input,
      Stream& compute_stream) override {
    auto stream_guard = compute_stream.set_stream_guard();
    CHECK(compute_stream.wait_event(input.metadata_ready_event));
    // Capture the validity mask as seen by the batch's execute: under overlap
    // the deferred restore inside step_for_schedule_overlap has already run
    // by this point, so a marked mask here proves the restore executed inside
    // the step task rather than after it.
    masks_at_execute_entry_.push_back(
        input.input_params.linear_state_validity_mask);
    snapshots_.push_back(clone_states(states_));
    advance_state(
        states_,
        input.input_params.embedding.linear_state_ids.front(),
        input.input_params.embedding.linear_state_read_ids.empty()
            ? input.input_params.embedding.linear_state_ids.front()
            : input.input_params.embedding.linear_state_read_ids.front(),
        stride_);
    return std::nullopt;
  }

  void finish() {
    auto stream_guard = compute_stream_->set_stream_guard();
    snapshots_.push_back(clone_states(states_));
    ASSERT_EQ(compute_stream_->synchronize(), 0);
  }

  const std::vector<StateSnapshot>& snapshots() const { return snapshots_; }

  const std::vector<std::vector<int64_t>>& masks_at_execute_entry() const {
    return masks_at_execute_entry_;
  }

 private:
  const int64_t stride_;
  RestoreFamily family_;
  StateSnapshot states_;
  std::vector<StateSnapshot> snapshots_;
  std::vector<std::vector<int64_t>> masks_at_execute_entry_;
};

using LifecycleParameters = std::tuple<int64_t, bool, bool>;

// Shared harness state driving a LifecycleWorker against a CPU reference
// snapshot sequence. Derived fixtures choose the RestoreFamily and the
// ModelConfig model_impl in SetUp().
class LinearStateLifecycleTestBase {
 protected:
  void construct_worker(const torch::Device& device, RestoreFamily family) {
    family_ = family;
    const ParallelArgs parallel_args(
        /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
    runtime::Options options;
    options.enable_schedule_overlap(overlap_);
    worker_ = std::make_unique<LifecycleWorker>(
        parallel_args, device, options, stride_, family_);
    reference_ = make_states(torch::Device(torch::kCPU), stride_);
    expected_.reserve(10);
  }

  ForwardInput prepare(
      std::vector<int32_t> write_ids,
      std::vector<int32_t> read_ids,
      std::vector<int32_t> cached_tokens,
      BatchForwardType forward_type = BatchForwardType::PREFILL,
      bool is_spec_verify = false) {
    ForwardInput input;
    const int32_t rows = static_cast<int32_t>(write_ids.size());
    input.token_ids = torch::ones({rows}, torch::kInt32);
    input.positions = torch::zeros_like(input.token_ids);
    input.input_params.meta.num_sequences = rows;
    input.input_params.meta.q_max_seq_len = 1;
    input.input_params.meta.batch_forward_type = forward_type;
    input.input_params.is_spec_verify = is_spec_verify;
    input.input_params.attention.host.q_seq_lens.reserve(rows + 1);
    for (int32_t row = 0; row <= rows; ++row) {
      input.input_params.attention.host.q_seq_lens.push_back(row);
    }
    input.input_params.attention.host.kv_cache_tokens_nums =
        std::move(cached_tokens);
    input.input_params.embedding.linear_state_ids = std::move(write_ids);
    input.input_params.embedding.linear_state_read_ids = std::move(read_ids);
    ForwardInput processed;
    worker_->prepare_input(input, processed);
    EXPECT_EQ(processed.input_params.embedding.linear_state_ids,
              input.input_params.embedding.linear_state_ids);
    EXPECT_EQ(processed.input_params.embedding.linear_state_read_ids,
              input.input_params.embedding.linear_state_read_ids);
    return processed;
  }

  void run(ForwardInput& input, const std::vector<int64_t>& expected_mask) {
    apply_reference(reference_,
                    input.input_params,
                    stride_,
                    reads_distinct_state_for(family_, input.input_params));
    expected_.push_back(clone_states(reference_));
    worker_->run(input);
    EXPECT_EQ(input.input_params.linear_state_validity_mask, expected_mask);
    advance_state(
        reference_,
        input.input_params.embedding.linear_state_ids.front(),
        input.input_params.embedding.linear_state_read_ids.empty()
            ? input.input_params.embedding.linear_state_ids.front()
            : input.input_params.embedding.linear_state_read_ids.front(),
        stride_);
  }

  void verify() {
    worker_->finish();
    expected_.push_back(clone_states(reference_));
    const auto& actual = worker_->snapshots();
    ASSERT_EQ(actual.size(), expected_.size());
    for (size_t step = 0; step < actual.size(); ++step) {
      SCOPED_TRACE(step);
      expect_bitwise_equal(actual[step], expected_[step]);
    }
  }

  int64_t stride_ = 0;
  bool overlap_ = false;
  RestoreFamily family_ = RestoreFamily::kLegacyInPlace;
  std::string previous_model_impl_;
  std::unique_ptr<LifecycleWorker> worker_;
  StateSnapshot reference_;
  std::vector<StateSnapshot> expected_;
};

class NpuLinearStateLifecycleTest
    : public LinearStateLifecycleTestBase,
      public ::testing::TestWithParam<LifecycleParameters> {
 protected:
  void SetUp() override {
    previous_model_impl_ = ModelConfig::get_instance().model_impl();
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "An NPU is required for LINEAR lifecycle tests.";
    }
    Device device(/*device_index=*/0);
    device.set_device();
    device.init_device_context();
    stride_ = std::get<0>(GetParam());
    overlap_ = std::get<1>(GetParam());
    ModelConfig::get_instance().model_impl("native");
    construct_worker(device.unwrap(),
                     std::get<2>(GetParam())
                         ? RestoreFamily::kNativeQwenDirectRead
                         : RestoreFamily::kLegacyInPlace);
  }

  void TearDown() override {
    ModelConfig::get_instance().model_impl(previous_model_impl_);
  }
};

TEST_P(NpuLinearStateLifecycleTest, PreservesReusedColdSlots) {
  std::vector<ForwardInput> inputs;
  inputs.reserve(2);
  for (int32_t request = 0; request < 2; ++request) {
    inputs.push_back(prepare({3}, {3}, {0}));
    run(inputs.back(), {0});
  }
  verify();
}

TEST_P(NpuLinearStateLifecycleTest, PreparesMixedRows) {
  ForwardInput input = prepare({1, 2, 3, 5}, {4, 2, 3, 5}, {0, 512, 0, 0});
  run(input, {1, 1, 0, 0});
  verify();
}

TEST_P(NpuLinearStateLifecycleTest, PreparesNineChunksInOrder) {
  constexpr int32_t kChunks = 9;
  const auto prepare_chunk = [this](int32_t chunk) {
    const int32_t write_id = chunk % 3 + 1;
    const int32_t read_id = chunk == 0 ? write_id : (chunk - 1) % 3 + 1;
    return prepare({write_id},
                   {read_id},
                   {chunk * 512},
                   BatchForwardType::CHUNKED_PREFILL);
  };
  std::vector<ForwardInput> inputs;
  inputs.reserve(kChunks);
  inputs.push_back(prepare_chunk(0));
  for (int32_t chunk = 0; chunk < kChunks; ++chunk) {
    if (overlap_ && chunk + 1 < kChunks) {
      inputs.push_back(prepare_chunk(chunk + 1));
    }
    run(inputs[chunk], {chunk == 0 ? 0 : 1});
    if (!overlap_ && chunk + 1 < kChunks) {
      inputs.push_back(prepare_chunk(chunk + 1));
    }
  }
  verify();
}

TEST_P(NpuLinearStateLifecycleTest, PreservesWarmInPlaceState) {
  ForwardInput input = prepare({2}, {2}, {2046}, BatchForwardType::DECODE);
  run(input, {1});
  verify();
}

TEST_P(NpuLinearStateLifecycleTest, RotatesDecodeSlots) {
  ForwardInput input = prepare({2}, {1}, {2050}, BatchForwardType::DECODE);
  run(input, {1});
  verify();
}

TEST_P(NpuLinearStateLifecycleTest, RotatesSpeculativeVerifySlots) {
  ForwardInput input =
      prepare({2}, {1}, {2050}, BatchForwardType::DECODE, true);
  run(input, {1});
  verify();
}

TEST_P(NpuLinearStateLifecycleTest, PreservesPaddingSlot) {
  ForwardInput input = prepare({2, 0}, {2, 0}, {0, 128});
  run(input, {0, 0});
  verify();
}

TEST_P(NpuLinearStateLifecycleTest, DefaultsMissingReadIdsToWriteIds) {
  ForwardInput input = prepare({2}, {}, {0});
  run(input, {0});
  verify();
}

TEST_P(NpuLinearStateLifecycleTest, PreservesColdSlotsDuringPrepare) {
  ForwardInput first = prepare({2}, {2}, {0});
  if (overlap_) {
    ForwardInput second = prepare({2}, {2}, {0});
    run(first, {0});
    run(second, {0});
  } else {
    run(first, {0});
    ForwardInput second = prepare({2}, {2}, {0});
    run(second, {0});
  }
  verify();
}

TEST_P(NpuLinearStateLifecycleTest, PreparesExpandedSpeculativeRows) {
  ForwardInput input =
      prepare({2, 3}, {1, 3}, {2048, 0}, BatchForwardType::DECODE, true);
  input.input_params.linear_state_validity_mask = {1, 1, 1, 0, 0, 0};
  worker_->prepare_state(input.input_params);
  run(input, {1, 1, 1, 0, 0, 0});
  verify();
}

TEST_P(NpuLinearStateLifecycleTest, EmptyShardDoesNotTouchCache) {
  ModelInputParams params;
  params.embedding.linear_state_ids = {2};
  params.embedding.linear_state_read_ids = {2};
  worker_->prepare_state(params);
  verify();
}

INSTANTIATE_TEST_SUITE_P(
    CheckpointLayouts,
    NpuLinearStateLifecycleTest,
    ::testing::Combine(::testing::Values(int64_t{1}, int64_t{3}),
                       ::testing::Bool(),
                       ::testing::Bool()),
    [](const ::testing::TestParamInfo<LifecycleParameters>& info) {
      return "Stride" + std::to_string(std::get<0>(info.param)) +
             (std::get<1>(info.param) ? "Overlap" : "Prepare") +
             (std::get<2>(info.param) ? "DirectRead" : "InPlaceBackend");
    });

// ===========================================================================
// Python glm5_next KDA linear-state restore under schedule overlap.
//
// The glm5_next CP overlap ban (master.cpp: "initially requires
// enable_schedule_overlap=false") keeps these paths dark today; these tests
// pin the two overlap dependencies the M11.6 gate lift relies on:
//
//   1. Deferral: with overlap ON, prepare_work_before_execute_on_stream
//      SKIPS prepare_linear_state_cache (worker_impl.cpp,
//      `if (!enable_schedule_overlap()) prepare_linear_state_cache(...)`);
//      the restore instead runs inside the step task on the compute stream
//      (LLMWorkerImpl::step_for_schedule_overlap, llm_worker_impl.cpp),
//      FIFO-serialized with the previous step's device work by the
//      single-threaded task pool (worker_impl.h). The base
//      WorkerImpl::step_for_schedule_overlap additionally orders it after the
//      metadata_ready_event wait (worker_impl.cpp).
//   2. Mark-only restore: for a glm5_next python prefill batch
//      (reads_distinct_state=true) restore_linear_state_slots only marks the
//      validity mask -- no slot copies (linear_state_restore.cpp,
//      `if (!reads_distinct_state) copy_linear_state_slot(...)`); decode and
//      spec-verify batches fall back to the copy path.
// ===========================================================================

using Glm5NextPythonRestoreParameters = std::tuple<int64_t, bool>;

class NpuGlm5NextPythonRestoreTest
    : public LinearStateLifecycleTestBase,
      public ::testing::TestWithParam<Glm5NextPythonRestoreParameters> {
 protected:
  void SetUp() override {
    previous_model_impl_ = ModelConfig::get_instance().model_impl();
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "An NPU is required for glm5_next python restore tests.";
    }
    Device device(/*device_index=*/0);
    device.set_device();
    device.init_device_context();
    stride_ = std::get<0>(GetParam());
    overlap_ = std::get<1>(GetParam());
    // The python-model branch of the reads_distinct_state predicate only
    // fires for python model impls (ModelConfig::is_python_model_impl).
    ModelConfig::get_instance().model_impl("python");
    construct_worker(device.unwrap(), RestoreFamily::kPythonGlm5Next);
  }

  void TearDown() override {
    ModelConfig::get_instance().model_impl(previous_model_impl_);
  }
};

TEST_P(NpuGlm5NextPythonRestoreTest,
       PrefillRestoreIsDeferredAndMarksValidityOnly) {
  // Cold rotated prefill row: the glm5_next python backend materializes slot
  // 2's distinct read of slot 1 itself, so the restore may only mark the
  // validity mask (no copy_linear_state_slot).
  ForwardInput input = prepare({2}, {1}, {0}, BatchForwardType::PREFILL);
  if (overlap_) {
    // Deferral pin: the prepare path skipped the restore, so the mask still
    // holds the pre-restore warm value (cold row -> 0). If the restore had
    // run at prepare time, the rotated row would already read 1 here.
    EXPECT_EQ(input.input_params.linear_state_validity_mask,
              std::vector<int64_t>({0}));
  } else {
    // Without overlap the restore runs inside
    // prepare_work_before_execute_on_stream (worker_impl.cpp), so the
    // rotated row is already marked before the step task exists.
    EXPECT_EQ(input.input_params.linear_state_validity_mask,
              std::vector<int64_t>({1}));
  }
  run(input, {1});
  // The restore ran inside step_for_schedule_overlap, before the batch's
  // execute observed the input: the execute-entry mask is already marked.
  EXPECT_EQ(worker_->masks_at_execute_entry(),
            std::vector<std::vector<int64_t>>({{1}}));
  // Mark-only pin: the worker's cache states match a reference where NO slot
  // copy was applied -- any restore-time copy would diverge bitwise.
  verify();
}

TEST_P(NpuGlm5NextPythonRestoreTest, DecodeRestoreTakesTheCopyPath) {
  // The python glm5_next branch of reads_distinct_state requires
  // no_decode(); a decode batch therefore restores in place (slot copy),
  // exactly like a legacy native model.
  ForwardInput input = prepare({2}, {1}, {2050}, BatchForwardType::DECODE);
  run(input, {1});
  verify();
}

TEST_P(NpuGlm5NextPythonRestoreTest, SpecVerifyPrefillRestoreTakesCopyPath) {
  // !params.is_spec_verify is the second gate of the python glm5_next
  // branch: a spec-verify batch typed as chunked prefill must take the copy
  // path even though no_decode() holds.
  ForwardInput input = prepare({2},
                               {1},
                               {0},
                               BatchForwardType::CHUNKED_PREFILL,
                               /*is_spec_verify=*/true);
  run(input, {1});
  verify();
}

INSTANTIATE_TEST_SUITE_P(
    CheckpointLayouts,
    NpuGlm5NextPythonRestoreTest,
    ::testing::Combine(::testing::Values(int64_t{1}, int64_t{3}),
                       ::testing::Bool()),
    [](const ::testing::TestParamInfo<Glm5NextPythonRestoreParameters>& info) {
      return "Stride" + std::to_string(std::get<0>(info.param)) +
             (std::get<1>(info.param) ? "Overlap" : "Prepare");
    });

}  // namespace
}  // namespace xllm
