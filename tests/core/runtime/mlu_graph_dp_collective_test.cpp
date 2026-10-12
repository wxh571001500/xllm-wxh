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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/framework/model/causal_vlm.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/runtime/executor.h"
#include "core/runtime/mlu_graph_executor_impl.h"
#include "core/runtime/options.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"

namespace xllm::mlu {
namespace {

constexpr int32_t kWorldSize = 2;
constexpr int32_t kHiddenSize = 16;
constexpr int32_t kSkip = 77;
constexpr std::chrono::seconds kChildTimeout{120};

enum class Scenario : int8_t {
  ALL_SUPPORTED,
  ONE_REJECTED,
  PLAN_MISMATCH,
  EMBEDDING_MISMATCH,
  PARTIAL_HIT,
  VLM_ALL_SUPPORTED,
  VLM_ONE_REJECTED,
  VLM_PARTIAL_HIT,
};

enum class ChildOutcome : int8_t { PASSED, SKIPPED, FAILED };

class CountingModel final : public CausalVLM {
 public:
  explicit CountingModel(const torch::TensorOptions& options)
      : options_(options) {}

  ModelOutput forward(const torch::Tensor& /*tokens*/,
                      const torch::Tensor& /*positions*/,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelInputParams& params) override {
    ++forward_count_;
    const torch::Tensor input =
        params.embedding.input_embedding.defined()
            ? params.embedding.input_embedding
            : torch::zeros({params.meta.num_sequences, kHiddenSize}, options_);
    return ModelOutput(input + 1);
  }

  MMDict encode(const ModelInputParams& /*params*/) override { return {}; }

  torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelInputParams& /*params*/) override {
    return torch::zeros({input_ids.size(0), kHiddenSize}, options_);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& /*selected_idxes*/) override {
    return hidden_states;
  }

  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  torch::Device device() const override { return options_.device(); }
  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {}
  void update_expert_weight(int32_t /*layer_id*/) override {}
  const torch::TensorOptions& options() const override { return options_; }
  int32_t forward_count() const { return forward_count_; }

 private:
  torch::TensorOptions options_;
  int32_t forward_count_ = 0;
};

struct StepInput {
  torch::Tensor tokens;
  torch::Tensor positions;
  ModelInputParams params;
  std::vector<KVCache> kv_caches;
};

StepInput make_input(int32_t rank,
                     const torch::Device& device,
                     int32_t batch_size) {
  const auto ints = torch::TensorOptions().dtype(torch::kInt32).device(device);
  const auto hidden =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  std::vector<int32_t> offsets(batch_size + 1);
  std::iota(offsets.begin(), offsets.end(), 0);

  StepInput input;
  input.tokens = torch::ones({batch_size}, ints);
  input.positions = torch::ones({batch_size}, ints);
  input.params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.params.meta.num_sequences = batch_size;
  input.params.meta.q_max_seq_len = 1;
  input.params.meta.kv_max_seq_len = 1;
  input.params.parallel.dp_global_token_nums = {batch_size, batch_size};
  input.params.parallel.dp_is_decode = {1, 1};
  input.params.attention.host.q_seq_lens = offsets;
  input.params.attention.host.kv_seq_lens = offsets;
  input.params.attention.device.q_seq_lens = torch::tensor(offsets, ints);
  input.params.attention.device.kv_seq_lens = torch::tensor(offsets, ints);
  input.params.attention.device.new_cache_slots =
      torch::arange(batch_size, ints);
  input.params.attention.device.block_tables =
      torch::zeros({batch_size, 513}, ints);
  input.params.embedding.input_embedding = torch::full(
      {batch_size, kHiddenSize}, static_cast<float>(rank + 1), hidden);
  input.kv_caches.resize(batch_size);
  return input;
}

void set_embedding(StepInput& input, int32_t rank, int32_t step) {
  input.params.embedding.input_embedding.fill_(
      static_cast<float>(rank + step + 1));
}

void run_step(MluGraphExecutorImpl& executor, StepInput& input) {
  const torch::Tensor output =
      executor.run(input.tokens, input.positions, input.kv_caches, input.params)
          .hidden_states.clone();
  const torch::Tensor expected =
      input.params.embedding.input_embedding.defined()
          ? input.params.embedding.input_embedding + 1
          : torch::ones(
                {input.tokens.size(0), kHiddenSize},
                input.params.attention.device.q_seq_lens.options().dtype(
                    torch::kBFloat16));
  CHECK(torch::equal(output.to(torch::kCPU), expected.to(torch::kCPU)))
      << "Graph/eager output differs from the current rank input";
}

void run_vlm_step(Executor& executor, StepInput& input) {
  const torch::Tensor output =
      executor
          .forward(input.tokens, input.positions, input.kv_caches, input.params)
          .hidden_states.clone();
  const torch::Tensor expected = input.params.embedding.input_embedding + 1;
  CHECK(torch::equal(output.to(torch::kCPU), expected.to(torch::kCPU)))
      << "VLM graph/eager output differs from the current rank input";
}

void prove_local_vlm_graph(Executor& executor,
                           CountingModel& model,
                           StepInput& input,
                           int32_t rank) {
  run_vlm_step(executor, input);
  CHECK_EQ(model.forward_count(), 1);
  set_embedding(input, rank, /*step=*/1);
  run_vlm_step(executor, input);
  CHECK_EQ(model.forward_count(), 1)
      << "Updated VLM output without a second forward proves local replay";
}

void prove_local_graph(MluGraphExecutorImpl& executor,
                       CountingModel& model,
                       StepInput& input,
                       int32_t rank) {
  run_step(executor, input);
  CHECK_EQ(model.forward_count(), 1);
  set_embedding(input, rank, /*step=*/1);
  run_step(executor, input);
  CHECK_EQ(model.forward_count(), 1)
      << "Updated output without a second model forward proves local replay";
}

void verify_group(ProcessGroup& group,
                  int32_t rank,
                  const torch::Device& device) {
  const auto ints = torch::TensorOptions().dtype(torch::kInt64).device(device);
  const torch::Tensor local = torch::full({1}, rank, ints);
  const torch::Tensor gathered =
      group.allgather_base_sync(local).to(torch::kCPU).reshape({kWorldSize});
  CHECK_EQ(gathered[0].item<int64_t>(), 0);
  CHECK_EQ(gathered[1].item<int64_t>(), 1);
}

int32_t run_child(Scenario scenario, int32_t rank, int32_t port) {
  try {
    if (Platform::device_count() < kWorldSize) {
      return kSkip;
    }
    Device xllm_device(rank);
    xllm_device.set_device();
    const torch::Device device = xllm_device.unwrap();
    auto group =
        create_process_group(rank,
                             kWorldSize,
                             kWorldSize,
                             port,
                             /*trans=*/false,
                             /*host=*/"127.0.0.1",
                             /*group_name=*/"mlu_graph_dp_collective_test",
                             device);
    CHECK(group);
    verify_group(*group, rank, device);

    const auto hidden =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    CountingModel model(hidden);
    ModelArgs args;
    args.model_type("qwen3_5_text")
        .dtype("bfloat16")
        .hidden_size(kHiddenSize)
        .max_position_embeddings(8192);
    runtime::Options options;
    options.block_size(16).dp_size(kWorldSize).world_size(kWorldSize);
    if (scenario == Scenario::VLM_ALL_SUPPORTED ||
        scenario == Scenario::VLM_ONE_REJECTED ||
        scenario == Scenario::VLM_PARTIAL_HIT) {
      ExecutionConfig::get_instance().enable_graph(true);
      options.backend("vlm").enable_graph(true);
      Executor executor(&model, args, device, options);
      StepInput input = make_input(rank, device, /*batch_size=*/3);
      if (scenario == Scenario::VLM_ONE_REJECTED) {
        if (rank == 1) {
          // Keep the VLM pure-decode path, but reject this rank in the graph
          // planner before the DP action is agreed.
          input.params.meta.q_max_seq_len = 0;
        } else {
          prove_local_vlm_graph(executor, model, input, rank);
        }
        executor.set_dp_process_group(group.get());
        set_embedding(input, rank, /*step=*/2);
        run_vlm_step(executor, input);
        CHECK_EQ(model.forward_count(), rank == 0 ? 2 : 1)
            << "VLM ranks must run eager when one rank rejects graph";
        set_embedding(input, rank, /*step=*/3);
        run_vlm_step(executor, input);
        CHECK_EQ(model.forward_count(), rank == 0 ? 3 : 2)
            << "VLM ranks must keep agreeing on eager";
        return 0;
      }
      if (scenario == Scenario::VLM_PARTIAL_HIT && rank == 0) {
        prove_local_vlm_graph(executor, model, input, rank);
      }
      executor.set_dp_process_group(group.get());
      set_embedding(input, rank, /*step=*/2);
      run_vlm_step(executor, input);
      CHECK_EQ(model.forward_count(),
               scenario == Scenario::VLM_PARTIAL_HIT && rank == 0 ? 2 : 1)
          << "VLM ranks must capture together";
      set_embedding(input, rank, /*step=*/3);
      run_vlm_step(executor, input);
      CHECK_EQ(model.forward_count(),
               scenario == Scenario::VLM_PARTIAL_HIT && rank == 0 ? 2 : 1)
          << "VLM ranks must replay together";
      return 0;
    }
    MluGraphExecutorImpl executor(&model, args, device, options);
    StepInput input = make_input(rank, device, /*batch_size=*/3);

    if (scenario == Scenario::ONE_REJECTED) {
      if (rank == 1) {
        input.params.meta.batch_forward_type = BatchForwardType::PREFILL;
      } else {
        prove_local_graph(executor, model, input, rank);
      }
      executor.set_dp_process_group(group.get());
      set_embedding(input, rank, /*step=*/2);
      run_step(executor, input);
      CHECK_EQ(model.forward_count(), rank == 0 ? 2 : 1)
          << "Both ranks must take eager when one rank rejects graph";
      set_embedding(input, rank, /*step=*/3);
      run_step(executor, input);
      CHECK_EQ(model.forward_count(), rank == 0 ? 3 : 2)
          << "Repeated one-sided rejection must remain eager on both ranks";
      return 0;
    }

    if (scenario == Scenario::PLAN_MISMATCH) {
      if (rank == 1) {
        const auto ints =
            torch::TensorOptions().dtype(torch::kInt32).device(device);
        input.params.multi_block_tables.emplace_back(
            torch::zeros({3, 2}, ints));
      }
      // Both plans capture and replay locally before their different
      // multi-table layouts are compared through the real DP group.
      prove_local_graph(executor, model, input, rank);
      executor.set_dp_process_group(group.get());
      set_embedding(input, rank, /*step=*/2);
      run_step(executor, input);
      CHECK_EQ(model.forward_count(), 2)
          << "Both ranks must take eager for incompatible graph packets";
      return 0;
    }

    if (scenario == Scenario::EMBEDDING_MISMATCH) {
      prove_local_graph(executor, model, input, rank);
      executor.set_dp_process_group(group.get());
      if (rank == 1) {
        input.params.embedding.input_embedding = torch::Tensor();
      } else {
        set_embedding(input, rank, /*step=*/2);
      }
      run_step(executor, input);
      CHECK_EQ(model.forward_count(), 2)
          << "Both ranks must run eager for different embedding presence";
      return 0;
    }

    if (scenario == Scenario::PARTIAL_HIT && rank == 0) {
      prove_local_graph(executor, model, input, rank);
    }
    executor.set_dp_process_group(group.get());
    set_embedding(input, rank, /*step=*/2);
    run_step(executor, input);
    CHECK_EQ(model.forward_count(),
             scenario == Scenario::PARTIAL_HIT && rank == 0 ? 2 : 1)
        << "All ranks must capture when any compatible rank misses";
    set_embedding(input, rank, /*step=*/3);
    run_step(executor, input);
    CHECK_EQ(model.forward_count(),
             scenario == Scenario::PARTIAL_HIT && rank == 0 ? 2 : 1)
        << "All ranks must replay after coordinated capture";
    return 0;
  } catch (const std::exception& error) {
    LOG(ERROR) << "DP graph collective rank " << rank
               << " failed: " << error.what();
    return 1;
  }
}

ChildOutcome wait_children(const std::vector<pid_t>& pids) {
  std::array<bool, kWorldSize> finished{};
  int32_t remaining = kWorldSize;
  int32_t skipped = 0;
  bool failed = false;
  const auto deadline = std::chrono::steady_clock::now() + kChildTimeout;
  while (remaining > 0 && !failed &&
         std::chrono::steady_clock::now() < deadline) {
    for (int32_t rank = 0; rank < kWorldSize; ++rank) {
      if (finished[rank]) {
        continue;
      }
      int status = 0;
      const pid_t result = ::waitpid(pids[rank], &status, WNOHANG);
      if (result == 0) {
        continue;
      }
      finished[rank] = true;
      --remaining;
      if (result > 0 && WIFEXITED(status) && WEXITSTATUS(status) == kSkip) {
        ++skipped;
      } else if (result < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        failed = true;
        LOG(ERROR) << "DP graph collective rank " << rank
                   << " exited abnormally, status=" << status;
      }
    }
    if (remaining > 0 && !failed) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    if (finished[rank]) {
      continue;
    }
    failed = true;
    LOG(ERROR) << "Stopping stalled DP graph collective rank " << rank;
    if (::kill(pids[rank], SIGKILL) != 0 && errno != ESRCH) {
      LOG(ERROR) << "Failed to kill rank " << rank << ": "
                 << std::strerror(errno);
    }
    int status = 0;
    ::waitpid(pids[rank], &status, 0);
  }
  if (failed || (skipped != 0 && skipped != kWorldSize)) {
    return ChildOutcome::FAILED;
  }
  return skipped == kWorldSize ? ChildOutcome::SKIPPED : ChildOutcome::PASSED;
}

void run_scenario(Scenario scenario) {
  const int32_t port = net::get_local_free_port();
  ASSERT_GT(port, 0);
  std::vector<pid_t> pids;
  pids.reserve(kWorldSize);
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    const pid_t pid = ::fork();
    if (pid == 0) {
      _exit(run_child(scenario, rank, port));
    }
    if (pid < 0) {
      for (const pid_t child : pids) {
        ::kill(child, SIGKILL);
        int status = 0;
        ::waitpid(child, &status, 0);
      }
      FAIL() << "Failed to fork DP graph collective rank " << rank;
      return;
    }
    pids.emplace_back(pid);
  }
  const ChildOutcome outcome = wait_children(pids);
  if (outcome == ChildOutcome::SKIPPED) {
    GTEST_SKIP() << "Requires two visible MLU devices";
  }
  EXPECT_EQ(outcome, ChildOutcome::PASSED);
}

TEST(MluGraphDpCollectiveTest, AllRanksCaptureThenReplay) {
  run_scenario(Scenario::ALL_SUPPORTED);
}

TEST(MluGraphDpCollectiveTest, OneRankRejectsSoAllRunEager) {
  run_scenario(Scenario::ONE_REJECTED);
}

TEST(MluGraphDpCollectiveTest, DifferentPlansMakeAllRunEager) {
  run_scenario(Scenario::PLAN_MISMATCH);
}

TEST(MluGraphDpCollectiveTest, DifferentEmbeddingPresenceMakesAllRunEager) {
  run_scenario(Scenario::EMBEDDING_MISMATCH);
}

TEST(MluGraphDpCollectiveTest, PartialCacheHitRecapturesThenReplays) {
  run_scenario(Scenario::PARTIAL_HIT);
}

TEST(MluGraphDpCollectiveTest, VlmAllRanksCaptureThenReplay) {
  run_scenario(Scenario::VLM_ALL_SUPPORTED);
}

TEST(MluGraphDpCollectiveTest, VlmOneRankRejectsSoAllRunEager) {
  run_scenario(Scenario::VLM_ONE_REJECTED);
}

TEST(MluGraphDpCollectiveTest, VlmPartialCacheHitRecapturesThenReplays) {
  run_scenario(Scenario::VLM_PARTIAL_HIT);
}

}  // namespace
}  // namespace xllm::mlu
