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

#include <framework/core/device.h>
#include <framework/graphs/MLUGraph.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cerrno>
#include <exception>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/config/eplb_config.h"
#include "framework/parallel_state/process_group.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_sparse_moe_block.h"
#include "layers/mlu/glm5_next/glm5_next_pcp_context.h"
#include "platform/device.h"
#include "platform/model_stream_registry.h"
#include "platform/platform.h"
#include "util/net.h"

namespace xllm::layer {
namespace {

constexpr int32_t kExitCodeSkip = 77;

ModelArgs make_model_args(bool with_shared) {
  ModelArgs args;
  args.model_type() = "glm5_next";
  args.hidden_size() = 256;
  args.moe_intermediate_size() = 256;
  args.n_routed_experts() = 4;
  args.num_experts_per_tok() = 2;
  args.n_shared_experts() = with_shared ? 1 : 0;
  args.n_group() = 1;
  args.topk_group() = 1;
  args.routed_scaling_factor() = 1.0f;
  args.norm_topk_prob() = true;
  args.hidden_act() = "silu";
  args.scoring_func() = "softmax";
  args.topk_method() = "greedy";
  args.swiglu_limit() = 10.0f;
  return args;
}

torch::Tensor make_values(int64_t rows,
                          int64_t columns,
                          float phase,
                          const torch::TensorOptions& options) {
  // Construct on CPU so every rank receives identical full weights and inputs.
  // Positive values avoid cancellation amplifying BF16 shard-rounding error
  // when TP partial sums are compared with a single unsharded GEMM.
  return (torch::arange(rows * columns, torch::kFloat32) * 0.17f + phase)
      .sin()
      .add(1.5f)
      .reshape({rows, columns})
      .mul(0.02f)
      .to(options);
}

StateDict make_weights(const ModelArgs& args,
                       const torch::TensorOptions& options) {
  std::unordered_map<std::string, torch::Tensor> weights;
  const int64_t hidden = args.hidden_size();
  const int64_t intermediate = args.moe_intermediate_size();
  const auto add_expert = [&](const std::string& prefix, float phase) {
    weights.emplace(prefix + "gate_proj.weight",
                    make_values(intermediate, hidden, phase, options));
    weights.emplace(prefix + "up_proj.weight",
                    make_values(intermediate, hidden, phase + 0.7f, options));
    weights.emplace(prefix + "down_proj.weight",
                    make_values(hidden, intermediate, phase + 1.3f, options));
  };
  for (int32_t expert = 0; expert < args.n_routed_experts(); ++expert) {
    add_expert("experts." + std::to_string(expert) + ".",
               static_cast<float>(expert));
  }
  if (args.n_shared_experts() > 0) {
    add_expert("shared_experts.", 5.0f);
  }
  weights.emplace("gate.weight",
                  make_values(args.n_routed_experts(), hidden, 0.0f, options));
  return StateDict(std::move(weights));
}

ParallelArgs make_parallel_args(int32_t rank,
                                int32_t world_size,
                                bool use_ep,
                                ProcessGroup* collective,
                                ProcessGroup* single_rank) {
  ParallelArgs args(rank, world_size, collective);
  args.tp_group_ = collective;
  args.single_rank_group_ = single_rank;
  args.moe_ep_group_ = use_ep ? collective : single_rank;
  args.moe_tp_group_ = use_ep ? single_rank : collective;
  args.ep_size_ = use_ep ? world_size : 1;
  return args;
}

void check_output(const torch::Tensor& actual, const torch::Tensor& expected) {
  CHECK_EQ(actual.sizes(), expected.sizes());
  CHECK(torch::isfinite(actual).all().item<bool>());
  CHECK_GT(expected.abs().max().item<float>(), 1e-4f);
  CHECK(torch::allclose(actual, expected, /*rtol=*/0.02, /*atol=*/2e-4))
      << "Selected MoE differs from the unsharded reference; maximum error: "
      << (actual - expected).abs().max().item<float>();
}

int32_t run_rank(int32_t rank,
                 int32_t world_size,
                 int32_t port,
                 bool use_ep,
                 bool with_shared,
                 bool graph_replay,
                 bool use_cp) {
  // Bound a failed collective, including a peer that exits during setup.
  ::alarm(180);
  if (Platform::device_count() < world_size) {
    return kExitCodeSkip;
  }
  torch::InferenceMode inference_mode;
  Device device(rank);
  device.set_device();
  EPLBConfig::get_instance().expert_parallel_degree(1);
  auto collective = create_process_group(rank,
                                         world_size,
                                         world_size,
                                         port,
                                         /*trans=*/false,
                                         "127.0.0.1",
                                         "selected_moe_collective_test",
                                         device.unwrap());
  ProcessGroup single_rank(/*rank=*/0, /*world_size=*/1, device.unwrap());
  ParallelArgs parallel_args = make_parallel_args(
      rank, world_size, use_ep, collective.get(), &single_rank);
  if (use_cp) {
    parallel_args.tp_group_ = &single_rank;
    parallel_args.cp_group_ = collective.get();
  }
  const ParallelArgs reference_args = make_parallel_args(
      /*rank=*/0,
      /*world_size=*/1,
      /*use_ep=*/false,
      &single_rank,
      &single_rank);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  const ModelArgs model_args = make_model_args(with_shared);
  ModelStreamRegistry streams(device.unwrap());
  DeepseekV4SparseMoEBlock block(
      model_args,
      QuantArgs(),
      parallel_args,
      options,
      /*use_hash=*/false,
      streams.get(ExecutionStreamRole::COMMUNICATION),
      streams.get(ExecutionStreamRole::AUXILIARY_COMPUTE));
  // No sharding or collective in this reference: all experts run on each rank.
  FusedMoE reference(
      model_args,
      FusedMoEArgs{.is_gated = true, .enable_result_reduction = false},
      QuantArgs(),
      reference_args,
      options,
      streams.get(ExecutionStreamRole::COMMUNICATION),
      streams.get(ExecutionStreamRole::AUXILIARY_COMPUTE));
  const StateDict weights = make_weights(model_args, options);
  block->load_state_dict(weights);
  reference->load_state_dict(weights);

  if (use_cp) {
    for (const std::vector<int32_t>& lengths :
         {std::vector<int32_t>{128}, {129}, {65, 129}, {1, 0, 7}}) {
      const auto context = glm5_next_pcp::build_context(
          lengths, rank, collective.get(), /*chunk_size=*/64, device.unwrap());
      const torch::Tensor full =
          make_values(context.geometry.total_tokens, 16, 0.5f, options);
      const torch::Tensor local = glm5_next_pcp::shard_rows(full, context);
      CHECK(torch::equal(glm5_next_pcp::gather_restore(local, context), full));
    }
    for (const std::vector<int32_t>& lengths :
         {std::vector<int32_t>{3, 5}, {0, 5}, {5, 0}}) {
      const int64_t offset = rank == 0 ? 0 : lengths[0];
      const torch::Tensor full =
          make_values(
              lengths[0] + lengths[1], model_args.hidden_size(), 0.5f, options)
              .mul(10.0f);
      torch::Tensor local = full.narrow(0, offset, lengths[rank]);
      const torch::Tensor actual =
          block->forward_cp(local, std::nullopt, lengths, collective.get());
      if (local.size(0) == 0) {
        CHECK_EQ(actual.sizes(), local.sizes());
        continue;
      }
      const auto route = reference->prep_route(local, std::nullopt);
      torch::Tensor expected = reference->forward_experts(
          local, /*enable_all2all_communication=*/false, route);
      const torch::Tensor shared = reference->forward_shared(local);
      if (shared.defined()) {
        expected.add_(shared);
      }
      check_output(actual, expected);
    }
    torch_mlu::synchronize();
    return 0;
  }

  torch::Tensor hidden =
      make_values(/*rows=*/8, model_args.hidden_size(), 0.5f, options)
          .mul(10.0f)
          .reshape({2, 4, model_args.hidden_size()});
  torch::Tensor ids =
      torch::tensor({0, 2}, options.dtype(torch::kInt32)).repeat({8, 1});
  torch::Tensor routing_weights =
      torch::tensor({0.25f, 0.75f}, options.dtype(torch::kFloat32))
          .repeat({8, 1});
  const auto selected = [&]() {
    return block->forward_selected(
        hidden, routing_weights, ids, ModelInputParams());
  };
  const auto expected = [&]() {
    const torch::Tensor rows = hidden.reshape({-1, model_args.hidden_size()});
    FusedMoEImpl::RouteInfo route;
    route.expert_id = ids;
    route.reduce_weight = routing_weights;
    torch::Tensor output = reference->forward_experts(
        rows, /*enable_all2all_communication=*/false, route);
    const torch::Tensor shared = reference->forward_shared(rows);
    if (shared.defined()) {
      output.add_(shared);
    }
    return output.reshape(hidden.sizes());
  };

  // Warm the real communicator and kernels before capture on a nondefault
  // stream.
  for (int32_t iteration = 0; iteration < 3; ++iteration) {
    check_output(selected(), expected());
  }
  torch_mlu::synchronize();
  if (graph_replay) {
    auto capture_stream = device.get_stream_from_pool();
    torch_mlu::MLUGraph graph;
    torch::Tensor captured;
    {
      auto guard = capture_stream->set_stream_guard();
      graph.capture_begin();
      captured = selected();
      graph.capture_end();
    }
    for (int32_t iteration = 0; iteration < 3; ++iteration) {
      hidden.mul_(0.8f);
      ids.fill_(iteration % model_args.n_routed_experts());
      graph.replay();
      check_output(captured, expected());
    }
    torch_mlu::synchronize();
  }
  return 0;
}

void run_collective_test(int32_t world_size,
                         bool use_ep,
                         bool with_shared,
                         bool graph_replay,
                         bool use_cp = false) {
  const int32_t port = net::get_local_free_port();
  std::vector<pid_t> children;
  children.reserve(world_size);
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const pid_t child = ::fork();
    CHECK_GE(child, 0) << "Cannot fork selected MoE rank " << rank;
    if (child == 0) {
      int32_t result = 1;
      // A child process boundary must report setup or device failures to GTest.
      try {
        result = run_rank(
            rank, world_size, port, use_ep, with_shared, graph_replay, use_cp);
      } catch (const std::exception& error) {
        LOG(ERROR) << "Selected MoE rank " << rank << ": " << error.what();
      }
      ::_exit(result);
    }
    children.emplace_back(child);
  }
  bool skipped = false;
  for (pid_t child : children) {
    int status = 0;
    pid_t waited;
    do {
      waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    EXPECT_EQ(waited, child);
    EXPECT_TRUE(WIFEXITED(status))
        << "Child " << child << ", status " << status;
    if (!WIFEXITED(status)) {
      continue;
    }
    const int32_t exit_code = WEXITSTATUS(status);
    skipped = skipped || exit_code == kExitCodeSkip;
    EXPECT_TRUE(exit_code == 0 || exit_code == kExitCodeSkip)
        << "Child " << child << " exited with " << exit_code;
  }
  if (skipped && !::testing::Test::HasFailure()) {
    GTEST_SKIP() << "Requires " << world_size << " visible MLU devices";
  }
}

TEST(DeepseekV4SparseMoECollectiveTest, EpSharedMatchesUnshardedReference) {
  run_collective_test(/*world_size=*/2,
                      /*use_ep=*/true,
                      /*with_shared=*/true,
                      /*graph_replay=*/false);
}

TEST(DeepseekV4SparseMoECollectiveTest, CpSharedPreservesRaggedAndEmptyRanks) {
  run_collective_test(/*world_size=*/2,
                      /*use_ep=*/true,
                      /*with_shared=*/true,
                      /*graph_replay=*/false,
                      /*use_cp=*/true);
}

TEST(DeepseekV4SparseMoECollectiveTest, CpWithoutSharedPreservesRaggedRanks) {
  run_collective_test(/*world_size=*/2,
                      /*use_ep=*/true,
                      /*with_shared=*/false,
                      /*graph_replay=*/false,
                      /*use_cp=*/true);
}

TEST(DeepseekV4SparseMoECollectiveTest, TpSharedMatchesUnshardedReference) {
  run_collective_test(/*world_size=*/2,
                      /*use_ep=*/false,
                      /*with_shared=*/true,
                      /*graph_replay=*/false);
}

TEST(DeepseekV4SparseMoECollectiveTest,
     EpWithoutSharedMatchesUnshardedReference) {
  run_collective_test(/*world_size=*/2,
                      /*use_ep=*/true,
                      /*with_shared=*/false,
                      /*graph_replay=*/false);
}

TEST(DeepseekV4SparseMoECollectiveTest,
     SingleRankWithoutSharedPreservesOutput) {
  run_collective_test(/*world_size=*/1,
                      /*use_ep=*/false,
                      /*with_shared=*/false,
                      /*graph_replay=*/false);
}

TEST(DeepseekV4SparseMoECollectiveTest,
     EpSharedGraphUsesUpdatedInputsAndRoutes) {
  run_collective_test(/*world_size=*/2,
                      /*use_ep=*/true,
                      /*with_shared=*/true,
                      /*graph_replay=*/true);
}

}  // namespace
}  // namespace xllm::layer
