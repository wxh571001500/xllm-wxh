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

#include <gtest/gtest.h>

#include "core/framework/parallel_state/collective_communicator.h"

namespace xllm {
namespace {

TEST(CollectiveCommunicatorPolicyTest,
     PreservesLegacyKvSplitRankWithoutTopology) {
  for (int32_t global_rank = 0; global_rank < 8; ++global_rank) {
    ParallelArgs args(global_rank, /*world_size=*/8, /*process_group=*/nullptr);
    args.kv_split_size(2);
    EXPECT_FALSE(args.dcp_topology_.has_value());
    EXPECT_EQ(args.kv_split_rank(), global_rank / 4);
  }
}

TEST(CollectiveCommunicatorPolicyTest,
     PreservesNativeDcpGroupRankWithoutTopology) {
  ProcessGroup dcp_group(
      /*rank=*/1, /*world_size=*/2, torch::Device(torch::kCPU));
  ParallelArgs args(/*rank=*/1, /*world_size=*/8, /*process_group=*/nullptr);
  args.kv_split_size(2);
  args.dcp_group_ = &dcp_group;
  EXPECT_FALSE(args.dcp_topology_.has_value());
  EXPECT_EQ(args.kv_split_rank(), 1);
}

TEST(CollectiveCommunicatorPolicyTest,
     CopiesFinalizedDcpRankWithoutCommunicator) {
  for (int32_t global_rank = 0; global_rank < 8; ++global_rank) {
    ParallelArgs args(global_rank, /*world_size=*/8, /*process_group=*/nullptr);
    args.kv_split_size(2);
    args.dcp_topology_ = parallel_state::build_contiguous_dcp_topology(
        global_rank, /*world_size=*/8, /*dp_size=*/1, /*dcp_size=*/2);

    // Worker and ModelContext copies must agree before any Python group exists.
    const ParallelArgs model_args = args;
    ASSERT_TRUE(model_args.dcp_topology_.has_value());
    EXPECT_EQ(model_args.dcp_group_, nullptr);
    EXPECT_EQ(model_args.rank(), global_rank);
    EXPECT_EQ(model_args.world_size(), 8);
    EXPECT_EQ(model_args.kv_split_rank(), global_rank % 2);
    const auto& topology = *model_args.dcp_topology_;
    EXPECT_EQ(
        topology.group_ranks[topology.group_index][model_args.kv_split_rank()],
        global_rank);
  }
}

TEST(CollectiveCommunicatorPolicyTest, ReusesOnlyEquivalentTpGroup) {
  EXPECT_TRUE(can_reuse_tp_group_for_moe(
      /*dp_size=*/1, /*tp_size=*/4, /*moe_tp_size=*/4));
  EXPECT_FALSE(can_reuse_tp_group_for_moe(
      /*dp_size=*/1, /*tp_size=*/2, /*moe_tp_size=*/4));
  EXPECT_FALSE(can_reuse_tp_group_for_moe(
      /*dp_size=*/2, /*tp_size=*/2, /*moe_tp_size=*/4));
}

TEST(CollectiveCommunicatorPolicyTest, MoETpGroupIdentityFollowsReusePolicy) {
  // Bare ProcessGroup instances carry no backend communicator: construction
  // and destruction are safe while the pg_ backend handle stays null.
  ProcessGroup tp_group(
      /*rank=*/0, /*world_size=*/4, torch::Device(torch::kCPU));
  ProcessGroup world_group(
      /*rank=*/0, /*world_size=*/4, torch::Device(torch::kCPU));
  ASSERT_NE(&tp_group, &world_group);

  // DP1/EP1 with an equivalent TP rank set: dense and MoE collectives share
  // the very same TP communicator object.
  EXPECT_EQ(select_moe_tp_group(&tp_group,
                                &world_group,
                                /*dp_size=*/1,
                                /*tp_size=*/4,
                                /*moe_tp_size=*/4),
            &tp_group);

  // DP>1 or a non-equivalent MoE TP size keeps the world process group.
  EXPECT_EQ(select_moe_tp_group(&tp_group,
                                &world_group,
                                /*dp_size=*/2,
                                /*tp_size=*/2,
                                /*moe_tp_size=*/4),
            &world_group);
  EXPECT_EQ(select_moe_tp_group(&tp_group,
                                &world_group,
                                /*dp_size=*/1,
                                /*tp_size=*/2,
                                /*moe_tp_size=*/4),
            &world_group);
  // dp_size > 1 with moe_tp_size == tp_size is the width
  // resolve_moe_tp_size() returns for an opted-in caller whatever dp_size says,
  // so it is what a wrongly-set `moe_tp_scoped_to_cp_cohort` flag would produce
  // here. The selection must still refuse the cohort's TP communicator: the
  // caller-side dp_size == 1 gate (worker_server.cpp) keeps the published flag
  // and the chosen group consistent, and this pins the half that makes a wrong
  // flag harmless instead of world-mixing.
  EXPECT_EQ(select_moe_tp_group(&tp_group,
                                &world_group,
                                /*dp_size=*/2,
                                /*tp_size=*/4,
                                /*moe_tp_size=*/4),
            &world_group);
}

}  // namespace
}  // namespace xllm
