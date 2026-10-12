/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/parallel_state/context_parallel_topology.h"

#include <gtest/gtest.h>

#include <vector>

#include "core/framework/parallel_state/parallel_args.h"

namespace xllm::parallel_state {
namespace {

TEST(DcpTopologyTest, FinalizesMembershipAndCoordinatesTogether) {
  const std::vector<std::vector<int32_t>> expected_groups{
      {0, 1}, {2, 3}, {4, 5}, {6, 7}};
  for (int32_t global_rank = 0; global_rank < 8; ++global_rank) {
    const DcpTopology topology = build_contiguous_dcp_topology(global_rank,
                                                               /*world_size=*/8,
                                                               /*dp_size=*/1,
                                                               /*dcp_size=*/2);
    EXPECT_EQ(topology.group_ranks, expected_groups);
    EXPECT_EQ(topology.rank, global_rank % 2);
    EXPECT_EQ(topology.group_index, global_rank / 2);
    EXPECT_EQ(topology.group_ranks[topology.group_index][topology.rank],
              global_rank);
  }
}

TEST(DcpTopologyTest, KeepsGroupsInsideEachDpReplica) {
  for (const int32_t dcp_size : {1, 2, 4, 8}) {
    for (int32_t global_rank = 0; global_rank < 16; ++global_rank) {
      const DcpTopology topology =
          build_contiguous_dcp_topology(global_rank,
                                        /*world_size=*/16,
                                        /*dp_size=*/2,
                                        dcp_size);
      EXPECT_EQ(topology.group_ranks.size(), 16 / dcp_size);
      const auto& own_group = topology.group_ranks[topology.group_index];
      ASSERT_EQ(own_group.size(), dcp_size);
      EXPECT_EQ(own_group[topology.rank], global_rank);
      for (const int32_t member : own_group) {
        EXPECT_EQ(member / 8, global_rank / 8);
      }
    }
  }
}

TEST(DcpTopologyTest, RejectsInvalidDimensionsAndRanks) {
  EXPECT_DEATH(build_contiguous_dcp_topology(/*global_rank=*/0,
                                             /*world_size=*/0,
                                             /*dp_size=*/1,
                                             /*dcp_size=*/2),
               "world_size must be positive");
  EXPECT_DEATH(build_contiguous_dcp_topology(/*global_rank=*/0,
                                             /*world_size=*/8,
                                             /*dp_size=*/0,
                                             /*dcp_size=*/2),
               "dp_size must be positive");
  EXPECT_DEATH(build_contiguous_dcp_topology(/*global_rank=*/0,
                                             /*world_size=*/8,
                                             /*dp_size=*/1,
                                             /*dcp_size=*/0),
               "dcp_size must be positive");
  EXPECT_DEATH(build_contiguous_dcp_topology(/*global_rank=*/-1,
                                             /*world_size=*/8,
                                             /*dp_size=*/1,
                                             /*dcp_size=*/2),
               "global_rank must be non-negative");
  EXPECT_DEATH(build_contiguous_dcp_topology(/*global_rank=*/8,
                                             /*world_size=*/8,
                                             /*dp_size=*/1,
                                             /*dcp_size=*/2),
               "global_rank must be in the world");
  EXPECT_DEATH(build_contiguous_dcp_topology(/*global_rank=*/0,
                                             /*world_size=*/8,
                                             /*dp_size=*/3,
                                             /*dcp_size=*/2),
               "world_size must be divisible by dp_size");
  // Dividing the global world is not enough: a group must not cross DP.
  EXPECT_DEATH(build_contiguous_dcp_topology(/*global_rank=*/0,
                                             /*world_size=*/12,
                                             /*dp_size=*/2,
                                             /*dcp_size=*/4),
               "dcp_size must divide the DP-local world size");
}

TEST(ContextParallelTopologyTest, SeparatesTpPcpAndFullDomainDcp) {
  const ContextParallelTopology topology(/*global_rank=*/3,
                                         /*world_size=*/8,
                                         /*dp_size=*/2,
                                         /*pcp_size=*/2,
                                         /*dcp_size=*/4);

  EXPECT_EQ(topology.dp_rank(), 0);
  EXPECT_EQ(topology.tp_size(), 2);
  EXPECT_EQ(topology.tp_rank(), 1);
  EXPECT_EQ(topology.pcp_rank(), 1);
  EXPECT_EQ(topology.dcp_rank(), 3);
  EXPECT_EQ(topology.pcp_group_ranks(), (std::vector<int32_t>{1, 3}));
  EXPECT_EQ(topology.dcp_group_ranks(), (std::vector<int32_t>{0, 1, 2, 3}));
}

TEST(ContextParallelTopologyTest, PartitionsFactorDcpInsidePcpGroup) {
  const ContextParallelTopology first_partition(/*global_rank=*/5,
                                                /*world_size=*/16,
                                                /*dp_size=*/2,
                                                /*pcp_size=*/4,
                                                /*dcp_size=*/2);
  EXPECT_EQ(first_partition.dp_rank(), 0);
  EXPECT_EQ(first_partition.pcp_group_ranks(),
            (std::vector<int32_t>{1, 3, 5, 7}));
  EXPECT_EQ(first_partition.dcp_rank(), 1);
  EXPECT_EQ(first_partition.dcp_group_ranks(), (std::vector<int32_t>{1, 5}));
  EXPECT_EQ(first_partition.dcp_group_ranks()[first_partition.dcp_rank()], 5);

  const ContextParallelTopology second_dp_partition(/*global_rank=*/13,
                                                    /*world_size=*/16,
                                                    /*dp_size=*/2,
                                                    /*pcp_size=*/4,
                                                    /*dcp_size=*/2);
  EXPECT_EQ(second_dp_partition.dp_rank(), 1);
  EXPECT_EQ(second_dp_partition.pcp_group_ranks(),
            (std::vector<int32_t>{9, 11, 13, 15}));
  EXPECT_EQ(second_dp_partition.dcp_rank(), 1);
  EXPECT_EQ(second_dp_partition.dcp_group_ranks(),
            (std::vector<int32_t>{9, 13}));
  EXPECT_EQ(
      second_dp_partition.dcp_group_ranks()[second_dp_partition.dcp_rank()],
      13);
}

TEST(ContextParallelTopologyTest, ParallelArgsFallbackUsesDpLocalDcpRank) {
  constexpr int32_t kWorldSize = 16;
  constexpr int32_t kDpSize = 2;
  constexpr int32_t kCpSize = 4;
  constexpr int32_t kKvSplitSize = 2;

  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    xllm::ParallelArgs args(rank,
                            kWorldSize,
                            kDpSize,
                            kCpSize,
                            nullptr,
                            /*ep_size=*/kWorldSize);
    args.kv_split_size(kKvSplitSize);

    const ContextParallelTopology topology(
        rank, kWorldSize, kDpSize, kCpSize, kKvSplitSize);
    EXPECT_EQ(args.kv_split_rank(), topology.dcp_rank()) << "rank=" << rank;
  }
}

TEST(ContextParallelTopologyTest, ParallelArgsFallbackHandlesTpOnlyDcp) {
  constexpr int32_t kWorldSize = 16;
  constexpr int32_t kDpSize = 2;
  constexpr int32_t kCpSize = 1;
  constexpr int32_t kKvSplitSize = 8;

  const std::vector<int32_t> expected_ranks = {
      0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
  for (int32_t rank = 0; rank < kWorldSize; ++rank) {
    xllm::ParallelArgs args(rank,
                            kWorldSize,
                            kDpSize,
                            kCpSize,
                            nullptr,
                            /*ep_size=*/kWorldSize);
    args.kv_split_size(kKvSplitSize);

    EXPECT_EQ(args.kv_split_rank(), expected_ranks[rank]) << "rank=" << rank;
  }
}

TEST(ContextParallelTopologyTest, SupportsPcpBoundaryDcpLayouts) {
  const ContextParallelTopology single_rank_dcp(/*global_rank=*/5,
                                                /*world_size=*/16,
                                                /*dp_size=*/2,
                                                /*pcp_size=*/4,
                                                /*dcp_size=*/1);
  EXPECT_EQ(single_rank_dcp.dcp_rank(), 0);
  EXPECT_EQ(single_rank_dcp.dcp_group_ranks(), (std::vector<int32_t>{5}));

  const ContextParallelTopology pcp_wide_dcp(/*global_rank=*/5,
                                             /*world_size=*/16,
                                             /*dp_size=*/2,
                                             /*pcp_size=*/4,
                                             /*dcp_size=*/4);
  EXPECT_EQ(pcp_wide_dcp.dcp_rank(), 2);
  EXPECT_EQ(pcp_wide_dcp.dcp_group_ranks(), (std::vector<int32_t>{1, 3, 5, 7}));
}

TEST(ContextParallelTopologyTest, SupportsFullDpLocalDcpLayout) {
  const ContextParallelTopology topology(/*global_rank=*/13,
                                         /*world_size=*/16,
                                         /*dp_size=*/2,
                                         /*pcp_size=*/4,
                                         /*dcp_size=*/8);
  EXPECT_EQ(topology.dp_rank(), 1);
  EXPECT_EQ(topology.dcp_rank(), 5);
  EXPECT_EQ(topology.dcp_group_ranks(),
            (std::vector<int32_t>{8, 9, 10, 11, 12, 13, 14, 15}));
  EXPECT_EQ(topology.dcp_group_ranks()[topology.dcp_rank()], 13);
}

TEST(ContextParallelTopologyTest, RejectsNonFactorDcpLayout) {
  EXPECT_DEATH(ContextParallelTopology(/*global_rank=*/0,
                                       /*world_size=*/8,
                                       /*dp_size=*/1,
                                       /*pcp_size=*/4,
                                       /*dcp_size=*/3),
               "dcp_size must divide pcp_size");
}

TEST(ContextParallelTopologyTest, LocalRankIndexesEveryLegalDcpGroup) {
  const std::vector<int32_t> dcp_sizes{1, 2, 4, 8};
  for (const int32_t dcp_size : dcp_sizes) {
    for (int32_t global_rank = 0; global_rank < 16; ++global_rank) {
      const ContextParallelTopology topology(global_rank,
                                             /*world_size=*/16,
                                             /*dp_size=*/2,
                                             /*pcp_size=*/4,
                                             dcp_size);
      EXPECT_EQ(topology.dcp_group_ranks()[topology.dcp_rank()], global_rank);
      for (const int32_t member_rank : topology.dcp_group_ranks()) {
        EXPECT_EQ(member_rank / 8, topology.dp_rank());
      }
    }
  }
}

TEST(ContextParallelTopologyTest, InterleavedDcpGroupsHaveDistinctIds) {
  const ContextParallelTopology even(/*global_rank=*/0,
                                     /*world_size=*/8,
                                     /*dp_size=*/1,
                                     /*pcp_size=*/8,
                                     /*dcp_size=*/4);
  const ContextParallelTopology odd(/*global_rank=*/1,
                                    /*world_size=*/8,
                                    /*dp_size=*/1,
                                    /*pcp_size=*/8,
                                    /*dcp_size=*/4);
  EXPECT_EQ(even.dcp_group_ranks(), (std::vector<int32_t>{0, 2, 4, 6}));
  EXPECT_EQ(odd.dcp_group_ranks(), (std::vector<int32_t>{1, 3, 5, 7}));
  EXPECT_EQ(even.dcp_group_index(), 0);
  EXPECT_EQ(odd.dcp_group_index(), 1);
  EXPECT_EQ(even.dcp_group_count(), 2);
  EXPECT_EQ(odd.dcp_group_count(), 2);
}

TEST(ContextParallelTopologyTest, GroupIdsBijectivelyIdentifyMemberSets) {
  // Exercise DP boundaries, TP lanes, PCP replicas, singleton and PCP-wide
  // reuse layouts, and DCP spanning the complete DP-local domain.
  for (const int32_t dp_size : {1, 2}) {
    for (const int32_t tp_size : {1, 2}) {
      for (const int32_t pcp_size : {1, 4, 8}) {
        for (const int32_t dcp_size : {1, 2, 4, 8, 16}) {
          const bool partitions_pcp = pcp_size % dcp_size == 0;
          if (!partitions_pcp && dcp_size != pcp_size * tp_size) {
            continue;
          }
          const int32_t world_size = dp_size * pcp_size * tp_size;
          const int32_t group_count = world_size / dcp_size;
          std::vector<std::vector<int32_t>> groups(group_count);
          std::vector<int32_t> member_counts(group_count, 0);
          for (int32_t rank = 0; rank < world_size; ++rank) {
            SCOPED_TRACE(::testing::Message()
                         << "dp=" << dp_size << " tp=" << tp_size
                         << " pcp=" << pcp_size << " dcp=" << dcp_size
                         << " rank=" << rank);
            const ContextParallelTopology topology(
                rank, world_size, dp_size, pcp_size, dcp_size);
            ASSERT_EQ(topology.dcp_group_count(), group_count);
            const int32_t index = topology.dcp_group_index();
            ASSERT_GE(index, 0);
            ASSERT_LT(index, group_count);
            // The first member records the group's identity. Every subsequent
            // member must agree, so no ID can alias a different member set.
            if (groups[index].empty()) {
              groups[index] = topology.dcp_group_ranks();
            }
            EXPECT_EQ(groups[index], topology.dcp_group_ranks());
            ++member_counts[index];
            for (const int32_t member : topology.dcp_group_ranks()) {
              const ContextParallelTopology peer(
                  member, world_size, dp_size, pcp_size, dcp_size);
              EXPECT_EQ(peer.dcp_group_index(), index);
            }
          }
          for (const int32_t count : member_counts) {
            EXPECT_EQ(count, dcp_size);
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace xllm::parallel_state
