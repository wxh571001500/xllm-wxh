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

#include "framework/kv_cache_transfer/prefetch_tail_probe.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace xllm {
namespace {

// Unit u holds gated_blocks_per_unit[u] gated blocks whose dst_block_id is
// u * 100 + block, so the flattened order can be checked.
StoragePrefetchRequest make_request(
    const std::vector<size_t>& gated_blocks_per_unit) {
  StoragePrefetchRequest request;
  request.units.reserve(gated_blocks_per_unit.size());
  for (size_t unit = 0; unit < gated_blocks_per_unit.size(); ++unit) {
    PrefetchUnit prefetch_unit;
    prefetch_unit.gated_blocks.reserve(gated_blocks_per_unit[unit]);
    for (size_t block = 0; block < gated_blocks_per_unit[unit]; ++block) {
      prefetch_unit.gated_blocks.emplace_back(
          /*src_block_id=*/-1,
          /*dst_block_id=*/static_cast<int32_t>(unit * 100 + block));
    }
    prefetch_unit.non_gated_blocks.emplace_back(/*src_block_id=*/-1,
                                                /*dst_block_id=*/-1);
    request.units.emplace_back(std::move(prefetch_unit));
  }
  return request;
}

std::vector<int32_t> dst_block_ids(PrefetchTailProbe& probe) {
  std::vector<int32_t> ids;
  ids.reserve(probe.transfers().size());
  for (const BlockTransferInfo& info : probe.transfers()) {
    ids.emplace_back(info.dst_block_id);
  }
  return ids;
}

TEST(PrefetchTailProbeTest, TailBeginsAtFirstGatedMiss) {
  EXPECT_EQ(PrefetchTailProbe::tail_begin(
                /*unit_begin=*/8, {1, 1, 0, 1}, /*unit_count=*/4),
            10U);
  EXPECT_EQ(PrefetchTailProbe::tail_begin(
                /*unit_begin=*/8, {0, 1, 1, 1}, /*unit_count=*/4),
            8U);
}

TEST(PrefetchTailProbeTest, FullyHitBatchMovesTailPastBatch) {
  EXPECT_EQ(PrefetchTailProbe::tail_begin(
                /*unit_begin=*/8, {1, 1, 1, 1}, /*unit_count=*/4),
            12U);
}

TEST(PrefetchTailProbeTest, PaddingOfShortLastBatchIsIgnored) {
  // Only the first two entries are real units; the zero padding is no miss.
  EXPECT_EQ(PrefetchTailProbe::tail_begin(
                /*unit_begin=*/4, {1, 1, 0, 0}, /*unit_count=*/2),
            6U);
}

TEST(PrefetchTailProbeTest, FlattensGatedBlocksOfTailUnitsOnly) {
  const StoragePrefetchRequest request = make_request({1, 2, 3, 1});
  PrefetchTailProbe probe(request, /*begin_unit=*/1);

  EXPECT_EQ(probe.unit_count(), 3U);
  EXPECT_EQ(dst_block_ids(probe),
            std::vector<int32_t>({100, 101, 200, 201, 202, 300}));
}

TEST(PrefetchTailProbeTest, UnitIsPresentOnlyWhenAllGatedBlocksArePresent) {
  const StoragePrefetchRequest request = make_request({1, 2, 3, 1});
  PrefetchTailProbe probe(request, /*begin_unit=*/1);

  // Unit 1 fully present, unit 2 misses its last block, unit 3 present.
  EXPECT_EQ(probe.count_present_units({1, 1, 1, 1, 0, 1}), 2U);
  EXPECT_EQ(probe.count_present_units({0, 0, 0, 0, 0, 0}), 0U);
  EXPECT_EQ(probe.count_present_units({1, 1, 1, 1, 1, 1}), 3U);
}

TEST(PrefetchTailProbeTest, TailAtOrPastEndIsEmpty) {
  const StoragePrefetchRequest request = make_request({1, 2});
  for (size_t begin_unit : {2U, 5U}) {
    SCOPED_TRACE(begin_unit);
    PrefetchTailProbe probe(request, begin_unit);
    EXPECT_EQ(probe.unit_count(), 0U);
    EXPECT_TRUE(probe.transfers().empty());
    EXPECT_EQ(probe.count_present_units({}), 0U);
  }
}

TEST(PrefetchTailProbeTest, WholeRequestIsProbedWhenNothingWasFetched) {
  const StoragePrefetchRequest request = make_request({2, 1});
  PrefetchTailProbe probe(request, /*begin_unit=*/0);

  EXPECT_EQ(probe.unit_count(), 2U);
  EXPECT_EQ(dst_block_ids(probe), std::vector<int32_t>({0, 1, 100}));
  EXPECT_EQ(probe.count_present_units({1, 0, 1}), 1U);
}

TEST(PrefetchTailProbeTest, RejectsPresenceOfWrongLengthWithoutTerminating) {
  const StoragePrefetchRequest request = make_request({2, 1});
  PrefetchTailProbe probe(request, /*begin_unit=*/0);
  EXPECT_FALSE(probe.count_present_units({1, 1}).has_value());
  EXPECT_FALSE(probe.count_present_units({1, 1, 1, 1}).has_value());
  EXPECT_EQ(probe.count_present_units({1, 1, 1}), 2U);
}

TEST(PrefetchTailProbeTest, TailIncludesReadHitsAfterPrefixBreak) {
  const StoragePrefetchRequest request = make_request({1, 1, 1, 1, 1});
  const size_t begin = PrefetchTailProbe::tail_begin(
      /*unit_begin=*/0, {1, 1, 0, 1}, /*unit_count=*/4);
  PrefetchTailProbe probe(request, begin);
  EXPECT_EQ(dst_block_ids(probe), std::vector<int32_t>({200, 300, 400}));
  EXPECT_EQ(probe.count_present_units({0, 1, 1}), 2U);
}

}  // namespace
}  // namespace xllm
