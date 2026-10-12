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

// The settled m7 PD pair declares DIFFERENT KV split widths: the PREFILL runs
// kv_split_size = 2 (owner-sharded KV under CP, the d20ec21a1 route) while the
// decode declares kv_split_size = 1 (cp=1, the flag's default and the settled
// decode width). For the same prompt the destination therefore holds
// kv_split_size times as many blocks as one source logical block covers, and
// the source's block mapping has to interleave.
//
// These tests drive the real mapping decision at the existing unit-test seam:
// BatchInputBuilder::build_step_transfer_info (private; reached through the
// friend name it already declares), plus filter_kv_split_infos, the width
// classifier plan_kv_split_widths, the production
// has_rank_preserving_kv_groups predicate, and the GetInstanceInfo reader
// instance_info_from_proto. No hardware, no production change.
//
// The pre-fix gate (disagg_pd_scheduler.cpp) derived rank_local_mapping from
// the source's own width and the response's group TYPES only, so the
// heterogeneous 2 -> 1 pair wrongly took the 1:1 path: every owner stripe
// addressed the same destination block and the stripe-mate blocks were never
// written. The fix keys the decision on the widths both sides declare.

#include <gtest/gtest.h>
#include <unistd.h>

#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "disagg_pd.pb.h"
#include "framework/batch/batch_input_builder.h"
#include "framework/block/block_manager_impl.h"
#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#include "framework/kv_cache_transfer/reshard_planner.h"
#include "framework/request/incremental_decoder.h"
#include "framework/request/sequence.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/xservice_client.h"
#include "scheduler/disagg_pd_scheduler.h"
#include "tests/core/framework/kv_cache_transfer/scoped_environment_variable.h"
#include "xservice.pb.h"

namespace xllm {

// The production method is private and befriends this exact class name
// (batch_input_builder.h), so it must live at namespace scope, not in the
// anonymous namespace below. tests/core/framework/batch/batch_test.cpp defines
// its own copy for its own binary; this TU is the only definition in this one.
class BatchInputBuilderTestPeer final {
 public:
  static TransferKVInfo build_step_transfer_info(
      const TransferKVInfo& full_info,
      Sequence* sequence,
      uint32_t seq_len,
      uint32_t kv_split_size) {
    return BatchInputBuilder::build_step_transfer_info(
        full_info, sequence, seq_len, kv_split_size);
  }
};

namespace {

// The mapping decision, bound to the production composition both dispatch
// overrides run (resolve_kv_split_plan, disagg_pd_scheduler.cpp /
// pd_ooc_scheduler.cpp): the declared widths and the source's write mode
// decide whether the rank-local shape is available, and the response's group
// types decide whether it is appropriate for this cache layout. The instances
// these tests model run the default replicated write mode.
//
// The pre-fix gate this replaces was
//   rank_local_mapping = instance_info_.kv_split_size > 1 &&
//                        has_rank_preserving_kv_groups(resp)
// -- a group-type proxy that never looked at the destination's declared width,
// so it was true for both a split and an unsplit destination.
bool rank_local_mapping_for_test(int32_t src_kv_split_size,
                                 int32_t dst_kv_split_size) {
  proto::DisaggResponse response;
  response.add_groups()->set_group_id(cache_group_id(BlockType::KV));
  response.add_groups()->set_group_id(cache_group_id(BlockType::LINEAR));
  InstanceInfo remote;
  remote.kv_split_size = dst_kv_split_size;
  return resolve_kv_split_plan(
             src_kv_split_size, kCpIndexWriteModeReplicated, remote, &response)
      .rank_local_mapping;
}

// Source logical block size, in the units the source block manager hands out:
// one logical block spans block_size * kv_split_size tokens, so one logical
// block id covers this many tokens and the destination (kv_split == 1) needs
// kSourceKvSplitSize blocks for each of them.
constexpr uint32_t kSourceLogicalBlockSize = 64;
constexpr uint32_t kSourceLogicalBlockCount = 4;
constexpr uint32_t kSeqLen = kSourceLogicalBlockSize * kSourceLogicalBlockCount;
constexpr int32_t kSourceKvSplitSize = 2;
constexpr uint64_t kFirstDestinationBlockId = 100;
// A destination that declares kv_split == 1 stores kSourceKvSplitSize times as
// many blocks for the same prompt.
constexpr uint32_t kDestinationBlockCount =
    kSourceLogicalBlockCount * static_cast<uint32_t>(kSourceKvSplitSize);

Sequence make_sequence(std::vector<Block> blocks) {
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker;

  SequenceParams seq_params;
  seq_params.seq_capacity = kSeqLen + 8;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = false;

  IncrementalDecoder decoder(/*prompt=*/"",
                             /*num_prompt_tokens=*/1,
                             /*echo=*/false,
                             /*skip_special_tokens=*/true);
  Sequence sequence(/*index=*/0,
                    /*prompt_token_ids=*/{1},
                    /*input_embedding=*/torch::Tensor(),
                    /*mm_data=*/MMData(),
                    decoder,
                    seq_params);
  sequence.add_blocks(BlockType::KV, std::move(blocks));
  return sequence;
}

// The destination hands back every block it allocated for the prompt: the
// source's own per-rank logical block count times kv_split_size when it
// declares kv_split == 1, and the logical block count itself when it declares
// the same width as the source.
std::vector<uint64_t> destination_block_ids(uint32_t block_count) {
  std::vector<uint64_t> ids;
  ids.reserve(block_count);
  for (uint32_t i = 0; i < block_count; ++i) {
    ids.emplace_back(kFirstDestinationBlockId + i);
  }
  return ids;
}

TransferKVInfo full_transfer_info(const std::vector<uint64_t>& remote_ids,
                                  bool rank_local_mapping) {
  TransferKVInfo info;
  info.request_id = "kv-split-width-mismatch";
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.remote_ids = remote_ids;
  mapping.remote_shared_num = 0;
  info.mappings.emplace_back(std::move(mapping));
  info.rank_local_mapping = rank_local_mapping;
  return info;
}

// One source shard rank's view of the transfer, exactly as the push path builds
// it: build_step_transfer_info, then filter_kv_split_infos when the source
// declares kv_split_size > 1 (kv_cache_transfer.cpp).
std::vector<uint64_t> source_rank_destination_blocks(int32_t kv_split_rank,
                                                     bool rank_local_mapping,
                                                     uint32_t dst_block_count) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(kSourceLogicalBlockSize);
  BlockManagerImpl manager(options);
  Sequence sequence = make_sequence(manager.allocate(kSourceLogicalBlockCount));

  const TransferKVInfo step =
      BatchInputBuilderTestPeer::build_step_transfer_info(
          full_transfer_info(destination_block_ids(dst_block_count),
                             rank_local_mapping),
          &sequence,
          /*seq_len=*/kSeqLen,
          /*kv_split_size=*/static_cast<uint32_t>(kSourceKvSplitSize));

  const std::vector<TransferKVInfo> filtered = filter_kv_split_infos(
      kv_split_rank, kSourceKvSplitSize, std::vector<TransferKVInfo>{step});
  if (filtered.empty()) {
    return {};
  }
  const KVTransferMapping& mapping = filtered.front().mappings.front();
  EXPECT_EQ(mapping.local_ids.size(), mapping.remote_ids.size())
      << "kv_split_rank=" << kv_split_rank
      << ", rank_local_mapping=" << rank_local_mapping
      << ": local=" << mapping.local_ids.size()
      << ", remote=" << mapping.remote_ids.size();
  return mapping.remote_ids;
}

struct Coverage {
  std::vector<std::vector<uint64_t>> per_shard_remote_ids;
  std::set<uint64_t> covered;
  std::map<uint64_t, int32_t> writers_per_block;
  // Sum over shard ranks of the source logical blocks that got a destination
  // block in that rank's step mapping. It counts (rank, block) pairs, so a
  // destination block written by several ranks is counted once per rank.
  size_t source_blocks_mapped = 0;
};

Coverage measure_coverage(bool rank_local_mapping, uint32_t dst_block_count) {
  Coverage coverage;
  for (int32_t kv_split_rank = 0; kv_split_rank < kSourceKvSplitSize;
       ++kv_split_rank) {
    const std::vector<uint64_t> remote_ids = source_rank_destination_blocks(
        kv_split_rank, rank_local_mapping, dst_block_count);
    coverage.source_blocks_mapped += remote_ids.size();
    coverage.per_shard_remote_ids.emplace_back(remote_ids);
    for (const uint64_t remote_id : remote_ids) {
      coverage.covered.emplace(remote_id);
      ++coverage.writers_per_block[remote_id];
    }
  }
  return coverage;
}

std::string describe(const Coverage& coverage, bool rank_local_mapping) {
  std::string text = "rank_local_mapping=";
  text += rank_local_mapping ? "true" : "false";
  text += ", covered=" + std::to_string(coverage.covered.size()) +
          ", mapped=" + std::to_string(coverage.source_blocks_mapped);
  for (size_t rank = 0; rank < coverage.per_shard_remote_ids.size(); ++rank) {
    text += "\n  shard " + std::to_string(rank) + " -> {";
    for (const uint64_t id : coverage.per_shard_remote_ids[rank]) {
      text += std::to_string(id) + " ";
    }
    text += "}";
  }
  text += "\n  writers per addressed block: {";
  for (const auto& [id, writers] : coverage.writers_per_block) {
    text += std::to_string(id) + ":" + std::to_string(writers) + " ";
  }
  text += "}";
  return text;
}

}  // namespace

// The response shape that made the pre-fix gate width-blind: the decode's
// AddNewRequests response carries KV (and LINEAR, for the KDA layers) groups
// only, so has_rank_preserving_kv_groups is true and
// `kv_split_size > 1 && predicate` was true for ANY destination width. No
// SWA / C4 / C128 group exists for glm5_next: composite manager types are built
// only for deepseek_v4.
TEST(KvSplitWidthMismatchTest, SettledResponseShapeIsRankPreserving) {
  proto::DisaggResponse response;
  response.add_groups()->set_group_id(cache_group_id(BlockType::KV));
  response.add_groups()->set_group_id(cache_group_id(BlockType::LINEAR));

  EXPECT_TRUE(has_rank_preserving_kv_groups(response));
}

// ACCEPTANCE. The settled m7 pair: source declares kv_split_size = 2 (cp8/tp2
// prefill), destination declares 1 (cp1 decode).
//
// With the pre-fix gate the derived rank_local_mapping is true, so the builder
// emits one destination block per source logical block and
// filter_kv_split_infos returns the mapping untouched. The two source shard
// ranks of one CP cohort receive the identical response and therefore emit the
// identical destination blocks: half the blocks are addressed by both shards
// and the stripe-mate blocks are never written. Nothing rejects this:
// validate_transfer_mappings only compares the two lists' lengths, and the
// rank-local short-circuit makes even that a length-only check.
//
// Red before the fix (both shards address the same blocks, two writers per
// addressed block), green after it: the gate must select the stride path this
// pair needs.
TEST(KvSplitWidthMismatchTest,
     SettledPairCoversEveryDestinationBlockExactlyOnce) {
  // The decision the settled pair must derive: destination width 1, so the
  // destination is unsplit and the stride path is required.
  const bool rank_local_mapping =
      rank_local_mapping_for_test(kSourceKvSplitSize,
                                  /*dst_kv_split_size=*/1);
  EXPECT_FALSE(rank_local_mapping)
      << "the destination declares kv_split_size=1 while the source declares "
      << kSourceKvSplitSize << ": the stride path is required";
  const Coverage coverage =
      measure_coverage(rank_local_mapping, kDestinationBlockCount);
  std::cout << "SETTLED " << describe(coverage, rank_local_mapping)
            << std::endl;

  SCOPED_TRACE(describe(coverage, rank_local_mapping));
  EXPECT_EQ(coverage.covered.size(), kDestinationBlockCount);
  for (const auto& [remote_id, writers] : coverage.writers_per_block) {
    EXPECT_EQ(writers, 1) << "destination block " << remote_id << " has "
                          << writers << " writers";
  }
}

// The same pair with rank_local_mapping = false selects remote_stride =
// kv_split_size and the shard filter. Every destination block is then addressed
// exactly once across the two source shards, so the mapping machinery already
// handles source kv_split 2 -> destination kv_split 1 and the defect is the
// gate, not the mapping.
TEST(KvSplitWidthMismatchTest,
     StridePathCoversEveryDestinationBlockExactlyOnce) {
  const Coverage coverage =
      measure_coverage(/*rank_local_mapping=*/false, kDestinationBlockCount);
  std::cout << "STRIDE " << describe(coverage, /*rank_local_mapping=*/false)
            << std::endl;

  SCOPED_TRACE(describe(coverage, /*rank_local_mapping=*/false));
  EXPECT_EQ(coverage.covered.size(), kDestinationBlockCount);
  for (const auto& [remote_id, writers] : coverage.writers_per_block) {
    EXPECT_EQ(writers, 1) << "destination block " << remote_id << " has "
                          << writers << " writers";
  }
}

// REGRESSION GUARD. Both sides declare kv_split_size = 2: the destination is
// sharded exactly like the source and holds one block per source logical
// block, so the rank-local one-to-one mapping is the correct one and must stay
// selected.
//
// This is the shape a naive "force rank_local_mapping = false whenever
// kv_split_size > 1" fix would silently break while turning the pair above
// green.
//
// The two shard ranks deliberately address the SAME block ids here: they write
// into different destination ranks (merge_kv_blocks picks the destination
// worker per source rank), so "one writer per block id" is the invariant of the
// unsplit destination, not of an equally split one. The invariants that do
// discriminate are per rank: every one of the local logical blocks keeps a
// destination, one-to-one, in order.
TEST(KvSplitWidthMismatchTest, EqualWidthPairKeepsRankLocalMapping) {
  const bool rank_local_mapping =
      rank_local_mapping_for_test(kSourceKvSplitSize,
                                  /*dst_kv_split_size=*/kSourceKvSplitSize);
  // ASSERT, not EXPECT: if the gate is wrong here, the measurement below would
  // itself trip the builder's coverage CHECK and abort the binary, hiding the
  // real failure behind a crash.
  ASSERT_TRUE(rank_local_mapping)
      << "equal declared widths must keep the rank-local mapping";

  const Coverage coverage =
      measure_coverage(rank_local_mapping, kSourceLogicalBlockCount);
  std::cout << "EQUAL_WIDTH " << describe(coverage, rank_local_mapping)
            << std::endl;

  SCOPED_TRACE(describe(coverage, rank_local_mapping));
  EXPECT_EQ(coverage.covered.size(), kSourceLogicalBlockCount);
  EXPECT_EQ(coverage.source_blocks_mapped,
            static_cast<size_t>(kSourceLogicalBlockCount) *
                static_cast<size_t>(kSourceKvSplitSize));
  const std::vector<uint64_t> expected =
      destination_block_ids(kSourceLogicalBlockCount);
  for (size_t rank = 0; rank < coverage.per_shard_remote_ids.size(); ++rank) {
    EXPECT_EQ(coverage.per_shard_remote_ids[rank], expected)
        << "shard " << rank
        << " must map its logical blocks one-to-one, in order";
  }

  // What the naive "false whenever the source is split" fix would do to this
  // pair: the builder expands the destination's four ids by the source's stride
  // and trips its own coverage CHECK (batch_input_builder.cpp, "KV remote id
  // coverage shortage"), so the transfer cannot be built at all. That CHECK is
  // a process-fatal assertion, not a return value, so the path has to be driven
  // out of process: the child must be killed by the assertion (SIGABRT), and
  // its death message is checked to be that coverage CHECK so an unrelated
  // crash cannot satisfy the clause.
  EXPECT_EXIT(
      {
        for (int32_t kv_split_rank = 0; kv_split_rank < kSourceKvSplitSize;
             ++kv_split_rank) {
          (void)source_rank_destination_blocks(kv_split_rank,
                                               /*rank_local_mapping=*/false,
                                               kSourceLogicalBlockCount);
        }
        std::cout << "no coverage CHECK fired" << std::endl;
        _exit(0);
      },
      ::testing::KilledBySignal(SIGABRT),
      "KV remote id coverage shortage")
      << "the guarded shape must not survive as a mapping";
}

// The classifier's own contract, including the fail-closed half: only equal
// declared widths and an unsplit destination are expressible, every other pair
// is refused with both widths named, and the settled pair is expressible
// (STRIDED) rather than a rejection -- refusing it would fail closed on the
// topology we actually ship.
TEST(KvSplitWidthMismatchTest, ClassifiesDeclaredWidthPairs) {
  std::string reason;
  // The write mode defaults to unspecified: these pairs model instances that
  // predate the field, which keep the legacy admission behavior.
  const auto plan = [&reason](int32_t src_kv_split_size,
                              int32_t dst_kv_split_size,
                              int32_t src_cp_index_write_mode =
                                  kCpIndexWriteModeUnspecified,
                              int32_t dst_cp_index_write_mode =
                                  kCpIndexWriteModeUnspecified) {
    reason.clear();
    return plan_kv_split_widths(src_kv_split_size,
                                dst_kv_split_size,
                                src_cp_index_write_mode,
                                dst_cp_index_write_mode,
                                &reason);
  };

  EXPECT_EQ(plan(/*src=*/1, /*dst=*/1), KvSplitWidthPlan::RANK_LOCAL);
  EXPECT_EQ(plan(/*src=*/kSourceKvSplitSize, /*dst=*/kSourceKvSplitSize),
            KvSplitWidthPlan::RANK_LOCAL);
  EXPECT_EQ(plan(/*src=*/kSourceKvSplitSize, /*dst=*/1),
            KvSplitWidthPlan::STRIDED);
  EXPECT_EQ(plan(/*src=*/2, /*dst=*/1), KvSplitWidthPlan::STRIDED);
  EXPECT_TRUE(reason.empty())
      << "an expressible pair must not name a reason: " << reason;

  const std::vector<std::pair<int32_t, int32_t>> refused = {
      {/*src=*/1, /*dst=*/kSourceKvSplitSize},
      {/*src=*/kSourceKvSplitSize, /*dst=*/3},
      {/*src=*/3, /*dst=*/kSourceKvSplitSize},
      {/*src=*/0, /*dst=*/1},
      {/*src=*/kSourceKvSplitSize, /*dst=*/0}};
  for (const auto& [src, dst] : refused) {
    EXPECT_EQ(plan(src, dst), KvSplitWidthPlan::UNRECONCILABLE)
        << "src=" << src << ", dst=" << dst;
    EXPECT_FALSE(reason.empty()) << "src=" << src << ", dst=" << dst;
    EXPECT_NE(reason.find(std::to_string(src)), std::string::npos) << reason;
    EXPECT_NE(reason.find(std::to_string(dst)), std::string::npos) << reason;
  }
}

// The declared width is the transfer plan's INPUT, and before it reaches the
// plan it crosses a process and a repository boundary: decode sets it -> etcd
// JSON -> the sibling xllm-service parses it -> GetInstanceInfo ->
// instance_info_from_proto(). A producer that does not carry
// InstanceMetaInfo.kv_split_size (field 13) -- a service binary older than the
// field -- answers with proto3's 0, and the reader must keep that as "not
// declared". Folding 0 into InstanceInfo's 1 default is what makes the two
// indistinguishable, and then the mirror image of the settled defect is
// silent: a decode that intends dcp=2 reads as 1 and takes the 1:1 path.
TEST(KvSplitWidthMismatchTest,
     UndeclaredDestinationWidthIsRefusedNotAssumedOne) {
  xllm_service::proto::InstanceMetaInfo legacy_response;
  legacy_response.set_name("decode-legacy-service");
  legacy_response.set_dp_size(4);
  const InstanceInfo legacy = instance_info_from_proto(legacy_response);

  // "not declared" stays distinguishable from "declares 1" ...
  EXPECT_EQ(legacy.kv_split_size, 0)
      << "a response without the field must not be read as a declared width";
  EXPECT_EQ(legacy.dp_size, 4);

  // ... and the plan refuses the pair rather than assuming the missing width.
  std::string reason;
  EXPECT_EQ(plan_kv_split_widths(kSourceKvSplitSize,
                                 legacy.kv_split_size,
                                 kCpIndexWriteModeUnspecified,
                                 legacy.cp_index_write_mode,
                                 &reason),
            KvSplitWidthPlan::UNRECONCILABLE);
  EXPECT_FALSE(reason.empty());

  // A peer that genuinely declares 1 keeps the settled topology expressible:
  // P kv_split=2 -> D kv_split=1 is the pair we ship, so the two cases must not
  // be conflated.
  reason.clear();
  EXPECT_EQ(plan_kv_split_widths(kSourceKvSplitSize,
                                 /*dst_kv_split_size=*/1,
                                 kCpIndexWriteModeUnspecified,
                                 kCpIndexWriteModeUnspecified,
                                 &reason),
            KvSplitWidthPlan::STRIDED);
  EXPECT_TRUE(reason.empty()) << reason;

  // A producer that does carry the field is read verbatim.
  xllm_service::proto::InstanceMetaInfo declared_response;
  declared_response.set_name("decode-dcp2");
  declared_response.set_kv_split_size(2);
  EXPECT_EQ(instance_info_from_proto(declared_response).kv_split_size, 2);
}

// The scheduler half of the contract (disagg_pd_scheduler.cpp,
// dispatch_requests): an unreconcilable pair must be refused with a named
// INVALID_ARGUMENT before the decode reservation and the AddNewRequests RPC,
// and the refusal decision is exactly the plan the gate consumes. The dispatch
// loop itself needs a live xservice registry, so this pins the decision it
// composes -- the same seam the existing scheduler tests use for
// has_rank_preserving_kv_groups.
TEST(KvSplitWidthMismatchTest,
     SchedulerRefusesUnreconcilableWidthPairBeforeReservation) {
  // A kv_split=2 prefill against a decode declaring 3: not expressible as a
  // mapping, so dispatch must refuse it, loudly, with both widths named.
  std::string reason;
  EXPECT_EQ(plan_kv_split_widths(/*src=*/kSourceKvSplitSize,
                                 /*dst=*/3,
                                 kCpIndexWriteModeUnspecified,
                                 kCpIndexWriteModeUnspecified,
                                 &reason),
            KvSplitWidthPlan::UNRECONCILABLE);
  EXPECT_NE(reason.find(std::to_string(kSourceKvSplitSize)), std::string::npos)
      << reason;
  EXPECT_NE(reason.find("3"), std::string::npos) << reason;

  // The message the scheduler emits is "decode instance <name>: " + reason, so
  // the widths above are what the operator sees in the INVALID_ARGUMENT.
  const std::string message = "decode instance decode-3: " + reason;
  EXPECT_NE(message.find(std::to_string(kSourceKvSplitSize)),
            std::string::npos);
  EXPECT_NE(message.find("destination=3"), std::string::npos) << message;

  // No mapping may be derived for the refused pair ...
  EXPECT_FALSE(rank_local_mapping_for_test(kSourceKvSplitSize,
                                           /*dst_kv_split_size=*/3));
  // ... while every pair dispatch admits still derives one: the settled 2 -> 1
  // pair takes the stride path and the equal-width pair stays rank-local.
  EXPECT_FALSE(rank_local_mapping_for_test(kSourceKvSplitSize,
                                           /*dst_kv_split_size=*/1));
  EXPECT_TRUE(rank_local_mapping_for_test(kSourceKvSplitSize,
                                          /*dst_kv_split_size=*/2));
}

// The write-mode half of the admission contract: a side in the SHARDED CP
// index write mode keeps only page 0 of each INDEX resource valid on any
// rank, so an equal-width kv_split_size > 1 pair -- whose RANK_LOCAL plan
// moves every page 1:1 -- would deliver stale peer pages with no error
// anywhere on the link. The refusal is SYMMETRIC: the sharded declaration
// poisons the pair whether it is the source or the destination that carries
// it, because the sharded side's non-zero pages are stale regardless of who
// wrote them. The plan must refuse both directions while keeping the
// supported sharded pairing (an unsplit kv_split_size == 1 destination, whose
// strided plan reads exactly page 0) and both legacy behaviors expressible.
TEST(KvSplitWidthMismatchTest, ShardedWriteModeRefusesEqualWidthPair) {
  std::string reason;
  const auto plan = [&reason](int32_t src_kv_split_size,
                              int32_t dst_kv_split_size,
                              int32_t src_cp_index_write_mode,
                              int32_t dst_cp_index_write_mode =
                                  kCpIndexWriteModeUnspecified) {
    reason.clear();
    return plan_kv_split_widths(src_kv_split_size,
                                dst_kv_split_size,
                                src_cp_index_write_mode,
                                dst_cp_index_write_mode,
                                &reason);
  };

  // The refused pair: sharded source, equal width > 1.
  EXPECT_EQ(
      plan(kSourceKvSplitSize, kSourceKvSplitSize, kCpIndexWriteModeSharded),
      KvSplitWidthPlan::UNRECONCILABLE);
  EXPECT_NE(reason.find("write mode"), std::string::npos) << reason;
  EXPECT_NE(reason.find(cp_index_write_mode_label(kCpIndexWriteModeSharded)),
            std::string::npos)
      << reason;
  EXPECT_NE(reason.find(std::to_string(kSourceKvSplitSize)), std::string::npos)
      << reason;

  // The mirror-image refusal: a replicated source against a destination that
  // declares sharded must not slip through as RANK_LOCAL -- the destination's
  // own INDEX pages are stale, so the same 1:1 moves corrupt silently.
  EXPECT_EQ(plan(kSourceKvSplitSize,
                 kSourceKvSplitSize,
                 kCpIndexWriteModeReplicated,
                 kCpIndexWriteModeSharded),
            KvSplitWidthPlan::UNRECONCILABLE);
  EXPECT_NE(reason.find("write mode"), std::string::npos) << reason;
  EXPECT_NE(reason.find("the destination"), std::string::npos) << reason;
  EXPECT_NE(reason.find(cp_index_write_mode_label(kCpIndexWriteModeSharded)),
            std::string::npos)
      << reason;

  // Both sides sharded names both in the reason rather than picking one.
  EXPECT_EQ(plan(kSourceKvSplitSize,
                 kSourceKvSplitSize,
                 kCpIndexWriteModeSharded,
                 kCpIndexWriteModeSharded),
            KvSplitWidthPlan::UNRECONCILABLE);
  EXPECT_NE(reason.find("both sides"), std::string::npos) << reason;

  // The supported sharded pairing: an unsplit destination plans STRIDED and
  // the page-level overlap only ever reads page 0, which is exactly the page
  // a sharded writer keeps valid. A sharded DESTINATION with an unsplit
  // source is the same settled shape viewed from the other end.
  EXPECT_EQ(plan(kSourceKvSplitSize, /*dst=*/1, kCpIndexWriteModeSharded),
            KvSplitWidthPlan::STRIDED);
  EXPECT_TRUE(reason.empty()) << reason;

  // Replicated writes make every page valid, so the equal-width pair keeps
  // its RANK_LOCAL plan ...
  EXPECT_EQ(
      plan(kSourceKvSplitSize, kSourceKvSplitSize, kCpIndexWriteModeReplicated),
      KvSplitWidthPlan::RANK_LOCAL);
  EXPECT_TRUE(reason.empty()) << reason;
  // ... and the mode is irrelevant at kv_split_size == 1 (no stripes exist).
  EXPECT_EQ(plan(1, 1, kCpIndexWriteModeSharded), KvSplitWidthPlan::RANK_LOCAL);

  // Unspecified (0) is a legacy instance that predates the field: today's
  // behavior, including the equal-width RANK_LOCAL plan.
  EXPECT_EQ(
      plan(
          kSourceKvSplitSize, kSourceKvSplitSize, kCpIndexWriteModeUnspecified),
      KvSplitWidthPlan::RANK_LOCAL);
  EXPECT_TRUE(reason.empty()) << reason;

  // The dispatch composition agrees: a sharded source derives no
  // rank_local_mapping for an equal-width destination because the pair is
  // refused before any mapping is built.
  proto::DisaggResponse response;
  response.add_groups()->set_group_id(cache_group_id(BlockType::KV));
  InstanceInfo remote;
  remote.kv_split_size = kSourceKvSplitSize;
  const KvSplitDispatchPlan sharded_plan = resolve_kv_split_plan(
      kSourceKvSplitSize, kCpIndexWriteModeSharded, remote, &response);
  EXPECT_EQ(sharded_plan.plan, KvSplitWidthPlan::UNRECONCILABLE);
  EXPECT_FALSE(sharded_plan.rank_local_mapping);
  EXPECT_FALSE(sharded_plan.reason.empty());

  // ... and the destination direction composes the same way: resolve reads
  // remote_info.cp_index_write_mode, so a sharded destination declaration is
  // refused before any mapping is built, not silently ignored.
  InstanceInfo sharded_remote;
  sharded_remote.kv_split_size = kSourceKvSplitSize;
  sharded_remote.cp_index_write_mode = kCpIndexWriteModeSharded;
  const KvSplitDispatchPlan sharded_dst_plan =
      resolve_kv_split_plan(kSourceKvSplitSize,
                            kCpIndexWriteModeReplicated,
                            sharded_remote,
                            &response);
  EXPECT_EQ(sharded_dst_plan.plan, KvSplitWidthPlan::UNRECONCILABLE);
  EXPECT_FALSE(sharded_dst_plan.rank_local_mapping);
  EXPECT_FALSE(sharded_dst_plan.reason.empty());
}

// The C++ declaration of the write mode must mirror the python switch
// (xllm/python/model_executor/cp_utils.py cp_index_write_mode): only an
// explicit "sharded" selects the sharded mode, and unset, empty, or
// unrecognized values keep the default replicated mode. register_instance_info
// stamps this value on the registration record, so a drift between the two
// readers would declare a mode the runtime does not enforce (or vice versa).
TEST(KvSplitWidthMismatchTest, DeclaredCpIndexWriteModeMirrorsPythonSwitch) {
  constexpr char kEnvName[] = "XLLM_CP_INDEX_WRITE_MODE";
  // Start from a pristine environment: whatever the harness inherited must
  // not leak into the expectations below, and the guard restores the
  // caller's original value (not an unconditional unset) on exit.
  const xllm::tests::ScopedEnvironmentVariable env_guard(kEnvName);
  ::unsetenv(kEnvName);
  EXPECT_EQ(declared_cp_index_write_mode(), kCpIndexWriteModeReplicated)
      << "an unset switch keeps the replicated default";

  const std::vector<std::pair<std::string, int32_t>> cases = {
      {"sharded", kCpIndexWriteModeSharded},
      {"replicated", kCpIndexWriteModeReplicated},
      // Python strips and lowercases before matching.
      {"  SHARDED  ", kCpIndexWriteModeSharded},
      {"Sharded", kCpIndexWriteModeSharded},
      // python's strip set is the ASCII whitespace set (cp_utils strips
      // " \t\n\v\f\r" explicitly); the C++ reader must agree
      // (absl::StripAsciiWhitespace covers the same set).
      {"\v\tSHARDED\f\r\n", kCpIndexWriteModeSharded},
      {"\f", kCpIndexWriteModeReplicated},
      // Non-ASCII whitespace (full-width space U+3000 from CJK input
      // methods) is NOT in the set on either side: both readers must fall
      // back to the default -- if python's old str.strip() ever comes back,
      // it would read sharded here while this C++ table reads replicated,
      // i.e. an owner-sharded instance declaring replicated to its peers.
      {"\xE3\x80\x80SHARDED\xE3\x80\x80", kCpIndexWriteModeReplicated},
      {"\xC2\xA0sharded", kCpIndexWriteModeReplicated},
      // An empty value is deliberately not a mode (an unset shell variable).
      {"", kCpIndexWriteModeReplicated},
      // Unrecognized values keep the default instead of flipping the mode.
      {"owner-sharded", kCpIndexWriteModeReplicated},
      {"bogus", kCpIndexWriteModeReplicated},
  };
  for (const auto& [value, expected] : cases) {
    ::setenv(kEnvName, value.c_str(), /*replace=*/1);
    EXPECT_EQ(declared_cp_index_write_mode(), expected)
        << "XLLM_CP_INDEX_WRITE_MODE=\"" << value << "\"";
  }
  EXPECT_EQ(declared_cp_index_write_mode(), kCpIndexWriteModeReplicated);
}

// The declared write mode crosses two boundaries before it reaches the plan:
// the etcd registration JSON (InstanceInfo::serialize_to_json -> the sibling
// xllm-service -> InstanceMetaInfo) and GetInstanceInfo ->
// instance_info_from_proto(). Both must carry it verbatim, with proto3's 0
// staying distinguishable from a declared mode the same way kv_split_size's
// 0 stays distinguishable from a declared 1.
TEST(KvSplitWidthMismatchTest, RegistrationCarriesCpIndexWriteMode) {
  // The proto response: absent means the peer predates the field ...
  xllm_service::proto::InstanceMetaInfo legacy_response;
  legacy_response.set_name("decode-legacy-mode");
  const InstanceInfo legacy = instance_info_from_proto(legacy_response);
  EXPECT_EQ(legacy.cp_index_write_mode, kCpIndexWriteModeUnspecified);

  // ... and a declared mode is read verbatim.
  xllm_service::proto::InstanceMetaInfo declared_response;
  declared_response.set_name("decode-sharded");
  declared_response.set_kv_split_size(kSourceKvSplitSize);
  declared_response.set_cp_index_write_mode(kCpIndexWriteModeSharded);
  const InstanceInfo declared = instance_info_from_proto(declared_response);
  EXPECT_EQ(declared.cp_index_write_mode, kCpIndexWriteModeSharded);
  EXPECT_EQ(declared.kv_split_size, kSourceKvSplitSize);

  // The etcd record the sibling service parses carries the same key, so a
  // name-driven master maps it onto the new proto field without further
  // changes on this side.
  const nlohmann::json record = declared.serialize_to_json();
  ASSERT_TRUE(record.contains("cp_index_write_mode"));
  EXPECT_EQ(record["cp_index_write_mode"].get<int32_t>(),
            kCpIndexWriteModeSharded);
  EXPECT_EQ(record["kv_split_size"].get<int32_t>(), kSourceKvSplitSize);
}

// INDEX-page characterization for the heterogeneous pair (the secondary risk
// the m7 diagnosis flagged): the PREFILL's INDEX manifest packs
// indexer_pages_per_block() = kv_split_size physical index pages into one
// logical resource (cache_layout_builder.cpp describe_replicated_index_pages:
// span repeat_count = kv_split_size, one resource per logical block), while the
// kv_split=1 decode has exactly one page per block. This test binds the strided
// block mapping through the real planner and pins where each owner's INDEX
// resource lands on the decode side.
namespace {

constexpr uint64_t kIndexPageTokens = 2;
constexpr uint64_t kIndexHeadDim = 4;
constexpr uint64_t kIndexPageBytes = kIndexPageTokens * kIndexHeadDim;
constexpr uint64_t kIndexBlocks = kSourceLogicalBlockCount;
constexpr uint64_t kDecodeIndexBlocks = kDestinationBlockCount;

WorkerCacheLayoutManifest make_index_side_manifest(
    const std::string& addr,
    uint64_t buffer_id,
    int32_t kv_split_rank,
    int32_t kv_split_size,
    uint64_t index_pages_per_block,
    uint64_t resource_count) {
  WorkerCacheLayoutManifest manifest;
  manifest.incarnation_id = addr + "-incarnation";
  manifest.layout_generation = 1;
  manifest.fingerprint = "kv-split-width-test-model";
  manifest.backend = "cpu";
  manifest.layout_family = "token_head_dim";
  manifest.cluster_id = static_cast<uint64_t>(kv_split_rank + 1);
  manifest.addr = addr;
  manifest.listen_port = 20000;
  manifest.coordinates.dp_rank = 0;
  manifest.coordinates.dp_size = 1;
  manifest.coordinates.tp_rank = 0;
  manifest.coordinates.tp_size = 1;
  manifest.coordinates.cp_rank = kv_split_rank;
  manifest.coordinates.cp_size = kv_split_size;
  manifest.coordinates.kv_split_rank = kv_split_rank;
  manifest.coordinates.kv_split_size = kv_split_size;

  // One KEY tensor (full logical block per resource) and one INDEX tensor
  // (index_pages_per_block pages per resource), both in the flat KV group --
  // the role set glm5_next registers under BlockType::KV. Each tensor gets its
  // own transfer buffer, exactly as register_kv_cache hands out per-tensor
  // buffers in production.
  const std::vector<KVCacheTensorRole> roles = {KVCacheTensorRole::KEY,
                                                KVCacheTensorRole::INDEX};
  for (size_t role_index = 0; role_index < roles.size(); ++role_index) {
    const KVCacheTensorRole& role = roles[role_index];
    const bool is_index = role == KVCacheTensorRole::INDEX;
    const uint64_t pages_per_resource = is_index ? index_pages_per_block : 1;
    CacheTensorManifest tensor;
    tensor.cache_namespace = CacheNamespace::MAIN;
    tensor.layer_id = 0;
    tensor.role =
        static_cast<int32_t>(static_cast<KVCacheTensorRole::Value>(role));
    tensor.group_id = cache_group_id(BlockType::KV);
    tensor.mooncake_buffer_id = static_cast<int64_t>(buffer_id + role_index);
    tensor.scalar_type = 0;
    tensor.element_bytes = 1;
    tensor.shape = {static_cast<int64_t>(resource_count * pages_per_resource),
                    static_cast<int64_t>(kIndexPageTokens),
                    1,
                    static_cast<int64_t>(kIndexHeadDim)};
    tensor.stride = {static_cast<int64_t>(kIndexPageTokens * kIndexHeadDim),
                     static_cast<int64_t>(kIndexHeadDim),
                     static_cast<int64_t>(kIndexHeadDim),
                     1};
    tensor.contiguous = true;
    tensor.resource_count = resource_count;
    tensor.physical_rows_per_resource = pages_per_resource;
    tensor.resource_stride_bytes = kIndexPageBytes * pages_per_resource;
    tensor.buffer_bytes = tensor.resource_count * tensor.resource_stride_bytes;
    tensor.block_token_capacity = kIndexPageTokens;
    tensor.shard.kind = LogicalShardKind::REPLICATED;
    tensor.shard.resource_scope = CacheResourceScope::BLOCK;
    LogicalSpan span;
    span.logical_tensor = role.to_string();
    span.logical_offset_bytes = 0;
    span.physical_offset_bytes = 0;
    span.bytes_per_region = kIndexPageBytes;
    span.repeat_count = pages_per_resource;
    span.logical_stride_bytes = kIndexPageBytes;
    span.physical_stride_bytes = kIndexPageBytes;
    span.owner_tp_rank = 0;
    tensor.shard.spans.emplace_back(std::move(span));
    manifest.tensors.emplace_back(std::move(tensor));
  }
  return manifest;
}

}  // namespace

// The strided mapping must route each owner's INDEX resource in lockstep with
// its KV blocks: owner r pushes local logical block k to decode block
// kv_split_rank + k * kv_split_size, and the INDEX resource k must land on the
// decode's INDEX resource for that same block. The page-level overlap is also
// pinned: the decode's one-page INDEX span only intersects PAGE 0 of the
// prefill's kv_split-page resource, so the byte region transferred is
// [resource k, page 0) -> [decode resource kv_split_rank + k * kv_split, page
// 0).
//
// The residual this characterization used to leave visible (m7 diagnosis
// section 6, secondary risk) is resolved: page 0 of the local resource IS
// always the owner's valid stripe page. The CP write path now stores each
// owner's stripe at the FIRST page of the logical block's group (row
// block-table entry * kv_split -- localize_pool_write_block_table for the pool
// writers, localize_index_write_slots for paged index writes, and the
// materialized gather in _materialize_cp_cache), so the page transferred here
// is exactly the page written, for every owner. The manifest still groups rows
// [block * kv_split, (block + 1) * kv_split) into the resource
// (validate_physical_coverage requires the spans to tile every physical byte),
// and the peer pages of each group stay unwritten on this rank: instance
// internal reads gather them over the CP group instead of the local rows.
TEST(KvSplitWidthMismatchTest,
     StridedMappingRoutesIndexResourcesInLockstepWithKvBlocks) {
  // The default replicated write mode: every rank persists every stripe's
  // page at its natural row, so rank r must ship page r of each source
  // resource -- the strided remote block r + k * kv_split needs source page
  // k * kv_split + r, and shipping page 0 from both owners would deliver
  // stripe 0's bytes twice ([A, A] instead of [A, B]).
  ReshardPlanner planner;
  for (int32_t kv_split_rank = 0; kv_split_rank < kSourceKvSplitSize;
       ++kv_split_rank) {
    // Prefill side: kv_split=2, INDEX resources carry 2 physical pages.
    const WorkerCacheLayoutManifest source = make_index_side_manifest(
        "prefill",
        /*buffer_id=*/3,
        /*kv_split_rank=*/kv_split_rank,
        /*kv_split_size=*/kSourceKvSplitSize,
        /*index_pages_per_block=*/static_cast<uint64_t>(kSourceKvSplitSize),
        /*resource_count=*/kIndexBlocks);
    // Decode side: kv_split=1 (cp1), one INDEX page per block, twice as many
    // blocks for the same prompt.
    const WorkerCacheLayoutManifest destination =
        make_index_side_manifest("decode",
                                 /*buffer_id=*/17,
                                 /*kv_split_rank=*/0,
                                 /*kv_split_size=*/1,
                                 /*index_pages_per_block=*/1,
                                 /*resource_count=*/kDecodeIndexBlocks);

    ReshardPlanTemplate plan;
    const Status plan_status =
        planner.build_outgoing_plan(source,
                                    destination,
                                    &plan,
                                    /*include_replicas=*/false,
                                    kCpIndexWriteModeReplicated);
    ASSERT_TRUE(plan_status.ok())
        << "kv_split_rank=" << kv_split_rank << ": " << plan_status.message();

    // The strided mapping this owner derives after filter_kv_split_infos:
    // local logical block k -> decode block kv_split_rank + k * kv_split.
    KVTransferMapping mapping;
    mapping.group_id = cache_group_id(BlockType::KV);
    for (uint64_t k = 0; k < kIndexBlocks; ++k) {
      mapping.local_ids.emplace_back(k);
      mapping.remote_ids.emplace_back(
          static_cast<uint64_t>(kv_split_rank) +
          k * static_cast<uint64_t>(kSourceKvSplitSize));
    }

    std::vector<ByteRegion> regions;
    ASSERT_TRUE(RequestRegionBinder()
                    .bind(plan,
                          {mapping},
                          CacheNamespace::MAIN,
                          /*layer_id=*/0,
                          &regions)
                    .ok())
        << "kv_split_rank=" << kv_split_rank;
    ASSERT_FALSE(regions.empty());

    // KEY regions (one page per resource on both sides) and INDEX regions
    // (two pages per resource on the prefill, one on the decode) are told
    // apart by their transfer buffers: make_index_side_manifest assigns
    // buffer_id + role_index, so KEY is 3/17 and INDEX is 4/18 here.
    constexpr uint64_t kSourceKeyBuffer = 3;
    constexpr uint64_t kSourceIndexBuffer = 4;
    uint32_t key_regions = 0;
    uint32_t index_regions = 0;
    for (const ByteRegion& region : regions) {
      const uint64_t local_page_bytes = kIndexPageBytes;
      const uint64_t local_resource_bytes =
          region.local_buffer_id == kSourceIndexBuffer ? kIndexPageBytes * 2
                                                       : kIndexPageBytes;
      // Every region is page-granular and stays inside one resource on both
      // sides: the decode's single-page spans cannot cross a resource.
      EXPECT_EQ(region.length, kIndexPageBytes);
      EXPECT_EQ(region.local_offset % local_page_bytes, 0U);
      EXPECT_EQ(region.remote_offset % kIndexPageBytes, 0U);
      const uint64_t local_resource =
          region.local_offset / local_resource_bytes;
      const uint64_t local_page =
          (region.local_offset % local_resource_bytes) / kIndexPageBytes;
      const uint64_t remote_resource = region.remote_offset / kIndexPageBytes;
      if (region.local_buffer_id == kSourceKeyBuffer) {
        ++key_regions;
        // The full-block KV transfer follows the strided mapping directly:
        // local block k -> decode block kv_split_rank + k * kv_split.
        EXPECT_EQ(local_page, 0U);
      } else {
        ASSERT_EQ(region.local_buffer_id, kSourceIndexBuffer);
        ++index_regions;
        // The INDEX page-level overlap under the replicated write mode: the
        // decode's one-page span intersects the source resource at THIS
        // owner's stripe page (kv_split_rank), not page 0.
        EXPECT_EQ(local_page, static_cast<uint64_t>(kv_split_rank))
            << "kv_split_rank=" << kv_split_rank
            << ", local_resource=" << local_resource;
      }
      // The lockstep property both tensors must keep: prefill resource k
      // lands on decode resource kv_split_rank + k * kv_split -- the same
      // block the strided KV mapping selected, never the 1:1 target k the
      // width-blind gate picked.
      EXPECT_EQ(remote_resource,
                static_cast<uint64_t>(kv_split_rank) +
                    local_resource * static_cast<uint64_t>(kSourceKvSplitSize))
          << "kv_split_rank=" << kv_split_rank
          << ", local_resource=" << local_resource;
    }
    // Every logical block contributes one KEY region and one INDEX region.
    EXPECT_EQ(key_regions, kIndexBlocks);
    EXPECT_EQ(index_regions, kIndexBlocks);
  }
}

TEST(KvSplitWidthMismatchTest,
     StridedMappingUnderShardedWritesKeepsPageZeroIndexOverlap) {
  // The sharded write mode keeps each rank's stripe at page 0 of its local
  // resource, so the width-collapsing overlap must stay at page 0 for every
  // rank -- the replicated-mode stripe-page shift must NOT apply.
  ReshardPlanner planner;
  for (int32_t kv_split_rank = 0; kv_split_rank < kSourceKvSplitSize;
       ++kv_split_rank) {
    const WorkerCacheLayoutManifest source = make_index_side_manifest(
        "prefill",
        /*buffer_id=*/3,
        kv_split_rank,
        kSourceKvSplitSize,
        /*index_pages_per_block=*/static_cast<uint64_t>(kSourceKvSplitSize),
        kIndexBlocks);
    const WorkerCacheLayoutManifest destination =
        make_index_side_manifest("decode",
                                 /*buffer_id=*/17,
                                 /*kv_split_rank=*/0,
                                 /*kv_split_size=*/1,
                                 /*index_pages_per_block=*/1,
                                 kDecodeIndexBlocks);

    ReshardPlanTemplate plan;
    const Status plan_status =
        planner.build_outgoing_plan(source,
                                    destination,
                                    &plan,
                                    /*include_replicas=*/false,
                                    kCpIndexWriteModeSharded);
    ASSERT_TRUE(plan_status.ok()) << plan_status.message();

    KVTransferMapping mapping;
    mapping.group_id = cache_group_id(BlockType::KV);
    for (uint64_t k = 0; k < kIndexBlocks; ++k) {
      mapping.local_ids.emplace_back(k);
      mapping.remote_ids.emplace_back(
          static_cast<uint64_t>(kv_split_rank) +
          k * static_cast<uint64_t>(kSourceKvSplitSize));
    }

    std::vector<ByteRegion> regions;
    ASSERT_TRUE(RequestRegionBinder()
                    .bind(plan,
                          {mapping},
                          CacheNamespace::MAIN,
                          /*layer_id=*/0,
                          &regions)
                    .ok());
    uint32_t index_regions = 0;
    for (const ByteRegion& region : regions) {
      if (region.local_buffer_id != 4) {  // kSourceKeyBuffer=3, INDEX=4.
        continue;
      }
      ++index_regions;
      const uint64_t local_resource_bytes = kIndexPageBytes * 2;
      const uint64_t local_page =
          (region.local_offset % local_resource_bytes) / kIndexPageBytes;
      EXPECT_EQ(local_page, 0U) << "kv_split_rank=" << kv_split_rank;
    }
    EXPECT_EQ(index_regions, kIndexBlocks);
  }
}

// Equal-width INDEX characterization (the replicated-write unlock): under
// XLLM_CP_INDEX_WRITE_MODE=replicated (the default) every rank persists every
// stripe's page of every logical block at its natural row, so a pair of EQUAL
// kv_split widths (src kv2 -> dst kv2, the D(kv_split SfaDcp) topology) can
// transfer INDEX resources 1:1 exactly like the PR370/M0 model: equal widths
// plan to RANK_LOCAL, and the plan's page-level overlap covers EVERY page of
// each source resource (both INDEX spans are kv_split pages wide), so any
// single active writer supplies all valid pages. Under the m7 owner-sharded
// writes only page 0 of each source resource was ever valid on any rank, so
// page 1 would have transferred stale rows and this topology was
// inexpressible.
TEST(KvSplitWidthMismatchTest,
     EqualWidthPairRoutesIndexResourcesOneToOneWithEveryPage) {
  ReshardPlanner planner;
  // Both sides: kv_split=2, INDEX resources carry 2 physical pages, the same
  // resource count for the same prompt.
  const WorkerCacheLayoutManifest source = make_index_side_manifest(
      "prefill",
      /*buffer_id=*/3,
      /*kv_split_rank=*/0,
      /*kv_split_size=*/kSourceKvSplitSize,
      /*index_pages_per_block=*/static_cast<uint64_t>(kSourceKvSplitSize),
      /*resource_count=*/kIndexBlocks);
  const WorkerCacheLayoutManifest destination = make_index_side_manifest(
      "decode",
      /*buffer_id=*/17,
      /*kv_split_rank=*/0,
      /*kv_split_size=*/kSourceKvSplitSize,
      /*index_pages_per_block=*/static_cast<uint64_t>(kSourceKvSplitSize),
      /*resource_count=*/kIndexBlocks);

  ReshardPlanTemplate plan;
  const Status plan_status =
      planner.build_outgoing_plan(source, destination, &plan);
  ASSERT_TRUE(plan_status.ok()) << plan_status.message();

  // Equal declared widths plan to RANK_LOCAL: the 1:1 block mapping every
  // owner derives (the heterogeneous counterpart above takes the strided
  // mapping instead). This pair runs the replicated write mode, whose every
  // INDEX page is valid on every rank.
  ASSERT_EQ(plan_kv_split_widths(kSourceKvSplitSize,
                                 kSourceKvSplitSize,
                                 kCpIndexWriteModeReplicated,
                                 kCpIndexWriteModeReplicated,
                                 /*reason=*/nullptr),
            KvSplitWidthPlan::RANK_LOCAL);
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  for (uint64_t k = 0; k < kIndexBlocks; ++k) {
    mapping.local_ids.emplace_back(k);
    mapping.remote_ids.emplace_back(k);
  }

  std::vector<ByteRegion> regions;
  ASSERT_TRUE(RequestRegionBinder()
                  .bind(plan,
                        {mapping},
                        CacheNamespace::MAIN,
                        /*layer_id=*/0,
                        &regions)
                  .ok());
  ASSERT_FALSE(regions.empty());

  constexpr uint64_t kSourceKeyBuffer = 3;
  constexpr uint64_t kSourceIndexBuffer = 4;
  constexpr uint64_t kIndexPagesPerResource =
      static_cast<uint64_t>(kSourceKvSplitSize);
  // Per-resource page coverage of the INDEX regions: both pages of every
  // resource must be transferable (the equal-width overlap), tracked so the
  // assertion reports which (resource, page) pair went missing.
  std::map<std::pair<uint64_t, uint64_t>, uint64_t> index_page_bytes;
  uint64_t key_regions = 0;
  for (const ByteRegion& region : regions) {
    const uint64_t local_page_bytes = kIndexPageBytes;
    const uint64_t local_resource_bytes =
        kIndexPageBytes * kIndexPagesPerResource;
    EXPECT_EQ(region.local_offset % local_page_bytes, 0U);
    EXPECT_EQ(region.remote_offset % kIndexPageBytes, 0U);
    EXPECT_EQ(region.length % kIndexPageBytes, 0U);
    if (region.local_buffer_id == kSourceKeyBuffer) {
      ++key_regions;
      // KEY resources hold one page per resource on both sides.
      const uint64_t local_resource = region.local_offset / kIndexPageBytes;
      const uint64_t remote_resource = region.remote_offset / kIndexPageBytes;
      EXPECT_EQ(remote_resource, local_resource);
      continue;
    }
    ASSERT_EQ(region.local_buffer_id, kSourceIndexBuffer);
    const uint64_t local_resource = region.local_offset / local_resource_bytes;
    const uint64_t local_page =
        (region.local_offset % local_resource_bytes) / kIndexPageBytes;
    const uint64_t pages = region.length / kIndexPageBytes;
    // The 1:1 lockstep: prefill resource k lands on decode resource k.
    const uint64_t remote_resource =
        region.remote_offset / local_resource_bytes;
    EXPECT_EQ(remote_resource, local_resource);
    for (uint64_t page = 0; page < pages; ++page) {
      index_page_bytes[{local_resource, local_page + page}] += kIndexPageBytes;
    }
  }
  // Every logical block contributes one KEY region...
  EXPECT_EQ(key_regions, kIndexBlocks);
  // ...and BOTH INDEX pages of every resource are covered by the plan: this
  // is what makes a single active writer sufficient under replicated writes
  // (and what made the equal-width pair inexpressible under owner-sharded
  // writes, where page 1 was stale on every rank).
  ASSERT_EQ(index_page_bytes.size(), kIndexBlocks * kIndexPagesPerResource);
  for (uint64_t resource = 0; resource < kIndexBlocks; ++resource) {
    for (uint64_t page = 0; page < kIndexPagesPerResource; ++page) {
      const std::pair<uint64_t, uint64_t> page_key(resource, page);
      const uint64_t covered_bytes = index_page_bytes[page_key];
      EXPECT_EQ(covered_bytes, kIndexPageBytes)
          << "resource=" << resource << ", page=" << page;
    }
  }
}

}  // namespace xllm
