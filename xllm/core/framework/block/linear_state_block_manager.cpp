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

#include "core/framework/block/linear_state_block_manager.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "core/framework/request/sequence.h"

namespace xllm {

namespace {

BlockManager::Options make_linear_state_options(
    uint32_t num_slots,
    int32_t chunk_stride,
    bool enable_prefix_cache,
    bool instance_is_decode,
    uint32_t num_speculative_tokens) {
  BlockManager::Options options;
  options.num_blocks(num_slots);
  options.block_size(chunk_stride);
  options.enable_prefix_cache(enable_prefix_cache);
  options.enable_disagg_pd(false);
  options.block_type(BlockType::LINEAR);
  options.instance_is_decode(instance_is_decode);
  options.num_speculative_tokens(num_speculative_tokens);
  return options;
}

bool crosses_checkpoint(size_t begin, size_t end, size_t checkpoint_stride) {
  CHECK_LE(begin, end);
  CHECK_GT(checkpoint_stride, 0u);
  return begin / checkpoint_stride != end / checkpoint_stride;
}

}  // namespace

bool last_decode_step_crosses_checkpoint(const Sequence& sequence,
                                         size_t checkpoint_stride) {
  return last_decode_step_checkpoint_row(sequence, checkpoint_stride)
      .has_value();
}

std::optional<size_t> last_decode_step_checkpoint_row(
    const Sequence& sequence,
    size_t checkpoint_stride) {
  CHECK_GT(checkpoint_stride, 0u);
  const size_t confirmed_tokens = sequence.last_confirmed_cached_tokens_num();
  const size_t accepted_length = sequence.accepted_length();
  // accepted_length is the maximum accepted draft offset. The preceding bonus
  // token also advances the recurrent cache, so the confirmed span is one
  // larger than that offset.
  if (confirmed_tokens <= accepted_length) {
    return std::nullopt;
  }
  const size_t confirmed_span = accepted_length + 1;
  const size_t begin = confirmed_tokens - confirmed_span;
  if (!crosses_checkpoint(begin, confirmed_tokens, checkpoint_stride)) {
    return std::nullopt;
  }
  const size_t checkpoint_tokens =
      (confirmed_tokens / checkpoint_stride) * checkpoint_stride;
  CHECK_GT(checkpoint_tokens, begin);
  return checkpoint_tokens - begin - 1;
}

LinearStateBlockManager::LinearStateBlockManager(
    uint32_t num_slots,
    int32_t chunk_stride,
    bool enable_prefix_cache,
    bool instance_is_decode,
    uint32_t num_speculative_tokens,
    bool preserve_decode_checkpoint)
    : BlockManagerImpl(make_linear_state_options(num_slots,
                                                 chunk_stride,
                                                 enable_prefix_cache,
                                                 instance_is_decode,
                                                 num_speculative_tokens)),
      preserve_decode_checkpoint_(preserve_decode_checkpoint) {
  CHECK_GT(num_slots, 1u)
      << "linear-state leaf needs at least one usable slot (plus padding)";
  CHECK_GT(chunk_stride, 0)
      << "linear-state leaf needs a positive chunk stride";
}

std::optional<std::vector<Block>>
LinearStateBlockManager::allocate_for_sequence(Sequence* seq,
                                               KVCacheState& kv_state,
                                               size_t /*num_tokens*/) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  if (seq->is_prefill_stage()) {
    if (!options_.instance_is_decode()) {
      return allocate_prefill(seq, kv_state);
    }
    retain_read_source(kv_state);
    return allocate_decode(seq, kv_state);
  }
  return allocate_decode(seq, kv_state);
}

bool LinearStateBlockManager::allocate_for_prefetch(Sequence* seq,
                                                    size_t num_tokens) {
  if (seq == nullptr || block_size() == 0) {
    return false;
  }

  KVCacheState& host_state = seq->host_kv_state();
  const size_t target_blocks = num_tokens / block_size();
  const size_t cached_blocks = host_state.num_cached_blocks(BlockType::LINEAR);
  const size_t shared_blocks = host_state.shared_blocks_num(BlockType::LINEAR);
  const size_t cursor = std::min(cached_blocks, target_blocks);
  const bool source_in_target =
      cached_blocks > 0 && cached_blocks <= target_blocks;

  std::vector<Block> old_blocks = host_state.take_blocks(BlockType::LINEAR);
  // The vector is indexed by logical checkpoint, while only the deepest
  // matched checkpoint has a physical source after prefix-cache admission.
  // Keep the earlier positions as invalid placeholders so Store transfers and
  // restore selection continue to use absolute checkpoint indices.
  std::vector<Block> blocks(target_blocks);
  Block source;
  if (source_in_target) {
    for (size_t index = std::min(cached_blocks, old_blocks.size()); index > 0;
         --index) {
      Block& candidate = old_blocks[index - 1];
      if (candidate.is_valid()) {
        source = std::move(candidate);
        break;
      }
    }
  }
  std::vector<Block> dropped;
  dropped.reserve(old_blocks.size());
  for (size_t index = 0; index < old_blocks.size(); ++index) {
    Block& block = old_blocks[index];
    if (!block.is_valid()) {
      continue;
    }
    if (index >= cursor && index < target_blocks) {
      blocks[index] = std::move(block);
    } else {
      dropped.emplace_back(std::move(block));
    }
  }
  if (!dropped.empty()) {
    deallocate(dropped);
  }

  const bool has_source = source_in_target && source.is_valid();
  if (has_source) {
    blocks[cached_blocks - 1] = std::move(source);
  }

  seq->update_linear_state_hashes(static_cast<uint32_t>(block_size()));
  const Slice<XXH3Key> hashes = seq->linear_state_hashes();
  CHECK_GE(hashes.size(), target_blocks);

  std::vector<size_t> missing;
  missing.reserve(target_blocks > cursor ? target_blocks - cursor : 0);
  for (size_t index = cursor; index < target_blocks; ++index) {
    if (!blocks[index].is_valid()) {
      missing.emplace_back(index);
    }
  }

  size_t allocatable = std::min(
      missing.size(), num_free_blocks() + num_blocks_in_prefix_cache());
  std::vector<Block> allocated = allocate(allocatable);
  if (allocated.empty() && allocatable > 0) {
    allocatable = std::min(missing.size(), num_free_blocks());
    allocated = allocate(allocatable);
  }
  for (size_t index = 0; index < allocated.size(); ++index) {
    const size_t block_index = missing[index];
    allocated[index].set_hash_value(hashes[block_index].data);
    blocks[block_index] = std::move(allocated[index]);
  }

  if (!blocks.empty()) {
    host_state.replace_composite_blocks(
        BlockType::LINEAR,
        std::move(blocks),
        has_source ? std::min(shared_blocks, target_blocks) : 0,
        cursor);
  }
  return allocated.size() == missing.size();
}

void LinearStateBlockManager::trim_prefetch_blocks(Sequence* seq,
                                                   size_t max_hit_tokens) {
  if (seq == nullptr || block_size() == 0) {
    return;
  }

  KVCacheState& host_state = seq->host_kv_state();
  std::vector<Block> blocks = host_state.take_blocks(BlockType::LINEAR);
  const size_t keep = std::min(max_hit_tokens / block_size(), blocks.size());

  // A Linear checkpoint vector is a logical index space. Keep only the
  // deepest valid checkpoint in the accepted gate range; all other physical
  // aliases, including valid blocks after the range, are temporary Store
  // state and must be released by this leaf.
  size_t source_index = keep;
  for (size_t index = keep; index > 0; --index) {
    if (blocks[index - 1].is_valid()) {
      source_index = index - 1;
      break;
    }
  }

  std::vector<Block> to_release;
  to_release.reserve(blocks.size());
  Block source;
  if (source_index < keep) {
    source = blocks[source_index];
  }
  for (size_t index = 0; index < blocks.size(); ++index) {
    if (!blocks[index].is_valid() || index == source_index) {
      continue;
    }
    to_release.emplace_back(std::move(blocks[index]));
  }
  if (!to_release.empty()) {
    deallocate(to_release);
  }

  if (!source.is_valid()) {
    host_state.erase_blocks(BlockType::LINEAR);
    return;
  }

  // LinearStatePrefixCache needs the absolute checkpoint index to derive the
  // chained key. Publish through a sparse logical vector, then collapse the
  // sequence-owned view to the single restore source required by Linear
  // runtime allocation.
  seq->update_linear_state_hashes(static_cast<uint32_t>(block_size()));
  const Slice<XXH3Key> hashes = seq->linear_state_hashes();
  CHECK_GT(hashes.size(), source_index);
  std::vector<Block> publish(source_index + 1);
  publish[source_index] = source;
  BlockManagerImpl::cache(seq->hash_tokens(options_.hasher_type()),
                          publish,
                          /*existed_shared_blocks_num=*/0,
                          seq->mm_data(),
                          hashes);

  std::vector<Block> retained(1);
  retained[0] = std::move(source);
  host_state.replace_composite_blocks(BlockType::LINEAR,
                                      std::move(retained),
                                      /*num_shared_blocks=*/1,
                                      /*cache_publish_cursor=*/1);
}

void LinearStateBlockManager::retain_read_source(KVCacheState& kv_state) {
  std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::LINEAR);
  if (blocks->empty()) {
    return;
  }
  CHECK(blocks->back().is_valid());
  const bool shared_source =
      kv_state.shared_blocks_num(BlockType::LINEAR) == blocks->size();
  const size_t keep_begin = blocks->size() - 1;
  deallocate(Slice<Block>(*blocks).slice(0, keep_begin));
  blocks->erase(blocks->begin(), blocks->begin() + keep_begin);
  kv_state.set_shared_blocks_num(BlockType::LINEAR, shared_source ? 1 : 0);
}

void LinearStateBlockManager::cache_read_source(Sequence* seq,
                                                KVCacheState& kv_state) {
  const Slice<Block> blocks = kv_state.blocks(BlockType::LINEAR);
  if (seq->is_graph_warmup() || prefix_cache_ == nullptr || blocks.empty()) {
    return;
  }
  const size_t chunk_stride = block_size();
  const size_t cached_tokens = seq->kv_cache_tokens_num();
  if (cached_tokens % chunk_stride != 0) {
    return;
  }
  const size_t checkpoint_index = cached_tokens / chunk_stride;
  if (checkpoint_index <= kv_state.num_cached_blocks(BlockType::LINEAR)) {
    return;
  }
  seq->update_linear_state_hashes(block_size());
  const Slice<XXH3Key> hashes = seq->linear_state_hashes();
  CHECK_GE(hashes.size(), checkpoint_index);
  auto* source = kv_state.mutable_blocks(BlockType::LINEAR);
  CHECK(source->back().is_valid());
  source->back().set_hash_value(hashes[checkpoint_index - 1].data);
  prefix_cache_->insert(
      Slice<Block>(*source).slice(source->size() - 1, source->size()));
  kv_state.set_num_cached_blocks(BlockType::LINEAR, checkpoint_index);
}

std::optional<std::vector<Block>> LinearStateBlockManager::allocate_prefill(
    Sequence* seq,
    KVCacheState& kv_state) {
  // If the sequence's effective restore boundary is ahead of this tier's
  // cursor, the existing slot belongs to a shorter cache hit. Drop it without
  // inserting into this tier's prefix cache, then build the two-slot
  // [restore, live] window. The rule applies identically to HBM and Host
  // states; the caller identifies the tier by the KVCacheState reference.
  if (seq->kv_cache_tokens_num() > kv_state.kv_cache_tokens_num()) {
    std::vector<Block> old_blocks = kv_state.take_blocks(BlockType::LINEAR);
    deallocate(old_blocks);
    std::vector<Block> allocated = BlockManagerImpl::allocate(2);
    if (allocated.size() != 2) {
      deallocate(allocated);
      return std::nullopt;
    }
    return allocated;
  }

  cache_read_source(seq, kv_state);
  retain_read_source(kv_state);
  std::vector<Block> allocated = BlockManagerImpl::allocate(1);
  if (allocated.empty()) {
    return std::nullopt;
  }
  return allocated;
}

std::optional<std::vector<Block>> LinearStateBlockManager::allocate_decode(
    Sequence* seq,
    KVCacheState& kv_state) {
  std::vector<Block>* blocks = kv_state.mutable_blocks(BlockType::LINEAR);
  CHECK_LE(blocks->size(), 2u);
  if (seq->is_prefill_stage()) {
    if (!blocks->empty()) {
      return std::vector<Block>{};
    }
    std::vector<Block> allocated = BlockManagerImpl::allocate(1);
    return allocated.empty()
               ? std::nullopt
               : std::optional<std::vector<Block>>(std::move(allocated));
  }

  const size_t confirmed_tokens = seq->last_confirmed_cached_tokens_num();
  const size_t tokens_per_step =
      static_cast<size_t>(options_.num_speculative_tokens()) + 1;
  const size_t checkpoint_stride = block_size();
  const bool completed_checkpoint =
      blocks->size() == 2 &&
      last_decode_step_crosses_checkpoint(*seq, checkpoint_stride);

  if (preserve_decode_checkpoint_ && completed_checkpoint) {
    // HierarchyBlockManagerPool owns the transfer boundary. It moves the
    // HBM and Host front blocks into an OffloadBlockPair after both tiers
    // preserve the same logical checkpoint. Hashing happens once at that
    // ownership boundary.
    return std::vector<Block>{};
  }

  // Prefill may leave its read source in front of the newest write block.
  // Decode starts from the newest state, so discard that historical prefill
  // source before applying the decode rolling rules.  In particular, do not
  // treat the prefill pair as a pending decode window.
  if (blocks->size() == 2 &&
      kv_state.kv_cache_tokens_num() <= seq->num_prompt_tokens()) {
    retain_read_source(kv_state);
  }

  if (blocks->size() == 2) {
    if (completed_checkpoint) {
      deallocate(Slice<Block>(*blocks).slice(0, 1));
      blocks->erase(blocks->begin());
      kv_state.set_shared_blocks_num(BlockType::LINEAR, 0);
    } else {
      // The previous decode step has not yet confirmed a checkpoint. Keep
      // both slots alive and alternate their roles so the next out-of-place
      // step reads the slot written by the preceding step. This is also
      // what makes overlap allocation safe: the old read slot cannot be
      // returned to the global pool while its in-flight step may still use it.
      std::iter_swap(blocks->begin(), blocks->begin() + 1);
      return std::vector<Block>{};
    }
  }

  if (blocks->empty()) {
    std::vector<Block> allocated = BlockManagerImpl::allocate(1);
    return allocated.empty()
               ? std::nullopt
               : std::optional<std::vector<Block>>(std::move(allocated));
  }

  // A stride of one is the legacy fallback when chunked prefill is disabled.
  // It does not describe a persistent LINEAR checkpoint window to roll.
  if (checkpoint_stride == 1) {
    return std::vector<Block>{};
  }

  CHECK_LE(tokens_per_step,
           std::numeric_limits<size_t>::max() - confirmed_tokens)
      << "linear-state confirmed-token cursor overflow";
  const size_t predicted_tokens = confirmed_tokens + tokens_per_step;
  if (!crosses_checkpoint(
          confirmed_tokens, predicted_tokens, checkpoint_stride)) {
    return std::vector<Block>{};
  }

  std::vector<Block> allocated = BlockManagerImpl::allocate(1);
  if (allocated.empty()) {
    // The extra slot only preserves a checkpoint for the future offload path;
    // recurrent decode itself supports reading and writing the live slot in
    // place. Under pressure, keep decoding instead of turning an optional
    // checkpoint copy into a sequence preemption.
    return std::vector<Block>{};
  }
  return allocated;
}

Block LinearStateBlockManager::allocate() {
  std::vector<Block> blocks = BlockManagerImpl::allocate(1);
  if (blocks.empty()) {
    return Block();
  }
  return std::move(blocks[0]);
}

std::vector<Block> LinearStateBlockManager::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/,
    const Slice<XXH3Key>& block_hashes) {
  if (token_ids.size() == 0 || prefix_cache_ == nullptr) {
    return {};
  }
  std::vector<Block> blocks = prefix_cache_->match(
      token_ids.slice(0, token_ids.size() - 1), {}, MMData(), block_hashes);
  VLOG(1) << "[HostCache][LinearMatch] manager=" << this
          << " decode=" << options_.instance_is_decode()
          << " blocks=" << blocks.size();
  for (const Block& block : blocks) {
    if (block.is_valid() && mark_used(&usage_accounted_ids_, block.id())) {
      num_used_blocks_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return blocks;
}

void LinearStateBlockManager::cache(const Slice<int32_t>& token_ids,
                                    std::vector<Block>& blocks,
                                    size_t existed_shared_blocks_num,
                                    const MMData& mm_data,
                                    const Slice<XXH3Key>& block_hashes) {
  if (blocks.empty() || prefix_cache_ == nullptr) {
    return;
  }
  const size_t publish_end =
      std::min(token_ids.size() / block_size(), blocks.size() - 1);
  VLOG(1) << "[HostCache][LinearCache] manager=" << this
          << " decode=" << options_.instance_is_decode()
          << " blocks=" << blocks.size() << " publish_end=" << publish_end;
  if (publish_end <= existed_shared_blocks_num) {
    return;
  }
  CHECK_GE(block_hashes.size(), publish_end);
  BlockManagerImpl::cache(token_ids.slice(0, publish_end * block_size()),
                          blocks,
                          existed_shared_blocks_num,
                          mm_data,
                          block_hashes);
}

}  // namespace xllm
