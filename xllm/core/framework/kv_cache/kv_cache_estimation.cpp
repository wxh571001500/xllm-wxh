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

#include "framework/kv_cache/kv_cache_estimation.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/layers/common/dsa_topk_share_plan.h"
#include "core/platform/platform.h"
#include "framework/block/block_utils.h"
#include "framework/config/kv_cache_config.h"
#include "framework/kv_cache/cache_layout_builder.h"
#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "framework/kv_cache/deepseek_v4_cache_policy.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/model/model_args.h"
#include "util/pretty_print.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {

namespace {

constexpr int32_t kNzAlignment = 16;

int64_t kv_cache_dtype_size(const std::string& kv_cache_dtype,
                            int64_t model_dtype_size) {
  if (kv_cache_dtype == "auto") {
    return model_dtype_size;
  }
  if (kv_cache_dtype == "int8") {
    return 1;
  }
  if (kv_cache_dtype == "fp8_e4m3" || kv_cache_dtype == "fp8_e5m2") {
    return 1;
  }
  return model_dtype_size;
}

int64_t kv_slot_size(const ModelArgs& model_args,
                     const KVCacheEstimateOptions& options,
                     int64_t cache_dtype_size) {
  if (model_args.enable_mla()) {
    // SFA C8 packed layout: one byte tensor holding [int8 nope] + [bf16 rope]
    // + [fp32 tile scale]; K == V so no separate value cache is counted. The
    // predicate carries its own build gate, so no #ifdef is needed here.
    if (util::enable_mla_packed_c8(options.kv_cache_dtype == "int8",
                                   model_args.model_type())) {
      const int64_t nope_bytes = model_args.kv_lora_rank();
      const int64_t rope_bytes =
          MlaPackedC8Layout::kRopeElementBytes * model_args.qk_rope_head_dim();
      const int64_t scale_bytes = MlaPackedC8Layout::kScaleElementBytes *
                                  (nope_bytes / MlaPackedC8Layout::kTileSize);
      return nope_bytes + rope_bytes + scale_bytes;
    }
#if defined(USE_NPU)
    if ((model_args.model_type() == "deepseek_v3" ||
         model_args.model_type() == "deepseek_v3_mtp") &&
        options.enable_prefix_cache) {
      return cache_dtype_size *
             ((model_args.kv_lora_rank() + kNzAlignment - 1) / kNzAlignment +
              (model_args.qk_rope_head_dim() + kNzAlignment - 1) /
                  kNzAlignment) *
             kNzAlignment;
    }
#endif
    return cache_dtype_size *
           (model_args.kv_lora_rank() + model_args.qk_rope_head_dim());
  }

  return 2 * cache_dtype_size * model_args.head_dim() *
         options.n_local_kv_heads;
}

struct IndexCacheFootprint {
  int64_t slot_size = 0;
  int64_t block_size = 0;
};

IndexCacheFootprint estimate_index_cache_footprint(
    const ModelArgs& model_args,
    bool enable_indexer_cache_quantization,
    int64_t dtype_size,
    int64_t block_size,
    KPoolCacheLayout kpool_layout) {
  if (model_args.index_n_heads() <= 0) {
    return {};
  }

  const int64_t index_n_head = 1;
  const bool uses_compressed_tail =
      kpool_layout == KPoolCacheLayout::COMPRESSED_WITH_TAIL &&
      model_args.index_kpool_compress();
  const bool uses_packed_kpool = kpool_layout == KPoolCacheLayout::PACKED &&
                                 model_args.index_kpool_compress();
  if (uses_compressed_tail || uses_packed_kpool) {
    CHECK(!enable_indexer_cache_quantization)
        << "KPool cache layouts do not support indexer_cache_dtype=\"int8\".";
    CHECK_GT(model_args.index_kpool(), 0) << "KPool requires index_kpool > 0.";
  }
  const int64_t replication_factor = indexer_pages_per_block();
  if (enable_indexer_cache_quantization) {
    // int8 index cache: one byte per element, plus an independent per-token
    // fp32 scale (kept separate from the main-KV scale_slot_size path).
    const int64_t slot_size =
        replication_factor * (static_cast<int64_t>(sizeof(int8_t)) *
                                  index_n_head * model_args.index_head_dim() +
                              static_cast<int64_t>(sizeof(float)));
    return {slot_size, slot_size * block_size};
  }
  if (uses_compressed_tail) {
    const int64_t pool_bytes = replication_factor * dtype_size * index_n_head *
                               model_args.index_head_dim();
    const int64_t slot_size =
        (pool_bytes + model_args.index_kpool() - 1) / model_args.index_kpool();
    CHECK_EQ(block_size % model_args.index_kpool(), 0);
    return {slot_size, block_size / model_args.index_kpool() * pool_bytes};
  }
  const int64_t index_width = uses_packed_kpool
                                  ? 2 * model_args.index_head_dim() + 1
                                  : model_args.index_head_dim();
  const int64_t slot_size =
      replication_factor * dtype_size * index_n_head * index_width;
  return {slot_size, slot_size * block_size};
}

int64_t scale_slot_size(const ModelArgs& model_args,
                        const KVCacheEstimateOptions& options) {
  if (options.kv_cache_dtype == "auto") {
    return 0;
  }
  if (model_args.enable_mla()) {
    // SFA C8 folds per-tile fp32 scales into the packed KV byte tensor
    // (kv_slot_size already counts them). Advertising an extra scale slot
    // here would double-book the memory. The predicate carries its own build
    // gate, so the estimator and shape stay symmetric on every backend: a
    // hypothetical fp8 KV path on a supported model must not short-circuit
    // here.
    if (util::enable_mla_packed_c8(options.kv_cache_dtype == "int8",
                                   model_args.model_type())) {
      return 0;
    }
    return sizeof(float);
  }
  return 2 * sizeof(float) * options.n_local_kv_heads;
}

int64_t standard_full_cache_block_size_in_bytes(
    const KVCacheCapacity& kv_cache_cap) {
  const int64_t full_attention_layers =
      std::max<int64_t>(kv_cache_cap.num_full_attention_layers(), 1);
  const int64_t indexer_layers = kv_cache_cap.num_indexer_layers();
  CHECK_GE(indexer_layers, 0) << "num_indexer_layers must be non-negative";
  CHECK_LE(indexer_layers, full_attention_layers)
      << "num_indexer_layers cannot exceed full-attention layers";
  const int64_t logical_block_bytes =
      kv_cache_cap.block_size() *
          (full_attention_layers *
           (kv_cache_cap.slot_size() + kv_cache_cap.scale_slot_size())) +
      indexer_layers * kv_cache_cap.index_block_size();
  CHECK_GT(logical_block_bytes, 0) << "logical block bytes must be positive";
  return logical_block_bytes;
}

int64_t layerwise_split_block_count(const ModelArgs& model_args,
                                    int32_t layerwise_split_size,
                                    const KVCacheCapacity& kv_cache_cap,
                                    int64_t available_bytes,
                                    int64_t additional_block_bytes) {
  const int64_t num_layers = kv_cache_cap.n_layers();
  const std::vector<bool> indexer_layer_mask =
      resolve_indexer_cache_enabled_layers(model_args, num_layers);

  std::vector<int64_t> layer_bytes(static_cast<size_t>(num_layers), 0);
  bool any_indexer_layer = false;
  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    if (!is_full_attention_layer(model_args, layer_id)) {
      continue;
    }
    const bool has_indexer =
        kv_cache_cap.index_slot_size() > 0 &&
        (indexer_layer_mask.empty() ||
         indexer_layer_mask[static_cast<size_t>(layer_id)]);
    any_indexer_layer = any_indexer_layer || has_indexer;
    const int64_t bytes =
        kv_cache_cap.block_size() *
            (kv_cache_cap.slot_size() + kv_cache_cap.scale_slot_size()) +
        (has_indexer ? kv_cache_cap.index_block_size() : 0);
    layer_bytes[static_cast<size_t>(layer_id)] = bytes;
  }

  // Scratch matches an owned layer, including indexer when any layer has one.
  const int64_t scratch_bytes_per_block =
      kv_cache_cap.block_size() *
          (kv_cache_cap.slot_size() + kv_cache_cap.scale_slot_size()) +
      (any_indexer_layer ? kv_cache_cap.index_block_size() : 0);
  CHECK_GT(scratch_bytes_per_block, 0);

  int64_t common_block_count = std::numeric_limits<int64_t>::max();
  for (int32_t split_rank = 0; split_rank < layerwise_split_size;
       ++split_rank) {
    const LayerwiseSplitLayout layout(
        /*enabled=*/true, layerwise_split_size, split_rank);
    const std::vector<bool> layer_cache_owned =
        build_layer_cache_owned(model_args, layout, num_layers);
    int64_t owned_bytes = 0;
    for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      if (layer_cache_owned[static_cast<size_t>(layer_id)]) {
        owned_bytes += layer_bytes[static_cast<size_t>(layer_id)];
      }
    }
    if (owned_bytes == 0) {
      continue;
    }
    const int64_t per_block_bytes =
        owned_bytes + scratch_bytes_per_block + additional_block_bytes;
    common_block_count =
        std::min(common_block_count, available_bytes / per_block_bytes);
  }

  CHECK_NE(common_block_count, std::numeric_limits<int64_t>::max())
      << "No layerwise split rank owns a model layer.";
  CHECK_GT(common_block_count, 0) << "No memory for one layerwise split block.";
  return common_block_count;
}

int64_t effective_num_speculative_tokens(
    const KVCacheEstimateOptions& options) {
  // A prefill instance never receives speculative decode tokens. Keep the
  // runtime MTP options intact for draft prefill, but do not reserve decode
  // checkpoints in its cache shape.
  return options.instance_role == InstanceRole::PREFILL
             ? 0
             : options.num_speculative_tokens;
}

bool enable_hybrid_spec_verify(const ModelArgs& model_args,
                               const KVCacheEstimateOptions& options) {
  return effective_num_speculative_tokens(options) > 0 &&
         !options.is_draft_engine &&
         (is_qwen3_5_target_model_type(model_args.model_type()) ||
          model_args.model_type() == "glm5_next" ||
          model_args.model_type() == "glm5_next_text");
}

int64_t linear_slot_size(const ModelArgs& model_args,
                         const KVCacheEstimateOptions& options,
                         int64_t dtype_size) {
  if (model_args.linear_num_value_heads() <= 0) {
    return 0;
  }
  const int64_t num_speculative_tokens =
      enable_hybrid_spec_verify(model_args, options)
          ? effective_num_speculative_tokens(options)
          : 0;

  const int64_t head_k_dim = model_args.linear_key_head_dim();
  const int64_t head_v_dim = model_args.linear_value_head_dim();
  const int64_t ssm_dtype_size =
      resolve_ssm_dtype_size(model_args.mamba_ssm_dtype(), dtype_size);

  const int64_t linear_ssm_slot_size =
      ssm_dtype_size * options.n_local_linear_v_heads * head_k_dim * head_v_dim;
  const int64_t linear_conv_state_len =
      model_args.linear_conv_kernel_dim() - 1 + num_speculative_tokens;
  const int64_t linear_conv_slot_size =
      dtype_size *
      (head_k_dim * options.n_local_linear_k_heads * 2 +
       head_v_dim * options.n_local_linear_v_heads) *
      linear_conv_state_len;
  return linear_conv_slot_size +
         linear_ssm_slot_size * (num_speculative_tokens + 1);
}

int64_t max_linear_state_blocks(int64_t cache_size_in_bytes,
                                int64_t state_slot_bytes,
                                int64_t full_cache_block_size_in_bytes) {
  if (state_slot_bytes <= 0) {
    return kPaddingLinearStateBlocks;
  }

  CHECK_GT(cache_size_in_bytes, 0);
  CHECK_GT(full_cache_block_size_in_bytes, 0);
  CHECK_GT(state_slot_bytes, 0);

  int64_t max_linear_blocks = (cache_size_in_bytes - 1) / state_slot_bytes;
  const int64_t balanced_max_linear_blocks =
      (cache_size_in_bytes +
       kPaddingLinearStateBlocks * full_cache_block_size_in_bytes) /
      (state_slot_bytes + full_cache_block_size_in_bytes);
  max_linear_blocks = std::min(max_linear_blocks, balanced_max_linear_blocks);

  return std::max<int64_t>(max_linear_blocks, kPaddingLinearStateBlocks);
}

int64_t calculate_linear_state_blocks(int64_t cache_size_in_bytes,
                                      int64_t state_slot_bytes,
                                      int64_t full_cache_block_size_in_bytes,
                                      int64_t max_linear_state_cache_slots) {
  CHECK_GE(max_linear_state_cache_slots, 0)
      << "max_linear_state_cache_slots must be greater than or equal to 0.";
  if (state_slot_bytes <= 0) {
    return kPaddingLinearStateBlocks;
  }
  const int64_t max_blocks = max_linear_state_blocks(
      cache_size_in_bytes, state_slot_bytes, full_cache_block_size_in_bytes);
  if (max_linear_state_cache_slots > 0) {
    const int64_t requested_blocks =
        max_linear_state_cache_slots + kPaddingLinearStateBlocks;
    CHECK_LE(requested_blocks, max_blocks)
        << "max_linear_state_cache_slots requires " << requested_blocks
        << " linear-state blocks, but only " << max_blocks
        << " fit in the configured KV cache budget.";
    return requested_blocks;
  }

  // Auto-size: allocate ~47% of cache bytes to linear-state slots (ratio 0.9
  // means linear fraction = 0.9 / 1.9).
  constexpr double kLinearStateFullKvMemoryRatio = 0.9;
  int64_t auto_blocks = kPaddingLinearStateBlocks;
  if (state_slot_bytes > 0) {
    const double linear_memory_fraction =
        kLinearStateFullKvMemoryRatio / (1.0 + kLinearStateFullKvMemoryRatio);
    const double linear_memory_bytes =
        static_cast<double>(cache_size_in_bytes) * linear_memory_fraction;
    auto_blocks = std::max<int64_t>(
        static_cast<int64_t>(linear_memory_bytes / state_slot_bytes),
        kPaddingLinearStateBlocks);
  }
  // Both bounds already sit at or above kPaddingLinearStateBlocks (auto_blocks
  // is floored above; max_blocks is floored in max_linear_state_blocks), so the
  // min of the two cannot drop below it.
  return std::min<int64_t>(auto_blocks, max_blocks);
}

int64_t dsv4_common_unit_bytes(const Dsv4KVCacheEstimateCost& cache_cost,
                               int64_t common_blocks_per_unit) {
  CHECK_GT(cache_cost.manager_blocks_per_unit, 0);
  CHECK_EQ(common_blocks_per_unit % cache_cost.manager_blocks_per_unit, 0);
  const int64_t compressed_units =
      common_blocks_per_unit / cache_cost.manager_blocks_per_unit;
  return compressed_units * cache_cost.token_unit_bytes;
}

// SWA ring sizing shared by every DSV4-family estimator: per-sequence window
// blocks, one prefill burst, one per-sequence tail block, and two guard
// blocks.
int64_t dsv4_swa_block_count(const ModelArgs& model_args,
                             const KVCacheEstimateOptions& options) {
  const int64_t block_size = options.block_size;
  const int64_t swa_blocks_per_seq =
      get_swa_blocks_per_seq(model_args.window_size(), block_size);
  const int64_t max_seqs =
      std::max(options.max_seqs_per_batch, static_cast<int64_t>(1));
  int64_t burst_budget =
      std::max(options.max_tokens_per_batch, static_cast<int64_t>(0));
  bool shrunk_to_per_group_share = false;
  if (options.enable_dp_fair_token_budget && options.dp_size > 1 &&
      options.instance_role == InstanceRole::PREFILL) {
    // The scheduler caps each DP group at max_tokens_per_batch / dp_size
    // tokens per scheduling round, floored at one prefill chunk, so the
    // prefill burst landing on any single rank is bounded by the same
    // expression. Keep this identical to the scheduler-side cap or the ring
    // under-reserves.
    int64_t per_group_cap =
        (burst_budget + options.dp_size - 1) / options.dp_size;
    if (options.enable_chunked_prefill) {
      per_group_cap =
          std::max(per_group_cap,
                   std::min<int64_t>(options.max_tokens_per_chunk_for_prefill,
                                     burst_budget));
    } else {
      per_group_cap = burst_budget;
    }
    burst_budget = per_group_cap;
    shrunk_to_per_group_share = true;
  }
  // The SWA ring backs two independent demands and has to be sized for both:
  //   (a) the live prefill burst, which the per-DP fair budget above bounds;
  //   (b) the rows of a published prefix-cache checkpoint, because the
  //       composite publishes prefix entries in C128-sized units and a unit is
  //       only restorable while the SWA rows at that checkpoint are still
  //       resident (trim_swa_compressed discards a checkpoint whose rows were
  //       recycled).
  // Sizing for (a) alone starves (b) as soon as the per-DP share drops below
  // one publish unit: the ring then recycles checkpoint rows before the next
  // request probes them, every cross-request prefix hit degrades to a miss,
  // and the request silently re-prefills its whole prompt. Keep at least one
  // publish unit worth of SWA rows alive whenever one can be published. The
  // unit is the largest compress ratio, i.e. the number of base blocks the
  // composite stages per published C128 checkpoint.
  const std::vector<int32_t>& compress_ratios = model_args.compress_ratios();
  int64_t publish_unit_blocks = 0;
  if (shrunk_to_per_group_share && options.enable_prefix_cache) {
    for (const int32_t ratio : compress_ratios) {
      if (ratio > 1) {
        publish_unit_blocks =
            std::max(publish_unit_blocks, static_cast<int64_t>(ratio));
      }
    }
  }
  const int64_t burst_blocks =
      std::max(util::ceil_div(burst_budget, block_size), publish_unit_blocks);
  return swa_blocks_per_seq * max_seqs + burst_blocks + max_seqs + 2;
}

void set_dsv4_compressed_counts(const Dsv4KVCacheEstimateCost& cache_cost,
                                int64_t token_unit_count,
                                KVCacheCapacity* kv_cache_cap) {
  CHECK(kv_cache_cap != nullptr);
  if (cache_cost.n_c2_layers > 0 || cache_cost.n_c1_layers > 0) {
    // V4.1: one allocation unit spans manager_blocks_per_unit engine blocks;
    // each group consumes manager_blocks_per_unit / ratio blocks per unit
    // (ratio 2 -> 1 c2 block, ratio 1 -> 2 c1 blocks).
    if (cache_cost.n_c2_layers > 0) {
      kv_cache_cap->c2_count(token_unit_count *
                             cache_cost.manager_blocks_per_unit / 2);
    }
    if (cache_cost.n_c1_layers > 0) {
      kv_cache_cap->c1_count(token_unit_count *
                             cache_cost.manager_blocks_per_unit / 1);
    }
    return;
  }
  if (cache_cost.n_c4_layers > 0) {
    kv_cache_cap->c4_count(token_unit_count);
  }
  if (cache_cost.n_c128_layers > 0) {
    kv_cache_cap->c128_count(token_unit_count);
  }
}

Dsv4KVCacheEstimateCost estimate_dsv4_kv_cache_cost(
    const ModelArgs& model_args,
    const KVCacheEstimateOptions& options) {
  const int64_t block_size = options.block_size;
  const int64_t head_dim = model_args.head_dim();
  const int64_t index_head_dim =
      std::max<int64_t>(model_args.index_head_dim(), 1);
  const std::vector<int32_t>& compress_ratios = model_args.compress_ratios();
  const int64_t float32_size = 4;
  const int64_t dtype_size =
      static_cast<int64_t>(torch::elementSize(options.dtype));
  const Dsv4CacheGeometry& geometry =
      KVCacheConfig::get_instance().dsv4_cache_geometry();
  const int64_t compressed_block_token_size =
      geometry.compressed_block_token_size();
  const int64_t c4_physical_dim = geometry.c4_physical_dim();
  const int64_t c128_physical_dim = geometry.c128_physical_dim();

  Dsv4KVCacheEstimateCost cache_cost;
  cache_cost.swa_count = dsv4_swa_block_count(model_args, options);

  for (int64_t i = 0; i < model_args.n_layers(); ++i) {
    const int32_t ratio = i < static_cast<int64_t>(compress_ratios.size())
                              ? compress_ratios[static_cast<size_t>(i)]
                              : 1;
    if (ratio == 4) {
      ++cache_cost.n_c4_layers;
    } else if (ratio == 128) {
      ++cache_cost.n_c128_layers;
    }
  }
  const int64_t n_c1_layers =
      model_args.n_layers() - cache_cost.n_c4_layers - cache_cost.n_c128_layers;

  const int64_t swa_bytes_per_c1_block = block_size * head_dim * dtype_size;
  const int64_t swa_bytes_per_c4_block =
      block_size * head_dim * dtype_size +
      block_size * (2 * head_dim * float32_size) * 2 +
      block_size * (2 * index_head_dim * float32_size) * 2;
  const int64_t swa_bytes_per_c128_block =
      block_size * head_dim * dtype_size +
      block_size * head_dim * float32_size * 2;

  cache_cost.swa_bytes_per_block =
      n_c1_layers * swa_bytes_per_c1_block +
      cache_cost.n_c4_layers * swa_bytes_per_c4_block +
      cache_cost.n_c128_layers * swa_bytes_per_c128_block;
  cache_cost.constant_swa_bytes =
      cache_cost.swa_count * cache_cost.swa_bytes_per_block;

  const DeepSeekV4CachePolicy cache_policy =
      get_dsv4_cache_policy(options.dtype);
  const int64_t scale_bytes =
      cache_policy.has_indexer_cache_scale ? cache_policy.scale_dtype_size : 0;
  const int64_t bytes_per_c4_block =
      c4_physical_dim *
      (head_dim * dtype_size + index_head_dim * cache_policy.index_dtype_size +
       scale_bytes);
  const int64_t bytes_per_c128_block =
      c128_physical_dim * head_dim * dtype_size;
  const int64_t manager_blocks_per_compressed_block =
      compressed_block_token_size / block_size;

  if (cache_cost.n_c4_layers > 0 && cache_cost.n_c128_layers > 0) {
    cache_cost.token_unit_bytes =
        cache_cost.n_c4_layers * bytes_per_c4_block +
        cache_cost.n_c128_layers * bytes_per_c128_block;
    cache_cost.manager_blocks_per_unit = manager_blocks_per_compressed_block;
  } else if (cache_cost.n_c4_layers > 0) {
    cache_cost.token_unit_bytes = cache_cost.n_c4_layers * bytes_per_c4_block;
    cache_cost.manager_blocks_per_unit = manager_blocks_per_compressed_block;
  } else if (cache_cost.n_c128_layers > 0) {
    cache_cost.token_unit_bytes =
        cache_cost.n_c128_layers * bytes_per_c128_block;
    cache_cost.manager_blocks_per_unit = manager_blocks_per_compressed_block;
  }
  return cache_cost;
}

// DeepSeek V4.1 cost model (contract §2). Only kv-source layers allocate
// TOKEN group tensors; reuse/reindex layers read them, so the compressed
// budget is driven solely by the kv-source layers.
//
// Derivation of the per-block byte costs (bs = engine block_size tokens):
// - Every layer stores its own SWA window KV: bs * head_dim * dtype_size.
// - A ratio-2 kv source additionally parks the trailing partial group in
//   compress_kv_state / compress_score_state, each [swa_count, bs, head_dim]
//   fp32 (ring slot = pos % bs), adding 2 * bs * head_dim * 4 bytes to the
//   SWA-group per-block cost.
// - A TOKEN block stores bs compressed rows of key [head_dim] plus index
//   [index_head_dim], both in the MODEL dtype (the V4.1 indexer cache is
//   bf16, unlike V4's int8 index cache), so
//   bytes_per_token_block = bs * (head_dim + index_head_dim) * dtype_size
//   for both ratios.
//
// Allocation unit: one c2 block spans bs*2 tokens = 2 engine blocks, one c1
// block spans bs*1 tokens = 1 engine block, so the common unit is
// manager_blocks_per_unit = 2 engine blocks (bs*2 tokens): per unit a ratio-2
// source consumes 1 c2 block and a ratio-1 source consumes 2 c1 blocks.
// This keeps c2_count : c1_count at the token-proportional 1 : 2 and makes
// n_blocks = max(c2_count * 2, c1_count) land on both managers evenly.
Dsv4KVCacheEstimateCost estimate_dsv41_kv_cache_cost(
    const ModelArgs& model_args,
    const KVCacheEstimateOptions& options) {
  const int64_t block_size = options.block_size;
  const int64_t head_dim = model_args.head_dim();
  const int64_t index_head_dim =
      std::max<int64_t>(model_args.index_head_dim(), 1);
  const int64_t float32_size = 4;
  const int64_t dtype_size =
      static_cast<int64_t>(torch::elementSize(options.dtype));

  Dsv4KVCacheEstimateCost cache_cost;
  cache_cost.swa_count = dsv4_swa_block_count(model_args, options);

  const std::vector<int32_t>& compress_ratios = model_args.compress_ratios();
  for (const int32_t layer_id : model_args.kv_source_layer_ids()) {
    if (layer_id < 0 ||
        layer_id >= static_cast<int64_t>(compress_ratios.size())) {
      continue;
    }
    const int32_t ratio = compress_ratios[static_cast<size_t>(layer_id)];
    if (ratio == 2) {
      ++cache_cost.n_c2_layers;
    } else if (ratio == 1) {
      ++cache_cost.n_c1_layers;
    }
  }

  const int64_t swa_bytes_per_plain_block = block_size * head_dim * dtype_size;
  const int64_t swa_bytes_per_c2_block =
      swa_bytes_per_plain_block + 2 * block_size * head_dim * float32_size;
  cache_cost.swa_bytes_per_block =
      (model_args.n_layers() - cache_cost.n_c2_layers) *
          swa_bytes_per_plain_block +
      cache_cost.n_c2_layers * swa_bytes_per_c2_block;
  cache_cost.constant_swa_bytes =
      cache_cost.swa_count * cache_cost.swa_bytes_per_block;

  const int64_t bytes_per_token_block =
      block_size * (head_dim + index_head_dim) * dtype_size;
  cache_cost.manager_blocks_per_unit = 2;
  cache_cost.token_unit_bytes =
      cache_cost.n_c2_layers * (cache_cost.manager_blocks_per_unit / 2) *
          bytes_per_token_block +
      cache_cost.n_c1_layers * (cache_cost.manager_blocks_per_unit / 1) *
          bytes_per_token_block;
  return cache_cost;
}

Dsv4KVCacheEstimateCost estimate_dsv4_family_kv_cache_cost(
    const ModelArgs& model_args,
    const KVCacheEstimateOptions& options) {
  if (util::is_deepseek_v41_model_type(model_args.model_type())) {
    return estimate_dsv41_kv_cache_cost(model_args, options);
  }
  return estimate_dsv4_kv_cache_cost(model_args, options);
}

void init_dsv4_counts(const ModelArgs& model_args,
                      const KVCacheEstimateOptions& options,
                      KVCacheCapacity* kv_cache_cap) {
  CHECK(kv_cache_cap != nullptr);
  const Dsv4KVCacheEstimateCost cache_cost =
      estimate_dsv4_family_kv_cache_cost(model_args, options);
  CHECK_GE(kv_cache_cap->cache_size_in_bytes(), cache_cost.constant_swa_bytes)
      << "no memory for the minimum DSV4 SWA cache, required="
      << readable_size(cache_cost.constant_swa_bytes)
      << ", available=" << readable_size(kv_cache_cap->cache_size_in_bytes());
  int64_t token_mem =
      kv_cache_cap->cache_size_in_bytes() - cache_cost.constant_swa_bytes;
  if (options.draft_model_args != nullptr) {
    CHECK(options.draft_options != nullptr)
        << "DSV4 draft options must be provided with draft model args";
    CHECK(
        util::is_deepseek_v4_model_type(options.draft_model_args->model_type()))
        << "DSV4 speculative kv cache estimation only supports DeepSeek V4 "
           "draft";
    const Dsv4KVCacheEstimateCost draft_cost =
        estimate_dsv4_family_kv_cache_cost(*options.draft_model_args,
                                           *options.draft_options);
    const int64_t constant_bytes =
        cache_cost.constant_swa_bytes + draft_cost.constant_swa_bytes;
    CHECK_GE(kv_cache_cap->cache_size_in_bytes(), constant_bytes)
        << "no memory left for speculative target/draft fixed kv cache "
           "allocation";

    const int64_t common_blocks_per_unit = std::max(
        cache_cost.manager_blocks_per_unit, draft_cost.manager_blocks_per_unit);
    const int64_t target_common_unit_bytes =
        dsv4_common_unit_bytes(cache_cost, common_blocks_per_unit);
    const int64_t draft_common_unit_bytes =
        dsv4_common_unit_bytes(draft_cost, common_blocks_per_unit);
    const int64_t combined_unit_bytes =
        target_common_unit_bytes + draft_common_unit_bytes;
    const int64_t remaining_bytes =
        kv_cache_cap->cache_size_in_bytes() - constant_bytes;
    if (combined_unit_bytes > 0) {
      CHECK_GE(remaining_bytes, combined_unit_bytes)
          << "minimum DSV4 target/draft SWA caches leave insufficient memory "
             "for one compressed cache unit, swa_required="
          << readable_size(constant_bytes)
          << ", compressed_unit_required=" << readable_size(combined_unit_bytes)
          << ", available="
          << readable_size(kv_cache_cap->cache_size_in_bytes());
    }
    const int64_t token_unit_count =
        combined_unit_bytes > 0 ? remaining_bytes / combined_unit_bytes : 0;

    const int64_t adjusted_cache_size_in_bytes =
        cache_cost.constant_swa_bytes +
        token_unit_count * target_common_unit_bytes;
    CHECK_GT(adjusted_cache_size_in_bytes, 0)
        << "no memory left for speculative target/draft kv cache allocation";
    LOG(INFO) << "speculative kv cache capacity adjusted from "
              << readable_size(kv_cache_cap->cache_size_in_bytes()) << " to "
              << readable_size(adjusted_cache_size_in_bytes)
              << ", target_constant_bytes=" << cache_cost.constant_swa_bytes
              << ", draft_constant_bytes=" << draft_cost.constant_swa_bytes
              << ", target_common_unit_bytes=" << target_common_unit_bytes
              << ", draft_common_unit_bytes=" << draft_common_unit_bytes
              << ", common_blocks_per_unit=" << common_blocks_per_unit
              << ", token_unit_count=" << token_unit_count;
    kv_cache_cap->cache_size_in_bytes(adjusted_cache_size_in_bytes);
    token_mem = token_unit_count * target_common_unit_bytes;
  } else {
    CHECK(options.draft_options == nullptr)
        << "DSV4 draft options require draft model args";
    if (cache_cost.token_unit_bytes > 0) {
      CHECK_GE(token_mem, cache_cost.token_unit_bytes)
          << "minimum DSV4 SWA cache leaves insufficient memory for one "
             "compressed cache unit, swa_required="
          << readable_size(cache_cost.constant_swa_bytes)
          << ", compressed_unit_required="
          << readable_size(cache_cost.token_unit_bytes) << ", available="
          << readable_size(kv_cache_cap->cache_size_in_bytes());
    }
  }

  kv_cache_cap->swa_count(cache_cost.swa_count);
  kv_cache_cap->c4_count(0);
  kv_cache_cap->c128_count(0);
  kv_cache_cap->c2_count(0);
  kv_cache_cap->c1_count(0);
  // Keep SWA at the operational minimum calculated above. Prefix-cache entries
  // share this pool; any remaining memory is reserved for compressed history.
  if (cache_cost.token_unit_bytes > 0 && token_mem > 0) {
    const int64_t token_unit_count = token_mem / cache_cost.token_unit_bytes;
    set_dsv4_compressed_counts(cache_cost, token_unit_count, kv_cache_cap);
  }

  CHECK_GT(kv_cache_cap->swa_count(), 0) << "DSV4 swa_count must be > 0";
  if (cache_cost.n_c4_layers > 0) {
    CHECK_GT(kv_cache_cap->c4_count(), 0)
        << "DSV4 c4_count must be > 0 when compress_ratio=4 layers exist";
  }
  if (cache_cost.n_c128_layers > 0) {
    CHECK_GT(kv_cache_cap->c128_count(), 0)
        << "DSV4 c128_count must be > 0 when compress_ratio=128 layers "
           "exist";
  }
  if (cache_cost.n_c2_layers > 0) {
    CHECK_GT(kv_cache_cap->c2_count(), 0)
        << "DSV4.1 c2_count must be > 0 when ratio-2 kv-source layers exist";
  }
  if (cache_cost.n_c1_layers > 0) {
    CHECK_GT(kv_cache_cap->c1_count(), 0)
        << "DSV4.1 c1_count must be > 0 when ratio-1 kv-source layers exist";
  }

  int64_t manager_base_blocks = 0;
  if (cache_cost.n_c4_layers > 0) {
    manager_base_blocks =
        std::max(manager_base_blocks,
                 kv_cache_cap->c4_count() * cache_cost.manager_blocks_per_unit);
  }
  if (cache_cost.n_c128_layers > 0) {
    manager_base_blocks = std::max(
        manager_base_blocks,
        kv_cache_cap->c128_count() * cache_cost.manager_blocks_per_unit);
  }
  // V4.1 TOKEN groups: a c2 block spans 2 engine blocks (bs*2 tokens), a c1
  // block spans 1 (bs tokens). n_blocks is in engine-block units, matching
  // the V4 managers above; the composite leaf divides by its ratio.
  if (cache_cost.n_c2_layers > 0) {
    manager_base_blocks =
        std::max(manager_base_blocks, kv_cache_cap->c2_count() * 2);
  }
  if (cache_cost.n_c1_layers > 0) {
    manager_base_blocks =
        std::max(manager_base_blocks, kv_cache_cap->c1_count() * 1);
  }
  kv_cache_cap->n_blocks(std::max<int64_t>(manager_base_blocks, 1));
}

void init_standard_counts(const ModelArgs& model_args,
                          const KVCacheEstimateOptions& options,
                          KVCacheCapacity* kv_cache_cap) {
  for (int64_t layer_id = 0; layer_id < kv_cache_cap->n_layers(); ++layer_id) {
    if (is_full_attention_layer(model_args, layer_id)) {
      ++kv_cache_cap->num_full_attention_layers();
    } else {
      ++kv_cache_cap->num_linear_attention_layers();
    }
  }

  if (kv_cache_cap->index_slot_size() > 0) {
    int64_t num_indexer_layers = kv_cache_cap->num_full_attention_layers();
    const std::vector<bool> indexer_layer_mask =
        resolve_indexer_cache_enabled_layers(model_args,
                                             kv_cache_cap->n_layers());
    if (!indexer_layer_mask.empty()) {
      num_indexer_layers = static_cast<int64_t>(std::count(
          indexer_layer_mask.begin(), indexer_layer_mask.end(), true));
    }
    kv_cache_cap->num_indexer_layers(num_indexer_layers);
  }

  const int64_t full_cache_block_size_in_bytes =
      standard_full_cache_block_size_in_bytes(*kv_cache_cap);
  const int64_t gdn_state_bytes = kv_cache_cap->num_linear_attention_layers() *
                                  kv_cache_cap->linear_slot_size();
  const int64_t kpool_state_bytes =
      kv_cache_cap->num_indexer_layers() * kv_cache_cap->kpool_tail_slot_size();
  const int64_t total_state_bytes = gdn_state_bytes + kpool_state_bytes;
  int64_t num_linear_state_blocks =
      calculate_linear_state_blocks(kv_cache_cap->cache_size_in_bytes(),
                                    total_state_bytes,
                                    full_cache_block_size_in_bytes,
                                    options.max_linear_state_cache_slots);
#if defined(USE_NPU)
  if (is_qwen3_5_target_model_type(model_args.model_type()) &&
      kv_cache_cap->num_linear_attention_layers() > 0) {
    constexpr int64_t kMegaGdnMaxLinearStateBlocks = 1024;
    CHECK_LE(options.max_linear_state_cache_slots + kPaddingLinearStateBlocks,
             kMegaGdnMaxLinearStateBlocks)
        << "Qwen3.5 MegaGdn supports at most " << kMegaGdnMaxLinearStateBlocks
        << " physical linear-state slots, "
        << "including " << kPaddingLinearStateBlocks << " padding slots.";
    num_linear_state_blocks =
        std::min(num_linear_state_blocks, kMegaGdnMaxLinearStateBlocks);
  }
#endif
  kv_cache_cap->num_linear_state_blocks(num_linear_state_blocks);
  kv_cache_cap->linear_cache_size_in_bytes(
      kv_cache_cap->num_linear_state_blocks() * total_state_bytes);
  const int64_t available_full_cache_size_in_bytes =
      kv_cache_cap->cache_size_in_bytes() -
      kv_cache_cap->linear_cache_size_in_bytes();
  if (total_state_bytes > 0) {
    CHECK_GT(kv_cache_cap->cache_size_in_bytes(),
             kv_cache_cap->linear_cache_size_in_bytes())
        << "failed to reserve linear state cache for linear-attention "
           "layers: "
        << "max_seqs_per_batch (" << options.max_seqs_per_batch
        << ") is too large. Please reduce max_seqs_per_batch to less than "
        << kv_cache_cap->cache_size_in_bytes() / total_state_bytes - 2;
  }
  CHECK_GT(available_full_cache_size_in_bytes, 0)
      << "no memory left for full-attention kv cache after reserving linear "
         "state cache";
  if (options.layerwise_split_size > 1) {
    kv_cache_cap->n_blocks(
        layerwise_split_block_count(model_args,
                                    options.layerwise_split_size,
                                    *kv_cache_cap,
                                    available_full_cache_size_in_bytes,
                                    /*additional_block_bytes=*/0));
  } else {
    kv_cache_cap->n_blocks(available_full_cache_size_in_bytes /
                           full_cache_block_size_in_bytes);
  }
  CHECK_GT(kv_cache_cap->n_blocks(), 0) << "no n_blocks for kv cache";
}

}  // namespace

int64_t estimate_shared_kpool_blocks(const KVCacheCapacity& target_cap,
                                     const KVCacheCapacity& draft_cap,
                                     const ModelArgs& draft_args) {
  CHECK_EQ(target_cap.block_size(), draft_cap.block_size());
  CHECK_GT(target_cap.kpool_tail_slot_size(), 0);
  CHECK_GT(draft_cap.kpool_tail_slot_size(), 0);
  CHECK_EQ(draft_cap.num_linear_attention_layers(), 0);
  // Draft request slots and the verify window are inherited from the target
  // shape, rather than from the draft's standalone capacity estimate.
  const int64_t draft_tail_len =
      std::max<int64_t>(target_cap.kpool_tail_len(), draft_args.index_kpool());
  const int64_t draft_state_bytes =
      draft_cap.num_indexer_layers() * target_cap.num_linear_state_blocks() *
      2 * static_cast<int64_t>(sizeof(uint16_t)) * draft_tail_len *
      draft_args.index_head_dim();
  const int64_t state_bytes =
      target_cap.linear_cache_size_in_bytes() + draft_state_bytes;
  const int64_t budget = std::min(target_cap.cache_size_in_bytes(),
                                  draft_cap.cache_size_in_bytes());
  CHECK_GT(budget, state_bytes)
      << "no memory left after reserving target and draft KPool request state";
  const int64_t block_bytes =
      standard_full_cache_block_size_in_bytes(target_cap) +
      standard_full_cache_block_size_in_bytes(draft_cap);
  const int64_t n_blocks = (budget - state_bytes) / block_bytes;
  CHECK_GT(n_blocks, 0) << "no memory for a shared KPool block";
  return n_blocks;
}

std::vector<bool> resolve_indexer_cache_enabled_layers(
    const ModelArgs& model_args,
    int64_t num_cache_layers) {
  if (!Platform::supports_dsa_indexer_cache_elision()) {
    return {};
  }
  std::vector<bool> layer_mask =
      layer::get_dsa_indexer_layer_mask(model_args, num_cache_layers);
  if (layer_mask.empty()) {
    return layer_mask;
  }
  for (int64_t layer_id = 0; layer_id < num_cache_layers; ++layer_id) {
    if (!is_full_attention_layer(model_args, layer_id)) {
      layer_mask[static_cast<size_t>(layer_id)] = false;
    }
  }
  return layer_mask;
}

std::vector<bool> build_layer_cache_owned(const ModelArgs& model_args,
                                          const LayerwiseSplitLayout& layout,
                                          int64_t num_layers) {
  std::vector<bool> layer_cache_owned;
  layer_cache_owned.reserve(static_cast<size_t>(num_layers));
  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    layer_cache_owned.emplace_back(
        !is_full_attention_layer(model_args, layer_id) ||
        layout.owns(layer_id));
  }
  return layer_cache_owned;
}

int64_t estimate_layerwise_split_block_count(
    const ModelArgs& model_args,
    int32_t layerwise_split_size,
    const KVCacheCapacity& kv_cache_cap,
    int64_t available_bytes,
    int64_t additional_block_bytes) {
  return layerwise_split_block_count(model_args,
                                     layerwise_split_size,
                                     kv_cache_cap,
                                     available_bytes,
                                     additional_block_bytes);
}

KVCacheCapacity estimate_kv_cache_capacity(
    const ModelArgs& model_args,
    const KVCacheEstimateOptions& options) {
  KVCacheCapacity kv_cache_cap;
  kv_cache_cap
      .cache_size_in_bytes(
          std::max(options.cache_size_in_bytes, static_cast<int64_t>(0)))
      .block_size(options.block_size);
  CHECK_GT(kv_cache_cap.cache_size_in_bytes(), 0)
      << "Available kv cache size must be greater than 0";
  const bool enable_dsv4_estimation =
      util::is_deepseek_v4_model_type(model_args.model_type());
  if (options.draft_model_args != nullptr) {
    CHECK(enable_dsv4_estimation)
        << "DSV4 speculative kv cache estimation only supports DeepSeek V4 "
           "target";
  }

  const int64_t dtype_size = static_cast<int64_t>(
      torch::scalarTypeToTypeMeta(options.dtype).itemsize());
  const int64_t cache_dtype_size =
      kv_cache_dtype_size(options.kv_cache_dtype, dtype_size);
  const bool enable_indexer_cache_quantization =
      options.indexer_cache_dtype == "int8";
  KPoolCacheLayout kpool_layout = options.kpool_layout;
  if (Platform::is_npu()) {
    kpool_layout = uses_npu_compressed_kpool_tail(model_args)
                       ? KPoolCacheLayout::COMPRESSED_WITH_TAIL
                       : KPoolCacheLayout::PACKED;
  }
  // Estimation is the only site that also honors the runtime enable_mla
  // config bit (every other caller has already committed to MLA via model
  // wiring), so AND it in here rather than pushing it into the helper.
  const bool enable_mla_kv_cache_quantization =
      model_args.enable_mla() &&
      util::enable_mla_packed_c8(options.kv_cache_dtype == "int8",
                                 model_args.model_type());
  const IndexCacheFootprint index_cache =
      estimate_index_cache_footprint(model_args,
                                     enable_indexer_cache_quantization,
                                     dtype_size,
                                     options.block_size,
                                     kpool_layout);

  kv_cache_cap.slot_size(kv_slot_size(model_args, options, cache_dtype_size))
      .index_cache_sizes(index_cache.slot_size, index_cache.block_size)
      .kpool_layout(kpool_layout)
      .enable_indexer_cache_quant(enable_indexer_cache_quantization)
      .enable_mla_kv_cache_quant(enable_mla_kv_cache_quantization)
      .scale_slot_size(scale_slot_size(model_args, options))
      .linear_slot_size(linear_slot_size(model_args, options, dtype_size))
      .n_layers(model_args.n_layers())
      .block_size(options.block_size);
  const int64_t num_speculative_tokens =
      enable_hybrid_spec_verify(model_args, options)
          ? effective_num_speculative_tokens(options)
          : 0;
  kv_cache_cap.linear_conv_state_len(model_args.linear_conv_kernel_dim() - 1 +
                                     num_speculative_tokens);
  kv_cache_cap.linear_ssm_checkpoint_stride(num_speculative_tokens + 1);
  if (options.is_draft_engine && model_args.num_nextn_predict_layers() > 0) {
    kv_cache_cap.n_layers(model_args.num_nextn_predict_layers());
  }

  if (model_args.index_kpool_compress() &&
      kpool_layout == KPoolCacheLayout::COMPRESSED_WITH_TAIL) {
    CHECK_EQ(options.layerwise_split_size, 1)
        << "Compressed KPool requires owned layer caches.";
    CHECK_EQ(options.dtype, torch::kBFloat16);
    CHECK_EQ(options.block_size % model_args.index_kpool(), 0);
    const int64_t num_speculative_tokens =
        effective_num_speculative_tokens(options);
    CHECK_GE(num_speculative_tokens, 0);
    if (uses_npu_compressed_kpool_tail(model_args)) {
      // Preserve one complete pool plus speculative rows that may be rejected.
      // The verify base token advances the committed sequence and is already
      // covered by the regular pool tail.
      kv_cache_cap.kpool_tail_len(model_args.index_kpool() +
                                  num_speculative_tokens);
    } else {
      // Preserve the MLU asynchronous handoff window.
      const int64_t window =
          num_speculative_tokens > 0 ? 2 * (num_speculative_tokens + 1) : 0;
      kv_cache_cap.kpool_tail_len(model_args.index_kpool() + window);
    }
    kv_cache_cap.kpool_tail_slot_size(2 * sizeof(uint16_t) *
                                      kv_cache_cap.kpool_tail_len() *
                                      model_args.index_head_dim());
  }

  if (enable_dsv4_estimation) {
    init_dsv4_counts(model_args, options, &kv_cache_cap);
  } else {
    init_standard_counts(model_args, options, &kv_cache_cap);
  }
  return kv_cache_cap;
}

}  // namespace xllm
