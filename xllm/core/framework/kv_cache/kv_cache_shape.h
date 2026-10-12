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

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_args.h"

namespace xllm {

namespace proto {
class KVCacheShape;
}

// SFA C8 packed-row constants. One physical row per token holds
// [int8 nope | bf16 rope | fp32 per-tile scale]; kernel fixes tile=128 and
// dequant scale as fp32. Shared by kv_cache_shape.cpp, kv_cache_estimation.cpp,
// and the Python write path so the layout stays byte-identical everywhere.
struct MlaPackedC8Layout {
  static constexpr int64_t kTileSize = 128;         // fixed by kernel
  static constexpr int64_t kRopeElementBytes = 2;   // bf16
  static constexpr int64_t kScaleElementBytes = 4;  // fp32
};

class KVCacheShape final {
 public:
  KVCacheShape() = default;

  KVCacheShape(const KVCacheCapacity& kv_cache_cap,
               const ModelArgs& model_args,
               int64_t world_size);

  const std::vector<int64_t>& key_cache_shape() const;
  const std::vector<int64_t>& value_cache_shape() const;
  const std::vector<int64_t>& index_cache_shape() const;
  const std::vector<int64_t>& kpool_tail_shape() const;
  bool has_kpool_tail_shape() const;
  KPoolCacheLayout kpool_layout() const { return kpool_layout_; }
  const std::vector<int64_t>& index_cache_scale_shape() const;
  const std::vector<int64_t>& conv_cache_shape() const;
  const std::vector<int64_t>& ssm_cache_shape() const;

  bool has_key_cache_shape() const;
  bool has_value_cache_shape() const;
  bool has_index_cache_shape() const;
  bool has_index_cache_scale_shape() const;
  bool has_conv_cache_shape() const;
  bool has_ssm_cache_shape() const;
  int64_t linear_ssm_checkpoint_stride() const;
  // Physical pages one logical block owns in this pool. 1 = the pool follows
  // the target's kv_split sharding; > 1 = the pool is replicated on every rank
  // and its slots are addressed by the logical token index.
  int64_t replicated_block_pages() const { return replicated_block_pages_; }
  bool has_grouped_cache_layout() const {
    return shape_kind_ == ShapeKind::GROUPED_POOL;
  }

  void print_shapes() const;

  void to_proto(proto::KVCacheShape* proto_shape) const;
  static KVCacheShape from_proto(const proto::KVCacheShape& proto_shape);

 private:
  enum class ShapeKind : int8_t {
    NORMAL = 0,
    GROUPED_POOL = 1,
  };

  void init_dsv4_pool_shape(const KVCacheCapacity& kv_cache_cap,
                            const ModelArgs& model_args);
  void init_key_cache_shape(const KVCacheCapacity& kv_cache_cap,
                            const ModelArgs& model_args,
                            int64_t world_size);
  void init_value_cache_shape(const KVCacheCapacity& kv_cache_cap,
                              const ModelArgs& model_args,
                              int64_t world_size);
  void init_index_cache_shape(const KVCacheCapacity& kv_cache_cap,
                              const ModelArgs& model_args);
  void init_mla_packed_c8_shape(const KVCacheCapacity& kv_cache_cap,
                                const ModelArgs& model_args);
  void init_index_cache_scale_shape();
  void init_conv_cache_shape(const KVCacheCapacity& kv_cache_cap,
                             const ModelArgs& model_args,
                             int64_t world_size);
  void init_ssm_cache_shape(const KVCacheCapacity& kv_cache_cap,
                            const ModelArgs& model_args,
                            int64_t world_size);
  void apply_device_layout(const ModelArgs& model_args);
  void print_dsv4_pool_shape() const;

  static const std::vector<int64_t>& empty_shape();

 private:
  KPoolCacheLayout kpool_layout_ = KPoolCacheLayout::PACKED;
  int64_t replicated_block_pages_ = 1;
  std::optional<std::vector<int64_t>> kpool_tail_shape_;
  ShapeKind shape_kind_ = ShapeKind::NORMAL;
  // True when the grouped pool slots carry the V4.1 meaning
  // [swa_count, c2_count, c1_count] instead of V4's
  // [swa_count, c4_count, c128_count].
  bool dsv41_pool_ = false;
  std::optional<std::vector<int64_t>> key_cache_shape_;
  std::optional<std::vector<int64_t>> value_cache_shape_;

  // for index cache
  std::optional<std::vector<int64_t>> index_cache_shape_;
  std::optional<std::vector<int64_t>> index_cache_scale_shape_;

  // for linear attention
  std::optional<std::vector<int64_t>> conv_cache_shape_;
  std::optional<std::vector<int64_t>> ssm_cache_shape_;
};

// Returns the cache layout requested by the model, including draft-specific
// layouts. Keeping this decision with cache geometry prevents workers from
// interpreting model attention semantics.
KVCacheCreateOptions get_kv_cache_create_options(const ModelArgs& model_args,
                                                 const KVCacheShape& shape,
                                                 bool is_spec_draft);

}  // namespace xllm
