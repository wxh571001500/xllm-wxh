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

// Host KV transfer under kv_split: the target INDEX cache (one logical block
// spans kv_split replicated pages on every NPU rank) and a replicated draft
// pool (replicated_block_pages > 1) must both fold into "one dim0 row per
// logical block" so the host transfer's block-id-as-row contract holds.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/config/parallel_config.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_capacity.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"
#include "framework/model/model_args.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace {

constexpr int32_t kKvSplitSize = 4;
constexpr int64_t kBlockCount = 2;
constexpr int64_t kBlockSize = 4;
constexpr int64_t kLayerCount = 2;

// Keeps ParallelConfig::kv_split_size pinned while a KVCacheShape and its
// transfer layout are built, then restores the previous value. The shape init
// (init_index_cache_shape) and the fold helpers both read
// indexer_pages_per_block(), so they must observe the same value.
class KvSplitSizeGuard final {
 public:
  explicit KvSplitSizeGuard(int32_t kv_split_size) {
    old_value_ = ParallelConfig::get_instance().kv_split_size();
    ParallelConfig::get_instance().kv_split_size(kv_split_size);
  }

  ~KvSplitSizeGuard() {
    ParallelConfig::get_instance().kv_split_size(old_value_);
  }

  KvSplitSizeGuard(const KvSplitSizeGuard&) = delete;
  KvSplitSizeGuard& operator=(const KvSplitSizeGuard&) = delete;

 private:
  int32_t old_value_ = 1;
};

// An indexed target pool: K/V carry one physical row per logical block while
// the INDEX cache carries kv_split replicated rows per logical block.
class IndexedPool final {
 public:
  std::vector<KVCache> caches;
  KVCacheShape cache_shape;
  KVCacheCreateOptions create_options;

  explicit IndexedPool(const Device& device) {
    KVCacheCapacity capacity;
    capacity.n_blocks(kBlockCount).block_size(kBlockSize);
    ModelArgs model_args;
    model_args.model_type("test_model")
        .n_layers(kLayerCount)
        .n_heads(2)
        .n_kv_heads(1)
        .head_dim(8)
        .index_n_heads(4)
        .index_head_dim(6);
    cache_shape = KVCacheShape(capacity, model_args, /*world_size=*/1);
    create_options.device(device.unwrap())
        .dtype(torch::kFloat32)
        .num_layers(kLayerCount)
        .enable_lighting_indexer(true)
        .model_type("test_model");
    allocate_kv_caches(caches, cache_shape, create_options);
  }
};

// A plain full-attention pool. A replicated draft pool sizes n_blocks in
// physical rows (logical_blocks * replicated_block_pages), mirroring
// build_speculative_draft_kv_cache_shape().
class PlainPool final {
 public:
  std::vector<KVCache> caches;
  KVCacheShape cache_shape;
  KVCacheCreateOptions create_options;

  PlainPool(const Device& device,
            int64_t physical_rows,
            int64_t replicated_block_pages) {
    KVCacheCapacity capacity;
    capacity.n_blocks(physical_rows)
        .block_size(kBlockSize)
        .replicated_block_pages(replicated_block_pages);
    ModelArgs model_args;
    model_args.model_type("test_model")
        .n_layers(kLayerCount)
        .n_heads(2)
        .n_kv_heads(1)
        .head_dim(8);
    cache_shape = KVCacheShape(capacity, model_args, /*world_size=*/1);
    create_options.device(device.unwrap())
        .dtype(torch::kFloat32)
        .num_layers(kLayerCount)
        .model_type("test_model");
    allocate_kv_caches(caches, cache_shape, create_options);
  }
};

HierarchyKVCacheTransfer::Options make_transfer_options() {
  HierarchyKVCacheTransfer::Options transfer_options;
  transfer_options.layers(kLayerCount)
      .host_blocks_factor(2.0)
      .layers_wise_copy_batchs(1);
  return transfer_options;
}

void run_kv_round_trip(HierarchyKVCacheTransfer& transfer, uint64_t batch_id) {
  BlockTransferInfo offload_info(/*src_block_id=*/0, /*dst_block_id=*/0);
  offload_info.block_type = BlockType::KV;
  offload_info.transfer_type = TransferType::D2H2G;
  ASSERT_EQ(transfer.transfer_kv_blocks(batch_id, {offload_info}), 1U);

  BlockTransferInfo load_info(/*src_block_id=*/0, /*dst_block_id=*/1);
  load_info.block_type = BlockType::KV;
  load_info.transfer_type = TransferType::H2D;
  ASSERT_EQ(transfer.transfer_kv_blocks(batch_id, {load_info}), 1U);

  ModelInputParams input_params;
  input_params.meta.batch_id = batch_id;
  input_params.meta.requires_host_restore = true;
  transfer.set_layer_synchronizer(input_params);
  ASSERT_TRUE(input_params.synchronize_all_layers());
}

TEST(KvSplitHostTransferTest, BuildsIndexedLayoutUnderKvSplit) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "An accelerator device is required for hierarchy KV transfer.";
  }
  if (!Platform::is_npu()) {
    GTEST_SKIP() << "Indexer cache replication is an NPU-only layout.";
  }
  const KvSplitSizeGuard kv_split_guard(kKvSplitSize);
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  IndexedPool pool(device);

  // The NPU indexer cache replicates every logical block across kv_split
  // ranks: the index tensor carries kBlockCount * kv_split physical rows
  // while the K/V tensors carry one row per logical block.
  ASSERT_TRUE(pool.cache_shape.has_index_cache_shape());
  ASSERT_EQ(pool.cache_shape.index_cache_shape()[0],
            kBlockCount * kKvSplitSize);
  ASSERT_EQ(pool.cache_shape.key_cache_shape()[0], kBlockCount);

  std::unique_ptr<Stream> compute_stream = device.current_stream();
  // Before the fold fix this constructor aborted inside HostKVLayout with
  // "host role block capacities must match" (KEY dim0 != INDEX dim0).
  HierarchyKVCacheTransfer transfer(make_transfer_options(),
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &pool.caches,
                                    pool.cache_shape,
                                    pool.create_options);
  EXPECT_TRUE(transfer.registration_finalized());
  EXPECT_TRUE(transfer.supports_block_type(BlockType::KV));
}

TEST(KvSplitHostTransferTest, RoundTripCopiesAllIndexPagesPerBlock) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "An accelerator device is required for hierarchy KV transfer.";
  }
  if (!Platform::is_npu()) {
    GTEST_SKIP() << "Indexer cache replication is an NPU-only layout.";
  }
  const KvSplitSizeGuard kv_split_guard(kKvSplitSize);
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  IndexedPool pool(device);

  // Fill every physical index page with a page-unique value so a copy that
  // only moves one page (or picks the wrong row) cannot match.
  for (KVCache& cache : pool.caches) {
    const torch::Tensor index_cache = cache.get_index_cache();
    ASSERT_EQ(index_cache.size(0), kBlockCount * kKvSplitSize);
    for (int64_t page = 0; page < index_cache.size(0); ++page) {
      index_cache[page].fill_(100.0 + static_cast<double>(page));
    }
    cache.get_k_cache()[0].fill_(3.0);
    cache.get_k_cache()[1].zero_();
  }
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  std::unique_ptr<Stream> compute_stream = device.current_stream();
  HierarchyKVCacheTransfer transfer(make_transfer_options(),
                                    device.unwrap(),
                                    compute_stream.get(),
                                    &pool.caches,
                                    pool.cache_shape,
                                    pool.create_options);
  run_kv_round_trip(transfer, /*batch_id=*/11);

  for (KVCache& cache : pool.caches) {
    const torch::Tensor index_cache = cache.get_index_cache();
    EXPECT_TRUE(torch::equal(index_cache.narrow(0, 0, kKvSplitSize),
                             index_cache.narrow(0, kKvSplitSize, kKvSplitSize)))
        << "H2D must restore all kv_split index pages of the logical block";
    EXPECT_TRUE(torch::equal(cache.get_k_cache()[0], cache.get_k_cache()[1]));
  }
}

TEST(KvSplitHostTransferTest, RoundTripReplicatedDraftPool) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "An accelerator device is required for hierarchy KV transfer.";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  // Target: plain sharded pool (2 logical blocks). Draft: replicated pool
  // with replicated_block_pages = kv_split, sized in physical rows.
  PlainPool target_pool(device,
                        /*physical_rows=*/kBlockCount,
                        /*replicated_block_pages=*/1);
  PlainPool draft_pool(device,
                       /*physical_rows=*/kBlockCount * kKvSplitSize,
                       /*replicated_block_pages=*/kKvSplitSize);
  ASSERT_EQ(draft_pool.caches[0].get_k_cache().size(0),
            kBlockCount * kKvSplitSize);

  for (KVCache& cache : draft_pool.caches) {
    const torch::Tensor k_cache = cache.get_k_cache();
    for (int64_t row = 0; row < k_cache.size(0); ++row) {
      k_cache[row].fill_(10.0 + static_cast<double>(row));
    }
    const torch::Tensor v_cache = cache.get_v_cache();
    for (int64_t row = 0; row < v_cache.size(0); ++row) {
      v_cache[row].fill_(20.0 + static_cast<double>(row));
    }
  }
  ASSERT_EQ(device.synchronize_default_stream(), 0);

  std::unique_ptr<Stream> compute_stream = device.current_stream();
  HierarchyKVCacheTransfer transfer(make_transfer_options(), device.unwrap());
  HierarchyKVCacheTransfer::CacheRegistration target_registration;
  target_registration.role = HierarchyKVCacheTransfer::CacheRole::TARGET;
  target_registration.device_kv_caches = &target_pool.caches;
  target_registration.kv_cache_shape = target_pool.cache_shape;
  target_registration.create_options = target_pool.create_options;
  target_registration.producer_stream = compute_stream.get();
  target_registration.store_key_component = "main";
  transfer.register_cache(std::move(target_registration));

  HierarchyKVCacheTransfer::CacheRegistration draft_registration;
  draft_registration.role = HierarchyKVCacheTransfer::CacheRole::DRAFT;
  draft_registration.device_kv_caches = &draft_pool.caches;
  draft_registration.kv_cache_shape = draft_pool.cache_shape;
  draft_registration.create_options = draft_pool.create_options;
  draft_registration.producer_stream = compute_stream.get();
  draft_registration.store_key_component = "draft";
  transfer.register_cache(std::move(draft_registration));
  ASSERT_TRUE(transfer.finalize_registration());

  // Block ids stay logical; the draft pool must copy all
  // replicated_block_pages rows of the logical block, not one wrong row.
  run_kv_round_trip(transfer, /*batch_id=*/12);

  for (KVCache& cache : draft_pool.caches) {
    const torch::Tensor k_cache = cache.get_k_cache();
    EXPECT_TRUE(torch::equal(k_cache.narrow(0, 0, kKvSplitSize),
                             k_cache.narrow(0, kKvSplitSize, kKvSplitSize)))
        << "H2D must restore all replicated draft rows of the logical block";
    const torch::Tensor v_cache = cache.get_v_cache();
    EXPECT_TRUE(torch::equal(v_cache.narrow(0, 0, kKvSplitSize),
                             v_cache.narrow(0, kKvSplitSize, kKvSplitSize)))
        << "H2D must restore all replicated draft VALUE rows of the "
           "logical block";
  }
}

#if defined(USE_NPU)
// deepseek_v3 / deepseek_v3_mtp pools with prefix cache enabled are
// allocated in FRACTAL_NZ (get_npu_kv_cache_format). Such a tensor reports
// contiguous logical strides, so the contiguity CHECK cannot catch it and
// only the storage-format guard keeps the folded view out of the transfer
// layout: view() cannot refold the interleaved physical tiling, and every
// host/device copy through the folded view would mis-address rows.
TEST(KvSplitHostTransferTest, FoldRejectsFractalNzCacheTensor) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP()
        << "An accelerator device is required for hierarchy KV transfer.";
  }
  if (!Platform::is_npu()) {
    GTEST_SKIP() << "FRACTAL_NZ pool casting is an NPU-only layout.";
  }
  const KvSplitSizeGuard kv_split_guard(kKvSplitSize);
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();

  const torch::TensorOptions pool_options =
      torch::dtype(torch::kFloat16).device(device.unwrap());
  const torch::Tensor nz_index = at_npu::native::npu_format_cast(
      torch::empty({kBlockCount * kKvSplitSize, kBlockSize * 16}, pool_options),
      ACL_FORMAT_FRACTAL_NZ);
  // The interleaved storage must look contiguous on the logical descriptor:
  // this is exactly why the storage-format guard below is load-bearing.
  ASSERT_TRUE(nz_index.is_contiguous());
  ASSERT_EQ(device.synchronize_default_stream(), 0);
  EXPECT_DEATH(fold_tensor_rows_for_transfer(nz_index,
                                             KVCacheTensorRole::INDEX,
                                             /*replicated_block_pages=*/1),
               "host transfer folding requires");

  // The guard only restricts the folded (pages > 1) path: an ND pool of the
  // same geometry still folds into one dim0 row per logical block.
  const torch::Tensor nd_index =
      torch::empty({kBlockCount * kKvSplitSize, kBlockSize, 8}, pool_options);
  const torch::Tensor folded = fold_tensor_rows_for_transfer(
      nd_index, KVCacheTensorRole::INDEX, /*replicated_block_pages=*/1);
  EXPECT_EQ(folded.size(0), kBlockCount);
  EXPECT_EQ(folded.size(1), kBlockSize * kKvSplitSize);
}
#endif

}  // namespace
}  // namespace xllm
