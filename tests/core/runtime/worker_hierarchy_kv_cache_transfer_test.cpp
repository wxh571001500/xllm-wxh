/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/kv_cache/kv_cache_capacity.h"
#include "core/framework/kv_cache/kv_cache_shape.h"
#include "core/framework/model/model_args.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/runtime/options.h"
#include "core/util/slice.h"

namespace xllm {
namespace {

constexpr int64_t kBlockCount = 2;
constexpr int64_t kBlockSize = 4;
constexpr int64_t kLayerCount = 1;

ModelArgs make_model_args() {
  ModelArgs model_args;
  model_args.model_type("test_model")
      .n_layers(kLayerCount)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(8);
  return model_args;
}

KVCacheShape make_cache_shape(const ModelArgs& model_args) {
  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount).block_size(kBlockSize);
  return KVCacheShape(capacity, model_args, /*world_size=*/1);
}

KVCacheCreateOptions make_create_options(const torch::Device& device) {
  KVCacheCreateOptions create_options;
  create_options.device(device)
      .dtype(torch::kFloat32)
      .num_layers(kLayerCount)
      .model_type("test_model");
  return create_options;
}

runtime::Options make_runtime_options(double host_blocks_factor) {
  runtime::Options options;
  options.model_id("test_model")
      .host_blocks_factor(host_blocks_factor)
      .world_size(1)
      .dp_size(1)
      .cp_size(1);
  return options;
}

class TestHierarchyWorker final : public LLMWorkerImpl {
 public:
  TestHierarchyWorker(const ParallelArgs& parallel_args,
                      const torch::Device& device,
                      const runtime::Options& options,
                      const ModelArgs& model_args)
      : LLMWorkerImpl(parallel_args, device, options) {
    dtype_ = torch::kFloat32;
    context_ =
        ModelContext(parallel_args,
                     model_args,
                     QuantArgs(),
                     torch::TensorOptions().dtype(dtype_).device(device));
  }

  void initialize_hierarchy_cache(const KVCacheShape& cache_shape,
                                  const KVCacheCreateOptions& create_options) {
    allocate_kv_caches(kv_caches_, cache_shape, create_options);
    init_hierarchy_kv_cache_transfer(cache_shape, create_options);
  }

  std::shared_ptr<HierarchyKVCacheTransfer> recreate_hierarchy_transfer() {
    std::shared_ptr<HierarchyKVCacheTransfer> transfer =
        create_hierarchy_kv_cache_transfer();
    register_hierarchy_kv_cache(*transfer,
                                HierarchyKVCacheTransfer::CacheRole::TARGET,
                                compute_stream_.get());
    EXPECT_TRUE(transfer->finalize_registration());
    return transfer;
  }

  bool owns_npu_parallel_input_prepare() const override { return false; }
};

class WorkerHierarchyKVCacheTransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP()
          << "An accelerator device is required for hierarchy KV transfer.";
    }

    device_ = std::make_unique<Device>(/*device_index=*/0);
    device_->set_device();
    device_->init_device_context();
  }

  std::unique_ptr<Device> device_;
};

TEST_F(WorkerHierarchyKVCacheTransferTest,
       DisabledHostCacheKeepsTransferUnbound) {
  const ModelArgs model_args = make_model_args();
  const KVCacheShape cache_shape = make_cache_shape(model_args);
  const KVCacheCreateOptions create_options =
      make_create_options(device_->unwrap());
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  TestHierarchyWorker worker(parallel_args,
                             device_->unwrap(),
                             make_runtime_options(/*host_blocks_factor=*/0.0),
                             model_args);
  worker.initialize_hierarchy_cache(cache_shape, create_options);

  EXPECT_EQ(worker.get_hierarchy_kv_cache_transfer(), nullptr);

  std::vector<BlockTransferInfo> transfer_info = {
      BlockTransferInfo(/*src_block_id=*/0, /*dst_block_id=*/1)};
  EXPECT_EQ(worker.transfer_kv_blocks(/*batch_id=*/1, transfer_info), 0U);
  Slice<BlockTransferInfo> transfer_slice(transfer_info);
  EXPECT_EQ(worker.prefetch_kv_blocks(transfer_slice, /*stats=*/nullptr),
            std::vector<uint8_t>({0}));
}

TEST_F(WorkerHierarchyKVCacheTransferTest,
       RequiredRestoreRejectsMissingManager) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const ModelArgs model_args = make_model_args();
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  TestHierarchyWorker worker(parallel_args,
                             device_->unwrap(),
                             make_runtime_options(/*host_blocks_factor=*/0.0),
                             model_args);
  ModelInputParams params;
  params.meta.batch_id = 91;
  worker.set_hierarchy_layer_synchronizer(params);
  EXPECT_EQ(params.parallel.layer_wise_load_synchronizer, nullptr);
  params.meta.requires_host_restore = true;
  EXPECT_DEATH(worker.set_hierarchy_layer_synchronizer(params),
               "Missing Host restore manager");
}

TEST_F(WorkerHierarchyKVCacheTransferTest,
       SharedTransferCanBeClearedAndRecreatedFromSavedContext) {
  const ModelArgs model_args = make_model_args();
  const KVCacheShape cache_shape = make_cache_shape(model_args);
  const KVCacheCreateOptions create_options =
      make_create_options(device_->unwrap());
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  TestHierarchyWorker worker(parallel_args,
                             device_->unwrap(),
                             make_runtime_options(/*host_blocks_factor=*/2.0),
                             model_args);
  worker.initialize_hierarchy_cache(cache_shape, create_options);

  std::shared_ptr<HierarchyKVCacheTransfer> transfer =
      worker.get_hierarchy_kv_cache_transfer();
  ASSERT_NE(transfer, nullptr);
  EXPECT_EQ(transfer.use_count(), 2);
  std::weak_ptr<HierarchyKVCacheTransfer> original_transfer = transfer;
  transfer.reset();
  EXPECT_EQ(original_transfer.use_count(), 1);

  worker.clear_hierarchy_kv_cache_transfer();
  EXPECT_TRUE(original_transfer.expired());
  worker.clear_hierarchy_kv_cache_transfer();

  std::shared_ptr<HierarchyKVCacheTransfer> recreated_transfer =
      worker.recreate_hierarchy_transfer();
  ASSERT_NE(recreated_transfer, nullptr);
  EXPECT_EQ(worker.get_hierarchy_kv_cache_transfer(), nullptr);
  HierarchyKVCacheTransfer* recreated_transfer_ptr = recreated_transfer.get();
  worker.set_hierarchy_kv_cache_transfer(recreated_transfer);
  EXPECT_EQ(worker.get_hierarchy_kv_cache_transfer().get(),
            recreated_transfer_ptr);
  recreated_transfer.reset();
  EXPECT_EQ(worker.get_hierarchy_kv_cache_transfer().use_count(), 2);
  worker.clear_hierarchy_kv_cache_transfer();
}

TEST_F(WorkerHierarchyKVCacheTransferTest,
       WorkerDestructionReleasesOwnedTransfer) {
  const ModelArgs model_args = make_model_args();
  const KVCacheShape cache_shape = make_cache_shape(model_args);
  const KVCacheCreateOptions create_options =
      make_create_options(device_->unwrap());
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  std::weak_ptr<HierarchyKVCacheTransfer> owned_transfer;

  {
    auto worker = std::make_unique<TestHierarchyWorker>(
        parallel_args,
        device_->unwrap(),
        make_runtime_options(/*host_blocks_factor=*/2.0),
        model_args);
    worker->initialize_hierarchy_cache(cache_shape, create_options);
    std::shared_ptr<HierarchyKVCacheTransfer> transfer =
        worker->get_hierarchy_kv_cache_transfer();
    ASSERT_NE(transfer, nullptr);
    owned_transfer = transfer;
    transfer.reset();
    EXPECT_EQ(owned_transfer.use_count(), 1);
  }

  EXPECT_TRUE(owned_transfer.expired());
}

TEST_F(WorkerHierarchyKVCacheTransferTest,
       EmptyHybridTargetInitializesAndResetsAcceptedTokens) {
  ModelArgs model_args = make_model_args();
  model_args.full_attention_interval(2);
  runtime::Options options = make_runtime_options(0.0);
  options.world_size(2)
      .dp_size(2)
      .num_decoding_tokens(4)
      .enable_speculative_decode(true)
      .enable_schedule_overlap(false);
  const ParallelArgs parallel_args(1, 2, 2, 1, nullptr, 1);
  TestHierarchyWorker worker(
      parallel_args, device_->unwrap(), options, model_args);

  for (bool has_previous_acceptance : {false, true}) {
    ForwardInput input;
    input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    input.input_params.parallel.dp_global_token_nums = {4, 0};
    input.input_params.parallel.raw_dp_global_token_nums = {4, 0};
    if (has_previous_acceptance) {
      input.input_params.num_accepted_tokens_host = {4, 2};
      input.input_params.num_accepted_tokens =
          torch::tensor({4, 2}, torch::kInt32);
    }
    ForwardInput prepared;
    worker.prepare_work_before_execute(input, prepared);
    auto stream = device_->current_stream();
    ASSERT_TRUE(stream->wait_event(prepared.metadata_ready_event));
    ASSERT_EQ(stream->synchronize(), 0);

    EXPECT_EQ(prepared.input_params.meta.num_sequences, 0);
    EXPECT_EQ(prepared.token_ids.numel(), 1);
    EXPECT_EQ(prepared.input_params.num_accepted_tokens_host,
              std::vector<int64_t>({1}));
    const torch::Tensor& accepted = prepared.input_params.num_accepted_tokens;
    ASSERT_TRUE(accepted.defined());
    EXPECT_EQ(accepted.scalar_type(), torch::kInt32);
    EXPECT_EQ(accepted.device(), device_->unwrap());
    EXPECT_TRUE(torch::equal(accepted.cpu(), torch::ones({1}, torch::kInt32)));
    EXPECT_TRUE(
        torch::equal(prepared.input_params.embedding.linear_state_indices.cpu(),
                     torch::zeros({1}, torch::kInt32)));
  }
}

TEST_F(WorkerHierarchyKVCacheTransferTest,
       EmptyNonSpeculativeOrDenseTargetLeavesAcceptanceAbsent) {
  for (int32_t decoding_tokens : {1, 4}) {
    ModelArgs model_args = make_model_args();
    model_args.full_attention_interval(decoding_tokens == 1 ? 2 : 1);
    runtime::Options options = make_runtime_options(0.0);
    options.world_size(2)
        .dp_size(2)
        .num_decoding_tokens(decoding_tokens)
        .enable_speculative_decode(decoding_tokens > 1)
        .enable_schedule_overlap(false);
    const ParallelArgs parallel_args(1, 2, 2, 1, nullptr, 1);
    TestHierarchyWorker worker(
        parallel_args, device_->unwrap(), options, model_args);
    ForwardInput input;
    input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    ForwardInput prepared;
    worker.prepare_work_before_execute(input, prepared);
    auto stream = device_->current_stream();
    ASSERT_TRUE(stream->wait_event(prepared.metadata_ready_event));
    ASSERT_EQ(stream->synchronize(), 0);

    EXPECT_EQ(prepared.token_ids.numel(), 1);
    EXPECT_TRUE(prepared.input_params.num_accepted_tokens_host.empty());
    EXPECT_FALSE(prepared.input_params.num_accepted_tokens.defined());
  }
}

TEST_F(WorkerHierarchyKVCacheTransferTest,
       RestoresParallelShardsAndDraftFromFreshHostCaches) {
  const char* master = std::getenv("XLLM_TEST_STORE_MASTER");
  if (master == nullptr) {
    GTEST_SKIP() << "Set XLLM_TEST_STORE_MASTER to a local TCP Mooncake Store.";
  }
  struct SplitCase {
    int32_t size;
    bool draft;
    std::array<int32_t, 8> values;
  };
  const std::vector<SplitCase> cases = {{0, true, {1, 1, 2, 2, 3, 3, 4, 4}},
                                        {1, false, {1, 1, 1, 1, 1, 1, 1, 1}},
                                        {2, false, {1, 1, 1, 1, 2, 2, 2, 2}},
                                        {4, true, {1, 1, 2, 2, 3, 3, 4, 4}},
                                        {8, true, {1, 2, 3, 4, 5, 6, 7, 8}}};
  const ModelArgs model_args = make_model_args();
  const KVCacheShape shape = make_cache_shape(model_args);
  const KVCacheCreateOptions create_options =
      make_create_options(device_->unwrap());
  std::unique_ptr<Stream> stream = device_->current_stream();
  for (const SplitCase& topology : cases) {
    runtime::Options options = make_runtime_options(/*host_blocks_factor=*/2.0);
    options.world_size(8)
        .cp_size(4)
        .enable_mla(true)
        .enable_kvcache_store(true)
        .store_protocol("tcp")
        .store_metadata_server("P2PHANDSHAKE")
        .store_master_server_address(master)
        .store_local_hostname("127.0.0.1:25100")
        .model_id("worker-split-test:" + std::to_string(getpid()) + ":" +
                  std::to_string(topology.size));
    const auto round_trip = [&](int32_t rank,
                                bool restore,
                                bool missing_draft = false) {
      SCOPED_TRACE(::testing::Message() << "size=" << topology.size << " rank="
                                        << rank << " restore=" << restore);
      ParallelArgs args(rank,
                        /*world_size=*/8,
                        /*dp_size=*/1,
                        /*cp_size=*/4,
                        /*process_group=*/nullptr,
                        /*ep_size=*/8);
      args.kv_split_size(topology.size);
      TestHierarchyWorker worker(args, device_->unwrap(), options, model_args);
      std::vector<KVCache> target;
      std::vector<KVCache> draft;
      allocate_kv_caches(target, shape, create_options);
      allocate_kv_caches(draft, shape, create_options);
      const int32_t value = topology.values[rank];
      target.front().get_k_cache().fill_(restore ? 0 : value);
      target.front().get_v_cache().fill_(restore ? 0 : value + 10);
      draft.front().get_k_cache().fill_(restore ? 0 : value + 20);
      draft.front().get_v_cache().fill_(restore ? 0 : value + 30);
      ASSERT_EQ(device_->synchronize_default_stream(), 0);
      auto transfer = worker.create_hierarchy_kv_cache_transfer();
      const auto register_cache = [&](std::vector<KVCache>& caches,
                                      HierarchyKVCacheTransfer::CacheRole role,
                                      const std::string& component) {
        HierarchyKVCacheTransfer::CacheRegistration registration;
        registration.role = role;
        registration.device_kv_caches = &caches;
        registration.kv_cache_shape = shape;
        registration.create_options = create_options;
        registration.producer_stream = stream.get();
        registration.store_key_component = component;
        transfer->register_cache(std::move(registration));
      };
      register_cache(
          target, HierarchyKVCacheTransfer::CacheRole::TARGET, "main");
      if (topology.draft) {
        register_cache(draft,
                       HierarchyKVCacheTransfer::CacheRole::DRAFT,
                       missing_draft ? "missing-draft" : "draft");
      }
      ASSERT_TRUE(transfer->finalize_registration());
      BlockTransferInfo info(/*src_block_id=*/0, /*dst_block_id=*/0);
      std::fill(std::begin(info.hash_key), std::end(info.hash_key), 93);
      info.transfer_type = TransferType::D2H2G;
      if (!restore) {
        ASSERT_EQ(transfer->transfer_kv_blocks(/*batch_id=*/1, {info}), 1U);
        return;
      }
      info.transfer_type = TransferType::G2H;
      std::vector<BlockTransferInfo> infos = {info};
      Slice<BlockTransferInfo> slice(infos);
      if (missing_draft) {
        EXPECT_EQ(transfer->prefetch_kv_blocks(slice, /*stats=*/nullptr),
                  std::vector<uint8_t>({0}));
        return;
      }
      ASSERT_EQ(transfer->prefetch_kv_blocks(slice, /*stats=*/nullptr),
                std::vector<uint8_t>({1}));
      info.transfer_type = TransferType::H2D;
      info.dst_block_id = 1;
      ASSERT_EQ(transfer->transfer_kv_blocks(/*batch_id=*/2, {info}), 1U);
      ModelInputParams params;
      params.meta.batch_id = 2;
      params.meta.requires_host_restore = true;
      transfer->set_layer_synchronizer(params);
      ASSERT_TRUE(params.synchronize_layer(/*layer_id=*/0));
      EXPECT_TRUE(torch::equal(
          target.front().get_k_cache()[1],
          torch::full_like(target.front().get_k_cache()[1], value)));
      EXPECT_TRUE(torch::equal(
          target.front().get_v_cache()[1],
          torch::full_like(target.front().get_v_cache()[1], value + 10)));
      if (topology.draft) {
        ASSERT_TRUE(params.synchronize_draft_layer());
        EXPECT_TRUE(torch::equal(
            draft.front().get_k_cache()[1],
            torch::full_like(draft.front().get_k_cache()[1], value + 20)));
        EXPECT_TRUE(torch::equal(
            draft.front().get_v_cache()[1],
            torch::full_like(draft.front().get_v_cache()[1], value + 30)));
      }
    };
    for (int32_t rank = 0; rank < 8; ++rank) {
      round_trip(rank, /*restore=*/false);
    }
    for (int32_t rank = 0; rank < 8; ++rank) {
      round_trip(rank, /*restore=*/true);
    }
    if (topology.draft) {
      round_trip(/*rank=*/7, /*restore=*/true, /*missing_draft=*/true);
    }
  }
}

}  // namespace
}  // namespace xllm
