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

#include "runtime/draft_model_spec_worker_impl.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <system_error>
#include <tuple>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/kv_cache/kv_cache_capacity.h"
#include "core/framework/kv_cache/kv_cache_estimation.h"
#include "core/framework/kv_cache/kv_cache_shape.h"
#include "core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"
#include "core/framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/adaptive_speculative_controller.h"
#include "core/framework/speculative/embedding_cache.h"
#include "runtime/llm_worker_impl.h"
#include "util/hash_util.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {

namespace {

int64_t get_dp_local_tp_size(const ParallelArgs& parallel_args) {
  const int64_t dp_size = std::max<int64_t>(parallel_args.dp_size(), 1);
  const int64_t cp_size = std::max<int64_t>(parallel_args.cp_size(), 1);
  return std::max<int64_t>(parallel_args.world_size() / dp_size / cp_size, 1);
}

std::string stable_path_digest(const std::string& path_string) {
  const std::filesystem::path path(path_string);
  std::error_code error;
  const std::filesystem::path canonical_path =
      std::filesystem::weakly_canonical(path, error);
  const std::string normalized_path =
      (error ? path.lexically_normal() : canonical_path).generic_string();
  const XXH3Key path_hash = hash_string(normalized_path);

  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string digest;
  digest.reserve(XXH3_128BITS_HASH_VALUE_LEN * 2);
  for (uint8_t byte : path_hash.data) {
    digest.push_back(kHexDigits[byte >> 4]);
    digest.push_back(kHexDigits[byte & 0x0f]);
  }
  return digest;
}

std::string draft_store_key_component(const runtime::Options& options) {
  std::string algorithm = options.speculative_algorithm();
  std::transform(
      algorithm.begin(),
      algorithm.end(),
      algorithm.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  const std::string draft_model_path = options.draft_model_path().value_or("");
  if (draft_model_path.empty()) {
    return "spec_draft::" + algorithm + "::embedded";
  }

  std::string draft_model_name = std::filesystem::path(draft_model_path)
                                     .lexically_normal()
                                     .filename()
                                     .generic_string();
  if (draft_model_name.empty()) {
    draft_model_name = "checkpoint";
  }
  return "spec_draft::" + algorithm + "::" + draft_model_name +
         "::" + stable_path_digest(draft_model_path);
}

KVCacheEstimateOptions make_kv_cache_estimate_options(
    const ModelArgs& model_args,
    const runtime::Options& options,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    int64_t cache_size_in_bytes) {
  const int64_t dp_local_tp_size = get_dp_local_tp_size(parallel_args);
  const int64_t n_heads = model_args.n_heads();
  const int64_t n_kv_heads = model_args.n_kv_heads().value_or(n_heads);

  KVCacheEstimateOptions estimate_options;
  estimate_options.dtype = dtype;
  estimate_options.kv_cache_dtype = options.kv_cache_dtype();
  estimate_options.indexer_cache_dtype =
      KVCacheConfig::get_instance().indexer_cache_dtype();
  estimate_options.cache_size_in_bytes = cache_size_in_bytes;
  estimate_options.block_size = options.block_size();
  estimate_options.world_size = dp_local_tp_size;
  estimate_options.n_local_kv_heads =
      std::max<int64_t>(n_kv_heads / dp_local_tp_size, 1);
  if (has_linear_attention_layers(model_args)) {
    estimate_options.n_local_linear_k_heads = std::max<int64_t>(
        model_args.linear_num_key_heads() / dp_local_tp_size, 1);
    estimate_options.n_local_linear_v_heads = std::max<int64_t>(
        model_args.linear_num_value_heads() / dp_local_tp_size, 1);
  }
  estimate_options.max_seqs_per_batch =
      static_cast<int64_t>(options.max_seqs_per_batch());
  estimate_options.num_speculative_tokens =
      static_cast<int64_t>(options.num_speculative_tokens());
  estimate_options.max_tokens_per_batch =
      static_cast<int64_t>(options.max_tokens_per_batch());
  estimate_options.max_tokens_per_chunk_for_prefill =
      static_cast<int64_t>(options.max_tokens_per_chunk_for_prefill());
  estimate_options.max_linear_state_cache_slots =
      options.max_linear_state_cache_slots();
  estimate_options.is_draft_engine = options.is_draft_engine();
  estimate_options.enable_chunked_prefill = options.enable_chunked_prefill();
  estimate_options.enable_schedule_overlap = options.enable_schedule_overlap();
  const KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  estimate_options.enable_prefix_cache =
      kv_cache_config.enable_prefix_cache() &&
      !kv_cache_config.enable_xtensor();
  estimate_options.enable_disagg_pd = options.enable_disagg_pd();
  estimate_options.instance_role = options.instance_role();
  return estimate_options;
}

}  // namespace

KVCacheShape build_speculative_draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape,
    const ModelArgs& draft_model_args,
    int64_t block_size,
    int64_t draft_world_size,
    const std::string& kv_cache_dtype,
    int64_t replicated_block_pages) {
  CHECK(!target_kv_cache_shape.key_cache_shape().empty())
      << "target KV cache shape must contain key cache shape";
  if (target_kv_cache_shape.has_grouped_cache_layout()) {
    return target_kv_cache_shape;
  }
  CHECK_GE(replicated_block_pages, 1)
      << "replicated_block_pages must be at least one";

  // Propagate the target-side packed-C8 layout to the draft pool so the two
  // allocators agree on 656B/token rows (glm_moe_dsa_mtp + int8). Every other
  // combination keeps the legacy BF16 layout: util::enable_mla_packed_c8()
  // returns false for unsupported model_types AND on non-NPU builds, so the
  // default-false capacity bit is left untouched.
  const bool draft_mla_packed_c8 =
      draft_model_args.enable_mla() &&
      util::enable_mla_packed_c8(kv_cache_dtype == "int8",
                                 draft_model_args.model_type());
  KVCacheCapacity draft_capacity;
  // A replicated draft pool owns `replicated_block_pages` physical rows per
  // logical block on every rank, so the block manager's logical block ids still
  // index it directly and the draft's slots stay logical token slots.
  draft_capacity.n_blocks(target_kv_cache_shape.key_cache_shape()[0] *
                          replicated_block_pages)
      .block_size(block_size)
      .replicated_block_pages(replicated_block_pages)
      .enable_mla_kv_cache_quant(draft_mla_packed_c8);
  draft_capacity.kpool_layout(target_kv_cache_shape.kpool_layout());
  if (target_kv_cache_shape.has_kpool_tail_shape()) {
    draft_capacity
        .num_linear_state_blocks(target_kv_cache_shape.kpool_tail_shape()[0])
        .kpool_tail_len(target_kv_cache_shape.kpool_tail_shape()[2]);
  }
  return KVCacheShape(draft_capacity, draft_model_args, draft_world_size);
}

DraftModelSpecWorkerImpl::DraftModelSpecWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    const runtime::Options& target_options,
    WorkerType worker_type,
    const std::function<std::unique_ptr<LLMWorkerImpl>()>& draft_factory)
    : SpeculativeWorkerImpl(parallel_args,
                            device,
                            options,
                            target_options,
                            worker_type),
      draft_sampling_mode_(
          parse_draft_sampling_mode(options.draft_sampling_mode())) {
  CHECK(draft_factory) << "Draft worker factory must not be empty.";
  draft_impl_ = draft_factory();
  CHECK(draft_impl_ != nullptr) << "Draft worker factory must return a worker.";
}

DraftModelSpecWorkerImpl::~DraftModelSpecWorkerImpl() {
  // Stop the async load threadpool before draft_impl_'s KV cache is freed;
  // shutdown() is idempotent, so later ref-zero owners re-run it as a no-op.
  if (hierarchy_kv_cache_transfer_ != nullptr) {
    hierarchy_kv_cache_transfer_->shutdown();
  }
}

KVCacheShape DraftModelSpecWorkerImpl::draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape) const {
  return build_draft_kv_cache_shape(target_kv_cache_shape);
}

KVCacheShape DraftModelSpecWorkerImpl::build_draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape,
    int64_t draft_world_size) const {
  if (draft_world_size <= 0 &&
      !target_kv_cache_shape.has_grouped_cache_layout()) {
    draft_world_size =
        get_dp_local_tp_size(draft_impl_->context_.get_parallel_args());
  }
  return build_speculative_draft_kv_cache_shape(
      target_kv_cache_shape,
      draft_impl_->context_.get_model_args(),
      options_.block_size(),
      draft_world_size,
      options_.kv_cache_dtype(),
      draft_replicated_block_pages());
}

std::tuple<int64_t, int64_t>
DraftModelSpecWorkerImpl::estimate_kv_cache_capacity_with_draft(
    const runtime::Options& target_options,
    const runtime::Options& draft_options) {
  const std::tuple<int64_t, int64_t> target_memory =
      impl_->estimate_kv_cache_capacity();
  const std::tuple<int64_t, int64_t> draft_memory =
      draft_impl_->estimate_kv_cache_capacity();
  const int64_t cache_size_in_bytes =
      std::min(std::get<0>(target_memory), std::get<0>(draft_memory));
  const int64_t total_memory =
      std::min(std::get<1>(target_memory), std::get<1>(draft_memory));

  const ModelArgs& target_model_args = impl_->context_.get_model_args();
  if (!util::is_deepseek_v4_model_type(target_model_args.model_type())) {
    return {cache_size_in_bytes, total_memory};
  }

  const ModelArgs& draft_model_args = draft_impl_->context_.get_model_args();
  KVCacheEstimateOptions target_estimate_options =
      make_kv_cache_estimate_options(target_model_args,
                                     target_options,
                                     parallel_args_,
                                     dtype_,
                                     cache_size_in_bytes);
  const KVCacheEstimateOptions draft_estimate_options =
      make_kv_cache_estimate_options(draft_model_args,
                                     draft_options,
                                     parallel_args_,
                                     dtype_,
                                     cache_size_in_bytes);
  target_estimate_options.draft_model_args = &draft_model_args;
  target_estimate_options.draft_options = &draft_estimate_options;

  const KVCacheCapacity capacity = ::xllm::estimate_kv_cache_capacity(
      target_model_args, target_estimate_options);
  return {capacity.cache_size_in_bytes(), total_memory};
}

void DraftModelSpecWorkerImpl::prepare_hierarchy_kv_cache_transfers() {
  if (options_.host_blocks_factor() <= 1.0) {
    return;
  }

  CHECK(impl_ != nullptr);
  std::shared_ptr<HierarchyKVCacheTransfer> unified_transfer =
      hierarchy_kv_cache_transfer_;
  if (unified_transfer == nullptr) {
    unified_transfer = impl_->get_hierarchy_kv_cache_transfer();
  }
  if (unified_transfer == nullptr) {
    unified_transfer = draft_impl_->get_hierarchy_kv_cache_transfer();
  }
  if (unified_transfer == nullptr) {
    unified_transfer = impl_->create_hierarchy_kv_cache_transfer();
  }

  if (impl_->get_hierarchy_kv_cache_transfer() == nullptr) {
    impl_->bind_hierarchy_kv_cache_transfer(
        unified_transfer,
        HierarchyKVCacheTransfer::CacheRole::TARGET,
        compute_stream_.get(),
        /*store_key_component=*/"main");
  } else {
    CHECK_EQ(impl_->get_hierarchy_kv_cache_transfer().get(),
             unified_transfer.get())
        << "Speculative target worker hierarchy KV cache transfer changed "
           "unexpectedly.";
  }

  if (draft_impl_->get_hierarchy_kv_cache_transfer() == nullptr) {
    draft_impl_->bind_hierarchy_kv_cache_transfer(
        unified_transfer,
        HierarchyKVCacheTransfer::CacheRole::DRAFT,
        compute_stream_.get(),
        draft_store_key_component(options_));
  } else {
    CHECK_EQ(draft_impl_->get_hierarchy_kv_cache_transfer().get(),
             unified_transfer.get())
        << "Speculative draft worker hierarchy KV cache transfer changed "
           "unexpectedly.";
  }

  if (hierarchy_kv_cache_transfer_ == nullptr) {
    set_hierarchy_kv_cache_transfer(std::move(unified_transfer));
  }
}

void DraftModelSpecWorkerImpl::finalize_hierarchy_kv_cache_transfers() {
  if (options_.host_blocks_factor() <= 1.0) {
    return;
  }

  CHECK(hierarchy_kv_cache_transfer_ != nullptr)
      << "Speculative hierarchy KV cache transfer is not prepared.";
  if (!hierarchy_kv_cache_transfer_->registration_finalized()) {
    CHECK(hierarchy_kv_cache_transfer_->finalize_registration());
  }
}

void DraftModelSpecWorkerImpl::init_embedding_cache(int64_t num_blocks) {
  embedding_cache_ = std::make_shared<EmbeddingCache>(
      num_blocks, requires_full_target_replay());
  const int64_t placeholder_size = get_embedding_placeholder_size();
  if (placeholder_size > 0) {
    embedding_cache_->set_placeholder(
        torch::zeros({placeholder_size}, torch::dtype(dtype_).device(device_)));
  }
}

bool DraftModelSpecWorkerImpl::allocate_pools(
    const KVCacheShape& kv_cache_shape,
    const AllocateFn& allocate) {
  const bool target_allocated = allocate_pool_if_loaded(
      *impl_,
      [&]() -> const KVCacheShape& { return kv_cache_shape; },
      allocate);
  const bool draft_allocated = allocate_pool_if_loaded(
      *draft_impl_,
      [&] { return draft_kv_cache_shape(kv_cache_shape); },
      allocate);

  init_embedding_cache(kv_cache_shape.key_cache_shape()[0]);
  const bool allocated = target_allocated && draft_allocated;
  if (allocated) {
    finalize_hierarchy_kv_cache_transfers();
  }
  return allocated;
}

bool DraftModelSpecWorkerImpl::allocate_kv_cache(
    const KVCacheShape& kv_cache_shape) {
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);
  // MTP must fix its context policy before either pool or transfer is created.
  static_cast<void>(requires_full_target_replay());
  prepare_hierarchy_kv_cache_transfers();

  return allocate_pools(kv_cache_shape,
                        [](WorkerImpl& worker, const KVCacheShape& shape) {
                          return worker.allocate_kv_cache(shape);
                        });
}

#if defined(USE_NPU) || defined(USE_MLU)
bool DraftModelSpecWorkerImpl::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);
  static_cast<void>(requires_full_target_replay());
  prepare_hierarchy_kv_cache_transfers();

  if (kv_cache_transfer_ == nullptr) {
    kv_cache_transfer_ = std::make_shared<MooncakeKVCacheTransferDefault>(
        device_.index(),
        options_.transfer_listen_port(),
        device_,
        context_.get_model_args().model_type());

    const int32_t device_id = device_.index();
    kv_cache_transfer_->initialize(device_id);
  }

  return allocate_pools(kv_cache_shape,
                        [this](WorkerImpl& worker, const KVCacheShape& shape) {
                          return worker.allocate_kv_cache_with_transfer(
                              kv_cache_transfer_, shape);
                        });
}
#endif

void DraftModelSpecWorkerImpl::force_greedy_draft_sampling(
    SamplingParameters& sampling_params) {
  if (sampling_params.do_sample.defined()) {
    sampling_params.do_sample = torch::zeros_like(sampling_params.do_sample);
  }
  sampling_params.all_random_sample = false;
  sampling_params.all_greedy_sample = true;
  sampling_params.logprobs = false;
  sampling_params.max_top_logprobs = 0;
  sampling_params.return_probs = false;
}

ForwardInput DraftModelSpecWorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
  return inputs;
}

void DraftModelSpecWorkerImpl::sync_dp_global_token_nums_after_prune(
    ModelInputParams& input_params,
    int32_t local_total_val_tokens) {
  // Only the adaptive controller makes the per-rank validate token count
  // data-dependent. When it is inactive the dense path already keeps
  // dp_global_token_nums identical across ranks (constant width multiplier), so
  // skip the collective entirely and leave static behavior byte-unchanged.
  if (adaptive_spec_controller_ == nullptr ||
      !adaptive_spec_controller_->enabled()) {
    return;
  }
  ProcessGroup* dp_group = parallel_args_.dp_local_process_group_;
  if (dp_group == nullptr || dp_group->world_size() <= 1) {
    return;
  }
  const int32_t dp_size = static_cast<int32_t>(dp_group->world_size());
  // Gather each DP peer's true post-pruning validate token count. The engine
  // pre-populates dp_global_token_nums assuming a uniform per-seq width; after
  // per-seq pruning that assumption is stale, so the padded and raw vectors are
  // both rewritten with the gathered per-rank counts. Every DP rank runs this,
  // including ranks that did not prune this step, so the collective matches.
  torch::Tensor local = torch::tensor(
      {local_total_val_tokens},
      torch::TensorOptions().dtype(torch::kInt32).device(device_.unwrap()));
  torch::Tensor gathered = dp_group->allgather_base_sync(local);
  torch::Tensor gathered_cpu =
      safe_to(gathered.view({dp_size}), torch::kCPU).contiguous();
  const int32_t* gathered_data = gathered_cpu.data_ptr<int32_t>();

  std::vector<int32_t>& token_nums = input_params.parallel.dp_global_token_nums;
  std::vector<int32_t>& raw_token_nums =
      input_params.parallel.raw_dp_global_token_nums;
  CHECK_EQ(static_cast<int32_t>(token_nums.size()), dp_size)
      << "dp_global_token_nums size must match DP group world size";
  for (int32_t dp_rank = 0; dp_rank < dp_size; ++dp_rank) {
    token_nums[static_cast<size_t>(dp_rank)] = gathered_data[dp_rank];
  }
  if (!raw_token_nums.empty()) {
    CHECK_EQ(static_cast<int32_t>(raw_token_nums.size()), dp_size)
        << "raw_dp_global_token_nums size must match DP group world size";
    for (int32_t dp_rank = 0; dp_rank < dp_size; ++dp_rank) {
      raw_token_nums[static_cast<size_t>(dp_rank)] = gathered_data[dp_rank];
    }
  }
}

void DraftModelSpecWorkerImpl::sync_dp_global_token_nums_for_idle_rank(
    ModelInputParams& input_params) {
  if (adaptive_spec_controller_ == nullptr ||
      !adaptive_spec_controller_->enabled()) {
    return;
  }
  ProcessGroup* dp_group = parallel_args_.dp_local_process_group_;
  if (dp_group == nullptr || dp_group->world_size() <= 1) {
    return;
  }
  // The idle rank's own validate width is already materialized in its
  // dp_global_token_nums entry (scaled to the uniform N+1 width by the caller).
  // Contribute exactly that so the gathered vector stays consistent with the
  // busy peers, which pass their pruned Σ per_seq_val_tokens.
  const int32_t dp_rank = static_cast<int32_t>(dp_group->rank());
  const std::vector<int32_t>& token_nums =
      input_params.parallel.dp_global_token_nums;
  CHECK_LT(dp_rank, static_cast<int32_t>(token_nums.size()))
      << "DP rank out of range for dp_global_token_nums";
  sync_dp_global_token_nums_after_prune(
      input_params, token_nums[static_cast<size_t>(dp_rank)]);
}

}  // namespace xllm
