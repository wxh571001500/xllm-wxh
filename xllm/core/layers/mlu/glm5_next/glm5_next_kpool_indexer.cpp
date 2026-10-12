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

#include "layers/mlu/glm5_next/glm5_next_kpool_indexer.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "common/constants.h"
#include "framework/config/kv_cache_config.h"
#include "framework/core/MLUStream.h"
#include "kernels/mlu/kpool.h"
#include "triton_jit/include/jit_kernel.h"
#include "util/linalg.h"

namespace xllm::layer {
namespace {

constexpr char kKPoolKernelPath[] =
    "xllm.core.kernels.mlu.triton_kernel.glm5_next_kpool";
constexpr char kKPoolExpandKernelPath[] =
    "xllm.core.kernels.mlu.triton_kernel.glm5_next_kpool_expand";
constexpr int64_t kWorkspaceBytes = 64 * 1024 * 1024;

int32_t max_sequence_length(const std::vector<int32_t>& lengths) {
  return lengths.empty() ? 0
                         : *std::max_element(lengths.begin(), lengths.end());
}

std::vector<int64_t> make_query_offsets(const std::vector<int32_t>& lengths) {
  std::vector<int64_t> offsets;
  offsets.reserve(lengths.size() + 1);
  offsets.emplace_back(0);
  for (const int32_t length : lengths) {
    offsets.emplace_back(offsets.back() + length);
  }
  return offsets;
}

std::vector<int32_t> resolve_seq_lens(const std::vector<int32_t>& host_lens,
                                      const torch::Tensor& device_lens,
                                      const char* name) {
  if (!host_lens.empty()) {
    if (device_lens.defined() && host_lens.front() == 0 &&
        host_lens.size() == static_cast<size_t>(device_lens.numel() + 1)) {
      std::vector<int32_t> per_sequence_lens;
      per_sequence_lens.reserve(host_lens.size() - 1);
      for (size_t i = 1; i < host_lens.size(); ++i) {
        per_sequence_lens.emplace_back(host_lens[i] - host_lens[i - 1]);
      }
      return per_sequence_lens;
    }
    return host_lens;
  }
  CHECK(device_lens.defined()) << name << " must be defined.";
  torch::Tensor cpu_lens =
      device_lens.to(torch::kCPU, torch::kInt32).contiguous();
  const int32_t* data = cpu_lens.data_ptr<int32_t>();
  return std::vector<int32_t>(data, data + cpu_lens.numel());
}

torch::Tensor hadamard_after_bf16_roundtrip(const torch::Tensor& input,
                                            const torch::Tensor& hadamard) {
  torch::Tensor rounded = input.to(torch::kBFloat16).to(torch::kFloat32);
  torch::Tensor matrix =
      hadamard.to(input.device(), torch::kFloat32, /*non_blocking=*/false);
  return util::hadamard_transform(rounded, matrix).to(torch::kBFloat16);
}

torch::Tensor make_row_batch(const std::vector<int32_t>& q_seq_lens,
                             const torch::Device& device) {
  std::vector<int64_t> rows;
  const int64_t row_count = std::accumulate(
      q_seq_lens.begin(), q_seq_lens.end(), static_cast<int64_t>(0));
  rows.reserve(static_cast<size_t>(row_count));
  for (size_t batch_id = 0; batch_id < q_seq_lens.size(); ++batch_id) {
    rows.insert(rows.end(),
                static_cast<size_t>(q_seq_lens[batch_id]),
                static_cast<int64_t>(batch_id));
  }
  return torch::tensor(rows, torch::TensorOptions().dtype(torch::kInt64))
      .to(device);
}

int64_t next_power_of_two(int64_t value) {
  int64_t result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

torch::Tensor make_kpool_rows(const torch::Tensor& starts, int64_t tokens) {
  torch::Tensor rows = torch::empty({tokens}, starts.options());
  if (tokens == 0) {
    return rows;
  }
  const int64_t requests = starts.numel() - 1;
  CHECK_GT(requests, 0);
  triton_jit::JITKernel::get(kKPoolKernelPath, "kpool_rows")
      .launch(static_cast<void*>(torch_mlu::getCurMLUStream()),
              {static_cast<uint32_t>((tokens + 63) / 64), 1, 1},
              {/*num_warps=*/1, /*num_stages=*/1},
              starts,
              rows,
              tokens,
              /*N=*/static_cast<int32_t>(requests),
              /*BN=*/static_cast<int32_t>(next_power_of_two(requests)),
              /*BT=*/64);
  return rows;
}

std::shared_ptr<KPoolBatchMetadata> make_kpool_batch_metadata(
    const AttentionMetadata& metadata,
    const torch::Device& device,
    const torch::Tensor& graph_rows) {
  auto batch = std::make_shared<KPoolBatchMetadata>();
  const auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
  const bool graph = metadata.enable_cuda_graph;
  batch->q_seq_lens = metadata.kpool_query_lens;
  if (batch->q_seq_lens.empty()) {
    if (graph && metadata.q_seq_lens_vec.empty()) {
      batch->q_seq_lens.assign(graph_rows.numel(), 1);
    } else {
      batch->q_seq_lens = resolve_seq_lens(
          metadata.q_seq_lens_vec, metadata.q_seq_lens, "KPool q lengths");
    }
  }
  const int64_t requests = static_cast<int64_t>(batch->q_seq_lens.size());
  const std::vector<int64_t> offsets = make_query_offsets(batch->q_seq_lens);
  const int64_t tokens = offsets.back();
  if (graph) {
    // The ordinary graph path has live cumulative query offsets. Explicit
    // speculative grouping is prepared before capture, or has uniform spans.
    if (metadata.kpool_query_lens.empty() && metadata.q_cu_seq_lens.defined() &&
        metadata.q_cu_seq_lens.numel() == requests + 1) {
      batch->query_starts = metadata.q_cu_seq_lens.to(torch::kInt64);
    } else {
      const int32_t width = batch->q_seq_lens.front();
      CHECK(std::all_of(batch->q_seq_lens.begin(),
                        batch->q_seq_lens.end(),
                        [width](int32_t len) { return len == width; }))
          << "Prepare ragged KPool metadata before graph capture.";
      batch->query_starts = torch::arange(requests + 1, options) * width;
    }
    batch->row_batch = make_kpool_rows(batch->query_starts, tokens);
  } else {
    batch->query_starts = torch::tensor(offsets, options);
    batch->row_batch = make_row_batch(batch->q_seq_lens, device);
  }
  batch->block_table = metadata.block_table;
  batch->tail_indices = metadata.linear_state_indices;
  const bool expanded = metadata.block_table.defined() &&
                        metadata.block_table.size(0) != requests;
  if (expanded) {
    CHECK_GE(metadata.block_table.size(0), tokens);
    batch->block_table = metadata.block_table.index_select(
        0, batch->query_starts.narrow(0, 0, requests));
  }
  if (batch->tail_indices.defined()) {
    CHECK_GE(batch->tail_indices.numel(), requests);
    batch->tail_indices = batch->tail_indices.narrow(0, 0, requests);
  }
  if (!graph) {
    batch->kv_seq_lens = resolve_seq_lens(
        metadata.kv_seq_lens_vec, metadata.kv_seq_lens, "KPool KV lengths");
    if (expanded) {
      std::vector<int32_t> logical_lens;
      logical_lens.reserve(requests);
      for (int64_t row = 0; row < requests; ++row) {
        logical_lens.emplace_back(batch->kv_seq_lens.at(offsets[row + 1] - 1));
      }
      batch->kv_seq_lens = std::move(logical_lens);
    }
    batch->max_kv_len = max_sequence_length(batch->kv_seq_lens);
  }
  return batch;
}

void update_prefill(const torch::Tensor& key,
                    const torch::Tensor& gate,
                    const torch::Tensor& ape,
                    const torch::Tensor& hadamard,
                    torch::Tensor& cache,
                    torch::Tensor& tail,
                    const torch::Tensor& tail_ids,
                    const torch::Tensor& block_table,
                    const torch::Tensor& positions,
                    const torch::Tensor& row_batch,
                    const torch::Tensor& starts,
                    int64_t block_size,
                    int64_t pool_size) {
  if (key.size(0) == 0) {
    return;
  }
  const torch::Tensor raw = key.contiguous();
  const torch::Tensor gates = gate.contiguous();
  const torch::Tensor a = ape.to(raw.device(), torch::kFloat32).contiguous();
  const torch::Tensor matrix = hadamard.contiguous();
  const torch::Tensor ids = tail_ids.contiguous();
  const torch::Tensor table = block_table.contiguous();
  const torch::Tensor pos = positions.contiguous();
  const torch::Tensor rows = row_batch.contiguous();
  const torch::Tensor offsets = starts.contiguous();
  CHECK_EQ(tail.dim(), 4);
  CHECK_EQ(tail.size(1), 2);
  CHECK_GE(tail.size(2), pool_size);
  CHECK_EQ(tail.size(3), key.size(1));
  CHECK_EQ(starts.numel(), tail_ids.numel() + 1);
  CHECK(tail.is_contiguous());
  const int64_t tail_length = tail.size(2);
  const int64_t dimension = raw.size(1);
  void* queue = static_cast<void*>(torch_mlu::getCurMLUStream());
  triton_jit::JITKernel::get(kKPoolKernelPath, "kpool_prefill_complete")
      .launch(queue,
              {static_cast<uint32_t>(raw.size(0)), 1, 1},
              {/*num_warps=*/1, /*num_stages=*/1},
              raw,
              gates,
              a,
              matrix,
              cache,
              tail,
              ids,
              table,
              pos,
              rows,
              offsets,
              raw.stride(0),
              gates.stride(0),
              table.stride(0),
              /*P=*/static_cast<int32_t>(pool_size),
              /*T=*/static_cast<int32_t>(tail_length),
              /*D=*/static_cast<int32_t>(dimension),
              /*BLOCK_P=*/static_cast<int32_t>(next_power_of_two(pool_size)),
              /*BLOCK_D=*/static_cast<int32_t>(next_power_of_two(dimension)),
              /*POOL_BLOCK=*/static_cast<int32_t>(block_size / pool_size));
  // Complete all old-tail reads before overwriting the request rings.
  triton_jit::JITKernel::get(kKPoolKernelPath, "kpool_prefill_stash")
      .launch(queue,
              {static_cast<uint32_t>(ids.numel()), 1, 1},
              {/*num_warps=*/1, /*num_stages=*/1},
              raw,
              gates,
              tail,
              ids,
              pos,
              offsets,
              raw.stride(0),
              gates.stride(0),
              /*T=*/static_cast<int32_t>(tail_length),
              /*D=*/static_cast<int32_t>(dimension),
              /*BLOCK_P=*/static_cast<int32_t>(next_power_of_two(tail_length)),
              /*BLOCK_D=*/static_cast<int32_t>(next_power_of_two(dimension)));
}

kernel::mlu::KPoolSelection expand_prefill(const torch::Tensor& pool_ids,
                                           const torch::Tensor& positions,
                                           const torch::Tensor& row_batch,
                                           const torch::Tensor& block_table,
                                           int64_t block_size,
                                           int64_t token_budget,
                                           int64_t pool_size,
                                           bool always_select_tail) {
  CHECK_LE(pool_ids.size(1), token_budget / pool_size);
  const int64_t queries = pool_ids.size(0);
  const int64_t selected_pools = pool_ids.size(1);
  const int64_t width = token_budget + pool_size - 1;
  const auto options = pool_ids.options().dtype(torch::kInt32);
  kernel::mlu::KPoolSelection result{torch::empty({queries, width}, options),
                                     torch::empty({queries}, options)};
  if (queries == 0) {
    return result;
  }
  int64_t pool_tile =
      pool_size > 16
          ? 16
          : std::min<int64_t>(
                selected_pools >= 1024 ? 1024 : 512,
                next_power_of_two(std::max<int64_t>(selected_pools, 1)));
  if (block_size % pool_size != 0) {
    pool_tile = std::min<int64_t>(pool_tile, 256);
  }
  const int64_t member_tile = std::min<int64_t>(pool_size > 16 ? 128 : 16,
                                                next_power_of_two(pool_size));
  const int64_t table_tile =
      block_table.size(1) <= 2048
          ? next_power_of_two(std::max<int64_t>(block_table.size(1), 1))
          : 0;
  const bool local_i32 =
      block_size % pool_size == 0 &&
      block_table.size(1) <= std::numeric_limits<int32_t>::max() / block_size;
  const int32_t tail_tile = local_i32 ? 1024 : 256;
  triton_jit::JITKernel::get(kKPoolExpandKernelPath, "kpool_prefill_expand")
      .launch(static_cast<void*>(torch_mlu::getCurMLUStream()),
              {static_cast<uint32_t>(std::min<int64_t>(queries, 65535)), 1, 1},
              {/*num_warps=*/1, /*num_stages=*/local_i32 ? 3 : 1},
              pool_ids,
              positions,
              row_batch,
              block_table,
              result.physical_slots,
              result.context_lens,
              pool_ids.stride(0),
              pool_ids.stride(1),
              positions.stride(0),
              row_batch.stride(0),
              block_table.stride(0),
              block_table.stride(1),
              block_table.size(1),
              queries,
              /*K=*/selected_pools,
              /*P=*/pool_size,
              /*B=*/block_size,
              /*W=*/width,
              /*BP=*/static_cast<int32_t>(pool_tile),
              /*BM=*/static_cast<int32_t>(member_tile),
              /*BT=*/static_cast<int32_t>(table_tile),
              /*TAIL=*/always_select_tail ? 1 : 0,
              /*BTAIL=*/tail_tile,
              /*LOCAL_I32=*/local_i32 ? 1 : 0);
  return result;
}

}  // namespace

void prepare_glm5_next_kpool_metadata(AttentionMetadata& metadata,
                                      const torch::Device& device) {
  metadata.kpool_batch_metadata =
      metadata.is_dummy
          ? nullptr
          : make_kpool_batch_metadata(metadata, device, metadata.kv_seq_lens);
}

torch::Tensor glm5_next_kpool_select(const torch::Tensor& query,
                                     const torch::Tensor& head_weights,
                                     const torch::Tensor& positions,
                                     const torch::Tensor& row_batch,
                                     const torch::Tensor& index_cache,
                                     const torch::Tensor& block_table,
                                     int64_t max_kv_seq_len,
                                     int64_t block_size,
                                     int64_t index_kpool,
                                     int64_t index_topk,
                                     double softmax_scale,
                                     int64_t workspace_bytes,
                                     const torch::Tensor& query_starts) {
  CHECK_GT(index_kpool, 0);
  CHECK_LE((max_kv_seq_len + block_size - 1) / block_size, block_table.size(1));
  const int64_t count = index_topk / index_kpool;
  const int64_t pools = (max_kv_seq_len + index_kpool - 1) / index_kpool;
  torch::Tensor result = torch::full(
      {query.size(0), count}, -1, query.options().dtype(torch::kInt64));
  if (pools == 0 || query.size(0) == 0 || count == 0) {
    return result;
  }
  CHECK_GE(workspace_bytes, pools * static_cast<int64_t>(sizeof(float)));
  const int64_t chunk = std::min<int64_t>(
      query.size(0), workspace_bytes / (pools * sizeof(float)));
  torch::Tensor workspace =
      torch::empty({chunk, pools}, query.options().dtype(torch::kFloat32));
  // Request offsets address the complete query tensor. Smaller workspaces use
  // the existing independent-query path instead of slicing request metadata.
  const torch::Tensor verify_starts =
      chunk == query.size(0) ? query_starts : torch::Tensor();
  for (int64_t start = 0; start < query.size(0); start += chunk) {
    const int64_t rows = std::min(chunk, query.size(0) - start);
    torch::Tensor scores = workspace.narrow(0, 0, rows);
    kernel::mlu::score_kpool(query.narrow(0, start, rows),
                             head_weights.narrow(0, start, rows),
                             index_cache,
                             block_table,
                             positions.narrow(0, start, rows),
                             row_batch.narrow(0, start, rows),
                             scores,
                             block_size,
                             index_kpool,
                             softmax_scale,
                             verify_starts);
    result.narrow(0, start, rows)
        .copy_(kernel::mlu::select_kpool(scores, count));
  }
  return result;
}

torch::Tensor glm5_next_kpool_normalize_key(torch::Tensor key,
                                            RMSNorm& key_norm) {
  const torch::ScalarType projection_dtype = key.scalar_type();
  key = key.to(torch::kFloat32);
  return std::get<0>(key_norm->forward(key)).to(projection_dtype);
}

Glm5NextKPoolIndexerImpl::Glm5NextKPoolIndexerImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const std::shared_ptr<RotaryEmbeddingBase>& rotary_emb,
    const torch::TensorOptions& options)
    : hidden_size_(args.hidden_size()),
      n_heads_(args.index_n_heads()),
      head_dim_(args.index_head_dim()),
      rope_head_dim_(args.qk_rope_head_dim()),
      index_topk_(args.index_topk()),
      index_kpool_(args.index_kpool()),
      block_size_(KVCacheConfig::get_instance().block_size()),
      softmax_scale_(std::pow(static_cast<double>(head_dim_), -0.5) *
                     std::pow(static_cast<double>(n_heads_), -0.5)),
      always_select_tail_(args.index_kpool_always_select_tail()),
      rotary_emb_(rotary_emb) {
  CHECK(options.device().is_privateuseone()) << "KPool requires MLU.";
  CHECK(options.dtype() == torch::kBFloat16) << "KPool requires BF16.";
  CHECK_GE(head_dim_, 16) << "Paged KPool scoring requires D >= 16.";
  CHECK_LE(head_dim_, 128);
  CHECK_EQ(head_dim_ & (head_dim_ - 1), 0);
  CHECK_GT(index_kpool_, 0);
  CHECK_GE(index_topk_, 0);
  CHECK_LE(index_topk_,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()) -
               index_kpool_ + 1)
      << "KPool context lengths must fit int32.";
  CHECK_EQ(index_topk_ % index_kpool_, 0);
  CHECK_GT(n_heads_, 0);
  CHECK_GT(block_size_, 0);
  CHECK_EQ(block_size_ % index_kpool_, 0);
  CHECK_EQ(parallel_args.kv_split_size_effective(), 1)
      << "KPool does not support DCP cache sharding.";

  CHECK(rope_head_dim_ == 0 || rotary_emb_ != nullptr)
      << "GLM5-Next KPool RoPE requires a rotary embedding.";

  wq_b_ = register_module("wq_b",
                          ReplicatedLinear(args.q_lora_rank(),
                                           n_heads_ * head_dim_,
                                           /*bias=*/false,
                                           quant_args,
                                           options));
  wk_ = register_module("wk",
                        ReplicatedLinear(hidden_size_,
                                         head_dim_,
                                         /*bias=*/false,
                                         QuantArgs(),
                                         options));
  weights_proj_ =
      register_module("weights_proj",
                      ReplicatedLinear(hidden_size_,
                                       n_heads_,
                                       /*bias=*/false,
                                       QuantArgs(),
                                       options.dtype(torch::kFloat32)));
  k_norm_ = register_module(
      "k_norm", RMSNorm(head_dim_, 1e-6, options.dtype(torch::kFloat32)));
  k_norm_->set_layernorm_mode();

  index_kpool_compress_gate_ = register_parameter(
      "index_kpool_compress_gate",
      torch::empty({head_dim_, hidden_size_}, options.dtype(torch::kBFloat16)),
      /*requires_grad=*/false);
  index_kpool_compress_ape_ = register_parameter(
      "index_kpool_compress_ape",
      torch::empty({index_kpool_, head_dim_}, options.dtype(torch::kFloat32)),
      /*requires_grad=*/false);
  hadamard_matrix_ = util::create_hadamard_matrix(
      head_dim_, torch::kFloat32, options.device(), /*normalize=*/true);
}

torch::Tensor Glm5NextKPoolIndexerImpl::project_query(
    const torch::Tensor& q_norm,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata) {
  torch::Tensor q =
      wq_b_->forward(q_norm).view({q_norm.size(0), n_heads_, head_dim_});
  if (rope_head_dim_ > 0) {
    torch::Tensor q_pe = q.slice(/*dim=*/-1, 0, rope_head_dim_);
    rotary_emb_->forward(q_pe,
                         positions,
                         attn_metadata.q_cu_seq_lens,
                         attn_metadata.max_query_len,
                         attn_metadata.is_prefill);
  }
  return hadamard_after_bf16_roundtrip(q, hadamard_matrix_);
}

torch::Tensor Glm5NextKPoolIndexerImpl::project_raw_k(
    const torch::Tensor& hidden_states,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata) {
  torch::Tensor k = wk_->forward(hidden_states);
  k = glm5_next_kpool_normalize_key(std::move(k), k_norm_);
  if (rope_head_dim_ > 0) {
    torch::Tensor k_pe =
        k.slice(/*dim=*/-1, 0, rope_head_dim_).unsqueeze(/*dim=*/1);
    rotary_emb_->forward(k_pe,
                         positions,
                         attn_metadata.q_cu_seq_lens,
                         attn_metadata.max_query_len,
                         attn_metadata.is_prefill);
  }
  return k.to(torch::kBFloat16);
}

Glm5NextKPoolIndexerImpl::Execution Glm5NextKPoolIndexerImpl::prepare_execution(
    const AttentionMetadata& metadata,
    const torch::Tensor& positions) const {
  Execution execution;
  execution.graph_decode = metadata.enable_cuda_graph;
  execution.batch = metadata.kpool_batch_metadata;
  if (!execution.batch) {
    execution.batch =
        make_kpool_batch_metadata(metadata, positions.device(), positions);
  }
  if (execution.graph_decode && metadata.kpool_batch_metadata) {
    auto batch = std::make_shared<KPoolBatchMetadata>(*execution.batch);
    const int64_t tokens = positions.numel();
    batch->row_batch = make_kpool_rows(batch->query_starts, tokens);
    execution.batch = std::move(batch);
  }
  // Draft extend carries prefill metadata; its active positions already mask
  // placeholders. Verify always uses the request-local fused ring update.
  execution.prefill = (metadata.is_prefill || metadata.is_chunked_prefill ||
                       max_sequence_length(execution.batch->q_seq_lens) > 1) &&
                      !metadata.is_spec_verify && !execution.graph_decode;
  // This is an update launch choice within the semantic stage. Tiny prefills
  // avoid the gather/softmax launch chain and share paged selection.
  execution.fused_update =
      !execution.prefill ||
      max_sequence_length(execution.batch->q_seq_lens) <= 32;
  execution.score_capacity =
      execution.graph_decode
          ? (metadata.max_seq_len > 0
                 ? metadata.max_seq_len
                 : execution.batch->block_table.size(1) * block_size_)
          : execution.batch->max_kv_len;
  return execution;
}

void Glm5NextKPoolIndexerImpl::update_cache(
    const torch::Tensor& raw_k,
    const torch::Tensor& gate_bf16,
    const torch::Tensor& positions,
    torch::Tensor& index_cache,
    torch::Tensor& tail_cache,
    const torch::Tensor& linear_state_indices,
    const torch::Tensor& block_table,
    const Execution& execution) {
  const auto& batch = *execution.batch;
  if (execution.prefill && !execution.fused_update) {
    update_prefill(raw_k,
                   gate_bf16,
                   index_kpool_compress_ape_,
                   hadamard_matrix_,
                   index_cache,
                   tail_cache,
                   linear_state_indices,
                   block_table,
                   positions,
                   batch.row_batch,
                   batch.query_starts,
                   block_size_,
                   index_kpool_);
    return;
  }
  const bool decode = execution.fused_update;
  kernel::mlu::update_kpool(raw_k,
                            gate_bf16,
                            index_kpool_compress_ape_,
                            hadamard_matrix_,
                            index_cache,
                            tail_cache,
                            linear_state_indices,
                            block_table,
                            positions,
                            batch.row_batch,
                            batch.query_starts,
                            block_size_,
                            index_kpool_,
                            decode);
}

torch::Tensor Glm5NextKPoolIndexerImpl::select_pools(
    const torch::Tensor& hidden_states,
    const torch::Tensor& q_norm,
    const torch::Tensor& positions,
    const torch::Tensor& index_cache,
    const AttentionMetadata& metadata,
    const Execution& execution) {
  const torch::Tensor query =
      project_query(q_norm, torch::clamp_min(positions, 0), metadata);
  const torch::Tensor weights =
      weights_proj_->forward(hidden_states.to(torch::kFloat32));
  return select_projected_pools(
      query, weights, positions, index_cache, metadata, execution);
}

torch::Tensor Glm5NextKPoolIndexerImpl::select_projected_pools(
    const torch::Tensor& query,
    const torch::Tensor& weights,
    const torch::Tensor& positions,
    const torch::Tensor& index_cache,
    const AttentionMetadata& metadata,
    const Execution& execution) {
  if (execution.prefill && !execution.fused_update) {
    // Score directly from the paged pool cache.  The previous prefill path
    // gathered every request's history into dense tensors and then reran an
    // FP32 matmul for each query tile.  The paged score kernel already applies
    // the request-local causal limit from positions and row_batch, so it is
    // equivalent while avoiding the dense gather and host-side grouping loops.
    return glm5_next_kpool_select(query,
                                  weights,
                                  positions,
                                  execution.batch->row_batch,
                                  index_cache,
                                  execution.batch->block_table,
                                  execution.batch->max_kv_len,
                                  block_size_,
                                  index_kpool_,
                                  index_topk_,
                                  softmax_scale_);
  }
  // Every verify query sees only pools complete at its own logical position.
  const torch::Tensor& score_positions = positions;
  const bool four_query_verify =
      metadata.is_spec_verify && metadata.max_query_len == 4 &&
      !execution.batch->q_seq_lens.empty() &&
      std::all_of(execution.batch->q_seq_lens.begin(),
                  execution.batch->q_seq_lens.end(),
                  [](int32_t length) { return length == 4; });
  return glm5_next_kpool_select(
      query,
      weights,
      score_positions,
      execution.batch->row_batch,
      index_cache,
      execution.batch->block_table,
      execution.score_capacity,
      block_size_,
      index_kpool_,
      index_topk_,
      softmax_scale_,
      /*workspace_bytes=*/kWorkspaceBytes,
      four_query_verify ? execution.batch->query_starts : torch::Tensor());
}

std::tuple<torch::Tensor, torch::Tensor> Glm5NextKPoolIndexerImpl::forward(
    const torch::Tensor& hidden_states,
    const torch::Tensor& q_norm,
    const torch::Tensor& positions,
    torch::Tensor& index_cache,
    torch::Tensor& tail_cache,
    const AttentionMetadata& attn_metadata) {
  CHECK(tail_cache.defined()) << "KPool requires framework-owned tail storage.";
  if (hidden_states.size(0) == 0) {
    const auto options = hidden_states.options().dtype(torch::kInt32);
    return {torch::empty({0, output_width()}, options),
            torch::empty({0}, options)};
  }
  const Execution execution = prepare_execution(attn_metadata, positions);
  const bool use_prefill_expand = execution.prefill && !execution.fused_update;
  torch::Tensor raw_k = project_raw_k(hidden_states, positions, attn_metadata);
  torch::Tensor gate_score =
      torch::nn::functional::linear(hidden_states, index_kpool_compress_gate_);
  const torch::Tensor gate_bf16 = gate_score.to(torch::kBFloat16);
  CHECK_EQ(execution.batch->row_batch.numel(), positions.numel());
  CHECK(execution.batch->tail_indices.defined());
  if (attn_metadata.is_spec_verify) {
    CHECK_LE(max_sequence_length(execution.batch->q_seq_lens),
             tail_cache.size(2) - index_kpool_)
        << "KPool tail capacity does not cover the speculative write window.";
  }
  torch::Tensor active =
      execution.batch->tail_indices.index_select(
          0, execution.batch->row_batch) > kPaddingLinearStateId;
  if (attn_metadata.slot_mapping.defined()) {
    active = active & (attn_metadata.slot_mapping.reshape({-1}) > 0);
  }
  const torch::Tensor cache_positions =
      torch::where(active, positions, torch::full_like(positions, -1));
  update_cache(raw_k,
               gate_bf16,
               cache_positions,
               index_cache,
               tail_cache,
               execution.batch->tail_indices,
               execution.batch->block_table,
               execution);
  const torch::Tensor pools = select_pools(hidden_states,
                                           q_norm,
                                           cache_positions,
                                           index_cache,
                                           attn_metadata,
                                           execution);
  kernel::mlu::KPoolSelection selection =
      use_prefill_expand
          ? expand_prefill(pools,
                           cache_positions,
                           execution.batch->row_batch,
                           execution.batch->block_table,
                           block_size_,
                           index_topk_,
                           index_kpool_,
                           always_select_tail_)
          : kernel::mlu::expand_kpool(pools,
                                      cache_positions,
                                      execution.batch->row_batch,
                                      execution.batch->block_table,
                                      block_size_,
                                      index_topk_,
                                      index_kpool_,
                                      always_select_tail_);
  return {std::move(selection.physical_slots),
          std::move(selection.context_lens)};
}

void Glm5NextKPoolIndexerImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  wq_b_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  wk_->load_state_dict(state_dict.get_dict_with_prefix("wk."));
  weights_proj_->load_state_dict(
      state_dict.get_dict_with_prefix("weights_proj."));
  k_norm_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));
  LOAD_WEIGHT(index_kpool_compress_gate);
  LOAD_WEIGHT(index_kpool_compress_ape);
}

}  // namespace xllm::layer
