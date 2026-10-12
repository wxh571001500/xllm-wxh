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

#include "framework/kv_cache/cache_layout_builder.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

#include "util/utils.h"

namespace xllm {

namespace {

bool fail(const std::string& message, std::string* error) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

bool checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t* result) {
  if (result == nullptr ||
      (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool get_head_placement(int64_t global_head_count,
                        int64_t local_head_count,
                        int32_t tp_rank,
                        int32_t tp_size,
                        int64_t* first_global_head,
                        int32_t* replica_count,
                        std::string* error) {
  if (global_head_count <= 0 || local_head_count <= 0 || tp_size <= 0 ||
      tp_rank < 0 || tp_rank >= tp_size || first_global_head == nullptr ||
      replica_count == nullptr) {
    return fail("invalid TP head placement", error);
  }

  if (global_head_count >= tp_size) {
    if (global_head_count % tp_size != 0 ||
        local_head_count != global_head_count / tp_size) {
      return fail("global head count is not evenly sharded by TP", error);
    }
    *first_global_head = static_cast<int64_t>(tp_rank) * local_head_count;
    *replica_count = 1;
    return true;
  }

  if (tp_size % global_head_count != 0 || local_head_count != 1) {
    return fail("KV-head replication requires TP to be a multiple of heads",
                error);
  }
  *replica_count = tp_size / static_cast<int32_t>(global_head_count);
  *first_global_head = tp_rank / *replica_count;
  return true;
}

uint64_t tensor_stride_bytes(const torch::Tensor& tensor, int64_t dim) {
  return static_cast<uint64_t>(tensor.stride(dim)) * tensor.element_size();
}

// Writes the descriptor every replicated entry point below ends with: the kind
// and the resource scope are shared, only the span geometry differs.
void store_replicated_descriptor(KVCacheTensor* cache_tensor,
                                 LogicalSpan span) {
  LogicalShardDescriptor descriptor;
  descriptor.kind = LogicalShardKind::REPLICATED;
  descriptor.resource_scope = cache_tensor->sequence_scoped
                                  ? CacheResourceScope::SEQUENCE
                                  : CacheResourceScope::BLOCK;
  descriptor.spans.emplace_back(std::move(span));
  cache_tensor->shard_descriptor = std::move(descriptor);
}

bool describe_replicated_tensor(KVCacheTensor* cache_tensor,
                                std::string* error) {
  const torch::Tensor& tensor = cache_tensor->tensor;
  if (!tensor.is_contiguous() || tensor.dim() == 0 || tensor.size(0) <= 0) {
    return fail("replicated cache tensor must be contiguous and non-empty",
                error);
  }

  const uint64_t resource_count = static_cast<uint64_t>(tensor.size(0));
  const uint64_t tensor_bytes = static_cast<uint64_t>(tensor.nbytes());
  if (tensor_bytes % resource_count != 0) {
    return fail("cache tensor bytes are not divisible by resource count",
                error);
  }

  LogicalSpan span;
  span.logical_tensor = cache_tensor->role.to_string();
  span.bytes_per_region = tensor_bytes / resource_count;
  span.owner_tp_rank = 0;

  store_replicated_descriptor(cache_tensor, std::move(span));
  return true;
}

// The NPU replicates the DSA indexer cache on every rank and packs
// `indexer_pages_per_block()` physical index pages into one logical cache
// block (platform.h). Grouping those pages into a single logical resource keeps
// the block-id mapping every transfer binder uses pointing at exactly the index
// pages the Python read/write path addresses, for every kv_split size.
bool describe_replicated_index_pages(int64_t indexer_pages,
                                     KVCacheTensor* cache_tensor,
                                     std::string* error) {
  const torch::Tensor& tensor = cache_tensor->tensor;
  if (!tensor.is_contiguous() || tensor.dim() == 0 || tensor.size(0) <= 0) {
    return fail(
        "replicated index cache tensor must be contiguous and non-empty",
        error);
  }
  if (tensor.size(0) % indexer_pages != 0) {
    return fail("index cache rows are not divisible by its pages per block",
                error);
  }
  const uint64_t page_bytes = tensor_stride_bytes(tensor, 0);
  if (page_bytes == 0) {
    return fail("index cache page must contain at least one element", error);
  }

  LogicalSpan span;
  span.logical_tensor = cache_tensor->role.to_string();
  span.bytes_per_region = page_bytes;
  span.repeat_count = static_cast<uint64_t>(indexer_pages);
  span.logical_stride_bytes = page_bytes;
  span.physical_stride_bytes = page_bytes;
  span.owner_tp_rank = 0;

  store_replicated_descriptor(cache_tensor, std::move(span));
  return true;
}

bool describe_attention_heads(const CacheTensorLayoutContext& context,
                              int64_t global_head_count,
                              int64_t replicated_block_pages,
                              KVCacheTensor* cache_tensor,
                              std::string* error) {
  const torch::Tensor& tensor = cache_tensor->tensor;
  if (!tensor.is_contiguous() || tensor.dim() < 3 || tensor.dim() > 4) {
    return fail("head-sharded cache tensor must be contiguous and 3D/4D",
                error);
  }

  const int64_t head_axis = context.head_major_layout ? 1 : 2;
  const int64_t token_axis = context.head_major_layout ? 2 : 1;
  if (head_axis >= tensor.dim() || token_axis >= tensor.dim()) {
    return fail("cache tensor does not contain head and token axes", error);
  }
  if (context.block_token_capacity > 0 &&
      tensor.size(token_axis) != context.block_token_capacity) {
    return fail("cache tensor token capacity differs from producer metadata",
                error);
  }

  const int64_t local_head_count = tensor.size(head_axis);
  int64_t first_global_head = 0;
  int32_t replica_count = 1;
  if (!get_head_placement(global_head_count,
                          local_head_count,
                          context.tp_rank,
                          context.tp_size,
                          &first_global_head,
                          &replica_count,
                          error)) {
    return false;
  }
  const bool replicated = replica_count > 1;

  uint64_t bytes_per_head = tensor.element_size();
  const int64_t inner_begin = std::max(head_axis, token_axis) + 1;
  for (int64_t dim = inner_begin; dim < tensor.dim(); ++dim) {
    uint64_t next_bytes = 0;
    if (!checked_multiply(bytes_per_head,
                          static_cast<uint64_t>(tensor.size(dim)),
                          &next_bytes)) {
      return fail("cache tensor head byte size overflow", error);
    }
    bytes_per_head = next_bytes;
  }

  LogicalShardDescriptor descriptor;
  descriptor.kind =
      replicated ? LogicalShardKind::REPLICATED : LogicalShardKind::SHARDED;
  descriptor.resource_scope = CacheResourceScope::BLOCK;
  descriptor.spans.reserve(static_cast<size_t>(local_head_count));
  const uint64_t logical_token_stride =
      static_cast<uint64_t>(global_head_count) * bytes_per_head;
  const uint64_t physical_token_stride =
      tensor_stride_bytes(tensor, token_axis);
  const uint64_t physical_head_stride = tensor_stride_bytes(tensor, head_axis);
  // A replicated pool stores `replicated_block_pages` physical pages per
  // logical block. Those pages are consecutive rows, so one span per head with
  // a token-major repeat covers them byte-exactly: row `page * token_count +
  // token` sits at exactly `(page * token_count + token) *
  // physical_token_stride` within the resource.
  CHECK_GE(replicated_block_pages, 1)
      << "cache replicated pages per block must be positive";
  const uint64_t token_count = static_cast<uint64_t>(tensor.size(token_axis)) *
                               static_cast<uint64_t>(replicated_block_pages);

  for (int64_t local_head = 0; local_head < local_head_count; ++local_head) {
    const int64_t global_head = first_global_head + local_head;
    LogicalSpan span;
    span.logical_tensor = cache_tensor->role.to_string();
    span.logical_offset_bytes =
        static_cast<uint64_t>(global_head) * bytes_per_head;
    span.physical_offset_bytes =
        static_cast<uint64_t>(local_head) * physical_head_stride;
    span.bytes_per_region = bytes_per_head;
    span.repeat_count = token_count;
    span.logical_stride_bytes = logical_token_stride;
    span.physical_stride_bytes = physical_token_stride;
    span.owner_tp_rank = replicated
                             ? static_cast<int32_t>(global_head * replica_count)
                             : context.tp_rank;
    descriptor.spans.emplace_back(std::move(span));
  }
  cache_tensor->shard_descriptor = std::move(descriptor);
  return true;
}

bool describe_ssm(const CacheTensorLayoutContext& context,
                  KVCacheTensor* cache_tensor,
                  std::string* error) {
  const torch::Tensor& tensor = cache_tensor->tensor;
  if (!tensor.is_contiguous() || tensor.dim() != 4 ||
      context.linear_ssm_checkpoint_stride <= 0 ||
      tensor.size(0) % context.linear_ssm_checkpoint_stride != 0) {
    return fail("SSM cache tensor must use [slot, head, key, value]", error);
  }

  const int64_t local_head_count = tensor.size(1);
  int64_t first_global_head = 0;
  int32_t replica_count = 1;
  if (!get_head_placement(context.linear_value_head_count,
                          local_head_count,
                          context.tp_rank,
                          context.tp_size,
                          &first_global_head,
                          &replica_count,
                          error)) {
    return false;
  }
  const bool replicated = replica_count > 1;

  uint64_t bytes_per_head = tensor.element_size();
  for (int64_t dim = 2; dim < tensor.dim(); ++dim) {
    uint64_t next_bytes = 0;
    if (!checked_multiply(bytes_per_head,
                          static_cast<uint64_t>(tensor.size(dim)),
                          &next_bytes)) {
      return fail("SSM cache head byte size overflow", error);
    }
    bytes_per_head = next_bytes;
  }

  LogicalShardDescriptor descriptor;
  descriptor.kind =
      replicated ? LogicalShardKind::REPLICATED : LogicalShardKind::SHARDED;
  descriptor.resource_scope = CacheResourceScope::SEQUENCE;
  descriptor.spans.reserve(static_cast<size_t>(local_head_count));
  const uint64_t checkpoint_stride =
      static_cast<uint64_t>(context.linear_ssm_checkpoint_stride);
  const uint64_t logical_checkpoint_stride =
      static_cast<uint64_t>(context.linear_value_head_count) * bytes_per_head;
  const uint64_t physical_checkpoint_stride = tensor_stride_bytes(tensor, 0);
  const uint64_t physical_head_stride = tensor_stride_bytes(tensor, 1);
  for (int64_t local_head = 0; local_head < local_head_count; ++local_head) {
    const int64_t global_head = first_global_head + local_head;
    LogicalSpan span;
    span.logical_tensor = cache_tensor->role.to_string();
    span.logical_offset_bytes =
        static_cast<uint64_t>(global_head) * bytes_per_head;
    span.physical_offset_bytes =
        static_cast<uint64_t>(local_head) * physical_head_stride;
    span.bytes_per_region = bytes_per_head;
    span.repeat_count = checkpoint_stride;
    span.logical_stride_bytes = logical_checkpoint_stride;
    span.physical_stride_bytes = physical_checkpoint_stride;
    span.owner_tp_rank = replicated
                             ? static_cast<int32_t>(global_head * replica_count)
                             : context.tp_rank;
    descriptor.spans.emplace_back(std::move(span));
  }
  cache_tensor->shard_descriptor = std::move(descriptor);
  return true;
}

bool append_conv_component(const CacheTensorLayoutContext& context,
                           const torch::Tensor& tensor,
                           const std::string& logical_tensor,
                           int64_t global_head_count,
                           int64_t local_head_count,
                           uint64_t component_offset_bytes,
                           LogicalShardDescriptor* descriptor,
                           std::string* error) {
  int64_t first_global_head = 0;
  int32_t replica_count = 1;
  if (!get_head_placement(global_head_count,
                          local_head_count,
                          context.tp_rank,
                          context.tp_size,
                          &first_global_head,
                          &replica_count,
                          error)) {
    return false;
  }
  const bool replicated = replica_count > 1;

  const uint64_t bytes_per_head =
      static_cast<uint64_t>(context.linear_key_head_dim) *
      tensor.element_size();
  const uint64_t state_count = static_cast<uint64_t>(tensor.size(1));
  const uint64_t physical_state_stride = tensor_stride_bytes(tensor, 1);
  const uint64_t logical_state_stride =
      static_cast<uint64_t>(global_head_count) * bytes_per_head;
  for (int64_t local_head = 0; local_head < local_head_count; ++local_head) {
    const int64_t global_head = first_global_head + local_head;
    LogicalSpan span;
    span.logical_tensor = logical_tensor;
    span.logical_offset_bytes =
        static_cast<uint64_t>(global_head) * bytes_per_head;
    span.physical_offset_bytes =
        component_offset_bytes +
        static_cast<uint64_t>(local_head) * bytes_per_head;
    span.bytes_per_region = bytes_per_head;
    span.repeat_count = state_count;
    span.logical_stride_bytes = logical_state_stride;
    span.physical_stride_bytes = physical_state_stride;
    span.owner_tp_rank = replicated
                             ? static_cast<int32_t>(global_head * replica_count)
                             : context.tp_rank;
    descriptor->spans.emplace_back(std::move(span));
  }
  return true;
}

bool describe_conv(const CacheTensorLayoutContext& context,
                   KVCacheTensor* cache_tensor,
                   std::string* error) {
  const torch::Tensor& tensor = cache_tensor->tensor;
  if (!tensor.is_contiguous() || tensor.dim() != 3 ||
      context.linear_key_head_dim <= 0) {
    return fail("CONV cache tensor must use [slot, state, packed_heads]",
                error);
  }

  const int64_t local_key_heads =
      context.linear_key_head_count >= context.tp_size
          ? context.linear_key_head_count / context.tp_size
          : 1;
  const int64_t local_value_heads =
      context.linear_value_head_count >= context.tp_size
          ? context.linear_value_head_count / context.tp_size
          : 1;
  const int64_t expected_features =
      context.linear_key_head_dim * (local_key_heads * 2 + local_value_heads);
  if (tensor.size(2) != expected_features) {
    return fail("CONV packed feature size differs from producer metadata",
                error);
  }

  LogicalShardDescriptor descriptor;
  descriptor.kind = LogicalShardKind::COMPOSITE;
  descriptor.resource_scope = CacheResourceScope::SEQUENCE;
  descriptor.spans.reserve(
      static_cast<size_t>(local_key_heads * 2 + local_value_heads));
  const uint64_t bytes_per_head =
      static_cast<uint64_t>(context.linear_key_head_dim) *
      tensor.element_size();
  if (!append_conv_component(context,
                             tensor,
                             "conv_key_a",
                             context.linear_key_head_count,
                             local_key_heads,
                             /*component_offset_bytes=*/0,
                             &descriptor,
                             error) ||
      !append_conv_component(
          context,
          tensor,
          "conv_key_b",
          context.linear_key_head_count,
          local_key_heads,
          static_cast<uint64_t>(local_key_heads) * bytes_per_head,
          &descriptor,
          error) ||
      !append_conv_component(
          context,
          tensor,
          "conv_value",
          context.linear_value_head_count,
          local_value_heads,
          static_cast<uint64_t>(local_key_heads * 2) * bytes_per_head,
          &descriptor,
          error)) {
    return false;
  }
  cache_tensor->shard_descriptor = std::move(descriptor);
  return true;
}

bool is_kv_head_role(KVCacheTensorRole role) {
  return role == KVCacheTensorRole::KEY || role == KVCacheTensorRole::VALUE ||
         role == KVCacheTensorRole::KEY_SCALE ||
         role == KVCacheTensorRole::VALUE_SCALE ||
         role == KVCacheTensorRole::CACHE_SCALE;
}

}  // namespace

int64_t physical_rows_per_resource(KVCacheTensorRole role,
                                   int64_t replicated_block_pages,
                                   int64_t ssm_checkpoint_stride) {
  CHECK_GT(ssm_checkpoint_stride, 0);
  CHECK_GE(replicated_block_pages, 1);
  if (role == KVCacheTensorRole::SSM) {
    return ssm_checkpoint_stride;
  }
  if (role == KVCacheTensorRole::INDEX ||
      role == KVCacheTensorRole::INDEX_SCALE) {
    return indexer_pages_per_block();
  }
  if (role == KVCacheTensorRole::KEY || role == KVCacheTensorRole::VALUE ||
      role == KVCacheTensorRole::WINDOW) {
    return std::max<int64_t>(replicated_block_pages, 1);
  }
  return 1;
}

void record_group_block_capacities(const std::vector<KVCache>& kv_caches,
                                   CacheTensorLayoutContext* tensor_layout) {
  CHECK(tensor_layout != nullptr);
  // MLA contexts describe every tensor through the replicated-tensor
  // contract, which never reads a pool span, and their K/V tensors (NPU NZ
  // layout) legitimately carry different token-axis spans within one pool.
  // Recording would only trip the pool-span consistency check.
  if (tensor_layout->enable_mla) {
    return;
  }
  const int64_t token_axis = tensor_layout->head_major_layout ? 2 : 1;
  for (const KVCache& cache : kv_caches) {
    for (const KVCacheTensor& cache_tensor : cache.get_cache_tensors()) {
      // Only the K/V roles define their pool's token-axis span: compressed
      // KPool gives INDEX and INDEX_SCALE a narrower span than the K/V
      // tensors sharing their group, and their geometry is already covered
      // by index_block_capacity and the group's K/V span.
      if (!is_kv_head_role(cache_tensor.role)) {
        continue;
      }
      const torch::Tensor& tensor = cache_tensor.tensor;
      // Tensors without the token axis define no pool span; leave them to
      // describe_cache_tensor, which rejects them with a precise error.
      if (!tensor.defined() || tensor.dim() <= token_axis) {
        continue;
      }
      const int64_t span = tensor.size(token_axis);
      // Uniform pools run at the declared block capacity: recording their
      // span would make describe validate the tensor against its own size,
      // muting the producer-capacity check for every untyped model. Only
      // pools deviating from the uniform capacity define an override.
      if (span == tensor_layout->block_token_capacity) {
        continue;
      }
      const auto [entry, inserted] =
          tensor_layout->group_block_capacities.emplace(cache_tensor.group_id,
                                                        span);
      CHECK(inserted || entry->second == span)
          << "Cache tensors of one pool must share the token-axis span, "
          << "group_id=" << cache_tensor.group_id;
    }
  }
}

int64_t effective_block_token_capacity(const CacheTensorLayoutContext& context,
                                       int32_t group_id) {
  const auto it = context.group_block_capacities.find(group_id);
  if (it != context.group_block_capacities.end()) {
    return it->second;
  }
  return context.block_token_capacity;
}

bool describe_cache_tensor(const CacheTensorLayoutContext& context,
                           KVCacheTensor* cache_tensor,
                           std::string* error) {
  if (cache_tensor == nullptr || !cache_tensor->tensor.defined() ||
      cache_tensor->tensor.numel() <= 0) {
    return fail("cache tensor must be defined and non-empty", error);
  }
  if (cache_tensor->shard_descriptor.has_value()) {
    return true;
  }
  // A logical index block owns `indexer_pages_per_block()` physical pages on
  // the NPU, so the replicated descriptor of an index cache must span the whole
  // block instead of a single page.
  const int64_t index_pages = indexer_pages_per_block();
  if (index_pages > 1 &&
      (cache_tensor->role == KVCacheTensorRole::INDEX ||
       cache_tensor->role == KVCacheTensorRole::INDEX_SCALE)) {
    return describe_replicated_index_pages(index_pages, cache_tensor, error);
  }

  // Typed-block pools cover a pool-specific token span per resource; resolve
  // it once so every describe path below validates against the tensor's own
  // pool geometry instead of the uniform scheduler block size.
  CacheTensorLayoutContext effective_context = context;
  effective_context.block_token_capacity =
      effective_block_token_capacity(context, cache_tensor->group_id);

#if !defined(USE_NPU) && !defined(USE_MLU)
  // Preserve the existing MLA transfer contract on other backends.
  if (effective_context.enable_mla) {
    return describe_replicated_tensor(cache_tensor, error);
  }
#endif
  // NPU/MLU hybrid MLA + linear-attention models shard recurrent state by TP.
  // Resolve CONV/SSM before the MLA replica fallback so each destination rank
  // receives the corresponding source rank's state rather than rank 0's copy.
  // The CONV/SSM descriptors consume only the linear-attention geometry and
  // never read a pool token span, so they take the unmodified context.
  if (cache_tensor->role == KVCacheTensorRole::CONV &&
      context.linear_key_head_count > 0 &&
      context.linear_value_head_count > 0) {
    return describe_conv(context, cache_tensor, error);
  }
  if (cache_tensor->role == KVCacheTensorRole::SSM &&
      context.linear_value_head_count > 0) {
    return describe_ssm(context, cache_tensor, error);
  }
#if defined(USE_NPU) || defined(USE_MLU)
  // NPU/MLU MLA attention and index caches are logical replicas even when
  // physical cache formats differ from ordinary K/V layouts. The deterministic
  // TP owner prevents duplicate writes while every destination replica is
  // populated.
  if (effective_context.enable_mla) {
    return describe_replicated_tensor(cache_tensor, error);
  }
#endif
  if (is_kv_head_role(cache_tensor->role) &&
      effective_context.kv_head_count > 0) {
    return describe_attention_heads(effective_context,
                                    effective_context.kv_head_count,
                                    effective_context.replicated_block_pages,
                                    cache_tensor,
                                    error);
  }
  if (cache_tensor->role == KVCacheTensorRole::INDEX ||
      cache_tensor->role == KVCacheTensorRole::INDEX_SCALE) {
    // The lightning indexer query uses index_head_count heads, but its cached
    // key is a single shared logical head. KVCacheShape and the DeepSeek V4
    // grouped cache both allocate INDEX/INDEX_SCALE with a physical head
    // dimension of one on every TP rank.
    CacheTensorLayoutContext index_context = effective_context;
    if (effective_context.index_block_capacity > 0) {
      index_context.block_token_capacity =
          effective_context.index_block_capacity;
    }
    return describe_attention_heads(index_context,
                                    /*global_head_count=*/1,
                                    /*replicated_block_pages=*/1,
                                    cache_tensor,
                                    error);
  }

  // Roles whose producer exposes no head axis are explicit whole-resource
  // replicas. This covers grouped state tensors and preserves their independent
  // logical identities instead of treating them as KEY-shaped bytes.
  return describe_replicated_tensor(cache_tensor, error);
}

}  // namespace xllm
