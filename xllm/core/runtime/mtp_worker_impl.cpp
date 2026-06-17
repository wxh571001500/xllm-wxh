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

#include "mtp_worker_impl.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_set>

#include "common/global_flags.h"
#include "common/metrics.h"
#if defined(USE_MLU)
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#endif
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/speculative_config.h"
#include "framework/model_loader.h"
#include "framework/request/mm_data.h"
#include "spec_input_builder.h"
#include "util/env_var.h"
#include "util/pretty_print.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
constexpr uint64_t MBUF_SIZE = 128 * 1024 * 1024;

namespace {
void scale_dp_global_token_nums_for_speculative_width(
    ModelInputParams& input_params,
    int32_t width,
    const char* /*stage*/) {
  CHECK_GT(width, 0) << "speculative width must be positive";
  std::vector<int32_t>& token_nums = input_params.parallel.dp_global_token_nums;
  if (token_nums.empty()) {
    return;
  }

  for (int32_t& token_num : token_nums) {
    if (token_num > 0) {
      token_num *= width;
    }
  }
  if (input_params.parallel.dp_is_decode.size() == token_nums.size()) {
    for (size_t i = 0; i < token_nums.size(); ++i) {
      input_params.parallel.dp_is_decode[i] = token_nums[i] > 0 ? 1 : 0;
    }
  } else if (token_nums.size() > 1) {
    input_params.parallel.dp_is_decode.reserve(token_nums.size());
    input_params.parallel.dp_is_decode.clear();
    for (const int32_t token_num : token_nums) {
      input_params.parallel.dp_is_decode.emplace_back(token_num > 0 ? 1 : 0);
    }
  }
}

torch::Tensor make_cpu_int_tensor(const std::vector<int32_t>& values) {
  return torch::tensor(values,
                       torch::TensorOptions()
                           .dtype(torch::kInt)
                           .device(torch::kCPU)
                           .pinned_memory(true));
}

bool mtp_speculative_algorithm_is_eagle3() {
  return ::xllm::SpeculativeConfig::get_instance().speculative_algorithm() ==
         "Eagle3";
}

bool is_kimi_k25_eagle3_draft(const std::string& target_model_type,
                              const std::string& draft_model_type) {
  return mtp_speculative_algorithm_is_eagle3() &&
         target_model_type == "kimi_k25" &&
         draft_model_type == "kimi_k25_eagle3";
}

int64_t local_tp_size_from_parallel_args(const ParallelArgs& parallel_args) {
  const int64_t world_size = std::max<int64_t>(parallel_args.world_size(), 1);
  const int64_t dp_size = std::max<int64_t>(parallel_args.dp_size(), 1);
  const int64_t cp_size = std::max<int64_t>(parallel_args.cp_size(), 1);
  const int64_t divisor = dp_size * cp_size;
  CHECK_EQ(world_size % divisor, 0)
      << "world_size must be divisible by dp_size * cp_size"
      << ", world_size=" << world_size << ", dp_size=" << dp_size
      << ", cp_size=" << cp_size;
  return std::max<int64_t>(1, world_size / divisor);
}

bool mtp_decode_kv_debug_enabled() {
  static const bool enabled =
      util::get_bool_env("XLLM_MTP_DECODE_KV_DEBUG", false);
  return enabled;
}

bool mtp_clear_rejected_target_kv_enabled() {
  static const bool enabled =
      util::get_bool_env("XLLM_MTP_CLEAR_REJECTED_TARGET_KV", true);
  return enabled;
}

bool mtp_preclear_multi_request_validate_kv_enabled() {
  static const bool enabled =
      util::get_bool_env("XLLM_MTP_PRECLEAR_MULTI_REQUEST_VALIDATE_KV", false);
  return enabled;
}

bool mtp_target_only_on_multi_request_enabled() {
  static const bool enabled =
      util::get_bool_env("XLLM_MTP_TARGET_ONLY_ON_MULTI_REQUEST", false);
  return enabled;
}

bool mtp_isolate_multi_request_validate_enabled() {
  static const bool enabled =
      util::get_bool_env("XLLM_MTP_ISOLATE_MULTI_REQUEST_VALIDATE", false);
  return enabled;
}

bool mtp_kimi_k25_eagle3_replica_draft_enabled() {
  static const bool enabled =
      util::get_bool_env("XLLM_KIMI_K25_EAGLE3_REPLICA_DRAFT", false);
  return enabled;
}

int64_t mtp_decode_kv_debug_rank_filter() {
  static const int64_t rank =
      util::get_int_env("XLLM_MTP_DECODE_KV_DEBUG_RANK", -1);
  return rank;
}

int64_t mtp_decode_kv_debug_max_logs() {
  static const int64_t max_logs =
      util::get_int_env("XLLM_MTP_DECODE_KV_DEBUG_MAX_LOGS", 512);
  return max_logs;
}

bool should_log_mtp_decode_kv_debug(int64_t rank) {
  if (!mtp_decode_kv_debug_enabled()) {
    return false;
  }
  const int64_t rank_filter = mtp_decode_kv_debug_rank_filter();
  if (rank_filter >= 0 && rank != rank_filter) {
    return false;
  }
  static std::atomic<int64_t> log_count{0};
  const int64_t current_count =
      log_count.fetch_add(1, std::memory_order_relaxed);
  return current_count < mtp_decode_kv_debug_max_logs();
}

std::string tensor_shape_debug_string(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream oss;
  oss << "[";
  for (int32_t i = 0; i < tensor.dim(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << tensor.size(i);
  }
  oss << "]";
  return oss.str();
}

std::string int_vector_debug_string(const std::vector<int32_t>& values,
                                    int64_t limit = 32) {
  std::ostringstream oss;
  oss << "[";
  const int64_t count =
      std::min<int64_t>(static_cast<int64_t>(values.size()), limit);
  for (int64_t i = 0; i < count; ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  if (static_cast<int64_t>(values.size()) > limit) {
    oss << ",...";
  }
  oss << "]";
  return oss.str();
}

std::string string_vector_debug_string(const std::vector<std::string>& values,
                                       int64_t limit = 16) {
  std::ostringstream oss;
  oss << "[";
  const int64_t count =
      std::min<int64_t>(static_cast<int64_t>(values.size()), limit);
  for (int64_t i = 0; i < count; ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  if (static_cast<int64_t>(values.size()) > limit) {
    oss << ",...";
  }
  oss << "]";
  return oss.str();
}

torch::Tensor cpu_flat_tensor(const torch::Tensor& tensor,
                              torch::ScalarType dtype) {
  if (!tensor.defined()) {
    return torch::Tensor();
  }
  return safe_to(tensor.flatten(),
                 torch::TensorOptions().dtype(dtype).device(torch::kCPU),
                 false)
      .contiguous();
}

std::string tensor_int_values_debug_string(const torch::Tensor& tensor,
                                           int64_t limit = 32) {
  torch::Tensor cpu_tensor = cpu_flat_tensor(tensor, torch::kLong);
  if (!cpu_tensor.defined()) {
    return "undefined";
  }
  const int64_t* values = cpu_tensor.data_ptr<int64_t>();
  std::ostringstream oss;
  oss << "[";
  const int64_t count = std::min<int64_t>(cpu_tensor.numel(), limit);
  for (int64_t i = 0; i < count; ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  if (cpu_tensor.numel() > limit) {
    oss << ",...";
  }
  oss << "]";
  return oss.str();
}

double tensor_l2_norm(const torch::Tensor& tensor) {
  torch::Tensor cpu_tensor = cpu_flat_tensor(tensor, torch::kFloat);
  if (!cpu_tensor.defined() || cpu_tensor.numel() == 0) {
    return -1.0;
  }
  const float* values = cpu_tensor.data_ptr<float>();
  double sum = 0.0;
  for (int64_t i = 0; i < cpu_tensor.numel(); ++i) {
    const double value = static_cast<double>(values[i]);
    sum += value * value;
  }
  return std::sqrt(sum);
}

torch::Tensor select_kv_slot_tensor(const torch::Tensor& cache,
                                    int64_t slot,
                                    int32_t block_size) {
  if (!cache.defined() || slot < 0 || block_size <= 0 || cache.dim() < 2) {
    return torch::Tensor();
  }
  const int64_t block_id = slot / block_size;
  const int64_t block_offset = slot % block_size;
  if (block_id < 0 || block_id >= cache.size(0)) {
    return torch::Tensor();
  }
  if (cache.size(1) == block_size && block_offset < cache.size(1)) {
    return cache.select(/*dim=*/0, block_id).select(/*dim=*/0, block_offset);
  }
  if (cache.dim() >= 3 && cache.size(2) == block_size &&
      block_offset < cache.size(2)) {
    return cache.select(/*dim=*/0, block_id).select(/*dim=*/1, block_offset);
  }
  return torch::Tensor();
}

torch::Tensor select_embedding_row_tensor(const torch::Tensor& embedding,
                                          size_t row,
                                          size_t request_idx,
                                          size_t slot_count,
                                          size_t request_count) {
  if (!embedding.defined() || embedding.numel() == 0) {
    return torch::Tensor();
  }
  if (embedding.dim() == 3 &&
      embedding.size(0) * embedding.size(1) ==
          static_cast<int64_t>(slot_count)) {
    return embedding.view({-1, embedding.size(-1)})
        .select(/*dim=*/0, static_cast<int64_t>(row));
  }
  if (embedding.dim() == 2 &&
      embedding.size(0) == static_cast<int64_t>(slot_count)) {
    return embedding.select(/*dim=*/0, static_cast<int64_t>(row));
  }
  if (embedding.dim() == 2 &&
      embedding.size(0) == static_cast<int64_t>(request_count)) {
    return embedding.select(/*dim=*/0, static_cast<int64_t>(request_idx));
  }
  if (embedding.dim() == 1) {
    return embedding;
  }
  return torch::Tensor();
}

std::vector<int32_t> debug_layer_indices(int32_t num_layers) {
  std::vector<int32_t> layers;
  if (num_layers <= 0) {
    return layers;
  }
  layers.emplace_back(0);
  if (num_layers > 2) {
    layers.emplace_back(num_layers / 2);
  }
  if (num_layers > 1) {
    layers.emplace_back(num_layers - 1);
  }
  return layers;
}

std::string kv_slot_l2_debug_string(const std::vector<KVCache>& kv_caches,
                                    int64_t slot,
                                    int32_t block_size) {
  if (slot < 0 || kv_caches.empty()) {
    return "[]";
  }
  std::ostringstream oss;
  oss << "[";
  const std::vector<int32_t> layers =
      debug_layer_indices(static_cast<int32_t>(kv_caches.size()));
  for (size_t i = 0; i < layers.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    const int32_t layer = layers[i];
    const torch::Tensor k_slot =
        select_kv_slot_tensor(kv_caches[layer].get_k_cache(), slot, block_size);
    const torch::Tensor v_slot =
        select_kv_slot_tensor(kv_caches[layer].get_v_cache(), slot, block_size);
    oss << "{layer=" << layer
        << ",k_shape=" << tensor_shape_debug_string(k_slot)
        << ",v_shape=" << tensor_shape_debug_string(v_slot)
        << ",k_l2=" << tensor_l2_norm(k_slot)
        << ",v_l2=" << tensor_l2_norm(v_slot) << "}";
  }
  oss << "]";
  return oss.str();
}

int64_t cpu_int_tensor_value(const torch::Tensor& tensor, int64_t index) {
  torch::Tensor cpu_tensor = cpu_flat_tensor(tensor, torch::kLong);
  if (!cpu_tensor.defined() || index < 0 || index >= cpu_tensor.numel()) {
    return -1;
  }
  return cpu_tensor.data_ptr<int64_t>()[index];
}

int64_t cpu_int_tensor_value_from_flat(const torch::Tensor& cpu_tensor,
                                       int64_t index) {
  if (!cpu_tensor.defined() || index < 0 || index >= cpu_tensor.numel()) {
    return -1;
  }
  return cpu_tensor.data_ptr<int64_t>()[index];
}

int32_t seq_len_debug_value(const std::vector<int32_t>& seq_lens, size_t row) {
  if (seq_lens.empty()) {
    return -1;
  }
  Slice<int32_t> seq_lens_slice = seq_lens;
  if (row >= seq_lens_slice.size()) {
    return -1;
  }
  return specBuilder::calc_kv_len(
      seq_lens_slice, static_cast<int32_t>(row), /*offset=*/0);
}

std::string position_row_debug_string(const torch::Tensor& cpu_positions,
                                      size_t row) {
  if (!cpu_positions.defined()) {
    return "undefined";
  }
  std::ostringstream oss;
  if (cpu_positions.dim() == 2) {
    if (cpu_positions.size(0) != 3 ||
        row >= static_cast<size_t>(cpu_positions.size(1))) {
      return "out_of_range";
    }
    const int32_t* data = cpu_positions.data_ptr<int32_t>();
    const int64_t stride = cpu_positions.size(1);
    oss << "[" << data[row] << "," << data[stride + row] << ","
        << data[stride * 2 + row] << "]";
    return oss.str();
  }
  if (cpu_positions.dim() == 1) {
    if (row >= static_cast<size_t>(cpu_positions.numel())) {
      return "out_of_range";
    }
    oss << cpu_positions.data_ptr<int32_t>()[row];
    return oss.str();
  }
  return tensor_shape_debug_string(cpu_positions);
}

std::string block_table_prefix_debug_string(const torch::Tensor& cpu_block_tables,
                                            size_t row,
                                            int32_t max_values = 8) {
  if (!cpu_block_tables.defined()) {
    return "undefined";
  }
  if (cpu_block_tables.dim() != 2 ||
      row >= static_cast<size_t>(cpu_block_tables.size(0))) {
    return "out_of_range";
  }
  const int32_t width = static_cast<int32_t>(cpu_block_tables.size(1));
  const int32_t count = std::min(max_values, width);
  const int32_t* data = cpu_block_tables[row].data_ptr<int32_t>();
  std::ostringstream oss;
  oss << "[";
  for (int32_t i = 0; i < count; ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << data[i];
  }
  if (count < width) {
    oss << ",...";
  }
  oss << "]";
  return oss.str();
}

std::string logits_topk_debug_string(const torch::Tensor& logits,
                                     size_t row,
                                     int32_t k = 5) {
  if (!logits.defined()) {
    return "undefined";
  }
  if (logits.dim() < 2 || logits.size(-1) <= 0) {
    return tensor_shape_debug_string(logits);
  }
  const int64_t vocab_size = logits.size(-1);
  torch::Tensor flat_logits = logits.reshape({-1, vocab_size});
  if (row >= static_cast<size_t>(flat_logits.size(0))) {
    return "out_of_range";
  }
  torch::Tensor row_logits =
      safe_to(flat_logits[static_cast<int64_t>(row)],
              torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU),
              false)
          .contiguous();
  const int64_t top_k = std::min<int64_t>(k, row_logits.numel());
  auto topk_result = torch::topk(row_logits, top_k);
  torch::Tensor values = std::get<0>(topk_result).contiguous();
  torch::Tensor indices = std::get<1>(topk_result).contiguous();
  const float* value_data = values.data_ptr<float>();
  const int64_t* index_data = indices.data_ptr<int64_t>();

  std::ostringstream oss;
  oss << "[";
  for (int64_t i = 0; i < top_k; ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << "{token=" << index_data[i] << ",logit=" << value_data[i] << "}";
  }
  oss << "]";
  return oss.str();
}

int64_t logits_row_count(const torch::Tensor& logits) {
  if (!logits.defined() || logits.dim() < 2 || logits.size(-1) <= 0) {
    return -1;
  }
  const int64_t vocab_size = logits.size(-1);
  return logits.reshape({-1, vocab_size}).size(0);
}

void clear_kv_slot(const KVCache& kv_cache, int64_t slot, int32_t block_size) {
  if (slot < 0) {
    return;
  }
  torch::Tensor k_slot = select_kv_slot_tensor(
      kv_cache.get_k_cache(), slot, block_size);
  if (k_slot.defined()) {
    k_slot.zero_();
  }
  torch::Tensor v_slot = select_kv_slot_tensor(
      kv_cache.get_v_cache(), slot, block_size);
  if (v_slot.defined()) {
    v_slot.zero_();
  }
}

void clear_kv_slots_tensor(torch::Tensor cache,
                           const std::vector<int32_t>& slots,
                           int32_t block_size) {
  if (!cache.defined() || slots.empty() || block_size <= 0 ||
      cache.dim() < 2) {
    return;
  }

  std::vector<int64_t> block_ids;
  std::vector<int64_t> block_offsets;
  block_ids.reserve(slots.size());
  block_offsets.reserve(slots.size());
  for (const int32_t slot : slots) {
    if (slot < 0) {
      continue;
    }
    const int64_t block_id = slot / block_size;
    const int64_t block_offset = slot % block_size;
    if (block_id < 0 || block_id >= cache.size(0)) {
      continue;
    }
    block_ids.emplace_back(block_id);
    block_offsets.emplace_back(block_offset);
  }
  if (block_ids.empty()) {
    return;
  }

  torch::TensorOptions index_options =
      torch::TensorOptions().dtype(torch::kLong).device(cache.device());
  torch::Tensor block_tensor = torch::tensor(block_ids, index_options);
  torch::Tensor offset_tensor = torch::tensor(block_offsets, index_options);

  if (cache.size(1) == block_size) {
    cache.index_put_({block_tensor, offset_tensor}, 0);
    return;
  }
  if (cache.dim() >= 3 && cache.size(2) == block_size) {
    cache.index_put_({block_tensor, torch::indexing::Slice(), offset_tensor},
                     0);
  }
}

void clear_kv_slots(const KVCache& kv_cache,
                    const std::vector<int32_t>& slots,
                    int32_t block_size) {
  clear_kv_slots_tensor(kv_cache.get_k_cache(), slots, block_size);
  clear_kv_slots_tensor(kv_cache.get_v_cache(), slots, block_size);
}

std::string target_kv_slot_owner_key(
    const std::vector<std::string>& request_ids,
    const std::vector<int32_t>& embedding_ids,
    int32_t seq_id) {
  if (seq_id >= 0 && seq_id < static_cast<int32_t>(request_ids.size()) &&
      !request_ids[seq_id].empty()) {
    return request_ids[seq_id];
  }
  if (seq_id >= 0 && seq_id < static_cast<int32_t>(embedding_ids.size())) {
    return std::to_string(embedding_ids[seq_id]);
  }
  return "";
}

bool has_multiple_request_ids(const std::vector<std::string>& request_ids) {
  std::unordered_set<std::string> unique_request_ids;
  for (const std::string& request_id : request_ids) {
    if (request_id.empty()) {
      continue;
    }
    unique_request_ids.insert(request_id);
    if (unique_request_ids.size() > 1) {
      return true;
    }
  }
  return false;
}

bool has_target_only_request_id(
    const std::vector<std::string>& request_ids,
    const std::unordered_set<std::string>& target_only_request_ids) {
  for (const std::string& request_id : request_ids) {
    if (!request_id.empty() &&
        target_only_request_ids.find(request_id) !=
            target_only_request_ids.end()) {
      return true;
    }
  }
  return false;
}

void mark_target_only_request_ids(
    const std::vector<std::string>& request_ids,
    std::unordered_set<std::string>& target_only_request_ids) {
  for (const std::string& request_id : request_ids) {
    if (!request_id.empty()) {
      target_only_request_ids.insert(request_id);
    }
  }
}

void refresh_decode_input_device_tensors(ForwardInput& input,
                                         const torch::Device& device) {
  CHECK(input.token_ids_host.defined())
      << "decode token_ids_host must be defined";
  CHECK(input.positions_host.defined())
      << "decode positions_host must be defined";
  torch::TensorOptions token_options = input.token_ids.options();
  torch::TensorOptions position_options = input.positions.options();
  input.device_tensors_ready = false;
  input.token_ids = safe_to(input.token_ids_host, token_options, true);
  input.positions = safe_to(input.positions_host, position_options, true);
  input.input_params.attention.rebuild_device_buffer(device);
  input.device_tensors_ready = true;
}

void update_target_decode_kv_slot_owners(
    const ForwardInput& input,
    const std::vector<std::string>& request_ids,
    const std::vector<int32_t>& embedding_ids,
    std::unordered_map<int32_t, std::string>& slot_owners) {
  const std::vector<int32_t>& slots =
      input.input_params.attention.host.new_cache_slots;
  const int32_t num_sequences =
      static_cast<int32_t>(input.input_params.embedding.embedding_ids.size());
  if (slots.size() < static_cast<size_t>(num_sequences)) {
    return;
  }
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t slot = slots[seq_id];
    if (slot < 0) {
      continue;
    }
    const std::string current_owner =
        target_kv_slot_owner_key(request_ids, embedding_ids, seq_id);
    if (!current_owner.empty()) {
      slot_owners[slot] = current_owner;
    }
  }
}

torch::Tensor make_long_index_tensor(const std::vector<int64_t>& indices,
                                     const torch::Device& device) {
  return torch::tensor(
      indices, torch::TensorOptions().dtype(torch::kLong).device(device));
}

std::vector<int32_t> slice_int_vector(const std::vector<int32_t>& values,
                                      int32_t row_start,
                                      int32_t row_count) {
  if (values.empty()) {
    return {};
  }
  CHECK_GE(row_start, 0);
  CHECK_GE(row_count, 0);
  CHECK_LE(static_cast<size_t>(row_start + row_count), values.size());
  return std::vector<int32_t>(values.begin() + row_start,
                              values.begin() + row_start + row_count);
}

std::vector<int32_t> slice_seq_lens_vector(const std::vector<int32_t>& values,
                                           int32_t row_start,
                                           int32_t row_count,
                                           int32_t total_rows) {
  if (values.empty()) {
    return {};
  }
  const bool is_cumsum =
      values.size() == static_cast<size_t>(total_rows + 1) &&
      !values.empty() && values.front() == 0;
  if (!is_cumsum) {
    return slice_int_vector(values, row_start, row_count);
  }

  std::vector<int32_t> result;
  result.reserve(static_cast<size_t>(row_count) + 1);
  result.emplace_back(0);
  for (int32_t row = 0; row < row_count; ++row) {
    const int32_t src_row = row_start + row;
    const int32_t len = values[src_row + 1] - values[src_row];
    result.emplace_back(result.back() + len);
  }
  return result;
}

std::vector<int32_t> make_q_cu_seq_lens(const std::vector<int32_t>& q_seq_lens,
                                        int32_t row_count) {
  std::vector<int32_t> result;
  result.reserve(row_count);
  if (q_seq_lens.empty()) {
    for (int32_t row = 0; row < row_count; ++row) {
      result.emplace_back(row + 1);
    }
    return result;
  }
  const bool is_cumsum =
      q_seq_lens.size() == static_cast<size_t>(row_count + 1) &&
      q_seq_lens.front() == 0;
  int32_t sum = 0;
  for (int32_t row = 0; row < row_count; ++row) {
    const int32_t len =
        is_cumsum ? q_seq_lens[row + 1] - q_seq_lens[row] : q_seq_lens[row];
    sum += len;
    result.emplace_back(sum);
  }
  return result;
}

torch::Tensor index_select_if_defined(const torch::Tensor& tensor,
                                      const std::vector<int64_t>& rows,
                                      int64_t dim) {
  if (!tensor.defined()) {
    return torch::Tensor();
  }
  torch::Tensor index = make_long_index_tensor(rows, tensor.device());
  return tensor.index_select(dim, index).contiguous();
}

void slice_sampling_params(SamplingParameters& sampling_params,
                           const std::vector<int64_t>& rows,
                           const torch::Device& device) {
  const int64_t row_count = static_cast<int64_t>(rows.size());
  torch::TensorOptions int_options =
      sampling_params.selected_token_idxes.defined()
          ? sampling_params.selected_token_idxes.options()
          : torch::TensorOptions().dtype(torch::kInt).device(device);
  sampling_params.selected_token_idxes =
      torch::arange(row_count, int_options).contiguous();
  sampling_params.sample_idxes =
      torch::arange(row_count, int_options).contiguous();
  sampling_params.frequency_penalties =
      index_select_if_defined(sampling_params.frequency_penalties, rows, 0);
  sampling_params.presence_penalties =
      index_select_if_defined(sampling_params.presence_penalties, rows, 0);
  sampling_params.repetition_penalties =
      index_select_if_defined(sampling_params.repetition_penalties, rows, 0);
  sampling_params.temperatures =
      index_select_if_defined(sampling_params.temperatures, rows, 0);
  sampling_params.top_p =
      index_select_if_defined(sampling_params.top_p, rows, 0);
  sampling_params.top_k =
      index_select_if_defined(sampling_params.top_k, rows, 0);
  sampling_params.unique_token_ids =
      index_select_if_defined(sampling_params.unique_token_ids, rows, 0);
  sampling_params.unique_token_counts =
      index_select_if_defined(sampling_params.unique_token_counts, rows, 0);
  sampling_params.unique_token_ids_lens =
      index_select_if_defined(sampling_params.unique_token_ids_lens, rows, 0);
  sampling_params.do_sample =
      index_select_if_defined(sampling_params.do_sample, rows, 0);
  sampling_params.acc_logprob =
      index_select_if_defined(sampling_params.acc_logprob, rows, 0);
}

ForwardInput slice_validate_input_rows(const ForwardInput& input,
                                       int32_t row_start,
                                       int32_t row_count,
                                       const torch::Device& device) {
  CHECK_GE(row_start, 0);
  CHECK_GT(row_count, 0);
  std::vector<int64_t> rows;
  rows.reserve(row_count);
  for (int32_t row = 0; row < row_count; ++row) {
    rows.emplace_back(row_start + row);
  }

  ForwardInput output = input;
  output.device_tensors_ready = false;
  torch::TensorOptions token_options = input.token_ids.options();
  torch::TensorOptions position_options = input.positions.options();

  output.token_ids_host =
      index_select_if_defined(input.token_ids_host, rows, 0)
          .to(torch::kCPU)
          .contiguous();
  if (input.positions_host.dim() == 2) {
    output.positions_host =
        index_select_if_defined(input.positions_host, rows, 1)
            .to(torch::kCPU)
            .contiguous();
  } else {
    output.positions_host =
        index_select_if_defined(input.positions_host, rows, 0)
            .to(torch::kCPU)
            .contiguous();
  }
  output.token_ids = safe_to(output.token_ids_host, token_options, true);
  output.positions = safe_to(output.positions_host, position_options, true);

  ModelInputParams& params = output.input_params;
  const int32_t total_rows = input.input_params.meta.num_sequences;
  params.meta.num_sequences = row_count;
  params.attention.host.new_cache_slots = slice_int_vector(
      input.input_params.attention.host.new_cache_slots, row_start, row_count);
  params.attention.host.q_seq_lens = slice_seq_lens_vector(
      input.input_params.attention.host.q_seq_lens,
      row_start,
      row_count,
      total_rows);
  params.attention.host.kv_seq_lens = slice_seq_lens_vector(
      input.input_params.attention.host.kv_seq_lens,
      row_start,
      row_count,
      total_rows);
  params.attention.host.q_cu_seq_lens =
      make_q_cu_seq_lens(params.attention.host.q_seq_lens, row_count);
  if (input.input_params.attention.host.block_tables.defined()) {
    params.attention.host.block_tables =
        index_select_if_defined(input.input_params.attention.host.block_tables,
                                rows,
                                0)
            .to(torch::kCPU)
            .contiguous();
  }
  params.meta.kv_max_seq_len = 0;
  Slice<int32_t> kv_seq_lens = params.attention.host.kv_seq_lens;
  for (int32_t row = 0; row < row_count; ++row) {
    params.meta.kv_max_seq_len = std::max(
        params.meta.kv_max_seq_len,
        specBuilder::calc_kv_len(kv_seq_lens, row, /*offset=*/0));
  }
  params.attention.rebuild_device_buffer(device);
  slice_sampling_params(output.sampling_params, rows, device);
  output.device_tensors_ready = true;
  return output;
}

void clear_rejected_target_kv(
    const ForwardInput& validate_input,
    const SampleOutput& accepted_output,
    const std::vector<KVCache>& kv_caches,
    int32_t block_size,
    int32_t num_sequences) {
  if (!mtp_clear_rejected_target_kv_enabled()) {
    return;
  }
  if (!accepted_output.next_tokens.defined() ||
      validate_input.input_params.attention.host.new_cache_slots.empty()) {
    return;
  }

  torch::Tensor accepted_tokens =
      cpu_flat_tensor(accepted_output.next_tokens, torch::kLong);
  if (!accepted_tokens.defined()) {
    return;
  }
  const std::vector<int32_t>& slots =
      validate_input.input_params.attention.host.new_cache_slots;
  const int32_t num_rows =
      static_cast<int32_t>(accepted_output.next_tokens.numel());
  if (num_sequences <= 0 || slots.empty()) {
    return;
  }
  if (num_rows % num_sequences != 0) {
    return;
  }

  const int32_t width = num_rows / num_sequences;
  if (width <= 0 || slots.size() < static_cast<size_t>(num_rows)) {
    return;
  }

  torch::NoGradGuard no_grad;
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t row_base = seq_id * width;
    int32_t accepted_len = 0;
    for (int32_t row_offset = 0; row_offset < width; ++row_offset) {
      const int64_t token = cpu_int_tensor_value_from_flat(
          accepted_tokens, row_base + row_offset);
      if (token < 0) {
        break;
      }
      ++accepted_len;
    }
    if (accepted_len <= 0 || accepted_len >= width) {
      continue;
    }
    for (int32_t row_offset = accepted_len; row_offset < width; ++row_offset) {
      const int64_t slot = slots[row_base + row_offset];
      for (const KVCache& kv_cache : kv_caches) {
        clear_kv_slot(kv_cache, slot, block_size);
      }
    }
  }
}

void preclear_multi_request_validate_kv(
    const ForwardInput& validate_input,
    const std::vector<KVCache>& kv_caches,
    int32_t block_size,
    int32_t num_sequences,
    const std::vector<std::string>& request_ids,
    const std::vector<int32_t>& embedding_ids,
    const std::unordered_map<int32_t, std::string>& slot_owners) {
  if (!mtp_preclear_multi_request_validate_kv_enabled() ||
      num_sequences <= 1) {
    return;
  }
  const std::vector<int32_t>& slots =
      validate_input.input_params.attention.host.new_cache_slots;
  if (slots.empty()) {
    return;
  }
  if (slots.size() % static_cast<size_t>(num_sequences) != 0) {
    return;
  }
  const int32_t width =
      static_cast<int32_t>(slots.size() / static_cast<size_t>(num_sequences));
  if (width <= 1) {
    return;
  }

  std::vector<int32_t> foreign_slots;
  foreign_slots.reserve(slots.size());
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const std::string current_owner =
        target_kv_slot_owner_key(request_ids, embedding_ids, seq_id);
    if (current_owner.empty()) {
      continue;
    }
    const int32_t row_base = seq_id * width;
    for (int32_t row_offset = 0; row_offset < width; ++row_offset) {
      const int32_t slot = slots[row_base + row_offset];
      const auto owner_it = slot_owners.find(slot);
      if (owner_it != slot_owners.end() && owner_it->second != current_owner) {
        foreign_slots.emplace_back(slot);
      }
    }
  }
  if (foreign_slots.empty()) {
    return;
  }

  torch::NoGradGuard no_grad;
  for (const KVCache& kv_cache : kv_caches) {
    clear_kv_slots(kv_cache, foreign_slots, block_size);
  }
}

void update_target_kv_slot_owners(
    const ForwardInput& validate_input,
    const SampleOutput& accepted_output,
    int32_t num_sequences,
    const std::vector<std::string>& request_ids,
    const std::vector<int32_t>& embedding_ids,
    std::unordered_map<int32_t, std::string>& slot_owners) {
  if (!accepted_output.next_tokens.defined()) {
    return;
  }
  const std::vector<int32_t>& slots =
      validate_input.input_params.attention.host.new_cache_slots;
  if (num_sequences <= 0 || slots.empty()) {
    return;
  }
  const int32_t num_rows =
      static_cast<int32_t>(accepted_output.next_tokens.numel());
  if (num_rows % num_sequences != 0) {
    return;
  }
  const int32_t width = num_rows / num_sequences;
  if (width <= 0 || slots.size() < static_cast<size_t>(num_rows)) {
    return;
  }
  torch::Tensor accepted_tokens =
      cpu_flat_tensor(accepted_output.next_tokens, torch::kLong);
  if (!accepted_tokens.defined()) {
    return;
  }

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const std::string current_owner =
        target_kv_slot_owner_key(request_ids, embedding_ids, seq_id);
    if (current_owner.empty()) {
      continue;
    }
    const int32_t row_base = seq_id * width;
    int32_t accepted_len = 0;
    for (int32_t row_offset = 0; row_offset < width; ++row_offset) {
      const int64_t token = cpu_int_tensor_value_from_flat(
          accepted_tokens, row_base + row_offset);
      if (token < 0) {
        break;
      }
      ++accepted_len;
    }
    for (int32_t row_offset = 0; row_offset < width; ++row_offset) {
      const int32_t slot = slots[row_base + row_offset];
      if (slot < 0) {
        continue;
      }
      if (row_offset < accepted_len || !mtp_clear_rejected_target_kv_enabled()) {
        slot_owners[slot] = current_owner;
      } else {
        slot_owners.erase(slot);
      }
    }
  }
}

void log_mtp_decode_input_debug(const char* stage,
                                int64_t rank,
                                const ForwardInput& input) {
  if (!should_log_mtp_decode_kv_debug(rank)) {
    return;
  }
  const ModelInputParams& params = input.input_params;
  const torch::Tensor& token_ids =
      input.token_ids_host.defined() ? input.token_ids_host : input.token_ids;
  const torch::Tensor& positions =
      input.positions_host.defined() ? input.positions_host : input.positions;
  LOG(INFO) << "[MTP_DECODE_KV_DEBUG]"
            << " stage=" << stage
            << " rank=" << rank
            << " batch_id=" << params.meta.batch_id
            << " batch_type=" << params.meta.batch_forward_type.to_string()
            << " num_sequences=" << params.meta.num_sequences
            << " request_ids="
            << string_vector_debug_string(params.embedding.request_ids)
            << " embedding_ids="
            << int_vector_debug_string(params.embedding.embedding_ids)
            << " token_ids=" << tensor_int_values_debug_string(token_ids)
            << " token_ids_host="
            << tensor_int_values_debug_string(input.token_ids_host)
            << " token_ids_device="
            << tensor_int_values_debug_string(input.token_ids)
            << " positions=" << tensor_int_values_debug_string(positions)
            << " q_seq_lens="
            << int_vector_debug_string(params.attention.host.q_seq_lens)
            << " kv_seq_lens="
            << int_vector_debug_string(params.attention.host.kv_seq_lens)
            << " new_cache_slots="
            << int_vector_debug_string(params.attention.host.new_cache_slots)
            << " input_embedding_shape="
            << tensor_shape_debug_string(params.embedding.input_embedding)
            << " selected_token_idxes="
            << tensor_int_values_debug_string(
                   input.sampling_params.selected_token_idxes)
            << " sample_idxes="
            << tensor_int_values_debug_string(input.sampling_params.sample_idxes)
            << " input_embedding_l2="
            << tensor_l2_norm(params.embedding.input_embedding);
}

void log_mtp_decode_kv_slots_debug(const char* stage,
                                   int64_t rank,
                                   const ForwardInput& input,
                                   const std::vector<KVCache>& kv_caches,
                                   const torch::Tensor& next_tokens,
                                   const torch::Tensor& logits,
                                   int32_t block_size) {
  if (!should_log_mtp_decode_kv_debug(rank)) {
    return;
  }
  const ModelInputParams& params = input.input_params;
  const std::vector<int32_t>& slots = params.attention.host.new_cache_slots;
  const size_t request_count = params.embedding.request_ids.size();
  const size_t rows_per_request =
      request_count > 0 && slots.size() % request_count == 0
          ? slots.size() / request_count
          : 1;
  torch::Tensor cpu_host_token_ids =
      cpu_flat_tensor(input.token_ids_host, torch::kLong);
  torch::Tensor cpu_device_token_ids =
      cpu_flat_tensor(input.token_ids, torch::kLong);
  torch::Tensor cpu_next_tokens = cpu_flat_tensor(next_tokens, torch::kLong);
  const torch::Tensor& positions =
      input.positions_host.defined() ? input.positions_host : input.positions;
  torch::Tensor cpu_positions =
      positions.defined()
          ? safe_to(positions,
                    torch::TensorOptions()
                        .dtype(torch::kInt)
                        .device(torch::kCPU),
                    false)
                .contiguous()
          : torch::Tensor();
  torch::Tensor cpu_block_tables =
      params.attention.host.block_tables.defined()
          ? safe_to(params.attention.host.block_tables,
                    torch::TensorOptions()
                        .dtype(torch::kInt)
                        .device(torch::kCPU),
                    false)
                .contiguous()
          : torch::Tensor();
  const int64_t logits_rows = logits_row_count(logits);
  for (size_t row = 0; row < slots.size(); ++row) {
    const int64_t row_index = static_cast<int64_t>(row);
    const int64_t token_id_host =
        cpu_int_tensor_value_from_flat(cpu_host_token_ids, row_index);
    const int64_t token_id_device =
        cpu_int_tensor_value_from_flat(cpu_device_token_ids, row_index);
    const int64_t token_id =
        token_id_device >= 0 ? token_id_device : token_id_host;
    const int64_t next_token =
        cpu_int_tensor_value_from_flat(cpu_next_tokens, row_index);
    const size_t request_idx =
        request_count == 0
            ? row
            : std::min(row / rows_per_request, request_count - 1);
    const size_t row_in_request =
        rows_per_request > 0 ? row % rows_per_request : row;
    const std::string request_id =
        request_idx < params.embedding.request_ids.size()
            ? params.embedding.request_ids[request_idx]
            : "";
    const int32_t embedding_id =
        request_idx < params.embedding.embedding_ids.size()
            ? params.embedding.embedding_ids[request_idx]
            : -1;
    const int32_t q_len =
        seq_len_debug_value(params.attention.host.q_seq_lens, row);
    const int32_t kv_len =
        seq_len_debug_value(params.attention.host.kv_seq_lens, row);
    const int32_t slot = slots[row];
    const int32_t slot_block = block_size > 0 ? slot / block_size : -1;
    const int32_t slot_offset = block_size > 0 ? slot % block_size : -1;
    const size_t logits_row =
        logits_rows == static_cast<int64_t>(slots.size()) ? row : request_idx;
    const torch::Tensor embedding_row = select_embedding_row_tensor(
        params.embedding.input_embedding,
        row,
        request_idx,
        slots.size(),
        request_count);
    LOG(INFO) << "[MTP_DECODE_KV_DEBUG]"
              << " stage=" << stage
              << " rank=" << rank
              << " batch_id=" << params.meta.batch_id
              << " row=" << row
              << " row_in_request=" << row_in_request
              << " request_id=" << request_id
              << " embedding_id=" << embedding_id
              << " token_id=" << token_id
              << " token_id_host=" << token_id_host
              << " token_id_device=" << token_id_device
              << " next_token=" << next_token
              << " position=" << position_row_debug_string(cpu_positions, row)
              << " q_len=" << q_len
              << " kv_len=" << kv_len
              << " new_cache_slot=" << slot
              << " slot_block=" << slot_block
              << " slot_offset=" << slot_offset
              << " block_table_prefix="
              << block_table_prefix_debug_string(cpu_block_tables, row)
              << " input_embedding_row_l2=" << tensor_l2_norm(embedding_row)
              << " top_logits="
              << logits_topk_debug_string(logits, logits_row)
              << " kv_l2="
              << kv_slot_l2_debug_string(kv_caches, slot, block_size);
  }
}

void log_mtp_sample_rows_debug(const char* stage,
                               int64_t rank,
                               const ForwardInput& input,
                               const SampleOutput& output,
                               const torch::Tensor& logits) {
  if (!should_log_mtp_decode_kv_debug(rank)) {
    return;
  }
  const ModelInputParams& params = input.input_params;
  torch::Tensor cpu_next_tokens =
      cpu_flat_tensor(output.next_tokens, torch::kLong);
  const int64_t row_count =
      output.next_tokens.defined() ? output.next_tokens.numel() : 0;
  for (int64_t row = 0; row < row_count; ++row) {
    const std::string request_id =
        row < static_cast<int64_t>(params.embedding.request_ids.size())
            ? params.embedding.request_ids[static_cast<size_t>(row)]
            : "";
    const int32_t embedding_id =
        row < static_cast<int64_t>(params.embedding.embedding_ids.size())
            ? params.embedding.embedding_ids[static_cast<size_t>(row)]
            : -1;
    const torch::Tensor embedding_row =
        output.embeddings.defined() && output.embeddings.dim() >= 2 &&
                row < output.embeddings.size(0)
            ? output.embeddings[row]
            : torch::Tensor();
    LOG(INFO) << "[MTP_DECODE_KV_DEBUG]"
              << " stage=" << stage
              << " rank=" << rank
              << " batch_id=" << params.meta.batch_id
              << " row=" << row
              << " request_id=" << request_id
              << " embedding_id=" << embedding_id
              << " next_token="
              << cpu_int_tensor_value_from_flat(cpu_next_tokens, row)
              << " selected_token_idxes="
              << tensor_int_values_debug_string(
                     input.sampling_params.selected_token_idxes)
              << " sample_idxes="
              << tensor_int_values_debug_string(input.sampling_params.sample_idxes)
              << " embedding_row_l2=" << tensor_l2_norm(embedding_row)
              << " logits_shape=" << tensor_shape_debug_string(logits)
              << " top_logits="
              << logits_topk_debug_string(logits, static_cast<size_t>(row));
  }
}

void log_mtp_validate_result_debug(const char* stage,
                                   int64_t rank,
                                   const ForwardInput& input,
                                   const SampleOutput& output,
                                   int32_t num_speculative_tokens) {
  if (!should_log_mtp_decode_kv_debug(rank)) {
    return;
  }
  const int32_t width = num_speculative_tokens + 1;
  torch::Tensor accepted_tokens = output.next_tokens;
  if (accepted_tokens.defined() && accepted_tokens.dim() == 1 &&
      accepted_tokens.numel() % width == 0) {
    accepted_tokens = accepted_tokens.view({-1, width});
  }
  const int64_t batch_size =
      accepted_tokens.defined() && accepted_tokens.dim() == 2
          ? accepted_tokens.size(0)
          : 0;
  torch::Tensor cpu_tokens = cpu_flat_tensor(accepted_tokens, torch::kLong);
  const int64_t* values =
      cpu_tokens.defined() ? cpu_tokens.data_ptr<int64_t>() : nullptr;
  for (int64_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    int32_t accepted_len = 0;
    std::ostringstream token_oss;
    token_oss << "[";
    for (int32_t j = 0; j < width; ++j) {
      if (j > 0) {
        token_oss << ",";
      }
      const int64_t token = values[seq_id * width + j];
      token_oss << token;
      if (token >= 0) {
        ++accepted_len;
      }
    }
    token_oss << "]";
    const int32_t accepted_draft_count =
        std::min(std::max(accepted_len - 1, 0), num_speculative_tokens);
    const bool all_draft_accepted =
        accepted_draft_count == num_speculative_tokens;
    const std::string request_id =
        seq_id < static_cast<int64_t>(
                     input.input_params.embedding.request_ids.size())
            ? input.input_params.embedding.request_ids[seq_id]
            : "";
    const int32_t embedding_id =
        seq_id < static_cast<int64_t>(
                     input.input_params.embedding.embedding_ids.size())
            ? input.input_params.embedding.embedding_ids[seq_id]
            : -1;
    LOG(INFO) << "[MTP_DECODE_KV_DEBUG]"
              << " stage=" << stage
              << " rank=" << rank
              << " batch_id=" << input.input_params.meta.batch_id
              << " seq_id=" << seq_id
              << " request_id=" << request_id
              << " embedding_id=" << embedding_id
              << " accepted_len=" << accepted_len
              << " accepted_draft_count=" << accepted_draft_count
              << " all_draft_accepted=" << all_draft_accepted
              << " accepted_tokens=" << token_oss.str()
              << " accepted_embeddings_l2="
              << tensor_l2_norm(output.embeddings.defined()
                                    ? output.embeddings[seq_id]
                                    : torch::Tensor());
  }
}

std::vector<int32_t> make_int_range(int32_t size) {
  CHECK_GE(size, 0) << "range size must be non-negative";
  std::vector<int32_t> values;
  values.reserve(size);
  for (int32_t i = 0; i < size; ++i) {
    values.emplace_back(i);
  }
  return values;
}

int32_t local_dp_token_count(const ModelInputParams& input_params,
                             const ParallelArgs& parallel_args) {
  const std::vector<int32_t>& token_nums =
      input_params.parallel.dp_global_token_nums;
  if (token_nums.empty() || parallel_args.dp_local_process_group_ == nullptr) {
    return 1;
  }

  const int64_t dp_rank = parallel_args.dp_local_process_group_->rank();
  CHECK_LT(dp_rank, static_cast<int64_t>(token_nums.size()))
      << "dp rank exceeds dp_global_token_nums size";
  return std::max<int32_t>(1, token_nums[dp_rank]);
}

void set_token_ids_device_tensor(ForwardInput& input,
                                 const torch::Tensor& token_ids,
                                 const torch::TensorOptions& token_options) {
  CHECK(token_ids.defined()) << "draft token_ids must be defined";
  torch::Tensor flat_token_ids = token_ids.flatten();
  CHECK_EQ(flat_token_ids.numel(), input.input_params.meta.num_sequences)
      << "draft token count must match num_sequences";

  input.device_tensors_ready = false;
  input.token_ids_host = torch::Tensor();
  input.token_ids = safe_to(flat_token_ids, token_options, true);
  input.device_tensors_ready = true;
}

torch::Tensor to_cpu_int_tensor_for_read(const torch::Tensor& values) {
  return safe_to(values.flatten(),
                 torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU),
                 false)
      .contiguous();
}

void replace_host_token_placeholders(ForwardInput& input,
                                     int32_t placeholder,
                                     const torch::Tensor& replacements,
                                     const torch::TensorOptions& token_options,
                                     bool refresh_device = true) {
  CHECK(replacements.defined())
      << "speculative replacement tokens must be defined";
  CHECK(input.token_ids_host.defined())
      << "token_ids_host must be defined before speculative token update";
  CHECK(input.token_ids_host.device().is_cpu())
      << "token_ids_host must stay on CPU";
  CHECK_EQ(input.token_ids_host.scalar_type(), torch::kInt)
      << "token_ids_host must be int32";

  input.device_tensors_ready = false;
  torch::Tensor replacement_cpu = to_cpu_int_tensor_for_read(replacements);
  int32_t* token_ids = input.token_ids_host.data_ptr<int32_t>();
  const size_t num_token_ids =
      static_cast<size_t>(input.token_ids_host.numel());
  Slice<int32_t> replacement_ids = {
      replacement_cpu.data_ptr<int32_t>(),
      static_cast<size_t>(replacement_cpu.numel())};

  size_t replacement_idx = 0;
  for (size_t i = 0; i < num_token_ids; ++i) {
    if (token_ids[i] != placeholder) {
      continue;
    }
    CHECK_LT(replacement_idx, replacement_ids.size())
        << "not enough speculative replacement tokens";
    token_ids[i] = replacement_ids[replacement_idx++];
  }
  CHECK_EQ(replacement_idx, replacement_ids.size())
      << "unused speculative replacement tokens";

  if (refresh_device) {
    input.token_ids = safe_to(input.token_ids_host, token_options, true);
    input.device_tensors_ready = true;
  }
}

void set_token_position_tensors(ForwardInput& input,
                                const std::vector<int32_t>& token_ids,
                                const torch::Tensor& positions,
                                const torch::TensorOptions& token_options,
                                const torch::TensorOptions& position_options) {
  input.device_tensors_ready = false;
  input.token_ids_host = make_cpu_int_tensor(token_ids);
  input.positions_host = positions;
  input.token_ids = safe_to(input.token_ids_host, token_options, true);
  input.positions = safe_to(input.positions_host, position_options, true);
  input.device_tensors_ready = true;
}

void set_positions_tensor(ForwardInput& input,
                          const torch::Tensor& positions,
                          const torch::TensorOptions& device_options) {
  input.device_tensors_ready = false;
  input.positions_host = positions;
  input.positions = safe_to(input.positions_host, device_options, true);
  input.device_tensors_ready = true;
}

ForwardInput make_fake_prefill_input_for_empty_shard(
    const ForwardInput& input,
    const torch::Device& device,
    torch::ScalarType dtype) {
  ForwardInput fake_input = input;
  fake_input.device_tensors_ready = false;
  fake_input.input_host_buffer = torch::Tensor();
  fake_input.device_input_buffer = torch::Tensor();
  fake_input.input_host_buffer_has_layout = false;

  auto int_options = torch::TensorOptions()
                         .dtype(torch::kInt)
                         .device(torch::kCPU)
                         .pinned_memory(true);
  fake_input.token_ids_host = torch::tensor({1}, int_options);
  if (input.positions_host.defined() && input.positions_host.dim() == 2) {
    fake_input.positions_host = torch::zeros({3, 1}, int_options);
  } else {
    fake_input.positions_host = torch::tensor({0}, int_options);
  }
  fake_input.token_ids = fake_input.token_ids_host;
  fake_input.positions = fake_input.positions_host;

  ModelInputParams& input_params = fake_input.input_params;
  input_params.meta.num_sequences = 1;
  input_params.meta.q_max_seq_len = 1;
  input_params.meta.kv_max_seq_len = 1;
  input_params.attention.host.q_seq_lens = {1};
  input_params.attention.host.q_cu_seq_lens = {1};
  input_params.attention.host.kv_seq_lens = {1};
  input_params.attention.host.new_cache_slots = {0};
  input_params.attention.host.kv_cache_tokens_nums = {0};
  input_params.attention.host.block_tables = torch::zeros({1, 1}, int_options);
  input_params.attention.device.q_seq_lens =
      torch::tensor(input_params.attention.host.q_seq_lens, int_options);
  input_params.attention.device.q_cu_seq_lens =
      torch::tensor(input_params.attention.host.q_cu_seq_lens, int_options);
  input_params.attention.device.kv_seq_lens =
      torch::tensor(input_params.attention.host.kv_seq_lens, int_options);
  input_params.attention.device.new_cache_slots =
      torch::tensor(input_params.attention.host.new_cache_slots, int_options);
  input_params.attention.device.kv_cache_tokens_nums = torch::tensor(
      input_params.attention.host.kv_cache_tokens_nums, int_options);
  input_params.attention.device.block_tables =
      input_params.attention.host.block_tables;
  input_params.attention.rebuild_device_buffer(device);

  input_params.embedding.input_embedding = torch::Tensor();
  input_params.embedding.embedding_ids.clear();
  input_params.embedding.request_ids.clear();
  input_params.embedding.extra_token_ids = {1};
  input_params.embedding.mtp_shifted_token_ids = torch::Tensor();

  for (int32_t& token_num : input_params.parallel.dp_global_token_nums) {
    if (token_num == 0) {
      token_num = 1;
    }
  }

  SamplingParameters& sampling_params = fake_input.sampling_params;
  sampling_params.selected_token_idxes = torch::tensor({0}, int_options);
  sampling_params.sample_idxes = torch::tensor({0}, int_options);
  sampling_params.do_sample = torch::tensor(
      {false}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
  sampling_params.all_greedy_sample = true;
  sampling_params.all_random_sample = false;

  return fake_input.to(device, dtype);
}

ForwardInput make_fake_decode_input_for_empty_shard(
    const ForwardInput& input,
    const torch::Device& device,
    torch::ScalarType dtype,
    int32_t num_tokens,
    int32_t block_size,
    const torch::Tensor& input_embedding) {
  CHECK_GT(num_tokens, 0) << "fake decode token count must be positive";
  CHECK_GT(block_size, 0) << "block size must be positive";

  ForwardInput fake_input = input;
  fake_input.device_tensors_ready = false;
  fake_input.input_host_buffer = torch::Tensor();
  fake_input.device_input_buffer = torch::Tensor();
  fake_input.input_host_buffer_has_layout = false;

  const torch::TensorOptions int_options = torch::TensorOptions()
                                               .dtype(torch::kInt)
                                               .device(torch::kCPU)
                                               .pinned_memory(true);
  const std::vector<int32_t> selected_idxes = make_int_range(num_tokens);
  std::vector<int32_t> token_ids(num_tokens, 0);
  std::vector<int32_t> positions = selected_idxes;
  fake_input.token_ids_host = make_cpu_int_tensor(token_ids);
  fake_input.positions_host = make_cpu_int_tensor(positions);
  fake_input.token_ids = fake_input.token_ids_host;
  fake_input.positions = fake_input.positions_host;

  ModelInputParams& input_params = fake_input.input_params;
  input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input_params.meta.num_sequences = num_tokens;
  input_params.meta.q_max_seq_len = 1;
  input_params.meta.kv_max_seq_len = num_tokens;

  input_params.attention.host.q_seq_lens.clear();
  input_params.attention.host.q_cu_seq_lens.clear();
  input_params.attention.host.kv_seq_lens.clear();
  input_params.attention.host.new_cache_slots.clear();
  input_params.attention.host.kv_cache_tokens_nums.clear();
  input_params.attention.host.q_seq_lens.reserve(num_tokens);
  input_params.attention.host.q_cu_seq_lens.reserve(num_tokens);
  input_params.attention.host.kv_seq_lens.reserve(num_tokens);
  input_params.attention.host.new_cache_slots.reserve(num_tokens);
  input_params.attention.host.kv_cache_tokens_nums.reserve(num_tokens);
  for (int32_t i = 0; i < num_tokens; ++i) {
    specBuilder::append_q_seq_len(input_params.attention.host.q_seq_lens,
                                  input_params.attention.host.q_cu_seq_lens,
                                  1);
    specBuilder::append_seq_len_by_layout(
        input_params.attention.host.kv_seq_lens, i + 1);
    input_params.attention.host.new_cache_slots.emplace_back(i);
    input_params.attention.host.kv_cache_tokens_nums.emplace_back(i);
  }

  const int32_t block_table_stride =
      std::max<int32_t>(1, (num_tokens + block_size - 1) / block_size);
  input_params.attention.host.block_tables =
      torch::zeros({num_tokens, block_table_stride}, int_options);
  input_params.attention.rebuild_device_buffer(device);

  if (input_embedding.defined()) {
    torch::Tensor embedding = input_embedding;
    if (embedding.dim() == 1) {
      embedding = embedding.unsqueeze(0).repeat({num_tokens, 1});
    } else {
      CHECK_EQ(embedding.dim(), 2)
          << "fake draft input embedding must be 1D or 2D";
      if (embedding.size(0) == 1 && num_tokens > 1) {
        embedding = embedding.repeat({num_tokens, 1});
      }
      CHECK_EQ(embedding.size(0), num_tokens)
          << "fake draft input embedding rows must match fake token count";
    }
    input_params.embedding.input_embedding = safe_to(embedding, device, true);
  } else {
    input_params.embedding.input_embedding = torch::Tensor();
  }
  input_params.embedding.embedding_ids.clear();
  input_params.embedding.request_ids.clear();
  input_params.embedding.extra_token_ids.clear();
  input_params.embedding.mtp_shifted_token_ids = torch::Tensor();

  SamplingParameters& sampling_params = fake_input.sampling_params;
  sampling_params.selected_token_idxes =
      safe_to(make_cpu_int_tensor(selected_idxes), device, true);
  sampling_params.sample_idxes =
      safe_to(make_cpu_int_tensor(selected_idxes), device, true);
  sampling_params.do_sample = safe_to(torch::zeros({num_tokens},
                                                   torch::TensorOptions()
                                                       .dtype(torch::kBool)
                                                       .device(torch::kCPU)
                                                       .pinned_memory(true)),
                                      device,
                                      true);
  sampling_params.all_greedy_sample = true;
  sampling_params.all_random_sample = false;

  fake_input.device_tensors_ready = false;
  return fake_input.to(device, dtype);
}

runtime::Options MTPTargetOptions(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false).is_draft_engine(false);
  return opts;
}

runtime::Options MTPDraftOptions(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(true)
      .num_decoding_tokens(1)
      .num_speculative_tokens(0);
  return opts;
}

}  // namespace

MTPWorkerImpl::MTPWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options,
                             WorkerType worker_type)
    : MTPWorkerImpl(parallel_args,
                    device,
                    options,
                    MTPTargetOptions(options),
                    MTPDraftOptions(options),
                    worker_type,
                    ::xllm::SpeculativeConfig::get_instance()
                        .enable_opt_validate_probs()) {}

MTPWorkerImpl::MTPWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options,
                             const runtime::Options& target_options,
                             const runtime::Options& draft_options,
                             WorkerType worker_type,
                             bool enable_opt_validate_probs)
    : SpeculativeWorkerImpl(parallel_args,
                            device,
                            options,
                            target_options,
                            worker_type),
      draft_options_(draft_options),
      enable_opt_validate_probs_(enable_opt_validate_probs) {
  draft_impl_ =
      std::make_unique<LLMWorkerImpl>(parallel_args, device, draft_options);
}

bool MTPWorkerImpl::init_model(const std::string& model_weights_path,
                               int32_t random_seed,
                               MasterStatus master_status) {
  // Load target model via base class
  bool result = true;
  if (target_impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = SpeculativeWorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::UNINITIALIZED);
    draft_impl_ = create_draft_worker(model_weights_path);
    result = draft_impl_->WorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
  }

  if (draft_impl_ != nullptr &&
      draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    // Share lm_head and word_embedding between target and draft models
#if defined(USE_NPU)
    if (::xllm::KernelConfig::get_instance().npu_kernel_backend() != "TORCH") {
      auto head = target_impl_->get_npu_lm_head();
      draft_impl_->set_npu_lm_head(head);
      auto word_embedding = target_impl_->get_npu_word_embedding();
      draft_impl_->set_npu_word_embedding(word_embedding);
    } else {
      auto head = target_impl_->get_lm_head();
      draft_impl_->set_lm_head(head);
      auto word_embedding = target_impl_->get_word_embedding();
      draft_impl_->set_word_embedding(word_embedding);
    }
#else
    auto head = target_impl_->get_lm_head();
    draft_impl_->set_lm_head(head);
    auto word_embedding = target_impl_->get_word_embedding();
    draft_impl_->set_word_embedding(word_embedding);
#endif
    // Sync context_ from target_impl_ for
    // WorkerImpl::prepare_work_before_execute.
    context_ = target_impl_->context_;
  }
  return result;
}

int64_t MTPWorkerImpl::get_embedding_placeholder_size() {
  return static_cast<int64_t>(embedding_size_);
}

std::unique_ptr<LLMWorkerImpl> MTPWorkerImpl::create_draft_worker(
    const std::string& model_weights_path) const {
  auto model_loader = ModelLoader::create(model_weights_path);
  const std::string& draft_model_type = model_loader->model_args().model_type();
  const std::string& target_model_type =
      target_impl_->context_.get_model_args().model_type();
  const bool is_kimi_k25_eagle3 =
      is_kimi_k25_eagle3_draft(target_model_type, draft_model_type);
  const bool use_replica_draft =
      mtp_kimi_k25_eagle3_replica_draft_enabled() && is_kimi_k25_eagle3;
  if (!use_replica_draft) {
    if (is_kimi_k25_eagle3) {
      LOG(INFO) << "Creating TP Eagle3 draft worker for kimi_k25"
                << ", target_model_type=" << target_model_type
                << ", draft_model_type=" << draft_model_type
                << ", target_rank=" << parallel_args_.rank()
                << ", target_world_size=" << parallel_args_.world_size()
                << ", target_dp_size=" << parallel_args_.dp_size()
                << ", target_cp_size=" << parallel_args_.cp_size()
                << ", target_ep_size=" << parallel_args_.ep_size();
    }
    return std::make_unique<LLMWorkerImpl>(
        parallel_args_, device(), draft_options_);
  }

  ParallelArgs draft_parallel_args(
      /*rank=*/0,
      /*world_size=*/1,
      /*dp_size=*/1,
      /*cp_size=*/1,
      /*process_group=*/nullptr,
      /*ep_size=*/1);
  LOG(INFO) << "Creating replica Eagle3 draft worker for kimi_k25"
            << ", target_model_type=" << target_model_type
            << ", draft_model_type=" << draft_model_type
            << ", target_rank=" << parallel_args_.rank()
            << ", target_world_size=" << parallel_args_.world_size()
            << ", target_dp_size=" << parallel_args_.dp_size()
            << ", target_cp_size=" << parallel_args_.cp_size()
            << ", target_ep_size=" << parallel_args_.ep_size();
  return std::make_unique<LLMWorkerImpl>(
      draft_parallel_args, device(), draft_options_);
}

bool MTPWorkerImpl::should_use_separate_draft_kv_cache_shape() const {
  if (target_impl_ == nullptr || draft_impl_ == nullptr) {
    return false;
  }
  return is_kimi_k25_eagle3_draft(
      target_impl_->context_.get_model_args().model_type(),
      draft_impl_->context_.get_model_args().model_type());
}

KVCacheShape MTPWorkerImpl::get_draft_kv_cache_shape(
    const KVCacheShape& target_kv_cache_shape) const {
  CHECK(should_use_separate_draft_kv_cache_shape())
      << "separate draft KV cache shape is only supported for kimi_k25 Eagle3";
  CHECK(!target_kv_cache_shape.key_cache_shape().empty())
      << "target KV cache shape must contain key cache shape";

  const int64_t num_blocks = target_kv_cache_shape.key_cache_shape()[0];
  CHECK_GT(num_blocks, 0) << "draft KV cache num_blocks must be positive";

  const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
  const ParallelArgs& draft_parallel_args =
      draft_impl_->context_.get_parallel_args();
  const int64_t draft_tp_size =
      local_tp_size_from_parallel_args(draft_parallel_args);

  KVCacheCapacity draft_kv_cache_cap;
  draft_kv_cache_cap.n_blocks(num_blocks)
      .block_size(options_.block_size())
      .n_layers(draft_args.num_nextn_predict_layers() > 0
                    ? draft_args.num_nextn_predict_layers()
                    : draft_args.n_layers());

  const int64_t dtype_size =
      static_cast<int64_t>(torch::elementSize(draft_impl_->dtype()));
  const int64_t total_kv_heads =
      draft_args.n_kv_heads().value_or(draft_args.n_heads());
  const int64_t local_kv_heads =
      std::max<int64_t>(1, total_kv_heads / draft_tp_size);
  const int64_t cache_dtype_size =
      options_.kv_cache_dtype() == "auto" ? dtype_size : 1;

  draft_kv_cache_cap
      .slot_size(2 * cache_dtype_size * draft_args.head_dim() * local_kv_heads)
      .index_slot_size(draft_args.index_n_heads() > 0
                           ? dtype_size * draft_args.index_head_dim()
                           : 0)
      .scale_slot_size(options_.kv_cache_dtype() == "auto"
                           ? 0
                           : 2 * sizeof(float) * local_kv_heads);

  LOG(INFO) << "Using separate Eagle3 draft KV cache shape for kimi_k25"
            << ", num_blocks=" << num_blocks
            << ", draft_tp_size=" << draft_tp_size
            << ", replica_draft="
            << mtp_kimi_k25_eagle3_replica_draft_enabled()
            << ", draft_world_size=" << draft_parallel_args.world_size()
            << ", draft_dp_size=" << draft_parallel_args.dp_size()
            << ", draft_cp_size=" << draft_parallel_args.cp_size()
            << ", local_kv_heads=" << local_kv_heads
            << ", slot_size=" << draft_kv_cache_cap.slot_size()
            << ", index_slot_size=" << draft_kv_cache_cap.index_slot_size()
            << ", scale_slot_size=" << draft_kv_cache_cap.scale_slot_size()
            << ", n_layers=" << draft_kv_cache_cap.n_layers();
  return KVCacheShape(draft_kv_cache_cap, draft_args, draft_tp_size);
}

bool MTPWorkerImpl::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape.key_cache_shape()[0];
  // init_model() must run first so dtype_/embedding_size_ are initialized.
  embedding_cache_ = std::make_shared<EmbeddingCache>(num_blocks);
  if (embedding_cache_) {
    embedding_cache_->set_probs_placeholder(
        torch::ones({}, torch::dtype(torch::kFloat32).device(torch::kCPU)));
    int64_t size = get_embedding_placeholder_size();
    if (size > 0) {
      embedding_cache_->set_placeholder(
          torch::zeros({size}, torch::dtype(dtype_).device(device_)));
    }
  }
  CHECK(target_impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);

  bool target_allocated = true;
  const auto target_status = target_impl_->get_status();
  if (target_status == WorkerImpl::Status::LOADED) {
    target_allocated = target_impl_->allocate_kv_cache(kv_cache_shape);
  } else {
    CHECK_EQ(target_status, WorkerImpl::Status::READY);
  }

  bool draft_allocated = true;
  const auto draft_status = draft_impl_->get_status();
  if (draft_status == WorkerImpl::Status::LOADED) {
    if (should_use_separate_draft_kv_cache_shape()) {
      draft_allocated = draft_impl_->allocate_kv_cache(
          get_draft_kv_cache_shape(kv_cache_shape));
    } else {
      draft_allocated = draft_impl_->allocate_kv_cache(kv_cache_shape);
    }
  } else {
    CHECK_EQ(draft_status, WorkerImpl::Status::READY);
  }

  return target_allocated && draft_allocated;
}

#if defined(USE_NPU) || defined(USE_MLU)
bool MTPWorkerImpl::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape.key_cache_shape()[0];
  CHECK(target_impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);

  if (kv_cache_transfer_ == nullptr) {
#if defined(USE_NPU)
    kv_cache_transfer_ = std::make_shared<SpecKVCacheTransfer>(
        options_.device_ip().value(),
        options_.transfer_listen_port(),
        options_.instance_role(),
        context_.get_model_args().model_type(),
        context_.get_model_args().index_n_heads() > 0);
#elif defined(USE_MLU)
    CHECK_EQ(::xllm::DisaggPDConfig::get_instance().kv_cache_transfer_type(),
             "Mooncake")
        << "MLU MTP push only supports Mooncake KV transfer.";
    kv_cache_transfer_ = std::make_shared<MooncakeKVCacheTransferDefault>(
        device_.index(),
        options_.transfer_listen_port(),
        device_,
        context_.get_model_args().model_type());
#endif

    int32_t device_id = device_.index();
    kv_cache_transfer_->initialize(device_id);
  }

  bool target_allocated = true;
  const auto target_status = target_impl_->get_status();
  if (target_status == WorkerImpl::Status::LOADED) {
    target_allocated = target_impl_->allocate_kv_cache_with_transfer(
        kv_cache_transfer_, kv_cache_shape);
  } else {
    CHECK_EQ(target_status, WorkerImpl::Status::READY);
  }

  bool draft_allocated = true;
  const auto draft_status = draft_impl_->get_status();
  if (draft_status == WorkerImpl::Status::LOADED) {
    if (should_use_separate_draft_kv_cache_shape()) {
      draft_allocated = draft_impl_->allocate_kv_cache_with_transfer(
          kv_cache_transfer_, get_draft_kv_cache_shape(kv_cache_shape));
    } else {
      draft_allocated = draft_impl_->allocate_kv_cache_with_transfer(
          kv_cache_transfer_, kv_cache_shape);
    }
  } else {
    CHECK_EQ(draft_status, WorkerImpl::Status::READY);
  }

  embedding_cache_ = std::make_shared<EmbeddingCache>(num_blocks);
  if (embedding_cache_) {
    embedding_cache_->set_probs_placeholder(
        torch::ones({}, torch::dtype(torch::kFloat32).device(device_)));
    int64_t size = get_embedding_placeholder_size();
    if (size > 0) {
      embedding_cache_->set_placeholder(
          torch::zeros({size}, torch::dtype(dtype_).device(device_)));
    }
  }
  return target_allocated && draft_allocated;
}
#endif

ForwardInput MTPWorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
  return inputs;
}

void MTPWorkerImpl::prepare_work_before_execute(const ForwardInput& input,
                                                ForwardInput& processed_input) {
  SpeculativeWorkerImpl::prepare_work_before_execute(input, processed_input);
}

std::optional<ForwardOutput> MTPWorkerImpl::step_empty(
    const ForwardInput& input) {
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    const bool empty_shard = input.input_params.meta.num_sequences == 0;
    const ForwardInput& target_input =
        empty_shard
            ? make_fake_prefill_input_for_empty_shard(input, device_, dtype_)
            : input;
    auto target_future = target_impl_->step_async(target_input);
    std::optional<ForwardOutput> output = std::move(target_future).get();
    ForwardInput prefill_input;
    prepare_prefill_inputs(target_input, prefill_input);
    if (!empty_shard && output.has_value()) {
      auto& embeddings = output->sample_output.embeddings;
      if (embeddings.defined()) {
        set_draft_input_embedding(prefill_input.input_params.embedding,
                                  embeddings.clone(),
                                  "prefill empty");
      }
      if (output->sample_output.next_tokens.defined()) {
        replace_host_token_placeholders(prefill_input,
                                        -1,
                                        output->sample_output.next_tokens,
                                        prefill_input.token_ids.options());
      }
    }
    auto draft_future = draft_impl_->step_async(prefill_input);
    std::optional<ForwardOutput> draft_output = std::move(draft_future).get();
    (void)draft_output;
    if (output.has_value()) {
      output->sample_output.embeddings = torch::Tensor();
    }
    return output;
  } else {
    ForwardInput new_input = input;
    scale_dp_global_token_nums_for_speculative_width(
        new_input.input_params, /*width=*/2, "empty_decode_draft_extend");
    const int32_t draft_extend_fake_tokens =
        local_dp_token_count(new_input.input_params, parallel_args_);
    CHECK(embedding_cache_ != nullptr)
        << "embedding cache must be initialized for fake draft decode";
    new_input = make_fake_decode_input_for_empty_shard(
        new_input,
        device_,
        dtype_,
        draft_extend_fake_tokens,
        options_.block_size(),
        embedding_cache_->embedding_placeholder());
    auto draft_extend_future = draft_impl_->step_async(new_input);
    ForwardOutput draft_extend_output =
        std::move(draft_extend_future).get().value();
    (void)draft_extend_output;

    for (int32_t i = 1; i < options_.num_speculative_tokens(); ++i) {
      ForwardInput draft_input = input;
      scale_dp_global_token_nums_for_speculative_width(
          draft_input.input_params, /*width=*/1, "empty_decode_draft");
      const int32_t draft_fake_tokens =
          local_dp_token_count(draft_input.input_params, parallel_args_);
      draft_input = make_fake_decode_input_for_empty_shard(
          draft_input,
          device_,
          dtype_,
          draft_fake_tokens,
          options_.block_size(),
          embedding_cache_->embedding_placeholder());
      auto draft_future = draft_impl_->step_async(draft_input);
      ForwardOutput draft_output = std::move(draft_future).get().value();
      (void)draft_output;
    }

    new_input = input;
    scale_dp_global_token_nums_for_speculative_width(
        new_input.input_params,
        options_.num_speculative_tokens() + 1,
        "empty_decode_target_validate");
    const int32_t validate_fake_tokens =
        local_dp_token_count(new_input.input_params, parallel_args_);
    new_input = make_fake_decode_input_for_empty_shard(new_input,
                                                       device_,
                                                       dtype_,
                                                       validate_fake_tokens,
                                                       options_.block_size(),
                                                       torch::Tensor());
    auto future = target_impl_->step_async(new_input);
    ForwardOutput output = std::move(future).get().value();
    output.sample_output.embeddings = torch::Tensor();
    return output;
  }
}

std::optional<ForwardOutput> MTPWorkerImpl::step_prefill(
    const ForwardInput& input) {
  Timer timer;
  // run the target model to get first token and hidden states
  auto future = target_impl_->step_async(input);
  ForwardOutput output = std::move(future).get().value();
  log_mtp_sample_rows_debug("prefill_target_output",
                            parallel_args_.rank(),
                            input,
                            output.sample_output,
                            output.logits);
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());
  // MTP path that depends on hidden states.
  ForwardInput prefill_input;
  prepare_prefill_inputs(input, prefill_input);
  // prepare input for draft model
  auto& embeddings = output.sample_output.embeddings;
  if (embeddings.defined()) {
    set_draft_input_embedding(prefill_input.input_params.embedding,
                              embeddings.clone(),
                              "prefill");
  }
  if (output.sample_output.next_tokens.defined()) {
    replace_host_token_placeholders(prefill_input,
                                    -1,
                                    output.sample_output.next_tokens,
                                    prefill_input.token_ids.options());
  }
  // generate kv cache for draft model
  timer.reset();
  auto draft_future = draft_impl_->step_async(prefill_input);
  ForwardOutput draft_output = std::move(draft_future).get().value();
  process_draft_sample_output(draft_output.sample_output);
  log_mtp_sample_rows_debug("prefill_draft_after_forward",
                            parallel_args_.rank(),
                            prefill_input,
                            draft_output.sample_output,
                            draft_output.logits);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  if (input.sampling_params.selected_token_idxes.defined()) {
    embedding_cache_->write_prefill_target_context(
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        output.sample_output.next_tokens,
        embeddings,
        input.sampling_params.selected_token_idxes);
  }
  output.sample_output.embeddings = torch::Tensor();

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

void MTPWorkerImpl::prepare_prefill_inputs(const ForwardInput& input,
                                           ForwardInput& prefill_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  prefill_input = input.to(device_, dtype_);
  auto& input_params = prefill_input.input_params;
  if (options_.cp_size() > 1) {
    CHECK(input_params.embedding.mtp_shifted_token_ids.defined());
    CHECK_EQ(input_params.embedding.mtp_shifted_token_ids.numel(),
             prefill_input.token_ids.numel());
    prefill_input.token_ids = input_params.embedding.mtp_shifted_token_ids;
    return;
  }

  auto& extra_token_ids = input_params.embedding.extra_token_ids;

  const torch::Tensor& token_ids = input.token_ids_host;
  if (input_params.meta.num_sequences == 0 || !token_ids.defined() ||
      token_ids.numel() == 0) {
    prefill_input.device_tensors_ready = false;
    torch::TensorOptions token_options =
        prefill_input.token_ids.defined()
            ? prefill_input.token_ids.options()
            : torch::TensorOptions().dtype(torch::kInt).device(device_);
    prefill_input.token_ids_host = make_cpu_int_tensor({});
    prefill_input.token_ids =
        safe_to(prefill_input.token_ids_host, token_options, true);
    prefill_input.device_tensors_ready = true;
    prepare_stream_->synchronize();
    return;
  }

  Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                     static_cast<size_t>(token_ids.numel())};

  int32_t start_idx = 0;
  std::vector<int32_t> new_token_ids;
  new_token_ids.reserve(token_ids.numel());
  for (int32_t i = 0; i < input_params.meta.num_sequences; ++i) {
    int32_t q_len = input_params.get_q_seq_len(i);
    Slice<int32_t> tokens_ids_slice_i =
        tokens_ids_slice.slice(start_idx + 1, start_idx + q_len);
    start_idx += q_len;
    new_token_ids.insert(new_token_ids.end(),
                         tokens_ids_slice_i.begin(),
                         tokens_ids_slice_i.end());
    new_token_ids.emplace_back(extra_token_ids[i]);
  }
  prefill_input.device_tensors_ready = false;
  prefill_input.token_ids_host = make_cpu_int_tensor(new_token_ids);
  prefill_input.token_ids = safe_to(
      prefill_input.token_ids_host, prefill_input.positions.options(), true);
  prefill_input.device_tensors_ready = true;
  prepare_stream_->synchronize();
}

std::optional<ForwardOutput> MTPWorkerImpl::step_decode(
    const ForwardInput& raw_input) {
  ForwardInput input = raw_input;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();

  std::vector<ForwardOutput> draft_outputs;
  ForwardInput current_draft_input, validate_input, next_step_input;
  Timer timer;

  // Get decode state of last step
  std::vector<EmbeddingCache::DecodeState> last_states =
      embedding_cache_->read_decode_states(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids);
  CHECK_EQ(last_states.size(),
           input.input_params.embedding.embedding_ids.size())
      << "decode target state count mismatch";
  update_decode_step_input(input, last_states);
  log_mtp_decode_input_debug("decode_base_after_update",
                             parallel_args_.rank(),
                             input);
  if (mtp_target_only_on_multi_request_enabled() &&
      (has_multiple_request_ids(input.input_params.embedding.request_ids) ||
       has_target_only_request_id(input.input_params.embedding.request_ids,
                                  target_only_request_ids_))) {
    refresh_decode_input_device_tensors(input, device_);
    log_mtp_decode_input_debug("target_only_multi_request_input",
                               parallel_args_.rank(),
                               input);
    auto future = target_impl_->step_async(input);
    std::optional<ForwardOutput> output = std::move(future).get();
    CHECK(output.has_value())
        << "target-only multi-request decode output is empty";
    log_mtp_sample_rows_debug("target_only_multi_request_output",
                              parallel_args_.rank(),
                              input,
                              output->sample_output,
                              output->logits);
    mark_target_only_request_ids(input.input_params.embedding.request_ids,
                                 target_only_request_ids_);
    if (output->sample_output.next_tokens.defined() &&
        output->sample_output.embeddings.defined()) {
      embedding_cache_->write_prefill_target_context(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids,
          output->sample_output.next_tokens,
          output->sample_output.embeddings,
          input.sampling_params.selected_token_idxes);
      update_target_decode_kv_slot_owners(
          input,
          input.input_params.embedding.request_ids,
          input.input_params.embedding.embedding_ids,
          target_kv_slot_owners_);
    }
    COUNTER_ADD(speculative_execution_latency_seconds_target,
                timer.elapsed_seconds());
    if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
      return std::nullopt;
    }
    output->sample_output.embeddings = torch::Tensor();
    return output;
  }
  prepare_draft_extend_inputs(input, last_states, current_draft_input);
  log_mtp_decode_input_debug("draft_extend_input",
                             parallel_args_.rank(),
                             current_draft_input);
  draft_outputs.reserve(num_speculative_tokens);
  for (int32_t draft_idx = 0; draft_idx < num_speculative_tokens; ++draft_idx) {
    auto future = draft_impl_->step_async(current_draft_input);
    // Overlap next-step input preparation with async draft forward.
    if (draft_idx == num_speculative_tokens - 1) {
      prepare_validate_inputs(input, validate_input);
    } else {
      prepare_draft_inputs(input, next_step_input, draft_idx + 1);
    }
    std::optional<ForwardOutput> draft_output_opt = std::move(future).get();
    CHECK(draft_output_opt.has_value())
        << "draft output is empty in speculative step";
    draft_outputs.push_back(std::move(draft_output_opt.value()));
    process_draft_sample_output(draft_outputs.back().sample_output);
    log_mtp_decode_kv_slots_debug(
        "draft_decode_after_forward",
        parallel_args_.rank(),
        current_draft_input,
        draft_impl_->kv_caches_for_debug(),
        draft_outputs.back().sample_output.next_tokens,
        draft_outputs.back().logits,
        options_.block_size());
    if (draft_idx == num_speculative_tokens - 1) {
      continue;
    }
    const SampleOutput& last_output = draft_outputs.back().sample_output;
    current_draft_input = next_step_input;
    set_token_ids_device_tensor(current_draft_input,
                                last_output.next_tokens,
                                current_draft_input.token_ids.options());
    set_draft_input_embedding(current_draft_input.input_params.embedding,
                              last_output.embeddings,
                              "decode");
    log_mtp_decode_input_debug("draft_next_input",
                               parallel_args_.rank(),
                               current_draft_input);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return run_validate(input, draft_outputs, validate_input);
}

void MTPWorkerImpl::fill_validate_input_from_draft_outputs(
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  CHECK_EQ(draft_outputs.size(), static_cast<size_t>(num_speculative_tokens))
      << "draft output count mismatch";
  CHECK(validate_input.token_ids.defined())
      << "validate token_ids must be prepared before draft token fill";
  CHECK_EQ(validate_input.token_ids.dim(), 1)
      << "validate token_ids must be flat";
  CHECK_EQ(validate_input.token_ids.numel() % num_val_tokens, 0)
      << "validate token_ids size must be divisible by validation width";

  const int64_t total_num_val_tokens = validate_input.token_ids.numel();
  const int64_t num_sequences = total_num_val_tokens / num_val_tokens;
  const auto token_options = validate_input.token_ids.options();
  torch::Tensor validate_token_rows =
      validate_input.token_ids.view({num_sequences, num_val_tokens});

  validate_input.device_tensors_ready = false;
  for (int32_t i = 0; i < num_speculative_tokens; ++i) {
    const auto& draft_output = draft_outputs[i];
    const torch::Tensor& next_tokens = draft_output.sample_output.next_tokens;
    CHECK(next_tokens.defined())
        << "draft next_tokens must be defined for validate token fill";
    torch::Tensor draft_tokens =
        safe_to(next_tokens.flatten(), token_options, /*non_blocking=*/true);
    CHECK_EQ(draft_tokens.numel(), num_sequences)
        << "draft token count must match validate sequence count";
    validate_token_rows.select(/*dim=*/1, /*index=*/i + 1)
        .copy_(draft_tokens, /*non_blocking=*/true);
  }
  validate_input.device_tensors_ready = true;
}

std::optional<ForwardOutput> MTPWorkerImpl::run_validate(
    const ForwardInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input) {
  // run the target model to get the verification scores
  Timer timer;
  fill_validate_input_from_draft_outputs(draft_outputs, validate_input);
  log_mtp_decode_input_debug("target_validate_input",
                             parallel_args_.rank(),
                             validate_input);
  const int32_t num_sequences =
      static_cast<int32_t>(input.input_params.embedding.embedding_ids.size());
  preclear_multi_request_validate_kv(validate_input,
                                     target_impl_->kv_caches_for_debug(),
                                     options_.block_size(),
                                     num_sequences,
                                     input.input_params.embedding.request_ids,
                                     input.input_params.embedding.embedding_ids,
                                     target_kv_slot_owners_);
  ForwardOutput target_output;
  const bool isolate_validate =
      mtp_isolate_multi_request_validate_enabled() &&
      has_multiple_request_ids(input.input_params.embedding.request_ids);
  if (isolate_validate) {
    const int32_t width = options_.num_speculative_tokens() + 1;
    std::vector<torch::Tensor> logits_parts;
    std::vector<torch::Tensor> next_token_parts;
    std::vector<torch::Tensor> embedding_parts;
    logits_parts.reserve(num_sequences);
    next_token_parts.reserve(num_sequences);
    embedding_parts.reserve(num_sequences);
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      ForwardInput seq_validate_input = slice_validate_input_rows(
          validate_input, seq_id * width, width, device_);
      auto future = target_impl_->step_async(seq_validate_input);
      std::optional<ForwardOutput> seq_output = std::move(future).get();
      CHECK(seq_output.has_value())
          << "isolated target validate output is empty, seq_id=" << seq_id;
      if (seq_id == 0) {
        target_output = std::move(seq_output.value());
        logits_parts.emplace_back(target_output.logits);
        next_token_parts.emplace_back(target_output.sample_output.next_tokens);
        if (target_output.sample_output.embeddings.defined()) {
          embedding_parts.emplace_back(target_output.sample_output.embeddings);
        }
        continue;
      }
      logits_parts.emplace_back(seq_output->logits);
      next_token_parts.emplace_back(seq_output->sample_output.next_tokens);
      if (seq_output->sample_output.embeddings.defined()) {
        embedding_parts.emplace_back(seq_output->sample_output.embeddings);
      }
    }
    target_output.logits = torch::cat(logits_parts, /*dim=*/0);
    target_output.sample_output.next_tokens =
        torch::cat(next_token_parts, /*dim=*/0);
    if (!embedding_parts.empty()) {
      target_output.sample_output.embeddings =
          torch::cat(embedding_parts, /*dim=*/0);
    }
  } else {
    auto future = target_impl_->step_async(validate_input);
    target_output = std::move(future).get().value();
  }
  log_mtp_decode_kv_slots_debug("target_validate_after_forward",
                                parallel_args_.rank(),
                                validate_input,
                                target_impl_->kv_caches_for_debug(),
                                target_output.sample_output.next_tokens,
                                target_output.logits,
                                options_.block_size());
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  // verify the proposals with target and update the batch
  timer.reset();
  SampleOutput val_output =
      validate(input.sampling_params, draft_outputs, target_output);
  log_mtp_validate_result_debug("validate_result",
                                parallel_args_.rank(),
                                input,
                                val_output,
                                options_.num_speculative_tokens());
  record_validate_metrics(val_output);
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  val_output.next_tokens = val_output.next_tokens.to(torch::kCPU);
  clear_rejected_target_kv(validate_input,
                           val_output,
                           target_impl_->kv_caches_for_debug(),
                           options_.block_size(),
                           num_sequences);
  update_target_kv_slot_owners(validate_input,
                               val_output,
                               num_sequences,
                               input.input_params.embedding.request_ids,
                               input.input_params.embedding.embedding_ids,
                               target_kv_slot_owners_);
  write_target_context_to_cache(input, val_output);

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

void MTPWorkerImpl::write_target_context_to_cache(
    const ForwardInput& input,
    const SampleOutput& validate_output) {
  CHECK(embedding_cache_ != nullptr)
      << "embedding_cache_ must be initialized before target cache write";
  CHECK(!input.input_params.embedding.embedding_ids.empty())
      << "target context cache write requires embedding ids";
  log_mtp_validate_result_debug("target_context_write",
                                parallel_args_.rank(),
                                input,
                                validate_output,
                                options_.num_speculative_tokens());
  embedding_cache_->write_target_context(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      validate_output.next_tokens,
      validate_output.embeddings,
      options_.num_speculative_tokens());
}

void MTPWorkerImpl::record_validate_metrics(
    const SampleOutput& validate_output) const {
  CHECK(validate_output.next_tokens.defined())
      << "validate output tokens are undefined";
  const int32_t batch_size =
      static_cast<int32_t>(validate_output.next_tokens.size(0));
  const int32_t num_draft_tokens =
      batch_size * options_.num_speculative_tokens();
  torch::Tensor mask = (validate_output.next_tokens == -1).to(torch::kInt64);
  const int64_t rejected_count = mask.sum().item<int64_t>();
  COUNTER_ADD(speculative_num_draft_tokens_total, num_draft_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_total,
              num_draft_tokens - rejected_count);
}

void MTPWorkerImpl::process_draft_sample_output(SampleOutput& sample_output) {
  if (sample_output.probs.defined()) {
    CHECK(sample_output.next_tokens.defined())
        << "draft sample_output.next_tokens must be defined when probs exist";
    CHECK_EQ(sample_output.next_tokens.dim(), 1)
        << "MTP draft cache expects next_tokens [batch], got "
        << sample_output.next_tokens.sizes();
    CHECK(sample_output.probs.dim() == 1 || sample_output.probs.dim() == 2)
        << "MTP draft cache expects probs [batch] or [batch,vocab], got "
        << sample_output.probs.sizes();
    CHECK_EQ(sample_output.probs.size(0), sample_output.next_tokens.size(0))
        << "MTP draft cache probs/token batch mismatch";
    // Cache always stores selected-only draft probs [batch_size] to reduce HBM.
    sample_output.probs = specBuilder::draftProbs::compress_for_cache(
        sample_output.probs, sample_output.next_tokens);
  }
}

void MTPWorkerImpl::update_decode_step_input(
    ForwardInput& input,
    const std::vector<EmbeddingCache::DecodeState>& last_states) const {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "decode context state count mismatch";
  const bool enable_cache_correction = enable_schedule_overlap();

  std::vector<int32_t> kv_seq_lens_vec;
#if defined(USE_NPU)
  kv_seq_lens_vec.reserve(num_sequences);
#else
  kv_seq_lens_vec.reserve(num_sequences + 1);
#endif

  const torch::Tensor& token_ids_cpu = input.token_ids_host;
  Slice<int32_t> input_token_ids = {token_ids_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(token_ids_cpu.numel())};
  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(num_sequences);
  buf.out_positions.reserve(input.positions_host.dim() == 2
                                ? static_cast<size_t>(num_sequences) * 3
                                : static_cast<size_t>(num_sequences));
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    CHECK_LT(static_cast<size_t>(seq_id), input_token_ids.size())
        << "decode context token seq_id out of range, seq_id=" << seq_id;
    const EmbeddingCache::DecodeState& state = last_states[seq_id];
    const int32_t input_token_id = input_token_ids[seq_id];
    const bool input_is_fake_token = input_token_id < 0;
    const bool use_cache_correction =
        enable_cache_correction && input_is_fake_token && state.valid;
    const bool use_fake_context =
        enable_cache_correction && input_is_fake_token && !state.valid;
    const int32_t position_offset =
        use_cache_correction ? state.position_offset : 0;
    const int32_t current_kv_len = specBuilder::calc_kv_len(
        input.input_params.attention.host.kv_seq_lens, seq_id, position_offset);

    if (input.positions_host.dim() != 2) {
      Slice<int32_t> input_positions = {
          input.positions_host.data_ptr<int32_t>(),
          static_cast<size_t>(input.positions_host.numel())};
      CHECK_LT(static_cast<size_t>(seq_id), input_positions.size())
          << "decode context position seq_id out of range, seq_id=" << seq_id;
      const int32_t current_position =
          input_positions[seq_id] + position_offset;
      CHECK_EQ(current_position + 1, current_kv_len)
          << "decode context position/kv_len mismatch, seq_id=" << seq_id
          << ", current_position=" << current_position
          << ", current_kv_len=" << current_kv_len;
    }

    specBuilder::RowSpec row;
    row.seq_id = seq_id;
    row.token_id = (use_cache_correction || use_fake_context) ? state.token_id
                                                              : input_token_id;
    row.position_offset = position_offset;
    row.append_kv_len = false;
    specBuilder::append_decode_row(row_ctx, row, options_.block_size(), buf);
    specBuilder::append_seq_len_by_layout(kv_seq_lens_vec, current_kv_len);
  }

  input.token_ids_host = make_cpu_int_tensor(buf.out_token_ids);
  input.positions_host = specBuilder::make_positions_tensor(buf);
  input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  input.device_tensors_ready = false;
}

void MTPWorkerImpl::prepare_validate_inputs(const ForwardInput& input,
                                            ForwardInput& validate_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  validate_input = input;
  validate_input.device_tensors_ready = false;
  auto& input_params = validate_input.input_params;
  torch::TensorOptions token_options = validate_input.token_ids.options();
  torch::TensorOptions position_options = validate_input.positions.options();

  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const int32_t total_num_val_tokens = num_sequences * num_val_tokens;
  const int32_t block_size = options_.block_size();
  const bool use_validate_chunked_prefill =
      ::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel();
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  Slice<int32_t> token_ids = {
      input.token_ids_host.data_ptr<int32_t>(),
      static_cast<size_t>(input.token_ids_host.numel())};
  Slice<int32_t> kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(total_num_val_tokens);
  buf.out_positions.reserve(input.positions_host.dim() == 2
                                ? static_cast<size_t>(total_num_val_tokens) * 3
                                : static_cast<size_t>(total_num_val_tokens));
  buf.out_new_cache_slots.reserve(total_num_val_tokens);
  if (!use_validate_chunked_prefill) {
    buf.out_kv_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_cu_seq_lens.reserve(total_num_val_tokens);
    buf.out_block_tables.reserve(static_cast<size_t>(total_num_val_tokens) *
                                 row_ctx.block_table_stride);
  }

  std::vector<int32_t> atb_kv_seq_lens_vec;
  std::vector<int32_t> atb_q_seq_lens_vec;
  std::vector<int32_t> atb_q_cu_seq_lens_vec;
  int32_t atb_kv_max_seq_len = 0;
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t kv_len =
        specBuilder::calc_kv_len(kv_seq_lens, seq_id, /*offset=*/0);
    if (input.positions_host.dim() != 2) {
      Slice<int32_t> positions = {
          input.positions_host.data_ptr<int32_t>(),
          static_cast<size_t>(input.positions_host.numel())};
      const int32_t start_position = positions[seq_id];
      CHECK_EQ(start_position + 1, kv_len)
          << "validate position/kv_len mismatch, seq_id=" << seq_id
          << ", start_position=" << start_position << ", kv_len=" << kv_len;
    }

    for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.token_id = val_idx == 0 ? token_ids[seq_id] : -val_idx;
      row.position_offset = val_idx;
      row.append_kv_len = !use_validate_chunked_prefill;
      row.append_q_len_one = !use_validate_chunked_prefill;
      row.append_block_table = !use_validate_chunked_prefill;
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
    }

    if (use_validate_chunked_prefill) {
      const int32_t kv_len_after_validation = kv_len + num_speculative_tokens;
      specBuilder::update_kv_seq_lens_and_max(
          atb_kv_seq_lens_vec, kv_len_after_validation, atb_kv_max_seq_len);
      specBuilder::append_q_seq_len(
          atb_q_seq_lens_vec, atb_q_cu_seq_lens_vec, num_val_tokens);
    }
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "validate kv slots/tokens mismatch";
  CHECK_EQ(specBuilder::position_column_count(buf), buf.out_token_ids.size())
      << "validate positions/tokens mismatch";

  set_token_position_tensors(validate_input,
                             buf.out_token_ids,
                             specBuilder::make_positions_tensor(buf),
                             token_options,
                             position_options);
  if (!use_validate_chunked_prefill) {
    input_params.meta.num_sequences = total_num_val_tokens;
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  } else {
    input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  }
  if (use_validate_chunked_prefill) {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     num_val_tokens,
                                     std::move(atb_q_seq_lens_vec),
                                     std::move(atb_q_cu_seq_lens_vec),
                                     atb_kv_max_seq_len,
                                     std::move(atb_kv_seq_lens_vec));
  } else {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     1,
                                     std::move(buf.out_q_seq_lens),
                                     std::move(buf.out_q_cu_seq_lens),
                                     buf.meta.kv_max_seq_len,
                                     std::move(buf.out_kv_seq_lens),
                                     /*update_block_tables=*/true);
  }

  update_sampling_params(
      validate_input.sampling_params, num_val_tokens, total_num_val_tokens);

  scale_dp_global_token_nums_for_speculative_width(
      input_params, num_val_tokens, "validate_target");

  input_params.attention.rebuild_device_buffer(device_);
  validate_input.device_tensors_ready = true;
  prepare_stream_->synchronize();
}

void MTPWorkerImpl::prepare_draft_extend_inputs(
    const ForwardInput& base_input,
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    ForwardInput& extend_input) {
  extend_input = base_input;
  extend_input.device_tensors_ready = false;
  auto& input_params = extend_input.input_params;
  const int32_t num_sequences = input_params.meta.num_sequences;

  const bool dp_enabled = parallel_args_.dp_size() > 1;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "draft extend state count mismatch";

  const int32_t block_size = options_.block_size();
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(base_input);
  torch::TensorOptions token_options = extend_input.token_ids.options();
  torch::TensorOptions position_options = extend_input.positions.options();
  Slice<int32_t> token_ids = {
      base_input.token_ids_host.data_ptr<int32_t>(),
      static_cast<size_t>(base_input.token_ids_host.numel())};

  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(num_sequences * 2);
  buf.out_positions.reserve(base_input.positions_host.dim() == 2
                                ? static_cast<size_t>(num_sequences) * 2 * 3
                                : static_cast<size_t>(num_sequences) * 2);
  buf.out_new_cache_slots.reserve(num_sequences * 2);
  buf.out_kv_seq_lens.reserve(num_sequences * 2);
  buf.out_q_seq_lens.reserve(num_sequences * 2);
  buf.out_q_cu_seq_lens.reserve(num_sequences * 2);
  buf.out_block_tables.reserve(static_cast<size_t>(num_sequences) * 2 *
                               row_ctx.block_table_stride);
  std::vector<torch::Tensor> expanded_embeddings;
  std::vector<int32_t> selected_row_idx;
  expanded_embeddings.reserve(num_sequences * 2);
  selected_row_idx.reserve(num_sequences);
  auto to_worker_device = [this](const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.device() == device_) {
      return tensor;
    }
    return tensor.to(device_);
  };

  torch::Tensor placeholder = embedding_cache_->embedding_placeholder();
  CHECK(placeholder.defined())
      << "embedding placeholder must be initialized for fake draft context";
  placeholder = to_worker_device(placeholder);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    auto add_row = [&](int32_t token_id,
                       int32_t position_offset,
                       const torch::Tensor& embedding) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.token_id = token_id >= 0 ? token_id : 0;
      row.position_offset = position_offset;
      row.append_q_len_one = true;
      row.append_block_table = true;
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
      if (embedding.defined()) {
        expanded_embeddings.emplace_back(to_worker_device(embedding));
      } else {
        expanded_embeddings.emplace_back(placeholder);
      }
    };

    EmbeddingCache::DecodeState state = last_states[seq_id];
    const int32_t current_token_id = token_ids[seq_id];
    if (!state.valid || state.token_id != current_token_id) {
      state = EmbeddingCache::DecodeState();
      state.token_id = current_token_id >= 0 ? current_token_id : 0;
    }
    if (state.all_draft_accepted) {
      int32_t prev_token_id = state.prev_token_id;
      int32_t prev_position_offset = -1;
      torch::Tensor prev_embedding = state.prev_embedding;
      if (prev_token_id < 0) {
        prev_token_id = state.token_id;
        prev_embedding = torch::Tensor();
      }
      add_row(prev_token_id, prev_position_offset, prev_embedding);
    }

    selected_row_idx.emplace_back(
        static_cast<int32_t>(expanded_embeddings.size()));
    add_row(state.token_id, /*position_offset=*/0, state.embedding);

    if (dp_enabled && !state.all_draft_accepted) {
      // Keep DP local token width aligned without rewriting historical KV.
      add_row(state.token_id, /*position_offset=*/1, torch::Tensor());
    }
  }

  CHECK_EQ(buf.out_new_cache_slots.size(),
           static_cast<size_t>(specBuilder::position_column_count(buf)))
      << "draft extend slots/positions mismatch";
  CHECK_EQ(expanded_embeddings.size(),
           static_cast<size_t>(specBuilder::position_column_count(buf)))
      << "draft extend embeddings/positions mismatch";

  set_token_position_tensors(extend_input,
                             buf.out_token_ids,
                             specBuilder::make_positions_tensor(buf),
                             token_options,
                             position_options);
  input_params.meta.num_sequences = specBuilder::position_column_count(buf);
  input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  specBuilder::update_input_params(input_params,
                                   buf,
                                   1,
                                   std::move(buf.out_q_seq_lens),
                                   std::move(buf.out_q_cu_seq_lens),
                                   buf.meta.kv_max_seq_len,
                                   std::move(buf.out_kv_seq_lens),
                                   /*update_block_tables=*/true);
  input_params.attention.rebuild_device_buffer(device_);
  set_draft_input_embedding(input_params.embedding,
                            torch::stack(expanded_embeddings),
                            "decode extend");

  if (!input_params.parallel.dp_global_token_nums.empty()) {
    if (dp_enabled) {
      constexpr int32_t num_extend_tokens = 2;
      scale_dp_global_token_nums_for_speculative_width(
          input_params, num_extend_tokens, "draft_extend");
    } else if (input_params.parallel.dp_global_token_nums.size() == 1) {
      input_params.parallel.dp_global_token_nums[0] =
          specBuilder::position_column_count(buf);
    }
  }

  auto& params = extend_input.sampling_params;
  torch::TensorOptions idx_options =
      params.selected_token_idxes.defined()
          ? params.selected_token_idxes.options()
          : torch::dtype(torch::kInt).device(device_);
  params.selected_token_idxes = torch::tensor(selected_row_idx, idx_options);
  if (!params.sample_idxes.defined()) {
    params.sample_idxes = torch::arange(num_sequences, idx_options);
  }
  extend_input.device_tensors_ready = true;
}

void MTPWorkerImpl::prepare_draft_inputs(const ForwardInput& input,
                                         ForwardInput& draft_input,
                                         int32_t position_offset) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  draft_input = input;
  draft_input.device_tensors_ready = false;

  auto& input_params = draft_input.input_params;
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t block_size = options_.block_size();
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  specBuilder::DecodeBuildBuffers buf;
  buf.out_positions.reserve(input.positions_host.dim() == 2
                                ? static_cast<size_t>(num_sequences) * 3
                                : static_cast<size_t>(num_sequences));
  buf.out_kv_seq_lens.reserve(num_sequences);
  buf.out_new_cache_slots.reserve(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    specBuilder::RowSpec row;
    row.seq_id = seq_id;
    row.position_offset = position_offset;
    row.append_token = false;
    specBuilder::append_decode_row(row_ctx, row, block_size, buf);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(),
           static_cast<size_t>(specBuilder::position_column_count(buf)))
      << "draft kv slots/positions mismatch";

  torch::TensorOptions position_options = input.positions.options();
  set_positions_tensor(
      draft_input, specBuilder::make_positions_tensor(buf), position_options);
  specBuilder::update_input_params(
      input_params,
      buf,
      input_params.meta.q_max_seq_len,
      std::move(input_params.attention.host.q_seq_lens),
      std::move(input_params.attention.host.q_cu_seq_lens),
      buf.meta.kv_max_seq_len,
      std::move(buf.out_kv_seq_lens));
  input_params.attention.rebuild_device_buffer(device_);
  // token_ids is intentionally filled later from the previous draft output.
  draft_input.device_tensors_ready = false;

  prepare_stream_->synchronize();
}

SampleOutput MTPWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const std::vector<ForwardOutput>& draft_outputs,
    const ForwardOutput& target_output) {
  const int32_t num_target_tokens =
      target_output.sample_output.next_tokens.numel();
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  CHECK_EQ(num_target_tokens % num_val_tokens, 0);
  const int32_t batch_size = num_target_tokens / num_val_tokens;
  const int32_t vocab_size = target_output.logits.size(/*dim=*/-1);

  std::vector<torch::Tensor> draft_token_ids_steps;
  std::vector<torch::Tensor> draft_probs_steps;
  draft_token_ids_steps.reserve(draft_outputs.size());
  draft_probs_steps.reserve(draft_outputs.size());
  for (const auto& draft_output : draft_outputs) {
    draft_token_ids_steps.push_back(draft_output.sample_output.next_tokens);
    draft_probs_steps.push_back(draft_output.sample_output.probs);
  }

  auto [draft_token_ids, draft_probs] =
      specBuilder::draftProbs::build_validate_tensors(
          draft_token_ids_steps,
          draft_probs_steps,
          batch_size,
          vocab_size,
          enable_opt_validate_probs_);
  return validate(sampling_params, draft_token_ids, draft_probs, target_output);
}

SampleOutput MTPWorkerImpl::validate(const SamplingParameters& sampling_params,
                                     const torch::Tensor& draft_token_ids,
                                     const torch::Tensor& draft_probs,
                                     const ForwardOutput& target_output) {
  const int32_t num_target_tokens =
      target_output.sample_output.next_tokens.numel();
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  CHECK_EQ(num_target_tokens % num_val_tokens, 0);
  const int32_t batch_size = num_target_tokens / num_val_tokens;
  const int32_t vocab_size = target_output.logits.size(/*dim=*/-1);

  using torch::indexing::None;
  using ISlice = torch::indexing::Slice;
  auto bonus_token_ids =
      target_output.sample_output.next_tokens
          .index({"...", ISlice(num_val_tokens - 1, None, num_val_tokens)})
          .view({-1, 1});

  auto target_logits =
      target_output.logits.view({batch_size, num_val_tokens, vocab_size});

  // prepare input for rejection sampling
  auto rejection_sampler =
      std::make_unique<RejectionSampler>(sampling_params.do_sample,
                                         sampling_params.all_random_sample,
                                         sampling_params.all_greedy_sample,
                                         target_output.logprobs,
                                         target_output.max_top_logprobs,
                                         enable_fused_kernel_);

  // get the accepted tokens
  SampleOutput sample_output =
      rejection_sampler->forward(draft_token_ids.to(bonus_token_ids),
                                 draft_probs.to(target_logits.device()),
                                 target_logits,
                                 bonus_token_ids,
                                 /*mask_out_rejected_tokens=*/true);

  // process embedding
  auto embeddings = target_output.sample_output.embeddings;
  sample_output.embeddings =
      embeddings.view({batch_size, num_val_tokens, embeddings.size(-1)});

  return sample_output;
}

}  // namespace xllm
