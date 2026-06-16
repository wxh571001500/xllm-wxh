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

#include "models/spec_feature_dump.h"

#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#include "util/tensor_helper.h"

namespace xllm::spec_feature_dump {

namespace {

constexpr int32_t kSchemaVersion = 1;
constexpr char kDefaultDumpRoot[] =
    "/export/home/weinan5/wangxiaohan/xllm-dump";
constexpr char kTargetModel[] = "target";
constexpr char kDraftModel[] = "draft";

std::atomic<int64_t> g_event_index{0};
std::mutex g_write_mutex;

struct RequestView {
  std::string request_id;
  int32_t request_index = -1;
  std::vector<int64_t> token_rows;
  int32_t q_seq_len = 0;
  int32_t kv_seq_len = 0;
  std::vector<int32_t> slots;
};

const char* getenv_value(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return nullptr;
  }
  return value;
}

bool env_bool(const char* name, bool default_value) {
  const char* value = getenv_value(name);
  if (value == nullptr) {
    return default_value;
  }
  const std::string text(value);
  return text == "1" || text == "true" || text == "TRUE" || text == "on" ||
         text == "ON" || text == "yes" || text == "YES";
}

std::string env_string(const char* name, const std::string& default_value) {
  const char* value = getenv_value(name);
  if (value == nullptr) {
    return default_value;
  }
  return std::string(value);
}

std::vector<std::string> split_list(const std::string& text) {
  std::vector<std::string> values;
  std::string current;
  for (char ch : text) {
    if (ch == ',' || ch == ';' || ch == ' ' || ch == '\t' || ch == '\n') {
      if (!current.empty()) {
        values.emplace_back(current);
        current.clear();
      }
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty()) {
    values.emplace_back(current);
  }
  return values;
}

std::filesystem::path dump_root_path() {
  return std::filesystem::path(
      env_string("XLLM_SPEC_FEATURE_DUMP_DIR", kDefaultDumpRoot));
}

std::optional<std::filesystem::path> available_dump_root() {
  if (!enabled()) {
    return std::nullopt;
  }
  const std::filesystem::path root = dump_root_path();
  std::error_code error_code;
  const bool exists = std::filesystem::exists(root, error_code);
  if (error_code || !exists) {
    return std::nullopt;
  }
  const bool is_directory = std::filesystem::is_directory(root, error_code);
  if (error_code || !is_directory) {
    return std::nullopt;
  }
  return root;
}

const std::unordered_set<std::string>& request_filter() {
  static const std::unordered_set<std::string> filter = [] {
    std::unordered_set<std::string> values;
    const char* configured = getenv_value("XLLM_SPEC_FEATURE_DUMP_REQUEST_IDS");
    if (configured == nullptr) {
      return values;
    }
    for (const std::string& value : split_list(configured)) {
      values.insert(value);
    }
    return values;
  }();
  return filter;
}

bool should_dump_request(const std::string& request_id) {
  if (request_id.empty()) {
    return false;
  }
  const std::unordered_set<std::string>& filter = request_filter();
  return filter.empty() || filter.find(request_id) != filter.end();
}

uint64_t fnv1a64(const std::string& value) {
  uint64_t hash = 1469598103934665603ull;
  for (unsigned char ch : value) {
    hash ^= static_cast<uint64_t>(ch);
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string hex_u64(uint64_t value) {
  std::ostringstream oss;
  oss << std::hex << std::setw(16) << std::setfill('0') << value;
  return oss.str();
}

std::string sanitize_path_component(const std::string& value) {
  std::string safe;
  safe.reserve(std::min<size_t>(value.size(), 128));
  for (char ch : value) {
    const bool keep = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                      (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' ||
                      ch == '.';
    safe.push_back(keep ? ch : '_');
    if (safe.size() >= 128) {
      break;
    }
  }
  if (safe.empty()) {
    return "empty";
  }
  return safe;
}

std::string host_name() {
  char buffer[256] = {};
  if (gethostname(buffer, sizeof(buffer) - 1) != 0) {
    return "unknown";
  }
  return sanitize_path_component(std::string(buffer));
}

std::filesystem::path request_dir_path(const std::filesystem::path& root,
                                       const std::string& request_id) {
  return root / ("request-" + sanitize_path_component(request_id) + "-" +
                 hex_u64(fnv1a64(request_id)));
}

bool ensure_directory(const std::filesystem::path& path) {
  std::error_code error_code;
  std::filesystem::create_directories(path, error_code);
  if (error_code) {
    LOG(WARNING) << "Failed to create spec feature dump directory: "
                 << path.string() << ", error=" << error_code.message();
    return false;
  }
  return true;
}

bool write_request_id_file(const std::filesystem::path& request_dir,
                           const std::string& request_id) {
  std::ofstream output(request_dir / "request_id.txt");
  if (!output.good()) {
    return false;
  }
  output << request_id << "\n";
  return output.good();
}

bool append_json_line(const std::filesystem::path& path,
                      const nlohmann::json& record) {
  std::lock_guard<std::mutex> guard(g_write_mutex);
  std::ofstream output(path, std::ios::app);
  if (!output.good()) {
    LOG(WARNING) << "Failed to open spec feature dump file: "
                 << path.string();
    return false;
  }
  output << record.dump(-1, ' ', false, nlohmann::json::error_handler_t::ignore)
         << "\n";
  if (!output.good()) {
    LOG(WARNING) << "Failed to write spec feature dump file: "
                 << path.string();
    return false;
  }
  return true;
}

int64_t now_unix_us() {
  const auto now = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(
             now.time_since_epoch())
      .count();
}

nlohmann::json shape_json(const torch::Tensor& tensor) {
  nlohmann::json shape = nlohmann::json::array();
  if (!tensor.defined()) {
    return shape;
  }
  for (int32_t dim = 0; dim < tensor.dim(); ++dim) {
    shape.push_back(tensor.size(dim));
  }
  return shape;
}

std::string scalar_type_string(torch::ScalarType scalar_type) {
  std::ostringstream oss;
  oss << scalar_type;
  return oss.str();
}

std::string device_string(const torch::Device& device) {
  std::ostringstream oss;
  oss << device;
  return oss.str();
}

nlohmann::json finite_number(double value) {
  if (!std::isfinite(value)) {
    return nullptr;
  }
  return value;
}

nlohmann::json int_vector_json(const std::vector<int32_t>& values) {
  nlohmann::json result = nlohmann::json::array();
  for (int32_t value : values) {
    result.push_back(value);
  }
  return result;
}

nlohmann::json int64_vector_json(const std::vector<int64_t>& values) {
  nlohmann::json result = nlohmann::json::array();
  for (int64_t value : values) {
    result.push_back(value);
  }
  return result;
}

std::vector<int32_t> seq_lengths_from_layout(
    const std::vector<int32_t>& values,
    int64_t row_count) {
  if (row_count <= 0) {
    return {};
  }
  if (values.size() == static_cast<size_t>(row_count + 1) &&
      !values.empty() && values.front() == 0) {
    std::vector<int32_t> lengths;
    lengths.reserve(static_cast<size_t>(row_count));
    for (int64_t i = 0; i < row_count; ++i) {
      lengths.emplace_back(values[static_cast<size_t>(i + 1)] -
                           values[static_cast<size_t>(i)]);
    }
    return lengths;
  }
  if (values.size() == static_cast<size_t>(row_count)) {
    return values;
  }
  return {};
}

std::vector<int64_t> equal_partition_rows(int64_t row_count,
                                          int32_t request_index,
                                          int64_t request_count) {
  std::vector<int64_t> rows;
  if (row_count <= 0 || request_count <= 0 || request_index < 0 ||
      request_index >= request_count || row_count % request_count != 0) {
    return rows;
  }
  const int64_t rows_per_request = row_count / request_count;
  const int64_t start =
      static_cast<int64_t>(request_index) * rows_per_request;
  rows.reserve(static_cast<size_t>(rows_per_request));
  for (int64_t i = 0; i < rows_per_request; ++i) {
    rows.emplace_back(start + i);
  }
  return rows;
}

std::vector<int64_t> token_rows_for_request_index(
    const ModelInputParams& input_params,
    int32_t request_index,
    int64_t token_row_count,
    int64_t request_count) {
  if (request_count <= 0 || request_index < 0 ||
      request_index >= request_count || token_row_count <= 0) {
    return {};
  }

  std::vector<int32_t> q_lengths = seq_lengths_from_layout(
      input_params.attention.host.q_seq_lens, request_count);
  if (!q_lengths.empty()) {
    int64_t total_tokens = 0;
    for (int32_t length : q_lengths) {
      total_tokens += static_cast<int64_t>(std::max(length, 0));
    }
    if (total_tokens == token_row_count) {
      int64_t start = 0;
      for (int32_t i = 0; i < request_index; ++i) {
        start += q_lengths[static_cast<size_t>(i)];
      }
      const int32_t length = q_lengths[static_cast<size_t>(request_index)];
      std::vector<int64_t> rows;
      rows.reserve(static_cast<size_t>(std::max(length, 0)));
      for (int32_t i = 0; i < length; ++i) {
        rows.emplace_back(start + i);
      }
      return rows;
    }
  }

  std::vector<int64_t> rows =
      equal_partition_rows(token_row_count, request_index, request_count);
  if (!rows.empty()) {
    return rows;
  }
  if (request_index < token_row_count) {
    return {request_index};
  }
  return {};
}

int32_t seq_len_for_request(const std::vector<int32_t>& values,
                            int32_t request_index,
                            int64_t request_count) {
  std::vector<int32_t> lengths =
      seq_lengths_from_layout(values, request_count);
  if (request_index >= 0 &&
      request_index < static_cast<int32_t>(lengths.size())) {
    return lengths[static_cast<size_t>(request_index)];
  }
  return 0;
}

torch::Tensor to_cpu_tensor(const torch::Tensor& tensor,
                            torch::ScalarType dtype) {
  if (!tensor.defined()) {
    return torch::Tensor();
  }
  try {
    return safe_to(tensor,
                   torch::TensorOptions().dtype(dtype).device(torch::kCPU),
                   /*non_blocking=*/false)
        .contiguous();
  } catch (const std::exception& exception) {
    LOG(WARNING) << "Failed to copy tensor for spec feature dump: "
                 << exception.what();
    return torch::Tensor();
  }
}

std::vector<double> row_l2_values(const torch::Tensor& tensor) {
  std::vector<double> values;
  if (!tensor.defined() || tensor.dim() != 2) {
    return values;
  }
  torch::Tensor cpu_tensor = to_cpu_tensor(tensor, torch::kFloat);
  if (!cpu_tensor.defined()) {
    return values;
  }

  const int64_t rows = cpu_tensor.size(0);
  const int64_t cols = cpu_tensor.size(1);
  const float* data = cpu_tensor.data_ptr<float>();
  values.reserve(static_cast<size_t>(rows));
  for (int64_t row = 0; row < rows; ++row) {
    double square_sum = 0.0;
    const int64_t row_offset = row * cols;
    for (int64_t col = 0; col < cols; ++col) {
      const double value = static_cast<double>(data[row_offset + col]);
      square_sum += value * value;
    }
    values.emplace_back(std::sqrt(square_sum));
  }
  return values;
}

torch::Tensor hidden_rows_2d(const torch::Tensor& hidden_states,
                             const std::vector<int64_t>& rows) {
  if (!hidden_states.defined() || hidden_states.dim() == 0 || rows.empty()) {
    return torch::Tensor();
  }
  if (hidden_states.size(0) <= 0) {
    return torch::Tensor();
  }

  std::vector<int64_t> valid_rows;
  valid_rows.reserve(rows.size());
  for (int64_t row : rows) {
    if (row >= 0 && row < hidden_states.size(0)) {
      valid_rows.emplace_back(row);
    }
  }
  if (valid_rows.empty()) {
    return torch::Tensor();
  }

  try {
    torch::Tensor index = torch::tensor(
        valid_rows,
        torch::TensorOptions().dtype(torch::kLong).device(
            hidden_states.device()));
    torch::Tensor selected =
        hidden_states.index_select(/*dim=*/0, index).contiguous();
    return selected.reshape({selected.size(0), -1});
  } catch (const std::exception& exception) {
    LOG(WARNING) << "Failed to select hidden rows for spec feature dump: "
                 << exception.what();
    return torch::Tensor();
  }
}

nlohmann::json double_vector_json(const std::vector<double>& values) {
  nlohmann::json result = nlohmann::json::array();
  for (double value : values) {
    result.push_back(finite_number(value));
  }
  return result;
}

nlohmann::json optional_double_vector_json(
    const std::vector<std::optional<double>>& values) {
  nlohmann::json result = nlohmann::json::array();
  for (const std::optional<double>& value : values) {
    if (value.has_value()) {
      result.push_back(finite_number(value.value()));
    } else {
      result.push_back(nullptr);
    }
  }
  return result;
}

torch::Tensor flatten_kv_cache_slots(const torch::Tensor& cache,
                                     int32_t block_size) {
  if (!cache.defined() || block_size <= 0 || cache.dim() < 2 ||
      cache.size(0) <= 0) {
    return torch::Tensor();
  }
  if (cache.size(1) == block_size) {
    return cache.reshape({cache.size(0) * block_size, -1});
  }
  if (cache.dim() >= 3 && cache.size(2) == block_size) {
    return cache.transpose(1, 2)
        .contiguous()
        .reshape({cache.size(0) * block_size, -1});
  }
  return torch::Tensor();
}

std::vector<std::optional<double>> slot_l2_values(
    const torch::Tensor& cache,
    const std::vector<int32_t>& slots,
    int32_t block_size) {
  std::vector<std::optional<double>> values(slots.size());
  torch::Tensor flat_cache = flatten_kv_cache_slots(cache, block_size);
  if (!flat_cache.defined()) {
    return values;
  }

  std::vector<int64_t> valid_slots;
  std::vector<size_t> valid_positions;
  valid_slots.reserve(slots.size());
  valid_positions.reserve(slots.size());
  for (size_t i = 0; i < slots.size(); ++i) {
    const int32_t slot = slots[i];
    if (slot >= 0 && static_cast<int64_t>(slot) < flat_cache.size(0)) {
      valid_slots.emplace_back(slot);
      valid_positions.emplace_back(i);
    }
  }
  if (valid_slots.empty()) {
    return values;
  }

  try {
    torch::Tensor index =
        torch::tensor(valid_slots,
                      torch::TensorOptions().dtype(torch::kLong).device(
                          flat_cache.device()));
    torch::Tensor selected =
        flat_cache.index_select(/*dim=*/0, index).contiguous();
    const std::vector<double> selected_values =
        row_l2_values(selected.reshape({selected.size(0), -1}));
    for (size_t i = 0; i < selected_values.size(); ++i) {
      values[valid_positions[i]] = selected_values[i];
    }
  } catch (const std::exception& exception) {
    LOG(WARNING) << "Failed to compute KV slot L2 for spec feature dump: "
                 << exception.what();
  }
  return values;
}

int32_t infer_block_size_from_cache(const torch::Tensor& cache) {
  if (!cache.defined() || cache.dim() < 2) {
    return 0;
  }
  if (cache.dim() >= 4) {
    return static_cast<int32_t>(cache.size(1));
  }
  return static_cast<int32_t>(cache.size(1));
}

torch::Tensor block_tables_cpu(const ModelInputParams& input_params) {
  const torch::Tensor& block_tables = input_params.attention.host.block_tables;
  if (!block_tables.defined()) {
    return torch::Tensor();
  }
  return to_cpu_tensor(block_tables, torch::kLong);
}

int64_t block_table_row_for_request(const torch::Tensor& block_tables,
                                    int32_t request_index,
                                    int64_t request_count,
                                    const std::vector<int64_t>& token_rows,
                                    int64_t token_row_count) {
  if (!block_tables.defined() || block_tables.dim() != 2 ||
      request_index < 0) {
    return -1;
  }
  const int64_t row_count = block_tables.size(0);
  if (row_count == request_count && request_index < row_count) {
    return request_index;
  }
  if (row_count == token_row_count && !token_rows.empty()) {
    const int64_t row = token_rows.front();
    if (row >= 0 && row < row_count) {
      return row;
    }
  }
  if (request_count > 0 && row_count % request_count == 0) {
    const int64_t rows_per_request = row_count / request_count;
    return static_cast<int64_t>(request_index) * rows_per_request;
  }
  if (request_index < row_count) {
    return request_index;
  }
  return -1;
}

std::vector<int32_t> slots_from_block_table(const ModelInputParams& input_params,
                                            const RequestView& view,
                                            int64_t token_row_count,
                                            int32_t block_size) {
  if (block_size <= 0 || view.kv_seq_len <= 0) {
    return {};
  }

  torch::Tensor block_tables = block_tables_cpu(input_params);
  if (!block_tables.defined() || block_tables.dim() != 2) {
    return {};
  }

  const int64_t request_count =
      static_cast<int64_t>(input_params.embedding.request_ids.size());
  const int64_t block_table_row = block_table_row_for_request(block_tables,
                                                              view.request_index,
                                                              request_count,
                                                              view.token_rows,
                                                              token_row_count);
  if (block_table_row < 0 || block_table_row >= block_tables.size(0)) {
    return {};
  }

  const int64_t table_width = block_tables.size(1);
  const int64_t* block_data = block_tables.data_ptr<int64_t>();
  std::vector<int32_t> slots;
  slots.reserve(static_cast<size_t>(view.kv_seq_len));
  for (int32_t token_index = 0; token_index < view.kv_seq_len; ++token_index) {
    const int64_t block_col = token_index / block_size;
    if (block_col < 0 || block_col >= table_width) {
      slots.emplace_back(-1);
      continue;
    }
    const int64_t physical_block =
        block_data[block_table_row * table_width + block_col];
    if (physical_block < 0 ||
        physical_block >
            static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      slots.emplace_back(-1);
      continue;
    }
    const int64_t slot =
        physical_block * block_size + token_index % block_size;
    if (slot > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      slots.emplace_back(-1);
      continue;
    }
    slots.emplace_back(static_cast<int32_t>(slot));
  }
  return slots;
}

std::vector<RequestView> build_request_views(
    const ModelInputParams& input_params,
    int64_t token_row_count) {
  std::vector<RequestView> views;
  const std::vector<std::string>& request_ids =
      input_params.embedding.request_ids;
  const int64_t request_count = static_cast<int64_t>(request_ids.size());
  if (request_count <= 0) {
    return views;
  }

  views.reserve(request_ids.size());
  for (int32_t request_index = 0;
       request_index < static_cast<int32_t>(request_ids.size());
       ++request_index) {
    const std::string& request_id =
        request_ids[static_cast<size_t>(request_index)];
    if (!should_dump_request(request_id)) {
      continue;
    }
    RequestView view;
    view.request_id = request_id;
    view.request_index = request_index;
    view.token_rows = token_rows_for_request_index(
        input_params, request_index, token_row_count, request_count);
    view.q_seq_len = seq_len_for_request(
        input_params.attention.host.q_seq_lens, request_index, request_count);
    view.kv_seq_len = seq_len_for_request(
        input_params.attention.host.kv_seq_lens, request_index, request_count);
    views.emplace_back(std::move(view));
  }
  return views;
}

std::string stage_name(const ModelInputParams& input_params) {
  const BatchForwardType& batch_type = input_params.meta.batch_forward_type;
  if (batch_type.is_decode()) {
    return "decode";
  }
  if (batch_type.no_decode()) {
    return "prefill";
  }
  if (batch_type.is_mixed()) {
    return "mixed";
  }
  return "empty";
}

nlohmann::json base_record(const FeatureMetadata& metadata,
                           const ModelInputParams& input_params,
                           const std::string& request_id,
                           int32_t request_index) {
  const int64_t event_index =
      g_event_index.fetch_add(1, std::memory_order_relaxed);
  nlohmann::json record;
  record["schema_version"] = kSchemaVersion;
  record["event_index"] = event_index;
  record["event_time_unix_us"] = now_unix_us();
  record["pid"] = static_cast<int64_t>(getpid());
  record["host"] = host_name();
  record["rank"] = metadata.rank;
  record["model"] = metadata.model;
  record["stage"] = stage_name(input_params);
  record["point"] = metadata.point;
  record["layer"] = metadata.layer;
  record["request_id"] = request_id;
  record["request_index"] = request_index;
  record["batch_id"] = input_params.meta.batch_id;
  record["batch_forward_type"] =
      input_params.meta.batch_forward_type.to_string();
  record["num_sequences"] = input_params.meta.num_sequences;
  record["q_max_seq_len"] = input_params.meta.q_max_seq_len;
  record["kv_max_seq_len"] = input_params.meta.kv_max_seq_len;
  record["empty_dp_request"] = false;
  return record;
}

nlohmann::json empty_dp_record(const FeatureMetadata& metadata,
                               const ModelInputParams& input_params,
                               const torch::Tensor& tensor) {
  nlohmann::json record =
      base_record(metadata, input_params, "__dp_empty__", -1);
  record["empty_dp_request"] = true;
  record["tensor_defined"] = tensor.defined();
  record["tensor_shape"] = shape_json(tensor);
  if (tensor.defined()) {
    record["tensor_dtype"] = scalar_type_string(tensor.scalar_type());
    record["tensor_device"] = device_string(tensor.device());
  }
  return record;
}

std::filesystem::path feature_file_path(const std::filesystem::path& request_dir,
                                        int64_t rank) {
  std::ostringstream file_name;
  file_name << "features_rank" << rank << "_host" << host_name() << "_pid"
            << static_cast<int64_t>(getpid()) << ".jsonl";
  return request_dir / file_name.str();
}

void write_feature_record(const std::filesystem::path& root,
                          const std::string& request_id,
                          int64_t rank,
                          const nlohmann::json& record) {
  const std::filesystem::path request_dir = request_dir_path(root, request_id);
  if (!ensure_directory(request_dir)) {
    return;
  }
  write_request_id_file(request_dir, request_id);
  append_json_line(feature_file_path(request_dir, rank), record);
}

const std::set<int32_t>& selected_layers_for_model(const std::string& model) {
  static const std::set<int32_t> target_layers = [] {
    const std::string configured =
        env_string("XLLM_SPEC_FEATURE_DUMP_TARGET_LAYERS", "0,30,60");
    std::set<int32_t> layers;
    if (configured == "all" || configured == "ALL") {
      layers.insert(-1);
      return layers;
    }
    for (const std::string& item : split_list(configured)) {
      try {
        layers.insert(std::stoi(item));
      } catch (const std::exception&) {
        LOG(WARNING) << "Invalid XLLM_SPEC_FEATURE_DUMP_TARGET_LAYERS item: "
                     << item;
      }
    }
    return layers;
  }();

  static const std::set<int32_t> draft_layers = [] {
    const std::string configured =
        env_string("XLLM_SPEC_FEATURE_DUMP_DRAFT_LAYERS", "0");
    std::set<int32_t> layers;
    if (configured == "all" || configured == "ALL") {
      layers.insert(-1);
      return layers;
    }
    for (const std::string& item : split_list(configured)) {
      try {
        layers.insert(std::stoi(item));
      } catch (const std::exception&) {
        LOG(WARNING) << "Invalid XLLM_SPEC_FEATURE_DUMP_DRAFT_LAYERS item: "
                     << item;
      }
    }
    return layers;
  }();

  if (model == kDraftModel) {
    return draft_layers;
  }
  return target_layers;
}

void dump_empty_dp_if_needed(const FeatureMetadata& metadata,
                             const ModelInputParams& input_params,
                             const torch::Tensor& tensor) {
  if (!input_params.embedding.request_ids.empty()) {
    return;
  }
  const std::optional<std::filesystem::path> root = available_dump_root();
  if (!root.has_value()) {
    return;
  }
  write_feature_record(root.value(),
                       "__dp_empty__",
                       metadata.rank,
                       empty_dp_record(metadata, input_params, tensor));
}

}  // namespace

bool enabled() {
  return env_bool("XLLM_SPEC_FEATURE_DUMP", false);
}

bool should_dump_layer(const std::string& model, int32_t layer) {
  if (!enabled()) {
    return false;
  }
  const std::set<int32_t>& layers = selected_layers_for_model(model);
  return layers.find(-1) != layers.end() || layers.find(layer) != layers.end();
}

void dump_hidden(const FeatureMetadata& metadata,
                 const torch::Tensor& hidden_states,
                 const ModelInputParams& input_params) {
  const std::optional<std::filesystem::path> root = available_dump_root();
  if (!root.has_value()) {
    return;
  }
  dump_empty_dp_if_needed(metadata, input_params, hidden_states);

  const int64_t token_row_count =
      hidden_states.defined() && hidden_states.dim() > 0 ? hidden_states.size(0)
                                                         : 0;
  const std::vector<RequestView> views =
      build_request_views(input_params, token_row_count);
  for (const RequestView& view : views) {
    nlohmann::json record =
        base_record(metadata, input_params, view.request_id, view.request_index);
    record["q_seq_len"] = view.q_seq_len;
    record["kv_seq_len"] = view.kv_seq_len;
    record["token_rows"] = int64_vector_json(view.token_rows);
    record["tensor_kind"] = "hidden";
    record["hidden_shape"] = shape_json(hidden_states);
    if (hidden_states.defined()) {
      record["hidden_dtype"] = scalar_type_string(hidden_states.scalar_type());
      record["hidden_device"] = device_string(hidden_states.device());
    }

    torch::Tensor rows = hidden_rows_2d(hidden_states, view.token_rows);
    record["token_l2"] = double_vector_json(row_l2_values(rows));
    write_feature_record(
        root.value(), view.request_id, metadata.rank, record);
  }
}

void dump_kv(const FeatureMetadata& metadata,
             const KVCache& kv_cache,
             const ModelInputParams& input_params) {
  const std::optional<std::filesystem::path> root = available_dump_root();
  if (!root.has_value()) {
    return;
  }
  const torch::Tensor k_cache = kv_cache.get_k_cache();
  const torch::Tensor v_cache = kv_cache.get_v_cache();
  dump_empty_dp_if_needed(metadata, input_params, k_cache);

  const int64_t request_count =
      static_cast<int64_t>(input_params.embedding.request_ids.size());
  int64_t token_row_count = 0;
  const std::vector<int32_t> q_lengths = seq_lengths_from_layout(
      input_params.attention.host.q_seq_lens, request_count);
  for (int32_t length : q_lengths) {
    token_row_count += static_cast<int64_t>(std::max(length, 0));
  }

  const std::vector<RequestView> views =
      build_request_views(input_params, token_row_count);
  const int32_t block_size =
      infer_block_size_from_cache(k_cache.defined() ? k_cache : v_cache);
  for (const RequestView& view : views) {
    const std::vector<int32_t> slots = slots_from_block_table(
        input_params, view, token_row_count, block_size);
    nlohmann::json record =
        base_record(metadata, input_params, view.request_id, view.request_index);
    record["block_size"] = block_size;
    record["q_seq_len"] = view.q_seq_len;
    record["kv_seq_len"] = view.kv_seq_len;
    record["token_rows"] = int64_vector_json(view.token_rows);
    record["tensor_kind"] = "kv";
    record["kv_token_start"] = 0;
    record["slots"] = int_vector_json(slots);
    record["k_shape"] = shape_json(k_cache);
    record["v_shape"] = shape_json(v_cache);
    if (k_cache.defined()) {
      record["k_dtype"] = scalar_type_string(k_cache.scalar_type());
      record["k_device"] = device_string(k_cache.device());
    }
    if (v_cache.defined()) {
      record["v_dtype"] = scalar_type_string(v_cache.scalar_type());
      record["v_device"] = device_string(v_cache.device());
    }
    record["k_l2"] =
        optional_double_vector_json(slot_l2_values(k_cache, slots, block_size));
    record["v_l2"] =
        optional_double_vector_json(slot_l2_values(v_cache, slots, block_size));
    write_feature_record(
        root.value(), view.request_id, metadata.rank, record);
  }
}

}  // namespace xllm::spec_feature_dump
