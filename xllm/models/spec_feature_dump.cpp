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
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "util/tensor_helper.h"

namespace xllm::spec_feature_dump {

namespace {

constexpr char kDefaultDumpRoot[] =
    "/export/home/weinan5/wangxiaohan/xllm-dump";
constexpr char kDraftModel[] = "draft";
constexpr int32_t kDefaultMaxKvTokens = 2048;

std::atomic<int64_t> g_event_index{0};

struct RequestView {
  std::string request_id;
  int32_t request_index = -1;
  int32_t q_seq_len = 0;
  int32_t kv_seq_len = 0;
  std::vector<int64_t> token_rows;
  std::vector<int32_t> token_slots;
  std::vector<int32_t> token_ids;
};

struct EventDump {
  int64_t event_index = -1;
  std::filesystem::path dir;
  std::string hidden_file;
  std::string k_file;
  std::string v_file;
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

int32_t env_int32(const char* name, int32_t default_value) {
  const char* value = getenv_value(name);
  if (value == nullptr) {
    return default_value;
  }
  try {
    return std::stoi(value);
  } catch (const std::exception&) {
    LOG(WARNING) << "Invalid integer env " << name << "=" << value;
    return default_value;
  }
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

const std::unordered_set<std::string>& request_filter() {
  static const std::unordered_set<std::string> filter = [] {
    std::unordered_set<std::string> values;
    const char* configured = getenv_value("XLLM_SPEC_FEATURE_DUMP_REQUEST_IDS");
    if (configured == nullptr) {
      configured = getenv_value("XLLM_SPEC_FEATURE_LOG_REQUEST_IDS");
    }
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

std::string sanitize_path_component(const std::string& value) {
  std::string result;
  result.reserve(std::min<size_t>(value.size(), 128));
  for (char ch : value) {
    const bool keep = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                      (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' ||
                      ch == '.';
    result.push_back(keep ? ch : '_');
    if (result.size() >= 128) {
      break;
    }
  }
  if (result.empty()) {
    return "empty";
  }
  return result;
}

std::string json_escape(const std::string& text) {
  std::ostringstream oss;
  for (char ch : text) {
    switch (ch) {
      case '\\':
        oss << "\\\\";
        break;
      case '"':
        oss << "\\\"";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        oss << ch;
        break;
    }
  }
  return oss.str();
}

std::string json_string(const std::string& text) {
  return "\"" + json_escape(text) + "\"";
}

template <typename T>
std::string json_vector(const std::vector<T>& values) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

std::string shape_json(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "[]";
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

std::vector<int32_t> seq_lengths_from_layout(
    const std::vector<int32_t>& values,
    int64_t sequence_count) {
  if (sequence_count <= 0) {
    return {};
  }
  if (values.size() == static_cast<size_t>(sequence_count + 1) &&
      !values.empty() && values.front() == 0) {
    std::vector<int32_t> lengths;
    lengths.reserve(static_cast<size_t>(sequence_count));
    for (int64_t i = 0; i < sequence_count; ++i) {
      lengths.emplace_back(values[static_cast<size_t>(i + 1)] -
                           values[static_cast<size_t>(i)]);
    }
    return lengths;
  }
  if (values.size() == static_cast<size_t>(sequence_count)) {
    return values;
  }
  return {};
}

int32_t seq_len_for_request(const std::vector<int32_t>& values,
                            int32_t request_index,
                            int64_t sequence_count) {
  const std::vector<int32_t> lengths =
      seq_lengths_from_layout(values, sequence_count);
  if (request_index < 0 ||
      request_index >= static_cast<int32_t>(lengths.size())) {
    return 0;
  }
  return lengths[static_cast<size_t>(request_index)];
}

std::vector<int64_t> token_rows_for_request(
    const ModelInputParams& input_params,
    int32_t request_index,
    int64_t token_row_count,
    int64_t request_count) {
  std::vector<int64_t> rows;
  if (request_index < 0 || request_index >= request_count ||
      token_row_count <= 0) {
    return rows;
  }

  const std::vector<int32_t> q_lengths = seq_lengths_from_layout(
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
      rows.reserve(static_cast<size_t>(std::max(length, 0)));
      for (int32_t i = 0; i < length; ++i) {
        rows.emplace_back(start + i);
      }
      return rows;
    }
  }

  if (request_count > 0 && token_row_count % request_count == 0) {
    const int64_t rows_per_request = token_row_count / request_count;
    const int64_t start =
        static_cast<int64_t>(request_index) * rows_per_request;
    rows.reserve(static_cast<size_t>(rows_per_request));
    for (int64_t i = 0; i < rows_per_request; ++i) {
      rows.emplace_back(start + i);
    }
    return rows;
  }

  if (request_index < token_row_count) {
    rows.emplace_back(request_index);
  }
  return rows;
}

std::vector<int32_t> token_ids_for_request(const torch::Tensor& token_ids,
                                           const std::vector<int64_t>& rows) {
  std::vector<int32_t> result;
  if (!token_ids.defined() || rows.empty() || token_ids.dim() < 1) {
    return result;
  }
  try {
    torch::Tensor cpu_tensor =
        token_ids.to(torch::TensorOptions().dtype(torch::kInt64).device(
            torch::kCPU));
    cpu_tensor = cpu_tensor.contiguous().view(-1);
    const int64_t* data = cpu_tensor.data_ptr<int64_t>();
    const int64_t size = cpu_tensor.size(0);
    result.reserve(rows.size());
    for (int64_t row : rows) {
      if (row < 0 || row >= size) {
        result.emplace_back(-1);
        continue;
      }
      result.emplace_back(static_cast<int32_t>(data[row]));
    }
  } catch (const std::exception& exception) {
    LOG(WARNING) << "Failed to copy token ids for spec feature dump: "
                 << exception.what();
  }
  return result;
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

int32_t max_kv_tokens_per_request() {
  return env_int32("XLLM_SPEC_FEATURE_DUMP_MAX_KV_TOKENS",
                   kDefaultMaxKvTokens);
}

bool dump_hidden_tensors() {
  return env_bool("XLLM_SPEC_FEATURE_DUMP_HIDDEN_TENSORS", true);
}

bool dump_kv_tensors() {
  return env_bool("XLLM_SPEC_FEATURE_DUMP_KV_TENSORS", true);
}

std::vector<std::vector<int32_t>> block_tables_to_vectors(
    const torch::Tensor& block_tables) {
  std::vector<std::vector<int32_t>> values;
  if (!block_tables.defined() || block_tables.dim() != 2) {
    return values;
  }
  try {
    torch::Tensor cpu_tensor =
        block_tables.to(torch::TensorOptions().dtype(torch::kInt).device(
            torch::kCPU));
    cpu_tensor = cpu_tensor.contiguous();
    const int64_t rows = cpu_tensor.size(0);
    const int64_t cols = cpu_tensor.size(1);
    const int32_t* data = cpu_tensor.data_ptr<int32_t>();
    values.resize(static_cast<size_t>(rows));
    for (int64_t row = 0; row < rows; ++row) {
      std::vector<int32_t>& current = values[static_cast<size_t>(row)];
      current.reserve(static_cast<size_t>(cols));
      for (int64_t col = 0; col < cols; ++col) {
        current.emplace_back(data[row * cols + col]);
      }
    }
  } catch (const std::exception& exception) {
    LOG(WARNING) << "Failed to copy block table for spec feature dump: "
                 << exception.what();
  }
  return values;
}

std::string block_tables_json(
    const std::vector<std::vector<int32_t>>& block_tables) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < block_tables.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << json_vector(block_tables[i]);
  }
  oss << "]";
  return oss.str();
}

int64_t block_table_row_for_request(
    const std::vector<std::vector<int32_t>>& block_tables,
    int32_t request_index,
    int64_t request_count,
    const std::vector<int64_t>& token_rows,
    int64_t token_row_count) {
  if (request_index < 0 || block_tables.empty()) {
    return -1;
  }
  const int64_t table_rows = static_cast<int64_t>(block_tables.size());
  if (table_rows == request_count && request_index < table_rows) {
    return request_index;
  }
  if (table_rows == token_row_count && !token_rows.empty()) {
    const int64_t row = token_rows.front();
    if (row >= 0 && row < table_rows) {
      return row;
    }
  }
  if (request_count > 0 && table_rows % request_count == 0) {
    return static_cast<int64_t>(request_index) * (table_rows / request_count);
  }
  if (request_index < table_rows) {
    return request_index;
  }
  return -1;
}

std::vector<int32_t> token_slots_for_request(
    const std::vector<std::vector<int32_t>>& block_tables,
    const RequestView& view,
    int64_t request_count,
    int64_t token_row_count,
    int32_t block_size) {
  std::vector<int32_t> slots;
  if (block_size <= 0 || view.kv_seq_len <= 0 || block_tables.empty()) {
    return slots;
  }

  const int64_t block_table_row =
      block_table_row_for_request(block_tables,
                                  view.request_index,
                                  request_count,
                                  view.token_rows,
                                  token_row_count);
  if (block_table_row < 0 ||
      block_table_row >= static_cast<int64_t>(block_tables.size())) {
    return slots;
  }

  const std::vector<int32_t>& table =
      block_tables[static_cast<size_t>(block_table_row)];
  const int32_t max_tokens = max_kv_tokens_per_request();
  const int32_t token_count =
      max_tokens >= 0 ? std::min(view.kv_seq_len, max_tokens)
                      : view.kv_seq_len;
  slots.reserve(static_cast<size_t>(std::max(token_count, 0)));
  for (int32_t token_index = 0; token_index < token_count; ++token_index) {
    const int64_t block_col = token_index / block_size;
    if (block_col < 0 || block_col >= static_cast<int64_t>(table.size())) {
      slots.emplace_back(-1);
      continue;
    }
    const int64_t physical_block = table[static_cast<size_t>(block_col)];
    const int64_t slot =
        physical_block * block_size + token_index % block_size;
    if (slot < 0 ||
        slot > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      slots.emplace_back(-1);
      continue;
    }
    slots.emplace_back(static_cast<int32_t>(slot));
  }
  return slots;
}

std::vector<RequestView> build_request_views(
    const ModelInputParams& input_params,
    int64_t token_row_count,
    int32_t block_size,
    const std::vector<std::vector<int32_t>>& block_tables) {
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
    view.q_seq_len = seq_len_for_request(
        input_params.attention.host.q_seq_lens, request_index, request_count);
    view.kv_seq_len = seq_len_for_request(
        input_params.attention.host.kv_seq_lens, request_index, request_count);
    view.token_rows = token_rows_for_request(
        input_params, request_index, token_row_count, request_count);
    view.token_slots = token_slots_for_request(
        block_tables, view, request_count, token_row_count, block_size);
    views.emplace_back(std::move(view));
  }
  return views;
}

int32_t infer_block_size_from_cache(const torch::Tensor& cache) {
  if (!cache.defined() || cache.dim() < 2) {
    return 0;
  }
  return static_cast<int32_t>(cache.size(1));
}

std::vector<int32_t> selected_slots_from_views(
    const std::vector<RequestView>& views) {
  std::vector<int32_t> slots;
  for (const RequestView& view : views) {
    for (int32_t slot : view.token_slots) {
      if (slot >= 0) {
        slots.emplace_back(slot);
      }
    }
  }
  std::sort(slots.begin(), slots.end());
  slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
  return slots;
}

std::vector<int32_t> selected_blocks_from_slots(
    const std::vector<int32_t>& slots,
    int32_t block_size) {
  std::vector<int32_t> blocks;
  if (block_size <= 0) {
    return blocks;
  }
  blocks.reserve(slots.size());
  for (int32_t slot : slots) {
    if (slot >= 0) {
      blocks.emplace_back(slot / block_size);
    }
  }
  std::sort(blocks.begin(), blocks.end());
  blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
  return blocks;
}

bool save_tensor_cpu(const torch::Tensor& tensor,
                     const std::filesystem::path& path,
                     const std::string& label) {
  if (!tensor.defined()) {
    return false;
  }
  try {
    torch::Tensor cpu_tensor = tensor.to(torch::kCPU).contiguous();
    save_tensor_as_pickle(cpu_tensor, path.string());
    return true;
  } catch (const std::exception& exception) {
    LOG(WARNING) << "Failed to save " << label
                 << " for spec feature dump to " << path.string()
                 << ": " << exception.what();
    return false;
  }
}

bool save_selected_kv_blocks(const torch::Tensor& cache,
                             const std::vector<int32_t>& selected_blocks,
                             const std::filesystem::path& path,
                             const std::string& label) {
  if (!cache.defined() || selected_blocks.empty() || cache.dim() < 1) {
    return false;
  }
  try {
    std::vector<torch::Tensor> cpu_blocks;
    cpu_blocks.reserve(selected_blocks.size());
    for (int32_t block : selected_blocks) {
      if (block < 0 || static_cast<int64_t>(block) >= cache.size(0)) {
        continue;
      }
      torch::Tensor block_tensor =
          cache.narrow(/*dim=*/0, static_cast<int64_t>(block), /*length=*/1);
      cpu_blocks.emplace_back(block_tensor.to(torch::kCPU).contiguous());
    }
    if (cpu_blocks.empty()) {
      return false;
    }
    torch::Tensor selected = torch::cat(cpu_blocks, /*dim=*/0);
    save_tensor_as_pickle(selected, path.string());
    return true;
  } catch (const std::exception& exception) {
    LOG(WARNING) << "Failed to save selected " << label
                 << " blocks for spec feature dump: " << exception.what();
    return false;
  }
}

EventDump create_event_dir(const std::filesystem::path& root,
                           const FeatureMetadata& metadata,
                           const ModelInputParams& input_params) {
  EventDump event;
  event.event_index = g_event_index.fetch_add(1, std::memory_order_relaxed);
  std::ostringstream name;
  name << "event-" << std::setw(10) << std::setfill('0') << event.event_index
       << "_rank" << metadata.rank << "_pid" << static_cast<int64_t>(getpid())
       << "_" << sanitize_path_component(metadata.model)
       << "_" << sanitize_path_component(stage_name(input_params))
       << "_" << sanitize_path_component(metadata.point)
       << "_layer" << metadata.layer;
  event.dir = root / name.str();
  std::error_code error_code;
  std::filesystem::create_directories(event.dir, error_code);
  if (error_code) {
    LOG(WARNING) << "Failed to create spec feature dump event dir: "
                 << event.dir.string() << ", error=" << error_code.message();
    event.dir.clear();
  }
  return event;
}

void write_common_json(std::ostream& output,
                       const EventDump& event,
                       const FeatureMetadata& metadata,
                       const ModelInputParams& input_params,
                        const std::vector<RequestView>& views,
                       const std::vector<std::vector<int32_t>>& block_tables,
                       int32_t block_size,
                       const std::vector<int32_t>& selected_slots,
                       const std::vector<int32_t>& selected_blocks,
                       const std::string& tensor_kind) {
  output << "{\n";
  output << "  \"schema_version\": 2,\n";
  output << "  \"event_index\": " << event.event_index << ",\n";
  output << "  \"pid\": " << static_cast<int64_t>(getpid()) << ",\n";
  output << "  \"rank\": " << metadata.rank << ",\n";
  output << "  \"model\": " << json_string(metadata.model) << ",\n";
  output << "  \"stage\": " << json_string(stage_name(input_params)) << ",\n";
  output << "  \"point\": " << json_string(metadata.point) << ",\n";
  output << "  \"layer\": " << metadata.layer << ",\n";
  output << "  \"batch_id\": " << input_params.meta.batch_id << ",\n";
  output << "  \"batch_forward_type\": "
         << json_string(input_params.meta.batch_forward_type.to_string())
         << ",\n";
  output << "  \"num_sequences\": " << input_params.meta.num_sequences
         << ",\n";
  output << "  \"q_seq_lens\": "
         << json_vector(input_params.attention.host.q_seq_lens) << ",\n";
  output << "  \"kv_seq_lens\": "
         << json_vector(input_params.attention.host.kv_seq_lens) << ",\n";
  output << "  \"tensor_kind\": " << json_string(tensor_kind) << ",\n";
  output << "  \"block_size\": " << block_size << ",\n";
  output << "  \"block_tables\": " << block_tables_json(block_tables)
         << ",\n";
  output << "  \"selected_slots\": " << json_vector(selected_slots) << ",\n";
  output << "  \"selected_blocks\": " << json_vector(selected_blocks)
         << ",\n";
  output << "  \"hidden_file\": " << json_string(event.hidden_file) << ",\n";
  output << "  \"k_file\": " << json_string(event.k_file) << ",\n";
  output << "  \"v_file\": " << json_string(event.v_file) << ",\n";
  output << "  \"requests\": [\n";
  for (size_t i = 0; i < views.size(); ++i) {
    const RequestView& view = views[i];
    output << "    {\n";
    output << "      \"request_id\": " << json_string(view.request_id)
           << ",\n";
    output << "      \"request_index\": " << view.request_index << ",\n";
    output << "      \"q_seq_len\": " << view.q_seq_len << ",\n";
    output << "      \"kv_seq_len\": " << view.kv_seq_len << ",\n";
    output << "      \"token_rows\": " << json_vector(view.token_rows)
           << ",\n";
    output << "      \"token_slots\": " << json_vector(view.token_slots)
           << ",\n";
    output << "      \"token_ids\": " << json_vector(view.token_ids)
           << "\n";
    output << "    }";
    if (i + 1 < views.size()) {
      output << ",";
    }
    output << "\n";
  }
  output << "  ]\n";
  output << "}\n";
}

void write_meta_json(const EventDump& event,
                     const FeatureMetadata& metadata,
                     const ModelInputParams& input_params,
                     const std::vector<RequestView>& views,
                     const std::vector<std::vector<int32_t>>& block_tables,
                     int32_t block_size,
                     const std::vector<int32_t>& selected_slots,
                     const std::vector<int32_t>& selected_blocks,
                     const std::string& tensor_kind) {
  if (event.dir.empty()) {
    return;
  }
  const std::filesystem::path path = event.dir / "meta.json";
  std::ofstream output(path);
  if (!output.good()) {
    LOG(WARNING) << "Failed to open spec feature dump meta file: "
                 << path.string();
    return;
  }
  write_common_json(output,
                    event,
                    metadata,
                    input_params,
                    views,
                    block_tables,
                    block_size,
                    selected_slots,
                    selected_blocks,
                    tensor_kind);
}

void dump_empty_dp_event(const FeatureMetadata& metadata,
                         const ModelInputParams& input_params,
                         const torch::Tensor& tensor) {
  if (!input_params.embedding.request_ids.empty()) {
    return;
  }
  const std::optional<std::filesystem::path> root = available_dump_root();
  if (!root.has_value()) {
    return;
  }
  EventDump event = create_event_dir(root.value(), metadata, input_params);
  if (event.dir.empty()) {
    return;
  }
  std::ofstream output(event.dir / "meta.json");
  output << "{\n";
  output << "  \"schema_version\": 2,\n";
  output << "  \"event_index\": " << event.event_index << ",\n";
  output << "  \"pid\": " << static_cast<int64_t>(getpid()) << ",\n";
  output << "  \"rank\": " << metadata.rank << ",\n";
  output << "  \"model\": " << json_string(metadata.model) << ",\n";
  output << "  \"stage\": " << json_string(stage_name(input_params)) << ",\n";
  output << "  \"point\": " << json_string(metadata.point) << ",\n";
  output << "  \"layer\": " << metadata.layer << ",\n";
  output << "  \"empty_dp_request\": true,\n";
  output << "  \"tensor_shape\": " << shape_json(tensor) << "\n";
  output << "}\n";
  LOG(INFO) << "SpecFeatureDump event_index=" << event.event_index
            << " request_id=__dp_empty__"
            << " rank=" << metadata.rank
            << " model=" << metadata.model
            << " stage=" << stage_name(input_params)
            << " point=" << metadata.point
            << " layer=" << metadata.layer
            << " dir=" << event.dir.string();
}

const std::set<int32_t>& selected_layers_for_model(const std::string& model) {
  static const std::set<int32_t> target_layers = [] {
    const std::string configured =
        env_string("XLLM_SPEC_FEATURE_DUMP_TARGET_LAYERS",
                   env_string("XLLM_SPEC_FEATURE_LOG_TARGET_LAYERS",
                              "0,30,60"));
    std::set<int32_t> layers;
    if (configured == "all" || configured == "ALL") {
      layers.insert(-1);
      return layers;
    }
    for (const std::string& item : split_list(configured)) {
      try {
        layers.insert(std::stoi(item));
      } catch (const std::exception&) {
        LOG(WARNING) << "Invalid target layer item for spec feature dump: "
                     << item;
      }
    }
    return layers;
  }();

  static const std::set<int32_t> draft_layers = [] {
    const std::string configured =
        env_string("XLLM_SPEC_FEATURE_DUMP_DRAFT_LAYERS",
                   env_string("XLLM_SPEC_FEATURE_LOG_DRAFT_LAYERS", "0"));
    std::set<int32_t> layers;
    if (configured == "all" || configured == "ALL") {
      layers.insert(-1);
      return layers;
    }
    for (const std::string& item : split_list(configured)) {
      try {
        layers.insert(std::stoi(item));
      } catch (const std::exception&) {
        LOG(WARNING) << "Invalid draft layer item for spec feature dump: "
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

void log_event_summary(const EventDump& event,
                       const FeatureMetadata& metadata,
                       const ModelInputParams& input_params,
                       const std::vector<RequestView>& views) {
  std::ostringstream request_ids;
  request_ids << "[";
  for (size_t i = 0; i < views.size(); ++i) {
    if (i > 0) {
      request_ids << ",";
    }
    request_ids << views[i].request_id;
  }
  request_ids << "]";
  LOG(INFO) << "SpecFeatureDump event_index=" << event.event_index
            << " rank=" << metadata.rank
            << " model=" << metadata.model
            << " stage=" << stage_name(input_params)
            << " point=" << metadata.point
            << " layer=" << metadata.layer
            << " request_ids=" << request_ids.str()
            << " dir=" << event.dir.string();
}

}  // namespace

bool enabled() {
  return env_bool("XLLM_SPEC_FEATURE_DUMP",
                  env_bool("XLLM_SPEC_FEATURE_LOG", false));
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
                 const ModelInputParams& input_params,
                 const torch::Tensor& token_ids) {
  if (!enabled()) {
    return;
  }
  if (input_params.embedding.request_ids.empty()) {
    dump_empty_dp_event(metadata, input_params, hidden_states);
    return;
  }
  const std::optional<std::filesystem::path> root = available_dump_root();
  if (!root.has_value()) {
    return;
  }

  const int64_t token_row_count =
      hidden_states.defined() && hidden_states.dim() > 0 ? hidden_states.size(0)
                                                         : 0;
  const std::vector<std::vector<int32_t>> block_tables =
      block_tables_to_vectors(input_params.attention.host.block_tables);
  const std::vector<RequestView> views =
      build_request_views(input_params,
                          token_row_count,
                          /*block_size=*/0,
                          block_tables);
  if (views.empty()) {
    return;
  }

  std::vector<RequestView> views_with_token_ids = views;
  if (metadata.point == "model_input_hidden") {
    for (RequestView& view : views_with_token_ids) {
      view.token_ids = token_ids_for_request(token_ids, view.token_rows);
    }
  }

  EventDump event = create_event_dir(root.value(), metadata, input_params);
  if (event.dir.empty()) {
    return;
  }
  if (dump_hidden_tensors() &&
      save_tensor_cpu(hidden_states, event.dir / "hidden.pt", "hidden")) {
    event.hidden_file = "hidden.pt";
  }

  write_meta_json(event,
                  metadata,
                  input_params,
                  views_with_token_ids,
                  block_tables,
                  /*block_size=*/0,
                  /*selected_slots=*/{},
                  /*selected_blocks=*/{},
                  "hidden");
  log_event_summary(event, metadata, input_params, views_with_token_ids);
}

void dump_kv(const FeatureMetadata& metadata,
             const KVCache& kv_cache,
             const ModelInputParams& input_params) {
  if (!enabled()) {
    return;
  }
  const torch::Tensor k_cache = kv_cache.get_k_cache();
  const torch::Tensor v_cache = kv_cache.get_v_cache();
  if (input_params.embedding.request_ids.empty()) {
    dump_empty_dp_event(metadata, input_params, k_cache);
    return;
  }
  const std::optional<std::filesystem::path> root = available_dump_root();
  if (!root.has_value()) {
    return;
  }

  const int64_t request_count =
      static_cast<int64_t>(input_params.embedding.request_ids.size());
  int64_t token_row_count = 0;
  const std::vector<int32_t> q_lengths = seq_lengths_from_layout(
      input_params.attention.host.q_seq_lens, request_count);
  for (int32_t length : q_lengths) {
    token_row_count += static_cast<int64_t>(std::max(length, 0));
  }

  const int32_t block_size =
      infer_block_size_from_cache(k_cache.defined() ? k_cache : v_cache);
  const std::vector<std::vector<int32_t>> block_tables =
      block_tables_to_vectors(input_params.attention.host.block_tables);
  const std::vector<RequestView> views = build_request_views(
      input_params, token_row_count, block_size, block_tables);
  if (views.empty()) {
    return;
  }
  const std::vector<int32_t> selected_slots = selected_slots_from_views(views);
  const std::vector<int32_t> selected_blocks =
      selected_blocks_from_slots(selected_slots, block_size);

  EventDump event = create_event_dir(root.value(), metadata, input_params);
  if (event.dir.empty()) {
    return;
  }
  if (dump_kv_tensors()) {
    if (save_selected_kv_blocks(
            k_cache, selected_blocks, event.dir / "k_blocks.pt", "k")) {
      event.k_file = "k_blocks.pt";
    }
    if (save_selected_kv_blocks(
            v_cache, selected_blocks, event.dir / "v_blocks.pt", "v")) {
      event.v_file = "v_blocks.pt";
    }
  }

  write_meta_json(event,
                  metadata,
                  input_params,
                  views,
                  block_tables,
                  block_size,
                  selected_slots,
                  selected_blocks,
                  "kv");
  log_event_summary(event, metadata, input_params, views);
}

}  // namespace xllm::spec_feature_dump
