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

#include "core/runtime/mlu_graph_executor_impl.h"

#include <cnrt.h>
#include <framework/core/MLUEvent.h>
#include <framework/core/caching_allocator.h>
#include <framework/core/stream_guard.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <sstream>
#include <string>

#include "common/metrics.h"
#include "core/common/constants.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/triton_jit/include/jit_kernel.h"

namespace {
using xllm::mlu::get_block_capacity;
using xllm::mlu::GraphLayout;
struct GraphPoolMemoryUsage {
  std::size_t reserved_bytes = 0;
  std::size_t allocated_bytes = 0;
  std::size_t active_bytes = 0;
  std::size_t segment_count = 0;
};

std::size_t tensor_bytes(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return 0;
  }

  return static_cast<std::size_t>(tensor.numel()) * tensor.element_size();
}

std::string format_memory_size(std::size_t bytes) {
  if (bytes < 1024) {
    return std::to_string(bytes) + " B";
  }

  double value = static_cast<double>(bytes) / 1024.0;
  std::string unit = " KiB";
  if (value >= 1024.0) {
    value /= 1024.0;
    unit = " MiB";
  }
  if (value >= 1024.0) {
    value /= 1024.0;
    unit = " GiB";
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << value << unit;
  return oss.str();
}

GraphPoolMemoryUsage get_graph_pool_usage(
    c10::DeviceIndex device_index,
    const torch_mlu::MempoolId_t& pool_id) {
  GraphPoolMemoryUsage usage;
  const auto snapshot = torch_mlu::MLUCachingAllocator::snapshot();
  for (const auto& segment : snapshot.segments) {
    if (segment.device != device_index ||
        segment.owner_private_pool_id.first != pool_id.first ||
        segment.owner_private_pool_id.second != pool_id.second) {
      continue;
    }
    usage.reserved_bytes += segment.total_size;
    usage.allocated_bytes += segment.allocated_size;
    usage.active_bytes += segment.active_size;
    usage.segment_count += 1;
  }
  return usage;
}

xllm::ModelOutput make_graph_output(const torch::Tensor& hidden_states,
                                    const torch::Tensor& aux_hidden_states,
                                    bool enable_aux_hidden_states) {
  if (enable_aux_hidden_states && aux_hidden_states.defined() &&
      aux_hidden_states.numel() > 0) {
    return xllm::ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
  }
  return xllm::ModelOutput(hidden_states);
}

int64_t ordinary_graph_capacity(const xllm::ModelArgs& args,
                                int32_t block_size) {
  if (args.max_position_embeddings() <= 0) {
    return 0;
  }
  const int64_t block_capacity =
      get_block_capacity(args.max_position_embeddings(), block_size) + 1;
  return block_capacity * block_size;
}

enum class GraphAction : int8_t { EAGER, CAPTURE, REPLAY };

// A fixed-size packet keeps admission, layout, and cache state in one
// collective. The complete compared fields are values, never a hash.
constexpr int64_t kGraphPacketSize = 32;
constexpr int64_t kGraphPacketFields = 11;

GraphAction agree_graph_action(const xllm::mlu::GraphDecision& decision,
                               bool cache_hit,
                               xllm::ProcessGroup* dp_group,
                               const torch::Device& device) {
  const auto* plan = std::get_if<xllm::mlu::GraphPlan>(&decision);
  std::array<int64_t, kGraphPacketSize> packet{};
  if (plan != nullptr && plan->key.multi_block_table_column_counts.size() <=
                             kGraphPacketSize - kGraphPacketFields) {
    packet[0] = 1;
    packet[1] = cache_hit ? 1 : 0;
    packet[2] = plan->layout.padded_num_tokens;
    packet[3] = plan->layout.padded_num_reqs;
    packet[4] = plan->layout.tokens_per_request;
    packet[5] = plan->layout.attention_row_query_len;
    packet[6] = plan->layout.cache_pad_slot;
    packet[7] = plan->history_capacity;
    packet[8] = plan->main_block_table_columns;
    packet[9] = plan->key.has_input_embedding;
    packet[10] = plan->key.multi_block_table_column_counts.size();
    std::copy(plan->key.multi_block_table_column_counts.begin(),
              plan->key.multi_block_table_column_counts.end(),
              packet.begin() + kGraphPacketFields);
  }
  if (dp_group == nullptr || dp_group->world_size() <= 1) {
    return packet[0] == 0
               ? GraphAction::EAGER
               : (cache_hit ? GraphAction::REPLAY : GraphAction::CAPTURE);
  }
  torch::Tensor local =
      torch::tensor(std::vector<int64_t>(packet.begin(), packet.end()),
                    torch::TensorOptions().dtype(torch::kInt64))
          .to(device);
  torch::Tensor gathered =
      dp_group->allgather_base_sync(local).to(torch::kCPU).contiguous();
  CHECK_EQ(gathered.numel(), dp_group->world_size() * kGraphPacketSize);
  const int64_t* values = gathered.data_ptr<int64_t>();
  bool all_hit = true;
  for (int32_t rank = 0; rank < dp_group->world_size(); ++rank) {
    const int64_t* peer = values + rank * kGraphPacketSize;
    if (peer[0] == 0 ||
        !std::equal(packet.begin() + 2, packet.end(), peer + 2)) {
      return GraphAction::EAGER;
    }
    all_hit = all_hit && peer[1] != 0;
  }
  return all_hit ? GraphAction::REPLAY : GraphAction::CAPTURE;
}

// Batch rows can vary within a bucket; all other dimensions and dtypes are
// model contracts. Validate against the owned buffers without caching schemas.
void check_graph_tensor(const torch::Tensor& buffer,
                        const torch::Tensor& input,
                        const char* name,
                        int32_t batch_dim = 0) {
  if (!input.defined()) {
    return;
  }
  CHECK(buffer.defined()) << name;
  CHECK(buffer.scalar_type() == input.scalar_type())
      << name << " graph input dtype changed";
  CHECK(buffer.device() == input.device())
      << name << " graph input device changed";
  CHECK_EQ(buffer.dim(), input.dim()) << name << " graph input rank changed";
  CHECK_LE(input.size(batch_dim), buffer.size(batch_dim)) << name;
  for (int32_t dim = 0; dim < input.dim(); ++dim) {
    if (dim == batch_dim) {
      continue;
    }
    CHECK_EQ(buffer.size(dim), input.size(dim))
        << name << " graph input feature dimension changed";
  }
}

}  // namespace

namespace xllm::mlu {

GraphPersistentParam::GraphPersistentParam(const torch::Tensor& tokens,
                                           const torch::Tensor& positions,
                                           ModelInputParams params,
                                           const GraphLayout& layout,
                                           int64_t graph_max_kv_seq_len,
                                           int64_t main_block_table_columns)
    : params_(std::move(params)) {
  const torch::Tensor& input_embedding = params_.embedding.input_embedding;
  if (input_embedding.defined()) {
    CHECK_EQ(input_embedding.dim(), 2)
        << "input_embedding graph input rank changed";
    CHECK_EQ(input_embedding.size(0), tokens.size(0))
        << "input_embedding graph input row count changed";
    CHECK(input_embedding.device() == tokens.device())
        << "input_embedding graph input device changed";
  }
  const int64_t graph_tokens = layout.padded_num_tokens;
  const int64_t extra_rows = (layout.padded_num_reqs - layout.num_reqs) *
                             layout.tokens_per_request /
                             layout.attention_row_query_len;
  const auto allocate_rows = [](const torch::Tensor& input, int64_t rows) {
    if (!input.defined()) {
      return torch::Tensor();
    }
    auto shape = input.sizes().vec();
    shape[0] = rows;
    return torch::empty(shape, input.options());
  };
  use_mrope_ = positions.dim() == 2;
  tokens_ = allocate_rows(tokens, graph_tokens);
  auto position_shape = positions.sizes().vec();
  position_shape[use_mrope_ ? 1 : 0] = graph_tokens;
  positions_ = torch::empty(position_shape, positions.options());
  params_.enable_graph = true;
  params_.meta.num_sequences = graph_tokens / layout.attention_row_query_len;
  if (params_.parallel.dp_global_token_nums.size() > 1) {
    std::fill(params_.parallel.dp_global_token_nums.begin(),
              params_.parallel.dp_global_token_nums.end(),
              graph_tokens);
  }
  params_.graph.num_valid_token_rows = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));
  params_.attn_metadata.reset();
  if (graph_max_kv_seq_len > 0) {
    params_.meta.kv_max_seq_len = graph_max_kv_seq_len;
  }
  auto& attention = params_.attention.device;
  attention.q_seq_lens = allocate_rows(
      attention.q_seq_lens, attention.q_seq_lens.numel() + extra_rows);
  attention.q_cu_seq_lens = attention.q_seq_lens;
  attention.kv_seq_lens = allocate_rows(
      attention.kv_seq_lens, attention.kv_seq_lens.numel() + extra_rows);
  attention.new_cache_slots =
      allocate_rows(attention.new_cache_slots, graph_tokens);
  if (attention.block_tables.defined()) {
    attention.block_tables = torch::empty(
        {attention.block_tables.size(0) + extra_rows, main_block_table_columns},
        attention.block_tables.options());
  }
  auto& embedding = params_.embedding;
  embedding.input_embedding =
      allocate_rows(embedding.input_embedding, graph_tokens);
  embedding.linear_state_indices =
      allocate_rows(embedding.linear_state_indices, layout.padded_num_reqs);
  if (!embedding.linear_state_indices.defined() &&
      !embedding.linear_state_ids.empty()) {
    embedding.linear_state_indices =
        torch::empty({layout.padded_num_reqs}, attention.q_seq_lens.options());
  }
  params_.num_accepted_tokens =
      allocate_rows(params_.num_accepted_tokens, layout.padded_num_reqs);
  params_.linear_state_validity_mask_tensor = allocate_rows(
      params_.linear_state_validity_mask_tensor, layout.padded_num_reqs);
  if (!params_.linear_state_validity_mask_tensor.defined() &&
      !params_.linear_state_validity_mask.empty()) {
    params_.linear_state_validity_mask_tensor = torch::empty(
        {layout.padded_num_reqs},
        torch::TensorOptions().dtype(torch::kBool).device(tokens.device()));
  }
  params_.expert.eplb_decode_token_mask = torch::Tensor();
  for (auto& table : params_.multi_block_tables) {
    table = allocate_rows(table, table.size(0) + extra_rows);
  }
}

void GraphPersistentParam::update_input_buffer(const torch::Tensor& tokens,
                                               const torch::Tensor& positions,
                                               const ModelInputParams& params,
                                               const GraphLayout& layout) {
  check_graph_tensor(tokens_, tokens, "tokens");
  check_graph_tensor(positions_, positions, "positions", use_mrope_ ? 1 : 0);
  const auto& source = params.attention.device;
  const auto& buffer = params_.attention.device;
  check_graph_tensor(buffer.q_seq_lens, source.q_seq_lens, "q_seq_lens");
  check_graph_tensor(buffer.kv_seq_lens, source.kv_seq_lens, "kv_seq_lens");
  check_graph_tensor(
      buffer.new_cache_slots, source.new_cache_slots, "new_cache_slots");
  // Main block-table columns are padded/truncated to the selected KV capacity.
  if (source.block_tables.defined()) {
    CHECK_EQ(source.block_tables.dim(), 2);
    CHECK(buffer.block_tables.scalar_type() ==
          source.block_tables.scalar_type())
        << "block_tables graph input dtype changed";
    CHECK_LE(source.block_tables.size(0), buffer.block_tables.size(0));
  }
  check_graph_tensor(params_.embedding.linear_state_indices,
                     params.embedding.linear_state_indices,
                     "state_indices");
  const torch::Tensor& input_embedding = params.embedding.input_embedding;
  CHECK_EQ(params_.embedding.input_embedding.defined(),
           input_embedding.defined())
      << "input_embedding graph input presence changed";
  if (input_embedding.defined()) {
    CHECK_EQ(input_embedding.dim(), 2)
        << "input_embedding graph input rank changed";
    check_graph_tensor(
        params_.embedding.input_embedding, input_embedding, "input_embedding");
    CHECK_EQ(input_embedding.size(0), tokens.size(0))
        << "input_embedding graph input row count changed";
  }
  if (params_.num_accepted_tokens.defined()) {
    check_graph_tensor(params_.num_accepted_tokens,
                       params.num_accepted_tokens,
                       "accepted_tokens");
  }
  if (params_.linear_state_validity_mask_tensor.defined()) {
    check_graph_tensor(params_.linear_state_validity_mask_tensor,
                       params.linear_state_validity_mask_tensor,
                       "state_mask");
  }
  CHECK_EQ(params_.multi_block_tables.size(), params.multi_block_tables.size());
  for (std::size_t i = 0; i < params.multi_block_tables.size(); ++i) {
    check_graph_tensor(params_.multi_block_tables[i],
                       params.multi_block_tables[i],
                       "multi_block_tables");
  }
  const int64_t extra_requests = layout.padded_num_reqs - layout.num_reqs;
  const int64_t extra_rows = extra_requests * layout.tokens_per_request /
                             layout.attention_row_query_len;
  const auto copy_host = [](torch::Tensor dst,
                            const std::vector<int32_t>& values) {
    torch::Tensor staging = torch::tensor(
        values,
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    dst.copy_(staging);
  };
  const auto extend_offsets = [extra_rows,
                               &layout](std::vector<int32_t>& offsets) {
    if (offsets.empty()) {
      return;
    }
    offsets.reserve(offsets.size() + extra_rows);
    for (int64_t row = 0; row < extra_rows; ++row) {
      offsets.emplace_back(offsets.back() + layout.attention_row_query_len);
    }
  };
  const auto fill_tail =
      [](torch::Tensor& dst, int64_t dim, int64_t valid, int64_t value) {
        if (valid < dst.size(dim)) {
          dst.narrow(dim, valid, dst.size(dim) - valid).fill_(value);
        }
      };

  const bool padded_reqs = extra_requests > 0;
  const int32_t dim = use_mrope_ ? 1 : 0;
  if (padded_reqs) {
    tokens_.zero_();
    positions_.zero_();
  } else {
    if (tokens.size(0) < tokens_.size(0)) {
      tokens_.zero_();
    }
    if (positions.size(dim) < positions_.size(dim)) {
      positions_.zero_();
    }
  }
  tokens_.narrow(0, 0, tokens.size(0)).copy_(tokens);
  params_.graph.num_valid_token_rows.fill_(
      static_cast<int32_t>(tokens.size(0)));
  positions_.narrow(dim, 0, positions.size(dim)).copy_(positions);
  params_.attention.host = params.attention.host;
  extend_offsets(params_.attention.host.q_seq_lens);
  extend_offsets(params_.attention.host.kv_seq_lens);
  params_.attention.host.q_cu_seq_lens = params_.attention.host.q_seq_lens;
  if (!params_.attention.host.kpool_query_lens.empty()) {
    params_.attention.host.kpool_query_lens.resize(layout.padded_num_reqs,
                                                   layout.tokens_per_request);
  }
  auto& attention = params_.attention.device;
  copy_host(attention.q_seq_lens, params_.attention.host.q_seq_lens);
  const int64_t kv_rows = params.attention.device.kv_seq_lens.numel();
  attention.kv_seq_lens.narrow(0, 0, kv_rows)
      .copy_(params.attention.device.kv_seq_lens);
  if (extra_rows > 0) {
    std::vector<int32_t> tail;
    tail.reserve(extra_rows);
    if (!params_.attention.host.kv_seq_lens.empty()) {
      tail.assign(params_.attention.host.kv_seq_lens.end() - extra_rows,
                  params_.attention.host.kv_seq_lens.end());
    } else {
      const int32_t last =
          params.attention.device.kv_seq_lens[-1].item<int32_t>();
      for (int64_t row = 1; row <= extra_rows; ++row) {
        tail.emplace_back(last + row * layout.attention_row_query_len);
      }
    }
    copy_host(attention.kv_seq_lens.narrow(0, kv_rows, extra_rows), tail);
  }
  if (padded_reqs || tokens.size(0) < attention.new_cache_slots.size(0)) {
    attention.new_cache_slots.fill_(layout.cache_pad_slot);
  }
  attention.new_cache_slots.narrow(0, 0, tokens.size(0))
      .copy_(params.attention.device.new_cache_slots);
  if (attention.block_tables.defined()) {
    const int64_t block_rows = params.attention.device.block_tables.size(0);
    const int64_t copy_width =
        std::min(attention.block_tables.size(1),
                 params.attention.device.block_tables.size(1));
    if (padded_reqs) {
      attention.block_tables.zero_();
      if (block_rows > 0 && copy_width > 0) {
        attention.block_tables.narrow(0, 0, block_rows)
            .narrow(1, 0, copy_width)
            .copy_(
                params.attention.device.block_tables.narrow(1, 0, copy_width));
      }
    } else {
      const bool has_right = copy_width < attention.block_tables.size(1);
      const bool has_bottom = block_rows < attention.block_tables.size(0);
      // Two disjoint fills cost an extra device operation in this shape.
      if (has_right && has_bottom) {
        attention.block_tables.zero_();
      }
      if (block_rows > 0 && copy_width > 0) {
        attention.block_tables.narrow(0, 0, block_rows)
            .narrow(1, 0, copy_width)
            .copy_(
                params.attention.device.block_tables.narrow(1, 0, copy_width));
      }
      if (block_rows > 0 && has_right && !has_bottom) {
        attention.block_tables.narrow(0, 0, block_rows)
            .narrow(1, copy_width, attention.block_tables.size(1) - copy_width)
            .zero_();
      }
      if (!has_right) {
        fill_tail(attention.block_tables, 0, block_rows, 0);
      }
    }
  }
  const auto copy_rows = [padded_reqs](torch::Tensor& dst,
                                       const torch::Tensor& src,
                                       int64_t fill) {
    if (!dst.defined()) {
      return;
    }
    if (padded_reqs) {
      dst.fill_(fill);
    }
    if (src.defined()) {
      if (!padded_reqs && src.size(0) < dst.size(0)) {
        dst.fill_(fill);
      }
      dst.narrow(0, 0, src.size(0)).copy_(src);
    } else if (!padded_reqs) {
      dst.fill_(fill);
    }
  };
  if (params.embedding.linear_state_indices.defined() ||
      params.embedding.linear_state_ids.empty()) {
    copy_rows(params_.embedding.linear_state_indices,
              params.embedding.linear_state_indices,
              kPaddingLinearStateId);
  } else if (params_.embedding.linear_state_indices.defined()) {
    if (extra_requests > 0) {
      params_.embedding.linear_state_indices.fill_(kPaddingLinearStateId);
    }
    copy_host(
        params_.embedding.linear_state_indices.narrow(0, 0, layout.num_reqs),
        params.embedding.linear_state_ids);
  }
  copy_rows(params_.num_accepted_tokens, params.num_accepted_tokens, 1);
  copy_rows(params_.embedding.input_embedding, input_embedding, 0);
  if (params.linear_state_validity_mask_tensor.defined() ||
      params.linear_state_validity_mask.empty()) {
    copy_rows(params_.linear_state_validity_mask_tensor,
              params.linear_state_validity_mask_tensor,
              0);
  } else if (params_.linear_state_validity_mask_tensor.defined()) {
    if (padded_reqs) {
      params_.linear_state_validity_mask_tensor.zero_();
    }
    torch::Tensor mask = torch::tensor(
        params.linear_state_validity_mask,
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    if (!padded_reqs &&
        mask.numel() < params_.linear_state_validity_mask_tensor.size(0)) {
      params_.linear_state_validity_mask_tensor.zero_();
    }
    params_.linear_state_validity_mask_tensor.narrow(0, 0, mask.numel())
        .copy_(mask);
  }
  for (size_t i = 0; i < params.multi_block_tables.size(); ++i) {
    if (padded_reqs) {
      params_.multi_block_tables[i].zero_();
      params_.multi_block_tables[i]
          .narrow(0, 0, params.multi_block_tables[i].size(0))
          .copy_(params.multi_block_tables[i]);
    } else {
      copy_rows(params_.multi_block_tables[i], params.multi_block_tables[i], 0);
    }
  }
  if (params_.linear_state_validity_mask_tensor.defined()) {
    params_.linear_state_validity_mask = params.linear_state_validity_mask;
  }
  if (!params_.linear_state_validity_mask.empty()) {
    params_.linear_state_validity_mask.resize(layout.padded_num_reqs, 0);
  }
  params_.embedding.linear_state_ids = params.embedding.linear_state_ids;
  if (!params_.embedding.linear_state_ids.empty()) {
    params_.embedding.linear_state_ids.resize(layout.padded_num_reqs,
                                              kPaddingLinearStateId);
  }
  if (params_.num_accepted_tokens.defined()) {
    params_.num_accepted_tokens_host = params.num_accepted_tokens_host;
  }
  if (!params_.num_accepted_tokens_host.empty()) {
    params_.num_accepted_tokens_host.resize(layout.padded_num_reqs, 1);
  }
}

MluGraphExecutorImpl::~MluGraphExecutorImpl() = default;

std::size_t GraphPersistentParam::get_persistent_tensor_bytes() const {
  std::size_t total = 0;
  for (const auto& tensor : {output_,
                             positions_,
                             tokens_,
                             aux_hidden_states_,
                             params_.attention.device.q_seq_lens,
                             params_.attention.device.kv_seq_lens,
                             params_.attention.device.new_cache_slots,
                             params_.attention.device.block_tables,
                             params_.embedding.linear_state_indices,
                             params_.embedding.input_embedding,
                             params_.num_accepted_tokens,
                             params_.linear_state_validity_mask_tensor}) {
    total += tensor_bytes(tensor);
  }
  total += tensor_bytes(params_.graph.num_valid_token_rows);
  for (const auto& table : params_.multi_block_tables) {
    total += tensor_bytes(table);
  }
  return total;
}

MluGraph::MluGraph(std::unique_ptr<GraphPersistentParam> persistent_param)
    : persistent_param_(std::move(persistent_param)) {}

std::size_t MluGraph::owned_persistent_tensor_bytes() const {
  return persistent_param_->get_persistent_tensor_bytes();
}

void MluGraph::prepare_model_graph_metadata(CausalLM* model) {
  if (!model->requires_graph_forward_metadata()) {
    return;
  }

  if (!model_graph_metadata_state_) {
    model_graph_metadata_state_ = model->create_graph_forward_metadata_state();
  }
  ModelInputParams& graph_params = persistent_param_->params_;
  model->prepare_graph_forward_metadata(model_graph_metadata_state_.get(),
                                        persistent_param_->positions_,
                                        graph_params);
}

void MluGraph::store_output(const ModelOutput& result) {
  auto store =
      [](torch::Tensor& buffer, const torch::Tensor& value, int64_t capacity) {
        if (!buffer.defined()) {
          auto shape = value.sizes().vec();
          shape[0] = capacity;
          buffer = torch::empty(shape, value.options());
        }
        buffer.narrow(/*dim=*/0, /*start=*/0, value.size(0))
            .copy_(value, /*non_blocking=*/true);
      };
  store(persistent_param_->output_,
        result.hidden_states,
        persistent_param_->tokens_.size(0));
  if (enable_aux_hidden_states_ && result.aux_hidden_states.defined()) {
    store(persistent_param_->aux_hidden_states_,
          result.aux_hidden_states,
          persistent_param_->output_.size(0));
  }
}

ModelOutput MluGraph::output() const {
  const auto& params = *persistent_param_;
  return make_graph_output(
      params.output_, params.aux_hidden_states_, enable_aux_hidden_states_);
}

ModelOutput MluGraph::capture(CausalLM* model,
                              std::vector<KVCache>& kv_cache,
                              const torch_mlu::MempoolId_t& pool,
                              const torch_mlu::MLUStream& capture_stream,
                              const runtime::Options& options) {
  enable_aux_hidden_states_ = options.enable_graph_aux_hidden_states();
  triton_jit::JITKernel::initialize_backend();
  auto forward = [&]() {
    return model->forward(persistent_param_->tokens_,
                          persistent_param_->positions_,
                          kv_cache,
                          persistent_param_->params_);
  };
  const bool has_state =
      std::any_of(kv_cache.begin(), kv_cache.end(), [](const KVCache& cache) {
        return cache.has_request_state();
      });
  if (has_state) {
    // Compile lazy kernels and execute the first request once. Capture only
    // records operations; replay here would advance recurrent state twice.
    store_output(forward());
  }
  const torch_mlu::MLUStream caller_stream =
      torch_mlu::getCurrentMLUStream(capture_stream.device_index());
  torch_mlu::MLUEvent inputs_ready;
  inputs_ready.place(caller_stream);
  inputs_ready.wait(capture_stream);
  {
    torch_mlu::mlu::MLUStreamGuard guard(capture_stream);
    graph_.capture_begin(pool, cnrtQueueCaptureModeRelaxed);
    store_output(forward());
    graph_.capture_end();
  }
  torch_mlu::MLUEvent capture_ready;
  capture_ready.place(capture_stream);
  capture_ready.wait(caller_stream);
  if (!has_state) {
    graph_.replay();
  }
  return output();
}

ModelOutput MluGraph::replay() {
  graph_.replay();
  return output();
}

void MluGraph::update_input_buffer(CausalLM* model,
                                   const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   const ModelInputParams& params,
                                   const GraphLayout& layout) {
  persistent_param_->update_input_buffer(tokens, positions, params, layout);
  // For some models (e.g. DeepSeekV4), the metadata depends on variable host
  // data, which needs to be updated outside of capture.
  prepare_model_graph_metadata(model);
}

MluGraphExecutorImpl::MluGraphExecutorImpl(CausalLM* model,
                                           const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : model_(model),
      args_(args),
      device_(device),
      options_(options),
      graph_pool_(torch_mlu::graph_pool_handle()) {
  mtp_capabilities_ = ModelRegistry::get_mtp_capabilities(args_.model_type());
  int64_t history_capacity =
      ordinary_graph_capacity(args_, options_.block_size());
  if (mtp_capabilities_.graph_history == MtpGraphHistoryPolicy::KPOOL_FIXED) {
    history_capacity = ModelRegistry::get_graph_history_capacity(
        args_.model_type(), args_, history_capacity);
  } else if (mtp_capabilities_.graph_history ==
             MtpGraphHistoryPolicy::QWEN_FULL_ATTENTION) {
    CHECK_GT(args_.max_position_embeddings(), 0)
        << "Qwen3.5 graph requires positive max_position_embeddings";
    history_capacity = args_.max_position_embeddings();
  }
  const int64_t max_tokens = std::max<int64_t>(
      ::xllm::ExecutionConfig::get_instance().max_tokens_for_graph_mode(),
      options_.max_seqs_per_batch());
  GraphPlannerConfig planner_config;
  planner_config.options = options_;
  planner_config.capabilities = mtp_capabilities_;
  planner_config.state_required = model_->requires_graph_forward_metadata();
  planner_config.history_capacity = history_capacity;
  planner_config.max_tokens = max_tokens;
  planner_ = std::make_unique<MluGraphPlanner>(std::move(planner_config));
}

void MluGraphExecutorImpl::set_dp_process_group(ProcessGroup* group) {
  dp_group_ = group;
}

ForwardInput MluGraphExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput MluGraphExecutorImpl::run_eager(const torch::Tensor& tokens,
                                            const torch::Tensor& positions,
                                            std::vector<KVCache>& kv_caches,
                                            const ModelInputParams& params) {
  COUNTER_INC(num_model_execution_total_eager);
  ModelOutput result = model_->forward(tokens, positions, kv_caches, params);
  ModelOutput output =
      make_graph_output(result.hidden_states,
                        result.aux_hidden_states,
                        options_.enable_graph_aux_hidden_states());
  output.mtp_topk_state = std::move(result.mtp_topk_state);
  return output;
}

void MluGraphExecutorImpl::log_memory_after_capture() {
  std::size_t reserved_bytes = 0;
  std::size_t allocated_bytes = 0;
  std::size_t active_bytes = 0;
  std::size_t segment_count = 0;

  try {
    const GraphPoolMemoryUsage usage =
        get_graph_pool_usage(device_.index(), graph_pool_);
    reserved_bytes = usage.reserved_bytes;
    allocated_bytes = usage.allocated_bytes;
    active_bytes = usage.active_bytes;
    segment_count = usage.segment_count;
  } catch (const std::exception& e) {
    VLOG(1) << "Skip MLU graph memory usage log: " << e.what();
  } catch (...) {
    VLOG(1) << "Skip MLU graph memory usage log: unknown allocator error";
  }

  std::size_t persistent_param_bytes = 0;
  for (const auto& [key, graph] : graphs_) {
    persistent_param_bytes += graph->owned_persistent_tensor_bytes();
  }
  // Per-capture delta of the shared graph pool. When scratch is being reused
  // across buckets, this collapses to ~0 after the first (largest) capture;
  // a steady positive delta means each bucket pins its own scratch instead.
  const bool reserved_grew = reserved_bytes >= last_pool_reserved_bytes_;
  const std::size_t reserved_delta =
      reserved_grew ? reserved_bytes - last_pool_reserved_bytes_
                    : last_pool_reserved_bytes_ - reserved_bytes;
  last_pool_reserved_bytes_ = reserved_bytes;
  peak_pool_reserved_bytes_ =
      std::max(peak_pool_reserved_bytes_, reserved_bytes);

  LOG(INFO) << "MluGraphExecutorMemory Usage:"
            << " graphs_cached=" << graphs_.size() << " persistent_param="
            << format_memory_size(persistent_param_bytes)
            << " pool_reserved=" << format_memory_size(reserved_bytes)
            << " pool_reserved_delta=" << (reserved_grew ? "+" : "-")
            << format_memory_size(reserved_delta) << " pool_reserved_peak="
            << format_memory_size(peak_pool_reserved_bytes_)
            << " pool_segments=" << segment_count
            << " allocated_pool_memory=" << format_memory_size(allocated_bytes)
            << " active_pool_memory=" << format_memory_size(active_bytes);
}

// Main execution method with graph optimization for decode phase
// tokens: [num_decode_tokens]
// positions: [num_decode_tokens] token pos in the sequence
// returns: ModelOutput
ModelOutput MluGraphExecutorImpl::run(const torch::Tensor& tokens,
                                      const torch::Tensor& positions,
                                      std::vector<KVCache>& kv_caches,
                                      const ModelInputParams& params) {
  const GraphDecision decision = planner_->plan({tokens.size(0), params});
  const GraphPlan* plan = std::get_if<GraphPlan>(&decision);
  const bool cache_hit =
      plan != nullptr && graphs_.find(plan->key) != graphs_.end();
  const GraphAction action =
      agree_graph_action(decision, cache_hit, dp_group_, device_);
  if (action == GraphAction::EAGER) {
    return run_eager(tokens, positions, kv_caches, params);
  }
  CHECK(plan != nullptr);
  if (action == GraphAction::CAPTURE &&
      params.embedding.input_embedding.defined()) {
    CHECK(params.embedding.input_embedding.device() == device_)
        << "input_embedding graph input device changed";
  }
  const GraphLayout& layout = plan->layout;
  const uint32_t actual_tokens = static_cast<uint32_t>(tokens.size(0));
  const uint32_t graph_tokens = static_cast<uint32_t>(layout.padded_num_tokens);
  const int64_t graph_max_kv_seq_len = plan->history_capacity;
  const bool canonical_model =
      mtp_capabilities_.replay_family == MtpReplayFamily::GLM5 ||
      mtp_capabilities_.replay_family == MtpReplayFamily::QWEN35;
  const GraphKey& key = plan->key;
  auto it = graphs_.find(key);
  ModelOutput result;
  if (action == GraphAction::REPLAY) {
    CHECK(it != graphs_.end());
    it->second->update_input_buffer(model_, tokens, positions, params, layout);
    result = it->second->replay();
  } else {
    // Normalize unused model inputs only when allocating a new graph. Replay
    // copies directly into its active buffers without cloning ModelInputParams.
    auto graph_params = params;
    if (canonical_model) {
      if (!params.is_spec_verify) {
        graph_params.num_accepted_tokens = torch::Tensor();
        graph_params.num_accepted_tokens_host.clear();
      }
      if (params.meta.batch_forward_type.is_decode()) {
        graph_params.linear_state_validity_mask.clear();
        graph_params.linear_state_validity_mask_tensor = torch::Tensor();
      }
    }
    auto buffers =
        std::make_unique<GraphPersistentParam>(tokens,
                                               positions,
                                               std::move(graph_params),
                                               layout,
                                               graph_max_kv_seq_len,
                                               plan->main_block_table_columns);
    auto graph = std::make_unique<MluGraph>(std::move(buffers));
    graph->update_input_buffer(model_, tokens, positions, params, layout);
    if (!graph_capture_stream_.has_value()) {
      graph_capture_stream_ = torch_mlu::getStreamFromPool(
          /*isHighPriority=*/false, device_.index());
    }
    result = graph->capture(
        model_, kv_caches, graph_pool_, *graph_capture_stream_, options_);
    if (std::any_of(
            kv_caches.begin(), kv_caches.end(), [](const KVCache& cache) {
              return cache.has_request_state();
            })) {
      COUNTER_INC(num_model_execution_total_eager);
    }
    // Publish the replacement only after capture succeeds. Old replay work may
    // still reference its graph resources on the caller stream.
    if (it != graphs_.end()) {
      torch_mlu::getCurrentMLUStream(device_.index()).synchronize();
      it->second = std::move(graph);
    } else {
      graphs_.emplace(key, std::move(graph));
    }
    LOG(INFO) << "MLU padded graph captured: padded_num_reqs="
              << layout.padded_num_reqs
              << ", tokens_per_request=" << layout.tokens_per_request
              << ", draft=" << options_.is_draft_engine()
              << ", verify=" << params.is_spec_verify
              << ", padded_num_tokens=" << graph_tokens;
    log_memory_after_capture();
  }
  return make_graph_output(
      result.hidden_states.slice(0, 0, actual_tokens),
      result.aux_hidden_states.defined()
          ? result.aux_hidden_states.slice(0, 0, actual_tokens)
          : torch::Tensor(),
      options_.enable_graph_aux_hidden_states());
}

}  // namespace xllm::mlu
