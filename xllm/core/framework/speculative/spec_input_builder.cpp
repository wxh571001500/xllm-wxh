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

#include "core/framework/speculative/spec_input_builder.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "framework/model/model_input_params.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/forward_params.h"
#include "util/tensor_helper.h"

namespace xllm::specBuilder {

namespace {

// Builds cumulative seq-lens layout: [0, l0, l0+l1, ...].
void push_cumsum(std::vector<int32_t>& vec, int32_t len) {
  if (vec.empty()) {
    vec.emplace_back(0);
  }
  vec.emplace_back(vec.back() + len);
}

Slice<int32_t> tensor_slice(const torch::Tensor& tensor) {
  return {tensor.data_ptr<int32_t>(), static_cast<size_t>(tensor.numel())};
}

Slice<int32_t> get_token_ids(const ForwardInput& input) {
  return tensor_slice(input.token_ids_host);
}

Slice<int32_t> get_positions(const ForwardInput& input) {
  return tensor_slice(input.positions_host);
}

Slice<int32_t> get_kv_seq_lens(const ForwardInput& input) {
  return input.input_params.attention.host.kv_seq_lens;
}

// Resolves a row token from either input token_ids[seq_id] or row.token_id.
int32_t resolve_row_token_id(const DecodeRowContext& ctx, const RowSpec& row) {
  if (!row.use_input_token) {
    return row.token_id;
  }
  CHECK_LT(static_cast<size_t>(row.seq_id), ctx.token_ids.size())
      << "seq_id out of range for token_ids, seq_id=" << row.seq_id
      << ", token_ids_size=" << ctx.token_ids.size();
  return ctx.token_ids[row.seq_id];
}

Slice<int32_t> get_block_table_slice(const DecodeRowContext& ctx,
                                     int32_t seq_id) {
  CHECK_GE(seq_id, 0) << "invalid seq_id=" << seq_id;
  CHECK_LT(seq_id, ctx.num_sequences)
      << "seq_id out of range for block tables, seq_id=" << seq_id
      << ", num_sequences=" << ctx.num_sequences;
  CHECK_GT(ctx.block_table_stride, 0)
      << "invalid block table row stride=" << ctx.block_table_stride;

  const size_t row_offset =
      static_cast<size_t>(seq_id) * static_cast<size_t>(ctx.block_table_stride);
  CHECK_LE(row_offset + static_cast<size_t>(ctx.block_table_stride),
           ctx.block_tables.size())
      << "block table row out of range, seq_id=" << seq_id
      << ", row_offset=" << row_offset
      << ", row_stride=" << ctx.block_table_stride
      << ", block_tables_size=" << ctx.block_tables.size();
  return {ctx.block_tables.data() + row_offset,
          static_cast<size_t>(ctx.block_table_stride)};
}

template <typename T>
void pad_2d_vector(std::vector<std::vector<T>>& vec, T pad_value) {
  size_t max_col_size = 0;
  for (const std::vector<T>& row : vec) {
    max_col_size = std::max(max_col_size, row.size());
  }

  for (std::vector<T>& row : vec) {
    row.resize(max_col_size, pad_value);
  }
}

torch::Tensor create_flat_2d_tensor(const std::vector<int32_t>& values,
                                    int32_t rows,
                                    int32_t stride) {
  if (rows == 0) {
    return torch::Tensor();
  }
  CHECK_GT(rows, 0) << "invalid rows=" << rows;
  CHECK_GT(stride, 0) << "invalid stride=" << stride;
  CHECK_EQ(values.size(), static_cast<size_t>(rows) * stride)
      << "flat 2D tensor size mismatch, rows=" << rows << ", stride=" << stride
      << ", values_size=" << values.size();
  auto tensor = torch::empty({rows, stride},
                             torch::TensorOptions()
                                 .dtype(torch::kInt)
                                 .device(torch::kCPU)
                                 .pinned_memory(true));
  std::copy(values.begin(), values.end(), tensor.data_ptr<int32_t>());
  return tensor;
}

void fill_multi_block_table_slices(DecodeRowContext& ctx) {
  ctx.model_managed_multiblock = !ctx.multi_block_tables_owner.empty();
  ctx.multi_block_tables.resize(ctx.multi_block_tables_owner.size());
  for (size_t m = 0; m < ctx.multi_block_tables_owner.size(); ++m) {
    const torch::Tensor& manager_table = ctx.multi_block_tables_owner[m];
    CHECK(manager_table.defined())
        << "multi_block_tables[" << m << "] is undefined";
    CHECK_EQ(manager_table.dim(), 2)
        << "multi_block_tables[" << m << "] must be 2D, got "
        << manager_table.sizes();
    CHECK_LE(ctx.num_sequences, manager_table.size(0))
        << "num_sequences exceeds multi_block_tables[" << m
        << "] rows, num_sequences=" << ctx.num_sequences
        << ", rows=" << manager_table.size(0);
    ctx.multi_block_tables[m].reserve(static_cast<size_t>(ctx.num_sequences));
    for (int32_t seq_id = 0; seq_id < ctx.num_sequences; ++seq_id) {
      torch::Tensor row = manager_table[seq_id];
      ctx.multi_block_tables[m].emplace_back(row.data_ptr<int32_t>(),
                                             static_cast<size_t>(row.numel()));
    }
  }
}

void append_paged_kv_row(const Slice<int32_t>& block_table,
                         int32_t kv_seq_len,
                         int32_t block_size,
                         DecodeBuildBuffers& buf) {
  CHECK_GT(block_size, 0) << "invalid block_size=" << block_size;
  const int32_t effective_kv_seq_len = std::max(kv_seq_len, 1);
  const int32_t page_count =
      (effective_kv_seq_len + block_size - 1) / block_size;
  CHECK_LE(static_cast<size_t>(page_count), block_table.size())
      << "row KV length exceeds block-table capacity, kv_seq_len=" << kv_seq_len
      << ", page_count=" << page_count
      << ", block_table_size=" << block_table.size();
  if (buf.out_paged_kv_indptr.empty()) {
    buf.out_paged_kv_indptr.emplace_back(0);
  }
  buf.out_paged_kv_indices.insert(buf.out_paged_kv_indices.end(),
                                  block_table.begin(),
                                  block_table.begin() + page_count);
  buf.out_paged_kv_indptr.emplace_back(buf.out_paged_kv_indptr.back() +
                                       page_count);
  buf.out_paged_kv_last_page_len.emplace_back(
      (effective_kv_seq_len - 1) % block_size + 1);
}

}  // namespace

MtpTopkStatePtr select_mtp_topk_state_for_next_step(
    const MtpTopkStatePtr& state,
    const SamplingParameters& sampling_params) {
  if (state == nullptr || !sampling_params.selected_token_idxes.defined()) {
    return state;
  }
  const torch::Tensor& selected_idxes = sampling_params.selected_token_idxes;
  if (selected_idxes.numel() == 0 ||
      state->num_rows() == selected_idxes.numel()) {
    return state;
  }
  if (selected_idxes.device().is_cpu()) {
    CHECK_GT(state->num_rows(), selected_idxes.max().item<int64_t>())
        << "MTP selected top-k index exceeds top-k rows.";
  }
  torch::Tensor index =
      selected_idxes.to(torch::dtype(torch::kLong).device(state->device()))
          .contiguous();
  return state->index_select_rows(index);
}

bool has_decode_lookahead_capacity(const DecodeRowContext& ctx,
                                   int32_t num_tokens,
                                   int32_t block_size,
                                   int64_t max_position_embeddings) {
  CHECK_GT(num_tokens, 0);
  CHECK_GT(block_size, 0);
  for (int32_t seq = 0; seq < ctx.num_sequences; ++seq) {
    const int64_t first_position = ctx.positions[seq];
    const int64_t last_position = first_position + num_tokens - 1;
    const int64_t last_block = last_position / block_size;
    if (first_position < 0 || last_position >= max_position_embeddings ||
        last_block >= ctx.block_table_stride) {
      return false;
    }
    for (int64_t block = first_position / block_size; block <= last_block;
         ++block) {
      if (ctx.block_tables[seq * ctx.block_table_stride + block] < 0) {
        return false;
      }
    }
  }
  return true;
}

int32_t calc_slot_id(int32_t position,
                     const Slice<int32_t>& block_table_slice,
                     int32_t block_size) {
  CHECK_GT(block_size, 0) << "invalid block_size=" << block_size;
  CHECK_GE(position, 0) << "invalid position=" << position;
  const int32_t block_idx = position / block_size;
  CHECK_LT(static_cast<size_t>(block_idx), block_table_slice.size())
      << "block table index out of range, block_idx=" << block_idx
      << ", block_table_size=" << block_table_slice.size()
      << ", position=" << position << ", block_size=" << block_size;
  const int32_t block_id = block_table_slice[block_idx];
  CHECK_GE(block_id, 0) << "invalid block_id=" << block_id;
  const int32_t block_offset = position % block_size;
  return block_id * block_size + block_offset;
}

int32_t calc_ring_slot_id(int32_t position,
                          const Slice<int32_t>& block_table_slice,
                          int32_t block_size) {
  CHECK_GT(block_size, 0) << "invalid block_size=" << block_size;
  CHECK_GE(position, 0) << "invalid position=" << position;
  CHECK(!block_table_slice.empty()) << "circular block table must not be empty";
  const int32_t block_idx =
      (position / block_size) % static_cast<int32_t>(block_table_slice.size());
  const int32_t block_id = block_table_slice[block_idx];
  CHECK_GE(block_id, 0) << "invalid circular block_id=" << block_id;
  const int32_t block_offset = position % block_size;
  return block_id * block_size + block_offset;
}

std::vector<int32_t> build_grouped_prefill_swa_slots(const ForwardInput& input,
                                                     int32_t block_size) {
  DecodeRowContext ctx = make_decode_row_context(input);
  CHECK(ctx.model_managed_multiblock)
      << "grouped prefill SWA slots require multi_block_tables";
  CHECK(!ctx.multi_block_tables.empty())
      << "grouped prefill SWA slots require manager 0";
  CHECK_EQ(ctx.multi_block_tables.front().size(),
           static_cast<size_t>(ctx.num_sequences))
      << "grouped prefill SWA manager row count mismatch";

  std::vector<int32_t> slots;
  slots.reserve(ctx.positions.size());
  for (int32_t seq_id = 0; seq_id < ctx.num_sequences; ++seq_id) {
    const int32_t kv_len = calc_kv_len(ctx.kv_seq_lens, seq_id, /*offset=*/0);
    const int32_t q_len = input.input_params.get_q_seq_len(seq_id);
    CHECK_GE(q_len, 0) << "grouped prefill q length must be non-negative";
    CHECK_GE(kv_len, q_len) << "grouped prefill kv length must cover the query";
    const int32_t query_start = kv_len - q_len;
    const Slice<int32_t>& swa_block_table =
        ctx.multi_block_tables.front()[static_cast<size_t>(seq_id)];
    for (int32_t query_idx = 0; query_idx < q_len; ++query_idx) {
      slots.emplace_back(calc_ring_slot_id(
          query_start + query_idx, swa_block_table, block_size));
    }
  }
  CHECK_EQ(slots.size(), ctx.positions.size())
      << "grouped prefill SWA slot/position count mismatch";
  return slots;
}

int32_t calc_kv_len(const Slice<int32_t>& kv_seq_lens_slice,
                    int32_t seq_id,
                    int32_t offset) {
  CHECK_GE(seq_id, 0) << "invalid seq_id=" << seq_id;
#if defined(USE_NPU)
  CHECK_LT(static_cast<size_t>(seq_id), kv_seq_lens_slice.size())
      << "seq_id out of range, seq_id=" << seq_id
      << ", kv_seq_lens_size=" << kv_seq_lens_slice.size();
  return kv_seq_lens_slice[seq_id] + offset;
#else
  CHECK_LT(static_cast<size_t>(seq_id + 1), kv_seq_lens_slice.size())
      << "seq_id out of range for cumulative layout, seq_id=" << seq_id
      << ", kv_seq_lens_size=" << kv_seq_lens_slice.size();
  return kv_seq_lens_slice[seq_id + 1] - kv_seq_lens_slice[seq_id] + offset;
#endif
}

void append_seq_len_by_layout(std::vector<int32_t>& vec, int32_t len) {
#if defined(USE_NPU)
  vec.emplace_back(len);
#else
  push_cumsum(vec, len);
#endif
}

void append_q_seq_len(std::vector<int32_t>& q_seq_lens,
                      std::vector<int32_t>& q_cu_seq_lens,
                      int32_t len) {
  append_seq_len_by_layout(q_seq_lens, len);
  q_cu_seq_lens.emplace_back(
      (q_cu_seq_lens.empty() ? 0 : q_cu_seq_lens.back()) + len);
}

void set_kpool_verify_metadata(ModelInputParams& input_params,
                               std::vector<int32_t> query_lens) {
  input_params.attention.host.kpool_query_lens = std::move(query_lens);
  input_params.is_spec_verify = true;
}

void update_kv_seq_lens_and_max(std::vector<int32_t>& kv_seq_lens_vec,
                                int32_t kv_len,
                                int32_t& kv_max_seq_len) {
  if (kv_len > kv_max_seq_len) {
    kv_max_seq_len = kv_len;
  }
  append_seq_len_by_layout(kv_seq_lens_vec, kv_len);
}

DecodeRowContext make_decode_row_context(const ForwardInput& input) {
  DecodeRowContext ctx;
  ctx.num_sequences = input.input_params.meta.num_sequences;
  CHECK_GE(ctx.num_sequences, 0) << "invalid num_sequences";

  if (input.token_ids_host.defined()) {
    ctx.token_ids = get_token_ids(input);
  }
  CHECK(input.positions_host.defined())
      << "positions_host must be defined for decode row build";
  CHECK(input.positions_host.is_contiguous());
  ctx.position_axes =
      input.positions_host.dim() == 2 ? input.positions_host.size(0) : 1;
  CHECK(ctx.position_axes == 1 || ctx.position_axes == 3);
  ctx.positions = get_positions(input);
  CHECK_GE(static_cast<int32_t>(ctx.positions.size()), ctx.num_sequences)
      << "positions size is smaller than num_sequences, positions_size="
      << ctx.positions.size() << ", num_sequences=" << ctx.num_sequences;

  ctx.kv_seq_lens = get_kv_seq_lens(input);
  if (!input.input_params.multi_block_tables.empty()) {
    ctx.multi_block_tables_owner.reserve(
        input.input_params.multi_block_tables.size());
    for (const torch::Tensor& block_table :
         input.input_params.multi_block_tables) {
      torch::Tensor cpu_block_table = block_table.device().is_cpu()
                                          ? block_table
                                          : block_table.to(torch::kCPU);
      ctx.multi_block_tables_owner.emplace_back(cpu_block_table.contiguous());
    }
    fill_multi_block_table_slices(ctx);
    return ctx;
  }

  CHECK(input.input_params.attention.host.block_tables.defined())
      << "host block_tables must be defined for decode row build";
  ctx.block_tables_owner =
      input.input_params.attention.host.block_tables.contiguous();
  CHECK_EQ(ctx.block_tables_owner.dim(), 2)
      << "block_tables must be 2D, got " << ctx.block_tables_owner.sizes();
  CHECK_LE(ctx.num_sequences, ctx.block_tables_owner.size(0))
      << "num_sequences exceeds block table rows, num_sequences="
      << ctx.num_sequences
      << ", block_table_rows=" << ctx.block_tables_owner.size(0);
  ctx.block_table_stride = static_cast<int32_t>(ctx.block_tables_owner.size(1));
  CHECK_GT(ctx.block_table_stride, 0)
      << "invalid block table row stride=" << ctx.block_table_stride;
  ctx.block_tables = tensor_slice(ctx.block_tables_owner);
  return ctx;
}

void append_decode_row(const DecodeRowContext& ctx,
                       const RowSpec& row,
                       int32_t block_size,
                       DecodeBuildBuffers& buf) {
  CHECK_GE(row.seq_id, 0);
  CHECK_LT(row.seq_id, ctx.num_sequences);
  CHECK_LT(static_cast<size_t>(row.seq_id), ctx.positions.size());
  const int32_t model_position =
      ctx.positions[row.seq_id] + row.position_offset;
  const int32_t cache_position =
      calc_kv_len(ctx.kv_seq_lens, row.seq_id, row.position_offset) - 1;
  CHECK_GE(model_position, 0) << "invalid decode model position";
  CHECK_GE(cache_position, 0) << "invalid decode cache position";

  // All decode paths can toggle which fields are emitted, so one row builder
  // can serve draft/validate/first-decode/update-last-step scenarios.
  if (row.append_token) {
    buf.out_token_ids.emplace_back(resolve_row_token_id(ctx, row));
  }
  buf.out_positions.emplace_back(model_position);

  std::optional<int32_t> row_kv_seq_len;
  if (row.append_kv_len) {
    const int32_t kv_len_offset =
        row.kv_len_offset.value_or(row.position_offset);
    row_kv_seq_len = calc_kv_len(ctx.kv_seq_lens, row.seq_id, kv_len_offset);
    update_kv_seq_lens_and_max(
        buf.out_kv_seq_lens, *row_kv_seq_len, buf.meta.kv_max_seq_len);
  }

  if (ctx.model_managed_multiblock) {
    CHECK(!ctx.multi_block_tables.empty())
        << "model-managed multiblock input requires an SWA manager";
    CHECK_LT(static_cast<size_t>(row.seq_id),
             ctx.multi_block_tables.front().size());
    buf.out_new_cache_slots.emplace_back(calc_ring_slot_id(
        cache_position,
        ctx.multi_block_tables.front()[static_cast<size_t>(row.seq_id)],
        block_size));
    if (row.append_block_table) {
      if (buf.out_multi_block_tables.size() < ctx.multi_block_tables.size()) {
        buf.out_multi_block_tables.resize(ctx.multi_block_tables.size());
      }
      for (size_t m = 0; m < ctx.multi_block_tables.size(); ++m) {
        CHECK_LT(static_cast<size_t>(row.seq_id),
                 ctx.multi_block_tables[m].size());
        const Slice<int32_t>& block_table_slice =
            ctx.multi_block_tables[m][row.seq_id];
        buf.out_multi_block_tables[m].emplace_back(block_table_slice.begin(),
                                                   block_table_slice.end());
      }
    }
  } else {
    const Slice<int32_t> block_table_slice =
        get_block_table_slice(ctx, row.seq_id);
    buf.out_new_cache_slots.emplace_back(
        calc_slot_id(cache_position, block_table_slice, block_size));
    if (row.append_block_table) {
      if (buf.out_block_table_stride == 0) {
        buf.out_block_table_stride =
            static_cast<int32_t>(block_table_slice.size());
      }
      CHECK_EQ(buf.out_block_table_stride,
               static_cast<int32_t>(block_table_slice.size()))
          << "block table stride mismatch";
      buf.out_block_tables.insert(buf.out_block_tables.end(),
                                  block_table_slice.begin(),
                                  block_table_slice.end());
      ++buf.out_block_table_rows;
      if (row_kv_seq_len.has_value()) {
        append_paged_kv_row(
            block_table_slice, *row_kv_seq_len, block_size, buf);
      }
    }
  }

  if (row.append_q_len_one) {
    append_q_seq_len(buf.out_q_seq_lens, buf.out_q_cu_seq_lens, 1);
  }
}

MtpReplayInputs build_mtp_replay_inputs(
    const DecodeRowContext& ctx,
    const std::vector<EmbeddingCache::DecodeState>& states,
    const torch::Tensor& placeholder,
    int32_t block_size,
    int32_t uniform_width,
    bool is_graph_warmup,
    int32_t pad_cache_slot) {
  CHECK_EQ(states.size(), static_cast<size_t>(ctx.num_sequences));
  CHECK_GE(uniform_width, 0);
  MtpReplayInputs result;
  int32_t capacity = 0;
  for (const auto& state : states) {
    capacity += uniform_width > 0
                    ? uniform_width
                    : std::max<int32_t>(1, state.replay_token_ids.size());
  }
  result.rows.out_token_ids.reserve(capacity);
  result.rows.out_positions.reserve(capacity);
  result.rows.out_new_cache_slots.reserve(capacity);
  result.rows.out_kv_seq_lens.reserve(capacity + 1);
  result.rows.out_q_seq_lens.reserve(capacity + 1);
  result.rows.out_q_cu_seq_lens.reserve(capacity + 1);
  result.rows.out_block_tables.reserve(capacity * ctx.block_table_stride);
  result.embeddings.reserve(capacity);
  result.selected_rows.reserve(ctx.num_sequences);
  result.source_sequences.reserve(capacity);
  result.valid_rows.reserve(capacity);
  if (ctx.position_axes == 3) {
    result.mrope_positions.reserve(capacity);
  }
  for (int32_t seq_id = 0; seq_id < ctx.num_sequences; ++seq_id) {
    const auto& state = states[seq_id];
    const int32_t span = static_cast<int32_t>(state.replay_token_ids.size());
    CHECK(state.valid || is_graph_warmup)
        << "MTP replay requires target context";
    if (state.valid) {
      CHECK_GT(span, 0);
      CHECK_EQ(state.replay_token_ids.back(), ctx.token_ids[seq_id]);
      CHECK(state.replay_embeddings.defined());
      CHECK_EQ(state.replay_embeddings.dim(), 2);
      CHECK_EQ(state.replay_embeddings.size(0), span);
      CHECK_GE(ctx.positions[seq_id], span);
    }
    const int32_t width = uniform_width > 0 ? uniform_width : std::max(1, span);
    CHECK_GE(width, span);
    const int32_t padding = width - span;
    for (int32_t row_id = 0; row_id < width; ++row_id) {
      const int32_t replay_id = row_id - padding;
      const bool valid = replay_id >= 0;
      RowSpec row;
      row.seq_id = seq_id;
      row.token_id = valid ? state.replay_token_ids[replay_id] : 0;
      row.position_offset = valid ? replay_id - span : -ctx.positions[seq_id];
      row.append_q_len_one = true;
      row.append_block_table = true;
      append_decode_row(ctx, row, block_size, result.rows);
      if (!valid) {
        result.rows.out_new_cache_slots.back() = pad_cache_slot;
      }
      result.embeddings.emplace_back(
          valid ? state.replay_embeddings.select(0, replay_id) : placeholder);
      result.source_sequences.emplace_back(seq_id);
      result.valid_rows.emplace_back(valid ? 1 : 0);
      if (ctx.position_axes == 3) {
        std::array<int32_t, 3> rotary = {0, 0, 0};
        if (valid) {
          if (state.replay_mrope_positions.defined()) {
            CHECK_EQ(state.replay_mrope_positions.sizes(),
                     torch::IntArrayRef({3, span}));
            auto stored = state.replay_mrope_positions.accessor<int32_t, 2>();
            for (int32_t axis = 0; axis < 3; ++axis) {
              rotary[axis] = stored[axis][replay_id];
            }
          } else {
            for (int32_t axis = 0; axis < 3; ++axis) {
              rotary[axis] = ctx.positions[axis * ctx.num_sequences + seq_id] +
                             row.position_offset;
            }
          }
        }
        result.mrope_positions.emplace_back(rotary);
      }
    }
    result.selected_rows.emplace_back(result.embeddings.size() - 1);
  }
  return result;
}

TokenWithOffset resolve_token_with_position_offset(
    int32_t input_token_id,
    int32_t seq_id,
    const Slice<int64_t>& last_step_tokens,
    int32_t last_step_decode_num) {
  CHECK_GT(last_step_decode_num, 0)
      << "invalid last_step_decode_num=" << last_step_decode_num;
  if (input_token_id >= 0) {
    TokenWithOffset direct;
    direct.token_id = input_token_id;
    direct.position_offset = 0;
    return direct;
  }

  const int32_t placeholder_idx = -input_token_id - 1;
  CHECK_GE(placeholder_idx, 0)
      << "invalid placeholder token id=" << input_token_id
      << ", seq_id=" << seq_id;
  const int32_t base_idx = placeholder_idx * last_step_decode_num;
  CHECK_LE(base_idx + last_step_decode_num,
           static_cast<int32_t>(last_step_tokens.size()))
      << "last_step_tokens out of range, seq_id=" << seq_id
      << ", placeholder_idx=" << placeholder_idx
      << ", last_step_decode_num=" << last_step_decode_num
      << ", last_step_tokens_size=" << last_step_tokens.size();

  TokenWithOffset resolved;
  resolved.position_offset = -1;
  for (int32_t i = 0; i < last_step_decode_num; ++i) {
    const int64_t candidate = last_step_tokens[base_idx + i];
    if (candidate >= 0) {
      CHECK_LE(candidate,
               static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
          << "token id overflow, seq_id=" << seq_id
          << ", candidate=" << candidate;
      resolved.token_id = static_cast<int32_t>(candidate);
      resolved.position_offset += 1;
    }
  }
  return resolved;
}

void append_decode_row_from_last_step(const DecodeRowContext& ctx,
                                      int32_t seq_id,
                                      int32_t input_token_id,
                                      const Slice<int64_t>& last_step_tokens,
                                      int32_t last_step_decode_num,
                                      int32_t block_size,
                                      DecodeBuildBuffers& buf) {
  // Placeholder tokens (-1/-2/...) are resolved from last-step outputs first,
  // then appended via the same row builder used by all decode paths.
  const TokenWithOffset resolved = resolve_token_with_position_offset(
      input_token_id, seq_id, last_step_tokens, last_step_decode_num);

  RowSpec row;
  row.seq_id = seq_id;
  row.token_id = resolved.token_id;
  row.position_offset = resolved.position_offset;
  append_decode_row(ctx, row, block_size, buf);
}

void update_input_params(ModelInputParams& input_params,
                         DecodeBuildBuffers& buf,
                         int32_t q_max_seq_len,
                         std::vector<int32_t> q_seq_lens_vec,
                         std::vector<int32_t> q_cu_seq_lens_vec,
                         int32_t kv_max_seq_len,
                         std::vector<int32_t> kv_seq_lens_vec,
                         bool update_block_tables) {
  CHECK_EQ(q_seq_lens_vec.empty(), q_cu_seq_lens_vec.empty())
      << "q_seq_lens and q_cu_seq_lens must be provided together";
  input_params.meta.q_max_seq_len = q_max_seq_len;
  input_params.attention.host.q_seq_lens = std::move(q_seq_lens_vec);
  input_params.attention.host.q_cu_seq_lens = std::move(q_cu_seq_lens_vec);
  input_params.meta.kv_max_seq_len = kv_max_seq_len;
  input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  input_params.attention.host.new_cache_slots =
      std::move(buf.out_new_cache_slots);
  if (update_block_tables) {
    if (!buf.out_multi_block_tables.empty()) {
      input_params.multi_block_tables.clear();
      input_params.multi_block_tables.reserve(
          buf.out_multi_block_tables.size());
      for (std::vector<std::vector<int32_t>>& manager_tables :
           buf.out_multi_block_tables) {
        pad_2d_vector(manager_tables, /*pad_value=*/-1);
        input_params.multi_block_tables.emplace_back(
            create_2d_tensor(manager_tables, torch::kInt));
      }
      input_params.attention.host.block_tables = torch::Tensor();
    } else {
      input_params.attention.host.block_tables =
          create_flat_2d_tensor(buf.out_block_tables,
                                buf.out_block_table_rows,
                                buf.out_block_table_stride);
      input_params.multi_block_tables.clear();
      if (!buf.out_paged_kv_indptr.empty()) {
        CHECK_EQ(buf.out_paged_kv_indptr.size(),
                 static_cast<size_t>(buf.out_block_table_rows + 1));
        CHECK_EQ(buf.out_paged_kv_last_page_len.size(),
                 static_cast<size_t>(buf.out_block_table_rows));
        CHECK_EQ(buf.out_paged_kv_indptr.back(),
                 static_cast<int32_t>(buf.out_paged_kv_indices.size()));
        input_params.attention.device.paged_kv_indptr =
            make_cpu_int_tensor(buf.out_paged_kv_indptr);
        input_params.attention.device.paged_kv_indices =
            make_cpu_int_tensor(buf.out_paged_kv_indices);
        input_params.attention.device.paged_kv_last_page_len =
            make_cpu_int_tensor(buf.out_paged_kv_last_page_len);
      }
    }
  }
}

void update_execution_batch_metadata(
    ModelInputParams& input_params,
    std::vector<int32_t> num_scheduled_tokens) {
  ExecutionBatchMetadata& metadata = input_params.execution_batch;
  CHECK_EQ(num_scheduled_tokens.size(), static_cast<size_t>(metadata.num_reqs))
      << "speculative execution widths must remain request-scoped";

  std::vector<int32_t> query_start_loc;
  query_start_loc.reserve(num_scheduled_tokens.size() + 1);
  query_start_loc.emplace_back(0);
  for (int32_t num_tokens : num_scheduled_tokens) {
    CHECK_GT(num_tokens, 0)
        << "each speculative request must execute at least one token";
    CHECK_LE(num_tokens,
             std::numeric_limits<int32_t>::max() - query_start_loc.back())
        << "speculative execution token count exceeds int32 range";
    query_start_loc.emplace_back(query_start_loc.back() + num_tokens);
  }

  metadata.num_tokens = query_start_loc.back();
  metadata.num_scheduled_tokens = std::move(num_scheduled_tokens);
  metadata.query_start_loc = std::move(query_start_loc);
}

torch::Tensor make_cpu_int_tensor(const std::vector<int32_t>& values) {
  return torch::tensor(values,
                       torch::TensorOptions()
                           .dtype(torch::kInt)
                           .device(torch::kCPU)
                           .pinned_memory(true));
}

void set_token_position_tensors(ForwardInput& input,
                                const std::vector<int32_t>& token_ids,
                                const std::vector<int32_t>& positions,
                                const torch::TensorOptions& token_options,
                                const torch::TensorOptions& position_options) {
  input.device_tensors_ready = false;
  input.token_ids_host = make_cpu_int_tensor(token_ids);
  input.positions_host = make_cpu_int_tensor(positions);
  input.token_ids =
      safe_to(input.token_ids_host, token_options, /*non_blocking=*/true);
  input.positions =
      safe_to(input.positions_host, position_options, /*non_blocking=*/true);
  input.device_tensors_ready = true;
}

DraftProposal build_validate_proposal(
    const std::vector<torch::Tensor>& draft_token_ids_steps,
    const std::vector<torch::Tensor>& draft_probs_steps,
    bool draft_probs_required) {
  CHECK(!draft_token_ids_steps.empty()) << "draft steps must not be empty";

  auto draft_token_ids =
      torch::stack(draft_token_ids_steps, /*dim=*/1).to(torch::kLong);
  if (!draft_probs_required) {
    return DraftProposal(std::move(draft_token_ids));
  }

  return DraftProposal(std::move(draft_token_ids),
                       torch::stack(draft_probs_steps, /*dim=*/1));
}

}  // namespace xllm::specBuilder
