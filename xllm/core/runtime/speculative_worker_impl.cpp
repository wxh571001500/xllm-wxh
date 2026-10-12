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

#include "speculative_worker_impl.h"

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/kv_cache/kv_cache_shape.h"
#include "core/framework/model/mtp_utils.h"
#include "core/framework/speculative/spec_input_builder.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/vlm_worker_impl.h"
#include "util/slice.h"
#include "util/tensor_helper.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {
#define TENSOR_REPEAT(tensor_, repeats)                                       \
  do {                                                                        \
    tensor_ = tensor_.defined()                                               \
                  ? tensor_.repeat_interleave(/*repeats=*/repeats, /*dim=*/0) \
                  : tensor_;                                                  \
  } while (0)

Slice<int32_t> tensor_slice(const torch::Tensor& tensor) {
  return {tensor.data_ptr<int32_t>(), static_cast<size_t>(tensor.numel())};
}

std::vector<SpeculativeTokenStats> calculate_contiguous_speculative_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens,
    int32_t accepted_token_offset) {
  CHECK(tokens.defined()) << "speculative output tokens are undefined";
  CHECK_EQ(tokens.dim(), 2) << "speculative output tokens should be 2D";
  const int64_t batch_size = tokens.size(0);
  const int64_t token_width = tokens.size(1);
  CHECK_EQ(proposed_tokens.size(), static_cast<size_t>(batch_size))
      << "proposed token count batch mismatch";

  torch::Tensor int_tokens = tokens.to(torch::kInt64).contiguous();
  const int64_t* data = int_tokens.const_data_ptr<int64_t>();
  std::vector<SpeculativeTokenStats> sequence_stats(
      static_cast<size_t>(batch_size));
  for (int64_t row = 0; row < batch_size; ++row) {
    const int64_t proposed = proposed_tokens[static_cast<size_t>(row)];
    CHECK_GE(proposed, 0) << "proposed token count should not be negative";
    CHECK_LE(proposed + accepted_token_offset, token_width)
        << "proposed token count exceeds output width";

    SpeculativeTokenStats& stats = sequence_stats[static_cast<size_t>(row)];
    stats.proposed_tokens = proposed;
    const int64_t* row_ptr = data + row * token_width;
    for (int64_t column = accepted_token_offset;
         column < proposed + accepted_token_offset;
         ++column) {
      if (row_ptr[column] < 0) {
        break;
      }
      ++stats.accepted_tokens;
    }
  }
  return sequence_stats;
}
}  // namespace

bool should_run_speculative_decode(const ModelInputParams& params) {
  if (!params.meta.batch_forward_type.is_decode()) {
    return false;
  }

  const auto& dp_token_nums = params.parallel.dp_global_token_nums;
  const auto& dp_is_decode = params.parallel.dp_is_decode;
  if (dp_is_decode.empty()) {
    return dp_token_nums.size() <= 1;
  }
  if (dp_is_decode.size() != dp_token_nums.size()) {
    return false;
  }

  // Idle DP ranks (no scheduled tokens this step) must not veto speculative
  // decode for the active ranks. Under enable_graph=False these idle ranks keep
  // dp_is_decode=0 (the backfill in llm_engine only fires when enable_graph=
  // True), which made an all-of-ones check fail for any bs<dp_size batch and
  // silently fell back to the non-speculative path (validate never ran). Only
  // ranks that actually carry tokens gate the decision; require every such rank
  // to be in decode.
  bool any_active = false;
  for (size_t i = 0; i < dp_is_decode.size(); ++i) {
    if (dp_token_nums[i] == 0) {
      continue;  // idle rank: does not participate in the vote
    }
    any_active = true;
    if (dp_is_decode[i] != 1) {
      return false;
    }
  }
  return any_active;
}

void scale_speculative_parallel_token_counts(ModelInputParams& params,
                                             int32_t multiplier) {
  for (int32_t& token_num : params.parallel.dp_global_token_nums) {
    token_num *= multiplier;
  }
  for (int32_t& token_num : params.parallel.raw_dp_global_token_nums) {
    token_num *= multiplier;
  }
  params.expert.eplb_decode_token_mask = eplb::expand_decode_token_mask(
      params.expert.eplb_decode_token_mask, multiplier);
}

// Validation replaces each request's single decode token with its full query
// width. num_reqs stays the logical request count; only the token layout grows
// so it matches the expanded token_ids tensor.
void align_execution_batch_query_widths(
    ExecutionBatchMetadata& batch,
    const std::vector<int32_t>& query_widths) {
  if (batch.num_reqs == 0) {
    return;
  }
  CHECK_EQ(static_cast<int32_t>(query_widths.size()), batch.num_reqs);
  CHECK_EQ(batch.num_scheduled_tokens.size(),
           static_cast<size_t>(batch.num_reqs));
  int64_t scheduled_tokens = 0;
  for (int32_t count : batch.num_scheduled_tokens) {
    scheduled_tokens += count;
  }
  int64_t expanded_tokens = 0;
  for (int32_t width : query_widths) {
    CHECK_GT(width, 0);
    expanded_tokens += width;
  }
  if (batch.num_tokens == expanded_tokens &&
      scheduled_tokens == expanded_tokens) {
    return;
  }
  CHECK_EQ(scheduled_tokens, static_cast<int64_t>(batch.num_reqs))
      << "speculative verify expected one scheduled token per request "
         "before expanding the validation width";
  batch.num_scheduled_tokens = query_widths;
  std::vector<int32_t> query_start_loc;
  query_start_loc.reserve(query_widths.size() + 1);
  query_start_loc.emplace_back(0);
  for (int32_t width : query_widths) {
    query_start_loc.emplace_back(query_start_loc.back() + width);
  }
  batch.query_start_loc = std::move(query_start_loc);
  batch.num_tokens = expanded_tokens;
}

std::vector<SpeculativeTokenStats> calculate_mtp_speculative_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens) {
  return calculate_contiguous_speculative_token_stats(
      tokens, proposed_tokens, /*accepted_token_offset=*/1);
}

std::vector<SpeculativeTokenStats> calculate_block_speculative_token_stats(
    const torch::Tensor& tokens,
    const std::vector<int32_t>& proposed_tokens) {
  // The contiguous output includes one target-model token: the replacement
  // at the first rejection, or the bonus after all drafts are accepted.
  // Count one fewer valid token, capped by each row's actual proposal width.
  return calculate_contiguous_speculative_token_stats(
      tokens, proposed_tokens, /*accepted_token_offset=*/1);
}

SpeculativeOutputStats calculate_speculative_output_stats(
    const torch::Tensor& tokens,
    int64_t num_speculative_tokens) {
  torch::Tensor int_tokens = tokens.to(torch::kInt64).contiguous();
  const int64_t* data = int_tokens.const_data_ptr<int64_t>();
  const int64_t batch_size = int_tokens.size(0);
  const int64_t token_width = int_tokens.size(1);
  CHECK_LE(token_width, num_speculative_tokens + 1)
      << "next_tokens width exceeds num_speculative_tokens + 1.";
  SpeculativeOutputStats stats;
  stats.accepted_per_position.resize(
      static_cast<size_t>(num_speculative_tokens));
  stats.sequence_stats.resize(static_cast<size_t>(batch_size));
  for (int64_t row = 0; row < batch_size; ++row) {
    const int64_t* row_ptr = data + row * token_width;
    SpeculativeTokenStats& sequence_stats =
        stats.sequence_stats[static_cast<size_t>(row)];
    sequence_stats.proposed_tokens = token_width - 1;
    for (int64_t column = 0; column < token_width; ++column) {
      if (row_ptr[column] < 0) {
        continue;
      }
      ++stats.committed_tokens;
      if (column > 0) {
        ++stats.accepted_per_position[static_cast<size_t>(column - 1)];
        ++sequence_stats.accepted_tokens;
      }
    }
  }
  return stats;
}

SpeculativeWorkerImpl::SpeculativeWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    const runtime::Options& target_options,
    WorkerType worker_type)
    : WorkerImpl(parallel_args, device, options) {
  if (worker_type == WorkerType::LLM) {
    impl_ =
        std::make_unique<LLMWorkerImpl>(parallel_args, device, target_options);
  } else if (worker_type == WorkerType::VLM) {
    impl_ =
        std::make_unique<VLMWorkerImpl>(parallel_args, device, target_options);
  } else {
    LOG(FATAL) << "Unsupported speculative worker type: "
               << worker_type.to_string();
  }
}

SpeculativeWorkerImpl::~SpeculativeWorkerImpl() {
  // Draft-model subclasses shut down the shared transfer's threadpool first
  // (see ~DraftModelSpecWorkerImpl); this base only releases refs.
  if (impl_ != nullptr) {
    impl_->clear_hierarchy_kv_cache_transfer();
  }
  clear_hierarchy_kv_cache_transfer();
}

bool SpeculativeWorkerImpl::init_model(const std::string& model_weights_path,
                                       int32_t random_seed,
                                       MasterStatus master_status) {
  // Base class only loads the target model.
  bool result = true;
  CHECK(impl_ != nullptr);
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = impl_->WorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
    if (result) {
      dtype_ = impl_->dtype();
    }
  }
  enable_fused_kernel_ =
      impl_->get_optimization_config().enable_fused_spec_kernel;
  return result;
}

bool SpeculativeWorkerImpl::allocate_kv_cache(
    const KVCacheShape& kv_cache_shape) {
  return impl_->allocate_kv_cache(kv_cache_shape);
}

#if defined(USE_NPU)
bool SpeculativeWorkerImpl::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  return impl_->allocate_kv_cache_with_transfer(kv_cache_shape);
}
#endif

std::optional<ForwardOutput> SpeculativeWorkerImpl::step(
    const ForwardInput& input) {
  ModelInputParams& mutable_params =
      const_cast<ModelInputParams&>(input.input_params);
  set_hierarchy_layer_synchronizer(mutable_params);
  const bool run_speculative_decode =
      should_run_speculative_decode(input.input_params);
  if (input.input_params.meta.num_sequences == 0 ||
      input.token_ids.numel() == 0) {
    if (input.input_params.meta.batch_forward_type.is_decode() &&
        !run_speculative_decode) {
      ForwardInput aligned_input = input;
      aligned_input.input_params.meta.batch_forward_type =
          BatchForwardType::EMPTY;
      return step_empty(aligned_input);
    }
    return step_empty(input);
  }

  if (run_speculative_decode) {
    return step_decode(input);
  }
  return step_prefill(input);
}

ForwardInput SpeculativeWorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
  // only process decode batch, so prepare draft input here.
  ForwardInput& new_inputs = inputs;

  auto& input_params = new_inputs.input_params;
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t block_size = logical_block_size();

  Slice<int32_t> token_ids = tensor_slice(inputs.token_ids_host);
  torch::Tensor last_token_ids = safe_to(
      last_step_output_.sample_output.next_tokens.flatten(), torch::kCPU);
  Slice<int64_t> last_tokens_ids_slice = {
      last_token_ids.data_ptr<int64_t>(),
      static_cast<size_t>(last_token_ids.numel())};

  // Determine how many tokens were decoded in the last step
  // If the output is 2D, it means multiple tokens were generated per sequence
  int32_t last_step_decode_num = 1;
  if (last_step_output_.sample_output.next_tokens.dim() == 2) {
    last_step_decode_num = last_step_output_.sample_output.next_tokens.size(1);
  }

  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(num_sequences);
  buf.out_positions.reserve(num_sequences);
  buf.out_kv_seq_lens.reserve(num_sequences);
  buf.out_new_cache_slots.reserve(num_sequences);
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(inputs);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    specBuilder::append_decode_row_from_last_step(row_ctx,
                                                  seq_id,
                                                  token_ids[seq_id],
                                                  last_tokens_ids_slice,
                                                  last_step_decode_num,
                                                  block_size,
                                                  buf);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "step-update kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "step-update positions/tokens mismatch";

  specBuilder::set_token_position_tensors(new_inputs,
                                          buf.out_token_ids,
                                          buf.out_positions,
                                          inputs.token_ids.options(),
                                          inputs.positions.options());
  // update the input_params
  input_params.meta.kv_max_seq_len = buf.meta.kv_max_seq_len;
  input_params.attention.host.kv_seq_lens = std::move(buf.out_kv_seq_lens);
  input_params.attention.host.new_cache_slots =
      std::move(buf.out_new_cache_slots);
  input_params.attention.rebuild_device_buffer(device_);
  new_inputs.device_tensors_ready = true;

  return new_inputs;
}

void SpeculativeWorkerImpl::update_sampling_params(
    SamplingParameters& sampling_params,
    const int32_t num_val_tokens,
    const int32_t total_num_val_tokens,
    const bool repeat_unique_history) {
  std::vector<int32_t> selected_token_idxes_vec;
  selected_token_idxes_vec.reserve(total_num_val_tokens);
  for (int32_t i = 0; i < total_num_val_tokens; i++) {
    selected_token_idxes_vec.emplace_back(i);
  }
  torch::Tensor selected_token_idxes = torch::tensor(selected_token_idxes_vec);

  // sample_idxes equals to selected_token_idxes since only process decode batch
  sampling_params.selected_token_idxes = selected_token_idxes.to(device_);
  sampling_params.sample_idxes = selected_token_idxes.to(device_);

  TENSOR_REPEAT(sampling_params.frequency_penalties, num_val_tokens);
  TENSOR_REPEAT(sampling_params.presence_penalties, num_val_tokens);
  TENSOR_REPEAT(sampling_params.repetition_penalties, num_val_tokens);
  TENSOR_REPEAT(sampling_params.temperatures, num_val_tokens);
  TENSOR_REPEAT(sampling_params.top_p, num_val_tokens);
  TENSOR_REPEAT(sampling_params.top_k, num_val_tokens);
  if (repeat_unique_history) {
    TENSOR_REPEAT(sampling_params.unique_token_ids, num_val_tokens);
    TENSOR_REPEAT(sampling_params.unique_token_counts, num_val_tokens);
    TENSOR_REPEAT(sampling_params.unique_token_ids_lens, num_val_tokens);
  }
  TENSOR_REPEAT(sampling_params.do_sample, num_val_tokens);
  TENSOR_REPEAT(sampling_params.filter_mask, num_val_tokens);
  TENSOR_REPEAT(sampling_params.filter_bitmask, num_val_tokens);
}

void SpeculativeWorkerImpl::update_sampling_params(
    SamplingParameters& sampling_params,
    const std::vector<int32_t>& per_seq_val_tokens,
    const int32_t total_num_val_tokens) {
  std::vector<int32_t> selected_token_idxes_vec;
  selected_token_idxes_vec.reserve(total_num_val_tokens);
  for (int32_t i = 0; i < total_num_val_tokens; i++) {
    selected_token_idxes_vec.emplace_back(i);
  }
  torch::Tensor selected_token_idxes = torch::tensor(selected_token_idxes_vec);
  sampling_params.selected_token_idxes = selected_token_idxes.to(device_);
  // Alias sample_idxes to the already-uploaded device tensor rather than
  // paying a second identical H2D copy.
  sampling_params.sample_idxes = sampling_params.selected_token_idxes;

  torch::Tensor repeats_tensor =
      torch::tensor(std::vector<int64_t>(per_seq_val_tokens.begin(),
                                         per_seq_val_tokens.end()),
                    torch::kLong)
          .to(device_);
  auto repeat_per_seq = [&](torch::Tensor& tensor) {
    if (!tensor.defined()) {
      return;
    }
    tensor = tensor.repeat_interleave(repeats_tensor, /*dim=*/0);
  };
  repeat_per_seq(sampling_params.frequency_penalties);
  repeat_per_seq(sampling_params.presence_penalties);
  repeat_per_seq(sampling_params.repetition_penalties);
  repeat_per_seq(sampling_params.temperatures);
  repeat_per_seq(sampling_params.top_p);
  repeat_per_seq(sampling_params.top_k);
  repeat_per_seq(sampling_params.unique_token_ids);
  repeat_per_seq(sampling_params.unique_token_counts);
  repeat_per_seq(sampling_params.unique_token_ids_lens);
  repeat_per_seq(sampling_params.do_sample);
}

void SpeculativeWorkerImpl::prepare_validate_inputs(
    const ForwardInput& input,
    ForwardInput& validate_input,
    const bool repeat_unique_history) {
  validate_input = input.to(device_, dtype_);
  validate_input.device_tensors_ready = false;
  auto& input_params = validate_input.input_params;
  torch::TensorOptions token_options = validate_input.token_ids.options();
  torch::TensorOptions position_options = validate_input.positions.options();

  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const int32_t total_num_val_tokens = num_sequences * num_val_tokens;
  const int32_t block_size = logical_block_size();
  // Hybrid targets (for example Qwen3.8 GDN) mark validation as spec-verify
  // before entering this generic builder.  They must keep one sequence row
  // with an N+1-token query so recurrent state is checkpointed and committed
  // by the model's spec-verify kernel instead of being expanded into N+1
  // independent decode rows.
  const bool use_chunked_spec_verify =
      ::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel() ||
      input.input_params.is_spec_verify;
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);

  Slice<int32_t> token_ids = tensor_slice(input.token_ids_host);
  Slice<int32_t> kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(total_num_val_tokens);
  buf.out_positions.reserve(total_num_val_tokens);
  buf.out_new_cache_slots.reserve(total_num_val_tokens);
  if (!use_chunked_spec_verify) {
    buf.out_kv_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_cu_seq_lens.reserve(total_num_val_tokens);
    buf.out_block_tables.reserve(static_cast<size_t>(total_num_val_tokens) *
                                 row_ctx.block_table_stride);
  }

  std::vector<int32_t> atb_kv_seq_lens_vec = {};
  std::vector<int32_t> atb_q_seq_lens_vec = {};
  std::vector<int32_t> atb_q_cu_seq_lens_vec = {};
  int32_t atb_kv_max_seq_len = 0;
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    int32_t kv_len =
        specBuilder::calc_kv_len(kv_seq_lens, seq_id, /*offset=*/0);
    for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      if (val_idx == 0) {
        row.token_id = token_ids[seq_id];
      } else {
        row.token_id = -val_idx;
      }
      row.position_offset = val_idx;
      row.append_kv_len = !use_chunked_spec_verify;
      row.append_q_len_one = !use_chunked_spec_verify;
      row.append_block_table = !use_chunked_spec_verify;
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
    }

    if (use_chunked_spec_verify) {
      const int32_t kv_len_after_validation = kv_len + num_speculative_tokens;
      specBuilder::update_kv_seq_lens_and_max(
          atb_kv_seq_lens_vec, kv_len_after_validation, atb_kv_max_seq_len);
      specBuilder::append_q_seq_len(
          atb_q_seq_lens_vec, atb_q_cu_seq_lens_vec, num_val_tokens);
    }
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "validate kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "validate positions/tokens mismatch";

  specBuilder::set_token_position_tensors(validate_input,
                                          buf.out_token_ids,
                                          buf.out_positions,
                                          token_options,
                                          position_options);
  // update the input_params
  if (!use_chunked_spec_verify) {
    input_params.meta.num_sequences = total_num_val_tokens;
    input_params.meta.q_max_seq_len = 1;
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  } else {
    input_params.meta.q_max_seq_len = num_val_tokens;
    input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  }
  if (use_chunked_spec_verify) {
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
  specBuilder::update_execution_batch_metadata(
      input_params,
      std::vector<int32_t>(static_cast<size_t>(num_sequences), num_val_tokens));
  input_params.attention.rebuild_device_buffer(device_);

  // update the sampling_params
  update_sampling_params(validate_input.sampling_params,
                         num_val_tokens,
                         total_num_val_tokens,
                         repeat_unique_history);

  scale_speculative_parallel_token_counts(input_params, num_val_tokens);
  align_execution_batch_query_widths(
      input_params.execution_batch,
      std::vector<int32_t>(static_cast<size_t>(num_sequences), num_val_tokens));
  validate_input.device_tensors_ready = true;
}

// Per-seq adaptive validate builder: each sequence contributes
// per_seq_val_tokens[i] rows instead of a uniform N+1. Only implements the
// chunked-prefill (non-atb_spec_kernel) path since DFlash/DSpark require
// --enable_chunked_prefill=true anyway.
void SpeculativeWorkerImpl::prepare_validate_inputs(
    const ForwardInput& input,
    ForwardInput& validate_input,
    const std::vector<int32_t>& per_seq_val_tokens) {
  validate_input = input.to(device_, dtype_);
  validate_input.device_tensors_ready = false;
  auto& input_params = validate_input.input_params;
  torch::TensorOptions token_options = validate_input.token_ids.options();
  torch::TensorOptions position_options = validate_input.positions.options();

  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input_params.meta.num_sequences;
  CHECK_EQ(static_cast<int32_t>(per_seq_val_tokens.size()), num_sequences)
      << "per_seq_val_tokens size must match num_sequences";
  int32_t total_num_val_tokens = 0;
  int32_t max_val_tokens = 0;
  for (int32_t v : per_seq_val_tokens) {
    CHECK_GE(v, 1) << "per_seq_val_tokens must be >= 1";
    CHECK_LE(v, num_speculative_tokens + 1)
        << "per_seq_val_tokens must be <= num_speculative_tokens + 1";
    total_num_val_tokens += v;
    if (v > max_val_tokens) {
      max_val_tokens = v;
    }
  }
  const int32_t block_size = logical_block_size();
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);

  Slice<int32_t> token_ids = tensor_slice(input.token_ids_host);
  Slice<int32_t> kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(total_num_val_tokens);
  buf.out_positions.reserve(total_num_val_tokens);
  buf.out_new_cache_slots.reserve(total_num_val_tokens);
  buf.out_kv_seq_lens.reserve(total_num_val_tokens);
  buf.out_q_seq_lens.reserve(total_num_val_tokens);
  buf.out_q_cu_seq_lens.reserve(total_num_val_tokens);
  buf.out_block_tables.reserve(static_cast<size_t>(total_num_val_tokens) *
                               row_ctx.block_table_stride);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    int32_t kv_len =
        specBuilder::calc_kv_len(kv_seq_lens, seq_id, /*offset=*/0);
    const int32_t seq_val_tokens =
        per_seq_val_tokens[static_cast<size_t>(seq_id)];

    for (int32_t val_idx = 0; val_idx < seq_val_tokens; ++val_idx) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      if (val_idx == 0) {
        row.token_id = token_ids[seq_id];
      } else {
        row.token_id = -val_idx;
      }
      row.position_offset = val_idx;
      row.append_kv_len = true;
      row.append_q_len_one = true;
      row.append_block_table = true;
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
    }
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "validate kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "validate positions/tokens mismatch";

  specBuilder::set_token_position_tensors(validate_input,
                                          buf.out_token_ids,
                                          buf.out_positions,
                                          token_options,
                                          position_options);
  // Match the dense (non-adaptive) validate path's DECODE-mode layout: each
  // validate row is an independent q=1 decode step, and causal visibility
  // across a seq's block comes from the per-row increasing kv_seq_lens (row j
  // sees anchor_kv + j tokens), NOT from a chunked-prefill block mask. Using
  // CHUNKED_PREFILL here (q_max_seq_len = max_val_tokens) gave the block a
  // prefill-style mask under which col>=1 could not attend to the accepted
  // draft tokens in col<j, so the target logits from col 1 onward diverged
  // from the dense path and produced garbled adaptive output. Flatten to
  // total_num_val_tokens q=1 rows exactly like the dense builder.
  input_params.meta.num_sequences = total_num_val_tokens;
  input_params.meta.q_max_seq_len = 1;
  input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  specBuilder::update_input_params(input_params,
                                   buf,
                                   /*val_tokens_per_seq=*/1,
                                   std::move(buf.out_q_seq_lens),
                                   std::move(buf.out_q_cu_seq_lens),
                                   buf.meta.kv_max_seq_len,
                                   std::move(buf.out_kv_seq_lens),
                                   /*update_block_tables=*/true);
  specBuilder::update_execution_batch_metadata(input_params,
                                               per_seq_val_tokens);
  input_params.attention.rebuild_device_buffer(device_);

  // update sampling params using the per-seq width.
  update_sampling_params(
      validate_input.sampling_params, per_seq_val_tokens, total_num_val_tokens);

  // Note: dp_global_token_nums is NOT scaled here. Under adaptive pruning each
  // DP rank's validate token count is data-dependent, so a rank-local estimate
  // (e.g. average width) would diverge across ranks and desync the MoE
  // all-to-all pads. The authoritative per-rank counts are gathered over the DP
  // group by the draft-model worker's sync_dp_global_token_nums_after_prune(),
  // called on every DP rank right before the target validate forward.
  align_execution_batch_query_widths(input_params.execution_batch,
                                     per_seq_val_tokens);
  validate_input.device_tensors_ready = true;
}

void SpeculativeWorkerImpl::restore_json_object_states(ForwardInput& input) {
  impl_->restore_json_object_states(input);
}
}  // namespace xllm
