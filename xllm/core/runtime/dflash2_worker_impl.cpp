/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "runtime/dflash2_worker_impl.h"

#include <glog/logging.h>

#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#if defined(USE_NPU)
#include "core/platform/npu/device_capture_lock.h"
#include "torch_npu/csrc/core/npu/NPUGraph.h"
#endif
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/speculative/dflash_async_input_builder.h"
#include "core/framework/speculative/spec_input_builder.h"
#include "framework/sampling/gumbel_sampling.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/attention_metadata_builder.h"
#include "runtime/speculative_worker_utils.h"
#include "util/timer.h"

namespace xllm {

namespace {

int64_t next_pow2_block_table_cols(int64_t cols) {
  CHECK_GE(cols, 0);
  int64_t padded = 16;
  while (padded < cols) {
    padded *= 2;
  }
  CHECK_LT(padded, 1LL << 28)
      << "DFlash2 block-table columns exceed graph capacity: " << cols;
  return padded;
}

}  // namespace

struct DFlash2WorkerImpl::DraftGraphResources {
  std::unique_ptr<Stream> capture_stream;
#if defined(USE_NPU)
  c10_npu::MempoolId_t pool = c10_npu::graph_pool_handle();
#endif
};

struct DFlash2WorkerImpl::DraftGraph {
  ForwardInput input;
  ForwardInput query;
  torch::Tensor accepted_tokens;
  torch::Tensor embeddings;
  torch::Tensor base_positions;
  torch::Tensor noise;
  DraftBlock output;
#if defined(USE_NPU)
  c10_npu::NPUGraph graph;
#endif
};

DFlash2WorkerImpl::DFlash2WorkerImpl(const ParallelArgs& parallel_args,
                                     const torch::Device& device,
                                     const runtime::Options& options)
    : DFlashWorkerImpl(parallel_args, device, options),
      sampling_process_group_(
          speculative_worker::sampling_process_group(parallel_args)) {
  prelaunch_enabled_ =
      enable_dflash_proposal_xfia(parallel_args, device, options);
  prelaunch_graph_enabled_ =
      prelaunch_enabled_ && ExecutionConfig::get_instance().enable_graph();
}

DFlash2WorkerImpl::~DFlash2WorkerImpl() { discard_draft_prelaunch(); }

DFlashWorkerImpl::DraftBlock DFlash2WorkerImpl::run_decode_draft(
    const ForwardInput& input,
    ForwardInput& validate_input) {
  if (pending_draft_.has_value()) {
    if (pending_draft_matches(input)) {
      if (pending_draft_->next_template.has_value()) {
        DraftTemplate& next = *pending_draft_->next_template;
        prepared_prelaunch_ = std::move(next.query);
        prepared_sampling_ = std::move(next.sampling);
        prepared_noise_ = std::move(next.noise);
        prepared_base_positions_ = std::move(next.base_positions);
      }
      if (pending_draft_->target_input.has_value()) {
        validate_input = std::move(*pending_draft_->target_input);
        validate_input.input_params.meta.batch_id =
            input.input_params.meta.batch_id;
      } else {
        prepare_validate_inputs(input, validate_input);
      }
      DraftBlock block = std::move(pending_draft_->block);
      // Keep producer tensors alive through this iteration's target verify.
      auto retained =
          std::make_shared<ForwardInput>(std::move(pending_draft_->identity));
      retained->input_params.embedding.input_embedding =
          std::move(pending_draft_->retained_embeddings);
      retained->metadata_ready_event = std::move(pending_draft_->ready);
      block.retained_inputs.emplace_back(std::move(retained));
      pending_draft_.reset();
      return block;
    }
    discard_draft_prelaunch();
  }
  ForwardInput query_input;
  prepare_query_inputs(input, query_input);
  // Proposal blocks share token counts with ordinary chunks. Mark only the
  // prelaunch mode, where XFIA is enabled, and do not reuse the target
  // spec-verify flag. Graph-warmup batches keep the plain chunk metadata:
  // their draft forwards must not take the XFIA proposal path.
  if (prelaunch_enabled_ && !input.input_params.meta.is_graph_warmup) {
    query_input.input_params.is_dflash_proposal = true;
  }

  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_speculative_tokens, 0);
  CHECK(input.token_ids_host.defined());
  CHECK_GE(input.token_ids_host.numel(), batch_size);
  torch::Tensor anchor_token_ids =
      input.token_ids_host.slice(/*dim=*/0, /*start=*/0, /*end=*/batch_size)
          .to(draft_impl_->device(), torch::kLong);

  // Build the Gumbel noise and finish its TP consensus up front: one
  // host-blocking broadcast here replaces the seven per-step index
  // broadcasts inside the previous path walk, so the loop below runs
  // device-side end to end and never blocks the host mid-loop.
  const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
  const int64_t selector_top_k = draft_args.dflash2_selector_top_k();
  c10::StreamGuard noise_guard = compute_stream_->set_stream_guard();
  torch::Tensor gumbel_noise = sample_gumbel_noise(batch_size,
                                                   num_speculative_tokens,
                                                   selector_top_k,
                                                   input.sampling_params,
                                                   draft_impl_->device());
  // Cross-rank RNG divergence must not fork the sampled path: unify the noise
  // once from rank 0.  The edge logits are TP-replicated, so identical noise
  // yields identical argmax on every rank for all steps.
  if (!input.sampling_params.all_greedy_sample &&
      sampling_process_group_ != nullptr &&
      sampling_process_group_->world_size() > 1) {
    gumbel_noise = gumbel_noise.contiguous();
    sampling_process_group_->broadcast(gumbel_noise, /*root_rank=*/0);
  }

  DraftBlock block = execute_draft_query(
      input, std::move(query_input), anchor_token_ids, gumbel_noise, false);
  prepare_validate_inputs(input, validate_input);
  return block;
}

DFlashWorkerImpl::DraftBlock DFlash2WorkerImpl::execute_draft_query(
    const ForwardInput& input,
    ForwardInput query_input,
    const torch::Tensor& anchor_token_ids,
    const torch::Tensor& gumbel_noise,
    bool prepared,
    Stream* execution_stream) {
  Stream& stream =
      execution_stream == nullptr ? *compute_stream_ : *execution_stream;
  Timer timer;
  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  query_input.skip_sampling_for_logits_only = true;
  query_input.return_selected_hidden = true;
  ForwardInput processed_input;
  if (prepared) {
    processed_input = std::move(query_input);
  } else {
    draft_impl_->prepare_work_before_execute_on_stream(
        query_input,
        processed_input,
        *prepare_stream_,
        /*record_ready_event=*/prepare_stream_.get() != compute_stream_.get());
  }
  draft_impl_->set_hierarchy_layer_synchronizer(processed_input.input_params);
  std::optional<ForwardOutput> draft_output =
      draft_impl_->execute_no_sync_on_stream(processed_input,
                                             stream,
                                             /*record_ready_event=*/false);
  CHECK(draft_output.has_value());
  CHECK(draft_output->logits.defined());
  CHECK(draft_output->selected_hidden.defined());
  const int64_t num_rows = draft_output->logits.size(0);
  CHECK_EQ(num_rows, static_cast<int64_t>(batch_size) * num_speculative_tokens);
  torch::Tensor unary_logits = draft_output->logits.view(
      {batch_size, num_speculative_tokens, draft_output->logits.size(-1)});
  torch::Tensor hidden_states = draft_output->selected_hidden.view(
      {batch_size,
       num_speculative_tokens,
       draft_output->selected_hidden.size(-1)});

  BlockSampleOutput sampled;
  {
    c10::StreamGuard stream_guard = stream.set_stream_guard();
    DFlash2CandidateOutput candidates = draft_impl_->dflash2_candidates(
        hidden_states, unary_logits, anchor_token_ids);
    SamplingParameters sampling_params = input.sampling_params.to(
        unary_logits.device(), unary_logits.scalar_type());
    sampled = sample_path(candidates,
                          sampling_params,
                          gumbel_noise,
                          unary_logits.size(/*dim=*/-1));
  }

  DraftBlock draft_block;
  // DFlash2 samples selector paths from a sparse top-k distribution; the
  // dense per-token proposal must be retained so rejection recovery stays
  // exact. The selected-only probs carry no extra information for the
  // verifier, so only token_ids and the dense proposal feed the DraftProposal.
  draft_block.proposal = DraftProposal(std::move(sampled.token_ids),
                                       std::move(sampled.dense_probs));
  draft_block.retained_inputs = take_retained_inputs(*draft_output);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return draft_block;
}

bool DFlash2WorkerImpl::can_prepare_without_compute_stream_wait(
    const ForwardInput& input) const {
  // Reuse the owner's actual enablement, including A3/Python, overlap,
  // CP, PD role and adaptive-decode restrictions. The outer worker stages
  // fresh metadata; it owns neither the leaf graph's persistent buffers nor
  // its KV cache. Do not extend this exemption to the leaf workers.
  return prelaunch_graph_enabled_ &&
         input.input_params.meta.batch_forward_type.is_decode() &&
         can_stage_prelaunch_input(input);
}

bool DFlash2WorkerImpl::can_stage_prelaunch_input(
    const ForwardInput& input) const {
  if (!prelaunch_enabled_ || input.input_params.meta.is_graph_warmup ||
      !input.input_params.multi_block_tables.empty() ||
      input.input_params.meta.requires_host_restore ||
      !input.input_params.block_copy.swap_blocks.empty() ||
      (input.input_params.block_copy.src_block_indices.defined() &&
       input.input_params.block_copy.src_block_indices.numel() > 0) ||
      !input.transfer_kv_infos.empty() || !input.json_object_states.empty() ||
      !input.json_object_state_snapshots.empty() ||
      input.input_params.embedding.embedding_ids.empty() ||
      input.input_params.embedding.request_ids.size() !=
          input.input_params.embedding.embedding_ids.size()) {
    return false;
  }
  return true;
}

bool DFlash2WorkerImpl::prepare_draft_prelaunch(const ForwardInput& input) {
  if (!can_stage_prelaunch_input(input)) {
    return false;
  }
  // A matching pending draft can supply the following draft's template too.
  // Its full lookahead block mapping was checked before consumption.
  if (prepared_prelaunch_.has_value()) {
    return true;
  }
  const int32_t width = options_.num_speculative_tokens() + 1;
  const auto rows = specBuilder::make_decode_row_context(input);
  if (rows.model_managed_multiblock) {
    return false;
  }
  // Capacity is checked using only known host geometry, before validation.
  if (!specBuilder::has_decode_lookahead_capacity(
          rows,
          /*num_tokens=*/2 * width,
          options_.block_size(),
          draft_impl_->context_.get_model_args().max_position_embeddings())) {
    return false;
  }
  // Noise is independent of the accepted prefix. Schedule any TP consensus
  // while preparing the template, before rejection sampling is submitted.
  c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
  prepared_noise_ = sample_gumbel_noise(
      rows.num_sequences,
      width - 1,
      draft_impl_->context_.get_model_args().dflash2_selector_top_k(),
      input.sampling_params,
      draft_impl_->device());
  if (!input.sampling_params.all_greedy_sample &&
      sampling_process_group_ != nullptr &&
      sampling_process_group_->world_size() > 1) {
    sampling_process_group_->broadcast(prepared_noise_, /*root_rank=*/0);
  }
  // Do not convert sampling control tensors from CPU after target launch.
  prepared_sampling_ =
      input.sampling_params.to(device_.unwrap(), torch::kFloat32);
  ForwardInput future = input;
  future.positions_host = input.positions_host.clone();
  future.positions_host.add_(width);
  future.token_ids_host = input.token_ids_host.clone();
  for (int32_t& length : future.input_params.attention.host.kv_seq_lens) {
    length += width;
  }
  ForwardInput query;
  prepare_query_inputs(future, query);
  query.input_params.is_dflash_proposal = true;
  query.skip_sampling_for_logits_only = true;
  query.return_selected_hidden = true;
  prepared_prelaunch_.emplace();
  // The query builder has allocated fresh metadata buffers on prepare_stream_.
  // It reads no in-flight draft outputs and performs no KV swaps/restores.
  // Only its explicit input-ready event is needed; waiting for the entire
  // compute stream here would serialize staging behind the pending draft.
  draft_impl_->prepare_work_before_execute_on_stream(
      query,
      *prepared_prelaunch_,
      *prepare_stream_,
      /*record_ready_event=*/true,
      /*wait_for_compute=*/false);
  prepared_base_positions_ =
      prepared_prelaunch_->positions.view({rows.num_sequences, width})
          .select(1, 0) -
      width;
  if (prelaunch_graph_enabled_) {
    const torch::Tensor table =
        prepared_prelaunch_->input_params.attention.device.block_tables;
    const int64_t cols = next_pow2_block_table_cols(table.size(1));
    torch::Tensor padded = torch::zeros({table.size(0), cols}, table.options());
    padded.slice(1, 0, table.size(1)).copy_(table);
    prepared_prelaunch_->input_params.attention.device.block_tables =
        std::move(padded);
  }
  // Covers the final staging operations as well as the earlier template.
  prepared_prelaunch_->metadata_ready_event = prepare_stream_->record_event();
  return true;
}

void DFlash2WorkerImpl::launch_draft_prelaunch(
    const ForwardInput& input,
    const SampleOutput& output,
    const torch::Tensor& accepted_host) {
  CHECK(prepared_prelaunch_.has_value());
  CHECK(!pending_draft_.has_value());
  c10::StreamGuard guard = compute_stream_->set_stream_guard();
  ForwardInput query = std::move(*prepared_prelaunch_);
  prepared_prelaunch_.reset();
  CHECK(compute_stream_->wait_event(query.metadata_ready_event));
  torch::Tensor noise = std::move(prepared_noise_);
  torch::Tensor base_positions = std::move(prepared_base_positions_);
  auto staged_query = std::make_shared<ForwardInput>(query);
  PendingDraft pending;
  pending.accepted_host = accepted_host;
  pending.retained_embeddings = output.embeddings;
  ForwardInput device_input = input;
  device_input.sampling_params = std::move(prepared_sampling_);
  pending.identity = device_input;
  if (prelaunch_graph_enabled_) {
    pending.block = replay_prelaunch_graph(
        device_input, std::move(query), output, noise, base_positions);
  } else {
    pending.block = execute_prelaunch_body(device_input,
                                           query,
                                           output.next_tokens,
                                           output.embeddings,
                                           base_positions,
                                           noise,
                                           *compute_stream_);
  }
  pending.block.retained_inputs.emplace_back(std::move(staged_query));
  pending.block.retained_tensors = {
      noise, base_positions, output.next_tokens, output.embeddings};
  pending.ready = compute_stream_->record_event();
  CHECK(pending.ready != nullptr);
  pending_draft_ = std::move(pending);
}

DFlashWorkerImpl::DraftBlock DFlash2WorkerImpl::execute_prelaunch_body(
    const ForwardInput& input,
    ForwardInput& query,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& embeddings,
    const torch::Tensor& base_positions,
    const torch::Tensor& noise,
    Stream& stream) {
  c10::StreamGuard stream_guard = stream.set_stream_guard();
  const int64_t batch = accepted_tokens.size(0);
  const int64_t width = accepted_tokens.size(1);
  dflash_async::NextDraftInputs next = dflash_async::prepare_next_draft(
      accepted_tokens,
      base_positions,
      query.input_params.attention.device.block_tables,
      mask_token_id_,
      options_.block_size());
  ModelOutput written =
      draft_impl_->write_context_kv(embeddings.reshape({batch * width, -1}),
                                    next.context_positions,
                                    next.context_slots,
                                    query.input_params);
  CHECK(written.hidden_states.defined());
  query.token_ids.copy_(next.query_tokens);
  query.positions.copy_(next.query_positions);
  query.input_params.attention.device.new_cache_slots.copy_(next.query_slots);
  query.input_params.attention.device.kv_seq_lens.copy_(next.kv_lengths);
  query.input_params.attn_metadata.reset();
  query.metadata_ready_event.reset();
  query.device_tensors_ready = true;
  return execute_draft_query(
      input, query, next.anchor_tokens, noise, /*prepared=*/true, &stream);
}

DFlashWorkerImpl::DraftBlock DFlash2WorkerImpl::replay_prelaunch_graph(
    const ForwardInput& input,
    ForwardInput query,
    const SampleOutput& output,
    const torch::Tensor& noise,
    const torch::Tensor& base_positions) {
#if defined(USE_NPU)
  torch::InferenceMode inference_guard;
  const int64_t batch = output.next_tokens.size(0);
  const torch::Tensor source_table =
      query.input_params.attention.device.block_tables;
  const int64_t source_cols = source_table.size(1);
  const int64_t table_cols = next_pow2_block_table_cols(source_cols);
  const uint64_t mode = (input.sampling_params.all_greedy_sample ? 1 : 0) |
                        (input.sampling_params.all_random_sample ? 2 : 0) |
                        (input.sampling_params.temperatures.defined() ? 4 : 0);
  const uint64_t key = (static_cast<uint64_t>(batch) << 32) |
                       (static_cast<uint64_t>(table_cols) << 4) | mode;
  auto iterator = draft_graphs_.find(key);
  const bool first_capture = iterator == draft_graphs_.end();
  if (first_capture) {
    if (draft_graph_resources_ == nullptr) {
      draft_graph_resources_ = std::make_unique<DraftGraphResources>();
      draft_graph_resources_->capture_stream = device_.get_stream_from_pool();
    }
    auto entry = std::make_unique<DraftGraph>();
    entry->input = input;
    entry->input.sampling_params.do_sample =
        input.sampling_params.do_sample.clone();
    if (input.sampling_params.temperatures.defined()) {
      entry->input.sampling_params.temperatures =
          input.sampling_params.temperatures.clone();
    }
    entry->query = query;
    entry->query.input_params.attention.device.block_tables =
        torch::zeros({batch, table_cols}, source_table.options());
    entry->accepted_tokens = torch::empty_like(output.next_tokens);
    entry->embeddings = torch::empty_like(output.embeddings);
    entry->base_positions = torch::empty({batch}, query.positions.options());
    entry->noise = torch::empty_like(noise);
    iterator = draft_graphs_.emplace(key, std::move(entry)).first;
  }
  DraftGraph& entry = *iterator->second;
  entry.accepted_tokens.copy_(output.next_tokens, /*non_blocking=*/true);
  entry.embeddings.copy_(output.embeddings, /*non_blocking=*/true);
  entry.base_positions.copy_(base_positions, /*non_blocking=*/true);
  entry.noise.copy_(noise, /*non_blocking=*/true);
  entry.query.input_params.attention.device.block_tables.copy_(
      source_table, /*non_blocking=*/true);
  entry.input.sampling_params.do_sample.copy_(input.sampling_params.do_sample);
  if (input.sampling_params.temperatures.defined()) {
    entry.input.sampling_params.temperatures.copy_(
        input.sampling_params.temperatures);
  }
  if (first_capture) {
    auto& capture_lock =
        npu::DeviceCaptureLock::get_instance().get_lock(device_.index());
    std::lock_guard<std::mutex> lock(capture_lock);
    Stream& capture_stream = *draft_graph_resources_->capture_stream;
    capture_stream.wait_stream(*compute_stream_);
    c10::StreamGuard capture_guard = capture_stream.set_stream_guard();
    // Only the first use of a bucket warms/captures. Warmup writes the same
    // accepted context/proposal positions as replay and is idempotent.
    for (int32_t step = 0; step < 2; ++step) {
      entry.output = execute_prelaunch_body(entry.input,
                                            entry.query,
                                            entry.accepted_tokens,
                                            entry.embeddings,
                                            entry.base_positions,
                                            entry.noise,
                                            capture_stream);
    }
    CHECK_EQ(capture_stream.synchronize(), 0);
    // Keep verifier outputs outside the shared pool: an earlier graph's
    // temporary storage can alias a later graph's captured output. Replays
    // are serialized, but target validation retains the proposal tensors.
    torch::Tensor token_ids =
        torch::empty_like(entry.output.proposal.token_ids());
    CHECK(entry.output.proposal.draft_probs().has_value());
    torch::Tensor draft_probs =
        torch::empty_like(*entry.output.proposal.draft_probs());
    // Share only the draft pool and capture stream. The target has its own
    // pool, and every bucket's input buffers were allocated before capture.
    entry.graph.capture_begin(draft_graph_resources_->pool,
                              ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    entry.output = execute_prelaunch_body(entry.input,
                                          entry.query,
                                          entry.accepted_tokens,
                                          entry.embeddings,
                                          entry.base_positions,
                                          entry.noise,
                                          capture_stream);
    token_ids.copy_(entry.output.proposal.token_ids());
    draft_probs.copy_(*entry.output.proposal.draft_probs());
    entry.output.proposal =
        DraftProposal(std::move(token_ids), std::move(draft_probs));
    entry.graph.capture_end();
  }
  entry.graph.replay();
  return entry.output;
#else
  LOG(FATAL) << "DFlash2 prelaunch graph requires NPU";
  return {};
#endif
}

bool DFlash2WorkerImpl::pending_draft_matches(const ForwardInput& input) const {
  const PendingDraft& pending = *pending_draft_;
  const auto& old = pending.identity.input_params;
  const auto& current = input.input_params;
  if (!can_stage_prelaunch_input(input) ||
      old.embedding.embedding_ids != current.embedding.embedding_ids ||
      old.embedding.request_ids != current.embedding.request_ids ||
      old.embedding.linear_state_ids != current.embedding.linear_state_ids ||
      old.parallel.dp_global_batch_generations !=
          current.parallel.dp_global_batch_generations ||
      old.parallel.dp_global_token_nums !=
          current.parallel.dp_global_token_nums ||
      pending.identity.sampling_params.all_greedy_sample !=
          input.sampling_params.all_greedy_sample) {
    return false;
  }
  const auto previous = specBuilder::make_decode_row_context(pending.identity);
  const auto next = specBuilder::make_decode_row_context(input);
  const int64_t width = pending.accepted_host.size(1);
  const int64_t* tokens = pending.accepted_host.const_data_ptr<int64_t>();
  for (int32_t seq = 0; seq < next.num_sequences; ++seq) {
    int64_t count = 0;
    while (count < width && tokens[seq * width + count] >= 0) {
      ++count;
    }
    if (count == 0 || next.positions[seq] != previous.positions[seq] + count ||
        next.token_ids[seq] != tokens[seq * width + count - 1]) {
      return false;
    }
    const int64_t lookahead =
        pending.next_template.has_value() ? 2 * width : width;
    const int64_t pages =
        (next.positions[seq] + lookahead + options_.block_size() - 1) /
        options_.block_size();
    if (pages > next.block_table_stride ||
        pages > previous.block_table_stride) {
      return false;
    }
    for (int64_t page = 0; page < pages; ++page) {
      if (next.block_tables[seq * next.block_table_stride + page] !=
          previous.block_tables[seq * previous.block_table_stride + page]) {
        return false;
      }
    }
  }
  return true;
}

void DFlash2WorkerImpl::finish_draft_prelaunch(const ForwardInput& input) {
  if (!pending_draft_.has_value()) {
    return;
  }
  // Keep next-target preparation overlapped with the running draft. Only
  // after that preparation, fence all context/query KV writes before this
  // rank returns accepted tokens. The scheduler may end any row (EOS, length,
  // stop sequence or cancellation) and immediately reuse its blocks. Retaining
  // tensors or waiting when a stale draft is consumed does not pin those
  // blocks.
  prepare_next_target(input);
  CHECK(pending_draft_->ready->synchronize())
      << "Failed to retire DFlash2 prelaunch KV writes before publishing "
         "tokens";
}

void DFlash2WorkerImpl::prepare_next_target(const ForwardInput& input) {
  if (!pending_draft_.has_value() || !prelaunch_graph_enabled_ ||
      input.sampling_params.unique_token_ids.defined() ||
      input.sampling_params.filter_mask.defined() ||
      input.sampling_params.filter_bitmask.defined()) {
    return;
  }
  PendingDraft& pending = *pending_draft_;
  CHECK(!pending.target_input.has_value());
  // run_validate has waited only for rejection sampling and its pinned D2H.
  // The prelaunched draft is still queued/running on compute_stream_. Build
  // the next verify metadata on prepare_stream_ using the exact host prefix.
  const auto rows = specBuilder::make_decode_row_context(input);
  const int64_t width = pending.accepted_host.size(1);
  const int64_t* accepted = pending.accepted_host.const_data_ptr<int64_t>();
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  std::vector<int32_t> kv_lengths;
  tokens.reserve(rows.num_sequences);
  positions.reserve(rows.num_sequences);
  kv_lengths.reserve(rows.num_sequences);
  for (int32_t seq = 0; seq < rows.num_sequences; ++seq) {
    int64_t count = 0;
    while (count < width && accepted[seq * width + count] >= 0) {
      ++count;
    }
    CHECK_GT(count, 0);
    tokens.emplace_back(
        static_cast<int32_t>(accepted[seq * width + count - 1]));
    const int32_t position = rows.positions[seq] + static_cast<int32_t>(count);
    positions.emplace_back(position);
    kv_lengths.emplace_back(position + 1);
  }
  ForwardInput future = input;
  future.token_ids_host = specBuilder::make_cpu_int_tensor(tokens);
  future.positions_host = specBuilder::make_cpu_int_tensor(positions);
  future.input_params.attention.host.kv_seq_lens = std::move(kv_lengths);
  future.device_tensors_ready = false;
  future.input_host_buffer_has_layout = false;
  future.metadata_ready_event.reset();
  future.input_params.attn_metadata.reset();
  pending.target_input.emplace();
  prepare_validate_inputs(future, *pending.target_input);
  // Finish the target worker's input preparation and the C++ -> Python
  // attention metadata here too. In particular, materializing the recurrent
  // validity mask on compute_stream_ would synchronize it behind the draft.
  ForwardInput processed_target;
  impl_->prepare_work_before_execute_on_stream(*pending.target_input,
                                               processed_target,
                                               *prepare_stream_,
                                               /*record_ready_event=*/false,
                                               /*wait_for_compute=*/false);
  {
    c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
    processed_target.input_params.attn_metadata =
        std::make_shared<layer::AttentionMetadata>(
            layer::AttentionMetadataBuilder::build(
                processed_target.input_params,
                impl_->context_.get_model_args().enable_mla(),
                std::nullopt,
                device_.unwrap()));
    processed_target.metadata_ready_event = prepare_stream_->record_event();
    CHECK(processed_target.metadata_ready_event != nullptr);
  }
  pending.target_input = std::move(processed_target);
  if (prepare_draft_prelaunch(future)) {
    pending.next_template = DraftTemplate{std::move(*prepared_prelaunch_),
                                          std::move(prepared_sampling_),
                                          std::move(prepared_noise_),
                                          std::move(prepared_base_positions_)};
    prepared_prelaunch_.reset();
  }
}

void DFlash2WorkerImpl::discard_draft_prelaunch() {
  if (pending_draft_.has_value()) {
    CHECK(pending_draft_->ready->synchronize())
        << "Failed to drain stale DFlash2 prelaunch";
    if (pending_draft_->target_input.has_value()) {
      CHECK(pending_draft_->target_input->metadata_ready_event->synchronize())
          << "Failed to drain stale DFlash2 target preparation";
    }
    if (pending_draft_->next_template.has_value()) {
      CHECK(pending_draft_->next_template->query.metadata_ready_event
                ->synchronize())
          << "Failed to drain stale DFlash2 draft template";
    }
    pending_draft_.reset();
  }
  prepared_prelaunch_.reset();
  prepared_noise_ = torch::Tensor();
  prepared_base_positions_ = torch::Tensor();
}

DFlash2WorkerImpl::BlockSampleOutput DFlash2WorkerImpl::sample_path(
    const DFlash2CandidateOutput& candidates,
    const SamplingParameters& sampling_params,
    const torch::Tensor& gumbel_noise,
    int64_t vocab_size) const {
  CHECK_EQ(candidates.candidate_ids.dim(), 3);
  CHECK_EQ(candidates.edge_logits.dim(), 4);
  const int64_t batch_size = candidates.candidate_ids.size(0);
  const int64_t num_steps = candidates.candidate_ids.size(1);
  const int64_t top_k = candidates.candidate_ids.size(2);
  CHECK_EQ(candidates.edge_logits.sizes(),
           torch::IntArrayRef({batch_size, num_steps, top_k, top_k}));
  CHECK_EQ(gumbel_noise.sizes(),
           torch::IntArrayRef({batch_size, num_steps, top_k}));
  const torch::Device device = candidates.edge_logits.device();
  const torch::TensorOptions float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  // Pre-compute log_softmax(edge_logits / temperature) for every step at
  // once.  The previous implementation re-scaled and re-normalized the
  // gathered row inside each step through the generic sampler, which lowered
  // to an AICPU multinomial plus a per-step consensus broadcast and left the
  // compute stream idle between steps.  Gumbel-max over the same
  // log-probabilities draws from exactly the same distribution:
  //   argmax(log_softmax(logits / T) + g) ~ categorical(softmax(logits / T))
  // and greedy rows (zeroed noise) collapse to plain argmax.  Target-side
  // truncation, penalties, and grammar constraints are applied by
  // verification and must not be applied a second time to the selector's
  // top-k candidate distribution.
  torch::Tensor edge_log_probs = candidates.edge_logits.to(torch::kFloat32);
  if (sampling_params.temperatures.defined()) {
    apply_selector_temperatures(
        edge_log_probs, sampling_params.temperatures, batch_size);
  }
  edge_log_probs = torch::log_softmax(edge_log_probs, /*dim=*/-1);

  torch::Tensor token_ids =
      torch::empty({batch_size, num_steps}, candidates.candidate_ids.options());
  torch::Tensor candidate_probs =
      torch::empty({batch_size, num_steps, top_k}, float_options);
  torch::Tensor previous_indices =
      torch::zeros({batch_size}, candidates.candidate_ids.options());
  torch::Tensor batch_offsets =
      torch::arange(batch_size, candidates.candidate_ids.options()) *
      (num_steps * top_k);
  // Flatten all batch/step/predecessor rows once. Selecting a step before
  // reshaping would copy a noncontiguous [B, K, K] slice when B > 1.
  torch::Tensor edge_rows = edge_log_probs.view({-1, top_k});

  using ISlice = torch::indexing::Slice;
  for (int64_t step = 0; step < num_steps; ++step) {
    // NPU gather on the middle dimension of a 3-D tensor transposes both
    // inputs and its output. A complete-row lookup preserves the same values
    // and ordering without those per-step layout conversions.
    torch::Tensor row_indices =
        previous_indices + (batch_offsets + step * top_k);
    torch::Tensor row_log_probs =
        edge_rows.index_select(/*dim=*/0, row_indices);
    torch::Tensor sampled_indices = gumbel_argmax(
        row_log_probs, gumbel_noise.select(/*dim=*/1, /*index=*/step));

    torch::Tensor step_candidates =
        candidates.candidate_ids.select(/*dim=*/1, /*index=*/step);
    torch::Tensor sampled_tokens =
        step_candidates.gather(/*dim=*/1, sampled_indices.view({-1, 1}))
            .view({-1});
    // Keep the per-step proposal distribution semantics identical to the
    // previous sampler path: random rows expose the full softmax over the
    // top-k candidates, deterministic rows expose a one-hot at the argmax,
    // and mixed batches select per row via do_sample. The exp() lives inside
    // the consuming branches so greedy-only batches skip it.
    torch::Tensor step_probs;
    if (sampling_params.all_random_sample) {
      step_probs = row_log_probs.exp();
    } else {
      torch::Tensor greedy_probs =
          torch::zeros({batch_size, top_k}, float_options);
      greedy_probs.scatter_(/*dim=*/1,
                            sampled_indices.view({-1, 1}),
                            /*value=*/1.0);
      if (sampling_params.all_greedy_sample) {
        step_probs = greedy_probs;
      } else {
        step_probs =
            torch::where(sampling_params.do_sample.view({batch_size, 1}),
                         row_log_probs.exp(),
                         greedy_probs);
      }
    }
    token_ids.index_put_({ISlice(), step}, sampled_tokens);
    candidate_probs.index_put_({ISlice(), step, ISlice()}, step_probs);
    previous_indices = sampled_indices;
  }

  torch::Tensor dense_probs =
      torch::zeros({batch_size, num_steps, vocab_size}, float_options);
  dense_probs.scatter_(
      /*dim=*/-1, candidates.candidate_ids, candidate_probs);
  return {.token_ids = std::move(token_ids),
          .dense_probs = std::move(dense_probs)};
}

}  // namespace xllm
