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

#include "llm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache_transfer/kv_transfer_completion.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_CUDA) || defined(USE_ILU) || defined(USE_MUSA)
#include "layers/cuda/flashinfer_workspace.h"
#endif
#include "models/model_registry.h"
#include "runtime/params_utils.h"
#include "runtime/speculative_worker_utils.h"
#include "util/env_var.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {

namespace {

void wait_input_ready_events(const ForwardInput& input, const Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait ForwardInput metadata ready event";
}

StreamEventPtr record_current_stream_event(const Device& device) {
  std::unique_ptr<Stream> stream = device.current_stream();
  return stream->record_event_or_sync();
}

}  // namespace

LLMWorkerImpl::LLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  device_.set_device();
#if defined(USE_CUDA) || defined(USE_MUSA)
  const auto& model_config = ModelConfig::get_instance();
  if (!ModelConfig::is_python_model_impl(model_config.model_impl())) {
    threadpool_.schedule([this]() mutable {
      // initialize flashinfer workspace
      ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
          device_);
    });
  }
#endif
}

bool LLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";
  const auto& model_config = ModelConfig::get_instance();
#if defined(USE_MUSA)
  static const bool use_pool_compute_stream = util::get_bool_env(
      "XLLM_MUSA_POOL_COMPUTE_STREAM", /*default_value=*/true);
  const bool is_qwen3 = context.get_model_args().model_type() == "qwen3";
  if (use_pool_compute_stream && !is_qwen3) {
    compute_stream_ = device_.get_stream_from_pool();
  } else if (use_pool_compute_stream) {
    LOG(WARNING) << "MUSA pool compute streams are not validated for Qwen3 "
                    "attention; using the default compute stream.";
  }

  const auto& beam_search_config = BeamSearchConfig::get_instance();
  CHECK(!has_linear_attention_layers(context.get_model_args()) ||
        (!beam_search_config.enable_beam_search_kernel() &&
         beam_search_config.beam_width() <= 1))
      << "MUSA beam search is not supported for models with linear-attention "
         "layers.";
#endif

#if defined(USE_CUDA) || defined(USE_MUSA)
  // Ensure FlashinferWorkspace is initialized on the calling thread before
  // constructing model layers. When called synchronously from
  // SpeculativeWorkerImpl (e.g. MTP target/draft setup), init_model runs on
  // the MTP worker's thread (T_MTP) rather than on the LLMWorkerImpl's own
  // threadpool thread (T_worker) where the scheduled initialize() runs.
  // FlashinferWorkspace is thread_local, so T_MTP's instance must be
  // explicitly initialized here; otherwise FlashInferAttentionImpl captures
  // an undefined int_workspace_buffer_ and crashes at prefill time.
  //
  // Skip when model_impl=python: Python executor uses flashinfer's Python API
  // directly; initializing the C++ workspace would conflict with Python-side
  // TVM-FFI type registration.
  if (!ModelConfig::is_python_model_impl(model_config.model_impl())) {
    auto& ws = ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance();
    if (!ws.get_int_workspace_buffer().defined()) {
      ws.initialize(device_);
    }
  }
#endif

  // Try to create a causal LM model
  context.set_model_impl(model_config.model_impl());
  model_ = create_llm_model(context);

  // Dont find model in causal models
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);
  model_executor_->set_dp_process_group(
      context.get_parallel_args().dp_local_process_group_);

  // MTP draft workers execute a separate one-layer model and must not consume
  // the target model's EPLB prepare/update commands. Only the target reports
  // expert load and participates in expert-weight transfer.
  if (::xllm::EPLBConfig::get_instance().enable_eplb() && !is_spec_draft_) {
    eplb_executor_ = std::make_unique<EplbExecutor>(*model_, device_);
  }

  if (::xllm::BeamSearchConfig::get_instance().enable_beam_search_kernel()) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }
  return true;
}

#if defined(USE_NPU)
bool LLMWorkerImpl::prepare_static_mtp_graph_tasks(
    const SpecVerifyGraphTaskSignal& signal,
    const Stream& signal_stream) {
  if (model_executor_ == nullptr) {
    return false;
  }
  return model_executor_->prepare_static_mtp_graph_tasks(signal, signal_stream);
}
#endif

std::optional<ForwardOutput> LLMWorkerImpl::execute_no_sync_on_stream(
    const ForwardInput& input,
    Stream& compute_stream) {
  return execute_no_sync_on_stream(
      input, compute_stream, /*record_ready_event=*/true);
}

std::optional<ForwardOutput> LLMWorkerImpl::execute_no_sync_on_stream(
    const ForwardInput& input,
    Stream& compute_stream,
    bool record_ready_event) {
  const ForwardSyncPolicy sync_policy = ForwardSyncPolicy::NO_SYNC;
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  if (::xllm::LoadConfig::get_instance().enable_manual_loader()) {
#if defined(USE_NPU)
    if (!enable_schedule_overlap() && options_.backend() == "llm") {
      aclrtStream current_acl_stream =
          c10_npu::getCurrentNPUStream(device_.index()).stream();
      atb::Context* atb_context =
          const_cast<atb::Context*>(context_.get_atb_context());
      atb_context->SetExecuteStream(current_acl_stream);
      wait_input_ready_events(input, compute_stream);
      return step_internal(input, sync_policy, record_ready_event);
    } else {
      SET_ATB_EXECUTE_STREAM((&compute_stream), device_, context_);
      wait_input_ready_events(input, compute_stream);
      return step_internal(input, sync_policy, record_ready_event);
    }
#else
    wait_input_ready_events(input, compute_stream);
    return step_internal(input, sync_policy, record_ready_event);
#endif
  }
  wait_input_ready_events(input, compute_stream);
  return step_internal(input, sync_policy, record_ready_event);
}

std::optional<ForwardOutput> LLMWorkerImpl::step(const ForwardInput& input) {
#if defined(USE_NPU)
  if (::xllm::LoadConfig::get_instance().enable_manual_loader()) {
    if (!enable_schedule_overlap() && options_.backend() == "llm") {
      aclrtStream current_stream =
          c10_npu::getCurrentNPUStream(device_.index()).stream();
      atb::Context* atb_context =
          const_cast<atb::Context*>(context_.get_atb_context());
      atb_context->SetExecuteStream(current_stream);
    } else {
      SET_ATB_EXECUTE_STREAM(compute_stream_, device_, context_);
      wait_input_ready_events(input, *compute_stream_);
      return step_internal(input, ForwardSyncPolicy::LEGACY);
    }
  }
#endif

  std::unique_ptr<Stream> stream = device_.current_stream();
  wait_input_ready_events(input, *stream);
  return step_internal(input, ForwardSyncPolicy::LEGACY);
}

std::optional<ForwardOutput> LLMWorkerImpl::step_for_schedule_overlap(
    ForwardInput& input) {
  if (has_request_state_cache()) {
    c10::StreamGuard state_guard = compute_stream_->set_stream_guard();
    prepare_linear_state_cache(input.input_params);
  }
  return execute_no_sync_on_stream(input, *compute_stream_);
}

ForwardInput
LLMWorkerImpl::update_input_by_last_step_output_for_schedule_overlap(
    ForwardInput& input) {
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  CHECK(compute_stream_->wait_event(last_step_output_.ready_event))
      << "failed to wait last step output ready event";
  // Order replace_token's in-place token_ids rewrite after the async H2D that
  // fills token_ids on prepare_stream_.
  wait_input_ready_events(input, *compute_stream_);
  return WorkerImpl::update_input_by_last_step_output_for_schedule_overlap(
      input);
}

std::optional<ForwardOutput> LLMWorkerImpl::step_internal(
    const ForwardInput& input,
    ForwardSyncPolicy sync_policy,
    bool record_ready_event) {
  MULTI_MODEL_STEP_LOCK(::xllm::KVCacheConfig::get_instance().enable_xtensor());

  Timer timer;
  auto& sampling_params = input.sampling_params;

  KVTransferCompletion kv_transfers;
  const std::vector<std::string> canonical_request_ids =
      canonical_transfer_request_ids(input.transfer_kv_infos);
  std::vector<std::string> failed_request_ids;
  bool kv_push_waited = false;

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
#if defined(USE_NPU)
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
#elif defined(USE_MLU)
    std::shared_ptr<MLULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<MLULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
#elif defined(USE_DCU)
    std::shared_ptr<DCULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<DCULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
#endif
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
    const_cast<ModelInputParams*>(&(input.input_params))
        ->parallel.layer_synchronizer = layer_synchronizer;

    kv_transfers.add(
        kv_cache_transfer_->push_kv_blocks_async(input.transfer_kv_infos,
                                                 context_.get_parallel_args(),
                                                 layer_synchronizer,
                                                 is_spec_draft_),
        canonical_request_ids);
#endif
  }
  auto wait_kv_push = [&]() {
    if (kv_push_waited) {
      return;
    }
    failed_request_ids =
        finalize_kv_push_failures(kv_transfers,
                                  input.transfer_kv_infos,
                                  options_.kv_cache_transfer_mode(),
                                  options_.instance_role(),
                                  context_.get_parallel_args(),
                                  device_);
    kv_push_waited = true;
  };
  if (eplb_executor_ != nullptr) {
    eplb_executor_->start_eplb_step(input.input_params.expert.eplb_info);
  }

  // call model executor forward to get hidden states
  auto model_output = model_executor_->forward(
      input.token_ids, input.positions, kv_caches_, input.input_params);
  if (eplb_executor_ != nullptr) {
    eplb_executor_->finish_eplb_step();
  }
  if (!model_output.hidden_states.defined()) {
    wait_kv_push();
    if (!failed_request_ids.empty()) {
      return make_failed_output(std::move(failed_request_ids));
    }
    return std::nullopt;
  }

  torch::Tensor logits;
  torch::Tensor lm_head_selected_token_idxes;
  torch::Tensor selected_hidden;
  if (sampling_params.selected_token_idxes.defined()) {
    torch::Tensor selected_token_idxes = choose_lm_head_selected_token_idxes(
        sampling_params.selected_token_idxes,
        input.input_params,
        context_.get_parallel_args(),
        model_output.hidden_states.size(0),
        model_output.hidden_states.device());
    if (model_output.hidden_states.defined() &&
        selected_token_idxes.device() != model_output.hidden_states.device()) {
      selected_token_idxes = selected_token_idxes
                                 .to(model_output.hidden_states.device(),
                                     /*non_blocking=*/false)
                                 .contiguous();
    }
    lm_head_selected_token_idxes = selected_token_idxes;
    const bool need_selected_hidden =
        input.return_selected_hidden ||
        (options_.cp_size() > 1 && options_.enable_speculative_decode());
    if (need_selected_hidden) {
      // Emit both selected hidden and logits from a single lm_head pass so the
      // ConfidenceHead and CP speculative paths can consume hidden without a
      // second projection.
      logits = model_->logits(
          model_output.hidden_states, selected_token_idxes, selected_hidden);
      if (!selected_hidden.defined() && model_output.hidden_states.defined()) {
        // ATB lm_head backend does not expose a second output tensor
        // (LmHeadParam::outputHidden is false), so we surface the hidden here.
        // selected_hidden must align row-for-row with `logits`, which is
        // produced in selected_token_idxes order. index_select reproduces that
        // order for any selection. We deliberately do NOT alias the full
        // hidden_states on a numel match: equal row count does not imply the
        // idxes are the identity permutation, and a full-but-reordered
        // selection would silently misalign hidden against logits. The gather
        // is one [num_selected, hidden] copy per decode step (not per layer),
        // negligible next to the forward.
        selected_hidden = model_output.hidden_states.index_select(
            /*dim=*/0, selected_token_idxes.to(torch::kLong));
      }
    } else {
      logits = model_->logits(model_output.hidden_states, selected_token_idxes);
    }
  }

  ForwardOutput output;
  output.mtp_topk_state = std::move(model_output.mtp_topk_state);
  if (eplb_executor_ != nullptr) {
    output.expert_load_data = expert_load_data_;
    output.prepared_token = eplb_executor_->consume_ready_prepare_token();
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    MULTI_MODEL_STEP_UNLOCK();
    if (sync_policy == ForwardSyncPolicy::NO_SYNC) {
      wait_kv_push();
      return make_non_driver_forward_output(ForwardOutput{},
                                            std::move(failed_request_ids),
                                            /*eplb_enabled=*/false);
    }
    int ret = device_.synchronize_default_stream();
    CHECK_EQ(ret, 0) << "synchronize_default_stream failed";
    wait_kv_push();
    return make_non_driver_forward_output(std::move(output),
                                          std::move(failed_request_ids),
                                          eplb_executor_ != nullptr);
  }

  // driver prepare model output
  if (sampling_params.selected_token_idxes.defined()) {
    output.logits = logits;
    output.selected_hidden = selected_hidden;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
    if (!input.skip_sampling_for_logits_only) {
      auto sample_output = sampler_->forward(logits, sampling_params);
      // Stochastic sampling under schedule overlap needs a CP-wide
      // consensus: the overlapped decode replacement
      // (update_input_by_last_step_output) resolves every rank's negative
      // placeholders against its own rank-local sample, so all ranks of the
      // plane must derive the same tokens. Greedy argmax over the
      // CP-replicated logits is deterministic, but random_sample draws from
      // the per-process default RNG, which is only seeded identically at
      // init -- nothing keeps the generators in lockstep afterwards -- so
      // unify stochastic batches with a root-0 broadcast, mirroring
      // enable_spec_token_broadcast and DFlash2's noise consensus. The
      // composition lands on the cohort's dp_driver, the worker whose
      // output the engine collects. Requiring overlap is what makes the
      // collective safe: without it only driver ranks sample.
      //
      // The tp axis runs where the backend publishes a tp group; the NPU ATB
      // path builds only the cp group when cp_size > 1 (tp_group_ stays
      // nullptr), so there the cp-axis broadcast is the operative consensus.
      // What must never happen is entering this block with no usable group
      // at all -- that would silently skip every axis -- so fail loudly
      // instead of degrading. (Deliberately NOT the peers' tp && cp
      // conjunction from mtp_worker_impl/dflash_worker_impl: under ATB both
      // process_group_ and tp_group_ are nullptr, and that branch would
      // broadcast on a null sampling group, skipping the cp axis too.)
      if (enable_schedule_overlap() && options_.cp_size() > 1 &&
          !options_.enable_speculative_decode() &&
          !sampling_params.all_greedy_sample) {
        CHECK(parallel_args_.tp_group_ != nullptr ||
              parallel_args_.cp_group_ != nullptr)
            << "cp_size=" << options_.cp_size()
            << " but no tp/cp broadcast group is available for the "
               "schedule-overlap token consensus";
        speculative_worker::broadcast_tokens_in_group(sample_output.next_tokens,
                                                      parallel_args_.tp_group_);
        // Only broadcast the CP axis when it is a genuinely orthogonal
        // group: backends that publish a single process group alias
        // cp_group_ to tp_group_, and a second broadcast over the same
        // group would be a redundant collective per decode step. Mirrors
        // broadcast_spec_tokens' guard in mtp_worker_impl.
        if (parallel_args_.cp_group_ != parallel_args_.tp_group_) {
          speculative_worker::broadcast_tokens_in_group(
              sample_output.next_tokens, parallel_args_.cp_group_);
        }
      }
      output.filter_bitmask_applied_to_logits =
          sampling_params.filter_bitmask.defined();

      // beam search kernel
      BeamSearchOutput beam_search_output;
      if (sampling_params.use_beam_search &&
          sampling_params.acc_logprob.defined() &&
          sampling_params.acc_logprob.numel() > 0) {
        beam_search_output =
            beam_searcher_->forward(sampling_params.acc_logprob,
                                    sample_output.top_tokens,
                                    sample_output.top_logprobs);
      }

      // set sample output to output
      output.sample_output = sample_output;
      // set beam search output to output
      output.beam_search_output = beam_search_output;
    }
  }

  if (options_.enable_speculative_decode()) {
    torch::Tensor embeddings;
    if (model_output.aux_hidden_states.defined()) {
      embeddings = model_output.aux_hidden_states;
    } else {
      embeddings = model_output.hidden_states;
    }
    if (!input.input_params.meta.batch_forward_type.is_decode() &&
        !is_spec_draft_) {
      // Target prefill: keep full embeddings (global-real under model-side CP).
      output.sample_output.embeddings = embeddings;
    } else if (sampling_params.selected_token_idxes.defined()) {
      if (options_.cp_size() > 1) {
        CHECK(selected_hidden.defined())
            << "selected_hidden must be defined when "
               "selected_token_idxes is defined.";
        output.sample_output.embeddings = selected_hidden;
      } else {
        output.sample_output.embeddings = embeddings.index_select(
            /*dim=*/0, lm_head_selected_token_idxes);
      }
    }
  }

  MULTI_MODEL_STEP_UNLOCK();
  bool should_sync_default_stream = true;
#if defined(USE_NPU)
  should_sync_default_stream =
      !can_skip_npu_graph_decode_sync(input.input_params);
#endif
  if (sync_policy == ForwardSyncPolicy::NO_SYNC) {
    wait_kv_push();
    output.failed_request_ids = std::move(failed_request_ids);
    output.retained_inputs.emplace_back(std::make_shared<ForwardInput>(input));
    if (enable_schedule_overlap() && record_ready_event) {
      output.ready_event = record_current_stream_event(device_);
    }
    return output;
  }
  if (should_sync_default_stream) {
    int ret = device_.synchronize_default_stream();
    CHECK_EQ(ret, 0) << "synchronize_default_stream failed";
  }

  wait_kv_push();
  output.failed_request_ids = std::move(failed_request_ids);

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  if (should_sync_default_stream) {
    DeviceMonitor::get_instance().update_active_activation_memory(
        device_.index());
  }

  return output;
}

}  // namespace xllm
