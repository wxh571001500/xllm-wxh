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

#include "suffix_worker_impl.h"

#include <algorithm>
#include <unordered_map>

#include "common/metrics.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/speculative/spec_verify.h"
#include "framework/sampling/sampler.h"
#include "framework/sampling/sampling_params.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {

void append_tokens_with_limit(std::vector<int32_t>& history,
                              std::span<const int32_t> tokens,
                              size_t max_size) {
  history.insert(history.end(), tokens.begin(), tokens.end());
  if (history.size() > max_size) {
    history.erase(history.begin(), history.end() - max_size);
  }
}

void extend_validation_history(const SamplingParameters& original,
                               const std::vector<int32_t>& proposals,
                               int32_t num_sequences,
                               int32_t draft_width,
                               SamplingParameters& expanded) {
  if (!original.unique_token_ids.defined()) {
    return;
  }
  const torch::Device device = expanded.unique_token_ids.device();
  torch::Tensor ids =
      original.unique_token_ids.to(torch::kCPU).to(torch::kLong).contiguous();
  torch::Tensor counts =
      original.unique_token_counts.to(torch::kCPU).to(torch::kInt).contiguous();
  torch::Tensor lengths = original.unique_token_ids_lens.to(torch::kCPU)
                              .to(torch::kInt)
                              .contiguous();
  CHECK_EQ(ids.size(0), num_sequences);
  const int32_t width = draft_width + 1;
  const int64_t capacity = ids.size(1) + draft_width;
  torch::Tensor new_ids =
      torch::zeros({num_sequences * width, capacity}, torch::kLong);
  torch::Tensor new_counts =
      torch::zeros({num_sequences * width, capacity}, torch::kInt);
  torch::Tensor new_lengths =
      torch::zeros({num_sequences * width}, torch::kInt);
  const auto old_ids = ids.accessor<int64_t, 2>();
  const auto old_counts = counts.accessor<int32_t, 2>();
  const auto old_lengths = lengths.accessor<int32_t, 1>();
  auto out_ids = new_ids.accessor<int64_t, 2>();
  auto out_counts = new_counts.accessor<int32_t, 2>();
  auto out_lengths = new_lengths.accessor<int32_t, 1>();
  for (int32_t seq = 0; seq < num_sequences; ++seq) {
    std::unordered_map<int64_t, int32_t> history;
    history.reserve(static_cast<size_t>(capacity));
    for (int32_t i = 0; i < old_lengths[seq]; ++i) {
      history[old_ids[seq][i]] = old_counts[seq][i];
    }
    for (int32_t pos = 0; pos < width; ++pos) {
      const int32_t row = seq * width + pos;
      int32_t column = 0;
      for (const auto& [token, count] : history) {
        out_ids[row][column] = token;
        out_counts[row][column] = count;
        ++column;
      }
      out_lengths[row] = column;
      if (pos < draft_width) {
        ++history[proposals[seq * draft_width + pos]];
      }
    }
  }
  expanded.unique_token_ids = new_ids.to(device);
  expanded.unique_token_counts = new_counts.to(device);
  expanded.unique_token_ids_lens = new_lengths.to(device);
}

// The target may emit one punctuation token that the CTC hypothesis omits
// (e.g. a pause marker); a single-token history gap bridges it. Gap 0
// measured strictly worse acceptance; gaps >= 2 measured no extra gain.
constexpr int32_t kMaxHistoryGap = 1;
// Longest prompt suffix considered when anchoring the hint tail.
constexpr size_t kMaxPromptMatchWidth = 16;

std::vector<int32_t> find_prompt_lookup_draft(std::span<const int32_t> history,
                                              std::span<const int32_t> hint,
                                              int32_t max_tokens) {
  if (history.empty() || hint.empty() || max_tokens <= 0) {
    return {};
  }
  const size_t max_gap =
      std::min<size_t>(static_cast<size_t>(kMaxHistoryGap), history.size() - 1);
  // Exact suffixes take priority. A bounded gap can bridge a punctuation
  // token emitted by the target but absent from the CTC hypothesis.
  for (size_t gap = 0; gap <= max_gap; ++gap) {
    const auto prefix = history.first(history.size() - gap);
    const size_t max_match = std::min(kMaxPromptMatchWidth, prefix.size());
    for (size_t width = max_match; width > 0; --width) {
      for (size_t end = width; end < hint.size(); ++end) {
        if (!std::equal(prefix.end() - width,
                        prefix.end(),
                        hint.begin() + end - width)) {
          continue;
        }
        const size_t count = std::min<size_t>(static_cast<size_t>(max_tokens),
                                              hint.size() - end);
        return std::vector<int32_t>(hint.begin() + end,
                                    hint.begin() + end + count);
      }
    }
  }
  return {};
}

std::string summarize_int32_span(std::span<const int32_t> values,
                                 size_t limit = 8) {
  std::string out = "[";
  const size_t n = std::min(values.size(), limit);
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      out += ",";
    }
    out += std::to_string(values[i]);
  }
  if (values.size() > n) {
    out += ",...";
  }
  out += "]";
  return out;
}

}  // namespace

namespace {
runtime::Options SuffixTargetOptions(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false);
  return opts;
}
}  // namespace

SuffixWorkerImpl::SuffixWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options,
                                   WorkerType worker_type)
    : SpeculativeWorkerImpl(parallel_args,
                            device,
                            options,
                            SuffixTargetOptions(options),
                            worker_type) {
  suffix_cache_ = std::make_unique<SuffixDecodingCache>(
      options_.speculative_suffix_cache_max_depth(),
      options_.speculative_suffix_max_cached_requests());
  vlm_target_ = worker_type == WorkerType::VLM;
}

void SuffixWorkerImpl::store_prompt_hint(const std::string& request_id,
                                         std::vector<int32_t> hint) {
  // Hints are advisory; the bound only stops unbounded growth from requests
  // that finish inside prefill and never reach decode cleanup. Eviction
  // follows insertion order and only degrades acceptance, never correctness.
  const size_t limit =
      static_cast<size_t>(std::max(1, options_.max_seqs_per_batch())) * 2;
  const bool is_new =
      model_prompt_hints_.find(request_id) == model_prompt_hints_.end();
  if (is_new && model_prompt_hints_.size() >= limit) {
    model_prompt_hints_.erase(model_prompt_hints_order_.front());
    model_prompt_hints_order_.pop_front();
  }
  if (is_new) {
    model_prompt_hints_order_.push_back(request_id);
  }
  model_prompt_hints_.insert_or_assign(request_id, std::move(hint));
}

void SuffixWorkerImpl::drop_prompt_hint(const std::string& request_id) {
  if (model_prompt_hints_.erase(request_id) > 0) {
    const auto it = std::find(model_prompt_hints_order_.begin(),
                              model_prompt_hints_order_.end(),
                              request_id);
    if (it != model_prompt_hints_order_.end()) {
      model_prompt_hints_order_.erase(it);
    }
  }
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_empty(
    const ForwardInput& input) {
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    auto output = impl_->step(input);
    output->sample_output.embeddings = torch::Tensor();
    return output;
  } else {
    ForwardInput new_input = input;
    for (auto& it : new_input.input_params.parallel.dp_global_token_nums) {
      it *= options_.num_speculative_tokens() + 1;
    }
    new_input.input_params.expert.eplb_decode_token_mask =
        eplb::expand_decode_token_mask(
            new_input.input_params.expert.eplb_decode_token_mask,
            options_.num_speculative_tokens() + 1);

    auto future = impl_->step_async(new_input);
    ForwardOutput output = std::move(future).get().value();
    output.sample_output.embeddings = torch::Tensor();
    return output;
  }
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_prefill(
    const ForwardInput& input) {
  Timer timer;
  // run the target model to get first token and hidden states
  auto future = impl_->step_async(input);
  ForwardOutput output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  const auto& input_params = input.input_params;
  const int32_t num_sequences = input_params.meta.num_sequences;
  const auto& request_ids = input_params.embedding.request_ids;

  if (suffix_cache_ != nullptr &&
      request_ids.size() == static_cast<size_t>(num_sequences)) {
    const torch::Tensor& token_ids = input.token_ids_host;
    Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                       static_cast<size_t>(token_ids.numel())};

    int32_t start_idx = 0;
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      int32_t q_len = input_params.get_q_seq_len(seq_id);
      Slice<int32_t> seq_tokens =
          tokens_ids_slice.slice(start_idx, start_idx + q_len);
      start_idx += q_len;

      const std::string req_id = request_ids[seq_id];
      if (req_id.empty()) {
        continue;
      }

      if (!suffix_cache_->has_active_request(req_id)) {
        suffix_cache_->start_request(req_id, seq_tokens);
        suffix_recent_tokens_[req_id].clear();
      } else {
        suffix_cache_->add_active_prompt(req_id, seq_tokens);
      }
      append_tokens_with_limit(
          suffix_recent_tokens_[req_id],
          seq_tokens,
          static_cast<size_t>(suffix_cache_->max_tree_depth()));
    }

    std::vector<int32_t> audio_sequences;
    audio_sequences.reserve(static_cast<size_t>(num_sequences));
    for (const auto& data : input_params.multimodal.mm_data.mm_data_vec()) {
      if (!data.hold<MMItemVec>()) {
        continue;
      }
      for (const auto& item : data.items<MMItemVec>()) {
        if (item.type() != MMType::AUDIO) {
          continue;
        }
        audio_sequences.emplace_back(item.state().seq_index());
      }
    }
    if (!audio_sequences.empty()) {
      auto hints = impl_->get_prompt_lookup_hints();
      if (!hints.empty()) {
        CHECK_EQ(hints.size(), audio_sequences.size());
        for (size_t i = 0; i < hints.size(); ++i) {
          const int32_t seq = audio_sequences[i];
          CHECK_GE(seq, 0);
          CHECK_LT(seq, num_sequences);
          store_prompt_hint(request_ids[seq], std::move(hints[i]));
        }
      }
    }

    torch::Tensor next_tokens =
        safe_to(output.sample_output.next_tokens, torch::kCPU);
    if (next_tokens.defined() &&
        next_tokens.numel() == static_cast<int64_t>(num_sequences)) {
      next_tokens = next_tokens.view({-1}).to(torch::kInt);
      Slice<int32_t> next_tokens_slice = {
          next_tokens.data_ptr<int32_t>(),
          static_cast<size_t>(next_tokens.numel())};
      for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
        int32_t token = next_tokens_slice[seq_id];
        if (token < 0) {
          continue;
        }
        const std::string req_id = request_ids[seq_id];
        if (req_id.empty()) {
          continue;
        }
        suffix_cache_->add_active_response(req_id,
                                           std::span<const int32_t>(&token, 1));
        append_tokens_with_limit(
            suffix_recent_tokens_[req_id],
            std::span<const int32_t>(&token, 1),
            static_cast<size_t>(suffix_cache_->max_tree_depth()));
      }
    }
  }

  output.sample_output.embeddings = torch::Tensor();
  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

std::optional<ForwardOutput> SuffixWorkerImpl::step_decode(
    const ForwardInput& input) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const auto& request_ids = input.input_params.embedding.request_ids;

  const bool has_request_ids =
      suffix_cache_ != nullptr &&
      request_ids.size() == static_cast<size_t>(num_sequences);
  if (has_request_ids) {
    std::unordered_set<std::string> current_req_ids;
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      if (!request_ids[seq_id].empty()) {
        current_req_ids.insert(request_ids[seq_id]);
      }
    }

    for (const auto& req_id : suffix_active_decode_req_ids_) {
      if (current_req_ids.find(req_id) == current_req_ids.end()) {
        if (suffix_cache_->has_active_request(req_id)) {
          suffix_cache_->stop_request(req_id);
        }
        suffix_recent_tokens_.erase(req_id);
        drop_prompt_hint(req_id);
      }
    }
    suffix_active_decode_req_ids_ = std::move(current_req_ids);
  }

  const torch::Tensor& input_token_ids = input.token_ids_host;
  Slice<int32_t> input_tokens_slice = {
      input_token_ids.data_ptr<int32_t>(),
      static_cast<size_t>(input_token_ids.numel())};

  Timer timer;

  // The CTC hint path requires greedy verification to reproduce the target's
  // autoregressive output exactly (per-position processor application plus
  // greedy compare). Non-greedy batches degrade to the standard suffix-cache
  // flow with raw-logits greedy verify — the same behavior as the LLM suffix
  // path — instead of failing the request.
  const bool vlm_greedy_verify =
      vlm_target_ && input.sampling_params.all_greedy_sample;

  std::vector<int32_t> draft_tokens_flat;
  draft_tokens_flat.reserve(num_sequences * num_speculative_tokens);
  std::vector<std::string> req_ids(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    int32_t fallback_token = input_tokens_slice[seq_id];
    for (int32_t i = 0; i < num_speculative_tokens; ++i) {
      draft_tokens_flat.emplace_back(fallback_token);
    }

    if (suffix_cache_ == nullptr ||
        request_ids.size() != static_cast<size_t>(num_sequences)) {
      continue;
    }

    const std::string req_id = request_ids[seq_id];
    if (req_id.empty()) {
      continue;
    }
    req_ids[seq_id] = req_id;

    if (!suffix_cache_->has_active_request(req_id)) {
      suffix_cache_->start_request(
          req_id, std::span<const int32_t>(&fallback_token, 1));
      suffix_recent_tokens_[req_id].clear();
      append_tokens_with_limit(
          suffix_recent_tokens_[req_id],
          std::span<const int32_t>(&fallback_token, 1),
          static_cast<size_t>(suffix_cache_->max_tree_depth()));
    }

    auto& history = suffix_recent_tokens_[req_id];
    if (history.empty()) {
      append_tokens_with_limit(
          history,
          std::span<const int32_t>(&fallback_token, 1),
          static_cast<size_t>(suffix_cache_->max_tree_depth()));
    }
    // A stored model hint takes priority over the suffix-cache draft for
    // this request until it finishes; the suffix history bookkeeping above
    // stays maintained either way. Hints are only consulted for greedy
    // batches. Slots the hint cannot fill (short tail at transcription end)
    // are completed from the suffix cache; if both run short, the remaining
    // slots keep the fallback token — the same short-draft semantics as the
    // standard suffix path below. Greedy verify truncates at the first
    // mismatch, so this only wastes slots, never correctness.
    const auto speculate_cache = [&](int32_t max_spec_tokens) {
      return suffix_cache_->speculate(
          req_id,
          std::span<const int32_t>(history.data(), history.size()),
          /*max_spec_tokens=*/max_spec_tokens,
          options_.speculative_suffix_max_spec_factor(),
          options_.speculative_suffix_max_spec_offset(),
          options_.speculative_suffix_min_token_prob(),
          options_.speculative_suffix_use_tree_spec());
    };
    const auto hint = vlm_greedy_verify ? model_prompt_hints_.find(req_id)
                                        : model_prompt_hints_.end();
    if (hint != model_prompt_hints_.end()) {
      const auto proposed = find_prompt_lookup_draft(
          history, hint->second, num_speculative_tokens);
      if (!proposed.empty()) {
        for (size_t i = 0; i < proposed.size(); ++i) {
          draft_tokens_flat[seq_id * num_speculative_tokens + i] = proposed[i];
        }
        const int32_t remaining =
            num_speculative_tokens - static_cast<int32_t>(proposed.size());
        if (remaining > 0) {
          const SuffixDecodingDraft tail = speculate_cache(remaining);
          const int32_t fill =
              std::min<int32_t>(remaining, tail.token_ids.size());
          for (int32_t i = 0; i < fill; ++i) {
            draft_tokens_flat[seq_id * num_speculative_tokens +
                              static_cast<int32_t>(proposed.size()) + i] =
                tail.token_ids[i];
          }
        }
        continue;
      }
    }

    SuffixDecodingDraft draft = speculate_cache(num_speculative_tokens);

    const int32_t fill_count =
        std::min<int32_t>(num_speculative_tokens, draft.token_ids.size());
    for (int32_t i = 0; i < fill_count; ++i) {
      draft_tokens_flat[seq_id * num_speculative_tokens + i] =
          draft.token_ids[i];
    }
  }

  ForwardInput validate_input;
  prepare_validate_inputs(
      input, validate_input, /*repeat_unique_history=*/!vlm_greedy_verify);
  validate_input.skip_sampling_for_logits_only = true;

  auto& validate_token_ids = validate_input.token_ids;
  for (int32_t i = 0; i < num_speculative_tokens; ++i) {
    std::vector<int32_t> draft_col;
    draft_col.reserve(num_sequences);
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      draft_col.emplace_back(
          draft_tokens_flat[seq_id * num_speculative_tokens + i]);
    }
    auto draft_col_tensor =
        torch::tensor(draft_col, validate_token_ids.options());
    auto mask = (validate_token_ids == -1 * (i + 1));
    validate_token_ids.masked_scatter_(mask, draft_col_tensor);
  }

  if (vlm_greedy_verify) {
    extend_validation_history(input.sampling_params,
                              draft_tokens_flat,
                              num_sequences,
                              num_speculative_tokens,
                              validate_input.sampling_params);
  }

  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  timer.reset();
  auto future = impl_->step_async(validate_input);
  ForwardOutput target_output = std::move(future).get().value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  if (vlm_greedy_verify) {
    // The target returned raw logits; apply the original request's
    // processors in place, using the history of each verification position.
    // validate() below samples the processed logits (bonus argmax plus the
    // rejection sampler's greedy compare), so the sampler's own argmax and
    // softmax passes would be discarded work.
    Sampler::apply_logits_processors(target_output.logits,
                                     validate_input.sampling_params);
  }

  torch::Tensor draft_token_ids =
      torch::tensor(draft_tokens_flat,
                    torch::TensorOptions().dtype(torch::kLong))
          .view({num_sequences, num_speculative_tokens})
          .to(target_output.logits.device());

  DraftProposal draft_proposal = DraftProposal(std::move(draft_token_ids));

  timer.reset();
  SampleOutput val_output =
      validate(input.sampling_params, draft_proposal, target_output);
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  if (suffix_cache_ != nullptr &&
      request_ids.size() == static_cast<size_t>(num_sequences)) {
    torch::Tensor accepted_tokens =
        safe_to(val_output.next_tokens, torch::kCPU).to(torch::kInt);
    accepted_tokens = accepted_tokens.view({num_sequences, num_val_tokens});
    Slice<int32_t> accepted_tokens_slice = {
        accepted_tokens.data_ptr<int32_t>(),
        static_cast<size_t>(accepted_tokens.numel())};

    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      const std::string& req_id = req_ids[seq_id];
      if (req_id.empty()) {
        continue;
      }

      std::vector<int32_t> accepted;
      accepted.reserve(num_val_tokens);
      int32_t first_reject_idx = -1;
      std::vector<int32_t> row_tokens;
      row_tokens.reserve(num_val_tokens);
      for (int32_t j = 0; j < num_val_tokens; ++j) {
        int32_t token = accepted_tokens_slice[seq_id * num_val_tokens + j];
        row_tokens.emplace_back(token);
        if (token < 0) {
          if (first_reject_idx < 0) {
            first_reject_idx = j;
          }
          break;
        }
        accepted.emplace_back(token);
      }

      if (seq_id < 8) {
        VLOG(3) << "[spec-validate-output] seq=" << seq_id
                << " req_id=" << req_id << " accepted_len=" << accepted.size()
                << " first_reject_idx=" << first_reject_idx << " accepted="
                << summarize_int32_span(std::span<const int32_t>(
                       accepted.data(), accepted.size()))
                << " row_tokens="
                << summarize_int32_span(std::span<const int32_t>(
                       row_tokens.data(), row_tokens.size()));
        VLOG(3) << "[spec-reject-handle] seq=" << seq_id
                << " accepted_prefix_len=" << accepted.size()
                << " first_reject_idx=" << first_reject_idx << " pos_offset = "
                << (static_cast<int32_t>(accepted.size()) - 1) << " row_tokens="
                << summarize_int32_span(std::span<const int32_t>(
                       row_tokens.data(), row_tokens.size()));
      }

      if (!accepted.empty()) {
        suffix_cache_->add_active_response(
            req_id, std::span<const int32_t>(accepted.data(), accepted.size()));
        append_tokens_with_limit(
            suffix_recent_tokens_[req_id],
            std::span<const int32_t>(accepted.data(), accepted.size()),
            static_cast<size_t>(suffix_cache_->max_tree_depth()));
      }
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

SampleOutput SuffixWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const DraftProposal& draft_proposal,
    const ForwardOutput& target_output) {
  (void)sampling_params;
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  const int32_t batch_size =
      static_cast<int32_t>(draft_proposal.token_ids().size(/*dim=*/0));
  const int32_t vocab_size =
      static_cast<int32_t>(target_output.logits.size(/*dim=*/-1));
  CHECK_EQ(target_output.logits.size(/*dim=*/0),
           static_cast<int64_t>(batch_size) * num_val_tokens)
      << "suffix validate logits shape mismatch";

  using ISlice = torch::indexing::Slice;
  auto target_logits =
      target_output.logits.view({batch_size, num_val_tokens, vocab_size});
  // Use target greedy token as the bonus token, consistent with greedy verify.
  auto bonus_token_ids =
      target_logits.index({ISlice(), num_val_tokens - 1, ISlice()})
          .argmax(/*dim=*/-1, /*keepdim=*/true);

  auto greedy_do_sample = torch::zeros({batch_size}, torch::kBool);
  return spec_verify::run_rejection_sampling({.do_sample = greedy_do_sample,
                                              .all_random_sample = false,
                                              .all_greedy_sample = true},
                                             draft_proposal,
                                             target_logits,
                                             target_output,
                                             bonus_token_ids,
                                             enable_fused_kernel_);
}

}  // namespace xllm
