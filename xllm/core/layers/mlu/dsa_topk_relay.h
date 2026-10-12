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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <optional>
#include <utility>

#include "core/layers/common/dsa_topk_share_plan.h"
#include "core/layers/mlu/dsa_topk_state.h"

namespace xllm::layer {

class DsaPrefillRemap final {
 public:
  DsaPrefillRemap(torch::Tensor block_table,
                  torch::Tensor topk,
                  torch::Tensor context_lens,
                  torch::Tensor sorted_slots,
                  torch::Tensor sorted_rows)
      : block_table_(std::move(block_table)),
        topk_(std::move(topk)),
        context_lens_(std::move(context_lens)),
        sorted_slots_(std::move(sorted_slots)),
        sorted_rows_(std::move(sorted_rows)) {}

  const torch::Tensor& block_table() const { return block_table_; }

  bool matches(const torch::Tensor& topk,
               const torch::Tensor& context_lens,
               const torch::Tensor& sorted_slots,
               const torch::Tensor& sorted_rows) const {
    return topk_.unsafeGetTensorImpl() == topk.unsafeGetTensorImpl() &&
           context_lens_.unsafeGetTensorImpl() ==
               context_lens.unsafeGetTensorImpl() &&
           sorted_slots_.unsafeGetTensorImpl() ==
               sorted_slots.unsafeGetTensorImpl() &&
           sorted_rows_.unsafeGetTensorImpl() ==
               sorted_rows.unsafeGetTensorImpl() &&
           block_table_.defined() && block_table_.sizes() == topk.sizes() &&
           block_table_.scalar_type() == topk.scalar_type() &&
           block_table_.device() == topk.device();
  }

 private:
  torch::Tensor block_table_;
  torch::Tensor topk_;
  torch::Tensor context_lens_;
  torch::Tensor sorted_slots_;
  torch::Tensor sorted_rows_;
};

// Per-attention transfer prepared by a relay or the MTP bridge. Factory
// methods expose only valid reuse/capture combinations, and output is
// published as one DsaTopkState rather than through independent pointers.
// DCP forwards additionally relay a rank-local projection of the shared
// state: the producer layer derives it once per forward so consumers skip
// per-layer re-localization. The projection is optional; MTP steps and
// non-DCP forwards transfer the global state only and localize on demand.
class DsaTopkTransfer final {
 public:
  static DsaTopkTransfer capture_output() {
    return DsaTopkTransfer(/*input=*/std::nullopt,
                           /*localized_input=*/std::nullopt,
                           /*captures_output=*/true);
  }

  static DsaTopkTransfer reuse(
      const DsaTopkState& input,
      const std::optional<DsaTopkState>& localized_input = std::nullopt,
      std::optional<DsaPrefillRemap> prefill_remap = std::nullopt) {
    return DsaTopkTransfer(input,
                           localized_input,
                           /*captures_output=*/false,
                           std::move(prefill_remap));
  }

  static DsaTopkTransfer reuse_and_capture(
      const DsaTopkState& input,
      std::optional<DsaPrefillRemap> prefill_remap = std::nullopt) {
    return DsaTopkTransfer(input,
                           /*localized_input=*/std::nullopt,
                           /*captures_output=*/true,
                           std::move(prefill_remap));
  }

  static DsaTopkTransfer prepare_mtp_step(
      const std::optional<DsaTopkState>& state,
      const torch::Device& device) {
    if (!state.has_value()) {
      return capture_output();
    }
    return reuse_and_capture(state->to(device));
  }

  const DsaTopkState* input() const {
    return input_.has_value() ? &input_.value() : nullptr;
  }

  const DsaTopkState* localized_input() const {
    return localized_input_.has_value() ? &localized_input_.value() : nullptr;
  }

  const DsaPrefillRemap* prefill_remap_input() const {
    return prefill_remap_input_.has_value() ? &prefill_remap_input_.value()
                                            : nullptr;
  }

  bool captures_output() const { return captures_output_; }

  void publish_output(DsaTopkState output) {
    CHECK(captures_output_) << "DSA top-k transfer does not capture output.";
    CHECK(!output_.has_value()) << "DSA top-k output was already published.";
    output_ = std::move(output);
  }

  void publish_localized(DsaTopkState localized) {
    CHECK(captures_output_) << "DSA top-k transfer does not capture output.";
    CHECK(output_.has_value())
        << "DSA top-k localized view requires a published global state.";
    CHECK(!localized_output_.has_value())
        << "DSA top-k localized view was already published.";
    localized_output_ = std::move(localized);
  }

  void publish_prefill_remap(DsaPrefillRemap remap) {
    CHECK(captures_output_) << "DSA transfer does not capture prefill remap.";
    CHECK(output_.has_value())
        << "DSA prefill remap requires a published top-k state.";
    CHECK(!prefill_remap_output_.has_value())
        << "DSA prefill remap was already published.";
    prefill_remap_output_ = std::move(remap);
  }

  void complete(
      const std::optional<DsaTopkState>& resolved_state,
      const std::optional<DsaTopkState>& localized_state = std::nullopt) {
    // A data-parallel dummy invocation resolves no state and publishes
    // nothing, and a reuse-only transfer ignores what it resolved.
    if (!captures_output_ || !resolved_state.has_value()) {
      return;
    }
    publish_output(resolved_state.value());
    if (localized_state.has_value()) {
      publish_localized(localized_state.value());
    }
  }

  const DsaTopkState* output() const {
    return output_.has_value() ? &output_.value() : nullptr;
  }

  const DsaTopkState* localized_output() const {
    return localized_output_.has_value() ? &localized_output_.value() : nullptr;
  }

  const DsaPrefillRemap* prefill_remap_output() const {
    return prefill_remap_output_.has_value() ? &prefill_remap_output_.value()
                                             : nullptr;
  }

  std::optional<DsaTopkState> mtp_output_state() const { return output_; }

 private:
  DsaTopkTransfer(std::optional<DsaTopkState> input,
                  std::optional<DsaTopkState> localized_input,
                  bool captures_output,
                  std::optional<DsaPrefillRemap> prefill_remap = std::nullopt)
      : input_(std::move(input)),
        localized_input_(std::move(localized_input)),
        prefill_remap_input_(std::move(prefill_remap)),
        captures_output_(captures_output) {}

  std::optional<DsaTopkState> input_;
  std::optional<DsaTopkState> localized_input_;
  std::optional<DsaPrefillRemap> prefill_remap_input_;
  bool captures_output_ = false;
  std::optional<DsaTopkState> output_;
  std::optional<DsaTopkState> localized_output_;
  std::optional<DsaPrefillRemap> prefill_remap_output_;
};

// Owns the forward-scoped producer-to-consumer state. The model only resets
// and passes this relay; layer-role interpretation and transfer validation stay
// below the model boundary.
class DsaTopkRelay final {
 public:
  void reset() {
    state_.reset();
    localized_state_.reset();
    prefill_remap_.reset();
  }

  std::optional<DsaTopkTransfer> prepare_layer(
      const DsaTopkShareDecision& decision) const {
    CHECK(!(decision.reuse_topk && decision.output_topk))
        << "A DSA layer cannot reuse and publish top-k simultaneously.";
    if (decision.reuse_topk) {
      CHECK(state_.has_value())
          << "DSA top-k reuse requires a previously published state.";
      return DsaTopkTransfer::reuse(
          state_.value(), localized_state_, prefill_remap_);
    }
    if (decision.output_topk) {
      return DsaTopkTransfer::capture_output();
    }
    return std::nullopt;
  }

  void finish_layer(const DsaTopkShareDecision& decision,
                    const DsaTopkTransfer& transfer) {
    if (!decision.output_topk) {
      if (decision.last_shared) {
        prefill_remap_.reset();
      }
      return;
    }
    const DsaTopkState* output = transfer.output();
    CHECK(output != nullptr) << "DSA top-k producer did not publish its state.";
    state_ = *output;
    if (const DsaPrefillRemap* remap = transfer.prefill_remap_output();
        remap != nullptr) {
      prefill_remap_ = *remap;
    } else {
      prefill_remap_.reset();
    }
    if (const DsaTopkState* localized = transfer.localized_output();
        localized != nullptr) {
      localized_state_ = *localized;
    } else {
      localized_state_.reset();
    }
  }

 private:
  std::optional<DsaTopkState> state_;
  std::optional<DsaTopkState> localized_state_;
  std::optional<DsaPrefillRemap> prefill_remap_;
};

}  // namespace xllm::layer
