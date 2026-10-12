/* Copyright 2026 The xLLM Authors.

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

// JoyOV2Transformer3DModel — Diffusers checkpoint layout:
//   x_embedder / context_embedder / time_embedder / audio_embedder
//   noise_refiner / context_text_refiner / audio_refiner
//   transformer_blocks / proj_out / audio_proj_out
// Reference: tools/SglangXvideo-f3a6983/.../models/dits/joyo_v2.py
//
// MoE on transformer_blocks.1+ (layer 0 stays dense). Experts are EP-sharded
// on dit_tp_group (ep=1 keeps all experts local). Attention / dense FFN are
// TP-sharded (ColumnParallel QKV + RowParallel out).

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/linear.h"
#include "core/layers/common/rms_norm.h"
#include "core/platform/device.h"
#include "core/util/utils.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/process_group.h"
#include "models/dit/utils/dit_parallel_linear.h"
#include "models/dit/utils/dit_parallel_mixin.h"
#include "models/model_registry.h"
#if defined(USE_NPU)
#include "models/dit/utils/dit_block_weight_manager.h"
#endif

namespace xllm {

namespace joyo_v2 {

// Joy-only dtype override (kept out of DiTModelLoader::get_torch_dtype so
// Wan / Flux / etc. keep the worker-wide first-hit seed).
//
// Why: DiTWorker seeds one tensor_options from get_torch_dtype() (unordered
// first non-empty component). Joy packages mix fp32 VAE/scheduler with bf16
// transformer; that seed can be float32 and would upcast DiT weights (~2x
// HBM) if used as-is.
//
// What: re-bind options to this component's own ModelArgs.dtype()
// (config.json). JoyOV2Transformer / pipeline / in-process Qwen3-VL use
// this; XVAE already forces float32.
inline torch::TensorOptions resolve_component_tensor_options(
    const torch::TensorOptions& base,
    const std::string& component_dtype) {
  if (component_dtype.empty()) {
    return base;
  }
  const auto dtype = util::parse_dtype(component_dtype, base.device());
  return base.dtype(dtype);
}

inline void empty_device_cache(const torch::Device& device) {
  if (!device.is_cpu()) {
    Device npu_dev(device);
    npu_dev.set_device();
    npu_dev.synchronize_default_stream();
    Device::empty_cache(static_cast<int32_t>(device.index()));
  }
}

}  // namespace joyo_v2

struct JoyOV2RopeMeta {
  torch::Tensor position_id;  // [3, S]
  torch::Tensor cu_seqlens;   // [n+1]
  torch::Tensor seqlens;      // [n]
  int64_t max_seq_len = 0;
};

struct JoyOV2PackedRopeMeta {
  JoyOV2RopeMeta text;
  std::optional<JoyOV2RopeMeta> pixel;
  std::optional<JoyOV2RopeMeta> audio;
  std::optional<JoyOV2RopeMeta> mix;
};

// JoyOV2 block primitives (model-local; unified layers live in core/layers).
// Module names match the Diffusers checkpoint: norm1 / attn / ff / mod.
// MoE FFN for transformer_blocks.1+.

class JoyOV2MoEFeedForwardImpl : public torch::nn::Module {
 public:
  JoyOV2MoEFeedForwardImpl(int64_t hidden_size,
                           int64_t moe_hidden_size,
                           int64_t num_experts,
                           float top_p,
                           int64_t share_expert_dim,
                           float routed_scaling_factor,
                           const ParallelArgs& parallel_args,
                           const QuantArgs& quant_args,
                           const torch::TensorOptions& options)
      : hidden_size_(hidden_size),
        moe_hidden_size_(moe_hidden_size),
        num_experts_(num_experts),
        top_p_(top_p),
        share_expert_dim_(share_expert_dim),
        routed_scaling_factor_(routed_scaling_factor),
        options_(options) {
    ProcessGroup* tp = parallel_args.dit_tp_group_;
    ep_size_ = tp != nullptr ? static_cast<int64_t>(tp->world_size()) : 1;
    ep_rank_ = tp != nullptr ? static_cast<int64_t>(tp->rank()) : 0;
    CHECK_GT(ep_size_, 0);
    CHECK_EQ(num_experts_ % ep_size_, 0)
        << "num_experts=" << num_experts_
        << " not divisible by ep_size=" << ep_size_;
    num_local_experts_ = num_experts_ / ep_size_;
    tp_group_ = tp;

    gate_ = register_module(
        "gate",
        layer::ReplicatedLinear(
            hidden_size, num_experts, /*bias=*/false, quant_args, options));

    // Local expert weights — Diffusers: [E, 2*H_moe, H] / [E, H, H_moe]
    experts_w1_ = register_parameter(
        "experts_w1",
        torch::empty({num_local_experts_, moe_hidden_size * 2, hidden_size},
                     options));
    experts_w2_ = register_parameter(
        "experts_w2",
        torch::empty({num_local_experts_, hidden_size, moe_hidden_size},
                     options));

    // Shared expert: TP-sharded (same process group as EP in JoyO eval)
    shared_w1_ =
        register_module("shared_expert_w1",
                        layer::ColumnParallelLinear(hidden_size,
                                                    share_expert_dim * 2,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    tp,
                                                    options));
    shared_w2_ = register_module(
        "shared_expert_w2",
        layer::RowParallelLinear(share_expert_dim,
                                 hidden_size,
                                 /*bias=*/false,
                                 /*input_is_parallelized=*/true,
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 tp,
                                 options));
    shared_gate_ = register_module(
        "shared_expert_gate",
        layer::ReplicatedLinear(
            hidden_size, 1, /*bias=*/false, quant_args, options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const std::optional<torch::Tensor>& seqlens) {
    // hidden_states: [B, S, D] (B=1 packed)
    const int64_t b = hidden_states.size(0);
    const int64_t s = hidden_states.size(1);
    const int64_t d = hidden_states.size(2);
    auto x = hidden_states.reshape({-1, d});
    const int64_t num_tokens = x.size(0);

    auto gate_logits = gate_->forward(x);
    auto gate_prob = torch::sigmoid(gate_logits.to(torch::kFloat));

    auto [expert_token_ids, expert_token_weights] =
        expert_chosen_routing(gate_prob, num_tokens, seqlens);

    torch::Tensor routed;
    if (ep_size_ == 1) {
      // All experts local: gather → bmm → index_add
      auto expert_input = x.index({expert_token_ids});  // [E, C, D]
      auto expert_out =
          local_expert_compute(expert_input, expert_token_weights);
      routed = torch::zeros_like(x);
      routed.index_add_(/*dim=*/0,
                        expert_token_ids.reshape({-1}),
                        expert_out.reshape({-1, d}).to(x.dtype()));
    } else {
      // EP>1 without Joytron dispatcher yet: each rank runs local experts on
      // tokens it was assigned, then all_reduce the dense output. Correctness
      // for multi-rank lands with the EC dispatcher; this keeps EP shard load
      // + local compute wired.
      CHECK(tp_group_ != nullptr);
      auto local_ids =
          expert_token_ids.slice(/*dim=*/0,
                                 ep_rank_ * num_local_experts_,
                                 (ep_rank_ + 1) * num_local_experts_);
      auto local_w =
          expert_token_weights.slice(/*dim=*/0,
                                     ep_rank_ * num_local_experts_,
                                     (ep_rank_ + 1) * num_local_experts_);
      auto expert_input = x.index({local_ids});
      auto expert_out = local_expert_compute(expert_input, local_w);
      routed = torch::zeros_like(x);
      routed.index_add_(/*dim=*/0,
                        local_ids.reshape({-1}),
                        expert_out.reshape({-1, d}).to(x.dtype()));
      tp_group_->allreduce(routed);
    }

    routed = routed * routed_scaling_factor_;
    auto shared = shared_expert_forward(x);
    return (routed + shared).view({b, s, d});
  }

  void load_state_dict(const StateDict& state_dict) {
    gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
    // experts.w1 / experts.w2 are full [E, ...] — take EP slice
    {
      auto w1 = state_dict.get_tensor("experts.w1");
      if (w1.defined()) {
        CHECK_EQ(w1.size(0), num_experts_);
        const int64_t start = ep_rank_ * num_local_experts_;
        experts_w1_.data().copy_(
            w1.slice(/*dim=*/0, start, start + num_local_experts_));
      }
      auto w2 = state_dict.get_tensor("experts.w2");
      if (w2.defined()) {
        CHECK_EQ(w2.size(0), num_experts_);
        const int64_t start = ep_rank_ * num_local_experts_;
        experts_w2_.data().copy_(
            w2.slice(/*dim=*/0, start, start + num_local_experts_));
      }
    }
    ProcessGroup* tp = shared_w1_->process_group();
    const int64_t tp_size =
        tp != nullptr ? static_cast<int64_t>(tp->world_size()) : 1;
    CHECK_EQ(share_expert_dim_ % tp_size, 0);
    const int64_t local = share_expert_dim_ / tp_size;
    shared_w1_->load_state_dict(
        state_dict.get_dict_with_prefix("shared_expert.w1."),
        /*shard_tensor_count=*/2,
        /*shard_sizes=*/{local, local});
    shared_w2_->load_state_dict(
        state_dict.get_dict_with_prefix("shared_expert.w2."));
    shared_gate_->load_state_dict(
        state_dict.get_dict_with_prefix("shared_expert.gate."));
  }

 private:
  torch::Tensor shared_expert_forward(const torch::Tensor& x) {
    auto gate_up = shared_w1_->forward(x);
    auto chunks = gate_up.chunk(2, /*dim=*/-1);
    auto h = torch::silu(chunks[0]) * chunks[1];
    auto out = shared_w2_->forward(h);
    auto g = torch::sigmoid(shared_gate_->forward(out));
    return g * out;
  }

  torch::Tensor local_expert_compute(const torch::Tensor& expert_input,
                                     const torch::Tensor& expert_weights) {
    // expert_input: [E_loc, C, D], experts_w1: [E_loc, 2H, D]
    const auto in_dtype = expert_input.scalar_type();
    auto h = torch::bmm(expert_input.to(experts_w1_.dtype()),
                        experts_w1_.transpose(1, 2));
    auto chunks = h.chunk(2, /*dim=*/-1);
    h = torch::silu(chunks[0]) * chunks[1];
    auto out = torch::bmm(h, experts_w2_.transpose(1, 2));
    out = out * expert_weights.unsqueeze(-1).to(out.dtype());
    return out.to(in_dtype);
  }

  std::pair<torch::Tensor, torch::Tensor> expert_chosen_routing(
      const torch::Tensor& gate_prob,  // [S, E] fp32
      int64_t num_tokens,
      const std::optional<torch::Tensor>& seqlens) {
    auto gate_t = gate_prob.transpose(0, 1);  // [E, S]
    torch::Tensor expert_token_ids;
    torch::Tensor expert_token_weights;

    if (seqlens.has_value() && seqlens->defined() && seqlens->size(0) > 1) {
      // Sample-level: ceil(seqlen_i * top_p) tokens per expert per sample
      std::vector<int64_t> lens(static_cast<size_t>(seqlens->size(0)));
      for (int64_t i = 0; i < seqlens->size(0); ++i) {
        lens[static_cast<size_t>(i)] = (*seqlens)[i].item<int64_t>();
      }
      std::vector<int64_t> kept_cols;
      int64_t start = 0;
      for (int64_t len : lens) {
        int64_t k = static_cast<int64_t>(
            std::ceil(static_cast<double>(len) * static_cast<double>(top_p_)));
        k = std::min(k, len);
        for (int64_t j = 0; j < k; ++j) {
          kept_cols.push_back(start + j);
        }
        start += len;
      }
      auto device = gate_t.device();
      auto sample_ids = torch::repeat_interleave(
          torch::arange(
              static_cast<int64_t>(lens.size()),
              torch::TensorOptions().dtype(torch::kLong).device(device)),
          seqlens->to(torch::kLong).to(device),
          /*dim=*/0,
          num_tokens);
      auto sort_key = gate_t - sample_ids.unsqueeze(0).to(gate_t.dtype()) * 2.0;
      auto sorted = torch::sort(sort_key,
                                /*dim=*/1,
                                /*descending=*/true,
                                /*stable=*/true);
      expert_token_ids = std::get<1>(sorted);
      auto kept = torch::tensor(
          kept_cols, torch::TensorOptions().dtype(torch::kLong).device(device));
      expert_token_ids = expert_token_ids.index_select(/*dim=*/1, kept);
      expert_token_weights = torch::gather(gate_t, /*dim=*/1, expert_token_ids);
    } else {
      const int64_t tokens_per_expert = static_cast<int64_t>(std::ceil(
          static_cast<double>(num_tokens) * static_cast<double>(top_p_)));
      auto topk = torch::topk(gate_t,
                              tokens_per_expert,
                              /*dim=*/1,
                              /*largest=*/true,
                              /*sorted=*/true);
      expert_token_weights = std::get<0>(topk);
      expert_token_ids = std::get<1>(topk);
    }

    // Normalize weights per token
    auto token_weight_sums =
        torch::zeros({num_tokens},
                     torch::TensorOptions()
                         .dtype(expert_token_weights.dtype())
                         .device(expert_token_weights.device()));
    token_weight_sums.scatter_add_(
        /*dim=*/0,
        expert_token_ids.reshape({-1}),
        expert_token_weights.reshape({-1}));
    auto gathered_sums =
        torch::clamp(token_weight_sums.index({expert_token_ids}),
                     /*min=*/1e-9);
    expert_token_weights = expert_token_weights / gathered_sums;

    // Scale by expert count^0.5 (Joytron default)
    auto token_counts = torch::zeros_like(token_weight_sums);
    token_counts.scatter_add_(
        /*dim=*/0,
        expert_token_ids.reshape({-1}),
        torch::ones_like(expert_token_ids.reshape({-1}),
                         expert_token_weights.dtype()));
    auto gathered_counts = token_counts.index({expert_token_ids});
    expert_token_weights =
        expert_token_weights * gathered_counts.pow(/*exponent=*/0.5);

    return {expert_token_ids, expert_token_weights};
  }

  int64_t hidden_size_ = 0;
  int64_t moe_hidden_size_ = 0;
  int64_t num_experts_ = 0;
  int64_t num_local_experts_ = 0;
  int64_t ep_size_ = 1;
  int64_t ep_rank_ = 0;
  float top_p_ = 0.125f;
  int64_t share_expert_dim_ = 0;
  float routed_scaling_factor_ = 10.0f;
  torch::TensorOptions options_;
  ProcessGroup* tp_group_ = nullptr;

  layer::ReplicatedLinear gate_{nullptr};
  torch::Tensor experts_w1_;
  torch::Tensor experts_w2_;
  layer::ColumnParallelLinear shared_w1_{nullptr};
  layer::RowParallelLinear shared_w2_{nullptr};
  layer::ReplicatedLinear shared_gate_{nullptr};
};
TORCH_MODULE(JoyOV2MoEFeedForward);

// JoyO-only: SP size from --sp_size. Avoid ProcessGroup::world_size() /
// all_to_all_4D mid-forward (those touch HCCL getSize and can abort SP ranks).
inline int64_t joyo_sp_world() {
  return std::max<int64_t>(1, ParallelConfig::get_instance().sp_size());
}

// Sync Ulysses all2all sized by --sp_size; sp_group is collective handle only.
inline torch::Tensor joyo_sp_all_to_all_single(const torch::Tensor& input_4d,
                                               int64_t sp_size,
                                               bool seq_to_head,
                                               ProcessGroup* sp_group) {
  if (sp_group == nullptr || sp_size <= 1) {
    return input_4d;
  }
  const auto sizes = input_4d.sizes().vec();
  const int64_t bs = sizes[0];
  std::vector<int64_t> empty_splits;
  if (seq_to_head) {
    // (bs, shard_seq, head, dim) -> (bs, seq, head/sp, dim)
    const int64_t shard_seqlen = sizes[1];
    const int64_t head_num = sizes[2];
    const int64_t head_size = sizes[3];
    const int64_t seqlen = shard_seqlen * sp_size;
    const int64_t shard_head_num = head_num / sp_size;
    auto input_t =
        input_4d.reshape({bs, shard_seqlen, sp_size, shard_head_num, head_size})
            .transpose(0, 2)
            .contiguous();
    torch::Tensor output = torch::empty_like(input_t);
    sp_group->all_to_all_single(
        output, input_t, empty_splits, empty_splits, /*async_op=*/false);
    return output.reshape({seqlen, bs, shard_head_num, head_size})
        .transpose(0, 1)
        .contiguous()
        .reshape({bs, seqlen, shard_head_num, head_size});
  }
  // (bs, seq, head/sp, dim) -> (bs, shard_seq, head, dim)
  const int64_t seqlen = sizes[1];
  const int64_t shard_head_num = sizes[2];
  const int64_t head_size = sizes[3];
  const int64_t shard_seqlen = seqlen / sp_size;
  const int64_t head_num = shard_head_num * sp_size;
  auto input_t =
      input_4d.reshape({bs, sp_size, shard_seqlen, shard_head_num, head_size})
          .transpose(0, 3)
          .transpose(0, 1)
          .contiguous();
  torch::Tensor output = torch::empty_like(input_t);
  sp_group->all_to_all_single(
      output, input_t, empty_splits, empty_splits, /*async_op=*/false);
  return output.reshape({head_num, shard_seqlen, bs, head_size})
      .transpose(0, 2)
      .contiguous()
      .reshape({bs, shard_seqlen, head_num, head_size});
}

inline torch::Tensor joyo_sp_all_to_all(const torch::Tensor& input,
                                        int64_t heads,
                                        int64_t dim_head,
                                        int64_t tp_size,
                                        ProcessGroup* sp_group) {
  const int64_t sp_size = joyo_sp_world();
  auto out = joyo_sp_all_to_all_single(
      input.view({input.size(0), -1, heads / tp_size, dim_head}),
      sp_size,
      /*seq_to_head=*/true,
      sp_group);
  return out.view({input.size(0), -1, heads * dim_head / (tp_size * sp_size)});
}

inline torch::Tensor joyo_sp_all_to_all_reverse(const torch::Tensor& input,
                                                int64_t heads,
                                                int64_t dim_head,
                                                int64_t tp_size,
                                                ProcessGroup* sp_group) {
  const int64_t sp_size = joyo_sp_world();
  auto out = joyo_sp_all_to_all_single(
      input.view({input.size(0), -1, heads / (tp_size * sp_size), dim_head}),
      sp_size,
      /*seq_to_head=*/false,
      sp_group);
  return out.view({input.size(0), -1, heads * dim_head / tp_size});
}

inline torch::Tensor expand_per_sample(
    const torch::Tensor& per_sample,  // [N, D]
    const torch::Tensor& seqlens,     // [N]
    int64_t total_s) {
  // repeat_interleave requires int64 repeats (matches SGLang).
  auto repeats = seqlens.to(torch::kLong).to(per_sample.device());
  return torch::repeat_interleave(per_sample, repeats, /*dim=*/0, total_s)
      .unsqueeze(0);
}

// NeoX-style RoPE. x: [B,S,H,D], cos/sin: [S, D/2]
inline torch::Tensor apply_rotary_neox(torch::Tensor x,
                                       const torch::Tensor& cos,
                                       const torch::Tensor& sin) {
  const auto orig_dtype = x.scalar_type();
  auto c = cos.to(orig_dtype).unsqueeze(0).unsqueeze(2);
  auto s = sin.to(orig_dtype).unsqueeze(0).unsqueeze(2);
  auto chunks = x.chunk(2, /*dim=*/-1);
  auto o1 = chunks[0] * c - chunks[1] * s;
  auto o2 = chunks[1] * c + chunks[0] * s;
  return torch::cat({o1, o2}, /*dim=*/-1);
}

class JoyOV2ModulateImpl : public torch::nn::Module {
 public:
  JoyOV2ModulateImpl(int64_t hidden_size,
                     const torch::TensorOptions& options,
                     int64_t factor = 6)
      : factor_(factor), hidden_size_(hidden_size) {
    // Checkpoint: mod.modulate_table [1, factor*H]
    modulate_table_ = register_parameter(
        "modulate_table",
        torch::zeros({1, factor * hidden_size}, options.dtype(torch::kFloat)));
  }

  // timestep_emb: [N, factor*H] -> 6 tensors of [N, H]
  // Match SGLang/Joytron: (modulate_table + x).chunk(factor) — additive, not
  // (1+table)*x. Block forward still applies x*(1+scale)+shift on top.
  std::vector<torch::Tensor> forward(const torch::Tensor& timestep_emb) {
    auto fused = (modulate_table_.to(timestep_emb.dtype()) + timestep_emb)
                     .view({-1, factor_, hidden_size_})
                     .unbind(/*dim=*/1);
    return std::vector<torch::Tensor>(fused.begin(), fused.end());
  }

  void load_state_dict(const StateDict& state_dict) {
    auto t = state_dict.get_tensor("modulate_table");
    if (t.defined()) {
      CHECK_EQ(t.sizes(), modulate_table_.sizes());
      modulate_table_.data().copy_(t);
    }
  }

 private:
  int64_t factor_ = 6;
  int64_t hidden_size_ = 0;
  torch::Tensor modulate_table_;
};
TORCH_MODULE(JoyOV2Modulate);

class JoyOV2DenseFFNImpl : public torch::nn::Module {
 public:
  JoyOV2DenseFFNImpl(int64_t hidden_size,
                     int64_t ffn_hidden_size,
                     const ParallelArgs& parallel_args,
                     const QuantArgs& quant_args,
                     const torch::TensorOptions& options)
      : ffn_hidden_size_(ffn_hidden_size) {
    ProcessGroup* tp = parallel_args.dit_tp_group_;
    w1_ = register_module("w1",
                          layer::ColumnParallelLinear(hidden_size,
                                                      ffn_hidden_size * 2,
                                                      /*bias=*/false,
                                                      /*gather_output=*/false,
                                                      quant_args,
                                                      tp,
                                                      options));
    w2_ = register_module(
        "w2",
        layer::RowParallelLinear(ffn_hidden_size,
                                 hidden_size,
                                 /*bias=*/false,
                                 /*input_is_parallelized=*/true,
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 tp,
                                 options));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    auto gate_up = w1_->forward(x);
    auto chunks = gate_up.chunk(2, /*dim=*/-1);
    return w2_->forward(torch::silu(chunks[0]) * chunks[1]);
  }

  void load_state_dict(const StateDict& state_dict) {
    // Checkpoint packs [gate|up]; TP must shard each half independently.
    // shard_sizes are per-rank (see load_merged_weight_v2).
    ProcessGroup* tp = w1_->process_group();
    const int64_t tp_size =
        tp != nullptr ? static_cast<int64_t>(tp->world_size()) : 1;
    CHECK_EQ(ffn_hidden_size_ % tp_size, 0);
    const int64_t local = ffn_hidden_size_ / tp_size;
    w1_->load_state_dict(state_dict.get_dict_with_prefix("w1."),
                         /*shard_tensor_count=*/2,
                         /*shard_sizes=*/{local, local});
    w2_->load_state_dict(state_dict.get_dict_with_prefix("w2."));
  }

 private:
  int64_t ffn_hidden_size_ = 0;
  layer::ColumnParallelLinear w1_{nullptr};
  layer::RowParallelLinear w2_{nullptr};
};
TORCH_MODULE(JoyOV2DenseFFN);

class JoyOV2AttentionImpl : public torch::nn::Module {
 public:
  JoyOV2AttentionImpl(int64_t hidden_size,
                      int64_t num_heads,
                      int64_t num_kv_heads,
                      int64_t head_dim,
                      double norm_eps,
                      const ParallelArgs& parallel_args,
                      const QuantArgs& quant_args,
                      const torch::TensorOptions& options)
      : hidden_size_(hidden_size),
        num_heads_(num_heads),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        sp_group_(parallel_args.dit_sp_group_) {
    ProcessGroup* tp = parallel_args.dit_tp_group_;
    tp_size_ = tp != nullptr ? static_cast<int64_t>(tp->world_size()) : 1;
    if (tp_size_ <= 1) {
      tp_size_ = std::max<int64_t>(
          1, static_cast<int64_t>(ParallelConfig::get_instance().tp_size()));
    }
    CHECK_EQ(num_heads % tp_size_, 0);
    CHECK_EQ(num_kv_heads % tp_size_, 0);
    local_heads_ = num_heads / tp_size_;
    local_kv_heads_ = num_kv_heads / tp_size_;
    const int64_t sp_cfg = ParallelConfig::get_instance().sp_size();
    if (sp_cfg > 1) {
      CHECK_EQ(local_heads_ % sp_cfg, 0)
          << "JoyOV2Attention: local_heads=" << local_heads_
          << " must divide sp_size=" << sp_cfg;
      CHECK_EQ(local_kv_heads_ % sp_cfg, 0)
          << "JoyOV2Attention: local_kv_heads=" << local_kv_heads_
          << " must divide sp_size=" << sp_cfg;
    }

    // Checkpoint packs QKV + per-Q-head gate scalar:
    //   (H + 2*KV)*head_dim + H  == 10288 for H=48, KV=16, D=128
    const int64_t qkv_out =
        (num_heads + 2 * num_kv_heads) * head_dim + num_heads;
    to_qkv_ =
        register_module("to_qkv",
                        layer::ColumnParallelLinear(hidden_size,
                                                    qkv_out,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    tp,
                                                    options));
    // Input is head-sharded (local_heads * head_dim); reduce across TP.
    to_out_ = register_module(
        "to_out",
        layer::RowParallelLinear(hidden_size,
                                 hidden_size,
                                 /*bias=*/false,
                                 /*input_is_parallelized=*/true,
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 tp,
                                 options));
    norm_q_ =
        register_module("norm_q", layer::RMSNorm(head_dim, norm_eps, options));
    norm_k_ =
        register_module("norm_k", layer::RMSNorm(head_dim, norm_eps, options));
  }

  // Per-sample SDPA over packed [sample0|sample1|...] (SGLang _sdpa_varlen).
  // Full-sequence SDPA would let CFG pos/neg attend across samples and
  // collapse generation into noise.
  static torch::Tensor sdpa_varlen(const torch::Tensor& q,
                                   const torch::Tensor& k,
                                   const torch::Tensor& v,
                                   const torch::Tensor& cu_seqlens) {
    // q/k/v: [B=1, H, S, D] after transpose
    CHECK_EQ(q.size(0), 1);
    const auto cu = cu_seqlens.to(torch::kCPU).to(torch::kInt64);
    const int64_t n = cu.numel() - 1;
    std::vector<torch::Tensor> outs;
    outs.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      const int64_t start = cu[i].item<int64_t>();
      const int64_t end = cu[i + 1].item<int64_t>();
      CHECK_GT(end, start);
      auto qi = q.slice(/*dim=*/2, start, end);
      auto ki = k.slice(/*dim=*/2, start, end);
      auto vi = v.slice(/*dim=*/2, start, end);
      outs.push_back(torch::scaled_dot_product_attention(
          qi, ki, vi, std::nullopt, /*dropout_p=*/0.0, /*is_causal=*/false));
    }
    return torch::cat(outs, /*dim=*/2);  // [1, H, S, D]
  }

  // ulysses_sp: caller already SP-split the sequence (mix blocks only).
  // Refiners must pass false — all2all on a full sequence doubles S.
  torch::Tensor forward(const torch::Tensor& x,
                        const std::optional<torch::Tensor>& freqs_cis,
                        const std::optional<torch::Tensor>& cu_seqlens,
                        bool ulysses_sp = false) {
    const int64_t b = x.size(0);
    const int64_t s = x.size(1);
    auto qkvg = to_qkv_->forward(x);
    // Checkpoint / SGLang layout: [Q | KV_interleaved | gate], where KV is
    // packed per-head as [K_i | V_i] (MergedColumnParallelLinear sizes
    // [q_dim, 2*kv_heads*head_dim, gate_dim]). Not contiguous [Q|K|V|gate].
    const int64_t q_size = local_heads_ * head_dim_;
    const int64_t kv_size = local_kv_heads_ * head_dim_ * 2;
    const int64_t gate_size = local_heads_;
    auto q = qkvg.slice(/*dim=*/-1, 0, q_size);
    auto kv = qkvg.slice(/*dim=*/-1, q_size, q_size + kv_size);
    auto gate = qkvg.slice(
        /*dim=*/-1, q_size + kv_size, q_size + kv_size + gate_size);

    q = q.view({b, s, local_heads_, head_dim_});
    auto kv_view = kv.view({b, s, local_kv_heads_, head_dim_ * 2});
    auto kv_chunks = kv_view.chunk(2, /*dim=*/-1);
    auto k = kv_chunks[0];
    auto v = kv_chunks[1].contiguous();

    auto q_flat = q.reshape({b * s * local_heads_, head_dim_});
    auto k_flat = k.reshape({b * s * local_kv_heads_, head_dim_});
    q = std::get<0>(norm_q_->forward(q_flat))
            .view({b, s, local_heads_, head_dim_});
    k = std::get<0>(norm_k_->forward(k_flat))
            .view({b, s, local_kv_heads_, head_dim_});

    if (local_kv_heads_ != local_heads_) {
      const int64_t rep = local_heads_ / local_kv_heads_;
      k = k.repeat_interleave(rep, /*dim=*/2);
      v = v.repeat_interleave(rep, /*dim=*/2);
    }

    const int64_t sp_size = ParallelConfig::get_instance().sp_size();
    const bool use_sp = ulysses_sp && sp_size > 1 && sp_group_ != nullptr;
    if (use_sp) {
      CHECK(freqs_cis.has_value() && freqs_cis->defined());
      auto q3 = q.reshape({b, s, local_heads_ * head_dim_}).contiguous();
      auto k3 = k.reshape({b, s, local_heads_ * head_dim_}).contiguous();
      auto v3 = v.reshape({b, s, local_heads_ * head_dim_}).contiguous();
      q3 = joyo_sp_all_to_all(q3, num_heads_, head_dim_, tp_size_, sp_group_);
      k3 = joyo_sp_all_to_all(k3, num_heads_, head_dim_, tp_size_, sp_group_);
      v3 = joyo_sp_all_to_all(v3, num_heads_, head_dim_, tp_size_, sp_group_);
      const int64_t attn_heads = local_heads_ / sp_size;
      q = q3.view({b, -1, attn_heads, head_dim_});
      k = k3.view({b, -1, attn_heads, head_dim_});
      v = v3.view({b, -1, attn_heads, head_dim_});
    }

    if (freqs_cis.has_value() && freqs_cis->defined()) {
      // freqs_cis: [S, head_dim] packed [cos|sin].
      // Ulysses: S is full sequence (after all2all). Else: local tokens.
      auto halves = freqs_cis->chunk(2, /*dim=*/-1);
      auto cos = halves[0];
      auto sin = halves[1];
      CHECK_EQ(cos.size(0), q.size(1))
          << "JoyOV2Attention RoPE length mismatch: freqs=" << cos.size(0)
          << " q_s=" << q.size(1) << " ulysses=" << use_sp;
      q = apply_rotary_neox(q, cos, sin);
      k = apply_rotary_neox(k, cos, sin);
    }

    q = q.transpose(1, 2);
    k = k.transpose(1, 2);
    v = v.transpose(1, 2);

    torch::Tensor out;
    // After Ulysses all2all, QKV have the full packed sequence; isolate CFG
    // samples with cu_seqlens built from the unsplit mix seqlens.
    const bool allow_varlen = cu_seqlens.has_value() && cu_seqlens->defined() &&
                              cu_seqlens->numel() > 2;
    if (allow_varlen) {
      out = sdpa_varlen(q, k, v, *cu_seqlens);
    } else {
      out = torch::scaled_dot_product_attention(
          q, k, v, std::nullopt, /*dropout_p=*/0.0, /*is_causal=*/false);
    }
    out = out.transpose(1, 2).contiguous();
    if (use_sp) {
      out = joyo_sp_all_to_all_reverse(out.view({b, out.size(1), -1}),
                                       num_heads_,
                                       head_dim_,
                                       tp_size_,
                                       sp_group_);
    }
    out = out.contiguous().view({b, out.size(1), local_heads_ * head_dim_});
    auto gate_s = torch::sigmoid(gate).view({b, s, local_heads_, 1});
    out = out.view({b, s, local_heads_, head_dim_}) * gate_s;
    out = out.contiguous().view({b, s, local_heads_ * head_dim_});
    return to_out_->forward(out);
  }

  void load_state_dict(const StateDict& state_dict) {
    // Match SGLang MergedColumnParallelLinear([q_dim, kv_dim, gate_dim]) with
    // kv_dim = 2 * num_kv_heads * head_dim (per-head [K|V] interleaved).
    // shard_sizes are per-rank.
    ProcessGroup* tp = to_qkv_->process_group();
    const int64_t tp_size =
        tp != nullptr ? static_cast<int64_t>(tp->world_size()) : 1;
    CHECK_EQ(num_heads_ % tp_size, 0);
    CHECK_EQ(num_kv_heads_ % tp_size, 0);
    const int64_t q_dim = (num_heads_ / tp_size) * head_dim_;
    const int64_t kv_dim = (num_kv_heads_ / tp_size) * head_dim_ * 2;
    const int64_t gate_dim = num_heads_ / tp_size;
    to_qkv_->load_state_dict(state_dict.get_dict_with_prefix("to_qkv."),
                             /*shard_tensor_count=*/3,
                             /*shard_sizes=*/{q_dim, kv_dim, gate_dim});
    to_out_->load_state_dict(state_dict.get_dict_with_prefix("to_out."));
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
  }

 private:
  int64_t hidden_size_ = 0;
  int64_t num_heads_ = 0;
  int64_t num_kv_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t local_heads_ = 0;
  int64_t local_kv_heads_ = 0;
  int64_t tp_size_ = 1;
  ProcessGroup* sp_group_ = nullptr;
  layer::ColumnParallelLinear to_qkv_{nullptr};
  layer::RowParallelLinear to_out_{nullptr};
  layer::RMSNorm norm_q_{nullptr};
  layer::RMSNorm norm_k_{nullptr};
};
TORCH_MODULE(JoyOV2Attention);

class JoyOV2TransformerBlockImpl : public torch::nn::Module {
 public:
  struct MoEArgs {
    int64_t moe_hidden_size = 4096;
    int64_t num_experts = 48;
    float top_p = 0.125f;
    int64_t share_expert_dim = 8192;
    float routed_scaling_factor = 10.0f;
  };

  JoyOV2TransformerBlockImpl(int64_t hidden_size,
                             int64_t num_heads,
                             int64_t num_kv_heads,
                             int64_t head_dim,
                             int64_t ffn_hidden_size,
                             double norm_eps,
                             bool sandwich_norm,
                             bool modulation,
                             bool use_moe,
                             const std::optional<MoEArgs>& moe_args,
                             const ParallelArgs& parallel_args,
                             const QuantArgs& quant_args,
                             const torch::TensorOptions& options)
      : sandwich_norm_(sandwich_norm),
        modulation_(modulation),
        use_moe_(use_moe) {
    norm1_ = register_module("norm1",
                             layer::RMSNorm(hidden_size, norm_eps, options));
    norm2_ = register_module("norm2",
                             layer::RMSNorm(hidden_size, norm_eps, options));
    if (sandwich_norm_) {
      norm1_post_ = register_module(
          "norm1_post", layer::RMSNorm(hidden_size, norm_eps, options));
      norm2_post_ = register_module(
          "norm2_post", layer::RMSNorm(hidden_size, norm_eps, options));
    }
    if (modulation_) {
      mod_ = register_module("mod", JoyOV2Modulate(hidden_size, options, 6));
    }
    attn_ = register_module("attn",
                            JoyOV2Attention(hidden_size,
                                            num_heads,
                                            num_kv_heads,
                                            head_dim,
                                            norm_eps,
                                            parallel_args,
                                            quant_args,
                                            options));
    if (use_moe_) {
      CHECK(moe_args.has_value());
      moe_ff_ =
          register_module("ff",
                          JoyOV2MoEFeedForward(hidden_size,
                                               moe_args->moe_hidden_size,
                                               moe_args->num_experts,
                                               moe_args->top_p,
                                               moe_args->share_expert_dim,
                                               moe_args->routed_scaling_factor,
                                               parallel_args,
                                               quant_args,
                                               options));
    } else {
      ff_ = register_module("ff",
                            JoyOV2DenseFFN(hidden_size,
                                           ffn_hidden_size,
                                           parallel_args,
                                           quant_args,
                                           options));
    }
  }

  torch::Tensor forward(
      torch::Tensor x,
      const std::optional<torch::Tensor>& timestep_emb,
      const std::optional<torch::Tensor>& freqs_cis,
      const std::optional<torch::Tensor>& seqlens,
      const std::optional<torch::Tensor>& /*attn_mask*/,
      bool ulysses_sp = false,
      const std::optional<torch::Tensor>& attn_seqlens = std::nullopt,
      const std::optional<torch::Tensor>& token_sample_ids = std::nullopt) {
    torch::Tensor scale_msa, gate_msa, shift_msa, scale_mlp, gate_mlp,
        shift_mlp;
    bool modulated = false;
    if (modulation_ && timestep_emb.has_value() && timestep_emb->defined()) {
      auto mods = mod_->forward(*timestep_emb);
      CHECK_EQ(mods.size(), 6u);
      scale_msa = mods[0];
      gate_msa = mods[1];
      shift_msa = mods[2];
      scale_mlp = mods[3];
      gate_mlp = mods[4];
      shift_mlp = mods[5];
      modulated = true;
    }

    const int64_t s = x.size(1);
    auto expand = [&](const torch::Tensor& t) {
      if (token_sample_ids.has_value() && token_sample_ids->defined()) {
        CHECK_EQ(token_sample_ids->numel(), s);
        return t.index_select(/*dim=*/0, *token_sample_ids).unsqueeze(0);
      }
      CHECK(seqlens.has_value());
      return expand_per_sample(t, *seqlens, s);
    };

    std::optional<torch::Tensor> cu_seqlens;
    const torch::Tensor* lens_for_cu = nullptr;
    if (attn_seqlens.has_value() && attn_seqlens->defined() &&
        attn_seqlens->numel() > 1) {
      lens_for_cu = &*attn_seqlens;
    } else if (seqlens.has_value() && seqlens->defined() &&
               seqlens->numel() > 1) {
      lens_for_cu = &*seqlens;
    }
    if (lens_for_cu != nullptr) {
      const auto device = x.device();
      auto lens = lens_for_cu->to(torch::kInt32).to(device);
      auto cu = torch::zeros(
          {lens.numel() + 1},
          torch::TensorOptions().dtype(torch::kInt32).device(device));
      cu.slice(0, 1, lens.numel() + 1) = torch::cumsum(lens, /*dim=*/0);
      cu_seqlens = cu;
    }

    auto attn_in = std::get<0>(norm1_->forward(x));
    if (modulated) {
      attn_in = attn_in * (1 + expand(scale_msa)) + expand(shift_msa);
    }
    auto attn_out = attn_->forward(attn_in, freqs_cis, cu_seqlens, ulysses_sp);
    if (sandwich_norm_) {
      attn_out = std::get<0>(norm1_post_->forward(attn_out));
    }
    if (modulated) {
      attn_out = attn_out * expand(gate_msa);
    }
    auto h = x + attn_out;

    auto ffn_in = std::get<0>(norm2_->forward(h));
    if (modulated) {
      ffn_in = ffn_in * (1 + expand(scale_mlp)) + expand(shift_mlp);
    }
    torch::Tensor ffn_out;
    if (use_moe_) {
      ffn_out = moe_ff_->forward(ffn_in, ulysses_sp ? std::nullopt : seqlens);
    } else {
      ffn_out = ff_->forward(ffn_in);
    }
    if (sandwich_norm_) {
      ffn_out = std::get<0>(norm2_post_->forward(ffn_out));
    }
    if (modulated) {
      ffn_out = ffn_out * expand(gate_mlp);
    }
    return h + ffn_out;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    if (sandwich_norm_) {
      norm1_post_->load_state_dict(
          state_dict.get_dict_with_prefix("norm1_post."));
      norm2_post_->load_state_dict(
          state_dict.get_dict_with_prefix("norm2_post."));
    }
    if (modulation_) {
      mod_->load_state_dict(state_dict.get_dict_with_prefix("mod."));
    }
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    if (use_moe_) {
      moe_ff_->load_state_dict(state_dict.get_dict_with_prefix("ff."));
    } else {
      ff_->load_state_dict(state_dict.get_dict_with_prefix("ff."));
    }
  }

#if defined(USE_NPU)
  void build_weight_loader() { weight_loader_.build_from_module(*this); }

  dit::BlockWeightLoader& weight_loader() { return weight_loader_; }
#endif

 private:
  bool sandwich_norm_ = true;
  bool modulation_ = true;
  bool use_moe_ = false;
  layer::RMSNorm norm1_{nullptr};
  layer::RMSNorm norm2_{nullptr};
  layer::RMSNorm norm1_post_{nullptr};
  layer::RMSNorm norm2_post_{nullptr};
  JoyOV2Modulate mod_{nullptr};
  JoyOV2Attention attn_{nullptr};
  JoyOV2DenseFFN ff_{nullptr};
  JoyOV2MoEFeedForward moe_ff_{nullptr};
#if defined(USE_NPU)
  dit::BlockWeightLoader weight_loader_;
#endif
};
TORCH_MODULE(JoyOV2TransformerBlock);

class JoyOV2Transformer3DModelImpl : public torch::nn::Module,
                                     public dit::SequenceParallelMixin {
 public:
  // Param/activation dtype from this transformer's config.json, not worker-wide
  // get_torch_dtype(); see joyo_v2::resolve_component_tensor_options.
  explicit JoyOV2Transformer3DModelImpl(const ModelContext& context)
      : dit::SequenceParallelMixin(context.get_parallel_args().dit_sp_group_),
        options_(joyo_v2::resolve_component_tensor_options(
            context.get_tensor_options(),
            context.get_model_args().dtype())),
        parallel_args_(context.get_parallel_args()),
        quant_args_(context.get_quant_args()) {
    const ModelArgs& args = context.get_model_args();
    hidden_size_ = args.hidden_size() > 0 ? args.hidden_size() : 6144;
    num_attention_heads_ = args.n_heads() > 0 ? args.n_heads() : 48;
    num_kv_heads_ = args.n_kv_heads().value_or(16);
    if (num_kv_heads_ <= 0) {
      num_kv_heads_ = 16;
    }
    head_dim_ = args.head_dim() > 0 ? args.head_dim() : 128;
    num_layers_ = args.num_layers() > 0
                      ? args.num_layers()
                      : (args.n_layers() > 0 ? args.n_layers() : 51);
    num_encoder_layers_ =
        args.num_encoder_layers() > 0 ? args.num_encoder_layers() : 2;
    num_experts_ = args.num_experts() > 0 ? args.num_experts() : 48;
    moe_hidden_size_ =
        args.moe_hidden_size() > 0 ? args.moe_hidden_size() : 4096;
    share_expert_dim_ =
        args.share_expert_dim() > 0 ? args.share_expert_dim() : 8192;
    ffn_hidden_size_ = args.ffn_dim() > 0 ? args.ffn_dim() : 16384;
    latent_channels_ =
        args.latent_channels() > 0 ? args.latent_channels() : 128;
    text_dim_ = args.text_dim() > 0 ? args.text_dim() : 5120;
    rope_theta_ = args.rope_theta() > 0
                      ? args.rope_theta()
                      : static_cast<float>(args.rope_theta_dit());
    if (rope_theta_ <= 0) {
      rope_theta_ = 10000.0f;
    }
    mrope_section_ = args.mrope_section();
    if (mrope_section_.empty()) {
      mrope_section_ = {16, 24, 24};
    }
    norm_eps_ = args.rms_norm_eps() > 0 ? args.rms_norm_eps() : 1e-6;
    sandwich_norm_ = args.sandwich_norm();
    top_p_ = args.dit_top_p() > 0 ? args.dit_top_p() : 0.125f;
    routed_scaling_factor_ =
        args.routed_scaling_factor() > 0 ? args.routed_scaling_factor() : 10.0f;

    CHECK_EQ(num_attention_heads_ * head_dim_, hidden_size_);
    ProcessGroup* tp = parallel_args_.dit_tp_group_;

    x_embedder_ =
        register_module("x_embedder",
                        layer::ColumnParallelLinear(latent_channels_,
                                                    hidden_size_,
                                                    /*bias=*/true,
                                                    /*gather_output=*/true,
                                                    quant_args_,
                                                    tp,
                                                    options_));
    audio_embedder_ =
        register_module("audio_embedder",
                        layer::ColumnParallelLinear(latent_channels_,
                                                    hidden_size_,
                                                    /*bias=*/true,
                                                    /*gather_output=*/true,
                                                    quant_args_,
                                                    tp,
                                                    options_));
    context_norm_ =
        register_module("context_embedder_norm",
                        layer::RMSNorm(text_dim_, norm_eps_, options_));
    context_proj_ =
        register_module("context_embedder_proj",
                        layer::ColumnParallelLinear(text_dim_,
                                                    hidden_size_,
                                                    /*bias=*/true,
                                                    /*gather_output=*/true,
                                                    quant_args_,
                                                    tp,
                                                    options_));
    // Checkpoint: time_embedder.linear_1 / linear_2 (no bias)
    time_linear_1_ =
        register_module("time_embedder_linear_1",
                        layer::ColumnParallelLinear(256,
                                                    hidden_size_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/true,
                                                    quant_args_,
                                                    tp,
                                                    options_));
    time_linear_2_ =
        register_module("time_embedder_linear_2",
                        layer::ColumnParallelLinear(hidden_size_,
                                                    hidden_size_ * 6,
                                                    /*bias=*/false,
                                                    /*gather_output=*/true,
                                                    quant_args_,
                                                    tp,
                                                    options_));
    proj_out_norm_ = register_module(
        "proj_out_norm", layer::RMSNorm(hidden_size_, norm_eps_, options_));
    proj_out_linear_ =
        register_module("proj_out_linear",
                        layer::ColumnParallelLinear(hidden_size_,
                                                    latent_channels_,
                                                    /*bias=*/true,
                                                    /*gather_output=*/true,
                                                    quant_args_,
                                                    tp,
                                                    options_));
    audio_proj_out_norm_ =
        register_module("audio_proj_out_norm",
                        layer::RMSNorm(hidden_size_, norm_eps_, options_));
    audio_proj_out_linear_ =
        register_module("audio_proj_out_linear",
                        layer::ColumnParallelLinear(hidden_size_,
                                                    latent_channels_,
                                                    /*bias=*/true,
                                                    /*gather_output=*/true,
                                                    quant_args_,
                                                    tp,
                                                    options_));

    const int64_t ep_size =
        tp != nullptr ? static_cast<int64_t>(tp->world_size()) : 1;
    // Layer 1+ always use Expert-Chosen MoE (ep=1 = all experts on this rank).
    JoyOV2TransformerBlockImpl::MoEArgs moe_args;
    moe_args.moe_hidden_size = moe_hidden_size_;
    moe_args.num_experts = num_experts_;
    moe_args.top_p = top_p_;
    moe_args.share_expert_dim = share_expert_dim_;
    moe_args.routed_scaling_factor = routed_scaling_factor_;

    auto make_block = [&](bool modulation, bool use_moe) {
      return JoyOV2TransformerBlock(
          hidden_size_,
          num_attention_heads_,
          num_kv_heads_,
          head_dim_,
          ffn_hidden_size_,
          norm_eps_,
          sandwich_norm_,
          modulation,
          use_moe,
          use_moe ? std::optional<JoyOV2TransformerBlockImpl::MoEArgs>(moe_args)
                  : std::nullopt,
          parallel_args_,
          quant_args_,
          options_);
    };

    noise_refiner_ = register_module("noise_refiner", torch::nn::ModuleList());
    context_text_refiner_ =
        register_module("context_text_refiner", torch::nn::ModuleList());
    audio_refiner_ = register_module("audio_refiner", torch::nn::ModuleList());
    // Ascend cannot allocate blocks on CPU; park each block immediately so
    // construct peak ≈ 1 block. Denoise streams blocks via DitRollingLoad.
    for (int64_t i = 0; i < num_encoder_layers_; ++i) {
      auto nr = make_block(/*modulation=*/true, /*use_moe=*/false);
      auto tr = make_block(/*modulation=*/false, /*use_moe=*/false);
      auto ar = make_block(/*modulation=*/true, /*use_moe=*/false);
      nr->to(torch::kCPU);
      tr->to(torch::kCPU);
      ar->to(torch::kCPU);
      noise_refiner_layers_.push_back(nr);
      context_text_refiner_layers_.push_back(tr);
      audio_refiner_layers_.push_back(ar);
      noise_refiner_->push_back(nr);
      context_text_refiner_->push_back(tr);
      audio_refiner_->push_back(ar);
    }

    transformer_blocks_ =
        register_module("transformer_blocks", torch::nn::ModuleList());
    layers_.reserve(static_cast<size_t>(num_layers_));
    for (int64_t i = 0; i < num_layers_; ++i) {
      // Layer 0 is dense; layers 1..N-1 are Expert-Chosen MoE.
      const bool use_moe = i > 0;
      auto block = make_block(/*modulation=*/true, use_moe);
      block->to(torch::kCPU);
      layers_.push_back(block);
      transformer_blocks_->push_back(block);
    }

    dit_sp_group_ = parallel_args_.dit_sp_group_;
    // SP size from --sp_size; boundary pad/scatter/gather uses
    // SequenceParallelMixin (same as JoyImage). Attention Ulysses still uses
    // cached sp size (see joyo_sp_all_to_all) — do not query PG mid-forward.
    sp_world_ = joyo_sp_world();
    LOG(INFO) << "JoyOV2Transformer3DModel: layers=" << num_layers_
              << " hidden=" << hidden_size_ << " experts=" << num_experts_
              << " ep_size=" << ep_size
              << " rolling=" << dit_rolling_load_enabled();
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    CHECK(loader != nullptr);
    for (auto& sd : loader->get_state_dicts()) {
      load_state_dict(*sd);
    }
    // Blocks stay on host until prepare_for_rolling_load or
    // prepare_resident_device.
    offload_blocks_to_cpu();
  }

  static bool dit_rolling_load_enabled() {
#if defined(USE_NPU)
    return LoadConfig::get_instance().enable_rolling_load();
#else
    return false;
#endif
  }

#if defined(USE_NPU)
  // Pack each block into host-pinned storage; keep embedders on `device`.
  // Device block weights are streamed later by DitRollingLoadManager.
  void prepare_for_rolling_load(const torch::Device& device) {
    auto build_all = [](std::vector<JoyOV2TransformerBlock>& blocks) {
      for (auto& block : blocks) {
        block->build_weight_loader();
      }
    };
    build_all(context_text_refiner_layers_);
    build_all(noise_refiner_layers_);
    build_all(audio_refiner_layers_);
    build_all(layers_);
    move_non_block_modules_to(device);
    joyo_v2::empty_device_cache(device);
    rolling_prepared_ = true;
  }

  // Forward order must match denoise traversal for wait_h2d indices.
  std::vector<dit::BlockWeightLoader*> get_block_weight_loaders() {
    std::vector<dit::BlockWeightLoader*> loaders;
    loaders.reserve(context_text_refiner_layers_.size() +
                    noise_refiner_layers_.size() +
                    audio_refiner_layers_.size() + layers_.size());
    for (auto& block : context_text_refiner_layers_) {
      loaders.push_back(&block->weight_loader());
    }
    for (auto& block : noise_refiner_layers_) {
      loaders.push_back(&block->weight_loader());
    }
    for (auto& block : audio_refiner_layers_) {
      loaders.push_back(&block->weight_loader());
    }
    for (auto& block : layers_) {
      loaders.push_back(&block->weight_loader());
    }
    return loaders;
  }

  bool rolling_prepared() const { return rolling_prepared_; }
#endif

  // Rolling denoise residency: only non-block modules need H2D; blocks use
  // DitRollingLoadManager slots.
  void prepare_rolling_denoise(const torch::Device& device) {
    move_non_block_modules_to(device);
    joyo_v2::empty_device_cache(device);
  }

  // Whole-DiT residency (no rolling): one block at a time to limit peak HBM.
  void prepare_resident_device(const torch::Device& device) {
    move_blocks_to(device);
    move_non_block_modules_to(device);
    joyo_v2::empty_device_cache(device);
  }

  void offload_blocks_to_cpu() { move_blocks_to(torch::Device(torch::kCPU)); }

  void move_blocks_to(const torch::Device& device) {
    auto move_list = [&](std::vector<JoyOV2TransformerBlock>& blocks) {
      for (auto& block : blocks) {
        block->to(device);
        if (!device.is_cpu()) {
          joyo_v2::empty_device_cache(device);
        }
      }
    };
    move_list(noise_refiner_layers_);
    move_list(context_text_refiner_layers_);
    move_list(audio_refiner_layers_);
    move_list(layers_);
  }

  void move_non_block_modules_to(const torch::Device& device) {
    x_embedder_->to(device);
    audio_embedder_->to(device);
    context_norm_->to(device);
    context_proj_->to(device);
    time_linear_1_->to(device);
    time_linear_2_->to(device);
    proj_out_norm_->to(device);
    proj_out_linear_->to(device);
    audio_proj_out_norm_->to(device);
    audio_proj_out_linear_->to(device);
  }

  template <typename BlockHolder>
  torch::Tensor run_block(
      BlockHolder& block,
      torch::Tensor x,
      const std::optional<torch::Tensor>& timestep_emb,
      const torch::Tensor& freqs,
      const torch::Tensor& seqlens,
      bool ulysses_sp = false,
      const std::optional<torch::Tensor>& attn_seqlens = std::nullopt,
      const std::optional<torch::Tensor>& token_sample_ids = std::nullopt) {
    return block->forward(x,
                          timestep_emb,
                          freqs,
                          seqlens,
                          /*attn_mask=*/std::nullopt,
                          ulysses_sp,
                          attn_seqlens,
                          token_sample_ids);
  }

  void load_state_dict(const StateDict& state_dict) {
    x_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("x_embedder."));
    audio_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("audio_embedder."));
    context_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("context_embedder.norm."));
    context_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("context_embedder.proj."));
    time_linear_1_->load_state_dict(
        state_dict.get_dict_with_prefix("time_embedder.linear_1."));
    time_linear_2_->load_state_dict(
        state_dict.get_dict_with_prefix("time_embedder.linear_2."));
    proj_out_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("proj_out.norm."));
    proj_out_linear_->load_state_dict(
        state_dict.get_dict_with_prefix("proj_out.linear."));
    audio_proj_out_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("audio_proj_out.norm."));
    audio_proj_out_linear_->load_state_dict(
        state_dict.get_dict_with_prefix("audio_proj_out.linear."));

    for (size_t i = 0; i < noise_refiner_layers_.size(); ++i) {
      noise_refiner_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "noise_refiner." + std::to_string(i) + "."));
      context_text_refiner_layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("context_text_refiner." +
                                          std::to_string(i) + "."));
      audio_refiner_layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "audio_refiner." + std::to_string(i) + "."));
    }
    // Dense bind for block 0; MoE blocks load ff.* via JoyOV2MoEFeedForward.
    if (!layers_.empty()) {
      layers_[0]->load_state_dict(
          state_dict.get_dict_with_prefix("transformer_blocks.0."));
    }
    for (size_t i = 1; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(state_dict.get_dict_with_prefix(
          "transformer_blocks." + std::to_string(i) + "."));
    }
  }

  static torch::Tensor timestep_embedding(const torch::Tensor& timesteps,
                                          int64_t dim) {
    const int64_t half = dim / 2;
    auto freqs = torch::exp(
        -std::log(10000.0) *
        torch::arange(half, timesteps.options().dtype(torch::kFloat)) /
        static_cast<double>(half));
    auto args = timesteps.to(torch::kFloat).unsqueeze(1) * freqs.unsqueeze(0);
    return torch::cat({torch::cos(args), torch::sin(args)}, /*dim=*/-1);
  }

  // Interleaved 3D MRoPE → [S, head_dim] = cat(cos, sin)
  torch::Tensor compute_freqs_cis(const torch::Tensor& position_ids) const {
    CHECK_EQ(position_ids.size(0), 3);
    const int64_t half = head_dim_ / 2;
    auto device = position_ids.device();
    auto inv_freq =
        1.0 /
        torch::pow(static_cast<double>(rope_theta_),
                   torch::arange(0, head_dim_, 2, device).to(torch::kFloat) /
                       static_cast<double>(head_dim_));
    auto freqs = position_ids.to(torch::kFloat).unsqueeze(-1) *
                 inv_freq.view({1, 1, -1});
    auto cos = freqs.cos();
    auto sin = freqs.sin();
    auto pack = [&](torch::Tensor x) {
      auto result = x[0].clone();
      const int64_t sec_h = mrope_section_.size() > 1 ? mrope_section_[1] : 0;
      const int64_t sec_w = mrope_section_.size() > 2 ? mrope_section_[2] : 0;
      for (int64_t i = 0; i < sec_h; ++i) {
        const int64_t idx = 1 + i * 3;
        if (idx < half) {
          result.select(/*dim=*/-1, idx) = x[1].select(/*dim=*/-1, idx);
        }
      }
      for (int64_t i = 0; i < sec_w; ++i) {
        const int64_t idx = 2 + i * 3;
        if (idx < half) {
          result.select(/*dim=*/-1, idx) = x[2].select(/*dim=*/-1, idx);
        }
      }
      return result;
    };
    return torch::cat({pack(cos), pack(sin)}, /*dim=*/-1);
  }

  // Optional before/after callbacks index blocks in forward order
  // (context_text_refiner → noise_refiner → audio_refiner → layers_), matching
  // get_block_weight_loaders() for DitRollingLoadManager wait/schedule.
  std::pair<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& text_embeddings,
      const torch::Tensor& timestep,
      const JoyOV2PackedRopeMeta& rope_meta,
      const std::optional<torch::Tensor>& audio_hidden_states,
      std::function<void(int32_t)> before_layer_cb = nullptr,
      std::function<void(int32_t)> after_layer_cb = nullptr) {
    CHECK(rope_meta.mix.has_value());
    CHECK(rope_meta.pixel.has_value());

    int32_t rolling_idx = 0;
    auto invoke_before = [&]() {
      if (before_layer_cb) {
        before_layer_cb(rolling_idx);
      }
    };
    auto invoke_after = [&]() {
      if (after_layer_cb) {
        after_layer_cb(rolling_idx);
      }
      ++rolling_idx;
    };

    auto t_emb =
        timestep_embedding(timestep * 1000.0, 256).to(options_.dtype());
    auto timestep_emb =
        time_linear_2_->forward(torch::silu(time_linear_1_->forward(t_emb)));

    auto pixel_emb = x_embedder_->forward(hidden_states);
    auto text_in = text_embeddings;
    auto text_emb =
        context_proj_->forward(std::get<0>(context_norm_->forward(text_in)));

    torch::Tensor audio_emb;
    if (audio_hidden_states.has_value() && audio_hidden_states->defined()) {
      audio_emb = audio_embedder_->forward(*audio_hidden_states);
    }

    // Refine text (no modulation). Refiners stay full-sequence (no Ulysses).
    auto text_freqs = compute_freqs_cis(rope_meta.text.position_id);
    const auto device = hidden_states.device();
    {
      auto x = text_emb.unsqueeze(0);
      for (size_t i = 0; i < context_text_refiner_layers_.size(); ++i) {
        invoke_before();
        x = run_block(context_text_refiner_layers_[i],
                      x,
                      /*timestep_emb=*/std::nullopt,
                      text_freqs,
                      rope_meta.text.seqlens,
                      /*ulysses_sp=*/false);
        invoke_after();
      }
      text_emb = x.squeeze(0);
    }
    // Refine pixel
    auto pixel_freqs = compute_freqs_cis(rope_meta.pixel->position_id);
    {
      auto x = pixel_emb.unsqueeze(0);
      for (size_t i = 0; i < noise_refiner_layers_.size(); ++i) {
        invoke_before();
        x = run_block(noise_refiner_layers_[i],
                      x,
                      timestep_emb,
                      pixel_freqs,
                      rope_meta.pixel->seqlens,
                      /*ulysses_sp=*/false);
        invoke_after();
      }
      pixel_emb = x.squeeze(0);
    }
    if (audio_emb.defined() && rope_meta.audio.has_value()) {
      auto audio_freqs = compute_freqs_cis(rope_meta.audio->position_id);
      auto x = audio_emb.unsqueeze(0);
      for (size_t i = 0; i < audio_refiner_layers_.size(); ++i) {
        invoke_before();
        x = run_block(audio_refiner_layers_[i],
                      x,
                      timestep_emb,
                      audio_freqs,
                      rope_meta.audio->seqlens,
                      /*ulysses_sp=*/false);
        invoke_after();
      }
      audio_emb = x.squeeze(0);
    } else if (before_layer_cb || after_layer_cb) {
      // Keep rolling indices aligned with get_block_weight_loaders().
      for (size_t i = 0; i < audio_refiner_layers_.size(); ++i) {
        invoke_before();
        invoke_after();
      }
    }
#if defined(USE_NPU)
    {
      const aclError sync_err = aclrtSynchronizeDevice();
      CHECK_EQ(sync_err, ACL_SUCCESS) << "NPU error after refiners";
    }
#endif

    // Pack [text | pixel | audio] per sample. Copy seqlens to CPU first so a
    // failed .item() is not an unlogged device abort.
    const auto& mix = *rope_meta.mix;
    auto text_lens =
        rope_meta.text.seqlens.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto pixel_lens =
        rope_meta.pixel->seqlens.to(torch::kCPU).to(torch::kInt64).contiguous();
    torch::Tensor audio_lens;
    if (audio_emb.defined() && rope_meta.audio.has_value()) {
      audio_lens = rope_meta.audio->seqlens.to(torch::kCPU)
                       .to(torch::kInt64)
                       .contiguous();
    }
    const int64_t n = text_lens.numel();
    CHECK_EQ(pixel_lens.numel(), n);
    std::vector<torch::Tensor> pieces;
    pieces.reserve(static_cast<size_t>(n));
    int64_t text_off = 0;
    int64_t pix_off = 0;
    int64_t aud_off = 0;
    const int64_t pix_per = pixel_lens[0].item<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
      const int64_t t_len = text_lens[i].item<int64_t>();
      auto t_slice = text_emb.slice(/*dim=*/0, text_off, text_off + t_len);
      text_off += t_len;
      auto p_slice = pixel_emb.slice(/*dim=*/0, pix_off, pix_off + pix_per);
      pix_off += pix_per;
      if (audio_emb.defined()) {
        CHECK_EQ(audio_lens.numel(), n);
        const int64_t a_len = audio_lens[i].item<int64_t>();
        auto a_slice = audio_emb.slice(/*dim=*/0, aud_off, aud_off + a_len);
        aud_off += a_len;
        pieces.push_back(torch::cat({t_slice, p_slice, a_slice}, /*dim=*/0));
      } else {
        pieces.push_back(torch::cat({t_slice, p_slice}, /*dim=*/0));
      }
    }
    auto x = torch::cat(pieces, /*dim=*/0).unsqueeze(0);
    auto mix_freqs = compute_freqs_cis(mix.position_id);

    const bool use_sp = sp_world_ > 1 && dit_sp_group_ != nullptr;
    torch::Tensor mix_attn_seqlens = mix.seqlens;
    if (use_sp) {
      // JoyO-only: pad length must also be reflected in the last sample's
      // attn seqlens (varlen SDPA). Mixin pads/scatters tensors; seqlens
      // stay on the host side of the boundary.
      const int64_t s_orig = x.size(1);
      const int64_t pad = (sp_world_ - (s_orig % sp_world_)) % sp_world_;
      auto mix_lens_cpu = mix.seqlens.to(torch::kCPU).to(torch::kInt64);
      auto sample_ids =
          torch::repeat_interleave(
              torch::arange(n, torch::TensorOptions().dtype(torch::kLong)),
              mix_lens_cpu,
              /*dim=*/0)
              .to(device);
      if (pad > 0) {
        auto attn_lens = mix_lens_cpu.clone();
        attn_lens[-1] += pad;
        mix_attn_seqlens = attn_lens.to(torch::kInt32).to(device);
        // RoPE is applied after Ulysses all2all on the full sequence; pad
        // only, do not scatter with x / sample_ids.
        mix_freqs = torch::nn::functional::pad(
            mix_freqs,
            torch::nn::functional::PadFuncOptions({0, 0, 0, pad})
                .mode(torch::kConstant));
      }

      dit::SequenceParallelTensorMap sp_outputs = sequence_parallel_forward(
          {{"x", {x, /*sequence_dim=*/1}},
           {"sample_ids", {sample_ids.unsqueeze(0), /*sequence_dim=*/1}}},
          [this,
           &timestep_emb,
           &mix,
           &mix_attn_seqlens,
           &mix_freqs,
           &invoke_before,
           &invoke_after](const dit::SequenceParallelTensorMap& local_inputs) {
            torch::Tensor local_x = local_inputs.at("x").first;
            std::optional<torch::Tensor> local_sample_ids =
                local_inputs.at("sample_ids").first.squeeze(0);
            for (size_t i = 0; i < layers_.size(); ++i) {
              invoke_before();
              local_x = run_block(layers_[i],
                                  local_x,
                                  timestep_emb,
                                  mix_freqs,
                                  mix.seqlens,
                                  /*ulysses_sp=*/true,
                                  mix_attn_seqlens,
                                  local_sample_ids);
              invoke_after();
            }
            return dit::SequenceParallelTensorMap{
                {"x", {local_x, /*sequence_dim=*/1}}};
          });
      x = sp_outputs.at("x").first;
    } else {
      for (size_t i = 0; i < layers_.size(); ++i) {
        invoke_before();
        x = run_block(layers_[i],
                      x,
                      timestep_emb,
                      mix_freqs,
                      mix.seqlens,
                      /*ulysses_sp=*/false,
                      mix_attn_seqlens,
                      /*token_sample_ids=*/std::nullopt);
        invoke_after();
      }
    }

    std::vector<torch::Tensor> pix_outs;
    std::vector<torch::Tensor> aud_outs;
    int64_t off = 0;
    for (int64_t i = 0; i < n; ++i) {
      const int64_t t_len = rope_meta.text.seqlens[i].item<int64_t>();
      off += t_len;
      pix_outs.push_back(x.select(0, 0).slice(/*dim=*/0, off, off + pix_per));
      off += pix_per;
      if (audio_emb.defined()) {
        const int64_t a_len = rope_meta.audio->seqlens[i].item<int64_t>();
        aud_outs.push_back(x.select(0, 0).slice(/*dim=*/0, off, off + a_len));
        off += a_len;
      }
    }
    auto pix_h = torch::cat(pix_outs, /*dim=*/0);
    auto pixel_vel =
        proj_out_linear_->forward(std::get<0>(proj_out_norm_->forward(pix_h)));

    torch::Tensor audio_vel;
    if (!aud_outs.empty()) {
      auto aud_h = torch::cat(aud_outs, /*dim=*/0);
      audio_vel = audio_proj_out_linear_->forward(
          std::get<0>(audio_proj_out_norm_->forward(aud_h)));
    }
    return {pixel_vel, audio_vel};
  }

  int64_t latent_channels() const { return latent_channels_; }
  int64_t text_dim() const { return text_dim_; }

 private:
  torch::TensorOptions options_;
  ParallelArgs parallel_args_;
  QuantArgs quant_args_;
  ProcessGroup* dit_sp_group_ = nullptr;
  int64_t sp_world_ = 1;

  int64_t hidden_size_ = 6144;
  int64_t num_attention_heads_ = 48;
  int64_t num_kv_heads_ = 16;
  int64_t head_dim_ = 128;
  int64_t num_layers_ = 51;
  int64_t num_encoder_layers_ = 2;
  int64_t num_experts_ = 48;
  int64_t moe_hidden_size_ = 4096;
  int64_t share_expert_dim_ = 8192;
  int64_t ffn_hidden_size_ = 16384;
  int64_t latent_channels_ = 128;
  int64_t text_dim_ = 5120;
  float rope_theta_ = 10000.0f;
  std::vector<int64_t> mrope_section_{16, 24, 24};
  double norm_eps_ = 1e-6;
  bool sandwich_norm_ = true;
  float top_p_ = 0.125f;
  float routed_scaling_factor_ = 10.0f;

  layer::ColumnParallelLinear x_embedder_{nullptr};
  layer::ColumnParallelLinear audio_embedder_{nullptr};
  layer::RMSNorm context_norm_{nullptr};
  layer::ColumnParallelLinear context_proj_{nullptr};
  layer::ColumnParallelLinear time_linear_1_{nullptr};
  layer::ColumnParallelLinear time_linear_2_{nullptr};
  layer::RMSNorm proj_out_norm_{nullptr};
  layer::ColumnParallelLinear proj_out_linear_{nullptr};
  layer::RMSNorm audio_proj_out_norm_{nullptr};
  layer::ColumnParallelLinear audio_proj_out_linear_{nullptr};

  torch::nn::ModuleList noise_refiner_{nullptr};
  torch::nn::ModuleList context_text_refiner_{nullptr};
  torch::nn::ModuleList audio_refiner_{nullptr};
  torch::nn::ModuleList transformer_blocks_{nullptr};
  std::vector<JoyOV2TransformerBlock> noise_refiner_layers_;
  std::vector<JoyOV2TransformerBlock> context_text_refiner_layers_;
  std::vector<JoyOV2TransformerBlock> audio_refiner_layers_;
  std::vector<JoyOV2TransformerBlock> layers_;
#if defined(USE_NPU)
  bool rolling_prepared_ = false;
#endif
};
TORCH_MODULE(JoyOV2Transformer3DModel);

REGISTER_MODEL_ARGS(JoyOV2Transformer3DModel, [&] {
  LOAD_ARG_OR(model_type, "_class_name", "JoyOV2Transformer3DModel");
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(hidden_size, "hidden_size", 6144);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 48);
  LOAD_ARG_OR(n_kv_heads, "num_kv_heads", 16);
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(n_layers, "num_layers", 51);
  LOAD_ARG_OR(num_layers, "num_layers", 51);
  LOAD_ARG_OR(num_encoder_layers, "num_encoder_layers", 2);
  LOAD_ARG_OR(num_experts, "num_experts", 48);
  LOAD_ARG_OR(moe_hidden_size, "moe_hidden_size", 4096);
  LOAD_ARG_OR(moe_intermediate_size, "moe_hidden_size", 4096);
  LOAD_ARG_OR(share_expert_dim, "share_expert_dim", 8192);
  LOAD_ARG_OR(ffn_dim, "ffn_hidden_size", 16384);
  LOAD_ARG_OR(latent_channels, "latent_channels", 128);
  LOAD_ARG_OR(text_dim, "text_dim", 5120);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(rope_theta_dit, "rope_theta", 10000);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 8192);
  LOAD_ARG_OR(rms_norm_eps, "norm_eps", 1e-6);
  LOAD_ARG_OR(sandwich_norm, "sandwich_norm", true);
  LOAD_ARG_OR(dit_top_p, "top_p", 0.125f);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 10.0f);
  // mrope is always interleaved in compute_freqs_cis; no ModelArgs flag.
  LOAD_ARG_OR(
      mrope_section, "mrope_section", (std::vector<int64_t>{16, 24, 24}));
});

}  // namespace xllm
