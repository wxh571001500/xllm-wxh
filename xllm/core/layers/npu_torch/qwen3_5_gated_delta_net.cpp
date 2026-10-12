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

#include "core/layers/npu_torch/qwen3_5_gated_delta_net.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "core/kernels/npu/xllm_ops/mega_gdn_constants.h"
#include "core/kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace {

// MegaGdnDecode and MegaGdnMtpDecode accept at most 32 sequences per call.
constexpr int64_t kMegaGdnMaxDecodeBatchSize = 32;

void check_tensor(const torch::Tensor& tensor,
                  const char* name,
                  torch::ScalarType dtype,
                  int64_t dim,
                  const torch::Device& device) {
  CHECK(tensor.defined()) << name << " must be defined.";
  CHECK_EQ(tensor.scalar_type(), dtype) << name << " has an invalid dtype.";
  CHECK_EQ(tensor.dim(), dim) << name << " has an invalid rank.";
  CHECK_EQ(tensor.device(), device) << name << " must be on the input device.";
}

torch::Tensor graph_safe_indices(const torch::Tensor& indices,
                                 const std::vector<int32_t>& host_values,
                                 int64_t batch_size,
                                 const torch::Device& device,
                                 const char* name) {
  if (!indices.defined()) {
    return qwen3_5_gdn_internal::slots_to_device(host_values, device);
  }
  check_tensor(indices, name, torch::kInt, 1, device);
  CHECK_EQ(indices.numel(), batch_size) << name << " must be sequence-scoped.";
  CHECK(indices.is_contiguous()) << name << " must be contiguous.";
  return indices;
}

bool is_supported_prefill_head_count(int64_t heads) {
  switch (heads) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 6:
    case 8:
    case 12:
    case 16:
    case 24:
    case 32:
    case 48:
    case 64:
      return true;
    default:
      return false;
  }
}

void check_decode_head_geometry(int64_t key_heads, int64_t value_heads) {
  CHECK_GE(key_heads, 1)
      << "MegaGdn decode operators require at least one local key head.";
  CHECK_LE(key_heads, 16)
      << "MegaGdn decode operators support at most 16 local key heads.";
  CHECK_EQ(key_heads & (key_heads - 1), 0)
      << "MegaGdn decode operators require a power-of-two local key-head "
         "count.";
  CHECK_EQ(value_heads % key_heads, 0)
      << "MegaGdn decode operators require NV to be divisible by NK.";
  const int64_t value_heads_per_key = value_heads / key_heads;
  CHECK_GE(value_heads_per_key, 1);
  CHECK_LE(value_heads_per_key, 4)
      << "MegaGdn decode operators support at most four value heads per key "
         "head.";
}

int64_t compute_num_matrices(const std::vector<int32_t>& query_lengths,
                             int64_t num_value_heads) {
  int64_t num_chunks = 0;
  for (const int32_t query_length : query_lengths) {
    CHECK_GT(query_length, 0) << "prefill query lengths must be positive.";
    num_chunks += (static_cast<int64_t>(query_length) +
                   kernel::npu::kMegaGdnChunkSize - 1) /
                  kernel::npu::kMegaGdnChunkSize;
  }
  return num_chunks * num_value_heads;
}

}  // namespace

Qwen3_5GatedDeltaNetImpl::Qwen3_5GatedDeltaNetImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : Qwen3NextGatedDeltaNetImpl(args,
                                 quant_args,
                                 parallel_args,
                                 options,
                                 /*init_projections=*/false) {
  in_proj_qkv_ = register_module("in_proj_qkv",
                                 ColumnParallelLinear(args.hidden_size(),
                                                      k_size_ * 2 + v_size_,
                                                      /*bias=*/false,
                                                      /*gather_output=*/false,
                                                      quant_args,
                                                      parallel_args.tp_group_,
                                                      options));
  in_proj_z_ = register_module("in_proj_z",
                               ColumnParallelLinear(args.hidden_size(),
                                                    v_size_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
  in_proj_b_ = register_module("in_proj_b",
                               ColumnParallelLinear(args.hidden_size(),
                                                    num_v_heads_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
  in_proj_a_ = register_module("in_proj_a",
                               ColumnParallelLinear(args.hidden_size(),
                                                    num_v_heads_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  if (attn_metadata.is_dummy) {
    return torch::zeros_like(hidden_states);
  }

  const FlashComm1Context* fc1_context = get_current_flash_comm1_context();
  torch::Tensor gathered_hidden_states = hidden_states;
  if (fc1_context != nullptr && is_sequence_sharded(*fc1_context)) {
    gathered_hidden_states = gather_sequence(hidden_states, *fc1_context);
  }
  const int64_t original_num_tokens = gathered_hidden_states.size(0);
  const bool use_spec_verify = input_params.is_spec_verify;
  const bool is_any_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  torch::Tensor mixed_qkv;
  torch::Tensor z;
  torch::Tensor b;
  torch::Tensor a;
  int64_t batch_size = 0;
  int64_t sequence_length = 0;
  std::optional<
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
      split_inputs =
          project_split_inputs(gathered_hidden_states, attn_metadata);
  if (split_inputs.has_value()) {
    std::tie(mixed_qkv, z, b, a) = split_inputs.value();
    batch_size = mixed_qkv.size(0);
    sequence_length = mixed_qkv.size(1);
  } else {
    auto [qkvz_padded, ba_padded] =
        project_padded_inputs(gathered_hidden_states, attn_metadata);
    batch_size = qkvz_padded.size(0);
    sequence_length = qkvz_padded.size(1);
    xllm::kernel::FusedQkvzbaSplitReshapeParams split_params;
    split_params.mixed_qkvz =
        qkvz_padded.view({batch_size * sequence_length, qkvz_padded.size(-1)});
    split_params.mixed_ba =
        ba_padded.view({batch_size * sequence_length, ba_padded.size(-1)});
    split_params.num_heads_qk = static_cast<int32_t>(num_k_heads_ / tp_size_);
    split_params.num_heads_v = static_cast<int32_t>(num_v_heads_ / tp_size_);
    split_params.head_qk = static_cast<int32_t>(head_k_dim_);
    split_params.head_v = static_cast<int32_t>(head_v_dim_);
    std::tie(mixed_qkv, z, b, a) =
        xllm::kernel::fused_qkvzba_split_reshape_cat(split_params);
    mixed_qkv = mixed_qkv.view({batch_size, sequence_length, -1});
    z = z.view(
        {batch_size, sequence_length, num_v_heads_ / tp_size_, head_v_dim_});
    b = b.view({batch_size, sequence_length, num_v_heads_ / tp_size_});
    a = a.view({batch_size, sequence_length, num_v_heads_ / tp_size_});
  }

  CHECK_GT(batch_size, 0) << "GDN batch size must be positive.";
  CHECK_GT(sequence_length, 0) << "GDN sequence length must be positive.";
  CHECK_EQ(num_k_heads_ % tp_size_, 0)
      << "MegaGdn operators require key heads divisible by TP size.";
  CHECK_EQ(num_v_heads_ % tp_size_, 0)
      << "MegaGdn operators require value heads divisible by TP size.";
  const int64_t local_value_heads = num_v_heads_ / tp_size_;
  const int64_t local_conv_dim = 2 * (num_k_heads_ / tp_size_) * head_k_dim_ +
                                 local_value_heads * head_v_dim_;
  const torch::Device device = mixed_qkv.device();
  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  torch::Tensor conv_weight = conv1d_->weight();

  check_tensor(mixed_qkv, "mixed_qkv", torch::kBFloat16, 3, device);
  check_tensor(z, "z", torch::kBFloat16, 4, device);
  check_tensor(b, "b", torch::kBFloat16, 3, device);
  check_tensor(a, "a", torch::kBFloat16, 3, device);
  check_tensor(conv_weight, "conv_weight", torch::kBFloat16, 2, device);
  check_tensor(conv_cache, "conv_cache", torch::kBFloat16, 3, device);
  check_tensor(ssm_cache, "ssm_cache", torch::kFloat32, 4, device);
  check_tensor(A_log_, "A_log", torch::kFloat32, 1, device);
  check_tensor(dt_bias_, "dt_bias", torch::kFloat32, 1, device);
  check_tensor(norm_->weight(), "norm_weight", torch::kBFloat16, 1, device);
  CHECK_EQ(mixed_qkv.size(0), batch_size);
  CHECK_EQ(mixed_qkv.size(1), sequence_length);
  CHECK_EQ(mixed_qkv.size(2), local_conv_dim)
      << "mixed_qkv channel count does not match the model geometry.";
  CHECK_EQ(z.size(0), batch_size);
  CHECK_EQ(z.size(1), sequence_length);
  CHECK_EQ(b.size(0), batch_size);
  CHECK_EQ(b.size(1), sequence_length);
  CHECK_EQ(a.size(0), batch_size);
  CHECK_EQ(a.size(1), sequence_length);
  CHECK_EQ(conv_kernel_size_, 4)
      << "MegaGdn operators require a width-four causal convolution.";
  CHECK_EQ(head_k_dim_, 128)
      << "MegaGdn operators require a key head dimension of 128.";
  CHECK_EQ(head_v_dim_, 128)
      << "MegaGdn operators require a value head dimension of 128.";
  const int64_t local_key_heads = num_k_heads_ / tp_size_;
  CHECK_GE(local_key_heads, 1)
      << "MegaGdn operators require at least one local key head.";
  CHECK_EQ(local_value_heads % local_key_heads, 0)
      << "MegaGdn operators require NV to be divisible by NK.";
  CHECK_EQ(conv_weight.size(0), conv_kernel_size_)
      << "conv weight kernel width does not match the model configuration.";
  CHECK_EQ(conv_weight.size(1), local_conv_dim)
      << "conv weight channel count does not match mixed_qkv.";
  CHECK_EQ(conv_cache.size(2), local_conv_dim)
      << "conv cache channel count does not match mixed_qkv.";
  CHECK_EQ(conv_cache.device(), ssm_cache.device())
      << "GDN cache tensors must reside on the same device.";
  CHECK_EQ(ssm_cache.size(1), local_value_heads)
      << "SSM cache head count does not match the model geometry.";
  CHECK_EQ(ssm_cache.size(2), head_k_dim_)
      << "SSM cache key dimension does not match the model geometry.";
  CHECK_EQ(ssm_cache.size(3), head_v_dim_)
      << "SSM cache value dimension does not match the model geometry.";
  CHECK_EQ(z.size(2), local_value_heads);
  CHECK_EQ(z.size(3), head_v_dim_);
  CHECK_EQ(b.size(2), local_value_heads);
  CHECK_EQ(a.size(2), local_value_heads);

  const int64_t num_slots = conv_cache.size(0);
  CHECK_GT(num_slots, 0) << "GDN cache must contain at least one slot.";
  CHECK_EQ(ssm_cache.size(0) % num_slots, 0)
      << "SSM cache rows must be divisible by conv cache slots.";
  const int64_t checkpoint_stride = ssm_cache.size(0) / num_slots;
  const std::vector<int32_t>& live_slots =
      input_params.embedding.linear_state_ids;
  if (use_spec_verify || !is_any_prefill) {
    qwen3_5_gdn_internal::check_live_slots(live_slots, batch_size, num_slots);
    CHECK_EQ(
        static_cast<int64_t>(input_params.linear_state_validity_mask.size()),
        batch_size)
        << "decode linear state validity must be sequence-scoped.";
    for (int64_t i = 0; i < batch_size; ++i) {
      const int64_t validity = input_params.linear_state_validity_mask[i];
      CHECK(validity == 1 ||
            (live_slots[i] == kPaddingLinearStateId && validity == 0))
          << "decode requires a warm GDN state for every real sequence.";
    }
  }

  const torch::Tensor contiguous_conv_weight = conv_weight.contiguous();
  const torch::Tensor contiguous_a_log = A_log_.contiguous();
  const torch::Tensor contiguous_dt_bias = dt_bias_.contiguous();
  const torch::Tensor contiguous_norm_weight = norm_->weight().contiguous();

  torch::Tensor fused_norm_output;
  const auto& read_slots = input_params.embedding.linear_state_read_ids.empty()
                               ? live_slots
                               : input_params.embedding.linear_state_read_ids;
  // Route order is semantic: spec verify is also decode-shaped and must win.
  if (use_spec_verify) {
    check_decode_head_geometry(local_key_heads, local_value_heads);
    CHECK_EQ(checkpoint_stride, sequence_length)
        << "MegaGdnMtpDecode requires one checkpoint per verify token.";
    CHECK_EQ(conv_cache.size(1), sequence_length + 2)
        << "MegaGdnMtpDecode requires an S+2 conv history.";
    CHECK_EQ(static_cast<int64_t>(attn_metadata.q_seq_lens_vec.size()),
             batch_size)
        << "MTP query lengths must be sequence-scoped.";
    CHECK_EQ(static_cast<int64_t>(input_params.num_accepted_tokens_host.size()),
             batch_size)
        << "MTP accepted-token values must be sequence-scoped host data.";
    CHECK_GE(sequence_length, 2)
        << "MegaGdnMtpDecode requires at least two verify tokens.";
    CHECK_LE(sequence_length, 17)
        << "MegaGdnMtpDecode supports at most 17 verify tokens.";
    std::vector<int32_t> accepted_tokens;
    accepted_tokens.reserve(batch_size);
    int64_t valid_verify_tokens = 0;
    for (int64_t i = 0; i < batch_size; ++i) {
      const int32_t query_length = attn_metadata.q_seq_lens_vec[i];
      CHECK_GT(query_length, 0)
          << "MTP query lengths must be positive, including padded graph rows.";
      CHECK_LE(query_length, sequence_length)
          << "MTP query length exceeds the dense operator width.";
      valid_verify_tokens += query_length;
      const int64_t accepted = input_params.num_accepted_tokens_host[i];
      CHECK_GE(accepted, 1) << "MTP accepted-token count must be in [1, S].";
      CHECK_LE(accepted, sequence_length)
          << "MTP accepted-token count must be in [1, S].";
      accepted_tokens.emplace_back(static_cast<int32_t>(accepted));
    }
    CHECK_EQ(valid_verify_tokens, original_num_tokens)
        << "MTP host query lengths do not match the unpadded model input.";
    const torch::Tensor state_indices =
        graph_safe_indices(input_params.embedding.linear_state_indices,
                           live_slots,
                           batch_size,
                           device,
                           "linear_state_indices");
    const torch::Tensor accepted_indices =
        graph_safe_indices(input_params.num_accepted_tokens,
                           accepted_tokens,
                           batch_size,
                           device,
                           "num_accepted_tokens");
    const torch::Tensor read_indices =
        graph_safe_indices(input_params.embedding.linear_state_read_indices,
                           read_slots,
                           batch_size,
                           device,
                           "linear_state_read_indices");
    std::vector<torch::Tensor> chunk_outputs;
    chunk_outputs.reserve((batch_size + kMegaGdnMaxDecodeBatchSize - 1) /
                          kMegaGdnMaxDecodeBatchSize);
    for (int64_t start = 0; start < batch_size;
         start += kMegaGdnMaxDecodeBatchSize) {
      const int64_t end =
          std::min(start + kMegaGdnMaxDecodeBatchSize, batch_size);
      xllm::kernel::MegaGdnMtpDecodeParams params;
      params.qkv = mixed_qkv.slice(0, start, end).contiguous();
      params.z = z.slice(0, start, end).contiguous();
      params.b = b.slice(0, start, end).contiguous();
      params.a = a.slice(0, start, end).contiguous();
      params.conv_weight = contiguous_conv_weight;
      params.conv_state = conv_cache;
      params.a_log = contiguous_a_log;
      params.dt_bias = contiguous_dt_bias;
      params.ssm_state = ssm_cache;
      params.read_state_indices =
          read_indices.slice(0, start, end).contiguous();
      params.write_state_indices =
          state_indices.slice(0, start, end).contiguous();
      params.num_accepted_tokens =
          accepted_indices.slice(0, start, end).contiguous();
      params.norm_weight = contiguous_norm_weight;
      params.fla_ssm_state_layout = use_fla_ssm_state_layout();
      torch::Tensor chunk_output;
      std::tie(std::ignore, std::ignore, std::ignore, chunk_output) =
          xllm::kernel::mega_gdn_mtp_decode(params);
      chunk_outputs.emplace_back(std::move(chunk_output));
    }
    fused_norm_output = chunk_outputs.size() == 1
                            ? std::move(chunk_outputs.front())
                            : torch::cat(chunk_outputs, 0);
  } else if (is_any_prefill) {
    CHECK(is_supported_prefill_head_count(local_value_heads))
        << "MegaGdnPrefill does not support this local value-head count.";
    CHECK_EQ(static_cast<int64_t>(attn_metadata.q_seq_lens_vec.size()),
             batch_size)
        << "prefill query lengths must be sequence-scoped.";
    CHECK_EQ(attn_metadata.q_cu_seq_lens.numel(), batch_size + 1)
        << "prefill cu_seqlens must contain B+1 entries.";
    CHECK_EQ(conv_cache.size(1), checkpoint_stride + 2)
        << "MegaGdnPrefill conv history must be R+2.";

    torch::Tensor packed_qkv = reshape_qkvz_unpad(
        attn_metadata,
        mixed_qkv.reshape({batch_size * sequence_length, local_conv_dim}));
    torch::Tensor packed_z =
        reshape_qkvz_unpad(attn_metadata,
                           z.reshape({batch_size * sequence_length,
                                      local_value_heads * head_v_dim_}));
    torch::Tensor packed_b = reshape_qkvz_unpad(
        attn_metadata,
        b.reshape({batch_size * sequence_length, local_value_heads}));
    torch::Tensor packed_a = reshape_qkvz_unpad(
        attn_metadata,
        a.reshape({batch_size * sequence_length, local_value_heads}));
    const int64_t total_tokens = packed_qkv.size(0);
    int64_t expected_tokens = 0;
    for (const int32_t query_length : attn_metadata.q_seq_lens_vec) {
      CHECK_GT(query_length, 0);
      CHECK_LE(query_length, sequence_length);
      expected_tokens += query_length;
    }
    CHECK_EQ(total_tokens, expected_tokens)
        << "packed prefill token count does not match host sequence lengths.";

    const MegaGdnPrefillIndicesCache& indices =
        qwen3_5_gdn_internal::get_or_build_prefill_indices(
            attn_metadata,
            live_slots,
            input_params.linear_state_validity_mask,
            input_params.embedding.linear_state_read_ids,
            checkpoint_stride,
            num_slots,
            device);
    xllm::kernel::MegaGdnPrefillParams params;
    params.mixed_qkv = packed_qkv.contiguous();
    params.b = packed_b.view({total_tokens, local_value_heads}).contiguous();
    params.a = packed_a.view({total_tokens, local_value_heads}).contiguous();
    params.z = packed_z.view({total_tokens, local_value_heads, head_v_dim_})
                   .contiguous();
    params.conv_weight = contiguous_conv_weight;
    params.conv_state = conv_cache;
    params.a_log = contiguous_a_log;
    params.dt_bias = contiguous_dt_bias;
    params.conv_state_read_indices = indices.conv_read;
    params.conv_state_write_indices = indices.conv_write;
    params.ssm_state_read_indices = indices.ssm_read;
    params.ssm_state_write_indices = indices.ssm_write;
    params.ssm_cache = ssm_cache;
    params.cu_seqlens =
        attn_metadata.q_cu_seq_lens.to(torch::kInt).contiguous();
    params.norm_weight = contiguous_norm_weight;
    params.num_matrices =
        compute_num_matrices(attn_metadata.q_seq_lens_vec, local_value_heads);
    std::tie(fused_norm_output, std::ignore, std::ignore) =
        xllm::kernel::mega_gdn_prefill(params);
  } else {
    check_decode_head_geometry(local_key_heads, local_value_heads);
    CHECK_EQ(sequence_length, 1)
        << "MegaGdnDecode requires one token per sequence.";
    CHECK_EQ(checkpoint_stride, 1)
        << "MegaGdnDecode requires one SSM row per slot.";
    CHECK_EQ(conv_cache.size(1), 3)
        << "MegaGdnDecode requires a three-token conv history.";
    const torch::Tensor state_indices =
        graph_safe_indices(input_params.embedding.linear_state_indices,
                           live_slots,
                           batch_size,
                           device,
                           "linear_state_indices");
    const torch::Tensor decode_qkv =
        mixed_qkv.reshape({batch_size, local_conv_dim});
    const torch::Tensor read_indices =
        graph_safe_indices(input_params.embedding.linear_state_read_indices,
                           read_slots,
                           batch_size,
                           device,
                           "linear_state_read_indices");
    const torch::Tensor decode_z =
        z.reshape({batch_size, local_value_heads, head_v_dim_});
    const torch::Tensor decode_b = b.reshape({batch_size, local_value_heads});
    const torch::Tensor decode_a = a.reshape({batch_size, local_value_heads});
    std::vector<torch::Tensor> chunk_outputs;
    chunk_outputs.reserve((batch_size + kMegaGdnMaxDecodeBatchSize - 1) /
                          kMegaGdnMaxDecodeBatchSize);
    for (int64_t start = 0; start < batch_size;
         start += kMegaGdnMaxDecodeBatchSize) {
      const int64_t end =
          std::min(start + kMegaGdnMaxDecodeBatchSize, batch_size);
      xllm::kernel::MegaGdnDecodeParams params;
      params.qkv = decode_qkv.slice(0, start, end).contiguous();
      params.z = decode_z.slice(0, start, end).contiguous();
      params.b = decode_b.slice(0, start, end).contiguous();
      params.a = decode_a.slice(0, start, end).contiguous();
      params.conv_weight = contiguous_conv_weight;
      params.conv_state = conv_cache;
      params.a_log = contiguous_a_log;
      params.dt_bias = contiguous_dt_bias;
      params.ssm_state = ssm_cache;
      params.read_state_indices =
          read_indices.slice(0, start, end).contiguous();
      params.write_state_indices =
          state_indices.slice(0, start, end).contiguous();
      params.norm_weight = contiguous_norm_weight;
      params.fla_ssm_state_layout = use_fla_ssm_state_layout();
      torch::Tensor chunk_output;
      std::tie(std::ignore, std::ignore, std::ignore, chunk_output) =
          xllm::kernel::mega_gdn_decode(params);
      chunk_outputs.emplace_back(std::move(chunk_output));
    }
    fused_norm_output = chunk_outputs.size() == 1
                            ? std::move(chunk_outputs.front())
                            : torch::cat(chunk_outputs, 0);
  }

  CHECK(fused_norm_output.defined()) << "MegaGdn operator returned no output.";
  CHECK(fused_norm_output.dim() == 3 || fused_norm_output.dim() == 4)
      << "MegaGdn norm output has an invalid rank.";
  torch::Tensor projected_input =
      fused_norm_output.reshape({-1, local_value_heads * head_v_dim_});
  // MegaGdnMtpDecode returns dense [B,S,...]. Spec verify is represented as
  // chunked prefill by the ACL executor, so remove padding per sequence rather
  // than truncating the flattened output (which would retain an earlier row's
  // padding and drop a later row's valid tokens for ragged metadata).
  if (use_spec_verify) {
    projected_input = reshape_qkvz_unpad(attn_metadata, projected_input);
  }
  CHECK_EQ(projected_input.size(0), original_num_tokens)
      << "MegaGdn output token layout does not match the model input.";
  if (fc1_context != nullptr && is_sequence_sharded(*fc1_context)) {
    return o_proj_->forward(projected_input,
                            row_parallel_reduce_mode_for_fc1(*fc1_context));
  }
  return o_proj_->forward(projected_input);
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::merge_qkvz_from_split_activations(
    const torch::Tensor& qkv,
    const torch::Tensor& z) const {
  CHECK_EQ(qkv.dim(), 3) << "Expected qkv activation to be 3D, got "
                         << qkv.sizes();
  CHECK_EQ(z.dim(), 3) << "Expected z activation to be 3D, got " << z.sizes();
  CHECK_EQ(qkv.size(0), z.size(0)) << "qkv/z batch size mismatch.";
  CHECK_EQ(qkv.size(1), z.size(1)) << "qkv/z sequence size mismatch.";
  CHECK_EQ(qkv.size(2), (2 * k_size_ + v_size_) / tp_size_)
      << "Unexpected qkv hidden size for Qwen3.5.";
  CHECK_EQ(z.size(2), v_size_ / tp_size_)
      << "Unexpected z hidden size for Qwen3.5.";
  CHECK_GT(num_k_heads_, 0) << "linear_num_key_heads must be positive.";
  CHECK_EQ(num_v_heads_ % num_k_heads_, 0)
      << "linear_num_value_heads must be divisible by linear_num_key_heads.";

  const int64_t bs = qkv.size(0);
  const int64_t seqlen = qkv.size(1);
  const int64_t local_k_heads = num_k_heads_ / tp_size_;
  const int64_t local_v_heads = num_v_heads_ / tp_size_;
  const int64_t num_v_heads_per_k = num_v_heads_ / num_k_heads_;

  auto qkv_split = torch::split(
      qkv, {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_}, 2);
  auto q = qkv_split[0].view({bs, seqlen, local_k_heads, head_k_dim_});
  auto k = qkv_split[1].view({bs, seqlen, local_k_heads, head_k_dim_});
  auto v = qkv_split[2].view({bs, seqlen, local_v_heads, head_v_dim_});
  auto z_view = z.view({bs, seqlen, local_v_heads, head_v_dim_});

  v = v.view({bs, seqlen, local_k_heads, num_v_heads_per_k * head_v_dim_});
  z_view =
      z_view.view({bs, seqlen, local_k_heads, num_v_heads_per_k * head_v_dim_});

  return torch::cat({q, k, v, z_view}, -1).view({bs, seqlen, -1}).contiguous();
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::merge_ba_from_split_activations(
    const torch::Tensor& b,
    const torch::Tensor& a) const {
  CHECK_EQ(b.dim(), 3) << "Expected b activation to be 3D, got " << b.sizes();
  CHECK_EQ(a.dim(), 3) << "Expected a activation to be 3D, got " << a.sizes();
  CHECK_EQ(b.size(0), a.size(0)) << "b/a batch size mismatch.";
  CHECK_EQ(b.size(1), a.size(1)) << "b/a sequence size mismatch.";
  CHECK_EQ(b.size(2), num_v_heads_ / tp_size_)
      << "Unexpected b hidden size for Qwen3.5.";
  CHECK_EQ(a.size(2), num_v_heads_ / tp_size_)
      << "Unexpected a hidden size for Qwen3.5.";
  CHECK_GT(num_k_heads_, 0) << "linear_num_key_heads must be positive.";
  CHECK_EQ(num_v_heads_ % num_k_heads_, 0)
      << "linear_num_value_heads must be divisible by linear_num_key_heads.";

  const int64_t bs = b.size(0);
  const int64_t seqlen = b.size(1);
  const int64_t local_k_heads = num_k_heads_ / tp_size_;
  const int64_t num_v_heads_per_k = num_v_heads_ / num_k_heads_;

  auto b_view = b.view({bs, seqlen, local_k_heads, num_v_heads_per_k});
  auto a_view = a.view({bs, seqlen, local_k_heads, num_v_heads_per_k});
  return torch::cat({b_view, a_view}, -1).view({bs, seqlen, -1}).contiguous();
}

std::optional<
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
Qwen3_5GatedDeltaNetImpl::project_split_inputs(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata) {
  auto qkv = reshape_projected_tokens_with_pad(
      attn_metadata, in_proj_qkv_->forward(hidden_states));
  auto z_proj = reshape_projected_tokens_with_pad(
      attn_metadata, in_proj_z_->forward(hidden_states));
  auto b_proj = reshape_projected_tokens_with_pad(
      attn_metadata, in_proj_b_->forward(hidden_states));
  auto a_proj = reshape_projected_tokens_with_pad(
      attn_metadata, in_proj_a_->forward(hidden_states));

  const int64_t batch_size = qkv.size(0);
  const int64_t seq_len = qkv.size(1);
  auto z =
      z_proj.view({batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  auto b = b_proj.view({batch_size, seq_len, num_v_heads_ / tp_size_});
  auto a = a_proj.view({batch_size, seq_len, num_v_heads_ / tp_size_});
  return std::make_tuple(qkv, z, b, a);
}

void Qwen3_5GatedDeltaNetImpl::load_projection_state_dict(
    const StateDict& state_dict) {
  auto in_proj_qkv_state_dict = state_dict.get_dict_with_prefix("in_proj_qkv.");
  if (in_proj_qkv_state_dict.size() > 0 && !in_proj_qkv_->is_weight_loaded()) {
    in_proj_qkv_->load_state_dict(
        in_proj_qkv_state_dict,
        /*shard_tensor_count=*/3,
        /*shard_sizes=*/
        {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_});
  }

  auto in_proj_z_state_dict = state_dict.get_dict_with_prefix("in_proj_z.");
  if (in_proj_z_state_dict.size() > 0 && !in_proj_z_->is_weight_loaded()) {
    in_proj_z_->load_state_dict(in_proj_z_state_dict);
  }

  auto in_proj_b_state_dict = state_dict.get_dict_with_prefix("in_proj_b.");
  if (in_proj_b_state_dict.size() > 0 && !in_proj_b_->is_weight_loaded()) {
    in_proj_b_->load_state_dict(in_proj_b_state_dict);
  }

  auto in_proj_a_state_dict = state_dict.get_dict_with_prefix("in_proj_a.");
  if (in_proj_a_state_dict.size() > 0 && !in_proj_a_->is_weight_loaded()) {
    in_proj_a_->load_state_dict(in_proj_a_state_dict);
  }
}

void Qwen3_5GatedDeltaNetImpl::verify_projection_weights(
    const std::string& prefix) const {
  CHECK(in_proj_qkv_ && in_proj_qkv_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_qkv.weight";
  CHECK(in_proj_z_ && in_proj_z_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_z.weight";
  CHECK(in_proj_b_ && in_proj_b_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_b.weight";
  CHECK(in_proj_a_ && in_proj_a_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_a.weight";
}

}  // namespace layer
}  // namespace xllm
