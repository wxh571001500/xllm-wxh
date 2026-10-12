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

#include "core/layers/mlu/dflash2_context_kv.h"

#include "core/kernels/ops_api.h"
#include "core/layers/common/rotary_embedding.h"

namespace xllm::layer {

class DFlash2ContextKV::Impl final {
 public:
  explicit Impl(const ModelContext& context) {
    const ModelArgs& args = context.get_model_args();
    const ParallelArgs& parallel = context.get_parallel_args();
    tensor_options_ = context.get_tensor_options();
    num_layers_ = args.n_layers();
    hidden_size_ = args.hidden_size();
    head_dim_ = args.head_dim();
    rms_norm_eps_ = args.rms_norm_eps();
    CHECK_GT(num_layers_, 0);
    CHECK_GT(hidden_size_, 0);
    CHECK_GT(head_dim_, 0);
    CHECK_GT(parallel.dp_size(), 0);
    CHECK_GT(parallel.cp_size(), 0);
    const int32_t replicas = parallel.dp_size() * parallel.cp_size();
    CHECK_GT(parallel.world_size(), 0);
    CHECK_EQ(parallel.world_size() % replicas, 0);
    const int32_t tp_size = parallel.world_size() / replicas;
    const int32_t tp_rank = parallel.rank() % tp_size;
    const int64_t kv_heads = args.n_kv_heads().value_or(args.n_heads());
    CHECK_GT(kv_heads, 0);
    CHECK_GT(args.n_heads(), 0);
    CHECK_EQ(args.n_heads() % tp_size, 0);
    CHECK_GE(parallel.rank(), 0);
    CHECK_LT(parallel.rank(), parallel.world_size());
    // Match QKVParallelLinear: adjacent TP ranks replicate a complete KV head.
    if (kv_heads >= tp_size) {
      CHECK_EQ(kv_heads % tp_size, 0);
      kv_shard_count_ = tp_size;
      kv_shard_rank_ = tp_rank;
      local_kv_heads_ = kv_heads / tp_size;
    } else {
      CHECK_EQ(tp_size % kv_heads, 0);
      kv_shard_count_ = static_cast<int32_t>(kv_heads);
      kv_shard_rank_ = tp_rank / (tp_size / kv_shard_count_);
      local_kv_heads_ = 1;
    }
    rotary_embedding_ =
        std::make_shared<RotaryEmbeddingImpl>(head_dim_,
                                              args.max_position_embeddings(),
                                              args.rope_theta(),
                                              /*interleaved=*/false,
                                              tensor_options_);
  }
  void load_state_dict(const StateDict& state_dict) {
    const int32_t num_layers = num_layers_;
    if (per_layer_k_proj_.empty()) {
      per_layer_k_proj_.resize(num_layers);
      per_layer_v_proj_.resize(num_layers);
      per_layer_k_norm_.resize(num_layers);
    }
    for (int32_t i = 0; i < num_layers; ++i) {
      StateDict layer_dict =
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + ".");
      torch::Tensor k_proj =
          layer_dict.get_sharded_tensor("self_attn.k_proj.weight",
                                        /*dim=*/0,
                                        kv_shard_rank_,
                                        kv_shard_count_);
      torch::Tensor v_proj =
          layer_dict.get_sharded_tensor("self_attn.v_proj.weight",
                                        /*dim=*/0,
                                        kv_shard_rank_,
                                        kv_shard_count_);
      torch::Tensor k_norm = layer_dict.get_tensor("self_attn.k_norm.weight");
      if (k_proj.defined()) {
        CHECK_EQ(
            k_proj.sizes(),
            torch::IntArrayRef({local_kv_heads_ * head_dim_, hidden_size_}));
        per_layer_k_proj_[i] = k_proj.to(tensor_options_);
      }
      if (v_proj.defined()) {
        CHECK_EQ(
            v_proj.sizes(),
            torch::IntArrayRef({local_kv_heads_ * head_dim_, hidden_size_}));
        per_layer_v_proj_[i] = v_proj.to(tensor_options_);
      }
      if (k_norm.defined()) {
        CHECK_EQ(k_norm.sizes(), torch::IntArrayRef({head_dim_}));
        per_layer_k_norm_[i] = k_norm.to(tensor_options_).to(torch::kFloat32);
      }
    }
  }

  void verify_loaded_weights() const {
    const int32_t num_layers = num_layers_;
    CHECK_EQ(static_cast<int32_t>(per_layer_k_proj_.size()), num_layers);
    CHECK_GT(local_kv_heads_, 0);
    for (int32_t i = 0; i < num_layers; ++i) {
      CHECK(per_layer_k_proj_[i].defined());
      CHECK(per_layer_v_proj_[i].defined());
      CHECK(per_layer_k_norm_[i].defined());
    }
  }

  void finalize_loaded_weights() {
    verify_loaded_weights();
    const int32_t num_layers = num_layers_;
    std::vector<torch::Tensor> kv_weights;
    std::vector<torch::Tensor> k_norm_weights;
    kv_weights.reserve(num_layers * 2);
    k_norm_weights.reserve(num_layers);
    for (int32_t i = 0; i < num_layers; ++i) {
      kv_weights.emplace_back(per_layer_k_proj_[i]);
      kv_weights.emplace_back(per_layer_v_proj_[i]);
      k_norm_weights.emplace_back(per_layer_k_norm_[i]);
    }
    fused_kv_weight_ = torch::cat(kv_weights, /*dim=*/0).contiguous();
    k_norm_weight_ =
        torch::stack(k_norm_weights, /*dim=*/0).view({num_layers, 1, 1, -1});
    per_layer_k_proj_.clear();
    per_layer_v_proj_.clear();
    per_layer_k_norm_.clear();
  }

  bool write(const torch::Tensor& projected_hidden,
             const torch::Tensor& positions,
             const torch::Tensor& device_cache_slots,
             std::vector<KVCache>& kv_caches,
             const ModelInputParams& input_params) const {
    const int64_t num_layers = num_layers_;
    CHECK_EQ(static_cast<int64_t>(kv_caches.size()), num_layers);
    CHECK(fused_kv_weight_.defined())
        << "DFlash2 context weights are not finalized.";
    CHECK_EQ(projected_hidden.dim(), 2);
    CHECK_EQ(projected_hidden.size(1), hidden_size_);
    CHECK_EQ(positions.dim(), 1);
    CHECK_EQ(positions.numel(), projected_hidden.size(0));
    CHECK_EQ(device_cache_slots.dim(), 1);
    CHECK_EQ(device_cache_slots.numel(), projected_hidden.size(0));
    CHECK(positions.device() == projected_hidden.device());
    CHECK(device_cache_slots.device() == projected_hidden.device());
    CHECK(positions.scalar_type() == torch::kInt32 ||
          positions.scalar_type() == torch::kInt64);
    CHECK(device_cache_slots.scalar_type() == torch::kInt32 ||
          device_cache_slots.scalar_type() == torch::kInt64);
    for (const KVCache& cache : kv_caches) {
      const torch::Tensor key_cache = cache.get_k_cache();
      const torch::Tensor value_cache = cache.get_v_cache();
      CHECK(!cache.get_k_cache_scale().has_value() &&
            !cache.get_v_cache_scale().has_value())
          << "DFlash2 context KV requires an unquantized draft cache.";
      CHECK_EQ(key_cache.dim(), 4);
      CHECK_EQ(value_cache.sizes(), key_cache.sizes());
      constexpr int64_t kHeadAxis = 1;
      CHECK_EQ(key_cache.size(kHeadAxis), local_kv_heads_);
      CHECK_EQ(key_cache.size(3), head_dim_);
      CHECK_EQ(key_cache.scalar_type(), projected_hidden.scalar_type());
      CHECK_EQ(value_cache.scalar_type(), projected_hidden.scalar_type());
      CHECK(key_cache.device() == projected_hidden.device());
      CHECK(value_cache.device() == projected_hidden.device());
    }

    // Temporary storage is O(L * N * local_kv_heads * head_dim), where N is
    // this write's token count, not the accumulated context length. On MLU590,
    // a standalone BF16 write with L=5, N=8192, TP=8, local_kv_heads=1 and
    // head_dim=128 peaked at 90.5 MiB above preallocated weights, inputs and
    // caches. Reassess this workspace when increasing the per-rank batch or
    // KV head count; the fused projection and FP32 norm span all draft layers.
    const int64_t num_context = projected_hidden.size(0);
    torch::Tensor all_kv =
        torch::nn::functional::linear(projected_hidden, fused_kv_weight_)
            .view({num_context, num_layers, 2, local_kv_heads_, head_dim_})
            .permute({2, 1, 0, 3, 4})
            .contiguous();
    torch::Tensor all_key =
        apply_k_norm(all_kv.select(/*dim=*/0, /*index=*/0), k_norm_weight_);
    torch::Tensor all_value = all_kv.select(/*dim=*/0, /*index=*/1);
    torch::Tensor flat_key =
        all_key.reshape({num_layers * num_context, local_kv_heads_, head_dim_});
    flat_key = apply_rope(flat_key, positions.repeat({num_layers}));
    all_key =
        flat_key.view({num_layers, num_context, local_kv_heads_, head_dim_});

    for (int64_t i = 0; i < num_layers; ++i) {
      kernel::ReshapePagedCacheParams params;
      params.key = all_key[i];
      params.value = all_value[i];
      params.k_cache = kv_caches[i].get_k_cache();
      params.v_cache = kv_caches[i].get_v_cache();
      params.slot_mapping = device_cache_slots;
      kernel::reshape_paged_cache(params);
      const bool recorded =
          input_params.record_layer(static_cast<uint32_t>(i), all_key.device());
      if (!recorded) {
        return false;
      }
    }
    return true;
  }

 private:
  torch::Tensor apply_k_norm(const torch::Tensor& key,
                             const torch::Tensor& weight) const {
    torch::Tensor key_fp32 = key.to(torch::kFloat32);
    torch::Tensor variance = key_fp32.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    return (key_fp32 * torch::rsqrt(variance + rms_norm_eps_) * weight)
        .to(key.scalar_type());
  }

  torch::Tensor apply_rope(const torch::Tensor& key,
                           const torch::Tensor& positions) const {
    CHECK(rotary_embedding_ != nullptr);
    torch::Tensor rotated_key = key.clone();
    const torch::Tensor cu_query_lens = torch::tensor(
        {int64_t{0}, key.size(0)}, positions.options().dtype(torch::kInt32));
    rotary_embedding_->forward(rotated_key,
                               positions.to(torch::kInt32),
                               cu_query_lens,
                               key.size(0),
                               /*is_prompt=*/false);
    return rotated_key;
  }

  std::vector<torch::Tensor> per_layer_k_proj_;
  std::vector<torch::Tensor> per_layer_v_proj_;
  std::vector<torch::Tensor> per_layer_k_norm_;
  torch::Tensor fused_kv_weight_;
  torch::Tensor k_norm_weight_;
  torch::TensorOptions tensor_options_;
  std::shared_ptr<RotaryEmbeddingImpl> rotary_embedding_;
  int32_t num_layers_ = 0;
  int64_t hidden_size_ = 0;
  int64_t head_dim_ = 0;
  int64_t local_kv_heads_ = 0;
  double rms_norm_eps_ = 1e-6;
  int32_t kv_shard_rank_ = 0;
  int32_t kv_shard_count_ = 1;
};

DFlash2ContextKV::DFlash2ContextKV(const ModelContext& context)
    : impl_(std::make_unique<Impl>(context)) {}
DFlash2ContextKV::~DFlash2ContextKV() = default;
void DFlash2ContextKV::load_state_dict(const StateDict& state_dict) {
  impl_->load_state_dict(state_dict);
}
void DFlash2ContextKV::verify_loaded_weights() const {
  impl_->verify_loaded_weights();
}
void DFlash2ContextKV::finalize_loaded_weights() {
  impl_->finalize_loaded_weights();
}
bool DFlash2ContextKV::write(const torch::Tensor& hidden,
                             const torch::Tensor& positions,
                             const torch::Tensor& cache_slots,
                             std::vector<KVCache>& kv_caches,
                             const ModelInputParams& input_params) const {
  return impl_->write(hidden, positions, cache_slots, kv_caches, input_params);
}

}  // namespace xllm::layer
