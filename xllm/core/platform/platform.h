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

#pragma once

#include <c10/core/Device.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>

namespace xllm {

class Platform final {
 public:
  static std::string type_str();
  static torch::DeviceType type_torch();

  static constexpr bool is_cuda() {
#if defined(USE_CUDA)
    return true;
#else
    return false;
#endif
  }

  static constexpr bool is_npu() {
#if defined(USE_NPU)
    return true;
#else
    return false;
#endif
  }

  static constexpr bool is_mlu() {
#if defined(USE_MLU)
    return true;
#else
    return false;
#endif
  }

  // Performance preference between equivalent mRoPE implementations, not a
  // statement of functional support.
  static constexpr bool prefers_sliced_mrope() { return is_mlu(); }

  // Model-side CP: shard after embed, gather+restore before LM head.
  static constexpr bool uses_model_cp_sharding() {
    return is_mlu() || is_npu();
  }

  // KPool stores compressed index pages with a separate per-request tail.
  static constexpr bool uses_compressed_kpool_cache() { return is_mlu(); }

  // Under KV splitting, each NPU rank retains the full indexer cache.
  // Logical block B maps to index pages [B * split, (B + 1) * split).
  // XLLM_CP_INDEX_WRITE_MODE=replicated (the Python-side default) writes
  // every stripe's page at that natural row, so each rank's tensor is a
  // physical full replica and any single transfer writer supplies every
  // valid page. The sharded mode stores each rank's valid stripe at the
  // FIRST page of the group (row B * split): the transfer plan's page-level
  // overlap against a kv_split_size == 1 destination only ever reads page 0
  // of a source resource, so that is where the owner's page must live.
  static constexpr bool requires_dsa_indexer_cache_replication() {
    return is_npu();
  }

  // Shared DSA layers reuse the previous full layer's top-k and never write
  // indexer cache, so those layers skip indexer-page allocation.
  static constexpr bool supports_dsa_indexer_cache_elision() {
    return is_mlu() || is_npu();
  }

  // Host KV offload requires a batch memcpy provider with stream
  // synchronization support.
  static constexpr bool supports_host_kv_offload() {
    return is_npu() || is_mlu();
  }

  static constexpr bool supports_compact_host_kv_transfer() { return is_mlu(); }

  static constexpr bool supports_mtp_decode_graph_warmup() {
    return is_mlu() || is_npu();
  }

  // DFlash2 drafts may run a proposal block shorter than the trained block.
  static constexpr bool supports_dflash2_runtime_block_size() {
    return is_mlu();
  }

  static constexpr bool is_ilu() {
#if defined(USE_ILU)
    return true;
#else
    return false;
#endif
  }

  static constexpr bool is_musa() {
#if defined(USE_MUSA)
    return true;
#else
    return false;
#endif
  }

  static constexpr bool is_dcu() {
#if defined(USE_DCU)
    return true;
#else
    return false;
#endif
  }

  static int32_t device_count();
  // Returns the logical index of the device bound to the current thread.
  // Valid only after the device has been set (e.g. via Device::set_device).
  static int32_t current_device();
  static int32_t sm_count();
  static bool is_enable_pdl();
  static bool is_support_sm90a();
  static bool is_support_sm100a();
  static bool is_support_sm100f();
  static bool is_support_sm120a();

  // Initialize the per-device capability flags. Must be called once a
  // physical device index is known (typically from Device's constructor).
  static void init_capabilities(int32_t device_index);

 private:
  // Only used for cuda; cached after the first init_capabilities call.
  static int32_t sm_count_;
  static bool enable_pdl_;
  static bool support_sm90a_;
  static bool support_sm100a_;
  static bool support_sm100f_;
  static bool support_sm120a_;
};

}  // namespace xllm
