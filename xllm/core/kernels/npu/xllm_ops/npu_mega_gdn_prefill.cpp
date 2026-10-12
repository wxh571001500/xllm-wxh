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

#include <glog/logging.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

#include "acl/acl.h"
#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/mega_gdn_constants.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

// rtGetC2cCtrlAddr lives in the CANN runtime (libruntime.so, already linked),
// but its only declaration sits under experiment/runtime/runtime/rt_ffts.h,
// which is not on the default include path. Declare the stable C ABI prototype
// directly to avoid pulling the experiment header tree (and its base.h
// transitive deps) into the build.
extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace xllm::kernel::npu {

namespace {

// Resolve the FFTS control address the fused prefill operator needs for its
// AIC/AIV stage synchronization. Contract (mega_gdn_prefill_op.md): A2/A3
// (Ascend910B / 910_93) consume a real FFTS address obtained from the runtime;
// A5 (Ascend950) uses the MIX block grid + SYNCALL and must receive 0. The
// address is device-local, so resolve and cache it separately for each NPU.
std::unordered_map<int32_t, int64_t> g_ffts_addr_cache;
std::mutex g_ffts_addr_cache_mutex;

int64_t get_ffts_addr(const torch::Device& device) {
  const int32_t device_index = static_cast<int32_t>(device.index());
  std::lock_guard<std::mutex> lock(g_ffts_addr_cache_mutex);
  auto it = g_ffts_addr_cache.find(device_index);
  if (it != g_ffts_addr_cache.end()) {
    return it->second;
  }

  const c10_npu::NPUGuard device_guard(device);
  const char* soc_name = aclrtGetSocName();
  const std::string soc = soc_name == nullptr ? "" : soc_name;
  if (soc.find("950") != std::string::npos) {
    g_ffts_addr_cache.emplace(device_index, 0);
    return 0;
  }

  uint64_t ffts_addr = 0;
  uint32_t ffts_len = 0;
  const int32_t ret = rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  CHECK(ret == 0)
      << "rtGetC2cCtrlAddr failed for the GDN prefill operator on device "
      << device_index << ", ret=" << ret;
  CHECK_GT(ffts_len, 0)
      << "rtGetC2cCtrlAddr returned an empty FFTS region on device "
      << device_index;
  CHECK_NE(ffts_addr, 0)
      << "rtGetC2cCtrlAddr returned a null FFTS address on device "
      << device_index;
  const int64_t resolved_addr = static_cast<int64_t>(ffts_addr);
  g_ffts_addr_cache.emplace(device_index, resolved_addr);
  return resolved_addr;
}

// The fused prefill operator consumes three constant 128x128 matrices
// (mask_lower / mask_full / minus_identity). They are device- and
// dtype-fixed, so cache one set per device to avoid rebuilding them on every
// layer call (48 GDN layers per forward).
struct PrefillConstCache {
  torch::Tensor mask_lower;
  torch::Tensor mask_full;
  torch::Tensor minus_identity_bf16;
};

std::unordered_map<int32_t, PrefillConstCache> g_prefill_const_cache;
std::mutex g_prefill_const_cache_mutex;

PrefillConstCache get_or_create_consts(const torch::Device& device) {
  const int32_t device_index = static_cast<int32_t>(device.index());
  std::lock_guard<std::mutex> lock(g_prefill_const_cache_mutex);
  auto it = g_prefill_const_cache.find(device_index);
  if (it != g_prefill_const_cache.end()) {
    return it->second;
  }
  PrefillConstCache cache;
  cache.mask_lower = torch::tril(
      torch::ones({kMegaGdnChunkSize, kMegaGdnChunkSize},
                  torch::TensorOptions(device).dtype(torch::kFloat32)),
      /*diagonal=*/-1);
  cache.mask_full = torch::tril(
      torch::ones({kMegaGdnChunkSize, kMegaGdnChunkSize},
                  torch::TensorOptions(device).dtype(torch::kFloat32)),
      /*diagonal=*/0);
  cache.minus_identity_bf16 =
      torch::zeros({kMegaGdnChunkSize, kMegaGdnChunkSize},
                   torch::TensorOptions(device).dtype(torch::kBFloat16));
  cache.minus_identity_bf16.diagonal().fill_(-1);
  g_prefill_const_cache[device_index] = cache;
  return cache;
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> npu_mega_gdn_prefill(
    torch::Tensor& mixed_qkv,
    torch::Tensor& b,
    torch::Tensor& a,
    torch::Tensor& z,
    torch::Tensor& conv_weight,
    torch::Tensor& conv_state,
    torch::Tensor& a_log,
    torch::Tensor& dt_bias,
    torch::Tensor& conv_state_read_indices,
    torch::Tensor& conv_state_write_indices,
    torch::Tensor& ssm_state_read_indices,
    torch::Tensor& ssm_state_write_indices,
    torch::Tensor& ssm_cache,
    torch::Tensor& cu_seqlens,
    torch::Tensor& norm_weight,
    int64_t num_matrices) {
  const PrefillConstCache consts = get_or_create_consts(mixed_qkv.device());

  const int64_t total_tokens = z.size(0);
  const int64_t num_value_heads = z.size(1);
  const int64_t head_dim = z.size(2);

  // Outputs. norm_output takes z's shape; conv_state_out / ssm_cache_out are
  // updated in place by aliasing the input cache buffers (the simplest and
  // documented default; see mega_gdn_prefill_op.md "输入/输出 buffer
  // 可不同址").
  auto norm_output =
      torch::empty({total_tokens, num_value_heads, head_dim},
                   torch::TensorOptions(z.device()).dtype(torch::kBFloat16));
  torch::Tensor& conv_state_out = conv_state;
  torch::Tensor& ssm_cache_out = ssm_cache;

  // FFTS control address for AIC/AIV stage sync: real runtime address on
  // A2/A3, 0 on A5. Resolved and cached per device by get_ffts_addr(). Passing
  // 0 on A2/A3 triggers an fftsplus aicore exception at kernel launch.
  const int64_t ffts_addr = get_ffts_addr(mixed_qkv.device());

  // The argument order MUST match the autogen aclnnMegaGdnPrefillOp signature:
  // 18 inputs, then attrs (ffts_addr, num_matrices), then 3 outputs. This
  // mirrors the mega_chunk_gdn convention (inputs, attrs, outputs) and is the
  // #1 build/run verification point once the operator is compiled into the
  // autogen headers.
  EXEC_NPU_CMD(aclnnMegaGdnPrefillOp,
               mixed_qkv,
               b,
               a,
               z,
               conv_weight,
               conv_state,
               a_log,
               dt_bias,
               conv_state_read_indices,
               conv_state_write_indices,
               ssm_state_read_indices,
               ssm_state_write_indices,
               ssm_cache,
               consts.mask_lower,
               consts.mask_full,
               consts.minus_identity_bf16,
               cu_seqlens,
               norm_weight,
               ffts_addr,
               num_matrices,
               norm_output,
               conv_state_out,
               ssm_cache_out);

  return {norm_output, conv_state_out, ssm_cache_out};
}

}  // namespace xllm::kernel::npu
