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

#include "master.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>

#include "common/metrics.h"
#include "common/types.h"
#include "core/common/xllm_build_info.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "dit_master.h"
#if defined(USE_NPU)
#include "framework/parallel_state/npu_rank_table_env.h"
#endif
#include "core/platform/device_name_utils.h"
#include "framework/kv_cache/layerwise_split_layout.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "llm_engine.h"
#include "llm_master.h"
#include "models/model_registry.h"
#include "platform/platform.h"
#include "rec_engine.h"
#include "rec_master.h"
#include "runtime/options.h"
#include "speculative_engine.h"
#include "util/model_config_utils.h"
#include "util/scope_guard.h"
#include "util/timer.h"
#include "util/utils.h"
#include "vlm_engine.h"
#include "vlm_master.h"

namespace brpc {
DECLARE_bool(graceful_quit_on_sigterm);
DECLARE_bool(graceful_quit_on_sighup);
}  // namespace brpc

namespace xllm {
namespace {

void apply_runtime_kv_cache_options(const Options& source,
                                    runtime::Options& destination) {
  destination.host_blocks_factor(source.host_blocks_factor())
      .enable_kvcache_store(source.enable_kvcache_store())
      .store_protocol(source.store_protocol())
      .store_rdma_devices(source.store_rdma_devices())
      .store_master_server_address(source.store_master_server_address())
      .store_metadata_server(source.store_metadata_server())
      .store_local_hostname(source.store_local_hostname())
      .prefetch_batch_size(source.prefetch_batch_size())
      .prefetch_timeout(source.prefetch_timeout())
      .layers_wise_copy_batchs(source.layers_wise_copy_batchs())
      .kv_cache_dtype(source.kv_cache_dtype());
}

void validate_layerwise_split_size_startup_config(const Options& options,
                                                  const std::string& model_type,
                                                  int32_t global_world_size) {
  const ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const int32_t layerwise_split_size = parallel_config.layerwise_split_size();
  validate_layerwise_split_size_config(layerwise_split_size);
  if (layerwise_split_size <= 1) {
    return;
  }

  CHECK_GE(options.dp_size(), 1) << "dp_size must be >= 1.";
  CHECK_GT(global_world_size, 0) << "world_size must be > 0.";
  const int32_t dp_cp_size = options.dp_size() * options.cp_size();
  CHECK_EQ(global_world_size % dp_cp_size, 0)
      << "world_size (" << global_world_size
      << ") must be divisible by dp_size * cp_size (" << dp_cp_size << ").";
  const int32_t attn_tp_size = global_world_size / dp_cp_size;
  std::string resolved_model_type = model_type;
  if (resolved_model_type.empty()) {
    resolved_model_type =
        util::get_model_type(options.model_path(), options.backend());
  }
  validate_layerwise_split_enablement(
      layerwise_split_size, attn_tp_size, resolved_model_type);

  CHECK_LE(options.host_blocks_factor(), 1.0)
      << "layerwise_split_size > 1 does not support hierarchy host cache.";
  CHECK_EQ(parallel_config.cp_size(), 1)
      << "layerwise_split_size > 1 does not support context parallelism.";
  CHECK_EQ(parallel_config.kv_split_size_effective(), 1)
      << "layerwise_split_size > 1 does not support KV split.";
}

bool is_python_cp_compatible_graph_backend(std::string_view graph_backend) {
  std::string normalized_backend(graph_backend);
  boost::algorithm::to_lower(normalized_backend);
  return normalized_backend.empty() || normalized_backend == "off" ||
         normalized_backend == "none" || normalized_backend == "0" ||
         normalized_backend == "aclgraph";
}

}  // namespace

std::optional<std::string> validate_model_cp(const Options& options,
                                             EngineType engine_type,
                                             const std::string& model_type,
                                             int32_t global_world_size) {
  if (options.cp_size() < 1) {
    return "cp_size must be greater than or equal to 1";
  }

  if (options.cp_size() == 1) {
    return std::nullopt;
  }

  if (Platform::is_mlu()) {
    if (engine_type != EngineType::LLM && engine_type != EngineType::SSM) {
      return "MLU CP supports only LLM text generation";
    }
    if (options.task_type() != "generate") {
      return "MLU CP supports only the generate task";
    }
    if (!is_mlu_model_cp_capable(model_type)) {
      return "MLU CP does not support model_type=" + model_type;
    }

    if (global_world_size % (options.dp_size() * options.cp_size()) != 0) {
      return "MLU CP requires world_size divisible by dp_size * cp_size "
             "(orthogonal PCP x TP layout)";
    }

    if (options.dp_size() != 1) {
      return "MLU CP requires dp_size == 1";
    }

    const int32_t kv_split_size =
        ParallelConfig::get_instance().kv_split_size_effective();
    if (kv_split_size > 1 && model_type != "glm_moe_dsa") {
      return "MLU CP with kv_split_size > 1 supports only glm_moe_dsa";
    }
    const int32_t tp_size =
        global_world_size / (options.dp_size() * options.cp_size());
    const bool partitions_pcp = kv_split_size > 0 &&
                                kv_split_size <= options.cp_size() &&
                                options.cp_size() % kv_split_size == 0;
    const bool spans_pcp_and_tp = kv_split_size == options.cp_size() * tp_size;
    if (!partitions_pcp && !spans_pcp_and_tp) {
      return "MLU CP requires kv_split_size to divide cp_size or equal "
             "cp_size * tp_size";
    }

    if (options.ep_size() != global_world_size) {
      return "MLU CP requires ep_size == global world size";
    }
    return std::nullopt;
  }

  if (Platform::is_npu()) {
    if (engine_type != EngineType::LLM && engine_type != EngineType::SSM) {
      return "Model-side CP supports only LLM text generation";
    }
    if (options.task_type() != "generate") {
      return "Model-side CP supports only the generate task";
    }
    const bool is_dsv4_model = util::is_deepseek_v4_model_type(model_type);
    // glm5_next is exempted alongside deepseek_v4 only for DFlash2
    // specifically: DFlash2's aux-hidden-buffer restore under CP is proven
    // correct (glm5_next.py's cp_merge_rows on aux_hidden_buffer, mirroring
    // DeepseekV4CpContext.gather_restore) and unit-tested
    // (test_glm5_next_cp.py's
    // test_layer_capture_restores_global_rows_with_context_parallelism).
    // Eagle3/DFlash(1)/DSpark are not exempted here: their glm5_next-specific
    // dispatch paths were not individually verified (design.md §3.1 Gate B1).
    const bool is_glm5_next_dflash2 = SpeculativeConfig::is_glm5_next_dflash2(
        model_type, options.speculative_algorithm());
    if (engine_type == EngineType::SSM &&
        SpeculativeConfig::requires_aux_hidden_capture(
            options.speculative_algorithm()) &&
        !is_dsv4_model && !is_glm5_next_dflash2) {
      return "Current model-side CP does not support aux-hidden-capture "
             "speculative algorithms (Eagle3/DFlash/DSpark); run speculative "
             "decoding on a cp_size=1 Decode instance.";
    }
    // Native model-side CP is compatible with graph mode because the two are
    // phase-disjoint: CP only engages on batch_forward_type.no_decode(), while
    // ACL graph captures/replays pure decode. The Python CP path applies the
    // same phase split only to decode-only ACLGraph; other graph backends are
    // rejected below.
    //
    // The one batch that satisfies both gates is spec-verify chunked prefill.
    // The graph executor only takes it for hybrid-linear-attention models, and
    // no CP-capable model is one, so it falls back to eager today; the guard in
    // AclGraphExecutorImpl::run() keeps that true if a future model is both.
    if (options.instance_role() != InstanceRole::DEFAULT &&
        options.instance_role() != InstanceRole::PREFILL) {
      return "Model-side CP supports only DEFAULT or PREFILL roles";
    }

    // Python model executor runs a standalone torch CP path (all-gather KV,
    // eager prefill only) that does not go through the ATB fused-attention op,
    // so it bypasses the ATB-backend requirement and the ATB CP capability
    // allowlist below. The safety constraints above (LLM/generate,
    // DEFAULT/PREFILL) still apply. Orthogonal TP x CP is supported (both may
    // be > 1, sharing world = cp * tp); the collective communicator builds the
    // narrowed TP group and the strided CP group as separate torch subgroups
    // off the shared world rendezvous endpoint. GLM-5 Next also restores
    // DP-local CP rows before routed MoE, supporting the dp * cp * tp layout.
    // Other Python CP models remain restricted to DP1.
    if (ModelConfig::is_python_model_impl(
            ModelConfig::get_instance().model_impl())) {
      // Only models whose Python forward actually shards the sequence (via
      // cp_shard_rows / cp_merge_rows) may enable CP. Other Python models keep
      // a full-sequence forward, so a cp_context would be built but never
      // consumed: qwen3_5 would reach _prefill_cp with unsharded rows (garbled
      // or out-of-bounds output) and unsupported MLA models would silently
      // recompute the whole sequence on every rank. Mirror the implementations
      // that explicitly consume cp_context rather than admitting every Python
      // model type.
      static const std::unordered_set<std::string> kPythonCpCapableModels = {
          "qwen3",
          "glm_moe_dsa",
          "glm5_next",
      };
      if (kPythonCpCapableModels.find(model_type) ==
          kPythonCpCapableModels.end()) {
        return "Python model-side CP does not support model_type=" +
               model_type +
               "; supported models are qwen3, glm_moe_dsa, and glm5_next.";
      }
      // Consume the shared glm5_next+DFlash2 pairing predicate (like the
      // aux-capture gate above) so the algorithm whitelist cannot drift
      // when further algorithms are cleared. The explicit model_type
      // qualifier stays: this gate scopes a glm5_next-specific refusal,
      // and dropping the qualifier would newly refuse qwen3/glm_moe_dsa
      // SSM shapes that pass today with a glm5_next-specific message.
      if (model_type == "glm5_next" && engine_type == EngineType::SSM &&
          !is_glm5_next_dflash2) {
        return "Python GLM-5 Next CP does not support this target-side "
               "speculative algorithm; run speculation on a cp_size=1 "
               "Decode instance";
      }
      if (model_type == "glm_moe_dsa" && engine_type == EngineType::SSM &&
          SpeculativeConfig::is_mtp_algorithm(
              options.speculative_algorithm())) {
        return "Python model-side CP does not support MTP speculative "
               "verification; run MTP on a cp_size=1 Decode instance";
      }
      // On NPU, the Python executor resolves enable_graph=true with an
      // off-like backend to ACLGraph. ACLGraph handles Decode only, so Prefill
      // still runs through EagerRunner and receives cp_context.
      if (!is_python_cp_compatible_graph_backend(
              ExecutionConfig::get_instance().python_graph_backend())) {
        return "Python model-side CP requires Prefill to use EagerRunner; use "
               "--python_graph_backend=off or decode-only aclgraph";
      }
      if (options.dp_size() != 1 && model_type != "glm5_next") {
        return "Python CP requires dp_size == 1";
      }
      if (options.dp_size() == 0) {
        return "Python CP requires dp_size >= 1";
      }
      const uint64_t parallel_width = static_cast<uint64_t>(options.dp_size()) *
                                      static_cast<uint64_t>(options.cp_size());
      if (global_world_size <= 0 ||
          static_cast<uint64_t>(global_world_size) % parallel_width != 0) {
        return "Python CP requires world_size divisible by dp_size * cp_size";
      }
      const int32_t kv_split =
          ParallelConfig::get_instance().kv_split_size_effective();
      if (model_type == "glm5_next") {
        if (options.ep_size() == 0 ||
            global_world_size % options.ep_size() != 0) {
          return "Python GLM-5 Next CP requires ep_size to be a positive "
                 "divisor of world_size";
        }
        // EPLv1 may use any positive divisor. EPLv2 owns one expert shard
        // per rank, so PCP still requires ep_size == world_size.
        if (options.expert_parallel_degree().value_or(0) == 2 &&
            options.ep_size() != global_world_size) {
          return "Python GLM-5 Next EPLv2 with PCP requires ep_size "
                 "equal to world_size";
        }
        // Owner-sharded KV takes the same constraint glm_moe_dsa uses below,
        // because the constraint is a property of the sharded-KV mechanism
        // rather than of that model: PyExecutorImpl builds
        // KVShardBatchMetadata only for prefill/chunked-prefill batches, so
        // the owner-local write plus the CP gather that reconstructs the
        // logical cache -- the only path that has been validated -- exists on
        // the prefill side alone. On decode the shards would have to be
        // combined by the dcp group through SfaDcpAttentionBackend, which is
        // a different mechanism this gate does not enable. A prefill-only
        // instance is the exact condition, not "role == PREFILL": with
        // enable_disagg_pd=false the continuous scheduler runs prefill and
        // decode on the same instance whatever the role says.
        if (kv_split > 1 &&
            (!options.enable_disagg_pd() ||
             options.instance_role() != InstanceRole::PREFILL)) {
          return "Python GLM-5 Next CP with kv_split_size > 1 requires "
                 "disaggregated PD with the PREFILL role; set "
                 "enable_disagg_pd=true and instance_role=PREFILL";
        }
        const SchedulerConfig& scheduler_config =
            SchedulerConfig::get_instance();
        // No mix-batch gate: resolve_batch_mode() force-disables mixed
        // batches whenever cp_size > 1 (and again for every PD prefill
        // instance), so the raw enable_mix_batch flag -- default true -- is
        // inert for every shape this block validates. Gating on it rejected
        // default-flag CP launches over behavior the scheduler never runs.
        // Prefix caching is admitted on the same prefill-only PD shape the
        // kv_split gate above admits. A prefix-cache hit reaches the model
        // exactly like a chunked prefill's cached suffix: a batch whose
        // kv_seq_lens exceed q_seq_lens over blocks the block manager
        // already allocated, which is the prefix-extended CP forward the
        // chunked-prefill admission proved (build_cp_context's
        // kv_gather_index covers each segment's causal prefix). The
        // prefill-only scope keeps the DEFAULT role's decode-on-CP ranks
        // outside the admission: decode there reads matched blocks with no
        // CP plan, and that combination stays untested. This admission is
        // also what makes host offload configurable for glm5_next CP:
        // validate_host_cache_options requires prefix caching, so without
        // this shape a CP prefill could never set --host_blocks_factor > 1.
        if (KVCacheConfig::get_instance().enable_prefix_cache() &&
            (!options.enable_disagg_pd() ||
             options.instance_role() != InstanceRole::PREFILL)) {
          return "Python GLM-5 Next CP with enable_prefix_cache=true "
                 "requires disaggregated PD with the PREFILL role; a "
                 "standalone CP instance must set enable_prefix_cache=false";
        }
        // Schedule overlap is admitted for glm5_next CP: the overlapped
        // decode token replacement is byte-identical across CP ranks
        // (greedy argmax over CP-replicated logits, plus the CP sample-token
        // broadcast for stochastic batches in LLMWorkerImpl), the deferred
        // KDA linear-state restore stays ordered on the compute stream, and
        // the single-threaded worker task pool keeps the block lifecycle
        // FIFO. The M11.5 overlap-invariant tests are the authority.
        if (options.enable_disagg_pd() &&
            options.instance_role() != InstanceRole::PREFILL) {
          return "Python GLM-5 Next CP with disaggregated PD requires the "
                 "PREFILL role";
        }
        if (options.enable_pd_ooc()) {
          return "Python GLM-5 Next CP initially requires enable_pd_ooc=false";
        }
      }
      if (kv_split < 1 || options.cp_size() % kv_split != 0) {
        return "Python CP requires kv_split_size effective value to be a "
               "positive divisor of cp_size";
      }
      if (model_type == "glm_moe_dsa" && kv_split > 1 &&
          (!options.enable_disagg_pd() ||
           options.instance_role() != InstanceRole::PREFILL)) {
        return "Python GLM CP with kv_split_size > 1 requires disaggregated "
               "PD with the PREFILL role; set enable_disagg_pd=true and "
               "instance_role=PREFILL";
      }
      return std::nullopt;
    }

    // Require registered NPU model-side CP capability. The backend is not
    // constrained: ATB models drive CP through NpuCpPlan, while TORCH models
    // (deepseek_v4) own their CP split inside the model. Both rely on the
    // orthogonal dp * cp * attn_tp == world layout validated below.
    std::string effective_backend;
    std::string resolved_name;
    std::string resolve_error;
    // Runtime platform branches are type-checked in every platform build.
    const std::string requested_backend = options.npu_kernel_backend();
    if (!resolve_model_registration(model_type,
                                    requested_backend,
                                    &effective_backend,
                                    &resolved_name,
                                    &resolve_error)) {
      return "Model-side CP rejected model_type=" + model_type + ": " +
             resolve_error;
    }
    if (!is_npu_model_cp_capable(resolved_name)) {
      return "NPU model-side CP does not support model_type=" + model_type +
             " (resolved=" + resolved_name +
             "); only deepseek_v32, deepseek_v32_mtp, deepseek_v4, "
             "deepseek_v4_mtp, glm_moe_dsa, glm_moe_dsa_mtp are registered as "
             "CP-capable.";
    }
    if (global_world_size % (options.dp_size() * options.cp_size()) != 0) {
      return "NPU CP requires world_size divisible by dp_size * cp_size "
             "(orthogonal CP x TP layout)";
    }
    const int32_t attn_tp_size =
        global_world_size / (options.dp_size() * options.cp_size());
    if (attn_tp_size < 1) {
      return "NPU CP requires attn_tp_size >= 1";
    }
    const int32_t kv_split =
        ParallelConfig::get_instance().kv_split_size_effective();
    if (kv_split < 1 || options.cp_size() % kv_split != 0) {
      return "NPU CP requires kv_split_size effective value to be a positive "
             "divisor of cp_size";
    }
    return std::nullopt;
  }

  return "cp_size > 1 is only supported on platforms with model-side CP "
         "(MLU/NPU); disable CP (cp_size=1) or use MLU/NPU.";
}

std::optional<std::string> validate_qwen_dcp_topology(int32_t global_world_size,
                                                      int32_t dp_size,
                                                      int32_t kv_split_size) {
  if (kv_split_size < 1) {
    return "kv_split_size must resolve to a value greater than or equal to 1";
  }
  if (kv_split_size == 1) {
    return std::nullopt;
  }
  if (global_world_size < 1 || dp_size < 1) {
    return "Qwen DCP requires positive world_size and dp_size";
  }
  if (global_world_size % dp_size != 0) {
    return "Qwen DCP requires world_size divisible by dp_size";
  }
  const int32_t tp_size = global_world_size / dp_size;
  if (kv_split_size > tp_size || tp_size % kv_split_size != 0) {
    return "Qwen DCP kv_split_size must divide the TP size within each DP "
           "replica";
  }
  return std::nullopt;
}

namespace {

void print_startup_banner(const std::filesystem::path& model_path,
                          const std::string& backend,
                          int32_t node_rank) {
  if (node_rank != 0) {
    return;
  }

  constexpr std::string_view kAnsiRed = "\033[31m";
  constexpr std::string_view kAnsiReset = "\033[0m";
  const bool use_color = ::isatty(::fileno(stderr));

  std::array<std::string_view, 4> x_logo = {
      "      ", "▀█▄ ▀ ", "  █▶  ", "▄█▀ ▄ "};
  std::array<std::string_view, 4> llm_logo = {"█     █     █▄   ▄█",
                                              "█     █     █ ▀▄▀ █",
                                              "█     █     █     █",
                                              "█▄▄▄▄ █▄▄▄▄ █     █"};

  LOG(INFO) << "";
  LOG(INFO) << x_logo[0] << llm_logo[0];
  if (use_color) {
    LOG(INFO) << kAnsiRed << x_logo[1] << kAnsiReset << llm_logo[1]
              << "  version " << XLLM_BUILD_VERSION;
    LOG(INFO) << kAnsiRed << x_logo[2] << kAnsiReset << llm_logo[2]
              << "  model   " << model_path.string();
    LOG(INFO) << kAnsiRed << x_logo[3] << kAnsiReset << llm_logo[3]
              << "  backend " << backend;
  } else {
    LOG(INFO) << x_logo[1] << llm_logo[1] << "  version " << XLLM_BUILD_VERSION;
    LOG(INFO) << x_logo[2] << llm_logo[2] << "  model   "
              << model_path.string();
    LOG(INFO) << x_logo[3] << llm_logo[3] << "  backend " << backend;
  }
  LOG(INFO) << "";
}

}  // namespace

namespace {

#if defined(USE_NPU)
void validate_rank_tablefile_backend() {
  const EPLBConfig& eplb_config = EPLBConfig::get_instance();
  const ParallelConfig& parallel_config = ParallelConfig::get_instance();
  if (!eplb_config.rank_tablefile().empty() &&
      parallel_config.communication_backend() != "hccl") {
    LOG(FATAL) << "--rank_tablefile requires --communication_backend=hccl, "
               << "but got --communication_backend="
               << parallel_config.communication_backend();
  }
}

void resolve_npu_kernel_backend_for_options(Options* options) {
  CHECK(options != nullptr) << "options must not be null";
  if (options->backend() == "dit") {
    return;
  }

  // Python model executor builds the compute graph in Python (torch/torch_npu),
  // bypassing ATB C++ kernels entirely — force TORCH backend so that kernel
  // dispatch picks pure-torch implementations for reshape_and_cache etc.
  if (ModelConfig::is_python_model_impl(
          ModelConfig::get_instance().model_impl())) {
    options->npu_kernel_backend("TORCH");
    KernelConfig::get_instance().npu_kernel_backend("TORCH");
    LOG(INFO) << "Forced npu_kernel_backend=TORCH for python model_impl";
    return;
  }

  const std::string model_type =
      util::get_model_type(options->model_path(), options->backend());
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;
  if (!resolve_model_registration(model_type,
                                  options->npu_kernel_backend(),
                                  &effective_backend,
                                  &resolved_name,
                                  &error_message)) {
    LOG(FATAL) << error_message;
  }

  options->npu_kernel_backend(effective_backend);
  KernelConfig::get_instance().npu_kernel_backend(effective_backend);
  LOG(INFO) << "Resolved npu_kernel_backend=" << effective_backend
            << " for model_type=" << model_type;
}
#endif

}  // namespace

Master::Master(const Options& options, EngineType type)
    : options_(options),
      engine_type_(type),
      master_status_(options.master_status()) {
  const auto model_path =
      std::filesystem::path(options_.model_path()).lexically_normal();
  if (options_.host_blocks_factor() > 1.0) {
    const bool supports_host_offload =
        type == EngineType::LLM ||
        (type == EngineType::SSM && SpeculativeConfig::supports_host_kv_cache(
                                        options_.speculative_algorithm()));
    CHECK(supports_host_offload)
        << "Basic host KV cache offload supports the LLM engine and "
           "model-based speculative engines only.";
  }
  // Multi-process serving runs one worker per process. Select one runtime
  // logical device from the process-visible devices while keeping node_rank as
  // the global distributed identity.
  const int32_t visible_device_count = Platform::device_count();
  const int32_t device_idx = DeviceNameUtils::get_device_idx(
      options_.node_rank(), options_.nnodes(), visible_device_count);
  const auto visible_devices = DeviceNameUtils::parse_devices("auto");
  const std::vector<torch::Device> devices = {visible_devices[device_idx]};
  // World size is the node count (one worker per process).
  const int32_t global_world_size = options_.nnodes();
  const int32_t kv_split_size =
      ParallelConfig::get_instance().kv_split_size_effective();
  const bool npu_dcp_requested =
      Platform::is_npu() && options_.cp_size() == 1 && kv_split_size > 1;
  const bool native_qwen_dcp_requested =
      npu_dcp_requested && !ModelConfig::is_python_model_impl(
                               ModelConfig::get_instance().model_impl());
  if (npu_dcp_requested) {
    const std::optional<std::string> dcp_topology_error =
        validate_qwen_dcp_topology(
            global_world_size, options_.dp_size(), kv_split_size);
    CHECK(!dcp_topology_error.has_value()) << dcp_topology_error.value();
  }
  std::string model_type;
  if (options_.backend() == "llm" || native_qwen_dcp_requested ||
      (options_.cp_size() > 1 && Platform::uses_model_cp_sharding()) ||
      (ModelConfig::is_python_model_impl(
           ModelConfig::get_instance().model_impl()) &&
       options_.num_speculative_tokens() > 0)) {
    model_type = util::get_model_type(model_path, options_.backend());
  }
  // V4.1 does not support KV cache Store, speculative decoding, or
  // disaggregated PD: the grouped KV (C1/C2) is not handled by the restore or
  // draft-accounting paths. Reject the options before any engine/worker is
  // built. Prefix-cache support is a property of the cache groups and is
  // resolved in the block layer (leaf_participates_in_prefix_cache), not here.
  if (util::is_deepseek_v41_model_type(model_type)) {
    CHECK(!options_.enable_kvcache_store())
        << "DeepSeek-V4.1 does not support KV cache Store: restore for the "
           "grouped KV (C1/C2) is not implemented.";
    const bool v41_speculative_requested =
        options_.num_speculative_tokens() > 0 ||
        !options_.draft_model_path().value_or("").empty();
    CHECK(!v41_speculative_requested)
        << "DeepSeek-V4.1 does not support speculative decoding: grouped KV "
           "(C1/C2) is not implemented in the speculative path.";
    CHECK(!options_.enable_disagg_pd())
        << "DeepSeek-V4.1 does not support disaggregated PD: the grouped "
           "TOKEN KV (C1/C2) transfer path is not implemented.";
  }
  const std::optional<std::string> speculative_error =
      ModelConfig::validate_python_speculative_decode(
          ModelConfig::get_instance().model_impl(),
          model_type,
          options_.speculative_algorithm(),
          options_.num_speculative_tokens());
  CHECK(!speculative_error.has_value()) << speculative_error.value();
  const std::optional<std::string> cp_error =
      validate_model_cp(options_, type, model_type, global_world_size);
  CHECK(!cp_error.has_value()) << cp_error.value();
  options_.enable_mla(util::should_enable_mla(model_path, options_.backend()));
  print_startup_banner(model_path, options_.backend(), options_.node_rank());
  LOG(INFO) << "Master init options: " << options_.to_string();
  ParallelConfig::get_instance().cp_size(options_.cp_size());
  // cp_size <= 1 -> "disabled", otherwise "model" (model-side CP).
  const char* cp_sharding_stage =
      options_.cp_size() <= 1 ? "disabled" : "model";
  LOG(INFO) << "Resolved CP config: cp_size=" << options_.cp_size()
            << ", world_size=" << global_world_size
            << ", dp_size=" << options_.dp_size()
            << ", ep_size=" << options_.ep_size()
            << ", cp_sharding_stage=" << cp_sharding_stage
            << ", instance_role=" << options_.instance_role().to_string();

  // Allow brpc receive SIGTREM and SIGINT signal.
  brpc::FLAGS_graceful_quit_on_sigterm = true;
  brpc::FLAGS_graceful_quit_on_sighup = true;

#if defined(USE_NPU)
  EPLBConfig& eplb_config = EPLBConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  if (options.rank_tablefile().has_value()) {
    eplb_config.rank_tablefile(options.rank_tablefile().value());
  }
  if (options.communication_backend().has_value()) {
    parallel_config.communication_backend(
        options.communication_backend().value());
  }
  validate_rank_tablefile_backend();
  parallel_state::sync_torch_npu_rank_table_file_env(
      eplb_config.rank_tablefile());
  if (options.expert_parallel_degree().has_value()) {
    eplb_config.expert_parallel_degree(
        options.expert_parallel_degree().value());
  }
  if (options.enable_eplb().has_value()) {
    eplb_config.enable_eplb(options.enable_eplb().value());
  }
  if (options.redundant_experts_num().has_value()) {
    eplb_config.redundant_experts_num(options.redundant_experts_num().value());
  }
  if (options.eplb_update_interval().has_value()) {
    eplb_config.eplb_update_interval(options.eplb_update_interval().value());
  }
  if (options.eplb_min_peak_load_improvement().has_value()) {
    eplb_config.eplb_min_peak_load_improvement(
        options.eplb_min_peak_load_improvement().value());
  }
  CHECK(!eplb_config.enable_eplb() ||
        (type != EngineType::VLM && type != EngineType::VLMSSM))
      << "EPLB does not support VLM engines because VLMEngine has no EPLB "
         "control plane.";
  resolve_npu_kernel_backend_for_options(&options_);
#endif
  if (native_qwen_dcp_requested) {
    const bool supported_runtime =
        options_.npu_kernel_backend() == "TORCH" && type == EngineType::LLM &&
        options_.task_type() == "generate" && options_.ep_size() == 1 &&
        !options_.enable_eplb().value_or(false) &&
        options_.expert_parallel_degree().value_or(0) <= 1 &&
        options_.is_local() && options_.rank_tablefile().value_or("").empty() &&
        options_.host_blocks_factor() <= 1.0 &&
        !options_.enable_kvcache_store() &&
        !KVCacheConfig::get_instance().enable_xtensor() &&
        !options_.enable_disagg_pd() &&
        options_.instance_role() == InstanceRole::DEFAULT &&
        options_.draft_model_path().value_or("").empty() &&
        options_.num_speculative_tokens() == 0 &&
        ExecutionConfig::get_instance().enable_fia_decode() &&
        (model_type == "qwen3_5_text" || model_type == "qwen3_5_moe_text");
    CHECK(supported_runtime)
        << "Qwen3.5 DCP supports only replicated-GQA local NPU TORCH LLM "
           "generation without CP, EP/EPLB, host/KVStore/XTensor offload, "
           "P/D, or "
           "speculative decoding, and requires enable_fia_decode=true";
  }
  validate_layerwise_split_size_startup_config(
      options_, model_type, global_world_size);
  ParallelConfig::get_instance().enable_multi_stream_parallel(
      options.enable_multi_stream_parallel() && (options.nnodes() > 1));
  if (ParallelConfig::get_instance().enable_multi_stream_parallel()) {
    LOG(FATAL)
        << "Multi-stream parallel is refactoring now, will be supported later.";
  }
  // construct engine
  LOG(INFO) << "Creating engine with devices: "
            << DeviceNameUtils::to_string(devices);

  if (options_.enable_disagg_pd()) {
    // Enable service routing in disagg pd mode
    options_.enable_service_routing(true);
    if (options_.instance_role() == InstanceRole::PREFILL) {
      // Disable schedule overlap for prefill instance in disagg pd mode
      options_.enable_schedule_overlap(false);
      LOG(WARNING) << "Force to disable schedule overlap for prefill instance "
                      "in disagg pd mode.";
    }
  }

  if (options_.task_type() == "mm_embed") {
    options_.enable_chunked_prefill(false);
    options_.enable_prefix_cache(false);
    SchedulerConfig::get_instance().enable_chunked_prefill(false);
    KVCacheConfig::get_instance().enable_prefix_cache(false);
    LOG(WARNING) << "Disabling chunked prefill and prefix cache for "
                    "task=mm_embed to process all multimodal inputs.";
  }

  if (type == EngineType::VLM) {
    runtime::Options eng_options;
    eng_options.model_path(options_.model_path())
        .devices(devices)
        .backend(options.backend())
        .block_size(options.block_size())
        .max_cache_size(options.max_cache_size())
        .max_memory_utilization(options.max_memory_utilization())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .max_encoder_cache_size(options.max_encoder_cache_size())
        .max_processor_cache_items(options.max_processor_cache_items())
        .max_linear_state_cache_slots(options.max_linear_state_cache_slots())
        .task_type(options.task_type())
        .enable_mla(options_.enable_mla())
        .enable_flashcomm1(options_.enable_flashcomm1())
        .flashcomm1_min_prefill_tokens(options_.flashcomm1_min_prefill_tokens())
        .enable_mmrs_fusion(options_.enable_mmrs_fusion())
        .mmrs_comm_mode(options_.mmrs_comm_mode())
        .cp_size(options_.cp_size())
        .instance_role(options_.instance_role())
        .enable_disagg_pd(options_.enable_disagg_pd())
        .npu_kernel_backend(options_.npu_kernel_backend())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .enable_offline_inference(options_.enable_offline_inference())
        .disable_log_stats(options_.disable_log_stats())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .input_shm_size(options_.input_shm_size() * 1024 * 1024)
        .output_shm_size(options_.output_shm_size() * 1024 * 1024)
        .is_local(options_.is_local())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .master_node_addr(options.master_node_addr())
        .nnodes(options.nnodes())
        .node_rank(options.node_rank())
        .dp_size(options.dp_size())
        .ep_size(options.ep_size())
        .max_tokens_per_batch(options_.max_tokens_per_batch())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .enable_graph(options_.enable_graph())
        .enable_graph_mode_decode_no_padding(
            options_.enable_graph_mode_decode_no_padding())
        .enable_prefill_piecewise_graph(
            options_.enable_prefill_piecewise_graph())
        .max_tokens_for_graph_mode(options_.max_tokens_for_graph_mode())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill());

    auto engine = std::make_unique<VLMEngine>(eng_options);
    engine_ = std::move(engine);
  } else if (type == EngineType::SSM || type == EngineType::VLMSSM) {
    if (type == EngineType::VLMSSM) {
      CHECK(!options_.enable_disagg_pd())
          << "VLM speculative decoding does not support disaggregated PD";
      CHECK(!options_.enable_pd_ooc())
          << "VLM speculative decoding does not support PD OOC";
      CHECK(!options_.enable_service_routing())
          << "VLM speculative decoding does not support service routing";
      CHECK(!options_.enable_adaptive_speculative_decode())
          << "VLM speculative decoding does not support adaptive speculative "
             "decode";
    }
    // create a speculative engine if draft model path is provided
    const std::string draft_model_path =
        options_.draft_model_path().value_or("");
    const bool use_suffix_spec = options_.speculative_algorithm() == "Suffix";
    CHECK(use_suffix_spec || !draft_model_path.empty())
        << "draft model path is required unless --speculative_algorithm=Suffix";
    // Draft model shares the same devices as the target model.
    const auto& draft_devices = devices;
    LOG(INFO) << "Using draft devices: "
              << DeviceNameUtils::to_string(draft_devices);
    runtime::Options spec_options;
    spec_options.model_path(options_.model_path())
        .model_id(options_.model_id())
        .draft_model_path(draft_model_path)
        .devices(devices)
        .draft_devices(draft_devices)
        .backend(options_.backend())
        .block_size(options_.block_size())
        .max_cache_size(options_.max_cache_size())
        .max_memory_utilization(options_.max_memory_utilization())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .max_linear_state_cache_slots(options_.max_linear_state_cache_slots())
        .num_speculative_tokens(options_.num_speculative_tokens())
        .speculative_algorithm(options_.speculative_algorithm())
        .draft_sampling_mode(options_.draft_sampling_mode())
        .enable_mtp_draft_body_tp1(options_.enable_mtp_draft_body_tp1())
        .speculative_suffix_cache_max_depth(
            options_.speculative_suffix_cache_max_depth())
        .speculative_suffix_max_spec_factor(
            options_.speculative_suffix_max_spec_factor())
        .speculative_suffix_max_spec_offset(
            options_.speculative_suffix_max_spec_offset())
        .speculative_suffix_min_token_prob(
            options_.speculative_suffix_min_token_prob())
        .speculative_suffix_max_cached_requests(
            options_.speculative_suffix_max_cached_requests())
        .speculative_suffix_use_tree_spec(
            options_.speculative_suffix_use_tree_spec())
        .enable_adaptive_speculative_decode(
            options_.enable_adaptive_speculative_decode())
        .adaptive_speculative_min_gain(options_.adaptive_speculative_min_gain())
        .task_type(options_.task_type())
        .enable_mla(options_.enable_mla())
        .npu_kernel_backend(options_.npu_kernel_backend())
        .master_node_addr(options.master_node_addr())
        .nnodes(options.nnodes())
        .node_rank(options.node_rank())
        .dp_size(options.dp_size())
        .ep_size(options.ep_size())
        .enable_flashcomm1(options_.enable_flashcomm1())
        .flashcomm1_min_prefill_tokens(options_.flashcomm1_min_prefill_tokens())
        .enable_mmrs_fusion(options_.enable_mmrs_fusion())
        .mmrs_comm_mode(options_.mmrs_comm_mode())
        .cp_size(options_.cp_size())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .max_tokens_per_batch(options_.max_tokens_per_batch())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill())
        .instance_role(options_.instance_role())
        .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
        .transfer_listen_port(options_.transfer_listen_port())
        .enable_disagg_pd(options_.enable_disagg_pd())
        .enable_pd_ooc(options_.enable_pd_ooc())
        .enable_service_routing(options_.enable_service_routing())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .enable_offline_inference(options_.enable_offline_inference())
        .disable_log_stats(options_.disable_log_stats())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .input_shm_size(options_.input_shm_size() * 1024 * 1024)
        .output_shm_size(options_.output_shm_size() * 1024 * 1024)
        .is_local(options_.is_local())
        .enable_graph(options_.enable_graph())
        .enable_graph_mode_decode_no_padding(
            options_.enable_graph_mode_decode_no_padding())
        .enable_prefill_piecewise_graph(
            options_.enable_prefill_piecewise_graph())
        .max_tokens_for_graph_mode(options_.max_tokens_for_graph_mode());
    apply_runtime_kv_cache_options(options_, spec_options);

    if (use_suffix_spec) {
      if (type == EngineType::VLMSSM) {
        engine_ = std::make_unique<SuffixVLMSpeculativeEngine>(spec_options);
      } else {
        engine_ = std::make_unique<SuffixSpeculativeEngine>(spec_options);
      }
    } else {
      if (type == EngineType::VLMSSM) {
        engine_ =
            std::make_unique<SpeculativeEngineBase<VLMEngine>>(spec_options);
      } else {
        engine_ =
            std::make_unique<SpeculativeEngineBase<LLMEngine>>(spec_options);
      }
    }
  } else if (type == EngineType::LLM) {
    if (options_.task_type() == "embed" || options.task_type() == "mm_embed") {
      options_.enable_schedule_overlap(false);
      LOG(WARNING) << "Force to disable schedule overlap for embedding model, "
                      "avoiding performance degradation.";
    }
    runtime::Options eng_options;
    eng_options.model_path(options_.model_path())
        .devices(devices)
        .backend(options_.backend())
        .block_size(options_.block_size())
        .max_cache_size(options_.max_cache_size())
        .max_memory_utilization(options_.max_memory_utilization())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .max_linear_state_cache_slots(options_.max_linear_state_cache_slots())
        .task_type(options_.task_type())
        .enable_mla(options_.enable_mla())
        .npu_kernel_backend(options_.npu_kernel_backend())
        .master_node_addr(options_.master_node_addr())
        .nnodes(options_.nnodes())
        .node_rank(options_.node_rank())
        .dp_size(options_.dp_size())
        .ep_size(options_.ep_size())
        .enable_flashcomm1(options_.enable_flashcomm1())
        .flashcomm1_min_prefill_tokens(options_.flashcomm1_min_prefill_tokens())
        .enable_mmrs_fusion(options_.enable_mmrs_fusion())
        .mmrs_comm_mode(options_.mmrs_comm_mode())
        .cp_size(options_.cp_size())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .max_tokens_per_batch(options_.max_tokens_per_batch())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill())
        .instance_role(options_.instance_role())
        .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
        .transfer_listen_port(options_.transfer_listen_port())
        .enable_disagg_pd(options_.enable_disagg_pd())
        .enable_service_routing(options_.enable_service_routing())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .enable_offline_inference(options_.enable_offline_inference())
        .disable_log_stats(options_.disable_log_stats())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .input_shm_size(options_.input_shm_size() * 1024 * 1024)
        .output_shm_size(options_.output_shm_size() * 1024 * 1024)
        .is_local(options_.is_local())
        .server_idx(options_.server_idx())
        .enable_graph(options_.enable_graph())
        .enable_graph_mode_decode_no_padding(
            options_.enable_graph_mode_decode_no_padding())
        .enable_prefill_piecewise_graph(
            options_.enable_prefill_piecewise_graph())
        .max_tokens_for_graph_mode(options_.max_tokens_for_graph_mode())
        .enable_sleep_mode(options_.enable_sleep_mode())
        .model_id(options_.model_id());
    apply_runtime_kv_cache_options(options_, eng_options);

    engine_ = std::make_unique<LLMEngine>(eng_options);
  } else if (type == EngineType::REC) {
    options_.enable_schedule_overlap(false);
    LOG(WARNING) << "Force to disable schedule overlap for REC model, not "
                    "supported yet.";
    runtime::Options eng_options;
    eng_options.model_path(options_.model_path())
        .devices(devices)
        .backend(options_.backend())
        .block_size(options_.block_size())
        .max_cache_size(options_.max_cache_size())
        .max_memory_utilization(options_.max_memory_utilization())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .task_type(options_.task_type())
        .npu_kernel_backend(options_.npu_kernel_backend())
        .enable_mla(options_.enable_mla())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .enable_offline_inference(options_.enable_offline_inference())
        .disable_log_stats(options_.disable_log_stats())
        .spawn_worker_path(options_.spawn_worker_path())
        .enable_shm(options_.enable_shm())
        .is_local(options_.is_local())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .master_node_addr(options_.master_node_addr())
        .nnodes(options_.nnodes())
        .node_rank(options_.node_rank())
        .dp_size(options_.dp_size())
        .ep_size(options_.ep_size())
        .cp_size(options_.cp_size())
        .max_seqs_per_batch(options_.max_seqs_per_batch())
        .beam_width(options_.beam_width())
        .max_tokens_per_batch(options_.max_tokens_per_batch())
        .enable_graph(options_.enable_graph())
        .enable_graph_mode_decode_no_padding(
            options_.enable_graph_mode_decode_no_padding())
        .enable_prefill_piecewise_graph(
            options_.enable_prefill_piecewise_graph())
        .max_tokens_for_graph_mode(options_.max_tokens_for_graph_mode())
        .max_tokens_per_chunk_for_prefill(
            options_.max_tokens_per_chunk_for_prefill())
        .rec_worker_max_concurrency(options_.rec_worker_max_concurrency());

    engine_ = std::make_unique<RecEngine>(eng_options);
  } else if (type == EngineType::DIT) {
    // construct dit engine
    runtime::Options eng_options;
    eng_options.model_path(options.model_path())
        .model_id(options.model_id())
        .devices(devices)
        .backend(options.backend())
        .npu_kernel_backend(options_.npu_kernel_backend())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .enable_chunked_prefill(options_.enable_chunked_prefill())
        .enable_offline_inference(options_.enable_offline_inference())
        .disable_log_stats(options_.disable_log_stats())
        .max_memory_utilization(options_.max_memory_utilization())
        .master_node_addr(options.master_node_addr())
        .nnodes(options.nnodes())
        .task_type(options_.task_type())
        .enable_shm(options_.enable_shm())
        .input_shm_size(options_.input_shm_size() * 1024 * 1024)
        .output_shm_size(options_.output_shm_size() * 1024 * 1024)
        .is_local(options_.is_local())
        .node_rank(options_.node_rank())
        .enable_schedule_overlap(options_.enable_schedule_overlap())
        .dp_size(options_.dp_size())
        .ep_size(options_.ep_size())
        .tp_size(options_.tp_size())
        .sp_size(options_.sp_size())
        .cfg_size(options_.cfg_size())
        .vae_size(options_.vae_size())
        .text_encoder_tp_size(options_.text_encoder_tp_size());

    auto dit_engine = std::make_unique<DiTEngine>(eng_options);
    engine_ = std::move(dit_engine);
  } else {
    LOG(WARNING) << "Not supported llm engine type: "
                 << static_cast<size_t>(type);
  }

  if (!is_leader()) {
    const std::string master_node_addr =
        options_.master_node_addr().value_or("");
    if (master_node_addr.empty()) {
      LOG(FATAL) << "Multi-node serving requires --master_node_addr, "
                    "current value is empty.";
    }
  }
}

std::atomic<bool> Master::idle_running_{false};

void Master::handle_shutdown_signal(int /*signum*/) {
  idle_running_.store(false, std::memory_order_relaxed);
}

Master::~Master() {
  idle_running_.store(false, std::memory_order_relaxed);
  if (idle_thread_.joinable()) {
    idle_thread_.join();
  }
}

void Master::wait() {
  if (idle_thread_.joinable()) {
    idle_thread_.join();
  }
}

void Master::run() {
  idle_running_.store(true, std::memory_order_relaxed);
  signal(SIGINT, Master::handle_shutdown_signal);
  signal(SIGTERM, Master::handle_shutdown_signal);

  idle_thread_ = std::thread([]() {
    while (idle_running_.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  });
}

std::unique_ptr<Master> create_master(const std::string& backend,
                                      const Options& options) {
  if (backend == "llm") {
    return std::make_unique<LLMMaster>(options);
  } else if (backend == "vlm") {
    return std::make_unique<VLMMaster>(options);
  } else if (backend == "dit") {
    LOG(INFO) << "creating dit master";
    return std::make_unique<DiTMaster>(options);
  } else if (backend == "rec") {
    LOG(INFO) << "creating rec master";
    return std::make_unique<RecMaster>(options);
  } else {
    LOG(FATAL) << "Failed to create master, backend is" << backend;
    return nullptr;
  }
}

std::unique_ptr<Master> fork_master(Master* master, const Options& options) {
  // sleep/wakeup/fork_master requires --enable_xtensor=true
  if (!::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
    LOG(WARNING) << "fork_master requires xtensor to be enabled";
    return nullptr;
  }

  static uint64_t server_idx = 1;
  CHECK(master != nullptr);

  Options new_options = master->options();

  if (!options.model_id().empty()) {
    new_options.model_id() = options.model_id();
  }
  if (!options.model_path().empty()) {
    new_options.model_path() = options.model_path();
  }
  new_options.master_node_addr() = options.master_node_addr();
  new_options.server_idx() = server_idx++;
  new_options.master_status() = options.master_status();
  // Set nnodes and dp_size from fork request (tp_size * dp_size = nnodes)
  if (options.nnodes() > 0 && new_options.nnodes() >= options.nnodes()) {
    new_options.nnodes() = options.nnodes();
  }
  if (options.dp_size() > 0 && new_options.dp_size() >= options.nnodes()) {
    new_options.dp_size() = options.dp_size();
  }
  std::unique_ptr<Master> new_master =
      create_master(new_options.backend(), new_options);
  new_master->run();

  return new_master;
}
}  // namespace xllm
