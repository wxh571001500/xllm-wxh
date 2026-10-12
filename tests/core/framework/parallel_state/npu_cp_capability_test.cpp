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

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>

#include "core/distributed_runtime/master.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/util/scope_guard.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

TEST(NpuDcpTopologyTest, KvOwnerIsLocalToDpCohort) {
  for (int32_t rank = 0; rank < 8; ++rank) {
    ParallelArgs args(rank,
                      /*world_size=*/8,
                      /*dp_size=*/2,
                      /*cp_size=*/1,
                      /*process_group=*/nullptr,
                      /*ep_size=*/8);
    args.kv_split_size(2);
    EXPECT_EQ(args.kv_split_rank(), (rank % 4) / 2);
    args.cp_size(2);
    EXPECT_EQ(args.kv_split_rank(), (rank % 4) / 2);
  }
}

TEST(NpuCpCapabilityTest, RegisteredCpCapableModels) {
  // The models that opt into NPU model-side CP. deepseek_v32 / glm_moe_dsa
  // drive it through the ATB NpuCpPlan pipeline; deepseek_v4 owns its split
  // inside the model on the TORCH backend. Both are advertised here because
  // this is the master-side startup gate, not the worker-side sharding switch.
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v32"));
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v32_mtp"));
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v4"));
  EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v4_mtp"));
  EXPECT_TRUE(is_npu_model_cp_capable("glm_moe_dsa"));
  EXPECT_TRUE(is_npu_model_cp_capable("glm_moe_dsa_mtp"));
  // The registry must advertise MODEL for these and NONE for the rest.
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("deepseek_v32"),
            CpShardingMode::MODEL);
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("glm_moe_dsa_mtp"),
            CpShardingMode::MODEL);
}

TEST(NpuCpCapabilityTest, UnregisteredModelsAreNotCapable) {
  // deepseek_v3_mtp uses the DeepSeekV2 decoder without the V3.2 ATB CP
  // metadata/TP contract; it must NOT be advertised as CP-capable so that
  // validate_model_cp rejects deepseek_v3_mtp + cp_size>1 at startup.
  EXPECT_FALSE(is_npu_model_cp_capable("deepseek_v3_mtp"));
  EXPECT_FALSE(is_npu_model_cp_capable("deepseek_v3"));
  // Unrelated NPU models are not CP-capable.
  EXPECT_FALSE(is_npu_model_cp_capable("qwen3"));
  EXPECT_FALSE(is_npu_model_cp_capable("qwen3_atb"));
  // Hybrid linear attention models are the only ones the graph executor takes
  // through spec-verify chunked prefill; none of them is CP-capable, which is
  // what keeps that capture path CP-free.
  EXPECT_FALSE(is_npu_model_cp_capable("qwen3_next"));
  // Unknown model names default to NONE.
  EXPECT_FALSE(is_npu_model_cp_capable("definitely_not_a_model"));
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("deepseek_v3_mtp"),
            CpShardingMode::NONE);
  EXPECT_EQ(ModelRegistry::get_cp_sharding_mode("definitely_not_a_model"),
            CpShardingMode::NONE);
}

TEST(NpuCpCapabilityTest, RegistrationIsIdempotent) {
  // Repeated calls must not flip the capability and must keep returning the
  // same result (std::call_once guards the one-shot registration).
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(is_npu_model_cp_capable("deepseek_v32"));
    EXPECT_FALSE(is_npu_model_cp_capable("deepseek_v3_mtp"));
  }
}

TEST(NpuCpCapabilityTest, PythonCpAllowsAclGraphAndRejectsCompileBackends) {
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_python_graph_backend =
      execution_config.python_graph_backend();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
    execution_config.python_graph_backend(original_python_graph_backend);
  });
  execution_config.python_graph_backend("off");
  model_config.model_impl("python");
  parallel_config.kv_split_size(1);

  Options options;
  options.task_type("generate")
      .cp_size(4)
      .dp_size(1)
      .ep_size(16)
      .instance_role(InstanceRole::PREFILL)
      .enable_graph(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());

  options.enable_graph(false);
  execution_config.python_graph_backend("aclgraph");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());

  execution_config.python_graph_backend("inductor");
  const std::optional<std::string> graph_error = std::optional<std::string>(
      "Python model-side CP requires Prefill to use EagerRunner; use "
      "--python_graph_backend=off or decode-only aclgraph");
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            graph_error);

  execution_config.python_graph_backend("off");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());
}

TEST(NpuCpCapabilityTest, PythonCpPreservesQwenAndRestrictsGlm) {
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_python_graph_backend =
      execution_config.python_graph_backend();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
    execution_config.python_graph_backend(original_python_graph_backend);
  });
  execution_config.python_graph_backend("off");
  model_config.model_impl("python");
  parallel_config.kv_split_size(1);

  Options options;
  options.task_type("generate")
      .cp_size(4)
      .dp_size(1)
      .ep_size(16)
      .instance_role(InstanceRole::PREFILL)
      .enable_graph(false)
      .speculative_algorithm("MTP");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "qwen3",
                                 /*global_world_size=*/16)
                   .has_value());
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::SSM,
                                 "qwen3",
                                 /*global_world_size=*/16)
                   .has_value());
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::SSM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python model-side CP does not support MTP speculative "
                "verification; run MTP on a cp_size=1 Decode instance"));

  options.speculative_algorithm("DSpark");
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::SSM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Current model-side CP does not support aux-hidden-capture "
                "speculative algorithms (Eagle3/DFlash/DSpark); run "
                "speculative decoding on a cp_size=1 Decode instance."));

  options.cp_size(1).instance_role(InstanceRole::DECODE).enable_disagg_pd(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());
  options.cp_size(4)
      .instance_role(InstanceRole::PREFILL)
      .enable_disagg_pd(false)
      .speculative_algorithm("MTP");

  // EP topology validation belongs to model construction, not this Python CP
  // capability gate.
  options.ep_size(2);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());

  parallel_config.kv_split_size(2);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "qwen3",
                                 /*global_world_size=*/16)
                   .has_value());
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python GLM CP with kv_split_size > 1 requires "
                "disaggregated PD with the PREFILL role; set "
                "enable_disagg_pd=true and instance_role=PREFILL"));

  options.enable_disagg_pd(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm_moe_dsa",
                                 /*global_world_size=*/16)
                   .has_value());

  options.instance_role(InstanceRole::DEFAULT);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python GLM CP with kv_split_size > 1 requires "
                "disaggregated PD with the PREFILL role; set "
                "enable_disagg_pd=true and instance_role=PREFILL"));
  options.instance_role(InstanceRole::PREFILL);

  parallel_config.kv_split_size(1);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa_mtp",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python model-side CP does not support "
                "model_type=glm_moe_dsa_mtp; supported models are qwen3, "
                "glm_moe_dsa, and glm5_next."));

  parallel_config.kv_split_size(3);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm_moe_dsa",
                              /*global_world_size=*/16),
            std::optional<std::string>(
                "Python CP requires kv_split_size effective value to be a "
                "positive divisor of cp_size"));
}

TEST(NpuCpCapabilityTest, PythonGlm5NextCapabilityGate) {
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  const std::string original_python_graph_backend =
      execution_config.python_graph_backend();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  const bool original_chunked_prefill =
      scheduler_config.enable_chunked_prefill();
  const bool original_mix_batch = scheduler_config.enable_mix_batch();
  const bool original_prefix_cache = kv_cache_config.enable_prefix_cache();
  const bool original_schedule_overlap =
      scheduler_config.enable_schedule_overlap();
  ScopeGuard config_guard([&] {
    scheduler_config.enable_schedule_overlap(original_schedule_overlap);
    kv_cache_config.enable_prefix_cache(original_prefix_cache);
    scheduler_config.enable_mix_batch(original_mix_batch);
    scheduler_config.enable_chunked_prefill(original_chunked_prefill);
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
    execution_config.python_graph_backend(original_python_graph_backend);
  });
  model_config.model_impl("python");
  execution_config.python_graph_backend("off");
  parallel_config.kv_split_size(1);
  scheduler_config.enable_chunked_prefill(false);
  scheduler_config.enable_mix_batch(false);
  kv_cache_config.enable_prefix_cache(false);
  scheduler_config.enable_schedule_overlap(false);

  Options options;
  options.task_type("generate")
      .cp_size(2)
      .dp_size(1)
      .ep_size(1)
      .instance_role(InstanceRole::PREFILL)
      .enable_graph(false);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());

  options.instance_role(InstanceRole::DEFAULT);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  options.instance_role(InstanceRole::PREFILL);

  execution_config.python_graph_backend("aclgraph");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());

  // B4 (implement.md, design.md §3.1 Gates B1/B2): the blanket
  // EngineType::SSM rejection for glm5_next+CP is narrowed to exclude only
  // DFlash2 — MTP/Suffix (and any other non-DFlash2 algorithm) stay
  // rejected with the new, narrower message text.
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::SSM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python GLM-5 Next CP does not support this target-side "
                "speculative algorithm; run speculation on a cp_size=1 "
                "Decode instance"));

  // DFlash2 is exempted from both the general aux-hidden-capture gate
  // (Gate B1, alongside deepseek_v4) and the glm5_next-specific blanket SSM
  // rejection (Gate B2): the combination must now be admitted.
  options.speculative_algorithm("DFlash2");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::SSM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  // Case-insensitive, mirroring SpeculativeConfig::is_dflash2_algorithm.
  options.speculative_algorithm("dflash2");
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::SSM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());

  // Eagle3/DFlash(1)/DSpark are aux-hidden-capture algorithms too, but were
  // not individually verified for glm5_next (design.md §3.1 Gate B1) — they
  // must still hit the general aux-hidden-capture gate, not the narrower
  // glm5_next-specific one (is_dsv4_model is false and is_glm5_next_dflash2
  // is false for these, so the earlier, general gate fires first).
  options.speculative_algorithm("Eagle3");
  EXPECT_EQ(
      validate_model_cp(options,
                        EngineType::SSM,
                        "glm5_next",
                        /*global_world_size=*/8),
      std::optional<std::string>(
          "Current model-side CP does not support aux-hidden-capture "
          "speculative algorithms (Eagle3/DFlash/DSpark); run speculative "
          "decoding on a cp_size=1 Decode instance."));

  options.speculative_algorithm("MTP");

  execution_config.python_graph_backend("inductor");
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python model-side CP requires Prefill to use EagerRunner; use "
                "--python_graph_backend=off or decode-only aclgraph"));

  execution_config.python_graph_backend("off");
  options.ep_size(2);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  options.dp_size(2).ep_size(8);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  options.dp_size(0);
  EXPECT_TRUE(validate_model_cp(options,
                                EngineType::LLM,
                                "glm5_next",
                                /*global_world_size=*/8)
                  .has_value());
  options.dp_size(uint32_t{1} << 31);
  EXPECT_TRUE(validate_model_cp(options,
                                EngineType::LLM,
                                "glm5_next",
                                /*global_world_size=*/8)
                  .has_value());
  options.dp_size(2);
  for (uint32_t ep_size : {0u, 3u}) {
    options.ep_size(ep_size);
    EXPECT_TRUE(validate_model_cp(options,
                                  EngineType::LLM,
                                  "glm5_next",
                                  /*global_world_size=*/8)
                    .has_value());
  }
  options.dp_size(1);
  options.ep_size(1);

  options.expert_parallel_degree(2).ep_size(8);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  options.ep_size(4);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python GLM-5 Next EPLv2 with PCP requires ep_size equal to "
                "world_size"));
  options.expert_parallel_degree(1).ep_size(1);

  // Owner-sharded KV is admitted under the same constraint glm_moe_dsa uses:
  // the KVShardBatchMetadata that the owner-local write and the CP gather
  // consume is built for prefill/chunked-prefill batches only, so the sharded
  // shape needs a prefill-only instance -- which in turn only exists under
  // disaggregated PD. A standalone instance runs prefill and decode on the
  // same ranks whatever the role says, so the role alone is not enough.
  options.cp_size(4);
  parallel_config.kv_split_size(2);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python GLM-5 Next CP with kv_split_size > 1 requires "
                "disaggregated PD with the PREFILL role; set "
                "enable_disagg_pd=true and instance_role=PREFILL"));
  // The DEFAULT role is refused by the same gate: under disaggregated PD it is
  // handed decode batches, whose shards no backend combines.
  options.enable_disagg_pd(true);
  options.instance_role(InstanceRole::DEFAULT);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python GLM-5 Next CP with kv_split_size > 1 requires "
                "disaggregated PD with the PREFILL role; set "
                "enable_disagg_pd=true and instance_role=PREFILL"));
  // The admitted shape: world=8, dp=1, tp=2, cp=4, kv_split == cp, on a
  // prefill-only instance under disaggregated PD.
  options.instance_role(InstanceRole::PREFILL);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  parallel_config.kv_split_size(4);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  parallel_config.kv_split_size(1);
  options.enable_disagg_pd(false);
  options.cp_size(2);

  // B3 (implement.md): enable_chunked_prefill is no longer rejected for
  // glm5_next+CP — the underlying is_mla-and-chunked-prefill cp_context
  // mechanism is generic and pre-dates this gate (R2-followup verdict B);
  // see tests/python/test_glm5_next_cp.py's
  // test_kda_cp_chunked_prefill_matches_noncp_nonchunked_baseline for the
  // model-level correctness proof this gate removal depends on.
  scheduler_config.enable_chunked_prefill(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  scheduler_config.enable_chunked_prefill(false);

  // A default enable_mix_batch=true no longer trips the gate: the flag is
  // inert under CP because resolve_batch_mode() force-disables mixed batches
  // whenever cp_size > 1, so the raw value is admitted untouched (the
  // scheduler, not the validator, owns the effective batch mode).
  scheduler_config.enable_mix_batch(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  EXPECT_TRUE(scheduler_config.enable_mix_batch());
  scheduler_config.enable_mix_batch(false);

  // The prefix-cache ban is narrowed to the standalone shapes: a non-PD
  // instance (whatever its role label) keeps the refusal with the scoped
  // message, while the PD PREFILL role is admitted below.
  kv_cache_config.enable_prefix_cache(true);
  EXPECT_EQ(
      validate_model_cp(options,
                        EngineType::LLM,
                        "glm5_next",
                        /*global_world_size=*/8),
      std::optional<std::string>(
          "Python GLM-5 Next CP with enable_prefix_cache=true requires "
          "disaggregated PD with the PREFILL role; a standalone CP instance "
          "must set enable_prefix_cache=false"));
  kv_cache_config.enable_prefix_cache(false);

  // M11.6: the schedule-overlap ban is lifted for glm5_next CP. The M11.5
  // pins make overlap safe: the overlapped decode token replacement is
  // byte-identical across CP ranks (greedy argmax over CP-replicated logits,
  // and the CP sample-token broadcast for stochastic batches in
  // LLMWorkerImpl::step_internal), the deferred KDA linear-state restore
  // stays on the compute stream, and the single-threaded worker task pool
  // keeps the block lifecycle FIFO.
  scheduler_config.enable_schedule_overlap(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());

  // The lift is scoped to overlap only: with overlap still enabled, the
  // other initially-required flags keep refusing (mix batch is the
  // exception -- the flag is inert under CP, admitted above).
  scheduler_config.enable_mix_batch(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  scheduler_config.enable_mix_batch(false);

  kv_cache_config.enable_prefix_cache(true);
  EXPECT_EQ(
      validate_model_cp(options,
                        EngineType::LLM,
                        "glm5_next",
                        /*global_world_size=*/8),
      std::optional<std::string>(
          "Python GLM-5 Next CP with enable_prefix_cache=true requires "
          "disaggregated PD with the PREFILL role; a standalone CP instance "
          "must set enable_prefix_cache=false"));
  kv_cache_config.enable_prefix_cache(false);
  // Overlap stays enabled through the checks below so the PD PREFILL-role
  // admission and the pd_ooc refusal are also pinned for the overlapped
  // combination (the L3 smoke topology); the config guard restores it.

  options.enable_disagg_pd(true);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  // The admitted prefix-cache shape: the PD PREFILL role carries prefix
  // caching (the prefill-only PD instance the kv_split admission already
  // scopes to), including the full serving topology -- CP4 x TP2,
  // kv_split_size=2, chunked prefill, and schedule overlap together.
  kv_cache_config.enable_prefix_cache(true);
  scheduler_config.enable_chunked_prefill(true);
  options.cp_size(4);
  parallel_config.kv_split_size(2);
  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/8)
                   .has_value());
  options.cp_size(2);
  parallel_config.kv_split_size(1);
  scheduler_config.enable_chunked_prefill(false);
  // Under PD, a non-PREFILL role with prefix caching is refused by the
  // prefix gate before the PD role gate fires (master.cpp checks the
  // prefix scope first).
  options.instance_role(InstanceRole::DEFAULT);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python GLM-5 Next CP with enable_prefix_cache=true requires "
                "disaggregated PD with the PREFILL role; a standalone CP "
                "instance must set enable_prefix_cache=false"));
  kv_cache_config.enable_prefix_cache(false);
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python GLM-5 Next CP with disaggregated PD requires the "
                "PREFILL role"));
  options.instance_role(InstanceRole::PREFILL)
      .enable_disagg_pd(false)
      .enable_pd_ooc(true);
  // pd_ooc keeps refusing with schedule overlap enabled, proving the lift
  // did not weaken the other initially-required bans.
  EXPECT_EQ(validate_model_cp(options,
                              EngineType::LLM,
                              "glm5_next",
                              /*global_world_size=*/8),
            std::optional<std::string>(
                "Python GLM-5 Next CP initially requires enable_pd_ooc=false"));
}

TEST(NpuCpCapabilityTest, PythonGlm5NextShardedKvIsAdmittedAtCpSizeOne) {
  // The cp_size == 1 Python DCP decode shape is admitted, and that admission
  // is the design: validate_model_cp returns nullopt for every cp_size == 1
  // instance before any kv_split logic runs (master.cpp's cp1 early return),
  // so a glm5_next decode instance with kv_split_size > 1 never sees the
  // "requires disaggregated PD with the PREFILL role" gate that governs the
  // cp > 1 shape. Safety comes from backend selection instead: an MLA model
  // at cp_size == 1 with a live DCP group selects SfaDcpAttentionBackend
  // (executor.py), which localizes its own slots in prepare() and never
  // writes through the global slot mapping -- pinning the selection is
  // test_model_executor.py's
  // test_glm_next_decode_cp1_selects_sfa_dcp_backend_from_dsa_layer.
  //
  // This is the inverted form of the pr-line's
  // PythonGlm5NextShardedKvRequiresCp (6d87b4f16), which asserted a staging
  // guard fired for this shape. m11 never carried that guard: the shape has
  // been admissible here since the cp1 early return landed, and nothing pinned
  // it until now.
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_python_graph_backend =
      execution_config.python_graph_backend();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
    execution_config.python_graph_backend(original_python_graph_backend);
  });
  execution_config.python_graph_backend("off");
  model_config.model_impl("python");
  parallel_config.kv_split_size(2);

  Options options;
  options.task_type("generate");
  options.cp_size(1);
  options.dp_size(1);
  options.ep_size(16);
  options.enable_disagg_pd(true);
  options.instance_role(InstanceRole::DECODE);

  // Locate the failure before asserting on the shape: if either of these
  // trips, the problem is the fixture, not the admission.
  ASSERT_EQ(options.cp_size(), 1) << "options.cp_size was not set";
  ASSERT_EQ(parallel_config.kv_split_size_effective(), 2)
      << "kv_split_size_effective was not 2";
  ASSERT_TRUE(ModelConfig::is_python_model_impl(
      ModelConfig::get_instance().model_impl()))
      << "model_impl is not the python path";

  EXPECT_FALSE(validate_model_cp(options,
                                 EngineType::LLM,
                                 "glm5_next",
                                 /*global_world_size=*/16)
                   .has_value())
      << "cp_size == 1 with kv_split_size > 1 on a DECODE-role python "
         "glm5_next instance must be admissible, got: "
      << validate_model_cp(options,
                           EngineType::LLM,
                           "glm5_next",
                           /*global_world_size=*/16)
             .value_or("");

  // The same shape must also clear the DCP topology gate the Master
  // constructor runs for NPU cp1 instances with kv_split_size > 1
  // (validate_qwen_dcp_topology): world=16, dp=1 gives tp=16, and 2 divides
  // 16, so the shard has a TP-local factor to live in.
  EXPECT_FALSE(validate_qwen_dcp_topology(
                   /*global_world_size=*/16,
                   /*dp_size=*/1,
                   /*kv_split_size=*/2)
                   .has_value())
      << "kv_split_size=2 must divide the TP size (16) within the DP replica";
}

TEST(NpuCpCapabilityTest, PythonDcpIsAdmittedAtDpGreaterThanOneAndCpSizeOne) {
  // The other half of the same admission: cp_size == 1 with BOTH dp_size > 1
  // and kv_split_size > 1. The pr-line staged this shape behind a
  // "dp_size == 1 until DP-local KV groups are implemented" guard
  // (6d87b4f16 removed it on hardware evidence); m11 never carried the guard
  // because the DCP group is DP-replica-local by construction here
  // (py_causal_lm.cpp derives dcp_group_index from dp_rank and tp_rank), so
  // the shape only has to clear validate_qwen_dcp_topology's TP-divisibility
  // rule. Pin that it does -- and that a width the TP size cannot host is
  // still refused, so the admission is a real decision, not a vacuous pass.
  ModelConfig& model_config = ModelConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const std::string original_model_impl = model_config.model_impl();
  const int32_t original_kv_split_size = parallel_config.kv_split_size();
  ScopeGuard config_guard([&] {
    parallel_config.kv_split_size(original_kv_split_size);
    model_config.model_impl(original_model_impl);
  });
  model_config.model_impl("python");
  parallel_config.kv_split_size(2);

  Options options;
  options.task_type("generate");
  options.cp_size(1);
  options.dp_size(2);
  options.ep_size(2);
  options.enable_disagg_pd(true);
  options.instance_role(InstanceRole::DECODE);

  ASSERT_EQ(options.dp_size(), 2) << "options.dp_size was not set";
  ASSERT_EQ(options.cp_size(), 1) << "options.cp_size was not set";
  ASSERT_EQ(parallel_config.kv_split_size_effective(), 2)
      << "kv_split_size_effective was not 2";

  const std::optional<std::string> dp_error =
      validate_model_cp(options,
                        EngineType::LLM,
                        "glm5_next",
                        /*global_world_size=*/4);
  EXPECT_FALSE(dp_error.has_value())
      << "dp_size > 1 with kv_split_size > 1 at cp_size == 1 must be "
         "admissible, got: "
      << dp_error.value_or("");

  // world=4, dp=2 gives tp=2: a kv_split of 2 fits, a kv_split of 4 does not.
  EXPECT_FALSE(validate_qwen_dcp_topology(
                   /*global_world_size=*/4,
                   /*dp_size=*/2,
                   /*kv_split_size=*/2)
                   .has_value());
  const std::optional<std::string> topology_error = validate_qwen_dcp_topology(
      /*global_world_size=*/4,
      /*dp_size=*/2,
      /*kv_split_size=*/4);
  ASSERT_TRUE(topology_error.has_value())
      << "kv_split_size=4 exceeds the TP size (2) and must be refused";
  EXPECT_EQ(topology_error.value(),
            "Qwen DCP kv_split_size must divide the TP size within each DP "
            "replica");
}

TEST(NpuDcpTopologyTest, AcceptsOnlyTpLocalFactors) {
  const auto is_valid = [](int32_t kv_split_size) {
    return !validate_qwen_dcp_topology(/*global_world_size=*/8,
                                       /*dp_size=*/2,
                                       kv_split_size)
                .has_value();
  };

  EXPECT_TRUE(is_valid(2));
  EXPECT_TRUE(is_valid(4));
  EXPECT_FALSE(is_valid(3));
  EXPECT_FALSE(is_valid(8));
}

}  // namespace
}  // namespace xllm
