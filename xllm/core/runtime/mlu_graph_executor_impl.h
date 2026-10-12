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

#pragma once

#include <framework/graphs/MLUGraph.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/runtime/executor_impl.h"
#include "core/runtime/executor_impl_factory.h"
#include "core/runtime/mlu_graph_planner.h"
#include "core/runtime/options.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "models/model_registry.h"

namespace xllm::mlu {
// Every graph owns its input addresses and metadata for its entire lifetime.
class GraphPersistentParam final {
 public:
  GraphPersistentParam(const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       ModelInputParams params,
                       const GraphLayout& layout,
                       int64_t graph_max_kv_seq_len,
                       int64_t main_block_table_columns);

  void update_input_buffer(const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           const ModelInputParams& params,
                           const GraphLayout& layout);
  std::size_t get_persistent_tensor_bytes() const;

  torch::Tensor tokens_;
  torch::Tensor positions_;
  ModelInputParams params_;
  bool use_mrope_ = false;
  torch::Tensor output_;
  torch::Tensor aux_hidden_states_;
};

// graph executor using libtorch MLUGraph for memory management
// MLUGraph provides mempool to manage temporary tensors during forward pass
class MluGraph final {
 public:
  explicit MluGraph(std::unique_ptr<GraphPersistentParam> persistent_param);

  // Capture computation graph for given bucket num_tokens.
  // All buckets must capture on the same MLU stream so the caching allocator
  // can reuse scratch freed by earlier captures within the shared mempool;
  // capturing on different streams defeats its per-stream block reuse.
  ModelOutput capture(CausalLM* model,
                      std::vector<KVCache>& kv_cache,
                      const torch_mlu::MempoolId_t& pool,
                      const torch_mlu::MLUStream& capture_stream,
                      const runtime::Options& options);

  // Replay captured graph with new input data
  ModelOutput replay();
  void update_input_buffer(CausalLM* model,
                           const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           const ModelInputParams& params,
                           const GraphLayout& layout);

  std::size_t owned_persistent_tensor_bytes() const;

 private:
  void prepare_model_graph_metadata(CausalLM* model);

  ModelOutput output() const;
  void store_output(const ModelOutput& result);
  bool enable_aux_hidden_states_ = false;

  // Stable storage owned by this graph.
  std::unique_ptr<GraphPersistentParam> persistent_param_;

  // Per-graph metadata state for models that require graph-forward
  // metadata preparation (e.g., DeepSeek V4 DSA metadata).
  std::unique_ptr<ModelGraphMetadataState> model_graph_metadata_state_;

  // Destroy the graph before releasing tensors referenced by captured kernels.
  torch_mlu::MLUGraph graph_;
};

// Executor implementation using MLU graph optimization
// Uses MLUGraph mempool to reduce memory allocation overhead during inference
class MluGraphExecutorImpl final : public ExecutorImpl {
 public:
  MluGraphExecutorImpl(CausalLM* model,
                       const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);

  ~MluGraphExecutorImpl() override;

  ForwardInput prepare_inputs(Batch& batch) override;

  // Execute model with graph optimization for decode phase
  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) override;

  void set_dp_process_group(ProcessGroup* group) override;

 private:
  ModelOutput run_eager(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& params);
  void log_memory_after_capture();

  CausalLM* model_;  // not owned
  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;
  torch_mlu::MempoolId_t graph_pool_;
  // Fixed capture stream shared by every bucket capture. Lazily initialized on
  // the first capture so the allocator can reuse pool scratch across buckets.
  std::optional<torch_mlu::MLUStream> graph_capture_stream_;
  std::unique_ptr<MluGraphPlanner> planner_;
  ProcessGroup* dp_group_ = nullptr;  // owned by the worker context
  MtpModelCapabilities mtp_capabilities_;
  std::size_t last_pool_reserved_bytes_ = 0;
  std::size_t peak_pool_reserved_bytes_ = 0;

  std::unordered_map<GraphKey, std::unique_ptr<MluGraph>, GraphKeyHash> graphs_;
};
REGISTER_EXECUTOR("mlu", MluGraphExecutorImpl);
}  // namespace xllm::mlu
