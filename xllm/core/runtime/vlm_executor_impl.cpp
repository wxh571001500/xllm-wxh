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

#include "vlm_executor_impl.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

VlmExecutorImpl::VlmExecutorImpl(CausalLM* model,
                                 const ModelArgs& args,
                                 const torch::Device& device,
                                 const runtime::Options& options)
    : model_(dynamic_cast<CausalVLM*>(model)),
      args_(args),
      device_(device),
      options_(options) {
  if (options_.max_encoder_cache_size() > 0) {
    encoder_cache_ = std::make_unique<EncoderCache>(
        options_.max_encoder_cache_size() * 1024 * 1024);
  }

  if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
    llm_executor_ = ExecutorImplFactory::get_instance().create_executor_impl(
        model, args, device, options, Platform::type_str());
  }
}

void VlmExecutorImpl::set_dp_process_group(ProcessGroup* group) {
  if (llm_executor_) {
    llm_executor_->set_dp_process_group(group);
  }
}

ForwardInput VlmExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

MMDict VlmExecutorImpl::encode(const ModelInputParams& params) {
  return model_->encode(params);
}

ModelOutput VlmExecutorImpl::run(const torch::Tensor& tokens,
                                 const torch::Tensor& positions,
                                 std::vector<KVCache>& kv_caches,
                                 const ModelInputParams& params) {
  torch::NoGradGuard no_grad;
  auto& mm_data = params.multimodal.mm_data;

  // Pure decode carries no multimodal data. Its mRoPE positions degenerate to
  // identical rows, so they collapse to the 1-D vector ordinary rope and the
  // graph's 1-D persistent positions buffer expect. We forward raw tokens
  // instead of a host-computed embedding: the language model runs embed_tokens
  // itself (identical to get_input_embeddings for a plain token lookup), and
  // under graph mode it does so on the persistent token buffer, keeping the
  // captured input address stable across double-buffered slots (a fresh host
  // embedding tensor would be reallocated every step and make graph replay read
  // a stale buffer).
  if (params.meta.batch_forward_type.is_decode() && !mm_data.valid()) {
    const torch::Tensor decode_positions =
        collapse_mrope_decode_positions(positions);
    if (llm_executor_) {
      return llm_executor_->run(tokens, decode_positions, kv_caches, params);
    }
    return model_->forward(tokens, decode_positions, kv_caches, params);
  }

  // Prefill / multimodal step: resolve encoder embeddings on the host (needed
  // to merge vision tokens into the token stream) and keep the real
  // [3, num_tokens] mRoPE rows. Under graph mode the executor demotes this
  // non-decode batch to eager forward.
  if (encoder_cache_) {
    EncoderCacheLookupVisitor lookup(encoder_cache_.get());
    mm_data.foreach (lookup);
  }

  EncoderInputGatherVisitor input_gather;
  mm_data.foreach (input_gather);
  CHECK(input_gather.finish(mm_data));

  MMDict embedding = encode(params);
  EncoderOutputScatterVisitor scatter(embedding);
  mm_data.foreach (scatter);
  CHECK(scatter.finish());

  if (encoder_cache_) {
    EncoderCacheInsertVisitor insert(encoder_cache_.get());
    mm_data.foreach (insert);
  }

  EncoderEmbeddingGatherVisitor gather(device_,
                                       mm_data.type(),
                                       params.attention.host.kv_seq_lens,
                                       params.attention.host.q_seq_lens);
  mm_data.foreach (gather);
  CHECK(gather.finish(mm_data));

  params.embedding.input_embedding =
      model_->get_input_embeddings(tokens, params);

  if (llm_executor_) {
    return llm_executor_->run(tokens, positions, kv_caches, params);
  }

  return model_->forward(tokens, positions, kv_caches, params);
}

void VlmExecutorImpl::prepare_graph_input(const torch::Tensor& tokens,
                                          const torch::Tensor& positions,
                                          std::vector<KVCache>& kv_caches,
                                          const ModelInputParams& params) {
  // Decode-only double-buffer graph prewarm. Delegate to the inner graph
  // executor built in enable_graph mode; multimodal prefill steps never reach
  // here because the worker gates this on decode-phase input params.
  if (llm_executor_) {
    // Prewarm only runs on decode batches, so match the 1-D positions the
    // graph's persistent buffer captures and replays; the worker otherwise
    // hands us the raw [3, num_tokens] mRoPE rows.
    llm_executor_->prepare_graph_input(
        tokens, collapse_mrope_decode_positions(positions), kv_caches, params);
  }
}

torch::Tensor VlmExecutorImpl::collapse_mrope_decode_positions(
    const torch::Tensor& positions) const {
  // mRoPE builds positions as [3, num_tokens]; on decode the three rows are
  // identical, so row 0 is the 1-D position vector ordinary rope and the
  // graph's persistent buffer expect. Guard on the 2-D shape so already-1-D
  // (non-mRoPE) positions pass through untouched.
  if (args_.rope_scaling_rope_type() == "mrope" && positions.dim() == 2) {
    return positions[0].contiguous();
  }
  return positions;
}

}  // namespace xllm
