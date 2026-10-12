/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/runtime/py_executor_impl.h"

#include <glog/logging.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/kv_cache/kv_shard_layout.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "core/framework/speculative/mtp_async_state.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/kv_shard_batch_metadata.h"
#include "core/runtime/py_attention_metadata.h"
#include "core/runtime/py_input_batch.h"
#include "core/util/pybind_helper.h"
#include "models/llm/py_causal_lm.h"
#include "platform/layer_synchronizer.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "core/kernels/npu/utils.h"
#include "platform/npu/npu_layer_synchronizer.h"
#endif

namespace py = pybind11;

namespace xllm {

namespace {

class AttentionMetadataView final {
 public:
  explicit AttentionMetadataView(
      std::shared_ptr<layer::AttentionMetadata> metadata)
      : metadata_(std::move(metadata)),
        kv_seq_lens_host_(make_kv_seq_lens_host(metadata_)) {}

  AttentionMetadataView(std::shared_ptr<layer::AttentionMetadata> metadata,
                        const ModelInputParams& params)
      : AttentionMetadataView(std::move(metadata)) {
    // Kimi K3 uses this compact metadata view because its KDA layers carry
    // recurrent-state scheduling separately.  Keep the DP protocol metadata
    // here as well: EP is global across DP groups, so Python MoE must know
    // when another DP peer is in prefill versus decode.
    dp_execution_token_counts_ = params.parallel.dp_global_token_nums;
    for (int32_t& count : dp_execution_token_counts_) {
      if (count == 0) {
        count = 1;
      }
    }
    dp_is_decode_ = params.parallel.dp_is_decode;
  }

  const torch::Tensor& slot_mapping() const { return metadata_->slot_mapping; }
  const torch::Tensor& paged_kv_indptr() const {
    return metadata_->paged_kv_indptr;
  }
  const torch::Tensor& paged_kv_indices() const {
    return metadata_->paged_kv_indices;
  }
  const torch::Tensor& paged_kv_last_page_len() const {
    return metadata_->paged_kv_last_page_len;
  }
  py::object qo_indptr() const {
    if (!metadata_->qo_indptr.has_value() || !metadata_->qo_indptr->defined()) {
      return py::none();
    }
    return py::cast(*metadata_->qo_indptr);
  }
  py::object q_cu_seq_lens() const {
    return optional_tensor(metadata_->q_cu_seq_lens);
  }
  py::object kv_cu_seq_lens() const {
    return optional_tensor(metadata_->kv_cu_seq_lens);
  }
  py::object kv_seq_lens_host() const {
    return optional_tensor(kv_seq_lens_host_);
  }
  py::object block_table() const {
    return optional_tensor(metadata_->block_table);
  }
  py::object kv_seq_lens() const {
    return optional_tensor(metadata_->kv_seq_lens);
  }
  bool is_prefill() const { return metadata_->is_prefill; }
  bool is_chunked_prefill() const { return metadata_->is_chunked_prefill; }
  const std::vector<int32_t>& dp_execution_token_counts() const {
    return dp_execution_token_counts_;
  }
  const std::vector<int32_t>& dp_is_decode() const { return dp_is_decode_; }
  bool is_mixed() const { return metadata_->is_mixed; }

 private:
  static torch::Tensor make_kv_seq_lens_host(
      const std::shared_ptr<layer::AttentionMetadata>& metadata) {
    if (metadata->is_dummy) {
      return torch::zeros(
          {2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    }
    if (metadata->kv_seq_lens_vec.empty()) {
      return torch::Tensor();
    }

    std::shared_ptr<layer::AttentionMetadata> owner = metadata;
    return torch::from_blob(
        metadata->kv_seq_lens_vec.data(),
        {static_cast<int64_t>(metadata->kv_seq_lens_vec.size())},
        [owner = std::move(owner)](void*) mutable { owner.reset(); },
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  }

  static py::object optional_tensor(const torch::Tensor& tensor) {
    return tensor.defined() ? py::cast(tensor) : py::none();
  }

  std::shared_ptr<layer::AttentionMetadata> metadata_;
  torch::Tensor kv_seq_lens_host_;
  std::vector<int32_t> dp_execution_token_counts_;
  std::vector<int32_t> dp_is_decode_;
};

// Read-only per-step view of the linear-attention (KDA) scheduling info the C++
// runtime already computes for GDN/linear layers. Mirrors the fields of the
// Python KimiK3KDAMetadata (xllm/python/layers/kda.py). Tensors are
// materialized in the constructor so the view does not outlive `params`.
class KimiK3KDAMetadataView final {
 public:
  explicit KimiK3KDAMetadataView(const ModelInputParams& params) {
    // state_indices: per-sequence linear-state slot id, [num_seqs] int64.
    state_indices_ = params.embedding.linear_state_indices;
    if (state_indices_.defined()) {
      state_indices_ = state_indices_.to(torch::kInt64);
    }

#if defined(USE_NPU)
    // query_start_loc / has_initial_state are host-side vectors populated by
    // WorkerImpl::prepare_input_params_for_linear_attention on the NPU path.
    const std::vector<int64_t>& qsl = params.parallel.query_start_loc;
    if (!qsl.empty()) {
      query_start_loc_ =
          torch::tensor(qsl, torch::TensorOptions().dtype(torch::kInt64))
              .to(torch::kInt32);
    }
    const std::vector<int64_t>& his = params.parallel.has_initial_state;
    if (!his.empty()) {
      has_initial_state_ =
          torch::tensor(his, torch::TensorOptions().dtype(torch::kInt64))
              .to(torch::kBool);
    }
#endif

    // Decode/prefill sequence split. Decodes are ordered before prefills, and
    // pure batches (the only supported paths) map directly from the batch type.
    const int32_t num_sequences = params.meta.num_sequences;
    const BatchForwardType batch_type = params.meta.batch_forward_type;
    if (batch_type.is_decode()) {
      num_decode_seqs_ = num_sequences;
      num_prefill_seqs_ = 0;
    } else if (batch_type.no_decode()) {
      num_decode_seqs_ = 0;
      num_prefill_seqs_ = num_sequences;
    } else {
      // MIXED: decodes lead the batch as single-token sequences. Count them via
      // query_start_loc deltas of 1; the remainder are prefills.
      int32_t num_decode = 0;
#if defined(USE_NPU)
      const std::vector<int64_t>& qsl = params.parallel.query_start_loc;
      for (size_t i = 1; i < qsl.size(); ++i) {
        if (qsl[i] - qsl[i - 1] != 1) {
          break;
        }
        ++num_decode;
      }
#endif
      num_decode_seqs_ = num_decode;
      num_prefill_seqs_ = num_sequences - num_decode;
    }

    // DP decode gating: the engine host-syncs every DP rank's batch type into
    // dp_is_decode, so no collective is needed here. The python decode graph
    // may only run when all DP ranks decode, keeping HCCL usage consistent
    // across ranks (mirrors AclGraphExecutorImpl's all-ranks-decode gate).
    const std::vector<int32_t>& dp_is_decode = params.parallel.dp_is_decode;
    const std::vector<int32_t>& dp_token_nums =
        params.parallel.dp_global_token_nums;
    dp_metadata_valid_ =
        !dp_token_nums.empty() && dp_is_decode.size() == dp_token_nums.size();
    all_dp_decode_ = dp_metadata_valid_ &&
                     std::all_of(dp_is_decode.begin(),
                                 dp_is_decode.end(),
                                 [](int32_t value) { return value != 0; });
    graph_num_tokens_ =
        dp_token_nums.empty()
            ? num_sequences
            : *std::max_element(dp_token_nums.begin(), dp_token_nums.end());
    empty_shard_ = num_sequences == 0;
  }

  py::object query_start_loc() const {
    return query_start_loc_.defined() ? py::cast(query_start_loc_) : py::none();
  }
  py::object state_indices() const {
    return state_indices_.defined() ? py::cast(state_indices_) : py::none();
  }
  py::object has_initial_state() const {
    return has_initial_state_.defined() ? py::cast(has_initial_state_)
                                        : py::none();
  }
  int32_t num_decode_seqs() const { return num_decode_seqs_; }
  int32_t num_prefill_seqs() const { return num_prefill_seqs_; }
  bool dp_metadata_valid() const { return dp_metadata_valid_; }
  bool all_dp_decode() const { return all_dp_decode_; }
  int32_t graph_num_tokens() const { return graph_num_tokens_; }
  bool empty_shard() const { return empty_shard_; }

 private:
  torch::Tensor query_start_loc_;
  torch::Tensor state_indices_;
  torch::Tensor has_initial_state_;
  int32_t num_decode_seqs_ = 0;
  int32_t num_prefill_seqs_ = 0;
  bool dp_metadata_valid_ = false;
  bool all_dp_decode_ = true;
  int32_t graph_num_tokens_ = 0;
  bool empty_shard_ = false;
};

// Slice each modality's embedding blocks to the in-chunk subrange on the
// scheduled items (drives ChunkEmbedSliceVisitor, the host-side twin of the
// C++ VLM path's gather visitor). Chunked prefill — where a chunk boundary
// can land inside an item's span — then scatters only the features whose
// placeholders are in `tokens`; fully scheduled items pass through unchanged,
// so the non-chunked case is a no-op.
torch::Tensor slice_chunk_embeds(MMBatchData& mm_data,
                                 const torch::Tensor& embeds,
                                 MMType modality) {
  if (!embeds.defined() || embeds.dim() == 0 || embeds.size(0) == 0) {
    return embeds;
  }
  ChunkEmbedSliceVisitor visitor(embeds, modality);
  CHECK(mm_data.foreach (visitor));
  return visitor.finish();
}

void register_xllm_runtime_module(py::module_& m) {
  register_attention_metadata_views(m);
  register_input_batch_metadata_view(m);
  py::class_<LayerSynchronizer, std::shared_ptr<LayerSynchronizer>>(
      m, "LayerLoadSynchronizer")
      .def(
          "synchronize_layer",
          [](LayerSynchronizer& self, int64_t event_index) {
            CHECK_GE(event_index, 0);
            CHECK_LT(static_cast<uint64_t>(event_index), self.size());
            return self.synchronize_layer(event_index);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("size", &LayerSynchronizer::size);
  py::class_<AttentionMetadataView>(m, "KimiAttentionMetadataView")
      .def_property_readonly("slot_mapping",
                             &AttentionMetadataView::slot_mapping)
      .def_property_readonly("paged_kv_indptr",
                             &AttentionMetadataView::paged_kv_indptr)
      .def_property_readonly("paged_kv_indices",
                             &AttentionMetadataView::paged_kv_indices)
      .def_property_readonly("paged_kv_last_page_len",
                             &AttentionMetadataView::paged_kv_last_page_len)
      .def_property_readonly("qo_indptr", &AttentionMetadataView::qo_indptr)
      .def_property_readonly("q_cu_seq_lens",
                             &AttentionMetadataView::q_cu_seq_lens)
      .def_property_readonly("kv_cu_seq_lens",
                             &AttentionMetadataView::kv_cu_seq_lens)
      .def_property_readonly("kv_seq_lens_host",
                             &AttentionMetadataView::kv_seq_lens_host)
      .def_property_readonly("block_table", &AttentionMetadataView::block_table)
      .def_property_readonly("kv_seq_lens", &AttentionMetadataView::kv_seq_lens)
      .def_property_readonly("is_prefill", &AttentionMetadataView::is_prefill)
      .def_property_readonly("is_chunked_prefill",
                             &AttentionMetadataView::is_chunked_prefill)
      .def_property_readonly("dp_execution_token_counts",
                             &AttentionMetadataView::dp_execution_token_counts)
      .def_property_readonly("dp_is_decode",
                             &AttentionMetadataView::dp_is_decode)
      .def_property_readonly("is_mixed", &AttentionMetadataView::is_mixed);

  // Kimi K3 KDA per-step scheduling view.
  py::class_<KimiK3KDAMetadataView>(m, "KimiK3KDAMetadataView")
      .def_property_readonly("query_start_loc",
                             &KimiK3KDAMetadataView::query_start_loc)
      .def_property_readonly("state_indices",
                             &KimiK3KDAMetadataView::state_indices)
      .def_property_readonly("has_initial_state",
                             &KimiK3KDAMetadataView::has_initial_state)
      .def_property_readonly("num_decode_seqs",
                             &KimiK3KDAMetadataView::num_decode_seqs)
      .def_property_readonly("num_prefill_seqs",
                             &KimiK3KDAMetadataView::num_prefill_seqs)
      .def_property_readonly("dp_metadata_valid",
                             &KimiK3KDAMetadataView::dp_metadata_valid)
      .def_property_readonly("all_dp_decode",
                             &KimiK3KDAMetadataView::all_dp_decode)
      .def_property_readonly("graph_num_tokens",
                             &KimiK3KDAMetadataView::graph_num_tokens)
      .def_property_readonly("empty_shard",
                             &KimiK3KDAMetadataView::empty_shard);

  m.def("native_collectives_available",
        []() { return PyCausalLM::active_instance() != nullptr; });

  m.def("tp_all_reduce", [](torch::Tensor tensor) {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm != nullptr) {
      py_causal_lm->tp_all_reduce(tensor);
    }
    return tensor;
  });
  m.def("tp_all_gather", [](torch::Tensor tensor, int64_t dim) {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm != nullptr) {
      return py_causal_lm->tp_all_gather(tensor, dim);
    }
    return tensor;
  });
  m.def("dp_all_gather",
        [](torch::Tensor tensor,
           const std::vector<int32_t>& execution_token_counts) {
          PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
          if (py_causal_lm != nullptr) {
            return py_causal_lm->dp_all_gather(tensor, execution_token_counts);
          }
          return tensor;
        });
  m.def("moe_tp_all_reduce", [](torch::Tensor tensor) {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm != nullptr) {
      py_causal_lm->moe_tp_all_reduce(tensor);
    }
    return tensor;
  });
  m.def("moe_ep_all_reduce", [](torch::Tensor tensor) {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm != nullptr) {
      py_causal_lm->moe_ep_all_reduce(tensor);
    }
    return tensor;
  });
  m.def("eplb_batch_isend_irecv",
        [](py::list operation_types, py::list tensors, py::list remote_ranks) {
          PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
          CHECK(py_causal_lm != nullptr)
              << "EPLB P2P transfer requires an active PyCausalLM.";
          py_causal_lm->eplb_batch_isend_irecv(
              operation_types, tensors, remote_ranks);
        });
  m.def("eplb_wait_batch_isend_irecv", []() {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    CHECK(py_causal_lm != nullptr)
        << "EPLB P2P transfer requires an active PyCausalLM.";
    py_causal_lm->eplb_wait_batch_isend_irecv();
  });

#if defined(USE_NPU)
  m.def("is_ascend_a3", &kernel::npu::is_ascend_a3);
  py::class_<NPULayerSynchronizerImpl,
             std::shared_ptr<NPULayerSynchronizerImpl>>(m, "LayerSynchronizer")
      .def("record_event",
           [](NPULayerSynchronizerImpl& self, int64_t layer_id) {
             int32_t device_id = static_cast<int32_t>(
                 c10_npu::getCurrentNPUStream().device_index());
             return self.record_event(layer_id, device_id);
           });
#endif
}

}  // namespace

PYBIND11_EMBEDDED_MODULE(xllm_runtime, m) { register_xllm_runtime_module(m); }

PyExecutorImpl::PyExecutorImpl(CausalLM* model,
                               const ModelArgs& args,
                               const torch::Device& device,
                               const runtime::Options& options)
    : py_model_bridge_(dynamic_cast<PyModelBridge*>(model)),
      py_causal_lm_(dynamic_cast<PyCausalLM*>(model)),
      args_(args),
      device_(device),
      options_(options),
      enable_mla_(args.enable_mla()),
      has_kda_layers_((args.model_type() == "kimi_k3" ||
                       args.model_type() == "kimi_linear") &&
                      has_linear_attention_layers(args)) {
  CHECK(py_model_bridge_ != nullptr)
      << "PyExecutorImpl requires a Python model bridge";

  py::gil_scoped_acquire gil;
  py::module_::import("xllm_runtime");
  py::module_ executor_module =
      py::module_::import("xllm.python.model_executor.executor");
  const bool is_spec_draft =
      options_.is_draft_engine() || (options_.enable_speculative_decode() &&
                                     options_.num_decoding_tokens() == 1);
  py_executor_ = executor_module.attr("ModelExecutor")(
      py_model_bridge_->python_model(),
      py_model_bridge_->config_dict(),
      options_.max_seqs_per_batch(),
      options_.num_decoding_tokens(),
      ExecutionConfig::get_instance().acl_graph_decode_batch_size_limit(),
      py::arg("is_spec_draft") = is_spec_draft);
}

PyExecutorImpl::~PyExecutorImpl() {
  if (PyCausalLM::active_instance() == py_causal_lm_) {
    PyCausalLM::set_active_instance(nullptr);
  }
  clear_python_object(py_executor_);
}

ForwardInput PyExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput PyExecutorImpl::run(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& params) {
  torch::NoGradGuard no_grad;
  COUNTER_INC(num_model_execution_total_eager);
  PyCausalLM::set_active_instance(py_causal_lm_);

  // Build or reuse attention metadata.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      params.attn_metadata;
  if (!attn_metadata) {
    attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(
            params, enable_mla_, std::nullopt, tokens.device()));
  }
  // The dense-DCP case is a model with no MLA latent cache (Qwen3.5), which
  // needs KV-shard metadata on every step. An MLA model with cp_size == 1 has
  // no CP group for the DSA prefill to shard across, so attaching the metadata
  // there only trips the sharded-MLA-prefill guard in npu_paged_attention.py;
  // keep the pre-main behavior for MLA and require the active CP group.
  const bool is_decode_context_parallel =
      !enable_mla_ && py_causal_lm_ != nullptr &&
      py_causal_lm_->cp_size() == 1 && py_causal_lm_->kv_split_size() > 1;
  const bool is_mla_prefill_kv_shard =
      enable_mla_ && py_causal_lm_ != nullptr && py_causal_lm_->cp_size() > 1 &&
      py_causal_lm_->kv_split_size() > 1 &&
      (attn_metadata->is_prefill || attn_metadata->is_chunked_prefill);
  if (is_decode_context_parallel || is_mla_prefill_kv_shard) {
    const KVShardLayout layout(options_.block_size(),
                               py_causal_lm_->kv_split_size(),
                               py_causal_lm_->kv_split_rank());
    attn_metadata->kv_shard_batch_metadata =
        layer::build_kv_shard_batch_metadata(*attn_metadata, layout);
  }

  py::gil_scoped_acquire gil;

  // Lazy bind KV caches on first call.
  int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  if (!kv_bound_) {
    py::list kv_caches_py;
    for (auto& kv : kv_caches) {
      // Slot order must match ``LayerCache`` on the Python side.
      // Keep this order synchronized with LayerCache/_LAYER_CACHE_SLOTS.
      // Generic caches use the first five entries; DeepSeek-V4 uses the
      // next six, and GLM5 uses the final KPOOL_TAIL entry.
      kv_caches_py.append(
          py::make_tuple(optional_tensor(kv.get_k_cache()),
                         optional_tensor(kv.get_v_cache()),
                         optional_tensor(kv.get_index_cache()),
                         optional_tensor(kv.get_conv_cache()),
                         optional_tensor(kv.get_ssm_cache()),
                         optional_tensor(kv.get_swa_cache()),
                         optional_tensor(kv.get_compress_kv_state()),
                         optional_tensor(kv.get_compress_score_state()),
                         optional_tensor(kv.get_compress_index_kv_state()),
                         optional_tensor(kv.get_compress_index_score_state()),
                         optional_tensor(kv.get_indexer_cache_scale()),
                         optional_tensor(kv.get_kpool_tail())));
    }
    py_executor_.attr("bind_kv_caches")(kv_caches_py);

    // Bind linear-attention (KDA) conv/recurrent caches by global layer id.
    // Full-attention layers own no conv/ssm tensors, so only linear layers are
    // forwarded to the Python KDA runtime.
    if (has_kda_layers_) {
      py::list kda_caches_py;
      for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
        if (is_full_attention_layer(args_, layer_id)) {
          continue;
        }
        KVCache& kv = kv_caches[static_cast<size_t>(layer_id)];
        kda_caches_py.append(
            py::make_tuple(layer_id, kv.get_conv_cache(), kv.get_ssm_cache()));
      }
      py_executor_.attr("bind_kda_caches")(kda_caches_py);
    }

    kv_bound_ = true;
    kv_layer_count_ = num_layers;
  } else {
    CHECK_EQ(num_layers, kv_layer_count_)
        << "KV cache layer count changed after initial bind";
  }

  py::object py_metadata =
      has_kda_layers_ ? py::cast(AttentionMetadataView(attn_metadata, params))
                      : py::cast(PyAttentionMetadataView(
                            attn_metadata, params, args_.dummy_token_count()));
  py::object py_input_batch_metadata =
      py::cast(PyInputBatchMetadataView(params));
  py::object input_embedding =
      optional_tensor(params.embedding.input_embedding);
  torch::Tensor execution_tokens =
      mtp_async::materialize_graph_speculative_verify_tokens(tokens,
                                                             params.graph);

  // Per-step KDA scheduling info (linear-state slots, query_start_loc,
  // has_initial_state, decode/prefill split). None for non-KDA models.
  py::object py_kda_metadata =
      has_kda_layers_ ? py::object(py::cast(KimiK3KDAMetadataView(params)))
                      : py::object(py::none());

  // --- VLM: vision encode + embedding merge on image/video prefill steps ---
  // On steps carrying multimodal input, ``params.multimodal.mm_data`` holds the
  // batched ``pixel_values`` + ``image_grid_thw`` (still images) and/or
  // ``pixel_values_videos`` + ``video_grid_thw`` (video) — same accessors the
  // C++ Qwen3-VL base uses in qwen3_vl_base.h. Drive the Python model's
  // ``encode`` -> ``get_input_embeddings`` pipeline: the latter scatters each
  // modality's embeddings at its placeholder-token positions and sets
  // ``model._inputs_embeds`` / ``deepstack_input_embeds`` for the runner-driven
  // ``Qwen3VLModel.forward``. Decode steps carry no mm_data, so the attributes
  // stay clear and the aclgraph embed path is used.
  //
  // NOTE: this scatters the FULL image/video embedding into the current
  // forward's tokens, so it assumes every multimodal token is in this batch
  // (i.e. enable_chunked_prefill=False). Chunked prefill — where a chunk
  // boundary can land inside an item's token span — needs item-level scatter
  // (reuse EncoderEmbeddingGatherVisitor + the NPU backend's paged mixed-batch
  // attention, both tracked for a follow-up PR).
  //
  // TODO: refactor the per-modality blocks below into a generic handoff.
  auto& mm_data = params.multimodal.mm_data;
  if (py_causal_lm_ != nullptr && mm_data.valid()) {
    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values")) {
      pixel_values = res.value();
    }
    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw")) {
      image_grid_thw = res.value();
    }
    torch::Tensor pixel_values_videos;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values_videos")) {
      pixel_values_videos = res.value();
    }
    torch::Tensor video_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("video_grid_thw")) {
      video_grid_thw = res.value();
    }
    torch::Tensor input_features;
    if (const auto& res = mm_data.get<torch::Tensor>("input_features")) {
      input_features = res.value();
    }
    torch::Tensor speech_lengths;
    if (const auto& res = mm_data.get<torch::Tensor>("speech_lengths")) {
      speech_lengths = res.value();
    }
    CHECK(input_features.defined() == speech_lengths.defined())
        << "input_features and speech_lengths must be provided together";

    if (pixel_values.defined() || pixel_values_videos.defined() ||
        input_features.defined()) {
      py::object top_model = py_causal_lm_->python_model();
      // encode() moves the tensors onto device internally. Slice each block to
      // the chunk's in-chunk subrange (see slice_chunk_embeds) so chunked
      // prefill does not feed full image/video features into a partial
      // placeholder span.
      py::object image_embeds = py::none();
      if (pixel_values.defined() && image_grid_thw.defined()) {
        torch::Tensor raw =
            top_model.attr("encode")(pixel_values, image_grid_thw)
                .cast<torch::Tensor>();
        image_embeds =
            py::cast(slice_chunk_embeds(mm_data, raw, MMType::IMAGE));
      }
      py::object video_embeds = py::none();
      if (pixel_values_videos.defined() && video_grid_thw.defined()) {
        torch::Tensor raw =
            top_model.attr("encode")(pixel_values_videos, video_grid_thw)
                .cast<torch::Tensor>();
        video_embeds =
            py::cast(slice_chunk_embeds(mm_data, raw, MMType::VIDEO));
      }
      py::object audio_embeds = py::none();
      py::object audio_mask = py::none();
      if (input_features.defined() && speech_lengths.defined()) {
        // Per-audio [hash, ctc_pad_num] rows, in item order.
        torch::Tensor audio_meta;
        if (const auto& res = mm_data.get<torch::Tensor>("audio_encode_meta")) {
          audio_meta = res.value();
        }
        CHECK(audio_meta.defined() &&
              audio_meta.size(0) == speech_lengths.numel())
            << "audio_encode_meta missing or misaligned with speech_lengths";
        torch::Tensor raw =
            top_model.attr("encode")(input_features, speech_lengths, audio_meta)
                .cast<torch::Tensor>();
        audio_embeds =
            py::cast(slice_chunk_embeds(mm_data, raw, MMType::AUDIO));
        // Chunk replacement mask: the Python merge scatters the sliced rows
        // at these positions (item state, not pad-id matching).
        AudioScatterMaskVisitor mask_visitor(
            /*seq_lens=*/params.attention.host.kv_seq_lens,
            /*scheduled_seq_lens=*/params.attention.host.q_seq_lens,
            execution_tokens);
        CHECK(mm_data.foreach (mask_visitor));
        audio_mask = py::cast(mask_visitor.finish());
      }
      // Sets top_model.model._inputs_embeds + deepstack_input_embeds.
      if (audio_embeds.is_none()) {
        top_model.attr("get_input_embeddings")(
            execution_tokens, image_embeds, video_embeds);
      } else {
        top_model.attr("get_input_embeddings")(execution_tokens,
                                               image_embeds,
                                               video_embeds,
                                               audio_embeds,
                                               audio_mask);
      }
    }
  }

  // --- mRoPE: collapse [3, N] decode positions to 1-D ---
  // Only PURE decode collapses to 1-D: decode rows are identical
  // (batch_input_builder get_mrope_positions), and mRoPE(p,p,p) == standard
  // RoPE at p, so a single row feeds the captured aclgraph's 1-D
  // static_positions unchanged. Chunked/mixed prefill (is_prefill=false but
  // is_chunked_prefill=true) still needs the full [3, N] for the Python mRoPE
  // path, so it is excluded here. The 2-D shape itself is the mRoPE signal:
  // non-mRoPE models never receive 2-D positions, so no config flag is needed.
  torch::Tensor positions_arg = positions;
  if (positions.dim() == 2 && !attn_metadata->is_prefill &&
      !attn_metadata->is_chunked_prefill) {
    positions_arg = positions.slice(/*dim=*/0, /*start=*/0, /*end=*/1)
                        .squeeze(0)
                        .contiguous();
  }

  py::object py_sync = py::none();
  // Host-load waits are separate from the PD push completion notifications.
  // Python eager waits before each attention; graph execution waits for all
  // loads before warmup/capture/replay, outside the recorded device graph.
  py::object py_load_sync = py::none();
  if (params.parallel.layer_wise_load_synchronizer) {
    py_load_sync = py::cast(params.parallel.layer_wise_load_synchronizer);
  }
#if defined(USE_NPU)
  if (params.parallel.layer_synchronizer) {
    py_sync = py::cast(params.parallel.layer_synchronizer);
  }
#endif

  py::object expert_load_data = optional_tensor(params.expert.expert_load_data);
  py::object eplb_decode_token_mask =
      optional_tensor(params.expert.eplb_decode_token_mask);

  // Execute: one C++ -> Python call per step. input_embedding stays None for
  // the Qwen3-VL python path (embeddings are merged via the attribute set by
  // get_input_embeddings above), so the runner takes the 2-arg model() branch
  // and Qwen3VLModel.forward reads _inputs_embeds. positions_arg carries the
  // mRoPE [3,N]->1-D decode collapse.
  py::object hidden_obj =
      py_executor_.attr("execute")(execution_tokens,
                                   positions_arg,
                                   py_metadata,
                                   input_embedding,
                                   py_sync,
                                   expert_load_data,
                                   eplb_decode_token_mask,
                                   params.meta.is_graph_warmup,
                                   py_input_batch_metadata,
                                   py_load_sync,
                                   params.parallel.layers_per_event,
                                   params.parallel.draft_load_event_index,
                                   py::arg("kda_metadata") = py_kda_metadata);
  if (py::isinstance<py::tuple>(hidden_obj)) {
    py::tuple output = hidden_obj.cast<py::tuple>();
    CHECK_EQ(output.size(), 2) << "Python model tuple output must be "
                                  "(hidden_states, aux_hidden_states)";
    return ModelOutput(output[0].cast<torch::Tensor>(),
                       torch::Tensor(),
                       output[1].cast<torch::Tensor>());
  }
  return ModelOutput(hidden_obj.cast<torch::Tensor>());
}

}  // namespace xllm
