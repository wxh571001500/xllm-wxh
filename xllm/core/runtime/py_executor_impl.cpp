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

#include "py_executor_impl.h"

#include <glog/logging.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "common/metrics.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "models/py_model_bridge.h"

namespace py = pybind11;

namespace xllm {

namespace {

class AttentionMetadataView final {
 public:
  explicit AttentionMetadataView(
      std::shared_ptr<layer::AttentionMetadata> metadata)
      : metadata_(std::move(metadata)),
        kv_seq_lens_host_(make_kv_seq_lens_host(metadata_)) {}

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

 private:
  static torch::Tensor make_kv_seq_lens_host(
      const std::shared_ptr<layer::AttentionMetadata>& metadata) {
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
    all_dp_decode_ = std::all_of(dp_is_decode.begin(),
                                 dp_is_decode.end(),
                                 [](int32_t v) { return v != 0; });
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
  bool all_dp_decode() const { return all_dp_decode_; }

 private:
  torch::Tensor query_start_loc_;
  torch::Tensor state_indices_;
  torch::Tensor has_initial_state_;
  int32_t num_decode_seqs_ = 0;
  int32_t num_prefill_seqs_ = 0;
  bool all_dp_decode_ = true;
};

}  // namespace

PYBIND11_EMBEDDED_MODULE(xllm_runtime, m) {
  py::class_<AttentionMetadataView>(m, "AttentionMetadataView")
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
                             &AttentionMetadataView::is_chunked_prefill);

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
      .def_property_readonly("all_dp_decode",
                             &KimiK3KDAMetadataView::all_dp_decode);
}

PyExecutorImpl::PyExecutorImpl(CausalLM* model,
                               const ModelArgs& args,
                               const torch::Device& device,
                               const runtime::Options& options)
    : py_model_bridge_(dynamic_cast<PyModelBridge*>(model)),
      args_(args),
      options_(options),
      enable_mla_(args.enable_mla()),
      has_kda_layers_(has_linear_attention_layers(args)) {
  CHECK(py_model_bridge_ != nullptr)
      << "PyExecutorImpl requires a Python model bridge";

  py::gil_scoped_acquire gil;
  py::module_::import("xllm_runtime");
  py::module_ executor_module =
      py::module_::import("xllm.python.model_executor.executor");
  py_executor_ =
      executor_module.attr("ModelExecutor")(py_model_bridge_->python_model(),
                                            py_model_bridge_->config_dict(),
                                            options_.max_seqs_per_batch());
}

PyExecutorImpl::~PyExecutorImpl() {
  py::gil_scoped_acquire gil;
  py_executor_ = py::object();
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

  // Build or reuse attention metadata.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      params.attn_metadata;
  if (!attn_metadata) {
    attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(
            params, enable_mla_, std::nullopt, tokens.device()));
  }

  py::gil_scoped_acquire gil;

  // Lazy bind KV caches on first call.
  int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  if (!kv_bound_) {
    py::list kv_caches_py;
    for (auto& kv : kv_caches) {
      kv_caches_py.append(py::make_tuple(
          kv.get_k_cache(), kv.get_v_cache(), kv.get_index_cache()));
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

  py::object py_metadata = py::cast(AttentionMetadataView(attn_metadata));

  // Per-step KDA scheduling info (linear-state slots, query_start_loc,
  // has_initial_state, decode/prefill split). None for non-KDA models.
  py::object py_kda_metadata =
      has_kda_layers_ ? py::object(py::cast(KimiK3KDAMetadataView(params)))
                      : py::object(py::none());

  // Execute: one C++ -> Python call per step.
  py::object input_embedding =
      params.embedding.input_embedding.defined()
          ? py::object(py::cast(params.embedding.input_embedding))
          : py::object(py::none());
  py::object hidden_obj = py_executor_.attr("execute")(
      tokens, positions, py_metadata, input_embedding, py_kda_metadata);
  return ModelOutput(hidden_obj.cast<torch::Tensor>());
}

}  // namespace xllm
