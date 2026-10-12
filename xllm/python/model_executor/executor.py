# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from torch import nn

from scripts.logger import logger
from xllm.python import distributed
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    KVCache,
    LayerCacheInput,
    normalize_layer_caches,
)
from xllm.python.layers.attention import Attention, AttentionLayerSpec, AttentionRuntimeLayer
from xllm.python.model_executor.execution_context import (
    LayerCacheAwareExecutionMetadataBuilder,
)
from xllm.python.model_executor.forward_context import (
    EplbRuntimeState,
    LayerLoadContext,
    LayerLoadSynchronizer,
    LayerSynchronizer,
)
from xllm.python.model_executor.input_batch import (
    InputBatch,
    InputBatchMetadata,
)
from xllm.python.model_executor.runners.base import ModelExecutionOutput
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.platform import current_platform
from xllm.python.registry import get_execution_metadata_builder_classes

# Backend strings that mean "no graph execution" (matched case-insensitively).
_DISABLED_GRAPH_BACKENDS = ("", "off", "none", "0")


def _is_deepseek_v4_model_type(model_type: str) -> bool:
    return model_type.startswith("deepseek_v4")


def _is_qwen3_5_model_type(model_type: str) -> bool:
    return model_type in {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }


def _validate_qwen3_5_dcp_config(
    config: dict,
    dcp_size: int,
) -> None:
    tp_size = int(config.get("tp_size", 1))
    if dcp_size > tp_size or tp_size % dcp_size:
        raise ValueError(f"Qwen3.5 DCP size must divide TP size: dcp_size={dcp_size}, tp_size={tp_size}")
    num_kv_heads_value = config.get("n_kv_heads")
    if num_kv_heads_value is None:
        num_kv_heads_value = config.get("num_key_value_heads")
    if num_kv_heads_value is None:
        raise ValueError("Qwen3.5 DCP requires the global KV-head count")
    num_kv_heads = int(num_kv_heads_value)
    if num_kv_heads <= 0:
        raise ValueError("Qwen3.5 DCP requires a positive global KV-head count")
    if num_kv_heads >= tp_size:
        kv_head_replicas = 1
    else:
        if tp_size % num_kv_heads:
            raise ValueError(
                "Qwen3.5 DCP requires TP size to be divisible by the global "
                f"KV-head count: tp_size={tp_size}, num_kv_heads={num_kv_heads}"
            )
        kv_head_replicas = tp_size // num_kv_heads
    if kv_head_replicas % dcp_size:
        raise ValueError(
            "Qwen3.5 DCP size must divide the replicated KV-head group: "
            f"dcp_size={dcp_size}, kv_head_replicas={kv_head_replicas}"
        )
    if bool(config.get("enable_disagg_pd", False)):
        raise NotImplementedError("Qwen3.5 Python DCP does not yet support disaggregated P/D")


def _is_npu_device(device: torch.device) -> bool:
    return device.type in ("npu", "privateuseone")


def _resolve_graph_backend(config: dict, device: torch.device) -> str:
    graph_backend = str(config.get("python_graph_backend", "off")).lower()
    graph_disabled = graph_backend in ("", "off", "none", "0")
    if graph_disabled and config.get("enable_graph", False) and _is_npu_device(device):
        return "aclgraph"
    return graph_backend


def _split_prefill_compile_backend(
    graph_backend: str,
) -> tuple[str, str | None]:
    """Split ``aclgraph+inductor`` into decode graph and prefill compile.

    The combined form keeps the existing low-latency decode graph and uses a
    torch.compile backend only for dynamic prefill shapes.
    """
    if not graph_backend.startswith("aclgraph+"):
        return graph_backend, None

    prefill_backend = graph_backend.split("+", 1)[1]
    return "aclgraph", prefill_backend or "inductor"


def _create_attention_backend(
    first_attention: AttentionLayerSpec,
    device: torch.device,
    dtype: torch.dtype,
    config: dict | None = None,
    max_num_reqs: int = 1,
    num_decoding_tokens: int = 1,
    acl_graph_enabled: bool = False,
    *,
    attention_kinds: set[str] | None = None,
) -> AttentionBackend:
    config = config or {}
    model_type = config.get("model_type", "")
    # DeepSeek-V4.1 is NPU-only: reject on any other platform before the generic
    # CUDA/NPU backend selection can return a backend it must not run on.
    if model_type == "deepseek_v41" and not current_platform.is_npu():
        raise NotImplementedError(
            f"DeepSeek-V4.1 runs on NPU only; no attention backend available for device type '{device.type}'"
        )
    if model_type in {"kimi_k3", "kimi_linear"} and _is_npu_device(device):
        from xllm.python.attention.kimi_paged_attention import KimiPagedAttentionBackend

        return KimiPagedAttentionBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            device=device,
            dtype=dtype,
            has_mha_layers="mha" in (attention_kinds or {first_attention.kind}),
        )
    if _is_deepseek_v4_model_type(model_type) and current_platform.is_npu():
        if model_type == "deepseek_v41":
            # The V4.1 CSA2 backend is NPU-only; it runs eagerly on NPU until
            # the kernel milestone replaces its internals.
            from xllm.python.attention.csa2_attention import Csa2AttentionBackend
            from xllm.python.models.deepseek_v41 import _detect_v41_quant_mode

            # Derive the served context length here so the backend can default
            # its ACL-graph token-capacity granularity when the deployment does
            # not set XLLM_V41_GRAPH_CAPACITY_GRANULARITY explicitly (an empty
            # env value counts as unset). Same source as the decode graph
            # runner's max_model_len below.
            serving_len = int(config.get("max_position_embeddings", 0) or 0)
            # Register the checkpoint's quantization mode so the backend can
            # reject the fp8/none torch MoE from ACL-graph capture (51361671).
            # The value comes from the shared model config, so every DP/TP rank
            # derives the same capability and the whole DP group agrees on the
            # graph-vs-eager decision; no per-rank env is consulted.
            quant_mode = _detect_v41_quant_mode(config)
            return Csa2AttentionBackend(
                compress_ratios=list(config.get("compress_ratios", [])),
                window_size=int(config.get("window_size", 128)),
                n_layers=int(config.get("n_layers", config.get("num_hidden_layers", 0))),
                num_heads=first_attention.num_heads,
                attn_head_dim=first_attention.head_dim,
                index_topk=int(config.get("index_topk", 512)),
                index_n_heads=int(config.get("index_n_heads", 32)),
                index_head_dim=int(config.get("index_head_dim", 128)),
                rope_head_dim=int(config.get("qk_rope_head_dim", 64)),
                device=device,
                dtype=dtype,
                kv_source_layer_ids=list(config.get("kv_source_layer_ids", [])),
                index_source_layer_ids=list(config.get("index_source_layer_ids", [])),
                candidate_source_layer_id=int(config.get("candidate_source_layer_id", -1)),
                candidate_topk_blocks=int(config.get("candidate_topk_blocks", 0)),
                candidate_block_size=int(config.get("candidate_block_size", 0)),
                max_model_len=serving_len,
                quant_mode=quant_mode,
                acl_graph_enabled=acl_graph_enabled,
            )
        from xllm.python.attention.dsa_attention import DsaAttentionBackend

        return DsaAttentionBackend(
            compress_ratios=list(config.get("compress_ratios", [])),
            window_size=int(config.get("window_size", 128)),
            n_layers=int(config.get("n_layers", config.get("num_hidden_layers", 0))),
            num_heads=first_attention.num_heads,
            attn_head_dim=first_attention.head_dim,
            index_topk=int(config.get("index_topk", 512)),
            index_n_heads=int(config.get("index_n_heads", 64)),
            index_head_dim=int(config.get("index_head_dim", 128)),
            rope_head_dim=int(config.get("qk_rope_head_dim", 64)),
            device=device,
            dtype=dtype,
            dspark_block_size=int(config.get("dspark_block_size", 0)),
            dspark_use_native_sas=bool(config.get("dspark_use_native_sas", False)),
        )
    if current_platform.is_npu():
        dcp_group = distributed.dcp_group(device)
        if int(config.get("cp_size", 1)) == 1 and dcp_group is not None and dcp_group.size() > 1:
            if _is_qwen3_5_model_type(str(model_type)):
                if current_platform.get_ascend_soc_generation() == "a5":
                    raise NotImplementedError(
                        "Qwen3.5 Python DCP is not supported on Ascend950 because "
                        "its TND attention fallback does not return softmax LSE"
                    )
                _validate_qwen3_5_dcp_config(config, dcp_group.size())
                from xllm.python.attention.dense_dcp_backend import (
                    DenseDcpAttentionBackend,
                )

                return DenseDcpAttentionBackend(
                    num_heads=first_attention.num_heads,
                    num_kv_heads=first_attention.num_kv_heads,
                    head_dim=first_attention.head_dim,
                    scale=first_attention.scale,
                    sliding_window=first_attention.sliding_window,
                    device=device,
                    dtype=dtype,
                    dcp_group=dcp_group,
                    num_decoding_tokens=max(num_decoding_tokens, 1),
                )
            # Only an MLA model has a latent KV for the DCP backend to shard.
            # The DFlash2 draft is GQA: it is not sharded at all but keeps a
            # full replica of its own pool, read through the expanded block
            # table, so it skips this branch and falls through to the plain
            # paged backend below. A dense target has no such replica, so it
            # still has to refuse.
            if bool(config.get("enable_mla", False)):
                from xllm.python.attention.sfa_dcp_backend import (
                    SfaDcpAttentionBackend,
                    dcp_layer_options,
                )

                index_topk = dcp_layer_options(first_attention)
                return SfaDcpAttentionBackend(
                    num_heads=first_attention.num_heads,
                    num_kv_heads=first_attention.num_kv_heads,
                    head_dim=first_attention.head_dim,
                    scale=first_attention.scale,
                    sliding_window=first_attention.sliding_window,
                    device=device,
                    dtype=dtype,
                    dcp_group=dcp_group,
                    index_topk=index_topk,
                    max_num_reqs=max(max_num_reqs, 1),
                    num_decoding_tokens=max(num_decoding_tokens, 1),
                )
            if not bool(config.get("is_draft_engine", False)):
                raise NotImplementedError(f"Dense DCP attention is not supported for model type '{model_type}'")
        from xllm.python.attention.npu_paged_attention import (
            NpuPagedAttentionBackend,
        )

        dflash2_block_size = int(config.get("dflash2_block_size", 0))
        use_xfia_decode = False
        if dflash2_block_size > 0 and bool(config.get("dflash_xfia_enabled", False)):
            import xllm_runtime

            # C++ enable_dflash_proposal_xfia already applied the prelaunch gate.
            # Confirm the device here so a stale config cannot enable XFIA off A3.
            use_xfia_decode = xllm_runtime.is_ascend_a3()
        return NpuPagedAttentionBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            is_mla=bool(config.get("enable_mla", False)),
            device=device,
            dtype=dtype,
            num_decoding_tokens=num_decoding_tokens,
            xfia_query_width=max(dflash2_block_size, 1),
            use_xfia_decode=use_xfia_decode,
        )
    if device.type == "cuda":
        from xllm.python.attention.flashinfer import FlashInferBackend

        if first_attention.kind != "mha":
            raise NotImplementedError("CUDA Python backend does not support MLA")
        return FlashInferBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            device=device,
            dtype=dtype,
        )
    raise NotImplementedError(f"No attention backend available for device type '{device.type}'")


def _acl_graph_unsupported_reason(
    attention_layers: list[AttentionRuntimeLayer],
    *,
    supports_kimi_k3_graph: bool = False,
) -> str | None:
    """Return why ACL graph is unsafe for this model, or ``None``."""
    kinds = {layer.attention_kind for layer in attention_layers}
    if supports_kimi_k3_graph:
        unsupported_kinds = kinds.difference({"mha", "mla", "linear"})
        if unsupported_kinds:
            return f"ACL graph does not support attention kinds {sorted(unsupported_kinds)}"
        return None
    if "linear" in kinds:
        return "ACL graph does not support linear-attention runtime state"
    if kinds.difference({"mha"}):
        return "ACL graph currently supports MHA attention only"
    return None


class ModelExecutor:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        max_seqs_per_batch: int,
        num_decoding_tokens: int = 1,
        acl_graph_decode_batch_size_limit: int | None = None,
        *,
        is_spec_draft: bool = False,
    ) -> None:
        self.model = model
        self._kv_bound = False
        self._is_spec_draft = is_spec_draft
        self._requires_framework_kpool_tail = bool(config.get("requires_framework_kpool_tail", False))

        attention_layers = sorted(
            (module for module in model.modules() if isinstance(module, AttentionRuntimeLayer)),
            key=lambda layer: layer.layer_id,
        )
        if not attention_layers:
            raise ValueError("Python model does not contain a runtime attention layer")

        layer_specs = [layer.attention_layer_spec() for layer in attention_layers]
        layer_ids = [spec.layer_id for spec in layer_specs]
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError(f"Runtime attention layer ids must be unique: got {layer_ids}")

        first_parameter = next(model.parameters())
        device = first_parameter.device
        self._device = device
        self._attention_layer_specs = layer_specs
        self._num_attention_layers = len(layer_specs)
        # DP size from the C++ parallel properties (1 when DP is disabled).
        # Used to gate decode-graph execution on all DP ranks decoding.
        self._dp_size = int(config.get("dp_size", 1) or 1)
        # GLM-Next mixes DSA (MLA) and KDA (linear-attention) layers with
        # different head/dim configs; the paged backend only serves the DSA
        # layers, so it is built from the first DSA layer and the "identical
        # config across all layers" check is skipped. Non-GLM-Next models keep
        # the upstream behavior: backend from the first layer plus the
        # identical-config check. DSA layers are tagged with the
        # ``is_glm_next_mla`` class attribute so this dispatch does not have to
        # import glm5_next (that import pulls KDA kernel transitive deps and
        # fails on builds without them).
        dsa_layers = [layer for layer in attention_layers if getattr(layer, "is_glm_next_mla", False)]
        has_kda_layers = any(getattr(layer, "is_glm_next_kda", False) for layer in attention_layers)
        is_draft_engine = bool(config.get("is_draft_engine", False))
        adaptive_speculative_decode_enabled = bool(config.get("runtime_adaptive_speculative_decode_enabled", False))
        if has_kda_layers and not is_draft_engine and adaptive_speculative_decode_enabled:
            raise ValueError(
                "GLM5 KDA does not support adaptive speculative decode; disable adaptive speculative decoding."
            )
        self._is_kimi_k3 = hasattr(model.model, "kda_runtime")
        if self._is_kimi_k3:
            first_attention = next(spec for spec in layer_specs if spec.kind != "linear")
            config = {**config, "model_type": "kimi_k3"}
        elif dsa_layers:
            first_attention = dsa_layers[0]
        else:
            first_attention = attention_layers[0]
            expected_config = self._attention_config(first_attention)
            for layer in attention_layers[1:]:
                if self._attention_config(layer) != expected_config:
                    raise ValueError("Attention backend requires identical attention configuration across all layers")

        first_parameter = next(model.parameters())
        device = first_parameter.device
        self._num_attention_layers = len(attention_layers)
        num_decoding_tokens = max(
            int(num_decoding_tokens),
            int(config.get("num_speculative_tokens", 0)) + 1,
        )
        # Resolve the graph backend before the attention backend so the V4.1
        # CSA2 backend can validate its ACL-graph capacity at startup (C of
        # 51361671) instead of at the first captured forward.
        graph_backend = _resolve_graph_backend(config, device)
        graph_backend, prefill_compile_backend = _split_prefill_compile_backend(graph_backend)
        self.attention_backend = _create_attention_backend(
            first_attention,
            device,
            first_parameter.dtype,
            config,
            max_seqs_per_batch,
            num_decoding_tokens=num_decoding_tokens,
            acl_graph_enabled=graph_backend == "aclgraph",
            attention_kinds={spec.kind for spec in layer_specs},
        )

        execution_model = model.model
        # The execution model (e.g. KimiK3TextModel) owns the KDA runtime when
        # the model has linear-attention (KDA) layers; other models leave it
        # unset and the KDA bind/metadata calls below become no-ops.
        self._execution_model = execution_model
        self.eager_runner = EagerRunner(execution_model, self.attention_backend, device)
        # Context-Parallel: shard prefill sequences across the CP group. Decode
        # stays on the non-CP path (CP is prefill-only, eager-only in v1).
        self.eager_runner.cp_size = int(config.get("cp_size", 1))
        self.eager_runner.cp_rank = int(config.get("cp_rank", 0))
        self.layerwise_split_size = int(config.get("layerwise_split_size", 1))
        self.layerwise_split_rank = int(config.get("layerwise_split_rank", 0))
        if self.layerwise_split_size > 1 and config.get("model_type") != "glm_moe_dsa":
            raise NotImplementedError("Python layerwise split is supported only for GLM5.2")
        self.decode_graph_runner = None
        self.inductor_runner = None

        model_type = str(config.get("model_type", ""))
        execution_metadata_builders = tuple(
            builder
            for builder_class in get_execution_metadata_builder_classes(model_type)
            if (builder := builder_class.from_config(config)) is not None
        )
        self._execution_metadata_builders = execution_metadata_builders
        self.eager_runner.bind_execution_metadata_builders(execution_metadata_builders)

        logger.info(
            f"Python model graph backend: backend={graph_backend}, "
            f"prefill_compile_backend={prefill_compile_backend}, "
            f"enable_graph={bool(config.get('enable_graph', False))}, "
            f"device={device}"
        )
        if self.layerwise_split_size > 1 and graph_backend not in _DISABLED_GRAPH_BACKENDS:
            raise NotImplementedError(
                "Python GLM5.2 layerwise split requires eager execution; "
                f"graph backend '{graph_backend}' is not supported."
            )
        dp_size = int(config.get("dp_size", 1))
        dp_rank = int(config.get("dp_rank", 0))
        self.dp_size = dp_size
        if dp_size > 1 and graph_backend not in (*_DISABLED_GRAPH_BACKENDS, "cudagraphs", "aclgraph"):
            raise NotImplementedError("Python data parallel graph execution supports cudagraphs and aclgraph only")
        if graph_backend in _DISABLED_GRAPH_BACKENDS:
            pass
        elif graph_backend == "cudagraphs":
            from xllm.python.model_executor.runners.decode_cuda_graph import (
                DecodeCudaGraphRunner,
            )

            self.decode_graph_runner = DecodeCudaGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_seqs_per_batch,
                int(config["max_position_embeddings"]),
                dp_size,
                dp_rank,
            )
        elif graph_backend == "aclgraph":
            if self._is_kimi_k3:
                unsupported_reason = _acl_graph_unsupported_reason(attention_layers, supports_kimi_k3_graph=True)
                if unsupported_reason is not None:
                    raise ValueError(unsupported_reason)
                from xllm.python.model_executor.runners.kimi_decode_acl_graph import KimiDecodeAclGraphRunner

                if num_decoding_tokens != 1 or int(config.get("num_speculative_tokens", 0)) != 0:
                    raise NotImplementedError("Kimi K3 decode graphs do not support speculative decoding")
                self.decode_graph_runner = KimiDecodeAclGraphRunner(
                    execution_model,
                    self.attention_backend,
                    device,
                    max_seqs_per_batch,
                    int(config["max_position_embeddings"]),
                    int(config.get("block_size", 128)),
                )
            else:
                from xllm.python.model_executor.runners.decode_acl_graph import (
                    DecodeAclGraphRunner,
                )

                # EPLv2 uses MC2 only while the per-TP-rank dispatch fits the HCCL
                # window. Larger batches must take the eager All-to-AllV path
                # before graph capture, including expanded DFlash2 verify rows.
                eplv2_policy = getattr(execution_model, "_comm_policy", None)
                eplv2_graph_kwargs: dict[str, int] = {}
                if eplv2_policy is not None and int(config.get("expert_parallel_degree", 0)) == 2:
                    tp_size = int(config.get("tp_size", 1))
                    capacity = 0 if eplv2_policy.mode == "alltoall" else eplv2_policy.mc2_capacity * tp_size
                    eplv2_graph_kwargs["eplv2_graph_token_limit"] = capacity
                self.decode_graph_runner = DecodeAclGraphRunner(
                    execution_model,
                    self.attention_backend,
                    device,
                    max_seqs_per_batch,
                    int(config["max_position_embeddings"]),
                    dp_size,
                    dp_rank,
                    acl_graph_decode_batch_size_limit,
                    # MTP spec-verify packs (num_speculative_tokens+1) token rows
                    # per logical sequence; the runner sizes its static paging
                    # buffer in token rows, so pass the row multiplier so capacity
                    # matches the captured expanded-verify graph (item 二). Take
                    # the max of the explicit ctor param and the config-derived
                    # width so a caller that passes num_decoding_tokens still wins
                    # (single source of truth, no param/config divergence).
                    num_decoding_tokens=num_decoding_tokens,
                    enable_mega_moe_token_mask=bool(config.get("enable_mega_moe", False)),
                    is_spec_draft=is_spec_draft,
                    draft_query_width=int(config.get("draft_query_width", 1) or 1),
                    **eplv2_graph_kwargs,
                )
                self.decode_graph_runner.bind_execution_metadata_builders(execution_metadata_builders)
            if prefill_compile_backend is not None:
                from xllm.python.model_executor.runners.inductor import (
                    InductorRunner,
                )

                self.inductor_runner = InductorRunner(
                    execution_model,
                    self.attention_backend,
                    device,
                    prefill_compile_backend,
                )
            else:
                if self.layerwise_split_size > 1:
                    raise NotImplementedError(
                        "Python layerwise split requires eager execution; graph "
                        f"backend '{graph_backend}' is not supported."
                    )
                if self.eager_runner.cp_size > 1:
                    # CP is prefill-only and lives on eager_runner; a compile
                    # backend serves prefill through InductorRunner, which carries
                    # no cp_context, so CP would silently no-op. Reject rather than
                    # run without the requested sharding.
                    raise NotImplementedError(
                        "Context-Parallel (cp_size > 1) is not supported with the "
                        f"'{graph_backend}' graph backend; CP is eager-only. Use "
                        "graph_backend=off/aclgraph, or set cp_size=1."
                    )
        else:
            from xllm.python.model_executor.runners.inductor import InductorRunner

            self.inductor_runner = InductorRunner(execution_model, self.attention_backend, device, graph_backend)
            self.inductor_runner.bind_execution_metadata_builders(execution_metadata_builders)

    @staticmethod
    def _attention_config(
        layer: AttentionRuntimeLayer,
    ) -> tuple[int, int, int, float, int]:
        return (
            layer.num_heads,
            layer.num_kv_heads,
            layer.head_dim,
            layer.scale,
            layer.sliding_window,
        )

    def bind_kv_caches(self, kv_caches: list[LayerCacheInput]) -> None:
        layer_caches = normalize_layer_caches(kv_caches)
        highest_layer_id = max(spec.layer_id for spec in self._attention_layer_specs)
        if len(layer_caches) <= highest_layer_id:
            raise ValueError("KV cache list does not cover runtime physical-layer KV caches")
        if self._kv_bound:
            return
        if self._requires_framework_kpool_tail:
            indexed_caches = [cache for cache in layer_caches if cache.index is not None]
            if not indexed_caches or any(cache.kpool_tail is None for cache in indexed_caches):
                raise ValueError("model requires a framework-managed kPool tail for every indexer layer")
        for builder in self._execution_metadata_builders:
            if isinstance(builder, LayerCacheAwareExecutionMetadataBuilder):
                builder.bind_layer_caches(layer_caches)
        self.attention_backend.bind_kv_caches(layer_caches)
        self.eager_runner.bind_layer_caches(layer_caches)
        if self.decode_graph_runner is not None:
            self.decode_graph_runner.bind_layer_caches(layer_caches)
        if self.inductor_runner is not None:
            self.inductor_runner.bind_layer_caches(layer_caches)
        self._kv_bound = True

    def bind_kda_caches(self, kda_caches: list[tuple[int, torch.Tensor, torch.Tensor]]) -> None:
        """Bind linear-attention (KDA) conv/recurrent caches by decoder layer id.

        Each entry is ``(layer_id, conv_state, recurrent_state)``. No-op for
        models without a KDA runtime (i.e. non-Kimi-K3 models).
        """
        kda_runtime = getattr(self._execution_model, "kda_runtime", None)
        if kda_runtime is None:
            return
        for layer_id, conv_state, recurrent_state in kda_caches:
            kda_runtime.caches[int(layer_id)] = (conv_state, recurrent_state)

    def set_kda_metadata(self, view: object, num_tokens: int) -> None:
        """Populate the KDA runtime metadata for the current step.

        ``view`` exposes the per-step linear-attention scheduling info the C++
        runtime already computes for GDN layers. No-op for models without a KDA
        runtime.
        """
        kda_runtime = getattr(self._execution_model, "kda_runtime", None)
        if kda_runtime is None:
            return
        from xllm.python.layers.kda import (
            PAD_SLOT_ID,
            KimiK3KDAMetadata,
            build_kda_prefill_host_metadata,
        )

        prefill_cu_seqlens_host, prefill_chunk_indices_host = (None, None)
        if view.num_prefill_seqs > 0:
            prefill_cu_seqlens_host, prefill_chunk_indices_host = build_kda_prefill_host_metadata(
                view.query_start_loc,
                view.num_decode_seqs,
            )

        # Empty DP shard: the C++ worker feeds fake token rows with zero real
        # sequences. Preserve the tensor shape, but describe zero-length
        # sequences so KDA projections run while stateful operators see no
        # actual tokens and do not touch conv/recurrent caches.
        if view.num_decode_seqs + view.num_prefill_seqs == 0 and num_tokens > 0:
            kda_runtime.metadata = KimiK3KDAMetadata(
                query_start_loc=torch.zeros(1, dtype=torch.int64, device=self._device),
                state_indices=torch.full(
                    (num_tokens,),
                    PAD_SLOT_ID,
                    dtype=torch.int64,
                    device=self._device,
                ),
                num_decode_seqs=0,
                num_prefill_seqs=0,
                has_initial_state=None,
                graph_num_tokens=int(getattr(view, "graph_num_tokens", num_tokens)),
                empty_shard=True,
            )
            return

        # The C++ view builds query_start_loc / has_initial_state as host
        # tensors (state_indices is already moved to device). KDA feeds all of
        # them to the AscendC operators, which require device tensors, so move
        # the host ones here to honor the on-device contract of
        # KimiK3KDAMetadata.
        def _to_device(
            tensor: torch.Tensor | None,
            dtype: torch.dtype | None = None,
        ) -> torch.Tensor | None:
            if tensor is None or tensor.device == self._device:
                return tensor if dtype is None else tensor.to(dtype=dtype)
            return tensor.to(self._device, dtype=dtype, non_blocking=True)

        kda_runtime.metadata = KimiK3KDAMetadata(
            query_start_loc=_to_device(view.query_start_loc, torch.int64),
            state_indices=_to_device(view.state_indices),
            num_decode_seqs=view.num_decode_seqs,
            num_prefill_seqs=view.num_prefill_seqs,
            has_initial_state=_to_device(view.has_initial_state),
            graph_num_tokens=int(getattr(view, "graph_num_tokens", num_tokens)),
            empty_shard=bool(getattr(view, "empty_shard", False)),
            prefill_cu_seqlens_host=prefill_cu_seqlens_host,
            prefill_chunk_indices_host=prefill_chunk_indices_host,
        )

    @torch.inference_mode()
    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        expert_load_data: torch.Tensor | None = None,
        eplb_decode_token_mask: torch.Tensor | None = None,
        is_graph_warmup: bool = False,
        input_batch_metadata: InputBatchMetadata | None = None,
        layer_load_synchronizer: LayerLoadSynchronizer | None = None,
        layers_per_event: int = 1,
        draft_load_event_index: int | None = None,
        kda_metadata: object | None = None,
    ) -> ModelExecutionOutput:
        if not self._kv_bound:
            raise RuntimeError("KV caches are not bound")
        if self.layerwise_split_size > 1 and (metadata.is_prefill or metadata.is_chunked_prefill):
            raise NotImplementedError("Python GLM5.2 layerwise split is decode-only")

        eplb = None
        if expert_load_data is not None:
            eplb = EplbRuntimeState(
                expert_load_data=expert_load_data,
                decode_token_mask=eplb_decode_token_mask,
                is_graph_warmup=is_graph_warmup,
            )

        input_batch = None
        if self._execution_metadata_builders:
            if input_batch_metadata is None:
                raise RuntimeError("execution metadata builders require upstream InputBatch metadata")
            input_batch = InputBatch.from_runtime(
                input_ids,
                positions,
                input_batch_metadata,
                is_dummy=bool(getattr(metadata, "is_dummy", False)),
            )

        layer_load_context = (
            LayerLoadContext(layer_load_synchronizer, layers_per_event) if layer_load_synchronizer is not None else None
        )
        if self._is_spec_draft and layer_load_context is not None:
            if draft_load_event_index is None:
                layer_load_context.synchronize_all_layers()
            else:
                layer_load_context.synchronize_event(draft_load_event_index)
            layer_load_context = None
        if kda_metadata is not None:
            self.set_kda_metadata(kda_metadata, input_ids.shape[0])
        graph_runner = self.decode_graph_runner
        if graph_runner is not None and self._is_kimi_k3:
            if layer_load_context is not None:
                layer_load_context.synchronize_all_layers()
            # Capture before prefill populates request caches.
            graph_runner.warmup(input_ids.device, input_ids.dtype, None)
            if graph_runner.can_execute(input_ids, metadata) and self._all_dp_decode(kda_metadata):
                return graph_runner.execute(input_ids, positions, metadata, input_embedding)
            return self.eager_runner.execute(
                input_ids,
                positions,
                metadata,
                input_embedding,
                layer_synchronizer=layer_synchronizer,
                eplb=eplb,
                input_batch=input_batch,
                layer_load_context=layer_load_context,
                kda_metadata=kda_metadata,
            )
        if (
            graph_runner is not None
            and (eplb is None or current_platform.is_npu())
            and graph_runner.can_execute(input_ids, metadata, input_embedding)
        ):
            # Replay bypasses Python attention calls. Also protect first-capture
            # state snapshots and warmup before entering the graph runner.
            if layer_load_context is not None:
                layer_load_context.synchronize_all_layers()
            graph_runner.warmup(
                input_ids,
                positions,
                metadata,
                input_embedding,
            )
            return graph_runner.execute(
                input_ids,
                positions,
                metadata,
                input_embedding,
                eplb=eplb,
                input_batch=input_batch,
            )
        if self.inductor_runner is not None:
            # Compiled execution must not capture request-specific Host events.
            if layer_load_context is not None:
                layer_load_context.synchronize_all_layers()
            return self.inductor_runner.execute(
                input_ids,
                positions,
                metadata,
                input_embedding,
                layer_synchronizer=layer_synchronizer,
                eplb=eplb,
                input_batch=input_batch,
            )
        return self.eager_runner.execute(
            input_ids,
            positions,
            metadata,
            input_embedding,
            layer_synchronizer=layer_synchronizer,
            eplb=eplb,
            input_batch=input_batch,
            layer_load_context=layer_load_context,
        )

    def _all_dp_decode(self, kda_metadata: object | None) -> bool:
        """Whether every DP rank can safely enter decode graph this step.

        The C++ engine host-syncs per-rank batch types (``dp_is_decode``), so
        this is a plain host-side check. DP graph execution requires complete
        metadata and every rank to be in decode; otherwise all ranks fall back
        to eager. Empty decode shards are marked as decode by the engine and
        participate with padded inputs.
        """
        if self._dp_size <= 1:
            return True
        if kda_metadata is None:
            return False
        return bool(getattr(kda_metadata, "dp_metadata_valid", False)) and bool(
            getattr(kda_metadata, "all_dp_decode", False)
        )
