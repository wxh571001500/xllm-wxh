# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
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
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata, KVCache
from xllm.python.layers.attention import (
    AttentionLayerSpec,
    AttentionRuntimeLayer,
)
from xllm.python.model_executor.runners.eager import EagerRunner


def _is_npu_device(device: torch.device) -> bool:
    return device.type in ("npu", "privateuseone")


def _resolve_graph_backend(config: dict, device: torch.device) -> str:
    graph_backend = str(config.get("python_graph_backend", "off")).lower()
    graph_disabled = graph_backend in ("", "off", "none", "0")
    if graph_disabled and config.get("enable_graph", False) and _is_npu_device(device):
        return "aclgraph"
    return graph_backend


def _create_attention_backend(
    first_attention: AttentionLayerSpec,
    device: torch.device,
    dtype: torch.dtype,
    attention_kinds: set[str] | dict | None = None,
    max_num_reqs: int = 1,
) -> AttentionBackend:
    config = attention_kinds if isinstance(attention_kinds, dict) else {}
    if config.get("model_type") == "deepseek_v4" and _is_npu_device(device):
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
        )
    if _is_npu_device(device):
        dcp_group = distributed.dcp_group(device)
        if dcp_group is not None and dcp_group.size() > 1:
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
            )
        from xllm.python.attention.npu_paged_attention import (
            NpuPagedAttentionBackend,
        )

        return NpuPagedAttentionBackend(
            num_heads=first_attention.num_heads,
            num_kv_heads=first_attention.num_kv_heads,
            head_dim=first_attention.head_dim,
            scale=first_attention.scale,
            sliding_window=first_attention.sliding_window,
            device=device,
            dtype=dtype,
            has_mha_layers=(first_attention.kind == "mha" if attention_kinds is None else "mha" in attention_kinds),
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
    ) -> None:
        self.model = model
        self._kv_bound = False

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
        # The paged-KV backend geometry (heads/dims) must come from a real
        # paged-attention layer. KDA ("linear") layers are runtime layers for
        # ordering only and carry no paged-KV geometry, so build the backend
        # from the first non-linear spec.
        backend_spec = next(
            (spec for spec in layer_specs if spec.kind != "linear"),
            layer_specs[0],
        )
        self.attention_backend = _create_attention_backend(
            backend_spec,
            device,
            first_parameter.dtype,
            config,
            max_seqs_per_batch,
        )

        execution_model = model.model
        # The execution model (e.g. KimiK3TextModel) owns the KDA runtime when
        # the model has linear-attention (KDA) layers; other models leave it
        # unset and the KDA bind/metadata calls below become no-ops.
        self._execution_model = execution_model
        self.eager_runner = EagerRunner(execution_model, self.attention_backend, device)
        self.decode_graph_runner = None
        self.inductor_runner = None

        graph_backend = _resolve_graph_backend(config, device)
        logger.info(
            f"Python model graph backend: backend={graph_backend}, "
            f"enable_graph={bool(config.get('enable_graph', False))}, "
            f"device={device}"
        )
        if graph_backend in ("", "off", "none", "0"):
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
            )
        elif graph_backend == "aclgraph":
            unsupported_reason = _acl_graph_unsupported_reason(
                attention_layers,
                supports_kimi_k3_graph=hasattr(execution_model, "kda_runtime"),
            )
            if unsupported_reason is not None:
                raise ValueError(unsupported_reason)
            from xllm.python.model_executor.runners.decode_acl_graph import (
                DecodeAclGraphRunner,
            )

            self.decode_graph_runner = DecodeAclGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_seqs_per_batch,
                int(config["max_position_embeddings"]),
                int(config.get("block_size", 128)),
            )
        else:
            from xllm.python.model_executor.runners.inductor import InductorRunner

            self.inductor_runner = InductorRunner(execution_model, self.attention_backend, device, graph_backend)

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

    def bind_kv_caches(self, kv_caches: list[KVCache]) -> None:
        # Runtime layers are indexed by physical decoder layer id. The cache
        # list may therefore contain slots for layers not present in a truncated
        # runtime model, including KDA conv/SSM slots that the paged-KV backend
        # never indexes directly.
        highest_layer_id = max(spec.layer_id for spec in self._attention_layer_specs)
        if len(kv_caches) <= highest_layer_id:
            raise ValueError(
                "KV cache list does not cover runtime physical-layer KV caches: "
                f"highest layer id is {highest_layer_id}, got {len(kv_caches)} caches"
            )
        if self._kv_bound:
            return
        self.attention_backend.bind_kv_caches(kv_caches)
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
        from xllm.python.layers.kda import PAD_SLOT_ID, KimiK3KDAMetadata

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
        )

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        inputs_embeds: torch.Tensor | None = None,
        kda_metadata: object | None = None,
    ) -> torch.Tensor:
        if not self._kv_bound:
            raise RuntimeError("KV caches are not bound")

        # Push per-step KDA scheduling info into the runtime before forward so
        # KDA layers can read it via kda_runtime.require(). No-op without KDA.
        if kda_metadata is not None:
            self.set_kda_metadata(kda_metadata, input_ids.shape[0])

        graph_runner = self.decode_graph_runner
        if graph_runner is not None:
            # Capture before eager prefill populates live KV/KDA caches. The
            # synthetic graph warmup writes its own cache slots, so deferring it
            # until the first decode would overwrite the active request state.
            graph_runner.warmup(
                input_ids.device,
                input_ids.dtype,
                inputs_embeds,
            )
            if graph_runner.can_execute(input_ids, metadata) and self._all_dp_decode(kda_metadata):
                return graph_runner.execute(
                    input_ids,
                    positions,
                    metadata,
                    inputs_embeds,
                )
        if inputs_embeds is not None:
            return self.eager_runner.execute(input_ids, positions, metadata, inputs_embeds)
        if self.inductor_runner is not None:
            return self.inductor_runner.execute(input_ids, positions, metadata)
        return self.eager_runner.execute(input_ids, positions, metadata)

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
