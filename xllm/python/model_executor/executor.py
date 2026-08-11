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
import torch.nn as nn

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
    if graph_disabled and config.get("enable_graph", False):
        if _is_npu_device(device):
            return "aclgraph"
    return graph_backend


def _create_attention_backend(
    first_attention: AttentionLayerSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> AttentionBackend:
    if _is_npu_device(device):
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
    raise NotImplementedError(
        f"No attention backend available for device type '{device.type}'"
    )


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
            (
                module
                for module in model.modules()
                if isinstance(module, AttentionRuntimeLayer)
            ),
            key=lambda layer: layer.layer_id,
        )
        if not attention_layers:
            raise ValueError(
                "Python model does not contain a runtime attention layer"
            )

        layer_specs = [layer.attention_layer_spec() for layer in attention_layers]
        layer_ids = [spec.layer_id for spec in layer_specs]
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError(
                "Runtime attention layer ids must be unique: "
                f"got {layer_ids}"
            )

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
            backend_spec, device, first_parameter.dtype
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
            from xllm.python.model_executor.runners.decode_acl_graph import (
                DecodeAclGraphRunner,
            )
            self.decode_graph_runner = DecodeAclGraphRunner(
                execution_model,
                self.attention_backend,
                device,
                max_seqs_per_batch,
                int(config["max_position_embeddings"]),
            )
        else:
            from xllm.python.model_executor.runners.inductor import InductorRunner
            self.inductor_runner = InductorRunner(
                execution_model, self.attention_backend, device, graph_backend
            )

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
        highest_layer_id = max(
            spec.layer_id for spec in self._attention_layer_specs
        )
        if len(kv_caches) <= highest_layer_id:
            raise ValueError(
                "KV cache list does not cover runtime physical-layer KV caches: "
                f"highest layer id is {highest_layer_id}, got {len(kv_caches)} caches"
            )
        if self._kv_bound:
            return
        self.attention_backend.bind_kv_caches(kv_caches)
        self._kv_bound = True

    def bind_kda_caches(
        self, kda_caches: list[tuple[int, torch.Tensor, torch.Tensor]]
    ) -> None:
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

        # Empty DP shard: the C++ worker feeds one fake token with zero real
        # sequences. Synthesize one dummy decode sequence per fake token on the
        # pad slot so KDA layers run shape-consistent kernels without touching
        # any real conv/recurrent state slot.
        if view.num_decode_seqs + view.num_prefill_seqs == 0 and num_tokens > 0:
            kda_runtime.metadata = KimiK3KDAMetadata(
                query_start_loc=torch.arange(
                    num_tokens + 1, dtype=torch.int32, device=self._device
                ),
                state_indices=torch.full(
                    (num_tokens,),
                    PAD_SLOT_ID,
                    dtype=torch.int64,
                    device=self._device,
                ),
                num_decode_seqs=num_tokens,
                num_prefill_seqs=0,
                has_initial_state=None,
            )
            return

        # The C++ view builds query_start_loc / has_initial_state as host
        # tensors (state_indices is already moved to device). KDA feeds all of
        # them to the AscendC operators, which require device tensors, so move
        # the host ones here to honor the on-device contract of
        # KimiK3KDAMetadata.
        def _to_device(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None or tensor.device == self._device:
                return tensor
            return tensor.to(self._device, non_blocking=True)

        kda_runtime.metadata = KimiK3KDAMetadata(
            query_start_loc=_to_device(view.query_start_loc),
            state_indices=_to_device(view.state_indices),
            num_decode_seqs=view.num_decode_seqs,
            num_prefill_seqs=view.num_prefill_seqs,
            has_initial_state=_to_device(view.has_initial_state),
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

        # Multimodal prefill supplies already-merged embeddings from the C++
        # VLM data path. Graph runners only accept token ids, so execute this
        # path eagerly until they grow an inputs_embeds input contract.
        if inputs_embeds is not None:
            return self.eager_runner.execute(
                input_ids, positions, metadata, inputs_embeds
            )

        graph_runner = self.decode_graph_runner
        if graph_runner is not None:
            graph_runner.warmup(input_ids.device, input_ids.dtype)
            if graph_runner.can_execute(input_ids, metadata) and self._all_dp_decode(
                kda_metadata
            ):
                return graph_runner.execute(input_ids, positions, metadata)
        if self.inductor_runner is not None:
            return self.inductor_runner.execute(input_ids, positions, metadata)
        return self.eager_runner.execute(input_ids, positions, metadata)

    def _all_dp_decode(self, kda_metadata: object | None) -> bool:
        """Whether every DP rank runs a decode batch this step.

        The C++ engine host-syncs per-rank batch types (``dp_is_decode``), so
        this is a plain host-side check. Decode graphs are only entered when
        all DP ranks decode, mirroring the C++ ACL graph executor gating.
        """
        if self._dp_size <= 1 or kda_metadata is None:
            return True
        return bool(getattr(kda_metadata, "all_dp_decode", True))
