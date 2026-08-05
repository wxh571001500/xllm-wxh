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
        expected_layer_ids = list(range(len(layer_specs)))
        if layer_ids != expected_layer_ids:
            raise ValueError(
                "Runtime attention layer ids must be unique and contiguous from zero: "
                f"got {layer_ids}"
            )

        first_parameter = next(model.parameters())
        device = first_parameter.device
        self._attention_layer_specs = layer_specs
        self._num_attention_layers = len(layer_specs)
        self.attention_backend = _create_attention_backend(
            layer_specs[0], device, first_parameter.dtype
        )

        execution_model = model.model
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
        if len(kv_caches) != self._num_attention_layers:
            raise ValueError(
                "KV cache layer count does not match model attention layer count"
            )
        if self._kv_bound:
            return
        self.attention_backend.bind_kv_caches(kv_caches)
        self._kv_bound = True

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._kv_bound:
            raise RuntimeError("KV caches are not bound")

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
            if graph_runner.can_execute(input_ids, metadata):
                return graph_runner.execute(input_ids, positions, metadata)
        if self.inductor_runner is not None:
            return self.inductor_runner.execute(input_ids, positions, metadata)
        return self.eager_runner.execute(input_ids, positions, metadata)
