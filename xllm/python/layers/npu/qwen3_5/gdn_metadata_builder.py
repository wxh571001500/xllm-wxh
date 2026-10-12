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

"""Execution metadata builder for the Qwen3.5 NPU GDN path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from xllm.python.attention.backend import LayerCache, resolve_linear_state_io_indices
from xllm.python.layers.npu.qwen3_5.gdn_metadata import (
    MEGA_GDN_MTP_MAX_VERIFY_TOKENS,
    MEGA_GDN_MTP_MIN_VERIFY_TOKENS,
    GdnDecodeMetadata,
    GdnMetadata,
    GdnPrefillMetadata,
    GdnSpecVerifyMetadata,
    GdnStateCache,
)
from xllm.python.model_executor.input_batch import InputBatch

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata


_MEGA_GDN_CHUNK_SIZE = 128
_MEGA_GDN_CONV_KERNEL_SIZE = 4
_MEGA_GDN_HEAD_DIM = 128
_SUPPORTED_MEGA_GDN_PREFILL_HEAD_COUNTS = frozenset((1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64))


def _validate_mega_gdn_config(
    config: Mapping[str, object],
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> None:
    """Reject configurations unsupported by the fused MegaGdn kernels."""
    if int(config.get("linear_conv_kernel_dim", _MEGA_GDN_CONV_KERNEL_SIZE)) != _MEGA_GDN_CONV_KERNEL_SIZE:
        raise NotImplementedError("Qwen3.5 MegaGdn requires convolution width 4")
    if key_head_dim != _MEGA_GDN_HEAD_DIM or value_head_dim != _MEGA_GDN_HEAD_DIM:
        raise NotImplementedError("Qwen3.5 MegaGdn requires K/V head dimension 128")
    if num_value_heads not in _SUPPORTED_MEGA_GDN_PREFILL_HEAD_COUNTS:
        raise NotImplementedError(f"Qwen3.5 MegaGdnPrefill does not support {num_value_heads} local value heads")
    if num_key_heads <= 0 or num_key_heads > 16:
        raise NotImplementedError("Qwen3.5 MegaGdnDecode supports 1 to 16 local key heads")
    if num_key_heads & (num_key_heads - 1):
        raise NotImplementedError("Qwen3.5 MegaGdnDecode requires a power-of-two local key-head count")
    if num_value_heads % num_key_heads != 0:
        raise NotImplementedError("Qwen3.5 MegaGdn requires Nv divisible by Nk")
    if num_value_heads // num_key_heads > 4:
        raise NotImplementedError("Qwen3.5 MegaGdnDecode supports at most four value heads per key head")

    dtype = config.get("dtype") or config.get("torch_dtype") or "bfloat16"
    if dtype != torch.bfloat16 and str(dtype).removeprefix("torch.") != "bfloat16":
        raise NotImplementedError("Qwen3.5 MegaGdn supports BF16 model weights only")


def _build_mega_prefill_indices(
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    checkpoint_stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the independent Conv and SSM read/write indices for MegaGdn."""
    if checkpoint_stride <= 0:
        raise ValueError("Qwen3.5 SSM checkpoint stride must be positive")
    if (
        read_state_indices.dim() != 1
        or write_state_indices.shape != read_state_indices.shape
        or has_initial_state.shape != read_state_indices.shape
    ):
        raise ValueError("Qwen3.5 state indices and validity must be sequence-scoped")

    conv_read_source = read_state_indices.to(dtype=torch.int32).contiguous()
    conv_write = write_state_indices.to(dtype=torch.int32).contiguous()
    valid_read = has_initial_state.to(dtype=torch.bool) & (conv_read_source > 0)
    invalid = torch.full_like(conv_read_source, -1)
    conv_read = torch.where(valid_read, conv_read_source, invalid).contiguous()
    ssm_write = (conv_write * checkpoint_stride).contiguous()
    ssm_read_source = conv_read_source * checkpoint_stride
    ssm_read = torch.where(valid_read, ssm_read_source, invalid).contiguous()
    return conv_read, conv_write, ssm_read, ssm_write


def _compute_mega_prefill_num_matrices(
    query_lengths: list[int],
    num_value_heads: int,
) -> int:
    """Return the number of 128-token/head matrices consumed by MegaGdn."""
    if not query_lengths or any(length <= 0 for length in query_lengths):
        raise ValueError("Qwen3.5 prefill query lengths must be positive")
    return (
        sum((length + _MEGA_GDN_CHUNK_SIZE - 1) // _MEGA_GDN_CHUNK_SIZE for length in query_lengths) * num_value_heads
    )


class Qwen3_5GdnMetadataBuilder:
    """Build eager GDN metadata and maintain its persistent Graph form."""

    metadata_type = GdnMetadata

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> Qwen3_5GdnMetadataBuilder | None:
        device = torch.device(str(config.get("device", "cuda")))
        if device.type not in ("npu", "privateuseone"):
            return None

        num_layers = int(config.get("n_layers", config.get("num_hidden_layers", 0)))
        layer_types_value = config.get("layer_types")
        if layer_types_value is not None:
            if not isinstance(layer_types_value, Sequence) or isinstance(layer_types_value, (str, bytes)):
                raise TypeError("Qwen3.5 layer_types must be a sequence")
            layer_types = tuple(str(layer_type) for layer_type in layer_types_value)
        else:
            layer_types = ()
        if not layer_types:
            full_attention_interval = int(config.get("full_attention_interval", 4))
            if full_attention_interval <= 0:
                raise ValueError("Qwen3.5 full-attention interval must be positive")
            layer_types = tuple(
                "full_attention" if (layer_id + 1) % full_attention_interval == 0 else "linear_attention"
                for layer_id in range(num_layers)
            )
        if len(layer_types) != num_layers:
            raise ValueError("Qwen3.5 layer_types must contain one entry per layer")
        layer_ids = tuple(
            layer_id for layer_id, layer_type in enumerate(layer_types) if layer_type == "linear_attention"
        )
        if not layer_ids:
            return None
        if int(config.get("cp_size", 1)) != 1:
            raise NotImplementedError("Qwen3.5 NPU GDN requires cp_size=1")
        if int(config.get("num_speculative_tokens", 0)) != 0:
            raise NotImplementedError("Qwen3.5 NPU GDN does not support speculative decoding")

        tp_size = int(config.get("tp_size", 1))
        if tp_size <= 0:
            raise ValueError("Qwen3.5 TP size must be positive")
        global_num_key_heads = int(config.get("linear_num_key_heads", 16))
        global_num_value_heads = int(config.get("linear_num_value_heads", 32))
        if global_num_key_heads % tp_size or global_num_value_heads % tp_size:
            raise ValueError("Qwen3.5 linear-attention heads must be divisible by TP size")
        num_key_heads = global_num_key_heads // tp_size
        num_value_heads = global_num_value_heads // tp_size
        key_head_dim = int(config.get("linear_key_head_dim", 128))
        value_head_dim = int(config.get("linear_value_head_dim", 128))
        _validate_mega_gdn_config(
            config,
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
        )
        conv_dim = 2 * num_key_heads * key_head_dim + num_value_heads * value_head_dim
        return cls(
            layer_ids=layer_ids,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            conv_dim=conv_dim,
        )

    def __init__(
        self,
        layer_ids: tuple[int, ...],
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_dim: int,
    ) -> None:
        if not layer_ids or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("Qwen3.5 GDN layer ids must be non-empty and unique")
        if any(layer_id < 0 for layer_id in layer_ids):
            raise ValueError("Qwen3.5 GDN layer ids must be non-negative")
        if min(num_value_heads, key_head_dim, value_head_dim, conv_dim) <= 0:
            raise ValueError("Qwen3.5 GDN geometry must be positive")
        self._layer_ids = layer_ids
        self._num_value_heads = num_value_heads
        self._key_head_dim = key_head_dim
        self._value_head_dim = value_head_dim
        self._conv_dim = conv_dim
        self._checkpoint_stride: int | None = None
        self._state_caches: dict[int, GdnStateCache] | None = None
        self._cache_device: torch.device | None = None

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        """Bind and validate the long-lived state cache for each GDN layer."""
        checkpoint_stride: int | None = None
        cache_device: torch.device | None = None
        state_caches: dict[int, GdnStateCache] = {}
        for layer_id in self._layer_ids:
            if layer_id >= len(layer_caches):
                raise ValueError(f"GDN cache is missing for layer {layer_id}")
            cache = layer_caches[layer_id]
            conv_state = cache.conv
            ssm_state = cache.ssm
            if conv_state is None or ssm_state is None:
                raise ValueError(f"GDN state cache is missing for layer {layer_id}")
            if conv_state.dim() != 3 or conv_state.shape[0] <= 0 or conv_state.shape[2] != self._conv_dim:
                raise ValueError(f"GDN Conv cache has invalid geometry for layer {layer_id}")
            if (
                ssm_state.dim() != 4
                or ssm_state.shape[0] <= 0
                or ssm_state.shape[0] % conv_state.shape[0] != 0
                or ssm_state.shape[1:]
                != (
                    self._num_value_heads,
                    self._key_head_dim,
                    self._value_head_dim,
                )
            ):
                raise ValueError(f"GDN SSM cache has invalid geometry for layer {layer_id}")
            if conv_state.dtype != torch.bfloat16 or ssm_state.dtype != torch.float32:
                raise ValueError(f"GDN state cache has invalid dtype for layer {layer_id}")
            if conv_state.device != ssm_state.device:
                raise ValueError(f"GDN state caches are on different devices for layer {layer_id}")
            if cache_device is None:
                cache_device = conv_state.device
            elif cache_device != conv_state.device:
                raise ValueError("Qwen3.5 GDN layers must use one cache device")

            layer_stride = ssm_state.shape[0] // conv_state.shape[0]
            if conv_state.shape[1] != layer_stride + 2:
                raise ValueError(f"GDN Conv history has invalid geometry for layer {layer_id}")
            if checkpoint_stride is None:
                checkpoint_stride = layer_stride
            elif checkpoint_stride != layer_stride:
                raise ValueError("Qwen3.5 GDN layers must use one checkpoint stride")
            state_caches[layer_id] = GdnStateCache(
                conv_state=conv_state,
                ssm_state=ssm_state,
            )

        self._checkpoint_stride = checkpoint_stride
        self._state_caches = state_caches
        self._cache_device = cache_device

    def build(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> GdnMetadata:
        state_caches = self._require_state_caches(input_batch.input_ids.device)
        read_indices, write_indices = self._resolve_indices(
            metadata,
            input_batch.input_ids.device,
        )
        if metadata.is_spec_verify:
            num_accepted = metadata.num_accepted_tokens
            if num_accepted is None:
                raise ValueError("Qwen3.5 GDN spec verify requires num_accepted_tokens")
            return GdnSpecVerifyMetadata(
                state_caches=state_caches,
                read_state_indices=read_indices,
                write_state_indices=write_indices,
                num_accepted_tokens=num_accepted.to(device=input_batch.input_ids.device, dtype=torch.int32),
            )
        if metadata.is_prefill or metadata.is_chunked_prefill:
            return self._build_prefill_metadata(
                input_batch,
                metadata,
                read_indices,
                write_indices,
                state_caches,
            )
        if read_indices.numel() != input_batch.num_tokens:
            raise ValueError("Qwen3.5 GDN decode requires one state slot per token")
        return GdnDecodeMetadata(
            state_caches=state_caches,
            read_state_indices=read_indices,
            write_state_indices=write_indices,
        )

    def allocate_persistent(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> GdnMetadata:
        device = input_batch.input_ids.device
        state_caches = self._require_state_caches(device)
        if metadata.is_spec_verify:
            request_capacity = self._spec_verify_request_capacity(input_batch)
            width = input_batch.num_scheduled_tokens[0]
            if not MEGA_GDN_MTP_MIN_VERIFY_TOKENS <= width <= MEGA_GDN_MTP_MAX_VERIFY_TOKENS:
                raise NotImplementedError(
                    f"Qwen3.5 MegaGdnMtpDecode supports {MEGA_GDN_MTP_MIN_VERIFY_TOKENS} "
                    f"to {MEGA_GDN_MTP_MAX_VERIFY_TOKENS} verify tokens"
                )
            return GdnSpecVerifyMetadata(
                state_caches=state_caches,
                read_state_indices=torch.zeros(request_capacity, dtype=torch.int32, device=device),
                write_state_indices=torch.zeros(request_capacity, dtype=torch.int32, device=device),
                num_accepted_tokens=torch.ones(request_capacity, dtype=torch.int32, device=device),
                num_tail_padding_tokens=input_batch.num_tokens_after_padding % width,
            )
        if metadata.is_prefill or metadata.is_chunked_prefill:
            raise NotImplementedError("Qwen3.5 GDN ACL Graph supports decode only")
        token_capacity = input_batch.num_tokens_after_padding
        if token_capacity <= 0:
            raise ValueError("Qwen3.5 GDN graph token capacity must be positive")
        return GdnDecodeMetadata(
            state_caches=state_caches,
            read_state_indices=torch.zeros(
                token_capacity,
                dtype=torch.int32,
                device=device,
            ),
            write_state_indices=torch.zeros(
                token_capacity,
                dtype=torch.int32,
                device=device,
            ),
        )

    def update_persistent(
        self,
        persistent_metadata: object,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> None:
        if isinstance(persistent_metadata, GdnSpecVerifyMetadata):
            self._update_persistent_spec_verify(persistent_metadata, input_batch, metadata)
            return
        if not isinstance(persistent_metadata, GdnDecodeMetadata):
            raise TypeError("Qwen3.5 GDN received invalid persistent metadata")
        if metadata.is_prefill or metadata.is_chunked_prefill:
            raise NotImplementedError("Qwen3.5 GDN ACL Graph supports decode only")

        token_capacity = persistent_metadata.read_state_indices.numel()
        if input_batch.num_tokens_after_padding != token_capacity:
            raise RuntimeError(
                "Qwen3.5 GDN metadata capacity does not match the execution batch: "
                f"metadata={token_capacity}, batch={input_batch.num_tokens_after_padding}"
            )
        if input_batch.num_tokens <= 0 or input_batch.num_tokens > token_capacity:
            raise ValueError("Qwen3.5 GDN token count must fit the graph capacity")
        read_indices, write_indices = self._resolve_indices(
            metadata,
            persistent_metadata.read_state_indices.device,
        )
        if read_indices.numel() != input_batch.num_tokens:
            raise ValueError("Qwen3.5 GDN decode requires one state slot per token")

        persistent_metadata.read_state_indices.zero_()
        persistent_metadata.write_state_indices.zero_()
        persistent_metadata.read_state_indices[: input_batch.num_tokens].copy_(read_indices)
        persistent_metadata.write_state_indices[: input_batch.num_tokens].copy_(write_indices)

    def _update_persistent_spec_verify(
        self,
        persistent: GdnSpecVerifyMetadata,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> None:
        if not metadata.is_spec_verify:
            raise ValueError("Qwen3.5 GDN verify graph requires speculative-verify metadata")
        device = input_batch.input_ids.device
        request_capacity = self._spec_verify_request_capacity(input_batch)
        if persistent.read_state_indices.shape != (request_capacity,):
            raise ValueError("Qwen3.5 GDN verify graph capacity does not match the execution batch")
        read_indices, write_indices = self._resolve_indices(metadata, device)
        num_reqs = input_batch.num_reqs
        if read_indices.numel() != num_reqs:
            raise ValueError("Qwen3.5 GDN verify requires one state slot per request")
        accepted = metadata.num_accepted_tokens
        if accepted is None or accepted.shape != (num_reqs,):
            raise ValueError("Qwen3.5 GDN verify requires one accepted-token count per request")
        # The worker validates accepted counts before staging them to the device.
        accepted = accepted.to(device=device, dtype=torch.int32)

        persistent.read_state_indices.zero_()
        persistent.write_state_indices.zero_()
        persistent.num_accepted_tokens.fill_(1)
        persistent.read_state_indices[:num_reqs].copy_(read_indices)
        persistent.write_state_indices[:num_reqs].copy_(write_indices)
        persistent.num_accepted_tokens[:num_reqs].copy_(accepted)

    def _spec_verify_request_capacity(self, input_batch: InputBatch) -> int:
        """Validate GDN-specific grouping on an already validated InputBatch."""
        if input_batch.num_reqs <= 0:
            raise ValueError("Qwen3.5 GDN verify requires at least one request")
        token_counts = input_batch.num_scheduled_tokens
        width = token_counts[0]
        if any(count != width for count in token_counts):
            raise ValueError("Qwen3.5 GDN verify requires equal-width request groups")
        if self._checkpoint_stride != width:
            raise ValueError("Qwen3.5 GDN verify width must match the bound cache checkpoint stride")
        return input_batch.num_tokens_after_padding // width

    def _build_prefill_metadata(
        self,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
        read_indices: torch.Tensor,
        write_indices: torch.Tensor,
        state_caches: Mapping[int, GdnStateCache],
    ) -> GdnPrefillMetadata:
        checkpoint_stride = self._checkpoint_stride
        if checkpoint_stride is None:
            raise RuntimeError("Qwen3.5 GDN state caches must be bound before execution")
        if input_batch.is_dummy:
            if input_batch.num_reqs != 0 or input_batch.num_tokens != 1:
                raise ValueError("Qwen3.5 GDN requires one execution row for an empty DP shard")
            num_sequences = 1
            query_start_loc = (0, 1)
            num_scheduled_tokens = (1,)
        else:
            num_sequences = input_batch.num_reqs
            query_start_loc = input_batch.query_start_loc
            num_scheduled_tokens = input_batch.num_scheduled_tokens

        if write_indices.numel() != num_sequences:
            raise ValueError("Qwen3.5 GDN prefill requires one state slot per request")
        has_initial_state = metadata.has_initial_state
        if has_initial_state is None:
            raise RuntimeError("Qwen3.5 GDN prefill requires initial-state validity")
        if has_initial_state.numel() != num_sequences:
            raise ValueError("Qwen3.5 GDN initial-state validity must be request scoped")
        if sum(num_scheduled_tokens) != input_batch.num_tokens:
            raise ValueError("Qwen3.5 GDN packed token count does not match scheduled tokens")

        device = input_batch.input_ids.device
        conv_read, conv_write, ssm_read, ssm_write = _build_mega_prefill_indices(
            read_indices,
            write_indices,
            has_initial_state.to(device=device, dtype=torch.bool),
            checkpoint_stride,
        )
        source_cu_seqlens = metadata.q_cu_seq_lens
        if source_cu_seqlens is None:
            cu_seqlens = torch.tensor(
                query_start_loc,
                dtype=torch.int32,
                device=device,
            )
        else:
            if source_cu_seqlens.dim() != 1 or source_cu_seqlens.numel() != num_sequences + 1:
                raise ValueError("Qwen3.5 cu_seqlens must contain one boundary per request")
            cu_seqlens = source_cu_seqlens.to(
                device=device,
                dtype=torch.int32,
            ).contiguous()

        return GdnPrefillMetadata(
            state_caches=state_caches,
            conv_read_indices=conv_read,
            conv_write_indices=conv_write,
            ssm_read_indices=ssm_read,
            ssm_write_indices=ssm_write,
            cu_seqlens=cu_seqlens,
            num_matrices=_compute_mega_prefill_num_matrices(
                list(num_scheduled_tokens),
                self._num_value_heads,
            ),
        )

    def _require_state_caches(
        self,
        device: torch.device,
    ) -> Mapping[int, GdnStateCache]:
        if self._state_caches is None:
            raise RuntimeError("Qwen3.5 GDN state caches must be bound before execution")
        if self._cache_device != device:
            raise ValueError(
                f"Qwen3.5 GDN state cache device does not match the input: cache={self._cache_device}, input={device}"
            )
        return self._state_caches

    @staticmethod
    def _resolve_indices(
        metadata: AttentionMetadata,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        read_indices, write_indices = resolve_linear_state_io_indices(metadata)
        if read_indices is None or write_indices is None:
            raise RuntimeError("Qwen3.5 GDN requires read/write state indices")
        if read_indices.dim() != 1 or write_indices.shape != read_indices.shape:
            raise ValueError("Qwen3.5 GDN state indices must be one-dimensional and shape-aligned")
        return (
            read_indices.to(device=device, dtype=torch.int32).contiguous(),
            write_indices.to(device=device, dtype=torch.int32).contiguous(),
        )
