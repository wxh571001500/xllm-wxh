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

"""Dense GQA decode-context-parallel attention for Ascend NPU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from xllm.python import kernels
from xllm.python.attention.backend import AttentionMetadata, LayerCache
from xllm.python.attention.dense_dcp_execution import DenseDcpFia, get_dense_dcp_merge_buffers
from xllm.python.attention.dense_dcp_metadata import DenseDcpMetadata
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
from xllm.python.model_executor.forward_context import get_execution_context

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention

_SPARSE_MODE_RIGHT_DOWN_CAUSAL = 3


class DcpGroupCoordinator(Protocol):
    world_size: int
    rank_in_group: int
    device_group: Any


@dataclass(frozen=True, slots=True)
class TorchDcpGroupCoordinator:
    world_size: int
    rank_in_group: int
    device_group: ProcessGroup

    @classmethod
    def from_process_group(
        cls,
        group: ProcessGroup,
    ) -> TorchDcpGroupCoordinator:
        return cls(
            world_size=group.size(),
            rank_in_group=group.rank(),
            device_group=group,
        )


class DcpGatherContext(NamedTuple):
    gathered: torch.Tensor
    handle: torch.distributed.Work | None
    restore_perm: tuple[int, ...] | None
    split_sizes: tuple[int, ...]


def normalize_dcp_lse(
    softmax_lse: torch.Tensor,
    num_tokens: int,
    num_heads: int,
) -> torch.Tensor:
    """Normalize FIA LSE variants to ``[tokens, heads]``."""
    lse = softmax_lse
    if lse.ndim >= 3 and lse.shape[-1] == 1:
        lse = lse.squeeze(-1)
    if lse.ndim >= 3 and lse.shape[0] == 1:
        lse = lse.squeeze(0)
    if tuple(lse.shape) != (num_tokens, num_heads):
        raise RuntimeError(f"softmax_lse must be [T, H]=[{num_tokens}, {num_heads}], got {tuple(softmax_lse.shape)}")
    return lse


def all_gather_dcp_async(
    tensor: torch.Tensor,
    group: DcpGroupCoordinator,
    output: torch.Tensor | None = None,
    async_op: bool = True,
) -> tuple[torch.Tensor, torch.distributed.Work | None]:
    if group.world_size == 1:
        return tensor, None
    if output is None:
        input_size = tensor.size()
        output_size = (input_size[0] * group.world_size,) + input_size[1:]
        output = torch.empty(output_size, dtype=tensor.dtype, device=tensor.device)
    return output, dist.all_gather_into_tensor(
        output,
        tensor,
        group=group.device_group,
        async_op=async_op,
    )


def start_dcp_gather(
    tensor: torch.Tensor,
    dim: int,
    split_sizes: tuple[int, ...],
    group: DcpGroupCoordinator,
) -> DcpGatherContext:
    if dim == 0:
        gathered, handle = all_gather_dcp_async(tensor.contiguous(), group)
        restore_perm = None
    else:
        perm = (dim, *[index for index in range(tensor.dim()) if index != dim])
        restore_perm = tuple(perm.index(index) for index in range(tensor.dim()))
        gathered, handle = all_gather_dcp_async(
            tensor.permute(perm).contiguous(),
            group,
        )
    return DcpGatherContext(
        gathered=gathered,
        handle=handle,
        restore_perm=restore_perm,
        split_sizes=split_sizes,
    )


def finish_dcp_gather(
    context: DcpGatherContext,
) -> tuple[torch.Tensor, ...]:
    if context.handle is not None:
        context.handle.wait()
    gathered = context.gathered
    if context.restore_perm is not None:
        gathered = gathered.permute(context.restore_perm).contiguous()
    return torch.split(gathered, context.split_sizes, dim=-1)


def merge_dcp_attention_outputs(
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    group: DcpGroupCoordinator,
    *,
    local_output: torch.Tensor | None = None,
    local_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exchange Query-head shards and merge all attention contributions."""
    if (local_output is None) != (local_lse is None):
        raise ValueError("local_output and local_lse must be supplied together")
    from xllm.python.kernels_npu.triton.dcp_packed_a2a import (
        fused_dcp_lse_combine,
        fused_dcp_lse_combine_with_local,
        pack_dcp_output_lse,
    )

    num_tokens, num_heads, head_dim = (int(size) for size in output.shape)
    lse = normalize_dcp_lse(softmax_lse, num_tokens, num_heads)
    dcp_size = int(group.world_size)
    send, recv, merged = get_dense_dcp_merge_buffers(output, dcp_size)
    pack_dcp_output_lse(
        output,
        lse,
        dcp_size,
        scatter_dim=1,
        send=send,
    )
    dist.all_to_all_single(recv, send, group=group.device_group)
    if local_output is None:
        return fused_dcp_lse_combine(
            recv,
            head_dim,
            scatter_dim=1,
            output=merged,
        )
    assert local_lse is not None
    return fused_dcp_lse_combine_with_local(
        recv,
        head_dim,
        scatter_dim=1,
        local_output=local_output,
        local_lse=local_lse,
        output=merged,
    )


class DenseDcpAttentionBackend(NpuPagedAttentionBackend):
    """FIA backend that shards replicated GQA KV cache across DCP ranks."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
        dcp_group: ProcessGroup,
        num_decoding_tokens: int = 1,
    ) -> None:
        if dcp_group.size() <= 1:
            raise ValueError("Dense DCP attention requires a group larger than one rank")
        if num_decoding_tokens != 1:
            raise NotImplementedError("Dense DCP attention does not support speculative decoding")
        if sliding_window:
            raise NotImplementedError("Dense DCP attention does not support sliding-window attention")
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
            sliding_window=sliding_window,
            is_mla=False,
            device=device,
            dtype=dtype,
            num_decoding_tokens=num_decoding_tokens,
        )
        self._dcp_group = TorchDcpGroupCoordinator.from_process_group(dcp_group)
        self._fia = DenseDcpFia(num_heads * dcp_group.size(), num_kv_heads, head_dim, scale, dtype)

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        super().bind_kv_caches(kv_caches)
        self._fia.bind_layer_caches(kv_caches)

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        # Retain the common runner interface; execution mechanics live below
        # the attention algorithm, and all batch derivation lives in the builder.
        self._fia.prepare(metadata, graph_mode=graph_mode)

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
    ) -> torch.Tensor:
        if not layer.causal:
            raise NotImplementedError("Dense DCP attention supports only causal attention")
        dcp_metadata = get_execution_context(DenseDcpMetadata)
        if dcp_metadata is None:
            raise RuntimeError("Dense DCP attention requires DenseDcpMetadata from its metadata builder")

        layer_cache = self._kv_caches[layer.layer_id]
        key_cache, value_cache = layer_cache.key, layer_cache.value
        if key_cache is None or value_cache is None:
            raise RuntimeError(f"KV cache is missing for layer {layer.layer_id}")

        num_tokens = int(q.shape[0])
        query = q.view(num_tokens, self.num_heads, self.head_dim).contiguous()
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()

        query_gather: DcpGatherContext | None = None
        if dcp_metadata.is_chunked_prefill or not dcp_metadata.is_prefill:
            query_gather = start_dcp_gather(
                query,
                dim=1,
                split_sizes=(self.head_dim,),
                group=self._dcp_group,
            )

        kernels.reshape_paged_cache(
            dcp_metadata.local_slot_mapping,
            key,
            value,
            key_cache,
            value_cache,
        )

        if dcp_metadata.is_prefill and not dcp_metadata.is_chunked_prefill:
            # All current K/V are available locally. A table, if supplied, only
            # addresses a DCP shard and must not select the parent's paged path.
            output, _ = self._run_current_chunk_attention(query, key, value, dcp_metadata)
            return output.reshape(num_tokens, self.num_heads * self.head_dim)

        assert query_gather is not None
        (gathered_query,) = finish_dcp_gather(query_gather)
        if dcp_metadata.is_chunked_prefill:
            return self._chunked_prefill_dcp(
                gathered_query,
                key,
                value,
                key_cache,
                value_cache,
                num_tokens,
                dcp_metadata,
            )
        return self._decode_dcp(
            gathered_query,
            key_cache,
            value_cache,
            num_tokens,
            dcp_metadata,
        )

    def _run_current_chunk_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: DenseDcpMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            pse_shift=None,
            atten_mask=self._causal_mask,
            actual_seq_lengths=metadata.query_seq_ends,
            actual_seq_lengths_kv=metadata.query_seq_ends,
            num_heads=query.shape[1],
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=_SPARSE_MODE_RIGHT_DOWN_CAUSAL,
            softmax_lse_flag=metadata.is_chunked_prefill,
        )

    def _chunked_prefill_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_tokens: int,
        metadata: DenseDcpMetadata,
    ) -> torch.Tensor:
        current_output, current_lse = self._run_current_chunk_attention(
            query,
            key,
            value,
            metadata,
        )
        local_head_start = self._dcp_group.rank_in_group * self.num_heads
        local_head_end = local_head_start + self.num_heads
        current_output = current_output[:, local_head_start:local_head_end]
        current_lse = normalize_dcp_lse(
            current_lse,
            num_tokens,
            self.num_heads * self._dcp_group.world_size,
        )[:, local_head_start:local_head_end]

        if not metadata.has_context:
            return current_output.reshape(num_tokens, self.num_heads * self.head_dim)

        if metadata.has_local_context:
            context_output, context_lse = self._fia.run(
                query,
                key_cache,
                value_cache,
                metadata,
            )
        else:
            context_output = torch.zeros_like(query)
            context_lse = torch.full(
                (*query.shape[:2], 1),
                -torch.inf,
                dtype=torch.float32,
                device=query.device,
            )
        context_lse = normalize_dcp_lse(
            context_lse,
            num_tokens,
            self.num_heads * self._dcp_group.world_size,
        )
        context_lse.masked_fill_(
            metadata.empty_kv_shards[:num_tokens],
            -torch.inf,
        )
        output = merge_dcp_attention_outputs(
            context_output,
            context_lse,
            self._dcp_group,
            local_output=current_output,
            local_lse=current_lse,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    def _decode_dcp(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_tokens: int,
        metadata: DenseDcpMetadata,
    ) -> torch.Tensor:
        partial_output, partial_lse = self._fia.run(
            query,
            key_cache,
            value_cache,
            metadata,
        )

        normalized_lse = normalize_dcp_lse(
            partial_lse,
            num_tokens,
            self.num_heads * self._dcp_group.world_size,
        )
        normalized_lse.masked_fill_(
            metadata.empty_kv_shards[:num_tokens],
            -torch.inf,
        )
        output = merge_dcp_attention_outputs(
            partial_output,
            normalized_lse,
            self._dcp_group,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)
