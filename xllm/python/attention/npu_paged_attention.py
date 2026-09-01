# Copyright 2025-2026 The xLLM Authors.
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

"""NPU attention backend using Fused-Infer-Attention (FIA).

Registers as the PrivateUse1 (NPU) backend for the Python model executor.
Prefill uses FIA TND with causal mask; decode uses FIA TND with block_table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch_npu

from xllm.python import ops
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    KVCache,
    MlaIndexContext,
    MlaUnabsorbedPrefill,
)
from xllm.python.model_executor.forward_context import (
    AclGraphTask,
    get_forward_context,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


_FIA_MLA_HEAD_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128)


def _pad_mla_query_heads(
    q_latent: torch.Tensor,
    q_pe: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad absorbed MLA queries for FIA's 512-dimension head constraint."""
    if q_latent.shape[-1] != 512 or num_heads in _FIA_MLA_HEAD_COUNTS:
        return q_latent, q_pe, num_heads

    padded_num_heads = next(
        (count for count in _FIA_MLA_HEAD_COUNTS if count > num_heads),
        None,
    )
    if padded_num_heads is None:
        raise RuntimeError(f"FIA does not support {num_heads} absorbed MLA heads")
    head_padding = padded_num_heads - num_heads
    return (
        F.pad(q_latent, (0, 0, 0, head_padding)),
        F.pad(q_pe, (0, 0, 0, head_padding)),
        padded_num_heads,
    )


class NpuPagedAttentionBackend(AttentionBackend):
    """NPU attention backend dispatching to npu_fused_infer_attention_score."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
        has_mha_layers: bool = True,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.sliding_window = sliding_window
        self.dtype = dtype
        self.device = device
        self._has_mha_layers = has_mha_layers

        self._kv_caches: list[KVCache] = []
        self._metadata: AttentionMetadata | None = None
        self._graph_workspace: torch.Tensor | None = None
        self._graph_outputs: dict[int, torch.Tensor] = {}
        self._graph_lses: dict[int, torch.Tensor] = {}
        self._current_graph_output: torch.Tensor | None = None
        self._current_graph_lse: torch.Tensor | None = None
        self._mla_v2_graph_workspaces: dict[
            tuple[int, int, int, int],
            torch.Tensor,
        ] = {}
        self._mla_v2_graph_outputs: dict[
            tuple[int, int, int, int, int],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._mla_actual_seq_q: torch.Tensor | None = None
        self._mla_actual_seq_kv: torch.Tensor | None = None
        self._causal_mask = (
            torch.triu(torch.ones(2048, 2048, dtype=torch.float32), 1)
            .to(torch.int8)
            .contiguous()
            .to(device)
        )

    @property
    def num_kv_blocks(self) -> int:
        cache = self._first_paged_cache()
        return 0 if cache is None else cache.shape[0]

    @property
    def page_size(self) -> int:
        cache = self._first_paged_cache()
        return 1 if cache is None else cache.shape[1]

    def _first_paged_cache(self) -> torch.Tensor | None:
        for cache in self._kv_caches:
            key_cache = cache.key if isinstance(cache, LayerCache) else cache[0]
            if key_cache is not None:
                return key_cache
        return None

    def bind_kv_caches(self, kv_caches: list[KVCache | LayerCache]) -> None:
        self._kv_caches = kv_caches

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        self._metadata = metadata
        if metadata.q_cu_seq_lens is not None:
            self._actual_seq_lens: list[int] | None = (
                metadata.q_cu_seq_lens[1:].cpu().tolist()
            )
        else:
            self._actual_seq_lens = None

        block_table = metadata.block_table
        if block_table is None and metadata.paged_kv_indices is not None:
            try:
                _indptr = metadata.paged_kv_indptr
                _indices = metadata.paged_kv_indices.to(torch.int32)
                _batch = _indptr.shape[0] - 1
                if _batch > 0:
                    _counts = _indptr[1:] - _indptr[:-1]
                    _max_blocks = int(_counts.max().item())
                    if _max_blocks > 0:
                        block_table = torch.zeros(
                            (_batch, _max_blocks), dtype=torch.int32,
                            device=_indices.device,
                        )
                        for _i in range(_batch):
                            _s = int(_indptr[_i].item())
                            _e = int(_indptr[_i + 1].item())
                            _n = _e - _s
                            if _n > 0:
                                block_table[_i, :_n] = _indices[_s:_e]
            except Exception:
                block_table = None

        if block_table is not None:
            self._block_table_i32 = block_table.to(torch.int32).to(self.device)

            real_batch = block_table.shape[0]

            kv_host = metadata.kv_seq_lens_host
            if kv_host is not None:
                kv_host = kv_host.cpu()
                if kv_host.numel() == real_batch + 1:
                    per_seq_kv = kv_host[1:] - kv_host[:-1]
                else:
                    per_seq_kv = kv_host
            else:
                per_seq_kv = torch.ones(real_batch, dtype=torch.int32)

            kv_list = per_seq_kv[:real_batch].tolist()

            self._actual_seq_q: list[int] = list(range(1, real_batch + 1))
            self._actual_seq_kv: list[int] = kv_list
        else:
            self._block_table_i32 = None

        if (
            graph_mode
            and self._has_mha_layers
            and self._block_table_i32 is not None
        ):
            graph_batch_size = self._block_table_i32.shape[0]
            if self._graph_workspace is None:
                block_size = self.page_size
                dummy_q = torch.empty(
                    graph_batch_size, self.num_heads, self.head_dim,
                    dtype=self.dtype, device=self.device,
                )
                dummy_kv = torch.empty(
                    self.num_kv_blocks, block_size,
                    self.num_kv_heads * self.head_dim,
                    dtype=self.dtype, device=self.device,
                )
                self._graph_workspace = (
                    torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        query=dummy_q,
                        key=dummy_kv,
                        value=dummy_kv,
                        block_table=self._block_table_i32,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_lengths=self._actual_seq_q,
                        actual_seq_lengths_kv=self._actual_seq_kv,
                        num_key_value_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        sparse_mode=0,
                        scale=self.scale,
                        softmax_lse_flag=False,
                    )
                )
            if graph_batch_size not in self._graph_outputs:
                self._graph_outputs[graph_batch_size] = torch.empty(
                    graph_batch_size,
                    self.num_heads,
                    self.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                self._graph_lses[graph_batch_size] = torch.empty(
                    0, dtype=self.dtype, device=self.device
                )
            self._current_graph_output = self._graph_outputs[graph_batch_size]
            self._current_graph_lse = self._graph_lses[graph_batch_size]

        # Pre-cache MLA (sparse SFA) seq-lens once per step; shared by
        # execute_mla / mla_index_context instead of re-derived per layer.
        if metadata.kv_seq_lens is not None:
            kv_seq_lens = metadata.kv_seq_lens
            mla_device = kv_seq_lens.device
            self._mla_actual_seq_kv = kv_seq_lens.to(torch.int32).to(mla_device)
            if metadata.q_cu_seq_lens is not None:
                self._mla_actual_seq_q = metadata.q_cu_seq_lens[1:].to(
                    torch.int32
                ).to(mla_device)
            else:
                batch = kv_seq_lens.size(0)
                self._mla_actual_seq_q = torch.arange(
                    1, batch + 1, dtype=torch.int32, device=mla_device
                )
        else:
            self._mla_actual_seq_q = None
            self._mla_actual_seq_kv = None

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
    ) -> torch.Tensor:
        metadata = self._metadata
        assert metadata is not None

        layer_id = layer.layer_id
        cache = self._kv_caches[layer_id]
        k_cache = cache.key if isinstance(cache, LayerCache) else cache[0]
        v_cache = cache.value if isinstance(cache, LayerCache) else cache[1]
        num_tokens = q.shape[0]

        # Write KV to paged cache (kernel expects [T, kv_heads, head_dim]).
        k_3d = k.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        v_3d = v.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        ops.reshape_paged_cache(
            metadata.slot_mapping, k_3d, v_3d, k_cache, v_cache
        )

        q_3d = q.view(num_tokens, self.num_heads, self.head_dim).contiguous()

        if metadata.is_prefill or metadata.is_chunked_prefill:
            return self._prefill(q_3d, k_3d, v_3d, metadata, num_tokens)
        return self._decode(q_3d, k_cache, v_cache, metadata, num_tokens)

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent_3d: torch.Tensor,
        k_pe_3d: torch.Tensor,
        layer: Attention,
        topk: torch.Tensor | None = None,
        unabsorbed_prefill: MlaUnabsorbedPrefill | None = None,
    ) -> torch.Tensor:
        """Absorbed-MLA attention. Returns [T, H, kv_lora]; caller bmm's W_UV."""
        metadata = self._metadata
        assert metadata is not None, "execute_mla called before prepare()"
        layer_id = layer.layer_id
        cache = self._kv_caches[layer_id]
        nope_cache = cache.key if isinstance(cache, LayerCache) else cache[0]
        rope_cache = cache.value if isinstance(cache, LayerCache) else cache[1]

        torch_npu._npu_reshape_and_cache(
            key=k_latent_3d,
            value=k_pe_3d,
            key_cache=nope_cache,
            value_cache=rope_cache,
            slot_indices=metadata.slot_mapping,
        )
        if topk is None:
            if unabsorbed_prefill is not None and self.use_unabsorbed_mla_prefill():
                return self._mla_unabsorbed_prefill(
                    unabsorbed_prefill,
                    q_pe,
                    k_pe_3d,
                    metadata,
                    layer,
                )
            if metadata.is_chunked_prefill:
                return self._mla_dense_chunked_prefill(
                    q_latent, q_pe, nope_cache, rope_cache, metadata, layer
                )
            if metadata.is_prefill:
                return self._mla_dense_prefill(
                    q_latent, q_pe, k_latent_3d, k_pe_3d, metadata, layer
                )
            if getattr(layer, "use_vllm_fia_v2_decode", False):
                return self._mla_dense_decode_v2(
                    q_latent,
                    q_pe,
                    nope_cache,
                    rope_cache,
                    layer,
                )
            return self._mla_dense_decode(q_latent, q_pe, nope_cache, rope_cache, layer)
        return self._mla_sparse(
            q_latent, q_pe, nope_cache, rope_cache, topk, metadata.block_table
        )

    def use_unabsorbed_mla_prefill(self) -> bool:
        metadata = self._metadata
        return bool(
            metadata is not None
            and metadata.is_prefill
            and not metadata.is_chunked_prefill
        )

    def mla_index_context(self, layer: Attention) -> MlaIndexContext:
        metadata = self._metadata
        assert metadata is not None, "mla_index_context called before prepare()"
        cache = self._kv_caches[layer.layer_id]
        index_cache = cache.index if isinstance(cache, LayerCache) else cache[2]
        return MlaIndexContext(
            index_cache=index_cache,
            slot_mapping=metadata.slot_mapping,
            block_table=metadata.block_table,
            actual_seq_q=self._mla_actual_seq_q,
            actual_seq_kv=self._mla_actual_seq_kv,
        )

    def _mla_sparse(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        topk: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.ops.xllm_ops.sparse_flash_attention(
            q_latent, nope_cache, nope_cache, topk,
            block_table,
            self._mla_actual_seq_q,
            self._mla_actual_seq_kv,
            q_pe, rope_cache, self.scale, 1,
            "TND", "PA_BSND", 3,
        )
        return out  # [T, H, kv_lora]

    def _mla_dense_prefill(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent: torch.Tensor,
        k_pe: torch.Tensor,
        metadata: AttentionMetadata,
        layer: Attention,
    ) -> torch.Tensor:
        """Dense absorbed MLA prefill with separate latent and positional scores."""
        num_tokens = q_latent.shape[0]
        actual_seq = self._cumulative_seq_lens(metadata, num_tokens)

        # FIA's packed TND path accepts BF16 inputs only. Preserve the caller's
        # dtype at the backend boundary so the absorbed W_UV projection sees the
        # same dtype as q_latent.
        original_dtype = q_latent.dtype
        if original_dtype != torch.bfloat16:
            q_latent = q_latent.to(torch.bfloat16)
            q_pe = q_pe.to(torch.bfloat16)
            k_latent = k_latent.to(torch.bfloat16)
            k_pe = k_pe.to(torch.bfloat16)
        q_latent, q_pe, fia_num_heads = _pad_mla_query_heads(
            q_latent, q_pe, layer.num_heads
        )

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_latent.contiguous(),
            k_latent.contiguous(),
            k_latent.contiguous(),
            query_rope=q_pe.contiguous(),
            key_rope=k_pe.contiguous(),
            pse_shift=None,
            atten_mask=self._causal_mask,
            actual_seq_lengths=actual_seq,
            actual_seq_lengths_kv=actual_seq,
            num_heads=fia_num_heads,
            scale=layer.scale,
            input_layout="TND",
            num_key_value_heads=layer.num_kv_heads,
            sparse_mode=3,
            softmax_lse_flag=False,
        )
        output = output.view(num_tokens, fia_num_heads, q_latent.shape[-1])
        output = output[:, : layer.num_heads]
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output

    def _mla_unabsorbed_prefill(
        self,
        inputs: MlaUnabsorbedPrefill,
        query_position: torch.Tensor,
        key_position: torch.Tensor,
        metadata: AttentionMetadata,
        layer: Attention,
    ) -> torch.Tensor:
        """Kimi-style no-RoPE prefill matching vLLM-Ascend's FIA path."""
        num_tokens = inputs.query_nope.shape[0]
        actual_seq = self._cumulative_seq_lens(metadata, num_tokens)
        expanded_key_position = key_position.expand(
            -1,
            layer.num_heads,
            -1,
        )
        query = torch.cat((inputs.query_nope, query_position), dim=-1)
        key = torch.cat((inputs.key_nope, expanded_key_position), dim=-1)
        output, _ = torch_npu.npu_fused_infer_attention_score(
            query.contiguous(),
            key.contiguous(),
            inputs.value.contiguous(),
            num_heads=layer.num_heads,
            num_key_value_heads=layer.num_heads,
            input_layout="TND",
            atten_mask=self._causal_mask,
            sparse_mode=3,
            scale=layer.scale,
            antiquant_mode=0,
            antiquant_scale=None,
            block_table=None,
            block_size=0,
            softmax_lse_flag=True,
            actual_seq_lengths=actual_seq,
            actual_seq_lengths_kv=actual_seq,
        )
        return output.reshape(num_tokens, layer.num_heads, -1)

    def _mla_dense_chunked_prefill(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        metadata: AttentionMetadata,
        layer: Attention,
    ) -> torch.Tensor:
        """Dense absorbed MLA chunked prefill over the updated paged caches."""
        num_tokens = q_latent.shape[0]
        if self._block_table_i32 is None:
            raise RuntimeError("dense MLA chunked prefill requires a block table")

        actual_seq_q = self._cumulative_seq_lens(metadata, num_tokens)
        actual_seq_kv = self._chunked_kv_seq_lens(metadata)
        if len(actual_seq_q) != len(actual_seq_kv):
            raise RuntimeError(
                "dense MLA chunked prefill requires matching query and KV batches"
            )

        block_size = nope_cache.size(1)
        nope_flat = nope_cache.view(nope_cache.size(0), block_size, -1)
        rope_flat = rope_cache.view(rope_cache.size(0), block_size, -1)
        original_dtype = q_latent.dtype
        if original_dtype != torch.bfloat16:
            q_latent = q_latent.to(torch.bfloat16)
            q_pe = q_pe.to(torch.bfloat16)
        q_latent, q_pe, fia_num_heads = _pad_mla_query_heads(
            q_latent, q_pe, layer.num_heads
        )
        if nope_flat.dtype != torch.bfloat16 or rope_flat.dtype != torch.bfloat16:
            raise RuntimeError(
                "dense MLA paged caches must use BF16 for NPU FIA chunked prefill"
            )

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_latent.contiguous(),
            nope_flat.contiguous(),
            nope_flat.contiguous(),
            query_rope=q_pe.contiguous(),
            key_rope=rope_flat.contiguous(),
            pse_shift=None,
            atten_mask=self._causal_mask,
            actual_seq_lengths=actual_seq_q,
            actual_seq_lengths_kv=actual_seq_kv,
            block_table=self._block_table_i32[: len(actual_seq_kv)],
            num_heads=fia_num_heads,
            scale=layer.scale,
            input_layout="TND",
            num_key_value_heads=layer.num_kv_heads,
            sparse_mode=3,
            block_size=block_size,
            softmax_lse_flag=False,
        )
        output = output.view(num_tokens, fia_num_heads, q_latent.shape[-1])
        output = output[:, : layer.num_heads]
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output

    def _mla_dense_decode(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        layer: Attention,
    ) -> torch.Tensor:
        """Dense absorbed MLA decode directly over paged latent caches."""
        num_tokens = q_latent.shape[0]
        if self._block_table_i32 is None:
            raise RuntimeError("dense MLA decode requires a block table")

        block_size = nope_cache.size(1)
        nope_flat = nope_cache.view(nope_cache.size(0), block_size, -1)
        rope_flat = rope_cache.view(rope_cache.size(0), block_size, -1)
        original_dtype = q_latent.dtype
        if original_dtype != torch.bfloat16:
            q_latent = q_latent.to(torch.bfloat16)
            q_pe = q_pe.to(torch.bfloat16)
        q_latent, q_pe, fia_num_heads = _pad_mla_query_heads(
            q_latent, q_pe, layer.num_heads
        )
        if nope_flat.dtype != torch.bfloat16 or rope_flat.dtype != torch.bfloat16:
            raise RuntimeError(
                "dense MLA paged caches must use BF16 for NPU FIA decode"
            )

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_latent.contiguous(),
            nope_flat.contiguous(),
            nope_flat.contiguous(),
            query_rope=q_pe.contiguous(),
            key_rope=rope_flat.contiguous(),
            pse_shift=None,
            atten_mask=None,
            actual_seq_lengths=self._actual_seq_q[:num_tokens],
            actual_seq_lengths_kv=self._actual_seq_kv[:num_tokens],
            block_table=self._block_table_i32[:num_tokens],
            num_heads=fia_num_heads,
            scale=layer.scale,
            input_layout="TND",
            num_key_value_heads=layer.num_kv_heads,
            sparse_mode=0,
            block_size=block_size,
            softmax_lse_flag=False,
        )
        output = output.view(num_tokens, fia_num_heads, q_latent.shape[-1])
        output = output[:, : layer.num_heads]
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output

    def _mla_dense_decode_v2(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        layer: Attention,
    ) -> torch.Tensor:
        """Kimi-style absorbed decode matching vLLM-Ascend's FIA v2 path."""
        num_tokens = q_latent.shape[0]
        if self._block_table_i32 is None:
            raise RuntimeError("dense MLA decode requires a block table")

        block_size = nope_cache.size(1)
        nope_cache = nope_cache.view(
            -1,
            layer.num_kv_heads,
            block_size,
            q_latent.shape[-1],
        )
        rope_cache = rope_cache.view(
            -1,
            layer.num_kv_heads,
            block_size,
            q_pe.shape[-1],
        )
        original_dtype = q_latent.dtype
        if original_dtype != torch.bfloat16:
            q_latent = q_latent.to(torch.bfloat16)
            q_pe = q_pe.to(torch.bfloat16)
        q_latent, q_pe, fia_num_heads = _pad_mla_query_heads(
            q_latent,
            q_pe,
            layer.num_heads,
        )
        query = q_latent.view(
            num_tokens,
            fia_num_heads,
            1,
            q_latent.shape[-1],
        ).contiguous()
        query_position = q_pe.view(
            num_tokens,
            fia_num_heads,
            1,
            q_pe.shape[-1],
        )
        graph_context = get_forward_context().acl_graph
        if graph_context is not None:
            shape_key = (
                num_tokens,
                fia_num_heads,
                layer.num_kv_heads,
                q_latent.shape[-1],
            )
            output_key = (layer.layer_id, *shape_key)
            outputs = self._mla_v2_graph_outputs.get(output_key)
            if outputs is None:
                # BNSD_NBSD returns head-major output. Allocate it explicitly,
                # matching vLLM-Ascend, so graph capture never depends on the
                # generic output-shape inference used by the default handler.
                # Keep outputs layer-local because all captured MLA layers
                # remain live in the same graph.
                output = torch.empty(
                    fia_num_heads,
                    num_tokens,
                    1,
                    q_latent.shape[-1],
                    dtype=query.dtype,
                    device=query.device,
                )
                lse = torch.empty(
                    num_tokens,
                    dtype=query.dtype,
                    device=query.device,
                )
                outputs = (output, lse)
                self._mla_v2_graph_outputs[output_key] = outputs
            attn_output, softmax_lse = outputs

            workspace = self._mla_v2_graph_workspaces.get(shape_key)
            if workspace is None:
                workspace = (
                    torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                        query,
                        nope_cache,
                        nope_cache,
                        query_rope=query_position,
                        key_rope=rope_cache,
                        num_query_heads=fia_num_heads,
                        num_key_value_heads=layer.num_kv_heads,
                        input_layout="BNSD_NBSD",
                        atten_mask=None,
                        sparse_mode=0,
                        softmax_scale=layer.scale,
                        block_table=self._block_table_i32[:num_tokens],
                        block_size=block_size,
                        actual_seq_qlen=None,
                        actual_seq_kvlen=self._actual_seq_kv[:num_tokens],
                        return_softmax_lse=False,
                    )
                )
                self._mla_v2_graph_workspaces[shape_key] = workspace

            def _run_v2_out() -> None:
                torch_npu.npu_fused_infer_attention_score_v2.out(
                    query,
                    nope_cache,
                    nope_cache,
                    query_rope=query_position,
                    key_rope=rope_cache,
                    num_query_heads=fia_num_heads,
                    num_key_value_heads=layer.num_kv_heads,
                    input_layout="BNSD_NBSD",
                    atten_mask=None,
                    sparse_mode=0,
                    softmax_scale=layer.scale,
                    block_table=self._block_table_i32[:num_tokens],
                    block_size=block_size,
                    actual_seq_qlen=None,
                    actual_seq_kvlen=self._actual_seq_kv[:num_tokens],
                    return_softmax_lse=False,
                    workspace=workspace,
                    out=[attn_output, softmax_lse],
                )

            stream = graph_context.stream
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            try:
                _run_v2_out()
            except Exception:
                torch.npu.graph_task_group_end(stream)
                raise
            handle = torch.npu.graph_task_group_end(stream)
            graph_context.tasks.append(AclGraphTask(event, handle, _run_v2_out))
        else:
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                nope_cache,
                nope_cache,
                query_rope=query_position,
                key_rope=rope_cache,
                num_query_heads=fia_num_heads,
                num_key_value_heads=layer.num_kv_heads,
                input_layout="BNSD_NBSD",
                atten_mask=None,
                sparse_mode=0,
                softmax_scale=layer.scale,
                block_table=self._block_table_i32[:num_tokens],
                block_size=block_size,
                actual_seq_qlen=None,
                actual_seq_kvlen=self._actual_seq_kv[:num_tokens],
                return_softmax_lse=False,
            )
        output = attn_output[: layer.num_heads]
        output = output.view(
            layer.num_heads,
            num_tokens,
            q_latent.shape[-1],
        ).transpose(0, 1)
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output

    # ------------------------------------------------------------------
    # Prefill: packed TND with causal mask
    # ------------------------------------------------------------------

    def _prefill(
        self, q_3d: torch.Tensor, k_3d: torch.Tensor, v_3d: torch.Tensor,
        metadata: AttentionMetadata, num_tokens: int,
    ) -> torch.Tensor:
        actual_seq = self._cumulative_seq_lens(metadata, num_tokens)

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_3d, k_3d, v_3d,
            pse_shift=None,
            atten_mask=self._causal_mask,
            actual_seq_lengths=actual_seq,
            actual_seq_lengths_kv=actual_seq,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=3,
            softmax_lse_flag=False,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Decode: FIA with block_table (paged KV, no gather)
    # ------------------------------------------------------------------

    def _fia_out(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        block_size: int,
    ) -> None:
        torch.ops.npu.npu_fused_infer_attention_score.out(
            q, k, v,
            pse_shift=None,
            atten_mask=None,
            actual_seq_lengths=self._actual_seq_q,
            actual_seq_lengths_kv=self._actual_seq_kv,
            block_table=self._block_table_i32,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=0,
            block_size=block_size,
            softmax_lse_flag=False,
            workspace=self._graph_workspace,
            out=[self._current_graph_output, self._current_graph_lse],
        )

    def _decode(
        self, q_3d: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
        metadata: AttentionMetadata, num_tokens: int,
    ) -> torch.Tensor:
        block_size = k_cache.size(1)
        k_flat = k_cache.view(k_cache.size(0), block_size, -1)
        v_flat = v_cache.view(v_cache.size(0), block_size, -1)

        graph_context = get_forward_context().acl_graph
        if graph_context is not None:
            if self._current_graph_output is None:
                raise RuntimeError("ACL graph output buffer is not prepared")
            stream = graph_context.stream
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            try:
                self._fia_out(q_3d, k_flat, v_flat, block_size)
            except Exception:
                torch.npu.graph_task_group_end(stream)
                raise
            handle = torch.npu.graph_task_group_end(stream)

            def _update_fia_args() -> None:
                self._fia_out(q_3d, k_flat, v_flat, block_size)

            graph_context.tasks.append(
                AclGraphTask(event, handle, _update_fia_args)
            )
            return self._current_graph_output.reshape(
                num_tokens, self.num_heads * self.head_dim
            )

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_3d, k_flat, v_flat,
            pse_shift=None,
            atten_mask=None,
            actual_seq_lengths=self._actual_seq_q[:num_tokens],
            actual_seq_lengths_kv=self._actual_seq_kv[:num_tokens],
            block_table=self._block_table_i32,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=0,
            block_size=block_size,
            softmax_lse_flag=False,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunked_kv_seq_lens(metadata: AttentionMetadata) -> list[int]:
        kv_host = metadata.kv_seq_lens_host
        if kv_host is None:
            raise RuntimeError(
                "dense MLA chunked prefill requires host KV sequence lengths"
            )
        kv_host = kv_host.cpu()
        batch_size = (
            metadata.q_cu_seq_lens.numel() - 1
            if metadata.q_cu_seq_lens is not None
            else metadata.block_table.shape[0]
        )
        # NPU metadata stores one total KV length per sequence. Keep support for
        # the cumulative layout used by non-NPU builders so contract tests and
        # metadata adapters fail safely instead of silently changing semantics.
        if kv_host.numel() == batch_size:
            return kv_host.tolist()
        if kv_host.numel() == batch_size + 1 and kv_host[0].item() == 0:
            return (kv_host[1:] - kv_host[:-1]).tolist()
        raise RuntimeError(
            "dense MLA chunked prefill received invalid host KV sequence lengths"
        )

    def _cumulative_seq_lens(
        self, metadata: AttentionMetadata, num_tokens: int,
    ) -> list[int]:
        if self._actual_seq_lens is not None:
            return self._actual_seq_lens
        return [num_tokens]
