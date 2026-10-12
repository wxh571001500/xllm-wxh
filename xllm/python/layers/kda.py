# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kimi-K3 Delta Attention (KDA), sharing the GLM AscendC KDA kernels.

Short convolution reuses FLA prefill and the xllm decode operator. Prefill and decode use
``fla_npu.ops.ascendc.chunk_kda_fwd`` and ``recurrent_kda`` respectively.
Beta is sigmoid-activated by the caller; the operators fuse gate activation
and decode Q/K normalization. Tensor parallelism only; PP/SP and speculative
decoding are out of scope.

Caller-owned caches:

- conv_state: [num_slots, conv_size - 1, 3 * local_proj], model dtype.
- recurrent_state: [num_slots, local_num_heads, V, K], float32.

Decode packs tokens as TND and updates indexed state slots in place. Prefill
uses packed BSND with sequence boundaries and returns VK states, avoiding a
transpose at the operator boundary. Padding decode slots use -1.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from fla_npu.ops.ascendc import chunk_kda_fwd, npu_causal_conv1d, recurrent_kda
from torch import nn

from xllm.python.layers.attention import AttentionRuntimeLayer
from xllm.python.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    W8A8DynamicLinearMethod,
)

KDA_CHUNK_SIZE = 64
# The executor uses this sentinel for dummy rows in a static decode graph.
PAD_SLOT_ID = -1
_L2NORM_EPS = 1e-6


@dataclass
class KimiK3KDAMetadata:
    """Per-step scheduling info for KDA layers.

    Tokens of the whole batch are packed along dim 0. The leading
    ``num_decode_seqs`` sequences are decode requests (exactly 1 token each);
    the following ``num_prefill_seqs`` sequences are (chunked-)prefill
    requests.
    """

    # [num_seqs + 1] int32 on device, cumulative token counts.
    query_start_loc: torch.Tensor
    # [num_seqs] int64 on device, conv/recurrent state slot id per sequence.
    state_indices: torch.Tensor
    num_decode_seqs: int
    num_prefill_seqs: int
    # [num_seqs] bool; True when the sequence's conv/recurrent state must be
    # loaded from the caches (always True for decode; True for chunked-prefill
    # continuations). Required whenever ``num_prefill_seqs > 0``.
    has_initial_state: torch.Tensor | None = None
    graph_num_tokens: int | None = None
    empty_shard: bool = False
    # Host-side metadata shared by all KDA layers in one prefill step.  The
    # executor builds it once so each layer does not call ``tolist()`` again.
    prefill_cu_seqlens_host: tuple[int, ...] | None = None
    prefill_chunk_indices_host: tuple[int, ...] | None = None


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt((x32 * x32).sum(-1, keepdim=True) + _L2NORM_EPS)).to(x.dtype)


def _build_chunk_indices(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    """Flat [seq_idx, chunk_idx] pairs, mirroring build_kda_chunk_indices."""
    indices: list[int] = []
    for seq in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq + 1] - cu_seqlens[seq]
        for chunk in range(-(-seq_len // chunk_size)):
            indices.extend((seq, chunk))
    return indices


def build_kda_prefill_host_metadata(
    query_start_loc: torch.Tensor | None,
    num_decode_seqs: int,
) -> tuple[tuple[int, ...], tuple[int, ...]] | tuple[None, None]:
    """Build host metadata shared by every KDA layer in one prefill step.

    The AscendC prefill kernels accept host-side cumulative lengths and chunk
    indices.  Building them once per scheduler step avoids a device-to-host
    sync and Python loop in every KDA layer.
    """
    if query_start_loc is None:
        return None, None

    prefill_cu = query_start_loc[num_decode_seqs:]
    if prefill_cu.numel() <= 1:
        return None, None

    cu_values = [int(value) for value in prefill_cu.detach().cpu().reshape(-1).tolist()]
    cu_list = [value - cu_values[0] for value in cu_values]
    chunk_indices = tuple(_build_chunk_indices(cu_list, KDA_CHUNK_SIZE))
    return tuple(cu_list), chunk_indices


class KimiK3DeltaAttention(AttentionRuntimeLayer, nn.Module):
    """Kimi-K3 KDA layer with a full-rank per-channel decay gate.

    A runtime attention layer (``attention_kind="linear"``) so the executor
    collects it alongside MLA/MHA layers and the decoder layer ids stay
    contiguous. Unlike paged-attention layers it reads its conv/recurrent state
    from ``kda_runtime`` rather than the attention backend, so its
    ``attention_layer_spec`` is only used for identity/ordering, not paged-KV
    geometry.

    Args:
        hidden_size: model hidden size.
        linear_attn_config: the HF config's ``linear_attn_config`` dict; must
            contain ``num_heads``, ``head_dim``, ``short_conv_kernel_size`` and
            ``use_full_rank_gate=True``. ``gate_lower_bound`` is optional.
        layer_id: global 0-based decoder layer index (used by the executor for
            runtime-layer ordering and KDA cache/metadata routing).
        tp_size / tp_rank: tensor-parallel world size and rank. Heads (and the
            tied projections / conv / biases) are head-sharded per rank; the
            low-rank ``f_a`` projection is replicated on every rank.
        rms_norm_eps: epsilon of the gated output RMSNorm (the model config's
            ``rms_norm_eps``).
        dtype / device: parameter dtype and device.
    """

    attention_kind = "linear"

    def __init__(
        self,
        hidden_size: int,
        linear_attn_config: dict,
        *,
        layer_id: int = 0,
        tp_size: int = 1,
        tp_rank: int = 0,
        rms_norm_eps: float = 1e-6,
        quantized: bool = False,
        reduce_o_proj: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        assert linear_attn_config.get("use_full_rank_gate", False), "KimiK3DeltaAttention requires a full-rank gate"
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.quantized = quantized
        self.head_dim = linear_attn_config["head_dim"]
        self.num_heads = linear_attn_config["num_heads"]
        assert self.num_heads % tp_size == 0
        self.local_num_heads = self.num_heads // tp_size
        self.projection_size = self.head_dim * self.num_heads
        self.local_projection_size = self.projection_size // tp_size
        self.conv_size = linear_attn_config["short_conv_kernel_size"]
        # Runtime-layer spec fields (see AttentionRuntimeLayer). KDA does not use
        # the paged-KV backend, so num_kv_heads/sliding_window are nominal.
        self.num_kv_heads = self.local_num_heads
        self.scale = self.head_dim**-0.5
        self.sliding_window = 0
        self.gate_lower_bound: float | None = linear_attn_config.get("gate_lower_bound")
        if self.gate_lower_bound is not None:
            assert -5.0 <= self.gate_lower_bound < 0, (
                f"KDA gate lower bound must be in [-5, 0), got {self.gate_lower_bound}."
            )
        self.o_norm_eps = rms_norm_eps

        local_proj = self.local_projection_size
        if quantized:
            # q/k/v are W8A8_DYNAMIC in the Kimi-K3 checkpoint (int8 weight +
            # per-token int8 activation quant); the gate/output/low-rank/beta
            # projections stay bf16. Each rank owns its head shard and feeds the
            # per-rank conv/recurrent kernels, so no output gather/reduce.
            self.q_proj = ColumnParallelLinear(
                hidden_size,
                local_proj,
                tp_size,
                bias=False,
                dtype=dtype,
                device=device,
                quant_method=W8A8DynamicLinearMethod(),
            )
            self.k_proj = ColumnParallelLinear(
                hidden_size,
                local_proj,
                tp_size,
                bias=False,
                dtype=dtype,
                device=device,
                quant_method=W8A8DynamicLinearMethod(),
            )
            self.v_proj = ColumnParallelLinear(
                hidden_size,
                local_proj,
                tp_size,
                bias=False,
                dtype=dtype,
                device=device,
                quant_method=W8A8DynamicLinearMethod(),
            )
        else:
            self.q_proj = ColumnParallelLinear(hidden_size, local_proj, tp_size, bias=False, dtype=dtype, device=device)
            self.k_proj = ColumnParallelLinear(hidden_size, local_proj, tp_size, bias=False, dtype=dtype, device=device)
            self.v_proj = ColumnParallelLinear(hidden_size, local_proj, tp_size, bias=False, dtype=dtype, device=device)
        # Full-rank output gate (sigmoid), applied by the gated output norm.
        self.g_proj = ColumnParallelLinear(hidden_size, local_proj, tp_size, bias=False, dtype=dtype, device=device)
        self.b_proj = ColumnParallelLinear(
            hidden_size,
            self.local_num_heads,
            tp_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        # Low-rank input of the decay gate; replicated on every TP rank.
        self.f_a_proj = ColumnParallelLinear(
            hidden_size,
            self.head_dim,
            tp_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.f_b_proj = ColumnParallelLinear(self.head_dim, local_proj, tp_size, bias=False, dtype=dtype, device=device)
        # Packed [q, k, v] short-conv weights, checkpoint layout, kept fp32.
        self.conv1d_weight = nn.Parameter(
            torch.empty(3 * local_proj, 1, self.conv_size, dtype=torch.float32, device=device)
        )
        # [conv_size, 3 * local_proj] copy in the model dtype consumed by the
        # AscendC conv operator; built by process_weights_after_loading().
        self.register_buffer(
            "conv_weight_t",
            torch.zeros(self.conv_size, 3 * local_proj, dtype=dtype, device=device),
            persistent=False,
        )
        self.A_log = nn.Parameter(torch.empty(self.local_num_heads, dtype=torch.float32, device=device))
        self.dt_bias = nn.Parameter(torch.empty(local_proj, dtype=torch.float32, device=device))
        self.o_norm_weight = nn.Parameter(torch.empty(self.head_dim, dtype=dtype, device=device))
        self.o_proj = RowParallelLinear(
            local_proj,
            hidden_size,
            tp_size,
            bias=False,
            dtype=dtype,
            device=device,
            reduce_results=reduce_o_proj,
        )

    # -- state cache contract --------------------------------------------------

    def conv_state_shape(self) -> tuple[int, ...]:
        return (self.conv_size - 1, 3 * self.local_projection_size)

    def recurrent_state_shape(self) -> tuple[int, ...]:
        return (self.local_num_heads, self.head_dim, self.head_dim)

    def state_dtypes(self) -> tuple[torch.dtype, torch.dtype]:
        return (self.o_norm_weight.dtype, torch.float32)

    # -- weight loading ----------------------------------------------------------

    def load_weights(
        self,
        prefix: str,
        find: Callable[[str], torch.Tensor | None],
    ) -> set[str]:
        """Load checkpoint shards into this layer's parameters.

        Args:
            prefix: this layer's checkpoint prefix (e.g.
                ``model.layers.0.self_attn``). Pass ``""`` to look weights up
                by their bare names.
            find: ``find(name) -> tensor | None`` lookup over the checkpoint
                state dicts.

        Returns:
            The set of checkpoint names (relative to ``prefix``) consumed.
        """
        consumed: set[str] = set()

        def get(name: str) -> torch.Tensor:
            full = f"{prefix}.{name}" if prefix else name
            tensor = find(full)
            if tensor is None:
                raise KeyError(f"missing checkpoint weight: {full}")
            consumed.add(name)
            return tensor

        def shard(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
            size = t.size(dim) // self.tp_size
            return t.narrow(dim, self.tp_rank * size, size).contiguous()

        # Head-tied rows are laid out head-major, so equal splits shard heads.
        for name in ("q_proj", "k_proj", "v_proj"):
            proj = getattr(self, name)
            if self.quantized:
                # W8A8_DYNAMIC: int8 weight + per-output-channel fp32
                # scale/offset, all sharded on the (head-major) output dim.
                for suffix in ("weight", "weight_scale", "weight_offset"):
                    proj.load_weight(suffix, shard(get(f"{name}.{suffix}")))
            else:
                proj.weight.data.copy_(shard(get(f"{name}.weight")))
        for name in ("g_proj", "b_proj", "f_b_proj"):
            proj = getattr(self, name)
            proj.weight.data.copy_(shard(get(f"{name}.weight")))
        self.f_a_proj.weight.data.copy_(get("f_a_proj.weight"))  # replicated

        conv_parts = []
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            w = get(f"{name}.weight")
            if w.dim() == 2:  # [P, W] -> [P, 1, W]
                w = w.unsqueeze(1)
            conv_parts.append(shard(w))
        self.conv1d_weight.data.copy_(torch.cat(conv_parts, dim=0).float())

        a_log = get("A_log")
        if a_log.dim() == 1:
            # Official K3 stores num_heads real entries followed by zero
            # padding (e.g. 96 of 128).
            a_log = a_log[: self.num_heads]
        elif a_log.dim() == 4:  # legacy (1, 1, H, 1) storage
            a_log = a_log.view(a_log.shape[2])
        self.A_log.data.copy_(shard(a_log).float())
        self.dt_bias.data.copy_(shard(get("dt_bias")).float())
        self.o_norm_weight.data.copy_(get("o_norm.weight"))
        # RowParallelLinear shards the input dim (dim 1).
        self.o_proj.weight.data.copy_(shard(get("o_proj.weight"), dim=1))
        return consumed

    def process_weights_after_loading(self) -> None:
        """Build the transposed, model-dtype conv weight for the AscendC op."""
        self.conv_weight_t = (
            self.conv1d_weight.data.view(3 * self.local_projection_size, -1)
            .t()
            .contiguous()
            .to(self.o_norm_weight.dtype)
        )
        if self.quantized:
            # Transpose the int8 q/k/v weights into the matmul's [in, out]
            # layout and flatten their per-channel scale/offset.
            for name in ("q_proj", "k_proj", "v_proj"):
                getattr(self, name).finish_weight_loading()

    # -- forward -----------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: KimiK3KDAMetadata,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        reduce_o_proj: bool | None = None,
    ) -> torch.Tensor:
        """Args:
        hidden_states: ``[num_tokens, hidden_size]`` packed batch.
        metadata: scheduling info, see :class:`KimiK3KDAMetadata`.
        conv_state: ``[num_slots, conv_size - 1, 3 * local_proj]``.
        recurrent_state: ``[num_slots, H, V, K]`` float32.
        """
        num_tokens = hidden_states.size(0)
        num_decode = metadata.num_decode_seqs
        num_decode_tokens = num_decode  # decode requests carry 1 token each

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        beta = self.b_proj(hidden_states).float().sigmoid().unsqueeze(0)
        raw_gate = self.f_b_proj(self.f_a_proj(hidden_states)).view(1, num_tokens, self.local_num_heads, self.head_dim)
        output_gate = self.g_proj(hidden_states).view(num_tokens, self.local_num_heads, self.head_dim)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if num_decode + metadata.num_prefill_seqs > 0:
            mixed_qkv = torch.cat((q, k, v), dim=-1)
            if metadata.num_prefill_seqs > 0:
                assert metadata.has_initial_state is not None
                mixed_qkv = self._causal_conv1d(
                    mixed_qkv,
                    conv_state,
                    query_start_loc=metadata.query_start_loc,
                    cache_indices=metadata.state_indices,
                    initial_state_mode=metadata.has_initial_state,
                    run_mode=0,
                )
            else:
                mixed_qkv = self._causal_conv1d(
                    mixed_qkv,
                    conv_state,
                    query_start_loc=metadata.query_start_loc,
                    cache_indices=metadata.state_indices,
                    initial_state_mode=None,
                    run_mode=1,
                )

            q, k, v = (
                x.reshape(1, num_tokens, self.local_num_heads, self.head_dim) for x in mixed_qkv.chunk(3, dim=-1)
            )

            if num_decode > 0:
                core_attn_out[:, :num_decode_tokens] = self._recurrent(
                    q[:, :num_decode_tokens],
                    k[:, :num_decode_tokens],
                    v[:, :num_decode_tokens],
                    raw_gate[:, :num_decode_tokens],
                    beta[:, :num_decode_tokens],
                    recurrent_state,
                    cu_seqlens=metadata.query_start_loc[: num_decode + 1],
                    state_indices=metadata.state_indices[:num_decode],
                )
            if metadata.num_prefill_seqs > 0:
                core_attn_out[:, num_decode_tokens:] = self._prefill(
                    q[:, num_decode_tokens:],
                    k[:, num_decode_tokens:],
                    v[:, num_decode_tokens:],
                    raw_gate[:, num_decode_tokens:],
                    beta[:, num_decode_tokens:],
                    recurrent_state,
                    metadata,
                )

        out = self._gated_rms_norm(core_attn_out, output_gate.unsqueeze(0))
        output = self.o_proj(
            out.reshape(num_tokens, self.local_projection_size),
            reduce_results=reduce_o_proj,
        )
        return output

    # -- internals ---------------------------------------------------------------

    def _causal_conv1d(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        *,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        initial_state_mode: torch.Tensor | None,
        run_mode: int,
    ) -> torch.Tensor:
        if run_mode == 0:
            return npu_causal_conv1d(
                mixed_qkv,
                self.conv_weight_t,
                conv_states=conv_state,
                cache_indices=cache_indices.tolist(),
                initial_state_mode=initial_state_mode.to(torch.int64).tolist(),
                query_start_loc=query_start_loc.tolist(),
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=0,
                head_num=0,
            )
        # The existing tiled kernel disables its padding check for a negative
        # sentinel. Use a positive out-of-pool sentinel so dummy graph rows
        # never access the cache; valid state slots keep their original IDs.
        pad_slot = conv_state.shape[0]
        slots = torch.where(cache_indices < 0, pad_slot, cache_indices)
        output = torch.ops.xllm_ops.causal_conv1d_update_v2(
            mixed_qkv,
            conv_state.transpose(1, 2),
            self.conv_weight_t.transpose(0, 1),
            True,
            None,
            slots,
            query_start_loc,
            1,
            pad_slot,
            None,
        )
        return output.masked_fill(cache_indices.reshape(-1, 1) < 0, 0)

    def _recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> torch.Tensor:
        # The shared recurrent kernel aborts a core on an invalid state slot.
        # Represent padding as empty sequences instead, and pack the valid
        # single-token rows without a host synchronization during graph replay.
        valid = state_indices >= 0
        num_sequences = state_indices.numel()
        order = None
        if num_sequences > 1:
            positions = torch.arange(num_sequences, device=state_indices.device)
            order = torch.argsort(((~valid).long() * num_sequences + positions).float())
        # A one-row graph has no later valid row to lose when the kernel
        # skips a padding slot. Preserve the existing boundaries in this case.
        packed_cu = (
            cu_seqlens
            if num_sequences == 1
            else torch.cat((valid.new_zeros(1, dtype=torch.int64), valid.long().cumsum(0)))
        )

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            flat = tensor.reshape(num_sequences, *tensor.shape[-2:])
            return (flat if order is None else flat.index_select(0, order)).contiguous()

        packed_beta = beta.reshape(num_sequences, self.local_num_heads)
        packed_slots = state_indices.to(torch.int64)
        if order is not None:
            packed_beta = packed_beta.index_select(0, order)
            packed_slots = packed_slots.index_select(0, order)
        output, _ = recurrent_kda(
            pack(q),
            pack(k),
            pack(v),
            pack(raw_gate),
            packed_beta.contiguous(),
            initial_state=recurrent_state,
            cu_seqlens=packed_cu,
            ssm_state_indices=packed_slots.contiguous(),
            A_log=self.A_log.reshape(-1).contiguous(),
            dt_bias=self.dt_bias.contiguous(),
            layout="TND",
            scale=self.scale,
            output_final_state=True,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound,
            state_v_first=True,
        )
        if num_sequences > 1:
            rows = (packed_cu[1:] - 1).clamp_min(0)
            output = output.index_select(0, rows)
        output = output.masked_fill(~valid.reshape(-1, 1, 1), 0)
        return output.reshape(v.shape)

    def _prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        metadata: KimiK3KDAMetadata,
    ) -> torch.Tensor:
        num_decode = metadata.num_decode_seqs
        cu_list = metadata.prefill_cu_seqlens_host
        chunk_indices = metadata.prefill_chunk_indices_host
        if cu_list is None or chunk_indices is None:
            # Fallback for callers that construct metadata without the shared
            # host metadata (for example unit tests).
            prefill_cu = metadata.query_start_loc[num_decode:]
            cu_list = [int(value) for value in prefill_cu.tolist()]
            cu_list = [value - cu_list[0] for value in cu_list]
            chunk_indices = tuple(_build_chunk_indices(cu_list, KDA_CHUNK_SIZE))
        slots = metadata.state_indices.long()[num_decode:]
        has_init = metadata.has_initial_state[num_decode:]

        initial_state = recurrent_state.index_select(0, slots)
        initial_state = initial_state * has_init.view(-1, 1, 1, 1)
        result = chunk_kda_fwd(
            _l2norm(q.contiguous()),
            _l2norm(k.contiguous()),
            v.contiguous(),
            raw_gate.contiguous(),
            beta.contiguous(),
            self.scale,
            chunk_size=KDA_CHUNK_SIZE,
            layout="BSND",
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_list,
            chunk_indices=chunk_indices,
            use_gate_in_kernel=True,
            A_log=self.A_log.reshape(-1).contiguous(),
            dt_bias=self.dt_bias.contiguous(),
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound,
            return_intermediate_states=False,
            state_v_first=True,
        )
        recurrent_state.index_copy_(0, slots, result[1].to(recurrent_state.dtype))
        return result[0]

    def _gated_rms_norm(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        y = x32 * torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + self.o_norm_eps) * self.o_norm_weight.float()
        return (y * torch.sigmoid(gate.float())).to(x.dtype)
