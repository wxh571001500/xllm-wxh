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

"""Kimi-K3 Delta Attention (KDA) layer, NPU implementation.

Pure-Python port of vllm-ascend's ``AscendKimiGatedDeltaNetAttention``
(vllm_ascend/ops/kimi_kda.py), built on the Kimi AscendC operators exposed as
``torch.ops._C_ascend.*`` (see the op schemas in vllm-ascend
csrc/torch_binding.cpp):

- ``npu_causal_conv1d_custom``: varlen causal short-conv (prefill run_mode=0,
  decode run_mode=1), silu-activated, updates the conv cache in place.
- ``recurrent_kda``: decode-time recurrent delta rule; applies L2Norm, the
  full-rank gate (``A_log`` / ``dt_bias`` / ``lower_bound``) and updates the
  recurrent state in place.
- ``kda_gate_cumsum``: prefill gate activation + chunk-local cumulative sum.
- ``chunk_kda_fwd``: chunked prefill delta rule; consumes L2-normalized q/k
  and the gate cumsum, returns the output and the final recurrent state.

Beta is sigmoid-activated outside the kernels (``use_beta_sigmoid_in_kernel``
is False), matching vllm-ascend. Tensor parallelism only; PP/SP and
speculative decoding are out of scope.

State caches are owned by the caller (one pair per KDA layer) and updated in
place:

- ``conv_state``:      ``[num_slots, conv_size - 1, 3 * local_proj]``, model
  dtype (width-first layout, raw pre-conv tokens).
- ``recurrent_state``: ``[num_slots * checkpoint_stride, local_num_heads,
  head_dim, head_dim]``, float32; per-slot layout is ``[H, V, K]``. During
  speculative verification, each logical slot owns one checkpoint row per
  token in the block. The ``chunk_kda_fwd`` operator uses the transposed
  ``[H, K, V]`` layout, so the initial/final states are transposed at that
  operator boundary only.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from xllm.python.layers.attention import AttentionRuntimeLayer
from xllm.python.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    W8A8DynamicLinearMethod,
)
from xllm.python.ascend_custom_ops import ensure_ascend_custom_ops

KDA_CHUNK_SIZE = 64
# Padding slot convention of the AscendC causal-conv operator.
PAD_SLOT_ID = -1
_L2NORM_EPS = 1e-6
_SOFTPLUS_THRESHOLD = 20.0
_KDA_CUSTOM_OPS = (
    "npu_causal_conv1d_custom",
    "recurrent_kda",
    "kda_gate_cumsum",
    "chunk_kda_fwd",
)


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
    # Speculative validation uses one accepted-token count per logical
    # sequence, matching vLLM's grouped q segments.
    num_accepted_tokens: torch.Tensor | None = None
    # Logical cumulative lengths for speculative rows. The token buffer is
    # flat, but causal-conv/recurrent kernels consume q segments per request.
    spec_query_start_loc: torch.Tensor | None = None
    is_spec_verify: bool = False
    graph_num_tokens: int | None = None
    empty_shard: bool = False


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt((x32 * x32).sum(-1, keepdim=True) + _L2NORM_EPS)).to(
        x.dtype
    )


def _causal_conv1d_torch_fallback(
    mixed_qkv: torch.Tensor,
    conv_weight_t: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    initial_state_mode: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
) -> torch.Tensor:
    """Reference causal convolution for environments without ACLNN tiling.

    This path is intentionally opt-in and is for functional validation on
    older CANN releases. It uses ordinary grouped ``conv1d`` and preserves the
    same width-first cache contract as the Ascend custom operator.
    """
    state_len = conv_weight_t.size(0) - 1
    feature_dim = mixed_qkv.size(-1)
    weight = conv_weight_t.transpose(0, 1).contiguous().unsqueeze(1)
    state_work = conv_state.contiguous()
    output = mixed_qkv.clone()

    for row in range(cache_indices.numel()):
        start = int(query_start_loc[row].item())
        end = int(query_start_loc[row + 1].item())
        if end <= start:
            continue
        slot = int(cache_indices[row].item())
        if slot == PAD_SLOT_ID:
            continue
        if slot < 0 or slot >= state_work.size(0):
            raise RuntimeError(
                "Kimi K3 causal-conv fallback received an invalid cache slot: "
                f"slot={slot}, cache_lines={state_work.size(0)}"
            )

        sequence = mixed_qkv[start:end]
        valid_tokens = sequence.size(0)
        if num_accepted_tokens is not None:
            valid_tokens = min(valid_tokens, int(num_accepted_tokens[row].item()))
        if valid_tokens <= 0:
            continue
        sequence = sequence[:valid_tokens]

        if initial_state_mode is not None and not bool(initial_state_mode[row].item()):
            history = torch.zeros(
                (state_len, feature_dim),
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )
        else:
            history = state_work[slot, :state_len, :]

        conv_input = torch.cat((history, sequence), dim=0)
        conv_input = conv_input.transpose(0, 1).unsqueeze(0)
        conv_output = F.conv1d(conv_input, weight, groups=feature_dim)
        output[start : start + valid_tokens] = F.silu(
            conv_output.squeeze(0).transpose(0, 1)
        ).to(mixed_qkv.dtype)

        if valid_tokens >= state_len:
            final_history = sequence[-state_len:]
        else:
            final_history = torch.cat((history[valid_tokens:], sequence), dim=0)
        state_work[slot, :state_len, :].copy_(final_history)

    if state_work is not conv_state:
        conv_state.copy_(state_work)
    return output


def _build_chunk_indices(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    """Flat [seq_idx, chunk_idx] pairs, mirroring build_kda_chunk_indices."""
    indices: list[int] = []
    for seq in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq + 1] - cu_seqlens[seq]
        for chunk in range(-(-seq_len // chunk_size)):
            indices.extend((seq, chunk))
    return indices


def _normalize_query_start_loc(
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Restore one cumulative q segment per logical KDA state slot.

    Some NPU decode/graph paths expose a block as ``[0, 1, ..., N]`` while
    still providing one linear-state slot. vLLM keeps this block as one
    ``[0, N]`` segment; passing the expanded boundaries to causal-conv makes
    the operator interpret one cache line as N independent sequences.
    """
    expected_rows = state_indices.numel()
    if query_start_loc.numel() == expected_rows + 1:
        return query_start_loc
    if (
        expected_rows == 1
        and query_start_loc.numel() == num_tokens + 1
    ):
        return torch.tensor(
            [0, num_tokens],
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
    raise RuntimeError(
        "Kimi K3 KDA query/state metadata mismatch: "
        f"query_start_loc={tuple(query_start_loc.shape)}, "
        f"state_indices={tuple(state_indices.shape)}, num_tokens={num_tokens}"
    )


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
        assert linear_attn_config.get("use_full_rank_gate", False), (
            "KimiK3DeltaAttention requires a full-rank gate"
        )
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
        self.gate_lower_bound: float | None = linear_attn_config.get(
            "gate_lower_bound", None
        )
        if self.gate_lower_bound is not None:
            assert -5.0 <= self.gate_lower_bound < 0, (
                "KDA gate lower bound must be in [-5, 0), "
                f"got {self.gate_lower_bound}."
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
            self.q_proj = ColumnParallelLinear(
                hidden_size, local_proj, tp_size, bias=False, dtype=dtype, device=device
            )
            self.k_proj = ColumnParallelLinear(
                hidden_size, local_proj, tp_size, bias=False, dtype=dtype, device=device
            )
            self.v_proj = ColumnParallelLinear(
                hidden_size, local_proj, tp_size, bias=False, dtype=dtype, device=device
            )
        # Full-rank output gate (sigmoid), applied by the gated output norm.
        self.g_proj = ColumnParallelLinear(
            hidden_size, local_proj, tp_size, bias=False, dtype=dtype, device=device
        )
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
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim, local_proj, tp_size, bias=False, dtype=dtype, device=device
        )
        # Packed [q, k, v] short-conv weights, checkpoint layout, kept fp32.
        self.conv1d_weight = nn.Parameter(
            torch.empty(
                3 * local_proj, 1, self.conv_size, dtype=torch.float32, device=device
            )
        )
        # [conv_size, 3 * local_proj] copy in the model dtype consumed by the
        # AscendC conv operator; built by process_weights_after_loading().
        self.register_buffer(
            "conv_weight_t",
            torch.zeros(self.conv_size, 3 * local_proj, dtype=dtype, device=device),
            persistent=False,
        )
        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32, device=device)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(local_proj, dtype=torch.float32, device=device)
        )
        self.o_norm_weight = nn.Parameter(
            torch.empty(self.head_dim, dtype=dtype, device=device)
        )
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
    ) -> torch.Tensor:
        """Args:
            hidden_states: ``[num_tokens, hidden_size]`` packed batch.
            metadata: scheduling info, see :class:`KimiK3KDAMetadata`.
            conv_state: ``[num_slots, conv_size - 1, 3 * local_proj]``.
            recurrent_state: ``[num_slots, H, V, K]`` float32.
        """
        num_tokens = hidden_states.size(0)
        num_decode = metadata.num_decode_seqs
        num_prefill = metadata.num_prefill_seqs
        num_decode_tokens = num_tokens if metadata.is_spec_verify else num_decode
        spec_num_accepted_tokens = (
            metadata.num_accepted_tokens if metadata.is_spec_verify else None
        )
        if recurrent_state.size(0) % conv_state.size(0) != 0:
            raise RuntimeError(
                "Kimi K3 recurrent cache rows must divide evenly by logical "
                f"slots: recurrent_rows={recurrent_state.size(0)}, "
                f"logical_slots={conv_state.size(0)}"
            )
        checkpoint_stride = recurrent_state.size(0) // conv_state.size(0)
        recurrent_state_indices = (
            metadata.state_indices.to(torch.int64) * checkpoint_stride
        )
        query_start_loc = metadata.query_start_loc
        if metadata.is_spec_verify:
            if metadata.spec_query_start_loc is None:
                raise RuntimeError(
                    "Kimi K3 speculative metadata is missing logical q lengths"
                )
            query_start_loc = metadata.spec_query_start_loc
            if num_decode <= 0 or num_tokens % num_decode != 0:
                raise RuntimeError(
                    "Kimi K3 speculative token rows must divide evenly by "
                    f"logical sequences: tokens={num_tokens}, sequences={num_decode}"
                )
            if spec_num_accepted_tokens is None:
                raise RuntimeError(
                    "Kimi K3 speculative metadata is missing accepted-token counts"
                )
            width = num_tokens // num_decode
            expected_conv_state_len = self.conv_size - 1 + width - 1
            if conv_state.size(1) < expected_conv_state_len:
                raise RuntimeError(
                    "Kimi K3 speculative convolution cache is too short: "
                    f"cache_state_len={conv_state.size(1)}, "
                    f"required={expected_conv_state_len}, width={width}, "
                    f"conv_kernel_size={self.conv_size}"
                )
            if checkpoint_stride < width:
                raise RuntimeError(
                    "Kimi K3 speculative recurrent cache has too few "
                    "checkpoints: "
                    f"checkpoint_stride={checkpoint_stride}, required={width}"
                )
            if metadata.state_indices.numel() != num_decode:
                raise RuntimeError(
                    "Kimi K3 speculative state slots must be sequence-scoped: "
                    f"slots={metadata.state_indices.numel()}, sequences={num_decode}"
                )
            if spec_num_accepted_tokens.numel() != num_decode:
                raise RuntimeError(
                    "Kimi K3 speculative accepted-token counts must be "
                    f"sequence-scoped: counts={spec_num_accepted_tokens.numel()}, "
                    f"sequences={num_decode}"
                )
            offsets = torch.arange(
                width, dtype=torch.int64, device=metadata.state_indices.device
            )
            recurrent_state_indices = (
                metadata.state_indices.to(torch.int64).unsqueeze(1)
                * checkpoint_stride
                + offsets.unsqueeze(0)
            ).contiguous()
        elif num_decode + num_prefill > 0 and not metadata.empty_shard:
            raw_query_start_loc = query_start_loc
            query_start_loc = _normalize_query_start_loc(
                query_start_loc,
                metadata.state_indices,
                num_tokens,
            )
            # A one-slot block can be represented by N q_len=1 rows by the
            # NPU graph builder. Once collapsed to one logical segment, run
            # recurrent KDA over the complete block and keep one output row
            # per input token.
            if query_start_loc.numel() != raw_query_start_loc.numel():
                num_decode = 1
                num_prefill = 0
                num_decode_tokens = num_tokens

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        beta = self.b_proj(hidden_states).float().sigmoid().unsqueeze(0)
        raw_gate = self.f_b_proj(self.f_a_proj(hidden_states)).view(
            1, num_tokens, self.local_num_heads, self.head_dim
        )
        output_gate = self.g_proj(hidden_states).view(
            num_tokens, self.local_num_heads, self.head_dim
        )

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if num_decode + num_prefill > 0:
            mixed_qkv = torch.cat((q, k, v), dim=-1)
            if num_prefill > 0:
                assert metadata.has_initial_state is not None
                mixed_qkv = self._causal_conv1d(
                    mixed_qkv,
                    conv_state,
                    query_start_loc=query_start_loc,
                    cache_indices=metadata.state_indices,
                    initial_state_mode=metadata.has_initial_state,
                    run_mode=0,
                )
            else:
                mixed_qkv = self._causal_conv1d(
                    mixed_qkv,
                    conv_state,
                    query_start_loc=query_start_loc,
                    cache_indices=metadata.state_indices,
                    initial_state_mode=None,
                    run_mode=1,
                    num_accepted_tokens=spec_num_accepted_tokens,
                )

            q, k, v = (
                x.reshape(1, num_tokens, self.local_num_heads, self.head_dim)
                for x in mixed_qkv.chunk(3, dim=-1)
            )

            if num_decode > 0:
                recurrent_cu_seqlens = (
                    query_start_loc
                    if metadata.is_spec_verify
                    else query_start_loc[: num_decode + 1]
                )
                recurrent_indices = (
                    recurrent_state_indices.reshape(-1)
                    if metadata.is_spec_verify
                    else recurrent_state_indices[:num_decode]
                )
                core_attn_out[:, :num_decode_tokens] = self._recurrent(
                    q[:, :num_decode_tokens],
                    k[:, :num_decode_tokens],
                    v[:, :num_decode_tokens],
                    raw_gate[:, :num_decode_tokens],
                    beta[:, :num_decode_tokens],
                    recurrent_state,
                    cu_seqlens=recurrent_cu_seqlens,
                    state_indices=recurrent_indices,
                    num_accepted_tokens=spec_num_accepted_tokens,
                )
            if num_prefill > 0:
                core_attn_out[:, num_decode_tokens:] = (
                    self._prefill(
                        q[:, num_decode_tokens:],
                        k[:, num_decode_tokens:],
                        v[:, num_decode_tokens:],
                        raw_gate[:, num_decode_tokens:],
                        beta[:, num_decode_tokens:],
                        recurrent_state,
                        metadata,
                        recurrent_state_indices,
                    )
                )

        out = self._gated_rms_norm(core_attn_out, output_gate.unsqueeze(0))
        return self.o_proj(out.reshape(num_tokens, self.local_projection_size))

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
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ensure_ascend_custom_ops(_KDA_CUSTOM_OPS)
        if conv_state.shape[-1] != mixed_qkv.shape[-1]:
            raise RuntimeError(
                "Ascend Kimi K3 KDA requires convolution cache layout "
                "[num_cache_lines, state_len, qkv_dim]: "
                f"cache={tuple(conv_state.shape)}, "
                f"mixed_qkv={tuple(mixed_qkv.shape)}"
            )
        if self.conv_weight_t.shape != (
            self.conv_size,
            mixed_qkv.shape[-1],
        ):
            raise RuntimeError(
                "Ascend Kimi K3 KDA convolution weight shape mismatch: "
                f"weight={tuple(self.conv_weight_t.shape)}, "
                f"expected=({self.conv_size}, {mixed_qkv.shape[-1]})"
            )
        if (
            mixed_qkv.dtype != conv_state.dtype
            or mixed_qkv.dtype != self.conv_weight_t.dtype
        ):
            raise RuntimeError(
                "Ascend Kimi K3 KDA causal-conv tensors must have one dtype: "
                f"input={mixed_qkv.dtype}, cache={conv_state.dtype}, "
                f"weight={self.conv_weight_t.dtype}"
            )
        if mixed_qkv.dim() != 2 or conv_state.dim() != 3:
            raise RuntimeError(
                "Ascend Kimi K3 KDA causal-conv expects input [tokens, qkv_dim] "
                "and cache [slots, state_len, qkv_dim]: "
                f"input={tuple(mixed_qkv.shape)}, cache={tuple(conv_state.shape)}"
            )

        # Keep the operator boundary identical to vLLM-Ascend.  In particular,
        # ACLNN requires device-side contiguous int32 row offsets/slot ids;
        # the C++ metadata view is already int32, but pybind tensors can retain
        # a non-contiguous view after graph/speculation slicing.
        mixed_qkv = mixed_qkv.contiguous()
        conv_weight_t = self.conv_weight_t.contiguous()
        query_start_loc = query_start_loc.to(torch.int32).contiguous()
        cache_indices = cache_indices.to(torch.int32).contiguous()
        if initial_state_mode is not None:
            initial_state_mode = initial_state_mode.to(torch.bool).contiguous()
        if num_accepted_tokens is not None:
            num_accepted_tokens = num_accepted_tokens.to(torch.int32).contiguous()
        conv_state_work = conv_state.contiguous()

        fallback_mode = os.getenv(
            "XLLM_KIMI_K3_CAUSAL_CONV_FALLBACK", ""
        ).lower()
        if fallback_mode == "torch":
            return _causal_conv1d_torch_fallback(
                mixed_qkv,
                conv_weight_t,
                conv_state,
                query_start_loc,
                cache_indices,
                initial_state_mode,
                num_accepted_tokens,
            )

        out = torch.empty_like(mixed_qkv)
        try:
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                out,
                mixed_qkv,
                conv_weight_t,
                conv_state=conv_state_work,
                bias_opt=None,
                query_start_loc_opt=query_start_loc,
                cache_indices_opt=cache_indices,
                initial_state_mode_opt=initial_state_mode,
                num_accepted_tokens_opt=num_accepted_tokens,
                activation_mode=1,  # silu
                pad_slot_id=PAD_SLOT_ID,
                run_mode=run_mode,
            )
        except RuntimeError as error:
            if fallback_mode != "auto":
                accepted_values = (
                    None
                    if num_accepted_tokens is None
                    else num_accepted_tokens.tolist()
                )
                raise RuntimeError(
                    "Ascend Kimi K3 causal-conv tiling failed with "
                    f"input={tuple(mixed_qkv.shape)}, "
                    f"weight={tuple(conv_weight_t.shape)}, "
                    f"cache={tuple(conv_state_work.shape)}, "
                    f"query_start_loc={query_start_loc.tolist()}, "
                    f"cache_indices_shape={tuple(cache_indices.shape)}, "
                    f"cache_indices={cache_indices.tolist()}, "
                    f"num_accepted_tokens={accepted_values}, "
                    f"run_mode={run_mode}, dtype={mixed_qkv.dtype}"
                ) from error
            return _causal_conv1d_torch_fallback(
                mixed_qkv,
                conv_weight_t,
                conv_state,
                query_start_loc,
                cache_indices,
                initial_state_mode,
                num_accepted_tokens,
            )
        if conv_state_work is not conv_state:
            conv_state.copy_(conv_state_work)
        return out

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
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops._C_ascend.recurrent_kda(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            raw_gate.contiguous(),
            beta.contiguous(),
            recurrent_state,
            cu_seqlens,
            state_indices,
            self.A_log.reshape(-1).contiguous(),
            self.dt_bias.contiguous(),
            num_accepted_tokens=num_accepted_tokens,
            scale=self.scale,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=(
                self.gate_lower_bound if self.gate_lower_bound is not None else -5.0
            ),
        )

    def _prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
        metadata: KimiK3KDAMetadata,
        recurrent_state_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_decode = metadata.num_decode_seqs
        prefill_cu = metadata.query_start_loc[num_decode:]
        # The AscendC prefill operators take host-side cumulative lengths.
        cu_list = (prefill_cu - prefill_cu[0]).tolist()
        cu_list = [int(x) for x in cu_list]
        slots = recurrent_state_indices[num_decode:]
        has_init = metadata.has_initial_state[num_decode:]

        # The recurrent cache is [H, V, K]; the chunk operator uses [H, K, V].
        initial_state_vk = recurrent_state.index_select(0, slots)
        initial_state_vk = initial_state_vk * has_init.view(-1, 1, 1, 1)
        initial_state_kv = initial_state_vk.transpose(-1, -2).contiguous()

        q = _l2norm(q.contiguous())
        k = _l2norm(k.contiguous())
        if self.gate_lower_bound is not None:
            gate_cumsum = torch.ops._C_ascend.kda_gate_cumsum(
                raw_gate.contiguous(),
                KDA_CHUNK_SIZE,
                A_log=self.A_log.reshape(-1).contiguous(),
                dt_bias=self.dt_bias.contiguous(),
                cu_seqlens=cu_list,
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=self.gate_lower_bound,
                layout="BSND",
            )
        else:
            gate_cumsum = torch.ops._C_ascend.kda_gate_cumsum(
                self._activate_gate(raw_gate).contiguous(),
                KDA_CHUNK_SIZE,
                cu_seqlens=cu_list,
                layout="BSND",
            )

        result = torch.ops._C_ascend.chunk_kda_fwd(
            q,
            k,
            v.contiguous(),
            gate_cumsum,
            beta.contiguous(),
            self.scale,
            KDA_CHUNK_SIZE,
            layout="BSND",
            initial_state=initial_state_kv,
            output_final_state=True,
            cu_seqlens=cu_list,
            chunk_indices=_build_chunk_indices(cu_list, KDA_CHUNK_SIZE),
            return_intermediate=False,
        )
        final_state = result[1].transpose(-1, -2).contiguous()
        recurrent_state.index_copy_(0, slots, final_state.to(recurrent_state.dtype))
        return result[0]

    def _activate_gate(self, raw_gate: torch.Tensor) -> torch.Tensor:
        """Unbounded decay gate: -exp(A_log) * softplus(raw + dt_bias)."""
        x = raw_gate.float() + self.dt_bias.view(1, 1, self.local_num_heads, -1)
        a = self.A_log.exp().view(1, 1, self.local_num_heads, 1)
        return -a * F.softplus(x, beta=1.0, threshold=_SOFTPLUS_THRESHOLD)

    def _gated_rms_norm(
        self, x: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        x32 = x.float()
        y = (
            x32
            * torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + self.o_norm_eps)
            * self.o_norm_weight.float()
        )
        return (y * torch.sigmoid(gate.float())).to(x.dtype)
