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
- ``recurrent_state``: ``[num_slots, local_num_heads, head_dim, head_dim]``,
  float32; per-slot layout is ``[H, V, K]``. The ``chunk_kda_fwd`` operator
  uses the transposed ``[H, K, V]`` layout, so the initial/final states are
  transposed at that operator boundary only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear

KDA_CHUNK_SIZE = 64
# Padding slot convention of the AscendC causal-conv operator.
PAD_SLOT_ID = -1
_L2NORM_EPS = 1e-6
_SOFTPLUS_THRESHOLD = 20.0


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


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt((x32 * x32).sum(-1, keepdim=True) + _L2NORM_EPS)).to(
        x.dtype
    )


def _build_chunk_indices(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    """Flat [seq_idx, chunk_idx] pairs, mirroring build_kda_chunk_indices."""
    indices: list[int] = []
    for seq in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq + 1] - cu_seqlens[seq]
        for chunk in range(-(-seq_len // chunk_size)):
            indices.extend((seq, chunk))
    return indices


class KimiK3DeltaAttention(nn.Module):
    """Kimi-K3 KDA layer with a full-rank per-channel decay gate.

    Args:
        hidden_size: model hidden size.
        linear_attn_config: the HF config's ``linear_attn_config`` dict; must
            contain ``num_heads``, ``head_dim``, ``short_conv_kernel_size`` and
            ``use_full_rank_gate=True``. ``gate_lower_bound`` is optional.
        tp_size / tp_rank: tensor-parallel world size and rank. Heads (and the
            tied projections / conv / biases) are head-sharded per rank; the
            low-rank ``f_a`` projection is replicated on every rank.
        rms_norm_eps: epsilon of the gated output RMSNorm (the model config's
            ``rms_norm_eps``).
        dtype / device: parameter dtype and device.
    """

    def __init__(
        self,
        hidden_size: int,
        linear_attn_config: dict,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        assert linear_attn_config.get("use_full_rank_gate", False), (
            "KimiK3DeltaAttention requires a full-rank gate"
        )
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.head_dim = linear_attn_config["head_dim"]
        self.num_heads = linear_attn_config["num_heads"]
        assert self.num_heads % tp_size == 0
        self.local_num_heads = self.num_heads // tp_size
        self.projection_size = self.head_dim * self.num_heads
        self.local_projection_size = self.projection_size // tp_size
        self.conv_size = linear_attn_config["short_conv_kernel_size"]
        self.gate_lower_bound: float | None = linear_attn_config.get(
            "gate_lower_bound", None
        )
        if self.gate_lower_bound is not None:
            assert -5.0 <= self.gate_lower_bound < 0, (
                "KDA gate lower bound must be in [-5, 0), "
                f"got {self.gate_lower_bound}."
            )
        self.scale = self.head_dim**-0.5
        self.o_norm_eps = rms_norm_eps

        local_proj = self.local_projection_size
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
            local_proj, hidden_size, tp_size, bias=False, dtype=dtype, device=device
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
        for name in ("q_proj", "k_proj", "v_proj", "g_proj", "b_proj", "f_b_proj"):
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
        num_decode_tokens = num_decode  # decode requests carry 1 token each

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
            x.reshape(1, num_tokens, self.local_num_heads, self.head_dim)
            for x in mixed_qkv.chunk(3, dim=-1)
        )

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
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
    ) -> torch.Tensor:
        out = torch.empty_like(mixed_qkv)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            out,
            mixed_qkv,
            self.conv_weight_t,
            conv_state=conv_state,
            bias_opt=None,
            query_start_loc_opt=query_start_loc,
            cache_indices_opt=cache_indices.to(torch.int32),
            initial_state_mode_opt=initial_state_mode,
            num_accepted_tokens_opt=None,
            activation_mode=1,  # silu
            pad_slot_id=PAD_SLOT_ID,
            run_mode=run_mode,
        )
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
            num_accepted_tokens=None,
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
    ) -> torch.Tensor:
        num_decode = metadata.num_decode_seqs
        prefill_cu = metadata.query_start_loc[num_decode:]
        # The AscendC prefill operators take host-side cumulative lengths.
        cu_list = (prefill_cu - prefill_cu[0]).tolist()
        cu_list = [int(x) for x in cu_list]
        slots = metadata.state_indices.long()[num_decode:]
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
