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

"""Qwen3-style DFlash2 draft model for the Python NPU executor.

DFlash2 extends the DFlash block-diffusion draft with two additions carried in
the checkpoint: per-layer two-tap dynamic convolutions that keep the block from
decaying toward its tail, and a candidate selector that scores edges between the
top-k proposals at adjacent positions so the worker can trace one coherent path.
The context-KV precompute, the fc/hidden_norm projection of target aux-hidden
states, and the QuaRot adaptation are inherited unchanged from DFlash.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader
from xllm.python.models.qwen3 import Qwen3DecoderLayer
from xllm.python.models.qwen3_dflash import (
    DFlashQwen3Config,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)


@dataclass
class DFlash2Qwen3Config(DFlashQwen3Config):
    dflash2_block_size: int = 0
    dflash2_conv_group_size: int = 0
    dflash2_conv_kernel_size: int = 0
    dflash2_selector_rank: int = 0
    dflash2_selector_top_k: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> DFlash2Qwen3Config:
        base = DFlashQwen3Config.from_dict(d)
        return cls(
            **base.__dict__,
            dflash2_block_size=int(d.get("dflash2_block_size", 0)),
            dflash2_conv_group_size=int(d.get("dflash2_conv_group_size", 0)),
            dflash2_conv_kernel_size=int(d.get("dflash2_conv_kernel_size", 0)),
            dflash2_selector_rank=int(d.get("dflash2_selector_rank", 0)),
            dflash2_selector_top_k=int(d.get("dflash2_selector_top_k", 0)),
        )

    def validate(self) -> None:
        super().validate()
        if self.sliding_window <= 0:
            raise ValueError("DFlash2 requires sliding_window > 0")
        if self.dflash2_block_size <= 0:
            raise ValueError("DFlash2 requires dflash2_block_size > 0")
        if self.dflash2_selector_rank <= 0:
            raise ValueError("DFlash2 requires dflash2_selector_rank > 0")
        if self.dflash2_selector_top_k <= 0:
            raise ValueError("DFlash2 requires dflash2_selector_top_k > 0")
        if self.dflash2_selector_top_k > self.vocab_size:
            raise ValueError(
                "dflash2_selector_top_k must not exceed vocab_size "
                f"(got top_k={self.dflash2_selector_top_k}, vocab_size={self.vocab_size})"
            )
        if self.dflash2_conv_kernel_size <= 0:
            raise ValueError("DFlash2 requires dflash2_conv_kernel_size > 0")
        if self.dflash2_conv_kernel_size > self.dflash2_block_size:
            raise ValueError(
                "dflash2_conv_kernel_size must not exceed dflash2_block_size "
                f"(got taps={self.dflash2_conv_kernel_size}, block_size={self.dflash2_block_size})"
            )
        if self.dflash2_conv_group_size <= 0 or self.hidden_size % self.dflash2_conv_group_size:
            raise ValueError("dflash2_conv_group_size must divide hidden_size")


class DFlash2GroupedConv(nn.Module):
    """N-tap dynamic depthwise-grouped convolution over one proposal block.

    ``prepare`` runs before the wrapped sublayer (attention or MLP) and returns
    the pre-convolved input plus the post coefficients; ``finish`` runs after it.
    Coefficients are per-token, so the kernel adapts to content while the
    ``base_kernel`` supplies a static prior. The convolution never crosses a
    ``block_size`` boundary, matching the block-diffusion draft layout. ``taps``
    (= ``dflash2_conv_kernel_size``) is fully general; DFlash2 currently ships
    with ``taps == 2`` but the math and validate() cover any ``taps in [1,
    block_size]``.
    """

    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.taps = taps
        self.group_size = group_size
        self.block_size = block_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, taps, hidden_size, dtype=dtype, device=device))
        self.kernel_projection = nn.Linear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def prepare(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden).view(hidden.size(0), 2, self.taps, self.num_groups)
        pre = self._convolve(hidden, coefficients[:, 0], self.base_kernel[0])
        return pre, coefficients[:, 1]

    def finish(self, hidden: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        return self._convolve(hidden, coefficients, self.base_kernel[1])

    def _convolve(
        self,
        hidden: torch.Tensor,
        delta: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden.size(0)
        if num_tokens % self.block_size:
            raise ValueError("DFlash2 convolution rows must contain complete blocks")
        num_blocks = num_tokens // self.block_size
        blocks = hidden.view(num_blocks, self.block_size, self.num_groups, self.group_size)
        coefficients = base.view(1, self.taps, self.num_groups, self.group_size) + delta.unsqueeze(-1)
        coefficients = coefficients.view(num_blocks, self.block_size, self.taps, self.num_groups, self.group_size)
        output = coefficients[:, :, 0] * blocks
        for tap in range(1, self.taps):
            output[:, tap:] += coefficients[:, tap:, tap] * blocks[:, : self.block_size - tap]
        return output.reshape(num_tokens, self.num_groups * self.group_size)


class DFlash2CandidateSelector(nn.Module):
    """Scores edges between adjacent-position top-k proposals.

    For each proposal position it keeps the ``selector_top_k`` highest unary
    logits and produces ``edge_logits[p, c] = unary(c) + <predecessor(p), proj(h),
    successor(c)>`` so the worker can pick one coherent path through the block.
    All tensors are TP-replicated: the edges are identical on every rank.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False, dtype=dtype, device=device)
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank, dtype=dtype, device=device))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank, dtype=dtype, device=device))

    def forward(
        self,
        hidden_states: torch.Tensor,
        unary_logits: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.dim() != 3 or unary_logits.dim() != 3:
            raise ValueError("DFlash2 selector expects [batch, steps, *] inputs")
        if unary_logits.size(-1) != self.successor_codebook.size(0):
            raise ValueError("DFlash2 selector unary logits width does not match vocab_size")

        values, candidate_ids = torch.topk(unary_logits, self.top_k, dim=-1)
        values = values.to(torch.float32)

        # Edge score = unary(c) + <predecessor(p), proj(h), successor(c)>. The
        # trilinear term drives argmax path selection, so keep it fp32-consistent
        # with the fp32 ``values`` above — bf16 einsum accumulation can perturb
        # ordering enough to flip predecessor selection at the block boundary.
        hidden = self.hidden_projection(hidden_states).to(torch.float32)
        successors = F.embedding(candidate_ids, self.successor_codebook).to(torch.float32)
        anchor = anchor_token_ids.view(-1, 1, 1).expand(-1, 1, self.top_k)
        predecessor_ids = torch.cat((anchor, candidate_ids[:, :-1, :]), dim=1)
        predecessors = F.embedding(predecessor_ids, self.predecessor_codebook).to(torch.float32)
        pair_scores = torch.einsum(
            "blpr,blcr->blpc",
            predecessors.mul_(hidden.unsqueeze(2)),
            successors,
        )
        edge_logits = pair_scores.add_(values.unsqueeze(-2))
        return candidate_ids, edge_logits


class DFlash2DecoderLayer(Qwen3DecoderLayer):
    """Qwen3 dense decoder layer wrapped by two-tap block convolutions.

    The convolutions bracket both the attention and the MLP. Attention is
    non-causal (``causal=False``) so each query attends the whole proposal
    block, which is denoised jointly.
    """

    def __init__(
        self,
        cfg: DFlash2Qwen3Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool = False,
    ) -> None:
        super().__init__(
            cfg,
            layer_id,
            dtype,
            device,
            causal=causal,
        )
        self.self_attn.attn.attention_window = (
            cfg.sliding_window - 1,
            cfg.dflash2_block_size - 1,
        )

        def _make_conv() -> DFlash2GroupedConv:
            return DFlash2GroupedConv(
                cfg.hidden_size,
                cfg.dflash2_conv_kernel_size,
                cfg.dflash2_conv_group_size,
                cfg.dflash2_block_size,
                dtype,
                device,
            )

        self.attention_conv = _make_conv()
        self.mlp_conv = _make_conv()

    def _attn_block(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mrope_section: list[int] | None,
    ) -> torch.Tensor:
        hidden, coefficients = self.attention_conv.prepare(hidden)
        hidden = self.self_attn(positions, hidden, cos_sin_cache, cos, sin, mrope_section)
        return self.attention_conv.finish(hidden, coefficients)

    def _mlp_block(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden, coefficients = self.mlp_conv.prepare(hidden)
        hidden = self.mlp(hidden)
        return self.mlp_conv.finish(hidden, coefficients)

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        super().load_weights(weights, context)
        # Replicated two-tap conv weights; loaded from the layer-scoped root.
        for conv in ("attention_conv", "mlp_conv"):
            module = getattr(self, conv)
            weights.load_tensor(module.base_kernel, f"{conv}.base_kernel")
            weights.load_tensor(module.kernel_projection.weight, f"{conv}.kernel_projection.weight")


class DFlash2Qwen3Model(DFlashQwen3Model):
    def __init__(self, cfg: DFlash2Qwen3Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(cfg, dtype, device, decoder_layer_cls=DFlash2DecoderLayer)
        self.candidate_selector = DFlash2CandidateSelector(
            cfg.hidden_size,
            cfg.vocab_size,
            cfg.dflash2_selector_rank,
            cfg.dflash2_selector_top_k,
            dtype,
            device,
        )

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> ScopedWeightLoader:
        all_weights = super().load_weights(state_dicts, tp_rank, tp_size)
        # Replicated selector codebooks + projection; loaded from the model root.
        selector = self.candidate_selector
        all_weights.load_tensor(selector.hidden_projection.weight, "candidate_selector.hidden_projection.weight")
        all_weights.load_tensor(selector.predecessor_codebook, "candidate_selector.predecessor_codebook")
        all_weights.load_tensor(selector.successor_codebook, "candidate_selector.successor_codebook")
        return all_weights


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    config_cls = DFlash2Qwen3Config
    model_cls = DFlash2Qwen3Model

    def dflash2_candidates(
        self,
        hidden_states: torch.Tensor,
        unary_logits: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = cast(DFlash2Qwen3Model, self.model)
        return model.candidate_selector(hidden_states, unary_logits, anchor_token_ids)
