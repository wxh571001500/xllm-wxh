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

"""JoyaiASR speech-to-text model (Python model executor target).

kaldi-fbank (C++ JoyaiASRAudioProcessor) -> Conformer encoder (a direct
port of vLLM conformer_encoder.py) -> 2-frame-concat projector -> Qwen2
LLM. Checkpoint: encoder.* (funasr names), encoder_projector.*, llm.*,
optional ctc_lo.* (CTC head; needs a template that reserves a CTC pad
segment after each audio).
"""

from __future__ import annotations

import itertools
import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.logger import logger
from xllm.python.layers import ColumnParallelLinear
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader, load_causal_lm_weights
from xllm.python.models.base import PyModelBase
from xllm.python.models.qwen2 import Qwen2Config, Qwen2Model

# ---------------------------------------------------------------------------
# Conformer encoder (port of vLLM conformer_encoder.py)
# ---------------------------------------------------------------------------


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim: int, d_model: int, out_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((idim - 1) // 2 - 1) // 2
        self.out = nn.Linear(out_channels * subsample_idim, d_model)

        self.subsampling = 4
        left_context = right_context = 3  # both exclude the current frame
        self.context = left_context + 1 + right_context  # 7

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        x = self.conv(x)
        n, c, t, d = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(n, t, c * d))
        mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        input_lengths = mask[:, -1, :].sum(dim=-1)
        return x, input_lengths, mask


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe_positive = torch.zeros(max_len, d_model, requires_grad=False)
        pe_negative = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        # Non-persistent: never loaded from the checkpoint (vLLM skip list).
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Read-only view; no per-encode [1, 2T, d_model] clone.
        tmax, t = self.pe.size(1), x.size(1)
        return self.pe[:, tmax // 2 - t + 1 : tmax // 2 + t].detach()


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.linear_expand = nn.Linear(d_model, d_model * 4)
        self.nonlinear = nn.SiLU()
        self.linear_project = nn.Linear(d_model * 4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_layer_norm(x)
        x = self.linear_expand(x)
        x = self.nonlinear(x)
        x = self.linear_project(x)
        return x + residual


class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)

    def forward_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_output(
        self,
        output: torch.Tensor,
        residual: torch.Tensor,
        sz_b: int,
        len_q: int,
    ) -> torch.Tensor:
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        fc_out = self.fc(output)
        return fc_out + residual

    def forward_attention(
        self,
        attn: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.eq(0)
            # In-place fills are safe (executor runs under no_grad) and
            # save two [batch, head, T, T] temporaries per layer.
            attn.masked_fill_(mask, -float("inf"))
            attn = torch.softmax(attn, dim=-1)
            attn.masked_fill_(mask, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn


class RelPosMultiHeadAttention(EncoderMultiHeadAttention):
    def __init__(self, n_head: int, d_model: int):
        super().__init__(n_head, d_model)
        d_k = d_model // n_head
        self.scale = 1.0 / (d_k**0.5)
        self.linear_pos = nn.Linear(d_model, n_head * d_k, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty([n_head, d_k]))
        self.pos_bias_v = nn.Parameter(torch.empty([n_head, d_k]))

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        n, h, t1, t2 = x.size()
        zero_pad = torch.zeros((n, h, t1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(n, h, t2 + 1, t1)
        x = x_padded[:, :, 1:].view_as(x)
        x = x[:, :, :, : x.size(-1) // 2 + 1]
        return x

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sz_b, len_q = q.size(0), q.size(1)
        residual = q
        q, k, v = self.forward_qkv(q, k, v)

        q = q.transpose(1, 2)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.n_head, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self._rel_shift(matrix_bd)

        attn_scores = matrix_ac + matrix_bd
        attn_scores.mul_(self.scale)

        output, attn = self.forward_attention(attn_scores, v, mask=mask)
        output = self.forward_output(output, residual, sz_b, len_q)
        return output, attn


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 33):
        super().__init__()
        assert kernel_size % 2 == 1
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 4, kernel_size=1, bias=False)
        self.padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model * 2,
            d_model * 2,
            kernel_size,
            stride=1,
            padding=self.padding,
            groups=d_model * 2,
            bias=False,
        )
        self.batch_norm = nn.LayerNorm(d_model * 2)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model * 2, d_model, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        out = self.pre_layer_norm(x)
        out = out.transpose(1, 2)
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        out = self.pointwise_conv1(out)
        out = F.glu(out, dim=1)
        out = self.depthwise_conv(out)
        out = out.transpose(1, 2)
        out = self.swish(self.batch_norm(out))
        out = out.transpose(1, 2)
        out = self.pointwise_conv2(out)
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        out = out.transpose(1, 2)
        return out + residual


class RelPosEmbConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, kernel_size: int = 33):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model)
        self.conv = ConformerConvolution(d_model, kernel_size)
        self.ffn2 = ConformerFeedForward(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        slf_attn_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = 0.5 * x + 0.5 * self.ffn1(x)
        out = self.mhsa(out, out, out, pos_emb, mask=slf_attn_mask)[0]
        out = self.conv(out, pad_mask)
        out = 0.5 * out + 0.5 * self.ffn2(out)
        out = self.layer_norm(out)
        return out


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        idim: int,
        n_layers_enc: int,
        n_head: int,
        d_model: int,
        kernel_size: int = 33,
        pe_maxlen: int = 5000,
    ):
        super().__init__()
        self.odim = d_model

        self.input_preprocessor = Conv2dSubsampling(idim, d_model)
        self.positional_encoding = RelPositionalEncoding(d_model, max_len=pe_maxlen)

        self.layer_stack = nn.ModuleList(
            [RelPosEmbConformerBlock(d_model, n_head, kernel_size) for _ in range(n_layers_enc)]
        )

    @staticmethod
    def padding_position_is_0(padded_input: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        n, t = padded_input.size()[:2]
        positions = torch.arange(t, device=padded_input.device).unsqueeze(0)
        mask = (positions < input_lengths.unsqueeze(1)).to(torch.uint8)
        return mask.unsqueeze(1)

    def forward(
        self,
        padded_input: torch.Tensor,
        input_lengths: torch.Tensor,
        pad: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pad:
            padded_input = F.pad(
                padded_input,
                (0, 0, 0, self.input_preprocessor.context - 1),
                "constant",
                0.0,
            )
        src_mask = self.padding_position_is_0(padded_input, input_lengths)

        embed_output, input_lengths, src_mask = self.input_preprocessor(padded_input, src_mask)
        enc_output = embed_output

        pos_emb = self.positional_encoding(embed_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, pos_emb, slf_attn_mask=src_mask, pad_mask=src_mask)

        return enc_output, input_lengths, src_mask


class JoyaiASRAdapter(nn.Module):
    """2-frame-concat projector: Linear(dim*2 -> llm) + ReLU + Linear."""

    def __init__(self, encoder_dim: int, llm_dim: int, downsample_rate: int = 2):
        super().__init__()
        self.ds = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.ds, feat_dim * self.ds)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        new_x_lens = torch.clamp(x_lens, max=seq_len) // self.ds
        return x, new_x_lens


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


def _batch_ctc_greedy_search(
    ctc_logits: torch.Tensor, speech_lengths: torch.Tensor, blank_id: int = 0
) -> list[list[int]]:
    """Batch CTC greedy decode -> token ids per item (vLLM-parity fold:
    collapse consecutive duplicates and drop blanks)."""
    lengths = speech_lengths.tolist()
    results: list[list[int]] = []
    for i in range(ctc_logits.size(0)):
        length = max(1, min(int(lengths[i]), ctc_logits.size(1)))
        hyp = ctc_logits[i, :length].max(dim=-1).indices.tolist()
        results.append([key for key, _ in itertools.groupby(hyp) if key != blank_id])
    return results


class JoyaiASRForConditionalGeneration(PyModelBase):
    """JoyaiASR top-level model: Conformer + projector + Qwen2 + lm_head.

    ``self.model`` (required by PyModelBase / the executor runner) is the
    :class:`Qwen2Model` LLM. Audio encoding and the embedding merge happen in
    :meth:`encode` / :meth:`get_input_embeddings`, driven by PyExecutorImpl.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        # Constructor-local tower config from the flat ModelArgs dict.
        encoder_cfg = {
            "idim": int(config.get("mm_audio_idim", 80)),
            "d_model": int(config.get("mm_audio_d_model", 1280)),
            "n_layers": int(config.get("mm_audio_n_layers", 16)),
            "n_head": int(config.get("mm_audio_n_head", 20)),
            "kernel_size": int(config.get("mm_audio_kernel_size", 33)),
            "pe_maxlen": int(config.get("mm_audio_pe_maxlen", 5000)),
            "downsample_rate": int(config.get("mm_audio_downsample_rate", 2)),
        }
        self.text_cfg = Qwen2Config.from_dict(config)

        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device

        # Also drives the chat-template variant.
        self.ctc_enable = bool(config.get("use_ctc", False))

        # Conformer tower (replicated across TP ranks).
        self.encoder = ConformerEncoder(
            encoder_cfg["idim"],
            encoder_cfg["n_layers"],
            encoder_cfg["n_head"],
            encoder_cfg["d_model"],
            encoder_cfg["kernel_size"],
            encoder_cfg["pe_maxlen"],
        ).to(dtype=dtype, device=device)
        self.encoder_projector = JoyaiASRAdapter(
            encoder_cfg["d_model"],
            self.text_cfg.hidden_size,
            encoder_cfg["downsample_rate"],
        ).to(dtype=dtype, device=device)

        # Optional CTC head, constructed at load time from the checkpoint's
        # ctc_lo shapes — config has no CTC vocab key and the LLM vocab_size
        # is not it (151647 vs 152064 for this family).
        self.ctc_lo: nn.Linear | None = None

        # LLM (self.model is required by PyModelBase / the executor runner).
        tp = self.text_cfg.tp_size
        assert self.text_cfg.vocab_size % tp == 0
        self.model = Qwen2Model(self.text_cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.text_cfg.hidden_size,
            self.text_cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

        # Per-audio encode cache keyed by content hash, byte-budgeted with
        # FIFO eviction, not CTC-gated (single-threaded executor). Raw tower
        # output + CTC candidates only; blocks assemble per request.
        self._encode_cache: dict[int, tuple[torch.Tensor, list[int]]] = {}
        self._encode_cache_bytes = 0
        self._encode_cache_budget_bytes = int(config.get("encode_cache_mb", 256)) * 1024 * 1024
        # Most-recent prefill's per-audio CTC hypotheses, in encode item
        # order; consumed by get_prompt_lookup_hints for speculative drafts.
        self._ctc_results: list[list[int]] = []

    # ------------------------------------------------------------------
    # Connection logic: Conformer -> LLM
    # ------------------------------------------------------------------
    def _cache_encode_entry(self, hash_key: int, embeds: torch.Tensor, hyp: list[int]) -> None:
        """Cache one audio's encode result, respecting the byte budget."""
        entry_bytes = embeds.element_size() * embeds.numel()
        while self._encode_cache_bytes + entry_bytes > self._encode_cache_budget_bytes and self._encode_cache:
            ev = self._encode_cache.pop(next(iter(self._encode_cache)))
            self._encode_cache_bytes -= ev[0].element_size() * ev[0].numel()
        if entry_bytes <= self._encode_cache_budget_bytes:
            # Release the old entry's bytes (get() must stay after the
            # eviction loop, or the same key double-counts).
            old = self._encode_cache.get(hash_key)
            if old is not None:
                self._encode_cache_bytes -= old[0].element_size() * old[0].numel()
            self._encode_cache[hash_key] = (embeds, hyp)
            self._encode_cache_bytes += entry_bytes

    def _ctc_block(self, hyp: list[int], pad_count: int) -> torch.Tensor:
        """[pad_count, hidden] CTC rows: embed_tokens(hyp) truncated to the
        reserved count, zero past its end (the pad id is also an EOS whose
        embedding leaks early-termination pressure); empty hyp -> zeros."""
        block = torch.zeros(
            pad_count,
            self.text_cfg.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        if hyp:
            take = hyp[:pad_count]
            ids = torch.tensor(take, dtype=torch.long, device=self.device)
            block[: len(take)] = self.model.embed_tokens(ids).to(self.dtype)
        return block

    def _assemble_blocks(
        self,
        entries: list[tuple[torch.Tensor, list[int]]],
        pad_counts: list[int],
    ) -> torch.Tensor:
        """One [audio; ctc] block per entry in item order; row count equals
        the entry's mm_token_num, matching the executor's mask-prefix-sum
        slicing."""
        parts: list[torch.Tensor] = []
        for i, (embeds, hyp) in enumerate(entries):
            pad_count = pad_counts[i] if i < len(pad_counts) else 0
            if pad_count > 0:
                parts.append(torch.cat([embeds, self._ctc_block(hyp, pad_count)], dim=0))
            else:
                parts.append(embeds)
        return torch.cat(parts, dim=0)

    def encode(
        self,
        input_features: torch.Tensor,
        speech_lengths: torch.Tensor,
        audio_meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage 1: fbank -> Conformer -> projector -> per-entry blocks.

        audio_meta: [hash, ctc_pad_num] per audio. Cache hits skip the
        tower (identical audios encode once); blocks assemble in item order
        with this call's pad counts, so downstream slicing and scattering
        are cache-state-free.
        """
        hash_keys: list[int] = []
        pad_counts: list[int] = []
        if audio_meta is not None:
            assert audio_meta.dim() == 2 and audio_meta.size(1) >= 2, (
                f"audio_meta must be [num_audios, 2] (hash, ctc_pad_num), got shape {tuple(audio_meta.shape)}"
            )
            hash_keys = [int(h) for h in audio_meta[:, 0].tolist()]
            pad_counts = [int(n) for n in audio_meta[:, 1].tolist()]
        cached: list[tuple[torch.Tensor, list[int]] | None] = [self._encode_cache.get(h) for h in hash_keys]
        if hash_keys and all(e is not None for e in cached):
            self._ctc_results = [list(e[1]) for e in cached]
            return self._assemble_blocks([(e[0], e[1]) for e in cached], pad_counts)

        # Host-side split; only the miss batch moves to the device below.
        lengths = speech_lengths.long().tolist()
        items = list(torch.split(input_features, lengths, dim=0))

        # Encode only the uncached items (all of them when no cache).
        if cached:
            miss_indices = [i for i, e in enumerate(cached) if e is None]
        else:
            miss_indices = list(range(len(items)))

        # Dedup identical audios among the misses; results fan out below.
        if hash_keys:
            slot_by_hash: dict[int, int] = {}
            encode_slots: list[int] = []  # unique slot -> first miss position
            slot_of_miss: list[int] = []  # miss position -> unique slot
            for pos, i in enumerate(miss_indices):
                h = hash_keys[i]
                slot = slot_by_hash.setdefault(h, len(encode_slots))
                if slot == len(encode_slots):
                    encode_slots.append(pos)
                slot_of_miss.append(slot)
        else:
            encode_slots = list(range(len(miss_indices)))
            slot_of_miss = list(range(len(miss_indices)))

        miss_items = [items[miss_indices[j]] for j in encode_slots]
        miss_lengths = [lengths[miss_indices[j]] for j in encode_slots]
        # Pad on the host, then one fused cast+H2D for the miss batch only.
        miss_batched = nn.utils.rnn.pad_sequence(miss_items, batch_first=True).to(dtype=self.dtype, device=self.device)
        miss_lens = torch.tensor(miss_lengths, dtype=torch.long, device=self.device)

        enc_output, enc_lengths, _ = self.encoder(miss_batched, miss_lens)

        if self.ctc_lo is not None:
            with torch.inference_mode():
                ctc_logits = self.ctc_lo(enc_output)
            miss_ctc = _batch_ctc_greedy_search(ctc_logits, enc_lengths)
        else:
            miss_ctc = []

        speech_features, speech_lens = self.encoder_projector(enc_output, enc_lengths)
        # Fan the per-slot results back out to miss positions.
        row_counts = speech_lens.tolist()
        miss_row_counts = [row_counts[s] for s in slot_of_miss]
        if miss_ctc:
            miss_ctc = [miss_ctc[s] for s in slot_of_miss]
        miss_embeds = [speech_features[slot_of_miss[j], : int(n), :] for j, n in enumerate(miss_row_counts)]

        # Unique slots store each distinct audio exactly once.
        if hash_keys:
            for j in encode_slots:
                i = miss_indices[j]
                h = hash_keys[i]
                n = int(miss_row_counts[j]) if j < len(miss_row_counts) else 0
                if n > 0:
                    # clone(): .contiguous() is a no-op view here and
                    # would pin the whole padded batch buffer.
                    item_embeds = miss_embeds[j].to(self.dtype).clone()
                    hyp = miss_ctc[j] if j < len(miss_ctc) else []
                    self._cache_encode_entry(h, item_embeds, hyp)

        # Assemble in original batch order.
        entries: list[tuple[torch.Tensor, list[int]]] = []
        miss_pos = 0
        for i in range(len(items)):
            if cached and i < len(cached) and cached[i] is not None:
                entries.append((cached[i][0], cached[i][1]))
            else:
                if miss_pos < len(miss_embeds):
                    entries.append(
                        (
                            miss_embeds[miss_pos].to(self.dtype),
                            miss_ctc[miss_pos] if miss_pos < len(miss_ctc) else [],
                        )
                    )
                miss_pos += 1
        self._ctc_results = [list(hyp) for _, hyp in entries]
        return self._assemble_blocks(entries, pad_counts)

    def get_prompt_lookup_hints(self) -> list[list[int]]:
        """Return current audio CTC proposals in encode item order.

        Proposals never bypass target-model verification. A request with no
        CTC hypothesis simply receives no useful draft.
        """
        return [list(tokens) for tokens in self._ctc_results]

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        image_embeds: torch.Tensor | None = None,
        video_embeds: torch.Tensor | None = None,
        audio_embeds: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage 2: merge text + audio embeddings.

        Audio rows scatter at the executor's chunk mask positions — not
        token-id matching (pad ids also occur as ordinary text). Sets
        ``model._inputs_embeds`` for the runner-driven forward.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)

        if audio_embeds is None:
            # Text-only step: clear so the runner falls back to embed_tokens.
            self.model._inputs_embeds = None
            return inputs_embeds

        assert audio_mask is not None, "audio_embeds requires the executor's chunk scatter mask"
        # Single host sync guarding the mask/embeds row invariant.
        num_mask_rows = int(audio_mask.sum().item())
        if num_mask_rows == 0:
            # Gap-text-only window: nothing to scatter; hand the runner the
            # embed_tokens rows instead of making it recompute them.
            self.model._inputs_embeds = inputs_embeds
            return inputs_embeds
        assert num_mask_rows == audio_embeds.size(0), (
            f"audio scatter mask rows ({num_mask_rows}) != audio embeds rows ({audio_embeds.size(0)})"
        )
        inputs_embeds[audio_mask] = audio_embeds.to(inputs_embeds.dtype)
        self.model._inputs_embeds = inputs_embeds
        return inputs_embeds

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(
        self,
        state_dicts: list,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        self._load_encoder_weights(state_dicts)
        if self.ctc_enable:
            self._load_ctc_weights(state_dicts)
        # LLM subtree via the shared causal-LM loader.
        ctx = ParallelLoadContext(tp_rank, tp_size)
        llm_weights = ScopedWeightLoader(state_dicts, src_prefixes=("llm.model.", "llm.", "model.", ""))
        load_causal_lm_weights(
            self.model,
            self.lm_head.weight,
            llm_weights,
            ctx,
            tie_word_embeddings=self.text_cfg.tie_word_embeddings,
        )

    def _find(self, state_dicts: list, name: str):
        for sd in state_dicts:
            if sd.has(name):
                return sd.get_tensor(name)
        return None

    def _load_replicated(self, module: nn.Module, state_dicts: list, key_fn: Callable[[str], str]) -> None:
        """Load every param of ``module`` from a replicated (non-TP) checkpoint
        namespace. ``key_fn(param_name)`` maps a module parameter name to its
        checkpoint key. ``copy_`` streams the H2D transfer and dtype cast in
        one shot, so no per-param ``.to(...)`` temporary is materialized
        (same pattern as :class:`WeightLoader.copy_in`)."""
        for param_name, param in module.named_parameters():
            key = key_fn(param_name)
            tensor = self._find(state_dicts, key)
            assert tensor is not None, f"checkpoint tensor not found: {key}"
            param.data.copy_(tensor)

    def _load_encoder_weights(self, state_dicts: list) -> None:
        """Load ``encoder.*`` + ``encoder_projector.*`` (replicated, no TP)."""

        # funasr Sequential names -> semantic module names (vLLM WeightsMapper).
        def ckpt_key(name: str) -> str:
            name = name.replace("ffn1.pre_layer_norm", "ffn1.net.0")
            name = name.replace("ffn1.linear_expand", "ffn1.net.1")
            name = name.replace("ffn1.linear_project", "ffn1.net.4")
            name = name.replace("ffn2.pre_layer_norm", "ffn2.net.0")
            name = name.replace("ffn2.linear_expand", "ffn2.net.1")
            name = name.replace("ffn2.linear_project", "ffn2.net.4")
            return "encoder." + name

        self._load_replicated(self.encoder, state_dicts, ckpt_key)
        self._load_replicated(self.encoder_projector, state_dicts, lambda n: "encoder_projector." + n)

    def _load_ctc_weights(self, state_dicts: list) -> None:
        """Construct + load the optional CTC head from ``ctc_lo.*``.

        Both dimensions come from the checkpoint: config.json has no CTC
        vocab key, and the LLM vocab_size is not the CTC vocab — the head
        ships a trimmed vocabulary (151647 vs 152064 for this family), so
        the shapes are derived where they live instead of being hardcoded.
        """
        weight = self._find(state_dicts, "ctc_lo.weight")
        assert weight is not None, (
            "--use_ctc is on but the checkpoint has no ctc_lo.weight "
            "(drop --use_ctc, or convert ctc_lo.pt into the safetensors "
            "shards first)"
        )
        if self.ctc_lo is None:
            self.ctc_lo = nn.Linear(
                weight.size(1),  # input: encoder d_model
                weight.size(0),  # output: CTC vocab (checkpoint-defined)
                bias=True,
            ).to(dtype=self.dtype, device=self.device)
            # Hypotheses are embedded via embed_tokens; ids must be valid.
            assert weight.size(0) <= self.text_cfg.vocab_size, (
                f"ctc vocab ({weight.size(0)}) exceeds the llm vocab "
                f"({self.text_cfg.vocab_size}): ctc hypotheses would not be "
                "embeddable by embed_tokens"
            )
        self._load_replicated(self.ctc_lo, state_dicts, lambda n: "ctc_lo." + n)
