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

"""Legacy Qwen3/GQA DSpark draft model.

Kimi K3 uses a separate MLA draft implementation.  Older DSpark checkpoints
identify their draft as ``DSparkDraftModel`` and contain ordinary Qwen3
``q_proj/k_proj/v_proj`` weights.  Keeping this implementation separate avoids
interpreting those weights as Kimi's fused MLA projections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch_npu


def _tensor_checksum(tensor, name, layer, call_idx):
    if tensor is None or not _debug_enabled():
        return
    torch.npu.synchronize()
    # Read full tensor via small slices to avoid FRACTAL_NZ issues
    t = tensor.detach()
    if t.device.type in ("npu", "privateuseone"):
        import torch_npu

        try:
            fmt = int(torch_npu.get_npu_format(t))
            if fmt != 2:
                t = torch_npu.npu_format_cast(t, 2)
        except Exception:
            pass
        t = t.cpu().contiguous()
    s = t.float()
    print(
        f"CSUM xl {name} L{layer} c{call_idx} shape={list(s.shape)} sum={s.sum().item():.6f} mean={s.mean().item():.6f} std={s.std().item():.6f} min={s.min().item():.6f} max={s.max().item():.6f} first5={s.flatten()[:5].tolist()}",
        flush=True,
    )


_xllm_save_counter = [0]


def _debug_enabled():
    import os

    return bool(os.getenv("XLLM_DSPARK_ACCURACY_DUMP_DIR"))


def _save_full_tensor_xllm(tensor, name, dump_dir):
    if tensor is None:
        return
    if tensor.device.type in ("npu", "privateuseone"):
        torch.npu.synchronize()
    if tensor.numel() < 5000000:
        vals = tensor.detach().cpu().reshape(-1).tolist()
        cpu_t = torch.tensor(vals, dtype=torch.float32).reshape(tensor.shape)
    else:
        cpu_t = tensor.detach().cpu().contiguous().clone()
    os.makedirs(dump_dir, exist_ok=True)
    call_idx = _xllm_save_counter[0]
    fpath = os.path.join(dump_dir, f"xlhook_{name}_call{call_idx:04d}.pt")
    torch.save(cpu_t, fpath)


import torch.nn as nn
import torch.nn.functional as F

from xllm.python import ops
from xllm.python.layers import GatedMLP, RMSNorm, RotaryEmbedding
from xllm.python.model_executor.forward_context import record_layer_event
from xllm.python.models.base import PyModelBase
from xllm.python.models.dspark_accuracy import dump_dspark_tensors, snapshot_for_dump
from xllm.python.models.kimi_k3_text import (
    _layer_ids,
    _MergedStateDict,
    _state_dict_sharded_tensor,
    _state_dict_tensor,
)
from xllm.python.models.qwen3 import Qwen3Attention, Qwen3Config

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


@dataclass
class Qwen3DSparkConfig(Qwen3Config):
    target_hidden_size: int = 0
    num_target_layers: int = 0
    markov_rank: int = 256
    mask_token_id: int = -1

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Qwen3DSparkConfig:
        base = Qwen3Config.from_dict(config)

        # The C++ config bridge strips nested dicts (e.g. rope_parameters is
        # serialized as -1), so recover YaRN RoPE from the raw config or
        # fall back to the Kimi-K3-DSpark defaults.
        rope_raw = config.get("rope_parameters") or config.get("rope_scaling")
        if isinstance(rope_raw, dict) and rope_raw.get("rope_type", "default") == "yarn":
            base.rope_type = "yarn"
            base.rope_theta = float(rope_raw.get("rope_theta", base.rope_theta))
            base.rope_scaling_factor = float(rope_raw.get("factor", rope_raw.get("scaling_factor", 1.0)))
            base.rope_original_max_position_embeddings = int(rope_raw.get("original_max_position_embeddings", 0))
            base.rope_beta_fast = int(rope_raw.get("beta_fast", 32))
            base.rope_beta_slow = int(rope_raw.get("beta_slow", 1))
            base.rope_mscale = float(rope_raw.get("mscale", 1.0))
            base.rope_mscale_all_dim = float(rope_raw.get("mscale_all_dim", 0.0))
        elif not isinstance(rope_raw, dict):
            # C++ bridge stripped rope_parameters; apply Kimi-K3-DSpark YaRN defaults.
            base.rope_type = "yarn"
            # The C++ bridge fills rope_theta/rope_scaling_factor with its
            # own defaults (1e6 / 0.0), so ignore them and use the known
            # Kimi-K3-DSpark YaRN values from the checkpoint config.json.
            base.rope_theta = 10000.0
            base.rope_scaling_factor = 16.0
            base.rope_original_max_position_embeddings = 65536
            base.rope_beta_fast = 32
            base.rope_beta_slow = 1
            base.rope_mscale = 1.0
            base.rope_mscale_all_dim = 0.0

        # vLLM derives the local tensor-parallel width from the active
        # parallel world, rather than from a checkpoint field.  Older xLLM
        # draft bridges may leave ``tp_size`` at its default value even when
        # the process world is sharded, so recover the same effective width
        # before constructing QKV and MLP layers.
        world_size = int(config.get("world_size", 0) or 0)
        dp_size = max(int(config.get("dp_size", 1) or 1), 1)
        cp_size = max(int(config.get("cp_size", 1) or 1), 1)
        effective_tp_size = world_size // (dp_size * cp_size) if world_size else 0
        if effective_tp_size > 1 and base.tp_size <= 1:
            base.tp_size = effective_tp_size
            base.tp_rank = int(config.get("rank", base.tp_rank) or 0) % effective_tp_size

        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in config and config[key] is not None:
                    return config[key]
            return default

        dflash = config.get("dflash_config", {})
        if not isinstance(dflash, dict):
            dflash = {}
        target_ids = (
            dflash.get("target_layer_ids")
            or config.get("target_layer_ids")
            or config.get("dspark_target_layer_ids")
            or []
        )
        configured_layers = int(pick("dspark_num_target_layers", "num_target_layers", default=0) or 0)
        num_target_layers = len(target_ids) or configured_layers or base.n_layers
        configured_hidden = int(pick("dspark_target_hidden_size", "target_hidden_size", default=0) or 0)
        target_hidden_size = configured_hidden or base.hidden_size
        mask_token_id = int(
            dflash.get(
                "mask_token_id",
                pick("mask_token_id", "dspark_noise_token_id", default=-1),
            )
        )
        return cls(
            **base.__dict__,
            target_hidden_size=target_hidden_size,
            num_target_layers=num_target_layers,
            markov_rank=(int(pick("markov_rank", "dspark_markov_rank", default=0) or 0) or 256),
            mask_token_id=mask_token_id,
        )


def _copy_parameter(parameter: torch.Tensor, tensor: torch.Tensor, name: str) -> None:
    if parameter.shape != tensor.shape:
        raise ValueError(f"Qwen3 DSpark parameter {name} expects {tuple(parameter.shape)}, got {tuple(tensor.shape)}")
    parameter.data.copy_(tensor.to(dtype=parameter.dtype, device=parameter.device))


def _first_tensor(state_dict: Any, names: tuple[str, ...]) -> torch.Tensor | None:
    for name in names:
        tensor = _state_dict_tensor(state_dict, name)
        if tensor is not None:
            return tensor
    return None


def _first_sharded_tensor(
    state_dict: Any,
    names: tuple[str, ...],
    dim: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor | None:
    for name in names:
        tensor = _state_dict_sharded_tensor(state_dict, name, dim, tp_rank, tp_size)
        if tensor is not None:
            return tensor
    return None


def _rope_cos_sin(
    positions: torch.Tensor,
    rotary,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions.to(torch.int64).contiguous()
    cos_sin = rotary.cos_sin_cache[positions]
    half = cos_sin.shape[-1] // 2
    cos_half, sin_half = cos_sin[..., :half], cos_sin[..., half:]
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1).unsqueeze(1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1).unsqueeze(1)
    return cos, sin


class _RotaryAdapter:
    """Adapter to expose cos_sin_cache for the interleaved rope helper."""

    def __init__(self, cos_sin_cache: torch.Tensor) -> None:
        self.cos_sin_cache = cos_sin_cache


def _apply_interleaved_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    rotary,
) -> torch.Tensor:
    cos, sin = _rope_cos_sin(positions, rotary)
    if tensor.device.type in ("npu", "privateuseone"):
        tokens, heads, dim = tensor.shape
        return torch_npu.npu_interleave_rope(tensor.view(tokens, heads, 1, dim), cos, sin).view(tokens, heads, dim)
    cos_sin = rotary.cos_sin_cache[positions.to(torch.int64).contiguous()]
    half = cos_sin.shape[-1] // 2
    cos_half, sin_half = cos_sin[..., :half], cos_sin[..., half:]
    pairs = tensor.unflatten(-1, (-1, 2))
    even, odd = pairs.unbind(dim=-1)
    cos_half, sin_half = cos_half.unsqueeze(1), sin_half.unsqueeze(1)
    return torch.stack((even * cos_half - odd * sin_half, odd * cos_half + even * sin_half), dim=-1).flatten(-2)


class Qwen3DSparkAttention(Qwen3Attention):
    """Qwen3 attention with the non-causal runtime flag used by DSpark."""

    def __init__(self, cfg: Qwen3Config, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(cfg, layer_id, dtype, device)
        self.attn.non_causal_block = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mrope_section: list[int] | None = None,
        trace_tensors: dict[str, torch.Tensor | None] | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        global _xllm_save_counter
        if layer_index == 0:
            _xllm_save_counter[0] += 1
        if layer_index == 0 and trace_tensors is not None:
            w = self.qkv_proj.weight
            if _debug_enabled():
                print(
                    f"DSpark FWD: qkv_proj.weight ptr={w.data_ptr()} first5={snapshot_for_dump(w)[0, :5].tolist()}",
                    flush=True,
                )
        qkv = self.qkv_proj(hidden)
        num_tokens = qkv.size(0)
        q = qkv[:, : self.q_size]
        k = qkv[:, self.q_size : self.q_size + self.kv_size]
        v = qkv[:, self.q_size + self.kv_size :]

        if trace_tensors is not None:
            if layer_index == 0:
                w = snapshot_for_dump(self.qkv_proj.weight)
                qkv_s = snapshot_for_dump(qkv)
            trace_tensors[f"draft.layer.{layer_index}.qkv_raw"] = snapshot_for_dump(qkv)
            _tensor_checksum(qkv, "qkv_raw", layer_index, _xllm_save_counter[0])
            trace_tensors[f"draft.layer.{layer_index}.q_raw"] = snapshot_for_dump(q)
            trace_tensors[f"draft.layer.{layer_index}.k_raw"] = snapshot_for_dump(k)
            # Also dump the actual weight used
            trace_tensors[f"draft.layer.{layer_index}.qkv_weight"] = snapshot_for_dump(self.qkv_proj.weight)

        # Per-head RMSNorm (decomposed to match vLLM DFlash attention).
        q = (
            torch.ops.xllm_ops.rms_norm(
                q.reshape(num_tokens * self.num_heads, self.head_dim),
                self.q_norm.weight,
                self.q_norm.eps,
            )
            .view(num_tokens, self.q_size)
            .clone()
        )
        k = (
            torch.ops.xllm_ops.rms_norm(
                k.reshape(num_tokens * self.num_kv_heads, self.head_dim),
                self.k_norm.weight,
                self.k_norm.eps,
            )
            .view(num_tokens, self.kv_size)
            .clone()
        )

        if trace_tensors is not None:
            trace_tensors[f"draft.layer.{layer_index}.q_normed"] = snapshot_for_dump(q)
            trace_tensors[f"draft.layer.{layer_index}.k_normed"] = snapshot_for_dump(k)

        # Apply RoPE using torch_npu._npu_rotary_embedding to match vLLM
        q_rot = q.view(num_tokens, -1).contiguous()
        k_rot = k.view(num_tokens, -1).contiguous()
        torch_npu._npu_rotary_embedding(
            positions,
            q_rot,
            k_rot,
            self.head_dim,
            cos_sin_cache,
            True,
        )
        q = q_rot.view(num_tokens, self.q_size)
        k = k_rot.view(num_tokens, self.kv_size)

        if trace_tensors is not None:
            _q_snap = snapshot_for_dump(q)
            _k_snap = snapshot_for_dump(k)
            trace_tensors[f"draft.layer.{layer_index}.q_rope"] = _q_snap
            trace_tensors[f"draft.layer.{layer_index}.k_rope"] = _k_snap
            _tensor_checksum(q, "q_rope", layer_index, _xllm_save_counter[0])
            _tensor_checksum(k, "k_rope", layer_index, _xllm_save_counter[0])

        attn_out = self.attn(q, k, v)

        if trace_tensors is not None:
            _attn_snap = snapshot_for_dump(attn_out)
            trace_tensors[f"draft.layer.{layer_index}.attn_out"] = _attn_snap
            _tensor_checksum(attn_out, "attn_out", layer_index, _xllm_save_counter[0])

        out = self.o_proj(attn_out)

        if trace_tensors is not None:
            trace_tensors[f"draft.layer.{layer_index}.o_proj_out"] = snapshot_for_dump(out)
            _tensor_checksum(out, "o_proj_out", layer_index, _xllm_save_counter[0])

        return out


class Qwen3DSparkDecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3DSparkConfig, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = Qwen3DSparkAttention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.mlp = GatedMLP(cfg.hidden_size, cfg.intermediate_size, cfg.tp_size, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        trace_tensors: dict[str, torch.Tensor | None] | None = None,
        layer_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)

        if trace_tensors is not None:
            trace_tensors[f"draft.layer.{layer_index}.normed_input"] = snapshot_for_dump(hidden)
            _tensor_checksum(hidden, "normed_input", layer_index, _xllm_save_counter[0])

        hidden = self.self_attn(
            positions,
            hidden,
            cos_sin_cache,
            None,
            None,
            trace_tensors=trace_tensors,
            layer_index=layer_index,
        )
        hidden, residual = self.post_attention_layernorm(hidden, residual)

        if trace_tensors is not None:
            trace_tensors[f"draft.layer.{layer_index}.post_attn_normed"] = snapshot_for_dump(hidden)

        hidden = self.mlp(hidden)
        return hidden, residual


def _neox_rope(tensor: torch.Tensor, positions: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    """Apply the NEOX half-split rotation used by Qwen3's RoPE table.

    Compute in float32 to match vLLM's Triton RoPE kernel precision."""
    orig_dtype = tensor.dtype
    cos_sin = cache[positions.to(torch.int64).contiguous()]
    half = cos_sin.shape[-1] // 2
    cos = cos_sin[..., :half].unsqueeze(1).float()
    sin = cos_sin[..., half:].unsqueeze(1).float()
    first, second = tensor[..., :half].float(), tensor[..., half:].float()
    result = torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
    return result.to(orig_dtype)


def _match_cache_heads(
    tensor: torch.Tensor,
    cache: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Validate the vLLM/xLLM local GQA head layout before cache insertion."""
    if cache.ndim < 3:
        raise ValueError(f"Qwen3 DSpark {name} cache must be at least 3D, got {tuple(cache.shape)}")
    expected_heads = cache.shape[2]
    actual_heads = tensor.shape[1]
    if actual_heads == expected_heads:
        return tensor
    raise ValueError(
        f"Qwen3 DSpark {name} local head mismatch: produced {actual_heads}, "
        f"cache requires {expected_heads}; tensor={tuple(tensor.shape)}, "
        f"cache={tuple(cache.shape)}"
    )


class Qwen3DSparkModel(nn.Module):
    def __init__(self, cfg: Qwen3DSparkConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.config = cfg
        self.cfg = cfg
        # Target weights are attached by share_weights_from after draft loading.
        self.embed_tokens: nn.Module | None = None
        self.context_proj = nn.Linear(
            cfg.target_hidden_size * cfg.num_target_layers,
            cfg.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.hidden_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        if cfg.rope_type == "yarn":
            print(
                f"DSpark: using YaRN RoPE, theta={cfg.rope_theta}, factor={cfg.rope_scaling_factor}, orig_max={cfg.rope_original_max_position_embeddings}",
                flush=True,
            )
            from xllm.python.models.deepseek_v32 import DeepseekYarnRotaryEmbedding

            self.rotary = DeepseekYarnRotaryEmbedding(
                cfg.head_dim,
                cfg.rope_original_max_position_embeddings or cfg.max_position_embeddings,
                cfg.rope_scaling_factor,
                cfg.rope_theta,
                cfg.rope_beta_fast,
                cfg.rope_beta_slow,
                cfg.rope_mscale,
                cfg.rope_mscale_all_dim,
                dtype=dtype,
                device=device,
                cache_max_position_embeddings=cfg.max_position_embeddings,
            )
        else:
            self.rotary = RotaryEmbedding(
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                dtype=dtype,
                device=device,
            )
        self.layers = nn.ModuleList([Qwen3DSparkDecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.markov_w1 = nn.Embedding(cfg.vocab_size, cfg.markov_rank, dtype=dtype, device=device)
        self.markov_w2 = nn.Linear(cfg.markov_rank, cfg.vocab_size, bias=False, dtype=dtype, device=device)
        self._fused_kv_weight: torch.Tensor | None = None
        self._fused_kv_bias: torch.Tensor | None = None
        self._k_norm_weights: torch.Tensor | None = None

    def rebuild_attention_layers(self) -> None:
        """Recreate layers after checkpoint geometry is inferred."""
        self.layers = nn.ModuleList(
            [
                Qwen3DSparkDecoderLayer(self.cfg, i, self.norm.weight.dtype, self.norm.weight.device)
                for i in range(self.cfg.n_layers)
            ]
        )
        self._fused_kv_weight = None
        self._fused_kv_bias = None
        self._k_norm_weights = None

    _inject_call_count = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Inject vLLM's input_ids for the first draft call to enable
        # like-for-like precision comparison. Only active when dump is enabled.
        if _debug_enabled():
            inject_path = "/export/home/wangxiaohan17/wangxiaohan/vllm_draft_input_0.pt"
            import os as _os

            if _os.path.isfile(inject_path) and Qwen3DSparkModel._inject_call_count == 0:
                Qwen3DSparkModel._inject_call_count += 1
                inj = torch.load(inject_path, weights_only=False)
                inj_ids = inj["input_ids"].to(device=input_ids.device, dtype=input_ids.dtype)
                inj_pos = inj["positions"].to(device=positions.device, dtype=positions.dtype)
                print(f"INJECT: overriding draft input_ids {input_ids.tolist()} -> {inj_ids.tolist()}", flush=True)
                input_ids = inj_ids
                positions = inj_pos
        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise RuntimeError("Qwen3 DSpark target embedding is not shared")
            inputs_embeds = self.embed_tokens(input_ids)
        hidden, residual = inputs_embeds, None
        trace_tensors: dict[str, torch.Tensor | None] = {
            "draft.input_ids": snapshot_for_dump(input_ids),
            "draft.positions": snapshot_for_dump(positions),
            "draft.inputs_embeds": snapshot_for_dump(inputs_embeds),
            "draft.cos_sin_cache": snapshot_for_dump(self.rotary.cos_sin_cache),
        }
        positions = positions.to(torch.int64).contiguous()
        for layer_index, layer in enumerate(self.layers):
            hidden, residual = layer(
                hidden,
                residual,
                positions,
                self.rotary.cos_sin_cache,
                trace_tensors,
                layer_index,
            )
            trace_tensors[f"draft.layer.{layer_index}.hidden"] = snapshot_for_dump(hidden)
            _tensor_checksum(hidden, "hidden", layer_index, _xllm_save_counter[0])
            trace_tensors[f"draft.layer.{layer_index}.residual"] = snapshot_for_dump(residual)
            _tensor_checksum(residual, "residual", layer_index, _xllm_save_counter[0])
            record_layer_event(layer.layer_id)
        hidden, _ = self.norm(hidden, residual)
        trace_tensors["draft.final_hidden"] = snapshot_for_dump(hidden)
        _tensor_checksum(hidden, "final_hidden", 0, _xllm_save_counter[0])
        dump_dspark_tensors("draft_forward", trace_tensors)
        return hidden

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError("Qwen3 DSpark target embedding is not shared")
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = self.context_proj.in_features
        if target_hidden.shape[-1] != expected:
            raise ValueError(
                f"Qwen3 DSpark context hidden size mismatch: expected {expected}, got {target_hidden.shape[-1]}"
            )
        projected = self.context_proj(target_hidden)
        dump_dspark_tensors(
            "context_projection",
            {
                "context.target_hidden": target_hidden,
                "context.projected_hidden": projected,
            },
        )
        return projected

    def _build_fused_kv_buffers(self) -> None:
        kv_weights: list[torch.Tensor] = []
        k_norm_weights: list[torch.Tensor] = []
        for layer in self.layers:
            attention = layer.self_attn
            qkv = attention.qkv_proj.weight
            kv_weights.append(qkv[attention.q_size :].contiguous())
            k_norm_weights.append(attention.k_norm.weight)
        self._fused_kv_weight = torch.cat(kv_weights, dim=0).contiguous()
        if self.layers[0].self_attn.qkv_proj.bias is not None:
            self._fused_kv_bias = torch.cat(
                [layer.self_attn.qkv_proj.bias[layer.self_attn.q_size :] for layer in self.layers], dim=0
            ).contiguous()
        self._k_norm_weights = torch.stack(k_norm_weights, dim=0).contiguous()

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if len(kv_caches) != len(self.layers):
            raise ValueError("Qwen3 DSpark KV cache count must match draft layers")
        if self._fused_kv_weight is None or self._k_norm_weights is None:
            self._build_fused_kv_buffers()

        hidden = self.hidden_norm(self.combine_hidden_states(target_hidden))
        fused = F.linear(hidden, self._fused_kv_weight, self._fused_kv_bias)
        tokens = fused.shape[0]
        num_layers = len(self.layers)
        num_kv_heads = self.layers[0].self_attn.num_kv_heads
        head_dim = self.cfg.head_dim
        # Match vLLM's DFlash layout exactly: project as [T, L, 2, H, D],
        # then make one contiguous layer-major copy [2, L, T, H, D].
        fused = fused.view(tokens, num_layers, 2, num_kv_heads, head_dim).permute(2, 1, 0, 3, 4).contiguous()
        keys = fused[0]
        values = fused[1]
        # The K norm is layer-specific and is applied before the same NEOX
        # rotation used by the normal Qwen3 attention path.
        # Use xllm_ops.rms_norm (NPU fused kernel, BF16) to match vLLM's
        # ops.rms_norm precision instead of manual float32 computation.
        k_eps = self.layers[0].self_attn.q_norm.eps
        normed_keys = []
        for layer_id in range(num_layers):
            k_layer = keys[layer_id]
            k_flat = k_layer.reshape(-1, head_dim)
            k_normed = torch.ops.xllm_ops.rms_norm(
                k_flat,
                self._k_norm_weights[layer_id],
                k_eps,
            )
            normed_keys.append(k_normed.view_as(k_layer))
        keys = torch.stack(normed_keys, dim=0)
        k_flat = keys.view(num_layers * tokens, -1).contiguous()
        positions_repeated = positions.repeat(num_layers)
        torch_npu._npu_rotary_embedding(
            positions_repeated,
            k_flat,
            k_flat.clone(),
            head_dim,
            self.rotary.cos_sin_cache,
            True,
        )
        keys = k_flat.view(num_layers, tokens, num_kv_heads, head_dim)
        for layer_id, (k_cache, v_cache, _) in enumerate(kv_caches):
            layer_key = _match_cache_heads(keys[layer_id], k_cache, "key")
            layer_value = _match_cache_heads(values[layer_id], v_cache, "value")
            if layer_key.shape != layer_value.shape:
                raise ValueError(
                    "Qwen3 DSpark key/value cache shape mismatch: "
                    f"key={tuple(layer_key.shape)}, value={tuple(layer_value.shape)}"
                )
            ops.reshape_paged_cache(
                slot_mapping,
                layer_key,
                layer_value,
                k_cache,
                v_cache,
            )
        return hidden

    def dspark_markov_bias(self, previous_token_ids: torch.Tensor) -> torch.Tensor:
        markov_embed = self.markov_w1(previous_token_ids)
        markov_bias = self.markov_w2(markov_embed)
        dump_dspark_tensors(
            "draft_markov",
            {
                "markov.previous_token_ids": previous_token_ids,
                "markov.embedding": markov_embed,
                "markov.bias": markov_bias,
            },
        )
        return markov_bias

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [f"layers.{layer.layer_id}.self_attn" for layer in self.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * len(self.layers)


class Qwen3DSparkForCausalLM(PyModelBase):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = Qwen3DSparkConfig.from_dict(config)
        self.dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        self.device = torch.device(config.get("device", "npu"))
        self.model = Qwen3DSparkModel(self.cfg, self.dtype, self.device)
        # Both are replaced with target modules by share_weights_from.
        self.lm_head: nn.Module | None = None

    def compute_logits(self, hidden: torch.Tensor, selected_idxes: torch.Tensor | None) -> torch.Tensor:
        if selected_idxes is not None and selected_idxes.numel() > 0:
            hidden = hidden.index_select(0, selected_idxes)
        if self.lm_head is None:
            raise RuntimeError("Qwen3 DSpark target LM head is not shared")
        logits = self.lm_head(hidden)
        dump_dspark_tensors(
            "draft_logits",
            {
                "draft.sample_hidden": hidden,
                "draft.base_logits": logits,
            },
        )
        return logits

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(target_hidden)

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        return self.model.write_context_kv(target_hidden, positions, slot_mapping, kv_caches)

    def dspark_markov_bias(self, previous_token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.dspark_markov_bias(previous_token_ids)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def get_draft_attn_causal(self) -> list[bool]:
        return self.model.get_draft_attn_causal()

    def load_weights(self, state_dicts: list[StateDict], tp_rank: int, tp_size: int) -> set[str]:
        merged = _MergedStateDict(list(state_dicts))
        state_dict = merged.get_dict_with_prefixes(["model.", ""])
        cfg = self.cfg
        loaded: set[str] = set()
        # The C++ bridge passes the effective TP group separately from the
        # checkpoint config. Draft configs often omit these runtime fields;
        # layer geometry must use the bridge values before constructing QKV.
        # A legacy bridge can still pass its default ``tp_size=1``; in that
        # case retain the effective world-derived width recovered by
        # Qwen3DSparkConfig.from_dict, just as vLLM does.
        runtime_tp_size = tp_size if tp_size > 1 else cfg.tp_size
        runtime_tp_rank = tp_rank if tp_size > 1 else cfg.tp_rank
        if _debug_enabled():
            print(
                f"DSpark load_weights: tp_rank={tp_rank} tp_size={tp_size} cfg.tp_rank={cfg.tp_rank} cfg.tp_size={cfg.tp_size} runtime_tp_rank={runtime_tp_rank} runtime_tp_size={runtime_tp_size}",
                flush=True,
            )
        tp_changed = (cfg.tp_size, cfg.tp_rank) != (runtime_tp_size, runtime_tp_rank)
        cfg.tp_size = runtime_tp_size
        cfg.tp_rank = runtime_tp_rank

        def copy(name: str, tensor: torch.Tensor) -> None:
            _copy_parameter(self.get_parameter(name), tensor, name)
            loaded.add(name)

        context = _first_tensor(state_dict, ("context_proj.weight", "fc.weight"))
        if context is None:
            raise KeyError("missing Qwen3 DSpark context projection weight")
        target_layers = cfg.num_target_layers
        if context.ndim != 2:
            raise ValueError(
                "Qwen3 DSpark context projection expects [hidden, target_hidden_size * "
                "target_layer_count], "
                f"got {tuple(context.shape)}"
            )
        if target_layers <= 0:
            if cfg.target_hidden_size <= 0 or context.shape[1] % cfg.target_hidden_size:
                raise ValueError(
                    "Qwen3 DSpark context projection expects [hidden, target_hidden_size * "
                    "target_layer_count], "
                    f"got {tuple(context.shape)}"
                )
            target_layers = context.shape[1] // cfg.target_hidden_size
            cfg.num_target_layers = target_layers
        if context.shape[1] % target_layers:
            # Some older configs carry zero/incorrect DSpark helper fields.
            # The draft layer count is still available from the model config,
            # while the fc width is authoritative for target_hidden_size.
            if cfg.target_hidden_size <= 0 or context.shape[1] % cfg.target_hidden_size:
                raise ValueError(
                    "Qwen3 DSpark context projection expects [hidden, target_hidden_size * "
                    "target_layer_count], "
                    f"got {tuple(context.shape)}"
                )
            target_layers = context.shape[1] // cfg.target_hidden_size
            cfg.num_target_layers = target_layers
        # The generic Qwen3 model args do not carry Kimi's target_hidden_size
        # override. The fc tensor is authoritative, because target_layer_count
        # comes from the draft config's capture-layer list.
        cfg.target_hidden_size = context.shape[1] // target_layers
        if self.model.context_proj.weight.shape != context.shape:
            self.model.context_proj = nn.Linear(
                context.shape[1],
                cfg.hidden_size,
                bias=False,
                dtype=self.dtype,
                device=self.device,
            )
        copy("model.context_proj.weight", context)

        layer_ids = _layer_ids(state_dict)
        if not layer_ids:
            raise KeyError("Qwen3 DSpark checkpoint has no decoder layers")
        # The model config (populated from C++ ModelArgs) is authoritative for
        # global head geometry.  A checkpoint tensor may be full-sized or
        # already sharded depending on the weight-loader backend; deriving
        # ``n_kv_heads`` from its row count would turn a full 64-head tensor
        # into a false 64-head model when the configured model has 16 KV heads.
        # vLLM keeps this geometry from the config and only shards weights.
        if tp_changed:
            self.model.rebuild_attention_layers()

        for name, aliases in (
            ("model.hidden_norm.weight", ("hidden_norm.weight", "context_norm.weight")),
            ("model.norm.weight", ("norm.weight", "final_norm.weight")),
            ("model.markov_w1.weight", ("markov_w1.weight", "markov_head.markov_w1.weight")),
            ("model.markov_w2.weight", ("markov_w2.weight", "markov_head.markov_w2.weight")),
        ):
            tensor = _first_tensor(state_dict, aliases)
            if tensor is None:
                raise KeyError(f"missing Qwen3 DSpark weight: {aliases[0]}")
            if name == "model.markov_w1.weight":
                markov_w2 = _first_tensor(
                    state_dict,
                    ("markov_w2.weight", "markov_head.markov_w2.weight"),
                )
                if markov_w2 is None or tensor.ndim != 2 or markov_w2.ndim != 2:
                    raise ValueError("Qwen3 DSpark Markov head weights must be rank-2")
                if tensor.shape[1] != markov_w2.shape[1]:
                    raise ValueError(
                        f"Qwen3 DSpark Markov head rank mismatch: w1={tuple(tensor.shape)}, w2={tuple(markov_w2.shape)}"
                    )
                if self.model.markov_w1.weight.shape != tensor.shape:
                    self.model.markov_w1 = nn.Embedding(
                        tensor.shape[0],
                        tensor.shape[1],
                        dtype=self.dtype,
                        device=self.device,
                    )
                if self.model.markov_w2.weight.shape != markov_w2.shape:
                    self.model.markov_w2 = nn.Linear(
                        markov_w2.shape[1],
                        markov_w2.shape[0],
                        bias=False,
                        dtype=self.dtype,
                        device=self.device,
                    )
            copy(name, tensor)

        total_kv_heads = cfg.n_kv_heads
        kv_replicas = runtime_tp_size // total_kv_heads if total_kv_heads < runtime_tp_size else 1
        kv_rank = runtime_tp_rank // kv_replicas if kv_replicas > 1 else runtime_tp_rank
        kv_world = runtime_tp_size // kv_replicas if kv_replicas > 1 else runtime_tp_size

        def shard(layer_state: Any, names: tuple[str, ...], dim: int, kv: bool = False) -> torch.Tensor:
            tensor = _first_sharded_tensor(
                layer_state,
                names,
                dim,
                kv_rank if kv else runtime_tp_rank,
                kv_world if kv else runtime_tp_size,
            )
            if tensor is None:
                raise KeyError(f"missing Qwen3 DSpark weight: {names[0]}")
            return tensor

        if len(layer_ids) != len(self.model.layers):
            raise KeyError(
                "Qwen3 DSpark checkpoint layer count does not match config: "
                f"got {layer_ids}, expected {len(self.model.layers)}"
            )
        for layer, layer_id in zip(self.model.layers, layer_ids, strict=True):
            prefix = f"layers.{layer_id}."
            layer_state = state_dict.get_dict_with_prefix(prefix)
            for parameter_name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                tensor = _first_tensor(layer_state, (parameter_name,))
                if tensor is None:
                    raise KeyError(f"missing Qwen3 DSpark weight: {prefix}{parameter_name}")
                _copy_parameter(layer.get_parameter(parameter_name), tensor, f"{prefix}{parameter_name}")
                loaded.add(f"{prefix}{parameter_name}")

            attention_state = layer_state.get_dict_with_prefix("self_attn.")
            for parameter_name in ("q_norm.weight", "k_norm.weight"):
                tensor = _first_tensor(attention_state, (parameter_name,))
                if tensor is None:
                    raise KeyError(f"missing Qwen3 DSpark weight: {prefix}self_attn.{parameter_name}")
                _copy_parameter(layer.self_attn.get_parameter(parameter_name), tensor, parameter_name)
                loaded.add(f"{prefix}self_attn.{parameter_name}")
            q = shard(attention_state, ("q_proj.weight",), 0)
            k = shard(attention_state, ("k_proj.weight",), 0, kv=True)
            v = shard(attention_state, ("v_proj.weight",), 0, kv=True)
            qkv_weight = torch.cat((q, k, v), dim=0)
            _copy_parameter(layer.self_attn.qkv_proj.weight, qkv_weight, "qkv_proj.weight")
            loaded.add(f"{prefix}self_attn.qkv_proj.weight")
            if layer_id == layer_ids[0]:
                actual_w = snapshot_for_dump(layer.self_attn.qkv_proj.weight)
                if _debug_enabled():
                    print(f"DSpark post_copy: qkv_proj.weight[0,:5]={actual_w[0, :5].tolist()}", flush=True)
                if _debug_enabled():
                    print(
                        f"DSpark weight debug: layer={layer_id} q_shard={q.shape} k_shard={k.shape} v_shard={v.shape} qkv={qkv_weight.shape}",
                        flush=True,
                    )

            if cfg.attention_bias:
                q_bias = shard(attention_state, ("q_proj.bias",), 0)
                k_bias = shard(attention_state, ("k_proj.bias",), 0, kv=True)
                v_bias = shard(attention_state, ("v_proj.bias",), 0, kv=True)
                _copy_parameter(
                    layer.self_attn.qkv_proj.bias,
                    torch.cat((q_bias, k_bias, v_bias), dim=0),
                    "qkv_proj.bias",
                )
                output_bias = _first_tensor(attention_state, ("o_proj.bias",))
                if output_bias is None:
                    raise KeyError(f"missing Qwen3 DSpark weight: {prefix}self_attn.o_proj.bias")
                _copy_parameter(layer.self_attn.o_proj.bias, output_bias, "o_proj.bias")
            output = shard(attention_state, ("o_proj.weight",), 1)
            _copy_parameter(layer.self_attn.o_proj.weight, output, "o_proj.weight")
            layer.self_attn.o_proj.format_npu_weight_()
            loaded.add(f"{prefix}self_attn.o_proj.weight")

            mlp_state = layer_state.get_dict_with_prefix("mlp.")
            gate = shard(mlp_state, ("gate_proj.weight",), 0)
            up = shard(mlp_state, ("up_proj.weight",), 0)
            _copy_parameter(layer.mlp.gate_up_proj.weight, torch.cat((gate, up), dim=0), "gate_up_proj.weight")
            down = shard(mlp_state, ("down_proj.weight",), 1)
            _copy_parameter(layer.mlp.down_proj.weight, down, "down_proj.weight")
            layer.mlp.down_proj.format_npu_weight_()
            loaded.update((f"{prefix}mlp.gate_up_proj.weight", f"{prefix}mlp.down_proj.weight"))

        self.model._build_fused_kv_buffers()
        w = self.model.layers[0].self_attn.qkv_proj.weight
        if _debug_enabled():
            print(
                f"DSpark END_LOAD: qkv_proj.weight ptr={w.data_ptr()} fmt={__import__('torch_npu').get_npu_format(w) if w.device.type in ('npu', 'privateuseone') else 'cpu'} first5={snapshot_for_dump(w)[0, :5].tolist()}",
                flush=True,
            )
        return loaded


DSparkDraftModel = Qwen3DSparkForCausalLM

__all__ = [
    "DSparkDraftModel",
    "Qwen3DSparkConfig",
    "Qwen3DSparkModel",
    "Qwen3DSparkForCausalLM",
]
