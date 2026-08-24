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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import ops
from xllm.python.layers import GatedMLP, RMSNorm, RotaryEmbedding
from xllm.python.model_executor.forward_context import record_layer_event
from xllm.python.models.base import PyModelBase
from xllm.python.models.kimi_k3_text import (
    _MergedStateDict,
    _layer_ids,
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
    def from_dict(cls, config: dict[str, Any]) -> "Qwen3DSparkConfig":
        base = Qwen3Config.from_dict(config)

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
        configured_layers = int(
            pick("dspark_num_target_layers", "num_target_layers", default=0) or 0
        )
        num_target_layers = len(target_ids) or configured_layers or base.n_layers
        configured_hidden = int(
            pick("dspark_target_hidden_size", "target_hidden_size", default=0) or 0
        )
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
            markov_rank=(
                int(pick("markov_rank", "dspark_markov_rank", default=0) or 0) or 256
            ),
            mask_token_id=mask_token_id,
        )


def _copy_parameter(parameter: torch.Tensor, tensor: torch.Tensor, name: str) -> None:
    if parameter.shape != tensor.shape:
        raise ValueError(
            f"Qwen3 DSpark parameter {name} expects {tuple(parameter.shape)}, "
            f"got {tuple(tensor.shape)}"
        )
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


class Qwen3DSparkAttention(Qwen3Attention):
    """Qwen3 attention with the non-causal runtime flag used by DSpark."""

    def __init__(self, cfg: Qwen3Config, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(cfg, layer_id, dtype, device)
        self.attn.non_causal_block = True


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden = self.self_attn(positions, hidden, cos_sin_cache, None, None)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        return self.mlp(hidden), residual


def _neox_rope(tensor: torch.Tensor, positions: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    """Apply the NEOX half-split rotation used by Qwen3's RoPE table."""
    cos_sin = cache[positions.to(torch.int64).contiguous()]
    half = cos_sin.shape[-1] // 2
    cos = cos_sin[..., :half].unsqueeze(1)
    sin = cos_sin[..., half:].unsqueeze(1)
    first, second = tensor[..., :half], tensor[..., half:]
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


def _match_cache_heads(
    tensor: torch.Tensor,
    cache: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Validate the vLLM/xLLM local GQA head layout before cache insertion."""
    if cache.ndim < 3:
        raise ValueError(
            f"Qwen3 DSpark {name} cache must be at least 3D, "
            f"got {tuple(cache.shape)}"
        )
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
        self.rotary = RotaryEmbedding(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList(
            [Qwen3DSparkDecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers)]
        )
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise RuntimeError("Qwen3 DSpark target embedding is not shared")
            inputs_embeds = self.embed_tokens(input_ids)
        positions = positions.to(torch.int64).contiguous()
        hidden, residual = inputs_embeds, None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, positions, self.rotary.cos_sin_cache)
            record_layer_event(layer.layer_id)
        hidden, _ = self.norm(hidden, residual)
        return hidden

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError("Qwen3 DSpark target embedding is not shared")
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = self.context_proj.in_features
        if target_hidden.shape[-1] != expected:
            raise ValueError(
                "Qwen3 DSpark context hidden size mismatch: "
                f"expected {expected}, got {target_hidden.shape[-1]}"
            )
        return self.context_proj(target_hidden)

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
                [layer.self_attn.qkv_proj.bias[layer.self_attn.q_size :]
                 for layer in self.layers], dim=0
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
        fused = (
            fused.view(tokens, num_layers, 2, num_kv_heads, head_dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        keys = fused[0]
        values = fused[1]
        # The K norm is layer-specific and is applied before the same NEOX
        # rotation used by the normal Qwen3 attention path.
        norm_weights = self._k_norm_weights.to(
            dtype=torch.float32, device=keys.device
        )
        variance = keys.float().pow(2).mean(dim=-1, keepdim=True)
        keys = keys.float() * torch.rsqrt(
            variance + self.layers[0].self_attn.q_norm.eps
        )
        keys = (keys * norm_weights.unsqueeze(1).unsqueeze(2)).to(dtype=fused.dtype)
        rotated = []
        for layer_id in range(num_layers):
            rotated.append(
                _neox_rope(keys[layer_id], positions, self.rotary.cos_sin_cache)
            )
        keys = torch.stack(rotated, dim=0)
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
        return self.markov_w2(self.markov_w1(previous_token_ids))

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
        return self.lm_head(hidden)

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
                context.shape[1], cfg.hidden_size, bias=False,
                dtype=self.dtype, device=self.device,
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
                        "Qwen3 DSpark Markov head rank mismatch: "
                        f"w1={tuple(tensor.shape)}, w2={tuple(markov_w2.shape)}"
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
        kv_replicas = (
            runtime_tp_size // total_kv_heads
            if total_kv_heads < runtime_tp_size
            else 1
        )
        kv_rank = (
            runtime_tp_rank // kv_replicas
            if kv_replicas > 1
            else runtime_tp_rank
        )
        kv_world = (
            runtime_tp_size // kv_replicas
            if kv_replicas > 1
            else runtime_tp_size
        )

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
            _copy_parameter(layer.self_attn.qkv_proj.weight, torch.cat((q, k, v), dim=0), "qkv_proj.weight")
            loaded.add(f"{prefix}self_attn.qkv_proj.weight")
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
        return loaded


DSparkDraftModel = Qwen3DSparkForCausalLM

__all__ = [
    "DSparkDraftModel",
    "Qwen3DSparkConfig",
    "Qwen3DSparkModel",
    "Qwen3DSparkForCausalLM",
]
