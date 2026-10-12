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

"""DeepSeek-V4 DSpark draft model for the Python NPU executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from xllm.python import distributed
from xllm.python.attention.backend import LayerCacheInput, normalize_layer_caches
from xllm.python.layers.layernorm import RMSNorm
from xllm.python.layers.qlinear import QLinear
from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.model_loader.module_loaders import load_w8a8_dynamic_projection
from xllm.python.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    _expand_half_rope_cos_sin,
    _find_checkpoint_prefix,
    _require_checkpoint_key,
)
from xllm.python.models.dspark import DSparkForCausalLMBase
from xllm.python.models.weight_utils import W8A8WeightLoader


class _DSparkQuantAwareLinear(QLinear):
    """FP or dynamic-W8A8 linear with optional output gathering."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_size: int,
        gather_output: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            device=device,
            dtype=dtype,
            kind="dynamic",
        )
        self.tp_size = tp_size
        self.gather_output = gather_output

    def resolve_quant(self, has_quant_tensors: bool) -> None:
        super().resolve_quant(has_quant_tensors)
        if self.use_w8a8:
            self.register_parameter("weight", None)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = super().forward(value)
        if self.gather_output and self.tp_size > 1:
            output = distributed.tp_all_gather(output, dim=-1, world_size=self.tp_size)
        return output


@dataclass
class DeepseekV4DSparkConfig(DeepseekV4Config):
    """DeepSeek-V4 target configuration narrowed to its DSpark draft."""

    dspark_num_layers: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    markov_rank: int = 0
    enable_confidence_head: bool = False
    confidence_head_with_markov: bool = True
    dspark_use_native_sas: bool = False

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> DeepseekV4DSparkConfig:
        base = DeepseekV4Config.from_dict(config)
        target_layer_ids = tuple(int(layer_id) for layer_id in config.get("dspark_target_layer_ids", ()))
        num_layers = int(config.get("dspark_num_layers", len(target_layer_ids)))
        values = dict(base.__dict__)
        values.update(
            model_type="deepseek_v4_dspark",
            n_layers=num_layers,
            n_hash_layers=0,
            compress_ratios=[1] * num_layers,
            dspark_num_layers=num_layers,
            dspark_target_layer_ids=target_layer_ids,
            dspark_block_size=int(config.get("dspark_block_size", 0)),
            dspark_noise_token_id=int(config.get("dspark_noise_token_id", 0)),
            markov_rank=int(config.get("markov_rank", config.get("dspark_markov_rank", 0))),
            enable_confidence_head=bool(config.get("enable_confidence_head", False)),
            confidence_head_with_markov=bool(config.get("confidence_head_with_markov", True)),
            dspark_use_native_sas=bool(config.get("dspark_use_native_sas", False)),
        )
        return cls(**values)

    def validate(self) -> None:
        if self.dspark_num_layers <= 0:
            raise ValueError("DeepSeek-V4 DSpark requires dspark_num_layers > 0")
        if self.dspark_target_layer_ids and len(self.dspark_target_layer_ids) != self.dspark_num_layers:
            raise ValueError("DeepSeek-V4 DSpark target layer count must match dspark_num_layers")
        if self.markov_rank <= 0:
            raise ValueError("DeepSeek-V4 DSpark requires dspark_markov_rank > 0")


class DeepseekV4DSparkModel(DeepseekV4Model):
    """DeepSeek-V4 decoder layers with shared target-context projection."""

    def __init__(
        self,
        cfg: DeepseekV4DSparkConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(cfg, dtype, device)
        self.cfg = cfg
        self.main_proj = _DSparkQuantAwareLinear(
            cfg.hidden_size * cfg.dspark_num_layers,
            cfg.hidden_size,
            tp_size=1,
            gather_output=False,
            dtype=dtype,
            device=device,
        )
        self.main_norm = RMSNorm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[LayerCacheInput],
        layer_synchronizer: LayerSynchronizer | None,
    ) -> torch.Tensor | None:
        if len(kv_caches) != self.cfg.n_layers:
            raise ValueError("DeepSeek-V4 DSpark cache/layer count mismatch")
        if cache_slots.numel() != target_hidden.size(0):
            raise ValueError("DeepSeek-V4 DSpark context slot count mismatch")
        expected_width = self.cfg.hidden_size * self.cfg.dspark_num_layers
        if target_hidden.dim() != 2 or target_hidden.size(-1) != expected_width:
            raise ValueError(
                f"DeepSeek-V4 DSpark target hidden width must equal dspark_num_layers * hidden_size ({expected_width})"
            )

        projected = self.main_norm(self.main_proj(target_hidden))
        positions = positions.to(torch.int64).contiguous()
        cos_sin = self.rotary.cos_sin_cache.index_select(0, positions)
        cos, sin = _expand_half_rope_cos_sin(cos_sin)
        layer_caches = normalize_layer_caches(kv_caches)
        for layer_id, (layer, layer_cache) in enumerate(zip(self.layers, layer_caches, strict=True)):
            if layer_cache.swa is None:
                raise RuntimeError(f"DeepSeek-V4 DSpark SWA cache is missing for layer {layer_id}")
            layer.write_context_kv(
                projected,
                cos,
                sin,
                cache_slots,
                layer_cache.swa,
            )
            if layer_synchronizer is not None and not layer_synchronizer.record_event(layer_id):
                return None
        return projected


class DeepseekV4DSparkForCausalLM(
    DeepseekV4ForCausalLM,
    DSparkForCausalLMBase,
):
    """DeepSeek-V4 DSpark draft calculator driven by the C++ worker."""

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = DeepseekV4DSparkConfig.from_dict(config)
        cfg.validate()
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0"))
        DSparkForCausalLMBase.__init__(
            self,
            vocab_size=cfg.vocab_size,
            draft_vocab_size=cfg.vocab_size,
            markov_rank=cfg.markov_rank,
            hidden_size=cfg.hidden_size,
            enable_confidence_head=cfg.enable_confidence_head,
            confidence_head_with_markov=cfg.confidence_head_with_markov,
            dtype=dtype,
            device=device,
            confidence_head_bias=False,
        )
        self.cfg = cfg
        self.model = DeepseekV4DSparkModel(cfg, dtype, device)
        self.lm_head = _DSparkQuantAwareLinear(
            cfg.hidden_size,
            cfg.vocab_size // cfg.tp_size,
            tp_size=cfg.tp_size,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[LayerCacheInput],
        layer_synchronizer: LayerSynchronizer | None,
    ) -> torch.Tensor | None:
        return self.model.write_context_kv(
            target_hidden,
            positions,
            cache_slots,
            kv_caches,
            layer_synchronizer,
        )

    def _load_dsv4_decoder_layer(
        self,
        loader: W8A8WeightLoader,
        checkpoint_prefix: str,
        parameter_prefix: str,
        layer_id: int,
    ) -> None:
        """Load one draft decoder layer without requiring the target MTP model."""

        def _w8a8(
            checkpoint_module: str,
            parameter_module: str,
            shard_dims: dict[str, int] | None = None,
        ) -> None:
            load_w8a8_dynamic_projection(
                loader,
                checkpoint_module,
                parameter_module,
                shard_dims,
                require_weight_and_scale=True,
                fill_missing_offset=True,
                description=f"DeepSeek-V4 DSpark {checkpoint_module}",
            )

        attention, _ = self._load_dsv4_attention(
            loader,
            checkpoint_prefix=checkpoint_prefix,
            parameter_prefix=parameter_prefix,
            layer_id=layer_id,
            w8a8_loader=_w8a8,
        )
        attention.process_weights_after_loading()

        mlp = self.model.layers[layer_id].mlp
        if hasattr(mlp, "experts_w13"):
            moe_prefix = _find_checkpoint_prefix(
                loader,
                (
                    checkpoint_prefix + "ffn.",
                    checkpoint_prefix + "mlp.",
                ),
                (
                    "experts.0.gate_proj.weight",
                    "experts.0.w1.weight",
                ),
            )
            if moe_prefix is None:
                raise KeyError(f"DeepSeek-V4 DSpark MoE weights not found under {checkpoint_prefix}")
            self._load_dsv4_moe(
                loader,
                checkpoint_prefix,
                parameter_prefix,
                layer_id,
                moe_prefix,
            )
            mlp.process_weights_after_loading()
        else:
            self._load_dsv4_dense_mlp(loader, checkpoint_prefix, parameter_prefix, mlp)

    @staticmethod
    def _load_quant_aware_linear(
        loader: W8A8WeightLoader,
        checkpoint_module: str,
        parameter_module: str,
        linear: _DSparkQuantAwareLinear,
        *,
        shard_dim: int | None = None,
    ) -> None:
        weight_key = _require_checkpoint_key(
            loader,
            (checkpoint_module + ".weight",),
            f"DeepSeek-V4 DSpark {checkpoint_module} weight",
        )
        weight = loader.load_tensor(weight_key)
        has_weight_scale = loader.has(checkpoint_module + ".weight_scale")
        if not has_weight_scale:
            if not weight.is_floating_point():
                raise KeyError(f"DeepSeek-V4 DSpark quantized {checkpoint_module}.weight requires weight_scale")
            linear.resolve_quant(False)
            if shard_dim is not None:
                weight = loader.shard(weight, dim=shard_dim)
            linear.load_weight(weight)
            return

        linear.resolve_quant(True)
        scale = loader.load_tensor(checkpoint_module + ".weight_scale")
        offset_key = checkpoint_module + ".weight_offset"
        offset = loader.load_tensor(offset_key) if loader.has(offset_key) else torch.zeros_like(scale)
        if shard_dim is not None:
            weight = loader.shard(weight, dim=shard_dim)
            scale = loader.shard(scale, dim=shard_dim)
            offset = loader.shard(offset, dim=shard_dim)
        quant_linear = linear._w8a8
        if quant_linear is None:
            raise RuntimeError(f"failed to initialize quantized {parameter_module}")
        quant_linear.weight.data.copy_(weight)
        quant_linear.weight_scale.data.copy_(scale)
        quant_linear.weight_offset.data.copy_(offset)
        linear.process_weights_after_loading()

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        del tp_rank, tp_size
        loader = W8A8WeightLoader(
            self,
            state_dicts,
            self.cfg.tp_size,
            self.cfg.tp_rank,
            src_prefixes=("", "model."),
        )
        layer_prefixes: list[str] = []
        for layer_id in range(self.cfg.n_layers):
            checkpoint_prefix = _find_checkpoint_prefix(
                loader,
                (f"mtp.{layer_id}.",),
                ("attn.wq_a.weight", "self_attn.wq_a.weight"),
            )
            if checkpoint_prefix is None:
                raise KeyError(f"DeepSeek-V4 DSpark layer {layer_id} weights not found")
            layer_prefixes.append(checkpoint_prefix)
            self._load_dsv4_decoder_layer(
                loader,
                checkpoint_prefix,
                f"model.layers.{layer_id}.",
                layer_id,
            )

        first_prefix = layer_prefixes[0]
        last_prefix = layer_prefixes[-1]
        main_proj_key = _require_checkpoint_key(
            loader,
            (first_prefix + "main_proj.weight",),
            "DeepSeek-V4 DSpark main projection",
        )
        self._load_quant_aware_linear(
            loader,
            main_proj_key.removesuffix(".weight"),
            "model.main_proj",
            self.model.main_proj,
        )
        main_norm_key = _require_checkpoint_key(
            loader,
            (first_prefix + "main_norm.weight",),
            "DeepSeek-V4 DSpark main norm",
        )
        loader.copy_in("model.main_norm.weight", loader.load_tensor(main_norm_key))

        embed_key = _require_checkpoint_key(
            loader,
            (
                first_prefix + "embed.weight",
                first_prefix + "emb.tok_emb.weight",
                "embed.weight",
                "embed_tokens.weight",
            ),
            "DeepSeek-V4 DSpark embedding weight",
        )
        loader.copy_in(
            "model.embed_tokens.weight",
            loader.shard(loader.load_tensor(embed_key), dim=1),
        )
        norm_key = _require_checkpoint_key(
            loader,
            (last_prefix + "norm.weight", "norm.weight", "final_layernorm.weight"),
            "DeepSeek-V4 DSpark final norm",
        )
        loader.copy_in("model.norm.weight", loader.load_tensor(norm_key))
        for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            key = _require_checkpoint_key(
                loader,
                (last_prefix + name,),
                f"DeepSeek-V4 DSpark {name}",
            )
            loader.copy_in("model." + name, loader.load_tensor(key))

        for name in ("markov_w1.weight", "markov_w2.weight"):
            key = _require_checkpoint_key(
                loader,
                (last_prefix + "markov_head." + name,),
                f"DeepSeek-V4 DSpark Markov {name}",
            )
            loader.copy_in("markov_head." + name, loader.load_tensor(key))
        if self.confidence_head is not None:
            confidence_key = _require_checkpoint_key(
                loader,
                (last_prefix + "confidence_head.proj.weight",),
                "DeepSeek-V4 DSpark confidence projection",
            )
            loader.copy_in("confidence_head.proj.weight", loader.load_tensor(confidence_key))

        head_key = _require_checkpoint_key(
            loader,
            (last_prefix + "head.weight", "head.weight", "lm_head.weight"),
            "DeepSeek-V4 DSpark output head",
        )
        self._load_quant_aware_linear(
            loader,
            head_key.removesuffix(".weight"),
            "lm_head",
            self.lm_head,
            shard_dim=0,
        )
