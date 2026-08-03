# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Kimi K3 MoonViT tower and PatchMergerV2 projector."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import ops
from xllm.python.layers import ColumnParallelLinear, RowParallelLinear


def _grid_thw_list(
    grid_thws: torch.Tensor | Sequence[Sequence[int]],
) -> list[tuple[int, int, int]]:
    values = grid_thws.tolist() if isinstance(grid_thws, torch.Tensor) else grid_thws
    grids: list[tuple[int, int, int]] = []
    for grid in values:
        if len(grid) != 3:
            raise ValueError(f"Kimi K3 grid must contain [t, h, w], got {grid}")
        time, height, width = (int(value) for value in grid)
        if time <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"Kimi K3 grid dimensions must be positive, got {grid}")
        grids.append((time, height, width))
    return grids


def _dtype_rms_norm_eps(dtype: torch.dtype) -> float:
    try:
        return torch.finfo(dtype).eps
    except TypeError as error:
        raise ValueError(f"Kimi K3 vision RMSNorm does not support {dtype}") from error


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.shape != key.shape or query.shape[0] != frequencies.shape[0]:
        raise ValueError("Kimi K3 RoPE input shapes do not match")
    if query.shape[-1] % 2 != 0:
        raise ValueError("Kimi K3 RoPE head dimension must be even")

    frequency_pairs = frequencies.unsqueeze(-2)
    query_complex = torch.view_as_complex(
        query.float().reshape(*query.shape[:-1], -1, 2)
    )
    key_complex = torch.view_as_complex(
        key.float().reshape(*key.shape[:-1], -1, 2)
    )
    query_out = torch.view_as_real(query_complex * frequency_pairs).flatten(-2)
    key_out = torch.view_as_real(key_complex * frequency_pairs).flatten(-2)
    return query_out.to(query.dtype), key_out.to(key.dtype)


@dataclass
class KimiK3VisionConfig:
    patch_size: int = 14
    in_channels: int = 3
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 12
    num_hidden_layers: int = 27
    qkv_hidden_size: int = 1536
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    init_pos_emb_time: int = 4
    patch_embed_proj_bias: bool = False
    attn_bias: bool = False
    linear_bias: bool = False
    activation_func: str = "gelu_pytorch_tanh"
    norm_type: str = "rmsnorm"
    mlp_type: str = "mlp2"
    pos_emb_type: str = "divided_fixed"
    merge_kernel_size: tuple[int, int] = (2, 2)
    merge_type: str = "sd2_tpool"
    mm_hidden_size: int = 1024
    text_hidden_size: int = 7168
    mm_projector_type: str = "patchmergerv2"
    projector_hidden_act: str = "gelu"
    projector_ln_eps: float = 1e-5
    tp_size: int = 1
    tp_rank: int = 0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "KimiK3VisionConfig":
        raw = config.get("vision_config", config)
        if not isinstance(raw, dict):
            raise TypeError("Kimi K3 vision_config must be a dictionary")

        merge_kernel_size = raw.get("merge_kernel_size", (2, 2))
        if not isinstance(merge_kernel_size, Sequence) or isinstance(
            merge_kernel_size, str
        ):
            raise TypeError("Kimi K3 merge_kernel_size must be a sequence")
        if len(merge_kernel_size) != 2:
            raise ValueError("Kimi K3 merge_kernel_size must contain two values")
        hidden_size = int(raw.get("vt_hidden_size", raw.get("hidden_size", 1024)))
        mm_hidden_size = raw.get("mm_hidden_size")
        return cls(
            patch_size=int(raw.get("patch_size", 14)),
            in_channels=int(raw.get("in_chans", raw.get("in_channels", 3))),
            hidden_size=hidden_size,
            intermediate_size=int(
                raw.get("vt_intermediate_size", raw.get("intermediate_size", 4096))
            ),
            num_attention_heads=int(
                raw.get(
                    "vt_num_attention_heads",
                    raw.get("num_attention_heads", 12),
                )
            ),
            num_hidden_layers=int(
                raw.get("vt_num_hidden_layers", raw.get("num_hidden_layers", 27))
            ),
            qkv_hidden_size=int(raw.get("qkv_hidden_size", 1536)),
            init_pos_emb_height=int(raw.get("init_pos_emb_height", 64)),
            init_pos_emb_width=int(raw.get("init_pos_emb_width", 64)),
            init_pos_emb_time=int(raw.get("init_pos_emb_time", 4)),
            patch_embed_proj_bias=bool(raw.get("patch_embed_proj_bias", False)),
            attn_bias=bool(raw.get("attn_bias", False)),
            linear_bias=bool(raw.get("linear_bias", False)),
            activation_func=str(raw.get("activation_func", "gelu_pytorch_tanh")),
            norm_type=str(raw.get("norm_type", "rmsnorm")),
            mlp_type=str(raw.get("mlp_type", "mlp2")),
            pos_emb_type=str(raw.get("pos_emb_type", "divided_fixed")),
            merge_kernel_size=(
                int(merge_kernel_size[0]),
                int(merge_kernel_size[1]),
            ),
            merge_type=str(raw.get("merge_type", "sd2_tpool")),
            mm_hidden_size=(
                hidden_size if mm_hidden_size is None else int(mm_hidden_size)
            ),
            text_hidden_size=int(raw.get("text_hidden_size", 7168)),
            mm_projector_type=str(raw.get("mm_projector_type", "patchmergerv2")),
            projector_hidden_act=str(raw.get("projector_hidden_act", "gelu")),
            projector_ln_eps=float(raw.get("projector_ln_eps", 1e-5)),
            tp_size=int(config.get("tp_size", raw.get("tp_size", 1))),
            tp_rank=int(config.get("tp_rank", raw.get("tp_rank", 0))),
        )

    def validate(self) -> None:
        if self.patch_size <= 0 or self.in_channels <= 0:
            raise ValueError("Kimi K3 patch size and input channels must be positive")
        if self.hidden_size <= 0 or self.hidden_size % 2 != 0:
            raise ValueError("Kimi K3 vision hidden size must be positive and even")
        if self.intermediate_size <= 0 or self.qkv_hidden_size <= 0:
            raise ValueError("Kimi K3 vision projection sizes must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("Kimi K3 vision layer count must be positive")
        if min(
            self.init_pos_emb_height,
            self.init_pos_emb_width,
            self.init_pos_emb_time,
        ) <= 0:
            raise ValueError("Kimi K3 initial position dimensions must be positive")
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("Kimi K3 TP rank and size are invalid")
        if self.num_attention_heads <= 0:
            raise ValueError("Kimi K3 vision head count must be positive")
        if self.qkv_hidden_size % self.num_attention_heads != 0:
            raise ValueError("Kimi K3 qkv_hidden_size must be divisible by heads")
        if self.qkv_hidden_size // self.num_attention_heads % 4 != 0:
            raise ValueError("Kimi K3 vision head dimension must be divisible by four")
        if self.num_attention_heads % self.tp_size != 0:
            raise ValueError("Kimi K3 vision heads must be divisible by tp_size")
        if self.intermediate_size % self.tp_size != 0:
            raise ValueError("Kimi K3 intermediate_size must be divisible by tp_size")
        if self.activation_func != "gelu_pytorch_tanh":
            raise ValueError(f"Unsupported Kimi K3 activation: {self.activation_func}")
        if self.norm_type != "rmsnorm":
            raise ValueError(f"Unsupported Kimi K3 norm: {self.norm_type}")
        if self.mlp_type != "mlp2":
            raise ValueError(f"Unsupported Kimi K3 MLP: {self.mlp_type}")
        if self.pos_emb_type != "divided_fixed":
            raise ValueError(
                f"Unsupported Kimi K3 position embedding: {self.pos_emb_type}"
            )
        if self.merge_type != "sd2_tpool":
            raise ValueError(f"Unsupported Kimi K3 merge type: {self.merge_type}")
        if self.mm_projector_type != "patchmergerv2":
            raise ValueError(
                f"Unsupported Kimi K3 projector: {self.mm_projector_type}"
            )
        if self.projector_hidden_act != "gelu":
            raise ValueError(
                f"Unsupported Kimi K3 projector activation: {self.projector_hidden_act}"
            )
        if any(size <= 0 for size in self.merge_kernel_size):
            raise ValueError("Kimi K3 merge dimensions must be positive")
        if self.mm_hidden_size != self.hidden_size:
            raise ValueError("Kimi K3 projector input must match vision hidden size")
        if self.text_hidden_size <= 0 or self.projector_ln_eps <= 0:
            raise ValueError(
                "Kimi K3 projector dimensions and epsilon must be positive"
            )


class KimiK3VisionPosEmbDivided(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.height = config.init_pos_emb_height
        self.width = config.init_pos_emb_width
        self.num_frames = config.init_pos_emb_time
        self.dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty(self.height, self.width, self.dim, dtype=dtype, device=device)
        )
        nn.init.normal_(self.weight)

        float_options = {"dtype": torch.float32, "device": device}
        omega = torch.arange(self.dim // 2, **float_options)
        omega = 1.0 / 10000.0 ** (omega / (self.dim / 2.0))
        positions = torch.arange(self.num_frames, **float_options)
        phases = torch.einsum("m,d->md", positions, omega)
        time_weight = torch.cat([phases.sin(), phases.cos()], dim=1)
        self.register_buffer(
            "time_weight",
            time_weight.unsqueeze(1).to(dtype=dtype),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        position_embeddings: list[torch.Tensor] = []
        for time, height, width in _grid_thw_list(grid_thws):
            if time > self.num_frames:
                raise ValueError(
                    f"Kimi K3 grid time {time} exceeds {self.num_frames}"
                )
            if height == self.height and width == self.width:
                spatial = self.weight.flatten(0, 1)
            else:
                spatial = F.interpolate(
                    self.weight.permute(2, 0, 1).unsqueeze(0),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0).flatten(0, 1)
            if time > 1:
                spatial = spatial.unsqueeze(0).repeat(time, 1, 1)
                spatial = spatial + self.time_weight[:time]
            position_embeddings.append(spatial.reshape(-1, self.dim))

        if not position_embeddings:
            return hidden_states
        positions = torch.cat(position_embeddings, dim=0).to(hidden_states)
        if positions.shape != hidden_states.shape:
            raise ValueError(
                f"Kimi K3 positions {positions.shape} do not match "
                f"patches {hidden_states.shape}"
            )
        return hidden_states + positions


class KimiK3VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=config.patch_embed_proj_bias,
            dtype=dtype,
            device=device,
        )
        self.pos_emb = KimiK3VisionPosEmbDivided(config, dtype, device)

    def forward(
        self,
        pixels: torch.Tensor,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        hidden_states = self.proj(pixels).flatten(1)
        return self.pos_emb(hidden_states, grid_thws)


class KimiK3VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_height: int = 512,
        max_width: int = 512,
    ) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("Kimi K3 vision head dimension must be divisible by four")
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.register_buffer("_frequencies", None, persistent=False)

    def _build_frequencies(self, device: torch.device) -> torch.Tensor:
        positions = torch.arange(
            self.max_height * self.max_width,
            dtype=torch.float32,
            device=device,
        )
        x_positions = positions.remainder(self.max_width)
        y_positions = positions.floor_divide(self.max_width)
        dim_range = torch.arange(0, self.dim, 4, dtype=torch.float32, device=device)
        frequencies = 1.0 / 10000.0 ** (dim_range / self.dim)
        x_phases = torch.outer(x_positions, frequencies)
        y_phases = torch.outer(y_positions, frequencies)
        x_complex = torch.polar(torch.ones_like(x_phases), x_phases)
        y_complex = torch.polar(torch.ones_like(y_phases), y_phases)
        return torch.stack([x_complex, y_complex], dim=-1).reshape(
            self.max_height,
            self.max_width,
            self.dim // 2,
        )

    def forward(
        self,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
        device: torch.device,
    ) -> torch.Tensor:
        if self._frequencies is None or self._frequencies.device != device:
            self._frequencies = self._build_frequencies(device)

        outputs: list[torch.Tensor] = []
        for time, height, width in _grid_thw_list(grid_thws):
            if height > self.max_height or width > self.max_width:
                raise ValueError(
                    f"Kimi K3 grid {(height, width)} exceeds RoPE cache "
                    f"{(self.max_height, self.max_width)}"
                )
            frequencies = self._frequencies[:height, :width].reshape(
                -1, self.dim // 2
            )
            outputs.append(frequencies.repeat(time, 1))
        if not outputs:
            return torch.empty(0, self.dim // 2, dtype=torch.complex64, device=device)
        return torch.cat(outputs, dim=0)


class KimiK3VisionMLP(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        intermediate_per_rank = config.intermediate_size // config.tp_size
        self.fc0 = ColumnParallelLinear(
            config.hidden_size,
            intermediate_per_rank,
            config.tp_size,
            bias=config.linear_bias,
            dtype=dtype,
            device=device,
        )
        self.fc1 = RowParallelLinear(
            intermediate_per_rank,
            config.hidden_size,
            config.tp_size,
            bias=config.linear_bias,
            dtype=dtype,
            device=device,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.fc0(hidden_states), approximate="tanh")
        return self.fc1(hidden_states)


class KimiK3VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads // config.tp_size
        self.head_dim = config.qkv_hidden_size // config.num_attention_heads
        qkv_hidden_per_rank = self.num_heads * self.head_dim
        norm_eps = _dtype_rms_norm_eps(dtype)
        self.norm0 = nn.RMSNorm(
            config.hidden_size,
            eps=norm_eps,
            dtype=dtype,
            device=device,
        )
        self.norm1 = nn.RMSNorm(
            config.hidden_size,
            eps=norm_eps,
            dtype=dtype,
            device=device,
        )
        self.mlp = KimiK3VisionMLP(config, dtype, device)
        self.wqkv = ColumnParallelLinear(
            config.hidden_size,
            3 * qkv_hidden_per_rank,
            config.tp_size,
            bias=config.attn_bias,
            dtype=dtype,
            device=device,
        )
        self.wo = RowParallelLinear(
            qkv_hidden_per_rank,
            config.hidden_size,
            config.tp_size,
            bias=config.attn_bias,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: Sequence[int],
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = self.norm0(hidden_states)
        qkv = self.wqkv(normalized).reshape(
            hidden_states.shape[0],
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=1)
        query, key = _apply_rope(query, key, frequencies)
        attended = ops.encoder_attention(
            query,
            key,
            value,
            sequence_lengths,
        )
        hidden_states = residual + self.wo(attended.flatten(1))
        return hidden_states + self.mlp(self.norm1(hidden_states))


class KimiK3VisionEncoder(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        head_dim = config.qkv_hidden_size // config.num_attention_heads
        self.rope_2d = KimiK3VisionRotaryEmbedding(head_dim)
        self.blocks = nn.ModuleList(
            [
                KimiK3VisionEncoderLayer(config, dtype, device)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=_dtype_rms_norm_eps(dtype),
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        grids = _grid_thw_list(grid_thws)
        sequence_lengths = [time * height * width for time, height, width in grids]
        frequencies = self.rope_2d(grids, hidden_states.device)
        for block in self.blocks:
            hidden_states = block(hidden_states, sequence_lengths, frequencies)
        return self.final_layernorm(hidden_states)


def tpool_patch_merger(
    hidden_states: torch.Tensor,
    grid_thws: torch.Tensor | Sequence[Sequence[int]],
    merge_kernel_size: tuple[int, int],
) -> list[torch.Tensor]:
    merge_height, merge_width = merge_kernel_size
    outputs: list[torch.Tensor] = []
    offset = 0
    for time, height, width in _grid_thw_list(grid_thws):
        if height % merge_height != 0 or width % merge_width != 0:
            raise ValueError(
                f"Kimi K3 grid {(height, width)} is not divisible by "
                f"merge kernel {merge_kernel_size}"
            )
        length = time * height * width
        sequence = hidden_states[offset : offset + length].reshape(
            time,
            height // merge_height,
            merge_height,
            width // merge_width,
            merge_width,
            hidden_states.shape[-1],
        )
        sequence = sequence.mean(dim=0).permute(0, 2, 1, 3, 4)
        outputs.append(
            sequence.reshape(
                (height // merge_height) * (width // merge_width),
                merge_height * merge_width,
                hidden_states.shape[-1],
            )
        )
        offset += length
    if offset != hidden_states.shape[0]:
        raise ValueError(
            f"Kimi K3 grids cover {offset} tokens, got {hidden_states.shape[0]}"
        )
    return outputs


class KimiK3VisionTower(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        config.validate()
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_embed = KimiK3VisionPatchEmbed(config, dtype, device)
        self.encoder = KimiK3VisionEncoder(config, dtype, device)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
    ) -> list[torch.Tensor]:
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        return tpool_patch_merger(
            hidden_states,
            grid_thws,
            self.merge_kernel_size,
        )


class KimiK3MultiModalProjector(nn.Module):
    def __init__(
        self,
        config: KimiK3VisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        merge_size = config.merge_kernel_size[0] * config.merge_kernel_size[1]
        self.input_size = config.mm_hidden_size * merge_size
        self.linear_1 = nn.Linear(
            self.input_size,
            self.input_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.linear_2 = nn.Linear(
            self.input_size,
            config.text_hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.post_norm = nn.RMSNorm(
            config.text_hidden_size,
            eps=config.projector_ln_eps,
            dtype=dtype,
            device=device,
        )
        self.rot_proj: nn.Linear | None = None

    def add_rot_proj(self) -> None:
        if self.rot_proj is not None:
            return
        reference = self.post_norm.weight
        self.rot_proj = nn.Linear(
            reference.shape[0],
            reference.shape[0],
            bias=False,
            dtype=reference.dtype,
            device=reference.device,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = image_features.reshape(-1, self.input_size)
        hidden_states = F.gelu(self.linear_1(hidden_states))
        hidden_states = self.post_norm(self.linear_2(hidden_states))
        if self.rot_proj is not None:
            hidden_states = self.rot_proj(hidden_states)
        return hidden_states


class KimiK3VisionModel(nn.Module):
    """Standalone vision path, ready to be composed with the K3 language model."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = KimiK3VisionConfig.from_dict(config)
        self.config.validate()
        dtype_value = config.get("dtype") or config.get("torch_dtype") or "bfloat16"
        dtype = (
            dtype_value
            if isinstance(dtype_value, torch.dtype)
            else getattr(torch, dtype_value)
        )
        device = torch.device(config.get("device", "cuda"))
        self.vision_tower = KimiK3VisionTower(self.config, dtype, device)
        self.mm_projector = KimiK3MultiModalProjector(self.config, dtype, device)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor | Sequence[Sequence[int]],
    ) -> list[torch.Tensor]:
        tower_outputs = self.vision_tower(pixel_values, grid_thws)
        if not tower_outputs:
            return []
        lengths = [output.shape[0] for output in tower_outputs]
        projected = self.mm_projector(torch.cat(tower_outputs, dim=0))
        return list(projected.split(lengths, dim=0))

    def load_weights(
        self,
        state_dicts: list[Any],
        tp_rank: int,
        tp_size: int,
    ) -> set[str]:
        if tp_rank != self.config.tp_rank or tp_size != self.config.tp_size:
            raise ValueError(
                "Kimi K3 loader TP rank/size must match model construction"
            )
        rot_proj_name = "mm_projector.rot_proj.weight"
        if (
            _find_tensor(
                state_dicts,
                (rot_proj_name, f"model.{rot_proj_name}"),
            )
            is not None
        ):
            self.mm_projector.add_rot_proj()

        loaded: set[str] = set()
        for name, parameter in self.named_parameters():
            checkpoint_names = _checkpoint_names(name)
            tensor = _find_tensor(state_dicts, checkpoint_names)
            if tensor is None:
                raise KeyError(
                    f"Kimi K3 checkpoint tensor not found: {checkpoint_names[0]}"
                )
            tensor = _shard_vision_weight(name, tensor, tp_rank, tp_size)
            if tensor.shape != parameter.shape:
                raise ValueError(
                    f"Kimi K3 weight {name} expects {parameter.shape}, "
                    f"got {tensor.shape}"
                )
            parameter.data.copy_(
                tensor.to(dtype=parameter.dtype, device=parameter.device)
            )
            loaded.add(name)
        return loaded


def _checkpoint_names(parameter_name: str) -> tuple[str, ...]:
    aliases = [parameter_name, f"model.{parameter_name}"]
    projector_aliases = {
        "mm_projector.linear_1.weight": "mm_projector.proj.0.weight",
        "mm_projector.linear_2.weight": "mm_projector.proj.2.weight",
    }
    if parameter_name in projector_aliases:
        alias = projector_aliases[parameter_name]
        aliases.extend([alias, f"model.{alias}"])
    return tuple(aliases)


def _find_tensor(
    state_dicts: Sequence[Any],
    names: Sequence[str],
) -> torch.Tensor | None:
    for name in names:
        for state_dict in state_dicts:
            if state_dict.has(name):
                return state_dict.get_tensor(name)
    return None


def _shard_tensor(
    tensor: torch.Tensor,
    dim: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    if world_size == 1:
        return tensor
    if tensor.shape[dim] % world_size != 0:
        raise ValueError(
            f"Kimi K3 weight dimension {tensor.shape[dim]} is not divisible by "
            f"tp_size {world_size}"
        )
    shard_size = tensor.shape[dim] // world_size
    return tensor.narrow(dim, rank * shard_size, shard_size).contiguous()


def _shard_vision_weight(
    name: str,
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    if ".wqkv." in name:
        chunks = tensor.chunk(3, dim=0)
        return torch.cat(
            [_shard_tensor(chunk, 0, rank, world_size) for chunk in chunks],
            dim=0,
        )
    if ".mlp.fc0." in name:
        return _shard_tensor(tensor, 0, rank, world_size)
    if ".mlp.fc1.weight" in name or ".wo.weight" in name:
        return _shard_tensor(tensor, 1, rank, world_size)
    return tensor


__all__ = [
    "KimiK3MultiModalProjector",
    "KimiK3VisionConfig",
    "KimiK3VisionEncoderLayer",
    "KimiK3VisionModel",
    "KimiK3VisionTower",
    "tpool_patch_merger",
]
