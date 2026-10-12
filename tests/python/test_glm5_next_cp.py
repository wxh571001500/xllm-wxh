# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import linear_state_checkpoint_stride  # noqa: E402
from xllm.python.attention.kv_shard_layout import (  # noqa: E402
    localize_pool_write_block_table,
    replicate_pool_write_block_table,
)
from xllm.python.layers import moe_parallel
from xllm.python.layers.moe_parallel import TokenParallelLayout
from xllm.python.model_executor import cp_utils
from xllm.python.models import glm5_next
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.glm5_next_kpool import read_pools

# ``xllm.python.attention.npu_paged_attention`` imports ``torch_npu`` at
# module scope; stub it when absent so the real-backend handoff tests below
# (which need the real cache-write code paths, not a mocked backend) can
# import ``NpuPagedAttentionBackend`` on a host without NPU hardware. The
# stub is installed only when torch_npu cannot actually be imported: a
# wheel-only install without CANN/libhccl.so still has a find_spec hit but
# fails the real import, so probe by importing. In the NPU container the
# real module stays in charge, and a bare entry a previously imported test
# module may have left in ``sys.modules`` satisfies the import directly.
# The stub is removed again after the imports below so the rest of the
# pytest session sees the true absence -- including glm5_next's own
# module-level reference: its try/except binds either the real module or
# None, and leaving it bound to the stub would make
# ``_load_compact_kpool_update_op``'s ``torch_npu is None`` capability
# check falsely pass on hosts without torch_npu, driving far-away failures
# in the Triton kpool loader. Restoring None keeps the fresh-process
# semantics for every later collector in the same session.
try:
    import torch_npu  # noqa: F401

    _torch_npu_stub_installed = False
except Exception:  # no CANN / missing libhccl.so fails at import time
    sys.modules.pop("torch_npu", None)
    sys.modules["torch_npu"] = types.ModuleType("torch_npu")
    _torch_npu_stub_installed = True

from xllm.python.attention import npu_paged_attention as npu_paged_attention_module
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend

if _torch_npu_stub_installed:
    del sys.modules["torch_npu"]
    # Both modules that carry a module-level torch_npu reference bound
    # during the stub window are reset to the fresh-process no-torch_npu
    # state: glm5_next's try/except (None fallback, keyed by capability
    # checks) and npu_paged_attention's plain import (no fallback -- a
    # bare None matches "module absent" for every reader that does not
    # call through it, and the stub never satisfied attribute calls
    # either). Leaving either bound to the stub would keep a state no
    # fresh process without torch_npu could reach.
    glm5_next.torch_npu = None
    npu_paged_attention_module.torch_npu = None


class _Embedding(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._events.append("embedding")
        return input_ids.to(torch.float32).unsqueeze(-1)


class _DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, events: list[str]) -> None:
        super().__init__()
        self.layer_id = layer_id
        self._events = events
        self.positions: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None
        self.prev_topk: torch.Tensor | None = None
        self.output_topk: torch.Tensor | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        prev_topk: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.positions = positions
        self.attention_mask = attention_mask
        self.prev_topk = prev_topk
        self.output_topk = hidden[:, :1, 0].clone()
        self._events.append(f"layer_{self.layer_id}")
        return hidden + self.layer_id + 1, self.output_topk


class _Norm(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self._events.append("norm")
        return hidden


class _HyperHead(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.mean(dim=2)


def _make_model(events: list[str]) -> tuple[glm5_next.Glm5NextModel, list[_DecoderLayer]]:
    model = glm5_next.Glm5NextModel.__new__(glm5_next.Glm5NextModel)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(hidden_size=1, hc_mult=4)
    model.embed_tokens = _Embedding(events)
    layers = [_DecoderLayer(layer_id, events) for layer_id in range(2)]
    model.layers = nn.ModuleList(layers)
    model.norm = _Norm(events)
    model.hc_head = _HyperHead()
    model.aux_hidden_capture = AuxHiddenCapture(())
    model._inputs_embeds = None
    return model, layers


def test_cp_model_loop_shards_rows_and_merges_final_hidden() -> None:
    events: list[str] = []
    model, layers = _make_model(events)
    cp_context = SimpleNamespace(shard_valid_mask=torch.tensor([True, False]))
    merged_output = torch.tensor([[13.0], [23.0], [33.0], [43.0]])

    def shard_rows(hidden: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        events.append("shard_rows")
        return hidden.index_select(0, torch.tensor([3, 0]))

    def shard_positions(positions: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        events.append("shard_positions")
        return positions.index_select(0, torch.tensor([3, 0]))

    def merge_rows(hidden: torch.Tensor, context: object) -> torch.Tensor:
        assert context is cp_context
        torch.testing.assert_close(hidden, torch.tensor([[43.0], [13.0]]))
        events.append("merge_rows")
        return merged_output

    with (
        patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)),
        patch.object(glm5_next, "cp_shard_rows", side_effect=shard_rows),
        patch.object(glm5_next, "cp_shard_positions", side_effect=shard_positions),
        patch.object(glm5_next, "cp_merge_rows", side_effect=merge_rows),
    ):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    assert events == [
        "embedding",
        "shard_rows",
        "shard_positions",
        "layer_0",
        "layer_1",
        "norm",
        "merge_rows",
    ]
    torch.testing.assert_close(layers[0].positions, torch.tensor([[3, 0]], dtype=torch.int32))
    torch.testing.assert_close(layers[0].attention_mask, torch.tensor([[True, False]]))
    assert layers[1].prev_topk is layers[0].output_topk
    torch.testing.assert_close(output, merged_output)


class _SpMoe(glm5_next.Glm5NextMoE):
    def __init__(self) -> None:
        nn.Module.__init__(self)


class _SpDecoder(nn.Module):
    """Last sparse layer: keep only this TP rank's shard row."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_id = 0
        self.mlp = _SpMoe()

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        prev_topk: torch.Tensor | None,
        *,
        input_layout: TokenParallelLayout | None = None,
        output_layout: TokenParallelLayout | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        assert output_layout is not None and output_layout.shard_tokens == 1
        local = hidden[:, :1] + 1
        return local, None


def test_pcp_sp_gathers_tp_rows_before_cp_merge() -> None:
    """CP=2, TP=4, 8 tokens: SP owns 1 row, CP merge needs all 4 local rows."""
    events: list[str] = []
    model, _layers = _make_model(events)
    model.cfg = SimpleNamespace(
        hidden_size=1,
        hc_mult=4,
        expert_parallel_degree=2,
        eplv2_sequence_parallel=True,
        tp_size=4,
        tp_rank=0,
        layers_to_capture=(),
    )
    model.layers = nn.ModuleList([_SpDecoder()])
    layout = TokenParallelLayout(4, 4, 0)
    model._sp_layout_and_mask = lambda hidden, attention_mask: (layout, torch.ones(1, dtype=torch.bool))
    cp_context = SimpleNamespace(
        cp_size=2,
        total_local=4,
        shard_gather_index=torch.arange(4),
        shard_valid_mask=torch.ones(4, dtype=torch.bool),
        restore_index=torch.arange(8),
    )

    def tp_all_gather(value: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
        assert dim == 0 and world_size == 4 and value.shape[0] == 1
        events.append("tp_gather")
        return value.repeat(4, *([1] * (value.dim() - 1)))

    def all_gather(value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0 and world_size == 2 and group_name == "cp"
        assert value.shape[0] == cp_context.total_local
        events.append(f"cp_merge:{tuple(value.shape)}")
        return torch.cat((value, value + 10), dim=0)

    with (
        patch.object(glm5_next, "get_forward_context_or_none", return_value=SimpleNamespace(cp_context=cp_context)),
        patch.object(moe_parallel.distributed, "tp_all_gather", side_effect=tp_all_gather, create=True),
        patch.object(cp_utils.distributed, "all_gather", side_effect=all_gather, create=True),
    ):
        output = model(torch.arange(8), torch.arange(8))

    assert events[-2:] == ["tp_gather", "cp_merge:(4, 1)"]
    assert output.shape == (8, 1)


def test_pcp_sp_aux_capture_merges_full_cp_local_rows() -> None:
    events: list[str] = []
    model, _layers = _make_model(events)
    model.cfg = SimpleNamespace(
        hidden_size=1,
        hc_mult=4,
        expert_parallel_degree=2,
        eplv2_sequence_parallel=True,
        tp_size=4,
        tp_rank=0,
        layers_to_capture=(0,),
    )
    model.layers = nn.ModuleList([_SpDecoder()])
    model.aux_hidden_capture = AuxHiddenCapture(
        (0,),
        transform=lambda streams: streams.mean(dim=2).reshape(-1, 1),
    )
    layout = TokenParallelLayout(4, 4, 0)
    model._sp_layout_and_mask = lambda hidden, attention_mask: (layout, torch.ones(1, dtype=torch.bool))
    cp_context = SimpleNamespace(
        cp_size=2,
        total_local=4,
        shard_gather_index=torch.arange(4),
        shard_valid_mask=torch.ones(4, dtype=torch.bool),
        restore_index=torch.arange(8),
    )
    merged_shapes: list[tuple[int, ...]] = []

    def tp_all_gather(value: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
        assert value.shape[0] == 1
        return value.repeat(4, *([1] * (value.dim() - 1)))

    def all_gather(value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert group_name == "cp" and value.shape[0] == 4
        merged_shapes.append(tuple(value.shape))
        return torch.cat((value, value + 10), dim=0)

    with (
        patch.object(glm5_next, "get_forward_context_or_none", return_value=SimpleNamespace(cp_context=cp_context)),
        patch.object(moe_parallel.distributed, "tp_all_gather", side_effect=tp_all_gather, create=True),
        patch.object(cp_utils.distributed, "all_gather", side_effect=all_gather, create=True),
    ):
        hidden, aux = model(torch.arange(8), torch.arange(8))

    assert hidden.shape == (8, 1)
    assert aux.shape[0] == 8
    # The model exit packs h [4, 1] and the capture [4, 1] along the feature
    # axis, so the full CP-local rows still reach the CP group -- now as one
    # collective of packed width 2 instead of two of width 1.
    assert merged_shapes == [(4, 2)]


def _cp2_context(rank: int) -> SimpleNamespace:
    shard_index = torch.tensor([0, 3], dtype=torch.int64) if rank == 0 else torch.tensor([1, 2], dtype=torch.int64)
    # Global 4 tokens, zigzag cp_size=2, chunk_len=1: every rank owns two real
    # rows, one per (sequence, half) segment (rank 0: global 0 then 3; rank 1:
    # global 1 then 2). query_index is therefore the identity here, while the
    # per-segment causal prefixes differ per rank.
    segment_kv_seq_lens = [1, 4] if rank == 0 else [2, 3]
    return SimpleNamespace(
        cp_size=2,
        cp_rank=rank,
        total_local=2,
        shard_index=shard_index,
        shard_gather_index=shard_index,
        shard_valid_mask=torch.ones(2, dtype=torch.bool),
        restore_index=torch.tensor([0, 2, 3, 1], dtype=torch.int64),
        query_index=torch.tensor([0, 1], dtype=torch.int64),
        q_cu_seqlens=[1, 2],
        q_cu_seqlens_tensor=torch.tensor([1, 2], dtype=torch.int32),
        segment_seq_indices=torch.tensor([0, 0], dtype=torch.int64),
        segment_kv_seq_lens=segment_kv_seq_lens,
        segment_kv_seq_lens_tensor=torch.tensor(segment_kv_seq_lens, dtype=torch.int32),
    )


def _patch_cp_gather(monkeypatch, remote_by_local: dict[tuple[object, ...], torch.Tensor]) -> MagicMock:
    def all_gather(value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0
        assert world_size == 2
        assert group_name == "cp"
        key = tuple(value.reshape(-1).tolist())
        # The peer's rows are supplied flat; match the gathered value's trailing
        # dims so one map serves 2-D and 3-D call sites.
        remote = remote_by_local[key].reshape(-1, *value.shape[1:])
        return torch.cat([value, remote], dim=0)

    gather = MagicMock(side_effect=all_gather)
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)
    return gather


def test_kda_cp_materializes_global_rows_and_reshards_output(monkeypatch) -> None:
    attention = glm5_next.Glm5NextKdaAttention.__new__(glm5_next.Glm5NextKdaAttention)
    nn.Module.__init__(attention)
    attention.layer_id = 0
    attention.hidden_size = 1
    attention.head_dim = 1
    attention.qkv_dim = 1
    attention.conv_dim = 3
    attention.num_heads_local = 1
    attention.tp = 1
    attention.in_proj_qkvbfg_a = lambda hidden: torch.cat(
        (hidden.expand(-1, -1, 3), hidden, hidden, hidden),
        dim=-1,
    )
    attention.input_projection_sizes = (3, 1, 2)
    attention.forget_gate = SimpleNamespace(raw_projection=lambda hidden: hidden.unsqueeze(-1) * 10)
    attention.g_b_proj = lambda hidden: hidden * 100
    attention._fg_b_weight = None
    attention.o_norm = MagicMock(side_effect=lambda core, gate: core + gate)
    attention.o_proj = nn.Identity()

    cp_context = _cp2_context(0)
    local_hidden = torch.tensor([[[1.0], [4.0]]])
    remote_hidden = torch.tensor([[2.0], [3.0]])
    remote_by_local = {
        (1.0, 1.0, 1.0, 4.0, 4.0, 4.0): remote_hidden.expand(-1, 3),
        (10.0, 40.0): remote_hidden.mul(10).unsqueeze(-1),
        (1.0, 4.0): remote_hidden,
    }
    gather = _patch_cp_gather(monkeypatch, remote_by_local)

    backend = MagicMock()
    backend.execute_linear.side_effect = lambda mixed_qkv, _beta, _layer, *, raw_gate_proj: (
        mixed_qkv[:, :1].transpose(1, 2).unsqueeze(2)
    )
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend, cp_context=cp_context),
    ):
        output = attention(local_hidden, torch.tensor([[0, 3]], dtype=torch.int32), torch.tensor([[True, True]]))

    mixed_qkv, beta, layer = backend.execute_linear.call_args.args
    raw_gate_proj = backend.execute_linear.call_args.kwargs["raw_gate_proj"]
    assert layer is attention
    expected_hidden = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(mixed_qkv, expected_hidden.view(1, 1, 4).expand(1, 3, 4))
    torch.testing.assert_close(raw_gate_proj, expected_hidden.view(1, 4, 1, 1) * 10)
    torch.testing.assert_close(beta, expected_hidden.view(1, 4, 1))
    assert gather.call_count == 3
    torch.testing.assert_close(attention.o_norm.call_args.args[1], local_hidden.unsqueeze(-1) * 100)
    torch.testing.assert_close(output, torch.tensor([[[101.0], [404.0]]]))


def test_kda_cp_one_keeps_full_postprocessing() -> None:
    attention = glm5_next.Glm5NextKdaAttention.__new__(glm5_next.Glm5NextKdaAttention)
    nn.Module.__init__(attention)
    attention.layer_id = 0
    attention.hidden_size = 1
    attention.head_dim = 1
    attention.qkv_dim = 1
    attention.conv_dim = 3
    attention.num_heads_local = 1
    attention.tp = 1
    attention.in_proj_qkvbfg_a = lambda hidden: torch.cat(
        (hidden.expand(-1, -1, 3), hidden, hidden, hidden),
        dim=-1,
    )
    attention.input_projection_sizes = (3, 1, 2)
    attention.forget_gate = SimpleNamespace(raw_projection=lambda hidden: hidden.unsqueeze(-1) * 10)
    attention.g_b_proj = lambda hidden: hidden * 100
    attention._fg_b_weight = None
    attention.o_norm = MagicMock(side_effect=lambda core, gate: core + gate)
    attention.o_proj = nn.Identity()
    hidden = torch.tensor([[[1.0], [2.0], [3.0]]])
    backend = MagicMock()
    backend.execute_linear.return_value = hidden.unsqueeze(2)

    with (
        patch.object(
            glm5_next,
            "get_forward_context_or_none",
            return_value=SimpleNamespace(attention_backend=backend, cp_context=None),
        ),
        patch.object(glm5_next, "cp_merge_rows") as merge_rows,
        patch.object(glm5_next, "cp_shard_rows") as shard_rows,
    ):
        output = attention(hidden, torch.tensor([[0, 1, 2]], dtype=torch.int32), torch.ones(1, 3, dtype=torch.bool))

    merge_rows.assert_not_called()
    shard_rows.assert_not_called()
    torch.testing.assert_close(output, hidden * 101)


def _cp2_padded_context() -> SimpleNamespace:
    """Global 3 tokens under zigzag cp_size=2, rank 0: one real row (global 0)
    plus one padding row. ``query_index`` is ``[0]`` -- not the identity -- so a
    bug that drops the query selection and uses every local row is observable.
    """
    return SimpleNamespace(
        cp_size=2,
        cp_rank=0,
        total_local=2,
        shard_index=torch.tensor([0, -1], dtype=torch.int64),
        shard_gather_index=torch.tensor([0, 0], dtype=torch.int64),
        shard_valid_mask=torch.tensor([True, False]),
        restore_index=torch.tensor([0, 2, 3], dtype=torch.int64),
        query_index=torch.tensor([0], dtype=torch.int64),
        q_cu_seqlens=[1],
        q_cu_seqlens_tensor=torch.tensor([1], dtype=torch.int32),
        segment_seq_indices=torch.tensor([0], dtype=torch.int64),
        segment_kv_seq_lens=[1],
        segment_kv_seq_lens_tensor=torch.tensor([1], dtype=torch.int32),
    )


class _KdaNorm(nn.Module):
    def forward(self, core: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        assert core.shape == (1, 2, 2, 2)
        assert gate.shape == (1, 2, 2, 2)
        return core


class _KdaOutputProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape: tuple[int, ...] | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.input_shape = tuple(hidden.shape)
        assert hidden.shape == (1, 2, 4)
        return torch.tensor(
            [[[1.0, 2.0, 3.0], [float("nan"), float("inf"), -float("inf")]]],
            device=hidden.device,
        )


class _MlaOutputProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape: tuple[int, ...] | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.input_shape = tuple(hidden.shape)
        assert hidden.shape == (2, 6)
        return torch.tensor(
            [[1.0, 2.0, 3.0], [float("nan"), float("inf"), -float("inf")]],
            device=hidden.device,
        )


def test_kda_cp_projects_only_local_rows_and_overwrites_padding(monkeypatch) -> None:
    attention = glm5_next.Glm5NextKdaAttention.__new__(glm5_next.Glm5NextKdaAttention)
    nn.Module.__init__(attention)
    attention.layer_id = 0
    attention.hidden_size = 3
    attention.head_dim = 2
    attention.qkv_dim = 4
    attention.conv_dim = 12
    attention.num_heads_local = 2
    attention.tp = 2
    attention.input_projection_sizes = (12, 2, 4)
    attention.in_proj_qkvbfg_a = lambda hidden: torch.cat(
        (
            hidden[..., :1].expand(-1, -1, 12),
            hidden[..., :1].expand(-1, -1, 2),
            hidden[..., :1].expand(-1, -1, 4),
        ),
        dim=-1,
    )
    attention._project_fg = lambda latents: (latents.view(1, 2, 2, 2), latents.view(1, 2, 2, 2))
    attention.o_norm = _KdaNorm()
    attention.o_proj = _KdaOutputProjection()

    cp_context = _cp2_padded_context()
    local_hidden = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]])
    global_core = torch.arange(12, dtype=torch.float32).view(1, 3, 2, 2)
    backend = MagicMock()
    backend.execute_linear.return_value = global_core
    merge = MagicMock(
        side_effect=[
            torch.ones(3, 12),
            torch.ones(3, 2, 2),
            torch.ones(3, 2),
        ]
    )

    def compensate(output: torch.Tensor) -> None:
        output.add_(7)

    with (
        patch.object(
            glm5_next,
            "get_forward_context_or_none",
            return_value=SimpleNamespace(attention_backend=backend, cp_context=cp_context),
        ),
        patch.object(glm5_next, "cp_merge_rows", merge),
        patch.object(glm5_next.distributed, "all_reduce_", side_effect=compensate, create=True) as all_reduce,
    ):
        output = attention(local_hidden, torch.tensor([[0, 0]], dtype=torch.int32), torch.tensor([[True, False]]))

    assert merge.call_count == 3
    assert backend.execute_linear.call_args.args[0].shape == (1, 12, 3)
    assert backend.execute_linear.call_args.args[1].shape == (1, 3, 2)
    assert backend.execute_linear.call_args.kwargs["raw_gate_proj"].shape == (1, 3, 2, 2)
    assert attention.o_proj.input_shape == (1, 2, 4)
    all_reduce.assert_called_once()
    torch.testing.assert_close(output[0, 0], torch.tensor([8.0, 9.0, 10.0]))
    assert torch.equal(output[0, 1], torch.zeros(3))


def test_kda_pcp_eplv2_sp_masks_padding_before_reducing() -> None:
    attention = glm5_next.Glm5NextKdaAttention.__new__(glm5_next.Glm5NextKdaAttention)
    nn.Module.__init__(attention)
    attention.layer_id = 0
    attention.hidden_size = 3
    attention.head_dim = 2
    attention.conv_dim = 12
    attention.num_heads_local = 2
    attention.input_projection_sizes = (12, 2, 4)
    attention.cfg = SimpleNamespace(hidden_size=3)
    attention.in_proj_qkvbfg_a = lambda hidden: torch.cat(
        (
            hidden[..., :1].expand(-1, -1, 12),
            hidden[..., :1].expand(-1, -1, 2),
            hidden[..., :1].expand(-1, -1, 4),
        ),
        dim=-1,
    )
    attention._project_fg = lambda latent: (latent.view(1, 2, 2, 2), latent.view(1, 2, 2, 2))
    attention.o_norm = _KdaNorm()
    attention.o_proj = _KdaOutputProjection()
    backend = MagicMock()
    backend.execute_linear.return_value = torch.arange(12, dtype=torch.float32).view(1, 3, 2, 2)
    with (
        patch.object(
            glm5_next,
            "get_forward_context_or_none",
            return_value=SimpleNamespace(attention_backend=backend, cp_context=_cp2_padded_context()),
        ),
        patch.object(
            glm5_next, "cp_merge_rows", side_effect=[torch.ones(3, 12), torch.ones(3, 2, 2), torch.ones(3, 2)]
        ),
        patch.object(glm5_next.distributed, "all_reduce_", side_effect=AssertionError("SP requires RS"), create=True),
    ):
        output = attention(
            torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]]),
            torch.tensor([[0, 0]], dtype=torch.int32),
            torch.tensor([[True, False]]),
            output_layout=TokenParallelLayout(2, 1, 0),
        )
    torch.testing.assert_close(output[0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(output[0, 1], torch.zeros(3))


def _make_mla_attention(indexer: object | None) -> glm5_next.Glm5NextMlaAttention:
    attention = glm5_next.Glm5NextMlaAttention.__new__(glm5_next.Glm5NextMlaAttention)
    nn.Module.__init__(attention)
    attention.layer_id = 0
    attention.hidden_size = 1
    attention.qk_nope_head_dim = 1
    attention.qk_rope_head_dim = 0
    attention.qk_head_dim = 1
    attention.kv_lora_rank = 1
    attention.num_heads_local = 1
    attention.v_head_dim = 1
    attention.cfg = SimpleNamespace(tp_size=1)
    attention.q_a_proj = nn.Identity()
    attention.q_a_layernorm = nn.Identity()
    attention.q_b_proj = nn.Identity()
    attention.kv_a_proj_with_mqa = nn.Identity()
    attention.kv_a_layernorm = nn.Identity()
    attention._qkv_a_weight = None
    attention.o_proj = nn.Identity()
    attention.register_buffer("W_UK", torch.ones(1, 1, 1), persistent=False)
    attention.register_buffer("W_UV", torch.ones(1, 1, 1), persistent=False)
    attention.indexer = indexer
    return attention


def test_mla_cp_one_keeps_full_postprocessing() -> None:
    attention = _make_mla_attention(MagicMock())
    attention.indexer.select_qli.return_value = torch.tensor([[[10]], [[20]], [[30]]], dtype=torch.int32)
    hidden = torch.tensor([[[1.0], [2.0], [3.0]]])
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = hidden.reshape(3, 1, 1)

    with (
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=None, attention_backend=backend),
        ),
        patch.object(glm5_next, "cp_merge_rows") as merge_rows,
        patch.object(glm5_next, "cp_shard_rows") as shard_rows,
    ):
        output, topk = attention(
            hidden,
            torch.tensor([[0, 1, 2]], dtype=torch.int32),
            torch.ones(1, 3, dtype=torch.bool),
        )

    merge_rows.assert_not_called()
    shard_rows.assert_not_called()
    torch.testing.assert_close(output, hidden.reshape(3, 1))
    torch.testing.assert_close(topk, torch.tensor([[[10]], [[20]], [[30]]], dtype=torch.int32))


def test_mla_cp_projects_global_rows_and_reshards_output() -> None:
    """The model-side CP row layout is partition-independent.

    ``Glm5NextMlaAttention`` merges to the complete logical stream, runs the
    indexer and every projection on those global rows, and reshards the
    backend output. Only the attention backend partitions the query rows, so
    the projection GEMMs keep ``M = the global token count`` exactly as in
    the ``cp_size == 1`` path.
    """
    indexer = MagicMock()
    global_topk = torch.arange(6, dtype=torch.int32).view(3, 1, 2)
    indexer.select_qli.return_value = global_topk
    attention = _make_mla_attention(indexer)
    attention.hidden_size = 3
    attention.qk_nope_head_dim = 2
    attention.qk_head_dim = 2
    attention.kv_lora_rank = 3
    attention.num_heads_local = 2
    attention.v_head_dim = 3
    attention.cfg = SimpleNamespace(tp_size=2)
    attention.q_b_proj = nn.Linear(3, 4, bias=False)
    attention.kv_a_proj_with_mqa = nn.Identity()
    attention.o_proj = _MlaOutputProjection()
    attention.W_UK = torch.ones(2, 2, 3)
    attention.W_UV = torch.ones(2, 3, 3)

    cp_context = _cp2_padded_context()
    global_hidden = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    global_positions = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    global_attn_out = torch.arange(18, dtype=torch.float32).view(3, 2, 3)
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = global_attn_out
    merge = MagicMock(side_effect=[global_hidden.squeeze(0), global_positions.reshape(-1, 1)])

    def fake_shard(value: torch.Tensor, _ctx: object) -> torch.Tensor:
        # CP-local layout: this rank's one real row plus one zeroed padding row.
        rows = value.index_select(0, torch.tensor([0]))
        padding = torch.zeros(1, *value.shape[1:], dtype=value.dtype)
        return torch.cat([rows, padding], dim=0)

    shard = MagicMock(side_effect=fake_shard)

    with (
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
        ),
        patch.object(glm5_next, "cp_merge_rows", merge),
        patch.object(glm5_next, "cp_shard_rows", shard),
        patch.object(glm5_next.distributed, "all_reduce_", create=True),
    ):
        output, topk = attention(
            torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]]),
            torch.tensor([[0, 0]], dtype=torch.int32),
            torch.tensor([[True, False]]),
        )

    # Merge to global rows, then reshard: identical for either switch value.
    assert merge.call_count == 2
    assert indexer.select_qli.call_args.args[0].shape == (1, 3, 3)
    torch.testing.assert_close(indexer.select_qli.call_args.args[0], global_hidden)
    assert backend.execute_mla.call_args.args[0].shape == (3, 2, 3)
    assert backend.execute_mla.call_args.args[2].shape == (3, 1, 3)
    torch.testing.assert_close(backend.execute_mla.call_args.kwargs["topk"], global_topk)
    assert shard.call_count == 2
    assert attention.o_proj.input_shape == (2, 6)


def test_mla_pcp_eplv2_sp_masks_padding_before_reducing_and_returns_local_topk(monkeypatch) -> None:
    indexer = MagicMock()
    global_topk = torch.tensor([[[10]], [[20]], [[30]]], dtype=torch.int32)
    indexer.select_qli.return_value = global_topk
    attention = _make_mla_attention(indexer)
    cp_context = _cp2_padded_context()
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = torch.tensor([[[11.0]], [[22.0]], [[33.0]]])
    merge = MagicMock(
        side_effect=[
            torch.tensor([[1.0], [2.0], [3.0]]),
            torch.tensor([[0], [1], [2]], dtype=torch.int32),
        ]
    )
    with (
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
        ),
        patch.object(glm5_next, "cp_merge_rows", merge),
        patch.object(
            glm5_next.distributed, "all_reduce_", side_effect=AssertionError("SP must reduce-scatter"), create=True
        ),
    ):
        output, topk = attention(
            torch.tensor([[[1.0], [0.0]]]),
            torch.tensor([[0, 0]], dtype=torch.int32),
            torch.tensor([[True, False]]),
            output_layout=TokenParallelLayout(2, 1, 0),
        )
    torch.testing.assert_close(output, torch.tensor([[11.0], [0.0]]))
    torch.testing.assert_close(topk, torch.tensor([[[10]], [[0]]], dtype=torch.int32))


def test_mla_cp_indexer_and_backend_receive_global_rows_for_each_rank(monkeypatch) -> None:
    """MLA/DSA counterpart of the KDA rank-invariance test.

    Under CP the model merges every rank up to the complete logical stream
    before touching the indexer or the backend, so both ranks hand them the
    *same* global rows. Every projection GEMM therefore runs with the global
    batch shape on every rank, exactly as in the ``cp_size == 1`` path.
    """
    indexer = MagicMock()
    indexer.select_qli.return_value = torch.zeros(4, 1, 1, dtype=torch.int32)
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = torch.zeros(4, 1, 1)
    global_hidden = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    global_positions = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    captured_by_rank: dict[int, dict[str, torch.Tensor]] = {}
    for rank, local_hidden, local_positions in (
        (0, torch.tensor([[[1.0], [4.0]]]), torch.tensor([[0, 3]], dtype=torch.int32)),
        (1, torch.tensor([[[2.0], [3.0]]]), torch.tensor([[1, 2]], dtype=torch.int32)),
    ):
        attention = _make_mla_attention(indexer)
        cp_context = _cp2_context(rank)
        merge = MagicMock(side_effect=[global_hidden.squeeze(0), global_positions.reshape(-1, 1)])
        indexer.select_qli.reset_mock()
        backend.execute_mla.reset_mock()
        with (
            patch.object(
                glm5_next,
                "get_forward_context",
                return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
            ),
            patch.object(glm5_next, "cp_merge_rows", merge),
            patch.object(glm5_next, "cp_shard_rows", MagicMock(return_value=torch.zeros(2, 1, 1))),
        ):
            attention(local_hidden, local_positions, torch.tensor([[True, True]]))
        assert merge.call_count == 2
        index_args = indexer.select_qli.call_args.args
        mla_args = backend.execute_mla.call_args.args
        captured_by_rank[rank] = {
            "index_hidden_states": index_args[0],
            "index_position_ids": index_args[2],
            "q_latent": mla_args[0],
            "k_latent_3d": mla_args[2],
        }

    for rank in (0, 1):
        torch.testing.assert_close(captured_by_rank[rank]["index_hidden_states"], global_hidden)
        torch.testing.assert_close(captured_by_rank[rank]["index_position_ids"], global_positions)
        torch.testing.assert_close(captured_by_rank[rank]["q_latent"], global_hidden.reshape(4, 1, 1))
        torch.testing.assert_close(captured_by_rank[rank]["k_latent_3d"], global_hidden.reshape(4, 1, 1))

    # The two ranks are no longer fed different token rows: global means global.
    assert torch.equal(
        captured_by_rank[0]["index_hidden_states"],
        captured_by_rank[1]["index_hidden_states"],
    )


def test_dsa_cp_full_indexer_uses_global_rows_and_reshards_topk(monkeypatch) -> None:
    """A full-indexer DSA layer under CP merges to global rows first: the
    indexer sees the complete logical stream, the MLA backend sees global
    q/kv/topk (so its projections keep ``M = T_global``), and the returned
    top-k is resharded to CP-local for the next shared-indexer layer.
    ``previous_topk_indices`` is ignored because a full indexer recomputes.
    """
    indexer = MagicMock()
    global_topk = torch.tensor([[[10]], [[20]], [[30]], [[40]]], dtype=torch.int32)
    indexer.select_qli.return_value = global_topk
    attention = _make_mla_attention(indexer)
    cp_context = _cp2_context(0)
    previous_topk = torch.tensor([[[50]], [[80]]], dtype=torch.int32)
    global_hidden = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    global_positions = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = torch.tensor([[[11.0]], [[22.0]], [[33.0]], [[44.0]]])

    with (
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
        ),
        patch.object(
            glm5_next,
            "cp_merge_rows",
            MagicMock(side_effect=[global_hidden.squeeze(0), global_positions.reshape(-1, 1)]),
        ) as merge,
        patch.object(
            glm5_next,
            "cp_shard_rows",
            side_effect=lambda value, _ctx: value.index_select(0, torch.tensor([0, 3])),
        ) as shard,
    ):
        output, topk = attention(
            torch.tensor([[[1.0], [4.0]]]),
            torch.tensor([[0, 3]], dtype=torch.int32),
            torch.tensor([[True, True]]),
            previous_topk,
        )

    # Hidden + positions are merged; a full indexer never merges prev top-k.
    assert merge.call_count == 2
    assert shard.call_count == 2
    index_args = indexer.select_qli.call_args.args
    torch.testing.assert_close(index_args[0], global_hidden)
    torch.testing.assert_close(index_args[1], global_hidden.reshape(4, 1))
    torch.testing.assert_close(index_args[2], global_positions)
    torch.testing.assert_close(index_args[3], torch.ones(1, 4, dtype=torch.bool))
    torch.testing.assert_close(backend.execute_mla.call_args.args[0], global_hidden.reshape(4, 1, 1))
    torch.testing.assert_close(backend.execute_mla.call_args.kwargs["topk"], global_topk)
    torch.testing.assert_close(output, torch.tensor([[11.0], [44.0]]))
    torch.testing.assert_close(topk, global_topk.index_select(0, torch.tensor([0, 3])))


def test_dsa_cp_shared_indexer_merges_local_topk_back_to_global(monkeypatch) -> None:
    """A shared-indexer DSA layer receives the preceding layer's CP-local top-k,
    merges it (with hidden and positions) back to global rows, and hands the
    global tensor to the backend. The returned top-k is resharded again for the
    next shared layer.
    """
    attention = _make_mla_attention(None)
    cp_context = _cp2_context(0)
    local_topk = torch.tensor([[[10]], [[40]]], dtype=torch.int32)
    global_topk = torch.tensor([[[10]], [[20]], [[30]], [[40]]], dtype=torch.int32)
    global_hidden = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    global_positions = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    backend = MagicMock()
    backend.execute_mla.return_value = torch.tensor([[[11.0]], [[22.0]], [[33.0]], [[44.0]]])

    with (
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
        ),
        patch.object(
            glm5_next,
            "cp_merge_rows",
            MagicMock(
                side_effect=[
                    global_hidden.squeeze(0),
                    global_positions.reshape(-1, 1),
                    global_topk.reshape(4, -1),
                ]
            ),
        ) as merge,
        patch.object(
            glm5_next,
            "cp_shard_rows",
            side_effect=lambda value, _ctx: value.index_select(0, torch.tensor([0, 3])),
        ),
    ):
        output, topk = attention(
            torch.tensor([[[1.0], [4.0]]]),
            torch.tensor([[0, 3]], dtype=torch.int32),
            torch.tensor([[True, True]]),
            local_topk,
        )

    assert merge.call_count == 3  # hidden + positions + the shared top-k
    torch.testing.assert_close(backend.execute_mla.call_args.kwargs["topk"], global_topk)
    torch.testing.assert_close(output, torch.tensor([[11.0], [44.0]]))
    torch.testing.assert_close(topk, local_topk)


# ---------------------------------------------------------------------------
# C1: the indexer under CP -- it receives global rows, so the key-side cache
# write and the query-side top-k both run on the complete logical stream.
# ---------------------------------------------------------------------------


def _make_simple_indexer() -> glm5_next.Glm5NextIndexer:
    """A real ``Glm5NextIndexer`` with the non-compressed (index_kpool=1) path,
    which is the smallest body that still exercises the key-side write and the
    query-side reshaping without NPU kPool kernels."""
    indexer = glm5_next.Glm5NextIndexer.__new__(glm5_next.Glm5NextIndexer)
    nn.Module.__init__(indexer)
    indexer.layer_id = 0
    indexer.n_heads = 1
    indexer.head_dim = 1
    indexer.topk = 1
    indexer.index_kpool = 1
    indexer.index_kpool_compress = False
    indexer.index_kpool_always_select_tail = False
    indexer._uses_npu_compressed_tail = False
    indexer._update_compact_kpool = None
    indexer._compact_kpool_model_compatible = False
    indexer.softmax_scale = 1.0
    indexer.wq_b = nn.Linear(1, 1, bias=False)
    indexer.wk = nn.Linear(1, 1, bias=False)
    indexer.weights_proj = nn.Linear(1, 1, bias=False)
    for module in (indexer.wq_b, indexer.wk, indexer.weights_proj):
        with torch.no_grad():
            module.weight.fill_(1.0)
    indexer.k_norm = nn.Identity()
    indexer.register_buffer("_wk_weights_weight", None, persistent=False)
    indexer.index_kpool_compress_ape = torch.zeros(1, 1)
    indexer.index_kpool_compress_gate = torch.zeros(1, 1)
    return indexer


def _index_context_recording(cp_context, recorded: dict[str, torch.Tensor]) -> SimpleNamespace:
    def update_index_cache(values: torch.Tensor, scales: torch.Tensor | None) -> None:
        del scales
        recorded["packed"] = values.detach().clone()

    return SimpleNamespace(
        block_table=torch.tensor([[0]], dtype=torch.int32),
        actual_seq_kv=torch.tensor([4], dtype=torch.int64),
        kpool_tail=None,
        kpool_tail_read_indices=None,
        kpool_tail_write_indices=None,
        slot_mapping=torch.arange(4, dtype=torch.int64),
        index_cache=torch.zeros(1, _MLA_HANDOFF_BLOCK_SIZE, 1, 1),
        index_cache_scale=None,
        cp_context=cp_context,
        kpool_query_lens=(),
        kpool_query_lens_device=None,
        kpool_cache_triton_compatible=False,
        has_kv_shard=False,
        materialized_block_table=None,
        localize_pool_block_table=lambda table: table,
        update_index_cache=update_index_cache,
    )


def test_dsa_cp_indexer_writes_global_key_and_selects_global_query_rows(monkeypatch) -> None:
    """The real ``Glm5NextIndexer.select_qli`` receives global rows under CP
    (the owning MLA layer merged them), so it must:
    (a) write the index cache in **global** token order with no extra gather,
    (b) run the top-k selection on those same global rows, and
    (c) return a global-layout top-k for the backend to partition.
    """
    indexer = _make_simple_indexer()
    cp_context = _cp2_context(0)
    recorded: dict[str, torch.Tensor] = {}
    ctx = _index_context_recording(cp_context, recorded)
    backend = MagicMock()
    backend.gather_index_history.return_value = torch.zeros(1, 4, 1)
    select_topk = MagicMock(return_value=torch.zeros(1, 4, 1, dtype=torch.long))
    indexer.select_topk = select_topk

    topk = indexer.select_qli(
        hidden_states=torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
        qr=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        positions=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        attention_mask=torch.tensor([[True, True, True, True]]),
        ctx=ctx,
        layer=None,
        backend=backend,
    )

    # (a) the global key side reaches the cache verbatim.
    torch.testing.assert_close(recorded["packed"].reshape(-1), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    # (b) the query side is the same global rows.
    select_args = select_topk.call_args.args
    torch.testing.assert_close(select_args[1].reshape(-1), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(
        select_topk.call_args.kwargs["query_positions"].reshape(-1),
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    # (c) the returned top-k is in the global [T_global] layout.
    assert topk.shape == (4, 1, 1)
    assert topk.dtype == torch.int32


def test_dsa_indexer_packs_global_rows_per_request_for_varlen_batch(monkeypatch) -> None:
    """The per-request query packing (``is_varlen`` branch of ``select_qli``)
    maps each request's **global** query rows to the right padded window slot,
    masks the pad rows, and gathers the surviving top-k rows back to the flat
    global order.

    Under CP the model has merged to global rows, so these are the original
    request boundaries (7 global tokens split 5/2). A request-blind layout --
    for example assuming every row belongs to the first request -- misplaces
    both the inputs and the returned top-k.
    """
    indexer = _make_simple_indexer()
    recorded: dict[str, torch.Tensor] = {}
    ctx = _index_context_recording(None, recorded)
    ctx.slot_mapping = torch.arange(7, dtype=torch.int64)
    ctx.actual_seq_kv = torch.tensor([5, 2], dtype=torch.int64)
    backend = MagicMock()
    backend.gather_index_history.return_value = torch.zeros(2, 5, 1)
    select_topk = MagicMock(return_value=torch.arange(10, dtype=torch.long).view(2, 5, 1))
    indexer.select_topk = select_topk

    hidden = torch.arange(10.0, 17.0).view(1, 7, 1)
    with patch.object(
        glm5_next,
        "get_forward_context",
        return_value=SimpleNamespace(metadata=SimpleNamespace(q_cu_seq_lens=torch.tensor([0, 5, 7]))),
    ):
        topk = indexer.select_qli(
            hidden_states=hidden,
            qr=hidden.reshape(7, 1),
            positions=torch.tensor([[0, 1, 2, 3, 4, 0, 1]], dtype=torch.int32),
            attention_mask=torch.ones(1, 7, dtype=torch.bool),
            ctx=ctx,
            layer=None,
            backend=backend,
        )

    # Padded window [num_seqs=2, max_q=5]: request 0 -> rows 10..14, request 1 ->
    # rows 15/16 with the last three slots masked (its real run is 2 rows).
    torch.testing.assert_close(
        select_topk.call_args.args[0],
        torch.tensor(
            [
                [[10.0], [11.0], [12.0], [13.0], [14.0]],
                [[15.0], [16.0], [16.0], [16.0], [16.0]],
            ]
        ),
    )
    torch.testing.assert_close(select_topk.call_args.args[1], select_topk.call_args.args[0])
    assert torch.equal(
        select_topk.call_args.args[2],
        torch.tensor([[True, True, True, True, True], [True, True, False, False, False]]),
    )
    torch.testing.assert_close(
        select_topk.call_args.kwargs["query_positions"],
        torch.tensor([[0, 1, 2, 3, 4], [0, 1, 1, 1, 1]], dtype=torch.int32),
    )
    # The seven real top-k rows survive the packed window and land back in the
    # flat global order.
    assert topk.shape == (7, 1, 1)
    assert topk.dtype == torch.int32
    assert torch.equal(topk, torch.arange(7, dtype=torch.int32).view(7, 1, 1))


# ---------------------------------------------------------------------------
# B1 (implement.md): GLM-5-Next model-level CP + aux-hidden-capture restore.
#
# Mirrors DSV4's
# ``test_layer_capture_restores_global_rows_with_context_parallelism``
# (test_deepseek_v4_model.py). ``Glm5NextModel.forward`` packs the exit hidden
# and the auxiliary capture into a single ``cp_merge_rows`` (one CP
# all-gather), so this section exercises the fused wiring end-to-end (real
# ``AuxHiddenCapture``/``cp_merge_rows``, mocked only at the
# ``distributed.all_gather`` collective): the fused result must stay
# element-wise equal to the former two-merge path, including the empty-local
# and capture-disabled boundaries.
# ---------------------------------------------------------------------------


def _make_model_with_capture(
    events: list[str], layers_to_capture: tuple[int, ...], hidden_size: int = 1
) -> tuple[glm5_next.Glm5NextModel, list[_DecoderLayer]]:
    """Like ``_make_model``, but wires ``aux_hidden_capture`` the way the real
    ``Glm5NextModel.__init__`` does (mean-collapse over the mHC streams via
    ``hc_head``) instead of the empty, capture-disabled default."""
    model, layers = _make_model(events)
    model.cfg.hidden_size = hidden_size
    model.cfg.layers_to_capture = layers_to_capture
    hc_head = model.hc_head
    model.aux_hidden_capture = AuxHiddenCapture(
        layers_to_capture,
        transform=lambda streams: hc_head(streams).reshape(-1, hidden_size),
    )
    return model, layers


def test_layer_capture_restores_global_rows_with_context_parallelism(monkeypatch) -> None:
    """No-padding case: 4 global tokens, zigzag cp_size=2, rank 0's local
    shard (global tokens 0 and 3; rank 1 owns 1 and 2 -- see ``_cp2_context``).

    Each mock ``_DecoderLayer`` adds ``layer_id + 1`` to its input, so after
    both layers a global token's final hidden value is ``value + 3`` and its
    captured (layer-0-only) auxiliary value is ``value + 1``. The model exit
    packs ``h`` and ``aux_hidden_buffer`` into a single all-gather, so
    ``h_gathered``/``aux_gathered`` below are supplied as one fused rank-major
    concatenation (rank 0's two local rows, then rank 1's two local rows, with
    the two tensors side by side on the feature axis); the test asserts the
    local shard fed to that collective is ``[h_local | aux_local]`` and that
    ``cp_merge_rows``' ``restore_index`` reassembles both parts back into
    original global token order [10, 20, 30, 40].
    """
    events: list[str] = []
    model, layers = _make_model_with_capture(events, layers_to_capture=(0,))
    del layers
    cp_context = _cp2_context(0)

    h_gathered = torch.tensor([[13.0], [43.0], [23.0], [33.0]])
    aux_gathered = torch.tensor([[11.0], [41.0], [21.0], [31.0]])
    fused_gathered = torch.cat([h_gathered, aux_gathered], dim=-1)
    gather = MagicMock(return_value=fused_gathered)
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)

    with patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    assert isinstance(output, tuple)
    hidden, aux_hidden = output
    torch.testing.assert_close(hidden, torch.tensor([[13.0], [23.0], [33.0], [43.0]]))
    torch.testing.assert_close(aux_hidden, torch.tensor([[11.0], [21.0], [31.0], [41.0]]))
    # One collective instead of two, carrying rank 0's local rows (globals 0
    # and 3) with h and aux packed on the feature axis.
    assert gather.call_count == 1
    torch.testing.assert_close(gather.call_args_list[0].args[0], torch.tensor([[13.0, 11.0], [43.0, 41.0]]))


def test_layer_capture_restores_global_rows_with_context_parallelism_padding_row(monkeypatch) -> None:
    """Padding-row case (mirrors DSV4's own parametrization, which covers a
    token count that does not divide evenly across CP ranks): 3 global
    tokens under zigzag cp_size=2 leave rank 0 with one real row and one
    padding row (``_cp2_padded_context``). The padding row's (garbage) value
    must never leak into ``restore_index``'s output -- ``restore_index``
    (``[0, 2, 3]``) never references gathered row 1, the padding slot.
    """
    events: list[str] = []
    model, layers = _make_model_with_capture(events, layers_to_capture=(0,))
    del layers
    cp_context = _cp2_padded_context()

    # Rank 0's local padding row (gathered index 1) carries a deterministic
    # but otherwise-irrelevant value; it is never selected by restore_index.
    h_gathered = torch.tensor([[13.0], [3.0], [23.0], [33.0]])
    aux_gathered = torch.tensor([[11.0], [1.0], [21.0], [31.0]])
    fused_gathered = torch.cat([h_gathered, aux_gathered], dim=-1)
    gather = MagicMock(return_value=fused_gathered)
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)

    with patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)):
        output = model(
            torch.tensor([10, 20, 30]),
            torch.tensor([0, 1, 2], dtype=torch.int32),
        )

    assert isinstance(output, tuple)
    hidden, aux_hidden = output
    torch.testing.assert_close(hidden, torch.tensor([[13.0], [23.0], [33.0]]))
    torch.testing.assert_close(aux_hidden, torch.tensor([[11.0], [21.0], [31.0]]))
    assert gather.call_count == 1
    torch.testing.assert_close(gather.call_args_list[0].args[0], torch.tensor([[13.0, 11.0], [3.0, 1.0]]))


def _cp2_empty_local_context() -> SimpleNamespace:
    """Zigzag cp_size=2 plan with no local rows at all (``T_local == 0``).

    ``total_local`` is identical on every CP rank (the sharding plan's
    contract), so an empty local shard is a group-wide state: every rank has
    to enter the exit all-gather even then, or the whole group deadlocks.
    """
    empty_indices = torch.empty(0, dtype=torch.int64)
    return SimpleNamespace(
        cp_size=2,
        cp_rank=0,
        total_local=0,
        shard_index=empty_indices,
        shard_gather_index=empty_indices,
        shard_valid_mask=torch.empty(0, dtype=torch.bool),
        restore_index=empty_indices,
        query_index=empty_indices,
        q_cu_seqlens=[],
        q_cu_seqlens_tensor=torch.empty(0, dtype=torch.int32),
        segment_seq_indices=empty_indices,
        segment_kv_seq_lens=[],
        segment_kv_seq_lens_tensor=torch.empty(0, dtype=torch.int32),
    )


def _patch_recording_cp_all_gather(monkeypatch: pytest.MonkeyPatch, peer_scale: float = 10.0) -> list[torch.Tensor]:
    """Patch ``distributed.all_gather`` with a recording, deterministic 2-rank
    collective and return the gathered inputs in call order.

    The peer rank's shard is ``shard * peer_scale``: arbitrary but fully
    deterministic and shape-compatible, so every merged element is specified
    by this rank's local shard alone. Only the axis semantics of the
    collective are under test; the peer payload just has to be distinct.
    """
    gathered_inputs: list[torch.Tensor] = []

    def all_gather(value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0
        assert world_size == 2
        assert group_name == "cp"
        gathered_inputs.append(value)
        return torch.cat([value, value * peer_scale], dim=dim)

    monkeypatch.setattr(glm5_next.distributed, "all_gather", all_gather, raising=False)
    return gathered_inputs


def test_model_exit_fuses_cp_merges_into_one_all_gather(monkeypatch) -> None:
    """The model exit issues one all-gather, element-wise equal to merging
    ``h`` and the auxiliary capture separately.

    ``h`` is ``[T_local, D]`` and the capture is ``[T_local, D * k]`` over the
    same local rows (``k = 2`` here). The reference is the pre-fusion code
    path replayed through the real ``cp_merge_rows``; because the packing
    happens on the feature axis and the merge on the row axis, both the
    intermediate packed shard and the split result have to match exactly.
    """
    hidden_size = 2
    captured_layers = (0, 1)  # k = 2 -> auxiliary width 4
    events: list[str] = []
    model, _ = _make_model_with_capture(events, layers_to_capture=captured_layers, hidden_size=hidden_size)
    cp_context = _cp2_context(0)
    gathered_inputs = _patch_recording_cp_all_gather(monkeypatch)
    # Global tokens 0..3 carry [value, value + 0.5]. The capture is a mean over
    # the identical mHC streams; the two mock layers add +1 and +2, so rank 0's
    # local rows (globals 0 and 3) hold ``embed + 3`` for the exit hidden and
    # ``[embed + 1 | embed + 3]`` for the two captured slots.
    embeds = torch.tensor([[[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]])
    local_h = torch.tensor([[3.0, 3.5], [6.0, 6.5]])
    local_aux = torch.tensor([[1.0, 1.5, 3.0, 3.5], [4.0, 4.5, 6.0, 6.5]])

    # "Two merges separately": the previous exit path, through the real merge.
    expected_h = glm5_next.cp_merge_rows(local_h, cp_context)
    expected_aux = glm5_next.cp_merge_rows(local_aux, cp_context)

    model._inputs_embeds = embeds
    with patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)):
        output = model(
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    assert isinstance(output, tuple)
    hidden, aux_hidden = output
    assert torch.equal(hidden, expected_h)
    assert torch.equal(aux_hidden, expected_aux)
    assert hidden.shape == (4, hidden_size)
    assert aux_hidden.shape == (4, hidden_size * len(captured_layers))
    # Two reference merges, then the single fused exit merge.
    assert len(gathered_inputs) == 3
    reference_h, reference_aux, fused_input = gathered_inputs
    assert torch.equal(reference_h, local_h)
    assert torch.equal(reference_aux, local_aux)
    assert fused_input.shape == (2, hidden_size * (1 + len(captured_layers)))
    assert torch.equal(fused_input, torch.cat([reference_h, reference_aux], dim=-1))


def test_model_exit_empty_local_rank_still_joins_the_all_gather(monkeypatch) -> None:
    """``T_local == 0`` must not turn into a skipped collective.

    The fused concat is ``[0, D]`` with ``[0, D * k]`` -> ``[0, D * (1 + k)]``;
    both ``all_gather`` and ``index_select`` handle the empty row axis, and the
    rank still enters the collective exactly once. Every rank sees the same
    empty shard, so an early return here would hang the CP group.
    """
    events: list[str] = []
    model, _ = _make_model_with_capture(events, layers_to_capture=(0,))
    cp_context = _cp2_empty_local_context()
    gathered_inputs = _patch_recording_cp_all_gather(monkeypatch)
    expected_h = glm5_next.cp_merge_rows(torch.empty(0, 1), cp_context)
    expected_aux = glm5_next.cp_merge_rows(torch.empty(0, 1), cp_context)

    with patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)):
        output = model(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int32))

    assert isinstance(output, tuple)
    hidden, aux_hidden = output
    assert hidden.shape == (0, 1)
    assert aux_hidden.shape == (0, 1)
    assert torch.equal(hidden, expected_h)
    assert torch.equal(aux_hidden, expected_aux)
    # Two reference merges, then the exit merge the empty rank still joins.
    assert len(gathered_inputs) == 3
    assert gathered_inputs[2].shape == (0, 2)
    assert torch.equal(gathered_inputs[2], torch.cat([gathered_inputs[0], gathered_inputs[1]], dim=-1))


def test_model_exit_without_aux_capture_keeps_a_single_merge(monkeypatch) -> None:
    """No capture buffer -> the exit falls back to merging ``h`` alone."""
    events: list[str] = []
    model, _ = _make_model(events)  # AuxHiddenCapture(()) -> create_buffer returns None
    cp_context = _cp2_context(0)
    gathered_inputs = _patch_recording_cp_all_gather(monkeypatch)
    # Rank 0's local rows are globals 0 and 3: values [10, 40] plus 2 layers.
    expected = glm5_next.cp_merge_rows(torch.tensor([[13.0], [43.0]]), cp_context)

    with patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=cp_context)):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    assert not isinstance(output, tuple)
    assert torch.equal(output, expected)
    assert len(gathered_inputs) == 2
    assert gathered_inputs[1].shape == (2, 1)  # h only: no fused width


def test_cp_one_preserves_full_rows_without_shard_or_merge(monkeypatch) -> None:
    events: list[str] = []
    model, layers = _make_model(events)
    # The CP package is stubbed on this host, so the collective only exists
    # once a test installs it; it must stay untouched for cp_size == 1.
    gather = MagicMock()
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)
    with (
        patch.object(glm5_next, "get_forward_context", return_value=SimpleNamespace(cp_context=None)),
        patch.object(glm5_next, "cp_shard_rows") as shard_rows,
        patch.object(glm5_next, "cp_shard_positions") as shard_positions,
        patch.object(glm5_next, "cp_merge_rows") as merge_rows,
    ):
        output = model(
            torch.tensor([10, 20, 30, 40]),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    shard_rows.assert_not_called()
    shard_positions.assert_not_called()
    merge_rows.assert_not_called()
    gather.assert_not_called()
    assert events == ["embedding", "layer_0", "layer_1", "norm"]
    torch.testing.assert_close(layers[0].positions, torch.tensor([[0, 1, 2, 3]], dtype=torch.int32))
    torch.testing.assert_close(output, torch.tensor([[13.0], [23.0], [33.0], [43.0]]))


# ---------------------------------------------------------------------------
# PCP x PD prefill-role evidence (from the pcp-pd-prefill-role branch):
# cross-rank rank-invariance of both persistent-cache writers, and the
# prefill-to-decode handoff parity suites with CPU reference kernels.
# ---------------------------------------------------------------------------


def test_kda_cp_backend_inputs_are_identical_regardless_of_local_rank(monkeypatch) -> None:
    """Every CP rank must feed the backend the exact same globally-merged
    tensors for the same logical batch — this is the R3/design.md §2 invariant
    ("every CP rank ends its forward holding the complete, globally-correct
    persistent-cache-write input") that Capability A's gate relaxation
    depends on. The existing sibling tests in this file only exercise
    ``_cp2_context(0)``; this test drives the *same* real (unmocked)
    ``Glm5NextKdaAttention.forward`` through both CP ranks of a zigzag
    ``cp_size=2`` batch and asserts the backend receives byte-identical
    ``mixed_qkv``/``beta``/``raw_gate_proj`` regardless of which physical rank
    ran the forward — i.e. the persistent linear-state write is rank-
    invariant, so a non-CP Decode instance pulling from any Prefill CP rank's
    cache (design.md §2.2) sees the same, globally-correct data.
    """
    # Global tokens [0, 1, 2, 3] with hidden values [1.0, 2.0, 3.0, 4.0].
    # Zigzag cp_size=2: rank 0 owns tokens [0, 3], rank 1 owns tokens [1, 2].
    # A real ``all_gather`` returns the same rank-major concatenation to every
    # caller; the fake below keys each call's local shard by its contents and
    # rebuilds the rank-major pair, so the gathered tensors -- and therefore
    # everything the backend receives -- are a function of the rows the
    # forward actually passed, not precomputed constants.
    mixed_r0 = torch.tensor([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]])
    mixed_r1 = torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    g_raw_r0 = torch.tensor([[[10.0]], [[40.0]]])
    g_raw_r1 = torch.tensor([[[20.0]], [[30.0]]])
    # beta_raw reaches the merge pre-activation: the recurrent kernel fuses
    # the beta sigmoid in-kernel (see the forward's comment), so the backend
    # must receive the merged raw beta, not a sigmoid of it.
    beta_r0 = torch.tensor([[1.0], [4.0]])
    beta_r1 = torch.tensor([[2.0], [3.0]])
    shards_by_local: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}
    shards_by_local.update(_shards_by_local(mixed_r0, mixed_r1))
    shards_by_local.update(_shards_by_local(g_raw_r0, g_raw_r1))
    shards_by_local.update(_shards_by_local(beta_r0, beta_r1))
    gather = _patch_cp_gather_rank_major(monkeypatch, shards_by_local)

    backend = MagicMock()
    backend.execute_linear.return_value = torch.zeros(1, 4, 1, 1)

    captured_by_rank: dict[int, dict[str, torch.Tensor]] = {}
    for rank, local_hidden in ((0, torch.tensor([[[1.0], [4.0]]])), (1, torch.tensor([[[2.0], [3.0]]]))):
        attention = _make_kda_attention_for_cross_rank_test()
        cp_context = _cp2_context(rank)
        backend.execute_linear.reset_mock()
        with patch.object(
            glm5_next,
            "get_forward_context_or_none",
            return_value=SimpleNamespace(attention_backend=backend, cp_context=cp_context),
        ):
            attention(local_hidden, torch.tensor([[0, 3]], dtype=torch.int32), torch.tensor([[True, True]]))
        mixed_qkv, beta, layer = backend.execute_linear.call_args.args
        assert layer is attention
        captured_by_rank[rank] = {
            "mixed_qkv": mixed_qkv,
            "beta": beta,
            "raw_gate_proj": backend.execute_linear.call_args.kwargs["raw_gate_proj"],
        }

    assert gather.call_count == 6
    torch.testing.assert_close(captured_by_rank[0]["mixed_qkv"], captured_by_rank[1]["mixed_qkv"])
    torch.testing.assert_close(captured_by_rank[0]["beta"], captured_by_rank[1]["beta"])
    torch.testing.assert_close(captured_by_rank[0]["raw_gate_proj"], captured_by_rank[1]["raw_gate_proj"])
    expected_hidden = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(captured_by_rank[0]["mixed_qkv"], expected_hidden.view(1, 1, 4).expand(1, 3, 4))
    torch.testing.assert_close(captured_by_rank[0]["raw_gate_proj"], expected_hidden.view(1, 4, 1, 1) * 10)
    # The merged raw beta: the sigmoid is fused inside the recurrent kernel,
    # so the persistent write the backend receives carries the pre-activation
    # rows in global order.
    torch.testing.assert_close(captured_by_rank[0]["beta"], expected_hidden.view(1, 4, 1))


# ---------------------------------------------------------------------------
# A3: CP-Prefill -> non-CP-Decode cache-handoff correctness (design.md §2.2).
#
# The rank-invariance tests above prove CP ranks feed the backend identical
# *inputs*. This section proves the stronger, distinct property design.md §2.2
# and R3 actually claim: a CP-Prefill's resulting *persistent cache state*,
# handed to a cp_size=1 Decode continuation, produces the same output as a
# plain cp_size=1 Prefill+Decode baseline. Everything here drives the real
# (unmocked) ``Glm5NextKdaAttention.forward``/``execute_linear`` and a real
# ``NpuPagedAttentionBackend`` with real CPU ``conv``/``ssm`` cache tensors,
# following ``test_glm53_linear_state_io.py``'s pattern; only the innermost
# NPU-only kernels (``chunk_kda_fwd``/``recurrent_kda``/native conv1d) are
# replaced by deterministic, content-sensitive CPU reference implementations
# (not exact hardware-numerics reproductions -- see docstring below).
# ---------------------------------------------------------------------------


def test_mla_cp_backend_inputs_are_identical_regardless_of_local_rank(monkeypatch) -> None:
    """MLA/DSA analog of ``test_kda_cp_backend_inputs_are_identical_regardless_of_local_rank``.

    R3's research (``research/r3-disagg-pd-pcp-feasibility.md`` §2.1) found
    the same "merge-before-write, fully-replicated" structural mechanism
    (``cp_merge_rows`` before the backend call) holds for
    ``Glm5NextMlaAttention``'s KV/MLA-latent cache write and raw index-cache
    write exactly as it does for KDA's ``linear_state`` write — both call
    ``cp_merge_rows`` on the CP-sharded local input before handing it to the
    backend/indexer. This test drives the same real (unmocked)
    ``Glm5NextMlaAttention.forward`` through both CP ranks of a zigzag
    ``cp_size=2`` batch and asserts the indexer (the raw index-cache/kPool
    writer) and ``backend.execute_mla`` (the MLA-latent-cache writer) receive
    byte-identical merged inputs regardless of which physical rank ran the
    forward — i.e. both persistent-cache writes are rank-invariant, so a
    non-CP Decode instance pulling from any Prefill CP rank's cache
    (design.md §2.2) sees the same, globally-correct data.
    """
    # Global tokens [0, 1, 2, 3] with hidden values [1.0, 2.0, 3.0, 4.0].
    # Zigzag cp_size=2: rank 0 owns tokens [0, 3], rank 1 owns tokens [1, 2].
    # A real ``all_gather`` returns the same rank-major concatenation to every
    # caller; the fake below keys each call's local shard by its contents and
    # rebuilds the rank-major pair, so the merged tensors the indexer and the
    # backend receive are a function of the rows the forward actually passed.
    hidden_r0 = torch.tensor([[1.0], [4.0]])
    hidden_r1 = torch.tensor([[2.0], [3.0]])
    positions_r0 = torch.tensor([[0], [3]], dtype=torch.int32)
    positions_r1 = torch.tensor([[1], [2]], dtype=torch.int32)
    # Two cp_merge_rows calls per forward (hidden_states, position_ids) since
    # this attention has its own indexer (no prev_topk_indices merge).
    shards_by_local: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}
    shards_by_local.update(_shards_by_local(hidden_r0, hidden_r1))
    shards_by_local.update(_shards_by_local(positions_r0, positions_r1))
    gather = _patch_cp_gather_rank_major(monkeypatch, shards_by_local)

    indexer = MagicMock()
    indexer.select_qli.return_value = torch.zeros(4, 1, 1, dtype=torch.int32)
    backend = MagicMock()
    backend.mla_index_context.return_value = object()
    backend.execute_mla.return_value = torch.zeros(4, 1, 1)

    captured_by_rank: dict[int, dict[str, torch.Tensor]] = {}
    for rank, local_hidden, local_positions in (
        (0, torch.tensor([[[1.0], [4.0]]]), torch.tensor([[0, 3]], dtype=torch.int32)),
        (1, torch.tensor([[[2.0], [3.0]]]), torch.tensor([[1, 2]], dtype=torch.int32)),
    ):
        attention = _make_mla_attention(indexer)
        cp_context = _cp2_context(rank)
        indexer.select_qli.reset_mock()
        backend.execute_mla.reset_mock()
        with patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
        ):
            attention(local_hidden, local_positions, torch.tensor([[True, True]]))
        index_args = indexer.select_qli.call_args.args
        mla_args = backend.execute_mla.call_args.args
        captured_by_rank[rank] = {
            "index_hidden_states": index_args[0],
            "index_position_ids": index_args[2],
            "q_latent": mla_args[0],
            "k_latent_3d": mla_args[2],
        }

    assert gather.call_count == 4
    for key in ("index_hidden_states", "index_position_ids", "q_latent", "k_latent_3d"):
        torch.testing.assert_close(captured_by_rank[0][key], captured_by_rank[1][key])
    expected_hidden = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(captured_by_rank[0]["index_hidden_states"].reshape(-1), expected_hidden)
    torch.testing.assert_close(captured_by_rank[0]["k_latent_3d"].reshape(-1), expected_hidden)


# ---------------------------------------------------------------------------
# A3 (MLA/DSA analog of the KDA handoff test above).
# ---------------------------------------------------------------------------

_MLA_HANDOFF_BLOCK_SIZE = 8


def test_kda_cp_prefill_to_decode_handoff_matches_noncp_baseline(monkeypatch, causal_conv1d_reference) -> None:
    """A3 (implement.md): a CP-Prefill's resulting cache state, handed off to
    a cp_size=1 Decode continuation, must produce the same output as a plain
    cp_size=1 Prefill+Decode baseline for the same prompt.

    This is a distinct, stronger property than the rank-invariance tests
    above: those prove CP ranks compute byte-identical backend *inputs*; this
    proves the resulting persistent *cache* is correct enough to continue
    generation correctly, by actually running the real backend's
    ``execute_linear`` (real conv1d + delta-rule state I/O, real
    ``conv``/``ssm`` cache tensors) for both the CP-Prefill and the baseline,
    then simulating the PD handoff described in design.md §2.2 by copying
    rank-0's post-Prefill cache tensors -- the shard
    ``pull_kv_blocks``' rank arithmetic actually transfers -- into a fresh
    cp_size=1 Decode worker's identical cache slots.

    ``causal_conv1d_reference`` (conftest.py) supplies the same CPU conv1d
    reference every other pure-Python KDA test in this repository uses (e.g.
    ``test_glm53_linear_state_io.py``); ``_install_kda_reference_kernels``
    supplies the delta-rule stand-in this test file needs on top of it (see
    its docstring for why exact hardware fidelity is not required).
    """
    del causal_conv1d_reference  # fixture side effect (patches native conv1d) is what's needed
    _install_kda_reference_kernels(monkeypatch)

    # ---- Baseline: plain cp_size=1 Prefill (4 tokens) + Decode (1 token). ----
    conv_cache_baseline = torch.zeros(1, 1, 3, dtype=torch.float32)
    ssm_cache_baseline = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    backend_baseline = _kda_backend_with_cache(conv_cache_baseline, ssm_cache_baseline)
    backend_baseline._metadata = _kda_prefill_metadata(4)
    attention_baseline = _make_kda_attention_for_handoff_test()
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend_baseline, cp_context=None),
    ):
        attention_baseline(
            torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
            torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            torch.ones(1, 4, dtype=torch.bool),
        )
    conv_cache_baseline_postprefill = conv_cache_baseline.clone()
    ssm_cache_baseline_postprefill = ssm_cache_baseline.clone()
    baseline_decode_output = _run_kda_decode_step(conv_cache_baseline, ssm_cache_baseline)

    # ---- CP case: same prompt's Prefill under cp_size=2, rank 0. Per
    # design.md §2.2, only rank 0's cache is what a cp_size=1 Decode instance
    # actually pulls -- rank 1's cache is proven byte-identical to it by the
    # rank-invariance test above, so re-deriving it here would not exercise
    # anything new. ----
    conv_cache_rank0 = torch.zeros(1, 1, 3, dtype=torch.float32)
    ssm_cache_rank0 = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    _run_kda_cp_rank_prefill(
        0,
        torch.tensor([[[1.0], [4.0]]]),
        torch.tensor([[0, 3]], dtype=torch.int32),
        conv_cache_rank0,
        ssm_cache_rank0,
        monkeypatch,
    )

    # Per design.md §2.2, rank 0's cache is the one a cp_size=1 Decode
    # instance's `pull_kv_blocks` rank arithmetic actually transfers; it must
    # exactly match the non-CP baseline's post-Prefill cache.
    torch.testing.assert_close(conv_cache_rank0, conv_cache_baseline_postprefill)
    torch.testing.assert_close(ssm_cache_rank0, ssm_cache_baseline_postprefill)

    # ---- Simulated handoff: copy rank-0's cache into a fresh cp_size=1 Decode
    # worker's identical cache-tensor slots (stands in for pull_kv_blocks). ----
    conv_cache_handoff = conv_cache_rank0.clone()
    ssm_cache_handoff = ssm_cache_rank0.clone()
    handoff_decode_output = _run_kda_decode_step(conv_cache_handoff, ssm_cache_handoff)

    torch.testing.assert_close(handoff_decode_output, baseline_decode_output)


def test_mla_cp_prefill_to_decode_handoff_matches_noncp_baseline(monkeypatch) -> None:
    """MLA/DSA analog of ``test_kda_cp_prefill_to_decode_handoff_matches_noncp_baseline``.

    Drives the real (unmocked) ``Glm5NextMlaAttention.forward`` /
    ``NpuPagedAttentionBackend.execute_mla`` -- including the real
    ``write_mla_paged_cache`` -> ``reshape_paged_cache`` KV/latent-cache write
    and the real ``_update_mla_index_cache`` raw index-cache write -- for both
    a plain cp_size=1 Prefill+Decode baseline and a cp_size=2 CP-Prefill,
    then simulates the design.md §2.2 PD handoff by copying rank 0's
    post-Prefill ``nope_cache``/``index_cache`` into a fresh cp_size=1 Decode
    worker. Only the innermost NPU-only sparse-attention score kernel
    (``_mla_sparse``) and the ``reshape_paged_cache`` custom op are replaced
    by deterministic CPU reference implementations (see their docstrings);
    the indexer's ``select_qli`` is a minimal content-writing stand-in (see
    ``_index_writer_with_attend_all_topk``) rather than the real kPool
    compression machinery, which is orthogonal to the KV/latent-cache and
    index-cache write paths this test targets.

    The final decode *output* assertion is sensitive to ``nope_cache``
    corruption (the reference attention kernel reads it) but not to
    ``index_cache`` corruption (the stand-in indexer does not gate its topk
    on index-cache content) -- so ``index_cache`` correctness is checked by
    a direct tensor-equality assertion instead, immediately after Prefill.
    """
    monkeypatch.setattr(torch.ops.xllm_ops, "reshape_paged_cache", _mla_reshape_paged_cache_reference, raising=False)

    # ---- Baseline: plain cp_size=1 Prefill (4 tokens) + Decode (1 token). ----
    nope_cache_baseline = torch.zeros(1, _MLA_HANDOFF_BLOCK_SIZE, 1, 1, dtype=torch.float32)
    index_cache_baseline = torch.zeros(1, _MLA_HANDOFF_BLOCK_SIZE, 1, 1, dtype=torch.float32)
    backend_baseline = _mla_handoff_backend(nope_cache_baseline, index_cache_baseline)
    indexer_baseline = MagicMock()
    indexer_baseline.select_qli.side_effect = _index_writer_with_attend_all_topk
    attention_baseline = _make_mla_attention(indexer_baseline)
    _mla_handoff_prepare(
        backend_baseline,
        slot_mapping=torch.arange(4, dtype=torch.int64),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        kv_seq_lens=torch.tensor([4], dtype=torch.int64),
        q_cu_seq_lens=torch.tensor([0, 4], dtype=torch.int32),
        is_prefill=True,
    )
    with (
        patch.object(
            backend_baseline,
            "_mla_sparse",
            lambda *args, **kwargs: _mla_sparse_reference(backend_baseline, *args, **kwargs),
        ),
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=None, attention_backend=backend_baseline),
        ),
        patch.object(
            npu_paged_attention_module,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=None, attention_backend=backend_baseline),
        ),
    ):
        attention_baseline(
            torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
            torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            torch.ones(1, 4, dtype=torch.bool),
        )
    nope_cache_baseline_postprefill = nope_cache_baseline.clone()
    index_cache_baseline_postprefill = index_cache_baseline.clone()
    baseline_decode_output = _run_mla_decode_step(nope_cache_baseline, index_cache_baseline)

    # ---- CP case: same prompt's Prefill under cp_size=2, rank 0. ----
    nope_cache_rank0 = torch.zeros(1, _MLA_HANDOFF_BLOCK_SIZE, 1, 1, dtype=torch.float32)
    index_cache_rank0 = torch.zeros(1, _MLA_HANDOFF_BLOCK_SIZE, 1, 1, dtype=torch.float32)
    _run_mla_cp_rank_prefill(
        0,
        torch.tensor([[[1.0], [4.0]]]),
        torch.tensor([[0, 3]], dtype=torch.int32),
        nope_cache_rank0,
        index_cache_rank0,
        monkeypatch,
    )

    # Per design.md §2.2, rank 0's cache is the one a cp_size=1 Decode
    # instance's `pull_kv_blocks` rank arithmetic actually transfers; it must
    # exactly match the non-CP baseline's post-Prefill cache -- both the
    # KV/latent cache (attends/decodes correctly, checked via decode output
    # below) and the raw index cache (checked directly here; see docstring).
    torch.testing.assert_close(nope_cache_rank0, nope_cache_baseline_postprefill)
    torch.testing.assert_close(index_cache_rank0, index_cache_baseline_postprefill)

    # ---- Simulated handoff: copy rank-0's cache into a fresh cp_size=1 Decode
    # worker's identical cache-tensor slots (stands in for pull_kv_blocks). ----
    nope_cache_handoff = nope_cache_rank0.clone()
    index_cache_handoff = index_cache_rank0.clone()

    # The decode-output assertion below is sensitive to nope_cache corruption
    # (the reference attention kernel reads it) but, per the docstring, not to
    # index_cache corruption (the stand-in indexer does not gate its topk on
    # index-cache content) -- so assert the handoff copy itself landed the
    # right content directly, rather than relying on an insensitive output.
    torch.testing.assert_close(index_cache_handoff, index_cache_baseline_postprefill)

    handoff_decode_output = _run_mla_decode_step(nope_cache_handoff, index_cache_handoff)

    torch.testing.assert_close(handoff_decode_output, baseline_decode_output)


def _make_kda_attention_for_cross_rank_test() -> glm5_next.Glm5NextKdaAttention:
    attention = glm5_next.Glm5NextKdaAttention.__new__(glm5_next.Glm5NextKdaAttention)
    nn.Module.__init__(attention)
    # main's forward opens with wait_for_layer_load(self.layer_id) (the host
    # cache store-reuse fix); the thread-local load context is absent in
    # these tests, so the call is a no-op, but the attribute must exist.
    attention.layer_id = 0
    attention.hidden_size = 1
    attention.head_dim = 1
    attention.qkv_dim = 1
    attention.conv_dim = 3
    attention.num_heads_local = 1
    attention.tp = 1
    attention.in_proj_qkvbfg_a = lambda hidden: torch.cat(
        (hidden.expand(-1, -1, 3), hidden, hidden, hidden),
        dim=-1,
    )
    attention.input_projection_sizes = (3, 1, 2)
    attention.forget_gate = SimpleNamespace(raw_projection=lambda hidden: hidden.unsqueeze(-1) * 10)
    attention.g_b_proj = lambda hidden: hidden * 100
    attention._fg_b_weight = None
    attention.o_norm = MagicMock(side_effect=lambda core, gate: core + gate)
    attention.o_proj = nn.Identity()
    return attention


def _make_kda_attention_for_handoff_test() -> glm5_next.Glm5NextKdaAttention:
    """Real ``Glm5NextKdaAttention`` driven through the real backend, unlike
    ``_make_kda_attention_for_cross_rank_test`` (mocked backend). Needs a real
    ``forget_gate`` (``raw_projection``/``gate_from_raw``) and conv weights
    because ``execute_linear`` calls them for real."""
    attention = glm5_next.Glm5NextKdaAttention.__new__(glm5_next.Glm5NextKdaAttention)
    nn.Module.__init__(attention)
    attention.layer_id = 0
    attention.hidden_size = 1
    attention.head_dim = 1
    attention.qkv_dim = 1
    attention.conv_dim = 3
    attention.conv_kernel_size = 2
    attention.conv_weight_t = torch.ones(2, 3, dtype=torch.float32)
    attention.activation = "identity"
    attention.num_heads_local = 1
    attention.tp = 1
    attention.in_proj_qkvbfg_a = lambda hidden: torch.cat(
        (hidden.expand(-1, -1, 3), hidden, hidden, hidden),
        dim=-1,
    )
    attention.input_projection_sizes = (3, 1, 2)
    attention.forget_gate = SimpleNamespace(
        raw_projection=lambda hidden: hidden.unsqueeze(-1),
        safe_gate_lower_bound=None,
        gate_from_raw=lambda raw: -0.5 * torch.sigmoid(raw),
    )
    attention.g_b_proj = lambda hidden: hidden * 100
    attention._fg_b_weight = None
    attention.o_norm = lambda core, gate: core + gate
    attention.o_proj = nn.Identity()
    return attention


def _shards_by_local(
    rank0_shard: torch.Tensor,
    rank1_shard: torch.Tensor,
) -> dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]]:
    """Map each rank's local shard to the (rank0, rank1) pair it belongs to."""
    return {
        tuple(rank0_shard.reshape(-1).tolist()): (rank0_shard, rank1_shard),
        tuple(rank1_shard.reshape(-1).tolist()): (rank0_shard, rank1_shard),
    }


def _patch_cp_gather_rank_major(
    monkeypatch,
    shards_by_local: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]],
) -> MagicMock:
    """Content-dependent rank-major fake for the CP ``all_gather``.

    A real all_gather returns the same rank-major concatenation -- rank 0's
    local shard first, rank 1's second -- to every caller, whichever rank
    invoked it. The fake identifies the caller's local shard by its contents
    and rebuilds that shard's rank-major pair, so the gathered tensor is a
    function of what the forward actually passed: a rank-mixup or a merge
    that consumes the wrong rank's rows changes the gathered tensor and
    fails the downstream asserts, instead of comparing two constants.
    """

    def all_gather(value: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0
        assert world_size == 2
        assert group_name == "cp"
        key = tuple(value.reshape(-1).tolist())
        rank0_shard, rank1_shard = shards_by_local[key]
        return torch.cat([rank0_shard, rank1_shard], dim=0)

    gather = MagicMock(side_effect=all_gather)
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)
    return gather


def _kda_backend_with_cache(conv_cache: torch.Tensor, ssm_cache: torch.Tensor) -> NpuPagedAttentionBackend:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._kv_caches = [LayerCache(key=None, value=None, conv=conv_cache, ssm=ssm_cache)]
    backend._metadata = None
    backend._kda_verify_width = 1
    # The spec-verify state prep reads this unconditionally; the
    # object.__new__ construction above bypasses __init__, so initialize it
    # here like a real backend would from the cache geometry.
    backend._linear_state_checkpoint_stride = linear_state_checkpoint_stride(conv_cache, ssm_cache)
    return backend


def _kda_prefill_metadata(num_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        linear_state_indices=torch.tensor([0], dtype=torch.int32),
        linear_state_read_indices=torch.tensor([0], dtype=torch.int32),
        linear_state_write_indices=torch.tensor([0], dtype=torch.int32),
        has_initial_state=torch.tensor([0], dtype=torch.int64),
        is_prefill=True,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.tensor([0, num_tokens], dtype=torch.int32),
        kv_seq_lens=None,
        expanded_decode_metadata=None,
    )


def _kda_decode_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        linear_state_indices=torch.tensor([0], dtype=torch.int32),
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        has_initial_state=None,
        is_prefill=False,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        kv_seq_lens=None,
        expanded_decode_metadata=None,
    )


def _install_kda_reference_kernels(monkeypatch) -> None:
    """Install the two reference kernels above as ``fla_npu.ops.ascendc``.

    ``execute_linear`` imports ``fla_npu.ops.ascendc`` lazily (inside the
    function body), so patching ``sys.modules`` -- reverted automatically by
    ``monkeypatch`` -- is enough; no need to reload any already-imported
    module.

    Also stubs ``kernels.causal_conv1d_update_v2`` (used only by
    ``_spec_verify_v3``'s framework-managed conv checkpoints) with an identity
    passthrough, following ``test_glm53_linear_state_io.py``'s
    ``kda_test_environment`` convention: no CPU reference for that native op
    exists, and both the CP and non-CP sides of every comparison below go
    through the same stub, so the CP shard/merge cycle under test is still
    compared over real, content-sensitive recurrence inputs.
    """
    ascendc = types.ModuleType("fla_npu.ops.ascendc")
    ascendc.chunk_kda_fwd = _chunk_kda_fwd_reference
    ascendc.recurrent_kda = _recurrent_kda_reference
    fla_npu_ops = types.ModuleType("fla_npu.ops")
    fla_npu_ops.ascendc = ascendc
    fla_npu = types.ModuleType("fla_npu")
    fla_npu.ops = fla_npu_ops
    monkeypatch.setitem(sys.modules, "fla_npu", fla_npu)
    monkeypatch.setitem(sys.modules, "fla_npu.ops", fla_npu_ops)
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", ascendc)
    monkeypatch.setattr(
        npu_paged_attention_module.kernels,
        "causal_conv1d_update_v2",
        lambda value, *_args, **_kwargs: value,
        raising=False,
    )


def _chunk_kda_fwd_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor | None,
    beta: torch.Tensor,
    scale: float,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic, content-sensitive stand-in for the real NPU kernel.

    Not a reproduction of the real delta-rule numerics (no such CPU kernel
    exists in this repository to fall back to -- see
    ``test_glm53_linear_state_io.py``'s ``_install_kda_stubs`` for the
    existing precedent of stubbing this same kernel with placeholder math).
    What A3 needs is a real, deterministic, *content-sensitive* recurrence
    that a corrupted cache handoff would visibly perturb; exact hardware
    fidelity is not required because both the CP-derived and non-CP baseline
    decode runs go through this same reference, so any mismatch between them
    can only come from the cache-handoff plumbing under test, not from
    reference-kernel error.
    """
    initial_state = kwargs["initial_state"].float()
    batch_size, seq_len, num_heads, head_dim = query.shape
    state = initial_state.clone()
    output = torch.zeros(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    query_f, key_f, value_f, beta_f = query.float(), key.float(), value.float(), beta.float()
    for b in range(batch_size):
        for h in range(num_heads):
            running_state = state[b, h].clone()
            for t in range(seq_len):
                decay = _kda_reference_gate_decay(gate[b, t, h].float()).mean() if gate is not None else 1.0
                running_state = decay * running_state + beta_f[b, t, h] * torch.outer(key_f[b, t, h], value_f[b, t, h])
                output[b, t, h] = scale * (query_f[b, t, h].unsqueeze(0) @ running_state).squeeze(0)
            state[b, h] = running_state
    return output.to(query.dtype), state


def _recurrent_kda_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor | None,
    beta: torch.Tensor,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-side counterpart of :func:`_chunk_kda_fwd_reference` (see its
    docstring for why exact hardware fidelity is not required here)."""
    initial_state = kwargs["initial_state"].float()
    scale = kwargs.get("scale", 1.0)
    total_tokens, num_heads, head_dim = query.shape
    state = initial_state.clone()
    output = torch.zeros(total_tokens, num_heads, head_dim, dtype=torch.float32)
    query_f, key_f, value_f, beta_f = query.float(), key.float(), value.float(), beta.float()
    for t in range(total_tokens):
        for h in range(num_heads):
            decay = _kda_reference_gate_decay(gate[t, h].float()).mean() if gate is not None else 1.0
            row = t if state.shape[0] == total_tokens else 0
            running_state = decay * state[row, h].clone() + beta_f[t, h] * torch.outer(key_f[t, h], value_f[t, h])
            output[t, h] = scale * (query_f[t, h].unsqueeze(0) @ running_state).squeeze(0)
            state[row, h] = running_state
    return output.to(query.dtype), state


def _kda_reference_gate_decay(gate: torch.Tensor) -> torch.Tensor:
    return torch.exp(gate)


def _index_writer_with_attend_all_topk(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    attention_mask: torch.Tensor,
    ctx: object,
    *rest: object,
) -> torch.Tensor:
    """``select_qli`` stand-in: writes real, content-derived rows into the
    real index cache via ``ctx.update_index_cache`` (exercising the real
    ``NpuPagedAttentionBackend._update_mla_index_cache`` write path this
    test targets), then returns an "attend to every valid KV position" topk.
    This sidesteps kPool's compression/Triton machinery (out of scope for
    A3, which targets the KV/latent-cache and raw index-cache *write* paths,
    not the indexer's internal selection logic) while still genuinely
    writing and later comparing the index cache's content.
    """
    del qr, attention_mask
    num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
    packed = hidden_states.reshape(num_tokens, -1) * 1000  # distinguishable from raw hidden values
    ctx.update_index_cache(packed, None)
    kv_len = int(ctx.actual_seq_kv.reshape(-1)[-1].item())
    return torch.arange(kv_len, dtype=torch.int32).view(1, 1, -1).expand(num_tokens, 1, -1)


def _run_kda_cp_rank_prefill(
    rank: int,
    local_hidden: torch.Tensor,
    local_positions: torch.Tensor,
    conv_cache: torch.Tensor,
    ssm_cache: torch.Tensor,
    monkeypatch,
) -> None:
    """Run one CP rank's real Prefill forward, writing into ``conv_cache``/
    ``ssm_cache`` for real via the real backend."""
    gathered_mixed_qkv = torch.tensor([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    gathered_g_raw = torch.tensor([[[1.0]], [[4.0]], [[2.0]], [[3.0]]])
    # Raw (pre-sigmoid) beta: the model layer merges ``beta_raw`` and
    # ``execute_linear`` applies the sigmoid itself, so a real ``all_gather``
    # of the merged tensor returns the raw rank-major projection.
    gathered_beta = torch.tensor([[1.0], [4.0], [2.0], [3.0]])
    gather = MagicMock(side_effect=[gathered_mixed_qkv, gathered_g_raw, gathered_beta])
    backend = _kda_backend_with_cache(conv_cache, ssm_cache)
    backend._metadata = _kda_prefill_metadata(4)
    attention = _make_kda_attention_for_handoff_test()
    cp_context = _cp2_context(rank)
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend, cp_context=cp_context),
    ):
        attention(local_hidden, local_positions, torch.tensor([[True, True]]))


def _run_kda_decode_step(
    conv_cache: torch.Tensor,
    ssm_cache: torch.Tensor,
) -> torch.Tensor:
    """Run one plain (cp_size=1) Decode step against the given cache and
    return the layer's output tensor."""
    backend = _kda_backend_with_cache(conv_cache, ssm_cache)
    backend._metadata = _kda_decode_metadata()
    attention = _make_kda_attention_for_handoff_test()
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend, cp_context=None),
    ):
        return attention(
            torch.tensor([[[5.0]]]), torch.tensor([[4]], dtype=torch.int32), torch.ones(1, 1, dtype=torch.bool)
        )


def _mla_handoff_backend(nope_cache: torch.Tensor, index_cache: torch.Tensor) -> NpuPagedAttentionBackend:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._kv_caches = [LayerCache(key=nope_cache, value=None, index=index_cache)]
    backend._metadata = None
    backend._is_mla = True
    backend._uses_sparse_mla = True
    backend.scale = 1.0
    backend._block_table_i32 = None
    backend._mla_quant_indexer_metadata = {}
    backend._graph_workspace = None
    backend._graph_outputs = {}
    backend._graph_lses = {}
    backend._page_size = _MLA_HANDOFF_BLOCK_SIZE
    backend._kpool_cache_triton_compatible = (False,)
    # ``prepare`` reads these two regardless of the metadata shape; the XFIA
    # decode fast path stays off and no KDA speculative checkpointing runs.
    backend._use_xfia_decode = False
    backend._linear_state_checkpoint_stride = None
    return backend


def _mla_handoff_prepare(
    backend: NpuPagedAttentionBackend,
    *,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    q_cu_seq_lens: torch.Tensor,
    is_prefill: bool,
) -> None:
    metadata = SimpleNamespace(
        slot_mapping=slot_mapping,
        block_table=block_table,
        kv_seq_lens=kv_seq_lens,
        q_cu_seq_lens=q_cu_seq_lens,
        q_seq_lens=None,
        is_prefill=is_prefill,
        is_chunked_prefill=False,
        has_kv_shard=False,
        kv_split_size=1,
        local_slot_mapping=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
        q_cu_host_values=None,
    )
    backend.prepare(metadata)


def _mla_reshape_paged_cache_reference(
    slot_mapping: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> None:
    """CPU reference for the real ``reshape_paged_cache`` op, mirroring
    ``xllm/core/kernels/cuda/reshape_paged_cache.cu``'s per-slot scatter:
    ``slot // block_size`` selects the block, ``slot % block_size`` the
    offset within it; negative slots are padding and are skipped. GLM-5.3's
    NoPE MLA aliases ``key_cache``/``value_cache`` to the same ``nope_cache``
    tensor (see ``write_mla_paged_cache``), so this writes it twice
    (redundant but harmless, matching the real op's contract).
    """
    slots = slot_mapping.reshape(-1).tolist()
    num_kv_heads, head_dim = keys.shape[-2], keys.shape[-1]
    key_flat = key_cache.view(-1, num_kv_heads, head_dim)
    value_flat = value_cache.view(-1, num_kv_heads, head_dim)
    for row, slot in enumerate(slots):
        if slot < 0:
            continue
        key_flat[slot].copy_(keys[row])
        value_flat[slot].copy_(values[row])


def _mla_sparse_reference(
    backend: NpuPagedAttentionBackend,
    q_latent: torch.Tensor,
    q_pe: torch.Tensor | None,
    nope_cache: torch.Tensor,
    rope_cache: torch.Tensor | None,
    topk: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_q: torch.Tensor,
    actual_seq_kv: torch.Tensor,
    layer_id: int,
) -> torch.Tensor:
    """Deterministic, content-sensitive CPU stand-in for the real
    ``npu_sparse_flash_attention`` kernel: dense scaled-dot-product softmax
    attention over every valid cached KV position addressed through
    ``block_table`` (this test's tiny per-sequence ``topk`` always selects
    every valid position -- a degenerate case of top-k -- so ``topk`` itself
    is unused here; see ``_index_writer_with_attend_all_topk`` below).

    As with the KDA reference kernels, exact hardware-numerics fidelity is
    not required: what A3 needs is a real read of ``nope_cache`` that is
    sensitive to its content, so a corrupted cache-handoff copy is visible in
    the decode output. Both the CP-derived and non-CP baseline runs go
    through this same reference.
    """
    del q_pe, rope_cache, topk, layer_id
    kv_lens = actual_seq_kv.reshape(-1).tolist()
    q_ends = actual_seq_q.reshape(-1).tolist()
    q_starts = [0, *q_ends[:-1]]
    num_heads = q_latent.shape[1]
    output = torch.zeros_like(q_latent)
    flat_cache = nope_cache.view(-1, nope_cache.shape[-2], nope_cache.shape[-1])
    for seq_idx, (kv_len, q_start, q_end) in enumerate(zip(kv_lens, q_starts, q_ends)):
        block_row = block_table[seq_idx]
        keys = []
        for position in range(kv_len):
            block_idx = int(block_row[position // _MLA_HANDOFF_BLOCK_SIZE].item())
            offset = position % _MLA_HANDOFF_BLOCK_SIZE
            keys.append(flat_cache[block_idx * _MLA_HANDOFF_BLOCK_SIZE + offset])
        keys_tensor = torch.stack(keys, dim=0).float()  # [kv_len, num_heads, kv_lora]
        for row in range(q_start, q_end):
            query_row = q_latent[row].float()  # [num_heads, kv_lora]
            for head in range(num_heads):
                scores = (keys_tensor[:, head, :] * query_row[head].unsqueeze(0)).sum(-1) * backend.scale
                weights = torch.softmax(scores, dim=0)
                output[row, head] = (weights.unsqueeze(-1) * keys_tensor[:, head, :]).sum(0).to(output.dtype)
    return output


def _run_mla_cp_rank_prefill(
    rank: int,
    local_hidden: torch.Tensor,
    local_positions: torch.Tensor,
    nope_cache: torch.Tensor,
    index_cache: torch.Tensor,
    monkeypatch,
) -> None:
    """Run one CP rank's real Prefill forward, writing into ``nope_cache``/
    ``index_cache`` for real via the real backend."""
    gathered_hidden = torch.tensor([[1.0], [4.0], [2.0], [3.0]])
    gathered_positions = torch.tensor([[0], [3], [1], [2]], dtype=torch.int32)
    gather = MagicMock(side_effect=[gathered_hidden, gathered_positions])
    backend = _mla_handoff_backend(nope_cache, index_cache)
    indexer = MagicMock()
    indexer.select_qli.side_effect = _index_writer_with_attend_all_topk
    attention = _make_mla_attention(indexer)
    cp_context = _cp2_context(rank)
    _mla_handoff_prepare(
        backend,
        slot_mapping=torch.arange(4, dtype=torch.int64),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        kv_seq_lens=torch.tensor([4], dtype=torch.int64),
        q_cu_seq_lens=torch.tensor([0, 4], dtype=torch.int32),
        is_prefill=True,
    )
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)
    with (
        patch.object(backend, "_mla_sparse", lambda *args, **kwargs: _mla_sparse_reference(backend, *args, **kwargs)),
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
        ),
        patch.object(
            npu_paged_attention_module,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context, attention_backend=backend),
        ),
    ):
        attention(local_hidden, local_positions, torch.tensor([[True, True]]))


def _run_mla_decode_step(nope_cache: torch.Tensor, index_cache: torch.Tensor) -> torch.Tensor:
    """Run one plain (cp_size=1) Decode step (5th token) against the given
    cache and return the layer's output tensor."""
    backend = _mla_handoff_backend(nope_cache, index_cache)
    indexer = MagicMock()
    indexer.select_qli.side_effect = _index_writer_with_attend_all_topk
    attention = _make_mla_attention(indexer)
    _mla_handoff_prepare(
        backend,
        slot_mapping=torch.tensor([4], dtype=torch.int64),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        kv_seq_lens=torch.tensor([5], dtype=torch.int64),
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        is_prefill=False,
    )
    with (
        patch.object(backend, "_mla_sparse", lambda *args, **kwargs: _mla_sparse_reference(backend, *args, **kwargs)),
        patch.object(
            glm5_next,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=None, attention_backend=backend),
        ),
        patch.object(
            npu_paged_attention_module,
            "get_forward_context",
            return_value=SimpleNamespace(cp_context=None, attention_backend=backend),
        ),
    ):
        output, _ = attention(
            torch.tensor([[[5.0]]]), torch.tensor([[4]], dtype=torch.int32), torch.ones(1, 1, dtype=torch.bool)
        )
    return output


# ---------------------------------------------------------------------------
# KV-split indexer-pool suite (from the cp-kv-split branch): the write-mode
# pin, the compressed-indexer select_qli fixtures, and the pool write
# localization / replicated-natural-rows / roundtrip parity tests.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _pin_cp_index_write_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the CP index-write mode switch instead of inheriting the ambient
    environment.

    The switch is an operator-facing runtime flag; a shell that exports
    ``XLLM_CP_INDEX_WRITE_MODE`` must not leak into these tests. The model-side
    tests hand-build their index contexts (the layout hooks are injected
    directly), so they are switch-independent by design; the replicated-mode
    tests override this with ``"replicated"`` through the same ``monkeypatch``
    instance.
    """
    monkeypatch.setenv("XLLM_CP_INDEX_WRITE_MODE", "sharded")


def _compressed_index_context(
    recorded: dict[str, object],
    *,
    has_kv_shard: bool,
) -> SimpleNamespace:
    """A compressed-tail context whose pool views are observable.

    ``update_compressed_kpool`` is replaced by a recorder so the write's page
    table can be read back, and the sharded case advertises a distinct
    materialized cache/page table so a read that forgot to materialize is
    visible.
    """
    ctx = _index_context_recording(None, recorded)
    ctx.block_table = torch.tensor([[3, 7]], dtype=torch.int32)
    ctx.actual_seq_kv = torch.tensor([4], dtype=torch.int64)
    ctx.slot_mapping = torch.arange(4, dtype=torch.int64)
    ctx.kpool_tail = torch.zeros(2, 2, 2, 1, dtype=torch.bfloat16)
    ctx.kpool_tail_read_indices = torch.tensor([2], dtype=torch.int64)
    ctx.kpool_tail_write_indices = ctx.kpool_tail_read_indices
    ctx.kpool_query_lens = (4,)
    ctx.index_cache = torch.full((8, 1, 1, 1), -7.0)
    ctx.materialized_index_cache = torch.full((8, 1, 1, 1), 7.0)
    ctx.materialized_block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    ctx.has_kv_shard = has_kv_shard
    ctx.materialize_index_cache = lambda: (
        ctx.materialized_index_cache,
        None,
        ctx.materialized_block_table,
    )
    ctx.localize_pool_block_table = (
        (lambda table: localize_pool_write_block_table(table, dcp_size=2, dcp_rank=0))
        if has_kv_shard
        # A full replica: the helper short-circuits to the caller's table, which
        # is what the backend closure does for kv_split_size == 1.
        else (lambda table: table)
    )
    return ctx


@pytest.mark.parametrize("has_kv_shard", [False, True])
def _make_compressed_indexer() -> glm5_next.Glm5NextIndexer:
    """The compressed-tail kPool body (``index_kpool > 1`` + per-request tail),
    which is the shape the GLM-5.3-Flash checkpoint config selects: only the
    pool write and the pool read are interesting here, so ``select_topk`` and
    ``update_compressed_kpool`` are the call sites under test rather than the
    kernel math they contain."""
    indexer = _make_simple_indexer()
    indexer.index_kpool = 2
    indexer.index_kpool_compress = True
    indexer.index_kpool_always_select_tail = True
    indexer._uses_npu_compressed_tail = True
    indexer._update_compact_kpool = None
    indexer._compact_kpool_model_compatible = False
    indexer.index_kpool_compress_ape = torch.zeros(2, 1)
    return indexer


def _replicated_compressed_index_context(
    recorded: dict[str, object],
) -> SimpleNamespace:
    """A compressed-tail context wired the way ``mla_index_context`` wires it
    in replicated write mode.

    Eight tokens of one logical block (entry 2, so its index-page group is
    [4, 5]); the pool write goes through the natural-row expansion and the
    read through the local full-replica cache with its physically expanded
    table. ``update_compressed_kpool`` stays REAL so the written pool rows can
    be read back."""
    ctx = _index_context_recording(None, recorded)
    ctx.block_table = torch.tensor([[2]], dtype=torch.int32)
    ctx.actual_seq_kv = torch.tensor([8], dtype=torch.int64)
    ctx.slot_mapping = torch.arange(16, 24, dtype=torch.int64)
    ctx.kpool_tail = torch.zeros(2, 2, 2, 1, dtype=torch.bfloat16)
    ctx.kpool_tail_read_indices = torch.tensor([2], dtype=torch.int64)
    ctx.kpool_tail_write_indices = ctx.kpool_tail_read_indices
    ctx.kpool_query_lens = (8,)
    ctx.index_cache = torch.full((8, 2, 1, 1), -7.0)
    ctx.has_kv_shard = True
    # The physically expanded local table: logical block 2 owns pages 4 and 5.
    ctx.materialized_block_table = torch.tensor([[4, 5]], dtype=torch.int32)
    ctx.materialize_index_cache = lambda: (
        ctx.index_cache,
        None,
        ctx.materialized_block_table,
    )
    ctx.localize_pool_block_table = lambda table: replicate_pool_write_block_table(table, dcp_size=2)
    return ctx


def _run_compressed_select_qli(ctx: SimpleNamespace) -> tuple[object, MagicMock]:
    """Drive ``select_qli`` over the 8-token stream and capture the call sites."""
    indexer = _make_compressed_indexer()
    select_topk = MagicMock(return_value=torch.zeros(1, 8, 1, dtype=torch.long))
    indexer.select_topk = select_topk
    hidden = torch.arange(1.0, 9.0).view(1, 8, 1)
    with patch.object(glm5_next, "update_compressed_kpool", wraps=glm5_next.update_compressed_kpool) as write_pools:
        indexer.select_qli(
            hidden_states=hidden,
            qr=hidden.reshape(8, 1),
            positions=torch.arange(8, dtype=torch.int32).view(1, 8),
            attention_mask=torch.ones(1, 8, dtype=torch.bool),
            ctx=ctx,
            layer=None,
            backend=MagicMock(),
        )
    return write_pools, select_topk


@pytest.mark.parametrize("has_kv_shard", [False, True])
def test_dsa_indexer_pool_write_is_localized_and_read_is_materialized(has_kv_shard: bool) -> None:
    """Owner-sharded index cache: the compressed-pool write must address this
    rank's own pool pages (peer pages marked invalid) while the read must
    address the reconstructed logical pages. With no shard both call sites see
    exactly the objects they saw before, so the kv_split_size == 1 launch is
    untouched.
    """
    indexer = _make_compressed_indexer()
    recorded: dict[str, object] = {}
    ctx = _compressed_index_context(recorded, has_kv_shard=has_kv_shard)
    select_topk = MagicMock(return_value=torch.zeros(1, 4, 1, dtype=torch.long))
    indexer.select_topk = select_topk
    backend = MagicMock()

    hidden = torch.arange(1.0, 5.0).view(1, 4, 1)
    with patch.object(glm5_next, "update_compressed_kpool") as write_pools:
        indexer.select_qli(
            hidden_states=hidden,
            qr=hidden.reshape(4, 1),
            positions=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            attention_mask=torch.ones(1, 4, dtype=torch.bool),
            ctx=ctx,
            layer=None,
            backend=backend,
        )

    # The write addresses pool pages through a table whose peer-owned columns
    # are invalid; only columns 0 and 2 (owner 0's half of each logical block)
    # survive, and each points at the FIRST page of the block's index-page
    # group (entry * kv_split_size) -- the page the PD transfer plan moves.
    write_pools.assert_called_once()
    write_table = write_pools.call_args.args[9]
    read_args = select_topk.call_args.kwargs
    if has_kv_shard:
        torch.testing.assert_close(
            write_table,
            torch.tensor([[6, -1, 14, -1]], dtype=torch.int32),
        )
        assert read_args["pool_cache"] is ctx.materialized_index_cache
        torch.testing.assert_close(
            read_args["pool_block_table"],
            ctx.materialized_block_table,
        )
        torch.testing.assert_close(
            read_args["pool_query_block_table"],
            ctx.materialized_block_table,
        )
    else:
        assert write_table is ctx.block_table
        assert read_args["pool_cache"] is ctx.index_cache
        assert write_table is read_args["pool_block_table"]
        assert read_args["pool_query_block_table"] is ctx.block_table


def test_dsa_indexer_pool_write_replicates_natural_rows_and_reads_local_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replicated write mode (``XLLM_CP_INDEX_WRITE_MODE=replicated``, the
    default): the compressed-pool write addresses every stripe's page at its
    natural row (no owner filter, no -1 peer columns), and the read goes to
    the LOCAL full-replica cache through the physically expanded table --
    no gathered reconstruction. The sharded battery's counterpart test pins
    the opposite geometry (owner columns only, group-first pages)."""
    monkeypatch.setenv("XLLM_CP_INDEX_WRITE_MODE", "replicated")
    recorded: dict[str, object] = {}
    ctx = _replicated_compressed_index_context(recorded)

    write_pools, select_topk = _run_compressed_select_qli(ctx)

    # The write table is the natural-row expansion of the logical table
    # [[2]]: stripes 0 and 1 address pages 4 and 5 -- never the owner-filtered
    # [[4, -1]] the sharded mode produces for rank 0.
    write_pools.assert_called_once()
    write_table = write_pools.call_args.args[9]
    torch.testing.assert_close(
        write_table,
        torch.tensor([[4, 5]], dtype=torch.int32),
    )
    # Every page of the block's group was persisted: pools 0 and 1 (stripe 0,
    # page 4) and pools 2 and 3 (stripe 1, page 5) all completed, so neither
    # group page keeps its -7 fill. In sharded mode page 5 would stay stale.
    flat_cache = ctx.index_cache.view(-1)
    assert not bool((flat_cache[8:12] == -7.0).any())

    # The read is the identity view: the local cache object itself plus the
    # physically expanded table, never a gathered copy.
    read_args = select_topk.call_args.kwargs
    assert read_args["pool_cache"] is ctx.index_cache
    torch.testing.assert_close(
        read_args["pool_block_table"],
        ctx.materialized_block_table,
    )
    torch.testing.assert_close(
        read_args["pool_query_block_table"],
        ctx.materialized_block_table,
    )


def test_dsa_indexer_replicated_pool_roundtrip_matches_kv1_full_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write-read roundtrip at the model seam: the real
    ``update_compressed_kpool`` plus ``read_pools`` over the same 8-token
    stream must produce byte-identical pool caches and identical read-back
    pools in the replicated kv_split_size == 2 layout and the
    kv_split_size == 1 full-replica layout. This is the property that makes
    the replicated index cache a drop-in full replica for the transfer side:
    any single writer's pages are all valid."""
    monkeypatch.setenv("XLLM_CP_INDEX_WRITE_MODE", "replicated")
    recorded: dict[str, object] = {}
    replicated_ctx = _replicated_compressed_index_context(recorded)
    _, replicated_topk = _run_compressed_select_qli(replicated_ctx)

    # The kv1 full-replica counterpart: page-granular block table [[4, 5]],
    # no shard views, the identity pool-table hook. The same 8-token stream
    # derives the same global slot mapping [16, 24).
    kv1_ctx = _index_context_recording(None, recorded)
    kv1_ctx.block_table = torch.tensor([[4, 5]], dtype=torch.int32)
    kv1_ctx.actual_seq_kv = torch.tensor([8], dtype=torch.int64)
    kv1_ctx.slot_mapping = torch.arange(16, 24, dtype=torch.int64)
    kv1_ctx.kpool_tail = torch.zeros(2, 2, 2, 1, dtype=torch.bfloat16)
    kv1_ctx.kpool_tail_read_indices = torch.tensor([2], dtype=torch.int64)
    kv1_ctx.kpool_tail_write_indices = kv1_ctx.kpool_tail_read_indices
    kv1_ctx.kpool_query_lens = (8,)
    kv1_ctx.index_cache = torch.full((8, 2, 1, 1), -7.0)
    kv1_ctx.has_kv_shard = False
    _, kv1_topk = _run_compressed_select_qli(kv1_ctx)

    # Identical cache bytes: both layouts wrote the same pools to the same
    # physical rows (pages 4 and 5, pool rows 8..11).
    torch.testing.assert_close(replicated_ctx.index_cache, kv1_ctx.index_cache)

    # Identical read-back through read_pools on each captured view.
    replicated_kwargs = replicated_topk.call_args.kwargs
    kv1_kwargs = kv1_topk.call_args.kwargs
    replicated_pools = read_pools(
        replicated_kwargs["pool_cache"],
        replicated_kwargs["pool_block_table"],
        kv1_ctx.actual_seq_kv,
        n_pools=4,
        rate=2,
    )
    kv1_pools = read_pools(
        kv1_kwargs["pool_cache"],
        kv1_kwargs["pool_block_table"],
        kv1_ctx.actual_seq_kv,
        n_pools=4,
        rate=2,
    )
    torch.testing.assert_close(replicated_pools[0], kv1_pools[0])
    torch.testing.assert_close(replicated_pools[1], kv1_pools[1])
    torch.testing.assert_close(replicated_pools[2], kv1_pools[2])


# ---------------------------------------------------------------------------
# M11.5 scheduler-overlap prework: the overlapped decode-input token
# replacement must be byte-identical across CP ranks.
#
# There is no Python-side replacement entry: under ``enable_schedule_overlap``
# the fake-token -> real-token rewrite happens in the worker
# (``WorkerImpl::update_input_by_last_step_output``, xllm/core/runtime/
# worker_impl.cpp -- the NPU path calls aclnnReplaceToken, whose golden is
# third_party/xllm_ops/test/python_test/test_replace_token.py). The scheduler
# feeds each sequence a 1-based negative placeholder produced by the driver
# worker's fake output (``torch::arange(-1, -(N + 1), -1)``,
# xllm/core/distributed_runtime/worker_service.cpp), the engine broadcasts one
# ``ForwardInput`` to every CP rank of a DP group (``LLMEngine::step``, "Engine
# sends full global tokens"), and every rank resolves the placeholder against
# its own rank-local ``last_step_output_``. The tests below pin that protocol
# at the narrowest Python-visible seam: the replacement rule itself.

# DFlash2 spec-verify and chunked-prefill suite (from the pcp-dflash2
# branch): the prefill-capture -> spec-verify roundtrip against the non-CP
# baseline, the DFlash2 v3 spec-verify shape, and the chunked-prefill
# KDA parity tests with their chunk plumbing.
# ---------------------------------------------------------------------------


def _cp2_chunk_context(rank: int) -> SimpleNamespace:
    """Zigzag CpContext for a 2-new-token chunk under cp_size=2: chunk-local
    token 0 is a lone early segment (rank 0), token 1 a lone late segment
    (rank 1) -- padded to 4 (2*cp_size) chunk slots per the zigzag scheme.
    Mirrors ``_cp2_padded_context`` but parametrized by rank (only rank 0's
    forward is actually driven by this test; rank 1's shard content is only
    needed to build the mocked rank-major ``all_gather`` return value).
    """
    if rank == 0:
        shard_index = torch.tensor([0, -1], dtype=torch.int64)
        shard_gather_index = torch.tensor([0, 0], dtype=torch.int64)
    else:
        shard_index = torch.tensor([1, -1], dtype=torch.int64)
        shard_gather_index = torch.tensor([1, 0], dtype=torch.int64)
    return SimpleNamespace(
        cp_size=2,
        cp_rank=rank,
        total_local=2,
        shard_index=shard_index,
        shard_gather_index=shard_gather_index,
        shard_valid_mask=torch.tensor([True, False]),
        restore_index=torch.tensor([0, 2], dtype=torch.int64),
    )


def _kda_chunked_prefill_metadata(num_tokens: int, *, has_initial_state: int) -> SimpleNamespace:
    return SimpleNamespace(
        linear_state_indices=torch.tensor([0], dtype=torch.int32),
        linear_state_read_indices=torch.tensor([0], dtype=torch.int32),
        linear_state_write_indices=torch.tensor([0], dtype=torch.int32),
        has_initial_state=torch.tensor([has_initial_state], dtype=torch.int64),
        is_prefill=False,
        is_chunked_prefill=True,
        q_cu_seq_lens=torch.tensor([0, num_tokens], dtype=torch.int32),
        kv_seq_lens=None,
        expanded_decode_metadata=None,
    )


def _run_kda_cp_rank_chunk(
    rank: int,
    real_token_value: float,
    other_rank_real_token_value: float,
    conv_cache: torch.Tensor,
    ssm_cache: torch.Tensor,
    has_initial_state: int,
    monkeypatch,
) -> None:
    """Run one CP rank's real chunked-prefill forward for a 2-new-token chunk
    (this rank's real token + a padding row), writing into ``conv_cache``/
    ``ssm_cache`` for real via the real backend. Only rank 0 is actually
    driven by the tests below; ``other_rank_real_token_value`` supplies what
    the mocked ``all_gather`` returns for the peer rank's real row.
    """
    rank0_value = real_token_value if rank == 0 else other_rank_real_token_value
    rank1_value = other_rank_real_token_value if rank == 0 else real_token_value
    # Rank-major gathered order: [rank0_real, rank0_pad, rank1_real, rank1_pad].
    gathered_mixed_qkv = torch.tensor([[rank0_value] * 3, [0.0] * 3, [rank1_value] * 3, [0.0] * 3])
    gathered_g_raw = torch.tensor([[[rank0_value]], [[0.0]], [[rank1_value]], [[0.0]]])
    # Raw (pre-sigmoid) beta projection: on main the model layer merges
    # ``beta_raw`` through cp_merge_rows and ``execute_linear`` applies the
    # sigmoid (fused into the kernel via ``use_beta_sigmoid_in_kernel``), so
    # the mocked gather must return the raw values a real CP all_gather sees.
    gathered_beta = torch.tensor([[rank0_value], [0.0], [rank1_value], [0.0]])
    gather = MagicMock(side_effect=[gathered_mixed_qkv, gathered_g_raw, gathered_beta])
    backend = _kda_backend_with_cache(conv_cache, ssm_cache)
    backend._metadata = _kda_chunked_prefill_metadata(2, has_initial_state=has_initial_state)
    attention = _make_kda_attention_for_handoff_test()
    cp_context = _cp2_chunk_context(rank)
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)
    local_hidden = torch.tensor([[[real_token_value], [0.0]]])
    local_positions = torch.tensor([[0, 0]], dtype=torch.int32)
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend, cp_context=cp_context),
    ):
        attention(local_hidden, local_positions, torch.tensor([[True, False]]))


def _spec_verify_metadata(num_seqs: int, rows_per_seq: int) -> SimpleNamespace:
    idx = torch.arange(num_seqs, dtype=torch.int32)
    return SimpleNamespace(
        linear_state_indices=idx,
        linear_state_read_indices=None,
        # A multi-element tensor here would make backend.py's
        # ``write_indices or read_indices`` raise (ambiguous truth value);
        # None makes it fall through to linear_state_indices, matching how a
        # real spec-verify metadata (no prefix-cache checkpoint rotation) is
        # built (R2-followup Q1: read/write always collapse for spec-verify).
        linear_state_write_indices=None,
        has_initial_state=None,
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=True,
        q_seq_lens_host=None,
        num_accepted_tokens=torch.ones(num_seqs, dtype=torch.int64),
        q_cu_seq_lens=torch.arange(0, num_seqs * rows_per_seq + 1, rows_per_seq, dtype=torch.int32),
        kv_seq_lens=None,
        expanded_decode_metadata=None,
    )


def test_kda_cp_chunked_prefill_matches_noncp_nonchunked_baseline(monkeypatch, causal_conv1d_reference) -> None:
    """B2 (implement.md): >=2 consecutive chunked-prefill forward calls under
    cp_size=2 must produce the same final ``conv_cache``/``ssm_cache`` content
    and the same subsequent decode output as a plain cp_size=1, non-chunked
    (single Prefill call) baseline for the same 4-token prompt.

    This directly exercises the R2-followup / design.md §3.1 Gate B3 claim:
    chunking only changes how many times the "read checkpoint, extend,
    write checkpoint" cycle runs (R2-followup Q4) -- it does not change what
    happens inside one cycle, and CP row-restore only ever needs to
    correctly reorder rows physically present in the current call, whether
    that call is a whole sequence or one chunk of one.
    """
    del causal_conv1d_reference  # fixture side effect (patches native conv1d) is what's needed
    _install_kda_reference_kernels(monkeypatch)

    # ---- Baseline: plain cp_size=1, non-chunked single Prefill call (4 tokens). ----
    conv_cache_baseline = torch.zeros(1, 1, 3, dtype=torch.float32)
    ssm_cache_baseline = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    backend_baseline = _kda_backend_with_cache(conv_cache_baseline, ssm_cache_baseline)
    backend_baseline._metadata = _kda_prefill_metadata(4)
    attention_baseline = _make_kda_attention_for_handoff_test()
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend_baseline, cp_context=None),
    ):
        attention_baseline(
            torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
            torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            torch.ones(1, 4, dtype=torch.bool),
        )
    conv_cache_baseline_postprefill = conv_cache_baseline.clone()
    ssm_cache_baseline_postprefill = ssm_cache_baseline.clone()
    baseline_decode_output = _run_kda_decode_step(conv_cache_baseline, ssm_cache_baseline)

    # ---- CP+chunked case: same 4-token prompt as 2 consecutive chunks of 2
    # tokens each, cp_size=2, rank 0. Chunk 1 = global tokens [1.0, 2.0]
    # (rank 0 owns 1.0, rank 1 owns 2.0); chunk 2 = global tokens [3.0, 4.0]
    # (rank 0 owns 3.0, rank 1 owns 4.0). Chunk 1 is cold-start
    # (has_initial_state=0); chunk 2 resumes chunk 1's checkpoint
    # (has_initial_state=1) -- the exact "resume, extend, checkpoint"
    # semantic R2-followup Q1/Q2 established for the non-CP chunked case. ----
    conv_cache_rank0 = torch.zeros(1, 1, 3, dtype=torch.float32)
    ssm_cache_rank0 = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    _run_kda_cp_rank_chunk(
        0,
        real_token_value=1.0,
        other_rank_real_token_value=2.0,
        conv_cache=conv_cache_rank0,
        ssm_cache=ssm_cache_rank0,
        has_initial_state=0,
        monkeypatch=monkeypatch,
    )
    _run_kda_cp_rank_chunk(
        0,
        real_token_value=3.0,
        other_rank_real_token_value=4.0,
        conv_cache=conv_cache_rank0,
        ssm_cache=ssm_cache_rank0,
        has_initial_state=1,
        monkeypatch=monkeypatch,
    )

    # Per design.md §2.2, rank 0's cache is what a cp_size=1 Decode instance's
    # pull_kv_blocks rank arithmetic actually transfers; the CP+chunked
    # 2-chunk run's final cache must exactly match the non-CP, non-chunked
    # single-call baseline's.
    torch.testing.assert_close(conv_cache_rank0, conv_cache_baseline_postprefill)
    torch.testing.assert_close(ssm_cache_rank0, ssm_cache_baseline_postprefill)

    chunked_decode_output = _run_kda_decode_step(conv_cache_rank0, ssm_cache_rank0)
    torch.testing.assert_close(chunked_decode_output, baseline_decode_output)


# ---------------------------------------------------------------------------
# B6 (implement.md): _spec_verify_v3's ephemeral multi-slot scratch-buffer
# mechanism under CP -- see
# research/b6-spec-verify-v3-cp-safety.md for the full trace. This test
# drives the real dispatch into ``_spec_verify_v3`` (num_seqs=2,
# rows_per_seq=2 -- DFlash2's own "bonus + 1 draft token" verify-block
# shape) under cp_size=2, and asserts the resulting conv_cache/ssm_cache and
# attention output exactly match a cp_size=1 baseline for the same 4-token
# verify batch, proving the ephemeral pool's sequence-major-contiguous-block
# assumption survives a real cp_merge_rows restore.
# ---------------------------------------------------------------------------


def test_kda_cp_dflash2_prefill_capture_then_spec_verify_matches_noncp_baseline(
    monkeypatch, causal_conv1d_reference
) -> None:
    """B8: cp_size=2 (rank 0) Prefill-with-aux-hidden-capture, immediately
    followed by a cp_size=2 (rank 0) spec-verify call resuming that exact
    KDA checkpoint, must produce the same final conv_cache/ssm_cache, the
    same restored (global-order) aux-hidden buffer, and the same rank-0
    attention output as a plain cp_size=1 baseline running the identical
    two-step sequence for the same tokens.

    Step 1 (Prefill, 4 global tokens [1.0, 2.0, 3.0, 4.0], capturing this
    layer's output into an aux_hidden buffer -- standing in for the model's
    real ``AuxHiddenCapture.capture_layer`` call, per B1) writes a KDA
    checkpoint into slot 0. Step 2 (spec-verify, 1 sequence, 2 verify rows
    [5.0, 6.0] = bonus + 1 draft token, num_accepted_tokens=1) resumes slot
    0's checkpoint through ``_spec_verify_v3``.
    """
    del causal_conv1d_reference
    _install_kda_reference_kernels(monkeypatch)

    # ---- Baseline: plain cp_size=1, both steps, single shared cache slot. ----
    # Same framework-managed checkpoint layout as B6 above: conv capacity 2
    # (conv_state_len + 2 verify rows - 1) and stride-2 ssm rows per slot, so
    # step 2's ``_spec_verify_v3`` has room for its 2-row verify block.
    conv_cache_baseline = torch.zeros(1, 2, 3, dtype=torch.float32)
    ssm_cache_baseline = torch.zeros(2, 1, 1, 1, dtype=torch.float32)
    backend_baseline = _kda_backend_with_cache(conv_cache_baseline, ssm_cache_baseline)
    attention_baseline = _make_kda_attention_for_handoff_test()

    backend_baseline._metadata = _kda_prefill_metadata(4)
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend_baseline, cp_context=None),
    ):
        baseline_prefill_output = attention_baseline(
            torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
            torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            torch.ones(1, 4, dtype=torch.bool),
        )
    # Stand-in for AuxHiddenCapture.capture_layer: this layer's output is the
    # only captured layer, so the buffer equals the (untransformed) output.
    baseline_aux_hidden = baseline_prefill_output.reshape(-1, 1).clone()

    backend_baseline._kda_verify_width = 2
    backend_baseline._metadata = _spec_verify_metadata(1, 2)
    backend_baseline._prepare_speculative_ssm_state_indices(backend_baseline._metadata, graph_mode=False)
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend_baseline, cp_context=None),
    ):
        attention_baseline(
            torch.tensor([[[5.0], [6.0]]]),
            torch.tensor([[4, 5]], dtype=torch.int32),
            torch.ones(1, 2, dtype=torch.bool),
        )
    conv_cache_baseline_post = conv_cache_baseline.clone()
    ssm_cache_baseline_post = ssm_cache_baseline.clone()

    # ---- CP case: same two steps, cp_size=2, rank 0 (+ real rank 1 prefill
    # to derive the true cross-rank gather for the aux-hidden-buffer merge). ----
    conv_cache_rank0 = torch.zeros(1, 2, 3, dtype=torch.float32)
    ssm_cache_rank0 = torch.zeros(2, 1, 1, 1, dtype=torch.float32)
    backend_cp = _kda_backend_with_cache(conv_cache_rank0, ssm_cache_rank0)
    backend_cp._metadata = _kda_prefill_metadata(4)
    attention_cp_rank0 = _make_kda_attention_for_handoff_test()

    gathered_mixed_qkv = torch.tensor([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    gathered_g_raw = torch.tensor([[[1.0]], [[4.0]], [[2.0]], [[3.0]]])
    gathered_beta = torch.tensor([[1.0], [4.0], [2.0], [3.0]])  # raw pre-sigmoid beta (see _run_kda_cp_rank_chunk)
    gather_step1_rank0 = MagicMock(side_effect=[gathered_mixed_qkv, gathered_g_raw, gathered_beta])
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather_step1_rank0, raising=False)
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend_cp, cp_context=_cp2_context(0)),
    ):
        cp_prefill_output_rank0 = attention_cp_rank0(
            torch.tensor([[[1.0], [4.0]]]),
            torch.tensor([[0, 3]], dtype=torch.int32),
            torch.tensor([[True, True]]),
        )

    # Real rank 1 prefill (scratch cache; only its local output is needed) to
    # derive the true rank-major all_gather this layer's output would
    # produce, rather than hand-deriving it -- the aux-hidden buffer merge
    # is exactly this all_gather + restore_index, per design.md §3.2.
    conv_cache_scratch_rank1 = torch.zeros(1, 2, 3, dtype=torch.float32)
    ssm_cache_scratch_rank1 = torch.zeros(2, 1, 1, 1, dtype=torch.float32)
    backend_rank1 = _kda_backend_with_cache(conv_cache_scratch_rank1, ssm_cache_scratch_rank1)
    backend_rank1._metadata = _kda_prefill_metadata(4)
    attention_cp_rank1 = _make_kda_attention_for_handoff_test()
    gather_step1_rank1 = MagicMock(side_effect=[gathered_mixed_qkv, gathered_g_raw, gathered_beta])
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather_step1_rank1, raising=False)
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend_rank1, cp_context=_cp2_context(1)),
    ):
        cp_prefill_output_rank1 = attention_cp_rank1(
            torch.tensor([[[2.0], [3.0]]]),
            torch.tensor([[1, 2]], dtype=torch.int32),
            torch.tensor([[True, True]]),
        )

    # Rank-major all_gather of this layer's (post cp_shard_rows) local
    # output, restored via the real cp_merge_rows -- the model-level
    # mechanism design.md §3.2/B1 already proved correct, now driven with
    # this test's own real per-rank outputs instead of a mocked baseline.
    gathered_aux = torch.cat([cp_prefill_output_rank0.reshape(-1, 1), cp_prefill_output_rank1.reshape(-1, 1)], dim=0)
    aux_gather_mock = MagicMock(return_value=gathered_aux)
    monkeypatch.setattr(glm5_next.distributed, "all_gather", aux_gather_mock, raising=False)
    cp_aux_hidden = glm5_next.cp_merge_rows(cp_prefill_output_rank0.reshape(-1, 1), _cp2_context(0))
    aux_gather_mock.assert_called_once()

    torch.testing.assert_close(cp_aux_hidden, baseline_aux_hidden)

    # ---- Step 2: spec-verify, cp_size=2 rank 0, resuming rank 0's own
    # post-Prefill checkpoint (the real DFlash2 continuation). This
    # sequence's 2-row verify block [5.0 (bonus), 6.0 (draft)] is CP-sharded
    # exactly like a 2-new-token chunk (``_cp2_chunk_context``): rank 0
    # owns global row 0 (bonus, real) + a padding row, rank 1 owns global
    # row 1 (draft, real) + a padding row. Gathered rank-major order is
    # therefore 4 rows: [rank0_real, rank0_pad, rank1_real, rank1_pad].
    backend_cp._kda_verify_width = 2
    backend_cp._metadata = _spec_verify_metadata(1, 2)
    backend_cp._prepare_speculative_ssm_state_indices(backend_cp._metadata, graph_mode=False)
    gathered_mixed_qkv_v2 = torch.tensor([[5.0, 5.0, 5.0], [0.0, 0.0, 0.0], [6.0, 6.0, 6.0], [0.0, 0.0, 0.0]])
    gathered_g_raw_v2 = torch.tensor([[[5.0]], [[0.0]], [[6.0]], [[0.0]]])
    gathered_beta_v2 = torch.tensor([[5.0], [0.0], [6.0], [0.0]])  # raw pre-sigmoid beta (see _run_kda_cp_rank_chunk)
    gather_step2 = MagicMock(side_effect=[gathered_mixed_qkv_v2, gathered_g_raw_v2, gathered_beta_v2])
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather_step2, raising=False)
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend_cp, cp_context=_cp2_chunk_context(0)),
    ):
        cp_verify_output = attention_cp_rank0(
            torch.tensor([[[5.0], [0.0]]]),
            torch.tensor([[4, 0]], dtype=torch.int32),
            torch.tensor([[True, False]]),
        )

    # Final cache state must match the non-CP baseline's exactly -- proving
    # the full Prefill(+capture)-then-spec-verify chain is CP-safe end to
    # end, not just each half in isolation.
    #
    # Known scope limitation (verified by checker-agent bug-injection during
    # review, not previously documented here): this test's shared attention
    # fixture (``_make_kda_attention_for_handoff_test``) uses
    # ``conv_kernel_size=2`` => ``conv_state_len=1``, a single-row conv
    # history. Because both the CP and non-CP baseline calls in this test
    # route through the *identical* ``execute_linear``/``_spec_verify_v3``
    # production code once their inputs are merged back to global row order,
    # a bug injected into that shared code (e.g. dropping the conv-state
    # checkpoint write between step 1 and step 2) corrupts both sides
    # identically and is therefore invisible to a CP-vs-baseline comparison
    # regardless of conv_state_len -- this is a structural property of any
    # "does CP change the result" comparison test, not something a larger
    # kernel size would fix. What conv_state_len=1 *does* additionally hide
    # is any bug that corrupts conv history beyond the single most-recent
    # position (e.g. an off-by-one in a multi-row conv_state slice) even in
    # a hypothetical asymmetric-injection scenario, since there is no
    # earlier position for such a bug to touch. General conv-checkpoint
    # read/extend/write correctness (content-verified, non-degenerate
    # conv_kernel_size=3) is separately covered by
    # ``test_glm53_linear_state_io.py::test_execute_linear_prefill_reads_source_and_writes_live``;
    # this test's own, narrower job is proving the CP shard/merge cycle
    # (``cp_merge_rows``/``restore_index``) does not disturb that already-
    # correct mechanism -- verified directly by checker-agent mutation
    # testing (temporarily reversing ``cp_utils.cp_merge_rows``'s
    # ``restore_index`` and confirming this test fails, then reverting).
    torch.testing.assert_close(conv_cache_rank0, conv_cache_baseline_post)
    torch.testing.assert_close(ssm_cache_rank0, ssm_cache_baseline_post)
    del cp_verify_output  # shape/content already covered by B6's dedicated test


def test_kda_cp_spec_verify_v3_matches_noncp_baseline(monkeypatch, causal_conv1d_reference) -> None:
    """B6: a real cp_size=2 dispatch into ``_spec_verify_v3`` (2 sequences,
    2 verify rows each -- DFlash2's bonus+1-draft-token shape) must produce
    the same final ``conv_cache``/``ssm_cache`` and the same rank-0-local
    attention output as a plain cp_size=1 baseline over the same 4-token
    verify batch, cold-started (num_accepted_tokens=1, i.e. this is the
    first verify call, no prior accepted draft to resume from).

    Global verify batch: sequence 0's block = tokens [1.0, 2.0]
    (bonus, draft0); sequence 1's block = tokens [3.0, 4.0]. Zigzag
    cp_size=2: rank 0 owns global tokens 0 and 3 (values 1.0, 4.0), rank 1
    owns 1 and 2 (values 2.0, 3.0) -- same token/rank assignment
    ``_cp2_context`` already uses elsewhere in this file, just now
    interpreted as a spec-verify block layout instead of a plain sequence.
    """
    del causal_conv1d_reference
    _install_kda_reference_kernels(monkeypatch)

    num_seqs, rows_per_seq = 2, 2

    # ---- Baseline: plain cp_size=1 spec-verify dispatch over all 4 rows. ----
    # Cache shapes follow main's framework-managed checkpoint layout: the conv
    # cache's second dim must hold ``conv_state_len + rows_per_seq - 1`` = 2
    # per-token conv checkpoints, and the ssm cache carries one checkpoint row
    # per conv slot per stride (2 slots x stride 2 = 4 rows), mirroring
    # ``test_glm53_linear_state_io.py``'s spec-verify caches.
    conv_cache_baseline = torch.zeros(2, 2, 3, dtype=torch.float32)
    ssm_cache_baseline = torch.zeros(4, 1, 1, 1, dtype=torch.float32)
    backend_baseline = _kda_backend_with_cache(conv_cache_baseline, ssm_cache_baseline)
    backend_baseline._kda_verify_width = rows_per_seq
    backend_baseline._metadata = _spec_verify_metadata(num_seqs, rows_per_seq)
    backend_baseline._prepare_speculative_ssm_state_indices(backend_baseline._metadata, graph_mode=False)
    attention_baseline = _make_kda_attention_for_handoff_test()
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend_baseline, cp_context=None),
    ):
        baseline_output = attention_baseline(
            torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
            torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            torch.ones(1, 4, dtype=torch.bool),
        )
    conv_cache_baseline_post = conv_cache_baseline.clone()
    ssm_cache_baseline_post = ssm_cache_baseline.clone()

    # ---- CP case: same verify batch, cp_size=2, rank 0. ----
    gathered_mixed_qkv = torch.tensor([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    gathered_g_raw = torch.tensor([[[1.0]], [[4.0]], [[2.0]], [[3.0]]])
    # Raw pre-sigmoid beta (the sigmoid is fused into the kernel on main;
    # see _run_kda_cp_rank_chunk).
    gathered_beta = torch.tensor([[1.0], [4.0], [2.0], [3.0]])
    gather = MagicMock(side_effect=[gathered_mixed_qkv, gathered_g_raw, gathered_beta])
    conv_cache_rank0 = torch.zeros(2, 2, 3, dtype=torch.float32)
    ssm_cache_rank0 = torch.zeros(4, 1, 1, 1, dtype=torch.float32)
    backend_cp = _kda_backend_with_cache(conv_cache_rank0, ssm_cache_rank0)
    backend_cp._kda_verify_width = rows_per_seq
    backend_cp._metadata = _spec_verify_metadata(num_seqs, rows_per_seq)
    backend_cp._prepare_speculative_ssm_state_indices(backend_cp._metadata, graph_mode=False)
    attention_cp = _make_kda_attention_for_handoff_test()
    cp_context = _cp2_context(0)
    monkeypatch.setattr(glm5_next.distributed, "all_gather", gather, raising=False)
    with patch.object(
        glm5_next,
        "get_forward_context_or_none",
        return_value=SimpleNamespace(attention_backend=backend_cp, cp_context=cp_context),
    ):
        cp_output = attention_cp(
            torch.tensor([[[1.0], [4.0]]]),
            torch.tensor([[0, 3]], dtype=torch.int32),
            torch.tensor([[True, True]]),
        )

    # Per design.md §2.2/R3, rank 0's post-verify cache is what a cp_size=1
    # continuation actually reads; it must exactly match the non-CP
    # baseline's -- proving _spec_verify_v3's ephemeral-pool computation and
    # final persistent write are unaffected by the CP shard/merge cycle.
    torch.testing.assert_close(conv_cache_rank0, conv_cache_baseline_post)
    torch.testing.assert_close(ssm_cache_rank0, ssm_cache_baseline_post)

    # cp_output is rank 0's local (sharded) output; compare against the
    # baseline's corresponding global rows (0 and 3, rank 0's owned tokens).
    # baseline_output/cp_output are [batch=1, seq_len, hidden]; select along
    # the sequence axis (dim=1), not the batch axis.
    baseline_rank0_rows = baseline_output.index_select(1, torch.tensor([0, 3]))
    torch.testing.assert_close(cp_output, baseline_rank0_rows)


# ---------------------------------------------------------------------------
# B8 (implement.md): end-to-end DFlash2 + PCP composition test.
#
# A full generate() call is impractical at this unit-test layer (it requires
# real weights, a tokenizer, the C++ scheduler, and the draft's own sampling
# logic -- none of which is available here or should be re-implemented in a
# Python unit test). Per B9's validation-ladder guidance (mirroring Phase
# A's precedent that a CPU-only simulation is a required, not merely
# optional, substitute before any live-hardware escalation), this test
# drives the real model-level mechanism DFlash2's actual two-step decode
# cycle depends on: (1) a CP-Prefill that captures the aux-hidden buffer
# (the draft's input, per B1) and writes the KDA linear-state checkpoint,
# immediately followed by (2) a CP spec-verify call that reads and extends
# that exact checkpoint through the real ``_spec_verify_v3`` path (B6) --
# both steps driven through the real (unmocked) ``Glm5NextKdaAttention``/
# ``execute_linear``, with only the innermost NPU-only kernels replaced by
# the same deterministic CPU references every other test in this file uses.
# This chains B1's and B6's individually-proven mechanisms into the single
# continuous pipeline DFlash2+PCP actually exercises, which neither B1 nor
# B6 alone did.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Schedule-overlap suite (from the sched-overlap branch): the driver
# worker's fake-output token replacement protocol and its CP-rank
# byte-identity invariant.
# ---------------------------------------------------------------------------


def _driver_fake_next_tokens(num_decode_seqs: int) -> torch.Tensor:
    """Driver worker's fake overlap output: 1-based negative row references.

    Mirrors the driver-side construction in ``WorkerService`` (worker_service.cpp):
    when the overlapped step returns an asynchronous (tokenless) output, the
    driver emits ``arange(-1, -(N + 1), -1)`` so the scheduler appends
    placeholder ``-(row + 1)`` into each sequence's token array.
    """
    return torch.arange(-1, -1 * (num_decode_seqs + 1), -1, dtype=torch.int32)


def _overlap_replace_token_golden(token_ids: torch.Tensor, next_tokens: torch.Tensor) -> torch.Tensor:
    """CPU golden of the overlapped decode-input token replacement.

    Mirrors the non-NPU fallback of ``WorkerImpl::update_input_by_last_step_output``
    and the aclnnReplaceToken kernel golden: a negative input value ``v`` is a
    1-based negative reference resolved as ``next_tokens[-v - 1]``;
    non-negative values pass through unchanged.
    """
    neg_mask = token_ids < 0
    clamped_neg_indices = torch.clamp(-token_ids, min=0)
    replacement = next_tokens[clamped_neg_indices - 1]
    return torch.where(neg_mask, replacement, token_ids)


def test_overlap_decode_fake_token_protocol_matches_worker_replacement_rule() -> None:
    """Placeholder encoding pin: ``-(row + 1)`` resolves to ``next_tokens[row]``
    and real tokens pass through.

    The overlapped decode input mixes negative placeholders (whose values the
    driver's fake output fixed at ``-(row + 1)``) with real tokens (e.g. the
    draft positions of a spec-verify row). The replacement must resolve each
    placeholder to the same-row sampled token of the previous step and leave
    real tokens untouched -- otherwise a CP rank would feed the model an input
    that disagrees with the KV state the other ranks wrote.
    """
    num_rows = 5
    # Distinct real tokens so a wrong row mapping cannot cancel out.
    real_next_tokens = torch.tensor([101, 202, 303, 404, 505], dtype=torch.int64)

    # Pure decode batch: every row's query token is the driver placeholder.
    decode_input = _driver_fake_next_tokens(num_rows).clone()
    replaced = _overlap_replace_token_golden(decode_input, real_next_tokens)
    assert replaced.dtype == torch.int64
    torch.testing.assert_close(replaced, real_next_tokens.to(torch.int64))

    # Spec-verify-shaped row: real draft tokens pass through, the trailing
    # placeholders still resolve through their row references.
    mixed_input = torch.tensor([7, 8, 909, -4, -5], dtype=torch.int32)
    mixed_replaced = _overlap_replace_token_golden(mixed_input, real_next_tokens)
    torch.testing.assert_close(mixed_replaced, torch.tensor([7, 8, 909, 404, 505], dtype=torch.int64))


def test_overlap_decode_token_replacement_is_byte_identical_across_cp_ranks() -> None:
    """CP-rank consistency pin for the overlapped decode-input replacement.

    Under CP with ``kv_split_size == 1`` decode runs replicated on every CP
    rank, so the master's glm5_next-CP overlap ban (master.cpp: "initially
    requires enable_schedule_overlap=false") can only be lifted if the
    replacement each rank performs locally is byte-identical across ranks.
    The simulation mirrors the real chain:

    1. The engine broadcasts ONE ``ForwardInput`` to both CP ranks -- each
       rank decodes its own fresh copy of the placeholder tensor.
    2. Each rank samples its own copy of the replicated logits (under overlap
       every rank runs the sampler; llm_worker_impl.cpp's non-driver early
       return only fires when ``enable_schedule_overlap()`` is false),
       producing its own rank-local ``last_step_output_``.
    3. Each rank applies the replacement rule independently.

    Deterministic sampling over replicated logits (greedy here) must give the
    same tokens on both ranks, hence byte-identical decode inputs. If the
    replacement rule, the placeholder encoding, or the per-rank sampling ever
    diverges, the two ranks would write disagreeing KV state for the same
    sequence.
    """
    num_rows = 4
    vocab_size = 32
    # Replicated CP decode computes the same logits on every rank (the same
    # invariant test_kda_cp_backend_inputs_are_identical_regardless_of_local_rank
    # pins at the model seam). Make the argmax distinct per row.
    logits = torch.randn(num_rows, vocab_size, dtype=torch.float32, generator=torch.Generator().manual_seed(2026))

    # 1. Engine broadcast: one placeholder tensor, fresh copy per CP rank.
    broadcast_input = _driver_fake_next_tokens(num_rows).clone()
    rank_inputs = [broadcast_input.clone() for _ in range(2)]
    assert rank_inputs[0].data_ptr() != rank_inputs[1].data_ptr()

    # 2. Per-rank sampling of replicated logits -> rank-local last_step_output.
    rank_next_tokens = []
    for _rank in range(2):
        local_logits = logits.clone()
        rank_next_tokens.append(torch.argmax(local_logits, dim=-1).to(torch.int64))

    # 3. Per-rank replacement.
    rank_replaced = [_overlap_replace_token_golden(rank_inputs[rank], rank_next_tokens[rank]) for rank in range(2)]

    # Byte-identical across ranks: same dtype, same values.
    assert rank_replaced[0].dtype == rank_replaced[1].dtype
    assert torch.equal(rank_replaced[0], rank_replaced[1])
    # And each row resolved to its own sampled token (row mapping is 1-based).
    torch.testing.assert_close(rank_replaced[0], rank_next_tokens[0])
    torch.testing.assert_close(rank_replaced[1], rank_next_tokens[1])
