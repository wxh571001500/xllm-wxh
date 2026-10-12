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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xllm.python.models import glm5_next


class _SharedPartial(nn.Module):
    def __init__(self, expected: torch.Tensor, multiplier: int) -> None:
        super().__init__()
        self._expected = expected
        self._multiplier = multiplier
        self.calls: list[bool] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        torch.testing.assert_close(value, self._expected)
        self.calls.append(True)
        return value * self._multiplier


def _config(rank: int, ep_size: int) -> glm5_next.Glm5NextConfig:
    return glm5_next.Glm5NextConfig.from_dict(
        dict(
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=8,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_layers=1,
            first_k_dense_replace=0,
            tp_size=2,
            tp_rank=rank % 2,
            cp_size=2,
            cp_rank=rank // 2,
            ep_size=ep_size,
            ep_rank=rank // (4 // ep_size),
            moe_tp_size=4 // ep_size,
            moe_tp_rank=rank % (4 // ep_size),
        )
    )


@pytest.mark.parametrize("ep_size", [1, 2, 4])
@pytest.mark.parametrize("rank", range(4))
def test_cp_config_preserves_independent_attention_and_moe_axes(rank: int, ep_size: int) -> None:
    config = _config(rank, ep_size)
    assert (config.cp_size, config.cp_rank) == (2, rank // 2)
    assert config.tp_size * config.cp_size == config.ep_size * config.moe_tp_size


def test_bf16_cp_is_rejected_before_using_tp_only_reduction() -> None:
    layer = glm5_next.Glm5NextMoE(_config(0, 1), torch.bfloat16, torch.device("cpu"))
    with pytest.raises(NotImplementedError, match="CP supports W8A8"):
        layer(torch.zeros(2, 8, dtype=torch.bfloat16))


@pytest.mark.parametrize("ep_size", [1, 2, 4])
@pytest.mark.parametrize("prefill", [False, True])
def test_cp_moe_reduces_global_rows_and_counts_shared_once(
    monkeypatch: pytest.MonkeyPatch, ep_size: int, prefill: bool
) -> None:
    hidden = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    partials = []
    for rank in range(4):
        config = _config(rank, ep_size)
        layer = glm5_next.Glm5NextMoE(config, torch.float32, torch.device("cpu"))
        layer.use_w8a8 = True
        for name in ("experts_w13", "experts_w2", "experts_w13_scale", "experts_w2_scale"):
            setattr(layer, name, torch.empty(0))
        shard_indices = torch.tensor([0, 3, 0]) if rank < 2 else torch.tensor([1, 2, 0])
        valid = torch.tensor([True, True, False])
        context = SimpleNamespace(shard_gather_index=shard_indices, shard_valid_mask=valid) if prefill else None
        local = hidden.index_select(0, shard_indices) * valid[:, None] if prefill else hidden
        monkeypatch.setattr(
            glm5_next, "get_forward_context", lambda context=context: SimpleNamespace(cp_context=context)
        )

        def merge_rows(
            value: torch.Tensor,
            ctx: object,
            expected_context: object = context,
            expected_local: torch.Tensor = local,
        ) -> torch.Tensor:
            assert ctx is expected_context
            torch.testing.assert_close(value, expected_local)
            return hidden.clone()

        monkeypatch.setattr(glm5_next, "cp_merge_rows", merge_rows)
        shared = _SharedPartial(hidden, config.tp_rank + 1)
        layer.shared_experts = shared
        monkeypatch.setattr(
            glm5_next.kernels,
            "moe_gate_routing",
            lambda *args, **kwargs: (torch.ones(4, 2), torch.zeros(4, 2, dtype=torch.int32)),
            raising=False,
        )

        def grouped(value: torch.Tensor, *args: object, rank: int = rank, **kwargs: object) -> torch.Tensor:
            torch.testing.assert_close(value, hidden)
            return value * (rank + 1)

        monkeypatch.setattr(glm5_next.kernels, "grouped_moe_with_selected_experts", grouped, raising=False)
        captures = []
        monkeypatch.setattr(
            glm5_next.distributed,
            "moe_tp_all_reduce",
            lambda value, captures=captures: captures.append(value.clone()),
            raising=False,
        )
        monkeypatch.setattr(
            glm5_next.distributed,
            "moe_ep_all_reduce",
            lambda value, captures=captures: captures.append(value.clone()),
            raising=False,
        )
        output = layer(local)
        expected = hidden * ((rank + 1) * config.routed_scaling_factor)
        if config.cp_rank == 0:
            expected = expected + hidden * (config.tp_rank + 1)
        assert len(shared.calls) == int(config.cp_rank == 0)
        assert len(captures) == int(config.moe_tp_size > 1) + int(config.ep_size > 1)
        for value in captures:
            torch.testing.assert_close(value, expected)
        partials.append(captures[0])
        expected_local = expected.index_select(0, shard_indices) * valid[:, None] if prefill else expected
        torch.testing.assert_close(output, expected_local)
    # Four routed shards and only one pair of shared attention-TP shards.
    torch.testing.assert_close(torch.stack(partials).sum(0), hidden * (10 * config.routed_scaling_factor + 3))


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("empty_dp", [False, True])
def test_dp_cp_moe_keeps_unequal_request_cohorts_separate(
    monkeypatch: pytest.MonkeyPatch, prefill: bool, empty_dp: bool
) -> None:
    from xllm.python.layers import moe_parallel

    counts = (1, 3) if empty_dp else (3, 5)
    hidden = torch.arange(sum(counts) * 8, dtype=torch.float32).reshape(-1, 8)
    if empty_dp:
        hidden[:1].zero_()
    partials = []
    for rank in range(4):
        dp_rank, cp_rank = divmod(rank, 2)
        offset = sum(counts[:dp_rank])
        dp_hidden = hidden[offset : offset + counts[dp_rank]]
        config = glm5_next.Glm5NextConfig.from_dict(
            dict(
                hidden_size=8,
                intermediate_size=16,
                moe_intermediate_size=8,
                n_routed_experts=4,
                num_experts_per_tok=2,
                n_layers=1,
                first_k_dense_replace=0,
                tp_size=1,
                dp_size=2,
                dp_rank=dp_rank,
                cp_size=2,
                cp_rank=cp_rank,
                ep_size=4,
                ep_rank=rank,
                moe_tp_size=1,
                moe_tp_rank=0,
            )
        )
        indices = list(range(cp_rank, counts[dp_rank], 2))
        width = (counts[dp_rank] + 1) // 2
        valid = torch.arange(width) < len(indices)
        shard_indices = torch.tensor(indices + [0] * (width - len(indices)))
        context = SimpleNamespace(shard_gather_index=shard_indices, shard_valid_mask=valid) if prefill else None
        local = dp_hidden.index_select(0, shard_indices) * valid[:, None] if prefill else dp_hidden
        forward_context = SimpleNamespace(
            cp_context=context, metadata=SimpleNamespace(dp_execution_token_counts=counts)
        )
        monkeypatch.setattr(glm5_next, "get_forward_context", lambda ctx=forward_context: ctx)
        monkeypatch.setattr(moe_parallel, "get_forward_context", lambda ctx=forward_context: ctx)
        monkeypatch.setattr(glm5_next, "cp_merge_rows", lambda value, ctx, dp_hidden=dp_hidden: dp_hidden.clone())

        def gather(
            value: torch.Tensor,
            execution_counts: tuple[int, ...],
            index: int,
            expected: torch.Tensor = dp_hidden,
            expected_rank: int = dp_rank,
            expected_offset: int = offset,
        ) -> tuple[torch.Tensor, int]:
            torch.testing.assert_close(value, expected)
            assert execution_counts == counts and index == expected_rank
            return hidden.clone(), expected_offset

        monkeypatch.setattr(glm5_next.distributed, "gather_dp_execution_tokens", gather, raising=False)
        layer = glm5_next.Glm5NextMoE(config, torch.float32, torch.device("cpu"))
        layer.use_w8a8 = True
        for name in ("experts_w13", "experts_w2", "experts_w13_scale", "experts_w2_scale"):
            setattr(layer, name, torch.empty(0))
        shared = _SharedPartial(dp_hidden, 1)
        layer.shared_experts = shared
        monkeypatch.setattr(
            glm5_next.kernels,
            "moe_gate_routing",
            lambda *args, **kwargs: (torch.ones(len(hidden), 2), torch.zeros(len(hidden), 2, dtype=torch.int32)),
            raising=False,
        )
        monkeypatch.setattr(
            glm5_next.kernels,
            "grouped_moe_with_selected_experts",
            lambda value, *args, rank=rank, **kwargs: value * (rank + 1),
            raising=False,
        )
        captured = []
        monkeypatch.setattr(
            glm5_next.distributed,
            "moe_ep_all_reduce",
            lambda value, captured=captured: captured.append(value.clone()),
            raising=False,
        )
        output = layer(local)
        expected = hidden * ((rank + 1) * config.routed_scaling_factor)
        if cp_rank == 0:
            expected[offset : offset + counts[dp_rank]].add_(dp_hidden)
        assert len(shared.calls) == int(cp_rank == 0)
        assert len(captured) == 1
        torch.testing.assert_close(captured[0], expected)
        partials.append(captured[0])
        expected_local = expected[offset : offset + counts[dp_rank]]
        if prefill:
            expected_local = expected_local.index_select(0, shard_indices) * valid[:, None]
        torch.testing.assert_close(output, expected_local)
    torch.testing.assert_close(torch.stack(partials).sum(0), hidden * (10 * config.routed_scaling_factor + 1))
