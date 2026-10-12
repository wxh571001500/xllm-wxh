# Copyright 2026 The xLLM Authors.
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

"""Check MC2 preparation with the framework's typed execution context."""

import pytest
import torch

from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.model_executor.runners.decode_acl_graph import _StaticAttentionMetadata
from xllm.python.models.kimi_k3_moe import MC2ExecutionContext, MC2PrepareAndFinalize, MoEParallelConfig


def _context(counts: tuple[int, ...], phase: str, mc2: MC2ExecutionContext | None) -> ForwardContext:
    metadata = _StaticAttentionMetadata(
        slot_mapping=torch.empty(0, dtype=torch.int32),
        paged_kv_indptr=torch.empty(0, dtype=torch.int32),
        paged_kv_indices=torch.empty(0, dtype=torch.int32),
        paged_kv_last_page_len=torch.empty(0, dtype=torch.int32),
        dp_execution_token_counts=counts,
    )
    return ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[],
        acl_graph=AclGraphCaptureContext(stream=object(), tasks=[]) if phase == "capture" else None,
        execution_state=AclGraphExecutionState(persistent_buffers={}) if phase != "eager" else None,
        execution_contexts={MC2ExecutionContext: mc2} if mc2 is not None else {},
    )


@pytest.mark.parametrize("counts", [(1, 0, 0, 0), (1, 0, 3, 2), (17, 1, 0, 2)])
@pytest.mark.parametrize("phase", ["eager", "warmup", "capture"])
def test_mc2_partition_and_mask(monkeypatch: pytest.MonkeyPatch, counts: tuple[int, ...], phase: str) -> None:
    def unexpected_gather(self: MC2PrepareAndFinalize, value: torch.Tensor) -> torch.Tensor:
        pytest.fail("published scheduler counts or a static bucket must not require token-count all-gather")

    monkeypatch.setattr(MC2PrepareAndFinalize, "_gather_token_counts", unexpected_gather)
    graph = phase != "eager"
    padded = (max(counts) + 15) // 16 * 16
    masks = []
    for dp_rank, count in enumerate(counts):
        hidden = torch.arange(max(count, 1) * 4).reshape(-1, 4).float()
        logits = torch.arange(max(count, 1) * 7).reshape(-1, 7).float()
        mc2 = MC2ExecutionContext(padded, torch.arange(padded) < count) if graph else None
        context = _context(counts, phase, mc2)
        assert not hasattr(context, "padded_num_tokens")
        assert not hasattr(context, "mc2_mask")
        parts = []
        for tp_rank in range(16):
            config = MoEParallelConfig(
                tp_size=1,
                tp_rank=0,
                ep_size=64,
                ep_rank=dp_rank * 16 + tp_rank,
                dp_size=4,
                dp_rank=dp_rank,
                input_tp_size=16,
                input_tp_rank=tp_rank,
            )
            with forward_context(context):
                part = MC2PrepareAndFinalize(config).prepare(hidden, logits)
            assert part.hidden_states.shape == (padded // 16, 4)
            assert part.router_logits.shape == (padded // 16, 7)
            assert part.active_mask.shape == (padded // 16,)
            masks.append(part.active_mask)
            parts.append(part)
        restored_hidden = torch.cat([part.hidden_states for part in parts])
        restored_logits = torch.cat([part.router_logits for part in parts])
        restored_mask = torch.cat([part.active_mask for part in parts])
        torch.testing.assert_close(restored_hidden[:count], hidden[:count])
        torch.testing.assert_close(restored_logits[:count], logits[:count])
        assert restored_mask.tolist() == (torch.arange(padded) < count).tolist()
        if mc2 is not None:
            mc2.active_mask.zero_()
            assert all(not part.active_mask.any() for part in parts)
    if not graph:
        assert sum(int(mask.sum()) for mask in masks) == sum(counts)


@pytest.mark.parametrize("padded,mask_size", [(1, 1), (16, 1), (16, 32)])
def test_mc2_rejects_invalid_static_mask(padded: int, mask_size: int) -> None:
    context = _context((1, 0, 0, 0), "capture", MC2ExecutionContext(padded, torch.ones(mask_size, dtype=torch.bool)))
    config = MoEParallelConfig(
        tp_size=1, tp_rank=0, ep_size=64, ep_rank=15, dp_size=4, dp_rank=0, input_tp_size=16, input_tp_rank=15
    )
    with forward_context(context), pytest.raises(ValueError, match="MC2"):
        MC2PrepareAndFinalize(config).prepare(torch.ones(1, 4), torch.ones(1, 7))


@pytest.mark.parametrize("phase", ["warmup", "capture"])
@pytest.mark.parametrize("tokens", [0, 1, 3])
def test_mc2_nonreplicated_graph_uses_static_context(monkeypatch: pytest.MonkeyPatch, phase: str, tokens: int) -> None:
    monkeypatch.setattr(
        MC2PrepareAndFinalize, "_gather_token_counts", lambda *args: pytest.fail("graph must not gather device counts")
    )
    config = MoEParallelConfig(tp_size=1, tp_rank=0, ep_size=2, ep_rank=0)
    mask = torch.arange(4) < tokens
    context = _context((), phase, MC2ExecutionContext(4, mask))
    hidden, logits = torch.ones(tokens, 2), torch.ones(tokens, 4)
    prepare = MC2PrepareAndFinalize(config)
    with forward_context(context):
        result = prepare.prepare(hidden, logits)
        restored = prepare.finalize(result.hidden_states, reduce_results=False)
    assert result.hidden_states.shape == (4, 2)
    assert result.active_mask is mask
    torch.testing.assert_close(restored, hidden)


@pytest.mark.parametrize("phase", ["warmup", "capture"])
def test_mc2_nonreplicated_graph_rejects_missing_context(phase: str) -> None:
    config = MoEParallelConfig(tp_size=1, tp_rank=0, ep_size=2, ep_rank=0)
    with forward_context(_context((), phase, None)), pytest.raises(ValueError, match="static token bucket"):
        MC2PrepareAndFinalize(config).prepare(torch.ones(1, 2), torch.ones(1, 4))
