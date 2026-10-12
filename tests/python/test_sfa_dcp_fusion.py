# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""SFA DCP remap precision: CPU golden and optional AOT kernel vs naive torch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.layers import sfa_dcp as sfa_dcp_module
from xllm.python.layers.sfa_dcp import (
    AscendSFADCPImpl,
    _can_use_fused_remap,
    _fill_wide_remap,
)
from xllm.python.layers.sfa_dcp_ref import remap_sparse_indices
from xllm.python.model_executor.forward_context import ForwardContext, forward_context

TOPK = 2048
# GLM-5.3: index_kpool=4 with always-select-tail, so the tail is index_kpool - 1.
KPOOL_TAIL_COLS = 3
# AOT fused-remap specialization bound (sfa_dcp.py): token rows beyond this
# width must take the torch reference path.
FUSED_REMAP_MAX_TOKENS = 256


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _aot_remap_available() -> bool:
    ops = getattr(torch.ops, "xllm_ops", None)
    return ops is not None and hasattr(ops, "sfa_dcp_remap_out")


def _make_logical_slots(
    num_tokens: int,
    layout: KVShardLayout,
    *,
    device: torch.device,
    width: int = TOPK,
) -> torch.Tensor:
    torch.manual_seed(0)
    slots = torch.randint(
        0,
        8 * layout.logical_block_size,
        (num_tokens, width),
        device=device,
        dtype=torch.int32,
    )
    mask = torch.rand((num_tokens, width), device=device) < 0.25
    return torch.where(mask, torch.full_like(slots, KVShardLayout.INVALID_SLOT), slots)


def _assert_valid_slots_form_one_run(slots: torch.Tensor) -> None:
    """SFA stops at the first ``-1``, so nothing valid may follow one."""
    for row in slots.tolist():
        invalid = [pos for pos, slot in enumerate(row) if slot < 0]
        if invalid:
            assert all(slot < 0 for slot in row[invalid[0] :]), f"valid slot behind the first -1: {row}"


def _cpu_forward_context() -> ForwardContext:
    return ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[],
        execution_state=None,
    )


class _IdentityDcpGroup:
    world_size = 1
    rank_in_group = 0
    device_group = None


def _naive_owned_pack(row: list[int], layout: KVShardLayout) -> list[int]:
    """Independent restatement of the remap coordinate contract.

    Walks one row of sequence-relative GLOBAL positions, keeps the slots this
    rank owns (stripe ``dcp_rank`` of their logical block), maps each to its
    LOCAL physical slot, packs them to the front in column order, and pads
    the rest with ``-1``.
    """
    owned: list[int] = []
    for slot in row:
        if slot < 0:
            continue
        block, offset = divmod(slot, layout.logical_block_size)
        stripe, local_offset = divmod(offset, layout.physical_block_size)
        if stripe == layout.dcp_rank:
            owned.append(block * layout.physical_block_size + local_offset)
    return owned + [KVShardLayout.INVALID_SLOT] * (len(row) - len(owned))


def test_remap_sparse_indices_packs_owned_slots() -> None:
    layout = KVShardLayout(physical_block_size=4, dcp_size=2, dcp_rank=0)
    slots = torch.tensor(
        [
            [0, 5, -1, 8],
            [1, 4, 9, -1],
        ],
        dtype=torch.int32,
    )
    remapped = remap_sparse_indices(slots, layout, index_topk=4)
    assert remapped.shape == slots.shape
    owned = remapped >= 0
    assert owned[0].tolist() == [True, True, False, False]
    assert owned[1].tolist() == [True, True, False, False]


def test_remap_sparse_indices_accepts_kpool_tail_wider_than_index_topk() -> None:
    layout = KVShardLayout(physical_block_size=4, dcp_size=2, dcp_rank=0)
    slots = torch.tensor([[0, 5, -1, 8, 12, 1, 4]], dtype=torch.int32)

    remapped = remap_sparse_indices(slots, layout, index_topk=4)

    assert remapped.shape == slots.shape
    # Owned prefix slots stay first, owned tail slots continue the same run.
    assert remapped[remapped >= 0].tolist() == [0, 4, 1]
    _assert_valid_slots_form_one_run(remapped)


def test_fill_wide_remap_matches_the_whole_row_reference() -> None:
    """The fused wide assembly has to equal packing the whole row at once.

    The AOT kernel only packs the configured prefix, so the tail is joined by
    ``_fill_wide_remap``; on CPU the packed prefix comes from the layout.
    """
    layout = KVShardLayout(physical_block_size=128, dcp_size=4, dcp_rank=0)
    slots = _make_logical_slots(4, layout, device=torch.device("cpu"), width=TOPK + KPOOL_TAIL_COLS)

    out = torch.empty_like(slots)
    _fill_wide_remap(
        out=out,
        packed_prefix=layout.pack_owned_slots(slots[..., :TOPK]),
        tail_slots=slots[..., TOPK:],
        layout=layout,
    )

    assert torch.equal(out, remap_sparse_indices(slots, layout, index_topk=TOPK))
    assert int((out >= 0).sum()) == int((layout.localize_slots(slots) >= 0).sum())
    _assert_valid_slots_form_one_run(out)


def test_fill_wide_remap_keeps_a_fully_owned_tail_contiguous() -> None:
    """With every slot owned there is no ``-1`` padding to hide the tail behind."""
    layout = KVShardLayout(physical_block_size=128, dcp_size=1, dcp_rank=0)
    slots = torch.cat(
        [
            torch.arange(TOPK, dtype=torch.int32),
            torch.arange(TOPK, TOPK + KPOOL_TAIL_COLS, dtype=torch.int32),
        ]
    ).unsqueeze(0)

    out = torch.empty_like(slots)
    _fill_wide_remap(
        out=out,
        packed_prefix=layout.pack_owned_slots(slots[..., :TOPK]),
        tail_slots=slots[..., TOPK:],
        layout=layout,
    )

    assert torch.equal(out, slots)
    _assert_valid_slots_form_one_run(out)


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
@pytest.mark.skipif(
    not _aot_remap_available(),
    reason="TileLang AOT sfa_dcp_remap_out is not registered",
)
def test_fused_remap_matches_naive() -> None:
    """Both widths the indexer emits: exactly ``index_topk``, and the wider
    kPool tail (``topk + index_kpool - 1``, what GLM-5.3 ships). The fused path
    packs the prefix with the AOT kernel and joins the tail through
    ``_fill_wide_remap``, which is what the model feeds SFA.
    """
    device = torch.device("npu")
    cases = (
        (128, 4, 2, 1),
        (128, 4, 2, 8),
        (64, 2, 1, 8),
        (128, 32, 7, 8),
    )
    for physical_block_size, dcp_size, dcp_rank, num_tokens in cases:
        for width in (TOPK, TOPK + KPOOL_TAIL_COLS):
            layout = KVShardLayout(
                physical_block_size=physical_block_size,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
            )
            slots = _make_logical_slots(num_tokens, layout, device=device, width=width)
            prefix = slots[..., :TOPK].contiguous()
            out = torch.empty_like(prefix)
            scratch = torch.empty(num_tokens * TOPK, dtype=torch.int32, device=device)
            fused = torch.ops.xllm_ops.sfa_dcp_remap_out(
                prefix,
                layout.physical_block_size,
                layout.dcp_size,
                layout.dcp_rank,
                out,
                scratch,
            )
            torch.npu.synchronize()
            if width > TOPK:
                wide = torch.empty_like(slots)
                _fill_wide_remap(
                    out=wide,
                    packed_prefix=fused,
                    tail_slots=slots[..., TOPK:],
                    layout=layout,
                )
                fused = wide
            naive = remap_sparse_indices(slots, layout, index_topk=TOPK)
            _assert_valid_slots_form_one_run(fused)
            assert torch.equal(fused, naive), (
                f"remap mismatch pb={physical_block_size} dcp={dcp_size} T={num_tokens} width={width}"
            )


def test_kpool_topk_coordinates_remap_through_production_entry() -> None:
    """M11.3 combination pin: glm5_next kPool ``select_topk`` output
    coordinates are sequence-relative GLOBAL positions, and the production
    remap entry (``AscendSFADCPImpl._remap_sparse_indices``) must map them to
    this rank's LOCAL physical slots with the owned run packed to the front
    and peer-owned positions dropped to ``-1``.

    The row below is GLM-5.3-shaped (width ``index_topk + index_kpool - 1``
    = 2048 + 3, int32): the prefix holds the kPool top-k and the trailing
    columns hold the always-select-tail positions. Expected values are
    hand-derived from the layout arithmetic, not from the code under test.
    """
    layout = KVShardLayout(physical_block_size=128, dcp_size=2, dcp_rank=0)
    impl = AscendSFADCPImpl(
        _IdentityDcpGroup(),
        scale=0.1,
        index_topk=TOPK,
        layout=layout,
    )
    width = TOPK + KPOOL_TAIL_COLS
    slots = torch.full((1, width), KVShardLayout.INVALID_SLOT, dtype=torch.int32)
    # Prefix (top-k): positions 5 / 130 / 260 / 300 / 550 of a sequence that
    # spans logical blocks of 256 tokens; rank 0 owns stripe 0 (offsets
    # 0-127) of every block.
    #   5   -> block 0 stripe 0 -> local   0*128+5   = 5
    #   130 -> block 0 stripe 1 -> peer (-1)
    #   260 -> block 1 stripe 0 -> local 1*128+4    = 132
    #   300 -> block 1 stripe 0 -> local 1*128+44   = 172
    #   550 -> block 2 stripe 0 -> local 2*128+38   = 294
    slots[0, :5] = torch.tensor([5, 130, 260, 300, 550], dtype=torch.int32)
    # Tail (always-select): 598 / 599 / 620 are all on block 2 stripe 0
    # (offsets 86 / 87 / 108) -> local 342 / 343 / 364.
    slots[0, TOPK:] = torch.tensor([598, 599, 620], dtype=torch.int32)

    with forward_context(_cpu_forward_context()):
        remapped = impl._remap_sparse_indices(slots)

    assert remapped.shape == slots.shape
    expected = torch.full((1, width), KVShardLayout.INVALID_SLOT, dtype=torch.int32)
    expected[0, :7] = torch.tensor([5, 132, 172, 294, 342, 343, 364], dtype=torch.int32)
    torch.testing.assert_close(remapped, expected)
    _assert_valid_slots_form_one_run(remapped)
    # The contract, restated independently: same expectation from the naive
    # walk over global positions.
    assert remapped[0].tolist() == _naive_owned_pack(slots[0].tolist(), layout)


def test_wide_spec_verify_remap_falls_back_and_stays_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M11.3 combination pin: a spec-verify batch wider than the AOT fused
    remap's 256-token specialization (``_can_use_fused_remap``) must fall
    back to the torch reference path, and that fallback must be CORRECT --
    the whole row (prefix and kPool tail) packed as one owned run.

    Part 1 pins the gate itself with every other fused precondition held;
    part 2 drives the production entry at 300 rows and checks the result
    against both the whole-row reference and an independent naive walk.
    """
    layout = KVShardLayout(physical_block_size=128, dcp_size=2, dcp_rank=0)

    # Part 1: the width constraint in isolation. All other fused-path
    # preconditions hold (index_topk, power-of-two sizes, int32, contiguous,
    # NPU tensor, AOT op registered); only the token count varies.
    with monkeypatch.context() as m:
        m.setattr(sfa_dcp_module, "_is_npu_tensor", lambda _tensor: True)
        m.setattr(sfa_dcp_module, "_fused_remap_available", lambda: True)
        at_limit = torch.zeros((FUSED_REMAP_MAX_TOKENS, TOPK), dtype=torch.int32)
        assert _can_use_fused_remap(
            at_limit,
            index_topk=TOPK,
            physical_block_size=128,
            dcp_size=2,
        )
        beyond_limit = torch.zeros((FUSED_REMAP_MAX_TOKENS + 1, TOPK), dtype=torch.int32)
        assert not _can_use_fused_remap(
            beyond_limit,
            index_topk=TOPK,
            physical_block_size=128,
            dcp_size=2,
        )

    # Part 2: the fallback. 300 spec-verify token rows exceed the fused
    # bound, so the production entry must take the reference path on this
    # CPU tensor and still produce the full packed result.
    impl = AscendSFADCPImpl(
        _IdentityDcpGroup(),
        scale=0.1,
        index_topk=TOPK,
        layout=layout,
    )
    num_tokens = FUSED_REMAP_MAX_TOKENS + 44
    slots = _make_logical_slots(
        num_tokens,
        layout,
        device=torch.device("cpu"),
        width=TOPK + KPOOL_TAIL_COLS,
    )
    with forward_context(_cpu_forward_context()):
        remapped = impl._remap_sparse_indices(slots)

    assert remapped.shape == slots.shape
    # The fallback equals the whole-row reference: prefix and tail packed
    # together, nothing truncated at the fused kernel's width.
    torch.testing.assert_close(remapped, remap_sparse_indices(slots, layout, index_topk=TOPK))
    _assert_valid_slots_form_one_run(remapped)
    # Independent contract check on sampled rows (first, middle, last).
    for row_idx in (0, num_tokens // 2, num_tokens - 1):
        assert remapped[row_idx].tolist() == _naive_owned_pack(slots[row_idx].tolist(), layout)
