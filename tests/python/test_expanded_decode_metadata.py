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

"""Logical-block-size validation of the expanded decode (spec-verify) metadata.

One block-table column is a *logical* block: with KV splitting it spans
``block_size * kv_split_size`` tokens, so the page count of a sequence has to be
computed against that logical size. Validating against the physical block size
rejects every sequence that merely crosses a physical block boundary.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention.expanded_decode_metadata import (
    resolve_expanded_decode_metadata,
)

# block_size=128 with kv_split_size=4.
_LOGICAL_BLOCK_SIZE = 512
_PHYSICAL_BLOCK_SIZE = 128
_PAGE_COUNT_ERROR = "page count exceeds block-table capacity"


def _metadata(
    kv_seq_len: int,
    block_table_columns: int,
    logical_block_size: int | None,
) -> SimpleNamespace:
    expanded = SimpleNamespace(
        enabled=True,
        kv_seq_lens=torch.tensor([kv_seq_len], dtype=torch.int32),
        block_table=torch.zeros((1, block_table_columns), dtype=torch.int32),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=[kv_seq_len],
    )
    metadata = SimpleNamespace(
        expanded_decode_metadata=expanded,
        slot_mapping=torch.zeros(1, dtype=torch.int32),
    )
    if logical_block_size is not None:
        metadata.logical_block_size = logical_block_size
    return metadata


def test_one_logical_block_covers_the_host_kv_length() -> None:
    """200 tokens stay inside a single 512-token logical block."""
    resolved = resolve_expanded_decode_metadata(_metadata(200, 1, _LOGICAL_BLOCK_SIZE))
    assert resolved is not None
    assert resolved.kv_seq_lens.tolist() == [200]


def test_physical_block_size_rejects_the_same_host_kv_length() -> None:
    """The same sequence needs two physical 128-token blocks."""
    metadata = _metadata(200, 1, None)
    with pytest.raises(RuntimeError, match=_PAGE_COUNT_ERROR):
        resolve_expanded_decode_metadata(metadata, block_size=_PHYSICAL_BLOCK_SIZE)


def test_host_kv_length_crosses_into_the_next_logical_block() -> None:
    """513 is the first length that needs the second block-table column."""
    with pytest.raises(RuntimeError, match=_PAGE_COUNT_ERROR):
        resolve_expanded_decode_metadata(_metadata(513, 1, _LOGICAL_BLOCK_SIZE))

    resolved = resolve_expanded_decode_metadata(_metadata(513, 2, _LOGICAL_BLOCK_SIZE))
    assert resolved is not None
    assert resolved.block_table.shape[1] == 2
