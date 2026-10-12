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

"""Tests for the NPU ACL decode-graph runner."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.dsa_metadata import DsaMetadataBuilder, build_cache_specs
from xllm.python.attention.expanded_decode_metadata import ExpandedDecodeMetadata
from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
)


def _runner() -> DecodeAclGraphRunner:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    return DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
    )


def _metadata(linear_state_indices: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        slot_mapping=torch.arange(4, dtype=torch.int32),
        paged_kv_indptr=torch.arange(5, dtype=torch.int32),
        paged_kv_indices=torch.tensor([10, 20, 30, 40], dtype=torch.int32),
        paged_kv_last_page_len=torch.arange(1, 5, dtype=torch.int32),
        block_table=torch.tensor(
            [[10, 0], [20, 0], [30, 0], [40, 0]],
            dtype=torch.int32,
        ),
        kv_seq_lens=torch.arange(1, 5, dtype=torch.int32),
        kv_seq_lens_host_values=[1, 2, 3, 4],
        kv_cu_seq_lens=torch.tensor([0, 1, 3, 6, 10], dtype=torch.int32),
        q_cu_seq_lens=torch.arange(5, dtype=torch.int32),
        linear_state_indices=linear_state_indices,
        num_accepted_tokens=torch.tensor([1, 2, 1, 2], dtype=torch.int32),
        expanded_decode_metadata=None,
        multi_block_tables=(),
        new_cache_slots_host_values=[0, 1, 2, 3],
        is_prefill=False,
        is_chunked_prefill=False,
    )


def test_eplv2_graph_admission_uses_padded_token_rows_for_dflash2() -> None:
    runner = _runner()
    runner.max_batch = 128
    runner.num_decoding_tokens = 8
    runner._eplv2_graph_token_limit = 16
    metadata = _metadata(torch.arange(3, dtype=torch.int32))
    with (
        patch.object(runner, "_has_compatible_decode_metadata", return_value=True),
        patch(
            "xllm.python.model_executor.runners.decode_acl_graph.resolve_expanded_decode_metadata",
            return_value=object(),
        ),
    ):
        # One and two 8-row sequences fit; three sequences would capture a
        # padded 32-row graph even though the active input has only 24 rows.
        for sequences, expected in ((1, True), (2, True), (3, False)):
            metadata.linear_state_indices = torch.arange(sequences)
            assert runner.can_execute(torch.arange(sequences * 8), metadata) is expected


def test_accepted_tokens_use_live_per_sequence_graph_buffer() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 3, 7, 7], dtype=torch.int32))
    metadata.num_accepted_tokens = torch.tensor([4, 2], dtype=torch.int32)
    entry = runner._allocate_entry(8, input_ids, positions, metadata)
    static_counts = entry.static_metadata.num_accepted_tokens
    address = static_counts.data_ptr()
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
        assert static_counts.tolist() == [4, 2, 1, 1, 1, 1, 1, 1]
        metadata.num_accepted_tokens.copy_(torch.tensor([1, 3]))
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
        assert static_counts.tolist() == [1, 3, 1, 1, 1, 1, 1, 1]
        metadata.num_accepted_tokens = torch.tensor([2], dtype=torch.int32)
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
        assert static_counts.tolist() == [2, 1, 1, 1, 1, 1, 1, 1]
        metadata.num_accepted_tokens = None
        metadata.is_dummy = True
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
    assert static_counts.data_ptr() == address
    assert static_counts.tolist() == [1] * 8

    plain_metadata = _metadata(torch.tensor([3, 7, 11, 15], dtype=torch.int32))
    plain_metadata.num_accepted_tokens = None
    plain_entry = runner._allocate_entry(8, input_ids, positions, plain_metadata)
    assert plain_entry.static_metadata.num_accepted_tokens is None


def test_verify_graph_key_tracks_width_not_active_sequence_count() -> None:
    runner = _runner()
    captured_keys = []

    def capture_key(key: tuple) -> None:
        captured_keys.append(key)
        raise LookupError("graph key recorded")

    runner._graphs = SimpleNamespace(get=capture_key)
    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.resolve_expanded_decode_metadata",
        return_value=object(),
    ):
        for sequence_count, width in ((3, 4), (4, 4), (8, 2)):
            input_ids = torch.arange(sequence_count * width, dtype=torch.int32)
            metadata = SimpleNamespace(linear_state_indices=torch.arange(sequence_count))
            with pytest.raises(LookupError, match="graph key recorded"):
                runner.execute(input_ids, input_ids, metadata)
    assert captured_keys[0] == captured_keys[1]
    assert captured_keys[1] != captured_keys[2]


@pytest.mark.parametrize(
    "accepted_counts,error_type",
    [
        (None, RuntimeError),
        (torch.ones(2, 2, dtype=torch.int32), ValueError),
        (torch.ones(0, dtype=torch.int32), ValueError),
        (torch.ones(5, dtype=torch.int32), ValueError),
    ],
)
def test_accepted_tokens_reject_missing_or_invalid_replay_metadata(
    accepted_counts: torch.Tensor | None, error_type: type[Exception]
) -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 3, 7, 7], dtype=torch.int32))
    metadata.num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    entry = runner._allocate_entry(8, input_ids, positions, metadata)
    entry.static_metadata.is_spec_verify = True
    metadata.num_accepted_tokens = accepted_counts
    with (
        patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True),
        pytest.raises(error_type, match="accepted-token metadata"),
    ):
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)


def test_slice_output_preserves_aux_hidden_tuple() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    aux_hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    output = DecodeAclGraphRunner._slice_output((hidden, aux_hidden), 3)

    assert isinstance(output, tuple)
    torch.testing.assert_close(output[0], hidden[:3])
    torch.testing.assert_close(output[1], aux_hidden[:3])


def test_linear_state_graph_buffers_are_stable() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 7, 11, 15], dtype=torch.int32))
    metadata.kpool_query_lens = [4]
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )
    static_indices = entry.static_metadata.linear_state_indices
    static_num_accepted_tokens = entry.static_metadata.num_accepted_tokens
    data_ptr = static_indices.data_ptr()
    accepted_data_ptr = static_num_accepted_tokens.data_ptr()
    assert entry.static_metadata.kpool_query_lens == (4, 4)
    assert entry.static_metadata.kpool_query_lens_device.tolist() == [4, 4]
    assert DecodeAclGraphRunner._graph_key(
        8,
        True,
        None,
        entry.static_metadata.kpool_query_lens,
    ) != DecodeAclGraphRunner._graph_key(8, True, None, (2, 2, 2, 2))

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )
        assert static_indices.tolist() == [3, 7, 11, 15, 0, 0, 0, 0]
        metadata.linear_state_indices = torch.tensor(
            [4, 8, 12, 16],
            dtype=torch.int32,
        )
        metadata.num_accepted_tokens = torch.tensor(
            [2, 1, 2, 1],
            dtype=torch.int32,
        )
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert static_indices.data_ptr() == data_ptr
    assert static_indices.tolist() == [4, 8, 12, 16, 0, 0, 0, 0]
    assert static_num_accepted_tokens.data_ptr() == accepted_data_ptr
    assert static_num_accepted_tokens.tolist() == [2, 1, 2, 1, 1, 1, 1, 1]


def test_dflash_proposal_mode_uses_a_distinct_graph_key() -> None:
    ordinary_key = DecodeAclGraphRunner._graph_key(8, False, None)
    proposal_key = DecodeAclGraphRunner._graph_key(
        8,
        False,
        None,
        is_dflash_proposal=True,
    )

    assert ordinary_key != proposal_key


def test_linear_state_snapshot_restores_all_ssm_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda: None),
        raising=False,
    )
    runner = _runner()
    conv = torch.arange(48, dtype=torch.float32).reshape(4, 2, 6)
    ssm = torch.arange(48, dtype=torch.float32).reshape(12, 1, 2, 2)
    runner.bind_layer_caches(
        [
            SimpleNamespace(
                conv=conv,
                ssm=ssm,
            )
        ]
    )
    entry = SimpleNamespace(
        static_metadata=SimpleNamespace(
            linear_state_indices=torch.tensor([1, 1, 3, 3], dtype=torch.int64),
        )
    )

    snapshot = runner._snapshot_linear_state(entry)
    assert torch.equal(snapshot[0][2], torch.tensor([1, 3], dtype=torch.int64))
    assert torch.equal(snapshot[0][3], torch.tensor([3, 4, 5, 9, 10, 11], dtype=torch.int64))
    expected_conv = conv.clone()
    expected_ssm = ssm.clone()
    conv.zero_()
    ssm.zero_()
    runner._restore_linear_state(entry, snapshot)

    assert torch.equal(conv[[1, 3]], expected_conv[[1, 3]])
    checkpoint_indices = torch.tensor([3, 4, 5, 9, 10, 11])
    assert torch.equal(ssm[checkpoint_indices], expected_ssm[checkpoint_indices])


def test_explicit_verify_reuses_row_aligned_paging_metadata() -> None:
    runner = _runner()
    metadata = _metadata(torch.tensor([1, 2], dtype=torch.int32))
    metadata.expanded_decode_metadata = ExpandedDecodeMetadata(
        kv_seq_lens=metadata.kv_seq_lens,
        block_table=metadata.block_table,
        paged_kv_indptr=metadata.paged_kv_indptr,
        paged_kv_indices=metadata.paged_kv_indices,
        paged_kv_last_page_len=metadata.paged_kv_last_page_len,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=metadata.kv_seq_lens_host_values,
    )
    metadata.is_spec_verify = True
    metadata.is_chunked_prefill = True
    metadata.num_accepted_tokens = torch.tensor([4, 2], dtype=torch.int32)
    metadata.q_seq_lens = torch.tensor([2, 2], dtype=torch.int32)
    metadata.q_cu_seq_lens = torch.tensor([0, 2, 4], dtype=torch.int32)
    metadata.kv_seq_lens = torch.tensor([2, 4], dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [2, 4]
    metadata.kv_cu_seq_lens = torch.tensor([0, 2, 6], dtype=torch.int32)
    metadata.block_table = metadata.block_table[1::2]

    with patch.object(
        runner,
        "_build_row_aligned_paged_kv_metadata",
        side_effect=AssertionError("row-aligned paging must come from C++"),
    ):
        resolved = runner._decode_metadata(metadata)

    assert resolved[3] is metadata.paged_kv_indptr
    assert resolved[4] is metadata.paged_kv_indices
    assert resolved[5] is metadata.paged_kv_last_page_len
    input_ids = torch.arange(4, dtype=torch.int32)
    assert runner.can_execute(input_ids, metadata)
    entry = runner._allocate_entry(8, input_ids, input_ids, metadata)
    assert entry.static_metadata.is_spec_verify
    assert entry.static_metadata.q_seq_lens.tolist() == [2, 2, 2, 2]
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=4, input_embedding=None)
    assert entry.static_metadata.num_accepted_tokens.tolist() == [4, 2, 1, 1, 1, 1, 1, 1]
    assert entry.static_metadata.linear_state_indices.tolist() == [1, 1, 2, 2, 0, 0, 0, 0]


def test_verify_graph_requires_explicit_metadata() -> None:
    runner = _runner()
    metadata = _metadata(torch.tensor([1, 2], dtype=torch.int32))
    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


@pytest.mark.parametrize("dp_size", [1, 2])
def test_fill_entry_refreshes_mega_moe_mask_in_place(dp_size: int) -> None:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        dp_size=dp_size,
        enable_mega_moe_token_mask=True,
    )
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )
    mask = entry.static_metadata.mega_moe_token_mask
    data_ptr = mask.data_ptr()

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        for remote_count in (3, 1):
            metadata.dp_execution_token_counts = (4, remote_count)[:dp_size]
            runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
            expected = [True] * 4 + [False] * 4
            if dp_size == 2:
                expected += [True] * remote_count + [False] * (8 - remote_count)
            assert mask.tolist() == expected
            assert entry.static_metadata.mega_moe_token_mask.data_ptr() == data_ptr


def test_dsa_graph_tables_use_compressed_block_counts() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    attention_backend = SimpleNamespace(
        page_size=128,
        is_mla=False,
        group_infos=group_infos,
    )
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=32768,
    )
    runner._max_blocks_per_sequence = 256

    tables = runner._build_static_multi_block_tables(4, torch.device("cpu"))

    assert [tuple(table.shape) for table in tables] == [(4, 256), (4, 16), (4, 16)]
    assert all(torch.all(table == 0) for table in tables)

    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    indptr, indices, last_page_len = runner._build_row_aligned_paged_kv_metadata(
        block_table,
        [129, 132],
    )
    assert indptr.tolist() == [0, 2, 4]
    assert indices.tolist() == [10, 11, 20, 21]
    assert last_page_len.tolist() == [1, 4]

    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.kv_seq_lens_host_values = None
    metadata.paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


def test_dsa_graph_refreshes_every_manager_and_clears_tails() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    attention_backend = SimpleNamespace(
        page_size=128,
        is_mla=False,
        group_infos=group_infos,
    )
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    static_metadata = SimpleNamespace(
        multi_block_tables=(
            torch.full((4, 4), 99, dtype=torch.int32),
            torch.full((4, 2), 99, dtype=torch.int32),
            torch.full((4, 1), 99, dtype=torch.int32),
        )
    )
    metadata = SimpleNamespace(
        multi_block_tables=(
            torch.tensor([[10, 11], [12, 13]], dtype=torch.int32),
            torch.tensor([[20], [21]], dtype=torch.int32),
            torch.tensor([[30], [31]], dtype=torch.int32),
        )
    )

    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        torch.empty(0, dtype=torch.int32),
        batch_size=2,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [10, 11, 0, 0],
        [12, 13, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert static_metadata.multi_block_tables[1].tolist() == [
        [20, 0],
        [21, 0],
        [0, 0],
        [0, 0],
    ]
    assert static_metadata.multi_block_tables[2].tolist() == [[30], [31], [0], [0]]

    metadata.multi_block_tables = (
        torch.tensor([[40]], dtype=torch.int32),
        torch.tensor([[50]], dtype=torch.int32),
        torch.tensor([[60]], dtype=torch.int32),
    )
    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        torch.empty(0, dtype=torch.int32),
        batch_size=1,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [40, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert static_metadata.multi_block_tables[1].tolist() == [
        [50, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    assert static_metadata.multi_block_tables[2].tolist() == [[60], [0], [0], [0]]


def test_dsa_graph_clamps_target_tables_to_draft_groups() -> None:
    _, group_infos = build_cache_specs([1], 128, 1)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False, group_infos=group_infos),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    static_metadata = SimpleNamespace(multi_block_tables=(torch.full((4, 2), 99, dtype=torch.int32),))
    metadata = SimpleNamespace(
        multi_block_tables=(
            torch.tensor([[10], [11]], dtype=torch.int32),
            torch.tensor([[20], [21]], dtype=torch.int32),
            torch.tensor([[30], [31]], dtype=torch.int32),
        )
    )

    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        torch.empty(0, dtype=torch.int32),
        batch_size=2,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [10, 0],
        [11, 0],
        [0, 0],
        [0, 0],
    ]


def test_dsa_graph_padding_uses_reserved_single_kv_length() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(
            page_size=128,
            is_mla=False,
            group_infos=group_infos,
        ),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(kv_seq_lens_host_values=[1, 1, 1, 1]),
    )

    runner._fill_host_metadata(entry, [9, 17], batch_size=2)

    assert entry.static_metadata.kv_seq_lens_host_values == [9, 17, 1, 1]


def test_dsa_graph_pads_new_cache_slots_to_graph_bucket() -> None:
    entry = SimpleNamespace(
        batch_size=16,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[]),
    )
    metadata = SimpleNamespace(new_cache_slots_host_values=[301, 302, 401, 402])

    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=4,
    )

    assert entry.static_metadata.new_cache_slots_host_values == [301, 302, 401, 402] + [0] * 12


def test_dsa_graph_padding_preserves_scheduler_resolved_swa_slots() -> None:
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[]),
    )
    metadata = SimpleNamespace(
        new_cache_slots_host_values=[1280, 2560, 3840],
    )
    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=3,
    )

    caches_info, group_infos = build_cache_specs([0, 4, 128, 4], 128, 4)
    builder = DsaMetadataBuilder(caches_info, group_infos)
    dsa = builder.build(
        multi_block_tables=[
            torch.tensor(
                [
                    [10, 11, 0, 0],
                    [20, 21, 0, 0],
                    [30, 31, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=torch.int32,
            ),
            torch.zeros((4, 4), dtype=torch.int32),
            torch.zeros((4, 4), dtype=torch.int32),
        ],
        kv_seq_lens=[257, 257, 257, 1],
        q_seq_lens=[1, 1, 1, 1],
        positions=torch.tensor([256, 256, 256, 0], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        new_cache_slots=entry.static_metadata.new_cache_slots_host_values,
        enable_graph=True,
        graph_block_table_capacity_cols=4,
    )

    assert dsa.slot_mappings[0][0].tolist() == [1280, 2560, 3840, 0]


def test_dsa_graph_allows_dummy_without_scheduler_cache_slots() -> None:
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[99]),
    )
    metadata = SimpleNamespace(new_cache_slots_host_values=[], is_dummy=True)

    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=1,
    )

    assert entry.static_metadata.new_cache_slots_host_values == []


def test_dsv4_decode_admits_scheduler_slots_when_top_level_mapping_is_empty() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.slot_mapping = torch.empty(0, dtype=torch.int32)
    metadata.block_table = None
    metadata.multi_block_tables = (torch.tensor([[10, 0], [20, 0], [30, 0], [40, 0]], dtype=torch.int32),)
    metadata.new_cache_slots_host_values = [301, 302, 401, 402]

    assert runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)
    assert runner._effective_slot_mapping(metadata, 4, torch.device("cpu")).tolist() == [
        301,
        302,
        401,
        402,
    ]


def test_decode_rejects_mismatched_linear_state_validity_mask() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.has_initial_state = torch.ones(3, dtype=torch.int32)

    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


def test_decode_rejects_mismatched_linear_state_indices_without_cache_probe() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(3, dtype=torch.int32))

    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


def test_fill_entry_uses_scheduler_slots_for_dsv4_decode() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.slot_mapping = torch.empty(0, dtype=torch.int32)
    metadata.new_cache_slots_host_values = [301, 302, 401, 402]
    entry = runner._allocate_entry(8, input_ids, positions, metadata)

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ) as update_metadata:
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert update_metadata.call_args.args[2].tolist() == [301, 302, 401, 402]
    assert entry.static_metadata.new_cache_slots_host_values == [301, 302, 401, 402, 0, 0, 0, 0]


def test_dp_real_and_dummy_decode_share_graph_admission() -> None:
    def make_metadata(*, is_dummy: bool) -> SimpleNamespace:
        metadata = _metadata(torch.tensor([0], dtype=torch.int32))
        metadata.slot_mapping = torch.tensor([0], dtype=torch.int32) if is_dummy else torch.empty(0, dtype=torch.int32)
        metadata.block_table = None
        metadata.multi_block_tables = (torch.tensor([[0]], dtype=torch.int32),)
        metadata.kv_seq_lens = torch.tensor([1], dtype=torch.int32)
        metadata.kv_seq_lens_host_values = [1]
        metadata.kv_cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32)
        metadata.q_cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32)
        metadata.paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
        metadata.paged_kv_indices = torch.tensor([0], dtype=torch.int32)
        metadata.paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
        metadata.new_cache_slots_host_values = [] if is_dummy else [301]
        metadata.dp_execution_token_counts = (1, 1)
        metadata.dp_is_decode = (1, 1)
        metadata.is_dummy = is_dummy
        return metadata

    real_runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )
    dummy_runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
    )
    input_ids = torch.tensor([1], dtype=torch.int32)

    assert real_runner.can_execute(input_ids, make_metadata(is_dummy=False))
    assert dummy_runner.can_execute(input_ids, make_metadata(is_dummy=True))


def test_dsa_graph_positions_are_refreshed_from_current_input() -> None:
    runner = _runner()
    entry = SimpleNamespace(
        batch_size=4,
        static_positions=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        static_metadata=SimpleNamespace(dsa_positions=None),
    )

    runner._fill_graph_dsa_positions(
        entry,
        torch.tensor([20, 21], dtype=torch.int32),
    )

    assert entry.static_metadata.dsa_positions.tolist() == [20, 21, 0, 0]


def _dp_metadata(
    token_counts: tuple[int, int],
    dp_is_decode: tuple[int, int] = (1, 1),
) -> SimpleNamespace:
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        dp_execution_token_counts=tuple(1 if count == 0 else count for count in token_counts),
        dp_is_decode=dp_is_decode,
    )


def test_dp_empty_rank_uses_group_wide_acl_graph_bucket() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
    )

    with patch.object(
        runner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runner.can_execute(
            torch.zeros(1, dtype=torch.int32),
            _dp_metadata((5, 0)),
        )


def test_dp_mtp_target_dummy_rank_uses_shared_graph_bucket() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    active_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_decoding_tokens=4,
    )
    empty_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
        num_decoding_tokens=4,
    )
    draft_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_decoding_tokens=1,
    )
    metadata = _dp_metadata((4, 0))

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert active_runner.can_execute(torch.zeros(4, dtype=torch.int32), metadata)
        assert empty_runner.can_execute(torch.zeros(1, dtype=torch.int32), metadata)
        assert draft_runner.can_execute(torch.zeros(4, dtype=torch.int32), metadata)


def test_dp_expanded_verify_and_dummy_rank_share_graph_decision() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=3,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=4,
        )
        for rank in range(2)
    ]
    active = _expanded_verify_metadata(4)
    active.dp_execution_token_counts = (4, 1)
    active.raw_dp_execution_token_counts = (4, 0)
    active.is_spec_verify = True
    active.is_chunked_prefill = True
    dummy = _dp_metadata((4, 0))
    dummy.raw_dp_execution_token_counts = (4, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0].can_execute(torch.zeros(4, dtype=torch.int32), active)
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy)
        busy_plan = runners[0]._shared_graph_plan(active, 4)
        empty_plan = runners[1]._shared_graph_plan(dummy, 1)
        assert busy_plan == empty_plan == (True, 4, 1)

    wide_runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=4,
        )
        for rank in range(2)
    ]
    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert wide_runners[0].can_execute(torch.zeros(4, dtype=torch.int32), active)
        assert wide_runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy)


def test_dp_plain_decode_with_empty_rank_keeps_width_one() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=3,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=4,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((4, 0))
    active.raw_dp_execution_token_counts = (4, 0)
    dummy = _dp_metadata((4, 0))
    dummy.raw_dp_execution_token_counts = (4, 0)

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 4) == (False, 1, 4)
        assert runners[1]._shared_graph_plan(dummy, 1) == (False, 1, 4)
        assert runners[0].can_execute(torch.zeros(4, dtype=torch.int32), active) is False
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy) is False


def test_dp_empty_spec_rank_matches_busy_verify_graph_key() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=3,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=4,
        )
        for rank in range(2)
    ]
    active = _expanded_verify_metadata(4)
    active.dp_execution_token_counts = (4, 1)
    active.raw_dp_execution_token_counts = (4, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((4, 0))
    dummy.raw_dp_execution_token_counts = (4, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        busy_plan = runners[0]._shared_graph_plan(active, 4)
        empty_plan = runners[1]._shared_graph_plan(dummy, 1)
        assert busy_plan == (True, 4, 1)
        assert empty_plan == busy_plan
        busy_key = DecodeAclGraphRunner._graph_key(4, busy_plan[0], None, (), verify_width=busy_plan[1])
        empty_key = DecodeAclGraphRunner._graph_key(4, empty_plan[0], None, (), verify_width=empty_plan[1])
        assert busy_key == empty_key
        assert runners[0].can_execute(torch.zeros(4, dtype=torch.int32), active)
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy)

        unmarked = _expanded_verify_metadata(4)
        unmarked.dp_execution_token_counts = (4, 1)
        unmarked.raw_dp_execution_token_counts = (4, 0)
        unmarked.is_spec_verify = False
        with pytest.raises(RuntimeError, match="group-wide spec verify"):
            runners[0]._shared_graph_plan(unmarked, 4)


def test_dflash_draft_graph_capacity_uses_sequence_count() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=1,
            is_spec_draft=True,
            draft_query_width=8,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((16, 0))
    active.raw_dp_execution_token_counts = (16, 0)
    dummy = _dp_metadata((16, 0))
    dummy.raw_dp_execution_token_counts = (16, 0)
    misaligned = _dp_metadata((10, 0))
    misaligned.raw_dp_execution_token_counts = (10, 0)

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 16) == (False, 1, 2)
        assert runners[1]._shared_graph_plan(dummy, 1) == (False, 1, 2)
        assert runners[0].can_execute(torch.zeros(16, dtype=torch.int32), active)
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy)
        assert runners[0].can_execute(torch.zeros(10, dtype=torch.int32), misaligned) is False
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), misaligned) is False


def test_dp_uneven_sequences_stay_in_graph_with_full_batch_capacity() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=6,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((48, 0))
    active.raw_dp_execution_token_counts = (48, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((48, 0))
    dummy.raw_dp_execution_token_counts = (48, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0].max_batch == 8
        assert runners[0].can_execute(torch.zeros(48, dtype=torch.int32), active)
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy)


def test_dp_spec_verify_width_six_pads_to_bucket_eight() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=64,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=6,
        )
        for rank in range(2)
    ]
    active = _expanded_verify_metadata(6)
    active.dp_execution_token_counts = (6, 6)
    active.raw_dp_execution_token_counts = (6, 6)
    active.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 6) == (True, 6, 1)
        assert runners[0].can_execute(torch.zeros(6, dtype=torch.int32), active)
        assert runners[1].can_execute(torch.zeros(6, dtype=torch.int32), active)


def test_dp_spec_verify_refuses_bucket_not_divisible_by_width() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=8,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=3,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((2, 0))
    active.raw_dp_execution_token_counts = (2, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((2, 0))
    dummy.raw_dp_execution_token_counts = (2, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 2) == (True, 3, None)
        assert runners[1]._shared_graph_plan(dummy, 1) == (True, 3, None)
        assert runners[0].can_execute(torch.zeros(2, dtype=torch.int32), active) is False
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy) is False


def test_dp_spec_verify_uses_num_decoding_tokens_on_every_rank() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=8,
        )
        for rank in range(2)
    ]
    active = _expanded_verify_metadata(8)
    active.dp_execution_token_counts = (8, 1)
    active.raw_dp_execution_token_counts = (8, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((8, 0))
    dummy.raw_dp_execution_token_counts = (8, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0]._shared_graph_plan(active, 8) == (True, 8, 1)
        assert runners[1]._shared_graph_plan(dummy, 1) == (True, 8, 1)


def test_dp_spec_verify_rejects_measured_width_mismatch() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_decoding_tokens=8,
    )
    active = _expanded_verify_metadata(4)
    active.dp_execution_token_counts = (4, 1)
    active.raw_dp_execution_token_counts = (4, 0)
    active.is_spec_verify = True

    with pytest.raises(RuntimeError, match="num_decoding_tokens"):
        runner._shared_graph_plan(active, 4)


def test_dp_spec_verify_stays_eager_when_active_count_is_not_a_multiple() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=4,
        )
        for rank in range(2)
    ]
    active = _dp_metadata((6, 0))
    active.raw_dp_execution_token_counts = (6, 0)
    active.is_spec_verify = True
    dummy = _dp_metadata((6, 0))
    dummy.raw_dp_execution_token_counts = (6, 0)
    dummy.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert runners[0].can_execute(torch.zeros(6, dtype=torch.int32), active) is False
        assert runners[1].can_execute(torch.zeros(1, dtype=torch.int32), dummy) is False


def test_dp_kpool_spans_must_match_shared_bucket_layout() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_decoding_tokens=8,
    )
    runner.layer_caches = [SimpleNamespace(kpool_tail=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16))]
    aligned = _dp_metadata((8, 8))
    aligned.is_spec_verify = True
    mismatched = _dp_metadata((8, 8))
    mismatched.is_spec_verify = True
    mismatched.kpool_query_lens = (1,) * 8

    with patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True):
        assert runner.can_execute(torch.zeros(8, dtype=torch.int32), aligned)
        with pytest.raises(RuntimeError, match="shared bucket layout"):
            runner.can_execute(torch.zeros(8, dtype=torch.int32), mismatched)
    assert runner._graph_kpool_query_lens(aligned, 8, 8, 8) == (8,)
    assert runner._graph_kpool_query_lens(aligned, 1, 8, 8) == (8,)

    uneven = _dp_metadata((16, 16))
    uneven.is_spec_verify = True
    uneven.kpool_query_lens = (10, 6)
    short = _dp_metadata((16, 16))
    short.is_spec_verify = True
    short.kpool_query_lens = (8,)
    nonpositive = _dp_metadata((8, 8))
    nonpositive.is_spec_verify = True
    nonpositive.kpool_query_lens = (0, 8)
    with (
        patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True),
        pytest.raises(RuntimeError, match="must be uniform"),
    ):
        runner.can_execute(torch.zeros(16, dtype=torch.int32), uneven)
    with (
        patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True),
        pytest.raises(RuntimeError, match="positive lengths"),
    ):
        runner.can_execute(torch.zeros(16, dtype=torch.int32), short)
    with (
        patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True),
        pytest.raises(RuntimeError, match="positive lengths"),
    ):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), nonpositive)


def _aligned_dp_decode_metadata(rows: int) -> SimpleNamespace:
    metadata = _dp_metadata((rows, rows))
    metadata.is_spec_verify = True
    metadata.slot_mapping = torch.arange(rows, dtype=torch.int32)
    metadata.block_table = torch.zeros((rows, 2), dtype=torch.int32)
    metadata.kv_seq_lens = torch.ones(rows, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = [1] * rows
    metadata.kv_cu_seq_lens = torch.arange(rows + 1, dtype=torch.int32)
    metadata.q_cu_seq_lens = torch.arange(rows + 1, dtype=torch.int32)
    metadata.paged_kv_indptr = torch.arange(rows + 1, dtype=torch.int32)
    metadata.paged_kv_indices = torch.zeros(rows, dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.ones(rows, dtype=torch.int32)
    metadata.linear_state_indices = torch.zeros(rows, dtype=torch.int32)
    metadata.expanded_decode_metadata = None
    return metadata


def test_dp_all_decode_row_mismatch_raises() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
        num_decoding_tokens=8,
    )
    aligned = _aligned_dp_decode_metadata(8)
    mismatched = _aligned_dp_decode_metadata(8)
    mismatched.block_table = torch.zeros((1, 2), dtype=torch.int32)
    mismatched.kv_seq_lens = torch.ones(1, dtype=torch.int32)
    mismatched.kv_seq_lens_host_values = [1]
    mismatched.slot_mapping = torch.zeros(1, dtype=torch.int32)

    assert runner.can_execute(torch.zeros(8, dtype=torch.int32), aligned)
    assert runner.can_execute(torch.zeros(8, dtype=torch.int32), aligned, torch.zeros(8, 4))
    narrow = _aligned_dp_decode_metadata(8)
    narrow.paged_kv_indptr = None
    narrow.paged_kv_indices = None
    narrow.paged_kv_last_page_len = None
    narrow.block_table = torch.zeros((8, 1), dtype=torch.int32)
    narrow.kv_seq_lens = torch.full((8,), 5, dtype=torch.int32)
    narrow.kv_seq_lens_host_values = [5] * 8
    with pytest.raises(RuntimeError, match="cannot hold the KV length"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), narrow)
    fits = _aligned_dp_decode_metadata(8)
    fits.paged_kv_indptr = None
    fits.paged_kv_indices = None
    fits.paged_kv_last_page_len = None
    fits.block_table = torch.zeros((8, 1), dtype=torch.int32)
    fits.kv_seq_lens = torch.full((8,), 4, dtype=torch.int32)
    fits.kv_seq_lens_host_values = [4] * 8
    assert runner.can_execute(torch.zeros(8, dtype=torch.int32), fits)
    with pytest.raises(RuntimeError, match="input_embedding does not match"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), aligned, torch.zeros(3, 4))
    with pytest.raises(RuntimeError, match="does not match token rows"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), mismatched)

    bad_q_cu = _aligned_dp_decode_metadata(8)
    bad_q_cu.q_cu_seq_lens = torch.zeros(3, dtype=torch.int32)
    bad_indices = _aligned_dp_decode_metadata(8)
    bad_indices.linear_state_indices = torch.zeros(3, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="q_cu_seq_lens does not match"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), bad_q_cu)
    with pytest.raises(RuntimeError, match="linear_state_indices does not match"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), bad_indices)

    bad_initial_state = _aligned_dp_decode_metadata(8)
    bad_initial_state.has_initial_state = torch.ones(3, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="has_initial_state does not match"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), bad_initial_state)

    runner.layer_caches = [SimpleNamespace(conv=object(), kpool_tail=None)]
    missing_indices = _aligned_dp_decode_metadata(8)
    missing_indices.linear_state_indices = None
    missing_indices.raw_dp_execution_token_counts = (8, 0)
    with pytest.raises(RuntimeError, match="missing linear_state_indices"):
        runner.can_execute(torch.zeros(8, dtype=torch.int32), missing_indices)

    empty_runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
        num_decoding_tokens=8,
    )
    empty_runner.layer_caches = [SimpleNamespace(conv=object(), kpool_tail=None)]
    placeholder = _aligned_dp_decode_metadata(1)
    placeholder.dp_execution_token_counts = (8, 1)
    placeholder.raw_dp_execution_token_counts = (8, 0)
    placeholder.linear_state_indices = None
    assert empty_runner.can_execute(torch.zeros(1, dtype=torch.int32), placeholder)


def test_dp_kpool_spans_keep_complete_groups_only() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=6,
        )
        for rank in range(2)
    ]
    for runner in runners:
        runner.layer_caches = [SimpleNamespace(kpool_tail=torch.zeros((1, 2, 1, 1), dtype=torch.bfloat16))]
    active = _dp_metadata((6, 6))
    active.is_spec_verify = True
    active.kpool_query_lens = (6,)
    empty = _dp_metadata((6, 6))
    empty.is_spec_verify = True

    with patch.object(DecodeAclGraphRunner, "_has_compatible_decode_metadata", return_value=True):
        assert runners[0].can_execute(torch.zeros(6, dtype=torch.int32), active)
        assert runners[1].can_execute(torch.zeros(6, dtype=torch.int32), empty)
    assert runners[0]._graph_kpool_query_lens(active, 6, 8, 6) == (6,)
    assert runners[1]._graph_kpool_query_lens(empty, 6, 8, 6) == (6,)
    assert runners[0]._graph_kpool_query_lens(empty, 8, 16, 8) == (8, 8)


def test_empty_rank_kpool_spans_follow_shared_verify_width() -> None:
    runner = _runner()
    runner.layer_caches = [SimpleNamespace(kpool_tail=torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16))]
    metadata = SimpleNamespace()

    busy = runner._padded_kpool_query_lens(metadata, 8, 16, verify_width=8)
    empty = runner._padded_kpool_query_lens(metadata, 1, 16, verify_width=8)

    assert busy == (8, 8)
    assert empty == busy


def test_dp_expanded_verify_without_placeholder_folds_to_sequences() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=3,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=4,
        )
        for rank in range(2)
    ]
    metadata = _expanded_verify_metadata(8)
    metadata.dp_execution_token_counts = (8, 8)
    metadata.raw_dp_execution_token_counts = (8, 8)
    metadata.linear_state_indices = torch.tensor([0, 1], dtype=torch.int32)
    metadata.is_spec_verify = True

    with patch.object(
        DecodeAclGraphRunner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        for runner in runners:
            assert runner.can_execute(torch.zeros(8, dtype=torch.int32), metadata)


def _expanded_verify_metadata(rows: int) -> SimpleNamespace:
    metadata = _dp_metadata((rows, rows))
    metadata.slot_mapping = torch.arange(rows, dtype=torch.int32)
    metadata.linear_state_indices = torch.tensor([0], dtype=torch.int32)
    metadata.expanded_decode_metadata = SimpleNamespace(
        enabled=True,
        kv_seq_lens=torch.ones(rows, dtype=torch.int32),
        block_table=torch.zeros((rows, 1), dtype=torch.int32),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
    )
    return metadata


@pytest.mark.parametrize(
    ("width", "disable_verify_graph"),
    [
        pytest.param(4, False, id="kda-verify-default"),
        pytest.param(4, True, id="verify-graph-disabled"),
        pytest.param(3, False, id="width-three-pads-inside-bucket"),
    ],
)
def test_dp_mtp_target_fallback_is_group_wide(
    width: int,
    disable_verify_graph: bool,
) -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runners = [
        DecodeAclGraphRunner(
            nn.Identity(),
            attention_backend,
            torch.device("cpu"),
            max_batch=16,
            max_model_len=8,
            dp_size=2,
            dp_rank=rank,
            num_decoding_tokens=width,
        )
        for rank in range(2)
    ]
    metadata = _expanded_verify_metadata(width)
    metadata.is_spec_verify = True
    admitted = not disable_verify_graph

    with (
        patch.dict(
            os.environ,
            {"XLLM_NO_VERIFY_GRAPH": "1" if disable_verify_graph else "0"},
        ),
        patch.object(
            DecodeAclGraphRunner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
    ):
        for runner in runners:
            assert runner.can_execute(torch.zeros(width, dtype=torch.int32), metadata) is admitted


def test_kda_verify_graph_is_enabled_by_default() -> None:
    runner = _runner()
    metadata = _metadata(torch.tensor([1, 3], dtype=torch.int32))
    metadata.q_cu_seq_lens = torch.arange(5, dtype=torch.int32)
    metadata.kv_seq_lens = torch.tensor([5, 6, 9, 10], dtype=torch.int32)
    with (
        patch.dict(os.environ, {"XLLM_NO_VERIFY_GRAPH": "0"}),
        patch.object(runner, "_has_compatible_decode_metadata", return_value=True),
    ):
        assert runner.can_execute(torch.zeros(4, dtype=torch.int32), metadata)


def test_dp_mixed_step_does_not_enter_acl_decode_graph() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )

    with patch.object(
        runner,
        "_has_compatible_decode_metadata",
        return_value=True,
    ):
        assert not runner.can_execute(
            torch.zeros(3, dtype=torch.int32),
            _dp_metadata((3, 2), dp_is_decode=(0, 1)),
        )
        missing_step_type = _dp_metadata((3, 2))
        del missing_step_type.dp_is_decode
        with pytest.raises(RuntimeError, match="requires dp_is_decode"):
            runner.can_execute(torch.zeros(3, dtype=torch.int32), missing_step_type)
        assert not runner.can_execute(
            torch.zeros(1, 3, dtype=torch.int32),
            _dp_metadata((3, 2), dp_is_decode=(0, 1)),
        )
        with pytest.raises(RuntimeError, match="must be one-dimensional"):
            runner.can_execute(
                torch.zeros(1, 8, dtype=torch.int32),
                _dp_metadata((8, 8)),
            )


def test_dp_acl_graph_requires_group_wide_token_counts() -> None:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=16,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )
    metadata = _dp_metadata((3, 2))
    metadata.dp_execution_token_counts = (3,)

    with (
        patch.object(
            runner,
            "_has_compatible_decode_metadata",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="valid dp_execution_token_counts"),
    ):
        runner.can_execute(torch.zeros(3, dtype=torch.int32), metadata)


def test_xfia_graph_accepts_device_only_lengths() -> None:
    runner = _runner()
    runner.attention_backend.requires_host_kv_lengths = False
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.kv_seq_lens_host_values = None
    assert runner._has_compatible_decode_metadata(torch.arange(4), metadata)
    table, lengths, host_lengths, *_ = runner._decode_metadata(metadata)
    assert table is metadata.block_table
    assert lengths is metadata.kv_seq_lens
    assert host_lengths is None
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(kv_seq_lens_host_values=[1] * 4),
    )
    runner._fill_host_metadata(entry, None, batch_size=4)
    assert entry.static_metadata.kv_seq_lens_host_values == [1] * 4
