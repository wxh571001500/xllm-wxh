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

"""Empty-DP coverage for the NPU ACL decode-graph runner."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.dsa_metadata import build_cache_specs
from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
)


def _decode_metadata(is_dummy: bool, accepted: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        slot_mapping=torch.tensor([-1 if is_dummy else 4], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0 if is_dummy else 1], dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(1, dtype=torch.int32),
        block_table=torch.tensor([[0 if is_dummy else 1]], dtype=torch.int32),
        kv_seq_lens=torch.ones(1, dtype=torch.int32),
        kv_seq_lens_host_values=[1],
        kv_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        linear_state_indices=torch.tensor([0 if is_dummy else 3], dtype=torch.int32),
        num_accepted_tokens=None if accepted is None else torch.tensor([accepted], dtype=torch.int32),
        expanded_decode_metadata=None,
        multi_block_tables=(),
        new_cache_slots_host_values=[] if is_dummy else [4],
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=is_dummy,
    )


def _linear_runner(num_decoding_tokens: int = 4, linear: bool = True) -> DecodeAclGraphRunner:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        dp_size=2,
        num_decoding_tokens=num_decoding_tokens,
    )
    if linear:
        runner.bind_layer_caches([LayerCache(None, None, conv=torch.zeros(8, 1, 3), ssm=torch.zeros(8, 1, 1, 1))])
    return runner


@pytest.mark.parametrize("starts_empty", [True, False])
def test_accepted_buffer_survives_empty_and_busy_dp_transitions(starts_empty: bool) -> None:
    runner = _linear_runner()
    input_ids = torch.ones(1, dtype=torch.int32)
    positions = torch.zeros_like(input_ids)
    initial = _decode_metadata(starts_empty, None if starts_empty else 4)
    entry = runner._allocate_entry(4, input_ids, positions, initial)
    counts = entry.static_metadata.num_accepted_tokens
    assert counts is not None
    address = counts.data_ptr()
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        for metadata in (
            initial,
            _decode_metadata(True),
            _decode_metadata(False, 3),
            _decode_metadata(True, 4),
            _decode_metadata(False, 2),
        ):
            runner._fill_entry(entry, input_ids, positions, metadata, batch_size=1, input_embedding=None)
            expected = 1 if metadata.is_dummy else int(metadata.num_accepted_tokens[0])
            assert counts.tolist() == [expected, 1, 1, 1]
            assert counts.data_ptr() == address
            assert entry.static_metadata.linear_state_indices.tolist() == [0 if metadata.is_dummy else 3, 0, 0, 0]


def test_speculative_linear_graph_rejects_missing_acceptance_on_first_real_batch() -> None:
    runner = _linear_runner()
    input_ids = torch.ones(1, dtype=torch.int32)
    metadata = _decode_metadata(False)
    entry = runner._allocate_entry(4, input_ids, input_ids, metadata)
    with (
        patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True),
        pytest.raises(RuntimeError, match="accepted-token metadata is missing"),
    ):
        runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=1, input_embedding=None)


@pytest.mark.parametrize("num_decoding_tokens,linear", [(1, True), (4, False)])
def test_empty_non_recurrent_or_non_speculative_graph_does_not_require_acceptance(
    num_decoding_tokens: int, linear: bool
) -> None:
    runner = _linear_runner(num_decoding_tokens, linear)
    input_ids = torch.ones(1, dtype=torch.int32)
    metadata = _decode_metadata(True)
    entry = runner._allocate_entry(4, input_ids, input_ids, metadata)
    assert entry.static_metadata.num_accepted_tokens is None
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        for metadata in (_decode_metadata(True), _decode_metadata(False)):
            runner._fill_entry(entry, input_ids, input_ids, metadata, batch_size=1, input_embedding=None)


def test_dsa_graph_empty_rank_uses_reserved_block_zero() -> None:
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
    static_metadata = SimpleNamespace(
        multi_block_tables=(
            torch.full((4, 4), 99, dtype=torch.int32),
            torch.full((4, 2), 99, dtype=torch.int32),
            torch.full((4, 1), 99, dtype=torch.int32),
        )
    )
    metadata = SimpleNamespace(multi_block_tables=(), is_dummy=True)

    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        torch.zeros((1, 1), dtype=torch.int32),
        batch_size=1,
    )

    for table in static_metadata.multi_block_tables:
        assert torch.all(table == 0)


def _dsa_metadata(tables: tuple = (), *, dummy: bool = True) -> SimpleNamespace:
    """One-row DSA decode batch as the C++ multi-block exporter emits it.

    Busy shards carry ``multi_block_tables`` only (flat ``block_table``
    undefined, ``slot_mapping`` empty); the empty DP shard additionally has no
    manager tables at all (its fake input never reaches the exporter).
    """
    return SimpleNamespace(
        slot_mapping=torch.empty(0, dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0], dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(1, dtype=torch.int32),
        block_table=None,
        kv_seq_lens=torch.ones(1, dtype=torch.int32),
        kv_seq_lens_host_values=[1],
        kv_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        linear_state_indices=None,
        expanded_decode_metadata=None,
        multi_block_tables=tables,
        new_cache_slots_host_values=[] if dummy else [301],
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=dummy,
        dp_execution_token_counts=(1, 1),
        dp_is_decode=(1, 1),
        dp_global_kv_max_seq_lens=(1, 1),
    )


def _dsa_runner(**backend_overrides) -> DecodeAclGraphRunner:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    backend = SimpleNamespace(page_size=128, is_mla=False, group_infos=group_infos, **backend_overrides)
    return DecodeAclGraphRunner(
        nn.Identity(),
        backend,
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
        dp_size=2,
        dp_rank=1,
    )


def test_empty_dp_shard_without_manager_tables_replays_the_graph() -> None:
    """A table-less empty DP shard must not split the group onto eager.

    The empty shard's fake input never reaches the composite block exporter,
    so left as-is it fails the flat-table admission check and takes the eager
    runner while the busy ranks capture/replay the graph -- the asymmetric
    collective schedule that hangs DP>1 (NPUGraph.cpp:223). Normalization
    synthesizes one-row reserved-block-0 manager tables so the shard joins the
    same graph step.
    """
    runner = _dsa_runner()
    input_ids = torch.tensor([1], dtype=torch.int32)
    metadata = _dsa_metadata()

    view = runner._normalize_dsa_metadata(metadata)
    assert view is not metadata
    assert tuple(view.block_table.shape) == (1, 1)
    assert view.slot_mapping.tolist() == [0]
    assert [tuple(table.shape) for table in view.multi_block_tables] == [(1, 1)] * 3
    # Read-through: untouched fields still come from the original metadata.
    assert view.kv_seq_lens_host_values == [1]
    assert view.is_dummy is True

    assert runner.can_execute(input_ids, metadata)


def test_empty_dp_shard_normalized_entry_fills_and_keeps_table_zero() -> None:
    """The synthesized empty-shard contract survives a real graph fill."""
    runner = _dsa_runner()
    input_ids = torch.tensor([1], dtype=torch.int32)
    positions = torch.zeros_like(input_ids)
    metadata = runner._normalize_dsa_metadata(_dsa_metadata())

    entry = runner._allocate_entry(4, input_ids, positions, metadata)
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=1, input_embedding=None)

    for table in entry.static_metadata.multi_block_tables:
        assert torch.all(table == 0)


def test_mismatched_manager_table_count_falls_back_to_eager() -> None:
    runner = _dsa_runner()
    input_ids = torch.tensor([1], dtype=torch.int32)
    metadata = _dsa_metadata((torch.zeros((1, 1), dtype=torch.int32),))

    # Normalization only synthesizes when the export is absent; a partial
    # manager export must stay on the eager runner rather than raise inside
    # the captured fill.
    assert runner._normalize_dsa_metadata(metadata) is metadata
    assert not runner.can_execute(input_ids, metadata)


def test_token_capacity_bucket_is_group_wide_and_bounded_by_max_model_len() -> None:
    runner = _dsa_runner(graph_token_capacity_granularity=128)
    input_ids = torch.tensor([1], dtype=torch.int32)

    in_range = _dsa_metadata()
    in_range.dp_global_kv_max_seq_lens = (1, 200)
    assert runner._token_capacity_bucket(in_range) == 2
    assert runner.can_execute(input_ids, in_range)

    beyond_model_len = _dsa_metadata()
    beyond_model_len.dp_global_kv_max_seq_lens = (1, 100000)
    assert runner._token_capacity_bucket(beyond_model_len) > (512 + 127) // 128
    assert not runner.can_execute(input_ids, beyond_model_len)


def test_graph_key_partitions_by_token_capacity_bucket() -> None:
    same = DecodeAclGraphRunner._graph_key(1, False, None, (), 1, 1)
    assert same == DecodeAclGraphRunner._graph_key(1, False, None, (), 1, 1)
    assert same != DecodeAclGraphRunner._graph_key(1, False, None, (), 1, 2)
    # Backends that do not bucket keep the legacy zero bucket.
    assert DecodeAclGraphRunner._graph_key(1, False, None, (), 1)[-1] == 0


def test_busy_dsa_rank_admits_without_flat_scheduler_slots() -> None:
    """A composite-block DSA batch must not be rejected for missing flat slots.

    The C++ builder appends nothing to ``new_token_slot_ids`` for sequences
    with composite blocks, so ``slot_mapping`` and
    ``new_cache_slots_host_values`` are empty on every real decode batch while
    the DSA builder derives the slots from the persistent manager tables.
    Requiring a flat slot source here sent busy DP shards to eager while the
    empty shards (whose C++ metadata is fully synthesized) stayed on the graph
    -- the asymmetric collective schedule that hangs DP>1.
    """
    runner = _dsa_runner()
    input_ids = torch.tensor([1], dtype=torch.int32)
    manager_table = torch.tensor([[3]], dtype=torch.int32)
    metadata = _dsa_metadata((manager_table,) * 3, dummy=False)
    metadata.new_cache_slots_host_values = []
    metadata.slot_mapping = torch.empty(0, dtype=torch.int32)

    assert runner._has_effective_slot_mapping(metadata, 1)
    # The DSA flat slots are a zero-reader field; keep the static buffer shaped.
    assert runner._effective_slot_mapping(metadata, 1, torch.device("cpu")).tolist() == [0]
    assert runner.can_execute(input_ids, metadata)

    positions = torch.zeros_like(input_ids)
    entry = runner._allocate_entry(4, input_ids, positions, metadata)
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=1, input_embedding=None)
    # Empty host slots let the DSA builder recompute the slot from the table.
    assert entry.static_metadata.new_cache_slots_host_values == []
    # The live row keeps block 3; spare columns and padded rows stay at the
    # reserved block 0 (gatherable) instead of the old -1 pads.
    zero_row = [0] * 4
    assert entry.static_metadata.multi_block_tables[0].tolist() == [[3, 0, 0, 0], zero_row, zero_row, zero_row]
    assert entry.static_metadata.multi_block_tables[1].tolist() == [[3], [0], [0], [0]]


def test_non_dsa_rank_still_requires_a_flat_slot_source() -> None:
    """Without DSA manager tables the flat slot contract stays mandatory."""
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
    )
    metadata = _dsa_metadata()
    metadata.multi_block_tables = ()
    metadata.new_cache_slots_host_values = []

    assert not runner._has_effective_slot_mapping(metadata, 1)
    with pytest.raises(RuntimeError, match="one scheduler cache slot per token"):
        runner._effective_slot_mapping(metadata, 1, torch.device("cpu"))


def test_non_capturable_moe_quant_mode_is_never_admitted() -> None:
    """The fp8/none torch MoE must keep every rank on the eager runner.

    The capability marker lives on the attention backend and is derived from
    the shared model config, so the admission check is static and each rank of
    a DP group returns the same verdict.
    """
    backend = SimpleNamespace(page_size=128, is_mla=False)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=512,
    )
    input_ids = torch.tensor([1], dtype=torch.int32)
    metadata = _decode_metadata(False)

    # No marker (non-V4.1 backends): the legacy admission is unchanged.
    assert runner.can_execute(input_ids, metadata)
    backend.moe_graph_capturable = True
    assert runner.can_execute(input_ids, metadata)
    # Marker present and False: never admit, regardless of the batch shape.
    backend.moe_graph_capturable = False
    assert not runner.can_execute(input_ids, metadata)
