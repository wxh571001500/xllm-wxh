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

"""Qwen3.5 DSpark target ACL graph integration test, invoked by the C++ driver."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.expanded_decode_metadata import ExpandedDecodeMetadata
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.models.qwen3_5 import Qwen3_5ForCausalLM

_DEVICE = torch.device("npu:0")


def _random(shape: tuple[int, ...], generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    return (torch.randn(shape, generator=generator) * 0.1).to(device=_DEVICE, dtype=dtype)


def _target_metadata(
    slots: list[int],
    accepted: list[int],
    width: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, SimpleNamespace, SimpleNamespace]:
    batch = len(slots)
    tokens = batch * width
    prefix = 2 + step * width
    logical_slots = torch.tensor(slots, dtype=torch.int32, device=_DEVICE)
    lengths = [prefix + offset + 1 for _ in slots for offset in range(width)]
    block_table = logical_slots[:, None]
    row_blocks = block_table.repeat_interleave(width, dim=0)
    slot_values = [slot * 128 + prefix + offset for slot in slots for offset in range(width)]
    positions = torch.tensor(list(range(prefix, prefix + width)) * batch, dtype=torch.int32, device=_DEVICE)
    input_ids = torch.tensor(
        [(slot + step + offset) % 128 for slot in slots for offset in range(width)], dtype=torch.int32, device=_DEVICE
    )
    expanded = ExpandedDecodeMetadata(
        kv_seq_lens=torch.tensor(lengths, dtype=torch.int32, device=_DEVICE),
        block_table=row_blocks,
        paged_kv_indptr=torch.arange(tokens + 1, dtype=torch.int32, device=_DEVICE),
        paged_kv_indices=row_blocks.flatten(),
        paged_kv_last_page_len=torch.tensor(lengths, dtype=torch.int32, device=_DEVICE),
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=lengths,
    )
    boundaries = list(range(0, tokens + 1, width))
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor(slot_values, dtype=torch.int32, device=_DEVICE),
        paged_kv_indptr=torch.arange(batch + 1, dtype=torch.int32, device=_DEVICE),
        paged_kv_indices=logical_slots,
        paged_kv_last_page_len=torch.full((batch,), prefix + width, dtype=torch.int32, device=_DEVICE),
        paged_kv_indptr_host=None,
        paged_kv_last_page_len_host=None,
        block_table=block_table,
        kv_seq_lens=torch.full((batch,), prefix + width, dtype=torch.int32, device=_DEVICE),
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=[prefix + width] * batch,
        kv_cu_seq_lens=torch.arange(batch + 1, dtype=torch.int32, device=_DEVICE) * (prefix + width),
        qo_indptr=None,
        q_cu_seq_lens=torch.tensor(boundaries, dtype=torch.int32, device=_DEVICE),
        q_cu_seq_lens_host_values=boundaries,
        q_seq_lens=torch.full((batch,), width, dtype=torch.int32, device=_DEVICE),
        q_seq_lens_host=torch.full((batch,), width, dtype=torch.int32),
        linear_state_indices=logical_slots,
        linear_state_read_indices=logical_slots,
        linear_state_write_indices=logical_slots,
        has_initial_state=torch.ones(batch, dtype=torch.bool, device=_DEVICE),
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32, device=_DEVICE),
        kpool_query_lens=(),
        expanded_decode_metadata=expanded,
        multi_block_tables=(),
        new_cache_slots_host_values=slot_values,
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=True,
        is_mixed=False,
        is_dummy=False,
        dp_execution_token_counts=(),
        raw_dp_execution_token_counts=(),
        dp_global_kv_max_seq_lens=(),
        dp_is_decode=(),
        max_query_len=width,
        max_seq_len=prefix + width,
        dsa_metadata=None,
        dsa_positions=None,
        local_slot_mapping=None,
        kv_split_size=1,
        kv_split_rank=0,
        has_kv_shard=False,
    )
    batch_metadata = SimpleNamespace(
        num_reqs=batch,
        num_tokens=tokens,
        num_scheduled_tokens=[width] * batch,
        num_computed_tokens=[prefix] * batch,
        query_start_loc=boundaries,
        is_prefilling=[False] * batch,
    )
    return input_ids, positions, metadata, batch_metadata


@torch.inference_mode()
def check_dspark_target_aclgraph() -> None:
    """Verify graph replay matches eager with complete groups and bucket tails."""
    for width in range(2, 18):
        _check_dspark_target_aclgraph_width(width)


def _check_dspark_target_aclgraph_width(width: int) -> None:
    generator = torch.Generator().manual_seed(238)
    config = {
        "model_type": "qwen3_5",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "intermediate_size": 256,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "vocab_size": 128,
        "max_position_embeddings": 128,
        "layers_to_capture": [0, 1],
        "device": "npu:0",
        "dtype": "bfloat16",
    }
    model = Qwen3_5ForCausalLM(config)
    for name, parameter in model.named_parameters():
        if name.endswith("norm_weight"):
            parameter.fill_(1)
        else:
            parameter.copy_(_random(tuple(parameter.shape), generator, parameter.dtype))
    conv = _random((10, width + 2, 512), generator, torch.bfloat16)
    ssm = _random((10 * width, 2, 128, 128), generator, torch.float32)
    conv[0].fill_(float("nan"))
    ssm[:width].fill_(float("nan"))
    key = _random((10, 128, 1, 128), generator, torch.bfloat16)
    value = _random((10, 128, 1, 128), generator, torch.bfloat16)
    caches = [LayerCache(None, None, conv=conv, ssm=ssm), LayerCache(key, value)]
    eager_caches = [
        LayerCache(None, None, conv=conv.clone(), ssm=ssm.clone()),
        LayerCache(key.clone(), value.clone()),
    ]
    graph = ModelExecutor(model, {**config, "python_graph_backend": "aclgraph"}, 32, width)
    eager = ModelExecutor(model, {**config, "python_graph_backend": "off"}, 32, width)
    graph.bind_kv_caches(caches)
    eager.bind_kv_caches(eager_caches)
    assert graph.decode_graph_runner is not None
    captured_outputs = {}
    steps = (
        ([1], [width]),
        ([1, 2, 3, 4, 5], [1, 2, width, 1, 2]),
        ([2, 3, 4, 5, 6, 7, 8], [2, width, 1, 2, width, 1, 2]),
        ([5, 4, 3, 2, 1], [width, 1, 2, width, 1]),
        ([1], [width]),
    )
    for step, (slots, accepted) in enumerate(steps):
        ids, positions, metadata, batch = _target_metadata(slots, accepted, width, step)
        assert graph.decode_graph_runner.can_execute(ids, metadata)
        expected = eager.execute(ids, positions, metadata, input_batch_metadata=batch)
        actual = graph.execute(ids, positions, metadata, input_batch_metadata=batch)
        assert isinstance(actual, tuple) and isinstance(expected, tuple)
        assert actual[0].shape == (len(slots) * width, 128)
        assert actual[1].shape == (len(slots) * width, 256)
        # Repeated batch sizes must reuse captured storage after B grows/shrinks.
        first_outputs = captured_outputs.setdefault(len(slots), actual)
        assert tuple(t.data_ptr() for t in actual) == tuple(t.data_ptr() for t in first_outputs)
        for output, reference in zip(actual, expected):
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output.cpu(), reference.cpu(), rtol=5e-3, atol=2e-2)
        # Padding owns slot zero; compare all real state rows/checkpoints.
        torch.testing.assert_close(conv[1:].cpu(), eager_caches[0].conv[1:].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(ssm[width:].cpu(), eager_caches[0].ssm[width:].cpu(), rtol=5e-3, atol=2.5e-5)
        # Different padded projection shapes can round BF16 KV by one ULP.
        torch.testing.assert_close(key[1:].cpu(), eager_caches[1].key[1:].cpu(), rtol=1e-2, atol=1e-4)
        torch.testing.assert_close(value[1:].cpu(), eager_caches[1].value[1:].cpu(), rtol=1e-2, atol=1e-4)
