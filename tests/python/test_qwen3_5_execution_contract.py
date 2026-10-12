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

"""Execution-contract tests for the official Qwen3.5 model."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import LayerCache
from xllm.python.layers.npu.qwen3_5 import gated_delta_net
from xllm.python.layers.npu.qwen3_5.attention import NpuQwen3_5Attention
from xllm.python.layers.npu.qwen3_5.gated_delta_net import NpuQwen3_5GatedDeltaNet
from xllm.python.layers.npu.qwen3_5.gdn_metadata import (
    GdnDecodeMetadata,
    GdnMetadata,
    GdnPrefillMetadata,
    GdnSpecVerifyMetadata,
    GdnStateCache,
)
from xllm.python.layers.npu.qwen3_5.gdn_metadata_builder import (
    Qwen3_5GdnMetadataBuilder,
)
from xllm.python.layers.npu.qwen3_5.moe import NpuQwen3_5SparseMoEBlock
from xllm.python.model_executor.forward_context import (
    ForwardContext,
    forward_context,
)
from xllm.python.model_executor.input_batch import InputBatch


def _input_batch(
    token_counts: tuple[int, ...],
    *,
    token_capacity: int | None = None,
) -> InputBatch:
    num_tokens = sum(token_counts)
    query_start_loc = [0]
    for token_count in token_counts:
        query_start_loc.append(query_start_loc[-1] + token_count)
    batch = InputBatch.from_runtime(
        torch.zeros(num_tokens, dtype=torch.int32),
        torch.arange(num_tokens, dtype=torch.int32),
        SimpleNamespace(
            num_reqs=len(token_counts),
            num_tokens=num_tokens,
            num_scheduled_tokens=list(token_counts),
            num_computed_tokens=[0] * len(token_counts),
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=query_start_loc,
            is_prefilling=[True] * len(token_counts),
        ),
        is_dummy=False,
    )
    if token_capacity is None:
        return batch
    return batch.bind_graph_inputs(
        torch.zeros(token_capacity, dtype=torch.int32),
        torch.arange(token_capacity, dtype=torch.int32),
        torch.arange(token_capacity) >= num_tokens,
    )


def _gdn_cache(checkpoint_stride: int = 4) -> LayerCache:
    return LayerCache(
        key=None,
        value=None,
        conv=torch.zeros(
            8,
            checkpoint_stride + 2,
            16,
            dtype=torch.bfloat16,
        ),
        ssm=torch.zeros(
            8 * checkpoint_stride,
            2,
            4,
            4,
            dtype=torch.float32,
        ),
    )


def _builder() -> Qwen3_5GdnMetadataBuilder:
    builder = Qwen3_5GdnMetadataBuilder(
        layer_ids=(0,),
        num_value_heads=2,
        key_head_dim=4,
        value_head_dim=4,
        conv_dim=16,
    )
    builder.bind_layer_caches([_gdn_cache()])
    return builder


def _metadata(
    *,
    read_indices: torch.Tensor | None,
    write_indices: torch.Tensor,
    is_prefill: bool,
    is_chunked_prefill: bool = False,
    is_spec_verify: bool = False,
    has_initial_state: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    q_cu_seq_lens: torch.Tensor | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        linear_state_indices=write_indices,
        linear_state_read_indices=read_indices,
        linear_state_write_indices=write_indices,
        has_initial_state=has_initial_state,
        q_cu_seq_lens=q_cu_seq_lens,
        is_prefill=is_prefill,
        is_chunked_prefill=is_chunked_prefill,
        is_spec_verify=is_spec_verify,
        num_accepted_tokens=num_accepted_tokens,
    )


def test_gdn_prefill_builder_materializes_operator_ready_metadata() -> None:
    metadata = _metadata(
        read_indices=torch.tensor([5, 7], dtype=torch.int64),
        write_indices=torch.tensor([9, 11], dtype=torch.int64),
        is_prefill=True,
        has_initial_state=torch.tensor([True, False]),
    )

    gdn_metadata = _builder().build(_input_batch((129, 1)), metadata)

    assert isinstance(gdn_metadata, GdnPrefillMetadata)
    torch.testing.assert_close(
        gdn_metadata.conv_read_indices,
        torch.tensor([5, -1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        gdn_metadata.conv_write_indices,
        torch.tensor([9, 11], dtype=torch.int32),
    )
    torch.testing.assert_close(
        gdn_metadata.ssm_read_indices,
        torch.tensor([20, -1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        gdn_metadata.ssm_write_indices,
        torch.tensor([36, 44], dtype=torch.int32),
    )
    torch.testing.assert_close(
        gdn_metadata.cu_seqlens,
        torch.tensor([0, 129, 130], dtype=torch.int32),
    )
    assert gdn_metadata.num_matrices == 6
    assert gdn_metadata.state_caches[0].conv_state is not None


def test_gdn_prefill_builder_reuses_materialized_cu_seqlens() -> None:
    cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)
    metadata = _metadata(
        read_indices=torch.tensor([1, 2], dtype=torch.int32),
        write_indices=torch.tensor([1, 2], dtype=torch.int32),
        is_prefill=True,
        has_initial_state=torch.tensor([True, True]),
        q_cu_seq_lens=cu_seqlens,
    )

    gdn_metadata = _builder().build(_input_batch((2, 1)), metadata)

    assert isinstance(gdn_metadata, GdnPrefillMetadata)
    assert gdn_metadata.cu_seqlens is cu_seqlens


@pytest.mark.parametrize(
    ("config_override", "error"),
    (
        ({"linear_conv_kernel_dim": 3}, "convolution width 4"),
        ({"linear_key_head_dim": 64}, "head dimension 128"),
        ({"dtype": "float16"}, "BF16 model weights"),
    ),
)
def test_gdn_builder_rejects_unsupported_kernel_config(
    config_override: dict[str, object],
    error: str,
) -> None:
    config: dict[str, object] = {
        "device": "npu",
        "n_layers": 1,
        "layer_types": ["linear_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "dtype": "bfloat16",
    }
    config.update(config_override)

    with pytest.raises(NotImplementedError, match=error):
        Qwen3_5GdnMetadataBuilder.from_config(config)


def test_gdn_builder_is_registered_only_for_npu_linear_attention() -> None:
    base_config: dict[str, object] = {
        "n_layers": 1,
        "layer_types": ["linear_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "dtype": "bfloat16",
    }

    assert Qwen3_5GdnMetadataBuilder.from_config({**base_config, "device": "cuda"}) is None
    assert (
        Qwen3_5GdnMetadataBuilder.from_config(
            {
                **base_config,
                "device": "npu",
                "layer_types": ["full_attention"],
                "cp_size": 2,
            }
        )
        is None
    )
    assert isinstance(
        Qwen3_5GdnMetadataBuilder.from_config({**base_config, "device": "npu"}),
        Qwen3_5GdnMetadataBuilder,
    )


def test_gdn_builder_derives_empty_cpp_layer_types_from_interval() -> None:
    builder = Qwen3_5GdnMetadataBuilder.from_config(
        {
            "device": "npu",
            "n_layers": 4,
            "layer_types": [],
            "full_attention_interval": 4,
            "dtype": "bfloat16",
        }
    )

    assert isinstance(builder, Qwen3_5GdnMetadataBuilder)
    assert builder._layer_ids == (0, 1, 2)


def test_gdn_prefill_builder_materializes_empty_dp_execution_row() -> None:
    input_batch = InputBatch.from_runtime(
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        SimpleNamespace(
            num_reqs=0,
            num_tokens=1,
            num_scheduled_tokens=[],
            num_computed_tokens=[],
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=[0],
            is_prefilling=[],
        ),
        is_dummy=True,
    )
    metadata = _metadata(
        read_indices=torch.tensor([0], dtype=torch.int32),
        write_indices=torch.tensor([0], dtype=torch.int32),
        is_prefill=True,
        has_initial_state=torch.tensor([False]),
    )

    gdn_metadata = _builder().build(input_batch, metadata)

    assert isinstance(gdn_metadata, GdnPrefillMetadata)
    torch.testing.assert_close(
        gdn_metadata.conv_read_indices,
        torch.tensor([-1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        gdn_metadata.conv_write_indices,
        torch.tensor([0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        gdn_metadata.cu_seqlens,
        torch.tensor([0, 1], dtype=torch.int32),
    )


def test_gdn_graph_builder_updates_stable_decode_buffers() -> None:
    builder = _builder()
    input_batch = _input_batch((1, 1), token_capacity=4)
    initial_metadata = _metadata(
        read_indices=None,
        write_indices=torch.tensor([1, 2], dtype=torch.int32),
        is_prefill=False,
    )
    gdn_metadata = builder.allocate_persistent(input_batch, initial_metadata)
    assert isinstance(gdn_metadata, GdnDecodeMetadata)
    read_pointer = gdn_metadata.read_state_indices.data_ptr()
    write_pointer = gdn_metadata.write_state_indices.data_ptr()

    builder.update_persistent(
        gdn_metadata,
        input_batch,
        _metadata(
            read_indices=None,
            write_indices=torch.tensor([4, 6], dtype=torch.int64),
            is_prefill=False,
        ),
    )

    assert gdn_metadata.read_state_indices.data_ptr() == read_pointer
    assert gdn_metadata.write_state_indices.data_ptr() == write_pointer
    assert gdn_metadata.read_state_indices.tolist() == [4, 6, 0, 0]
    assert gdn_metadata.write_state_indices.tolist() == [4, 6, 0, 0]


def test_gdn_eager_decode_preserves_scheduler_slots() -> None:
    indices = torch.tensor([3, 7], dtype=torch.int64)

    gdn_metadata = _builder().build(
        _input_batch((1, 1)),
        _metadata(
            read_indices=None,
            write_indices=indices,
            is_prefill=False,
        ),
    )

    assert isinstance(gdn_metadata, GdnDecodeMetadata)
    torch.testing.assert_close(
        gdn_metadata.read_state_indices,
        indices.to(torch.int32),
    )
    torch.testing.assert_close(
        gdn_metadata.write_state_indices,
        indices.to(torch.int32),
    )


def test_gdn_spec_verify_takes_precedence_over_chunked_prefill() -> None:
    write_indices = torch.tensor([5, 9], dtype=torch.int64)
    num_accepted_tokens = torch.tensor([4, 2], dtype=torch.int64)

    gdn_metadata = _builder().build(
        _input_batch((4, 4)),
        _metadata(
            read_indices=None,
            write_indices=write_indices,
            is_prefill=False,
            is_chunked_prefill=True,
            is_spec_verify=True,
            num_accepted_tokens=num_accepted_tokens,
        ),
    )

    assert isinstance(gdn_metadata, GdnSpecVerifyMetadata)
    torch.testing.assert_close(
        gdn_metadata.read_state_indices,
        write_indices.to(torch.int32),
    )
    torch.testing.assert_close(
        gdn_metadata.write_state_indices,
        write_indices.to(torch.int32),
    )
    torch.testing.assert_close(
        gdn_metadata.num_accepted_tokens,
        num_accepted_tokens.to(torch.int32),
    )


def test_gdn_prefill_layer_consumes_builder_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = NpuQwen3_5GatedDeltaNet.__new__(NpuQwen3_5GatedDeltaNet)
    nn.Module.__init__(layer)
    layer.layer_id = 0
    layer.value_dim = 4
    layer.out_proj = nn.Identity()
    layer.conv1d_weight = nn.Parameter(torch.zeros(4, 16, dtype=torch.bfloat16))
    layer.A_log = nn.Parameter(torch.zeros(2, dtype=torch.float32))
    layer.dt_bias = nn.Parameter(torch.zeros(2, dtype=torch.float32))
    layer.norm_weight = nn.Parameter(torch.zeros(2, 4, dtype=torch.bfloat16))

    hidden = torch.zeros(3, 4, dtype=torch.bfloat16)
    mixed_qkv = torch.zeros(3, 16, dtype=torch.bfloat16)
    a = torch.zeros(3, 2, dtype=torch.bfloat16)
    b = torch.zeros(3, 2, dtype=torch.bfloat16)
    z = torch.zeros(3, 2, 4, dtype=torch.bfloat16)
    kernel_output = torch.zeros(3, 4, dtype=torch.bfloat16)
    kernel = MagicMock(return_value=kernel_output)

    monkeypatch.setattr(
        layer,
        "_project_prefill_inputs",
        MagicMock(return_value=(mixed_qkv, a, b, z)),
    )
    monkeypatch.setattr(
        gated_delta_net.kernels,
        "mega_gdn_prefill",
        kernel,
        raising=False,
    )

    conv_state = torch.zeros(4, 3, 16, dtype=torch.bfloat16)
    ssm_state = torch.zeros(4, 2, 4, 4, dtype=torch.float32)
    conv_read = torch.tensor([1], dtype=torch.int32)
    conv_write = torch.tensor([2], dtype=torch.int32)
    ssm_read = torch.tensor([4], dtype=torch.int32)
    ssm_write = torch.tensor([8], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    context = ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=None,
        layer_caches=[],
        execution_contexts={
            GdnMetadata: GdnPrefillMetadata(
                state_caches={0: GdnStateCache(conv_state, ssm_state)},
                conv_read_indices=conv_read,
                conv_write_indices=conv_write,
                ssm_read_indices=ssm_read,
                ssm_write_indices=ssm_write,
                cu_seqlens=cu_seqlens,
                num_matrices=6,
            )
        },
    )

    with forward_context(context):
        output = layer(hidden)

    args = kernel.call_args.args
    assert args[5] is conv_state
    assert args[8] is conv_read
    assert args[9] is conv_write
    assert args[10] is ssm_read
    assert args[11] is ssm_write
    assert args[12] is ssm_state
    assert args[13] is cu_seqlens
    assert args[15] == 6
    torch.testing.assert_close(output, kernel_output)


def test_gdn_decode_layer_consumes_only_typed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = NpuQwen3_5GatedDeltaNet.__new__(NpuQwen3_5GatedDeltaNet)
    nn.Module.__init__(layer)
    layer.layer_id = 0
    layer.value_dim = 4
    layer.out_proj = nn.Identity()
    read_indices = torch.tensor([2, 3], dtype=torch.int32)
    write_indices = torch.tensor([5, 7], dtype=torch.int32)
    conv_state = torch.zeros(4, 3, 16, dtype=torch.bfloat16)
    ssm_state = torch.zeros(4, 1, 4, 4, dtype=torch.float32)
    decode_call: dict[str, torch.Tensor] = {}

    def _decode(
        _self: NpuQwen3_5GatedDeltaNet,
        hidden: torch.Tensor,
        actual_conv_state: torch.Tensor,
        actual_ssm_state: torch.Tensor,
        actual_read_indices: torch.Tensor,
        actual_write_indices: torch.Tensor,
    ) -> torch.Tensor:
        decode_call["conv_state"] = actual_conv_state
        decode_call["ssm_state"] = actual_ssm_state
        decode_call["read"] = actual_read_indices
        decode_call["write"] = actual_write_indices
        return hidden

    monkeypatch.setattr(
        NpuQwen3_5GatedDeltaNet,
        "_decode",
        _decode,
    )
    context = ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=None,
        layer_caches=[],
        execution_contexts={
            GdnMetadata: GdnDecodeMetadata(
                state_caches={0: GdnStateCache(conv_state, ssm_state)},
                read_state_indices=read_indices,
                write_state_indices=write_indices,
            )
        },
    )
    hidden = torch.randn(2, 4)

    with forward_context(context):
        output = layer(hidden)

    assert decode_call["conv_state"] is conv_state
    assert decode_call["ssm_state"] is ssm_state
    assert decode_call["read"] is read_indices
    assert decode_call["write"] is write_indices
    torch.testing.assert_close(output, hidden)


def test_npu_decoder_reuses_existing_attention_and_mega_moe() -> None:
    from xllm.python.layers.npu.qwen3_5.decoder_layer import NpuQwen3_5DecoderLayer

    assert NpuQwen3_5DecoderLayer.attention_cls is NpuQwen3_5Attention
    assert NpuQwen3_5DecoderLayer.sparse_moe_cls is NpuQwen3_5SparseMoEBlock
    assert NpuQwen3_5DecoderLayer.gated_delta_net_cls is NpuQwen3_5GatedDeltaNet


def test_qwen35_model_source_has_no_execution_mode_dependencies() -> None:
    from xllm.python.models import qwen3_5

    source = inspect.getsource(qwen3_5) + inspect.getsource(gated_delta_net)
    for forbidden_name in (
        "AttentionMetadata",
        "InputBatch",
        "get_execution_buffer",
        "get_forward_context",
        "in_acl_graph",
        "is_dummy",
        "is_padding",
    ):
        assert forbidden_name not in source
