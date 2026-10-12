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

"""A3 XFIA decode: numerical equivalence and replay with changing device metadata.

Set XLLM_XFIA_TEST_LIBRARY to an operator-only library registering
xllm_ops::x_flash_attention_decode_out, and XLLM_XFIA_TEST_DEVICE=npu:8.
The service represents the eight proposal tokens by eight independent rows;
this deliberately tests shared paged caches with distinct per-row KV lengths.
The last INT32 tensor supplies the logical start of each row's KV window.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


@pytest.fixture(scope="module")
def device() -> torch.device:
    torch.set_num_threads(2)
    dev = torch.device(os.environ.get("XLLM_XFIA_TEST_DEVICE", "npu:8"))
    torch.npu.set_device(dev)
    library = os.environ.get("XLLM_XFIA_TEST_LIBRARY")
    if library:
        torch.ops.load_library(library)
    assert hasattr(torch.ops.xllm_ops, "x_flash_attention_decode_out")
    return dev


def _inputs(
    device: torch.device, lengths: list[int], dtype: torch.dtype, heads: int = 4, kv_heads: int = 1
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(137)
    rows = len(lengths)
    pages = (max(lengths) + 127) // 128
    # All proposal rows share their context cache, just as expanded decode does.
    table = torch.randperm(pages, dtype=torch.int32).expand(rows, -1).contiguous()
    query = torch.randn(rows, heads, 128, dtype=dtype)
    key = torch.randn(pages, 128, kv_heads, 128, dtype=dtype)
    value = torch.randn_like(key)
    return tuple(
        t.to(device)
        for t in (
            query,
            key,
            value,
            table,
            torch.arange(1, rows + 1, dtype=torch.int32),
            torch.tensor(lengths, dtype=torch.int32),
            torch.zeros(rows, dtype=torch.int32),
        )
    )


def _reference(inputs: tuple[torch.Tensor, ...], lengths: list[int], scale: float) -> torch.Tensor:
    q, k, v, table, _, _, starts = (t.cpu() for t in inputs)
    outputs = []
    for row, length in enumerate(lengths):
        ids = table[row, : (length + 127) // 128].long()
        key = k[ids].flatten(0, 1)[int(starts[row]) : length].float().repeat_interleave(q.shape[1] // k.shape[2], dim=1)
        value = (
            v[ids].flatten(0, 1)[int(starts[row]) : length].float().repeat_interleave(q.shape[1] // k.shape[2], dim=1)
        )
        scores = torch.einsum("hd,thd->ht", q[row].float(), key) * scale
        outputs.append(torch.einsum("ht,thd->hd", scores.softmax(-1), value))
    return torch.stack(outputs)


def _fia(inputs: tuple[torch.Tensor, ...], lengths: list[int], scale: float) -> torch.Tensor:
    q, k, v, table, *_ = inputs
    return torch.ops.npu.npu_fused_infer_attention_score_v2(
        q,
        k.flatten(2),
        v.flatten(2),
        block_table=table,
        actual_seq_qlen=list(range(1, len(lengths) + 1)),
        actual_seq_kvlen=lengths,
        num_query_heads=q.shape[1],
        num_key_value_heads=k.shape[2],
        softmax_scale=scale,
        input_layout="TND",
        sparse_mode=0,
        block_size=128,
        return_softmax_lse=False,
    )[0]


def _assert_accuracy(actual: torch.Tensor, inputs: tuple[torch.Tensor, ...], lengths: list[int], scale: float) -> None:
    actual = actual.cpu().float()
    expected = _reference(inputs, lengths, scale)
    fia = _fia(inputs, lengths, scale).cpu().float() if not bool(inputs[-1].any().cpu()) else None
    assert torch.isfinite(actual).all()
    # BF16 output plus online-softmax/P matmul rounding; also constrain relative
    # RMS error so long-context outputs cannot pass on absolute tolerance alone.
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.02)
    if fia is not None:
        torch.testing.assert_close(actual, fia, atol=0.004, rtol=0.02)
    assert (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt() < 0.008


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "lengths",
    [
        [1],
        [8],
        [127, 128, 129, 255, 256, 257, 511, 513],
        [2047, 2048, 2049, 2303, 2304, 2305, 4095, 4097],
        [1, 129, 513, 2305, 8192, 1, 1, 1] * 4,
        [4096 + i for i in range(64)],
        [32768],
    ],
)
@torch.inference_mode()
def test_xfia_matches_fia_and_fp32(device: torch.device, lengths: list[int], dtype: torch.dtype) -> None:
    inputs = _inputs(device, lengths, dtype)
    scale = 128**-0.5
    output = torch.full_like(inputs[0], float("nan"))
    torch.ops.xllm_ops.x_flash_attention_decode_out(*inputs, scale, output)
    _assert_accuracy(output, inputs, lengths, scale)


@torch.inference_mode()
def test_xfia_scale_and_gqa(device: torch.device) -> None:
    lengths = [129, 257, 1025, 2305, 4097, 4098, 4099, 4100]
    inputs = _inputs(device, lengths, torch.bfloat16, heads=32, kv_heads=8)
    output = torch.empty_like(inputs[0])
    scale = 0.065
    torch.ops.xllm_ops.x_flash_attention_decode_out(*inputs, scale, output)
    _assert_accuracy(output, inputs, lengths, scale)


@torch.inference_mode()
def test_xfia_graph_dynamic_lengths_and_pages(device: torch.device) -> None:
    lengths = [4097] * 8
    inputs = _inputs(device, lengths, torch.bfloat16)
    output = torch.empty_like(inputs[0])
    scale = 128**-0.5
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            torch.ops.xllm_ops.x_flash_attention_decode_out(*inputs, scale, output)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        torch.ops.xllm_ops.x_flash_attention_decode_out(*inputs, scale, output)
    for lengths in ([1, 127, 129, 511, 513, 2303, 2304, 2305], [4097, 1024, 8, 1, 1, 1, 1, 1]):
        inputs[5].copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        inputs[3].copy_(inputs[3].flip(1))
        inputs[-1].copy_(torch.tensor([max(0, length - 17) for length in lengths], dtype=torch.int32, device=device))
        inputs[0].copy_(torch.randn_like(inputs[0]))
        graph.replay()
        _assert_accuracy(output, inputs, lengths, scale)


@pytest.mark.parametrize("lengths", [[8], [2048], [2055], [2303], [2304], [2305], [8192], [129, 2055, 2305, 8192]])
@torch.inference_mode()
def test_dflash2_window_matches_fia_band(device: torch.device, lengths: list[int]) -> None:
    width = 8
    row_lengths = [length for length in lengths for _ in range(width)]
    inputs = _inputs(device, row_lengths, torch.bfloat16)
    starts = [max(0, length - width + i - 2047) for length in lengths for i in range(width)]
    inputs[-1].copy_(torch.tensor(starts, dtype=torch.int32, device=device))
    output = torch.empty_like(inputs[0])
    torch.ops.xllm_ops.x_flash_attention_decode_out(*inputs, 128**-0.5, output)
    _assert_accuracy(output, inputs, row_lengths, 128**-0.5)
    q, k, v, table, *_ = inputs
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), 1).to(device)
    reference = torch.ops.npu.npu_fused_infer_attention_score(
        q,
        k.flatten(2),
        v.flatten(2),
        block_table=table[::width].contiguous(),
        actual_seq_lengths=[width * (i + 1) for i in range(len(lengths))],
        actual_seq_lengths_kv=lengths,
        atten_mask=mask,
        sparse_mode=4,
        pre_tokens=2047,
        next_tokens=7,
        input_layout="TND",
        block_size=128,
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[2],
        scale=128**-0.5,
    )[0]
    torch.testing.assert_close(output.cpu().float(), reference.cpu().float(), atol=0.004, rtol=0.02)
