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

import sys
import types
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch


def _backend_for_linear_cache(
    conv_cache: torch.Tensor,
    ssm_cache: torch.Tensor,
    verify_width: int = 1,
) -> Any:
    from xllm.python.attention.backend import linear_state_checkpoint_stride
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend

    backend = NpuPagedAttentionBackend(
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
        scale=2**-0.5,
        sliding_window=0,
        is_mla=False,
        device=conv_cache.device,
        dtype=conv_cache.dtype,
        num_decoding_tokens=verify_width,
    )
    backend._kv_caches = [SimpleNamespace(conv=conv_cache, ssm=ssm_cache)]
    backend._linear_state_checkpoint_stride = (
        linear_state_checkpoint_stride(conv_cache, ssm_cache) if conv_cache.shape[0] > 0 else None
    )
    return backend


def _forget_gate(lower_bound: float | None = -5.0) -> SimpleNamespace:
    """Stub Glm5NextForgetGate: raw_projection is identity on the handed raw,
    gate_from_raw applies the safe-gate the backend fuses in-kernel on the
    plain path (only the MTP / non-fused paths materialize it here)."""
    num_heads, head_dim = 1, 2

    def gate_from_raw(raw: torch.Tensor) -> torch.Tensor:
        g = raw.float() + torch.zeros(num_heads * head_dim, dtype=torch.float32).view(1, 1, num_heads, head_dim)
        decay = torch.exp(torch.zeros(num_heads, dtype=torch.float32).view(1, 1, num_heads, 1))
        if lower_bound is not None:
            return lower_bound * torch.sigmoid(decay * g)
        softplus = torch.where(g > 20.0, g, torch.log1p(torch.exp(g)))
        return -decay * softplus

    return SimpleNamespace(
        A_log=torch.zeros(num_heads, dtype=torch.float32),
        dt_bias=torch.zeros(num_heads * head_dim, dtype=torch.float32),
        safe_gate_lower_bound=lower_bound,
        gate_from_raw=gate_from_raw,
    )


def _layer(lower_bound: float | None = -5.0) -> SimpleNamespace:
    conv_weight = torch.ones(6, 1, 3, dtype=torch.bfloat16)
    return SimpleNamespace(
        layer_id=0,
        conv_kernel_size=3,
        head_dim=2,
        num_heads_local=1,
        qkv_dim=2,
        conv_dim=6,
        conv_weight=conv_weight,
        conv_weight_t=conv_weight.squeeze(1).t().contiguous(),
        activation="silu",
        forget_gate=_forget_gate(lower_bound),
    )


@pytest.fixture
def kda_test_environment(monkeypatch: pytest.MonkeyPatch, causal_conv1d_reference: list[dict]) -> Iterator[list[dict]]:
    with monkeypatch.context() as context:
        kernel_calls = _install_kda_stubs(context)
        yield kernel_calls


def _install_kda_stubs(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.npu = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    fla_npu = types.ModuleType("fla_npu")
    fla_npu_ops = types.ModuleType("fla_npu.ops")
    ascendc = types.ModuleType("fla_npu.ops.ascendc")

    kernel_calls: list[dict] = []

    def chunk_kda_fwd(
        query: torch.Tensor,
        *_args: Any,
        initial_state: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # _args[2] is the gate tensor the backend hands the kernel (raw on the
        # fused plain path, materialized gate otherwise); record the fuse flags.
        gate_arg = _args[2] if len(_args) > 2 else None
        beta_arg = _args[3] if len(_args) > 3 else None
        kernel_calls.append(
            {
                "op": "chunk",
                "gate": gate_arg,
                "beta": beta_arg,
                "value": _args[1].clone(),
                **kwargs,
            }
        )
        output = torch.zeros(1, query.shape[1], 1, 2, dtype=query.dtype)
        return output, initial_state + 200

    def recurrent_kda(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        gate_arg = args[3] if len(args) > 3 else None
        beta_arg = args[4] if len(args) > 4 else None
        kernel_calls.append({"op": "recurrent", "gate": gate_arg, "beta": beta_arg, **kwargs})
        return torch.zeros_like(args[0]), kwargs["initial_state"] + 100

    ascendc.chunk_kda_fwd = chunk_kda_fwd
    ascendc.recurrent_kda = recurrent_kda
    fla_npu.ops = fla_npu_ops
    fla_npu_ops.ascendc = ascendc
    monkeypatch.setitem(sys.modules, "fla_npu", fla_npu)
    monkeypatch.setitem(sys.modules, "fla_npu.ops", fla_npu_ops)
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", ascendc)

    from xllm.python.attention import npu_paged_attention

    monkeypatch.setattr(
        npu_paged_attention.kernels,
        "causal_conv1d_update_v2",
        lambda value, *_args, **_kwargs: value,
        raising=False,
    )

    monkeypatch.setattr(
        sys.modules["xllm.python.kernels"],
        "l2_norm",
        lambda value, _eps=1e-6: value,
        raising=False,
    )

    return kernel_calls


def test_execute_linear_without_state_indices(kda_test_environment: None) -> None:
    backend = _backend_for_linear_cache(torch.empty(0), torch.empty(0))
    backend._metadata = SimpleNamespace(
        linear_state_indices=None,
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        has_initial_state=None,
        is_prefill=True,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        kv_seq_lens=None,
        expanded_decode_metadata=None,
    )

    output = backend.execute_linear(
        torch.ones(1, 6, 1, dtype=torch.bfloat16),
        torch.ones(1, 1, 1, dtype=torch.float32),
        _layer(),
        raw_gate_proj=torch.zeros(1, 1, 1, 2, dtype=torch.float32),
    )

    assert output.shape == (1, 1, 1, 2)


def test_merged_spec_verify_uses_remapped_state_indices(
    kda_test_environment: list[dict],
) -> None:
    conv_cache = torch.arange(4 * 3 * 6, dtype=torch.bfloat16).reshape(4, 3, 6)
    ssm_cache = torch.arange(16 * 1 * 2 * 2, dtype=torch.float32).reshape(16, 1, 2, 2)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache, verify_width=4)
    per_row_indices = torch.tensor([1, 1, 3, 3], dtype=torch.int32)
    metadata = SimpleNamespace(
        linear_state_indices=per_row_indices,
        linear_state_read_indices=per_row_indices.clone(),
        linear_state_write_indices=per_row_indices,
        has_initial_state=torch.tensor([1, 1], dtype=torch.int64),
        is_prefill=True,
        is_chunked_prefill=True,
        q_cu_seq_lens=torch.tensor([0, 2, 4], dtype=torch.int32),
        kv_seq_lens=torch.tensor([3, 4, 7, 8], dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        expanded_decode_metadata=None,
        block_table=None,
    )
    backend.prepare(metadata, graph_mode=False)

    output = backend.execute_linear(
        torch.ones(1, 6, 4, dtype=torch.bfloat16),
        torch.ones(1, 4, 1, dtype=torch.float32),
        _layer(),
        raw_gate_proj=torch.zeros(1, 4, 1, 2, dtype=torch.float32),
    )

    assert output.shape == (1, 4, 1, 2)
    assert len(kda_test_environment) == 1
    verify_call = kda_test_environment[0]
    assert verify_call["op"] == "recurrent"
    assert verify_call["inplace_final_state"] is True
    assert verify_call["ssm_state_indices"].tolist() == [
        [4, 5, 6, 7],
        [12, 13, 14, 15],
    ]
    assert verify_call["cu_seqlens"].tolist() == [0, 2, 4]
    assert verify_call["num_accepted_tokens"].tolist() == [1, 1]


def test_execute_linear_prefill_reads_source_and_writes_live(
    kda_test_environment: None,
) -> None:
    conv_cache = torch.arange(4 * 2 * 6, dtype=torch.bfloat16).reshape(4, 2, 6)
    ssm_cache = torch.arange(4 * 1 * 2 * 2, dtype=torch.float32).reshape(4, 1, 2, 2)
    original_conv = conv_cache.clone()
    original_ssm = ssm_cache.clone()
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    backend._metadata = SimpleNamespace(
        linear_state_indices=torch.tensor([3], dtype=torch.int32),
        linear_state_read_indices=torch.tensor([1], dtype=torch.int32),
        linear_state_write_indices=torch.tensor([3], dtype=torch.int32),
        has_initial_state=torch.tensor([1], dtype=torch.int64),
        is_prefill=True,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        kv_seq_lens=None,
        expanded_decode_metadata=None,
    )

    output = backend.execute_linear(
        torch.ones(1, 6, 1, dtype=torch.bfloat16),
        torch.ones(1, 1, 1, dtype=torch.float32),
        _layer(),
        raw_gate_proj=torch.zeros(1, 1, 1, 2, dtype=torch.float32),
    )

    assert output.shape == (1, 1, 1, 2)
    assert torch.equal(conv_cache[1], original_conv[1])
    assert torch.equal(ssm_cache[1], original_ssm[1])
    assert torch.equal(conv_cache[2], original_conv[2])
    assert torch.equal(ssm_cache[2], original_ssm[2])
    assert not torch.equal(conv_cache[3], original_conv[3])
    assert torch.equal(ssm_cache[3], original_ssm[1] + 200)


def _plain_prefill_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        linear_state_indices=torch.tensor([0], dtype=torch.int32),
        linear_state_read_indices=torch.tensor([0], dtype=torch.int32),
        linear_state_write_indices=torch.tensor([0], dtype=torch.int32),
        has_initial_state=torch.tensor([0], dtype=torch.int64),
        is_prefill=True,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        kv_seq_lens=None,
        expanded_decode_metadata=None,
    )


def test_plain_prefill_fuses_gate_in_kernel(kda_test_environment: list[dict]) -> None:
    # Plain (non-MTP) prefill: the kernel must compute the safe-gate itself
    # (use_gate_in_kernel=True + safe_gate + lower_bound) and receive the raw
    # projection verbatim — never a python-materialized gate (the double
    # safe-gate bug). The gate is not materialized on this path at all.
    conv_cache = torch.zeros(4, 2, 6, dtype=torch.bfloat16)
    ssm_cache = torch.zeros(4, 1, 2, 2, dtype=torch.float32)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    backend._metadata = _plain_prefill_metadata()
    raw = torch.full((1, 1, 1, 2), 0.5, dtype=torch.float32)

    backend.execute_linear(
        torch.ones(1, 6, 1, dtype=torch.bfloat16),
        torch.ones(1, 1, 1, dtype=torch.float32),
        _layer(lower_bound=-5.0),
        raw_gate_proj=raw,
    )

    chunk_calls = [c for c in kda_test_environment if c["op"] == "chunk"]
    assert len(chunk_calls) == 1
    call = chunk_calls[0]
    assert call["use_gate_in_kernel"] is True
    assert call["safe_gate"] is True
    assert call["lower_bound"] == -5.0
    # The kernel gets the raw projection (0.5), not lb*sigmoid(...) of it.
    assert torch.allclose(call["gate"].reshape(-1), raw.reshape(-1))
    torch.testing.assert_close(call["beta"], torch.ones(1, 1, 1).sigmoid())


def test_lower_bound_out_of_range_falls_back_to_python_gate(
    kda_test_environment: list[dict],
) -> None:
    # AscendC safe_gate requires lower_bound in [-5, 0); an out-of-range config
    # must fall back to the materialized python gate (use_gate_in_kernel=False)
    # rather than passing an illegal lower_bound to the kernel.
    conv_cache = torch.zeros(4, 2, 6, dtype=torch.bfloat16)
    ssm_cache = torch.zeros(4, 1, 2, 2, dtype=torch.float32)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache)
    backend._metadata = _plain_prefill_metadata()
    raw = torch.full((1, 1, 1, 2), 0.5, dtype=torch.float32)

    backend.execute_linear(
        torch.ones(1, 6, 1, dtype=torch.bfloat16),
        torch.ones(1, 1, 1, dtype=torch.float32),
        _layer(lower_bound=-6.0),
        raw_gate_proj=raw,
    )

    chunk_calls = [c for c in kda_test_environment if c["op"] == "chunk"]
    assert len(chunk_calls) == 1
    call = chunk_calls[0]
    assert call["use_gate_in_kernel"] is False
    assert "lower_bound" not in call
    # The kernel gets the materialized gate lb*sigmoid(decay*raw), not the raw.
    expected = _forget_gate(-6.0).gate_from_raw(raw)
    assert torch.allclose(call["gate"].reshape(-1), expected.reshape(-1))


def test_chunked_spec_verify_fuses_gate_and_beta(
    kda_test_environment: list[dict],
) -> None:
    conv_cache = torch.arange(4 * 3 * 6, dtype=torch.bfloat16).reshape(4, 3, 6)
    ssm_cache = torch.arange(16 * 1 * 2 * 2, dtype=torch.float32).reshape(16, 1, 2, 2)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache, verify_width=4)
    per_row_indices = torch.tensor([1, 1, 3, 3], dtype=torch.int32)
    metadata = SimpleNamespace(
        linear_state_indices=per_row_indices,
        linear_state_read_indices=per_row_indices.clone(),
        linear_state_write_indices=per_row_indices,
        has_initial_state=torch.tensor([1, 1], dtype=torch.int64),
        is_prefill=True,
        is_chunked_prefill=True,
        q_cu_seq_lens=torch.tensor([0, 2, 4], dtype=torch.int32),
        kv_seq_lens=torch.tensor([3, 4, 7, 8], dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        expanded_decode_metadata=None,
        block_table=None,
    )
    backend.prepare(metadata, graph_mode=False)

    backend.execute_linear(
        torch.ones(1, 6, 4, dtype=torch.bfloat16),
        torch.ones(1, 4, 1, dtype=torch.float32),
        _layer(lower_bound=-5.0),
        raw_gate_proj=torch.zeros(1, 4, 1, 2, dtype=torch.float32),
    )

    assert len(kda_test_environment) == 1
    verify_call = kda_test_environment[0]
    assert verify_call["op"] == "recurrent"
    assert verify_call["use_gate_in_kernel"] is True
    assert verify_call["use_beta_sigmoid_in_kernel"] is True
    assert torch.equal(verify_call["gate"], torch.zeros(4, 1, 2))
    assert torch.equal(verify_call["beta"], torch.ones(4, 1))


@pytest.mark.parametrize("in_graph", [False, True])
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("chunked_verify", [False, True])
def test_spec_verify_uses_v3_without_environment_flags(
    kda_test_environment: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    in_graph: bool,
    width: int,
    chunked_verify: bool,
) -> None:
    from xllm.python.attention import kda_linear_attention

    monkeypatch.setattr(kda_linear_attention, "in_acl_graph", lambda: in_graph)
    backend = _backend_for_linear_cache(torch.zeros(4, 2, 6), torch.zeros(4, 1, 2, 2), verify_width=4)
    slots = torch.tensor([1, 3], dtype=torch.int32)
    total_rows = 2 * width
    backend._metadata = SimpleNamespace(
        linear_state_indices=slots.repeat_interleave(width) if in_graph else slots,
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        has_initial_state=None,
        is_prefill=False,
        is_chunked_prefill=chunked_verify,
        is_spec_verify=chunked_verify,
        q_cu_seq_lens=torch.arange(
            0, total_rows + 1, width if chunked_verify and not in_graph else 1, dtype=torch.int32
        ),
        q_seq_lens=torch.full((2,), width, dtype=torch.int32),
        kv_seq_lens=torch.arange(total_rows, dtype=torch.int32) + 10,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        expanded_decode_metadata=SimpleNamespace() if in_graph else None,
    )
    expected_output = torch.zeros(1, total_rows, 1, 2)
    verify_calls = []

    def spec_verify_v3(*args: Any) -> torch.Tensor:
        verify_calls.append(args)
        return expected_output

    monkeypatch.setattr(backend, "_spec_verify_v3", spec_verify_v3)
    output = backend.execute_linear(
        torch.ones(1, 6, total_rows, dtype=torch.bfloat16),
        torch.ones(1, total_rows, 1),
        _layer(),
        raw_gate_proj=torch.zeros(1, total_rows, 1, 2),
    )

    assert output is expected_output
    assert len(verify_calls) == 1
    assert torch.equal(verify_calls[0][4], slots.to(torch.int64))
    assert not kda_test_environment


def test_spec_verify_rejects_nonuniform_widths(kda_test_environment: list[dict]) -> None:
    conv_cache = torch.ones(4, 2, 6)
    ssm_cache = torch.ones(4, 1, 2, 2)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache, verify_width=4)
    backend._metadata = SimpleNamespace(
        linear_state_indices=torch.tensor([1, 1, 3, 3, 3], dtype=torch.int32),
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        is_prefill=False,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.arange(6, dtype=torch.int32),
        expanded_decode_metadata=None,
    )

    with pytest.raises(RuntimeError, match="uniform rows per sequence"):
        backend.execute_linear(
            torch.ones(1, 6, 5, dtype=torch.bfloat16),
            torch.ones(1, 5, 1),
            _layer(),
            raw_gate_proj=torch.zeros(1, 5, 1, 2),
        )

    assert not kda_test_environment
    assert torch.equal(conv_cache, torch.ones_like(conv_cache))
    assert torch.equal(ssm_cache, torch.ones_like(ssm_cache))


def test_plain_decode_keeps_fused_gate_without_verify_state(kda_test_environment: list[dict]) -> None:
    backend = _backend_for_linear_cache(torch.zeros(4, 2, 6, dtype=torch.bfloat16), torch.zeros(4, 1, 2, 2))
    backend._metadata = SimpleNamespace(
        linear_state_indices=torch.tensor([1, 3], dtype=torch.int32),
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        has_initial_state=None,
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
        q_cu_seq_lens=torch.arange(3, dtype=torch.int32),
        expanded_decode_metadata=None,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
    )

    output = backend.execute_linear(
        torch.ones(1, 6, 2, dtype=torch.bfloat16),
        torch.ones(1, 2, 1),
        _layer(),
        raw_gate_proj=torch.zeros(1, 2, 1, 2),
    )

    assert output.shape == (1, 2, 1, 2)
    assert len(kda_test_environment) == 1
    assert kda_test_environment[0]["op"] == "recurrent"
    assert kda_test_environment[0]["use_gate_in_kernel"] is True
    assert kda_test_environment[0]["use_beta_sigmoid_in_kernel"] is True
    assert torch.equal(kda_test_environment[0]["beta"], torch.ones(2, 1))
    assert kda_test_environment[0]["cu_seqlens"].tolist() == [0, 1, 2]


def test_varlen_prefill_uses_one_native_convolution(
    kda_test_environment: list[dict], causal_conv1d_reference: list[dict]
) -> None:
    backend = _backend_for_linear_cache(torch.zeros(4, 2, 6, dtype=torch.bfloat16), torch.zeros(4, 1, 2, 2))
    backend._metadata = SimpleNamespace(
        linear_state_indices=torch.tensor([0, 1, 3], dtype=torch.int32),
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        has_initial_state=None,
        is_prefill=True,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.tensor([0, 1, 3, 5], dtype=torch.int32),
        q_cu_seq_lens_host_values=[1, 3, 5],
        expanded_decode_metadata=None,
    )

    output = backend.execute_linear(
        torch.arange(30, dtype=torch.bfloat16).reshape(1, 6, 5),
        torch.ones(1, 5, 1),
        _layer(),
        raw_gate_proj=torch.zeros(1, 5, 1, 2),
    )

    assert output.shape == (1, 5, 1, 2)
    assert len(causal_conv1d_reference) == 1
    assert causal_conv1d_reference[0]["query_start_loc"] == [0, 1, 3, 5]
    assert causal_conv1d_reference[0]["run_mode"] == 0
    assert [call["op"] for call in kda_test_environment] == ["chunk"]
    assert [call["cu_seqlens"].tolist() for call in kda_test_environment] == [[0, 1, 3, 5]]
    expected_value = causal_conv1d_reference[0]["output"][:, -2:].reshape(1, 5, 1, 2)
    actual_value = kda_test_environment[0]["value"]
    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    torch.testing.assert_close(
        backend._kv_caches[0].ssm[[0, 1, 3]],
        torch.full((3, 1, 2, 2), 200.0),
        rtol=0,
        atol=0,
    )


def test_varlen_prefill_uses_host_q_cu_without_device_to_host_sync(
    kda_test_environment: list[dict], causal_conv1d_reference: list[dict]
) -> None:
    class _DeviceCuSeqLens:
        def numel(self) -> int:
            return 4

        def to(self, dtype: torch.dtype) -> torch.Tensor:
            del dtype
            return torch.tensor([0, 1, 3, 5], dtype=torch.int32)

        def tolist(self) -> list[int]:
            raise AssertionError("q_cu_seq_lens must not be copied from device")

        @property
        def device(self) -> SimpleNamespace:
            return SimpleNamespace(type="npu")

    backend = _backend_for_linear_cache(
        torch.zeros(4, 2, 6, dtype=torch.bfloat16),
        torch.zeros(4, 1, 2, 2),
    )
    backend._metadata = SimpleNamespace(
        linear_state_indices=torch.tensor([0, 1, 3], dtype=torch.int32),
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        has_initial_state=None,
        is_prefill=True,
        is_chunked_prefill=False,
        q_cu_seq_lens=_DeviceCuSeqLens(),
        q_cu_seq_lens_host_values=[0, 1, 3, 5],
        expanded_decode_metadata=None,
    )

    backend.execute_linear(
        torch.arange(30, dtype=torch.bfloat16).reshape(1, 6, 5),
        torch.ones(1, 5, 1),
        _layer(),
        raw_gate_proj=torch.zeros(1, 5, 1, 2),
    )

    assert causal_conv1d_reference[0]["query_start_loc"] == [0, 1, 3, 5]
    assert kda_test_environment[0]["cu_seqlens"].tolist() == [0, 1, 3, 5]


def test_varlen_prefill_falls_back_to_matching_host_q_lens(
    kda_test_environment: list[dict], causal_conv1d_reference: list[dict]
) -> None:
    backend = _backend_for_linear_cache(
        torch.zeros(4, 2, 6, dtype=torch.bfloat16),
        torch.zeros(4, 1, 2, 2),
    )
    backend._metadata = SimpleNamespace(
        linear_state_indices=torch.tensor([0, 1, 3], dtype=torch.int32),
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        has_initial_state=None,
        is_prefill=True,
        is_chunked_prefill=False,
        q_cu_seq_lens=torch.tensor([0, 1, 3, 5], dtype=torch.int32),
        q_cu_seq_lens_host_values=[0, 1],
        q_seq_lens_host=torch.tensor([1, 2, 2], dtype=torch.int32),
        expanded_decode_metadata=None,
    )

    backend.execute_linear(
        torch.arange(30, dtype=torch.bfloat16).reshape(1, 6, 5),
        torch.ones(1, 5, 1),
        _layer(),
        raw_gate_proj=torch.zeros(1, 5, 1, 2),
    )

    assert causal_conv1d_reference[0]["query_start_loc"] == [0, 1, 3, 5]
    assert kda_test_environment[0]["cu_seqlens"].tolist() == [0, 1, 3, 5]


def _verify_metadata(
    slots: torch.Tensor,
    base_lengths: torch.Tensor,
    width: int,
    in_graph: bool = False,
) -> SimpleNamespace:
    from xllm.python.attention.expanded_decode_metadata import ExpandedDecodeMetadata

    per_row_slots = slots.repeat_interleave(width)
    kv_seq_lens = (base_lengths[:, None] + torch.arange(width)).flatten()
    expanded = None
    if in_graph and width > 1:
        expanded = ExpandedDecodeMetadata(
            kv_seq_lens=kv_seq_lens,
            block_table=torch.zeros(per_row_slots.numel(), 1, dtype=torch.int32),
            paged_kv_indptr=None,
            paged_kv_indices=None,
            paged_kv_last_page_len=None,
            paged_attention_tiling_data=None,
            kv_seq_lens_host=None,
            kv_seq_lens_host_values=None,
        )
    return SimpleNamespace(
        linear_state_indices=per_row_slots,
        slot_mapping=torch.arange(per_row_slots.numel(), dtype=torch.int32),
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        has_initial_state=None,
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=True,
        q_cu_seq_lens=torch.arange(per_row_slots.numel() + 1, dtype=torch.int32),
        q_seq_lens=torch.full((slots.numel(),), width, dtype=torch.int32),
        kv_seq_lens=kv_seq_lens,
        num_accepted_tokens=torch.ones(slots.numel(), dtype=torch.int32),
        expanded_decode_metadata=expanded,
    )


@pytest.mark.parametrize("in_graph", [False, True])
@pytest.mark.parametrize("widths", [(1, 2, 4), (2, 4, 4), (4, 4, 4), (1, 4, 4)])
def test_v3_consecutive_widths_use_framework_ssm_checkpoints(
    kda_test_environment: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    in_graph: bool,
    widths: tuple[int, ...],
) -> None:
    from fla_npu.ops import ascendc

    from xllm.python.attention import kda_linear_attention, npu_paged_attention

    monkeypatch.setattr(kda_linear_attention, "in_acl_graph", lambda: in_graph)
    monkeypatch.setattr(npu_paged_attention, "in_acl_graph", lambda: in_graph)
    conv_cache = torch.zeros(4, 5, 6, dtype=torch.bfloat16)
    ssm_cache = torch.zeros(16, 1, 2, 2)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache, verify_width=4)
    slots = torch.tensor([3, 1])
    base_lengths = torch.tensor([16, 32])
    base_indices = slots * 4
    ssm_cache[base_indices] = torch.tensor([1.0, 3.0]).view(2, 1, 1, 1)
    expected_ssm = ssm_cache[base_indices].clone()
    previous_checkpoints = None
    previous_width = 1

    def _make_recurrent_kda(
        expected_state: torch.Tensor,
        current_width: int,
        expected_accepted: torch.Tensor,
    ) -> Any:
        def recurrent_kda(
            query: torch.Tensor,
            *_args: Any,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            state_indices = kwargs["ssm_state_indices"].long()
            kernel_accepted = kwargs["num_accepted_tokens"].long()
            assert torch.equal(kernel_accepted.cpu(), expected_accepted.long())
            assert bool(((kernel_accepted >= 1) & (kernel_accepted <= current_width)).all())
            selected_indices = state_indices[
                torch.arange(2),
                kernel_accepted - 1,
            ]
            state_pool = kwargs["initial_state"]
            initial_state = state_pool[selected_indices].clone()
            torch.testing.assert_close(initial_state, expected_state, rtol=0, atol=0)
            for row in range(current_width):
                state_pool.index_copy_(
                    0,
                    state_indices[:, row],
                    initial_state + row + 1,
                )
            return torch.zeros_like(query), state_pool

        return recurrent_kda

    for width in widths:
        accepted = torch.ones(2, dtype=torch.int32)
        if previous_checkpoints is not None:
            accepted = torch.tensor([previous_width, 1], dtype=torch.int32)
            expected_ssm = previous_checkpoints[
                torch.arange(2),
                accepted.to(torch.int64) - 1,
            ]
            base_lengths += accepted

        monkeypatch.setattr(
            ascendc,
            "recurrent_kda",
            _make_recurrent_kda(expected_ssm, width, accepted),
        )
        metadata = _verify_metadata(slots, base_lengths, width, in_graph)
        metadata.num_accepted_tokens = accepted
        metadata.block_table = None
        backend.prepare(metadata, graph_mode=False)
        output = backend.execute_linear(
            torch.ones(1, 6, 2 * width, dtype=torch.bfloat16),
            torch.ones(1, 2 * width, 1),
            _layer(),
            raw_gate_proj=torch.zeros(1, 2 * width, 1, 2),
        )

        assert output.shape == (1, 2 * width, 1, 2)
        checkpoint_indices = base_indices[:, None] + torch.arange(width)
        previous_checkpoints = ssm_cache[checkpoint_indices].clone()
        expected_checkpoints = expected_ssm[:, None] + torch.arange(1, width + 1).view(
            1,
            width,
            1,
            1,
            1,
        )
        torch.testing.assert_close(previous_checkpoints, expected_checkpoints)
        previous_width = width


def test_spec_conv_treats_reserved_padding_slot_as_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python.kernels_npu.causal_conv1d import causal_conv1d_update_v2

    captured_pad_slot_ids = []

    def native_conv(*args: Any) -> torch.Tensor:
        captured_pad_slot_ids.append(args[8])
        return args[0]

    monkeypatch.setattr(torch.ops.xllm_ops, "causal_conv1d_update_v2", native_conv, raising=False)
    causal_conv1d_update_v2(
        torch.ones(2, 6),
        torch.ones(6, 3),
        torch.zeros(2, 5, 6),
        torch.tensor([1, 0]),
        torch.tensor([0, 1, 2]),
        1,
        torch.ones(2, dtype=torch.int32),
    )

    assert captured_pad_slot_ids == [-1]


def test_v3_rejects_width_above_capacity_before_mutation(kda_test_environment: list[dict]) -> None:
    conv_cache = torch.ones(4, 2, 6, dtype=torch.bfloat16)
    ssm_cache = torch.ones(4, 1, 2, 2)
    backend = _backend_for_linear_cache(conv_cache, ssm_cache, verify_width=4)
    backend._metadata = _verify_metadata(torch.tensor([1, 3]), torch.tensor([16, 32]), 5)

    with pytest.raises(RuntimeError, match="conv checkpoint capacity"):
        backend.execute_linear(
            torch.ones(1, 6, 10, dtype=torch.bfloat16),
            torch.ones(1, 10, 1),
            _layer(),
            raw_gate_proj=torch.zeros(1, 10, 1, 2),
        )

    assert not kda_test_environment
    assert torch.equal(conv_cache, torch.ones_like(conv_cache))
    assert torch.equal(ssm_cache, torch.ones_like(ssm_cache))
