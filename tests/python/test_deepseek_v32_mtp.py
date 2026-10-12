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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from xllm.python.models import deepseek_v32_mtp


def _mtp_input_stub(enable_rot: bool) -> SimpleNamespace:
    hidden_size = 8
    enorm = nn.RMSNorm(hidden_size, eps=1e-6)
    hnorm = nn.RMSNorm(hidden_size, eps=1e-6)
    with torch.no_grad():
        enorm.weight.copy_(torch.linspace(0.5, 1.5, hidden_size))
        hnorm.weight.copy_(torch.linspace(1.5, 0.5, hidden_size))
    return SimpleNamespace(
        embed_tokens=nn.Embedding(16, hidden_size),
        rot=nn.Linear(hidden_size, hidden_size, bias=False),
        enorm=enorm,
        hnorm=hnorm,
        eh_proj=nn.Identity(),
        rotary=lambda positions: (None, None, None, None),
        layers=(),
        norm=lambda hidden, residual: (hidden, residual),
        enable_rot=enable_rot,
    )


def _cpu_fused_eh_reference(
    embed: torch.Tensor,
    carried: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.cat(
        (
            nn.functional.rms_norm(embed, (embed.shape[-1],), enorm_weight, eps),
            nn.functional.rms_norm(carried, (carried.shape[-1],), hnorm_weight, eps),
        ),
        dim=-1,
    )


@pytest.mark.parametrize("enable_rot", [False, True])
@pytest.mark.parametrize("with_carried", [False, True])
def test_mtp_fused_eh_input_matches_reference(
    enable_rot: bool,
    with_carried: bool,
) -> None:
    torch.manual_seed(2026)
    model = _mtp_input_stub(enable_rot)
    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    carried = torch.randn(2, 8) if with_carried else None
    token_hidden = model.embed_tokens(input_ids)
    input_embedding = token_hidden if carried is None else carried
    rotated_embedding = model.rot(input_embedding) if enable_rot else input_embedding
    expected = _cpu_fused_eh_reference(
        token_hidden,
        rotated_embedding,
        model.enorm.weight,
        model.hnorm.weight,
        model.enorm.eps,
    )

    with patch.object(
        deepseek_v32_mtp.kernels, "fused_eh_norm", side_effect=_cpu_fused_eh_reference, create=True
    ) as fused:
        result = deepseek_v32_mtp.DeepseekV32MtpModel.forward(model, input_ids, positions, carried)
        fused.assert_called_once()

    torch.testing.assert_close(result, expected, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(1, 7168), (8, 128), (2, 3, 128)])
def test_npu_fused_eh_norm_matches_legacy_bf16(shape: tuple[int, ...]) -> None:
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")

    from xllm.python.kernels_npu.triton.eh_norm import fused_eh_norm

    torch.manual_seed(2026)
    eps = 1e-6
    embed = torch.randn(shape, dtype=torch.float32).to(device="npu", dtype=torch.bfloat16)
    carried = torch.randn(shape, dtype=torch.float32).to(device="npu", dtype=torch.bfloat16)
    # Both paths normalize the trailing dimension, including for 3D inputs.
    enorm_weight = torch.randn(shape[-1], dtype=torch.float32).to(device="npu", dtype=torch.bfloat16)
    hnorm_weight = torch.randn(shape[-1], dtype=torch.float32).to(device="npu", dtype=torch.bfloat16)

    # xllm_ops::rms_norm calls torch_npu's npu_rms_norm in the NPU backend.
    old_embed, _ = torch_npu.npu_rms_norm(embed, enorm_weight, eps)
    old_carried, _ = torch_npu.npu_rms_norm(carried, hnorm_weight, eps)
    legacy = torch.cat((old_embed, old_carried), dim=-1)
    fused = fused_eh_norm(embed, carried, enorm_weight, hnorm_weight, eps)
    torch.npu.synchronize()

    assert fused.shape == legacy.shape
    assert fused.dtype == legacy.dtype == torch.bfloat16
    torch.testing.assert_close(fused.cpu(), legacy.cpu(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", [(0, 8), (2, 0, 8)])
def test_npu_fused_eh_norm_accepts_empty_input(shape: tuple[int, ...]) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")

    from xllm.python.kernels_npu.triton.eh_norm import fused_eh_norm

    embed = torch.empty(shape, dtype=torch.bfloat16, device="npu")
    weight = torch.ones(shape[-1], dtype=torch.bfloat16, device="npu")
    result = fused_eh_norm(embed, embed, weight, weight, 1e-6)

    assert result.shape == (*shape[:-1], 2 * shape[-1])
    assert result.dtype == embed.dtype
    assert result.device == embed.device


def test_mtp_constructor_defers_shared_target_modules() -> None:
    config = {
        "device": "cpu",
        "dtype": "float32",
        "tp_size": 1,
        "tp_rank": 0,
    }
    model_config = SimpleNamespace(vocab_size=16)
    mtp_body = nn.Module()
    mtp_body.embed_tokens = None

    with (
        patch.object(
            deepseek_v32_mtp.DeepseekV3Config,
            "from_dict",
            return_value=model_config,
        ),
        patch.object(
            deepseek_v32_mtp.DeepseekV3ForCausalLM,
            "resolve_dtype",
            return_value=torch.float32,
        ),
        patch.object(
            deepseek_v32_mtp,
            "DeepseekV32MtpModel",
            return_value=mtp_body,
        ),
    ):
        draft = deepseek_v32_mtp.DeepseekV32MtpForCausalLM(config)

    assert draft.lm_head is None
    assert draft.model is mtp_body
    assert draft.model.embed_tokens is None

    target_lm_head = nn.Linear(4, 16, bias=False)
    target_embedding = nn.Embedding(16, 4)
    draft.lm_head = target_lm_head
    draft.model.embed_tokens = target_embedding

    assert draft.lm_head is target_lm_head
    assert draft.model.embed_tokens is target_embedding


def test_mtp_load_rejects_missing_required_weights() -> None:
    config = {
        "device": "cpu",
        "dtype": "float32",
        "tp_size": 1,
        "tp_rank": 0,
    }
    model_config = SimpleNamespace(vocab_size=16)
    mtp_body = nn.Module()
    mtp_body.embed_tokens = None

    with (
        patch.object(
            deepseek_v32_mtp.DeepseekV3Config,
            "from_dict",
            return_value=model_config,
        ),
        patch.object(
            deepseek_v32_mtp.DeepseekV3ForCausalLM,
            "resolve_dtype",
            return_value=torch.float32,
        ),
        patch.object(
            deepseek_v32_mtp,
            "DeepseekV32MtpModel",
            return_value=mtp_body,
        ),
        patch.object(deepseek_v32_mtp.DeepseekV3ForCausalLM, "load_weights"),
    ):
        draft = deepseek_v32_mtp.DeepseekV32MtpForCausalLM(config)
        with pytest.raises(KeyError, match="missing required MTP weight"):
            draft.load_weights([], tp_rank=0, tp_size=1)
