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

from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.model_executor.forward_context import AclGraphExecutionState, ForwardContext, forward_context
from xllm.python.models import deepseek_v4


def test_graph_silu_uses_aten_path_without_changing_eager_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = torch.tensor([-2.0, 0.0, 2.0])
    native_silu = MagicMock(return_value=torch.full_like(gate, 7.0))
    monkeypatch.setattr(deepseek_v4.torch_npu, "npu_silu", native_silu)

    eager = deepseek_v4._silu_for_execution(gate)
    context = ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[],
        execution_state=AclGraphExecutionState(persistent_buffers={}),
    )
    with forward_context(context):
        graph = deepseek_v4._silu_for_execution(gate)

    torch.testing.assert_close(eager, torch.full_like(gate, 7.0))
    torch.testing.assert_close(graph, torch.nn.functional.silu(gate))
    native_silu.assert_called_once_with(gate)
