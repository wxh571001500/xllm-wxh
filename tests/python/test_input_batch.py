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

import pytest
import torch

from xllm.python.model_executor.input_batch import InputBatch


def _metadata(**overrides):
    values = {
        "num_reqs": 2,
        "num_tokens": 6,
        "num_scheduled_tokens": [4, 2],
        "num_computed_tokens": [10, 20],
        "query_start_loc": [0, 4, 6],
        "is_prefilling": [0, 0],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_input_batch_preserves_upstream_request_layout() -> None:
    batch = InputBatch.from_runtime(
        torch.arange(6, dtype=torch.int32),
        torch.arange(6, dtype=torch.int32),
        _metadata(),
        is_dummy=False,
    )

    assert batch.num_reqs == 2
    assert batch.num_tokens == 6
    assert batch.num_scheduled_tokens == (4, 2)
    assert batch.num_computed_tokens == (10, 20)
    assert batch.query_start_loc == (0, 4, 6)


def test_bind_graph_inputs_only_changes_graph_owned_fields() -> None:
    batch = InputBatch.from_runtime(
        torch.arange(6, dtype=torch.int32),
        torch.arange(6, dtype=torch.int32),
        _metadata(),
        is_dummy=False,
    )
    is_padding = torch.tensor([False, False, False, False, False, False, True, True])

    padded = batch.bind_graph_inputs(
        torch.zeros(8, dtype=torch.int32),
        torch.zeros(8, dtype=torch.int32),
        is_padding,
    )

    assert padded.num_reqs == 2
    assert padded.num_tokens == 6
    assert padded.num_tokens_after_padding == 8
    assert padded.num_scheduled_tokens == (4, 2)
    assert padded.is_padding is is_padding


def test_input_batch_rejects_inconsistent_upstream_fields() -> None:
    with pytest.raises(ValueError, match="query_start_loc disagrees"):
        InputBatch.from_runtime(
            torch.arange(6, dtype=torch.int32),
            torch.arange(6, dtype=torch.int32),
            _metadata(query_start_loc=[0, 3, 6]),
            is_dummy=False,
        )


@pytest.mark.parametrize("num_dummy_tokens", [1, 4])
def test_empty_dp_rank_preserves_dummy_execution_rows(num_dummy_tokens: int) -> None:
    batch = InputBatch.from_runtime(
        torch.zeros(num_dummy_tokens, dtype=torch.int32),
        torch.zeros(num_dummy_tokens, dtype=torch.int32),
        _metadata(
            num_reqs=0,
            num_tokens=num_dummy_tokens,
            num_scheduled_tokens=[],
            num_computed_tokens=[],
            query_start_loc=[0],
            is_prefilling=[],
        ),
        is_dummy=True,
    )

    assert batch.num_reqs == 0
    assert batch.num_tokens == num_dummy_tokens
    assert batch.is_dummy
