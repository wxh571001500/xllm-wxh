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

"""Host-side MoE parallelism policy, separate from numerical kernel tests."""

import pytest

from xllm.python.layers.moe_parallel import Eplv2CommPolicy, mc2_buffer_bytes_per_source_row


def test_a3_window_bound_and_capacity() -> None:
    assert mc2_buffer_bytes_per_source_row(4096, 36, 8, 8) == 5160960
    policy = Eplv2CommPolicy.from_geometry(4096, 288, 8, 8, 1280)
    assert policy.mc2_capacity == 260
    assert policy.select(256) == "mc2"
    assert policy.select(261) == "alltoall"
    with pytest.raises(ValueError, match="Graph"):
        policy.select(261, graph=True)


@pytest.mark.parametrize("rows,expected", [(1, "mc2"), (512, "mc2"), (513, "alltoall"), (1024, "alltoall")])
def test_operator_token_limit_independent_of_phase(rows: int, expected: str) -> None:
    policy = Eplv2CommPolicy.from_geometry(4096, 288, 8, 8, 3072)
    assert policy.mc2_capacity == 512
    assert policy.select(rows) == expected


def test_explicit_modes_fail_before_execution_not_after_collectives() -> None:
    mc2 = Eplv2CommPolicy.from_geometry(4096, 288, 8, 8, 200, mode="mc2")
    with pytest.raises(ValueError, match="exceed capacity"):
        mc2.select(512)
    a2a = Eplv2CommPolicy.from_geometry(4096, 288, 8, 8, 200, mode="alltoall")
    assert a2a.select(1) == "alltoall"
    with pytest.raises(ValueError, match="eager"):
        a2a.select(1, graph=True)


@pytest.mark.parametrize("limit", [0, 513, -1])
def test_invalid_token_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="token limit"):
        Eplv2CommPolicy.from_geometry(4096, 288, 8, 8, 3072, token_limit=limit)


def test_invalid_geometry_never_enters_mc2() -> None:
    policy = Eplv2CommPolicy.from_geometry(16, 32, 2, 2, 3072)
    assert policy.select(1) == "alltoall"
    with pytest.raises(ValueError, match="source rows"):
        policy.select(0)
