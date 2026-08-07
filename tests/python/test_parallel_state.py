# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).parents[2]
    / "xllm"
    / "python"
    / "distributed"
    / "parallel_state.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "xllm_parallel_state_test_module",
    _MODULE_PATH,
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)
_normalize_group_specs = _MODULE._normalize_group_specs
get_parallel_group = _MODULE.get_parallel_group
init_parallel_groups = _MODULE.init_parallel_groups


def _group_spec(
    name: str,
    ranks: list[int],
    local_rank: int,
    alias_of: str | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "ranks": ranks,
        "local_rank": local_rank,
        "alias_of": alias_of,
        "group_id": "" if alias_of is not None else f"xllm_python_{name}_0",
    }


def test_accepts_cpp_process_group_specs() -> None:
    specs = _normalize_group_specs(
        [
            _group_spec("world", list(range(8)), 5),
            _group_spec("tp", [4, 5, 6, 7], 1),
            _group_spec("attention_tp", [4, 5], 1),
            _group_spec("single", [5], 0),
            _group_spec("dp", [1, 5], 1),
            _group_spec("cp", [5, 7], 0),
            _group_spec("moe_tp", [4, 5], 1),
            _group_spec("moe_ep", [1, 3, 5, 7], 2),
            _group_spec("encoder_dp", [4, 5, 6, 7], 1),
            _group_spec("attn_tp", [4, 5], 1, "attention_tp"),
            _group_spec("ep", [1, 3, 5, 7], 2, "moe_ep"),
            _group_spec("embedding", [4, 5, 6, 7], 1, "tp"),
            _group_spec("lm_head", [4, 5, 6, 7], 1, "tp"),
        ],
        rank=5,
        world_size=8,
    )
    groups = {spec.name: spec for spec in specs}

    assert groups["world"].ranks == tuple(range(8))
    assert groups["tp"].ranks == (4, 5, 6, 7)
    assert groups["attention_tp"].ranks == (4, 5)
    assert groups["dp"].ranks == (1, 5)
    assert groups["cp"].ranks == (5, 7)
    assert groups["moe_tp"].ranks == (4, 5)
    assert groups["moe_ep"].ranks == (1, 3, 5, 7)
    assert groups["encoder_dp"].alias_of is None
    assert groups["encoder_dp"].group_id != groups["tp"].group_id


def test_initializes_single_rank_groups_from_cpp_specs() -> None:
    _MODULE._contexts.clear()
    group_specs = [
        _group_spec("world", [0], 0),
        _group_spec("tp", [0], 0),
        _group_spec("single", [0], 0),
        _group_spec("dp", [0], 0, "single"),
        _group_spec("cp", [0], 0, "single"),
        _group_spec("moe_tp", [0], 0, "world"),
        _group_spec("moe_ep", [0], 0, "single"),
        _group_spec("lm_head", [0], 0, "tp"),
    ]

    groups = init_parallel_groups(
        host="127.0.0.1",
        port=29999,
        rank=0,
        world_size=1,
        device="cpu",
        group_specs=group_specs,
    )

    assert groups["tp"].world_size == 1
    assert groups["dp"].process_group is groups["single"].process_group
    assert groups["moe_tp"].process_group is groups["world"].process_group
    assert get_parallel_group("lm_head", "cpu").local_rank == 0
    _MODULE._contexts.clear()


def test_rejects_incorrect_cpp_local_rank() -> None:
    with pytest.raises(ValueError, match="local rank mismatch"):
        _normalize_group_specs(
            [
                _group_spec("world", list(range(4)), 2),
                _group_spec("tp", [2, 3], 1),
            ],
            rank=2,
            world_size=4,
        )


def test_rejects_unknown_cpp_alias() -> None:
    with pytest.raises(ValueError, match="aliases unknown group"):
        _normalize_group_specs(
            [
                _group_spec("world", list(range(4)), 2),
                _group_spec("tp", [2, 3], 0, "missing"),
            ],
            rank=2,
            world_size=4,
        )


def test_rejects_duplicate_physical_group_id() -> None:
    with pytest.raises(ValueError, match="duplicate or empty group_id"):
        _normalize_group_specs(
            [
                _group_spec("world", list(range(4)), 2),
                {
                    **_group_spec("tp", [2, 3], 0),
                    "group_id": "xllm_python_world_0",
                },
            ],
            rank=2,
            world_size=4,
        )


def test_requires_cpp_world_group() -> None:
    with pytest.raises(ValueError, match="full world group"):
        _normalize_group_specs(
            [_group_spec("tp", [2, 3], 0)],
            rank=2,
            world_size=4,
        )


@pytest.mark.parametrize("rank", [-1, 4])
def test_rejects_invalid_global_rank(rank: int) -> None:
    with pytest.raises(ValueError, match="must be in"):
        _normalize_group_specs(
            [_group_spec("world", list(range(4)), 0)],
            rank=rank,
            world_size=4,
        )
