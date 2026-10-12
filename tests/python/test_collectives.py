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

"""Tests for Python process-group rendezvous ownership."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from xllm.python.models import glm5_2

_MODULE_PATH = Path(__file__).parents[2] / "xllm" / "python" / "distributed" / "collectives.py"
_SPEC = importlib.util.spec_from_file_location("_xllm_collectives_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
collectives = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collectives)


class _FakeGroup:
    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def test_moe_ep_hccl_info_uses_existing_group() -> None:
    queried = []
    backend = SimpleNamespace(get_hccl_comm_name=lambda rank: queried.append(rank) or "ep_test_group")
    group = SimpleNamespace(rank=lambda: 1, size=lambda: 2, _get_backend=lambda device: backend)
    collectives._groups[("moe_ep", "cpu")] = group
    assert collectives.moe_ep_hccl_info("cpu") == ("ep_test_group", 1, 2)
    assert queried == [1]


def test_moe_ep_hccl_info_rejects_missing_group() -> None:
    with pytest.raises(RuntimeError, match="initialized process group"):
        collectives.moe_ep_hccl_info("cpu")


@pytest.mark.parametrize("comm_name", ["", None, 1])
def test_moe_ep_hccl_info_rejects_invalid_name(comm_name: object) -> None:
    backend = SimpleNamespace(get_hccl_comm_name=lambda rank: comm_name)
    collectives._groups[("moe_ep", "cpu")] = SimpleNamespace(
        rank=lambda: 0, size=lambda: 2, _get_backend=lambda device: backend
    )
    with pytest.raises(RuntimeError, match="nonempty HCCL"):
        collectives.moe_ep_hccl_info("cpu")


def test_moe_ep_hccl_info_rejects_non_hccl_group() -> None:
    collectives._groups[("moe_ep", "cpu")] = SimpleNamespace(
        rank=lambda: 0, size=lambda: 2, _get_backend=lambda device: object()
    )
    with pytest.raises(RuntimeError, match="HCCL process group"):
        collectives.moe_ep_hccl_info("cpu")


@pytest.fixture(autouse=True)
def _clear_collective_state():
    def reset():
        collectives._groups.clear()
        collectives._group_ranks.clear()
        collectives._stores.clear()
        collectives._symm_eligible.clear()
        collectives._symm_buffers.clear()
        collectives._world_topology = None
        collectives._world_initialized = False

    reset()
    yield
    reset()


class _FakeStore:
    def __init__(self, topology: list[dict[str, object]] | None = None) -> None:
        self.values: dict[str, bytes] = {}
        if topology is not None:
            for rank, entry in enumerate(topology):
                self.values[f"xllm/python_collectives/topology/v1/{rank}"] = json.dumps(entry).encode("utf-8")

    def set(self, key: str, value: str) -> None:
        self.values[key] = value.encode("utf-8")

    def get(self, key: str) -> bytes:
        return self.values[key]


def _mock_process_groups(
    monkeypatch: pytest.MonkeyPatch,
    global_rank: int,
    topology: list[dict[str, object]] | None = None,
):
    """Stand in for c10d so the rendezvous can be inspected without a world.

    ``new_group`` reports this rank's position inside the membership it is
    handed, which is what the module checks its caller's rank against.
    """
    if topology is None:
        topology = [{"hostname": "node-0", "device_index": rank} for rank in range(16)]
    base_store = _FakeStore(topology)
    tcp_store = MagicMock(return_value=base_store)
    init_world = MagicMock()
    new_group = MagicMock(
        side_effect=lambda ranks, timeout, backend: _FakeGroup(
            ranks.index(global_rank) if global_rank in ranks else -1, len(ranks)
        )
    )
    monkeypatch.setattr(dist, "TCPStore", tcp_store)
    monkeypatch.setattr(dist, "init_process_group", init_world)
    monkeypatch.setattr(dist, "new_group", new_group)
    monkeypatch.setattr(collectives.socket, "gethostname", lambda: "node-0")
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda _a, _b: True)
    return base_store, tcp_store, init_world, new_group


def _run_glm_ep1_tp_collective(global_rank: int, rendezvous_path: str) -> None:
    world_size = 4
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous_path}",
            rank=global_rank,
            world_size=world_size,
            timeout=timedelta(seconds=20),
        )
        tp_groups = [
            dist.new_group(
                ranks=[0, 1],
                backend="gloo",
                timeout=timedelta(seconds=20),
            ),
            dist.new_group(
                ranks=[2, 3],
                backend="gloo",
                timeout=timedelta(seconds=20),
            ),
        ]
        collectives._groups[("tp", "cpu")] = tp_groups[global_rank // 2]
        # This is the topology that exposed the bug: with EP1, moe_tp spans
        # both CP cohorts and must not be used to combine expert partials.
        collectives._groups[("moe_tp", "cpu")] = dist.group.WORLD

        cp_rank = global_rank // 2
        tp_rank = global_rank % 2
        local_value = float(cp_rank * 10 + tp_rank + 1)
        routed = torch.tensor([[local_value]])
        shared = torch.tensor([[local_value * 10]])
        moe = SimpleNamespace(
            ep_size=1,
            moe_tp_size=world_size,
            cfg=SimpleNamespace(tp_size=2),
        )

        with patch.object(
            glm5_2.distributed,
            "all_reduce_",
            collectives.all_reduce_,
            create=True,
        ):
            output = glm5_2.Glm52MoE._combine_expert_outputs(moe, routed, shared, False)

        expected = torch.tensor([[33.0 if cp_rank == 0 else 253.0]])
        torch.testing.assert_close(output, expected)
    finally:
        collectives._groups.clear()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_glm53_dp_cp_ep_collectives(global_rank: int, rendezvous_path: str) -> None:
    world_size = 8
    dp_rank, local_rank = divmod(global_rank, 4)
    cp_rank, tp_rank = divmod(local_rank, 2)
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous_path}",
            rank=global_rank,
            world_size=world_size,
            timeout=timedelta(seconds=20),
        )
        for name, size, stride, expected_rank in (
            ("tp", 2, None, tp_rank),
            ("dp", 2, None, dp_rank),
            ("moe_ep", world_size, None, global_rank),
            ("cp", 2, 2, cp_rank),
        ):
            memberships = collectives._group_memberships(name, size, world_size, stride)
            for index, ranks in enumerate(memberships):
                group = dist.new_group(ranks=ranks, backend="gloo", timeout=timedelta(seconds=20))
                if global_rank in ranks:
                    assert group.rank() == expected_rank
                    if name == "cp":
                        assert index == dp_rank * 2 + tp_rank
                    collectives._groups[(name, "cpu")] = group

        cp_rows = collectives.all_gather(torch.tensor([[float(global_rank)]]), 0, 2, "cp")
        expected_cp = torch.tensor([[float(dp_rank * 4 + tp_rank)], [float(dp_rank * 4 + 2 + tp_rank)]])
        torch.testing.assert_close(cp_rows, expected_cp)

        local_rows = torch.full((1 if dp_rank == 0 else 3, 1), float(global_rank))
        gathered, offset = collectives.gather_dp_execution_tokens(local_rows, (1, 3), dp_rank)
        expected_dp = torch.tensor([float(local_rank)] + [float(local_rank + 4)] * 3).unsqueeze(-1)
        assert offset == dp_rank
        torch.testing.assert_close(gathered, expected_dp)

        expert_partial = torch.tensor([[float(global_rank + 1)]])
        collectives.moe_ep_all_reduce(expert_partial)
        torch.testing.assert_close(expert_partial, torch.tensor([[36.0]]))
    finally:
        collectives._groups.clear()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_parallel_groups_share_one_multitenant_tcp_store(monkeypatch):
    base_store, tcp_store, init_world, new_group = _mock_process_groups(monkeypatch, global_rank=0)

    collectives.init_process_group("tp", "127.0.0.1", 46001, 0, 2, "cuda:0", 0, 2, 0)
    collectives.init_process_group("moe_tp", "127.0.0.1", 46001, 0, 2, "cuda:0", 0, 2, 0)

    tcp_store.assert_called_once()
    assert tcp_store.call_args.args[:4] == ("127.0.0.1", 46001, 2, True)
    assert tcp_store.call_args.kwargs["wait_for_workers"] is False
    assert tcp_store.call_args.kwargs["multi_tenant"] is True

    # Every parallel group is a subgroup of one world, so the world rendezvous
    # happens once no matter how many groups the caller asks for.
    init_world.assert_called_once()
    assert init_world.call_args.kwargs["store"] is base_store
    assert init_world.call_args.kwargs["rank"] == 0
    assert init_world.call_args.kwargs["world_size"] == 2
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [
        [0, 1],
        [0, 1],
    ]


def test_native_runtime_bridge_bypasses_python_process_groups(monkeypatch):
    calls: list[str] = []

    runtime = SimpleNamespace(
        tp_all_reduce=lambda tensor: (calls.append("tp_reduce"), tensor.add_(1)),
        tp_all_gather=lambda tensor, dim: (
            calls.append(f"tp_gather:{dim}"),
            torch.cat((tensor, tensor), dim=dim),
        )[1],
        dp_all_gather=lambda tensor, counts: (
            calls.append(f"dp_gather:{counts}"),
            torch.cat((tensor, tensor), dim=0),
        )[1],
        moe_tp_all_reduce=lambda tensor: (
            calls.append("moe_tp_reduce"),
            tensor.add_(2),
        ),
        moe_ep_all_reduce=lambda tensor: (
            calls.append("moe_ep_reduce"),
            tensor.add_(4),
        ),
    )
    monkeypatch.setitem(sys.modules, "xllm_runtime", runtime)
    python_reduce = MagicMock(side_effect=AssertionError("c10d fallback used"))
    python_gather = MagicMock(side_effect=AssertionError("c10d fallback used"))
    monkeypatch.setattr(collectives, "all_reduce_", python_reduce)
    monkeypatch.setattr(collectives, "all_gather", python_gather)

    value = torch.tensor([[1.0]])
    collectives.tp_all_reduce(value)
    gathered = collectives.tp_all_gather(value, 1, 2)
    dp_gathered, dp_offset = collectives.gather_dp_execution_tokens(
        value,
        (1, 1),
        rank=1,
    )
    collectives.moe_tp_all_reduce(value)
    collectives.moe_ep_all_reduce(value)

    assert calls == [
        "tp_reduce",
        "tp_gather:1",
        "dp_gather:[1, 1]",
        "moe_tp_reduce",
        "moe_ep_reduce",
    ]
    assert gathered.tolist() == [[2.0, 2.0]]
    assert dp_gathered.tolist() == [[2.0], [2.0]]
    assert dp_offset == 1
    assert value.tolist() == [[8.0]]
    python_reduce.assert_not_called()
    python_gather.assert_not_called()


def _run_gloo_workers(worker: Callable[[int, str], None], nprocs: int, rendezvous_path: Path) -> None:
    process_context = torch.multiprocessing.start_processes(
        worker,
        args=(str(rendezvous_path),),
        nprocs=nprocs,
        join=False,
        start_method="fork",
    )
    deadline = time.monotonic() + 30.0
    try:
        while not process_context.join(
            timeout=max(0.0, deadline - time.monotonic()),
            grace_period=5.0,
        ):
            if time.monotonic() >= deadline:
                pytest.fail(f"{worker.__name__} Gloo collective test timed out")
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
        cleanup_deadline = time.monotonic() + 5.0
        for process in process_context.processes:
            process.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        for process in process_context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend is unavailable")
def test_glm_ep1_tp_reduce_does_not_mix_cp_cohorts(tmp_path: Path) -> None:
    _run_gloo_workers(_run_glm_ep1_tp_collective, 4, tmp_path / "glm-ep1-tp-reduce")


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend is unavailable")
def test_glm53_dp2_cp2_tp2_ep8_collectives_keep_dp_cohorts(tmp_path: Path) -> None:
    _run_gloo_workers(_run_glm53_dp_cp_ep_collectives, 8, tmp_path / "glm53-dp2-cp2-tp2-ep8")


def test_dcp_group_is_strided_like_kv_split_rank(monkeypatch):
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=2)

    # world=8, dcp=2 → group_count=4; membership matches rank/(world/dcp).
    collectives.init_process_group("dcp", "127.0.0.1", 46001, 0, 2, "cuda:0", 2, 8, 2)

    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]


@pytest.mark.parametrize(
    "world, cp, tp, expected",
    [
        (4, 2, 1, [[0, 1], [2, 3]]),
        (8, 2, 2, [[0, 2], [1, 3], [4, 6], [5, 7]]),
        (8, 4, 1, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        (4, 2, 2, [[0, 2], [1, 3]]),
    ],
)
def test_cp_memberships_preserve_dp_cohorts(world: int, cp: int, tp: int, expected: list[list[int]]) -> None:
    assert collectives._group_memberships("cp", cp, world, tp) == expected


@pytest.mark.parametrize("stride", [0, -1, 3])
def test_cp_memberships_reject_invalid_cohorts(stride: int) -> None:
    with pytest.raises(ValueError, match="complete DP cohorts"):
        collectives._group_memberships("cp", 2, 8, stride)


def test_cp_group_initialization_uses_tp_stride(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=3)
    group = collectives.init_process_group("cp", "127.0.0.1", 46001, 1, 2, "cuda:0", 3, 4, 1, 1)
    assert group.rank() == 1
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [[0, 1], [2, 3]]


def test_dcp_group_initialization_stays_within_dp(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=5)
    group = collectives.init_process_group("dcp", "127.0.0.1", 46001, 0, 2, "cuda:0", 5, 8, 3, 2)
    assert group.rank() == 0
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
    ]


@pytest.mark.parametrize("dim", [0, 1])
def test_all_gather_materializes_strided_activations(monkeypatch: pytest.MonkeyPatch, dim: int) -> None:
    value = torch.arange(24, dtype=torch.float32).reshape(4, 6).transpose(0, 1)
    assert not value.is_contiguous()
    group = _FakeGroup(0, 2)
    monkeypatch.setattr(collectives, "_require_group", lambda value, name: group)

    def gather(chunks: list[torch.Tensor], tensor: torch.Tensor, group: object) -> None:
        assert tensor.is_contiguous()
        assert all(chunk.is_contiguous() for chunk in chunks)
        torch.testing.assert_close(tensor, value)
        chunks[0].copy_(tensor)
        chunks[1].copy_(tensor + 100)

    def gather_rows(output: torch.Tensor, tensor: torch.Tensor, group: object) -> None:
        assert tensor.is_contiguous()
        assert output.is_contiguous()
        torch.testing.assert_close(tensor, value)
        output.copy_(torch.cat((tensor, tensor + 100), dim=0))

    monkeypatch.setattr(dist, "all_gather", gather)
    monkeypatch.setattr(dist, "all_gather_into_tensor", gather_rows)
    actual = collectives.all_gather(value, dim, 2, "cp")
    torch.testing.assert_close(actual, torch.cat((value, value + 100), dim=dim))
    assert not value.is_contiguous()


@pytest.mark.parametrize("dim", [0, 1])
def test_all_gather_keeps_contiguous_input_buffer(monkeypatch: pytest.MonkeyPatch, dim: int) -> None:
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    group = _FakeGroup(0, 2)
    monkeypatch.setattr(collectives, "_require_group", lambda tensor, name: group)

    def gather(chunks: list[torch.Tensor], tensor: torch.Tensor, group: object) -> None:
        assert tensor.data_ptr() == value.data_ptr()
        for index, chunk in enumerate(chunks):
            chunk.copy_(tensor + 100 * index)

    def gather_rows(output: torch.Tensor, tensor: torch.Tensor, group: object) -> None:
        assert tensor.data_ptr() == value.data_ptr()
        output.copy_(torch.cat((tensor, tensor + 100)))

    monkeypatch.setattr(dist, "all_gather", gather)
    monkeypatch.setattr(dist, "all_gather_into_tensor", gather_rows)
    result = collectives.all_gather(value, dim, 2, "cp")
    torch.testing.assert_close(result, torch.cat((value, value + 100), dim=dim))


@pytest.mark.parametrize("global_rank", range(8))
def test_dcp_group_consumes_upstream_qwen_memberships(monkeypatch: pytest.MonkeyPatch, global_rank: int) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=global_rank)
    memberships = [[0, 1], [2, 3], [4, 5], [6, 7]]
    rank, group_index = global_rank % 2, global_rank // 2

    # The C++ bridge passes the new optional argument positionally.
    group = collectives.init_process_group(
        "dcp",
        "127.0.0.1",
        46001,
        rank,
        2,
        "cuda:0",
        global_rank,
        8,
        group_index,
        group_ranks=memberships,
    )

    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == memberships
    own_ranks = collectives._group_ranks[("dcp", "cuda:0")]
    assert own_ranks[group.rank()] == global_rank
    assert group.rank() == global_rank % 2
    assert collectives.dcp_group("cuda:0") is group


@pytest.mark.parametrize("group_name", ["tp", "dp", "moe_tp", "moe_ep", "cp", "layerwise", "dcp"])
@pytest.mark.parametrize("pass_none", [False, True])
def test_default_topologies_remain_compatible(
    monkeypatch: pytest.MonkeyPatch, group_name: str, pass_none: bool
) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=3)
    contiguous = group_name in ("tp", "moe_tp", "layerwise")
    rank, group_index = (1, 1) if contiguous else (0, 3)
    kwargs = {"group_ranks": None} if pass_none else {}
    groups = [
        collectives.init_process_group(group_name, "127.0.0.1", 46001, rank, 2, "cuda:0", 3, 8, group_index, **kwargs)
        for _ in range(2)
    ]

    assert groups[0] is groups[1]
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == (
        [[0, 1], [2, 3], [4, 5], [6, 7]] if contiguous else [[0, 4], [1, 5], [2, 6], [3, 7]]
    )


@pytest.mark.parametrize("group_name", ["tp", "dp", "moe_tp", "moe_ep", "cp", "layerwise", "dcp"])
def test_explicit_topology_overrides_defaults_and_preserves_group_order(
    monkeypatch: pytest.MonkeyPatch, group_name: str
) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=3)
    # Neither built-in layout has this group order. Consume it without deriving.
    memberships = [[2, 3], [0, 1], [6, 7], [4, 5]]
    derive_memberships = MagicMock(side_effect=AssertionError("topology was re-derived"))
    monkeypatch.setattr(collectives, "_group_memberships", derive_memberships)

    groups = [
        collectives.init_process_group(group_name, "127.0.0.1", 46001, 1, 2, "cuda:0", 3, 8, 0, group_ranks=memberships)
        for _ in range(2)
    ]

    assert groups[0] is groups[1]
    assert groups[0].rank() == 1
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == memberships
    assert collectives._group_ranks[(group_name, "cuda:0")] == (2, 3)
    derive_memberships.assert_not_called()


def test_tp_accepts_explicit_strided_memberships(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=5)
    memberships = [[0, 4], [1, 5], [2, 6], [3, 7]]

    group = collectives.init_process_group("tp", "127.0.0.1", 46001, 1, 2, "cuda:0", 5, 8, 1, group_ranks=memberships)

    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == memberships
    assert group.rank() == 1
    assert collectives._group_ranks[("tp", "cuda:0")] == (1, 5)


def test_dcp_groups_preserve_upstream_data_parallel_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=10)
    memberships = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]

    collectives.init_process_group("dcp", "127.0.0.1", 46001, 0, 2, "cuda:0", 10, 16, 5, group_ranks=memberships)

    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == memberships
    assert collectives._group_ranks[("dcp", "cuda:0")] == (10, 11)


def test_explicit_topology_does_not_affect_other_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=2)
    collectives.init_process_group(
        "dcp",
        "127.0.0.1",
        46001,
        0,
        2,
        "cuda:0",
        2,
        8,
        1,
        group_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
    )
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    new_group.reset_mock()

    # Another group and another device still use the original default rules.
    collectives.init_process_group("dp", "127.0.0.1", 46001, 0, 2, "cuda:0", 2, 8, 2)
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    new_group.reset_mock()
    collectives.init_process_group("dcp", "127.0.0.1", 46001, 0, 2, "cuda:1", 2, 8, 2)
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [[0, 4], [1, 5], [2, 6], [3, 7]]


@pytest.mark.parametrize("group_name", ["tp", "dp", "moe_tp", "moe_ep", "cp", "layerwise", "dcp"])
@pytest.mark.parametrize("explicit_topology_first", [False, True])
def test_rejects_different_members_with_same_rank_and_size(
    monkeypatch: pytest.MonkeyPatch, group_name: str, explicit_topology_first: bool
) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=0)
    # Choose an explicit layout different from this group type's default.
    explicit_memberships = (
        [[0, 4], [1, 5], [2, 6], [3, 7]]
        if group_name in ("tp", "moe_tp", "layerwise")
        else [[0, 1], [2, 3], [4, 5], [6, 7]]
    )
    first_memberships = explicit_memberships if explicit_topology_first else None
    second_memberships = None if explicit_topology_first else explicit_memberships
    collectives.init_process_group(
        group_name, "127.0.0.1", 46001, 0, 2, "cuda:0", 0, 8, 0, group_ranks=first_memberships
    )
    new_group.reset_mock()

    # Both layouts give rank 0 local rank 0 and size 2; only the peers differ.
    with pytest.raises(RuntimeError, match="different members"):
        collectives.init_process_group(
            group_name, "127.0.0.1", 46001, 0, 2, "cuda:0", 0, 8, 0, group_ranks=second_memberships
        )
    new_group.assert_not_called()


@pytest.mark.parametrize(
    ("rank", "group_index", "world_size", "global_world_size"),
    [(0, 1, 2, 8), (1, 0, 2, 8), (-1, 1, 2, 8), (1, -1, 2, 8), (1, 1, 4, 8), (1, 1, 2, 16)],
)
def test_rejects_inconsistent_explicit_topology_before_rendezvous(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    group_index: int,
    world_size: int,
    global_world_size: int,
) -> None:
    _, tcp_store, init_world, new_group = _mock_process_groups(monkeypatch, global_rank=3)

    with pytest.raises((ValueError, RuntimeError), match="dcp topology"):
        collectives.init_process_group(
            "dcp",
            "127.0.0.1",
            46001,
            rank,
            world_size,
            "cuda:0",
            3,
            global_world_size,
            group_index,
            group_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
        )
    tcp_store.assert_not_called()
    init_world.assert_not_called()
    new_group.assert_not_called()


def test_explicit_topology_creation_failure_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=3)
    new_group.side_effect = RuntimeError("collective creation failed")
    derive_memberships = MagicMock(side_effect=AssertionError("default topology used"))
    monkeypatch.setattr(collectives, "_group_memberships", derive_memberships)

    with pytest.raises(RuntimeError, match="collective creation failed"):
        collectives.init_process_group(
            "dcp",
            "127.0.0.1",
            46001,
            1,
            2,
            "cuda:0",
            3,
            8,
            1,
            group_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
        )
    assert collectives.dcp_group("cuda:0") is None
    assert new_group.call_count == 1
    derive_memberships.assert_not_called()


@pytest.mark.parametrize(
    "memberships",
    [[], [[]], [[0, 1], [2]], [[0, 1], [1, 2]], [[0, 1], [2, 4]], [[-1, 0], [1, 2]], [[1, 0], [2, 3]]],
)
def test_rejects_invalid_memberships_before_rendezvous(
    monkeypatch: pytest.MonkeyPatch, memberships: list[list[int]]
) -> None:
    _, tcp_store, init_world, new_group = _mock_process_groups(monkeypatch, global_rank=3)
    with pytest.raises(ValueError, match="dcp topology"):
        collectives.init_process_group("dcp", "127.0.0.1", 46001, 1, 2, "cuda:0", 3, 4, 1, group_ranks=memberships)
    tcp_store.assert_not_called()
    init_world.assert_not_called()
    new_group.assert_not_called()


@pytest.mark.parametrize("invalid_rank", [True, "0", 0.0])
def test_rejects_noninteger_topology_ranks_before_rendezvous(
    monkeypatch: pytest.MonkeyPatch, invalid_rank: object
) -> None:
    _, tcp_store, init_world, new_group = _mock_process_groups(monkeypatch, global_rank=3)
    with pytest.raises(TypeError, match="ranks must be integers"):
        collectives.init_process_group(
            "dcp",
            "127.0.0.1",
            46001,
            1,
            2,
            "cuda:0",
            3,
            4,
            1,
            group_ranks=[[invalid_rank, 1], [2, 3]],
        )
    tcp_store.assert_not_called()
    init_world.assert_not_called()
    new_group.assert_not_called()


def test_explicit_topology_keeps_cached_members_independent_of_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_process_groups(monkeypatch, global_rank=3)
    memberships = [[0, 1], [2, 3]]
    group = collectives.init_process_group("dcp", "127.0.0.1", 46001, 1, 2, "cuda:0", 3, 4, 1, group_ranks=memberships)
    assert memberships == [[0, 1], [2, 3]]
    memberships[1][:] = [0, 1]
    memberships.clear()

    assert group.rank() == 1
    assert collectives._group_ranks[("dcp", "cuda:0")] == (2, 3)


def test_dp_execution_gather_uses_fixed_collective_for_equal_counts(monkeypatch):
    value = torch.zeros(4, 8)
    gathered = torch.zeros(8, 8)
    fixed_gather = MagicMock(return_value=gathered)
    variable_gather = MagicMock()
    monkeypatch.setattr(collectives, "all_gather", fixed_gather)
    monkeypatch.setattr(collectives, "all_gather_variable", variable_gather)

    output, offset = collectives.gather_dp_execution_tokens(
        value,
        (4, 4),
        rank=1,
    )

    assert output is gathered
    assert offset == 4
    fixed_gather.assert_called_once_with(
        value,
        dim=0,
        world_size=2,
        group_name="dp",
    )
    variable_gather.assert_not_called()


def test_dp_execution_gather_uses_variable_collective_for_uneven_counts(monkeypatch):
    value = torch.zeros(1, 8)
    gathered = torch.zeros(4, 8)
    fixed_gather = MagicMock()
    variable_gather = MagicMock(return_value=gathered)
    monkeypatch.setattr(collectives, "all_gather", fixed_gather)
    monkeypatch.setattr(collectives, "all_gather_variable", variable_gather)

    output, offset = collectives.gather_dp_execution_tokens(
        value,
        (3, 1),
        rank=1,
    )

    assert output is gathered
    assert offset == 3
    variable_gather.assert_called_once_with(value, [3, 1], 1, "dp")
    fixed_gather.assert_not_called()


def test_dp_execution_gather_rejects_local_shape_mismatch() -> None:
    with pytest.raises(RuntimeError, match="does not match the local tensor"):
        collectives.gather_dp_execution_tokens(
            torch.zeros(1, 8),
            (3, 2),
            rank=1,
        )


def test_dp_execution_gather_rejects_zero_execution_count() -> None:
    with pytest.raises(RuntimeError, match="must be positive"):
        collectives.gather_dp_execution_tokens(
            torch.zeros(1, 8),
            (3, 0),
            rank=1,
        )


def test_tcp_store_master_is_global_rank_zero_not_group_rank_zero(monkeypatch):
    _, tcp_store, _, _ = _mock_process_groups(monkeypatch, global_rank=2)

    collectives.init_process_group("tp", "127.0.0.1", 46001, 0, 2, "cuda:0", 2, 4, 1)

    assert tcp_store.call_args.args[:4] == ("127.0.0.1", 46001, 4, False)


def test_symmetric_memory_rejects_cross_host_group(monkeypatch):
    collectives._world_topology = [
        {"hostname": "node-0", "device_index": 0},
        {"hostname": "node-1", "device_index": 0},
    ]
    can_access_peer = MagicMock(return_value=True)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", can_access_peer)

    assert not collectives._supports_symmetric_memory(torch.device("cuda:0"), [0, 1])
    can_access_peer.assert_not_called()


def test_symmetric_memory_rejects_incomplete_peer_domain(monkeypatch):
    collectives._world_topology = [
        {"hostname": "node-0", "device_index": 0},
        {"hostname": "node-0", "device_index": 1},
    ]
    monkeypatch.setattr(
        torch.cuda,
        "can_device_access_peer",
        lambda source, destination: (source, destination) != (1, 0),
    )

    assert not collectives._supports_symmetric_memory(torch.device("cuda:0"), [0, 1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
def test_symmetric_buffer_rejects_unsupported_dtype(monkeypatch, dtype):
    group_name = "tp"
    device = torch.device("cuda:0")
    collectives._symm_eligible[(group_name, str(device))] = True
    tensor = MagicMock()
    tensor.device = device
    tensor.dtype = dtype
    tensor.is_contiguous.return_value = True
    tensor.numel.return_value = 8
    tensor.element_size.return_value = torch.empty((), dtype=dtype).element_size()
    empty = MagicMock()
    rendezvous = MagicMock()
    monkeypatch.setattr(collectives.symm_mem, "empty", empty)
    monkeypatch.setattr(collectives.symm_mem, "rendezvous", rendezvous)

    assert collectives._symm_buffer(_FakeGroup(0, 2), group_name, tensor) is None
    empty.assert_not_called()
    rendezvous.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_symmetric_buffer_accepts_supported_dtype(monkeypatch, dtype):
    group_name = "tp"
    device = torch.device("cuda:0")
    collectives._symm_eligible[(group_name, str(device))] = True
    tensor = MagicMock()
    tensor.device = device
    tensor.dtype = dtype
    tensor.is_contiguous.return_value = True
    tensor.numel.return_value = 8
    tensor.element_size.return_value = torch.empty((), dtype=dtype).element_size()
    buffer = object()
    group = _FakeGroup(0, 2)
    group.group_name = "tp-group"
    empty = MagicMock(return_value=buffer)
    rendezvous = MagicMock()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(collectives.symm_mem, "empty", empty)
    monkeypatch.setattr(collectives.symm_mem, "rendezvous", rendezvous)

    assert collectives._symm_buffer(group, group_name, tensor) is buffer
    empty.assert_called_once_with(8, dtype=dtype, device=device)
    rendezvous.assert_called_once_with(buffer, "tp-group")


def test_row_all_gather_uses_one_contiguous_output(monkeypatch: pytest.MonkeyPatch) -> None:
    group = _FakeGroup(0, 2)
    collectives._groups[("tp", "cpu")] = group
    value = torch.arange(24, dtype=torch.float64).reshape(4, 6)[:, ::2]
    calls = []

    def gather(output: torch.Tensor, source: torch.Tensor, *, group: object) -> None:
        assert source.is_contiguous()
        calls.append(output.data_ptr())
        output.copy_(torch.cat((source, source + 1)))

    monkeypatch.setattr(dist, "all_gather_into_tensor", gather)
    old = MagicMock(side_effect=AssertionError("row gather must not allocate a tensor list"))
    monkeypatch.setattr(dist, "all_gather", old)
    result = collectives.all_gather(value, 0, 2)
    torch.testing.assert_close(result, torch.cat((value, value + 1)))
    assert len(calls) == 1 and result.is_contiguous()
    old.assert_not_called()


def test_reduce_scatter_uses_sum_and_local_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    group = _FakeGroup(1, 2)
    collectives._groups[("moe_ep", "cpu")] = group
    value = torch.arange(48, dtype=torch.float32).reshape(6, 8)[:, ::2]
    observed = []

    def reduce(output: torch.Tensor, source: torch.Tensor, *, group: object) -> None:
        assert source.is_contiguous()
        observed.append(source.shape)
        output.copy_((source * 3).chunk(2)[1])

    monkeypatch.setattr(dist, "reduce_scatter_tensor", reduce)
    result = collectives.reduce_scatter(value, 2, "moe_ep")
    torch.testing.assert_close(result, value[3:] * 3)
    assert observed == [value.shape]
    assert result.shape == collectives._reduce_scatter_fake(value, 2).shape


@pytest.mark.parametrize("rows,world", [(3, 2), (0, 2), (4, 0), (4, -1)])
def test_reduce_scatter_rejects_invalid_row_contract(rows: int, world: int) -> None:
    with pytest.raises(ValueError, match="nonempty input rows"):
        collectives.reduce_scatter(torch.zeros(rows, 4), world)


def test_reduce_scatter_one_rank_does_not_alias_input() -> None:
    value = torch.arange(12).reshape(3, 4)
    actual = collectives.reduce_scatter(value, 1)
    torch.testing.assert_close(actual, value)
    assert actual.data_ptr() != value.data_ptr()
