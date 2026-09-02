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

"""Process-group topology for Python-composed xLLM models."""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

_GROUP_TIMEOUT = timedelta(minutes=5)
_contexts: dict[str, _ParallelContext] = {}
_contexts_lock = threading.Lock()


@dataclass(frozen=True)
class ParallelGroup:
    """A named communication domain containing the current global rank."""

    name: str
    ranks: tuple[int, ...]
    local_rank: int
    process_group: Any | None

    @property
    def world_size(self) -> int:
        return len(self.ranks)


@dataclass(frozen=True)
class _GroupSpec:
    name: str
    ranks: tuple[int, ...]
    local_rank: int
    alias_of: str | None
    group_id: str


@dataclass(frozen=True)
class _ParallelTopology:
    host: str
    port: int
    rank: int
    world_size: int
    device: str
    group_specs: tuple[_GroupSpec, ...]


@dataclass
class _ParallelContext:
    topology: _ParallelTopology
    store: Any | None
    groups: dict[str, ParallelGroup]


def _normalize_group_specs(
    group_specs: list[dict[str, Any]],
    rank: int,
    world_size: int,
) -> tuple[_GroupSpec, ...]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} must be in [0, {world_size})")

    normalized: list[_GroupSpec] = []
    names: set[str] = set()
    group_ids: set[str] = set()
    for raw_spec in group_specs:
        name = str(raw_spec["name"])
        if not name or name in names:
            raise ValueError(f"duplicate or empty process-group name: {name}")
        names.add(name)

        ranks = tuple(int(group_rank) for group_rank in raw_spec["ranks"])
        if not ranks or len(set(ranks)) != len(ranks):
            raise ValueError(f"process group {name} has invalid ranks: {ranks}")
        if any(group_rank < 0 or group_rank >= world_size for group_rank in ranks):
            raise ValueError(f"process group {name} has rank outside world size: {ranks}")
        if rank not in ranks:
            raise ValueError(f"global rank {rank} is not in process group {name}")

        local_rank = int(raw_spec["local_rank"])
        if local_rank != ranks.index(rank):
            raise ValueError(
                f"process group {name} local rank mismatch: got {local_rank}, expected {ranks.index(rank)}"
            )

        raw_alias = raw_spec.get("alias_of")
        alias_of = None if raw_alias is None else str(raw_alias)
        group_id = str(raw_spec.get("group_id", ""))
        if alias_of is None:
            if not group_id or group_id in group_ids:
                raise ValueError(f"process group {name} has duplicate or empty group_id: {group_id}")
            group_ids.add(group_id)

        normalized.append(
            _GroupSpec(
                name=name,
                ranks=ranks,
                local_rank=local_rank,
                alias_of=alias_of,
                group_id=group_id,
            )
        )

    specs_by_name = {spec.name: spec for spec in normalized}
    world = specs_by_name.get("world")
    if world is None or world.ranks != tuple(range(world_size)):
        raise ValueError("C++ process-group specs must contain the full world group")

    for spec in normalized:
        if spec.alias_of is None:
            continue
        target = specs_by_name.get(spec.alias_of)
        if target is None:
            raise ValueError(f"process group {spec.name} aliases unknown group {spec.alias_of}")
        if spec.ranks != target.ranks or spec.local_rank != target.local_rank:
            raise ValueError(f"process group alias {spec.name} does not match {spec.alias_of}")
    return tuple(normalized)


def _create_process_group(
    store: Any,
    spec: _GroupSpec,
    device: torch.device,
) -> Any | None:
    if len(spec.ranks) == 1:
        return None

    prefix = f"xllm_python/{spec.group_id}/" + ",".join(str(rank) for rank in spec.ranks)
    group_store = dist.PrefixStore(prefix, store)
    if device.type == "cuda":
        return dist.ProcessGroupNCCL(
            group_store,
            spec.local_rank,
            len(spec.ranks),
            _GROUP_TIMEOUT,
        )

    if device.type != "npu":
        raise ValueError(f"unsupported distributed device: {device}")

    import torch_npu  # noqa: F401
    from torch_npu._C._distributed_c10d import ProcessGroupHCCL

    options = ProcessGroupHCCL.Options()
    options.group_id = spec.group_id
    options.global_ranks_in_group = list(spec.ranks)
    options._timeout = _GROUP_TIMEOUT
    pg = ProcessGroupHCCL(
        group_store,
        spec.local_rank,
        len(spec.ranks),
        options,
    )
    return pg


def _make_group(
    spec: _GroupSpec,
    process_group: Any | None,
) -> ParallelGroup:
    return ParallelGroup(
        name=spec.name,
        ranks=spec.ranks,
        local_rank=spec.local_rank,
        process_group=process_group,
    )


def init_parallel_groups(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    device: str,
    group_specs: list[dict[str, Any]],
) -> dict[str, ParallelGroup]:
    """Create the process groups described by the C++ parallel runtime."""

    device_obj = torch.device(device)
    device_key = str(device_obj)
    normalized_specs = _normalize_group_specs(group_specs, rank, world_size)
    topology = _ParallelTopology(
        host=host,
        port=port,
        rank=rank,
        world_size=world_size,
        device=device_key,
        group_specs=normalized_specs,
    )

    # Thread-safe access to global context cache
    with _contexts_lock:
        context = _contexts.get(device_key)
        if context is not None:
            if context.topology != topology:
                raise RuntimeError(
                    f"parallel groups for {device_key} are already initialized "
                    f"with {context.topology}, requested {topology}"
                )
            return context.groups

    store = None
    if world_size > 1:
        os.environ.pop("RANK_TABLE_FILE", None)
        store = dist.TCPStore(
            host,
            port,
            -1,
            rank == 0,
            _GROUP_TIMEOUT,
            wait_for_workers=False,
        )

    groups: dict[str, ParallelGroup] = {}
    for spec in normalized_specs:
        if spec.alias_of is not None:
            continue
        process_group = None if store is None else _create_process_group(store, spec, device_obj)
        groups[spec.name] = _make_group(spec, process_group)

    unresolved_aliases = [spec for spec in normalized_specs if spec.alias_of is not None]
    while unresolved_aliases:
        remaining: list[_GroupSpec] = []
        for spec in unresolved_aliases:
            target = groups.get(spec.alias_of or "")
            if target is None:
                remaining.append(spec)
                continue
            groups[spec.name] = ParallelGroup(
                name=spec.name,
                ranks=spec.ranks,
                local_rank=spec.local_rank,
                process_group=target.process_group,
            )
        if len(remaining) == len(unresolved_aliases):
            aliases = {spec.name: spec.alias_of for spec in remaining}
            raise ValueError(f"cyclic process-group aliases: {aliases}")
        unresolved_aliases = remaining

    with _contexts_lock:
        _contexts[device_key] = _ParallelContext(
            topology=topology,
            store=store,
            groups=groups,
        )

    # Bridge into the collectives registry so the xllm_ops::* custom ops
    # (all_gather / all_reduce_ / ...) used by the shared Python layers resolve
    # the same c10d process groups. Both registries are keyed by
    # (group_name, str(device)); publishing the groups here instead of
    # re-creating communicators keeps the kimi parallel_state init and the
    # coding-main collectives custom ops consistent.
    from xllm.python.distributed import collectives as _collectives

    for _name, _pg in groups.items():
        if _pg.process_group is None:
            continue
        _gkey = (_name, device_key)
        if _gkey not in _collectives._groups:
            _collectives._groups[_gkey] = _pg.process_group
            _collectives._symm_eligible[_gkey] = _collectives._supports_symmetric_memory(device_obj, _pg.ranks)

    return groups


def get_parallel_group(name: str, device: object) -> ParallelGroup:
    device_key = str(torch.device(device))
    context = _contexts.get(device_key)
    if context is None:
        raise RuntimeError(f"parallel groups were not initialized for device {device_key}")
    group = context.groups.get(name)
    if group is None:
        raise KeyError(f"unknown parallel group: {name}")
    return group


def parallel_group_rank(name: str, device: object) -> int:
    return get_parallel_group(name, device).local_rank


def parallel_group_world_size(name: str, device: object) -> int:
    return get_parallel_group(name, device).world_size


__all__ = [
    "ParallelGroup",
    "get_parallel_group",
    "init_parallel_groups",
    "parallel_group_rank",
    "parallel_group_world_size",
]
