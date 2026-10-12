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

"""Tensor-parallel process groups and collectives.

The collectives themselves are hardware-neutral: only the ProcessGroup backend
differs (NCCL on CUDA, HCCL on NPU), and torch picks it from the device. They
are graph operators so that a compiled graph keeps the communication node in
place instead of splitting around it.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Sequence
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed import ProcessGroup

_GROUP_NAMES = frozenset(("tp", "dp", "moe_tp", "moe_ep", "cp", "layerwise", "dcp"))
# ``tp`` and ``moe_tp`` own a contiguous block of global ranks, while ``dp``,
# ``moe_ep``, ``cp`` and ``dcp`` stride across those blocks. Context groups also
# need the within-DP stride when their world contains multiple DP cohorts.
# ``cp`` strides by the attention TP size: ranks sharing a (dp, tp) slot but
# holding different sequence shards form one CP group, matching the C++
# compute_cp_group_ranks layout (rank = dp*cp*tp + cp_rank*tp + tp_rank).
# CP and legacy DCP callers with DP pass an explicit within-cohort stride.
# Explicit memberships can override these defaults when the upstream runtime
# has already finalized a model-specific topology.
_CONTIGUOUS_GROUPS = frozenset(("tp", "moe_tp", "layerwise"))

_groups = {}
_group_ranks = {}
_stores = {}
_world_topology = None
_symm_eligible = {}
_symm_buffers = {}
_world_initialized = False


def _backend_for(device: torch.device) -> str:
    if device.type == "cuda":
        return "nccl"
    import torch_npu  # noqa: F401

    return "hccl"


def _shared_store(host: str, port: int, global_rank: int, global_world_size: int) -> dist.Store:
    store_key = (host, port)
    store = _stores.get(store_key)
    if store is None:
        store = dist.TCPStore(
            host,
            port,
            global_world_size,
            global_rank == 0,
            timedelta(minutes=5),
            wait_for_workers=False,
            multi_tenant=True,
        )
        _stores[store_key] = store
    return store


def _exchange_world_topology(
    store: dist.Store,
    device: torch.device,
    global_rank: int,
    global_world_size: int,
) -> list[dict[str, object]]:
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    local = {"hostname": socket.gethostname(), "device_index": device_index}
    key_prefix = "xllm/python_collectives/topology/v1"
    store.set(f"{key_prefix}/{global_rank}", json.dumps(local))
    return [json.loads(store.get(f"{key_prefix}/{rank}").decode("utf-8")) for rank in range(global_world_size)]


def _ensure_world(
    host: str,
    port: int,
    device: torch.device,
    global_rank: int,
    global_world_size: int,
) -> None:
    """Initialize the default process group covering every rank.

    The parallel groups are subgroups of it. Building them with ``new_group``
    instead of constructing communicators directly registers them with c10d,
    which is what lets a group back a symmetric-memory allocation.
    """
    global _world_initialized, _world_topology
    if _world_initialized:
        return
    store = _shared_store(host, port, global_rank, global_world_size)
    dist.init_process_group(
        backend=_backend_for(device),
        store=store,
        rank=global_rank,
        world_size=global_world_size,
        timeout=timedelta(minutes=5),
    )
    _world_topology = _exchange_world_topology(store, device, global_rank, global_world_size)
    _world_initialized = True


def _group_memberships(
    group_name: str,
    world_size: int,
    global_world_size: int,
    group_stride: int | None = None,
) -> list[list[int]]:
    """Every group of this kind, in an order all ranks agree on.

    ``new_group`` is collective over the whole world, so each rank has to create
    all groups of a kind in the same order, not only the one it belongs to.
    """
    if world_size <= 0 or global_world_size % world_size:
        raise ValueError(f"{group_name} size {world_size} does not divide the world size {global_world_size}")
    if group_stride is not None:
        if group_name not in ("cp", "dcp") or group_stride <= 0 or global_world_size % (world_size * group_stride):
            raise ValueError("CP/DCP group stride must describe complete DP cohorts")
        # Context parallelism varies within one DP cohort; a global stride instead mixes
        # different requests whenever DP > 1.
        cohort_size = world_size * group_stride
        return [
            [base + tp_rank + cp_rank * group_stride for cp_rank in range(world_size)]
            for base in range(0, global_world_size, cohort_size)
            for tp_rank in range(group_stride)
        ]
    count = global_world_size // world_size
    if group_name in _CONTIGUOUS_GROUPS:
        return [[index * world_size + offset for offset in range(world_size)] for index in range(count)]
    return [[index + offset * count for offset in range(world_size)] for index in range(count)]


def _validate_group_topology(
    group_name: str,
    memberships: list[list[int]],
    rank: int,
    world_size: int,
    global_rank: int,
    global_world_size: int,
    group_index: int,
    *,
    explicit_memberships: bool,
) -> None:
    """Validate topology and caller coordinates before creating or reusing a group."""
    if (
        global_world_size <= 0
        or world_size <= 0
        or len(memberships) * world_size != global_world_size
        or any(len(ranks) != world_size for ranks in memberships)
    ):
        raise ValueError(f"{group_name} topology sizes do not match the process-group initialization")
    if explicit_memberships:
        all_ranks = [member for ranks in memberships for member in ranks]
        if any(not isinstance(member, int) or isinstance(member, bool) for member in all_ranks):
            raise TypeError(f"{group_name} topology ranks must be integers")
        if sorted(all_ranks) != list(range(global_world_size)):
            raise ValueError(f"{group_name} topology must partition all global ranks exactly once")
        # c10d new_group sorts its rank list; reject a conflicting local-rank
        # order instead of silently changing the upstream topology.
        if any(ranks != sorted(ranks) for ranks in memberships):
            raise ValueError(f"{group_name} topology members within each group must be sorted")
    if (
        not 0 <= group_index < len(memberships)
        or not 0 <= rank < world_size
        or memberships[group_index][rank] != global_rank
    ):
        raise RuntimeError(f"{group_name} topology rank or group index does not match the caller")


def _supports_symmetric_memory(device: torch.device, ranks: list[int]) -> bool:
    if device.type != "cuda" or _world_topology is None:
        return False
    topology = [_world_topology[rank] for rank in ranks]
    hostnames = {entry["hostname"] for entry in topology}
    if len(hostnames) != 1:
        return False
    device_indices = [entry["device_index"] for entry in topology]
    if any(not isinstance(index, int) or index < 0 for index in device_indices):
        return False
    if len(set(device_indices)) != len(device_indices):
        return False
    return all(
        torch.cuda.can_device_access_peer(source, destination)
        for source in device_indices
        for destination in device_indices
        if source != destination
    )


def init_process_group(
    group_name: str,
    host: str,
    port: int,
    rank: int,
    world_size: int,
    device: str,
    global_rank: int,
    global_world_size: int,
    group_index: int,
    group_stride: int | None = None,
    group_ranks: Sequence[Sequence[int]] | None = None,
) -> ProcessGroup:
    """Create a parallel group from explicit memberships or the legacy layout.

    ``group_ranks`` lists every group of this kind in the same order on all
    processes, not just the current process's peers. Each sorted inner list
    contains global ranks; ``group_index`` selects our list and ``rank`` our
    position in it. None preserves the default layout for ``group_name``.
    """
    if group_name not in _GROUP_NAMES:
        raise ValueError(f"unsupported parallel group: {group_name}")
    device_obj = torch.device(device)
    group_key = (group_name, str(device_obj))
    if group_stride is not None and group_ranks is not None:
        raise ValueError("group_stride and group_ranks are mutually exclusive")
    if group_ranks is not None:
        memberships = [list(ranks) for ranks in group_ranks]
    else:
        memberships = _group_memberships(group_name, world_size, global_world_size, group_stride)
    _validate_group_topology(
        group_name,
        memberships,
        rank,
        world_size,
        global_rank,
        global_world_size,
        group_index,
        explicit_memberships=group_ranks is not None,
    )
    group = _groups.get(group_key)
    if group is not None:
        if group.rank() != rank or group.size() != world_size:
            raise RuntimeError(
                f"{group_name} group for {group_key[1]} is already initialized as "
                f"rank {group.rank()}/{group.size()}, requested "
                f"rank {rank}/{world_size}"
            )
        if _group_ranks.get(group_key) != tuple(memberships[group_index]):
            raise RuntimeError(f"{group_name} group for {group_key[1]} is already initialized with different members")
        return group

    _ensure_world(host, port, device_obj, global_rank, global_world_size)
    backend = _backend_for(device_obj)

    own = None
    own_ranks = None
    for index, ranks in enumerate(memberships):
        candidate = dist.new_group(ranks=ranks, timeout=timedelta(minutes=5), backend=backend)
        if global_rank in ranks:
            own = candidate
            own_index = index
            own_ranks = ranks
    if own is None or own.rank() != rank:
        raise RuntimeError(
            f"derived {group_name} membership disagrees with the caller: global "
            f"rank {global_rank} of {global_world_size} expected rank {rank} of "
            f"{world_size}, got {None if own is None else own.rank()}"
        )
    if own_index != group_index:
        raise RuntimeError(f"derived {group_name} group index {own_index} does not match the caller's {group_index}")

    assert own_ranks is not None
    _groups[group_key] = own
    _group_ranks[group_key] = tuple(own_ranks)
    _symm_eligible[group_key] = _supports_symmetric_memory(device_obj, own_ranks)
    return own


def init_tp_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    device: str,
    global_rank: int,
    global_world_size: int,
    group_index: int,
) -> ProcessGroup:
    return init_process_group(
        "tp",
        host,
        port,
        rank,
        world_size,
        device,
        global_rank,
        global_world_size,
        group_index,
    )


def _require_group(x: torch.Tensor, group_name: str) -> ProcessGroup:
    group = _groups.get((group_name, str(x.device)))
    if group is None:
        raise RuntimeError(f"{group_name} collective called before its process group was initialized for {x.device}")
    return group


def tp_rank(device: torch.device | str) -> int:
    """Rank in the TP group for ``device`` (0 when no TP group exists)."""
    group = _groups.get(("tp", str(torch.device(device))))
    return group.rank() if group is not None else 0


def cp_rank(device: torch.device | str) -> int:
    """Rank in the CP group for ``device`` (0 when no CP group exists)."""
    group = _groups.get(("cp", str(torch.device(device))))
    return group.rank() if group is not None else 0


def cp_world_size(device: torch.device | str) -> int:
    """Size of the CP group for ``device`` (1 when no CP group exists)."""
    group = _groups.get(("cp", str(torch.device(device))))
    return group.size() if group is not None else 1


def layerwise_rank(device: torch.device | str) -> int:
    """Rank in the layerwise split group, or zero when disabled."""
    group = _groups.get(("layerwise", str(torch.device(device))))
    return group.rank() if group is not None else 0


def _native_runtime_op(name: str) -> object | None:
    """Return an embedded C++ collective when running under PyExecutorImpl."""
    try:
        import xllm_runtime
    except ImportError:
        return None
    return getattr(xllm_runtime, name, None)


def tp_all_reduce(x: torch.Tensor) -> None:
    op = _native_runtime_op("tp_all_reduce")
    if op is not None:
        op(x)
        return
    all_reduce_(x, "tp")


def tp_all_gather(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    op = _native_runtime_op("tp_all_gather")
    if op is not None:
        return op(x, dim)
    return all_gather(x, dim, world_size, "tp")


def dp_all_gather(
    x: torch.Tensor,
    token_counts: list[int],
    rank: int,
) -> torch.Tensor:
    op = _native_runtime_op("dp_all_gather")
    if op is not None:
        return op(x, token_counts)
    if all(count == token_counts[0] for count in token_counts):
        return all_gather(
            x,
            dim=0,
            world_size=len(token_counts),
            group_name="dp",
        )
    return all_gather_variable(x, token_counts, rank, "dp")


def moe_tp_all_reduce(x: torch.Tensor) -> None:
    op = _native_runtime_op("moe_tp_all_reduce")
    if op is not None:
        op(x)
        return
    all_reduce_(x, "moe_tp")


def moe_ep_all_reduce(x: torch.Tensor) -> None:
    op = _native_runtime_op("moe_ep_all_reduce")
    if op is not None:
        op(x)
        return
    all_reduce_(x, "moe_ep")


def moe_ep_hccl_info(device: torch.device | str) -> tuple[str, int, int]:
    """Return the initialized expert group's HCCL name, rank and size."""
    device_obj = torch.device(device)
    group = _groups.get(("moe_ep", str(device_obj)))
    if group is None:
        raise RuntimeError(f"MoE EP dispatch requires an initialized process group for {device_obj}")
    rank, world_size = group.rank(), group.size()
    if world_size <= 1 or not 0 <= rank < world_size:
        raise RuntimeError(f"MoE EP dispatch requires an initialized multi-rank group: rank {rank}/{world_size}")
    get_comm_name = getattr(group._get_backend(device_obj), "get_hccl_comm_name", None)
    if get_comm_name is None:
        raise RuntimeError("MoE EP dispatch requires an HCCL process group")
    comm_name = get_comm_name(rank)
    if not isinstance(comm_name, str) or not comm_name:
        raise RuntimeError("MoE EP dispatch requires a nonempty HCCL communication name")
    return comm_name, rank, world_size


def moe_ep_group(device: torch.device | str) -> ProcessGroup:
    """Return the existing EP group without creating a second communicator."""
    group = _groups.get(("moe_ep", str(torch.device(device))))
    if group is None:
        raise RuntimeError(f"MoE EP group is not initialized for {device}")
    return group


def dcp_group(device: torch.device | str) -> ProcessGroup | None:
    return _groups.get(("dcp", str(torch.device(device))))


# A one-shot symmetric-memory reduction is an ordinary kernel on the current
# stream, so a captured graph runs it inline. NCCL runs collectives on its own
# stream, which costs a fork/join per call -- measured at ~32us of device idle
# before every all-reduce in the decode graph, and there are dozens per step.
# Past this size the staging copy stops paying for itself and NCCL's ring wins,
# and the bound also keeps the buffer cache to decode-sized shapes rather than
# one entry per distinct prefill length.
_SYMM_MEM_MAX_BYTES = 512 * 1024
# Distinct shapes still accumulate one buffer each; stop allocating rather than
# grow without limit when a workload sweeps many small shapes.
_SYMM_MEM_MAX_BUFFERS = 64
_SYMM_MEM_DTYPES = frozenset((torch.float32, torch.bfloat16))


def _symm_buffer(group: ProcessGroup, group_name: str, x: torch.Tensor) -> torch.Tensor | None:
    """Return a symmetric-memory staging buffer for ``x``, or None.

    Allocation and rendezvous are collective and cannot run inside a graph
    capture, so a shape first seen during capture falls back to NCCL. The
    capture path is warmed up eagerly beforehand, which is where the decode
    shapes get their buffers.
    """
    group_key = (group_name, str(x.device))
    if not _symm_eligible.get(group_key, False):
        return None
    if not x.is_contiguous() or x.dtype not in _SYMM_MEM_DTYPES:
        return None
    if x.numel() * x.element_size() > _SYMM_MEM_MAX_BYTES:
        return None

    key = (group_name, str(x.device), x.dtype, x.numel())
    buffer = _symm_buffers.get(key)
    if buffer is not None:
        return buffer
    if torch.cuda.is_current_stream_capturing():
        return None
    if len(_symm_buffers) >= _SYMM_MEM_MAX_BUFFERS:
        return None

    buffer = symm_mem.empty(x.numel(), dtype=x.dtype, device=x.device)
    symm_mem.rendezvous(buffer, group.group_name)
    _symm_buffers[key] = buffer
    return buffer


@torch.library.custom_op("xllm_ops::all_reduce_", mutates_args={"x"})
def all_reduce_(x: torch.Tensor, group_name: str = "tp") -> None:
    group = _require_group(x, group_name)
    buffer = _symm_buffer(group, group_name, x)
    if buffer is None:
        dist.all_reduce(x, group=group)
        return
    flat = x.view(-1)
    buffer.copy_(flat)
    torch.ops.symm_mem.one_shot_all_reduce_out(buffer, "sum", group.group_name, flat)


@torch.library.custom_op("xllm_ops::broadcast_", mutates_args={"x"})
def broadcast_(x: torch.Tensor, src: int, group_name: str = "tp") -> None:
    group = _require_group(x, group_name)
    if not 0 <= src < group.size():
        raise RuntimeError(f"invalid {group_name} source rank {src}")
    ranks = _group_ranks.get((group_name, str(x.device)))
    if ranks is None:
        raise RuntimeError(f"{group_name} group rank map is unavailable")
    dist.broadcast(x, src=ranks[src], group=group)


@broadcast_.register_fake
def _(x: torch.Tensor, src: int, group_name: str = "tp") -> None:
    del src, group_name
    return None


@all_reduce_.register_fake
def _(x: torch.Tensor, group_name: str = "tp") -> None:
    return None


@torch.library.custom_op("xllm_ops::all_gather", mutates_args=())
def all_gather(x: torch.Tensor, dim: int, world_size: int, group_name: str = "tp") -> torch.Tensor:
    group = _require_group(x, group_name)
    if group.size() != world_size:
        raise RuntimeError(f"{group_name} world-size mismatch: expected {world_size}, got {group.size()}")
    if dim == 0:
        output = x.new_empty((x.shape[0] * world_size, *x.shape[1:]))
        dist.all_gather_into_tensor(output, x.contiguous(), group=group)
        return output
    # Non-row gathers can receive strided inputs. HCCL requires contiguous
    # buffers; contiguous() reuses storage when the input is already contiguous.
    x = x.contiguous()
    chunks = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(chunks, x, group=group)
    return torch.cat(chunks, dim=dim)


@all_gather.register_fake
def _(x: torch.Tensor, dim: int, world_size: int, group_name: str = "tp") -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] *= world_size
    return x.new_empty(shape)


def _reduce_scatter_shape(x: torch.Tensor, world_size: int) -> tuple[int, ...]:
    if world_size <= 0 or x.ndim == 0 or x.shape[0] <= 0 or x.shape[0] % world_size:
        raise ValueError("reduce_scatter requires a positive world size dividing nonempty input rows")
    return (x.shape[0] // world_size, *x.shape[1:])


@torch.library.custom_op("xllm_ops::reduce_scatter", mutates_args=())
def reduce_scatter(x: torch.Tensor, world_size: int, group_name: str = "tp") -> torch.Tensor:
    """Reduce full-hidden partials directly into disjoint contiguous token rows."""
    shape = _reduce_scatter_shape(x, world_size)
    if world_size == 1:
        return x.clone()
    group = _require_group(x, group_name)
    if group.size() != world_size:
        raise RuntimeError(f"{group_name} world-size mismatch: expected {world_size}, got {group.size()}")
    output = x.new_empty(shape)
    dist.reduce_scatter_tensor(output, x.contiguous(), group=group)
    return output


@reduce_scatter.register_fake
def _reduce_scatter_fake(x: torch.Tensor, world_size: int, group_name: str = "tp") -> torch.Tensor:
    return x.new_empty(_reduce_scatter_shape(x, world_size))


@torch.library.custom_op("xllm_ops::all_gather_variable", mutates_args=())
def all_gather_variable(
    x: torch.Tensor,
    token_counts: list[int],
    rank: int,
    group_name: str,
) -> torch.Tensor:
    group = _require_group(x, group_name)
    if group.size() != len(token_counts):
        raise RuntimeError(
            f"{group_name} size mismatch: group has {group.size()} ranks, "
            f"but token_counts has {len(token_counts)} entries"
        )
    if not 0 <= rank < len(token_counts):
        raise RuntimeError(f"invalid {group_name} rank {rank}")

    local_tokens = token_counts[rank]
    if local_tokens < 0 or local_tokens > x.shape[0]:
        raise RuntimeError(f"invalid local token count {local_tokens} for input with {x.shape[0]} rows")
    padded_tokens = max(max(token_counts, default=0), 1)
    padded_shape = list(x.shape)
    padded_shape[0] = padded_tokens
    padded = x.new_zeros(padded_shape)
    if local_tokens:
        padded[:local_tokens].copy_(x[:local_tokens])

    chunks = [torch.empty_like(padded) for _ in token_counts]
    dist.all_gather(chunks, padded, group=group)
    valid_chunks = [chunk[:count] for chunk, count in zip(chunks, token_counts) if count]
    if not valid_chunks:
        empty_shape = list(x.shape)
        empty_shape[0] = 0
        return x.new_empty(empty_shape)
    return torch.cat(valid_chunks, dim=0)


@all_gather_variable.register_fake
def _(
    x: torch.Tensor,
    token_counts: list[int],
    rank: int,
    group_name: str,
) -> torch.Tensor:
    del rank, group_name
    shape = list(x.shape)
    shape[0] = sum(token_counts)
    return x.new_empty(shape)


def gather_dp_execution_tokens(
    x: torch.Tensor,
    execution_token_counts: Sequence[int],
    rank: int,
) -> tuple[torch.Tensor, int]:
    """Gather the execution rows materialized by every DP rank."""
    counts = [int(count) for count in execution_token_counts]
    if not counts:
        return x, 0
    if any(count <= 0 for count in counts):
        raise RuntimeError(f"DP execution token counts must be positive, got {counts}")
    if not 0 <= rank < len(counts):
        raise RuntimeError(f"invalid DP rank {rank} for token counts {counts}")
    if counts[rank] != x.shape[0]:
        raise RuntimeError(
            "DP execution token count does not match the local tensor: "
            f"rank={rank}, rows={x.shape[0]}, token_counts={counts}"
        )

    return dp_all_gather(x, counts, rank), sum(counts[:rank])


__all__ = [
    "init_process_group",
    "init_tp_group",
    "tp_rank",
    "cp_rank",
    "cp_world_size",
    "layerwise_rank",
    "dcp_group",
    "all_reduce_",
    "broadcast_",
    "all_gather",
    "reduce_scatter",
    "all_gather_variable",
    "dp_all_gather",
    "gather_dp_execution_tokens",
]
