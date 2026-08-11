from __future__ import annotations

from datetime import timedelta

import torch
import torch.distributed as dist

from xllm.python.distributed import (
    ParallelGroup,
    get_parallel_group,
    parallel_group_rank,
    parallel_group_world_size,
)

_tp_groups = {}
_tp_stores = {}


def _create_process_group(
    host: str, port: int, rank: int, world_size: int, device: str
):
    """Create HCCL or NCCL ProcessGroup depending on device type."""
    store = dist.TCPStore(
        host,
        port,
        world_size,
        rank == 0,
        timedelta(minutes=5),
        wait_for_workers=False,
    )
    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        group = dist.ProcessGroupNCCL(store, rank, world_size, timedelta(minutes=5))
    else:
        import torch_npu  # noqa: F401
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL

        group = ProcessGroupHCCL(store, rank, world_size, timedelta(minutes=5))
    return store, group


def init_tp_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    device: str,
):
    try:
        registered_group = get_parallel_group("tp", device)
    except (KeyError, RuntimeError):
        registered_group = None
    if registered_group is not None:
        if (
            registered_group.local_rank != rank
            or registered_group.world_size != world_size
        ):
            raise RuntimeError(
                f"TP group for {device} is already initialized as "
                f"rank {registered_group.local_rank}/"
                f"{registered_group.world_size}, requested "
                f"rank {rank}/{world_size}"
            )
        return registered_group.process_group

    device_key = str(torch.device(device))
    group = _tp_groups.get(device_key)
    if group is not None:
        if group.rank() != rank or group.size() != world_size:
            raise RuntimeError(
                f"TP group for {device_key} is already initialized as "
                f"rank {group.rank()}/{group.size()}, requested "
                f"rank {rank}/{world_size}"
            )
        return group

    store, group = _create_process_group(host, port, rank, world_size, device)
    _tp_stores[device_key] = store
    _tp_groups[device_key] = group
    return group


def _require_group(x: torch.Tensor, group_name: str) -> ParallelGroup:
    try:
        group = get_parallel_group(group_name, x.device)
    except (KeyError, RuntimeError):
        group = None

    if group is not None:
        if group.world_size > 1 and group.process_group is None:
            raise RuntimeError(
                f"process group {group_name} for {x.device} has no backend"
            )
        return group

    if group_name != "tp":
        raise RuntimeError(
            f"parallel group {group_name} was not initialized for {x.device}"
        )

    legacy_group = _tp_groups.get(str(x.device))
    if legacy_group is None:
        raise RuntimeError(
            "tensor-parallel collective called before the TP process group "
            f"was initialized for {x.device}"
        )
    return ParallelGroup(
        name="tp",
        ranks=tuple(range(legacy_group.size())),
        local_rank=legacy_group.rank(),
        process_group=legacy_group,
    )


def tp_rank(device: object) -> int:
    """Rank in the TP group for ``device`` (0 when no TP group exists)."""
    try:
        return parallel_group_rank("tp", device)
    except (KeyError, RuntimeError):
        pass
    group = _tp_groups.get(str(torch.device(device)))
    return group.rank() if group is not None else 0


@torch.library.custom_op("xllm_ops::all_reduce_", mutates_args={"x"})
def all_reduce_(x: torch.Tensor, group_name: str = "tp") -> None:
    group = _require_group(x, group_name)
    if group.world_size == 1:
        return None
    dist.all_reduce(x, group=group.process_group)


@all_reduce_.register_fake
def _(x: torch.Tensor, group_name: str = "tp") -> None:
    del group_name
    return None


@torch.library.custom_op("xllm_ops::all_gather", mutates_args=())
def all_gather(
    x: torch.Tensor,
    dim: int,
    world_size: int,
    group_name: str = "tp",
) -> torch.Tensor:
    group = _require_group(x, group_name)
    if group.world_size != world_size:
        raise RuntimeError(
            f"{group_name} world-size mismatch: expected {world_size}, "
            f"got {group.world_size}"
        )
    if group.world_size == 1:
        return x.clone()
    chunks = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(chunks, x, group=group.process_group)
    return torch.cat(chunks, dim=dim)


@all_gather.register_fake
def _(
    x: torch.Tensor,
    dim: int,
    world_size: int,
    group_name: str = "tp",
) -> torch.Tensor:
    del group_name
    shape = list(x.shape)
    shape[dim] *= world_size
    return x.new_empty(shape)


@torch.library.custom_op("xllm_ops::reduce_scatter", mutates_args=())
def reduce_scatter(
    x: torch.Tensor,
    dim: int,
    world_size: int,
    group_name: str,
) -> torch.Tensor:
    group = _require_group(x, group_name)
    if group.world_size != world_size:
        raise RuntimeError(
            f"{group_name} world-size mismatch: expected {world_size}, "
            f"got {group.world_size}"
        )
    if x.shape[dim] % world_size != 0:
        raise ValueError("reduce-scatter dimension must divide world size")
    if group.world_size == 1:
        return x.clone()
    moved = x.movedim(dim, 0).contiguous()
    output = moved.new_empty((moved.shape[0] // world_size, *moved.shape[1:]))
    dist.reduce_scatter_tensor(output, moved, group=group.process_group)
    return output.movedim(0, dim)


@reduce_scatter.register_fake
def _(
    x: torch.Tensor,
    dim: int,
    world_size: int,
    group_name: str,
) -> torch.Tensor:
    del group_name
    shape = list(x.shape)
    shape[dim] //= world_size
    return x.new_empty(shape)


@torch.library.custom_op("xllm_ops::all_to_all_single", mutates_args=())
def all_to_all_single(
    x: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group_name: str,
) -> torch.Tensor:
    group = _require_group(x, group_name)
    if len(output_split_sizes) != group.world_size:
        raise ValueError("all-to-all output splits must match group size")
    if len(input_split_sizes) != group.world_size:
        raise ValueError("all-to-all input splits must match group size")
    if sum(input_split_sizes) != x.shape[0]:
        raise ValueError("all-to-all input splits must match tensor size")
    output = x.new_empty((sum(output_split_sizes), *x.shape[1:]))
    if group.world_size == 1:
        output.copy_(x)
        return output
    dist.all_to_all_single(
        output,
        x,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group.process_group,
    )
    return output


@all_to_all_single.register_fake
def _(
    x: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group_name: str,
) -> torch.Tensor:
    del input_split_sizes, group_name
    return x.new_empty((sum(output_split_sizes), *x.shape[1:]))
