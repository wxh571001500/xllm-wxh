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

"""Custom AllGather for NZ tensors as a raw byte array.

Payload path: HcclAllGather(send=data_ptr, recv=data_ptr,
count=storage_nbytes/itemsize, dtype). No Transdata / npu_reshape on the
communicated bytes.

Only K-axis sharding (local K = full_K / world) is supported. Storage is
``[K1, N, C0]``; HCCL concat is already valid NZ ``[N, K * world]``.

Reuses the existing ProcessGroupHCCL communicator (no new HCCL communicator setup).
"""

from __future__ import annotations

import ctypes

import torch
import torch.distributed as dist
import torch_npu

ACL_FORMAT_FRACTAL_NZ = 29
FRACTAL_N0 = 16
HCCL_SUCCESS = 0
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
# FRACTAL_NZ cube is 32 bytes: C0 = 32 / itemsize (int8=32, fp16/bf16=16, fp32=8).
_NZ_CUBE_BYTES = 32

# torch_npu getHcclDataType: AllGather volume is count * sizeof(dtype).
_DTYPE_TO_HCCL = {
    torch.int8: 0,
    torch.float16: 3,
    torch.float32: 4,
    torch.bfloat16: 11,
}
_NZ_DTYPES = frozenset(_DTYPE_TO_HCCL)

_HCCL: ctypes.CDLL | None = None
_ACL: ctypes.CDLL | None = None
_COMM_CACHE: dict[int, int] = {}
_STREAM_CACHE: tuple[int, ctypes.c_void_p] | None = None
_INTERNAL_FORMAT_OK = False


def _load_hccl() -> ctypes.CDLL:
    global _HCCL
    if _HCCL is None:
        _HCCL = ctypes.CDLL("libhccl.so")
        _HCCL.HcclAllGather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        _HCCL.HcclAllGather.restype = ctypes.c_int32
    return _HCCL


def _load_acl() -> ctypes.CDLL:
    global _ACL
    if _ACL is None:
        _ACL = ctypes.CDLL("libascendcl.so")
        _ACL.aclrtMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        _ACL.aclrtMemcpyAsync.restype = ctypes.c_int32
    return _ACL


def fractal_k0(dtype: torch.dtype) -> int:
    if dtype not in _NZ_DTYPES:
        raise ValueError(f"unsupported NZ dtype {dtype}")
    item = dtype.itemsize
    if item <= 0 or _NZ_CUBE_BYTES % item != 0:
        raise ValueError(f"unsupported NZ itemsize {item} for {dtype}")
    return _NZ_CUBE_BYTES // item


def is_fractal_nz(t: torch.Tensor) -> bool:
    return int(torch_npu.get_npu_format(t)) == ACL_FORMAT_FRACTAL_NZ


def nz_storage_nbytes(t: torch.Tensor) -> int:
    """Physical NZ storage bytes from aligned logical shape (no NZ padding)."""
    return int(t.shape[0]) * int(t.shape[1]) * t.dtype.itemsize


def _internal_format_allowed() -> bool:
    global _INTERNAL_FORMAT_OK
    if _INTERNAL_FORMAT_OK:
        return True
    # torch.npu.config.allow_internal_format is write-only (no __getattr__).
    opt = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    _INTERNAL_FORMAT_OK = opt is not None and opt.decode() == "enable"
    return _INTERNAL_FORMAT_OK


def empty_nz(n: int, k: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Allocate an uninitialized 2D FRACTAL_NZ buffer (no ND→NZ Transdata).

    Does not change ``torch.npu.config.allow_internal_format``; the caller must
    already have set it to True (required by ``empty_with_format`` for NZ).
    """
    if not _internal_format_allowed():
        raise RuntimeError("torch.npu.config.allow_internal_format must be True to allocate FRACTAL_NZ")
    return torch_npu.empty_with_format((n, k), dtype=dtype, device=device, acl_format=ACL_FORMAT_FRACTAL_NZ)


def _require_nz2d(x: torch.Tensor) -> tuple[int, int, int]:
    if not x.is_npu:
        raise ValueError("expected an NPU tensor")
    if x.dim() != 2 or not is_fractal_nz(x):
        raise ValueError(f"expected 2D FRACTAL_NZ, got dim={x.dim()} format={int(torch_npu.get_npu_format(x))}")
    if x.storage_offset() != 0:
        raise ValueError("expected storage_offset=0 (whole NZ storage, not a view)")
    n, k = int(x.shape[0]), int(x.shape[1])
    k0 = fractal_k0(x.dtype)
    if n % FRACTAL_N0 != 0 or k % k0 != 0:
        raise ValueError(f"N={n} must be divisible by {FRACTAL_N0} and K={k} divisible by C0={k0} (no NZ padding)")
    return n, k, k0


def _require_out_nz(
    out: torch.Tensor,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    src: torch.Tensor,
) -> torch.Tensor:
    _require_nz2d(out)
    if out.dtype != dtype or out.device != device:
        raise ValueError("out dtype/device must match x")
    if tuple(out.shape) != (n, k):
        raise ValueError(f"out shape {tuple(out.shape)} != ({n}, {k})")
    if out.data_ptr() == src.data_ptr():
        raise ValueError("out must not alias x")
    return out


def _resolve_rank_world(
    rank_world: tuple[int, int] | None,
    group: dist.ProcessGroup | None,
) -> tuple[int, int]:
    if rank_world is None:
        if not dist.is_initialized():
            raise RuntimeError("distributed is not initialized; pass rank_world")
        rank, world = dist.get_rank(group), dist.get_world_size(group)
    else:
        rank, world = rank_world
    if world <= 0 or rank < 0 or rank >= world:
        raise ValueError(f"invalid rank/world: rank={rank} world={world}")
    return int(rank), int(world)


def _stream_handle() -> ctypes.c_void_p:
    global _STREAM_CACHE
    ptr = int(torch.npu.current_stream().npu_stream)
    cached = _STREAM_CACHE
    if cached is not None and cached[0] == ptr:
        return cached[1]
    handle = ctypes.c_void_p(ptr)
    _STREAM_CACHE = (ptr, handle)
    return handle


def _memcpy_d2d(
    dst: torch.Tensor,
    src: torch.Tensor,
    nbytes: int,
    src_off: int = 0,
    stream: ctypes.c_void_p | None = None,
) -> None:
    """Raw D2D copy of ``nbytes`` from ``src`` at byte offset ``src_off``."""
    acl = _load_acl()
    ret = acl.aclrtMemcpyAsync(
        ctypes.c_void_p(dst.data_ptr()),
        nbytes,
        ctypes.c_void_p(src.data_ptr() + src_off),
        nbytes,
        ACL_MEMCPY_DEVICE_TO_DEVICE,
        stream if stream is not None else _stream_handle(),
    )
    if ret != 0:
        raise RuntimeError(f"aclrtMemcpyAsync failed, ret={ret}")


def shard_nz(
    x: torch.Tensor,
    rank_world: tuple[int, int] | None = None,
    group: dist.ProcessGroup | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Slice a full FRACTAL_NZ weight along K into a local NZ shard (no Transdata).

    Storage is ``[K1, N, C0]``; copy this rank's ``K1/world`` tiles → NZ ``[N, K/world]``.
    When ``out is None``, returns a buffer distinct from ``x`` (including world==1).
    Pass ``rank_world=(rank, world)`` or omit it to use ``dist.get_rank/get_world_size``.
    """
    n, k, k0 = _require_nz2d(x)
    rank, world = _resolve_rank_world(rank_world, group)
    k1 = k // k0
    if k1 % world != 0:
        raise ValueError(f"K1={k1} must be divisible by world={world} for K-shard")
    local_k = k // world
    shard_nb = n * local_k * x.dtype.itemsize
    stream = _stream_handle()
    if out is None:
        out = empty_nz(n, local_k, x.dtype, x.device)
    else:
        _require_out_nz(out, n, local_k, x.dtype, x.device, src=x)
    _memcpy_d2d(out, x, shard_nb, src_off=rank * shard_nb, stream=stream)
    return out


def _hccl_dtype_and_count(x: torch.Tensor, nbytes: int) -> tuple[int, int]:
    hccl_ty = _DTYPE_TO_HCCL.get(x.dtype)
    if hccl_ty is None:
        raise ValueError(f"unsupported dtype for HcclAllGather: {x.dtype}")
    return hccl_ty, nbytes // x.dtype.itemsize


def _warmup_hccl(device: torch.device, group: dist.ProcessGroup | None) -> None:
    """Create the ProcessGroupHCCL communicator with a 1-element AllGather."""
    world = dist.get_world_size(group)
    send = torch.zeros(1, dtype=torch.int8, device=device)
    recv = torch.empty(world, dtype=torch.int8, device=device)
    dist.all_gather_into_tensor(recv, send, group=group)


def _hccl_comm(group: dist.ProcessGroup | None, device: torch.device) -> int:
    pg = dist.group.WORLD if group is None else group
    key = id(pg)
    cached = _COMM_CACHE.get(key)
    if cached is not None:
        return cached
    backend = pg._get_backend(torch.device("npu"))
    rank = dist.get_rank(pg)
    comm = backend.get_hccl_comm(rank)
    if not comm:
        _warmup_hccl(device, group)
        comm = backend.get_hccl_comm(rank)
    if not comm:
        raise RuntimeError("ProcessGroupHCCL.get_hccl_comm returned empty handle")
    _COMM_CACHE[key] = int(comm)
    return _COMM_CACHE[key]


def _allgather_bytes_direct(
    send: torch.Tensor,
    recv: torch.Tensor,
    nbytes: int,
    group: dist.ProcessGroup | None,
    stream: ctypes.c_void_p,
) -> None:
    hccl = _load_hccl()
    comm = _hccl_comm(group, send.device)
    hccl_ty, count = _hccl_dtype_and_count(send, nbytes)
    ret = hccl.HcclAllGather(
        ctypes.c_void_p(send.data_ptr()),
        ctypes.c_void_p(recv.data_ptr()),
        ctypes.c_uint64(count),
        hccl_ty,
        ctypes.c_void_p(comm),
        stream,
    )
    if ret != HCCL_SUCCESS:
        raise RuntimeError(f"HcclAllGather(dtype={hccl_ty}, count={count}, nbytes={nbytes}) failed, HcclResult={ret}")


def all_gather_nz_bytes(
    x: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """AllGather NZ storage as bytes along K; return FRACTAL_NZ ``[N, K*world]``.

    Pass ``out`` (FRACTAL_NZ ``[N, K*world]``, distinct from ``x``) to skip allocation.
    world==1 copies into a distinct buffer (or ``out``).
    """
    if not dist.is_initialized():
        raise RuntimeError("distributed is not initialized")
    n, k, _ = _require_nz2d(x)
    world = dist.get_world_size(group)
    send_nb = nz_storage_nbytes(x)
    out_n, out_k = n, k * world
    stream = _stream_handle()
    recv = (
        empty_nz(out_n, out_k, x.dtype, x.device)
        if out is None
        else _require_out_nz(out, out_n, out_k, x.dtype, x.device, src=x)
    )
    if world == 1:
        _memcpy_d2d(recv, x, send_nb, stream=stream)
        return recv
    _allgather_bytes_direct(x, recv, send_nb, group, stream)
    return recv


def raw_storage_uint8(t: torch.Tensor) -> torch.Tensor:
    """ND uint8 copy of physical NZ storage (no Transdata)."""
    _require_nz2d(t)
    nbytes = nz_storage_nbytes(t)
    buf = torch.empty(nbytes, dtype=torch.uint8, device=t.device)
    _memcpy_d2d(buf, t, nbytes, stream=_stream_handle())
    return buf
