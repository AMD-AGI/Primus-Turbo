###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import operator
import warnings
from functools import reduce
from typing import Optional, Sequence

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch._tensor import _dtype_to_typestr

from primus_turbo.pytorch.core.pyhip_runtime_wrapper import (
    get_hip_runtime_lib,
    hipMemcpyKindEnum,
)


@triton.jit
def _barrier_all_ipc_kernel(rank, num_ranks, comm_buf_base_ptrs):
    for i in range(num_ranks):
        remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
        while tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0:
            pass

    for i in range(num_ranks):
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass

    tl.debug_barrier()


_group_name_to_workspace_tensor = {}

# ---------------------------------------------------------------------------
# Zero-copy GPU tensor from raw device pointer via __cuda_array_interface__
#
# torch.as_tensor() natively consumes objects exposing this protocol (even on
# ROCm where the HIPified code path handles it identically).
#
# Typestr generation reuses torch._tensor._dtype_to_typestr (the same mapping
# that Tensor.__cuda_array_interface__ uses for export).  However, the C++
# import path (tensor_from_cuda_array_interface) parses the typestr via the
# NumPy C API (PyArray_DescrConverter) and then maps the numpy type_num to a
# torch ScalarType (numpy_dtype_to_aten).  That mapping only covers standard
# numpy type codes and rejects void ("V").  bfloat16's typestr is "<V2", so
# we override it with "<u2" (uint16, same byte width) and .view() back.
# ---------------------------------------------------------------------------

_PROXY_TYPESTR: dict[torch.dtype, str] = {
    torch.bfloat16: "<u2",
    torch.float8_e4m3fn: "|u1",
    torch.float8_e5m2: "|u1",
    torch.float8_e4m3fnuz: "|u1",
    torch.float8_e5m2fnuz: "|u1",
}


class _DeviceArrayView:
    """Lightweight wrapper exposing ``__cuda_array_interface__`` for a raw
    GPU pointer so that ``torch.as_tensor`` can consume it zero-copy."""

    __slots__ = ("__cuda_array_interface__",)

    def __init__(self, ptr: int, shape: tuple[int, ...], typestr: str):
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),
            "strides": None,
        }


def _tensor_from_device_ptr(
    ptr_int: int,
    shape: Sequence[int],
    dtype: torch.dtype,
    device_id: int,
) -> torch.Tensor:
    """Create a non-owning PyTorch tensor backed by an existing device pointer.

    Equivalent to ``at::for_blob(ptr, sizes).options(opts).make_tensor()``
    in C++, implemented purely in Python via ``__cuda_array_interface__``.
    """
    typestr = _PROXY_TYPESTR.get(dtype) or _dtype_to_typestr(dtype)
    view = _DeviceArrayView(ptr_int, tuple(shape), typestr)
    tensor = torch.as_tensor(view, device=torch.device("cuda", device_id))
    if tensor.dtype != dtype:
        tensor = tensor.view(dtype)
    return tensor


_DEFAULT_SIGNAL_PAD_SIZE = 1024


class SymmetricMemory:
    """GPU symmetric memory backed by HIP IPC.

    The caller must set the correct CUDA device (via torch.cuda.set_device
    with the *local* rank) before constructing this object.
    """

    def __init__(
        self, group: dist.ProcessGroup, alloc_size: int, signal_pad_size: int = _DEFAULT_SIGNAL_PAD_SIZE
    ):
        if alloc_size <= 0:
            raise ValueError(f"requested alloc size must be greater than 0, got {alloc_size}")
        if signal_pad_size <= 0:
            raise ValueError(f"requested signal_pad_size must be greater than 0, got {signal_pad_size}")
        if not torch.cuda.is_available():
            raise RuntimeError("SymmetricMemory requires CUDA/HIP device support.")

        self.lib = get_hip_runtime_lib()
        self.group = group
        self.rank = group.rank()
        self.world_size = group.size()
        self.buffer_size = alloc_size
        self.signal_pad_size = signal_pad_size
        self.device = torch.cuda.current_device()

        self.buffer_ptrs: list[int] = []
        self.signal_pad_ptrs: list[int] = []
        self.buffer_ptrs_dev: torch.Tensor = None
        self.signal_pad_ptrs_dev: torch.Tensor = None
        self.is_destroyed = False

        buffer_ptr = None
        signal_pad_ptr = None
        try:
            buffer_ptr = self.lib.hipMalloc(alloc_size)
            signal_pad_ptr = self.lib.hipMalloc(self.signal_pad_size)
            self.lib.hipMemset(buffer_ptr, 0, alloc_size)
            self.lib.hipMemset(signal_pad_ptr, 0, self.signal_pad_size)

            self.buffer_ptrs = self._rendezvous(buffer_ptr)
            self.signal_pad_ptrs = self._rendezvous(signal_pad_ptr)
        except Exception:
            if not self.buffer_ptrs and buffer_ptr is not None:
                self._try_free(buffer_ptr)
            if not self.signal_pad_ptrs and signal_pad_ptr is not None:
                self._try_free(signal_pad_ptr)
            self._cleanup_ptrs()
            self.is_destroyed = True
            raise

        self.buffer_ptrs_dev = torch.tensor(
            self.buffer_ptrs,
            dtype=torch.int64,
            device="cuda",
        )
        self.signal_pad_ptrs_dev = torch.tensor(
            self.signal_pad_ptrs,
            dtype=torch.int64,
            device="cuda",
        )

        self.group.barrier()

    @staticmethod
    def _c_void_p_to_int(ptr) -> int:
        """Convert a ctypes.c_void_p to a plain Python int.

        c_void_p.value is *None* (not 0) when the pointer is NULL — this is
        an easy-to-miss ctypes gotcha.  By normalizing to int once at storage
        time, the rest of the code only ever deals with plain ints.
        """
        val = ptr.value
        if val is None or val == 0:
            raise ValueError(f"NULL device pointer: {ptr!r}")
        return val

    def _rendezvous(self, ptr) -> list[int]:
        """Exchange IPC handles across all ranks and open remote memory.

        Returns a ``list[int]`` of device-pointer values (one per rank),
        uniformly typed to avoid mixed c_void_p / int issues downstream.
        """
        mem_handle = self.lib.hipIpcGetMemHandle(ptr)
        mem_handle_bytes = self.lib.mem_handle_to_bytes(mem_handle)
        mem_handle_list = [None] * self.world_size
        dist.all_gather_object(mem_handle_list, mem_handle_bytes, group=self.group)

        ptr_list: list[int] = [0] * self.world_size
        opened_ranks: list[int] = []
        try:
            for rank in range(self.world_size):
                if rank == self.rank:
                    ptr_list[rank] = self._c_void_p_to_int(ptr)
                else:
                    ipc_ptr = self.lib.hipIpcOpenMemHandle(mem_handle_list[rank])
                    ptr_list[rank] = self._c_void_p_to_int(ipc_ptr)
                    opened_ranks.append(rank)
        except Exception:
            for r in opened_ranks:
                try:
                    self.lib.hipIpcCloseMemHandle(ptr_list[r])
                except Exception:
                    pass
            raise
        return ptr_list

    def _try_free(self, ptr):
        try:
            self.lib.hipFree(ptr)
        except Exception as e:
            warnings.warn(f"hipFree failed during cleanup: {e}")

    def _cleanup_ptrs(self):
        """Release all allocated memory and IPC handles, tolerating individual failures."""
        for ptrs in (self.buffer_ptrs, self.signal_pad_ptrs):
            for rank in range(min(self.world_size, len(ptrs))):
                if ptrs[rank] == 0:
                    continue
                try:
                    if rank == self.rank:
                        self.lib.hipFree(ptrs[rank])
                    else:
                        self.lib.hipIpcCloseMemHandle(ptrs[rank])
                except Exception as e:
                    warnings.warn(f"SymmetricMemory cleanup error for rank {rank}: {e}")
        self.buffer_ptrs = []
        self.signal_pad_ptrs = []

    # ------------------------------------------------------------------
    #  get_buffer / get_signal_pad  –  mirror of the C++ SymmetricMemory API
    # ------------------------------------------------------------------

    def get_buffer(
        self,
        rank: int,
        sizes: Sequence[int],
        dtype: torch.dtype,
        storage_offset: int = 0,
    ) -> torch.Tensor:
        """Return a zero-copy tensor view into *rank*'s buffer.

        Mirrors ``SymmetricMemory::get_buffer`` in
        ``torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp``.

        Args:
            rank: peer rank whose buffer to access.
            sizes: desired tensor shape.
            dtype: desired tensor dtype.
            storage_offset: offset in *elements* (not bytes) from the start
                of the buffer.
        """
        if not (0 <= rank < self.world_size):
            raise ValueError(f"Invalid peer rank: {rank}")

        element_size = dtype.itemsize
        offset_bytes = storage_offset * element_size
        numel = 1
        for s in sizes:
            numel *= s
        req_bytes = offset_bytes + numel * element_size
        if req_bytes > self.buffer_size:
            raise ValueError(
                f"SymmetricMemory.get_buffer: the requested size "
                f"({req_bytes} bytes) exceeds the allocated size "
                f"({self.buffer_size} bytes)"
            )

        return _tensor_from_device_ptr(
            self.buffer_ptrs[rank] + offset_bytes,
            tuple(sizes),
            dtype,
            self.device,
        )

    def get_signal_pad(
        self,
        rank: int,
        sizes: Optional[Sequence[int]] = None,
        dtype: Optional[torch.dtype] = None,
        storage_offset: int = 0,
    ) -> torch.Tensor:
        """Return a zero-copy tensor view into *rank*'s signal pad.

        Args:
            rank: peer rank whose signal pad to access.
            sizes: desired tensor shape.  When *None*, the signal pad is
                treated as a flat 1-D tensor of ``signal_pad_size / elem``
                elements.
            dtype: defaults to ``torch.uint32``.
            storage_offset: offset in *elements*.
        """
        if not (0 <= rank < self.world_size):
            raise ValueError(f"Invalid peer rank: {rank}")

        if dtype is None:
            dtype = torch.uint32
        element_size = dtype.itemsize

        if sizes is None:
            shape: tuple[int, ...] = (self.signal_pad_size // element_size,)
        else:
            shape = tuple(sizes)

        numel = reduce(operator.mul, shape, 1)
        offset_bytes = storage_offset * element_size
        req_bytes = offset_bytes + numel * element_size
        if req_bytes > self.signal_pad_size:
            raise ValueError(
                f"SymmetricMemory.get_signal_pad: the requested size "
                f"({req_bytes} bytes) exceeds the allocated size "
                f"({self.signal_pad_size} bytes)"
            )

        return _tensor_from_device_ptr(
            self.signal_pad_ptrs[rank] + offset_bytes,
            shape,
            dtype,
            self.device,
        )

    def barrier(self, stream: torch.cuda.Stream | None = None) -> None:
        """GPU-side IPC barrier across all ranks on the given stream.

        Each rank atomically signals its arrival to every peer and then
        spins until all peers have signalled, using the signal-pad memory
        that is already IPC-mapped.
        """
        if stream is None:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            _barrier_all_ipc_kernel[(1,)](
                self.rank,
                self.world_size,
                self.signal_pad_ptrs_dev,
            )

    def destroy(self):
        """Release all memory and IPC handles.

        Callers should ensure all GPU operations on symmetric memory are
        complete (torch.cuda.synchronize) and all ranks are synchronized
        (group.barrier) before calling destroy.
        """
        if self.is_destroyed:
            return
        self.is_destroyed = True
        self._cleanup_ptrs()
        self.buffer_ptrs_dev = None
        self.signal_pad_ptrs_dev = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False


def get_symm_mem_workspace(
    group: dist.ProcessGroup,
    min_size: int,
    min_signal_pad_size: int = 0,
) -> SymmetricMemory:
    """
    Get the symmetric memory workspace associated with the process group. If
    ``min_size`` is greater than the workspace associated with ``group_name``,
    the workspace will be re-allocated and re-rendezvous'd.

    Args:
        group: the process group.
        min_size: the size requirement for the buffer workspace in bytes.
        min_signal_pad_size: the size requirement for the per-rank signal pad
            in bytes.  When zero the default (1024 B) is used for new
            allocations.

    Returns:
        SymmetricMemory: the symmetric memory workspace associated with the
        group.
    """
    symm_mem = _group_name_to_workspace_tensor.get(group.group_name, None)
    buf_size = symm_mem.buffer_size if symm_mem is not None else 0
    sig_size = symm_mem.signal_pad_size if symm_mem is not None else 0
    needs_realloc = symm_mem is None or buf_size < min_size or sig_size < min_signal_pad_size
    if needs_realloc:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "get_symm_mem_workspace(): cannot resize the symmetric-memory "
                f"workspace during CUDA graph capture. Requested {min_size} buffer bytes "
                f"and {min_signal_pad_size} signal-pad bytes, "
                f"but the current workspace has {buf_size} buffer bytes "
                f"and {sig_size} signal-pad bytes. Call "
                f"`get_symm_mem_workspace(group={group.group_name!r}, "
                f"min_size={min_size}, min_signal_pad_size={min_signal_pad_size})` "
                f"before starting graph capture."
            )
        old_symm_mem = symm_mem
        symm_mem = SymmetricMemory(
            group,
            max(buf_size, min_size),
            signal_pad_size=max(sig_size, min_signal_pad_size, _DEFAULT_SIGNAL_PAD_SIZE),
        )
        _group_name_to_workspace_tensor[group.group_name] = symm_mem
        if old_symm_mem is not None:
            old_symm_mem.destroy()
    return symm_mem


if __name__ == "__main__":
    import os

    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(rank)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("nccl", world_size=world_size, rank=rank)
    group = dist.new_group(list(range(world_size)))
    symm_mem = SymmetricMemory(group, 1024)

    def self_test_ring_write(symm: SymmetricMemory, base_value: int = 1000):
        """Write to next rank's buffer_ptr via IPC and verify local receive."""
        if symm.world_size < 2:
            if symm.rank == 0:
                print("Skip IPC ring-write self-test: world_size < 2")
            return

        bytes_needed = torch.tensor([], dtype=torch.int32).element_size()
        if symm.buffer_size < bytes_needed:
            raise ValueError(
                f"buffer_size={symm.buffer_size} is too small for self-test ({bytes_needed} bytes)"
            )

        dst_rank = (symm.rank + 1) % symm.world_size
        src_rank = (symm.rank - 1 + symm.world_size) % symm.world_size
        stream_ptr = torch.cuda.current_stream().cuda_stream

        send = torch.tensor([base_value + symm.rank], dtype=torch.int32, device="cuda")
        recv = torch.zeros(1, dtype=torch.int32, device="cuda")

        symm.lib.hipMemcpyAsync(
            symm.buffer_ptrs[dst_rank],
            int(send.data_ptr()),
            send.numel() * send.element_size(),
            hipMemcpyKindEnum.hipMemcpyDeviceToDevice,
            stream_ptr,
        )
        torch.cuda.synchronize()
        symm.group.barrier()

        symm.lib.hipMemcpyAsync(
            int(recv.data_ptr()),
            symm.buffer_ptrs[symm.rank],
            recv.numel() * recv.element_size(),
            hipMemcpyKindEnum.hipMemcpyDeviceToDevice,
            stream_ptr,
        )
        torch.cuda.synchronize()

        expected = base_value + src_rank
        got = int(recv.item())
        ok = got == expected

        gathered = [None] * symm.world_size
        dist.all_gather_object(gathered, (symm.rank, got, expected, ok), group=symm.group)
        if symm.rank == 0:
            print("SymmetricMemory IPC ring-write self-test results:")
            for item in gathered:
                print(f"  rank={item[0]} got={item[1]} expected={item[2]} ok={item[3]}")

        if not all(item[3] for item in gathered):
            raise AssertionError(f"IPC ring-write self-test failed: {gathered}")

    self_test_ring_write(symm_mem)

    symm_mem.destroy()
