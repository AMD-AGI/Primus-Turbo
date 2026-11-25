###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import operator
import os
import socket
import warnings
from functools import reduce
from typing import Optional, Sequence

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch._tensor import _dtype_to_typestr

import primus_turbo.common.envs as envs
from primus_turbo.pytorch.core.pyhip_runtime_wrapper import (
    HIPRuntimeLibrary,
    get_hip_runtime_lib,
    hipDeviceAttributeEnum,
    hipMemAllocationGranularityEnum,
    hipMemAllocationHandleTypeEnum,
    hipMemAllocationProp,
    hipMemAllocationTypeEnum,
    hipMemcpyKindEnum,
    hipMemLocationTypeEnum,
)
from primus_turbo.pytorch.core.utils import get_device_compute_capability

# Realtime-clock cycles (100 MHz) a barrier spin may wait before trapping. A spin that never
# ends wedges the queue, and on gfx1250 tearing down a spinning queue can leave the GPU's
# scheduler firmware unrecoverable until reboot, so a dead peer must abort the kernel instead.
_SPIN_TIMEOUT_CYCLES = 3_000_000_000


@triton.jit
def _read_realtime(GFX12: tl.constexpr):
    if GFX12:
        t = tl.inline_asm_elementwise(
            "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\n\ts_wait_kmcnt 0",
            "=s",
            [],
            dtype=tl.int64,
            is_pure=False,
            pack=1,
        )
    else:
        t = tl.inline_asm_elementwise(
            "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", [], dtype=tl.int64, is_pure=False, pack=1
        )
    return t


@triton.jit
def _trap():
    tl.inline_asm_elementwise("s_trap 2\n\ts_mov_b32 $0, 0", "=s", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _barrier_all_ipc_kernel(rank, num_ranks, comm_buf_base_ptrs, GFX12: tl.constexpr, TIMEOUT: tl.constexpr):
    for i in range(num_ranks):
        remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
        start = _read_realtime(GFX12)
        while tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0:
            if _read_realtime(GFX12) - start > TIMEOUT:
                _trap()

    for i in range(num_ranks):
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
        start = _read_realtime(GFX12)
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            if _read_realtime(GFX12) - start > TIMEOUT:
                _trap()

    tl.debug_barrier()


_group_name_to_workspace_tensor = {}

# Zero-copy GPU tensor from raw device ptr via __cuda_array_interface__; override bfloat16 typestr to "<u2" since C++ import rejects void "V".

_PROXY_TYPESTR: dict[torch.dtype, str] = {
    torch.bfloat16: "<u2",
    torch.float8_e4m3fn: "|u1",
    torch.float8_e5m2: "|u1",
    torch.float8_e4m3fnuz: "|u1",
    torch.float8_e5m2fnuz: "|u1",
}


class _DeviceArrayView:
    """Wrapper exposing __cuda_array_interface__ for a raw GPU ptr so torch.as_tensor can consume it zero-copy."""

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
    """Create a non-owning tensor backed by an existing device ptr via __cuda_array_interface__."""
    typestr = _PROXY_TYPESTR.get(dtype) or _dtype_to_typestr(dtype)
    view = _DeviceArrayView(ptr_int, tuple(shape), typestr)
    tensor = torch.as_tensor(view, device=torch.device("cuda", device_id))
    if tensor.dtype != dtype:
        tensor = tensor.view(dtype)
    return tensor


_DEFAULT_SIGNAL_PAD_SIZE = 1024

_SYMM_MEM_BACKENDS = ("auto", "ipc", "fabric")


def _os_instance_id() -> str:
    """Identity of the kernel this process runs on: legacy HIP IPC handles only open on the same one
    (containers on one host share it)."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()
    except OSError:
        return socket.gethostname()


def resolve_symm_mem_backend(group: dist.ProcessGroup, backend: Optional[str] = None) -> str:
    """Collective: the heap backend every rank of ``group`` uses.

    ``backend`` (else ``$PRIMUS_TURBO_SYMM_MEM_BACKEND``, else ``"auto"``) is ``"ipc"``, ``"fabric"`` or
    ``"auto"``. ``"auto"`` keeps legacy IPC when the group lives on one OS instance and takes fabric
    handles when it spans several, e.g. the trays of an MI455X UALink pod, which only fabric-exported
    memory can be mapped across."""
    requested = (backend or envs.PRIMUS_TURBO_SYMM_MEM_BACKEND).lower()
    if requested not in _SYMM_MEM_BACKENDS:
        raise ValueError(f"symmetric memory backend must be one of {_SYMM_MEM_BACKENDS}, got {requested!r}")
    infos = [None] * group.size()
    dist.all_gather_object(infos, (_os_instance_id(), requested, _fabric_supported()), group=group)
    if len({req for _, req, _ in infos}) != 1:
        raise ValueError(f"ranks of one group requested different symmetric memory backends: {infos}")
    num_os = len({os_id for os_id, _, _ in infos})
    if requested == "auto":
        requested = "ipc" if num_os == 1 else "fabric"
    if requested == "ipc" and num_os > 1:
        raise RuntimeError(
            f"the group spans {num_os} OS instances, which legacy HIP IPC handles cannot cross; "
            "use the 'fabric' symmetric memory backend"
        )
    if requested == "fabric" and not all(ok for _, _, ok in infos):
        bad = [rank for rank, (_, _, ok) in enumerate(infos) if not ok]
        raise RuntimeError(
            f"the 'fabric' symmetric memory backend needs fabric handle support on every rank's device; "
            f"group ranks {bad} lack it"
            + (
                f" (the group spans {num_os} OS instances, so legacy IPC is no alternative)"
                if num_os > 1
                else ""
            )
        )
    return requested


def _fabric_supported() -> bool:
    try:
        return bool(
            get_hip_runtime_lib().hipDeviceGetAttribute(
                hipDeviceAttributeEnum.hipDeviceAttributeHandleTypeFabricSupported,
                torch.cuda.current_device(),
            )
        )
    except Exception:
        return False


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class _IpcHeap:
    """One ``hipMalloc`` allocation per rank; peers open it through legacy HIP IPC (same OS instance)."""

    def __init__(self, lib: HIPRuntimeLibrary, group: dist.ProcessGroup, size: int, uncached: bool):
        self.lib = lib
        self.rank = group.rank()
        self.ptrs: list[int] = []
        world_size = group.size()
        ptr = (lib.hipMallocUncached if uncached else lib.hipMalloc)(size)
        ptrs = [0] * world_size
        ptrs[self.rank] = SymmetricMemory._c_void_p_to_int(ptr)
        try:
            lib.hipMemset(ptr, 0, size)
            handle = lib.mem_handle_to_bytes(lib.hipIpcGetMemHandle(ptr))
            handles = [None] * world_size
            dist.all_gather_object(handles, handle, group=group)
            for rank in range(world_size):
                if rank != self.rank:
                    ptrs[rank] = SymmetricMemory._c_void_p_to_int(lib.hipIpcOpenMemHandle(handles[rank]))
        except Exception:
            self.ptrs = ptrs
            self.close()
            raise
        self.ptrs = ptrs

    def close(self) -> None:
        for rank, ptr in enumerate(self.ptrs):
            if ptr == 0:
                continue
            try:
                if rank == self.rank:
                    self.lib.hipFree(ptr)
                else:
                    self.lib.hipIpcCloseMemHandle(ptr)
            except Exception as e:
                warnings.warn(f"SymmetricMemory cleanup error for rank {rank}: {e}")
        self.ptrs = []


class _FabricHeap:
    """A VMM allocation per rank exported as a fabric handle; every rank imports and maps the others'.

    Fabric handles open on any OS instance of the scale-up domain (the UALink pod of MI455X trays), so
    the group may span hosts. Each mapping is visible to this process's device only."""

    def __init__(
        self, lib: HIPRuntimeLibrary, group: dist.ProcessGroup, size: int, uncached: bool, device: int
    ):
        self.lib = lib
        self.device = device
        self.ptrs: list[int] = []
        self._mappings: list[tuple[int, object]] = []
        prop = hipMemAllocationProp()
        prop.type = (
            hipMemAllocationTypeEnum.hipMemAllocationTypeUncached
            if uncached
            else hipMemAllocationTypeEnum.hipMemAllocationTypePinned
        )
        prop.requestedHandleType = hipMemAllocationHandleTypeEnum.hipMemHandleTypeFabric
        prop.location.type = hipMemLocationTypeEnum.hipMemLocationTypeDevice
        prop.location.id = device
        self.granularity = lib.hipMemGetAllocationGranularity(
            prop, hipMemAllocationGranularityEnum.hipMemAllocationGranularityRecommended
        )
        self.size = _align(size, self.granularity)
        rank, world_size = group.rank(), group.size()
        ptrs = [0] * world_size
        try:
            ptrs[rank] = self._map(lib.hipMemCreate(self.size, prop))
            lib.hipMemset(ptrs[rank], 0, self.size)
            handle = lib.hipMemExportToFabricHandle(self._mappings[0][1])
            handles = [None] * world_size
            dist.all_gather_object(handles, handle, group=group)
            for peer in range(world_size):
                if peer != rank:
                    ptrs[peer] = self._map(lib.hipMemImportFromFabricHandle(handles[peer]))
        except Exception:
            self.close()
            raise
        self.ptrs = ptrs

    def _map(self, handle) -> int:
        """Reserve a VA range, map ``handle`` (owned from here on) there and open it to this device."""
        va, mapped = None, False
        try:
            va = self.lib.hipMemAddressReserve(self.size, self.granularity).value
            self.lib.hipMemMap(va, self.size, 0, handle)
            mapped = True
            self.lib.hipMemSetAccess(va, self.size, [self.device])
        except Exception:
            self._release(va, mapped, handle)
            raise
        self._mappings.append((va, handle))
        return va

    def _release(self, va: Optional[int], mapped: bool, handle) -> None:
        steps = []
        if mapped:
            steps.append(("hipMemUnmap", lambda: self.lib.hipMemUnmap(va, self.size)))
        if va is not None:
            steps.append(("hipMemAddressFree", lambda: self.lib.hipMemAddressFree(va, self.size)))
        steps.append(("hipMemRelease", lambda: self.lib.hipMemRelease(handle)))
        for name, step in steps:
            try:
                step()
            except Exception as e:
                warnings.warn(f"SymmetricMemory fabric cleanup: {name} failed: {e}")

    def close(self) -> None:
        for va, handle in reversed(self._mappings):
            self._release(va, True, handle)
        self._mappings = []
        self.ptrs = []


class SymmetricMemory:
    """GPU symmetric memory: one heap per rank, every rank holding device pointers to all of them.

    The heaps come from legacy HIP IPC (one OS instance) or fabric handles (a scale-up domain that
    may span hosts); see ``resolve_symm_mem_backend``. The caller must set the correct CUDA device
    (local rank) first."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        alloc_size: int,
        signal_pad_size: int = _DEFAULT_SIGNAL_PAD_SIZE,
        uncached_signal_pad: bool = False,
        uncached_buffer: bool = False,
        backend: Optional[str] = None,
    ):
        if alloc_size <= 0:
            raise ValueError(f"requested alloc size must be greater than 0, got {alloc_size}")
        if signal_pad_size < 0:
            raise ValueError(f"requested signal_pad_size must be non-negative, got {signal_pad_size}")
        if not torch.cuda.is_available():
            raise RuntimeError("SymmetricMemory requires CUDA/HIP device support.")

        self.lib = get_hip_runtime_lib()
        self.group = group
        self.rank = group.rank()
        self.world_size = group.size()
        self.buffer_size = alloc_size
        self.signal_pad_size = signal_pad_size
        self.device = torch.cuda.current_device()
        self.backend = resolve_symm_mem_backend(group, backend)

        self.buffer_ptrs: list[int] = []
        self.signal_pad_ptrs: list[int] = []
        self.buffer_ptrs_dev: torch.Tensor = None
        self.signal_pad_ptrs_dev: torch.Tensor = None
        self.is_destroyed = False
        self._heaps: list = []

        try:
            # ``uncached_buffer``: same trade-off as ``uncached_signal_pad`` below, for a caller whose
            # cross-rank protocol relies on every access to the heap bypassing L2 (DeepEP's channel
            # ring buffers, which drop the release/acquire fences around their head/tail updates).
            self.buffer_ptrs = self._open_heap(alloc_size, uncached_buffer).ptrs
            # signal_pad allocated only when requested (bf16 mega passes 0 and skips it).
            #
            # ``uncached_signal_pad`` is for a caller that puts DATA in the pad, not just flags. The
            # fp8 mega combine does: its two-heap layout keeps the combine buffer here, a peer PUSHes
            # the payload + E8M0 and the local reduce reads it, and a cached pad leaves that write in
            # a stale L2 line -- the reduce then reads a garbage E8M0 exponent (=+inf) and the output
            # goes non-finite. Uncached bypasses L2, so the cross-rank read sees what landed.
            #
            # It is off by default because it is not free: uncached traffic skips L2 for local access
            # too, and a pad holding only spin-wait flags (async_tp's few bytes per split, or the
            # 1 KB default this class hands out) does not need it.
            if self.signal_pad_size > 0:
                self.signal_pad_ptrs = self._open_heap(self.signal_pad_size, uncached_signal_pad).ptrs
        except Exception:
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
        """Convert a ctypes.c_void_p to a plain int, treating NULL (value is None) as an error."""
        val = ptr.value
        if val is None or val == 0:
            raise ValueError(f"NULL device pointer: {ptr!r}")
        return val

    def _open_heap(self, size: int, uncached: bool):
        """Collective: allocate this rank's zeroed heap and map every rank's; returns the heap."""
        if self.backend == "fabric":
            heap = _FabricHeap(self.lib, self.group, size, uncached, self.device)
        else:
            heap = _IpcHeap(self.lib, self.group, size, uncached)
        self._heaps.append(heap)
        return heap

    def _cleanup_ptrs(self):
        """Release all allocated memory and peer mappings, tolerating individual failures."""
        for heap in reversed(self._heaps):
            heap.close()
        self._heaps = []
        self.buffer_ptrs = []
        self.signal_pad_ptrs = []

    # get_buffer / get_signal_pad: mirror of the C++ SymmetricMemory API

    def get_buffer(
        self,
        rank: int,
        sizes: Sequence[int],
        dtype: torch.dtype,
        storage_offset: int = 0,
    ) -> torch.Tensor:
        """Return a zero-copy tensor view into rank's buffer (storage_offset in elements)."""
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
        """Return a zero-copy tensor view into rank's signal pad (None sizes = flat 1-D, dtype defaults to uint32)."""
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
        """GPU-side IPC barrier across all ranks on the given stream via the signal-pad memory."""
        if self.signal_pad_size <= 0:
            raise RuntimeError("barrier() requires signal_pad_size > 0")
        if stream is None:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            _barrier_all_ipc_kernel[(1,)](
                self.rank,
                self.world_size,
                self.signal_pad_ptrs_dev,
                GFX12=get_device_compute_capability()[0] >= 12,
                TIMEOUT=_SPIN_TIMEOUT_CYCLES,
            )

    def destroy(self):
        """Release all memory and IPC handles; sync GPU ops and ranks before calling."""
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
    """Get the group's symmetric memory workspace, reallocating if min_size/min_signal_pad_size exceeds the current one."""
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
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(rank % torch.cuda.device_count())
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
