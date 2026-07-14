###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import ctypes
from typing import Callable, List, Optional, Tuple

import torch
from flydsl.expr.numeric import Int64
from flydsl.expr.typing import AddressSpace, Pointer, PointerType, address_space_from_attr

try:
    import torch.distributed._symmetric_memory as symm_mem
except Exception as exception:  # pragma: no cover
    print(f"Failed to load mega symmetric memory, please check your PyTorch version: {exception}")


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


NUM_MAX_RANKS = 72
BLOCK_M = 256  # pool-block granularity + pool alignment

# default dispatched-token dtype; override via ``token_dtype`` to target fp8, etc.
TOKEN_DTYPE = torch.bfloat16
TOKEN_ALIGNMENT = 128  # get_token_alignment_for_mega_moe


def get_num_max_pool_tokens(
    num_ranks: int, num_max_tokens_per_rank: int, num_topk: int, num_experts_per_rank: int
) -> int:
    """Worst-case shared expert pool capacity + per-expert BLOCK_M padding (256-aligned)."""
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    return align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (BLOCK_M - 1),
        BLOCK_M,
    )


class SymBuffer:
    """Pure addressing handle (base + peer deltas + ``map``); one i64 table (base at [NUM_MAX_RANKS]) so it loop-carries through kernel control flow."""

    def __init__(self, container: List[int], rank_idx: int, device: Optional[torch.device] = None) -> None:
        self.rank_idx = rank_idx
        self.num_ranks = len(container)
        size = len(container)
        self.base = container[rank_idx]
        # _offsets[i] = peer_base - base (zero-padded to NUM_MAX_RANKS); private -- reach peers via map()
        self._offsets = [container[i] - self.base if i < size else 0 for i in range(NUM_MAX_RANKS)]
        self._device = device
        self._offsets_tensor = None  # lazy device-side table for kernel launches

    def _offsets_device_ptr(self) -> int:
        # build the i64 table once (deltas + base packed at [NUM_MAX_RANKS]); keep it alive on self
        if self._offsets_tensor is None:
            self._offsets_tensor = torch.tensor(
                self._offsets + [self.base],
                dtype=torch.int64,
                device=self._device if self._device is not None else "cuda",
            )
        return self._offsets_tensor.data_ptr()

    # -------- JitArgument protocol (host side) --------
    def __get_ir_types__(self) -> list:
        i64 = Int64.ir_type
        space = address_space_from_attr(AddressSpace.Global)
        return [PointerType.get(i64, space, 8)]

    def __cache_signature__(self) -> tuple:
        # only rank count specializes it; dims ride each kernel's constexpr, not the handle
        return (type(self), self.num_ranks)

    def __c_abi_spec__(self) -> list:
        def fill_offsets(a, s):
            s.value = a._offsets_device_ptr()

        return [(ctypes.c_void_p, fill_offsets)]

    # -------- DslType protocol (device side) --------
    @classmethod
    def __construct_from_ir_values__(
        cls, values: list, exemplar: Optional["SymBuffer"] = None
    ) -> "SymBuffer":
        obj = cls.__new__(cls)  # skip host __init__; rebuild from block args
        obj._offsets = Pointer(values[0])  # i64* table (deltas + base at [NUM_MAX_RANKS])
        obj.base = None  # device: read from the table via get_base_ptr()
        obj.num_ranks = exemplar.num_ranks if exemplar is not None else None
        obj.rank_idx = exemplar.rank_idx if exemplar is not None else None
        obj._device = None
        obj._offsets_tensor = None
        return obj

    def __extract_to_ir_values__(self) -> list:
        return [self._offsets]

    # -------- get_base_ptr / map (python int on host, device ops in a kernel) --------
    def get_base_ptr(self):
        if isinstance(self._offsets, list):  # host
            return self.base
        # device: base is packed at the end of the table
        return self._offsets[NUM_MAX_RANKS]

    def map(self, ptr, dst_rank_idx):
        if self.num_ranks == 1:
            return ptr
        # host: python int add; device: Pointer load (i64) + ptr
        return self._offsets[dst_rank_idx] + ptr


class Workspace:
    """Heap layout owner: 32B barrier header, then each kernel region; ``get_*_ptr`` walk the offsets."""

    NUM_BARRIER_SIGNAL_BYTES = 32
    NUM_MAX_GRID_SYNC_COUNTERS = 4

    def __init__(
        self,
        base,
        num_ranks: int,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int = 0,
        token_dtype: torch.dtype = TOKEN_DTYPE,
    ) -> None:
        self.base = base
        self.num_ranks = num_ranks
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.num_experts_per_rank = num_experts // num_ranks
        self.num_max_pool_tokens = get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk, self.num_experts_per_rank
        )
        self.hidden = hidden
        self.token_dtype = token_dtype
        self.num_max_pool_blocks = self.num_max_pool_tokens // BLOCK_M
        self.num_combine_slots = num_max_tokens_per_rank * num_topk

    # each region base = previous + its size, 256B-aligned (gfx950 cross-XCD flag coherence)
    def get_dispatch_token_pool_ptr(self):
        return self.base + align(self.NUM_BARRIER_SIGNAL_BYTES, BLOCK_M)

    def get_l2_token_buffer_ptr(self):
        pool_bytes = align(self.num_max_pool_tokens * self.hidden * self.token_dtype.itemsize, BLOCK_M)
        return self.get_dispatch_token_pool_ptr() + pool_bytes

    def get_combine_token_buffer_ptr(self):
        pool_bytes = align(self.num_max_pool_tokens * self.hidden * self.token_dtype.itemsize, BLOCK_M)
        return self.get_l2_token_buffer_ptr() + pool_bytes

    def get_dispatch_flag_ptr(self):
        combine_pool_bytes = align(self.num_combine_slots * self.hidden * self.token_dtype.itemsize, BLOCK_M)
        return self.get_combine_token_buffer_ptr() + combine_pool_bytes

    def get_combine_flag_ptr(self):
        return self.get_dispatch_flag_ptr() + align(2 * self.num_max_pool_blocks * 8, BLOCK_M)

    def get_reduce_flag_ptr(self):
        return self.get_combine_flag_ptr() + align(2 * self.num_max_pool_blocks * 8, BLOCK_M)

    def get_expert_count_buffer_ptr(self):
        return self.get_reduce_flag_ptr() + align(2 * self.num_combine_slots * 8, BLOCK_M)

    def get_pool_src_rank_ptr(self):
        return self.get_expert_count_buffer_ptr() + align(self.num_ranks * self.num_experts * 4, BLOCK_M)

    def get_pool_src_slot_ptr(self):
        return self.get_pool_src_rank_ptr() + align(self.num_max_pool_tokens * 4, BLOCK_M)

    def get_weight_recv_buf_ptr(self):
        return self.get_pool_src_slot_ptr() + align(self.num_max_pool_tokens * 4, BLOCK_M)

    def get_combine_gate_ptr(self):
        return self.get_weight_recv_buf_ptr() + align(self.num_max_pool_tokens * 4, BLOCK_M)

    def get_end_ptr(self):
        # past the last region; on a base-0 Workspace this offset is the total heap size
        return self.get_combine_gate_ptr() + align(self.num_max_tokens_per_rank * self.num_topk * 4, BLOCK_M)

    # Barrier: [0..15] 4 grid sync counters, [16..19] XGMI counter, [20..27] 2 signals

    def get_grid_sync_count_ptr(self, index: int = 0):
        assert index < self.NUM_MAX_GRID_SYNC_COUNTERS, "Grid sync index out of bounds"
        return self.base + index * 4

    def get_xgmi_barrier_counter_ptr(self):
        return self.base + self.NUM_MAX_GRID_SYNC_COUNTERS * 4

    def get_xgmi_barrier_signal_ptr(self, phase):
        return self.base + (self.NUM_MAX_GRID_SYNC_COUNTERS + 1) * 4 + phase * 4


def get_symm_buffer_size_for_mega_moe(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    token_dtype: torch.dtype = TOKEN_DTYPE,
) -> Tuple[int, Callable[["torch.Tensor"], Tuple[torch.Tensor, ...]]]:
    """Return ``(num_bytes, slice_input_buffers)``: the one owner of heap sizing + host slicing (offsets from Workspace)."""

    # end offset on a base-0 Workspace == total heap size
    num_bytes = Workspace(
        0,
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        token_dtype,
    ).get_end_ptr()

    def slice_input_buffers(buffer: "torch.Tensor") -> Tuple[torch.Tensor, ...]:
        # lazy: top-level import would cycle (pytorch.core -> pytorch -> flydsl.mega)
        from primus_turbo.pytorch.core.symm_mem import _tensor_from_device_ptr

        workspace = Workspace(
            buffer.data_ptr(),
            num_ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            token_dtype,
        )
        dev = buffer.device.index
        npt = workspace.num_max_pool_tokens
        return (
            _tensor_from_device_ptr(
                int(workspace.get_dispatch_token_pool_ptr()), (npt, hidden), token_dtype, dev
            ),
            _tensor_from_device_ptr(int(workspace.get_weight_recv_buf_ptr()), (npt,), torch.float32, dev),
            _tensor_from_device_ptr(
                int(workspace.get_l2_token_buffer_ptr()), (npt, hidden), token_dtype, dev
            ),
            _tensor_from_device_ptr(
                int(workspace.get_combine_token_buffer_ptr()),
                (workspace.num_combine_slots, hidden),
                token_dtype,
                dev,
            ),
            _tensor_from_device_ptr(int(workspace.get_pool_src_slot_ptr()), (npt,), torch.int32, dev),
            _tensor_from_device_ptr(int(workspace.get_pool_src_rank_ptr()), (npt,), torch.int32, dev),
            _tensor_from_device_ptr(
                int(workspace.get_dispatch_flag_ptr()), (2 * workspace.num_max_pool_blocks,), torch.int64, dev
            ),
            _tensor_from_device_ptr(
                int(workspace.get_combine_flag_ptr()), (2 * workspace.num_max_pool_blocks,), torch.int64, dev
            ),
            _tensor_from_device_ptr(
                int(workspace.get_reduce_flag_ptr()), (2 * workspace.num_combine_slots,), torch.int64, dev
            ),
        )

    return num_bytes, slice_input_buffers


class SymmBuffer:
    """Host owner of the IPC heap + parity bookkeeping; hands the kernel a lean ``SymBuffer``."""

    def __init__(
        self,
        group: "torch.distributed.ProcessGroup",
        *,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        token_dtype: torch.dtype = TOKEN_DTYPE,
    ) -> None:
        self.group = group
        self.rank = group.rank()
        self.world = group.size()
        self.num_experts = int(num_experts)
        self.num_max_tokens_per_rank = int(num_max_tokens_per_rank)
        self.num_topk = int(num_topk)
        self.hidden = int(hidden)
        self.intermediate_hidden = int(intermediate_hidden)
        self.key = (
            self.world,
            self.num_experts,
            self.num_max_tokens_per_rank,
            self.num_topk,
            self.hidden,
            self.intermediate_hidden,
            token_dtype,
        )

        # single layout owner: allocation size + the host-slicing hook
        self.num_bytes, slice_input_buffers = get_symm_buffer_size_for_mega_moe(
            self.world,
            self.num_experts,
            self.num_max_tokens_per_rank,
            self.num_topk,
            self.hidden,
            token_dtype,
        )
        # derived pool dims for parity bookkeeping / callers
        workspace = Workspace(
            0,
            self.world,
            self.num_experts,
            self.num_max_tokens_per_rank,
            self.num_topk,
            self.hidden,
            token_dtype,
        )
        self.num_max_pool_tokens = workspace.num_max_pool_tokens
        self.num_combine_slots = workspace.num_combine_slots
        self.num_tokens = self.num_max_tokens_per_rank  # back-compat alias

        # allocate the single symmetric-memory heap (torch official; empty() is uninit -> zero it)
        self.buffer = symm_mem.empty(self.num_bytes, dtype=torch.int8, device="cuda")
        self.symm_mem = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        heap = self.buffer
        self.group.barrier()
        torch.cuda.synchronize()

        # host-side region views, sliced by the layout owner from this rank's heap
        (
            self.dispatch_token_pool,
            self.weight_recv_buf,
            self.l2_token_buffer,
            self.combine_token_buffer,
            self.pool_src_slot,
            self.pool_src_rank,
            self.dispatch_flag,
            self.combine_flag,
            self.reduce_flag,
        ) = slice_input_buffers(heap)

        self.num_tokens_per_rank = torch.full(
            (self.world,), self.num_tokens, dtype=torch.int32, device="cuda"
        )

        # device epoch state (parity + per-bank expected); bumped by the device bump kernel
        self._disp_parity = torch.zeros(1, dtype=torch.int64, device="cuda")  # index into the 2 banks
        self._disp_expected = torch.zeros(2, dtype=torch.int64, device="cuda")
        self._combine_parity = torch.zeros(1, dtype=torch.int64, device="cuda")
        self._combine_expected = torch.zeros(2, dtype=torch.int64, device="cuda")
        self._reduce_expected = torch.zeros(2, dtype=torch.int64, device="cuda")
        self._sym_buffer = None  # cached so its peer-delta table stays alive with this heap

    def get_sym_buffer(self) -> SymBuffer:
        """Build (once) the SymBuffer handle; cached so its peer-delta table outlives async launches."""
        if self._sym_buffer is None:
            sym = SymBuffer(self.symm_mem.buffer_ptrs, self.rank)
            sym._offsets_device_ptr()  # materialize the delta table now, keep it alive on `sym`
            self._sym_buffer = sym
        return self._sym_buffer

    def destroy(self) -> None:
        global _CURRENT_SYMM_BUFFER
        if _CURRENT_SYMM_BUFFER is self:
            _CURRENT_SYMM_BUFFER = None
        # torch symmetric memory frees on GC; drop refs (region views alias buffer -> clear too)
        self._sym_buffer = None
        self.symm_mem = None
        self.buffer = None


_CURRENT_SYMM_BUFFER = None


def get_symm_buffer_for_mega_moe(
    group: Optional["torch.distributed.ProcessGroup"] = None,
    *,
    num_experts: Optional[int] = None,
    num_max_tokens_per_rank: Optional[int] = None,
    num_topk: Optional[int] = None,
    hidden: Optional[int] = None,
    intermediate_hidden: Optional[int] = None,
    token_dtype: torch.dtype = TOKEN_DTYPE,
) -> SymmBuffer:
    """Cached per-(group, dims, dtype) SymmBuffer; no-arg call returns the active one."""
    global _CURRENT_SYMM_BUFFER
    if group is None:
        if _CURRENT_SYMM_BUFFER is None:
            raise RuntimeError(
                "no symmetric buffer is active; call get_symm_buffer_for_mega_moe(group, ...) first"
            )
        return _CURRENT_SYMM_BUFFER

    num_max_tokens_per_rank = align(int(num_max_tokens_per_rank), TOKEN_ALIGNMENT)
    key = (
        group.size(),
        int(num_experts),
        num_max_tokens_per_rank,
        int(num_topk),
        int(hidden),
        int(intermediate_hidden),
        token_dtype,
    )
    symm = _CURRENT_SYMM_BUFFER
    if symm is None or symm.group is not group or symm.key != key:
        if symm is not None:
            symm.destroy()
        symm = SymmBuffer(
            group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            token_dtype=token_dtype,
        )
        _CURRENT_SYMM_BUFFER = symm
    return symm
