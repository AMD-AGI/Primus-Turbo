###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepGEMM-style ``SymBuffer`` for FlyDSL -- one delta table maps the whole arena.

Port of ``deep_gemm::layout::SymBuffer`` (sym_buffer.cuh): a symmetric allocation has
the same layout on every rank but a different base. The struct stores

    base        = this rank's arena base
    offsets[i]  = peer_base[i] - base        # precomputed per-peer delta

so ``map(ptr, dst) = ptr + offsets[dst]`` translates ANY local arena pointer to peer
``dst`` with a single add -- one table serves every sub-buffer carved from the arena
(unlike a per-tensor base table or an embedded header, cf. [[symm_tensor.py]]).

DeepGEMM passes the struct by value as ``__grid_constant__`` (offsets live in const
memory -> map is a register add). FlyDSL kernels take tensors/scalars, so we pass
``offsets`` as a small i64[world] tensor; the device ``sym_map`` does one i64 load
(L2/constant-cached) then the add. Self maps to identity (offsets[self] == 0).
"""

import flydsl.expr as fx
from flydsl.expr.buffer_ops import buffer_load

from primus_turbo.flydsl.mega.prims import tensor_base

_ALIGN = 256  # sub-buffer base alignment (matches the mega kernels' bump cursor)


def _align_up(x, a):
    return (x + a - 1) // a * a


# ---------------------------------------------------------------------------
# Device side
# ---------------------------------------------------------------------------
def sym_map(local_ptr, offsets_res, dst_rank):
    """Translate local arena pointer ``local_ptr`` (i64) to peer ``dst_rank`` (i64).

    ``offsets_res`` is a buffer resource over the SymBuffer's i64[world] delta table;
    returns ``local_ptr + offsets[dst_rank]``. Self -> identity (delta 0)."""
    delta = buffer_load(offsets_res, dst_rank, vec_width=1, dtype=fx.T.i64())
    return local_ptr + delta


def sym_map_tensor(local_tensor, offsets_res, dst_rank):
    """``sym_map`` starting from a sub-buffer tensor's base address."""
    return sym_map(tensor_base(local_tensor), offsets_res, dst_rank)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------
class SymBuffer:
    """One symmetric arena + an i64[world] peer-delta table (DeepGEMM SymBuffer).

    Carve sub-buffers with ``alloc``; pass ``.offsets`` to a FlyDSL kernel and call
    ``sym_map(ptr, offsets_res, rank)`` inside it to reach any peer's matching ptr."""

    def __init__(self, mem, arena_bytes):
        import torch

        self.mem = mem
        self.rank = mem.rank
        self.world = mem.world_size
        self.arena_bytes = arena_bytes
        self._cursor = 0  # bump allocator

        # base = this rank's arena base; offsets[i] = peer_base[i] - base
        base = mem.buffer_ptrs[self.rank]
        self.base = base
        self.offsets = torch.tensor(
            [mem.buffer_ptrs[i] - base for i in range(self.world)],
            dtype=torch.int64,
            device="cuda",
        )

    @classmethod
    def create(cls, group, arena_bytes):
        """Allocate a symmetric arena of ``arena_bytes`` over ``group`` (collective)."""
        import torch

        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        mem = SymmetricMemory(group, arena_bytes)
        mem.get_buffer(mem.rank, (arena_bytes,), torch.int8).zero_()
        mem.group.barrier()
        return cls(mem, arena_bytes)

    def alloc(self, shape, dtype):
        """Carve a 256B-aligned sub-buffer; return a local zero-copy tensor view."""

        numel = 1
        for s in shape:
            numel *= s
        off = _align_up(self._cursor, _ALIGN)
        nbytes = numel * dtype.itemsize
        if off + nbytes > self.arena_bytes:
            raise ValueError(f"arena overflow: need {off + nbytes}, have {self.arena_bytes}")
        self._cursor = off + nbytes
        assert off % dtype.itemsize == 0
        return self.mem.get_buffer(self.rank, tuple(shape), dtype, storage_offset=off // dtype.itemsize)

    def barrier(self):
        self.mem.group.barrier()

    def destroy(self):
        self.mem.destroy()
