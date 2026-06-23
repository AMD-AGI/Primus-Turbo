###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Self-describing symmetric tensor for FlyDSL -- single-arg ``symm_at``.

Triton-distributed exposes ``symm_at(ptr, rank)``: pass a local symmetric pointer,
get the same offset on a peer rank, no base table threaded through. FlyDSL has no
nvshmem/rocshmem heap intrinsic, so the per-rank base table must live somewhere.
``prims.symm_at`` passes it as an extra kernel arg; here we instead EMBED it inside
the symmetric allocation as a fixed header:

    allocation = [ i64 peer_base[world] | pad -> 256B ][ payload ... ]

The layout is symmetric, so on every rank the header sits at the same offset and
holds all ranks' allocation bases. From the payload pointer alone the kernel then
recovers any peer's address:

    my_alloc   = base(payload) - 256        # header sits right before payload
    peer_alloc = header[dst_rank]           # i64 load from the embedded table
    peer_ptr   = base(payload) + (peer_alloc - my_alloc)

Host side: ``SymmTensor.empty(group, shape, dtype)`` allocates + fills the header.
Device side: ``symm_at(payload_tensor, dst_rank)`` -> raw i64 peer address; pair with
``prims.addr_buffer_resource`` (vectorized) or ``prims.addr_elem_ptr_i32`` (scalar).
"""

import flydsl.expr as fx
from flydsl.expr.buffer_ops import buffer_load

from primus_turbo.flydsl.mega.prims import addr_buffer_resource, tensor_base

# Fixed self-describing header: 256B holds up to 32 i64 peer bases. Bump for >32 ranks.
SYMM_HDR_BYTES = 256


# ---------------------------------------------------------------------------
# Device side
# ---------------------------------------------------------------------------
def symm_at(payload_tensor, dst_rank):
    """Translate the local symmetric payload base to peer ``dst_rank`` (raw i64).

    ``payload_tensor`` is a ``SymmTensor.tensor`` view (payload at offset
    ``SYMM_HDR_BYTES``); ``dst_rank`` is an fx i32. Self maps to identity. The
    embedded header before the payload supplies every peer's allocation base."""
    local = tensor_base(payload_tensor)  # local payload base (i64)
    my_alloc = local - fx.Int64(SYMM_HDR_BYTES)  # allocation base == my header base
    hdr_res = addr_buffer_resource(my_alloc, num_records_bytes=SYMM_HDR_BYTES)
    peer_alloc = buffer_load(hdr_res, dst_rank, vec_width=1, dtype=fx.T.i64())
    return local + (peer_alloc - my_alloc)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------
class SymmTensor:
    """A symmetric allocation that carries its own peer-base table in a header.

    Use ``SymmTensor.empty(group, shape, dtype)``; pass ``.tensor`` to the kernel
    and call ``symm_at(tensor, rank)`` inside it -- no separate base table needed."""

    def __init__(self, mem, shape, dtype):
        self.mem = mem
        self.rank = mem.rank
        self.world = mem.world_size
        self.shape = tuple(shape)
        self.dtype = dtype
        self._elem_off = SYMM_HDR_BYTES // dtype.itemsize

    @classmethod
    def empty(cls, group, shape, dtype):
        """Allocate a symmetric tensor of ``shape``/``dtype`` over ``group``.

        Allocation is zero-initialized (payload) with the peer-base table written
        into the header. All ranks must call this collectively."""
        import torch

        from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

        if group.size() * 8 > SYMM_HDR_BYTES:
            raise ValueError(f"world={group.size()} exceeds header capacity {SYMM_HDR_BYTES // 8}")
        if SYMM_HDR_BYTES % dtype.itemsize != 0:
            raise ValueError(f"SYMM_HDR_BYTES={SYMM_HDR_BYTES} not divisible by itemsize {dtype.itemsize}")

        numel = 1
        for s in shape:
            numel *= s
        payload_bytes = numel * dtype.itemsize
        mem = SymmetricMemory(group, SYMM_HDR_BYTES + payload_bytes)

        # write header[r] = allocation base of rank r (== mem.buffer_ptrs[r])
        hdr = mem.get_buffer(mem.rank, (mem.world_size,), torch.int64, storage_offset=0)
        hdr.copy_(torch.tensor(mem.buffer_ptrs, dtype=torch.int64, device="cuda"))
        mem.group.barrier()
        return cls(mem, shape, dtype)

    @property
    def tensor(self):
        """Local payload view (the tensor to pass to the kernel)."""
        return self.mem.get_buffer(self.rank, self.shape, self.dtype, storage_offset=self._elem_off)

    def peer_view(self, rank):
        """Host-side payload view into peer ``rank`` (for verification)."""
        return self.mem.get_buffer(rank, self.shape, self.dtype, storage_offset=self._elem_off)

    def barrier(self):
        self.mem.group.barrier()

    def destroy(self):
        self.mem.destroy()
