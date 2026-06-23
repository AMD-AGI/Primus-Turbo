###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Cross-rank symmetric-memory address primitives (FlyDSL).

A symmetric allocation has the SAME layout on every rank but a different base
address. ``symm_at`` translates a LOCAL heap address into the same offset on a peer
rank -- the FlyDSL analog of ``triton_dist.language.symm_at(ptr, rank)``. FlyDSL has
no nvshmem/rocshmem heap intrinsic, so the per-rank base table is supplied explicitly
(from torch ``SymmetricMemory.buffer_ptrs`` / ``signal_pad_ptrs``):

    peer_addr = local_addr + (peer_base[rank] - my_base)

This is a pure base-offset translation (intra-node HIP-IPC P2P) and is valid only
WITHIN one symmetric allocation -- pass the base table of the heap the address lives
in (cached buffer vs uncached signal pad are separate heaps -> separate tables).

Returns raw i64 addresses; pair with ``addr_buffer_resource`` (vectorized buffer
load/store) or ``addr_elem_ptr_i32`` (scalar / atomic) to actually touch memory.
"""

import flydsl.expr as fx
from flydsl.expr.buffer_ops import (
    _unwrap_value,
    buffer_load,
    create_buffer_resource_from_addr,
    create_llvm_ptr,
    extract_base_index,
    get_element_ptr,
)

_I4 = 4  # int32 byte stride
_GLOBAL = 1  # LLVM global address space


def heap_base(peer_base_res, rank):
    """Peer ``rank``'s symmetric-heap base address (i64) from the [world] base table."""
    return buffer_load(peer_base_res, rank, vec_width=1, dtype=fx.T.i64())


def tensor_base(tensor):
    """i64 base address (data_ptr) of a local symmetric tensor.

    ``extract_base_index`` yields an ``index``; cast to i64 so it matches the base
    table loads in address arithmetic (MLIR forbids mixed index/i64 operands)."""
    addr = extract_base_index(tensor, address_space=_GLOBAL)
    return fx.arith.ArithValue(fx.arith.index_cast(fx.T.i64(), addr), signed=True)


def symm_at(local_addr, peer_base_res, my_base, rank):
    """Translate local symmetric address ``local_addr`` to peer ``rank`` (i64).

    ``peer_addr = local_addr + (peer_base[rank] - my_base)``. ``peer_base_res`` is a
    buffer resource over the [world] i64 table of every rank's heap base; ``my_base``
    is this rank's base (i64, e.g. ``heap_base(peer_base_res, fx.Int32(rank_self))``).
    Self maps to identity (``rank == rank_self`` -> ``peer_base - my_base == 0``)."""
    return local_addr + (heap_base(peer_base_res, rank) - my_base)


def symm_at_offset(peer_base_res, rank, byte_offset):
    """Peer ``rank`` address (i64) of a sub-buffer at ``byte_offset`` in the heap.

    Shortcut for ``symm_at`` when the offset is known: ``peer_base[rank] + byte_offset``
    (== translating a local addr ``my_base + byte_offset``). ``byte_offset`` may be a
    python int (folded to an i64 constant) or an fx i64 value."""
    if isinstance(byte_offset, int):
        byte_offset = fx.Int64(byte_offset)
    return heap_base(peer_base_res, rank) + byte_offset


def addr_buffer_resource(addr_i64, num_records_bytes):
    """Buffer resource over a raw i64 address (for vectorized buffer_load/store)."""
    return create_buffer_resource_from_addr(addr_i64, num_records_bytes=num_records_bytes)


def addr_elem_ptr_i32(addr_i64, idx):
    """LLVM global ptr to the int32 element at ``(addr_i64)[idx]`` (scalar / atomic)."""
    base = create_llvm_ptr(_unwrap_value(addr_i64), _GLOBAL)
    byte_off = _unwrap_value(idx * fx.Int32(_I4))
    return get_element_ptr(base, byte_offset=byte_off, elem_type=fx.T.i8())
