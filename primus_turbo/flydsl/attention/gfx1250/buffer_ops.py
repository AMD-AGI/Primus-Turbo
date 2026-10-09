###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

# FlyDSL's kernels/common/buffer_ops.py, as vendored by aiter
# (https://github.com/ROCm/aiter, commit 6963ae9d).

"""AMD buffer-resource stores and an LLVM GEP helper for the gfx1250 attention forward.

flydsl moved these from ``flydsl.expr.buffer_ops`` to its repo-level ``kernels/common/``,
which its wheel does not ship, so a trimmed copy lives here. It keeps only what the
forward kernel calls: ``create_buffer_resource`` + ``buffer_store`` (the LSE store) and
``get_element_ptr`` (a byte-offset GEP on an LDS pointer, so ``ds_load`` / ``ds_store``
fold the offset into the immediate).

The ops are built with the raw ``flydsl._mlir`` dialects (``llvm.getelementptr``,
``rocdl.make.buffer.rsrc``, ``rocdl.raw.ptr.buffer.store``). FlyDSL has no op for the GEP;
the stores could use ``fx.rocdl.make_buffer_tensor`` + a copy atom, as the backward does,
but keep the builders of the tuned kernels (a code-generation change to validate on its own).

Buffer instructions are an AMD hardware feature (buffer resource descriptor plus ROCDL
intrinsics) that give out-of-bounds protection; plain memref stores are not a substitute.

The flydsl 0.3.4.1 upgrade of Primus-Turbo adds a fuller copy of the same FlyDSL module,
``primus_turbo/flydsl/utils/buffer_ops.py``. The forward keeps this trimmed copy so that its
generated code stays the verified code; moving it to the shared copy is a separate change
that needs that check repeated.

Example:
    >>> rsrc = buffer_ops.create_buffer_resource(ptr_O, num_records_bytes=nbytes)
    >>> buffer_ops.buffer_store(data, rsrc, byte_off, offset_is_bytes=True)
"""

from __future__ import annotations

from .flydsl_version import require_flydsl

require_flydsl()

import flydsl.expr as fx

# UNSTABLE(gfx1250): raw MLIR dialects -- the LDS GEP has no FlyDSL op, and the buffer rsrc / store
# keep the raw builders of the tuned kernels.
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, rocdl
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.runtime.device import is_rdna_arch

# Largest num_records the 32-bit descriptor field holds; Python-int sizes are clamped to it.
_MAX_NUM_RECORDS = 0xFFFFFFFF


def _get_buffer_flags(arch=None):
    """Get AMD buffer resource descriptor (V#) flags word (bits 127:96).

    Constructs the 32-bit flags field for rocdl.make.buffer.rsrc, following the
    same logic as LLVM's AMDGPUToROCDL makeBufferRsrc():
      https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp

    Bit layout (common to all architectures):
      bits [11:0]  - DST_SEL: ignored by raw buffer intrinsics
      bits [14:12] - DATA_FORMAT: must be nonzero, 7 = float
      bits [18:15] - NUM_FORMAT:  must be nonzero, 4 = 32-bit
      bit  [19]    - In nested heap (0)
      bit  [20]    - Behavior on unmap (0 = return 0 / ignore)
      bits [22:21] - Index stride for swizzles (0)
      bit  [23]    - Add thread ID (0)
      bit  [24]    - Reserved: must be 1 on RDNA, 0 on CDNA
      bits [26:25] - Reserved (0)
      bit  [27]    - Non-volatile (CDNA only, 0)
      bits [29:28] - OOB_SELECT (RDNA only): 0=structured, 2=none, 3=check offset
      bits [31:30] - Type (must be 0)

    CDNA (gfx9xx):    (7 << 12) | (4 << 15)                         = 0x27000
    RDNA (gfx10+):    (7 << 12) | (4 << 15) | (1 << 24) | (2 << 28) = 0x21027000
      - bit 24 set to 1 (required on RDNA)
      - OOB_SELECT=2 (no bounds checking, matching LLVM boundsCheck=false)

    is_rdna_arch("gfx1250") is False, so gfx1250 takes the CDNA value.
    """
    import os

    if arch is None:
        arch = os.environ.get("FLYDSL_GPU_ARCH")
    flags = (7 << 12) | (4 << 15)
    if is_rdna_arch(arch):
        flags |= 1 << 24  # reserved bit, must be 1 on RDNA
        flags |= 2 << 28  # OOB_SELECT = 2 (no bounds checking)
    return flags


__all__ = [
    "buffer_store",
    "create_buffer_resource",
    "get_element_ptr",
]


@dsl_loc_tracing
def get_element_ptr(base_ptr, static_byte_offset: int = 0) -> ir.Value:
    """``llvm.getelementptr i8, base_ptr, static_byte_offset`` (a compile-time byte offset)."""
    if not isinstance(static_byte_offset, int):
        raise TypeError(f"static_byte_offset must be int, got {type(static_byte_offset).__name__}")
    base_ptr = fx.as_ir_value(base_ptr)
    return llvm.GEPOp(
        base_ptr.type,
        base_ptr,
        [],
        [static_byte_offset],
        fx.Int8.ir_type,
        None,
    ).result


@dsl_loc_tracing
def create_buffer_resource(memref_val, *, num_records_bytes: int | ir.Value) -> ir.Value:
    """AMD buffer resource descriptor (``!llvm.ptr<8>``) over ``memref_val``.

    ``num_records_bytes`` is the buffer size in BYTES used by the hardware OOB check: a
    Python int (clamped to [0, 0xFFFFFFFF]) or an integer value (widened to i64).
    """
    raw_val = fx.as_ir_value(memref_val)
    from flydsl._mlir.dialects import fly as _fly

    ptr_type = ir.Type.parse("!llvm.ptr")
    base_ptr = _fly.extract_aligned_pointer_as_index(ptr_type, raw_val)

    flags = fx.Int32(_get_buffer_flags()).ir_value()
    stride_val = fx.Int16(0).ir_value()
    if isinstance(num_records_bytes, int):
        # Descriptor uses i32 bytes; clamp to the representable range.
        nbytes = min(max(0, num_records_bytes), _MAX_NUM_RECORDS)
        num_records = fx.Int64(nbytes).ir_value()
    else:
        num_records = fx.Int64(fx.as_ir_value(num_records_bytes)).ir_value()

    rsrc_type = ir.Type.parse("!llvm.ptr<8>")
    return rocdl.MakeBufferRsrcOp(rsrc_type, base_ptr, stride_val, num_records, flags).result


@dsl_loc_tracing
def buffer_store(
    data: ir.Value,
    rsrc: ir.Value,
    offset: ir.Value,
    mask: ir.Value | None = None,
    cache_modifier: int = 0,
    *,
    soffset_bytes: int | ir.Value | None = None,
    offset_is_bytes: bool = False,
):
    """AMD buffer store operation.

    Store data to global memory using buffer descriptor and offset.

    Args:
        data: Data to store (scalar or vector)
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 type)
        mask: Optional mask for predicated store (i1 type)
        cache_modifier: Cache control flags (0 for default)

    Example:
        >>> buffer_store(data, rsrc, offset)
        >>>
        >>> # Store with mask
        >>> buffer_store(data, rsrc, offset, mask=valid)
    """
    data = fx.as_ir_value(data)
    rsrc = fx.as_ir_value(rsrc)
    offset = fx.Int32(fx.as_ir_value(offset))

    # IMPORTANT: RawPtrBufferStoreOp offset is in BYTES.
    # For backward compat, `buffer_store()` accepts element offsets by default
    # and scales them to bytes. Set `offset_is_bytes=True` to skip scaling.
    if not offset_is_bytes:
        # Get element size from data type
        data_type = data.type
        if hasattr(data_type, "element_type"):  # Vector type
            element_type = data_type.element_type
        else:  # Scalar type
            element_type = data_type
        element_bytes = element_type.width // 8
        offset = offset * element_bytes

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        offset = fx.Boolean(fx.as_ir_value(mask)).select(offset, 0x7FFFFFFF)

    # Create instruction offset (soffset) and aux flags
    soffset = fx.Int32(0 if soffset_bytes is None else fx.as_ir_value(soffset_bytes)).ir_value()
    aux = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_modifier)  # cache-policy bits

    # Emit buffer store
    rocdl.RawPtrBufferStoreOp(
        data,
        rsrc,
        offset.ir_value(),
        soffset,
        aux=aux,
    )
