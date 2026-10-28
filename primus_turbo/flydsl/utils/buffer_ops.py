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

"""AMD buffer load / store helpers on top of FlyDSL 0.3 buffer-descriptor pointers.

FlyDSL 0.3 dropped ``flydsl.expr.buffer_ops``. Descriptors here are built with
:func:`flydsl.expr.rocdl.make_buffer_ptr`, and :func:`create_buffer_resource` returns
the raw ``!llvm.ptr<8>`` resource (:func:`flydsl.expr.rocdl.get_buffer_rsrc`), which the
kernels also hand straight to ``rocdl.raw_ptr_buffer_*`` ops and inline asm.

Offsets of :func:`buffer_load` / :func:`buffer_store` are in elements unless
``offset_is_bytes=True``; ``mask`` moves masked-off lanes to byte offset ``0x7FFFFFFF``.

Example:
    >>> from primus_turbo.flydsl.utils import buffer_ops
    >>> rsrc = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=nbytes)
    >>> data = buffer_ops.buffer_load(rsrc, row * 4096 + col, vec_width=4)
    >>> buffer_ops.buffer_store(data, rsrc, row * 4096 + col)
"""

from typing import Optional, Union

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm
from flydsl.expr.meta import dsl_loc_tracing

__all__ = [
    "create_llvm_ptr",
    "get_element_ptr",
    "create_buffer_resource",
    "create_buffer_resource_from_addr",
    "buffer_load",
    "buffer_store",
    "extract_base_index",
]

_MAX_NUM_RECORDS = 0xFFFFFFFF
_MASKED_OFFSET = 0x7FFFFFFF


def _ir(value) -> ir.Value:
    """Raw ``ir.Value`` of a DSL value, a Python scalar or a single-result op."""
    if isinstance(value, ir.OpView):
        value = value.result
    return fx.as_ir_value(value)


def _i32_type() -> ir.Type:
    return ir.IntegerType.get_signless(32)


def _i64_type() -> ir.Type:
    return ir.IntegerType.get_signless(64)


def _const_i32(value: int) -> ir.Value:
    """i32 constant; values in ``[2**31, 2**32)`` wrap to their two's-complement bits."""
    if value > 0x7FFFFFFF:
        value -= 2**32
    return arith.constant(_i32_type(), value)


def _as_i32(value) -> ir.Value:
    """An offset as i32: a Python int, an i32 value, or an index value."""
    if isinstance(value, int):
        return _const_i32(value)
    value = _ir(value)
    if not isinstance(value.type, ir.IntegerType) or value.type.width != 32:
        value = arith.index_cast(_i32_type(), value)
    return value


def _as_i64(value) -> ir.Value:
    """An index or integer value as i64 (narrower integers are sign-extended)."""
    value = _ir(value)
    if isinstance(value.type, ir.IndexType):
        return arith.index_cast(_i64_type(), value)
    if value.type.width < 64:
        return arith.extsi(_i64_type(), value)
    return value


def _as_num_records(num_records_bytes) -> Optional[fx.Int64]:
    """Descriptor byte count for ``make_buffer_ptr``; ``None`` keeps its ``0xFFFFFFFF`` default."""
    if num_records_bytes is None:
        return None
    if isinstance(num_records_bytes, int):
        return fx.Int64(max(0, min(num_records_bytes, _MAX_NUM_RECORDS)))
    return fx.Int64(_as_i64(num_records_bytes))


def _num_records_from_memref_type(memref_val) -> Optional[int]:
    """Byte size of a statically shaped builtin memref, else ``None``."""
    try:
        mt = ir.MemRefType(_ir(memref_val).type)
    except (TypeError, ValueError):
        return None
    if any(int(d) < 0 for d in mt.shape):
        return None
    elem_bits = getattr(mt.element_type, "width", None)
    if not elem_bits or elem_bits < 8:
        return None
    num_elems = 1
    for d in mt.shape:
        num_elems *= int(d)
    return num_elems * (elem_bits // 8)


@dsl_loc_tracing
def _buffer_rsrc(global_ptr, num_records_bytes=None, base_byte_offset=None) -> ir.Value:
    """``!llvm.ptr<8>`` resource of a global pointer, optionally moved by a byte offset.

    A runtime offset is widened to ``index`` (64-bit) before the pointer add: per-batch
    bases can exceed 2**31 bytes.
    """
    if base_byte_offset is not None:
        if not isinstance(base_byte_offset, int):
            base_byte_offset = _ir(base_byte_offset)
            if not isinstance(base_byte_offset.type, ir.IndexType):
                base_byte_offset = arith.index_cast(ir.IndexType.get(), base_byte_offset)
        global_ptr = fx.recast_iter(fx.Int8, global_ptr) + base_byte_offset
    buf_ptr = fx.rocdl.make_buffer_ptr(global_ptr, num_records_bytes=_as_num_records(num_records_bytes))
    return fx.rocdl.get_buffer_rsrc(buf_ptr)


def _aux_attr(cache_modifier) -> ir.IntegerAttr:
    """Cache-policy bits for the raw buffer ops, which take ``aux`` as an i32 attribute.

    It is an immediate in the instruction encoding, so it must be a Python int.
    """
    if not isinstance(cache_modifier, int):
        raise TypeError(f"cache_modifier must be a Python int, got {type(cache_modifier).__name__}")
    return ir.IntegerAttr.get(_i32_type(), cache_modifier)


def _soffset(soffset_bytes) -> ir.Value:
    return _const_i32(0) if soffset_bytes is None else _as_i32(soffset_bytes)


def _scale_and_mask(offset: ir.Value, element_bytes: Optional[int], mask) -> ir.Value:
    if element_bytes is not None:
        offset = arith.muli(offset, _const_i32(element_bytes))
    if mask is not None:
        offset = arith.select(_ir(mask), offset, _const_i32(_MASKED_OFFSET))
    return offset


@dsl_loc_tracing
def _ptr8_to_v4i32(ptr8_val) -> ir.Value:
    """Reinterpret a buffer resource (!llvm.ptr<8>) as a <4 x i32> vector.

    Required by the scalar ``s.buffer.load`` intrinsic, whose resource operand is
    a v4i32 rather than the opaque buffer pointer used by the vector path.
    """
    i128_ty = ir.IntegerType.get_signless(128)
    v4i32_ty = ir.VectorType.get([4], _i32_type())
    return llvm.bitcast(v4i32_ty, llvm.ptrtoint(i128_ty, _ir(ptr8_val)))


@dsl_loc_tracing
def create_llvm_ptr(value, address_space: int = 0) -> ir.Value:
    """Create an LLVM pointer from an integer or index value."""
    value = _ir(value)
    if isinstance(value.type, ir.IndexType):
        value = arith.index_cast(_i64_type(), value)
    return llvm.inttoptr(ir.Type.parse(f"!llvm.ptr<{address_space}>"), value)


@dsl_loc_tracing
def extract_base_index(tensor, address_space: int = 1) -> ir.Value:
    """Extract the base address of a fly.memref as an index value.

    Inverse of :func:`create_llvm_ptr` (index -> ptr). Useful when ISA
    requires a raw pointer instead of a buffer resource descriptor
    (e.g. global_atomic_pk_add_bf16 on gfx942).
    """
    from flydsl._mlir.dialects import fly as _fly
    from flydsl._mlir.dialects import memref as _memref

    raw = _ir(tensor)
    if isinstance(raw.type, ir.MemRefType):
        return _memref.extract_aligned_pointer_as_index(raw)

    ptr = _fly.extract_aligned_pointer_as_index(ir.Type.parse(f"!llvm.ptr<{address_space}>"), raw)
    return arith.index_cast(ir.IndexType.get(), llvm.ptrtoint(_i64_type(), ptr))


@dsl_loc_tracing
def get_element_ptr(
    base_ptr,
    byte_offset: Union[int, ir.Value, None] = None,
    static_byte_offset: int = 0,
    elem_type: Optional[ir.Type] = None,
    no_wrap_flags=None,
) -> ir.Value:
    """Build an LLVM GEP from a base pointer plus byte offsets."""
    gep_dynamic_index_sentinel = -(2**31)

    base_ptr = _ir(base_ptr)
    if not isinstance(static_byte_offset, int):
        raise TypeError(f"static_byte_offset must be int, got {type(static_byte_offset).__name__}")
    if elem_type is None:
        elem_type = ir.IntegerType.get_signless(8)
    elif callable(elem_type):
        elem_type = elem_type()

    if byte_offset is None:
        dynamic_indices = []
        raw_constant_indices = [static_byte_offset]
    elif isinstance(byte_offset, int):
        dynamic_indices = []
        raw_constant_indices = [byte_offset + static_byte_offset]
    else:
        offset_val = _ir(byte_offset)
        if isinstance(offset_val.type, ir.IndexType):
            offset_val = arith.index_cast(_i64_type(), offset_val)
        elif not isinstance(offset_val.type, ir.IntegerType):
            raise TypeError(
                f"byte_offset must be int, index, or integer-typed MLIR value; got {offset_val.type}"
            )
        if static_byte_offset != 0:
            offset_val = arith.addi(offset_val, arith.constant(offset_val.type, static_byte_offset))
        dynamic_indices = [offset_val]
        raw_constant_indices = [gep_dynamic_index_sentinel]

    return llvm.GEPOp(
        base_ptr.type,
        base_ptr,
        dynamic_indices,
        raw_constant_indices,
        elem_type,
        no_wrap_flags,
    ).result


@dsl_loc_tracing
def create_buffer_resource_from_addr(
    addr_i64: ir.Value,
    *,
    num_records_bytes: Optional[Union[int, ir.Value]] = None,
) -> ir.Value:
    """Create AMD buffer resource descriptor from a raw i64 device address.

    Useful when working with runtime pointer arrays (e.g. IPC-mapped addresses
    or device-side pointer tables) where no fly.memref is available.
    The full address is encoded as the buffer base; callers should pass
    byte offset 0 to buffer_load / buffer_store.

    Args:
        addr_i64: Raw 64-bit device address (i64 MLIR value).
        num_records_bytes: Optional buffer size in bytes for hardware OOB checking.

    Returns:
        ROCDL buffer resource descriptor (!llvm.ptr<8>).

    Example:
        >>> rsrc = create_buffer_resource_from_addr(raw_addr_i64)
        >>> data = buffer_load(rsrc, 0, vec_width=4, dtype=fx.Int32)
    """
    ptr_ty = fx.PointerType.get(elem_ty=fx.Int8.ir_type, address_space=fx.AddressSpace.Global)
    base_ptr = fx.inttoptr(ptr_ty, fx.Int64(_ir(addr_i64)))
    return _buffer_rsrc(base_ptr, num_records_bytes)


@dsl_loc_tracing
def create_buffer_resource(
    memref_val: ir.Value,
    stride: int = 0,
    max_size: bool = True,
    *,
    num_records_bytes: Optional[Union[int, ir.Value]] = None,
    base_byte_offset: Optional[Union[int, ir.Value]] = None,
) -> ir.Value:
    """Create AMD buffer resource descriptor from memref.

    Args:
        memref_val: Memref value
        stride: Buffer stride; only 0 (contiguous) is supported
        max_size: Use the maximum buffer size (``0xFFFFFFFF``) when ``num_records_bytes``
            is not given; otherwise the static memref size, if known
        num_records_bytes: Override buffer size in bytes.
        base_byte_offset: Optional byte offset added to the descriptor base pointer.

    Returns:
        ROCDL buffer resource descriptor (!llvm.ptr<8>)

    Example:
        >>> rsrc = create_buffer_resource(A)
        >>> data = buffer_load(rsrc, offset)
    """
    if stride != 0:
        raise ValueError(f"create_buffer_resource: only stride=0 (contiguous) is supported, got {stride}")
    if num_records_bytes is None and not max_size:
        num_records_bytes = _num_records_from_memref_type(memref_val)
    return _buffer_rsrc(fx.get_iter(memref_val), num_records_bytes, base_byte_offset)


@dsl_loc_tracing
def buffer_load(
    rsrc: ir.Value,
    offset: ir.Value,
    vec_width: int = 4,
    dtype=None,
    mask: Optional[ir.Value] = None,
    cache_modifier: int = 0,
    soffset_bytes: Optional[Union[int, ir.Value]] = None,
    is_scalar: bool = False,
) -> ir.Value:
    """AMD buffer load operation.

    Load data from global memory using buffer descriptor and offset.
    Uses hardware-level bounds checking and vectorization.

    Args:
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 or index)
        vec_width: Vector width (1, 2, or 4)
        dtype: Element data type (None for f32; an ir.Type or a DSL Numeric class)
        mask: Optional mask for predicated load (i1 type)
        cache_modifier: Cache control flags (0 for default)
        soffset_bytes: Optional scalar offset (in BYTES) added by the buffer instruction (soffset).
                      Use this to fold small constant deltas into the instruction instead of emitting
                      extra VGPR address arithmetic.
        is_scalar: Emit a uniform/SGPR scalar load (llvm.amdgcn.s.buffer.load) instead of the
                      vector buffer load. Use only for wave-uniform addresses to route through the
                      SMEM cache and land the result directly in SGPRs. Restricted to vec_width 1 or 4;
                      dtype is forced to i32 (the result is raw i32 dwords). mask and soffset_bytes
                      are not supported in this mode and raise ValueError if provided.

    Returns:
        Loaded data (scalar or vector depending on vec_width)

    Example:
        >>> data = buffer_load(rsrc, offset, vec_width=4)
        >>> data = buffer_load(rsrc, offset, vec_width=4, mask=valid)
    """
    if is_scalar:
        if vec_width not in (1, 4):
            raise ValueError(f"buffer_load(is_scalar=True): unsupported vec_width={vec_width}")
        if mask is not None or soffset_bytes is not None:
            raise ValueError("buffer_load(is_scalar=True) does not support mask or soffset_bytes")
        dtype = _i32_type()
    elif dtype is None:
        dtype = ir.F32Type.get()
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type

    offset = _scale_and_mask(_as_i32(offset), dtype.width // 8, mask)
    result_type = dtype if vec_width == 1 else ir.VectorType.get([vec_width], dtype)

    if is_scalar:
        suffix = "i32" if vec_width == 1 else "v4i32"
        return llvm.call_intrinsic(
            result_type,
            f"llvm.amdgcn.s.buffer.load.{suffix}",
            [_ptr8_to_v4i32(rsrc), offset, _const_i32(cache_modifier)],
            [],
            [],
        )

    return fx.rocdl.raw_ptr_buffer_load(
        result_type, _ir(rsrc), offset, _soffset(soffset_bytes), aux=_aux_attr(cache_modifier)
    )


@dsl_loc_tracing
def buffer_store(
    data: ir.Value,
    rsrc: ir.Value,
    offset: ir.Value,
    mask: Optional[ir.Value] = None,
    cache_modifier: int = 0,
    *,
    soffset_bytes: Optional[Union[int, ir.Value]] = None,
    offset_is_bytes: bool = False,
):
    """AMD buffer store operation.

    Store data to global memory using buffer descriptor and offset.

    Args:
        data: Data to store (scalar or vector)
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 or index), or in bytes with ``offset_is_bytes=True``
        mask: Optional mask for predicated store (i1 type)
        cache_modifier: Cache control flags (0 for default)
        soffset_bytes: Optional scalar offset (in BYTES) added by the buffer instruction (soffset).
        offset_is_bytes: ``offset`` is already a byte offset; skip the element-size scaling.

    Example:
        >>> buffer_store(data, rsrc, offset)
        >>> buffer_store(data, rsrc, offset, mask=valid)
    """
    data = _ir(data)
    element_bytes = None
    if not offset_is_bytes:
        element_type = getattr(data.type, "element_type", data.type)
        element_bytes = element_type.width // 8
    offset = _scale_and_mask(_as_i32(offset), element_bytes, mask)
    fx.rocdl.raw_ptr_buffer_store(
        data, _ir(rsrc), offset, _soffset(soffset_bytes), aux=_aux_attr(cache_modifier)
    )
