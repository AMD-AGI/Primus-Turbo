###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple, Union

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm
from flydsl._mlir.extras import types as T
from flydsl.expr import range_constexpr
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.utils.buffer_ops import (
    _create_i32_constant,
    _create_i64_constant,
    _unwrap_value,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    create_llvm_ptr,
    get_element_ptr,
)

_WARP = 64
_COPY_VEC_I32 = 4  # 4 i32 = 16 B (b128) per lane per step
_I32_BYTES = 4  # word stride: i32-word offset -> byte address

# Watchdog budget for cross-rank / grid spin loops (realtime clock cycles).
SPIN_TIMEOUT_CYCLES = 3_000_000_000


def read_clock() -> ArithValue:
    # Realtime counter for spin-wait watchdogs; unsigned so deltas compare right.
    op = llvm.inline_asm(T.i64(), [], "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", has_side_effects=True)
    return ArithValue(op, signed=False)


def spin_timed_out(spin_start: ArithValue, timeout: int = SPIN_TIMEOUT_CYCLES) -> ArithValue:
    # Pure predicate (raw cmpi) for the watchdog `if`; the loop must stay inline for the AST rewriter (spin_start is loop-carried).
    return (read_clock() - spin_start) > fx.Int64(timeout)


def cast(val: Union[int, ArithValue], dtype) -> ArithValue:
    # Cast a scalar to dtype, picking the right widen/narrow/convert op.
    if hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type
    signed = getattr(val, "signed", True)
    src = _as_value(val)  # bare python int -> i32 constant
    src_ty = src.type
    if src_ty == dtype:
        return ArithValue(src, signed=signed)

    src_int, dst_int = isinstance(src_ty, ir.IntegerType), isinstance(dtype, ir.IntegerType)
    src_idx, dst_idx = isinstance(src_ty, ir.IndexType), isinstance(dtype, ir.IndexType)
    src_flt, dst_flt = isinstance(src_ty, ir.FloatType), isinstance(dtype, ir.FloatType)

    if src_idx or dst_idx:
        op = std_arith.IndexCastOp(dtype, src)
    elif src_int and dst_int:
        if dtype.width > src_ty.width:
            op = (std_arith.ExtSIOp if signed else std_arith.ExtUIOp)(dtype, src)
        else:
            op = std_arith.TruncIOp(dtype, src)
    elif src_flt and dst_flt:
        op = (std_arith.ExtFOp if dtype.width > src_ty.width else std_arith.TruncFOp)(dtype, src)
    elif src_int and dst_flt:
        op = (std_arith.SIToFPOp if signed else std_arith.UIToFPOp)(dtype, src)
    elif src_flt and dst_int:
        op = (std_arith.FPToSIOp if signed else std_arith.FPToUIOp)(dtype, src)
    else:
        raise ValueError(f"cannot cast {src_ty} to {dtype}")
    return ArithValue(op.result, signed=signed)


_ADDR_SPACES = {"global": 1, "gmem": 1, "lds": 3, "shared": 3, "smem": 3}
_ATOMIC_ORDERINGS = {
    "relaxed": llvm.AtomicOrdering.monotonic,
    "acquire": llvm.AtomicOrdering.acquire,
    "release": llvm.AtomicOrdering.release,
    "acq_rel": llvm.AtomicOrdering.acq_rel,
    "seq_cst": llvm.AtomicOrdering.seq_cst,
}


def _unwrap_scope(scope: Optional[str]) -> Optional[str]:
    if scope == "sys":
        return None
    return scope


def _unwrap_space(space: Union[int, str]) -> int:
    if isinstance(space, int):
        return space
    try:
        return _ADDR_SPACES[space]
    except KeyError:
        raise ValueError(f"bad space {space!r}; expected one of {sorted(_ADDR_SPACES)} or an int") from None


def _unwrap_order(order: Optional[str]) -> llvm.AtomicOrdering:
    if order is None:
        return llvm.AtomicOrdering.monotonic
    try:
        return _ATOMIC_ORDERINGS[order]
    except KeyError:
        raise ValueError(
            f"bad order {order!r}; expected None or one of {sorted(_ATOMIC_ORDERINGS)}"
        ) from None


def _as_value(v: Union[int, ArithValue]) -> ir.Value:
    # Coerce python int / ArithValue / raw ir value to a raw ir value (bare int -> i32; pass typed for i64).
    if isinstance(v, int):
        v = _create_i32_constant(v)
    elif hasattr(v, "ir_value"):
        v = v.ir_value()
    return _unwrap_value(v)


def memory_fence(order: Optional[str] = None, scope: Optional[str] = None) -> None:
    order_enum = _unwrap_order(order)
    llvm.fence(order_enum, syncscope=_unwrap_scope("agent" if scope is None else scope))


def addr_buffer_resource(addr_i64: ArithValue, num_records_bytes: int) -> ArithValue:
    return create_buffer_resource_from_addr(addr_i64, num_records_bytes=num_records_bytes)


def elem_ptr(
    base: Union[int, ArithValue],
    idx: Union[int, ArithValue],
    space: Union[int, str],
    elem_bytes: int = 4,
) -> ir.Value:
    ptr = create_llvm_ptr(_unwrap_value(base), _unwrap_space(space))
    idx_val = _unwrap_value(idx)
    if isinstance(idx_val.type, ir.IndexType):
        idx_val = _unwrap_value(std_arith.IndexCastOp(T.i64(), idx_val).result)
    elif isinstance(idx_val.type, ir.IntegerType) and idx_val.type.width < 64:
        idx_val = _unwrap_value(std_arith.ExtSIOp(T.i64(), idx_val).result)
    byte_off = _unwrap_value(std_arith.MulIOp(idx_val, _create_i64_constant(elem_bytes)).result)
    return get_element_ptr(ptr, byte_offset=byte_off, elem_type=T.i8())


def addr_elem_ptr_i32(addr_i64: Union[int, ArithValue], idx: Union[int, ArithValue]) -> ir.Value:
    return elem_ptr(addr_i64, idx, "global")


def atomic_add(
    base: Union[int, ArithValue],
    offset: Union[int, ArithValue],
    val: Union[int, ArithValue],
    scope: str = "agent",
    space: Union[int, str] = "global",
    order: str = "relaxed",
) -> ArithValue:
    val = _as_value(val)
    elem_bytes = val.type.width // 8
    ptr = elem_ptr(base, offset, space, elem_bytes)
    res = llvm.atomicrmw(
        llvm.AtomicBinOp.add,
        ptr,
        val,
        _unwrap_order(order),
        syncscope=_unwrap_scope(scope),
        alignment=elem_bytes,
    )
    return ArithValue(res, signed=True)


def ld(
    base: Union[int, ArithValue],
    offset: Union[int, ArithValue],
    *,
    scope: str = "agent",
    space: Union[int, str] = "global",
    order: str = "relaxed",
    dtype: Optional[object] = None,
) -> ArithValue:
    if dtype is None:
        dtype = T.i32()
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type
    elem_bytes = dtype.width // 8
    ptr = elem_ptr(base, offset, space, elem_bytes)
    op = llvm.LoadOp(
        dtype,
        ptr,
        ordering=_unwrap_order(order),
        syncscope=_unwrap_scope(scope),
        alignment=elem_bytes,
    )
    return ArithValue(op.result, signed=True)


def st(
    base: Union[int, ArithValue],
    offset: Union[int, ArithValue],
    val: Union[int, ArithValue],
    *,
    scope: str = "agent",
    space: Union[int, str] = "global",
    order: str = "relaxed",
) -> None:
    val = _as_value(val)
    elem_bytes = val.type.width // 8
    ptr = elem_ptr(base, offset, space, elem_bytes)
    llvm.StoreOp(
        val, ptr, ordering=_unwrap_order(order), syncscope=_unwrap_scope(scope), alignment=elem_bytes
    )


def copy_warp(
    dst: Union[int, ArithValue],
    src: Union[int, ArithValue],
    nbytes: int,
    dst_off: Union[int, ArithValue] = 0,
    src_off: Union[int, ArithValue] = 0,
    load_cache_modifier: int = 0,
    store_cache_modifier: int = 0,
) -> None:
    def _addr_i64(addr: Union[int, ArithValue]) -> ArithValue:
        if isinstance(addr, int):
            return fx.Int64(addr)
        v = _unwrap_value(addr)
        if isinstance(v.type, ir.IndexType):
            v = std_arith.IndexCastOp(T.i64(), v).result
        elif isinstance(v.type, ir.IntegerType) and v.type.width < 64:
            v = std_arith.ExtSIOp(T.i64(), v).result
        return ArithValue(v, signed=True)

    def _copy_operand(
        operand: Union[int, ArithValue], word_off: Union[int, ArithValue], nbytes: int
    ) -> Tuple[ArithValue, ArithValue]:
        if "ptr" in str(_unwrap_value(operand).type):
            return operand, fx.Int32(word_off) if isinstance(word_off, int) else word_off
        base = _addr_i64(operand) + _addr_i64(word_off) * fx.Int64(_I32_BYTES)
        return create_buffer_resource_from_addr(base, num_records_bytes=nbytes), fx.Int32(0)

    assert nbytes % (_WARP * 16) == 0, "copy_warp nbytes must be a multiple of 1024"
    src, src_off = _copy_operand(src, src_off, nbytes)
    dst, dst_off = _copy_operand(dst, dst_off, nbytes)
    lane_off = (fx.thread_idx.x % fx.Int32(_WARP)) * fx.Int32(_COPY_VEC_I32)
    cols = _WARP * _COPY_VEC_I32
    offs = [fx.Int32(c * cols) + lane_off for c in range_constexpr(nbytes // 4 // cols)]
    vals = [
        buffer_load(
            src, src_off + o, vec_width=_COPY_VEC_I32, dtype=T.i32(), cache_modifier=load_cache_modifier
        )
        for o in offs
    ]
    for o, v in zip(offs, vals):
        buffer_store(v, dst, dst_off + o, cache_modifier=store_cache_modifier)
