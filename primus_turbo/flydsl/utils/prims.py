###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

import math as host_math
from typing import Optional, Tuple, Union

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.buffer_ops import (
    _create_i32_constant,
    _create_i64_constant,
    _unwrap_value,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    create_llvm_ptr,
    get_element_ptr,
)
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import ArithValue

_WARP = 64
_COPY_VEC_I32 = 4  # 4 i32 = 16 B (b128) per lane per step
_I32_BYTES = 4  # word stride: i32-word offset -> byte address

LOG2E = host_math.log2(host_math.e)  # folds a natural exp into exp2

# Watchdog budget for cross-rank / grid spin loops (realtime clock cycles).
SPIN_TIMEOUT_CYCLES = 3_000_000_000


def read_clock() -> fx.ArithValue:
    # Realtime counter for spin-wait watchdogs; unsigned so deltas compare right.
    op = llvm.inline_asm(
        fx.T.i64(), [], "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", has_side_effects=True
    )
    return fx.arith.ArithValue(op, signed=False)


def spin_timed_out(spin_start: fx.ArithValue, timeout: int = SPIN_TIMEOUT_CYCLES) -> fx.ArithValue:
    # Pure predicate (raw cmpi) for the watchdog `if`; the loop must stay inline for the AST rewriter (spin_start is loop-carried).
    return (read_clock() - spin_start) > fx.Int64(timeout)


def cast(val: Union[int, fx.ArithValue], dtype) -> fx.ArithValue:
    # Cast a scalar to dtype, picking the right widen/narrow/convert op.
    if hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type
    signed = getattr(val, "signed", True)
    src = _as_value(val)  # bare python int -> i32 constant
    src_ty = src.type
    if src_ty == dtype:
        return fx.arith.ArithValue(src, signed=signed)

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
    return fx.arith.ArithValue(op.result, signed=signed)


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


def _as_value(v: Union[int, fx.ArithValue]) -> ir.Value:
    # Coerce python int / ArithValue / raw ir value to a raw ir value (bare int -> i32; pass typed for i64).
    if isinstance(v, int):
        v = _create_i32_constant(v)
    elif hasattr(v, "ir_value"):
        v = v.ir_value()
    return _unwrap_value(v)


def memory_fence(order: Optional[str] = None, scope: Optional[str] = None) -> None:
    order_enum = _unwrap_order(order)
    llvm.fence(order_enum, syncscope=_unwrap_scope("agent" if scope is None else scope))


def addr_buffer_resource(addr_i64: fx.ArithValue, num_records_bytes: int) -> fx.ArithValue:
    return create_buffer_resource_from_addr(addr_i64, num_records_bytes=num_records_bytes)


def elem_ptr(
    base: Union[int, fx.ArithValue],
    idx: Union[int, fx.ArithValue],
    space: Union[int, str],
    elem_bytes: int = 4,
) -> ir.Value:
    ptr = create_llvm_ptr(_unwrap_value(base), _unwrap_space(space))
    idx_val = _unwrap_value(idx)
    if isinstance(idx_val.type, ir.IndexType):
        idx_val = _unwrap_value(std_arith.IndexCastOp(fx.T.i64(), idx_val).result)
    elif isinstance(idx_val.type, ir.IntegerType) and idx_val.type.width < 64:
        idx_val = _unwrap_value(std_arith.ExtSIOp(fx.T.i64(), idx_val).result)
    byte_off = _unwrap_value(std_arith.MulIOp(idx_val, _create_i64_constant(elem_bytes)).result)
    return get_element_ptr(ptr, byte_offset=byte_off, elem_type=fx.T.i8())


def atomic_add(
    base: Union[int, fx.ArithValue],
    offset: Union[int, fx.ArithValue],
    val: Union[int, fx.ArithValue],
    scope: str = "agent",
    space: Union[int, str] = "global",
    order: str = "relaxed",
) -> fx.ArithValue:
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
    return fx.arith.ArithValue(res, signed=True)


def ld(
    base: Union[int, fx.ArithValue],
    offset: Union[int, fx.ArithValue],
    *,
    scope: str = "agent",
    space: Union[int, str] = "global",
    order: str = "relaxed",
    dtype: Optional[object] = None,
) -> fx.ArithValue:
    if dtype is None:
        dtype = fx.T.i32()
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
    return fx.arith.ArithValue(op.result, signed=True)


def st(
    base: Union[int, fx.ArithValue],
    offset: Union[int, fx.ArithValue],
    val: Union[int, fx.ArithValue],
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
    dst: Union[int, fx.ArithValue],
    src: Union[int, fx.ArithValue],
    nbytes: int,
    dst_off: Union[int, fx.ArithValue] = 0,
    src_off: Union[int, fx.ArithValue] = 0,
    load_cache_modifier: int = 0,
    store_cache_modifier: int = 0,
) -> None:
    def _addr_i64(addr: Union[int, fx.ArithValue]) -> fx.ArithValue:
        if isinstance(addr, int):
            return fx.Int64(addr)
        v = _unwrap_value(addr)
        if isinstance(v.type, ir.IndexType):
            v = std_arith.IndexCastOp(fx.T.i64(), v).result
        elif isinstance(v.type, ir.IntegerType) and v.type.width < 64:
            v = std_arith.ExtSIOp(fx.T.i64(), v).result
        return fx.arith.ArithValue(v, signed=True)

    def _copy_operand(
        operand: Union[int, fx.ArithValue], word_off: Union[int, fx.ArithValue], nbytes: int
    ) -> Tuple[fx.ArithValue, fx.ArithValue]:
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
            src, src_off + o, vec_width=_COPY_VEC_I32, dtype=fx.T.i32(), cache_modifier=load_cache_modifier
        )
        for o in offs
    ]
    for o, v in zip(offs, vals):
        buffer_store(v, dst, dst_off + o, cache_modifier=store_cache_modifier)


# ── Scalar and index arithmetic ──────────────────────────────────────────────


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def ceildiv_pow2(a, b: int):
    """``ceildiv(a, b)`` for a power-of-two ``b`` and a non-negative device value ``a``.
    Signed ``a // b`` lowers to arith.floordivsi (divide + remainder + sign fixup); the shift
    is one. Use on runtime hot-path values; plain ``ceildiv`` stays for host-side ints."""
    assert b > 0 and (b & (b - 1)) == 0
    return (a + (b - 1)) >> (b.bit_length() - 1)


def floordiv_pow2(a, b: int):
    """``a // b`` for a power-of-two ``b``, shifting past what ``floordivsi`` would lower to."""
    assert b > 0 and (b & (b - 1)) == 0
    return a >> (b.bit_length() - 1)


def _u32(v):
    return _raw(fx.Int32(v) if isinstance(v, int) else v)


def udiv(a, b):
    """``a // b`` for device values proven non-negative (tile ids, group tile counts).
    Python ``//`` on a device Int32 is arith.floordivsi (magic multiply plus a remainder and
    sign fixup); the unsigned form is far cheaper on the exposed grouped tile-decode chain."""
    return ArithValue(arith.divui(_u32(a), _u32(b)))


def umod(a, b):
    """``a % b`` for device values proven non-negative; see ``udiv``."""
    return ArithValue(arith.remui(_u32(a), _u32(b)))


def uindex(v):
    """``arith.index_cast(T.index, v)`` for a device value proven non-negative (row/tile/group
    offsets). The signed cast sign-extends into every derived SRD base/extent; the unsigned cast
    zero-extends so the high half folds to zero."""
    return ArithValue(arith.index_castui(T.index, _raw(v)))


def _as_index(v):
    # An extent may be a runtime value or a compile-time int; coerce both to an MLIR index.
    return arith.index(v) if isinstance(v, int) else arith.index_cast(T.index, v)


def _i64(v):
    # widen an i32 runtime value to i64 (avoids overflow in worst-case base offsets)
    return ArithValue(arith.extsi(T.i64, _unwrap_value(v)), signed=True)


# ── Wave and lane primitives ─────────────────────────────────────────────────

# gfx9 DPP controls: ROW_SHR|n shifts right by n within a 16-lane row; ROW_BCAST15/31 feed a
# row's last lane into following rows; QUAD_SWAP exchanges within a group of four.
_DPP_ROW_SHR = 0x110
_DPP_ROW_BCAST15 = 0x142
_DPP_ROW_BCAST31 = 0x143
_DPP_QUAD_SWAP1 = 0xB1  # quad_perm:[1,0,3,2] -- exchange with the neighbouring lane
_DPP_QUAD_SWAP2 = 0x4E  # quad_perm:[2,3,0,1] -- exchange with the lane two over


def _res_of(op):
    """Unwrap an op builder's single result (some rocdl builders already return one)."""
    return op.result if hasattr(op, "result") else op


def _readfirstlane_i32(v):
    """Force a wave-uniform-in-value i32 into an SGPR via s_readfirstlane.

    A value the compiler's divergence analysis cannot prove uniform lands in VGPRs, and a
    buffer descriptor built from one puts every store behind a readfirstlane/saveexec
    waterfall. Pinning the value collapses the SRD to scalar regs and drops the waterfall."""
    return ArithValue(_res_of(rocdl.readfirstlane(res=_raw(v).type, src=_raw(v))))


def _dpp_add_i32(acc, ctrl, row_mask=0xF):
    """acc + DPP(acc, ctrl); masked-off and shifted-in lanes contribute 0."""
    raw = _raw(acc)
    r = rocdl.update_dpp(raw.type, _raw(fx.Int32(0)), raw, ctrl, row_mask, 0xF, True)
    return acc + ArithValue(_res_of(r))


def _dpp_add_f32(acc, ctrl, row_mask=0xF):
    """acc + DPP(acc, ctrl) for f32; masked-off and shifted-in lanes contribute 0."""
    raw = _raw(acc)
    r = rocdl.update_dpp(raw.type, _raw(fx.Float32(0.0)), raw, ctrl, row_mask, 0xF, True)
    return acc + fx.Float32(_res_of(r))


def _row16_sum_f32(v):
    """Sum an f32 across each 16-lane DPP row; lane 15 of the row ends with the total.

    Four ROW_SHR adds, i.e. an inclusive scan whose last lane holds the row sum.
    All full-rate VALU. The obvious alternative -- a ``gpu.shuffle`` XOR butterfly
    -- lowers to ``ds_bpermute_b32``, one LDS crossbar op per step, and at the
    rate a GEMM epilogue calls this that measured +0.59 ms.
    """
    for _sh in (1, 2, 4, 8):
        v = _dpp_add_f32(v, _DPP_ROW_SHR + _sh)
    return v


def _dpp_max_f32(acc, ctrl, row_mask=0xF):
    """max(acc, DPP(acc, ctrl)) for f32; masked-off and shifted-in lanes contribute 0.

    Zero is the identity only over non-negative values, which is what the callers
    reduce: an abs-max.
    """
    raw = _raw(acc)
    r = rocdl.update_dpp(raw.type, _raw(fx.Float32(0.0)), raw, ctrl, row_mask, 0xF, True)
    return fx.Float32(_res_of(arith.MaxNumFOp(raw, _res_of(r))))


def _wave_max_f32(v):
    """Max a non-negative f32 across the whole wave; lane 63 ends with the total.

    The scan shape of :func:`_wave_prefix_add_i32` with max for add. Requires a full
    EXEC mask, so a lane whose value should not count has to be zeroed first rather
    than branched around.
    """
    for _sh in (1, 2, 4, 8):
        v = _dpp_max_f32(v, _DPP_ROW_SHR + _sh)
    v = _dpp_max_f32(v, _DPP_ROW_BCAST15, row_mask=0xA)
    return _dpp_max_f32(v, _DPP_ROW_BCAST31, row_mask=0xC)


def _wave_prefix_add_i32(v):
    """Wave64 inclusive add-scan of a per-lane i32 (lane l ends with the sum of 0..l). Six DPP
    steps replace the serial carry (bound_ctrl zeroes shifted-in lanes). Requires a full EXEC
    mask (kernel entry)."""
    for _sh in (1, 2, 4, 8):
        v = _dpp_add_i32(v, _DPP_ROW_SHR + _sh)
    v = _dpp_add_i32(v, _DPP_ROW_BCAST15, row_mask=0xA)
    return _dpp_add_i32(v, _DPP_ROW_BCAST31, row_mask=0xC)


def _readlane_i32(v, lane):
    """Broadcast one lane of a per-lane i32 into an SGPR; lane must be wave-uniform."""
    raw = _raw(v)
    return ArithValue(_res_of(rocdl.readlane(res=raw.type, src=raw, lane=lane)))


def _wave_count_le_i32(v, bound):
    """Number of lanes whose per-lane i32 is <= the wave-uniform bound. One ballot plus one
    s_bcnt1; on a monotone table this is the first lane above bound, an O(1) stand-in for a
    G-wide boundary compare chain."""
    m = _res_of(rocdl.ballot(res=ir.IntegerType.get_signless(64), pred=_raw(v <= bound)))
    n = _res_of(llvm.intr_ctpop(m))
    return ArithValue(arith.trunci(T.i32, n))


def _lane_load_i32(rsrc, idx):
    """One per-lane i32 gather from a buffer resource; out-of-range lanes read 0."""
    return ArithValue(buffer_load(rsrc, idx, vec_width=1, dtype=T.i32))


def _sload_i32(rsrc, idx):
    """One wave-uniform i32 read on the scalar path (``s_buffer_load`` into an SGPR). The value
    never enters the VGPR file, so a consumer waits on lgkmcnt and the read hits the scalar
    cache, not the g2s-evicted vL1D. (Raw intrinsic: buffer_load(is_scalar=) is not universal.)"""
    i32_t = ir.IntegerType.get_signless(32)
    rsrc_v4 = llvm.bitcast(
        ir.VectorType.get([4], i32_t), llvm.ptrtoint(ir.IntegerType.get_signless(128), _raw(rsrc))
    )
    args = [rsrc_v4, _raw(fx.Int32(idx * 4)), _raw(fx.Int32(0))]  # rsrc, byte offset, cache policy
    return ArithValue(llvm.call_intrinsic(i32_t, "llvm.amdgcn.s.buffer.load.i32", args, [], []))


# ── Synchronisation and LDS addressing ───────────────────────────────────────


def wait_lgkmcnt(n=0, memory=False):
    """Drain LDS/scalar traffic to at most ``n`` outstanding ops.

    ``memory`` clobbers memory as well, which is needed wherever the drain has to order
    against ops the asm blob does not name -- g2s copies are intrinsics, so a blob with no
    memory effect may be scheduled ahead of them. Leave it off when a wave only reads back
    what it alone wrote, so the drain is the whole of the ordering required.
    """
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt lgkmcnt({n})",
        constraints="~{memory}" if memory else "",
        has_side_effects=True,
    )


def _lds_barrier(vmcnt=None):
    # Drain outstanding LDS writes (lgkmcnt) BEFORE the workgroup barrier, else
    # readers may observe stale LDS (a bare s_barrier doesn't wait on ds_write).
    # ``vmcnt`` also drains direct global-to-LDS copies, which lgkmcnt does not track:
    # repurposing an operand pool rather than refilling it has to wait on the
    # mainloop's last prefetch too. That form clobbers memory as well -- the g2s
    # copies are intrinsics, and an asm blob with no memory effect could schedule
    # ahead of them and count loads that have not been issued.
    wait = "lgkmcnt(0)" if vmcnt is None else f"vmcnt({vmcnt}) lgkmcnt(0)"
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt {wait}\ns_barrier",
        constraints="" if vmcnt is None else "~{memory}",
        has_side_effects=True,
    )


def _inttoptr_lds(byte_addr):
    """Integer byte address -> !llvm.ptr<3> (LDS). Parsed per call: the type is
    bound to the current MLIRContext and cannot be cached across compiles."""
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


def _lds_ptr_from_i32(addr_i32, byte_offset=0):
    """Build an LDS pointer (ptr<3>) from an i32 byte address + optional static offset."""
    ptr = _inttoptr_lds(ArithValue(addr_i32).extui(T.i64))
    if byte_offset != 0:
        ptr = get_element_ptr(ptr, static_byte_offset=byte_offset)
    return ptr
