###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Low-level FlyDSL device primitives for the mega-MoE kernels.

Pointer / atomic / fence primitives shared by the LDS (addrspace-3) and global
(addrspace-1) paths: the caller supplies a base address + element offset and picks
``space`` / ``scope``, so one ``elem_ptr`` / ``atomic_add`` serves both memory spaces
and both tensor-derived and raw symmetric peer addresses. Depends only on ``flydsl``.
"""

import os

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm
from flydsl.expr.buffer_ops import (
    _create_i32_constant,
    _create_i64_constant,
    _unwrap_value,
    create_buffer_resource_from_addr,
    create_llvm_ptr,
    get_element_ptr,
)

# Spin-wait timeout in s_memrealtime ticks (~100 MHz on gfx950 -> default ~30s).
# Override via env for slower/faster debug; a spinning kernel printf's once per window.
SPIN_TIMEOUT_CYCLES = int(os.environ.get("MEGA_SPIN_TIMEOUT_CYCLES", str(3_000_000_000)))


def read_clock():
    """Read the constant-rate GPU realtime counter (s_memrealtime); for spin-wait timeouts."""
    op = llvm.inline_asm(
        fx.T.i64(), [], "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", has_side_effects=True
    )
    return fx.arith.ArithValue(op, signed=False)


# str space selector -> LLVM int address space (1 global / 3 LDS aka workgroup).
_ADDR_SPACES = {"global": 1, "gmem": 1, "lds": 3, "shared": 3, "smem": 3}
# str order selector -> llvm.AtomicOrdering (None / 'relaxed' -> monotonic).
_ATOMIC_ORDERINGS = {
    "relaxed": llvm.AtomicOrdering.monotonic,
    "acquire": llvm.AtomicOrdering.acquire,
    "release": llvm.AtomicOrdering.release,
    "acq_rel": llvm.AtomicOrdering.acq_rel,
    "seq_cst": llvm.AtomicOrdering.seq_cst,
}


def _unwrap_scope(scope):
    """str scope -> LLVM syncscope: 'sys' -> None (system default); 'agent'/'workgroup'
    (or any explicit syncscope string) pass through."""
    if scope == "sys":
        return None  # None == LLVM system-wide syncscope
    return scope


def _unwrap_space(space):
    """str space -> int address space: 'global'/'gmem' -> 1, 'lds'/'shared'/'smem' -> 3.
    An int (1 global / 3 LDS) passes through for back-compat."""
    if isinstance(space, int):
        return space  # already an address-space int
    try:
        return _ADDR_SPACES[space]
    except KeyError:
        raise ValueError(f"bad space {space!r}; expected one of {sorted(_ADDR_SPACES)} or an int")


def _unwrap_order(order):
    """str order -> llvm.AtomicOrdering: None/'relaxed' -> monotonic; 'acquire'/'release'/
    'acq_rel'/'seq_cst' map directly."""
    if order is None:
        return llvm.AtomicOrdering.monotonic
    try:
        return _ATOMIC_ORDERINGS[order]
    except KeyError:
        raise ValueError(f"bad order {order!r}; expected None or one of {sorted(_ATOMIC_ORDERINGS)}")


def _mem_fence():
    """deep_ep cheap fence: s_waitcnt drain + compiler barrier, so a following
    relaxed atomic gets release/acquire ordering."""
    llvm.inline_asm(fx.T.i32(), [], "s_waitcnt lgkmcnt(0) vmcnt(0)", "=r,~{memory}", has_side_effects=True)


def l2_invalidate():
    """gfx950: invalidate this CU's L2 view so a following load re-fetches from the coherence
    point (HBM). Pairs with a producer's write-through (sc1) store on another CU whose data is
    in HBM but left a stale L2 line here."""
    llvm.inline_asm(fx.T.i32(), [], "buffer_inv sc1", "=r,~{memory}", has_side_effects=True)


def memory_fence(order=None, scope=None):
    """Standalone memory fence; defaults to the cheap deep_ep s_waitcnt drain.

    ``order`` (str, same vocab as :func:`ld` / :func:`st`): ``None`` / ``"relaxed"`` ->
    cheap :func:`_mem_fence` (scope ignored, wave-local drain); ``"acquire"`` /
    ``"release"`` / ``"acq_rel"`` / ``"seq_cst"`` -> a real LLVM ``fence``.
    ``scope`` (str): 'sys' -> system default, else 'agent'/'workgroup' (default 'agent')."""
    order_enum = _unwrap_order(order)
    if order_enum == llvm.AtomicOrdering.monotonic:
        _mem_fence()  # no real ordering requested -> cheap drain
        return
    llvm.fence(order_enum, syncscope=_unwrap_scope("agent" if scope is None else scope))


def addr_buffer_resource(addr_i64, num_records_bytes):
    """Buffer resource over a raw i64 address (for vectorized buffer_load/store)."""
    return create_buffer_resource_from_addr(addr_i64, num_records_bytes=num_records_bytes)


def elem_ptr(base, idx, space, elem_bytes=4):
    """LLVM ptr to element ``(base as addrspace-`space`)[idx]`` (``elem_bytes`` stride).

    ``base`` is the integer/index/i64 base already living in ``space`` (tensor
    ``extract_base_index``, a raw symmetric i64 addr, or LDS ``ptrtoint``); ``space`` (str
    'global'/'lds', or the int 1 global / 3 LDS); ``elem_bytes`` the element size (4 i32,
    8 i64). One impl serves both spaces -- only the base source and the address space (a
    type, not a runtime value) differ. ``base`` is unwrapped (DSL Numeric / ArithValue ->
    ir.Value, like official ``buffer_load``), so callers can pass a
    ``sym_layout.<region>_ptr`` property or any DSL value directly."""
    ptr = create_llvm_ptr(_unwrap_value(base), _unwrap_space(space))
    # byte offset in i64 (mirror official get_element_ptr) so it can't wrap at +-2GB
    idx_val = _unwrap_value(idx)
    if isinstance(idx_val.type, ir.IndexType):
        idx_val = _unwrap_value(std_arith.IndexCastOp(fx.T.i64(), idx_val).result)  # index -> i64
    elif isinstance(idx_val.type, ir.IntegerType) and idx_val.type.width < 64:
        idx_val = _unwrap_value(std_arith.ExtSIOp(fx.T.i64(), idx_val).result)  # widen i32 -> i64
    byte_off = _unwrap_value(std_arith.MulIOp(idx_val, _create_i64_constant(elem_bytes)).result)
    return get_element_ptr(ptr, byte_offset=byte_off, elem_type=fx.T.i8())  # i8 elem -> byte GEP


def addr_elem_ptr_i32(addr_i64, idx):
    """LLVM global ptr to the int32 element at ``(addr_i64)[idx]`` (scalar / atomic)."""
    return elem_ptr(addr_i64, idx, "global")


def atomic_add(base, offset, val, scope="agent", space="global", release=False):
    """atomicrmw add into the element ``base[offset]``; returns the OLD value (signed).

    The single canonical atomic add: ``base`` may be a tensor ``extract_base_index``, an
    LDS ``ptrtoint`` base, OR a raw symmetric i64 peer address (``create_llvm_ptr`` accepts
    index or i64 alike), so the old ``atomic_add_addr`` is just this with ``space='global'``.
    Mirrors ``buffer_load(rsrc, offset)`` (base + element offset); the element width
    (i32 / i64) is inferred from ``val``. i32/i64 atomics go through ``llvm.atomicrmw``
    (AMD has no integer raw-buffer-atomic-add intrinsic), so the handle is a base address,
    not a V# descriptor.

    Args:
        base: integer/index/i64 base in address space ``space``.
        offset: element offset into ``base`` -- scaled by the element size internally.
        val: addend; a Python int is always materialized as i32 -- for an i64
            atomic pass a typed i64 DSL Numeric / ir.Value, not a Python int.
        scope: str syncscope matching ``space`` ('workgroup' LDS, 'agent' device-wide,
            'sys' -> system/None).
        space: str 'global'/'lds' (or the int 1 global / 3 LDS).
        release: if True, drain (``_mem_fence``) before the atomic for release ordering.
    """
    # normalize val -> ir.Value (Python int -> i32 const; DSL Numeric -> ir_value)
    if isinstance(val, int):
        val = _create_i32_constant(val)
    elif hasattr(val, "ir_value"):
        val = val.ir_value()
    val = _unwrap_value(val)
    elem_bytes = val.type.width // 8  # 4 (i32) or 8 (i64)
    ptr = elem_ptr(base, offset, space, elem_bytes)
    if release:
        _mem_fence()  # publish prior writes before the (monotonic) atomic
    res = llvm.atomicrmw(
        llvm.AtomicBinOp.add,
        ptr,
        val,
        llvm.AtomicOrdering.monotonic,
        syncscope=_unwrap_scope(scope),
        alignment=elem_bytes,
    )
    return fx.arith.ArithValue(res, signed=True)


def atomic_cas(base, offset, cmp, val, scope="agent", space="global", release=False):
    """atomic compare-exchange ``base[offset]``: if it equals ``cmp`` set it to ``val``.
    Returns the OLD value (signed). First-writer-wins election (token dedup): exactly one
    caller sees ``old == cmp`` (the winner); losers read the winner's ``val``.

    Same base/offset/space convention as :func:`atomic_add`; ``release`` drains
    (``_mem_fence``) before the CAS for release ordering. i32 only."""
    # normalize cmp / val -> ir.Value (Python int -> i32 const; DSL Numeric -> ir_value)
    if isinstance(cmp, int):
        cmp = _create_i32_constant(cmp)
    elif hasattr(cmp, "ir_value"):
        cmp = cmp.ir_value()
    cmp = _unwrap_value(cmp)
    if isinstance(val, int):
        val = _create_i32_constant(val)
    elif hasattr(val, "ir_value"):
        val = val.ir_value()
    val = _unwrap_value(val)
    elem_bytes = val.type.width // 8
    ptr = elem_ptr(base, offset, space, elem_bytes)
    if release:
        _mem_fence()  # publish prior writes before the CAS
    # success + failure orderings both monotonic (manual fences handle visibility)
    pair = llvm.cmpxchg(
        ptr,
        cmp,
        val,
        llvm.AtomicOrdering.monotonic,
        llvm.AtomicOrdering.monotonic,
        syncscope=_unwrap_scope(scope),
        alignment=elem_bytes,
    )
    old = llvm.extractvalue(val.type, pair, [0])  # cmpxchg returns {old, success}; take old
    return fx.arith.ArithValue(old, signed=True)


def ld(base, offset, *, scope="agent", space="global", order="relaxed", dtype=None):
    """Scalar load of ``base[offset]``; returns a signed value.

    Same base/offset/space convention as :func:`atomic_add`. ``order`` (str) "acquire"
    drains (``_mem_fence``) BEFORE the load, which forces a fresh re-read (spin-wait), not
    full acquire ordering -- before consuming peer data call :func:`memory_fence`
    ("acquire"). "relaxed" skips the drain. ``dtype`` is the element type (defaults i32)."""
    # normalize dtype -> ir.Type (DSL type wrapper -> ir_type)
    if dtype is None:
        dtype = fx.T.i32()
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type
    elem_bytes = dtype.width // 8
    if _unwrap_order(order) in (
        llvm.AtomicOrdering.acquire,
        llvm.AtomicOrdering.acq_rel,
        llvm.AtomicOrdering.seq_cst,
    ):
        _mem_fence()  # drain so the load re-reads (spin-wait freshness)
    ptr = elem_ptr(base, offset, space, elem_bytes)
    op = llvm.LoadOp(
        dtype,
        ptr,
        ordering=llvm.AtomicOrdering.monotonic,
        syncscope=_unwrap_scope(scope),
        alignment=elem_bytes,
    )
    return fx.arith.ArithValue(op.result, signed=True)


def st(base, offset, val, *, scope="agent", space="global", order="relaxed"):
    """Scalar store ``base[offset] = val`` (element width inferred from ``val``).

    Same convention as :func:`atomic_add`. ``order`` (str) "release" drains (``_mem_fence``)
    before the store so prior writes are visible; "relaxed" does not."""
    # normalize val -> ir.Value (Python int -> i32 const; DSL Numeric -> ir_value)
    if isinstance(val, int):
        val = _create_i32_constant(val)
    elif hasattr(val, "ir_value"):
        val = val.ir_value()
    val = _unwrap_value(val)
    elem_bytes = val.type.width // 8
    if _unwrap_order(order) in (
        llvm.AtomicOrdering.release,
        llvm.AtomicOrdering.acq_rel,
        llvm.AtomicOrdering.seq_cst,
    ):
        _mem_fence()  # publish prior writes before this (monotonic) store
    ptr = elem_ptr(base, offset, space, elem_bytes)
    llvm.StoreOp(
        val, ptr, ordering=llvm.AtomicOrdering.monotonic, syncscope=_unwrap_scope(scope), alignment=elem_bytes
    )
