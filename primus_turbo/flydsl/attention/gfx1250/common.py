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

"""Helpers shared by the gfx1250 attention kernels.

The few pieces of aiter's ``kernels_common`` / ``tensor_shim`` the kernels reach, kept
here so the package depends on flydsl, torch and the standard library only.

The kernels were developed against flydsl 0.3.4, while Primus-Turbo pins 0.2.4. The 0.3.4
expression helpers they use that 0.2.4 lacks -- ``fx.ceildiv``, integer ``fx.max`` /
``fx.min`` and ``fx.to_llvm_ptr`` -- are provided below: the 0.3.4 function when it exists,
otherwise the same MLIR op built directly. Under 0.3.4 the kernels therefore compile to the
same instruction stream as the originals.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import rocdl as _rocdl

LOG2E = 1.4426950408889634

_LLVM_ADDRESS_SPACE = {1: 1, 3: 3, fx.AddressSpace.Global: 1, fx.AddressSpace.Shared: 3}


def create_llvm_ptr(value, address_space=1):
    """Raw LLVM pointer for atomics and intrinsic APIs."""
    # Accept either the LLVM number (1 global / 3 LDS) or an fx.AddressSpace, so a caller
    # cannot silently pass the wrong one.
    if hasattr(fx, "to_llvm_ptr"):
        space = {1: fx.AddressSpace.Global, 3: fx.AddressSpace.Shared}.get(address_space, address_space)
        pt = fx.PointerType.get(fx.Int32.ir_type, address_space=space, alignment=4)
        return fx.as_ir_value(fx.to_llvm_ptr(fx.inttoptr(pt, value)))
    # UNSTABLE(flydsl 0.2.4 fallback): raw llvm.inttoptr, the op fx.to_llvm_ptr emits.
    ptr_type = ir.Type.parse(f"!llvm.ptr<{_LLVM_ADDRESS_SPACE[address_space]}>")
    return _llvm.IntToPtrOp(ptr_type, fx.as_ir_value(value)).result


def _as_int(v, like):
    """`v` as a value of the flydsl integer class `like` (python ints become constants)."""
    return v if isinstance(v, like) else like(v)


def _int_binop(signed_op, unsigned_op, lhs, rhs):
    cls = type(lhs) if isinstance(lhs, (fx.Int32, fx.Uint32)) else type(rhs)
    if not isinstance(lhs, (fx.Int32, fx.Uint32)) and not isinstance(rhs, (fx.Int32, fx.Uint32)):
        raise TypeError(f"expected an fx.Int32/fx.Uint32 operand, got {type(lhs)}, {type(rhs)}")
    op = unsigned_op if cls is fx.Uint32 else signed_op
    # UNSTABLE(flydsl 0.2.4 fallback): raw arith ops, the ones fx.ceildiv/max/min emit.
    return cls(op(fx.as_ir_value(_as_int(lhs, cls)), fx.as_ir_value(_as_int(rhs, cls))))


def ceildiv(lhs, rhs):
    """Integer ``lhs / rhs`` rounded up (``arith.ceildivsi`` / ``ceildivui``)."""
    if hasattr(fx, "ceildiv"):
        return fx.ceildiv(lhs, rhs)
    return _int_binop(_arith.ceildivsi, _arith.ceildivui, lhs, rhs)


def imax(lhs, rhs):
    """Integer maximum (``arith.maxsi`` / ``maxui``)."""
    if hasattr(fx, "max"):
        return fx.max(lhs, rhs)
    return _int_binop(_arith.maxsi, _arith.maxui, lhs, rhs)


def imin(lhs, rhs):
    """Integer minimum (``arith.minsi`` / ``minui``)."""
    if hasattr(fx, "min"):
        return fx.min(lhs, rhs)
    return _int_binop(_arith.minsi, _arith.minui, lhs, rhs)


def shuffle_xor(value, offset, width):
    """Lane ``k`` reads lane ``k ^ offset`` (``fx.gpu.shuffle_xor``; 0.2.4 only has the
    since-deprecated ``Numeric.shuffle_xor`` method, which emits the same gpu.shuffle)."""
    if hasattr(fx.gpu, "shuffle_xor"):
        return fx.gpu.shuffle_xor(value, offset, width)
    return value.shuffle_xor(offset, width)


# The forward's fastest O writer stages O through LDS and drains it with
# global_store_async_from_lds_b128. flydsl 0.2.4 has no op for it, and the intrinsic, emitted
# by hand, faults on the card: the LLVM bundled with 0.2.4 addresses global memory with the
# LDS operand (faulting addresses are LDS offsets). Only a flydsl that ships the op gets that
# writer; otherwise the kernels fall back to the buffer_store writer, ~1.3% slower at
# Llama-3.1-8B shapes.
HAS_ASYNC_LDS_STORE = hasattr(_rocdl, "global_store_async_from_lds_b128")


def _to_raw(v):
    """A typed flydsl value (Int32, Vector, ...) as a raw ir.Value."""
    return fx.as_ir_value(v)


_COMPILED = {}


def _run_compiled(exe, *args):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** launches the kernel.
    Later calls go straight to the cached ``CompiledFunction``.
    """
    cf = _COMPILED.get(exe)
    if cf is not None:
        cf(*args)
        return
    try:
        _COMPILED[exe] = flyc.compile(exe, *args)
    except Exception:
        # flyc.compile leaks ir.Context on failure; pop it so a retry takes the right path.
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:  # noqa: BLE001, S110
            pass
        raise
