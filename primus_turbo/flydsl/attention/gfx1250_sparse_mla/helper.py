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

"""Register-level primitives shared by the gfx1250 (wave32, WMMA) sparse-MLA kernels.

Measured fragment layout of ``v_wmma_f32_16x16x32_bf16`` (D = A @ B, A is 16x32, B is 32x16),
with ``l16 = lane % 16`` and ``hi = lane // 16``:

    A: lane holds row ``l16``,    k = hi*8 + [0..7] then 16 + hi*8 + [0..7]   (v16bf16)
    B: lane holds column ``l16``, k = hi*8 + [0..7] then 16 + hi*8 + [0..7]   (v16bf16)
    D: lane holds column ``l16``, rows hi*8 + [0..7]                           (v8f32)

So two D tiles stacked along M are, unchanged, one B (or A) operand whose reduction axis is
that M. The kernels lean on this: a score tile comes out of one WMMA already in the operand
layout the next WMMA consumes, with no cross-lane movement.

``ds_load_tr16_b128`` reads an 8x8 block of 16-bit elements from LDS transposed: within each
group of 8 lanes, lane j supplies the address of row j (8 consecutive elements, 16-byte
aligned) and lane l receives element l of every row. A misaligned address silently degrades
to a plain load, so every LDS row pitch here is a multiple of 16 bytes.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import arith, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

WAVE = 32
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453

# Buffer descriptor flags for gfx12-class targets with raw bounds checking (OOB_SELECT=3):
# a load at byte offset >= num_records returns 0 and a store there is dropped. FlyDSL's
# default RDNA flags use OOB_SELECT=2, which does no checking at all.
_RSRC_FLAGS_OOB_CHECKED = (7 << 12) | (4 << 15) | (1 << 24) | (3 << 28)

# Everything but nnan/ninf: the online softmax carries -inf for rows with nothing visible
# yet, and ninf licenses the compiler to fold the guards on it away.
_FM = (
    arith.FastMathFlags.reassoc
    | arith.FastMathFlags.nsz
    | arith.FastMathFlags.arcp
    | arith.FastMathFlags.contract
    | arith.FastMathFlags.afn
)


def _ptr3_type():
    return ir.Type.parse("!llvm.ptr<3>")


def _res(x):
    return x if isinstance(x, ir.Value) else x.result


def vec_type(n, elem):
    return ir.VectorType.get([n], elem.ir_type)


def tensor_addr(t):
    """Device address of a kernel tensor argument, as Int64."""
    return fx.Int64(fx.ptrtoint(fx.get_iter(t)))


def make_rsrc(addr_i64, nbytes):
    """Bounds-checked buffer descriptor at ``addr_i64`` covering ``nbytes`` bytes."""
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), _raw(fx.Int64(addr_i64))).result
    return _rocdl.MakeBufferRsrcOp(
        ir.Type.parse("!llvm.ptr<8>"),
        ptr,
        _raw(fx.Int16(0)),
        _raw(fx.Int64(nbytes)),
        _raw(fx.Int32(_RSRC_FLAGS_OOB_CHECKED)),
    ).result


def buf_load(rsrc, byte_off, res_type, soff=0):
    return _rocdl.raw_ptr_buffer_load(res_type, rsrc, _raw(fx.Int32(byte_off)), _raw(fx.Int32(soff)))


# Cache policy (gfx12 temporal hint) for data written once and read back only much later.
CPOL_NT = 1


def buf_store(data, rsrc, byte_off, soff=0, aux=0):
    attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), aux) if aux else None
    _rocdl.RawPtrBufferStoreOp(_raw(data), rsrc, _raw(fx.Int32(byte_off)), _raw(fx.Int32(soff)), aux=attr)


def buf_load_b128(rsrc, byte_off, elem=fx.BFloat16):
    """16 bytes as v8 of a 16-bit element type."""
    v = buf_load(rsrc, byte_off, vec_type(4, fx.Int32))
    return Vec(v).bitcast(elem)


def lds_ptr(addr_i32):
    return llvm.inttoptr(_ptr3_type(), _raw(fx.Int32(addr_i32)))


def lds_load(addr_i32, res_type, align=16):
    return llvm.load(res_type, lds_ptr(addr_i32), alignment=align)


def lds_store(val, addr_i32, align=16):
    llvm.store(_raw(val), lds_ptr(addr_i32), alignment=align)


def lds_load_tr16(addr_i32, elem=fx.BFloat16):
    return Vec(_res(_rocdl.ds_load_tr16_b128(vec_type(8, elem), lds_ptr(addr_i32))))


def cat(a, b):
    a, b = Vec(a), Vec(b)
    return a.shuffle(b, list(range(a.numel + b.numel)))


def wmma_bf16(a, b, c):
    """c += a @ b for one 16x16x32 bf16 tile (a, b: v16bf16; c: v8f32)."""
    return Vec(_res(_rocdl.wmma_f32_16x16x32_bf16(vec_type(8, fx.Float32), _raw(a), _raw(b), _raw(c))))


def wmma_f16(a, b, c):
    return Vec(_res(_rocdl.wmma_f32_16x16x32_f16(vec_type(8, fx.Float32), _raw(a), _raw(b), _raw(c))))


def swap_halves(x):
    """Value of ``x`` held by lane ``lane ^ 16`` (f32 scalar)."""
    return fx.Float32(
        _rocdl.permlanex16(
            T.f32, _raw(x), _raw(x), _raw(fx.Int32(0x76543210)), _raw(fx.Int32(-19088744)), False, False
        )
    )


def fmax(a, b):
    return fx.Float32(arith.MaxNumFOp(_raw(fx.Float32(a)), _raw(fx.Float32(b)), fastmath=_FM).result)


def fadd(a, b):
    return fx.Float32(arith.addf(_raw(fx.Float32(a)), _raw(fx.Float32(b)), fastmath=_FM))


def fsub(a, b):
    return fx.Float32(arith.subf(_raw(fx.Float32(a)), _raw(fx.Float32(b)), fastmath=_FM))


def fmul(a, b):
    return fx.Float32(arith.mulf(_raw(fx.Float32(a)), _raw(fx.Float32(b)), fastmath=_FM))


def exp2(x):
    """Hardware exp2 (v_exp_f32): exp2(-inf) = 0."""
    return fx.Float32(rocdl.exp2(T.f32, _raw(fx.Float32(x))))


def log2(x):
    return fx.Float32(rocdl.log(T.f32, _raw(fx.Float32(x))))


def rcp(x):
    return fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(x))))


def vmul_scalar(v, s):
    """v8f32 * scalar."""
    v = Vec(v)
    sv = Vec.from_elements([fx.Float32(s)], fx.Float32).broadcast_to(v.numel)
    return Vec(arith.mulf(_raw(v), _raw(sv), fastmath=_FM))


def vsub_scalar(v, s):
    v = Vec(v)
    sv = Vec.from_elements([fx.Float32(s)], fx.Float32).broadcast_to(v.numel)
    return Vec(arith.subf(_raw(v), _raw(sv), fastmath=_FM))


def vfma_scalar(v, s, t):
    """v * s - t, elementwise with scalar s and scalar t (one v_fma per element)."""
    v = Vec(v)
    sv = Vec.from_elements([fx.Float32(s)], fx.Float32).broadcast_to(v.numel)
    tv = Vec.from_elements([fx.Float32(t)], fx.Float32).broadcast_to(v.numel)
    return Vec(arith.subf(arith.mulf(_raw(v), _raw(sv), fastmath=_FM), _raw(tv), fastmath=_FM))


def to_bf16(v, elem=fx.BFloat16):
    """v8f32 -> v8 bf16/f16 (round to nearest even)."""
    v = Vec(v)
    op = llvm.FPTruncOp(vec_type(v.numel, elem), _raw(v))
    return Vec(op.result)


def imin(a, b):
    a, b = fx.Int32(a), fx.Int32(b)
    return fx.Int32((a < b).select(a, b))


def imax(a, b):
    a, b = fx.Int32(a), fx.Int32(b)
    return fx.Int32((a > b).select(a, b))


def uniform(x):
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(x))))
