###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared 1x128 / 128x1 UE8M0 FP8 quant helpers for producer epilogues on gfx1250.

Producers (dispatch gather, SwiGLU fwd/bwd, GEMM epilogues) call these on a register tile so they write FP8
data + E8M0 scales directly in the layout the consumer GEMM reads; no separate quant pass.

Exponent (no division, exact): with FP8_MAX = 1.75 * 2^MAX_EXP (E4M3 448: MAX_EXP 8, E5M2 57344: 15) and
amax = 1.m * 2^ea (fp32 bits),
    ceil(log2(amax / FP8_MAX)) = ea - MAX_EXP + (mantissa(amax) > mantissa(1.75)).
This equals the bit-pattern rounding of ``r = amax / FP8_MAX`` (quantization_fp8blk_moe_kernel) and
``pow2_floor(FP8_MAX / max(amax, 1e-4))`` (g90 / Turbo POW2) for every finite amax: a correctly rounded quotient
never crosses a power of two because FP8_MAX's mantissa 1.75 keeps both neighbours of an exact power of two more
than half an ulp away. The scale is then applied as an exact power-of-two multiply built from the exponent bits
(the v_ldexp form), one RNE rounding to FP8.

Conventions:
  FLY (MegaMoE flygg, quantization_fp8blk_moe_kernel): all-zero block -> e = 0; e clamped to [-126, 126]; no output clamp.
  G90 (dense g90, quantization_fp8blk_dual_kernel / Turbo POW2 blockwise): amax floored at 1e-4; inf amax -> multiplier 0, byte 255;
       q clamped to +-FP8_MAX.

Scale layouts (SL constexpr):
  SL_X4     = 0: E8M0 bytes [R, K/32], the block-128 exponent written 4x as one int32 (flygg MX planes)
  SL_BYTE   = 1: E8M0 bytes [R, K/128] (g90)
  SL_PACKED = 2: dsv4up / upstream mxfp_gemm_cdna5.pack_scale BK256 layout [R/128, K/32*128]:
                 byte (r//128)*(K/32*128) + kb*512 + (r%32)*16 + ((r%128)//32)*4 + s, s = 0..3 (one int32 per row)
"""

import triton
import triton.language as tl

SL_X4 = 0
SL_BYTE = 1
SL_PACKED = 2
_SL_NAMES = {
    "x4": SL_X4,
    "fly": SL_X4,
    "byte": SL_BYTE,
    "g90": SL_BYTE,
    "packed": SL_PACKED,
    "dsv4": SL_PACKED,
}

F32_MANT_1P75 = 0x600000  # mantissa field of 1.75


def scale_layout_id(name) -> int:
    if isinstance(name, int):
        assert name in (SL_X4, SL_BYTE, SL_PACKED)
        return name
    return _SL_NAMES[name]


def fp8_max_exp(fp8_max: float) -> int:
    """FP8_MAX = 1.75 * 2^k (E4M3 448 -> 8, E5M2 57344 -> 15); asserted so the integer exponent rule is exact."""
    import math

    m, e = math.frexp(fp8_max)  # fp8_max = m * 2^e, m in [0.5, 1)
    assert m == 0.875, f"FP8_MAX {fp8_max} is not 1.75 * 2^k"
    return e - 1


@triton.jit
def ue8m0_exp_fly(amax, MAX_EXP: tl.constexpr):
    """smallest e with amax <= FP8_MAX * 2^e, 0 for amax == 0, clamped to [-126, 126] (inf/nan -> 126)."""
    bits = amax.to(tl.int32, bitcast=True)
    fld = (bits >> 23) & 0xFF
    e = fld - (127 + MAX_EXP) + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
    e = tl.where(fld == 0xFF, 126, e)
    e = tl.where(amax > 0, e, 0)
    return tl.minimum(tl.maximum(e, -126), 126)


@triton.jit
def ue8m0_exp_g90(amax, MAX_EXP: tl.constexpr):
    """exponent of Turbo POW2 scale_inv = 1 / pow2_floor(FP8_MAX / max(amax, 1e-4)); 128 for inf amax."""
    a = tl.maximum(amax, 1e-4)
    bits = a.to(tl.int32, bitcast=True)
    fld = (bits >> 23) & 0xFF
    e = fld - (127 + MAX_EXP) + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
    return tl.where(fld == 0xFF, 128, e)


@triton.jit
def pow2_neg(e):
    """2^-e as fp32 built from the exponent bits (exact; e in [-126, 127]); e == 128 (g90 inf) -> 0."""
    return tl.where(e > 127, 0, (127 - e) << 23).to(tl.float32, bitcast=True)


@triton.jit
def pow2_neg_fly(e):
    """2^-e for e in [-126, 126] (FLY convention, no special case)."""
    return ((127 - e) << 23).to(tl.float32, bitcast=True)


@triton.jit
def pow2_pos(e):
    """2^e as fp32 (scale_inv); e == 128 -> inf, like 1 / 0."""
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def e8m0_word(e):
    return ((e + 127) & 0xFF) * 0x01010101


@triton.jit
def store_scale(S, r, kb, e, R, KB, SL: tl.constexpr, mask=None):
    """E8M0 of 1x128 block (row r, 128-block kb) of an [R, KB*128] operand; r is a 1-D int tensor, kb a scalar
    (or broadcastable). R only matters for the packed layout (R % 128 == 0 there)."""
    r64 = r.to(tl.int64)
    if SL == 0:
        tl.store(S.to(tl.pointer_type(tl.int32)) + r64 * KB + kb, e8m0_word(e), mask=mask)
    elif SL == 1:
        tl.store(S + r64 * KB + kb, ((e + 127) & 0xFF).to(tl.uint8), mask=mask)
    else:
        w = (r64 >> 7) * (KB * 128) + kb * 128 + (r64 & 31) * 4 + ((r64 >> 5) & 3)
        tl.store(S.to(tl.pointer_type(tl.int32)) + w, e8m0_word(e), mask=mask)
