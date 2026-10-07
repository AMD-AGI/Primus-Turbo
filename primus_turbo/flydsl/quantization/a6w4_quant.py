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

"""A6W4 operand quantizers for FLUX: MXFP6 (E2M3) activations and MXFP4 (E2M1) weights.

Both operands are rotated with the same block-diagonal H16 along K, which cancels in the
product, then quantized per 32 with an E8M0 scale:
  activation: scale = 2^(floor(log2 amax) - 2), E2M3 round-half-even, saturating at 7.5
  weight:     amax mantissa rounded at 2^21 first (the MXFP4 packer's default), E2M1 RNE
The H16, amax and scale code is mxfp4_quant_kernel's (in-register H16 with the 0.25 folded
into the scale); the converters are gfx950's scaled pk16 FP6 / pk FP4 conversions.

quant_act_a6w4 and quant_w_a6w4 return exactly what primus_turbo.flydsl.gemm.gemm_a6w4_kernel
consumes, with every shuffle done by the stores:
  A: FP8-padded packed FP6 [M, K] uint8 (per 32 values, 24 bytes holding a little-endian
     6-bit code stream, then 8 zero bytes), scales in shuffle_scale_w4 layout
  W: packed FP4 (even element in the low nibble) in shuffle_weight_w4(16) layout, scales in
     shuffle_scale_w4 layout
Only the bytes are a contract; the returned tensors' shapes are not.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, math, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

BLK = 128
_RPB = 4  # rows per block
_MPR = BLK // _RPB  # 32 micro-blocks per block row; one thread owns one micro-block
_NREC_MAX = 0x7FFFFFFF


def _imax(a, b):
    return arith.select(a < b, b, a)


def _compute_scale_native(amax_bits, scale_rounding_bias, exp_up=0):
    """E8M0 scale, all-int32; mxfp4_quant_kernel's. Returns (native f32 bits, biased e8m0).
    E2M1 and E2M3 share the largest power of two (2), so one recipe serves both."""
    extracted = ((amax_bits + scale_rounding_bias) >> 23) & 0x1FF
    extracted = extracted - 127 - 2 - exp_up
    extracted = _imax(extracted, -127)
    extracted = arith.select(extracted < 128, extracted, 128)
    biased = extracted + 127
    native_bits = (biased + exp_up) << 23
    return native_bits, biased


def _h4(v0, v1, v2, v3):
    a0 = v0 + v1
    a1 = v0 - v1
    a2 = v2 + v3
    a3 = v2 - v3
    return a0 + a2, a1 + a3, a0 - a2, a1 - a3


def _rht16_pair(v):
    """Both H16 groups of a 32-element micro-block as <2 x float>, without the trailing 0.25
    (folded into the scale as exp_up=2)."""
    p = [Vec.from_elements([v[i], v[i + 16]], fx.Float32) for i in range_constexpr(16)]
    o = [None] * 16
    for b in range_constexpr(4):
        y = _h4(p[4 * b + 0], p[4 * b + 1], p[4 * b + 2], p[4 * b + 3])
        for j in range_constexpr(4):
            o[4 * b + j] = y[j]
    r = [None] * 16
    for lc in range_constexpr(4):
        y = _h4(o[0 * 4 + lc], o[1 * 4 + lc], o[2 * 4 + lc], o[3 * 4 + lc])
        for j in range_constexpr(4):
            r[j * 4 + lc] = y[j]
    return [r[i][0] for i in range_constexpr(16)] + [r[i][1] for i in range_constexpr(16)]


def _amax_bits(vf):
    cur = math.absf(vf[0])
    for i in range_constexpr(1, 32):
        cur = cur.maximumf(math.absf(vf[i]))
    return Vec.from_elements([cur], fx.Float32).bitcast(fx.Int32)[0]


def _cvt_fp4(vf, scale_f32):
    """32 f32 -> 4 i32 words, 8 E2M1 codes each, even element in the low nibble."""
    words = []
    for wi in range_constexpr(4):
        acc = fx.Int32(0)
        for pair in range_constexpr(4):
            i = wi * 8 + pair * 2
            acc = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, acc, vf[i], vf[i + 1], scale_f32, pair)
        words.append(acc)
    return words


def _cvt_fp6(vf, scale_f32):
    """32 f32 -> 6 i32 words: a little-endian stream of 32 E2M3 codes. The 2xpk16 convert
    interleaves its sources (code 2i from src0[i], code 2i+1 from src1[i]), so feed it the
    even and odd elements."""
    src = [
        _raw(Vec.from_elements([fx.Float32(_raw(vf[2 * i + h])) for i in range_constexpr(16)], fx.Float32))
        for h in range_constexpr(2)
    ]
    v = Vec(rocdl.cvt_scalef32_2xpk16_fp6_f32(T.vec(6, T.i32), src[0], src[1], _raw(scale_f32)))
    return [v[i] for i in range_constexpr(6)]


def _srd_at(t, elem_off, elem_bytes, nrec_bytes):
    base = arith.index_cast(T.i64, buffer_ops.extract_base_index(t))
    boff = arith.index_cast(T.i64, arith.index_cast(T.index, elem_off) * arith.index(elem_bytes))
    raw = arith._to_raw(base + boff)
    r = rocdl.readfirstlane(res=raw.type, src=raw)
    base_v = r.result if hasattr(r, "result") else r
    nr = arith.minui(arith.index_cast(T.index, nrec_bytes), arith.index(_NREC_MAX))
    return buffer_ops.create_buffer_resource_from_addr(base_v, num_records_bytes=nr)


def _build_launch(C, act):
    """Row quantizer for a [R, C] bf16 tensor, R % 32 == 0, C % 1024 == 0. One thread per
    32-element micro-block, a block per 4 rows x 32 micro-blocks, so loads and the A store
    are contiguous per row."""
    CB = C // 32  # micro-blocks per row
    NCH = CB // _MPR
    C8 = CB // 8
    KP64 = C // 128  # 64-byte chunks per packed FP4 row

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def _kern(X: fx.Tensor, OUT: fx.Tensor, SC: fx.Tensor, BIAS: fx.Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        by = bid // NCH
        lr = tid >> 5
        c = (bid % NCH) * _MPR + (tid & 31)
        r = by * _RPB + lr
        row0 = arith.index_cast(T.index, by * _RPB)

        xsrc = _srd_at(X, row0 * arith.index(C // 2), 4, fx.Int32(_RPB * C * 2))
        bits = []
        for q in range_constexpr(4):
            v4 = buffer_ops.buffer_load(xsrc, lr * (C // 2) + c * 16 + q * 4, vec_width=4, dtype=T.i32)
            for j in range_constexpr(4):
                word = v4[j]
                bits.append(word << 16)
                bits.append(word & 0xFFFF0000)
        vf = [Vec.from_elements([b], fx.Int32).bitcast(fx.Float32)[0] for b in bits]
        vf = _rht16_pair(vf)
        native_bits, biased = _compute_scale_native(_amax_bits(vf), BIAS, exp_up=2)
        scale_f32 = arith.bitcast(T.f32, native_bits)

        if act:
            words = _cvt_fp6(vf, scale_f32)
            osrc = _srd_at(OUT, row0 * arith.index(C // 4), 4, fx.Int32(_RPB * C))
            ow = lr * (C // 4) + c * 8
            buffer_ops.buffer_store(Vec.from_elements(words[0:4], fx.Int32), osrc, ow)
            buffer_ops.buffer_store(
                Vec.from_elements([words[4], words[5], fx.Int32(0), fx.Int32(0)], fx.Int32), osrc, ow + 4
            )
        else:
            words = _cvt_fp4(vf, scale_f32)
            # shuffle_weight_w4(16): byte (n, kb) -> [n/16, kb/64, (kb/16)%4, n%16, kb%16]
            osrc = _srd_at(OUT, arith.index(0), 4, fx.Int32(_NREC_MAX))
            ow = ((((r >> 4) * KP64 + (c >> 2)) * 4 + (c & 3)) * 16 + (r & 15)) * 4
            buffer_ops.buffer_store(Vec.from_elements(words, fx.Int32), osrc, ow)

        # shuffle_scale_w4: (r, c) -> [r/32, c/8, c%4, r%16, (c/4)%2, (r/16)%2]
        ssrc = _srd_at(SC, arith.index(0), 1, fx.Int32(_NREC_MAX))
        so = (((((r >> 5) * C8 + (c >> 3)) * 4 + (c & 3)) * 16 + (r & 15)) * 2 + ((c >> 2) & 1)) * 2 + (
            (r >> 4) & 1
        )
        buffer_ops.buffer_store(arith.trunci(T.i8, biased & 0xFF), ssrc, so)

    @flyc.jit
    def _launch(X, OUT, SC, BIAS: fx.Int32, gx: fx.Int32, stream: fx.Stream):
        _kern(X, OUT, SC, BIAS).launch(grid=(gx, 1, 1), block=(BLK, 1, 1), stream=stream)

    return _launch


_COMPILED = {}


def _quant(x, act):
    import torch

    r, c = x.shape
    if r % 32 or c % 1024:
        raise ValueError(f"a6w4 quant needs rows % 32 == 0 and K % 1024 == 0, got {tuple(x.shape)}")
    out = torch.empty(r, c if act else c // 2, dtype=torch.uint8, device=x.device)
    sc = torch.empty(r * c // 32, dtype=torch.uint8, device=x.device)
    bias = 0 if act else 1 << 21
    gx = (r // _RPB) * (c // 32 // _MPR)
    xi, oi = x.view(torch.int32), out.view(torch.int32)
    key = (int(c), bool(act))
    fn = _COMPILED.get(key)
    if fn is None:
        fn = flyc.compile(_build_launch(int(c), bool(act)), xi, oi, sc, bias, gx, torch.cuda.current_stream())
        _COMPILED[key] = fn
    fn(xi, oi, sc, bias, gx, torch._C._cuda_getCurrentRawStream(x.device.index))
    return out, sc


def quant_act_a6w4(x):
    """bf16 [M, K] contiguous, M % 32 == 0, K % 1024 == 0 -> (FP6 A, shuffled A scales) for gemm_a6w4."""
    return _quant(x, True)


def quant_w_a6w4(w):
    """bf16 [N, K] contiguous, N % 32 == 0, K % 1024 == 0 -> (preshuffled FP4 W, shuffled W scales)."""
    return _quant(w, False)
