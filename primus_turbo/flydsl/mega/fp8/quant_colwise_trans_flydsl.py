###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL colwise (along-M) MXFP8 transpose-quant kernel.

dW2 (the fc2 variable-K weight gradient) needs its two bf16 operands quantized
*colwise* -- the E8M0 block groups 32 consecutive M rows and the fp8 output is the
transpose ``[F, M]``.  The production path calls the C++
``grouped_quantize_mxfp8_dual`` which *also* emits the (here unused) rowwise half;
at the DSv3 dW2 shape that dual costs ~1.0 ms and makes the fp8 dW2 net-negative vs
bf16.  This kernel emits *only* the colwise (transposed) operand, roughly halving the
write traffic (HBM floor ~0.37 ms for both dW2 operands), so the fp8 dW2 path can go
net-positive.

Output layout matches the dual's colwise outputs (indices 2,3 of its 8-tuple):
  * Q : fp8 ``[F, M]`` row-major (transpose of the bf16 ``[M, F]`` input)
  * S : raw E8M0 bytes ``[F, M//32]``
The E8M0 + fp8 math mirrors ``quant_flydsl._quant_block_words`` (round-even exp,
target 2^8, soft-clamp).  ``out_dtype`` selects E4M3 (``cvt_pk_fp8_f32``, clamp 448)
or E5M2 (``cvt_pk_bf8_f32``, clamp 57344; the dW2 default = grad range).

v1: one workgroup per 32-M block; each thread owns one output column ``f`` and
privately reduces its 32 M-values.  Consecutive threads own consecutive ``f`` so the
bf16 reads are coalesced (128B/wavefront).  At the DSv3 dW2 shape this is 0.876 ms for
both operands vs the dual's ~1.03 ms (1.17x), exact vs the dual for E5M2 + E4M3.
Single 2D group (caller pads M to a multiple of 32); grouped 128-M padding + an
LDS-staged variant (to close the gap to the ~0.36 ms HBM floor) come next.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr
from flydsl.expr.buffer_ops import buffer_load, buffer_store, create_buffer_resource
from flydsl.expr.rocdl import cvt_pk_bf8_f32, cvt_pk_fp8_f32
from flydsl.expr.typing import Vector as Vec

_BLK = 32  # mxfp8 block (M rows per E8M0 scale)


@functools.lru_cache(maxsize=64)
def _compile_colwise_quant(M: int, F: int, is_e5m2: bool, BT: int = 256):
    assert M % _BLK == 0, f"M={M} must be a multiple of {_BLK}"
    n_mblk = M // _BLK      # E8M0 blocks along M (per output row)
    M_i32 = M // 4          # fp8 output row viewed as i32 words
    blk_i32 = _BLK // 4     # 8 i32 words per 32-block
    fp8_max = 57344.0 if is_e5m2 else 448.0
    cvt = cvt_pk_bf8_f32 if is_e5m2 else cvt_pk_fp8_f32
    # E8M0 scale params (mirror csrc compute_tile_scale): round-even add = 1<<(22-mbits),
    # exponent target = elem emax.  E5M2: mbits=2, emax=15;  E4M3: mbits=3, emax=8.
    mbits = 2 if is_e5m2 else 3
    round_add = 1 << (22 - mbits)
    target_pow2 = 15 if is_e5m2 else 8

    @flyc.kernel(known_block_size=[BT, 1, 1])
    def kern(X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor):
        tid = fx.thread_idx.x
        mb = fx.block_idx.x  # one workgroup per 32-M block (E8M0 block along M)

        xr = create_buffer_resource(X, max_size=True)  # bf16 [M, F]
        qr = create_buffer_resource(Q, max_size=True)  # fp8 [F, M] viewed i32 [F, M//4]
        sr = create_buffer_resource(S, max_size=True)  # raw uint8 [F, M//32]

        lo = fx.arith.constant(-fp8_max, type=fx.T.f32())
        hi = fx.arith.constant(fp8_max, type=fx.T.f32())
        zero_i32 = fx.arith.constant(0, type=fx.T.i32())

        f = tid  # this thread owns output column f (== input free-col); one 32-M block
        while f < fx.Int32(F):
            # x[mb*32, f]; the 32 block rows step by F.  Consecutive threads own
            # consecutive f, so each of the 32 loads is coalesced across the workgroup.
            base = (mb * fx.Int32(_BLK)) * fx.Int32(F) + f

            vals = []
            amax = None
            for i in range_constexpr(_BLK):
                v = buffer_load(xr, base + fx.Int32(i * F), vec_width=1, dtype=fx.T.bf16())
                fv = fx.arith.extf(fx.T.f32(), v)
                vals.append(fv)
                a = fmath.absf(fv)
                amax = a if amax is None else fx.arith.maximumf(amax, a)

            # E8M0 scale: round-even exponent, target 2^emax, clamp to E8M0 range.
            amax_bits = fx.arith.ArithValue(amax).bitcast(fx.T.i32())
            t = amax_bits + fx.Int32(round_add)
            exp = ((t >> fx.Int32(23)) & fx.Int32(0x1FF)) - fx.Int32(127 + target_pow2)
            exp = fx.arith.select(exp < fx.Int32(-127), fx.Int32(-127), exp)
            exp = fx.arith.select(exp > fx.Int32(128), fx.Int32(128), exp)
            biased = fx.arith.ArithValue(exp) + fx.Int32(127)
            scale = (biased << fx.Int32(23)).bitcast(fx.T.f32())  # 2^exp as f32
            inv_scale = fx.Float32(1.0) / fx.arith.ArithValue(scale)

            # quantize + soft-clamp, pack 32 fp8 into 8 i32 words, store contiguous.
            qs = []
            for i in range_constexpr(_BLK):
                q = fmath.clampf(fx.arith.ArithValue(vals[i]) * inv_scale, lo, hi)
                qs.append(fx.arith._to_raw(q))
            base_i32 = f * fx.Int32(M_i32) + mb * fx.Int32(blk_i32)
            words = []
            for wi in range_constexpr(blk_i32):
                j = wi * 4
                w = cvt(fx.T.i32(), qs[j], qs[j + 1], zero_i32, False)
                w = cvt(fx.T.i32(), qs[j + 2], qs[j + 3], w, True)
                words.append(w)
            # two dwordx4 (16B) stores instead of 8 scalar stores (32B = 2x HW max store width).
            buffer_store(Vec.from_elements(words[0:4], fx.Int32).ir_value(), qr, base_i32)
            buffer_store(Vec.from_elements(words[4:8], fx.Int32).ir_value(), qr, base_i32 + fx.Int32(4))

            # store E8M0 byte at S[f, mb].
            buffer_store(fx.arith.ArithValue(biased).trunci(fx.T.i8()), sr, f * fx.Int32(n_mblk) + mb)

            f = f + fx.Int32(BT)

    @flyc.jit
    def launch(X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kern(X, Q, S).launch(grid=(n_mblk, 1, 1), block=(BT, 1, 1), stream=stream)

    return launch


def colwise_quant_mxfp8_flydsl(x: torch.Tensor, out_dtype: torch.dtype):
    """Colwise (along-M) MXFP8 transpose-quant of a single 2D group.

    Args:
        x: bf16 ``[M, F]`` with ``M`` a multiple of 32.
        out_dtype: ``torch.float8_e5m2`` (default dW2 grad range) or ``float8_e4m3fn``.

    Returns ``(q, s)``:
        q: fp8 ``[F, M]`` -- the transpose of ``x``, per-1x32-M E8M0 scaled.
        s: uint8 ``[F, M//32]`` -- raw E8M0 exponent bytes.
    """
    assert x.dim() == 2 and x.dtype == torch.bfloat16, "x must be bf16 [M, F]"
    M, F = x.shape
    is_e5m2 = out_dtype == torch.float8_e5m2
    q = torch.empty((F, M), dtype=out_dtype, device=x.device)
    s = torch.empty((F, M // _BLK), dtype=torch.uint8, device=x.device)
    _compile_colwise_quant(M, F, is_e5m2)(x, q, s)
    return q, s
