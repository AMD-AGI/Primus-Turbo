###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL rowwise MXFP8 (per-1x32 E8M0) quantization kernel.

Foundational piece for fusing quantization into the fp8 dispatch push: quantize a
bf16 ``[M, K]`` tensor to fp8 (E4M3) + raw E8M0 block scales ``[M, K//32]`` entirely
in a FlyDSL kernel (vs the current torch/HIP quant done as a separate op). One thread
owns one 1x32 K-block (no cross-lane reduction). The E8M0 + fp8 math mirrors the
production HIP kernel (``csrc/kernels/quantization/quantization_mxfp8.hip``):

  exp = ((float_as_uint(amax) + (1<<19)) >> 23 & 0x1ff) - 127 - 8   # round-even, target 2^8
  exp = clamp(exp, -127, 128);  e8m0_byte = exp + 127;  scale = 2^exp
  q_i = round_to_fp8(x_i / scale)

Standalone + validated vs ``quantize_rowwise_mxfp8`` before wiring into the push.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr import arith, range_constexpr
from flydsl.expr.buffer_ops import buffer_load, buffer_store, create_buffer_resource
from flydsl.expr.rocdl import cvt_pk_fp8_f32
from flydsl.expr.typing import Vector as Vec

_BLK = 32  # mxfp8 block (elements per E8M0 scale)
_VEC = 8   # sub-vector width for the bf16 load / f32 compute


def _quant_block_words(xr, base_elem):
    """Quantize one 1x32 bf16 block at ``xr[base_elem : base_elem+32]`` to MXFP8.

    Returns ``(words, biased)``: ``words`` = 8 i32 packing the 32 fp8 (E4M3) values
    (4 fp8/word, via the HW ``cvt_pk_fp8_f32``); ``biased`` = the E8M0 scale byte in an
    i32. Mirrors the production ``compute_tile_scale`` (round-even exp, target 2^8,
    soft-clamp before cvt). Shared by the standalone quant kernel and the fused push."""
    f32v = fx.T.VectorType.get([_VEC], fx.T.f32())
    neg1 = fx.arith.constant_vector(-1.0, f32v)
    lim = fx.arith.constant_vector(448.0, f32v)
    neglim = fx.arith.constant_vector(-448.0, f32v)
    subs = _BLK // _VEC  # 4

    fvs = []
    sub_amax = []
    for s in range_constexpr(subs):
        vv = buffer_load(xr, base_elem + fx.Int32(s * _VEC), vec_width=_VEC, dtype=fx.T.bf16())
        fv = fx.arith.extf(f32v, vv)
        fvs.append(fv)
        av = fx.arith.maximumf(fv, fx.arith.mulf(fv, neg1))  # |fv|
        sub_amax.append(
            fx.arith.ArithValue(_vector.reduction(fx.T.f32(), _vector.CombiningKind.MAXIMUMF, av))
        )
    amax = sub_amax[0]
    for s in range_constexpr(1, subs):
        amax = fx.arith.maximumf(amax, sub_amax[s])

    # E8M0 scale: round-even exponent, target 2^8, clamp to E8M0 range.
    amax_bits = fx.arith.ArithValue(amax).bitcast(fx.T.i32())
    t = amax_bits + fx.Int32(1 << 19)
    exp = ((t >> fx.Int32(23)) & fx.Int32(0x1FF)) - fx.Int32(127 + 8)
    exp = fx.arith.select(exp < fx.Int32(-127), fx.Int32(-127), exp)
    exp = fx.arith.select(exp > fx.Int32(128), fx.Int32(128), exp)
    biased = fx.arith.ArithValue(exp) + fx.Int32(127)
    scale = (biased << fx.Int32(23)).bitcast(fx.T.f32())  # 2^exp as f32
    inv_scale = fx.Float32(1.0) / fx.arith.ArithValue(scale)
    inv_v = _vector.broadcast(f32v, arith._to_raw(inv_scale))

    words = []
    for s in range_constexpr(subs):
        qraw = fx.arith.mulf(fvs[s], inv_v)
        qf = Vec(fx.arith.minimumf(fx.arith.maximumf(qraw, neglim), lim))  # soft-clamp to fp8 max
        e = [qf[i] for i in range_constexpr(_VEC)]
        w0 = cvt_pk_fp8_f32(fx.T.i32(), e[0], e[1], fx.Int32(0), False)
        w0 = cvt_pk_fp8_f32(fx.T.i32(), e[2], e[3], w0, True)
        w1 = cvt_pk_fp8_f32(fx.T.i32(), e[4], e[5], fx.Int32(0), False)
        w1 = cvt_pk_fp8_f32(fx.T.i32(), e[6], e[7], w1, True)
        words.append(w0)
        words.append(w1)
    return words, biased


def _e8m0_broadcast_i32(biased):
    """Broadcast the E8M0 byte (i32) into all 4 bytes of an i32 (ScaleS2R operand)."""
    bb = fx.arith.ArithValue(biased) & fx.Int32(0xFF)
    return bb | (bb << fx.Int32(8)) | (bb << fx.Int32(16)) | (bb << fx.Int32(24))


def _preshuffle_a_idx(dest_row, b, K128):
    """ScaleS2R layout-1 slot for row ``dest_row``, micro-block ``b``:
    ``((grp*K128 + gk)*64 + (g*16+r))*4 + s``, grp=row//64, s=(row%64)//16, r=row%16,
    gk=b//4, g=b%4. Returns the i32 element index into the broadcast a_sp buffer."""
    grp = dest_row // fx.Int32(64)
    s_row = (dest_row % fx.Int32(64)) // fx.Int32(16)
    r_row = dest_row % fx.Int32(16)
    gk = b // fx.Int32(4)
    g = b % fx.Int32(4)
    lane = g * fx.Int32(16) + r_row
    return ((grp * fx.Int32(K128) + gk) * fx.Int32(64) + lane) * fx.Int32(4) + s_row


@functools.lru_cache(maxsize=32)
def _compile_quant(K: int, BT: int = 256, preshuffle: bool = False):
    assert K % _BLK == 0, f"K={K} must be a multiple of {_BLK}"
    n_blk = K // _BLK          # E8M0 blocks (micro-blocks) per row
    K128 = K // 128            # K-iters (= n_blk // 4)
    K_fp8_i32 = K // 4         # fp8 row viewed as i32 words
    blk_i32 = _BLK // 4        # fp8 i32 words per block (=8)

    @flyc.kernel(known_block_size=[BT, 1, 1])
    def kern(X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor, c_m: fx.Int32):
        tid = fx.thread_idx.x
        row = fx.block_idx.x  # one workgroup per token row

        xr = create_buffer_resource(X, max_size=True)  # bf16 [M, K]
        qr = create_buffer_resource(Q, max_size=True)  # fp8 data viewed int32 [M, K//4]
        sr = create_buffer_resource(S, max_size=True)  # raw uint8 [M,K//32] OR broadcast i32 a_sp

        b = tid
        while b < fx.Int32(n_blk):
            base = row * fx.Int32(K) + b * fx.Int32(_BLK)
            words, biased = _quant_block_words(xr, base)

            # store E8M0 scale: raw [row, b] byte, OR broadcast into the ScaleS2R layout-1.
            if preshuffle:
                buffer_store(_e8m0_broadcast_i32(biased), sr, _preshuffle_a_idx(row, b, K128))
            else:
                buffer_store(fx.arith.ArithValue(biased).trunci(fx.T.i8()), sr, row * fx.Int32(n_blk) + b)

            # store the 8 packed fp8 i32 words to Q (int32 view) at this block's columns.
            base_i32 = row * fx.Int32(K_fp8_i32) + b * fx.Int32(blk_i32)
            for wi in range_constexpr(blk_i32):
                buffer_store(words[wi], qr, base_i32 + fx.Int32(wi))

            b = b + fx.Int32(BT)

    @flyc.jit
    def launch(X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor, M: int, stream: fx.Stream = fx.Stream(None)):
        kern(X, Q, S, M).launch(grid=(M, 1, 1), block=(BT, 1, 1), stream=stream)

    return launch


_QUANT_COMPILED: dict = {}
_BSCALE_PS_COMPILED: dict = {}


def preshuffle_b_scale(b_scale: torch.Tensor, G: int, N: int, K: int):
    """Host preshuffle of a grouped weight E8M0 scale into the ScaleBComb layout-3 ``b_sp``.

    ``b_scale`` = raw E8M0 [G, N, K//32] (or [G*N, K//32]) uint8 -> ``b_sp`` int32
    [b_ngrp*K128*256], b_ngrp=ceildiv(G*N,256)*4, read by ``ScaleBComb``. Runs the shared
    ``build_preshuffle_ab_kernel`` (B region only; A is a 64-row dummy). Weights are static,
    so callers cache the result per (G,N,K)."""
    from flydsl.expr.buffer_ops import buffer_load as _bl  # noqa: F401  (ensure module init)
    from primus_turbo.flydsl.utils.gemm_helper import (
        _PRESHUF_KT,
        build_preshuffle_ab_kernel,
        ceildiv,
    )

    GN = G * N
    K128 = K // 128
    dev = b_scale.device
    b_raw = b_scale.contiguous().reshape(GN, K // 32).view(torch.int32).reshape(-1)
    a_ngrp = 1
    a_blocks = a_ngrp * ceildiv(K128, _PRESHUF_KT)
    b_ngrp = ((GN + 255) // 256) * 4
    a_raw = torch.zeros(64 * K128, dtype=torch.int32, device=dev)  # dummy A (64 rows)
    a_sp = torch.zeros(a_ngrp * K128 * 256, dtype=torch.int32, device=dev)
    b_sp = torch.zeros(b_ngrp * K128 * 256, dtype=torch.int32, device=dev)
    pre_kern, n_kt = build_preshuffle_ab_kernel(K128)

    @flyc.jit
    def _launch(a_raw, b_raw, a_sp, b_sp, a_blocks: fx.Int32, a_ngrp: fx.Int32, b_ngrp: fx.Int32,
                stream: fx.Stream = fx.Stream(None)):
        pre_kern(a_raw, b_raw, a_sp, b_sp, fx.Int32(64), fx.Int32(GN), a_blocks, a_ngrp, b_ngrp).launch(
            grid=(a_blocks + b_ngrp * n_kt, 1, 1), block=(256, 1, 1), stream=stream
        )

    args = (a_raw, b_raw, a_sp, b_sp, a_blocks, a_ngrp, b_ngrp, torch.cuda.current_stream())
    ck = (GN, K128)
    compiled = _BSCALE_PS_COMPILED.get(ck)
    if compiled is None:
        compiled = flyc.compile(_launch, *args)
        _BSCALE_PS_COMPILED[ck] = compiled
    compiled(*args)
    return b_sp


def quantize_rowwise_mxfp8_flydsl(x: torch.Tensor, preshuffle: bool = False):
    """Rowwise MXFP8 quant of ``x`` [M, K] bf16 in one FlyDSL kernel.

    ``preshuffle=False``: returns ``(q fp8 [M,K], s uint8 [M, K//32])`` raw E8M0 (matches
    ``quantize_rowwise_mxfp8``). ``preshuffle=True``: the scale is written directly in the
    ScaleS2R broadcast layout-1 ``a_sp`` (int32 [ceildiv(M,64)*K128*256]) so a GEMM can
    read it with ``ScaleS2R`` (no separate preshuffle pass); returns ``(q, a_sp)``."""
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    x = x.contiguous()
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    q_i32 = q.view(torch.int32)  # fp8 data as int32 words (4 fp8/word), for the packed store
    K128 = K // 128
    if preshuffle:
        a_ngrp = (M + 63) // 64
        s = torch.zeros(a_ngrp * K128 * 256, dtype=torch.int32, device=x.device)
    else:
        s = torch.empty((M, K // _BLK), dtype=torch.uint8, device=x.device)
    launch = _compile_quant(int(K), preshuffle=bool(preshuffle))
    args = (x, q_i32, s, M, torch.cuda.current_stream())
    ck = (M, K, preshuffle)
    compiled = _QUANT_COMPILED.get(ck)
    if compiled is None:
        compiled = flyc.compile(launch, *args)
        _QUANT_COMPILED[ck] = compiled
    compiled(*args)
    return q, s
