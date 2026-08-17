###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL MXFP8 (per-1x32 E8M0) quantization kernels for the fused mega MoE.

Every operand the mxfp8 mega GEMMs consume is block-scaled with one E8M0 byte per 32
elements along the contracted dim.  This module holds all of it: the shared block math,
the rowwise (along-K) quant used by activations and weights, and the colwise (along-M)
transpose-quant the variable-K wgrads need.

E8M0 + fp8 math (mirrors the production HIP kernel
``csrc/kernels/quantization/quantization_mxfp8.hip``, ``compute_tile_scale``)::

    exp = ((float_as_uint(amax) + round_add) >> 23 & 0x1ff) - 127 - target_pow2  # round-even
    exp = clamp(exp, -127, 128);  e8m0_byte = exp + 127;  scale = 2^exp
    q_i = round_to_fp8(x_i / scale)                       # soft-clamped to the fp8 max first

``round_add``/``target_pow2``/the clamp follow the output encoding: E4M3 (``cvt_pk_fp8_f32``,
max 448, target 2^8) or E5M2 (``cvt_pk_bf8_f32``, max 57344, target 2^15; the dW2 default,
matching the grad range).

ROWWISE (along K): one thread owns one 1x32 K-block, so there is no cross-lane reduction.
Optionally fuses the A-scale ScaleS2R preshuffle (``pack=4`` repacks through LDS) so the GEMM's
scale operand comes straight out of the quant kernel.

COLWISE (along M, transposed out): dW2/dW1 contract over M, so their operands need the E8M0
block to group 32 consecutive M rows and the fp8 output to be the transpose ``[F, M]``.  Only
that colwise half is emitted; the production C++ ``grouped_quantize_mxfp8_dual`` also writes a
rowwise half nothing here reads.  Each thread owns one output column and privately reduces its
32 M-values, consecutive threads owning consecutive columns to keep the reads coalesced.  One
workgroup handles one (32-M block, ``BT``-wide F-tile): the kernel is memory-*latency* bound,
so the large grid is what hides the strided-load latency.  A fp8-in variant fuses the
dequant->requant round-trip for the L2-dgrad pool operand.
"""

import functools
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr import arith, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.buffer_ops import buffer_load, buffer_store, create_buffer_resource
from flydsl.expr.rocdl import cvt_f32_fp8, cvt_pk_bf8_f32, cvt_pk_fp8_f32
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.gemm_helper import (
    _PRESHUF_KT,
    build_preshuffle_ab_kernel,
    ceildiv,
    run_compiled,
)

MXFP8_BLOCK = 32  # mxfp8 block (elements per E8M0 scale)
MXFP8_VEC = 8  # sub-vector width for the bf16 load / f32 compute
# Match grouped GEMM / gemm_mxfp8_nt_tile (its local _SCALE_PACK).
MXFP8_SCALE_PACK = 4


# ── shared MXFP8 block math ───────────────────────────────────────────────────────────────


def mxfp8_words_from_f32_subvecs(fvs):
    """Quantize one 1x32 f32 block, given as ``subs`` vectors of width ``MXFP8_VEC``, to E4M3.

    Returns ``(words, biased)``, the layout ``_quant_block_words`` also returns."""
    f32v = fx.T.VectorType.get([MXFP8_VEC], fx.T.f32())
    neg1 = fx.arith.constant_vector(-1.0, f32v)
    lim = fx.arith.constant_vector(448.0, f32v)
    neglim = fx.arith.constant_vector(-448.0, f32v)
    subs = MXFP8_BLOCK // MXFP8_VEC

    sub_amax = []
    for s in range_constexpr(subs):
        fv = fvs[s]
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
    # Build 1/scale from the exponent bits rather than dividing by `bits(biased << 23)`: an
    # all-zero block clamps biased to 0, where the divide form gives 1.0/0.0 = inf and every
    # element quantizes to 0*inf = NaN. Loss-masked tokens have exactly-zero gradient rows, so
    # training hits that case. 2^(127-biased) is also what the stored E8M0 byte means.
    inv_scale = fx.arith.ArithValue((fx.Int32(254) - biased) << fx.Int32(23)).bitcast(fx.T.f32())
    inv_v = _vector.broadcast(f32v, arith._to_raw(inv_scale))

    words = []
    for s in range_constexpr(subs):
        qraw = fx.arith.mulf(fvs[s], inv_v)
        qf = Vec(fx.arith.minimumf(fx.arith.maximumf(qraw, neglim), lim))  # soft-clamp to fp8 max
        e = [qf[i] for i in range_constexpr(MXFP8_VEC)]
        w0 = cvt_pk_fp8_f32(fx.T.i32(), e[0], e[1], fx.Int32(0), False)
        w0 = cvt_pk_fp8_f32(fx.T.i32(), e[2], e[3], w0, True)
        w1 = cvt_pk_fp8_f32(fx.T.i32(), e[4], e[5], fx.Int32(0), False)
        w1 = cvt_pk_fp8_f32(fx.T.i32(), e[6], e[7], w1, True)
        words.append(w0)
        words.append(w1)
    return words, biased


def _quant_block_words(xr, base_elem):
    """Quantize one 1x32 bf16 block at ``xr[base_elem : base_elem+32]`` to MXFP8.

    Returns ``(words, biased)``: ``words`` = 8 i32 packing the 32 E4M3 values (4 fp8/word, via
    the HW ``cvt_pk_fp8_f32``); ``biased`` = the E8M0 scale byte in an i32."""
    f32v = fx.T.VectorType.get([MXFP8_VEC], fx.T.f32())
    subs = MXFP8_BLOCK // MXFP8_VEC

    fvs = []
    for s in range_constexpr(subs):
        vv = buffer_load(xr, base_elem + fx.Int32(s * MXFP8_VEC), vec_width=MXFP8_VEC, dtype=fx.T.bf16())
        fvs.append(fx.arith.extf(f32v, vv))
    return mxfp8_words_from_f32_subvecs(fvs)


def _e8m0_quant_pack(vals, round_add, target_pow2, lo, hi, cvt, zero_i32):
    """Scalar-input MXFP8 block math: 32 f32 ``vals`` -> (8 packed i32 fp8 words, E8M0 biased byte).

    Encoding-parametric twin of ``mxfp8_words_from_f32_subvecs``, which is hard-wired to E4M3 and
    takes vectors; the colwise kernels gather their 32 values stride-wise, one scalar at a time."""
    amax = None
    for fv in vals:
        a = fmath.absf(fv)
        amax = a if amax is None else fx.arith.maximumf(amax, a)
    amax_bits = fx.arith.ArithValue(amax).bitcast(fx.T.i32())
    t = amax_bits + fx.Int32(round_add)
    exp = ((t >> fx.Int32(23)) & fx.Int32(0x1FF)) - fx.Int32(127 + target_pow2)
    exp = fx.arith.select(exp < fx.Int32(-127), fx.Int32(-127), exp)
    exp = fx.arith.select(exp > fx.Int32(128), fx.Int32(128), exp)
    biased = fx.arith.ArithValue(exp) + fx.Int32(127)
    # scale is an exact power of two, so 1/scale = 2^(127-biased) = float bits ((254-biased) << 23):
    # bit-identical to 1.0/scale but a sub+shift instead of an fdiv. See the all-zero-block trap in
    # mxfp8_words_from_f32_subvecs.
    inv_scale = fx.arith.ArithValue((fx.Int32(254) - biased) << fx.Int32(23)).bitcast(fx.T.f32())
    qs = []
    for fv in vals:
        q = fmath.clampf(fx.arith.ArithValue(fv) * inv_scale, lo, hi)
        qs.append(fx.arith._to_raw(q))
    words = []
    for wi in range_constexpr(MXFP8_BLOCK // 4):
        j = wi * 4
        w = cvt(fx.T.i32(), qs[j], qs[j + 1], zero_i32, False)
        w = cvt(fx.T.i32(), qs[j + 2], qs[j + 3], w, True)
        words.append(w)
    return words, biased


# ── A-scale (ScaleS2R) preshuffle slot index ──────────────────────────────────────────────


def _preshuffle_a_pack4_idx(dest_row, kkp, g, K128p):
    """ScaleS2R pack=4 slot for row ``dest_row``, packed-K index ``kkp``, micro-group ``g``."""
    grp = dest_row // fx.Int32(64)
    s_row = (dest_row % fx.Int32(64)) // fx.Int32(16)
    r_row = dest_row % fx.Int32(16)
    lane = g * fx.Int32(16) + r_row
    return ((grp * fx.Int32(K128p) + kkp) * fx.Int32(64) + lane) * fx.Int32(4) + s_row


# ── rowwise (along-K) quant ───────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=32)
def _compile_quant(K: int, BT: int = 256):
    assert K % MXFP8_BLOCK == 0, f"K={K} must be a multiple of {MXFP8_BLOCK}"
    n_blk = K // MXFP8_BLOCK
    K_fp8_i32 = K // 4
    blk_i32 = MXFP8_BLOCK // 4

    @flyc.kernel(known_block_size=[BT, 1, 1])
    def kern(X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor, c_m: fx.Int32):
        tid = fx.thread_idx.x
        row = fx.block_idx.x

        xr = create_buffer_resource(X, max_size=True)
        qr = create_buffer_resource(Q, max_size=True)
        sr = create_buffer_resource(S, max_size=True)

        b = tid
        while b < fx.Int32(n_blk):
            base = row * fx.Int32(K) + b * fx.Int32(MXFP8_BLOCK)
            words, biased = _quant_block_words(xr, base)

            buffer_store(fx.arith.ArithValue(biased).trunci(fx.T.i8()), sr, row * fx.Int32(n_blk) + b)

            base_i32 = row * fx.Int32(K_fp8_i32) + b * fx.Int32(blk_i32)
            for wi in range_constexpr(blk_i32):
                buffer_store(words[wi], qr, base_i32 + fx.Int32(wi))

            b = b + fx.Int32(BT)

    @flyc.jit
    def launch(X: fx.Tensor, Q: fx.Tensor, S: fx.Tensor, M: int, stream: fx.Stream):
        kern(X, Q, S, M).launch(grid=(M, 1, 1), block=(BT, 1, 1), stream=stream)

    return launch


_QUANT_COMPILED: dict = {}
_BSCALE_PS_COMPILED: dict = {}


def preshuffle_b_scale(b_scale: torch.Tensor, G: int, N: int, K: int, *, pack: int = MXFP8_SCALE_PACK):
    """Host preshuffle of a grouped weight E8M0 scale into the ScaleBComb layout-3 ``b_sp``.

    ``b_scale`` = raw E8M0 [G, N, K//32] (or [G*N, K//32]) uint8 -> ``b_sp`` int32
    [b_ngrp*K128*256], b_ngrp=ceildiv(G*N,256)*4, read by ``ScaleBComb``. Runs the shared
    ``build_preshuffle_ab_kernel`` on the B region only (A is a 64-row dummy). Weights are
    static, so callers cache the result per (G,N,K)."""
    GN = G * N
    K128 = K // 128
    K128p = ceildiv(K128, pack)
    dev = b_scale.device
    b_raw = b_scale.contiguous().reshape(GN, K // 32).view(torch.int32).reshape(-1)
    a_ngrp = 1
    a_blocks = a_ngrp * ceildiv(K128, _PRESHUF_KT)
    b_ngrp = ((GN + 255) // 256) * 4
    a_raw = torch.zeros(64 * K128, dtype=torch.int32, device=dev)  # dummy A (64 rows)
    a_sp = torch.zeros(a_ngrp * K128p * 256, dtype=torch.int32, device=dev)
    b_sp = torch.zeros(b_ngrp * K128p * 256, dtype=torch.int32, device=dev)
    pre_kern, n_kt = build_preshuffle_ab_kernel(K128, pack=pack)

    @flyc.jit
    def _launch(
        a_raw, b_raw, a_sp, b_sp, a_blocks: fx.Int32, a_ngrp: fx.Int32, b_ngrp: fx.Int32, stream: fx.Stream
    ):
        pre_kern(a_raw, b_raw, a_sp, b_sp, fx.Int32(64), fx.Int32(GN), a_blocks, a_ngrp, b_ngrp).launch(
            grid=(a_blocks + b_ngrp * n_kt, 1, 1), block=(256, 1, 1), stream=stream
        )

    args = (a_raw, b_raw, a_sp, b_sp, a_blocks, a_ngrp, b_ngrp, torch.cuda.current_stream())
    ck = (GN, K128, pack)
    run_compiled(_BSCALE_PS_COMPILED, ck, _launch, *args)
    return b_sp


def quantize_rowwise_mxfp8_flydsl(x: torch.Tensor):
    """Rowwise MXFP8 quant of ``x`` [M, K] bf16 -> ``(q fp8 [M,K], s uint8 [M, K//32])`` raw E8M0.

    Callers that need the A-scale already in the ScaleS2R broadcast layout get it from the kernel
    that produces the activation (``swiglu_mxfp8_flydsl_kernel`` forward,
    ``swiglu_bwd_rowcol_dual_quant_mxfp8_flydsl`` backward), which fuses the preshuffle in."""
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    x = x.contiguous()
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    q_i32 = q.view(torch.int32)
    s = torch.empty((M, K // MXFP8_BLOCK), dtype=torch.uint8, device=x.device)
    launch = _compile_quant(int(K))
    args = (x, q_i32, s, M, torch.cuda.current_stream())
    ck = (M, K)
    run_compiled(_QUANT_COMPILED, ck, launch, *args)
    return q, s


def quantize_grouped_weight_mxfp8_flydsl(w: torch.Tensor):
    """Per-group MXFP8 quant of grouped weights ``[G, N, K]`` along K (block=32), E4M3.

    Rowwise-along-K quant is per-row independent, so group boundaries don't matter: reshape to
    ``[G*N, K]``, run the rowwise kernel above in ONE launch (not a ``G``-launch Python loop),
    reshape back. The scale is viewed as ``float8_e8m0fnu`` (byte-identical raw E8M0). Returns
    ``(w_fp8 [G,N,K] e4m3, w_scale [G,N,K//32] e8m0)``."""
    assert w.dim() == 3, f"expected 3D [G,N,K], got {tuple(w.shape)}"
    G, N, K = w.shape
    q, s = quantize_rowwise_mxfp8_flydsl(w.reshape(G * N, K))  # q e4m3 [G*N,K], s uint8 [G*N,K//32]
    return q.view(G, N, K), s.view(torch.float8_e8m0fnu).view(G, N, K // MXFP8_BLOCK)


# ── colwise (along-M) transpose-quant, bf16 in ────────────────────────────────────────────


@functools.lru_cache(maxsize=64)
def _compile_colwise_quant_grouped(F: int, is_e5m2: bool, BT: int = 256):
    """Grouped colwise MXFP8 transpose-quant.  One workgroup == one (padded 32-M block,
    BT-wide F-tile).  ``blk2grp[pmb]`` -> group g; the block's 32 padded-M rows map to input
    rows ``group_offs[g] + (pmb*32 - offs_pc[g] + i)`` when that local row < ``group_lens[g]``,
    else they are pad (value 0 => fp8 0).  Runtime row strides (padded total M) come in as the
    ``mpad_i32`` / ``npblk`` scalar args."""
    assert F % BT == 0, f"F={F} must be a multiple of BT={BT}"
    blk_i32 = MXFP8_BLOCK // 4
    n_ftile = F // BT
    fp8_max = 57344.0 if is_e5m2 else 448.0
    cvt = cvt_pk_bf8_f32 if is_e5m2 else cvt_pk_fp8_f32
    mbits = 2 if is_e5m2 else 3
    round_add = 1 << (22 - mbits)
    target_pow2 = 15 if is_e5m2 else 8

    @flyc.kernel(known_block_size=[BT, 1, 1])
    def kern(
        X: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        BLK2GRP: fx.Tensor,
        LENS: fx.Tensor,
        OFFS: fx.Tensor,
        OFFS_PC: fx.Tensor,
        mpad_i32: fx.Int32,
        npblk: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        pmb = bid // fx.Int32(n_ftile)  # padded 32-M block index
        f = (bid % fx.Int32(n_ftile)) * fx.Int32(BT) + tid  # output column (input free-col)

        xr = create_buffer_resource(X, max_size=True)  # bf16 [total_M, F]
        qr = create_buffer_resource(Q, max_size=True)  # fp8 [F, total_M_pad] i32 [F, mpad_i32]
        sr = create_buffer_resource(S, max_size=True)  # raw uint8 [F, npblk]
        b2g = create_buffer_resource(BLK2GRP, max_size=True)  # i32 [npblk] -> group id
        lr = create_buffer_resource(LENS, max_size=True)  # i32 [G] unpadded lens
        ofr = create_buffer_resource(OFFS, max_size=True)  # i32 [G+1] unpadded row offsets
        opr = create_buffer_resource(OFFS_PC, max_size=True)  # i32 [G+1] padded block-row offsets

        lo = fx.arith.constant(-fp8_max, type=fx.T.f32())
        hi = fx.arith.constant(fp8_max, type=fx.T.f32())
        zero_i32 = fx.arith.constant(0, type=fx.T.i32())

        # workgroup-uniform group metadata (pmb == same for all threads in the WG).
        g = buffer_load(b2g, pmb, vec_width=1, dtype=fx.T.i32())
        offs_pc_g = buffer_load(opr, g, vec_width=1, dtype=fx.T.i32())
        in_off_g = buffer_load(ofr, g, vec_width=1, dtype=fx.T.i32())
        len_g = buffer_load(lr, g, vec_width=1, dtype=fx.T.i32())
        m_local0 = pmb * fx.Int32(MXFP8_BLOCK) - fx.arith.ArithValue(offs_pc_g)  # M-offset within group

        vals = []
        for i in range_constexpr(MXFP8_BLOCK):
            m_local = fx.arith.ArithValue(m_local0) + fx.Int32(i)
            real = m_local < fx.arith.ArithValue(len_g)
            # clamp pad rows to the group's first row (in-bounds); select 0 for the value.
            m_eff = fx.arith.select(real, m_local, fx.Int32(0))
            row = fx.arith.ArithValue(in_off_g) + fx.arith.ArithValue(m_eff)
            v = buffer_load(xr, row * fx.Int32(F) + f, vec_width=1, dtype=fx.T.bf16())
            fv = fx.arith.select(real, fx.arith.extf(fx.T.f32(), v), fx.Float32(0.0))
            vals.append(fx.arith._to_raw(fv))

        words, biased = _e8m0_quant_pack(vals, round_add, target_pow2, lo, hi, cvt, zero_i32)
        base_i32 = f * fx.arith.ArithValue(mpad_i32) + pmb * fx.Int32(blk_i32)
        buffer_store(Vec.from_elements(words[0:4], fx.Int32).ir_value(), qr, base_i32)
        buffer_store(Vec.from_elements(words[4:8], fx.Int32).ir_value(), qr, base_i32 + fx.Int32(4))
        buffer_store(fx.arith.ArithValue(biased).trunci(fx.T.i8()), sr, f * fx.arith.ArithValue(npblk) + pmb)

    @flyc.jit
    def launch(X, Q, S, BLK2GRP, LENS, OFFS, OFFS_PC, mpad_i32, npblk, n_pblk, stream: fx.Stream):
        kern(X, Q, S, BLK2GRP, LENS, OFFS, OFFS_PC, mpad_i32, npblk).launch(
            grid=(n_pblk * n_ftile, 1, 1), block=(BT, 1, 1), stream=stream
        )

    return launch


def colwise_grouped_meta(group_lens: torch.Tensor, group_offs: torch.Tensor, pool_rows: Optional[int] = None):
    """Precompute the (device) grouping metadata for the grouped colwise quant.  Both dW2 operands
    share the same group structure, so compute this ONCE and pass it to both calls.

    ``pool_rows`` is the M extent of the pool the groups live in; given it, the padded total is
    bounded without reading the device (each group is padded to 128 here and to BLOCK_M=256 in the
    pool, and 128 divides 256, so the 128-padded total can never exceed the pool). Taking the bound
    avoids a D2H, which would block until every queued kernel retired and serialise the training
    step. Rows past the real groups are masked by ``len_g`` in the kernels.
    """
    dev = group_lens.device
    lens = group_lens.to(torch.int32)
    lens_pc = ((lens + 127) // 128) * 128  # pad each group M to 128
    offs_pc = torch.cat([torch.zeros(1, dtype=torch.int32, device=dev), torch.cumsum(lens_pc, 0)]).to(
        torch.int32
    )
    if pool_rows is None:
        total_M_pad = int(offs_pc[-1].item())  # D2H (sizes output/grid)
    else:
        assert int(pool_rows) % MXFP8_BLOCK == 0, f"pool_rows {pool_rows} must be a multiple of {MXFP8_BLOCK}"
        total_M_pad = int(pool_rows)
    n_pblk = total_M_pad // MXFP8_BLOCK
    # group id per 32-M block via searchsorted on the block offsets: fixed output size, so no
    # hidden D2H (repeat_interleave would sync to size its dynamic output). Blocks past the last
    # real group land on index G, which no per-group table below can hold, so clamp them onto the
    # last group; their rows fail the len_g mask anyway.
    offs32 = group_offs.to(torch.int32)
    blk2grp = (
        (
            torch.searchsorted(
                offs_pc // MXFP8_BLOCK, torch.arange(n_pblk, dtype=torch.int32, device=dev), right=True
            )
            - 1
        )
        .clamp_(0, lens.numel() - 1)
        .to(torch.int32)
    )
    grp = blk2grp
    offs_pc_g = offs_pc[grp]
    in_off_g = offs32[grp]
    len_g = lens[grp]
    m_local0 = torch.arange(n_pblk, dtype=torch.int32, device=dev) * MXFP8_BLOCK - offs_pc_g
    # Per-pmb WG metadata [offs_pc_g, in_off_g, len_g, m_local0] — avoids dependent VMEM chain + barrier.
    pmb_meta = torch.stack([offs_pc_g, in_off_g, len_g, m_local0], dim=1).contiguous()
    return {
        "lens": lens,
        "lens_pc": lens_pc,
        "offs_pc": offs_pc,
        "offs32": offs32,
        "blk2grp": blk2grp,
        "pmb_meta": pmb_meta,
        "total_M_pad": total_M_pad,
        "n_pblk": n_pblk,
    }


# ── fp8-in fused dequant->colwise-requant (a-branch producer fusion) ─────────────────────────
# The dW2 `a` operand comes out of the L2-dgrad pool as ROWWISE MXFP8 (E4M3 [P,H] + E8M0 [P,H//32]),
# the layout that MMA contracts, but dW2 contracts over P and needs it COLWISE. This kernel reads
# the rowwise-fp8 pool directly, dequants in-register (cvt_f32_fp8 * 2^(e8m0-127)) and emits the
# transposed operand, with no bf16 intermediate. The pool fp8 dequants exactly, so the result is
# numerically identical to a dequant->requant round-trip through HBM.


@functools.lru_cache(maxsize=64)
def _compile_colwise_requant_grouped_fp8in(F: int, is_e5m2_out: bool, BT: int = 128, MB: int = 4):
    """Grouped fp8-in colwise MXFP8 transpose-requant with an LDS OUTPUT-transpose stage.

    One workgroup owns (``MB`` padded 32-M blocks) x (BT-wide F-tile).  Each thread owns one output
    column ``f`` and colwise-quantizes its ``MB*32`` M-values (decode ROWWISE-fp8 E4M3 + raw E8M0
    ``2^(e8m0-127)``, then per-32-M-block amax/requant).

    The LDS stage exists for write coalescing: writing each column straight to the transposed output
    ``[F, total_M_pad]`` puts consecutive threads on consecutive output ROWS (stride ``mpad``), so a
    wavefront scatters over 64 cache lines. Staging the fp8 in LDS (``[BT col][MB*8 i32]``) lets the
    write-back re-map threads so 32 consecutive lanes emit 32 consecutive M-i32 of ONE row (a full
    128 B line)."""
    assert F % BT == 0, f"F={F} must be a multiple of BT={BT}"
    assert F % MXFP8_BLOCK == 0, f"F={F} must be a multiple of {MXFP8_BLOCK} (rowwise scale columns)"
    blk_i32 = MXFP8_BLOCK // 4  # fp8 i32 words per 32-block (=8)
    n_ftile = F // BT
    F32 = F // MXFP8_BLOCK  # rowwise E8M0 scale columns (per input row)
    RPT = MB * blk_i32  # output i32 per column (MB blocks * 8 words)
    TILE_I32 = BT * RPT  # LDS out-tile size
    assert TILE_I32 % BT == 0
    n_wr = TILE_I32 // BT  # coalesced-write iterations (= RPT)
    SCPT = BT // MXFP8_BLOCK  # rowwise scale columns per F-tile (shared 32-wide)
    MROWS = MB * MXFP8_BLOCK  # M-rows per workgroup
    SC_N = MROWS * SCPT  # scale slab entries (= MB*BT)
    fp8_max = 57344.0 if is_e5m2_out else 448.0
    cvt = cvt_pk_bf8_f32 if is_e5m2_out else cvt_pk_fp8_f32
    mbits = 2 if is_e5m2_out else 3
    round_add = 1 << (22 - mbits)
    target_pow2 = 15 if is_e5m2_out else 8

    @fx.struct
    class Smem:
        outq: fx.Array[fx.Int32, TILE_I32, 16]  # [BT col][MB*8 i32] fp8 output tile (for coalesced write)
        scale: fx.Array[fx.Int32, SC_N, 16]  # [MROWS][SCPT] staged rowwise E8M0 (cut 32x reload)

    @flyc.kernel(known_block_size=[BT, 1, 1])
    def kern(
        XQ: fx.Tensor,
        XS: fx.Tensor,
        Q: fx.Tensor,
        S: fx.Tensor,
        BLK2GRP: fx.Tensor,
        LENS: fx.Tensor,
        OFFS: fx.Tensor,
        OFFS_PC: fx.Tensor,
        mpad_i32: fx.Int32,
        npblk: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        mwg = bid // fx.Int32(n_ftile)  # M-workgroup (owns MB blocks)
        ftile = bid % fx.Int32(n_ftile)
        f = ftile * fx.Int32(BT) + tid  # output column (input free-col)
        f_base = ftile * fx.Int32(BT)
        pmb0 = mwg * fx.Int32(MB)  # first padded 32-M block

        xqr = create_buffer_resource(XQ, max_size=True)  # fp8 (i8 view) [total_M, F]
        xsr = create_buffer_resource(XS, max_size=True)  # raw E8M0 uint8 [total_M, F//32]
        qr = create_buffer_resource(Q, max_size=True)  # fp8 [F, total_M_pad] i32 [F, mpad_i32]
        sr = create_buffer_resource(S, max_size=True)  # raw uint8 [F, npblk]
        b2g = create_buffer_resource(BLK2GRP, max_size=True)  # i32 [npblk] -> group id
        lr = create_buffer_resource(LENS, max_size=True)  # i32 [G] unpadded lens
        ofr = create_buffer_resource(OFFS, max_size=True)  # i32 [G+1] unpadded row offsets
        opr = create_buffer_resource(OFFS_PC, max_size=True)  # i32 [G+1] padded block-row offsets
        lds = fx.SharedAllocator().allocate(Smem).peek()
        outq_lds = lds.outq
        scale_lds = lds.scale

        lo = fx.arith.constant(-fp8_max, type=fx.T.f32())
        hi = fx.arith.constant(fp8_max, type=fx.T.f32())
        zero_i32 = fx.arith.constant(0, type=fx.T.i32())

        g = buffer_load(b2g, pmb0, vec_width=1, dtype=fx.T.i32())
        offs_pc_g = buffer_load(opr, g, vec_width=1, dtype=fx.T.i32())
        in_off_g = buffer_load(ofr, g, vec_width=1, dtype=fx.T.i32())
        len_g = buffer_load(lr, g, vec_width=1, dtype=fx.T.i32())
        m_local0 = pmb0 * fx.Int32(MXFP8_BLOCK) - fx.arith.ArithValue(offs_pc_g)
        scol_base = ftile * fx.Int32(SCPT)  # first rowwise scale col of this F-tile
        scol_local = tid // fx.Int32(MXFP8_BLOCK)  # this column's scol within the tile

        def row_of(r):
            m_local = fx.arith.ArithValue(m_local0) + r
            real = m_local < fx.arith.ArithValue(len_g)
            m_eff = fx.arith.select(real, m_local, fx.Int32(0))
            return real, fx.arith.ArithValue(in_off_g) + fx.arith.ArithValue(m_eff)

        # ── stage the rowwise E8M0 scale slab (MROWS x SCPT) in LDS once (SC_N entries, MB/thread) ──
        for p in range_constexpr(MB):
            e = tid + fx.Int32(p * BT)
            r = e // fx.Int32(SCPT)
            j = e % fx.Int32(SCPT)
            _, srow = row_of(r)
            se_ld = buffer_load(xsr, srow * fx.Int32(F32) + (scol_base + j), vec_width=1, dtype=fx.T.i8())
            se_ld_i32 = fx.arith.ArithValue(se_ld).extui(fx.T.i32())
            fx.make_view(fx.add_offset(scale_lds.ptr, fx.make_int_tuple(e)), fx.make_layout(1, 1)).store(
                Vec.from_elements([fx.arith._to_raw(se_ld_i32)], fx.Int32)
            )
        fx.gpu.barrier()

        # ── compute: each column's MB blocks -> fp8 words into LDS out-tile + scale to global ──
        for mb in range_constexpr(MB):
            vals = []
            for i in range_constexpr(MXFP8_BLOCK):
                r = mb * MXFP8_BLOCK + i
                real, row = row_of(fx.Int32(r))
                qb = buffer_load(xqr, row * fx.Int32(F) + f, vec_width=1, dtype=fx.T.i8())
                qb_i32 = fx.arith.ArithValue(qb).extui(fx.T.i32())
                fq = cvt_f32_fp8(fx.T.f32(), fx.arith._to_raw(qb_i32), 0)
                sv_s = Vec(
                    fx.make_view(
                        fx.add_offset(scale_lds.ptr, fx.make_int_tuple(fx.Int32(r * SCPT) + scol_local)),
                        fx.make_layout(1, 1),
                    ).load()
                )
                sc = (fx.arith.ArithValue(fx.Int32(sv_s[0])) << fx.Int32(23)).bitcast(fx.T.f32())
                dv = fx.arith.mulf(fx.arith.ArithValue(fq), sc)
                fv = fx.arith.select(real, dv, fx.Float32(0.0))
                vals.append(fx.arith._to_raw(fv))
            words, biased = _e8m0_quant_pack(vals, round_add, target_pow2, lo, hi, cvt, zero_i32)
            lds_base = tid * fx.Int32(RPT) + fx.Int32(mb * blk_i32)
            for w in range_constexpr(blk_i32):
                fx.make_view(
                    fx.add_offset(outq_lds.ptr, fx.make_int_tuple(lds_base + fx.Int32(w))),
                    fx.make_layout(1, 1),
                ).store(Vec.from_elements([words[w]], fx.Int32))
            buffer_store(
                fx.arith.ArithValue(biased).trunci(fx.T.i8()),
                sr,
                f * fx.arith.ArithValue(npblk) + (pmb0 + fx.Int32(mb)),
            )
        fx.gpu.barrier()

        # ── coalesced write: 32 consecutive lanes -> 32 consecutive M-i32 of one output row ──
        for it in range_constexpr(n_wr):
            k = tid + fx.Int32(it * BT)
            col = k // fx.Int32(RPT)
            j = k % fx.Int32(RPT)
            sv = Vec(
                fx.make_view(fx.add_offset(outq_lds.ptr, fx.make_int_tuple(k)), fx.make_layout(1, 1)).load()
            )
            gi = (f_base + col) * fx.arith.ArithValue(mpad_i32) + pmb0 * fx.Int32(blk_i32) + j
            buffer_store(fx.Int32(sv[0]), qr, gi)

    @flyc.jit
    def launch(XQ, XS, Q, S, BLK2GRP, LENS, OFFS, OFFS_PC, mpad_i32, npblk, n_mwg, stream: fx.Stream):
        kern(XQ, XS, Q, S, BLK2GRP, LENS, OFFS, OFFS_PC, mpad_i32, npblk).launch(
            grid=(n_mwg * n_ftile, 1, 1), block=(BT, 1, 1), stream=stream
        )

    return launch


def colwise_requant_mxfp8_grouped_fp8in_flydsl(
    q_in: torch.Tensor,
    s_in: torch.Tensor,
    out_dtype: torch.dtype,
    group_lens: torch.Tensor = None,
    group_offs: torch.Tensor = None,
    meta: dict = None,
    BT: int = 256,
):
    """Grouped fp8-in colwise (along-M) MXFP8 transpose-requant, per-group M padded to 128.

    Takes the operand straight from the L2-dgrad rowwise-fp8 pool, fusing away the
    dequant->bf16->requant round-trip.

    Args:
        q_in: rowwise-fp8 (E4M3) ``[total_M, F]`` (the dispatched-dy pool).
        s_in: raw E8M0 rowwise scale, uint8 ``[total_M, F//32]``.
        out_dtype: colwise output fp8 encoding (``float8_e5m2`` default dW2 / ``float8_e4m3fn``).
        group_lens/group_offs: int ``[G]`` / ``[G+1]`` (unpadded) -- used if ``meta`` is None.
        meta: precomputed ``colwise_grouped_meta`` (share across both dW2 operands).

    Returns ``(q, s, lens_pc, offs_pc)``:
        q: fp8 ``[F, total_M_pad]``   s: uint8 ``[F, total_M_pad//32]``.
    """
    assert q_in.dim() == 2 and s_in.dim() == 2, "q_in [total_M, F], s_in [total_M, F//32]"
    M, F = q_in.shape
    assert s_in.shape[1] == F // MXFP8_BLOCK, f"s_in cols {s_in.shape[1]} != F//32 {F // MXFP8_BLOCK}"
    is_e5m2_out = out_dtype == torch.float8_e5m2
    if meta is None:
        meta = colwise_grouped_meta(group_lens, group_offs, pool_rows=M)
    total_M_pad, n_pblk = meta["total_M_pad"], meta["n_pblk"]
    q = torch.empty((F, total_M_pad), dtype=out_dtype, device=q_in.device)
    s = torch.empty((F, n_pblk), dtype=torch.uint8, device=q_in.device)
    while F % BT != 0:
        BT //= 2
    # MB 32-M blocks per workgroup (contiguous in the transposed output); n_pblk is a multiple of 4
    # (per-group M padded to 128), so MB|n_pblk and each WG's blocks stay within one group.
    MB = 4 if n_pblk % 4 == 0 else (2 if n_pblk % 2 == 0 else 1)
    n_mwg = n_pblk // MB
    xq = q_in.view(torch.uint8)
    xs = s_in.contiguous().view(torch.uint8)
    _compile_colwise_requant_grouped_fp8in(F, is_e5m2_out, BT, MB)(
        xq,
        xs,
        q,
        s,
        meta["blk2grp"],
        meta["lens"],
        meta["offs32"],
        meta["offs_pc"],
        total_M_pad // 4,
        n_pblk,
        n_mwg,
        stream=fx.Stream(None),  # default stream, unchanged
    )
    return q, s, meta["lens_pc"], meta["offs_pc"]


# ── dW2 dual-launch: fp8-in pool requant (a) + bf16 act colwise-quant (b) ────────────────────
# Two independent grids (heavy requant + light LDS-free quant) sharing one ``meta``, back-to-back on
# one stream. Byte-exact to calling the two kernels separately, and unlike a single-workgroup fusion
# it does not hold the requant's 36KB LDS through the act phase.


@functools.lru_cache(maxsize=64)
def _compile_dw2_colwise_dual_launch(
    F_a: int,
    F_b: int,
    is_e5m2_out: bool,
    BT_a: int,
    BT_b: int,
    MB: int,
):
    """Return a launch closure that dispatches requant then quant on the same stream."""
    launch_a = _compile_colwise_requant_grouped_fp8in(F_a, is_e5m2_out, BT_a, MB)
    launch_b = _compile_colwise_quant_grouped(F_b, is_e5m2_out, BT_b)

    def launch(
        XQ,
        XS,
        QA,
        SA,
        XB,
        QB,
        SB,
        BLK2GRP,
        LENS,
        OFFS,
        OFFS_PC,
        mpad_i32,
        npblk,
        n_mwg,
        stream: fx.Stream,
    ):
        launch_a(
            XQ,
            XS,
            QA,
            SA,
            BLK2GRP,
            LENS,
            OFFS,
            OFFS_PC,
            mpad_i32,
            npblk,
            n_mwg,
            stream=stream,
        )
        launch_b(
            XB,
            QB,
            SB,
            BLK2GRP,
            LENS,
            OFFS,
            OFFS_PC,
            mpad_i32,
            npblk,
            npblk,
            stream=stream,
        )

    return launch


def colwise_requant_fp8in_and_quant_bf16_grouped_flydsl(
    q_in: torch.Tensor,
    s_in: torch.Tensor,
    x_bf16: torch.Tensor,
    out_dtype: torch.dtype,
    group_lens: torch.Tensor = None,
    group_offs: torch.Tensor = None,
    meta: dict = None,
    BT: int = 256,
):
    """dW2 colwise operands: pool rowwise-fp8 requant (a) + bf16 act quant (b), dual-launch.

    Uses two independent kernels (heavy requant grid + light quant grid) on one stream with
    shared ``meta``.  Returns ``(a_t, a_ts, b_t, b_ts, lens_pc, offs_pc)``; the ``a`` half is
    byte-exact to ``colwise_requant_mxfp8_grouped_fp8in_flydsl``, and the roles reverse for dW1
    (bf16 operand as ``a``), so callers there read the two halves swapped."""
    assert q_in.dim() == 2 and s_in.dim() == 2 and x_bf16.dim() == 2
    assert x_bf16.dtype == torch.bfloat16
    M, F_a = q_in.shape
    M_b, F_b = x_bf16.shape
    assert M == M_b, f"pool rows {M} != act rows {M_b}"
    assert s_in.shape[1] == F_a // MXFP8_BLOCK
    is_e5m2_out = out_dtype == torch.float8_e5m2
    if meta is None:
        meta = colwise_grouped_meta(group_lens, group_offs, pool_rows=M)
    total_M_pad, n_pblk = meta["total_M_pad"], meta["n_pblk"]
    q_a = torch.empty((F_a, total_M_pad), dtype=out_dtype, device=q_in.device)
    s_a = torch.empty((F_a, n_pblk), dtype=torch.uint8, device=q_in.device)
    q_b = torch.empty((F_b, total_M_pad), dtype=out_dtype, device=q_in.device)
    s_b = torch.empty((F_b, n_pblk), dtype=torch.uint8, device=q_in.device)
    BT_a = BT
    while F_a % BT_a != 0:
        BT_a //= 2
    BT_b = BT
    while F_b % BT_b != 0:
        BT_b //= 2
    MB = 4 if n_pblk % 4 == 0 else (2 if n_pblk % 2 == 0 else 1)
    n_mwg = n_pblk // MB
    x_bf16 = x_bf16.contiguous()
    _compile_dw2_colwise_dual_launch(F_a, F_b, is_e5m2_out, BT_a, BT_b, MB)(
        q_in.view(torch.uint8),
        s_in.contiguous().view(torch.uint8),
        q_a,
        s_a,
        x_bf16,
        q_b,
        s_b,
        meta["blk2grp"],
        meta["lens"],
        meta["offs32"],
        meta["offs_pc"],
        total_M_pad // 4,
        n_pblk,
        n_mwg,
        stream=fx.Stream(None),  # default stream, unchanged
    )
    return q_a, s_a, q_b, s_b, meta["lens_pc"], meta["offs_pc"]


# ── FUSED rowwise + colwise dual-quant (one read of grad_l1 -> both operands) ─────────────────
# grad_l1 [P, F] bf16 is needed BOTH rowwise-preshuffled (E4M3, the L1 fc1-dgrad) and colwise-grouped
# (E5M2, dW1 wgrad). Both come from ONE read via a 32xBT bf16 tile staged in LDS: colwise reads down
# columns, rowwise across each 32-feature block. Rowwise ``q`` matches
# ``quantize_rowwise_mxfp8_flydsl``; the pack=4 ``a_sp`` preshuffle is fused in, one workgroup per
# pool M-block looping all F-tiles then packing, so it stays a single launch.


@functools.lru_cache(maxsize=64)
def compile_rowcol_dual_pack_grouped(F: int, BT: int = 256):
    """Pack=4 a_sp preshuffle for grouped rowcol dual-quant (reads s_raw written by quant kernel)."""
    n_blk = F // MXFP8_BLOCK
    K128p = ceildiv(F // 128, 4)
    n_out_pack = K128p * 4
    n_row_pack_slots = MXFP8_BLOCK * n_out_pack
    n_pack_rounds = (n_row_pack_slots + BT - 1) // BT

    @flyc.kernel(known_block_size=[BT, 1, 1])
    def pack_kern(
        ASP: fx.Tensor,
        SRAW: fx.Tensor,
        PMB_META: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        pmb = bid // fx.Int32(n_pack_rounds)
        pi = bid % fx.Int32(n_pack_rounds)

        aspr = create_buffer_resource(ASP, max_size=True)
        srr = create_buffer_resource(SRAW, max_size=True)
        pmbr = create_buffer_resource(PMB_META, max_size=True)
        mb = pmb * fx.Int32(4)
        in_off_g = buffer_load(pmbr, mb + fx.Int32(1), vec_width=1, dtype=fx.T.i32())
        len_g = buffer_load(pmbr, mb + fx.Int32(2), vec_width=1, dtype=fx.T.i32())
        m_local0 = fx.arith.ArithValue(buffer_load(pmbr, mb + fx.Int32(3), vec_width=1, dtype=fx.T.i32()))

        flat = tid + pi * BT
        if flat < fx.Int32(n_row_pack_slots):
            row_i = flat // fx.Int32(n_out_pack)
            pidx = flat % fx.Int32(n_out_pack)
            m_local = fx.arith.ArithValue(m_local0) + fx.arith.ArithValue(row_i)
            if m_local < fx.arith.ArithValue(len_g):
                global_row = fx.arith.ArithValue(in_off_g) + m_local
                kkp = pidx // fx.Int32(4)
                g_out = pidx % fx.Int32(4)
                packed = fx.Int32(0)
                for bb in range_constexpr(4):
                    raw_b = (kkp * fx.Int32(4) + fx.Int32(bb)) * fx.Int32(4) + g_out
                    scale_byte = buffer_load(
                        srr, global_row * fx.Int32(n_blk) + raw_b, vec_width=1, dtype=fx.T.i8()
                    )
                    b_i32 = fx.arith.extui(fx.T.i32(), scale_byte)
                    packed = packed | ((b_i32 & fx.Int32(0xFF)) << (fx.Int32(bb) * fx.Int32(8)))
                buffer_store(packed, aspr, _preshuffle_a_pack4_idx(global_row, kkp, g_out, K128p))

    @flyc.jit
    def launch_pack(ASP, SRAW, PMB_META, n_pblk, stream: fx.Stream):
        pack_kern(ASP, SRAW, PMB_META).launch(
            grid=(n_pblk * n_pack_rounds, 1, 1), block=(BT, 1, 1), stream=stream
        )

    return launch_pack
