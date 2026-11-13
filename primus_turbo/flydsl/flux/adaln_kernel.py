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

"""FlyDSL kernels for the FLUX single-stream block's adaLN: cat + LayerNorm + modulation.

The forward reads the text and image rows, writes copies of them for the residual add and the
modulated LayerNorm of their concatenation, ``LN(cat(x_txt, x_img)) * (1 + scale) + shift``, and
keeps each row's FP32 mean and rstd for the backward. The backward computes the input gradients,
written as separate text and image tensors, and the shift and scale gradients in the same pass.

One wavefront per token row: 64 lanes x 6 vectors of 8 BF16 cover the 3,072 columns, so the row sums
are 64-lane butterflies. The backward accumulates the shift and scale gradients per lane over its
wave's 32 rows (all in one sample); the caller adds the per-wave partials. A backward block's 128 rows
lie in one segment (text or image), so it selects its input and gradient tensors once.

Inductor's arithmetic: ``1 + scale`` rounded to BF16, FP32 inside, one rounding per output; only the
order of the sums differs, so a small fraction of outputs and gradients differ from Inductor's by
one BF16 step, all as close to an FP64 reference as Inductor's.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.math as fmath
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
)

VEC, LANES = 8, 64
FWD_ROWS, FWD_WAVES = 8, 4
BWD_ROWS, BWD_WAVES = 32, 4


def adaln_single_supported(
    x_txt: torch.Tensor, x_img: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> bool:
    if not (x_txt.dim() == 3 and x_img.dim() == 3 and scale.dim() == 2 and shift.dim() == 2):
        return False
    batch, s_txt, c = x_txt.shape
    s_img = x_img.shape[1]
    bwd_block = BWD_ROWS * BWD_WAVES
    return (
        all(t.is_cuda and t.dtype == torch.bfloat16 for t in (x_txt, x_img, scale, shift))
        and x_img.shape[0] == batch
        and x_img.shape[2] == c
        and c % (LANES * VEC) == 0
        and scale.stride(-1) == 1
        and shift.stride(-1) == 1
        and scale.stride(0) == shift.stride(0)
        and all(s % FWD_ROWS == 0 and (batch * s) % (FWD_ROWS * FWD_WAVES) == 0 for s in (s_txt, s_img))
        and s_txt % bwd_block == 0
        and s_img % bwd_block == 0
    )


def wave_sum(vs):
    # Sum of the vectors' elements over the 64 lanes; every lane gets the scalar.
    s = ArithValue(_vector.reduction(fx.T.f32, _vector.CombiningKind.ADD, vs[0]))
    for v in vs[1:]:
        s = s.addf(ArithValue(_vector.reduction(fx.T.f32, _vector.CombiningKind.ADD, v)))
    for o in (1, 2, 4, 8, 16, 32):
        s = s.addf(ArithValue(s.shuffle_xor(o, LANES)))
    return s


def _make_fwd_kernel(c, s_seg, s_out, seq_off, mod_stride, eps):
    nv = c // (LANES * VEC)
    threads = LANES * FWD_WAVES

    @flyc.kernel(name="flux_adaln_single_fwd_kernel", known_block_size=[threads, 1, 1])
    def fwd(
        SRC: fx.Tensor, SCALE: fx.Tensor, SHIFT: fx.Tensor, RES: fx.Tensor, Y: fx.Tensor, STATS: fx.Tensor
    ):
        f32v = fx.T.vec(VEC, fx.T.f32)
        bf16v = fx.T.vec(VEC, fx.T.bf16)
        bf16, f32 = fx.T.bf16, fx.T.f32
        t = fx.thread_idx.x
        bx, _, _ = fx.block_idx
        lane = t % fx.Int32(LANES)
        row0 = (bx * fx.Int32(FWD_WAVES) + t // fx.Int32(LANES)) * fx.Int32(FWD_ROWS)
        b = row0 // fx.Int32(s_seg)
        out0 = b * fx.Int32(s_out) + fx.Int32(seq_off) + row0 % fx.Int32(s_seg)
        cols = [(lane + fx.Int32(k * LANES)) * fx.Int32(VEC) for k in range(nv)]
        src_r, res_r, y_r = (create_buffer_resource(x, max_size=True) for x in (SRC, RES, Y))
        sc_r, sh_r, st_r = (create_buffer_resource(x, max_size=True) for x in (SCALE, SHIFT, STATS))
        one = fx.arith.constant_vector(1.0, f32v)
        inv_c = fx.arith.constant(1.0 / c, type=f32)
        eps_s = fx.arith.constant(eps, type=f32)
        mbase = b * fx.Int32(mod_stride)
        # 1 + scale rounded to BF16, as Inductor stores it; both stay in registers for the wave's rows.
        onep = [
            fx.arith.extf(
                f32v,
                fx.arith.trunc_f(
                    bf16v,
                    fx.arith.addf(
                        fx.arith.extf(f32v, buffer_load(sc_r, mbase + col, vec_width=VEC, dtype=bf16)),
                        one,
                    ),
                ),
            )
            for col in cols
        ]
        shift = [
            fx.arith.extf(f32v, buffer_load(sh_r, mbase + col, vec_width=VEC, dtype=bf16)) for col in cols
        ]

        def body(it):
            src_off = (row0 + ArithValue(it)) * fx.Int32(c)
            out_row = out0 + ArithValue(it)
            out_off = out_row * fx.Int32(c)
            xs = [buffer_load(src_r, src_off + col, vec_width=VEC, dtype=bf16) for col in cols]
            for x, col in zip(xs, cols):
                buffer_store(x, res_r, src_off + col)
            xf = [fx.arith.extf(f32v, x) for x in xs]
            mean_s = fx.arith.mulf(wave_sum(xf), inv_c)
            d = [fx.arith.subf(x, _vector.broadcast(f32v, mean_s)) for x in xf]
            rstd_s = fmath.rsqrt(
                fx.arith.addf(fx.arith.mulf(wave_sum([fx.arith.mulf(v, v) for v in d]), inv_c), eps_s)
            )
            rstd = _vector.broadcast(f32v, rstd_s)
            for v, op, sh, col in zip(d, onep, shift, cols):
                y = fx.arith.addf(fx.arith.mulf(fx.arith.mulf(v, rstd), op), sh)
                buffer_store(fx.arith.trunc_f(bf16v, y), y_r, out_off + col)
            # Every lane stores the same two words: no branch needed.
            buffer_store(mean_s, st_r, out_row * fx.Int32(2))
            buffer_store(rstd_s, st_r, out_row * fx.Int32(2) + fx.Int32(1))

        for it in range(fx.Int32(FWD_ROWS)):
            body(it)

    return fwd


def _make_bwd_kernel(c, s_txt, s_img, mod_stride):
    nv = c // (LANES * VEC)
    threads = LANES * BWD_WAVES
    s_tok = s_txt + s_img

    @flyc.kernel(name="flux_adaln_single_bwd_kernel", known_block_size=[threads, 1, 1])
    def bwd(
        G: fx.Tensor,
        XT: fx.Tensor,
        XI: fx.Tensor,
        GRT: fx.Tensor,
        GRI: fx.Tensor,
        SCALE: fx.Tensor,
        STATS: fx.Tensor,
        DXT: fx.Tensor,
        DXI: fx.Tensor,
        PART: fx.Tensor,
    ):
        f32v = fx.T.vec(VEC, fx.T.f32)
        f32v4 = fx.T.vec(4, fx.T.f32)
        bf16v = fx.T.vec(VEC, fx.T.bf16)
        bf16, f32 = fx.T.bf16, fx.T.f32
        t = fx.thread_idx.x
        bx, _, _ = fx.block_idx
        lane = t % fx.Int32(LANES)
        # The block's rows are in one sample and one segment: choose that segment's tensors once.
        blk_row0 = bx * fx.Int32(BWD_ROWS * BWD_WAVES)
        b = blk_row0 // fx.Int32(s_tok)
        s_blk = blk_row0 % fx.Int32(s_tok)
        is_txt = s_blk < fx.Int32(s_txt)
        cond = fx.as_ir_value(is_txt)

        def pick(a, i):
            return _llvm.SelectOp(
                cond, create_buffer_resource(a, max_size=True), create_buffer_resource(i, max_size=True)
            ).result

        x_r, res_r, dx_r = pick(XT, XI), pick(GRT, GRI), pick(DXT, DXI)
        g_r, sc_r, st_r, part_r = (create_buffer_resource(x, max_size=True) for x in (G, SCALE, STATS, PART))
        seg_len = fx.arith.select(cond, fx.Int32(s_txt).ir_value(), fx.Int32(s_img).ir_value())
        seg_off = fx.arith.select(cond, fx.Int32(0).ir_value(), fx.Int32(s_txt).ir_value())
        wave = bx * fx.Int32(BWD_WAVES) + t // fx.Int32(LANES)
        row0 = wave * fx.Int32(BWD_ROWS)
        seg_row0 = b * ArithValue(seg_len) + row0 % fx.Int32(s_tok) - ArithValue(seg_off)
        cols = [(lane + fx.Int32(k * LANES)) * fx.Int32(VEC) for k in range(nv)]
        one = fx.arith.constant_vector(1.0, f32v)
        zero = fx.arith.constant_vector(0.0, f32v)
        inv_c = fx.arith.constant_vector(1.0 / c, f32v)
        c_v = fx.arith.constant_vector(float(c), f32v)
        mbase = b * fx.Int32(mod_stride)
        # 1 + scale, rounded to BF16 as Inductor stores it, kept packed: 24 registers instead of 48.
        onep = [
            fx.arith.trunc_f(
                bf16v,
                fx.arith.addf(
                    fx.arith.extf(f32v, buffer_load(sc_r, mbase + col, vec_width=VEC, dtype=bf16)), one
                ),
            )
            for col in cols
        ]
        acc_sh = [fx.make_rmem_tensor(fx.make_layout(VEC, 1), fx.Float32) for _ in range(nv)]
        acc_sc = [fx.make_rmem_tensor(fx.make_layout(VEC, 1), fx.Float32) for _ in range(nv)]
        for a in acc_sh + acc_sc:
            fx.memref_store_vec(zero, a)

        def load_row(rsrc, off, policy):
            return [
                buffer_load(rsrc, off + col, vec_width=VEC, dtype=bf16, cache_modifier=policy) for col in cols
            ]

        def terms(gs, xs, mean, rstd):
            gf = [fx.arith.extf(f32v, g) for g in gs]
            xhat = [fx.arith.mulf(fx.arith.subf(fx.arith.extf(f32v, x), mean), rstd) for x in xs]
            dxh = [fx.arith.mulf(g, fx.arith.extf(f32v, op)) for g, op in zip(gf, onep)]
            return gf, xhat, dxh

        def body(it):
            row = row0 + ArithValue(it)
            off = row * fx.Int32(c)
            seg = (seg_row0 + ArithValue(it)) * fx.Int32(c)
            mean = _vector.broadcast(f32v, buffer_load(st_r, row * fx.Int32(2), vec_width=1, dtype=f32))
            rstd = _vector.broadcast(
                f32v, buffer_load(st_r, row * fx.Int32(2) + fx.Int32(1), vec_width=1, dtype=f32)
            )
            # Pass 1: the row sums. Its temporaries die here; pass 2 reloads the row (from L2) with
            # another cache policy, so LLVM cannot merge the loads and keep 144 FP32 values live.
            _, xhat, dxh = terms(load_row(g_r, off, 0), load_row(x_r, seg, 0), mean, rstd)
            s1 = _vector.broadcast(f32v, wave_sum(dxh))
            s2 = _vector.broadcast(f32v, wave_sum([fx.arith.mulf(a, h) for a, h in zip(dxh, xhat)]))
            gf, xhat, dxh = terms(load_row(g_r, off, 1), load_row(x_r, seg, 1), mean, rstd)
            rs = load_row(res_r, seg, 0)
            # Inductor's native_layer_norm_backward: rstd / C * (C * dxh - s1 - xhat * s2), + residual.
            scale_r = fx.arith.mulf(rstd, inv_c)
            for k, col in enumerate(cols):
                dx = fx.arith.mulf(
                    scale_r,
                    fx.arith.subf(fx.arith.subf(fx.arith.mulf(dxh[k], c_v), s1), fx.arith.mulf(xhat[k], s2)),
                )
                dx = fx.arith.addf(fx.arith.extf(f32v, rs[k]), dx)
                buffer_store(fx.arith.trunc_f(bf16v, dx), dx_r, seg + col)
                fx.memref_store_vec(fx.arith.addf(fx.memref_load_vec(acc_sh[k]), gf[k]), acc_sh[k])
                fx.memref_store_vec(
                    fx.arith.addf(fx.memref_load_vec(acc_sc[k]), fx.arith.mulf(gf[k], xhat[k])), acc_sc[k]
                )

        for it in range(fx.Int32(BWD_ROWS)):
            body(it)

        # PART [waves, 2, C]: the wave's shift- and scale-gradient column sums over its rows.
        pbase = wave * fx.Int32(2 * c)
        for k, col in enumerate(cols):
            for j, acc in ((0, acc_sh[k]), (1, acc_sc[k])):
                v = fx.memref_load_vec(acc)
                for half in (0, 1):
                    buffer_store(
                        _vector.extract_strided_slice(f32v4, v, [half * 4], [4], [1]),
                        part_r,
                        pbase + fx.Int32(j * c) + col + fx.Int32(half * 4),
                    )

    return bwd


@flyc.jit
def _compiled_fwd(
    SRC,
    SCALE,
    SHIFT,
    RES,
    Y,
    STATS,
    c: fx.Constexpr[int],
    n_rows: fx.Constexpr[int],
    s_seg: fx.Constexpr[int],
    s_out: fx.Constexpr[int],
    seq_off: fx.Constexpr[int],
    mod_stride: fx.Constexpr[int],
    eps: fx.Constexpr[float],
    stream: fx.Stream,
):
    _make_fwd_kernel(c, s_seg, s_out, seq_off, mod_stride, eps)(SRC, SCALE, SHIFT, RES, Y, STATS).launch(
        grid=(n_rows // (FWD_ROWS * FWD_WAVES), 1, 1), block=(LANES * FWD_WAVES, 1, 1), stream=stream
    )


@flyc.jit
def _compiled_bwd(
    G,
    XT,
    XI,
    GRT,
    GRI,
    SCALE,
    STATS,
    DXT,
    DXI,
    PART,
    c: fx.Constexpr[int],
    n_rows: fx.Constexpr[int],
    s_txt: fx.Constexpr[int],
    s_img: fx.Constexpr[int],
    mod_stride: fx.Constexpr[int],
    stream: fx.Stream,
):
    _make_bwd_kernel(c, s_txt, s_img, mod_stride)(G, XT, XI, GRT, GRI, SCALE, STATS, DXT, DXI, PART).launch(
        grid=(n_rows // (BWD_ROWS * BWD_WAVES), 1, 1), block=(LANES * BWD_WAVES, 1, 1), stream=stream
    )


def flydsl_adaln_single_fwd(x_txt, x_img, scale, shift, eps):
    """``(x_txt, x_img)`` copied, ``LN(cat(x_txt, x_img)) * (1 + scale) + shift`` and each row's FP32
    mean and rstd, ``[B, S, 2]``. The caller checks :func:`adaln_single_supported`."""
    x_txt, x_img = x_txt.contiguous(), x_img.contiguous()
    batch, s_txt, c = x_txt.shape
    s_tok = s_txt + x_img.shape[1]
    r_txt, r_img = torch.empty_like(x_txt), torch.empty_like(x_img)
    y = torch.empty((batch, s_tok, c), device=x_txt.device, dtype=x_txt.dtype)
    stats = torch.empty((batch, s_tok, 2), device=x_txt.device, dtype=torch.float32)
    stream = torch.cuda.current_stream(x_txt.device)
    for src, res, off in ((x_txt, r_txt, 0), (x_img, r_img, s_txt)):
        _compiled_fwd(
            src,
            scale,
            shift,
            res,
            y,
            stats,
            c=c,
            n_rows=batch * src.shape[1],
            s_seg=src.shape[1],
            s_out=s_tok,
            seq_off=off,
            mod_stride=scale.stride(0),
            eps=float(eps),
            stream=stream,
        )
    return r_txt, r_img, y, stats


def flydsl_adaln_single_bwd(g_y, g_r_txt, g_r_img, x_txt, x_img, scale, stats):
    """The input gradients ``(dx_txt, dx_img)`` and the FP32 per-wave partial sums of the shift and
    scale gradients, ``[B, S / BWD_ROWS, 2 C]``, to be summed per sample."""
    g_y, g_r_txt, g_r_img = g_y.contiguous(), g_r_txt.contiguous(), g_r_img.contiguous()
    batch, s_txt, c = x_txt.shape
    s_img = x_img.shape[1]
    s_tok = s_txt + s_img
    dx_txt, dx_img = torch.empty_like(x_txt), torch.empty_like(x_img)
    part = torch.empty((batch, s_tok // BWD_ROWS, 2 * c), device=x_txt.device, dtype=torch.float32)
    _compiled_bwd(
        g_y,
        x_txt,
        x_img,
        g_r_txt,
        g_r_img,
        scale,
        stats,
        dx_txt,
        dx_img,
        part,
        c=c,
        n_rows=batch * s_tok,
        s_txt=s_txt,
        s_img=s_img,
        mod_stride=scale.stride(0),
        stream=torch.cuda.current_stream(x_txt.device),
    )
    return dx_txt, dx_img, part
