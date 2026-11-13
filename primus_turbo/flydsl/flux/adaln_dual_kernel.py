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

"""FlyDSL kernels for the FLUX dual-stream block's gated residual + second adaLN.

For both streams in one launch the forward computes ``x1 = x + g * a`` (``a`` the attention's output
projection), the modulated LayerNorm ``y = LN(x1) * (1 + s) + h`` that feeds the stream's MLP, and
each row's FP32 mean and rstd for the backward; ``g``, ``s`` and ``h`` are [B, C] chunks of the
stream's modulation GEMM output. The backward computes the gradients of ``x1`` (adding the residual's)
and of ``a``, and the column sums for the shift, scale and gate gradients inside the same row pass,
from the normalized rows it recomputes in registers.

Three waves share each token row, 1,024 columns each (two vectors of 8 BF16 per lane), and add their
row sums through LDS, double-buffered by row parity so one barrier per row suffices; with one wave per
row the three column accumulators would spill out of the backward's registers. The backward
accumulates the shift, scale and gate gradients per lane over the workgroup's 32 rows (all in one
sample of one stream); the caller adds the per-workgroup partials.

Inductor's arithmetic: ``x1`` in FP32, ``1 + s`` in FP32, one rounding per output; each wave's row
mean and squared deviations are combined exactly (Chan et al.) instead of with Welford's update.
``x1`` is stored in BF16.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.math as fmath
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
)

VEC, LANES = 8, 64
# Three waves share each row, 1,024 columns each (two vectors of 8 per lane); they combine their
# row sums through LDS, double-buffered by row parity so one barrier per row suffices.
WAVES = 3
FWD_ROWS, BWD_ROWS = 8, 32  # rows per workgroup, in sequence
N_SUMS = 3  # shift, scale and gate gradients


def adaln_dual_supported(x_img: torch.Tensor, x_txt: torch.Tensor, vectors) -> bool:
    if not (x_img.dim() == 3 and x_txt.dim() == 3):
        return False
    batch, s_img, c = x_img.shape
    s_txt = x_txt.shape[1]
    return (
        all(t.is_cuda and t.dtype == torch.bfloat16 for t in (x_img, x_txt, *vectors))
        and x_txt.shape[0] == batch
        and x_txt.shape[2] == c
        and c % (WAVES * LANES * VEC) == 0
        and all(
            v.dim() == 2
            and v.shape == (batch, c)
            and v.stride(-1) == 1
            and v.stride(0) == vectors[0].stride(0)
            for v in vectors
        )
        and s_img % BWD_ROWS == 0
        and s_txt % BWD_ROWS == 0
        and s_img == s_txt
    )


# LDS exchange: [parity][wave][lane][2] floats; every lane writes its own pair, readers take lane 0's.
lds_floats = 2 * WAVES * LANES * 2


def wave_sum(vs):
    # Sum of the vectors' elements over the 64 lanes; every lane gets the scalar.
    s = ArithValue(_vector.reduction(fx.T.f32, _vector.CombiningKind.ADD, vs[0]))
    for v in vs[1:]:
        s = s.addf(ArithValue(_vector.reduction(fx.T.f32, _vector.CombiningKind.ADD, v)))
    for o in (1, 2, 4, 8, 16, 32):
        s = s.addf(ArithValue(s.shuffle_xor(o, LANES)))
    return s


def exchange(red, it, wave, lane, a, b):
    # Every wave's (a, b), after one barrier; slots alternate with the row's parity.
    base = (ArithValue(it) % fx.Int32(2)) * fx.Int32(WAVES * LANES * 2)
    mine = fx.add_offset(red.ptr, fx.make_int_tuple(base + (wave * fx.Int32(LANES) + lane) * fx.Int32(2)))
    fx.make_view(mine, fx.make_layout(2, 1)).store(Vec.from_elements([a, b], fx.Float32))
    fx.barrier()
    out = []
    for w in range(WAVES):
        p = fx.add_offset(red.ptr, fx.make_int_tuple(base + fx.Int32(w * LANES * 2)))
        v = Vec(fx.make_view(p, fx.make_layout(2, 1)).load())
        out.append((v[0], v[1]))
    return out


def stream_select(blk_row0, n_img_rows, s_img, s_txt):
    # A workgroup's rows lie in one stream: image rows first, then text rows.
    is_img = blk_row0 < fx.Int32(n_img_rows)
    cond = fx.as_ir_value(is_img)
    off = ArithValue(fx.arith.select(cond, fx.Int32(0).ir_value(), fx.Int32(n_img_rows).ir_value()))
    s_seg = ArithValue(fx.arith.select(cond, fx.Int32(s_img).ir_value(), fx.Int32(s_txt).ir_value()))

    def pick(i, t):
        return _llvm.SelectOp(
            cond, create_buffer_resource(i, max_size=True), create_buffer_resource(t, max_size=True)
        ).result

    return pick, off, s_seg


def lane_cols(c, wave, lane):
    per_wave = c // WAVES
    return [
        wave * fx.Int32(per_wave) + (lane + fx.Int32(k * LANES)) * fx.Int32(VEC)
        for k in range(per_wave // (LANES * VEC))
    ]


def _make_fwd_kernel(c, n_img_rows, s_img, s_txt, mod_stride, eps):
    @flyc.kernel(name="flux_adaln_dual_fwd_kernel", known_block_size=[LANES * WAVES, 1, 1])
    def fwd(
        XI: fx.Tensor,
        AI: fx.Tensor,
        GI: fx.Tensor,
        SI: fx.Tensor,
        HI: fx.Tensor,
        X1I: fx.Tensor,
        YI: fx.Tensor,
        XT: fx.Tensor,
        AT: fx.Tensor,
        GT: fx.Tensor,
        ST: fx.Tensor,
        HT: fx.Tensor,
        X1T: fx.Tensor,
        YT: fx.Tensor,
        STATS: fx.Tensor,
    ):
        @fx.struct
        class Smem:
            red: fx.Array[fx.Float32, lds_floats, 16]

        red = fx.SharedAllocator().allocate(Smem).peek().red
        f32v = fx.T.vec(VEC, fx.T.f32)
        bf16v = fx.T.vec(VEC, fx.T.bf16)
        bf16, f32 = fx.T.bf16, fx.T.f32
        t = fx.thread_idx.x
        bx, _, _ = fx.block_idx
        lane, wave = t % fx.Int32(LANES), t // fx.Int32(LANES)
        blk_row0 = bx * fx.Int32(FWD_ROWS)
        pick, off, s_seg = stream_select(blk_row0, n_img_rows, s_img, s_txt)
        x_r, a_r, g_r, s_r, h_r = pick(XI, XT), pick(AI, AT), pick(GI, GT), pick(SI, ST), pick(HI, HT)
        x1_r, y_r = pick(X1I, X1T), pick(YI, YT)
        st_r = create_buffer_resource(STATS, max_size=True)
        seg_row0 = blk_row0 - off
        mbase = (seg_row0 // s_seg) * fx.Int32(mod_stride)
        cols = lane_cols(c, wave, lane)
        one = fx.arith.constant_vector(1.0, f32v)
        n_w = float(c // WAVES)
        inv_nw = fx.arith.constant(1.0 / n_w, type=f32)
        inv_c = fx.arith.constant(1.0 / c, type=f32)
        eps_s = fx.arith.constant(eps, type=f32)
        # The sample's gate, scale and shift, kept packed in BF16 for the workgroup's rows.
        gv, sv, hv = (
            [buffer_load(r, mbase + col, vec_width=VEC, dtype=bf16) for col in cols] for r in (g_r, s_r, h_r)
        )

        def body(it):
            seg = (seg_row0 + ArithValue(it)) * fx.Int32(c)
            row = blk_row0 + ArithValue(it)
            xs = [buffer_load(x_r, seg + col, vec_width=VEC, dtype=bf16) for col in cols]
            as_ = [buffer_load(a_r, seg + col, vec_width=VEC, dtype=bf16) for col in cols]
            x1 = [
                fx.arith.addf(
                    fx.arith.extf(f32v, x), fx.arith.mulf(fx.arith.extf(f32v, g), fx.arith.extf(f32v, a))
                )
                for x, a, g in zip(xs, as_, gv)
            ]
            for v, col in zip(x1, cols):
                buffer_store(fx.arith.trunc_f(bf16v, v), x1_r, seg + col)
            # Each wave's mean and sum of squared deviations over its columns, combined as in
            # Chan et al. (equal counts): exact two-pass statistics of the whole row.
            mean_w = fx.arith.mulf(wave_sum(x1), inv_nw)
            d = [fx.arith.subf(v, _vector.broadcast(f32v, mean_w)) for v in x1]
            m2_w = wave_sum([fx.arith.mulf(v, v) for v in d])
            parts = exchange(red, it, wave, lane, mean_w, m2_w)
            mean_s = (parts[0][0] + parts[1][0] + parts[2][0]) * fx.Float32(1.0 / WAVES)
            m2 = parts[0][1] + parts[1][1] + parts[2][1]
            for mw, _ in parts:
                dm = mw - mean_s
                m2 = m2 + dm * dm * fx.Float32(n_w)
            mean_v = mean_s.ir_value() if hasattr(mean_s, "ir_value") else mean_s
            m2_v = m2.ir_value() if hasattr(m2, "ir_value") else m2
            rstd_s = fmath.rsqrt(fx.arith.addf(fx.arith.mulf(m2_v, inv_c), eps_s))
            rstd = _vector.broadcast(f32v, rstd_s)
            mean = _vector.broadcast(f32v, mean_v)
            for v, s, h, col in zip(x1, sv, hv, cols):
                y = fx.arith.addf(
                    fx.arith.mulf(
                        fx.arith.mulf(fx.arith.subf(v, mean), rstd),
                        fx.arith.addf(fx.arith.extf(f32v, s), one),
                    ),
                    fx.arith.extf(f32v, h),
                )
                buffer_store(fx.arith.trunc_f(bf16v, y), y_r, seg + col)
            # Every lane stores the same two words: no branch needed.
            buffer_store(mean_v, st_r, row * fx.Int32(2))
            buffer_store(rstd_s, st_r, row * fx.Int32(2) + fx.Int32(1))

        for it in range(fx.Int32(FWD_ROWS)):
            body(it)

    return fwd


def _make_bwd_kernel(c, n_img_rows, s_img, s_txt, mod_stride):
    @flyc.kernel(name="flux_adaln_dual_bwd_kernel", known_block_size=[LANES * WAVES, 1, 1])
    def bwd(
        GYI: fx.Tensor,
        GXI: fx.Tensor,
        XI: fx.Tensor,
        AI: fx.Tensor,
        GI: fx.Tensor,
        SI: fx.Tensor,
        DXI: fx.Tensor,
        DAI: fx.Tensor,
        GYT: fx.Tensor,
        GXT: fx.Tensor,
        XT: fx.Tensor,
        AT: fx.Tensor,
        GT: fx.Tensor,
        ST: fx.Tensor,
        DXT: fx.Tensor,
        DAT: fx.Tensor,
        STATS: fx.Tensor,
        PART: fx.Tensor,
    ):
        @fx.struct
        class Smem:
            red: fx.Array[fx.Float32, lds_floats, 16]

        red = fx.SharedAllocator().allocate(Smem).peek().red
        f32v = fx.T.vec(VEC, fx.T.f32)
        f32v4 = fx.T.vec(4, fx.T.f32)
        bf16v = fx.T.vec(VEC, fx.T.bf16)
        bf16, f32 = fx.T.bf16, fx.T.f32
        t = fx.thread_idx.x
        bx, _, _ = fx.block_idx
        lane, wave = t % fx.Int32(LANES), t // fx.Int32(LANES)
        blk_row0 = bx * fx.Int32(BWD_ROWS)
        pick, off, s_seg = stream_select(blk_row0, n_img_rows, s_img, s_txt)
        gy_r, gx_r, x_r, a_r = pick(GYI, GYT), pick(GXI, GXT), pick(XI, XT), pick(AI, AT)
        g_r, s_r, dx_r, da_r = pick(GI, GT), pick(SI, ST), pick(DXI, DXT), pick(DAI, DAT)
        st_r, part_r = (
            create_buffer_resource(STATS, max_size=True),
            create_buffer_resource(PART, max_size=True),
        )
        seg_row0 = blk_row0 - off
        mbase = (seg_row0 // s_seg) * fx.Int32(mod_stride)
        cols = lane_cols(c, wave, lane)
        nv = len(cols)
        one = fx.arith.constant_vector(1.0, f32v)
        zero = fx.arith.constant_vector(0.0, f32v)
        inv_c = fx.arith.constant_vector(1.0 / c, f32v)
        c_v = fx.arith.constant_vector(float(c), f32v)
        gv = [buffer_load(g_r, mbase + col, vec_width=VEC, dtype=bf16) for col in cols]
        sv = [buffer_load(s_r, mbase + col, vec_width=VEC, dtype=bf16) for col in cols]
        accs = [
            [fx.make_rmem_tensor(fx.make_layout(VEC, 1), fx.Float32) for _ in range(nv)]
            for _ in range(N_SUMS)
        ]
        for acc in accs:
            for a in acc:
                fx.memref_store_vec(zero, a)

        def body(it):
            row = blk_row0 + ArithValue(it)
            seg = (seg_row0 + ArithValue(it)) * fx.Int32(c)
            mean = _vector.broadcast(f32v, buffer_load(st_r, row * fx.Int32(2), vec_width=1, dtype=f32))
            rstd = _vector.broadcast(
                f32v, buffer_load(st_r, row * fx.Int32(2) + fx.Int32(1), vec_width=1, dtype=f32)
            )
            gys = [buffer_load(gy_r, seg + col, vec_width=VEC, dtype=bf16) for col in cols]
            xs = [buffer_load(x_r, seg + col, vec_width=VEC, dtype=bf16) for col in cols]
            as_ = [buffer_load(a_r, seg + col, vec_width=VEC, dtype=bf16) for col in cols]
            gxs = [buffer_load(gx_r, seg + col, vec_width=VEC, dtype=bf16) for col in cols]
            gyf = [fx.arith.extf(f32v, g) for g in gys]
            af = [fx.arith.extf(f32v, a) for a in as_]
            xhat = [
                fx.arith.mulf(
                    fx.arith.subf(
                        fx.arith.addf(fx.arith.extf(f32v, x), fx.arith.mulf(fx.arith.extf(f32v, g), a)),
                        mean,
                    ),
                    rstd,
                )
                for x, a, g in zip(xs, af, gv)
            ]
            dxh = [fx.arith.mulf(gy, fx.arith.addf(fx.arith.extf(f32v, s), one)) for gy, s in zip(gyf, sv)]
            parts = exchange(
                red,
                it,
                wave,
                lane,
                wave_sum(dxh),
                wave_sum([fx.arith.mulf(a, h) for a, h in zip(dxh, xhat)]),
            )
            s1_s = parts[0][0] + parts[1][0] + parts[2][0]
            s2_s = parts[0][1] + parts[1][1] + parts[2][1]
            s1 = _vector.broadcast(f32v, s1_s.ir_value() if hasattr(s1_s, "ir_value") else s1_s)
            s2 = _vector.broadcast(f32v, s2_s.ir_value() if hasattr(s2_s, "ir_value") else s2_s)
            # Inductor's native_layer_norm_backward: rstd / C * (C * dxh - s1 - xhat * s2), + residual.
            scale_r = fx.arith.mulf(rstd, inv_c)
            for k, col in enumerate(cols):
                dx = fx.arith.mulf(
                    scale_r,
                    fx.arith.subf(fx.arith.subf(fx.arith.mulf(dxh[k], c_v), s1), fx.arith.mulf(xhat[k], s2)),
                )
                dx = fx.arith.addf(fx.arith.extf(f32v, gxs[k]), dx)
                buffer_store(fx.arith.trunc_f(bf16v, dx), dx_r, seg + col)
                buffer_store(
                    fx.arith.trunc_f(bf16v, fx.arith.mulf(dx, fx.arith.extf(f32v, gv[k]))),
                    da_r,
                    seg + col,
                )
                for acc, v in (
                    (accs[0][k], gyf[k]),
                    (accs[1][k], fx.arith.mulf(gyf[k], xhat[k])),
                    (accs[2][k], fx.arith.mulf(af[k], dx)),
                ):
                    fx.memref_store_vec(fx.arith.addf(fx.memref_load_vec(acc), v), acc)

        for it in range(fx.Int32(BWD_ROWS)):
            body(it)

        # PART [workgroups, 3, C]: the shift-, scale- and gate-gradient column sums over the
        # workgroup's rows; each wave writes its own columns.
        pbase = bx * fx.Int32(N_SUMS * c)
        for k, col in enumerate(cols):
            for j in range(N_SUMS):
                v = fx.memref_load_vec(accs[j][k])
                for half in (0, 1):
                    buffer_store(
                        _vector.extract_strided_slice(f32v4, v, [half * 4], [4], [1]),
                        part_r,
                        pbase + fx.Int32(j * c) + col + fx.Int32(half * 4),
                    )

    return bwd


@flyc.jit
def _compiled_fwd(
    XI,
    AI,
    GI,
    SI,
    HI,
    X1I,
    YI,
    XT,
    AT,
    GT,
    ST,
    HT,
    X1T,
    YT,
    STATS,
    c: fx.Constexpr[int],
    n_rows: fx.Constexpr[int],
    n_img_rows: fx.Constexpr[int],
    s_img: fx.Constexpr[int],
    s_txt: fx.Constexpr[int],
    mod_stride: fx.Constexpr[int],
    eps: fx.Constexpr[float],
    stream: fx.Stream,
):
    _make_fwd_kernel(c, n_img_rows, s_img, s_txt, mod_stride, eps)(
        XI, AI, GI, SI, HI, X1I, YI, XT, AT, GT, ST, HT, X1T, YT, STATS
    ).launch(grid=(n_rows // FWD_ROWS, 1, 1), block=(LANES * WAVES, 1, 1), stream=stream)


@flyc.jit
def _compiled_bwd(
    GYI,
    GXI,
    XI,
    AI,
    GI,
    SI,
    DXI,
    DAI,
    GYT,
    GXT,
    XT,
    AT,
    GT,
    ST,
    DXT,
    DAT,
    STATS,
    PART,
    c: fx.Constexpr[int],
    n_rows: fx.Constexpr[int],
    n_img_rows: fx.Constexpr[int],
    s_img: fx.Constexpr[int],
    s_txt: fx.Constexpr[int],
    mod_stride: fx.Constexpr[int],
    stream: fx.Stream,
):
    _make_bwd_kernel(c, n_img_rows, s_img, s_txt, mod_stride)(
        GYI, GXI, XI, AI, GI, SI, DXI, DAI, GYT, GXT, XT, AT, GT, ST, DXT, DAT, STATS, PART
    ).launch(grid=(n_rows // BWD_ROWS, 1, 1), block=(LANES * WAVES, 1, 1), stream=stream)


def flydsl_adaln_dual_fwd(x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps):
    """For both streams ``x1 = x + g * a`` and ``y = LN(x1) * (1 + s) + h``, and each row's FP32 mean
    and rstd, ``[B * (S_img + S_txt), 2]`` (image rows first). The caller checks
    :func:`adaln_dual_supported`."""
    x_img, a_img, x_txt, a_txt = (t.contiguous() for t in (x_img, a_img, x_txt, a_txt))
    batch, n_img, c = x_img.shape
    n_txt = x_txt.shape[1]
    x1_img, y_img, x1_txt, y_txt = (torch.empty_like(t) for t in (x_img, x_img, x_txt, x_txt))
    n_rows = batch * (n_img + n_txt)
    stats = torch.empty((n_rows, 2), device=x_img.device, dtype=torch.float32)
    _compiled_fwd(
        x_img,
        a_img,
        g_img,
        s_img,
        h_img,
        x1_img,
        y_img,
        x_txt,
        a_txt,
        g_txt,
        s_txt,
        h_txt,
        x1_txt,
        y_txt,
        stats,
        c=c,
        n_rows=n_rows,
        n_img_rows=batch * n_img,
        s_img=n_img,
        s_txt=n_txt,
        mod_stride=g_img.stride(0),
        eps=float(eps),
        stream=torch.cuda.current_stream(x_img.device),
    )
    return x1_img, y_img, x1_txt, y_txt, stats


def flydsl_adaln_dual_bwd(
    gy_img, gx_img, x_img, a_img, g_img, s_img, gy_txt, gx_txt, x_txt, a_txt, g_txt, s_txt, stats
):
    """The gradients of ``x`` and ``a`` for both streams and the FP32 per-workgroup partial sums of the
    shift, scale and gate gradients, ``[B * (S_img + S_txt) / BWD_ROWS, N_SUMS * C]``: image samples
    first, each sample's partials consecutive."""
    gy_img, gx_img, gy_txt, gx_txt = (t.contiguous() for t in (gy_img, gx_img, gy_txt, gx_txt))
    batch, n_img, c = x_img.shape
    n_txt = x_txt.shape[1]
    dx_img, da_img, dx_txt, da_txt = (torch.empty_like(t) for t in (x_img, x_img, x_txt, x_txt))
    n_rows = batch * (n_img + n_txt)
    part = torch.empty((n_rows // BWD_ROWS, N_SUMS * c), device=x_img.device, dtype=torch.float32)
    _compiled_bwd(
        gy_img,
        gx_img,
        x_img,
        a_img,
        g_img,
        s_img,
        dx_img,
        da_img,
        gy_txt,
        gx_txt,
        x_txt,
        a_txt,
        g_txt,
        s_txt,
        dx_txt,
        da_txt,
        stats,
        part,
        c=c,
        n_rows=n_rows,
        n_img_rows=batch * n_img,
        s_img=n_img,
        s_txt=n_txt,
        mod_stride=g_img.stride(0),
        stream=torch.cuda.current_stream(x_img.device),
    )
    return dx_img, da_img, dx_txt, da_txt, part
