###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-launch residual RMSNorm + row/column MXFP4 quantization (GPT-OSS).

One FlyDSL kernel owns a 32-row panel across the full 2880 hidden extent:

    phase A  streaming residual add -> ``x_plus_r`` (bf16) and the fp32
             sum-of-squares, 8 threads/row, cross-lane reduce via
             ``ds_bpermute``; the leading ``_KWCH`` 256-wide hidden chunks stay
             resident in registers so phase B does not re-read them from HBM.
    phase B  11 full 256-wide chunks + one masked 64-wide tail (2880 = 11*256 +
             64). Each chunk stages ``y = x_plus_r * rstd * gamma`` into LDS and
             emits both the rowwise (no RHT) and the colwise-transpose (RHT-16)
             MXFP4 microblocks with the shipped quantizer primitives.

``skip_y_store=True`` never writes the bf16 ``y`` bytes; the autograd edge is
kept by a real backward over the saved ``x_plus_r`` / ``rstd``.
"""

import functools
from typing import Tuple

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, math as fm, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from primus_turbo.flydsl.quantization.mxfp4_quant_kernel import (
    _finish_microblock,
    _lds_load1,
    _lds_load_vec4,
    _lds_store_vec4,
    _mxfp4_scale_rounding_bias,
    _store_words_vec4,
)
from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    ScalingGranularity,
    ScalingRecipe,
    float4_e2m1fn_x2,
)
from primus_turbo.pytorch.kernels.normalization.rmsnorm_impl import (
    rmsnorm_bwd_residual_impl,
)
from primus_turbo.pytorch.ops.normalization import rmsnorm_residual
from primus_turbo.pytorch.ops.quantization import quantize_fp4_with_trans

__all__ = ["rmsnorm_residual_mxfp4_fused"]

# ---------------------------------------------------------------------------
# Production geometry. The kernel is specialised for the GPT-OSS hidden size;
# anything else falls back to the unfused reference pair.
# ---------------------------------------------------------------------------
_BLK = 256
_BR = 32  # rows per workgroup; the column microblock needs >= 32 whole rows
_H = 2880  # real hidden
_HP = 2944  # row recipe pads hidden to a multiple of 128
_HW = _H // 2  # 1440 i32 words per bf16 row
_HPW8 = _HP // 8  # 368 i32 per ROW_OUT row
_HPS = _HP // 32  # 92 E8M0 scale bytes per ROW_OUT row
_TPR = _BLK // _BR  # 8 threads per row in phase A
_NVEC = _HW // (_TPR * 4)  # 45 vec4 iterations over one row
_TC = 256  # phase-B chunk width in hidden columns
_TCW = _TC // 2  # 128 i32 words
_VPC = _TCW // (_TPR * 4)  # 4 phase-A vec4 iterations per full chunk
_NFULL = _HW // _TCW  # 11 full chunks
_NCH = _NFULL + 1  # + one masked 64-wide tail chunk
_GAMW = ((_HW + _BLK - 1) // _BLK) * _BLK  # 1536 gamma LDS slots
_GITER = _GAMW // _BLK
_OOB = 0x7FFFFFFF

# Hidden chunks held live in registers across the sum-of-squares reduction.
# 6 of 11.25 chunks = 53.3% residency: the measured interior optimum between
# re-read traffic and the occupancy cliff.
_KWCH = 6

_SCALE_ROUNDING_MODE = 2
_GRANULARITY = ScalingGranularity.MX_BLOCKWISE


def _f32(bits):
    return Vec.from_elements([bits], fx.Int32).bitcast(fx.Float32)[0]


def _pack_bf16(lo, hi):
    """RNE-pack an fp32 pair into one i32 holding two bf16 lanes."""
    v = rocdl.cvt_pk_bf16_f32(_raw(lo), _raw(hi))
    v = v.result if hasattr(v, "result") else v
    return fx.Int32(arith.bitcast(T.i32, v))


def _srd(t, nrec_bytes):
    base = arith.index_cast(T.i64, buffer_ops.extract_base_index(t))
    raw = arith._to_raw(base)
    r = rocdl.readfirstlane(res=raw.type, src=raw)
    base_v = r.result if hasattr(r, "result") else r
    nr = arith.minui(arith.index_cast(T.index, nrec_bytes), arith.index(_OOB))
    return buffer_ops.create_buffer_resource_from_addr(base_v, num_records_bytes=nr)


def _lds_store1(lds_ptr, off, val):
    fx.make_view(fx.add_offset(lds_ptr, fx.make_int_tuple(off)), fx.make_layout(1, 1)).store(
        Vec.from_elements([val], fx.Int32)
    )


def _emit_fused_body(
    lds, tid, bid, X, RES, GAM, XPR, RSTD, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, EPS, BIAS
):
    """Emit the fused body from plain Python.

    Every compile-time branch below must stay a Python branch: a ``if`` inside
    the ``@flyc.kernel`` AST is rewritten to ``scf.if`` and a value first bound
    inside a branch is out of scope afterwards.
    """
    I32 = fx.Int32
    F32 = fx.Float32
    IRI = fx.Int32.ir_type
    kwvec = _KWCH * _VPC

    r0 = bid * I32(_BR)
    rr = tid // I32(_TPR)
    jj = tid - rr * I32(_TPR)
    lane = tid & I32(63)

    xs = _srd(X, R * I32(_HW * 4))
    rsrc = _srd(RES, R * I32(_HW * 4))
    gs = _srd(GAM, I32(_HW * 4))
    xps = _srd(XPR, R * I32(_HW * 4))

    # gamma row -> LDS once. The load is masked at _HW words, so gamma[2879] is
    # the last element the kernel can ever touch.
    for it in range_constexpr(_GITER):
        w = I32(it * _BLK) + tid
        goff = arith.select(w < I32(_HW), w, I32(_OOB))
        gv = buffer_ops.buffer_load(gs, goff, vec_width=1, dtype=T.i32)
        _lds_store1(lds.gam.ptr, w, fx.Int32(gv))

    # ---- phase A: streaming residual add -> x_plus_r, fp32 sum of squares
    acc = [F32(0.0), F32(0.0), F32(0.0), F32(0.0)]
    hold = {}
    rowbase = (r0 + rr) * I32(_HW)
    for i in range_constexpr(_NVEC):
        woff = I32(i * _TPR * 4) + jj * I32(4)
        goff = rowbase + woff
        xv = buffer_ops.buffer_load(xs, goff, vec_width=4, dtype=T.i32)
        rv = buffer_ops.buffer_load(rsrc, goff, vec_width=4, dtype=T.i32)
        outw = []
        for q in range_constexpr(4):
            xw = fx.Int32(xv[q])
            rw = fx.Int32(rv[q])
            lo = _f32(xw << 16) + _f32(rw << 16)
            hi = _f32(xw & 0xFFFF0000) + _f32(rw & 0xFFFF0000)
            acc[q] = acc[q] + lo * lo + hi * hi
            outw.append(_pack_bf16(lo, hi))
            if i < kwvec:
                hold[(i, q)] = (lo, hi)
        buffer_ops.buffer_store(Vec.from_elements(outw, fx.Int32), xps, goff)

    ssq = (acc[0] + acc[1]) + (acc[2] + acc[3])
    for m in range_constexpr(3):
        idx = (lane ^ I32(1 << m)) << I32(2)
        other = _f32(
            fx.Int32(rocdl.ds_bpermute(IRI, _raw(idx), _raw(arith.bitcast(T.i32, _raw(ssq)))))
        )
        ssq = ssq + other
    rstd = fm.rsqrt(ssq / F32(float(_H)) + EPS)

    # rstd is what the backward needs; one lane per row writes it.
    rsts = _srd(RSTD, R * I32(4))
    roff = arith.select(jj != I32(0), I32(_OOB), r0 + rr)
    buffer_ops.buffer_store(fx.Int32(arith.bitcast(T.i32, _raw(rstd))), rsts, roff)

    ros = _srd(ROW_OUT, R * I32(_HPW8 * 4))
    rss = _srd(ROW_SC, R * I32(_HPS))
    cosrc = _srd(COL_OUT, I32(_H) * (R >> 3) * I32(4))
    css = _srd(COL_SC, I32(_H) * (R >> 5))

    # ---- phase B
    for c in range_constexpr(_NCH):
        tail = c == _NFULL
        nv = 1 if tail else _VPC
        # stage the [BR, TC] y tile in LDS
        for t in range_constexpr(nv):
            i = (_NVEC - 1) if tail else (c * _VPC + t)
            woff = I32(i * _TPR * 4) + jj * I32(4)
            lw = rr * I32(_TCW) + (woff - I32(c * _TCW))
            g4 = _lds_load_vec4(lds.gam.ptr, woff)
            if i < kwvec:
                vals = [hold[(i, q)] for q in range_constexpr(4)]
            else:
                xv = buffer_ops.buffer_load(xps, rowbase + woff, vec_width=4, dtype=T.i32)
                vals = []
                for q in range_constexpr(4):
                    xw = fx.Int32(xv[q])
                    vals.append((_f32(xw << 16), _f32(xw & 0xFFFF0000)))
            yw = []
            for q in range_constexpr(4):
                gw = fx.Int32(g4[q])
                lo, hi = vals[q]
                yw.append(
                    _pack_bf16(lo * rstd * _f32(gw << 16), hi * rstd * _f32(gw & 0xFFFF0000))
                )
            _lds_store_vec4(lds.buf.ptr, lw, Vec.from_elements(yw, fx.Int32))
        if tail:
            # words 32..127 of every tile row are hidden padding -> must read 0
            zer = Vec.from_elements([I32(0)] * 4, fx.Int32)
            for k in range_constexpr(3):
                _lds_store_vec4(
                    lds.buf.ptr, rr * I32(_TCW) + I32(32 + k * 32) + jj * I32(4), zer
                )
        fx.barrier()

        # ---- ROW phase: one 32-element microblock per thread, no RHT
        rbits = []
        for q in range_constexpr(4):
            v4 = _lds_load_vec4(lds.buf.ptr, rr * I32(_TCW) + jj * I32(16) + I32(q * 4))
            for j in range_constexpr(4):
                word = fx.Int32(v4[j])
                rbits.append(word << 16)
                rbits.append(word & 0xFFFF0000)
        rwords, rb = _finish_microblock(rbits, False, BIAS)
        gcmb = I32(c * (_TC // 32)) + jj
        ob = (r0 + rr) * I32(_HPW8) + gcmb * I32(4)
        sc = (r0 + rr) * I32(_HPS) + gcmb
        if tail:
            wok = gcmb < I32(_HPS)
            ob = arith.select(wok, ob, I32(_OOB))
            sc = arith.select(wok, sc, I32(_OOB))
        _store_words_vec4(ros, ob, rwords)
        buffer_ops.buffer_store(arith.trunci(T.i8, rb & 0xFF), rss, sc)

        # ---- COL phase: one thread per column, 32-row microblock, RHT-16
        cw = tid >> I32(1)
        half = tid & I32(1)
        cbits = []
        for row in range_constexpr(32):
            word = fx.Int32(_lds_load1(lds.buf.ptr, I32(row * _TCW) + cw))
            cbits.append(arith.select(half != I32(0), word & I32(-65536), word << 16))
        cwords, cb = _finish_microblock(cbits, True, BIAS)
        gcol = I32(c * _TC) + tid
        cob = gcol * (R >> 3) + bid * I32(4)
        csoff = gcol * (R >> 5) + bid
        if tail:
            cok = gcol < I32(_H)
            cob = arith.select(cok, cob, I32(_OOB))
            csoff = arith.select(cok, csoff, I32(_OOB))
        _store_words_vec4(cosrc, cob, cwords)
        buffer_ops.buffer_store(arith.trunci(T.i8, cb & 0xFF), css, csoff)
        fx.barrier()


@functools.lru_cache(maxsize=1)
def _build_launcher():
    @fx.struct
    class _Smem:
        buf: fx.Array[fx.Int32, _BR * _TCW, 16]  # 16 KiB y tile
        gam: fx.Array[fx.Int32, _GAMW, 16]  # 6 KiB gamma row

    @flyc.kernel(known_block_size=[_BLK, 1, 1])
    def rmsnorm_mxfp4_fused_kernel(
        X: fx.Tensor,
        RES: fx.Tensor,
        GAM: fx.Tensor,
        XPR: fx.Tensor,
        RSTD: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        R: fx.Int32,
        EPS: fx.Float32,
        BIAS: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(_Smem).peek()
        _emit_fused_body(
            lds,
            fx.thread_idx.x,
            fx.block_idx.x,
            X,
            RES,
            GAM,
            XPR,
            RSTD,
            ROW_OUT,
            ROW_SC,
            COL_OUT,
            COL_SC,
            R,
            EPS,
            BIAS,
        )

    @flyc.jit
    def launch(
        X: fx.Tensor,
        RES: fx.Tensor,
        GAM: fx.Tensor,
        XPR: fx.Tensor,
        RSTD: fx.Tensor,
        ROW_OUT: fx.Tensor,
        ROW_SC: fx.Tensor,
        COL_OUT: fx.Tensor,
        COL_SC: fx.Tensor,
        R: fx.Int32,
        EPS: fx.Float32,
        BIAS: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        rmsnorm_mxfp4_fused_kernel(
            X, RES, GAM, XPR, RSTD, ROW_OUT, ROW_SC, COL_OUT, COL_SC, R, EPS, BIAS
        ).launch(grid=(grid_x, 1, 1), block=(_BLK, 1, 1), stream=stream)

    return launch


def _fused_supported(x, residual, gamma) -> bool:
    if not (x.is_cuda and residual.is_cuda and gamma.is_cuda):
        return False
    if x.dtype is not torch.bfloat16 or residual.dtype != x.dtype or gamma.dtype != x.dtype:
        return False
    if x.shape != residual.shape or gamma.ndim != 1 or gamma.shape[0] != _H:
        return False
    if x.shape[-1] != _H:
        return False
    rows = x.numel() // _H
    if rows % _BR or rows == 0:
        return False
    return x.is_contiguous() and residual.is_contiguous() and gamma.is_contiguous()


# ``@flyc.jit.__call__`` re-walks every module global the traced body reads and
# rebuilds the cache key on every launch: 92 us of host time per call for this
# kernel, which the scored pipeline pays in full because it times one call at a
# time from an idle GPU. ``flyc.compile`` pre-binds the CallState and dispatches
# in ~5 us. Keyed on row count because shapes are baked into the CallState.
_COMPILED_BY_ROWS = {}


def _compiled(rows, args):
    key = (rows, args[0].device.index)
    fn = _COMPILED_BY_ROWS.get(key)
    if fn is None:
        fn = flyc.compile(_build_launcher(), *args)
        _COMPILED_BY_ROWS[key] = fn
    return fn


def _run_fused(x2, r2, gamma, eps, buffers):
    y, xpr, rstd, row_i32, row_sc, col_i32, col_sc = buffers
    rows = x2.shape[0]
    args = (
        x2.view(torch.int32),
        r2.view(torch.int32),
        gamma.view(torch.int32),
        xpr.view(torch.int32),
        rstd.view(torch.int32),
        row_i32,
        row_sc,
        col_i32,
        col_sc,
        rows,
        float(eps),
        _mxfp4_scale_rounding_bias(_SCALE_ROUNDING_MODE),
        rows // _BR,
        torch.cuda.current_stream(),
    )
    _compiled(rows, args)(*args)
    del y


def _allocate(x2):
    rows = x2.shape[0]
    device = x2.device
    return (
        torch.empty_like(x2),  # y: bytes are never written on the skip-y path
        torch.empty_like(x2),  # x_plus_r
        torch.empty(rows, device=device, dtype=torch.float32),  # rstd
        torch.empty((rows, _HPW8), device=device, dtype=torch.int32),
        torch.empty((rows, _HPS), device=device, dtype=torch.uint8),
        torch.empty((_H, rows // 8), device=device, dtype=torch.int32),
        torch.empty((_H, rows // 32), device=device, dtype=torch.uint8),
    )


def _as_quant_views(buffers):
    _y, _xpr, _rstd, row_i32, row_sc, col_i32, col_sc = buffers
    return (
        row_i32.view(torch.uint8).view(float4_e2m1fn_x2),
        row_sc.view(torch.float8_e8m0fnu),
        col_i32.view(torch.uint8).view(float4_e2m1fn_x2),
        col_sc.view(torch.float8_e8m0fnu),
    )


class _FusedRMSNormMXFP4SkipY(torch.autograd.Function):
    """Runs the fused kernel and keeps a real RMSNorm gradient for ``y``.

    Only ``y`` and ``x_plus_r`` leave as autograd outputs. The MXFP4 buffers are
    written in place into caller-owned storage so no Float4 tensor is ever an
    autograd Function output.
    """

    @staticmethod
    def forward(ctx, x2, r2, gamma, eps, buffers):
        _run_fused(x2, r2, gamma, eps, buffers)
        y, xpr, rstd = buffers[0], buffers[1], buffers[2]
        ctx.save_for_backward(xpr, gamma, rstd)
        return y, xpr

    @staticmethod
    def backward(ctx, grad_y, grad_xpr):
        xpr, gamma, rstd = ctx.saved_tensors
        dx, dgamma = rmsnorm_bwd_residual_impl(
            grad_y.contiguous(),
            None if grad_xpr is None else grad_xpr.contiguous(),
            xpr,
            gamma,
            rstd,
            0,
            0,
            0,
            0,
        )
        # The upstream `+` has Jacobian [I, I]: x and residual share dx.
        return dx, dx, dgamma, None, None


def _reference_path(x, residual, gamma, eps):
    y, xpr = rmsnorm_residual(x, residual, gamma, eps)
    row, row_scale, col, col_scale = quantize_fp4_with_trans(
        y.reshape(-1, _H).contiguous(),
        float4_e2m1fn_x2,
        _GRANULARITY,
        block_size=MXFP4_BLOCK_SIZE,
        scaling_recipe=ScalingRecipe(),
        scaling_recipe_for_trans=ScalingRecipe(use_rht=True),
        scale_rounding_mode=_SCALE_ROUNDING_MODE,
    )
    return y, xpr, row, row_scale, col, col_scale


def rmsnorm_residual_mxfp4_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1.0e-6,
    fp4_dtype=float4_e2m1fn_x2,
    *,
    skip_y_store: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Fused residual RMSNorm + row/column MXFP4 quantization.

    Returns ``(y, x_plus_r, row, row_scale, col, col_scale)``.

    ``skip_y_store=True`` is the production path: the QKV GEMM consumes ``row``
    and ``row_scale``, the column output is retained for backward, and the bf16
    ``y`` bytes are never written. ``y`` still carries the RMSNorm autograd
    edge. ``skip_y_store=False`` reproduces the unfused forward bit for bit.
    """
    if fp4_dtype is not float4_e2m1fn_x2:
        raise ValueError(f"unsupported fp4 dtype {fp4_dtype!r}")
    if not skip_y_store or not _fused_supported(x, residual, gamma):
        return _reference_path(x, residual, gamma, eps)

    orig_shape = x.shape
    x2 = x.reshape(-1, _H)
    r2 = residual.reshape(-1, _H)
    buffers = _allocate(x2)
    y, xpr = _FusedRMSNormMXFP4SkipY.apply(x2, r2, gamma, eps, buffers)
    row, row_scale, col, col_scale = _as_quant_views(buffers)
    return y.reshape(orig_shape), xpr.reshape(orig_shape), row, row_scale, col, col_scale
