###############################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""1x128 FP8 activation quantization for the gfx1250 block-FP8 GEMMs (UE8M0 scales).

One pass over a bf16 tensor x[M, N] (M % 128 == N % 128 == 0) writes
  * the row-quantized copy (1x128 tiles along N): fp8 [M, N] and E8M0 scale bytes,
  * the token-major transposed copy quantized along M (128x1 tiles): fp8 [N, M] and E8M0 scale bytes,
so the wgrad GEMM (reduction over tokens) is a plain NT GEMM with per-row scales on both operands. Each scale tensor
is written in the layout of its consumer GEMM: "g90" (one byte per block) or "packed" (dsv4up pack_scale layout).
Numerics match Turbo's POW2 blockwise quant: scale = pow2_floor(FP8_MAX / max(amax, 1e-4)), q = clamp(x * scale),
scale_inv = 1 / scale (exact). The exponent is taken from the amax bits and the scale applied as an exact power of
two (no division, quantization_fp8blk_utils.ue8m0_exp_g90).

With the Gluon gate on, bf16 inputs run the quant2 core; the Triton kernels below are the path for Triton < 3.9,
PRIMUS_GLUON=0 and inputs the Gluon core does not take.
"""

import logging

import torch
import triton
import triton.language as tl

import primus_turbo.common.envs as envs
from primus_turbo.gluon.gate import gluon_ok, why_not
from primus_turbo.gluon.gemm.gemm_fp8blk_utils import pack_job
from primus_turbo.triton.quantization.quantization_fp8blk_utils import (
    SL_BYTE,
    SL_PACKED,
    fp8_max_exp,
    pow2_neg,
    scale_layout_id,
    store_scale,
    ue8m0_exp_g90,
)


@triton.jit
def _quant_dual_tile(
    x_ptr,
    qr_ptr,
    er_ptr,
    qt_ptr,
    et_ptr,
    pid_m,
    pid_n,
    M,
    N,
    stride_x,
    FP8_MAX: tl.constexpr,
    MAX_EXP: tl.constexpr,
    SLR: tl.constexpr,
    SLT: tl.constexpr,
    TMODE: tl.constexpr,
):
    """Row (1x128) and transposed (128x1) quant of the 128x128 tile (pid_m, pid_n), M % 128 == N % 128 == 0 (no masks).
    The row and the transposed E8M0 bytes go out in their own scale layouts (SLR / SLT: dsv4up packed or g90 byte), so
    each consumer GEMM reads them directly. TMODE 1: four consecutive tokens of a channel are packed into one int32
    before the transpose (4-byte stores)."""
    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * stride_x + offs_n[None, :]).to(tl.float32)
    ax = tl.abs(x)
    nb_n = N // 128
    nb_m = M // 128
    e = ue8m0_exp_g90(tl.max(ax, axis=1), MAX_EXP)  # [128] per row
    q = tl.clamp(x * pow2_neg(e)[:, None], -FP8_MAX, FP8_MAX)
    tl.store(qr_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], q.to(qr_ptr.dtype.element_ty))
    store_scale(er_ptr, offs_m, pid_n, e, M, nb_n, SLR)
    e = ue8m0_exp_g90(tl.max(ax, axis=0), MAX_EXP)  # [128] per column
    q = tl.clamp(x * pow2_neg(e)[None, :], -FP8_MAX, FP8_MAX).to(qt_ptr.dtype.element_ty)
    if TMODE == 1:
        u = tl.reshape(q.to(tl.uint8, bitcast=True).to(tl.int32), (32, 4, 128))
        w = tl.sum(u << (tl.arange(0, 4) * 8)[None, :, None], axis=1)  # [32, 128]: tokens 4a..4a+3
        pw = pid_m * 32 + tl.arange(0, 32)
        tl.store(
            qt_ptr.to(tl.pointer_type(tl.int32)) + offs_n[:, None].to(tl.int64) * (M // 4) + pw[None, :],
            tl.trans(w),
        )
    else:
        tl.store(qt_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :], tl.trans(q))
    store_scale(et_ptr, offs_n, pid_m, e, N, nb_m, SLT)


@triton.jit
def _quant_dual_t2_kernel(
    x_ptr,
    qr_ptr,
    er_ptr,
    qt_ptr,
    et_ptr,
    M,
    N,
    stride_x,
    FP8_MAX: tl.constexpr,
    MAX_EXP: tl.constexpr,
    SLR: tl.constexpr,
    SLT: tl.constexpr,
    TMODE: tl.constexpr,
):
    """Grid (M/128, N/128): one tile per program (see _quant_dual_tile)."""
    _quant_dual_tile(
        x_ptr,
        qr_ptr,
        er_ptr,
        qt_ptr,
        et_ptr,
        tl.program_id(0),
        tl.program_id(1),
        M,
        N,
        stride_x,
        FP8_MAX,
        MAX_EXP,
        SLR,
        SLT,
        TMODE,
    )


@triton.jit
def _quant_dual_t2p_kernel(
    x_ptr,
    qr_ptr,
    er_ptr,
    qt_ptr,
    et_ptr,
    M,
    N,
    stride_x,
    s0p,
    d0p,
    a0,
    b0,
    k0,
    r0,
    n0,
    c0,
    s1p,
    d1p,
    a1,
    b1,
    k1,
    r1,
    n1,
    c1,
    FP8_MAX: tl.constexpr,
    MAX_EXP: tl.constexpr,
    SLR: tl.constexpr,
    SLT: tl.constexpr,
):
    """_quant_dual_t2_kernel (row + transposed, TMODE 0) plus up to two dsv4up weight-scale pack jobs
    (gemm_fp8blk_utils.pack_scales) in the same launch: grid (M/128, N/128 + extra), the extra columns' programs
    (flattened tokens-fastest) run the pack jobs. Same bytes as quant + pack_scales."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    nb_n = N // 128
    nb_m = M // 128
    if pid_n >= nb_n:
        j = (pid_n - nb_n) * nb_m + pid_m
        if j < n0:
            pack_job(s0p, d0p, a0, b0, k0, r0, j, c0, 8)
        elif j < n0 + n1:
            pack_job(s1p, d1p, a1, b1, k1, r1, j - n0, c1, 8)
    else:
        _quant_dual_tile(
            x_ptr, qr_ptr, er_ptr, qt_ptr, et_ptr, pid_m, pid_n, M, N, stride_x, FP8_MAX, MAX_EXP, SLR, SLT, 0
        )


def quant_dual_pack(x, dtype, row_layout, t_layout, wjobs, site=None, num_warps=2):
    """quant_dual_t_e8m0_v2(x, dtype, row_layout, t_layout) (TMODE 0) plus
    gemm_fp8blk_utils.pack_scales(wjobs) (<= 2 jobs (src u8 2D, rows_per_scale, R, K), None entries allowed) in ONE launch.
    Returns ((q_row, None, e8m0_row, q_t, e8m0_t), packed list aligned with wjobs)."""
    assert x.dim() == 2 and x.stride(1) == 1 and len(wjobs) <= 2
    M, N = x.shape
    assert M % 128 == 0 and N % 128 == 0
    if _q2_use(x):
        from primus_turbo.gluon.quantization.quantization_fp8blk_dual_kernel import q2_dense_pack

        return q2_dense_pack(x, dtype, row_layout, t_layout, wjobs)
    v = (
        _MB_QDUAL.get(f"{site}{N}", _MB_QDUAL.get(str(N), _MB_QDUAL.get("*")))
        if (_MB_QDUAL and site)
        else None
    )
    if v is not None and v.startswith("t2w"):
        num_warps = int(v[3:])
    slr, slt = scale_layout_id(row_layout), scale_layout_id(t_layout)
    dev = x.device
    qr = torch.empty((M, N), dtype=dtype, device=dev)
    er = _scale_buf(M, N, slr, dev)
    qt = torch.empty((N, M), dtype=dtype, device=dev)
    et = _scale_buf(N, M, slt, dev)
    outs, args, nprogs = [], [], []
    for jb in wjobs:
        if jb is None:
            outs.append(None)
            continue
        src, rps, R, K = jb
        assert src.dtype == torch.uint8 and src.dim() == 2 and R % 128 == 0 and K % 128 == 0
        KB = K // 128
        assert tuple(src.shape) == ((R // rps), KB)
        dst = torch.empty((R // 128, K * 4), dtype=torch.uint8, device=dev)
        outs.append(dst)
        n_kcb = triton.cdiv(KB, 8)
        nprogs.append((R // 128) * n_kcb)
        args += [src, dst, src.stride(0), src.stride(1), KB, rps, nprogs[-1], n_kcb]
    while len(nprogs) < 2:
        args += [er, er, 0, 0, 0, 1, 0, 1]  # unused slot (uint8 dummies)
        nprogs.append(0)
    extra = triton.cdiv(sum(nprogs), M // 128)
    fmax = float(torch.finfo(dtype).max)
    _quant_dual_t2p_kernel[(M // 128, N // 128 + extra)](
        x,
        qr,
        er,
        qt,
        et,
        M,
        N,
        x.stride(0),
        *args,
        FP8_MAX=fmax,
        MAX_EXP=fp8_max_exp(fmax),
        SLR=slr,
        SLT=slt,
        num_warps=num_warps,
    )
    return (qr, None, er, qt, et), outs


def _mb_qdual_spec():
    """PRIMUS_MB_QDUAL: comma list of key:variant selecting the dual-quant kernel per call
    site and column count N. Keys, first match wins: <site><N> (site f = FP8 Linear forward input, c = concat-N
    Linear forward input, b = dY in backward), <N>, '*'. Variants: t2wW = _quant_dual_t2_kernel with W warps.
    Example: 'f1024:t2w8,f4096:t2w8'. Unlisted keep the caller's kernel."""
    spec = {}
    for item in envs.PRIMUS_MB_QDUAL.split(","):
        if ":" in item:
            k, v = item.split(":", 1)
            spec[k.strip()] = v.strip()
    return spec


_MB_QDUAL = _mb_qdual_spec()


_log = logging.getLogger("primus_turbo.quantization_fp8blk_dual_kernel")
_noted = False


def _q2_use(x) -> bool:
    """bf16 dual quants run the Gluon quant2 core (quantization_fp8blk_dual_kernel.q2_dense / q2_dense_pack, byte-identical to the
    Triton kernels here) whenever the Gluon gate is on and x is 16-byte aligned; otherwise the Triton kernels run."""
    global _noted
    if not gluon_ok():
        if not _noted:
            _noted = True
            _log.info(
                "quantization_fp8blk_dual_kernel: Gluon quant2 is off (%s); using the Triton dual-quant kernels",
                why_not(),
            )
        return False
    return x.dtype == torch.bfloat16 and (x.stride(0) * 2) % 16 == 0 and x.data_ptr() % 16 == 0


def _scale_buf(R, K, sl, dev):
    if sl == SL_PACKED:
        return torch.empty((R // 128, K // 32 * 128), dtype=torch.uint8, device=dev)
    return torch.empty((R, K // 128), dtype=torch.uint8, device=dev)


def quant_dual_t_e8m0_v2(
    x: torch.Tensor,
    dtype=torch.float8_e4m3fn,
    row_layout="g90",
    t_layout="g90",
    tmode=0,
    num_warps=2,
    site=None,
):
    """Row (1x128) and transposed (128x1) UE8M0 quant of x [M, N], M % 128 == N % 128 == 0, with per-output scale
    layouts ``row_layout`` / ``t_layout`` in {"g90" (byte), "packed" (dsv4up)}. Returns
    (q_row, None, e8m0_row, q_t, e8m0_t). ``site`` ("f", "c", "b") and N select the Triton variant through
    PRIMUS_MB_QDUAL when the Gluon quant2 core is off."""
    assert x.dim() == 2 and x.stride(1) == 1
    M, N = x.shape
    assert M % 128 == 0 and N % 128 == 0
    slr, slt = scale_layout_id(row_layout), scale_layout_id(t_layout)
    assert slr in (SL_BYTE, SL_PACKED) and slt in (SL_BYTE, SL_PACKED)
    if _q2_use(x):
        from primus_turbo.gluon.quantization.quantization_fp8blk_dual_kernel import q2_dense

        return q2_dense(x, dtype, row_layout=slr, t_layout=slt)
    dev = x.device
    qr = torch.empty((M, N), dtype=dtype, device=dev)
    er = _scale_buf(M, N, slr, dev)
    qt = torch.empty((N, M), dtype=dtype, device=dev)
    et = _scale_buf(N, M, slt, dev)
    fmax = float(torch.finfo(dtype).max)
    v = _MB_QDUAL.get(f"{site}{N}", _MB_QDUAL.get(str(N), _MB_QDUAL.get("*"))) if _MB_QDUAL else None
    if v is not None and v.startswith("t2w"):
        num_warps, tmode = int(v[3:]), 0
    _quant_dual_t2_kernel[(M // 128, N // 128)](
        x,
        qr,
        er,
        qt,
        et,
        M,
        N,
        x.stride(0),
        FP8_MAX=fmax,
        MAX_EXP=fp8_max_exp(fmax),
        SLR=slr,
        SLT=slt,
        TMODE=tmode,
        num_warps=num_warps,
    )
    return qr, None, er, qt, et


def e8m0_of(scale_inv: torch.Tensor) -> torch.Tensor:
    """fp32 power-of-two scale_inv -> E8M0 exponent bytes (same shape)."""
    return ((scale_inv.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
