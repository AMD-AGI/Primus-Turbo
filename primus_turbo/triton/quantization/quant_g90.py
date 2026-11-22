###############################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""1x128 FP8 quantization for the gfx1250 g90 blockwise GEMM (UE8M0, paper recipe).

One pass over a 2D bf16/fp32 tensor x[M, N] writes
  * row-quantized x (1x128 tiles along N): fp8 [M, N], fp32 scale_inv [M, N/128], E8M0 bytes [M, N/128]
  * token-major transposed copy quantized along M (1x128 tiles along M): fp8 [N, M], E8M0 bytes [N, M/128]
so the wgrad GEMM (reduction over tokens) is a plain NT GEMM with per-row scales on both operands,
and no per-call fp32 -> E8M0 conversion is needed. Numerics match Turbo's POW2 blockwise quant:
scale = pow2_floor(FP8_MAX / max(amax, 1e-4)), q = clamp(x * scale), scale_inv = 1 / scale (exact).
The exponent is taken from the amax bits and the scale applied as an exact power of two (no division,
quant_epi_gfx1250.ue8m0_exp_g90); bytes are identical to the former division form.

Options: ``scale_layout="packed"`` writes the E8M0 bytes in the dsv4up pack_scale BK256 layout (4x per 128
block) instead of one byte per block; ``rowmap`` fuses a row gather (MoE permute / dispatch): row i of the
outputs is x[rowmap[i]] (negative -> a zero row).
"""

import os

import torch
import triton
import triton.language as tl

from primus_turbo.triton.quantization.quant_epi_gfx1250 import (
    SL_BYTE,
    SL_PACKED,
    fp8_max_exp,
    pow2_neg,
    pow2_pos,
    scale_layout_id,
    store_scale,
    ue8m0_exp_g90,
)


@triton.jit
def _pow2_floor(s):
    return (s.to(tl.int32, bitcast=True) & 0x7F800000).to(tl.float32, bitcast=True)


@triton.jit
def _quant_dual_t_kernel(
    x_ptr, qr_ptr, sr_ptr, er_ptr, qt_ptr, et_ptr, map_ptr,
    M, N, stride_x,
    BLOCK: tl.constexpr, FP8_MAX: tl.constexpr, WITH_ROW: tl.constexpr, WITH_T: tl.constexpr,
    MAX_EXP: tl.constexpr = 8, SL: tl.constexpr = 1, HAS_MAP: tl.constexpr = 0,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = tl.cast(pid_m * BLOCK + tl.arange(0, BLOCK), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK + tl.arange(0, BLOCK), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if HAS_MAP:
        src = tl.load(map_ptr + offs_m, mask=offs_m < M, other=-1).to(tl.int64)
        lmask = mask & (src >= 0)[:, None]
        x = tl.load(x_ptr + tl.maximum(src, 0)[:, None] * stride_x + offs_n[None, :], mask=lmask,
                    other=0.0).to(tl.float32)
    else:
        x = tl.load(x_ptr + offs_m[:, None] * stride_x + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    ax = tl.abs(x)
    nb_n = tl.cdiv(N, BLOCK)
    nb_m = tl.cdiv(M, BLOCK)
    if WITH_ROW:
        e = ue8m0_exp_g90(tl.max(ax, axis=1), MAX_EXP)  # [BLOCK] per row
        q = tl.clamp(x * pow2_neg(e)[:, None], -FP8_MAX, FP8_MAX)
        tl.store(qr_ptr + offs_m[:, None] * N + offs_n[None, :], q.to(qr_ptr.dtype.element_ty), mask=mask)
        rmask = offs_m < M
        tl.store(sr_ptr + offs_m * nb_n + pid_n, pow2_pos(e), mask=rmask)
        store_scale(er_ptr, offs_m, pid_n, e, M, nb_n, SL, mask=rmask)
    if WITH_T:
        e = ue8m0_exp_g90(tl.max(ax, axis=0), MAX_EXP)  # [BLOCK] per column
        q = tl.clamp(x * pow2_neg(e)[None, :], -FP8_MAX, FP8_MAX)
        qt = tl.trans(q.to(qt_ptr.dtype.element_ty))
        tmask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
        tl.store(qt_ptr + offs_n[:, None] * M + offs_m[None, :], qt, mask=tmask)
        store_scale(et_ptr, offs_n, pid_m, e, N, nb_m, SL, mask=offs_n < N)


def quant_dual_t_e8m0(x: torch.Tensor, dtype=torch.float8_e4m3fn, with_row=True, with_t=True, block=128,
                      scale_layout="g90", rowmap: torch.Tensor = None):
    """Returns (q_row [M,N], scale_inv_row fp32 [M,N/128], e8m0_row u8 [M,N/128], q_t [N,M], e8m0_t u8 [N,M/128]).
    Outputs not requested are None. With ``rowmap`` (int32 [M]) row i is x[rowmap[i]] (negative -> zeros).
    ``scale_layout="packed"``: e8m0_row [M/128, N/32*128], e8m0_t [N/128, M/32*128] (dsv4up layout)."""
    assert x.dim() == 2 and x.stride(1) == 1 and (rowmap is not None or x.is_contiguous())
    sl = scale_layout_id(scale_layout)
    N = x.shape[1]
    M = rowmap.shape[0] if rowmap is not None else x.shape[0]
    nb_m, nb_n = triton.cdiv(M, block), triton.cdiv(N, block)
    if sl == SL_PACKED:
        assert block == 128 and M % 128 == 0 and N % 128 == 0
    else:
        assert sl == SL_BYTE
    dev = x.device
    qr = torch.empty((M, N), dtype=dtype, device=dev) if with_row else None
    sr = torch.empty((M, nb_n), dtype=torch.float32, device=dev) if with_row else None
    qt = torch.empty((N, M), dtype=dtype, device=dev) if with_t else None
    if sl == SL_PACKED:
        er = torch.empty((M // 128, N // 32 * 128), dtype=torch.uint8, device=dev) if with_row else None
        et = torch.empty((N // 128, M // 32 * 128), dtype=torch.uint8, device=dev) if with_t else None
    else:
        er = torch.empty((M, nb_n), dtype=torch.uint8, device=dev) if with_row else None
        et = torch.empty((N, nb_m), dtype=torch.uint8, device=dev) if with_t else None
    dummy = x
    fmax = float(torch.finfo(dtype).max)
    _quant_dual_t_kernel[(nb_m, nb_n)](
        x, qr if with_row else dummy, sr if with_row else dummy, er if with_row else dummy,
        qt if with_t else dummy, et if with_t else dummy, rowmap if rowmap is not None else dummy,
        M, N, x.stride(0), BLOCK=block, FP8_MAX=fmax, WITH_ROW=with_row, WITH_T=with_t,
        MAX_EXP=fp8_max_exp(fmax), SL=sl, HAS_MAP=int(rowmap is not None),
        num_warps=8,
    )
    return qr, sr, er, qt, et


@triton.jit
def _quant_dual_t2_kernel(
    x_ptr, qr_ptr, sr_ptr, er_ptr, qt_ptr, et_ptr,
    M, N, stride_x,
    FP8_MAX: tl.constexpr, WITH_ROW: tl.constexpr, WITH_T: tl.constexpr, WITH_SR: tl.constexpr,
    MAX_EXP: tl.constexpr, SLR: tl.constexpr, SLT: tl.constexpr, TMODE: tl.constexpr,
):
    """fuseq variant of _quant_dual_t_kernel for M % 128 == N % 128 == 0 (no masks): same bytes, plus
    * separate scale layouts for the row (SLR) and transposed (SLT) E8M0 bytes, so each consumer gets its GEMM's
      layout (dsv4up packed or g90 byte) straight from the quant -- no dsv4up pack_scales launch;
    * the fp32 scale_inv is optional (WITH_SR; the g90 / dsv4up GEMMs never read it);
    * TMODE 1: four consecutive tokens of a channel packed into one int32 before the transpose (4-byte stores).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * stride_x + offs_n[None, :]).to(tl.float32)
    ax = tl.abs(x)
    nb_n = N // 128
    nb_m = M // 128
    if WITH_ROW:
        e = ue8m0_exp_g90(tl.max(ax, axis=1), MAX_EXP)  # [128] per row
        q = tl.clamp(x * pow2_neg(e)[:, None], -FP8_MAX, FP8_MAX)
        tl.store(qr_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], q.to(qr_ptr.dtype.element_ty))
        if WITH_SR:
            tl.store(sr_ptr + offs_m * nb_n + pid_n, pow2_pos(e))
        store_scale(er_ptr, offs_m, pid_n, e, M, nb_n, SLR)
    if WITH_T:
        e = ue8m0_exp_g90(tl.max(ax, axis=0), MAX_EXP)  # [128] per column
        q = tl.clamp(x * pow2_neg(e)[None, :], -FP8_MAX, FP8_MAX).to(qt_ptr.dtype.element_ty)
        if TMODE == 1:
            u = tl.reshape(q.to(tl.uint8, bitcast=True).to(tl.int32), (32, 4, 128))
            w = tl.sum(u << (tl.arange(0, 4) * 8)[None, :, None], axis=1)  # [32, 128]: tokens 4a..4a+3
            pw = pid_m * 32 + tl.arange(0, 32)
            tl.store(qt_ptr.to(tl.pointer_type(tl.int32)) + offs_n[:, None].to(tl.int64) * (M // 4) + pw[None, :],
                     tl.trans(w))
        else:
            tl.store(qt_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :], tl.trans(q))
        store_scale(et_ptr, offs_n, pid_m, e, N, nb_m, SLT)


@triton.jit
def _quant_dual_t3_kernel(x_ptr, qr_ptr, er_ptr, qt_ptr, et_ptr, M, N, stride_x,
                          FP8_MAX: tl.constexpr, MAX_EXP: tl.constexpr, SLR: tl.constexpr, SLT: tl.constexpr,
                          BN: tl.constexpr, BM: tl.constexpr):
    """membound (PRIMUS_MB_QDUAL): _quant_dual_t2_kernel's bytes (row + transposed, no fp32 scale_inv, TMODE 0) for
    a BM x BN tile (multiples of 128) per program on a 1-D grid; with more warps this keeps the per-thread tile, the
    VGPRs (~120-200 instead of ~470 at 2 warps) and the unrolled code (11-16 KB instead of 29 KB) small. Large-code
    variants run 3-5x slower in the step when their code is cold (first launch after a GEMM)."""
    pid = tl.program_id(0)
    nbm_t = M // BM
    pid_m = pid % nbm_t  # tokens fastest, like the 2-D grid of _quant_dual_t2_kernel
    pid_n = pid // nbm_t
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    x = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * stride_x + offs_n[None, :]).to(tl.float32)
    ax = tl.abs(x)
    nb_n = N // 128
    nb_m = M // 128
    er = ue8m0_exp_g90(tl.max(tl.reshape(ax, (BM, BN // 128, 128)), axis=2), MAX_EXP)  # [BM, BN/128]
    q = tl.clamp(tl.reshape(x, (BM, BN // 128, 128)) * pow2_neg(er)[:, :, None], -FP8_MAX, FP8_MAX)
    tl.store(qr_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :],
             tl.reshape(q, (BM, BN)).to(qr_ptr.dtype.element_ty))
    kb = pid_n * (BN // 128) + tl.arange(0, BN // 128)
    r64 = offs_m.to(tl.int64)[:, None]
    if SLR == 2:
        w = (r64 >> 7) * (nb_n * 128) + kb[None, :] * 128 + (r64 & 31) * 4 + ((r64 >> 5) & 3)
        tl.store(er_ptr.to(tl.pointer_type(tl.int32)) + w, ((er + 127) & 0xFF) * 0x01010101)
    else:
        tl.store(er_ptr + r64 * nb_n + kb[None, :], ((er + 127) & 0xFF).to(tl.uint8))
    et = ue8m0_exp_g90(tl.max(tl.reshape(ax, (BM // 128, 128, BN)), axis=1), MAX_EXP)  # [BM/128, BN]
    qc = tl.clamp(tl.reshape(x, (BM // 128, 128, BN)) * pow2_neg(et)[:, None, :], -FP8_MAX, FP8_MAX)
    qc = tl.reshape(qc.to(qt_ptr.dtype.element_ty), (BM, BN))
    tl.store(qt_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :], tl.trans(qc))
    mb = pid_m * (BM // 128) + tl.arange(0, BM // 128)
    c64 = offs_n.to(tl.int64)[None, :]
    if SLT == 2:
        w = (c64 >> 7) * (nb_m * 128) + mb[:, None] * 128 + (c64 & 31) * 4 + ((c64 >> 5) & 3)
        tl.store(et_ptr.to(tl.pointer_type(tl.int32)) + w, ((et + 127) & 0xFF) * 0x01010101)
    else:
        tl.store(et_ptr + c64 * nb_m + mb[:, None], ((et + 127) & 0xFF).to(tl.uint8))


@triton.jit
def _mb_pack_job(src_ptr, dst_ptr, s0, s1, KB, RPS, pid, n_kcb):
    """dsv4up_pack._pack_job (BKC = 8): 128-row block c, 8 K-chunks of 128 -> packed int32 words (same bytes)."""
    c = pid // n_kcb
    kc = (pid % n_kcb) * 8 + tl.arange(0, 8)
    t = tl.arange(0, 128)
    row = c * 128 + (t % 4) * 32 + t // 4
    srow = row // RPS
    kmask = kc < KB
    v = tl.load(src_ptr + srow[None, :].to(tl.int64) * s0 + kc[:, None].to(tl.int64) * s1, mask=kmask[:, None],
                other=127)
    w = v.to(tl.int32) * 0x01010101
    dst = dst_ptr.to(tl.pointer_type(tl.int32)) + c.to(tl.int64) * KB * 128 + kc[:, None].to(tl.int64) * 128 + t[None, :]
    tl.store(dst, w, mask=kmask[:, None])


@triton.jit
def _quant_dual_t2p_kernel(x_ptr, qr_ptr, er_ptr, qt_ptr, et_ptr, M, N, stride_x,
                           s0p, d0p, a0, b0, k0, r0, n0, c0, s1p, d1p, a1, b1, k1, r1, n1, c1,
                           FP8_MAX: tl.constexpr, MAX_EXP: tl.constexpr, SLR: tl.constexpr, SLT: tl.constexpr):
    """membound (PRIMUS_MB_PACKQ): _quant_dual_t2_kernel (row + transposed, TMODE 0, no scale_inv) plus up to two
    dsv4up weight-scale pack jobs (dsv4up_pack.pack_scales) in the same launch: grid (M/128, N/128 + extra), the
    extra columns' programs (flattened tokens-fastest) run the pack jobs. Same bytes as quant + pack_scales."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    nb_n = N // 128
    nb_m = M // 128
    if pid_n >= nb_n:
        j = (pid_n - nb_n) * nb_m + pid_m
        if j < n0:
            _mb_pack_job(s0p, d0p, a0, b0, k0, r0, j, c0)
        elif j < n0 + n1:
            _mb_pack_job(s1p, d1p, a1, b1, k1, r1, j - n0, c1)
    else:
        offs_m = pid_m * 128 + tl.arange(0, 128)
        offs_n = pid_n * 128 + tl.arange(0, 128)
        x = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * stride_x + offs_n[None, :]).to(tl.float32)
        ax = tl.abs(x)
        e = ue8m0_exp_g90(tl.max(ax, axis=1), MAX_EXP)  # [128] per row
        q = tl.clamp(x * pow2_neg(e)[:, None], -FP8_MAX, FP8_MAX)
        tl.store(qr_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], q.to(qr_ptr.dtype.element_ty))
        store_scale(er_ptr, offs_m, pid_n, e, M, nb_n, SLR)
        e = ue8m0_exp_g90(tl.max(ax, axis=0), MAX_EXP)  # [128] per column
        q = tl.clamp(x * pow2_neg(e)[None, :], -FP8_MAX, FP8_MAX).to(qt_ptr.dtype.element_ty)
        tl.store(qt_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :], tl.trans(q))
        store_scale(et_ptr, offs_n, pid_m, e, N, nb_m, SLT)


def mb_packq_enabled() -> bool:
    """PRIMUS_MB_PACKQ=1 (membound, default off): the forward activation quant also packs the weight scales."""
    return os.environ.get("PRIMUS_MB_PACKQ", "0") == "1"


def quant_dual_pack(x, dtype, row_layout, t_layout, wjobs, site=None, num_warps=2):
    """quant_dual_t_e8m0_v2(x, dtype, row_layout, t_layout) (with_row, with_t, TMODE 0, no scale_inv) plus
    dsv4up_pack.pack_scales(wjobs) (<= 2 jobs (src u8 2D, rows_per_scale, R, K), None entries allowed) in ONE launch.
    Returns ((q_row, None, e8m0_row, q_t, e8m0_t), packed list aligned with wjobs)."""
    assert x.dim() == 2 and x.stride(1) == 1 and len(wjobs) <= 2
    M, N = x.shape
    assert M % 128 == 0 and N % 128 == 0
    v = _MB_QDUAL.get(f"{site}{N}", _MB_QDUAL.get(str(N), _MB_QDUAL.get("*"))) if (_MB_QDUAL and site) else None
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
        x, qr, er, qt, et, M, N, x.stride(0), *args, FP8_MAX=fmax, MAX_EXP=fp8_max_exp(fmax), SLR=slr, SLT=slt,
        num_warps=num_warps)
    return (qr, None, er, qt, et), outs


def _mb_qdual_spec():
    """PRIMUS_MB_QDUAL (membound, default off): comma list of key:variant selecting the dual-quant kernel per call
    site and column count N. Keys, first match wins: <site><N> (site f = FP8 Linear forward input, c = concat-N
    Linear forward input, b = dY in backward), <N>, '*'. Variants: t2wW = _quant_dual_t2_kernel with W warps;
    t3BNxBMwW = _quant_dual_t3_kernel. Example: 'f1024:t2w8,f4096:t2w8'. Unlisted keep the caller's kernel."""
    spec = {}
    for item in os.environ.get("PRIMUS_MB_QDUAL", "").split(","):
        if ":" in item:
            k, v = item.split(":", 1)
            spec[k.strip()] = v.strip()
    return spec


_MB_QDUAL = _mb_qdual_spec()


def _scale_buf(R, K, sl, dev):
    if sl == SL_PACKED:
        return torch.empty((R // 128, K // 32 * 128), dtype=torch.uint8, device=dev)
    return torch.empty((R, K // 128), dtype=torch.uint8, device=dev)


def quant_dual_t_e8m0_v2(x: torch.Tensor, dtype=torch.float8_e4m3fn, with_row=True, with_t=True,
                         row_layout="g90", t_layout="g90", with_sr=False, tmode=0, num_warps=2, site=None):
    """Like quant_dual_t_e8m0 (same FP8 bytes and E8M0 codes) for M % 128 == N % 128 == 0, with per-output scale
    layouts: ``row_layout`` / ``t_layout`` in {"g90" (byte), "packed" (dsv4up)}. scale_inv is None unless
    ``with_sr``. Returns (q_row, scale_inv_row, e8m0_row, q_t, e8m0_t)."""
    assert x.dim() == 2 and x.stride(1) == 1
    M, N = x.shape
    assert M % 128 == 0 and N % 128 == 0
    slr, slt = scale_layout_id(row_layout), scale_layout_id(t_layout)
    assert slr in (SL_BYTE, SL_PACKED) and slt in (SL_BYTE, SL_PACKED)
    dev = x.device
    qr = torch.empty((M, N), dtype=dtype, device=dev) if with_row else None
    sr = torch.empty((M, N // 128), dtype=torch.float32, device=dev) if (with_row and with_sr) else None
    er = _scale_buf(M, N, slr, dev) if with_row else None
    qt = torch.empty((N, M), dtype=dtype, device=dev) if with_t else None
    et = _scale_buf(N, M, slt, dev) if with_t else None
    fmax = float(torch.finfo(dtype).max)
    d = x
    v = _MB_QDUAL.get(f"{site}{N}", _MB_QDUAL.get(str(N), _MB_QDUAL.get("*"))) if _MB_QDUAL else None
    if v is not None and v.startswith("t2w"):
        num_warps, tmode = int(v[3:]), 0
    elif v is not None and v.startswith("t3") and with_row and with_t and sr is None and x.stride(0) % 16 == 0:
        bn, rest = v[2:].split("x")
        bm, w = rest.split("w")
        bn, bm, w = int(bn), int(bm), int(w)
        if N % bn == 0 and M % bm == 0:
            _quant_dual_t3_kernel[((M // bm) * (N // bn),)](
                x, qr, er, qt, et, M, N, x.stride(0), FP8_MAX=fmax, MAX_EXP=fp8_max_exp(fmax), SLR=slr, SLT=slt,
                BN=bn, BM=bm, num_warps=w)
            return qr, sr, er, qt, et
    _quant_dual_t2_kernel[(M // 128, N // 128)](
        x, qr if with_row else d, sr if sr is not None else d, er if with_row else d,
        qt if with_t else d, et if with_t else d, M, N, x.stride(0),
        FP8_MAX=fmax, WITH_ROW=with_row, WITH_T=with_t, WITH_SR=sr is not None, MAX_EXP=fp8_max_exp(fmax),
        SLR=slr, SLT=slt, TMODE=tmode, num_warps=num_warps,
    )
    return qr, sr, er, qt, et


def e8m0_of(scale_inv: torch.Tensor) -> torch.Tensor:
    """fp32 power-of-two scale_inv -> E8M0 exponent bytes (same shape)."""
    return ((scale_inv.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
