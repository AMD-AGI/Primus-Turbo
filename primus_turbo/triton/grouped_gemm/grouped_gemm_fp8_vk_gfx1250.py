###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""gfx1250 FP8 tensorwise variable-K (wgrad) grouped GEMM on group-padded, token-contiguous operands.

``out[g] (+)= lhs[rows_g]^T @ rhs[rows_g] * lhs_scale * rhs_scale``  (same contract as
``grouped_gemm_fp8_tensorwise_variable_k_triton_kernel``).

Turbo's generic variable-K kernel reads ``lhs``/``rhs`` token-major, i.e. with the contraction axis
strided. gfx1250's fp8 WMMA (16x16x128) wants both operands K-contiguous, so that kernel pays an
in-register/LDS transpose of every tile and reaches ~0.6 PFLOP/s. Here both fp8 operands are first
transposed once into ``[C, P]`` buffers in which every expert's token range starts on an
``ALIGN``-aligned column and is zero-padded to a multiple of ``ALIGN`` (``t_pad``; one HBM pass per
operand, no host sync: the padded offsets are computed on device). The GEMM then runs with plain,
unmasked 16-byte loads along the contraction axis and accumulates into ``out`` in its epilogue.
"""

import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.core.utils import get_num_cus

ALIGN = 128
# Each expert segment spans at least MIN_T aligned tiles, so a pipelined consumer may prefetch one
# tile ahead without leaving the segment; empty experts then contribute exact zeros.
MIN_T = 2


@triton.jit
def _t_pad_kernel(
    X,
    Y,
    offs_ptr,
    POFFS,
    M,
    C,
    stride_x,
    stride_y,
    G,
    GP: tl.constexpr,
    ALIGN: tl.constexpr,
    MIN_T: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    """Y[c, d] = X[src(d), c] for every padded column d < total; pad columns are written as zeros."""
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    d = pid_t * BT + tl.arange(0, BT)
    rc = pid_c * BC + tl.arange(0, BC)
    idx = tl.arange(0, GP)
    lo = tl.load(offs_ptr + idx, mask=idx < G, other=0).to(tl.int32)
    hi = tl.load(offs_ptr + idx + 1, mask=idx < G, other=0).to(tl.int32)
    plen = tl.where(idx < G, tl.maximum((hi - lo + ALIGN - 1) // ALIGN, MIN_T) * ALIGN, 0)
    pend = tl.cumsum(plen, 0)
    if (pid_t == 0) & (pid_c == 0):
        tl.store(POFFS + idx + 1, pend.to(tl.int32), mask=idx < G)
        tl.store(POFFS + idx, tl.zeros_like(pend).to(tl.int32), mask=idx == 0)
    ptotal = tl.sum(plen)
    pstart = pend - plen
    pend_v = tl.where(idx < G, pend, 2147483647)
    gid = tl.sum((pend_v[None, :] <= d[:, None]).to(tl.int32), 1)
    sel = idx[None, :] == gid[:, None]
    shift = tl.sum(tl.where(sel, lo[None, :] - pstart[None, :], 0), 1)
    g_hi = tl.sum(tl.where(sel, hi[None, :], 0), 1)
    src = d + shift
    valid = (d < ptotal) & (src < g_hi)
    cm = rc < C
    x = tl.load(
        X + src[:, None].to(tl.int64) * stride_x + rc[None, :],
        mask=valid[:, None] & cm[None, :],
        other=0.0,
    )
    tl.store(Y + rc[None, :].to(tl.int64) * stride_y + d[:, None], x, mask=(d < ptotal)[:, None] & cm[None, :])


def pad_pitch(M: int, G: int) -> int:
    return (M + G * ALIGN * MIN_T + 15) // 16 * 16


def t_pad(x: torch.Tensor, group_offs: torch.Tensor, n_cols: int | None = None):
    """``x`` [M, C] (only the first ``n_cols`` columns) -> (``y`` [n_cols, P], padded offsets int32 [G+1])."""
    M = x.shape[0]
    C = x.shape[1] if n_cols is None else n_cols
    G = group_offs.numel() - 1
    P = pad_pitch(M, G)
    y = torch.empty((C, P), device=x.device, dtype=x.dtype)
    poffs = torch.empty(G + 1, device=x.device, dtype=torch.int32)
    BT, BC = 64, 64
    _t_pad_kernel[(triton.cdiv(P, BT), triton.cdiv(C, BC))](
        x,
        y,
        group_offs,
        poffs,
        M,
        C,
        x.stride(0),
        y.stride(0),
        G,
        GP=triton.next_power_of_2(G + 1),
        ALIGN=ALIGN,
        MIN_T=MIN_T,
        BT=BT,
        BC=BC,
        num_warps=4,
    )
    return y, poffs


@triton.jit
def _vk_fp8_padded_kernel(
    LHS,
    RHS,
    C,
    ls_ptr,
    rs_ptr,
    poffs_ptr,  # [G + 1] int32 padded offsets
    offs_ptr,  # [G + 1] int64 real offsets
    G,
    OUT_M,
    OUT_N,
    stride_l,
    stride_r,
    stride_cg,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BETA_IS_ONE: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(OUT_M, BLOCK_M)
    tiles_n = tl.cdiv(OUT_N, BLOCK_N)
    tpg = tiles_m * tiles_n
    total = G * tpg
    scale = tl.load(ls_ptr) * tl.load(rs_ptr)
    tl.assume(stride_l > 0)
    tl.assume(stride_r > 0)
    tl.assume(stride_cg > 0)
    tl.assume(stride_cm > 0)
    for t in range(pid, total, NUM_SMS):
        g = t // tpg
        lt = t - g * tpg
        npg = GROUP_M * tiles_n
        first_m = (lt // npg) * GROUP_M
        gsm = min(tiles_m - first_m, GROUP_M)
        pid_m = first_m + (lt % npg) % gsm
        pid_n = (lt % npg) // gsm
        k0 = tl.load(poffs_ptr + g)
        k0 = tl.multiple_of(k0, 128)
        n_blk = (tl.load(poffs_ptr + g + 1) - k0) // BLOCK_K
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % OUT_M
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % OUT_N
        rk = tl.arange(0, BLOCK_K)
        A = LHS + rm[:, None].to(tl.int64) * stride_l + (k0 + rk)[None, :]
        B = RHS + rn[:, None].to(tl.int64) * stride_r + (k0 + rk)[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(n_blk):
            a = tl.load(tl.multiple_of(A, (1, 16)))
            b = tl.load(tl.multiple_of(B, (1, 16)))
            acc += tl.dot(a, tl.trans(b))
            A += BLOCK_K
            B += BLOCK_K
        acc = acc * scale
        rms = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rns = tl.max_contiguous(tl.multiple_of(rns, BLOCK_N), BLOCK_N)
        cmask = (rms[:, None] < OUT_M) & (rns[None, :] < OUT_N)
        Cp = C + g.to(tl.int64) * stride_cg + rms[:, None].to(tl.int64) * stride_cm + rns[None, :]
        if BETA_IS_ONE:
            acc += tl.load(Cp, mask=cmask, other=0.0).to(tl.float32)
        tl.store(Cp, acc.to(C.dtype.element_ty), mask=cmask)


# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)
_TRITON_CFG = (256, 256, 128, 4, 8, 2)


def grouped_gemm_fp8_tensorwise_variable_k_gfx1250(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    beta: float = 0.0,
    out: torch.Tensor | None = None,
    m_real: "int | None" = None,
    n_real: "int | None" = None,
) -> torch.Tensor:
    """lhs [M_total, OUT_M] fp8, rhs [M_total, OUT_N] fp8 -> out [G, OUT_M, OUT_N] (+= when beta == 1)."""
    assert beta in (0.0, 1.0)
    OUT_M = m_real if m_real is not None else lhs.shape[1]
    OUT_N = n_real if n_real is not None else rhs.shape[1]
    G = group_offs.numel() - 1
    if out is None:
        assert beta == 0.0, "beta=1.0 requires an explicit `out` buffer"
        out = torch.empty((G, OUT_M, OUT_N), device=lhs.device, dtype=out_dtype)
    assert tuple(out.shape) == (G, OUT_M, OUT_N) and out.is_contiguous()
    lhs_p, poffs = t_pad(lhs, group_offs, OUT_M)
    rhs_p, _ = t_pad(rhs, group_offs, OUT_N)
    bm, bn, bk, gm, nw, ns = _TRITON_CFG
    total = G * triton.cdiv(OUT_M, bm) * triton.cdiv(OUT_N, bn)
    nsm = max(1, min(total, get_num_cus()))
    _vk_fp8_padded_kernel[(nsm,)](
        lhs_p,
        rhs_p,
        out,
        lhs_scale,
        rhs_scale,
        poffs,
        group_offs,
        G,
        OUT_M,
        OUT_N,
        lhs_p.stride(0),
        rhs_p.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=gm,
        NUM_SMS=nsm,
        BETA_IS_ONE=(beta == 1.0),
        num_warps=nw,
        num_stages=ns,
    )
    return out


# ---------------------------------------------------------------------------------------------
# Fused tensorwise quantization that emits both layouts in one pass over the bf16 input:
#   q  [M, C]  row-major fp8 (the forward / dgrad operand)
#   qT [C, P]  token-contiguous fp8, each expert ALIGN-aligned and zero-padded (the wgrad operand)
# The abs-max needs a pass of its own (tensorwise scale), as in Turbo's quantize op.
# ---------------------------------------------------------------------------------------------


@triton.jit
def _amax_kernel(X, AMAX, n, NPROG: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    m = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(pid * BLOCK, n, NPROG * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(X + offs, mask=offs < n, other=0.0).to(tl.float32)
        m = tl.maximum(m, tl.abs(x))
    tl.atomic_max(AMAX, tl.max(m, 0))


@triton.jit
def _quant_dual_kernel(
    X,
    Q,
    QT,
    AMAX,
    SINV,
    offs_ptr,
    POFFS,
    C,
    stride_x,
    stride_q,
    stride_qt,
    G,
    FP8_MAX: tl.constexpr,
    GP: tl.constexpr,
    ALIGN: tl.constexpr,
    MIN_T: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    PACK: tl.constexpr = 1,
    STORE_Q: tl.constexpr = 1,
    STORE_QT: tl.constexpr = 1,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    amax = tl.load(AMAX)
    scale = tl.where(amax > 0, FP8_MAX / amax, 1.0)
    d = pid_t * BT + tl.arange(0, BT)
    rc = pid_c * BC + tl.arange(0, BC)
    idx = tl.arange(0, GP)
    lo = tl.load(offs_ptr + idx, mask=idx < G, other=0).to(tl.int32)
    hi = tl.load(offs_ptr + idx + 1, mask=idx < G, other=0).to(tl.int32)
    plen = tl.where(idx < G, tl.maximum((hi - lo + ALIGN - 1) // ALIGN, MIN_T) * ALIGN, 0)
    pend = tl.cumsum(plen, 0)
    if (pid_t == 0) & (pid_c == 0):
        tl.store(POFFS + idx + 1, pend.to(tl.int32), mask=idx < G)
        tl.store(POFFS + idx, tl.zeros_like(pend).to(tl.int32), mask=idx == 0)
        tl.store(SINV, 1.0 / scale)
    ptotal = tl.sum(plen)
    pstart = pend - plen
    # BT == ALIGN and segments are ALIGN-aligned: the whole block belongs to one expert (scalar lookup).
    d0 = pid_t * BT
    pend_v = tl.where(idx < G, pend, 2147483647)
    gid = tl.sum((pend_v <= d0).to(tl.int32))
    sel = idx == gid
    shift = tl.sum(tl.where(sel, lo - pstart, 0))
    g_hi = tl.sum(tl.where(sel, hi, 0))
    src = d + shift
    valid = (d < ptotal) & (src < g_hi)
    cm = rc < C
    x = tl.load(X + src[:, None].to(tl.int64) * stride_x + rc[None, :], mask=valid[:, None] & cm[None, :], other=0.0)
    q = tl.clamp(x.to(tl.float32) * scale, -FP8_MAX, FP8_MAX).to(QT.dtype.element_ty)
    if d0 >= ptotal:
        return
    if STORE_Q:
        if PACK:
            # 4 consecutive channels per int32 word -> 4-byte row-major stores (fp8 stores are byte-wide otherwise)
            ur = tl.reshape(q.to(tl.uint8, bitcast=True).to(tl.int32), (BT, BC // 4, 4))
            wr = tl.sum(ur << (tl.arange(0, 4) * 8)[None, None, :], axis=2)  # [BT, BC//4]
            cw = pid_c * (BC // 4) + tl.arange(0, BC // 4)
            QR = Q.to(tl.pointer_type(tl.int32))
            tl.store(QR + src[:, None].to(tl.int64) * (stride_q // 4) + cw[None, :], wr,
                     mask=valid[:, None] & (cw < C // 4)[None, :])
        else:
            tl.store(Q + src[:, None].to(tl.int64) * stride_q + rc[None, :], q, mask=valid[:, None] & cm[None, :])
    if STORE_QT:
        if PACK:
            # pack 4 consecutive tokens of a channel into one int32 word -> 4-byte (vectorizable) transposed stores
            u = q.to(tl.uint8, bitcast=True).to(tl.int32)  # [BT, BC]
            u = tl.reshape(u, (BT // 4, 4, BC))
            sh = (tl.arange(0, 4) * 8)[None, :, None]
            w = tl.sum(u << sh, axis=1)  # [BT//4, BC]
            wt = tl.trans(w)  # [BC, BT//4]
            dw = d0 // 4 + tl.arange(0, BT // 4)
            QW = QT.to(tl.pointer_type(tl.int32))
            tl.store(QW + rc[:, None].to(tl.int64) * (stride_qt // 4) + dw[None, :], wt, mask=cm[:, None])
        else:
            qt = tl.trans(q)
            tl.store(QT + rc[:, None].to(tl.int64) * stride_qt + d[None, :], qt, mask=cm[:, None] & (d < ptotal)[None, :])


def quantize_fp8_tensorwise_dual(x: torch.Tensor, dtype: torch.dtype, group_offs: torch.Tensor, **variant):
    """bf16 x [M, C] -> (q [M, C] fp8, scale_inv [1] fp32, qT [C, P] padded token-contiguous fp8, poffs int32 [G+1])."""
    assert x.dim() == 2 and x.is_contiguous()
    M, C = x.shape
    G = group_offs.numel() - 1
    P = pad_pitch(M, G)
    q = torch.empty((M, C), device=x.device, dtype=dtype)
    qT = torch.empty((C, P), device=x.device, dtype=dtype)
    poffs = torch.empty(G + 1, device=x.device, dtype=torch.int32)
    sinv = torch.empty(1, device=x.device, dtype=torch.float32)
    amax = torch.zeros(1, device=x.device, dtype=torch.float32)
    n = x.numel()
    nprog = max(1, min(1024, triton.cdiv(n, 8192)))
    _amax_kernel[(nprog,)](x, amax, n, NPROG=nprog, BLOCK=8192, num_warps=8)
    BT, BC = ALIGN, 128
    # real padded tiles <= M // BT + G * MIN_T + 1 (programs past the padded total exit at once)
    n_t = min(triton.cdiv(P, BT), M // BT + G * MIN_T + 1)
    _quant_dual_kernel[(n_t, triton.cdiv(C, BC))](
        x,
        q,
        qT,
        amax,
        sinv,
        group_offs,
        poffs,
        C,
        x.stride(0),
        q.stride(0),
        qT.stride(0),
        G,
        FP8_MAX=torch.finfo(dtype).max,
        GP=triton.next_power_of_2(G + 1),
        ALIGN=ALIGN,
        MIN_T=MIN_T,
        BT=BT,
        BC=BC,
        num_warps=8,
        **variant,
    )
    return q, sinv, qT, poffs


def variable_k_padded(lhs_p, rhs_p, lhs_scale, rhs_scale, poffs, group_offs, OUT_M, OUT_N, out, beta):
    """wgrad on prebuilt padded operands. FlyDSL WMMA kernel for a bf16 ``out``; Triton otherwise."""
    G = poffs.numel() - 1
    if out.dtype == torch.bfloat16 and OUT_N % 256 == 0:
        from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_vk_flydsl_gfx1250 import flyvk_fp8

        return flyvk_fp8(lhs_p, rhs_p, lhs_scale, rhs_scale, poffs, OUT_M, OUT_N, out, beta=beta,
                         tile=(256, 256, 128), m_warp=2, n_warp=4, num_buffers=2)
    bm, bn, bk, gm, nw, ns = _TRITON_CFG
    total = G * triton.cdiv(OUT_M, bm) * triton.cdiv(OUT_N, bn)
    nsm = max(1, min(total, get_num_cus()))
    _vk_fp8_padded_kernel[(nsm,)](
        lhs_p, rhs_p, out, lhs_scale, rhs_scale, poffs, group_offs, G, OUT_M, OUT_N,
        lhs_p.stride(0), rhs_p.stride(0), out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, NUM_SMS=nsm, BETA_IS_ONE=(beta == 1.0),
        num_warps=nw, num_stages=ns,
    )
    return out
