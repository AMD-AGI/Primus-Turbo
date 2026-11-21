###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""gfx1250 tensorwise FP8 quantization at the HBM floor (MoE grouped-GEMM operands).

amax:  ``amax_partials`` writes one abs-max per program into a small fp32 buffer. No same-address
       atomics: on gfx1250 they serialize at ~65 ns each, which is what made the earlier
       ``_amax_kernel`` (1024 programs -> 1024 atomics) take 70-100 us on any tensor size.
       Consumers reduce the partials themselves (a few KB from L2), so no scale kernel/launch.
cast:  ``quant_flat`` row-major cast (weights [G, N, K] with N, K already aligned).
dual:  ``quant_dual`` row-major fp8 + per-expert ALIGN-padded transposed fp8 copy (wgrad operand).
A max is exact and order-independent: the scale and the fp8 bytes equal Turbo's tensorwise quantize.
"""

import torch
import triton
import triton.language as tl

ALIGN = 128
MIN_T = 2
AMAX_NP = 1024  # partial slots (power of two)


def pad_pitch(M: int, G: int) -> int:
    return (M + G * ALIGN * MIN_T + 15) // 16 * 16


@triton.jit
def _amax_partial_kernel(X, P, n, CHUNK: tl.constexpr, BLOCK: tl.constexpr, MASK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    m = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, CHUNK, BLOCK):
        offs = base + off + tl.arange(0, BLOCK)
        if MASK:
            x = tl.load(X + offs, mask=offs < n, other=0.0)
        else:
            x = tl.load(X + offs)
        m = tl.maximum(m, tl.abs(x.to(tl.float32)))
    tl.store(P + pid, tl.max(m, 0))


@triton.jit
def _scale_from_partials(P, NP: tl.constexpr, n_part, FP8_MAX: tl.constexpr):
    i = tl.arange(0, NP)
    amax = tl.max(tl.load(P + i, mask=i < n_part, other=0.0), 0)
    # same expression as Turbo's tensorwise_amax_scale_kernel (eps clamp) -> bit-identical scale
    return FP8_MAX / tl.maximum(amax, 1e-12)


def amax_partials(x: torch.Tensor, block: int = 8192, nw: int = 8, max_prog: int = AMAX_NP):
    """-> (partials fp32 [max_prog], n_part). Non-negative abs-max per program."""
    n = x.numel()
    nprog = max(1, min(max_prog, triton.cdiv(n, block)))
    chunk = triton.cdiv(triton.cdiv(n, nprog), block) * block
    nprog = triton.cdiv(n, chunk)
    p = torch.empty(max_prog, device=x.device, dtype=torch.float32)
    _amax_partial_kernel[(nprog,)](x, p, n, CHUNK=chunk, BLOCK=block, MASK=(nprog * chunk != n), num_warps=nw)
    return p, nprog


@triton.jit
def _quant_flat_kernel(X, Q, P, n_part, SINV, n, NP: tl.constexpr, FP8_MAX: tl.constexpr,
                       BLOCK: tl.constexpr, PACK: tl.constexpr):
    pid = tl.program_id(0)
    scale = _scale_from_partials(P, NP, n_part, FP8_MAX)
    if pid == 0:
        tl.store(SINV, 1.0 / scale)
    if PACK:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs, mask=offs < n, other=0.0)
        q = tl.clamp(x.to(tl.float32) * scale, -FP8_MAX, FP8_MAX).to(Q.dtype.element_ty)
        u = tl.reshape(q.to(tl.uint8, bitcast=True).to(tl.int32), (BLOCK // 4, 4))
        w = tl.sum(u << (tl.arange(0, 4) * 8)[None, :], axis=1)
        wo = pid.to(tl.int64) * (BLOCK // 4) + tl.arange(0, BLOCK // 4)
        tl.store(Q.to(tl.pointer_type(tl.int32)) + wo, w, mask=wo < n // 4)
    else:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs, mask=offs < n, other=0.0)
        q = tl.clamp(x.to(tl.float32) * scale, -FP8_MAX, FP8_MAX).to(Q.dtype.element_ty)
        tl.store(Q + offs, q, mask=offs < n)


def quant_flat(x: torch.Tensor, dtype: torch.dtype, parts=None, block: int = 8192, nw: int = 8, pack: int = 1):
    """Tensorwise cast of a contiguous tensor (any shape, numel % 4 == 0) -> (q same shape, scale_inv [1])."""
    assert x.is_contiguous() and x.numel() % 4 == 0
    if parts is None:
        parts = amax_partials(x)
    p, n_part = parts
    q = torch.empty(x.shape, device=x.device, dtype=dtype)
    sinv = torch.empty(1, device=x.device, dtype=torch.float32)
    n = x.numel()
    _quant_flat_kernel[(triton.cdiv(n, block),)](
        x, q, p, n_part, sinv, n, NP=p.numel(), FP8_MAX=torch.finfo(dtype).max, BLOCK=block, PACK=pack, num_warps=nw
    )
    return q, sinv


@triton.jit
def _quant_dual_kernel(
    X, Q, QT, P, n_part, SINV, offs_ptr, POFFS, C, stride_x, stride_q, stride_qt, G, n_t,
    NP: tl.constexpr, FP8_MAX: tl.constexpr, GP: tl.constexpr, ALIGN: tl.constexpr, MIN_T: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, CB: tl.constexpr, EMAJ: tl.constexpr = 0,
):
    """grid (n_t, cdiv(C, BC*CB)); token-block axis fastest so neighbouring programs share rows."""
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    scale = _scale_from_partials(P, NP, n_part, FP8_MAX)
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
    d0 = pid_t * BT
    if d0 >= ptotal:
        return
    pstart = pend - plen
    pend_v = tl.where(idx < G, pend, 2147483647)
    gid = tl.sum((pend_v <= d0).to(tl.int32))
    sel = idx == gid
    shift = tl.sum(tl.where(sel, lo - pstart, 0))
    ps_g = tl.sum(tl.where(sel, pstart, 0))
    pl_g = tl.sum(tl.where(sel, plen, 0))
    g_hi = tl.sum(tl.where(sel, hi, 0))
    d = d0 + tl.arange(0, BT)
    src = d + shift
    valid = src < g_hi
    QR = Q.to(tl.pointer_type(tl.int32))
    QW = QT.to(tl.pointer_type(tl.int32))
    for cb in tl.static_range(CB):
        c0 = (pid_c * CB + cb) * BC
        rc = c0 + tl.arange(0, BC)
        x = tl.load(X + src[:, None].to(tl.int64) * stride_x + rc[None, :], mask=valid[:, None], other=0.0,
                    )
        q = tl.clamp(x.to(tl.float32) * scale, -FP8_MAX, FP8_MAX).to(QT.dtype.element_ty)
        u8 = q.to(tl.uint8, bitcast=True).to(tl.int32)
        ur = tl.reshape(u8, (BT, BC // 4, 4))
        wr = tl.sum(ur << (tl.arange(0, 4) * 8)[None, None, :], axis=2)
        cw = c0 // 4 + tl.arange(0, BC // 4)
        tl.store(QR + src[:, None].to(tl.int64) * (stride_q // 4) + cw[None, :], wr, mask=valid[:, None])
        ut = tl.reshape(u8, (BT // 4, 4, BC))
        wt = tl.trans(tl.sum(ut << (tl.arange(0, 4) * 8)[None, :, None], axis=1))
        dw = d0 // 4 + tl.arange(0, BT // 4)
        if EMAJ:
            # expert-major: expert g's block [C, plen_g] starts at pstart_g * C (contiguous per expert)
            eb = (ps_g.to(tl.int64) * C + (d0 - ps_g)) // 4
            tl.store(QW + eb + rc[:, None].to(tl.int64) * (pl_g // 4) + tl.arange(0, BT // 4)[None, :], wt)
        else:
            tl.store(QW + rc[:, None].to(tl.int64) * (stride_qt // 4) + dw[None, :], wt)


def quant_dual(x: torch.Tensor, dtype: torch.dtype, group_offs: torch.Tensor, parts=None,
               bt: int = ALIGN, bc: int = 64, cb: int = 4, nw: int = 4, emaj: bool = False):
    """bf16 x [M, C] -> (q [M, C], scale_inv [1], qT [C, P] padded token-contiguous, poffs int32 [G+1]).
    Same outputs as grouped_gemm_fp8_vk_gfx1250.quantize_fp8_tensorwise_dual. C % (bc*cb) == 0."""
    assert x.dim() == 2 and x.is_contiguous()
    M, C = x.shape
    assert C % (bc * cb) == 0 and bt == ALIGN
    G = group_offs.numel() - 1
    P = pad_pitch(M, G)
    if parts is None:
        parts = amax_partials(x)
    p, n_part = parts
    q = torch.empty((M, C), device=x.device, dtype=dtype)
    # emaj: flat [P * C] buffer, expert g's [C, plen_g] block at poffs[g] * C (rows of one expert contiguous)
    qT = torch.empty((C * P,) if emaj else (C, P), device=x.device, dtype=dtype)
    poffs = torch.empty(G + 1, device=x.device, dtype=torch.int32)
    sinv = torch.empty(1, device=x.device, dtype=torch.float32)
    n_t = min(triton.cdiv(P, bt), M // bt + G * MIN_T + 1)
    _quant_dual_kernel[(n_t, C // (bc * cb))](
        x, q, qT, p, n_part, sinv, group_offs, poffs, C, x.stride(0), q.stride(0), P if emaj else qT.stride(0), G, n_t,
        NP=p.numel(), FP8_MAX=torch.finfo(dtype).max, GP=triton.next_power_of_2(G + 1), ALIGN=ALIGN, MIN_T=MIN_T,
        BT=bt, BC=bc, CB=cb, EMAJ=int(emaj), num_warps=nw,
    )
    return q, sinv, qT, poffs
