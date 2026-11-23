###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Concat-N block-FP8 linear: Y_i = X . W_i^T for weights sharing X, one launch per pass (gfx1250 dsv4up path).

For weights that read the same input (DeepSeek-V4 wq_a || wkv), with the DSv4 recipe (1x128 activation /
128x128 weight blocks, UE8M0 scales, ``PRIMUS_TURBO_FP8_BLOCK_KERNEL=dsv4up``):
  fwd    one 1x128 quant of X, one launch writing one CONTIGUOUS output per weight (split-output epilogue
         for two weights; bits = the separate GEMMs')
  bwd    one 1x128 quant of [dY_0 | dY_1] read from the separate grads (no bf16 concat copy); with two
         outputs the op that consumes both may run that quant in its own backward launch
         (DyPrequantSlot: take_dy_prequant / dy_prequant_fill)
  dgrad  one GEMM with K = sum N_i                       (fp32 over all K, one rounding: NOT the bits of
                                                           separate dgrads + a bf16 add)
  wgrad  one GEMM into the joint main_grad if the weights' main_grads are adjacent in that order, else one GEMM
         per weight on its row slice of dY^T             (bits = the separate GEMMs')
Weights: with the FP8 weight cache (PRIMUS_TURBO_FP8_WCACHE) the optimizer writes each weight's FP8 W / scales into its row slice of
one joint buffer (fp8blk_weight_cache.attach_joint), so the forward reads the joint W with no copy; the dgrad
operand W^T is one concat of the per-weight W^T copies per step. Without a ready cache each weight is quantized
here (128-row blocks never straddle two weights, so this equals the per-weight quant).
"""

from typing import List, Optional, Sequence

import torch
import triton
import triton.language as tl

import primus_turbo.common.envs as envs
from primus_turbo.pytorch.core import fp8blk_weight_cache
from primus_turbo.pytorch.core import overlap as _overlap
from primus_turbo.pytorch.ops.utils import (
    _get_dummy_wgrad,
    _setup_fused_grad_accum,
    main_grad_first_write,
)
from primus_turbo.triton.quantization.quantization_fp8blk_utils import (
    SL_PACKED,
    fp8_max_exp,
    pow2_neg,
    store_scale,
    ue8m0_exp_g90,
)

__all__ = [
    "gemm_fp8_block_concat_n",
    "gemm_fp8_block_concat_n_supported",
    "request_dx_accum",
    "take_dy_prequant",
    "dy_prequant_fill",
]

# Optional hand-off from the framework: an object with ``take(M, K, dtype) -> Tensor | None`` and a ``fused`` attribute,
# handed to the next concat-N forward. Its backward then writes dX into the returned tensor as
# bf16(dX) + tensor (dsv4up accumulate=2: torch's bf16 add) and records it in ``fused``.
_DX_ACCUM = [None]


def request_dx_accum(slot) -> None:
    _DX_ACCUM[0] = slot


def _dgrad(g_row, b_t, pg, pbt, out_dtype, slot):
    from primus_turbo.pytorch.kernels.gemm import gemm_fp8_gluon_gfx1250_impl

    if slot is not None:
        M, K = g_row.shape[0], b_t.shape[0]
        acc = slot.take(M, K, out_dtype)
        if (
            acc is not None
            and not gemm_fp8_gluon_gfx1250_impl.small_tile_of(M, K, g_row.shape[1], True)
            and not gemm_fp8_gluon_gfx1250_impl.small_tile_of(M, K, g_row.shape[1], False)
        ):
            from primus_turbo.gluon.gemm.gemm_fp8blk_kernel import dsv4up_gemm

            dsv4up_gemm(g_row, b_t, pg, pbt, out=acc, accumulate=2)
            slot.fused = acc
            return acc
        if acc is not None:  # cannot fuse: give the buffer back untouched, autograd adds as usual
            slot.untake(acc)
    return gemm_fp8_gluon_gfx1250_impl.gemm_nt(g_row, b_t, pg, pbt, out_dtype)


def gemm_fp8_block_concat_n_supported(x: torch.Tensor, weights: Sequence[torch.Tensor], config) -> bool:
    """True if the concat-N op can run all three passes on dsv4up for these shapes / this config."""
    try:
        from primus_turbo.pytorch.core.low_precision import ScalingGranularity
        from primus_turbo.pytorch.kernels.gemm import gemm_fp8_gluon_gfx1250_impl
    except Exception:
        return False
    if not envs.PRIMUS_TURBO_FP8_BLOCK_UE8M0 or not gemm_fp8_gluon_gfx1250_impl.enabled():
        return False
    if (
        config is None
        or config.granularity != ScalingGranularity.BLOCKWISE
        or getattr(config, "block_size", 128) != 128
    ):
        return False
    if x.dim() != 2 or not x.is_cuda or len(weights) < 2:
        return False
    M, K = x.shape
    if any(w.dim() != 2 or w.shape[1] != K or w.shape[0] % 128 for w in weights):
        return False
    N = sum(int(w.shape[0]) for w in weights)
    if M % 512 or K % 512:
        return False
    sh = gemm_fp8_gluon_gfx1250_impl.shape_supported
    return sh(M, N, K) and sh(M, K, N) and all(sh(int(w.shape[0]), K, M) for w in weights)


def _weights_fp8(weights, b_dtype):
    """-> (w_joint [N,K] fp8, e8m0_joint [N/128,K/128] u8, wt_list [K,N_i] or None). Joint cache when ready."""
    jc = fp8blk_weight_cache.attach_joint(list(weights), b_dtype)
    if jc is not None and jc.ready(weights, b_dtype):
        return jc.w, jc.e8m0, [c.wt for c in jc.caches]
    caches = [fp8blk_weight_cache.lookup(w, b_dtype) for w in weights]
    if all(c is not None for c in caches):  # separate (non-joint) caches: one concat per step
        return (
            torch.cat([c.w for c in caches], 0),
            torch.cat([c.e8m0 for c in caches], 0),
            [c.wt for c in caches],
        )
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity, ScalingRecipe
    from primus_turbo.pytorch.ops.quantization import quantize_fp8
    from primus_turbo.triton.quantization.quantization_fp8blk_dual_kernel import e8m0_of

    qs, es = [], []
    for w in weights:  # same quant as FP8GemmBlockFunction's no-cache path, per weight
        fp8blk_weight_cache.materialize(w)  # BF16 reader (no-op unless the optimizer skipped its store)
        q, s = quantize_fp8(
            w,
            b_dtype,
            ScalingGranularity.BLOCKWISE,
            block_size=128,
            scaling_recipe=ScalingRecipe(use_2d_block=True),
        )
        qs.append(q)
        es.append(e8m0_of(s))
    return torch.cat(qs, 0), torch.cat(es, 0), None


@triton.jit
def _cat_cols2_kernel(s0, s1, d, R, C0: tl.constexpr, C1: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    # d[R, C0 + C1] = [s0 | s1] (int32 words; all row-major contiguous)
    rows = tl.program_id(0) * BR + tl.arange(0, BR)[:, None]
    cols = tl.program_id(1) * BC + tl.arange(0, BC)[None, :]
    m = (rows < R) & (cols < C0 + C1)
    in0 = cols < C0
    v0 = tl.load(s0 + rows * C0 + cols, mask=m & in0, other=0)
    v1 = tl.load(s1 + rows * C1 + (cols - C0), mask=m & (cols >= C0), other=0)
    tl.store(d + rows * (C0 + C1) + cols, tl.where(in0, v0, v1), mask=m)


def _cat_cols(ts):
    """torch.cat(ts, 1) of row-major fp8 [R, C_i] (C_i % 4 == 0) in one Triton launch for 2 parts."""
    if len(ts) != 2 or any(not t.is_contiguous() or t.shape[1] % 4 for t in ts):
        return torch.cat([t.view(torch.int32) for t in ts], 1).view(ts[0].dtype)
    R = ts[0].shape[0]
    c0, c1 = ts[0].shape[1] // 4, ts[1].shape[1] // 4
    out = torch.empty((R, (c0 + c1) * 4), dtype=ts[0].dtype, device=ts[0].device)
    for t in (*ts, out):
        assert t.is_cuda and t.data_ptr() != 0 and t.shape[0] == R
    BR, BC = 32, 128
    _cat_cols2_kernel[(triton.cdiv(R, BR), triton.cdiv(c0 + c1, BC))](
        ts[0].view(torch.int32),
        ts[1].view(torch.int32),
        out.view(torch.int32),
        R,
        C0=c0,
        C1=c1,
        BR=BR,
        BC=BC,
        num_warps=4,
    )
    return out


@triton.jit
def _quant_cat2_kernel(
    x0_ptr,
    x1_ptr,
    qr_ptr,
    er_ptr,
    qt_ptr,
    et_ptr,
    M,
    N,
    N0,
    stride_x0,
    stride_x1,
    FP8_MAX: tl.constexpr,
    MAX_EXP: tl.constexpr,
    SL: tl.constexpr,
):
    """quant_dual_t_e8m0_v2 (TMODE 0, packed row / packed t scales) of the virtual [M, N] = [x0 | x1] (x0 [M, N0]):
    each 128x128 tile reads its own source, so no concatenated bf16 copy; the bytes equal the quant of the cat."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    if pid_n * 128 < N0:
        x = tl.load(x0_ptr + offs_m[:, None].to(tl.int64) * stride_x0 + offs_n[None, :]).to(tl.float32)
    else:
        x = tl.load(x1_ptr + offs_m[:, None].to(tl.int64) * stride_x1 + (offs_n[None, :] - N0)).to(tl.float32)
    ax = tl.abs(x)
    nb_n = N // 128
    nb_m = M // 128
    e = ue8m0_exp_g90(tl.max(ax, axis=1), MAX_EXP)  # [128] per row
    q = tl.clamp(x * pow2_neg(e)[:, None], -FP8_MAX, FP8_MAX)
    tl.store(qr_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], q.to(qr_ptr.dtype.element_ty))
    store_scale(er_ptr, offs_m, pid_n, e, M, nb_n, SL)
    e = ue8m0_exp_g90(tl.max(ax, axis=0), MAX_EXP)  # [128] per column
    q = tl.clamp(x * pow2_neg(e)[None, :], -FP8_MAX, FP8_MAX).to(qt_ptr.dtype.element_ty)
    tl.store(qt_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :], tl.trans(q))
    store_scale(et_ptr, offs_n, pid_m, e, N, nb_m, SL)


def _quant_cat(parts, dtype):
    """(q_row [M,N], packed row scales [M/128, N*4], q_t [N,M], packed t scales [N/128, M*4]) of cat(parts, 1)."""
    from primus_turbo.triton.quantization.quantization_fp8blk_dual_kernel import quant_dual_t_e8m0_v2

    if len(parts) != 2:
        g = torch.cat(parts, 1)
        qr, _, er, qt, et = quant_dual_t_e8m0_v2(g, dtype, row_layout="packed", t_layout="packed")
        return qr, er, qt, et
    x0, x1 = parts
    M, N0 = x0.shape
    N = N0 + x1.shape[1]
    assert M % 128 == 0 and N0 % 128 == 0 and N % 128 == 0 and x0.stride(1) == 1 and x1.stride(1) == 1
    dev = x0.device
    qr = torch.empty((M, N), dtype=dtype, device=dev)
    er = torch.empty((M // 128, N // 32 * 128), dtype=torch.uint8, device=dev)
    qt = torch.empty((N, M), dtype=dtype, device=dev)
    et = torch.empty((N // 128, M // 32 * 128), dtype=torch.uint8, device=dev)
    for t in (x0, x1, qr, er, qt, et):
        assert t.is_cuda and t.data_ptr() != 0
    fmax = float(torch.finfo(dtype).max)
    _quant_cat2_kernel[(M // 128, N // 128)](
        x0,
        x1,
        qr,
        er,
        qt,
        et,
        M,
        N,
        N0,
        x0.stride(0),
        x1.stride(0),
        FP8_MAX=fmax,
        MAX_EXP=fp8_max_exp(fmax),
        SL=SL_PACKED,
        num_warps=2,
    )
    return qr, er, qt, et


@triton.jit
def _quant_cat2_cols_kernel(
    x0_ptr,
    x1_ptr,
    qr_ptr,
    er_ptr,
    qt_ptr,
    et_ptr,
    M,
    N,
    N0,
    stride_x0,
    stride_x1,
    s0,
    s1,
    d,
    R,
    NQ,
    NBN,
    NCB,
    FP8_MAX: tl.constexpr,
    MAX_EXP: tl.constexpr,
    SL: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    """``_quant_cat2_kernel`` (programs [0, NQ)) and ``_cat_cols2_kernel``
    (programs [NQ, ...)) in one launch; each program runs the original body (same bytes)."""
    pid = tl.program_id(0)
    if pid < NQ:
        pid_m = pid // NBN
        pid_n = pid % NBN
        offs_m = pid_m * 128 + tl.arange(0, 128)
        offs_n = pid_n * 128 + tl.arange(0, 128)
        if pid_n * 128 < N0:
            x = tl.load(x0_ptr + offs_m[:, None].to(tl.int64) * stride_x0 + offs_n[None, :]).to(tl.float32)
        else:
            x = tl.load(x1_ptr + offs_m[:, None].to(tl.int64) * stride_x1 + (offs_n[None, :] - N0)).to(
                tl.float32
            )
        ax = tl.abs(x)
        nb_n = N // 128
        nb_m = M // 128
        e = ue8m0_exp_g90(tl.max(ax, axis=1), MAX_EXP)  # [128] per row
        q = tl.clamp(x * pow2_neg(e)[:, None], -FP8_MAX, FP8_MAX)
        tl.store(qr_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], q.to(qr_ptr.dtype.element_ty))
        store_scale(er_ptr, offs_m, pid_n, e, M, nb_n, SL)
        e = ue8m0_exp_g90(tl.max(ax, axis=0), MAX_EXP)  # [128] per column
        q = tl.clamp(x * pow2_neg(e)[None, :], -FP8_MAX, FP8_MAX).to(qt_ptr.dtype.element_ty)
        tl.store(qt_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :], tl.trans(q))
        store_scale(et_ptr, offs_n, pid_m, e, N, nb_m, SL)
    else:
        c = pid - NQ
        rows = (c // NCB) * BR + tl.arange(0, BR)[:, None]
        cols = (c % NCB) * BC + tl.arange(0, BC)[None, :]
        m = (rows < R) & (cols < C0 + C1)
        in0 = cols < C0
        v0 = tl.load(s0 + rows * C0 + cols, mask=m & in0, other=0)
        v1 = tl.load(s1 + rows * C1 + (cols - C0), mask=m & (cols >= C0), other=0)
        tl.store(d + rows * (C0 + C1) + cols, tl.where(in0, v0, v1), mask=m)


def _quant_cat_cols(parts, dtype, ts):
    """(_quant_cat(parts, dtype), _cat_cols(ts)) in ONE launch, or None when either would not take its 2-part
    Triton kernel."""
    if len(parts) != 2 or len(ts) != 2 or any(not t.is_contiguous() or t.shape[1] % 4 for t in ts):
        return None
    x0, x1 = parts
    M, N0 = x0.shape
    N = N0 + x1.shape[1]
    if not (M % 128 == 0 and N0 % 128 == 0 and N % 128 == 0 and x0.stride(1) == 1 and x1.stride(1) == 1):
        return None
    R = ts[0].shape[0]
    if ts[1].shape[0] != R:
        return None
    c0, c1 = ts[0].shape[1] // 4, ts[1].shape[1] // 4
    dev = x0.device
    qr = torch.empty((M, N), dtype=dtype, device=dev)
    er = torch.empty((M // 128, N // 32 * 128), dtype=torch.uint8, device=dev)
    qt = torch.empty((N, M), dtype=dtype, device=dev)
    et = torch.empty((N // 128, M // 32 * 128), dtype=torch.uint8, device=dev)
    out = torch.empty((R, (c0 + c1) * 4), dtype=ts[0].dtype, device=ts[0].device)
    for t in (x0, x1, qr, er, qt, et, *ts, out):
        assert t.is_cuda and t.data_ptr() != 0
    fmax = float(torch.finfo(dtype).max)
    BR, BC = 32, 128
    nbn = N // 128
    nq = (M // 128) * nbn
    ncb = triton.cdiv(c0 + c1, BC)
    ncat = triton.cdiv(R, BR) * ncb
    _quant_cat2_cols_kernel[(nq + ncat,)](
        x0,
        x1,
        qr,
        er,
        qt,
        et,
        M,
        N,
        N0,
        x0.stride(0),
        x1.stride(0),
        ts[0].view(torch.int32),
        ts[1].view(torch.int32),
        out.view(torch.int32),
        R,
        nq,
        nbn,
        ncb,
        FP8_MAX=fmax,
        MAX_EXP=fp8_max_exp(fmax),
        SL=SL_PACKED,
        C0=c0,
        C1=c1,
        BR=BR,
        BC=BC,
        num_warps=2,
    )
    return (qr, er, qt, et), out


@triton.jit
def _quant_cat2_cols_red_kernel(
    x0_ptr,
    x1_ptr,
    qr_ptr,
    er_ptr,
    qt_ptr,
    et_ptr,
    M,
    N,
    N0,
    stride_x0,
    stride_x1,
    s0,
    s1,
    d,
    R,
    NQ,
    NBN,
    NCB,
    NCAT,
    dgp0,
    dgo0,
    dgp1,
    dgo1,
    PR,
    FP8_MAX: tl.constexpr,
    MAX_EXP: tl.constexpr,
    SL: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    H0R: tl.constexpr,
    H1R: tl.constexpr,
    RBH: tl.constexpr,
    RBP: tl.constexpr,
):
    """``_quant_cat2_cols_kernel`` (programs [0, NQ + NCAT), same bodies) plus, in programs [NQ + NCAT, ...), a
    fixed-order column sum of two fp32 partial sets [PR, H0R] / [PR, H1R] into dgo0 / dgo1 ([RBP, RBH] tiles summed
    over PR in steps of RBP, then one tl.sum over the tile rows; 4 warps). The quant / cat parts only take maxima and
    elementwise ops, so their bytes do not depend on the warp count."""
    pid = tl.program_id(0)
    if pid < NQ:
        pid_m = pid // NBN
        pid_n = pid % NBN
        offs_m = pid_m * 128 + tl.arange(0, 128)
        offs_n = pid_n * 128 + tl.arange(0, 128)
        if pid_n * 128 < N0:
            x = tl.load(x0_ptr + offs_m[:, None].to(tl.int64) * stride_x0 + offs_n[None, :]).to(tl.float32)
        else:
            x = tl.load(x1_ptr + offs_m[:, None].to(tl.int64) * stride_x1 + (offs_n[None, :] - N0)).to(
                tl.float32
            )
        ax = tl.abs(x)
        nb_n = N // 128
        nb_m = M // 128
        e = ue8m0_exp_g90(tl.max(ax, axis=1), MAX_EXP)  # [128] per row
        q = tl.clamp(x * pow2_neg(e)[:, None], -FP8_MAX, FP8_MAX)
        tl.store(qr_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], q.to(qr_ptr.dtype.element_ty))
        store_scale(er_ptr, offs_m, pid_n, e, M, nb_n, SL)
        e = ue8m0_exp_g90(tl.max(ax, axis=0), MAX_EXP)  # [128] per column
        q = tl.clamp(x * pow2_neg(e)[None, :], -FP8_MAX, FP8_MAX).to(qt_ptr.dtype.element_ty)
        tl.store(qt_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :], tl.trans(q))
        store_scale(et_ptr, offs_n, pid_m, e, N, nb_m, SL)
    elif pid < NQ + NCAT:
        c = pid - NQ
        rows = (c // NCB) * BR + tl.arange(0, BR)[:, None]
        cols = (c % NCB) * BC + tl.arange(0, BC)[None, :]
        m = (rows < R) & (cols < C0 + C1)
        in0 = cols < C0
        v0 = tl.load(s0 + rows * C0 + cols, mask=m & in0, other=0)
        v1 = tl.load(s1 + rows * C1 + (cols - C0), mask=m & (cols >= C0), other=0)
        tl.store(d + rows * (C0 + C1) + cols, tl.where(in0, v0, v1), mask=m)
    else:
        r = pid - NQ - NCAT
        NR0: tl.constexpr = H0R // RBH
        if r < NR0:
            rcols = r * RBH + tl.arange(0, RBH)
            racc = tl.zeros((RBP, RBH), dtype=tl.float32)
            for p0 in range(0, PR, RBP):
                pr = p0 + tl.arange(0, RBP)
                racc += tl.load(dgp0 + pr[:, None] * H0R + rcols[None, :], mask=(pr[:, None] < PR), other=0.0)
            tl.store(dgo0 + rcols, tl.sum(racc, axis=0).to(dgo0.dtype.element_ty))
        else:
            rcols = (r - NR0) * RBH + tl.arange(0, RBH)
            racc = tl.zeros((RBP, RBH), dtype=tl.float32)
            for p0 in range(0, PR, RBP):
                pr = p0 + tl.arange(0, RBP)
                racc += tl.load(dgp1 + pr[:, None] * H1R + rcols[None, :], mask=(pr[:, None] < PR), other=0.0)
            tl.store(dgo1 + rcols, tl.sum(racc, axis=0).to(dgo1.dtype.element_ty))


class DyPrequantSlot:
    """Hand-off between a two-output concat-N forward and the op that consumes both outputs (DeepSeek-V4: the
    q / kv post-processing after wq_a || wkv). The consumer's backward produces the concat's output gradients; it
    quantizes them right there (plus the W^T column cat and a reduction of its own) in ONE launch
    (``dy_prequant_fill``), and the concat backward takes that result instead of running ``_quant_cat_cols``: one
    launch fewer, same bytes. The concat backward checks that the gradients it receives are exactly the tensors that
    were quantized (pointer, shape, strides) and quantizes them itself otherwise."""

    __slots__ = ("g_dtype", "wts", "out_ptrs", "result", "keys")

    def __init__(self, g_dtype, wts, out_ptrs):
        self.g_dtype, self.wts, self.out_ptrs = g_dtype, wts, out_ptrs
        self.result = self.keys = None


# The slot of the latest two-output concat-N forward, consumed by ``take_dy_prequant``.
_DY_PREQ = [None]


def take_dy_prequant(a: torch.Tensor, b: torch.Tensor):
    """``(slot, swapped)``: the slot of the latest concat-N forward if its two outputs are ``(a, b)`` (by data
    pointer; ``swapped`` when they are ``(b, a)`` in its output order), else ``(None, False)``. Consumed once."""
    slot, _DY_PREQ[0] = _DY_PREQ[0], None
    if slot is not None:
        if slot.out_ptrs == (a.data_ptr(), b.data_ptr()):
            return slot, False
        if slot.out_ptrs == (b.data_ptr(), a.data_ptr()):
            return slot, True
    return None, False


def _part_key(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()))


def dy_prequant_fill(slot, parts, red) -> bool:
    """Quantize ``parts`` (the concat's output gradients in its output order, [M, N_i] each) like the concat
    backward's ``_quant_cat_cols`` and, in the same launch, reduce ``red = (dgp0, dgo0, dgp1, dgo1, P)``: fp32
    partials dgp_i [P, H_i] summed over P into dgo_i [H_i] (see ``_quant_cat2_cols_red_kernel`` for the order).
    Returns False, with nothing launched, when the shapes do not fit; the caller then does its own reduction."""
    ts = slot.wts
    if (
        len(parts) != 2
        or ts is None
        or len(ts) != 2
        or any(not t.is_contiguous() or t.shape[1] % 4 for t in ts)
    ):
        return False
    x0, x1 = parts
    M, N0 = x0.shape
    N = N0 + x1.shape[1]
    if not (M % 128 == 0 and N0 % 128 == 0 and N % 128 == 0 and x0.stride(1) == 1 and x1.stride(1) == 1):
        return False
    R = ts[0].shape[0]
    if ts[1].shape[0] != R:
        return False
    dgp0, dgo0, dgp1, dgo1, PR = red
    H0R, H1R = dgp0.shape[1], dgp1.shape[1]
    RBH, RBP = 64, 64
    if H0R % RBH or H1R % RBH or not (dgo0.is_contiguous() and dgo1.is_contiguous()):
        return False
    dtype = slot.g_dtype
    c0, c1 = ts[0].shape[1] // 4, ts[1].shape[1] // 4
    dev = x0.device
    qr = torch.empty((M, N), dtype=dtype, device=dev)
    er = torch.empty((M // 128, N // 32 * 128), dtype=torch.uint8, device=dev)
    qt = torch.empty((N, M), dtype=dtype, device=dev)
    et = torch.empty((N // 128, M // 32 * 128), dtype=torch.uint8, device=dev)
    out = torch.empty((R, (c0 + c1) * 4), dtype=ts[0].dtype, device=ts[0].device)
    for t in (x0, x1, qr, er, qt, et, *ts, out, dgp0, dgo0, dgp1, dgo1):
        assert t.is_cuda and t.data_ptr() != 0
    fmax = float(torch.finfo(dtype).max)
    BR, BC = 32, 128
    nbn = N // 128
    nq = (M // 128) * nbn
    ncb = triton.cdiv(c0 + c1, BC)
    ncat = triton.cdiv(R, BR) * ncb
    nred = H0R // RBH + H1R // RBH
    _quant_cat2_cols_red_kernel[(nq + ncat + nred,)](
        x0,
        x1,
        qr,
        er,
        qt,
        et,
        M,
        N,
        N0,
        x0.stride(0),
        x1.stride(0),
        ts[0].view(torch.int32),
        ts[1].view(torch.int32),
        out.view(torch.int32),
        R,
        nq,
        nbn,
        ncb,
        ncat,
        dgp0,
        dgo0,
        dgp1,
        dgo1,
        PR,
        FP8_MAX=fmax,
        MAX_EXP=fp8_max_exp(fmax),
        SL=SL_PACKED,
        C0=c0,
        C1=c1,
        BR=BR,
        BC=BC,
        H0R=H0R,
        H1R=H1R,
        RBH=RBH,
        RBP=RBP,
        num_warps=4,
    )
    slot.result = ((qr, er, qt, et), out)
    slot.keys = tuple(_part_key(p) for p in parts)
    return True


def _joint_main_grad(params, main_grads):
    """[sum N, K] view over adjacent main_grads (same dtype, same storage, contiguous, in this order), else None.
    Address adjacency alone is not enough: two separate allocations can sit back to back, and a view over both would
    run past the end of the first one's storage."""
    mg0 = main_grads[0]
    if mg0 is None or not mg0.is_contiguous():
        return None
    storage = mg0.untyped_storage().data_ptr()
    ptr = mg0.data_ptr()
    for mg in main_grads:
        if (
            mg is None
            or not mg.is_contiguous()
            or mg.dtype != mg0.dtype
            or mg.data_ptr() != ptr
            or mg.untyped_storage().data_ptr() != storage
        ):
            return None
        ptr += mg.numel() * mg.element_size()
    N = sum(int(mg.shape[0]) for mg in main_grads)
    return mg0.as_strided((N, mg0.shape[1]), (mg0.shape[1], 1))


class FP8GemmBlockConcatNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, config, fuse_patterns, *weights):
        from primus_turbo.pytorch.kernels.gemm import gemm_fp8_gluon_gfx1250_impl
        from primus_turbo.pytorch.ops.gemm_fp8 import _get_fp8_dtype

        a_dtype = _get_fp8_dtype(config.format, True)
        assert a_dtype == torch.float8_e4m3fn
        fused = []
        for w, pat in zip(weights, fuse_patterns):
            f, mg = _setup_fused_grad_accum(w, pat)
            fused.append((f, mg))
        ctx.fused = fused
        ctx.wparams = list(weights)
        ctx.ns = [int(w.shape[0]) for w in weights]
        x = x.contiguous()
        M, K = x.shape
        N = sum(ctx.ns)
        from primus_turbo.triton.quantization import quantization_fp8blk_dual_kernel as _qg

        # Weight-scale packing inside the activation quant launch (same bytes as a separate pack launch).
        w_j, e_j, wts = _weights_fp8(weights, a_dtype)
        (a_row, _, pa, a_t, pat), (pb, pbt) = _qg.quant_dual_pack(
            x,
            a_dtype,
            "packed",
            "packed",
            [(e_j, 128, N, K), (e_j.t(), 128, K, N)],
            site="c",
        )
        # one contiguous output per weight (no strided views downstream): GEMM i = X . W_joint[rows of W_i]^T
        outs, jobs, off = [], [], 0
        for n in ctx.ns:
            o = torch.empty((M, n), dtype=x.dtype, device=x.device)
            outs.append(o)
            jobs.append((a_row, w_j[off : off + n], pa, pb[off // 128 : (off + n) // 128], o, False))
            off += n
        if len(jobs) == 2 and ctx.ns[0] % 256 == 0 and N % 256 == 0:
            # ONE GEMM over the joint N, each 256-column tile stored into its own output (split-output
            # epilogue); same per-tile main loop, so the bits of the separate GEMMs.
            gemm_fp8_gluon_gfx1250_impl.gemm_nt_split(a_row, w_j, pa, pb, outs[0], outs[1], "n")
        else:
            for j in jobs:
                gemm_fp8_gluon_gfx1250_impl.gemm_nt(j[0], j[1], j[2], j[3], x.dtype, out=j[4])
        ctx.wts = wts
        ctx.dx_slot, _DX_ACCUM[0] = _DX_ACCUM[0], None
        # The consumer of both outputs may quantize their gradients for the backward (see DyPrequantSlot).
        ctx.preq = None
        if wts is not None and len(outs) == 2:
            ctx.preq = DyPrequantSlot(
                _get_fp8_dtype(config.format, False), wts, (outs[0].data_ptr(), outs[1].data_ptr())
            )
        _DY_PREQ[0] = ctx.preq
        ctx.save_for_backward(a_t, pat, pbt, w_j)
        ctx.config, ctx.out_dtype, ctx.K = config, x.dtype, K
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grad_outs):
        from primus_turbo.pytorch.kernels.gemm import (
            gemm_fp8_flydsl_gfx1250_impl,
            gemm_fp8_gluon_gfx1250_impl,
        )
        from primus_turbo.pytorch.ops.gemm_fp8 import _get_fp8_dtype

        a_t, pat, pbt, w_j = ctx.saved_tensors
        g_dtype = _get_fp8_dtype(ctx.config.format, False)
        M = a_t.shape[1]
        N = sum(ctx.ns)
        parts = []
        for go, n in zip(grad_outs, ctx.ns):
            if go is None:
                go = torch.zeros((M, n), dtype=ctx.out_dtype, device=a_t.device)
            parts.append(go.reshape(M, n) if go.stride(-1) == 1 else go.reshape(M, n).contiguous())
        preq, ctx.preq = ctx.preq, None
        if preq is not None and preq.result is not None and preq.keys == tuple(_part_key(p) for p in parts):
            qc = preq.result  # quantized by the consumer's backward (same kernel bodies, same bytes)
        else:
            qc = _quant_cat_cols(parts, g_dtype, ctx.wts) if ctx.wts is not None else None
        if qc is not None:  # dY quant + W^T column-cat in one launch (same bytes)
            (g_row, pg, g_t, pgt), b_t = qc
        else:
            g_row, pg, g_t, pgt = _quant_cat(parts, g_dtype)  # one quant of [dY_0 | dY_1 ...], no bf16 cat
        g = parts[0]
        # dgrad: dX[M,K] = dY[M,N] . W_joint[N,K]  ->  NT with B = W_joint^T [K, N]
        if qc is not None:
            pass
        elif ctx.wts is not None:  # one Triton column-cat of the per-weight W^T (faster than torch's fp8 cat)
            b_t = _cat_cols(ctx.wts)
        else:
            b_t = w_j.t().contiguous()

        # wgrad: dW_i[N_i,K] = dY^T[rows of W_i] . X^T
        grads = [None] * len(ctx.ns)
        grad_x = None
        if all(f for f, _ in ctx.fused):
            mgs = [mg for _, mg in ctx.fused]
            ows = [main_grad_first_write(w) for w in ctx.wparams]
            ok = all(
                gemm_fp8_flydsl_gfx1250_impl.accum_supported(mg, n, ctx.K)
                and not (ow and mg.dtype != ctx.out_dtype)
                for mg, n, ow in zip(mgs, ctx.ns, ows)
            )
            jmg = _joint_main_grad(ctx.wparams, mgs) if (ok and len(set(ows)) == 1) else None
            if jmg is not None:
                jw = lambda: gemm_fp8_gluon_gfx1250_impl.gemm_nt(
                    g_t,
                    a_t,
                    pgt,
                    pat,
                    jmg.dtype,
                    out=jmg,  # noqa: E731
                    accumulate=not ows[0],
                )
                if _overlap.enabled("WGRAD"):  # joint wgrad on the side stream (same bits)
                    _overlap.defer(jw, g_t, a_t, pgt, pat, jmg)
                    grad_x = _dgrad(g_row, b_t, pg, pbt, ctx.out_dtype, ctx.dx_slot)
                else:
                    grad_x = _dgrad(g_row, b_t, pg, pbt, ctx.out_dtype, ctx.dx_slot)
                    jw()
            else:
                # main_grads not adjacent (DSv4: wq_b's lies between wkv's and wq_a's): one wgrad per weight on its
                # row slice of dY^T (same bits as the joint GEMM)
                ovl = _overlap.enabled("WGRAD")
                if not ovl:
                    grad_x = _dgrad(g_row, b_t, pg, pbt, ctx.out_dtype, ctx.dx_slot)
                off, jobs, posts = 0, [], []
                for mg, n, ow in zip(mgs, ctx.ns, ows):
                    gt_i, pgt_i = g_t[off : off + n], pgt[off // 128 : (off + n) // 128]
                    if gemm_fp8_flydsl_gfx1250_impl.accum_supported(mg, n, ctx.K) and not (
                        ow and mg.dtype != ctx.out_dtype
                    ):
                        jobs.append((gt_i, a_t, pgt_i, pat, mg, not ow))
                    else:
                        tmp = torch.empty((n, ctx.K), dtype=ctx.out_dtype, device=g.device)
                        jobs.append((gt_i, a_t, pgt_i, pat, tmp, False))
                        posts.append((mg, tmp, ow))
                    off += n

                def wgrads():
                    if (
                        len(jobs) == 2
                        and not posts
                        and jobs[0][5] == jobs[1][5]
                        and jobs[0][4].dtype == jobs[1][4].dtype
                        and ctx.ns[0] % 256 == 0
                        and N % 256 == 0
                    ):
                        # ONE wgrad GEMM over the joint dY^T rows, each 256-row tile
                        # stored into its weight's main_grad (split-output epilogue); bits = the per-weight GEMMs'.
                        gemm_fp8_gluon_gfx1250_impl.gemm_nt_split(
                            g_t, a_t, pgt, pat, jobs[0][4], jobs[1][4], "m", accumulate=jobs[0][5]
                        )
                    else:
                        for j in jobs:
                            gemm_fp8_gluon_gfx1250_impl.gemm_nt(
                                j[0], j[1], j[2], j[3], j[4].dtype, out=j[4], accumulate=j[5]
                            )
                    for mg, tmp, ow in posts:
                        mg.copy_(tmp) if ow else mg.add_(tmp)

                if ovl:
                    # the wgrads (main_grad only) on the side stream, then the dgrad here
                    _overlap.defer(wgrads, g_t, a_t, pgt, pat, *[t for j in jobs for t in j[:5]])
                    grad_x = _dgrad(g_row, b_t, pg, pbt, ctx.out_dtype, ctx.dx_slot)
                else:
                    wgrads()
            for i, mg in enumerate(mgs):
                grads[i] = _get_dummy_wgrad(list(mg.shape), ctx.out_dtype)
        else:
            grad_x = _dgrad(g_row, b_t, pg, pbt, ctx.out_dtype, ctx.dx_slot)
            gw = gemm_fp8_gluon_gfx1250_impl.gemm_nt(g_t, a_t, pgt, pat, ctx.out_dtype)
            off = 0
            for i, ((f, mg), n, w) in enumerate(zip(ctx.fused, ctx.ns, ctx.wparams)):
                if f:
                    ow = main_grad_first_write(w)
                    mg.copy_(gw[off : off + n]) if ow else mg.add_(gw[off : off + n])
                    grads[i] = _get_dummy_wgrad(list(mg.shape), ctx.out_dtype)
                else:
                    grads[i] = gw[off : off + n]
                off += n
        ctx.wts = ctx.wparams = ctx.fused = ctx.dx_slot = ctx.preq = None
        return (grad_x, None, None, *grads)


def gemm_fp8_block_concat_n(
    x: torch.Tensor, weights: List[torch.Tensor], config, fuse_patterns: Optional[List[Optional[str]]] = None
) -> torch.Tensor:
    """(Y_0, Y_1, ..) with Y_i[M, N_i] = X[M,K] . W_i^T in block FP8 (see module doc). Call only when
    ``gemm_fp8_block_concat_n_supported`` is True. ``fuse_patterns[i]`` = the weight's fused main_grad pattern
    ("megatron") or None (gradient returned to autograd)."""
    if fuse_patterns is None:
        fuse_patterns = [None] * len(weights)
    return FP8GemmBlockConcatNFunction.apply(x, config, list(fuse_patterns), *weights)
