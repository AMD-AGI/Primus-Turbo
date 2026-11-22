###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Concat-N block-FP8 linear: Y_i = X . W_i^T for weights sharing X, one launch per pass (gfx1250 dsv4up path).

For weights that read the same input (DeepSeek-V4 wq_a || wkv), with the DSv4 recipe (1x128 activation /
128x128 weight blocks, UE8M0 scales, ``PRIMUS_TURBO_FP8_BLOCK_KERNEL=dsv4up``):
  fwd    one 1x128 quant of X, one launch writing one CONTIGUOUS output per weight (dsv4up dual for two
         weights; bits = the separate GEMMs')
  bwd    one 1x128 quant of [dY_0 | dY_1] read from the separate grads (no bf16 concat copy)
  dgrad  one GEMM with K = sum N_i                       (fp32 over all K, one rounding: NOT the bits of
                                                           separate dgrads + a bf16 add)
  wgrad  one GEMM into the joint main_grad if the weights' main_grads are adjacent in that order, else one GEMM
         per weight on its row slice of dY^T             (bits = the separate GEMMs')
Weights: with PRIMUS_TURBO_FP8_WCACHE the optimizer writes each weight's FP8 W / scales into its row slice of
one joint buffer (fp8blk_weight_cache.attach_joint), so the forward reads the joint W with no copy; the dgrad
operand W^T is one concat of the per-weight W^T copies per step. Without a ready cache each weight is quantized
here (128-row blocks never straddle two weights, so this equals the per-weight quant).
"""

import os
from typing import List, Optional, Sequence

import torch
import triton
import triton.language as tl

from primus_turbo.pytorch.core import fp8blk_weight_cache
from primus_turbo.pytorch.core import overlap as _overlap
from primus_turbo.triton.quantization.quant_epi_gfx1250 import (
    SL_PACKED,
    fp8_max_exp,
    pow2_neg,
    store_scale,
    ue8m0_exp_g90,
)
from primus_turbo.pytorch.ops.utils import (
    _get_dummy_wgrad,
    _setup_fused_grad_accum,
    main_grad_first_write,
)

__all__ = ["gemm_fp8_block_concat_n", "gemm_fp8_block_concat_n_supported", "request_dx_accum"]

# fuse2 (PRIMUS_FUSE2_ATT_GADD): an object with ``take(M, K, dtype) -> Tensor | None`` and a ``fused`` attribute,
# handed to the next concat-N forward. Its backward then writes dX into the returned tensor as
# bf16(dX) + tensor (dsv4up accumulate=2: torch's bf16 add) and records it in ``fused``.
_DX_ACCUM = [None]


def request_dx_accum(slot) -> None:
    _DX_ACCUM[0] = slot


def _dgrad(g_row, b_t, pg, pbt, out_dtype, slot):
    from primus_turbo.gluon.gemm import dsv4up_block

    if slot is not None:
        M, K = g_row.shape[0], b_t.shape[0]
        acc = slot.take(M, K, out_dtype)
        if (acc is not None and dsv4up_block.split_k_of(M, K, g_row.shape[1]) == 1
                and not dsv4up_block.small_tile_of(M, K, g_row.shape[1], True)
                and not dsv4up_block.small_tile_of(M, K, g_row.shape[1], False)):
            from primus_turbo.gluon.gemm.dsv4up_gemm import dsv4up_gemm

            dsv4up_gemm(g_row, b_t, pg, pbt, out=acc, accumulate=2)
            slot.fused = acc
            return acc
        if acc is not None:  # cannot fuse: give the buffer back untouched, autograd adds as usual
            slot.untake(acc)
    return dsv4up_block.gemm_nt(g_row, b_t, pg, pbt, out_dtype)


def _fuseq_kw():
    if os.environ.get("PRIMUS_TURBO_FUSEQ", "0") != "1":
        return None
    return {"tmode": int(os.environ.get("PRIMUS_TURBO_FUSEQ_TMODE", "0")),
            "num_warps": int(os.environ.get("PRIMUS_TURBO_FUSEQ_NW", "2"))}


def gemm_fp8_block_concat_n_supported(x: torch.Tensor, weights: Sequence[torch.Tensor], config) -> bool:
    """True if the concat-N op can run all three passes on dsv4up for these shapes / this config."""
    try:
        from primus_turbo.gluon.gemm import dsv4up_block
        from primus_turbo.pytorch.core.low_precision import ScalingGranularity
    except Exception:
        return False
    if os.environ.get("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "0") != "1" or not dsv4up_block.enabled():
        return False
    if config is None or config.granularity != ScalingGranularity.BLOCKWISE or getattr(config, "block_size", 128) != 128:
        return False
    if x.dim() != 2 or not x.is_cuda or len(weights) < 2:
        return False
    M, K = x.shape
    if any(w.dim() != 2 or w.shape[1] != K or w.shape[0] % 128 for w in weights):
        return False
    N = sum(int(w.shape[0]) for w in weights)
    if M % 512 or K % 512:
        return False
    sh = dsv4up_block.shape_supported
    if not (sh(M, N, K) and sh(M, K, N) and all(sh(int(w.shape[0]), K, M) for w in weights)):
        return False
    # an explicit PRIMUS_TURBO_DSV4UP_PREFER_G90 entry for one of these passes keeps the old (separate) path
    pref = dsv4up_block._prefer_g90()
    return not ({("fwd", M, N, K), ("dgrad", M, K, N)} & pref)


def _weights_fp8(weights, b_dtype):
    """-> (w_joint [N,K] fp8, e8m0_joint [N/128,K/128] u8, wt_list [K,N_i] or None). Joint cache when ready."""
    jc = fp8blk_weight_cache.attach_joint(list(weights), b_dtype)
    if jc is not None and jc.ready(weights, b_dtype):
        for c in jc.caches:
            c.hits += 1
        return jc.w, jc.e8m0, [c.wt for c in jc.caches]
    caches = [fp8blk_weight_cache.lookup(w, b_dtype) for w in weights]
    if all(c is not None for c in caches):  # separate (non-joint) caches: one concat per step
        return (torch.cat([c.w for c in caches], 0), torch.cat([c.e8m0 for c in caches], 0),
                [c.wt for c in caches])
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity, ScalingRecipe
    from primus_turbo.pytorch.ops.quantization import quantize_fp8
    from primus_turbo.triton.quantization.quant_g90 import e8m0_of

    qs, es = [], []
    for w in weights:  # same quant as FP8GemmBlockFunction's no-cache path, per weight
        q, s = quantize_fp8(w, b_dtype, ScalingGranularity.BLOCKWISE, block_size=128,
                            scaling_recipe=ScalingRecipe(use_2d_block=True))
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
        ts[0].view(torch.int32), ts[1].view(torch.int32), out.view(torch.int32), R, C0=c0, C1=c1, BR=BR, BC=BC,
        num_warps=4)
    return out


@triton.jit
def _quant_cat2_kernel(x0_ptr, x1_ptr, qr_ptr, er_ptr, qt_ptr, et_ptr, M, N, N0, stride_x0, stride_x1,
                       FP8_MAX: tl.constexpr, MAX_EXP: tl.constexpr, SL: tl.constexpr):
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
    from primus_turbo.triton.quantization.quant_g90 import quant_dual_t_e8m0_v2

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
    _quant_cat2_kernel[(M // 128, N // 128)](x0, x1, qr, er, qt, et, M, N, N0, x0.stride(0), x1.stride(0),
                                             FP8_MAX=fmax, MAX_EXP=fp8_max_exp(fmax), SL=SL_PACKED, num_warps=2)
    return qr, er, qt, et


def _joint_main_grad(params, main_grads):
    """[sum N, K] view over adjacent main_grads (same dtype, contiguous, in this order), else None."""
    mg0 = main_grads[0]
    if mg0 is None or not mg0.is_contiguous():
        return None
    ptr = mg0.data_ptr()
    for mg in main_grads:
        if mg is None or not mg.is_contiguous() or mg.dtype != mg0.dtype or mg.data_ptr() != ptr:
            return None
        ptr += mg.numel() * mg.element_size()
    N = sum(int(mg.shape[0]) for mg in main_grads)
    return mg0.as_strided((N, mg0.shape[1]), (mg0.shape[1], 1))


class FP8GemmBlockConcatNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, config, fuse_patterns, *weights):
        from primus_turbo.gluon.gemm import dsv4up_block
        from primus_turbo.pytorch.ops.gemm_fp8 import _get_fp8_dtype
        from primus_turbo.triton.quantization.quant_g90 import quant_dual_t_e8m0, quant_dual_t_e8m0_v2

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
        fq = _fuseq_kw()
        from primus_turbo.triton.quantization import quant_g90 as _qg

        if fq is not None and fq["tmode"] == 0 and _qg.mb_packq_enabled():
            # membound PRIMUS_MB_PACKQ=1: weight-scale packing inside the activation quant launch (same bytes)
            w_j, e_j, wts = _weights_fp8(weights, a_dtype)
            (a_row, _, pa, a_t, pat), (pb, pbt) = _qg.quant_dual_pack(
                x, a_dtype, "packed", "packed", [(e_j, 128, N, K), (e_j.t(), 128, K, N)], site="c",
                num_warps=fq["num_warps"])
        elif fq is not None:
            a_row, _, pa, a_t, pat = quant_dual_t_e8m0_v2(x, a_dtype, row_layout="packed", t_layout="packed",
                                                          site="c", **fq)
            w_j, e_j, wts = _weights_fp8(weights, a_dtype)
            pb, pbt = dsv4up_block.pack_scales([(e_j, 128, N, K), (e_j.t(), 128, K, N)])
        else:
            a_row, _, a_re, a_t, a_te = quant_dual_t_e8m0(x, a_dtype)
            w_j, e_j, wts = _weights_fp8(weights, a_dtype)
            pa, pb, pbt, pat = dsv4up_block.pack_scales([(a_re, 1, M, K), (e_j, 128, N, K), (e_j.t(), 128, K, N),
                                                          (a_te, 1, K, M)])
        # one contiguous output per weight (no strided views downstream): GEMM i = X . W_joint[rows of W_i]^T, both
        # in ONE launch for two weights (dsv4up dual: same per-tile code as separate launches -> same bits)
        outs, jobs, off = [], [], 0
        for n in ctx.ns:
            o = torch.empty((M, n), dtype=x.dtype, device=x.device)
            outs.append(o)
            jobs.append((a_row, w_j[off:off + n], pa, pb[off // 128:(off + n) // 128], o, False))
            off += n
        tiles = [(M // 256) * (n // 256) for n in ctx.ns]
        if len(jobs) == 2 and dsv4up_block.split_epi_enabled() and ctx.ns[0] % 256 == 0 and N % 256 == 0:
            # PRIMUS_TURBO_CONCAT_SPLIT_EPI=1: ONE GEMM over the joint N, each 256-column tile stored into its own
            # output (split-output epilogue); same per-tile main loop -> the separate GEMMs' bits.
            dsv4up_block.gemm_nt_split(a_row, w_j, pa, pb, outs[0], outs[1], "n")
        elif len(jobs) == 2 and (tiles[0] % 8 == 0 or tiles[1] % 8 == 0) and K <= 8192:
            dsv4up_block.gemm_nt_dual(*(jobs if tiles[0] % 8 == 0 else jobs[::-1]))
        else:
            for j in jobs:
                dsv4up_block.gemm_nt(j[0], j[1], j[2], j[3], x.dtype, out=j[4])
        ctx.wts = wts
        ctx.dx_slot, _DX_ACCUM[0] = _DX_ACCUM[0], None
        ctx.save_for_backward(a_t, pat, pbt, w_j)
        ctx.config, ctx.out_dtype, ctx.K = config, x.dtype, K
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grad_outs):
        from primus_turbo.gluon.gemm import dsv4up_block
        from primus_turbo.flydsl.gemm import g90_block
        from primus_turbo.pytorch.ops.gemm_fp8 import _get_fp8_dtype
        from primus_turbo.triton.quantization.quant_g90 import quant_dual_t_e8m0, quant_dual_t_e8m0_v2

        a_t, pat, pbt, w_j = ctx.saved_tensors
        g_dtype = _get_fp8_dtype(ctx.config.format, False)
        M = a_t.shape[1]
        N = sum(ctx.ns)
        parts = []
        for go, n in zip(grad_outs, ctx.ns):
            if go is None:
                go = torch.zeros((M, n), dtype=ctx.out_dtype, device=a_t.device)
            parts.append(go.reshape(M, n) if go.stride(-1) == 1 else go.reshape(M, n).contiguous())
        g_row, pg, g_t, pgt = _quant_cat(parts, g_dtype)  # one quant of [dY_0 | dY_1 ...], no bf16 cat
        g = parts[0]
        # dgrad: dX[M,K] = dY[M,N] . W_joint[N,K]  ->  NT with B = W_joint^T [K, N]
        if ctx.wts is not None:  # one Triton column-cat of the per-weight W^T (torch's fp8 cat runs ~9 us)
            b_t = _cat_cols(ctx.wts)
        else:
            b_t = w_j.t().contiguous()
        # wgrad: dW_i[N_i,K] = dY^T[rows of W_i] . X^T
        grads = [None] * len(ctx.ns)
        grad_x = None
        if all(f for f, _ in ctx.fused):
            mgs = [mg for _, mg in ctx.fused]
            ows = [main_grad_first_write(w) for w in ctx.wparams]
            ok = all(g90_block.accum_supported(mg, n, ctx.K) and not (ow and mg.dtype != ctx.out_dtype)
                     for mg, n, ow in zip(mgs, ctx.ns, ows))
            jmg = _joint_main_grad(ctx.wparams, mgs) if (ok and len(set(ows)) == 1) else None
            if jmg is not None and dsv4up_block.dual_enabled() and (N // 256) * (ctx.K // 256) % 8 == 0 and N <= 8192:
                # PRIMUS_TURBO_DSV4UP_DUAL: joint wgrad (K = tokens, first) and dgrad (K = sum N_i) in one launch
                grad_x = torch.empty((M, ctx.K), dtype=ctx.out_dtype, device=g.device)
                dsv4up_block.gemm_nt_dual((g_t, a_t, pgt, pat, jmg, not ows[0]), (g_row, b_t, pg, pbt, grad_x, False))
            elif jmg is not None:
                jw = lambda: dsv4up_block.gemm_nt(g_t, a_t, pgt, pat, jmg.dtype, out=jmg,  # noqa: E731
                                                  accumulate=not ows[0])
                if _overlap.enabled("WGRAD"):  # PRIMUS_OVERLAP_WGRAD=1: joint wgrad on the side stream (same bits)
                    _overlap.defer(jw, g_t, a_t, pgt, pat, jmg)
                    grad_x = _dgrad(g_row, b_t, pg, pbt, ctx.out_dtype, ctx.dx_slot)
                else:
                    grad_x = _dgrad(g_row, b_t, pg, pbt, ctx.out_dtype, ctx.dx_slot)
                    jw()
            else:
                # main_grads not adjacent (DSv4: wq_b's lies between wkv's and wq_a's): one wgrad per weight on its
                # row slice of dY^T, the two in ONE dsv4up dual launch (same bits as separate launches)
                ovl = _overlap.enabled("WGRAD")
                if not ovl:
                    grad_x = _dgrad(g_row, b_t, pg, pbt, ctx.out_dtype, ctx.dx_slot)
                off, jobs, posts = 0, [], []
                for mg, n, ow in zip(mgs, ctx.ns, ows):
                    gt_i, pgt_i = g_t[off:off + n], pgt[off // 128:(off + n) // 128]
                    if g90_block.accum_supported(mg, n, ctx.K) and not (ow and mg.dtype != ctx.out_dtype):
                        jobs.append((gt_i, a_t, pgt_i, pat, mg, not ow))
                    else:
                        tmp = torch.empty((n, ctx.K), dtype=ctx.out_dtype, device=g.device)
                        jobs.append((gt_i, a_t, pgt_i, pat, tmp, False))
                        posts.append((mg, tmp, ow))
                    off += n
                tiles = [(n // 256) * (ctx.K // 256) for n in ctx.ns]

                def wgrads():
                    if (len(jobs) == 2 and not posts and dsv4up_block.split_epi_enabled() and jobs[0][5] == jobs[1][5]
                            and jobs[0][4].dtype == jobs[1][4].dtype and ctx.ns[0] % 256 == 0 and N % 256 == 0):
                        # PRIMUS_TURBO_CONCAT_SPLIT_EPI=1: ONE wgrad GEMM over the joint dY^T rows, each 256-row tile
                        # stored into its weight's main_grad (split-output epilogue); bits = the per-weight GEMMs'.
                        dsv4up_block.gemm_nt_split(g_t, a_t, pgt, pat, jobs[0][4], jobs[1][4], "m",
                                                   accumulate=jobs[0][5])
                    elif len(jobs) == 2 and M <= 8192 and (tiles[0] % 8 == 0 or tiles[1] % 8 == 0):
                        dsv4up_block.gemm_nt_dual(*(jobs if tiles[0] % 8 == 0 else jobs[::-1]))
                    else:
                        for j in jobs:
                            dsv4up_block.gemm_nt(j[0], j[1], j[2], j[3], j[4].dtype, out=j[4], accumulate=j[5])
                    for mg, tmp, ow in posts:
                        mg.copy_(tmp) if ow else mg.add_(tmp)

                if ovl:
                    # PRIMUS_OVERLAP_WGRAD=1: the wgrads (main_grad only) on the side stream, then the dgrad here
                    _overlap.defer(wgrads, g_t, a_t, pgt, pat, *[t for j in jobs for t in j[:5]])
                    grad_x = _dgrad(g_row, b_t, pg, pbt, ctx.out_dtype, ctx.dx_slot)
                else:
                    wgrads()
            for i, mg in enumerate(mgs):
                grads[i] = _get_dummy_wgrad(list(mg.shape), ctx.out_dtype)
        else:
            grad_x = _dgrad(g_row, b_t, pg, pbt, ctx.out_dtype, ctx.dx_slot)
            gw = dsv4up_block.gemm_nt(g_t, a_t, pgt, pat, ctx.out_dtype)
            off = 0
            for i, ((f, mg), n, w) in enumerate(zip(ctx.fused, ctx.ns, ctx.wparams)):
                if f:
                    ow = main_grad_first_write(w)
                    mg.copy_(gw[off:off + n]) if ow else mg.add_(gw[off:off + n])
                    grads[i] = _get_dummy_wgrad(list(mg.shape), ctx.out_dtype)
                else:
                    grads[i] = gw[off:off + n]
                off += n
        ctx.wts = ctx.wparams = ctx.fused = ctx.dx_slot = None
        return (grad_x, None, None, *grads)


def gemm_fp8_block_concat_n(x: torch.Tensor, weights: List[torch.Tensor], config,
                            fuse_patterns: Optional[List[Optional[str]]] = None) -> torch.Tensor:
    """(Y_0, Y_1, ..) with Y_i[M, N_i] = X[M,K] . W_i^T in block FP8 (see module doc). Call only when
    ``gemm_fp8_block_concat_n_supported`` is True. ``fuse_patterns[i]`` = the weight's fused main_grad pattern
    ("megatron") or None (gradient returned to autograd)."""
    if fuse_patterns is None:
        fuse_patterns = [None] * len(weights)
    return FP8GemmBlockConcatNFunction.apply(x, config, list(fuse_patterns), *weights)
