###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegaMoE experts in FP8 with the DeepSeek-V3/V4 recipe on gfx1250: E4M3, power-of-two (UE8M0) scales per
1x128 activation tile, per 128x128 weight block and per 128x1 (token-direction) tile of the wgrad operands.

EP=1 (one rank holds every expert); the EP>1 dispatch/combine stay with the bf16 / mxfp8 MegaMoE ops.
Everything is device-side and fixed-shape, so a layer's forward + backward can be captured in a CUDA graph:
  * routing tables from three small Triton kernels (no host sync), pool sized for the worst case
    (T*K routes + 256 rows of padding per expert),
  * expert weights quantized (with transposed copies for the dgrads) into fixed buffers by
    ``refresh_fp8blk_weights()``, to be called once after every optimizer step -- never in the forward.

Kernels: primus_turbo.triton.quantization.quantization_fp8blk_moe_kernel (routing, dispatch/SwiGLU fused quant, combine) and
primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_{mx,nt} (fwd / dgrad grouped GEMMs) and _{vkq,vkp,vk} (wgrads),
all on the gfx1250 WMMA-scale instruction.

Switches (primus_turbo.common.envs): PRIMUS_TURBO_MOE_FP8BLK_GG (grouped GEMM chain), _WGRAD (wgrad kernel), _VERIFY.
"""

import logging
import weakref
from typing import Optional

import torch

import primus_turbo.common.envs as envs
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_mxgp_kernel import mxgp_fp8blk_nt
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_nt_kernel import flygg_fp8blk_nt
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_vk_kernel import flyvkblk_fp8
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_vkp_kernel import vkpblk_sumsq_numel
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_vkq_kernel import flyvkqblk_fp8
from primus_turbo.pytorch.core import overlap as _overlap
from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, main_grad_first_write
from primus_turbo.triton.quantization.quantization_fp8blk_moe_kernel import (
    MX,
    build_routing_blk,
    combine_rows,
    combine_rows_route_dw,
    expert_order_lpt_fused,
    quant_dual_blk,
    quant_w_128x128,
    swiglu_bwd_dual_blk,
    swiglu_fwd_dual_blk,
)

__all__ = [
    "FP8BlockWeightCache",
    "early_routing_tables",
    "fused_mega_moe_fp8blk",
    "refresh_fp8blk_weights",
    "routing_pool_rows",
]

_log = logging.getLogger("primus_turbo.moe.fp8blk")
_FP8 = torch.float8_e4m3fn
_verify_logged = False
_ALIGN = 256
# PRIMUS_TURBO_MOE_FP8BLK_GG: grouped fwd/dgrad GEMM schedule (all bit-identical). Each mode falls back to the next
# when its shape test fails (mxp -> k256 -> tile):
#   mxp:  mxgp -- persistent (grid = #CUs, live tiles bounded on the device from the group offsets: no dead
#         workgroups, EP>1 worst-case pools OK), 256x256x256 k-tiles in a 2-stage ring, fragments pipelined across
#         k-tiles, next tile's k-tiles / scale planes issued during the current tile's last k-tiles; C via
#         line-scratch VMEM stores. Needs K % 512 == 0 and K a power of two;
#   k256: flygg 256x256x256 k-tiles in a 2-stage ring + B scale plane of one row per 16-row fragment (128x128-block
#         weight scales) so it fits LDS; needs K % 256 == 0;
#   tile: flygg 256x256x128, 3-stage ring, one tile per workgroup, grid bound M/256 (pool offsets are 256-aligned).
_GG_MODE = envs.PRIMUS_TURBO_MOE_FP8BLK_GG
assert _ALIGN % 256 == 0
_K256 = dict(tile=(256, 256, 256), num_buffers=2)
_TILE = dict(tile=(256, 256, 128), num_buffers=3)
_VK = dict(num_buffers=2)
_CACHES = weakref.WeakSet()
# wgrad kernel (PRIMUS_TURBO_MOE_FP8BLK_WGRAD): "vkq" (persistent, experts longest-first, old-C prefetch for beta = 1,
# scale planes on VMEM; needs the compact 1-byte-per-128-token col scales from the quant kernels) or "vk" (one tile
# per workgroup).
_VKP_LDS_KWIN = 5120  # longest expert window the vkq LDS arena holds (scale planes grow with it)
# Every expert window of the pool is a multiple of _ALIGN = 256 tokens, so the beta-0 vkq wgrads run as vkw
# (4 waves of 128x128, 256-token k-tiles; main_grad and the sumsq partials bit-identical to vkq, 7-10% faster);
# beta-1 (accumulating) calls keep vkq.
_VKW = _ALIGN % 256 == 0
# longest expert window the vk LDS arena holds (a power of two: the scale-plane TDM pad interval)
_VK_LDS_KWIN = 8192


def _wgrad_kernel(T: int) -> str:
    mode = envs.PRIMUS_TURBO_MOE_FP8BLK_WGRAD
    if mode == "vkq" and _vkp_kwin(T) > _VKP_LDS_KWIN:
        return "vk"
    return mode


def _vk_kwin(window: int) -> int:
    """kwin_max of the vk wgrad for expert windows of at most ``window`` tokens: the next power of two >= max(4096,
    window), capped at what the LDS arena holds (``_VK_LDS_KWIN``)."""
    kwin = 4096
    while kwin < window and kwin < _VK_LDS_KWIN:
        kwin *= 2
    return kwin


def _vkp_kwin(T: int) -> int:
    # an expert window holds each token at most once, padded to 256 rows (and >= 256)
    return max(4096, -(-T // _ALIGN) * _ALIGN)


def _materialize(w: torch.Tensor):
    """Run the weight's materialize hook, if an optimizer registered one: an optimizer that keeps only the FP32
    master (and this cache) current between steps, without the BF16 weight, rebuilds the BF16 weight before
    anything reads it."""
    hook = getattr(w, FP8BlockWeightCache.MATERIALIZE_HOOK, None)
    if hook is not None:
        hook()


class FP8BlockWeightCache:
    """Fixed buffers holding one expert weight [E, N, K] in 128x128-block FP8 (+ the transposed copy)."""

    # Name of the optional zero-argument hook on ``weight`` that brings a lazily updated BF16 weight up to date;
    # refresh() and the VERIFY check call it before reading the weight.
    MATERIALIZE_HOOK = "_fp8blk_materialize"

    def __init__(self, w: torch.Tensor):
        E, N, K = w.shape
        dev = w.device
        self.weight = w
        self.q = torch.empty((E, N, K), device=dev, dtype=_FP8)
        self.s = torch.empty((E, N, K // MX), device=dev, dtype=torch.uint8)
        self.qT = torch.empty((E, K, N), device=dev, dtype=_FP8)
        self.sT = torch.empty((E, K, N // MX), device=dev, dtype=torch.uint8)
        self.valid = False
        # Set by an optimizer that writes q / s / qT / sT itself with the bytes of refresh() after every
        # update of ``weight``; refresh_fp8blk_weights() then skips it.
        self.adam_managed = False
        _CACHES.add(self)

    @torch.no_grad()
    def refresh(self):
        _materialize(self.weight)
        quant_w_128x128(self.weight.detach(), _FP8, transposed=True, out=(self.q, self.s, self.qT, self.sT))
        self.valid = True


@torch.no_grad()
def refresh_fp8blk_weights():
    """Re-quantize every live FP8-block expert weight not written by the optimizer itself (``adam_managed``).
    Call once after each optimizer step (outside graphs)."""
    verify = envs.PRIMUS_TURBO_MOE_FP8BLK_VERIFY
    n_refreshed = n_verified = 0
    for c in list(_CACHES):
        if not (c.adam_managed and c.valid):
            c.refresh()
            n_refreshed += 1
        elif verify:  # debug: the optimizer-written bytes must equal a fresh quant_w_128x128 (costs a sync)
            _materialize(c.weight)
            ref = quant_w_128x128(c.weight.detach(), _FP8, transposed=True)
            bad = [
                n
                for n, a, b in zip(("q", "s", "qT", "sT"), (c.q, c.s, c.qT, c.sT), ref)
                if not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            ]
            if bad:
                raise RuntimeError(
                    f"PRIMUS_TURBO_MOE_FP8BLK_VERIFY: optimizer-written {bad} differ from "
                    f"quant_w_128x128 of {tuple(c.weight.shape)}"
                )
            c.verified = getattr(c, "verified", 0) + 1
            n_verified += 1
    global _verify_logged
    if verify and not _verify_logged:  # once per process
        _verify_logged = True
        _log.info(
            "PRIMUS_TURBO_MOE_FP8BLK_VERIFY: caches=%d refreshed=%d adam-written verified byte-equal=%d",
            len(_CACHES),
            n_refreshed,
            n_verified,
        )


def _gg(a, a_s, c, transposed, poffs):
    """Grouped NT GEMM over the expert pool: a [P, K] fp8 x expert weights (c.q, or c.qT for the dgrads)."""
    w, ws = (c.qT, c.sT) if transposed else (c.q, c.s)
    P, K = a.shape
    if _GG_MODE == "mxp" and K % 512 == 0 and (K & (K - 1)) == 0 and w.shape[1] % 256 == 0 and P % 256 == 0:
        return mxgp_fp8blk_nt(a, w, a_s, ws, poffs)
    return flygg_fp8blk_nt(a, w, a_s, ws, poffs, **(_K256 if _GG_MODE != "tile" and K % 256 == 0 else _TILE))


def _wgrad_sumsq_buf(w: torch.Tensor, mg: torch.Tensor):
    """Persistent fp32 partial buffer the vkq wgrad fills with the sum of squares of
    ``w.main_grad`` (its final bf16 values) in its epilogue; exposed as ``w._primus_main_grad_sumsq`` (a tensor whose
    sum is sum(main_grad^2)) and flagged ``w._primus_main_grad_sumsq_fresh`` for the grad-norm pass, which then skips
    re-reading this weight's grad."""
    buf = getattr(w, "_primus_main_grad_sumsq", None)
    n = vkpblk_sumsq_numel(mg)
    if buf is None or buf.numel() != n or buf.device != mg.device:
        buf = torch.zeros(n, dtype=torch.float32, device=mg.device)
        w._primus_main_grad_sumsq = buf
    return buf


def _setup_accum(w: torch.Tensor, accumulate: bool):
    mg = getattr(w, "main_grad", None) if accumulate else None
    if mg is not None and mg.dtype == torch.bfloat16 and mg.shape == w.shape and mg.is_contiguous():
        return mg
    return None


def routing_pool_rows(T: int, K: int, E: int) -> int:
    """Rows of the fixed-size expert pool (T*K routes + ``_ALIGN`` rows of padding per expert)."""
    return T * K + E * _ALIGN


def early_routing_tables(topk_idx: torch.Tensor, topk_w: torch.Tensor, E: int):
    """The pool tables of ``_FP8BlockMoE.forward`` (same kernels, same inputs -> same bytes), built by the caller
    wherever it likes (e.g. before a stream join); pass the result as ``routing=``. The fifth entry is None (the
    backward builds the LPT order)."""
    T, K = topk_idx.shape
    P = routing_pool_rows(T, K, E)
    rowmap, rw, inv, poffs, _ = build_routing_blk(
        topk_idx.to(torch.int64).contiguous(), topk_w.float(), E, P, align=_ALIGN
    )
    return (rowmap, rw, inv, poffs, None)


def _run_wgrad(wk, lhs, rhs, lhs_s, rhs_s, poffs, n, k, out, beta, perm, window, sq=None):
    """One expert wgrad launch: out [E, n, k] bf16 (+)= per-expert lhs[:, seg] @ rhs[:, seg]^T. ``window``: upper
    bound (tokens) of any expert window; ``perm``: LPT expert order (vkq only); ``sq``: sumsq partials (vkq only)."""
    if wk == "vkq":
        flyvkqblk_fp8(
            lhs,
            rhs,
            lhs_s,
            rhs_s,
            poffs,
            n,
            k,
            out,
            beta=beta,
            perm=perm,
            kwin_max=_vkp_kwin(window),
            sumsq_out=sq,
            vkw=_VKW,
        )
    else:
        bound = _vkp_kwin(window)  # an expert window holds each token at most once, padded to 256 rows
        kwin = _vk_kwin(bound)
        if bound > kwin:
            # the bound exceeds what vk's LDS scale planes hold: a longer expert window would read the wrong scale
            # bytes, so check the real windows on the device (no host sync; aborts instead of a silent wrong result)
            torch._assert_async((poffs[1:] - poffs[:-1]).max() <= kwin)
        flyvkblk_fp8(lhs, rhs, lhs_s, rhs_s, poffs, n, k, out, beta=beta, kwin_max=kwin, **_VK)


class _FP8BlockMoE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        topk_idx,
        topk_w,
        w1,
        w2,
        c1,
        c2,
        activation,
        accumulate,
        add=None,
        routing=None,
    ):
        T, H = x.shape
        K = topk_idx.shape[1]
        E, I2, _ = w1.shape
        I = I2 // 2
        P = T * K + E * _ALIGN
        if not (c1.valid and c2.valid):  # first use only (eager, before any capture)
            c1.refresh()
            c2.refresh()
        if routing is not None:
            # precomputed pool tables (early_routing_tables), same bytes as below
            rowmap, route_w, inv, poffs, _ = routing
            assert (
                rowmap.numel() == P
                and route_w.numel() == P
                and inv.numel() == T * K
                and poffs.numel() == E + 1
            )
            preal = poffs[E:]
        else:
            rowmap, route_w, inv, poffs, preal = build_routing_blk(
                topk_idx, topk_w.float(), E, P, align=_ALIGN
            )
        wk = _wgrad_kernel(T)
        ct = wk == "vkq"  # col (token-direction) scales compact for the persistent wgrad
        xq, xs, xT, xTs = quant_dual_blk(x, P, rowmap, preal=preal, compact_t=ct)
        l1 = _gg(xq, xs, c1, False, poffs)
        aq, as_, aT, aTs = swiglu_fwd_dual_blk(l1, route_w, activation, preal=preal, compact_t=ct)
        l2 = _gg(aq, as_, c2, False, poffs)
        y = combine_rows(l2, inv, T, K, add=add)
        ctx.has_add = add is not None
        ctx.save_for_backward(xT, xTs, aT, aTs, l1, route_w, rowmap, inv, poffs, preal)
        ctx.c1, ctx.c2, ctx.w1, ctx.w2 = c1, c2, w1, w2
        ctx.activation, ctx.accumulate, ctx.dims = activation, accumulate, (T, H, K, E, I, P)
        ctx.wk = wk
        ctx.topk_dtype = topk_w.dtype
        return y

    @staticmethod
    def backward(ctx, dy):
        xT, xTs, aT, aTs, l1, route_w, rowmap, inv, poffs, preal = ctx.saved_tensors
        T, H, K, E, I, P = ctx.dims
        c1, c2, w1, w2 = ctx.c1, ctx.c2, ctx.w1, ctx.w2
        vkq = ctx.wk == "vkq"  # compact col scales, LPT order, sumsq fold
        dq, ds, dT, dTs = quant_dual_blk(
            dy.contiguous().to(torch.bfloat16), P, rowmap, preal=preal, compact_t=vkq
        )
        dact = _gg(dq, ds, c2, True, poffs)
        gq, gs, gT, gTs, dwr = swiglu_bwd_dual_blk(
            dact, l1, route_w, ctx.activation, preal=preal, compact_t=vkq, raw_dw=True
        )
        jobs = ((w2, dT, dTs, aT, aTs, H, I), (w1, gT, gTs, xT, xTs, 2 * I, H))
        mgs = [_setup_accum(w, ctx.accumulate) for w, *_ in jobs]
        deferred = _overlap.enabled("MOE_WGRAD") and all(mg is not None for mg in mgs)
        if deferred:
            # MOE_WGRAD overlap: the LPT order and both expert wgrads (they only feed the optimizer) run on
            # the side stream next to the fc1 dgrad / combine / router backward; same kernels and inputs -> same bits
            # (main_grad and the sumsq partials are per weight; nothing reads them before the join).
            betas = [0.0 if main_grad_first_write(w) else 1.0 for w, *_ in jobs]
            sqs = [_wgrad_sumsq_buf(w, mg) if vkq else None for (w, *_), mg in zip(jobs, mgs)]

            def side_wgrads():
                # experts longest-first (one tiny order kernel shared by both wgrads)
                perm_s = expert_order_lpt_fused(poffs) if vkq else None
                for (w, lhs, lhs_s, rhs, rhs_s, n, k), mg, beta, sq in zip(jobs, mgs, betas, sqs):
                    _run_wgrad(ctx.wk, lhs, rhs, lhs_s, rhs_s, poffs, n, k, mg, beta, perm_s, T, sq)
                return perm_s

            _overlap.defer(side_wgrads, dT, dTs, aT, aTs, gT, gTs, xT, xTs, poffs, *mgs, *sqs)
            for (w, *_), sq in zip(jobs, sqs):
                w._primus_main_grad_sumsq_fresh = sq is not None
                w.grad_added_to_main_grad = True
            jobs = ()  # nothing left for the serial loop below
        perm = expert_order_lpt_fused(poffs) if (vkq and not deferred) else None

        grads = []
        for (w, lhs, lhs_s, rhs, rhs_s, n, k), mg in zip(jobs, mgs):
            if mg is not None:
                # beta 0 on the step's first write into main_grad (framework skipped its clear), else accumulate
                # every call writes all of main_grad, so with the fold the last call of the step leaves the sum of
                # squares of the final grad (multi-microbatch accumulation included)
                sq = _wgrad_sumsq_buf(w, mg) if vkq else None
                beta = 0.0 if main_grad_first_write(w) else 1.0
                _run_wgrad(ctx.wk, lhs, rhs, lhs_s, rhs_s, poffs, n, k, mg, beta, perm, T, sq)
                w._primus_main_grad_sumsq_fresh = sq is not None
                w.grad_added_to_main_grad = True
                grads.append(_get_dummy_wgrad(w.shape, w.dtype))  # dummy, never read
            else:
                g = torch.empty((E, n, k), device=w.device, dtype=torch.bfloat16)
                _run_wgrad(ctx.wk, lhs, rhs, lhs_s, rhs_s, poffs, n, k, g, 0.0, perm, T)
                grads.append(g.to(w.dtype))
        dW2, dW1 = grads if not deferred else (_get_dummy_wgrad(w.shape, w.dtype) for w in (w2, w1))
        dxp = _gg(gq, gs, c1, True, poffs)
        # combine + d topk-weights (independent kernels) as one launch
        dx, dtw = combine_rows_route_dw(dxp, inv, T, K, dwr, ctx.topk_dtype)
        dtw = dtw.view(T, K)
        return dx, None, dtw, dW1, dW2, None, None, None, None, (dy if ctx.has_add else None), None


def fused_mega_moe_fp8blk(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    cache1: FP8BlockWeightCache,
    cache2: FP8BlockWeightCache,
    activation=None,
    accumulate_wgrad: bool = False,
    add: Optional[torch.Tensor] = None,
    routing=None,
) -> torch.Tensor:
    """y [T, H] = sum_k topk_w[t, k] * w2[e] act(w1[e] x[t]) with e = topk_idx[t, k], all GEMMs FP8 128-block.

    x [T, H] bf16, w1 [E, 2I, H], w2 [E, H, I] (H, I multiples of 256). ``cache1``/``cache2`` hold the quantized
    weights (``FP8BlockWeightCache(w)``, refreshed by ``refresh_fp8blk_weights()`` after each optimizer step).
    ``accumulate_wgrad``: add dW straight into ``w.main_grad`` (bf16, Megatron DDP) and set
    ``grad_added_to_main_grad``; the returned weight gradient is then a dummy.
    ``add`` [T, H] bf16 (e.g. the shared expert output): returned y is ``y + add`` with the add fused into the
    combine kernel (bit-identical to the separate bf16 add; its gradient is dy).
    ``routing``: precomputed pool tables from ``early_routing_tables`` for these ``topk_idx`` / ``topk_weights``;
    skips the three table kernels."""
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    assert w1.shape[0] == w2.shape[0] and w1.shape[2] == x.shape[1] and w2.shape[1] == x.shape[1]
    assert w1.shape[1] % 256 == 0 and x.shape[1] % 256 == 0 and w2.shape[2] * 2 == w1.shape[1]
    if add is not None:
        add = add.reshape(x.shape).contiguous()
    return _FP8BlockMoE.apply(
        x.contiguous(),
        topk_idx.to(torch.int64).contiguous(),
        topk_weights,
        w1,
        w2,
        cache1,
        cache2,
        activation,
        accumulate_wgrad,
        add,
        routing,
    )
