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

Kernels: primus_turbo.triton.quantization.quant_blk_gfx1250 (routing, dispatch/SwiGLU fused quant, combine),
primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_{nt,vk}_flydsl_gfx1250 (WMMA-scale GEMMs).
"""

import os
import weakref
from typing import Optional

import torch

from primus_turbo.pytorch.core import overlap as _overlap
from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, main_grad_first_write
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_mx_flydsl_gfx1250 import mxgg_fp8blk_nt, mxgk_fp8blk_nt, mxgp_fp8blk_nt
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_nt_flydsl_gfx1250 import flygg_fp8blk_nt
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_vk_flydsl_gfx1250 import flyvkblk_fp8
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_vkp_flydsl_gfx1250 import (
    expert_order_lpt,
    flyvkpblk_fp8,
    vkpblk_sumsq_numel,
)
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_vkq_flydsl_gfx1250 import flyvkqblk_fp8
from primus_turbo.triton.quantization.quant_blk_gfx1250 import (
    MX,
    build_routing_blk,
    combine_rows,
    expert_order_lpt_fused,
    route_dw_partials,
    quant_dual_blk,
    quant_w_128x128,
    swiglu_bwd_dual_blk,
    swiglu_fwd_dual_blk,
)

__all__ = ["FP8BlockWeightCache", "fused_mega_moe_fp8blk", "refresh_fp8blk_weights"]

_FP8 = torch.float8_e4m3fn
_ALIGN = 256
# PRIMUS_TURBO_MOE_GG_DSV4=1: grouped fwd/dgrad GEMMs on the Gluon DSv4 kernel (gg_dsv4_gfx1250: upstream
# warp-pipeline mainloop, persistent grid + split-K tail; needs Triton main with gluon cdna5) instead of
# flygg_fp8blk_nt.  Default off (not recommended: tile 159.2 us vs flygg 162.1 us, gate failed).
# GG_DSV4_MODE=tile|persistent|split selects the variant (default tile; persistent/split measured slower).
_GG_DSV4 = os.environ.get("PRIMUS_TURBO_MOE_GG_DSV4", "0") == "1"
_GEMM = dict(num_buffers=3)
# moegemm track. PRIMUS_TURBO_MOE_FP8BLK_GG: grouped fwd/dgrad GEMM schedule (all bit-identical):
#   tile (default): flygg 256x256x128, 3-stage ring, one tile per workgroup, grid bound M/256 + G m-tiles;
#   aligned: same kernel, grid bound M/256 (pool offsets are 256-aligned) -> no G x N/256 dead workgroups;
#   k256: aligned + 256x256x256 k-tiles in a 2-stage ring (half the k-step barriers/waits per FLOP, same bytes in
#         flight) + B scale plane of one row per 16-row fragment (128x128-block weight scales) so it fits LDS;
#   persistent: flygg ggq (grid = #CUs, cross-tile TDM stream, C via LDS line scratch) -- measured slower;
#   k256p (moegemm2): persistent k256 (launch_ggk: grid = #CUs, one TDM stream across tiles, scale words straight from
#         global memory one k-step ahead, C through the LDS line scratch with non-TDM stores).
_GG_MODE = os.environ.get("PRIMUS_TURBO_MOE_FP8BLK_GG", "tile").strip().lower()
#   mx (mxgemm): mxgg -- 4-stage ring of 128-deep k-tiles, fragments of the next k-step read during the current one
#         (grouped_gemm_fp8blk_mx_flydsl_gfx1250.py); falls back to k256 when a shape constraint fails.
#   mxk (mxgemm): mxgk -- k256's TDM traffic (2 x 256-deep stages, whole-K scale planes) + a fragment pipeline that
#         crosses k-tiles (early barrier in each k-tile's second half); falls back to k256.
#   mxp (mxgemm phase 2): mxgp -- persistent mxgk (grid = #CUs, live tiles bounded on the device from the group offsets:
#         no dead workgroups, EP>1 worst-case pools OK; next tile's k-tiles / planes issued during the current tile's
#         last k-tiles; C via line-scratch VMEM stores). Needs K % 512 == 0, else mxgk / k256. Bit-identical.
assert _GG_MODE in ("tile", "aligned", "k256", "persistent", "k256p", "mx", "mxk", "mxp"), f"PRIMUS_TURBO_MOE_FP8BLK_GG={_GG_MODE!r}"
if _GG_MODE != "tile":
    assert _ALIGN % 256 == 0
    _GEMM.update(aligned=True, persistent={"persistent": 1, "k256p": 2}.get(_GG_MODE, 0))
if _GG_MODE in ("k256", "k256p", "mx", "mxk", "mxp"):
    _GEMM.update(tile=(256, 256, 256), num_buffers=2, bblk=1)
# mxgemm. PRIMUS_TURBO_MOE_FP8BLK_GG_BPF=1 (with GG=k256): fragment software pipeline inside each 256-deep k-tile (B
# fragments of the next 128-k step double-buffered, A one row group ahead). Bit-identical. Default 0.
if _GG_MODE == "k256" and os.environ.get("PRIMUS_TURBO_MOE_FP8BLK_GG_BPF", "0") == "1":
    _GEMM.update(bpf=1)
_VK = dict(num_buffers=2)
# PRIMUS_TURBO_MOE_FP8BLK_VKP_NB=3: vkp wgrad with a 3-stage ring (2 k-tiles in flight) for beta 0 (bit-identical;
# measured neutral to slightly slower: the direct C stores cost what the deeper ring saves). Default 2.
_VKP_NB = int(os.environ.get("PRIMUS_TURBO_MOE_FP8BLK_VKP_NB", "2"))
assert _VKP_NB in (2, 3)
# moegemm2. PRIMUS_TURBO_MOE_FP8BLK_VKP_SCHED2=1: vkp wgrad with scalar-loaded tile parameters, batched + carried E8M0
# scale words and A fragments read ahead of their WMMAs (bit-identical C and sumsq partials). Default 0.
# fuse track. PRIMUS_FUSE_MOE_BWD=1: d topk_weights in one launch from the unsummed SwiGLU-bwd partials (no zero fill,
# sum, index and casts) and the LPT expert order in one launch (bit-identical order); default off.
_FUSE_BWD = os.environ.get("PRIMUS_FUSE_MOE_BWD", "0") == "1"
_VKP_SCHED2 = int(os.environ.get("PRIMUS_TURBO_MOE_FP8BLK_VKP_SCHED2", "0"))
_CACHES = weakref.WeakSet()
# wgrad kernel: "vk" (one tile per workgroup) or "vkp" (persistent, experts longest-first, old-C prefetch for
# beta = 1; needs the compact 1-byte-per-128-token col scales from the quant kernels). PRIMUS_TURBO_MOE_FP8BLK_WGRAD.
# wgrad line: "vkq" = vkp's contract, tiles, LPT order and sumsq layout (main_grad and the sumsq partials
# bit-identical to vkp, beta 0 and 1) with the scale planes on VMEM instead of in the TDM queue (~-13% per call).
# PRIMUS_TURBO_MOE_FP8BLK_VKQ_NB=4 (beta 0 only; experimental, not faster): 4-stage ring with next-step fragment
# prefetch and C through a per-wave line scratch. Same window limit as vkp.
_VKP_LDS_KWIN = 5120  # longest expert window the vkp / vkq LDS arenas hold (scale planes grow with it)
_VKQ_NB = int(os.environ.get("PRIMUS_TURBO_MOE_FP8BLK_VKQ_NB", "2"))
assert _VKQ_NB in (2, 3, 4)


def _wgrad_kernel(T: int) -> str:
    mode = os.environ.get("PRIMUS_TURBO_MOE_FP8BLK_WGRAD", "vk").strip().lower()
    assert mode in ("vk", "vkp", "vkq"), f"PRIMUS_TURBO_MOE_FP8BLK_WGRAD={mode!r}"
    if mode in ("vkp", "vkq") and _vkp_kwin(T) > _VKP_LDS_KWIN:
        return "vk"
    return mode


def _vkp_kwin(T: int) -> int:
    # an expert window holds each token at most once, padded to 256 rows (and >= 256)
    return max(4096, -(-T // _ALIGN) * _ALIGN)


def _materialize(w: torch.Tensor):
    """Run the weight's materialize hook, if an optimizer registered one: an optimizer that keeps only the FP32
    master (and this cache) current between steps, without the BF16 weight (Primus PRIMUS_FP8BLK_ADAM_SKIP_BF16=1),
    rebuilds the BF16 weight before anything reads it."""
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
        self.s_pk = self.sT_pk = None  # gg_dsv4 packed MX scales (only when PRIMUS_TURBO_MOE_GG_DSV4=1)
        self.valid = False
        # Set by an optimizer that writes q / s / qT / sT itself with the bytes of refresh() after every
        # update of ``weight`` (Primus PRIMUS_FP8BLK_ADAM_MEGAMOE=1); refresh_fp8blk_weights() then skips it.
        self.adam_managed = False
        _CACHES.add(self)

    @torch.no_grad()
    def refresh(self):
        _materialize(self.weight)
        quant_w_128x128(self.weight.detach(), _FP8, transposed=True, out=(self.q, self.s, self.qT, self.sT))
        if _GG_DSV4:
            from primus_turbo.triton.grouped_gemm.gg_dsv4_gfx1250 import pack_rows

            E, N, K = self.q.shape
            if self.s_pk is None:
                self.s_pk = torch.empty((E * N // 128, K * 4), device=self.q.device, dtype=torch.uint8)
                self.sT_pk = torch.empty((E * K // 128, N * 4), device=self.q.device, dtype=torch.uint8)
            self.s_pk.copy_(pack_rows(self.s.view(E * N, K // MX)))
            self.sT_pk.copy_(pack_rows(self.sT.view(E * K, N // MX)))
        self.valid = True


@torch.no_grad()
def refresh_fp8blk_weights():
    """Re-quantize every live FP8-block expert weight not written by the optimizer itself (``adam_managed``).
    Call once after each optimizer step (outside graphs)."""
    verify = os.environ.get("PRIMUS_TURBO_MOE_FP8BLK_VERIFY", "0") == "1"
    n_refreshed = n_verified = 0
    for c in list(_CACHES):
        if not (c.adam_managed and c.valid):
            c.refresh()
            n_refreshed += 1
        elif verify:  # debug: the optimizer-written bytes must equal a fresh quant_w_128x128 (costs a sync)
            _materialize(c.weight)
            ref = quant_w_128x128(c.weight.detach(), _FP8, transposed=True)
            bad = [n for n, a, b in zip(("q", "s", "qT", "sT"), (c.q, c.s, c.qT, c.sT), ref)
                   if not torch.equal(a.view(torch.uint8), b.view(torch.uint8))]
            if bad:
                raise RuntimeError(f"PRIMUS_TURBO_MOE_FP8BLK_VERIFY: optimizer-written {bad} differ from "
                                   f"quant_w_128x128 of {tuple(c.weight.shape)}")
            c.verified = getattr(c, "verified", 0) + 1
            n_verified += 1
    if verify:
        print(f"[PRIMUS_TURBO_MOE_FP8BLK_VERIFY] caches={len(_CACHES)} refreshed={n_refreshed} "
              f"adam-written verified byte-equal={n_verified}", flush=True)


def _gg(a, a_s, c, transposed, poffs):
    """Grouped NT GEMM over the expert pool: a [P, K] fp8 x expert weights (c.q, or c.qT for the dgrads)."""
    w, ws = (c.qT, c.sT) if transposed else (c.q, c.s)
    if _GG_MODE in ("mx", "mxk", "mxp") and not _GG_DSV4:
        P, K = a.shape
        if _GG_MODE == "mx" and K % 128 == 0 and K >= 384 and w.shape[1] % 256 == 0 and P % 256 == 0:
            return mxgg_fp8blk_nt(a, w, a_s, ws, poffs)
        if _GG_MODE == "mxp" and K % 512 == 0 and (K & (K - 1)) == 0 and w.shape[1] % 256 == 0 and P % 256 == 0:
            return mxgp_fp8blk_nt(a, w, a_s, ws, poffs)
        if _GG_MODE in ("mxk", "mxp") and K >= 512 and (K & (K - 1)) == 0 and w.shape[1] % 256 == 0 and P % 256 == 0:
            return mxgk_fp8blk_nt(a, w, a_s, ws, poffs, sched=1)
    if not _GG_DSV4:
        kw = _GEMM
        if kw.get("tile", (0, 0, 128))[2] == 256 and a.shape[1] % 256:  # k256 needs K % 256 == 0
            kw = dict(kw, tile=(256, 256, 128), num_buffers=3, persistent=0)
        return flygg_fp8blk_nt(a, w, a_s, ws, poffs, **kw)
    from primus_turbo.triton.grouped_gemm.gg_dsv4_gfx1250 import gg_dsv4, pack_rows, tile_table

    E, N, K = w.shape
    P = a.shape[0]
    mte, nmt = tile_table(poffs, P)  # device-only (searchsorted), capturable
    return gg_dsv4(a, w.view(E * N, K), pack_rows(a_s), c.sT_pk if transposed else c.s_pk, mte, nmt, N)


def _wgrad_sumsq_buf(w: torch.Tensor, mg: torch.Tensor):
    """PRIMUS_TURBO_MOE_WGRAD_SUMSQ=1: persistent fp32 partial buffer the vkp wgrad fills with the sum of squares of
    ``w.main_grad`` (its final bf16 values) in its epilogue; exposed as ``w._primus_main_grad_sumsq`` (a tensor whose
    sum is sum(main_grad^2)) and flagged ``w._primus_main_grad_sumsq_fresh`` for the grad-norm pass, which then skips
    re-reading this weight's grad. None when off. The epilogue is on the nbuf 2 path: with
    PRIMUS_TURBO_MOE_FP8BLK_VKP_NB=3 the beta 0 wgrads of the fold run nbuf 2 (bit-identical C)."""
    if os.environ.get("PRIMUS_TURBO_MOE_WGRAD_SUMSQ", "0") != "1":
        return None
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


class _FP8BlockMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, topk_idx, topk_w, w1, w2, c1, c2, activation, accumulate, add=None):
        T, H = x.shape
        K = topk_idx.shape[1]
        E, I2, _ = w1.shape
        I = I2 // 2
        P = T * K + E * _ALIGN
        if not (c1.valid and c2.valid):  # first use only (eager, before any capture)
            c1.refresh(); c2.refresh()
        rowmap, route_w, inv, poffs, preal = build_routing_blk(topk_idx, topk_w.float(), E, P, align=_ALIGN)
        wk = _wgrad_kernel(T)
        ct = wk in ("vkp", "vkq")  # col (token-direction) scales compact for the persistent wgrads
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
        vkp = ctx.wk in ("vkp", "vkq")  # vkq: the vkp family (compact col scales, LPT, sumsq fold)
        dq, ds, dT, dTs = quant_dual_blk(dy.contiguous().to(torch.bfloat16), P, rowmap, preal=preal, compact_t=vkp)
        dact = _gg(dq, ds, c2, True, poffs)
        gq, gs, gT, gTs, dwr = swiglu_bwd_dual_blk(dact, l1, route_w, ctx.activation, preal=preal, compact_t=vkp,
                                                   raw_dw=_FUSE_BWD)
        # experts longest-first (one tiny argsort shared by both wgrads); PRIMUS_TURBO_MOE_FP8BLK_WGRAD_LPT=0: index order
        lpt = os.environ.get("PRIMUS_TURBO_MOE_FP8BLK_WGRAD_LPT", "1") == "1"
        order_lpt = expert_order_lpt_fused if _FUSE_BWD else expert_order_lpt
        jobs = ((w2, dT, dTs, aT, aTs, H, I), (w1, gT, gTs, xT, xTs, 2 * I, H))
        mgs = [_setup_accum(w, ctx.accumulate) for w, *_ in jobs]
        deferred = _overlap.enabled("MOE_WGRAD") and all(mg is not None for mg in mgs)
        if deferred:
            # PRIMUS_OVERLAP_MOE_WGRAD=1: the LPT order and both expert wgrads (they only feed the optimizer) run on
            # the side stream next to the fc1 dgrad / combine / router backward; same kernels and inputs -> same bits
            # (main_grad and the sumsq partials are per weight; nothing reads them before the join).
            betas = [0.0 if main_grad_first_write(w) else 1.0 for w, *_ in jobs]
            sqs = [_wgrad_sumsq_buf(w, mg) if vkp else None for (w, *_), mg in zip(jobs, mgs)]

            def side_wgrads():
                perm_s = order_lpt(poffs) if (vkp and lpt) else None
                for (w, lhs, lhs_s, rhs, rhs_s, n, k), mg, beta, sq in zip(jobs, mgs, betas, sqs):
                    if ctx.wk == "vkq":
                        flyvkqblk_fp8(lhs, rhs, lhs_s, rhs_s, poffs, n, k, mg, beta=beta, perm=perm_s,
                                      kwin_max=_vkp_kwin(T), sumsq_out=sq, nbuf=_VKQ_NB if _ALIGN % 256 == 0 else 2)
                    elif vkp:
                        flyvkpblk_fp8(lhs, rhs, lhs_s, rhs_s, poffs, n, k, mg, beta=beta, perm=perm_s,
                                      kwin_max=_vkp_kwin(T), nbuf=_VKP_NB, sumsq_out=sq, sched2=_VKP_SCHED2)
                    else:
                        flyvkblk_fp8(lhs, rhs, lhs_s, rhs_s, poffs, n, k, mg, beta=beta, **_VK)
                return perm_s

            _overlap.defer(side_wgrads, dT, dTs, aT, aTs, gT, gTs, xT, xTs, poffs, *mgs, *sqs)
            for (w, *_), sq in zip(jobs, sqs):
                w._primus_main_grad_sumsq_fresh = sq is not None
                w.grad_added_to_main_grad = True
            jobs = ()  # nothing left for the serial loop below
        perm = order_lpt(poffs) if (vkp and lpt and not deferred) else None

        def wgrad(lhs, rhs, lhs_s, rhs_s, n, k, out, beta, sq=None):
            if ctx.wk == "vkq":
                # nbuf 4 needs every expert window a multiple of 256 tokens: the pool alignment (_ALIGN)
                flyvkqblk_fp8(lhs, rhs, lhs_s, rhs_s, poffs, n, k, out, beta=beta, perm=perm, kwin_max=_vkp_kwin(T),
                              sumsq_out=sq, nbuf=_VKQ_NB if _ALIGN % 256 == 0 else 2)
            elif vkp:
                flyvkpblk_fp8(lhs, rhs, lhs_s, rhs_s, poffs, n, k, out, beta=beta, perm=perm, kwin_max=_vkp_kwin(T),
                              nbuf=_VKP_NB, sumsq_out=sq, sched2=_VKP_SCHED2)
            else:
                flyvkblk_fp8(lhs, rhs, lhs_s, rhs_s, poffs, n, k, out, beta=beta, **_VK)

        grads = []
        for (w, lhs, lhs_s, rhs, rhs_s, n, k), mg in zip(jobs, mgs):
            if mg is not None:
                # beta 0 on the step's first write into main_grad (framework skipped its clear), else accumulate
                # every call writes all of main_grad, so with the fold the last call of the step leaves the sum of
                # squares of the final grad (multi-microbatch accumulation included)
                sq = _wgrad_sumsq_buf(w, mg) if vkp else None
                wgrad(lhs, rhs, lhs_s, rhs_s, n, k, mg, 0.0 if main_grad_first_write(w) else 1.0, sq)
                w._primus_main_grad_sumsq_fresh = sq is not None
                w.grad_added_to_main_grad = True
                grads.append(_get_dummy_wgrad(w.shape, w.dtype))  # dummy, never read
            else:
                g = torch.empty((E, n, k), device=w.device, dtype=torch.bfloat16)
                wgrad(lhs, rhs, lhs_s, rhs_s, n, k, g, 0.0)
                grads.append(g.to(w.dtype))
        dW2, dW1 = grads if not deferred else (_get_dummy_wgrad(w.shape, w.dtype) for w in (w2, w1))
        dxp = _gg(gq, gs, c1, True, poffs)
        dx = combine_rows(dxp, inv, T, K)
        if _FUSE_BWD:
            dtw = route_dw_partials(dwr, inv, ctx.topk_dtype).view(T, K)
        else:
            dtw = dwr[inv.long()].view(T, K).to(ctx.topk_dtype)
        return dx, None, dtw, dW1, dW2, None, None, None, None, (dy if ctx.has_add else None)


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
) -> torch.Tensor:
    """y [T, H] = sum_k topk_w[t, k] * w2[e] act(w1[e] x[t]) with e = topk_idx[t, k], all GEMMs FP8 128-block.

    x [T, H] bf16, w1 [E, 2I, H], w2 [E, H, I] (H, I multiples of 256). ``cache1``/``cache2`` hold the quantized
    weights (``FP8BlockWeightCache(w)``, refreshed by ``refresh_fp8blk_weights()`` after each optimizer step).
    ``accumulate_wgrad``: add dW straight into ``w.main_grad`` (bf16, Megatron DDP) and set
    ``grad_added_to_main_grad``; the returned weight gradient is then a dummy.
    ``add`` [T, H] bf16 (e.g. the shared expert output): returned y is ``y + add`` with the add fused into the
    combine kernel (bit-identical to the separate bf16 add; its gradient is dy)."""
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    assert w1.shape[0] == w2.shape[0] and w1.shape[2] == x.shape[1] and w2.shape[1] == x.shape[1]
    assert w1.shape[1] % 256 == 0 and x.shape[1] % 256 == 0 and w2.shape[2] * 2 == w1.shape[1]
    if add is not None:
        add = add.reshape(x.shape).contiguous()
    return _FP8BlockMoE.apply(x.contiguous(), topk_idx.to(torch.int64).contiguous(), topk_weights, w1, w2, cache1,
                              cache2, activation, accumulate_wgrad, add)
