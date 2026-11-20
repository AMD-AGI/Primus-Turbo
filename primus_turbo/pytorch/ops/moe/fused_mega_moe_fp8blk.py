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

from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, main_grad_first_write
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_nt_flydsl_gfx1250 import flygg_fp8blk_nt
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_vk_flydsl_gfx1250 import flyvkblk_fp8
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_vkp_flydsl_gfx1250 import expert_order_lpt, flyvkpblk_fp8
from primus_turbo.triton.quantization.quant_blk_gfx1250 import (
    MX,
    build_routing_blk,
    combine_rows,
    quant_dual_blk,
    quant_w_128x128,
    swiglu_bwd_dual_blk,
    swiglu_fwd_dual_blk,
)

__all__ = ["FP8BlockWeightCache", "fused_mega_moe_fp8blk", "refresh_fp8blk_weights"]

_FP8 = torch.float8_e4m3fn
_ALIGN = 256
_GEMM = dict(num_buffers=3)
_VK = dict(num_buffers=2)
_CACHES = weakref.WeakSet()
# wgrad kernel: "vk" (one tile per workgroup) or "vkp" (persistent, experts longest-first, old-C prefetch for
# beta = 1; needs the compact 1-byte-per-128-token col scales from the quant kernels). PRIMUS_TURBO_MOE_FP8BLK_WGRAD.
_VKP_LDS_KWIN = 5120  # longest expert window the vkp LDS arena holds (scale planes grow with it)


def _wgrad_kernel(T: int) -> str:
    mode = os.environ.get("PRIMUS_TURBO_MOE_FP8BLK_WGRAD", "vk").strip().lower()
    assert mode in ("vk", "vkp"), f"PRIMUS_TURBO_MOE_FP8BLK_WGRAD={mode!r}"
    if mode == "vkp" and _vkp_kwin(T) > _VKP_LDS_KWIN:
        return "vk"
    return mode


def _vkp_kwin(T: int) -> int:
    # an expert window holds each token at most once, padded to 256 rows (and >= 256)
    return max(4096, -(-T // _ALIGN) * _ALIGN)


class FP8BlockWeightCache:
    """Fixed buffers holding one expert weight [E, N, K] in 128x128-block FP8 (+ the transposed copy)."""

    def __init__(self, w: torch.Tensor):
        E, N, K = w.shape
        dev = w.device
        self.weight = w
        self.q = torch.empty((E, N, K), device=dev, dtype=_FP8)
        self.s = torch.empty((E, N, K // MX), device=dev, dtype=torch.uint8)
        self.qT = torch.empty((E, K, N), device=dev, dtype=_FP8)
        self.sT = torch.empty((E, K, N // MX), device=dev, dtype=torch.uint8)
        self.valid = False
        _CACHES.add(self)

    @torch.no_grad()
    def refresh(self):
        quant_w_128x128(self.weight.detach(), _FP8, transposed=True, out=(self.q, self.s, self.qT, self.sT))
        self.valid = True


@torch.no_grad()
def refresh_fp8blk_weights():
    """Re-quantize every live FP8-block expert weight. Call once after each optimizer step (outside graphs)."""
    for c in list(_CACHES):
        c.refresh()


def _setup_accum(w: torch.Tensor, accumulate: bool):
    mg = getattr(w, "main_grad", None) if accumulate else None
    if mg is not None and mg.dtype == torch.bfloat16 and mg.shape == w.shape and mg.is_contiguous():
        return mg
    return None


class _FP8BlockMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, topk_idx, topk_w, w1, w2, c1, c2, activation, accumulate):
        T, H = x.shape
        K = topk_idx.shape[1]
        E, I2, _ = w1.shape
        I = I2 // 2
        P = T * K + E * _ALIGN
        if not (c1.valid and c2.valid):  # first use only (eager, before any capture)
            c1.refresh(); c2.refresh()
        rowmap, route_w, inv, poffs, preal = build_routing_blk(topk_idx, topk_w.float(), E, P, align=_ALIGN)
        wk = _wgrad_kernel(T)
        ct = wk == "vkp"  # col (token-direction) scales compact for the persistent wgrad
        xq, xs, xT, xTs = quant_dual_blk(x, P, rowmap, preal=preal, compact_t=ct)
        l1 = flygg_fp8blk_nt(xq, c1.q, xs, c1.s, poffs, **_GEMM)
        aq, as_, aT, aTs = swiglu_fwd_dual_blk(l1, route_w, activation, preal=preal, compact_t=ct)
        l2 = flygg_fp8blk_nt(aq, c2.q, as_, c2.s, poffs, **_GEMM)
        y = combine_rows(l2, inv, T, K)
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
        vkp = ctx.wk == "vkp"
        dq, ds, dT, dTs = quant_dual_blk(dy.contiguous().to(torch.bfloat16), P, rowmap, preal=preal, compact_t=vkp)
        dact = flygg_fp8blk_nt(dq, c2.qT, ds, c2.sT, poffs, **_GEMM)
        gq, gs, gT, gTs, dwr = swiglu_bwd_dual_blk(dact, l1, route_w, ctx.activation, preal=preal, compact_t=vkp)
        # experts longest-first (one tiny argsort shared by both wgrads); PRIMUS_TURBO_MOE_FP8BLK_WGRAD_LPT=0: index order
        lpt = os.environ.get("PRIMUS_TURBO_MOE_FP8BLK_WGRAD_LPT", "1") == "1"
        perm = expert_order_lpt(poffs) if (vkp and lpt) else None

        def wgrad(lhs, rhs, lhs_s, rhs_s, n, k, out, beta):
            if vkp:
                flyvkpblk_fp8(lhs, rhs, lhs_s, rhs_s, poffs, n, k, out, beta=beta, perm=perm, kwin_max=_vkp_kwin(T))
            else:
                flyvkblk_fp8(lhs, rhs, lhs_s, rhs_s, poffs, n, k, out, beta=beta, **_VK)

        grads = []
        for w, lhs, lhs_s, rhs, rhs_s, n, k in ((w2, dT, dTs, aT, aTs, H, I), (w1, gT, gTs, xT, xTs, 2 * I, H)):
            mg = _setup_accum(w, ctx.accumulate)
            if mg is not None:
                # beta 0 on the step's first write into main_grad (framework skipped its clear), else accumulate
                wgrad(lhs, rhs, lhs_s, rhs_s, n, k, mg, 0.0 if main_grad_first_write(w) else 1.0)
                w.grad_added_to_main_grad = True
                grads.append(_get_dummy_wgrad(w.shape, w.dtype))  # dummy, never read
            else:
                g = torch.empty((E, n, k), device=w.device, dtype=torch.bfloat16)
                wgrad(lhs, rhs, lhs_s, rhs_s, n, k, g, 0.0)
                grads.append(g.to(w.dtype))
        dW2, dW1 = grads
        dxp = flygg_fp8blk_nt(gq, c1.qT, gs, c1.sT, poffs, **_GEMM)
        dx = combine_rows(dxp, inv, T, K)
        dtw = dwr[inv.long()].view(T, K).to(ctx.topk_dtype)
        return dx, None, dtw, dW1, dW2, None, None, None, None


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
) -> torch.Tensor:
    """y [T, H] = sum_k topk_w[t, k] * w2[e] act(w1[e] x[t]) with e = topk_idx[t, k], all GEMMs FP8 128-block.

    x [T, H] bf16, w1 [E, 2I, H], w2 [E, H, I] (H, I multiples of 256). ``cache1``/``cache2`` hold the quantized
    weights (``FP8BlockWeightCache(w)``, refreshed by ``refresh_fp8blk_weights()`` after each optimizer step).
    ``accumulate_wgrad``: add dW straight into ``w.main_grad`` (bf16, Megatron DDP) and set
    ``grad_added_to_main_grad``; the returned weight gradient is then a dummy."""
    assert x.dim() == 2 and x.dtype == torch.bfloat16
    assert w1.shape[0] == w2.shape[0] and w1.shape[2] == x.shape[1] and w2.shape[1] == x.shape[1]
    assert w1.shape[1] % 256 == 0 and x.shape[1] % 256 == 0 and w2.shape[2] * 2 == w1.shape[1]
    return _FP8BlockMoE.apply(x.contiguous(), topk_idx.to(torch.int64).contiguous(), topk_weights, w1, w2, cache1,
                              cache2, activation, accumulate_wgrad)
