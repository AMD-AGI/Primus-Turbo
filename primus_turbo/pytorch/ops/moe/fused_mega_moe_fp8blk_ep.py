###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Expert-parallel (EP>1) MegaMoE experts in FP8 with the DeepSeek-V3/V4 recipe on gfx1250: the EP>1 counterpart of
``fused_mega_moe_fp8blk`` (E4M3, UE8M0 scales per 1x128 activation tile / 128x128 weight block / 128x1 wgrad-operand
tile), with FP8 dispatch and BF16 combine.

Two autograd stages, the split of the bf16 / mxfp8 MegaMoE (``fused_mega_moe{,_fp8}_stage1/2``):
  stage1 (owns w1): source-side 1x128 quant of x -> dispatch (one FP8 row + H/128 scale bytes per (token, destination
         rank), plus the token's local routes) -> local routing tables -> pool operands -> grouped fc1.
         Outputs l1 [P, 2I] and the pool routing weights route_w [P]; the gradient of route_w (d act_w . act) is how
         d topk_weights travels from stage2.backward back to stage1.backward.
  stage2 (owns w2): SwiGLU x route_w + quant -> grouped fc2 -> each route's BF16 row back to its token's rank ->
         y[t] = sum_k row(t, k) (fp32, k order; the EP=1 combine_rows kernel).
  stage2.backward: source-side 1x128 quant of dy -> dispatch (same slots as x) -> dact (fc2 dgrad), SwiGLU^T, dW2.
  stage1.backward: dW1, fc1 dgrad, each route's BF16 dx row and fp32 d route_w back -> dx (combine_rows), d topk_w.
Stage-to-stage state (tables, saved operands, the quantized grad_l1) rides a ``_EPState`` object (as in mxfp8).

Numerics vs EP=1 ``fused_mega_moe_fp8blk`` on the same tokens (the EP group's tokens in rank order = EP=1 rows):
  * every expert's pool segment holds the same routes in the same (global token) order at a 256-aligned start, so
    every 1x128 / 128x1 tile, GEMM row and wgrad k-sequence is the same: y, dx, d topk_w are bit-identical.
  * ``dispatch="fp8"`` (default, the recipe): the 128x1 token-direction operands of the wgrads (x^T for dW1, dy^T for
    dW2) are re-quantized on the expert's rank from the received FP8 rows (dequantize -> 128x1 quant, as DeepSeek-V3
    describes), while EP=1 quantizes them from the BF16 values: dW1 / dW2 differ by that double rounding.
    The 128x1 tiles span tokens of several source ranks, so the source cannot build them; sending them would need
    the BF16 rows (2H bytes per row instead of H + H/128).
  * ``dispatch="bf16"``: BF16 rows are dispatched (2x the dispatch bytes) and quantized both ways on the expert's
    rank with the EP=1 kernel: everything bit-identical to EP=1.

Fixed shapes, no host syncs (CUDA-graph capturable, given a capturable all-to-all): with R ranks of T tokens each,
  dispatch buffers [R, T, H] (a token goes at most once to a rank), combine buffers [R, T*min(K, E_l), H], expert pool
  P = R*T*min(K, E_l) + E_l*256 rows (rounded to 256): the true worst case (all tokens to one rank), no token drop.
  Requires: the same T on every rank, distinct experts per token, E_l a power of two, H >= 1024 (the fp8_block grouped
  GEMMs return zeros / NaN at H <= 512), H % 1024 == 0 (the validated shapes), I % 256 == 0.
Communication goes through an ``EPComm``: ``DistEPComm(group)`` (torch.distributed all_to_all_single with equal
  per-peer blocks: sends the whole fixed-capacity buffer) or ``EmulatedEPComm(R)``, which drives all R ranks of the group
  in one process (lists of per-rank tensors) with the same block exchange done as device copies -- the single-GPU
  emulation used by the tests.
"""

import os
from typing import List, Optional, Sequence

import torch

from primus_turbo.pytorch.core import overlap as _overlap
from primus_turbo.pytorch.ops.moe import fused_mega_moe_fp8blk as _blk
from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk import FP8BlockWeightCache
from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, main_grad_first_write
from primus_turbo.triton.quantization.quantization_fp8blk_ep_kernel import (
    build_routing_blk_masked,
    build_routing_blk_meta,
    combine_ranks,
    gather_dual_fp8,
    pack_pool_rows,
    presum_routes,
    quant_rows_send,
    scatter_pool_ext,
    src_layout_ep,
)
from primus_turbo.triton.quantization.quantization_fp8blk_moe_kernel import (
    combine_rows,
    expert_order_lpt_fused,
    quant_dual_blk,
    swiglu_bwd_dual_blk,
    swiglu_fwd_dual_blk,
)

__all__ = [
    "EPComm",
    "DistEPComm",
    "EmulatedEPComm",
    "FP8BlockWeightCache",
    "fused_mega_moe_fp8blk_ep",
    "fused_mega_moe_fp8blk_ep_multi",
    "fused_mega_moe_fp8blk_ep_stage1",
    "fused_mega_moe_fp8blk_ep_stage2",
]

_FP8 = torch.float8_e4m3fn
_MIN_H = 1024


# ---------------------------------------------------------------------------------------------------------------------
# communication
# ---------------------------------------------------------------------------------------------------------------------
class EPComm:
    """Fixed-shape all-to-all inside an EP group of ``size`` ranks, for the ranks this process drives (``local_ranks``).

    ``all_to_all(sends)``: ``sends[i]`` [R, ...] from local rank i holds block d for rank d; returns ``recvs[i]``
    [R, ...] holding block s from rank s."""

    size: int
    local_ranks: Sequence[int]

    def all_to_all(self, sends: List[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError


class DistEPComm(EPComm):
    """torch.distributed: one rank per process, ``all_to_all_single`` with equal per-peer blocks (no split lists, so no
    host sync; capturable where the backend supports graph capture). Sends whole fixed-capacity blocks: a count-based
    transport (per-peer valid prefix only) is the real-hardware follow-up."""

    def __init__(self, group):
        import torch.distributed as dist

        self._dist = dist
        self.group = group
        self.size = group.size()
        self.local_ranks = [group.rank()]

    def all_to_all(self, sends):
        (s,) = sends
        s = s.contiguous()
        wire = s.view(torch.uint8) if s.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else s
        out = torch.empty_like(wire)
        self._dist.all_to_all_single(out, wire, group=self.group)
        return [out.view(s.dtype)]


class EmulatedEPComm(EPComm):
    """Single-process emulation of an EP group: this process drives all ``size`` ranks; the exchange copies block d of
    rank s into block s of rank d (the all-to-all's data movement, as device copies)."""

    def __init__(self, size: int):
        self.size = size
        self.local_ranks = list(range(size))

    def all_to_all(self, sends):
        R = self.size
        assert len(sends) == R and all(s.shape[0] == R for s in sends)
        recvs = [torch.empty_like(sends[0]) for _ in range(R)]
        for d in range(R):
            for s in range(R):
                recvs[d][s].copy_(sends[s][d])
        return recvs


# ---------------------------------------------------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------------------------------------------------
class _Rank:
    """Per driven rank: source-side layout + expert-side tables and saved operands."""

    __slots__ = (
        "T",
        "slot",
        "cinv",
        "rowmap",
        "rw",
        "inv",
        "poffs",
        "preal",
        "p2s",
        "xT",
        "xTs",
        "l1",
        "aT",
        "aTs",
        "g",
        "perm",
        "topk_dtype",
    )

    def __init__(self):
        for a in self.__slots__:
            setattr(self, a, None)


class _EPState:
    def __init__(self, comm: EPComm, E_l: int, K: int, H: int, I: int, T: int, dispatch: str):
        R = comm.size
        self.comm, self.R, self.E_l, self.K, self.H, self.I, self.T = comm, R, E_l, K, H, I, T
        self.dispatch = dispatch
        self.cap = T  # dispatch slots per (source, destination): a token goes at most once to a rank
        self.kl = min(K, E_l)  # routes of one token to one rank
        self.ccap = T * self.kl  # combine slots per (expert rank, source)
        self.N = R * self.cap  # received token rows per rank
        align = _blk._ALIGN
        self.P = -(-(self.N * self.kl + E_l * align) // 256) * 256
        # Longest expert window the wgrad is built for. The true bound is N tokens, which is past what the persistent
        # vkq / vkw wgrad holds in LDS; PRIMUS_TURBO_EP_WGRAD_WINDOW (default 5120, 0 = N) caps it so they run, and
        # every wgrad asserts on the device that the real windows fit (no host sync; aborts instead of a wrong result,
        # as the vk path does past its own cap).
        cap_win = int(os.environ.get("PRIMUS_TURBO_EP_WGRAD_WINDOW", "5120"))
        self.wwin = min(self.N, cap_win) if cap_win > 0 else self.N
        self.wk = _blk._wgrad_kernel(self.wwin)
        self.ct = self.wk == "vkq"  # compact col scales + LPT order + sumsq fold: the persistent wgrad
        self.ranks = [_Rank() for _ in comm.local_ranks]
        self.activation = None
        self.acc1 = self.acc2 = False


def _src_layout(st: _EPState, rk: _Rank, idx: torch.Tensor, tw: torch.Tensor):
    """Source rank: dispatch slots, the routes' combine slots and the per-slot route metadata [R, cap, 3K]."""
    rk.slot, rk.cinv, meta, cnt = src_layout_ep(idx, tw, st.R, st.E_l, st.cap, st.ccap)
    return meta, cnt.view(st.R, 1)


def _recv_tables(st: _EPState, rk: _Rank, meta, cnt):
    """Expert rank: pool routing tables over the local experts from the received route metadata and per-source
    counts, with the pool row -> combine slot map (p2s). ``meta`` / ``cnt``: (flat int32 buffer, element offset,
    per-source stride) of the [cap, 3K] metadata blocks / the counts (slots past a source's count are dead; the
    kernels mask them, no materialized columns)."""
    (mb, moff, mstride), (cb, coff, cstride) = meta, cnt
    rk.rowmap, rk.rw, rk.inv, rk.poffs, rk.preal, rk.p2s = build_routing_blk_meta(
        mb, moff, mstride, cb, coff, cstride, st.N, st.K, st.E_l, st.P, st.cap, st.ccap, align=_blk._ALIGN
    )


def _flat_src(t: torch.Tensor):
    """(flat int32 view, offset 0, per-source stride) of a contiguous [R, ...] int32 tensor."""
    t = t.contiguous()
    return t.view(-1), 0, t[0].numel()


def _merge_small() -> bool:
    return os.environ.get("PRIMUS_TURBO_EP_MERGE_SMALL", "1") != "0"


def _a2a_small(comm: EPComm, ps, pm, pc):
    """The dispatch's three small exchanges (E8M0 scales [R, cap, H/128] u8, route metadata [R, cap, 3K] i32, counts
    [R, 1] i32) as ONE all-to-all of their per-peer bytes: one launch instead of three, and it avoids a slow message
    size (the metadata alone, 1152 KiB per rank at the DSv4 shapes, takes ~3x a 2 MiB message on MI455X RCCL)."""
    sends, parts = [], None
    for s, m, c in zip(ps, pm, pc):
        R = s.shape[0]
        parts = [t.reshape(R, -1).view(torch.uint8) for t in (s, m, c)]
        widths = [p.shape[1] for p in parts]
        blk = -(-sum(widths) // 16) * 16
        buf = torch.empty((R, blk), dtype=torch.uint8, device=s.device)
        off = 0
        for p, w in zip(parts, widths):
            buf[:, off : off + w].copy_(p)
            off += w
        sends.append(buf)
    recvs = comm.all_to_all(sends)
    rs, rm, rc = [], [], []
    for r, s, m, c in zip(recvs, ps, pm, pc):
        ws = s[0].numel() * s.element_size()
        wm = m[0].numel() * m.element_size()
        rs.append(r[:, :ws].contiguous().view(s.dtype).view(s.shape))
        # metadata / counts are read in place by the routing kernels: (int32 buffer, offset, per-source stride)
        b32 = r.view(torch.int32).view(-1)
        stride = r.shape[1] // 4
        rm.append((b32, ws // 4, stride))
        rc.append((b32, (ws + wm) // 4, stride))
    return rs, rm, rc


def _dispatch_fwd(st: _EPState, xs, idxs, tws):
    """x -> pool operands + routing tables on every driven rank (stage1 forward minus fc1)."""
    H, cap, N, P = st.H, st.cap, st.N, st.P
    fp8 = st.dispatch == "fp8"
    pq, ps, pm, pc = [], [], [], []
    for rk, x, idx, tw in zip(st.ranks, xs, idxs, tws):
        meta, cnt = _src_layout(st, rk, idx, tw)
        q, s = quant_rows_send(x, rk.slot, cap, quant=fp8)
        pq.append(q), ps.append(s), pm.append(meta), pc.append(cnt)
    comm = st.comm
    rq = comm.all_to_all(pq)
    if fp8 and _merge_small():
        rs, rm, rc = _a2a_small(comm, ps, pm, pc)
    else:
        rs = comm.all_to_all(ps) if fp8 else [None] * len(rq)
        rm = [_flat_src(m) for m in comm.all_to_all(pm)]
        rc = [_flat_src(c) for c in comm.all_to_all(pc)]
    del pq, ps, pm, pc
    out = []
    for rk, q, s, meta, cnt in zip(st.ranks, rq, rs, rm, rc):
        _recv_tables(st, rk, meta, cnt)
        if fp8:
            xq, xs_, rk.xT, rk.xTs = gather_dual_fp8(
                q.view(N, H), s.view(N, H // 128), P, rk.rowmap, preal=rk.preal, compact_t=st.ct
            )
        else:
            xq, xs_, rk.xT, rk.xTs = quant_dual_blk(
                q.view(N, H), P, rk.rowmap, preal=rk.preal, compact_t=st.ct
            )
        out.append((xq, xs_))
    return out


def _dispatch_dy(st: _EPState, dys):
    """dy -> pool operands (1x128 for the fc2 dgrad, 128x1 for dW2) through the forward's dispatch slots."""
    H, N, P, cap = st.H, st.N, st.P, st.cap
    fp8 = st.dispatch == "fp8"
    pq, ps = [], []
    for rk, dy in zip(st.ranks, dys):
        q, s = quant_rows_send(dy.contiguous().to(torch.bfloat16), rk.slot, cap, quant=fp8)
        pq.append(q), ps.append(s)
    rq = st.comm.all_to_all(pq)
    rs = st.comm.all_to_all(ps) if fp8 else [None] * len(rq)
    del pq, ps
    out = []
    for rk, q, s in zip(st.ranks, rq, rs):
        if fp8:
            out.append(
                gather_dual_fp8(
                    q.view(N, H), s.view(N, H // 128), P, rk.rowmap, preal=rk.preal, compact_t=st.ct
                )
            )
        else:
            out.append(quant_dual_blk(q.view(N, H), P, rk.rowmap, preal=rk.preal, compact_t=st.ct))
    return out


def _presum() -> bool:
    return os.environ.get("PRIMUS_TURBO_EP_PRESUM", "1") != "0"


def _combine_presum(st: _EPState, pool_rows, extra=None):
    """Combine with the routes of a token summed on the expert rank first: one bf16 row per (token, expert rank)
    ([R, cap, H], the dispatch slots) instead of one per route ([R, ccap, H], ccap = T * min(K, E_l)), then the
    per-rank partials summed on the token's rank in rank order. Not bit-identical to EP=1 (sum order and one more
    bf16 rounding). ``extra`` (d route_w) still travels per route ([R, ccap] fp32)."""
    R, H, ccap, K, cap, N = st.R, st.H, st.ccap, st.K, st.cap, st.N
    sends, exts = [], []
    for i, (rk, rows) in enumerate(zip(st.ranks, pool_rows)):
        if extra is None:
            sends.append(presum_routes(rows, rk.inv, N, K).view(R, cap, H))
        else:  # d route_w scattered to its combine slot inside the same launch
            o, eo = presum_routes(rows, rk.inv, N, K, ext=extra[i], p2s=rk.p2s, nslots=R * ccap)
            sends.append(o.view(R, cap, H))
            exts.append(eo.view(R, ccap))
    recvs = st.comm.all_to_all(sends)
    del sends
    if extra is not None:
        rws = st.comm.all_to_all(exts)
    outs = []
    for i, (rk, rv) in enumerate(zip(st.ranks, recvs)):
        if extra is None:
            outs.append(combine_ranks(rv.view(R * cap, H), rk.slot, rk.T, R, cap))
        else:  # d topk_w gathered by combine slot inside the same launch
            outs.append(combine_ranks(rv.view(R * cap, H), rk.slot, rk.T, R, cap, rws=rws[i].view(-1), cinv=rk.cinv, K=K))
    return outs


def _combine(st: _EPState, pool_rows, extra=None):
    """Each live route's pool row (bf16 [P, H]) back to its token's rank, summed there in k order (combine_rows).
    ``extra``: per-rank fp32 [P] (d route_w) sent along per route -> [T, K]."""
    if _presum():
        return _combine_presum(st, pool_rows, extra)
    R, H, ccap, K = st.R, st.H, st.ccap, st.K
    sends, exts = [], []
    for i, (rk, rows) in enumerate(zip(st.ranks, pool_rows)):
        o, eo = pack_pool_rows(rows, rk.p2s, rk.preal, R * ccap, None if extra is None else extra[i])
        sends.append(o.view(R, ccap, H))
        exts.append(eo)
    recvs = st.comm.all_to_all(sends)
    del sends
    if extra is not None:
        rws = st.comm.all_to_all([e.view(R, ccap) for e in exts])
    outs = []
    for i, (rk, rv) in enumerate(zip(st.ranks, recvs)):
        o = combine_rows(rv.view(R * ccap, H), rk.cinv, rk.T, K)
        if extra is not None:
            o = (o, rws[i].view(-1)[rk.cinv.long()].view(rk.T, K))
        outs.append(o)
    return outs


def _wgrad(st: _EPState, rk: _Rank, w, lhs, lhs_s, rhs, rhs_s, n, k, accumulate):
    """fused_mega_moe_fp8blk's wgrad step (main_grad accumulation, sumsq fold, dummy grad) for one weight."""

    def run(out, beta, sq=None):
        if st.wwin < st.N:
            torch._assert_async((rk.poffs[1:] - rk.poffs[:-1]).max() <= _blk._vkp_kwin(st.wwin))
        _blk._run_wgrad(st.wk, lhs, rhs, lhs_s, rhs_s, rk.poffs, n, k, out, beta, rk.perm, st.wwin, sq)

    mg = _blk._setup_accum(w, accumulate)
    if mg is not None:
        sq = _blk._wgrad_sumsq_buf(w, mg) if st.wk == "vkq" else None
        beta = 0.0 if main_grad_first_write(w) else 1.0
        if _overlap.enabled("MOE_WGRAD"):
            # the expert wgrad only feeds the optimizer: run it on the side stream next to the dgrad chain and the
            # EP all-to-alls (same kernel and inputs -> same main_grad bits; joined after the micro-batch backward)
            _overlap.defer(lambda: run(mg, beta, sq), lhs, lhs_s, rhs, rhs_s, rk.poffs, rk.perm, mg, sq)
        else:
            run(mg, beta, sq)
        w._primus_main_grad_sumsq_fresh = sq is not None
        w.grad_added_to_main_grad = True
        return _get_dummy_wgrad(w.shape, w.dtype)
    g = torch.empty((w.shape[0], n, k), device=w.device, dtype=torch.bfloat16)
    run(g, 0.0)
    return g.to(w.dtype)


# ---------------------------------------------------------------------------------------------------------------------
# autograd stages (lists: one entry per driven rank)
# ---------------------------------------------------------------------------------------------------------------------
class _Stage1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, st: _EPState, caches1, *args):
        n = len(st.ranks)
        xs, tws, w1s = args[:n], args[n : 2 * n], args[2 * n : 3 * n]
        for rk, x, tw in zip(st.ranks, xs, tws):
            rk.T, rk.topk_dtype = x.shape[0], tw.dtype
        pools = _dispatch_fwd(st, xs, st.idxs, tws)
        l1s = []
        for rk, (xq, xs_), c1 in zip(st.ranks, pools, caches1):
            if not c1.valid:  # first use only (eager, before any capture)
                c1.refresh()
            l1s.append(_blk._gg(xq, xs_, c1, False, rk.poffs))
        del pools
        ctx.st, ctx.caches1, ctx.w1s = st, caches1, w1s
        ctx.set_materialize_grads(False)
        return (*l1s, *[rk.rw for rk in st.ranks])

    @staticmethod
    @torch.no_grad()
    def backward(ctx, *grads):
        st = ctx.st
        n = len(st.ranks)
        d_rw = grads[n : 2 * n]
        if all(g is None for g in d_rw) or st.ranks[0].g is None:
            return (None, None) + (None,) * (3 * n)
        dW1s, dxps = [], []
        for rk, w1, c1 in zip(st.ranks, ctx.w1s, ctx.caches1):
            gq, gs, gT, gTs = rk.g
            dW1s.append(_wgrad(st, rk, w1, gT, gTs, rk.xT, rk.xTs, 2 * st.I, st.H, st.acc1))
            rk.xT = rk.xTs = None
            dxps.append(_blk._gg(gq, gs, c1, True, rk.poffs))
            rk.g = None
        outs = _combine(st, dxps, [g for g in d_rw])
        del dxps
        dxs = [o[0] for o in outs]
        dtws = [o[1].to(rk.topk_dtype) for o, rk in zip(outs, st.ranks)]
        ctx.st = ctx.caches1 = ctx.w1s = None
        return (None, None, *dxs, *dtws, *dW1s)


class _Stage2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, st: _EPState, caches2, *args):
        n = len(st.ranks)
        l1s, rws, w2s = args[:n], args[n : 2 * n], args[2 * n : 3 * n]
        l2s = []
        for rk, l1, rw, c2 in zip(st.ranks, l1s, rws, caches2):
            if not c2.valid:
                c2.refresh()
            aq, as_, rk.aT, rk.aTs = swiglu_fwd_dual_blk(
                l1, rw, st.activation, preal=rk.preal, compact_t=st.ct
            )
            l2s.append(_blk._gg(aq, as_, c2, False, rk.poffs))
            rk.l1 = l1
        ys = _combine(st, l2s)
        del l2s
        ctx.st, ctx.caches2, ctx.w2s = st, caches2, w2s
        ctx.set_materialize_grads(False)
        return tuple(ys)

    @staticmethod
    @torch.no_grad()
    def backward(ctx, *dys):
        st = ctx.st
        n = len(st.ranks)
        if all(d is None for d in dys):
            return (None, None) + (None,) * (3 * n)
        assert all(d is not None for d in dys), "every rank's y needs a gradient"
        pools = _dispatch_dy(st, dys)
        dW2s, dws = [], []
        for rk, (dq, ds, dT, dTs), w2, c2 in zip(st.ranks, pools, ctx.w2s, ctx.caches2):
            dact = _blk._gg(dq, ds, c2, True, rk.poffs)
            del dq, ds
            gq, gs, gT, gTs, dwr = swiglu_bwd_dual_blk(
                dact, rk.l1, rk.rw, st.activation, preal=rk.preal, compact_t=st.ct
            )
            del dact
            rk.perm = expert_order_lpt_fused(rk.poffs) if st.wk == "vkq" else None
            dW2s.append(_wgrad(st, rk, w2, dT, dTs, rk.aT, rk.aTs, st.H, st.I, st.acc2))
            rk.aT = rk.aTs = rk.l1 = None
            rk.g = (gq, gs, gT, gTs)
            dws.append(dwr)
        del pools
        ctx.st = ctx.caches2 = ctx.w2s = None
        # (st, caches2, *l1s, *route_ws, *w2s): d route_w carries d topk_w (and keeps stage1.backward live)
        return (None, None, *([None] * n), *dws, *dW2s)


# ---------------------------------------------------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------------------------------------------------
def _check(xs, idxs, tws, ws, comm, w_is_w1):
    n = len(comm.local_ranks)
    assert len(xs) == len(idxs) == len(tws) == len(ws) == n, (
        "one x / topk_idx / topk_weights / weight per driven rank"
    )
    T, H = xs[0].shape
    for x, idx, tw in zip(xs, idxs, tws):
        assert x.dim() == 2 and x.dtype == torch.bfloat16 and x.shape == (T, H), (
            "same [T, H] bf16 x on every rank"
        )
        assert idx.shape == tw.shape and idx.shape[0] == T
    E_l = ws[0].shape[0]
    assert all(w.shape == ws[0].shape for w in ws)
    # H <= 512: the fp8_block grouped GEMMs return zeros / NaN; the EP kernels are validated for H % 1024 == 0
    assert H >= _MIN_H and H % 1024 == 0, (
        f"fp8_block EP MegaMoE needs H >= {_MIN_H}, H % 1024 == 0 (got H={H})"
    )
    if w_is_w1:
        assert ws[0].shape[2] == H and ws[0].shape[1] % 512 == 0, "w1 [E_l, 2I, H] with I % 256 == 0"
    return T, H, E_l


def fused_mega_moe_fp8blk_ep_stage1(
    xs,
    topk_idxs,
    topk_weights,
    w1s,
    caches1,
    comm: EPComm,
    accumulate_wgrad: bool = False,
    dispatch: Optional[str] = None,
):
    """Stage1 (dispatch + fc1, owns w1) for the driven ranks of ``comm`` (lists, one entry per rank).
    ``topk_idxs`` hold GLOBAL expert ids (rank d owns experts [d*E_l, (d+1)*E_l)). Returns (l1s, route_ws, state) to
    feed stage2."""
    T, H, E_l = _check(xs, topk_idxs, topk_weights, w1s, comm, True)
    K = topk_idxs[0].shape[1]
    I = w1s[0].shape[1] // 2
    dispatch = dispatch or "fp8"
    assert dispatch in ("fp8", "bf16"), f"dispatch={dispatch!r}: must be 'fp8' or 'bf16'"
    st = _EPState(comm, E_l, K, H, I, T, dispatch)
    st.idxs = [i.to(torch.int64).contiguous() for i in topk_idxs]
    st.acc1 = accumulate_wgrad
    out = _Stage1.apply(st, list(caches1), *[x.contiguous() for x in xs], *topk_weights, *w1s)
    n = len(xs)
    return list(out[:n]), list(out[n:]), st


def fused_mega_moe_fp8blk_ep_stage2(
    l1s, route_ws, state: _EPState, w2s, caches2, activation=None, accumulate_wgrad: bool = False
):
    """Stage2 (SwiGLU + fc2 + combine, owns w2). Returns the per-rank y [T, H] bf16."""
    st = state
    assert all(w.shape == (st.E_l, st.H, st.I) for w in w2s)
    st.activation = activation
    st.acc2 = accumulate_wgrad
    return list(_Stage2.apply(st, list(caches2), *l1s, *route_ws, *w2s))


def fused_mega_moe_fp8blk_ep_multi(
    xs,
    topk_idxs,
    topk_weights,
    w1s,
    w2s,
    caches1,
    caches2,
    comm: EPComm,
    activation=None,
    accumulate_wgrad: bool = False,
    dispatch: Optional[str] = None,
):
    """Both stages for all ranks ``comm`` drives (lists). With ``EmulatedEPComm(R)``: the whole EP group in one
    process."""
    l1s, rws, st = fused_mega_moe_fp8blk_ep_stage1(
        xs, topk_idxs, topk_weights, w1s, caches1, comm, accumulate_wgrad, dispatch
    )
    return fused_mega_moe_fp8blk_ep_stage2(l1s, rws, st, w2s, caches2, activation, accumulate_wgrad)


def fused_mega_moe_fp8blk_ep(
    x,
    topk_idx,
    topk_weights,
    w1,
    w2,
    cache1: FP8BlockWeightCache,
    cache2: FP8BlockWeightCache,
    group_or_comm,
    activation=None,
    accumulate_wgrad: bool = False,
    dispatch: Optional[str] = None,
) -> torch.Tensor:
    """One rank of an EP group (torch.distributed group, or an EPComm driving one rank): y [T, H] for this rank's
    tokens. x [T, H] bf16, topk_idx [T, K] global expert ids, w1 [E_l, 2I, H] / w2 [E_l, H, I] this rank's experts,
    caches as for fused_mega_moe_fp8blk. Same contract as fused_mega_moe_fp8blk otherwise."""
    comm = group_or_comm if isinstance(group_or_comm, EPComm) else DistEPComm(group_or_comm)
    assert len(comm.local_ranks) == 1
    (y,) = fused_mega_moe_fp8blk_ep_multi(
        [x],
        [topk_idx],
        [topk_weights],
        [w1],
        [w2],
        [cache1],
        [cache2],
        comm,
        activation,
        accumulate_wgrad,
        dispatch,
    )
    return y
