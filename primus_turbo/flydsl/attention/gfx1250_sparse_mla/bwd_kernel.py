###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""DeepSeek-V4 sparse-MLA attention backward for gfx1250 (MI455X): wave32, WMMA 16x16x32.

Same contract as ``sparse_mla_bwd_triton`` (dq / dkv rope columns are zero, dsink from
``delta`` and ``lse``), deterministic: no atomics.

One fused kernel per token, all heads in one workgroup (16 heads per wave group; with fewer
than 128 heads, two waves share a group and split its slots and dQ^T columns):

1. ``S = Q K^T`` and ``dP = dO K^T`` with heads as rows, so a lane holds one top-k slot and
   eight heads; with lse known, ``P`` and ``dS = P (dP - delta) scale`` are elementwise.
2. ``dS`` and ``P`` go to LDS as ``[slot][head]``.
3. ``dQ^T += K^T dS^T`` per wave (both operands are ``ds_load_tr16_b128`` reads).
4. Waves split the latent width and form each slot's dKV contribution
   ``C^T = Q^T dS + dO^T P`` (reduction over all heads, Q / dO staged in LDS), written to a
   bf16 ``[T, topk, 512]`` intermediate.

The intermediate is reduced per kv row through the CSR inverse of the top-k map by the
Triton gather kernel the Triton backend uses.

The default kernel (``build_sparse_mla_bwd_gfx1250_w1``) runs the same steps with one wave per
SIMD and 32 heads per wave (two 16-head groups), so each K / K^T fragment read from LDS feeds
two WMMAs; kv tiles arrive by TDM gather and LDS operands are prefetched several WMMA steps
ahead. The 8-wave kernel above remains for kv tiles that do not divide top-k.

Pure SWA (cr0: top-k is the 128-row window ``[t-127, t]``) skips the intermediate and the
inverse: ``build_sparse_mla_bwd_gfx1250_swa`` accumulates dKV over 16 consecutive tokens in
VGPRs and writes one fp32 partial per token block, which ``build_swa_dkv_reduce`` sums per kv
row into the bf16 dKV.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from .fwd_kernel import (
    COMPILE_HINTS,
    DQK,
    DV,
    ROW,
    _sbuf_load_v8i32,
    _tdm_gather,
)
from .helper import (
    CPOL_NT,
    LOG2E,
    WAVE,
    buf_load,
    buf_store,
    cat,
    exp2,
    fadd,
    fmul,
    fsub,
    lds_load,
    lds_load_tr16,
    lds_store,
    make_rsrc,
    swap_halves,
    tensor_addr,
    to_bf16,
    uniform,
    vec_type,
    vmul_scalar,
    wmma_bf16,
)

_LDS_CAP = 320 * 1024
# Per-chunk bound on the dKV intermediate. Besides capping memory, the gather's scattered row
# reads run ~2.5x faster once the buffer spans less than ~2 GB.
_INTERM_BYTES = 1536 << 20
_GATHER_SEG = 256  # CSR entries per gather program
# global_prefetch_b8 cache policy: SCOPE_DEV (fill L2, not the WGP cache), non-speculative.
_PF_L2 = (2 << 3) | 1
_ROW_XMASK0 = 0x160  # DPP row_xmask:0


def _row_xor(x, m):
    """f32 ``x`` of lane ``lane ^ m`` within its 16-lane row (``m`` < 16)."""
    v = _rocdl.update_dpp(T.f32, _raw(x), _raw(x), _ROW_XMASK0 + m, 0xF, 0xF, True)
    return fx.Float32(v if isinstance(v, ir.Value) else v.result)


def _l2_prefetch(base, nbytes, tid, block):
    """Prefetch ``nbytes`` at ``base`` into L2, one 128 B line per lane per instruction."""
    for k in range_constexpr(-(-nbytes // (block * 128))):
        off = fx.Int32((k * block + tid) * 128)
        off = fx.Int32((off < fx.Int32(nbytes)).select(off, fx.Int32(nbytes - 128)))
        ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), _raw(base + fx.Int64(off))).result
        llvm.call_intrinsic(None, "llvm.amdgcn.global.prefetch", [ptr, _raw(fx.Int32(_PF_L2))], [], [])


@functools.lru_cache(maxsize=64)
def build_sparse_mla_bwd_gfx1250(
    num_heads, topk, scale, block_k=32, d_split=None, prefetch=None, pin=None, ct_group=None
):
    H = num_heads
    if H % 32 or 128 % H:
        raise ValueError(f"gfx1250 sparse-MLA backward needs num_heads in (32, 64, 128), got {H}")
    BN = block_k
    if topk % BN or BN % 32:
        raise ValueError(f"topk {topk} must be a multiple of the kv tile {BN}")
    HG = H // 16  # 16-head groups
    KT = BN // 16  # 16-slot sub-tiles
    # Waves per head group: they split the S / dP slot sub-tiles and the dQ^T d range, so small
    # head counts still run 8 waves.
    DS = d_split or max(1, min(8 // HG, KT))
    WAVES = HG * DS
    BLOCK = WAVES * WAVE
    NT = topk // BN
    KTW = KT // DS  # slot sub-tiles per wave (S, dP)
    PC = BN // 32  # 32-slot reduction steps (dQ)
    KS = DV // 32  # d reduction steps (S, dP)
    DT = DV // 16  # d tiles of dQ^T
    DTQ = DT // DS  # dQ^T d tiles per wave
    HC = H // 32  # head reduction steps (dKV)
    DTW = DT // WAVES  # d tiles of C^T per wave
    RZ = (DQK - DV) * 2 // 32  # 16-byte rope-zero chunks per half row of dq
    if KT % DS or DT % WAVES or RZ % DS:
        raise ValueError(f"d_split {DS} does not divide the tile for num_heads {H}")
    ROW = DV * 2 + 16  # LDS pitch of latent rows (bytes)
    SP = H * 2 + 16  # LDS pitch of the [slot][head] dS / P tiles (bytes)
    QO_B = H * ROW
    TILE_B = BN * ROW
    DS_B = BN * SP
    fixed = 2 * QO_B + 2 * DS_B + DS * H * 4
    NBUF = 2 if fixed + 2 * TILE_B <= _LDS_CAP else 1
    if fixed + NBUF * TILE_B > _LDS_CAP:
        raise ValueError("sparse-MLA backward tile does not fit in LDS")
    CPR = DV * 2 // 16  # 16-byte chunks per latent row
    TPR = BLOCK // BN
    CH = CPR // TPR
    QCH = H * CPR // BLOCK  # staging chunks per thread for Q / dO
    c_log2 = float(scale) * LOG2E
    # 128 heads already hold ~450 VGPRs (16x512 fp32 dQ^T per wave); prefetched fragments spill
    # there, and pinned order without prefetch only constrains the scheduler.
    if prefetch is None:
        prefetch = 8 if H < 128 else 0
    if pin is None:
        pin = 4 if prefetch else 0
    pin_every = pin if pin > 0 else 1 << 30
    # Without pinned prefetch the backend already merges the repeated Q^T / dO^T loads.
    CTG = (KT if prefetch else 1) if ct_group is None else ct_group
    if KT % CTG:
        raise ValueError(f"ct_group {CTG} must divide the {KT} slot sub-tiles")

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def sparse_mla_bwd_gfx1250_kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DQ: fx.Tensor,
        DELTA: fx.Tensor,
        INTERM: fx.Tensor,
        NKV: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = uniform(tid // WAVE)
        hg = wave % HG
        ds = wave // HG
        lane = tid % WAVE
        l16 = lane % 16
        hi = lane // 16
        l8 = lane % 8
        tok = fx.Int32(fx.block_idx.x)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(fixed + NBUF * TILE_B)
        lds0 = fx.Int32(fx.Int64(fx.ptrtoint(arena.base_ptr)))
        q_lds = lds0
        do_lds = lds0 + QO_B
        ds_lds = lds0 + 2 * QO_B
        p_lds = ds_lds + DS_B
        dl_lds = p_lds + DS_B
        kv_lds = dl_lds + DS * H * 4

        v4i = vec_type(4, fx.Int32)
        v4f = vec_type(4, fx.Float32)
        v8e = vec_type(8, fx.BFloat16)
        tok_q = fx.Int64(tok) * (H * DQK * 2)
        tok_o = fx.Int64(tok) * (H * DV * 2)
        rsrc_q = make_rsrc(tensor_addr(Q) + tok_q, H * DQK * 2)
        rsrc_do = make_rsrc(tensor_addr(DO) + tok_o, H * DV * 2)
        rsrc_o = make_rsrc(tensor_addr(O) + tok_o, H * DV * 2)
        rsrc_idx = make_rsrc(tensor_addr(TOPK) + fx.Int64(tok) * (topk * 4), topk * 4)
        kv_bytes = NKV * (DQK * 2)
        rsrc_kv = make_rsrc(tensor_addr(KV), kv_bytes)

        # ---- stage Q and dO (all heads) in LDS.
        for c in range_constexpr(QCH):
            idx = tid + c * BLOCK
            r, col = idx // CPR, idx % CPR
            lds_store(buf_load(rsrc_q, r * (DQK * 2) + col * 16, v4i), q_lds + r * ROW + col * 16)
            lds_store(buf_load(rsrc_do, r * (DV * 2) + col * 16, v4i), do_lds + r * ROW + col * 16)

        # ---- delta = rowsum(O * dO), dO from LDS: lane (l16, hi) of split ds sums part ds*2 + hi
        # of head hg*16 + l16; the DS partials are summed when read back.
        h_own = hg * 16 + l16
        CPP = CPR // (2 * DS)
        part = ds * 2 + hi
        o_regs = [
            buf_load(rsrc_o, h_own * (DV * 2) + (part * CPP + j) * 16, v4i) for j in range_constexpr(CPP)
        ]
        gpu.barrier()
        acc = fx.Float32(0.0)
        for j in range_constexpr(CPP):
            off = (part * CPP + j) * 16
            ov = Vec(o_regs[j]).bitcast(fx.BFloat16)
            dv = Vec(lds_load(do_lds + h_own * ROW + off, v4i)).bitcast(fx.BFloat16)
            for e in range_constexpr(8):
                acc = fadd(acc, fmul(fx.Float32(ov[e]), fx.Float32(dv[e])))
        lds_store(fadd(acc, swap_halves(acc)), dl_lds + (ds * H + h_own) * 4, align=4)

        # ---- KV gather (same scheme as the forward).
        g_row = tid // TPR
        g_col = tid % TPR

        def load_tile(t):
            idx = fx.Int32(buf_load(rsrc_idx, (t * BN + g_row) * 4, T.i32))
            row_off = fx.Int32((idx >= 0).select(idx * (DQK * 2), kv_bytes)) + g_col * 16
            return [buf_load(rsrc_kv, row_off + c * (TPR * 16), v4i) for c in range_constexpr(CH)]

        def store_tile(regs, buf):
            base = kv_lds + buf * TILE_B + g_row * ROW + g_col * 16
            for c in range_constexpr(CH):
                lds_store(regs[c], base + c * (TPR * 16))

        store_tile(load_tile(fx.Int32(0)), fx.Int32(0))
        gpu.barrier()

        delta_own = fx.Float32(lds_load(dl_lds + h_own * 4, T.f32, align=4))
        for k in range_constexpr(1, DS):
            delta_own = fadd(delta_own, fx.Float32(lds_load(dl_lds + (k * H + h_own) * 4, T.f32, align=4)))
        rsrc_dl = make_rsrc(tensor_addr(DELTA) + fx.Int64(tok) * (H * 4), H * 4)
        buf_store(delta_own, rsrc_dl, fx.Int32((part == 0).select(h_own * 4, fx.Int32(0x7FFFFFF0))))

        # Per-lane row statistics for heads hg*16 + hi*8 + i (rows of this lane's S tile).
        rsrc_lse = make_rsrc(tensor_addr(LSE) + fx.Int64(tok) * (H * 4), H * 4)
        hrow = hg * 16 + hi * 8
        lsev = cat(Vec(buf_load(rsrc_lse, hrow * 4, v4f)), Vec(buf_load(rsrc_lse, hrow * 4 + 16, v4f)))
        dlt = [fx.Float32(0.0)] * 8
        for k in range_constexpr(DS):
            o = dl_lds + (k * H + hrow) * 4
            dlv = cat(lds_load(o, v4f), lds_load(o + 16, v4f))
            dlt = [fadd(dlt[i], fx.Float32(dlv[i])) for i in range_constexpr(8)]
        lse2 = []
        for i in range_constexpr(8):
            l_ = fx.Float32(lsev[i])
            # No visible key and no sink: lse = -inf, P must be 0.
            lse2.append(
                fx.Float32(
                    (l_ == fx.Float32(float("-inf"))).select(fx.Float32(float("inf")), fmul(l_, LOG2E))
                )
            )

        a_lane = l16 * ROW + hi * 16  # non-transposed fragment read (lane = row)
        tr_lane = (hi * 8 + l8) * ROW + (l16 // 8) * 16
        tr_ds = (hi * 8 + l8) * SP + (l16 // 8) * 16
        q_row = q_lds + (hg * 16) * ROW + a_lane
        do_row = do_lds + (hg * 16) * ROW + a_lane
        kt0 = ds * KTW  # first slot sub-tile of this wave
        dt0 = ds * DTQ  # first dQ^T d tile of this wave
        interm_tok = fx.Int64(tok) * fx.Int64(topk * DV * 2)
        rsrc_im = make_rsrc(tensor_addr(INTERM) + interm_tok, topk * DV * 2)

        def frag(addr):
            return cat(lds_load(addr, v8e), lds_load(addr + 32, v8e))

        def frag_tr(addr, pitch):
            return cat(lds_load_tr16(addr), lds_load_tr16(addr + 16 * pitch))

        # A step is (load: () -> fragments, mma: fragments -> None); loads run PREFETCH steps
        # ahead of their WMMAs (the backend does not hoist LDS reads on its own).
        def run_steps(steps):
            q = [steps[i][0]() for i in range_constexpr(min(prefetch, len(steps)))]
            for i in range_constexpr(len(steps)):
                if const_expr(i + prefetch < len(steps)):
                    q.append(steps[i + prefetch][0]())
                steps[i][1](q.pop(0))
                if const_expr((i + 1) % pin_every == 0):
                    fx.rocdl.sched_barrier(0)

        def tile_compute(t, buf, dq_acc):
            kbase = kv_lds + buf * TILE_B
            kb_w = kbase + a_lane + kt0 * 16 * ROW
            s = [Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(KTW)]
            dp = [Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(KTW)]
            cur = {}
            steps = []
            for ks in range_constexpr(KS):
                for kk in range_constexpr(KTW):

                    def ld(ks=ks, kk=kk):
                        bk = frag(kb_w + kk * 16 * ROW + ks * 64)
                        if const_expr(kk == 0):
                            return (frag(q_row + ks * 64), frag(do_row + ks * 64), bk)
                        return (bk,)

                    def mma(fr, kk=kk):
                        if const_expr(kk == 0):
                            cur["aq"], cur["ad"] = fr[0], fr[1]
                        s[kk] = wmma_bf16(cur["aq"], fr[-1], s[kk])
                        dp[kk] = wmma_bf16(cur["ad"], fr[-1], dp[kk])

                    steps.append((ld, mma))
            run_steps(steps)
            # P, dS: lane = slot t*BN + kt*16 + l16, element i = head hg*16 + hi*8 + i.
            for kk in range_constexpr(KTW):
                kt = kt0 + kk
                idx = fx.Int32(buf_load(rsrc_idx, (t * BN + kt * 16 + l16) * 4, T.i32))
                ok = idx >= fx.Int32(0)
                sv = vmul_scalar(s[kk], c_log2)
                pv, dsv = [], []
                for i in range_constexpr(8):
                    p = fx.Float32(ok.select(exp2(fsub(sv[i], lse2[i])), fx.Float32(0.0)))
                    pv.append(p)
                    dsv.append(fmul(fmul(p, fsub(dp[kk][i], dlt[i])), scale))
                slot_off = (kt * 16 + l16) * SP + hrow * 2
                lds_store(to_bf16(Vec.from_elements(dsv, fx.Float32)), ds_lds + slot_off)
                lds_store(to_bf16(Vec.from_elements(pv, fx.Float32)), p_lds + slot_off)
            gpu.barrier()
            # dQ^T[d, own heads] += K^T[d, slot] dS^T[slot, h].
            new = list(dq_acc)
            steps = []
            for pc in range_constexpr(PC):
                for j in range_constexpr(DTQ):

                    def ld(pc=pc, j=j):
                        a = frag_tr(kbase + tr_lane + pc * 32 * ROW + (dt0 + j) * 32, ROW)
                        if const_expr(j == 0):
                            return (frag_tr(ds_lds + tr_ds + pc * 32 * SP + hg * 32, SP), a)
                        return (a,)

                    def mma(fr, j=j):
                        if const_expr(j == 0):
                            cur["b"] = fr[0]
                        new[j] = wmma_bf16(fr[-1], cur["b"], new[j])

                    steps.append((ld, mma))
            # C^T[d, slot] = Q^T dS + dO^T P over this wave's d tiles; each Q^T / dO^T fragment
            # feeds the CTG slot tiles of a group.
            for kg in range_constexpr(KT // CTG):
                for kt in range_constexpr(kg * CTG, kg * CTG + CTG):
                    for hc in range_constexpr(HC):

                        def ld(kt=kt, hc=hc):
                            o = (kt * 16 + l16) * SP + hc * 64 + hi * 16
                            return (frag(ds_lds + o), frag(p_lds + o))

                        def keep(fr, kt=kt, hc=hc):
                            cur[("bs", kt, hc)], cur[("bp", kt, hc)] = fr

                        steps.append((ld, keep))
                for j in range_constexpr(DTW):
                    dt = wave * DTW + j
                    for hc in range_constexpr(HC):

                        def ld(hc=hc, dt=dt):
                            o = tr_lane + hc * 32 * ROW + dt * 32
                            return (frag_tr(q_lds + o, ROW), frag_tr(do_lds + o, ROW))

                        def mma(fr, kg=kg, hc=hc, dt=dt):
                            for kt in range_constexpr(kg * CTG, kg * CTG + CTG):
                                c = Vec.filled(8, 0.0, fx.Float32) if hc == 0 else cur[("c", kt)]
                                c = wmma_bf16(fr[0], cur[("bs", kt, hc)], c)
                                c = wmma_bf16(fr[1], cur[("bp", kt, hc)], c)
                                cur[("c", kt)] = c
                                if const_expr(hc == HC - 1):
                                    im_off = (t * BN + kt * 16 + l16) * (DV * 2) + dt * 32 + hi * 16
                                    buf_store(to_bf16(c).bitcast(fx.Int32), rsrc_im, im_off, aux=CPOL_NT)

                        steps.append((ld, mma))
            run_steps(steps)
            return new

        init = [Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(DTQ)]
        for t, st in range(fx.Index(0), fx.Index(NT), fx.Index(1), init=init):
            ti = fx.Int32(t)
            nxt = load_tile(ti + 1)
            if const_expr(NBUF == 2):
                buf = ti % 2
                acc_ = tile_compute(ti, buf, [st[j] for j in range_constexpr(DTQ)])
                store_tile(nxt, 1 - buf)
                gpu.barrier()
            else:
                acc_ = tile_compute(ti, fx.Int32(0), [st[j] for j in range_constexpr(DTQ)])
                gpu.barrier()
                store_tile(nxt, fx.Int32(0))
                gpu.barrier()
            res = yield list(acc_)

        # dQ^T tiles: lane holds head hg*16 + l16, d rows (dt0 + j)*16 + hi*8 + [0..7].
        rsrc_dq = make_rsrc(tensor_addr(DQ) + tok_q, H * DQK * 2)
        dq_off = h_own * (DQK * 2) + hi * 16
        for j in range_constexpr(DTQ):
            buf_store(to_bf16(res[j]).bitcast(fx.Int32), rsrc_dq, dq_off + (dt0 + j) * 32)
        zero = Vec.filled(4, 0, fx.Int32)
        rz_off = h_own * (DQK * 2) + DV * 2 + hi * (RZ * 16) + ds * (RZ // DS * 16)
        for j in range_constexpr(RZ // DS):
            buf_store(zero, rsrc_dq, rz_off + j * 16)

    @flyc.jit
    def launch_sparse_mla_bwd_gfx1250(
        Q: fx.Tensor,
        KV: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DQ: fx.Tensor,
        DELTA: fx.Tensor,
        INTERM: fx.Tensor,
        n_tok: fx.Int32,
        NKV: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        sparse_mla_bwd_gfx1250_kernel(Q, KV, O, DO, TOPK, LSE, DQ, DELTA, INTERM, NKV).launch(
            grid=(fx.Index(n_tok), 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    compiled = {}

    def run(q, kv, o, do, topk_indices, lse, dq, delta, interm, stream=None):
        args = (q, kv, o, do, topk_indices, lse, dq, delta, interm, int(q.shape[0]), int(kv.shape[0]))
        fn = compiled.get("fn")
        if fn is None:
            fn = flyc.compile[COMPILE_HINTS](launch_sparse_mla_bwd_gfx1250, *args, fx.Stream(stream))
            compiled["fn"] = fn
        return fn(*args, stream)

    return run


def _frag(addr):
    v8e = vec_type(8, fx.BFloat16)
    return cat(lds_load(addr, v8e), lds_load(addr + 32, v8e))


def _frag_tr(addr, pitch):
    return cat(lds_load_tr16(addr), lds_load_tr16(addr + 16 * pitch))


def _run_steps(steps, prefetch, pin, batch=0):
    """Run ``(load, mma)`` steps with loads ``prefetch`` steps ahead, ``sched_barrier`` every ``pin``.

    ``batch > 0``: batch b's mmas run while batch b+1's loads go to other registers, with
    ``sched_barrier`` only at batch edges. LLVM 24 waits ``va_vdst(0)`` before a ds_load that
    overwrites a register some WMMA read; batching makes that once per batch, not per step."""
    if const_expr(batch > 0):
        n = len(steps)
        cur = [steps[i][0]() for i in range_constexpr(min(batch, n))]
        for b0 in range_constexpr(0, n, batch):
            nxt = [steps[i][0]() for i in range_constexpr(b0 + batch, min(b0 + 2 * batch, n))]
            fx.rocdl.sched_barrier(0)
            for j in range_constexpr(len(cur)):
                steps[b0 + j][1](cur[j])
            fx.rocdl.sched_barrier(0)
            cur = nxt
        return
    q = [steps[i][0]() for i in range_constexpr(min(prefetch, len(steps)))]
    for i in range_constexpr(len(steps)):
        if const_expr(i + prefetch < len(steps)):
            q.append(steps[i + prefetch][0]())
        steps[i][1](q.pop(0))
        if const_expr((i + 1) % pin == 0):
            fx.rocdl.sched_barrier(0)


def _lds_barrier():
    """Workgroup barrier that orders LDS only (``gpu.barrier()`` also waits for every outstanding
    global store, which serializes a loop on store latency)."""
    fx.rocdl.s_wait_dscnt(0)
    fx.rocdl.s_barrier_signal(-1)
    fx.rocdl.s_barrier_wait(-1)


def _w1_block_k(num_heads):
    return 32 if num_heads >= 128 else 64


@functools.lru_cache(maxsize=64)
def build_sparse_mla_bwd_gfx1250_w1(
    num_heads, topk, scale, block_k=None, prefetch=8, pin=None, ahead=256, ct_resident=None, batch=4
):
    """One wave per SIMD (4 waves), 32 heads per wave; the kv tiles arrive by TDM gather.

    With fewer than 128 heads, ``4 * 32 / H`` waves share a head pair and split its slot
    sub-tiles (S / dP) and dQ^T d tiles. C^T does not read the kv tile, so with a single kv
    buffer the next gather is issued after dQ^T and lands under C^T.

    A token walks only its tiles up to the last valid slot (``ntile``, per token): the indexer
    leaves the causally masked picks as trailing -1, about half the slots at T = 4096.

    After its last tile, a workgroup prefetches into L2 the Q / dO / O / indices / lse of token
    ``tok + ahead`` (one workgroup per CU: the token that takes over this CU next).

    ``batch``: LDS operands load in batches of that many steps (see ``_run_steps``); 4 beats
    3 / 5 / 6 within noise at the fewest VGPRs, 8 spills. 0 restores the ``prefetch`` / ``pin``
    stream."""
    H = num_heads
    WAVES = 4
    HP = H // 32  # 32-head pairs
    if H % 32 or WAVES % HP:
        raise ValueError(f"num_heads {H} must be 32, 64 or 128")
    DS = WAVES // HP  # waves per head pair
    BN = block_k or _w1_block_k(H)
    KT = BN // 16
    if topk % BN or KT % DS:
        raise ValueError(f"topk {topk} must be a multiple of the kv tile {BN}")
    G = 2  # 16-head groups per wave
    BLOCK = WAVES * WAVE
    KTW = KT // DS
    PC = BN // 32
    KS = DV // 32
    DT = DV // 16
    DTQ = DT // DS
    HC = H // 32
    DTW = DT // WAVES
    RZ = (DQK - DV) * 2 // 32
    if RZ % DS:
        raise ValueError(f"num_heads {H} does not split the dq rope columns")
    SP = H * 2 + 16
    QO_B = H * ROW
    TILE_B = BN * ROW
    DS_B = BN * SP
    PPH = BLOCK // H  # delta partials per head
    fixed = 2 * QO_B + 2 * DS_B + PPH * H * 4
    NBUF = 2 if fixed + 2 * TILE_B <= _LDS_CAP else 1
    if fixed + NBUF * TILE_B > _LDS_CAP:
        raise ValueError("sparse-MLA backward tile does not fit in LDS")

    # C^T goes out through a per-wave LDS stage of 16 slots x GW d tiles: the widest rows (all
    # DTW tiles, else 4 = 128 B) that fit next to the kv buffers.
    def _stage_b(gw):
        return 16 * (gw * 32 + 16)

    fits = [
        gw for gw in (DTW, 4) if DTW % gw == 0 and fixed + NBUF * TILE_B + WAVES * _stage_b(gw) <= _LDS_CAP
    ]
    GW = fits[0] if fits else 0
    STAGE = GW > 0
    SROW = GW * 32 + 16
    STG_B = _stage_b(GW)
    LPR = max(GW * 2, 1)  # lanes per staged row (16 B each)
    RPS = WAVE // LPR  # rows per buffer_store
    RPO = 8  # rows per TDM gather (i32 indices)
    if (BN // RPO) % WAVES:
        raise ValueError(f"kv tile {BN} does not split into TDM ops over {WAVES} waves")
    OPW = BN // RPO // WAVES
    CPR = DV * 2 // 16
    QCH = H * CPR // BLOCK
    CPP = CPR // PPH
    QCPR = DQK * 2 // 16  # 16 B chunks per dq row
    QROW = DQK * 2 + 16  # LDS pitch of the dq stage
    if (H * QCPR) % BLOCK or H * QROW > 2 * QO_B:
        raise ValueError(f"num_heads {H} does not fit the dq stage")
    c_log2 = float(scale) * LOG2E
    if pin is None:
        pin = 4 if H >= 128 else 2
    pin_every = pin if pin > 0 else 1 << 30
    # The C^T A operands (Q^T / dO^T of this wave's d tiles) are the same for every kv tile:
    # with fewer heads (smaller dQ accumulator) they fit in registers for the whole token.
    CT_RES = H < 128 if ct_resident is None else ct_resident

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def sparse_mla_bwd_gfx1250_w1_kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DQ: fx.Tensor,
        DELTA: fx.Tensor,
        INTERM: fx.Tensor,
        NTILE: fx.Tensor,
        NKV: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = uniform(tid // WAVE)
        hp = wave % HP
        ds = wave // HP
        lane = tid % WAVE
        l16 = lane % 16
        hi = lane // 16
        l8 = lane % 8
        tok = fx.Int32(fx.block_idx.x)
        rsrc_nt = make_rsrc(tensor_addr(NTILE), (tok + fx.Int32(1)) * 4)
        n_tile = uniform(fx.Int32(buf_load(rsrc_nt, tok * 4, T.i32)))
        last_t = fx.Int32((n_tile > fx.Int32(0)).select(n_tile - fx.Int32(1), fx.Int32(0)))

        arena = fx.SharedAllocator(static=False)
        arena.allocate(fixed + NBUF * TILE_B + (WAVES * STG_B if STAGE else 0))
        lds0 = fx.Int32(fx.Int64(fx.ptrtoint(arena.base_ptr)))
        q_lds = lds0
        do_lds = lds0 + QO_B
        ds_lds = lds0 + 2 * QO_B
        p_lds = ds_lds + DS_B
        dl_lds = p_lds + DS_B
        kv_lds = dl_lds + PPH * H * 4
        stg = kv_lds + NBUF * TILE_B + wave * STG_B

        v4i = vec_type(4, fx.Int32)
        v4f = vec_type(4, fx.Float32)
        tok_q = fx.Int64(tok) * (H * DQK * 2)
        tok_o = fx.Int64(tok) * (H * DV * 2)
        rsrc_q = make_rsrc(tensor_addr(Q) + tok_q, H * DQK * 2)
        rsrc_do = make_rsrc(tensor_addr(DO) + tok_o, H * DV * 2)
        rsrc_o = make_rsrc(tensor_addr(O) + tok_o, H * DV * 2)
        rsrc_idx = make_rsrc(tensor_addr(TOPK) + fx.Int64(tok) * (topk * 4), topk * 4)
        kv_addr = tensor_addr(KV)

        def issue(t, buf):
            # Past the last walked tile, re-fetch it into a buffer that is not read again.
            ct = fx.Int32((t < n_tile).select(t, last_t))
            base = ct * BN + wave * (OPW * RPO)
            for k in range_constexpr(OPW):
                oct_ = _sbuf_load_v8i32(rsrc_idx, (base + k * RPO) * 4)
                words = [oct_[i] for i in range_constexpr(8)]
                # An unused slot (-1) is out of range: the row lands as zeros (P is masked).
                _tdm_gather(kv_lds + buf * TILE_B + (wave * OPW + k) * (RPO * ROW), kv_addr, words, RPO, NKV)

        issue(fx.Int32(0), fx.Int32(0))

        # ---- stage Q and dO (all heads) in LDS; O rows for delta go out alongside.
        h_d = tid // PPH
        p_d = tid % PPH
        o_regs = [buf_load(rsrc_o, h_d * (DV * 2) + (p_d * CPP + j) * 16, v4i) for j in range_constexpr(CPP)]
        for c in range_constexpr(QCH):
            idx = tid + c * BLOCK
            r, col = idx // CPR, idx % CPR
            lds_store(buf_load(rsrc_q, r * (DQK * 2) + col * 16, v4i), q_lds + r * ROW + col * 16)
            lds_store(buf_load(rsrc_do, r * (DV * 2) + col * 16, v4i), do_lds + r * ROW + col * 16)
        gpu.barrier()

        # ---- delta = rowsum(O * dO): thread (h_d, p_d) sums part p_d of head h_d.
        acc = fx.Float32(0.0)
        for j in range_constexpr(CPP):
            ov = Vec(o_regs[j]).bitcast(fx.BFloat16)
            dv = Vec(lds_load(do_lds + h_d * ROW + (p_d * CPP + j) * 16, v4i)).bitcast(fx.BFloat16)
            for e in range_constexpr(8):
                acc = fadd(acc, fmul(fx.Float32(ov[e]), fx.Float32(dv[e])))
        lds_store(acc, dl_lds + (p_d * H + h_d) * 4, align=4)
        gpu.barrier()
        delta_h = fx.Float32(lds_load(dl_lds + h_d * 4, T.f32, align=4))
        for k in range_constexpr(1, PPH):
            delta_h = fadd(delta_h, fx.Float32(lds_load(dl_lds + (k * H + h_d) * 4, T.f32, align=4)))
        rsrc_dl = make_rsrc(tensor_addr(DELTA) + fx.Int64(tok) * (H * 4), H * 4)
        buf_store(delta_h, rsrc_dl, fx.Int32((p_d == 0).select(h_d * 4, fx.Int32(0x7FFFFFF0))))

        # Per-lane row statistics of group g: heads hp*32 + g*16 + hi*8 + i.
        rsrc_lse = make_rsrc(tensor_addr(LSE) + fx.Int64(tok) * (H * 4), H * 4)
        lse2, dlt = [], []
        for g in range_constexpr(G):
            hrow = hp * 32 + g * 16 + hi * 8
            lsev = cat(Vec(buf_load(rsrc_lse, hrow * 4, v4f)), Vec(buf_load(rsrc_lse, hrow * 4 + 16, v4f)))
            dg = [fx.Float32(0.0)] * 8
            for k in range_constexpr(PPH):
                o = dl_lds + (k * H + hrow) * 4
                dlv = cat(lds_load(o, v4f), lds_load(o + 16, v4f))
                dg = [fadd(dg[i], fx.Float32(dlv[i])) for i in range_constexpr(8)]
            dlt.append(dg)
            lg = []
            for i in range_constexpr(8):
                l_ = fx.Float32(lsev[i])
                # No visible key and no sink: lse = -inf, P must be 0.
                lg.append(
                    fx.Float32(
                        (l_ == fx.Float32(float("-inf"))).select(fx.Float32(float("inf")), fmul(l_, LOG2E))
                    )
                )
            lse2.append(lg)

        a_lane = l16 * ROW + hi * 16
        tr_lane = (hi * 8 + l8) * ROW + (l16 // 8) * 16
        tr_ds = (hi * 8 + l8) * SP + (l16 // 8) * 16
        q_row = [q_lds + (hp * 32 + g * 16) * ROW + a_lane for g in range_constexpr(G)]
        do_row = [do_lds + (hp * 32 + g * 16) * ROW + a_lane for g in range_constexpr(G)]
        kt0 = ds * KTW
        dt0 = ds * DTQ
        ct_a = {}
        if const_expr(CT_RES):
            for j in range_constexpr(DTW):
                for hc in range_constexpr(HC):
                    o = tr_lane + hc * 32 * ROW + (wave * DTW + j) * 32
                    ct_a[(j, hc)] = (_frag_tr(q_lds + o, ROW), _frag_tr(do_lds + o, ROW))
        rsrc_im = make_rsrc(tensor_addr(INTERM) + fx.Int64(tok) * fx.Int64(topk * DV * 2), topk * DV * 2)

        def run(steps):
            _run_steps(steps, prefetch, pin_every, batch)

        def tile_compute(t, buf, dq):
            kbase = kv_lds + buf * TILE_B
            kb_w = kbase + a_lane + kt0 * 16 * ROW
            s = [[Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(KTW)] for _ in range_constexpr(G)]
            dp = [[Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(KTW)] for _ in range_constexpr(G)]
            cur = {}
            # ---- S = Q K^T, dP = dO K^T (heads as rows): every K fragment feeds 2 x G WMMAs.
            steps = []
            for ks in range_constexpr(KS):
                for kk in range_constexpr(KTW):

                    def ld(ks=ks, kk=kk):
                        bk = _frag(kb_w + kk * 16 * ROW + ks * 64)
                        if const_expr(kk == 0):
                            aq = [_frag(q_row[g] + ks * 64) for g in range_constexpr(G)]
                            ad = [_frag(do_row[g] + ks * 64) for g in range_constexpr(G)]
                            return (*aq, *ad, bk)
                        return (bk,)

                    def mma(fr, kk=kk):
                        if const_expr(kk == 0):
                            cur["aq"], cur["ad"] = fr[:G], fr[G : 2 * G]
                        for g in range_constexpr(G):
                            s[g][kk] = wmma_bf16(cur["aq"][g], fr[-1], s[g][kk])
                            dp[g][kk] = wmma_bf16(cur["ad"][g], fr[-1], dp[g][kk])

                    steps.append((ld, mma))
            run(steps)
            # ---- P, dS: lane = slot t*BN + kt*16 + l16, element i = head hp*32 + g*16 + hi*8 + i.
            for kk in range_constexpr(KTW):
                kt = kt0 + kk
                idx = fx.Int32(buf_load(rsrc_idx, (t * BN + kt * 16 + l16) * 4, T.i32))
                ok = idx >= fx.Int32(0)
                for g in range_constexpr(G):
                    sv = vmul_scalar(s[g][kk], c_log2)
                    pv, dsv = [], []
                    for i in range_constexpr(8):
                        p = fx.Float32(ok.select(exp2(fsub(sv[i], lse2[g][i])), fx.Float32(0.0)))
                        pv.append(p)
                        dsv.append(fmul(fmul(p, fsub(dp[g][kk][i], dlt[g][i])), scale))
                    slot_off = (kt * 16 + l16) * SP + (hp * 32 + g * 16 + hi * 8) * 2
                    lds_store(to_bf16(Vec.from_elements(dsv, fx.Float32)), ds_lds + slot_off)
                    lds_store(to_bf16(Vec.from_elements(pv, fx.Float32)), p_lds + slot_off)
            _lds_barrier()
            # ---- dQ^T[d, h] += K^T[d, slot] dS^T[slot, h]: every K^T fragment feeds G WMMAs.
            new = [list(dq[g]) for g in range_constexpr(G)]
            steps = []
            for pc in range_constexpr(PC):
                for j in range_constexpr(DTQ):

                    def ld(pc=pc, j=j):
                        a = _frag_tr(kbase + tr_lane + pc * 32 * ROW + (dt0 + j) * 32, ROW)
                        if const_expr(j == 0):
                            bb = ds_lds + tr_ds + pc * 32 * SP + hp * 64
                            return (*[_frag_tr(bb + g * 32, SP) for g in range_constexpr(G)], a)
                        return (a,)

                    def mma(fr, j=j):
                        if const_expr(j == 0):
                            cur["b"] = fr[:G]
                        for g in range_constexpr(G):
                            new[g][j] = wmma_bf16(fr[-1], cur["b"][g], new[g][j])

                    steps.append((ld, mma))
            if const_expr(NBUF == 1):
                run(steps)
                steps = []
                # Every wave is done with the kv tile: gather the next one under C^T.
                _lds_barrier()
                issue(t + 1, fx.Int32(0))
            # ---- C^T[d, slot] = Q^T dS + dO^T P (all heads) over this wave's d tiles.
            for kt in range_constexpr(KT):
                for hc in range_constexpr(HC):

                    def ld(kt=kt, hc=hc):
                        o = (kt * 16 + l16) * SP + hc * 64 + hi * 16
                        return (_frag(ds_lds + o), _frag(p_lds + o))

                    def keep(fr, kt=kt, hc=hc):
                        cur[("bs", kt, hc)], cur[("bp", kt, hc)] = fr

                    steps.append((ld, keep))
            for j in range_constexpr(DTW):
                dt = wave * DTW + j
                for hc in range_constexpr(HC):

                    def ld(hc=hc, dt=dt, j=j):
                        if const_expr(CT_RES):
                            return ct_a[(j, hc)]
                        o = tr_lane + hc * 32 * ROW + dt * 32
                        return (_frag_tr(q_lds + o, ROW), _frag_tr(do_lds + o, ROW))

                    def mma(fr, hc=hc, dt=dt, j=j):
                        for kt in range_constexpr(KT):
                            c = Vec.filled(8, 0.0, fx.Float32) if hc == 0 else cur[("c", kt)]
                            c = wmma_bf16(fr[0], cur[("bs", kt, hc)], c)
                            c = wmma_bf16(fr[1], cur[("bp", kt, hc)], c)
                            cur[("c", kt)] = c
                            if const_expr(hc == HC - 1 and not STAGE):
                                im_off = (t * BN + kt * 16 + l16) * (DV * 2) + dt * 32 + hi * 16
                                buf_store(to_bf16(c).bitcast(fx.Int32), rsrc_im, im_off, aux=CPOL_NT)
                            if const_expr(hc == HC - 1 and STAGE):
                                cur[("o", kt, j % GW)] = to_bf16(c)
                                if const_expr(j % GW == GW - 1):
                                    for g in range_constexpr(GW):
                                        lds_store(cur[("o", kt, g)], stg + l16 * SROW + g * 32 + hi * 16)
                                    row0 = t * BN + kt * 16 + lane // LPR
                                    col = (dt - (GW - 1)) * 32 + (lane % LPR) * 16
                                    for r in range_constexpr(16 // RPS):
                                        v = lds_load(
                                            stg + (r * RPS + lane // LPR) * SROW + (lane % LPR) * 16, v4i
                                        )
                                        buf_store(v, rsrc_im, (row0 + r * RPS) * (DV * 2) + col, aux=CPOL_NT)

                    steps.append((ld, mma))
            run(steps)
            return new

        init = [Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(G * DTQ)]
        for t, st in range(fx.Index(0), fx.Index(n_tile), fx.Index(1), init=init):
            ti = fx.Int32(t)
            # Tile t landed, and every wave is done with tile t-1 (its kv buffer and dS / P).
            _rocdl.s_wait_tensorcnt(0)
            _lds_barrier()
            if const_expr(NBUF == 2):
                issue(ti + 1, (ti + 1) % 2)
                buf = ti % 2
            else:
                buf = fx.Int32(0)
            dq = [[st[g * DTQ + j] for j in range_constexpr(DTQ)] for g in range_constexpr(G)]
            new = tile_compute(ti, buf, dq)
            res = yield [new[g][j] for g in range_constexpr(G) for j in range_constexpr(DTQ)]
        # The over-issued gather must land before the workgroup's LDS is released.
        _rocdl.s_wait_tensorcnt(0)
        nxt = tok + fx.Int32(ahead)
        nxt = fx.Int32((nxt < fx.Int32(fx.grid_dim.x)).select(nxt, tok))
        _l2_prefetch(tensor_addr(Q) + fx.Int64(nxt) * (H * DQK * 2), H * DQK * 2, tid, BLOCK)
        _l2_prefetch(tensor_addr(DO) + fx.Int64(nxt) * (H * DV * 2), H * DV * 2, tid, BLOCK)
        _l2_prefetch(tensor_addr(O) + fx.Int64(nxt) * (H * DV * 2), H * DV * 2, tid, BLOCK)
        _l2_prefetch(tensor_addr(TOPK) + fx.Int64(nxt) * (topk * 4), topk * 4, tid, BLOCK)
        _l2_prefetch(tensor_addr(LSE) + fx.Int64(nxt) * (H * 4), H * 4, tid, BLOCK)
        # dQ (rope columns zeroed) goes out through the Q / dO area, free once every wave is past
        # its last C^T, so the token's rows leave as one contiguous run.
        gpu.barrier()
        zero = Vec.filled(4, 0, fx.Int32)
        for g in range_constexpr(G):
            row = q_lds + (hp * 32 + g * 16 + l16) * QROW
            for j in range_constexpr(DTQ):
                lds_store(to_bf16(res[g * DTQ + j]), row + (dt0 + j) * 32 + hi * 16)
            for j in range_constexpr(RZ // DS):
                lds_store(zero, row + DV * 2 + hi * (RZ * 16) + ds * (RZ // DS * 16) + j * 16)
        gpu.barrier()
        rsrc_dq = make_rsrc(tensor_addr(DQ) + tok_q, H * DQK * 2)
        for i in range_constexpr(H * QCPR // BLOCK):
            c = tid + i * BLOCK
            buf_store(lds_load(q_lds + (c + c // QCPR) * 16, v4i), rsrc_dq, c * 16)

    @flyc.jit
    def launch_sparse_mla_bwd_gfx1250_w1(
        Q: fx.Tensor,
        KV: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DQ: fx.Tensor,
        DELTA: fx.Tensor,
        INTERM: fx.Tensor,
        NTILE: fx.Tensor,
        n_tok: fx.Int32,
        NKV: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        sparse_mla_bwd_gfx1250_w1_kernel(
            Q, KV, O, DO, TOPK, LSE, DQ, DELTA, INTERM, NTILE, NKV, value_attrs={"rocdl.waves_per_eu": 1}
        ).launch(grid=(fx.Index(n_tok), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    compiled = {}

    def run_(q, kv, o, do, topk_indices, lse, dq, delta, interm, ntile, stream=None):
        args = (q, kv, o, do, topk_indices, lse, dq, delta, interm, ntile, int(q.shape[0]), int(kv.shape[0]))
        fn = compiled.get("fn")
        if fn is None:
            fn = flyc.compile[COMPILE_HINTS](launch_sparse_mla_bwd_gfx1250_w1, *args, fx.Stream(stream))
            compiled["fn"] = fn
        return fn(*args, stream)

    return run_


_SWA_W = 128  # cr0 window: slot s of token t is kv row t - 127 + s
_SWA_HW = 64  # heads per workgroup of the SWA kernel
_SWA_BT = 16  # tokens per SWA workgroup
_SWA_POOL_MAX = 32  # pool slots the SWA kernel's VGPR budget holds (cr128 up to T = 4096)


def _swa_rows(block_t):
    """Rows of a token block's kv window (``block_t + 127``, whole 16-row tiles)."""
    return -(-(block_t + _SWA_W - 1) // 16) * 16


@functools.lru_cache(maxsize=64)
def build_sparse_mla_bwd_gfx1250_swa(
    num_heads,
    scale,
    block_t=16,
    prefetch=None,
    pin=4,
    c_dt=4,
    pool_p=0,
    pool_cr=0,
    pool_la=2,
    bf_hold=True,
    batch=None,
    qgw=None,
):
    """cr0 backward (pure SWA, top-k = the 128-row window ``[t-127, t]`` in slot order).

    A workgroup owns ``block_t`` consecutive tokens and 64 heads (one wave per SIMD, waves split
    as in the w1 kernel). The block's ``block_t + 127`` kv rows are gathered to LDS once; token
    ``i`` reads its 128 rows at offset ``i``. Each token does S / dP, P / dS, dQ written per d tile
    (all 128 slots in one pass), and adds ``C^T = Q^T dS + dO^T P`` into fp32 dKV tiles of the
    whole window held in VGPRs (dS / P rows read shifted by ``i``). The tiles go out once per
    block as a ``[window, 512]`` fp32 partial, summed per kv row by ``build_swa_dkv_reduce``.

    ``pool_p > 0`` adds cr128's causal pool: slots 128 .. 128 + pool_p - 1 are kv rows
    ``T + p``, visible iff ``t >= pool_cr * (p + 1) - 1``. The pool rows are shared by every token,
    so their K / K^T fragments come from global memory (``KVT`` is the pool's K^T); pool dS / P
    take the window P tile's LDS once the window C^T is done, pool C^T accumulates over the block
    in VGPRs (one ``[NPP, 512]`` partial per workgroup in ``PPART``) and dQ runs over the window
    and pool slots in one pass.

    ``batch`` (see ``_run_steps``): 4 without a pool (cr0 -5..-6%); 0 (the ``prefetch`` / ``pin``
    stream) with one, where batching is 18-36% slower at 4K."""
    H, HW, W, BT = num_heads, _SWA_HW, _SWA_W, block_t
    if H % HW:
        raise ValueError(f"SWA backward needs num_heads % {HW} == 0, got {H}")
    NHW = H // HW
    WAVES = 4
    BLOCK = WAVES * WAVE
    HP = HW // 32
    DS = WAVES // HP
    G = 2
    KT = W // 16
    KTW = KT // DS
    PC = W // 32
    KS = DV // 32
    DT = DV // 16
    DTQ = DT // DS
    HC = HW // 32
    DTW = DT // WAVES
    RZ = (DQK - DV) * 2 // 32
    WR = _swa_rows(BT)
    NR = WR // 16
    RPO = 8
    NOP = WR // RPO
    OPW = -(-NOP // WAVES)
    QOP = HW // RPO // WAVES
    SP = HW * 2 + 16
    QO_B = HW * ROW
    DS_B = W * SP
    PPH = BLOCK // HW
    LDS_B = 2 * QO_B + WR * ROW + 2 * DS_B + PPH * HW * 4 + SP
    if LDS_B > _LDS_CAP:
        raise ValueError(f"SWA backward block {BT} does not fit in LDS")
    if DTW % c_dt:
        raise ValueError(f"c_dt {c_dt} must divide {DTW}")
    NACC = NR * DTW
    POOL = pool_p > 0
    # The pool accumulators take ~128 VGPRs: shallower prefetch.
    if prefetch is None:
        prefetch = 3 if POOL else 4
    if batch is None:
        batch = 0 if POOL else 4
    NPP = -(-pool_p // 32) * 32  # pool slots processed (dQ steps of 32)
    NPT = NPP // 16
    KTWP = NPT // DS
    NPC = NPP // 32
    NPACC = NPT * DTW
    if 2 * NPP > W:
        raise ValueError(f"pool of {pool_p} rows does not fit the window P tile")
    # dQ goes out through a per-wave stage of QGW d tiles (the widest that fit; direct fragment
    # stores when none does, ~25% slower). Without a pool: 16 heads in the window P tile (dead
    # during the dQ pass), tiles held in VGPRs until a row is whole. The pool kernel has no VGPRs
    # to hold them (it spills): each tile goes to the stage as it completes, both head groups, in
    # the window dS tile once every wave holds its dS^T fragments (bf_hold).
    QS_ROWS = 16 * (G if POOL else 1)
    qfits = [gw for gw in (8, 4) if DTQ % gw == 0 and WAVES * QS_ROWS * (gw * 32 + 16) <= DS_B]
    if qgw is None:
        qgw = qfits[0] if qfits and (bf_hold or not POOL) else 1
    QGW = qgw
    QSROW = QGW * 32 + 16
    QLPR = QGW * 2  # lanes per staged row (16 B each)
    QRPS = WAVE // QLPR  # rows per buffer_store
    c_log2 = float(scale) * LOG2E
    pin_every = pin if pin > 0 else 1 << 30
    # Row tiles whose shifted dS / P rows can fall outside the 128 slots (read as zeros).
    edge = [r * 16 - (BT - 1) < 0 or r * 16 + 15 >= W for r in range(NR)]

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def sparse_mla_bwd_gfx1250_swa_kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DQ: fx.Tensor,
        DELTA: fx.Tensor,
        PART: fx.Tensor,
        KVT: fx.Tensor,
        PPART: fx.Tensor,
        N_TOK: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = uniform(tid // WAVE)
        hp = wave % HP
        ds = wave // HP
        lane = tid % WAVE
        l16 = lane % 16
        hi = lane // 16
        l8 = lane % 8
        bid = fx.Int32(fx.block_idx.x)
        t0 = (bid // NHW) * BT
        h0 = (bid % NHW) * HW
        n_i = fx.Int32((N_TOK - t0 < fx.Int32(BT)).select(N_TOK - t0, fx.Int32(BT)))

        arena = fx.SharedAllocator(static=False)
        arena.allocate(LDS_B)
        lds0 = fx.Int32(fx.Int64(fx.ptrtoint(arena.base_ptr)))
        q_lds = lds0
        do_lds = q_lds + QO_B
        kw_lds = do_lds + QO_B
        ds_lds = kw_lds + WR * ROW
        p_lds = ds_lds + DS_B
        dl_lds = p_lds + DS_B
        zr_lds = dl_lds + PPH * HW * 4
        qstg = (ds_lds if POOL else p_lds) + wave * (QS_ROWS * QSROW)
        q_ld = qstg + (lane // QLPR) * QSROW + (lane % QLPR) * 16
        q_st = (lane // QLPR) * (DQK * 2) + (lane % QLPR) * 16

        v4i = vec_type(4, fx.Int32)
        v4f = vec_type(4, fx.Float32)
        n_rows = N_TOK * H

        # ---- the block's kv window, once: row r is kv row t0 - 127 + r (out of range -> zeros).
        kv0 = t0 - (W - 1)
        for k in range_constexpr(OPW):
            op = wave + k * WAVES
            op = fx.Int32((op < NOP).select(op, fx.Int32(NOP - 1)))
            r0 = kv0 + op * RPO
            _tdm_gather(kw_lds + op * (RPO * ROW), tensor_addr(KV), [r0 + e for e in range(RPO)], RPO, N_TOK)

        def issue_qdo(tok, ok):
            # Past the last token every row index is negative: nothing is fetched.
            base = fx.Int32(ok.select(tok * H + h0, fx.Int32(-(1 << 30))))
            for k in range_constexpr(QOP):
                op = wave * QOP + k
                r0 = base + op * RPO
                words = [r0 + e for e in range(RPO)]
                _tdm_gather(q_lds + op * (RPO * ROW), tensor_addr(Q), words, RPO, n_rows)
                _tdm_gather(do_lds + op * (RPO * ROW), tensor_addr(DO), words, RPO, n_rows, row_len=DV)

        issue_qdo(t0, fx.Int32(0) == fx.Int32(0))
        lds_store(Vec.filled(4, 0, fx.Int32), zr_lds + (tid % (SP // 16)) * 16)

        a_lane = l16 * ROW + hi * 16
        tr_lane = (hi * 8 + l8) * ROW + (l16 // 8) * 16
        tr_ds = (hi * 8 + l8) * SP + (l16 // 8) * 16
        q_row = [q_lds + (hp * 32 + g * 16) * ROW + a_lane for g in range_constexpr(G)]
        do_row = [do_lds + (hp * 32 + g * 16) * ROW + a_lane for g in range_constexpr(G)]
        kt0 = ds * KTW
        dt0 = ds * DTQ

        def run(steps):
            _run_steps(steps, prefetch, pin_every, batch)

        def sdp(kbase, tok, lse2):
            # Window S / dP -> (P, dP) per (g, kk): lane = slot kt*16 + l16, element i = head row.
            kb_w = kbase + a_lane + kt0 * 16 * ROW
            s = [[Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(KTW)] for _ in range_constexpr(G)]
            dp = [[Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(KTW)] for _ in range_constexpr(G)]
            cur = {}
            steps = []
            for ks in range_constexpr(KS):
                for kk in range_constexpr(KTW):

                    def ld(ks=ks, kk=kk):
                        bk = _frag(kb_w + kk * 16 * ROW + ks * 64)
                        if const_expr(kk == 0):
                            aq = [_frag(q_row[g] + ks * 64) for g in range_constexpr(G)]
                            ad = [_frag(do_row[g] + ks * 64) for g in range_constexpr(G)]
                            return (*aq, *ad, bk)
                        return (bk,)

                    def mma(fr, kk=kk):
                        if const_expr(kk == 0):
                            cur["aq"], cur["ad"] = fr[:G], fr[G : 2 * G]
                        for g in range_constexpr(G):
                            s[g][kk] = wmma_bf16(cur["aq"][g], fr[-1], s[g][kk])
                            dp[g][kk] = wmma_bf16(cur["ad"][g], fr[-1], dp[g][kk])

                    steps.append((ld, mma))
            run(steps)
            pv = []
            for kk in range_constexpr(KTW):
                slot = (kt0 + kk) * 16 + l16
                ok = (tok - (W - 1) + slot) >= fx.Int32(0)
                for g in range_constexpr(G):
                    sv = vmul_scalar(s[g][kk], c_log2)
                    p = [
                        fx.Float32(ok.select(exp2(fsub(sv[i], lse2[g][i])), fx.Float32(0.0)))
                        for i in range(8)
                    ]
                    pv.append((g, slot, p, dp[g][kk]))
            return pv

        def delta_pdp(tok, pv):
            # delta = rowsum(O * dO) taken as rowsum(P * dP) over every slot (the sink adds mass but
            # no value row): this wave's slots (then the 16 slot lanes), then the DS slot-split
            # waves through LDS, in a fixed order so all of them hold the same delta.
            dlt = []
            for g in range_constexpr(G):
                part = []
                for i in range_constexpr(8):
                    x = fx.Float32(0.0)
                    for gg, _, p, d in pv:
                        if const_expr(gg == g):
                            x = fadd(x, fmul(p[i], d[i]))
                    for m in (1, 2, 4, 8):
                        x = fadd(x, _row_xor(x, m))
                    part.append(x)
                hrow = hp * 32 + g * 16 + hi * 8
                o = dl_lds + (ds * HW + hrow) * 4
                lds_store(Vec.from_elements(part[:4], fx.Float32), o)
                lds_store(Vec.from_elements(part[4:], fx.Float32), o + 16)
            gpu.barrier()
            rsrc_dl = make_rsrc(tensor_addr(DELTA) + fx.Int64(tok * H + h0) * 4, HW * 4)
            for g in range_constexpr(G):
                hrow = hp * 32 + g * 16 + hi * 8
                dg = [fx.Float32(0.0)] * 8
                for k in range_constexpr(DS):
                    o = dl_lds + (k * HW + hrow) * 4
                    dlv = cat(lds_load(o, v4f), lds_load(o + 16, v4f))
                    dg = [fadd(dg[e], fx.Float32(dlv[e])) for e in range_constexpr(8)]
                dlt.append(dg)
                off = fx.Int32((l16 == 0).select(hrow * 4, fx.Int32(0x7FFFFFE0)))
                off = fx.Int32((ds == 0).select(off, fx.Int32(0x7FFFFFE0)))
                buf_store(Vec.from_elements(dg[:4], fx.Float32), rsrc_dl, off)
                buf_store(Vec.from_elements(dg[4:], fx.Float32), rsrc_dl, off + 16)
            return dlt

        def ds_p(pv, dlt):
            # bf16 (slot, dS, P) per (g, kk).
            out = []
            for g, slot, p, d in pv:
                dsv = [fmul(fmul(p[i], fsub(d[i], dlt[g][i])), scale) for i in range_constexpr(8)]
                col = (hp * 32 + g * 16 + hi * 8) * 2
                out.append(
                    (
                        slot * SP + col,
                        to_bf16(Vec.from_elements(dsv, fx.Float32)),
                        to_bf16(Vec.from_elements(p, fx.Float32)),
                    )
                )
            return out

        def ct(acc, ii):
            # acc[r * DTW + j] += C^T[d tile wave*DTW + j, window rows r*16..] for token ii: window
            # row r*16 + l16 is slot r*16 + l16 - ii.
            new = list(acc)
            cur = {}
            for half in range_constexpr(DTW // c_dt):
                steps = []
                for j in range_constexpr(c_dt):
                    dt = wave * DTW + half * c_dt + j
                    for hc in range_constexpr(HC):

                        def ld(dt=dt, hc=hc):
                            o = tr_lane + hc * 32 * ROW + dt * 32
                            return (_frag_tr(q_lds + o, ROW), _frag_tr(do_lds + o, ROW))

                        def keep(fr, j=j, hc=hc):
                            cur[("a", j, hc)] = fr

                        steps.append((ld, keep))
                for r in range_constexpr(NR):
                    for hc in range_constexpr(HC):

                        def ld(r=r, hc=hc):
                            slot = fx.Int32(r * 16) + l16 - ii
                            col = hc * 64 + hi * 16
                            a_s = ds_lds + slot * SP + col
                            a_p = p_lds + slot * SP + col
                            if const_expr(edge[r]):
                                z = zr_lds + col
                                bad = fx.Int32((slot < fx.Int32(0)).select(fx.Int32(1), fx.Int32(0)))
                                bad = bad | fx.Int32((slot >= fx.Int32(W)).select(fx.Int32(1), fx.Int32(0)))
                                a_s = fx.Int32((bad != fx.Int32(0)).select(z, a_s))
                                a_p = fx.Int32((bad != fx.Int32(0)).select(z, a_p))
                            return (_frag(a_s), _frag(a_p))

                        def mma(fr, r=r, hc=hc, half=half):
                            for j in range_constexpr(c_dt):
                                k = r * DTW + half * c_dt + j
                                a = cur[("a", j, hc)]
                                new[k] = wmma_bf16(a[0], fr[0], new[k])
                                new[k] = wmma_bf16(a[1], fr[1], new[k])

                        steps.append((ld, mma))
                run(steps)
            return new

        if const_expr(POOL):
            rsrc_kvp = make_rsrc(tensor_addr(KV) + fx.Int64(N_TOK) * (DQK * 2), pool_p * DQK * 2)
            rsrc_kvt = make_rsrc(tensor_addr(KVT), DV * NPP * 2)
            kp0 = ds * KTWP

        def gfrag(rsrc, o, soff):
            # Wave-uniform parts of the offset go to soffset so the constants fold into the
            # instruction offset instead of one hoisted VGPR per fragment.
            lo = Vec(buf_load(rsrc, o, v4i, soff=soff)).bitcast(fx.BFloat16)
            return cat(lo, Vec(buf_load(rsrc, o + 32, v4i, soff=soff)).bitcast(fx.BFloat16))

        def pool_k(kk, ks):
            # Pool K fragment (lane = pool row (kp0 + kk) * 16 + l16): rows past the pool read zeros.
            return gfrag(rsrc_kvp, (kk * 16 + l16) * (DQK * 2) + ks * 64 + hi * 16, kp0 * (16 * DQK * 2))

        def pool_kt(j, pc):
            # Pool K^T fragment (lane = latent column (dt0 + j) * 16 + l16) from the transposed copy.
            o = (j * 16 + l16) * (NPP * 2) + (pc * 32 + hi * 8) * 2
            return gfrag(rsrc_kvt, o, dt0 * (16 * NPP * 2))

        def sdp_pool(tok, lse2):
            # Pool S / dP -> (P, dP) per (g, kk), lane = pool slot.
            s = [[Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(KTWP)] for _ in range_constexpr(G)]
            dp = [[Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(KTWP)] for _ in range_constexpr(G)]
            cur = {}
            steps = []
            for ks in range_constexpr(KS):
                for kk in range_constexpr(KTWP):

                    def ld(ks=ks, kk=kk):
                        bk = pool_k(kk, ks)
                        if const_expr(kk == 0):
                            aq = [_frag(q_row[g] + ks * 64) for g in range_constexpr(G)]
                            ad = [_frag(do_row[g] + ks * 64) for g in range_constexpr(G)]
                            return (*aq, *ad, bk)
                        return (bk,)

                    def mma(fr, kk=kk):
                        if const_expr(kk == 0):
                            cur["aq"], cur["ad"] = fr[:G], fr[G : 2 * G]
                        for g in range_constexpr(G):
                            s[g][kk] = wmma_bf16(cur["aq"][g], fr[-1], s[g][kk])
                            dp[g][kk] = wmma_bf16(cur["ad"][g], fr[-1], dp[g][kk])

                    steps.append((ld, mma))
            run(steps)
            pv = []
            for kk in range_constexpr(KTWP):
                slot = (kp0 + kk) * 16 + l16
                lim = fx.Int32(
                    (slot < fx.Int32(pool_p)).select(slot * pool_cr + (pool_cr - 1), fx.Int32(1 << 30))
                )
                ok = tok >= lim
                for g in range_constexpr(G):
                    sv = vmul_scalar(s[g][kk], c_log2)
                    p = [
                        fx.Float32(ok.select(exp2(fsub(sv[i], lse2[g][i])), fx.Float32(0.0)))
                        for i in range(8)
                    ]
                    pv.append((g, slot, p, dp[g][kk]))
            return pv

        def ct_pool(pacc):
            # pacc[r * DTW + j] += C^T[d tile wave*DTW + j, pool rows r*16..] (no shift).
            new = list(pacc)
            cur = {}
            for half in range_constexpr(DTW // c_dt):
                steps = []
                for j in range_constexpr(c_dt):
                    dt = wave * DTW + half * c_dt + j
                    for hc in range_constexpr(HC):

                        def ld(dt=dt, hc=hc):
                            o = tr_lane + hc * 32 * ROW + dt * 32
                            return (_frag_tr(q_lds + o, ROW), _frag_tr(do_lds + o, ROW))

                        def keep(fr, j=j, hc=hc):
                            cur[("a", j, hc)] = fr

                        steps.append((ld, keep))
                for r in range_constexpr(NPT):
                    for hc in range_constexpr(HC):

                        def ld(r=r, hc=hc):
                            o = (r * 16 + l16) * SP + hc * 64 + hi * 16
                            return (_frag(p_lds + o), _frag(p_lds + NPP * SP + o))

                        def mma(fr, r=r, hc=hc, half=half):
                            for j in range_constexpr(c_dt):
                                k = r * DTW + half * c_dt + j
                                a = cur[("a", j, hc)]
                                new[k] = wmma_bf16(a[0], fr[0], new[k])
                                new[k] = wmma_bf16(a[1], fr[1], new[k])

                        steps.append((ld, mma))
                run(steps)
            return new

        def dq_pass(kbase, tok):
            rsrc_dq = make_rsrc(tensor_addr(DQ) + fx.Int64(tok * H + h0) * (DQK * 2), HW * DQK * 2)
            NS = PC + NPC

            def b_frags(pc):
                # dS^T fragments of slot step pc (window slots, then the pool's).
                base = ds_lds + tr_ds + hp * 64 + pc * 32 * SP
                if const_expr(pc >= PC):
                    base = p_lds + tr_ds + hp * 64 + (pc - PC) * 32 * SP
                return [_frag_tr(base + g * 32, SP) for g in range_constexpr(G)]

            bf = [b_frags(pc) for pc in range_constexpr(NS)] if bf_hold else None
            if const_expr(POOL and QGW > 1):
                gpu.barrier()
            dq_off = [(hp * 32 + g * 16 + l16) * (DQK * 2) + hi * 16 for g in range_constexpr(G)]
            order = [(j, pc) for j in range_constexpr(DTQ) for pc in range_constexpr(NS)]
            # Pool K^T comes from global memory: its loads run pool_la d tiles ahead.
            pk = [(j, pc - PC) for j, pc in order if pc >= PC]
            la = min(pool_la * NPC, len(pk))
            kq = [pool_kt(*k) for k in pk[:la]]
            cur = {}
            steps = []
            for j, pc in order:

                def ld(j=j, pc=pc):
                    bq = () if bf_hold else tuple(b_frags(pc))
                    if const_expr(pc < PC):
                        return (_frag_tr(kbase + tr_lane + pc * 32 * ROW + (dt0 + j) * 32, ROW), *bq)
                    i = cur.get("pn", 0)
                    cur["pn"] = i + 1
                    if const_expr(i + la < len(pk)):
                        kq.append(pool_kt(*pk[i + la]))
                    return (kq.pop(0), *bq)

                def mma(fr, j=j, pc=pc):
                    for g in range_constexpr(G):
                        c = Vec.filled(8, 0.0, fx.Float32) if pc == 0 else cur[(j, g)]
                        cur[(j, g)] = wmma_bf16(fr[0], bf[pc][g] if bf_hold else fr[1 + g], c)
                        if const_expr(pc == NS - 1 and QGW == 1):
                            buf_store(
                                to_bf16(cur[(j, g)]).bitcast(fx.Int32),
                                rsrc_dq,
                                dq_off[g] + j * 32,
                                soff=dt0 * 32,
                            )
                        elif const_expr(pc == NS - 1 and POOL):
                            o = qstg + (g * 16 + l16) * QSROW + (j % QGW) * 32 + hi * 16
                            lds_store(to_bf16(cur[(j, g)]), o)
                        elif const_expr(pc == NS - 1):
                            cur[("o", g, j % QGW)] = to_bf16(cur[(j, g)])
                    if const_expr(QGW > 1 and pc == NS - 1 and j % QGW == QGW - 1):
                        # 16 heads x QGW d tiles through the stage, out as rows of QGW * 32 B.
                        col = (dt0 + j - (QGW - 1)) * 32
                        for g in range_constexpr(G):
                            if const_expr(POOL):
                                # One lane-varying address each side, the rest in immediates /
                                # soffset: hoisted per-store address VGPRs spill here.
                                for r in range_constexpr(16 // QRPS):
                                    v = lds_load(q_ld + (g * 16 + r * QRPS) * QSROW, v4i)
                                    soff = (hp * 32 + g * 16 + r * QRPS) * (DQK * 2) + col
                                    buf_store(v, rsrc_dq, q_st, soff=soff)
                            else:
                                col_l = col + (lane % QLPR) * 16
                                for jj in range_constexpr(QGW):
                                    lds_store(cur[("o", g, jj)], qstg + l16 * QSROW + jj * 32 + hi * 16)
                                for r in range_constexpr(16 // QRPS):
                                    row = r * QRPS + lane // QLPR
                                    v = lds_load(qstg + row * QSROW + (lane % QLPR) * 16, v4i)
                                    buf_store(v, rsrc_dq, (hp * 32 + g * 16 + row) * (DQK * 2) + col_l)

                steps.append((ld, mma))
            run(steps)
            zero = Vec.filled(4, 0, fx.Int32)
            for g in range_constexpr(G):
                rz_off = dq_off[g] - hi * 16 + DV * 2 + hi * (RZ * 16) + ds * (RZ // DS * 16)
                for j in range_constexpr(RZ // DS):
                    buf_store(zero, rsrc_dq, rz_off + j * 16)

        init = [Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(NACC + NPACC)]
        for i, st in range(fx.Index(0), fx.Index(n_i), fx.Index(1), init=init):
            ii = fx.Int32(i)
            tok = t0 + ii
            acc = [st[k] for k in range_constexpr(NACC)]
            pacc = [st[NACC + k] for k in range_constexpr(NPACC)]
            # Q / dO of this token (and on the first pass the kv window) landed.
            _rocdl.s_wait_tensorcnt(0)
            gpu.barrier()

            rsrc_lse = make_rsrc(tensor_addr(LSE) + fx.Int64(tok * H + h0) * 4, HW * 4)
            lse2 = []
            for g in range_constexpr(G):
                hrow = hp * 32 + g * 16 + hi * 8
                lsev = cat(
                    Vec(buf_load(rsrc_lse, hrow * 4, v4f)), Vec(buf_load(rsrc_lse, hrow * 4 + 16, v4f))
                )
                lg = []
                for e in range_constexpr(8):
                    l_ = fx.Float32(lsev[e])
                    # No visible key and no sink: lse = -inf, P must be 0.
                    lg.append(
                        fx.Float32(
                            (l_ == fx.Float32(float("-inf"))).select(
                                fx.Float32(float("inf")), fmul(l_, LOG2E)
                            )
                        )
                    )
                lse2.append(lg)

            kbase = kw_lds + ii * ROW
            ppv = sdp_pool(tok, lse2) if POOL else []
            pv = sdp(kbase, tok, lse2)
            dlt = delta_pdp(tok, pv + ppv)
            for off, dsv, p in ds_p(pv, dlt):
                lds_store(dsv, ds_lds + off)
                lds_store(p, p_lds + off)
            # Pool dS / P wait in registers until the window C^T frees the P tile.
            pool_dsp = ds_p(ppv, dlt)
            gpu.barrier()
            acc = ct(acc, ii)
            gpu.barrier()
            if const_expr(POOL):
                # Pool dS rows 0..NPP and P rows NPP..2*NPP go to the (dead) window P tile.
                for off, dsv, p in pool_dsp:
                    lds_store(dsv, p_lds + off)
                    lds_store(p, p_lds + NPP * SP + off)
                gpu.barrier()
                pacc = ct_pool(pacc)
                gpu.barrier()
            # Every wave is done with Q / dO: the next token's land under dQ.
            issue_qdo(tok + fx.Int32(1), ii + fx.Int32(1) < n_i)
            dq_pass(kbase, tok)
            res = yield acc + pacc
        _rocdl.s_wait_tensorcnt(0)

        if const_expr(POOL):
            rsrc_pp = make_rsrc(tensor_addr(PPART) + fx.Int64(bid) * (NPP * DV * 4), NPP * DV * 4)
            for r in range_constexpr(NPT):
                for j in range_constexpr(DTW):
                    v = Vec(res[NACC + r * DTW + j])
                    off = (r * 16 + l16) * (DV * 4) + ((wave * DTW + j) * 16 + hi * 8) * 4
                    buf_store(v.shuffle(v, [0, 1, 2, 3]), rsrc_pp, off)
                    buf_store(v.shuffle(v, [4, 5, 6, 7]), rsrc_pp, off + 16)

        # ---- this block's fp32 partial: lane holds window row r*16 + l16, d = dt*16 + hi*8 + [0, 8).
        rsrc_pt = make_rsrc(tensor_addr(PART) + fx.Int64(bid) * (WR * DV * 4), WR * DV * 4)
        for r in range_constexpr(NR):
            for j in range_constexpr(DTW):
                v = Vec(res[r * DTW + j])
                off = (r * 16 + l16) * (DV * 4) + ((wave * DTW + j) * 16 + hi * 8) * 4
                buf_store(v.shuffle(v, [0, 1, 2, 3]), rsrc_pt, off)
                buf_store(v.shuffle(v, [4, 5, 6, 7]), rsrc_pt, off + 16)

    @flyc.jit
    def launch_sparse_mla_bwd_gfx1250_swa(
        Q: fx.Tensor,
        KV: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DQ: fx.Tensor,
        DELTA: fx.Tensor,
        PART: fx.Tensor,
        KVT: fx.Tensor,
        PPART: fx.Tensor,
        n_tok: fx.Int32,
        n_prog: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        sparse_mla_bwd_gfx1250_swa_kernel(
            Q, KV, O, DO, LSE, DQ, DELTA, PART, KVT, PPART, n_tok, value_attrs={"rocdl.waves_per_eu": 1}
        ).launch(grid=(fx.Index(n_prog), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    compiled = {}

    def run_(q, kv, o, do, lse, dq, delta, part, kvt=None, ppart=None, stream=None):
        n_tok = int(q.shape[0])
        kvt = part if kvt is None else kvt
        ppart = part if ppart is None else ppart
        args = (q, kv, o, do, lse, dq, delta, part, kvt, ppart, n_tok, -(-n_tok // BT) * NHW)
        fn = compiled.get("fn")
        if fn is None:
            fn = flyc.compile[COMPILE_HINTS](launch_sparse_mla_bwd_gfx1250_swa, *args, fx.Stream(stream))
            compiled["fn"] = fn
        return fn(*args, stream)

    return run_


@functools.lru_cache(maxsize=16)
def build_swa_dkv_reduce(num_heads, block_t=16, pool_p=0):
    """bf16 dKV ``[NKV + pool_p, 576]`` from the SWA kernel's partials: kv row j < NKV sums window
    row ``j + 127 - b * block_t`` of every block b (and head group) whose window holds it; rope
    columns are zero. One workgroup per kv row, 4 columns per thread.

    Pool row p (kv row NKV + p) sums its ``[NPP, 512]`` pool partial over every workgroup of the
    SWA kernel: PCS workgroups per row, one per column slice, whose thread groups sum interleaved
    partials (PU loads in flight) and then each other in a fixed order."""
    NHW, W, BT = num_heads // _SWA_HW, _SWA_W, block_t
    WR = _swa_rows(BT)
    NBW = (WR - 1) // BT + 1  # blocks whose window can hold a row
    BLOCK = DV // 4
    RZ = (DQK - DV) * 2 // 16
    POOL = pool_p > 0
    NPP = -(-pool_p // 32) * 32
    PCS = 8
    PQ = DV // PCS // 4  # column quads per slice
    PG = BLOCK // PQ  # thread groups
    PU = 16

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def swa_dkv_reduce_kernel(PART: fx.Tensor, PPART: fx.Tensor, DKV: fx.Tensor, NKV: fx.Int32, NB: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        j = fx.Int32(fx.block_idx.x)
        v4f = vec_type(4, fx.Float32)
        zero = Vec.filled(4, 0, fx.Int32)
        if const_expr(POOL):
            arena = fx.SharedAllocator(static=False)
            arena.allocate(BLOCK * 16)
            red = fx.Int32(fx.Int64(fx.ptrtoint(arena.base_ptr)))
        if j < NKV:
            rsrc_pt = make_rsrc(tensor_addr(PART), NB * (NHW * WR * DV * 4))
            b_hi = (j + (W - 1)) // BT
            acc = [fx.Float32(0.0)] * 4
            for m in range_constexpr(NBW):
                b = b_hi - m
                r = j + (W - 1) - b * BT
                ok = fx.Int32((b >= fx.Int32(0)).select(fx.Int32(1), fx.Int32(0)))
                ok = ok & fx.Int32((b < NB).select(fx.Int32(1), fx.Int32(0)))
                ok = ok & fx.Int32((r < fx.Int32(WR)).select(fx.Int32(1), fx.Int32(0)))
                for hw in range_constexpr(NHW):
                    off = ((b * NHW + hw) * WR + r) * (DV * 4) + tid * 16
                    off = fx.Int32((ok != fx.Int32(0)).select(off, fx.Int32(0x7FFFFFF0)))
                    v = Vec(buf_load(rsrc_pt, off, v4f))
                    acc = [fadd(acc[e], fx.Float32(v[e])) for e in range_constexpr(4)]
            rsrc_o = make_rsrc(tensor_addr(DKV) + fx.Int64(j) * (DQK * 2), DQK * 2)
            buf_store(to_bf16(Vec.from_elements(acc, fx.Float32)).bitcast(fx.Int32), rsrc_o, tid * 8)
            buf_store(
                zero, rsrc_o, fx.Int32((tid < fx.Int32(RZ)).select(DV * 2 + tid * 16, fx.Int32(0x7FFFFFF0)))
            )
        if const_expr(POOL):
            if j >= NKV:
                p = (j - NKV) // PCS
                col = ((j - NKV) % PCS) * (PQ * 4) + (tid % PQ) * 4
                grp = tid // PQ
                n_part = NB * NHW
                rsrc_pp = make_rsrc(tensor_addr(PPART), n_part * (NPP * DV * 4))
                n_it = (n_part + (PG * PU - 1)) // (PG * PU)
                init = [fx.Float32(0.0)] * 4
                for it, st in range(fx.Index(0), fx.Index(n_it), fx.Index(1), init=init):
                    a = [st[e] for e in range_constexpr(4)]
                    vs = []
                    for u in range_constexpr(PU):
                        b = (fx.Int32(it) * PU + u) * PG + grp
                        off = (b * NPP + p) * (DV * 4) + col * 4
                        off = fx.Int32((b < n_part).select(off, fx.Int32(0x7FFFFFF0)))
                        vs.append(Vec(buf_load(rsrc_pp, off, v4f)))
                    for v in vs:
                        a = [fadd(a[e], fx.Float32(v[e])) for e in range_constexpr(4)]
                    res = yield a
                lds_store(Vec.from_elements([res[e] for e in range_constexpr(4)], fx.Float32), red + tid * 16)
                gpu.barrier()
                s = [fx.Float32(0.0)] * 4
                for g in range_constexpr(PG):
                    v = lds_load(red + (g * PQ + tid % PQ) * 16, v4f)
                    s = [fadd(s[e], fx.Float32(v[e])) for e in range_constexpr(4)]
                rsrc_o = make_rsrc(tensor_addr(DKV) + fx.Int64(NKV + p) * (DQK * 2), DQK * 2)
                off = fx.Int32((grp == fx.Int32(0)).select(col * 2, fx.Int32(0x7FFFFFF0)))
                buf_store(to_bf16(Vec.from_elements(s, fx.Float32)).bitcast(fx.Int32), rsrc_o, off)
                zoff = fx.Int32((tid < fx.Int32(RZ)).select(DV * 2 + tid * 16, fx.Int32(0x7FFFFFF0)))
                zoff = fx.Int32(((j - NKV) % PCS == fx.Int32(0)).select(zoff, fx.Int32(0x7FFFFFF0)))
                buf_store(zero, rsrc_o, zoff)

    @flyc.jit
    def launch_swa_dkv_reduce(
        PART: fx.Tensor,
        PPART: fx.Tensor,
        DKV: fx.Tensor,
        n_kv: fx.Int32,
        n_b: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        swa_dkv_reduce_kernel(PART, PPART, DKV, n_kv, n_b).launch(
            grid=(fx.Index(n_kv + pool_p * PCS), 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    compiled = {}

    def run_(part, dkv, ppart=None, stream=None):
        ppart = part if ppart is None else ppart
        args = (part, ppart, dkv, int(dkv.shape[0]) - pool_p, int(part.shape[0]) // NHW)
        fn = compiled.get("fn")
        if fn is None:
            fn = flyc.compile[COMPILE_HINTS](launch_swa_dkv_reduce, *args, fx.Stream(stream))
            compiled["fn"] = fn
        return fn(*args, stream)

    return run_


@functools.lru_cache(maxsize=None)
def _num_cus(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.lru_cache(maxsize=None)
def _side_stream(device):
    return torch.cuda.Stream(device)


def _inverted_topk(topk_indices, num_kv):
    """CSR inverse of the top-k map: ``inv_data`` lists flat ``(tok, slot)`` indices grouped by kv
    row in stable order (invalid entries first, skipped by ``inv_ptr[0]``), ``keys`` the kv row
    of each listed entry. Built from the sorted keys alone, so no host sync."""
    flat = topk_indices.reshape(-1)
    if num_kv < torch.iinfo(torch.int16).max:
        flat = flat.to(torch.int16)  # half the radix passes
    keys, inv_data = torch.sort(flat, stable=True)
    bounds = torch.arange(num_kv + 1, device=keys.device, dtype=keys.dtype)
    inv_ptr = torch.searchsorted(keys, bounds, out_int32=True)
    return keys, inv_ptr, inv_data


def _inverted_topk_chunks(topk_indices, num_kv, chunk):
    """``_inverted_topk`` of every ``chunk``-token slice from one sort: the key of a valid entry
    is ``c * num_kv + kv row`` for its chunk ``c``, so chunk ``c``'s rows are
    ``inv_ptr[c * num_kv : (c + 1) * num_kv + 1]`` (in the same stable order as a per-chunk sort)
    and ``inv_data`` holds flat ``(tok, slot)`` indices of the whole map."""
    from primus_turbo.triton.attention.sparse_mla_dkv_gather import chunk_keys

    n_chunks = -(-topk_indices.shape[0] // chunk)
    dt = torch.int16 if n_chunks * num_kv < torch.iinfo(torch.int16).max else torch.int32
    keys, inv_data = torch.sort(chunk_keys(topk_indices, num_kv, chunk, dt), stable=True)
    bounds = torch.arange(n_chunks * num_kv + 1, device=keys.device, dtype=keys.dtype)
    inv_ptr = torch.searchsorted(keys, bounds, out_int32=True)
    return keys, inv_ptr, inv_data


def sparse_mla_bwd_gfx1250(
    q,
    kv,
    o,
    do,
    topk_indices,
    lse,
    attn_sink=None,
    kv_lora_rank=512,
    scale=None,
    stream=None,
    interm_bytes=_INTERM_BYTES,
    use_w1=True,
    use_swa=True,
):
    """``(dq, dkv, dsink)``; dkv is shaped like ``kv`` and dsink is ``None`` without a sink.

    Tokens are processed in chunks whose bf16 ``[tokens, topk, 512]`` intermediate fits in
    ``interm_bytes`` (no chunking when ``None``)."""
    # Deferred: primus_turbo.triton imports primus_turbo.pytorch, which imports this module.
    from primus_turbo.triton.attention.sparse_mla_dkv_gather import dkv_gather_acc
    from primus_turbo.triton.attention.sparse_mla_dsink import sparse_mla_dsink

    assert kv_lora_rank == DV and q.shape[-1] == DQK, "gfx1250 sparse-MLA is the 512 + 64 latent layout"
    for x in (q, o, do, topk_indices, lse):
        assert x.is_contiguous()
    T_, H, _ = q.shape
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / DQK**0.5
    kv_shape = kv.shape
    kv2 = kv.reshape(kv.shape[0], DQK)
    assert kv2.is_contiguous()
    num_kv = kv2.shape[0]
    dq = torch.empty_like(q)
    delta = torch.empty(T_, H, dtype=torch.float32, device=q.device)
    dsink = None
    # Closed-form bands (same tests as the forward): cr0 = SWA window only; cr128 = window + causal
    # pool of P = num_kv - T rows (kv row T + p visible iff t >= cr * (p + 1) - 1, cr = T / P).
    # No intermediate / inverse. Shapes only, the indices are not read: other indices with these
    # shapes (e.g. packed sequences) get wrong results; see ``sparse_mla_func``.
    pool_p = num_kv - T_
    if pool_p > 0 and not (T_ % pool_p == 0 and T_ // pool_p >= 64 and topk >= _SWA_W + pool_p):
        pool_p = -1
    if pool_p == 0 and topk != _SWA_W:
        pool_p = -1
    npp = -(-pool_p // 32) * 32 if pool_p >= 0 else -1
    n_part = -(-T_ // _SWA_BT) * (H // _SWA_HW)
    part_bytes = n_part * (_swa_rows(_SWA_BT) + npp) * DV * 4
    if (
        use_swa
        and T_ > 0
        and 0 <= npp <= _SWA_POOL_MAX
        and H % _SWA_HW == 0
        and kv.dtype == torch.bfloat16
        and (interm_bytes is None or part_bytes <= interm_bytes)
        # Each workgroup walks its 16 tokens serially: with too few of them the GPU idles.
        and 2 * n_part >= _num_cus(q.device)
    ):
        part = torch.empty(n_part, _swa_rows(_SWA_BT), DV, dtype=torch.float32, device=q.device)
        kvt = ppart = None
        if pool_p > 0:
            kvt = torch.zeros(DV, npp, dtype=kv.dtype, device=q.device)
            kvt[:, :pool_p] = kv2[T_:, :DV].t()
            ppart = torch.empty(n_part, npp, DV, dtype=torch.float32, device=q.device)
        build_sparse_mla_bwd_gfx1250_swa(
            H, float(scale), block_t=_SWA_BT, pool_p=max(pool_p, 0), pool_cr=T_ // pool_p if pool_p > 0 else 0
        )(q, kv2, o, do, lse, dq, delta, part, kvt, ppart, stream=stream)
        dkv = torch.empty(num_kv, DQK, dtype=kv.dtype, device=q.device)
        build_swa_dkv_reduce(H, block_t=_SWA_BT, pool_p=max(pool_p, 0))(part, dkv, ppart, stream=stream)
        if attn_sink is not None:
            dsink = sparse_mla_dsink(attn_sink, lse, delta)
        return dq, dkv.view(kv_shape), dsink
    # One workgroup per token and per CU: chunks are whole multiples of the CU count.
    cus = _num_cus(q.device)
    units = -(-T_ // cus)
    n_chunks = 1 if interm_bytes is None else -(-T_ * topk * DV * 2 // interm_bytes)
    n_chunks = max(1, min(n_chunks, units))
    chunk = -(-units // n_chunks) * cus
    # Two intermediates when chunked: chunk c's inverse + gather (side stream) run under chunk
    # c + 1's main kernel and fill its tail.
    interms = [
        torch.empty(chunk, topk, DV, dtype=torch.bfloat16, device=q.device) for _ in range(min(n_chunks, 2))
    ]
    dkv_acc = torch.zeros(num_kv, DQK, dtype=torch.float32, device=q.device)
    if T_ > 0:
        w1 = use_w1 and H % 32 == 0 and topk % _w1_block_k(H) == 0
        if w1:
            run = build_sparse_mla_bwd_gfx1250_w1(H, topk, float(scale), ahead=cus)
            # Tiles up to each token's last valid slot (slots past it are all -1).
            bn = _w1_block_k(H)
            slot = torch.arange(1, topk + 1, dtype=torch.int32, device=q.device)
            ntile = ((topk_indices >= 0) * slot).amax(1).add_(bn - 1).div_(bn, rounding_mode="floor").int()
        else:
            run = build_sparse_mla_bwd_gfx1250(H, topk, float(scale))
        main = torch.cuda.current_stream(q.device) if stream is None else stream
        side = _side_stream(q.device)
        # The main kernel holds every CU, so side-stream work is not free: when the chunk keys fit
        # int16, one sort for all chunks costs about half of the per-chunk sorts.
        joint = 1 < n_chunks and n_chunks * num_kv < torch.iinfo(torch.int16).max
        if joint:
            side.wait_stream(main)
            with torch.cuda.stream(side):
                keys, inv_ptr_all, inv_data = _inverted_topk_chunks(topk_indices, num_kv, chunk)
        freed = [None] * len(interms)
        for c, t0 in enumerate(range(0, T_, chunk)):
            t1 = min(t0 + chunk, T_)
            b = c % len(interms)
            if freed[b] is not None:
                main.wait_event(freed[b])
            idx = topk_indices[t0:t1]
            im = interms[b][: t1 - t0]
            args = (q[t0:t1], kv2, o[t0:t1], do[t0:t1], idx, lse[t0:t1], dq[t0:t1], delta[t0:t1], im)
            run(*args, *((ntile[t0:t1],) if w1 else ()), stream=main)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                if joint:
                    dkv_gather_acc(
                        im,
                        keys,
                        inv_ptr_all[c * num_kv :],
                        inv_data,
                        dkv_acc,
                        d=DV,
                        seg=_GATHER_SEG,
                        num_kv=num_kv,
                        max_entries=(t1 - t0) * topk,
                        row_base=c * num_kv,
                        ent_base=t0 * topk,
                    )
                else:
                    keys, inv_ptr, inv_data = _inverted_topk(idx, num_kv)
                    dkv_gather_acc(im, keys, inv_ptr, inv_data, dkv_acc, d=DV, seg=_GATHER_SEG)
                freed[b] = torch.cuda.Event()
                freed[b].record(side)
        main.wait_stream(side)
    if attn_sink is not None:
        dsink = sparse_mla_dsink(attn_sink, lse, delta)
    return dq, dkv_acc.to(kv.dtype).view(kv_shape), dsink
