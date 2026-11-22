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

"""Warp-specialized (ping-pong) variant of the w1 sparse-MLA backward kernel (gfx1250, H = 64).

Same contract and the same results bit for bit as ``build_sparse_mla_bwd_gfx1250_w1`` (with
``qdo_tdm``, no L2 prefetch): one workgroup per token, kv tiles of 64 slots by TDM gather, a bf16
``[T, topk, 512]`` dKV intermediate.

The w1 kernel runs every phase of a tile on all four waves, one after the other: S / dP (LDS
bound), softmax, dQ^T (LDS bound), C^T (WMMA bound, plus the intermediate stores). Here eight
waves (two per SIMD) split them into two roles that run one tile apart:

* score waves 0-3 (same partition as w1): S = Q K^T and dP = dO K^T, P / dS to LDS, dQ^T += K^T dS^T,
  and the kv tile gathers;
* C^T waves 4-7: C^T = Q^T dS + dO^T P over 128 latent columns each (Q^T / dO^T fragments resident),
  staged per 16 slots through LDS and stored to the intermediate. The WMMAs of a 16-slot block run in
  an order without back-to-back dependences, and the previous block's stage / stores go out behind
  them (``ctp``; without it this role was the critical one: 2 us per tile alone).

Step ``s`` runs the score phases of tile ``s`` next to the C^T of tile ``s - 1``, so the SIMDs
overlap an LDS-bound phase with a WMMA-bound one. P / dS are double-buffered. Two workgroup
barriers per step: X (tile landed; the P / dS buffer being overwritten is free) and Y (P / dS of
tile ``s`` complete before dQ^T). C^T waves signal Y right after X and wait for it only at the end
of their step (split barrier), so they never stall on the score waves' softmax. Each role keeps
its own registers (<= 512 VGPRs, two waves per SIMD): the dQ^T accumulators live only in the score
waves, the C^T operands only in the C^T waves.

Every output element is accumulated in the same order as in w1 (the delta partials use w1's
128-thread split), so dq / delta / interm are bit-identical.

CSA shape (T 4096, H 64, topk 640, L2 flushed): w1 ~530 -> ~440 us; the whole backward call ~755 -> ~670 us.
Waves go round-robin over the SIMDs, so waves w and w + 4 share one: an even / odd role split
(``interleave``) puts two score waves on a SIMD and is slower (~490 us). Storing C^T straight from
the WMMA layout (``stage=False``: 32 B row pieces) doubles the time.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .fwd_kernel import COMPILE_HINTS, DQK, DV, ROW, _if_wave, _l2_prefetch, _sbuf_load_v8i32, _tdm_gather
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
    tensor_addr,
    to_bf16,
    uniform,
    vec_type,
    vmul_scalar,
    wmma_bf16,
)

_LDS_CAP = 320 * 1024


def _frag(addr):
    v8e = vec_type(8, fx.BFloat16)
    return cat(lds_load(addr, v8e), lds_load(addr + 32, v8e))


def _frag_tr(addr, pitch):
    return cat(lds_load_tr16(addr), lds_load_tr16(addr + 16 * pitch))


def _run_steps(steps, prefetch, pin, batch=0):
    """``(load, mma)`` steps, loads ``prefetch`` steps ahead (or in batches), as in bwd_kernel."""
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
    fx.rocdl.s_wait_dscnt(0)
    fx.rocdl.s_barrier_signal(-1)
    fx.rocdl.s_barrier_wait(-1)


@functools.lru_cache(maxsize=32)
def build_sparse_mla_bwd_gfx1250_pp(
    num_heads, topk, scale, batch_a=2, batch_q=4, prefetch_c=1, ablate="", dqd=True, wpe=2, st_group=8, stage=True,
    interleave=False,
    l2pf=0,
    ctp=True,
):
    """Probes / options (the defaults are the tuned kernel):
    ``dqd``: the score waves split dQ^T by latent columns (8 d tiles x 64 heads each) instead of
    w1's (32 heads x 16 d tiles): each K^T fragment feeds four WMMAs, not two (same sums).
    ``ablate`` (timing probes only, wrong results): sdp, dq, ct, st.
    ``l2pf = a > 0``: during step 0 (nothing to do yet) the C^T waves prefetch into L2 the Q / dO / O / indices / lse
    of token ``tok + a`` (the token that takes this CU next: one workgroup per CU, same XCD)."""
    H = num_heads
    if H != 64:
        raise ValueError("the ping-pong kernel is built for 64 heads")
    AW = 4  # score waves
    BW = 4  # C^T waves
    WAVES = AW + BW
    BLOCK = WAVES * WAVE
    ABLOCK = AW * WAVE
    HP = H // 32
    DS = AW // HP
    BN = 64
    KT = BN // 16
    if topk % BN:
        raise ValueError(f"topk {topk} must be a multiple of {BN}")
    G = 2
    KTW = KT // DS
    PC = BN // 32
    KS = DV // 32
    DT = DV // 16
    DTQ = DT // DS
    HC = H // 32
    DTW = DT // BW
    DTA = DT // AW  # dqd: d tiles per score wave
    GQ = H // 16  # dqd: head groups per score wave
    RZ = (DQK - DV) * 2 // 32
    SP = H * 2 + 16
    QO_B = H * ROW
    TILE_B = BN * ROW
    DS_B = BN * SP
    PPH = ABLOCK // H  # delta partials per head: w1's split (same sums)
    NBUF = 2
    VLD_B = 3 * BN * 4
    GW = DTW
    SROW = GW * 32 + 16
    STG_B = 16 * SROW
    LPR = GW * 2
    RPS = WAVE // LPR
    STG = max(1, st_group)
    RPO = 8
    OPW = BN // RPO // AW
    CPR = DV * 2 // 16
    CPP = CPR // PPH
    QCPR = DQK * 2 // 16
    QROW = DQK * 2 + 16
    NQS = H * QCPR // BLOCK
    if (H * QCPR) % BLOCK or H * QROW > 2 * QO_B or (H // RPO) % WAVES:
        raise ValueError("bad head count")
    total = 2 * QO_B + 2 * (2 * DS_B) + PPH * H * 4 + NBUF * TILE_B + BW * STG_B + VLD_B
    if total > _LDS_CAP:
        raise ValueError(f"ping-pong LDS {total} > {_LDS_CAP}")
    c_log2 = float(scale) * LOG2E
    NACC = (GQ * DTA) if dqd else (G * DTQ)

    @flyc.kernel(name="sparse_mla_bwd_gfx1250_pp_kernel", known_block_size=[BLOCK, 1, 1])
    def sparse_mla_bwd_gfx1250_pp_kernel(
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
        lane = tid % WAVE
        l16 = lane % 16
        hi = lane // 16
        l8 = lane % 8
        tok = fx.Int32(fx.block_idx.x)
        rsrc_nt = make_rsrc(tensor_addr(NTILE), (tok + fx.Int32(1)) * 4)
        n_tile = uniform(fx.Int32(buf_load(rsrc_nt, tok * 4, T.i32)))
        last_t = fx.Int32((n_tile > fx.Int32(0)).select(n_tile - fx.Int32(1), fx.Int32(0)))

        arena = fx.SharedAllocator(static=False)
        arena.allocate(total)
        lds0 = fx.Int32(fx.Int64(fx.ptrtoint(arena.base_ptr)))
        q_lds = lds0
        do_lds = lds0 + QO_B
        dsp0 = lds0 + 2 * QO_B  # 2 sets of (dS, P)
        dl_lds = dsp0 + 4 * DS_B
        kv_lds = dl_lds + PPH * H * 4
        stg0 = kv_lds + NBUF * TILE_B
        vld0 = stg0 + BW * STG_B

        v4i = vec_type(4, fx.Int32)
        v4f = vec_type(4, fx.Float32)
        tok_q = fx.Int64(tok) * (H * DQK * 2)
        rsrc_idx = make_rsrc(tensor_addr(TOPK) + fx.Int64(tok) * (topk * 4), topk * 4)
        kv_addr = tensor_addr(KV)
        # Roles: waves 0-3 score / 4-7 C^T (a SIMD holds waves w and w + 4 if waves go round-robin over
        # SIMDs); ``interleave``: even waves score, odd waves C^T.
        if const_expr(interleave):
            is_c = (wave % 2) == fx.Int32(1)
            aw = wave // 2
        else:
            is_c = wave >= fx.Int32(AW)
            aw = fx.Int32(is_c.select(wave - fx.Int32(AW), wave))
        a_tr_lane = (hi * 8 + l8) * ROW + (l16 // 8) * 16
        tr_ds = (hi * 8 + l8) * SP + (l16 // 8) * 16
        a_lane = l16 * ROW + hi * 16

        # score-wave helpers (wave < AW)
        def load_octs(t, static_tile=None):
            if const_expr(static_tile is not None):
                ct = fx.Int32(min(static_tile, topk // BN - 1))
            else:
                ct = fx.Int32((t < n_tile).select(t, last_t))
            base = ct * BN + aw * (OPW * RPO)
            out = []
            for k in range_constexpr(OPW):
                oct_ = _sbuf_load_v8i32(rsrc_idx, (base + k * RPO) * 4)
                out += [oct_[i] for i in range_constexpr(8)]
            return out

        def issue_octs(octs, buf):
            for k in range_constexpr(OPW):
                words = octs[k * 8 : (k + 1) * 8]
                _tdm_gather(kv_lds + buf * TILE_B + (aw * OPW + k) * (RPO * ROW), kv_addr, words, RPO, NKV)
                va = vld0 + (buf * BN + (aw * OPW + k) * RPO) * 4
                dump = vld0 + NBUF * BN * 4 + aw * (OPW * 32) + k * 32
                va = fx.Int32((lane == fx.Int32(0)).select(va, dump))
                lds_store(Vec.from_elements(words[0:4], fx.Int32), va)
                lds_store(Vec.from_elements(words[4:8], fx.Int32), va + 16)

        # ---- Q / dO / O rows by TDM gather (one 8-row op of each per wave); tile 0 by the score waves.
        n_rows_all = fx.Int32(fx.grid_dim.x) * H
        for k in range_constexpr(H // RPO // WAVES):
            op = wave * (H // RPO // WAVES) + k
            words = [tok * H + op * RPO + e for e in range(RPO)]
            _tdm_gather(q_lds + op * (RPO * ROW), tensor_addr(Q), words, RPO, n_rows_all)
            _tdm_gather(do_lds + op * (RPO * ROW), tensor_addr(DO), words, RPO, n_rows_all, row_len=DV)
            _tdm_gather(kv_lds + TILE_B + op * (RPO * ROW), tensor_addr(O), words, RPO, n_rows_all, row_len=DV)

        def _issue0():
            issue_octs(load_octs(None, static_tile=0), fx.Int32(0))
            return []

        _if_wave(is_c, [], lambda: [], _issue0)
        _rocdl.s_wait_tensorcnt(0)
        gpu.barrier()

        # ---- delta = rowsum(O * dO) on the score waves, w1's 128-thread split (same sums).
        t2 = aw * WAVE + lane
        h_d = t2 // PPH
        p_d = t2 % PPH

        def _delta_parts():
            acc = fx.Float32(0.0)
            for j in range_constexpr(CPP):
                ov = Vec(lds_load(kv_lds + TILE_B + h_d * ROW + (p_d * CPP + j) * 16, v4i)).bitcast(fx.BFloat16)
                dv = Vec(lds_load(do_lds + h_d * ROW + (p_d * CPP + j) * 16, v4i)).bitcast(fx.BFloat16)
                for e in range_constexpr(8):
                    acc = fadd(acc, fmul(fx.Float32(ov[e]), fx.Float32(dv[e])))
            lds_store(acc, dl_lds + (p_d * H + h_d) * 4, align=4)
            return []

        _if_wave(is_c, [], lambda: [], _delta_parts)
        _lds_barrier()
        delta_h = fx.Float32(lds_load(dl_lds + h_d * 4, T.f32, align=4))
        for k in range_constexpr(1, PPH):
            delta_h = fadd(delta_h, fx.Float32(lds_load(dl_lds + (k * H + h_d) * 4, T.f32, align=4)))
        rsrc_dl = make_rsrc(tensor_addr(DELTA) + fx.Int64(tok) * (H * 4), H * 4)
        dl_off = fx.Int32((p_d == fx.Int32(0)).select(h_d * 4, fx.Int32(0x7FFFFFF0)))
        dl_off = fx.Int32(is_c.select(fx.Int32(0x7FFFFFF0), dl_off))
        buf_store(delta_h, rsrc_dl, dl_off)
        rsrc_im = make_rsrc(tensor_addr(INTERM) + fx.Int64(tok) * fx.Int64(topk * DV * 2), topk * DV * 2)

        # ================= score waves =================
        def score_role():
            hp = aw % HP
            ds = aw // HP
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
                    lg.append(
                        fx.Float32(
                            (l_ == fx.Float32(float("-inf"))).select(fx.Float32(float("inf")), fmul(l_, LOG2E))
                        )
                    )
                lse2.append(lg)
            q_row = [q_lds + (hp * 32 + g * 16) * ROW + a_lane for g in range_constexpr(G)]
            do_row = [do_lds + (hp * 32 + g * 16) * ROW + a_lane for g in range_constexpr(G)]
            kt0 = ds * KTW
            dt0 = ds * DTQ

            def tile_score(t, buf, dq):
                kbase = kv_lds + buf * TILE_B
                kb_w = kbase + a_lane + kt0 * 16 * ROW
                sb = t % 2
                ds_l = dsp0 + sb * (2 * DS_B)
                p_l = ds_l + DS_B
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
                if const_expr("sdp" not in ablate):
                    _run_steps(steps, 8, 2, batch_a)
                for kk in range_constexpr(KTW):
                    kt = kt0 + kk
                    idx = fx.Int32(lds_load(vld0 + (buf * BN + kt * 16 + l16) * 4, T.i32, align=4))
                    ok = idx >= fx.Int32(0)
                    for g in range_constexpr(G):
                        sv = vmul_scalar(s[g][kk], c_log2)
                        pv, dsv = [], []
                        for i in range_constexpr(8):
                            p = fx.Float32(ok.select(exp2(fsub(sv[i], lse2[g][i])), fx.Float32(0.0)))
                            pv.append(p)
                            dsv.append(fmul(fmul(p, fsub(dp[g][kk][i], dlt[g][i])), scale))
                        slot_off = (kt * 16 + l16) * SP + (hp * 32 + g * 16 + hi * 8) * 2
                        lds_store(to_bf16(Vec.from_elements(dsv, fx.Float32)), ds_l + slot_off)
                        lds_store(to_bf16(Vec.from_elements(pv, fx.Float32)), p_l + slot_off)
                # Y: P / dS of this tile complete (C^T waves signalled Y at the top of their step).
                _lds_barrier()
                new = list(dq)
                steps = []
                if const_expr(dqd):
                    # dQ^T[d tiles wave*DTA + j, all heads]
                    for pc in range_constexpr(PC):
                        for j in range_constexpr(DTA):

                            def ld(pc=pc, j=j):
                                a = _frag_tr(kbase + a_tr_lane + pc * 32 * ROW + (aw * DTA + j) * 32, ROW)
                                if const_expr(j == 0):
                                    bb = ds_l + tr_ds + pc * 32 * SP
                                    return (*[_frag_tr(bb + g * 32, SP) for g in range_constexpr(GQ)], a)
                                return (a,)

                            def mma(fr, j=j):
                                if const_expr(j == 0):
                                    cur["b"] = fr[:GQ]
                                for g in range_constexpr(GQ):
                                    new[g * DTA + j] = wmma_bf16(fr[-1], cur["b"][g], new[g * DTA + j])

                            steps.append((ld, mma))
                else:
                    for pc in range_constexpr(PC):
                        for j in range_constexpr(DTQ):

                            def ld(pc=pc, j=j):
                                a = _frag_tr(kbase + a_tr_lane + pc * 32 * ROW + (dt0 + j) * 32, ROW)
                                if const_expr(j == 0):
                                    bb = ds_l + tr_ds + pc * 32 * SP + hp * 64
                                    return (*[_frag_tr(bb + g * 32, SP) for g in range_constexpr(G)], a)
                                return (a,)

                            def mma(fr, j=j):
                                if const_expr(j == 0):
                                    cur["b"] = fr[:G]
                                for g in range_constexpr(G):
                                    new[g * DTQ + j] = wmma_bf16(fr[-1], cur["b"][g], new[g * DTQ + j])

                            steps.append((ld, mma))
                if const_expr("dq" not in ablate):
                    _run_steps(steps, 8, 2, batch_q)
                return new

            init = [Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(NACC)]
            init = init + load_octs(None, static_tile=1)
            for t, st in range(fx.Index(0), fx.Index(n_tile), fx.Index(1), init=init):
                ti = fx.Int32(t)
                # X: tile t landed; every wave is done with tile t-1's kv buffer and with the P / dS
                # buffer this tile overwrites (C^T of tile t-2).
                _rocdl.s_wait_tensorcnt(0)
                _lds_barrier()
                issue_octs([st[NACC + i] for i in range_constexpr(OPW * 8)], (ti + 1) % 2)
                nxt = load_octs(ti + 2)
                new = tile_score(ti, ti % 2, [st[i] for i in range_constexpr(NACC)])
                res = yield list(new) + nxt
            _rocdl.s_wait_tensorcnt(0)
            _lds_barrier()  # X of the drain step
            # dQ (rope columns zero) to the Q / dO area: no score wave reads it any more.
            zero = Vec.filled(4, 0, fx.Int32)
            if const_expr(dqd):
                for g in range_constexpr(GQ):
                    row = q_lds + (g * 16 + l16) * QROW
                    for j in range_constexpr(DTA):
                        lds_store(to_bf16(res[g * DTA + j]), row + (aw * DTA + j) * 32 + hi * 16)
                    lds_store(zero, row + DV * 2 + (hi * AW + aw) * 16)
            else:
                for g in range_constexpr(G):
                    row = q_lds + (hp * 32 + g * 16 + l16) * QROW
                    for j in range_constexpr(DTQ):
                        lds_store(to_bf16(res[g * DTQ + j]), row + (dt0 + j) * 32 + hi * 16)
                    for j in range_constexpr(RZ // DS):
                        lds_store(zero, row + DV * 2 + hi * (RZ * 16) + ds * (RZ // DS * 16) + j * 16)
            _lds_barrier()  # Y of the drain step
            return []

        # ================= C^T waves =================
        def ct_role():
            wb = aw
            stg = stg0 + wb * STG_B
            ct_a = {}
            for j in range_constexpr(DTW):
                for hc in range_constexpr(HC):
                    o = a_tr_lane + hc * 32 * ROW + (wb * DTW + j) * 32
                    ct_a[(j, hc)] = (_frag_tr(q_lds + o, ROW), _frag_tr(do_lds + o, ROW))
            col = wb * (DTW * 32) + (lane % LPR) * 16

            def emit_out(outs, t, kt):
                if const_expr(not stage):
                    # straight from the WMMA layout: 16 B (8 latent columns of one slot) per lane
                    for j in range_constexpr(DTW):
                        off = (t * BN + kt * 16 + l16) * (DV * 2) + (wb * DTW + j) * 32 + hi * 16
                        if const_expr("st" in ablate):
                            off = fx.Int32(0x7FFFFFF0)
                        buf_store(outs[j].bitcast(fx.Int32), rsrc_im, off, aux=CPOL_NT)
                    return
                for g in range_constexpr(GW):
                    lds_store(outs[g], stg + l16 * SROW + g * 32 + hi * 16)
                row0 = t * BN + kt * 16 + lane // LPR
                for r0 in range_constexpr(0, 16 // RPS, STG):
                    rr = list(range_constexpr(r0, min(r0 + STG, 16 // RPS)))
                    vs = [lds_load(stg + (r * RPS + lane // LPR) * SROW + (lane % LPR) * 16, v4i) for r in rr]
                    for v, r in zip(vs, rr):
                        off = (row0 + r * RPS) * (DV * 2) + col
                        if const_expr("st" in ablate):
                            off = fx.Int32(0x7FFFFFF0)
                        buf_store(v, rsrc_im, off, aux=CPOL_NT)

            def tile_ct(t):
                sb = t % 2
                ds_l = dsp0 + sb * (2 * DS_B)
                p_l = ds_l + DS_B

                def ld(kt):
                    out = []
                    for hc in range_constexpr(HC):
                        o = (kt * 16 + l16) * SP + hc * 64 + hi * 16
                        out.append((_frag(ds_l + o), _frag(p_l + o)))
                    return out

                if const_expr("ct" in ablate):
                    return
                if const_expr(ctp):
                    # 16 slots at a time; the WMMAs of slot block kt run in an order without back-to-back
                    # dependences (per element still dS(hc 0), P(hc 0), dS(hc 1), P(hc 1)), and block
                    # kt - 1's stage / stores go out behind them.
                    fr = ld(0)
                    prev = None
                    for kt in range_constexpr(KT):
                        nfr = ld(kt + 1) if kt + 1 < KT else None
                        cs = [Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(DTW)]
                        for hc in range_constexpr(HC):
                            for term in range_constexpr(2):
                                for j in range_constexpr(DTW):
                                    cs[j] = wmma_bf16(ct_a[(j, hc)][term], fr[hc][term], cs[j])
                        if const_expr(prev is not None):
                            emit_out(prev, t, kt - 1)
                        prev = [to_bf16(c) for c in cs]
                        fr = nfr
                        fx.rocdl.sched_barrier(0)
                    emit_out(prev, t, KT - 1)
                    return
                steps = []
                for kt in range_constexpr(KT):

                    def mma(fr, kt=kt):
                        outs = []
                        for j in range_constexpr(DTW):
                            c = Vec.filled(8, 0.0, fx.Float32)
                            for hc in range_constexpr(HC):
                                c = wmma_bf16(ct_a[(j, hc)][0], fr[hc][0], c)
                                c = wmma_bf16(ct_a[(j, hc)][1], fr[hc][1], c)
                            outs.append(to_bf16(c))
                        emit_out(outs, t, kt)

                    steps.append((lambda kt=kt: ld(kt), mma))
                _run_steps(steps, prefetch_c, 1, 0)

            # Step 0: nothing to do (tile 0's P / dS are not ready before Y of step 0).
            _lds_barrier()  # X(0)
            if const_expr(l2pf > 0):
                fx.rocdl.s_barrier_signal(-1)  # Y(0)
                nx = tok + fx.Int32(l2pf)
                nx = fx.Int32((nx < fx.Int32(fx.grid_dim.x)).select(nx, tok))
                bt = aw * WAVE + lane
                _l2_prefetch(tensor_addr(Q) + fx.Int64(nx) * (H * DQK * 2), H * DQK * 2, bt, BW * WAVE)
                _l2_prefetch(tensor_addr(DO) + fx.Int64(nx) * (H * DV * 2), H * DV * 2, bt, BW * WAVE)
                _l2_prefetch(tensor_addr(O) + fx.Int64(nx) * (H * DV * 2), H * DV * 2, bt, BW * WAVE)
                _l2_prefetch(tensor_addr(TOPK) + fx.Int64(nx) * (topk * 4), topk * 4, bt, BW * WAVE)
                _l2_prefetch(tensor_addr(LSE) + fx.Int64(nx) * (H * 4), H * 4, bt, BW * WAVE)
                fx.rocdl.s_barrier_wait(-1)
            else:
                _lds_barrier()  # Y(0)
            for t in range(fx.Index(0), fx.Index(n_tile), fx.Index(1)):
                ti = fx.Int32(t)
                _lds_barrier()  # X(t + 1)
                fx.rocdl.s_barrier_signal(-1)  # Y(t + 1): nothing of ours to publish
                tile_ct(ti)
                fx.rocdl.s_barrier_wait(-1)
            return []

        _if_wave(is_c, [], ct_role, score_role)
        _lds_barrier()  # E: dQ staged
        rsrc_dq = make_rsrc(tensor_addr(DQ) + tok_q, H * DQK * 2)
        for i0 in range_constexpr(0, NQS, STG):
            ii = list(range_constexpr(i0, min(i0 + STG, NQS)))
            cs = [tid + i * BLOCK for i in ii]
            vs = [lds_load(q_lds + (c + c // QCPR) * 16, v4i) for c in cs]
            for v, c in zip(vs, cs):
                buf_store(v, rsrc_dq, c * 16)

    @flyc.jit
    def launch_sparse_mla_bwd_gfx1250_pp(
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
        sparse_mla_bwd_gfx1250_pp_kernel(
            Q, KV, O, DO, TOPK, LSE, DQ, DELTA, INTERM, NTILE, NKV, value_attrs={"rocdl.waves_per_eu": wpe}
        ).launch(grid=(fx.Index(n_tok), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    compiled = {}

    def run_(q, kv, o, do, topk_indices, lse, dq, delta, interm, ntile, stream=None):
        args = (q, kv, o, do, topk_indices, lse, dq, delta, interm, ntile, int(q.shape[0]), int(kv.shape[0]))
        fn = compiled.get("fn")
        if fn is None:
            fn = flyc.compile[COMPILE_HINTS](launch_sparse_mla_bwd_gfx1250_pp, *args, fx.Stream(stream))
            compiled["fn"] = fn
        return fn(*args, stream)

    return run_
