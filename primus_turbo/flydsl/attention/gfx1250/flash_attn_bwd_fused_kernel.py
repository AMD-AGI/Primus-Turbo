###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""gfx1250 flash-attention backward with dQ fused into the dK/dV kernel (4 waves, fp32 dQ atomics).

flash_attn_bwd_kernel.py's backward runs two kernels that both recompute S and dP: k_dkdv
(4 GEMMs per 32q x 32kv unit) and k_dqg (3 GEMMs). Here one 4-wave workgroup owns 128 kv rows of
one (batch, kv head) and does all 5 GEMMs: wave w keeps the K/V fragments of its 32 kv rows
resident, and dQ[32q][32d slice of wave w] = dS[32q][128kv] . K[128kv][32d] is added into an fp32
workspace with device-scope atomics, so dQ is not bitwise reproducible run to run (dK/dV are).
interface.flash_attn_bwd takes this path only when the caller does not ask for a deterministic
backward.

Per step s (one 32-query pair x one q head of the GQA group; descending pairs, heads inner):
  (C) S^T / dP^T: 32 WMMAs on the step's Q/dO B operands, read back from LDS ring slot s % 4
      during step s-1 and carried in VGPRs, as are the step's LSE and delta.
  (D) softmax: exp2(S * scale * log2e - lse * log2e) as one fma; P goes to the wave's private P
      tile, the unscaled dS to the wave's columns of the shared dS buffer s % 2. dK and dQ are
      scaled once, at the dK store and in the dQ conversion.
  (E) split barrier B_s signal, after the stores and the first dV/dK operand loads.
  (F) dV/dK: 32 WMMAs; between the first and second group of 8, the 32 dQ atomics of step s-1
      as one burst (the next group's operand loads are issued before it).
  (G) barrier wait B_s; TDM of step s+3's Q/dO into slot (s+3) % 4.
  (H) one DS burst: dS A fragments of all 128 kv columns, then step s+1's LSE/delta and Q/dO
      readback from slot (s+1) % 4; 16 dQ WMMAs against the resident K^T fragments; TDM of step
      s+3's LSE/delta (the same batch).
One split barrier per step suffices:
  dS RAW   writers retire their stores before signal B_s; readers read after wait B_s.
  dS WAR   buffer s % 2 is rewritten at step s+2, after wait B_{s+1}; every read of it retires
           before the reader's signal B_{s+1}.
  ring RAW each wave retires its TDM of step s+1 before signal B_s; slot (s+1) % 4 is read after
           wait B_s.
  ring WAR TDM(s+3) overwrites slot (s-1) % 4 after wait B_s; every read of that slot retired
           before its reader's signal B_{s-1} or B_s.
The per-step tensor wait retires only TDM(s+1) and leaves TDM(s+2) in flight, relying on TDM
batches of one wave completing in order.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops

from .flash_attn_bwd_kernel import (
    BLOCK_KV,
    DV8,
    NDO,
    NDT,
    NEG,
    NKV,
    NST,
    S_ROW_B,
    X_ROW_B,
    D,
    _bv,
    _exp2,
    _ldv,
    _st1,
    _stv,
)
from .flash_attn_utils import LOG2E, WAVE_SIZE, create_llvm_ptr

_ir = fx.as_ir_value

W4 = 4  # waves per workgroup
W4_THREADS = W4 * WAVE_SIZE
W4_BKV = W4 * BLOCK_KV  # 128 kv rows per workgroup
W4_RING = 4  # Q/dO/LSE/delta ring slots
W4_LEAD = W4_RING - 1  # TDM of step s + W4_LEAD is issued after wait B_s
W4_TDM_OPS = 4  # tensor ops per wave per step: dO, Q, LSE, delta
W4_TW_LOOP = W4_TDM_OPS  # per-step tensor wait: retire TDM(s+1), leave TDM(s+2) in flight
W4_TW_PRO = 2 * W4_TDM_OPS  # prologue: only TDM(0) of TDM(0..2) must have landed
W4_QDO_B = 2 * 32 * X_ROW_B  # dO [32][272 B] at 0, Q at 8704
W4_LSE_OFF = W4_QDO_B  # LSE [32] fp32 at +17408, delta [32] fp32 at +17536
W4_STAGE_B = W4_QDO_B + 256  # one ring slot
W4_DSBUF_B = 32 * X_ROW_B  # one shared dS buffer [32 q][128 kv] bf16, 272 B rows
W4_P_B = 32 * S_ROW_B  # one wave's private P tile [32 q][32 kv]
W4_OFF_RING = 0
W4_OFF_DS = 131072  # dS double buffer, its own LDS segment
W4_OFF_P = W4_OFF_DS + 2 * W4_DSBUF_B
W4_OFF_K = 196608  # K image, prologue only
W4_K_B = W4_BKV * X_ROW_B
W4_LDS = W4_OFF_K + W4_K_B
W4_EPI_CB = 48
W4_EPI_B = 4 * 128 * W4_EPI_CB  # one wave's dK/dV epilogue staging
W4_R1_LOADS = 18  # DS loads issued between the P/dS store burst and signal B_s
assert W4_RING * W4_STAGE_B <= W4_OFF_DS
assert W4_OFF_P + W4 * W4_P_B <= W4_OFF_K
assert W4 * W4_EPI_B <= W4_LDS

# One shape rule for the host: the kv rows split into 128-row workgroup blocks, the queries into
# 32-row pairs (and the 64-row tiles of k_delta / the dQ conversion).
SEQLEN_KV_MULTIPLE = W4_BKV
SEQLEN_Q_MULTIPLE = 64


def _dkdv_fused_impl(
    PARTIAL, G_, Q, K, V, DO, LSE, DEL, DV_, DK, DQA, scale, Sq, Skv, Hq, Hkv, nqt, cshift, causal, B_, nsp
):
    """grid = (Hkv [* nsp], Skv / 128, B), block = 128 (4 waves). G_ (Hq / Hkv) is a Python int:
    the step -> (q pair, q head) split below is a shift and a mask, not a runtime division."""
    G = fx.Int32(G_)
    tid = fx.Int32(fx.thread_idx.x)
    lane = tid % fx.Int32(WAVE_SIZE)
    # wave index as an SGPR: the TDM descriptor and every LDS slice base must be uniform
    wv = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, tid // fx.Int32(WAVE_SIZE)))
    _gx = fx.Int32(fx.block_idx.x)
    if PARTIAL:
        hkv = _gx // nsp
        sp = _gx - hkv * nsp
    else:
        hkv = _gx
        sp = fx.Int32(0)
    bid = fx.Int32(fx.block_idx.y)  # 128-row kv block; ascending = longest first
    bat = fx.Int32(fx.block_idx.z)
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    kvw0 = bid * fx.Int32(W4_BKV)  # workgroup's first kv row
    kv0 = kvw0 + wv * fx.Int32(BLOCK_KV)  # this wave's first kv row

    nkv_b = B_ * Skv * Hkv * fx.Int32(D * 2)
    ndkv_b = B_ * Skv * Hkv * fx.Int32(D * 2)
    g_k = _bv(K, nkv_b, fx.BFloat16, 8)
    g_v = _bv(V, nkv_b, fx.BFloat16, 8)
    if PARTIAL:
        _np_b = nsp * B_ * Skv * Hkv * fx.Int32(D * 4)
        g_dv = _bv(DV_, _np_b, fx.Float32)
        g_dk = _bv(DK, _np_b, fx.Float32)

    rs_kv = Hkv * fx.Int32(DV8)
    base_kv = bat * Skv * rs_kv + hkv * fx.Int32(DV8)

    smem = fx.SharedAllocator().allocate(W4_LDS)
    _lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    lds_ring = _lds0 + fx.Int32(W4_OFF_RING)
    lds_dsb = _lds0 + fx.Int32(W4_OFF_DS)
    lds_p = _lds0 + fx.Int32(W4_OFF_P) + wv * fx.Int32(W4_P_B)
    lds_k = _lds0 + fx.Int32(W4_OFF_K)
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)

    def gfrag2(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return (_ldv(buf, t, fx.BFloat16, 8), _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8))

    def tr(base, rowb):
        return fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base, address_space=3))).shuffle(
            fx.Vector(
                rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base + fx.Int32(16 * rowb), address_space=3))
            ),
            list(range(16)),
        )

    def _l8(addr):
        return fx.Vector(llvm_dialect.load(v8b, create_llvm_ptr(addr, address_space=3)))

    def _st8(vals, addr):
        llvm_dialect.store(
            fx.as_ir_value(fx.Vector.from_elements(vals, dtype=fx.BFloat16)),
            create_llvm_ptr(addr, address_space=3),
        )

    def _split_barrier_signal(tw=0):
        # Explicit waits, then the signal: fx.barrier()'s release fence would also drain the
        # in-flight dQ atomics.
        rocdl.sched_barrier(0)
        rocdl.s_wait_dscnt(0)
        tdm_ops.tensor_wait(tw)
        rocdl.s_barrier_signal(-1)
        rocdl.sched_barrier(0)

    def _split_barrier_wait():
        rocdl.sched_barrier(0)
        rocdl.s_barrier_wait(-1)
        rocdl.sched_barrier(0)

    # ---- TDM: wave w copies rows [8w, 8w+8) of the step's dO, Q, LSE and delta ----------------
    X_ROW_EL = X_ROW_B // 2
    _lds_bf_ty = fx.PointerType.get(
        elem_ty=fx.BFloat16.ir_type, address_space=fx.AddressSpace.Shared, alignment=16
    )
    _lds_f32_ty = fx.PointerType.get(
        elem_ty=fx.Float32.ir_type, address_space=fx.AddressSpace.Shared, alignment=16
    )
    _q_rs_el = Hq * fx.Int32(D)
    v1f = fx.Vector.make_type(1, fx.Float32)

    def _tdm(qt, gh, slot_off, part):
        """part "qdo": the dO and Q rows; part "lse": the LSE and delta rows."""
        qh = hkv * G + gh
        r0 = qt * fx.Int32(32) + wv * fx.Int32(8)
        if part == "qdo":
            off = (fx.Int64(bat * Sq + r0) * fx.Int64(Hq) + fx.Int64(qh)) * fx.Int64(D)
            valid = Sq - r0  # >= 8 for every issued tile
            lb0 = lds_ring + slot_off + wv * fx.Int32(8 * X_ROW_B)
            for src, lb in ((DO, lb0), (Q, lb0 + fx.Int32(32 * X_ROW_B))):
                g_view = fx.Tensor(
                    fx.make_view(fx.add_offset(fx.get_iter(src), off), fx.make_layout((8, D), (D, 1)))
                )
                atom = fx.rocdl.cdna5.make_tdm_atom(
                    g_view,
                    [valid, None],
                    strides=[_q_rs_el, None],
                    num_warps=1,
                    pad_interval=D,
                    pad_amount=X_ROW_EL - D,
                )
                l_view = fx.Tensor(
                    fx.make_view(fx.inttoptr(_lds_bf_ty, lb), fx.make_layout((8, D), (X_ROW_EL, 1)))
                )
                fx.copy_atom_call(atom, g_view, l_view)
        else:
            loff = (bat * Hq + qh) * Sq + r0
            lvalid = Sq - r0  # >= 8
            for src, lo in ((LSE, 0), (DEL, 128)):
                g_view = fx.Tensor(
                    fx.make_view(fx.add_offset(fx.get_iter(src), loff), fx.make_layout((1, 8), (8, 1)))
                )
                atom = fx.rocdl.cdna5.make_tdm_atom(
                    g_view, [None, lvalid], strides=[fx.Int32(8), None], num_warps=1
                )
                l_view = fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(
                            _lds_f32_ty, lds_ring + slot_off + fx.Int32(W4_LSE_OFF + lo) + wv * fx.Int32(32)
                        ),
                        fx.make_layout((1, 8), (8, 1)),
                    )
                )
                fx.copy_atom_call(atom, g_view, l_view)

    def _rdl(slot_off):
        """This lane's LSE and delta for rows hh*16 + row, from a ring slot."""
        out = []
        for hh in range_constexpr(2):
            o = (fx.Int32(hh * 16) + row) * fx.Int32(4)
            for lo in (0, 128):
                out.append(
                    fx.Vector(
                        llvm_dialect.load(
                            v1f,
                            create_llvm_ptr(
                                lds_ring + slot_off + fx.Int32(W4_LSE_OFF + lo) + o, address_space=3
                            ),
                        )
                    )
                )
        return out

    # ---- step schedule: workgroup-uniform (only kvw0, not the wave's kv0, enters) ------------
    nqt2 = nqt // fx.Int32(2)

    def _clampqt(t):
        t = (t < nqt2).select(t, nqt2 - fx.Int32(1))
        return (t < fx.Int32(0)).select(fx.Int32(0), t)

    _c = kvw0 - cshift
    qp_start = ((_c < fx.Int32(0)).select(fx.Int32(0), _c)) // fx.Int32(32)
    qp_start = (causal != fx.Int32(0)).select(qp_start, fx.Int32(0))
    nqp_eff = nqt2 - qp_start
    nqp_eff = (nqp_eff < fx.Int32(0)).select(fx.Int32(0), nqp_eff)
    # pair qt is unmasked for the whole 128-row block iff kvw0 + 127 <= qt*32 + cshift
    _u = kvw0 + fx.Int32(W4_BKV - 1) - cshift
    _qsf = (_u < fx.Int32(0)).select(fx.Int32(0), (_u + fx.Int32(31)) // fx.Int32(32))
    _qsf = (_qsf < nqt2).select(_qsf, nqt2)
    _nm = _qsf - qp_start
    _nm = (_nm < fx.Int32(0)).select(fx.Int32(0), _nm)
    _nm = (_nm < nqp_eff).select(_nm, nqp_eff)
    nmaskp = (causal != fx.Int32(0)).select(_nm, fx.Int32(0))
    if PARTIAL:
        _fn = nqp_eff - nmaskp
        _ch = (_fn + nsp - fx.Int32(1)) // nsp
        _off = sp * _ch
        _cnt = _fn - _off
        _cnt = (_cnt < fx.Int32(0)).select(fx.Int32(0), _cnt)
        _cnt = (_cnt < _ch).select(_cnt, _ch)
        _mk = (sp != fx.Int32(0)).select(fx.Int32(0), nmaskp)
        p_begin = qp_start + (sp != fx.Int32(0)).select(nmaskp + _off, fx.Int32(0))
    else:
        _mk = nmaskp
        _cnt = nqp_eff - nmaskp
        p_begin = qp_start
    n_mask = G * _mk
    n_all = G * (_mk + _cnt)

    def _tile(s):
        """(qt, gh) of step s, clamped into [0, n_all-1] and qt into [0, nqt2-1]. Pairs run
        descending from the top of the range: unmasked pairs first, the masked diagonal ones last,
        and workgroups of one (batch, kv head) reach a pair together (dQ atomic locality)."""
        sc = (s < n_all).select(s, n_all - fx.Int32(1))
        sc = (sc < fx.Int32(0)).select(fx.Int32(0), sc)
        qi = sc // G
        return _clampqt(p_begin + (_mk + _cnt) - fx.Int32(1) - qi), sc - qi * G

    def _dq_atomics(dq, qt, gh):
        """32 closures, one fp32 atomic each, for this wave's 32 d columns of the step's dQ.

        The fp32 workspace is in WMMA C-fragment order: per (b, qh, q pair) a 16 KB block; element
        (q, d) sits at slot k = (qh16*2 + j)*8 + si (512 B) + wave*128 B + lane*4 B, where
        q = qh16*16 + half*8 + si, d = 32*wave + 16*j + row and lane = half*16 + row. One atomic
        instruction is then one full, aligned 128 B line. The constant part of each index goes
        into an inbounds GEP, so it becomes the instruction's immediate offset.
        """
        qh = hkv * G + gh
        base = ((bat * Hq + qh) * nqt2 + qt) * fx.Int32(32 * D) + wv * fx.Int32(WAVE_SIZE) + lane
        pb = fx.to_llvm_ptr(fx.add_offset(fx.get_iter(DQA), base))
        fns = []
        for qh16 in range_constexpr(2):
            for j in range_constexpr(2):
                for si in range_constexpr(8):

                    def _g(qh16=qh16, j=j, si=si):
                        v = fx.Vector(dq[qh16 * 2 + j])[si]
                        p = llvm_dialect.GEPOp(
                            pb.type,
                            pb,
                            [],
                            [((qh16 * 2 + j) * 8 + si) * (W4 * WAVE_SIZE)],
                            fx.Float32.ir_type,
                            llvm_dialect.GEPNoWrapFlags.inbounds,
                        ).result
                        llvm_dialect.atomicrmw(
                            llvm_dialect.AtomicBinOp.fadd,
                            p,
                            fx.as_ir_value(v),
                            llvm_dialect.AtomicOrdering.monotonic,
                            syncscope=rocdl.SyncScope.Agent,
                            alignment=4,
                        )

                    fns.append(_g)
        return fns

    # ---- prologue: TDM steps 0..2, K/V fragments, K image, one barrier ----------------------
    _t0 = _tile(fx.Int32(0))
    _t1 = _tile(fx.Int32(1))
    _t2 = _tile(fx.Int32(2))
    for _t, _slot in ((_t0, 0), (_t1, 1), (_t2, 2)):
        _tdm(_t[0], _t[1], fx.Int32(_slot * W4_STAGE_B), "qdo")
        _tdm(_t[0], _t[1], fx.Int32(_slot * W4_STAGE_B), "lse")
    k2 = [
        [gfrag2(g_k, base_kv, rs_kv, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT)] for kh in range(NKV)
    ]
    kf = [[k2[kh][dt][0].shuffle(k2[kh][dt][1], list(range(16))) for dt in range(NDT)] for kh in range(NKV)]
    vf = []
    for kh in range(NKV):
        _r = []
        for dt in range(NDT):
            a, b = gfrag2(g_v, base_kv, rs_kv, kv0 + fx.Int32(kh * 16), dt)
            _r.append(a.shuffle(b, list(range(16))))
        vf.append(_r)
    for kh in range_constexpr(NKV):
        ko = (wv * fx.Int32(BLOCK_KV) + fx.Int32(kh * 16) + row) * fx.Int32(X_ROW_B) + half * fx.Int32(16)
        for dt in range_constexpr(NDT):
            for u in range_constexpr(2):
                llvm_dialect.store(
                    fx.as_ir_value(k2[kh][dt][u]),
                    create_llvm_ptr(lds_k + ko + fx.Int32(dt * 64 + u * 32), address_space=3),
                )
    _split_barrier_signal(W4_TW_PRO)
    _split_barrier_wait()
    # K^T B fragments for dQ, resident: kt[j][kc] covers d in [32*wv + 16*j, +16) x kv in [kvw0 + 32*kc, +32)
    kt = [
        [
            tr(
                lds_k
                + (fx.Int32(kc * 32) + lane_r) * fx.Int32(X_ROW_B)
                + (lane_c + wv * fx.Int32(32) + fx.Int32(j * 16)) * fx.Int32(2),
                X_ROW_B,
            )
            for kc in range_constexpr(W4)
        ]
        for j in range_constexpr(2)
    ]

    def _rdqd(slot_off):
        """The 32 ds_load_b128 of a ring slot's S/dP B operands, index hh*16 + {Q 0, dO 8} + dt*2 + u."""
        lq = lds_ring + slot_off + fx.Int32(32 * X_ROW_B)
        ld = lds_ring + slot_off
        out = []
        for hh in range_constexpr(2):
            xo = (fx.Int32(hh * 16) + row) * fx.Int32(X_ROW_B) + half * fx.Int32(16)
            for base in (lq, ld):
                for dt in range_constexpr(NDT):
                    for u in range_constexpr(2):
                        out.append(_l8(base + xo + fx.Int32(dt * 64 + u * 32)))
        return out

    def _body(st, s, do_mask):
        acc = st[:NST]
        dqp = st[NST : NST + 4]
        pre = [fx.Vector(v) for v in st[NST + 4 : NST + 8]]
        qd = [fx.Vector(v) for v in st[NST + 8 : NST + 40]]
        qt, gh = _tile(s)
        q0 = qt * fx.Int32(32)
        qtp, ghp = _tile(s - fx.Int32(1))
        atomics = _dq_atomics(dqp, qtp, ghp)  # step s-1's dQ, issued in (F)
        lds_do = lds_ring + (s % fx.Int32(W4_RING)) * fx.Int32(W4_STAGE_B)
        lds_q = lds_do + fx.Int32(32 * X_ROW_B)
        lds_ds = lds_dsb + (s % fx.Int32(2)) * fx.Int32(W4_DSBUF_B)

        # (C)+(D) S^T/dP^T on the carried operands, then softmax; the 8 P/dS stores as one burst
        stores = []
        sl2 = scale * fx.Float32(LOG2E)
        for hh in range_constexpr(2):
            qfr = [
                qd[hh * 16 + dt * 2].shuffle(qd[hh * 16 + dt * 2 + 1], list(range(16)))
                for dt in range_constexpr(NDT)
            ]
            dfr = [
                qd[hh * 16 + 8 + dt * 2].shuffle(qd[hh * 16 + 8 + dt * 2 + 1], list(range(16)))
                for dt in range_constexpr(NDT)
            ]
            q_glob = q0 + fx.Int32(hh * 16) + row
            lse_q = pre[hh * 2][0]
            del_q = pre[hh * 2 + 1][0]
            for kh in range_constexpr(NKV):
                s_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                p_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                for dt in range_constexpr(NDT):
                    s_acc = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(kf[kh][dt]), _ir(qfr[dt]), s_acc, reuseA=False, reuseB=False
                    ).result
                    p_acc = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(vf[kh][dt]), _ir(dfr[dt]), p_acc, reuseA=False, reuseB=False
                    ).result
                sv, pv_ = fx.Vector(s_acc), fx.Vector(p_acc)
                kvb = kv0 + fx.Int32(kh * 16) + half * fx.Int32(8)
                nl2 = fx.Float32(0.0) - lse_q * fx.Float32(LOG2E)
                arg = [fx.fma(sv[si], sl2, nl2) for si in range(8)]
                if do_mask:
                    arg = [
                        ((kvb + fx.Int32(si) > q_glob + cshift) & (causal != fx.Int32(0))).select(
                            fx.Float32(NEG), arg[si]
                        )
                        for si in range(8)
                    ]
                pf = [_exp2(arg[si]) for si in range(8)]
                p_l = [x.to(fx.BFloat16) for x in pf]
                ds_l = [(pf[si] * (pv_[si] - del_q)).to(fx.BFloat16) for si in range(8)]
                stores.append(
                    (
                        p_l,
                        lds_p
                        + (fx.Int32(hh * 16) + row) * fx.Int32(S_ROW_B)
                        + fx.Int32(kh * 32)
                        + half * fx.Int32(16),
                    )
                )
                stores.append(
                    (
                        ds_l,
                        lds_ds
                        + (fx.Int32(hh * 16) + row) * fx.Int32(X_ROW_B)
                        + wv * fx.Int32(64)
                        + fx.Int32(kh * 32)
                        + half * fx.Int32(16),
                    )
                )
        rocdl.sched_barrier(0)
        for vals, addr in stores:
            _st8(vals, addr)

        # (E)+(F) dV/dK: A from the private P tile and this wave's columns of the dS buffer, B from
        # ring slot s % 4, in four 8-WMMA groups.
        def _a(base, rowb, kh):
            return tr(base + lane_r * fx.Int32(rowb) + lane_c * fx.Int32(2) + fx.Int32(kh * 32), rowb)

        def _b(base, dtile):
            c = (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2)
            return tr(base + lane_r * fx.Int32(X_ROW_B) + c, X_ROW_B)

        lds_dsw = lds_ds + wv * fx.Int32(64)
        new = [None] * NST

        def _mm(kh, a, bs, which):
            for dtile in range_constexpr(NDO):
                k = (which * NKV + kh) * NDO + dtile
                new[k] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a), _ir(bs[dtile]), acc[k], reuseA=(dtile > 0), reuseB=False
                ).result

        rocdl.sched_barrier(0)
        a_p0 = _a(lds_p, S_ROW_B, 0)
        b_do = [_b(lds_do, dtile) for dtile in range_constexpr(NDO)]
        rocdl.sched_barrier(0)
        # signal B_s: the W4_R1_LOADS loads above are the only younger DS ops, and DS completes in
        # order, so this retires every P/dS store
        rocdl.s_wait_dscnt(W4_R1_LOADS)
        tdm_ops.tensor_wait(W4_TW_LOOP)
        rocdl.s_barrier_signal(-1)
        rocdl.sched_barrier(0)
        a_ds0 = _a(lds_dsw, X_ROW_B, 0)
        b_q = [_b(lds_q, dtile) for dtile in range_constexpr(NDO)]
        rocdl.sched_barrier(0)
        _mm(0, a_p0, b_do, 0)
        rocdl.sched_barrier(0)
        a_p1 = _a(lds_p, S_ROW_B, 1)
        a_ds1 = _a(lds_dsw, X_ROW_B, 1)
        rocdl.sched_barrier(0)
        for _f in atomics:  # one burst; the loads above run under it
            _f()
        rocdl.sched_barrier(0)
        _mm(0, a_ds0, b_q, 1)
        rocdl.sched_barrier(0)
        _mm(1, a_p1, b_do, 0)
        _mm(1, a_ds1, b_q, 1)

        # (G) wait B_s, then refill the slot step s-1 used with step s+3's dO/Q
        _split_barrier_wait()
        qt2, gh2 = _tile(s + fx.Int32(W4_LEAD))
        _slot2 = ((s + fx.Int32(W4_LEAD)) % fx.Int32(W4_RING)) * fx.Int32(W4_STAGE_B)
        _tdm(qt2, gh2, _slot2, "qdo")
        rocdl.sched_barrier(0)

        # (H) one DS burst: dS A fragments (all 128 kv columns), then step s+1's LSE/delta and
        # S/dP operands; 16 dQ WMMAs (2 q halves x 2 d tiles, 4-deep chains over the 128 kv)
        afrs = []
        for qh16 in range_constexpr(2):
            xo = (fx.Int32(qh16 * 16) + row) * fx.Int32(X_ROW_B) + half * fx.Int32(16)
            afrs.append(
                [
                    _l8(lds_ds + xo + fx.Int32(kc * 64)).shuffle(
                        _l8(lds_ds + xo + fx.Int32(kc * 64 + 32)), list(range(16))
                    )
                    for kc in range_constexpr(W4)
                ]
            )
        _slot1 = ((s + fx.Int32(1)) % fx.Int32(W4_RING)) * fx.Int32(W4_STAGE_B)
        nxt = _rdl(_slot1)
        rb = _rdqd(_slot1)
        rocdl.sched_barrier(0)
        dq = []
        for qh16 in range_constexpr(2):
            for j in range_constexpr(2):
                c = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                for kc in range_constexpr(W4):
                    c = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(afrs[qh16][kc]), _ir(kt[j][kc]), c, reuseA=False, reuseB=False
                    ).result
                dq.append(c)
        rocdl.sched_barrier(0)
        _tdm(qt2, gh2, _slot2, "lse")  # same TDM batch as the dO/Q above, issued last
        rocdl.sched_barrier(0)
        return new + dq + [_ir(v) for v in nxt] + [_ir(v) for v in rb]

    @flyc.jit
    def loop_masked(state, n, i0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            final = yield _body(list(carried), fx.Int32(it) + i0, True)
        return final

    @flyc.jit
    def loop_full(state, n, i0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            final = yield _body(list(carried), fx.Int32(it) + i0, False)
        return final

    init = (
        [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NST)]
        + [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(4)]
        + [_ir(v) for v in _rdl(fx.Int32(0))]
        + [_ir(v) for v in _rdqd(fx.Int32(0))]
    )
    # descending pairs: the unmasked steps come first, the masked ones last
    out = loop_full(init, n_all - n_mask, fx.Int32(0))
    out = list(loop_masked(list(out), n_mask, n_all - n_mask))

    # flush the last step's dQ (zeros to a clamped, in-bounds tile if n_all == 0)
    _tl = _tile(n_all - fx.Int32(1))
    for f in _dq_atomics(out[NST : NST + 4], _tl[0], _tl[1]):
        f()
    # epilogue barrier: every wave's TDM retired and every LDS read done before any wave
    # overwrites the ring/dS/P area with its epilogue staging
    _split_barrier_signal()
    _split_barrier_wait()

    if PARTIAL:
        base_o = ((sp * B_ + bat) * Skv * Hkv) * fx.Int32(D) + hkv * fx.Int32(D)
        for kh in range_constexpr(NKV):
            for dtile in range_constexpr(NDO):
                ov = fx.Vector(out[kh * NDO + dtile])
                ok_ = fx.Vector(out[(NKV + kh) * NDO + dtile])
                for si in range_constexpr(8):
                    kv = kv0 + fx.Int32(kh * 16) + half * fx.Int32(8) + fx.Int32(si)
                    idx = base_o + kv * Hkv * fx.Int32(D) + fx.Int32(dtile * 16) + row
                    _st1(ov[si], g_dv, idx, fx.Float32)
                    _st1(ok_[si] * scale, g_dk, idx, fx.Float32)
    else:
        # LDS-transposed bf16 epilogue, one W4_EPI_B staging area per wave
        g_dv8 = _bv(DV_, ndkv_b, fx.BFloat16, 8)
        g_dk8 = _bv(DK, ndkv_b, fx.BFloat16, 8)
        lds_e = _lds0 + wv * fx.Int32(W4_EPI_B)
        for kh in range_constexpr(NKV):
            lds_ev = lds_e + fx.Int32((kh * 2 + 0) * 128 * W4_EPI_CB)
            lds_ek = lds_e + fx.Int32((kh * 2 + 1) * 128 * W4_EPI_CB)
            for dtile in range_constexpr(NDO):
                ov = fx.Vector(out[kh * NDO + dtile])
                ok_ = fx.Vector(out[(NKV + kh) * NDO + dtile])
                o = (fx.Int32(dtile * 16) + row) * fx.Int32(W4_EPI_CB) + half * fx.Int32(16)
                _st8([ov[si].to(fx.BFloat16) for si in range_constexpr(8)], lds_ev + o)
                _st8([(ok_[si] * scale).to(fx.BFloat16) for si in range_constexpr(8)], lds_ek + o)
            gt = base_kv + (kv0 + fx.Int32(kh * 16) + row) * rs_kv
            for sub in range_constexpr(NDO):
                a = (fx.Int32(sub * 16) + lane_r) * fx.Int32(W4_EPI_CB) + lane_c * fx.Int32(2)
                vv = fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(lds_ev + a, address_space=3)))
                kk = fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(lds_ek + a, address_space=3)))
                t = gt + fx.Int32(sub * 2) + half
                _stv([vv[e] for e in range_constexpr(8)], g_dv8, t, fx.BFloat16)
                _stv([kk[e] for e in range_constexpr(8)], g_dk8, t, fx.BFloat16)


@functools.lru_cache(maxsize=None)
def _dkdv_fused_kernels(group: int):
    """The fused kernels for one GQA group size (Hq / Hkv); it is baked into the code."""

    @flyc.kernel(known_block_size=[W4_THREADS, 1, 1])
    def k_dkdv_fused(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DEL: fx.Tensor,
        DV_: fx.Tensor,
        DK: fx.Tensor,
        DQA: fx.Tensor,
        scale: fx.Float32,
        Sq: fx.Int32,
        Skv: fx.Int32,
        Hq: fx.Int32,
        Hkv: fx.Int32,
        nqt: fx.Int32,
        cshift: fx.Int32,
        causal: fx.Int32,
        B_: fx.Int32,
    ):
        _dkdv_fused_impl(
            False,
            group,
            Q,
            K,
            V,
            DO,
            LSE,
            DEL,
            DV_,
            DK,
            DQA,
            scale,
            Sq,
            Skv,
            Hq,
            Hkv,
            nqt,
            cshift,
            causal,
            B_,
            fx.Int32(1),
        )

    @flyc.kernel(known_block_size=[W4_THREADS, 1, 1])
    def k_dkdv_fused_sp(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DEL: fx.Tensor,
        DV_: fx.Tensor,
        DK: fx.Tensor,
        DQA: fx.Tensor,
        scale: fx.Float32,
        Sq: fx.Int32,
        Skv: fx.Int32,
        Hq: fx.Int32,
        Hkv: fx.Int32,
        nqt: fx.Int32,
        cshift: fx.Int32,
        causal: fx.Int32,
        B_: fx.Int32,
        nsp: fx.Int32,
    ):
        _dkdv_fused_impl(
            True,
            group,
            Q,
            K,
            V,
            DO,
            LSE,
            DEL,
            DV_,
            DK,
            DQA,
            scale,
            Sq,
            Skv,
            Hq,
            Hkv,
            nqt,
            cshift,
            causal,
            B_,
            nsp,
        )

    return k_dkdv_fused, k_dkdv_fused_sp


@flyc.jit
def launch_dkdv_fused(
    Q,
    K,
    V,
    DO,
    LSE,
    DEL,
    DV_,
    DK,
    DQA,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    nqt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    nblk: fx.Int32,
    nhkv: fx.Int32,
    nb: fx.Int32,
    group: fx.Constexpr[int],
    stream: fx.Stream,
):
    _dkdv_fused_kernels(group)[0](
        Q, K, V, DO, LSE, DEL, DV_, DK, DQA, scale, Sq, Skv, Hq, Hkv, nqt, cshift, causal, nb
    ).launch(grid=(nhkv, nblk, nb), block=(W4_THREADS, 1, 1), stream=stream)


@flyc.jit
def launch_dkdv_fused_sp(
    Q,
    K,
    V,
    DO,
    LSE,
    DEL,
    DV_,
    DK,
    DQA,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    nqt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    nblk: fx.Int32,
    nhkv: fx.Int32,
    nb: fx.Int32,
    nsp: fx.Int32,
    nxs: fx.Int32,
    group: fx.Constexpr[int],
    stream: fx.Stream,
):
    _dkdv_fused_kernels(group)[1](
        Q, K, V, DO, LSE, DEL, DV_, DK, DQA, scale, Sq, Skv, Hq, Hkv, nqt, cshift, causal, nb, nsp
    ).launch(grid=(nxs, nblk, nb), block=(W4_THREADS, 1, 1), stream=stream)


# ---- dQ conversion: fp32 C-fragment-order workspace -> scaled bf16 [B, Sq, Hq, D] ---------------
# One 256-thread workgroup per (b, qh, q pair): its 16 KB fp32 tile is read contiguously (4 vec4
# per thread, every 128 B line read once and whole), scattered into an LDS [32 q][128 d] bf16 tile
# and written back as whole 256 B dQ rows (2 x 16 B per thread).
CVT_THREADS = 256
CVT_ROW_B = D * 2 + 16  # LDS row pitch


@flyc.kernel(known_block_size=[CVT_THREADS, 1, 1])
def k_dq_cvt(DQA: fx.Tensor, DQ: fx.Tensor, Sq: fx.Int32, Hq: fx.Int32, n_tiles: fx.Int32, scale: fx.Float32):
    tid = fx.Int32(fx.thread_idx.x)
    tile = fx.Int32(fx.block_idx.x)  # (b * Hq + qh) * (Sq / 32) + qt
    nqt2 = Sq // fx.Int32(32)
    qt = tile % nqt2
    bh = tile // nqt2
    qh = bh % Hq
    b = bh // Hq
    smem = fx.SharedAllocator().allocate(32 * CVT_ROW_B)
    lds = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    g_a = _bv(DQA, n_tiles * fx.Int32(32 * D * 4), fx.Float32, 4)
    g_q = _bv(DQ, n_tiles * fx.Int32(32 * D * 2), fx.BFloat16, 8)
    src0 = tile * fx.Int32(32 * D // 4)  # in vec4 units
    for i in range_constexpr(4):
        v = tid + fx.Int32(i * CVT_THREADS)  # vec4 index inside the 16 KB tile
        x = _ldv(g_a, src0 + v, fx.Float32, 4)
        p = v * fx.Int32(4)
        k = p // fx.Int32(W4 * WAVE_SIZE)  # 128-float slot
        e = p - k * fx.Int32(W4 * WAVE_SIZE)
        w = e // fx.Int32(WAVE_SIZE)
        hf = (e // fx.Int32(16)) % fx.Int32(2)
        r16 = e % fx.Int32(16)
        qh16 = k // fx.Int32(16)
        j = (k // fx.Int32(8)) % fx.Int32(2)
        si = k % fx.Int32(8)
        q = qh16 * fx.Int32(16) + hf * fx.Int32(8) + si
        d = w * fx.Int32(32) + j * fx.Int32(16) + r16
        vals = fx.Vector.from_elements(
            [(x[c] * scale).to(fx.BFloat16) for c in range_constexpr(4)], dtype=fx.BFloat16
        )
        llvm_dialect.store(
            fx.as_ir_value(vals),
            create_llvm_ptr(lds + q * fx.Int32(CVT_ROW_B) + d * fx.Int32(2), address_space=3),
        )
    fx.barrier()
    for i in range_constexpr(2):
        c = tid + fx.Int32(i * CVT_THREADS)  # 16 B chunk: row c // 16, d chunk c % 16
        r = c // fx.Int32(16)
        dc = c - r * fx.Int32(16)
        val = fx.Vector(
            llvm_dialect.load(
                v8b, create_llvm_ptr(lds + r * fx.Int32(CVT_ROW_B) + dc * fx.Int32(16), address_space=3)
            )
        )
        out_row = (b * Sq + qt * fx.Int32(32) + r) * Hq + qh
        _stv([val[e_] for e_ in range_constexpr(8)], g_q, out_row * fx.Int32(D // 8) + dc, fx.BFloat16)


@flyc.jit
def launch_dq_cvt(
    DQA, DQ, Sq: fx.Int32, Hq: fx.Int32, n_tiles: fx.Int32, scale: fx.Float32, stream: fx.Stream
):
    k_dq_cvt(DQA, DQ, Sq, Hq, n_tiles, scale).launch(
        grid=(n_tiles, 1, 1), block=(CVT_THREADS, 1, 1), stream=stream
    )
