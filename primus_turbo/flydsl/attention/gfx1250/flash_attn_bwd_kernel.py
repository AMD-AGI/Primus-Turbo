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

"""gfx1250 (MI455X) FlyDSL flash-attention backward, head_dim 128, batched and GQA-aware.

Written for Primus-Turbo from the gfx1250 bring-up kernels, extended with a batch/head
grid, GQA, bottom-right causal masking and split-K variants for small grids.

    delta[b,h,s] = sum_d dO*O                              k_delta_bshd
    dV[kv,d] = sum_q P^T dO ; dK[kv,d] = sum_q dS^T Q      k_dkdv   (k_dkdv_sp: split-K)
    dQ[q,d]  = sum_kv dS K                                 k_dqg    (k_dq / k_dq_sp)
    fold of split-K fp32 partials -> bf16                  k_redsp (dK/dV), k_redsp_q (dQ)

Layouts: q/o/do [B, Sq, Hq, D] bf16; k/v [B, Skv, Hkv, D] bf16; lse/delta [B, Hq, Sq] fp32,
lse in natural log; dq [B, Sq, Hq, D] and dk/dv [B, Skv, Hkv, D] bf16 (split-K partials are
fp32 [nsp, ...] workspaces folded by k_redsp / k_redsp_q).

GQA is reduced INSIDE k_dkdv: one workgroup owns a kv tile of one kv head and streams every
query tile of all `G = Hq/Hkv` q heads that share it, accumulating into the same registers.
Every output element is written exactly once, with no atomics; the split-K folds sum the
partials in a fixed ascending order. The backward is therefore deterministic run to run.

Divisibility is required, not handled: interface.py enforces Sq % 64 == 0 and
Skv % 32 == 0.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import const_expr, range_constexpr, rocdl

from .common import _to_raw as _ir
from .common import create_llvm_ptr

D = 128
DV8 = D // 8  # vec8 tiles per row of one head
NDT = D // 32  # WMMA k-steps to contract 128
NDO = D // 16  # 16-wide output tiles across d
NKV = 2  # 16-row kv sub-tiles per workgroup
NST = 2 * NKV * NDO  # dV/dK accumulators carried by k_dkdv's q loop
WAVE = 32  # gfx1250 dispatches wave32
LOG2E = 1.4426950408889634
NEG = -3.0e38
# Disabled variants, kept for experimentation (measured no gain; off):
VF_KV = False  # k_dkdv: softmax as one fma per element, scale applied at store
VF_Q = False  # k_dqg: same
KV_U2 = False  # k_dkdv: full q loop unrolled by 2
BLOCK_KV = 32  # kv rows one k_dkdv workgroup owns (NKV x 16)
S_ROW_B = BLOCK_KV * 2 + 16  # 64 -> 80 B LDS row pitch. 16 dwords is a 4-way bank
# collision at 64 banks; 20 dwords walks all 64.
X_ROW_B = D * 2 + 16  # 256 -> 272 B LDS row pitch. 256 B = 64 dwords is exactly
# the full 64-way collision stride on gfx1250's 64x4B
# LDS: every staged row starts on bank 0, so the 16
# rows a ds_load_tr16_b128 phase touches all collide.
# 68 dwords gives row*4 mod 64 -- 16 distinct groups.

DELTA_THREADS = 256
LANES_PER_ROW = D // 8
ROWS_PER_PASS = DELTA_THREADS // LANES_PER_ROW
ROWS_DELTA = 32
PASSES_PER_WG = ROWS_DELTA // ROWS_PER_PASS


# ---- shared helpers -----------------------------------------------------------------
def _bv(t, nb, dt, vec=1):
    nt = (1 << 31) // (vec * (dt.width // 8))
    b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=fx.Int64(nb))
    return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt, vec), (vec, 1))))


def _atom(dt, vec):
    return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * dt.width), dt)


def _ldv(buf, tile, dt, vec):
    f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
    fx.copy_atom_call(_atom(dt, vec), fx.slice(buf, (tile, None)), f)
    return f.load()


def _ld1(buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.copy_atom_call(_atom(dt, 1), fx.slice(buf, (idx, None)), f)
    return f.load()[0]


def _st1(v, buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.memref_store_vec(fx.Vector.from_elements([v], dtype=dt), f)
    fx.copy_atom_call(_atom(dt, 1), f, fx.slice(buf, (idx, None)))


def _stv(vals, buf, tile, dt):
    """Store a `len(vals)`-wide vector of `dt` at vector-index `tile`. Mirrors _ldv."""
    n = len(vals)
    f = fx.make_rmem_tensor(fx.make_layout(n, 1), dt)
    fx.memref_store_vec(fx.Vector.from_elements(vals, dtype=dt), f)
    fx.copy_atom_call(_atom(dt, n), f, fx.slice(buf, (tile, None)))


def _exp2(x):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


# ================================================================== delta ==========
@flyc.kernel(known_block_size=[DELTA_THREADS, 1, 1])
def k_delta_bshd(DO: fx.Tensor, O: fx.Tensor, DEL: fx.Tensor, S: fx.Int32, H: fx.Int32, n_rows: fx.Int32):
    """delta[b, h, s] = sum_d dO[b, s, h, d] * O[b, s, h, d], fp32, written [B, H, S]."""
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    g_do = _bv(DO, n_rows * (D * 2), fx.BFloat16, 8)
    g_o = _bv(O, n_rows * (D * 2), fx.BFloat16, 8)
    g_delta = _bv(DEL, n_rows * 4, fx.Float32)

    tile = bid * (ROWS_DELTA * DV8) + tid
    do_vecs = [
        _ldv(g_do, tile + u * (ROWS_PER_PASS * DV8), fx.BFloat16, 8).ir_value()
        for u in range_constexpr(PASSES_PER_WG)
    ]
    o_vecs = [
        _ldv(g_o, tile + u * (ROWS_PER_PASS * DV8), fx.BFloat16, 8).ir_value()
        for u in range_constexpr(PASSES_PER_WG)
    ]

    lane_in_row = tid % fx.Int32(LANES_PER_ROW)
    row_in_group = tid // fx.Int32(LANES_PER_ROW)
    for u in range_constexpr(PASSES_PER_WG):
        do8 = fx.Vector(do_vecs[u])
        o8 = fx.Vector(o_vecs[u])
        e0 = fx.Float32(0.0)
        e1 = fx.Float32(0.0)
        for c in range_constexpr(4):
            e0 = e0 + fx.Float32(do8[2 * c]) * fx.Float32(o8[2 * c])
            e1 = e1 + fx.Float32(do8[2 * c + 1]) * fx.Float32(o8[2 * c + 1])
        acc = e0 + e1
        for sft in range_constexpr(4):
            acc = acc + acc.shuffle_xor(1 << sft, WAVE)
        idx = bid * fx.Int32(ROWS_DELTA) + u * fx.Int32(ROWS_PER_PASS) + row_in_group
        ok = (lane_in_row == fx.Int32(0)) & (idx < n_rows)
        b = idx // (S * H)
        rem = idx - b * (S * H)
        s_ = rem // H
        h = rem - s_ * H
        dst = (b * H + h) * S + s_
        _st1(acc, g_delta, ok.select(dst, n_rows), fx.Float32)


@flyc.jit
def launch_delta(DO, O, DEL, S: fx.Int32, H: fx.Int32, n_rows: fx.Int32, nblk: fx.Int32, stream: fx.Stream):
    k_delta_bshd(DO, O, DEL, S, H, n_rows).launch(
        grid=(nblk, 1, 1), block=(DELTA_THREADS, 1, 1), stream=stream
    )


# =================================================================== dkdv =========
# Split-K over the q loop. When k_dkdv's grid is about one dispatch wave (1 wave/SIMD, so
# ~1024 workgroups on MI455X) and the causal work skew is large, makespan is the longest
# workgroup and much of the machine idles; only splitting helps. PARTIAL=True
# (k_dkdv_sp) splits the unmasked q-pair range `nsp` ways, each split writing its own fp32
# workspace slice, folded by k_redsp. PARTIAL=False (k_dkdv) is the path for grids that
# already span several dispatch waves, where splitting is pure cost.
def _dkdv_impl(
    PARTIAL, Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, B_, nsp
):
    """One wave owns a BLOCK_KV-row kv tile of one kv head and streams every (q head, q tile).

    grid = (Hkv [* nsp], Skv/BLOCK_KV, B). The accumulators persist across the G q heads
    that share this kv head, so the GQA reduction happens in registers -- atomic-free,
    written once.
    """
    lane = fx.Int32(fx.thread_idx.x)
    # kv head on grid.x (fastest-varying, uniform work), kv tile on grid.y.
    if PARTIAL:
        _x = fx.Int32(fx.block_idx.x)  # kv head * nsp + split
        hkv = _x // nsp
        sp = _x - hkv * nsp
    else:
        hkv = fx.Int32(fx.block_idx.x)  # kv head
        sp = fx.Int32(0)
    bid = fx.Int32(fx.block_idx.y)  # kv tile
    bat = fx.Int32(fx.block_idx.z)  # batch
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    kv0 = bid * fx.Int32(BLOCK_KV)

    # Every buffer descriptor carries the tensor's TRUE byte extent, so the hardware bound
    # check makes any over-read return 0 instead of reading past the allocation. A clamp
    # that ever hit a live access would wreck accuracy, so it cannot hide a real bug.
    nq_b = B_ * Sq * Hq * fx.Int32(D * 2)  # q / do  bf16 [B, Sq, Hq, D]
    nkv_b = B_ * Skv * Hkv * fx.Int32(D * 2)  # k / v   bf16 [B, Skv, Hkv, D]
    nl_b = B_ * Hq * Sq * fx.Int32(4)  # lse / delta fp32 [B, Hq, Sq]
    # dK/dV are stored as bf16 directly (the fp32 accumulator rounded once), saving the
    # extra full-tensor pass and launch a host-side fp32 -> bf16 conversion would cost.
    ndkv_b = B_ * Skv * Hkv * fx.Int32(D * 2)  # dk / dv bf16 [B, Skv, Hkv, D]
    g_q = _bv(Q, nq_b, fx.BFloat16, 8)
    g_k = _bv(K, nkv_b, fx.BFloat16, 8)
    g_v = _bv(V, nkv_b, fx.BFloat16, 8)
    g_do = _bv(DO, nq_b, fx.BFloat16, 8)
    g_lse = _bv(LSE, nl_b, fx.Float32)
    g_del = _bv(DEL, nl_b, fx.Float32)
    if PARTIAL:
        _np_b = nsp * B_ * Skv * Hkv * fx.Int32(D * 4)  # [nsp, B, Skv, Hkv, D] fp32
        g_dv = _bv(DV_, _np_b, fx.Float32)
        g_dk = _bv(DK, _np_b, fx.Float32)
    else:
        g_dv = _bv(DV_, ndkv_b, fx.BFloat16)
        g_dk = _bv(DK, ndkv_b, fx.BFloat16)

    rs_q = Hq * fx.Int32(DV8)  # vec8 tiles between consecutive q rows
    rs_kv = Hkv * fx.Int32(DV8)
    base_kv = bat * Skv * rs_kv + hkv * fx.Int32(DV8)

    # The workgroup owns BLOCK_KV key rows as NKV 16-row WMMA accumulator sets.
    # LDS segment separation: gfx1250's LDS is organised in 64 KB segments served by two
    # read ports; reads in different segments can issue in the same cycle. The B ring
    # (dO, Q) sits at offset 0 and the A ring (P, dS) exactly 65536 B up, which flips
    # address bit 16 so the rings land in adjacent segments whatever the physical LDS
    # base. Occupancy is unchanged: 70656 B still allows 4 workgroups per CU, and VGPR
    # use (1 wave/SIMD) caps it at 4 anyway. Placing the SMALL ring high keeps the total
    # under that rung.
    LDS_SEG = 65536
    smem = fx.SharedAllocator().allocate(LDS_SEG + 2 * 32 * S_ROW_B)
    _lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    lds_do = _lds0
    lds_q = lds_do + fx.Int32(32 * X_ROW_B)
    lds_p = _lds0 + fx.Int32(LDS_SEG)
    lds_ds = lds_p + fx.Int32(32 * S_ROW_B)
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16))
        )

    def gfrag2(buf, base, rs, r, dt):
        """gfrag's two halves before the shuffle: cols [c, c+8) and [c+16, c+24) of
        row r+row, where c = half*8 + dt*32. Each is exactly one 16-byte LDS chunk."""
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return (_ldv(buf, t, fx.BFloat16, 8), _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8))

    # K and V fragments for this kv tile are invariant over every query and every q head.
    # Prefetch: the 32 Q/dO buffer_load_b128 an iteration consumes are issued one
    # iteration early and carried in the scf.for state, so the consuming wait sits
    # ~450 instructions after issue instead of 2.
    def _ldqd(qt, gh):
        qh = hkv * G + gh
        base_q = bat * Sq * rs_q + qh * fx.Int32(DV8)
        q0 = qt * fx.Int32(32)
        out = []
        # The 4 LSE/delta buffer_load_b32 join the cross-iteration prefetch and are
        # issued FIRST, at the head of the loadcnt FIFO. Loaded in place they had the
        # shortest load-to-use distance in the body and caused most of its stall cycles.
        # Head-of-FIFO placement keeps the consuming wait partial (the 32 b128 still in
        # flight) instead of a full drain, which measured -14.6% when it happened.
        base_l = (bat * Hq + qh) * Sq
        for hh in range_constexpr(2):
            qg = q0 + fx.Int32(hh * 16) + row
            # carried as 1-wide vectors: the scf.for carried tuple is _ir()'d, and
            # _to_raw only accepts vector values.
            out.append(_ldv(g_lse, base_l + qg, fx.Float32, 1))
            out.append(_ldv(g_del, base_l + qg, fx.Float32, 1))
        # Source order is not issue order: without this fence the scheduler sinks the
        # 32 b128 below the b32 consumption point and the wait becomes a full drain.
        # sched_barrier(0) (nothing may cross) pins the 4 b32 ahead of the 32 b128.
        rocdl.sched_barrier(0)
        for hh in range_constexpr(2):
            qh0 = q0 + fx.Int32(hh * 16)
            for buf in (g_q, g_do):
                for dt in range_constexpr(NDT):
                    a, b = gfrag2(buf, base_q, rs_q, qh0, dt)
                    out.append(a)
                    out.append(b)
        return out

    kf = [[gfrag(g_k, base_kv, rs_kv, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT)] for kh in range(NKV)]
    vf = [[gfrag(g_v, base_kv, rs_kv, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT)] for kh in range(NKV)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)

    # CAUSAL TILE SKIP. Under bottom-right causal, kv row j is attended only by queries
    # q >= j - cshift, so this workgroup's kv tile [kv0, kv0+16) is untouched by every
    # query tile below qt_start = max(0, (kv0 - cshift) // 16). Tiles below it contributed
    # exactly zero to dK/dV (p = exp2(NEG*LOG2E) = 0), so skipping them is not an
    # approximation -- the result is bit-identical, only the work is gone.
    nqt2 = nqt // fx.Int32(2)  # query tiles come in PAIRS now

    def _clampqt(t):
        """Clamp a prefetch's query pair into [0, nqt2-1] so it never reads out of range."""
        t = (t < nqt2).select(t, nqt2 - fx.Int32(1))
        return (t < fx.Int32(0)).select(fx.Int32(0), t)

    _c = kv0 - cshift
    qp_start = ((_c < fx.Int32(0)).select(fx.Int32(0), _c)) // fx.Int32(32)
    qp_start = (causal != fx.Int32(0)).select(qp_start, fx.Int32(0))
    nqp_eff = nqt2 - qp_start

    # (a) Loop order: query-pair OUTER, q-head INNER (qi = ii // G, gh = ii - qi*G).
    #     This puts every masked iteration at the front, which makes (b) a clean loop
    #     split, and the G q heads of one GQA group are G*D*2 contiguous bytes of one q
    #     row, so sweeping them back to back reads each row once. The fp32 accumulation
    #     order is fixed, so the result is deterministic run to run.
    # (b) Mask split. Under bottom-right causality at most one query pair per q head
    #     straddles this workgroup's kv tile; every later pair has
    #     kv0+BLOCK_KV-1 <= q0 + cshift, so the mask predicate is provably false there.
    #     The mask costs ~14% of the loop body's instructions and the body is issue-bound
    #     (~10% WMMA), so the unmasked loop is emitted without it.
    def _body(acc, pre, qt, gh, do_mask, qt_n, gh_n, carry=True):
        q0 = qt * fx.Int32(32)

        # The masked body does NOT carry the prefetch (carry=False): it runs only a
        # handful of iterations, and giving both loops identically-shaped carried
        # tuples made the allocator duplicate ~200 VGPR. It loads its own Q/dO and
        # returns only the accumulators.
        if const_expr(carry):
            nxt = _ldqd(qt_n, gh_n)
        else:
            nxt = None
            pre = _ldqd(qt, gh)

        # Q and dO are staged to LDS from the registers the S/P GEMM already holds
        # instead of being read from global a second time: gfrag's halves tile the
        # [32][D] LDS image exactly (row hh*16+row, byte col half*16 + dt*64, second
        # half at +32). The Q/dO fragments feed both kv sub-tiles, once per 16-query half.
        for hh in range_constexpr(2):
            qp = [(pre[4 + hh * 16 + dt * 2], pre[4 + hh * 16 + dt * 2 + 1]) for dt in range_constexpr(NDT)]
            dp = [
                (pre[4 + hh * 16 + 8 + dt * 2], pre[4 + hh * 16 + 8 + dt * 2 + 1])
                for dt in range_constexpr(NDT)
            ]
            xo = (fx.Int32(hh * 16) + row) * fx.Int32(X_ROW_B) + half * fx.Int32(16)
            for dt in range_constexpr(NDT):
                for u in range_constexpr(2):
                    o = xo + fx.Int32(dt * 64 + u * 32)
                    llvm_dialect.store(
                        fx.as_ir_value(dp[dt][u]), create_llvm_ptr(lds_do + o, address_space=3)
                    )
                    llvm_dialect.store(fx.as_ir_value(qp[dt][u]), create_llvm_ptr(lds_q + o, address_space=3))
            qfr = [qp[dt][0].shuffle(qp[dt][1], list(range(16))) for dt in range_constexpr(NDT)]
            dfr = [dp[dt][0].shuffle(dp[dt][1], list(range(16))) for dt in range_constexpr(NDT)]
            # LSE/delta were prefetched an iteration early (see _ldqd); q_glob is pure
            # VALU and is still needed by the causal mask below.
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
                # causal, BOTTOM-RIGHT: query q attends kv <= q + (Skv - Sq).
                kvb = kv0 + fx.Int32(kh * 16) + half * fx.Int32(8)
                if const_expr(VF_KV):
                    # VF_KV (off): exp2(s*scale*log2e - lse*log2e) as one fma per element,
                    # dS stored without the softmax scale (applied once to dK at the store).
                    # Deterministic, but not bitwise identical to the default path.
                    _c1 = scale * fx.Float32(LOG2E)
                    _nl = lse_q * fx.Float32(-LOG2E)
                    tt = [fx.Float32(fx.fma(sv[si], _c1, _nl)) for si in range(8)]
                    if const_expr(do_mask):
                        tt = [
                            ((kvb + fx.Int32(si) > q_glob + cshift) & (causal != fx.Int32(0))).select(
                                fx.Float32(NEG), tt[si]
                            )
                            for si in range(8)
                        ]
                    pf = [_exp2(tt[si]) for si in range(8)]
                    p_l = [x.to(fx.BFloat16) for x in pf]
                    ds_l = [(pf[si] * (pv_[si] - del_q)).to(fx.BFloat16) for si in range(8)]
                else:
                    if const_expr(do_mask):
                        masked = [
                            ((kvb + fx.Int32(si) > q_glob + cshift) & (causal != fx.Int32(0))).select(
                                fx.Float32(NEG), sv[si] * scale
                            )
                            for si in range(8)
                        ]
                    else:
                        masked = [sv[si] * scale for si in range(8)]
                    pf = [_exp2((masked[si] - lse_q) * fx.Float32(LOG2E)) for si in range(8)]
                    p_l = [x.to(fx.BFloat16) for x in pf]
                    ds_l = [(pf[si] * (pv_[si] - del_q) * scale).to(fx.BFloat16) for si in range(8)]
                off = (fx.Int32(hh * 16) + row) * fx.Int32(S_ROW_B) + fx.Int32(kh * 32) + half * fx.Int32(16)
                llvm_dialect.store(
                    fx.as_ir_value(fx.Vector.from_elements(p_l, dtype=fx.BFloat16)),
                    create_llvm_ptr(lds_p + off, address_space=3),
                )
                llvm_dialect.store(
                    fx.as_ir_value(fx.Vector.from_elements(ds_l, dtype=fx.BFloat16)),
                    create_llvm_ptr(lds_ds + off, address_space=3),
                )
        # No fx.barrier() here: the workgroup is one wave32, so s_barrier is a no-op, and
        # the all-counter wait it would bring (s_wait_loadcnt_dscnt 0x0) would drain the
        # in-flight prefetch. The LDS RAW wait is derived from the memory dependence.

        # 32 queries staged, so the contraction is FULL: rows [lane_r] and
        # [lane_r+16] concatenate in lane into a v16 operand, no padding zeros.
        def tr(base, rowb):
            return fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base, address_space=3))).shuffle(
                fx.Vector(
                    rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base + fx.Int32(16 * rowb), address_space=3))
                ),
                list(range(16)),
            )

        # The B operands (dO, Q) are shared by BOTH kv sub-tiles -- one transpose
        # load feeds two WMMAs instead of one. The A operands differ only by a
        # 16-column (32-byte) offset into the same [q][kv] tile.
        b_do = []
        b_q = []
        for dtile in range_constexpr(NDO):
            c = (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2)
            b_do.append(tr(lds_do + lane_r * fx.Int32(X_ROW_B) + c, X_ROW_B))
            b_q.append(tr(lds_q + lane_r * fx.Int32(X_ROW_B) + c, X_ROW_B))
        new = [None] * (2 * NKV * NDO)
        for kh in range_constexpr(NKV):
            col = lane_c * fx.Int32(2) + fx.Int32(kh * 32)
            a_p = tr(lds_p + lane_r * fx.Int32(S_ROW_B) + col, S_ROW_B)
            a_ds = tr(lds_ds + lane_r * fx.Int32(S_ROW_B) + col, S_ROW_B)
            # A-operand reuse: gfx1250 WMMA can hint that A (or B) is identical to the
            # preceding matrix instruction's. Emitting two runs of NDO (a_p then a_ds)
            # rather than interleaving them holds A fixed across each run, so every WMMA
            # after the first in a run sets reuseA. B changes every op.
            for dtile in range_constexpr(NDO):
                new[kh * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_p), _ir(b_do[dtile]), acc[kh * NDO + dtile], reuseA=(dtile > 0), reuseB=False
                ).result
            for dtile in range_constexpr(NDO):
                new[(NKV + kh) * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f,
                    _ir(a_ds),
                    _ir(b_q[dtile]),
                    acc[(NKV + kh) * NDO + dtile],
                    reuseA=(dtile > 0),
                    reuseB=False,
                ).result
        # No fx.barrier() here either (single wave; see above).
        if const_expr(carry):
            return new + [_ir(v) for v in nxt]
        return new

    @flyc.jit
    def qloop_mask(state, n, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            qi = ii // G
            st = list(carried)
            final = yield _body(st, None, qt0 + qi, ii - qi * G, True, None, None, False)
        return final

    @flyc.jit
    def qloop_full(state, n, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            qi = ii // G
            jj = ii + fx.Int32(1)
            # The last iteration's prefetch (ii+1) would name a pair that does not exist;
            # clamp it to the last one, as k_dq's kvloop_full does. Value-neutral: the
            # redirected loads are consumed by iterations that never run.
            jj = (jj < n).select(jj, n - fx.Int32(1))
            qj = jj // G
            st = list(carried)
            final = yield _body(
                st[:NST],
                [fx.Vector(v) for v in st[NST:]],
                qt0 + qi,
                ii - qi * G,
                False,
                qt0 + qj,
                jj - qj * G,
            )
        return final

    # KV_U2 (off, measured no gain): qloop_full unrolled by 2, as k_dqg's DQ_U2. Two
    # bodies per scf.for trip let the second prefetch be allocated onto the dead carried
    # set, removing the back-edge v_mov_b64 copies. n2 = pairs; an odd leftover iteration
    # runs in qloop_tail, which loads its own Q/dO. Bitwise identical to the one-body loop.
    @flyc.jit
    def qloop_full2(state, n2, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n2), 1, init=state):
            ii = fx.Int32(it) * fx.Int32(2)
            i1 = ii + fx.Int32(1)
            jj = ii + fx.Int32(2)
            jj = (jj < n2 * fx.Int32(2)).select(jj, n2 * fx.Int32(2) - fx.Int32(1))
            qi = ii // G
            q1 = i1 // G
            qj = jj // G
            st = list(carried)
            mid = _body(
                st[:NST],
                [fx.Vector(v) for v in st[NST:]],
                qt0 + qi,
                ii - qi * G,
                False,
                qt0 + q1,
                i1 - q1 * G,
            )
            final = yield _body(
                mid[:NST],
                [fx.Vector(v) for v in mid[NST:]],
                qt0 + q1,
                i1 - q1 * G,
                False,
                qt0 + qj,
                jj - qj * G,
            )
        return final

    @flyc.jit
    def qloop_tail(state, n, i0, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it) + i0
            qi = ii // G
            final = yield _body(list(carried), None, qt0 + qi, ii - qi * G, False, None, None, False)
        return final

    def _qloop_full_any(state, n, qt0):
        if KV_U2:
            n2 = n // fx.Int32(2)
            out = qloop_full2(state, n2, qt0)
            return qloop_tail(list(out)[:NST], n - n2 * fx.Int32(2), n2 * fx.Int32(2), qt0)
        return qloop_full(state, n, qt0)

    # Query pair `qt` is fully unmasked iff this workgroup's LARGEST key index is
    # attended by the pair's SMALLEST query: kv0 + BLOCK_KV-1 <= qt*32 + cshift, i.e.
    # qt >= ceil((kv0 + BLOCK_KV-1 - cshift)/32). cshift = Skv - Sq can be negative.
    _u = kv0 + fx.Int32(BLOCK_KV - 1) - cshift
    _qsf = (_u < fx.Int32(0)).select(fx.Int32(0), (_u + fx.Int32(31)) // fx.Int32(32))
    _qsf = (_qsf < nqt2).select(_qsf, nqt2)
    _nm = _qsf - qp_start
    _nm = (_nm < fx.Int32(0)).select(fx.Int32(0), _nm)
    _nm = (_nm < nqp_eff).select(_nm, nqp_eff)
    nmaskp = (causal != fx.Int32(0)).select(_nm, fx.Int32(0))

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(2 * NKV * NDO)]
    # The prologue prefetch for qloop_full's first iteration is issued between the masked
    # loop (which carries no prefetch) and the full loop.
    if PARTIAL:
        # The masked query pairs come first (from qp_start) and stay with split 0, so the
        # mask split is unchanged. The remaining `_fn` unmasked pairs are cut into
        # contiguous chunks; every pair costs the same, so contiguous is balanced.
        _fn = nqp_eff - nmaskp
        _fn = (_fn < fx.Int32(0)).select(fx.Int32(0), _fn)
        _ch = (_fn + nsp - fx.Int32(1)) // nsp
        _off = sp * _ch
        _cnt = _fn - _off
        _cnt = (_cnt < fx.Int32(0)).select(fx.Int32(0), _cnt)
        _cnt = (_cnt < _ch).select(_cnt, _ch)
        _mk = (sp != fx.Int32(0)).select(fx.Int32(0), nmaskp)
        _qt0 = qp_start + nmaskp + _off
        out = qloop_mask(init, G * _mk, qp_start)
        # The prologue issues iteration 0 unconditionally, even when the chunk is empty
        # (_cnt == 0). Clamp the query pair to the last legal one; value-neutral.
        _pc0 = _clampqt(_qt0)
        out = list(out) + [_ir(v) for v in _ldqd(_pc0, fx.Int32(0))]
        out = _qloop_full_any(out, G * _cnt, _qt0)
        base_o = ((sp * B_ + bat) * Skv * Hkv) * fx.Int32(D) + hkv * fx.Int32(D)
    else:
        out = qloop_mask(init, G * nmaskp, qp_start)
        # Prologue prefetch for qloop_full's first iteration (qi = 0, gh = 0) at query
        # pair qp_start + nmaskp, clamped as above.
        _pc0 = _clampqt(qp_start + nmaskp)
        out = list(out) + [_ir(v) for v in _ldqd(_pc0, fx.Int32(0))]
        out = _qloop_full_any(out, G * (nqp_eff - nmaskp), qp_start + nmaskp)
        base_o = bat * Skv * Hkv * fx.Int32(D) + hkv * fx.Int32(D)
    if PARTIAL:
        # fp32 workspace path (nsp > 1), narrow stores. The LDS-staged wide stores of the
        # bf16 path below do not apply: gfx1250's transposing LDS loads stop at 16-bit
        # (no 32-bit form), so an fp32 tile could only be staged row-major, costing
        # 64 ds_write_b32 to save 48 buffer_store -- a net loss.
        for kh in range_constexpr(NKV):
            for dtile in range_constexpr(NDO):
                ov = fx.Vector(out[kh * NDO + dtile])
                ok_ = fx.Vector(out[(NKV + kh) * NDO + dtile])
                for si in range_constexpr(8):
                    kv = kv0 + fx.Int32(kh * 16) + half * fx.Int32(8) + fx.Int32(si)
                    idx = base_o + kv * Hkv * fx.Int32(D) + fx.Int32(dtile * 16) + row
                    _st1(ov[si], g_dv, idx, fx.Float32)
                    _st1(ok_[si], g_dk, idx, fx.Float32)
    else:
        # Stage the dK/dV epilogue through the finished LDS ring. The WMMA output layout
        # gives each lane ONE d-column and 8 kv rows, so a direct bf16 epilogue can only
        # emit 256 buffer_store_b16 per workgroup, each covering 32 B of a 128 B line.
        #
        # The accumulator is transposed through LDS instead, and the transpose is
        # FREE in both directions:
        #   write  -- the image is COLUMN major, M[d][kv], so a lane's 8 kv rows for
        #             one d column are contiguous: one ds_write_b128 per d tile.
        #   read   -- `ds_load_tr16_b128` gives lane l element e `src[(l//16)*8+e, l%16]`,
        #             i.e. reading M column-major returns ROW
        #             major: lane l holds kv row l%16 and 8 CONSECUTIVE d columns.
        # 64 narrow stores per (kv half, output) become 8 ds_write_b128 + 8
        # ds_load_tr16_b128 + 8 buffer_store_b128: 256 -> 96 instructions, of which
        # VMEM 256 -> 32.
        #
        # It costs no LDS: the Q/dO ring (17408 B at offset 0) is dead here, and the
        # four staging images are 4 * 128 * EPI_CB = 24576 B, still inside segment 0,
        # so the allocation and therefore the 4-workgroup occupancy rung are unchanged.
        # EPI_CB = 48 B = 32 B of kv row + 16 B pad: 12 dwords walks c*12 mod 64,
        # 16 distinct bank groups, the same granularity X_ROW_B was chosen for.
        EPI_CB = 48
        g_dv8 = _bv(DV_, ndkv_b, fx.BFloat16, 8)
        g_dk8 = _bv(DK, ndkv_b, fx.BFloat16, 8)
        for kh in range_constexpr(NKV):
            lds_ev = _lds0 + fx.Int32((kh * 2 + 0) * 128 * EPI_CB)
            lds_ek = _lds0 + fx.Int32((kh * 2 + 1) * 128 * EPI_CB)
            for dtile in range_constexpr(NDO):
                ov = fx.Vector(out[kh * NDO + dtile])
                ok_ = fx.Vector(out[(NKV + kh) * NDO + dtile])
                o = (fx.Int32(dtile * 16) + row) * fx.Int32(EPI_CB) + half * fx.Int32(16)
                llvm_dialect.store(
                    fx.as_ir_value(
                        fx.Vector.from_elements(
                            [ov[si].to(fx.BFloat16) for si in range_constexpr(8)], dtype=fx.BFloat16
                        )
                    ),
                    create_llvm_ptr(lds_ev + o, address_space=3),
                )
                llvm_dialect.store(
                    fx.as_ir_value(
                        fx.Vector.from_elements(
                            [ok_[si].to(fx.BFloat16) for si in range_constexpr(8)], dtype=fx.BFloat16
                        )
                    ),
                    create_llvm_ptr(lds_ek + o, address_space=3),
                )
            # lane_r / lane_c are the transposing-load address form already used by
            # `tr()` in the body: lane l addresses row (l//16)*8 + l%8, column
            # ((l//8)%2)*8, and receives column l%16 of rows (l//16)*8 + e.
            gt = base_kv + (kv0 + fx.Int32(kh * 16) + row) * rs_kv
            for sub in range_constexpr(NDO):
                a = (fx.Int32(sub * 16) + lane_r) * fx.Int32(EPI_CB) + lane_c * fx.Int32(2)
                vv = fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(lds_ev + a, address_space=3)))
                kk = fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(lds_ek + a, address_space=3)))
                t = gt + fx.Int32(sub * 2) + half
                _stv([vv[e] for e in range_constexpr(8)], g_dv8, t, fx.BFloat16)
                _stv([kk[e] for e in range_constexpr(8)], g_dk8, t, fx.BFloat16)


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dkdv(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    DO: fx.Tensor,
    LSE: fx.Tensor,
    DEL: fx.Tensor,
    DV_: fx.Tensor,
    DK: fx.Tensor,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nqt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    B_: fx.Int32,
):
    _dkdv_impl(
        False,
        Q,
        K,
        V,
        DO,
        LSE,
        DEL,
        DV_,
        DK,
        scale,
        Sq,
        Skv,
        Hq,
        Hkv,
        G,
        nqt,
        cshift,
        causal,
        B_,
        fx.Int32(1),
    )


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dkdv_sp(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    DO: fx.Tensor,
    LSE: fx.Tensor,
    DEL: fx.Tensor,
    DV_: fx.Tensor,
    DK: fx.Tensor,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nqt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    B_: fx.Int32,
    nsp: fx.Int32,
):
    _dkdv_impl(True, Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, B_, nsp)


@flyc.jit
def launch_dkdv(
    Q,
    K,
    V,
    DO,
    LSE,
    DEL,
    DV_,
    DK,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nqt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    nblk: fx.Int32,
    nhkv: fx.Int32,
    nb: fx.Int32,
    stream: fx.Stream,
):
    k_dkdv(Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, nb).launch(
        grid=(nhkv, nblk, nb), block=(32, 1, 1), stream=stream
    )


@flyc.jit
def launch_dkdv_sp(
    Q,
    K,
    V,
    DO,
    LSE,
    DEL,
    DV_,
    DK,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nqt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    nblk: fx.Int32,
    nhkv: fx.Int32,
    nb: fx.Int32,
    nsp: fx.Int32,
    nxs: fx.Int32,
    stream: fx.Stream,
):
    # nxs = nhkv * nsp, computed on the host: grid.x carries the split index in its low
    # digits so the kv tile stays on grid.y.
    k_dkdv_sp(
        Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, nb, nsp
    ).launch(grid=(nxs, nblk, nb), block=(32, 1, 1), stream=stream)


# ===================================================================== dq =========
KV_STEP = 32  # two 16-kv tiles: exactly one WMMA contraction of dS
NKT = KV_STEP // 16
BLOCK_Q = 64  # queries one k_dq workgroup owns (NQ x 16)
NQ = BLOCK_Q // 16


# Split-K over k_dq's kv loop, the mirror of k_dkdv_sp. k_dq's workgroup count is
# ceil(Sq/BLOCK_Q)*Hq*B and one workgroup is one wave32, so at 1 wave/SIMD the device
# holds ~1024 at once. Large grids saturate it and use the unsplit bf16 path; a small
# grid (e.g. 128 workgroups) leaves most SIMDs idle, and splitting is the only way to add
# workgroups. PARTIAL=True (k_dq_sp) cuts the UNMASKED kv range `nsp` ways, each split
# writing its own fp32 [nsp, B, Sq, Hq, D] workspace slice, folded by k_redsp_q.
def _dq_impl(PARTIAL, Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, B_, nsp):
    """One wave owns BLOCK_Q queries of one q head and streams every 32-key block.

    grid = (Hq [* nsp], Sq/BLOCK_Q, B). The A operand of the dQ GEMM comes free: the two
    kv-tile dS^T accumulators concatenate in lane into one v16, so the only LDS traffic is
    staging K.
    """
    lane = fx.Int32(fx.thread_idx.x)
    # Longest-first dispatch. Under bottom-right causality query tile t streams
    # ceil((t*BLOCK_Q + BLOCK_Q + cshift) / KV_STEP) kv blocks, so work grows with t.
    # The q head is grid.x (cheap, uniform work, fastest-varying) and the query tile is
    # grid.y walked DESCENDING, so the longest tiles are issued first and the tail of the
    # dispatch is the cheapest. Pure dispatch order: results are bitwise unchanged.
    # XCD-major q-head remap. MI455X has 8 XCDs and workgroups are handed to them
    # round-robin on the linearised id, so with grid.x = Hq = 32 one XCD would get q heads
    # {c, c+8, c+16, c+24} -- four different kv heads. Permuting x by (x%8)*(Hq/8) + x/8
    # puts q heads {4c..4c+3} on XCD c: one kv head per XCD, K/V L2 footprint cut 4x.
    # It is a bijection of [0, Hq) whenever Hq % 8 == 0 (identity otherwise).
    # In the split kernel grid.x carries the split index in its low digits, as in
    # launch_dkdv_sp; the remap is applied to the decoded q head.
    _nx = fx.Int32(8)
    if PARTIAL:
        _fx = fx.Int32(fx.block_idx.x)  # q head * nsp + split
        _bx = _fx // nsp
        sp = _fx - _bx * nsp
    else:
        _bx = fx.Int32(fx.block_idx.x)
        sp = fx.Int32(0)
    qh = (Hq % _nx == fx.Int32(0)).select((_bx % _nx) * (Hq // _nx) + _bx // _nx, _bx)
    bid = (
        (Sq + fx.Int32(BLOCK_Q - 1)) // fx.Int32(BLOCK_Q) - fx.Int32(1) - fx.Int32(fx.block_idx.y)
    )  # query tile, descending
    bat = fx.Int32(fx.block_idx.z)  # batch
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    q0 = bid * fx.Int32(BLOCK_Q)
    hkv = qh // G

    g_q = _bv(Q, 1 << 30, fx.BFloat16, 8)
    g_k = _bv(K, 1 << 30, fx.BFloat16, 8)
    g_v = _bv(V, 1 << 30, fx.BFloat16, 8)
    g_do = _bv(DO, 1 << 30, fx.BFloat16, 8)
    g_lse = _bv(LSE, 1 << 28, fx.Float32)
    g_del = _bv(DEL, 1 << 28, fx.Float32)
    if PARTIAL:
        # [nsp, B, Sq, Hq, D] fp32. TRUE byte extent, not the flat 1 GiB the bf16 path
        # carries: the workspace is the one buffer in this kernel whose size depends on
        # nsp, and an over-read of it would land in another split's slice.
        # The extent is computed in Int64: in Int32 it can wrap (e.g. to 0 at
        # nsp=8, B=4, Sq=8192, Hq=32), and a descriptor with num_records 0 silently
        # drops every dQ write.
        g_dq = _bv(
            DQ,
            (nsp.to(fx.Int64) * B_.to(fx.Int64) * Sq.to(fx.Int64) * Hq.to(fx.Int64) * fx.Int64(D * 4)),
            fx.Float32,
        )
    else:
        g_dq = _bv(DQ, 1 << 30, fx.BFloat16)  # bf16 output, no host-side conversion

    rs_q = Hq * fx.Int32(DV8)
    rs_kv = Hkv * fx.Int32(DV8)
    base_q = bat * Sq * rs_q + qh * fx.Int32(DV8)
    base_kv = bat * Skv * rs_kv + hkv * fx.Int32(DV8)
    base_l = (bat * Hq + qh) * Sq

    smem = fx.SharedAllocator().allocate(KV_STEP * X_ROW_B)
    lds_k = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16))
        )

    def gfrag2(buf, base, rs, r, dt):
        """gfrag's two halves before the shuffle: cols [c, c+8) and [c+16, c+24) of
        row r+row, where c = half*8 + dt*32. Each is exactly one 16-byte LDS chunk."""
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return (_ldv(buf, t, fx.BFloat16, 8), _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8))

    # Q, dO, lse and delta are invariant over the whole kv loop: hoist them.
    # The workgroup owns BLOCK_Q queries as NQ 16-row tiles. K and V (global fragments,
    # LDS staging and the tr16 transpose of it) are invariant over queries, so each of
    # them serves all NQ tiles.
    qf = [[gfrag(g_q, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT)] for qh_ in range(NQ)]
    dof = [[gfrag(g_do, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT)] for qh_ in range(NQ)]
    q_glob = [q0 + fx.Int32(qh_ * 16) + row for qh_ in range(NQ)]
    lse_q = [_ld1(g_lse, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQ)]
    del_q = [_ld1(g_del, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQ)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)

    # Prefetch, as in k_dkdv: the 32 K/V buffer_load_b128 an iteration consumes are
    # issued one iteration early and carried in the kvloop_full scf.for state. Only the
    # FULL loop carries it; the masked loop (a handful of iterations) loads its own, so
    # the two carried tuples differ in shape and the allocator does not duplicate them.
    def _ldkv(kv0):
        out = []
        for kt in range_constexpr(NKT):
            for dt in range_constexpr(NDT):
                a, b = gfrag2(g_k, base_kv, rs_kv, kv0 + fx.Int32(kt * 16), dt)
                c, d = gfrag2(g_v, base_kv, rs_kv, kv0 + fx.Int32(kt * 16), dt)
                out += [a, b, c, d]
        return out

    # Mask split. Under bottom-right causality only the last one or two kv blocks a
    # query block touches straddle the diagonal; in every earlier block the predicate
    # `kv > q + cshift` is provably false. The mask costs ~23% of the loop body's
    # instructions and the body is issue-bound (~12% WMMA), so the loop is split:
    # [0, nfull) with no mask emitted, then [nfull, nkvt_eff) with it. Bitwise identical.
    def _body(acc, pre, kv0, do_mask, kv0_n, carry=True):
        if const_expr(carry):
            nxt = _ldkv(kv0_n)
        else:
            nxt = None
            pre = _ldkv(kv0)

        # K is staged to LDS from the registers the S = K Q^T GEMM already holds instead
        # of being read from global a second time: gfrag's two halves tile the [32][D]
        # LDS image exactly (row kt*16+row, byte col half*16 + dt*64, second half at +32).
        # X_ROW_B's padding makes these 16 ds_store_b128 conflict-free.
        ds_halves = [[] for _ in range_constexpr(NQ)]
        for kt in range_constexpr(NKT):
            s_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(NQ)]
            p_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(NQ)]
            ko = (fx.Int32(kt * 16) + row) * fx.Int32(X_ROW_B) + half * fx.Int32(16)
            for dt in range_constexpr(NDT):
                # One pair of K/V fragments (from the carried prefetch) feeds all NQ
                # query tiles.
                pi = (kt * NDT + dt) * 4
                kp = (pre[pi], pre[pi + 1])
                for u in range_constexpr(2):
                    llvm_dialect.store(
                        fx.as_ir_value(kp[u]),
                        create_llvm_ptr(lds_k + ko + fx.Int32(dt * 64 + u * 32), address_space=3),
                    )
                kfr = kp[0].shuffle(kp[1], list(range(16)))
                vfr = pre[pi + 2].shuffle(pre[pi + 3], list(range(16)))
                for qh_ in range_constexpr(NQ):
                    s_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(kfr), _ir(qf[qh_][dt]), s_acc[qh_], reuseA=False, reuseB=False
                    ).result
                    p_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(vfr), _ir(dof[qh_][dt]), p_acc[qh_], reuseA=False, reuseB=False
                    ).result
            for qh_ in range_constexpr(NQ):
                sv, pv_ = fx.Vector(s_acc[qh_]), fx.Vector(p_acc[qh_])
                if const_expr(do_mask):
                    masked = [
                        (
                            (
                                kv0 + fx.Int32(kt * 16) + half * fx.Int32(8) + fx.Int32(si)
                                > q_glob[qh_] + cshift
                            )
                            & (causal != fx.Int32(0))
                        ).select(fx.Float32(NEG), sv[si] * scale)
                        for si in range(8)
                    ]
                else:
                    masked = [sv[si] * scale for si in range(8)]
                pf = [_exp2((masked[si] - lse_q[qh_]) * fx.Float32(LOG2E)) for si in range(8)]
                ds_halves[qh_].append(
                    [(pf[si] * (pv_[si] - del_q[qh_]) * scale).to(fx.BFloat16) for si in range(8)]
                )
        # FREE: the two kv-tile accumulators concatenate in-lane into the dS A-operand.
        a_ds = [
            fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1], dtype=fx.BFloat16)
            for qh_ in range_constexpr(NQ)
        ]
        # No fx.barrier() here either (single wave; see above).

        new = [None] * (NQ * NDO)
        for dtile in range_constexpr(NDO):
            base = lds_k + lane_r * fx.Int32(X_ROW_B) + (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2)
            b_k = fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base, address_space=3))).shuffle(
                fx.Vector(
                    rocdl.ds_load_tr16_b128(
                        v8b, create_llvm_ptr(base + fx.Int32(16 * X_ROW_B), address_space=3)
                    )
                ),
                list(range(16)),
            )
            for qh_ in range_constexpr(NQ):
                new[qh_ * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds[qh_]), _ir(b_k), acc[qh_ * NDO + dtile], reuseA=False, reuseB=False
                ).result
        # No fx.barrier() here either (single wave; see above).
        if const_expr(carry):
            return new + [_ir(v) for v in nxt]
        return new

    if PARTIAL:
        # Identical body, with the iteration index rebased onto this split's chunk. The
        # prefetch clamp stays RELATIVE (jj in [0, n)) and the base is added after it, so
        # it never reaches past this chunk's last block.
        @flyc.jit
        def kvloop_full(state, n, it0):
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                ii = fx.Int32(it)
                jj = ii + fx.Int32(1)
                jj = (jj < n).select(jj, n - fx.Int32(1))
                st = list(carried)
                final = yield _body(
                    st[: NQ * NDO],
                    [fx.Vector(v) for v in st[NQ * NDO :]],
                    (ii + it0) * fx.Int32(KV_STEP),
                    False,
                    (jj + it0) * fx.Int32(KV_STEP),
                )
            return final
    else:

        @flyc.jit
        def kvloop_full(state, n):
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                ii = fx.Int32(it)
                # k_dq's descriptors carry a flat 1 GiB num_records (no true-extent
                # clamp), so a prefetch past the last block would be a live out-of-bounds
                # read. The last iteration re-issues its own block instead.
                jj = ii + fx.Int32(1)
                jj = (jj < n).select(jj, n - fx.Int32(1))
                st = list(carried)
                final = yield _body(
                    st[: NQ * NDO],
                    [fx.Vector(v) for v in st[NQ * NDO :]],
                    ii * fx.Int32(KV_STEP),
                    False,
                    jj * fx.Int32(KV_STEP),
                )
            return final

    @flyc.jit
    def kvloop_mask(state, n, it0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            final = yield _body(
                list(carried), None, (fx.Int32(it) + it0) * fx.Int32(KV_STEP), True, None, False
            )
        return final

    # Causal tile skip, the mirror of k_dkdv's. Queries [q0, q0+BLOCK_Q) reach at most key
    # q0 + BLOCK_Q-1 + cshift, so every kv block past that is fully masked and contributes
    # exactly zero to dQ. Bit-identical, not an approximation.
    _lim = (q0 + fx.Int32(BLOCK_Q) + cshift + fx.Int32(KV_STEP - 1)) // fx.Int32(KV_STEP)
    _lim = (_lim < fx.Int32(1)).select(fx.Int32(1), _lim)
    _lim = (_lim < nkvt).select(_lim, nkvt)
    nkvt_eff = (causal != fx.Int32(0)).select(_lim, nkvt)

    # Block `it` is fully unmasked iff its largest key index is attended by this
    # block's SMALLEST query: it*KV_STEP + KV_STEP-1 <= q0 + cshift, i.e.
    # it < (q0 + cshift + 1) // KV_STEP. cshift = Skv - Sq can be negative (Sq > Skv),
    # so the numerator is guarded before the division.
    _t = q0 + cshift + fx.Int32(1)
    _nf = (_t < fx.Int32(0)).select(fx.Int32(0), _t // fx.Int32(KV_STEP))
    _nf = (_nf < nkvt_eff).select(_nf, nkvt_eff)
    nfull = (causal != fx.Int32(0)).select(_nf, nkvt_eff)

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NQ * NDO)]
    if PARTIAL:
        # The UNMASKED range [0, nfull) is cut into nsp contiguous chunks;
        # every block in it costs the same, so contiguous is balanced. The masked tail
        # [nfull, nkvt_eff) is at most two blocks and is NOT cut -- it stays whole on
        # the LAST split, which makes the fold order over sp = 0..nsp-1 exactly the
        # ascending kv-block order the unsplit kernel accumulates in.
        _ch = (nfull + nsp - fx.Int32(1)) // nsp
        _off = sp * _ch
        _cnt = nfull - _off
        _cnt = (_cnt < fx.Int32(0)).select(fx.Int32(0), _cnt)
        _cnt = (_cnt < _ch).select(_cnt, _ch)
        # The prologue prefetch is issued whether or not this split has work, and k_dq's
        # K/V descriptors carry a flat 1 GiB num_records, so an empty split at _off >=
        # nkvt_eff would issue a LIVE out-of-bounds read. Clamp the prefetch address;
        # the loaded value is dead in that case.
        _pf = (_off < nkvt_eff).select(_off, nkvt_eff - fx.Int32(1))
        _mk = (sp == nsp - fx.Int32(1)).select(nkvt_eff - nfull, fx.Int32(0))
        out = kvloop_full(init + [_ir(v) for v in _ldkv(_pf * fx.Int32(KV_STEP))], _cnt, _off)
        out = kvloop_mask(list(out)[: NQ * NDO], _mk, nfull)
        base_o = ((sp * B_ + bat) * Sq * Hq) * fx.Int32(D) + qh * fx.Int32(D)
    else:
        out = kvloop_full(init + [_ir(v) for v in _ldkv(fx.Int32(0))], nfull)
        out = kvloop_mask(list(out)[: NQ * NDO], nkvt_eff - nfull, nfull)
        base_o = bat * Sq * Hq * fx.Int32(D) + qh * fx.Int32(D)
    for qh_ in range_constexpr(NQ):
        for dtile in range_constexpr(NDO):
            ov = fx.Vector(out[qh_ * NDO + dtile])
            for si in range_constexpr(8):
                q_i = q0 + fx.Int32(qh_ * 16) + half * fx.Int32(8) + fx.Int32(si)
                # The address is written out in both branches on purpose: hoisting it into
                # a shared variable reorders the emitted ops and changes the unsplit
                # k_dq's ISA.
                if PARTIAL:
                    _st1(
                        ov[si], g_dq, base_o + q_i * Hq * fx.Int32(D) + fx.Int32(dtile * 16) + row, fx.Float32
                    )
                else:
                    _st1(
                        ov[si].to(fx.BFloat16),
                        g_dq,
                        base_o + q_i * Hq * fx.Int32(D) + fx.Int32(dtile * 16) + row,
                        fx.BFloat16,
                    )


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dq(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    DO: fx.Tensor,
    LSE: fx.Tensor,
    DEL: fx.Tensor,
    DQ: fx.Tensor,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nkvt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
):
    _dq_impl(
        False,
        Q,
        K,
        V,
        DO,
        LSE,
        DEL,
        DQ,
        scale,
        Sq,
        Skv,
        Hq,
        Hkv,
        G,
        nkvt,
        cshift,
        causal,
        fx.Int32(1),
        fx.Int32(1),
    )


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dq_sp(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    DO: fx.Tensor,
    LSE: fx.Tensor,
    DEL: fx.Tensor,
    DQ: fx.Tensor,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nkvt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    B_: fx.Int32,
    nsp: fx.Int32,
):
    _dq_impl(True, Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, B_, nsp)


@flyc.jit
def launch_dq(
    Q,
    K,
    V,
    DO,
    LSE,
    DEL,
    DQ,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nkvt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    nblk: fx.Int32,
    nhq: fx.Int32,
    nb: fx.Int32,
    stream: fx.Stream,
):
    k_dq(Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal).launch(
        grid=(nhq, nblk, nb), block=(32, 1, 1), stream=stream
    )  # q head on x, see _dq_impl


@flyc.jit
def launch_dq_sp(
    Q,
    K,
    V,
    DO,
    LSE,
    DEL,
    DQ,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nkvt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    nblk: fx.Int32,
    nhq: fx.Int32,
    nb: fx.Int32,
    nsp: fx.Int32,
    nxq: fx.Int32,
    stream: fx.Stream,
):
    # nxq = nhq * nsp, computed on the host -- same shape as launch_dkdv_sp's nxs.
    k_dq_sp(Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, nb, nsp).launch(
        grid=(nxq, nblk, nb), block=(32, 1, 1), stream=stream
    )


# ===================================================================== dqg ==========
# Grouped k_dq ("k_dqg"), used by interface.py on the unsplit path (nsp_q == 1); the
# split path uses k_dq_sp. The defaults below reproduce k_dq's schedule plus DQ_U2.
#
#   DQ_NW    waves per workgroup. Wave w of group x owns q head x*DQ_NW + w, so with
#            DQ_NW == G the waves of one workgroup are the G q heads that share ONE kv
#            head, on the same query tile: they stream identical K/V rows on the same CU,
#            so most K/V fetches can hit the CU-local cache. Waves do not communicate
#            (own LDS K slice, own registers), so there is no barrier. Default 1.
#   DQ_BQW   queries per wave (64; 32 halves the accumulators and Q/dO fragments so that,
#            without the K/V prefetch, the wave fits <= 512 VGPR, i.e. 2 waves per SIMD).
#   DQ_PF    carry the one-iteration K/V prefetch, or load in-iteration.
#   DQ_DFUSE (off, measured no gain) compute delta = rowsum(dO * O) in the prologue from
#            the dO fragments the wave already holds plus one O fragment load, and write
#            it to DEL for k_dkdv (which must then run after k_dqg) instead of k_delta.
#   DQ_U2    unroll the full kv loop by 2 (see kvloop_full below).
DQ_NW = 1
DQ_BQW = 64
DQ_PF = True
DQ_DFUSE = False
DQ_U2 = True


def _dqg_impl(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal):
    NW, BQW, NQW = DQ_NW, DQ_BQW, DQ_BQW // 16
    tid = fx.Int32(fx.thread_idx.x)
    if NW > 1:
        lane = tid % fx.Int32(WAVE)
        wave = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, (tid // fx.Int32(WAVE)).ir_value()))
    else:
        lane = tid
        wave = fx.Int32(0)
    # Head GROUP on grid.x; k_dq's XCD-major remap is applied to the group index
    # (a bijection of [0, ngrp) whenever ngrp % 8 == 0, identity otherwise), so with
    # Hq = 32: NW = 4 -> 8 groups, group c = kv head c on XCD c; NW = 2 -> 16 groups,
    # XCD c gets groups 2c, 2c+1 = q heads 4c..4c+3 = kv head c; NW = 1 = k_dq's remap.
    _nx = fx.Int32(8)
    _gx = fx.Int32(fx.block_idx.x)
    ngrp = Hq // fx.Int32(NW)
    grp = (ngrp % _nx == fx.Int32(0)).select((_gx % _nx) * (ngrp // _nx) + _gx // _nx, _gx)
    qh = grp * fx.Int32(NW) + wave
    bid = Sq // fx.Int32(BQW) - fx.Int32(1) - fx.Int32(fx.block_idx.y)  # descending
    bat = fx.Int32(fx.block_idx.z)
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    q0 = bid * fx.Int32(BQW)
    hkv = qh // G

    g_q = _bv(Q, 1 << 30, fx.BFloat16, 8)
    g_k = _bv(K, 1 << 30, fx.BFloat16, 8)
    g_v = _bv(V, 1 << 30, fx.BFloat16, 8)
    g_do = _bv(DO, 1 << 30, fx.BFloat16, 8)
    g_lse = _bv(LSE, 1 << 28, fx.Float32)
    g_del = _bv(DEL, 1 << 28, fx.Float32)
    g_dq = _bv(DQ, 1 << 30, fx.BFloat16)
    if DQ_DFUSE:
        g_o = _bv(O, 1 << 30, fx.BFloat16, 8)

    rs_q = Hq * fx.Int32(DV8)
    rs_kv = Hkv * fx.Int32(DV8)
    base_q = bat * Sq * rs_q + qh * fx.Int32(DV8)
    base_kv = bat * Skv * rs_kv + hkv * fx.Int32(DV8)
    base_l = (bat * Hq + qh) * Sq

    smem = fx.SharedAllocator().allocate(NW * KV_STEP * X_ROW_B)
    lds_k = fx.Int32(fx.ptrtoint(smem.peek().ptr)) + wave * fx.Int32(KV_STEP * X_ROW_B)
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16))
        )

    def gfrag2(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return (_ldv(buf, t, fx.BFloat16, 8), _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8))

    qf = [[gfrag(g_q, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT)] for qh_ in range(NQW)]
    dof = [
        [gfrag(g_do, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT)] for qh_ in range(NQW)
    ]
    q_glob = [q0 + fx.Int32(qh_ * 16) + row for qh_ in range(NQW)]
    lse_q = [_ld1(g_lse, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQW)]
    if DQ_DFUSE:
        # gfrag gives lane (row, half) columns [8h+32dt, +8) and [8h+32dt+16, +8) of its
        # row for dt = 0..3: 64 of the 128 columns; the lane with the other `half` (lane
        # ^ 16) holds the other 64. So delta(row) = own partial + partner's partial.
        # fp32 addition is commutative, so both lanes hold the bitwise-same total and
        # both store it to the same address (a benign same-value duplicate).
        del_q = []
        for qh_ in range(NQW):
            e0 = fx.Float32(0.0)
            e1 = fx.Float32(0.0)
            for dt in range(NDT):
                o16 = gfrag(g_o, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt)
                d16 = dof[qh_][dt]
                for i in range(8):
                    e0 = e0 + fx.Float32(d16[2 * i]) * fx.Float32(o16[2 * i])
                    e1 = e1 + fx.Float32(d16[2 * i + 1]) * fx.Float32(o16[2 * i + 1])
            part = e0 + e1
            tot = part + part.shuffle_xor(16, WAVE)
            _st1(tot, g_del, base_l + q_glob[qh_], fx.Float32)
            del_q.append(tot)
    else:
        del_q = [_ld1(g_del, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQW)]
    _c1q = scale * fx.Float32(LOG2E)  # VF_Q only
    _nlq = [lse_q[qh_] * fx.Float32(-LOG2E) for qh_ in range(NQW)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)

    def _ldkv(kv0):
        out = []
        for kt in range_constexpr(NKT):
            for dt in range_constexpr(NDT):
                a, b = gfrag2(g_k, base_kv, rs_kv, kv0 + fx.Int32(kt * 16), dt)
                c, d = gfrag2(g_v, base_kv, rs_kv, kv0 + fx.Int32(kt * 16), dt)
                out += [a, b, c, d]
        return out

    def _body(acc, pre, kv0, do_mask, kv0_n, carry=True):
        if const_expr(carry):
            nxt = _ldkv(kv0_n)
        else:
            nxt = None
            pre = _ldkv(kv0)
        ds_halves = [[] for _ in range_constexpr(NQW)]
        for kt in range_constexpr(NKT):
            s_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(NQW)]
            p_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(NQW)]
            ko = (fx.Int32(kt * 16) + row) * fx.Int32(X_ROW_B) + half * fx.Int32(16)
            for dt in range_constexpr(NDT):
                pi = (kt * NDT + dt) * 4
                kp = (pre[pi], pre[pi + 1])
                for u in range_constexpr(2):
                    llvm_dialect.store(
                        fx.as_ir_value(kp[u]),
                        create_llvm_ptr(lds_k + ko + fx.Int32(dt * 64 + u * 32), address_space=3),
                    )
                kfr = kp[0].shuffle(kp[1], list(range(16)))
                vfr = pre[pi + 2].shuffle(pre[pi + 3], list(range(16)))
                for qh_ in range_constexpr(NQW):
                    s_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(kfr), _ir(qf[qh_][dt]), s_acc[qh_], reuseA=False, reuseB=False
                    ).result
                    p_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(vfr), _ir(dof[qh_][dt]), p_acc[qh_], reuseA=False, reuseB=False
                    ).result
            for qh_ in range_constexpr(NQW):
                sv, pv_ = fx.Vector(s_acc[qh_]), fx.Vector(p_acc[qh_])
                if const_expr(VF_Q):
                    tt = [fx.Float32(fx.fma(sv[si], _c1q, _nlq[qh_])) for si in range(8)]
                    if const_expr(do_mask):
                        tt = [
                            (
                                (
                                    kv0 + fx.Int32(kt * 16) + half * fx.Int32(8) + fx.Int32(si)
                                    > q_glob[qh_] + cshift
                                )
                                & (causal != fx.Int32(0))
                            ).select(fx.Float32(NEG), tt[si])
                            for si in range(8)
                        ]
                    pf = [_exp2(tt[si]) for si in range(8)]
                    ds_halves[qh_].append(
                        [(pf[si] * (pv_[si] - del_q[qh_])).to(fx.BFloat16) for si in range(8)]
                    )
                else:
                    if const_expr(do_mask):
                        masked = [
                            (
                                (
                                    kv0 + fx.Int32(kt * 16) + half * fx.Int32(8) + fx.Int32(si)
                                    > q_glob[qh_] + cshift
                                )
                                & (causal != fx.Int32(0))
                            ).select(fx.Float32(NEG), sv[si] * scale)
                            for si in range(8)
                        ]
                    else:
                        masked = [sv[si] * scale for si in range(8)]
                    pf = [_exp2((masked[si] - lse_q[qh_]) * fx.Float32(LOG2E)) for si in range(8)]
                    ds_halves[qh_].append(
                        [(pf[si] * (pv_[si] - del_q[qh_]) * scale).to(fx.BFloat16) for si in range(8)]
                    )
        a_ds = [
            fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1], dtype=fx.BFloat16)
            for qh_ in range_constexpr(NQW)
        ]
        new = [None] * (NQW * NDO)
        for dtile in range_constexpr(NDO):
            base = lds_k + lane_r * fx.Int32(X_ROW_B) + (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2)
            b_k = fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base, address_space=3))).shuffle(
                fx.Vector(
                    rocdl.ds_load_tr16_b128(
                        v8b, create_llvm_ptr(base + fx.Int32(16 * X_ROW_B), address_space=3)
                    )
                ),
                list(range(16)),
            )
            for qh_ in range_constexpr(NQW):
                new[qh_ * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds[qh_]), _ir(b_k), acc[qh_ * NDO + dtile], reuseA=False, reuseB=False
                ).result
        if const_expr(carry):
            return new + [_ir(v) for v in nxt]
        return new

    if DQ_PF and DQ_U2:
        # DQ_U2: the full loop unrolled by 2. With one body per scf.for trip the carried
        # prefetch lands in fresh registers and the back-edge copies it into the
        # loop-carried ones (~65 v_mov_b64 per trip). With two bodies per trip the second
        # prefetch is issued after the carried set is dead and is allocated onto it.
        # n = number of PAIRS (blocks [0, 2n)); an odd leftover block runs in kvloop_mask,
        # whose causal predicate is identically false on it -- bitwise identical.
        @flyc.jit
        def kvloop_full(state, n):
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                ii = fx.Int32(it) * fx.Int32(2)
                jj = ii + fx.Int32(2)
                jj = (jj < n * fx.Int32(2)).select(jj, n * fx.Int32(2) - fx.Int32(1))
                st = list(carried)
                mid = _body(
                    st[: NQW * NDO],
                    [fx.Vector(v) for v in st[NQW * NDO :]],
                    ii * fx.Int32(KV_STEP),
                    False,
                    (ii + fx.Int32(1)) * fx.Int32(KV_STEP),
                )
                final = yield _body(
                    mid[: NQW * NDO],
                    [fx.Vector(v) for v in mid[NQW * NDO :]],
                    (ii + fx.Int32(1)) * fx.Int32(KV_STEP),
                    False,
                    jj * fx.Int32(KV_STEP),
                )
            return final
    elif DQ_PF:

        @flyc.jit
        def kvloop_full(state, n):
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                ii = fx.Int32(it)
                jj = ii + fx.Int32(1)
                jj = (jj < n).select(jj, n - fx.Int32(1))
                st = list(carried)
                final = yield _body(
                    st[: NQW * NDO],
                    [fx.Vector(v) for v in st[NQW * NDO :]],
                    ii * fx.Int32(KV_STEP),
                    False,
                    jj * fx.Int32(KV_STEP),
                )
            return final
    else:

        @flyc.jit
        def kvloop_full(state, n):
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                final = yield _body(list(carried), None, fx.Int32(it) * fx.Int32(KV_STEP), False, None, False)
            return final

    @flyc.jit
    def kvloop_mask(state, n, it0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            final = yield _body(
                list(carried), None, (fx.Int32(it) + it0) * fx.Int32(KV_STEP), True, None, False
            )
        return final

    _lim = (q0 + fx.Int32(BQW) + cshift + fx.Int32(KV_STEP - 1)) // fx.Int32(KV_STEP)
    _lim = (_lim < fx.Int32(1)).select(fx.Int32(1), _lim)
    _lim = (_lim < nkvt).select(_lim, nkvt)
    nkvt_eff = (causal != fx.Int32(0)).select(_lim, nkvt)
    _t = q0 + cshift + fx.Int32(1)
    _nf = (_t < fx.Int32(0)).select(fx.Int32(0), _t // fx.Int32(KV_STEP))
    _nf = (_nf < nkvt_eff).select(_nf, nkvt_eff)
    nfull = (causal != fx.Int32(0)).select(_nf, nkvt_eff)

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NQW * NDO)]
    if DQ_PF and DQ_U2:
        npair = nfull // fx.Int32(2)
        out = kvloop_full(init + [_ir(v) for v in _ldkv(fx.Int32(0))], npair)
        nfull = npair * fx.Int32(2)  # the mask loop takes the odd leftover
    elif DQ_PF:
        # prologue prefetch of block 0: always a real block (nkvt_eff >= 1).
        out = kvloop_full(init + [_ir(v) for v in _ldkv(fx.Int32(0))], nfull)
    else:
        out = kvloop_full(init, nfull)
    out = kvloop_mask(list(out)[: NQW * NDO], nkvt_eff - nfull, nfull)
    base_o = bat * Sq * Hq * fx.Int32(D) + qh * fx.Int32(D)
    for qh_ in range_constexpr(NQW):
        for dtile in range_constexpr(NDO):
            ov = fx.Vector(out[qh_ * NDO + dtile])
            ovs = (
                [ov[si] * scale for si in range(8)] if VF_Q else [ov[si] for si in range(8)]
            )  # VF_Q: scale applied at store
            for si in range_constexpr(8):
                q_i = q0 + fx.Int32(qh_ * 16) + half * fx.Int32(8) + fx.Int32(si)
                _st1(
                    ovs[si].to(fx.BFloat16),
                    g_dq,
                    base_o + q_i * Hq * fx.Int32(D) + fx.Int32(dtile * 16) + row,
                    fx.BFloat16,
                )


@flyc.kernel(known_block_size=[DQ_NW * 32, 1, 1])
def k_dqg(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    DO: fx.Tensor,
    O: fx.Tensor,
    LSE: fx.Tensor,
    DEL: fx.Tensor,
    DQ: fx.Tensor,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nkvt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
):
    _dqg_impl(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal)


@flyc.jit
def launch_dqg(
    Q,
    K,
    V,
    DO,
    O,
    LSE,
    DEL,
    DQ,
    scale: fx.Float32,
    Sq: fx.Int32,
    Skv: fx.Int32,
    Hq: fx.Int32,
    Hkv: fx.Int32,
    G: fx.Int32,
    nkvt: fx.Int32,
    cshift: fx.Int32,
    causal: fx.Int32,
    nblk: fx.Int32,
    ngrp: fx.Int32,
    nb: fx.Int32,
    stream: fx.Stream,
):
    # grid = (Hq / DQ_NW head groups, Sq / DQ_BQW query tiles, B)
    k_dqg(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal).launch(
        grid=(ngrp, nblk, nb), block=(DQ_NW * 32, 1, 1), stream=stream
    )


_G07_CLAMP = True  # k_dkdv takes the batch count as its last argument (true-extent clamp)


# ================================================================= redsp ==========
# Split-K fold for dK/dV: ONE kernel, ONE flat pass over BOTH tensors, bf16 written
# straight out with no fp32 temporary. A host-side `sum(0)` + `.to(bf16)` per tensor is
# four launches and latency-bound (measured ~18% of a small shape's backward time).
# The summation order is `sp = 0, 1, .. nsp-1`, ascending and fixed, which is the
# determinism guarantee; it matches `torch.sum(0)`'s order.
RED_THREADS = 256
RED_VEC = 4  # fp32 dwordx4 in, bf16 x4 out


@flyc.kernel(known_block_size=[RED_THREADS, 1, 1])
def k_redsp(DKP: fx.Tensor, DVP: fx.Tensor, DK: fx.Tensor, DV: fx.Tensor, n_vec: fx.Int32, nsp: fx.Int32):
    """dk[i] = sum_{sp<nsp} dkp[sp, i], dv likewise; fp32 in, bf16 out, one pass."""
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    tile = bid * fx.Int32(RED_THREADS) + tid  # index in units of RED_VEC elements

    # Bound by an explicit predicate, not by the buffer descriptor. With D = 128 every
    # accepted shape divides RED_THREADS*RED_VEC, so the clamp never fires; it keeps a
    # non-dividing shape correct.
    ok = tile < n_vec
    t = ok.select(tile, fx.Int32(0))

    g_dkp = _bv(DKP, nsp * n_vec * fx.Int32(RED_VEC * 4), fx.Float32, RED_VEC)
    g_dvp = _bv(DVP, nsp * n_vec * fx.Int32(RED_VEC * 4), fx.Float32, RED_VEC)
    g_dk = _bv(DK, n_vec * fx.Int32(RED_VEC * 2), fx.BFloat16, RED_VEC)
    g_dv = _bv(DV, n_vec * fx.Int32(RED_VEC * 2), fx.BFloat16, RED_VEC)

    @flyc.jit
    def redloop(state, n):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            kacc = fx.Vector(carried[0])
            vacc = fx.Vector(carried[1])
            off = t + ii * n_vec
            kx = _ldv(g_dkp, off, fx.Float32, RED_VEC)
            vx = _ldv(g_dvp, off, fx.Float32, RED_VEC)
            kn = [kacc[c] + kx[c] for c in range_constexpr(RED_VEC)]
            vn = [vacc[c] + vx[c] for c in range_constexpr(RED_VEC)]
            final = yield [
                _ir(fx.Vector.from_elements(kn, dtype=fx.Float32)),
                _ir(fx.Vector.from_elements(vn, dtype=fx.Float32)),
            ]
        return final

    init = [_ir(fx.Vector.filled(RED_VEC, 0.0, fx.Float32)) for _ in range(2)]
    out = redloop(init, nsp)
    kacc = fx.Vector(out[0])
    vacc = fx.Vector(out[1])
    dst = ok.select(tile, n_vec)  # off-range lanes are parked past the view
    _stv([kacc[c].to(fx.BFloat16) for c in range_constexpr(RED_VEC)], g_dk, dst, fx.BFloat16)
    _stv([vacc[c].to(fx.BFloat16) for c in range_constexpr(RED_VEC)], g_dv, dst, fx.BFloat16)


@flyc.jit
def launch_redsp(DKP, DVP, DK, DV, n_vec: fx.Int32, nsp: fx.Int32, nblk: fx.Int32, stream: fx.Stream):
    k_redsp(DKP, DVP, DK, DV, n_vec, nsp).launch(grid=(nblk, 1, 1), block=(RED_THREADS, 1, 1), stream=stream)


# Split-K fold for dQ: k_redsp with one tensor. The determinism guarantee is the fixed
# ascending order `sp = 0, 1, .. nsp-1`; since k_dq_sp keeps the masked tail whole on the
# LAST split (see _dq_impl), folding ascending in sp adds each query's kv blocks in
# ascending kv order, the same order the unsplit k_dq uses.
@flyc.kernel(known_block_size=[RED_THREADS, 1, 1])
def k_redsp_q(DQP: fx.Tensor, DQ: fx.Tensor, n_vec: fx.Int32, nsp: fx.Int32):
    """dq[i] = sum_{sp<nsp} dqp[sp, i]; fp32 in, bf16 out, one pass."""
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    tile = bid * fx.Int32(RED_THREADS) + tid  # index in units of RED_VEC elements

    ok = tile < n_vec
    t = ok.select(tile, fx.Int32(0))

    g_dqp = _bv(DQP, nsp * n_vec * fx.Int32(RED_VEC * 4), fx.Float32, RED_VEC)
    g_dq = _bv(DQ, n_vec * fx.Int32(RED_VEC * 2), fx.BFloat16, RED_VEC)

    @flyc.jit
    def redloop_q(state, n):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            qacc = fx.Vector(carried[0])
            off = t + ii * n_vec
            qx = _ldv(g_dqp, off, fx.Float32, RED_VEC)
            qn = [qacc[c] + qx[c] for c in range_constexpr(RED_VEC)]
            final = yield [_ir(fx.Vector.from_elements(qn, dtype=fx.Float32))]
        return final

    init = [_ir(fx.Vector.filled(RED_VEC, 0.0, fx.Float32))]
    out = redloop_q(init, nsp)
    # k_redsp carries TWO vectors and gets a sequence back; this loop carries ONE and
    # flyc.jit unpacks a single-element carried state to the value itself at the return
    # boundary (inside the body `carried[0]` still indexes a sequence). Normalise.
    out = list(out) if isinstance(out, (list, tuple)) else [out]
    qacc = fx.Vector(out[0])
    dst = ok.select(tile, n_vec)  # off-range lanes are parked past the view
    _stv([qacc[c].to(fx.BFloat16) for c in range_constexpr(RED_VEC)], g_dq, dst, fx.BFloat16)


@flyc.jit
def launch_redsp_q(DQP, DQ, n_vec: fx.Int32, nsp: fx.Int32, nblk: fx.Int32, stream: fx.Stream):
    k_redsp_q(DQP, DQ, n_vec, nsp).launch(grid=(nblk, 1, 1), block=(RED_THREADS, 1, 1), stream=stream)
