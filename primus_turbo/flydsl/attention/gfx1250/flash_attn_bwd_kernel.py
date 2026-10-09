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

Written for Primus-Turbo: a batch/head grid with GQA and bottom-right causal masking, plus
split-K variants for small grids.

    delta[b,h,s] = sum_d dO*O                              k_delta_bshd
    dV[kv,d] = sum_q P^T dO ; dK[kv,d] = sum_q dS^T Q      k_dkdv   (k_dkdv_sp: split-K)
    dQ[q,d]  = sum_kv dS K                                 k_dqg    (k_dq_sp: split-K)
    fold of split-K fp32 partials -> bf16                  k_redsp (dK/dV), k_redsp_q (dQ)

Layouts: q/o/do [B, Sq, Hq, D] bf16; k/v [B, Skv, Hkv, D] bf16; lse/delta [B, Hq, Sq] fp32,
lse in natural log; dq [B, Sq, Hq, D] and dk/dv [B, Skv, Hkv, D] bf16 (split-K partials are
fp32 [nsp, ...] workspaces folded by k_redsp / k_redsp_q).

GQA is reduced INSIDE k_dkdv: one workgroup owns a kv tile of one kv head and streams every
query tile of all `G = Hq/Hkv` q heads that share it, accumulating into the same registers.
Every output element is written exactly once, with no atomics; the split-K folds sum the
partials in a fixed ascending order. The backward is therefore deterministic run to run.

Every k_dkdv / k_dqg / k_dq_sp workgroup is a single wave32, so none of them needs an
s_barrier. k_dkdv and k_dqg feed their hot loops from a TDM_DEPTH-stage LDS ring that the
Tensor Data Mover (TDM) fills two iterations ahead: Q/dO for k_dkdv, K/V for k_dqg. The
S/dP GEMM operands of the next iteration are read back from the ring into VGPRs one
iteration ahead, so neither hot loop issues a global load.

Divisibility is required, not handled: interface.py enforces Sq % 64 == 0 and
Skv % 32 == 0.
"""

from .flydsl_version import require_flydsl

require_flydsl()

# UNSTABLE(gfx1250): raw boundaries with no stable equivalent in flydsl 0.3.4.1:
#   - upstream llvm.load / llvm.store of 16 B at a hand-built LDS !llvm.ptr<3> (the padded
#     X_ROW_B / S_ROW_B / EPI_CB images and the TDM ring read-back);
#   - ROCDL builders outside rocdl.__all__: wmma_f32_16x16x32_bf16 (reuseA hint),
#     ds_load_tr16_b128, exp2, readfirstlane, raw_ptr_buffer_load (voffset + soffset form),
#     sched_barrier and sched_group_barrier;
#   - the Tensor Data Mover: fx.rocdl.cdna5.make_tdm_atom (cdna5 is not in rocdl.__all__ yet)
#     and rocdl.tdm_ops.tensor_wait (s_wait_tensorcnt).
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops

from .flash_attn_utils import LOG2E, WAVE_SIZE, create_llvm_ptr

D = 128
DV8 = D // 8  # vec8 tiles per row of one head
NDT = D // 32  # WMMA k-steps to contract 128
NDO = D // 16  # 16-wide output tiles across d
NKV = 2  # 16-row kv sub-tiles per k_dkdv workgroup
NST = 2 * NKV * NDO  # dV/dK accumulators carried by k_dkdv's q loop
NEG = -3.0e38
BLOCK_KV = 32  # kv rows one k_dkdv workgroup owns (NKV x 16)
S_ROW_B = BLOCK_KV * 2 + 16  # 64 -> 80 B LDS row pitch. 16 dwords is a 4-way bank
# collision at 64 banks; 20 dwords walks all 64.
X_ROW_B = D * 2 + 16  # 256 -> 272 B LDS row pitch. 256 B = 64 dwords is exactly
# the full 64-way collision stride on gfx1250's 64x4B
# LDS: every staged row starts on bank 0, so the 16
# rows a ds_load_tr16_b128 phase touches all collide.
# 68 dwords gives row*4 mod 64 -- 16 distinct groups.

# Stages of the TDM LDS rings. While iteration i computes from stage i % 3, the tile of
# iteration i+1 sits in the next stage and the tile of iteration i+2 is being written into
# the one after. A stage is filled by two TDM ops (dO + Q, or K + V) and TDM ops retire in
# order, so waiting until TDM_IN_FLIGHT ops remain retires exactly the older stage in flight.
TDM_DEPTH = 3
TDM_IN_FLIGHT = 2 * (TDM_DEPTH - 2)

# sched_group_barrier instruction classes (LLVM AMDGPU scheduling-group mask bits).
_SG_VALU = 0x002
_SG_WMMA = 0x008
_SG_TRANS = 0x400

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
    # UNSTABLE(gfx1250): raw rocdl.exp2 is a bare v_exp_f32; fx.math.exp2 adds denorm range scaling.
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


def _wmma(a, b, c, reuse_a=False):
    """D = A x B + C on 16x16x32 bf16 -> fp32 tiles; reuse_a hints that A is the previous WMMA's A."""
    return rocdl.wmma_f32_16x16x32_bf16(
        fx.Vector.make_type(8, fx.Float32),
        fx.as_ir_value(a),
        fx.as_ir_value(b),
        c,
        reuseA=reuse_a,
        reuseB=False,
    ).result


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

    lane_in_row = tid % LANES_PER_ROW
    row_in_group = tid // LANES_PER_ROW
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
            acc = acc + fx.gpu.shuffle_xor(acc, 1 << sft, WAVE_SIZE)
        idx = bid * ROWS_DELTA + u * ROWS_PER_PASS + row_in_group
        ok = (lane_in_row == 0) & (idx < n_rows)
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
# Split-K over the q loop. When k_dkdv's grid is about one dispatch wave (one workgroup per
# SIMD) and the causal work skew is large, makespan is the longest workgroup and much of the
# machine idles; only splitting helps. PARTIAL=True
# (k_dkdv_sp) splits the unmasked q-pair range `nsp` ways, each split writing its own fp32
# workspace slice, folded by k_redsp. PARTIAL=False (k_dkdv) is the path for grids that
# already span several dispatch waves, where splitting is pure cost.
def _dkdv_impl(
    PARTIAL, Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, B_, nsp
):
    """One wave owns a BLOCK_KV-row kv tile of one kv head and streams every (q head, q pair).

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
    bid = fx.Int32(fx.block_idx.y)  # kv tile
    bat = fx.Int32(fx.block_idx.z)  # batch
    row = lane % 16
    half = lane // 16
    kv0 = bid * BLOCK_KV

    # Every buffer descriptor carries the tensor's TRUE byte extent, so the hardware bound
    # check makes any over-read return 0 instead of reading past the allocation. A clamp
    # that ever hit a live access would wreck accuracy, so it cannot hide a real bug.
    # Q and dO take no descriptor: the TDM reads them (_tdm_qdo).
    nkv_b = B_ * Skv * Hkv * (D * 2)  # k / v   bf16 [B, Skv, Hkv, D]
    nl_b = B_ * Hq * Sq * 4  # lse / delta fp32 [B, Hq, Sq]
    # dK/dV are stored as bf16 directly (the fp32 accumulator rounded once), saving the
    # extra full-tensor pass and launch a host-side fp32 -> bf16 conversion would cost.
    ndkv_b = B_ * Skv * Hkv * (D * 2)  # dk / dv bf16 [B, Skv, Hkv, D]
    g_k = _bv(K, nkv_b, fx.BFloat16, 8)
    g_v = _bv(V, nkv_b, fx.BFloat16, 8)
    g_lse = _bv(LSE, nl_b, fx.Float32)
    g_del = _bv(DEL, nl_b, fx.Float32)
    if PARTIAL:
        _np_b = nsp * B_ * Skv * Hkv * (D * 4)  # [nsp, B, Skv, Hkv, D] fp32
        g_dv = _bv(DV_, _np_b, fx.Float32)
        g_dk = _bv(DK, _np_b, fx.Float32)

    rs_kv = Hkv * DV8  # vec8 tiles between consecutive kv rows
    base_kv = bat * Skv * rs_kv + hkv * DV8

    # LDS: the Q/dO ring at offset 0 and the P/dS tiles exactly 65536 B up, so the output
    # GEMM's A reads (P, dS) and B reads (dO, Q) differ in address bit 16 whatever the physical
    # LDS base. This placement was chosen by measurement: it ran faster than packing the two
    # regions together. Occupancy is unchanged: 70656 B still allows 4 workgroups per CU, and
    # VGPR use (1 wave/SIMD) caps it at 4 anyway.
    LDS_SEG = 65536
    smem = fx.SharedAllocator().allocate(LDS_SEG + 2 * 32 * S_ROW_B)
    _lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    lds_p = _lds0 + LDS_SEG
    lds_ds = lds_p + 32 * S_ROW_B
    v8b = fx.Vector.make_type(8, fx.BFloat16)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + dt * 4
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(_ldv(buf, t + 2, fx.BFloat16, 8), list(range(16)))

    # LSE/delta of the full loop: voffset = 4*row (a loop-invariant VGPR) plus a wave-uniform
    # soffset (SGPR), so the loads cost no per-iteration VALU address math.
    rsrc_lse = rocdl.get_buffer_rsrc(fx.get_iter(g_lse))
    rsrc_del = rocdl.get_buffer_rsrc(fx.get_iter(g_del))
    voff_l = row * 4

    def _ldl(qt, gh, split_offset=True):
        """The 4 LSE/delta loads (hh x {lse, delta}) of query pair qt of q head hkv*G + gh,
        issued one iteration early and carried in the loop state as 1-wide vectors.

        split_offset=True (full loop and its prologue): the byte address
        4*(base_l + q0 + hh*16 + row) is split as voffset 4*row + soffset 4*(base_l + q0) + 64*hh.
        split_offset=False (masked loop): plain per-lane buffer loads."""
        qh = hkv * G + gh
        q0 = qt * 32
        out = []
        base_l = (bat * Hq + qh) * Sq
        if const_expr(split_offset):
            sb = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, ((base_l + q0) * 4).ir_value()))
            for hh in range_constexpr(2):
                so = sb + hh * 64
                for rs_ in (rsrc_lse, rsrc_del):
                    x = fx.Float32(rocdl.raw_ptr_buffer_load(fx.Float32.ir_type, rs_, voff_l, so))
                    out.append(fx.Vector.from_elements([x], dtype=fx.Float32))
            return out
        for hh in range_constexpr(2):
            qg = q0 + hh * 16 + row
            out.append(_ldv(g_lse, base_l + qg, fx.Float32, 1))
            out.append(_ldv(g_del, base_l + qg, fx.Float32, 1))
        return out

    # Q/dO staging through the TDM into the TDM_DEPTH-stage ring. One TDM op per tensor writes
    # the [32 q rows][D] bf16 tile: row r at r*X_ROW_B, element c at 2c, 16 B pad per row
    # (pad_interval = D elements, pad_amount = 8 elements). Stage s is [s*QDO_B, (s+1)*QDO_B):
    # dO at +0, Q at +32*X_ROW_B; the three stages (52224 B) stay in LDS segment 0.
    # Tile origin element ((bat*Sq + q0)*Hq + qh)*D, row stride Hq*D, outer extent Sq - q0,
    # which is >= 32 for every issued tile (Sq % 64 == 0 and qt is clamped into [0, nqt2-1]).
    QDO_B = 2 * 32 * X_ROW_B
    X_ROW_EL = X_ROW_B // 2
    _lds_bf_ty = fx.PointerType.get(
        elem_ty=fx.BFloat16.ir_type, address_space=fx.AddressSpace.Shared, alignment=16
    )
    _q_rs_el = Hq * D  # elements between consecutive q rows

    def _tdm_qdo(qt, gh, stage_off):
        qh = hkv * G + gh
        q0 = qt * 32
        off = (fx.Int64(bat * Sq + q0) * fx.Int64(Hq) + fx.Int64(qh)) * fx.Int64(D)
        valid = Sq - q0
        for src, lb in ((DO, _lds0 + stage_off), (Q, _lds0 + stage_off + 32 * X_ROW_B)):
            g_view = fx.Tensor(
                fx.make_view(fx.add_offset(fx.get_iter(src), off), fx.make_layout((32, D), (D, 1)))
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
                fx.make_view(fx.inttoptr(_lds_bf_ty, lb), fx.make_layout((32, D), (X_ROW_EL, 1)))
            )
            fx.copy_atom_call(atom, g_view, l_view)

    # K and V fragments for this kv tile are invariant over every query and every q head.
    kf = [[gfrag(g_k, base_kv, rs_kv, kv0 + kh * 16, dt) for dt in range(NDT)] for kh in range(NKV)]
    vf = [[gfrag(g_v, base_kv, rs_kv, kv0 + kh * 16, dt) for dt in range(NDT)] for kh in range(NKV)]
    lane_r = (lane // 16) * 8 + lane % 8
    lane_c = ((lane // 8) % 2) * 8
    # Lane-only parts of the two DS-phase address families. Every per-op difference is a
    # compile-time constant that folds into the 16-bit DS immediate (the largest is 13280), so
    # each family costs one v_add per iteration.
    lb_tr = lane_r * X_ROW_B + lane_c * 2  # b_do / b_q transposing loads
    lb_rd = row * X_ROW_B + half * 16  # _rdqd read-back

    # CAUSAL TILE SKIP. Under bottom-right causal, kv row j is attended only by queries
    # q >= j - cshift, so this workgroup's kv tile [kv0, kv0+BLOCK_KV) is untouched by every
    # query pair below qp_start = max(0, (kv0 - cshift) // 32). Pairs below it contribute
    # exactly zero to dK/dV (p = exp2(NEG*LOG2E) = 0), so skipping them is not an
    # approximation -- the result is bit-identical, only the work is gone.
    nqt2 = nqt // 2  # query tiles come in PAIRS

    def _clampqt(t):
        """Clamp a prefetch's query pair into [0, nqt2-1] so it never reads out of range."""
        t = (t < nqt2).select(t, nqt2 - 1)
        return (t < 0).select(fx.Int32(0), t)

    _c = kv0 - cshift
    qp_start = ((_c < 0).select(fx.Int32(0), _c)) // 32
    qp_start = (causal != 0).select(qp_start, fx.Int32(0))
    nqp_eff = nqt2 - qp_start

    def _rdqd(stage_off, rbase=None):
        """The 32 ds_load_b128 that read one ring stage's S/dP B operands into VGPRs. Entry
        hh*16 + dt*2 + u is Q row hh*16+row, bytes half*16 + dt*64 + u*32; entry
        hh*16 + 8 + dt*2 + u is the same chunk of dO. Every address is rbase + a constant."""
        if const_expr(rbase is None):
            rbase = _lds0 + stage_off + lb_rd
        out = []
        for hh in range_constexpr(2):
            for qo in (32 * X_ROW_B, 0):  # Q, then dO
                for dt in range_constexpr(NDT):
                    for u in range_constexpr(2):
                        addr = rbase + (hh * 16 * X_ROW_B + qo + dt * 64 + u * 32)
                        out.append(fx.Vector(llvm_dialect.load(v8b, create_llvm_ptr(addr, address_space=3))))
        return out

    # (a) Loop order: query-pair OUTER, q-head INNER (iteration ii = qi*G + gh). This puts
    #     every masked iteration at the front, which makes (b) a clean loop split, and the G
    #     q heads of one GQA group are G*D*2 contiguous bytes of one q row, so sweeping them
    #     back to back reads each row once. The fp32 accumulation order is fixed, so the
    #     result is deterministic run to run.
    # (b) Mask split. Under bottom-right causality at most one query pair per q head
    #     straddles this workgroup's kv tile; every later pair has
    #     kv0+BLOCK_KV-1 <= q0 + cshift, so the mask predicate is provably false there.
    #     qloop_mask runs the straddling pairs with the mask; qloop_full, the hot loop,
    #     carries no mask at all.
    #
    # qloop_full (carry=True) gets this iteration's S/dP B operands in VGPRs (qd), read back
    # from ring stage ii % 3 during iteration ii-1 (or by the prologue), and its LSE/delta
    # (pre) likewise. At the top it TDM-issues the tile of iteration ii+2 into stage
    # (ii+2) % 3 == (ii-1) % 3 with no wait: that stage's last readers (the transposing loads
    # of ii-1, the read-back in ii-2) were consumed by WMMAs of ii-1, hence retired.
    # qloop_mask (carry=False) runs a handful of iterations: it loads its own tile into stage 0
    # and waits for it, and carries nothing but the accumulators.
    def _body(
        acc,
        pre,
        qt,
        gh,
        do_mask,
        qt_n,
        gh_n,
        carry=True,
        cur_off=None,
        nxt_off=None,
        pf_qt=None,
        pf_gh=None,
        qd=None,
        rb_off=None,
    ):
        q0 = qt * 32
        if const_expr(carry):
            # TDM first, then the LSE/delta prefetch, so the TDM descriptor's SALU does not
            # recycle the soffset SGPRs right behind the loads.
            rocdl.sched_barrier(0)
            _tdm_qdo(pf_qt, pf_gh, nxt_off)
            rocdl.sched_barrier(0)
            nxt = _ldl(qt_n, gh_n)
        else:
            nxt = None
            cur_off = fx.Int32(0)
            pre = _ldl(qt, gh, split_offset=False)
            _tdm_qdo(qt, gh, cur_off)
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(0)
            rocdl.sched_barrier(0)
        lds_do = _lds0 + cur_off

        # The S/dP B operands: lane (row, half) holds row hh*16+row, bytes half*16 + dt*64
        # (+32) of Q and dO, the halves gfrag would have loaded from global memory.
        ops = qd if const_expr(carry) else _rdqd(cur_off)
        pst = []  # P/dS stores, deferred into the DS phase
        for hh in range_constexpr(2):
            qp = [(ops[hh * 16 + dt * 2], ops[hh * 16 + dt * 2 + 1]) for dt in range_constexpr(NDT)]
            dp = [(ops[hh * 16 + 8 + dt * 2], ops[hh * 16 + 8 + dt * 2 + 1]) for dt in range_constexpr(NDT)]
            qfr = [qp[dt][0].shuffle(qp[dt][1], list(range(16))) for dt in range_constexpr(NDT)]
            dfr = [dp[dt][0].shuffle(dp[dt][1], list(range(16))) for dt in range_constexpr(NDT)]
            # LSE/delta were prefetched an iteration early (_ldl); q_glob is pure VALU and is
            # still needed by the causal mask below.
            q_glob = q0 + hh * 16 + row
            lse_q = pre[hh * 2][0]
            del_q = pre[hh * 2 + 1][0]
            for kh in range_constexpr(NKV):
                s_acc = fx.as_ir_value(fx.Vector.filled(8, 0.0, fx.Float32))
                p_acc = fx.as_ir_value(fx.Vector.filled(8, 0.0, fx.Float32))
                for dt in range_constexpr(NDT):
                    s_acc = _wmma(kf[kh][dt], qfr[dt], s_acc)
                    p_acc = _wmma(vf[kh][dt], dfr[dt], p_acc)
                sv, pv_ = fx.Vector(s_acc), fx.Vector(p_acc)
                # causal, BOTTOM-RIGHT: query q attends kv <= q + (Skv - Sq).
                kvb = kv0 + kh * 16 + half * 8
                if const_expr(do_mask):
                    masked = [
                        ((kvb + si > q_glob + cshift) & (causal != 0)).select(fx.Float32(NEG), sv[si] * scale)
                        for si in range(8)
                    ]
                else:
                    masked = [sv[si] * scale for si in range(8)]
                pf = [_exp2((masked[si] - lse_q) * LOG2E) for si in range(8)]
                p_l = [x.to(fx.BFloat16) for x in pf]
                ds_l = [(pf[si] * (pv_[si] - del_q) * scale).to(fx.BFloat16) for si in range(8)]
                off = (hh * 16 + row) * S_ROW_B + kh * 32 + half * 16
                pst.append((fx.Vector.from_elements(p_l, dtype=fx.BFloat16), lds_p + off))
                pst.append((fx.Vector.from_elements(ds_l, dtype=fx.BFloat16), lds_ds + off))

        # ONE DS phase per iteration, fenced off from the S/dP/softmax part. In order: the 8
        # P/dS stores, the transposing loads of the dK/dV GEMM operands, and (carry) the tensor
        # wait plus the 32 ds_load_b128 read-back of the NEXT iteration's stage. DS completes
        # in order, so the first dK/dV WMMA waits only for the stores and its own operands.
        # The DS bases are computed BEFORE the fence, so their VALU->DS latency hides under the
        # S/dP phase instead of stalling the first load of each family.
        tr_base = lds_do + lb_tr
        if const_expr(carry):
            rb_base = _lds0 + rb_off + lb_rd
        rocdl.sched_barrier(0)
        for v_, a_ in pst:
            llvm_dialect.store(fx.as_ir_value(v_), create_llvm_ptr(a_, address_space=3))
        # No fx.barrier() here: the workgroup is one wave32, so s_barrier is a no-op, and the
        # all-counter wait it would bring (s_wait_loadcnt_dscnt 0x0) would drain the in-flight
        # prefetch. The LDS RAW wait is derived from the memory dependence.

        # 32 queries staged, so the contraction is FULL: rows [lane_r] and [lane_r+16]
        # concatenate in lane into a v16 operand, no padding zeros.
        def tr(base, rowb):
            return fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base, address_space=3))).shuffle(
                fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base + 16 * rowb, address_space=3))),
                list(range(16)),
            )

        # The B operands (dO, Q) are shared by BOTH kv sub-tiles -- one transposing load feeds
        # two WMMAs. The A operands differ only by a 16-column (32-byte) offset into the same
        # [q][kv] tile.
        def _a_tr(kh, base_):
            col = lane_c * 2 + kh * 32
            return tr(base_ + lane_r * S_ROW_B + col, S_ROW_B)

        if const_expr(carry):
            # Loads issued in the order the dK/dV WMMAs consume them (kh 0: a_p x b_do,
            # a_ds x b_q; then kh 1's A operands), each group fenced so the scheduler keeps it:
            # the first 8 dK/dV WMMAs then wait for 26 DS ops (the 8 stores and 18 loads)
            # instead of nearly all of them.
            a_pk = [None] * NKV
            a_dsk = [None] * NKV
            a_pk[0] = _a_tr(0, lds_p)
            rocdl.sched_barrier(0)
            b_do = [tr(tr_base + dtile * 32, X_ROW_B) for dtile in range_constexpr(NDO)]
            rocdl.sched_barrier(0)
            a_dsk[0] = _a_tr(0, lds_ds)
            rocdl.sched_barrier(0)
            b_q = [tr(tr_base + (32 * X_ROW_B + dtile * 32), X_ROW_B) for dtile in range_constexpr(NDO)]
            rocdl.sched_barrier(0)
            for kh in range_constexpr(1, NKV):
                a_pk[kh] = _a_tr(kh, lds_p)
                a_dsk[kh] = _a_tr(kh, lds_ds)
            # Outstanding TDM here: stage (ii+1) % 3 (issued in ii-1) and (ii+2) % 3 (issued at
            # the top of this iteration). The wait retires the older one, about to be read.
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(TDM_IN_FLIGHT)
            rocdl.sched_barrier(0)
            rb = _rdqd(rb_off, rb_base)
        else:
            a_pk = []
            a_dsk = []
            for kh in range_constexpr(NKV):
                a_pk.append(_a_tr(kh, lds_p))
                a_dsk.append(_a_tr(kh, lds_ds))
            b_do = []
            b_q = []
            for dtile in range_constexpr(NDO):
                b_do.append(tr(tr_base + dtile * 32, X_ROW_B))
                b_q.append(tr(tr_base + (32 * X_ROW_B + dtile * 32), X_ROW_B))
        rocdl.sched_barrier(0)
        new = [None] * (2 * NKV * NDO)
        for kh in range_constexpr(NKV):
            # A-operand reuse: gfx1250 WMMA can hint that A (or B) is identical to the
            # preceding matrix instruction's. Emitting two runs of NDO (a_p then a_ds) rather
            # than interleaving them holds A fixed across each run, so every WMMA after the
            # first in a run sets reuseA. B changes every op.
            for dtile in range_constexpr(NDO):
                new[kh * NDO + dtile] = _wmma(a_pk[kh], b_do[dtile], acc[kh * NDO + dtile], dtile > 0)
            for dtile in range_constexpr(NDO):
                i = (NKV + kh) * NDO + dtile
                new[i] = _wmma(a_dsk[kh], b_q[dtile], acc[i], dtile > 0)
        if const_expr(carry):
            # Fence the dK/dV WMMAs off from the back-edge copies of the LSE/delta prefetch,
            # so their s_wait_loadcnt sits at the END of the body.
            rocdl.sched_barrier(0)
            return new + [fx.as_ir_value(v) for v in nxt] + [fx.as_ir_value(v) for v in rb]
        return new

    # Division-free loop index: the decomposition ii == qi*G + gh (0 <= gh < G) is carried in
    # the loop state as two wave-uniform i32 counters advanced by a wrap, gh+1 == G -> (qi+1, 0),
    # instead of runtime divisions by G every iteration. qloop_full also carries the ring
    # offset cur = (ii % 3) * QDO_B.
    def _wrap(qc, gc):
        g1 = gc + 1
        nw = g1 < G
        return fx.Int32(nw.select(qc, qc + 1)), fx.Int32(nw.select(g1, fx.Int32(0)))

    @flyc.jit
    def qloop_mask(state, n, qt0):
        final = state
        for it, carried in range(fx.Index(0), fx.Index(n), 1, init=state):
            st = list(carried)
            qi, gh = fx.Int32(st[NST]), fx.Int32(st[NST + 1])
            qn, gn = _wrap(qi, gh)
            res = _body(st[:NST], None, qt0 + qi, gh, True, None, None, False)
            final = yield res + [fx.as_ir_value(qn), fx.as_ir_value(gn)]
        return final

    @flyc.jit
    def qloop_full(state, n, qt0):
        final = state
        for it, carried in range(fx.Index(0), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            jj = ii + 1
            st = list(carried)
            # carried: accumulators st[:NST], LSE/delta st[NST:NST+4], the read-back B
            # operands st[NST+4:-3], then the counters (cur, qi, gh).
            cur, qi, gh = fx.Int32(st[-3]), fx.Int32(st[-2]), fx.Int32(st[-1])
            qn, gn = _wrap(qi, gh)  # decomposition of ii+1
            # The prefetches of the last iterations would name pairs that do not exist; they
            # are clamped to the last one (n-1). Value-neutral: the redirected loads are
            # consumed by iterations that never run.
            live = jj < n
            qj = fx.Int32(live.select(qn, qi))
            gj = fx.Int32(live.select(gn, gh))
            q2, g2 = _wrap(qn, gn)  # decomposition of ii+2
            live2 = (ii + 2) < n
            pf_qt = qt0 + fx.Int32(live2.select(q2, qj))
            pf_gh = fx.Int32(live2.select(g2, gj))
            # TDM target stage (ii+2) % 3 == (ii-1) % 3: the stage before cur.
            nxo = fx.Int32((cur == 0).select(fx.Int32(2 * QDO_B), cur - QDO_B))
            # Stage (ii+1) % 3: the next iteration's, read back during this one.
            ncur = fx.Int32((cur == 2 * QDO_B).select(fx.Int32(0), cur + QDO_B))
            res = _body(
                st[:NST],
                [fx.Vector(v) for v in st[NST : NST + 4]],
                qt0 + qi,
                gh,
                False,
                qt0 + qj,
                gj,
                True,
                cur,
                nxo,
                pf_qt,
                pf_gh,
                [fx.Vector(v) for v in st[NST + 4 : -3]],
                ncur,
            )
            final = yield res + [fx.as_ir_value(ncur), fx.as_ir_value(qn), fx.as_ir_value(gn)]
        return final

    def _tdm_prologue(qt0, n):
        """Fill stages 0 and 1 for qloop_full's first two iterations, then retire stage 0 and
        read back iteration 0's B operands. Stage 1 gets iteration min(1, n-1)'s tile and both
        query pairs are clamped, so an empty or 1-iteration loop still only issues in-bounds
        tiles (never read)."""
        _tdm_qdo(_clampqt(qt0), fx.Int32(0), fx.Int32(0))
        # iteration min(1, n-1) is wrap(0, 0) when n > 1, else iteration 0.
        q1w, g1w = _wrap(fx.Int32(0), fx.Int32(0))
        two = fx.Int32(1) < n
        q1 = fx.Int32(two.select(q1w, fx.Int32(0)))
        g1 = fx.Int32(two.select(g1w, fx.Int32(0)))
        _tdm_qdo(_clampqt(qt0 + q1), g1, fx.Int32(QDO_B))
        rocdl.sched_barrier(0)
        tdm_ops.tensor_wait(TDM_IN_FLIGHT)
        rocdl.sched_barrier(0)
        return [fx.as_ir_value(v) for v in _rdqd(fx.Int32(0))]

    # Query pair `qt` is fully unmasked iff this workgroup's LARGEST key index is attended by
    # the pair's SMALLEST query: kv0 + BLOCK_KV-1 <= qt*32 + cshift, i.e.
    # qt >= ceil((kv0 + BLOCK_KV-1 - cshift)/32). cshift = Skv - Sq can be negative.
    _u = kv0 + (BLOCK_KV - 1) - cshift
    _qsf = (_u < 0).select(fx.Int32(0), (_u + 31) // 32)
    _qsf = (_qsf < nqt2).select(_qsf, nqt2)
    _nm = _qsf - qp_start
    _nm = (_nm < 0).select(fx.Int32(0), _nm)
    _nm = (_nm < nqp_eff).select(_nm, nqp_eff)
    nmaskp = (causal != 0).select(_nm, fx.Int32(0))

    init = [fx.as_ir_value(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(2 * NKV * NDO)]
    # Both loops start at ii = 0: (qi, gh) = (0, 0) and ring offset cur = 0.
    _z2 = [fx.as_ir_value(fx.Int32(0)), fx.as_ir_value(fx.Int32(0))]
    _z3 = [fx.as_ir_value(fx.Int32(0))] + _z2
    # The LSE/delta prefetch and the TDM prologue of qloop_full's first iteration
    # (qi = 0, gh = 0) sit between the two loops; the prologue issues even when the full loop
    # is empty, so its query pair is clamped. The last tensor_wait retires the final
    # (clamped, never read) prefetches -- or the prologue's, when the loop ran 0 times --
    # before the epilogue reuses the start of LDS, which overlaps the ring.
    if PARTIAL:
        # The masked query pairs come first (from qp_start) and stay with split 0, so the
        # mask split is unchanged. The remaining `_fn` unmasked pairs are cut into
        # contiguous chunks; every pair costs the same, so contiguous is balanced.
        _fn = nqp_eff - nmaskp
        _fn = (_fn < 0).select(fx.Int32(0), _fn)
        _ch = (_fn + nsp - 1) // nsp
        _off = sp * _ch
        _cnt = _fn - _off
        _cnt = (_cnt < 0).select(fx.Int32(0), _cnt)
        _cnt = (_cnt < _ch).select(_cnt, _ch)
        _mk = (sp != 0).select(fx.Int32(0), nmaskp)
        _qt0 = qp_start + nmaskp + _off
        out = qloop_mask(init + _z2, G * _mk, qp_start)
        _pc0 = _clampqt(_qt0)
        out = list(out)[:NST] + [fx.as_ir_value(v) for v in _ldl(_pc0, fx.Int32(0))]
        out = out + _tdm_prologue(_qt0, G * _cnt) + _z3
        out = qloop_full(out, G * _cnt, _qt0)
        tdm_ops.tensor_wait(0)
        base_o = ((sp * B_ + bat) * Skv * Hkv) * D + hkv * D
    else:
        out = qloop_mask(init + _z2, G * nmaskp, qp_start)
        _pc0 = _clampqt(qp_start + nmaskp)
        out = list(out)[:NST] + [fx.as_ir_value(v) for v in _ldl(_pc0, fx.Int32(0))]
        out = out + _tdm_prologue(qp_start + nmaskp, G * (nqp_eff - nmaskp)) + _z3
        out = qloop_full(out, G * (nqp_eff - nmaskp), qp_start + nmaskp)
        tdm_ops.tensor_wait(0)
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
                    kv = kv0 + kh * 16 + half * 8 + si
                    idx = base_o + kv * Hkv * D + dtile * 16 + row
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
        # It costs no LDS: the ring is dead here, and the four staging images are
        # 4 * 128 * EPI_CB = 24576 B, still inside segment 0, so the allocation and therefore
        # the 4-workgroup occupancy rung are unchanged.
        # EPI_CB = 48 B = 32 B of kv row + 16 B pad: 12 dwords walks c*12 mod 64,
        # 16 distinct bank groups, the same granularity X_ROW_B was chosen for.
        EPI_CB = 48
        g_dv8 = _bv(DV_, ndkv_b, fx.BFloat16, 8)
        g_dk8 = _bv(DK, ndkv_b, fx.BFloat16, 8)
        for kh in range_constexpr(NKV):
            lds_ev = _lds0 + (kh * 2 + 0) * 128 * EPI_CB
            lds_ek = _lds0 + (kh * 2 + 1) * 128 * EPI_CB
            for dtile in range_constexpr(NDO):
                ov = fx.Vector(out[kh * NDO + dtile])
                ok_ = fx.Vector(out[(NKV + kh) * NDO + dtile])
                o = (dtile * 16 + row) * EPI_CB + half * 16
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
            gt = base_kv + (kv0 + kh * 16 + row) * rs_kv
            for sub in range_constexpr(NDO):
                a = (sub * 16 + lane_r) * EPI_CB + lane_c * 2
                vv = fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(lds_ev + a, address_space=3)))
                kk = fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(lds_ek + a, address_space=3)))
                t = gt + sub * 2 + half
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


# ================================================================== dq_sp =========
KV_STEP = 32  # two 16-kv tiles: exactly one WMMA contraction of dS
NKT = KV_STEP // 16
BLOCK_Q = 64  # queries one k_dq_sp workgroup owns (NQ x 16)
NQ = BLOCK_Q // 16


# Split-K over the kv loop of dQ, the mirror of k_dkdv_sp. The dQ grid is
# ceil(Sq/BLOCK_Q)*Hq*B workgroups of one wave32 each, so at 1 wave/SIMD a small grid (e.g.
# 128 workgroups) leaves most SIMDs idle, and splitting is the only way to add workgroups.
# k_dq_sp cuts the UNMASKED kv range `nsp` ways, each split
# writing its own fp32 [nsp, B, Sq, Hq, D] workspace slice, folded by k_redsp_q. Grids that
# fill the machine take k_dqg instead.
def _dq_sp_impl(Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, B_, nsp):
    """One wave owns BLOCK_Q queries of one q head and one split of the 32-key blocks.

    grid = (Hq * nsp, Sq/BLOCK_Q, B). The A operand of the dQ GEMM comes free: the two
    kv-tile dS^T accumulators concatenate in lane into one v16, so the only LDS traffic is
    staging K.
    """
    lane = fx.Int32(fx.thread_idx.x)
    # Longest-first dispatch. Under bottom-right causality query tile t streams
    # ceil((t*BLOCK_Q + BLOCK_Q + cshift) / KV_STEP) kv blocks, so work grows with t.
    # The q head is grid.x (cheap, uniform work, fastest-varying) and the query tile is
    # grid.y walked DESCENDING, so the longest tiles are issued first and the tail of the
    # dispatch is the cheapest. Pure dispatch order: results are bitwise unchanged.
    # XCD-major q-head remap, tuned on the assumption (not a documented fact) that
    # workgroups go round-robin on the linearised id to 8 XCDs: with grid.x = Hq = 32 one
    # XCD would then get q heads {c, c+8, c+16, c+24} -- four different kv heads. Permuting
    # x by (x%8)*(Hq/8) + x/8 puts q heads {4c..4c+3} on XCD c: one kv head per XCD, a
    # quarter of the K/V L2 footprint. It is a bijection of [0, Hq) whenever Hq % 8 == 0
    # (identity otherwise), so it only changes dispatch order, never results.
    # grid.x carries the split index in its low digits, as in launch_dkdv_sp; the remap is
    # applied to the decoded q head.
    _nx = fx.Int32(8)
    _fx = fx.Int32(fx.block_idx.x)  # q head * nsp + split
    _bx = _fx // nsp
    sp = _fx - _bx * nsp
    qh = (Hq % _nx == 0).select((_bx % _nx) * (Hq // _nx) + _bx // _nx, _bx)
    bid = (Sq + (BLOCK_Q - 1)) // BLOCK_Q - 1 - fx.Int32(fx.block_idx.y)  # query tile, descending
    bat = fx.Int32(fx.block_idx.z)  # batch
    row = lane % 16
    half = lane // 16
    q0 = bid * BLOCK_Q
    hkv = qh // G

    g_q = _bv(Q, 1 << 30, fx.BFloat16, 8)
    g_k = _bv(K, 1 << 30, fx.BFloat16, 8)
    g_v = _bv(V, 1 << 30, fx.BFloat16, 8)
    g_do = _bv(DO, 1 << 30, fx.BFloat16, 8)
    g_lse = _bv(LSE, 1 << 28, fx.Float32)
    g_del = _bv(DEL, 1 << 28, fx.Float32)
    # [nsp, B, Sq, Hq, D] fp32. TRUE byte extent, not the flat 1 GiB of the inputs: the
    # workspace is the one buffer in this kernel whose size depends on nsp, and an over-read
    # of it would land in another split's slice. The extent is computed in Int64: in Int32 it
    # can wrap (e.g. to 0 at nsp=8, B=4, Sq=8192, Hq=32), and a descriptor with num_records 0
    # silently drops every dQ write.
    g_dq = _bv(
        DQ,
        (nsp.to(fx.Int64) * B_.to(fx.Int64) * Sq.to(fx.Int64) * Hq.to(fx.Int64) * fx.Int64(D * 4)),
        fx.Float32,
    )

    rs_q = Hq * DV8
    rs_kv = Hkv * DV8
    base_q = bat * Sq * rs_q + qh * DV8
    base_kv = bat * Skv * rs_kv + hkv * DV8
    base_l = (bat * Hq + qh) * Sq

    smem = fx.SharedAllocator().allocate(KV_STEP * X_ROW_B)
    lds_k = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    v8b = fx.Vector.make_type(8, fx.BFloat16)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + dt * 4
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(_ldv(buf, t + 2, fx.BFloat16, 8), list(range(16)))

    def gfrag2(buf, base, rs, r, dt):
        """gfrag's two halves before the shuffle: cols [c, c+8) and [c+16, c+24) of
        row r+row, where c = half*8 + dt*32. Each is exactly one 16-byte LDS chunk."""
        t = base + (r + row) * rs + half + dt * 4
        return (_ldv(buf, t, fx.BFloat16, 8), _ldv(buf, t + 2, fx.BFloat16, 8))

    # Q, dO, lse and delta are invariant over the whole kv loop: hoist them.
    # The workgroup owns BLOCK_Q queries as NQ 16-row tiles. K and V (global fragments,
    # LDS staging and the tr16 transpose of it) are invariant over queries, so each of
    # them serves all NQ tiles.
    qf = [[gfrag(g_q, base_q, rs_q, q0 + qh_ * 16, dt) for dt in range(NDT)] for qh_ in range(NQ)]
    dof = [[gfrag(g_do, base_q, rs_q, q0 + qh_ * 16, dt) for dt in range(NDT)] for qh_ in range(NQ)]
    q_glob = [q0 + qh_ * 16 + row for qh_ in range(NQ)]
    lse_q = [_ld1(g_lse, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQ)]
    del_q = [_ld1(g_del, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQ)]
    lane_r = (lane // 16) * 8 + lane % 8
    lane_c = ((lane // 8) % 2) * 8

    # Prefetch: the 32 K/V buffer_load_b128 an iteration consumes are issued one iteration
    # early and carried in the kvloop_full scf.for state. Only the FULL loop carries it; the
    # masked loop (a handful of iterations) loads its own, so the two carried tuples differ
    # in shape and the allocator does not duplicate them.
    def _ldkv(kv0):
        out = []
        for kt in range_constexpr(NKT):
            for dt in range_constexpr(NDT):
                a, b = gfrag2(g_k, base_kv, rs_kv, kv0 + kt * 16, dt)
                c, d = gfrag2(g_v, base_kv, rs_kv, kv0 + kt * 16, dt)
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
            s_acc = [fx.as_ir_value(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(NQ)]
            p_acc = [fx.as_ir_value(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(NQ)]
            ko = (kt * 16 + row) * X_ROW_B + half * 16
            for dt in range_constexpr(NDT):
                # One pair of K/V fragments (from the carried prefetch) feeds all NQ
                # query tiles.
                pi = (kt * NDT + dt) * 4
                kp = (pre[pi], pre[pi + 1])
                for u in range_constexpr(2):
                    llvm_dialect.store(
                        fx.as_ir_value(kp[u]),
                        create_llvm_ptr(lds_k + ko + (dt * 64 + u * 32), address_space=3),
                    )
                kfr = kp[0].shuffle(kp[1], list(range(16)))
                vfr = pre[pi + 2].shuffle(pre[pi + 3], list(range(16)))
                for qh_ in range_constexpr(NQ):
                    s_acc[qh_] = _wmma(kfr, qf[qh_][dt], s_acc[qh_])
                    p_acc[qh_] = _wmma(vfr, dof[qh_][dt], p_acc[qh_])
            for qh_ in range_constexpr(NQ):
                sv, pv_ = fx.Vector(s_acc[qh_]), fx.Vector(p_acc[qh_])
                if const_expr(do_mask):
                    masked = [
                        ((kv0 + kt * 16 + half * 8 + si > q_glob[qh_] + cshift) & (causal != 0)).select(
                            fx.Float32(NEG), sv[si] * scale
                        )
                        for si in range(8)
                    ]
                else:
                    masked = [sv[si] * scale for si in range(8)]
                pf = [_exp2((masked[si] - lse_q[qh_]) * LOG2E) for si in range(8)]
                ds_halves[qh_].append(
                    [(pf[si] * (pv_[si] - del_q[qh_]) * scale).to(fx.BFloat16) for si in range(8)]
                )
        # FREE: the two kv-tile accumulators concatenate in-lane into the dS A-operand.
        a_ds = [
            fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1], dtype=fx.BFloat16)
            for qh_ in range_constexpr(NQ)
        ]
        # No fx.barrier() between the K stores and their transposing loads: one wave32 per
        # workgroup; the LDS RAW wait is derived from the memory dependence.

        new = [None] * (NQ * NDO)
        for dtile in range_constexpr(NDO):
            base = lds_k + lane_r * X_ROW_B + (lane_c + dtile * 16) * 2
            b_k = fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base, address_space=3))).shuffle(
                fx.Vector(
                    rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base + 16 * X_ROW_B, address_space=3))
                ),
                list(range(16)),
            )
            for qh_ in range_constexpr(NQ):
                new[qh_ * NDO + dtile] = _wmma(a_ds[qh_], b_k, acc[qh_ * NDO + dtile])
        if const_expr(carry):
            return new + [fx.as_ir_value(v) for v in nxt]
        return new

    # The iteration index is rebased onto this split's chunk. The prefetch clamp stays
    # RELATIVE (jj in [0, n)) and the base is added after it, so it never reaches past this
    # chunk's last block.
    @flyc.jit
    def kvloop_full(state, n, it0):
        final = state
        for it, carried in range(fx.Index(0), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            jj = ii + 1
            jj = (jj < n).select(jj, n - 1)
            st = list(carried)
            final = yield _body(
                st[: NQ * NDO],
                [fx.Vector(v) for v in st[NQ * NDO :]],
                (ii + it0) * KV_STEP,
                False,
                (jj + it0) * KV_STEP,
            )
        return final

    @flyc.jit
    def kvloop_mask(state, n, it0):
        final = state
        for it, carried in range(fx.Index(0), fx.Index(n), 1, init=state):
            final = yield _body(list(carried), None, (fx.Int32(it) + it0) * KV_STEP, True, None, False)
        return final

    # Causal tile skip, the mirror of k_dkdv's. Queries [q0, q0+BLOCK_Q) reach at most key
    # q0 + BLOCK_Q-1 + cshift, so every kv block past that is fully masked and contributes
    # exactly zero to dQ. Bit-identical, not an approximation.
    _lim = (q0 + BLOCK_Q + cshift + (KV_STEP - 1)) // KV_STEP
    _lim = (_lim < 1).select(fx.Int32(1), _lim)
    _lim = (_lim < nkvt).select(_lim, nkvt)
    nkvt_eff = (causal != 0).select(_lim, nkvt)

    # Block `it` is fully unmasked iff its largest key index is attended by this
    # block's SMALLEST query: it*KV_STEP + KV_STEP-1 <= q0 + cshift, i.e.
    # it < (q0 + cshift + 1) // KV_STEP. cshift = Skv - Sq can be negative (Sq > Skv),
    # so the numerator is guarded before the division.
    _t = q0 + cshift + 1
    _nf = (_t < 0).select(fx.Int32(0), _t // KV_STEP)
    _nf = (_nf < nkvt_eff).select(_nf, nkvt_eff)
    nfull = (causal != 0).select(_nf, nkvt_eff)

    init = [fx.as_ir_value(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NQ * NDO)]
    # The UNMASKED range [0, nfull) is cut into nsp contiguous chunks; every block in it
    # costs the same, so contiguous is balanced. The masked tail [nfull, nkvt_eff) is at
    # most two blocks and is NOT cut -- it stays whole on the LAST split, which makes the
    # fold order over sp = 0..nsp-1 exactly the ascending kv-block order of an unsplit pass.
    _ch = (nfull + nsp - 1) // nsp
    _off = sp * _ch
    _cnt = nfull - _off
    _cnt = (_cnt < 0).select(fx.Int32(0), _cnt)
    _cnt = (_cnt < _ch).select(_cnt, _ch)
    # The prologue prefetch is issued whether or not this split has work, and the K/V
    # descriptors carry a flat 1 GiB num_records, so an empty split at _off >= nkvt_eff
    # would issue a LIVE out-of-bounds read. Clamp the prefetch address; the loaded value
    # is dead in that case.
    _pf = (_off < nkvt_eff).select(_off, nkvt_eff - 1)
    _mk = (sp == nsp - 1).select(nkvt_eff - nfull, fx.Int32(0))
    out = kvloop_full(init + [fx.as_ir_value(v) for v in _ldkv(_pf * KV_STEP)], _cnt, _off)
    out = kvloop_mask(list(out)[: NQ * NDO], _mk, nfull)
    base_o = ((sp * B_ + bat) * Sq * Hq) * D + qh * D
    for qh_ in range_constexpr(NQ):
        for dtile in range_constexpr(NDO):
            ov = fx.Vector(out[qh_ * NDO + dtile])
            for si in range_constexpr(8):
                q_i = q0 + qh_ * 16 + half * 8 + si
                _st1(ov[si], g_dq, base_o + q_i * Hq * D + dtile * 16 + row, fx.Float32)


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
    _dq_sp_impl(Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, B_, nsp)


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
# k_dqg, the dQ kernel of the unsplit path (interface.py: nsp_q == 1). One wave owns DQ_BQW
# queries of one q head and streams every KV_STEP-row kv block. K and V arrive through the
# TDM_DEPTH-stage LDS ring; with i the kv-block index over both loops (full, then masked):
#   stage i % 3       tile i: its K half is the dQ GEMM's B operand (transposing loads)
#   stage (i+1) % 3   tile min(i+1, N-1): read back (32 ds_load_b128) into the S/dP A
#                     operands iteration i+1 carries in VGPRs
#   stage (i+2) % 3   tile min(i+2, N-1): TDM-issued at the top of iteration i
# where N is the number of kv blocks this query tile reaches. The masked loop continues the
# same ring, so the handoff from the full loop is just the carried state.
DQ_BQW = 64  # queries per k_dqg wave
KV_B = 2 * KV_STEP * X_ROW_B  # one ring stage: K [32][X_ROW_B], then V at +32*X_ROW_B


def _dqg_impl(Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal):
    """grid = (Hq, Sq/DQ_BQW, B); the q head is XCD-remapped and the query tile walked
    descending, as in k_dq_sp."""
    NQW = DQ_BQW // 16  # 16-row query tiles per wave
    lane = fx.Int32(fx.thread_idx.x)
    _nx = fx.Int32(8)
    _gx = fx.Int32(fx.block_idx.x)
    qh = (Hq % _nx == 0).select((_gx % _nx) * (Hq // _nx) + _gx // _nx, _gx)
    bid = Sq // DQ_BQW - 1 - fx.Int32(fx.block_idx.y)  # descending
    bat = fx.Int32(fx.block_idx.z)
    row = lane % 16
    half = lane // 16
    q0 = bid * DQ_BQW
    hkv = qh // G

    # k_dqg takes no batch count, so q / do / dq carry a flat 1 GiB extent and lse / delta
    # 256 MiB; interface.py refuses problems whose tensors would exceed them.
    g_q = _bv(Q, 1 << 30, fx.BFloat16, 8)
    g_do = _bv(DO, 1 << 30, fx.BFloat16, 8)
    g_lse = _bv(LSE, 1 << 28, fx.Float32)
    g_del = _bv(DEL, 1 << 28, fx.Float32)
    g_dq = _bv(DQ, 1 << 30, fx.BFloat16)

    rs_q = Hq * DV8
    base_q = bat * Sq * rs_q + qh * DV8
    base_l = (bat * Hq + qh) * Sq

    smem = fx.SharedAllocator().allocate(TDM_DEPTH * KV_B)
    _lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    v8b = fx.Vector.make_type(8, fx.BFloat16)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + dt * 4
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(_ldv(buf, t + 2, fx.BFloat16, 8), list(range(16)))

    # Q, dO, lse and delta are invariant over the whole kv loop.
    qf = [[gfrag(g_q, base_q, rs_q, q0 + qh_ * 16, dt) for dt in range(NDT)] for qh_ in range(NQW)]
    dof = [[gfrag(g_do, base_q, rs_q, q0 + qh_ * 16, dt) for dt in range(NDT)] for qh_ in range(NQW)]
    q_glob = [q0 + qh_ * 16 + row for qh_ in range(NQW)]
    lse_q = [_ld1(g_lse, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQW)]
    del_q = [_ld1(g_del, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQW)]
    # Per-row constants of the softmax and dS, hoisted out of the kv loops:
    #   p  = exp2(s*scale*LOG2E - lse*LOG2E) = exp2(fma(s, scale*LOG2E, -lse*LOG2E))
    #   ds = p * (dp - delta) * scale        = p * fma(dp, scale, -delta*scale)
    _c1q = scale * LOG2E
    _nlq = [lse_q[qh_] * -LOG2E for qh_ in range(NQW)]
    _ndq = [del_q[qh_] * (fx.Float32(0.0) - scale) for qh_ in range(NQW)]
    lane_r = (lane // 16) * 8 + lane % 8
    lane_c = ((lane // 8) % 2) * 8
    # Lane-only parts of the two DS address families; every per-op difference is a
    # compile-time immediate (at most 32*X_ROW_B + 16*X_ROW_B + 7*64 + 32 + 224 < 65536).
    lb_tr = lane_r * X_ROW_B + lane_c * 2  # dQ B operand (K) transposing loads
    lb_rd = row * X_ROW_B + half * 16  # S/dP A operand read-back

    X_ROW_EL = X_ROW_B // 2
    _lds_bf_ty = fx.PointerType.get(
        elem_ty=fx.BFloat16.ir_type, address_space=fx.AddressSpace.Shared, alignment=16
    )
    _kv_rs_el = Hkv * D  # elements between consecutive kv rows

    def _tdm_kv(kv0p, stage_off):
        """K, then V, tile [kv0p, kv0p+32) x [0, D) of (bat, hkv) into stage_off / +32 rows, in
        the padded X_ROW_B image. Origin element ((bat*Skv + kv0p)*Hkv + hkv)*D, row stride
        Hkv*D, outer extent Skv - kv0p >= 32 (every issued kv0p <= (nkvt-1)*32)."""
        off = (fx.Int64(bat * Skv + kv0p) * fx.Int64(Hkv) + fx.Int64(hkv)) * fx.Int64(D)
        valid = Skv - kv0p
        for src, lb in ((K, _lds0 + stage_off), (V, _lds0 + stage_off + KV_STEP * X_ROW_B)):
            g_view = fx.Tensor(
                fx.make_view(fx.add_offset(fx.get_iter(src), off), fx.make_layout((KV_STEP, D), (D, 1)))
            )
            atom = fx.rocdl.cdna5.make_tdm_atom(
                g_view,
                [valid, None],
                strides=[_kv_rs_el, None],
                num_warps=1,
                pad_interval=D,
                pad_amount=X_ROW_EL - D,
            )
            l_view = fx.Tensor(
                fx.make_view(fx.inttoptr(_lds_bf_ty, lb), fx.make_layout((KV_STEP, D), (X_ROW_EL, 1)))
            )
            fx.copy_atom_call(atom, g_view, l_view)

    def _rdkv(stage_off):
        """32 ds_load_b128: entry (kt*NDT + dt)*4 + {0,1,2,3} = K u0, K u1, V u0, V u1 of row
        kt*16+row, bytes half*16 + dt*64 + u*32 -- the 16 B chunks of the S/dP A operands."""
        rb = _lds0 + stage_off + lb_rd
        out = []
        for kt in range_constexpr(NKT):
            for dt in range_constexpr(NDT):
                for vo in (0, KV_STEP * X_ROW_B):
                    for u in range_constexpr(2):
                        addr = rb + (kt * 16 * X_ROW_B + vo + dt * 64 + u * 32)
                        out.append(fx.Vector(llvm_dialect.load(v8b, create_llvm_ptr(addr, address_space=3))))
        return out

    def _bks(cur):
        """The NDO dQ B operands: K of stage `cur`, transposed by ds_load_tr16_b128."""
        tb = _lds0 + cur + lb_tr
        out = []
        for dtile in range_constexpr(NDO):
            base = tb + dtile * 32
            out.append(
                fx.Vector(rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base, address_space=3))).shuffle(
                    fx.Vector(
                        rocdl.ds_load_tr16_b128(v8b, create_llvm_ptr(base + 16 * X_ROW_B, address_space=3))
                    ),
                    list(range(16)),
                )
            )
        return out

    def _smx(sv, pv_, qh_, kt, kv0, do_mask):
        """Softmax and dS of one (kv sub-tile kt, query tile qh_) -> 8 bf16 of the dS operand."""
        tt = [fx.Float32(fx.fma(sv[si], _c1q, _nlq[qh_])) for si in range(8)]
        if const_expr(do_mask):
            # causal, BOTTOM-RIGHT: query q attends kv <= q + (Skv - Sq).
            tt = [
                ((kv0 + kt * 16 + half * 8 + si > q_glob[qh_] + cshift) & (causal != 0)).select(
                    fx.Float32(NEG), tt[si]
                )
                for si in range(8)
            ]
        pf = [_exp2(tt[si]) for si in range(8)]
        return [(pf[si] * fx.Float32(fx.fma(pv_[si], scale, _ndq[qh_]))).to(fx.BFloat16) for si in range(8)]

    def _sgb(n_wmma):
        """Ask the scheduler for {1 WMMA, 2 VALU, 1 transcendental} per WMMA of the region."""
        for _ in range_constexpr(n_wmma):
            rocdl.sched_group_barrier(_SG_WMMA, 1, 0)
            rocdl.sched_group_barrier(_SG_VALU, 2, 0)
            rocdl.sched_group_barrier(_SG_TRANS, 1, 0)

    def _sdp(pre, kt, dt, s_acc, p_acc):
        """The S and dP WMMAs of k-step dt of kv sub-tile kt, for all NQW query tiles."""
        pi = (kt * NDT + dt) * 4
        kfr = pre[pi].shuffle(pre[pi + 1], list(range(16)))
        vfr = pre[pi + 2].shuffle(pre[pi + 3], list(range(16)))
        for qh_ in range_constexpr(NQW):
            s_acc[qh_] = _wmma(kfr, qf[qh_][dt], s_acc[qh_])
            p_acc[qh_] = _wmma(vfr, dof[qh_][dt], p_acc[qh_])

    def _ds_phase(cur, ncur):
        """The one DS phase of an iteration, fenced: the dQ B operands (transposing loads of
        stage `cur`), then retire stage (i+1) % 3 -- in flight with stage (i+2) % 3, issued
        at the top of this iteration -- and read it back for iteration i+1."""
        rocdl.sched_barrier(0)
        b_ks = _bks(cur)
        rocdl.sched_barrier(0)
        tdm_ops.tensor_wait(TDM_IN_FLIGHT)
        rocdl.sched_barrier(0)
        rb = _rdkv(ncur)
        rocdl.sched_barrier(0)
        return b_ks, rb

    def _zeros():
        return [fx.as_ir_value(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range_constexpr(NQW)]

    # Iteration i starts by TDM-issuing tile min(i+2, N-1) into stage (i+2) % 3 == (i-1) % 3
    # with no wait: that stage's last readers (the dQ transposing loads of i-1, the read-back
    # in i-2) were consumed by WMMAs of iteration i-1.
    def _body_mask(acc, pre, kv0, pf_kv0, cur, nxo, ncur):
        """Masked loop: S/dP WMMAs | DS phase | softmax and dS VALU (under the DS latency) | dQ."""
        rocdl.sched_barrier(0)
        _tdm_kv(pf_kv0, nxo)
        rocdl.sched_barrier(0)
        sp_acc = []
        for kt in range_constexpr(NKT):
            s_acc, p_acc = _zeros(), _zeros()
            for dt in range_constexpr(NDT):
                _sdp(pre, kt, dt, s_acc, p_acc)
            sp_acc.append((s_acc, p_acc))
        b_ks, rb = _ds_phase(cur, ncur)
        ds_halves = [[] for _ in range_constexpr(NQW)]
        for kt in range_constexpr(NKT):
            s_acc, p_acc = sp_acc[kt]
            for qh_ in range_constexpr(NQW):
                ds_halves[qh_].append(_smx(fx.Vector(s_acc[qh_]), fx.Vector(p_acc[qh_]), qh_, kt, kv0, True))
        a_ds = [
            fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1], dtype=fx.BFloat16)
            for qh_ in range_constexpr(NQW)
        ]
        new = [None] * (NQW * NDO)
        for qh_ in range_constexpr(NQW):
            for dtile in range_constexpr(NDO):
                new[qh_ * NDO + dtile] = _wmma(a_ds[qh_], b_ks[dtile], acc[qh_ * NDO + dtile])
        return new + [fx.as_ir_value(v) for v in rb]

    # The full loop computes exactly what _body_mask computes (minus the mask), with the
    # softmax/dS VALU interleaved with WMMAs that do not depend on it:
    #   - the kv sub-tile 0 S/dP WMMAs, fenced as their own region;
    #   - the sub-tile 1 S/dP WMMAs in NDT chunks: chunk j = the 2*NQW WMMAs of k-step j plus
    #     the softmax/dS of (sub-tile 0, query tile j);
    #   - the DS phase, then the softmax/dS of (sub-tile 1, query tile 0);
    #   - the dQ WMMAs in NQW chunks: chunk j = the NDO WMMAs of query tile j plus the
    #     softmax/dS of (sub-tile 1, query tile j+1).
    # Each chunk ends in sched_barrier(0), and inside it _sgb spreads the chunk's 24 VALU
    # (16 plain, 8 transcendental) over its 8 WMMAs. Operands, accumulators and the WMMA order
    # per accumulator are those of _body_mask; only the VALU placement differs.
    assert NDT == DQ_BQW // 16, "a sub-tile 1 chunk pairs k-step j with query tile j"

    def _body_full(acc, pre, kv0, pf_kv0, cur, nxo, ncur):
        rocdl.sched_barrier(0)
        _tdm_kv(pf_kv0, nxo)
        rocdl.sched_barrier(0)
        ds_halves = [[] for _ in range_constexpr(NQW)]
        s0, p0 = _zeros(), _zeros()
        for dt in range_constexpr(NDT):
            _sdp(pre, 0, dt, s0, p0)
        rocdl.sched_barrier(0)
        s1, p1 = _zeros(), _zeros()
        for dt in range_constexpr(NDT):
            _sdp(pre, 1, dt, s1, p1)
            ds_halves[dt].append(_smx(fx.Vector(s0[dt]), fx.Vector(p0[dt]), dt, 0, kv0, False))
            _sgb(2 * NQW)
            rocdl.sched_barrier(0)
        b_ks, rb = _ds_phase(cur, ncur)
        ds_halves[0].append(_smx(fx.Vector(s1[0]), fx.Vector(p1[0]), 0, 1, kv0, False))
        rocdl.sched_barrier(0)
        new = [None] * (NQW * NDO)
        for qh_ in range_constexpr(NQW):
            a_ds = fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1], dtype=fx.BFloat16)
            for dtile in range_constexpr(NDO):
                new[qh_ * NDO + dtile] = _wmma(a_ds, b_ks[dtile], acc[qh_ * NDO + dtile])
            if const_expr(qh_ + 1 < NQW):
                ds_halves[qh_ + 1].append(
                    _smx(fx.Vector(s1[qh_ + 1]), fx.Vector(p1[qh_ + 1]), qh_ + 1, 1, kv0, False)
                )
                _sgb(NDO)
            rocdl.sched_barrier(0)
        return new + [fx.as_ir_value(v) for v in rb]

    NACC = NQW * NDO
    NP = 4 * NKT * NDT

    def _kvloop(body):
        @flyc.jit
        def kvloop(state, n, i0, nlast):
            # carried: acc[NACC] + read-back S/dP A operands[NP] + cur (ring offset of stage i % 3)
            final = state
            for it, carried in range(fx.Index(0), fx.Index(n), 1, init=state):
                st = list(carried)
                cur = fx.Int32(st[-1])
                ii = fx.Int32(it) + i0
                kk = ii + (TDM_DEPTH - 1)
                kk = (kk < nlast).select(kk, nlast)  # min(i+2, N-1)
                # stage (i+2) % 3 == (i-1) % 3, the one before cur
                nxo = fx.Int32((cur == 0).select(fx.Int32((TDM_DEPTH - 1) * KV_B), cur - KV_B))
                # stage (i+1) % 3, the one after cur
                ncur = fx.Int32((cur == (TDM_DEPTH - 1) * KV_B).select(fx.Int32(0), cur + KV_B))
                res = body(
                    st[:NACC],
                    [fx.Vector(v) for v in st[NACC : NACC + NP]],
                    ii * KV_STEP,
                    kk * KV_STEP,
                    cur,
                    nxo,
                    ncur,
                )
                final = yield res + [fx.as_ir_value(ncur)]
            return final

        return kvloop

    kvloop_full = _kvloop(_body_full)
    kvloop_mask = _kvloop(_body_mask)

    # Causal tile skip and mask split, as in k_dq_sp: blocks [0, nfull) are fully unmasked,
    # [nfull, nkvt_eff) straddle the diagonal, and blocks past nkvt_eff contribute nothing.
    _lim = (q0 + DQ_BQW + cshift + (KV_STEP - 1)) // KV_STEP
    _lim = (_lim < 1).select(fx.Int32(1), _lim)
    _lim = (_lim < nkvt).select(_lim, nkvt)
    nkvt_eff = (causal != 0).select(_lim, nkvt)
    _t = q0 + cshift + 1
    _nf = (_t < 0).select(fx.Int32(0), _t // KV_STEP)
    _nf = (_nf < nkvt_eff).select(_nf, nkvt_eff)
    nfull = (causal != 0).select(_nf, nkvt_eff)
    nlast = nkvt_eff - 1  # >= 0: nkvt_eff >= 1

    # Prologue: stage s gets tile min(s, N-1) for s = 0..TDM_DEPTH-2; retire stage 0 and read
    # it back.
    _tdm_kv(fx.Int32(0), fx.Int32(0))
    for s_ in range_constexpr(1, TDM_DEPTH - 1):
        t1 = (fx.Int32(s_) < nlast).select(fx.Int32(s_), nlast)
        _tdm_kv(t1 * KV_STEP, fx.Int32(s_ * KV_B))
    rocdl.sched_barrier(0)
    tdm_ops.tensor_wait(TDM_IN_FLIGHT)
    rocdl.sched_barrier(0)
    rb0 = _rdkv(fx.Int32(0))

    init = [fx.as_ir_value(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NACC)]
    out = kvloop_full(
        init + [fx.as_ir_value(v) for v in rb0] + [fx.as_ir_value(fx.Int32(0))], nfull, fx.Int32(0), nlast
    )
    out = kvloop_mask(list(out), nkvt_eff - nfull, nfull, nlast)
    # Retire the last two (clamped, never read) prefetches before the wave ends: the
    # workgroup's LDS must not be written after it is released.
    tdm_ops.tensor_wait(0)
    base_o = bat * Sq * Hq * D + qh * D
    for qh_ in range_constexpr(NQW):
        for dtile in range_constexpr(NDO):
            ov = fx.Vector(out[qh_ * NDO + dtile])
            for si in range_constexpr(8):
                q_i = q0 + qh_ * 16 + half * 8 + si
                _st1(ov[si].to(fx.BFloat16), g_dq, base_o + q_i * Hq * D + dtile * 16 + row, fx.BFloat16)


@flyc.kernel(known_block_size=[32, 1, 1])
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
    # O is not read (delta comes from k_delta_bshd); it stays in the argument list, which is
    # part of the tuned kernel's ABI.
    _dqg_impl(Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal)


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
    nhq: fx.Int32,
    nb: fx.Int32,
    stream: fx.Stream,
):
    # grid = (Hq q heads, Sq / DQ_BQW query tiles, B)
    k_dqg(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal).launch(
        grid=(nhq, nblk, nb), block=(32, 1, 1), stream=stream
    )


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
    tile = bid * RED_THREADS + tid  # index in units of RED_VEC elements

    # Bound by an explicit predicate, not by the buffer descriptor. With D = 128 every
    # accepted shape divides RED_THREADS*RED_VEC, so the clamp never fires; it keeps a
    # non-dividing shape correct.
    ok = tile < n_vec
    t = ok.select(tile, fx.Int32(0))

    g_dkp = _bv(DKP, nsp * n_vec * (RED_VEC * 4), fx.Float32, RED_VEC)
    g_dvp = _bv(DVP, nsp * n_vec * (RED_VEC * 4), fx.Float32, RED_VEC)
    g_dk = _bv(DK, n_vec * (RED_VEC * 2), fx.BFloat16, RED_VEC)
    g_dv = _bv(DV, n_vec * (RED_VEC * 2), fx.BFloat16, RED_VEC)

    @flyc.jit
    def redloop(state, n):
        final = state
        for it, carried in range(fx.Index(0), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            kacc = fx.Vector(carried[0])
            vacc = fx.Vector(carried[1])
            off = t + ii * n_vec
            kx = _ldv(g_dkp, off, fx.Float32, RED_VEC)
            vx = _ldv(g_dvp, off, fx.Float32, RED_VEC)
            kn = [kacc[c] + kx[c] for c in range_constexpr(RED_VEC)]
            vn = [vacc[c] + vx[c] for c in range_constexpr(RED_VEC)]
            final = yield [
                fx.as_ir_value(fx.Vector.from_elements(kn, dtype=fx.Float32)),
                fx.as_ir_value(fx.Vector.from_elements(vn, dtype=fx.Float32)),
            ]
        return final

    init = [fx.as_ir_value(fx.Vector.filled(RED_VEC, 0.0, fx.Float32)) for _ in range(2)]
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
# LAST split (see _dq_sp_impl), folding ascending in sp adds each query's kv blocks in
# ascending kv order.
@flyc.kernel(known_block_size=[RED_THREADS, 1, 1])
def k_redsp_q(DQP: fx.Tensor, DQ: fx.Tensor, n_vec: fx.Int32, nsp: fx.Int32):
    """dq[i] = sum_{sp<nsp} dqp[sp, i]; fp32 in, bf16 out, one pass."""
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    tile = bid * RED_THREADS + tid  # index in units of RED_VEC elements

    ok = tile < n_vec
    t = ok.select(tile, fx.Int32(0))

    g_dqp = _bv(DQP, nsp * n_vec * (RED_VEC * 4), fx.Float32, RED_VEC)
    g_dq = _bv(DQ, n_vec * (RED_VEC * 2), fx.BFloat16, RED_VEC)

    @flyc.jit
    def redloop_q(state, n):
        final = state
        for it, carried in range(fx.Index(0), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            qacc = fx.Vector(carried[0])
            off = t + ii * n_vec
            qx = _ldv(g_dqp, off, fx.Float32, RED_VEC)
            qn = [qacc[c] + qx[c] for c in range_constexpr(RED_VEC)]
            final = yield [fx.as_ir_value(fx.Vector.from_elements(qn, dtype=fx.Float32))]
        return final

    init = [fx.as_ir_value(fx.Vector.filled(RED_VEC, 0.0, fx.Float32))]
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
