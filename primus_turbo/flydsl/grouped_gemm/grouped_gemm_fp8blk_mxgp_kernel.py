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

"""FlyDSL gfx1250 FP8 block-scaled (UE8M0) M-grouped NT GEMM, persistent, with a cross-k-tile fragment pipeline
("mxgp"). Same math, operands, WMMA (v_wmma_scale_f32_16x16x128_f8f6f4, MX block 32 with the 1x128 / 128x128
exponents repeated 4x), per-element k order (ascending 128-k steps into one fp32 accumulator) and bf16 rounding as
flygg (grouped_gemm_fp8blk_nt_kernel.kernel_gg): C is bit-identical.

Tile 256 x 256 x 256 on 4 waves (2 x 2), a 2-stage ring of 256-deep k-tiles and whole-K scale planes loaded once per
tile (A: one row per row, weights: one row per 16-row WMMA fragment, as the 128x128-block scales are constant
there). The fragments of the next k-step (B double-buffered, A one row group ahead) are read from LDS during the row
groups of the current one, each group pinned with sched_barrier, so no k-step starts with a fragment-load bubble.
The grid is the number of CUs; each workgroup walks the live tiles (bounded on the device by the group offsets).
Requires: aligned group offsets (multiples of 256: the caller's pool alignment), K % 512 == 0 with K a power of two
(the scale-plane TDM pad interval), N % 256 == 0, weight scales constant over 16-row groups of N.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Constexpr
from flydsl.expr.typing import T as _T

from primus_turbo.flydsl.utils import buffer_ops as _bops

from .grouped_gemm_fp8blk_utils import _A_ID, WAVE, _make_lds_load, _ptr


def _store_c_lines(
    base_ptr,
    scr_off,
    lane,
    c_frags,
    m_reps,
    n_reps,
    warp_n,
    gC8,
    row0,
    col0,
    ldc64,
    out_cls,
    lds_b128,
    bands=1,
):
    """Store a wave's (m_reps*16) x warp_n accumulator block as bf16 with whole-line global stores.
    The WMMA C layout gives each lane 8 consecutive columns of one row, so a direct global_store_b128 of a fragment
    touches 16 rows x 32 B (16 cache lines per instruction). Here each
    16-row band goes through a wave-private LDS scratch (16 x (2*warp_n + 16) B at scr_off) and is read back
    row-contiguous, so one global_store_b128 covers 32/LPR rows x 2*warp_n B (whole lines). LDS ops of one wave
    execute in order, so no barrier is needed. row0/col0: first output row/column of the wave block (elements).
    bands: 16-row bands staged per LDS round trip (scratch = bands x 16 x (2*warp_n + 16) B per wave)."""
    RB = 2 * warp_n + 16
    LPR = (2 * warp_n) // 16  # lanes per output row
    assert 32 % LPR == 0
    RPI = 32 // LPR  # rows per store instruction
    lrow = lane // LPR
    lcol = lane % LPR
    lane16 = lane % 16
    kgrp = lane // 16
    assert m_reps % bands == 0
    for mb in range_constexpr(m_reps // bands):
        for bi in range_constexpr(bands):
            mi = mb * bands + bi
            for ni in range_constexpr(n_reps):
                h = c_frags[mi * n_reps + ni].load().to(out_cls)
                fx.ptr_store(
                    h.bitcast(fx.Int8),
                    base_ptr + (scr_off + (bi * 16 + lane16) * RB + (ni * 16 + kgrp * 8) * 2),
                )
        vs = []
        for bi in range_constexpr(bands):
            for j in range_constexpr(16 // RPI):
                r = bi * 16 + j * RPI + lrow
                vs.append(fx.Vector(lds_b128(fx.Int32(fx.ptrtoint(base_ptr)), scr_off + r * RB + lcol * 16)))
        for bi in range_constexpr(bands):
            for j in range_constexpr(16 // RPI):
                r = j * RPI + lrow
                off = (fx.Int64(row0 + (mb * bands + bi) * 16 + r) * ldc64 + fx.Int64(col0 + lcol * 8)) * 2
                fx.ptr_store(vs[bi * (16 // RPI) + j].bitcast(fx.Int8), gC8 + off)


@flyc.jit
def launch_mxgp_fp8blk_nt_gfx1250(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_offs: fx.Pointer,
    arg_sa: fx.Pointer,
    arg_sb: fx.Pointer,
    i32_nwg: fx.Int32,
    stream: fx.Stream,
    i32_n: fx.Int32,
    i32_k: fx.Int32,
    n_groups: Constexpr[int],
    a_id: Constexpr[int],
    b_id: Constexpr[int],
    k_max: Constexpr[int] = 4096,
    early: Constexpr[int] = 4,
):
    """Persistent: grid = #CUs; workgroup bid computes live tiles t = bid, bid + nwg, ... (live = offs[G] / 256 m-tiles
    x N / 256, read on the device: no dead workgroups, usable for EP>1 worst-case pools). Across tiles: k-tile K-2's
    early barrier issues the NEXT tile's k-tile 0; the last k-tile's early barrier issues the next tile's scale planes
    (the current tile's scale words are all in registers by then) and its row groups prefetch the next tile's
    fragments from stage 0; the epilogue writes C from the accumulators through a per-wave line scratch in stage 1
    (free: its last reads were before that early barrier) with whole-line VMEM stores, not waited on; then the next
    tile's k-tile 1 goes to stage 1 and its scale words are read once the planes landed. No C staging in LDS."""
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    tile_m = tile_n = 256
    tile_k = 256
    m_warp = n_warp = 2
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    out_cls = fx.BFloat16
    LDS_PAD = 16
    A_ROW = tile_k + LDS_PAD
    B_ROW = tile_k + LDS_PAD
    STAGE_A = tile_m * A_ROW
    PITCH = STAGE_A + tile_n * B_ROW
    MX_BLOCK = 32
    SK = k_max // MX_BLOCK
    SP = SK + 4
    PLANE_SA = tile_m * SP
    B_PROWS = tile_n // WMMA_N
    PLANE_SB = (B_PROWS * SP + 15) // 16 * 16
    SC_OFF = 2 * PITCH
    OFF_LDS = SC_OFF + PLANE_SA + PLANE_SB  # 32 x i32: slot l = offs[min(l + 1, G)]
    PRM_LDS = OFF_LDS + 4 * WAVE  # 8 x i32 tile parameters
    ARENA = PRM_LDS + 32
    assert ARENA <= 320 * 1024, f"LDS arena {ARENA} B exceeds 320 KiB"
    G = n_groups
    m_reps = n_reps = 8
    warp_n = 128
    SCR_W = 16 * (2 * warp_n + 16)  # per-wave C line scratch, inside stage 1
    assert num_waves * SCR_W <= PITCH
    assert 2 <= early <= 6 and n_reps % (m_reps - early) == 0
    BPG = n_reps // (m_reps - early)
    NA = 2
    kernel_name = f"moe_mxgp_fp8blk_nt_gfx1250_t256x256x256_a{a_id}{b_id}_g{G}_k{k_max}_e{early}"

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_mxgp(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_offs: fx.Pointer,
        arg_sa: fx.Pointer,
        arg_sb: fx.Pointer,
        i32_nwg: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, _, _ = fx.block_idx
        bid = fx.Int32(bid_x)
        n_tiles_n = i32_n // tile_n
        i32_ptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4
        )
        offs_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(fx.recast_iter(i32_ptr, arg_offs)))
        zero = bid * fx.Int32(0)
        one = zero + fx.Int32(1)
        assert G <= WAVE
        # group offsets: lane i holds offs[i+1] in one VGPR (loaded once); per tile the expert index is the count of
        # lanes with offs[i+1] <= m_row (xor-shuffle sum): no SGPRs held across the loop
        lane0 = tid % WAVE  # noqa: F841
        live_rows = fx.Int32(_bops.buffer_load(offs_rsrc, zero + fx.Int32(G), vec_width=1, is_scalar=True))
        total = (live_rows // tile_m) * n_tiles_n
        my_cnt = (total - bid + i32_nwg - 1) // i32_nwg
        my_cnt = (my_cnt > zero).select(my_cnt, zero)
        K_TILES = i32_k // tile_k
        lda64 = fx.Int64(i32_k)
        ldc64 = fx.Int64(i32_n)
        sk = i32_k // MX_BLOCK
        sk64 = fx.Int64(sk)

        def tile_params(t):
            # scalar loads + SALU scan at the tile start only (results go to the LDS parameter block)
            m_row = (t // n_tiles_n) * tile_m
            g = zero
            for i_ in range_constexpr(G):
                e = fx.Int32(
                    _bops.buffer_load(offs_rsrc, zero + fx.Int32(i_ + 1), vec_width=1, is_scalar=True)
                )
                g = g + (e <= m_row).select(one, zero)
            g = (g < fx.Int32(G)).select(g, fx.Int32(G - 1))
            return (
                rocdl.readfirstlane(_T.i32, g),
                rocdl.readfirstlane(_T.i32, m_row),
                rocdl.readfirstlane(_T.i32, (t % n_tiles_n) * tile_n),
            )

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA)
        base_ptr = arena.base_ptr

        def _stage(s):
            return fx.add_offset(base_ptr, s * PITCH)

        def _view(ptr, cls, shape, stride, align=16):
            return fx.Tensor(
                fx.make_view(
                    fx.recast_iter(fx.PointerType.get(cls.ir_type, ptr.address_space, align), ptr),
                    fx.make_layout(shape, stride),
                )
            )

        def _gview(base, off, shape, stride):
            return fx.Tensor(fx.make_view(fx.add_offset(base, off), fx.make_layout(shape, stride)))

        def _cast(cls, p, align=16):
            return fx.recast_iter(fx.PointerType.get(cls.ir_type, p.address_space, align), p)

        gA = _gview(_cast(a_cls, arg_a), fx.Int64(0), (tile_m, tile_k), (i32_k, 1))
        gB = _gview(_cast(b_cls, arg_b), fx.Int64(0), (tile_n, tile_k), (i32_k, 1))
        atomA = fx.rocdl.make_tdm_atom(
            gA,
            [None, None],
            strides=[lda64, None],
            num_warps=num_waves,
            pad_interval=tile_k,
            pad_amount=LDS_PAD,
        )
        atomB = fx.rocdl.make_tdm_atom(
            gB,
            [None, None],
            strides=[lda64, None],
            num_warps=num_waves,
            pad_interval=tile_k,
            pad_amount=LDS_PAD,
        )
        gSA_base = _cast(fx.Uint8, arg_sa, 1)
        gSB_base = _cast(fx.Uint8, arg_sb, 1)

        def issue(s, tp, kt):
            g, m_row, blk_n = tp
            k = fx.Int64(kt) * tile_k
            fx.copy(
                atomA,
                gA,
                _view(_stage(s), a_cls, (tile_m, tile_k), (A_ROW, 1)),
                imm_offset=fx.Int64(m_row) * lda64 + k,
            )
            fx.copy(
                atomB,
                gB,
                _view(fx.add_offset(_stage(s), STAGE_A), b_cls, (tile_n, tile_k), (B_ROW, 1)),
                imm_offset=(fx.Int64(g) * fx.Int64(i32_n) + fx.Int64(blk_n)) * lda64 + k,
            )

        def issue_planes(tp):
            g, m_row, blk_n = tp
            gsa = _gview(gSA_base, fx.Int64(m_row) * sk64, (tile_m, SK), (sk, 1))
            gsb = _gview(
                gSB_base,
                (fx.Int64(g) * fx.Int64(i32_n) + fx.Int64(blk_n)) * sk64,
                (B_PROWS, SK),
                (sk * WMMA_N, 1),
            )
            asa = fx.rocdl.make_tdm_atom(
                gsa, [None, sk], strides=[sk64, None], num_warps=num_waves, pad_interval=SK, pad_amount=4
            )
            asb = fx.rocdl.make_tdm_atom(
                gsb,
                [None, sk],
                strides=[sk64 * WMMA_N, None],
                num_warps=num_waves,
                pad_interval=SK,
                pad_amount=4,
            )
            fx.copy(asa, gsa, _view(fx.add_offset(base_ptr, SC_OFF), fx.Uint8, (tile_m, SK), (SP, 1), 1))
            fx.copy(
                asb,
                gsb,
                _view(fx.add_offset(base_ptr, SC_OFF + PLANE_SA), fx.Uint8, (B_PROWS, SK), (SP, 1), 1),
            )

        mma_atom = fx.make_mma_atom(
            fx.rocdl.WMMAScale(WMMA_M, WMMA_N, WMMA_K, b_cls, a_cls, fx.Float32, block_size=MX_BLOCK)
        )
        lds_b128 = _make_lds_load(128)
        lds_b32 = _make_lds_load(32)
        wave = rocdl.readfirstlane(_T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wmb = (wave // n_warp) * 128
        wnb = (wave % n_warp) * 128
        c_frags = [fx.make_rmem_tensor(WMMA_N // 2, fx.Float32) for _ in range_constexpr(m_reps * n_reps)]
        lds_base = fx.Int32(fx.ptrtoint(base_ptr))
        lane_a = (wmb + lane16) * A_ROW + kgrp * 16
        lane_b = STAGE_A + (wnb + lane16) * B_ROW + kgrp * 16
        lane_sa = SC_OFF + (wmb + lane16) * SP
        zl = lane * (i32_n >> 30)
        zs = i32_n >> 30
        sb_row0 = SC_OFF + PLANE_SA + (wnb // WMMA_N) * SP

        def addrs(kt):
            st = fx.Int32(fx.ptrtoint(_stage((kt + zs) % 2)))
            return dict(
                a=st + lane_a,
                b=st + lane_b,
                sa=lds_base + lane_sa + kt * 8,
                sb=zl + (lds_base + sb_row0 + kt * 8),
            )

        def _ld_frag(base, off, dst=None):
            v = [fx.Vector(lds_b128(base, off + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            reg = fx.make_rmem_tensor(16, fx.Int32) if dst is None else dst
            reg.store(v01.shuffle(v23, list(range(16))))
            return reg

        def frag_a(ad, ks, mi, dst=None):
            return _ld_frag(ad["a"], mi * WMMA_M * A_ROW + ks * WMMA_K, dst)

        def frag_b(ad, ks, ni, dst=None):
            return _ld_frag(ad["b"], ni * WMMA_N * B_ROW + ks * WMMA_K, dst)

        def scale_a(ad, ks, mi):
            return lds_b32(ad["sa"], mi * WMMA_M * SP + ks * 4)

        def scale_b(ad, ks, ni):
            return lds_b32(ad["sb"], ni * SP + ks * 4)

        def _set():
            return dict(
                b=[fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(n_reps)],
                a2=[fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(NA)],
                sa2=[fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(NA)],
                sb=[fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(n_reps)],
            )

        SETS = [_set(), _set()]

        def load_frags(kt, dst):
            ad = addrs(kt)
            for ni in range_constexpr(n_reps):
                frag_b(ad, 0, ni, dst["b"][ni])
            for j in range_constexpr(NA):
                frag_a(ad, 0, j, dst["a2"][j])

        def load_scales(kt, dst):
            ad = addrs(kt)
            for ni in range_constexpr(n_reps):
                dst["sb"][ni].store(scale_b(ad, 0, ni))
            for j in range_constexpr(NA):
                dst["sa2"][j].store(scale_a(ad, 0, j))

        def _wmmas(bfr, afr, sav, sbv, mi, lo, hi):
            for ni in range_constexpr(lo, hi):
                idx = mi * n_reps + ni
                fx.gemm(mma_atom, c_frags[idx], bfr[ni], afr, c_frags[idx], scale_a=sbv[ni], scale_b=sav)

        def bar():
            rocdl.sched_barrier(0)
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)
            rocdl.sched_barrier(0)

        REST = list(range(2, m_reps))
        CH = -(-len(REST) // (early - 1))

        def step(kt, src, dst):
            """k-tile kt of the current tile (tp); dst = carrier set for the next k-tile (this tile's kt+1, or the
            next tile's k-tile 0 when kt is the last one: fragments only, its scale words are read after its planes)."""
            rocdl.sched_barrier(0)
            last = kt + 1 == K_TILES
            ad = addrs(kt)
            an = addrs(kt + 1)  # stage (kt+1) % 2; for the last k-tile = stage 0 = next tile's k-tile 0
            h = n_reps // 2
            sb0 = [src["sb"][ni].load()[0] for ni in range_constexpr(n_reps)]
            a0 = [src["a2"][0], src["a2"][1]] + [None] * (m_reps - 2)
            s0 = [src["sa2"][0].load()[0], src["sa2"][1].load()[0]] + [None] * (m_reps - 2)
            a1 = [None] * m_reps
            s1 = [None] * m_reps
            b1 = [None] * n_reps
            sb1 = [None] * n_reps
            for mi in range_constexpr(m_reps):
                rocdl.sched_barrier(0)
                _wmmas(src["b"], a0[mi], s0[mi], sb0, mi, 0, h)
                rocdl.sched_barrier(0)
                if const_expr(mi + 2 < m_reps):
                    a0[mi + 2] = frag_a(ad, 0, mi + 2)
                    s0[mi + 2] = scale_a(ad, 0, mi + 2)[0]
                else:
                    j = mi + 2 - m_reps
                    a1[j] = frag_a(ad, 1, j)
                    s1[j] = scale_a(ad, 1, j)[0]
                b1[mi] = frag_b(ad, 1, mi)
                if const_expr(mi == 0):
                    for ni in range_constexpr(n_reps):
                        sb1[ni] = scale_b(ad, 1, ni)[0]
                rocdl.sched_barrier(0)
                _wmmas(src["b"], a0[mi], s0[mi], sb0, mi, h, n_reps)
            for mi in range_constexpr(m_reps):
                rocdl.sched_barrier(0)
                if const_expr(mi == early):
                    rocdl.s_wait_dscnt(0)
                    tdm_ops.tensor_wait(0)
                    bar()
                    # stage kt % 2 and (at the last k-tile) the scale planes are no longer read by any wave
                    if kt + 2 < K_TILES:
                        issue(kt % 2, tp_cur(), kt + 2)
                    else:
                        if prm(6) > zero:
                            if last:
                                issue_planes(tp_nxt())
                            else:
                                issue(kt % 2, tp_nxt(), 0)
                _wmmas(b1, a1[mi], s1[mi], sb1, mi, 0, h)
                rocdl.sched_barrier(0)
                if const_expr(mi < early - 1):
                    for j in REST[mi * CH : (mi + 1) * CH]:
                        a1[j] = frag_a(ad, 1, j)
                        s1[j] = scale_a(ad, 1, j)[0]
                if const_expr(mi >= early):
                    for q in range_constexpr(BPG):
                        frag_b(an, 0, (mi - early) * BPG + q, dst["b"][(mi - early) * BPG + q])
                    # scale words of step kt+1: at the last k-tile these read the planes' pad column (in bounds,
                    # garbage) and are replaced at the next tile start (an unconditional load avoids phi merges
                    # that keep two carrier sets of scale words live in the low VGPRs)
                    if const_expr(mi == early):
                        for ni in range_constexpr(n_reps):
                            dst["sb"][ni].store(scale_b(an, 0, ni))
                    if const_expr(mi >= m_reps - NA):
                        j = mi - (m_reps - NA)
                        frag_a(an, 0, j, dst["a2"][j])
                        dst["sa2"][j].store(scale_a(an, 0, j))
                rocdl.sched_barrier(0)
                _wmmas(b1, a1[mi], s1[mi], sb1, mi, h, n_reps)

        # tile parameters [g, m_row, blk_n, next g, next m_row, next blk_n, has_next] in LDS (no VGPR / SGPR held
        # across the k loop: the scale operands of V_WMMA_SCALE need the low VGPRs); rewritten only after the tile's
        # last barrier, read by each wave after its own write (LDS ops of a wave are in order)
        def prm(i):
            return rocdl.readfirstlane(_T.i32, lds_b32(lds_base, PRM_LDS + 4 * i)[0])

        def tp_cur():
            return (prm(0), prm(1), prm(2))

        def tp_nxt():
            return (prm(3), prm(4), prm(5))

        def set_prm(tp, ntp, hn):
            v = fx.Vector.from_elements([tp[0], tp[1], tp[2], ntp[0], ntp[1], ntp[2], hn, zero], fx.Int32)
            fx.ptr_store(v.bitcast(fx.Int8), base_ptr + PRM_LDS)

        gC8 = _cast(fx.Int8, _cast(out_cls, arg_c))
        # first tile: planes + k-tile 0, wait, its fragments + scale words, k-tile 1 in flight
        t0 = (my_cnt > zero).select(bid, zero)
        tp0 = tile_params(t0)
        if my_cnt > zero:
            issue_planes(tp0)
            issue(0, tp0, 0)
            tdm_ops.tensor_wait(0)
            fx.barrier()
            load_frags(0, SETS[0])
            load_scales(0, SETS[0])
            issue(1, tp0, 1)
        for it in range(my_cnt):
            t = bid + it * i32_nwg
            hn = (it + 1 < my_cnt).select(one, zero)
            set_prm(tile_params(t), tile_params((hn > zero).select(t + i32_nwg, t)), hn)
            for cf in c_frags:
                cf.fill(0.0)
            for kp in range(K_TILES // 2):
                step(kp * 2, SETS[0], SETS[1])
                step(kp * 2 + 1, SETS[1], SETS[0])
            # epilogue: C via the wave's line scratch in stage 1 (its last reads preceded the last early barrier),
            # whole-line VMEM stores, not waited on
            tp = tp_cur()
            _store_c_lines(
                base_ptr,
                PITCH + wave * SCR_W,
                lane,
                c_frags,
                m_reps,
                n_reps,
                warp_n,
                gC8,
                tp[1] + wmb,
                tp[2] + wnb,
                ldc64,
                out_cls,
                lds_b128,
                bands=1,
            )
            if prm(6) > zero:
                bar()  # every wave is done with its stage-1 scratch
                issue(1, tp_nxt(), 1)
                tdm_ops.tensor_wait(2)  # the next tile's planes (issued before k-tile 1)
                bar()
                load_scales(0, SETS[0])

    kernel_mxgp(arg_c, arg_a, arg_b, arg_offs, arg_sa, arg_sb, i32_nwg, i32_n, i32_k).launch(
        grid=(i32_nwg, 1, 1), block=(block, 1, 1), stream=stream
    )


launch_mxgp_fp8blk_nt_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}
_COMPILED_P: dict = {}


def mxgp_fp8blk_nt(a, b, sa, sb, offs_i32, out=None, early=4, nwg=None):
    """out [M, N] bf16, bit-identical to flygg_fp8blk_nt(..., tile=(256, 256, 256), num_buffers=2): a [M, K] fp8,
    b [G, N, K] fp8, sa [M, K/32] / sb [G, N, K/32] uint8 E8M0. Needs K % 512 == 0, power-of-two K, N % 256 == 0,
    group offsets multiples of 256 (live tiles are bounded on the device by offs[G])."""
    assert sa.dtype == torch.uint8 and sb.dtype == torch.uint8 and sa.is_contiguous() and sb.is_contiguous()
    M, K = a.shape
    G, N, Kb = b.shape
    assert K == Kb and a.is_contiguous() and b.is_contiguous()
    assert K % 512 == 0 and (K & (K - 1)) == 0 and N % 256 == 0 and M % 256 == 0
    assert sa.shape == (M, K // 32) and sb.shape == (G, N, K // 32)
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    for t in (a, b, sa, sb, offs_i32, out):
        assert t.data_ptr() != 0
    if nwg is None:
        nwg = torch.cuda.get_device_properties(a.device).multi_processor_count
    nwg = max(1, min(nwg, (M // 256) * (N // 256)))
    a_id = 0 if a.dtype == torch.float8_e4m3fn else 1
    b_id = 0 if b.dtype == torch.float8_e4m3fn else 1
    args = (
        _ptr(out),
        _ptr(a),
        _ptr(b),
        _ptr(offs_i32),
        _ptr(sa),
        _ptr(sb),
        nwg,
        fx.Stream(torch.cuda.current_stream(device=a.device)),
        N,
        K,
        G,
        a_id,
        b_id,
        K,
        early,
    )
    key = (G, a_id, b_id, K, early, nwg)
    c = _COMPILED_P.get(key)
    if c is None:
        _COMPILED_P[key] = flyc.compile(launch_mxgp_fp8blk_nt_gfx1250, *args)
    else:
        c(*args)
    return out
