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

"""FlyDSL gfx1250 FP8 block-scaled (UE8M0, 128x1 along K) variable-K wgrad, persistent.

out[g] (+)= lhsP[:, seg_g] @ rhsP[:, seg_g]^T with E8M0 exponents per (row, 128 pool rows):
  lhsP [OUT_M, P], rhsP [OUT_N, P] fp8 (pool rows contiguous), lhs_s [OUT_M, P/128], rhs_s [OUT_N, P/128] uint8
  (one byte per 128 tokens; the kernel repeats it 4x for the MX WMMA, block 32), seg_g = [poffs[g], poffs[g+1]).

Each workgroup walks tiles bid, bid + grid, ...; all their k-tiles are one stream through a 2-stage LDS ring (the
next tile's first k-tile and its scale planes are issued during the current tile's last k-tile); the C tile is
staged in its own LDS region and its TDM store stays in flight while the next tile runs. Scale planes are
double-buffered by tile parity. This is the fallback of the vkq kernel (same bits). Wait accounting (2 TDM ops per stage, LA = 1 stage ahead): the store is the only op newer than stage 0 of the
next tile -> wait 1 at k-step 0, 0 after; before sC is rewritten the next tile's planes + stage (4 ops) are newer than
the previous store -> wait 4.
Requires OUT_M % tile_m == 0, OUT_N % tile_n == 0, every window a multiple of tile_k, >= 1 k-tile, <= kwin_max.
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


@flyc.jit
def launch_vkpblk_fp8_gfx1250(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_offs: fx.Pointer,
    arg_perm: fx.Pointer,
    arg_sa: fx.Pointer,
    arg_sb: fx.Pointer,
    arg_sq: fx.Pointer,
    i32_m: fx.Int32,
    stream: fx.Stream,
    i32_n: fx.Int32,
    i32_g: fx.Int32,
    i32_lda: fx.Int32,
    i32_ldb: fx.Int32,
    i32_nwg: fx.Int32,
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    a_id: Constexpr[int],
    b_id: Constexpr[int],
    beta_one: Constexpr[int],
    group_m: Constexpr[int] = 4,
    kwin_max: Constexpr[int] = 4096,
    lpt: Constexpr[int] = 0,
    sumsq: Constexpr[int] = 0,
):
    """Tile parameters (perm / group offsets) are read by scalar buffer loads; the E8M0 scale words of k-step kt+1 are
    read from the LDS planes at the end of k-step kt (one batch, live across the next barrier) and the byte -> 4x
    repeat is one v_perm_b32; the M-side operand fragments are read ``sched2`` row groups ahead of their WMMAs,
    pinned with sched_barrier. beta 1 loads each tile's old C behind its k-tile 2 (a prefetch) instead of in the
    epilogue.
    sumsq: every thread also sums the squares of the bf16 values it stores (the final C, after
    the beta = 1 add) over all of its tiles, and writes that fp32 partial to arg_sq[bid * block + tid] (fixed order,
    no atomics)."""
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    num_buffers = 2
    sched2 = 1  # row groups of A fragments read ahead of their WMMAs
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    MX_BLOCK = 32
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    assert tile_m % (m_warp * WMMA_M) == 0 and tile_n % (n_warp * WMMA_N) == 0
    assert tile_k == WMMA_K, "one 128-token scale per k-tile"
    assert tile_m % num_waves == 0 and tile_n % num_waves == 0
    out_cls = fx.BFloat16
    LDS_PAD = 16
    A_ROW = tile_k + LDS_PAD
    B_ROW = tile_k + LDS_PAD
    STAGE_A = tile_m * A_ROW
    STAGE_B = tile_n * B_ROW
    PITCH = STAGE_A + STAGE_B
    C_ROW = tile_n + 8
    C_OFF = num_buffers * PITCH
    SPW = kwin_max // 128  # scale bytes per row of the longest window
    SPB = SPW + 4  # padded LDS row (multiple of 4)
    PL_A = tile_m * SPB
    PL_B = tile_n * SPB
    PSET = PL_A + PL_B
    PL_OFF = C_OFF + tile_m * C_ROW * 2
    ARENA = PL_OFF + 2 * PSET
    assert ARENA <= 320 * 1024, f"LDS arena {ARENA} B exceeds 320 KiB"

    kernel_name = (
        f"moe_vkpblk_fp8_gfx1250_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}_a{a_id}{b_id}"
        f"_b{beta_one}_w{kwin_max}"
        + ("_lpt" if lpt else "")
        + ("_pfc" if beta_one else "")
        + ("_sq" if sumsq else "")
        + f"_s{sched2}"
    )
    PFC = bool(beta_one)

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_vkpblk(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_offs: fx.Pointer,
        arg_perm: fx.Pointer,
        arg_sa: fx.Pointer,
        arg_sb: fx.Pointer,
        arg_sq: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_g: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldb: fx.Int32,
        i32_nwg: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, _, _ = fx.block_idx
        bid = fx.Int32(bid_x)
        n_tiles_m = i32_m // tile_m
        n_tiles_n = i32_n // tile_n
        tpg = n_tiles_m * n_tiles_n
        total = tpg * i32_g
        my_cnt = (total - bid + i32_nwg - 1) // i32_nwg
        last_t = bid + (my_cnt - 1) * i32_nwg

        i32_ptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4
        )
        lda64 = fx.Int64(i32_lda)
        ldb64 = fx.Int64(i32_ldb)
        ska = i32_lda // 128
        skb = i32_ldb // 128

        offs_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(fx.recast_iter(i32_ptr, arg_offs)))
        perm_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(fx.recast_iter(i32_ptr, arg_perm)))

        def _sld(rsrc, i):
            return fx.Int32(_bops.buffer_load(rsrc, i, vec_width=1, is_scalar=True))

        def tile_params(t):
            # lpt: tile slot t // tpg -> expert perm[slot] (experts sorted by window length, longest first), so the
            # round-robin t = bid + it * nwg hands out the longest experts' tiles first and the short ones fill the tail
            g = t // tpg
            if const_expr(lpt):
                g = _sld(perm_rsrc, g)
            pid = t % tpg
            per_group = group_m * n_tiles_n
            first_m = (pid // per_group) * group_m
            rem = n_tiles_m - first_m
            gsize = (rem < group_m).select(rem, fx.Int32(group_m))
            in_g = pid % per_group
            pm = first_m + in_g % gsize
            pn = in_g // gsize
            kb = _sld(offs_rsrc, g)
            ke = _sld(offs_rsrc, g + fx.Int32(1))
            return g, pm * tile_m, pn * tile_n, kb, (ke - kb) // tile_k

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

        gA = _gview(_cast(a_cls, arg_a), fx.Int64(0), (tile_m, tile_k), (i32_lda, 1))
        gB = _gview(_cast(b_cls, arg_b), fx.Int64(0), (tile_n, tile_k), (i32_ldb, 1))
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
            strides=[ldb64, None],
            num_warps=num_waves,
            pad_interval=tile_k,
            pad_amount=LDS_PAD,
        )
        gSA_base = _cast(fx.Uint8, arg_sa, 1)
        gSB_base = _cast(fx.Uint8, arg_sb, 1)

        def sA_of(s):
            return _view(_stage(s), a_cls, (tile_m, tile_k), (A_ROW, 1))

        def sB_of(s):
            return _view(fx.add_offset(_stage(s), STAGE_A), b_cls, (tile_n, tile_k), (B_ROW, 1))

        def issue(s, offa, offb):
            fx.copy(atomA, gA, sA_of(s), imm_offset=offa)
            fx.copy(atomB, gB, sB_of(s), imm_offset=offb)

        def issue_planes(pset, blk_m, blk_n, kb, kt_n):
            """the tile's window of both scale planes -> LDS set pset (2 TDM ops)"""
            gsa = _gview(
                gSA_base, fx.Int64(blk_m) * fx.Int64(ska) + fx.Int64(kb // 128), (tile_m, SPW), (ska, 1)
            )
            gsb = _gview(
                gSB_base, fx.Int64(blk_n) * fx.Int64(skb) + fx.Int64(kb // 128), (tile_n, SPW), (skb, 1)
            )
            asa = fx.rocdl.make_tdm_atom(
                gsa,
                [None, kt_n],
                strides=[fx.Int64(ska), None],
                num_warps=num_waves,
                pad_interval=SPW,
                pad_amount=4,
            )
            asb = fx.rocdl.make_tdm_atom(
                gsb,
                [None, kt_n],
                strides=[fx.Int64(skb), None],
                num_warps=num_waves,
                pad_interval=SPW,
                pad_amount=4,
            )
            poff = PL_OFF + pset * PSET
            fx.copy(asa, gsa, _view(fx.add_offset(base_ptr, poff), fx.Uint8, (tile_m, SPW), (SPB, 1), 1))
            fx.copy(
                asb, gsb, _view(fx.add_offset(base_ptr, poff + PL_A), fx.Uint8, (tile_n, SPW), (SPB, 1), 1)
            )

        mma_atom = fx.make_mma_atom(
            fx.rocdl.WMMAScale(WMMA_M, WMMA_N, WMMA_K, b_cls, a_cls, fx.Float32, block_size=MX_BLOCK)
        )
        warp_m = tile_m // m_warp
        warp_n = tile_n // n_warp
        m_reps = warp_m // WMMA_M
        n_reps = warp_n // WMMA_N
        lds_b128 = _make_lds_load(128)
        lds_b32 = _make_lds_load(32)
        wave = rocdl.readfirstlane(_T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wmb = (wave // n_warp) * warp_m
        wnb = (wave % n_warp) * warp_n
        n_acc = m_reps * n_reps
        c_frags = [fx.make_rmem_tensor(WMMA_N // 2, fx.Float32) for _ in range_constexpr(n_acc)]
        gk_reg = fx.make_rmem_tensor(1, fx.Int32)
        gk_reg.fill(0)
        if const_expr(sumsq):
            sq_reg = fx.make_rmem_tensor(WMMA_N // 2, fx.Float32)
            sq_reg.fill(0.0)
        lds_base = fx.Int32(fx.ptrtoint(base_ptr))

        def _operand(base, plane_off, row, ks):
            b0 = plane_off + row * A_ROW + ks * WMMA_K + kgrp * 16
            v = [fx.Vector(lds_b128(base, b0 + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            reg = fx.make_rmem_tensor(16, fx.Int32)
            reg.store(v01.shuffle(v23, list(range(16))))
            return reg

        NSW = n_reps + m_reps
        sw_regs = [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(NSW)]

        def load_words(kt, pset):
            """the raw scale words of k-step kt (B rows of the wave, then A rows) -> sw_regs"""
            poff = PL_OFF + pset * PSET
            koff = (kt // 4) * 4
            for ni in range_constexpr(n_reps):
                sw_regs[ni].store(lds_b32(lds_base, poff + PL_A + (wnb + ni * WMMA_N + lane16) * SPB + koff))
            for mi in range_constexpr(m_reps):
                sw_regs[n_reps + mi].store(
                    lds_b32(lds_base, poff + (wmb + mi * WMMA_M + lane16) * SPB + koff)
                )

        def compute2(s, kt, pset, kt_n):
            base = fx.Int32(fx.ptrtoint(_stage(s)))
            sel = (kt % 4) * 0x01010101  # v_perm_b32: every result byte = byte kt%4 of the word
            sw = [r.load()[0] for r in sw_regs]
            sv = [fx.Int32(rocdl.perm_b32(w, w, sel)) for w in sw]
            sbv, sav_all = sv[:n_reps], sv[n_reps:]
            bf = [_operand(base, STAGE_A, wnb + ni * WMMA_N + lane16, 0) for ni in range_constexpr(n_reps)]
            afs = [None] * m_reps
            for mi in range_constexpr(min(sched2, m_reps)):
                afs[mi] = _operand(base, 0, wmb + mi * WMMA_M + lane16, 0)
            for mi in range_constexpr(m_reps):
                if const_expr(mi + sched2 < m_reps):
                    afs[mi + sched2] = _operand(base, 0, wmb + (mi + sched2) * WMMA_M + lane16, 0)
                rocdl.sched_barrier(0)
                for ni in range_constexpr(n_reps):
                    idx = mi * n_reps + ni
                    fx.gemm(
                        mma_atom,
                        c_frags[idx],
                        bf[ni],
                        afs[mi],
                        c_frags[idx],
                        scale_a=sbv[ni],
                        scale_b=sav_all[mi],
                    )
                rocdl.sched_barrier(0)

        def offs_of(blk_m, blk_n, kb, kt):
            k = fx.Int64(kb) + fx.Int64(kt) * tile_k
            return fx.Int64(blk_m) * lda64 + k, fx.Int64(blk_n) * ldb64 + k

        c_ld = fx.Int64(i32_n)
        gC_base = _cast(out_cls, arg_c)
        sC = _view(fx.add_offset(base_ptr, C_OFF), out_cls, (tile_m, C_ROW), (C_ROW, 1))

        def c_view(g, blk_m, blk_n):
            gtC = _gview(
                gC_base,
                fx.Int64(g) * fx.Int64(i32_m) * c_ld + fx.Int64(blk_m) * c_ld + fx.Int64(blk_n),
                (tile_m, C_ROW),
                (C_ROW, 1),
            )
            return gtC, fx.rocdl.make_tdm_atom(gtC, [None, tile_n], strides=[c_ld, None], num_warps=num_waves)

        # prologue: tile 0's planes (set 0) and first k-tile, then one TDM op standing in for the C store that
        # later tiles keep in flight (keeps the wait counts uniform)
        g0, bm0, bn0, kb0, ktn0 = tile_params(bid)
        issue_planes(0, bm0, bn0, kb0, ktn0)
        oa, ob = offs_of(bm0, bn0, kb0, 0)
        issue(0, oa, ob)
        gt0, at0 = c_view(g0, bm0, bn0)
        fx.copy(at0, gt0, sC)

        for it in range(my_cnt):
            t = bid + it * i32_nwg
            g, blk_m, blk_n, kb, kt_n = tile_params(t)
            t_nx = (t + i32_nwg < total).select(t + i32_nwg, last_t)
            _, nbm, nbn, nkb, nktn = tile_params(t_nx)
            pset = it % 2
            for cf in c_frags:
                cf.fill(0.0)
            gk = gk_reg.load()[0]
            gtC, atomC = c_view(g, blk_m, blk_n)
            for kt in range(kt_n):
                if kt == 0:
                    tdm_ops.tensor_wait(1)  # the previous tile's C store may still be in flight
                else:
                    if const_expr(PFC):
                        if kt == 2:
                            tdm_ops.tensor_wait(
                                1
                            )  # the old-C prefetch (newest op, issued at kt 1) may stay in flight
                        else:
                            tdm_ops.tensor_wait(0)
                    else:
                        tdm_ops.tensor_wait(0)
                fx.barrier()
                if kt == 0:
                    load_words(0, pset)
                f = kt + 1
                if f < kt_n:
                    oa0, ob0 = offs_of(blk_m, blk_n, kb, f)
                    issue((gk + f) % num_buffers, oa0, ob0)
                else:
                    # next tile: its scale planes (other set), then its first k-tile
                    issue_planes(1 - pset, nbm, nbn, nkb, nktn)
                    oa1, ob1 = offs_of(nbm, nbn, nkb, 0)
                    issue((gk + f) % num_buffers, oa1, ob1)
                if const_expr(PFC):
                    # beta = 1: this tile's old C -> sC, behind k-tile 2. The wait(0) at kt 1 retired the previous
                    # tile's C store (older than every op since), so sC is free. Needs kt_n >= 3: the epilogue's
                    # wait(4) (planes + stage of the next tile issued later) then also covers this load.
                    if kt == 1:
                        if kt_n >= 3:
                            fx.copy(atomC, gtC, sC)
                compute2((gk + kt) % num_buffers, kt, pset, kt_n)
                # next k-step's scale words (same tile only: the next tile's planes are still in flight)
                if kt + 1 < kt_n:
                    load_words(kt + 1, pset)
            gk_reg.store(gk_reg.load() + kt_n)
            # epilogue: the previous C store must be done before sC is rewritten (4 newer ops: planes + stage)
            tdm_ops.tensor_wait(4)
            fx.barrier()
            if const_expr(beta_one):
                if kt_n < 3:  # short window: no prefetch slot, load old C here (exposed)
                    fx.copy(atomC, gtC, sC)
                    tdm_ops.tensor_wait(0)
                    fx.barrier()
            cbase = fx.Int32(fx.ptrtoint(fx.add_offset(base_ptr, C_OFF)))
            for mi in range_constexpr(m_reps):
                row = wmb + mi * WMMA_M + lane16
                for ni in range_constexpr(n_reps):
                    col = wnb + ni * WMMA_N + kgrp * (WMMA_N // 2)
                    v = c_frags[mi * n_reps + ni].load()
                    if const_expr(beta_one):
                        old = (
                            fx.Vector(lds_b128(cbase, (row * C_ROW + col) * 2))
                            .bitcast(out_cls)
                            .to(fx.Float32)
                        )
                        v = v + old
                    h = v.to(out_cls)
                    if const_expr(sumsq):
                        hf = h.to(fx.Float32)
                        sq_reg.store(sq_reg.load() + hf * hf)
                    fx.ptr_store(h.bitcast(fx.Int8), fx.add_offset(base_ptr, C_OFF) + (row * C_ROW + col) * 2)
            fx.barrier()
            fx.copy(atomC, sC, gtC)
        tdm_ops.tensor_wait(0)
        if const_expr(sumsq):
            sv = sq_reg.load()
            tot = sv[0]
            for i_ in range_constexpr(1, WMMA_N // 2):
                tot = tot + sv[i_]
            sq_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(_cast(fx.Float32, arg_sq, 4)))
            _bops.buffer_store(tot, sq_rsrc, bid * block + tid)

    kernel_vkpblk(
        arg_c,
        arg_a,
        arg_b,
        arg_offs,
        arg_perm,
        arg_sa,
        arg_sb,
        arg_sq,
        i32_m,
        i32_n,
        i32_g,
        i32_lda,
        i32_ldb,
        i32_nwg,
    ).launch(grid=(i32_nwg, 1, 1), block=(block, 1, 1), stream=stream)


launch_vkpblk_fp8_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}

_COMPILED: dict = {}


def vkpblk_sumsq_numel(out, tile=(256, 256, 128), m_warp=2, n_warp=4, nwg=None):
    """fp32 elements of the ``sumsq_out`` partial buffer flyvkpblk_fp8 fills (one per thread of every workgroup)."""
    if nwg is None:
        nwg = torch.cuda.get_device_properties(out.device).multi_processor_count
    G, M, N = out.shape
    return min(nwg, (M // tile[0]) * (N // tile[1]) * G) * m_warp * n_warp * 32


def flyvkpblk_fp8(
    lhsP,
    rhsP,
    lhs_s,
    rhs_s,
    poffs_i32,
    OUT_M,
    OUT_N,
    out,
    beta=1.0,
    tile=(256, 256, 128),
    m_warp=2,
    n_warp=4,
    group_m=4,
    kwin_max=4096,
    nwg=None,
    perm=None,
    sumsq_out=None,
):
    """out [G, OUT_M, OUT_N] bf16 (+)= per-expert lhsP[:, seg] @ rhsP[:, seg]^T; lhs_s/rhs_s compact E8M0
    [rows, P/128] uint8. beta 1 accumulates into out, 0 overwrites it.
    perm: int32 [G] expert visiting order (``quantization_fp8blk_moe_kernel.expert_order_lpt_fused``) or None (expert index order). Every output tile is
    computed by one workgroup with the same k sequence whatever the order, so the result is bit-identical.
    sumsq_out: fp32 [vkpblk_sumsq_numel(out)] or None. When given, the kernel also writes per-thread partial sums of
    squares of the bf16 values of ``out`` it stores (all of ``out``: sumsq_out.sum() == out.float().square().sum()
    up to fp32 summation order); every element is overwritten, so no clearing is needed."""
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    assert lhs_s.dtype == torch.uint8 and lhs_s.shape[1] * 128 == lhsP.shape[1]
    G = poffs_i32.numel() - 1
    tm, tn, tk = tile
    assert OUT_M % tm == 0 and OUT_N % tn == 0
    total = (OUT_M // tm) * (OUT_N // tn) * G
    if nwg is None:
        nwg = torch.cuda.get_device_properties(out.device).multi_processor_count
    nwg = min(nwg, total)
    a_id = 0 if lhsP.dtype == torch.float8_e4m3fn else 1
    b_id = 0 if rhsP.dtype == torch.float8_e4m3fn else 1
    assert beta in (0.0, 1.0)
    bo = 1 if beta == 1.0 else 0
    lpt = 0 if perm is None else 1
    if lpt:
        assert perm.dtype == torch.int32 and perm.numel() == G and perm.is_contiguous()
    sq = 0
    if sumsq_out is not None:
        assert sumsq_out.dtype == torch.float32 and sumsq_out.is_contiguous()
        assert sumsq_out.numel() >= nwg * m_warp * n_warp * 32, (sumsq_out.numel(), nwg)
        sq = 1
    args = (
        _ptr(out),
        _ptr(lhsP),
        _ptr(rhsP),
        _ptr(poffs_i32),
        _ptr(perm if lpt else poffs_i32),
        _ptr(lhs_s),
        _ptr(rhs_s),
        _ptr(sumsq_out if sq else out),
        OUT_M,
        fx.Stream(torch.cuda.current_stream(device=lhsP.device)),
        OUT_N,
        G,
        lhsP.stride(0),
        rhsP.stride(0),
        nwg,
        tm,
        tn,
        tk,
        m_warp,
        n_warp,
        a_id,
        b_id,
        bo,
        group_m,
        kwin_max,
        lpt,
        sq,
    )
    key = (
        tm,
        tn,
        tk,
        m_warp,
        n_warp,
        a_id,
        b_id,
        bo,
        group_m,
        kwin_max,
        lpt,
        sq,
    )
    c = _COMPILED.get(key)
    if c is None:
        _COMPILED[key] = flyc.compile(launch_vkpblk_fp8_gfx1250, *args)
    else:
        c(*args)
    return out
