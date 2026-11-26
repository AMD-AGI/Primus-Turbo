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

"""FlyDSL gfx1250 FP8 block-scaled (UE8M0, 128x1 along K) variable-K wgrad, persistent: "vkq".

Same contract, tiling, k order, epilogue and sumsq layout as vkp (grouped_gemm_fp8blk_vkp_kernel.py), so C
and the sumsq partials are bit-identical to vkp:
  out[g] (+)= lhsP[:, seg_g] @ rhsP[:, seg_g]^T, lhsP [OUT_M, P], rhsP [OUT_N, P] fp8, compact E8M0 col scales
  lhs_s [OUT_M, P/128], rhs_s [OUT_N, P/128] uint8 (one byte per 128 tokens), seg_g = [poffs[g], poffs[g+1]).
Difference from vkp: each tile's window of the two scale planes (512 rows x kt_n bytes) is loaded with VMEM
global_load_async_to_lds_b32 (one lane per plane row, dword-aligned, ASYNCcnt) one whole tile ahead, instead of two
TDM ops in the in-order TDM queue between the k-tiles. Calls whose scale planes are not dword-addressable, or with
another tiling, run vkp.
vkw (launch_vkw_fp8_gfx1250, beta 0, every window a multiple of 256 tokens): the same contract on 4 waves of
128x128 with 256-token k-tiles and C stored through the freed ring stage as whole 256-B row segments; C and the
sumsq partials bit-identical to vkq. flyvkqblk_fp8(vkw=True) selects it for beta 0; beta 1 runs vkq.
Requires OUT_M % 256 == 0, OUT_N % 256 == 0, every window a multiple of 128 and >= 1 k-tile, <= kwin_max.
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
from .grouped_gemm_fp8blk_vkp_kernel import vkpblk_sumsq_numel

__all__ = ["flyvkqblk_fp8", "vkpblk_sumsq_numel"]


@flyc.jit
def launch_vkqblk_fp8_gfx1250(
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
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    sched2 = 1  # row groups of A fragments read ahead of their WMMAs
    num_buffers = 2
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    MX_BLOCK = 32
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    assert tile_m % (m_warp * WMMA_M) == 0 and tile_n % (n_warp * WMMA_N) == 0
    assert tile_k == WMMA_K, "one 128-token scale per k-tile"
    assert tile_m == block and tile_n == block, "planes: one lane per plane row"
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
    SPB = SPW + 4  # LDS plane row: the window from its dword-aligned start (<= 2 bytes earlier) fits
    PL_A = tile_m * SPB
    PL_B = tile_n * SPB
    PSET = PL_A + PL_B
    PL_OFF = C_OFF + tile_m * C_ROW * 2
    ARENA = PL_OFF + 2 * PSET
    assert ARENA <= 320 * 1024, f"LDS arena {ARENA} B exceeds 320 KiB"

    kernel_name = (
        f"moe_vkqblk_fp8_gfx1250_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}_a{a_id}{b_id}"
        f"_b{beta_one}_w{kwin_max}"
        + ("_lpt" if lpt else "")
        + ("_pfc" if beta_one else "")
        + ("_sq" if sumsq else "")
        + f"_s{sched2}"
    )
    PFC = bool(beta_one)

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_vkqblk(
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
        gSA8 = _cast(fx.Int8, arg_sa, 4)
        gSB8 = _cast(fx.Int8, arg_sb, 4)

        def sA_of(s):
            return _view(_stage(s), a_cls, (tile_m, tile_k), (A_ROW, 1))

        def sB_of(s):
            return _view(fx.add_offset(_stage(s), STAGE_A), b_cls, (tile_n, tile_k), (B_ROW, 1))

        def issue(s, offa, offb):
            fx.copy(atomA, gA, sA_of(s), imm_offset=offa)
            fx.copy(atomB, gB, sB_of(s), imm_offset=offb)

        def issue_planes(pset, blk_m, blk_n, kb, kt_n):
            """E3: the tile's window of both scale planes -> LDS set pset by VMEM async loads (ASYNCcnt): thread tid
            copies A-plane row blk_m + tid and B-plane row blk_n + tid, as dwords from the window's dword-aligned
            start c0 & ~3 (the reader adds the shift c0 & 3)."""
            c0 = kb // 128
            c0a = c0 - c0 % 4
            nd = (c0 % 4 + kt_n + 3) // 4
            ga = fx.Int64(blk_m + tid) * fx.Int64(ska) + fx.Int64(c0a)
            gb = fx.Int64(blk_n + tid) * fx.Int64(skb) + fx.Int64(c0a)
            la = PL_OFF + pset * PSET + tid * SPB
            lb = la + PL_A
            for d in range(nd):
                d64 = fx.Int64(d * 4)
                rocdl.global_load_async_to_lds_b32(
                    fx.add_offset(gSA8, ga + d64).llvm_ptr,
                    fx.recast_iter(fx.Int8, fx.add_offset(base_ptr, la + d * 4)).llvm_ptr,
                    0,
                )
                rocdl.global_load_async_to_lds_b32(
                    fx.add_offset(gSB8, gb + d64).llvm_ptr,
                    fx.recast_iter(fx.Int8, fx.add_offset(base_ptr, lb + d * 4)).llvm_ptr,
                    0,
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

        def load_words(kx, pset):
            """the raw scale words of plane byte kx (= k-step + the window's dword shift) -> sw_regs"""
            poff = PL_OFF + pset * PSET
            koff = (kx // 4) * 4
            for ni in range_constexpr(n_reps):
                sw_regs[ni].store(lds_b32(lds_base, poff + PL_A + (wnb + ni * WMMA_N + lane16) * SPB + koff))
            for mi in range_constexpr(m_reps):
                sw_regs[n_reps + mi].store(
                    lds_b32(lds_base, poff + (wmb + mi * WMMA_M + lane16) * SPB + koff)
                )

        def compute2(s, kx):
            base = fx.Int32(fx.ptrtoint(_stage(s)))
            sel = (kx % 4) * 0x01010101  # v_perm_b32: every result byte = byte kx%4 of the word
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

        # prologue: tile 0's planes (set 0, async) and first k-tile, then one TDM op standing in for the C store that
        # later tiles keep in flight (keeps the TDM wait counts uniform)
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
            ksh = (kb // 128) % 4  # this tile's plane rows start at its window's dword-aligned column
            for cf in c_frags:
                cf.fill(0.0)
            gk = gk_reg.load()[0]
            gtC, atomC = c_view(g, blk_m, blk_n)
            for kt in range(kt_n):
                if kt == 0:
                    tdm_ops.tensor_wait(1)  # the previous tile's C store may still be in flight
                    rocdl.s_wait_asynccnt(0)  # this tile's planes (issued a whole tile ago)
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
                    load_words(ksh, pset)
                    # the next tile's planes -> the other set (its last reader, the previous tile, is past this
                    # barrier); waited for at the next tile's k-step 0
                    issue_planes(1 - pset, nbm, nbn, nkb, nktn)
                f = kt + 1
                if f < kt_n:
                    oa0, ob0 = offs_of(blk_m, blk_n, kb, f)
                    issue((gk + f) % num_buffers, oa0, ob0)
                else:
                    oa1, ob1 = offs_of(nbm, nbn, nkb, 0)
                    issue((gk + f) % num_buffers, oa1, ob1)
                if const_expr(PFC):
                    # beta = 1: this tile's old C -> sC, behind k-tile 2; needs kt_n >= 3
                    if kt == 1:
                        if kt_n >= 3:
                            fx.copy(atomC, gtC, sC)
                compute2((gk + kt) % num_buffers, kt + ksh)
                if kt + 1 < kt_n:
                    load_words(kt + 1 + ksh, pset)
            gk_reg.store(gk_reg.load() + kt_n)
            # epilogue: the previous C store must be done before sC is rewritten (2 newer ops: the next k-tile 0)
            tdm_ops.tensor_wait(2)
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
        rocdl.s_wait_asynccnt(0)  # the clamped last "next tile" planes
        if const_expr(sumsq):
            sv = sq_reg.load()
            tot = sv[0]
            for i_ in range_constexpr(1, WMMA_N // 2):
                tot = tot + sv[i_]
            sq_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(_cast(fx.Float32, arg_sq, 4)))
            _bops.buffer_store(tot, sq_rsrc, bid * block + tid)

    kernel_vkqblk(
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


launch_vkqblk_fp8_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}


@flyc.jit
def launch_vkw_fp8_gfx1250(
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
    a_id: Constexpr[int],
    b_id: Constexpr[int],
    group_m: Constexpr[int] = 4,
    kwin_max: Constexpr[int] = 4096,
    lpt: Constexpr[int] = 0,
    sumsq: Constexpr[int] = 0,
):
    """ "vkw" (beta 0): the vkq wgrad contract with a 4-wave main loop.
    * 4 waves (2x2) of 128x128 per 256x256 tile, 256-token k-tiles (two WMMA k-steps each, in k order), 2-stage TDM
      ring (2 x 139 KB, 272-B rows), persistent LPT tile -> workgroup map as vkq, scale planes per tile loaded with
      VMEM as in vkq (2 sets); each k-step's 16 scale operands are loaded from the planes and byte-replicated
      (v_perm_b32) just in time.
    * epilogue: after a barrier the last k-step's stage is free; each wave writes its 128x128 bf16 C there (its own
      128 x 272 B region) and stores it as whole 256-B row segments (global_store_b128, 2 rows per instruction).
    * bit-identical to vkq: the same WMMAs per output fragment in the same k order, the same tiles -> workgroups, and
      the sumsq partial of every vkq thread (8-wave 2x4 map, 128x64 per wave) is one of two accumulators of the vkw
      thread that holds its fragments, summed in the same (tile, mi, ni) order and stored at the vkq index.
    Requires every window a multiple of 256 tokens (>= 256), <= kwin_max; OUT_M, OUT_N multiples of 256."""
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    tile_m = tile_n = 256
    TK = 256
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    KS = TK // WMMA_K
    MX_BLOCK = 32
    m_warp, n_warp = 2, 2
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    out_cls = fx.BFloat16
    ROW = TK + 16
    STAGE_A = tile_m * ROW
    PITCH = 2 * STAGE_A
    SPW = kwin_max // 128
    SPB = SPW + 4
    PL_A = tile_m * SPB
    PSET = 2 * PL_A
    PL_OFF = 2 * PITCH
    ARENA = PL_OFF + 2 * PSET
    assert ARENA <= 320 * 1024, f"LDS arena {ARENA} B exceeds 320 KiB"
    warp_m, warp_n = tile_m // m_warp, tile_n // n_warp
    m_reps, n_reps = warp_m // WMMA_M, warp_n // WMMA_N
    assert warp_m * ROW * num_waves == PITCH  # the per-wave C regions tile one stage exactly
    VW = 2  # vkq waves (n direction) per vkw wave: sumsq accumulators per thread
    kernel_name = (
        f"moe_vkwblk_fp8_gfx1250_t256x256x256_w2x2_a{a_id}{b_id}_w{kwin_max}"
        + ("_lpt" if lpt else "")
        + ("_sq" if sumsq else "")
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_vkw(
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
            return g, pm * tile_m, pn * tile_n, kb, (ke - kb) // WMMA_K

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA)
        base_ptr = arena.base_ptr

        def _cast(cls, p, align=16):
            return fx.recast_iter(fx.PointerType.get(cls.ir_type, p.address_space, align), p)

        def _view(ptr, cls, shape, stride, align=16):
            return fx.Tensor(fx.make_view(_cast(cls, ptr, align), fx.make_layout(shape, stride)))

        def _gview(base, off, shape, stride):
            return fx.Tensor(fx.make_view(fx.add_offset(base, off), fx.make_layout(shape, stride)))

        gA = _gview(_cast(a_cls, arg_a), fx.Int64(0), (tile_m, TK), (i32_lda, 1))
        gB = _gview(_cast(b_cls, arg_b), fx.Int64(0), (tile_n, TK), (i32_ldb, 1))
        atomA = fx.rocdl.make_tdm_atom(
            gA, [None, None], strides=[lda64, None], num_warps=num_waves, pad_interval=TK, pad_amount=ROW - TK
        )
        atomB = fx.rocdl.make_tdm_atom(
            gB, [None, None], strides=[ldb64, None], num_warps=num_waves, pad_interval=TK, pad_amount=ROW - TK
        )
        gSA8 = _cast(fx.Int8, arg_sa, 4)
        gSB8 = _cast(fx.Int8, arg_sb, 4)

        def issue(s, offa, offb):
            st = fx.add_offset(base_ptr, s * PITCH)
            fx.copy(atomA, gA, _view(st, a_cls, (tile_m, TK), (ROW, 1)), imm_offset=offa)
            fx.copy(
                atomB, gB, _view(fx.add_offset(st, STAGE_A), b_cls, (tile_n, TK), (ROW, 1)), imm_offset=offb
            )

        def offs_of(blk_m, blk_n, kb, kt):
            k = fx.Int64(kb) + fx.Int64(kt) * TK
            return fx.Int64(blk_m) * lda64 + k, fx.Int64(blk_n) * ldb64 + k

        def issue_planes(pset, blk_m, blk_n, kb, kt_n):
            """plane rows of the tile's window -> plane set pset (VMEM async, one lane per row, dwords from the
            window's dword-aligned start c0 & ~3); 128 threads: rows tid and tid + 128 of each plane"""
            c0 = kb // 128
            c0a = c0 - c0 % 4
            nd = (c0 % 4 + kt_n + 3) // 4
            for h in range_constexpr(2):
                r = tid + h * block
                ga = fx.Int64(blk_m + r) * fx.Int64(ska) + fx.Int64(c0a)
                gb = fx.Int64(blk_n + r) * fx.Int64(skb) + fx.Int64(c0a)
                la = PL_OFF + pset * PSET + r * SPB
                lb = la + PL_A
                for d in range(nd):
                    d64 = fx.Int64(d * 4)
                    rocdl.global_load_async_to_lds_b32(
                        fx.add_offset(gSA8, ga + d64).llvm_ptr,
                        fx.recast_iter(fx.Int8, fx.add_offset(base_ptr, la + d * 4)).llvm_ptr,
                        0,
                    )
                    rocdl.global_load_async_to_lds_b32(
                        fx.add_offset(gSB8, gb + d64).llvm_ptr,
                        fx.recast_iter(fx.Int8, fx.add_offset(base_ptr, lb + d * 4)).llvm_ptr,
                        0,
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
        wmb = (wave // n_warp) * warp_m
        wnb = (wave % n_warp) * warp_n
        v0 = (wave // n_warp) * 4 + (wave % n_warp) * VW  # first vkq wave whose fragments this thread holds
        c_frags = [fx.make_rmem_tensor(WMMA_N // 2, fx.Float32) for _ in range_constexpr(m_reps * n_reps)]
        for cf in c_frags:
            cf.fill(0.0)
        gk_reg = fx.make_rmem_tensor(1, fx.Int32)
        gk_reg.fill(0)
        if const_expr(sumsq):
            sq_regs = [fx.make_rmem_tensor(WMMA_N // 2, fx.Float32) for _ in range_constexpr(VW)]
            for q_ in range_constexpr(VW):
                sq_regs[q_].fill(0.0)
        lds_base = fx.Int32(fx.ptrtoint(base_ptr))

        def frag(base, row, ks):
            b0 = row * ROW + ks * WMMA_K + kgrp * 16
            v = [fx.Vector(lds_b128(base, b0 + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            reg = fx.make_rmem_tensor(16, fx.Int32)
            reg.store(v01.shuffle(v23, list(range(16))))
            return reg

        NW = n_reps + m_reps
        WR = [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(2 * NW)]

        def load_raw(kx0, pset):
            """the two plane dwords holding bytes kx0, kx0 + 1 of the wave's B rows (ni) and A rows (mi)"""
            poff = PL_OFF + pset * PSET
            koff = (kx0 // 4) * 4
            for q in range_constexpr(NW):
                if const_expr(q < n_reps):
                    ad = poff + PL_A + (wnb + q * WMMA_N + lane16) * SPB + koff
                else:
                    ad = poff + (wmb + (q - n_reps) * WMMA_M + lane16) * SPB + koff
                WR[2 * q].store(lds_b32(lds_base, ad))
                WR[2 * q + 1].store(lds_b32(lds_base, ad + 4))

        def perms(kx0, ks):
            """k-step ks's scale operands: byte kx0 + ks of each row replicated (v_perm_b32 over {w1:w0})"""
            sel = (kx0 % 4) * 0x01010101 + ks * 0x01010101
            w = [r.load()[0] for r in WR]
            sb = [fx.Int32(rocdl.perm_b32(w[2 * ni + 1], w[2 * ni], sel)) for ni in range_constexpr(n_reps)]
            sa = [
                fx.Int32(rocdl.perm_b32(w[2 * (n_reps + mi) + 1], w[2 * (n_reps + mi)], sel))
                for mi in range_constexpr(m_reps)
            ]
            return sb, sa

        H = m_reps // 2

        def block_mma(ms, ns, af, bf, sb, sa):
            for i in range_constexpr(len(ms)):
                for j in range_constexpr(len(ns)):
                    mi, ni = ms[i], ns[j]
                    idx = mi * n_reps + ni
                    fx.gemm(
                        mma_atom, c_frags[idx], bf[j], af[i], c_frags[idx], scale_a=sb[ni], scale_b=sa[mi]
                    )

        def compute(s, kx0, pset):
            base = lds_base + s * PITCH
            load_raw(kx0, pset)
            for ks in range_constexpr(KS):
                rocdl.sched_barrier(0)
                sb, sa = perms(kx0, ks)
                bl = [frag(base + STAGE_A, wnb + ni * WMMA_N + lane16, ks) for ni in range_constexpr(H)]
                at = [frag(base, wmb + mi * WMMA_M + lane16, ks) for mi in range_constexpr(H)]
                rocdl.sched_barrier(0)
                ab = [frag(base, wmb + (H + mi) * WMMA_M + lane16, ks) for mi in range_constexpr(H)]
                rocdl.sched_barrier(0)
                block_mma(list(range(H)), list(range(H)), at, bl, sb, sa)
                rocdl.sched_barrier(0)
                br = [frag(base + STAGE_A, wnb + (H + ni) * WMMA_N + lane16, ks) for ni in range_constexpr(H)]
                rocdl.sched_barrier(0)
                block_mma(list(range(H, m_reps)), list(range(H)), ab, bl, sb, sa)
                rocdl.sched_barrier(0)
                block_mma(list(range(H)), list(range(H, n_reps)), at, br, sb, sa)
                rocdl.sched_barrier(0)
                block_mma(list(range(H, m_reps)), list(range(H, n_reps)), ab, br, sb, sa)
            rocdl.sched_barrier(0)

        gC8 = _cast(fx.Int8, _cast(out_cls, arg_c))
        c_ld = fx.Int64(i32_n)

        def epilogue(g, blk_m, blk_n, s_last):
            """accumulators -> bf16 (+ sumsq, vkq order) -> this wave's 128 x 272 B region of the freed stage -> whole
            256-B row segments to global (lane l: row 2q + l // 16, 16 B at column (l % 16) * 8); zeroed"""
            fx.barrier()  # every wave is done reading the last step's stage
            scr = s_last * PITCH + wave * (warp_m * ROW)
            for mi in range_constexpr(m_reps):
                rocdl.sched_barrier(0)
                for nj in range_constexpr(n_reps):
                    h = c_frags[mi * n_reps + nj].load().to(out_cls)
                    if const_expr(sumsq):
                        hf = h.to(fx.Float32)
                        q = sq_regs[nj // (n_reps // VW)]
                        q.store(q.load() + hf * hf)
                    fx.ptr_store(
                        h.bitcast(fx.Int8),
                        base_ptr + (scr + (mi * WMMA_M + lane16) * ROW + (nj * WMMA_N + kgrp * 8) * 2),
                    )
                    c_frags[mi * n_reps + nj].fill(0.0)
            cb = fx.Int64(g) * fx.Int64(i32_m) * c_ld + fx.Int64(blk_m + wmb) * c_ld + fx.Int64(blk_n + wnb)
            lr = lane // 16
            lc = lane % 16
            for q2 in range_constexpr(warp_m // 2):
                if const_expr(q2 % 8 == 0):
                    rocdl.sched_barrier(0)
                r = q2 * 2 + lr
                v = fx.Vector(lds_b128(lds_base, scr + r * ROW + lc * 16))
                fx.ptr_store(v.bitcast(fx.Int8), gC8 + (cb + fx.Int64(r) * c_ld + fx.Int64(lc * 8)) * 2)

        # prologue: the first tile's planes (set 0) and k-tile 0 (stage 0)
        g0, bm0, bn0, kb0, ktn0 = tile_params(bid)
        issue_planes(0, bm0, bn0, kb0, ktn0)
        oa, ob = offs_of(bm0, bn0, kb0, 0)
        issue(0, oa, ob)

        for it in range(my_cnt):
            t = bid + it * i32_nwg
            g, blk_m, blk_n, kb, kt_n = tile_params(t)
            t_nx = (t + i32_nwg < total).select(t + i32_nwg, last_t)
            _, nbm, nbn, nkb, nktn = tile_params(t_nx)
            pset = it % 2
            ksh = (kb // 128) % 4
            n2 = kt_n // KS
            gk = gk_reg.load()[0]
            for j in range(n2):
                tdm_ops.tensor_wait(0)
                if j == 0:
                    rocdl.s_wait_asynccnt(0)  # this tile's planes
                fx.barrier()
                if j == 0:
                    issue_planes(1 - pset, nbm, nbn, nkb, nktn)
                f = j + 1
                if f < n2:
                    oa0, ob0 = offs_of(blk_m, blk_n, kb, f)
                    issue((gk + f) % 2, oa0, ob0)
                else:
                    oa1, ob1 = offs_of(nbm, nbn, nkb, 0)
                    issue((gk + f) % 2, oa1, ob1)
                compute((gk + j) % 2, j * KS + ksh, pset)
            gk_reg.store(gk_reg.load() + n2)
            epilogue(g, blk_m, blk_n, (gk + n2 - 1) % 2)
        tdm_ops.tensor_wait(0)
        rocdl.s_wait_asynccnt(0)
        if const_expr(sumsq):
            sq_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(_cast(fx.Float32, arg_sq, 4)))
            for q_ in range_constexpr(VW):
                sv = sq_regs[q_].load()
                tot = sv[0]
                for i_ in range_constexpr(1, WMMA_N // 2):
                    tot = tot + sv[i_]
                _bops.buffer_store(tot, sq_rsrc, bid * 256 + (v0 + q_) * WAVE + lane)

    kernel_vkw(
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


launch_vkw_fp8_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}


_COMPILED: dict = {}


def flyvkqblk_fp8(
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
    vkw=False,
):
    """Drop-in for flyvkpblk_fp8 (same arguments, bit-identical out and sumsq_out).
    out [G, OUT_M, OUT_N] bf16 (+)= per-expert lhsP[:, seg] @ rhsP[:, seg]^T; lhs_s/rhs_s compact E8M0
    [rows, P/128] uint8 (row stride P/128 a multiple of 4). beta 1 accumulates into out, 0 overwrites it.
    vkw: the caller guarantees every expert window is a multiple of 256 tokens (the MegaMoE pool alignment; not
    checked here, that would need a host sync); beta-0 calls then run the vkw kernel (same bits, faster)."""
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    assert lhs_s.dtype == torch.uint8 and lhs_s.shape[1] * 128 == lhsP.shape[1]
    if not (
        lhs_s.is_contiguous()
        and rhs_s.is_contiguous()
        and lhs_s.shape[1] % 4 == 0
        and lhs_s.data_ptr() % 4 == 0
        and rhs_s.data_ptr() % 4 == 0
        and tuple(tile) == (256, 256, 128)
        and m_warp * n_warp * 32 == 256
    ):
        # plane rows not dword-addressable (or another tiling): vkp computes the same bits
        from .grouped_gemm_fp8blk_vkp_kernel import flyvkpblk_fp8

        return flyvkpblk_fp8(
            lhsP,
            rhsP,
            lhs_s,
            rhs_s,
            poffs_i32,
            OUT_M,
            OUT_N,
            out,
            beta=beta,
            tile=tile,
            m_warp=m_warp,
            n_warp=n_warp,
            group_m=group_m,
            kwin_max=kwin_max,
            nwg=nwg,
            perm=perm,
            sumsq_out=sumsq_out,
        )
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
    if vkw and not bo:
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
            a_id,
            b_id,
            group_m,
            kwin_max,
            lpt,
            sq,
        )
        key = ("vkw", a_id, b_id, group_m, kwin_max, lpt, sq)
        c = _COMPILED.get(key)
        if c is None:
            _COMPILED[key] = flyc.compile(launch_vkw_fp8_gfx1250, *args)
        else:
            c(*args)
        return out
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
    key = (tm, tn, tk, m_warp, n_warp, a_id, b_id, bo, group_m, kwin_max, lpt, sq)
    c = _COMPILED.get(key)
    if c is None:
        _COMPILED[key] = flyc.compile(launch_vkqblk_fp8_gfx1250, *args)
    else:
        c(*args)
    return out
