"""FlyDSL gfx1250 FP8 block-scaled (UE8M0, 128x1 along K) variable-K wgrad, persistent: "vkq" (wgrad line).

Same contract, tiling, k order, epilogue and sumsq layout as vkp (grouped_gemm_fp8blk_vkp_flydsl_gfx1250.py, sched2
path, nbuf 2), so C and the sumsq partials are bit-identical to vkp:
  out[g] (+)= lhsP[:, seg_g] @ rhsP[:, seg_g]^T, lhsP [OUT_M, P], rhsP [OUT_N, P] fp8, compact E8M0 col scales
  lhs_s [OUT_M, P/128], rhs_s [OUT_N, P/128] uint8 (one byte per 128 tokens), seg_g = [poffs[g], poffs[g+1]).
Differences from vkp (wgrad ledger):
- E3 (planes): each tile's window of the two scale planes (512 rows x kt_n bytes) is loaded with VMEM
  global_load_async_to_lds_b32 (one lane per plane row, dword-aligned, ASYNCcnt) one whole tile ahead, instead of two
  TDM ops in the in-order TDM queue between the k-tiles (vkp: 512 tiny TDM rows per tile, 17% of the kernel).
  This is the default path (nbuf 2, beta 0 and 1): -13% per call vs vkp at the DSv4 proxy shapes.
- E6 nbuf 4 (beta 0, experimental, not faster than nbuf 2): launch_vkq4_fp8_gfx1250, see its docstring.
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
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm

from primus_turbo.flydsl.utils import buffer_ops as _bops

from .grouped_gemm_fp8blk_vkp_flydsl_gfx1250 import _make_lds_load, expert_order_lpt, vkpblk_sumsq_numel

WAVE = 32
_A_ID = {0: fx.Float8E4M3FN, 1: fx.Float8E5M2}

__all__ = ["flyvkqblk_fp8", "expert_order_lpt", "vkpblk_sumsq_numel"]


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
    prefetch_c: Constexpr[int] = 0,
    sumsq: Constexpr[int] = 0,
    sched2: Constexpr[int] = 1,
):
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    num_buffers = 2
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    MX_BLOCK = 32
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    assert tile_m % (m_warp * WMMA_M) == 0 and tile_n % (n_warp * WMMA_N) == 0
    assert tile_k == WMMA_K, "one 128-token scale per k-tile"
    assert tile_m == block and tile_n == block, "planes: one lane per plane row"
    assert sched2 >= 1
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

    kernel_name = (f"moe_vkqblk_fp8_gfx1250_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}_a{a_id}{b_id}"
                   f"_b{beta_one}_w{kwin_max}" + ("_lpt" if lpt else "") + ("_pfc" if (prefetch_c and beta_one) else "")
                   + ("_sq" if sumsq else "") + f"_s{sched2}")
    PFC = bool(prefetch_c and beta_one)

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

        i32_ptr = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
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
        atomA = fx.rocdl.make_tdm_atom(gA, [None, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)
        atomB = fx.rocdl.make_tdm_atom(gB, [None, None], strides=[ldb64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)
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
                    fx.recast_iter(fx.Int8, fx.add_offset(base_ptr, la + d * 4)).llvm_ptr, 0)
                rocdl.global_load_async_to_lds_b32(
                    fx.add_offset(gSB8, gb + d64).llvm_ptr,
                    fx.recast_iter(fx.Int8, fx.add_offset(base_ptr, lb + d * 4)).llvm_ptr, 0)

        mma_atom = fx.make_mma_atom(fx.rocdl.WMMAScale(WMMA_M, WMMA_N, WMMA_K, b_cls, a_cls, fx.Float32,
                                                       block_size=MX_BLOCK))
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
                sw_regs[n_reps + mi].store(lds_b32(lds_base, poff + (wmb + mi * WMMA_M + lane16) * SPB + koff))

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
                    fx.gemm(mma_atom, c_frags[idx], bf[ni], afs[mi], c_frags[idx], scale_a=sbv[ni], scale_b=sav_all[mi])
                rocdl.sched_barrier(0)

        def offs_of(blk_m, blk_n, kb, kt):
            k = fx.Int64(kb) + fx.Int64(kt) * tile_k
            return fx.Int64(blk_m) * lda64 + k, fx.Int64(blk_n) * ldb64 + k

        c_ld = fx.Int64(i32_n)
        gC_base = _cast(out_cls, arg_c)
        sC = _view(fx.add_offset(base_ptr, C_OFF), out_cls, (tile_m, C_ROW), (C_ROW, 1))

        def c_view(g, blk_m, blk_n):
            gtC = _gview(gC_base, fx.Int64(g) * fx.Int64(i32_m) * c_ld + fx.Int64(blk_m) * c_ld + fx.Int64(blk_n),
                         (tile_m, C_ROW), (C_ROW, 1))
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
                            tdm_ops.tensor_wait(1)  # the old-C prefetch (newest op, issued at kt 1) may stay in flight
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
                    # beta = 1: this tile's old C -> sC, behind k-tile 2 (see vkp); needs kt_n >= 3
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
                if const_expr(PFC):
                    if kt_n < 3:  # short window: no prefetch slot, load old C here (exposed)
                        fx.copy(atomC, gtC, sC)
                        tdm_ops.tensor_wait(0)
                        fx.barrier()
                else:
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
                        old = fx.Vector(lds_b128(cbase, (row * C_ROW + col) * 2)).bitcast(out_cls).to(fx.Float32)
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

    kernel_vkqblk(arg_c, arg_a, arg_b, arg_offs, arg_perm, arg_sa, arg_sb, arg_sq, i32_m, i32_n, i32_g, i32_lda,
                  i32_ldb, i32_nwg).launch(grid=(i32_nwg, 1, 1), block=(block, 1, 1), stream=stream)


launch_vkqblk_fp8_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}


@flyc.jit
def launch_vkq4_fp8_gfx1250(
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
    early: Constexpr[int] = 2,
    sig: Constexpr[int] = -1,
    abl: Constexpr[int] = 0,
    clk: Constexpr[int] = 0,
):
    """E6 vkq4 (beta 0): 4-stage ring of 128-token k-steps over one flat stream of all the workgroup's tiles, 2 k-steps
    (128 KB) in flight, next-step fragment prefetch, C from the accumulators through a per-wave line scratch.
    Same tiles -> workgroups (persistent LPT round robin), same 2x4 wave map (128x64 per wave), same WMMAs / operands /
    per-element k order, same bf16 rounding and same per-thread sumsq order as vkp: C and the sumsq partials are
    bit-identical.
    * LDS: two 272 B-pitch slots per operand (mx8w form: each wave TDM-loads its own whole [32, 128 B] tile, pad
      interval 128 / amount 144; stage s at slot s // 2, columns (s % 2) * 128); 16-row fragment reads conflict-free.
      Scale planes (one tile window, 2 sets by tile parity) on VMEM as vkq E3, issued in the epilogue two tiles
      ahead (no WMMA in flight there). 4 x 69632 + 2 x 512 x (kwin/128 + 4) B = 315392 B at kwin 4096.
    * k-step J (global): its B fragments, A fragments of row groups 0 / 1 and its 12 scale words were read during step
      J-1. At row group `sig` each wave waits for its TDM ops of step J+1 (and, at a tile's last step, for the next
      tile's planes) and signals; at row group `early` it waits for the barrier: step J+1 is visible to every wave and
      the stage of step J-1 is free -> step J+3 is issued into it (J+3 may belong to the next tile or the one after:
      every window is >= 2 k-steps), then step J+1's B fragments (row groups early..early+3), its scale words and its
      A fragments of row groups 0 / 1 (row groups 6 / 7) are read into the other register set. A fragments of this
      step: two row groups ahead, in the middle of a row group.
    * epilogue: barrier (the last step's stage is free) -> each wave writes its 8 row groups x 64 columns as bf16
      through a 16-row band of that stage's half-rows and stores whole 128-B lines (mx8w store_c); the next tile's
      step 3 is issued into that stage at the next barrier.
    Requires every window a multiple of 256 tokens (even k-step count; the MegaMoE pool aligns each expert to 256
    rows), >= 256 tokens, <= kwin_max; OUT_M, OUT_N multiples of 256.
    abl (timing ablations, wrong output): 1 no C global stores (sumsq of the accumulators kept), 2 no refill after the
    prologue, 4 no fragment loads / WMMAs, 8 no epilogue (tile summed into one store per lane, no sumsq), 16 no
    per-step barrier / TDM waits / issue (with 2). clk (diagnostic): thread 0 writes [cycles, 100 MHz realtime] at start and
    end (4 x i64) at arg_sq + nwg * 256 * 4 + bid * 32 (needs sumsq and a larger sumsq buffer)."""
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    tile_m = tile_n = 256
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    MX_BLOCK = 32
    m_warp, n_warp = 2, 4
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    out_cls = fx.BFloat16
    PITCH = 272
    HALF = 128
    SLOT = tile_m * PITCH
    B_OFF = 2 * SLOT
    SPW = kwin_max // 128
    SPB = SPW + 4
    PL_A = tile_m * SPB
    PSET = 2 * PL_A
    PL_OFF = 4 * SLOT
    ARENA = PL_OFF + 2 * PSET
    assert ARENA <= 320 * 1024, f"LDS arena {ARENA} B exceeds 320 KiB"
    ROWS_PW = tile_m // num_waves
    m_reps, n_reps = 8, 4
    warp_m, warp_n = 128, 64
    assert 1 <= early <= 4
    SIG = early if sig < 0 else sig
    assert 0 <= SIG <= early
    assert clk == 0 or sumsq
    kernel_name = (f"moe_vkq4blk_fp8_gfx1250_t256x256x128_w2x4_a{a_id}{b_id}_w{kwin_max}" + ("_lpt" if lpt else "")
                   + ("_sq" if sumsq else "") + f"_e{early}" + (f"_s{SIG}" if SIG != early else "")
                   + (f"_abl{abl}" if abl else "") + ("_clk" if clk else ""))

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_vkq4(
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

        def _clock():
            i64 = _ir.IntegerType.get_signless(64)
            return (fx.Int64(_llvm.call_intrinsic(i64, "llvm.readcyclecounter", [], [], [])),
                    fx.Int64(_llvm.call_intrinsic(i64, "llvm.readsteadycounter", [], [], [])))

        if const_expr(clk):
            clk0 = _clock()
        zero = bid * fx.Int32(0)
        one = zero + fx.Int32(1)
        n_tiles_m = i32_m // tile_m
        n_tiles_n = i32_n // tile_n
        tpg = n_tiles_m * n_tiles_n
        total = tpg * i32_g
        my_cnt = rocdl.readfirstlane(_T.i32, (total - bid + i32_nwg - 1) // i32_nwg)

        i32_ptr = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        lda64 = fx.Int64(i32_lda)
        ldb64 = fx.Int64(i32_ldb)
        ska = i32_lda // 128
        skb = i32_ldb // 128
        offs_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(fx.recast_iter(i32_ptr, arg_offs)))
        perm_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(fx.recast_iter(i32_ptr, arg_perm)))

        def _sld(rsrc, i):
            return fx.Int32(_bops.buffer_load(rsrc, i, vec_width=1, is_scalar=True))

        def tile_params(it):
            """(g, blk_m, blk_n, kb, kt_n) of this workgroup's tile number it (clamped to its first tile past the
            end) and whether it exists"""
            ok = (it < my_cnt).select(one, zero)
            t = bid + (it < my_cnt).select(it, zero) * i32_nwg
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
            return (g, pm * tile_m, pn * tile_n, kb, (ke - kb) // WMMA_K), rocdl.readfirstlane(_T.i32, ok)

        # E7: tile parameters carried across tiles (cur, next, next-next) so each tile does one fresh lookup, in the
        # epilogue (its scalar-load latency overlaps the C stores) instead of three dependent lookups at tile start
        TPK = [fx.make_rmem_tensor(4, fx.Int32) for _ in range_constexpr(3)]

        def tp_put(i, tpv, ok):
            g, blk_m, blk_n, kb, kt_n = tpv
            TPK[i].store(fx.Vector.from_elements([g, blk_m // tile_m + (blk_n // tile_n) * 65536, kb,
                                                  kt_n + ok * 65536], fx.Int32))

        def tp_get(i):
            v = TPK[i].load()
            w = [rocdl.readfirstlane(_T.i32, v[q]) for q in range_constexpr(4)]
            return (w[0], (w[1] % 65536) * tile_m, (w[1] // 65536) * tile_n, w[2], w[3] % 65536), w[3] // 65536

        def tp_shift(it_new):
            for i_ in range_constexpr(2):
                TPK[i_].store(TPK[i_ + 1].load())
            tpv, ok = tile_params(it_new)
            tp_put(2, tpv, ok)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA)
        base_ptr = arena.base_ptr

        def _cast(cls, p, align=16):
            return fx.recast_iter(fx.PointerType.get(cls.ir_type, p.address_space, align), p)

        def _view(ptr, cls, shape, stride, align=16):
            return fx.Tensor(fx.make_view(_cast(cls, ptr, align), fx.make_layout(shape, stride)))

        def _gview(base, off, shape, stride):
            return fx.Tensor(fx.make_view(fx.add_offset(base, off), fx.make_layout(shape, stride)))

        wave = rocdl.readfirstlane(_T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        zl = lane * (i32_n >> 30)  # opaque VGPR zero
        zs = i32_n >> 30  # opaque uniform zero: stage offsets must not fold into DS immediates (mxgemm E4)
        wmb = (wave // n_warp) * warp_m
        wnb = (wave % n_warp) * warp_n

        gA = _gview(_cast(a_cls, arg_a), fx.Int64(0), (ROWS_PW, WMMA_K), (i32_lda, 1))
        gB = _gview(_cast(b_cls, arg_b), fx.Int64(0), (ROWS_PW, WMMA_K), (i32_ldb, 1))
        atomA = fx.rocdl.make_tdm_atom(gA, [None, None], strides=[lda64, None], num_warps=1, pad_interval=HALF,
                                       pad_amount=PITCH - HALF)
        atomB = fx.rocdl.make_tdm_atom(gB, [None, None], strides=[ldb64, None], num_warps=1, pad_interval=HALF,
                                       pad_amount=PITCH - HALF)
        gSA8 = _cast(fx.Int8, arg_sa, 4)
        gSB8 = _cast(fx.Int8, arg_sb, 4)

        def stage_off(s):
            return (s // 2) * SLOT + (s % 2) * HALF

        def issue(s, tp, ks):
            g, blk_m, blk_n, kb, _ = tp
            so = stage_off(s) + wave * (ROWS_PW * PITCH)
            k = fx.Int64(kb) + fx.Int64(ks) * WMMA_K
            fx.copy(atomA, gA, _view(fx.add_offset(base_ptr, so), a_cls, (ROWS_PW, WMMA_K), (PITCH, 1)),
                    imm_offset=fx.Int64(blk_m + wave * ROWS_PW) * lda64 + k)
            fx.copy(atomB, gB, _view(fx.add_offset(base_ptr, so + B_OFF), b_cls, (ROWS_PW, WMMA_K), (PITCH, 1)),
                    imm_offset=fx.Int64(blk_n + wave * ROWS_PW) * ldb64 + k)

        def issue_planes(pset, tp):
            """vkq E3: thread tid copies A-plane row blk_m + tid and B-plane row blk_n + tid of the tile's window
            (dwords from c0 & ~3) into plane set pset (VMEM async, ASYNCcnt)"""
            g, blk_m, blk_n, kb, kt_n = tp
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
                    fx.recast_iter(fx.Int8, fx.add_offset(base_ptr, la + d * 4)).llvm_ptr, 0)
                rocdl.global_load_async_to_lds_b32(
                    fx.add_offset(gSB8, gb + d64).llvm_ptr,
                    fx.recast_iter(fx.Int8, fx.add_offset(base_ptr, lb + d * 4)).llvm_ptr, 0)

        mma_atom = fx.make_mma_atom(fx.rocdl.WMMAScale(WMMA_M, WMMA_N, WMMA_K, b_cls, a_cls, fx.Float32,
                                                       block_size=MX_BLOCK))
        lds_b128 = _make_lds_load(128)
        lds_b32 = _make_lds_load(32)
        lds_base = fx.Int32(fx.ptrtoint(base_ptr))
        c_frags = [fx.make_rmem_tensor(WMMA_N // 2, fx.Float32) for _ in range_constexpr(m_reps * n_reps)]
        gk_reg = fx.make_rmem_tensor(1, fx.Int32)
        gk_reg.fill(0)
        if const_expr(sumsq):
            sq_reg = fx.make_rmem_tensor(WMMA_N // 2, fx.Float32)
            sq_reg.fill(0.0)
        lane_a = (wmb + lane16) * PITCH + kgrp * 16
        lane_b = B_OFF + (wnb + lane16) * PITCH + kgrp * 16
        lane_wa = PL_OFF + (wmb + lane16) * SPB
        lane_wb = PL_OFF + PL_A + (wnb + lane16) * SPB

        def faddr(J):
            """fragment address VGPRs of global k-step J's stage"""
            st = lds_base + stage_off((J + zs) % 4)
            return dict(a=st + lane_a, b=st + lane_b)

        def _ld_frag(base, off, dst=None):
            v = [fx.Vector(lds_b128(base, off + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            reg = fx.make_rmem_tensor(16, fx.Int32) if dst is None else dst
            reg.store(v01.shuffle(v23, list(range(16))))
            return reg

        def frag_a(ad, j, dst=None):
            return _ld_frag(ad["a"], j * WMMA_M * PITCH, dst)

        def frag_b(ad, ni, dst=None):
            return _ld_frag(ad["b"], ni * WMMA_N * PITCH, dst)

        def waddr(pset, kx):
            """scale-word address VGPRs of plane byte kx in set pset + the v_perm selector (formed at a step's start,
            so no word load waits on a just-computed address)"""
            koff = (kx // 4) * 4 + pset * PSET
            return dict(wb=zl + lds_base + lane_wb + koff, wa=zl + lds_base + lane_wa + koff,
                        sel=(kx % 4) * 0x01010101)

        def load_words(wd, dst):
            """raw scale words (B rows of the wave, then A row groups) + the selector -> register set dst"""
            for ni in range_constexpr(n_reps):
                dst["sw"][ni].store(lds_b32(wd["wb"], ni * WMMA_N * SPB))
            for j in range_constexpr(m_reps):
                dst["sw"][n_reps + j].store(lds_b32(wd["wa"], j * WMMA_M * SPB))
            dst["sel"].store(fx.Vector.from_elements([wd["sel"]], fx.Int32))

        def _set():
            return dict(b=[fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(n_reps)],
                        a2=[fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(2)],
                        sw=[fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(n_reps + m_reps)],
                        sel=fx.make_rmem_tensor(1, fx.Int32))

        SETS = [_set(), _set()]

        def _wmmas(bfr, afr, sav, sbv, j, lo, hi):
            for ni in range_constexpr(lo, hi):
                idx = j * n_reps + ni
                fx.gemm(mma_atom, c_frags[idx], bfr[ni], afr, c_frags[idx], scale_a=sbv[ni], scale_b=sav)

        def bar():
            rocdl.sched_barrier(0)
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)
            rocdl.sched_barrier(0)

        def step(ks, src, dst, S):
            """k-step ks of the current tile (global J = gk + ks) from register set src; reads step J+1's fragments
            and scale words into dst."""
            tp, ntp, nntp, hn, hnn, pset, ksh, nksh, gk = S
            kt_n = tp[4]
            nkt_n = ntp[4]
            J = gk + ks
            rocdl.sched_barrier(0)
            ad = faddr(J)
            an = faddr(J + 1)
            lastk = (ks + 1 >= kt_n)
            pset_n = lastk.select(one - pset, pset)
            kx_n = lastk.select(nksh, ks + 1 + ksh)
            wdn = waddr(pset_n, kx_n)
            h = n_reps // 2
            if const_expr(not (abl & 4)):
                sel = src["sel"].load()[0]
                sw = [r.load()[0] for r in src["sw"]]
                sv = [fx.Int32(rocdl.perm_b32(w, w, sel)) for w in sw]
                sbv, sav = sv[:n_reps], sv[n_reps:]
                a = [src["a2"][0], src["a2"][1]] + [None] * (m_reps - 2)
            for j in range_constexpr(m_reps):
                rocdl.sched_barrier(0)
                if const_expr(j == SIG and not (abl & 16)):
                    # this wave's TDM ops of step J+1 complete: newer is step J+2 if it exists
                    e2 = (ks + 2 < kt_n).select(one, hn)
                    if e2 == zero:
                        tdm_ops.tensor_wait(0)
                    tdm_ops.tensor_wait(2)
                    if lastk:
                        rocdl.s_wait_asynccnt(0)  # the next tile's planes (this thread's rows)
                    rocdl.s_barrier_signal(-1)
                if const_expr(j == early and not (abl & 16)):
                    # every wave: step J+1 visible, done with step J-1's stage (at ks 0: the previous tile's C
                    # scratch)
                    rocdl.s_barrier_wait(-1)
                    rocdl.sched_barrier(0)
                    if const_expr(not (abl & 2)):
                        s3 = (J + 3 + zs) % 4
                        k3 = ks + 3
                        if k3 < kt_n:
                            issue(s3, tp, k3)
                        else:
                            k3n = k3 - kt_n
                            if k3n < nkt_n:
                                if hn > zero:
                                    issue(s3, ntp, k3n)
                            else:
                                if hnn > zero:
                                    issue(s3, nntp, k3n - nkt_n)
                    rocdl.sched_barrier(0)
                if const_expr(abl & 4):
                    continue
                _wmmas(src["b"], a[j], sav[j], sbv, j, 0, h)
                rocdl.sched_barrier(0)
                if const_expr(j + 2 < m_reps):
                    a[j + 2] = frag_a(ad, j + 2)
                else:
                    jj = j + 2 - m_reps
                    frag_a(an, jj, dst["a2"][jj])
                if const_expr(early <= j < early + n_reps):
                    frag_b(an, j - early, dst["b"][j - early])
                if const_expr(j == early):
                    load_words(wdn, dst)
                rocdl.sched_barrier(0)
                _wmmas(src["b"], a[j], sav[j], sbv, j, h, n_reps)

        gC8 = _cast(fx.Int8, _cast(out_cls, arg_c))
        c_ld = fx.Int64(i32_n)

        def store_c(tp, scr_stage):
            """C of the wave via its 16-row band of the free stage's A half-rows (mx8w store_c), sumsq in (mi, ni)
            order as vkp"""
            g, blk_m, blk_n, _, _ = tp
            if const_expr(abl & 8):  # timing ablation: no epilogue (tile summed -> one 16-byte store per lane)
                acc_v = c_frags[0].load()
                for cf in c_frags[1:]:
                    acc_v = acc_v + cf.load()
                fx.ptr_store(acc_v.to(out_cls).bitcast(fx.Int8),
                             gC8 + (fx.Int64(g) * fx.Int64(i32_m) * c_ld + fx.Int64(blk_m + wmb + lane16) * c_ld
                                    + fx.Int64(blk_n + wnb + kgrp * 8)) * 2)
                return
            scr = stage_off(scr_stage) + wave * (WMMA_M * PITCH)
            lrow = lane // 8
            lcol = lane % 8
            cb = fx.Int64(g) * fx.Int64(i32_m) * c_ld
            for j in range_constexpr(m_reps):
                row0 = blk_m + wmb + j * WMMA_M
                for ni in range_constexpr(n_reps):
                    hv = c_frags[j * n_reps + ni].load().to(out_cls)
                    if const_expr(sumsq):
                        hf = hv.to(fx.Float32)
                        sq_reg.store(sq_reg.load() + hf * hf)
                    if const_expr(not (abl & 1)):
                        fx.ptr_store(hv.bitcast(fx.Int8), base_ptr + (scr + lane16 * PITCH + (ni * 16 + kgrp * 8) * 2))
                if const_expr(not (abl & 1)):
                    vs = [fx.Vector(lds_b128(lds_base, scr + (q * 4 + lrow) * PITCH + lcol * 16))
                          for q in range_constexpr(4)]
                    for q in range_constexpr(4):
                        off = (cb + fx.Int64(row0 + q * 4 + lrow) * c_ld + fx.Int64(blk_n + wnb + lcol * 8)) * 2
                        fx.ptr_store(vs[q].bitcast(fx.Int8), gC8 + off)

        # ---- prologue: tile 0's planes, global k-steps 0..2, wait for step 0 + planes, its fragments and words
        tp0, _ok0 = tile_params(zero)
        tp1, hn0 = tile_params(one)
        tp2, hnn0 = tile_params(one + one)
        tp_put(0, tp0, one)
        tp_put(1, tp1, hn0)
        tp_put(2, tp2, hnn0)
        issue_planes(zero, tp0)
        if hn0 > zero:
            issue_planes(one, tp1)
        issue(zero, tp0, zero)
        issue(one, tp0, one)
        e02 = (tp0[4] > 2).select(one, hn0)
        if tp0[4] > 2:
            issue(one + one, tp0, one + one)
        else:
            if hn0 > zero:
                issue(one + one, tp1, zero)
        if e02 == zero:
            tdm_ops.tensor_wait(2)
        tdm_ops.tensor_wait(4)
        rocdl.s_wait_asynccnt(0)
        fx.barrier()
        ad0 = faddr(zero)
        for ni in range_constexpr(n_reps):
            frag_b(ad0, ni, SETS[0]["b"][ni])
        for j in range_constexpr(2):
            frag_a(ad0, j, SETS[0]["a2"][j])
        load_words(waddr(zero, (tp0[3] // 128) % 4), SETS[0])

        for it in range(my_cnt):
            tp, _ok = tp_get(0)
            ntp, hn = tp_get(1)
            nntp, hnn = tp_get(2)
            pset = it % 2
            ksh = (tp[3] // 128) % 4
            nksh = (ntp[3] // 128) % 4
            gk = gk_reg.load()[0]
            S = (tp, ntp, nntp, hn, hnn, pset, ksh, nksh, gk)
            for cf in c_frags:
                cf.fill(0.0)
            for kp in range(tp[4] // 2):
                step(kp * 2, SETS[0], SETS[1], S)
                step(kp * 2 + 1, SETS[1], SETS[0], S)
            # epilogue: every wave is done with the last step's stage -> C line scratch
            bar()
            store_c(tp, (gk + tp[4] - 1 + zs) % 4)
            gk_reg.store(gk_reg.load() + tp[4])
            # planes of tile it+2 -> this tile's set (all of its words are in registers; no WMMA in flight here, so
            # the address VALU of the async loads costs no WMMA drain); waited for at tile it+1's last k-step
            if hnn > zero:
                issue_planes(pset, nntp)
            tp_shift(it + 3)
        tdm_ops.tensor_wait(0)
        rocdl.s_wait_asynccnt(0)
        if const_expr(sumsq):
            sv_ = sq_reg.load()
            tot = sv_[0]
            for i_ in range_constexpr(1, WMMA_N // 2):
                tot = tot + sv_[i_]
            sq_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(_cast(fx.Float32, arg_sq, 4)))
            _bops.buffer_store(tot, sq_rsrc, bid * block + tid)
            if const_expr(clk):
                clk1 = _clock()
                if tid == zero:
                    v = fx.Vector.from_elements([clk0[0], clk0[1], clk1[0], clk1[1]], fx.Int64)
                    gS8 = _cast(fx.Int8, arg_sq, 4)
                    fx.ptr_store(v.bitcast(fx.Int8), gS8 + (fx.Int64(i32_nwg) * (block * 4) + fx.Int64(bid * 32)))

    kernel_vkq4(arg_c, arg_a, arg_b, arg_offs, arg_perm, arg_sa, arg_sb, arg_sq, i32_m, i32_n, i32_g, i32_lda,
                i32_ldb, i32_nwg).launch(grid=(i32_nwg, 1, 1), block=(block, 1, 1), stream=stream)


launch_vkq4_fp8_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}

_COMPILED: dict = {}


def _ptr(t):
    return flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)


def flyvkqblk_fp8(lhsP, rhsP, lhs_s, rhs_s, poffs_i32, OUT_M, OUT_N, out, beta=1.0, tile=(256, 256, 128),
                  m_warp=2, n_warp=4, group_m=4, kwin_max=4096, nwg=None, perm=None, prefetch_c=True,
                  sumsq_out=None, sched2=1, nbuf=2, early=2, sig=-1, abl=0, clk=0):
    """Drop-in for flyvkpblk_fp8(..., sched2=sched2) (same arguments, bit-identical out and sumsq_out).
    nbuf 4 (beta 0 only; beta 1 runs nbuf 2): the experimental vkq4 kernel; needs every window a multiple of 256
    tokens (not checked: would sync). early / sig / abl / clk: vkq4 options (see its docstring).
    out [G, OUT_M, OUT_N] bf16 (+)= per-expert lhsP[:, seg] @ rhsP[:, seg]^T; lhs_s/rhs_s compact E8M0
    [rows, P/128] uint8 (row stride P/128 a multiple of 4). beta 1 accumulates into out, 0 overwrites it."""
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    assert lhs_s.dtype == torch.uint8 and lhs_s.shape[1] * 128 == lhsP.shape[1]
    if not (lhs_s.is_contiguous() and rhs_s.is_contiguous() and lhs_s.shape[1] % 4 == 0
            and lhs_s.data_ptr() % 4 == 0 and rhs_s.data_ptr() % 4 == 0 and tuple(tile) == (256, 256, 128)
            and m_warp * n_warp * 32 == 256):
        # plane rows not dword-addressable (or another tiling): vkp computes the same bits
        from .grouped_gemm_fp8blk_vkp_flydsl_gfx1250 import flyvkpblk_fp8
        return flyvkpblk_fp8(lhsP, rhsP, lhs_s, rhs_s, poffs_i32, OUT_M, OUT_N, out, beta=beta, tile=tile,
                             m_warp=m_warp, n_warp=n_warp, group_m=group_m, kwin_max=kwin_max, nwg=nwg, perm=perm,
                             prefetch_c=prefetch_c, sumsq_out=sumsq_out, sched2=max(1, int(sched2)))
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
    pfc = 1 if (prefetch_c and bo) else 0
    nbuf = 2 if bo else int(nbuf)
    assert nbuf in (2, 4)
    sq = 0
    if sumsq_out is not None:
        assert sumsq_out.dtype == torch.float32 and sumsq_out.is_contiguous()
        assert sumsq_out.numel() >= nwg * m_warp * n_warp * 32, (sumsq_out.numel(), nwg)
        sq = 1
    if nbuf == 4:
        if clk:
            assert sq and sumsq_out.numel() >= nwg * (m_warp * n_warp * 32 + 8), "clk: sumsq buffer + 8 words per wg"
        args = (_ptr(out), _ptr(lhsP), _ptr(rhsP), _ptr(poffs_i32), _ptr(perm if lpt else poffs_i32), _ptr(lhs_s),
                _ptr(rhs_s), _ptr(sumsq_out if sq else out), OUT_M,
                fx.Stream(torch.cuda.current_stream(device=lhsP.device)), OUT_N, G, lhsP.stride(0), rhsP.stride(0), nwg,
                a_id, b_id, group_m, kwin_max, lpt, sq, int(early), int(sig), int(abl), int(clk))
        key = ("q4", a_id, b_id, group_m, kwin_max, lpt, sq, int(early), int(sig), int(abl), int(clk))
        c = _COMPILED.get(key)
        if c is None:
            _COMPILED[key] = flyc.compile(launch_vkq4_fp8_gfx1250, *args)
        else:
            c(*args)
        return out
    args = (_ptr(out), _ptr(lhsP), _ptr(rhsP), _ptr(poffs_i32), _ptr(perm if lpt else poffs_i32), _ptr(lhs_s),
            _ptr(rhs_s), _ptr(sumsq_out if sq else out), OUT_M, fx.Stream(torch.cuda.current_stream(device=lhsP.device)),
            OUT_N, G, lhsP.stride(0), rhsP.stride(0), nwg, tm, tn, tk, m_warp, n_warp, a_id, b_id, bo, group_m, kwin_max,
            lpt, pfc, sq, int(sched2))
    key = (tm, tn, tk, m_warp, n_warp, a_id, b_id, bo, group_m, kwin_max, lpt, pfc, sq, int(sched2))
    c = _COMPILED.get(key)
    if c is None:
        _COMPILED[key] = flyc.compile(launch_vkqblk_fp8_gfx1250, *args)
    else:
        c(*args)
    return out
