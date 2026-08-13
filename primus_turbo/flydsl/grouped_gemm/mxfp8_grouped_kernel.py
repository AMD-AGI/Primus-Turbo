###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL) (kernels/gemm/).
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""FlyDSL MXFP8 (per-1x32 E8M0 block-scaled) grouped GEMM for gfx950 (NT fwd/dgrad)."""

import math

import torch

# isort: off
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    MfmaScale16x16x128,
    S2RLoader,
    ScaleBComb,
    ScaleS2R,
    StoreCPerTensor,
    StoreCPerTensorCShuffle,
    _PRESHUF_KT,
    scale_opsel,
    _emit_lds_repack,
    _lane_tbl_count_le,
    _lane_tbl_get,
    _lane_tbl_load,
    _lane_tbl_scan,
    _readfirstlane_i32,
    _readlane_i32,
    _robust_ab_ratio,
    _robust_time,
    ceildiv,
    ceildiv_pow2,
    compute_global_swizzle,
    floordiv_pow2,
    make_fp8_buffer_tensor_rebased,
    make_row_band_resource,
    make_value_attrs,
    wait_barrier,
    xcd_remap_pid,
)
from primus_turbo.flydsl.grouped_gemm.gemm_fp8_grouped_kernel import (
    _WGRAD_RED_VEC,
    _WGRAD_RED_WPT,
    _grouped_block_mn,
    _load_go,
    _num_cus,
    _wgrad_block_mn,
    _wgrad_split_div,
    _wgrad_split_geom,
    _wgrad_split_policy,
    _wgrad_split_rcp_cfg,
    _wgrad_split_ws,
)

import flydsl.expr.buffer_ops as _buffer_ops

# isort: on


_BLOCK_N = 256
# `nt` aux bit for the NT C store: C is write-once (0.8-1.5 GB per launch against a 32 MB
# L2), so caching it normally turns the L2 over every few tiles and evicts the A/B band
# working set the L2 swizzle exists to keep resident. Marking those lines evict-first keeps
# the write combining (sc1, i.e. bypassing L2 outright, instead costs 2.5%).
_GNT_CSTORE_AUX = 2
_PRESHUF_BLK = 256

# E8M0 scale packing factor for the fwd/dgrad NT path (grouped-only opt-in; the shared
# gemm_helper defaults to unpacked). 4 is the hardware maximum: scale_opsel feeds the MFMA
# op_sel field, which selects one of the 4 bytes of a dword.
_GG_SCALE_PACK = 4


def run_eager_or_capture(entry, args, compiled_idx):
    """Eager runs the compiled object; capture runs the raw closure (compiled regresses under capture)."""
    if torch.cuda.is_current_stream_capturing():
        entry[0](*args)
    else:
        if entry[compiled_idx] is None:
            entry[compiled_idx] = flyc.compile(entry[0], *args)
        entry[compiled_idx](*args)


def _wgrad_mx_accum(mfma, a_frags, b_frags, acc_regs, sa, sb):
    """One quadrant's scaled MMA accumulate into the rmem accumulators (in place)."""
    c = [Vec(fx.memref_load_vec(r)) for r in acc_regs]
    c = mfma.call(a_frags, b_frags, c, sa, sb)
    for idx in range_constexpr(len(acc_regs)):
        fx.memref_store_vec(c[idx], acc_regs[idx])


def _wgrad_mx_body_4buf(
    k,
    kp0,
    BLOCK_K,
    A1off,
    B1off,
    a_g2s,
    b_g2s,
    a_s2r,
    b_s2r,
    sa_s2r,
    sb_s2r,
    mfma,
    a_cur0,
    a_cur1,
    b_cur0,
    b_cur1,
    a_next0,
    a_next1,
    b_next0,
    b_next1,
    acc00,
    acc01,
    acc10,
    acc11,
    sa_base0,
    sa_base1,
    sb_base0,
    NA,
    NB,
    opsel=0,
    quads=(2, 2),
):
    """One K-tile of the wgrad distance-2 4-buffer pipeline (scales loaded inline).

    ``opsel``: packed-dword byte immediate for this tile (caller computes it from the
    constexpr tail index; k is runtime so k%pack can't be an op_sel immediate).
    ``kp0``: this group's packed-scale base (scale k is group-local, see the kernel).
    ``quads``: live (M, N) output halves; 1 drops the padding half's MFMAs and its LDS
    reads while leaving every g2s, barrier and vmcnt wait in place. Dropping the padding
    half's g2s as well (as the NT kernel does) measured net-negative here: A1 is this
    pipeline's only distance-1 pool, so losing it also shortens the drain tail and the
    saved DMAs are paid back as an earlier wait."""
    qm, qn = quads
    k1 = k + 1
    k2 = k + 2
    mfma.opsel = opsel
    sa0 = sa_s2r.load(sa_base0, k, kp0)
    if qm == 2:
        sa1 = sa_s2r.load(sa_base1, k, kp0)
    sb_all = sb_s2r.load(sb_base0, k, kp0)
    sb0, sb1 = sb_all[0:2], sb_all[2:4]

    b0 = b_s2r.load(b_cur0)
    a0 = a_s2r.load(a_cur0)
    a_g2s.load(a_next1, A1off + k1 * BLOCK_K)
    rocdl.s_barrier()
    rocdl.s_setprio(1)
    _wgrad_mx_accum(mfma, a0, b0, acc00, sa0, sb0)
    rocdl.s_setprio(0)
    rocdl.s_barrier()
    if qn == 2:
        b1 = b_s2r.load(b_cur1)
    b_g2s.load(b_cur0, 0 + k2 * BLOCK_K)
    rocdl.s_barrier()
    if qn == 2:
        rocdl.s_setprio(1)
        _wgrad_mx_accum(mfma, a0, b1, acc01, sa0, sb1)
        rocdl.s_setprio(0)
    rocdl.s_barrier()
    if qm == 2:
        a1 = a_s2r.load(a_cur1)
    a_g2s.load(a_cur0, 0 + k2 * BLOCK_K)
    rocdl.s_barrier()
    if qm == 2:
        rocdl.s_setprio(1)
        _wgrad_mx_accum(mfma, a1, b0, acc10, sa1, sb0)
        rocdl.s_setprio(0)
    rocdl.s_barrier()
    b_g2s.load(b_cur1, B1off + k2 * BLOCK_K)
    wait_barrier(2 * NA + NB)
    if qm == 2 and qn == 2:
        rocdl.s_setprio(1)
        _wgrad_mx_accum(mfma, a1, b1, acc11, sa1, sb1)
        rocdl.s_setprio(0)
    rocdl.s_barrier()


def _wgrad_ssa_chunk(
    base_k,
    chunk,
    kp0,
    BLOCK_K,
    A1off,
    B1off,
    a_g2s,
    b_g2s,
    a_s2r,
    b_s2r,
    sa_s2r,
    sb_s2r,
    mfma,
    a_cur0,
    a_cur1,
    b_cur0,
    b_cur1,
    a_next0,
    a_next1,
    b_next0,
    b_next1,
    acc00,
    acc01,
    acc10,
    acc11,
    sa_base0,
    sa_base1,
    sb_base0,
    N_ACCUMS,
    N_LDS_STEPS_A,
    N_LDS_STEPS_B,
    sc_pf,
    pack=1,
    quads=(2, 2),
):
    """One constexpr chunk of the wgrad K-loop with SSA-register accumulators.

    ``quads``: live (M, N) output halves; see ``_wgrad_mx_body_4buf``. ``sc_pf``: rmem
    carriers (A half 0, B combined) that hand this chunk's first packed scale dword across
    the chunk boundary -- the dynamic-loop equivalent of the accumulators -- so it is not the
    one latency-exposed VMEM load left at chunk top. A half 1 stays an ordinary chunk-top
    load (feeds phase C, 4 barriers later; carrying it too would cost 4 spill dwords)."""
    qm, qn = quads
    c00 = [Vec(fx.memref_load_vec(r)) for r in acc00]
    if qn == 2:
        c01 = [Vec(fx.memref_load_vec(r)) for r in acc01]
    if qm == 2:
        c10 = [Vec(fx.memref_load_vec(r)) for r in acc10]
    if qm == 2 and qn == 2:
        c11 = [Vec(fx.memref_load_vec(r)) for r in acc11]
    sa0 = sa_s2r.split(Vec(fx.memref_load_vec(sc_pf[0])))
    if qm == 2:
        sa1 = sa_s2r.load(sa_base1, base_k, kp0)
    sb_all = sb_s2r.split(Vec(fx.memref_load_vec(sc_pf[1])))
    sb0, sb1 = sb_all[0:2], sb_all[2:4]
    for _j in range_constexpr(chunk):
        k = base_k + _j
        k1 = k + 1
        k2 = k + 2
        # pack>1: k is group-local and runtime (base_k=_c*chunk) but base_k%pack==0 and the
        # group packs from its own start -> the byte within the packed dword is the constexpr
        # _j%pack (op_sel immediate), whatever the group's contraction offset is.
        # The scale dword is shared by `pack` consecutive K-iters; load it only once
        # per pack-group and reuse (cuts scale-load vmem instructions PACK-fold). The
        # last pack-group prefetches ACROSS the chunk boundary into the rmem carriers, so
        # the next chunk starts with its first dword already in flight.
        mfma.opsel = scale_opsel(_j, pack)
        _reload = (_j + 1) % pack == 0
        _cross = (_j + 1) == chunk
        if _reload:
            _va0 = sa_s2r.load_vec(sa_base0, k1, kp0)
            if _cross:
                fx.memref_store_vec(_va0, sc_pf[0])
            else:
                sa0n = sa_s2r.split(_va0)
        b0 = b_s2r.load(b_cur0)
        a0 = a_s2r.load(a_cur0)
        a_g2s.load(a_next1, A1off + k1 * BLOCK_K)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c00 = mfma.call(a0, b0, c00, sa0, sb0)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        if qn == 2:
            b1 = b_s2r.load(b_cur1)
        b_g2s.load(b_cur0, 0 + k2 * BLOCK_K)
        if _reload:
            _vb = sb_s2r.load_vec(sb_base0, k1, kp0)
            if _cross:
                fx.memref_store_vec(_vb, sc_pf[1])
            else:
                sb_alln = sb_s2r.split(_vb)
        rocdl.s_barrier()
        if qn == 2:
            rocdl.s_setprio(1)
            c01 = mfma.call(a0, b1, c01, sa0, sb1)
            rocdl.s_setprio(0)
        rocdl.s_barrier()
        if qm == 2:
            a1 = a_s2r.load(a_cur1)
        a_g2s.load(a_cur0, 0 + k2 * BLOCK_K)
        if _reload and not _cross and qm == 2:
            sa1n = sa_s2r.load(sa_base1, k1, kp0)
        rocdl.s_barrier()
        if qm == 2:
            rocdl.s_setprio(1)
            c10 = mfma.call(a1, b0, c10, sa1, sb0)
            rocdl.s_setprio(0)
        rocdl.s_barrier()
        b_g2s.load(b_cur1, B1off + k2 * BLOCK_K)
        wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
        if qm == 2 and qn == 2:
            rocdl.s_setprio(1)
            c11 = mfma.call(a1, b1, c11, sa1, sb1)
            rocdl.s_setprio(0)
        rocdl.s_barrier()
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1
        if _reload and not _cross:
            sa0 = sa0n
            if qm == 2:
                sa1 = sa1n
            sb_all = sb_alln
            sb0, sb1 = sb_all[0:2], sb_all[2:4]
    for _i in range_constexpr(N_ACCUMS):
        fx.memref_store_vec(c00[_i], acc00[_i])
        if qn == 2:
            fx.memref_store_vec(c01[_i], acc01[_i])
        if qm == 2:
            fx.memref_store_vec(c10[_i], acc10[_i])
        if qm == 2 and qn == 2:
            fx.memref_store_vec(c11[_i], acc11[_i])
    return a_cur0, a_cur1, b_cur0, b_cur1, a_next0, a_next1, b_next0, b_next1


def _gnt_pair_n(N: int, BLOCK_N: int = _BLOCK_N) -> bool:
    """Adjacent-column B pairing gate for the NT path. Pairing writes a lane's two
    n-fragments as one dword, so the stored column span must lie inside N: true iff the
    last N-block is full or exactly the b0 half (the b1-half tile the kernel already
    skips). Shared by the kernel, its B feed and the B-scale preshuffle so the three
    cannot drift apart."""
    return N % BLOCK_N in (0, BLOCK_N // 2)


def _build_grouped_preshuffle_kernel(
    K128: int, G: int, N: int, KT: int = _PRESHUF_KT, BLK: int = 256, b_pair: bool = False
):
    """Fused per-group A (layout 1) + B (B-comb layout 3) E8M0 scale preshuffle.

    Returns ``(kern, n_kt, b_blocks_pg)``. Per-group into 64-row slabs (go_pre) so a
    32-aligned group data base needs no 64-alignment from the quantizer.
    """
    TILE = 64 * KT  # noqa: F841 (mirrors build_preshuffle_ab_kernel; sized in Smem)
    n_kt = ceildiv(K128, KT)
    K128p = ceildiv(K128, _GG_SCALE_PACK)  # packed K-groups (PACK scales / dword)
    _a_slab_i32 = K128p * 256  # per-slab a_sp span (i32 elems, packed)
    b_ngrp_pg = ((N + 255) // 256) * 4
    b_blocks_pg = b_ngrp_pg * n_kt
    _b_slab_i32 = b_ngrp_pg * K128p * 256  # per-group b_sp slab (i32 elems, packed)

    @fx.struct
    class Smem:
        tile: fx.Array[fx.Int32, 64 * KT, 16]

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def kern(
        a_raw: fx.Tensor,
        b_raw: fx.Tensor,
        a_sp: fx.Tensor,
        b_sp: fx.Tensor,
        group_offs: fx.Tensor,  # padded read offsets (int32 view of int64 [G+1])
        m_pad: fx.Int32,
        a_blocks: fx.Int32,
        a_ngrp: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        tile = fx.SharedAllocator().allocate(Smem).peek().tile
        rin_a = _buffer_ops.create_buffer_resource(a_raw, max_size=False, num_records_bytes=m_pad * K128 * 4)
        rin_b = _buffer_ops.create_buffer_resource(b_raw, max_size=False, num_records_bytes=G * N * K128 * 4)
        rout_a = _buffer_ops.create_buffer_resource(
            a_sp, max_size=False, num_records_bytes=a_ngrp * K128p * 256 * 4
        )
        rout_b = _buffer_ops.create_buffer_resource(
            b_sp, max_size=False, num_records_bytes=G * _b_slab_i32 * 4
        )
        go_pad = fx.rocdl.make_buffer_tensor(group_offs, max_size=False, num_records_bytes=(G + 1) * 8)
        go_pad_div = fx.logical_divide(go_pad, fx.make_layout(1, 1))
        if bid < a_blocks:
            slab_g = bid // n_kt
            k0 = (bid % n_kt) * KT
            # O(G) scan: locate the group owning global slab ``slab_g`` and capture
            # (go_pre[g], go_pad[g], M_g_pad, within-group slab j).
            acc = fx.Int32(0)
            pre_g = fx.Int32(0)
            base_row = fx.Int32(0)
            mgpad = fx.Int32(0)
            jj = fx.Int32(0)
            valid = fx.Int32(0)
            pp = _load_go(go_pad_div, 0)
            for g in range_constexpr(G):
                pn = _load_go(go_pad_div, g + 1)
                ns = ceildiv(pn - pp, 64)
                inq = (slab_g >= acc) & (slab_g < acc + ns)
                pre_g = arith.select(inq, acc, pre_g)
                base_row = arith.select(inq, pp, base_row)
                mgpad = arith.select(inq, pn - pp, mgpad)
                jj = arith.select(inq, slab_g - acc, jj)
                valid = arith.select(inq, fx.Int32(1), valid)
                acc = acc + ns
                pp = pn
            if valid > fx.Int32(0):
                _emit_lds_repack(
                    True,
                    jj,
                    k0,
                    tile,
                    rin_a,
                    rout_a,
                    mgpad,
                    K128,
                    KT,
                    tid,
                    BLK,
                    rd_base=base_row * K128,
                    wr_base=pre_g * _a_slab_i32,
                    pack=_GG_SCALE_PACK,
                )
        if bid >= a_blocks:
            bb = bid - a_blocks
            g = bb // b_blocks_pg
            loc = bb % b_blocks_pg
            grp = loc // n_kt
            k0 = (loc % n_kt) * KT
            _emit_lds_repack(
                False,
                grp,
                k0,
                tile,
                rin_b,
                rout_b,
                fx.Int32(N),
                K128,
                KT,
                tid,
                BLK,
                rd_base=g * (N * K128),
                wr_base=g * _b_slab_i32,
                pack=_GG_SCALE_PACK,
                b_pair=b_pair,
            )

    return kern, n_kt, b_blocks_pg


def _build_grouped_mxfp8_nt_kernel(
    K: int,
    G: int,
    N: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    group_m: int = 4,
    group_n: int = 0,
    num_xcd: int = 8,
    waves_per_eu: int = 2,
    cbsz: int = 0,
    blgp: int = 0,
    out_fp16: bool = False,
    persistent: bool = False,
    store_cshuffle: bool = False,  # vectorized 128b CShuffle store_c; NET-NEGATIVE for mxfp8
    # (scale is folded -> the scalar store is already a plain write, so CShuffle's LDS
    #  staging only adds cost; mxfp8 NT is mainloop-bound, not store-bound). Kept for probes.
    cstore_aux=None,  # aux immediate for the C store (None = _GNT_CSTORE_AUX)
):
    """Grouped MXFP8 NT (out = a @ b^T) with grouped per-tile addressing."""
    BLOCK_K = 128
    assert BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0 and BLOCK_M >= 128 and BLOCK_N >= 256
    assert K % BLOCK_K == 0 and G >= 1
    K_ITERS = K // BLOCK_K
    assert K_ITERS >= 2

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K
    SA_TILES = N_TILES_A
    # Boundary N-block skip: when the last N-block holds <= LDS_BLOCK_N valid columns its b1
    # half is all padding, so that tile's acc01/acc11 quadrants (half its MFMAs) and the b1
    # LDS reads are dead work -- the tile runs a b0-only body. Compile-time gated on the
    # static N (the mirror of the runtime c_n); a 256-aligned N emits only the full body.
    _HALF_N = (N % BLOCK_N != 0) and (N % BLOCK_N <= LDS_BLOCK_N)
    _N_BLOCKS = ceildiv(N, BLOCK_N)
    _LAST_BN = _N_BLOCKS - 1
    # Column span actually stored: BLOCK_N for the full body, LDS_BLOCK_N for the b0-only
    # one, and only the last N-block can overhang. Both spans stay inside N iff the residue
    # is 0 or exactly LDS_BLOCK_N, in which case the epilogue's per-store OOB select is dead
    # work. Unlike the wgrad kernel, the b0-only body drops its c01/c11 stores outright, so
    # nothing here relies on the column clamp.
    _COL_SAFE = _gnt_pair_n(N, BLOCK_N)
    # Pair-major B: renumber the n-operand's LDS rows so a lane's two mfma fragments are
    # ADJACENT output columns (see pair_major_row), letting the epilogue store both as one
    # dword and pack the two f32->bf16 converts into one v_cvt_pk_bf16_f32.
    _PAIR_N = 2 if (_COL_SAFE and N_TILES_B == 2) else 0
    # CShuffle epilogue staging (store_cshuffle only): one 16-row sub-tile per wave.
    _cshuf_ty = fx.Float16 if out_fp16 else fx.BFloat16
    _cshuf_n = 8 * 16 * (N_TILES_B * 16)
    _cstore_aux = _GNT_CSTORE_AUX if cstore_aux is None else int(cstore_aux)

    _ss_anns = {
        "A_lds_cur_0": fx.Array[fx.Float8E4M3FN, a_lds_size, 16],
        "A_lds_cur_1": fx.Array[fx.Float8E4M3FN, a_lds_size, 16],
        "A_lds_next_0": fx.Array[fx.Float8E4M3FN, a_lds_size, 16],
        "A_lds_next_1": fx.Array[fx.Float8E4M3FN, a_lds_size, 16],
        "B_lds_cur_0": fx.Array[fx.Float8E4M3FN, b_lds_size, 16],
        "B_lds_cur_1": fx.Array[fx.Float8E4M3FN, b_lds_size, 16],
        "B_lds_next_0": fx.Array[fx.Float8E4M3FN, b_lds_size, 16],
        "B_lds_next_1": fx.Array[fx.Float8E4M3FN, b_lds_size, 16],
    }
    if store_cshuffle:  # only allocate the staging region when the CShuffle epilogue is used
        _ss_anns["C_lds_shuffle"] = fx.Array[_cshuf_ty, _cshuf_n, 16]
    SharedStorage = fx.struct(type("SharedStorage", (), {"__annotations__": _ss_anns}))

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_mxfp8_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        group_offs: fx.Tensor,  # padded read offsets (int32 view of int64 [G+1])
        group_offs_out: fx.Tensor,  # tight write offsets (int32 view of int64 [G+1])
        c_scale_rows: fx.Int32,  # A-scale slab rows = a_ngrp_ub*64 -> ScaleS2R buffer sizing
        c_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        # c_n mirrors the static N at every entry, so the tile decode can divide by the
        # compile-time block count. gfx9 has no scalar integer divide, so a runtime divisor
        # would expand every quotient below into v_rcp_iflag_f32 + readfirstlane.
        n_blocks = _N_BLOCKS

        # One on-device lane-parallel group scan (no host read). Lane g owns group g, so
        # both prefixes -- _tcs[g] = tiles owned by groups < g (TIGHT sizes) and _sas[g]
        # = 64-row A-scale slabs before group g (PADDED sizes) -- come out of one wave
        # inclusive scan each and stay resident in lanes. That removes the whole serial
        # carry: no 2*(G+1) live scalars (they overflow the SGPR file), no LDS prefix
        # table, no publish barrier, and the per-tile decode below is a ballot instead of
        # a G-wide compare tree. Lanes past group G-1 read out of range (0) and add
        # nothing, so their scan value is total_tiles and they never win that compare.
        # Both round-ups go through ceildiv_pow2: signed floordiv would put a
        # divide/remainder/sign-select chain on every lane.
        lane_g = fx.thread_idx.x % 64
        go_out_rs = _buffer_ops.create_buffer_resource(
            group_offs_out, max_size=False, num_records_bytes=(G + 1) * 8
        )
        go_pad_rs = _buffer_ops.create_buffer_resource(
            group_offs, max_size=False, num_records_bytes=(G + 1) * 8
        )
        # int32 view of the int64 [G+1] tables: entry g is at i32 element 2*g (token
        # offsets are < 2^31, so the high word is 0).
        _gout0 = _lane_tbl_load(go_out_rs, lane_g, G + 1, stride=2)
        _gout1 = _lane_tbl_load(go_out_rs, lane_g, G + 1, stride=2, first=1)
        _gpad0 = _lane_tbl_load(go_pad_rs, lane_g, G + 1, stride=2)
        _gpad1 = _lane_tbl_load(go_pad_rs, lane_g, G + 1, stride=2, first=1)
        _own = [lane_g + fx.Int32(64 * c) < fx.Int32(G) for c in range_constexpr(len(_gout0))]
        _nt = [
            arith.select(_own[c], ceildiv_pow2(_gout1[c] - _gout0[c], BLOCK_M) * n_blocks, fx.Int32(0))
            for c in range_constexpr(len(_gout0))
        ]
        _ns = [
            arith.select(_own[c], ceildiv_pow2(_gpad1[c] - _gpad0[c], 64), fx.Int32(0))
            for c in range_constexpr(len(_gout0))
        ]
        _tcs_end = _lane_tbl_scan(_nt)  # entry g = tiles owned by groups <= g
        _sas_end = _lane_tbl_scan(_ns)
        _tcs = [_tcs_end[c] - _nt[c] for c in range_constexpr(len(_nt))]
        _sas = [_sas_end[c] - _ns[c] for c in range_constexpr(len(_ns))]
        total_tiles = _readlane_i32(_tcs_end[-1], 63)
        m_total_pad = _lane_tbl_get(_gpad0, G)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pid = fx.block_idx.x
        nsms = fx.grid_dim.x

        if const_expr(not persistent):
            _llvm.inline_asm(
                None,
                [pid.ir_value(), arith._to_raw(total_tiles)],
                "s_cmp_lt_u32 $0, $1\n\ts_cbranch_scc1 1f\n\ts_endpgm\n\t1:",
                "s,s,~{scc},~{memory}",
                has_side_effects=True,
            )

        def _do_tile(t):
            tt = xcd_remap_pid(t, total_tiles, num_xcd)
            # tt -> owning group: the lane-resident tile prefix is monotone, so the
            # number of its entries at or below tt is the group index (one ballot + one
            # s_bcnt1). Its tile base, A-scale slab base and row bounds are one
            # v_readlane each -- no LDS lookup and no per-tile group-offset load.
            group_idx = _lane_tbl_count_le(_tcs_end, tt)
            tile_start = _lane_tbl_get(_tcs, group_idx)
            sa_pre = _lane_tbl_get(_sas, group_idx)

            m_start = _lane_tbl_get(_gout0, group_idx)  # tight C base
            m_end = _lane_tbl_get(_gout1, group_idx)  # tight C end (store bound)
            m_start_pad = _lane_tbl_get(_gpad0, group_idx)  # padded A DATA base (32-aligned)
            local = tt - tile_start
            local_block_m, block_n = _grouped_block_mn(
                local, m_start, m_end, n_blocks, BLOCK_M, group_m, group_n
            )

            lane_id = fx.thread_idx.x % 64
            wave_id = fx.thread_idx.x // 64
            wave_m = wave_id // 4
            wave_n = wave_id % 4

            m_row_c = m_start + local_block_m * BLOCK_M
            m_row_a = m_start_pad + local_block_m * BLOCK_M

            cn_i = arith.index_cast(T.index, c_n)
            a_base = arith.index_cast(T.index, m_row_a) * arith.index(K)
            b_base = (
                arith.index_cast(T.index, group_idx) * cn_i + arith.index_cast(T.index, block_n * BLOCK_N)
            ) * arith.index(K)
            a_nrec = (
                arith.index_cast(T.index, m_total_pad) - arith.index_cast(T.index, m_row_a)
            ) * arith.index(K)
            b_nrec = arith.index(G) * cn_i * arith.index(K) - b_base
            A0_gl_offset = 0
            A1_gl_offset = LDS_BLOCK_M * K
            B0_gl_offset = 0
            B1_gl_offset = LDS_BLOCK_N * K

            gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
            gB = make_fp8_buffer_tensor_rebased(B_T, F8_IR_t, b_base, b_nrec)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
            gl_off_b = compute_global_swizzle(
                lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False, row_pair=bool(_PAIR_N)
            )
            mfma = MfmaScale16x16x128(N_TILES_A, N_TILES_B, cbsz=cbsz, blgp=blgp)

            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            a_s2r = S2RLoader(wave_m, N_TILES_A)
            b_s2r = S2RLoader(wave_n, N_TILES_B)

            sa_s2r = ScaleS2R(A_scale, c_scale_rows, K, SA_TILES, pack=_GG_SCALE_PACK)
            sb_s2r = ScaleBComb(B_scale, c_n, K, n_slabs=G, pack=_GG_SCALE_PACK)
            # Scale folded into the accumulator by the scaled MMA -> A_scale/B_scale=None
            # (plain store). CShuffle vectorizes the 128b store + cstore_aux keeps write-once
            # C out of L2 (mirrors the per-tensor NT epilogue).
            if const_expr(store_cshuffle):
                store_c = StoreCPerTensorCShuffle(
                    None,
                    None,
                    C,
                    m_end,
                    c_n,
                    mfma.idx,
                    N_TILES_A,
                    N_TILES_B,
                    _out_ty,
                    lds.C_lds_shuffle,
                    wave_id,
                    store_aux=_cstore_aux,
                )
            else:
                store_c = StoreCPerTensor(
                    None,
                    None,
                    C,
                    m_end,
                    c_n,
                    mfma.idx,
                    N_TILES_A,
                    N_TILES_B,
                    _out_ty,
                    col_safe=_COL_SAFE,
                    pack_cols=_PAIR_N,
                    store_aux=_cstore_aux,
                )
            wave_m_offset = wave_m * (N_TILES_A * 16)
            wave_n_offset = wave_n * (N_TILES_B * 16)
            # A-scale slab base = go_pre[g]*64 (per-group 64-aligned), NOT the data row
            # base m_row_a (32-aligned): the two are decoupled (see go_pre scan above).
            sa_row_base = sa_pre * fx.Int32(64) + local_block_m * BLOCK_M
            sa_base0 = sa_row_base + wave_m_offset
            sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
            sb_base0 = block_n * BLOCK_N + wave_n_offset

            # nq = live N-quadrants: 2 = full tile, 1 = b0 only (b1 half is all padding, so
            # its g2s is dropped too and B costs one step per K-iter instead of two).
            # All four LDS pools run at prefetch distance 2 (K-iter k stages k+2), so a g2s
            # has a whole K-iter to land instead of A1's old "inside its own iteration".
            # Each drain leaves one K-iter of issues in flight, minus the first step group:
            # vmcnt retires out of order, so a tail longer than that shows up as a
            # repeat-to-repeat race even though issue order alone would allow it (pitfalls/04).
            _NB_DRAIN = 2 * N_LDS_STEPS_A + N_LDS_STEPS_B
            _NB_DRAIN_HALF = 2 * N_LDS_STEPS_A

            def _body(nq):
                _full = nq == 2
                _nd = _NB_DRAIN if _full else _NB_DRAIN_HALF

                a_cur0 = lds.A_lds_cur_0
                a_cur1 = lds.A_lds_cur_1
                a_next0 = lds.A_lds_next_0
                a_next1 = lds.A_lds_next_1
                b_cur0 = lds.B_lds_cur_0
                b_cur1 = lds.B_lds_cur_1
                b_next0 = lds.B_lds_next_0
                b_next1 = lds.B_lds_next_1

                c00_frag = [mfma.zero_value] * N_ACCUMS
                c10_frag = [mfma.zero_value] * N_ACCUMS
                if const_expr(_full):
                    c01_frag = [mfma.zero_value] * N_ACCUMS
                    c11_frag = [mfma.zero_value] * N_ACCUMS

                b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
                a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
                if const_expr(_full):
                    b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
                a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)
                if const_expr(persistent):
                    rocdl.s_barrier()
                else:
                    if wave_m == 1:
                        rocdl.s_barrier()
                wait_barrier(_nd)
                b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
                a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
                if const_expr(_full):
                    b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)
                a_g2s.load(a_next1, A1_gl_offset + 1 * BLOCK_K)
                wait_barrier(_nd)
                sa0 = sa_s2r.load(sa_base0, 0)
                sa1 = sa_s2r.load(sa_base1, 0)
                sb_all = sb_s2r.load(sb_base0, 0, slab=group_idx)
                sb0, sb1 = sb_all[0:2], sb_all[2:4]

                for k in range_constexpr(K_ITERS - 2):
                    mfma.opsel = scale_opsel(k, _GG_SCALE_PACK)  # select packed scale byte for K-iter k
                    # One packed dword feeds _GG_SCALE_PACK consecutive K-iters; reload it only
                    # at the pack boundary (op_sel picks the byte). Cuts scale-load vmem PACK-fold.
                    # The boundary always lands on k+1==K_ITERS-2's dword, so the epilogue below
                    # keeps reading the correct retained dword via op_sel.
                    _rl = (k + 1) % _GG_SCALE_PACK == 0
                    if const_expr(_rl):
                        sa0n = sa_s2r.load(sa_base0, k + 1)
                    b0_frag = b_s2r.load(b_cur0)
                    a0_frag = a_s2r.load(a_cur0)
                    if const_expr(_full):
                        b1_frag = b_s2r.load(b_cur1)
                    rocdl.s_barrier()
                    rocdl.s_setprio(1)
                    c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
                    rocdl.s_setprio(0)
                    rocdl.s_barrier()
                    b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K)
                    if const_expr(_rl):
                        sb_alln = sb_s2r.load(sb_base0, k + 1, slab=group_idx)
                    # The c01/c11 rendezvous pairs only align the waves around an MFMA
                    # phase; in the b0-only body that phase is empty, so the pair degrades
                    # to two back-to-back s_barrier bracketing nothing. Every wave of the
                    # workgroup picks the same body (the predicate is wave-uniform), and
                    # the LDS WAR guards are the barriers that follow c00 and c10, so
                    # dropping these leaves the buffer protection intact.
                    if const_expr(_full):
                        rocdl.s_barrier()
                        rocdl.s_setprio(1)
                        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
                        rocdl.s_setprio(0)
                        rocdl.s_barrier()
                    a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
                    if const_expr(_rl):
                        sa1n = sa_s2r.load(sa_base1, k + 1)
                    a1_frag = a_s2r.load(a_cur1)
                    rocdl.s_barrier()
                    rocdl.s_setprio(1)
                    c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
                    rocdl.s_setprio(0)
                    rocdl.s_barrier()
                    # Both b1 and a1 LDS halves are read by now, so their k+2 stage issues
                    # here; the drain below only has to retire the k-1 tail.
                    if const_expr(_full):
                        b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K)
                    a_g2s.load(a_cur1, A1_gl_offset + (k + 2) * BLOCK_K)
                    wait_barrier(_nd)
                    if const_expr(_full):
                        rocdl.s_setprio(1)
                        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
                        rocdl.s_setprio(0)
                        rocdl.s_barrier()
                    a_cur0, a_next0 = a_next0, a_cur0
                    a_cur1, a_next1 = a_next1, a_cur1
                    b_cur0, b_next0 = b_next0, b_cur0
                    b_cur1, b_next1 = b_next1, b_cur1
                    if const_expr(_rl):
                        sa0, sa1 = sa0n, sa1n
                        sb_all = sb_alln
                        sb0, sb1 = sb_all[0:2], sb_all[2:4]

                # Step K_ITERS-2 (prefetch last iter's scales). Distance 2 means every stage
                # is already issued by now, so both tail steps only read.
                mfma.opsel = scale_opsel(K_ITERS - 2, _GG_SCALE_PACK)
                sa0n = sa_s2r.load(sa_base0, K_ITERS - 1)
                sa1n = sa_s2r.load(sa_base1, K_ITERS - 1)
                sb_alln = sb_s2r.load(sb_base0, K_ITERS - 1, slab=group_idx)
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                if const_expr(_full):
                    b1_frag = b_s2r.load(b_cur1)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                if const_expr(_full):
                    rocdl.s_barrier()
                    rocdl.s_setprio(1)
                    c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
                    rocdl.s_setprio(0)
                    rocdl.s_barrier()
                a1_frag = a_s2r.load(a_cur1)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                if const_expr(_full):
                    rocdl.s_setprio(1)
                    c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
                    rocdl.s_setprio(0)
                    rocdl.s_barrier()
                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1
                sa0, sa1 = sa0n, sa1n
                sb_all = sb_alln
                sb0, sb1 = sb_all[0:2], sb_all[2:4]

                # Step K_ITERS-1: last stage was issued one K-iter ago, so drain, then read.
                mfma.opsel = scale_opsel(K_ITERS - 1, _GG_SCALE_PACK)
                wait_barrier(0)
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                if const_expr(_full):
                    b1_frag = b_s2r.load(b_cur1)
                rocdl.s_setprio(1)
                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                if const_expr(_full):
                    rocdl.s_barrier()
                    rocdl.s_setprio(1)
                    c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1)
                    rocdl.s_setprio(0)
                    rocdl.s_barrier()

                base_row = m_row_c + wave_m_offset
                base_col = block_n * BLOCK_N + wave_n_offset
                a1_frag = a_s2r.load(a_cur1)
                rocdl.s_barrier()
                rocdl.s_setprio(1)
                c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0)
                if const_expr(_full):
                    c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1)
                rocdl.s_setprio(0)
                rocdl.s_barrier()
                # Both N-quadrants of a row band go through one store call so the two
                # quadrants and the two fragments of a row share one byte address.
                _q0 = ((LDS_BLOCK_N, c01_frag),) if const_expr(_full) else ()
                _q1 = ((LDS_BLOCK_N, c11_frag),) if const_expr(_full) else ()
                store_c.store(c00_frag, base_row + 0, base_col + 0, _q0)
                store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0, _q1)

            if const_expr(_HALF_N):
                # Scalar (wave-uniform) predicate: every wave must take the same path,
                # the bodies contain s_barrier.
                if _readfirstlane_i32(block_n) == fx.Int32(_LAST_BN):
                    _body(1)
                else:
                    _body(2)
            else:
                _body(2)

        if const_expr(persistent):
            for t in range(pid, total_tiles, nsms):
                _do_tile(t)
        else:
            _do_tile(pid)

    return kernel_grouped_mxfp8_nt, BLOCK_M, BLOCK_N, waves_per_eu


# ── Host wrapper ─────────────────────────────────────────────────────────────

_GNT_FUSED_CACHE: dict = {}  # (K, G, N, bm, gm, xcd, gn, cbsz, blgp, out_fp16, persist) -> launch
_GNT_WS_CACHE: dict = {}  # (M_pad, N, K128, G, device, stream) -> (a_sp, b_sp, a_blocks, a_ngrp)
_GNT_AT_CACHE: dict = {}  # (M_pad, N, K, G, cbsz, blgp, out_fp16, persist) -> [raw, compiled]
_GNT_CFG_CACHE: dict = {}  # cfg_key (NO M_pad) -> (bm, gm, xcd, gn) chosen by autotune

# fwd/dgrad NT autotune. The launch is M-generic (M is a runtime arg), so the config race
# keys on the static shape plus the band-width regime (cfg_key, no M_pad) and is reused for
# every M inside that regime.
_GNT_NT_DEFAULT_CFG = (256, 4, 4, 0)  # (BLOCK_M, GROUP_M, num_xcd, group_n); cand[0] = base ref
# One XCD's private L2 slice; MI355X has 8 of them, so a band is only re-read out of L2
# while its A footprint fits in one slice.
_L2_SLICE_BYTES = 4 << 20
# Tokens/group from which the wide M band wins whatever K is. The band width trades B
# traffic (B is streamed once per band, so it scales as 1/gm) against A residency (a band's
# A footprint is gm*BLOCK_M*K bytes). With a handful of M blocks per group a group spans one
# or two bands, the B term is already paid off and the residency term picks the width; at
# training-scale tokens/group a group spans tens of bands and the B term takes over.
_GNT_WIDE_BAND_PM = 16384

# (tokens/group, skewed) points the race times on. The swizzle is invariant in neither M nor
# the token distribution: a single midpoint mis-picks a cfg that wins there but loses at the
# range ends, and a balanced-only race adopts cfgs that win on even groups yet cost several %
# on the top-heavy routing real MoE produces. Two points spread both axes.
_GNT_PM_CANON = ((2048, False), (8192, True))
# Per-point hysteresis: a candidate must beat the base at EVERY point by this factor. A
# geomean-only gate hides cfgs that trade one regime for another -- one cfg serves every
# distribution, since the cfg cache keys on the static shape alone.
_GNT_AT_MARGIN = 0.985


def _gnt_nt_candidates(N, K, pm):
    """Flat (bm,gm,xcd,gn) autotune candidate list (4 max); cand[0] is the base reference.

    Trimmed 2026-07-09 from 5-7 to 4 via per-candidate AT_DBG timings across the MoE
    bench shapes (mi355x_vs_b200_grouped_gemm_fp8_tensorwise.md): the 2D N-band (gn>0)
    and (256,8,8,0) candidates never won on any fwd/dgrad shape, so only gm/xcd swizzles
    are kept.

    GROUP_M widens the M band an A tile-row block is re-read over: B is streamed once
    per band, so B traffic scales as 1/gm, while the band's A footprint (gm*bm*K bytes)
    eventually outgrows an XCD's private L2 slice and the A re-reads stop hitting. The
    trade point therefore moves with BOTH K and the tokens/group `pm` a band is cut out
    of, and a single global band width mis-serves long-K and training-scale shapes alike.
    Measured tw/mx over gm 2/4/8/16 at MoE bench scale (2048-8192 tokens/group) --
    N=2944 K=5760: 1.010 / 1.036 / 1.047 / 1.031, against 1.089 / 1.122 / 1.086 at
    K=2944, i.e. long K asks for the wide band and short K for the narrow one. At the
    gpt-oss e2e scale (G=4, 32768 tokens/group) the short-K shapes flip to the wide band
    as well: gm=8 over gm=4 is +1.9% on N=5760 K=2944 and +4.3% on N=2944 K=2944, +1.3%
    over the whole fwd/dgrad/wgrad set.
    """
    gm0, gm1 = (8, 4) if (4 * 256 * K > _L2_SLICE_BYTES or pm >= _GNT_WIDE_BAND_PM) else (4, 8)
    return [
        (256, gm0, 4, 0),  # base ref (default); the band width K and pm ask for
        (256, gm1, 4, 0),  # the other band width
        (256, 4, 8, 0),  # xcd=8 — wins qwen3-Down fwd (M=2048)
        (256, 1, 4, 0),  # gm=1 — wins several other MX MoE shapes (off-bench)
    ]


def _get_nt_launch(
    K,
    G,
    N,
    bm,
    gm,
    xcd,
    gn,
    cbsz,
    blgp,
    out_fp16,
    persistent,
    store_cshuffle=False,
    cstore_aux=None,
    preshuffle=True,
):
    fk = (
        K,
        G,
        N,
        bm,
        gm,
        xcd,
        gn,
        cbsz,
        blgp,
        out_fp16,
        persistent,
        store_cshuffle,
        cstore_aux,
        preshuffle,
    )
    launch = _GNT_FUSED_CACHE.get(fk)
    if launch is None:
        launch = _compile_grouped_mxfp8_nt_fused(
            K,
            G,
            N,
            bm,
            gm,
            xcd,
            gn,
            cbsz,
            blgp,
            out_fp16,
            persistent,
            store_cshuffle,
            cstore_aux,
            preshuffle=preshuffle,
        )
        _GNT_FUSED_CACHE[fk] = launch
    return launch


def _canon_go(G, pm, skew, dev):
    """[G+1] group offsets holding G*pm rows: evenly split, or (skew) a top-heavy power law
    in BLOCK_M units -- the canonical shape of MoE routing, where a few experts take most of
    the tokens and the tail groups fall back to row-major tiles."""
    if not skew:
        return torch.arange(0, G + 1, dtype=torch.int64, device=dev) * pm
    nu = G * pm // 256
    w = [(i + 1) ** -2.0 for i in range(G)]
    tw = sum(w)
    u = [max(1, round(nu * x / tw)) for x in w]
    u[0] += nu - sum(u)
    offs, acc = [0], 0
    for x in u:
        acc += x
        offs.append(acc * 256)
    return torch.tensor(offs, dtype=torch.int64, device=dev)


def _canon_nt_targs(args, K, G, N, pm, skew):
    """Synthetic args at `pm` tokens/group (dummy content, only shapes drive timing), groups
    evenly split or top-heavy per `skew`; reuses the M-independent b-side from `args`.
    Returns (targs, out_c=numeric-guard ref)."""
    dev = args[2].device
    stream = args[15]
    M_c = G * pm
    K128 = K // 128
    nsc = K // 32  # nsc % 4 == 0 (K % 128 == 0) so the int32 view is exact

    a8_c = torch.randint(0, 127, (M_c, K), device=dev, dtype=torch.int8)
    out_c = torch.empty((M_c, N), device=dev, dtype=args[2].dtype)
    a_scale_c = torch.randint(120, 128, (M_c, nsc), device=dev, dtype=torch.uint8)
    a_raw_c = a_scale_c.view(torch.int32).reshape(-1)
    a_sp_c, b_sp_c, a_blocks_c, a_ngrp_c = _get_grouped_mx_workspace(M_c, N, K128, G, dev, stream)

    n_blocks = (N + _BLOCK_N - 1) // _BLOCK_N
    grid_upper_c = ((M_c + 255) // 256 + G) * n_blocks
    go_c = _canon_go(G, pm, skew, dev).view(torch.int32)

    targs = (
        a8_c,
        args[1],  # b8 (weights, M-independent)
        out_c,
        a_raw_c,
        args[4],  # b_raw (b scales, M-independent)
        a_sp_c,
        b_sp_c,
        go_c,  # go_pad (fully packed)
        go_c,  # go_out (== go_pad)
        M_c,
        a_ngrp_c * 64,
        N,
        a_blocks_c,
        a_ngrp_c,
        grid_upper_c,
        stream,
    )
    return targs, out_c


def _select_nt_cfg(cfg_key, K, G, N, pm, cbsz, blgp, out_fp16, persistent, args):
    """First-call race on synthetic canonical tensors; cache the winning cfg per static shape
    (cfg_key, no M_pad -> reused for every M of the same band-width regime)."""
    cached = _GNT_CFG_CACHE.get(cfg_key)
    if cached is not None:
        return cached

    cands = _gnt_nt_candidates(N, K, pm)
    if pm >= _GNT_WIDE_BAND_PM:
        # Scoring this regime would mean synthesising operands at the caller's own
        # tokens/group -- a multi-GB transient on the first call -- because the canonical
        # points are an order of magnitude smaller and rank the band widths the other way
        # round. Nothing else in the table wins here either: xcd=8 gains 1.5% on even
        # groups and gives 1-2% back on the top-heavy routing that dominates the geomean.
        _GNT_CFG_CACHE[cfg_key] = cands[0]
        return cands[0]
    # one (targs, out_view) per steady point; candidates scored by their geomean cand/base ratio
    points = [_canon_nt_targs(args, K, G, N, pm, skew) for pm, skew in _GNT_PM_CANON]

    def _geomean(ts):
        return math.exp(sum(math.log(t) for t in ts) / len(ts))

    try:
        base = _get_nt_launch(K, G, N, *cands[0], cbsz, blgp, out_fp16, persistent)
        refs = []
        for targs, out_view in points:
            base(*targs)
            torch.cuda.synchronize()
            r = out_view.detach().clone().float()
            if not torch.isfinite(r.reshape(-1)[:1024]).all().item():
                raise RuntimeError("base cfg produced non-finite output")
            refs.append((r, float((r * r).sum().item()) or 1.0))
        _robust_time(base, points[0][0])  # ramp to the sustained-load clock before racing
    except Exception:
        _GNT_CFG_CACHE[cfg_key] = _GNT_NT_DEFAULT_CFG
        return _GNT_NT_DEFAULT_CFG

    best_cfg, best_ratio = cands[0], 1.0
    for cfg in cands[1:]:
        try:
            launch = _get_nt_launch(K, G, N, *cfg, cbsz, blgp, out_fp16, persistent)
            rs, matched = [], True
            for (targs, out_view), (ref, ref_n) in zip(points, refs):
                launch(*targs)
                torch.cuda.synchronize()
                o = out_view.detach().float()
                err = float(((o - ref) * (o - ref)).sum().item())
                # never adopt a config that drifts from the base at any point
                if not ((err / ref_n) < (2e-2**2) and torch.isfinite(o.reshape(-1)[:1024]).all().item()):
                    matched = False
                    break
                rs.append(_robust_ab_ratio(base, launch, targs))
            if not matched:
                continue
        except Exception:
            continue
        # Adopt only a cfg that clears the hysteresis at EVERY point, then keep the fastest
        # such cfg -- ties keep the earlier candidate, so the fixed list order decides.
        score = _geomean(rs)
        if max(rs) < _GNT_AT_MARGIN and score < best_ratio:
            best_cfg, best_ratio = cfg, score

    _GNT_CFG_CACHE[cfg_key] = best_cfg
    return best_cfg


def _compile_grouped_mxfp8_nt_fused(
    K,
    G,
    N,
    bm,
    gm,
    xcd,
    gn,
    cbsz,
    blgp,
    out_fp16,
    persistent,
    store_cshuffle=False,
    cstore_aux=None,
    preshuffle=True,
):
    K128 = K // 128
    pre_kern, n_kt, b_blocks_pg = _build_grouped_preshuffle_kernel(K128, G, N, b_pair=_gnt_pair_n(N))
    gemm_kern, BM, BN, wpe = _build_grouped_mxfp8_nt_kernel(
        K=K,
        G=G,
        N=N,
        BLOCK_M=bm,
        BLOCK_N=_BLOCK_N,
        group_m=gm,
        group_n=gn,
        num_xcd=xcd,
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_fp16,
        persistent=persistent,
        store_cshuffle=store_cshuffle,
        cstore_aux=cstore_aux,
    )

    @flyc.jit
    def launch_grouped_mxfp8_nt_fused(
        a8: fx.Tensor,
        b8: fx.Tensor,
        C: fx.Tensor,
        a_raw: fx.Tensor,
        b_raw: fx.Tensor,
        a_sp: fx.Tensor,
        b_sp: fx.Tensor,
        group_offs: fx.Tensor,
        group_offs_out: fx.Tensor,
        c_m_pad: fx.Int32,  # padded A rows (M_pad) -> preshuffle raw-read bound
        c_scale_rows: fx.Int32,  # a_ngrp_ub*64 -> A-scale slab buffer sizing
        c_n: fx.Int32,
        a_blocks: fx.Int32,
        a_ngrp: fx.Int32,
        grid_upper: fx.Int32,
        stream: fx.Stream,
    ):
        # preshuffle=False: skip the E8M0 scale preshuffle (assumes a_sp/b_sp already
        # populated by a prior call) so the GEMM main kernel can be timed in isolation.
        if const_expr(preshuffle):
            pre_kern(a_raw, b_raw, a_sp, b_sp, group_offs, c_m_pad, a_blocks, a_ngrp).launch(
                grid=(a_blocks + G * b_blocks_pg, 1, 1), block=(_PRESHUF_BLK, 1, 1), stream=stream
            )
        if const_expr(persistent):
            ncus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
            grid_x = arith.select(grid_upper < fx.Int32(ncus), grid_upper, fx.Int32(ncus))
        else:
            grid_x = grid_upper
        gemm_kern(
            a8,
            b8,
            C,
            a_sp,
            b_sp,
            group_offs,
            group_offs_out,
            c_scale_rows,
            c_n,
            value_attrs=make_value_attrs(wpe, 0, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_grouped_mxfp8_nt_fused


def _get_grouped_mx_workspace(M_pad, N, K128, G, device, stream):
    key = (M_pad, N, K128, G, device, stream)
    e = _GNT_WS_CACHE.get(key)
    if e is None:
        # a_ngrp = per-group 64-row-slab count upper bound; host over-allocates since
        # the exact count is device-only (each group adds at most one partial slab).
        a_ngrp = (M_pad + 63) // 64 + G
        b_ngrp_pg = ((N + 255) // 256) * 4
        n_kt = (K128 + _PRESHUF_KT - 1) // _PRESHUF_KT
        K128p = ceildiv(K128, _GG_SCALE_PACK)  # packed K-groups (PACK scales / dword)
        a_blocks = a_ngrp * n_kt
        a_sp = torch.empty(a_ngrp * K128p * 256, dtype=torch.int32, device=device)
        b_sp = torch.empty(G * b_ngrp_pg * K128p * 256, dtype=torch.int32, device=device)
        e = (a_sp, b_sp, a_blocks, a_ngrp)
        _GNT_WS_CACHE[key] = e
    return e


def grouped_gemm_mxfp8_flydsl_kernel(
    a: "torch.Tensor",
    a_scale: "torch.Tensor",
    b: "torch.Tensor",
    b_scale: "torch.Tensor",
    group_offs: "torch.Tensor",  # padded read offsets [G+1]
    N: int,
    K: int,
    group_offs_out: "torch.Tensor | None" = None,  # tight write offsets [G+1]; None => group_offs
    out_dtype: torch.dtype = torch.bfloat16,
    num_cu: "int | None" = -1,
    preshuffle: bool = True,
) -> "torch.Tensor":
    """FlyDSL MXFP8 grouped NT GEMM (fwd / dgrad). Returns C [M_pad, N].

    preshuffle=False skips the fused E8M0 scale preshuffle (assumes the cached
    workspace was populated by a prior preshuffle=True call on the same shape) so the
    GEMM main kernel can be timed alone; used by benches, not the training path."""
    assert a.ndim == 2 and b.ndim == 3
    M_pad = a.shape[0]
    G = b.shape[0]
    assert b.shape[1] == N and b.shape[2] == K and a.shape[1] == K
    assert K % 128 == 0 and K >= 256
    out_fp16 = out_dtype == torch.float16
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0
    K128 = K // 128

    a_raw = (a_scale if a_scale.is_contiguous() else a_scale.contiguous()).view(torch.int32).reshape(-1)
    b_raw = (b_scale if b_scale.is_contiguous() else b_scale.contiguous()).view(torch.int32).reshape(-1)
    a8 = a.contiguous().view(torch.int8)
    b8 = b.contiguous().view(torch.int8)
    out = torch.empty((M_pad, N), dtype=out_dtype, device=a.device)

    _go_pad = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    # None => write in the padded read layout (matches the Triton MX grouped wrapper).
    if group_offs_out is None:
        group_offs_out = group_offs
    _go_out = group_offs_out if group_offs_out.dtype == torch.int64 else group_offs_out.to(torch.int64)
    go_pad = _go_pad.view(torch.int32)
    go_out = _go_out.view(torch.int32)

    stream = torch.cuda.current_stream()
    a_sp, b_sp, a_blocks, a_ngrp = _get_grouped_mx_workspace(M_pad, N, K128, G, a.device, stream)

    n_blocks = (N + _BLOCK_N - 1) // _BLOCK_N
    grid_upper = ((M_pad + 255) // 256 + G) * n_blocks

    persistent = num_cu is not None and num_cu > 0
    args = (
        a8,
        b8,
        out,
        a_raw,
        b_raw,
        a_sp,
        b_sp,
        go_pad,
        go_out,
        M_pad,
        a_ngrp * 64,
        N,
        a_blocks,
        a_ngrp,
        grid_upper,
        stream,
    )
    # at_key bakes buffers/workspace per M_pad; cfg_key (no M_pad, only the band-width regime
    # the mean tokens/group falls in) picks the swizzle once per shape.
    pm = M_pad // G
    at_key = (M_pad, N, K, G, cbsz, blgp, out_fp16, persistent, preshuffle)
    cfg_key = (N, K, G, cbsz, blgp, out_fp16, persistent, pm >= _GNT_WIDE_BAND_PM)
    entry = _GNT_AT_CACHE.get(at_key)
    if entry is None:
        # race on canonical synthetic tensors -> needs only the static shape (args' b-side)
        bm, gm, xcd, gn = _select_nt_cfg(cfg_key, K, G, N, pm, cbsz, blgp, out_fp16, persistent, args)
        launch = _get_nt_launch(
            K, G, N, bm, gm, xcd, gn, cbsz, blgp, out_fp16, persistent, preshuffle=preshuffle
        )
        entry = [launch, None]
        _GNT_AT_CACHE[at_key] = entry
    run_eager_or_capture(entry, args, 1)
    return out


# WGRAD (variable-K TN): C[g] (OUT_M, OUT_N) = LHS[:, g] @ RHS[:, g]^T, contraction =
# per-group M_g tokens (padded to a multiple of BLOCK_K=128). K128 = M_total//128 is a
# runtime arg (no per-group slab rebase; whole-tensor A/B preshuffle).


def _build_grouped_wgrad_preshuffle_kernel(
    OUT_M: int, OUT_N: int, G: int, KT: int = _PRESHUF_KT, BLK: int = 256, pack: int = 1
):
    """LHS (layout 1) + RHS (B-comb layout 3) E8M0 scale preshuffle for the wgrad.

    Returns ``(kern, a_ngrp, b_ngrp)``. ``pack`` packs PACK consecutive contraction
    (m_total) K-blocks per output dword (reader selects the byte via MFMA op_sel).
    Each group is packed from its OWN contraction start, so pack>1 needs no alignment
    of the group offsets: group g's packed region begins at ``go[g]//BLOCK_K//pack + g``
    and the K-stride is ``k128//pack + G``. The single spare dword per group is what
    keeps the regions disjoint (floor(a/p)+ceil(n/p) <= floor((a+n)/p)+1), and paying it
    here keeps the timed main kernel at one extra integer add.
    """
    a_ngrp = ceildiv(OUT_M, 64)
    b_ngrp = ((OUT_N + 255) // 256) * 4

    @fx.struct
    class Smem:
        tile: fx.Array[fx.Int32, 64 * KT, 16]

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def kern(
        a_raw: fx.Tensor,
        b_raw: fx.Tensor,
        a_sp: fx.Tensor,
        b_sp: fx.Tensor,
        group_offs: fx.Tensor,  # padded per-group M offsets (int32 view of int64 [G+1])
        k128: fx.Int32,  # contraction blocks = M_total // 128
        n_ck: fx.Int32,  # per-row-group KT-chunk slots = k128//KT + G (upper bound)
        a_blocks: fx.Int32,  # a_ngrp * n_ck
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        tile = fx.SharedAllocator().allocate(Smem).peek().tile
        k128p = k128 // pack + fx.Int32(G)  # packed K-stride (one spare dword per group)
        rin_a = _buffer_ops.create_buffer_resource(
            a_raw, max_size=False, num_records_bytes=fx.Int32(OUT_M) * k128 * 4
        )
        rin_b = _buffer_ops.create_buffer_resource(
            b_raw, max_size=False, num_records_bytes=fx.Int32(OUT_N) * k128 * 4
        )
        rout_a = _buffer_ops.create_buffer_resource(
            a_sp, max_size=False, num_records_bytes=fx.Int32(a_ngrp) * k128p * 256 * 4
        )
        rout_b = _buffer_ops.create_buffer_resource(
            b_sp, max_size=False, num_records_bytes=fx.Int32(b_ngrp) * k128p * 256 * 4
        )
        go = fx.rocdl.make_buffer_tensor(group_offs, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go, fx.make_layout(1, 1))

        # Chunk slot -> owning group: O(G) scan over the per-group KT-chunk counts (this
        # pass is not on the timed path). Slots past the real count fall through unwritten.
        bb = arith.select(bid < a_blocks, bid, bid - a_blocks)
        row_grp = bb // n_ck
        ck = bb % n_ck
        acc = fx.Int32(0)
        ks0 = fx.Int32(0)
        k_iters = fx.Int32(0)
        j0 = fx.Int32(0)
        gidx = fx.Int32(0)
        valid = fx.Int32(0)
        prev = _load_go(go_div, 0) // 128
        for g in range_constexpr(G):
            nxt = _load_go(go_div, g + 1) // 128
            ki = nxt - prev
            nc = ceildiv(ki, KT)
            inq = (ck >= acc) & (ck < acc + nc)
            ks0 = arith.select(inq, prev, ks0)
            k_iters = arith.select(inq, ki, k_iters)
            j0 = arith.select(inq, (ck - acc) * KT, j0)
            gidx = arith.select(inq, fx.Int32(g), gidx)
            valid = arith.select(inq, fx.Int32(1), valid)
            acc = acc + nc
            prev = nxt
        kp0 = ks0 // pack + gidx  # group's packed base; 256 i32 (64 lanes x 4) per dword

        if valid > fx.Int32(0):
            if bid < a_blocks:
                _emit_lds_repack(
                    True,
                    row_grp,
                    j0,
                    tile,
                    rin_a,
                    rout_a,
                    fx.Int32(OUT_M),
                    k128,
                    KT,
                    tid,
                    BLK,
                    rd_base=ks0,
                    wr_base=kp0 * 256,
                    pack=pack,
                    kbound=k_iters,
                    k128p=k128p,
                )
            if bid >= a_blocks:
                _emit_lds_repack(
                    False,
                    row_grp,
                    j0,
                    tile,
                    rin_b,
                    rout_b,
                    fx.Int32(OUT_N),
                    k128,
                    KT,
                    tid,
                    BLK,
                    rd_base=ks0,
                    wr_base=kp0 * 256,
                    pack=pack,
                    kbound=k_iters,
                    k128p=k128p,
                )

    return kern, a_ngrp, b_ngrp


def _build_grouped_mxfp8_wgrad_kernel(
    OUT_M: int,
    OUT_N: int,
    G: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    group_m: int = 4,
    group_n: int = 0,
    num_xcd: int = 8,
    waves_per_eu: int = 2,
    cbsz: int = 0,
    blgp: int = 0,
    out_fp16: bool = False,
    chunk: int = 8,
    pack: int = 1,
    split_k: bool = True,
):
    """Grouped MXFP8 variable-K wgrad (runtime per-group contraction M_g).

    ``pack``: E8M0 scale packing (PACK scales/dword, op_sel byte select). pack>1 shrinks
    the preshuffled scale buffer + its mainloop scale traffic PACK-fold. Each group is
    packed from its own contraction start (base ``ks0//pack + g``), so the op_sel byte is
    the group-local k%pack for any group offset -- no alignment of group_offs required."""
    BLOCK_K = 128
    assert BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0 and BLOCK_M >= 128 and BLOCK_N >= 256
    assert G >= 1
    assert chunk % 2 == 0, "chunk must be even so the distance-2 ping-pong resets at the chunk boundary"
    assert chunk % pack == 0, "chunk must be a multiple of pack so op_sel = k%pack holds per chunk"

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K
    SA_TILES = N_TILES_A
    N_BLOCKS_M = ceildiv(OUT_M, BLOCK_M)
    N_BLOCKS_N = ceildiv(OUT_N, BLOCK_N)
    TILES_PER_GROUP = N_BLOCKS_M * N_BLOCKS_N
    TOTAL = G * TILES_PER_GROUP
    # Boundary half-tile skip: a trailing block with <= half a tile of valid rows/cols has
    # an all-padding second half, so that tile's MFMAs and LDS reads for it are dead work.
    # Compile-time gated per side; a BLOCK-aligned OUT_M/OUT_N emits only the full body.
    _HALF_M = (OUT_M % BLOCK_M != 0) and (OUT_M % BLOCK_M <= LDS_BLOCK_M)
    _HALF_N = (OUT_N % BLOCK_N != 0) and (OUT_N % BLOCK_N <= LDS_BLOCK_N)

    # Single-window adaptive split-K (ported from the tensorwise wgrad). One WG owns one
    # output tile's whole contraction, so the wall is set either by the last partial round
    # (rule A) or by a hot group's own tile chain (rule B); both are fixed by slicing ONE
    # window of tiles' contraction dim -- slice 0 writes C, slices 1.. write scratch bands a
    # reduce kernel folds back. The window is picked on-device from group_offs; the grid is a
    # compile-time bound truncated by a runtime live count (dead workgroups exit before any memory op).
    # `split_k` gates the whole path: the window only pays on grids shallow enough for one CU
    # round to be a big share of the makespan (see _wgrad_split_pays), and it needs num_xcd <= 1
    # because the window is a contiguous run of the PRE-remap pid space -- under an XCD remap that
    # run scatters across groups and the reduce would fold bands the policy never sliced.
    _NCU = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    _S_MAX, _S_A, _S_B, _N_MAX, _SP_EXT = (
        _wgrad_split_geom(TILES_PER_GROUP, TOTAL, _NCU) if (split_k and num_xcd <= 1) else (1, 1, 1, 0, 0)
    )
    _SPLIT = _S_MAX > 1
    _SP_POW2 = _wgrad_split_rcp_cfg(TILES_PER_GROUP, _S_A, _S_B, _NCU)[0]
    # Scales are packed PACK per dword from each group's own contraction start, so a slice may
    # only begin on a packed-dword boundary (the op_sel byte is the constexpr k%pack). Both
    # slice bounds are floored to `pack`, which costs a slice at most pack-1 K-blocks; the
    # policy guardrail rises to 2*pack so the shortest slice still keeps >= pack blocks.
    _SP_FLOOR = max(6, 2 * pack)
    _GRID_X = TOTAL + _SP_EXT
    # Reduce: one workgroup folds BLOCK_M//_WGRAD_RED_WPT rows x BLOCK_N cols of a window tile.
    _RED_ROWS = BLOCK_M // _WGRAD_RED_WPT
    _RED_LPR = BLOCK_N // _WGRAD_RED_VEC  # lanes spanning one tile row
    _RED_RPP = 256 // _RED_LPR  # rows one 256-thread pass covers
    _RED_L2WPT = _WGRAD_RED_WPT.bit_length() - 1
    _RED_GRID = max(1, _N_MAX * _WGRAD_RED_WPT)
    assert not _SPLIT or (
        _WGRAD_RED_WPT & (_WGRAD_RED_WPT - 1) == 0
        and _RED_ROWS % _RED_RPP == 0
        and OUT_N % _WGRAD_RED_VEC == 0
    ), "split-K reduce needs a pow2 WPT, row-aligned passes and a vector-aligned OUT_N"

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_grouped_mxfp8_wgrad(
        A: fx.Tensor,  # LHS [OUT_M, M_total] fp8
        B: fx.Tensor,  # RHS [OUT_N, M_total] fp8
        C: fx.Tensor,  # [G, OUT_M, OUT_N]
        A_scale: fx.Tensor,  # preshuffled LHS scale (layout 1)
        B_scale: fx.Tensor,  # preshuffled RHS scale (B-comb layout 3)
        group_offs: fx.Tensor,  # padded per-group M offsets (int32 view of int64 [G+1])
        WS: fx.Tensor,  # split-K slice scratch, (S_MAX-1) row bands at C's pitch
        m_total: fx.Int32,  # total padded contraction length (LHS/RHS leading dim)
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        go = fx.rocdl.make_buffer_tensor(group_offs, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go, fx.make_layout(1, 1))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pid = fx.block_idx.x

        def _do_tile(t, slice_id=None, split_s=None, split_code=None):
            tt = xcd_remap_pid(t, TOTAL, num_xcd)
            # interleave=False: group-major tile order, i.e. longest-processing-time-first
            # for the usual histogram (low group ids hold the most tokens), instead of
            # band-cyclic, which pushes a hot group's later M-bands to the tail of the
            # launch sequence. num_xcd=1 then leaves consecutive tiles on the HW XCD
            # round-robin so every XCD gets an even slice of each group; group_m still
            # keeps the B-stripe L2 reuse inside the band.
            group_idx, block_m, block_n = _wgrad_block_mn(
                tt, G, TILES_PER_GROUP, N_BLOCKS_M, N_BLOCKS_N, group_m, group_n, False
            )
            m_start = _load_go(go_div, group_idx)
            m_end = _load_go(go_div, group_idx + 1)
            # All three quotients are non-negative with power-of-two divisors, so they go
            # through floordiv_pow2: signed floordiv would put a divide/remainder/sign-select
            # chain on the per-workgroup prologue, which has no MFMA to hide behind.
            k_iters = floordiv_pow2(m_end - m_start, BLOCK_K)  # M_g padded to 128 -> exact
            row_shift = None
            store_base = None
            if slice_id is not None:
                # Contraction (token) split: slice s owns K-blocks [kb0, kb1) of this tile.
                # Both bounds are floored to a multiple of `pack` so the slice starts on a
                # packed-scale dword and the op_sel byte stays the constexpr k%pack; the
                # per-group packed base below then picks the shift up automatically, since
                # kb0*BLOCK_K is a whole number of packed dwords. slice_id < 0 marks a tile
                # outside the window: whole K range, writes C. Only the trip count, the
                # operand row origin and the store band move, so the loop body is untouched.
                _s = fx.Int32(_readfirstlane_i32(slice_id))
                _whole = _s < fx.Int32(0)
                _sc = fx.Int32(arith.select(_whole, fx.Int32(0), _s))
                _ki = fx.Int32(k_iters)
                _nxt = _sc + fx.Int32(1)
                _b0 = _wgrad_split_div(_ki * _sc, split_code, _SP_POW2)
                _b1 = _wgrad_split_div(_ki * _nxt, split_code, _SP_POW2)
                if const_expr(pack > 1):
                    _b0 = fx.Int32(_b0 & fx.Int32(-pack))
                    _b1 = fx.Int32(_b1 & fx.Int32(-pack))
                kb0 = fx.Int32(arith.select(_whole, fx.Int32(0), _b0))
                kb1 = fx.Int32(arith.select(_whole, _ki, arith.select(_nxt < split_s, _b1, _ki)))
                m_start = _readfirstlane_i32(m_start + kb0 * fx.Int32(BLOCK_K))
                k_iters = _readfirstlane_i32(kb1 - kb0)
                _part = _s > fx.Int32(0)
                _band = (_s - fx.Int32(1) - group_idx) * fx.Int32(OUT_M)
                row_shift = _readfirstlane_i32(arith.select(_part, _band, fx.Int32(0)))
                store_base = arith.select(
                    _part, _buffer_ops.extract_base_index(WS), _buffer_ops.extract_base_index(C)
                )
            # Scales are packed per group from the group's own contraction start, so the
            # scale index is group-local and the base is (ks0//pack + g): the one spare
            # dword per group keeps the regions disjoint for any group offset (no 512
            # alignment assumed), at the cost of G spare dwords in the K-stride.
            kp0 = floordiv_pow2(m_start, BLOCK_K * pack) + group_idx
            k128p = floordiv_pow2(m_total, BLOCK_K * pack) + fx.Int32(G)

            lane_id = fx.thread_idx.x % 64
            wave_id = fx.thread_idx.x // 64
            wave_m = wave_id // 4
            wave_n = wave_id % 4

            a_cur0 = lds.A_lds_cur_0
            a_cur1 = lds.A_lds_cur_1
            a_next0 = lds.A_lds_next_0
            a_next1 = lds.A_lds_next_1
            b_cur0 = lds.B_lds_cur_0
            b_cur1 = lds.B_lds_cur_1
            b_next0 = lds.B_lds_next_0
            b_next1 = lds.B_lds_next_1

            # SRD base = row_base*m_total + m_start; num_records bounds to the tensor end.
            mt_i = arith.index_cast(T.index, m_total)
            a_row = block_m * BLOCK_M
            b_row = block_n * BLOCK_N
            ms_i = arith.index_cast(T.index, m_start)
            a_base = arith.index_cast(T.index, a_row) * mt_i + ms_i
            b_base = arith.index_cast(T.index, b_row) * mt_i + ms_i
            a_nrec = arith.index_cast(T.index, fx.Int32(OUT_M)) * mt_i - a_base
            b_nrec = arith.index_cast(T.index, fx.Int32(OUT_N)) * mt_i - b_base
            gA = make_fp8_buffer_tensor_rebased(A, F8_IR_t, a_base, a_nrec)
            gB = make_fp8_buffer_tensor_rebased(B, F8_IR_t, b_base, b_nrec)
            a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
            b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

            gl_off_a = compute_global_swizzle(lane_id, wave_id, m_total, N_LDS_ROUNDS, preshuffled=False)
            gl_off_b = compute_global_swizzle(lane_id, wave_id, m_total, N_LDS_ROUNDS, preshuffled=False)

            A1off = LDS_BLOCK_M * m_total  # region1 = OUT_M rows [LDS_BLOCK_M, BLOCK_M)
            B1off = LDS_BLOCK_N * m_total

            mfma = MfmaScale16x16x128(N_TILES_A, N_TILES_B, cbsz=cbsz, blgp=blgp)
            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            a_s2r = S2RLoader(wave_m, N_TILES_A)
            b_s2r = S2RLoader(wave_n, N_TILES_B)

            sa_s2r = ScaleS2R(A_scale, OUT_M, m_total, SA_TILES, pack=pack, k128p=k128p)
            sb_s2r = ScaleBComb(B_scale, OUT_N, m_total, pack=pack, k128p=k128p)
            # A partial slice stores into scratch band s-1 instead of C: same row pitch, so
            # only the SRD base and the band end move with row_shift.
            _c_rows = (group_idx + 1) * OUT_M
            if row_shift is not None:
                _c_rows = _c_rows + row_shift
            store_c = StoreCPerTensor(
                None,
                None,
                C,
                _c_rows,
                OUT_N,
                mfma.idx,
                N_TILES_A,
                N_TILES_B,
                _out_ty,
                c_base=store_base,
            )

            wave_m_offset = wave_m * (N_TILES_A * 16)
            wave_n_offset = wave_n * (N_TILES_B * 16)
            sa_base0 = a_row + wave_m_offset
            sa_base1 = sa_base0 + fx.Int32(LDS_BLOCK_M)
            sb_base0 = b_row + wave_n_offset

            # Accumulators (rmem) are allocated just before the K-loop below.

            # Prologue: tile 0 -> cur, tile 1 -> next (distance-2 prelude).
            b_g2s.load(b_cur0, 0 + 0 * BLOCK_K)
            a_g2s.load(a_cur0, 0 + 0 * BLOCK_K)
            b_g2s.load(b_cur1, B1off + 0 * BLOCK_K)
            a_g2s.load(a_cur1, A1off + 0 * BLOCK_K)
            if wave_m == 1:
                rocdl.s_barrier()
            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
            b_g2s.load(b_next0, 0 + 1 * BLOCK_K)
            a_g2s.load(a_next0, 0 + 1 * BLOCK_K)
            b_g2s.load(b_next1, B1off + 1 * BLOCK_K)
            wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

            base_row = group_idx * OUT_M + a_row + wave_m_offset
            if row_shift is not None:
                base_row = base_row + row_shift
            base_col = b_row + wave_n_offset

            # rmem accumulators carry across the dynamic chunk loop via memory; zero-init once.
            acc00 = [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range_constexpr(N_ACCUMS)]
            acc01 = [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range_constexpr(N_ACCUMS)]
            acc10 = [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range_constexpr(N_ACCUMS)]
            acc11 = [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range_constexpr(N_ACCUMS)]
            for q in (acc00, acc01, acc10, acc11):
                for r in q:
                    fx.memref_store_vec(mfma.zero_value, r)
            # Scale-prefetch carriers: (A half 0, B combined). Same mechanism as the
            # accumulators -- the chunk body hands the next chunk's first packed dword across
            # the dynamic loop through memory, which promotes to a register phi.
            sc_pf = (
                fx.make_rmem_tensor(fx.make_layout(SA_TILES, 1), fx.Int32),
                fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32),
            )
            # Chunk 0's dword. Issued after the distance-2 prelude's vmcnt fences so it does
            # not shift their outstanding-load counts.
            fx.memref_store_vec(sa_s2r.load_vec(sa_base0, fx.Int32(0), kp0), sc_pf[0])
            fx.memref_store_vec(sb_s2r.load_vec(sb_base0, fx.Int32(0), kp0), sc_pf[1])

            # MAIN: every FULL chunk runs the SSA-accumulator body (memref RMW only at the
            # chunk boundary); the dynamic loop carries just the 8 swapped buffer refs.
            # `quads` selects the live output halves for this tile's boundary position;
            # g2s / barrier / vmcnt sequences are identical across the variants.
            def _run(quads):
                a_cur0 = lds.A_lds_cur_0
                a_cur1 = lds.A_lds_cur_1
                a_next0 = lds.A_lds_next_0
                a_next1 = lds.A_lds_next_1
                b_cur0 = lds.B_lds_cur_0
                b_cur1 = lds.B_lds_cur_1
                b_next0 = lds.B_lds_next_0
                b_next1 = lds.B_lds_next_1
                _nfull = k_iters // chunk
                for _c in range(_nfull):
                    (a_cur0, a_cur1, b_cur0, b_cur1, a_next0, a_next1, b_next0, b_next1) = _wgrad_ssa_chunk(
                        _c * chunk,
                        chunk,
                        kp0,
                        BLOCK_K,
                        A1off,
                        B1off,
                        a_g2s,
                        b_g2s,
                        a_s2r,
                        b_s2r,
                        sa_s2r,
                        sb_s2r,
                        mfma,
                        a_cur0,
                        a_cur1,
                        b_cur0,
                        b_cur1,
                        a_next0,
                        a_next1,
                        b_next0,
                        b_next1,
                        acc00,
                        acc01,
                        acc10,
                        acc11,
                        sa_base0,
                        sa_base1,
                        sb_base0,
                        N_ACCUMS,
                        N_LDS_STEPS_A,
                        N_LDS_STEPS_B,
                        sc_pf,
                        pack,
                        quads,
                    )

                # TAIL: the k_iters%chunk remainder tiles, per-tile (k_abs<k_iters) guarded.
                _tail0 = _nfull * chunk
                for _j in range_constexpr(chunk):
                    k_abs = _tail0 + _j
                    if k_abs < k_iters:
                        _wgrad_mx_body_4buf(
                            k_abs,
                            kp0,
                            BLOCK_K,
                            A1off,
                            B1off,
                            a_g2s,
                            b_g2s,
                            a_s2r,
                            b_s2r,
                            sa_s2r,
                            sb_s2r,
                            mfma,
                            a_cur0,
                            a_cur1,
                            b_cur0,
                            b_cur1,
                            a_next0,
                            a_next1,
                            b_next0,
                            b_next1,
                            acc00,
                            acc01,
                            acc10,
                            acc11,
                            sa_base0,
                            sa_base1,
                            sb_base0,
                            N_LDS_STEPS_A,
                            N_LDS_STEPS_B,
                            scale_opsel(_j, pack),
                            quads,
                        )
                    a_cur0, a_next0 = a_next0, a_cur0
                    a_cur1, a_next1 = a_next1, a_cur1
                    b_cur0, b_next0 = b_next0, b_cur0
                    b_cur1, b_next1 = b_next1, b_cur1

            # Scalar (wave-uniform) boundary predicates: the bodies contain s_barrier, so
            # every wave of the WG must take the same path.
            if const_expr(_HALF_M and _HALF_N):
                if _readfirstlane_i32(block_m) == fx.Int32(N_BLOCKS_M - 1):
                    if _readfirstlane_i32(block_n) == fx.Int32(N_BLOCKS_N - 1):
                        _run((1, 1))
                    else:
                        _run((1, 2))
                else:
                    if _readfirstlane_i32(block_n) == fx.Int32(N_BLOCKS_N - 1):
                        _run((2, 1))
                    else:
                        _run((2, 2))
            elif const_expr(_HALF_M):
                if _readfirstlane_i32(block_m) == fx.Int32(N_BLOCKS_M - 1):
                    _run((1, 2))
                else:
                    _run((2, 2))
            elif const_expr(_HALF_N):
                if _readfirstlane_i32(block_n) == fx.Int32(N_BLOCKS_N - 1):
                    _run((2, 1))
                else:
                    _run((2, 2))
            else:
                _run((2, 2))

            # All four quadrants are stored unconditionally: a quadrant skipped above keeps
            # its zero-init and lies entirely outside the row/col bounds, which StoreC
            # clamps. Keeping the stores here also keeps them out of `_run`, whose scf.for
            # body extraction only captures names the loop itself references.
            c00_frag = [Vec(fx.memref_load_vec(r)) for r in acc00]
            c01_frag = [Vec(fx.memref_load_vec(r)) for r in acc01]
            c10_frag = [Vec(fx.memref_load_vec(r)) for r in acc10]
            c11_frag = [Vec(fx.memref_load_vec(r)) for r in acc11]
            store_c.store(c00_frag, base_row + 0, base_col + 0, ((LDS_BLOCK_N, c01_frag),))
            store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0, ((LDS_BLOCK_N, c11_frag),))

        if const_expr(_SPLIT):
            _lo, _n, _s, _code = _wgrad_split_policy(
                go_div, G, TILES_PER_GROUP, TOTAL, BLOCK_K, _NCU, _S_A, _S_B, _SP_FLOOR
            )
            _nsl = _readfirstlane_i32(_n * _s)  # dispatch ids the window expands to
            _live = _readfirstlane_i32(fx.Int32(TOTAL) + (_nsl - _n))
            # Window ids [lo, lo+n*S) carry (tile, slice); ids above it shift back by n*(S-1).
            _rel = pid - _lo
            _pre = _rel < fx.Int32(0)
            _in = _rel < _nsl
            _q = _wgrad_split_div(_rel, _code, _SP_POW2)
            _t = _readfirstlane_i32(arith.select(_pre, pid, arith.select(_in, _lo + _q, pid - (_nsl - _n))))
            _sid = _readfirstlane_i32(
                arith.select(_pre, fx.Int32(-1), arith.select(_in, _rel - _q * _s, fx.Int32(-1)))
            )
            # The grid is the worst-case window expansion; the ids the policy left unused exit
            # here, before any LDS or global traffic.
            if pid < _live:
                _do_tile(_t, slice_id=_sid, split_s=_s, split_code=_code)
        else:
            _do_tile(pid)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_grouped_mxfp8_wgrad_reduce(C: fx.Tensor, group_offs: fx.Tensor, WS: fx.Tensor):
        """Fold the split-K scratch bands back into C. The window policy is recomputed here
        (identical wave-uniform scalars), so this only touches the n window tiles -- the rest
        of a band is never read and needs no zeroing pass. Slots are summed in a fixed 0..S-2
        order in fp32, keeping the store bit-reproducible."""
        _ = str(fx.thread_idx.x)
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        _ir_ty = _out_ty.ir_type
        f32v = fx.T.VectorType.get([_WGRAD_RED_VEC], fx.T.f32())
        outv = fx.T.VectorType.get([_WGRAD_RED_VEC], _ir_ty)
        go = fx.rocdl.make_buffer_tensor(group_offs, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go, fx.make_layout(1, 1))
        _lo, _n, _s_run, _ = _wgrad_split_policy(
            go_div, G, TILES_PER_GROUP, TOTAL, BLOCK_K, _NCU, _S_A, _S_B, _SP_FLOOR
        )
        c_base = _buffer_ops.extract_base_index(C)
        ws_base = _buffer_ops.extract_base_index(WS)
        tid = fx.thread_idx.x
        col_l = (tid % fx.Int32(_RED_LPR)) * fx.Int32(_WGRAD_RED_VEC)
        row_l = tid // fx.Int32(_RED_LPR)
        live = _readfirstlane_i32(_n * fx.Int32(_WGRAD_RED_WPT))
        for w in range(fx.block_idx.x, live, fx.grid_dim.x):
            slot = _readfirstlane_i32(w >> _RED_L2WPT)
            sub = _readfirstlane_i32(w & fx.Int32(_WGRAD_RED_WPT - 1))
            _tt = xcd_remap_pid(_lo + slot, TOTAL, num_xcd)
            _gi, _bm, _bn = _wgrad_block_mn(
                _tt, G, TILES_PER_GROUP, N_BLOCKS_M, N_BLOCKS_N, group_m, group_n, False
            )
            gi = _readfirstlane_i32(_gi)
            bm_off = _readfirstlane_i32(_bm * fx.Int32(BLOCK_M))
            bn_off = _readfirstlane_i32(_bn * fx.Int32(BLOCK_N))
            col = bn_off + col_l
            col_ok = col < fx.Int32(OUT_N)
            # Rows past OUT_M fall outside the band SRD (dropped); columns would wrap, hence col_ok.
            rs_c = make_row_band_resource(c_base, gi * OUT_M + bm_off, (gi + 1) * OUT_M, OUT_N, 2)
            rs_w = [
                make_row_band_resource(ws_base, bm_off + fx.Int32((s - 1) * OUT_M), s * OUT_M, OUT_N, 2)
                for s in range_constexpr(1, _S_MAX)
            ]
            off0 = (sub * fx.Int32(_RED_ROWS) + row_l) * fx.Int32(OUT_N) + col
            for p in range_constexpr(_RED_ROWS // _RED_RPP):
                off = off0 + fx.Int32(p * _RED_RPP * OUT_N)
                acc = arith.extf(
                    f32v,
                    _buffer_ops.buffer_load(rs_c, off, vec_width=_WGRAD_RED_VEC, dtype=_ir_ty, mask=col_ok),
                )
                for s in range_constexpr(1, _S_MAX):
                    # slots >= S were never written: address them out of bounds (HW returns 0).
                    off_s = arith.select(fx.Int32(s) < _s_run, off, fx.Int32(0x3FFFFFFF))
                    acc = arith.addf(
                        acc,
                        arith.extf(
                            f32v,
                            _buffer_ops.buffer_load(
                                rs_w[s - 1],
                                off_s,
                                vec_width=_WGRAD_RED_VEC,
                                dtype=_ir_ty,
                                mask=col_ok,
                            ),
                        ),
                    )
                _buffer_ops.buffer_store(arith.trunc_f(outv, acc), rs_c, off, mask=col_ok)

    _red = kernel_grouped_mxfp8_wgrad_reduce if _SPLIT else None
    return kernel_grouped_mxfp8_wgrad, _red, waves_per_eu, _GRID_X, _RED_GRID


# ── wgrad host wrapper ───────────────────────────────────────────────────────

_GWG_FUSED_CACHE: dict = {}  # (OUT_M, OUT_N, G, bm, bn, gm, xcd, gn, cbsz, blgp, out_fp16) -> launch
_GWG_WS_CACHE: dict = {}  # (OUT_M, OUT_N, K128, device, stream) -> (a_sp, b_sp)
_GWG_AT_CACHE: dict = {}  # (OUT_M, OUT_N, M_total, G, cbsz, blgp, out_fp16) -> [raw, compiled]
_GWG_CFG_CACHE: dict = {}  # cfg_key (NO M_total) -> (gm, xcd, gn) chosen by autotune

# wgrad tile-swizzle autotune. The tile grid is G*ceildiv(OUT_M,bm)*ceildiv(OUT_N,bn) -- fixed
# by the static shape alone -- so the race keys on that shape and is reused for every M_total.
_GWG_DEFAULT_CFG = (4, 1, 0)  # (GROUP_M, num_xcd, group_n); the base every candidate must beat
# gn>0 switches the walk to _band_block_mn's 2D super-block: N is cut into width-gn bands and a
# gm x gn block of tiles is walked inside one, so a tile's A slab is re-read gn times and its B
# slab gm times over a working set of (gm + gn) slabs -- gm=4/gn=2 asks 6 slabs of an XCD's 4 MB
# L2 slice where gn=0 (gm M blocks against the whole N row) asks gm + N_BLOCKS_N = 27.
# Every candidate keeps num_xcd=1: an XCD remap wins 3.8-4.5% on evenly split groups and gives
# 50-63% back on top-heavy routing (it reorders the walk across groups, so a tail group's tiles
# land on CUs still chewing the head group's), besides disabling split-K in the factory.
_GWG_CANDS = ((4, 1, 2), (2, 1, 2), (2, 1, 4))
# One tokens/group point, both distributions: the swizzle only reorders the tile grid, which is
# M-independent, but WHICH tiles are hot is not -- a balanced-only race adopts cfgs that cost
# several % on the top-heavy routing real MoE produces.
_GWG_PM_CANON = ((2048, False), (2048, True))
# Per-point hysteresis, above the interleaved-A/B noise band (<=0.5%, worst 0.9%).
_GWG_AT_MARGIN = 0.99
# Share of the makespan the split-K window must win back to be worth compiling in. Slicing one
# window's contraction dim buys at best the exposed tail round, (1 - 1/S_A) of one CU round out
# of the ceildiv(TOTAL, NCU) a launch takes. It costs the per-tile slice bookkeeping (runtime
# slice bounds, the C-vs-scratch store base select, the row-band shift) in EVERY tile whether the
# window fires or not, and at 1 WG/CU that sits latency-exposed in the tile prologue/epilogue:
# +1483 static instructions, measured 2.3% (12x23 tile grid) / 3.2% (12x12) of the kernel on
# distributions where the policy picks S=1.
_GWG_SPLIT_TAX = 0.03


def _compile_grouped_mxfp8_wgrad_fused(
    OUT_M, OUT_N, G, bm, bn, gm, xcd, gn, cbsz, blgp, out_fp16, pack=1, preshuffle=True, split_k=True
):
    pre_kern, a_ngrp, b_ngrp = _build_grouped_wgrad_preshuffle_kernel(OUT_M, OUT_N, G, pack=pack)
    gemm_kern, red_kern, wpe, grid_x, red_grid = _build_grouped_mxfp8_wgrad_kernel(
        OUT_M=OUT_M,
        OUT_N=OUT_N,
        G=G,
        BLOCK_M=bm,
        BLOCK_N=bn,
        group_m=gm,
        group_n=gn,
        num_xcd=xcd,
        cbsz=cbsz,
        blgp=blgp,
        out_fp16=out_fp16,
        pack=pack,
        split_k=split_k,
    )

    @flyc.jit
    def launch_grouped_mxfp8_wgrad_fused(
        a8: fx.Tensor,
        b8: fx.Tensor,
        C: fx.Tensor,
        a_raw: fx.Tensor,
        b_raw: fx.Tensor,
        a_sp: fx.Tensor,
        b_sp: fx.Tensor,
        group_offs: fx.Tensor,
        WS: fx.Tensor,
        m_total: fx.Int32,
        k128: fx.Int32,
        n_ck: fx.Int32,
        a_blocks: fx.Int32,
        pre_grid: fx.Int32,
        stream: fx.Stream,
    ):
        # preshuffle=False: skip the E8M0 scale preshuffle (assumes a_sp/b_sp already
        # populated by a prior call) so the GEMM main kernel can be timed in isolation.
        if const_expr(preshuffle):
            pre_kern(a_raw, b_raw, a_sp, b_sp, group_offs, k128, n_ck, a_blocks).launch(
                grid=(pre_grid, 1, 1), block=(_PRESHUF_BLK, 1, 1), stream=stream
            )
        gemm_kern(
            a8,
            b8,
            C,
            a_sp,
            b_sp,
            group_offs,
            WS,
            m_total,
            value_attrs=make_value_attrs(wpe, 0, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)
        if const_expr(red_kern is not None):
            # Same stream: the reduce sees every slice partial. The grid is the compile-time
            # window bound; a policy that picks S=1 leaves live=0 and every WG exits at once.
            red_kern(C, group_offs, WS).launch(grid=(red_grid, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_grouped_mxfp8_wgrad_fused


def _get_grouped_wgrad_workspace(OUT_M, OUT_N, K128, G, pack, device, stream):
    key = (OUT_M, OUT_N, K128, G, pack, device, stream)
    e = _GWG_WS_CACHE.get(key)
    if e is None:
        a_ngrp = ceildiv(OUT_M, 64)
        b_ngrp = ((OUT_N + 255) // 256) * 4
        # Packed K-stride: each group packs from its own contraction start, so the stride
        # holds one spare dword per group (see _build_grouped_wgrad_preshuffle_kernel).
        K128p = K128 // pack + G
        a_sp = torch.empty(a_ngrp * K128p * 256, dtype=torch.int32, device=device)
        b_sp = torch.empty(b_ngrp * K128p * 256, dtype=torch.int32, device=device)
        e = (a_sp, b_sp)
        _GWG_WS_CACHE[key] = e
    return e


def _wgrad_split_pays(OUT_M, OUT_N, G, bm, bn, ncu):
    """Is the split-K window worth compiling into every tile of THIS tile grid?

    Rule A -- the distribution-independent half of the policy -- slices the `TOTAL % NCU` tiles of
    the one exposed partial round S_A-fold, so it buys at most (1 - 1/S_A) of a CU round out of the
    ceildiv(TOTAL, NCU) rounds a launch takes, and its extra slices always fit that round (rule A
    picks S_A*rem <= NCU). Against that, the per-tile slice bookkeeping is paid by every tile of
    every round whether the window fires or not: _GWG_SPLIT_TAX. So the window pays on a shallow
    grid, where one round is a large share of the makespan, and stops paying as the grid deepens.
    Rule B (slicing a hot group's own tile chain) rides along whenever the path is compiled in;
    its payoff is a function of the runtime token distribution, which no compile-time key sees."""
    tiles_per_group = ceildiv(OUT_M, bm) * ceildiv(OUT_N, bn)
    total = G * tiles_per_group
    s_max, s_a, _s_b, _n_max, _ext = _wgrad_split_geom(tiles_per_group, total, ncu)
    if s_max <= 1 or s_a <= 1:
        return False
    return (1.0 - 1.0 / s_a) / ceildiv(total, ncu) >= _GWG_SPLIT_TAX


def _get_wgrad_launch(
    OUT_M, OUT_N, G, bm, bn, gm, xcd, gn, cbsz, blgp, out_fp16, pack, preshuffle, split_k
):
    fk = (OUT_M, OUT_N, G, bm, bn, gm, xcd, gn, cbsz, blgp, out_fp16, pack, preshuffle, split_k)
    launch = _GWG_FUSED_CACHE.get(fk)
    if launch is None:
        launch = _compile_grouped_mxfp8_wgrad_fused(
            OUT_M,
            OUT_N,
            G,
            bm,
            bn,
            gm,
            xcd,
            gn,
            cbsz,
            blgp,
            out_fp16,
            pack=pack,
            preshuffle=preshuffle,
            split_k=split_k,
        )
        _GWG_FUSED_CACHE[fk] = launch
    return launch


def _canon_wgrad_targs(args, OUT_M, OUT_N, G, pack, pm, skew):
    """Synthetic args at `pm` tokens/group (dummy content, only shapes drive timing), groups evenly
    split or top-heavy per `skew`. Both operands carry the contraction dim here, so neither side can
    be reused from `args`; C is M-independent and every launch overwrites all of it, so the race
    scores into the caller's own output buffer instead of a second one."""
    dev, stream = args[2].device, args[14]
    M_c = G * pm
    K128 = M_c // 128
    a8_c = torch.randint(0, 127, (OUT_M, M_c), device=dev, dtype=torch.int8)
    b8_c = torch.randint(0, 127, (OUT_N, M_c), device=dev, dtype=torch.int8)
    a_raw_c = torch.randint(120, 128, (OUT_M, M_c // 32), device=dev, dtype=torch.uint8)
    b_raw_c = torch.randint(120, 128, (OUT_N, M_c // 32), device=dev, dtype=torch.uint8)
    a_sp_c, b_sp_c = _get_grouped_wgrad_workspace(OUT_M, OUT_N, K128, G, pack, dev, stream)
    n_ck = K128 // _PRESHUF_KT + G
    a_blocks = ceildiv(OUT_M, 64) * n_ck
    pre_grid = a_blocks + ((OUT_N + 255) // 256) * 4 * n_ck
    targs = (
        a8_c,
        b8_c,
        args[2],  # C (M-independent, fully overwritten by every launch)
        a_raw_c.view(torch.int32).reshape(-1),
        b_raw_c.view(torch.int32).reshape(-1),
        a_sp_c,
        b_sp_c,
        _canon_go(G, pm, skew, dev).view(torch.int32),
        args[8],  # split-K scratch (M-independent)
        M_c,
        K128,
        n_ck,
        a_blocks,
        pre_grid,
        stream,
    )
    return targs, args[2]


def _select_wgrad_cfg(cfg_key, OUT_M, OUT_N, G, bm, bn, cbsz, blgp, out_fp16, pack, split_k, args):
    """First-call race on synthetic canonical tensors; cache the winning swizzle per static shape
    (cfg_key, no M_total -> reused for every token count of the same tile grid).

    Both points share the canonical shape's scale workspace, and the packing is per-group, so each
    point's is refilled (one preshuffling launch) before it is used and only the GEMM is timed: the
    preshuffle dispatches thousands of tiny workgroups over the whole scale buffer, which would both
    dilute every ratio and hand the GEMM a perturbed L2 -- the very thing this race scores."""
    cached = _GWG_CFG_CACHE.get(cfg_key)
    if cached is not None:
        return cached
    _GWG_CFG_CACHE[cfg_key] = _GWG_DEFAULT_CFG  # any failure below leaves the default in place

    def _launch_of(cfg, preshuffle=False):
        return _get_wgrad_launch(
            OUT_M, OUT_N, G, bm, bn, *cfg, cbsz, blgp, out_fp16, pack, preshuffle, split_k
        )

    try:
        points = [
            _canon_wgrad_targs(args, OUT_M, OUT_N, G, pack, pm, skew) for pm, skew in _GWG_PM_CANON
        ]
        fill = _launch_of(_GWG_DEFAULT_CFG, preshuffle=True)
        base = _launch_of(_GWG_DEFAULT_CFG)
        refs = []
        for targs, out_view in points:
            fill(*targs)
            base(*targs)
            torch.cuda.synchronize()
            r = out_view.detach().clone().float()
            if not torch.isfinite(r.reshape(-1)[:1024]).all().item():
                raise RuntimeError("base cfg produced non-finite output")
            refs.append((r, float((r * r).sum().item()) or 1.0))
        _robust_time(base, points[0][0])  # ramp to the sustained-load clock before racing
    except Exception:
        return _GWG_DEFAULT_CFG

    best_cfg, best_ratio = _GWG_DEFAULT_CFG, 1.0
    for cfg in _GWG_CANDS:
        try:
            launch = _launch_of(cfg)
            rs, matched = [], True
            for (targs, out_view), (ref, ref_n) in zip(points, refs):
                fill(*targs)  # the other point's packing is in the shared workspace
                launch(*targs)
                torch.cuda.synchronize()
                o = out_view.detach().float()
                err = float(((o - ref) * (o - ref)).sum().item())
                # never adopt a cfg that drifts from the base at any point: a swizzle change moves
                # the split-K window, and a window over different tiles reduces in another order
                if not ((err / ref_n) < (2e-2**2) and torch.isfinite(o.reshape(-1)[:1024]).all().item()):
                    matched = False
                    break
                rs.append(_robust_ab_ratio(base, launch, targs))
            if not matched:
                continue
        except Exception:
            continue
        # Adopt only a cfg that clears the hysteresis at EVERY point, then keep the fastest such
        # cfg -- ties keep the earlier candidate, so the fixed list order decides.
        score = math.exp(sum(math.log(r) for r in rs) / len(rs))
        if max(rs) < _GWG_AT_MARGIN and score < best_ratio:
            best_cfg, best_ratio = cfg, score

    _GWG_CFG_CACHE[cfg_key] = best_cfg
    return best_cfg


def grouped_gemm_mxfp8_variable_k_flydsl_kernel(
    lhs: "torch.Tensor",
    lhs_scale: "torch.Tensor",
    rhs: "torch.Tensor",
    rhs_scale: "torch.Tensor",
    group_offs: "torch.Tensor",  # padded per-group M offsets [G+1]
    OUT_M: int,
    OUT_N: int,
    G: int,
    out_dtype: torch.dtype = torch.bfloat16,
    num_cu: "int | None" = -1,
    pack: int = 4,
    preshuffle: bool = True,
) -> "torch.Tensor":
    """FlyDSL MXFP8 grouped variable-K wgrad. Returns C [G, OUT_M, OUT_N].

    preshuffle=False skips the fused E8M0 scale preshuffle (assumes the cached workspace
    was populated by a prior preshuffle=True call on the same shape) so the GEMM main
    kernel can be timed alone; used by benches, not the training path.

    pack>1 packs `pack` consecutive contraction-K-blocks' E8M0 scales into one dword
    (op_sel immediate byte select), which folds the mainloop scale-load count and the
    scale workspace by PACK. Each group is packed from its own contraction start, so any
    128-aligned group_offs works (the preshuffle absorbs the per-group phase).
    """
    assert lhs.ndim == 2 and rhs.ndim == 2
    assert lhs.shape[0] == OUT_M and rhs.shape[0] == OUT_N
    M_total = lhs.shape[1]
    assert rhs.shape[1] == M_total
    assert M_total % 128 == 0
    out_fp16 = out_dtype == torch.float16
    cbsz = 1 if lhs.dtype == torch.float8_e5m2 else 0
    blgp = 1 if rhs.dtype == torch.float8_e5m2 else 0
    K128 = M_total // 128

    a_raw = (lhs_scale if lhs_scale.is_contiguous() else lhs_scale.contiguous()).view(torch.int32).reshape(-1)
    b_raw = (rhs_scale if rhs_scale.is_contiguous() else rhs_scale.contiguous()).view(torch.int32).reshape(-1)
    a8 = lhs.contiguous().view(torch.int8)
    b8 = rhs.contiguous().view(torch.int8)
    out = torch.empty((G, OUT_M, OUT_N), dtype=out_dtype, device=lhs.device)

    _go = group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)
    go = _go.view(torch.int32)

    stream = torch.cuda.current_stream()
    a_sp, b_sp = _get_grouped_wgrad_workspace(OUT_M, OUT_N, K128, G, pack, lhs.device, stream)

    a_ngrp = ceildiv(OUT_M, 64)
    b_ngrp = ((OUT_N + 255) // 256) * 4
    # Per-group KT-chunk slots. sum_g ceildiv(k_g, KT) <= K128//KT + G, so this bound holds
    # for every distribution (slots past the real count exit the preshuffle immediately).
    n_ck = K128 // _PRESHUF_KT + G
    a_blocks = a_ngrp * n_ck
    pre_grid = a_blocks + b_ngrp * n_ck

    bm, bn = 256, 256
    ncu = _num_cus()
    split_k = _wgrad_split_pays(OUT_M, OUT_N, G, bm, bn, ncu)
    # Split-K slice scratch (S_MAX-1 row bands at C's pitch), persistent per shape+device.
    ws = _wgrad_split_ws(OUT_M, OUT_N, G, lhs.device, out_dtype, BLOCK_M=bm, BLOCK_N=bn)
    args = (a8, b8, out, a_raw, b_raw, a_sp, b_sp, go, ws, M_total, K128, n_ck, a_blocks, pre_grid, stream)
    # Single universal variable-K kernel: chunk-local SSA accumulation, no balance detection.
    # The tile swizzle and the split-K window geometry are pure functions of (OUT_M, OUT_N, G, bm,
    # bn) plus the device CU count, so the cfg race keys on those alone and every token count of
    # the shape reuses its winner.
    cfg_key = (OUT_M, OUT_N, G, bm, bn, cbsz, blgp, out_fp16, pack, split_k, ncu)
    gm, xcd, gn = _select_wgrad_cfg(
        cfg_key, OUT_M, OUT_N, G, bm, bn, cbsz, blgp, out_fp16, pack, split_k, args
    )
    launch = _get_wgrad_launch(
        OUT_M, OUT_N, G, bm, bn, gm, xcd, gn, cbsz, blgp, out_fp16, pack, preshuffle, split_k
    )
    at_key = (OUT_M, OUT_N, M_total, G, cbsz, blgp, out_fp16, pack, preshuffle)
    entry = _GWG_AT_CACHE.get(at_key)
    if entry is None:
        entry = [launch, None]
        _GWG_AT_CACHE[at_key] = entry
    run_eager_or_capture(entry, args, 1)
    return out
