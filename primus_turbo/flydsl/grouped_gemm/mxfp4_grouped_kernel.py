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
"""FlyDSL MXFP4 (per-32-K E8M0 block-scaled) grouped GEMM for gfx950 (NT fwd/dgrad).

A [total_M, K] fp4 (groups along M), B [G, N, K] fp4, out [total_M, N] bf16;
``group_offs`` [G+1] int64 splits M. Reuses the dense mxfp4 whole-loop compute
(``MfmaScaleFp4.call_mxfp4_wholeloop`` + ``S2RLoaderFp4`` + ``StoreCPlain``) with the
fp8-grouped addressing (O(G) tile scan, per-group A row offset, per-expert B offset,
C store bounded to the group's tight end). E8M0 scales are repacked into the lane-
contiguous layout the whole-loop reads: A into per-group 256-aligned slabs (so the
tile's scale soffset stays 128-region aligned), B per expert.
"""

import gc

import torch

# isort: off
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    _lane_tbl_count_le,
    _lane_tbl_get,
    _lane_tbl_load,
    _lane_tbl_scan,
    _readfirstlane_i32,
    _readlane_i32,
    ceildiv,
    ceildiv_pow2,
    make_fp8_rebased_tensor_and_srd,
    wait_barrier,
    xcd_band_remap_pid,
    xcd_remap_pid,
)
from primus_turbo.flydsl.gemm.mxfp4_gemm_kernel import (
    _MXFP4_PRESHUF_BLK,
    _MXFP4_PRESHUF_FO,
    MfmaScaleFp4,
    S2RLoaderFp4,
    ScaleS2RPacked,
    StoreCPlain,
    _build_mxfp4_preshuffle_kernel_ab,
    _mxfp4_grp_from,
    _mxfp4_pack_cell,
    fp4_g2s_offsets,
)
from primus_turbo.flydsl.grouped_gemm.gemm_fp8_grouped_kernel import (
    _grouped_block_mn,
    _load_go,
    _wgrad_block_mn,
)
from primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel import run_eager_or_capture

# isort: on

_BLOCK = 256  # BLOCK_M = BLOCK_N = BLOCK_K
_PRESHUF_BLK = 256
_PRESHUF_NG = 4  # g bytes packed by one preshuffle thread
_PRESHUF_ND = 4  # (r_region, K sub-block) cells packed by one preshuffle thread
_PRESHUF_FO = _PRESHUF_NG * _PRESHUF_ND  # output dwords per thread


def _build_grouped_mxfp4_ab_preshuffle(K128: int, G: int, N: int, k128_rd: int = None):
    """Merged A-slab + B-per-expert scale preshuffle in ONE launch (matches the fp8/dense
    single-preshuffle structure -> one fewer in-stream kernel launch + gap per grouped GEMM,
    a bigger relative win on the small/short-K shapes). Blocks [0, a_grid) do the A slab
    (mode 0, inline a_pre scan over GO); [a_grid, ...) do the B per-expert (mode 1). The
    two paths are computed then segment-selected (SGPR rsrc via arith.select, values via
    select) -- no per-thread divergence. ``k128_rd`` real read + 256-block mask = zero pad,
    no host F.pad."""
    _KRD = K128 if k128_rd is None else k128_rd
    # ScaleS2RPacked interleaves four 64-row groups at a time, so its physical
    # row extent must be a 256-multiple even though the FP4 operand keeps the
    # real N row stride. Without this padding, the last partial quartet leaves
    # holes in one expert's workspace and spills stores into the next expert.
    N_SCALE = ceildiv(N, 256) * 256
    n_sub, nd, KK = 2, _PRESHUF_ND, K128 // 2
    n_rr = nd // n_sub
    b_dwords_pe = N_SCALE * K128 // _PRESHUF_FO
    # One thread per (wi, kk, r) cell and 16 r per cell => a wave spans 64/16 cells, so wi --
    # wave-uniform when a thread emitted a single dword -- now takes up to _NWI values.
    _NWI = 1 + ceildiv(64 // 16 - 1, KK)

    @flyc.kernel(known_block_size=[_PRESHUF_BLK, 1, 1])
    def kern(
        a_raw: fx.Tensor,
        a_out: fx.Tensor,
        b_raw: fx.Tensor,
        b_out: fx.Tensor,
        go_out: fx.Tensor,
        total_M: fx.Int32,
        slab_rows: fx.Int32,
        a_grid: fx.Int32,
    ):
        I32 = fx.Int32
        a_rin = buffer_ops.create_buffer_resource(
            a_raw, max_size=False, num_records_bytes=total_M * I32(_KRD) * 4
        )
        a_rout = buffer_ops.create_buffer_resource(
            a_out, max_size=False, num_records_bytes=slab_rows * I32(K128) * 4
        )
        b_rin = buffer_ops.create_buffer_resource(
            b_raw, max_size=False, num_records_bytes=I32(G * N * _KRD) * 4
        )
        b_rout = buffer_ops.create_buffer_resource(
            b_out, max_size=False, num_records_bytes=I32(G * N_SCALE * K128) * 4
        )
        bid = rocdl.readfirstlane(T.i32, fx.block_idx.x)
        is_b = bid >= a_grid
        local = arith.select(is_b, bid - a_grid, bid)
        lane_id = fx.thread_idx.x % 64
        gid_all = local * I32(_PRESHUF_BLK) + fx.thread_idx.x
        rin = arith.select(is_b, b_rin, a_rin)
        rout = arith.select(is_b, b_rout, a_rout)

        # One thread owns a whole (wi, kk, r) cell -- both r_region halves and both K
        # sub-blocks on top of the 4 g bytes: gather becomes dwordx2, scatter dwordx4, and
        # the index/group-scan math amortises over 16 output dwords instead of 4. A workgroup
        # now spans one full 128B source line instead of 32B of it (fewer redundant re-fetches
        # by workgroups on other XCDs' L2 slices).
        b_expert = gid_all // I32(b_dwords_pe)
        a_total = slab_rows * I32(K128) // I32(_PRESHUF_FO)
        gid = arith.select(is_b, gid_all - b_expert * I32(b_dwords_pe), gid_all)
        total = arith.select(is_b, I32(b_dwords_pe), a_total)
        r = gid % I32(16)
        e2 = gid // I32(16)
        kk = e2 % I32(KK)
        wi = e2 // I32(KK)
        k128 = kk * I32(n_sub)  # the thread's n_sub K sub-blocks are adjacent source dwords
        _blk = ((wi * I32(KK) + kk) * I32(64) + r) * I32(nd)
        base = arith.select(is_b, b_expert * I32(N_SCALE * K128) + _blk, _blk)

        # A: lane-resident group scan (owning group -> tight source rows rd0..rd_end). Lane g
        # owns group g, so ONE wave inclusive scan of the per-group 64-row-block count plus a
        # few O(1) lookups cover the wave (wi spans at most _NWI values inside a wave). Replaces
        # the G-wide serial compare chain that made this preshuffle VALU-issue bound.
        go_rs = buffer_ops.create_buffer_resource(go_out, max_size=False, num_records_bytes=(G + 1) * 8)
        _go0 = _lane_tbl_load(go_rs, lane_id, G + 1, stride=2)
        _go1 = _lane_tbl_load(go_rs, lane_id, G + 1, stride=2, first=1)
        _own = [lane_id + I32(64 * c) < I32(G) for c in range_constexpr(len(_go0))]
        _nb = [
            arith.select(_own[c], ceildiv_pow2(_go1[c] - _go0[c], 256) * I32(4), I32(0))
            for c in range_constexpr(len(_go0))
        ]
        _nbs_end = _lane_tbl_scan(_nb)  # entry g = 64-row groups owned by groups <= g
        _nbs = [_nbs_end[c] - _nb[c] for c in range_constexpr(len(_nb))]
        _ngrp = _readlane_i32(_nbs_end[-1], 63)

        def _a_rows(q):
            gq = _lane_tbl_count_le(_nbs_end, q)
            r0 = _lane_tbl_get(_go0, gq) + (q - _lane_tbl_get(_nbs, gq)) * I32(64)
            return r0, _lane_tbl_get(_go1, gq)

        _wi_u = _readfirstlane_i32(wi)
        _rows_q = [_a_rows(I32(2) * _wi_u + I32(q)) for q in range_constexpr(2 * _NWI)]
        _dwi = wi - _wi_u
        rd_base = b_expert * I32(N)  # B source row base
        in_grid = arith.select(is_b, (gid < I32(b_dwords_pe)) & (b_expert < I32(G)), gid < a_total) & (
            gid < total
        )

        dws = []
        for r_region in range_constexpr(n_rr):
            rd0, rd_end = _rows_q[r_region]
            for q in range_constexpr(1, _NWI):
                _hit = _dwi == I32(q)
                rd0 = arith.select(_hit, _rows_q[2 * q + r_region][0], rd0)
                rd_end = arith.select(_hit, _rows_q[2 * q + r_region][1], rd_end)
            grp_a = _mxfp4_grp_from(wi, r_region, 0)
            grp_b = _mxfp4_grp_from(wi, r_region, 1)
            # slab-pad groups past the last expert: nothing to read
            okc = arith.select(is_b, in_grid, in_grid & (grp_a < _ngrp))
            for t in range_constexpr(nd):
                b_row = grp_b * I32(64) + I32(t * 16) + r
                row = arith.select(is_b, rd_base + b_row, rd0 + I32(t * 16) + r)
                valid = okc & arith.select(is_b, b_row < I32(N), row < rd_end)
                v = Vec(
                    buffer_ops.buffer_load(
                        rin, row * I32(_KRD) + k128, vec_width=n_sub, dtype=T.i32, mask=valid
                    )
                )
                if const_expr(_KRD % n_sub != 0):
                    # odd real K128: the pair's tail sub-block is the next row's first dword
                    v = Vec.from_elements(
                        [v[0]]
                        + [
                            arith.select(k128 + I32(j) < I32(_KRD), v[j], I32(0))
                            for j in range_constexpr(1, n_sub)
                        ]
                    )
                dws.append(v)
        words = _mxfp4_pack_cell(dws, n_sub, nd, _PRESHUF_NG)
        # store ALL in-range blocks: invalid/pad blocks got words=0 from the masked reads,
        # so slab-pad / 256-pad regions are written 0 (matches the split a_pre_shuf).
        for g in range_constexpr(_PRESHUF_NG):
            buffer_ops.buffer_store(Vec.from_elements(words[g]), rout, base + I32(g * 64), mask=gid < total)

    return kern


def _build_grouped_mxfp4_nt_kernel(
    K, G, N, group_m=4, num_xcds=8, group_n=0, wlv=10, elgk=9, out_fp16=False, k_real=None, xcd_span=16
):
    """Grouped MXFP4 NT (out = a @ b^T), per-group A rows + per-expert B, whole-loop compute.

    K = 256-rounded tile/scale extent (the tiny E8M0 scale is zero-padded to it so the
    preshuffle packs whole 256-blocks). ``k_real`` (<= K, 128-multiple) = the operands' TRUE
    contraction (row stride, no operand pad). When k_real%256==128 the loop runs
    k_real//256 full 256-blocks + a trailing block whose s=1 sub-step is past the real K
    (its scale is the zero pad, so only s=0 contributes) -- zero operand copy, and that
    dead sub-step and everything feeding it are dropped rather than executed."""
    BLOCK_M = BLOCK_N = BLOCK_K = _BLOCK
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    swizzle = True
    _KR = K if k_real is None else k_real  # operand true contraction (128-multiple)
    assert K % 256 == 0 and _KR % 128 == 0
    KI = _KR // BLOCK_K  # FULL 256-blocks over the REAL K
    _K128 = (_KR // 128) % 2  # 1 => trailing 128-K block, handled by scale-pad-zero below
    # Trailing 128-K: round KI up so the last 256-block is fully pipelined into the do-while,
    # and tell the whole-loop that this block's s=1 sub-step is past the contraction (it would
    # only multiply the zero-padded scale) so it drops that sub-step and the prefetch stream
    # that exists solely to feed a phase after it.
    KI_LOOP = KI + 1 if _K128 else KI
    NABUF, NBB, OCC = 2, 2, 2  # fwd waves_per_eu=2: hide the latency-bound short-K/small-tile GEMM
    N_SUB = BLOCK_K // 128
    BPR = BLOCK_K // 2
    KSTEP = BPR
    K2 = _KR // 2  # operand row stride (bytes) = real K (no operand K-pad)
    N_TILES_A = BLOCK_M // 32
    LDS_BN_HALF = BLOCK_N // 2
    N_TILES_BH = LDS_BN_HALF // 32
    LDS_ROW_STRIDE = BPR
    a_lds_size = BLOCK_M * LDS_ROW_STRIDE
    bh_lds_size = LDS_BN_HALF * LDS_ROW_STRIDE
    _ROWS_PER_STEP = 64 // (BPR // 16) * (256 // 64)
    N_LDS_STEPS_A = BLOCK_M // _ROWS_PER_STEP
    N_LDS_STEPS_BH = LDS_BN_HALF // _ROWS_PER_STEP
    _PRELL, _NSCBUF = 2, 2
    K128 = K // 128
    N_SCALE = ceildiv(N, 256) * 256
    _SCBUF = 4 * 4 * (BLOCK_K // 128) * 64
    _SCW = 4 * N_SUB * 64
    NBK = ceildiv(N, BLOCK_N)  # n_blocks
    _NV = N if (N % BLOCK_N != 0) else None  # non-256 N: mask store cols >= N (no host N-pad)
    # Boundary N-block skip: when the last n-block holds <= LDS_BN_HALF valid columns its BR
    # half is all free-dim padding, so that half's MFMAs and its B g2s are dead work (accR
    # keeps its zeros and the store is column-masked). The whole-loop then runs a second,
    # BR-less body selected by a wave-uniform SGPR flag.
    _HALF_N = (N % BLOCK_N != 0) and (N % BLOCK_N <= LDS_BN_HALF)

    _anns = {f"A_lds{i}": fx.Array[fx.Float8E4M3FN, a_lds_size, 16] for i in range_constexpr(NABUF)}
    for _b in range_constexpr(NBB):
        _anns[f"BL_lds{_b}"] = fx.Array[fx.Float8E4M3FN, bh_lds_size, 16]
    for _b in range_constexpr(NBB):
        _anns[f"BR_lds{_b}"] = fx.Array[fx.Float8E4M3FN, bh_lds_size, 16]
    for _b in range_constexpr(_NSCBUF):
        _anns[f"SC_lds{_b}"] = fx.Array[fx.Int32, _SCBUF, 16]
    SS = fx.struct(type("SSFp4Grp", (), {"__annotations__": _anns}))

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kern(
        A: fx.Tensor,  # a_row [total_M, K/2] fp4 (flat int8)
        B_T: fx.Tensor,  # b_row [G, N, K/2] fp4 (flat int8)
        C: fx.Tensor,  # out [total_M, N]
        A_scale: fx.Tensor,  # packed A slabs (int32)
        B_scale: fx.Tensor,  # packed B per-expert (int32)
        GO: fx.Tensor,  # tight offs (int32 view int64 [G+1])
        c_m: fx.Int32,  # total_M
        c_n: fx.Int32,  # N
        slab_rows: fx.Int32,  # padded A-slab rows
    ):
        F8 = fx.Float8E4M3FN.ir_type
        lds = fx.SharedAllocator().allocate(SS).peek()
        A_buf = [getattr(lds, f"A_lds{i}") for i in range_constexpr(NABUF)]
        BL_buf = [getattr(lds, f"BL_lds{i}") for i in range_constexpr(NBB)]
        BR_buf = [getattr(lds, f"BR_lds{i}") for i in range_constexpr(NBB)]
        SC_buf = [getattr(lds, f"SC_lds{b}") for b in range_constexpr(_NSCBUF)]
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2
        I32 = fx.Int32

        # ---- tile-independent setup (operand SRDs/loaders built per-tile below, rebased) ----
        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_BH, packed=True, wlv=wlv, elgk=elgk)
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, _KR, N_LDS_STEPS_A, BPR, swizzle=swizzle)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, _KR, N_LDS_STEPS_BH, BPR, swizzle=swizzle)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, LDS_ROW_STRIDE, swizzle=swizzle)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_BH, LDS_ROW_STRIDE, swizzle=swizzle)
        sa_s2r = ScaleS2RPacked(A_scale, slab_rows, K, 4)
        sb_s2r = ScaleS2RPacked(B_scale, I32(N_SCALE * G), K, 4)
        wave_m_off = wave_m * (N_TILES_A * 16)
        wave_n_off = wave_n * (N_TILES_BH * 16)

        a_base6 = [
            [a_s2r.base_addr(A_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NABUF)
        ]
        bl_base6 = [
            [b_s2r.base_addr(BL_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)
        ]
        br_base6 = [
            [b_s2r.base_addr(BR_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)
        ]

        def _gbase(buf):
            v = fx.Int32(fx.ptrtoint(buf.ptr)) + fx.Int32(wave_id) * fx.Int32(1024)
            return rocdl.readfirstlane(T.i32, v)

        abase6 = [_gbase(A_buf[b]) for b in range_constexpr(NABUF)]
        blbase6 = [_gbase(BL_buf[b]) for b in range_constexpr(NBB)]
        brbase6 = [_gbase(BR_buf[b]) for b in range_constexpr(NBB)]
        gl_a6 = [fx.Int32(gl_off_a[st]) for st in range_constexpr(N_LDS_STEPS_A)]
        gl_b6 = [fx.Int32(gl_off_b[st]) for st in range_constexpr(N_LDS_STEPS_BH)]
        scv6 = fx.Int32(0x7F7F7F7F)
        sc_rb6 = [
            fx.ptrtoint(
                fx.add_offset(SC_buf[b].ptr, fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW) + lane_id))
            )
            for b in range_constexpr(_NSCBUF)
        ]
        sc_gb6 = [
            rocdl.readfirstlane(
                T.i32,
                fx.Int32(
                    fx.ptrtoint(
                        fx.add_offset(SC_buf[b].ptr, fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW)))
                    )
                ),
            )
            for b in range_constexpr(_NSCBUF)
        ]
        _scrsa_v = sa_s2r.rsrc
        _scrsb_v = sb_s2r.rsrc
        sc_voff6 = lane_id * fx.Int32(8 * N_SUB)

        def _scsoff(base, extra):
            grp = (base + fx.Int32(extra)) // fx.Int32(64)
            return rocdl.readfirstlane(
                T.i32, (grp * fx.Int32(K128) + fx.Int32(_PRELL * N_SUB)) * fx.Int32(256)
            )

        # ---- lane-resident group scan: pid -> (group_idx, local tile, m_start, m_end, a_pre) ----
        # Lane g owns group g, so the tile prefix and the A-scale slab prefix both come out
        # of ONE wave inclusive scan of the per-group 256-row block count. That replaces the
        # serial G-wide compare tree (which needs 2*(G+1) live scalars and one group-offset
        # load per group) with two lane-table loads, one scan and per-tile readlanes.
        go_rs = buffer_ops.create_buffer_resource(GO, max_size=False, num_records_bytes=(G + 1) * 8)
        _go0 = _lane_tbl_load(go_rs, lane_id, G + 1, stride=2)
        _go1 = _lane_tbl_load(go_rs, lane_id, G + 1, stride=2, first=1)
        _own = [lane_id + I32(64 * c) < I32(G) for c in range_constexpr(len(_go0))]
        _nb = [
            arith.select(_own[c], ceildiv_pow2(_go1[c] - _go0[c], BLOCK_M), I32(0))
            for c in range_constexpr(len(_go0))
        ]
        _nbs_end = _lane_tbl_scan(_nb)
        _tcs_end = [v * I32(NBK) for v in _nbs_end]  # entry g = tiles owned by groups <= g
        _tcs = [_tcs_end[c] - _nb[c] * I32(NBK) for c in range_constexpr(len(_nb))]
        _sas = [(_nbs_end[c] - _nb[c]) * I32(4) for c in range_constexpr(len(_nb))]
        total_tiles = _readlane_i32(_tcs_end[-1], 63)
        bid = fx.block_idx.x
        # non-persistent grid: WGs past total_tiles hardware-exit (scf.if cannot
        # carry the Python-object loader state, so guard via s_endpgm). The grid is an
        # upper bound (worst-case per-group round-up), so the exit MUST test the raw
        # block id: the XCD remap is then a bijection over the LIVE tiles only, which
        # spreads the over-launched WGs round-robin over the XCDs instead of stacking
        # the whole dead tail onto the last one (it owns the top of the remapped range).
        _llvm.inline_asm(
            None,
            [bid.ir_value(), arith._to_raw(total_tiles)],
            "s_cmp_lt_u32 $0, $1\n\ts_cbranch_scc1 1f\n\ts_endpgm\n\t1:",
            "s,s,~{scc},~{memory}",
            has_side_effects=True,
        )
        # Band-cyclic (not contiguous) XCD partition: an XCD owns every 8th run of xcd_span
        # M-blocks, keeping the intra-run L2 reuse of a contiguous remap while still drawing
        # tiles from the whole token range. Needed because per-tile cost is NOT uniform under
        # skewed routing -- a contiguous partition would strand a small-expert's cheap tail
        # tiles on one static XCD (bid % 8), which then sets the makespan.
        pid = xcd_band_remap_pid(bid, total_tiles, num_xcds, xcd_span * NBK)
        group_idx = _lane_tbl_count_le(_tcs_end, pid)
        tile_start = _lane_tbl_get(_tcs, group_idx)
        a_pre_g = _lane_tbl_get(_sas, group_idx)
        m_start = _lane_tbl_get(_go0, group_idx)
        m_end = _lane_tbl_get(_go1, group_idx)
        local = pid - tile_start
        bm, bn = _grouped_block_mn(local, m_start, m_end, NBK, BLOCK_M, group_m, group_n)

        m_row = m_start + bm * I32(BLOCK_M)  # tight A/C row base
        # Fold the tile's A row base + per-expert/col B base into the operand SRDs in int64:
        # large-G MoE (group_idx*c_n*K2) / large total_M (m_row*K2) exceed 2^31, which the
        # whole-loop's int32 voffset/soffset cannot reach. Residual offsets stay intra-tile.
        a_base_e = arith.index_cast(T.index, m_row) * arith.index(K2)
        b_base_e = (
            arith.index_cast(T.index, group_idx) * arith.index_cast(T.index, c_n)
            + arith.index_cast(T.index, bn) * arith.index(BLOCK_N)
        ) * arith.index(K2)
        a_nrec = (arith.index_cast(T.index, c_m) - arith.index_cast(T.index, m_row)) * arith.index(K2)
        b_nrec = arith.index(G) * arith.index_cast(T.index, c_n) * arith.index(K2) - b_base_e
        gA, rsrc_a = make_fp8_rebased_tensor_and_srd(A, F8, a_base_e, a_nrec)
        gB, rsrc_b = make_fp8_rebased_tensor_and_srd(B_T, F8, b_base_e, b_nrec)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8, wave_id)
        bl_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8, wave_id)
        br_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8, wave_id)
        a_off = I32(0)  # A/B tile+expert bases folded into the SRDs above; only the LDS-half
        bl_off = I32(0)  # column shift (br) survives as an int32-safe intra-tile residual.
        br_off = I32(LDS_BN_HALF) * K2
        # A-scale slab row base (256-aligned): a_pre_g*64 + bm*256 + wave off
        sa_b = a_pre_g * I32(64) + bm * I32(BLOCK_M) + I32(wave_m_off)
        sbl_b = bn * I32(BLOCK_N) + I32(wave_n_off)
        sbr_b = bn * I32(BLOCK_N) + I32(LDS_BN_HALF) + I32(wave_n_off)
        b_exp_bytes = group_idx * I32(N_SCALE * K128 * 4)  # padded per-expert B-scale base (bytes)

        # ---- fill operand buffers ----
        for _pp in range_constexpr(0, _PRELL):
            if const_expr(KI_LOOP > _pp):
                a_g2s.load(A_buf[_pp], a_off + _pp * KSTEP)
        for _pp in range_constexpr(0, _PRELL):
            if const_expr(KI_LOOP > _pp):
                bl_g2s.load(BL_buf[_pp], bl_off + _pp * KSTEP)
                br_g2s.load(BR_buf[_pp], br_off + _pp * KSTEP)
        _llvm.inline_asm(
            res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)", constraints="", has_side_effects=True
        )
        wait_barrier(0)

        accL = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)
        accR = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)
        soff6_a = rocdl.readfirstlane(T.i32, a_off + fx.Int32(_PRELL * KSTEP))
        soff6_bl = rocdl.readfirstlane(T.i32, bl_off + fx.Int32(_PRELL * KSTEP))
        soff6_br = rocdl.readfirstlane(T.i32, br_off + fx.Int32(_PRELL * KSTEP))
        _sc1 = _scsoff(sa_b, 64)
        _wia = sa_b // I32(128)
        _soa = rocdl.readfirstlane(T.i32, _wia * I32(K128) * I32(512))
        _sc3 = rocdl.readfirstlane(T.i32, b_exp_bytes + _scsoff(sbr_b, 0))
        _wib = (sbl_b // I32(256)) * I32(2) + (sbl_b % I32(256)) // I32(64)
        _sob = rocdl.readfirstlane(T.i32, b_exp_bytes + _wib * I32(K128) * I32(512))
        sc_soff06 = [_soa, _sc1, _sob, _sc3]
        _half_n = None
        if const_expr(_HALF_N):
            _half_n = _readfirstlane_i32(arith.select(bn == I32(NBK - 1), I32(1), I32(0)))
        accL, accR = mfma.call_mxfp4_wholeloop(
            a_base6,
            bl_base6,
            br_base6,
            a_s2r.tile_stride,
            b_s2r.tile_stride,
            abase6,
            blbase6,
            brbase6,
            gl_a6,
            gl_b6,
            rsrc_a,
            rsrc_b,
            fx.Int32(KSTEP),
            scv6,
            accL,
            accR,
            N_SUB,
            N_LDS_STEPS_A,
            N_LDS_STEPS_BH,
            fx.Int32((KI_LOOP // 2) * 2),
            soff6_a,
            soff6_bl,
            soff6_br,
            sc_rb6,
            sc_gb6,
            _scrsa_v,
            _scrsb_v,
            sc_voff6,
            sc_soff06,
            ki=KI_LOOP,
            sc_buf_stride=(_SCBUF * 4),
            half_n=_half_n,
            half_k=bool(_K128),
        )
        base_row = m_row + I32(wave_m_off)
        base_col_l = bn * I32(BLOCK_N) + I32(wave_n_off)
        # store bounded to the group's tight end: StoreCPlain's SRD num_records =
        # m_end*c_n -> partial-tile rows >= m_end (next group) HW-drop.
        store_c = StoreCPlain(C, m_end, c_n, mfma.idx, N_TILES_A, N_TILES_BH, _out_ty)
        store_c.store(accL, base_row, base_col_l, n_valid=_NV)
        base_col_r = bn * I32(BLOCK_N) + I32(LDS_BN_HALF) + I32(wave_n_off)
        store_c.store(accR, base_row, base_col_r, n_valid=_NV)

    _pt = {"passthrough": [["amdgpu-agpr-alloc", "256"]]}
    attrs = {"rocdl.flat_work_group_size": "256,256", "rocdl.waves_per_eu": OCC, **_pt}
    return kern, attrs, NBK


_GMXFP4_LAUNCH_CACHE: dict = {}
_GMXFP4_WS_CACHE: dict = {}
_GMXFP4_AT_CACHE: dict = {}  # (total_M, N, K, G, gm, xcd, gn, out_fp16) -> [raw_launch, compiled]
# Fixed tile-blocking config per path: NT is (group_m, num_xcds, group_n, xcd_span), wgrad is
# the first three. NT's tiles-per-group is skew-sensitive so it needs the band-cyclic XCD
# partition; wgrad's skew lives in its per-group contraction length instead, so it stays on
# plain hardware round-robin (xcd=1) and takes L2 reuse from the group-major gn=4 band.
_GMXFP4_NT_CFG = (2, 8, 0, 16)
_GMXFP4_WGRAD_CFG = (2, 1, 4)
# JIT compile-cache bound: each distinct shape compiles one FlyDSL kernel (GPU code object).
# Real MoE uses a handful of shapes; a broad test sweep (~480 shapes) accumulates enough
# code objects to exhaust memory -> drop the caches (and gc the modules) past this cap. A
# real workload stays well under it, so its kernels are never evicted.
_GMXFP4_CACHE_CAP = 32


def _bound_caches(*caches):
    if any(len(c) > _GMXFP4_CACHE_CAP for c in caches):
        for c in caches:
            c.clear()
        gc.collect()


def _compile_grouped_mxfp4_nt_fused(K, G, N, gm, xcd, gn, wlv, elgk, out_fp16, k_real=None, span=16):
    K128 = K // 128
    N_SCALE = ceildiv(N, 256) * 256
    k128_rd = (K if k_real is None else k_real) // 128  # real raw K128 (scale not host-padded)
    ab_pre_shuf = _build_grouped_mxfp4_ab_preshuffle(K128, G, N, k128_rd)  # merged A+B, 1 launch
    gemm_k, attrs, NBK = _build_grouped_mxfp4_nt_kernel(
        K,
        G,
        N,
        group_m=gm,
        num_xcds=xcd,
        group_n=gn,
        wlv=wlv,
        elgk=elgk,
        out_fp16=out_fp16,
        k_real=k_real,
        xcd_span=span,
    )
    b_pre_grid = ceildiv(G * N_SCALE * K128, _PRESHUF_FO * _PRESHUF_BLK)

    @flyc.jit
    def launch(
        a8: fx.Tensor,
        b8: fx.Tensor,
        C: fx.Tensor,
        a_raw: fx.Tensor,
        b_raw: fx.Tensor,
        a_sp: fx.Tensor,
        b_sp: fx.Tensor,
        GO: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        slab_rows: fx.Int32,
        a_pre_grid: fx.Int32,
        grid_upper: fx.Int32,
        stream: fx.Stream,
    ):
        ab_pre_shuf(a_raw, a_sp, b_raw, b_sp, GO, c_m, slab_rows, a_pre_grid).launch(
            grid=(a_pre_grid + b_pre_grid, 1, 1), block=(_PRESHUF_BLK, 1, 1), stream=stream
        )
        gemm_k(a8, b8, C, a_sp, b_sp, GO, c_m, c_n, slab_rows, value_attrs=attrs).launch(
            grid=(grid_upper, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch, NBK


def _get_grouped_mxfp4_ws(total_M, N, K128, G, device):
    # M-generic workspace: key on the static shape (no total_M) and grow the A-slab
    # buffer only when a larger total_M arrives. The kernel is passed the live slab_rows
    # (a_pre_grid derives from it), so a larger cached buffer is safe -- only the needed
    # prefix is written/read. Keeping total_M out of the key stops per-total_M churn from
    # tripping _bound_caches and evicting the compiled kernels (mirrors #419 for mxfp8).
    slab_rows = (ceildiv(total_M, 256) + G) * 256  # padded A-slab upper bound for this call
    n_scale = ceildiv(N, 256) * 256
    key = (N, K128, G, device)
    e = _GMXFP4_WS_CACHE.get(key)
    if e is None or e[2] < slab_rows:
        a_sp = torch.empty(slab_rows * K128, dtype=torch.int32, device=device)
        b_sp = e[1] if e is not None else torch.empty(G * n_scale * K128, dtype=torch.int32, device=device)
        e = (a_sp, b_sp, slab_rows)
        _GMXFP4_WS_CACHE[key] = e
    return e[0], e[1], slab_rows


def grouped_gemm_mxfp4_flydsl_kernel(
    a, a_scale, b, b_scale, group_offs, N, K, group_offs_out=None, out_dtype=torch.bfloat16, num_cu=-1
):
    """FlyDSL MXFP4 grouped NT GEMM (fwd / dgrad). a [total_M, K/2] fp4, b [G, N, K/2] fp4,
    a_scale [total_M, K/32] / b_scale [G, N, K/32] canonical E8M0. Returns C [total_M, N]."""
    assert a.ndim == 2 and b.ndim == 3
    total_M = int(a.shape[0])
    G = int(b.shape[0])
    out_fp16 = out_dtype == torch.float16
    dev = a.device
    N_out = N  # true free dim to return

    # ZERO host pad / ZERO torch copies. Free dim N: kernel tiles the real N + masks store
    # cols >= N. Contraction K: operands stay real (k_real row stride); the whole-loop runs
    # k_real//256 full 256-blocks + a 128-tail. The tiny E8M0 SCALE is zero-padded to 256
    # entirely INSIDE the preshuffle (k128_rd real read + 256-block mask) -- no F.pad.
    k_real = K
    K256 = (K + 255) // 256 * 256
    au = a.contiguous().view(torch.uint8)  # [total_M, k_real/2] -- real K
    asu = a_scale.contiguous().view(torch.uint8)  # [total_M, k_real/32] -- real K
    bu = b.contiguous().view(torch.uint8)  # [G, N, k_real/2]
    bsu = b_scale.contiguous().view(torch.uint8)  # [G, N, k_real/32]
    K = K256
    K128 = K // 128

    a_raw = asu.contiguous().view(torch.int32).reshape(-1)
    b_raw = bsu.contiguous().view(torch.int32).reshape(-1)
    # Keep the fp4 operands multi-dim (do NOT flatten to 1D): a large-G MoE B holds
    # G*N*K/2 > 2^31 int8s, and flydsl packs a 1D tensor's single dim as int32 (host
    # CABI overflow). Multi-dim keeps every dim/stride int32; the kernel addresses via
    # the rebased base (make_fp8_rebased_tensor_and_srd), independent of the shape.
    a8 = au.contiguous().view(torch.int8)
    b8 = bu.contiguous().view(torch.int8)
    # 2D C (NOT flattened): StoreCPlain re-bases per row band from C's base + c_n, so the
    # shape only needs each dim int32; a 1D total_M*N view overflows the CABI for large M*N.
    out = torch.empty((total_M, N), dtype=out_dtype, device=dev)

    go = (group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)).view(torch.int32)
    a_sp, b_sp, slab_rows = _get_grouped_mxfp4_ws(total_M, N, K128, G, dev)

    n_blocks = (N + 255) // 256
    grid_upper = (ceildiv(total_M, 256) + G) * n_blocks
    a_pre_grid = ceildiv(slab_rows * K128, _PRESHUF_FO * _PRESHUF_BLK)

    stream = torch.cuda.current_stream()
    wlv, elgk = 10, 9
    args = (
        a8,
        b8,
        out,
        a_raw,
        b_raw,
        a_sp,
        b_sp,
        go,
        total_M,
        N,
        slab_rows,
        a_pre_grid,
        grid_upper,
        stream,
    )

    def _entry(cfg):
        gm, xcd, gn, span = cfg
        lk = (K, G, N, gm, xcd, gn, span, wlv, elgk, out_fp16, k_real)
        ent = _GMXFP4_LAUNCH_CACHE.get(lk)
        if ent is None:
            ent = _compile_grouped_mxfp4_nt_fused(
                K, G, N, gm, xcd, gn, wlv, elgk, out_fp16, k_real=k_real, span=span
            )
            _GMXFP4_LAUNCH_CACHE[lk] = ent
        # M-generic launch (total_M is a runtime arg, not compiled in) -> key on the static
        # shape only, reuse the compiled object for every total_M. Dropping total_M here stops
        # per-total_M entries from evicting the cache (mirrors #419's mxfp8 M-decoupling).
        atk = (N, K, G, gm, xcd, gn, span, out_fp16, k_real)  # same K256 diff real K must not collide
        e2 = _GMXFP4_AT_CACHE.get(atk)
        if e2 is None:
            e2 = [ent[0], None]
            _GMXFP4_AT_CACHE[atk] = e2
        return e2

    run_eager_or_capture(_entry(_GMXFP4_NT_CFG), args, 1)
    _bound_caches(_GMXFP4_LAUNCH_CACHE, _GMXFP4_AT_CACHE, _GMXFP4_WS_CACHE)
    return out[:, :N_out] if N_out != N else out


# ── WGRAD (variable-K TN via NT compute): C[g] (OUT_M, OUT_N) = lhs[:, g] @ rhs[:, g]^T,
# contraction = per-group padded M. lhs [OUT_M, M_total/2] / rhs [OUT_N, M_total/2] fp4;
# scales whole-tensor (rows OUT_M/OUT_N are 256-tiled -> no per-group slab). The whole-loop
# runs a RUNTIME nval = M_g/256 (even; balanced 256-aligned groups). ──


def _build_grouped_mxfp4_wgrad_kernel(
    OUT_M, OUT_N, G, M_total, group_m=4, num_xcds=8, group_n=0, wlv=10, elgk=9, out_fp16=False
):
    BLOCK_M = BLOCK_N = BLOCK_K = _BLOCK
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    swizzle = True
    NABUF, NBB, OCC = 2, 2, 1  # wgrad keeps occ=1 (feed-bound; occ measured non-lever for wgrad)
    N_SUB = BLOCK_K // 128
    BPR = BLOCK_K // 2
    KSTEP = BPR
    M2 = M_total // 2  # operand row stride (bytes) = full contraction
    N_TILES_A = BLOCK_M // 32
    LDS_BN_HALF = BLOCK_N // 2
    N_TILES_BH = LDS_BN_HALF // 32
    LDS_ROW_STRIDE = BPR
    a_lds_size = BLOCK_M * LDS_ROW_STRIDE
    bh_lds_size = LDS_BN_HALF * LDS_ROW_STRIDE
    _ROWS_PER_STEP = 64 // (BPR // 16) * (256 // 64)
    N_LDS_STEPS_A = BLOCK_M // _ROWS_PER_STEP
    N_LDS_STEPS_BH = LDS_BN_HALF // _ROWS_PER_STEP
    _PRELL, _NSCBUF = 2, 2
    K128m = M_total // 128  # scale packed row stride (contraction blocks)
    _SCBUF = 4 * 4 * (BLOCK_K // 128) * 64
    _SCW = 4 * N_SUB * 64
    _SCVSTEP = 64 * (2 * N_SUB) * 4  # scale byte advance per 256-K iter (whole-loop internal)
    N_BLOCKS_M = ceildiv(OUT_M, BLOCK_M)
    N_BLOCKS_N = ceildiv(OUT_N, BLOCK_N)
    TILES_PER_GROUP = N_BLOCKS_M * N_BLOCKS_N
    _NV = OUT_N if (OUT_N % BLOCK_N != 0) else None  # non-256 OUT_N: mask store cols >= OUT_N
    _HALF_N = (OUT_N % BLOCK_N != 0) and (OUT_N % BLOCK_N <= LDS_BN_HALF)  # see the NT kernel
    TOTAL = G * TILES_PER_GROUP

    _anns = {f"A_lds{i}": fx.Array[fx.Float8E4M3FN, a_lds_size, 16] for i in range_constexpr(NABUF)}
    for _b in range_constexpr(NBB):
        _anns[f"BL_lds{_b}"] = fx.Array[fx.Float8E4M3FN, bh_lds_size, 16]
    for _b in range_constexpr(NBB):
        _anns[f"BR_lds{_b}"] = fx.Array[fx.Float8E4M3FN, bh_lds_size, 16]
    for _b in range_constexpr(_NSCBUF):
        _anns[f"SC_lds{_b}"] = fx.Array[fx.Int32, _SCBUF, 16]
    SS = fx.struct(type("SSFp4Wgrad", (), {"__annotations__": _anns}))

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kern(
        A: fx.Tensor,  # lhs [OUT_M, M_total/2] fp4 (flat int8)
        B_T: fx.Tensor,  # rhs [OUT_N, M_total/2] fp4 (flat int8)
        C: fx.Tensor,  # [G, OUT_M, OUT_N]
        A_scale: fx.Tensor,  # packed lhs scale (whole-tensor)
        B_scale: fx.Tensor,  # packed rhs scale
        GO: fx.Tensor,  # padded per-group M offs (int32 view int64 [G+1])
    ):
        F8 = fx.Float8E4M3FN.ir_type
        lds = fx.SharedAllocator().allocate(SS).peek()
        A_buf = [getattr(lds, f"A_lds{i}") for i in range_constexpr(NABUF)]
        BL_buf = [getattr(lds, f"BL_lds{i}") for i in range_constexpr(NBB)]
        BR_buf = [getattr(lds, f"BR_lds{i}") for i in range_constexpr(NBB)]
        SC_buf = [getattr(lds, f"SC_lds{b}") for b in range_constexpr(_NSCBUF)]
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2
        I32 = fx.Int32

        # operand SRDs/loaders built per-tile below (rebased); tile-independent parts here.
        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_BH, packed=True, wlv=wlv, elgk=elgk)
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, M_total, N_LDS_STEPS_A, BPR, swizzle=swizzle)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, M_total, N_LDS_STEPS_BH, BPR, swizzle=swizzle)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, LDS_ROW_STRIDE, swizzle=swizzle)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_BH, LDS_ROW_STRIDE, swizzle=swizzle)
        _qm = ((OUT_M + 63) // 64) * 64
        _qn = ((OUT_N + 63) // 64) * 64
        sa_s2r = ScaleS2RPacked(A_scale, _qm, M_total, 4)
        sb_s2r = ScaleS2RPacked(B_scale, _qn, M_total, 4)
        wave_m_off = wave_m * (N_TILES_A * 16)
        wave_n_off = wave_n * (N_TILES_BH * 16)

        a_base6 = [
            [a_s2r.base_addr(A_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NABUF)
        ]
        bl_base6 = [
            [b_s2r.base_addr(BL_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)
        ]
        br_base6 = [
            [b_s2r.base_addr(BR_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)
        ]

        def _gbase(buf):
            v = fx.Int32(fx.ptrtoint(buf.ptr)) + fx.Int32(wave_id) * fx.Int32(1024)
            return rocdl.readfirstlane(T.i32, v)

        abase6 = [_gbase(A_buf[b]) for b in range_constexpr(NABUF)]
        blbase6 = [_gbase(BL_buf[b]) for b in range_constexpr(NBB)]
        brbase6 = [_gbase(BR_buf[b]) for b in range_constexpr(NBB)]
        gl_a6 = [fx.Int32(gl_off_a[st]) for st in range_constexpr(N_LDS_STEPS_A)]
        gl_b6 = [fx.Int32(gl_off_b[st]) for st in range_constexpr(N_LDS_STEPS_BH)]
        scv6 = fx.Int32(0x7F7F7F7F)
        sc_rb6 = [
            fx.ptrtoint(
                fx.add_offset(SC_buf[b].ptr, fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW) + lane_id))
            )
            for b in range_constexpr(_NSCBUF)
        ]
        sc_gb6 = [
            rocdl.readfirstlane(
                T.i32,
                fx.Int32(
                    fx.ptrtoint(
                        fx.add_offset(SC_buf[b].ptr, fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW)))
                    )
                ),
            )
            for b in range_constexpr(_NSCBUF)
        ]
        _scrsa_v = sa_s2r.rsrc
        _scrsb_v = sb_s2r.rsrc
        sc_voff6 = lane_id * fx.Int32(8 * N_SUB)

        def _scsoff(base, extra, ksb):
            grp = (base + fx.Int32(extra)) // fx.Int32(64)
            return rocdl.readfirstlane(
                T.i32, (grp * fx.Int32(K128m) + fx.Int32(_PRELL * N_SUB)) * fx.Int32(256) + ksb
            )

        go_t = rocdl.make_buffer_tensor(GO, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go_t, fx.make_layout(1, 1))
        pid = xcd_remap_pid(fx.block_idx.x, I32(TOTAL), num_xcds)
        group_idx, block_m, block_n = _wgrad_block_mn(
            pid, G, TILES_PER_GROUP, N_BLOCKS_M, N_BLOCKS_N, group_m, group_n, False
        )
        m_start = _load_go(go_div, group_idx)
        m_end = _load_go(go_div, group_idx + 1)
        nval = ((m_end - m_start) // I32(512)) * I32(2)  # even 256-block count

        a_row = block_m * I32(BLOCK_M)
        b_row = block_n * I32(BLOCK_N)
        # Fold the tile's row base + per-group contraction start into the operand SRDs in
        # int64: a_row*M2 (large OUT_M) and the m_start byte offset (large M_total) push the
        # start past 2^31, unreachable by the whole-loop's int32 voffset/soffset.
        _ms2 = arith.index_cast(T.index, m_start >> 1)
        a_base_e = arith.index_cast(T.index, a_row) * arith.index(M2) + _ms2
        b_base_e = arith.index_cast(T.index, b_row) * arith.index(M2) + _ms2
        a_nrec = arith.index(OUT_M) * arith.index(M2) - a_base_e
        b_nrec = arith.index(OUT_N) * arith.index(M2) - b_base_e
        gA, rsrc_a = make_fp8_rebased_tensor_and_srd(A, F8, a_base_e, a_nrec)
        gB, rsrc_b = make_fp8_rebased_tensor_and_srd(B_T, F8, b_base_e, b_nrec)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8, wave_id)
        bl_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8, wave_id)
        br_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8, wave_id)
        a_off = I32(0)  # tile row base + contraction start folded into the SRDs above; only
        bl_off = I32(0)  # br's LDS-half row shift survives as an int32-safe residual.
        br_off = I32(LDS_BN_HALF) * I32(M2)
        sa_b = a_row + I32(wave_m_off)
        sbl_b = b_row + I32(wave_n_off)
        sbr_b = b_row + I32(LDS_BN_HALF) + I32(wave_n_off)
        ksb = (m_start // I32(256)) * I32(_SCVSTEP)  # contraction-start scale byte offset

        for _pp in range_constexpr(0, _PRELL):
            a_g2s.load(A_buf[_pp], a_off + _pp * KSTEP)
        for _pp in range_constexpr(0, _PRELL):
            bl_g2s.load(BL_buf[_pp], bl_off + _pp * KSTEP)
            br_g2s.load(BR_buf[_pp], br_off + _pp * KSTEP)
        _llvm.inline_asm(
            res=None, operands_=[], asm_string="s_waitcnt lgkmcnt(0)", constraints="", has_side_effects=True
        )
        wait_barrier(0)

        accL = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)
        accR = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)
        soff6_a = rocdl.readfirstlane(T.i32, a_off + fx.Int32(_PRELL * KSTEP))
        soff6_bl = rocdl.readfirstlane(T.i32, bl_off + fx.Int32(_PRELL * KSTEP))
        soff6_br = rocdl.readfirstlane(T.i32, br_off + fx.Int32(_PRELL * KSTEP))
        _sc1 = _scsoff(sa_b, 64, ksb)
        _sc3 = _scsoff(sbr_b, 0, ksb)
        _wia = sa_b // I32(128)
        _wib = (sbl_b // I32(256)) * I32(2) + (sbl_b % I32(256)) // I32(64)
        _soa = rocdl.readfirstlane(T.i32, _wia * I32(K128m) * I32(512) + ksb)
        _sob = rocdl.readfirstlane(T.i32, _wib * I32(K128m) * I32(512) + ksb)
        sc_soff06 = [_soa, _sc1, _sob, _sc3]
        _half_n = None
        if const_expr(_HALF_N):
            _half_n = _readfirstlane_i32(arith.select(block_n == I32(N_BLOCKS_N - 1), I32(1), I32(0)))
        accL, accR = mfma.call_mxfp4_wholeloop(
            a_base6,
            bl_base6,
            br_base6,
            a_s2r.tile_stride,
            b_s2r.tile_stride,
            abase6,
            blbase6,
            brbase6,
            gl_a6,
            gl_b6,
            rsrc_a,
            rsrc_b,
            fx.Int32(KSTEP),
            scv6,
            accL,
            accR,
            N_SUB,
            N_LDS_STEPS_A,
            N_LDS_STEPS_BH,
            nval,
            soff6_a,
            soff6_bl,
            soff6_br,
            sc_rb6,
            sc_gb6,
            _scrsa_v,
            _scrsb_v,
            sc_voff6,
            sc_soff06,
            ki=None,
            sc_buf_stride=(_SCBUF * 4),
            half_n=_half_n,
        )
        base_row = group_idx * I32(OUT_M) + a_row + I32(wave_m_off)
        base_col_l = b_row + I32(wave_n_off)
        base_col_r = b_row + I32(LDS_BN_HALF) + I32(wave_n_off)
        store_c = StoreCPlain(
            C, (group_idx + I32(1)) * I32(OUT_M), OUT_N, mfma.idx, N_TILES_A, N_TILES_BH, _out_ty
        )
        store_c.store(accL, base_row, base_col_l, n_valid=_NV)
        store_c.store(accR, base_row, base_col_r, n_valid=_NV)

    _pt = {"passthrough": [["amdgpu-agpr-alloc", "256"]]}
    attrs = {"rocdl.flat_work_group_size": "256,256", "rocdl.waves_per_eu": OCC, **_pt}
    return kern, attrs, TOTAL


_GMXFP4_WGRAD_LAUNCH_CACHE: dict = {}
_GMXFP4_WGRAD_WS_CACHE: dict = {}
_GMXFP4_WGRAD_AT_CACHE: dict = {}  # (OUT_M_p, OUT_N_p, M_alloc, G, out_fp16) -> [raw, compiled]


def _get_grouped_mxfp4_wgrad_ws(OUT_M, OUT_N, K128m, device):
    key = (OUT_M, OUT_N, K128m, device)
    e = _GMXFP4_WGRAD_WS_CACHE.get(key)
    if e is None:
        qm = ((OUT_M + 63) // 64) * 64
        qn = ((OUT_N + 63) // 64) * 64
        a_sp = torch.empty(qm * K128m, dtype=torch.int32, device=device)
        b_sp = torch.empty(qn * K128m, dtype=torch.int32, device=device)
        e = (a_sp, b_sp)
        _GMXFP4_WGRAD_WS_CACHE[key] = e
    return e


def _compile_grouped_mxfp4_wgrad_fused(OUT_M, OUT_N, G, M_total, gm, xcd, gn, wlv, elgk, out_fp16):
    K128m = M_total // 128
    pre_ab = _build_mxfp4_preshuffle_kernel_ab()
    gemm_k, attrs, TOTAL = _build_grouped_mxfp4_wgrad_kernel(
        OUT_M, OUT_N, G, M_total, group_m=gm, num_xcds=xcd, group_n=gn, wlv=wlv, elgk=elgk, out_fp16=out_fp16
    )
    _PGRID = _MXFP4_PRESHUF_FO * _MXFP4_PRESHUF_BLK

    @flyc.jit
    def launch(
        a8: fx.Tensor,
        b8: fx.Tensor,
        C: fx.Tensor,
        a_raw: fx.Tensor,
        b_raw: fx.Tensor,
        a_sp: fx.Tensor,
        b_sp: fx.Tensor,
        GO: fx.Tensor,
        stream: fx.Stream,
    ):
        grid_a = ceildiv(fx.Int32(OUT_M) * fx.Int32(K128m), _PGRID)
        grid_b = ceildiv(fx.Int32(OUT_N) * fx.Int32(K128m), _PGRID)
        pre_ab(a_raw, a_sp, b_raw, b_sp, fx.Int32(OUT_M), fx.Int32(OUT_N), fx.Int32(K128m), grid_a).launch(
            grid=(grid_a + grid_b, 1, 1), block=(_MXFP4_PRESHUF_BLK, 1, 1), stream=stream
        )
        gemm_k(a8, b8, C, a_sp, b_sp, GO, value_attrs=attrs).launch(
            grid=(TOTAL, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch, TOTAL


def grouped_gemm_mxfp4_variable_k_flydsl_kernel(
    lhs, lhs_scale, rhs, rhs_scale, group_offs, OUT_M, OUT_N, G, out_dtype=torch.bfloat16, num_cu=-1
):
    """FlyDSL MXFP4 grouped variable-K wgrad (bare-asm whole-loop). lhs [OUT_M, M/2] /
    rhs [OUT_N, M/2] fp4 in the FlyDSL-quant colwise layout (each group's M already
    512-aligned), group_offs [G+1] the matching 512-padded per-group M offsets. Runs the
    NT whole-loop with a runtime nval directly -- no on-GPU repack, no free-dim pad (the
    non-256 OUT_M rows are SRD-dropped, OUT_N cols masked in the store). Returns
    C [G, OUT_M, OUT_N]."""
    assert lhs.ndim == 2 and rhs.ndim == 2
    assert lhs.shape[0] == OUT_M and rhs.shape[0] == OUT_N
    M_total = lhs.shape[1] * 2  # colwise contraction width (512-padded per group by the quant)
    assert rhs.shape[1] * 2 == M_total
    dev = lhs.device
    out_fp16 = out_dtype == torch.float16

    # Keep the fp4 operands 2D (do NOT flatten): a large total_M makes OUT_*/2 * M/2
    # exceed 2^31 int8s, which flydsl packs as an int32 dim (host CABI overflow). The
    # kernel addresses via the rebased base, independent of the operand shape.
    a8 = lhs.contiguous().view(torch.int8)
    b8 = rhs.contiguous().view(torch.int8)
    a_raw = lhs_scale.contiguous().view(torch.int32).reshape(-1)
    b_raw = rhs_scale.contiguous().view(torch.int32).reshape(-1)
    go_pad = (group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)).view(torch.int32)

    K128m = M_total // 128
    a_sp, b_sp = _get_grouped_mxfp4_wgrad_ws(OUT_M, OUT_N, K128m, dev)
    # 3D C (NOT flattened): StoreCPlain re-bases per row band from C's base + OUT_N; a 1D
    # G*OUT_M*OUT_N view overflows the CABI for large-G MoE grad_b (> 2^31 elems).
    out = torch.empty((G, OUT_M, OUT_N), dtype=out_dtype, device=dev)

    stream = torch.cuda.current_stream()
    wlv, elgk = 10, 9
    args = (a8, b8, out, a_raw, b_raw, a_sp, b_sp, go_pad, stream)

    def _entry(cfg):
        gm, xcd, gn = cfg
        lk = (OUT_M, OUT_N, G, M_total, gm, xcd, gn, wlv, elgk, out_fp16)
        ent = _GMXFP4_WGRAD_LAUNCH_CACHE.get(lk)
        if ent is None:
            ent = _compile_grouped_mxfp4_wgrad_fused(
                OUT_M, OUT_N, G, M_total, gm, xcd, gn, wlv, elgk, out_fp16
            )
            _GMXFP4_WGRAD_LAUNCH_CACHE[lk] = ent
        atk = (OUT_M, OUT_N, M_total, G, gm, xcd, gn, out_fp16)
        e2 = _GMXFP4_WGRAD_AT_CACHE.get(atk)
        if e2 is None:
            e2 = [ent[0], None]
            _GMXFP4_WGRAD_AT_CACHE[atk] = e2
        return e2

    run_eager_or_capture(_entry(_GMXFP4_WGRAD_CFG), args, 1)
    _bound_caches(_GMXFP4_WGRAD_LAUNCH_CACHE, _GMXFP4_WGRAD_AT_CACHE, _GMXFP4_WGRAD_WS_CACHE)
    return out
