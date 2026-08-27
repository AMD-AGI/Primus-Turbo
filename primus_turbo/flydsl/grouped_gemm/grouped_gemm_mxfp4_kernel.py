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

"""FlyDSL MXFP4 (per-32-K E8M0 block-scaled) grouped GEMM for gfx950 (NT fwd/dgrad):
reuses the dense mxfp4 whole-loop compute with fp8-grouped addressing and lane-packed scales."""

import gc

import torch

# isort: off
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    _lane_tbl_count_le,
    _lane_tbl_get,
    _lane_tbl_load,
    _lane_tbl_scan,
    make_fp8_rebased_tensor_and_srd,
    resolve_accum_out,
    xcd_band_remap_pid,
    xcd_remap_pid,
)
from primus_turbo.flydsl.utils.prims import (
    ceildiv,
    ceildiv_pow2,
    _lds_barrier,
    _readfirstlane_i32,
    _readlane_i32,
)
from primus_turbo.flydsl.utils.gemm_epilogue_helper import (
    DGLU_BAND_ROWS,
    LDS_WORDS_PER_WAVE,
    MXFP4DualQuantStore,
    MXFP4DualQuantStoreDglu,
    StoreCdSwiGLUQuadCShuffle,
    StoreCdSwiGLUQuadQuant,
    StoreCSwiGLU,
    StoreCSwiGLUQuant,
)
from primus_turbo.flydsl.gemm.gemm_mxfp4_kernel import (
    _MXFP4_PRESHUF_BLK,
    _MXFP4_PRESHUF_FO,
    MfmaScaleFp4,
    S2RLoaderFp4,
    S2RLoaderFp4Split,
    ScaleS2RPacked,
    StoreCPlain,
    _build_mxfp4_preshuffle_kernel_ab,
    _mxfp4_grp_from,
    _mxfp4_pack_cell,
    fp4_g2s_offsets,
    fp4_g2s_offsets_split,
)
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_kernel import (
    _grouped_block_mn,
    _wgrad_block_mn,
)
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp8_kernel import run_eager_or_capture

# isort: on

_BLOCK = 256  # BLOCK_M = BLOCK_N = BLOCK_K
_PRESHUF_BLK = 256
_PRESHUF_NG = 4  # g bytes packed by one preshuffle thread
_PRESHUF_ND = 4  # (r_region, K sub-block) cells packed by one preshuffle thread
_PRESHUF_FO = _PRESHUF_NG * _PRESHUF_ND  # output dwords per thread
_GMXFP4_XCD_BAND_STEP = 2  # M-block granularity a per-group tile count is a multiple of


_GMXFP4_SCHED_HINTS = {
    "llvm_options": {
        "amdgpu-sched-strategy": "iterative-ilp",
        "enable-post-misched": True,
        "lsr-drop-solution": True,
    }
}


_GMXFP4_SKEW_CUS = 256  # one skew rank per CU
_GMXFP4_SKEW_STEP = 2  # s_sleep units (~64 clocks) per skew rank


def _emit_launch_skew(bid):
    step = _GMXFP4_SKEW_STEP
    _llvm.inline_asm(
        T.i32,
        [bid.ir_value()],
        f"s_cmp_lt_u32 $1, {_GMXFP4_SKEW_CUS}\n\ts_cselect_b32 $0, $1, 0\n"
        f"1:\n\ts_cmp_eq_u32 $0, 0\n\ts_cbranch_scc1 2f\n\ts_sleep {step}\n"
        "\ts_sub_u32 $0, $0, 1\n\ts_branch 1b\n2:",
        "=&s,s,~{scc},~{memory}",
        has_side_effects=True,
    )


def _run_mxfp4_sched(entry, args, compiled_idx):
    """run_eager_or_capture with the mxfp4 grouped NT schedule hints applied."""
    if torch.cuda.is_current_stream_capturing():
        entry[0](*args)
        return
    if entry[compiled_idx] is None:
        with CompilationContext.compile_hints(_GMXFP4_SCHED_HINTS):
            entry[compiled_idx] = flyc.compile(entry[0], *args)
    entry[compiled_idx](*args)


def _build_grouped_mxfp4_ab_preshuffle(
    K128: int, G: int, N: int, k128_rd: int = None, b_ilv: int = 0, glu_i: int = 0
):
    """Merged A-slab + B-per-expert scale preshuffle in ONE launch (one fewer in-stream launch
    per grouped GEMM). Blocks [0, a_grid) do the A slab (mode 0), the rest the B per-expert;
    the two paths are segment-selected with no per-thread divergence. Read is real-K masked.

    ``glu_i`` (the gate width, with ``N == 2 * glu_i``) makes the B side emit the fused
    forward's row order instead of the plain one -- see the permutation at the read."""
    _KRD = K128 if k128_rd is None else k128_rd
    N_SCALE = ceildiv(N, 256) * 256  # 256-multiple: ScaleS2RPacked packs four 64-row groups
    n_sub, nd, KK = 2, _PRESHUF_ND, K128 // 2
    assert not b_ilv or b_ilv == nd
    n_rr = nd // n_sub
    b_dwords_pe = N_SCALE * K128 // _PRESHUF_FO
    _NWI = 1 + ceildiv(64 // 16 - 1, KK)  # wi values a wave spans (one (wi,kk,r) cell/thread)

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
        _bq = wi // I32(2)  # band pair a packed slot sits in (glu row remap below)
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
            okc = arith.select(is_b, in_grid, in_grid & (grp_a < _ngrp))  # skip slab-pad groups
            for t in range_constexpr(nd):
                b_row = grp_b * I32(64) + (r * I32(b_ilv) + I32(t) if b_ilv else I32(t * 16) + r)
                b_ok = b_row < I32(N)
                if const_expr(bool(glu_i)):
                    # The packed row space alternates 128-row gate/up bands, so the R
                    # pool -- a fixed 128 rows past L -- lands on the up band for any I.
                    # r_region is that band's parity.
                    b_row = b_row - (_bq + I32(r_region)) * I32(128)
                    b_ok = b_row < I32(glu_i)
                    b_row = b_row + I32(r_region * glu_i)
                row = arith.select(is_b, rd_base + b_row, rd0 + I32(t * 16) + r)
                valid = okc & arith.select(is_b, b_ok, row < rd_end)
                v = Vec(
                    buffer_ops.buffer_load(
                        rin, row * I32(_KRD) + k128, vec_width=n_sub, dtype=T.i32, mask=valid
                    )
                )
                if const_expr(_KRD % n_sub != 0):  # odd real K128: zero the past-K tail sub-block
                    v = Vec.from_elements(
                        [v[0]]
                        + [
                            arith.select(k128 + I32(j) < I32(_KRD), v[j], I32(0))
                            for j in range_constexpr(1, n_sub)
                        ]
                    )
                dws.append(v)
        words = _mxfp4_pack_cell(dws, n_sub, nd, _PRESHUF_NG)
        for g in range_constexpr(_PRESHUF_NG):  # pad regions store 0 (masked reads gave words=0)
            buffer_ops.buffer_store(Vec.from_elements(words[g]), rout, base + I32(g * 64), mask=gid < total)

    return kern


def _build_grouped_mxfp4_nt_kernel(
    K,
    G,
    N,
    group_m=4,
    num_xcds=8,
    group_n=0,
    wlv=10,
    elgk=9,
    out_fp16=False,
    k_real=None,
    xcd_span=16,
    cst_nt=False,
    glu=False,
    dglu=False,
    glu_i=0,
    glu_skip_act=False,
    glu_quant_row=False,
    glu_act_quant=False,
    dglu_act_quant=False,
    epi_row_sr=False,
    epi_col_sr=False,
):
    """Grouped MXFP4 NT (out = a @ b^T), per-group A rows + per-expert B, whole-loop compute.
    K is the 256-rounded scale extent; ``k_real`` (<=K, 128-multiple) is the operands' true
    contraction, its %256==128 tail run as a trailing block with zero-pad scale (no operand copy).

    ``glu``/``dglu`` fuse the SwiGLU forward / gradient into the epilogue. ``glu`` re-points
    the second B LDS pool at the weight's ``up`` band (row offset ``I``) so gate and up for
    one output column land in the same lane; ``dglu`` keeps the plain tiling (its N axis is
    already ``I``) and stages the accumulator through LDS so the ``l1``/``dl1`` traffic
    vectorises. Both take ``N == glu_i``."""
    BLOCK_M = BLOCK_N = BLOCK_K = _BLOCK
    _KR = K if k_real is None else k_real  # operand true contraction (128-multiple)
    assert K % 256 == 0 and _KR % 128 == 0
    KI = _KR // BLOCK_K  # FULL 256-blocks over the REAL K
    _K128 = (_KR // 128) % 2  # 1 => trailing 128-K block, handled by scale-pad-zero below
    KI_LOOP = KI + 1 if _K128 else KI  # trailing 128-K: last block's past-K s=1 sub-step drops
    NABUF, NBB, OCC = 2, 2, 2  # fwd waves_per_eu=2: hide the latency-bound short-K/small-tile GEMM
    N_SUB = BLOCK_K // 128
    BPR = BLOCK_K // 2
    KSTEP = BPR
    K2 = _KR // 2  # operand row stride (bytes) = real K (no operand K-pad)
    N_TILES_A = BLOCK_M // 32
    LDS_BN_HALF = BLOCK_N // 2
    N_TILES_BH = LDS_BN_HALF // 32
    LDS_ROW_STRIDE = BPR
    _ROWS_PER_STEP = 64 // (BPR // 16) * (256 // 64)
    N_LDS_STEPS_A = BLOCK_M // _ROWS_PER_STEP
    N_LDS_STEPS_BH = LDS_BN_HALF // _ROWS_PER_STEP
    NSA_H = N_LDS_STEPS_A // 2  # g2s steps per parity region
    NSB_H = N_LDS_STEPS_BH // 2
    _PRELL, _NSCBUF = 2, 2
    K128 = K // 128
    assert not (glu and dglu)
    assert not (glu or dglu) or N == glu_i, "fused GLU needs N == glu_i (the gate width)"
    # Under glu the second B pool holds the ``up`` band rather than the tile's next
    # 128 columns, so a tile block is 128 gate columns wide, not 256.
    _NCB = LDS_BN_HALF if glu else BLOCK_N  # output columns one tile block advances by
    _BROWS = 2 * glu_i if glu else N  # B rows per expert (gate||up under glu)
    _RSHIFT = glu_i if glu else LDS_BN_HALF  # column distance from the L pool to the R pool
    _SNCB, _SRSHIFT = BLOCK_N, LDS_BN_HALF  # the same two, in the packed-scale row space
    N_SCALE = ceildiv(_BROWS, 256) * 256
    NBK = ceildiv(N, _NCB)  # n_blocks
    # Narrowest XCD band (M-blocks) a per-group tile count still divides: the ragged fallback.
    _SPAN_NARROW = min(xcd_span, _GMXFP4_XCD_BAND_STEP)
    _WIDE_MB = num_xcds * xcd_span  # M-blocks a group needs to reach every XCD by itself
    _NV = N if (N % _NCB != 0) else None  # non-256 N: mask store cols >= N (no host N-pad)
    # The R pool is the up band under glu, so a ragged last block cannot drop it; the fused
    # epilogue masks the past-I columns of both pools instead.
    _HALF_N = (not glu) and (N % BLOCK_N != 0) and (N % BLOCK_N <= LDS_BN_HALF)
    # A gate width in whole 64-column bands turns the ragged edge into an empty band
    # SRD, which is what lets l1 go through the mainloop's store slot -- that store
    # has no mask to give, so short of this the edge is dropped via the address.
    _GLU_BAND = glu and (glu_i % (16 * N_TILES_BH) == 0)
    _BILV_OK = LDS_ROW_STRIDE == 128 and N_TILES_BH == 4
    _CSTORE = (
        (not out_fp16)
        and (not dglu)
        and ((not glu) or _GLU_BAND or (_BILV_OK and glu_i % N_TILES_BH == 0))
        and bool(_K128)
        and (KI_LOOP % 2 == 1 or KI_LOOP >= 4)
    )
    _BILV = N_TILES_BH if (_CSTORE and _BILV_OK) else 0
    _COL_SAFE = (N % _NCB == 0) if glu else (N % BLOCK_N == 0)
    # dglu stages one 16-row quadrant band as f32 so the l1 read and dl1 write become
    # 128-bit; the two waves of a wave_m group share it. The padding puts the four row
    # groups a staging write spans on disjoint banks.
    _DGLU_BAND_PAD = 4
    # l1 and dl1 are streamed once and dwarf L2, so non-temporal leaves that room to
    # the operands the tiles do re-read.
    _DGLU_AUX = 2
    _GLU_ACT_AUX = 2  # act is streamed once too
    # Quantising act in the epilogue needs the l1 store out of the way (in-mainloop)
    # and the wave tile to be the 128x64 the dual-quant geometry is built on.
    assert not glu_act_quant or (glu and _CSTORE and N_TILES_A % 2 == 0 and N_TILES_BH == 4), (
        "fused act quant needs glu with the in-mainloop l1 store and a 128x64 wave tile"
    )
    assert not (glu_act_quant and glu_skip_act), "fused act quant is what replaces the act store"
    # Quantising grad_l1 in the dGLU epilogue needs the band to be two sub-tiles tall
    # (the col-wise micro-block) and a wave_m group to span the 128 columns the
    # col-wise staging is laid out for. A 32-multiple I keeps a row-wise micro-block
    # from straddling the dg/du halves of grad_l1.
    assert not dglu_act_quant or (dglu and N_TILES_A % 2 == 0 and N_TILES_BH == 4 and glu_i % 32 == 0), (
        "fused grad_l1 quant needs dglu with an even sub-tile count and a 128x64 wave tile"
    )
    # Stochastic rounding is per-operand and belongs to whichever quant epilogue is on;
    # the two are mutually exclusive, and neither non-quant path has anything to round.
    assert not (epi_row_sr or epi_col_sr) or glu_act_quant or dglu_act_quant, (
        "stochastic rounding needs one of the quantising epilogues"
    )

    # parity-split LDS ring: skewed rows (odd 64-multiple stride) straddle two 128B lines
    _A_SLOT = (BLOCK_M // 2) * LDS_ROW_STRIDE  # skewed rows need 3 slots, aligned rows 2
    _B_SLOT = (LDS_BN_HALF // 2) * LDS_ROW_STRIDE
    _SK = 64 * _K128  # skewed region's byte offset from its 128B-aligned line
    _NOBUF = 3 if _K128 else 2  # a skewed in-place refill needs 2 live + 1 landing slot
    _anns = {"A_e": fx.Array[fx.Float8E4M3FN, NABUF * _A_SLOT, 16]}
    _anns["A_o"] = fx.Array[fx.Float8E4M3FN, _NOBUF * _A_SLOT, 16]
    for _h in ("BL", "BR"):
        _anns[f"{_h}_e"] = fx.Array[fx.Float8E4M3FN, NBB * _B_SLOT, 16]
        _anns[f"{_h}_o"] = fx.Array[fx.Float8E4M3FN, _NOBUF * _B_SLOT, 16]
    SS = fx.struct(type("SSFp4Grp", (), {"__annotations__": _anns}))

    def _body(
        A,
        B_T,
        C,
        ACT,
        PROBS,
        GRAD_PROBS,
        A_scale,
        B_scale,
        GO,
        c_m,
        c_n,
        slab_rows,
        gp_stride,
        AQ_OUT=None,
        AQ_SC=None,
        AQ_TOUT=None,
        AQ_TSC=None,
        aq_col_rows=None,
        SR_SEED=None,
    ):
        """A: [total_M, K/2] fp4 (flat int8); B_T: [G, N, K/2]; C: [total_M, N].
        A_scale/B_scale are the packed slabs, GO the tight offs (int32 view of int64 [G+1]).
        ACT/PROBS/GRAD_PROBS carry the fused GLU streams and are None on the plain path."""
        F8 = fx.Float8E4M3FN.ir_type
        lds = fx.SharedAllocator().allocate(SS).peek()
        A_lds = [lds.A_e, lds.A_o]
        BL_lds = [lds.BL_e, lds.BL_o]
        BR_lds = [lds.BR_e, lds.BR_o]
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2
        I32 = fx.Int32

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_BH, packed=True, wlv=wlv, elgk=elgk)
        gl_b_e = fp4_g2s_offsets_split(lane_id, wave_id, _KR, NSB_H, 0, 0, ilv=_BILV)
        gl_b_o = fp4_g2s_offsets_split(lane_id, wave_id, _KR, NSB_H, 1, _SK, ilv=_BILV)
        gl_b_o0 = fp4_g2s_offsets_split(lane_id, wave_id, _KR, NSB_H, 1, -_SK, ilv=_BILV)
        b_s2r = S2RLoaderFp4Split(wave_n, N_TILES_BH, LDS_BN_HALF, 0, _K128, ilv=_BILV)
        sa_s2r = ScaleS2RPacked(A_scale, slab_rows, K, 4)
        sb_s2r = ScaleS2RPacked(B_scale, I32(N_SCALE * G), K, 4)
        wave_m_off = wave_m * (N_TILES_A * 16)
        wave_n_off = wave_n * (N_TILES_BH * 16)

        def _b_bases(lds, b):
            if const_expr(bool(_BILV)):
                p = [b_s2r.f_base_ilv(lds[0], lds[1], b, s) for s in range_constexpr(N_SUB)]
                return [x[1] for x in p], [x[0] for x in p]
            return [b_s2r.f_base(lds[0], lds[1], b, s) for s in range_constexpr(N_SUB)], None

        _bl = [_b_bases(BL_lds, b) for b in range_constexpr(NBB)]
        _br = [_b_bases(BR_lds, b) for b in range_constexpr(NBB)]
        bl_base6 = [x[0] for x in _bl]
        br_base6 = [x[0] for x in _br]
        b_even6 = ([x[1] for x in _bl], [x[1] for x in _br]) if const_expr(bool(_BILV)) else None
        qu_b6 = b_s2r.q_unit() if const_expr(_K128) else None

        def _gbase(buf, slot=0):
            v = fx.Int32(fx.ptrtoint(buf.ptr)) + fx.Int32(wave_id) * fx.Int32(1024) + fx.Int32(slot)
            return rocdl.readfirstlane(T.i32, v)

        blbase6 = [_gbase(BL_lds[0], b * _B_SLOT) for b in range_constexpr(NBB)]
        brbase6 = [_gbase(BR_lds[0], b * _B_SLOT) for b in range_constexpr(NBB)]
        bl_od6 = [_gbase(BL_lds[1], j * _B_SLOT) for j in range_constexpr(_NOBUF)]
        br_od6 = [_gbase(BR_lds[1], j * _B_SLOT) for j in range_constexpr(_NOBUF)]
        gl_b6 = [fx.Int32(o) for o in gl_b_e] + [fx.Int32(o) for o in gl_b_o]
        scv6 = fx.Int32(0x7F7F7F7F)
        sc_rb6 = [fx.Int32(0) for _b in range_constexpr(_NSCBUF)]  # reserved (VGPR-direct scales)
        sc_gb6 = [fx.Int32(0) for _b in range_constexpr(_NSCBUF)]
        _scrsa_v = sa_s2r.rsrc
        _scrsb_v = sb_s2r.rsrc
        sc_voff6 = lane_id * fx.Int32(8 * N_SUB)

        def _scsoff(base, extra):
            grp = (base + fx.Int32(extra)) // fx.Int32(64)
            return rocdl.readfirstlane(
                T.i32, (grp * fx.Int32(K128) + fx.Int32(_PRELL * N_SUB)) * fx.Int32(256)
            )

        # lane-resident group scan (lane g owns group g) replaces the serial G-wide compare tree
        go_rs = buffer_ops.create_buffer_resource(GO, max_size=False, num_records_bytes=(G + 1) * 8)
        _go0 = _lane_tbl_load(go_rs, lane_id, G + 1, stride=2)
        _go1 = _lane_tbl_load(go_rs, lane_id, G + 1, stride=2, first=1)
        _own = [lane_id + I32(64 * c) < I32(G) for c in range_constexpr(len(_go0))]
        _nb = [
            arith.select(_own[c], ceildiv_pow2(_go1[c] - _go0[c], BLOCK_M), I32(0))
            for c in range_constexpr(len(_go0))
        ]
        # A band dividing every group's M-block count keeps an XCD's reads in one expert slab.
        if const_expr(_SPAN_NARROW < xcd_span):
            _span_res = [
                arith.select(_own[c], _nb[c] % I32(xcd_span), I32(0)) for c in range_constexpr(len(_nb))
            ]
            _span_div = _readlane_i32(_lane_tbl_scan(_span_res)[-1], 63) == I32(0)
            _nb_wide = I32(64 * len(_nb)) - _lane_tbl_count_le(_nb, I32(_WIDE_MB - 1))
            _span_ok = _span_div | (_nb_wide == I32(0))
        _nbs_end = _lane_tbl_scan(_nb)
        _tcs_end = [v * I32(NBK) for v in _nbs_end]  # entry g = tiles owned by groups <= g
        _tcs = [_tcs_end[c] - _nb[c] * I32(NBK) for c in range_constexpr(len(_nb))]
        _sas = [(_nbs_end[c] - _nb[c]) * I32(4) for c in range_constexpr(len(_nb))]
        # The col-wise operand rounds each group's rows up to BLOCK_M, and _nb is
        # already that block count, so the exclusive scan is the padded row base --
        # no host-side offset table needed.
        if const_expr(glu_act_quant or dglu_act_quant):
            _pad0 = [(_nbs_end[c] - _nb[c]) * I32(BLOCK_M) for c in range_constexpr(len(_nb))]
        total_tiles = _readlane_i32(_tcs_end[-1], 63)
        bid = fx.block_idx.x
        _llvm.inline_asm(
            None,
            [bid.ir_value(), arith._to_raw(total_tiles)],
            "s_cmp_lt_u32 $0, $1\n\ts_cbranch_scc1 1f\n\ts_endpgm\n\t1:",
            "s,s,~{scc},~{memory}",
            has_side_effects=True,
        )
        _emit_launch_skew(bid)
        if const_expr(_SPAN_NARROW < xcd_span):
            pid = arith.select(  # skew-robust band, group-aligned
                _span_ok,
                xcd_band_remap_pid(bid, total_tiles, num_xcds, xcd_span * NBK),
                xcd_band_remap_pid(bid, total_tiles, num_xcds, _SPAN_NARROW * NBK),
            )
        else:
            pid = xcd_band_remap_pid(bid, total_tiles, num_xcds, xcd_span * NBK)
        group_idx = _lane_tbl_count_le(_tcs_end, pid)
        tile_start = _lane_tbl_get(_tcs, group_idx)
        a_pre_g = _lane_tbl_get(_sas, group_idx)
        m_start = _lane_tbl_get(_go0, group_idx)
        m_end = _lane_tbl_get(_go1, group_idx)
        local = pid - tile_start
        bm, bn = _grouped_block_mn(local, m_start, m_end, NBK, BLOCK_M, group_m, group_n)

        m_row = m_start + bm * I32(BLOCK_M)  # tight A/C row base
        a_par = m_row % I32(2)
        a_sh = a_par * I32(64)
        gl_a_e = fp4_g2s_offsets_split(lane_id, wave_id, _KR, NSA_H, a_par, a_sh)
        gl_a_o = fp4_g2s_offsets_split(lane_id, wave_id, _KR, NSA_H, I32(1) - a_par, a_sh + I32(_SK))
        gl_a_o0 = fp4_g2s_offsets_split(lane_id, wave_id, _KR, NSA_H, I32(1) - a_par, a_sh - I32(_SK))
        a_s2r = S2RLoaderFp4Split(wave_m, N_TILES_A, BLOCK_M, a_par, _K128)
        a_base6 = [
            [a_s2r.f_base(A_lds[0], A_lds[1], b, s) for s in range_constexpr(N_SUB)]
            for b in range_constexpr(NABUF)
        ]
        qu_a6 = a_s2r.q_unit() if const_expr(_K128) else None
        abase6 = [_gbase(A_lds[0], b * _A_SLOT) for b in range_constexpr(NABUF)]
        a_od6 = [_gbase(A_lds[1], j * _A_SLOT) for j in range_constexpr(_NOBUF)]
        gl_a6 = [fx.Int32(o) for o in gl_a_e] + [fx.Int32(o) for o in gl_a_o]
        # Fold A/B bases into int64 SRDs: large-G/large-M exceeds the int32 voffset.
        a_base_e = arith.index_cast(T.index, m_row) * arith.index(K2) - arith.index_cast(T.index, a_sh)
        # Under glu the weight is gate||up, so its row pitch is 2I while c_n stays the gate width.
        _b_rows = arith.index(_BROWS) if glu else arith.index_cast(T.index, c_n)
        b_base_e = (
            arith.index_cast(T.index, group_idx) * _b_rows + arith.index_cast(T.index, bn) * arith.index(_NCB)
        ) * arith.index(K2)
        a_nrec = (arith.index_cast(T.index, c_m) - arith.index_cast(T.index, m_row)) * arith.index(
            K2
        ) + arith.index_cast(T.index, a_sh)
        b_nrec = arith.index(G) * _b_rows * arith.index(K2) - b_base_e
        gA, rsrc_a = make_fp8_rebased_tensor_and_srd(A, F8, a_base_e, a_nrec)
        gB, rsrc_b = make_fp8_rebased_tensor_and_srd(B_T, F8, b_base_e, b_nrec)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        a_g2s = [G2SLoader(a_div, g, NSA_H, F8, wave_id) for g in (gl_a_e, gl_a_o0)]
        bl_g2s = [G2SLoader(b_div, g, NSB_H, F8, wave_id) for g in (gl_b_e, gl_b_o0)]
        br_g2s = [G2SLoader(b_div, g, NSB_H, F8, wave_id) for g in (gl_b_e, gl_b_o0)]
        a_off = I32(0)  # A/B tile+expert bases folded into the SRDs above; only the LDS-half
        bl_off = I32(0)  # column shift (br) survives as an int32-safe intra-tile residual.
        br_off = I32(_RSHIFT) * K2
        sa_b = a_pre_g * I32(64) + bm * I32(BLOCK_M) + I32(wave_m_off)  # 256-aligned slab row base
        # Scale coordinates stay in the packed row space: it binds a 64-row group to the
        # one 128 rows below, so the up band's real offset (I) is inexpressible there.
        # The preshuffle lays the up rows where the plain tiling expects the R pool.
        sbl_b = bn * I32(_SNCB) + I32(wave_n_off)
        sbr_b = bn * I32(_SNCB) + I32(_SRSHIFT) + I32(wave_n_off)
        b_exp_bytes = group_idx * I32(N_SCALE * K128 * 4)  # padded per-expert B-scale base (bytes)

        for _pp in range_constexpr(0, _PRELL - 1):
            if const_expr(KI_LOOP > _pp):
                a_g2s[0].load(A_lds[0], a_off + _pp * KSTEP, base_off=I32(_pp * _A_SLOT))
        for _pp in range_constexpr(0, _NOBUF - 1):
            if const_expr(KI_LOOP + 1 > _pp):
                a_g2s[1].load(A_lds[1], a_off + _pp * KSTEP, base_off=I32(_pp * _A_SLOT))
        for _pp in range_constexpr(0, _PRELL - 1):
            if const_expr(KI_LOOP > _pp):
                bl_g2s[0].load(BL_lds[0], bl_off + _pp * KSTEP, base_off=I32(_pp * _B_SLOT))
                br_g2s[0].load(BR_lds[0], br_off + _pp * KSTEP, base_off=I32(_pp * _B_SLOT))
        for _pp in range_constexpr(0, _NOBUF - 1):
            if const_expr(KI_LOOP + 1 > _pp):
                bl_g2s[1].load(BL_lds[1], bl_off + _pp * KSTEP, base_off=I32(_pp * _B_SLOT))
                br_g2s[1].load(BR_lds[1], br_off + _pp * KSTEP, base_off=I32(_pp * _B_SLOT))

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
        base_row = m_row + I32(wave_m_off)
        base_col_l = bn * I32(_NCB) + I32(wave_n_off)
        base_col_r = base_col_l + I32(_RSHIFT)
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        if glu:
            _glu_kw = dict(
                col_safe=_COL_SAFE,
                ilv=_BILV,
                band_drop=(not _COL_SAFE) and _GLU_BAND,
                cst=_CSTORE,
                act_aux=_GLU_ACT_AUX,
            )
            _glu_args = (
                None,
                None,
                C,
                # Under glu_act_quant nothing is written through the act stream, but
                # the base still gets extracted; alias C rather than pass a null.
                C if const_expr(glu_act_quant) else ACT,
                PROBS,
                m_end,
                glu_i,
                mfma.idx,
                N_TILES_A,
                N_TILES_BH,
                _out_ty,
            )
            if const_expr(glu_act_quant):
                # The A/B pools are dead by the epilogue; the staging borrows BL like
                # dglu's does. 4 waves x 32 rows x 64 cols x 2B = 16 KB.
                _q = MXFP4DualQuantStore(
                    AQ_OUT,
                    AQ_SC,
                    AQ_TOUT,
                    AQ_TSC,
                    c_m,
                    glu_i,
                    ceildiv(glu_i, 128) * 128,  # the quantiser's row-wise pad
                    aq_col_rows,
                    # The pool is typed fp8 for the mainloop; the staging addresses it
                    # as the i32-packed bf16 pairs the quantiser's LDS helpers expect.
                    fx.recast_iter(fx.Int32, lds.BL_e.ptr),
                    wave_id,
                    lane_id,
                    row_sr=epi_row_sr,
                    col_sr=epi_col_sr,
                    sr_seed=SR_SEED,
                )
                assert 4 * LDS_WORDS_PER_WAVE * 4 <= (NBB + _NOBUF) * _B_SLOT
                store_c = StoreCSwiGLUQuant(*_glu_args, quant_store=_q, **_glu_kw)
                pad_row_base = _lane_tbl_get(_pad0, group_idx) + bm * I32(BLOCK_M) + I32(wave_m_off)
            else:
                store_c = StoreCSwiGLU(*_glu_args, skip_act=glu_skip_act, **_glu_kw)
        elif dglu:
            _dglu_args = (
                # Under dglu_act_quant nothing goes through the dl1 stream -- that is
                # what the quant replaces -- but the base still gets extracted, so
                # alias l1 rather than pass a null.
                ACT if const_expr(dglu_act_quant) else C,
                ACT,
                PROBS,
                GRAD_PROBS,
                bn,
                gp_stride,
                m_end,
                glu_i,
                mfma.idx,
                N_TILES_A,
                N_TILES_BH,
                _out_ty,
                lds.BL_e,  # dead once the mainloop drains; see the class docstring
                wave_id,
            )
            # The quant band is a whole 32-row micro-block, which only fits the two
            # dead BL pools unpadded; the pad is worth its bank-disjoint staging
            # writes only where there is room for it (see the assert below).
            _dglu_pad = 0 if dglu_act_quant else _DGLU_BAND_PAD
            _dglu_kw = dict(
                row_pad=_dglu_pad,
                col_safe=_COL_SAFE,
                store_aux=_DGLU_AUX,
            )
            if const_expr(dglu_act_quant):
                _row_stride = 2 * N_TILES_BH * 16 + _dglu_pad
                _q = MXFP4DualQuantStoreDglu(
                    AQ_OUT,
                    AQ_SC,
                    AQ_TOUT,
                    AQ_TSC,
                    c_m,
                    glu_i,
                    ceildiv(2 * glu_i, 128) * 128,  # the quantiser's row-wise pad
                    aq_col_rows,
                    I32(fx.ptrtoint(lds.BL_e.ptr)),
                    I32(wave_m) * I32(DGLU_BAND_ROWS * _row_stride),
                    _row_stride,
                    lane_id,
                    wave_n,
                    row_sr=epi_row_sr,
                    col_sr=epi_col_sr,
                    sr_seed=SR_SEED,
                )
                store_c = StoreCdSwiGLUQuadQuant(*_dglu_args, quant_store=_q, **_dglu_kw)
                pad_row_base = _lane_tbl_get(_pad0, group_idx) + bm * I32(BLOCK_M) + I32(wave_m_off)
            else:
                store_c = StoreCdSwiGLUQuadCShuffle(*_dglu_args, **_dglu_kw)
            # The band is wider than BL_e alone, so it runs on into BL_o; both are
            # dead by then and the struct lays them out adjacent. BR is not covered
            # by this bound and is not free whatever the drains above say -- a band
            # overrunning into it reads back what the mainloop left there.
            assert store_c.lds_bytes() <= (NBB + _NOBUF) * _B_SLOT
        else:
            store_c = StoreCPlain(C, m_end, c_n, mfma.idx, N_TILES_A, N_TILES_BH, _out_ty, ilv=_BILV)
        if const_expr(not _CSTORE):
            _cst = None
        elif glu:
            # The up band rides its own SRD rather than a gap: I*2 overflows the immediate.
            _cst = store_c.fused_operands(base_row, base_col_l)
        else:
            _cst = store_c.fused_operands(base_row, base_col_l, base_col_r, n_valid=_NV)
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
            half_n=_half_n,
            half_k=bool(_K128),
            split=(a_od6, bl_od6, br_od6, qu_a6, qu_b6),
            cst=_cst,
            cst_gap=0 if glu else LDS_BN_HALF * 2,
            cst_ilv=_BILV,
            cst_nt=cst_nt,
            b_base_even=b_even6,
        )
        if glu and const_expr(glu_act_quant):
            # The staging borrows the BL pool, so the mainloop's in-flight ds_reads
            # and the next tile's g2s prefetch have to retire first -- see the dglu
            # path below for why it takes two vmcnt drains rather than one.
            _lds_barrier(vmcnt=0)
            _lds_barrier(vmcnt=0)
            store_c.store_pair_quant(accL, accR, base_row, base_col_l, pad_row_base, m_end)
        elif glu:
            # accL is gate, accR the up column it pairs with: the R pool read the up band.
            store_c.store_pair(accL, accR, base_row, base_col_l)
        elif dglu:
            # The band borrows the BL pool, so the mainloop's in-flight ds_reads and
            # g2s prefetch must retire before the first staging write. Both fences
            # have to drain vmcnt, and both are needed: one leaves a g2s the
            # scheduler placed after the first drain free to overwrite the band.
            _lds_barrier(vmcnt=0)
            _lds_barrier(vmcnt=0)
            if const_expr(dglu_act_quant):
                store_c.store_pair_quant(accL, accR, base_row, base_col_l, base_col_r, pad_row_base, m_end)
            else:
                store_c.store_pair(accL, accR, base_row, base_col_l, base_col_r)
        elif const_expr(not _CSTORE):
            store_c.store(accL, base_row, base_col_l, n_valid=_NV)
            store_c.store(accR, base_row, base_col_r, n_valid=_NV)

    if dglu_act_quant:

        @flyc.kernel(known_block_size=[256, 1, 1])
        def kern(
            A: fx.Tensor,
            B_T: fx.Tensor,
            # dl1 never reaches HBM here, so l1 stands in for the store stream too.
            L1: fx.Tensor,  # l1 [total_M, 2I] in
            PROBS: fx.Tensor,  # [total_M] fp32
            GRAD_PROBS: fx.Tensor,  # fp32 partials
            AQ_OUT: fx.Tensor,  # grad_l1 row-wise fp4 [total_M, 2I/8] i32
            AQ_SC: fx.Tensor,  # grad_l1 row-wise E8M0 [total_M, 2I/32] i8
            AQ_TOUT: fx.Tensor,  # grad_l1 col-wise fp4 [2I, pad_M/8] i32
            AQ_TSC: fx.Tensor,  # grad_l1 col-wise E8M0 [2I, pad_M/32] i8
            A_scale: fx.Tensor,
            B_scale: fx.Tensor,
            GO: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
            slab_rows: fx.Int32,
            gp_stride: fx.Int32,
            aq_col_rows: fx.Int32,  # the col-wise operand's 256-aligned row extent
            sr_seed: fx.Int32,
        ):
            _body(
                A,
                B_T,
                L1,
                L1,
                PROBS,
                GRAD_PROBS,
                A_scale,
                B_scale,
                GO,
                c_m,
                c_n,
                slab_rows,
                gp_stride,
                AQ_OUT,
                AQ_SC,
                AQ_TOUT,
                AQ_TSC,
                aq_col_rows,
                sr_seed,
            )

    elif glu_act_quant:

        @flyc.kernel(known_block_size=[256, 1, 1])
        def kern(
            A: fx.Tensor,
            B_T: fx.Tensor,
            C: fx.Tensor,  # l1 [total_M, 2I]
            PROBS: fx.Tensor,  # [total_M] fp32
            AQ_OUT: fx.Tensor,  # act row-wise fp4 [total_M, I/8] i32
            AQ_SC: fx.Tensor,  # act row-wise E8M0 [total_M, I/32] i8
            AQ_TOUT: fx.Tensor,  # act col-wise fp4 [I, pad_M/8] i32
            AQ_TSC: fx.Tensor,  # act col-wise E8M0 [I, pad_M/32] i8
            A_scale: fx.Tensor,
            B_scale: fx.Tensor,
            GO: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
            slab_rows: fx.Int32,
            aq_col_rows: fx.Int32,  # the col-wise operand's 256-aligned row extent
            sr_seed: fx.Int32,
        ):
            _body(
                A,
                B_T,
                C,
                None,
                PROBS,
                None,
                A_scale,
                B_scale,
                GO,
                c_m,
                c_n,
                slab_rows,
                None,
                AQ_OUT,
                AQ_SC,
                AQ_TOUT,
                AQ_TSC,
                aq_col_rows,
                sr_seed,
            )

    elif glu or dglu:

        @flyc.kernel(known_block_size=[256, 1, 1])
        def kern(
            A: fx.Tensor,
            B_T: fx.Tensor,
            C: fx.Tensor,  # glu: l1 [total_M, 2I];  dglu: dl1 [total_M, 2I]
            ACT: fx.Tensor,  # glu: act [total_M, I] out;  dglu: l1 [total_M, 2I] in
            PROBS: fx.Tensor,  # [total_M] fp32
            GRAD_PROBS: fx.Tensor,  # dglu only: fp32 partials, aliased to C under glu
            A_scale: fx.Tensor,
            B_scale: fx.Tensor,
            GO: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
            slab_rows: fx.Int32,
            gp_stride: fx.Int32,
        ):
            _body(A, B_T, C, ACT, PROBS, GRAD_PROBS, A_scale, B_scale, GO, c_m, c_n, slab_rows, gp_stride)

    else:

        @flyc.kernel(known_block_size=[256, 1, 1])
        def kern(
            A: fx.Tensor,
            B_T: fx.Tensor,
            C: fx.Tensor,
            A_scale: fx.Tensor,
            B_scale: fx.Tensor,
            GO: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
            slab_rows: fx.Int32,
        ):
            _body(A, B_T, C, None, None, None, A_scale, B_scale, GO, c_m, c_n, slab_rows, None)

    _pt = {"passthrough": [["amdgpu-agpr-alloc", "256"]]}
    attrs = {"rocdl.flat_work_group_size": "256,256", "rocdl.waves_per_eu": OCC, **_pt}
    return kern, attrs, NBK, _BILV


_GMXFP4_LAUNCH_CACHE: dict = {}
_GMXFP4_WS_CACHE: dict = {}
_GMXFP4_AT_CACHE: dict = {}  # (total_M, N, K, G, gm, xcd, gn, out_fp16) -> [raw_launch, compiled]
_GMXFP4_NT_CFG = (4, 8, 0, 16, False)
# Thin groups: the write-only C stream evicts re-read weights, so non-temporal buys them back.
_GMXFP4_NT_CFG_THIN = (4, 8, 0, 16, True)
_GMXFP4_WGRAD_CFG = (2, 1, 4, False, 1, False)
_GMXFP4_WGRAD_CFG_SHORT = (4, 1, 6, True, 2, True)  # short per-group contraction: see selector
# When a group's tiles outnumber the CUs the band shape alone decides residency: narrower M.
_GMXFP4_WGRAD_CFG_SHORT_SPAN = (2, 1, 8, True, 2, True)
_GMXFP4_WGRAD_SHORT_MG = 8192  # per-group contraction at/below which the short-M blocking applies
_GMXFP4_CACHE_CAP = 32  # drop caches past this; real MoE uses few shapes, a test sweep many
_N_CU = 256  # gfx950 compute units, i.e. the width of one dispatch generation


def _bound_caches(*caches):
    if any(len(c) > _GMXFP4_CACHE_CAP for c in caches):
        for c in caches:
            c.clear()
        gc.collect()


def _select_gmxfp4_nt_cfg(total_M, G):
    """Pick the NT tile blocking from the runtime shape (host-side extents only). A group at
    least ``num_xcds`` bands wide (in M-blocks) gives every XCD a band inside one expert slab;
    below that width every XCD sees a different expert, so that regime gets its own blocking."""
    _gm, xcd, _gn, span, _nt = _GMXFP4_NT_CFG
    mb = ceildiv(total_M // max(G, 1), _BLOCK)
    return _GMXFP4_NT_CFG if mb >= xcd * span else _GMXFP4_NT_CFG_THIN


def _compile_grouped_mxfp4_nt_fused(
    K, G, N, gm, xcd, gn, wlv, elgk, out_fp16, k_real=None, span=16, cst_nt=False
):
    K128 = K // 128
    N_SCALE = ceildiv(N, 256) * 256
    k128_rd = (K if k_real is None else k_real) // 128  # real raw K128 (scale not host-padded)
    gemm_k, attrs, NBK, b_ilv = _build_grouped_mxfp4_nt_kernel(
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
        cst_nt=cst_nt,
    )
    ab_pre_shuf = _build_grouped_mxfp4_ab_preshuffle(K128, G, N, k128_rd, b_ilv=b_ilv)  # 1 launch
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


def _compile_grouped_mxfp4_nt_glu(
    K,
    G,
    N,
    gm,
    xcd,
    gn,
    wlv,
    elgk,
    out_fp16,
    k_real=None,
    span=16,
    glu=False,
    dglu=False,
    glu_i=0,
    glu_quant_row=False,
    fuse_act_quant=False,
    quant_total_M=0,
    epi_act_quant=False,
    dglu_epi_quant=False,
    epi_row_sr=False,
    epi_col_sr=False,
):
    """The NT compile of :func:`_compile_grouped_mxfp4_nt_fused` with a fused GLU epilogue.

    Kept apart from the plain entry rather than folded into it: the fused kernels carry
    three more tensor arguments and a stride, and the plain launch is the tuned production
    path whose argument list should not move for them.
    """
    K128 = K // 128
    N_b = 2 * glu_i if glu else N  # B rows per expert: gate||up under glu
    N_SCALE = ceildiv(N_b, 256) * 256
    k128_rd = (K if k_real is None else k_real) // 128  # real raw K128 (scale not host-padded)
    gemm_k, attrs, NBK, b_ilv = _build_grouped_mxfp4_nt_kernel(
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
        # l1 goes through the mainloop's store slot and is never re-read here, so
        # holding it out of L2 keeps the re-read A/B lines resident.
        cst_nt=True,
        glu=glu,
        dglu=dglu,
        glu_i=glu_i,
        # the row-wise quant epilogue is what replaces the bf16 act store
        glu_skip_act=glu_quant_row,
        glu_quant_row=glu_quant_row,
        glu_act_quant=epi_act_quant,
        dglu_act_quant=dglu_epi_quant,
        epi_row_sr=epi_row_sr,
        epi_col_sr=epi_col_sr,
    )
    ab_pre_shuf = _build_grouped_mxfp4_ab_preshuffle(
        K128, G, N_b, k128_rd, b_ilv=b_ilv, glu_i=glu_i if glu else 0
    )
    b_pre_grid = ceildiv(G * N_SCALE * K128, _PRESHUF_FO * _PRESHUF_BLK)

    quant_launch = None
    if fuse_act_quant:
        from primus_turbo.flydsl.quantization.mxfp4_grouped_quant import compile_grouped_mxfp4_qdual

        N_pad = (glu_i + 127) // 128 * 128
        M_pad_col = (quant_total_M + G * 256 + 255) // 256 * 256
        quant_launch = compile_grouped_mxfp4_qdual(
            quant_total_M,
            glu_i,
            G,
            M_pad_col,
            N_pad,
            False,
            True,
            is_fp16=out_fp16,
        )

    if dglu_epi_quant:

        @flyc.jit
        def launch(
            a8: fx.Tensor,
            b8: fx.Tensor,
            L1: fx.Tensor,
            PROBS: fx.Tensor,
            GRAD_PROBS: fx.Tensor,
            a_raw: fx.Tensor,
            b_raw: fx.Tensor,
            a_sp: fx.Tensor,
            b_sp: fx.Tensor,
            GO: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
            slab_rows: fx.Int32,
            gp_stride: fx.Int32,
            a_pre_grid: fx.Int32,
            grid_upper: fx.Int32,
            ROW_OUT: fx.Tensor,
            ROW_SC: fx.Tensor,
            COL_OUT: fx.Tensor,
            COL_SC: fx.Tensor,
            col_rows: fx.Int32,
            sr_seed: fx.Int32,
            stream: fx.Stream,
        ):
            ab_pre_shuf(a_raw, a_sp, b_raw, b_sp, GO, c_m, slab_rows, a_pre_grid).launch(
                grid=(a_pre_grid + b_pre_grid, 1, 1), block=(_PRESHUF_BLK, 1, 1), stream=stream
            )
            gemm_k(
                a8,
                b8,
                L1,
                PROBS,
                GRAD_PROBS,
                ROW_OUT,
                ROW_SC,
                COL_OUT,
                COL_SC,
                a_sp,
                b_sp,
                GO,
                c_m,
                c_n,
                slab_rows,
                gp_stride,
                col_rows,
                sr_seed,
                value_attrs=attrs,
            ).launch(grid=(grid_upper, 1, 1), block=(256, 1, 1), stream=stream)

    elif epi_act_quant:

        @flyc.jit
        def launch(
            a8: fx.Tensor,
            b8: fx.Tensor,
            C: fx.Tensor,
            PROBS: fx.Tensor,
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
            ROW_OUT: fx.Tensor,
            ROW_SC: fx.Tensor,
            COL_OUT: fx.Tensor,
            COL_SC: fx.Tensor,
            col_rows: fx.Int32,
            sr_seed: fx.Int32,
            stream: fx.Stream,
        ):
            ab_pre_shuf(a_raw, a_sp, b_raw, b_sp, GO, c_m, slab_rows, a_pre_grid).launch(
                grid=(a_pre_grid + b_pre_grid, 1, 1), block=(_PRESHUF_BLK, 1, 1), stream=stream
            )
            gemm_k(
                a8,
                b8,
                C,
                PROBS,
                ROW_OUT,
                ROW_SC,
                COL_OUT,
                COL_SC,
                a_sp,
                b_sp,
                GO,
                c_m,
                c_n,
                slab_rows,
                col_rows,
                sr_seed,
                value_attrs=attrs,
            ).launch(grid=(grid_upper, 1, 1), block=(256, 1, 1), stream=stream)

    elif fuse_act_quant:

        @flyc.jit
        def launch(
            a8: fx.Tensor,
            b8: fx.Tensor,
            C: fx.Tensor,
            ACT: fx.Tensor,
            ACT_Q: fx.Tensor,
            PROBS: fx.Tensor,
            GRAD_PROBS: fx.Tensor,
            a_raw: fx.Tensor,
            b_raw: fx.Tensor,
            a_sp: fx.Tensor,
            b_sp: fx.Tensor,
            GO: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
            slab_rows: fx.Int32,
            gp_stride: fx.Int32,
            a_pre_grid: fx.Int32,
            grid_upper: fx.Int32,
            ROW_OUT: fx.Tensor,
            ROW_SC: fx.Tensor,
            COL_OUT: fx.Tensor,
            COL_SC: fx.Tensor,
            LC: fx.Tensor,
            OC: fx.Tensor,
            SR_SEED: fx.Int32,
            stream: fx.Stream,
        ):
            ab_pre_shuf(a_raw, a_sp, b_raw, b_sp, GO, c_m, slab_rows, a_pre_grid).launch(
                grid=(a_pre_grid + b_pre_grid, 1, 1), block=(_PRESHUF_BLK, 1, 1), stream=stream
            )
            gemm_k(
                a8,
                b8,
                C,
                ACT,
                PROBS,
                GRAD_PROBS,
                a_sp,
                b_sp,
                GO,
                c_m,
                c_n,
                slab_rows,
                gp_stride,
                value_attrs=attrs,
            ).launch(grid=(grid_upper, 1, 1), block=(256, 1, 1), stream=stream)
            quant_launch(ACT_Q, ROW_OUT, ROW_SC, COL_OUT, COL_SC, GO, LC, OC, SR_SEED, stream)

    elif glu_quant_row:

        @flyc.jit
        def launch(
            a8: fx.Tensor,
            b8: fx.Tensor,
            C: fx.Tensor,
            ACT: fx.Tensor,
            PROBS: fx.Tensor,
            GRAD_PROBS: fx.Tensor,
            a_raw: fx.Tensor,
            b_raw: fx.Tensor,
            a_sp: fx.Tensor,
            b_sp: fx.Tensor,
            GO: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
            slab_rows: fx.Int32,
            gp_stride: fx.Int32,
            a_pre_grid: fx.Int32,
            grid_upper: fx.Int32,
            ROW_OUT: fx.Tensor,
            ROW_SC: fx.Tensor,
            SR_SEED: fx.Int32,
            stream: fx.Stream,
        ):
            ab_pre_shuf(a_raw, a_sp, b_raw, b_sp, GO, c_m, slab_rows, a_pre_grid).launch(
                grid=(a_pre_grid + b_pre_grid, 1, 1), block=(_PRESHUF_BLK, 1, 1), stream=stream
            )
            gemm_k(
                a8,
                b8,
                C,
                ACT,
                PROBS,
                GRAD_PROBS,
                a_sp,
                b_sp,
                GO,
                c_m,
                c_n,
                slab_rows,
                gp_stride,
                ROW_OUT,
                ROW_SC,
                SR_SEED,
                value_attrs=attrs,
            ).launch(grid=(grid_upper, 1, 1), block=(256, 1, 1), stream=stream)

    else:

        @flyc.jit
        def launch(
            a8: fx.Tensor,
            b8: fx.Tensor,
            C: fx.Tensor,
            ACT: fx.Tensor,
            PROBS: fx.Tensor,
            GRAD_PROBS: fx.Tensor,
            a_raw: fx.Tensor,
            b_raw: fx.Tensor,
            a_sp: fx.Tensor,
            b_sp: fx.Tensor,
            GO: fx.Tensor,
            c_m: fx.Int32,
            c_n: fx.Int32,
            slab_rows: fx.Int32,
            gp_stride: fx.Int32,
            a_pre_grid: fx.Int32,
            grid_upper: fx.Int32,
            stream: fx.Stream,
        ):
            ab_pre_shuf(a_raw, a_sp, b_raw, b_sp, GO, c_m, slab_rows, a_pre_grid).launch(
                grid=(a_pre_grid + b_pre_grid, 1, 1), block=(_PRESHUF_BLK, 1, 1), stream=stream
            )
            gemm_k(
                a8,
                b8,
                C,
                ACT,
                PROBS,
                GRAD_PROBS,
                a_sp,
                b_sp,
                GO,
                c_m,
                c_n,
                slab_rows,
                gp_stride,
                value_attrs=attrs,
            ).launch(grid=(grid_upper, 1, 1), block=(256, 1, 1), stream=stream)

    return launch, NBK


def _get_grouped_mxfp4_ws(total_M, N, K128, G, device):
    # key on static shape, grow the A-slab only for larger total_M so churn can't evict kernels
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

    k_real = K  # kernel tiles real N/K; the E8M0 scale is zero-padded to 256 in the preshuffle
    K256 = (K + 255) // 256 * 256
    au = a.contiguous().view(torch.uint8)  # [total_M, k_real/2] -- real K
    asu = a_scale.contiguous().view(torch.uint8)  # [total_M, k_real/32] -- real K
    bu = b.contiguous().view(torch.uint8)  # [G, N, k_real/2]
    bsu = b_scale.contiguous().view(torch.uint8)  # [G, N, k_real/32]
    K = K256
    K128 = K // 128

    a_raw = asu.contiguous().view(torch.int32).reshape(-1)
    b_raw = bsu.contiguous().view(torch.int32).reshape(-1)
    a8 = au.contiguous().view(torch.int8)  # keep multi-dim: 1D view of >2^31-elem MoE tensor overflows CABI
    b8 = bu.contiguous().view(torch.int8)
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
        gm, xcd, gn, span, nt = cfg
        lk = (K, G, N, gm, xcd, gn, span, nt, wlv, elgk, out_fp16, k_real)
        ent = _GMXFP4_LAUNCH_CACHE.get(lk)
        if ent is None:
            ent = _compile_grouped_mxfp4_nt_fused(
                K, G, N, gm, xcd, gn, wlv, elgk, out_fp16, k_real=k_real, span=span, cst_nt=nt
            )
            _GMXFP4_LAUNCH_CACHE[lk] = ent
        atk = (N, K, G, gm, xcd, gn, span, nt, out_fp16, k_real)  # same K256 diff real K must not collide
        e2 = _GMXFP4_AT_CACHE.get(atk)
        if e2 is None:
            e2 = [ent[0], None]
            _GMXFP4_AT_CACHE[atk] = e2
        return e2

    _run_mxfp4_sched(_entry(_select_gmxfp4_nt_cfg(total_M, G)), args, 1)
    _bound_caches(_GMXFP4_LAUNCH_CACHE, _GMXFP4_AT_CACHE, _GMXFP4_WS_CACHE)
    return out[:, :N_out] if N_out != N else out


# WGRAD (variable-K TN via NT compute): C[g] = lhs[:, g] @ rhs[:, g]^T over per-group padded M


def _build_grouped_mxfp4_wgrad_kernel(
    OUT_M,
    OUT_N,
    G,
    M_total,
    group_m=4,
    num_xcds=8,
    group_n=0,
    wlv=10,
    elgk=9,
    out_fp16=False,
    cst_nt=False,
    wg_tiles=1,
    half_m=False,
    beta_is_one=False,  # epilogue accumulates (C += acc) instead of overwriting
):
    BLOCK_M = BLOCK_N = BLOCK_K = _BLOCK
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
    # Last M block is half padding, so idling wave_m==1 would not shorten the tile; instead it
    # re-points its operand and store at the R half and runs the R-dropped body.
    _HALF_M = half_m and (OUT_M % BLOCK_M != 0) and (OUT_M % BLOCK_M <= BLOCK_M // 2)
    # The in-loop fused store cannot read C back, so an accumulate takes the standalone one.
    _CSTORE = (not out_fp16) and not beta_is_one
    _BILV = N_TILES_BH if (_CSTORE and LDS_ROW_STRIDE == 128 and N_TILES_BH == 4) else 0
    TOTAL = G * TILES_PER_GROUP
    # One WG walks WGT tiles from opposite ends; backfires once the halved grid under-covers.
    _WGT = wg_tiles if (wg_tiles > 1 and TOTAL % wg_tiles == 0 and TOTAL // wg_tiles >= _N_CU) else 1
    GRID = TOTAL // _WGT

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

        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_BH, packed=True, wlv=wlv, elgk=elgk)
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, M_total, N_LDS_STEPS_A, BPR, swizzle=swizzle)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, M_total, N_LDS_STEPS_BH, BPR, swizzle=swizzle, ilv=_BILV)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, LDS_ROW_STRIDE, swizzle=swizzle)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_BH, LDS_ROW_STRIDE, swizzle=swizzle)
        _qm = ceildiv(OUT_M, 256) * 256
        _qn = ceildiv(OUT_N, 256) * 256
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
        if const_expr(_HALF_M):
            a_s2r_h = S2RLoaderFp4(0, N_TILES_A, LDS_ROW_STRIDE, swizzle=swizzle)
            a_base6_h = [
                [a_s2r_h.base_addr(A_buf[b], s) for s in range_constexpr(N_SUB)]
                for b in range_constexpr(NABUF)
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

        # Lane-resident group table: bounds cost a v_readlane, not a load occ=1 cannot hide.
        go_rs = buffer_ops.create_buffer_resource(GO, max_size=False, num_records_bytes=(G + 1) * 8)
        go_tbl = _lane_tbl_load(go_rs, lane_id, G + 1, stride=2)
        # LPT order: tile cost tracks token count, so two half-sums pick the walk direction.
        _go_half = _lane_tbl_get(go_tbl, I32(G // 2))
        _head_tokens = _go_half - _lane_tbl_get(go_tbl, I32(0))
        _tail_tokens = _lane_tbl_get(go_tbl, I32(G)) - _go_half
        _tail_heavy = _tail_tokens > _head_tokens
        for _tt in range_constexpr(_WGT):
            if const_expr(_tt):
                _lds_barrier()  # the previous tile's ds_reads must retire before its buffers refill
            # Sub-tiles alternate ends sequentially: co-residency would cost more turnaround.
            if const_expr(_tt % 2 == 0):
                _pl = fx.block_idx.x + I32((_tt // 2) * GRID)
            else:
                _pl = I32(TOTAL - 1 - (_tt // 2) * GRID) - fx.block_idx.x
            pid_lin = _readfirstlane_i32(arith.select(_tail_heavy, I32(TOTAL - 1) - _pl, _pl))
            pid = xcd_remap_pid(pid_lin, I32(TOTAL), num_xcds)
            group_idx, block_m, block_n = _wgrad_block_mn(
                pid, G, TILES_PER_GROUP, N_BLOCKS_M, N_BLOCKS_N, group_m, group_n, False
            )
            _gi = _readfirstlane_i32(group_idx)
            m_start = _lane_tbl_get(go_tbl, _gi)
            m_end = _lane_tbl_get(go_tbl, _gi + I32(1))
            nval = (m_end - m_start) // I32(256)  # raw zero/even/odd 256-block count

            a_row = block_m * I32(BLOCK_M)
            b_row = block_n * I32(BLOCK_N)
            # M-side half tile: wave_m == 1 re-points at the R half and runs the R-dropped body.
            if const_expr(_HALF_M):
                _lm = block_m == I32(N_BLOCKS_M - 1)
                if const_expr(_HALF_N):
                    _ln = block_n == I32(N_BLOCKS_N - 1)
                    _hmf = I32(arith.select(_ln, I32(0), arith.select(_lm, I32(1), I32(0))))
                else:
                    _hmf = I32(arith.select(_lm, I32(1), I32(0)))
                _hm = _hmf == I32(1)
                _hmw = (_hmf + wave_m) == I32(2)  # this wave swaps onto the R column half
            # fold row base + contraction start into the int64 SRDs: large OUT_M/M_total pass 2^31
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
            if const_expr(_HALF_M):
                sa_b = I32(arith.select(_hm, a_row, sa_b))
            ksb = (m_start // I32(256)) * I32(_SCVSTEP)  # contraction-start scale byte offset

            for _pp in range_constexpr(0, _PRELL):
                a_g2s.load(A_buf[_pp], a_off + _pp * KSTEP)
            # B stops one ring slot short: the whole-loop asm issues and drains it behind its k=0 loads
            for _pp in range_constexpr(0, _PRELL - 1):
                bl_g2s.load(BL_buf[_pp], bl_off + _pp * KSTEP)
                br_g2s.load(BR_buf[_pp], br_off + _pp * KSTEP)

            accL = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)
            accR = [mfma.zero_value] * (N_TILES_A * N_TILES_BH)
            soff6_a = rocdl.readfirstlane(T.i32, a_off + fx.Int32(_PRELL * KSTEP))
            _blo = bl_off + fx.Int32(_PRELL * KSTEP)
            _bro = br_off + fx.Int32(_PRELL * KSTEP)
            if const_expr(_HALF_M and _HALF_N):
                # half_g2s off here, so a boundary N block's padding R half folds onto L rows.
                _bro = I32(arith.select(_ln, _blo, _bro))
            soff6_bl = rocdl.readfirstlane(T.i32, _blo)
            soff6_br = rocdl.readfirstlane(T.i32, _bro)
            _sc1 = _scsoff(sa_b, 64, ksb)
            _sc3 = _scsoff(sbr_b, 0, ksb)
            _wia = sa_b // I32(128)
            _wib = (sbl_b // I32(256)) * I32(2) + (sbl_b % I32(256)) // I32(64)
            _sob_v = _wib * I32(K128m) * I32(512) + ksb
            if const_expr(_HALF_M):
                _sob_v = I32(arith.select(_hmw, _sob_v + I32(8), _sob_v))
            _soa = rocdl.readfirstlane(T.i32, _wia * I32(K128m) * I32(512) + ksb)
            _sob = rocdl.readfirstlane(T.i32, _sob_v)
            sc_soff06 = [_soa, _sc1, _sob, _sc3]
            _half_n = None
            if const_expr(_HALF_N or _HALF_M):
                _hnv = I32(0)
                if const_expr(_HALF_N):
                    _hnv = I32(arith.select(block_n == I32(N_BLOCKS_N - 1), I32(1), _hnv))
                if const_expr(_HALF_M):
                    _hnv = I32(arith.select(_lm, I32(1), _hnv))
                _half_n = _readfirstlane_i32(_hnv)
            base_row = group_idx * I32(OUT_M) + a_row + I32(wave_m_off)
            base_col_l = b_row + I32(wave_n_off)
            base_col_r = b_row + I32(LDS_BN_HALF) + I32(wave_n_off)
            if const_expr(_HALF_M):
                base_row = I32(arith.select(_hm, group_idx * I32(OUT_M) + a_row, base_row))
                base_col_l = I32(arith.select(_hmw, base_col_r, base_col_l))
            _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
            store_c = StoreCPlain(
                C,
                (group_idx + I32(1)) * I32(OUT_M),
                OUT_N,
                mfma.idx,
                N_TILES_A,
                N_TILES_BH,
                _out_ty,
                ilv=_BILV,
                beta_is_one=beta_is_one,
            )
            _cst = store_c.fused_operands(base_row, base_col_l, base_col_r, n_valid=_NV) if _CSTORE else None
            if const_expr(_HALF_M):
                a_base_t = [
                    [arith.select(_hmw, a_base6_h[b][s], a_base6[b][s]) for s in range_constexpr(N_SUB)]
                    for b in range_constexpr(NABUF)
                ]
                bl_base_t = [
                    [arith.select(_hmw, br_base6[b][s], bl_base6[b][s]) for s in range_constexpr(N_SUB)]
                    for b in range_constexpr(NBB)
                ]
            else:
                a_base_t, bl_base_t = a_base6, bl_base6
            accL, accR = mfma.call_mxfp4_wholeloop(
                a_base_t,
                bl_base_t,
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
                half_g2s=not _HALF_M,
                cst=_cst,
                cst_gap=LDS_BN_HALF * 2,
                cst_ilv=_BILV,
                cst_nt=cst_nt,
            )
            if const_expr(not _CSTORE):
                store_c.store(accL, base_row, base_col_l, n_valid=_NV)
                store_c.store(accR, base_row, base_col_r, n_valid=_NV)

    _pt = {"passthrough": [["amdgpu-agpr-alloc", "256"]]}
    attrs = {"rocdl.flat_work_group_size": "256,256", "rocdl.waves_per_eu": OCC, **_pt}
    return kern, attrs, GRID, _BILV


_GMXFP4_WGRAD_LAUNCH_CACHE: dict = {}
_GMXFP4_WGRAD_WS_CACHE: dict = {}
_GMXFP4_WGRAD_AT_CACHE: dict = {}  # (OUT_M_p, OUT_N_p, M_alloc, G, out_fp16, beta1) -> [raw, compiled]


def _get_grouped_mxfp4_wgrad_ws(OUT_M, OUT_N, K128m, device):
    key = (OUT_M, OUT_N, K128m, device)
    e = _GMXFP4_WGRAD_WS_CACHE.get(key)
    if e is None:
        qm = ceildiv(OUT_M, 256) * 256
        qn = ceildiv(OUT_N, 256) * 256
        a_sp = torch.empty(qm * K128m, dtype=torch.int32, device=device)
        b_sp = torch.empty(qn * K128m, dtype=torch.int32, device=device)
        e = (a_sp, b_sp)
        _GMXFP4_WGRAD_WS_CACHE[key] = e
    return e


def _select_gmxfp4_wgrad_cfg(M_total, G, OUT_M=0, OUT_N=0):
    """Pick the wgrad tile blocking from the mean per-group contraction: a short one writes
    G x more C per FLOP and needs the non-temporal store, and splits again on the per-group
    tile count, where a group wider than one CU keeps its own rows resident."""
    if M_total // max(G, 1) > _GMXFP4_WGRAD_SHORT_MG:
        return _GMXFP4_WGRAD_CFG
    if ceildiv(OUT_M, _BLOCK) * ceildiv(OUT_N, _BLOCK) > _N_CU:
        return _GMXFP4_WGRAD_CFG_SHORT_SPAN
    return _GMXFP4_WGRAD_CFG_SHORT


def _compile_grouped_mxfp4_wgrad_fused(
    OUT_M, OUT_N, G, M_total, gm, xcd, gn, nt, wgt, hm, wlv, elgk, out_fp16, beta_is_one=False
):
    K128m = M_total // 128
    gemm_k, attrs, GRID, b_ilv = _build_grouped_mxfp4_wgrad_kernel(
        OUT_M,
        OUT_N,
        G,
        M_total,
        group_m=gm,
        num_xcds=xcd,
        group_n=gn,
        wlv=wlv,
        elgk=elgk,
        out_fp16=out_fp16,
        cst_nt=nt,
        wg_tiles=wgt,
        half_m=hm,
        beta_is_one=beta_is_one,
    )
    pre_ab = _build_mxfp4_preshuffle_kernel_ab(b_ilv=b_ilv)  # b_ilv: rhs scale follows rhs row map
    _PGRID = _MXFP4_PRESHUF_FO * _MXFP4_PRESHUF_BLK
    QM = ceildiv(OUT_M, 256) * 256  # 256-rounded packed extent; surplus rows masked off the read
    QN = ceildiv(OUT_N, 256) * 256

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
        grid_a = ceildiv(fx.Int32(QM) * fx.Int32(K128m), _PGRID)
        grid_b = ceildiv(fx.Int32(QN) * fx.Int32(K128m), _PGRID)
        pre_ab(
            a_raw,
            a_sp,
            b_raw,
            b_sp,
            fx.Int32(QM),
            fx.Int32(QN),
            fx.Int32(OUT_M),
            fx.Int32(OUT_N),
            fx.Int32(K128m),
            grid_a,
        ).launch(grid=(grid_a + grid_b, 1, 1), block=(_MXFP4_PRESHUF_BLK, 1, 1), stream=stream)
        gemm_k(a8, b8, C, a_sp, b_sp, GO, value_attrs=attrs).launch(
            grid=(GRID, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch, GRID


def grouped_gemm_mxfp4_variable_k_flydsl_kernel(
    lhs,
    lhs_scale,
    rhs,
    rhs_scale,
    group_offs,
    OUT_M,
    OUT_N,
    G,
    out_dtype=torch.bfloat16,
    num_cu=-1,
    beta=0.0,
    out=None,
):
    """MXFP4 grouped variable-K wgrad (bare-asm whole-loop) -> C [G, OUT_M, OUT_N].

    ``lhs``/``rhs`` use the colwise 256-aligned quant layout and ``group_offs`` contains
    the matching padded offsets. The NT whole-loop runs at a runtime ``nval`` without an
    on-GPU repack. ``beta=1.0`` accumulates into ``out`` instead of overwriting it.
    """
    assert lhs.ndim == 2 and rhs.ndim == 2
    assert lhs.shape[0] == OUT_M and rhs.shape[0] == OUT_N
    M_total = lhs.shape[1] * 2  # colwise contraction width (256-padded per group by the quant)
    assert rhs.shape[1] * 2 == M_total
    dev = lhs.device
    out_fp16 = out_dtype == torch.float16

    # keep fp4 operands 2D: a flat view of >2^31-int8 total_M overflows the CABI int32 dim
    a8 = lhs.contiguous().view(torch.int8)
    b8 = rhs.contiguous().view(torch.int8)
    a_raw = lhs_scale.contiguous().view(torch.int32).reshape(-1)
    b_raw = rhs_scale.contiguous().view(torch.int32).reshape(-1)
    go_pad = (group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)).view(torch.int32)

    K128m = M_total // 128
    a_sp, b_sp = _get_grouped_mxfp4_wgrad_ws(OUT_M, OUT_N, K128m, dev)
    # 3D: a 1D G*OUT_M*OUT_N view overflows the CABI int32 dim on large-G grad_b.
    out = resolve_accum_out(out, beta, (G, OUT_M, OUT_N), dev, out_dtype)
    beta_is_one = beta == 1.0

    stream = torch.cuda.current_stream()
    wlv, elgk = 10, 9
    args = (a8, b8, out, a_raw, b_raw, a_sp, b_sp, go_pad, stream)

    def _entry(cfg):
        gm, xcd, gn, nt, wgt, hm = cfg
        lk = (OUT_M, OUT_N, G, M_total, gm, xcd, gn, nt, wgt, hm, wlv, elgk, out_fp16, beta_is_one)
        ent = _GMXFP4_WGRAD_LAUNCH_CACHE.get(lk)
        if ent is None:
            ent = _compile_grouped_mxfp4_wgrad_fused(
                OUT_M, OUT_N, G, M_total, gm, xcd, gn, nt, wgt, hm, wlv, elgk, out_fp16, beta_is_one
            )
            _GMXFP4_WGRAD_LAUNCH_CACHE[lk] = ent
        atk = (OUT_M, OUT_N, M_total, G, gm, xcd, gn, nt, wgt, hm, out_fp16, beta_is_one)
        e2 = _GMXFP4_WGRAD_AT_CACHE.get(atk)
        if e2 is None:
            e2 = [ent[0], None]
            _GMXFP4_WGRAD_AT_CACHE[atk] = e2
        return e2

    run_eager_or_capture(_entry(_select_gmxfp4_wgrad_cfg(M_total, G, OUT_M, OUT_N)), args, 1)
    _bound_caches(_GMXFP4_WGRAD_LAUNCH_CACHE, _GMXFP4_WGRAD_AT_CACHE, _GMXFP4_WGRAD_WS_CACHE)
    return out
