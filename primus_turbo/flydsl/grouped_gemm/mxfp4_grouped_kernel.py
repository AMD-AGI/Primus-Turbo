###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
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

import torch
import torch.nn.functional as F

# isort: off
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    _readfirstlane_i32,
    ceildiv,
    make_fp8_buffer_tensor,
    wait_barrier,
    xcd_remap_pid,
)
from primus_turbo.flydsl.gemm.mxfp4_gemm_kernel import (
    MfmaScaleFp4,
    S2RLoaderFp4,
    ScaleS2RPacked,
    StoreCPlain,
    _mxfp4_grp_from,
    fp4_g2s_offsets,
)
from primus_turbo.flydsl.grouped_gemm.gemm_fp8_grouped_kernel import _grouped_block_mn, _load_go

# isort: on

_BLOCK = 256  # BLOCK_M = BLOCK_N = BLOCK_K
_PRESHUF_BLK = 256
_PRESHUF_NG = 4  # g_byte fan-out per preshuffle thread


def _pack4(rin, rows, k128, valid, K128):
    """Pack the 4 source E8M0 rows (byte t <- rows[t]) into NG dwords; returns list[NG]."""
    I32 = fx.Int32
    dws = []
    for t in range_constexpr(4):
        dws.append(
            fx.Int32(
                buffer_ops.buffer_load(
                    rin, rows[t] * I32(K128) + k128, vec_width=1, dtype=T.i32, mask=valid[t]
                )
            )
        )
    out = []
    for g in range_constexpr(_PRESHUF_NG):
        sh = I32(g * 8)
        p = I32(0)
        for t in range_constexpr(4):
            p = p | (((dws[t] >> sh) & I32(0xFF)) << I32(t * 8))
        out.append(p)
    return out


def _build_grouped_mxfp4_a_preshuffle(K128: int, G: int):
    """Repack canonical A E8M0 [total_M, K128] -> per-group 256-aligned packed slabs.

    Grid: one thread per NG-folded output dword over the padded slab space. Each thread
    decodes its slab row-group ``grp`` (dense mxfp4 map), finds the owning group via
    ``a_pre`` (64-row-group cumulative), reads the 4 tight source rows go[g]+... (pad
    rows past go[g+1] -> 0), and packs. ``a_pre[g]`` is 4*ceil(M_g/256) so each slab is
    256-aligned."""
    n_sub, nd, KK = 2, 4, K128 // 2

    @flyc.kernel(known_block_size=[_PRESHUF_BLK, 1, 1])
    def kern(
        a_raw: fx.Tensor,
        a_out: fx.Tensor,
        go_out: fx.Tensor,  # tight offs (int32 view int64 [G+1])
        a_pre: fx.Tensor,  # 64-grp slab base (int32 view int64 [G+1])
        total_M: fx.Int32,
        slab_rows: fx.Int32,
    ):
        I32 = fx.Int32
        rin = buffer_ops.create_buffer_resource(
            a_raw, max_size=False, num_records_bytes=total_M * I32(K128) * 4
        )
        rout = buffer_ops.create_buffer_resource(
            a_out, max_size=False, num_records_bytes=slab_rows * I32(K128) * 4
        )
        gid4 = fx.block_idx.x * I32(_PRESHUF_BLK) + fx.thread_idx.x
        total4 = slab_rows * I32(K128) // I32(_PRESHUF_NG)
        last = gid4 % I32(nd)
        e1 = gid4 // I32(nd)
        r = e1 % I32(16)
        e2 = e1 // I32(16)
        kk = e2 % I32(KK)
        wi = e2 // I32(KK)
        r_region = last // I32(n_sub)
        s = last % I32(n_sub)
        k128 = kk * I32(n_sub) + s
        base = ((wi * I32(KK) + kk) * I32(64) + r) * I32(nd) + last
        grp = _mxfp4_grp_from(wi, r_region, 0)  # slab row-group

        go_t = rocdl.make_buffer_tensor(go_out, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go_t, fx.make_layout(1, 1))
        ap_t = rocdl.make_buffer_tensor(a_pre, max_size=False, num_records_bytes=(G + 1) * 8)
        ap_div = fx.logical_divide(ap_t, fx.make_layout(1, 1))
        rd0 = I32(0)
        rd_end = I32(0)
        ok = I32(0)
        for g in range_constexpr(G):
            p0 = _load_go(ap_div, g)
            p1 = _load_go(ap_div, g + 1)
            inq = (grp >= p0) & (grp < p1)
            rd0 = arith.select(inq, _load_go(go_div, g) + (grp - p0) * I32(64), rd0)
            rd_end = arith.select(inq, _load_go(go_div, g + 1), rd_end)
            ok = arith.select(inq, I32(1), ok)
        base_ok = (gid4 < total4) & (ok != I32(0))
        rows = [rd0 + I32(t * 16) + r for t in range_constexpr(4)]
        valid = [base_ok & (rows[t] < rd_end) for t in range_constexpr(4)]
        words = _pack4(rin, rows, k128, valid, K128)
        for g in range_constexpr(_PRESHUF_NG):
            buffer_ops.buffer_store(words[g], rout, base + I32(g * 64), mask=gid4 < total4)

    return kern


def _build_grouped_mxfp4_b_preshuffle(K128: int, G: int, N: int):
    """Repack canonical B E8M0 [G*N, K128] -> per-expert packed slabs (dense map per
    expert, expert = block // blocks_per_expert)."""
    n_sub, nd, KK = 2, 4, K128 // 2
    dwords_pe = N * K128 // _PRESHUF_NG  # output threads per expert

    @flyc.kernel(known_block_size=[_PRESHUF_BLK, 1, 1])
    def kern(b_raw: fx.Tensor, b_out: fx.Tensor):
        I32 = fx.Int32
        rin = buffer_ops.create_buffer_resource(
            b_raw, max_size=False, num_records_bytes=I32(G * N * K128) * 4
        )
        rout = buffer_ops.create_buffer_resource(
            b_out, max_size=False, num_records_bytes=I32(G * N * K128) * 4
        )
        gid_all = fx.block_idx.x * I32(_PRESHUF_BLK) + fx.thread_idx.x
        expert = gid_all // I32(dwords_pe)
        gid4 = gid_all - expert * I32(dwords_pe)
        total4 = I32(dwords_pe)
        last = gid4 % I32(nd)
        e1 = gid4 // I32(nd)
        r = e1 % I32(16)
        e2 = e1 // I32(16)
        kk = e2 % I32(KK)
        wi = e2 // I32(KK)
        r_region = last // I32(n_sub)
        s = last % I32(n_sub)
        k128 = kk * I32(n_sub) + s
        base = expert * I32(N * K128) + ((wi * I32(KK) + kk) * I32(64) + r) * I32(nd) + last
        grp = _mxfp4_grp_from(wi, r_region, 1)
        rd_base = expert * I32(N)
        ok = (gid4 < total4) & (expert < I32(G))
        rows = [rd_base + grp * I32(64) + I32(t * 16) + r for t in range_constexpr(4)]
        valid = [ok & (grp * I32(64) + I32(t * 16) + r < I32(N)) for t in range_constexpr(4)]
        words = _pack4(rin, rows, k128, valid, K128)
        for g in range_constexpr(_PRESHUF_NG):
            buffer_ops.buffer_store(words[g], rout, base + I32(g * 64), mask=ok)

    return kern


def _build_grouped_mxfp4_nt_kernel(K, G, N, group_m=4, num_xcds=8, group_n=0, wlv=10, elgk=9, out_fp16=False):
    """Grouped MXFP4 NT (out = a @ b^T), per-group A rows + per-expert B, whole-loop compute."""
    BLOCK_M = BLOCK_N = BLOCK_K = _BLOCK
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    swizzle = True
    assert K % BLOCK_K == 0
    KI = K // BLOCK_K
    NABUF, NBB, OCC = 2, 2, 1
    N_SUB = BLOCK_K // 128
    BPR = BLOCK_K // 2
    KSTEP = BPR
    K2 = K // 2
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
    _SCBUF = 4 * 4 * (BLOCK_K // 128) * 64
    _SCW = 4 * N_SUB * 64
    NBK = ceildiv(N, BLOCK_N)  # n_blocks

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
        AP: fx.Tensor,  # 64-grp A-slab base (int32 view int64 [G+1])
        c_m: fx.Int32,  # total_M
        c_n: fx.Int32,  # N
        slab_rows: fx.Int32,  # padded A-slab rows
        grid_upper: fx.Int32,
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

        # ---- tile-independent setup ----
        gA = make_fp8_buffer_tensor(A, F8)
        gB = make_fp8_buffer_tensor(B_T, F8)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_BH, packed=True, wlv=wlv, elgk=elgk)
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_A, BPR, swizzle=swizzle)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, K, N_LDS_STEPS_BH, BPR, swizzle=swizzle)
        rsrc_a = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=c_m * K2)
        rsrc_b = buffer_ops.create_buffer_resource(B_T, max_size=False, num_records_bytes=I32(G) * c_n * K2)
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8, wave_id)
        bl_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8, wave_id)
        br_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8, wave_id)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, LDS_ROW_STRIDE, swizzle=swizzle)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_BH, LDS_ROW_STRIDE, swizzle=swizzle)
        _qn = ((c_n + 63) // 64) * 64
        sa_s2r = ScaleS2RPacked(A_scale, slab_rows, K, 4)
        sb_s2r = ScaleS2RPacked(B_scale, _qn * I32(G), K, 4)
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

        # ---- O(G) tile scan: pid -> (group_idx, local tile, m_start, m_end, a_pre) ----
        go_t = rocdl.make_buffer_tensor(GO, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go_t, fx.make_layout(1, 1))
        ap_t = rocdl.make_buffer_tensor(AP, max_size=False, num_records_bytes=(G + 1) * 8)
        ap_div = fx.logical_divide(ap_t, fx.make_layout(1, 1))
        pid = xcd_remap_pid(fx.block_idx.x, grid_upper, num_xcds)
        total_tiles = I32(0)
        prev = _load_go(go_div, 0)
        for g in range_constexpr(G):
            nxt = _load_go(go_div, g + 1)
            total_tiles = total_tiles + ceildiv(nxt - prev, BLOCK_M) * I32(NBK)
            prev = nxt
        # non-persistent grid: WGs past total_tiles hardware-exit (scf.if cannot
        # carry the Python-object loader state, so guard via s_endpgm).
        _tt = _readfirstlane_i32(total_tiles)
        _llvm.inline_asm(
            None,
            [pid.ir_value(), arith._to_raw(_tt)],
            "s_cmp_lt_u32 $0, $1\n\ts_cbranch_scc1 1f\n\ts_endpgm\n\t1:",
            "s,s,~{scc},~{memory}",
            has_side_effects=True,
        )
        cum = I32(0)
        group_idx = I32(0)
        tile_start = I32(0)
        m_start = I32(0)
        m_end = I32(0)
        a_pre_g = I32(0)
        p2 = _load_go(go_div, 0)
        for g in range_constexpr(G):
            nx = _load_go(go_div, g + 1)
            tg = ceildiv(nx - p2, BLOCK_M) * I32(NBK)
            nc = cum + tg
            inq = (pid >= cum) & (pid < nc)
            group_idx = arith.select(inq, I32(g), group_idx)
            tile_start = arith.select(inq, cum, tile_start)
            m_start = arith.select(inq, p2, m_start)
            m_end = arith.select(inq, nx, m_end)
            a_pre_g = arith.select(inq, _load_go(ap_div, g), a_pre_g)
            cum = nc
            p2 = nx
        local = pid - tile_start
        bm, bn = _grouped_block_mn(local, m_start, m_end, NBK, BLOCK_M, group_m, group_n)

        m_row = m_start + bm * I32(BLOCK_M)  # tight A/C row base
        a_off = m_row * K2
        b_base = group_idx * c_n * K2
        bl_off = b_base + bn * I32(BLOCK_N) * K2
        br_off = b_base + (bn * I32(BLOCK_N) + I32(LDS_BN_HALF)) * K2
        # A-scale slab row base (256-aligned): a_pre_g*64 + bm*256 + wave off
        sa_b = a_pre_g * I32(64) + bm * I32(BLOCK_M) + I32(wave_m_off)
        sbl_b = bn * I32(BLOCK_N) + I32(wave_n_off)
        sbr_b = bn * I32(BLOCK_N) + I32(LDS_BN_HALF) + I32(wave_n_off)
        b_exp_bytes = group_idx * c_n * I32(K128) * I32(4)  # per-expert B-scale base (bytes)

        # ---- fill operand buffers ----
        for _pp in range_constexpr(0, _PRELL):
            if const_expr(KI > _pp):
                a_g2s.load(A_buf[_pp], a_off + _pp * KSTEP)
        for _pp in range_constexpr(0, _PRELL):
            if const_expr(KI > _pp):
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
        _sc3 = rocdl.readfirstlane(T.i32, b_exp_bytes + _scsoff(sbr_b, 0))
        _wia = sa_b // I32(128)
        _wib = (sbl_b // I32(256)) * I32(2) + (sbl_b % I32(256)) // I32(64)
        _soa = rocdl.readfirstlane(T.i32, _wia * I32(K128) * I32(512))
        _sob = rocdl.readfirstlane(T.i32, b_exp_bytes + _wib * I32(K128) * I32(512))
        sc_soff06 = [_soa, _sc1, _sob, _sc3]
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
            fx.Int32((KI // 2) * 2),
            soff6_a,
            soff6_bl,
            soff6_br,
            sc_rb6,
            sc_gb6,
            _scrsa_v,
            _scrsb_v,
            sc_voff6,
            sc_soff06,
            ki=KI,
            sc_buf_stride=(_SCBUF * 4),
        )
        base_row = m_row + I32(wave_m_off)
        base_col_l = bn * I32(BLOCK_N) + I32(wave_n_off)
        base_col_r = bn * I32(BLOCK_N) + I32(LDS_BN_HALF) + I32(wave_n_off)
        # store bounded to the group's tight end: StoreCPlain's SRD num_records =
        # m_end*c_n -> partial-tile rows >= m_end (next group) HW-drop.
        store_c = StoreCPlain(C, m_end, c_n, mfma.idx, N_TILES_A, N_TILES_BH, _out_ty)
        store_c.store(accL, base_row, base_col_l)
        store_c.store(accR, base_row, base_col_r)

    _pt = {"passthrough": [["amdgpu-agpr-alloc", "256"]]}
    attrs = {"rocdl.flat_work_group_size": "256,256", "rocdl.waves_per_eu": OCC, **_pt}
    return kern, attrs, NBK


def _build_a_pre_kernel(G: int):
    """1-thread prologue: a_pre[g] = cumulative 4*ceil(M_g/256) (64-row-grp units),
    so each group's A-scale slab is 256-aligned. a_pre[0]=0."""

    @flyc.kernel(known_block_size=[1, 1, 1])
    def kern(GO: fx.Tensor, AP: fx.Tensor):
        I32 = fx.Int32
        z = I32(0)
        go_t = rocdl.make_buffer_tensor(GO, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go_t, fx.make_layout(1, 1))
        ap_r = buffer_ops.create_buffer_resource(AP, max_size=False, num_records_bytes=I32((G + 1) * 8))
        buffer_ops.buffer_store(z, ap_r, I32(0))
        buffer_ops.buffer_store(z, ap_r, I32(1))
        acc = z
        prev = _load_go(go_div, 0)
        for g in range_constexpr(G):
            nxt = _load_go(go_div, g + 1)
            acc = acc + ((nxt - prev + I32(255)) // I32(256)) * I32(4)  # 4 64-grps per 256-slab
            buffer_ops.buffer_store(acc, ap_r, I32(2 * (g + 1)))
            buffer_ops.buffer_store(z, ap_r, I32(2 * (g + 1) + 1))
            prev = nxt

    return kern


_GMXFP4_LAUNCH_CACHE: dict = {}
_GMXFP4_WS_CACHE: dict = {}


def _compile_grouped_mxfp4_nt_fused(K, G, N, gm, xcd, gn, wlv, elgk, out_fp16):
    K128 = K // 128
    a_pre_k = _build_a_pre_kernel(G)
    a_pre_shuf = _build_grouped_mxfp4_a_preshuffle(K128, G)
    b_pre_shuf = _build_grouped_mxfp4_b_preshuffle(K128, G, N)
    gemm_k, attrs, NBK = _build_grouped_mxfp4_nt_kernel(
        K, G, N, group_m=gm, num_xcds=xcd, group_n=gn, wlv=wlv, elgk=elgk, out_fp16=out_fp16
    )
    b_pre_grid = ceildiv(G * N * K128, _PRESHUF_NG * _PRESHUF_BLK)

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
        AP: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        slab_rows: fx.Int32,
        a_pre_grid: fx.Int32,
        grid_upper: fx.Int32,
        stream: fx.Stream,
    ):
        a_pre_k(GO, AP).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
        a_pre_shuf(a_raw, a_sp, GO, AP, c_m, slab_rows).launch(
            grid=(a_pre_grid, 1, 1), block=(_PRESHUF_BLK, 1, 1), stream=stream
        )
        b_pre_shuf(b_raw, b_sp).launch(grid=(b_pre_grid, 1, 1), block=(_PRESHUF_BLK, 1, 1), stream=stream)
        gemm_k(a8, b8, C, a_sp, b_sp, GO, AP, c_m, c_n, slab_rows, grid_upper, value_attrs=attrs).launch(
            grid=(grid_upper, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch, NBK


def _get_grouped_mxfp4_ws(total_M, N, K128, G, device):
    key = (total_M, N, K128, G, device)
    e = _GMXFP4_WS_CACHE.get(key)
    if e is None:
        slab_rows = (ceildiv(total_M, 256) + G) * 256  # padded A-slab upper bound
        a_sp = torch.empty(slab_rows * K128, dtype=torch.int32, device=device)
        b_sp = torch.empty(G * N * K128, dtype=torch.int32, device=device)
        e = (a_sp, b_sp, slab_rows)
        _GMXFP4_WS_CACHE[key] = e
    return e


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

    # The whole-loop tiles K/N in 256 blocks; K (already 128-padded by the quant) or N
    # may be an odd 128-multiple. Pad both to 256 with zero fp4 / E8M0 (0*2^-127 = 0
    # contributes nothing), so every shape runs natively (no Triton fallback). The pad
    # is a small zero-fill on the fp4 / e8m0 uint8 tensors.
    K256 = (K + 255) // 256 * 256
    N256 = (N + 255) // 256 * 256
    au = a.contiguous().view(torch.uint8)  # [total_M, K/2]
    asu = a_scale.contiguous().view(torch.uint8)  # [total_M, K/32]
    bu = b.contiguous().view(torch.uint8)  # [G, N, K/2]
    bsu = b_scale.contiguous().view(torch.uint8)  # [G, N, K/32]
    if K256 != K:
        au = F.pad(au, (0, (K256 - K) // 2))
        asu = F.pad(asu, (0, (K256 - K) // 32))
        bu = F.pad(bu, (0, (K256 - K) // 2))
        bsu = F.pad(bsu, (0, (K256 - K) // 32))
    if N256 != N:
        bu = F.pad(bu, (0, 0, 0, N256 - N))
        bsu = F.pad(bsu, (0, 0, 0, N256 - N))
    K, N = K256, N256
    K128 = K // 128

    a_raw = asu.contiguous().view(torch.int32).reshape(-1)
    b_raw = bsu.contiguous().view(torch.int32).reshape(-1)
    a8 = au.contiguous().view(torch.int8).reshape(-1)
    b8 = bu.contiguous().view(torch.int8).reshape(-1)
    out = torch.empty((total_M, N), dtype=out_dtype, device=dev)
    out_flat = out.view(-1)

    go = (group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)).view(torch.int32)
    ap = torch.empty(G + 1, dtype=torch.int64, device=dev).view(torch.int32)
    a_sp, b_sp, slab_rows = _get_grouped_mxfp4_ws(total_M, N, K128, G, dev)

    n_blocks = (N + 255) // 256
    grid_upper = (ceildiv(total_M, 256) + G) * n_blocks
    a_pre_grid = ceildiv(slab_rows * K128, _PRESHUF_NG * _PRESHUF_BLK)

    stream = torch.cuda.current_stream()
    gm, xcd, gn, wlv, elgk = 4, 8, 0, 10, 9
    lk = (K, G, N, gm, xcd, gn, wlv, elgk, out_fp16)
    ent = _GMXFP4_LAUNCH_CACHE.get(lk)
    if ent is None:
        ent = _compile_grouped_mxfp4_nt_fused(K, G, N, gm, xcd, gn, wlv, elgk, out_fp16)
        _GMXFP4_LAUNCH_CACHE[lk] = ent
    launch, _NBK = ent
    args = (
        a8,
        b8,
        out_flat,
        a_raw,
        b_raw,
        a_sp,
        b_sp,
        go,
        ap,
        total_M,
        N,
        slab_rows,
        a_pre_grid,
        grid_upper,
        stream,
    )
    comp = flyc.compile(launch, *args)
    comp(*args)
    return out[:, :N_out] if N_out != N else out
