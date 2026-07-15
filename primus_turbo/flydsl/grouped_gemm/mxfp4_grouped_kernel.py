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

import gc

import torch

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
    _MXFP4_PRESHUF_BLK,
    _MXFP4_PRESHUF_NG,
    MfmaScaleFp4,
    S2RLoaderFp4,
    ScaleS2RPacked,
    StoreCPlain,
    _build_mxfp4_preshuffle_kernel_ab,
    _mxfp4_grp_from,
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
_PRESHUF_NG = 4  # g_byte fan-out per preshuffle thread
# Trailing-128 handling: KI (full 256-block count) >= this uses the zero-waste 128-tail;
# below it, round up + zero-scale pad (avoids the tail's serial g2s+barrier bubble, which
# only amortizes on long K). Env-overridable for tuning.
import os as _os  # noqa: E402

# Trailing-128: scale-pad-zero (round KI up, last block's s=1 * zero-padded scale) beats the
# "zero-waste" 128-tail once the other overheads are gone -- the tail's fresh g2s + barrier
# is a serial bubble larger than the one padded MFMA. So default the 128-tail OFF (high
# threshold => scale-pad-zero); the tail stays available via the env knob.
_MXFP4_TAIL_MIN = int(_os.environ.get("MXFP4_TAIL_MIN", "100000"))


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


def _build_grouped_mxfp4_ab_preshuffle(K128: int, G: int, N: int, k128_rd: int = None):
    """Merged A-slab + B-per-expert scale preshuffle in ONE launch (matches the fp8/dense
    single-preshuffle structure -> one fewer in-stream kernel launch + gap per grouped GEMM,
    a bigger relative win on the small/short-K shapes). Blocks [0, a_grid) do the A slab
    (mode 0, inline a_pre scan over GO); [a_grid, ...) do the B per-expert (mode 1). The
    two paths are computed then segment-selected (SGPR rsrc via arith.select, values via
    select) -- no per-thread divergence. ``k128_rd`` real read + 256-block mask = zero pad,
    no host F.pad."""
    _KRD = K128 if k128_rd is None else k128_rd
    n_sub, nd, KK = 2, 4, K128 // 2
    b_dwords_pe = N * K128 // _PRESHUF_NG

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
        a_rin = buffer_ops.create_buffer_resource(a_raw, max_size=False, num_records_bytes=total_M * I32(_KRD) * 4)
        a_rout = buffer_ops.create_buffer_resource(a_out, max_size=False, num_records_bytes=slab_rows * I32(K128) * 4)
        b_rin = buffer_ops.create_buffer_resource(b_raw, max_size=False, num_records_bytes=I32(G * N * _KRD) * 4)
        b_rout = buffer_ops.create_buffer_resource(b_out, max_size=False, num_records_bytes=I32(G * N * K128) * 4)
        bid = rocdl.readfirstlane(T.i32, fx.block_idx.x)
        is_b = bid >= a_grid
        local = arith.select(is_b, bid - a_grid, bid)
        gid_all = local * I32(_PRESHUF_BLK) + fx.thread_idx.x
        rin = arith.select(is_b, b_rin, a_rin)
        rout = arith.select(is_b, b_rout, a_rout)

        b_expert = gid_all // I32(b_dwords_pe)
        a_total4 = slab_rows * I32(K128) // I32(_PRESHUF_NG)
        gid4 = arith.select(is_b, gid_all - b_expert * I32(b_dwords_pe), gid_all)
        total4 = arith.select(is_b, I32(b_dwords_pe), a_total4)
        last = gid4 % I32(nd)
        e1 = gid4 // I32(nd)
        r = e1 % I32(16)
        e2 = e1 // I32(16)
        kk = e2 % I32(KK)
        wi = e2 // I32(KK)
        r_region = last // I32(n_sub)
        s = last % I32(n_sub)
        k128 = kk * I32(n_sub) + s
        grp_a = _mxfp4_grp_from(wi, r_region, 0)
        grp_b = _mxfp4_grp_from(wi, r_region, 1)
        _blk = ((wi * I32(KK) + kk) * I32(64) + r) * I32(nd) + last
        base = arith.select(is_b, b_expert * I32(N * K128) + _blk, _blk)

        # A: inline a_pre scan (owning group -> tight source rows rd0..rd_end)
        go_t = rocdl.make_buffer_tensor(go_out, max_size=False, num_records_bytes=(G + 1) * 8)
        go_div = fx.logical_divide(go_t, fx.make_layout(1, 1))
        rd0 = I32(0)
        rd_end = I32(0)
        ok_a = I32(0)
        apre = I32(0)
        m_prev = _load_go(go_div, 0)
        for g in range_constexpr(G):
            m_nxt = _load_go(go_div, g + 1)
            p0 = apre
            p1 = apre + ((m_nxt - m_prev + I32(255)) // I32(256)) * I32(4)
            inq = (grp_a >= p0) & (grp_a < p1)
            rd0 = arith.select(inq, m_prev + (grp_a - p0) * I32(64), rd0)
            rd_end = arith.select(inq, m_nxt, rd_end)
            ok_a = arith.select(inq, I32(1), ok_a)
            apre = p1
            m_prev = m_nxt
        rd_base = b_expert * I32(N)  # B source row base

        okc = arith.select(is_b, (gid4 < I32(b_dwords_pe)) & (b_expert < I32(G)), (gid4 < a_total4) & (ok_a != I32(0)))
        okc = okc & (k128 < I32(_KRD)) & (gid4 < total4)
        rsrc_rows = [
            arith.select(is_b, rd_base + grp_b * I32(64) + I32(t * 16) + r, rd0 + I32(t * 16) + r)
            for t in range_constexpr(4)
        ]
        valid = [
            okc & arith.select(is_b, grp_b * I32(64) + I32(t * 16) + r < I32(N), rsrc_rows[t] < rd_end)
            for t in range_constexpr(4)
        ]
        words = _pack4(rin, rsrc_rows, k128, valid, _KRD)
        # store ALL in-range blocks: invalid/pad blocks got words=0 from the masked reads,
        # so slab-pad / 256-pad regions are written 0 (matches the split a_pre_shuf).
        for g in range_constexpr(_PRESHUF_NG):
            buffer_ops.buffer_store(words[g], rout, base + I32(g * 64), mask=gid4 < total4)

    return kern


def _build_grouped_mxfp4_nt_kernel(
    K, G, N, group_m=4, num_xcds=8, group_n=0, wlv=10, elgk=9, out_fp16=False, k_real=None
):
    """Grouped MXFP4 NT (out = a @ b^T), per-group A rows + per-expert B, whole-loop compute.

    K = 256-rounded tile/scale extent (the tiny E8M0 scale is zero-padded to it so the
    preshuffle packs whole 256-blocks). ``k_real`` (<= K, 128-multiple) = the operands' TRUE
    contraction (row stride, no operand pad). When k_real%256==128 the loop runs
    k_real//256 full 256-blocks + ONE 128-K tail MFMA (reads the real last 128 K, its scale
    is the block's s=0 in the padded packing) -- zero operand copy, zero wasted compute."""
    BLOCK_M = BLOCK_N = BLOCK_K = _BLOCK
    _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
    swizzle = True
    _KR = K if k_real is None else k_real  # operand true contraction (128-multiple)
    assert K % 256 == 0 and _KR % 128 == 0
    KI = _KR // BLOCK_K  # FULL 256-blocks over the REAL K; trailing 128 -> 128-tail
    _K128 = (_KR // 128) % 2  # 1 => there is a trailing 128-K block
    # 128-tail (zero-waste) vs scale-pad-zero for the trailing 128: MEASURED -- scale-pad-zero
    # wins even against a bubble-free tail (operand g2s hidden behind the odd-256 tail MFMAs).
    # Reason: scale-pad-zero folds the extra block into the do-while so it is FULLY PIPELINED
    # (one wasted zero-scale MFMA), whereas any 128-tail is a separate post-loop phase whose
    # barrier+ds_read breaks the pipeline by more than that one MFMA costs. Keep the env
    # override (MXFP4_TAIL_MIN) for A/B only; default => scale-pad-zero everywhere.
    _USE_TAIL = bool(_K128) and KI >= _MXFP4_TAIL_MIN
    _K128TAIL = 1 if _USE_TAIL else 0
    KI_LOOP = KI if (_USE_TAIL or not _K128) else (KI + 1)
    NABUF, NBB, OCC = 2, 2, 1
    N_SUB = BLOCK_K // 128
    BPR = BLOCK_K // 2
    KSTEP = BPR
    K2 = _KR // 2  # operand row stride (bytes) = real K (no operand K-pad)
    N_TILES_A = BLOCK_M // 32
    # block_n=128 => n_halves=1: the 128-N tile IS the "BL" span (2 wave_n x 64 N), no BR.
    # LDS_BN_HALF stays 128 (per-2-wave N span) in both cases; only tile width + BR differ.
    _n_halves = 1 if BLOCK_N < 256 else 2
    LDS_BN_HALF = BLOCK_N if _n_halves == 1 else BLOCK_N // 2
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
    _NV = N if (N % BLOCK_N != 0) else None  # non-256 N: mask store cols >= N (no host N-pad)

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
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, _KR, N_LDS_STEPS_A, BPR, swizzle=swizzle)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, _KR, N_LDS_STEPS_BH, BPR, swizzle=swizzle)
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
        apre = I32(0)  # inline 64-grp slab-base cumsum (no AP tensor / cumsum launch)
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
            a_pre_g = arith.select(inq, apre, a_pre_g)
            apre = apre + ceildiv(nx - p2, BLOCK_M) * I32(4)
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
            if const_expr(KI_LOOP > _pp):
                a_g2s.load(A_buf[_pp], a_off + _pp * KSTEP)
        for _pp in range_constexpr(0, _PRELL):
            if const_expr(KI_LOOP > _pp):
                bl_g2s.load(BL_buf[_pp], bl_off + _pp * KSTEP)
                if const_expr(_n_halves == 2):
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
        if const_expr(_n_halves == 1):
            # block_n=128: tile bn's N-64-groups (2bn, 2bn+1) live in tile (bn//2)'s BL
            # (even bn -> r_region=0) or BR (odd bn -> r_region=1) packed region. Select the
            # region soffset by bn parity; BR slot unused so point it at a valid soffset.
            bn256 = bn // I32(2)
            is_odd = (bn % I32(2)) != I32(0)
            sbl_e = bn256 * I32(256) + I32(wave_n_off)
            sbr_o = bn256 * I32(256) + I32(128) + I32(wave_n_off)
            _wib_e = (sbl_e // I32(256)) * I32(2) + (sbl_e % I32(256)) // I32(64)
            sob_e = b_exp_bytes + _wib_e * I32(K128) * I32(512)
            sc3_o = b_exp_bytes + _scsoff(sbr_o, 0)
            _sob = rocdl.readfirstlane(T.i32, arith.select(is_odd, sc3_o, sob_e))
            _sc3 = _sob
        else:
            _sc3 = rocdl.readfirstlane(T.i32, b_exp_bytes + _scsoff(sbr_b, 0))
            _wib = (sbl_b // I32(256)) * I32(2) + (sbl_b % I32(256)) // I32(64)
            _sob = rocdl.readfirstlane(T.i32, b_exp_bytes + _wib * I32(K128) * I32(512))
        sc_soff06 = [_soa, _sc1, _sob, _sc3]
        _tail128 = None
        if const_expr(_K128TAIL):
            # trailing 128-K block (block KI): operand + s=0 scale soffsets. _scvt = the
            # whole-loop's per-256-block scale advance (64*(2*N_SUB)*4).
            _scvt = 64 * (2 * N_SUB) * 4
            _tail128 = (
                rocdl.readfirstlane(T.i32, a_off + I32(KI * KSTEP)),
                rocdl.readfirstlane(T.i32, bl_off + I32(KI * KSTEP)),
                rocdl.readfirstlane(T.i32, br_off + I32(KI * KSTEP)),
                rocdl.readfirstlane(T.i32, _soa + I32(KI * _scvt)),
                rocdl.readfirstlane(T.i32, _sob + I32(KI * _scvt)),
            )
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
            tail128=_tail128,
            n_halves=_n_halves,
        )
        base_row = m_row + I32(wave_m_off)
        base_col_l = bn * I32(BLOCK_N) + I32(wave_n_off)
        # store bounded to the group's tight end: StoreCPlain's SRD num_records =
        # m_end*c_n -> partial-tile rows >= m_end (next group) HW-drop.
        store_c = StoreCPlain(C, m_end, c_n, mfma.idx, N_TILES_A, N_TILES_BH, _out_ty)
        store_c.store(accL, base_row, base_col_l, n_valid=_NV)
        if const_expr(_n_halves == 2):
            base_col_r = bn * I32(BLOCK_N) + I32(LDS_BN_HALF) + I32(wave_n_off)
            store_c.store(accR, base_row, base_col_r, n_valid=_NV)

    _pt = {"passthrough": [["amdgpu-agpr-alloc", "256"]]}
    attrs = {"rocdl.flat_work_group_size": "256,256", "rocdl.waves_per_eu": OCC, **_pt}
    return kern, attrs, NBK


_GMXFP4_LAUNCH_CACHE: dict = {}
_GMXFP4_WS_CACHE: dict = {}
_GMXFP4_AT_CACHE: dict = {}  # (total_M, N, K, G, gm, xcd, gn, out_fp16) -> [raw_launch, compiled]
# Fixed grouped NT config: xcd=1 (group-major XCD order) is the dominant L2-locality lever
# (xcd=8 measured ~20% slower); gm=2/gn=4 (N-band) is within ~1.5% of the per-shape optimum
# across aligned + fat-N + square shapes. A single fixed config keeps ONE compiled kernel
# per shape (a timed 7-candidate sweep exploded compile time + code-object memory on the
# full test sweep of ~480 shapes -> segfault), while retaining the xcd=1 win.
_GMXFP4_DEFAULT_CFG = tuple(int(x) for x in _os.environ.get("MXFP4_CFG", "2,1,4").split(","))  # (gm,xcd,gn)
_MXFP4_SMALL_CFG = tuple(int(x) for x in _os.environ.get("MXFP4_SMALL_CFG", "1,8,0").split(","))  # few-tile
# xcd=1 (2,1,4) wins across the board once the scale F.pad + split preshuffle overheads are
# gone (an earlier xcd=8 few-tile heuristic was tuned against that stale overhead and now
# regresses the small-square shapes); keep the knob but default it off (grid<0 never fires).
_MXFP4_SMALL_GRID = int(_os.environ.get("MXFP4_SMALL_GRID", "0"))  # grid_upper < this -> small cfg
_MXFP4_WLV = int(_os.environ.get("MXFP4_WLV", "10"))  # whole-loop vmcnt-in-flight hint (sweepable)
_MXFP4_ELGK = int(_os.environ.get("MXFP4_ELGK", "9"))  # whole-loop lgkmcnt-at-barrier hint
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


def _compile_grouped_mxfp4_nt_fused(K, G, N, gm, xcd, gn, wlv, elgk, out_fp16, k_real=None):
    K128 = K // 128
    k128_rd = (K if k_real is None else k_real) // 128  # real raw K128 (scale not host-padded)
    ab_pre_shuf = _build_grouped_mxfp4_ab_preshuffle(K128, G, N, k128_rd)  # merged A+B, 1 launch
    gemm_k, attrs, NBK = _build_grouped_mxfp4_nt_kernel(
        K, G, N, group_m=gm, num_xcds=xcd, group_n=gn, wlv=wlv, elgk=elgk, out_fp16=out_fp16, k_real=k_real
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
        gemm_k(a8, b8, C, a_sp, b_sp, GO, c_m, c_n, slab_rows, grid_upper, value_attrs=attrs).launch(
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
    a8 = au.contiguous().view(torch.int8).reshape(-1)
    b8 = bu.contiguous().view(torch.int8).reshape(-1)
    out = torch.empty((total_M, N), dtype=out_dtype, device=dev)
    out_flat = out.view(-1)

    go = (group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)).view(torch.int32)
    a_sp, b_sp, slab_rows = _get_grouped_mxfp4_ws(total_M, N, K128, G, dev)

    n_blocks = (N + 255) // 256
    grid_upper = (ceildiv(total_M, 256) + G) * n_blocks
    a_pre_grid = ceildiv(slab_rows * K128, _PRESHUF_NG * _PRESHUF_BLK)

    stream = torch.cuda.current_stream()
    wlv, elgk = _MXFP4_WLV, _MXFP4_ELGK
    args = (
        a8,
        b8,
        out_flat,
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
        gm, xcd, gn = cfg
        lk = (K, G, N, gm, xcd, gn, wlv, elgk, out_fp16, k_real)
        ent = _GMXFP4_LAUNCH_CACHE.get(lk)
        if ent is None:
            ent = _compile_grouped_mxfp4_nt_fused(K, G, N, gm, xcd, gn, wlv, elgk, out_fp16, k_real=k_real)
            _GMXFP4_LAUNCH_CACHE[lk] = ent
        atk = (total_M, N, K, G, gm, xcd, gn, out_fp16)
        e2 = _GMXFP4_AT_CACHE.get(atk)
        if e2 is None:
            e2 = [ent[0], None]
            _GMXFP4_AT_CACHE[atk] = e2
        return e2

    # Deterministic config by output-tile count (1 compile/shape, no timed autotune):
    # few tiles under-fill the CUs -> xcd=8 spreads WGs across all 8 XCDs for occupancy
    # (measured +20-30% on the small/short-K worst shapes); many tiles -> xcd=1 group-major
    # for L2 locality (xcd=8 there regresses ~20%). The large-shape (2,1,4) default is kept.
    cfg = _MXFP4_SMALL_CFG if grid_upper < _MXFP4_SMALL_GRID else _GMXFP4_DEFAULT_CFG
    run_eager_or_capture(_entry(cfg), args, 1)
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
    NABUF, NBB, OCC = 2, 2, 1
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

        gA = make_fp8_buffer_tensor(A, F8)
        gB = make_fp8_buffer_tensor(B_T, F8)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        mfma = MfmaScaleFp4(N_TILES_A, N_TILES_BH, packed=True, wlv=wlv, elgk=elgk)
        gl_off_a = fp4_g2s_offsets(lane_id, wave_id, M_total, N_LDS_STEPS_A, BPR, swizzle=swizzle)
        gl_off_b = fp4_g2s_offsets(lane_id, wave_id, M_total, N_LDS_STEPS_BH, BPR, swizzle=swizzle)
        rsrc_a = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=I32(OUT_M) * I32(M2))
        rsrc_b = buffer_ops.create_buffer_resource(B_T, max_size=False, num_records_bytes=I32(OUT_N) * I32(M2))
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8, wave_id)
        bl_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8, wave_id)
        br_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_BH, F8, wave_id)
        a_s2r = S2RLoaderFp4(wave_m, N_TILES_A, LDS_ROW_STRIDE, swizzle=swizzle)
        b_s2r = S2RLoaderFp4(wave_n, N_TILES_BH, LDS_ROW_STRIDE, swizzle=swizzle)
        _qm = ((OUT_M + 63) // 64) * 64
        _qn = ((OUT_N + 63) // 64) * 64
        sa_s2r = ScaleS2RPacked(A_scale, _qm, M_total, 4)
        sb_s2r = ScaleS2RPacked(B_scale, _qn, M_total, 4)
        wave_m_off = wave_m * (N_TILES_A * 16)
        wave_n_off = wave_n * (N_TILES_BH * 16)

        a_base6 = [[a_s2r.base_addr(A_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NABUF)]
        bl_base6 = [[b_s2r.base_addr(BL_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)]
        br_base6 = [[b_s2r.base_addr(BR_buf[b], s) for s in range_constexpr(N_SUB)] for b in range_constexpr(NBB)]

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
            fx.ptrtoint(fx.add_offset(SC_buf[b].ptr, fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW) + lane_id)))
            for b in range_constexpr(_NSCBUF)
        ]
        sc_gb6 = [
            rocdl.readfirstlane(
                T.i32,
                fx.Int32(fx.ptrtoint(fx.add_offset(SC_buf[b].ptr, fx.make_int_tuple(fx.Int32(wave_id) * fx.Int32(_SCW))))),
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
        a_off = a_row * I32(M2) + (m_start >> 1)
        bl_off = b_row * I32(M2) + (m_start >> 1)
        br_off = (b_row + I32(LDS_BN_HALF)) * I32(M2) + (m_start >> 1)
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
        accL, accR = mfma.call_mxfp4_wholeloop(
            a_base6, bl_base6, br_base6, a_s2r.tile_stride, b_s2r.tile_stride,
            abase6, blbase6, brbase6, gl_a6, gl_b6, rsrc_a, rsrc_b, fx.Int32(KSTEP), scv6,
            accL, accR, N_SUB, N_LDS_STEPS_A, N_LDS_STEPS_BH, nval,
            soff6_a, soff6_bl, soff6_br, sc_rb6, sc_gb6, _scrsa_v, _scrsb_v, sc_voff6, sc_soff06,
            ki=None, sc_buf_stride=(_SCBUF * 4),
        )
        base_row = group_idx * I32(OUT_M) + a_row + I32(wave_m_off)
        base_col_l = b_row + I32(wave_n_off)
        base_col_r = b_row + I32(LDS_BN_HALF) + I32(wave_n_off)
        store_c = StoreCPlain(C, (group_idx + I32(1)) * I32(OUT_M), OUT_N, mfma.idx, N_TILES_A, N_TILES_BH, _out_ty)
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
    _PGRID = _MXFP4_PRESHUF_NG * _MXFP4_PRESHUF_BLK

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

    a8 = lhs.contiguous().view(torch.int8).reshape(-1)
    b8 = rhs.contiguous().view(torch.int8).reshape(-1)
    a_raw = lhs_scale.contiguous().view(torch.int32).reshape(-1)
    b_raw = rhs_scale.contiguous().view(torch.int32).reshape(-1)
    go_pad = (group_offs if group_offs.dtype == torch.int64 else group_offs.to(torch.int64)).view(torch.int32)

    K128m = M_total // 128
    a_sp, b_sp = _get_grouped_mxfp4_wgrad_ws(OUT_M, OUT_N, K128m, dev)
    out = torch.empty((G, OUT_M, OUT_N), dtype=out_dtype, device=dev)
    out_flat = out.view(-1)

    stream = torch.cuda.current_stream()
    wlv, elgk = _MXFP4_WLV, _MXFP4_ELGK
    args = (a8, b8, out_flat, a_raw, b_raw, a_sp, b_sp, go_pad, stream)

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

    run_eager_or_capture(_entry(_GMXFP4_DEFAULT_CFG), args, 1)
    _bound_caches(_GMXFP4_WGRAD_LAUNCH_CACHE, _GMXFP4_WGRAD_AT_CACHE, _GMXFP4_WGRAD_WS_CACHE)
    return out
