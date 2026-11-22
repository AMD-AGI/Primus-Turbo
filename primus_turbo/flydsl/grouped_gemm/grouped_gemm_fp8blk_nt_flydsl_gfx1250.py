"""FlyDSL gfx1250 FP8 block-scaled (UE8M0) M-grouped NT GEMM: the DeepSeek-V3/V4 FP8 recipe on WMMA-scale.

out[rows_g] = (a * 2^sa)[rows_g] @ (b[g] * 2^sb[g])^T,  a: [M_total, K] fp8, b: [G, N, K] fp8, out bf16.
The recipe's scales are power-of-two (UE8M0) per 1x128 activation tile and per 128x128 weight block. A 128
block is four MX blocks of 32 with the same exponent, so the GEMM is the gfx1250 MX WMMA
(v_wmma_scale_f32_16x16x128_f8f6f4, block 32) with each exponent repeated four times: the hardware applies
the scales, no fp32 promotion step in the K loop. Scale planes as the MX kernel reads them:
  sa [M_total, K/32] uint8 (row r, 32-block j -> 1x128 tile j//4 of row r),
  sb [G, N, K/32] uint8   (row n of expert g -> its 128x128 block (n//128, j//4)).
Derived from primus_turbo/flydsl/gemm/gemm_gfx1250_kernel.py (TDM + wave32 WMMA 16x16x128 fp8). The M axis is
cut per expert: grid = (M_total // tile_m + G) m-tiles x N tiles (an upper bound, no host sync); each m-tile
decodes (expert, first row, rows) from the group offsets with an unrolled scan, TDM clamps the ragged rows, and
dead tiles (past the last expert) run only the pipeline prologue with zero rows.
"""
from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL import MmaOpGFX1250_WMMAType
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Constexpr
from flydsl.expr.typing import T as _T

from primus_turbo.flydsl.utils import buffer_ops as _bops

WAVE = 32
_A_ID = {0: fx.Float8E4M3FN, 1: fx.Float8E5M2}


def _make_lds_load(bits):
    layout = fx.make_layout(bits // 32, 1)
    atom = fx.make_copy_atom(fx.UniversalCopy(bits), fx.Int32)
    ptr_ty = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, bits // 8)

    def load(base, byte_offset):
        ptr = fx.inttoptr(ptr_ty, fx.Int32(base) + fx.Int32(byte_offset))
        reg = fx.make_rmem_tensor(layout, fx.Int32)
        fx.copy_atom_call(atom, fx.Tensor(fx.make_view(ptr, layout)), reg)
        return reg.load()

    return load


def _store_c_lines(base_ptr, scr_off, lane, c_frags, m_reps, n_reps, warp_n, gC8, row0, col0, ldc64, out_cls,
                   lds_b128, cbase64=None, bands=1, sq_reg=None):
    """Store a wave's (m_reps*16) x warp_n accumulator block as bf16 with whole-line global stores.
    The WMMA C layout gives each lane 8 consecutive columns of one row, so a direct global_store_b128 of a fragment
    touches 16 rows x 32 B (16 cache lines per instruction); measured ~4-5 TB/s, +30-60 us per flygg call. Here each
    16-row band goes through a wave-private LDS scratch (16 x (2*warp_n + 16) B at scr_off) and is read back
    row-contiguous, so one global_store_b128 covers 32/LPR rows x 2*warp_n B (whole lines). LDS ops of one wave
    execute in order, so no barrier is needed. row0/col0: first output row/column of the wave block (elements).
    bands: 16-row bands staged per LDS round trip (scratch = bands x 16 x (2*warp_n + 16) B per wave).
    sq_reg: optional fp32 rmem vector (8): += the squares of the bf16 values, fragments in (mi, ni) order."""
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
                if sq_reg is not None:
                    hf = h.to(fx.Float32)
                    sq_reg.store(sq_reg.load() + hf * hf)
                fx.ptr_store(h.bitcast(fx.Int8),
                             base_ptr + (scr_off + (bi * 16 + lane16) * RB + (ni * 16 + kgrp * 8) * 2))
        vs = []
        for bi in range_constexpr(bands):
            for j in range_constexpr(16 // RPI):
                r = bi * 16 + j * RPI + lrow
                vs.append(fx.Vector(lds_b128(fx.Int32(fx.ptrtoint(base_ptr)), scr_off + r * RB + lcol * 16)))
        for bi in range_constexpr(bands):
            for j in range_constexpr(16 // RPI):
                r = j * RPI + lrow
                off = (fx.Int64(row0 + (mb * bands + bi) * 16 + r) * ldc64 + fx.Int64(col0 + lcol * 8)) * 2
                if cbase64 is not None:
                    off = off + cbase64 * 2
                fx.ptr_store(vs[bi * (16 // RPI) + j].bitcast(fx.Int8), gC8 + off)


@flyc.jit
def launch_gg_fp8blk_nt_gfx1250(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_offs: fx.Pointer,
    arg_sa: fx.Pointer,
    arg_sb: fx.Pointer,
    i32_mtiles: fx.Int32,
    stream: fx.Stream,
    i32_n: fx.Int32,
    i32_k: fx.Int32,
    n_groups: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    num_buffers: Constexpr[int],
    a_id: Constexpr[int],
    b_id: Constexpr[int],
    k_max: Constexpr[int] = 4096,
    dstore: Constexpr[int] = 0,
    lds_pad: Constexpr[int] = 16,
    bblk: Constexpr[int] = 0,
    xdl_nostall: Constexpr[int] = 0,
    bpf: Constexpr[int] = 0,
    hwsched: Constexpr[int] = 0,
):
    """xdl_nostall (moegemm2): set SCHED_MODE.DISABLE_XDL_ARB_STALL at kernel start (no WMMA multicycle
    arbitration stall between back-to-back matrix ops; scheduling only, bit-identical).
    bpf (mxgemm): fragment software pipeline inside a k-tile (needs tile_k >= 256 and m_reps == n_reps). The B
    fragments of 128-k step ks+1 are read one per row group during step ks (double-buffered B, +128 VGPR), the A
    fragment of row group mi+1 during mi, each group pinned with sched_barrier. Same WMMAs, operands and k order:
    bit-identical. (probe mxgemm E0c: flygg LDS pattern 15.3 -> 11.1 cycles/WMMA with B double-buffered.)
    hwsched (mxgemm): compiled WITHOUT amdgpu-expert-scheduling-mode (hardware dependency interlocks, as hipcc and
    Triton with TRITON_HIP_USE_EXPERT_SCHEDULING=0). In expert mode the compiler guards every LDS load that reuses a
    register an in-flight WMMA reads with s_wait_alu depctr_va_vdst(0), draining the WMMA pipe once per row group.
    Only the name differs in the IR (cache key); the wrapper flips the llvm option for that compile. Bit-identical."""
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    assert tile_m % (m_warp * WMMA_M) == 0 and tile_n % (n_warp * WMMA_N) == 0
    assert tile_k % WMMA_K == 0 and num_buffers >= 2
    assert tile_m % num_waves == 0 and tile_n % num_waves == 0
    k_iters = tile_k // WMMA_K
    out_cls = fx.BFloat16
    LDS_PAD = lds_pad
    A_ROW = tile_k + LDS_PAD
    B_ROW = tile_k + LDS_PAD
    STAGE_A = tile_m * A_ROW
    STAGE_B = tile_n * B_ROW
    MX_BLOCK = 32
    SC_ROW = tile_k // MX_BLOCK
    # the whole-K scale planes of the tile are loaded once (two TDM ops) instead of 4-byte-wide rows per
    # k-tile; row pitch padded by 4 B against bank conflicts
    assert k_max % tile_k == 0
    SK = k_max // MX_BLOCK
    SP = SK + 4
    PLANE_SA = tile_m * SP
    # bblk: the B plane keeps one row per 16-row WMMA fragment (128x128-block weight scales are constant over it)
    B_PROWS = tile_n // WMMA_N if bblk else tile_n
    PLANE_SB = (B_PROWS * SP + 15) // 16 * 16
    PITCH = STAGE_A + STAGE_B
    SC_OFF = num_buffers * PITCH
    C_ROW = tile_n + 8
    SCR_OFF = SC_OFF + PLANE_SA + PLANE_SB  # dstore: wave-private C line scratch after the planes
    SCR_W = 16 * (2 * (tile_n // n_warp) + 16)
    ARENA = max(SC_OFF + PLANE_SA + PLANE_SB + (num_waves * SCR_W if dstore else 0), tile_m * C_ROW * 2)
    assert ARENA <= 320 * 1024, f"LDS arena {ARENA} B exceeds 320 KiB"
    n_ops = 2
    G = n_groups

    kernel_name = f"moe_gg_fp8blk_nt_gfx1250_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}_nb{num_buffers}_a{a_id}{b_id}_g{G}" + ("_ds" if dstore else "") + (f"_p{lds_pad}" if lds_pad != 16 else "") + ("_bb" if bblk else "") + ("_xn" if xdl_nostall else "") + ("_bpf" if bpf else "") + ("_hw" if hwsched else "")
    if bpf:
        assert k_iters >= 2 and tile_m // m_warp == tile_n // n_warp, "bpf: tile_k >= 256, square wave tile"

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_gg(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_offs: fx.Pointer,
        arg_sa: fx.Pointer,
        arg_sb: fx.Pointer,
        i32_mtiles: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
    ):
        if const_expr(xdl_nostall):
            rocdl.disable_xdl_arb_stall()
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, _, _ = fx.block_idx
        n_tiles_n = (i32_n + (tile_n - 1)) // tile_n
        bid = fx.Int32(bid_x)
        t = rocdl.readfirstlane(_T.i32, bid // n_tiles_n)
        pn = bid % n_tiles_n
        blk_n = pn * tile_n
        n_oob = i32_n - blk_n

        i32_ptr = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        f32_ptr = fx.PointerType.get(elem_ty=fx.Float32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        mt = fx.recast_iter(i32_ptr, arg_offs)
        # unrolled scan: which expert owns m-tile t, its first row and row count
        zero = t * fx.Int32(0)
        one = zero + fx.Int32(1)
        acc = zero
        found = zero
        g_idx = zero
        first_g = zero
        len_g = zero
        row_beg = zero
        prev = mt[zero]
        for i in range_constexpr(G):
            nxt = mt[zero + fx.Int32(i + 1)]
            li = nxt - prev
            bi = (li + fx.Int32(tile_m - 1)) // fx.Int32(tile_m)
            acc2 = acc + bi
            take = (t < acc2).select(one, zero) * (one - found)
            g_idx = g_idx + take * (fx.Int32(i) - g_idx)
            first_g = first_g + take * (acc - first_g)
            len_g = len_g + take * (li - len_g)
            row_beg = row_beg + take * (prev - row_beg)
            found = found + take
            acc = acc2
            prev = nxt
        len_g = len_g * found
        row_beg = row_beg * found
        local = t - first_g
        raw = len_g - local * fx.Int32(tile_m)
        rows = (raw < fx.Int32(tile_m)).select(raw, fx.Int32(tile_m))
        rows = (rows > zero).select(rows, zero)
        m_row = (rows > zero).select(row_beg + local * fx.Int32(tile_m), zero)
        g_idx = rocdl.readfirstlane(_T.i32, g_idx)
        rows = rocdl.readfirstlane(_T.i32, rows)
        m_row = rocdl.readfirstlane(_T.i32, m_row)
        K_TILES = (rows > zero).select(i32_k // tile_k, fx.Int32(num_buffers - 1))


        lda64 = fx.Int64(i32_k)
        ldc64 = fx.Int64(i32_n)

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

        gA_base = _cast(a_cls, arg_a)
        gB_base = _cast(b_cls, arg_b)
        gC_base = _cast(out_cls, arg_c)
        gA = _gview(gA_base, fx.Int64(m_row) * lda64, (tile_m, tile_k), (i32_k, 1))
        gB = _gview(gB_base, (fx.Int64(g_idx) * fx.Int64(i32_n) + fx.Int64(blk_n)) * lda64, (tile_n, tile_k), (i32_k, 1))
        atomA = fx.rocdl.make_tdm_atom(gA, [rows, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)
        atomB = fx.rocdl.make_tdm_atom(gB, [n_oob, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)

        sk = i32_k // MX_BLOCK
        sk64 = fx.Int64(sk)
        gSA = _gview(_cast(fx.Uint8, arg_sa, 1), fx.Int64(m_row) * sk64, (tile_m, SK), (sk, 1))
        BST = WMMA_N if bblk else 1  # B plane row step (bblk: row 16*i of the tile)
        gSB = _gview(_cast(fx.Uint8, arg_sb, 1), (fx.Int64(g_idx) * fx.Int64(i32_n) + fx.Int64(blk_n)) * sk64,
                     (B_PROWS, SK), (sk * BST, 1))
        atomSA = fx.rocdl.make_tdm_atom(gSA, [rows, sk], strides=[sk64, None], num_warps=num_waves,
                                        pad_interval=SK, pad_amount=4)
        atomSB = fx.rocdl.make_tdm_atom(gSB, [(n_oob + (BST - 1)) // BST, sk], strides=[sk64 * BST, None],
                                        num_warps=num_waves, pad_interval=SK, pad_amount=4)
        sSA = _view(fx.add_offset(base_ptr, SC_OFF), fx.Uint8, (tile_m, SK), (SP, 1), 1)
        sSB = _view(fx.add_offset(base_ptr, SC_OFF + PLANE_SA), fx.Uint8, (B_PROWS, SK), (SP, 1), 1)

        def sA_of(s):
            return _view(_stage(s), a_cls, (tile_m, tile_k), (A_ROW, 1))

        def sB_of(s):
            return _view(fx.add_offset(_stage(s), STAGE_A), b_cls, (tile_n, tile_k), (B_ROW, 1))

        def issue(s, kt):
            kt64 = fx.Int64(kt)
            fx.copy(atomA, gA, sA_of(s), imm_offset=kt64 * tile_k)
            fx.copy(atomB, gB, sB_of(s), imm_offset=kt64 * tile_k)

        # slot 1 = N side (b), slot 2 = M side (a), as in the dense kernel
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
        for cf in c_frags:
            cf.fill(0.0)

        def _operand(base, plane_off, row, row_bytes, ks):
            b0 = plane_off + row * row_bytes + ks * WMMA_K + kgrp * 16
            v = [fx.Vector(lds_b128(base, b0 + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            reg = fx.make_rmem_tensor(16, fx.Int32)
            reg.store(v01.shuffle(v23, list(range(16))))
            return reg

        sc_base = fx.Int32(fx.ptrtoint(fx.add_offset(base_ptr, SC_OFF)))

        def _sbw(kc, ni):
            if const_expr(bblk):
                return lds_b32(sc_base, PLANE_SA + (wnb // WMMA_N + ni) * SP + kc)[0]
            return lds_b32(sc_base, PLANE_SA + (wnb + ni * WMMA_N + lane16) * SP + kc)[0]

        def compute_bpf(s, kt):
            base = fx.Int32(fx.ptrtoint(_stage(s)))
            bcur = [_operand(base, STAGE_A, wnb + ni * WMMA_N + lane16, B_ROW, 0) for ni in range_constexpr(n_reps)]
            sbv = [_sbw(kt * k_iters * 4, ni) for ni in range_constexpr(n_reps)]
            for ks in range_constexpr(k_iters):
                kc = (kt * k_iters + ks) * 4
                if const_expr(ks + 1 < k_iters):
                    sbn = [_sbw(kc + 4, ni) for ni in range_constexpr(n_reps)]  # next step's B scale words, early
                afs = [None] * m_reps
                savs = [None] * m_reps
                afs[0] = _operand(base, 0, wmb + lane16, A_ROW, ks)
                savs[0] = lds_b32(sc_base, (wmb + lane16) * SP + kc)[0]
                bnx = [None] * n_reps
                for mi in range_constexpr(m_reps):
                    if const_expr(mi + 1 < m_reps):
                        afs[mi + 1] = _operand(base, 0, wmb + (mi + 1) * WMMA_M + lane16, A_ROW, ks)
                        savs[mi + 1] = lds_b32(sc_base, (wmb + (mi + 1) * WMMA_M + lane16) * SP + kc)[0]
                    if const_expr(ks + 1 < k_iters):
                        bnx[mi] = _operand(base, STAGE_A, wnb + mi * WMMA_N + lane16, B_ROW, ks + 1)
                    rocdl.sched_barrier(0)
                    for ni in range_constexpr(n_reps):
                        idx = mi * n_reps + ni
                        fx.gemm(mma_atom, c_frags[idx], bcur[ni], afs[mi], c_frags[idx], scale_a=sbv[ni],
                                scale_b=savs[mi])
                    rocdl.sched_barrier(0)
                bcur = bnx
                if const_expr(ks + 1 < k_iters):
                    sbv = sbn

        def compute(s, kt):
            if const_expr(bpf):
                compute_bpf(s, kt)
                return
            base = fx.Int32(fx.ptrtoint(_stage(s)))
            for ks in range_constexpr(k_iters):
                kc = (kt * k_iters + ks) * 4  # byte column of this 128-wide k-step in the scale planes
                bf = [_operand(base, STAGE_A, wnb + ni * WMMA_N + lane16, B_ROW, ks) for ni in range_constexpr(n_reps)]
                if const_expr(bblk):
                    sbv = [lds_b32(sc_base, PLANE_SA + (wnb // WMMA_N + ni) * SP + kc)[0]
                           for ni in range_constexpr(n_reps)]
                else:
                    sbv = [lds_b32(sc_base, PLANE_SA + (wnb + ni * WMMA_N + lane16) * SP + kc)[0]
                           for ni in range_constexpr(n_reps)]
                for mi in range_constexpr(m_reps):
                    af = _operand(base, 0, wmb + mi * WMMA_M + lane16, A_ROW, ks)
                    sav = lds_b32(sc_base, (wmb + mi * WMMA_M + lane16) * SP + kc)[0]
                    for ni in range_constexpr(n_reps):
                        idx = mi * n_reps + ni
                        fx.gemm(mma_atom, c_frags[idx], bf[ni], af, c_frags[idx], scale_a=sbv[ni], scale_b=sav)

        # scale planes first: older than every stage, so the first stage wait covers them
        fx.copy(atomSA, gSA, sSA)
        fx.copy(atomSB, gSB, sSB)
        for j in range_constexpr(num_buffers - 1):
            issue(j, j)
        n_steady = K_TILES - (num_buffers - 1)
        for kt in range(n_steady):
            tdm_ops.tensor_wait((num_buffers - 2) * n_ops)
            fx.barrier()
            nxt = kt + (num_buffers - 1)
            issue(nxt % num_buffers, nxt)
            compute(kt % num_buffers, kt)
        for j in range_constexpr(num_buffers - 1):
            tdm_ops.tensor_wait((num_buffers - 2 - j) * n_ops)
            fx.barrier()
            compute((n_steady + j) % num_buffers, n_steady + j)

        if const_expr(dstore):
            # moegemm: C straight from the accumulators (16 B per lane per fragment row-half), no LDS staging, no
            # barrier, no wait for the stores before the workgroup ends. Whole tiles only (group offsets multiples
            # of tile_m, checked by the wrapper's aligned flag); dead tiles (rows 0) store nothing.
            if rows > zero:
                _store_c_lines(base_ptr, SCR_OFF + wave * SCR_W, lane, c_frags, m_reps, n_reps, warp_n,
                               _cast(fx.Int8, gC_base), m_row + wmb, blk_n + wnb, ldc64, out_cls, lds_b128)
        else:
            fx.barrier()
            sC = _view(base_ptr, out_cls, (tile_m, C_ROW), (C_ROW, 1))
            for mi in range_constexpr(m_reps):
                row = wmb + mi * WMMA_M + lane16
                for ni in range_constexpr(n_reps):
                    col = wnb + ni * WMMA_N + kgrp * (WMMA_N // 2)
                    h = c_frags[mi * n_reps + ni].load().to(out_cls)
                    fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row * C_ROW + col) * 2)
            fx.barrier()
            gtC = _gview(gC_base, fx.Int64(m_row) * ldc64 + fx.Int64(blk_n), (tile_m, C_ROW), (C_ROW, 1))
            atomC = fx.rocdl.make_tdm_atom(gtC, [rows, tile_n], strides=[ldc64, None], num_warps=num_waves)
            fx.copy(atomC, sC, gtC)
            tdm_ops.tensor_wait(0)

    kernel_gg(arg_c, arg_a, arg_b, arg_offs, arg_sa, arg_sb, i32_mtiles, i32_n, i32_k).launch(
        grid=(i32_mtiles * ((i32_n + (tile_n - 1)) // tile_n), 1, 1), block=(block, 1, 1), stream=stream
    )


launch_gg_fp8blk_nt_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}


@flyc.jit
def launch_ggq_fp8blk_nt_gfx1250(
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
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    a_id: Constexpr[int],
    b_id: Constexpr[int],
    k_max: Constexpr[int] = 4096,
):
    """Persistent flygg (moegemm track): same operands, math and per-tile k order as kernel_gg (bit-identical).
    * grid = #CUs; workgroup bid walks the live output tiles t = bid, bid + nwg, ... (live = offs[G] / tile_m m-tiles,
      read on the device; group offsets must be multiples of tile_m, so every tile is whole);
    * one TDM stream per workgroup through a 3-stage ring, 2 k-tiles in flight; the next tile's scale planes and
      first two k-tiles are loaded during the current tile's last two k-steps (no per-tile prologue bubble);
    * whole-K scale planes double-buffered by tile parity; the B plane holds one row per 16-row WMMA fragment
      (row 16*i of the tile: the weight scale is per 128x128 block, so the rows of a fragment share it);
    * C stored straight from the accumulators (global_store_b128), not waited on: it overlaps the next tile.
    Wait accounting (2 TDM ops per k-tile, 2 per plane pair issued just before a tile's k-tile 0): newer than
    k-tile kt when it is consumed = k-tile kt+1, plus the next tile's planes at the last k-step -> wait 2 / 4."""
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    tile_k = WMMA_K
    NB = 3
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    assert tile_m % (m_warp * WMMA_M) == 0 and tile_n % (n_warp * WMMA_N) == 0
    assert tile_m % num_waves == 0 and tile_n % 128 == 0
    out_cls = fx.BFloat16
    LDS_PAD = 16
    A_ROW = tile_k + LDS_PAD
    B_ROW = tile_k + LDS_PAD
    STAGE_A = tile_m * A_ROW
    STAGE_B = tile_n * B_ROW
    PITCH = STAGE_A + STAGE_B
    MX_BLOCK = 32
    SK = k_max // MX_BLOCK
    SP = SK + 4
    BROWS = tile_n // WMMA_N  # one B-scale row per 16-row fragment
    assert BROWS % num_waves == 0
    PLANE_A = tile_m * SP
    PSET = (PLANE_A + BROWS * SP + 15) // 16 * 16
    SC_OFF = NB * PITCH
    SCR_OFF = SC_OFF + 2 * PSET  # wave-private C line scratch
    SCR_W = 16 * (2 * (tile_n // n_warp) + 16)
    ARENA = SCR_OFF + num_waves * SCR_W
    assert ARENA <= 320 * 1024, f"LDS arena {ARENA} B exceeds 320 KiB"
    G = n_groups
    kernel_name = f"moe_ggq_fp8blk_nt_gfx1250_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}_a{a_id}{b_id}_g{G}_k{k_max}"

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_ggq(
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
        i32_ptr = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        offs = fx.recast_iter(i32_ptr, arg_offs)
        zero = bid * fx.Int32(0)
        one = zero + fx.Int32(1)
        live_rows = rocdl.readfirstlane(_T.i32, offs[zero + fx.Int32(G)])
        total = (live_rows // tile_m) * n_tiles_n
        my_cnt = (total - bid + i32_nwg - 1) // i32_nwg
        my_cnt = (my_cnt > zero).select(my_cnt, zero)
        last_t = bid + (my_cnt - 1) * i32_nwg
        KT = i32_k // tile_k
        lda64 = fx.Int64(i32_k)
        ldc64 = fx.Int64(i32_n)
        sk = i32_k // MX_BLOCK
        sk64 = fx.Int64(sk)

        def tile_params(t):
            """tile t -> (expert, first pool row, first n column); group offsets are tile_m aligned"""
            mt = t // n_tiles_n
            m_row = mt * tile_m
            g = zero
            for i in range_constexpr(G):
                e = offs[zero + fx.Int32(i + 1)]
                g = g + (e <= m_row).select(one, zero)
            g = (g < fx.Int32(G)).select(g, fx.Int32(G - 1))
            return (rocdl.readfirstlane(_T.i32, g), rocdl.readfirstlane(_T.i32, m_row),
                    rocdl.readfirstlane(_T.i32, (t % n_tiles_n) * tile_n))

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
        atomA = fx.rocdl.make_tdm_atom(gA, [None, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)
        atomB = fx.rocdl.make_tdm_atom(gB, [None, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)
        gSA_base = _cast(fx.Uint8, arg_sa, 1)
        gSB_base = _cast(fx.Uint8, arg_sb, 1)

        def sA_of(s):
            return _view(_stage(s), a_cls, (tile_m, tile_k), (A_ROW, 1))

        def sB_of(s):
            return _view(fx.add_offset(_stage(s), STAGE_A), b_cls, (tile_n, tile_k), (B_ROW, 1))

        def offs_of(g, m_row, blk_n, kt):
            k = fx.Int64(kt) * tile_k
            return fx.Int64(m_row) * lda64 + k, (fx.Int64(g) * fx.Int64(i32_n) + fx.Int64(blk_n)) * lda64 + k

        def issue(s, oa, ob):
            fx.copy(atomA, gA, sA_of(s), imm_offset=oa)
            fx.copy(atomB, gB, sB_of(s), imm_offset=ob)

        def issue_planes(pset, g, m_row, blk_n):
            gsa = _gview(gSA_base, fx.Int64(m_row) * sk64, (tile_m, SK), (sk, 1))
            gsb = _gview(gSB_base, (fx.Int64(g) * fx.Int64(i32_n) + fx.Int64(blk_n)) * sk64, (BROWS, SK),
                         (sk * WMMA_N, 1))
            asa = fx.rocdl.make_tdm_atom(gsa, [None, sk], strides=[sk64, None], num_warps=num_waves,
                                         pad_interval=SK, pad_amount=4)
            asb = fx.rocdl.make_tdm_atom(gsb, [None, sk], strides=[sk64 * WMMA_N, None], num_warps=num_waves,
                                         pad_interval=SK, pad_amount=4)
            poff = SC_OFF + pset * PSET
            fx.copy(asa, gsa, _view(fx.add_offset(base_ptr, poff), fx.Uint8, (tile_m, SK), (SP, 1), 1))
            fx.copy(asb, gsb, _view(fx.add_offset(base_ptr, poff + PLANE_A), fx.Uint8, (BROWS, SK), (SP, 1), 1))

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
        lds_base = fx.Int32(fx.ptrtoint(base_ptr))

        def _operand(base, plane_off, row, row_bytes):
            b0 = plane_off + row * row_bytes + kgrp * 16
            v = [fx.Vector(lds_b128(base, b0 + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            reg = fx.make_rmem_tensor(16, fx.Int32)
            reg.store(v01.shuffle(v23, list(range(16))))
            return reg

        def compute(s, pset, kt):
            base = fx.Int32(fx.ptrtoint(_stage(s)))
            poff = SC_OFF + pset * PSET
            kc = kt * 4
            bf = [_operand(base, STAGE_A, wnb + ni * WMMA_N + lane16, B_ROW) for ni in range_constexpr(n_reps)]
            sbv = [lds_b32(lds_base, poff + PLANE_A + (wnb // WMMA_N + ni) * SP + kc)[0]
                   for ni in range_constexpr(n_reps)]
            for mi in range_constexpr(m_reps):
                af = _operand(base, 0, wmb + mi * WMMA_M + lane16, A_ROW)
                sav = lds_b32(lds_base, poff + (wmb + mi * WMMA_M + lane16) * SP + kc)[0]
                for ni in range_constexpr(n_reps):
                    idx = mi * n_reps + ni
                    fx.gemm(mma_atom, c_frags[idx], bf[ni], af, c_frags[idx], scale_a=sbv[ni], scale_b=sav)

        gC8 = _cast(fx.Int8, _cast(out_cls, arg_c))
        t0 = (my_cnt > zero).select(bid, zero)
        g0, mr0, bn0 = tile_params(t0)
        issue_planes(0, g0, mr0, bn0)
        for j0 in range_constexpr(NB - 1):
            oa, ob = offs_of(g0, mr0, bn0, j0)
            issue(j0, oa, ob)
        for it in range(my_cnt):
            t = bid + it * i32_nwg
            g, m_row, blk_n = tile_params(t)
            t_nx = (t + i32_nwg < total).select(t + i32_nwg, last_t)
            ng, nm_row, nblk_n = tile_params(t_nx)
            pset = it % 2
            for cf in c_frags:
                cf.fill(0.0)
            gk = gk_reg.load()[0]
            for kt in range(KT):
                if kt == KT - 1:
                    tdm_ops.tensor_wait(4)
                else:
                    tdm_ops.tensor_wait(2)
                fx.barrier()
                f = kt + 2
                if f < KT:
                    oa0, ob0 = offs_of(g, m_row, blk_n, f)
                    issue((gk + f) % NB, oa0, ob0)
                else:
                    if f == KT:
                        issue_planes(1 - pset, ng, nm_row, nblk_n)
                        oa1, ob1 = offs_of(ng, nm_row, nblk_n, 0)
                        issue((gk + f) % NB, oa1, ob1)
                    else:
                        oa2, ob2 = offs_of(ng, nm_row, nblk_n, 1)
                        issue((gk + f) % NB, oa2, ob2)
                compute((gk + kt) % NB, pset, kt)
            gk_reg.store(gk_reg.load() + KT)
            _store_c_lines(base_ptr, SCR_OFF + wave * SCR_W, lane, c_frags, m_reps, n_reps, warp_n, gC8,
                           m_row + wmb, blk_n + wnb, ldc64, out_cls, lds_b128)
        tdm_ops.tensor_wait(0)

    kernel_ggq(arg_c, arg_a, arg_b, arg_offs, arg_sa, arg_sb, i32_nwg, i32_n, i32_k).launch(
        grid=(i32_nwg, 1, 1), block=(block, 1, 1), stream=stream
    )


launch_ggq_fp8blk_nt_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}


@flyc.jit
def launch_ggk_fp8blk_nt_gfx1250(
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
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    a_id: Constexpr[int],
    b_id: Constexpr[int],
):
    """Persistent flygg with k256 k-tiles (moegemm2). Same operands, WMMAs, per-tile k order and bf16 rounding as
    kernel_gg (bit-identical); removes the per-tile prologue / epilogue that ATT showed in the one-tile-per-workgroup
    kernel (first k-tile wait ~1.4 us + C store wait ~1 us per tile, ~6 tiles per CU):
    * grid = #CUs, workgroup bid walks the live output tiles t = bid, bid + nwg, ... (live = offs[G] / tile_m m-tiles;
      group offsets are tile_m multiples, read once into SGPRs by scalar buffer loads);
    * one TDM stream per workgroup through a 2-stage ring of tile_k-deep k-tiles (2 TDM ops per k-tile, wait 0 per
      step as in kernel_gg); the next tile's k-tile 0 is issued at the current tile's last k-step;
    * no scale planes in LDS: each k-step's E8M0 words (4x-repeated bytes = the WMMA scale operand) come straight
      from global memory with vector loads issued one k-step ahead (carried in registers, also across tiles);
    * C leaves through the wave-private LDS line scratch with whole-line global stores (_store_c_lines), which are not
      tensorcnt ops, so the next tile's TDM waits never wait for them.
    Needs aligned group offsets, K % tile_k == 0, N % tile_n == 0, sb constant over 16-row groups of N (128x128-block
    weight scales) -- the B word of fragment row 16*i stands for its 16 rows."""
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    MX_BLOCK = 32
    NB = 2
    k_iters = tile_k // WMMA_K
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    assert tile_m % (m_warp * WMMA_M) == 0 and tile_n % (n_warp * WMMA_N) == 0
    assert tile_k % WMMA_K == 0 and tile_m % num_waves == 0 and tile_n % num_waves == 0
    out_cls = fx.BFloat16
    LDS_PAD = 16
    A_ROW = tile_k + LDS_PAD
    B_ROW = tile_k + LDS_PAD
    STAGE_A = tile_m * A_ROW
    STAGE_B = tile_n * B_ROW
    PITCH = STAGE_A + STAGE_B
    SCR_OFF = NB * PITCH
    SCR_W = 16 * (2 * (tile_n // n_warp) + 16)
    ARENA = SCR_OFF + num_waves * SCR_W
    assert ARENA <= 320 * 1024, f"LDS arena {ARENA} B exceeds 320 KiB"
    G = n_groups
    kernel_name = (f"moe_ggk_fp8blk_nt_gfx1250_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}_a{a_id}{b_id}_g{G}")

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_ggk(
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
        i32_ptr = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        offs_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(fx.recast_iter(i32_ptr, arg_offs)))
        sa_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(fx.recast_iter(i32_ptr, arg_sa)))
        sb_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(fx.recast_iter(i32_ptr, arg_sb)))
        zero = bid * fx.Int32(0)
        one = zero + fx.Int32(1)
        offv = [fx.Int32(_bops.buffer_load(offs_rsrc, zero + fx.Int32(i), vec_width=1, is_scalar=True))
                for i in range_constexpr(G + 1)]
        live_rows = offv[G]
        total = (live_rows // tile_m) * n_tiles_n
        my_cnt = (total - bid + i32_nwg - 1) // i32_nwg
        my_cnt = (my_cnt > zero).select(my_cnt, zero)
        last_t = bid + (my_cnt - 1) * i32_nwg
        KT = i32_k // tile_k
        lda64 = fx.Int64(i32_k)
        ldc64 = fx.Int64(i32_n)
        sk = i32_k // MX_BLOCK  # scale bytes per row (4x-repeated E8M0 per 128-block)
        skw = sk // 4  # scale words per row

        def tile_params(t):
            """tile t -> (expert, first pool row, first n column); group offsets are tile_m aligned"""
            m_row = (t // n_tiles_n) * tile_m
            g = zero
            for i in range_constexpr(G):
                g = g + (offv[i + 1] <= m_row).select(one, zero)
            g = (g < fx.Int32(G)).select(g, fx.Int32(G - 1))
            return g, m_row, (t % n_tiles_n) * tile_n

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
        atomA = fx.rocdl.make_tdm_atom(gA, [None, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)
        atomB = fx.rocdl.make_tdm_atom(gB, [None, None], strides=[lda64, None], num_warps=num_waves,
                                       pad_interval=tile_k, pad_amount=LDS_PAD)

        def sA_of(s):
            return _view(_stage(s), a_cls, (tile_m, tile_k), (A_ROW, 1))

        def sB_of(s):
            return _view(fx.add_offset(_stage(s), STAGE_A), b_cls, (tile_n, tile_k), (B_ROW, 1))

        def issue(s, g, m_row, blk_n, kt):
            k = fx.Int64(kt) * tile_k
            fx.copy(atomA, gA, sA_of(s), imm_offset=fx.Int64(m_row) * lda64 + k)
            fx.copy(atomB, gB, sB_of(s), imm_offset=(fx.Int64(g) * fx.Int64(i32_n) + fx.Int64(blk_n)) * lda64 + k)

        mma_atom = fx.make_mma_atom(fx.rocdl.WMMAScale(WMMA_M, WMMA_N, WMMA_K, b_cls, a_cls, fx.Float32,
                                                       block_size=MX_BLOCK))
        warp_m = tile_m // m_warp
        warp_n = tile_n // n_warp
        m_reps = warp_m // WMMA_M
        n_reps = warp_n // WMMA_N
        lds_b128 = _make_lds_load(128)
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
        # scale words of the next k-step (k_iters words per fragment row), carried across steps and tiles
        sa_regs = [fx.make_rmem_tensor(k_iters, fx.Int32) for _ in range_constexpr(m_reps)]
        sb_regs = [fx.make_rmem_tensor(k_iters, fx.Int32) for _ in range_constexpr(n_reps)]

        def load_scales(g, m_row, blk_n, kt):
            kw0 = kt * k_iters
            for mi in range_constexpr(m_reps):
                sa_regs[mi].store(fx.Vector(_bops.buffer_load(
                    sa_rsrc, (m_row + wmb + mi * WMMA_M + lane16) * skw + kw0, vec_width=k_iters, dtype=fx.Int32)))
            for ni in range_constexpr(n_reps):
                sb_regs[ni].store(fx.Vector(_bops.buffer_load(
                    sb_rsrc, (g * i32_n + blk_n + wnb + ni * WMMA_N) * skw + kw0, vec_width=k_iters, dtype=fx.Int32)))

        def _operand(base, plane_off, row, row_bytes, ks):
            b0 = plane_off + row * row_bytes + ks * WMMA_K + kgrp * 16
            v = [fx.Vector(lds_b128(base, b0 + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            reg = fx.make_rmem_tensor(16, fx.Int32)
            reg.store(v01.shuffle(v23, list(range(16))))
            return reg

        def compute(s, sav_w, sbv_w):
            base = fx.Int32(fx.ptrtoint(_stage(s)))
            for ks in range_constexpr(k_iters):
                bf = [_operand(base, STAGE_A, wnb + ni * WMMA_N + lane16, B_ROW, ks) for ni in range_constexpr(n_reps)]
                for mi in range_constexpr(m_reps):
                    af = _operand(base, 0, wmb + mi * WMMA_M + lane16, A_ROW, ks)
                    for ni in range_constexpr(n_reps):
                        idx = mi * n_reps + ni
                        fx.gemm(mma_atom, c_frags[idx], bf[ni], af, c_frags[idx], scale_a=sbv_w[ni][ks],
                                scale_b=sav_w[mi][ks])

        gC8 = _cast(fx.Int8, _cast(out_cls, arg_c))
        t0 = (my_cnt > zero).select(bid, zero)
        g0, mr0, bn0 = tile_params(t0)
        issue(0, g0, mr0, bn0, zero)
        load_scales(g0, mr0, bn0, zero)
        for it in range(my_cnt):
            t = bid + it * i32_nwg
            g, m_row, blk_n = tile_params(t)
            t_nx = (t + i32_nwg < total).select(t + i32_nwg, last_t)
            ng, nm_row, nblk_n = tile_params(t_nx)
            for cf in c_frags:
                cf.fill(0.0)
            gk = gk_reg.load()[0]
            for kt in range(KT):
                tdm_ops.tensor_wait(0)
                fx.barrier()
                # next k-step: k-tile kt+1 of this tile, or k-tile 0 of the next one (branch-free)
                nxt_last = kt + 1 >= KT
                sg = nxt_last.select(ng, g)
                sm = nxt_last.select(nm_row, m_row)
                sn = nxt_last.select(nblk_n, blk_n)
                skt = nxt_last.select(zero, kt + 1)
                issue((gk + kt + 1) % NB, sg, sm, sn, skt)
                sav_w = [r.load() for r in sa_regs]
                sbv_w = [r.load() for r in sb_regs]
                load_scales(sg, sm, sn, skt)
                compute((gk + kt) % NB, sav_w, sbv_w)
            gk_reg.store(gk_reg.load() + KT)
            _store_c_lines(base_ptr, SCR_OFF + wave * SCR_W, lane, c_frags, m_reps, n_reps, warp_n, gC8,
                           m_row + wmb, blk_n + wnb, ldc64, out_cls, lds_b128)
        tdm_ops.tensor_wait(0)

    kernel_ggk(arg_c, arg_a, arg_b, arg_offs, arg_sa, arg_sb, i32_nwg, i32_n, i32_k).launch(
        grid=(i32_nwg, 1, 1), block=(block, 1, 1), stream=stream
    )


launch_ggk_fp8blk_nt_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}

_COMPILED: dict = {}


def _ptr(t):
    return flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)


def flygg_fp8blk_nt(a, b, sa, sb, offs_i32, out=None, tile=(256, 256, 128), m_warp=2, n_warp=2, num_buffers=3,
                    aligned=False, dstore=0, persistent=0, lds_pad=16, bblk=0, xdl_nostall=0, bpf=0, hwsched=0):
    """a [M, K] fp8, b [G, N, K] fp8 contiguous, sa [M, K/32] / sb [G, N, K/32] uint8 E8M0 (block-128 exponents
    repeated 4x), offs_i32 [G+1] int32. Returns out [M, N] bf16 (rows past offs[G] are not written).
    aligned: the caller guarantees every offs_i32 entry is a multiple of tile[0] (not checked: that would sync), so
    the live m-tiles are <= M // tile[0] and the grid needs no +G per-group slack (fewer dead workgroups).
    dstore: epilogue stores C from the accumulators (no LDS staging / TDM store / final wait); needs aligned.
    persistent: launch_ggq (persistent, cross-tile TDM stream, direct C stores); needs aligned, K <= 4096 and
    sb constant over each 16-row group of N (true for 128x128-block weight scales).
    lds_pad: LDS row pad of the operand stages (8 lets num_buffers=4 fit with bblk). bblk: B scale plane with one
    row per 16-row fragment (needs sb constant over 16-row groups of N, true for 128x128-block weight scales).
    bpf (mxgemm): in-k-tile fragment pipeline of kernel_gg (tile_k >= 256 only; ignored otherwise).
    hwsched (mxgemm): kernel_gg built without expert scheduling mode (hardware interlocks).
    All are bit-identical to the default path."""
    assert sa.dtype == torch.uint8 and sb.dtype == torch.uint8 and sa.is_contiguous() and sb.is_contiguous()
    M, K = a.shape
    G, N, Kb = b.shape
    assert K == Kb and a.is_contiguous() and b.is_contiguous()
    tm, tn, tk = tile
    assert K % tk == 0 and N % tn == 0 and K // tk >= num_buffers - 1
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    a_id = 0 if a.dtype == torch.float8_e4m3fn else 1
    b_id = 0 if b.dtype == torch.float8_e4m3fn else 1
    assert not dstore or aligned, "dstore needs whole tiles (aligned group offsets)"
    if persistent == 2:
        # moegemm2: persistent k256 (launch_ggk); scales read straight from sa / sb, C through _store_c_lines
        assert aligned and tk % 128 == 0 and K % tk == 0 and N % tn == 0
        assert sa.shape == (M, K // 32) and sb.shape == (G, N, K // 32)
        nwg = torch.cuda.get_device_properties(a.device).multi_processor_count
        nwg = max(1, min(nwg, (M // tm) * (N // tn)))
        args = (_ptr(out), _ptr(a), _ptr(b), _ptr(offs_i32), _ptr(sa), _ptr(sb), nwg,
                fx.Stream(torch.cuda.current_stream(device=a.device)), N, K, G, tm, tn, tk, m_warp, n_warp, a_id, b_id)
        key = ("k", G, tm, tn, tk, m_warp, n_warp, a_id, b_id, nwg)
        c = _COMPILED.get(key)
        if c is None:
            _COMPILED[key] = flyc.compile(launch_ggk_fp8blk_nt_gfx1250, *args)
        else:
            c(*args)
        return out
    if persistent:
        assert aligned and tk == 128 and K // tk >= 2 and K <= 4096 and tn % 128 == 0
        nwg = torch.cuda.get_device_properties(a.device).multi_processor_count
        nwg = max(1, min(nwg, (M // tm) * (N // tn)))
        args = (_ptr(out), _ptr(a), _ptr(b), _ptr(offs_i32), _ptr(sa), _ptr(sb), nwg,
                fx.Stream(torch.cuda.current_stream(device=a.device)), N, K, G, tm, tn, m_warp, n_warp, a_id, b_id,
                4096)
        key = ("q", G, tm, tn, m_warp, n_warp, a_id, b_id, nwg)
        c = _COMPILED.get(key)
        if c is None:
            _COMPILED[key] = flyc.compile(launch_ggq_fp8blk_nt_gfx1250, *args)
        else:
            c(*args)
        return out
    mtiles = M // tm if aligned else M // tm + G
    args = (_ptr(out), _ptr(a), _ptr(b), _ptr(offs_i32), _ptr(sa), _ptr(sb), mtiles,
            fx.Stream(torch.cuda.current_stream(device=a.device)), N, K, G, tm, tn, tk, m_warp, n_warp,
            num_buffers, a_id, b_id, K, int(dstore), lds_pad, int(bblk), int(xdl_nostall), int(bpf and tk >= 256),
            int(hwsched))
    key = (G, tm, tn, tk, m_warp, n_warp, num_buffers, a_id, b_id, K, int(dstore), lds_pad, int(bblk), int(xdl_nostall),
           int(bpf and tk >= 256), int(hwsched))
    c = _COMPILED.get(key)
    if c is None:
        _COMPILED[key] = _compile_sched(launch_gg_fp8blk_nt_gfx1250, hwsched, args)
    else:
        c(*args)
    return out


def _compile_sched(launch, hwsched, args):
    """flyc.compile with amdgpu-expert-scheduling-mode off for hwsched variants (the kernel name carries _hw)."""
    hints = launch.compile_hints
    old = hints.get("llvm_options")
    if hwsched:
        hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": False}
    try:
        return flyc.compile(launch, *args)
    finally:
        hints["llvm_options"] = old
