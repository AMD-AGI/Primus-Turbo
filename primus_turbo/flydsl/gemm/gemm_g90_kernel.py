###############################################################################
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""gfx1250 dense GEMM with a continuously prefetched LDS->VGPR main loop (g90).

Same operand staging as ``gemm_gfx1250_kernel.py`` (TDM global->LDS, padded LDS rows,
LDS-staged TDM C store), new main loop:

* every WMMA slot of a K-tile issues (at most) one ``ds_load_b128``; the load stream runs
  a fixed ``S`` slots ahead of use, so the fragments of K-tile k+1 load while K-tile k
  computes (no per-K-tile exposed LDS latency, which cost ~50% in the original loop);
* the TDM wait + workgroup barrier + next TDM issue sit at the slot where the stream
  crosses into the next stage (mid K-tile);
* the K loop is unrolled by ``num_buffers`` so all LDS stage offsets are constants, the
  last unrolled iteration is peeled in Python, and ``sched_barrier`` keeps LLVM from
  re-hoisting the loads;
* 1 workgroup per CU (up to 320 KB LDS, ~1024 VGPR per wave).

NT layout: ``C[M, N] = A[M, K] @ B[N, K]^T``, all row-major.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster as _cluster
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.primitive import atom_set_value
from flydsl.expr.typing import Constexpr
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import as_ir_value
from flydsl._mlir.dialects import llvm as _llvm

WAVE = 32
# kind id -> (name, element class, bytes per element, WMMA K)
_KINDS = {
    0: ("fp8", fx.Float8E4M3FN, 1, 128),
    1: ("bf16", fx.BFloat16, 2, 32),
    # FP8 E4M3 with DeepSeek-V3/V4 block scaling: A 1x128 (per row, per 128-K block), B 128x128,
    # UE8M0 scales applied by V_WMMA_SCALE inside the K loop.
    2: ("fp8blk", fx.Float8E4M3FN, 1, 128),
    # Same recipe, plain FP8 WMMA + exact power-of-two accumulator rescale on the VALU between
    # K-tiles. V_WMMA_SCALE needs its accumulator in v0..v255 (no VGPR-MSB form), which caps a
    # wave at 30 accumulator tiles; the 128x128-per-wave tile needs 64, so kind 2 spills.
    3: ("fp8blkv", fx.Float8E4M3FN, 1, 128),
}
_TORCH_KIND = {torch.float8_e4m3fn: 0, torch.bfloat16: 1}


def _lds_loader(bits):
    layout = fx.make_layout(bits // 32, 1)
    atom = fx.make_copy_atom(fx.UniversalCopy(bits), fx.Int32)
    ptr_ty = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, bits // 8)

    def load(base, byte_offset):
        ptr = fx.inttoptr(ptr_ty, fx.Int32(base) + fx.Int32(byte_offset))
        reg = fx.make_rmem_tensor(layout, fx.Int32)
        fx.copy_atom_call(atom, fx.Tensor(fx.make_view(ptr, layout)), reg)
        return reg.load()

    return load


def _opaque_v(x):
    """Return x unchanged in a VGPR, hidden from constant folding / rematerialization."""
    op = _llvm.InlineAsmOp(res=_T.i32, operands_=[as_ir_value(fx.Int32(x))], asm_string="",
                           constraints="=v,0", has_side_effects=False, is_align_stack=False)
    return fx.Int32(op.res)


def _if_then(cond, fn):
    """scf.if without else around ``fn()`` (uniform condition)."""
    from flydsl._mlir import ir as _ir
    from flydsl._mlir.dialects import scf as _scf

    if_op = _scf.IfOp(as_ir_value(cond), [], has_else=False)
    if len(if_op.regions[0].blocks) == 0:
        if_op.regions[0].blocks.append(*[])
    with _ir.InsertionPoint(if_op.regions[0].blocks[0]):
        fn()
        _scf.YieldOp([])


def make_plan(MR, NR, KI, LPF, lead):
    """Static schedule of one K-tile.

    WMMA slot w = (ks, mi, ni) in row-major order. Fragments are streamed in order of
    first use; chunk c of the K-tile's stream has unshifted slot u(c) = c*W//L and is
    issued at slot u(c) - S of the same K-tile (negative = previous K-tile).
    """
    W = KI * MR * NR
    fr = []
    for ks in range(KI):
        for ni in range(NR):
            fr.append(("B", ks, ni, ks * MR * NR + ni))
        for mi in range(MR):
            fr.append(("A", ks, mi, ks * MR * NR + mi * NR))
    fr.sort(key=lambda f: f[3])
    L = len(fr) * LPF
    assert (LPF * W) % L == 0
    step = LPF * W // L  # slots per fragment

    def u(c):
        return (c * W) // L

    S0 = max(u(i * LPF + LPF - 1) - f[3] for i, f in enumerate(fr))
    S = -(-(S0 + lead) // step) * step
    assert S < W, f"lead {lead} too large (S={S}, W={W})"
    loads = [[] for _ in range(W)]
    for i in range(len(fr)):
        for j in range(LPF):
            loads[u(i * LPF + j)].append((i, j))
    idx = {(f[0], f[1], f[2]): i for i, f in enumerate(fr)}
    return W, fr, S, loads, S // step, idx


@flyc.jit
def launch_gemm_g90(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_sa: fx.Pointer,
    arg_sb: fx.Pointer,
    i32_m: fx.Int32,
    stream: fx.Stream,
    i32_n: fx.Int32,
    i32_k: fx.Int32,
    i32_lda: fx.Int32,
    i32_ldb: fx.Int32,
    i32_ldc: fx.Int32,
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    num_buffers: Constexpr[int],
    kind_id: Constexpr[int],
    lead: Constexpr[int],
    group_m: Constexpr[int],
    out_is_f16: Constexpr[int],
    abl: Constexpr[int] = 0,
    cluster_m: Constexpr[int] = 1,
    cluster_n: Constexpr[int] = 1,
    mc_mode: Constexpr[int] = 1,
    tdm_mode: Constexpr[int] = 0,
    tk_bytes: Constexpr[int] = 128,
    b_path: Constexpr[int] = 0,
    ws_rows: Constexpr[int] = 0,
    split_k: Constexpr[int] = 1,
    b_rowscale: Constexpr[int] = 0,
    accumulate: Constexpr[int] = 0,
):
    """abl (ablation, timing only): bit0 = no TDM refill in the K loop, bit1 = no ds_load
    (WMMAs reuse the carried fragments), bit2 = no WMMA, bit3 = TDM re-reads K-tiles 0..NB-1 (L2-hot)."""
    kname, e_cls, eb, WMMA_K = _KINDS[kind_id]
    is_bf16 = eb == 2
    is_blk = kind_id in (2, 3)
    use_wscale = kind_id == 2
    # kind 3: the first ws_rows accumulator rows use V_WMMA_SCALE (their tiles must fit in
    # v0..v255), the rest plain WMMA + VALU rescale -- trades VALU work for low-bank VGPRs.
    WS_ROWS = ws_rows if kind_id == 3 else 0
    RS_AHEAD = 8  # accumulator tiles rescaled ahead of their first WMMA in the next K-tile
    WMMA_M = WMMA_N = 16
    NB = num_buffers
    U = NB
    assert NB >= 3
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    TK = tk_bytes // eb  # K-tile row bytes
    KI = TK // WMMA_K
    LPF = WMMA_K * eb // 32  # ds_load_b128 per fragment per lane
    FW = LPF * 4  # i32 words per fragment
    warp_m = tile_m // m_warp
    warp_n = tile_n // n_warp
    MR = warp_m // WMMA_M
    NR = warp_n // WMMA_N
    W, FR, S, LOADS, NCARRY, FIDX = make_plan(MR, NR, KI, LPF, lead)

    LDS_PAD = 16
    ROW_B = TK * eb + LDS_PAD  # bytes per LDS row (both operands)
    STAGE_A = tile_m * ROW_B
    STAGE_B = tile_n * ROW_B
    PITCH = STAGE_A + STAGE_B
    C_ROW = tile_n + 8
    # split-K: grid.x = tiles * split_k; split z covers K-tiles [z*KT/split_k, (z+1)*KT/split_k)
    # and writes fp32 partials to C + z * M * ldc (summed by the wrapper).
    # accumulate: C += A @ B^T -- the wgrad main_grad epilogue (beta = 1). 1: fp32 C; 2: 16-bit C (bf16, or
    # f16 with out_is_f16) read, added in fp32 and rounded once (Megatron main_grads_dtype=bf16).
    assert not (accumulate and split_k > 1)
    acc16 = accumulate == 2
    OB = 4 if (split_k > 1 or accumulate == 1) else 2
    ARENA = max(NB * PITCH, tile_m * C_ROW * OB)
    assert ARENA <= 320 * 1024, f"LDS {ARENA} > 320 KiB"
    # tdm_mode 0: every wave fetches 1/num_waves of each operand (2 TDM ops per wave);
    # tdm_mode 1: wave 0 fetches all of A, wave 1 all of B (1 TDM op per issuing wave).
    n_ops = 2 if tdm_mode in (0, 2) else 1
    # b_path 1: B rides global_load_async_to_lds_b128 (vector-memory path, no VGPR data) instead
    # of TDM, so the two operands use the two global->LDS paths in parallel.
    if b_path == 1:
        n_ops = 1
    B_ROWS_PW = tile_n // (m_warp * n_warp)  # B rows fetched per wave (async path)
    B_ASYNC_PER_TILE = B_ROWS_PW * (tk_bytes // 16) // WAVE  # b128 async ops per wave per K-tile
    out_cls = fx.Float32 if (split_k > 1 or accumulate == 1) else (fx.Float16 if out_is_f16 else fx.BFloat16)

    kernel_name = f"g90_gemm_{kname}_t{tile_m}x{tile_n}_w{m_warp}x{n_warp}_nb{NB}_l{lead}_g{group_m}" + (f"_abl{abl}" if abl else "") + (
        f"_c{cluster_m}x{cluster_n}m{mc_mode}" if cluster_m * cluster_n > 1 else "") + (
        f"_tm{tdm_mode}" if tdm_mode else "") + (f"_tk{tk_bytes}" if tk_bytes != 128 else "") + (f"_bp{b_path}" if b_path else "") + (f"_ws{ws_rows}" if ws_rows else "") + (f"_sk{split_k}" if split_k > 1 else "") + ("_brs" if b_rowscale else "") + ("_acc" if accumulate == 1 else "") + ("_acc16" if acc16 else "")
    CM, CN = cluster_m, cluster_n
    CS = CM * CN

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_sa: fx.Pointer,
        arg_sb: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldb: fx.Int32,
        i32_ldc: fx.Int32,
    ):
        K_TILES = i32_k // TK // split_k
        n_iter = K_TILES // U
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, _, _ = fx.block_idx

        n_tiles_m = (i32_m + (tile_m - 1)) // tile_m
        n_tiles_n = (i32_n + (tile_n - 1)) // tile_n
        pid = fx.Int32(bid_x)
        if const_expr(split_k > 1):
            n_tiles_all = n_tiles_m * n_tiles_n
            kz = pid // n_tiles_all
            pid = pid % n_tiles_all
            kt0 = kz * K_TILES
        else:
            kz = fx.Int32(0)
            kt0 = fx.Int32(0)
        # Clusters of CS consecutive workgroups cover a CM x CN block of tiles; the
        # grouped (L2-friendly) ordering runs over those blocks.
        if const_expr(CS > 1):
            # position inside the hardware cluster (x-inner), cluster = CS consecutive ids
            lx = fx.Int32(rocdl.cluster_workgroup_id_x())
            q = pid // CS
            ca = lx % CM
            cb = lx // CM
        else:
            q = pid
        st_m = n_tiles_m // CM
        st_n = n_tiles_n // CN
        if const_expr(group_m > 1):
            per_group = group_m * st_n
            first_m = (q // per_group) * group_m
            rem = st_m - first_m
            gsize = (rem < group_m).select(rem, fx.Int32(group_m))
            in_g = q % per_group
            pm = first_m + in_g % gsize
            pn = in_g // gsize
        else:
            pm = q % st_m
            pn = q // st_m
        if const_expr(CS > 1):
            pm = pm * CM + ca
            pn = pn * CN + cb
        blk_m = pm * tile_m
        blk_n = pn * tile_n
        m_oob = i32_m - blk_m
        n_oob = i32_n - blk_n
        lda64 = fx.Int64(i32_lda)
        ldb64 = fx.Int64(i32_ldb)
        ldc64 = fx.Int64(i32_ldc)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA)
        base_ptr = arena.base_ptr

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

        gA_base = _cast(e_cls, arg_a)
        gB_base = _cast(e_cls, arg_b)
        gC_base = _cast(out_cls, arg_c)
        if const_expr(split_k > 1):
            gC_base = fx.add_offset(gC_base, fx.Int64(kz) * fx.Int64(i32_m) * fx.Int64(i32_ldc))
        # Multicast (mc_mode 1): the CN workgroups sharing an A tile each fetch 1/CN of its
        # rows and broadcast them to all CN; likewise B over the CM sharers.
        # mc_mode 2: every sharer fetches the whole tile with the mask (relies on GL1 merge).
        split = CS > 1 and mc_mode == 1
        RA = tile_m // CN if split else tile_m
        RB = tile_n // CM if split else tile_n
        # tdm_mode 2: each wave fetches its own 1/num_waves row block of A and of B as an
        # independent whole-tile TDM op (num_warps=1); no descriptor is split across waves.
        wsplit = tdm_mode == 2
        if const_expr(wsplit):
            assert CS == 1
            RA = tile_m // num_waves
            RB = tile_n // num_waves
            wv_ = fx.Int32(rocdl.readfirstlane(_T.i32, tid // WAVE))
            ra0 = wv_ * RA
            rb0 = wv_ * RB
        elif const_expr(split):
            ra0 = cb * RA
            rb0 = ca * RB
        else:
            ra0 = fx.Int32(0)
            rb0 = fx.Int32(0)
        gA = _gview(gA_base, fx.Int64(blk_m + ra0) * lda64, (RA, TK), (i32_lda, 1))
        gB = _gview(gB_base, fx.Int64(blk_n + rb0) * ldb64, (RB, TK), (i32_ldb, 1))
        early = CS > 1 and mc_mode == 2
        atomA = fx.rocdl.make_tdm_atom(
            gA, [((m_oob - ra0) < 0).select(fx.Int32(0), m_oob - ra0), None], strides=[lda64, None], num_warps=num_waves if tdm_mode == 0 else 1,
            pad_interval=TK, pad_amount=LDS_PAD // eb, early_timeout=early,
        )
        atomB = fx.rocdl.make_tdm_atom(
            gB, [((n_oob - rb0) < 0).select(fx.Int32(0), n_oob - rb0), None], strides=[ldb64, None], num_warps=num_waves if tdm_mode == 0 else 1,
            pad_interval=TK, pad_amount=LDS_PAD // eb, early_timeout=early,
        )

        if const_expr(CS > 1):
            a_pat = 0
            for b_ in range(CN):
                a_pat |= 1 << (b_ * CM)
            a_mask = fx.Int32(a_pat) << ca
            b_mask = fx.Int32((1 << CM) - 1) << (cb * CM)
            atomA = atom_set_value(atomA, "workgroup_mask", a_mask)
            atomB = atom_set_value(atomB, "workgroup_mask", b_mask)

        def _sync():
            if const_expr(CS > 1):
                _cluster.cluster_barrier()
            else:
                fx.barrier()

        def sA_of(s):
            p_ = fx.add_offset(base_ptr, s * PITCH)
            if const_expr(split or wsplit):
                p_ = fx.add_offset(p_, ra0 * ROW_B)
            return _view(p_, e_cls, (RA, TK), (ROW_B // eb, 1))

        def sB_of(s):
            p_ = fx.add_offset(base_ptr, s * PITCH + STAGE_A)
            if const_expr(split or wsplit):
                p_ = fx.add_offset(p_, rb0 * ROW_B)
            return _view(p_, e_cls, (RB, TK), (ROW_B // eb, 1))

        wave_u = rocdl.readfirstlane(_T.i32, tid // WAVE)

        if const_expr(b_path == 1):
            assert tdm_mode == 2 and CS == 1, "async B path pairs with per-wave whole-tile A loads"
            CH_PER_ROW = TK * eb // 16
            ROWS_PER_OP = WAVE // CH_PER_ROW
            gB_u8 = _cast(fx.Uint8, arg_b, 16)
            b_row0 = fx.Int32(wave_u) * B_ROWS_PW + (tid % WAVE) // CH_PER_ROW
            b_ch = (tid % WAVE) % CH_PER_ROW
            # per-lane global byte offset of op 0 (row b_row0, chunk b_ch) and LDS byte offset
            b_goff = fx.Int64(blk_n + b_row0) * ldb64 * eb + fx.Int64(b_ch * 16)
            b_loff = STAGE_A + b_row0 * ROW_B + b_ch * 16

        def issue_b_async(s, kt64):
            for i in range_constexpr(B_ASYNC_PER_TILE):
                goff = b_goff + fx.Int64(i * ROWS_PER_OP) * ldb64 * eb + kt64 * (TK * eb)
                gptr = fx.add_offset(gB_u8, goff).llvm_ptr
                lptr = fx.recast_iter(fx.Int8, fx.add_offset(base_ptr, s * PITCH + b_loff + i * ROWS_PER_OP * ROW_B)).llvm_ptr
                rocdl.global_load_async_to_lds_b128(gptr, lptr, 0)

        def issue(s, kt):
            kt64 = fx.Int64(kt + kt0) if const_expr(not (abl & 8)) else fx.Int64(s)
            if const_expr(b_path == 1):
                fx.copy(atomA, gA, sA_of(s), imm_offset=kt64 * (TK * eb))
                issue_b_async(s, kt64)
            elif const_expr(tdm_mode in (0, 2)):
                fx.copy(atomA, gA, sA_of(s), imm_offset=kt64 * (TK * eb))
                fx.copy(atomB, gB, sB_of(s), imm_offset=kt64 * (TK * eb))
            else:
                _if_then(fx.Int32(wave_u) == 0, lambda: fx.copy(atomA, gA, sA_of(s), imm_offset=kt64 * (TK * eb)))
                _if_then(fx.Int32(wave_u) == 1, lambda: fx.copy(atomB, gB, sB_of(s), imm_offset=kt64 * (TK * eb)))

        lds_b128 = _lds_loader(128)
        wave = rocdl.readfirstlane(_T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wmb = (wave // n_warp) * warp_m
        wnb = (wave % n_warp) * warp_n
        base_i = fx.Int32(fx.ptrtoint(base_ptr))
        a_lane = base_i + (wmb + lane16) * ROW_B + kgrp * 16
        b_lane = base_i + STAGE_A + (wnb + lane16) * ROW_B + kgrp * 16
        # One VGPR base per (stage, operand): every ds_load then uses a 16-bit immediate
        # offset. Without this the stage offset (> 64 KB) costs a v_add per load, and the
        # v_add -> ds_load dependency makes the hardware wait on the in-flight WMMA.
        a_base = [_opaque_v(a_lane + s_ * PITCH) for s_ in range_constexpr(NB)]
        b_base = [_opaque_v(b_lane + s_ * PITCH) for s_ in range_constexpr(NB)]

        acc_ty = fx.Vector.make_type(8, fx.Float32)
        if const_expr(is_bf16):
            op_ty = fx.Vector.make_type(16, fx.BFloat16)

        def wmma(bv, av, cv, sb_=None, sa_=None):
            if const_expr(use_wscale or sa_ is not None):
                # slot a = B fragment (N side) -> B scale; slot b = A fragment -> A scale
                r = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                    acc_ty, bv, av, cv, sb_, sa_, fmtA=0, fmtB=0, fmtScaleA=0, fmtScaleB=0
                )
                return fx.Vector(r)
            if const_expr(is_bf16):
                bb = fx.Vector(bv).bitcast(fx.BFloat16)
                aa = fx.Vector(av).bitcast(fx.BFloat16)
                r = rocdl.wmma_f32_16x16x32_bf16_(acc_ty, as_ir_value(bb), as_ir_value(aa), as_ir_value(cv))
            else:
                r = rocdl.wmma_f32_16x16x128_fp8_fp8(acc_ty, bv, av, cv)
            return fx.Vector(r)

        def load_chunk(stage, i, j):
            kind, ks, idx = FR[i][0], FR[i][1], FR[i][2]
            off = idx * 16 * ROW_B + ks * WMMA_K * eb + 32 * j
            return fx.Vector(lds_b128(a_base[stage] if kind == "A" else b_base[stage], off))

        def combine(ch):
            if const_expr(LPF == 4):
                v01 = ch[0].shuffle(ch[1], list(range(8)))
                v23 = ch[2].shuffle(ch[3], list(range(8)))
                return v01.shuffle(v23, list(range(16)))
            return ch[0].shuffle(ch[1], list(range(8)))

        # ---- block scales (fp8blk): A [M, K/128] and B [N/128, K/128] E8M0 bytes, row-major.
        # One b32 load fetches the 4 K-tiles of one unrolled loop iteration (U = 4).
        if const_expr(is_blk):
            assert U == 4, "fp8blk packs 4 K-tile scales per word: num_buffers must be 4"
            KS = i32_k // 128  # scale bytes per row
            sa_tile = fx.add_offset(_cast(fx.Uint8, arg_sa, 4), fx.Int64(blk_m) * fx.Int64(KS))
            sa_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(sa_tile))
            sa_voff = [_opaque_v((wmb + mi * 16 + lane16) * KS) for mi in range_constexpr(MR)]
            sb_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(_cast(fx.Uint8, arg_sb, 4)))
            # b_rowscale: B scales are 1x128 per row like A ([N, K/128]; wgrad: both operands
            # quantized along the token/reduction dim). Kind 2 (V_WMMA_SCALE) only.
            assert not b_rowscale or use_wscale, "b_rowscale needs the V_WMMA_SCALE kind"
            if const_expr(b_rowscale):
                sb_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(
                    fx.add_offset(_cast(fx.Uint8, arg_sb, 4), fx.Int64(blk_n) * fx.Int64(KS))))
                sb_voffs = [_opaque_v((wnb + ni * 16 + lane16) * KS) for ni in range_constexpr(NR)]
            sb_row = ((blk_n + wnb) // 128) * KS
            sb_voff = _opaque_v(lane * 0)

            def load_scales(kt0):
                sa_w = [
                    fx.Int32(rocdl.raw_ptr_buffer_load(_T.i32, sa_rsrc, sa_voff[mi], fx.Int32(kt0), 0))
                    for mi in range_constexpr(MR)
                ]
                if const_expr(b_rowscale):
                    sb_w = [
                        fx.Int32(rocdl.raw_ptr_buffer_load(_T.i32, sb_rsrc, sb_voffs[ni], fx.Int32(kt0), 0))
                        for ni in range_constexpr(NR)
                    ]
                else:
                    sb_w = fx.Int32(rocdl.raw_ptr_buffer_load(_T.i32, sb_rsrc, sb_voff, sb_row + fx.Int32(kt0), 0))
                return sa_w, sb_w

            def rep_byte(word, r):
                # E8M0 byte r of the packed word, replicated into all 4 MX sub-block slots
                return fx.Int32(
                    rocdl.perm_b32(as_ir_value(word), as_ir_value(word), as_ir_value(fx.Int32(r * 0x01010101)))
                )

            def byte_of(word, r):
                return (fx.Int32(word) >> (8 * r)) & 0xFF

            def factor(a_from, a_to, b_from, b_to):
                # 2^((a_from - a_to) + (b_from - b_to)) as f32 bits, splatted to an acc tile
                # f32 bits of the factor; splatted at the use so only 1 VGPR per factor stays live
                return ((a_from - a_to) + (b_from - b_to) + 127) << 23

            def splat(bits):
                return fx.full(8, bits, fx.Int32).bitcast(fx.Float32)

            NSB = NR if b_rowscale else 1
            sc_carry = [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(MR + NSB)]
            sa0, sb0 = load_scales(kt0)
            for mi in range_constexpr(MR):
                sc_carry[mi].store(fx.full(1, sa0[mi], fx.Int32))
            for j_ in range_constexpr(NSB):
                sc_carry[MR + j_].store(fx.full(1, sb0[j_] if b_rowscale else sb0, fx.Int32))
            # "previous K-tile" bytes, so the first transition factor is exactly 1
            if const_expr(not use_wscale):
                pv_carry = [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(MR + 1)]
                for mi in range_constexpr(MR):
                    pv_carry[mi].store(fx.full(1, byte_of(sa0[mi], 0), fx.Int32))
                pv_carry[MR].store(fx.full(1, byte_of(sb0, 0), fx.Int32))

        c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(MR * NR)]
        for cf in c_frags:
            cf.fill(0.0)
        carry = [fx.make_rmem_tensor(FW, fx.Int32) for _ in range_constexpr(NCARRY)]

        # ---- prologue: TDM tiles 0..NB-2, then the first S slots of K-tile 0's stream ----
        for j in range_constexpr(NB - 1):
            issue(j, j)
        tdm_ops.tensor_wait((NB - 2) * n_ops)
        if const_expr(b_path == 1):
            rocdl.s_wait_asynccnt((NB - 2) * B_ASYNC_PER_TILE)
        _sync()
        pro = {}
        for t in range_constexpr(S):
            for (i, j) in LOADS[t]:
                pro.setdefault(i, []).append(load_chunk(0, i, j))
        for i in range_constexpr(NCARRY):
            carry[i].store(combine(pro[i]))

        def body(kt_base, final):
            fv = {}
            for i in range_constexpr(NCARRY):
                fv[(0, i)] = carry[i].load()
            acc = [cf.load() for cf in c_frags]
            chunks = {}
            if const_expr(is_blk):
                sa_cur = [sc_carry[mi].load()[0] for mi in range_constexpr(MR)]
                if const_expr(b_rowscale):
                    sb_cur = [sc_carry[MR + j_].load()[0] for j_ in range_constexpr(NR)]
                else:
                    sb_cur = sc_carry[MR].load()[0]
                if const_expr(not final):
                    sa_nxt, sb_nxt = load_scales(kt0 + kt_base + U)
                srep = {}
                if const_expr(not use_wscale):
                    pa_prev = [pv_carry[mi].load()[0] for mi in range_constexpr(MR)]
                    pb_prev = pv_carry[MR].load()[0]

                    def a_byte(r, mi):
                        # A scale byte of K-tile r of this iteration (r == U: next iteration's first)
                        if const_expr(r < 0):
                            return pa_prev[mi]
                        if const_expr(r < U):
                            return byte_of(sa_cur[mi], r)
                        return byte_of(sa_nxt[mi], 0)

                    def b_byte(r):
                        if const_expr(r < 0):
                            return pb_prev
                        if const_expr(r < U):
                            return byte_of(sb_cur, r)
                        return byte_of(sb_nxt, 0)

                    def fac(r_to, mi):
                        key = ("f", r_to, mi)
                        if const_expr(key not in srep):
                            srep[key] = factor(a_byte(r_to - 1, mi), a_byte(r_to, mi), b_byte(r_to - 1), b_byte(r_to))
                        return srep[key]

            rocdl.sched_barrier(0)
            for r in range_constexpr(U):
                for w in range_constexpr(W):
                    t = w + S
                    rr, tt = (r, t) if t < W else (r + 1, t - W)
                    if const_expr(t == W and not (final and r == U - 1)):
                        if const_expr(final):
                            nw = max(0, min(r + NB - 2, U - 1) - (r + 1))
                        else:
                            nw = NB - 3
                        tdm_ops.tensor_wait(nw * n_ops)
                        if const_expr(b_path == 1):
                            rocdl.s_wait_asynccnt(nw * B_ASYNC_PER_TILE)
                        _sync()
                        if const_expr(((not final) or (r + NB - 1 < U)) and not (abl & 1)):
                            issue((r + NB - 1) % NB, kt_base + (r + NB - 1))
                    if const_expr(not (final and rr >= U) and not (abl & 2)):
                        for (i, j) in LOADS[tt]:
                            chunks.setdefault((rr, i), []).append(load_chunk(rr % NB, i, j))
                            if const_expr(j == LPF - 1):
                                fv[(rr, i)] = combine(chunks[(rr, i)])
                    ks = w // (MR * NR)
                    mi = (w % (MR * NR)) // NR
                    ni = w % NR
                    ai = FIDX[("A", ks, mi)]
                    bi = FIDX[("B", ks, ni)]
                    if const_expr(abl & 2):
                        fb_, fa_ = fv[(0, bi % NCARRY)], fv[(0, ai % NCARRY)]
                    else:
                        fb_, fa_ = fv[(r, bi)], fv[(r, ai)]
                    if const_expr(is_blk and not use_wscale):
                        # exact power-of-two rescale of one accumulator tile per slot:
                        # acc holds sum / s_k during K-tile k; moving to k+1 multiplies by s_k / s_{k+1}.
                        if const_expr(w < W - RS_AHEAD):
                            q = w + RS_AHEAD  # into this K-tile (its WMMA here is still ahead)
                            if const_expr(q // NR >= WS_ROWS):
                                acc[q] = acc[q] * splat(fac(r, q // NR))
                        elif const_expr(not (final and r == U - 1)):
                            q = w - (W - RS_AHEAD)  # into the next K-tile (its WMMA here is done)
                            if const_expr(q // NR >= WS_ROWS):
                                acc[q] = acc[q] * splat(fac(r + 1, q // NR))
                        if const_expr(mi < WS_ROWS):
                            if const_expr((r, -1) not in srep):
                                srep[(r, -1)] = rep_byte(sb_cur, r)
                            if const_expr((r, mi) not in srep):
                                srep[(r, mi)] = rep_byte(sa_cur[mi], r)
                            acc[mi * NR + ni] = wmma(fb_, fa_, acc[mi * NR + ni], srep[(r, -1)], srep[(r, mi)])
                        else:
                            acc[mi * NR + ni] = wmma(fb_, fa_, acc[mi * NR + ni])
                    elif const_expr(is_blk):
                        bkey = (r, -1 - ni) if const_expr(b_rowscale) else (r, -1)
                        if const_expr(bkey not in srep):
                            srep[bkey] = rep_byte(sb_cur[ni] if b_rowscale else sb_cur, r)
                        if const_expr((r, mi) not in srep):
                            srep[(r, mi)] = rep_byte(sa_cur[mi], r)
                        acc[mi * NR + ni] = wmma(fb_, fa_, acc[mi * NR + ni], srep[bkey], srep[(r, mi)])
                    elif const_expr(not (abl & 4)):
                        acc[mi * NR + ni] = wmma(fb_, fa_, acc[mi * NR + ni])
                    rocdl.sched_barrier(0)
            if const_expr(not (is_blk and final and not use_wscale)):
                for q in range_constexpr(MR * NR):
                    c_frags[q].store(acc[q])
            if const_expr(is_blk and not final):
                for mi in range_constexpr(MR):
                    sc_carry[mi].store(fx.full(1, sa_nxt[mi], fx.Int32))
                for j_ in range_constexpr(NSB):
                    sc_carry[MR + j_].store(fx.full(1, sb_nxt[j_] if b_rowscale else sb_nxt, fx.Int32))
                if const_expr(not use_wscale):
                    for mi in range_constexpr(MR):
                        pv_carry[mi].store(fx.full(1, byte_of(sa_cur[mi], U - 1), fx.Int32))
                    pv_carry[MR].store(fx.full(1, byte_of(sb_cur, U - 1), fx.Int32))
            if const_expr(is_blk and final and not use_wscale):
                # leave the true sum: multiply by s_{K-1} = 2^(a + b - 254)
                for q in range_constexpr(MR * NR):
                    if const_expr(q // NR < WS_ROWS):
                        c_frags[q].store(acc[q])
                    else:
                        bits = (byte_of(sa_cur[q // NR], U - 1) + byte_of(sb_cur, U - 1) - 127) << 23
                        c_frags[q].store(acc[q] * fx.full(8, bits, fx.Int32).bitcast(fx.Float32))
            if const_expr(not final and not (abl & 2)):
                for i in range_constexpr(NCARRY):
                    carry[i].store(fv[(U, i)])

        for it in range(n_iter - 1):
            body(it * U, False)
        body((n_iter - 1) * U, True)

        # ---- epilogue: registers -> LDS -> global (TDM, OOB-clamped) ----
        tdm_ops.tensor_wait(0)
        if const_expr(b_path == 1):
            rocdl.s_wait_asynccnt(0)
        _sync()
        # each wave loads/stores its own whole row block (no descriptor split across waves)
        RC = tile_m // num_waves
        wvc = fx.Int32(rocdl.readfirstlane(_T.i32, tid // WAVE))
        gtC = _gview(gC_base, fx.Int64(blk_m + wvc * RC) * ldc64 + fx.Int64(blk_n), (RC, C_ROW), (C_ROW, 1))
        sCw = _view(fx.add_offset(base_ptr, wvc * (RC * C_ROW * OB)), out_cls, (RC, C_ROW), (C_ROW, 1))
        c_oob = m_oob - wvc * RC
        c_oob = (c_oob < 0).select(fx.Int32(0), c_oob)
        atomC = fx.rocdl.make_tdm_atom(gtC, [c_oob, tile_n], strides=[ldc64, None], num_warps=1)
        if const_expr(accumulate):
            fx.copy(atomC, gtC, sCw)
            tdm_ops.tensor_wait(0)
            fx.barrier()
        c_base_i = fx.Int32(fx.ptrtoint(base_ptr))
        for mi in range_constexpr(MR):
            row = wmb + mi * WMMA_M + lane16
            for ni in range_constexpr(NR):
                col = wnb + ni * WMMA_N + kgrp * (WMMA_N // 2)
                accv = c_frags[mi * NR + ni].load()
                if const_expr(accumulate == 1):
                    o0 = fx.Vector(lds_b128(c_base_i, (row * C_ROW + col) * 4))
                    o1 = fx.Vector(lds_b128(c_base_i, (row * C_ROW + col) * 4 + 16))
                    accv = accv + o0.shuffle(o1, list(range(8))).bitcast(fx.Float32)
                elif const_expr(acc16):
                    # 8 contiguous 16-bit old-C values = one b128; each lane rewrites exactly what it read
                    o16 = fx.Vector(lds_b128(c_base_i, (row * C_ROW + col) * 2)).bitcast(out_cls)
                    accv = accv + o16.to(fx.Float32)
                h = accv.to(out_cls)
                fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row * C_ROW + col) * OB)
        fx.barrier()
        fx.copy(atomC, sCw, gtC)
        tdm_ops.tensor_wait(0)

    kernel(
        arg_c, arg_a, arg_b, arg_sa, arg_sb, i32_m, i32_n, i32_k, i32_lda, i32_ldb, i32_ldc,
        value_attrs={"rocdl.cluster_dims": (f"{CS},1,1" if CS > 1 else None)},
    ).launch(
        grid=(((i32_m + (tile_m - 1)) // tile_m) * ((i32_n + (tile_n - 1)) // tile_n) * split_k, 1, 1),
        block=(block, 1, 1),
        stream=stream,
        cluster=((CS, 1, 1) if CS > 1 else None),
    )


launch_gemm_g90.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}
launch_gemm_g90.compile_hints["waves_per_eu"] = 1

_COMPILED: dict = {}


def _ptr(t):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _e8m0(s):
    """fp32 power-of-two scales -> uint8 E8M0 exponent bytes."""
    if s.dtype == torch.uint8:
        return s.contiguous()
    return ((s.float().contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def gemm_g90(a, b, out=None, *, scale_a=None, scale_b=None, tile=(256, 256), m_warp=2, n_warp=2, num_buffers=4, lead=16,
             group_m=4, out_dtype=torch.bfloat16, abl=0, cluster=(1, 1), mc_mode=1, tdm_mode=2, tk_bytes=128, wscale=False, b_path=0, ws_rows=0, split_k=1, accumulate=False):
    """``C = A @ B^T`` (NT). a (M, K), b (N, K) row-major; fp8 e4m3 or bf16."""
    M, K = a.shape
    N, Kb = b.shape
    assert K == Kb and a.dtype == b.dtype
    kind_id = _TORCH_KIND[a.dtype]
    if scale_a is not None:
        kind_id = 2 if wscale else 3
        sa8, sb8 = _e8m0(scale_a), _e8m0(scale_b)
        b_rowscale = 1 if tuple(sb8.shape) == (N, K // 128) else 0
        if b_rowscale:
            kind_id = 2  # per-row B scales: V_WMMA_SCALE path (keep per-wave acc <= 30 tiles: 128x128 tile)
        assert sa8.shape == (M, K // 128) and (b_rowscale or sb8.shape == (N // 128, K // 128))
    else:
        sa8 = sb8 = a
        b_rowscale = 0
    TK = tk_bytes // a.element_size()
    tm, tn = tile
    cm, cn = cluster
    if tdm_mode == 0 or cm * cn > 1:
        raise ValueError("tdm_mode 0 (descriptor split across waves) and clusters are disabled by the GPU safety rules")
    if cm * cn > 1 and (M % (tm * cm) or N % (tn * cn)):
        raise ValueError("cluster needs M, N multiples of the cluster block")
    if N % tn or K % (TK * num_buffers * split_k):
        raise ValueError(f"N={N} % {tn} or K={K} % {TK * num_buffers}")
    final_out = out
    if split_k > 1:
        out = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
    elif out is None:
        assert not accumulate, "accumulate needs out"
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    acc_mode = 0
    if accumulate:
        assert out.dtype in (torch.float32, torch.bfloat16, torch.float16), out.dtype
        assert tuple(out.shape) == (M, N) and out.stride(1) == 1 and out.stride(0) % 8 == 0
        acc_mode = 1 if out.dtype == torch.float32 else 2
    out_is_f16 = 1 if out.dtype == torch.float16 else 0
    args = (
        _ptr(out), _ptr(a), _ptr(b), _ptr(sa8), _ptr(sb8), M,
        fx.Stream(torch.cuda.current_stream(device=a.device)),
        N, K, a.stride(0), b.stride(0), out.stride(-2),
        tm, tn, m_warp, n_warp, num_buffers, kind_id, lead, group_m, out_is_f16, abl, cm, cn, mc_mode, tdm_mode, tk_bytes, b_path, ws_rows, split_k, b_rowscale,
        acc_mode,
    )
    key = (kind_id, tm, tn, m_warp, n_warp, num_buffers, lead, group_m, out_is_f16, abl, cm, cn, mc_mode,
           tdm_mode, tk_bytes, b_path, ws_rows, split_k, b_rowscale, acc_mode)
    fn = _COMPILED.get(key)
    if fn is None:
        _COMPILED[key] = flyc.compile(launch_gemm_g90, *args)
    else:
        fn(*args)
    if split_k > 1:
        red = out.sum(0)
        if final_out is not None:
            final_out.copy_(red)
            return final_out
        return red.to(out_dtype)
    return out


# =============================================================================
# Variant "b": A through TDM -> LDS -> ds_load, B straight from global into the
# WMMA registers (buffer_load_b128, issued DB K-tiles ahead).
#
# Why: the TDM global->LDS path saturates at ~30 TB/s device-wide (117 GB/s per CU,
# measured with L2-hot data), while 256x256 FP8 tiles at 90% of MAF need ~39 TB/s.
# The vector-memory path sustains ~45-48 TB/s on its own, so splitting the operand
# traffic between the two paths lifts the input-bandwidth ceiling. With n_warp = 4
# (m_warp = 1) every wave owns distinct B columns, so the direct B loads are not
# duplicated; only A lives in LDS, which leaves room for up to 8 TDM stages.
# =============================================================================


def make_plan_a(MR, NR, KI, LPF, lead):
    W = KI * MR * NR
    fr = [("A", ks, mi, ks * MR * NR + mi * NR) for ks in range(KI) for mi in range(MR)]
    L = len(fr) * LPF
    assert (LPF * W) % L == 0
    step = LPF * W // L

    def u(c):
        return (c * W) // L

    S0 = max(u(i * LPF + LPF - 1) - f[3] for i, f in enumerate(fr))
    S = -(-(S0 + lead) // step) * step
    assert S < W, f"lead {lead} too large (S={S}, W={W})"
    loads = [[] for _ in range(W)]
    for i in range(len(fr)):
        for j in range(LPF):
            loads[u(i * LPF + j)].append((i, j))
    idx = {(f[1], f[2]): i for i, f in enumerate(fr)}
    return W, fr, S, loads, S // step, idx


@flyc.jit
def launch_gemm_g90b(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    i32_m: fx.Int32,
    stream: fx.Stream,
    i32_n: fx.Int32,
    i32_k: fx.Int32,
    i32_lda: fx.Int32,
    i32_ldb: fx.Int32,
    i32_ldc: fx.Int32,
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    num_buffers: Constexpr[int],
    kind_id: Constexpr[int],
    lead: Constexpr[int],
    group_m: Constexpr[int],
    out_is_f16: Constexpr[int],
    b_ahead: Constexpr[int],
    abl: Constexpr[int] = 0,
):
    kname, e_cls, eb, WMMA_K = _KINDS[kind_id]
    is_bf16 = eb == 2
    WMMA_M = WMMA_N = 16
    NB = num_buffers
    U = NB
    DB = b_ahead
    assert NB >= 3 and 1 <= DB <= U
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    TK = 128 // eb
    KI = TK // WMMA_K
    LPF = WMMA_K * eb // 32
    FW = LPF * 4
    warp_m = tile_m // m_warp
    warp_n = tile_n // n_warp
    MR = warp_m // WMMA_M
    NR = warp_n // WMMA_N
    W, FR, S, LOADS, NCARRY, FIDX = make_plan_a(MR, NR, KI, LPF, lead)
    LB = NR * KI * LPF  # global B chunks per K-tile per lane
    assert W % LB == 0
    BSTEP = W // LB

    LDS_PAD = 16
    ROW_B = TK * eb + LDS_PAD
    STAGE_A = tile_m * ROW_B
    PITCH = STAGE_A
    C_ROW = tile_n + 8
    ARENA = max(NB * PITCH, tile_m * C_ROW * 2)
    assert ARENA <= 320 * 1024, f"LDS {ARENA} > 320 KiB"
    out_cls = fx.Float16 if out_is_f16 else fx.BFloat16
    kernel_name = (
        f"g90b_gemm_{kname}_t{tile_m}x{tile_n}_w{m_warp}x{n_warp}_nb{NB}_l{lead}_db{DB}_g{group_m}"
        + (f"_abl{abl}" if abl else "")
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldb: fx.Int32,
        i32_ldc: fx.Int32,
    ):
        K_TILES = i32_k // TK
        n_iter = K_TILES // U
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, _, _ = fx.block_idx
        n_tiles_m = (i32_m + (tile_m - 1)) // tile_m
        n_tiles_n = (i32_n + (tile_n - 1)) // tile_n
        pid = fx.Int32(bid_x)
        if const_expr(group_m > 1):
            per_group = group_m * n_tiles_n
            first_m = (pid // per_group) * group_m
            rem = n_tiles_m - first_m
            gsize = (rem < group_m).select(rem, fx.Int32(group_m))
            in_g = pid % per_group
            pm = first_m + in_g % gsize
            pn = in_g // gsize
        else:
            pm = pid % n_tiles_m
            pn = pid // n_tiles_m
        blk_m = pm * tile_m
        blk_n = pn * tile_n
        m_oob = i32_m - blk_m
        lda64 = fx.Int64(i32_lda)
        ldb64 = fx.Int64(i32_ldb)
        ldc64 = fx.Int64(i32_ldc)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA)
        base_ptr = arena.base_ptr

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

        gA_base = _cast(e_cls, arg_a)
        gC_base = _cast(out_cls, arg_c)
        gA = _gview(gA_base, fx.Int64(blk_m) * lda64, (tile_m, TK), (i32_lda, 1))
        atomA = fx.rocdl.make_tdm_atom(
            gA, [m_oob, None], strides=[lda64, None], num_warps=num_waves,
            pad_interval=TK, pad_amount=LDS_PAD // eb,
        )

        def sA_of(s):
            return _view(fx.add_offset(base_ptr, s * PITCH), e_cls, (tile_m, TK), (ROW_B // eb, 1))

        def issue(s, kt):
            fx.copy(atomA, gA, sA_of(s), imm_offset=fx.Int64(kt) * (TK * eb))

        lds_b128 = _lds_loader(128)
        wave = rocdl.readfirstlane(_T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wmb = (wave // n_warp) * warp_m
        wnb = (wave % n_warp) * warp_n
        base_i = fx.Int32(fx.ptrtoint(base_ptr))
        a_lane = base_i + (wmb + lane16) * ROW_B + kgrp * 16
        a_base = [_opaque_v(a_lane + s_ * PITCH) for s_ in range_constexpr(NB)]

        # B: buffer resource at this tile's first row; per-lane row offsets in VGPRs,
        # the K-tile / k-step / chunk offset rides in the scalar soffset.
        gB_tile = fx.add_offset(_cast(fx.Uint8, arg_b, 16), fx.Int64(blk_n) * ldb64 * eb)
        b_rsrc = rocdl.get_buffer_rsrc(rocdl.make_buffer_ptr(gB_tile))
        b_voff = [
            _opaque_v((wnb + ni * 16 + lane16) * (i32_ldb * eb) + kgrp * 16) for ni in range_constexpr(NR)
        ]
        v4_ty = fx.Vector.make_type(4, fx.Int32)

        def load_b_chunk(kt, ks, ni, j):
            soff = fx.Int32(kt) * (TK * eb) + (ks * WMMA_K * eb + 32 * j)
            return fx.Vector(rocdl.raw_ptr_buffer_load(v4_ty, b_rsrc, b_voff[ni], soff, 0))

        acc_ty = fx.Vector.make_type(8, fx.Float32)

        def wmma(bv, av, cv):
            if const_expr(is_bf16):
                bb = fx.Vector(bv).bitcast(fx.BFloat16)
                aa = fx.Vector(av).bitcast(fx.BFloat16)
                r = rocdl.wmma_f32_16x16x32_bf16_(acc_ty, as_ir_value(bb), as_ir_value(aa), as_ir_value(cv))
            else:
                r = rocdl.wmma_f32_16x16x128_fp8_fp8(acc_ty, bv, av, cv)
            return fx.Vector(r)

        def load_a_chunk(stage, i, j):
            ks, mi = FR[i][1], FR[i][2]
            off = mi * 16 * ROW_B + ks * WMMA_K * eb + 32 * j
            return fx.Vector(lds_b128(a_base[stage], off))

        def combine(ch):
            if const_expr(LPF == 4):
                v01 = ch[0].shuffle(ch[1], list(range(8)))
                v23 = ch[2].shuffle(ch[3], list(range(8)))
                return v01.shuffle(v23, list(range(16)))
            return ch[0].shuffle(ch[1], list(range(8)))

        # B chunk order inside a K-tile: (ks, ni, j)
        BCH = [(ks, ni, j) for ks in range(KI) for ni in range(NR) for j in range(LPF)]

        c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(MR * NR)]
        for cf in c_frags:
            cf.fill(0.0)
        carry_a = [fx.make_rmem_tensor(FW, fx.Int32) for _ in range_constexpr(NCARRY)]
        carry_b = [fx.make_rmem_tensor(FW, fx.Int32) for _ in range_constexpr(DB * KI * NR)]

        # ---- prologue ----
        for d in range_constexpr(DB):
            ch = {}
            for (ks, ni, j) in BCH:
                ch.setdefault((ks, ni), []).append(load_b_chunk(d, ks, ni, j))
            for ks in range_constexpr(KI):
                for ni in range_constexpr(NR):
                    carry_b[(d * KI + ks) * NR + ni].store(combine(ch[(ks, ni)]))
        for j in range_constexpr(NB - 1):
            issue(j, j)
        tdm_ops.tensor_wait(NB - 2)
        fx.barrier()
        pro = {}
        for t in range_constexpr(S):
            for (i, j) in LOADS[t]:
                pro.setdefault(i, []).append(load_a_chunk(0, i, j))
        for i in range_constexpr(NCARRY):
            carry_a[i].store(combine(pro[i]))

        def body(kt_base, final):
            fa = {}
            fb = {}
            for i in range_constexpr(NCARRY):
                fa[(0, i)] = carry_a[i].load()
            for d in range_constexpr(DB):
                for ks in range_constexpr(KI):
                    for ni in range_constexpr(NR):
                        fb[(d, ks, ni)] = carry_b[(d * KI + ks) * NR + ni].load()
            acc = [cf.load() for cf in c_frags]
            cha = {}
            chb = {}
            rocdl.sched_barrier(0)
            for r in range_constexpr(U):
                for w in range_constexpr(W):
                    t = w + S
                    rr, tt = (r, t) if t < W else (r + 1, t - W)
                    if const_expr(t == W and not (final and r == U - 1)):
                        if const_expr(final):
                            nw = max(0, min(r + NB - 2, U - 1) - (r + 1))
                        else:
                            nw = NB - 3
                        tdm_ops.tensor_wait(nw)
                        fx.barrier()
                        if const_expr(((not final) or (r + NB - 1 < U)) and not (abl & 1)):
                            issue((r + NB - 1) % NB, kt_base + (r + NB - 1))
                    if const_expr(not (final and rr >= U) and not (abl & 2)):
                        for (i, j) in LOADS[tt]:
                            cha.setdefault((rr, i), []).append(load_a_chunk(rr % NB, i, j))
                            if const_expr(j == LPF - 1):
                                fa[(rr, i)] = combine(cha[(rr, i)])
                    # B for K-tile r + DB, one chunk every BSTEP slots
                    bt = r + DB
                    if const_expr(w % BSTEP == 0 and not (final and bt >= U) and not (abl & 16)):
                        ks_, ni_, j_ = BCH[w // BSTEP]
                        chb.setdefault((bt, ks_, ni_), []).append(load_b_chunk(kt_base + bt, ks_, ni_, j_))
                        if const_expr(j_ == LPF - 1):
                            fb[(bt, ks_, ni_)] = combine(chb[(bt, ks_, ni_)])
                    ks = w // (MR * NR)
                    mi = (w % (MR * NR)) // NR
                    ni = w % NR
                    if const_expr(abl & 2):
                        av = fa[(0, FIDX[(ks, mi)] % NCARRY)]
                    else:
                        av = fa[(r, FIDX[(ks, mi)])]
                    bv = fb[(r % DB, ks, ni)] if const_expr(abl & 16) else fb[(r, ks, ni)]
                    if const_expr(not (abl & 4)):
                        acc[mi * NR + ni] = wmma(bv, av, acc[mi * NR + ni])
                    rocdl.sched_barrier(0)
            for q in range_constexpr(MR * NR):
                c_frags[q].store(acc[q])
            if const_expr(not final):
                if const_expr(not (abl & 2)):
                    for i in range_constexpr(NCARRY):
                        carry_a[i].store(fa[(U, i)])
                if const_expr(not (abl & 16)):
                    for d in range_constexpr(DB):
                        for ks in range_constexpr(KI):
                            for ni in range_constexpr(NR):
                                carry_b[(d * KI + ks) * NR + ni].store(fb[(U + d, ks, ni)])

        for it in range(n_iter - 1):
            body(it * U, False)
        body((n_iter - 1) * U, True)

        tdm_ops.tensor_wait(0)
        fx.barrier()
        sC = _view(base_ptr, out_cls, (tile_m, C_ROW), (C_ROW, 1))
        for mi in range_constexpr(MR):
            row = wmb + mi * WMMA_M + lane16
            for ni in range_constexpr(NR):
                col = wnb + ni * WMMA_N + kgrp * (WMMA_N // 2)
                h = c_frags[mi * NR + ni].load().to(out_cls)
                fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row * C_ROW + col) * 2)
        fx.barrier()
        gtC = _gview(gC_base, fx.Int64(blk_m) * ldc64 + fx.Int64(blk_n), (tile_m, C_ROW), (C_ROW, 1))
        atomC = fx.rocdl.make_tdm_atom(gtC, [m_oob, tile_n], strides=[ldc64, None], num_warps=num_waves)
        fx.copy(atomC, sC, gtC)
        tdm_ops.tensor_wait(0)

    kernel(arg_c, arg_a, arg_b, i32_m, i32_n, i32_k, i32_lda, i32_ldb, i32_ldc).launch(
        grid=(((i32_m + (tile_m - 1)) // tile_m) * ((i32_n + (tile_n - 1)) // tile_n), 1, 1),
        block=(block, 1, 1),
        stream=stream,
    )


launch_gemm_g90b.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}
launch_gemm_g90b.compile_hints["waves_per_eu"] = 1


def gemm_g90b(a, b, out=None, *, tile=(256, 256), m_warp=1, n_warp=4, num_buffers=8, lead=16,
              b_ahead=3, group_m=4, out_dtype=torch.bfloat16, abl=0):
    """``C = A @ B^T`` (NT), A via TDM/LDS, B direct global->VGPR."""
    M, K = a.shape
    N, Kb = b.shape
    assert K == Kb and a.dtype == b.dtype
    kind_id = _TORCH_KIND[a.dtype]
    TK = 128 // a.element_size()
    tm, tn = tile
    if N % tn or K % (TK * num_buffers):
        raise ValueError(f"N={N} % {tn} or K={K} % {TK * num_buffers}")
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    out_is_f16 = 1 if out.dtype == torch.float16 else 0
    args = (
        _ptr(out), _ptr(a), _ptr(b), M,
        fx.Stream(torch.cuda.current_stream(device=a.device)),
        N, K, a.stride(0), b.stride(0), out.stride(0),
        tm, tn, m_warp, n_warp, num_buffers, kind_id, lead, group_m, out_is_f16, b_ahead, abl,
    )
    key = ("b", kind_id, tm, tn, m_warp, n_warp, num_buffers, lead, group_m, out_is_f16, b_ahead, abl)
    fn = _COMPILED.get(key)
    if fn is None:
        _COMPILED[key] = flyc.compile(launch_gemm_g90b, *args)
    else:
        fn(*args)
    return out
