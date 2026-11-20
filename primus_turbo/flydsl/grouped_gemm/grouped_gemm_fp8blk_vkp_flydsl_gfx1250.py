"""FlyDSL gfx1250 FP8 block-scaled (UE8M0, 128x1 along K) variable-K wgrad, persistent.

out[g] (+)= lhsP[:, seg_g] @ rhsP[:, seg_g]^T with E8M0 exponents per (row, 128 pool rows):
  lhsP [OUT_M, P], rhsP [OUT_N, P] fp8 (pool rows contiguous), lhs_s [OUT_M, P/128], rhs_s [OUT_N, P/128] uint8
  (one byte per 128 tokens; the kernel repeats it 4x for the MX WMMA, block 32), seg_g = [poffs[g], poffs[g+1]).

Why persistent (profile of the one-tile-per-workgroup kernel, ledger P2): 56% of the stall was s_wait_tensorcnt, half
of it once per tile (scale planes, old-C load, final C store) with K = 768 tokens = 6 k-tiles per output tile. Here
each workgroup walks tiles bid, bid + grid, ...; all their k-tiles are one stream through a 2-stage LDS ring (the next
tile's first k-tile and its scale planes are issued during the current tile's last k-tile); the C tile is staged in its
own LDS region and its TDM store stays in flight while the next tile runs. Scale planes are double-buffered by tile
parity. Wait accounting (2 TDM ops per stage, LA = 1 stage ahead): the store is the only op newer than stage 0 of the
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


@flyc.jit
def launch_vkpblk_fp8_gfx1250(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_offs: fx.Pointer,
    arg_sa: fx.Pointer,
    arg_sb: fx.Pointer,
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

    kernel_name = (f"moe_vkpblk_fp8_gfx1250_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}_a{a_id}{b_id}"
                   f"_b{beta_one}_w{kwin_max}")

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_vkpblk(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_offs: fx.Pointer,
        arg_sa: fx.Pointer,
        arg_sb: fx.Pointer,
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
        offs = fx.recast_iter(i32_ptr, arg_offs)
        lda64 = fx.Int64(i32_lda)
        ldb64 = fx.Int64(i32_ldb)
        ska = i32_lda // 128
        skb = i32_ldb // 128

        def tile_params(t):
            g = t // tpg
            pid = t % tpg
            per_group = group_m * n_tiles_n
            first_m = (pid // per_group) * group_m
            rem = n_tiles_m - first_m
            gsize = (rem < group_m).select(rem, fx.Int32(group_m))
            in_g = pid % per_group
            pm = first_m + in_g % gsize
            pn = in_g // gsize
            kb = rocdl.readfirstlane(_T.i32, offs[g])
            ke = rocdl.readfirstlane(_T.i32, offs[g + fx.Int32(1)])
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
            gsa = _gview(gSA_base, fx.Int64(blk_m) * fx.Int64(ska) + fx.Int64(kb // 128), (tile_m, SPW), (ska, 1))
            gsb = _gview(gSB_base, fx.Int64(blk_n) * fx.Int64(skb) + fx.Int64(kb // 128), (tile_n, SPW), (skb, 1))
            asa = fx.rocdl.make_tdm_atom(gsa, [None, kt_n], strides=[fx.Int64(ska), None], num_warps=num_waves,
                                         pad_interval=SPW, pad_amount=4)
            asb = fx.rocdl.make_tdm_atom(gsb, [None, kt_n], strides=[fx.Int64(skb), None], num_warps=num_waves,
                                         pad_interval=SPW, pad_amount=4)
            poff = PL_OFF + pset * PSET
            fx.copy(asa, gsa, _view(fx.add_offset(base_ptr, poff), fx.Uint8, (tile_m, SPW), (SPB, 1), 1))
            fx.copy(asb, gsb, _view(fx.add_offset(base_ptr, poff + PL_A), fx.Uint8, (tile_n, SPW), (SPB, 1), 1))

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

        def _operand(base, plane_off, row, ks):
            b0 = plane_off + row * A_ROW + ks * WMMA_K + kgrp * 16
            v = [fx.Vector(lds_b128(base, b0 + 32 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            reg = fx.make_rmem_tensor(16, fx.Int32)
            reg.store(v01.shuffle(v23, list(range(16))))
            return reg

        def _scale(off_row, kt):
            """E8M0 byte kt of a plane row, repeated into the 4 MX-block bytes of one i32"""
            w = lds_b32(lds_base, off_row + (kt // 4) * 4)[0]
            return ((w >> ((kt % 4) * 8)) & 0xFF) * 0x01010101

        def compute(s, kt, pset):
            base = fx.Int32(fx.ptrtoint(_stage(s)))
            poff = PL_OFF + pset * PSET
            # every scale word of this k-step first (oldest LDS reads, so later waits stay partial), then the
            # fragments; the byte -> 4x repeat runs once per word (ledger P2b: per-mi scale load + convert
            # forced s_wait_dscnt 0 in front of every WMMA group)
            koff = (kt // 4) * 4
            sh = (kt % 4) * 8
            sbw = [lds_b32(lds_base, poff + PL_A + (wnb + ni * WMMA_N + lane16) * SPB + koff)[0]
                   for ni in range_constexpr(n_reps)]
            saw = [lds_b32(lds_base, poff + (wmb + mi * WMMA_M + lane16) * SPB + koff)[0] for mi in range_constexpr(m_reps)]
            bf = [_operand(base, STAGE_A, wnb + ni * WMMA_N + lane16, 0) for ni in range_constexpr(n_reps)]
            sbv = [((w >> sh) & 0xFF) * 0x01010101 for w in sbw]
            sav_all = [((w >> sh) & 0xFF) * 0x01010101 for w in saw]
            for mi in range_constexpr(m_reps):
                af = _operand(base, 0, wmb + mi * WMMA_M + lane16, 0)
                sav = sav_all[mi]
                for ni in range_constexpr(n_reps):
                    idx = mi * n_reps + ni
                    fx.gemm(mma_atom, c_frags[idx], bf[ni], af, c_frags[idx], scale_a=sbv[ni], scale_b=sav)

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
            for kt in range(kt_n):
                if kt == 0:
                    tdm_ops.tensor_wait(1)  # the previous tile's C store may still be in flight
                else:
                    tdm_ops.tensor_wait(0)
                fx.barrier()
                f = kt + 1
                if f < kt_n:
                    oa0, ob0 = offs_of(blk_m, blk_n, kb, f)
                    issue((gk + f) % num_buffers, oa0, ob0)
                else:
                    # next tile: its scale planes (other set), then its first k-tile
                    issue_planes(1 - pset, nbm, nbn, nkb, nktn)
                    oa1, ob1 = offs_of(nbm, nbn, nkb, 0)
                    issue((gk + f) % num_buffers, oa1, ob1)
                compute((gk + kt) % num_buffers, kt, pset)
            gk_reg.store(gk_reg.load() + kt_n)
            # epilogue: the previous C store must be done before sC is rewritten (4 newer ops: planes + stage)
            gtC, atomC = c_view(g, blk_m, blk_n)
            tdm_ops.tensor_wait(4)
            fx.barrier()
            if const_expr(beta_one):
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
                    fx.ptr_store(v.to(out_cls).bitcast(fx.Int8), fx.add_offset(base_ptr, C_OFF) + (row * C_ROW + col) * 2)
            fx.barrier()
            fx.copy(atomC, sC, gtC)
        tdm_ops.tensor_wait(0)

    kernel_vkpblk(arg_c, arg_a, arg_b, arg_offs, arg_sa, arg_sb, i32_m, i32_n, i32_g, i32_lda, i32_ldb,
                  i32_nwg).launch(grid=(i32_nwg, 1, 1), block=(block, 1, 1), stream=stream)


launch_vkpblk_fp8_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}

_COMPILED: dict = {}


def _ptr(t):
    return flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)


def flyvkpblk_fp8(lhsP, rhsP, lhs_s, rhs_s, poffs_i32, OUT_M, OUT_N, out, beta=1.0, tile=(256, 256, 128),
                  m_warp=2, n_warp=4, group_m=4, kwin_max=4096, nwg=None):
    """out [G, OUT_M, OUT_N] bf16 (+)= per-expert lhsP[:, seg] @ rhsP[:, seg]^T; lhs_s/rhs_s compact E8M0
    [rows, P/128] uint8. beta 1 accumulates into out, 0 overwrites it."""
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
    bo = 1 if beta == 1.0 else 0
    args = (_ptr(out), _ptr(lhsP), _ptr(rhsP), _ptr(poffs_i32), _ptr(lhs_s), _ptr(rhs_s), OUT_M,
            fx.Stream(torch.cuda.current_stream(device=lhsP.device)), OUT_N, G, lhsP.stride(0), rhsP.stride(0), nwg,
            tm, tn, tk, m_warp, n_warp, a_id, b_id, bo, group_m, kwin_max)
    key = (tm, tn, tk, m_warp, n_warp, a_id, b_id, bo, group_m, kwin_max)
    c = _COMPILED.get(key)
    if c is None:
        _COMPILED[key] = flyc.compile(launch_vkpblk_fp8_gfx1250, *args)
    else:
        c(*args)
    return out
