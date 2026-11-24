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

"""FlyDSL gfx1250 FP8 block-scaled (UE8M0 128x1 along K) variable-K wgrad on pool-ordered transposed operands.

The contraction runs over pool rows (tokens); each operand row carries one E8M0 exponent per 128 tokens,
repeated 4x into the MX planes sa [OUT_M, P/32], sb [OUT_N, P/32] that the WMMA-scale instruction reads.

out[g] (+)= lhsP[:, seg_g] @ rhsP[:, seg_g]^T
  lhsP: [OUT_M, P] fp8, rhsP: [OUT_N, P] fp8 (token axis contiguous),
  seg_g = [poffs[g], poffs[g+1]), 128-aligned and zero-padded, at least num_buffers - 1 k-tiles long.
TDM global->LDS loads, wave32 WMMA 16x16x128 fp8, packed-i32 operand fragments. One grid over (tiles_per_group x G),
per-group K window and K tile count read from a device table, and an optional beta=1 epilogue that TDM-loads the old
C tile into the LDS staging buffer and adds it before the store. This is the fallback wgrad (not persistent) for
expert windows the persistent kernels cannot hold; its scale planes hold ``kwin_max`` tokens per row.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Constexpr
from flydsl.expr.typing import T as _T

from .grouped_gemm_fp8blk_utils import _A_ID, WAVE, _make_lds_load, _ptr


@flyc.jit
def launch_vkblk_fp8_gfx1250(
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
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    num_buffers: Constexpr[int],
    a_id: Constexpr[int],
    b_id: Constexpr[int],
    beta_one: Constexpr[int],
    group_m: Constexpr[int] = 4,
    kwin_max: Constexpr[int] = 4096,
):
    a_cls, b_cls = _A_ID[a_id], _A_ID[b_id]
    elem_bytes = 1
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    assert tile_m % (m_warp * WMMA_M) == 0 and tile_n % (n_warp * WMMA_N) == 0
    assert tile_k % WMMA_K == 0 and num_buffers >= 2
    assert tile_m % num_waves == 0 and tile_n % num_waves == 0
    k_iters = tile_k // WMMA_K
    out_cls = fx.BFloat16

    LDS_PAD = 16
    A_ROW = tile_k + LDS_PAD
    B_ROW = tile_k + LDS_PAD
    STAGE_A = tile_m * A_ROW
    STAGE_B = tile_n * B_ROW
    PITCH = STAGE_A + STAGE_B
    MX_BLOCK = 32
    SKW = kwin_max // MX_BLOCK  # scale bytes of the longest expert window
    SP = SKW + 4
    SC_OFF = num_buffers * PITCH
    PLANE_SA = tile_m * SP
    PLANE_SB = tile_n * SP
    C_ROW = tile_n + 8
    ARENA = max(SC_OFF + PLANE_SA + PLANE_SB, tile_m * C_ROW * 2)
    assert ARENA <= 320 * 1024, f"LDS arena {ARENA} B exceeds 320 KiB"
    n_ops = 2

    kernel_name = f"moe_vkblk_fp8_gfx1250_w{kwin_max}_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}_nb{num_buffers}_a{a_id}{b_id}_b{beta_one}"

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_vk(
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
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, _, _ = fx.block_idx
        n_tiles_m = (i32_m + (tile_m - 1)) // tile_m
        n_tiles_n = (i32_n + (tile_n - 1)) // tile_n
        tpg = n_tiles_m * n_tiles_n
        bid = fx.Int32(bid_x)
        g = bid // tpg
        pid = bid % tpg
        per_group = group_m * n_tiles_n
        first_m = (pid // per_group) * group_m
        rem = n_tiles_m - first_m
        gsize = (rem < group_m).select(rem, fx.Int32(group_m))
        in_g = pid % per_group
        pm = first_m + in_g % gsize
        pn = in_g // gsize
        blk_m = pm * tile_m
        blk_n = pn * tile_n
        m_oob = i32_m - blk_m
        n_oob = i32_n - blk_n

        i32_ptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4
        )
        f32_ptr = fx.PointerType.get(  # noqa: F841
            elem_ty=fx.Float32.ir_type, address_space=fx.AddressSpace.Global, alignment=4
        )
        offs = fx.recast_iter(i32_ptr, arg_offs)
        kb = rocdl.readfirstlane(_T.i32, offs[g])
        ke = rocdl.readfirstlane(_T.i32, offs[g + fx.Int32(1)])
        K_TILES = (ke - kb) // tile_k

        lda64 = fx.Int64(i32_lda)
        ldb64 = fx.Int64(i32_ldb)
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

        gA = _gview(gA_base, fx.Int64(blk_m) * lda64 + fx.Int64(kb), (tile_m, tile_k), (i32_lda, 1))
        gB = _gview(gB_base, fx.Int64(blk_n) * ldb64 + fx.Int64(kb), (tile_n, tile_k), (i32_ldb, 1))
        atomA = fx.rocdl.make_tdm_atom(
            gA,
            [m_oob, None],
            strides=[lda64, None],
            num_warps=num_waves,
            pad_interval=tile_k,
            pad_amount=LDS_PAD,
        )
        atomB = fx.rocdl.make_tdm_atom(
            gB,
            [n_oob, None],
            strides=[ldb64, None],
            num_warps=num_waves,
            pad_interval=tile_k,
            pad_amount=LDS_PAD,
        )

        # the window's scale planes, loaded once per tile (expert window <= kwin_max tokens)
        ska = i32_lda // MX_BLOCK
        skb = i32_ldb // MX_BLOCK
        kw = (ke - kb) // MX_BLOCK
        gSA = _gview(
            _cast(fx.Uint8, arg_sa, 1),
            fx.Int64(blk_m) * fx.Int64(ska) + fx.Int64(kb // MX_BLOCK),
            (tile_m, SKW),
            (ska, 1),
        )
        gSB = _gview(
            _cast(fx.Uint8, arg_sb, 1),
            fx.Int64(blk_n) * fx.Int64(skb) + fx.Int64(kb // MX_BLOCK),
            (tile_n, SKW),
            (skb, 1),
        )
        atomSA = fx.rocdl.make_tdm_atom(
            gSA,
            [m_oob, kw],
            strides=[fx.Int64(ska), None],
            num_warps=num_waves,
            pad_interval=SKW,
            pad_amount=4,
        )
        atomSB = fx.rocdl.make_tdm_atom(
            gSB,
            [n_oob, kw],
            strides=[fx.Int64(skb), None],
            num_warps=num_waves,
            pad_interval=SKW,
            pad_amount=4,
        )
        sSA = _view(fx.add_offset(base_ptr, SC_OFF), fx.Uint8, (tile_m, SKW), (SP, 1), 1)
        sSB = _view(fx.add_offset(base_ptr, SC_OFF + PLANE_SA), fx.Uint8, (tile_n, SKW), (SP, 1), 1)

        def sA_of(s):
            return _view(_stage(s), a_cls, (tile_m, tile_k), (A_ROW, 1))

        def sB_of(s):
            return _view(fx.add_offset(_stage(s), STAGE_A), b_cls, (tile_n, tile_k), (B_ROW, 1))

        def issue(s, kt):
            kt64 = fx.Int64(kt)
            fx.copy(atomA, gA, sA_of(s), imm_offset=kt64 * (tile_k * elem_bytes))
            fx.copy(atomB, gB, sB_of(s), imm_offset=kt64 * (tile_k * elem_bytes))

        mma_atom = fx.make_mma_atom(
            fx.rocdl.WMMAScale(WMMA_M, WMMA_N, WMMA_K, b_cls, a_cls, fx.Float32, block_size=MX_BLOCK)
        )

        warp_m = tile_m // m_warp
        warp_n = tile_n // n_warp
        m_reps = warp_m // WMMA_M
        n_reps = warp_n // WMMA_N
        lds_b128 = _make_lds_load(128)
        lds_b32 = _make_lds_load(32)
        sc_base = fx.Int32(fx.ptrtoint(fx.add_offset(base_ptr, SC_OFF)))
        wave = rocdl.readfirstlane(_T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wmb = (wave // n_warp) * warp_m
        wnb = (wave % n_warp) * warp_n
        n_acc = m_reps * n_reps
        A_ROW_B = A_ROW * elem_bytes
        B_ROW_B = B_ROW * elem_bytes

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

        def compute(s, kt):
            base = fx.Int32(fx.ptrtoint(_stage(s)))
            for ks in range_constexpr(k_iters):
                kc = (kt * k_iters + ks) * 4
                bf = [
                    _operand(base, STAGE_A, wnb + ni * WMMA_N + lane16, B_ROW_B, ks)
                    for ni in range_constexpr(n_reps)
                ]
                sbv = [
                    lds_b32(sc_base, PLANE_SA + (wnb + ni * WMMA_N + lane16) * SP + kc)[0]
                    for ni in range_constexpr(n_reps)
                ]
                for mi in range_constexpr(m_reps):
                    af = _operand(base, 0, wmb + mi * WMMA_M + lane16, A_ROW_B, ks)
                    sav = lds_b32(sc_base, (wmb + mi * WMMA_M + lane16) * SP + kc)[0]
                    for ni in range_constexpr(n_reps):
                        idx = mi * n_reps + ni
                        fx.gemm(
                            mma_atom, c_frags[idx], bf[ni], af, c_frags[idx], scale_a=sbv[ni], scale_b=sav
                        )

        # K pipeline (K_TILES >= num_buffers - 1 is guaranteed by the padded layout)
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

        # epilogue: (old C tile via TDM into LDS) + scale * acc -> LDS -> TDM store
        fx.barrier()
        sC = _view(base_ptr, out_cls, (tile_m, C_ROW), (C_ROW, 1))
        gtC = _gview(
            gC_base,
            fx.Int64(g) * fx.Int64(i32_m) * ldc64 + fx.Int64(blk_m) * ldc64 + fx.Int64(blk_n),
            (tile_m, C_ROW),
            (C_ROW, 1),
        )
        atomC = fx.rocdl.make_tdm_atom(gtC, [m_oob, tile_n], strides=[ldc64, None], num_warps=num_waves)
        if const_expr(beta_one):
            fx.copy(atomC, gtC, sC)
            tdm_ops.tensor_wait(0)
            fx.barrier()
        cbase = fx.Int32(fx.ptrtoint(base_ptr))
        for mi in range_constexpr(m_reps):
            row = wmb + mi * WMMA_M + lane16
            for ni in range_constexpr(n_reps):
                col = wnb + ni * WMMA_N + kgrp * (WMMA_N // 2)
                v = c_frags[mi * n_reps + ni].load()
                if const_expr(beta_one):
                    old = fx.Vector(lds_b128(cbase, (row * C_ROW + col) * 2)).bitcast(out_cls).to(fx.Float32)
                    v = v + old
                h = v.to(out_cls)
                fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row * C_ROW + col) * 2)
        fx.barrier()
        fx.copy(atomC, sC, gtC)
        tdm_ops.tensor_wait(0)

    G_TILES = ((i32_m + (tile_m - 1)) // tile_m) * ((i32_n + (tile_n - 1)) // tile_n) * i32_g
    kernel_vk(arg_c, arg_a, arg_b, arg_offs, arg_sa, arg_sb, i32_m, i32_n, i32_g, i32_lda, i32_ldb).launch(
        grid=(G_TILES, 1, 1), block=(block, 1, 1), stream=stream
    )


launch_vkblk_fp8_gfx1250.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}

_COMPILED: dict = {}


def flyvkblk_fp8(
    lhsP,
    rhsP,
    lhs_scale,
    rhs_scale,
    poffs_i32,
    OUT_M,
    OUT_N,
    out,
    beta=1.0,
    tile=(256, 256, 128),
    m_warp=2,
    n_warp=4,
    num_buffers=2,
    group_m=4,
    *,
    kwin_max,
    check_windows=False,
):
    """out [G, OUT_M, OUT_N] bf16 contiguous (+)= per-expert lhsP[:, seg] @ rhsP[:, seg]^T.
    lhs_scale [OUT_M, P/32], rhs_scale [OUT_N, P/32] uint8 E8M0 (128x1 exponents x4). poffs_i32: [G+1] int32 padded
    offsets: every expert window a multiple of tile_k, at least num_buffers-1 k-tiles and at most ``kwin_max``
    tokens.
    kwin_max (required): the LDS scale planes hold ``kwin_max`` tokens of each row; a longer window would read the
    wrong scale bytes, so it must be a power of two (the TDM pad interval) bounding every window, and the arena
    must fit in LDS (kwin_max <= 8192 at the default tile). check_windows: verify the bound on the device offsets
    (one host sync) and raise ValueError if a window is longer."""
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    if kwin_max < 128 or kwin_max & (kwin_max - 1):
        raise ValueError(f"kwin_max={kwin_max} must be a power of two >= 128")
    if check_windows:
        longest = int((poffs_i32[1:] - poffs_i32[:-1]).max())
        if longest > kwin_max:
            raise ValueError(f"an expert window of {longest} tokens exceeds kwin_max={kwin_max}")
    G = poffs_i32.numel() - 1
    tm, tn, tk = tile
    assert OUT_N % tn == 0
    a_id = 0 if lhsP.dtype == torch.float8_e4m3fn else 1
    b_id = 0 if rhsP.dtype == torch.float8_e4m3fn else 1
    bo = 1 if beta == 1.0 else 0
    args = (
        _ptr(out),
        _ptr(lhsP),
        _ptr(rhsP),
        _ptr(poffs_i32),
        _ptr(lhs_scale),
        _ptr(rhs_scale),
        OUT_M,
        fx.Stream(torch.cuda.current_stream(device=lhsP.device)),
        OUT_N,
        G,
        lhsP.stride(0),
        rhsP.stride(0),
        tm,
        tn,
        tk,
        m_warp,
        n_warp,
        num_buffers,
        a_id,
        b_id,
        bo,
        group_m,
        kwin_max,
    )
    key = (tm, tn, tk, m_warp, n_warp, num_buffers, a_id, b_id, bo, group_m, kwin_max)
    c = _COMPILED.get(key)
    if c is None:
        _COMPILED[key] = flyc.compile(launch_vkblk_fp8_gfx1250, *args)
    else:
        c(*args)
    return out
