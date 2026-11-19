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

# The mainloop is adapted from aiter (https://github.com/ROCm/aiter, amd-aiter 0.1.24.post1):
# aiter/ops/flydsl/kernels/gemm_a8w8_256x256_gfx1250.py, gemm_a8w8_gfx1250.py and gemm_common_gfx1250.py, where
# aiter is Copyright (C) 2024-2026, Advanced Micro Devices, Inc., under the MIT license.

"""MXFP8 NT GEMMs for gfx1250: one mainloop for three modes, fp32 accumulation, ``out`` bf16 or fp32, optionally
``out += result`` (beta = 1, rounded once).

* ``gemm_mxfp8_gfx1250``: dense ``out[M, N] (+)= (a * sa)[M, K] @ (b * sb)[N, K]^T``;
* ``grouped_gemm_mxfp8_gfx1250``: routed experts fwd / dgrad, ``out[offs[g] + i] = a[offs_pad[g] + i] @ b[g]^T``;
* ``grouped_gemm_mxfp8_variable_k_gfx1250``: routed experts wgrad, ``out[g] (+)= a[:, T_g] @ b[:, T_g]^T``.

Operands, in the layout the MX quantizers write: E4M3 ``a`` row-major (K contiguous, 16-byte rows), ``b`` in 16x16
blocks ``[N/16][K/16][16][16]``, one E8M0 scale byte per 32 K packed ``[ceil(rows/32)][K/128][32][4]``; K is a
multiple of 128 (the quantizers pad it). A half last K step (kpair 2, K % 256 == 128) loads whole block rows of
``b``, 2048 B past its last one (the TDM clamps the data, not the addressing). The quantizers allocate that slack;
without it the entry points use kpair 1, which has no partial step.

The mainloop is aiter's gfx1250 tile: 2x2 waves of up to 128x128, ``v_wmma_scale_f32_16x16x128_f8f6f4`` with one
scale VGPR feeding two WMMAs, ``num_buffers`` LDS slots of ``kpair`` 128-wide K steps filled by per-wave TDM
descriptors (wave 0: A, 1: B, 2: A scales, 3: B scales), a split signal / wait barrier per slot, cluster multicast
of the shared tiles. Added here: row-major A in padded LDS rows, mx32 scales, M / N / K tails by TDM out-of-bounds
zero fill (a half last K step loads half rows), an LDS-staged epilogue with 16-byte buffer stores, beta = 1 (the
old C fetched during the last K step where it fits), persistent workgroups and the grouped modes. A grouped-M
cluster along M whose m tiles straddle two experts (or the end) loads B per workgroup instead of multicasting it;
its dead workgroups run the K loop (the cluster barriers) and store nothing.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.rocdl import cluster, tdm_ops
from flydsl.expr.typing import Constexpr, T, as_ir_value
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.gemm.gemm_mxfp8_gfx1250_configs import (
    DENSE,
    GROUPED_M,
    VARIABLE_K,
    MXGemmConfig,
    pick_config,
    pick_grouped_config,
)
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import _make_lds_copy_ops
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from primus_turbo.pytorch.core.utils import get_num_cus, is_gfx1250

WAVE = 32
WMMA_K = 128  # K per WMMA = per scale word (4 E8M0 bytes)
LDS_PAD = 16  # bytes after every A LDS row: conflict-free ds_load_b128 fragment reads
C_PAD_B = 16  # bytes after every row of the epilogue's C staging
LDS_BYTES = 320 * 1024
CLUSTER_SYNC_REVS = 8  # a cluster barrier every this many pipeline revolutions keeps multicast partners close
_WG_BARRIER = -1
_E8M0 = (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))


def _tdm_group1(mask, tile0: int, tile1: int, dim0, dim1, stride, pad: int):
    """GROUP1 of a 2-D TDM load of 1-byte elements, inner dim0 (``tile0`` bytes per row), outer dim1 (``tile1`` rows),
    early timeout on; ``pad`` bytes of LDS padding after every row; ``dim0`` / ``dim1`` are the (>= 0) extents from
    the descriptor's address, past which the load zero-fills."""
    cfg = 1 << 21
    if pad:
        cfg |= (1 << 20) | (((tile0 // 4).bit_length() - 2) << 22) | ((pad // 4 - 1) << 25)
    return [
        fx.Int32(cfg) | (fx.Int32(mask) & 0xFFFF),
        (dim0 & 0xFFFF) << 16,
        ((dim0 >> 16) & 0xFFFF) | ((dim1 & 0xFFFF) << 16),
        ((dim1 >> 16) & 0xFFFF) | fx.Int32(tile0 << 16),
        fx.Int32(tile1),
        fx.Int32(stride),
        fx.Int32(0),
        fx.Int32(0),
    ]


@flyc.jit
def _launch_gemm_mxfp8(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_sa: fx.Pointer,
    arg_sb: fx.Pointer,
    arg_g0: fx.Pointer,
    arg_g1: fx.Pointer,
    i32_m: fx.Int32,
    i32_n: fx.Int32,
    i32_k: fx.Int32,
    i32_lda: fx.Int32,
    i32_ldb: fx.Int32,
    i32_ldc: fx.Int32,
    i32_ldsa: fx.Int32,
    i32_ldsb: fx.Int32,
    stream: fx.Stream,
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    num_buffers: Constexpr[int],
    kpair: Constexpr[int],
    cluster_m: Constexpr[int],
    cluster_n: Constexpr[int],
    k_tail: Constexpr[bool],
    mode: Constexpr[int],
    n_groups: Constexpr[int],
    out_f32: Constexpr[bool],
    beta1: Constexpr[bool],
    c_early: Constexpr[bool],
    persist: Constexpr[bool],
    n_cus: Constexpr[int],
):
    """``arg_g0`` / ``arg_g1``: grouped-M: the tight and the 32-padded row offsets of the experts (int64, G + 1);
    variable-K: the 128-padded token offsets (``arg_g0``). Unused in dense mode. ``c_early`` (beta = 1 into bf16,
    kpair 2, whole N tiles): the old C tile comes into the free A + B slot during the last K step. ``persist``
    (2 slots of kpair 2, bf16 staging): ``n_cus`` workgroups, each loading its next tile during the stores."""
    assert tile_m in (128, 256) and tile_n in (128, 256) and kpair in (1, 2) and num_buffers in (2, 3, 4)
    assert not c_early or (beta1 and not out_f32 and kpair == 2)
    assert not persist or (num_buffers == 2 and kpair == 2 and not out_f32 and (c_early or not beta1))
    G, NB, KPAIR = n_groups, num_buffers, kpair
    SUPER_K = WMMA_K * KPAIR
    STG = 4 if out_f32 or (beta1 and not c_early) else 2  # bytes per staged output element
    OB = 4 if out_f32 else 2  # bytes per output element
    C_ROW_B = tile_n * STG + C_PAD_B
    # LDS plan: NB slots of A (rows padded), B (16x16 blocks) and the two scale tiles; the epilogue stages C over
    # them. Slot j of A and B is one [B_j | A_j] block where a free slot must hold a whole bf16 C tile (rows padded
    # as staged), else aiter's order: all B slots (64 KiB aligned base), then all A slots.
    LDS_ROW = SUPER_K + LDS_PAD
    STAGE = (tile_m * LDS_ROW, tile_n * SUPER_K, tile_m // 32 * SUPER_K, tile_n // 32 * SUPER_K)
    PITCH = STAGE[0] + STAGE[1]
    slot_major = c_early or persist
    assert not slot_major or tile_m * C_ROW_B <= PITCH
    BASE = (STAGE[1] if slot_major else NB * STAGE[1], 0, NB * PITCH, NB * (PITCH + STAGE[2]))
    STAGE = (PITCH, PITCH, STAGE[2], STAGE[3]) if slot_major else STAGE
    ARENA = max(BASE[3] + NB * STAGE[3], tile_m * C_ROW_B)
    assert ARENA <= LDS_BYTES, f"t{tile_m}x{tile_n} nb{NB} kp{KPAIR} needs {ARENA} B of LDS"
    # c_pre (persistent c_early): the old C's first c_pre rows, as many as the LDS past the slots holds, come there
    # during the previous tile's epilogue (once its stores have read them); the last K step fetches only the rest
    # into the free slot
    PRE_BASE = (ARENA + 15) // 16 * 16
    c_pre = min((LDS_BYTES - PRE_BASE) // C_ROW_B // 16 * 16, tile_m // 2) if persist and c_early else 0
    if c_pre:
        ARENA = PRE_BASE + c_pre * C_ROW_B
    use_cluster = cluster_m * cluster_n > 1
    UNROLL = KPAIR * NB
    HALF_TAIL = KPAIR == 2 and k_tail  # a last K step of <= 128: half a slot, its second WMMA step skipped
    WM, WN = tile_m // 2, tile_n // 2  # 2x2 waves
    m_rep, n_rep = WM // 16, WN // 16
    half_m, half_n = m_rep // 2, n_rep // 2
    n_acc = m_rep * n_rep
    N_SA, N_SB = WM // 32, WN // 32  # scale VGPRs: one per 32 rows (ROW0 / ROW1 feed two WMMAs)
    BLOCK = 4 * WAVE
    DS_PER_FRAG = 4  # ds_load_b128 per 16-dword fragment
    # fragment geometry (16-row step, 16-byte K-chunk step): A's row-major rows, and aiter's 16x16 blocks of B (per
    # 16 rows [K/16][16 rows][16 B]: one TDM row of SUPER_K * 16 bytes per 16 rows)
    GEO_A = (16 * LDS_ROW, 32)
    GEO_B = (SUPER_K * 16, 512)
    kname = (
        f"gemm_mxfp8_gfx1250_{('dense', f'grouped{G}', f'vark{G}')[mode]}_t{tile_m}x{tile_n}_nb{NB}_kp{KPAIR}"
        f"_c{cluster_m}x{cluster_n}"
        + ("_kt" if k_tail else "")
        + ("_f32" if out_f32 else "")
        + ("_acc" if beta1 else "")
        + ("_ce" if c_early else "")
        + ("_p" if persist else "")
    )

    @flyc.kernel(name=kname, known_block_size=[BLOCK, 1, 1])
    def kernel_gemm_mxfp8(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_sa: fx.Pointer,
        arg_sb: fx.Pointer,
        arg_g0: fx.Pointer,
        arg_g1: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldb: fx.Int32,
        i32_ldc: fx.Int32,
        i32_ldsa: fx.Int32,
        i32_ldsb: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16, kgrp = lane % 16, lane // 16
        wave_m, wave_n = wave // 2, wave % 2
        zero = fx.Int32(0)

        def _table(arg):  # an offsets table (int64 [G + 1]) as a buffer resource
            return create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg)), num_records_bytes=8 * (G + 1))

        def _addr_words(addr):  # the address words of a TDM descriptor: lo, hi | type
            return [fx.Int32(addr), fx.Int32(addr >> 32) | fx.Int32(-(1 << 31))]

        def _end_unless_lt(a, b):  # the workgroup ends here unless a < b (uniform values)
            asm = "s_cmp_lt_u32 $0, $1\n\ts_cbranch_scc1 1f\n\ts_endpgm\n\t1:"
            ops = [as_ir_value(a), as_ir_value(b)]
            _llvm.inline_asm(None, ops, asm, "s,s,~{scc},~{memory}", has_side_effects=True)

        vk_tab = None
        if const_expr(mode == VARIABLE_K and G <= 32):
            # the experts' token offsets, lane e holding offs[e] / offs[e + 1] (read once; a tile then needs no load)
            rs_vk = _table(arg_g0)
            vk_tab = (
                fx.Int32(buffer_load(rs_vk, lane * 2, vec_width=1, dtype=T.i32)),
                fx.Int32(buffer_load(rs_vk, lane * 2 + 2, vec_width=1, dtype=T.i32)),
            )

        def _decode_vk(bx, by, e):
            """variable-K: output tile (bx, by) of expert e over its tokens [offs[e], offs[e + 1])."""
            blk_m, blk_n = bx * tile_m, by * tile_n
            if const_expr(vk_tab is not None):
                k0 = fx.Int32(rocdl.readlane(T.i32, vk_tab[0], e))
                klen = fx.Int32(rocdl.readlane(T.i32, vk_tab[1], e)) - k0
            else:
                o2 = Vec(buffer_load(_table(arg_g0), e * 2, vec_width=4, is_scalar=True))
                k0, klen = fx.Int32(o2[0]), fx.Int32(o2[2]) - fx.Int32(o2[0])
            supers = (klen + (SUPER_K - 1)) // SUPER_K
            # an empty expert: one zero-extent K step, from the operands' first column (offs[e] may be T, the end)
            k0 = (klen > zero).select(k0, zero)
            supers = (supers > zero).select(supers, fx.Int32(1))
            # (a_row0, out_row0, m_valid, blk_n, n_valid, expert, k0, klen, supers)
            return blk_m, blk_m, i32_m - blk_m, blk_n, i32_n - blk_n, e, k0, klen, supers

        def _gm_table():
            """grouped-M, once per workgroup: lane l of chunk c holds expert 32 c + l's first m tile (a DPP prefix sum
            of the tile counts ceil(len_g / tile_m)), tile count, rows and row offsets; and the number of m tiles."""
            rs0, rs1 = _table(arg_g0), _table(arg_g1)

            def _scan(v):  # inclusive prefix over the wave: row_shr within 16-lane rows, then row 0's total
                for k in (1, 2, 4, 8):
                    v = v + fx.Int32(_rocdl.update_dpp(T.i32, _raw(zero), _raw(v), 0x110 + k, 0xF, 0xF, True))
                return v + (lane >= 16).select(fx.Int32(rocdl.readlane(T.i32, v, 15)), zero)

            acc, chunks = zero, []
            for ch in range_constexpr((G + 31) // 32):
                g = lane + ch * 32
                o0 = fx.Int32(buffer_load(rs0, g * 2, vec_width=1, dtype=T.i32))  # low words (int64 tables)
                o1 = fx.Int32(buffer_load(rs0, g * 2 + 2, vec_width=1, dtype=T.i32))
                ln = (g < G).select(o1 - o0, zero)
                p0 = fx.Int32(buffer_load(rs1, g * 2, vec_width=1, dtype=T.i32))
                n_g = (ln + (tile_m - 1)) // tile_m
                incl = _scan(n_g)
                chunks.append((acc + incl - n_g, n_g, ln, p0, o0))
                acc = acc + fx.Int32(rocdl.readlane(T.i32, incl, 31))
            return chunks, acc

        def _decode_gm(tab, t, blk_n, pair=False):
            """grouped-M: m tile t of the experts' tiles in order; a tile past them is dead (no rows, no B). A ballot
            of the expert whose tiles hold t, its entries by readlane (no load). ``pair`` (cluster_m > 1): also
            whether the cluster's m tiles t - t % cluster_m ... all belong to t's expert (they share its B)."""
            chunks, acc = tab
            first, len_g, in0, out0, expert = zero, zero, zero, zero, zero
            for ch, (excl, n_g, ln, p0, o0) in enumerate(chunks):
                hit = fx.Int32(rocdl.ballot(T.i32, (t >= excl) & (t < excl + n_g)))
                src = fx.Int32(
                    _llvm.call_intrinsic(T.i32, "llvm.cttz.i32", [_raw(hit), _raw(fx.Boolean(False))], [], [])
                )
                found = hit != zero
                expert = found.select(src + ch * 32, expert)
                first = found.select(fx.Int32(rocdl.readlane(T.i32, excl, src)), first)
                len_g = found.select(fx.Int32(rocdl.readlane(T.i32, ln, src)), len_g)
                in0 = found.select(fx.Int32(rocdl.readlane(T.i32, p0, src)), in0)
                out0 = found.select(fx.Int32(rocdl.readlane(T.i32, o0, src)), out0)
            local = (t - first) * tile_m
            rows = len_g - local
            m_valid = (rows < tile_m).select(rows, fx.Int32(tile_m))
            m_valid = (t < acc).select(m_valid, zero)
            n_valid = (t < acc).select(i32_n - blk_n, zero)
            # (a dead tile of a live cluster runs the K steps too: the cluster barriers)
            supers = (i32_k + (SUPER_K - 1)) // SUPER_K
            dec = (in0 + local, out0 + local, m_valid, blk_n, n_valid, expert, zero, i32_k, supers)
            if const_expr(pair):
                t0 = t // cluster_m * cluster_m
                return dec, (t0 >= first) & ((t0 + (cluster_m - 1) - first) * tile_m < len_g)
            return dec

        if const_expr(mode == DENSE and not persist):
            blk_n = fx.Int32(bid_y) * tile_n
            blk_m = (fx.Int32(bid_z) * fx.Int32(fx.grid_dim.x) + fx.Int32(bid_x)) * tile_m
            supers = (i32_k + (SUPER_K - 1)) // SUPER_K
            # m_valid <= 0 for a workgroup past the edge
            dec = (blk_m, blk_m, i32_m - blk_m, blk_n, i32_n - blk_n, zero, zero, i32_k, supers)
        elif const_expr(mode == GROUPED_M and not persist):
            t = fx.Int32(bid_z) * fx.Int32(fx.grid_dim.x) + fx.Int32(bid_x)
            gm_tab = _gm_table()
            acc = gm_tab[1]
            if const_expr(cluster_m > 1):  # (consecutive m tiles: grid.x is a multiple of cluster_m)
                dec, gm_same = _decode_gm(gm_tab, t, fx.Int32(bid_y) * tile_n, pair=True)
                t_end = t // cluster_m * cluster_m
            else:
                dec, t_end = _decode_gm(gm_tab, t, fx.Int32(bid_y) * tile_n), t
            # a tile past the experts' (the grid is an upper bound) ends its workgroup here: uniform, and a cluster
            # (along N) shares the m tile, so whole clusters end (along M: when its first m tile is past them)
            _end_unless_lt(t_end, acc)
        elif const_expr(mode == VARIABLE_K and not persist):
            dec = _decode_vk(fx.Int32(bid_x), fx.Int32(bid_y), fx.Int32(bid_z))
        a_mask = b_mask = 0
        if const_expr(use_cluster):
            lx = fx.Int32(rocdl.cluster_workgroup_id_x())
            ly = fx.Int32(rocdl.cluster_workgroup_id_y())
            # flat id in the cluster = x + y * cluster_m; A is shared along y (same M tile), B along x
            a_mask = fx.Int32(sum(1 << (y * cluster_m) for y in range(cluster_n))) << lx
            b_mask = fx.Int32((1 << cluster_m) - 1) << (ly * cluster_m)
            if const_expr(mode == GROUPED_M and cluster_m > 1):  # m tiles of two experts: B per workgroup
                b_mask = gm_same.select(b_mask, zero)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA)
        base_ptr = arena.base_ptr
        lds0 = fx.Int32(fx.ptrtoint(base_ptr))

        # ---- per-wave TDM descriptor of a tile (wave 0: A, 1: B, 2: A scales, 3: B scales) ----
        def _owner(w, dec):
            """[addr lo, addr hi | type, LDS slot 0, slot step, K step, inner extent, GROUP1 words 0, 2 (dim1
            half), 3, 4, 5, and words 0, 3 of a half K step] of operand ``w`` of the tile ``dec``; the K step and
            inner extent are in the operand's bytes."""
            a_row0, _, m_valid, blk_n, n_valid, expert, k0, klen, _ = dec
            rows_valid, blk = (m_valid, a_row0) if w % 2 == 0 else (n_valid, blk_n)
            rows = tile_m if w % 2 == 0 else tile_n
            ld, ptr, mask = (
                (i32_lda, arg_a, a_mask),
                (i32_ldb, arg_b, b_mask),
                (i32_ldsa, arg_sa, a_mask),
                (i32_ldsb, arg_sb, b_mask),
            )[w]
            pad = LDS_PAD if w < 2 else 0
            eoff = fx.Int64(0)
            if const_expr(mode == GROUPED_M and w % 2 == 1):  # b [G, N, K], its scales [G, N / 32, ...]
                eoff = fx.Int64(expert) * fx.Int64(i32_n // (32 if w == 3 else 1)) * fx.Int64(ld)
            if const_expr(w == 1):  # [N/16][K * 16]: one TDM row per 16 rows of B
                rows, row0, rows_valid = tile_n // 16, blk // 16, (rows_valid + 15) // 16
                inner = step = SUPER_K * 16
                total, kb, ld, pad = klen * 16, k0 * 16, i32_ldb * 16, 0
            elif const_expr(w == 0):
                inner = step = SUPER_K
                total, row0, kb = klen, blk, k0
            else:  # [R/32][K/128][32][4] (variable-K token offsets are multiples of 128)
                rows, inner = rows // 32, SUPER_K
                step, total, row0 = SUPER_K, (klen + 127) // 128 * 128, blk // 32
                rows_valid, kb = (rows_valid + 31) // 32, k0
            ext1 = (rows_valid > zero).select(rows_valid, zero)
            addr = fx.Int64(fx.ptrtoint(ptr)) + eoff + fx.Int64(row0) * fx.Int64(ld) + fx.Int64(kb)
            g1 = _tdm_group1(mask, inner, rows, total, ext1, ld, pad)
            # a half K step (HALF_TAIL): half the row, the rest of the LDS row skipped as padding, so nothing past
            # the data is addressed (the inner extent of a TDM load clamps the data, not the addressing). The pad
            # field holds at most 512 B, the interval 1 KiB: not B's 2 KiB halves, which load the whole step
            g1h = _tdm_group1(mask, inner // 2, rows, total, ext1, ld, pad + inner // 2) if w != 1 else g1
            words = [*_addr_words(addr), lds0 + BASE[w], fx.Int32(STAGE[w]), fx.Int32(step), total]
            return words + [g1[0], g1[2] & fx.Int32(-65536), g1[3], g1[4], g1[5], g1h[0], g1h[3]]

        def _desc(dec):
            d = [zero] * 13
            if wave == 0:
                d = _owner(0, dec)
            elif wave == 1:
                d = _owner(1, dec)
            elif wave == 2:
                d = _owner(2, dec)
            else:
                d = _owner(3, dec)
            return d

        def _make_prepare(d, supers, rot):
            """The load of K offset ``delta`` (this wave's units, clamped to the last K step) into logical slot
            ``slot`` of a tile (descriptor words ``d``); slot j is physical slot j ^ ``rot`` (persistent tiles).
            c_early / persistent: loads past the last K step (re-loads of it) go into the last step's own slot (the
            same bytes), never into the free slot (the old C, or the next tile's first loads, go there)."""
            t_lo, t_hi, t_lds, t_lds_step, t_step, t_total = d[:6]
            t_g10, t_g12_hi, t_g13, t_g14, t_g15, t_g10h, t_g13h = d[6:]
            last_delta = (supers - 1) * t_step
            last_slot = (supers - 1) % NB

            def _prepare(slot, delta):
                past = delta > last_delta
                delta = past.select(last_delta, delta)
                if const_expr(c_early or persist):
                    slot = past.select(last_slot, fx.Int32(slot))
                if const_expr(persist):
                    slot = fx.Int32(slot) ^ rot
                lo, hi = tdm_ops.add_addr_with_carry(t_lo, t_hi, delta)
                g0 = Vec.from_elements(
                    [fx.Int32(1), t_lds + t_lds_step * slot, fx.Int32(lo), fx.Int32(hi)], fx.Int32
                )
                d0 = t_total - delta if const_expr(k_tail) else t_total
                d0 = (d0 > zero).select(d0, zero)
                w0, w3 = t_g10, t_g13
                if const_expr(HALF_TAIL):
                    half = d0 <= (t_step >> 1)
                    w0, w3 = half.select(t_g10h, t_g10), half.select(t_g13h, t_g13)
                g1 = Vec.from_elements(
                    [w0, (d0 & 0xFFFF) << 16, ((d0 >> 16) & 0xFFFF) | t_g12_hi, w3, t_g14, t_g15, zero, zero],
                    fx.Int32,
                )
                return tdm_ops.TDMDescriptor2D(as_ir_value(g0), as_ir_value(g1))

            return _prepare, t_step, last_slot

        wmb, wnb = wave_m * WM, wave_n * WN
        lds_load_b32, _ = _make_lds_copy_ops(32)
        lds_load_b128, _ = _make_lds_copy_ops(128)

        # ---- fragment / scale read addresses per unrolled stage (displacements stay DS immediates) ----
        a_byte = (wmb + lane16) * LDS_ROW + kgrp * 16
        b_byte = (wnb // 16) * GEO_B[0] + kgrp * 256 + lane16 * 16
        sa_byte, sb_byte = (wmb // 32) * SUPER_K + lane * 4, (wnb // 32) * SUPER_K + lane * 4
        # per operand (A, B, SA, SB): the lane's offset in a slot and the offset of a slot's second K step
        READS = ((a_byte, WMMA_K), (b_byte, WMMA_K * 16), (sa_byte, WMMA_K), (sb_byte, WMMA_K))

        def _planar(off, lane_byte):
            # a pointer offset, not an integer add: LLVM keeps one base VGPR per stage and the fragment
            # displacements in the 16-bit DS offset instead of folding the stage base into every load
            base = fx.index_cast(T.index, fx.ptrtoint(fx.add_offset(base_ptr, off)))
            return base + fx.index_cast(T.index, lane_byte)

        def _stage_addrs(rot):
            """Per unrolled stage: the fragment / scale base addresses of A, B, SA, SB (physical slot = slot ^ rot)."""
            addrs = ([], [], [], [])
            for st in range_constexpr(UNROLL):
                slot, par = st // KPAIR, st % KPAIR
                for w, (lb, po) in enumerate(READS):
                    if const_expr(persist):
                        addrs[w].append(_planar(BASE[w] + par * po, lb + (fx.Int32(slot) ^ rot) * STAGE[w]))
                    else:
                        addrs[w].append(_planar(BASE[w] + slot * STAGE[w] + par * po, lb))
            return addrs

        def _join(v):  # four 4-dword loads -> one 16-dword fragment
            lo8 = list(range(8))
            return v[0].shuffle(v[1], lo8).shuffle(v[2].shuffle(v[3], lo8), list(range(16)))

        def _load_frag(addr, idx, geo=GEO_A):
            return _join([Vec(lds_load_b128(addr, idx * geo[0] + geo[1] * j)) for j in range_constexpr(4)])

        def _rmem(n, v):
            t = fx.make_rmem_tensor(n, fx.Int32)
            t.store(v)
            return t

        def _wmma(sel_n, sel_m):  # opsel: which half of its scale VGPR (ROW0 / ROW1) an operand's 16 rows use
            f8 = fx.Float8E4M3FN
            op = fx.rocdl.WMMAScale(16, 16, WMMA_K, f8, f8, fx.Float32, opsel_a=sel_n, opsel_b=sel_m)
            return fx.make_mma_atom(op)

        wmma = [[_wmma(sel_n, sel_m) for sel_m in range_constexpr(2)] for sel_n in range_constexpr(2)]
        c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_acc)]

        def _mma(wm, wn, act, wt, sa_k, sb_k):
            # WMMA "A" = the B fragment (N rows), "B" = the A fragment: each lane's 8 results are 8 consecutive N
            acc = c_frags[wm * n_rep + wn]
            fx.gemm(wmma[wn % 2][wm % 2], acc, wt, act, acc, scale_a=sb_k[wn // 2], scale_b=sa_k[wm // 2])

        seed_a = [fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(half_m)]
        seed_b = [fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(half_n)]
        seed_sa = [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(N_SA)]
        seed_sb = [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(N_SB)]

        def _chain(d, pos, t):
            d.setdefault(pos, []).append(t)

        # epilogue store geometry: lane tid stores 16-byte chunk tid % CH of rows tid // CH + i * RPI
        EPC = 16 // OB  # output elements per chunk
        CH = tile_n // EPC
        RPI = BLOCK // CH
        c16, r0 = tid % CH, tid // CH
        GRP = 65536 // (RPI * C_ROW_B)  # iterations per LDS base: displacements stay 16-bit DS offsets
        wave_id = [as_ir_value(rocdl.wave_id())]
        par = _llvm.inline_asm(T.i32, wave_id, "s_and_b32 $0, $1, 1", "=s,s,~{scc}", has_side_effects=True)
        wave_parity = fx.Int32(par)

        def _c_geo(dec):
            """(m_ext, n_ext, c_addr) of output tile ``dec``: rows / columns past M / N are dropped by the stores
            (and not read by the old-C loads)."""
            _, out_row0, m_valid, blk_n, n_valid, expert, _, _, _ = dec
            m_ext = (m_valid < tile_m).select(m_valid, fx.Int32(tile_m))
            if const_expr(mode == GROUPED_M):  # rows past m_out: a bad offset table must not store there
                rows_left = i32_m - out_row0
                m_ext = (m_ext < rows_left).select(m_ext, rows_left)
            m_ext = (m_ext > zero).select(m_ext, zero)
            n_ext = (n_valid < tile_n).select(n_valid, fx.Int32(tile_n))
            c_addr = (
                fx.Int64(fx.ptrtoint(arg_c)) + (fx.Int64(out_row0) * fx.Int64(i32_ldc) + fx.Int64(blk_n)) * OB
            )
            if const_expr(mode == VARIABLE_K):  # out [G, M, ldc]
                c_addr = c_addr + fx.Int64(expert) * fx.Int64(i32_m) * fx.Int64(i32_ldc) * OB
            return m_ext, n_ext, c_addr

        def _load_c(lds, addr, rows, ext):
            """One TDM load of ``rows`` bf16 rows of the old C at ``addr`` (``ext`` of them inside out) into the LDS
            at ``lds``, padded to C_ROW_B like the staging."""
            row_b = tile_n * 2
            g0 = Vec.from_elements([fx.Int32(1), lds, *_addr_words(addr)], fx.Int32)
            g1 = Vec.from_elements(
                _tdm_group1(0, row_b, rows, fx.Int32(row_b), ext, i32_ldc * 2, C_PAD_B), fx.Int32
            )
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(as_ir_value(g0), as_ir_value(g1)))

        def _c_pre_load(dec):
            """c_pre (wave 3): the first ``c_pre`` rows of tile ``dec``'s old C into their own LDS."""
            m_ext, _, c_addr = _c_geo(dec)
            _load_c(lds0 + PRE_BASE, c_addr, c_pre, m_ext)

        def _tile(dec, rot, nxt=None):
            """One output tile ``dec``: mainloop + epilogue. Persistent (``nxt`` = (next tile, whether there is
            one)): this tile's first two slots were loaded already (physical slot = logical ^ ``rot``), and the
            epilogue loads the next tile's while it stores; returns the next tile's ``rot``."""
            klen, supers = dec[7], dec[8]
            _prepare, t_step, last_slot = _make_prepare(_desc(dec), supers, rot)
            addr_a, addr_b, addr_sa, addr_sb = _stage_addrs(rot)
            for cf in c_frags:
                cf.store(Vec.filled(8, 0.0, fx.Float32))

            def _seed_thunks(stage, parity=0):
                """Producers of the next stage's scales and first-half fragments: (early, tail)."""

                def _scale(seed, addr, i):  # one scale VGPR: the E8M0 bytes of 32 rows x 128 K, 4 per lane
                    def load():
                        seed[i].store(Vec.from_elements([lds_load_b32(addr, i * SUPER_K)[0]], fx.Int32))

                    return load

                head = [_scale(seed_sa, addr_sa[stage], i) for i in range_constexpr(N_SA)]
                head += [_scale(seed_sb, addr_sb[stage], i) for i in range_constexpr(N_SB)]
                a_th = [
                    (lambda i=i: seed_a[i].store(_load_frag(addr_a[stage], i)))
                    for i in range_constexpr(half_m)
                ]
                b_th = [
                    (lambda i=i: seed_b[i].store(_load_frag(addr_b[stage], i, GEO_B)))
                    for i in range_constexpr(half_n)
                ]
                return (head + a_th, b_th) if parity == 0 else (head + b_th, a_th)

            def _stage(stage, future_delta, fence_out, has_next, parity=0, more=None):
                """One 128-wide K step: four quadrants of the wave tile, at most three fragment groups live; the
                second-half fragments of this step and the first half of the next are read in between the WMMAs.
                At a slot's last step the slot is refilled with the K offset ``future_delta`` (None: not).
                ``more`` (drain): whether a later step runs, else its slot fence and barrier are skipped."""
                next_stage, future_slot, boundary = (
                    (stage + 1) % UNROLL,
                    stage // KPAIR,
                    stage % KPAIR == KPAIR - 1,
                )
                sa_k = [seed_sa[i].load()[0] for i in range_constexpr(N_SA)]
                sb_k = [seed_sb[i].load()[0] for i in range_constexpr(N_SB)]
                a0, b0 = seed_a, seed_b
                rocdl.sched_barrier(0)
                refill = boundary and future_delta is not None
                prepared = _prepare(future_slot, future_delta) if const_expr(refill) else None
                n_slots = half_m * half_n
                nxt = {}

                def _quad(wm0, wn0, act, wt, produce, n_fast, pre=None):
                    if const_expr(pre is not None):
                        pre()
                    last = max(produce) if produce else -1
                    for pos in range_constexpr(n_slots):
                        i, j = (
                            (pos // len(wt), pos % len(wt)) if n_fast else (pos % len(act), pos // len(act))
                        )
                        _mma(wm0 + i, wn0 + j, act[i], wt[j], sa_k, sb_k)
                        if const_expr(pos in produce):
                            for thunk in produce[pos]:
                                thunk()
                        if const_expr(pos == last):
                            for _ in range_constexpr(2 * len(produce)):
                                rocdl.sched_mfma(1)
                                rocdl.sched_dsrd(DS_PER_FRAG // 2)
                            rocdl.sched_barrier(0)

                def _mk(kind, key):
                    n = half_m if kind == "a" else half_n
                    addr, geo = (addr_a[stage], GEO_A) if kind == "a" else (addr_b[stage], GEO_B)
                    nxt[key] = [None] * n

                    def _go(i):
                        nxt[key][i] = _rmem(16, _load_frag(addr, n + i, geo))

                    return {i: [lambda i=i: _go(i)] for i in range_constexpr(n)}

                def _sig():
                    rocdl.sched_barrier(0)
                    rocdl.s_wait_dscnt(0)
                    rocdl.sched_barrier(0)
                    if const_expr(boundary and has_next and more is None):
                        tdm_ops.tensor_wait(fence_out)
                        rocdl.s_barrier_signal(_WG_BARRIER)
                    elif const_expr(boundary and has_next):
                        if more:
                            tdm_ops.tensor_wait(fence_out)
                            rocdl.s_barrier_signal(_WG_BARRIER)
                    rocdl.sched_barrier(0)

                def _wait_refill():
                    if const_expr(boundary and has_next and more is None):
                        # (LLVM hoists the wait above the slack WMMAs)
                        rocdl.s_barrier_wait(_WG_BARRIER)
                        if const_expr(refill):
                            tdm_ops.tensor_load_2d(prepared)
                    elif const_expr(boundary and has_next):
                        if more:
                            rocdl.s_barrier_wait(_WG_BARRIER)
                    rocdl.sched_barrier(0)

                slack = max(1, n_slots // 4)
                early, tail = _seed_thunks(next_stage, parity) if const_expr(has_next) else ([], [])
                q2, q3 = {slack - 1: [_wait_refill]}, {}
                for i, t in enumerate(early):  # Q2 after the slack, then the front of Q3 ...
                    p = slack + i
                    if p < n_slots:
                        _chain(q2, p, t)
                    else:
                        _chain(q3, min(p - n_slots, n_slots - 1), t)
                for i, t in enumerate(tail):  # ... and the last fragment group's reuse slots at the end of Q3
                    _chain(q3, n_slots - len(tail) + i, t)

                def fence():
                    rocdl.sched_barrier(0)

                if const_expr(parity == 0):  # (A0,B0) -> (A0,B1) -> (A1,B1) -> (A1,B0)
                    _quad(0, 0, a0, b0, _mk("b", "b1"), False)
                    _quad(0, half_n, a0, nxt["b1"], _mk("a", "a1"), True, pre=fence)
                    _quad(half_m, half_n, nxt["a1"], nxt["b1"], q2, True, pre=_sig)
                    _quad(half_m, 0, nxt["a1"], b0, q3, True, pre=fence)
                else:  # (A0,B0) -> (A1,B0) -> (A1,B1) -> (A0,B1)
                    _quad(0, 0, a0, b0, _mk("a", "a1"), True)
                    _quad(half_m, 0, nxt["a1"], b0, _mk("b", "b1"), False, pre=fence)
                    _quad(half_m, half_n, nxt["a1"], nxt["b1"], q2, False, pre=_sig)
                    _quad(0, half_n, a0, nxt["b1"], q3, False, pre=fence)

            # the output tile: rows / columns past M / N dropped by the stores (and not read by the old-C load)
            m_ext, n_ext, c_addr = _c_geo(dec)
            free_slot = (last_slot + (NB - 1)) % NB  # c_early: the previous K step's slot (logical)
            if const_expr(persist):
                free_slot = free_slot ^ rot

            # ---- prologue: fill every slot (past the last K step: reload it), wait for slot 0, read its seeds ----
            if const_expr(not persist):  # (persistent: loaded by the previous tile's epilogue)
                if const_expr(use_cluster):
                    cluster.cluster_barrier()
                for i in range_constexpr(NB):
                    tdm_ops.tensor_load_2d(_prepare(i, fx.Int32(i) * t_step))
            if const_expr(c_pre):  # wave 3 has this tile's early old-C rows in flight after its slots
                if wave == 3:
                    tdm_ops.tensor_wait(NB)
                else:
                    tdm_ops.tensor_wait(NB - 1)
            else:
                tdm_ops.tensor_wait(NB - 1)
            gpu.barrier()
            for group in _seed_thunks(0):
                for thunk in group:
                    thunk()

            n_full = (supers + (NB - 1)) // NB - 1
            drain = supers - n_full * NB  # 1..NB slots left after the steady revolutions
            slot_delta = [fx.Int32(c + NB) * t_step for c in range_constexpr(NB)]
            n_drain = drain * KPAIR  # K steps of the drain (a half last step skips its second one)
            if const_expr(HALF_TAIL):
                n_drain = n_drain - (klen - (supers - 1) * SUPER_K <= SUPER_K // 2).select(fx.Int32(1), zero)

            def _run_steady(parity):
                for rev in range(n_full):
                    rev_delta = rev * NB * t_step
                    for g in range_constexpr(UNROLL):
                        delta = rev_delta + slot_delta[g // KPAIR]
                        _stage(g, delta, NB - 2, True, parity)
                    if const_expr(use_cluster):
                        if (rev % CLUSTER_SYNC_REVS) == (CLUSTER_SYNC_REVS - 1):
                            cluster.cluster_barrier()

            def _c_load():
                """c_early: the old C tile (bf16 rows padded to C_ROW_B) into the free slot (the previous K
                step's, never refilled: see ``_prepare``); waited for by the epilogue's tensor_wait(0)."""
                src, ext = c_addr, m_ext
                if const_expr(c_pre):  # (the early rows are in their own LDS)
                    src = c_addr + fx.Int64(c_pre) * fx.Int64(i32_ldc) * 2
                    ext = (m_ext > c_pre).select(m_ext - c_pre, zero)
                _load_c(lds0 + free_slot * PITCH + c_pre * C_ROW_B, src, tile_m - c_pre, ext)

            rot_n = rot
            if const_expr(persist):
                # the next tile: its first slot goes into the slot the epilogue does not stage in (c_early: the last
                # K step's, after the mainloop; else the free one, at the last K step's start), its second one into
                # the staging slot once the stores have read it
                nd, more_tiles = nxt
                last_phys = last_slot ^ rot
                rot_n = last_phys if const_expr(c_early) else last_phys ^ 1
                prep_n, t_step_n, _ = _make_prepare(_desc(nd), nd[8], rot_n)

            # odd waves run the quadrants B-first: the two wave pairs read different fragment halves at a time
            if wave_parity == 0:
                _run_steady(0)
            else:
                _run_steady(1)
            rocdl.s_wait_dscnt(0)
            rocdl.sched_barrier(0)

            def _more(g):  # (drain) whether a K step runs after step g
                return (fx.Int32(g) < n_drain - 1) if const_expr(c_early or persist) else None

            for g in range_constexpr(UNROLL):
                if g < n_drain:
                    if const_expr((c_early or persist) and g % KPAIR == 0):
                        # the last K step begins: the free slot is free
                        if fx.Int32(g) == (drain - 1) * KPAIR:
                            if const_expr(use_cluster):
                                cluster.cluster_barrier()  # (in every partner: loads multicast into it)
                            if const_expr(c_early):
                                if wave == 3:
                                    _c_load()
                            else:
                                if more_tiles:
                                    tdm_ops.tensor_load_2d(prep_n(0, zero))
                    _stage(g, None, NB - 2 - g // KPAIR, g < UNROLL - 1, more=_more(g))

            # ---- epilogue: accumulators -> rows in LDS (bf16, or fp32 for fp32 out / beta = 1 without c_early)
            # -> 16-byte buffer stores; beta = 1: + the old C (c_early: from the free slot, added in registers and
            # staged in place; else read at the store's offset), rounded once; rows / columns past M / N dropped
            accs = [c_frags[i].load() for i in range_constexpr(n_acc)]
            rocdl.s_wait_dscnt(0)
            tdm_ops.tensor_wait(0)
            if const_expr(use_cluster):
                cluster.cluster_barrier()  # no multicast into this LDS is still in flight
            else:
                gpu.barrier()
            stg_off = zero
            if const_expr(c_early):  # staged in place over the old C
                stg_off = free_slot * PITCH
            if const_expr(persist):
                if const_expr(c_early):
                    if more_tiles:
                        tdm_ops.tensor_load_2d(prep_n(0, zero))
                else:
                    stg_off = last_phys * PITCH
            # c_pre: the early rows (wave row 0's first fragments) are read and staged in place in their own LDS
            pre_sel = (wave_m == 0).select(fx.Int32(PRE_BASE) - stg_off, zero) if const_expr(c_pre) else zero
            for wm in range_constexpr(m_rep):
                sel = pre_sel if const_expr(wm * 16 < c_pre) else zero
                for wn in range_constexpr(n_rep):
                    h = accs[wm * n_rep + wn]
                    row, col = wmb + wm * 16 + lane16, wnb + wn * 16 + kgrp * 8
                    if const_expr(c_early):
                        at = lds0 + stg_off + sel + (wmb + lane16) * C_ROW_B + (wnb + kgrp * 8) * 2
                        old = Vec(lds_load_b128(at, wm * 16 * C_ROW_B + wn * 32))
                        h = h + old.bitcast(fx.BFloat16).to(fx.Float32)
                    h = h if const_expr(STG == 4) else h.to(fx.BFloat16)
                    fx.ptr_store(h.bitcast(fx.Int8), base_ptr + stg_off + sel + row * C_ROW_B + col * STG)
            gpu.barrier()
            ldc_b = i32_ldc * OB
            c_res = create_buffer_resource_from_addr(c_addr, num_records_bytes=m_ext * ldc_b)
            # one column per lane, so the column mask is folded into the lane's base (past num_records, and every
            # later row of it too: dropped)
            lane_b = (c16 * EPC < n_ext).select(r0 * ldc_b + c16 * 16, m_ext * ldc_b)
            step_b = RPI * ldc_b
            NPRE = c_pre // RPI  # store iterations of the early rows (read from their own LDS)
            for i in range_constexpr(tile_m // RPI):
                ig = i if i < NPRE else i - NPRE
                if const_expr(ig % GRP == 0):
                    base = (PRE_BASE, zero) if const_expr(i < NPRE) else (0, stg_off)
                    rd = _planar(base[0] + i * RPI * C_ROW_B, base[1] + r0 * C_ROW_B + c16 * EPC * STG)
                off = lane_b + step_b * i
                if const_expr(STG == 2):
                    v = lds_load_b128(rd, (ig % GRP) * RPI * C_ROW_B)
                else:
                    acc_v = [
                        Vec(lds_load_b128(rd, (ig % GRP) * RPI * C_ROW_B + 16 * j))
                        for j in range_constexpr(STG // OB)
                    ]
                    acc_v = acc_v[0].shuffle(acc_v[1], list(range(8))) if const_expr(OB == 2) else acc_v[0]
                    acc_v = acc_v.bitcast(fx.Float32)
                    if const_expr(beta1):
                        old = Vec(buffer_load(c_res, off // 4, vec_width=4, dtype=fx.Int32))
                        acc_v = acc_v + (
                            old.bitcast(fx.BFloat16).to(fx.Float32)
                            if const_expr(OB == 2)
                            else old.bitcast(fx.Float32)
                        )
                    v = (acc_v.to(fx.BFloat16) if const_expr(OB == 2) else acc_v).bitcast(fx.Int32)
                buffer_store(v, c_res, off, offset_is_bytes=True)
            if const_expr(persist):
                rocdl.s_wait_dscnt(0)
                # the staging slot is read (in every workgroup of the cluster: the loads multicast into it)
                if const_expr(use_cluster):
                    cluster.cluster_barrier()
                else:
                    gpu.barrier()
                if more_tiles:
                    tdm_ops.tensor_load_2d(prep_n(1, t_step_n))
                    if const_expr(c_pre):  # (the stores have read this tile's early rows)
                        if wave == 3:
                            _c_pre_load(nd)
            return rot_n

        if const_expr(persist):
            # persistent: cluster c (a workgroup without one) runs cluster tiles c, c + NC, ...; its workgroup
            # (lx, ly) takes tile (xg * cluster_m + lx, yg * cluster_n + ly) of cluster tile (xg, yg)
            c_id, n_cl = fx.Int32(bid_x) // cluster_m, fx.Int32(fx.grid_dim.x) // cluster_m
            lxp, lyp = fx.Int32(bid_x) % cluster_m, fx.Int32(bid_y)
            # cluster tiles along M and N (the entry's cluster divides the tile counts)
            mtg = (i32_m + (tile_m - 1)) // tile_m // cluster_m
            ntg = (i32_n + (tile_n - 1)) // tile_n // cluster_n
            if const_expr(mode == VARIABLE_K):  # m tile fastest, then n tile, then expert
                n_tiles = mtg * ntg * G

                def _decode_lin(tau):
                    xg, yg = tau % mtg, (tau // mtg) % ntg
                    return _decode_vk(xg * cluster_m + lxp, yg * cluster_n + lyp, tau // (mtg * ntg))
            elif const_expr(mode == DENSE):
                # chunks of 32 m tiles x all n groups (the grid's order), the last short
                n_tiles = mtg * ntg
                RX = max(1, 32 // cluster_m)
                full = mtg // RX * RX
                rem = mtg - full
                rem = (rem > zero).select(rem, fx.Int32(1))
                supers_d = (i32_k + (SUPER_K - 1)) // SUPER_K

                def _decode_lin(tau):
                    head = tau < full * ntg
                    tail = tau - full * ntg
                    xg = head.select(tau // (RX * ntg) * RX + tau % RX, full + tail % rem)
                    yg = head.select((tau // RX) % ntg, tail // rem)
                    blk_m, blk_n = (xg * cluster_m + lxp) * tile_m, (yg * cluster_n + lyp) * tile_n
                    return blk_m, blk_m, i32_m - blk_m, blk_n, i32_n - blk_n, zero, zero, i32_k, supers_d
            else:  # GROUPED_M (cluster_m == 1): the same order over the experts' m tiles
                gm_tab = _gm_table()
                n_m = gm_tab[1]
                n_tiles = n_m * ntg
                full = n_m // 32 * 32
                rem = n_m - full
                rem = (rem > zero).select(rem, fx.Int32(1))

                def _decode_lin(tau):
                    head = tau < full * ntg
                    tail = tau - full * ntg
                    t = head.select(tau // (32 * ntg) * 32 + tau % 32, full + tail % rem)
                    yg = head.select((tau // 32) % ntg, tail // rem)
                    return _decode_gm(gm_tab, t, (yg * cluster_n + lyp) * tile_n)

            n_iter = (n_tiles - c_id + (n_cl - 1)) // n_cl
            _end_unless_lt(c_id, n_tiles)  # a cluster with no tile (uniform over it)
            if const_expr(use_cluster):
                cluster.cluster_barrier()  # every partner resident before the first multicast
            d0 = _decode_lin(c_id)
            prep0, t_step0, _ = _make_prepare(_desc(d0), d0[8], zero)
            for i in range_constexpr(NB):
                tdm_ops.tensor_load_2d(prep0(i, fx.Int32(i) * t_step0))
            if const_expr(c_pre):
                if wave == 3:
                    _c_pre_load(d0)
            rot = zero
            for it in range(n_iter):
                tau = c_id + it * n_cl
                more_tiles = it + 1 < n_iter
                nxt_tau = more_tiles.select(tau + n_cl, tau)
                rot = _tile(_decode_lin(tau), rot, (_decode_lin(nxt_tau), more_tiles))
        else:
            _tile(dec, 0)

    # Launch: grid.x runs up to 32 M tiles (a power of two dividing them), grid.y the N tiles, grid.z the M chunks
    # (grouped-M: the m tiles of all experts, at most ceil(M / tile_m) + G; variable-K: grid.z = the expert;
    # persistent: n_cus workgroups).
    gx = (i32_m + (tile_m - 1)) // tile_m + (G if mode == GROUPED_M else 0)
    gx = (gx + (cluster_m - 1)) // cluster_m * cluster_m
    gy = ((i32_n + (tile_n - 1)) // tile_n + (cluster_n - 1)) // cluster_n * cluster_n
    pow2 = gx & -gx
    capped = (pow2 < 32).select(pow2, fx.Int32(32))
    m_run = ((gx > 32) & (pow2 >= 8) & ((capped % cluster_m) == 0)).select(capped, gx)
    grid = (m_run, gy, gx // m_run)
    if const_expr(mode == VARIABLE_K):
        grid = (gx, gy, G)
    if const_expr(persist):  # n_cus workgroups in clusters along x; variable-K: at most one cluster per tile
        n_cl = n_cus // (cluster_m * cluster_n)
        if const_expr(mode != GROUPED_M):
            n_ct = gx // cluster_m * (gy // cluster_n) * (G if mode == VARIABLE_K else 1)
            n_cl = (n_ct < n_cl).select(n_ct, fx.Int32(n_cl))
        grid = (n_cl * cluster_m, cluster_n, 1)
    attrs = {"rocdl.cluster_dims": f"{cluster_m},{cluster_n},1"} if use_cluster else {}
    kernel_gemm_mxfp8(
        arg_c,
        arg_a,
        arg_b,
        arg_sa,
        arg_sb,
        arg_g0,
        arg_g1,
        i32_m,
        i32_n,
        i32_k,
        i32_lda,
        i32_ldb,
        i32_ldc,
        i32_ldsa,
        i32_ldsb,
        value_attrs=attrs,
    ).launch(
        grid=grid,
        block=(BLOCK, 1, 1),
        stream=stream,
        cluster=(cluster_m, cluster_n, 1) if use_cluster else None,
    )


_launch_gemm_mxfp8.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
    "amdgpu-kernarg-preload": True,
    "amdgpu-kernarg-preload-count": 16,
}

_COMPILED: dict = {}  # constexprs -> compiled launcher
# Host plans: a call signature -> (launcher, its int arguments, constexprs, the bytes of storage after b the config
# relies on, out shape), made by the full path; a later call with the same signature only checks alignment, that
# slack and that a is on the current device, and launches. The signature has every other property the full path
# checks, each tensor's device included.
_PLANS: dict = {}
_MAX_PLANS = 4096
_WHO = ("gemm_mxfp8_gfx1250", "grouped_gemm_mxfp8_gfx1250", "grouped_gemm_mxfp8_variable_k_gfx1250")
_NAMES = ("a", "b", "a_scale", "b_scale", "out", "group_offs", "group_offs_padded")


def _divisor_cluster(tiles: int, c: int) -> int:
    """The largest power of two <= ``c`` that divides ``tiles``."""
    while tiles % c:
        c //= 2
    return c


def constexprs(mode, cfg, M, N, G, k_tail, out_f32, beta1) -> tuple:
    """The constexpr tail of ``_launch_gemm_mxfp8`` (the compiled-kernel key) of a call. A dense cluster shrinks to
    what the tile counts divide (a workgroup wholly past M or N must not sit in a multicast group); a grouped-M one
    clusters along M only when not persistent (the kernel multicasts B where the m tiles share an expert). beta = 1
    into bf16 fetches the old C early (``c_early``) with kpair 2 and whole N tiles: the last K step leaves one A + B
    slot free, and it holds a C tile. A persistent config runs persistent for a bf16 out (beta = 1: with c_early)."""
    if cfg.persist and (cfg.num_buffers, cfg.kpair) != (2, 2):
        raise ValueError(f"persist=True needs num_buffers 2 and kpair 2, got {tuple(cfg)}")
    if mode == GROUPED_M:
        cm = 1 if cfg.persist else cfg.cluster_m
    else:
        cm = _divisor_cluster(-(-M // cfg.tile_m), cfg.cluster_m)
    cn = _divisor_cluster(-(-N // cfg.tile_n), cfg.cluster_n)
    c_early = bool(beta1 and not out_f32 and N % cfg.tile_n == 0 and cfg.kpair == 2)
    persist = bool(cfg.persist and not out_f32 and (c_early or not beta1))
    n_cus = get_num_cus() if persist else 0  # one workgroup per CU
    return (*cfg[:4], cm, cn, bool(k_tail), mode, G, bool(out_f32), bool(beta1), c_early, persist, n_cus)


def _slack(t: torch.Tensor) -> int:
    """Bytes of ``t``'s storage after its last element."""
    end = t.storage_offset() + sum((n - 1) * st for n, st in zip(t.shape, t.stride())) + 1
    return t.untyped_storage().nbytes() - end * t.element_size()


def _tail_safe(cfg: MXGemmConfig, K: int, b) -> tuple:
    """(config, the bytes of storage it relies on after ``b``) for a K that may end in a half step (kpair 2,
    K % 256 == 128): the step loads half rows of a and of the scales, but whole block rows of b, 2048 B past its
    last one. Without that slack: the kpair-1 config (never persistent) and None."""
    if cfg.kpair == 1 or K % (2 * WMMA_K) == 0:
        return cfg, 0
    if _slack(b) >= 16 * WMMA_K:
        return cfg, 16 * WMMA_K
    return cfg._replace(kpair=1, num_buffers=min(4, 2 * cfg.num_buffers), persist=False), None


def _sig(t):
    return t.shape, t.is_contiguous(), t.dtype, t.get_device()


def _gemm(mode, a, sa, b, sb, out_dtype, out, accumulate, config, tabs=(None, None), m_out=None):
    """The entry points' shared body: the plan of the call's signature where there is one, else every check, the
    config (``config`` or the static pick, made safe for the K tail), the compiled launcher of the call's
    constexprs and the plan for the next call. ``tabs``: the mode's two offset tables, ``m_out``: grouped-M's."""
    g0, g1 = tabs
    dev = a.get_device()
    key = (
        (mode, config, out_dtype, accumulate, m_out, dev, a.shape, a.stride(), a.dtype),
        (b.shape, b.is_contiguous(), b.dtype, b.get_device()),
        (sa.shape, sa.is_contiguous(), sa.dtype, sa.get_device()),
        (sb.shape, sb.is_contiguous(), sb.dtype, sb.get_device()),
        out is not None and (out.shape, out.stride(), out.dtype, out.get_device()),
        g0 is not None and _sig(g0),
        g1 is not None and _sig(g1),
    )
    plan = _PLANS.get(key)
    cur = torch._C._cuda_getDevice()
    if plan is not None:
        fn, ints, const, slack, shape = plan
        res = torch.empty(shape, dtype=out_dtype, device=a.device) if out is None else out
        o, pa, pb = res.data_ptr(), a.data_ptr(), b.data_ptr()
        # launched if out, a and b are 16-byte aligned, a (so every tensor: the key has their devices) is on the
        # current device and b has the storage slack the plan's config relies on (else the full path); an unused
        # offset table: any valid pointer
        if not (o | pa | pb) & 15 and dev == cur and not (slack and _slack(b) < slack):
            p0, p1 = o if g0 is None else g0.data_ptr(), o if g1 is None else g1.data_ptr()
            stream = torch._C._cuda_getCurrentRawStream(cur)  # (the raw handle: no Stream object per launch)
            fn(o, pa, pb, sa.data_ptr(), sb.data_ptr(), p0, p1, *ints, stream, *const)
            return res
    who = _WHO[mode]
    if not is_gfx1250():
        raise RuntimeError(f"{who}: runs on gfx1250 only")
    tabs = [t for t in tabs if t is not None]
    G = tabs[0].numel() - 1 if tabs else 1
    grouped = mode == GROUPED_M
    if a.dim() != 2 or b.dim() != 2 + grouped or a.shape[1] != b.shape[-1] or (grouped and b.shape[0] != G):
        raise ValueError(
            f"{who}: a [rows, K] and b [{'G, ' * grouped}N, K], got {tuple(a.shape)}, {tuple(b.shape)}"
        )
    bad = any(t.dtype != torch.int64 or not t.is_contiguous() or t.numel() != G + 1 for t in tabs)
    if bad or G < 0 or len(tabs) != (0, 2, 1)[mode]:
        raise ValueError(f"{who}: group offsets must be contiguous int64 [G + 1]")
    K, N = a.shape[1], b.shape[-2]
    M = m_out if grouped else a.shape[0]
    shape = (G, M, N) if mode == VARIABLE_K else (M, N)
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise NotImplementedError(f"{who}: out_dtype {out_dtype} (bf16 or fp32)")
    if out is None and accumulate:
        raise ValueError(f"{who}: accumulate=True needs out")
    if out is None:
        out = torch.empty(shape, dtype=out_dtype, device=a.device)
    elif out.dtype != out_dtype or tuple(out.shape) != shape:
        raise ValueError(f"{who}: out must be {out_dtype} {shape}, got {out.dtype} {tuple(out.shape)}")
    if out.numel() == 0:
        return out
    if K == 0:
        return out if accumulate else out.zero_()
    fp8 = torch.float8_e4m3fn
    if a.dtype != fp8 or b.dtype != fp8 or a.stride(1) != 1 or a.stride(0) % 16 or not b.is_contiguous():
        raise ValueError(
            f"{who}: a / b must be float8_e4m3fn, a's rows K-contiguous at a 16-byte stride, b dense"
        )
    if K % WMMA_K or N % (32 if grouped else 16):
        raise ValueError(
            f"{who}: K ({K}) must be a multiple of 128 and N ({N}) of {32 if grouped else 16} (b in 16x16 blocks"
            f"{', scales per 32 rows of an expert' * grouped})"
        )
    want = ((-(-a.shape[0] // 32), K // WMMA_K, 32, 4), (*b.shape[:-2], -(-N // 32), K // WMMA_K, 32, 4))
    if (tuple(sa.shape), tuple(sb.shape)) != want or not (sa.is_contiguous() and sb.is_contiguous()):
        raise ValueError(
            f"{who}: scales must be packed and contiguous, {want[0]} and {want[1]}, got "
            f"{tuple(sa.shape)}, {tuple(sb.shape)}"
        )
    if sa.dtype not in _E8M0 or sb.dtype not in _E8M0:
        raise ValueError(f"{who}: scales must be E8M0 bytes, got {sa.dtype}, {sb.dtype}")
    ldc = out.stride(-2)
    if out.stride(-1) != 1 or ldc % 8 or ldc < N or (mode == VARIABLE_K and out.stride(0) != M * ldc):
        raise ValueError(
            f"{who}: the out row stride (>= N) must be a multiple of 8 (16-byte output rows); variable-K: "
            f"out [G, M, ldc] with expert stride M * ldc"
        )
    off = [
        f"{n} is on {t.device}" for n, t in zip(_NAMES, (a, b, sa, sb, out, *tabs)) if t.get_device() != cur
    ]
    if off:
        raise ValueError(f"{who}: every tensor must be on the current device cuda:{cur} ({', '.join(off)})")
    if (a.data_ptr() | b.data_ptr() | out.data_ptr()) % 16:
        raise ValueError(f"{who}: a, b and out must be 16-byte aligned")
    bf16 = out.dtype == torch.bfloat16
    cfg = config or (
        pick_config(M, N, K, accumulate and bf16) if mode == DENSE else pick_grouped_config(mode, N, K, bf16)
    )
    # variable-K: any expert may have an odd count of 128 tokens
    cfg, slack = _tail_safe(cfg, WMMA_K if mode == VARIABLE_K else K, b)
    k_tail = mode == VARIABLE_K or K % (WMMA_K * cfg.kpair) != 0
    const = constexprs(mode, cfg, M, N, G, k_tail, not bf16, accumulate)
    ints = (M, N, K, a.stride(0), K, ldc, K, K)  # (b's rows and the scales' 32-row groups are K bytes apart)
    ptrs = [t.data_ptr() for t in (out, a, b, sa, sb, out if g0 is None else g0, out if g1 is None else g1)]
    fn = _COMPILED.get(const)
    if fn is not None:
        fn(*ptrs, *ints, torch._C._cuda_getCurrentRawStream(cur), *const)
    else:
        typed = [flyc.from_c_void_p(fx.Int8, p, assumed_align=16) for p in ptrs]
        # compiles and runs this call (None: compile only)
        fn = flyc.compile(_launch_gemm_mxfp8, *typed, *ints, torch.cuda.current_stream(), *const)
        if fn is None:
            return out
        _COMPILED[const] = fn
    if slack is not None:
        if len(_PLANS) >= _MAX_PLANS:
            _PLANS.clear()
        _PLANS[key] = (fn, ints, const, slack, shape)
    return out


def gemm_mxfp8_gfx1250(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
    config: MXGemmConfig | None = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """``out[M, N] (+)= (a * a_scale) @ (b * b_scale)^T`` with E4M3 ``a [M, K]`` (K contiguous, a row stride that
    is a multiple of 16 bytes), ``b [N, K]`` dense in the 16x16 order ``[N/16][K/16][16][16]`` and E8M0 scales
    (uint8 or float8_e8m0fnu) packed ``[ceil(rows/32), K/128, 32, 4]``; K a multiple of 128, N of 16. ``out`` bf16
    or fp32; ``accumulate``: ``out = round(out + fp32 result)``. ``config`` overrides the static per-shape pick."""
    return _gemm(DENSE, a, a_scale, b, b_scale, out_dtype, out, accumulate, config)


def grouped_gemm_mxfp8_gfx1250(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    group_offs: torch.Tensor,
    group_offs_padded: torch.Tensor,
    m_out: int,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
    config: MXGemmConfig | None = None,
) -> torch.Tensor:
    """Routed experts fwd / dgrad: ``out[offs[g] + i] = a[offs_pad[g] + i] @ b[g]^T`` for ``i < offs[g + 1] -
    offs[g]``. ``a [M_pad, K]`` holds each expert's rows from ``group_offs_padded[g]`` (int64 [G + 1], multiples of
    32: Turbo's rowwise-padded MX layout), ``b [G, N, K]`` (16x16 blocks per expert, N a multiple of 32), scales
    packed ``[ceil(M_pad/32), ..]`` / ``[G, N/32, ..]``, ``group_offs`` (int64 [G + 1]) the tight offsets of the
    ``m_out`` output rows. Device tables only: no host sync."""
    tabs = (group_offs, group_offs_padded)
    return _gemm(GROUPED_M, a, a_scale, b, b_scale, out_dtype, out, False, config, tabs, m_out)


def grouped_gemm_mxfp8_variable_k_gfx1250(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    group_offs: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
    config: MXGemmConfig | None = None,
) -> torch.Tensor:
    """Routed experts wgrad: ``out[g] (+)= a[:, T_g] @ b[:, T_g]^T`` over expert g's tokens
    ``T_g = [group_offs[g], group_offs[g + 1])`` (int64 [G + 1], multiples of 128: the column copies' layout).
    ``a [M, T]`` (dY^T), ``b [N, T]`` (X^T, in 16x16 blocks), scales packed ``[ceil(M/32), T/128, 32, 4]`` /
    ``[ceil(N/32), ..]``; ``out [G, M, N]`` bf16 or fp32, ``accumulate``: ``out[g] = round(out[g] + result)`` (the
    fused main_grad update; an empty expert leaves its slice unchanged, or zero without ``accumulate``)."""
    return _gemm(VARIABLE_K, a, a_scale, b, b_scale, out_dtype, out, accumulate, config, (group_offs, None))
