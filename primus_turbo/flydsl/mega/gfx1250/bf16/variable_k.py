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

"""Gathered-operand variable-K grouped GEMM for gfx1250 Mega MoE wgrads.

One operand's reduction rows come through a row-index table (TDM gather), so the
wgrad can read the dispatched token buffer in pool order. Lives under
``mega.gfx1250.bf16`` so the shared ``grouped_gemm_bf16_kernel_gfx1250`` stays free
of Mega-specific gather paths.

Shared tile helpers and the dense / non-gather variable-K path stay in
``primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250``.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _std_arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.rocdl import cluster as _cluster
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

# Shared helpers / constants from the gfx1250 grouped GEMM (non-gather).
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    _F16,
    _GFX1250_LDS_BYTES,
    _STACK_KEYS,
    _STACK_VK,
    _TDM_TH_LOAD,
    _TDM_TH_STORE,
    _WG_BARRIER,
    C_PAD_E,
    _as_ptr,
    _check_smem,
    _check_tdm_cache,
    _get_arch,
    _group_m_tile_decode,
    _i32_table,
    _pick_epi_chunk,
    _pick_lds_cosched,
    _pick_stack,
    _pick_wmma_b2b,
    _require_gfx1250,
    _stack_off,
    _vk_auto_split_bar,
    _vk_lds_pad,
    _vk_stack_cluster_ok,
    _vk_wave_grid,
    _xcd_band_remap,
    build_expert_table,
    check_prebuilt,
)
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    _pick_variable_k_config as _v3_pick_variable_k_config,
)

# ---------------------------------------------------------------------------
# Gathered operand (variable-K)
# ---------------------------------------------------------------------------
# One operand's reduction rows can come through a row-index table (reduction row -> source
# row) as TDM gather loads, 16 rows per instruction with 16-bit indices or 8 with 32-bit
# ones, so a caller whose rows live in another order (Mega MoE's pool order over the
# dispatch buffer) need not materialize them. Built by hand rather than through
# tdm_ops.make_tensor_gather_descriptor, which wants the LDS side as a memref and a constant
# row stride; the field layout is CDNA5 ISA 4.10.3.2, the bits that builder packs.


class _GatherDesc:
    def __init__(self, dgroup0, dgroup1, dgroup2, dgroup3):
        self.dgroup0, self.dgroup1, self.dgroup2, self.dgroup3 = dgroup0, dgroup1, dgroup2, dgroup3


def _readlane_i32(v, lane: int):
    """Lane ``lane`` of a per-lane i32, as a wave-uniform SGPR value."""
    op = rocdl.readlane(res=T.i32, src=_raw(v), lane=_raw(fx.Int32(lane)))
    return ArithValue(op.result if hasattr(op, "result") else op)


def _vk_gather_desc(
    lds_addr, src_addr, rows, idx_bits, row_width, num_src_rows, row_stride, lds_pad, mask=None, early=False
):
    """One gather load of ``rows`` (wave-uniform i32 indices < ``num_src_rows``), each
    ``row_width`` bf16 elements from ``src_addr`` (i64 bytes) with ``row_stride`` (i32
    elements), into LDS rows ``row_width * 2 + lds_pad`` bytes apart from ``lds_addr`` (i32 bytes).
    ``mask`` (wave-uniform i32, cluster-relative workgroup bits) multicasts it to those peers."""
    eb = 2
    i32 = ir.IntegerType.get_signless(32)

    def c32(v):
        return fx.Int32(v).ir_value()

    interval_dw, amount_dw = row_width * eb // 4, lds_pad // 4
    assert interval_dw & (interval_dw - 1) == 0, "TDM pad interval must be a power of two dwords"
    g = _raw(src_addr)
    lo = _std_arith.TruncIOp(i32, g).result
    hi = _std_arith.TruncIOp(i32, _std_arith.ShRUIOp(g, _raw(fx.Int64(32))).result).result
    hi = _std_arith.OrIOp(hi, c32(1 << 31)).result
    pred = c32(1 | ((1 if idx_bits == 32 else 0) << 30) | (1 << 31))
    dgroup0 = _vector.from_elements(T.vec(4, T.i32), [pred, _raw(lds_addr), lo, hi])
    s0 = (eb.bit_length() - 1) << 16 | 1 << 20 | (interval_dw.bit_length() - 2) << 22 | (amount_dw - 1) << 25
    if mask is not None and early:
        s0 |= 1 << 21
    words = [
        s0,
        (row_width & 0xFFFF) << 16,
        (row_width >> 16) & 0xFFFF | (num_src_rows & 0xFFFF) << 16,
        (num_src_rows >> 16) & 0xFFFF | row_width << 16,
        len(rows),
    ]
    dgroup1 = _vector.from_elements(
        T.vec(8, T.i32),
        [_raw(fx.Int32(words[0]) | mask) if mask is not None else c32(words[0])]
        + [c32(w) for w in words[1:]]
        + [_raw(fx.Int32(row_stride)), c32(0), c32(0)],
    )
    if idx_bits == 32:
        packed = [_raw(r) for r in rows]
    else:
        # indices are < num_src_rows < 2^16, so the high half needs no mask
        packed = [_raw(rows[2 * w] | (rows[2 * w + 1] << fx.Int32(16))) for w in range(len(rows) // 2)]
    packed += [c32(0)] * (8 - len(packed))
    dgroup2 = _vector.from_elements(T.vec(4, T.i32), packed[:4])
    dgroup3 = _vector.from_elements(T.vec(4, T.i32), packed[4:])
    return _GatherDesc(dgroup0, dgroup1, dgroup2, dgroup3)


@flyc.jit
def _launch_grouped_bf16_variable_k(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_etab: fx.Pointer,
    arg_ridx: fx.Pointer,
    stream: fx.Stream,
    i32_n: fx.Int32,
    i32_k: fx.Int32,
    G: Constexpr[int],
    N_BLOCKS_N: Constexpr[int],
    N_BLOCKS_K: Constexpr[int],
    C_GRP_ELEMS: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    num_buffers: Constexpr[int],
    is_f16: Constexpr[int] = 0,
    group_m: Constexpr[int] = 16,
    num_xcd: Constexpr[int] = 1,
    xcd_band: Constexpr[int] = 32,
    wmma_b2b: Constexpr[int] = 0,
    frag_pipeline: Constexpr[int] = 1,
    split_bar: Constexpr[int] = 0,
    k_tiles_const: Constexpr[int] = 0,
    # `c_off` is `g_idx * C_GRP_ELEMS + blk_n * ldc + blk_k`, so the stride and the per-expert
    # span must move TOGETHER or the store walks off the allocation; the entry point derives
    # `C_GRP_ELEMS` from this value and checks `out` against it.
    out_ldc: Constexpr[int] = 0,
    tdm_load_cache: Constexpr[int] = 0,
    tdm_store_cache: Constexpr[int] = 0,
    lds_pad: Constexpr[int] = 16,
    wave_grid: Constexpr[int] = 0,
    # Persistent workgroups.  0 = one workgroup per output tile, grid = TOTAL_TILES. > 0
    # launches exactly `persist` workgroups, each walking `TOTAL_TILES // persist` tiles; it
    # must DIVIDE TOTAL_TILES so the trip count stays a compile-time constant, and leave at
    # least 2 tiles per workgroup.
    persist: Constexpr[int] = 0,
    issue_first: Constexpr[int] = 0,
    # Bit-identical output.  n must divide ``wmma_m_rep`` so a chunk boundary never cuts a WMMA
    # fragment. A band's dim-0 extent goes NEGATIVE on a short tail or dead tile and the
    # descriptor field is unsigned -- a negative value is the FULL window -- so every extent is
    # clamped.
    epi_chunk: Constexpr[int] = 1,
    # 1: the stage-closing LDS wait of a deferred last sub-step is emitted after the k-tile's
    #    closing fence, so the scheduler cannot hoist it above the in-body sub-steps' matrix ops.
    # 2: 1, and each sub-step's fragment reads are issued one per matrix op at the head of the
    #    previous sub-step's run, one scheduling region per sub-step, with the running sub-step's
    #    fragments held live to the end of its run (see the NT launcher's `lds_cosched`).
    lds_cosched: Constexpr[int] = 0,
    # Workgroup cluster of mc_m x mc_n output tiles of ONE expert (1-D grid, cluster = mc_m*mc_n
    # consecutive workgroup ids) with TDM multicast: peers on one OUT_M block share the A block,
    # peers on one OUT_N block share the B block. All tiles of an expert have the same K range, so
    # peers agree on the k-tile count and on every load's extent. 1 x 1 = off.
    mc_m: Constexpr[int] = 1,
    mc_n: Constexpr[int] = 1,
    # early_timeout bit of the multicast loads.
    mc_early: Constexpr[int] = 1,
    # Cluster barrier every mc_sync steady k-tiles (0 = none).
    mc_sync: Constexpr[int] = 0,
    # 1 = cluster barrier at the start of every tile.
    mc_tile_sync: Constexpr[int] = 0,
    # 1 = cluster barrier after the last k-tile's fence, before the epilogue restages C.
    mc_epi_sync: Constexpr[int] = 0,
    # > 0 (needs lds_cosched=2 and split_bar): the next k-tile's sub-step-0 fragments are read
    # inside this k-tile's barrier window, after the first `head_pipe` deferred matrix ops and the
    # barrier wait, and carried into the next k-tile; the steady loop runs whole ring rotations so
    # every slot address is a loop-invariant register.
    head_pipe: Constexpr[int] = 0,
    # 1: with head_pipe, each run's fragment reads are issued as one burst at its head instead of
    #    one per matrix op, and the previous k-tile's deferred fragments are held live across it.
    hp_burst: Constexpr[int] = 0,
    # 1: with head_pipe, the rotation's k-tile index is made opaque to the loop optimizer, so the
    #    unrolled copies recompute their TDM addresses instead of each carrying strength-reduced
    #    64-bit induction variables, which spill the scalar file.
    hp_opaque: Constexpr[int] = 0,
    # bit 0: every wave issues one TDM load per k-tile -- waves [0, W/2) a row band of A, the others
    #        the same band of B -- with a wave-uniform descriptor built from per-wave constants, so
    #        the issue block has no branch. Same LDS image as the two-wave issue (needs tile_m ==
    #        tile_n, so both operands share the row pitch and the padding).
    # bit 1: with head_pipe, the issue block moves from the head of the k-tile's first run to behind
    #        that run's fragment reads, with its scalar ops spread between the matrix ops.
    tdm_deal: Constexpr[int] = 0,
    # Persistent tile seam (needs persist and two ring slots past the C staging).
    # bit 0: the epilogue prefetches the next tile's k-tiles 0 AND 1 (both slots clear of the staging).
    # bit 1 (needs bit 0, head_pipe, num_buffers=4): no store wait and seam barrier at the end of the
    #        epilogue. The next tile waits for its k-tile 0 only (TENSORcnt retires in order, and k-tiles 0
    #        and 1 were issued ahead of the stores), runs k-tile 0, then waits for everything -- this
    #        wave's share of the store included -- and a barrier before it issues k-tiles 2 and 3, the
    #        first ones that land on the staging. K_TILES is clamped to >= 4 so that k-tile is a steady one.
    # bit 2: the next tile's decode (expert table reads included) is done at the top of the tile.
    vk_ovl: Constexpr[int] = 0,
    # 0 = dense. 1 / 2: A / B reduction row m is row `arg_ridx[m]` of a `gather_rows`-row
    # tensor, loaded with TDM gather ops.
    gather: Constexpr[int] = 0,
    gather_rows: Constexpr[int] = 0,
):
    WMMA_M = WMMA_N = 16
    WMMA_K = 32
    WAVE = 32
    EB = 2
    K_WS = tile_k // WMMA_K
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    TILES_PER_EXPERT = N_BLOCKS_N * N_BLOCKS_K
    TOTAL_TILES = G * TILES_PER_EXPERT
    PERSIST = persist if persist else TOTAL_TILES
    TILES_PER_WG = TOTAL_TILES // PERSIST
    if persist:
        if TOTAL_TILES % persist:
            raise ValueError(f"persist={persist} must divide TOTAL_TILES={TOTAL_TILES}")
        if TILES_PER_WG < 2:
            raise ValueError(
                f"persist={persist} leaves {TILES_PER_WG} tile(s) per workgroup; with one "
                f"tile 'the accumulator was not cleared between tiles' is unobservable"
            )
    if epi_chunk < 1 or wmma_m_rep % epi_chunk:
        raise ValueError(f"epi_chunk must divide wmma_m_rep={wmma_m_rep}, got {epi_chunk}")
    NB1 = num_buffers - 1
    MC = mc_m * mc_n
    if mc_m not in (1, 2, 4) or mc_n not in (1, 2, 4):
        raise ValueError(f"mc_m and mc_n must be 1, 2 or 4, got {mc_m} x {mc_n}")
    if MC > 1:
        # The tile walk runs over whole clusters inside one expert, so a peer never sits in
        # another expert (whose K range differs) and every peer walks the same trip count.
        if N_BLOCKS_N % mc_m or N_BLOCKS_K % mc_n:
            raise ValueError(
                f"a {mc_m} x {mc_n} cluster needs whole clusters per expert: "
                f"N_BLOCKS_N={N_BLOCKS_N}, N_BLOCKS_K={N_BLOCKS_K}"
            )
        if PERSIST % MC:
            raise ValueError(f"the grid ({PERSIST}) must be a whole number of {MC}-workgroup clusters")
        if group_m % mc_m or (num_xcd > 1 and xcd_band % MC):
            raise ValueError(f"group_m={group_m} / xcd_band={xcd_band} must be multiples of the cluster")
    if mc_sync < 0:
        raise ValueError(f"mc_sync must be >= 0, got {mc_sync}")
    C_TOTAL = TOTAL_TILES // MC
    C_PER_EXPERT = TILES_PER_EXPERT // MC
    C_STEP = PERSIST // MC
    # Peer patterns at lx = ly = 0: A is shared down a column of the cluster (stride mc_m), B
    # along a row.
    MC_A_PAT = (
        1 | ((1 << mc_m) if mc_n > 1 else 0) | (((1 << (2 * mc_m)) | (1 << (3 * mc_m))) if mc_n > 2 else 0)
    )
    MC_B_PAT = (1 << mc_m) - 1

    if tile_k % WMMA_K or warp_tile_m % WMMA_M or warp_tile_n % WMMA_N:
        raise ValueError(f"tile ({tile_m},{tile_n},{tile_k}) / warps ({m_warp},{n_warp}) not WMMA-aligned")
    if K_WS < 2:
        raise ValueError(f"tile_k={tile_k} needs at least 2 WMMA K-steps")
    if num_waves < 2:
        raise ValueError("need >= 2 waves so the two TDM jobs land on different waves")
    if num_buffers < 2:
        raise ValueError("num_buffers must be >= 2")
    if split_bar not in (0, 1):
        raise ValueError(
            f"split_bar must be 0 or 1 by the time it reaches the launcher, got {split_bar}; "
            f"-1 (auto) is resolved by the entry point via _auto_split_bar()"
        )
    if split_bar and frag_pipeline == 2:
        raise ValueError("split_bar is not supported with frag_pipeline=2")
    if lds_cosched not in (0, 1, 2):
        raise ValueError(f"lds_cosched must be 0, 1 or 2 on the variable-K path, got {lds_cosched}")
    if lds_cosched and frag_pipeline != 1:
        raise ValueError(f"lds_cosched={lds_cosched} needs frag_pipeline=1, got {frag_pipeline}")
    if lds_cosched >= 2 and 2 * (warp_tile_m // WMMA_M + warp_tile_n // WMMA_N) >= n_acc:
        raise ValueError(f"lds_cosched={lds_cosched} needs fewer fragment reads per sub-step than matrix ops")
    if head_pipe and (lds_cosched != 2 or not split_bar or issue_first or k_tiles_const):
        raise ValueError(
            f"head_pipe={head_pipe} needs lds_cosched=2, split_bar=1, issue_first=0 and k_tiles_const=0"
        )
    if head_pipe < 0 or head_pipe + 2 * (warp_tile_m // WMMA_M + warp_tile_n // WMMA_N) > n_acc:
        raise ValueError(f"head_pipe={head_pipe} leaves too few deferred matrix ops for the head reads")
    if (hp_burst or hp_opaque) and not head_pipe:
        raise ValueError("hp_burst / hp_opaque need head_pipe")
    if tdm_deal not in (0, 1, 3):
        raise ValueError(f"tdm_deal must be 0, 1 or 3 (bit 1 places the branch-free issue), got {tdm_deal}")
    if tdm_deal and (num_waves % 2 or tile_k % (num_waves // 2) or tile_m != tile_n):
        raise ValueError(
            f"tdm_deal needs an even wave count whose half divides tile_k and tile_m == tile_n, got "
            f"{num_waves} waves, tile ({tile_m},{tile_n},{tile_k})"
        )
    if (tdm_deal & 2) and (not head_pipe or hp_burst):
        raise ValueError("tdm_deal bit 1 needs head_pipe with hp_burst=0")
    _check_tdm_cache(tdm_load_cache, _TDM_TH_LOAD, "tdm_load_cache")
    _check_tdm_cache(tdm_store_cache, _TDM_TH_STORE, "tdm_store_cache")

    if lds_pad <= 0 or lds_pad % 16:
        raise ValueError(f"lds_pad must be a positive multiple of 16, got {lds_pad}")
    LDS_PAD = lds_pad
    LDS_A_ROW = tile_m * EB + LDS_PAD
    LDS_B_ROW = tile_n * EB + LDS_PAD
    STAGE_A = ((tile_k * LDS_A_ROW + 15) // 16) * 16
    STAGE_B = ((tile_k * LDS_B_ROW + 15) // 16) * 16
    PITCH = ((STAGE_A + STAGE_B + 1023) // 1024) * 1024
    elem_cls = fx.Float16 if is_f16 else fx.BFloat16
    C_ROW_E = tile_n + C_PAD_E
    C_STORE_B = ((tile_m * C_ROW_E * EB + 127) // 128) * 128
    ARENA_B = max(num_buffers * PITCH, C_STORE_B)
    # The next tile's k-tile 0 is prefetched into the first buffer past the C staging, which covers
    # C_PITCHES buffers from the arena base.
    C_PITCHES = -(-C_STORE_B // PITCH)
    PREFETCH_OK = bool(persist) and C_PITCHES < num_buffers
    if ARENA_B > _GFX1250_LDS_BYTES:
        raise ValueError(f"LDS arena {ARENA_B} B exceeds the gfx1250 budget {_GFX1250_LDS_BYTES} B")
    _check_smem(ARENA_B, str(_get_arch()))
    if vk_ovl < 0 or vk_ovl > 7 or vk_ovl == 2 or vk_ovl == 6:
        raise ValueError(f"vk_ovl must be a bitfield of 1, 2 (needs 1) and 4, got {vk_ovl}")
    if vk_ovl and not (PREFETCH_OK and C_PITCHES + 2 <= num_buffers):
        raise ValueError(
            f"vk_ovl={vk_ovl} needs persist and two ring slots past the C staging "
            f"(C_PITCHES={C_PITCHES}, num_buffers={num_buffers})"
        )
    if (vk_ovl & 2) and (not head_pipe or num_buffers != 4 or C_PITCHES != 2 or MC > 1 and mc_sync):
        raise ValueError("vk_ovl bit 1 needs head_pipe, num_buffers=4, C_PITCHES=2 and mc_sync=0")
    # k-tiles the epilogue prefetches for the next tile, and TDM ops one wave issues for the C store.
    PRE_KT = 2 if vk_ovl else 1
    EPI_OPS = (epi_chunk * m_warp) if epi_chunk > 1 else 1
    if gather not in (0, 1, 2):
        raise ValueError(f"gather must be 0 (none), 1 (A) or 2 (B), got {gather}")
    IDX_BITS = 16 if gather_rows < 0xFFFF else 32
    GATHER_ROWS_PER_OP = 16 if IDX_BITS == 16 else 8
    # Gathered, every wave issues one row band of the k-tile: the dense operand's band as one
    # TDM load and the gathered operand's as GATHER_OPS gather loads, so the per-wave op count
    # the fences are written against is the same on every wave.
    G_BAND = tile_k // num_waves
    GATHER_OPS = G_BAND // GATHER_ROWS_PER_OP
    if gather and (tile_k % num_waves or G_BAND % GATHER_ROWS_PER_OP or G_BAND > WAVE):
        raise ValueError(
            f"gather splits tile_k={tile_k} rows over {num_waves} waves into whole ops of "
            f"{GATHER_ROWS_PER_OP}, at most one row per lane"
        )

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel_grouped_variable_k(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_etab: fx.Pointer,
        arg_ridx: fx.Pointer,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
    ):
        if const_expr(wmma_b2b):
            rocdl.disable_xdl_arb_stall()
        ldn_b = fx.Int64(i32_n) * EB
        ldk_b = fx.Int64(i32_k) * EB
        ldc64 = fx.Int64(out_ldc) if const_expr(out_ldc) else fx.Int64(i32_k)

        tid = fx.Int32(fx.thread_idx.x)
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        l8 = lane % 8
        hi8 = lane16 // 8
        wave_m = (wave % m_warp) if const_expr(wave_grid) else (wave // n_warp)
        wave_n = (wave // m_warp) if const_expr(wave_grid) else (wave % n_warp)

        pid = fx.Int32(fx.block_idx.x)
        mc_lx = mc_ly = mc_a_pat = mc_b_pat = None
        if const_expr(MC > 1):
            mc_cid = pid // fx.Int32(MC)
            mc_lid = pid - mc_cid * fx.Int32(MC)
            if const_expr(mc_m > 1):
                mc_lx = mc_lid % fx.Int32(mc_m)
            if const_expr(mc_n > 1):
                mc_ly = mc_lid // fx.Int32(mc_m)
            # The peers (self included) that issue the same A / B load; the hardware's flat id in
            # the cluster is lx + ly * mc_m. Masks are workgroup-uniform and symmetric: every peer
            # a mask names computes the same mask, so each multicast load has exactly one matching
            # load per peer.
            if const_expr(mc_n > 1):
                mc_a_pat = fx.Int32(MC_A_PAT)
                if const_expr(mc_m > 1):
                    mc_a_pat = mc_a_pat << mc_lx
            if const_expr(mc_m > 1):
                mc_b_pat = fx.Int32(MC_B_PAT)
                if const_expr(mc_n > 1):
                    mc_b_pat = mc_b_pat << (fx.Int32(mc_m) * mc_ly)

        dl_pat = None
        if const_expr(tdm_deal):
            # Per-wave descriptor constants, as integer arithmetic on the 0/1 operand index: a
            # loop-invariant compare would be carried into the loop as a lane mask.
            DL_PARTS = num_waves // 2
            DL_ROWS = tile_k // DL_PARTS
            dl_b = wave // fx.Int32(DL_PARTS)
            dl_part = wave - dl_b * fx.Int32(DL_PARTS)
            dl_rows = dl_part * fx.Int32(DL_ROWS)
            dl_b64 = fx.Int64(dl_b)
            dl_ld = ldn_b + dl_b64 * (ldk_b - ldn_b)
            dl_ga = fx.Int64(fx.ptrtoint(fx.recast_iter(fx.Int8, arg_a)))
            dl_gb = fx.Int64(fx.ptrtoint(fx.recast_iter(fx.Int8, arg_b)))
            dl_gbase = dl_ga + dl_b64 * (dl_gb - dl_ga)
            dl_lds = dl_b * fx.Int32(STAGE_A) + dl_part * fx.Int32(DL_ROWS * LDS_A_ROW)
            DL_GPTR = fx.PointerType.get(
                elem_ty=fx.Int8.ir_type, address_space=fx.AddressSpace.Global, alignment=16
            )
            if const_expr(MC > 1):
                pa_ = mc_a_pat if const_expr(mc_a_pat is not None) else fx.Int32(0)
                pb_ = mc_b_pat if const_expr(mc_b_pat is not None) else fx.Int32(0)
                dl_pat = pa_ + dl_b * (pb_ - pa_)

        def _cluster_tile(c_raw):
            ct = _xcd_band_remap(c_raw, C_TOTAL, num_xcd, xcd_band // MC)
            g = rocdl.readfirstlane(T.i32, ct // fx.Int32(C_PER_EXPERT))
            lc = ct % fx.Int32(C_PER_EXPERT)
            mp, kp = _group_m_tile_decode(lc, N_BLOCKS_N // mc_m, N_BLOCKS_K // mc_n, group_m // mc_m)
            bn = (mp * fx.Int32(mc_m) + mc_lx) if const_expr(mc_m > 1) else mp
            bk = (kp * fx.Int32(mc_n) + mc_ly) if const_expr(mc_n > 1) else kp
            return g, bn, bk

        def _one_output_tile(tile_raw):
            if const_expr(MC > 1):
                # `tile_raw` counts whole clusters.
                g_idx, blk_n_idx, blk_k_idx = _cluster_tile(tile_raw)
            else:
                tile = _xcd_band_remap(tile_raw, TOTAL_TILES, num_xcd, xcd_band)
                g_idx = rocdl.readfirstlane(T.i32, tile // fx.Int32(TILES_PER_EXPERT))
                local = tile % fx.Int32(TILES_PER_EXPERT)
                blk_n_idx, blk_k_idx = _group_m_tile_decode(local, N_BLOCKS_N, N_BLOCKS_K, group_m)

            # (m_start, m_len) per expert. m_len is the *valid* token count, which is
            # `masked_k[g]` when the caller passed one -- the padded tail between m_start+m_len
            # and the next expert's m_start is then never addressed.
            et = _i32_table(arg_etab)
            e = g_idx * fx.Int32(2)
            m_start = rocdl.readfirstlane(T.i32, et[e])
            m_len = rocdl.readfirstlane(T.i32, et[e + fx.Int32(1)])

            blk_n = blk_n_idx * tile_m
            blk_k = blk_k_idx * tile_n
            n_valid = i32_n - blk_n
            k_valid = i32_k - blk_k

            # CORRECTNESS: back the *read* descriptors off so their window never leaves the row.
            # A descriptor claims tile_m*EB bytes per row but a short last block has only
            # n_valid*EB valid, and the extents are not symmetric: a dim-0 extent bounds
            # addressing, a dim-1 extent bounds addressing on a STORE, but a dim-1 extent on a
            # LOAD clamps the DATA only -- an over-claimed load walks off the allocation and
            # faults the device.
            blk_n_eff = (blk_n > (i32_n - fx.Int32(tile_m))).select(i32_n - fx.Int32(tile_m), blk_n)
            blk_k_eff = (blk_k > (i32_k - fx.Int32(tile_n))).select(i32_k - fx.Int32(tile_n), blk_k)
            n_back = blk_n - blk_n_eff
            k_back = blk_k - blk_k_eff

            def _next_dec():
                # The index is CLAMPED rather than guarded, because `tile_raw + PERSIST` on the
                # last tile decodes an expert past the table.
                if const_expr(MC > 1):
                    # Same clamp for every peer, so the (never consumed) last prefetch still pairs.
                    nxt_raw = tile_raw + fx.Int32(C_STEP)
                    nxt_raw = (nxt_raw > fx.Int32(C_TOTAL - 1)).select(fx.Int32(C_TOTAL - 1), nxt_raw)
                    g2, bn2_idx, bk2_idx = _cluster_tile(nxt_raw)
                else:
                    nxt_raw = tile_raw + fx.Int32(PERSIST)
                    nxt_raw = (nxt_raw > fx.Int32(TOTAL_TILES - 1)).select(fx.Int32(TOTAL_TILES - 1), nxt_raw)
                    nxt = _xcd_band_remap(nxt_raw, TOTAL_TILES, num_xcd, xcd_band)
                    g2 = rocdl.readfirstlane(T.i32, nxt // fx.Int32(TILES_PER_EXPERT))
                    l2 = nxt % fx.Int32(TILES_PER_EXPERT)
                    bn2_idx, bk2_idx = _group_m_tile_decode(l2, N_BLOCKS_N, N_BLOCKS_K, group_m)
                e2 = g2 * fx.Int32(2)
                ms2 = rocdl.readfirstlane(T.i32, et[e2])
                ml2 = rocdl.readfirstlane(T.i32, et[e2 + fx.Int32(1)])
                bn2 = bn2_idx * tile_m
                bk2 = bk2_idx * tile_n
                bne2 = (bn2 > (i32_n - fx.Int32(tile_m))).select(i32_n - fx.Int32(tile_m), bn2)
                bke2 = (bk2 > (i32_k - fx.Int32(tile_n))).select(i32_k - fx.Int32(tile_n), bk2)
                return ms2, ml2, bne2, bke2

            nxt_dec = _next_dec() if const_expr(PREFETCH_OK and (vk_ovl & 4)) else None

            arena = fx.SharedAllocator(static=False)
            arena.allocate(ARENA_B)
            base_ptr = arena.base_ptr

            def _bidx(p):
                return fx.Int64(fx.ptrtoint(p))

            def _buf_ptr(s):
                return fx.add_offset(base_ptr, s * PITCH)

            # Identity when `persist == 0`; otherwise a CONSTANT +1, so every tile starts its
            # k-loop at buffer 1 and the epilogue's C staging -- always at the arena base,
            # inside buffer 0 -- is never the target of a cross-tile prefetch. It must be a
            # constant and not an alternation, or every other tile's prefetch lands in the
            # staging it is supposed to avoid.
            BUF_SHIFT = (C_PITCHES % num_buffers) if persist else 0

            def _slot(kt):
                return ((kt + BUF_SHIFT) % num_buffers) if BUF_SHIFT else (kt % num_buffers)

            def _gv(base, off, shape, stride):
                return fx.Tensor(fx.make_view(fx.add_offset(base, off), fx.make_layout(shape, stride)))

            def _lv(ptr, shape, stride):
                return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

            def _tdm(gt, rows, cols_b, stride, pad_interval, mask=None):
                atom = fx.rocdl.make_tdm_atom(
                    gt,
                    [rows, cols_b],
                    strides=[stride, None],
                    num_warps=1,
                    pad_interval=pad_interval,
                    pad_amount=LDS_PAD,
                    cache_modifier=tdm_load_cache,
                    early_timeout=True if const_expr(mask is None) else bool(mc_early),
                )
                if const_expr(mask is not None):
                    atom = fx.atom_set_value(atom, "workgroup_mask", mask)
                return atom

            def _mc_mask(pat, rem):
                # Peers share the expert, hence `rem`: an empty window is a plain load on all of them.
                return None if const_expr(pat is None) else (rem > fx.Int32(0)).select(pat, fx.Int32(0))

            gA_base = fx.recast_iter(fx.Int8, arg_a)
            gB_base = fx.recast_iter(fx.Int8, arg_b)
            gC_base = fx.recast_iter(fx.PointerType.get(elem_cls.ir_type, arg_c.address_space), arg_c)

            if const_expr(gather):
                ridx = _i32_table(arg_ridx)
                g_band = wave * fx.Int32(G_BAND)
                g_row = g_band + lane % fx.Int32(G_BAND)
                if const_expr(gather == 1):
                    g_base, g_w, g_lds_row, g_stride, g_pat = gA_base, tile_m, LDS_A_ROW, i32_n, mc_a_pat
                    d_base, d_w, d_lds_row, d_ld, d_pat = gB_base, tile_n, LDS_B_ROW, ldk_b, mc_b_pat
                else:
                    g_base, g_w, g_lds_row, g_stride, g_pat = gB_base, tile_n, LDS_B_ROW, i32_k, mc_b_pat
                    d_base, d_w, d_lds_row, d_ld, d_pat = gA_base, tile_m, LDS_A_ROW, ldn_b, mc_a_pat

                def _gather_src_rows(ms, ml, kt):
                    # rows past the expert re-read its last one, which is finite, against the
                    # dense operand's zero-filled rows
                    has_rows = ml > fx.Int32(0)
                    last_row = has_rows.select(ml - fx.Int32(1), fx.Int32(0))
                    # an empty expert's m_start can be the table's end
                    first_row = has_rows.select(ms, fx.Int32(0))
                    r = fx.Int32(kt) * fx.Int32(tile_k) + g_row
                    r = (r > last_row).select(last_row, r)
                    return Vec.from_elements([ridx[first_row + r]], fx.Int32)

                def _empty_expert_rows(src, ml, kt):
                    # Gathers of one hot row now and then corrupt a neighbouring workgroup's LDS writes
                    # (see gemm_nt_tile): an empty expert, whose accumulators are zeroed, reads
                    # distinct rows instead of re-reading one. Applied where the rows are used, a
                    # k-tile after their load, so the issue never waits on that load.
                    r = fx.Int32(kt) * fx.Int32(tile_k) + g_row
                    return (ml > fx.Int32(0)).select(src, (r < fx.Int32(gather_rows)).select(r, fx.Int32(0)))

                # This lane's source row for the next k-tile this tile issues itself, loaded one issue
                # ahead. Past the first tile the epilogue of the previous one issued k-tiles
                # [0, PRE_KT), so the tile's own run starts at PRE_KT.
                g_kt0 = 0
                if const_expr(PREFETCH_OK):
                    g_kt0 = (tile_raw < fx.Int32(C_STEP)).select(fx.Int32(0), fx.Int32(PRE_KT))
                gather_cell = fx.make_rmem_tensor(1, fx.Int32)
                gather_cell.store(_gather_src_rows(m_start, m_len, g_kt0))
                g_nxt = None
                if const_expr(nxt_dec is not None):
                    g_nxt = [_gather_src_rows(nxt_dec[0], nxt_dec[1], pk) for pk in range_constexpr(PRE_KT)]

                def _issue_gband(pa, kt, ms, ml, bne, bke, nxt_tile):
                    ktk = fx.Int32(kt) * fx.Int32(tile_k)
                    d_col, g_col = (bke, bne) if const_expr(gather == 1) else (bne, bke)
                    # The tile decode divides in vector registers; read the columns back so the
                    # descriptors are built in scalar registers.
                    d_col = rocdl.readfirstlane(T.i32, d_col)
                    g_col = rocdl.readfirstlane(T.i32, g_col)
                    d_stage = fx.add_offset(pa, STAGE_A) if const_expr(gather == 1) else pa
                    g_stage = pa if const_expr(gather == 1) else fx.add_offset(pa, STAGE_A)
                    rem = (ml - g_band) - ktk
                    rem = (rem < fx.Int32(0)).select(fx.Int32(0), rem)
                    gt = _gv(
                        d_base,
                        fx.Int64(fx.Int32(ms) + g_band + ktk) * d_ld + fx.Int64(d_col) * fx.Int64(EB),
                        (G_BAND, d_w * EB),
                        (d_ld, 1),
                    )
                    atom = _tdm(gt, rem, fx.Int32(d_w * EB), d_ld, d_w * EB, _mc_mask(d_pat, rem))
                    fx.copy(
                        atom,
                        gt,
                        _lv(
                            fx.add_offset(d_stage, g_band * fx.Int32(d_lds_row)),
                            (G_BAND, d_w * EB),
                            (d_lds_row, 1),
                        ),
                    )
                    if const_expr(nxt_tile):
                        src_rows = (
                            g_nxt[kt] if const_expr(g_nxt is not None) else _gather_src_rows(ms, ml, kt)
                        )[0]
                        src_rows = _empty_expert_rows(src_rows, ml, kt)
                    else:
                        src_rows = _empty_expert_rows(Vec(gather_cell.load())[0], m_len, kt)
                        gather_cell.store(_gather_src_rows(m_start, m_len, fx.Int32(kt) + fx.Int32(1)))
                    # Every row of a gather load is real (the tail re-reads the last one), so only an
                    # empty expert makes it a plain load, as `_mc_mask` does for an empty window.
                    g_mask = _mc_mask(g_pat, ml)
                    lds0 = fx.Int32(_bidx(g_stage)) + g_band * fx.Int32(g_lds_row)
                    src = fx.Int64(fx.ptrtoint(g_base)) + fx.Int64(g_col) * fx.Int64(EB)
                    per_op = GATHER_ROWS_PER_OP
                    for q in range_constexpr(GATHER_OPS):
                        rows = [_readlane_i32(src_rows, q * per_op + i) for i in range_constexpr(per_op)]
                        desc = _vk_gather_desc(
                            lds0 + fx.Int32(q * per_op * g_lds_row),
                            src,
                            rows,
                            IDX_BITS,
                            g_w,
                            gather_rows,
                            g_stride,
                            LDS_PAD,
                            mask=g_mask,
                            early=bool(mc_early),
                        )
                        tdm_ops.tensor_load_gather(desc)

            def _issue_dealt(pa, kt, ms, ml, bne, bke, col=None):
                ktk = fx.Int32(kt) * fx.Int32(tile_k)
                rem = (ml - dl_rows) - ktk
                rem = (rem < fx.Int32(0)).select(fx.Int32(0), rem)
                if const_expr(col is None):
                    # The tile decode divides in vector registers; read the column back so the
                    # descriptor is built in scalar registers.
                    col = rocdl.readfirstlane(T.i32, bne + dl_b * (bke - bne))
                off = fx.Int64(fx.Int32(ms) + dl_rows + ktk) * dl_ld + fx.Int64(col) * fx.Int64(EB)
                gt = fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(DL_GPTR, dl_gbase + off),
                        fx.make_layout((DL_ROWS, tile_m * EB), (dl_ld, 1)),
                    )
                )
                atom = _tdm(gt, rem, fx.Int32(tile_m * EB), dl_ld, tile_m * EB, _mc_mask(dl_pat, rem))
                fx.copy(atom, gt, _lv(fx.add_offset(pa, dl_lds), (DL_ROWS, tile_m * EB), (LDS_A_ROW, 1)))

            dl_col = None
            if const_expr(tdm_deal):
                dl_col = rocdl.readfirstlane(T.i32, blk_n_eff + dl_b * (blk_k_eff - blk_n_eff))

            def issue(s, kt, dec=None):
                pa = _buf_ptr(s)
                ms, ml, bne, bke = dec if dec is not None else (m_start, m_len, blk_n_eff, blk_k_eff)
                if const_expr(gather):
                    _issue_gband(pa, kt, ms, ml, bne, bke, dec is not None)
                elif const_expr(tdm_deal):
                    _issue_dealt(pa, kt, ms, ml, bne, bke, dl_col if const_expr(dec is None) else None)
                else:
                    _issue_pair(pa, kt, ms, ml, bne, bke)

            def _issue_pair(pa, kt, ms, ml, bne, bke):
                m_off = fx.Int32(ms) + fx.Int32(kt) * fx.Int32(tile_k)
                rem = ml - fx.Int32(kt) * fx.Int32(tile_k)
                rem = (rem < fx.Int32(0)).select(fx.Int32(0), rem)
                m_off64 = fx.Int64(m_off)
                if wave == 0 % num_waves:
                    gt = _gv(
                        gA_base,
                        m_off64 * ldn_b + fx.Int64(bne) * fx.Int64(EB),
                        (tile_k, tile_m * EB),
                        (ldn_b, 1),
                    )
                    # Full width: the backed-off window is entirely valid columns, so there is
                    # nothing left for a dim-1 extent to clamp.
                    atom = _tdm(gt, rem, fx.Int32(tile_m * EB), ldn_b, tile_m * EB, _mc_mask(mc_a_pat, rem))
                    fx.copy(atom, gt, _lv(pa, (tile_k, tile_m * EB), (LDS_A_ROW, 1)))
                if wave == 1 % num_waves:
                    gt = _gv(
                        gB_base,
                        m_off64 * ldk_b + fx.Int64(bke) * fx.Int64(EB),
                        (tile_k, tile_n * EB),
                        (ldk_b, 1),
                    )
                    atom = _tdm(gt, rem, fx.Int32(tile_n * EB), ldk_b, tile_n * EB, _mc_mask(mc_b_pat, rem))
                    fx.copy(atom, gt, _lv(fx.add_offset(pa, STAGE_A), (tile_k, tile_n * EB), (LDS_B_ROW, 1)))

            wmb = wave_m * warp_tile_m
            wnb = wave_n * warp_tile_n

            a_lane = (l8 + kgrp * fx.Int32(8)) * fx.Int32(LDS_A_ROW) + (
                wmb + hi8 * fx.Int32(8) + n_back
            ) * fx.Int32(EB)
            b_lane = (
                (l8 + kgrp * fx.Int32(8)) * fx.Int32(LDS_B_ROW)
                + (wnb + hi8 * fx.Int32(8) + k_back) * fx.Int32(EB)
                + fx.Int32(STAGE_A)
            )
            vec8 = ir.VectorType.get([8], elem_cls.ir_type)
            _PTR3 = ir.Type.parse("!llvm.ptr<3>")

            def _tr(addr, const_off):
                ptr_val = _llvm.inttoptr(_PTR3, _raw(addr + fx.Int32(const_off)))
                return Vec(_rocdl.ds_load_tr16_b128(vec8, ptr_val))

            def _frag(addr, row_stride, k_row_off, col_off):
                base = k_row_off * row_stride + col_off
                v0 = _tr(addr, base)
                v1 = _tr(addr, base + 16 * row_stride)
                return v0.shuffle(v1, list(range(16))).bitcast(fx.Int32)

            wmma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_cls, fx.Float32))
            c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_acc)]
            for cf in c_frags:
                cf.store(Vec.filled(8, 0.0, fx.Float32))

            def _rmem(n, v):
                t = fx.make_rmem_tensor(n, fx.Int32)
                t.store(v)
                return t

            DS_A = DS_B = 2
            KS_DS = wmma_m_rep * DS_A + wmma_n_rep * DS_B

            def _load_ks(addrs, ks):
                a_addr, b_addr = addrs
                act = [
                    _rmem(8, _frag(a_addr, LDS_A_ROW, ks * WMMA_K, wm * WMMA_M * EB))
                    for wm in range_constexpr(wmma_m_rep)
                ]
                wt = [
                    _rmem(8, _frag(b_addr, LDS_B_ROW, ks * WMMA_K, wn * WMMA_N * EB))
                    for wn in range_constexpr(wmma_n_rep)
                ]
                return act, wt

            def _mma_ks(state):
                act, wt = state
                for wm in range_constexpr(wmma_m_rep):
                    for wn_raw in range_constexpr(wmma_n_rep):
                        wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                        idx = wm * wmma_n_rep + wn
                        fx.gemm(wmma_atom, c_frags[idx], wt[wn], act[wm], c_frags[idx])

            def _hold_live(state):
                act, wt = state
                for t in act + wt:
                    _llvm.call_intrinsic(None, "llvm.fake.use", [_raw(t.load())], [], [])

            def compute_ktile(buf, prefetch_kt, defer_last=False):
                addrs = (fx.Int32(buf) + a_lane, fx.Int32(buf) + b_lane)
                n_mma = K_WS - 1 if const_expr(defer_last) else K_WS
                if const_expr(lds_cosched >= 2):
                    if const_expr(issue_first and prefetch_kt is not None):
                        rocdl.sched_barrier(0)
                        issue(_slot(prefetch_kt), prefetch_kt)
                        rocdl.sched_barrier(0)
                    ring = [_load_ks(addrs, 0), None]
                    rocdl.sched_dsrd(KS_DS)
                    rocdl.sched_barrier(0)
                    for ks in range_constexpr(K_WS):
                        load_ks = ks + 1
                        # Only this sub-step's reads are in flight here.
                        rocdl.s_wait_dscnt(0)
                        if const_expr(ks == 0 and prefetch_kt is not None and not issue_first):
                            rocdl.sched_barrier(0)
                            issue(_slot(prefetch_kt), prefetch_kt)
                            rocdl.sched_barrier(0)
                        if const_expr(load_ks < K_WS):
                            ring[load_ks % 2] = _load_ks(addrs, load_ks)
                        if const_expr(ks < n_mma):
                            _mma_ks(ring[ks % 2])
                            if const_expr(load_ks < K_WS):
                                for _j in range_constexpr(KS_DS):
                                    rocdl.sched_dsrd(1)
                                    rocdl.sched_mfma(1)
                                rocdl.sched_mfma(n_acc - KS_DS)
                                # Must follow the group hints; placed before them, the hints are ignored.
                                _hold_live(ring[ks % 2])
                            else:
                                rocdl.sched_mfma(n_acc)
                        elif const_expr(load_ks < K_WS):
                            rocdl.sched_dsrd(KS_DS)
                        rocdl.sched_barrier(0)
                    return ring[(K_WS - 1) % 2] if const_expr(defer_last) else None
                late_waits = []
                if const_expr(frag_pipeline == 2):
                    cur_a, cur_b = _load_ks(addrs, 0)
                    for ks in range_constexpr(K_WS):
                        nxt_a = (
                            [
                                _rmem(8, _frag(addrs[0], LDS_A_ROW, (ks + 1) * WMMA_K, wm * WMMA_M * EB))
                                for wm in range_constexpr(wmma_m_rep)
                            ]
                            if const_expr(ks + 1 < K_WS)
                            else None
                        )
                        rocdl.s_wait_dscnt(wmma_m_rep * DS_A if const_expr(nxt_a is not None) else 0)
                        if const_expr(ks == 0 and prefetch_kt is not None):
                            rocdl.sched_barrier(0)
                            issue(_slot(prefetch_kt), prefetch_kt)
                            rocdl.sched_barrier(0)
                        _mma_ks((cur_a, cur_b))
                        if const_expr(ks + 1 < K_WS):
                            cur_a = nxt_a
                            cur_b = [
                                _rmem(8, _frag(addrs[1], LDS_B_ROW, (ks + 1) * WMMA_K, wn * WMMA_N * EB))
                                for wn in range_constexpr(wmma_n_rep)
                            ]
                            rocdl.s_wait_dscnt(0)
                elif const_expr(frag_pipeline):
                    if const_expr(issue_first and prefetch_kt is not None):
                        rocdl.sched_barrier(0)
                        issue(_slot(prefetch_kt), prefetch_kt)
                        rocdl.sched_barrier(0)
                    cur = _load_ks(addrs, 0)
                    for ks in range_constexpr(K_WS):
                        nxt = _load_ks(addrs, ks + 1) if const_expr(ks + 1 < K_WS) else None
                        if const_expr(lds_cosched == 1 and ks >= n_mma):
                            late_waits.append(KS_DS if const_expr(nxt is not None) else 0)
                        else:
                            rocdl.s_wait_dscnt(KS_DS if const_expr(nxt is not None) else 0)
                        if const_expr(ks == 0 and prefetch_kt is not None and not issue_first):
                            rocdl.sched_barrier(0)
                            issue(_slot(prefetch_kt), prefetch_kt)
                            rocdl.sched_barrier(0)
                        if const_expr(ks < n_mma):
                            _mma_ks(cur)
                        if const_expr(nxt is not None):
                            cur = nxt
                else:
                    for ks in range_constexpr(K_WS):
                        cur = _load_ks(addrs, ks)
                        rocdl.s_wait_dscnt(0)
                        if const_expr(ks == 0 and prefetch_kt is not None):
                            rocdl.sched_barrier(0)
                            issue(_slot(prefetch_kt), prefetch_kt)
                            rocdl.sched_barrier(0)
                        if const_expr(ks < n_mma):
                            _mma_ks(cur)
                rocdl.sched_barrier(0)
                for w in late_waits:
                    rocdl.s_wait_dscnt(w)
                return cur if const_expr(defer_last) else None

            def _vk_split_fence(tail, outstanding):
                tdm_ops.tensor_wait(outstanding)
                rocdl.sched_barrier(0)
                rocdl.s_barrier_signal(_WG_BARRIER)
                rocdl.sched_barrier(0)
                _mma_ks(tail)
                rocdl.sched_barrier(0)
                rocdl.s_barrier_wait(_WG_BARRIER)

            # CORRECTNESS: the TDM outstanding budget is counted PER WAVE, not per workgroup.
            TDM_PW = (1 + GATHER_OPS) if const_expr(gather) else 1

            def _fence(outstanding):
                tdm_ops.tensor_wait(outstanding)
                gpu.barrier()

            # Clamp it up to the pipeline depth: a short expert then runs a few extra k-tiles
            # whose reads are all past m_len, hence zeroed, and the barriers stay matched.
            # vk_ovl bit 1 peels steady k-tile 0, so it keeps one more.
            KT_MIN = NB1 + 1 if const_expr(vk_ovl & 2) else NB1
            if const_expr(k_tiles_const):
                K_TILES = max(k_tiles_const, KT_MIN)
            else:
                k_tiles_raw = (m_len + fx.Int32(tile_k - 1)) // fx.Int32(tile_k)
                K_TILES = (k_tiles_raw < fx.Int32(KT_MIN)).select(fx.Int32(KT_MIN), k_tiles_raw)

            # Every peer walks the same number of tiles, each with the same K_TILES (same expert), so
            # every cluster barrier below is taken the same number of times by all of them.
            if const_expr(MC > 1 and mc_tile_sync):
                _cluster.cluster_barrier()
            for i in range_constexpr(NB1):
                if PREFETCH_OK and i < PRE_KT:
                    # `tile_raw < PERSIST` is exactly `_t == 0` and is workgroup-uniform.
                    if tile_raw < fx.Int32(C_STEP):
                        issue(_slot(i), i)
                elif const_expr(not (vk_ovl & 2)):
                    issue(_slot(i), i)
            n_steady = (K_TILES - NB1) if const_expr(k_tiles_const) else (K_TILES - fx.Int32(NB1))

            if const_expr(vk_ovl & 2):
                # In flight: k-tiles 0 and 1, then (past the first tile) this wave's EPI_OPS store ops of
                # the previous tile. In-order retirement: at most TDM_PW + EPI_OPS left = k-tile 0 landed.
                # The first tile has no store behind k-tile 1, so it waits for TDM_PW on its own.
                if tile_raw < fx.Int32(C_STEP):
                    tdm_ops.tensor_wait(TDM_PW)
                _fence(outstanding=TDM_PW + EPI_OPS)
            elif const_expr(split_bar):
                _fence(outstanding=TDM_PW * (num_buffers - 2))

            MMA_ORDER = [
                (wm, (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw)
                for wm in range_constexpr(wmma_m_rep)
                for wn_raw in range_constexpr(wmma_n_rep)
            ]

            def _mma_ks_range(state, lo, hi):
                act, wt = state
                for wm, wn in MMA_ORDER[lo:hi]:
                    idx = wm * wmma_n_rep + wn
                    fx.gemm(wmma_atom, c_frags[idx], wt[wn], act[wm], c_frags[idx])

            def _addrs(kt):
                buf = _bidx(_buf_ptr(_slot(kt)))
                return (fx.Int32(buf) + a_lane, fx.Int32(buf) + b_lane)

            def _load_into(addrs, ks, dst):
                act, wt = dst
                for wm in range_constexpr(wmma_m_rep):
                    act[wm].store(_frag(addrs[0], LDS_A_ROW, ks * WMMA_K, wm * WMMA_M * EB))
                for wn in range_constexpr(wmma_n_rep):
                    wt[wn].store(_frag(addrs[1], LDS_B_ROW, ks * WMMA_K, wn * WMMA_N * EB))

            def _opaque(v):
                # An asm result cannot be rematerialized, so the allocator keeps the address live
                # instead of recomputing it with a VALU add inside the loop.
                i32 = ir.IntegerType.get_signless(32)
                return fx.Int32(
                    _llvm.inline_asm(i32, [_raw(v)], "v_mov_b32 $0, $1", "=v,v", has_side_effects=True)
                )

            def _hp_run_issue(addrs, ring, prefetch_kt):
                # Sub-step 0 with the TDM issue inside its run: sub-step 1's reads ride the first KS_DS
                # matrix ops as usual, then the issue block's scalar ops are dealt DL_SALU per matrix op.
                # The issue follows the reads in program order, so its LDS write cannot hold them back.
                DL_SALU = 2
                DL_ILV = min(12, n_acc - KS_DS)
                ring[1] = _load_ks(addrs, 1)
                _mma_ks_range(ring[0], 0, KS_DS)
                for _j in range_constexpr(KS_DS):
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                rocdl.sched_barrier(0)
                issue(_slot(prefetch_kt), prefetch_kt)
                _mma_ks_range(ring[0], KS_DS, n_acc)
                for _j in range_constexpr(DL_ILV):
                    rocdl.sched_mfma(1)
                    rocdl.sched_group_barrier(4, DL_SALU, 0)
                rocdl.sched_mfma(n_acc - KS_DS - DL_ILV)
                _hold_live(ring[0])
                rocdl.sched_barrier(0)

            def _hp_ktile(addrs, prefetch_kt, head, prev_tail=None):
                # `head` holds this k-tile's sub-step 0, read in the previous barrier window.
                # `prev_tail` (the previous k-tile's deferred sub-step, whose matrix ops may still be
                # in flight) is held live across this run so the reads here cannot reuse its registers.
                ring = [head, None]
                for ks in range_constexpr(K_WS):
                    load_ks = ks + 1
                    rocdl.s_wait_dscnt(0)
                    if const_expr(ks == 0 and prefetch_kt is not None and (tdm_deal & 2)):
                        _hp_run_issue(addrs, ring, prefetch_kt)
                    else:
                        if const_expr(ks == 0 and prefetch_kt is not None):
                            rocdl.sched_barrier(0)
                            issue(_slot(prefetch_kt), prefetch_kt)
                            rocdl.sched_barrier(0)
                        if const_expr(load_ks < K_WS):
                            ring[load_ks % 2] = _load_ks(addrs, load_ks)
                        if const_expr(ks < K_WS - 1):
                            _mma_ks(ring[ks % 2])
                            if const_expr(load_ks < K_WS and hp_burst):
                                rocdl.sched_dsrd(KS_DS)
                                rocdl.sched_mfma(n_acc)
                                _hold_live(ring[ks % 2])
                                if const_expr(ks == 0 and prev_tail is not None):
                                    _hold_live(prev_tail)
                            elif const_expr(load_ks < K_WS):
                                for _j in range_constexpr(KS_DS):
                                    rocdl.sched_dsrd(1)
                                    rocdl.sched_mfma(1)
                                rocdl.sched_mfma(n_acc - KS_DS)
                                _hold_live(ring[ks % 2])
                            else:
                                rocdl.sched_mfma(n_acc)
                        elif const_expr(load_ks < K_WS):
                            rocdl.sched_dsrd(KS_DS)
                        rocdl.sched_barrier(0)
                return ring[(K_WS - 1) % 2]

            def _hp_fence(tail, outstanding, nxt_addrs, head, late=()):
                tdm_ops.tensor_wait(outstanding)
                rocdl.sched_barrier(0)
                rocdl.s_barrier_signal(_WG_BARRIER)
                rocdl.sched_barrier(0)
                _mma_ks_range(tail, 0, head_pipe)
                rocdl.sched_mfma(head_pipe)
                rocdl.sched_barrier(0)
                rocdl.s_barrier_wait(_WG_BARRIER)
                rocdl.sched_barrier(0)
                if const_expr(len(late) > 0):
                    for lk in range_constexpr(len(late)):
                        issue(_slot(late[lk]), late[lk])
                    rocdl.sched_barrier(0)
                if const_expr(nxt_addrs is not None):
                    _load_into(nxt_addrs, 0, head)
                _mma_ks_range(tail, head_pipe, n_acc)
                if const_expr(nxt_addrs is not None and hp_burst):
                    rocdl.sched_dsrd(KS_DS)
                    rocdl.sched_mfma(n_acc - head_pipe)
                    _hold_live(tail)
                elif const_expr(nxt_addrs is not None):
                    for _j in range_constexpr(KS_DS):
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                    rocdl.sched_mfma(n_acc - head_pipe - KS_DS)
                    _hold_live(tail)
                else:
                    rocdl.sched_mfma(n_acc - head_pipe)
                rocdl.sched_barrier(0)

            if const_expr(head_pipe):
                HEAD = (
                    [fx.make_rmem_tensor(8, fx.Int32) for _ in range_constexpr(wmma_m_rep)],
                    [fx.make_rmem_tensor(8, fx.Int32) for _ in range_constexpr(wmma_n_rep)],
                )
                # One address pair per ring slot, computed once: inside the loop a VALU add feeding
                # a fragment read is waited on with a full `depctr_va_vdst(0)` behind the WMMAs.
                SLOT_ADDRS = [
                    tuple(_opaque(x) for x in _addrs(s - BUF_SHIFT)) for s in range_constexpr(num_buffers)
                ]
                _load_into(SLOT_ADDRS[_slot(0)], 0, HEAD)
                rocdl.sched_barrier(0)

            def _steady_hp(kt, addrs, nxt_addrs, prev_tail=None):
                tail = _hp_ktile(addrs, kt + NB1, HEAD, prev_tail)
                _hp_fence(tail, TDM_PW * (num_buffers - 2), nxt_addrs, HEAD)
                return tail

            def _steady_ktile(kt):
                buf = _bidx(_buf_ptr(_slot(kt)))
                if const_expr(head_pipe):
                    _steady_hp(kt, _addrs(kt), _addrs(kt + 1))
                elif const_expr(split_bar):
                    tail = compute_ktile(buf, kt + NB1, defer_last=True)
                    _vk_split_fence(tail, TDM_PW * (num_buffers - 2))
                else:
                    _fence(outstanding=TDM_PW * (num_buffers - 2))
                    compute_ktile(buf, kt + NB1)
                if const_expr(MC > 1 and mc_sync > 0):
                    if fx.Int32(kt % mc_sync) == fx.Int32(mc_sync - 1):
                        _cluster.cluster_barrier()

            if const_expr(k_tiles_const):
                for kt in range_constexpr(n_steady):
                    _steady_ktile(kt)
            elif const_expr(head_pipe and (vk_ovl & 2)):
                # Steady k-tile 0 (K_TILES >= NB1 + 1) issues nothing: its prefetch target and the
                # remaining prologue slot are the C staging. Its fence retires everything (k-tile 1 and
                # this wave's store ops), and the barrier makes every wave's store complete, before
                # k-tiles NB1 - 1 and NB1 are issued; from k-tile 1 on the ring runs as usual.
                tail0 = _hp_ktile(SLOT_ADDRS[_slot(0)], None, HEAD)
                _hp_fence(tail0, 0, SLOT_ADDRS[_slot(1)], HEAD, late=(NB1 - 1, NB1))
                n_st = n_steady - fx.Int32(1)
                n_rot = n_st // fx.Int32(num_buffers)
                for it in range(n_rot):
                    prev = None
                    for r in range_constexpr(num_buffers):
                        kt_r = it * num_buffers + (r + 1)
                        if const_expr(hp_opaque):
                            kt_r = rocdl.readfirstlane(T.i32, fx.Int32(kt_r))
                        prev = _steady_hp(kt_r, SLOT_ADDRS[_slot(r + 1)], SLOT_ADDRS[_slot(r + 2)], prev)
                n_done = n_rot * fx.Int32(num_buffers)
                for kr in range(n_st - n_done):
                    kt = n_done + kr + fx.Int32(1)
                    _steady_ktile(kt)
            elif const_expr(head_pipe):
                # Whole ring rotations first, so every slot (and its addresses) is a constant.
                n_rot = n_steady // fx.Int32(num_buffers)
                for it in range(n_rot):
                    prev = None
                    for r in range_constexpr(num_buffers):
                        kt_r = it * num_buffers + r
                        if const_expr(hp_opaque):
                            kt_r = rocdl.readfirstlane(T.i32, fx.Int32(kt_r))
                        prev = _steady_hp(kt_r, SLOT_ADDRS[_slot(r)], SLOT_ADDRS[_slot(r + 1)], prev)
                        if const_expr(MC > 1 and mc_sync > 0):
                            if fx.Int32(kt_r % mc_sync) == fx.Int32(mc_sync - 1):
                                _cluster.cluster_barrier()
                n_done = n_rot * fx.Int32(num_buffers)
                for kr in range(n_steady - n_done):
                    kt = n_done + kr
                    _steady_ktile(kt)
            else:
                for kt in range(n_steady):
                    _steady_ktile(kt)

            for j in range_constexpr(NB1):
                kt = n_steady + j
                buf = _bidx(_buf_ptr(_slot(kt)))
                if const_expr(head_pipe):
                    tail = _hp_ktile(_addrs(kt), None, HEAD)
                    if const_expr(j < num_buffers - 2):
                        _hp_fence(tail, TDM_PW * (num_buffers - 3 - j), _addrs(kt + 1), HEAD)
                    else:
                        _mma_ks(tail)
                elif const_expr(split_bar):
                    tail = compute_ktile(buf, None, defer_last=True)
                    if const_expr(j < num_buffers - 2):
                        _vk_split_fence(tail, TDM_PW * (num_buffers - 3 - j))
                    else:
                        _mma_ks(tail)
                else:
                    _fence(outstanding=TDM_PW * (num_buffers - 2 - j))
                    compute_ktile(buf, None)

            # epilogue: out[g] tile, both output axes clamped
            accs = [c_frags[idx].load() for idx in range_constexpr(n_acc)]
            if const_expr(gather):
                # an empty expert has no row to re-read: its gathered LDS rows can be stale
                zero_acc = Vec.filled(8, 0.0, fx.Float32)
                accs = [Vec((m_len > fx.Int32(0)).select(acc, zero_acc)) for acc in accs]
            _fence(outstanding=0)
            if const_expr(MC > 1 and mc_epi_sync):
                _cluster.cluster_barrier()
            if PREFETCH_OK:
                dec2 = nxt_dec if const_expr(nxt_dec is not None) else _next_dec()
                # Ahead of the store ops, so the next tile can wait for these alone (in-order TENSORcnt).
                for pk in range_constexpr(PRE_KT):
                    issue(_slot(pk), pk, dec=dec2)
            c_off = fx.Int64(g_idx) * fx.Int64(C_GRP_ELEMS) + fx.Int64(blk_n) * ldc64 + fx.Int64(blk_k)

            def _epi_stage(wm_lo, wm_hi):
                for wm in range_constexpr(wm_hi - wm_lo):
                    wmg = wm_lo + wm
                    row_rel = wmb + wmg * WMMA_M + lane16
                    for wn in range_constexpr(wmma_n_rep):
                        col_rel = wnb + wn * WMMA_N + kgrp * 8
                        h = accs[wmg * wmma_n_rep + wn].to(elem_cls)
                        fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row_rel * C_ROW_E + col_rel) * EB)

            def _epi_store(r0, rows):
                if const_expr(r0):
                    ext0 = n_valid - fx.Int32(r0)
                    ext0 = (ext0 > fx.Int32(0)).select(ext0, fx.Int32(0))
                    off = c_off + fx.Int64(r0) * ldc64
                else:
                    ext0, off = n_valid, c_off
                gts = _gv(gC_base, off, (rows, C_ROW_E), (C_ROW_E, 1))
                atom = fx.rocdl.make_tdm_atom(
                    gts,
                    [ext0, (k_valid < fx.Int32(tile_n)).select(k_valid, fx.Int32(tile_n))],
                    strides=[ldc64, None],
                    num_warps=num_waves,
                    cache_modifier=tdm_store_cache,
                    early_timeout=False,
                )
                src = (
                    fx.recast_iter(elem_cls, base_ptr)
                    if const_expr(r0 == 0)
                    else fx.recast_iter(elem_cls, fx.add_offset(base_ptr, r0 * C_ROW_E * EB))
                )
                fx.copy(atom, _lv(src, (rows, C_ROW_E), (C_ROW_E, 1)), gts)

            if const_expr(epi_chunk > 1):
                EPI_REP = wmma_m_rep // epi_chunk
                EPI_ROWS = EPI_REP * WMMA_M
                for ch in range_constexpr(epi_chunk):
                    _epi_stage(ch * EPI_REP, (ch + 1) * EPI_REP)
                    gpu.barrier()
                    for i in range_constexpr(m_warp):
                        _epi_store(i * warp_tile_m + ch * EPI_ROWS, EPI_ROWS)
            else:
                _epi_stage(0, wmma_m_rep)
                gpu.barrier()
                _epi_store(0, tile_m)  # one band covering the whole tile
            if const_expr(not (vk_ovl & 2)):
                tdm_ops.tensor_wait(0)
                # Seam barrier: without it the persistent path is not reentrant.
                if persist:
                    gpu.barrier()

        # With a cluster the walk counts whole clusters: cluster c takes cluster tiles
        # c, c + C_STEP, ..., and its peers the adjacent tiles inside each.
        walk = mc_cid if const_expr(MC > 1) else pid
        if persist:
            for _t in range(TILES_PER_WG):
                _one_output_tile(walk + _t * fx.Int32(C_STEP))
            if const_expr(vk_ovl & 2):
                # The last tile's store ops (and its never-consumed prefetch) are still in flight.
                tdm_ops.tensor_wait(0)
        else:
            _one_output_tile(walk)

    # The cluster geometry must reach both the kernel and the launch, or the cluster never forms.
    kernel_grouped_variable_k(
        arg_c,
        arg_a,
        arg_b,
        arg_etab,
        arg_ridx,
        i32_n,
        i32_k,
        value_attrs=({"rocdl.cluster_dims": f"{MC},1,1"} if MC > 1 else None),
    ).launch(
        grid=(PERSIST, 1, 1),
        block=(block, 1, 1),
        stream=stream,
        cluster=((MC, 1, 1) if MC > 1 else None),
    )


_VK_CONSTEXPR0 = 8

_VK_COMPILED: dict = {}


def _vk_launch(args: tuple):
    """Same pre-bound CallState shortcut as ``_nt_launch``, for the variable-K kernel."""
    key = args[_VK_CONSTEXPR0:]
    fn = _VK_COMPILED.get(key)
    if fn is None:
        fn = flyc.compile(_launch_grouped_bf16_variable_k, *args)
        if fn is None:  # COMPILE_ONLY: nothing was executed, nothing to cache
            return None
        _VK_COMPILED[key] = fn
    return fn(*args)


@functools.lru_cache(maxsize=64)
def _pick_variable_k_config(OUT_M: int, OUT_N: int, avg_m: int):
    """(tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers) for the variable-K kernel.

    ``tile_m`` tiles the output's first axis and ``tile_n`` its second while
    ``tile_k`` is the reduction over tokens, so unlike the forward ``avg_m`` is the
    **reduction** dimension here: the output is fixed by the weights, and shrinking
    the tile for a short token run buys no occupancy while multiplying the output
    traffic. The 2x2 warp grid the NT kernel uses carries over.

    There is no ``tile_n = 192`` branch here, unlike ``_pick_config``: this kernel's
    LDS rows feed the TDM atom's ``pad_interval``, which flydsl requires to be a
    power of two, so a 192-wide output tile aborts inside MLIR rather than raising.
    """
    # The read descriptors back their base off by up to a full tile (see the kernel
    # body), so a narrow output axis has nothing to back off into.
    if OUT_M < 256 or OUT_N < 256:
        return (64, 64, 64, 2, 2, 2)
    # At avg_m <= 128 the shipped tile runs exactly ONE k-tile per expert
    # (`K_TILES = max(ceil(avg_m/tile_k), num_buffers-1)` is 1 when tile_k == 128 ==
    # avg_m), so the double buffer never reaches steady state and the whole TDM
    # latency is exposed. Halving tile_k gives it two k-tiles to overlap, and 2x2
    # drops to one wave per SIMD, which both removes the co-resident LDS phase
    # collision `_vk_wave_grid` is about and gives each wave the full 128x128
    # accumulator (spill 0, checked at the compile gate).
    #
    # Pinned to avg_m == 128 because the effect REVERSES above it, and that is the
    # only batch in the delivery matrix where avg_m == tile_k.
    if avg_m <= 128:
        return (256, 256, 64, 2, 2, 2)
    # 2x2 instead of 4x2. ONLY `m_warp` moves -- tile area, `tile_k` and
    # `num_buffers` are untouched, so this is the warp grid on its own. 4x2 puts
    # eight waves in a workgroup, two per SIMD, and splits the accumulator; 2x2 is
    # four waves, one per SIMD, each holding the whole 128x128 accumulator (spill 0,
    # checked at the compile gate -- `(256,256,128,4,4,2)` spills and is refused).
    #
    # ⚠️ MECHANISM NOT ESTABLISHED, and in particular `_vk_wave_grid` is NOT voided
    # at four waves: forcing the other spelling on this very tile costs about 10%,
    # which rules out SIMD co-residency as the story, since at 2x2 there is no
    # co-resident pair to mis-pair. The explanation left standing is the one written
    # down for the NN path (see `_nn_native_launch`): `issue()` assigns the
    # global->LDS copies by RAW WAVE ID, so the relabelling re-pairs which loader
    # wave feeds which output sub-tile. That is a hypothesis, recorded as one.
    return (256, 256, 128, 2, 2, 2)


def grouped_gemm_bf16_variable_k_supported(OUT_M: int, OUT_N: int, avg_m: int = 0) -> bool:
    """Whether ``grouped_gemm_bf16_variable_k_flydsl_kernel`` can serve this shape.

    The non-throwing form of the public entry's assertions, for a dispatcher that
    has to pick a backend before committing. The read descriptors back their base
    off by up to a whole tile so the claimed window stays inside the row whatever
    the remainder, which is why there has to be a whole tile to back off into.
    """
    tile_m, tile_n = _pick_variable_k_config(OUT_M, OUT_N, avg_m)[:2]
    return OUT_M >= tile_m and OUT_N >= tile_n


# Back-compat alias for the bring-up scripts, which know this kernel as "wgrad_tn".
wgrad_tn_ok = grouped_gemm_bf16_variable_k_supported

# The grouped GEMM's tuned variable-K group (`_STACK_KEYS["vk"]`, its tile and its riders), looked
# up by weight shape alone: the pool a Mega MoE wgrad reduces over is sized by the dispatch
# capacity, not by the routed tokens, so its avg_m is an upper bound and cannot select a key.
# What the group needs from the token counts is checked on the device side (`K_TILES` is clamped).
_VK_STACK_KEY = {(n, k): (n, k, m) for (n, k, m) in _STACK_KEYS["vk"]}


def grouped_gemm_bf16_variable_k_flydsl_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    group_k_offsets: torch.Tensor,
    masked_k: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
    # 0 = pick from the shape (recommended). The gfx950 entry defaults these to
    # 256; passing 256 explicitly here gives the same tile, but leaving them at 0
    # lets `_pick_variable_k_config` also choose tile_k / warps / buffer count,
    # which is where most of the measured win is.
    BLOCK_M: int = 0,
    BLOCK_N: int = 0,
    # Reduction depth in tokens, and the 4-buffer-style TDM pipeline depth.
    # gfx1250-only knobs; 0 = from the shape.
    BLOCK_K: int = 0,
    m_warp: int = 0,
    n_warp: int = 0,
    num_buffers: int = 0,
    # Super-tile width in output blocks: co-resident workgroups share a column
    # slice. 16, not the gfx950 default of 4; re-measured after `split_bar` was
    # wired onto the same resource and it still holds.
    group_m: int = 16,
    # XCD band-cyclic remap; members of the stack group (-1 = from it, 1 / 32 off it).
    # Correctness does not depend on the value.
    num_xcd: int = -1,
    xcd_band: int = -1,
    wmma_b2b: int = -1,
    frag_pipeline: int = 1,
    # 1 = run the k-tile's last sub-step of WMMA inside the workgroup barrier's
    # signal-to-wait window. Same knob as on the NT/NN entry.
    #
    # -1 = resolve from the shape (recommended). This kernel does NOT use the NT/NN
    # rule: WG/CU is 1 on every tile it returns, so that rule would turn the knob on
    # everywhere and it measured 4/6 negative when it did. What governs here is
    #   (a) executed steady-loop iterations >= 11, because the fence's cost is per
    #       workgroup and its benefit is per iteration, and
    #   (b) both operands' global row strides being a multiple of 256 B.
    # Both are measured; see `_vk_auto_split_bar`.
    split_bar: int = -1,
    # ⚠️ EXPERIMENT ONLY -- see the launcher parameter of the same name. Pins the
    # per-expert k-tile count to a compile-time constant so the steady-state loop
    # unrolls. WRONG ANSWERS, not slow ones, if the constant disagrees with the real
    # per-expert row counts -- hence the acknowledgement flag below.
    k_tiles_const: int = 0,
    # > 0 over-allocates `out` along its last axis and writes with that row stride,
    # leaving the returned view logically [G, OUT_M, OUT_N]. Separates the write-side
    # stride from `b`'s read-side stride, which are otherwise the same number.
    # Requires `out` to be absent or to already have the padded shape.
    out_ldc: int = 0,
    i_am_running_an_experiment: bool = False,
    # A (start, len) table from a previous call on the same `group_k_offsets` and
    # `masked_k`. Both variable-K calls in one MoE layer share it, so passing it
    # back removes one of the two builds. Provenance-checked -- a table from a
    # different call raises rather than silently computing the wrong thing.
    expert_table: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    # gfx950 parameters accepted for signature compatibility; see below.
    trans_c: bool = False,
    cap_cu: int = 0,
    # A/B only; see the NT entry for the encoding and for why no value here ships.
    tdm_load_cache: int = 0,
    tdm_store_cache: int = 0,
    # Byte pad on both transposed LDS stage rows; multiple of 16. 0 = auto, which
    # is `_vk_lds_pad(tile_m, tile_n)` -- the conflict-free residue class. Pass an
    # explicit value only to measure the alternative; see the launcher.
    lds_pad: int = 0,
    # -1 = auto (`_vk_wave_grid`). 0/1 force one of the two warp-grid spellings.
    wave_grid: int = -1,
    # The persistent-cluster group of the grouped GEMM's variable-K launcher (see its parameters).
    # -1 = from `_VK_STACK_KEY` for this weight shape; a call that sets any member gets exactly
    # what it passed and the rest at their off values. `persist` must divide TOTAL_TILES and
    # leave >= 2 tiles per workgroup; both are checked in the launcher.
    persist: int = -1,
    issue_first: int = 0,
    epi_chunk: int = -1,
    lds_cosched: int = -1,
    mc_m: int = -1,
    mc_n: int = -1,
    mc_early: int = -1,
    mc_sync: int = -1,
    mc_tile_sync: int = -1,
    mc_epi_sync: int = -1,
    head_pipe: int = -1,
    hp_burst: int = -1,
    hp_opaque: int = -1,
    tdm_deal: int = -1,
    vk_ovl: int = -1,
    # [M_total] int32: reduction row m of `a` (or `b`) is row `a_index[m]` of the tensor
    # passed, which may then hold any number of rows. At most one of the two.
    a_index: torch.Tensor | None = None,
    b_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Variable-K grouped wgrad: ``out[g] = a[rows_g].T @ b[rows_g]``.

    Drop-in for the gfx950 entry point of the same name. ``a`` is
    ``[M_total, OUT_M]`` and ``b`` is ``[M_total, OUT_N]`` with the groups
    concatenated along the reduction (token) dim, ``group_k_offsets`` ``[G+1]``
    gives each group's row start, and ``masked_k`` ``[G]`` the per-group **valid**
    row count so a padded tail is never read. Returns ``[G, OUT_M, OUT_N]``.

    In autograd terms with ``a = dout[M, N]`` and ``b = x[M, K]`` this is
    ``db[g] = dout[rows_g].T @ x[rows_g]`` of shape ``[G, N, K]`` -- the forward
    weight's layout. It is **end to end**: both operands are read token-major and
    the transpose happens inside LDS, so nothing is hidden outside the timed region.

    Differences from the gfx950 entry point
    ---------------------------------------
    * ``BLOCK_M`` / ``BLOCK_N`` default to 0 ("pick from the shape") rather than
      256. Pass 256 to reproduce the gfx950 default exactly.
    * ``trans_c=True`` is **not implemented** -- the epilogue would need a
      transposed TDM store -- and neither is ``cap_cu``, since the kernel launches
      one workgroup per output tile with no persistent loop to bound. Both raise
      rather than being silently ignored.
    * Requires ``OUT_M >= tile_m`` and ``OUT_N >= tile_n`` (see
      ``grouped_gemm_bf16_variable_k_supported``), because the read descriptors back
      their base off by a whole tile instead of bounding reads with an SRD
      ``num_records`` the way gfx950 does.
    * On the weight shapes the grouped GEMM has a tuned variable-K group for, the call runs that
      group (persistent clusters with TDM multicast; with a row index, its gather variant);
      elsewhere one workgroup per output tile.
    """
    assert a.dim() == 2 and b.dim() == 2, f"a must be [M,OUT_M] and b [M,OUT_N], got {a.dim()}/{b.dim()}"
    assert a.dtype == b.dtype and a.dtype in _F16, f"16-bit float operands only, got {a.dtype}/{b.dtype}"
    assert a.is_contiguous() and b.is_contiguous(), "operands must be contiguous"
    M_total, OUT_M = a.shape
    M2, OUT_N = b.shape
    assert a_index is None or b_index is None, "only one operand can be gathered"
    gather, gather_rows, row_index = 0, 0, None
    if a_index is not None:
        gather, gather_rows, row_index, M_total = 1, a.shape[0], a_index, M2
    elif b_index is not None:
        gather, gather_rows, row_index = 2, b.shape[0], b_index
    else:
        assert M2 == M_total, f"token counts differ: a M={M_total} vs b M={M2}"
    if row_index is not None:
        assert row_index.dtype == torch.int32 and row_index.is_contiguous(), "row index: contiguous int32"
        assert row_index.numel() >= M_total, f"row index covers {row_index.numel()} of {M_total} rows"
    G = group_k_offsets.numel() - 1
    _require_gfx1250(a.device)
    if trans_c:
        raise NotImplementedError(
            "trans_c=True is not implemented on gfx1250: the epilogue stores the tile through a "
            "TDM copy whose layout is fixed at [OUT_M, OUT_N]. Transpose the result, or use the "
            "gfx950 kernel."
        )
    if cap_cu:
        raise NotImplementedError(
            f"cap_cu={cap_cu} is not implemented on gfx1250: the kernel launches one workgroup per "
            f"output tile and has no persistent loop to bound. Pass cap_cu=0 (whole device)."
        )
    if masked_k is not None:
        assert masked_k.numel() == G, f"masked_k len {masked_k.numel()} != G {G}"
    if out_ldc and not i_am_running_an_experiment:
        raise ValueError(
            f"out_ldc={out_ldc} is an EXPERIMENT-ONLY override. Unlike the other knobs "
            f"here it can fault the device rather than return a wrong answer, because "
            f"the store's row stride and its per-expert span have to move together. "
            f"Pass i_am_running_an_experiment=True if that is genuinely what you want."
        )
    if k_tiles_const and not i_am_running_an_experiment:
        raise ValueError(
            f"k_tiles_const={k_tiles_const} is an EXPERIMENT-ONLY override that pins the "
            f"per-expert k-tile count at compile time. It computes WRONG ANSWERS, not slow "
            f"ones, on any input whose per-expert row counts disagree with it. Pass "
            f"i_am_running_an_experiment=True if that is genuinely what you want."
        )

    _avg_m = max(1, M_total // max(G, 1))
    _passed = (
        persist,
        mc_m,
        mc_n,
        mc_early,
        mc_sync,
        mc_tile_sync,
        mc_epi_sync,
        head_pipe,
        hp_burst,
        hp_opaque,
        tdm_deal,
        vk_ovl,
        num_xcd,
        xcd_band,
    )
    # The grouped GEMM's key for this weight shape: the group, its tile and its riders are read
    # there, so the two kernels' tuned values cannot drift apart.
    _key = _VK_STACK_KEY.get((OUT_M, OUT_N))
    _want = (
        _key is not None
        and not (BLOCK_M or BLOCK_N or BLOCK_K or m_warp or n_warp or num_buffers)
        and all(v < 0 for v in _passed)
        and lds_cosched in (-1, 2)
        and split_bar in (-1, 1)
        and frag_pipeline == 1
        and not (issue_first or k_tiles_const)
    )

    def _resolve(cfg):
        tile = (
            BLOCK_M or cfg[0],
            BLOCK_N or cfg[1],
            BLOCK_K or cfg[2],
            m_warp or cfg[3],
            n_warp or cfg[4],
            num_buffers or cfg[5],
        )
        # Pad, warp grid and split barrier are resolved against the effective tile, so an
        # explicit BLOCK_* / num_buffers override is judged on what it actually produces.
        return (
            tile,
            lds_pad or _vk_lds_pad(tile[0], tile[1], 2),
            _vk_wave_grid(tile[3], tile[4]) if wave_grid == -1 else wave_grid,
            _vk_auto_split_bar(tile, OUT_M, OUT_N, _avg_m) if split_bar == -1 else split_bar,
        )

    _stack = None
    if _want:
        tile, pad, grid, sbar = _resolve(_v3_pick_variable_k_config(*_key, True))
        _stack = _pick_stack(
            "vk", *_key, _STACK_VK, _passed, tile, sbar == 1 and pad == _vk_lds_pad(*tile[:2], 2)
        )
        n_bn, n_bk = -(-OUT_M // tile[0]), -(-OUT_N // tile[1])
        if _stack is not None and not _vk_stack_cluster_ok(_stack, n_bn, n_bk, G * n_bn * n_bk, group_m):
            _stack = None
    if _stack is None:
        tile, pad, grid, sbar = _resolve(_pick_variable_k_config(OUT_M, OUT_N, _avg_m))
    (tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers), lds_pad, wave_grid, split_bar = (
        tile,
        pad,
        grid,
        sbar,
    )
    # The read descriptors back their base off by up to a full tile to keep the
    # claimed window inside the row, so there has to be a full tile to back off
    # into. grouped_gemm_bf16_variable_k_supported() is the non-throwing form.
    assert OUT_M >= tile_m, f"OUT_M={OUT_M} must be >= tile_m={tile_m} for the descriptor back-off"
    assert OUT_N >= tile_n, f"OUT_N={OUT_N} must be >= tile_n={tile_n} for the descriptor back-off"
    if wave_grid not in (0, 1):
        raise ValueError(f"wave_grid must be 0 or 1 (or -1 for auto), got {wave_grid}")
    _n_blocks_n = (OUT_M + tile_m - 1) // tile_m
    _n_blocks_k = (OUT_N + tile_n - 1) // tile_n
    # The group's riders: keyed values on it, off values everywhere else.
    if epi_chunk < 0:
        epi_chunk = _pick_epi_chunk("vk", *_key, True) if _stack is not None else 1
    if wmma_b2b < 0:
        wmma_b2b = _pick_wmma_b2b("vk", *_key) if _stack is not None else 0
    if lds_cosched < 0:
        lds_cosched = _pick_lds_cosched("vk", *_key, True) if _stack is not None else 0
    (
        persist,
        mc_m,
        mc_n,
        mc_early,
        mc_sync,
        mc_tile_sync,
        mc_epi_sync,
        head_pipe,
        hp_burst,
        hp_opaque,
        tdm_deal,
        vk_ovl,
        num_xcd,
        xcd_band,
    ) = ((_stack or _stack_off(_STACK_VK, _passed))[n] for n in _STACK_VK)
    if gather and _stack is not None:
        # The gathered operand's per-wave loads replace the dealt issue (`tdm_deal`), and between
        # them the per-op fragment interleave stalls the WMMA chain: 1.52 vs 0.82 ms on dW2 with
        # `hp_burst` 0 / 1. And with the reduction being the expert, an XCD band of consecutive
        # tiles holds one expert's tiles, so a skewed routing loads its whole band onto a few
        # XCDs: 1.16 vs 0.84 ms on dW2 with one expert at half the rows, < 1% on an even one.
        # The early multicast timeout is worth 4-6% on that skewed case and nothing on the even one.
        hp_burst, tdm_deal, num_xcd, xcd_band = (1 if head_pipe else hp_burst), 0, 1, 32
        mc_early = 1 if mc_m * mc_n > 1 else mc_early
    # Off (or without its head_pipe placement) where the shape does not meet its preconditions.
    if tdm_deal and ((m_warp * n_warp) % 2 or tile_k % ((m_warp * n_warp) // 2) or tile_m != tile_n):
        tdm_deal = 0
    if (tdm_deal & 2) and (not head_pipe or hp_burst):
        tdm_deal &= 1

    # The store walks `g*OUT_M*ldc + blk_n*ldc + blk_k`, so the allocation has to be
    # OUT_M*ldc per expert, not OUT_M*OUT_N. Checked rather than assumed: this is the
    # one place in this entry where a mismatch is an out-of-bounds device write.
    c_ldc = out_ldc or OUT_N
    if out_ldc:
        if out_ldc < OUT_N:
            raise ValueError(f"out_ldc={out_ldc} must be >= OUT_N={OUT_N}")
        if out is None:
            out = torch.empty(G, OUT_M, out_ldc, device=a.device, dtype=out_dtype)
        if tuple(out.shape) != (G, OUT_M, out_ldc) or not out.is_contiguous():
            raise ValueError(
                f"out_ldc={out_ldc} requires a contiguous out of shape "
                f"{(G, OUT_M, out_ldc)}, got shape {tuple(out.shape)} "
                f"contiguous={out.is_contiguous()}. Refusing: the store would write "
                f"past the allocation."
            )
        assert out.numel() >= G * OUT_M * out_ldc, "out too small for the padded stride"
    elif out is None:
        out = torch.empty(G, OUT_M, OUT_N, device=a.device, dtype=out_dtype)
    if expert_table is not None:
        check_prebuilt(expert_table, group_k_offsets, 0, G, 2 * G, "expert_table", masked_k=masked_k)
        etab = expert_table.reshape(-1)
    else:
        etab = build_expert_table(group_k_offsets, masked_k).reshape(-1)

    _vk_launch(
        (
            _as_ptr(out),
            _as_ptr(a),
            _as_ptr(b),
            _as_ptr(etab),
            _as_ptr(etab if row_index is None else row_index),
            torch.cuda.current_stream(),
            OUT_M,
            OUT_N,
            G,
            _n_blocks_n,
            _n_blocks_k,
            OUT_M * c_ldc,
            tile_m,
            tile_n,
            tile_k,
            m_warp,
            n_warp,
            num_buffers,
            1 if a.dtype == torch.float16 else 0,
            group_m,
            num_xcd,
            xcd_band,
            wmma_b2b,
            frag_pipeline,
            split_bar,
            k_tiles_const,
            out_ldc,
            tdm_load_cache,
            tdm_store_cache,
            lds_pad,
            wave_grid,
            persist,
            issue_first,
            epi_chunk,
            lds_cosched,
            mc_m,
            mc_n,
            mc_early,
            mc_sync,
            mc_tile_sync,
            mc_epi_sync,
            head_pipe,
            hp_burst,
            hp_opaque,
            tdm_deal,
            vk_ovl,
            gather,
            gather_rows,
        )
    )
    # The caller asked for [G, OUT_M, OUT_N]; the padding is this function's business.
    return out[:, :, :OUT_N] if out_ldc else out
