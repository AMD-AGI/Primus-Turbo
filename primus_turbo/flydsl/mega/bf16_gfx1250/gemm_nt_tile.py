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

"""Mega MoE NT GEMM tile for gfx1250: ``C[pool block] = A[rows] @ B[expert].T``.

One workgroup computes one ``tile_m x tile_n`` block of one 256-row pool block, the unit Mega
schedules on. The pipeline is the shipped configuration of
``grouped_gemm_bf16_kernel_gfx1250._launch_grouped_gemm_bf16_nt`` (TDM global -> LDS, two
buffers, WMMA 16x16x32, ``split_bar`` with one deferred sub-step, a two-deep fragment ring),
with two changes Mega needs:

* **A can be gathered.** In ``gather`` mode the A rows come from a row-index table (pool row ->
  dispatched token) through TDM gather loads, 16 rows per instruction with 16-bit indices or 8
  with 32-bit ones. An index past ``a_rows`` -- the pool's padding rows carry -1 -- is out of
  range for the TDM engine, which leaves that LDS row unwritten; the row's output is padding
  that nothing downstream reads.
* **The epilogue stores with plain buffer stores**, staged through LDS so every lane writes 16
  contiguous bytes. The grouped kernel's ``tensor_store_from_lds`` epilogue is the remaining
  suspect of its rare silent wrong result, so Mega does not use it.

A dead tile (past the device-side tile count) is not branched around -- a dynamic branch over the
TDM pipeline is what the grouped kernel's ``dead_tile_exit`` found to fault intermittently.
It runs with every extent at 0: no k-tile, no TDM traffic, a store with ``num_records`` 0.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import vector
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Constexpr, T, as_ir_value
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    _WG_BARRIER,
    _as_ptr,
    _i32_table,
    _make_lds_copy_ops,
    _pipeline_fence,
    _require_gfx1250,
    _workgroup_barrier,
)
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_store,
    create_buffer_resource_from_addr,
)

WAVE = 32
WMMA_M = WMMA_N = 16
WMMA_K = 32
EB = 2  # bf16
LDS_PAD = 16  # bytes after every LDS row: keeps rows 4 dwords apart for conflict-free b128 reads
NUM_BUFFERS = 2
GFX1250_LDS_BYTES = 320 * 1024


def tile_geometry(tile_m: int, tile_n: int, tile_k: int):
    """(LDS row bytes, A stage bytes, buffer pitch, arena bytes) of one tile configuration."""
    lds_row = tile_k * EB + LDS_PAD
    stage = ((tile_m * lds_row + 15) // 16) * 16
    stage_b = ((tile_n * lds_row + 15) // 16) * 16
    pitch = ((stage + stage_b + 1023) // 1024) * 1024
    arena = max(NUM_BUFFERS * pitch, tile_m * tile_n * EB)
    return lds_row, stage, pitch, arena


# ── TDM gather descriptors ───────────────────────────────────────────────────
# Built by hand rather than through tdm_ops.make_tensor_gather_descriptor, which wants the LDS
# side as a memref; this tile's LDS is a pointer arena. Field layout per CDNA5 ISA 4.10.3.2,
# the same bits tdm_ops packs.


def _c32(v):
    return arith.constant(v, type=T.i32)


def _pad_encoding(row_bytes: int, pad_bytes: int):
    interval_dw, amount_dw = row_bytes // 4, pad_bytes // 4
    assert interval_dw & (interval_dw - 1) == 0, "TDM pad interval must be a power of two dwords"
    return interval_dw.bit_length() - 2, amount_dw - 1


def _gather_static_groups(row_indices, idx_bits: int, row_width: int, tensor_dim0: int, tensor_dim1, stride: int):
    """dgroup1..3: element size, padding, OOB dims, row width, row stride and the row indices."""
    enc_interval, enc_amount = _pad_encoding(row_width * EB, LDS_PAD)
    data_size_code = EB.bit_length() - 1
    s0 = _c32((data_size_code << 16) | (1 << 20) | (enc_interval << 22) | (enc_amount << 25))
    s1 = _c32((tensor_dim0 & 0xFFFF) << 16)
    td1 = _raw(tensor_dim1)
    s2 = arith.ori(_c32((tensor_dim0 >> 16) & 0xFFFF), arith.shli(arith.andi(td1, _c32(0xFFFF)), _c32(16)))
    s3 = arith.ori(arith.andi(arith.shrui(td1, _c32(16)), _c32(0xFFFF)), _c32(row_width << 16))
    s4 = _c32(len(row_indices))
    s5 = _c32(stride)
    zero = _c32(0)
    dgroup1 = vector.from_elements(T.vec(8, T.i32), [as_ir_value(v) for v in (s0, s1, s2, s3, s4, s5, zero, zero)])
    if idx_bits == 32:
        words = list(row_indices)
    else:
        words = [
            arith.ori(
                arith.andi(_raw(row_indices[2 * w]), _c32(0xFFFF)),
                arith.shli(arith.andi(_raw(row_indices[2 * w + 1]), _c32(0xFFFF)), _c32(16)),
            )
            for w in range(len(row_indices) // 2)
        ]
    dgroup2 = vector.from_elements(T.vec(4, T.i32), [as_ir_value(_raw(w)) for w in words[:4]])
    dgroup3 = vector.from_elements(T.vec(4, T.i32), [as_ir_value(_raw(w)) for w in words[4:8]])
    return dgroup1, dgroup2, dgroup3


def _gather_dgroup0(lds_addr_i32, global_addr_i64, idx_bits: int):
    """dgroup0: gather-mode predicate, LDS byte address and the 57-bit global base."""
    i32 = ir.IntegerType.get_signless(32)
    g = _raw(global_addr_i64)
    lo = std_arith.TruncIOp(i32, g).result
    hi = std_arith.TruncIOp(i32, std_arith.ShRUIOp(g, _raw(fx.Int64(32))).result).result
    hi = arith.ori(hi, _c32(1 << 31))
    pred = _c32(1 | ((1 if idx_bits == 32 else 0) << 30) | (1 << 31))
    return vector.from_elements(
        T.vec(4, T.i32), [as_ir_value(v) for v in (pred, _raw(lds_addr_i32), lo, hi)]
    )


class _GatherDesc:
    def __init__(self, dgroup0, dgroup1, dgroup2, dgroup3):
        self.dgroup0, self.dgroup1, self.dgroup2, self.dgroup3 = dgroup0, dgroup1, dgroup2, dgroup3


# ── the tile ─────────────────────────────────────────────────────────────────


@ASTRewriter.transform
def nt_gemm_tile(
    base_ptr,
    *,
    gA_base,
    lda: int,
    a_row0,
    a_rows,
    row_idx_table,
    row_idx_base,
    gB_base,
    ldb: int,
    b_slab,
    blk_n,
    c_addr,
    ldc: int,
    m_rows,
    n_valid,
    i32_k,
    live,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    gather: bool,
    idx_bits: int = 16,
    c_cache_modifier: int = 0,
):
    """One ``tile_m x tile_n`` NT block; must be called by all ``m_warp * n_warp`` waves.

    ``base_ptr`` is the LDS arena (``tile_geometry(...)[3]`` bytes). A is ``gA_base`` (Int8
    iterator) with row stride ``lda`` elements: rows ``a_row0 ..`` when dense, rows
    ``row_idx_table[row_idx_base ..][:tile_m]`` (an i32 iterator) when
    ``gather``, bounded by ``a_rows``. B is ``gB_base + b_slab`` with row stride ``ldb``; this
    block reads its rows ``blk_n ..``. C is written at byte address ``c_addr`` with row stride
    ``ldc``, ``m_rows`` rows and ``n_valid`` columns kept. ``live`` (wave-uniform i1) turns the
    tile into a no-op. ``c_cache_modifier`` is the C stores' cache policy (``cpol_device_stream()``
    when another workgroup consumes C in the same launch).
    """
    KB = tile_k * EB
    KSB = WMMA_K * EB
    K_WS = tile_k // WMMA_K
    warp_tile_m, warp_tile_n = tile_m // m_warp, tile_n // n_warp
    wmma_m_rep, wmma_n_rep = warp_tile_m // WMMA_M, warp_tile_n // WMMA_N
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    LDS_ROW, STAGE_A, PITCH, _arena = tile_geometry(tile_m, tile_n, tile_k)
    B_DELTA = STAGE_A
    R = 2  # fragment ring depth
    ROWS_PER_OP = 16 if idx_bits == 16 else 8
    OPS = tile_m // ROWS_PER_OP
    assert OPS % num_waves == 0, "gather ops must split evenly over the waves"
    OPS_PER_WAVE = OPS // num_waves
    assert ldc % 2 == 0, "the epilogue stores whole i32 words"

    tid = fx.Int32(fx.thread_idx.x)
    wave = rocdl.readfirstlane(T.i32, tid // WAVE)
    lane = tid % WAVE
    lane16 = lane % 16
    kgrp = lane // 16
    wave_m = wave // n_warp
    wave_n = wave % n_warp

    K_TILES = i32_k // tile_k
    # a dead tile runs no k-tile at all; every extent below is 0 as well
    n_steady = live.select(K_TILES - fx.Int32(NUM_BUFFERS - 1), fx.Int32(0))
    zero32 = fx.Int32(0)
    m_ext = live.select(m_rows, zero32)
    n_ext = live.select(n_valid, zero32)
    lda_b = fx.Int64(lda * EB)
    ldb_b = fx.Int64(ldb * EB)

    def _bidx(p):
        return fx.Int64(fx.ptrtoint(p))

    def _buf_ptr(s):
        return fx.add_offset(base_ptr, s * PITCH)

    def _gv(base, off, shape, stride):
        return fx.Tensor(fx.make_view(fx.add_offset(base, off), fx.make_layout(shape, stride)))

    def _lv(ptr, shape, stride):
        return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

    def _tdm(gt, outer, stride):
        return fx.rocdl.make_tdm_atom(
            gt,
            [outer, None],
            strides=[stride, None],
            num_warps=1,
            pad_interval=KB,
            pad_amount=LDS_PAD,
            cache_modifier=0,
            early_timeout=True,
        )

    gt_b = _gv(gB_base, b_slab + fx.Int64(blk_n) * ldb_b, (tile_n, KB), (KB, 1))
    atom_b = _tdm(gt_b, n_ext, ldb_b)
    if const_expr(gather):
        a_addr = fx.Int64(fx.ptrtoint(gA_base))
        # descriptors for this wave's ops: indices are K-invariant, so groups 1-3 are built once
        a_dim1 = live.select(a_rows, zero32)
        gather_groups = []
        for q in range_constexpr(OPS_PER_WAVE):
            first = (wave * fx.Int32(OPS_PER_WAVE) + fx.Int32(q)) * fx.Int32(ROWS_PER_OP)
            rows = [
                rocdl.readfirstlane(T.i32, row_idx_table[row_idx_base + first + fx.Int32(i)])
                for i in range_constexpr(ROWS_PER_OP)
            ]
            groups = _gather_static_groups(rows, idx_bits, tile_k, tile_k, a_dim1, lda)
            gather_groups.append((first * fx.Int32(LDS_ROW), groups))
    else:
        gt_a = _gv(gA_base, fx.Int64(a_row0) * lda_b, (tile_m, KB), (KB, 1))
        atom_a = _tdm(gt_a, m_ext, lda_b)

    def issue(s, kt):
        pa = _buf_ptr(s)
        koff = fx.Int64(kt) * fx.Int64(KB)
        if const_expr(gather):
            lds_base = fx.Int32(_bidx(pa))
            for q in range_constexpr(OPS_PER_WAVE):
                lds_off, (dg1, dg2, dg3) = gather_groups[q]
                dg0 = _gather_dgroup0(lds_base + lds_off, a_addr + koff, idx_bits)
                tdm_ops.tensor_load_gather(_GatherDesc(dg0, dg1, dg2, dg3))
        else:
            if wave == fx.Int32(0):
                fx.copy(atom_a, gt_a, _lv(pa, (tile_m, KB), (LDS_ROW, 1)), imm_offset=koff)
        if wave == fx.Int32(1 % num_waves):
            fx.copy(atom_b, gt_b, _lv(fx.add_offset(pa, B_DELTA), (tile_n, KB), (LDS_ROW, 1)), imm_offset=koff)

    wmb = wave_m * warp_tile_m
    wnb = wave_n * warp_tile_n
    lds_load_b128, _ = _make_lds_copy_ops(128)

    def _frag(buf, b0):
        v0 = Vec(lds_load_b128(buf, b0))
        v1 = Vec(lds_load_b128(buf, b0 + 32))
        return v0.shuffle(v1, list(range(8)))

    def load_a(buf, wm, ks):
        row = wmb + wm * WMMA_M + lane16
        return _frag(buf, fx.Int64(row * LDS_ROW + ks * KSB + kgrp * 16))

    def load_b(buf, wn, ks):
        col = wnb + wn * WMMA_N + lane16
        return _frag(buf, fx.Int64(B_DELTA + col * LDS_ROW + ks * KSB + kgrp * 16))

    wmma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fx.BFloat16, fx.Float32))
    c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_acc)]
    for cf in c_frags:
        cf.store(Vec.filled(8, 0.0, fx.Float32))

    def _rmem(n, v):
        t = fx.make_rmem_tensor(n, fx.Int32)
        t.store(v)
        return t

    KS_DS = (wmma_m_rep + wmma_n_rep) * 2

    def _load_ks(buf, ks):
        act = [_rmem(8, load_a(buf, wm, ks)) for wm in range_constexpr(wmma_m_rep)]
        wt = [_rmem(8, load_b(buf, wn, ks)) for wn in range_constexpr(wmma_n_rep)]
        return act, wt

    def _mma_ks(state):
        act, wt = state
        for wm in range_constexpr(wmma_m_rep):
            for wn_raw in range_constexpr(wmma_n_rep):
                wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                idx = wm * wmma_n_rep + wn
                fx.gemm(wmma_atom, c_frags[idx], wt[wn], act[wm], c_frags[idx])

    def compute_ktile(buf, prefetch_kt):
        """K_WS sub-steps; the last one is returned to run inside the barrier window."""
        ring = [None] * R
        ring[0] = _load_ks(buf, 0)
        for ks in range_constexpr(K_WS):
            if const_expr(ks + 1 < K_WS):
                ring[(ks + 1) % R] = _load_ks(buf, ks + 1)
            rocdl.s_wait_dscnt(min(K_WS - 1 - ks, R - 1) * KS_DS)
            if const_expr(ks == 0 and prefetch_kt is not None):
                rocdl.sched_barrier(0)
                issue(prefetch_kt % NUM_BUFFERS, prefetch_kt)
                rocdl.sched_barrier(0)
            if const_expr(ks < K_WS - 1):
                _mma_ks(ring[ks % R])
        rocdl.sched_dsrd(KS_DS)
        for _ks in range_constexpr(K_WS - 1):
            if const_expr(_ks + 1 < K_WS):
                rocdl.sched_dsrd(KS_DS)
            rocdl.sched_mfma(n_acc)
        rocdl.sched_barrier(0)
        return ring[(K_WS - 1) % R]

    def split_fence(tail):
        tdm_ops.tensor_wait(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier_signal(_WG_BARRIER)
        rocdl.sched_barrier(0)
        _mma_ks(tail)
        rocdl.sched_barrier(0)
        rocdl.s_barrier_wait(_WG_BARRIER)

    issue(0, 0)
    _pipeline_fence(outstanding=0)
    for kt in range(n_steady):
        buf = _bidx(_buf_ptr(kt % NUM_BUFFERS))
        split_fence(compute_ktile(buf, kt + 1))
    buf = _bidx(_buf_ptr(n_steady % NUM_BUFFERS))
    _mma_ks(compute_ktile(buf, None))

    # ---- epilogue: accumulators -> LDS (row-major tile) -> 16 B buffer stores ----
    accs = [c_frags[idx].load() for idx in range_constexpr(n_acc)]
    _pipeline_fence(outstanding=0)
    for wm in range_constexpr(wmma_m_rep):
        row_rel = wmb + wm * WMMA_M + lane16
        for wn in range_constexpr(wmma_n_rep):
            col_rel = wnb + wn * WMMA_N + kgrp * 8
            h = accs[wm * wmma_n_rep + wn].to(fx.BFloat16)
            fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row_rel * tile_n + col_rel) * EB)
    _workgroup_barrier()
    chunks_per_row = tile_n * EB // 16
    c_res = create_buffer_resource_from_addr(c_addr, num_records_bytes=m_ext * fx.Int32(ldc * EB))
    dropped = m_ext * fx.Int32(ldc // 2)  # first word past num_records: the store is dropped
    lds_c = fx.Int32(_bidx(base_ptr))
    for i in range_constexpr(tile_m * chunks_per_row // block):
        chunk = tid + fx.Int32(i * block)
        row = chunk // fx.Int32(chunks_per_row)
        c16 = chunk % fx.Int32(chunks_per_row)
        v = lds_load_b128(lds_c, row * fx.Int32(tile_n * EB) + c16 * fx.Int32(16))
        word = row * fx.Int32(ldc // 2) + c16 * fx.Int32(4)
        word = (c16 * fx.Int32(8) < n_ext).select(word, dropped)
        buffer_store(v, c_res, word, cache_modifier=c_cache_modifier)


# ── standalone launcher: one tile per (pool block, N block) ──────────────────


@flyc.jit
def _launch_mega_nt(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_row_idx: fx.Pointer,
    arg_tile_to_expert: fx.Pointer,
    arg_num_tile_blocks: fx.Pointer,
    stream: fx.Stream,
    i32_k: fx.Int32,
    i32_n: fx.Int32,
    i32_a_rows: fx.Int32,
    MAX_TILES: Constexpr[int],
    N_BLOCKS_N: Constexpr[int],
    LDA: Constexpr[int],
    LDB: Constexpr[int],
    LDC: Constexpr[int],
    B_GRP_ELEMS: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    group_m: Constexpr[int],
    gather: Constexpr[int],
    idx_bits: Constexpr[int],
):
    block = m_warp * n_warp * WAVE
    arena_bytes = tile_geometry(tile_m, tile_n, tile_k)[3]
    if arena_bytes > GFX1250_LDS_BYTES:
        raise ValueError(f"LDS arena {arena_bytes} B exceeds {GFX1250_LDS_BYTES} B")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel_mega_nt(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_row_idx: fx.Pointer,
        arg_tile_to_expert: fx.Pointer,
        arg_num_tile_blocks: fx.Pointer,
        i32_k: fx.Int32,
        i32_n: fx.Int32,
        i32_a_rows: fx.Int32,
    ):
        arena = fx.SharedAllocator(static=False)
        arena.allocate(arena_bytes)
        base_ptr = arena.base_ptr

        # Mega's tile order: GROUP_M super-tiles over the pool blocks that are live this call
        pid = fx.Int32(fx.block_idx.x)
        real_tiles = rocdl.readfirstlane(T.i32, _i32_table(arg_num_tile_blocks)[fx.Int32(0)])
        live = pid < real_tiles * fx.Int32(N_BLOCKS_N)
        num_pid_in_group = fx.Int32(group_m * N_BLOCKS_N)
        first_pid_m = (pid // num_pid_in_group) * fx.Int32(group_m)
        remaining_m = real_tiles - first_pid_m
        group_size_m = (remaining_m < fx.Int32(group_m)).select(remaining_m, fx.Int32(group_m))
        group_size_m = (group_size_m > fx.Int32(0)).select(group_size_m, fx.Int32(1))
        pid_in_group = pid % num_pid_in_group
        block_m = live.select(first_pid_m + pid_in_group % group_size_m, fx.Int32(0))
        block_n = live.select(pid_in_group // group_size_m, fx.Int32(0))
        block_m = rocdl.readfirstlane(T.i32, block_m)
        block_n = rocdl.readfirstlane(T.i32, block_n)
        expert = rocdl.readfirstlane(T.i32, _i32_table(arg_tile_to_expert)[block_m])

        blk_n = block_n * fx.Int32(tile_n)
        m_row = block_m * fx.Int32(tile_m)
        gA = fx.recast_iter(fx.Int8, arg_a)
        gB = fx.recast_iter(fx.Int8, arg_b)
        c_addr = (
            fx.Int64(fx.ptrtoint(fx.recast_iter(fx.Int8, arg_c)))
            + (fx.Int64(m_row) * fx.Int64(LDC) + fx.Int64(blk_n)) * fx.Int64(EB)
        )
        nt_gemm_tile(
            base_ptr,
            gA_base=gA,
            lda=LDA,
            a_row0=m_row,
            a_rows=i32_a_rows,
            row_idx_table=_i32_table(arg_row_idx),
            row_idx_base=m_row,
            gB_base=gB,
            ldb=LDB,
            b_slab=fx.Int64(expert) * fx.Int64(B_GRP_ELEMS * EB),
            blk_n=blk_n,
            c_addr=c_addr,
            ldc=LDC,
            m_rows=fx.Int32(tile_m),
            n_valid=i32_n - blk_n,
            i32_k=i32_k,
            live=live,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            m_warp=m_warp,
            n_warp=n_warp,
            gather=bool(gather),
            idx_bits=idx_bits,
        )

    kernel_mega_nt(
        arg_c, arg_a, arg_b, arg_row_idx, arg_tile_to_expert, arg_num_tile_blocks, i32_k, i32_n, i32_a_rows
    ).launch(grid=(MAX_TILES * N_BLOCKS_N, 1, 1), block=(block, 1, 1), stream=stream)


_launch_mega_nt.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}

_COMPILED: dict = {}
_CONSTEXPR0 = 10


def mega_grouped_gemm_nt_gfx1250(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    tile_to_expert: torch.Tensor,
    num_tile_blocks: torch.Tensor,
    row_idx: torch.Tensor | None = None,
    *,
    tile_m: int = 256,
    tile_n: int = 256,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 2,
    group_m: int = 16,
    idx_bits: int = 0,
) -> torch.Tensor:
    """``out[p] = a[row_idx[p]] @ b[e(p)].T`` per pool row ``p`` (``a[p]`` when ``row_idx`` is None).

    ``out`` is the pool, ``[num_pool_rows, N]`` with ``num_pool_rows`` a multiple of ``tile_m``;
    pool block ``t`` belongs to expert ``tile_to_expert[t]``, and only the first
    ``num_tile_blocks[0]`` blocks are computed. ``row_idx`` entries < 0 mark padding rows.
    """
    _require_gfx1250(a.device)
    G, N, K = b.shape
    pool_rows = out.shape[0]
    assert a.dtype == b.dtype == out.dtype == torch.bfloat16
    assert a.shape[1] == K and out.shape[1] == N and pool_rows % tile_m == 0
    assert K % tile_k == 0 and K // tile_k >= NUM_BUFFERS, "K must be a whole number of >= 2 k-tiles"
    assert a.is_contiguous() and b.is_contiguous() and out.is_contiguous()
    gather = row_idx is not None
    if not idx_bits:
        idx_bits = 16 if a.shape[0] < 0xFFFF else 32
    if gather:
        assert row_idx.dtype == torch.int32 and row_idx.numel() >= pool_rows
        assert idx_bits == 32 or a.shape[0] < 0xFFFF, "16-bit indices need A rows < 65535 (0xFFFF is padding)"
    else:
        row_idx = tile_to_expert  # unread
    args = (
        _as_ptr(out),
        _as_ptr(a),
        _as_ptr(b),
        _as_ptr(row_idx),
        _as_ptr(tile_to_expert),
        _as_ptr(num_tile_blocks),
        torch.cuda.current_stream(),
        K,
        N,
        a.shape[0],
        pool_rows // tile_m,
        (N + tile_n - 1) // tile_n,
        K,
        K,
        N,
        N * K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        group_m,
        int(gather),
        idx_bits,
    )
    key = args[_CONSTEXPR0:]
    fn = _COMPILED.get(key)
    if fn is None:
        fn = flyc.compile(_launch_mega_nt, *args)
        if fn is None:
            return out
        _COMPILED[key] = fn
    fn(*args)
    return out


@functools.lru_cache(maxsize=None)
def arena_bytes(tile_m: int = 256, tile_n: int = 256, tile_k: int = 128) -> int:
    return tile_geometry(tile_m, tile_n, tile_k)[3]
