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

"""Mega MoE GEMM tile for gfx1250: ``C[pool block] = A[rows] @ B[expert].T`` (NT), or
``A[rows] @ B[expert]`` with ``b_nn`` (B stored ``[K, N]``, the backward's dgrad).

One workgroup computes one ``tile_m x tile_n`` block of one 256-row pool block, the unit Mega
schedules on. The pipeline is the shipped configuration of
``grouped_gemm_bf16_kernel_gfx1250._launch_grouped_gemm_bf16_nt`` (TDM global -> LDS, two
buffers, WMMA 16x16x32, ``split_bar`` with one deferred sub-step, a two-deep fragment ring,
its keyed ``lds_cosched = 3`` / ``issue_first`` schedule), with two changes Mega needs:

* **A can be gathered.** In ``gather`` mode the A rows come from a row-index table (pool row ->
  dispatched token). Every lane copies 16-byte chunks of them straight into LDS with
  ``global_load_async_to_lds_b128`` (16 lanes per 256-byte row, 16 rows per instruction,
  waited with ``s_wait_asynccnt``). An index outside ``[0, a_rows)`` -- the pool's padding rows
  carry -1 -- reads one of the tile's own rows instead; the row's output is padding that
  nothing downstream reads. No other row may be read: one the dispatch is still writing leaves
  stale lines in the WGP's vector cache, which the tile owning that row then reads. TDM gather
  loads (``tensor_load_to_lds`` gather mode, 16 rows per instruction) need no VMEM issue slots
  and are about 2.5% faster, but on gfx1250 they now and
  then corrupt the LDS writes of neighbouring workgroups -- silently, about one fused launch in
  500, triggered by gathers that barely touch memory (out-of-range, repeated or hot rows) --
  so the tile does not use them.
* **B can be NN.** With ``b_nn`` the B stage is ``[tile_k, tile_n]`` as stored and its fragments
  come out of the hardware LDS transpose read ``ds_load_tr16_b128``, the native NN path of the
  grouped kernel: no transposed copy of the weight.
* **The epilogue stores with plain buffer stores**, staged through LDS so every lane writes 16
  contiguous bytes. The grouped kernel's ``tensor_store_from_lds`` epilogue is the remaining
  suspect of its rare silent wrong result, so Mega does not use it.

A dead tile (past the device-side tile count) is not branched around -- a dynamic branch over the
TDM pipeline is what the grouped kernel's ``dead_tile_exit`` found to fault intermittently.
It runs with every extent at 0: no k-tile, no TDM traffic, a store with ``num_records`` 0. A
gathered A's prologue loads sit behind a uniform ``live`` branch, the one branch in the tile.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl._mlir.dialects import vector
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Constexpr, T, as_ir_value
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (
    _WG_BARRIER,
    _as_ptr,
    _i32_table,
    _make_lds_copy_ops,
    _nn_b_pad,
    _pipeline_fence,
    _require_gfx1250,
    _workgroup_barrier,
)
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
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
NUM_XCD = 8  # a workgroup runs on XCD bid % NUM_XCD
XCD_BAND = 16


def tile_geometry(tile_m: int, tile_n: int, tile_k: int, b_nn: bool = False):
    """(LDS row bytes, A stage bytes, buffer pitch, arena bytes) of one tile configuration.

    The B stage is ``tile_n`` rows of ``tile_k`` (NT) or ``tile_k`` rows of ``tile_n`` (``b_nn``)."""
    lds_row = tile_k * EB + LDS_PAD
    stage = ((tile_m * lds_row + 15) // 16) * 16
    b_rows, b_row = (tile_k, tile_n * EB + _nn_b_pad(tile_n, EB)) if b_nn else (tile_n, lds_row)
    stage_b = ((b_rows * b_row + 15) // 16) * 16
    pitch = ((stage + stage_b + 1023) // 1024) * 1024
    arena = max(NUM_BUFFERS * pitch, tile_m * (tile_n * EB + LDS_PAD))
    return lds_row, stage, pitch, arena


def persist_geometry(tile_m: int, tile_n: int, tile_k: int, b_nn: bool = False):
    """(C staging offset, flag word offset, arena bytes) of a tile run back to back in one workgroup.

    The next tile's k-tile 0 lands in buffer 0 while this tile's epilogue runs, so the C staging
    starts at buffer 1; the caller's flag word sits past both buffers."""
    _, _, pitch, _ = tile_geometry(tile_m, tile_n, tile_k, b_nn)
    c_bytes = tile_m * (tile_n * EB + LDS_PAD)
    flag_off = max(NUM_BUFFERS * pitch, pitch + c_bytes)
    return pitch, flag_off, flag_off + 16


def resolve_lds_cosched(
    lds_cosched: int, issue_first: int, tile_m: int, tile_n: int, m_warp: int, n_warp: int
):
    """``lds_cosched`` as ``nt_gemm_tile`` takes it: -1 turns it on where the tile allows it.

    It issues one fragment read per matrix op of the previous sub-step, so it needs ``issue_first``
    and fewer reads per sub-step (two per fragment row and column) than matrix ops."""
    m_rep, n_rep = tile_m // m_warp // WMMA_M, tile_n // n_warp // WMMA_N
    allowed = bool(issue_first) and 2 * (m_rep + n_rep) < m_rep * n_rep
    if lds_cosched == -1:
        return int(allowed)
    if lds_cosched not in (0, 1):
        raise ValueError(f"lds_cosched must be -1 (auto), 0 or 1, got {lds_cosched}")
    if lds_cosched and not allowed:
        raise ValueError(
            f"lds_cosched=1 needs issue_first=1 and fewer fragment reads per sub-step than matrix ops, "
            f"got issue_first={int(issue_first)} and {2 * (m_rep + n_rep)} reads for {m_rep * n_rep} "
            f"matrix ops (tile {tile_m}x{tile_n}, {m_warp}x{n_warp} waves)"
        )
    return lds_cosched


def head_pipe_ok(
    head_pipe: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    lds_cosched: int,
    gather: bool,
) -> bool:
    """Whether ``nt_gemm_tile`` takes ``head_pipe`` (> 0) on this tile.

    The sub-step-0 reads ride on the co-scheduled loop, one per deferred matrix op after the first
    ``head_pipe``, and must not land in the deferred sub-step's registers (two sub-steps per k-tile).
    Dense A only."""
    m_rep, n_rep = tile_m // m_warp // WMMA_M, tile_n // n_warp // WMMA_N
    return (
        head_pipe > 0
        and bool(lds_cosched)
        and not gather
        and tile_k // WMMA_K >= 2
        and head_pipe + 2 * (m_rep + n_rep) <= m_rep * n_rep
    )


def check_wmma_b2b(wmma_b2b: int, gather: bool):
    """``wmma_b2b`` (the WMMA arbitration stall off) is for a dense A only.

    With the stall off and the async A gather issued at the k-tile head, a wave's last async loads
    of a k-tile now and then land in LDS after the barrier that covers them, and the tile reads
    the buffer's previous k-tile: about one NN dispatch GEMM launch in 25000 at EP4."""
    if wmma_b2b not in (0, 1):
        raise ValueError(f"wmma_b2b must be 0 or 1, got {wmma_b2b}")
    if wmma_b2b and gather:
        raise ValueError("wmma_b2b=1 needs a dense A: with the async A gather it races the k-tile barrier")


def check_gather_rows(a_rows: int, lda: int, tile_m: int, tile_k: int):
    """The async A gather's limits: some row to read, and 32-bit byte offsets into A."""
    assert a_rows > 0, "a gathered A needs a row"
    assert a_rows * lda * EB + tile_m * (tile_k * EB + LDS_PAD) < 2**32, "A too big for 32-bit row offsets"


def pool_tile_order(pid, n_blocks_m, n_blocks_n: int, group_m: int, num_xcd: int):
    """``(block_m, block_n)`` of grid tile ``pid`` over ``n_blocks_m`` (device-side) x ``n_blocks_n``.

    ``group_m`` pool blocks share each N block in turn, and with ``num_xcd`` > 1 every
    ``XCD_BAND`` consecutive tiles of that order go to one XCD (workgroup ``pid`` runs on XCD
    ``pid % num_xcd``, so ``pid`` must start on an XCD boundary): an XCD reads few of an expert's
    weight columns."""
    if num_xcd > 1:
        span = fx.Int32(XCD_BAND * num_xcd)
        n_full = (n_blocks_m * fx.Int32(n_blocks_n)) // span * span
        r = pid % span
        banded = pid // span * span + r % fx.Int32(num_xcd) * fx.Int32(XCD_BAND) + r // fx.Int32(num_xcd)
        pid = (pid < n_full).select(banded, pid)
    if group_m <= 1:
        return pid // fx.Int32(n_blocks_n), pid % fx.Int32(n_blocks_n)
    num_pid_in_group = fx.Int32(group_m * n_blocks_n)
    first_m = pid // num_pid_in_group * fx.Int32(group_m)
    size_m = n_blocks_m - first_m
    size_m = (size_m < fx.Int32(group_m)).select(size_m, fx.Int32(group_m))
    # a dead tile past the pool decodes to some block; it is masked anyway
    size_m = (size_m > fx.Int32(0)).select(size_m, fx.Int32(1))
    pid_in_group = pid % num_pid_in_group
    return first_m + pid_in_group % size_m, pid_in_group // size_m


# ── TDM gather-store descriptors ─────────────────────────────────────────────
# Built by hand rather than through tdm_ops.make_tensor_gather_descriptor, which wants the LDS
# side as a memref; this tile's LDS is a pointer arena. Field layout per CDNA5 ISA 4.10.3.2,
# the same bits tdm_ops packs.


def _c32(v):
    return arith.constant(v, type=T.i32)


def _gather_dgroup0(lds_addr_i32, global_addr_i64, idx_bits: int):
    """dgroup0: gather-mode predicate, LDS byte address and the 57-bit global base."""
    i32 = ir.IntegerType.get_signless(32)
    g = _raw(global_addr_i64)
    lo = std_arith.TruncIOp(i32, g).result
    hi = std_arith.TruncIOp(i32, std_arith.ShRUIOp(g, _raw(fx.Int64(32))).result).result
    hi = arith.ori(hi, _c32(1 << 31))
    pred = _c32(1 | ((1 if idx_bits == 32 else 0) << 30) | (1 << 31))
    return vector.from_elements(T.vec(4, T.i32), [as_ir_value(v) for v in (pred, _raw(lds_addr_i32), lo, hi)])


class _GatherDesc:
    def __init__(self, dgroup0, dgroup1, dgroup2, dgroup3):
        self.dgroup0, self.dgroup1, self.dgroup2, self.dgroup3 = dgroup0, dgroup1, dgroup2, dgroup3


def _readlane_i32(v, lane: int):
    """Lane ``lane`` of a per-lane i32, as a wave-uniform SGPR value."""
    op = rocdl.readlane(res=T.i32, src=_raw(v), lane=_raw(fx.Int32(lane)))
    return ArithValue(op.result if hasattr(op, "result") else op)


# rows per TDM gather-store with 32-bit row indices
_SCATTER_ROWS = 8


def _scatter_dgroups(row_indices, row_cols, num_rows: int, row_width: int, stride: int):
    """dgroup1..3 of a gather-store of dense LDS rows: ``row_cols`` (i32) columns of each row kept
    out of ``row_width``, row indices past ``num_rows`` dropped, rows ``stride`` elements apart."""
    cols = _raw(row_cols)
    s1 = arith.shli(arith.andi(cols, _c32(0xFFFF)), _c32(16))
    s2 = arith.ori(arith.andi(arith.shrui(cols, _c32(16)), _c32(0xFFFF)), _c32((num_rows & 0xFFFF) << 16))
    s3 = _c32(((num_rows >> 16) & 0xFFFF) | (row_width << 16))
    words = (
        _c32((EB.bit_length() - 1) << 16),
        s1,
        s2,
        s3,
        _c32(len(row_indices)),
        _c32(stride),
        _c32(0),
        _c32(0),
    )
    dgroup1 = vector.from_elements(T.vec(8, T.i32), [as_ir_value(v) for v in words])
    dgroup2 = vector.from_elements(T.vec(4, T.i32), [as_ir_value(_raw(w)) for w in row_indices[:4]])
    dgroup3 = vector.from_elements(T.vec(4, T.i32), [as_ir_value(_raw(w)) for w in row_indices[4:]])
    return dgroup1, dgroup2, dgroup3


# ── the tile ─────────────────────────────────────────────────────────────────

# `compute_ktile`'s prefetch slot taking the next tile's k-tile 0
_NEXT_TILE_K0 = object()


@ASTRewriter.transform
def issue_tile_k0(
    base_ptr,
    *,
    gA_base,
    lda: int,
    a_row0,
    gB_base,
    ldb: int,
    b_slab,
    blk_n,
    m_rows,
    n_valid,
    live,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_waves: int,
    b_nn: bool = False,
    i32_n=None,
):
    """Issue k-tile 0 of a dense-A ``nt_gemm_tile`` into buffer 0: the loads of its prologue, same
    arguments. A ``live`` False tile loads nothing (every extent 0)."""
    KB = tile_k * EB
    LDS_ROW, STAGE_A, _pitch, _arena = tile_geometry(tile_m, tile_n, tile_k, b_nn)
    LDS_B_ROW = tile_n * EB + _nn_b_pad(tile_n, EB)
    wave = rocdl.readfirstlane(T.i32, fx.Int32(fx.thread_idx.x) // WAVE)
    zero32 = fx.Int32(0)
    lda_b = fx.Int64(lda * EB)
    ldb_b = fx.Int64(ldb * EB)

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

    gt_a = _gv(gA_base, fx.Int64(a_row0) * lda_b, (tile_m, KB), (KB, 1))
    atom_a = _tdm(gt_a, live.select(m_rows, zero32), lda_b)
    if const_expr(b_nn):
        blk_n_eff = (blk_n > i32_n - fx.Int32(tile_n)).select(i32_n - fx.Int32(tile_n), blk_n)
        gt_b = _gv(gB_base, b_slab + fx.Int64(blk_n_eff) * fx.Int64(EB), (tile_k, tile_n * EB), (ldb_b, 1))
        atom_b = fx.rocdl.make_tdm_atom(
            gt_b,
            [live.select(fx.Int32(tile_k), zero32), fx.Int32(tile_n * EB)],
            strides=[ldb_b, None],
            num_warps=1,
            pad_interval=tile_n * EB,
            pad_amount=LDS_B_ROW - tile_n * EB,
            cache_modifier=0,
            early_timeout=True,
        )
    else:
        gt_b = _gv(gB_base, b_slab + fx.Int64(blk_n) * ldb_b, (tile_n, KB), (KB, 1))
        atom_b = _tdm(gt_b, live.select(n_valid, zero32), ldb_b)
    if wave == fx.Int32(0):
        fx.copy(atom_a, gt_a, _lv(base_ptr, (tile_m, KB), (LDS_ROW, 1)), imm_offset=fx.Int64(0))
    if wave == fx.Int32(1 % num_waves):
        if const_expr(b_nn):
            fx.copy(
                atom_b, gt_b, _lv(fx.add_offset(base_ptr, STAGE_A), (tile_k, tile_n * EB), (LDS_B_ROW, 1))
            )
        else:
            fx.copy(
                atom_b,
                gt_b,
                _lv(fx.add_offset(base_ptr, STAGE_A), (tile_n, KB), (LDS_ROW, 1)),
                imm_offset=fx.Int64(0),
            )


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
    b_nn: bool = False,
    i32_n=None,
    c_rows=None,
    c_rank_bases=None,
    c_num_rows: int = 0,
    c_bias=None,
    issue_first: bool = False,
    lds_cosched: bool = False,
    wmma_b2b: bool = False,
    k0_issued: bool = False,
    next_tile=None,
    c_off: int = 0,
    head_pipe: int = 0,
):
    """One ``tile_m x tile_n`` block; must be called by all ``m_warp * n_warp`` waves.

    ``base_ptr`` is the LDS arena (``tile_geometry(...)[3]`` bytes). A is ``gA_base`` (Int8
    iterator) with row stride ``lda`` elements: rows ``a_row0 ..`` when dense, rows
    ``row_idx_table[row_idx_base ..][:tile_m]`` (an i32 iterator) when
    ``gather``, bounded by ``a_rows``. B is ``gB_base + b_slab`` with row stride ``ldb``; this
    block reads its rows ``blk_n ..``. With ``b_nn`` B is ``[K, N]`` with row stride ``ldb``, this
    block reads its columns ``blk_n ..``, and ``i32_n`` (N, at least ``tile_n``) bounds that
    read. C is written at byte address ``c_addr`` with row stride
    ``ldc``, ``m_rows`` rows and ``n_valid`` columns kept. ``live`` (wave-uniform i1) turns the
    tile into a no-op. ``c_cache_modifier`` is the C stores' cache policy (``CPOL_DEVICE_STREAM``
    when another workgroup consumes C in the same launch).

    ``c_rows`` scatters C by row instead (``c_addr`` unused), with 8-row TDM gather-stores out
    of LDS: C lives on several ranks, ``c_rank_bases[r]`` the i64 byte address of rank ``r``'s
    row 0, ``c_num_rows`` rows of ``ldc`` elements. Called once with each lane's tile row (lane
    ``l`` of wave ``w`` holds row ``8 * (w + num_waves * (l // 8)) + l % 8``, so lanes
    ``8j .. 8j + 7`` hold the wave's ``j``-th group of 8 rows), it returns
    ``(rank, row, ok, extra)``: where the row goes and whether to store it at all. The tile
    returns that tuple, so the caller can reuse ``extra`` without loading it again. ``c_bias``
    ``(addr, scale)`` adds ``bias * scale(tile row)`` in f32 before the bf16 rounding, ``addr``
    the i64 address of the expert's bias row.

    ``issue_first`` issues the next k-tile's loads at the top of the k-tile body, ahead of its
    fragment reads; the buffer they fill was drained by the previous k-tile's fence.
    ``lds_cosched`` (needs ``issue_first``; ``resolve_lds_cosched``) is the grouped kernel's
    ``lds_cosched = 3``: each sub-step is one scheduling region that waits for its own fragments
    only and reads the next sub-step's one per matrix op, holding the running sub-step's
    fragments live to its end. ``wmma_b2b`` turns the WMMA arbitration stall off; not with a
    gathered A. Same results with any of them.

    Tiles run back to back in one workgroup (dense A, an even number of k-tiles): ``k0_issued``
    says this tile's k-tile 0 is already in flight to buffer 0 (``issue_tile_k0``), ``next_tile``
    (``issue_tile_k0``'s per-tile arguments: ``a_row0``, ``b_slab``, ``blk_n``, ``n_valid``,
    ``live``) issues the next tile's at the top of this tile's last k-tile, which runs from buffer
    1, and ``c_off`` moves the C staging off buffer 0 (``persist_geometry``).

    ``head_pipe`` > 0 (``head_pipe_ok``) reads each k-tile's sub-step-0 fragments in the barrier
    window before it, behind the barrier wait and the first ``head_pipe`` matrix ops of the deferred
    sub-step, and interleaved with the rest of them, instead of at the top of the k-tile where the
    first matrix op waits for them. Same results.
    """
    KB = tile_k * EB
    KSB = WMMA_K * EB
    K_WS = tile_k // WMMA_K
    warp_tile_m, warp_tile_n = tile_m // m_warp, tile_n // n_warp
    wmma_m_rep, wmma_n_rep = warp_tile_m // WMMA_M, warp_tile_n // WMMA_N
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    LDS_ROW, STAGE_A, PITCH, _arena = tile_geometry(tile_m, tile_n, tile_k, b_nn)
    LDS_B_ROW = tile_n * EB + _nn_b_pad(tile_n, EB)
    B_DELTA = STAGE_A
    R = 2  # fragment ring depth
    # Not an assert: without issue_first the co-scheduled loop never issues the next k-tile's
    # loads, so `python -O` would turn a bad combination into stale LDS reads.
    resolve_lds_cosched(int(lds_cosched), issue_first, tile_m, tile_n, m_warp, n_warp)
    check_wmma_b2b(wmma_b2b, gather)
    assert ldc % 2 == 0, "the epilogue stores whole i32 words"
    assert not (gather and (k0_issued or next_tile is not None)), "chained tiles have a dense A"
    assert next_tile is None or num_waves > 1, "the epilogue fence counts one chained load per wave"
    assert head_pipe == 0 or head_pipe_ok(
        head_pipe, tile_m, tile_n, tile_k, m_warp, n_warp, lds_cosched, gather
    ), f"head_pipe={head_pipe} needs a co-scheduled dense-A tile and room among the matrix ops"

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

    if const_expr(b_nn):
        # N is B's dim 1 here, and a dim-1 extent clamps a load's data but not its address: a
        # short last block reads the in-row window [N - tile_n, N) and shifts its fragment
        # columns by n_back instead
        blk_n_eff = (blk_n > i32_n - fx.Int32(tile_n)).select(i32_n - fx.Int32(tile_n), blk_n)
        n_back = blk_n - blk_n_eff
        b_rows_ext = live.select(fx.Int32(tile_k), zero32)
    else:
        gt_b = _gv(gB_base, b_slab + fx.Int64(blk_n) * ldb_b, (tile_n, KB), (KB, 1))
        atom_b = _tdm(gt_b, n_ext, ldb_b)
    if const_expr(gather):
        # lane tid copies chunk tid % CH of row tid // CH, and RPI rows further per instruction
        CH = KB // 16
        RPI = block // CH
        N_LD = tile_m // RPI
        assert block % CH == 0 and tile_m % RPI == 0
        # an instruction offset of 64 KiB or more makes the async loads read the wrong rows
        assert (N_LD - 1) * RPI * LDS_ROW < 1 << 16, "the async A gather needs more waves at this tile_k"
        A_BIAS = tile_m * LDS_ROW
        c16 = tid % fx.Int32(CH)
        rsub = tid // fx.Int32(CH)
        a_addr = fx.Int64(fx.ptrtoint(gA_base))
        a_lane_lds = rsub * fx.Int32(LDS_ROW) + c16 * fx.Int32(16)
        # K-invariant 32-bit row offsets. An instruction offset moves the global and the LDS
        # address alike, so load i's carries its LDS step, taken back out of its row offset here
        # (A_BIAS keeps that unsigned). A padding row reads the lane's row of load 0, or the
        # tile's first row if that is padding too: only this tile's rows are known written
        # (before its expert's flag), and reading a row the dispatch is still writing can leave
        # stale lines in the WGP's vector cache that the workgroup owning the row then hits.
        # One shared fallback row would put every padding row's loads on one address.
        first = rocdl.readfirstlane(T.i32, row_idx_table[row_idx_base])
        fallback = ((first >= zero32) & (first < a_rows)).select(first, zero32)
        a_off = []
        for i in range_constexpr(N_LD):
            pos = fx.Int32(i * RPI) + rsub
            idx = row_idx_table[row_idx_base + pos]
            ok = (idx >= zero32) & (idx < a_rows)
            row = ok.select(idx, fallback)
            if const_expr(i == 0):
                fallback = row
            a_off.append(row * fx.Int32(lda * EB) + c16 * fx.Int32(16) + fx.Int32(A_BIAS - i * RPI * LDS_ROW))
    else:
        gt_a = _gv(gA_base, fx.Int64(a_row0) * lda_b, (tile_m, KB), (KB, 1))
        atom_a = _tdm(gt_a, m_ext, lda_b)

    def issue(s, kt, prologue=False):
        pa = _buf_ptr(s)
        koff = fx.Int64(kt) * fx.Int64(KB)
        if const_expr(gather):
            # k goes into the 32-bit offset so its zext stays in the loop and the loads keep the
            # SADDR form; a hoisted 64-bit offset costs a v_add_u64 per load and twice the VGPRs.
            # No per-row branch around a load either: that serializes them
            i64 = ir.IntegerType.get_signless(64)
            a_base = _raw(a_addr - fx.Int64(A_BIAS))
            lds = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int32(_bidx(pa)) + a_lane_lds))
            koff32 = fx.Int32(kt) * fx.Int32(KB)

            def _a_loads():
                for i in range_constexpr(N_LD):
                    g = std_arith.AddIOp(
                        a_base, std_arith.ExtUIOp(i64, _raw(a_off[i] + koff32)).result
                    ).result
                    gp = _llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), g)
                    _rocdl.global_load_async_to_lds_b128(gp, lds, i * RPI * LDS_ROW)

            if const_expr(prologue):
                # a dead tile waits on no flag, so it reads no A at all (only the prologue runs
                # for it); its LDS stays garbage and its store is dropped
                if live:
                    _a_loads()
            else:
                _a_loads()
        else:
            if wave == fx.Int32(0):
                fx.copy(atom_a, gt_a, _lv(pa, (tile_m, KB), (LDS_ROW, 1)), imm_offset=koff)
        if wave == fx.Int32(1 % num_waves):
            if const_expr(b_nn):
                # k is B's row axis: fold it into the base (a TDM extent counts from imm_offset)
                gt_bk = _gv(
                    gB_base,
                    b_slab + fx.Int64(kt) * fx.Int64(tile_k) * ldb_b + fx.Int64(blk_n_eff) * fx.Int64(EB),
                    (tile_k, tile_n * EB),
                    (ldb_b, 1),
                )
                atom_bk = fx.rocdl.make_tdm_atom(
                    gt_bk,
                    [b_rows_ext, fx.Int32(tile_n * EB)],
                    strides=[ldb_b, None],
                    num_warps=1,
                    pad_interval=tile_n * EB,
                    pad_amount=LDS_B_ROW - tile_n * EB,
                    cache_modifier=0,
                    early_timeout=True,
                )
                fx.copy(
                    atom_bk, gt_bk, _lv(fx.add_offset(pa, B_DELTA), (tile_k, tile_n * EB), (LDS_B_ROW, 1))
                )
            else:
                fx.copy(
                    atom_b, gt_b, _lv(fx.add_offset(pa, B_DELTA), (tile_n, KB), (LDS_ROW, 1)), imm_offset=koff
                )

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

    if const_expr(b_nn):
        # ds_load_tr16_b128: lanes act in groups of 8, lane j of a group points at 8 consecutive
        # elements of LDS row (reduction k) ks*32 + kgrp*8 + j, and lane l gets column l of
        # that 8x8 block -- 8 consecutive k at output column wnb + wn*16 + lane16; a second read
        # 16 rows down gives k + 16.. (the NT loader's fragment order)
        l8 = lane % 8
        hi8 = lane16 // 8
        b_lane = (
            (l8 + kgrp * fx.Int32(8)) * fx.Int32(LDS_B_ROW)
            + (wnb + hi8 * fx.Int32(8) + n_back) * fx.Int32(EB)
            + fx.Int32(B_DELTA)
        )
        vec8 = ir.VectorType.get([8], T.bf16)
        ptr3 = ir.Type.parse("!llvm.ptr<3>")

        def _tr(addr, const_off):
            return Vec(_rocdl.ds_load_tr16_b128(vec8, _llvm.inttoptr(ptr3, _raw(addr + fx.Int32(const_off)))))

        def load_b(buf, wn, ks):
            addr = fx.Int32(buf) + b_lane
            base = ks * WMMA_K * LDS_B_ROW + wn * WMMA_N * EB
            return (
                _tr(addr, base).shuffle(_tr(addr, base + 16 * LDS_B_ROW), list(range(16))).bitcast(fx.Int32)
            )

    else:

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

    # `_mma_ks`'s order, to split one sub-step's matrix ops
    MMA_ORDER = [
        (wm, (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw)
        for wm in range_constexpr(wmma_m_rep)
        for wn_raw in range_constexpr(wmma_n_rep)
    ]

    def _mma_range(state, lo, hi):
        act, wt = state
        for wm, wn in MMA_ORDER[lo:hi]:
            idx = wm * wmma_n_rep + wn
            fx.gemm(wmma_atom, c_frags[idx], wt[wn], act[wm], c_frags[idx])

    def _load_into(buf, ks, dst):
        act, wt = dst
        for wm in range_constexpr(wmma_m_rep):
            act[wm].store(load_a(buf, wm, ks))
        for wn in range_constexpr(wmma_n_rep):
            wt[wn].store(load_b(buf, wn, ks))

    def _hold_live(state):
        # Otherwise the allocator hands the next sub-step's reads registers an in-flight WMMA still
        # reads, and expert scheduling covers each such overwrite with a full VALU-dependency wait.
        # `llvm.fake.use` emits nothing.
        act, wt = state
        for t in act + wt:
            _llvm.call_intrinsic(None, "llvm.fake.use", [_raw(t.load())], [], [])

    def _prefetch(kt):
        if const_expr(kt is _NEXT_TILE_K0):
            issue_tile_k0(
                base_ptr,
                gA_base=gA_base,
                lda=lda,
                gB_base=gB_base,
                ldb=ldb,
                m_rows=m_rows,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                num_waves=num_waves,
                b_nn=b_nn,
                i32_n=i32_n,
                **next_tile,
            )
        else:
            issue(kt % NUM_BUFFERS, kt)

    def compute_ktile(buf, prefetch_kt, head=None):
        """K_WS sub-steps; the last one is returned to run inside the barrier window. ``head``
        (``head_pipe``) holds sub-step 0, read in the previous barrier window."""
        ring = [None] * R
        if const_expr(issue_first and prefetch_kt is not None):
            rocdl.sched_barrier(0)
            _prefetch(prefetch_kt)
            rocdl.sched_barrier(0)
        if const_expr(head is None):
            ring[0] = _load_ks(buf, 0)
        else:
            ring[0] = head
        if const_expr(lds_cosched):
            if const_expr(head is None):
                rocdl.sched_dsrd(KS_DS)
            rocdl.sched_barrier(0)
            for ks in range_constexpr(K_WS):
                # only this sub-step's reads are in flight here (R = 2)
                rocdl.s_wait_dscnt(0)
                if const_expr(ks + 1 < K_WS):
                    ring[(ks + 1) % R] = _load_ks(buf, ks + 1)
                if const_expr(ks < K_WS - 1):
                    _mma_ks(ring[ks % R])
                    for _j in range_constexpr(KS_DS):
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(1)
                    rocdl.sched_mfma(n_acc - KS_DS)
                    # after the group hints: placed before them, the hints are ignored
                    _hold_live(ring[ks % R])
                rocdl.sched_barrier(0)
            return ring[(K_WS - 1) % R]
        for ks in range_constexpr(K_WS):
            if const_expr(ks + 1 < K_WS):
                ring[(ks + 1) % R] = _load_ks(buf, ks + 1)
            rocdl.s_wait_dscnt(min(K_WS - 1 - ks, R - 1) * KS_DS)
            if const_expr(ks == 0 and prefetch_kt is not None and not issue_first):
                rocdl.sched_barrier(0)
                _prefetch(prefetch_kt)
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
        if const_expr(gather):
            _rocdl.s_wait_asynccnt(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier_signal(_WG_BARRIER)
        rocdl.sched_barrier(0)
        _mma_ks(tail)
        rocdl.sched_barrier(0)
        rocdl.s_barrier_wait(_WG_BARRIER)

    def head_fence(tail, nxt_buf, head):
        """``split_fence`` that also reads the next k-tile's sub-step 0 (in ``nxt_buf``) into ``head``:
        the first ``head_pipe`` matrix ops of ``tail`` cover the barrier wait, the rest the reads."""
        tdm_ops.tensor_wait(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier_signal(_WG_BARRIER)
        rocdl.sched_barrier(0)
        _mma_range(tail, 0, head_pipe)
        rocdl.sched_mfma(head_pipe)
        rocdl.sched_barrier(0)
        rocdl.s_barrier_wait(_WG_BARRIER)
        rocdl.sched_barrier(0)
        _load_into(nxt_buf, 0, head)
        _mma_range(tail, head_pipe, n_acc)
        for _j in range_constexpr(KS_DS):
            rocdl.sched_dsrd(1)
            rocdl.sched_mfma(1)
        rocdl.sched_mfma(n_acc - head_pipe - KS_DS)
        _hold_live(tail)
        rocdl.sched_barrier(0)

    if const_expr(wmma_b2b):
        # SCHED_MODE bit 2: no arbitration stall after a WMMA, without which a read placed between
        # two WMMAs cannot issue until the first has executed. After the index-table reads (placed
        # ahead of them it turns their scalar loads into vector loads), ahead of every matrix op.
        rocdl.disable_xdl_arb_stall()
    if const_expr(not k0_issued):
        issue(0, 0, prologue=True)
    if const_expr(gather):
        _rocdl.s_wait_asynccnt(0)
    _pipeline_fence(outstanding=0)
    head = None
    if const_expr(head_pipe):
        head = (
            [fx.make_rmem_tensor(8, fx.Int32) for _ in range_constexpr(wmma_m_rep)],
            [fx.make_rmem_tensor(8, fx.Int32) for _ in range_constexpr(wmma_n_rep)],
        )
        _load_into(_bidx(_buf_ptr(0)), 0, head)
        rocdl.sched_barrier(0)
    for kt in range(n_steady):
        buf = _bidx(_buf_ptr(kt % NUM_BUFFERS))
        if const_expr(head_pipe):
            head_fence(compute_ktile(buf, kt + 1, head), _bidx(_buf_ptr((kt + 1) % NUM_BUFFERS)), head)
        else:
            split_fence(compute_ktile(buf, kt + 1))
    buf = _bidx(_buf_ptr(n_steady % NUM_BUFFERS))
    # the next tile's k-tile 0 goes to buffer 0, which the last k-tile (buffer 1) leaves alone
    last_prefetch = None
    epi_outstanding = 0
    if const_expr(next_tile is not None):
        last_prefetch = _NEXT_TILE_K0
        epi_outstanding = 1
    _mma_ks(compute_ktile(buf, last_prefetch, head))

    # ---- epilogue: accumulators -> LDS (row-major tile) -> 16 B buffer stores ----
    chunks_per_row = tile_n * EB // 16
    # the 16 lanes of a half-wave store one column block of 16 rows: padded rows land them on
    # 16 different bank groups instead of one
    c_lds_row = tile_n * EB + LDS_PAD
    rows_out = None
    if const_expr(c_rows is not None):
        assert tile_m == num_waves * WAVE and WAVE % _SCATTER_ROWS == 0, (
            "a scattered C wants one row per lane"
        )
        # issued before the accumulators go to LDS, which hides the index loads
        rows_out = c_rows(
            fx.Int32(_SCATTER_ROWS) * (wave + fx.Int32(num_waves) * (lane // fx.Int32(_SCATTER_ROWS)))
            + lane % fx.Int32(_SCATTER_ROWS)
        )
    accs = [c_frags[idx].load() for idx in range_constexpr(n_acc)]
    f32x8 = T.vec(8, T.f32)
    bf16x8 = T.vec(8, T.bf16)
    if const_expr(c_bias is not None):
        bias_addr, bias_scale = c_bias
        # columns past N read past num_records as 0
        bias_res = create_buffer_resource_from_addr(
            bias_addr + fx.Int64(blk_n) * fx.Int64(EB), num_records_bytes=n_ext * fx.Int32(EB)
        )
        bias_words = [
            buffer_load(bias_res, (wnb + wn * WMMA_N + kgrp * 8) // 2, vec_width=4, dtype=T.i32)
            for wn in range_constexpr(wmma_n_rep)
        ]
        bias_scales = [bias_scale(wmb + wm * WMMA_M + lane16) for wm in range_constexpr(wmma_m_rep)]
    # waves 0 and 1 may have the next tile's k-tile 0 in flight
    _pipeline_fence(outstanding=epi_outstanding)
    for wm in range_constexpr(wmma_m_rep):
        row_rel = wmb + wm * WMMA_M + lane16
        for wn in range_constexpr(wmma_n_rep):
            col_rel = wnb + wn * WMMA_N + kgrp * 8
            acc = accs[wm * wmma_n_rep + wn]
            if const_expr(c_bias is not None):
                bias = std_arith.ExtFOp(f32x8, vector.bitcast(bf16x8, _raw(bias_words[wn]))).result
                term = std_arith.MulFOp(bias, vector.broadcast(f32x8, _raw(bias_scales[wm]))).result
                acc = Vec(std_arith.AddFOp(as_ir_value(acc), term).result)
            h = acc.to(fx.BFloat16)
            if const_expr(c_off):
                fx.ptr_store(h.bitcast(fx.Int8), base_ptr + (row_rel * c_lds_row + col_rel * EB + c_off))
            else:
                fx.ptr_store(h.bitcast(fx.Int8), base_ptr + row_rel * c_lds_row + col_rel * EB)
    _workgroup_barrier()
    lds_c = fx.Int32(_bidx(base_ptr))
    if const_expr(c_off):
        lds_c = lds_c + fx.Int32(c_off)
    if const_expr(c_rows is not None):
        row_rank, row_idx, row_ok, _extra = rows_out
        # a store whose tile is wider than its column bound loses every row when the row bound's
        # low 16 bits are zero (65536 routes), so the bound sits a row or two past the last one
        row_bound = c_num_rows + (1 if (c_num_rows + 1) & 0xFFFF else 2)
        # per destination rank, the lane's row index, or past the rows (dropped) when it goes elsewhere
        rank_rows = [
            (row_ok & (row_rank == fx.Int32(r))).select(row_idx, fx.Int32(row_bound))
            for r in range_constexpr(len(c_rank_bases))
        ]
        rank_bit = row_ok.select(
            ArithValue(std_arith.ShLIOp(_raw(fx.Int32(1)), _raw(row_rank)).result), zero32
        )
        rank_bases = [base + fx.Int64(blk_n) * fx.Int64(EB) for base in c_rank_bases]
        # TDM stores ignore the LDS pad fields: each store reads whole padded rows and drops the
        # pad columns as out of bounds, past the tile's last column
        c_cols = (n_ext < fx.Int32(tile_n)).select(n_ext, fx.Int32(tile_n))
        for j in range_constexpr(tile_m // num_waves // _SCATTER_ROWS):
            lanes = [j * _SCATTER_ROWS + i for i in range(_SCATTER_ROWS)]
            ranks_hit = _readlane_i32(rank_bit, lanes[0])
            for group_lane in lanes[1:]:
                ranks_hit = ranks_hit | _readlane_i32(rank_bit, group_lane)
            lds_row = fx.Int32(_SCATTER_ROWS) * (wave + fx.Int32(j * num_waves))
            lds_addr = lds_c + lds_row * fx.Int32(c_lds_row)
            # a pool block holds its rows by source rank, so most groups go to one rank
            for r in range_constexpr(len(c_rank_bases)):
                if (ranks_hit & fx.Int32(1 << r)) != zero32:
                    dgroup1, dgroup2, dgroup3 = _scatter_dgroups(
                        [_readlane_i32(rank_rows[r], group_lane) for group_lane in lanes],
                        c_cols,
                        row_bound,
                        c_lds_row // EB,
                        ldc,
                    )
                    tdm_ops.tensor_store_gather(
                        _GatherDesc(_gather_dgroup0(lds_addr, rank_bases[r], 32), dgroup1, dgroup2, dgroup3),
                        cache_policy=c_cache_modifier,
                    )
        tdm_ops.tensor_wait(0)
    else:
        c_res = create_buffer_resource_from_addr(c_addr, num_records_bytes=m_ext * fx.Int32(ldc * EB))
        dropped = m_ext * fx.Int32(ldc // 2)  # first word past num_records: the store is dropped
        for i in range_constexpr(tile_m * chunks_per_row // block):
            chunk = tid + fx.Int32(i * block)
            row = chunk // fx.Int32(chunks_per_row)
            c16 = chunk % fx.Int32(chunks_per_row)
            v = lds_load_b128(lds_c, row * fx.Int32(c_lds_row) + c16 * fx.Int32(16))
            word = row * fx.Int32(ldc // 2) + c16 * fx.Int32(4)
            word = (c16 * fx.Int32(8) < n_ext).select(word, dropped)
            buffer_store(v, c_res, word, cache_modifier=c_cache_modifier)
    return rows_out


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
    b_nn: Constexpr[int],
    issue_first: Constexpr[int],
    lds_cosched: Constexpr[int],
    wmma_b2b: Constexpr[int],
):
    block = m_warp * n_warp * WAVE
    arena_bytes = tile_geometry(tile_m, tile_n, tile_k, bool(b_nn))[3]
    if arena_bytes > GFX1250_LDS_BYTES:
        raise ValueError(f"LDS arena {arena_bytes} B exceeds {GFX1250_LDS_BYTES} B")

    @flyc.kernel(name="kernel_mega_nt", known_block_size=[block, 1, 1])
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

        # Mega's tile order over the pool blocks that are live this call
        pid = fx.Int32(fx.block_idx.x)
        real_tiles = rocdl.readfirstlane(T.i32, _i32_table(arg_num_tile_blocks)[fx.Int32(0)])
        live = pid < real_tiles * fx.Int32(N_BLOCKS_N)
        tile_m_idx, tile_n_idx = pool_tile_order(pid, real_tiles, N_BLOCKS_N, group_m, NUM_XCD)
        block_m = rocdl.readfirstlane(T.i32, live.select(tile_m_idx, fx.Int32(0)))
        block_n = rocdl.readfirstlane(T.i32, live.select(tile_n_idx, fx.Int32(0)))
        expert = rocdl.readfirstlane(T.i32, _i32_table(arg_tile_to_expert)[block_m])

        blk_n = block_n * fx.Int32(tile_n)
        m_row = block_m * fx.Int32(tile_m)
        gA = fx.recast_iter(fx.Int8, arg_a)
        gB = fx.recast_iter(fx.Int8, arg_b)
        c_addr = fx.Int64(fx.ptrtoint(fx.recast_iter(fx.Int8, arg_c))) + (
            fx.Int64(m_row) * fx.Int64(LDC) + fx.Int64(blk_n)
        ) * fx.Int64(EB)
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
            b_nn=bool(b_nn),
            i32_n=i32_n,
            issue_first=bool(issue_first),
            lds_cosched=bool(lds_cosched),
            wmma_b2b=bool(wmma_b2b),
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
    tile_k: int = 0,
    m_warp: int = 4,
    n_warp: int = 2,
    group_m: int = 16,
    idx_bits: int = 0,
    b_nn: bool = False,
    issue_first: int = 1,
    lds_cosched: int = -1,
    wmma_b2b: int | None = None,
) -> torch.Tensor:
    """``out[p] = a[row_idx[p]] @ b[e(p)].T`` per pool row ``p`` (``a[p]`` when ``row_idx`` is None);
    ``b`` is ``[G, N, K]``, or ``[G, K, N]`` with ``b_nn`` (``out[p] = a[...] @ b[e(p)]``).

    ``out`` is the pool, ``[num_pool_rows, N]`` with ``num_pool_rows`` a multiple of ``tile_m``;
    pool block ``t`` belongs to expert ``tile_to_expert[t]``, and only the first
    ``num_tile_blocks[0]`` blocks are computed. ``row_idx`` entries < 0 mark padding rows.
    ``tile_k`` 0 is 128 when it divides K, else 64. ``issue_first``, ``lds_cosched`` (-1: wherever
    the tile and ``issue_first`` allow it) and ``wmma_b2b`` (None: with a co-scheduled dense A) are
    ``nt_gemm_tile``'s.
    """
    _require_gfx1250(a.device)
    G, N, K = b.shape
    if b_nn:
        K, N = N, K
        assert N >= tile_n, "an NN block backs its B read off into the row, so N >= tile_n"
    pool_rows = out.shape[0]
    tile_k = tile_k or (128 if K % 128 == 0 else 64)
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
        check_gather_rows(a.shape[0], K, tile_m, tile_k)
    else:
        row_idx = tile_to_expert  # unread
    lds_cosched = resolve_lds_cosched(lds_cosched, issue_first, tile_m, tile_n, m_warp, n_warp)
    if wmma_b2b is None:
        wmma_b2b = int(lds_cosched and not gather)
    check_wmma_b2b(wmma_b2b, gather)
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
        N if b_nn else K,
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
        int(b_nn),
        int(issue_first),
        int(lds_cosched),
        int(wmma_b2b),
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
def arena_bytes(tile_m: int = 256, tile_n: int = 256, tile_k: int = 128, b_nn: bool = False) -> int:
    return tile_geometry(tile_m, tile_n, tile_k, b_nn)[3]
