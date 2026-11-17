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

"""Ulysses A2AGemm: an all-to-all from head-sharded to sequence-sharded rows fused into the GEMM
that consumes them (push mode), with an optional residual epilogue.

Every rank holds rows of the full sequence for its ``K_r`` columns (its heads), as up to three
tensors (``q``, ``k``, ``v`` grads ``[n * M, h_x, d]``) whose heads interleave in ``G`` groups of
``(nq, nk, nv) / G`` heads, the weight's row order; the output projection is ``G = 1``, ``nk = nv = 0``. Push work moves the rows of destination ``dst``'s sequence slice into ``dst``'s symmetric
recv bank at column block ``rank`` and raises ``dst``'s flag ``(m_tile, rank)``. The recv bank is
row-major ``[M, n * K_r]``: the GEMM's ``A``. A GEMM tile walks K one source segment at a time in
the order the segments land, waiting for each segment's flag before its first load of it; the order
is fixed per (m-tile, rank), so the accumulation is bitwise deterministic.

The first ``num_push`` blocks claim push units (m-major, so early m-tiles complete first) until none
are left, then compute the last tiles. A block only waits once every unit is claimed by a running
block, so the launch cannot deadlock whatever the grid size.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.deep_ep.common import ld_lds, st_lds
from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel import (
    grouped_gemm_bf16_variable_k_tile,
)
from primus_turbo.flydsl.mega.bf16.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_tile,
)
from primus_turbo.flydsl.ulysses.common import (
    BLOCK_THREADS,
    STATE_TILE,
    SYS_NT,
    VEC_BF16,
    VEC_I32,
    WARP,
    _all_reached,
    claim_unit,
    exit_and_bump_epoch,
    i64_base,
    opaque_base,
    opaque_tid,
    read_clock,
    spin_flag,
    spin_flags,
)
from primus_turbo.flydsl.ulysses.workspace import (
    FLAG_BYTES,
    FLAG_REGION_BYTES,
    FLAGS_PER_REGION,
)
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
    extract_base_index,
)
from primus_turbo.flydsl.utils.gemm_helper import (
    StoreCBf16,
    _pack_out_pair,
    emit_if_then,
    make_bf16_fp16_tile_tensor,
    make_row_band_resource,
    make_value_attrs,
)
from primus_turbo.flydsl.utils.prims import _i64, _readfirstlane_i32, ceildiv, ld, memory_fence, st

# 16 B vectors whose loads are issued before their stores
_BATCH = 8
# K chunk of the variable-K wgrad tile (CHUNK x BLOCK_K rows) and its LDS chunk stride
_WGRAD_CHUNK_ROWS = 256
_WGRAD_LDS_CHUNK_STRIDE = 1152
# m-tile flags a wgrad tile polls at its start
_LOOKAHEAD = 16


def _residual_store_c(residual):
    """A ``StoreCBf16`` writing ``C = acc + residual``. Defined in a function so the kernel cache key
    (which hashes the source of functions the launcher reaches) covers the class body."""

    class _StoreCBf16Residual(StoreCBf16):
        """``C = acc + R`` with the add in fp32 before the one bf16 rounding (an unfused ``+ residual``
        after a bf16 GEMM output rounds twice). Only the paired-column store of a full ``BLOCK_N = 256``
        tile is implemented: the GEMM asserts ``N % BLOCK_N == 0``."""

        def __init__(self, C, c_rows, c_cols, out_ty, cache_modifier=0):
            super().__init__(C, c_rows, c_cols, out_ty, cache_modifier=cache_modifier)
            self.r_base = extract_base_index(residual)

        def store_band_pair16(self, frag_even, frag_odd, base_row, base_col, n_tiles_b, mask_cols=True):
            rsrc = make_row_band_resource(self.c_base, base_row, self.c_rows, self.c_cols, 2)
            res_rsrc = make_row_band_resource(self.r_base, base_row, self.c_rows, self.c_cols, 2)
            n_tiles_a = len(frag_even) // n_tiles_b
            lane_col = self.lane_id % 16
            row_bytes = self.c_cols * 2
            col_ok = [
                (base_col + (j * 16 + lane_col) * 2 < self.c_cols) if mask_cols else None
                for j in range_constexpr(n_tiles_b)
            ]
            base_off = ((self.lane_id // 16) * 4) * row_bytes + (base_col + lane_col * 2) * 2
            for ti in range_constexpr(n_tiles_a):
                offs = [base_off + (ti * 16 + r) * row_bytes for r in range_constexpr(4)]
                res = [
                    [
                        Vec.from_elements(
                            [
                                buffer_load(
                                    res_rsrc, (off + j * 64) // 4, vec_width=1, dtype=fx.T.i32, mask=col_ok[j]
                                )
                            ],
                            fx.Int32,
                        ).bitcast(fx.BFloat16)
                        for j in range_constexpr(n_tiles_b)
                    ]
                    for off in offs
                ]
                for r in range_constexpr(4):
                    for j in range_constexpr(n_tiles_b):
                        t = ti * n_tiles_b + j
                        packed = _pack_out_pair(
                            Vec(frag_even[t])[r] + res[r][j][0].to(fx.Float32),
                            Vec(frag_odd[t])[r] + res[r][j][1].to(fx.Float32),
                            self.out_ty,
                        )
                        buffer_store(
                            Vec.from_elements([packed], fx.Int32).bitcast(self.out_ty),
                            rsrc,
                            offs[r] + j * 64,
                            mask=col_ok[j],
                            cache_modifier=self.cache_modifier,
                            offset_is_bytes=True,
                        )

        def _unsupported(self, *args, **kwargs):
            raise NotImplementedError("the residual epilogue only covers full 256-wide column tiles")

        store = store_band16 = store_trans16 = _unsupported

    return _StoreCBf16Residual


def _copy_rows(thread_index, num_vecs, vec_coords, src_res, dst_res, src_cm, dst_cm, batch=_BATCH):
    """Copy ``num_vecs`` 16 B vectors, ``batch`` loads in flight per thread; ``vec_coords(v)`` gives
    ``(src_word, dst_word, valid)``."""
    per_iter = BLOCK_THREADS * batch
    for it in range_constexpr(ceildiv(num_vecs, per_iter)):
        coords = [
            vec_coords(fx.Int32(it * per_iter + j * BLOCK_THREADS) + thread_index)
            for j in range_constexpr(batch)
            if it * per_iter + j * BLOCK_THREADS < num_vecs
        ]
        vals = [
            buffer_load(src_res, src, vec_width=VEC_I32, dtype=fx.T.i32, mask=valid, cache_modifier=src_cm)
            for src, _, valid in coords
        ]
        for val, (_, dst, valid) in zip(vals, coords):
            buffer_store(val, dst_res, dst, mask=valid, cache_modifier=dst_cm)


def _o_row(t, dst, M, seq_local, num_ranks, is_bshd):
    """Row of ``O`` (full sequence) holding token ``t`` of ``dst``'s sequence slice."""
    if is_bshd:
        bi = t // fx.Int32(seq_local)
        return t + bi * fx.Int32(seq_local * (num_ranks - 1)) + dst * fx.Int32(seq_local)
    return dst * fx.Int32(M) + t


@ASTRewriter.transform
def _push_unit(thread_index, m_sub, dst, dst_heap, bank_offset, o_res, epoch, **c):
    """Rows of sub-tile ``m_sub`` of ``dst``'s sequence slice into ``dst``'s recv bank, then ``dst``'s flag."""
    M, K, K_r, rank, num_ranks = c["M"], c["K"], c["K_r"], c["rank"], c["num_ranks"]
    rows = c["BM"] // c["push_split"]
    G, d = c["G"], c["head_dim"]
    recv_res = create_buffer_resource_from_addr(dst_heap + bank_offset, num_records_bytes=c["M_pad"] * K * 2)
    heads = (c["nq"], c["nk"], c["nv"])
    vecs_per_head = d // VEC_BF16
    group_vecs = sum(heads) // G * vecs_per_head
    for x in range_constexpr(3):
        h_x = heads[x]
        if const_expr(h_x > 0):
            vecs_per_row = h_x * vecs_per_head
            per_group = h_x // G
            head_off = sum(heads[:x]) // G

            def coords(v, vecs_per_row=vecs_per_row, per_group=per_group, head_off=head_off):
                r = v // fx.Int32(vecs_per_row)
                col = v % fx.Int32(vecs_per_row)
                head = col // fx.Int32(vecs_per_head)
                dst_col = (
                    (head // fx.Int32(per_group)) * fx.Int32(group_vecs)
                    + (head % fx.Int32(per_group) + fx.Int32(head_off)) * fx.Int32(vecs_per_head)
                    + col % fx.Int32(vecs_per_head)
                )
                t = m_sub * fx.Int32(rows) + r
                o_row = _o_row(t, dst, M, c["seq_local"], num_ranks, c["is_bshd"])
                src_word = (o_row * fx.Int32(vecs_per_row) + col) * fx.Int32(VEC_I32)
                dst_word = t * fx.Int32(K // 2) + (fx.Int32(rank * K_r // VEC_BF16) + dst_col) * fx.Int32(
                    VEC_I32
                )
                return src_word, dst_word, (r < fx.Int32(rows)) & (t < fx.Int32(M))

            _copy_rows(thread_index, rows * vecs_per_row, coords, o_res[x], recv_res, 0, SYS_NT)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        flag_base = dst_heap + fx.Int64(c["flag_region"] * FLAG_REGION_BYTES)
        # no release (an L2 writeback per unit): the bank stores are system scope (SYS_NT), and the
        # wait above retired them before the flag goes out
        st(flag_base, m_sub * fx.Int32(num_ranks) + fx.Int32(rank), epoch, scope="sys", order="relaxed")


@ASTRewriter.transform
def _push_loop(thread_index, state_base, claim_slot, peers_base, bank_offset, o_res, epoch, trace_base, **c):
    """Claim ``(m_sub, dst = rank + i)`` units (``m_sub``: a ``BM / push_split``-row sub-tile) until
    none are left and push each one. Rank ``r`` walks the m-tiles from ``r * stagger`` on, so the
    sources of one m-tile land at ``n`` different times (see ``_arrival_k_seg``)."""
    num_ranks, S = c["num_ranks"], c["push_split"]
    m_tiles = ceildiv(c["M"], c["BM"])
    num_units = num_ranks * m_tiles * S
    unit = claim_unit(thread_index, state_base, claim_slot)
    while unit < fx.Int32(num_units):
        pos = unit // fx.Int32(num_ranks)
        m_tile = (pos // fx.Int32(S) + fx.Int32(c["rank"] * c["stagger"])) % fx.Int32(m_tiles)
        m_sub = m_tile * fx.Int32(S) + pos % fx.Int32(S)
        dst = (fx.Int32(c["rank"]) + unit % fx.Int32(num_ranks)) % fx.Int32(num_ranks)
        dst_heap = ld(peers_base, dst, dtype=fx.T.i64)
        if const_expr(trace_base is not None):
            if thread_index == fx.Int32(0):
                st(trace_base, fx.Int32(c["trace_units_at"]) + unit * fx.Int32(2), read_clock())
        _push_unit(thread_index, m_sub, dst, dst_heap, bank_offset, o_res, epoch, **c)
        if const_expr(trace_base is not None):
            if thread_index == fx.Int32(0):
                st(trace_base, fx.Int32(c["trace_units_at"]) + unit * fx.Int32(2) + 1, read_clock())
        unit = claim_unit(thread_index, state_base, claim_slot)


def _arrival_k_seg(flag_base, tile_m, epoch, trace_base=None, trace_idx=None, landed=False, **c):
    """The K walk of an m-tile: one segment per source, in the order the sources' rows land.

    Source ``s`` pushes m-tile ``i`` at position ``(i - s * D) mod m_tiles`` of its walk, so the
    earliest source of ``i`` is ``first = min(i // D, n - 1)`` and the rest follow ``first - 1, ...``
    cyclically. In step (``D = 0``) every source pushes ``i`` to itself first, then to ``s + 1, ...``,
    so this rank's own rows land first and source ``rank - j`` lands ``j``-th. The order depends on
    ``i`` and the rank only: the accumulation stays bitwise deterministic. Each wave waits for a
    segment's flags itself before its first load of it; ``landed`` (every flag already checked) walks
    the same order without the waits."""
    num_ranks, D, S, rank = c["num_ranks"], c["stagger"], c["push_split"], c["rank"]

    class _ArrivalKSeg:
        seg_len = c["K_r"] // 64

        def __init__(self):
            if D > 0:
                lead = tile_m // fx.Int32(D)
                self.first = arith.select(lead < fx.Int32(num_ranks - 1), lead, fx.Int32(num_ranks - 1))
            else:
                self.first = fx.Int32(rank)

        def src(self, seg):
            return (self.first + fx.Int32(num_ranks) - seg) % fx.Int32(num_ranks)

        def chunk_base(self, seg):
            return self.src(seg) * fx.Int32(self.seg_len)

        def wait(self, seg):
            if const_expr(landed):
                return
            src = self.src(seg)
            slots = [(tile_m * fx.Int32(S) + fx.Int32(sub)) * fx.Int32(num_ranks) + src for sub in range(S)]
            spin_flags(flag_base, slots, epoch, rank, src, "a2a_gemm")
            # recv rows are rewritten by peers every 2 epochs: the tile's first wait drops what this CU /
            # XCD still caches of them. No block reads a row before its flag, so later segments only
            # need their loads ordered after the flag.
            memory_fence(order="acquire", scope="sys" if seg == 0 else "wavefront")
            if const_expr(trace_base is not None and seg == num_ranks - 1):
                emit_if_then(fx.thread_idx.x == fx.Int32(0), lambda: st(trace_base, trace_idx, read_clock()))

    return _ArrivalKSeg()


def _wgrad_k_hook(flag_base, src, epoch, horizon_slot, m0, **c):
    """Waits of a weight-gradient tile part, whose ``A`` columns all come from source ``src`` and
    rows from the m-tiles ``[m0, m0 + m_tiles / wgrad_split)``: before the loads of its K chunk
    ``k`` (``_WGRAD_CHUNK_ROWS`` rows, two 64-row steps prefetched past it), the m-tiles those loads
    reach; only ``src``'s flags. Returns ``(hook, k_hook_from)``.

    A wait drains the wave's prefetches and costs a system-scope round trip, and a hook in the chunk
    loop slows it even when it does not wait (a wgrad tile ran 2.4x as long). So the tile first polls
    the flags of its first ``_LOOKAHEAD`` m-tiles at once: the chunks they cover run a loop without
    the hook (all of them once the exchange is done). Past them each wave keeps a horizon in LDS
    (``horizon_slot``), every m-tile below which has landed: the prologue polls the next
    ``_LOOKAHEAD`` m-tiles without blocking and moves it past them when all have landed, and chunks
    below the horizon skip the wait. A wait itself does not poll ahead: with the rows still in
    flight that only costs (Qwen3-30B bwd QKV 8K +3 %)."""
    num_ranks, S, rank, BM = c["num_ranks"], c["push_split"], c["rank"], c["BM"]
    # m-tiles of the part, counted from m0
    m_tiles = ceildiv(c["M"], BM) // c["wgrad_split"]
    per_chunk = _WGRAD_CHUNK_ROWS // BM
    # last row a chunk loads: (k + 1) * CHUNK_ROWS + 2 * 64 - 1
    lead = (_WGRAD_CHUNK_ROWS + 2 * 64 - 1) // BM
    look = max(1, _LOOKAHEAD // S)
    wave_slot = fx.Int32(1) + fx.thread_idx.x // fx.Int32(WARP)

    def clip(t):
        return arith.select(t < fx.Int32(m_tiles), t, fx.Int32(m_tiles - 1))

    def slots(tiles):
        return [
            ((t + m0) * fx.Int32(S) + fx.Int32(sub)) * fx.Int32(num_ranks) + src
            for t in tiles
            for sub in range(S)
        ]

    def wait(tiles):
        spin_flags(flag_base, slots(tiles), epoch, rank, src, "a2a_gemm wgrad")

    def advance(nxt):
        """m-tiles below ``nxt`` have landed: the horizon is ``nxt``, or ``nxt + look`` if those have too."""
        ahead = _all_reached(flag_base, slots([clip(nxt + fx.Int32(i)) for i in range(look)]), epoch)
        st_lds(horizon_slot, wave_slot, arith.select(ahead, nxt + fx.Int32(look), nxt))

    # m-tiles [0, landed) have landed; ordered before the loads by the prologue's acquire
    first = min(look, m_tiles)
    first_landed = _all_reached(flag_base, slots([fx.Int32(t) for t in range(first)]), epoch)
    landed = arith.select(first_landed, fx.Int32(first), fx.Int32(0))
    # first chunk whose loads reach m-tile ``landed``: hi(k) = k * per_chunk + lead >= landed
    k_hook_from = arith.select(
        landed >= fx.Int32(m_tiles),
        fx.Int32(2**30),
        arith.select(
            landed > fx.Int32(lead),
            (landed - fx.Int32(lead - per_chunk + 1)) // fx.Int32(per_chunk),
            fx.Int32(0),
        ),
    )

    def hook(k):
        if const_expr(isinstance(k, int)):
            # prologue: the first two 64-row steps; see _arrival_k_seg.wait for the sys scope
            wait([fx.Int32(0)])
            advance(arith.select(landed > fx.Int32(1), landed, fx.Int32(1)))
            memory_fence(order="acquire", scope="sys")
        else:
            hi = k * fx.Int32(per_chunk) + fx.Int32(lead)

            def wait_chunk():
                wait([clip(hi - fx.Int32(i)) for i in range(per_chunk)])
                st_lds(horizon_slot, wave_slot, hi + fx.Int32(1))
                memory_fence(order="acquire", scope="wavefront")

            horizon = _readfirstlane_i32(ld_lds(horizon_slot, wave_slot))
            emit_if_then(clip(hi) >= horizon, wait_chunk)

    return hook, k_hook_from


def _claimed_tile(q, **c):
    """Tile of claim ``q``: m-tiles in the order their last source lands (equal ``i mod D`` together)."""
    num_ranks, D, n_tiles = c["num_ranks"], c["stagger"], c["n_tiles"]
    q_m = q // fx.Int32(n_tiles)
    tile_m = q_m
    if D > 0:
        tile_m = arith.select(
            q_m < fx.Int32(num_ranks * D),
            (q_m % fx.Int32(num_ranks)) * fx.Int32(D) + q_m // fx.Int32(num_ranks),
            q_m,
        )
    return tile_m, q % fx.Int32(n_tiles)


@ASTRewriter.transform
def _all_landed(thread_index, flag_base, claim_slot, tile_m, epoch, **c):
    """1 when every source's rows of ``tile_m`` have landed (all its sub-tiles), else 0; one poll by
    thread 0, broadcast so the whole block takes the same tile body."""
    num_ranks, S = c["num_ranks"], c["push_split"]
    if thread_index == fx.Int32(0):
        slots = [tile_m * fx.Int32(S * num_ranks) + fx.Int32(i) for i in range(S * num_ranks)]
        ok = _all_reached(flag_base, slots, epoch)
        st_lds(claim_slot, 0, arith.select(ok, fx.Int32(1), fx.Int32(0)))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    landed = _readfirstlane_i32(ld_lds(claim_slot, 0))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    return landed


@ASTRewriter.transform
def _wait_all(thread_index, flag_base, tile_m, epoch, **c):
    """Every source's rows of ``tile_m`` have landed (all its sub-tiles)."""
    num_ranks, S = c["num_ranks"], c["push_split"]
    if thread_index == fx.Int32(0):
        slots = [tile_m * fx.Int32(S * num_ranks) + fx.Int32(i) for i in range(S * num_ranks)]
        spin_flags(flag_base, slots, epoch, c["rank"], -1, "a2a_gemm")
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    # recv rows are rewritten by peers every 2 epochs: drop what this CU / XCD still caches of them
    memory_fence(order="acquire", scope="sys")


def _part_flag(w, part, **c):
    """This rank's flag of part ``part`` of wgrad tile ``w`` being done, past the push flags."""
    return fx.Int32(c["wflag_at"]) + part * fx.Int32(c["num_wtiles"]) + w


@ASTRewriter.transform
def _wait_flag(thread_index, flag_base, slot, epoch, **c):
    """Block-wide wait for one of this rank's own flags (agent scope: no peer wrote what it guards)."""
    if thread_index == fx.Int32(0):
        spin_flag(flag_base, slot, epoch, c["rank"], -1, "a2a_gemm wgrad part")
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    memory_fence(order="acquire", scope="agent")


@ASTRewriter.transform
def _raise_flag(thread_index, flag_base, slot, epoch):
    """This block's stores so far (system scope) are retired, then ``slot`` = ``epoch``."""
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        st(flag_base, slot, epoch, scope="sys", order="relaxed")


@ASTRewriter.transform
def _wgrad_tile(
    thread_index, w, part, flag_base, recv_addr, XB, DW, WACC, lds, horizon_slot, epoch, trace_base, **c
):
    """``DW[tile] = A[:, tile rows]^T @ XB[:, tile cols]`` (the variable-K wgrad tile) over part
    ``part`` of the ``wgrad_split`` row ranges of M, in row order, each chunk after the m-tiles it reads
    have landed (``_wgrad_k_hook``). A part starts from the fp32 accumulators the previous part of the
    tile left in ``WACC`` and leaves its own there, so the result has the bits of one pass over M."""
    M, K, N, BMw, BN, S_w = c["M"], c["K"], c["N"], c["BMw"], c["BN"], c["wgrad_split"]
    wn_tiles = N // BN
    tile_k, tile_n = w // fx.Int32(wn_tiles), w % fx.Int32(wn_tiles)
    # trace slot of the part
    tw = part * fx.Int32(c["num_wtiles"]) + w
    if const_expr(trace_base is not None):
        if thread_index == fx.Int32(0):
            st(trace_base, fx.Int32(c["trace_wgrad_at"]) + tw * fx.Int32(2), read_clock())
    src = tile_k * fx.Int32(BMw) // fx.Int32(c["K_r"])
    seg_rows = M // S_w
    k_hook, k_hook_from = _wgrad_k_hook(
        flag_base, src, epoch, horizon_slot, part * fx.Int32(seg_rows // c["BM"]), **c
    )
    acc_in, acc_out = None, None
    if const_expr(S_w > 1):
        tile_bytes = BMw * BN * 4
        # a lane's accumulator vectors, each 16 B apart across the block
        acc_res = create_buffer_resource_from_addr(
            i64_base(WACC) + _i64(w) * fx.Int64(tile_bytes), num_records_bytes=tile_bytes
        )

        # one lane offset, the vector's in soffset: per-vector lane offsets get hoisted, spilled and
        # reloaded before each access, which serialises the 32 of them (a part ran ~25 us longer)
        lane_f32 = thread_index * fx.Int32(4)

        def save(i, v):
            buffer_store(v, acc_res, lane_f32, cache_modifier=SYS_NT, soffset_bytes=i * BLOCK_THREADS * 16)

        def load(i):
            return buffer_load(
                acc_res, lane_f32, vec_width=4, dtype=fx.T.f32, soffset_bytes=i * BLOCK_THREADS * 16
            )

        acc_in = (part > fx.Int32(0), load)
        acc_out = (part < fx.Int32(S_w - 1), part == fx.Int32(S_w - 1), save)
        if part > fx.Int32(0):
            _wait_flag(thread_index, flag_base, _part_flag(w, part - fx.Int32(1), **c), epoch, **c)
    grouped_gemm_bf16_variable_k_tile(
        None,
        XB,
        DW,
        fx.Int32(0),
        tile_k,
        tile_n,
        part * fx.Int32(seg_rows),
        part * fx.Int32(seg_rows) + fx.Int32(seg_rows),
        lds,
        fx.Int32(K),
        fx.Int32(N),
        G=1,
        OUT_M=K,
        OUT_N=N,
        BLOCK_M=BMw,
        BLOCK_N=BN,
        persistent="staggered",
        lds_chunk_stride=_WGRAD_LDS_CHUNK_STRIDE,
        a_base=recv_addr,
        k_hook=k_hook,
        k_hook_from=k_hook_from,
        acc_in=acc_in,
        acc_out=acc_out,
    )
    if const_expr(S_w > 1):
        if part < fx.Int32(S_w - 1):
            _raise_flag(thread_index, flag_base, _part_flag(w, part, **c), epoch)
    if const_expr(trace_base is not None):
        fx.rocdl.s_waitcnt(0)
        if thread_index == fx.Int32(0):
            st(trace_base, fx.Int32(c["trace_wgrad_at"]) + tw * fx.Int32(2) + 1, read_clock())


@ASTRewriter.transform
def _tile_loop(
    thread_index,
    state_base,
    claim_slot,
    flag_base,
    recv_addr,
    W,
    Y,
    RES,
    OG,
    XB,
    DW,
    WACC,
    lds,
    epoch,
    trace_base,
    **c,
):
    """Claim tiles until none are left: the ``num_wtiles`` weight-gradient tiles sit at claim
    positions ``[wgrad_at, wgrad_at + num_wtiles)``, the GEMM tiles around them. Split over
    ``wgrad_split`` row ranges, each range's GEMM tiles come first, then that part of every wgrad tile."""
    num_tiles, num_wtiles, w_at, S_w = c["num_tiles"], c["num_wtiles"], c["wgrad_at"], c["wgrad_split"]
    seg_tiles = num_tiles // S_w
    q = claim_unit(thread_index, state_base, claim_slot, STATE_TILE)
    while q < fx.Int32(num_tiles + S_w * num_wtiles):
        if const_expr(num_wtiles > 0 and S_w > 1):
            part = q // fx.Int32(seg_tiles + num_wtiles)
            r = q - part * fx.Int32(seg_tiles + num_wtiles)
            if r >= fx.Int32(seg_tiles):
                _wgrad_tile(
                    thread_index,
                    r - fx.Int32(seg_tiles),
                    part,
                    flag_base,
                    recv_addr,
                    XB,
                    DW,
                    WACC,
                    lds,
                    claim_slot,
                    epoch,
                    trace_base,
                    **c,
                )
            else:
                _gemm_tile(
                    thread_index,
                    part * fx.Int32(seg_tiles) + r,
                    flag_base,
                    claim_slot,
                    recv_addr,
                    W,
                    Y,
                    RES,
                    OG,
                    lds,
                    epoch,
                    trace_base,
                    **c,
                )
        elif const_expr(num_wtiles > 0):
            w = q - fx.Int32(w_at)
            if (w >= fx.Int32(0)) & (w < fx.Int32(num_wtiles)):
                _wgrad_tile(
                    thread_index,
                    w,
                    fx.Int32(0),
                    flag_base,
                    recv_addr,
                    XB,
                    DW,
                    WACC,
                    lds,
                    claim_slot,
                    epoch,
                    trace_base,
                    **c,
                )
            else:
                qd = arith.select(q < fx.Int32(w_at), q, q - fx.Int32(num_wtiles))
                _gemm_tile(
                    thread_index,
                    qd,
                    flag_base,
                    claim_slot,
                    recv_addr,
                    W,
                    Y,
                    RES,
                    OG,
                    lds,
                    epoch,
                    trace_base,
                    **c,
                )
        else:
            _gemm_tile(
                thread_index, q, flag_base, claim_slot, recv_addr, W, Y, RES, OG, lds, epoch, trace_base, **c
            )
        q = claim_unit(thread_index, state_base, claim_slot, STATE_TILE)


@ASTRewriter.transform
def _gemm_tile(thread_index, q, flag_base, claim_slot, recv_addr, W, Y, RES, OG, lds, epoch, trace_base, **c):
    """GEMM tile of claim ``q``. With whole K chunks per source it walks K by arrival
    (``_arrival_k_seg``); otherwise it waits for the whole m-tile, copies it out, then runs.

    The arrival walk first polls all of the m-tile's flags: once its rows have landed (most tiles
    when the GEMM outweighs the exchange) it runs the same K order without the per-segment waits.
    A wait inside the unrolled K loop costs even when the flag is set: it drains the prefetches and
    spills accumulators around the poll (DeepSeek-V3 o_proj n=4 128K: ~18 % of the kernel)."""
    M, M_pad, K, N, BM, BN = c["M"], c["M_pad"], c["K"], c["N"], c["BM"], c["BN"]
    n_tiles, progressive = c["n_tiles"], c["progressive"]
    gemm_tile = functools.partial(gemm_bf16_tile, c["layout"], pair_n=True)
    epilogue = {}
    if const_expr(c["has_residual"]):
        epilogue["store_c_cls"] = _residual_store_c(RES)
    og_vecs_per_row = K // VEC_BF16
    og_chunk = ceildiv(og_vecs_per_row, n_tiles)

    def copy_o(tile_m, tile_n):
        recv_res = create_buffer_resource_from_addr(recv_addr, num_records_bytes=M_pad * K * 2)
        og_res = create_buffer_resource(OG, num_records_bytes=M * K * 2)

        def og_coords(v):
            r = v // fx.Int32(og_chunk)
            t = tile_m * fx.Int32(BM) + r
            col = tile_n * fx.Int32(og_chunk) + v % fx.Int32(og_chunk)
            word = t * fx.Int32(K // 2) + col * fx.Int32(VEC_I32)
            valid = (r < fx.Int32(BM)) & (t < fx.Int32(M)) & (col < fx.Int32(og_vecs_per_row))
            return word, word, valid

        _copy_rows(thread_index, BM * og_chunk, og_coords, recv_res, og_res, SYS_NT, 0)

    tile_m, tile_n = _claimed_tile(q, **c)
    tile_index = tile_m * fx.Int32(n_tiles) + tile_n
    if const_expr(trace_base is not None):
        if thread_index == fx.Int32(0):
            st(trace_base, tile_index * fx.Int32(3), read_clock())

    # 128-row nt tiles: keep the lane offsets and bases out of LICM (see opaque_tid), which otherwise
    # spills them around the claim loop (GLM-4.7 o_proj: 182 scratch ops -> 0). The accumulators of a
    # 256-row tile leave no room to re-materialise them: there it spills more (DeepSeek-V3: 247 -> 445)
    opaque = c["layout"] == "nt" and BM == 128
    base = opaque_base if opaque else i64_base

    def run(k_seg):
        if const_expr(opaque):
            epilogue["tid"] = opaque_tid(thread_index)
        gemm_tile(
            make_bf16_fp16_tile_tensor(recv_addr, fx.Int64(0), M_pad * K),
            make_bf16_fp16_tile_tensor(base(W), fx.Int64(0), K * N),
            make_bf16_fp16_tile_tensor(base(Y), fx.Int64(0), M * N),
            fx.Int32(M),
            fx.Int32(N),
            lds,
            tile_m,
            tile_n,
            K=K,
            BLOCK_M=BM,
            BLOCK_N=BN,
            out_fp16=False,
            nt_vmcnt=c["nt_vmcnt"],
            n_tail=N % BN,
            persistent="staggered",
            k_seg=k_seg,
            **epilogue,
        )

    if const_expr(progressive):
        landed = _all_landed(thread_index, flag_base, claim_slot, tile_m, epoch, **c)
        if landed != fx.Int32(0):
            # see _arrival_k_seg.wait for the sys scope
            memory_fence(order="acquire", scope="sys")
            if const_expr(trace_base is not None):
                if thread_index == fx.Int32(0):
                    st(trace_base, tile_index * fx.Int32(3) + 1, read_clock())
            run(_arrival_k_seg(flag_base, tile_m, epoch, landed=True, **c))
        else:
            run(_arrival_k_seg(flag_base, tile_m, epoch, trace_base, tile_index * fx.Int32(3) + 1, **c))
    else:
        _wait_all(thread_index, flag_base, tile_m, epoch, **c)
        if const_expr(trace_base is not None):
            if thread_index == fx.Int32(0):
                st(trace_base, tile_index * fx.Int32(3) + 1, read_clock())
        if const_expr(c["save_o"]):
            copy_o(tile_m, tile_n)
        run(None)
    if const_expr(progressive and c["save_o"]):
        # every wave has waited for all of the m-tile's sources inside the K walk
        copy_o(tile_m, tile_n)
    if const_expr(trace_base is not None):
        fx.rocdl.s_waitcnt(0)
        if thread_index == fx.Int32(0):
            st(trace_base, tile_index * fx.Int32(3) + 2, read_clock())


@functools.lru_cache(maxsize=64)
def _make_kernel(
    M,
    K,
    N,
    BM,
    BN,
    rank,
    num_ranks,
    seq_local,
    is_bshd,
    has_residual,
    save_o,
    num_push,
    push_split,
    flag_region,
    bank_bytes,
    nt_vmcnt,
    layout,
    head_dim,
    G,
    nq,
    nk,
    nv,
    num_cus,
    BMw=0,
    wgrad_at=0,
    wgrad_split=1,
    trace=False,
):
    assert K % num_ranks == 0 and N % VEC_BF16 == 0 and head_dim % VEC_BF16 == 0
    K_r = K // num_ranks
    assert K_r == (nq + nk + nv) * head_dim and nq % G == 0 and nk % G == 0 and nv % G == 0
    assert layout in ("nt", "nn") and not (has_residual and layout == "nn")
    m_tiles = ceildiv(M, BM)
    M_pad = m_tiles * BM
    n_tiles = ceildiv(N, BN)
    num_tiles = m_tiles * n_tiles
    assert BM % push_split == 0 and m_tiles * push_split * num_ranks <= FLAGS_PER_REGION, (
        "too many m-tiles for one flag region"
    )
    assert M_pad * K < 2**31 and M * N < 2**31 and K * N < 2**31, "i32 element offsets overflow"
    assert M_pad * K * 2 <= bank_bytes
    assert not has_residual or (N % BN == 0 and BN == 256), "the residual epilogue needs N % 256 == 0"
    num_wtiles = 0
    if BMw:
        # recv pad rows are never written: the M walk needs whole m-tiles, and a tile's A columns
        # must come from one source
        assert M == M_pad and K_r % BMw == 0 and N % BN == 0 and BN == 256 and _WGRAD_CHUNK_ROWS % BM == 0
        num_wtiles = K // BMw * (N // BN)
        assert 0 <= wgrad_at <= num_tiles
    grid = min(num_tiles + num_wtiles + num_push, num_cus)
    num_push = min(num_push, grid)
    # Staggered sources only pay off when every tile has a block from the start: a tile claimed late
    # would walk most of its K after the last arrival. Otherwise all sources push m-major in step.
    stagger = m_tiles // num_ranks if num_tiles + num_wtiles <= grid - num_push else 0
    # in step, one m-tile per pusher unit: sub-tiles only pace the staggered arrivals
    push_split = push_split if stagger else 1
    assert not stagger or K_r % 64 == 0, "a staggered K walk needs whole K chunks per source"
    progressive = K_r % 64 == 0
    # a part's rows are whole m-tiles and whole wgrad K chunks; its done flags follow the push flags
    wflag_at = m_tiles * push_split * num_ranks
    assert wgrad_split == 1 or (
        BMw
        and m_tiles % wgrad_split == 0
        and (M // wgrad_split) % _WGRAD_CHUNK_ROWS == 0
        and wflag_at + (wgrad_split - 1) * num_wtiles <= FLAGS_PER_REGION
    )
    # i32 slots: the block-wide broadcast of a claimed unit, then each wave's wgrad horizon
    SharedStorage = _make_shared_storage(
        max(BM, BMw),
        BN,
        chunk_stride=_WGRAD_LDS_CHUNK_STRIDE if BMw else 1024,
        num_lds_row_idx_entries=1 + BLOCK_THREADS // WARP,
    )
    consts = dict(
        M=M,
        M_pad=M_pad,
        K=K,
        K_r=K_r,
        BM=BM,
        push_split=push_split,
        rank=rank,
        num_ranks=num_ranks,
        seq_local=seq_local,
        is_bshd=is_bshd,
        flag_region=flag_region,
        G=G,
        head_dim=head_dim,
        nq=nq,
        nk=nk,
        nv=nv,
        trace_units_at=3 * num_tiles,
        trace_wgrad_at=3 * num_tiles + 2 * num_ranks * m_tiles * push_split,
        BMw=BMw,
        num_wtiles=num_wtiles,
        wgrad_at=wgrad_at,
        wgrad_split=wgrad_split,
        wflag_at=wflag_at,
        stagger=stagger,
        progressive=progressive,
        N=N,
        BN=BN,
        n_tiles=n_tiles,
        num_tiles=num_tiles,
        layout=layout,
        nt_vmcnt=nt_vmcnt,
        has_residual=has_residual,
        save_o=save_o,
    )

    @flyc.kernel(name="ulysses_a2a_gemm_kernel", known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        OQ: fx.Tensor,
        OK: fx.Tensor,
        OV: fx.Tensor,
        W: fx.Tensor,
        RES: fx.Tensor,
        Y: fx.Tensor,
        OG: fx.Tensor,
        XB: fx.Tensor,
        DW: fx.Tensor,
        WACC: fx.Tensor,
        PEER_PTRS: fx.Tensor,
        STATE: fx.Tensor,
        TRACE: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        claim_slot = fx.Int32(fx.ptrtoint(lds.lds_row_idx.ptr))
        state_base = i64_base(STATE)
        peers_base = i64_base(PEER_PTRS)
        epoch = ld(state_base, fx.Int32(0), dtype=fx.T.i64)
        bank_offset = fx.Int64(FLAG_BYTES) + (epoch & fx.Int64(1)) * fx.Int64(bank_bytes)
        my_base = ld(peers_base, fx.Int32(rank), dtype=fx.T.i64)
        # debug timeline (s_memrealtime): [start, ready, end] per GEMM tile, then [start, end] per push unit
        trace_base = i64_base(TRACE) if trace else None

        if block_index < fx.Int32(num_push):
            o_res = [
                create_buffer_resource(t, num_records_bytes=max(num_ranks * M * h * head_dim * 2, 4))
                for t, h in ((OQ, nq), (OK, nk), (OV, nv))
            ]
            _push_loop(
                thread_index,
                state_base,
                claim_slot,
                peers_base,
                bank_offset,
                o_res,
                epoch,
                trace_base,
                **consts,
            )
        flag_base = my_base + fx.Int64(flag_region * FLAG_REGION_BYTES)
        _tile_loop(
            thread_index,
            state_base,
            claim_slot,
            flag_base,
            my_base + bank_offset,
            W,
            Y,
            RES,
            OG,
            XB,
            DW,
            WACC,
            lds,
            epoch,
            trace_base,
            **consts,
        )
        exit_and_bump_epoch(thread_index, state_base, epoch, grid, order="relaxed")

    return kernel, grid


@flyc.jit
def _launch(
    OQ,
    OK,
    OV,
    W,
    RES,
    Y,
    OG,
    XB,
    DW,
    WACC,
    PEER_PTRS,
    STATE,
    TRACE,
    M: fx.Constexpr[int],
    K: fx.Constexpr[int],
    N: fx.Constexpr[int],
    BM: fx.Constexpr[int],
    BN: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    seq_local: fx.Constexpr[int],
    is_bshd: fx.Constexpr[bool],
    has_residual: fx.Constexpr[bool],
    save_o: fx.Constexpr[bool],
    num_push: fx.Constexpr[int],
    push_split: fx.Constexpr[int],
    flag_region: fx.Constexpr[int],
    bank_bytes: fx.Constexpr[int],
    nt_vmcnt: fx.Constexpr[int],
    layout: fx.Constexpr[str],
    head_dim: fx.Constexpr[int],
    G: fx.Constexpr[int],
    nq: fx.Constexpr[int],
    nk: fx.Constexpr[int],
    nv: fx.Constexpr[int],
    num_cus: fx.Constexpr[int],
    BMw: fx.Constexpr[int],
    wgrad_at: fx.Constexpr[int],
    wgrad_split: fx.Constexpr[int],
    trace: fx.Constexpr[bool],
    stream: fx.Stream,
):
    kernel, grid = _make_kernel(
        M,
        K,
        N,
        BM,
        BN,
        rank,
        num_ranks,
        seq_local,
        is_bshd,
        has_residual,
        save_o,
        num_push,
        push_split,
        flag_region,
        bank_bytes,
        nt_vmcnt,
        layout,
        head_dim,
        G,
        nq,
        nk,
        nv,
        num_cus,
        BMw,
        wgrad_at,
        wgrad_split,
        trace,
    )
    kernel(
        OQ,
        OK,
        OV,
        W,
        RES,
        Y,
        OG,
        XB,
        DW,
        WACC,
        PEER_PTRS,
        STATE,
        TRACE,
        value_attrs=make_value_attrs(2, 0, "512,512"),
    ).launch(grid=(grid, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)


def a2a_gemm_bank_bytes(M, K, BM=256):
    return ceildiv(M, BM) * BM * K * 2


def ulysses_a2a_gemm(
    o,
    w,
    residual,
    y,
    o_gathered,
    workspace,
    *,
    seq_local,
    qkv_format,
    flag_region,
    layout="nt",
    head_dim=None,
    groups=1,
    BM=None,
    BN=256,
    num_push=None,
    push_split=8,
    wgrad=None,
    BMw=256,
    wgrad_at=None,
    wgrad_split=None,
    trace=None,
):
    """``o`` this rank's heads over the full sequence (``qkv_format`` row order): one ``[n * M, K_r]``
    tensor, or ``(dq, dk, dv)`` ``[n * M, h_x, d]`` whose heads interleave in ``groups`` groups per
    rank (the weight's row order). ``y = A @ w^T`` (``layout="nt"``, ``w`` ``[N, K]``) or ``A @ w``
    (``"nn"``, ``w`` ``[K, N]``), ``+ residual`` (nt only), where ``A`` is the gathered ``[M, K]``,
    ``K = n * K_r``; ``A`` is also copied to ``o_gathered`` unless it is ``None``.

    ``wgrad=(x, dw)`` (``nn`` only) also computes ``dw = A^T @ x`` (``x`` ``[M, N]``, ``dw``
    ``[K, N]``) in the same kernel, from ``BMw x 256`` tiles placed at claim position ``wgrad_at``
    (default: by shape, see below), or with ``wgrad_split > 1`` in that many parts over consecutive
    row ranges of M, part ``j`` claimed after range ``j``'s GEMM tiles (``wgrad_at`` unused)."""
    num_ranks = workspace.num_ranks
    if isinstance(o, torch.Tensor):
        head_dim = head_dim or o.shape[-1]
        parts = (o.view(o.shape[0], -1, head_dim), None, None)
    else:
        parts = tuple(o)
        head_dim = parts[0].shape[-1]
    heads = tuple(0 if p is None else p.shape[1] for p in parts)
    rows = parts[0].shape[0]
    K, N = (w.shape[1], w.shape[0]) if layout == "nt" else w.shape
    M = rows // num_ranks
    assert K == num_ranks * sum(heads) * head_dim and tuple(y.shape) == (M, N)
    assert all(p is None or (p.is_contiguous() and p.shape[0] == rows) for p in parts)
    assert w.is_contiguous() and y.is_contiguous()
    assert residual is None or (residual.is_contiguous() and residual.shape == y.shape)
    assert o_gathered is None or (o_gathered.is_contiguous() and tuple(o_gathered.shape) == (M, K))
    oq, ok, ov = (parts[0] if p is None else p for p in parts)
    num_cus = torch.cuda.get_device_properties(y.device).multi_processor_count
    num_wtiles = 0
    if wgrad is not None:
        xb, dw = wgrad
        assert layout == "nn" and o_gathered is None
        assert (
            xb.is_contiguous()
            and tuple(xb.shape) == (M, N)
            and dw.is_contiguous()
            and tuple(dw.shape) == (K, N)
        )
        num_wtiles = K // BMw * (N // BN)
    # Weight-gradient tiles reduce over all of M, so taken first they hold their CUs at the pace the
    # rows land. That pays while the exchange dominates (Qwen3-30B: 25-45 % slower otherwise); once
    # the GEMM outweighs it (N * n >= 4096 * (n - 1), see gemm_bound below) the GEMM tiles go first
    # with 256-row tiles and the wgrad tiles run on landed rows (Llama3.1-8B bwd QKV 4K 592 -> 516 us,
    # MiniMax-M2.5 689 -> 612 us). Over two waves of wgrad tiles (GLM-4.7: 1120) they always go last
    # and get 32 pushers (GLM-4.7 16 pushers: +16 %).
    many_wtiles = num_wtiles > 2 * num_cus
    wgrad_last = wgrad is not None and (many_wtiles or N * num_ranks >= 4096 * (num_ranks - 1))
    # GEMM / exchange time grows with N * n / (n - 1): ~1.1 for Llama-70B (hidden 8192, n = 4), 0.55
    # for Llama3.1-8B, 0.3 for Qwen3-30B. Past ~1 the CUs, not the links, bound the launch: 128-row
    # tiles cost 35 % more GEMM time than 256-row ones and pushers beyond what keeps ahead of the K walk
    # take CUs from it (Llama-70B o_proj 4K: 844 -> 748 us, 8K: 1506 -> 1252 us). Not with in-kernel
    # wgrad: its tiles read the recv bank too, and fewer pushers starve them (Llama-70B bwd QKV 8K
    # +12 %). From 8192 rows, 8 pushers suffice for a single peer but fall behind three or more once
    # landed tiles skip the per-segment waits (DeepSeek-V3 32K n = 4: 2130 -> 2003 us with 16; n = 2:
    # 4801 -> 5030 us).
    gemm_bound = wgrad is None and N * num_ranks >= 8192 * (num_ranks - 1)
    if num_push is None:
        if gemm_bound:
            num_push = 8 if M >= 8192 and num_ranks == 2 else 16
        elif wgrad is None:
            # ~5 pushers per peer keep the links busy, more take CUs from the K walk (32K, n = 4:
            # GLM-4.7 1627 -> 1441 us with 16, Qwen3-235B 923 -> 806, MiniMax-M2.5 609 -> 587; n = 2
            # MiniMax-M2.5 1942 -> 1874). n = 8 and the exchange-dominated launches below half the
            # GEMM-bound ratio need 32 (GLM-4.7 n = 8: 760 us with 16, 732; Qwen3-30B: 388, 383)
            exchange_bound = 2 * N * num_ranks < 8192 * (num_ranks - 1)
            num_push = 32 if num_ranks > 4 or exchange_bound else 16
        else:
            # with few weight-gradient tiles the launch is compute-bound: 16 pushers still fill the links
            num_push = 32 if many_wtiles else 16
    if BM is None:
        # 256-row tiles while every tile has a CU (the staggered K walk) or the GEMM dominates;
        # otherwise 128 rows halve the tail of the m-tile that lands last
        fits = ceildiv(M, 256) * ceildiv(N, BN) + num_wtiles + num_push <= num_cus
        BM = 256 if fits or wgrad_last or gemm_bound else 128
    if wgrad_at is None:
        wgrad_at = ceildiv(M, BM) * ceildiv(N, BN) if wgrad_last else 0
    if wgrad is None:
        wgrad_split = 1
    elif wgrad_split is None:
        # Behind all GEMM tiles the wgrad tiles run after the last rows land; in two halves of M the
        # first runs on landed rows during the exchange (32K bwd QKV n = 4: MiniMax-M2.5 1106 -> 1018 us).
        # Not over two waves of wgrad tiles (GLM-4.7 n = 4 2413 -> 2470: the GEMM dominates and each
        # hand-over moves 256 KB per tile), nor with the wgrad tiles first (Qwen3-30B 529 -> 535);
        # 4 parts lose to 2 everywhere.
        flags = ceildiv(M, BM) * push_split * num_ranks + num_wtiles
        halves = wgrad_last and not many_wtiles and M % (2 * 256) == 0 and flags <= FLAGS_PER_REGION
        wgrad_split = 2 if halves else 1
    # fp32 accumulators a wgrad tile part hands to the next (one buffer: the parts run in order)
    wacc = torch.empty(K * N, dtype=torch.float32, device=y.device) if wgrad_split > 1 else y
    run_launcher(
        _launch,
        oq,
        ok,
        ov,
        w,
        y if residual is None else residual,
        y,
        y if o_gathered is None else o_gathered,
        y if wgrad is None else wgrad[0],
        y if wgrad is None else wgrad[1],
        wacc,
        workspace.peer_ptrs,
        workspace.state,
        workspace.state if trace is None else trace,
        M,
        K,
        N,
        BM,
        BN,
        workspace.rank,
        num_ranks,
        seq_local,
        qkv_format == "bshd",
        residual is not None,
        o_gathered is not None,
        num_push,
        push_split,
        flag_region,
        workspace.bank_bytes,
        3,
        layout,
        head_dim,
        groups,
        *heads,
        num_cus,
        BMw if wgrad is not None else 0,
        wgrad_at if wgrad is not None else 0,
        wgrad_split,
        trace is not None,
        torch.cuda.current_stream(),
    )
