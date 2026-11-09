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
row-major ``[M, n * K_r]``: the GEMM's ``A``. A GEMM tile waits for the flags of its m-tile from all
sources, then runs the plain K loop, so the accumulation order is fixed (bitwise deterministic).

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
    claim_unit,
    exit_and_bump_epoch,
    i64_base,
    read_clock,
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
from primus_turbo.flydsl.utils.prims import ceildiv, ld, memory_fence, st

# 16 B vectors whose loads are issued before their stores
_BATCH = 8
# K chunk of the variable-K wgrad tile (CHUNK x BLOCK_K rows) and its LDS chunk stride
_WGRAD_CHUNK_ROWS = 256
_WGRAD_LDS_CHUNK_STRIDE = 1152


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


def _copy_rows(thread_index, num_vecs, vec_coords, src_res, dst_res, src_cm, dst_cm):
    """Copy ``num_vecs`` 16 B vectors; ``vec_coords(v)`` gives ``(src_word, dst_word, valid)``."""
    per_iter = BLOCK_THREADS * _BATCH
    for it in range_constexpr(ceildiv(num_vecs, per_iter)):
        coords = [
            vec_coords(fx.Int32(it * per_iter + j * BLOCK_THREADS) + thread_index)
            for j in range_constexpr(_BATCH)
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
        st(flag_base, m_sub * fx.Int32(num_ranks) + fx.Int32(rank), epoch, scope="sys", order="release")


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


def _arrival_k_seg(flag_base, tile_m, epoch, trace_base=None, trace_idx=None, **c):
    """The K walk of an m-tile: one segment per source, in the order the sources' rows land.

    Source ``s`` pushes m-tile ``i`` at position ``(i - s * D) mod m_tiles`` of its walk, so the
    earliest source of ``i`` is ``first = min(i // D, n - 1)`` and the rest follow ``first - 1, ...``
    cyclically. The order depends on ``i`` only: the accumulation stays bitwise deterministic. Each
    wave waits for a segment's flags itself before its first load of it."""
    num_ranks, D, S, rank = c["num_ranks"], c["stagger"], c["push_split"], c["rank"]

    class _ArrivalKSeg:
        seg_len = c["K_r"] // 64

        def __init__(self):
            if D > 0:
                lead = tile_m // fx.Int32(D)
                self.first = arith.select(lead < fx.Int32(num_ranks - 1), lead, fx.Int32(num_ranks - 1))
            else:
                self.first = fx.Int32(0)

        def src(self, seg):
            return (self.first + fx.Int32(num_ranks) - seg) % fx.Int32(num_ranks)

        def chunk_base(self, seg):
            return self.src(seg) * fx.Int32(self.seg_len)

        def wait(self, seg):
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


def _wgrad_k_hook(flag_base, src, epoch, **c):
    """Waits of a weight-gradient tile, whose ``A`` columns all come from source ``src``: before the
    loads of K chunk ``k`` (``_WGRAD_CHUNK_ROWS`` rows, two 64-row steps prefetched past it), the
    m-tiles those loads reach; only ``src``'s flags."""
    num_ranks, S, rank, BM = c["num_ranks"], c["push_split"], c["rank"], c["BM"]
    m_tiles = ceildiv(c["M"], BM)
    per_chunk = _WGRAD_CHUNK_ROWS // BM
    # last row a chunk loads: (k + 1) * CHUNK_ROWS + 2 * 64 - 1
    lead = (_WGRAD_CHUNK_ROWS + 2 * 64 - 1) // BM

    def wait(tiles):
        slots = [
            (t * fx.Int32(S) + fx.Int32(sub)) * fx.Int32(num_ranks) + src for t in tiles for sub in range(S)
        ]
        spin_flags(flag_base, slots, epoch, rank, src, "a2a_gemm wgrad")

    def hook(k):
        if const_expr(isinstance(k, int)):
            # prologue: the first two 64-row steps; see _arrival_k_seg.wait for the sys scope
            wait([fx.Int32(0)])
            memory_fence(order="acquire", scope="sys")
        else:
            hi = k * fx.Int32(per_chunk) + fx.Int32(lead)
            tiles = [hi - fx.Int32(i) for i in range(per_chunk)]
            wait([arith.select(t < fx.Int32(m_tiles), t, fx.Int32(m_tiles - 1)) for t in tiles])
            memory_fence(order="acquire", scope="wavefront")

    return hook


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


@ASTRewriter.transform
def _wgrad_tile(thread_index, w, flag_base, recv_addr, XB, DW, lds, epoch, trace_base, **c):
    """``DW[tile] = A[:, tile rows]^T @ XB[:, tile cols]`` over all of M (the variable-K wgrad tile),
    in row order, each chunk after the m-tiles it reads have landed (``_wgrad_k_hook``)."""
    M, K, N, BMw, BN = c["M"], c["K"], c["N"], c["BMw"], c["BN"]
    wn_tiles = N // BN
    tile_k, tile_n = w // fx.Int32(wn_tiles), w % fx.Int32(wn_tiles)
    if const_expr(trace_base is not None):
        if thread_index == fx.Int32(0):
            st(trace_base, fx.Int32(c["trace_wgrad_at"]) + w * fx.Int32(2), read_clock())
    src = tile_k * fx.Int32(BMw) // fx.Int32(c["K_r"])
    grouped_gemm_bf16_variable_k_tile(
        None,
        XB,
        DW,
        fx.Int32(0),
        tile_k,
        tile_n,
        fx.Int32(0),
        fx.Int32(M),
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
        k_hook=_wgrad_k_hook(flag_base, src, epoch, **c),
    )
    if const_expr(trace_base is not None):
        fx.rocdl.s_waitcnt(0)
        if thread_index == fx.Int32(0):
            st(trace_base, fx.Int32(c["trace_wgrad_at"]) + w * fx.Int32(2) + 1, read_clock())


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
    lds,
    epoch,
    trace_base,
    **c,
):
    """Claim tiles until none are left: the ``num_wtiles`` weight-gradient tiles sit at claim
    positions ``[wgrad_at, wgrad_at + num_wtiles)``, the GEMM tiles around them."""
    num_tiles, num_wtiles, w_at = c["num_tiles"], c["num_wtiles"], c["wgrad_at"]
    q = claim_unit(thread_index, state_base, claim_slot, STATE_TILE)
    while q < fx.Int32(num_tiles + num_wtiles):
        if const_expr(num_wtiles > 0):
            w = q - fx.Int32(w_at)
            if (w >= fx.Int32(0)) & (w < fx.Int32(num_wtiles)):
                _wgrad_tile(thread_index, w, flag_base, recv_addr, XB, DW, lds, epoch, trace_base, **c)
            else:
                qd = arith.select(q < fx.Int32(w_at), q, q - fx.Int32(num_wtiles))
                _gemm_tile(thread_index, qd, flag_base, recv_addr, W, Y, RES, OG, lds, epoch, trace_base, **c)
        else:
            _gemm_tile(thread_index, q, flag_base, recv_addr, W, Y, RES, OG, lds, epoch, trace_base, **c)
        q = claim_unit(thread_index, state_base, claim_slot, STATE_TILE)


@ASTRewriter.transform
def _gemm_tile(thread_index, q, flag_base, recv_addr, W, Y, RES, OG, lds, epoch, trace_base, **c):
    """GEMM tile of claim ``q``. Staggered (``D > 0``), it walks K by arrival (``_arrival_k_seg``);
    in step, it waits for the whole m-tile, copies it out, then runs."""
    M, M_pad, K, N, BM, BN = c["M"], c["M_pad"], c["K"], c["N"], c["BM"], c["BN"]
    n_tiles, progressive = c["n_tiles"], c["stagger"] > 0
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
    k_seg = None
    if const_expr(progressive):
        k_seg = _arrival_k_seg(flag_base, tile_m, epoch, trace_base, tile_index * fx.Int32(3) + 1, **c)
    else:
        _wait_all(thread_index, flag_base, tile_m, epoch, **c)
        if const_expr(trace_base is not None):
            if thread_index == fx.Int32(0):
                st(trace_base, tile_index * fx.Int32(3) + 1, read_clock())
        if const_expr(c["save_o"]):
            copy_o(tile_m, tile_n)
    gemm_tile(
        make_bf16_fp16_tile_tensor(recv_addr, fx.Int64(0), M_pad * K),
        make_bf16_fp16_tile_tensor(i64_base(W), fx.Int64(0), K * N),
        make_bf16_fp16_tile_tensor(i64_base(Y), fx.Int64(0), M * N),
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
    # one extra i32: the block-wide broadcast of a claimed push unit
    SharedStorage = _make_shared_storage(
        max(BM, BMw), BN, chunk_stride=_WGRAD_LDS_CHUNK_STRIDE if BMw else 1024, num_lds_row_idx_entries=1
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
        stagger=stagger,
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
            lds,
            epoch,
            trace_base,
            **consts,
        )
        exit_and_bump_epoch(thread_index, state_base, epoch, grid)

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
    trace=None,
):
    """``o`` this rank's heads over the full sequence (``qkv_format`` row order): one ``[n * M, K_r]``
    tensor, or ``(dq, dk, dv)`` ``[n * M, h_x, d]`` whose heads interleave in ``groups`` groups per
    rank (the weight's row order). ``y = A @ w^T`` (``layout="nt"``, ``w`` ``[N, K]``) or ``A @ w``
    (``"nn"``, ``w`` ``[K, N]``), ``+ residual`` (nt only), where ``A`` is the gathered ``[M, K]``,
    ``K = n * K_r``; ``A`` is also copied to ``o_gathered`` unless it is ``None``.

    ``wgrad=(x, dw)`` (``nn`` only) also computes ``dw = A^T @ x`` (``x`` ``[M, N]``, ``dw``
    ``[K, N]``) in the same kernel, from ``BMw x 256`` tiles placed at claim position ``wgrad_at``
    (default: by shape, see below)."""
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
    # Over two waves of weight-gradient tiles (GLM-4.7: 1120): taken first they fill the CUs and the
    # GEMM tiles run after the exchange, so the GEMM goes first, keeps its 256-row tiles and gets 32
    # pushers. Not below that: Qwen3-30B (160) and MiniMax-M2.5 (384) are slower this way.
    many_wtiles = num_wtiles > 2 * num_cus
    if num_push is None:
        # with few weight-gradient tiles the launch is compute-bound: 16 pushers still fill the links
        num_push = 32 if wgrad is None or many_wtiles else 16
    if BM is None:
        # 256-row tiles while every tile has a CU (the staggered K walk); otherwise 128 rows halve the
        # tail of the m-tile that lands last
        fits = ceildiv(M, 256) * ceildiv(N, BN) + num_wtiles + num_push <= num_cus
        BM = 256 if fits or many_wtiles else 128
    if wgrad_at is None:
        wgrad_at = ceildiv(M, BM) * ceildiv(N, BN) if many_wtiles else 0
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
        trace is not None,
        torch.cuda.current_stream(),
    )
