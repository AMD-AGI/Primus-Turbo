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

"""Weight gradients ``dw = a^T @ b`` split over K, as tiles a fused launch runs next to its own work.

K is the rows (``R``, long against the ``M x N`` output, which alone has too few tiles to fill the
device). It is cut into ``S`` slices, one fp32 partial each. Once all are out, every slice of an
output tile sums one row band of it over them, in slice order (deterministic). The slices wait for
each other, so a launch dispatches them co-resident, at most one tile per CU.

``a`` is one ``[R, M]`` operand, or a pair of ``[R, G, 128]`` views stacking two weight gradients that
share ``b``: ``dw[g] = [a0[:, g] | a1[:, g]]^T @ b`` (``dw`` ``[G, 256, N]``). Each 256-row output tile
is then one ``g`` and reads ``b`` once for both halves (e.g. the k and v projections of one head, from
the q / k / v gradients in place, no ``cat``).
"""

import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.deep_ep.common import _asm
from primus_turbo.flydsl.fused_collective_gemm.common import SPIN_TIMEOUT, SYS_NT, i64_base
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel import grouped_gemm_bf16_variable_k_tile
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    extract_base_index,
)
from primus_turbo.flydsl.utils.gemm_helper import BLOCK_K, make_row_band_resource
from primus_turbo.flydsl.utils.prims import (
    atomic_add,
    ceildiv,
    ld,
    memory_fence,
    read_clock,
    spin_timed_out,
    st,
)

_BM = 256
_BN = 256
WGRAD_LDS_CHUNK_STRIDE = 1152
_THREADS = 512
# 16 B partial loads in flight per thread in the split-K fixup (vectors x slices)
_FIXUP_LOADS = 32

_COUNTERS: dict = {}


def split_k_counters(device, n, name="wgrad"):
    """Arrival counters of the split-K fixup, zero between launches (the last arrival resets its own);
    one set per ``name``, for the fixups sharing a launch."""
    c = _COUNTERS.get((device, name))
    if c is None or c.numel() < n:
        c = torch.zeros(max(n, 1024), dtype=torch.int32, device=device)
        _COUNTERS[(device, name)] = c
    return c


class _StoreBandF32:
    """``StoreCBf16.store_band16`` for an fp32 ``C``: the slices' partials stay unrounded. System
    scope stores: the fixup reading them may run on another XCD."""

    def __init__(self, C, c_cols):
        self.c_base = extract_base_index(C)
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64

    def store_band16(
        self, c_frags, base_row, base_col, col_step, n_tiles_a, n_tiles_b, row_bound, mask_n=False
    ):
        assert not mask_n
        rsrc = make_row_band_resource(self.c_base, base_row, row_bound, self.c_cols, 4)
        row_bytes = self.c_cols * 4
        base_off = ((self.lane_id // 16) * 4) * row_bytes + (base_col + self.lane_id % 16) * 4
        for ti in range_constexpr(n_tiles_a):
            for r in range_constexpr(4):
                off = base_off + (ti * 16 + r) * row_bytes
                for q in range_constexpr(len(c_frags)):
                    for j in range_constexpr(n_tiles_b):
                        val = Vec(c_frags[q][ti * n_tiles_b + j])[r]
                        buffer_store(
                            val,
                            rsrc,
                            off + q * col_step * 4 + j * 64,
                            cache_modifier=SYS_NT,
                            offset_is_bytes=True,
                        )


@ASTRewriter.transform
def _split_k_fixup(PART, OUT, CNT, lds, s, tile, row0, col0, *, S, OUT_M, OUT_N, num_out):
    """Slice ``s`` of output tile ``tile`` (rows / columns from ``row0`` / ``col0``): wait until the
    ``S`` slices' fp32 partials ``PART`` [S, OUT_M, OUT_N] are all out, then sum row band ``s`` of the
    tile over them in slice order (deterministic) into the bf16 ``OUT`` [OUT_M, OUT_N]. ``CNT[tile]``
    counts the arrivals, ``CNT[num_out + tile]`` the departures; the last to leave resets both. The
    slices must be co-resident: the caller dispatches them last, at most one tile per CU."""
    thread_index = fx.thread_idx.x
    cnt_base = i64_base(CNT)
    # every wave's partial stores retired (system scope) before the count goes out; the waves are
    # aligned here only after a "staggered" tile
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        atomic_add(cnt_base, tile, fx.Int32(1), scope="agent")
        spin_start = read_clock()
        while ld(cnt_base, tile, scope="agent") < fx.Int32(S):
            fx.rocdl.s_sleep(fx.Int32(1))
            if spin_timed_out(spin_start, SPIN_TIMEOUT):
                fx.printf("[fused_collective_gemm wgrad] split-K wait timeout: tile={} slice={}\n", tile, s)
                _asm("s_trap 2")
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    # the loads below bypass the caches; this only keeps them below the wait
    memory_fence(order="acquire", scope="workgroup")
    part_res = create_buffer_resource(PART, num_records_bytes=S * OUT_M * OUT_N * 4)
    out_res = create_buffer_resource(OUT, num_records_bytes=OUT_M * OUT_N * 2)
    band = ceildiv(_BM, S)
    vpr = _BN // 4
    num_vecs = band * vpr
    batch = max(1, _FIXUP_LOADS // S)
    per_iter = _THREADS * batch
    band0 = s * fx.Int32(band)
    for it in range_constexpr(ceildiv(num_vecs, per_iter)):
        coords = []
        for j in range_constexpr(batch):
            if const_expr(it * per_iter + j * _THREADS < num_vecs):
                v = fx.Int32(it * per_iter + j * _THREADS) + thread_index
                r = band0 + v // fx.Int32(vpr)
                ok = (r < fx.Int32(_BM)) & (r < band0 + fx.Int32(band)) & (v < fx.Int32(num_vecs))
                coords.append(((row0 + r) * fx.Int32(OUT_N) + col0 + (v % fx.Int32(vpr)) * fx.Int32(4), ok))
        # the loads bypass the caches: other XCDs wrote the slices
        vals = [
            [
                buffer_load(
                    part_res,
                    wd + fx.Int32(k * OUT_M * OUT_N),
                    vec_width=4,
                    dtype=fx.T.i32,
                    mask=ok,
                    cache_modifier=SYS_NT,
                )
                for k in range_constexpr(S)
            ]
            for wd, ok in coords
        ]
        for parts, (wd, ok) in zip(vals, coords):
            acc = Vec(parts[0]).bitcast(fx.Float32)
            for k in range_constexpr(1, S):
                acc = acc + Vec(parts[k]).bitcast(fx.Float32)
            buffer_store(acc.to(fx.BFloat16).bitcast(fx.Int32), out_res, wd // fx.Int32(2), mask=ok)
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        if atomic_add(cnt_base, fx.Int32(num_out) + tile, fx.Int32(1), scope="agent") == fx.Int32(S - 1):
            st(cnt_base, tile, fx.Int32(0))
            st(cnt_base, fx.Int32(num_out) + tile, fx.Int32(0))


def wgrad_tiles(M, N, S):
    return S * (M // _BM) * (N // _BN)


def wgrad_splits(R, tiles, num_cus, S=None):
    """(slices, rows per slice) of ``tiles`` output tiles over ``R`` rows: by default about one tile
    per CU; every slice non-empty and BLOCK_K-aligned."""
    if S is None:
        S = max(1, min(num_cus // tiles, ceildiv(R, BLOCK_K)))
    k_split = ceildiv(ceildiv(R, S), BLOCK_K) * BLOCK_K
    return ceildiv(R, k_split), k_split


def wgrad_tail_splits(R, tiles, num_cus):
    """Split-K behind a fused launch's tail, where CUs free over about one tile's length:
    ``((S, k_split), (S2, k_split2, t2))``, the last ``t2`` output tiles on ``S2 = S + 1`` shorter
    slices for the CUs that free last. About 3/4 of the CUs take a slice, so the very last ones to free
    (the tail of the launch's own work) take none."""
    S, k_split = wgrad_splits(R, tiles, num_cus, max(1, num_cus * 3 // 4 // tiles))
    S2, k_split2 = wgrad_splits(R, tiles, num_cus, S + 1)
    t2 = tiles * 3 // 8 if S2 > S else 0
    if (tiles - t2) * S + t2 * S2 > num_cus:
        t2 = 0
    return (S, k_split), (S2, k_split2, t2)


def wgrad_lead_splits(R, tiles, num_cus, k_rows=None):
    """Split-K ahead of a fused launch's own work: slices of about ``k_rows`` rows; by default the
    fewest slices whose tiles cover half the CUs (a whole-K tile runs about as long as the launch when
    they cover fewer), at most one tile per CU."""
    if k_rows is None:
        S = max(1, min(ceildiv(num_cus // 2, tiles), num_cus // tiles, ceildiv(R, BLOCK_K)))
    else:
        S = max(1, ceildiv(R, k_rows))
    k_split = ceildiv(ceildiv(R, S), BLOCK_K) * BLOCK_K
    return ceildiv(R, k_split), k_split


def _row_span(x):
    """The row-major ``[R, row stride]`` operand a 3D view reads (the tile indexes it by strides)."""
    return x.as_strided((x.shape[0], x.stride(0)), (x.stride(0), 1))


def wgrad_operands(a, b, dw):
    """Host side of ``wgrad_tile`` for ``dw = a^T @ b``: ``(A, A1, B, dims)``, with ``dims`` the
    tile's ``R``, ``M``, ``N`` and ``pair`` (``()`` for a single ``a``)."""
    R, N = b.shape
    assert b.is_contiguous() and N % _BN == 0 and dw.is_contiguous(), (
        "b [R, N] and dw contiguous, N % 256 == 0"
    )
    if isinstance(a, torch.Tensor):
        M = a.shape[1]
        assert tuple(a.shape) == (R, M) and a.is_contiguous() and M % _BM == 0 and tuple(dw.shape) == (M, N)
        return a, a, b, dict(R=R, M=M, N=N, pair=())
    a0, a1 = a
    G = a0.shape[1]
    for x in a:
        assert x.dim() == 3 and tuple(x.shape) == (R, G, _BM // 2) and x.stride(2) == 1, (
            "a pair of [R, G, 128] views, unit stride along the last dim"
        )
    assert tuple(dw.shape) == (G, _BM, N)
    pair = (a0.stride(0), a1.stride(0), a0.stride(1), a1.stride(1))
    return _row_span(a0), _row_span(a1), b, dict(R=R, M=G * _BM, N=N, pair=pair)


def wgrad_tile(w, A, A1, B, PART, OUT, CNT, lds, *, R, M, N, S, k_split, pair=(), tile0=0, persistent=False):
    """Tile ``w`` (tile-major, from output tile ``tile0``) of ``OUT = A^T @ B`` ``[M, N]`` (bf16; ``B``
    ``[R, N]``), on ``S`` K slices of ``k_split`` rows: fp32 partials ``PART`` ``[>= S, M, N]``, then
    every slice of an output tile sums its row band of them (``CNT``: ``split_k_counters`` of ``2 *
    wgrad_tiles(M, N, 1)``); at ``S == 1`` straight into ``OUT``. ``A`` is ``[R, M]``, or with ``pair =
    (lo_stride, hi_stride, lo_gstride, hi_gstride)`` the two halves of output tile ``g`` read ``A`` /
    ``A1`` (rows ``lo_stride`` / ``hi_stride`` apart) from column ``g * lo_gstride`` / ``g *
    hi_gstride``. ``lds`` needs the ``WGRAD_LDS_CHUNK_STRIDE`` layout."""
    assert M % _BM == 0 and N % _BN == 0
    assert S == 1 or persistent == "staggered", "the fixup's barriers need the tile's waves realigned"
    n_tiles = N // _BN
    # tile-major: an output tile's slices dispatch together, so the fixup barely waits
    out_tile = w // fx.Int32(S)
    if tile0:
        out_tile = fx.Int32(tile0) + out_tile
    s = w % fx.Int32(S)
    tile_m = out_tile // fx.Int32(n_tiles)
    tile_n = out_tile % fx.Int32(n_tiles)
    m_start = s * fx.Int32(k_split)
    m_end = arith.select(m_start + fx.Int32(k_split) < fx.Int32(R), m_start + fx.Int32(k_split), fx.Int32(R))
    a_split = None
    if pair:
        lo_stride, hi_stride, lo_gstride, hi_gstride = pair
        a_split = (A1, lo_stride, hi_stride, tile_m * fx.Int32(lo_gstride), tile_m * fx.Int32(hi_gstride))
    grouped_gemm_bf16_variable_k_tile(
        A,
        B,
        OUT if S == 1 else PART,
        s,
        tile_m,
        tile_n,
        m_start,
        m_end,
        lds,
        fx.Int32(M),
        fx.Int32(N),
        G=S,
        OUT_M=M,
        OUT_N=N,
        BLOCK_M=_BM,
        BLOCK_N=_BN,
        persistent=persistent,
        lds_chunk_stride=WGRAD_LDS_CHUNK_STRIDE,
        a_split=a_split,
        store_c=None if S == 1 else _StoreBandF32(PART, N),
    )
    if S > 1:
        _split_k_fixup(
            PART,
            OUT,
            CNT,
            lds,
            s,
            out_tile,
            tile_m * fx.Int32(_BM),
            tile_n * fx.Int32(_BN),
            S=S,
            OUT_M=M,
            OUT_N=N,
            num_out=(M // _BM) * n_tiles,
        )
