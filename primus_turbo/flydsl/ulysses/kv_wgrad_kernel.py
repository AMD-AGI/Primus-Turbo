###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The up-projection weight gradients of the Ulysses MLA latent backward, split over K.

``d_w_kv[h] = [dk_nope[:, h] | dv[:, h]]^T @ c_kv`` for the rank's ``hl`` heads, with K the gathered
sequence (``n * T`` rows, long against the ``hl * (dn + dv)`` x ``Ckv`` output). A 256-row output
tile is one head: its A halves read ``dk_nope`` and ``dv`` of that head in place, so no ``cat``
packs them. K is cut into ``S`` slices, one fp32 partial each: the output alone has too few tiles
to fill the device. Once all are out, every slice of an output tile sums one row band of it over
them, in slice order. ``q_wgrad_tile`` is the same for ``d_w_q = gq^T @ c_q``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.deep_ep.common import _asm
from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.gemm.gemm_bf16_kernel import _make_shared_storage
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel import grouped_gemm_bf16_variable_k_tile
from primus_turbo.flydsl.ulysses.common import SPIN_TIMEOUT, SYS_NT, i64_base
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    extract_base_index,
)
from primus_turbo.flydsl.utils.gemm_helper import BLOCK_K, make_row_band_resource, make_value_attrs
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
KV_WGRAD_LDS_CHUNK_STRIDE = 1152
_THREADS = 512
# 16 B partial loads in flight per thread in the split-K fixup (vectors x slices)
_FIXUP_LOADS = 32

_COUNTERS: dict = {}


def split_k_counters(device, n, name="kv"):
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
                fx.printf("[ulysses wgrad] split-K wait timeout: tile={} slice={}\n", tile, s)
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


def kv_wgrad_splits(R, hl, Ckv, num_cus, S=None):
    """(slices, rows per slice): by default about one tile per CU; every slice non-empty and
    BLOCK_K-aligned."""
    if S is None:
        S = max(1, min(num_cus // kv_wgrad_tiles(hl, Ckv, 1), ceildiv(R, BLOCK_K)))
    k_split = ceildiv(ceildiv(R, S), BLOCK_K) * BLOCK_K
    return ceildiv(R, k_split), k_split


def kv_wgrad_tiles(hl, Ckv, S):
    return S * hl * (Ckv // _BN)


def kv_wgrad_tail_splits(R, hl, Ckv, num_cus):
    """Split-K of the kv weight gradient behind a fused launch's tail, where CUs free over about one
    tile's length: ``((S, k_split), (S2, k_split2, t2))``, the last ``t2`` output tiles on ``S2 = S + 1``
    shorter slices for the CUs that free last. About 3/4 of the CUs take a slice, so the very last
    ones to free (the tail reduce) take none."""
    tiles = kv_wgrad_tiles(hl, Ckv, 1)
    S, k_split = kv_wgrad_splits(R, hl, Ckv, num_cus, max(1, num_cus * 3 // 4 // tiles))
    S2, k_split2 = kv_wgrad_splits(R, hl, Ckv, num_cus, S + 1)
    t2 = tiles * 3 // 8 if S2 > S else 0
    if (tiles - t2) * S + t2 * S2 > num_cus:
        t2 = 0
    return (S, k_split), (S2, k_split2, t2)


def q_wgrad_splits(R, Nq, Cq, num_cus, k_rows=None):
    """(slices, rows per slice) of the q weight gradient: slices of about ``k_rows`` gathered rows;
    by default the fewest slices whose tiles cover half the CUs (a whole-K tile runs about as long
    as the launch when they cover fewer), at most one tile per CU."""
    if k_rows is None:
        base = q_wgrad_tiles(Nq, Cq, 1)
        S = max(1, min(ceildiv(num_cus // 2, base), num_cus // base, ceildiv(R, BLOCK_K)))
    else:
        S = max(1, ceildiv(R, k_rows))
    k_split = ceildiv(ceildiv(R, S), BLOCK_K) * BLOCK_K
    return ceildiv(R, k_split), k_split


def q_wgrad_tiles(Nq, Cq, S):
    return S * (Nq // _BM) * (Cq // _BN)


def q_wgrad_tile(w, GQ, LQ, PART, OUT, CNT, lds, *, R, Nq, Cq, S, k_split, persistent=False):
    """Tile ``w`` (tile-major) of the split-K q weight gradient ``OUT = gq^T @ lq`` [Nq, Cq] (bf16;
    ``gq`` [R, Nq], ``lq`` [R, Cq]): fp32 partials ``PART`` [S, Nq, Cq], summed as in
    ``kv_wgrad_tile`` (``CNT``: ``2 * q_wgrad_tiles(Nq, Cq, 1)`` counters); at ``S == 1`` straight
    into ``OUT``. ``lds`` needs the ``KV_WGRAD_LDS_CHUNK_STRIDE`` layout."""
    assert Nq % _BM == 0 and Cq % _BN == 0
    assert S == 1 or persistent == "staggered", "the fixup's barriers need the tile's waves realigned"
    n_tiles = Cq // _BN
    out_tile = w // fx.Int32(S)
    s = w % fx.Int32(S)
    tile_m = out_tile // fx.Int32(n_tiles)
    tile_n = out_tile % fx.Int32(n_tiles)
    m_start = s * fx.Int32(k_split)
    m_end = arith.select(m_start + fx.Int32(k_split) < fx.Int32(R), m_start + fx.Int32(k_split), fx.Int32(R))
    grouped_gemm_bf16_variable_k_tile(
        GQ,
        LQ,
        PART,
        s,
        tile_m,
        tile_n,
        m_start,
        m_end,
        lds,
        fx.Int32(Nq),
        fx.Int32(Cq),
        G=S,
        OUT_M=Nq,
        OUT_N=Cq,
        BLOCK_M=_BM,
        BLOCK_N=_BN,
        persistent=persistent,
        lds_chunk_stride=KV_WGRAD_LDS_CHUNK_STRIDE,
        store_c=None if S == 1 else _StoreBandF32(PART, Cq),
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
            OUT_M=Nq,
            OUT_N=Cq,
            num_out=(Nq // _BM) * n_tiles,
        )


def kv_wgrad_tile(
    w, GK, GV, LC, PART, OUT, CNT, lds, *, R, hl, dn, dr, dv, Ckv, S, k_split, tile0=0, persistent=False
):
    """Tile ``w`` (tile-major, from output tile ``tile0``) of the split-K kv weight gradient ``OUT``
    [hl, dn + dv, Ckv] (bf16): fp32 partials ``PART`` [>= S, hl, dn + dv, Ckv], then every slice of an
    output tile sums its row band of them (``CNT``: ``split_k_counters`` of ``2 * kv_wgrad_tiles(hl,
    Ckv, 1)``, ``_split_k_fixup``); at ``S == 1`` straight into ``OUT``. ``lds`` needs the
    ``KV_WGRAD_LDS_CHUNK_STRIDE`` layout."""
    assert dn == _BM // 2 and dv == _BM // 2, "a 256-row tile is one head's [dk_nope | dv]"
    assert Ckv % _BN == 0
    assert S == 1 or persistent == "staggered", "the fixup's barriers need the tile's waves realigned"
    OUT_M, OUT_N = hl * (dn + dv), Ckv
    n_tiles = Ckv // _BN
    per_split = hl * n_tiles
    # tile-major: an output tile's slices dispatch together, so the fixup barely waits
    out_tile = fx.Int32(tile0) + w // fx.Int32(S)
    s = w % fx.Int32(S)
    head = out_tile // fx.Int32(n_tiles)
    m_start = s * fx.Int32(k_split)
    m_end = arith.select(m_start + fx.Int32(k_split) < fx.Int32(R), m_start + fx.Int32(k_split), fx.Int32(R))
    tile_n = out_tile % fx.Int32(n_tiles)
    grouped_gemm_bf16_variable_k_tile(
        GK,
        LC,
        OUT if S == 1 else PART,
        s,
        head,
        tile_n,
        m_start,
        m_end,
        lds,
        fx.Int32(OUT_M),
        fx.Int32(OUT_N),
        G=S,
        OUT_M=OUT_M,
        OUT_N=OUT_N,
        BLOCK_M=_BM,
        BLOCK_N=_BN,
        persistent=persistent,
        lds_chunk_stride=KV_WGRAD_LDS_CHUNK_STRIDE,
        a_split=(GV, hl * (dn + dr), hl * dv, head * fx.Int32(dn + dr), head * fx.Int32(dv)),
        store_c=None if S == 1 else _StoreBandF32(PART, OUT_N),
    )
    if S > 1:
        _split_k_fixup(
            PART,
            OUT,
            CNT,
            lds,
            s,
            out_tile,
            head * fx.Int32(_BM),
            tile_n * fx.Int32(_BN),
            S=S,
            OUT_M=OUT_M,
            OUT_N=OUT_N,
            num_out=per_split,
        )


@functools.lru_cache(maxsize=64)
def _make_kernel(R, hl, dn, dr, dv, Ckv, S, k_split):
    SharedStorage = _make_shared_storage(_BM, _BN, chunk_stride=KV_WGRAD_LDS_CHUNK_STRIDE)
    dims = dict(R=R, hl=hl, dn=dn, dr=dr, dv=dv, Ckv=Ckv, S=S, k_split=k_split)

    @flyc.kernel(name="ulysses_mla_kv_wgrad_kernel", known_block_size=[_THREADS, 1, 1])
    def kernel(GK: fx.Tensor, GV: fx.Tensor, LC: fx.Tensor, PART: fx.Tensor, OUT: fx.Tensor, CNT: fx.Tensor):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pid, _b, _c = fx.block_idx
        kv_wgrad_tile(pid, GK, GV, LC, PART, OUT, CNT, lds, persistent="staggered", **dims)

    return kernel, kv_wgrad_tiles(hl, Ckv, S)


@flyc.jit
def _launch(
    GK,
    GV,
    LC,
    PART,
    OUT,
    CNT,
    R: fx.Constexpr[int],
    hl: fx.Constexpr[int],
    dn: fx.Constexpr[int],
    dr: fx.Constexpr[int],
    dv: fx.Constexpr[int],
    Ckv: fx.Constexpr[int],
    S: fx.Constexpr[int],
    k_split: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel, grid = _make_kernel(R, hl, dn, dr, dv, Ckv, S, k_split)
    kernel(GK, GV, LC, PART, OUT, CNT, value_attrs=make_value_attrs(2, 0, "512,512")).launch(
        grid=(grid, 1, 1), block=(_THREADS, 1, 1), stream=stream
    )


def ulysses_mla_kv_wgrad(gk, gv, lc, out, *, dn, dr, dv, S=None):
    """``gk`` [R, hl, dn + dr], ``gv`` [R, hl, dv] and ``lc`` [R, Ckv] (contiguous); writes ``out``
    [hl, dn + dv, Ckv] (rows ``[:, :dn]`` from ``gk[..., :dn]``, ``[:, dn:]`` from ``gv``)."""
    R, hl = gk.shape[0], gk.shape[1]
    Ckv = lc.shape[1]
    assert tuple(gk.shape) == (R, hl, dn + dr) and tuple(gv.shape) == (R, hl, dv) and lc.shape[0] == R
    assert tuple(out.shape) == (hl, dn + dv, Ckv) and out.is_contiguous()
    assert gk.is_contiguous() and gv.is_contiguous() and lc.is_contiguous()
    num_cus = torch.cuda.get_device_properties(gk.device).multi_processor_count
    S, k_split = kv_wgrad_splits(R, hl, Ckv, num_cus, S)
    assert S == 1 or kv_wgrad_tiles(hl, Ckv, S) <= num_cus, "the split-K slices wait for each other"
    part = out if S == 1 else torch.empty((S,) + tuple(out.shape), dtype=torch.float32, device=out.device)
    cnt = split_k_counters(out.device, 2 * kv_wgrad_tiles(hl, Ckv, 1))
    run_launcher(
        _launch, gk, gv, lc, part, out, cnt, R, hl, dn, dr, dv, Ckv, S, k_split, torch.cuda.current_stream()
    )
    return out
