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

"""Ulysses AgGemm for MLA: the all-gather of the latent rows fused into the up-projections of the
local heads (push mode, sbhd).

Every rank holds ``T`` tokens of ``[q_latent | c_kv | k_pe]``. Push work copies sub-tiles of
``SUB`` local rows into every rank's symmetric recv bank, at the rows the tokens take in the gathered
sequence (``src * T + t``), and raises the destination's flag ``(src, sub)``. The bank holds three
row-major regions, each the ``A`` of one GEMM: ``Lq [R, Cq]``, ``Lc [R, Ckv]`` and ``PE [R, dr]``.

GEMM tiles (claimed m-major, m-tiles in the order their rows land) wait for the flags of their rows
and then read ``A`` straight from the bank:

    q tiles     q = Lq @ Wq^T                                   ``[R, h * (dn + dr)]``
    k tiles     k[..., :dn] = Lc @ Wk^T                         ``[R, h, dn]`` of ``k``
    v tiles     v = Lc @ Wv^T                                   ``[R, h * dv]``

``Wkv`` is the rank's rows of the kv up-projection reordered to ``[k_nope of every local head; v
of every local head]``, so a k tile covers ``BN / dn`` whole heads. After its GEMM tiles every
m-tile has a copy unit: ``k[..., dn:] = PE`` (k_pe shared by the heads) and, for the weight
gradients, its rows of ``Lq`` / ``Lc`` out of the bank into ordinary tensors.

The first ``num_push`` blocks push until every unit is claimed, then compute tiles; a block only
waits once every push unit is claimed by a running block, so the launch cannot deadlock.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr

from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.mega.bf16.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_tile,
)
from primus_turbo.flydsl.ulysses.a2a_gemm_kernel import _copy_rows
from primus_turbo.flydsl.ulysses.common import (
    BLOCK_THREADS,
    STATE_TILE,
    SYS_NT,
    VEC_BF16,
    VEC_I32,
    claim_unit,
    exit_and_bump_epoch,
    i64_base,
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
)
from primus_turbo.flydsl.utils.gemm_helper import (
    BLOCK_K,
    StoreCBf16,
    make_bf16_fp16_tile_tensor,
    make_value_attrs,
)
from primus_turbo.flydsl.utils.prims import (
    _readfirstlane_i32,
    ceildiv,
    ld,
    memory_fence,
    read_clock,
    st,
)

# 16 B loads in flight per thread in a copy unit: under the GEMM's traffic a copy is latency bound
_COPY_BATCH = 32


def _k_nope_store_c(dn, dr):
    """A ``StoreCBf16`` writing the ``[R, h * dn]`` k_nope columns into ``k [R, h, dn + dr]``. Defined
    in a function so the kernel cache key (the source of reached functions) covers the class body."""

    class _StoreCKNope(StoreCBf16):
        """Only the paired-column store of a full 256-wide tile: a wave writes 32 aligned columns, which
        never straddle a head (``dn % 32 == 0``), so one per-wave shift maps them into ``k``."""

        def store_band_pair16(self, frag_even, frag_odd, base_row, base_col, n_tiles_b, mask_cols=True):
            col = base_col + (base_col // fx.Int32(dn)) * fx.Int32(dr)
            super().store_band_pair16(frag_even, frag_odd, base_row, col, n_tiles_b, mask_cols=False)

        def _unsupported(self, *args, **kwargs):
            raise NotImplementedError("the k_nope epilogue only covers full 256-wide column tiles")

        store = store_band16 = store_trans16 = _unsupported

    return _StoreCKNope


@ASTRewriter.transform
def _push_unit(thread_index, sub, dst, dst_heap, bank_offset, xq_res, xkv_res, epoch, **c):
    """Local rows of sub-tile ``sub`` into ``dst``'s three bank regions, then ``dst``'s flag."""
    T, R_pad, Cq, Ckv, dr, SUB, rank = (c[k] for k in ("T", "R_pad", "Cq", "Ckv", "dr", "SUB", "rank"))
    bank_res = create_buffer_resource_from_addr(
        dst_heap + bank_offset, num_records_bytes=R_pad * (Cq + Ckv + dr) * 2
    )
    vq, vkv, vc, vpe = Cq // VEC_BF16, (Ckv + dr) // VEC_BF16, Ckv // VEC_BF16, dr // VEC_BF16
    # (source, source vectors per row, first source vector, region word offset, vectors per row)
    parts = (
        (xq_res, vq, 0, 0, vq),
        (xkv_res, vkv, 0, R_pad * Cq // 2, vc),
        (xkv_res, vkv, vc, R_pad * (Cq + Ckv) // 2, vpe),
    )
    for i in range_constexpr(len(parts)):
        src_res, src_vpr, src_v0, region, vpr = parts[i]

        def coords(v, src_vpr=src_vpr, src_v0=src_v0, region=region, vpr=vpr):
            r = v // fx.Int32(vpr)
            col = v % fx.Int32(vpr)
            t = sub * fx.Int32(SUB) + r
            src_word = (t * fx.Int32(src_vpr) + fx.Int32(src_v0) + col) * fx.Int32(VEC_I32)
            dst_word = fx.Int32(region) + ((t + fx.Int32(rank * T)) * fx.Int32(vpr) + col) * fx.Int32(VEC_I32)
            return src_word, dst_word, t < fx.Int32(T)

        _copy_rows(thread_index, SUB * vpr, coords, src_res, bank_res, 0, SYS_NT)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        flag_base = dst_heap + fx.Int64(c["flag_region"] * FLAG_REGION_BYTES)
        st(flag_base, fx.Int32(rank * c["n_sub"]) + sub, epoch, scope="sys", order="release")


@ASTRewriter.transform
def _push_loop(
    thread_index, state_base, claim_slot, peers_base, bank_offset, xq_res, xkv_res, epoch, trace_base, **c
):
    """Claim ``(sub, dst = rank + i)`` units, sub-major so every rank's early rows land first."""
    num_ranks = c["num_ranks"]
    num_units = c["n_sub"] * num_ranks
    unit = claim_unit(thread_index, state_base, claim_slot)
    while unit < fx.Int32(num_units):
        sub = unit // fx.Int32(num_ranks)
        dst = (fx.Int32(c["rank"]) + unit % fx.Int32(num_ranks)) % fx.Int32(num_ranks)
        dst_heap = ld(peers_base, dst, dtype=fx.T.i64)
        if const_expr(trace_base is not None):
            if thread_index == fx.Int32(0):
                st(trace_base, fx.Int32(c["trace_units_at"]) + unit * fx.Int32(2), read_clock())
        _push_unit(thread_index, sub, dst, dst_heap, bank_offset, xq_res, xkv_res, epoch, **c)
        if const_expr(trace_base is not None):
            if thread_index == fx.Int32(0):
                st(trace_base, fx.Int32(c["trace_units_at"]) + unit * fx.Int32(2) + 1, read_clock())
        unit = claim_unit(thread_index, state_base, claim_slot)


def _row_slots(tile_m, BM, SUB, T, R, n_sub, i32=fx.Int32, select=arith.select, **_):
    """Flags of ``tile_m``'s rows. With ``T % SUB == 0`` the ``SUB``-row pieces of the tile are whole
    sub-tiles: their first rows. Otherwise a piece can cross a source boundary ``b``, and the run
    before it can hold a short last sub-tile besides the tail of the one before: each piece's first
    and last row, plus ``b - 1`` and ``b`` for every boundary in the tile (any run of at most ``SUB``
    rows inside one source spans at most two sub-tiles, its first and last row's).

    ``i32`` / ``select`` default to the device ops; ``int`` and a Python conditional evaluate the same
    slots on the host."""
    m0 = tile_m * i32(BM)
    rows = [m0 + i32(p * SUB) for p in range(BM // SUB)]
    if T % SUB:
        rows += [m0 + i32(p * SUB + SUB - 1) for p in range(BM // SUB)]
        first_bound = (m0 + i32(T - 1)) // i32(T) * i32(T)
        for k in range(ceildiv(BM, T) + 1):
            for e in (-1, 0):
                row = first_bound + i32(k * T + e)
                inside = (row >= m0) & (row < m0 + i32(BM))
                rows.append(select(inside, row, m0))
    slots = []
    for row in rows:
        row = select(row < i32(R), row, i32(R - 1))
        src = row // i32(T)
        slots.append(src * i32(n_sub) + (row - src * i32(T)) // i32(SUB))
    return slots


@ASTRewriter.transform
def _wait_rows(thread_index, flag_base, tile_m, epoch, **c):
    """Every row of ``tile_m`` has landed."""
    if thread_index == fx.Int32(0):
        spin_flags(flag_base, _row_slots(tile_m, **c), epoch, c["rank"], -1, "ag_gemm")
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    # no cache invalidate here (``buffer_inv`` per tile costs ~15% of the kernel): stale lines of the
    # bank went with the block's system-scope acquire before its first tile; this only keeps the
    # compiler from hoisting the tile's loads above the flag wait
    memory_fence(order="acquire", scope="workgroup")


def _copy_out(thread_index, bank_res, region, out_res, cols, row0, rows, **c):
    """Rows ``[row0, row0 + rows)`` of a bank region into an ordinary tensor, ``_COPY_BATCH`` loads in
    flight per thread."""
    R, batch = c["R"], _COPY_BATCH
    vpr = cols // VEC_BF16
    num_vecs = rows * vpr
    per_iter = BLOCK_THREADS * batch
    for it in range_constexpr(ceildiv(num_vecs, per_iter)):
        coords = []
        for j in range_constexpr(batch):
            if const_expr(it * per_iter + j * BLOCK_THREADS < num_vecs):
                v = fx.Int32(it * per_iter + j * BLOCK_THREADS) + thread_index
                t = row0 + v // fx.Int32(vpr)
                word = (t * fx.Int32(vpr) + v % fx.Int32(vpr)) * fx.Int32(VEC_I32)
                coords.append((word, (t < fx.Int32(R)) & (v < fx.Int32(num_vecs))))
        vals = [
            buffer_load(bank_res, fx.Int32(region) + w, vec_width=VEC_I32, dtype=fx.T.i32, mask=ok)
            for w, ok in coords
        ]
        for val, (w, ok) in zip(vals, coords):
            buffer_store(val, out_res, w, mask=ok)


def _copy_unit(thread_index, bank_res, tile_m, part, OK, LQ, LC, **c):
    """Rows ``part`` of ``copy_parts`` of the m-tile: ``k[rows, :, dn:] = PE[rows]`` for every local
    head and, when saved, the rows of ``Lq`` / ``Lc``. Separate units rather than a share of every
    GEMM tile: stores count in ``vmcnt``, so a copy's loads wait for the C stores before them to drain,
    once per unit; parts spread an m-tile's copies over blocks (a single block is latency bound)."""
    BM, R, R_pad, Cq, Ckv, dn, dr, hl = (c[k] for k in ("BM", "R", "R_pad", "Cq", "Ckv", "dn", "dr", "hl"))
    rows = BM // c["copy_parts"]
    row0 = tile_m * fx.Int32(BM) + part * fx.Int32(rows)
    k_res = create_buffer_resource(OK, num_records_bytes=R * hl * (dn + dr) * 2)
    vpe = dr // VEC_BF16
    region = R_pad * (Cq + Ckv) // 2
    for it in range_constexpr(ceildiv(rows * vpe, BLOCK_THREADS)):
        v = fx.Int32(it * BLOCK_THREADS) + thread_index
        t = row0 + v // fx.Int32(vpe)
        col = v % fx.Int32(vpe)
        valid = (t < fx.Int32(R)) & (v < fx.Int32(rows * vpe))
        val = buffer_load(
            bank_res,
            fx.Int32(region) + (t * fx.Int32(vpe) + col) * fx.Int32(VEC_I32),
            vec_width=VEC_I32,
            dtype=fx.T.i32,
            mask=valid,
        )
        word = (t * fx.Int32(hl * (dn + dr)) + fx.Int32(dn)) // fx.Int32(2) + col * fx.Int32(VEC_I32)
        for h in range_constexpr(hl):
            buffer_store(val, k_res, word + fx.Int32(h * (dn + dr) // 2), mask=valid)
    if const_expr(c["save"]):
        lq_res = create_buffer_resource(LQ, num_records_bytes=R * Cq * 2)
        lc_res = create_buffer_resource(LC, num_records_bytes=R * Ckv * 2)
        _copy_out(thread_index, bank_res, 0, lq_res, Cq, row0, rows, **c)
        _copy_out(thread_index, bank_res, R_pad * Cq // 2, lc_res, Ckv, row0, rows, **c)


@ASTRewriter.transform
def _unit(
    thread_index, q, order_res, flag_base, bank_addr, WQ, WKV, OQ, OK, OV, LQ, LC, lds, epoch, trace_base, **c
):
    """Claim ``q``: per m-tile (in landing order) its GEMM tiles, then its copy units."""
    R, R_pad, Cq, Ckv, dn, dr, dv, hl, BM, BN = (
        c[k] for k in ("R", "R_pad", "Cq", "Ckv", "dn", "dr", "dv", "hl", "BM", "BN")
    )
    nq_tiles, nk_tiles, nv_tiles = c["nq_tiles"], c["nk_tiles"], c["nv_tiles"]
    n_tiles = nq_tiles + nk_tiles + nv_tiles
    per_m = n_tiles + c["copy_parts"]
    Nq = hl * (dn + dr)
    gemm_tile = functools.partial(
        gemm_bf16_tile, "nt", pair_n=True, BLOCK_M=BM, BLOCK_N=BN, out_fp16=False, nt_vmcnt=c["nt_vmcnt"]
    )
    tile_m = _readfirstlane_i32(buffer_load(order_res, q // fx.Int32(per_m), vec_width=1, dtype=fx.T.i32))
    tile_n = q % fx.Int32(per_m)
    if const_expr(trace_base is not None):
        if thread_index == fx.Int32(0):
            st(trace_base, q * fx.Int32(3), read_clock())
    _wait_rows(thread_index, flag_base, tile_m, epoch, **c)
    if const_expr(trace_base is not None):
        if thread_index == fx.Int32(0):
            st(trace_base, q * fx.Int32(3) + 1, read_clock())
    bank_res = create_buffer_resource_from_addr(bank_addr, num_records_bytes=R_pad * (Cq + Ckv + dr) * 2)
    lc_bytes = R_pad * Cq * 2
    if tile_n >= fx.Int32(n_tiles):
        _copy_unit(thread_index, bank_res, tile_m, tile_n - fx.Int32(n_tiles), OK, LQ, LC, **c)
    elif tile_n < fx.Int32(nq_tiles):
        gemm_tile(
            make_bf16_fp16_tile_tensor(bank_addr, fx.Int64(0), R_pad * Cq),
            make_bf16_fp16_tile_tensor(i64_base(WQ), fx.Int64(0), Nq * Cq),
            make_bf16_fp16_tile_tensor(i64_base(OQ), fx.Int64(0), R * Nq),
            fx.Int32(R),
            fx.Int32(Nq),
            lds,
            tile_m,
            tile_n,
            K=Cq,
            n_tail=Nq % BN,
            persistent="staggered",
        )
    else:
        tk = tile_n - fx.Int32(nq_tiles)
        a_c = make_bf16_fp16_tile_tensor(bank_addr, fx.Int64(lc_bytes), R_pad * Ckv)
        if tk < fx.Int32(nk_tiles):
            gemm_tile(
                a_c,
                make_bf16_fp16_tile_tensor(i64_base(WKV), fx.Int64(0), hl * dn * Ckv),
                make_bf16_fp16_tile_tensor(i64_base(OK), fx.Int64(0), R * hl * (dn + dr)),
                fx.Int32(R),
                fx.Int32(hl * (dn + dr)),
                lds,
                tile_m,
                tk,
                K=Ckv,
                n_tail=0,
                persistent="staggered",
                store_c_cls=_k_nope_store_c(dn, dr),
            )
        else:
            gemm_tile(
                a_c,
                make_bf16_fp16_tile_tensor(i64_base(WKV), fx.Int64(hl * dn * Ckv * 2), hl * dv * Ckv),
                make_bf16_fp16_tile_tensor(i64_base(OV), fx.Int64(0), R * hl * dv),
                fx.Int32(R),
                fx.Int32(hl * dv),
                lds,
                tile_m,
                tk - fx.Int32(nk_tiles),
                K=Ckv,
                n_tail=0,
                persistent="staggered",
            )
    if const_expr(trace_base is not None):
        fx.rocdl.s_waitcnt(0)
        if thread_index == fx.Int32(0):
            st(trace_base, q * fx.Int32(3) + 2, read_clock())


@ASTRewriter.transform
def _tile_loop(
    thread_index,
    state_base,
    claim_slot,
    order_res,
    flag_base,
    bank_addr,
    tensors,
    lds,
    epoch,
    trace_base,
    **c,
):
    num_units = c["num_units"]
    q = claim_unit(thread_index, state_base, claim_slot, STATE_TILE)
    while q < fx.Int32(num_units):
        _unit(thread_index, q, order_res, flag_base, bank_addr, *tensors, lds, epoch, trace_base, **c)
        q = claim_unit(thread_index, state_base, claim_slot, STATE_TILE)


@functools.lru_cache(maxsize=64)
def _make_kernel(
    T,
    Cq,
    Ckv,
    hl,
    dn,
    dr,
    dv,
    rank,
    num_ranks,
    BM,
    BN,
    SUB,
    num_push,
    copy_parts,
    num_cus,
    save,
    flag_region,
    bank_bytes,
    nt_vmcnt,
    trace,
):
    assert Cq % BLOCK_K == 0 and Ckv % BLOCK_K == 0, f"latent widths must divide by {BLOCK_K}"
    assert BN == 256 and dn % 32 == 0 and BN % dn == 0 and dr % VEC_BF16 == 0
    assert (hl * dn) % BN == 0 and (hl * dv) % BN == 0, "k_nope / v columns must fill whole tiles"
    assert BM % SUB == 0 and BM % copy_parts == 0
    R = num_ranks * T
    m_tiles = ceildiv(R, BM)
    R_pad = m_tiles * BM
    n_sub = ceildiv(T, SUB)
    nq_tiles = ceildiv(hl * (dn + dr), BN)
    nk_tiles, nv_tiles = hl * dn // BN, hl * dv // BN
    num_units = m_tiles * (nq_tiles + nk_tiles + nv_tiles + copy_parts)
    assert num_ranks * n_sub <= FLAGS_PER_REGION, "too many sub-tiles for one flag region"
    assert R_pad * (Cq + Ckv + dr) * 2 <= bank_bytes and R_pad * (Cq + Ckv + dr) < 2**31
    assert R * hl * (dn + dr) < 2**31, "i32 element offsets overflow"
    grid = min(num_units + num_push, num_cus)
    num_push = min(num_push, grid)
    SharedStorage = _make_shared_storage(BM, BN, num_lds_row_idx_entries=1)
    consts = dict(
        T=T,
        R=R,
        R_pad=R_pad,
        Cq=Cq,
        Ckv=Ckv,
        hl=hl,
        dn=dn,
        dr=dr,
        dv=dv,
        rank=rank,
        num_ranks=num_ranks,
        BM=BM,
        BN=BN,
        SUB=SUB,
        n_sub=n_sub,
        nq_tiles=nq_tiles,
        nk_tiles=nk_tiles,
        nv_tiles=nv_tiles,
        num_units=num_units,
        copy_parts=copy_parts,
        save=save,
        flag_region=flag_region,
        nt_vmcnt=nt_vmcnt,
        trace_units_at=3 * num_units,
    )

    @flyc.kernel(name="ulysses_mla_ag_gemm_kernel", known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        XQ: fx.Tensor,
        XKV: fx.Tensor,
        WQ: fx.Tensor,
        WKV: fx.Tensor,
        OQ: fx.Tensor,
        OK: fx.Tensor,
        OV: fx.Tensor,
        LQ: fx.Tensor,
        LC: fx.Tensor,
        ORDER: fx.Tensor,
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
            xq_res = create_buffer_resource(XQ, num_records_bytes=T * Cq * 2)
            xkv_res = create_buffer_resource(XKV, num_records_bytes=T * (Ckv + dr) * 2)
            _push_loop(
                thread_index,
                state_base,
                claim_slot,
                peers_base,
                bank_offset,
                xq_res,
                xkv_res,
                epoch,
                trace_base,
                **consts,
            )
        order_res = create_buffer_resource(ORDER, num_records_bytes=m_tiles * 4)
        # bank rows are rewritten by peers every 2 epochs: drop what this CU / XCD still caches of
        # them. Once per block is enough: a tile reads only rows whose flag it saw, so no line read
        # after this predates its producer's write (an m-tile's rows of a region are whole 128 B lines)
        memory_fence(order="acquire", scope="sys")
        _tile_loop(
            thread_index,
            state_base,
            claim_slot,
            order_res,
            my_base + fx.Int64(flag_region * FLAG_REGION_BYTES),
            my_base + bank_offset,
            (WQ, WKV, OQ, OK, OV, LQ, LC),
            lds,
            epoch,
            trace_base,
            **consts,
        )
        exit_and_bump_epoch(thread_index, state_base, epoch, grid)

    return kernel, grid


@flyc.jit
def _launch(
    XQ,
    XKV,
    WQ,
    WKV,
    OQ,
    OK,
    OV,
    LQ,
    LC,
    ORDER,
    PEER_PTRS,
    STATE,
    TRACE,
    T: fx.Constexpr[int],
    Cq: fx.Constexpr[int],
    Ckv: fx.Constexpr[int],
    hl: fx.Constexpr[int],
    dn: fx.Constexpr[int],
    dr: fx.Constexpr[int],
    dv: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    BM: fx.Constexpr[int],
    BN: fx.Constexpr[int],
    SUB: fx.Constexpr[int],
    num_push: fx.Constexpr[int],
    copy_parts: fx.Constexpr[int],
    num_cus: fx.Constexpr[int],
    save: fx.Constexpr[bool],
    flag_region: fx.Constexpr[int],
    bank_bytes: fx.Constexpr[int],
    nt_vmcnt: fx.Constexpr[int],
    trace: fx.Constexpr[bool],
    stream: fx.Stream,
):
    kernel, grid = _make_kernel(
        T,
        Cq,
        Ckv,
        hl,
        dn,
        dr,
        dv,
        rank,
        num_ranks,
        BM,
        BN,
        SUB,
        num_push,
        copy_parts,
        num_cus,
        save,
        flag_region,
        bank_bytes,
        nt_vmcnt,
        trace,
    )
    kernel(
        XQ,
        XKV,
        WQ,
        WKV,
        OQ,
        OK,
        OV,
        LQ,
        LC,
        ORDER,
        PEER_PTRS,
        STATE,
        TRACE,
        value_attrs=make_value_attrs(2, 0, "512,512"),
    ).launch(grid=(grid, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)


def ag_gemm_bank_bytes(R, Cq, Ckv, dr, BM=256):
    return ceildiv(R, BM) * BM * (Cq + Ckv + dr) * 2


def ag_gemm_trace_len(T, num_ranks, hl, dn, dr, dv, BM=256, BN=256, SUB=64, copy_parts=4):
    """Entries of the debug timeline: 3 per claim (an m-tile's GEMM tiles, then its copy units),
    then 2 per push unit."""
    units_per_m = ceildiv(hl * (dn + dr), BN) + hl * dn // BN + hl * dv // BN + copy_parts
    return 3 * ceildiv(num_ranks * T, BM) * units_per_m + 2 * ceildiv(T, SUB) * num_ranks


@functools.lru_cache(maxsize=64)
def _tile_order(T, num_ranks, rank, BM, SUB, device):
    """m-tiles by the push step their last row lands at (all ranks push in step), own rows first."""
    R = num_ranks * T
    keys = []
    for m in range(ceildiv(R, BM)):
        last = min((m + 1) * BM, R) - 1
        src_first, src_last = (m * BM) // T, last // T
        land = max(
            ((last if s == src_last else (s + 1) * T - 1) - s * T) // SUB
            for s in range(src_first, src_last + 1)
        )
        keys.append((land, (src_first - rank) % num_ranks, m))
    return torch.tensor([m for *_, m in sorted(keys)], dtype=torch.int32, device=device)


def ulysses_mla_ag_gemm(
    xq,
    xkv,
    wq,
    wkv,
    q,
    k,
    v,
    lq,
    lc,
    workspace,
    *,
    dn,
    dr,
    dv,
    flag_region,
    BM=256,
    BN=256,
    SUB=64,
    num_push=None,
    copy_parts=4,
    trace=None,
):
    """``xq`` [T, Cq] / ``xkv`` [T, Ckv + dr] this rank's latent rows (sbhd order); ``wq``
    [hl * (dn + dr), Cq] and ``wkv`` [hl * (dn + dv), Ckv] (rows: k_nope of every local head, then v)
    this rank's up-projection rows. Writes ``q`` / ``k`` [n * T, hl, dn + dr] and ``v`` [n * T, hl, dv]
    over the gathered sequence; ``lq`` [n * T, Cq] / ``lc`` [n * T, Ckv] receive the gathered latents
    unless they are ``None``. ``trace`` (debug): an int64 tensor of ``ag_gemm_trace_len(...)``
    entries receiving the ``s_memrealtime`` timeline."""
    num_ranks, rank = workspace.num_ranks, workspace.rank
    # enough push blocks to keep the links busy, measured on MI355X: 8 / 16 / 32 for 2 / 4 / 8 ranks
    num_push = 4 * num_ranks if num_push is None else num_push
    T, Cq = xq.shape
    Ckv = xkv.shape[1] - dr
    hl = wq.shape[0] // (dn + dr)
    R = num_ranks * T
    assert xkv.shape[0] == T and tuple(wkv.shape) == (hl * (dn + dv), Ckv)
    assert tuple(q.shape) == (R, hl, dn + dr) and tuple(k.shape) == q.shape and tuple(v.shape) == (R, hl, dv)
    tensors = (xq, xkv, wq, wkv, q, k, v)
    assert all(t.is_contiguous() for t in tensors)
    save = lq is not None
    assert not save or (
        lq.is_contiguous() and lc.is_contiguous() and lq.shape == (R, Cq) and lc.shape == (R, Ckv)
    )
    num_cus = torch.cuda.get_device_properties(xq.device).multi_processor_count
    order = _tile_order(T, num_ranks, rank, BM, SUB, xq.device)
    run_launcher(
        _launch,
        *tensors,
        lq if save else q,
        lc if save else q,
        order,
        workspace.peer_ptrs,
        workspace.state,
        workspace.state if trace is None else trace,
        T,
        Cq,
        Ckv,
        hl,
        dn,
        dr,
        dv,
        rank,
        num_ranks,
        BM,
        BN,
        SUB,
        num_push,
        copy_parts,
        num_cus,
        save,
        flag_region,
        workspace.bank_bytes,
        3,
        trace is not None,
        torch.cuda.current_stream(),
    )
