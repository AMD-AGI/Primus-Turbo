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

"""Ulysses GemmRS for MLA: the latent-gradient GEMMs of the local heads fused with the
reduce-scatter of the partial latent gradients (pull mode, sbhd).

Every rank holds the output gradients of its ``hl`` heads over the gathered sequence (``R = n * T``
rows, row ``src * T + t``): ``gq``, ``gk`` ``[R, hl * (dn + dr)]`` and ``gv`` ``[R, hl * dv]``. GEMM
units write this rank's partial latent gradients into its symmetric staging bank, two row-major
regions ``Q [R, Cq]`` and ``KV [R, Ckv + dr]``, and raise one flag per unit:

    q units     Q = gq @ Wq                                          ``[R, Cq]``
    kv units    KV[:, :Ckv] = gk @ B' + gv @ Wv                      (the second GEMM adds onto the first)
    pe unit     KV[:, Ckv:] = gk @ B' = sum over heads of gk[..., dn:]

``B' [hl * (dn + dr), Ckv + dr]`` stacks every head's ``Wk_nope`` over zero rows for ``k_pe`` and an
identity block mapping each head's ``k_pe`` columns onto the ``dr`` shared columns, so the head sum
of ``dk_pe`` rides the k GEMM. Weights come transposed (``B_T``, the NT tile's operand).

Reduce units pull this rank's ``T`` rows of every peer's staging, sum the ``n`` partials in a fixed
order in fp32 (bitwise reproducible) and write ``d_q_latent [T, Cq]`` / ``d_kv_latent [T, Ckv + dr]``.

All units come from one claim table: GEMM units m-major, m-tiles in the same order on every rank,
and each reduce unit after every GEMM unit it waits for (plus a lag of about one wave). A waiting
reduce unit depends only on GEMM units claimed before it on every rank, and GEMM units never wait,
so the launch cannot deadlock.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
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
    _pack_out_pair,
    make_bf16_fp16_tile_tensor,
    make_row_band_resource,
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

# 16 B loads in flight per thread in a reduce unit (sources x vectors): remote reads are latency bound
_REDUCE_LOADS = 32


def _accumulate_store_c():
    """A ``StoreCBf16`` writing ``C = acc + C``. Defined in a function so the kernel cache key (the
    source of reached functions) covers the class body."""

    class _StoreCBf16Accumulate(StoreCBf16):
        """The add is in fp32 before one bf16 rounding. ``C`` was written by the same lanes with
        system-scope stores, so it is read back bypassing the caches. Only the paired-column store of
        a full 256-wide tile is implemented."""

        def store_band_pair16(self, frag_even, frag_odd, base_row, base_col, n_tiles_b, mask_cols=True):
            rsrc = make_row_band_resource(self.c_base, base_row, self.c_rows, self.c_cols, 2)
            n_tiles_a = len(frag_even) // n_tiles_b
            lane_col = self.lane_id % 16
            row_bytes = self.c_cols * 2
            base_off = ((self.lane_id // 16) * 4) * row_bytes + (base_col + lane_col * 2) * 2
            for ti in range_constexpr(n_tiles_a):
                offs = [base_off + (ti * 16 + r) * row_bytes for r in range_constexpr(4)]
                prev = [
                    [
                        Vec.from_elements(
                            [
                                buffer_load(
                                    rsrc,
                                    (off + j * 64) // 4,
                                    vec_width=1,
                                    dtype=fx.T.i32,
                                    cache_modifier=SYS_NT,
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
                            Vec(frag_even[t])[r] + prev[r][j][0].to(fx.Float32),
                            Vec(frag_odd[t])[r] + prev[r][j][1].to(fx.Float32),
                            self.out_ty,
                        )
                        buffer_store(
                            Vec.from_elements([packed], fx.Int32).bitcast(self.out_ty),
                            rsrc,
                            offs[r] + j * 64,
                            cache_modifier=self.cache_modifier,
                            offset_is_bytes=True,
                        )

        def _unsupported(self, *args, **kwargs):
            raise NotImplementedError("the accumulate epilogue only covers full 256-wide column tiles")

        store = store_band16 = store_trans16 = _unsupported

    return _StoreCBf16Accumulate


@ASTRewriter.transform
def _gemm_unit(thread_index, g, order_res, flag_base, bank_addr, GQ, GK, GV, WQT, WKT, WVT, lds, epoch, **c):
    """GEMM unit ``g`` (m-major over the m-tile order) into the own staging, then its flag."""
    R, R_pad, Cq, Ckv, dn, dr, dv, hl, BM, BN = (
        c[k] for k in ("R", "R_pad", "Cq", "Ckv", "dn", "dr", "dv", "hl", "BM", "BN")
    )
    nq_tiles, nkv_tiles, per_m = c["nq_tiles"], c["nkv_tiles"], c["per_m"]
    Nk, W2 = hl * (dn + dr), Ckv + dr
    gemm_tile = functools.partial(
        gemm_bf16_tile,
        "nt",
        pair_n=True,
        BLOCK_M=BM,
        BLOCK_N=BN,
        out_fp16=False,
        nt_vmcnt=c["nt_vmcnt"],
        c_cache_modifier=SYS_NT,
        persistent="staggered",
    )
    tile_m = _readfirstlane_i32(buffer_load(order_res, g // fx.Int32(per_m), vec_width=1, dtype=fx.T.i32))
    tile_n = g % fx.Int32(per_m)
    q_c = make_bf16_fp16_tile_tensor(bank_addr, fx.Int64(0), R_pad * Cq)
    kv_c = make_bf16_fp16_tile_tensor(bank_addr, fx.Int64(R_pad * Cq * 2), R_pad * W2)
    gk = make_bf16_fp16_tile_tensor(i64_base(GK), fx.Int64(0), R * Nk)
    wkt = make_bf16_fp16_tile_tensor(i64_base(WKT), fx.Int64(0), W2 * Nk)
    if tile_n < fx.Int32(nq_tiles):
        gemm_tile(
            make_bf16_fp16_tile_tensor(i64_base(GQ), fx.Int64(0), R * Nk),
            make_bf16_fp16_tile_tensor(i64_base(WQT), fx.Int64(0), Cq * Nk),
            q_c,
            fx.Int32(R),
            fx.Int32(Cq),
            lds,
            tile_m,
            tile_n,
            K=Nk,
            n_tail=0,
        )
    else:
        tk = tile_n - fx.Int32(nq_tiles)
        gemm_tile(gk, wkt, kv_c, fx.Int32(R), fx.Int32(W2), lds, tile_m, tk, K=Nk, n_tail=dr)
        if tk < fx.Int32(nkv_tiles):
            # the v GEMM reads back what the k GEMM's lanes stored
            fx.rocdl.s_waitcnt(0)
            fx.gpu.barrier()
            gemm_tile(
                make_bf16_fp16_tile_tensor(i64_base(GV), fx.Int64(0), R * hl * dv),
                make_bf16_fp16_tile_tensor(i64_base(WVT), fx.Int64(0), Ckv * hl * dv),
                kv_c,
                fx.Int32(R),
                fx.Int32(W2),
                lds,
                tile_m,
                tk,
                K=hl * dv,
                n_tail=0,
                store_c_cls=_accumulate_store_c(),
            )
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        st(flag_base, tile_m * fx.Int32(per_m) + tile_n, epoch, scope="sys", order="release")


def _reduce_rows(thread_index, srcs, region, out_res, vpr, t0, **c):
    """Rows ``[t0, t0 + RB)`` of this rank's slice of one staging region, summed over the sources
    in order, into ``out_res`` (``vpr`` 16 B vectors per row)."""
    T, RB, rank = c["T"], c["RB"], c["rank"]
    n = len(srcs)
    batch = max(1, _REDUCE_LOADS // n)
    num_vecs = RB * vpr
    per_iter = BLOCK_THREADS * batch
    for it in range_constexpr(ceildiv(num_vecs, per_iter)):
        coords = []
        for j in range_constexpr(batch):
            if const_expr(it * per_iter + j * BLOCK_THREADS < num_vecs):
                v = fx.Int32(it * per_iter + j * BLOCK_THREADS) + thread_index
                t = t0 + v // fx.Int32(vpr)
                col = v % fx.Int32(vpr)
                src_word = fx.Int32(region) + ((t + fx.Int32(rank * T)) * fx.Int32(vpr) + col) * fx.Int32(
                    VEC_I32
                )
                dst_word = (t * fx.Int32(vpr) + col) * fx.Int32(VEC_I32)
                coords.append((src_word, dst_word, (t < fx.Int32(T)) & (v < fx.Int32(num_vecs))))
        vals = [
            [
                buffer_load(src, sw, vec_width=VEC_I32, dtype=fx.T.i32, mask=ok, cache_modifier=SYS_NT)
                for src in srcs
            ]
            for sw, _, ok in coords
        ]
        for parts, (_, dw, ok) in zip(vals, coords):
            acc = Vec(parts[0]).bitcast(fx.BFloat16).to(fx.Float32)
            for s in range_constexpr(1, n):
                acc = acc + Vec(parts[s]).bitcast(fx.BFloat16).to(fx.Float32)
            buffer_store(acc.to(fx.BFloat16).bitcast(fx.Int32), out_res, dw, mask=ok)


@ASTRewriter.transform
def _reduce_unit(thread_index, i, peers_base, bank_offset, dlq_res, dlkv_res, epoch, **c):
    """Rows ``[i * RB, (i + 1) * RB)`` of this rank's slice: wait for every source's GEMM units over
    them, then sum the partials."""
    T, RB, BM, R_pad, Cq, Ckv, dr, rank, num_ranks, per_m = (
        c[k] for k in ("T", "RB", "BM", "R_pad", "Cq", "Ckv", "dr", "rank", "num_ranks", "per_m")
    )
    t0 = i * fx.Int32(RB)
    g0 = fx.Int32(rank * T) + t0
    g_last = arith.select(t0 + fx.Int32(RB) <= fx.Int32(T), g0 + fx.Int32(RB - 1), fx.Int32(rank * T + T - 1))
    m_first, m_last = g0 // fx.Int32(BM), g_last // fx.Int32(BM)
    slots = [m * fx.Int32(per_m) + fx.Int32(j) for m in (m_first, m_last) for j in range(per_m)]
    if thread_index == fx.Int32(0):
        for s in range_constexpr(num_ranks):
            heap = ld(peers_base, fx.Int32(s), dtype=fx.T.i64)
            spin_flags(
                heap + fx.Int64(c["flag_region"] * FLAG_REGION_BYTES), slots, epoch, rank, s, "gemm_rs"
            )
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    # the loads below bypass the caches; this only keeps them below the flag wait
    memory_fence(order="acquire", scope="workgroup")
    W2 = Ckv + dr
    srcs = [
        create_buffer_resource_from_addr(
            ld(peers_base, fx.Int32(s), dtype=fx.T.i64) + bank_offset, num_records_bytes=R_pad * (Cq + W2) * 2
        )
        for s in range_constexpr(num_ranks)
    ]
    _reduce_rows(thread_index, srcs, 0, dlq_res, Cq // VEC_BF16, t0, **c)
    _reduce_rows(thread_index, srcs, R_pad * Cq // 2, dlkv_res, W2 // VEC_BF16, t0, **c)


@ASTRewriter.transform
def _unit_loop(
    thread_index,
    state_base,
    claim_slot,
    table_res,
    order_res,
    peers_base,
    bank_offset,
    my_base,
    tensors,
    lds,
    epoch,
    trace_base,
    **c,
):
    """Claim table entries: ``g >= 0`` a GEMM unit, ``-1 - i`` reduce unit ``i``."""
    T, Cq, Ckv, dr = c["T"], c["Cq"], c["Ckv"], c["dr"]
    GQ, GK, GV, WQT, WKT, WVT, DLQ, DLKV = tensors
    dlq_res = create_buffer_resource(DLQ, num_records_bytes=T * Cq * 2)
    dlkv_res = create_buffer_resource(DLKV, num_records_bytes=T * (Ckv + dr) * 2)
    flag_base = my_base + fx.Int64(c["flag_region"] * FLAG_REGION_BYTES)
    num_entries = c["num_entries"]
    q = claim_unit(thread_index, state_base, claim_slot, STATE_TILE)
    while q < fx.Int32(num_entries):
        if const_expr(trace_base is not None):
            if thread_index == fx.Int32(0):
                st(trace_base, q * fx.Int32(2), read_clock())
        entry = _readfirstlane_i32(buffer_load(table_res, q, vec_width=1, dtype=fx.T.i32))
        if entry >= fx.Int32(0):
            _gemm_unit(
                thread_index,
                entry,
                order_res,
                flag_base,
                my_base + bank_offset,
                GQ,
                GK,
                GV,
                WQT,
                WKT,
                WVT,
                lds,
                epoch,
                **c,
            )
        else:
            _reduce_unit(
                thread_index, fx.Int32(-1) - entry, peers_base, bank_offset, dlq_res, dlkv_res, epoch, **c
            )
        if const_expr(trace_base is not None):
            fx.rocdl.s_waitcnt(0)
            if thread_index == fx.Int32(0):
                st(trace_base, q * fx.Int32(2) + 1, read_clock())
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
    RB,
    num_entries,
    num_cus,
    flag_region,
    bank_bytes,
    nt_vmcnt,
    trace,
):
    assert BN == 256 and Cq % BN == 0 and Ckv % BN == 0, "latent widths must fill whole column tiles"
    assert dr % VEC_BF16 == 0 and 0 < dr <= 128 and ceildiv(dr, 32) in (2, 4)
    assert (hl * (dn + dr)) % BLOCK_K == 0 and (hl * dv) % BLOCK_K == 0
    R = num_ranks * T
    m_tiles = ceildiv(R, BM)
    R_pad = m_tiles * BM
    nq_tiles, nkv_tiles = Cq // BN, Ckv // BN
    per_m = nq_tiles + nkv_tiles + 1
    assert RB <= BM, "a reduce unit spans at most two m-tiles"
    assert m_tiles * per_m <= FLAGS_PER_REGION, "too many GEMM units for one flag region"
    assert R_pad * (Cq + Ckv + dr) * 2 <= bank_bytes and R_pad * (Cq + Ckv + dr) < 2**31
    assert R * hl * (dn + dr) < 2**31, "i32 element offsets overflow"
    grid = min(num_entries, num_cus)
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
        RB=RB,
        nq_tiles=nq_tiles,
        nkv_tiles=nkv_tiles,
        per_m=per_m,
        num_entries=num_entries,
        flag_region=flag_region,
        nt_vmcnt=nt_vmcnt,
    )

    @flyc.kernel(name="ulysses_mla_gemm_rs_kernel", known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        GQ: fx.Tensor,
        GK: fx.Tensor,
        GV: fx.Tensor,
        WQT: fx.Tensor,
        WKT: fx.Tensor,
        WVT: fx.Tensor,
        DLQ: fx.Tensor,
        DLKV: fx.Tensor,
        TABLE: fx.Tensor,
        ORDER: fx.Tensor,
        PEER_PTRS: fx.Tensor,
        STATE: fx.Tensor,
        TRACE: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        claim_slot = fx.Int32(fx.ptrtoint(lds.lds_row_idx.ptr))
        state_base = i64_base(STATE)
        peers_base = i64_base(PEER_PTRS)
        epoch = ld(state_base, fx.Int32(0), dtype=fx.T.i64)
        bank_offset = fx.Int64(FLAG_BYTES) + (epoch & fx.Int64(1)) * fx.Int64(bank_bytes)
        my_base = ld(peers_base, fx.Int32(rank), dtype=fx.T.i64)
        # debug timeline (s_memrealtime): [start, end] per claim
        trace_base = i64_base(TRACE) if trace else None
        _unit_loop(
            thread_index,
            state_base,
            claim_slot,
            create_buffer_resource(TABLE, num_records_bytes=num_entries * 4),
            create_buffer_resource(ORDER, num_records_bytes=m_tiles * 4),
            peers_base,
            bank_offset,
            my_base,
            (GQ, GK, GV, WQT, WKT, WVT, DLQ, DLKV),
            lds,
            epoch,
            trace_base,
            **consts,
        )
        exit_and_bump_epoch(thread_index, state_base, epoch, grid)

    return kernel, grid


@flyc.jit
def _launch(
    GQ,
    GK,
    GV,
    WQT,
    WKT,
    WVT,
    DLQ,
    DLKV,
    TABLE,
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
    RB: fx.Constexpr[int],
    num_entries: fx.Constexpr[int],
    num_cus: fx.Constexpr[int],
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
        RB,
        num_entries,
        num_cus,
        flag_region,
        bank_bytes,
        nt_vmcnt,
        trace,
    )
    kernel(
        GQ,
        GK,
        GV,
        WQT,
        WKT,
        WVT,
        DLQ,
        DLKV,
        TABLE,
        ORDER,
        PEER_PTRS,
        STATE,
        TRACE,
        value_attrs=make_value_attrs(2, 0, "512,512"),
    ).launch(grid=(grid, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)


def gemm_rs_bank_bytes(R, Cq, Ckv, dr, BM=256):
    return ceildiv(R, BM) * BM * (Cq + Ckv + dr) * 2


@functools.lru_cache(maxsize=64)
def _claim_table(T, num_ranks, rank, Cq, Ckv, BM, BN, RB, lag, device):
    """The m-tile order (by the last row of any slice the tile covers, so every rank's early rows
    finish first; the same on every rank) and the claim table: GEMM units m-major, each reduce unit
    ``lag`` GEMM units after the last one it waits for."""
    R = num_ranks * T
    m_tiles = ceildiv(R, BM)
    per_m = Cq // BN + Ckv // BN + 1
    keys = []
    for m in range(m_tiles):
        last = min((m + 1) * BM, R) - 1
        src_first, src_last = (m * BM) // T, last // T
        land = max(
            (last if s == src_last else (s + 1) * T - 1) - s * T for s in range(src_first, src_last + 1)
        )
        keys.append((land, src_first, m))
    order = [m for *_, m in sorted(keys)]
    pos = {m: p for p, m in enumerate(order)}
    num_gemm = m_tiles * per_m
    after = [[] for _ in range(num_gemm)]
    for i in range(ceildiv(T, RB)):
        g0, g_last = rank * T + i * RB, rank * T + min((i + 1) * RB, T) - 1
        dep = max(pos[g0 // BM], pos[g_last // BM]) * per_m + per_m - 1
        after[min(dep + lag, num_gemm - 1)].append(-1 - i)
    table = []
    for g in range(num_gemm):
        table.append(g)
        table += after[g]
    return (
        torch.tensor(order, dtype=torch.int32, device=device),
        torch.tensor(table, dtype=torch.int32, device=device),
    )


def gemm_rs_trace_len(T, num_ranks, Cq, Ckv, BM=256, BN=256, RB=32):
    """Entries of the debug timeline: 2 per claim-table entry."""
    return 2 * (ceildiv(num_ranks * T, BM) * (Cq // BN + Ckv // BN + 1) + ceildiv(T, RB))


def ulysses_mla_gemm_rs(
    gq,
    gk,
    gv,
    wq_t,
    wk_t,
    wv_t,
    d_q_latent,
    d_kv_latent,
    workspace,
    *,
    dn,
    dr,
    dv,
    flag_region,
    BM=256,
    BN=256,
    RB=32,
    lag=None,
    trace=None,
):
    """``gq`` / ``gk`` [n * T, hl * (dn + dr)] and ``gv`` [n * T, hl * dv] this rank's head gradients
    over the gathered sequence (sbhd order); ``wq_t`` [Cq, hl * (dn + dr)] (``Wq`` rows of the local
    heads, transposed), ``wk_t`` [Ckv + dr, hl * (dn + dr)] (``B'`` transposed) and ``wv_t`` [Ckv, hl *
    dv]. Writes this rank's rows of the summed latent gradients, ``d_q_latent`` [T, Cq] and
    ``d_kv_latent`` [T, Ckv + dr]. ``trace`` (debug): an int64 tensor of ``gemm_rs_trace_len(...)``
    entries receiving the ``s_memrealtime`` timeline."""
    num_ranks, rank = workspace.num_ranks, workspace.rank
    T, Cq = d_q_latent.shape
    Ckv = d_kv_latent.shape[1] - dr
    hl = gv.shape[1] // dv
    R = num_ranks * T
    Nk = hl * (dn + dr)
    assert tuple(gq.shape) == (R, Nk) and tuple(gk.shape) == (R, Nk) and tuple(gv.shape) == (R, hl * dv)
    assert tuple(wq_t.shape) == (Cq, Nk) and tuple(wk_t.shape) == (Ckv + dr, Nk)
    assert tuple(wv_t.shape) == (Ckv, hl * dv) and d_kv_latent.shape[0] == T
    tensors = (gq, gk, gv, wq_t, wk_t, wv_t, d_q_latent, d_kv_latent)
    assert all(t.is_contiguous() for t in tensors)
    num_cus = torch.cuda.get_device_properties(gq.device).multi_processor_count
    lag = num_cus if lag is None else lag
    order, table = _claim_table(T, num_ranks, rank, Cq, Ckv, BM, BN, RB, lag, gq.device)
    run_launcher(
        _launch,
        *tensors,
        table,
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
        RB,
        table.numel(),
        num_cus,
        flag_region,
        workspace.bank_bytes,
        3,
        trace is not None,
        torch.cuda.current_stream(),
    )
