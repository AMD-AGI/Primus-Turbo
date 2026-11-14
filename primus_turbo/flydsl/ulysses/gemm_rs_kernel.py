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
    kv units    KV[:, :Ckv] = gk @ B' + gv @ Wv                      (one K loop over both products)
    pe unit     KV[:, Ckv:] = sum over heads of gk[..., dn:]           (a vector sum, no GEMM)

``B' [hl * (dn + dr), Ckv + dr]`` stacks every head's ``Wk_nope`` over zero rows for ``k_pe``; the kv
units use its first ``Ckv`` columns. Weights come transposed (``B_T``, the NT tile's operand).

Reduce units pull this rank's ``T`` rows of every peer's staging, sum the ``n`` partials in a fixed
order in fp32 (bitwise reproducible) and write ``d_q_latent [T, Cq]`` / ``d_kv_latent [T, Ckv + dr]``.

The launch is persistent: one block per CU (``num_blocks``) claims the items of a fixed dispatch
order from a counter. GEMM units go m-major, m-tiles in the same order on every rank; a reduce item
claims reduce units in row order from a second counter and waits for their flags. Reduce items sit
after the first GEMM wave (``num_red_cu``) and after the last GEMM unit (``num_tail_red``), as in
the GemmA2A. Claims only grow and GEMM units never wait, so the launch cannot deadlock.

Optionally the launch also computes the q and kv weight gradients (``kv_wgrad_kernel``), whose
tiles wait for nothing but their own K slices. The q tiles (the longest units of the launch, split
over K until they cover half the CUs) go ahead of the first GEMM wave: the GEMM and the reduce share
the other CUs while they run. The kv split-K tiles follow GEMM unit ``wgrad_at`` or, at ``wgrad_at
= num_gemm``, the tail reduce items, filling the CUs the last GEMM wave and the reduce tail leave
idle. Those CUs free over about one kv tile's length, so the last output tiles take one slice more
(shorter ones) than the first, and the CUs that free last take none.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.deep_ep.common import _asm, _uniform_i64, ld_lds, st_lds
from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.mega.bf16.gemm_bf16_kernel import (
    _make_shared_storage,
    gemm_bf16_tile,
)
from primus_turbo.flydsl.ulysses.common import (
    BLOCK_THREADS,
    SPIN_TIMEOUT,
    STATE_CLAIM,
    STATE_TILE,
    SYS_NT,
    VEC_BF16,
    VEC_I32,
    claim_unit,
    exit_and_bump_epoch,
    i64_base,
    opaque_base,
    opaque_tid,
)
from primus_turbo.flydsl.ulysses.kv_wgrad_kernel import (
    KV_WGRAD_LDS_CHUNK_STRIDE,
    kv_wgrad_splits,
    kv_wgrad_tail_splits,
    kv_wgrad_tile,
    kv_wgrad_tiles,
    q_wgrad_splits,
    q_wgrad_tile,
    q_wgrad_tiles,
    split_k_counters,
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
    make_bf16_fp16_tile_tensor,
    make_value_attrs,
)
from primus_turbo.flydsl.utils.prims import (
    _readfirstlane_i32,
    ceildiv,
    ld,
    memory_fence,
    read_clock,
    spin_timed_out,
    st,
)

# 16 B loads in flight per thread in a reduce unit (sources x vectors): remote reads are latency bound
_REDUCE_LOADS = 32


def _pe_unit(thread_index, tile_m, gk_res, kv_res, **c):
    """``KV[:, Ckv:]`` of m-tile ``tile_m``: the head sum of ``gk[..., dn:]``, in fp32 with one bf16
    rounding (what the identity block of ``B'`` computes, without a GEMM)."""
    R, BM, Ckv, dn, dr, hl = (c[k] for k in ("R", "BM", "Ckv", "dn", "dr", "hl"))
    vpr, head_vecs = dr // VEC_BF16, (dn + dr) // VEC_BF16
    row_vecs, out_row_vecs = hl * head_vecs, (Ckv + dr) // VEC_BF16
    num_vecs = BM * vpr
    for it in range_constexpr(ceildiv(num_vecs, BLOCK_THREADS)):
        v = fx.Int32(it * BLOCK_THREADS) + thread_index
        row = tile_m * fx.Int32(BM) + v // fx.Int32(vpr)
        j = v % fx.Int32(vpr)
        ok = (row < fx.Int32(R)) & (v < fx.Int32(num_vecs))
        src = row * fx.Int32(row_vecs) + fx.Int32(dn // VEC_BF16) + j
        parts = [
            buffer_load(
                gk_res,
                (src + fx.Int32(h * head_vecs)) * fx.Int32(VEC_I32),
                vec_width=VEC_I32,
                dtype=fx.T.i32,
                mask=ok,
            )
            for h in range_constexpr(hl)
        ]
        acc = Vec(parts[0]).bitcast(fx.BFloat16).to(fx.Float32)
        for h in range_constexpr(1, hl):
            acc = acc + Vec(parts[h]).bitcast(fx.BFloat16).to(fx.Float32)
        dst = (row * fx.Int32(out_row_vecs) + fx.Int32(Ckv // VEC_BF16) + j) * fx.Int32(VEC_I32)
        buffer_store(acc.to(fx.BFloat16).bitcast(fx.Int32), kv_res, dst, mask=ok, cache_modifier=SYS_NT)


def _nope_k_seg(hl, dn, dr):
    """The K walk of the kv units' k GEMM: every head's ``dn`` columns, skipping its ``dr`` ones
    (zero rows of ``B'``). None when a head's columns do not split into whole K chunks."""
    if dn % BLOCK_K or (dn + dr) % BLOCK_K:
        return None

    class _NopeKSeg:
        seg_len = dn // BLOCK_K
        num_chunks = hl * dn // BLOCK_K

        def chunk_base(self, seg):
            return seg * ((dn + dr) // BLOCK_K)

        def wait(self, seg):
            pass

    return _NopeKSeg()


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
        # realigned at the tile's end, so one barrier puts the flag after every wave's stores
        persistent="staggered",
    )
    tile_m = _readfirstlane_i32(buffer_load(order_res, g // fx.Int32(per_m), vec_width=1, dtype=fx.T.i32))
    tile_n = g % fx.Int32(per_m)
    q_c = make_bf16_fp16_tile_tensor(bank_addr, fx.Int64(0), R_pad * Cq)
    kv_c = make_bf16_fp16_tile_tensor(bank_addr, fx.Int64(R_pad * Cq * 2), R_pad * W2)
    if tile_n < fx.Int32(nq_tiles):
        gemm_tile(
            make_bf16_fp16_tile_tensor(opaque_base(GQ), fx.Int64(0), R * Nk),
            make_bf16_fp16_tile_tensor(opaque_base(WQT), fx.Int64(0), Cq * Nk),
            q_c,
            fx.Int32(R),
            fx.Int32(Cq),
            lds,
            tile_m,
            tile_n,
            K=Nk,
            n_tail=0,
            tid=opaque_tid(thread_index),
        )
    elif tile_n < fx.Int32(nq_tiles + nkv_tiles):
        tk = tile_n - fx.Int32(nq_tiles)
        gemm_tile(
            make_bf16_fp16_tile_tensor(opaque_base(GK), fx.Int64(0), R * Nk),
            make_bf16_fp16_tile_tensor(opaque_base(WKT), fx.Int64(0), W2 * Nk),
            kv_c,
            fx.Int32(R),
            fx.Int32(W2),
            lds,
            tile_m,
            tk,
            K=Nk,
            n_tail=0,
            tid=opaque_tid(thread_index),
            k_seg=_nope_k_seg(hl, dn, dr),
            k_ext=(
                make_bf16_fp16_tile_tensor(opaque_base(GV), fx.Int64(0), R * hl * dv),
                make_bf16_fp16_tile_tensor(opaque_base(WVT), fx.Int64(0), Ckv * hl * dv),
                hl * dv,
            ),
        )
    else:
        _pe_unit(
            opaque_tid(thread_index),
            tile_m,
            create_buffer_resource_from_addr(opaque_base(GK), num_records_bytes=R * Nk * 2),
            create_buffer_resource_from_addr(
                bank_addr + fx.Int64(R_pad * Cq * 2), num_records_bytes=R_pad * W2 * 2
            ),
            **c,
        )
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    # no release (an L2 writeback per unit): every staging store is system scope (SYS_NT), and the
    # waits above retired them before the flag goes out
    if thread_index == fx.Int32(0):
        st(flag_base, tile_m * fx.Int32(per_m) + tile_n, epoch, scope="sys", order="relaxed")


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


def _reduce_m_tiles(i, **c):
    """The (one or two) m-tiles the rows of reduce unit ``i`` fall in."""
    T, RB, BM, rank = (c[k] for k in ("T", "RB", "BM", "rank"))
    t0 = i * fx.Int32(RB)
    g0 = fx.Int32(rank * T) + t0
    g_last = arith.select(t0 + fx.Int32(RB) <= fx.Int32(T), g0 + fx.Int32(RB - 1), fx.Int32(rank * T + T - 1))
    return g0 // fx.Int32(BM), g_last // fx.Int32(BM)


def _peer_heaps(peers_base, num_ranks):
    """Every rank's heap base, pinned to SGPRs (buffer descriptors and the per-lane select below)."""
    return [_uniform_i64(ld(peers_base, fx.Int32(s), dtype=fx.T.i64)) for s in range(num_ranks)]


@ASTRewriter.transform
def _flags_pending(thread_index, i, heaps, epoch, claim_slot, **c):
    """Nonzero while some source has not raised the flags of reduce unit ``i``'s m-tiles. One flag per
    lane, so the ``n * 2 * per_m`` system-scope loads go out together (one round trip)."""
    num_ranks, per_m = c["num_ranks"], c["per_m"]
    per_src = 2 * per_m
    m_first, m_last = _reduce_m_tiles(i, **c)
    src, k = thread_index // fx.Int32(per_src), thread_index % fx.Int32(per_src)
    m = arith.select(k < fx.Int32(per_m), m_first, m_last)
    heap = heaps[0]
    for s in range_constexpr(1, num_ranks):
        heap = arith.select(src == fx.Int32(s), heaps[s], heap)
    if thread_index == fx.Int32(0):
        st_lds(claim_slot, 1, fx.Int32(0))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if thread_index < fx.Int32(num_ranks * per_src):
        flag_base = heap + fx.Int64(c["flag_region"] * FLAG_REGION_BYTES)
        if ld(flag_base, m * fx.Int32(per_m) + k % fx.Int32(per_m), scope="sys", dtype=fx.T.i64) < epoch:
            st_lds(claim_slot, 1, fx.Int32(1))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    pending = _readfirstlane_i32(ld_lds(claim_slot, 1))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    return pending


@ASTRewriter.transform
def _reduce_unit(thread_index, i, heaps, bank_offset, dlq_res, dlkv_res, epoch, claim_slot, **c):
    """Rows ``[i * RB, (i + 1) * RB)`` of this rank's slice: wait for every source's GEMM units over
    them, then sum the partials."""
    T, RB, R_pad, Cq, Ckv, dr, rank, num_ranks = (
        c[k] for k in ("T", "RB", "R_pad", "Cq", "Ckv", "dr", "rank", "num_ranks")
    )
    t0 = i * fx.Int32(RB)
    spin_start = read_clock()
    pending = _flags_pending(thread_index, i, heaps, epoch, claim_slot, **c)
    while pending != fx.Int32(0):
        fx.rocdl.s_sleep(fx.Int32(1))
        if spin_timed_out(spin_start, SPIN_TIMEOUT):
            if thread_index == fx.Int32(0):
                fx.printf(f"[ulysses rank={rank} gemm_rs] flag wait timeout: reduce unit={{}}\n", i)
            _asm("s_trap 2")
        pending = _flags_pending(thread_index, i, heaps, epoch, claim_slot, **c)
    # the loads below bypass the caches; this only keeps them below the flag wait
    memory_fence(order="acquire", scope="workgroup")
    W2 = Ckv + dr
    srcs = [
        create_buffer_resource_from_addr(heaps[s] + bank_offset, num_records_bytes=R_pad * (Cq + W2) * 2)
        for s in range_constexpr(num_ranks)
    ]
    _reduce_rows(thread_index, srcs, 0, dlq_res, Cq // VEC_BF16, t0, **c)
    _reduce_rows(thread_index, srcs, R_pad * Cq // 2, dlkv_res, W2 // VEC_BF16, t0, **c)


@ASTRewriter.transform
def _reduce_loop(
    thread_index, state_base, claim_slot, peers_base, bank_offset, DLQ, DLKV, epoch, trace_base, **c
):
    """Claim reduce units in row order until none are left. Trace entry ``num_gemm + i`` is reduce
    unit ``i``."""
    T, Cq, Ckv, dr = c["T"], c["Cq"], c["Ckv"], c["dr"]
    num_gemm, num_red = c["num_gemm"], c["num_red"]
    dlq_res = create_buffer_resource(DLQ, num_records_bytes=T * Cq * 2)
    dlkv_res = create_buffer_resource(DLKV, num_records_bytes=T * (Ckv + dr) * 2)
    heaps = _peer_heaps(peers_base, c["num_ranks"])
    r = claim_unit(thread_index, state_base, claim_slot, STATE_CLAIM)
    while r < fx.Int32(num_red):
        _trace(thread_index, trace_base, r + fx.Int32(num_gemm), 0)
        _reduce_unit(thread_index, r, heaps, bank_offset, dlq_res, dlkv_res, epoch, claim_slot, **c)
        _trace(thread_index, trace_base, r + fx.Int32(num_gemm), 1)
        r = claim_unit(thread_index, state_base, claim_slot, STATE_CLAIM)


@ASTRewriter.transform
def _trace(thread_index, trace_base, entry, end):
    """Debug timeline (``s_memrealtime``): ``[start, end]`` of each unit."""
    if const_expr(trace_base is not None):
        if const_expr(end):
            fx.rocdl.s_waitcnt(0)
        if thread_index == fx.Int32(0):
            st(trace_base, entry * fx.Int32(2) + fx.Int32(end), read_clock())


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
    num_lead,
    num_red_cu,
    num_tail_red,
    flag_region,
    bank_bytes,
    nt_vmcnt,
    trace,
    wgrad_s,
    wgrad_k_split,
    wgrad_at,
    wq_s,
    wq_k_split,
    wq_first,
    num_blocks,
    wgrad_s2=0,
    wgrad_k_split2=0,
    wgrad_t2=0,
):
    assert BN == 256 and Cq % BN == 0 and Ckv % BN == 0, "latent widths must fill whole column tiles"
    assert dr % VEC_BF16 == 0 and dn % VEC_BF16 == 0 and 0 < dr <= 128
    assert (hl * (dn + dr)) % BLOCK_K == 0 and (hl * dv) % BLOCK_K == 0
    R = num_ranks * T
    m_tiles = ceildiv(R, BM)
    R_pad = m_tiles * BM
    nq_tiles, nkv_tiles = Cq // BN, Ckv // BN
    per_m = nq_tiles + nkv_tiles + 1
    assert RB <= BM, "a reduce unit spans at most two m-tiles"
    assert m_tiles * per_m <= FLAGS_PER_REGION, "too many GEMM units for one flag region"
    assert num_ranks * 2 * per_m <= BLOCK_THREADS, "one flag per lane in a reduce unit's poll"
    assert R_pad * (Cq + Ckv + dr) * 2 <= bank_bytes and R_pad * (Cq + Ckv + dr) < 2**31
    assert R * hl * (dn + dr) < 2**31, "i32 element offsets overflow"
    num_gemm, num_red = m_tiles * per_m, ceildiv(T, RB)
    num_lead = min(num_lead, num_gemm)
    assert num_red_cu + num_tail_red > 0
    Nq = hl * (dn + dr)
    kv_out_tiles = kv_wgrad_tiles(hl, Ckv, 1)
    assert 0 <= wgrad_t2 <= kv_out_tiles and (not wgrad_t2 or wgrad_s2)
    num_w2 = wgrad_t2 * wgrad_s2
    num_w = (kv_out_tiles - wgrad_t2) * wgrad_s if wgrad_s else 0
    num_wq = q_wgrad_tiles(Nq, Cq, wq_s) if wq_s else 0
    wgrad_at = min(max(wgrad_at, num_lead), num_gemm)
    # dispatch order: [q wgrad] | lead GEMM | reduce | GEMM up to wgrad_at | wgrad | rest of GEMM | tail
    # reduce, the wgrad behind the tail reduce at wgrad_at = num_gemm. The tail reduce blocks follow
    # every GEMM block, so none of them holds a CU a GEMM block they wait on still needs
    w_last = wgrad_at == num_gemm
    tail = [("gemm", num_gemm - wgrad_at), ("red", num_tail_red)]
    wgrad = [("w", num_w), ("w2", num_w2)] if wq_first else [("wq", num_wq), ("w", num_w), ("w2", num_w2)]
    segs = [("wq", num_wq)] if wq_first else []
    segs += [("gemm", num_lead), ("red", num_red_cu), ("gemm", wgrad_at - num_lead)]
    segs += tail + wgrad if w_last else wgrad + tail
    seg_seq = []
    start, gemm_base = 0, 0
    for kind, count in segs:
        if count:
            seg_seq.append((kind, start, start + count, gemm_base if kind == "gemm" else 0))
            gemm_base += count if kind == "gemm" else 0
            start += count
    grid = start
    assert grid == num_gemm + num_red_cu + num_tail_red + num_w + num_w2 + num_wq
    num_blocks = min(num_blocks, grid)
    SharedStorage = _make_shared_storage(
        BM,
        BN,
        chunk_stride=KV_WGRAD_LDS_CHUNK_STRIDE if num_w + num_w2 + num_wq else 1024,
        num_lds_row_idx_entries=3,
    )
    wgrad_dims = dict(R=R, hl=hl, dn=dn, dr=dr, dv=dv, Ckv=Ckv, S=wgrad_s, k_split=wgrad_k_split)
    wgrad_dims2 = dict(
        R=R,
        hl=hl,
        dn=dn,
        dr=dr,
        dv=dv,
        Ckv=Ckv,
        S=wgrad_s2,
        k_split=wgrad_k_split2,
        tile0=kv_out_tiles - wgrad_t2,
    )
    wq_dims = dict(R=R, Nq=Nq, Cq=Cq, S=wq_s, k_split=wq_k_split)
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
        num_gemm=num_gemm,
        num_red=num_red,
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
        ORDER: fx.Tensor,
        PEER_PTRS: fx.Tensor,
        STATE: fx.Tensor,
        TRACE: fx.Tensor,
        LC: fx.Tensor,
        PART: fx.Tensor,
        WKV: fx.Tensor,
        CNT: fx.Tensor,
        LQ: fx.Tensor,
        PARTQ: fx.Tensor,
        DWQ: fx.Tensor,
        CNTQ: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        claim_slot = fx.Int32(fx.ptrtoint(lds.lds_row_idx.ptr))
        state_base = i64_base(STATE)
        peers_base = i64_base(PEER_PTRS)
        # uniform in value, pinned to SGPRs: the descriptors built from them skip a waterfall
        epoch = _uniform_i64(ld(state_base, fx.Int32(0), dtype=fx.T.i64))
        bank_offset = fx.Int64(FLAG_BYTES) + (epoch & fx.Int64(1)) * fx.Int64(bank_bytes)
        my_base = _uniform_i64(ld(peers_base, fx.Int32(rank), dtype=fx.T.i64))
        trace_base = i64_base(TRACE) if trace else None
        # one resident block per CU walks the dispatch order through a counter: the hardware deals
        # blocks to the shader engines round-robin, so a launched grid stalls on the busiest one
        # its own LDS slot: a reduce item's first unit claim follows this one with no store between
        # that LLVM sees, and on a shared slot it reads this claim back
        item_slot = claim_slot + fx.Int32(8)
        item = claim_unit(thread_index, state_base, item_slot, STATE_TILE)
        # claims only grow, so a block walks the segments in order: one loop per segment keeps each
        # body's loop-invariant setup out of the others' register budget (one loop over all spills)
        for si in range_constexpr(len(seg_seq)):
            kind, lo, hi, base = seg_seq[si]
            while item < fx.Int32(hi):
                if const_expr(kind == "red"):
                    _reduce_loop(
                        thread_index,
                        state_base,
                        claim_slot,
                        peers_base,
                        bank_offset,
                        DLQ,
                        DLKV,
                        epoch,
                        trace_base,
                        **consts,
                    )
                if const_expr(kind == "w"):
                    w = item - fx.Int32(lo)
                    _trace(thread_index, trace_base, w + fx.Int32(num_gemm + num_red + num_wq), 0)
                    kv_wgrad_tile(w, GK, GV, LC, PART, WKV, CNT, lds, persistent="staggered", **wgrad_dims)
                    _trace(thread_index, trace_base, w + fx.Int32(num_gemm + num_red + num_wq), 1)
                if const_expr(kind == "w2"):
                    w = item - fx.Int32(lo)
                    _trace(thread_index, trace_base, w + fx.Int32(num_gemm + num_red + num_wq + num_w), 0)
                    kv_wgrad_tile(w, GK, GV, LC, PART, WKV, CNT, lds, persistent="staggered", **wgrad_dims2)
                    _trace(thread_index, trace_base, w + fx.Int32(num_gemm + num_red + num_wq + num_w), 1)
                if const_expr(kind == "wq"):
                    wq = item - fx.Int32(lo)
                    _trace(thread_index, trace_base, wq + fx.Int32(num_gemm + num_red), 0)
                    q_wgrad_tile(wq, GQ, LQ, PARTQ, DWQ, CNTQ, lds, persistent="staggered", **wq_dims)
                    _trace(thread_index, trace_base, wq + fx.Int32(num_gemm + num_red), 1)
                if const_expr(kind == "gemm"):
                    g = item - fx.Int32(lo - base)
                    _trace(thread_index, trace_base, g, 0)
                    _gemm_unit(
                        thread_index,
                        g,
                        create_buffer_resource(ORDER, num_records_bytes=m_tiles * 4),
                        my_base + fx.Int64(flag_region * FLAG_REGION_BYTES),
                        my_base + bank_offset,
                        GQ,
                        GK,
                        GV,
                        WQT,
                        WKT,
                        WVT,
                        lds,
                        epoch,
                        **consts,
                    )
                    _trace(thread_index, trace_base, g, 1)
                item = claim_unit(thread_index, state_base, item_slot, STATE_TILE)
        # once per block: an L2 writeback / invalidate here hits the GEMMs still running
        exit_and_bump_epoch(thread_index, state_base, epoch, num_blocks, order="relaxed")

    return kernel, num_blocks


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
    ORDER,
    PEER_PTRS,
    STATE,
    TRACE,
    LC,
    PART,
    WKV,
    CNT,
    LQ,
    PARTQ,
    DWQ,
    CNTQ,
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
    num_lead: fx.Constexpr[int],
    num_red_cu: fx.Constexpr[int],
    num_tail_red: fx.Constexpr[int],
    flag_region: fx.Constexpr[int],
    bank_bytes: fx.Constexpr[int],
    nt_vmcnt: fx.Constexpr[int],
    trace: fx.Constexpr[bool],
    wgrad_s: fx.Constexpr[int],
    wgrad_k_split: fx.Constexpr[int],
    wgrad_at: fx.Constexpr[int],
    wq_s: fx.Constexpr[int],
    wq_k_split: fx.Constexpr[int],
    wq_first: fx.Constexpr[bool],
    num_blocks: fx.Constexpr[int],
    wgrad_s2: fx.Constexpr[int],
    wgrad_k_split2: fx.Constexpr[int],
    wgrad_t2: fx.Constexpr[int],
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
        num_lead,
        num_red_cu,
        num_tail_red,
        flag_region,
        bank_bytes,
        nt_vmcnt,
        trace,
        wgrad_s,
        wgrad_k_split,
        wgrad_at,
        wq_s,
        wq_k_split,
        wq_first,
        num_blocks,
        wgrad_s2,
        wgrad_k_split2,
        wgrad_t2,
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
        ORDER,
        PEER_PTRS,
        STATE,
        TRACE,
        LC,
        PART,
        WKV,
        CNT,
        LQ,
        PARTQ,
        DWQ,
        CNTQ,
        value_attrs=make_value_attrs(2, 0, "512,512"),
    ).launch(grid=(grid, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)


def gemm_rs_bank_bytes(R, Cq, Ckv, dr, BM=256):
    return ceildiv(R, BM) * BM * (Cq + Ckv + dr) * 2


@functools.lru_cache(maxsize=64)
def _m_tile_order(T, num_ranks, BM, device):
    """The m-tiles by the last row of any slice they cover, so every rank's early rows finish first
    (the same order on every rank)."""
    R = num_ranks * T
    keys = []
    for m in range(ceildiv(R, BM)):
        last = min((m + 1) * BM, R) - 1
        src_first, src_last = (m * BM) // T, last // T
        land = max(
            (last if s == src_last else (s + 1) * T - 1) - s * T for s in range(src_first, src_last + 1)
        )
        keys.append((land, src_first, m))
    return torch.tensor([m for *_, m in sorted(keys)], dtype=torch.int32, device=device)


def _default_rb(num_ranks, BM):
    """Reduce-unit rows: ``RB * n`` source rows per unit keeps a unit's loads (and its batching
    over the lanes) the same for every group size; n = 8 measured best at 32, n = 4 at 64."""
    return min(BM, max(16, 256 // num_ranks))


def gemm_rs_trace_len(T, num_ranks, Cq, Ckv, BM=256, BN=256, RB=None, num_wgrad_tiles=0):
    """Entries of the debug timeline: 2 per unit (GEMM units, reduce units, then the q and kv
    weight-gradient tiles, ``num_wgrad_tiles`` of them)."""
    RB = _default_rb(num_ranks, BM) if RB is None else RB
    return 2 * (ceildiv(num_ranks * T, BM) * (Cq // BN + Ckv // BN + 1) + ceildiv(T, RB) + num_wgrad_tiles)


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
    RB=None,
    num_lead=None,
    num_red_cu=16,
    num_tail_red=16,
    trace=None,
    kv_wgrad=None,
    wgrad_s=None,
    wgrad_at=None,
    q_wgrad=None,
    wq_rows=None,
    wq_first=True,
):
    """``gq`` / ``gk`` [n * T, hl * (dn + dr)] and ``gv`` [n * T, hl * dv] this rank's head gradients
    over the gathered sequence (sbhd order); ``wq_t`` [Cq, hl * (dn + dr)] (``Wq`` rows of the local
    heads, transposed), ``wk_t`` [Ckv + dr, hl * (dn + dr)] (``B'`` transposed) and ``wv_t`` [Ckv, hl *
    dv]. Writes this rank's rows of the summed latent gradients, ``d_q_latent`` [T, Cq] and
    ``d_kv_latent`` [T, Ckv + dr]. ``num_lead`` GEMM blocks (default: one per CU) go ahead of the
    ``num_red_cu`` reduce blocks, ``num_tail_red`` more follow the last GEMM block. ``trace`` (debug):
    an int64 tensor of ``gemm_rs_trace_len(...)`` entries receiving the ``s_memrealtime`` timeline.
    ``kv_wgrad=(lc, d_w_kv)`` (``lc`` [n * T, Ckv], ``d_w_kv`` [hl, dn + dv, Ckv]) also writes the kv
    weight gradient, on ``wgrad_s`` K slices dispatched after GEMM unit ``wgrad_at`` (default: behind
    the tail reduce, on ``kv_wgrad_tail_splits``: the last output tiles on shorter slices). ``q_wgrad=(lq, d_w_q)`` (``lq`` [n * T, Cq],
    ``d_w_q`` [hl * (dn + dr), Cq]) adds the q weight gradient, on K slices of about ``wq_rows`` rows
    (default: ``q_wgrad_splits``), dispatched ahead of everything (``wq_first``; split-K needs it) or
    just ahead of the kv one."""
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
    if RB is None:
        RB = _default_rb(num_ranks, BM)
    num_cus = torch.cuda.get_device_properties(gq.device).multi_processor_count
    if num_lead is None:
        # reduce blocks dispatch once the first GEMM wave retires: nothing is ready to sum before that
        num_lead = num_cus
    num_gemm = ceildiv(R, BM) * (Cq // BN + Ckv // BN + 1)
    wgrad_s2 = wgrad_k_split2 = wgrad_t2 = 0
    if kv_wgrad is None:
        lc = part = d_w_kv = cnt = workspace.state
        wgrad_s = wgrad_k_split = 0
    else:
        lc, d_w_kv = kv_wgrad
        assert tuple(lc.shape) == (R, Ckv) and lc.is_contiguous()
        assert tuple(d_w_kv.shape) == (hl, dn + dv, Ckv) and d_w_kv.is_contiguous()
        if wgrad_s is None and wgrad_at in (None, num_gemm):
            (wgrad_s, wgrad_k_split), (wgrad_s2, wgrad_k_split2, wgrad_t2) = kv_wgrad_tail_splits(
                R, hl, Ckv, num_cus
            )
        else:
            wgrad_s, wgrad_k_split = kv_wgrad_splits(R, hl, Ckv, num_cus, wgrad_s)
        part = d_w_kv
        if max(wgrad_s, wgrad_s2) > 1:
            part = d_w_kv.new_empty((max(wgrad_s, wgrad_s2),) + tuple(d_w_kv.shape), dtype=torch.float32)
        cnt = split_k_counters(gq.device, 2 * kv_wgrad_tiles(hl, Ckv, 1))
        num_w = (kv_wgrad_tiles(hl, Ckv, 1) - wgrad_t2) * wgrad_s + wgrad_t2 * wgrad_s2
        # the split-K slices wait for each other: the last blocks of the grid, at most one per CU
        assert max(wgrad_s, wgrad_s2) == 1 or (num_w <= num_cus and wgrad_at in (None, num_gemm)), (
            "split-K kv wgrad tiles must all be co-resident: behind the tail reduce, one per CU"
        )
    if q_wgrad is None:
        lq = partq = d_w_q = cntq = workspace.state
        wq_s = wq_k_split = 0
    else:
        lq, d_w_q = q_wgrad
        assert tuple(lq.shape) == (R, Cq) and lq.is_contiguous()
        assert tuple(d_w_q.shape) == (Nk, Cq) and d_w_q.is_contiguous()
        wq_s, wq_k_split = q_wgrad_splits(R, Nk, Cq, num_cus, wq_rows)
        partq = d_w_q
        if wq_s > 1:
            partq = d_w_q.new_empty((wq_s,) + tuple(d_w_q.shape), dtype=torch.float32)
        cntq = split_k_counters(gq.device, 2 * q_wgrad_tiles(Nk, Cq, 1), name="q")
        # the split-K slices wait for each other: the first blocks of the grid, at most one per CU
        assert wq_s == 1 or (wq_first and q_wgrad_tiles(Nk, Cq, wq_s) <= num_cus), (
            "split-K q wgrad tiles must all be co-resident: ahead of everything, one per CU"
        )
    if wgrad_at is None:
        wgrad_at = num_gemm
    order = _m_tile_order(T, num_ranks, BM, gq.device)
    run_launcher(
        _launch,
        *tensors,
        order,
        workspace.peer_ptrs,
        workspace.state,
        workspace.state if trace is None else trace,
        lc,
        part,
        d_w_kv,
        cnt,
        lq,
        partq,
        d_w_q,
        cntq,
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
        num_lead,
        num_red_cu,
        num_tail_red,
        flag_region,
        workspace.bank_bytes,
        3,
        trace is not None,
        wgrad_s,
        wgrad_k_split,
        min(wgrad_at, num_gemm),
        wq_s,
        wq_k_split,
        wq_first and wq_s > 0,
        num_cus,
        wgrad_s2,
        wgrad_k_split2,
        wgrad_t2,
        torch.cuda.current_stream(),
    )
