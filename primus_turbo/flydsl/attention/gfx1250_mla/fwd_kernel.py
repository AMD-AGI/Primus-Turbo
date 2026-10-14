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
# Adapted from the gfx1250 FMHA forward prefill of aiter (https://github.com/ROCm/aiter, commit 6963ae9d),
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc., under the MIT license.

"""gfx1250 FlyDSL flash-attention forward kernel at the DeepSeek-V3 MLA head dims.

A workgroup of 8 wave32 waves owns 256 query rows of one (batch, head) and streams 64-row K/V
tiles through two LDS slots. Both GEMMs run on ``v_wmma_f32_16x16x32_bf16``; the online softmax
works in fp32 with ``p = exp2(s * scale * log2(e) - m)`` and lets the running max lag by up to
one natural-log unit before it rescales O.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

# UNSTABLE(gfx1250): the rocdl / tdm_ops calls (WMMA, permlanex16, ballot, mbcnt_lo, wave_id,
# sched_barrier, the split workgroup barrier, s_wait_dscnt, tensor_wait) have no stable API in
# FlyDSL 0.3.4.1; arith is stable.
from flydsl.expr import arith, rocdl
from flydsl.expr.rocdl import tdm_ops

from primus_turbo.flydsl.utils.buffer_ops import buffer_store, create_buffer_resource
from primus_turbo.flydsl.utils.prims import LOG2E

from . import SOURCE_DIGEST
from .fwd_staging import (
    BLOCK_M,
    D_QK,
    D_V,
    K_LDS_BYTES,
    K_ROW_ELEMS,
    N_BLOCK,
    NUM_WAVES,
    O_LDS_BYTES,
    Q_LDS_BYTES,
    Q_TILES,
    ROWS_PER_WAVE,
    V_LDS_BYTES,
    V_ROW_ELEMS,
    WAVE_SIZE,
    WMMA_K,
    WMMA_M,
    OutputWriter,
    issue_q_copy,
    k_read_base,
    kv_copies,
    load_k,
    load_v,
    read_q_fragments,
    v_read_base,
)

BLOCK_SIZE = WAVE_SIZE * NUM_WAVES
WMMA_N = 16
D_TILES = D_V // WMMA_M  # 16-column O accumulators per q tile
# LDS: two K/V slots, K and V each at the start of a 64 KB segment. Q is staged in slot 1 before
# the loop, O in the slot the last tile does not use.
KV_BLOCK_BYTES = 64 * 1024
SLOT_BYTES = 2 * KV_BLOCK_BYTES
LDS_BYTES = 320 * 1024  # a gfx1250 workgroup's whole LDS, as the source kernel takes it; the slots use 256 KB
assert max(K_LDS_BYTES, V_LDS_BYTES) <= KV_BLOCK_BYTES and max(Q_LDS_BYTES, O_LDS_BYTES) <= SLOT_BYTES
# Head-grouped dispatch: KV_GROUP q blocks of KV_HEAD_CHUNK heads run together (powers of two).
KV_GROUP = 16
KV_HEAD_CHUNK = 16
# The running max may lag the row max by this many natural-log units before O is rescaled.
RESCALE_THRESHOLD = 1.0
# Finite running-max seed: a fully masked row keeps a finite m, so no -inf - -inf.
BIG_NEG = -1.0e30

_FAST = arith.FastMathFlags.fast
# Fast math without reassoc, so LLVM keeps the sum trees instead of re-linearizing them.
_NO_REASSOC = (
    arith.FastMathFlags.nnan
    | arith.FastMathFlags.ninf
    | arith.FastMathFlags.nsz
    | arith.FastMathFlags.arcp
    | arith.FastMathFlags.contract
    | arith.FastMathFlags.afn
)


def _drain_and_arrive():
    """Retire this wave's LDS reads and TDM copies, then arrive at the workgroup barrier."""
    rocdl.sched_barrier(0)
    rocdl.s_wait_dscnt(0)
    tdm_ops.tensor_wait(0)
    # The raw split barrier has no memory semantics: fence it the way gpu.barrier() does.
    fx.llvm.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Release)
    rocdl.s_barrier_signal(-1)
    rocdl.sched_barrier(0)


def _barrier_wait():
    rocdl.sched_barrier(0)
    rocdl.s_barrier_wait(-1)
    fx.llvm.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Acquire)
    rocdl.sched_barrier(0)


def _block_id(axis):
    """Block id along ``axis`` after the longest-first remap of the dispatch order.

    Causal work grows with the q block index, so the hardware's linear order is remapped to start
    the last q block of every (head, batch) first. When the grid divides evenly, windows of
    KV_GROUP ranks go out KV_HEAD_CHUNK heads at a time so a head's K/V is shared in L2.
    """
    gx, gy, gz = fx.grid_dim.x, fx.grid_dim.y, fx.grid_dim.z
    lin = fx.block_idx.x + gx * (fx.block_idx.y + gy * fx.block_idx.z)
    gyz = gy * gz
    rank = lin // gyz
    rem = lin - rank * gyz
    lg, lhb = KV_GROUP.bit_length() - 1, KV_HEAD_CHUNK.bit_length() - 1
    r = (rank & fx.Int32(KV_GROUP - 1)) * gyz + rem
    c = r & fx.Int32(KV_GROUP * KV_HEAD_CHUNK - 1)
    grouped = ((gx & fx.Int32(KV_GROUP - 1)) | (gyz & fx.Int32(KV_HEAD_CHUNK - 1))) == fx.Int32(0)
    lin = grouped.select(
        ((rank & fx.Int32(-KV_GROUP)) + (c >> fx.Int32(lhb))) * gyz
        + ((r >> fx.Int32(lg + lhb)) << fx.Int32(lhb))
        + (c & fx.Int32(KV_HEAD_CHUNK - 1)),
        lin,
    )
    rank = lin // gyz
    rem = lin - rank * gyz
    if axis == "x":
        return gx - fx.Int32(1) - rank
    if axis == "y":
        return rem % gy
    return rem // gy


def _wmma(a, b, c):
    """C[16x16 f32] += A[16x32 bf16] @ B[32x16 bf16]."""
    v8f32 = fx.Vector.make_type(8, fx.Float32)
    return rocdl.wmma_f32_16x16x32_bf16(
        v8f32, fx.as_ir_value(a), fx.as_ir_value(b), fx.as_ir_value(c), reuseA=False, reuseB=False
    ).result


def _qk_gemm(k_vals, q_frags):
    """S^T = K Q^T of one tile: [q tile][kv tile] v8 f32 (kv on the rows, q on the columns).

    Runs k-step major in the order ``load_k`` issued the K reads, fenced so every LDS wait
    releases at most one k-step; each accumulator still sums its k-steps in order.
    """
    nkv, nks = N_BLOCK // WMMA_N, D_QK // WMMA_K
    zero = fx.Vector.filled(8, 0.0, fx.Float32)
    acc = [[zero] * nkv for _ in range(Q_TILES)]
    for ks in range(nks):
        if ks > 0:
            rocdl.sched_barrier(0)
        for kv in range(nkv):
            if kv > 0:
                rocdl.sched_barrier(0)
            j = (kv * nks + ks) * 2
            k_frag = k_vals[j].shuffle(k_vals[j + 1], list(range(16)))
            for qt in range(Q_TILES):
                acc[qt][kv] = _wmma(k_frag, q_frags[qt][ks], acc[qt][kv])
    return acc


def _pv_gemm(v_vals, p, o_acc):
    """O^T += V^T P^T of one tile for both q tiles: [q tile][d tile] v8 f32."""
    nkt = N_BLOCK // WMMA_K
    rocdl.s_wait_dscnt(0)
    out = [[None] * D_TILES for _ in range(Q_TILES)]
    for dt in range(D_TILES):
        accs = [o_acc[qt][dt] for qt in range(Q_TILES)]
        j = dt * nkt * 2
        for kt in range(nkt):
            v_frag = v_vals[j].shuffle(v_vals[j + 1], list(range(16)))
            j += 2
            for qt in range(Q_TILES):
                p_frag = p[qt][2 * kt].shuffle(p[qt][2 * kt + 1], list(range(16)))
                accs[qt] = _wmma(v_frag, p_frag, accs[qt])
        for qt in range(Q_TILES):
            out[qt][dt] = accs[qt]
    return out


def _tree_reduce(lists, op3, op2):
    """Balanced 3-ary reduction of each list, emitted position-major across the lists so the
    independent rows interleave."""
    curs = [list(v) for v in lists]
    while max(len(c) for c in curs) > 1:
        nxts = [[] for _ in curs]
        idxs = [0] * len(curs)
        while any(idxs[k] < len(curs[k]) for k in range(len(curs))):
            for k in range(len(curs)):
                cur, i, n = curs[k], idxs[k], len(curs[k])
                if i >= n:
                    continue
                if n - i >= 3:
                    nxts[k].append(op3(cur[i], cur[i + 1], cur[i + 2]))
                    idxs[k] += 3
                elif n - i == 2:
                    nxts[k].append(op2(cur[i], cur[i + 1]))
                    idxs[k] += 2
                else:
                    nxts[k].append(cur[i])
                    idxs[k] += 1
        curs = nxts
    return [c[0] for c in curs]


def _fmax(a, b):
    return arith.maxnumf(a, b, fastmath=_FAST)


def _fsub(a, b):
    with arith.fastmath(_FAST):
        return a - b


def _fmul(a, b):
    with arith.fastmath(_FAST):
        return a * b


def _fadd(a, b):
    with arith.fastmath(_NO_REASSOC):
        return a + b


def _exp2(x):
    return fx.math.exp2(x, fastmath=arith.FastMathFlags.afn)


def _softmax(*, s, m_prev, d_prev, lane_idx, c_exp2, kv_pos_base, kv_last):
    """Online-softmax update of both q tiles for one kv tile.

    ``s`` is the unscaled S^T: lane l holds query l % 16 and kv rows (l // 16) * 8 + [0, 8) of each
    16-row kv tile, lane l ^ 16 the other eight. ``kv_last`` is the last key each q tile's row may
    see (None on unmasked tiles). m is carried in log2 units of the scaled score. Returns the bf16
    probabilities, the new m and d, the O correction and the wave-uniform rescale flag.
    """
    nkv = N_BLOCK // WMMA_N
    neg_inf = fx.Float32(float("-inf"))
    zero = fx.Float32(0.0)
    sel_lo, sel_hi = fx.as_ir_value(fx.Int32(0x76543210)), fx.as_ir_value(fx.Int32(0xFEDCBA98))

    def peer(x):  # the value held by lane l ^ 16; fx.gpu.shuffle_xor changes the generated code
        x = fx.as_ir_value(x)
        return fx.Float32(
            rocdl.permlanex16(fx.Float32.ir_type, x, x, sel_lo, sel_hi, fi=False, bound_control=False)
        )

    khalf = lane_idx // fx.Int32(WMMA_M)
    s_masked = []
    for qt in range(Q_TILES):
        row = []
        for kvt in range(nkv):
            svec = fx.Vector(s[qt][kvt])
            for i in range(8):
                sval = fx.Float32(svec[i])
                if kv_last is not None:
                    kv_pos = kv_pos_base + khalf * fx.Int32(8) + fx.Int32(kvt * WMMA_N + i)
                    sval = (kv_pos > kv_last[qt]).select(neg_inf, sval)
                row.append(sval)
        s_masked.append(row)

    local_max = _tree_reduce(s_masked, lambda a, b, c: _fmax(_fmax(a, b), c), _fmax)
    m_new, corr, neg_m, rescale = [], [], [], []
    for qt in range(Q_TILES):
        row_max = _fmul(_fmax(local_max[qt], peer(local_max[qt])), c_exp2)
        m_full = _fmax(m_prev[qt], row_max)
        # Wave-uniform: rescale O once any row's max grew by more than the threshold.
        need = _fsub(row_max, m_prev[qt]) > fx.Float32(RESCALE_THRESHOLD * LOG2E)
        do_rescale = fx.Int32(rocdl.ballot(fx.Int32.ir_type, need)) != fx.Int32(0)
        m = do_rescale.select(m_full, m_prev[qt])
        corr.append(_exp2(_fsub(m_prev[qt], m)))
        neg_m.append(_fsub(zero, m))
        m_new.append(m)
        rescale.append(do_rescale)

    # p = exp2(s * c - m), two elements per packed FMA.
    p, p_flat = [], []
    c2 = fx.Vector.from_elements([c_exp2], fx.Float32).broadcast_to(2)
    for qt in range(Q_TILES):
        tiles, flat, idx = [], [], 0
        n2 = fx.Vector.from_elements([neg_m[qt]], fx.Float32).broadcast_to(2)
        for _ in range(nkv):
            pe = []
            for _ in range(4):
                sv = fx.Vector.from_elements([s_masked[qt][idx], s_masked[qt][idx + 1]], fx.Float32)
                av = fx.math.fma(sv, c2, n2)
                for e in range(2):
                    pj = _exp2(fx.Float32(av[e]))
                    pe.append(pj)
                    flat.append(pj)
                idx += 2
            tiles.append(fx.Vector.from_elements(pe, fx.Float32).to(fx.BFloat16))
        p.append(tiles)
        p_flat.append(flat)

    # Both rows' sum trees run as one tree of (row 0, row 1) pairs.
    leaves = [fx.Vector.from_elements([a, b], fx.Float32) for a, b in zip(p_flat[0], p_flat[1])]
    (tot,) = _tree_reduce([leaves], lambda a, b, c: _fadd(_fadd(a, b), c), _fadd)
    local_sum = [fx.Float32(tot[0]), fx.Float32(tot[1])]
    d_new = [
        _fadd(_fmul(corr[qt], d_prev[qt]), _fadd(local_sum[qt], peer(local_sum[qt]))) for qt in range(Q_TILES)
    ]
    return p, m_new, d_new, corr, rescale


@flyc.jit
def _copy_if(cond, copies):
    if cond:
        for c in copies:
            fx.copy_atom_call(*c)


@flyc.jit
def _rescale_if(o_vecs, corr_vec, do_rescale):
    result = o_vecs
    if do_rescale:
        result = [ov * corr_vec for ov in o_vecs]
    return result


def _attention(
    *,
    o,
    q,
    k,
    v,
    lse,
    softmax_scale,
    stride_q_seq,
    stride_k_seq,
    stride_v_seq,
    stride_o_seq,
    stride_q_head,
    stride_k_head,
    stride_v_head,
    stride_o_head,
    stride_lse_seq,
    stride_lse_head,
    lse_base,
    lse_bytes,
    q_start,
    q_len,
    kv_start,
    kv_len,
    window_right,
    warp_idx,
    hi_wave,
    lds_base,
):
    """One wave's share of the workgroup: 32 query rows of one (batch, head)."""
    # Lane id within the wave32: mbcnt of all lanes below this one (fx.gpu.lane_id changes the code).
    lane_idx = fx.Int32(rocdl.mbcnt_lo(fx.Int32.ir_type, fx.Int32(-1).ir_value(), fx.Int32(0).ir_value()))
    head = _block_id("y")
    row0 = _block_id("x") * BLOCK_M + warp_idx * ROWS_PER_WAVE
    seq_idx = [row0 + qt * WMMA_M + lane_idx % WMMA_M for qt in range(Q_TILES)]

    def k_slot(pp):
        return lds_base + fx.Int32(pp) * fx.Int32(SLOT_BYTES)

    def v_slot(pp):
        return k_slot(pp) + fx.Int32(KV_BLOCK_BYTES)

    q_lds = lds_base + fx.Int32(SLOT_BYTES)
    q_region = issue_q_copy(
        q=q,
        stride_q_seq=stride_q_seq,
        stride_q_head=stride_q_head,
        q_start=q_start,
        q_len=q_len,
        head=head,
        block_x=_block_id("x"),
        warp_idx=warp_idx,
        lds_base=q_lds,
    )

    # Tiles [0, n_tiles) hold every key the workgroup's last query attends.
    block_x = _block_id("x")
    causal_off = kv_len - q_len
    wg_max_seq = fx.min(block_x * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1), q_len - fx.Int32(1))
    kv_len_wg = fx.max(fx.min(wg_max_seq + causal_off + window_right + fx.Int32(1), kv_len), fx.Int32(1))
    n_tiles = fx.ceildiv(kv_len_wg, fx.Int32(N_BLOCK))

    def kv_valid(row):
        return fx.min(fx.max(kv_len_wg - row, fx.Int32(0)), fx.Int32(N_BLOCK))

    def tile_copies(pp, row, valid):
        """K then V copies of the tile at kv row ``row`` into slot ``pp``."""
        kc = kv_copies(
            lds_base=k_slot(pp),
            src=k,
            stride_seq=stride_k_seq,
            stride_head=stride_k_head,
            head=head,
            row0=kv_start + row,
            valid=valid,
            width=D_QK,
            row_elems=K_ROW_ELEMS,
        )
        vc = kv_copies(
            lds_base=v_slot(pp),
            src=v,
            stride_seq=stride_v_seq,
            stride_head=stride_v_head,
            head=head,
            row0=kv_start + row,
            valid=valid,
            width=D_V,
            row_elems=V_ROW_ELEMS,
        )
        return kc + vc

    first = tile_copies(0, fx.Int32(0), kv_valid(fx.Int32(0)))
    q_frags = read_q_fragments(q_region, lane_idx)
    for c in first:
        fx.copy_atom_call(*c)
    # Split workgroup barrier, one arrive and one wait per tile: a wave arrives once its reads of
    # the current slot and its share of the next tile's copy are done and waits before the next
    # tile, so it may run one PV GEMM ahead of the slowest wave.
    _drain_and_arrive()

    c_exp2 = softmax_scale * fx.Float32(LOG2E)
    init = []
    for _ in range(Q_TILES):
        init += [fx.Float32(BIG_NEG), fx.Float32(0.0)]
        init += [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(D_TILES)]
    nst = 2 + D_TILES  # (m, d, O accumulators) per q tile
    ptr_base = len(init)
    k0, v0 = k_read_base(k_slot(0), lane_idx), v_read_base(v_slot(0), lane_idx)
    k1, v1 = k_read_base(k_slot(1), lane_idx), v_read_base(v_slot(1), lane_idx)
    init += [k0, k1, v0, v1]

    def tile(t, state, *, masked, sync_only=False):
        """One kv tile: wait for its K/V, prefetch the next one, then QK, softmax, rescale, PV.

        ``masked`` tiles apply the causal edge and the kv tail. ``sync_only`` tiles are fully masked
        for every row of this wave and keep only the barrier and the prefetch.
        """
        pb = 0 if sync_only else ptr_base
        nxt_pp = (t + fx.Int32(1)) % fx.Int32(2)
        kv_tile_start = t * fx.Int32(N_BLOCK)
        if not sync_only:
            m_prev = [fx.Float32(state[qt * nst]) for qt in range(Q_TILES)]
            d_prev = [fx.Float32(state[qt * nst + 1]) for qt in range(Q_TILES)]
            o_acc = [[fx.Vector(state[qt * nst + 2 + dt]) for dt in range(D_TILES)] for qt in range(Q_TILES)]
        k_cur, k_nxt, v_cur, v_nxt = state[pb : pb + 4]
        nxt = t + fx.Int32(1)
        nxt_row = nxt * fx.Int32(N_BLOCK)
        nxt_copies = tile_copies(nxt_pp, nxt_row, kv_valid(nxt_row))
        swapped = [k_nxt, k_cur, v_nxt, v_cur]
        if sync_only:
            _barrier_wait()
            _copy_if(nxt < n_tiles, nxt_copies)
            _drain_and_arrive()
            return swapped
        _barrier_wait()
        # Waves i and i + 4 share a SIMD; the high half prefetches before reading K so the pair
        # does not run the same phase in lockstep.
        if hi_wave:
            _copy_if(nxt < n_tiles, nxt_copies)
            k_vals = load_k(k_cur)
        else:
            k_vals = load_k(k_cur)
            _copy_if(nxt < n_tiles, nxt_copies)
        rocdl.sched_barrier(0)
        s = _qk_gemm(k_vals, q_frags)
        rocdl.sched_barrier(0)
        v_vals = load_v(v_cur)
        rocdl.sched_barrier(0)
        kv_last = None
        if masked:
            kv_last = [
                fx.min(seq_idx[qt] + causal_off + window_right, kv_len - fx.Int32(1)) for qt in range(Q_TILES)
            ]
        p, m_new, d_new, corr, rescale = _softmax(
            s=s,
            m_prev=m_prev,
            d_prev=d_prev,
            lane_idx=lane_idx,
            c_exp2=c_exp2,
            kv_pos_base=kv_tile_start,
            kv_last=kv_last,
        )
        _drain_and_arrive()
        o_scaled = []
        for qt in range(Q_TILES):
            corr_vec = fx.Vector.from_elements([corr[qt]], fx.Float32).broadcast_to(8)
            o_scaled.append(list(_rescale_if(o_acc[qt], corr_vec, rescale[qt])))
        o_new = _pv_gemm(v_vals, p, o_scaled)
        out = []
        for qt in range(Q_TILES):
            out += [m_new[qt], d_new[qt]] + o_new[qt]
        return out + swapped

    @flyc.jit
    def run_tiles(state, lo, hi, *, masked):
        final_state = state
        for t, carried in range(fx.Index(lo), fx.Index(hi), 1, init=state):
            next_state = tile(fx.Int32(t), list(carried), masked=masked)
            final_state = yield next_state
        return final_state

    @flyc.jit
    def run_sync_tiles(ptrs, lo, hi):
        final_ptrs = ptrs
        for t, carried in range(fx.Index(lo), fx.Index(hi), 1, init=ptrs):
            next_ptrs = tile(fx.Int32(t), list(carried), masked=False, sync_only=True)
            final_ptrs = yield next_ptrs
        return final_ptrs

    # Tiles [0, clean_hi) need no mask for any row of the workgroup, [clean_hi, wave_end) are
    # masked, and [wave_end, n_tiles) are fully masked for every row of this wave.
    n_last = n_tiles - fx.Int32(1)
    qmax_min = fx.max(block_x * fx.Int32(BLOCK_M) + causal_off + window_right, fx.Int32(0))
    clean_hi = fx.max(fx.min((qmax_min + fx.Int32(1)) // fx.Int32(N_BLOCK), n_last), fx.Int32(0))
    state = run_tiles(init, fx.Int32(0), clean_hi, masked=False)
    wave_max_seq = fx.min(
        block_x * fx.Int32(BLOCK_M) + warp_idx * fx.Int32(ROWS_PER_WAVE) + fx.Int32(ROWS_PER_WAVE - 1),
        q_len - fx.Int32(1),
    )
    wave_q_max = fx.max(wave_max_seq + causal_off + window_right, fx.Int32(-1))
    wave_end = fx.max(fx.min((wave_q_max + fx.Int32(N_BLOCK)) // fx.Int32(N_BLOCK), n_tiles), clean_hi)
    state = run_tiles(state, clean_hi, wave_end, masked=True)
    run_sync_tiles(list(state[ptr_base:]), wave_end, n_tiles)
    _barrier_wait()

    # O / d goes out through the LDS slot the last tile did not use.
    writer = OutputWriter(
        o=o,
        stride_o_seq=stride_o_seq,
        stride_o_head=stride_o_head,
        q_start=q_start,
        q_len=q_len,
        head=head,
        block_x=block_x,
        warp_idx=warp_idx,
        lane_idx=lane_idx,
    )
    o_lds = k_slot(n_tiles % fx.Int32(2))
    for qt in range(Q_TILES):
        d_final = fx.Float32(state[qt * nst + 1])
        inv = (d_final > fx.Float32(0.0)).select(fx.Float32(1.0) / d_final, fx.Float32(0.0))
        inv_vec = fx.Vector.from_elements([inv], fx.Float32).broadcast_to(8)
        o_norm = [fx.Vector(state[qt * nst + 2 + dt]) * inv_vec for dt in range(D_TILES)]
        if qt > 0:
            rocdl.s_wait_dscnt(0)  # nothing is pending; the wait only pins the epilogue schedule
        writer.stage(o_lds, o_norm, qt)
    writer.flush()

    # lse = (m + log2(d)) * ln(2), stored by the lanes holding kv half 0. Masked lanes store at
    # byte offset 0x7FFFFFFF, past the descriptor bound, so the store is dropped.
    khalf0 = (lane_idx // fx.Int32(WMMA_M)) == fx.Int32(0)
    lse_rsrc = create_buffer_resource(lse, num_records_bytes=lse_bytes)
    for qt in range(Q_TILES):
        m_final = fx.Float32(state[qt * nst])
        d_final = fx.Float32(state[qt * nst + 1])
        lse_val = (m_final + fx.log2(d_final)) * fx.Float32(1.0 / LOG2E)
        mask = khalf0 & (seq_idx[qt] < q_len)
        off = lse_base + seq_idx[qt] * stride_lse_seq + head * stride_lse_head
        off = mask.select(off * fx.Int32(4), fx.Int32(0x7FFFFFFF))
        buffer_store(lse_val, lse_rsrc, off, mask=None, offset_is_bytes=True)


@flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
def mla_fwd_kernel(
    o: fx.Tensor,
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    lse: fx.Tensor,
    unused_tensor: fx.Tensor,
    softmax_scale: fx.Float32,
    stride_q_seq: fx.Int32,
    stride_k_seq: fx.Int32,
    stride_v_seq: fx.Int32,
    stride_o_seq: fx.Int32,
    stride_q_head: fx.Int32,
    stride_k_head: fx.Int32,
    stride_v_head: fx.Int32,
    stride_o_head: fx.Int32,
    stride_lse_seq: fx.Int32,
    stride_lse_head: fx.Int32,
    stride_lse_batch: fx.Int32,
    unused_i32: fx.Int32,
    window_right: fx.Int32,
    seq_len_q: fx.Int32,
    seq_len_k: fx.Int32,
):
    """Grid (q blocks, heads, batch). Query i sees keys j <= i + seq_len_k - seq_len_q + window_right.

    The two unused arguments keep the argument layout the kernel was tuned with.
    """
    batch = _block_id("z")
    lse_base = batch * stride_lse_batch
    lse_bytes = fx.Int64(lse_base + stride_lse_batch) * fx.Int64(4)
    q_start = batch * seq_len_q
    kv_start = batch * seq_len_k
    lds_base = fx.Int32(fx.ptrtoint(fx.SharedAllocator().allocate(LDS_BYTES).peek().ptr))
    warp_idx = fx.Int32(rocdl.wave_id())
    args = dict(
        o=o,
        q=q,
        k=k,
        v=v,
        lse=lse,
        softmax_scale=softmax_scale,
        stride_q_seq=stride_q_seq,
        stride_k_seq=stride_k_seq,
        stride_v_seq=stride_v_seq,
        stride_o_seq=stride_o_seq,
        stride_q_head=stride_q_head,
        stride_k_head=stride_k_head,
        stride_v_head=stride_v_head,
        stride_o_head=stride_o_head,
        stride_lse_seq=stride_lse_seq,
        stride_lse_head=stride_lse_head,
        lse_base=lse_base,
        lse_bytes=lse_bytes,
        q_start=q_start,
        q_len=seq_len_q,
        kv_start=kv_start,
        kv_len=seq_len_k,
        window_right=window_right,
        warp_idx=warp_idx,
        lds_base=lds_base,
    )
    if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):
        _attention(hi_wave=False, **args)
    else:
        _attention(hi_wave=True, **args)


@flyc.jit
def launch_mla_fwd(
    o: fx.Tensor,
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    lse: fx.Tensor,
    unused_tensor: fx.Tensor,
    softmax_scale: fx.Float32,
    stride_q_seq: fx.Int32,
    stride_k_seq: fx.Int32,
    stride_v_seq: fx.Int32,
    stride_o_seq: fx.Int32,
    stride_q_head: fx.Int32,
    stride_k_head: fx.Int32,
    stride_v_head: fx.Int32,
    stride_o_head: fx.Int32,
    stride_lse_seq: fx.Int32,
    stride_lse_head: fx.Int32,
    stride_lse_batch: fx.Int32,
    unused_i32: fx.Int32,
    window_right: fx.Int32,
    seq_len_q: fx.Int32,
    seq_len_k: fx.Int32,
    num_heads: fx.Int32,
    batch: fx.Int32,
    stream: fx.Stream,
):
    _ = SOURCE_DIGEST  # FlyDSL disk-cache key
    grid_x = fx.Index(fx.ceildiv(fx.Uint32(seq_len_q), fx.Uint32(BLOCK_M)))
    mla_fwd_kernel(
        o,
        q,
        k,
        v,
        lse,
        unused_tensor,
        softmax_scale,
        stride_q_seq,
        stride_k_seq,
        stride_v_seq,
        stride_o_seq,
        stride_q_head,
        stride_k_head,
        stride_v_head,
        stride_o_head,
        stride_lse_seq,
        stride_lse_head,
        stride_lse_batch,
        unused_i32,
        window_right,
        seq_len_q,
        seq_len_k,
    ).launch(grid=(grid_x, fx.Index(num_heads), fx.Index(batch)), block=(BLOCK_SIZE, 1, 1), stream=stream)


# Expert scheduling mode turns the hardware dependency interlocks off and has LLVM insert the
# dependency-counter waits, so memory ops must stay plain intrinsics (no inline asm) it can see.
# waves_per_eu = 2: the 8 waves of a workgroup run two per SIMD, so each may use 512 VGPRs.
launch_mla_fwd.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}
launch_mla_fwd.compile_hints["waves_per_eu"] = 2
