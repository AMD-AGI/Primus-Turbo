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
one natural-log unit before it rescales O. On grids that allow it the workgroups are persistent
(at most one per CU, taking q blocks from global counters), and a block's Q and first K/V tile are
copied in during the previous block.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm

# UNSTABLE(gfx1250): the rocdl / tdm_ops calls (WMMA, permlanex16, ballot, mbcnt_lo, wave_id,
# sched_barrier, the split workgroup barrier, s_wait_dscnt, tensor_wait) have no stable API in
# FlyDSL 0.3.4.1; arith is stable.
from flydsl.expr import arith, rocdl
from flydsl.expr.rocdl import tdm_ops

from primus_turbo.flydsl.utils.buffer_ops import buffer_store, create_buffer_resource_from_addr
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
    O_ROW_ELEMS,
    Q_ROW_ELEMS,
    Q_TILES,
    ROWS_PER_WAVE,
    V_LDS_BYTES,
    V_ROW_ELEMS,
    WAVE_SIZE,
    WMMA_K,
    WMMA_M,
    KVTileLoader,
    OutputWriter,
    global_ptr,
    issue_q_copy,
    k_read_base,
    lds_ptr,
    load_k,
    load_v,
    read_q_fragments,
    v_read_base,
)

BLOCK_SIZE = WAVE_SIZE * NUM_WAVES
WMMA_N = 16
D_TILES = D_V // WMMA_M  # 16-column O accumulators per q tile
# LDS: two K/V slots, per-wave Q and O regions, the kernel arguments and two block records.
KV_BLOCK_BYTES = 64 * 1024
SLOT_BYTES = 2 * KV_BLOCK_BYTES
LDS_BYTES = 320 * 1024  # a gfx1250 workgroup's whole LDS, as the source kernel takes it
assert max(K_LDS_BYTES, V_LDS_BYTES) <= KV_BLOCK_BYTES
# LDS is 64 KB segments behind two 256 B/clk read ports that only read in the same cycle from
# different segments. Each K / V tile straddles a segment boundary at its middle row, so the reads of
# kv rows [0, 32) and [32, 64) alternate between two segments: K slot p is centred on (2p + 1) * 64 KB,
# V slot p on (2p + 2) * 64 KB.
K_SLOT_OFF = KV_BLOCK_BYTES - K_LDS_BYTES // 2
V_SLOT_OFF = 2 * KV_BLOCK_BYTES - V_LDS_BYTES // 2
# Q's per-wave regions lie outside both K/V slots, so the next block's Q can come in while a block
# runs: waves 0-3 below K slot 0, waves 4-7 above V slot 1. The kernel arguments follow the low Q
# regions; the two block records (current and next block) sit between V slot 0 and K slot 1.
Q_REGION_BYTES = ROWS_PER_WAVE * Q_ROW_ELEMS * 2
Q_HI_OFF = SLOT_BYTES + V_SLOT_OFF + V_LDS_BYTES
ARGS_OFF = 4 * Q_REGION_BYTES
REC_OFF = V_SLOT_OFF + V_LDS_BYTES
REC_BYTES = 16  # (ticket, q block, head, batch)
# O leaves through per-wave regions no copy writes: waves 0-3 in the gap between K and V of slot 0,
# waves 4-7 after the block records (between V of slot 0 and K of slot 1). An O store of one block is
# waited for before the next block stages its O.
O_WAVE_BYTES = ROWS_PER_WAVE * O_ROW_ELEMS * 2
O_LO_OFF = K_SLOT_OFF + K_LDS_BYTES
O_HI_OFF = REC_OFF + 2 * REC_BYTES
assert O_LO_OFF + 4 * O_WAVE_BYTES <= V_SLOT_OFF and O_HI_OFF + 4 * O_WAVE_BYTES <= SLOT_BYTES + K_SLOT_OFF
assert (K_LDS_BYTES // 2) % (K_ROW_ELEMS * 2) == 0 and (V_LDS_BYTES // 2) % (V_ROW_ELEMS * 2) == 0
assert K_SLOT_OFF + K_LDS_BYTES <= V_SLOT_OFF
assert ARGS_OFF + 4 * 32 <= K_SLOT_OFF and Q_HI_OFF + 4 * Q_REGION_BYTES <= LDS_BYTES
assert REC_OFF + 2 * REC_BYTES <= SLOT_BYTES + K_SLOT_OFF
# Wave w owns the block rows ROWS_PER_WAVE * w + [0, ROWS_PER_WAVE) (two q tiles QT_STRIDE rows apart).
QT_STRIDE = WMMA_M
# Head-grouped dispatch: KV_GROUP q blocks of KV_HEAD_CHUNK heads run together (powers of two).
KV_GROUP = 16
KV_HEAD_CHUNK = 16
# The running max may lag the row max by this many natural-log units before O is rescaled.
RESCALE_THRESHOLD = 1.0
# Ticket groups: workgroup p takes the tickets of group g = p % TICKET_GROUPS first, which keeps a head's
# blocks on one XCD (the hardware hands workgroups to the XCDs round robin; a head's K/V is then read
# through one L2). Group g holds two of every 16 consecutive tickets, head lanes g and 15 - g (one even,
# one odd: measured slightly faster than lanes g and g + 8).
TICKET_GROUPS = 8
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


def _remap(lin, gx, gy, gz):
    """(q block, head, batch) of dispatch rank ``lin`` after the longest-first remap.

    Causal work grows with the q block index, so the linear order is remapped to start the last q
    block of every (head, batch) first. When the grid divides evenly, windows of KV_GROUP ranks go
    out KV_HEAD_CHUNK heads at a time so a head's K/V is shared in L2.
    """
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
    return gx - fx.Int32(1) - rank, rem % gy, rem // gy


# Persistent mode (persist_hb = heads | batch << 16 != 0; the host enables it for causal grids of full
# q blocks with seqlen_q == seqlen_kv, q blocks and heads * batch multiples of 16): a grid of at most one
# workgroup per CU, each taking q blocks (tickets in the dispatch order of the non-persistent grid)
# from the ticket counters in ``unused_tensor`` (0 between calls). Wave 0 takes the next ticket as its
# masked tile begins (four tiles before the block's end) and stores the next block's record before the
# arrive of its first sync-only tile; the low waves copy the next block's Q and first K/V tile in
# during the block's last tile (sync-only for them).


def _grid(persist_hb, q_len):
    """(persist, gx, gy, gz, number of blocks) of the logical block grid (q blocks, heads, batch)."""
    persist = persist_hb != fx.Int32(0)
    gx = persist.select(fx.ceildiv(q_len, fx.Int32(BLOCK_M)), fx.grid_dim.x)
    gy = persist.select(persist_hb & fx.Int32(0xFFFF), fx.grid_dim.y)
    gz = persist.select(persist_hb.shrui(fx.Int32(16)), fx.grid_dim.z)
    return persist, gx, gy, gz, gx * gy * gz


# Counter layout (int32s 256 B apart): group g's counter at g * 256 B, the done counter after them.
COUNTER_STRIDE = 256


@flyc.jit
def _atomic_ticket(cond, addr, total):
    """Post-increment of this workgroup's group counter (in ``addr``'s counter block) by the lanes where
    ``cond`` holds (0 elsewhere). A wrapping increment (bound 2^31 - 1, never reached): LLVM's atomic
    optimizer leaves it alone (it would wait for an atomic add's result right after the add)."""
    r = fx.Int32(0)
    if cond:
        g = fx.block_idx.x & fx.Int32(TICKET_GROUPS - 1)
        r = fx.Int32(
            llvm.atomicrmw(
                llvm.AtomicBinOp.uinc_wrap,
                global_ptr(addr + fx.Int64(g * fx.Int32(COUNTER_STRIDE))),
                fx.as_ir_value(fx.Int32(0x7FFFFFFF)),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
            )
        )
    return r


@flyc.jit
def _atomic_add_if(cond, addr, val):
    r = fx.Int32(0)
    if cond:
        r = fx.llvm.atomic_add(global_ptr(addr), val, syncscope="agent")
    return r


@flyc.jit
def _reset_counters_if(cond, addr):
    if cond:
        for g in range(TICKET_GROUPS + 1):
            fx.llvm.atomic_xchg(
                global_ptr(addr + fx.Int64(g * COUNTER_STRIDE)), fx.Int32(0), syncscope="agent"
            )


# The HOLD_BACK shortest q blocks of every head go to the end of each group's sequence.
HOLD_BACK = 2


def _group_ticket(g, k, gx, gyz):
    """The k-th ticket (in the hardware dispatch order) of group g: two of every 16 consecutive tickets
    (head lanes g and 15 - g), the positions (16 tickets each) in the hardware order except that the
    HOLD_BACK shortest q blocks of every head chunk come last, chunk by chunk (grouped grids only)."""
    lane = ((k & fx.Int32(1)) == fx.Int32(0)).select(g, fx.Int32(15) - g)
    i = k >> fx.Int32(1)
    hc = gyz >> fx.Int32(4)
    reg = (gx - fx.Int32(16)) * hc  # positions of the q groups before the last one
    main = reg + hc * fx.Int32(16 - HOLD_BACK)
    j = i - reg
    jt = i - main
    p_main = reg + (j // fx.Int32(16 - HOLD_BACK)) * fx.Int32(16) + j % fx.Int32(16 - HOLD_BACK)
    p_tail = (
        reg + (jt // fx.Int32(HOLD_BACK)) * fx.Int32(16) + fx.Int32(16 - HOLD_BACK) + jt % fx.Int32(HOLD_BACK)
    )
    p = (i < reg).select(i, (i < main).select(p_main, p_tail))
    return (p << fx.Int32(4)) + lane


def _uniform(raw):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.as_ir_value(raw)))


@flyc.jit
def _steal(t0, total, g, addr, lane, persist, gx, gyz):
    """Past this workgroup's own group: the first unexhausted other group's next ticket, or total
    (persistent mode only)."""
    t = t0
    kmax = total >> fx.Int32(3)
    h = persist.select(fx.Int32(1), fx.Int32(TICKET_GROUPS))
    while (h < fx.Int32(TICKET_GROUPS)) & (t >= total):
        gh = (g + h) & fx.Int32(TICKET_GROUPS - 1)
        k = _uniform(
            _atomic_add_if(lane == fx.Int32(0), addr + fx.Int64(gh * fx.Int32(COUNTER_STRIDE)), fx.Int32(1))
        )
        t = (k < kmax).select(_group_ticket(gh, k, gx, gyz), total)
        h = h + fx.Int32(1)
    return t


def _finish_ticket(raw, persist, total, addr, lane, gx, gyz):
    """The wave-uniform ticket from ``_atomic_ticket``'s lane-0 result (else past the last block): from
    this workgroup's group, else stolen from another group; with none left the workgroup counts itself
    done, and the last one done resets the counters (persistent mode; else past the last block)."""
    g = fx.block_idx.x & fx.Int32(TICKET_GROUPS - 1)
    k = _uniform(raw)
    t = (k < (total >> fx.Int32(3))).select(_group_ticket(g, k, gx, gyz), total)
    t = _steal(t, total, g, addr, lane, persist, gx, gyz)
    none = persist & (t >= total)
    done_addr = addr + fx.Int64(TICKET_GROUPS * COUNTER_STRIDE)
    d = _uniform(_atomic_add_if(none & (lane == fx.Int32(0)), done_addr, fx.Int32(1)))
    _reset_counters_if(none & (lane == fx.Int32(0)) & (d == fx.Int32(fx.grid_dim.x) - fx.Int32(1)), addr)
    return persist.select(t, total)


def _rec_load(rec):
    """(ticket, q block, head, batch) of the block record at LDS byte address ``rec``."""
    v4 = fx.Vector.make_type(4, fx.Int32)
    w = fx.Vector(llvm.load(v4, lds_ptr(rec)))
    return [fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.as_ir_value(w[i]))) for i in range(4)]


@flyc.jit
def _rec_store(cond, rec, vals):
    if cond:
        llvm.store(fx.as_ir_value(fx.Vector.from_elements(list(vals), fx.Int32)), lds_ptr(rec), alignment=16)


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


def _issue_if(cond, loader, slot_base, row):
    """This wave's copies of the tile at kv row ``row`` into the slot at ``slot_base``, if ``cond``."""

    @flyc.jit
    def _go(cond, slot_base, row):
        if cond:
            loader.issue(slot_base, row)

    _go(cond, slot_base, row)


@flyc.jit
def _rescale_if(o_vecs, corr_vec, do_rescale):
    result = o_vecs
    if do_rescale:
        result = [ov * corr_vec for ov in o_vecs]
    return result


def _call_if(cond, fn):
    """fn() (which emits IR) under a scalar condition."""

    @flyc.jit
    def _go(cond):
        if cond:
            fn()

    _go(cond)


class _Args:
    """The per-block kernel arguments, kept in LDS (ARGS_OFF) and read back into scalar registers at
    each use site, so they are not live across the tile loops of a persistent workgroup. ``zero`` is
    an opaque 0: lane / wave / LDS values derived from it in a block are not hoisted out of the block
    loop (where they would stay live across the tile loops)."""

    NAMES = (
        "q", "k", "v", "o", "lse", "ctr",  # 64-bit addresses, two words each
        "stride_q_seq", "stride_k_seq", "stride_v_seq", "stride_o_seq",
        "stride_q_head", "stride_k_head", "stride_v_head", "stride_o_head",
        "stride_lse_seq", "stride_lse_head", "stride_lse_batch", "persist_hb", "zero",
    )  # fmt: skip
    PTRS = 6
    WORDS = 2 * PTRS + len(NAMES) - PTRS
    LOADS = (WORDS + 3) // 4

    @staticmethod
    def store(lds, vals):
        words = []
        for name in _Args.NAMES:
            x = vals[name]
            if isinstance(x, fx.Int64):
                words += [fx.Int32(x), fx.Int32(x >> fx.Int64(32))]
            else:
                words.append(x)
        words += [fx.Int32(0)] * (4 * _Args.LOADS - len(words))
        for i in range(_Args.LOADS):
            vec = fx.Vector.from_elements(words[4 * i : 4 * i + 4], fx.Int32)
            llvm.store(fx.as_ir_value(vec), lds_ptr(lds + fx.Int32(16 * i)), alignment=16)

    def __init__(self, lds):
        v4 = fx.Vector.make_type(4, fx.Int32)
        words = []
        for i in range(_Args.LOADS):
            vec = fx.Vector(llvm.load(v4, lds_ptr(lds + fx.Int32(16 * i))))
            words += [
                fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.as_ir_value(vec[j]))) for j in range(4)
            ]
        for i, name in enumerate(_Args.NAMES):
            if i < _Args.PTRS:
                lo, hi = words[2 * i], words[2 * i + 1]
                val = (fx.Int64(hi) << fx.Int64(32)) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))
            else:
                val = words[_Args.PTRS + i]
            setattr(self, name, val)


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
    stride_lse_batch,
    q_len,
    kv_len,
    window_right,
    persist_hb,
    ctr,
    warp_idx,
    hi_wave,
    lds_base,
):
    """One wave's share of the workgroup's q blocks: 32 query rows of one (batch, head) per block."""
    # Lane id within the wave32: mbcnt of all lanes below this one (fx.gpu.lane_id changes the code).
    lane_idx = fx.Int32(rocdl.mbcnt_lo(fx.Int32.ir_type, fx.Int32(-1).ir_value(), fx.Int32(0).ir_value()))
    args_lds = lds_base + fx.Int32(ARGS_OFF)
    ctr_addr = fx.Int64(fx.ptrtoint(fx.get_iter(ctr)))

    @flyc.jit
    def _setup(cond):
        # Wave 0 keeps the kernel arguments in LDS and stores the record of the workgroup's first block.
        if cond:
            _Args.store(
                args_lds,
                dict(
                    q=fx.Int64(fx.ptrtoint(fx.get_iter(q))),
                    k=fx.Int64(fx.ptrtoint(fx.get_iter(k))),
                    v=fx.Int64(fx.ptrtoint(fx.get_iter(v))),
                    o=fx.Int64(fx.ptrtoint(fx.get_iter(o))),
                    lse=fx.Int64(fx.ptrtoint(fx.get_iter(lse))),
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
                    stride_lse_batch=stride_lse_batch,
                    persist_hb=persist_hb,
                    zero=fx.Int32(0),
                    ctr=ctr_addr,
                ),
            )
            # The workgroup's first block: a ticket (persistent mode) or its hardware dispatch rank.
            persist, gx, gy, gz, total = _grid(persist_hb, q_len)
            raw = _atomic_ticket(persist & (lane_idx == fx.Int32(0)), ctr_addr, total)
            t = _finish_ticket(raw, persist, total, ctr_addr, lane_idx, gx, gy * gz)
            lin_hw = fx.block_idx.x + fx.grid_dim.x * (fx.block_idx.y + fx.grid_dim.y * fx.block_idx.z)
            t = persist.select(t, lin_hw)
            bx, hd, bt = _remap(t, gx, gy, gz)
            _rec_store(lane_idx == fx.Int32(0), lds_base + fx.Int32(REC_OFF), [t, bx, hd, bt])

    _setup(warp_idx == fx.Int32(0))
    _drain_and_arrive()
    _barrier_wait()

    def rec_addr(lds, sl):
        return lds + fx.Int32(REC_OFF) + sl * fx.Int32(REC_BYTES)

    def slot_k(lds, pp):
        return lds + fx.Int32(K_SLOT_OFF) + pp * fx.Int32(SLOT_BYTES)

    def slot_v(lds, pp):
        return lds + fx.Int32(V_SLOT_OFF) + pp * fx.Int32(SLOT_BYTES)

    causal_off = kv_len - q_len

    def kv_extent(block_x):
        """(keys the block's last query attends, its number of K/V tiles)."""
        wg_max_seq = fx.min(block_x * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1), q_len - fx.Int32(1))
        kv_len_wg = fx.max(fx.min(wg_max_seq + causal_off + window_right + fx.Int32(1), kv_len), fx.Int32(1))
        return kv_len_wg, fx.ceildiv(kv_len_wg, fx.Int32(N_BLOCK))

    def make_loader(a, ids, wid):
        block_x, head, batch = ids
        return KVTileLoader(
            k=a.k,
            v=a.v,
            stride_k_seq=a.stride_k_seq,
            stride_k_head=a.stride_k_head,
            stride_v_seq=a.stride_v_seq,
            stride_v_head=a.stride_v_head,
            head=head,
            kv_start=batch * kv_len,
            kv_len_wg=kv_extent(block_x)[0],
            warp_idx=wid,
            v_lds_off=V_SLOT_OFF - K_SLOT_OFF,
        )

    def stage(ids, pp, z):
        """(Low waves) the Q copies of block ``ids`` (q block, head, batch) for this wave and its high
        partner (wave + 4) and this wave's share of the block's first K/V tile, into K/V slot ``pp``.
        ``z`` is an opaque 0 (the argument block's LDS address is not a loop invariant)."""
        a = _Args(args_lds + z)
        wid = warp_idx + a.zero
        lds = lds_base + a.zero
        block_x, head, batch = ids
        for w, region in (
            (wid, lds + wid * fx.Int32(Q_REGION_BYTES)),
            (wid + fx.Int32(NUM_WAVES // 2), lds + fx.Int32(Q_HI_OFF) + wid * fx.Int32(Q_REGION_BYTES)),
        ):
            issue_q_copy(
                q_addr=a.q,
                stride_q_seq=a.stride_q_seq,
                stride_q_head=a.stride_q_head,
                q_start=batch * q_len,
                q_len=q_len,
                head=head,
                block_x=block_x,
                warp_idx=w,
                region=region,
                tile_stride=QT_STRIDE,
            )
        make_loader(a, ids, wid).issue(slot_k(lds, pp), fx.Int32(0))

    def stage_next(sl, pp, z):
        """(Low waves, past the barrier after wave 0 stored the next block's record ``sl``) if the record
        is a block, stage it into K/V slot ``pp``."""
        a = _Args(args_lds + z)
        lds = lds_base + a.zero
        t, bx, hd, bt = _rec_load(rec_addr(lds, sl))
        persist, _, _, _, total = _grid(a.persist_hb, q_len)
        _call_if(persist & (t < total), lambda: stage((bx, hd, bt), pp, a.zero))

    if not hi_wave:
        t0, bx0, hd0, bt0 = _rec_load(rec_addr(lds_base, fx.Int32(0)))
        _call_if(t0 < _grid(persist_hb, q_len)[4], lambda: stage((bx0, hd0, bt0), fx.Int32(0), fx.Int32(0)))
    c_exp2 = softmax_scale * fx.Float32(LOG2E)

    def one_block(sl, par, z, ids):
        """The block ``ids`` (q block, head, batch) of record ``sl``; its tile 0 is in K/V slot ``par``.
        Returns (whether a next block follows, its record, its tile-0 slot, an opaque 0, its ids). ``z``
        is an opaque 0 carried by the block loop, so the LDS address of the argument block (a constant)
        is not hoisted out of the loop into a VGPR live across it."""
        a = _Args(args_lds + z)
        # Lane, wave and LDS values recomputed from an opaque zero, so LLVM does not hoist them out of
        # the block loop (they would stay live across the tile loops).
        lane = lane_idx + a.zero
        wid = warp_idx + a.zero
        lds = lds_base + a.zero
        sl_next = sl ^ fx.Int32(1)
        block_x, head, batch = ids
        seq_idx = [
            block_x * BLOCK_M + qt * QT_STRIDE + wid * ROWS_PER_WAVE + lane % WMMA_M for qt in range(Q_TILES)
        ]
        if hi_wave:
            q_blk = lds + fx.Int32(Q_HI_OFF) + (wid - fx.Int32(NUM_WAVES // 2)) * fx.Int32(Q_REGION_BYTES)
        else:
            q_blk = lds + wid * fx.Int32(Q_REGION_BYTES)

        # Tiles [0, n_tiles) hold every key the workgroup's last query attends.
        n_tiles = kv_extent(block_x)[1]
        par_next = (par + n_tiles) & fx.Int32(1)
        loader = make_loader(a, ids, wid)
        # The low waves copied this block's Q and first K/V tile during the previous block (or before
        # the loop): they wait for the copies, then all waves sync.
        _drain_and_arrive()
        _barrier_wait()
        q_frags = read_q_fragments(q_blk, lane)
        # Split workgroup barrier, one arrive and one wait per tile: a wave arrives once its reads of
        # the current slot and its share of the next tile's copy (low waves only) are done and waits
        # before the next tile, so it may run one PV GEMM ahead of the slowest wave.
        _drain_and_arrive()

        # The O / d seeds are +0.0 from the opaque zero: a constant would be hoisted out of the block
        # loop into a VGPR live across it.
        fz = a.zero.bitcast(fx.Float32)
        init = []
        for _ in range(Q_TILES):
            init += [fx.Float32(BIG_NEG), fz]
            init += [fx.Vector.from_elements([fz], fx.Float32).broadcast_to(8) for _ in range(D_TILES)]
        nst = 2 + D_TILES  # (m, d, O accumulators) per q tile
        ptr_base = len(init)
        par1 = par ^ fx.Int32(1)
        init += [
            k_read_base(slot_k(lds, par), lane),
            k_read_base(slot_k(lds, par1), lane),
            v_read_base(slot_v(lds, par), lane),
            v_read_base(slot_v(lds, par1), lane),
        ]

        def tile(t, state, *, masked, sync_only=False):
            """One kv tile: wait for its K/V, prefetch the next one, then QK, softmax, rescale, PV.

            ``masked`` tiles apply the causal edge and the kv tail. ``sync_only`` tiles are fully
            masked for every row of this wave and keep only the barrier and the prefetch.
            """
            pb = 0 if sync_only else ptr_base
            nxt_pp = (t + fx.Int32(1) + par) & fx.Int32(1)
            kv_tile_start = t * fx.Int32(N_BLOCK)
            if not sync_only:
                m_prev = [fx.Float32(state[qt * nst]) for qt in range(Q_TILES)]
                d_prev = [fx.Float32(state[qt * nst + 1]) for qt in range(Q_TILES)]
                o_acc = [
                    [fx.Vector(state[qt * nst + 2 + dt]) for dt in range(D_TILES)] for qt in range(Q_TILES)
                ]
            k_cur, k_nxt, v_cur, v_nxt = state[pb : pb + 4]
            nxt = t + fx.Int32(1)
            nxt_row = nxt * fx.Int32(N_BLOCK)
            swapped = [k_nxt, k_cur, v_nxt, v_cur]
            if sync_only:
                _barrier_wait()
                if not hi_wave:
                    _issue_if(nxt < n_tiles, loader, slot_k(lds, nxt_pp), nxt_row)
                _drain_and_arrive()
                if not hi_wave:
                    # The last tile reads the other slot; past this tile's barrier every wave has
                    # read this block's Q and the K/V of the tile before. Waited for at the next
                    # block's first arrive.
                    _call_if(nxt == n_tiles, lambda: stage_next(sl_next, par_next, a.zero))
                return swapped
            _barrier_wait()
            # Waves i and i + 4 share a SIMD and the high one loses issue arbitration to the low one,
            # so it arrives last at every barrier: only the low waves copy K / V (after reading K).
            if hi_wave:
                k_vals = load_k(k_cur)
            else:
                k_vals = load_k(k_cur)
                _issue_if(nxt < n_tiles, loader, slot_k(lds, nxt_pp), nxt_row)
            rocdl.sched_barrier(0)
            s = _qk_gemm(k_vals, q_frags)
            rocdl.sched_barrier(0)
            v_vals = load_v(v_cur)
            rocdl.sched_barrier(0)
            kv_last = None
            if masked:
                kv_last = [
                    fx.min(seq_idx[qt] + causal_off + window_right, kv_len - fx.Int32(1))
                    for qt in range(Q_TILES)
                ]
            p, m_new, d_new, corr, rescale = _softmax(
                s=s,
                m_prev=m_prev,
                d_prev=d_prev,
                lane_idx=lane,
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

        # Tiles [0, clean_hi) need no mask for any row of this wave, [clean_hi, wave_end) are masked,
        # and [wave_end, n_tiles) are fully masked for every row of this wave. Per wave, not per
        # workgroup: a wave's rows need no mask on a tile that ends at or before its first row's
        # diagonal, so only its own diagonal tile (and the kv tail) runs masked. The masked softmax
        # selects nothing on such tiles, so the result bits are the same either way.
        n_last = n_tiles - fx.Int32(1)
        wave_row0 = block_x * fx.Int32(BLOCK_M) + wid * fx.Int32(ROWS_PER_WAVE)
        qmax_min = fx.max(wave_row0 + causal_off + window_right, fx.Int32(0))
        clean_hi = fx.max(fx.min((qmax_min + fx.Int32(1)) // fx.Int32(N_BLOCK), n_last), fx.Int32(0))
        state = run_tiles(init, fx.Int32(0), clean_hi, masked=False)
        if not hi_wave:
            # Wave 0 takes the next ticket as its masked tile begins (persistent mode) ...
            c = _Args(args_lds + a.zero)
            persist_c, _, _, _, total_c = _grid(c.persist_hb, q_len)
            raw = _atomic_ticket(persist_c & (wid == fx.Int32(0)) & (lane == fx.Int32(0)), c.ctr, total_c)
        wave_max_seq = fx.min(wave_row0 + fx.Int32(ROWS_PER_WAVE - 1), q_len - fx.Int32(1))
        wave_q_max = fx.max(wave_max_seq + causal_off + window_right, fx.Int32(-1))
        wave_end = fx.max(fx.min((wave_q_max + fx.Int32(N_BLOCK)) // fx.Int32(N_BLOCK), n_tiles), clean_hi)
        state = run_tiles(state, clean_hi, wave_end, masked=True)
        if not hi_wave:
            # ... and stores the next block's record before the arrive of its first sync-only tile: the
            # low waves read it in their last tile, all waves before the block's last barrier wait.
            c = _Args(args_lds + a.zero)
            persist_c, _, _, _, total_c = _grid(c.persist_hb, q_len)
            lane_c = lane_idx + c.zero

            def put_ticket():
                _, gx, gy, gz, _ = _grid(c.persist_hb, q_len)
                t = _finish_ticket(raw, persist_c, total_c, c.ctr, lane_c, gx, gy * gz)
                bx, hd, bt = _remap(t, gx, gy, gz)
                _rec_store(lane_c == fx.Int32(0), rec_addr(lds_base + c.zero, sl_next), [t, bx, hd, bt])

            _call_if(persist_c & (warp_idx + c.zero == fx.Int32(0)), put_ticket)
        run_sync_tiles(list(state[ptr_base:]), wave_end, n_tiles)
        # The epilogue's LDS loads (independent: addresses from the block's opaque zero) go out before
        # the last barrier wait; the next block's record was stored before this wave's last tile.
        lane_e = lane_idx + a.zero
        wid_e = warp_idx + a.zero
        lds_e = lds_base + a.zero
        b = _Args(args_lds + a.zero)
        nxt_rec = _rec_load(rec_addr(lds_e, sl_next))
        _barrier_wait()
        if not hi_wave:
            # A low wave without a sync-only tile copies the next block's Q and first tile here.
            _call_if(wave_end == n_tiles, lambda: stage_next(sl_next, par_next, a.zero))

        # O goes out through this wave's O region; the previous block's O stores have read it by now.
        rocdl.s_wait_asynccnt(0)
        writer = OutputWriter(
            o_addr=b.o,
            stride_o_seq=b.stride_o_seq,
            stride_o_head=b.stride_o_head,
            q_start=batch * q_len,
            q_len=q_len,
            head=head,
            block_x=block_x,
            warp_idx=wid_e,
            lane_idx=lane_e,
            tile_stride=QT_STRIDE,
        )
        if hi_wave:
            o_lds = lds_e + fx.Int32(O_HI_OFF - 4 * O_WAVE_BYTES)
        else:
            o_lds = lds_e + fx.Int32(O_LO_OFF)
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
        lse_base = batch * b.stride_lse_batch
        lse_bytes = fx.Int64(lse_base + b.stride_lse_batch) * fx.Int64(4)
        khalf0 = (lane_e // fx.Int32(WMMA_M)) == fx.Int32(0)
        lse_rsrc = create_buffer_resource_from_addr(
            fx.as_ir_value(b.lse), num_records_bytes=fx.as_ir_value(lse_bytes)
        )
        for qt in range(Q_TILES):
            m_final = fx.Float32(state[qt * nst])
            d_final = fx.Float32(state[qt * nst + 1])
            lse_val = (m_final + fx.log2(d_final)) * fx.Float32(1.0 / LOG2E)
            mask = khalf0 & (seq_idx[qt] < q_len)
            off = lse_base + seq_idx[qt] * b.stride_lse_seq + head * b.stride_lse_head
            off = mask.select(off * fx.Int32(4), fx.Int32(0x7FFFFFFF))
            buffer_store(lse_val, lse_rsrc, off, mask=None, offset_is_bytes=True)
        persist_e, _, _, _, total_e = _grid(b.persist_hb, q_len)
        more = persist_e & (nxt_rec[0] < total_e)
        return [more.select(fx.Int32(1), fx.Int32(0)), sl_next, par_next, a.zero] + nxt_rec[1:]

    @flyc.jit
    def run_blocks():
        t0, bx, hd, bt = _rec_load(rec_addr(lds_base, fx.Int32(0)))
        more = (t0 < _grid(persist_hb, q_len)[4]).select(fx.Int32(1), fx.Int32(0))
        sl = fx.Int32(0)
        par = fx.Int32(0)
        z = fx.Int32(0)
        while more != fx.Int32(0):
            nxt = one_block(sl, par, z, (bx, hd, bt))
            more = fx.Int32(nxt[0])
            sl = fx.Int32(nxt[1])
            par = fx.Int32(nxt[2])
            z = fx.Int32(nxt[3])
            bx = fx.Int32(nxt[4])
            hd = fx.Int32(nxt[5])
            bt = fx.Int32(nxt[6])

    run_blocks()


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
    persist_hb: fx.Int32,
    window_right: fx.Int32,
    seq_len_q: fx.Int32,
    seq_len_k: fx.Int32,
):
    """Grid (q blocks, heads, batch), or (workgroups, 1, 1) with ``persist_hb`` = heads | batch << 16 and
    the ticket counter in ``unused_tensor`` (persistent mode). Query i sees keys
    j <= i + seq_len_k - seq_len_q + window_right.
    """
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
        stride_lse_batch=stride_lse_batch,
        q_len=seq_len_q,
        kv_len=seq_len_k,
        window_right=window_right,
        persist_hb=persist_hb,
        ctr=unused_tensor,
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
    persist_hb: fx.Int32,
    window_right: fx.Int32,
    seq_len_q: fx.Int32,
    seq_len_k: fx.Int32,
    num_heads: fx.Int32,
    batch: fx.Int32,
    persist_wgs: fx.Int32,
    stream: fx.Stream,
):
    _ = SOURCE_DIGEST  # FlyDSL disk-cache key
    persist = persist_hb != fx.Int32(0)
    grid_x = fx.Index(
        persist.select(persist_wgs, fx.Int32(fx.ceildiv(fx.Uint32(seq_len_q), fx.Uint32(BLOCK_M))))
    )
    grid_y = fx.Index(persist.select(fx.Int32(1), num_heads))
    grid_z = fx.Index(persist.select(fx.Int32(1), batch))
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
        persist_hb,
        window_right,
        seq_len_q,
        seq_len_k,
    ).launch(grid=(grid_x, grid_y, grid_z), block=(BLOCK_SIZE, 1, 1), stream=stream)


# Expert scheduling mode turns the hardware dependency interlocks off and has LLVM insert the
# dependency-counter waits, so memory ops must stay plain intrinsics (no inline asm) it can see.
# greedy-regclass-priority-trumps-globalness: with the block loop around the tile loops, the greedy
# register allocator otherwise splits a Q fragment of the high waves' tile loop to scratch (with it,
# the tile loops get the baseline's register assignment: the same s_set_vgpr_msb count, no spill).
# waves_per_eu = 2: the 8 waves of a workgroup run two per SIMD, so each may use 512 VGPRs.
launch_mla_fwd.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
    "greedy-regclass-priority-trumps-globalness": True,
}
launch_mla_fwd.compile_hints["waves_per_eu"] = 2
