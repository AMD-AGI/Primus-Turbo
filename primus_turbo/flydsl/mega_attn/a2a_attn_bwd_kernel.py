###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses all-to-all + causal attention backward + all-to-all in one kernel (single node, symmetric memory).

Every rank holds dO for its ``M`` tokens with all heads (``[M, n * hq_r, d]``, batch 1) and, from the
forward, q / k / v / O / LSE of its own heads over the whole sequence. One launch:

* every block but the pull blocks stages this rank's dO rows into its symmetric bank as one column
  block per destination rank, raising one flag per (row piece, destination);
* pull blocks (the first ``num_pull``) claim pull units ``(m_tile, src)``, copy ``src``'s rows of this
  rank's column block into dO ``[S, hq_r, d]`` (sequence order), fold the identity delta
  ``-rowsum(O * dO)`` of those rows on the way, write the rows' scaled, transposed LSE (the stock
  backward's ``_prescale_lse``), raise a local ready flag per unit and count it; then
  they join the workers. Stage and pull order: the rows of the first ``num_prefix`` bands from the
  bottom up (the diagonal q blocks the first wave of items starts on), then the rest from the top down;
* every block runs as a persistent worker over the FlyDSL backward's work items ``(band, split, kv head)``
  (the dK / dV KV-outer body, one 256-row kv band each): before each q block's first dO / delta fetch
  the body waits for the ready flag of the unit holding those rows (until the block has seen every unit
  counted), and the item counts itself on its band. The band's last item folds the band's ``q_split``
  dK / dV slots and pushes them into the recv bank of the ranks owning those tokens, then marks the band
  done. Claim order: the top ``num_early`` bands from the top down, then band 0 up (longest first);
* after every item is claimed: dQ push units, one per 256-row q block ``j`` in sequence order, wait for
  bands ``0..j`` (every band that adds into those rows of the atomic dQ image), un-permute and scale the
  rows, push them to their owners and zero them in the image for the next call; then gather units
  ``(m-tile, src)`` copy ``src``'s column blocks of the recv rows into dq / dk / dv ``[M, n * h_r, d]``.

Pull units are claimed by running blocks before those turn worker, every item is claimed before any dQ
push or gather unit, and an item's own waits are only on pull units, so no wait can deadlock whatever the
grid. Sequence split as in ``a2a_attn_kernel``.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.attention import flash_attn_bwd_stock as _bwd
from primus_turbo.flydsl.deep_ep.common import _asm, ld_lds, st_lds
from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.ulysses.common import (
    SPIN_TIMEOUT,
    STATE_ASSIST,
    STATE_ASSIST_DONE,
    STATE_CLAIM,
    STATE_TILE,
    SYS_NT,
    VEC_BF16,
    claim_unit,
    exit_and_bump_epoch,
    i64_base,
    opaque_tid,
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
from primus_turbo.flydsl.utils.prims import (
    _readfirstlane_i32,
    atomic_add,
    _i64,
    cast,
    ld,
    memory_fence,
    read_clock,
    spin_timed_out,
    st,
)

REGION_BWD_STAGE = 0
REGION_BWD_READY = 1
REGION_BWD_RECV = 2
# per band: the done flag (the item counts live in a private tensor: other kernels on this heap write
# their epoch-stamped flags here, which only an epoch-stamped protocol tolerates)
REGION_BWD_BAND = 3
BAND_DONE_AT = FLAGS_PER_REGION // 2

THREADS = 256
WARP = 64
# rows of one pull unit, of one push / gather tile
BM = 128
STAGE_SPLIT = 4
# kv rows of one backward band
BAND = 256
# 16 B vectors per thread in flight in the copies
_BATCH = 8

_BF16 = fx.BFloat16

# LDS stash of the work loop's 64-bit bases (see _stashed)
_S_PEERS, _S_MY, _S_EPOCH, _S_COUNT, _S_WSK, _S_WSV, _S_IMG, _S_DQ, _S_STATE, _S_TRACE = 0, 1, 2, 3, 4, 5, 6, 7, 10, 11
_STASH_LEN = 12
# i32 word after the stash: the block has seen every pull unit counted
_DONE_AT = 2 * _STASH_LEN
# pull units done (reset at exit)
STATE_PULLED = STATE_ASSIST_DONE


def _ceildiv(a, b):
    return (a + b - 1) // b


def a2a_attn_bwd_bank_bytes(M, num_ranks, hq_r, hkv_r, head_dim):
    """Bank bytes: the dO staging rows, then the dq / dk / dv recv rows."""
    return M * num_ranks * (2 * hq_r + 2 * hkv_r) * head_dim * 2


def _plan(S, hq_r, hkv_r, d):
    """The stock dense SBHD backward plan for one rank's heads: ``(q_split, dkdv launcher)``."""
    p = _bwd._dense_plan(1, S, S, hq_r, hkv_r, d, 1.0 / math.sqrt(d), -1, True, False)
    q_split, block_kv, a16, a16_nat, dkdv_l, bat_plan = p[1], p[2], p[4], p[5], p[11], p[13]
    assert a16 and not a16_nat and block_kv == BAND and len(bat_plan) == 1, (
        "the fused backward runs the a16 dQ-image body with 256-row bands"
    )
    return q_split, dkdv_l


def _global_row(t, src, cs, n):
    """Sequence position of local token ``t`` of rank ``src``."""
    return (t // fx.Int32(cs) * fx.Int32(n) + src) * fx.Int32(cs) + t % fx.Int32(cs)


def _owner_row(pos, cs, n):
    """``(owner rank, local token)`` of sequence position ``pos``."""
    chunk = pos // fx.Int32(cs)
    return chunk % fx.Int32(n), chunk // fx.Int32(n) * fx.Int32(cs) + pos % fx.Int32(cs)


def _copy(thread_index, num_vecs, coords, src_res, dst_res, src_cm, dst_cm):
    """Copy ``num_vecs`` 16 B vectors (unrolled); ``coords(v)`` gives ``(src_word, dst_word)``."""
    per_iter = THREADS * _BATCH
    for it in range_constexpr(_ceildiv(num_vecs, per_iter)):
        words = [
            coords(fx.Int32(it * per_iter + j * THREADS) + thread_index)
            for j in range_constexpr(_BATCH)
            if it * per_iter + j * THREADS < num_vecs
        ]
        vals = [buffer_load(src_res, s, vec_width=4, dtype=fx.T.i32, cache_modifier=src_cm) for s, _ in words]
        for val, (_, dw) in zip(vals, words):
            buffer_store(val, dst_res, dw, cache_modifier=dst_cm)


def _fadd(a, b):
    return arith.addf(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fx.arith.FastMathFlags.fast)


def _row_dot(o_vec, do_vec, lane, lanes):
    """``sum(O * dO)`` over a row held as one 8-element vector by each of ``lanes`` consecutive lanes
    (the odo kernel's order: a serial fold per lane, then an xor butterfly)."""
    prod = Vec(o_vec).to(fx.Float32) * Vec(do_vec).to(fx.Float32)
    acc = fx.Float32(0.0)
    for i in range_constexpr(VEC_BF16):
        acc = fx.Float32(_fadd(acc, Vec(prod)[i]))
    for m in [1 << i for i in range_constexpr(lanes.bit_length() - 1)]:
        idx = fx.as_ir_value((lane ^ fx.Int32(m)) * fx.Int32(4))
        part = fx.as_ir_value(Vec.from_elements([acc], fx.Float32).bitcast(fx.Int32)[0])
        peer = rocdl.ds_bpermute(fx.Int32.ir_type, idx, part)
        peer_f = fx.Float32(fx.as_ir_value(Vec.from_elements([fx.Int32(peer)], fx.Int32).bitcast(fx.Float32)[0]))
        acc = fx.Float32(_fadd(acc, peer_f))
    return acc


@ASTRewriter.transform
def _stage_piece(thread_index, m_sub, dst, in_res, stage_res, **c):
    """Rows ``[m_sub * rows, +rows)`` of ``dst``'s heads of this rank's dO into the same place of the
    staging bank (dst's column block), retired on return."""
    h_vecs = c["hq_r"] * c["head_dim"] // VEC_BF16
    row_vecs = c["num_ranks"] * h_vecs
    rows = BM // STAGE_SPLIT
    t0 = m_sub * fx.Int32(rows)

    def coords(v):
        w = (t0 + v // fx.Int32(h_vecs)) * fx.Int32(row_vecs) + dst * fx.Int32(h_vecs) + v % fx.Int32(h_vecs)
        return w * fx.Int32(4), w * fx.Int32(4)

    _copy(thread_index, rows * h_vecs, coords, in_res, stage_res, 0, SYS_NT)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()


@ASTRewriter.transform
def _stage_role(thread_index, state_base, claim_slot, my_base, bank_offset, in_res, epoch, trace_base, **c):
    """Stage pieces ``(m_sub, dst)`` in the pull order (see the module doc) until none are left."""
    n, rank, m_tiles = c["num_ranks"], c["rank"], c["m_tiles"]
    stage_res = create_buffer_resource_from_addr(my_base + bank_offset, num_records_bytes=c["stage_bytes"])
    stage_flags = my_base + fx.Int64(REGION_BWD_STAGE * FLAG_REGION_BYTES)
    num_pieces = m_tiles * STAGE_SPLIT * n
    more = fx.Int32(1)
    while more == fx.Int32(1):
        piece = claim_unit(opaque_tid(thread_index), state_base, claim_slot, STATE_ASSIST)
        if piece < fx.Int32(num_pieces):
            k = piece // fx.Int32(n)
            pre = c["prefix_m_tiles"] * STAGE_SPLIT
            m_sub = arith.select(k < fx.Int32(pre), k, fx.Int32(m_tiles * STAGE_SPLIT - 1 + pre) - k)
            # the pullers of an m-tile start with src = dst - i: stage the other ranks' blocks first
            dst = (fx.Int32(rank + 1) + piece % fx.Int32(n)) % fx.Int32(n)
            _stage_piece(opaque_tid(thread_index), m_sub, dst, in_res, stage_res, **c)
            if thread_index == fx.Int32(0):
                st(stage_flags, m_sub * fx.Int32(n) + dst, epoch, scope="sys", order="relaxed")
                if const_expr(trace_base is not None):
                    st(trace_base, fx.Int32(c["trace_stage_at"]) + m_sub * fx.Int32(n) + dst, read_clock())
        more = arith.select(piece < fx.Int32(num_pieces), fx.Int32(1), fx.Int32(0))


@ASTRewriter.transform
def _pull_unit(thread_index, src, m_tile, src_heap, bank_offset, do_res, o_res, delta_res, prep, epoch, **c):
    """``src``'s staged dO rows ``[m_tile * BM, +BM)`` of this rank's column block into dO, with their delta;
    ``prep`` = (lse, scaled LSE) resources: those rows of the latter, as the stock ``_prescale_lse``
    rounds them (the Q prescale stays a launch: HBM-bound, it would slow the pull the items wait on)."""
    n, d, rank, cs, hq_r, S = c["num_ranks"], c["head_dim"], c["rank"], c["seq_chunk"], c["hq_r"], c["S"]
    lanes = d // VEC_BF16
    h_vecs = hq_r * lanes
    row_vecs = n * h_vecs
    if thread_index == fx.Int32(0):
        flag_base = src_heap + fx.Int64(REGION_BWD_STAGE * FLAG_REGION_BYTES)
        flag0 = m_tile * fx.Int32(STAGE_SPLIT * n) + fx.Int32(rank)
        spin_flags(flag_base, [flag0 + fx.Int32(i * n) for i in range(STAGE_SPLIT)], epoch, rank, src, "bwd pull", report=False)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    stage_res = create_buffer_resource_from_addr(src_heap + bank_offset, num_records_bytes=c["stage_bytes"])
    t0 = m_tile * fx.Int32(BM)
    # a unit lies inside one sequence chunk (cs % BM == 0): its rows land contiguously
    g0 = _global_row(t0, src, cs, n)
    lane = thread_index % fx.Int32(WARP)
    first_lane = thread_index % fx.Int32(lanes) == fx.Int32(0)
    per_iter = THREADS * _BATCH
    # the lanes of one (row, head) are consecutive: their partial sums fold within the wave
    for it in range(BM * h_vecs // per_iter):
        v0 = fx.Int32(it * per_iter) + thread_index
        vs = [v0 + fx.Int32(j * THREADS) for j in range_constexpr(_BATCH)]
        src_elems = [
            ((t0 + v // fx.Int32(h_vecs)) * fx.Int32(row_vecs) + fx.Int32(rank * h_vecs) + v % fx.Int32(h_vecs))
            * fx.Int32(VEC_BF16)
            for v in vs
        ]
        dst_elems = [(g0 * fx.Int32(h_vecs) + v) * fx.Int32(VEC_BF16) for v in vs]
        dos = [buffer_load(stage_res, e, vec_width=VEC_BF16, dtype=_BF16, cache_modifier=SYS_NT) for e in src_elems]
        os_ = [buffer_load(o_res, e, vec_width=VEC_BF16, dtype=_BF16) for e in dst_elems]
        for val, e in zip(dos, dst_elems):
            buffer_store(val, do_res, e)
        for j in range_constexpr(_BATCH):
            acc = _row_dot(os_[j], dos[j], lane, lanes)
            neg = arith.subf(fx.as_ir_value(fx.Float32(0.0)), fx.as_ir_value(acc), fastmath=fx.arith.FastMathFlags.fast)
            row = g0 + vs[j] // fx.Int32(h_vecs)
            head = vs[j] % fx.Int32(h_vecs) // fx.Int32(lanes)
            buffer_store(fx.Float32(neg), delta_res, head * fx.Int32(S) + row, mask=first_lane)
    lse_res, lsef_res = prep
    for it in range_constexpr(BM * c["hq_r"] // THREADS):
        x = fx.Int32(it * THREADS) + thread_index
        r = g0 + x % fx.Int32(BM)
        h = x // fx.Int32(BM)
        v = buffer_load(lse_res, r * fx.Int32(c["hq_r"]) + h, vec_width=1, dtype=fx.Float32)
        buffer_store(fx.Float32(v) * fx.Float32(-_bwd._LOG2E), lsef_res, h * fx.Int32(S) + r)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()


@ASTRewriter.transform
def _pull_role(thread_index, state_base, claim_slot, peers_base, my_base, bank_offset, do_res, o_res, delta_res, prep, epoch, trace_base, **c):
    """Pull units ``(m_tile, src)``, one 128-row block of the sequence each, the first ``prefix_tiles``
    from the bottom up and then the rest from the top down, until none are left."""
    n, m_tiles, cs, pre = c["num_ranks"], c["m_tiles"], c["seq_chunk"], c["prefix_tiles"]
    ready_base = my_base + fx.Int64(REGION_BWD_READY * FLAG_REGION_BYTES)
    num_units = m_tiles * n
    more = fx.Int32(1)
    while more == fx.Int32(1):
        unit = claim_unit(opaque_tid(thread_index), state_base, claim_slot, STATE_CLAIM)
        if unit < fx.Int32(num_units):
            g = arith.select(unit < fx.Int32(pre), unit, fx.Int32(num_units - 1 + pre) - unit)
            src, t = _owner_row(g * fx.Int32(BM), cs, n)
            m_tile = t // fx.Int32(BM)
            src_heap = ld(peers_base, src, dtype=fx.T.i64)
            _pull_unit(opaque_tid(thread_index), src, m_tile, src_heap, bank_offset, do_res, o_res, delta_res, prep, epoch, **c)
            if thread_index == fx.Int32(0):
                st(ready_base, m_tile * fx.Int32(n) + src, epoch, scope="agent", order="release")
                atomic_add(state_base, fx.Int32(STATE_PULLED), fx.Int64(1))
                if const_expr(trace_base is not None):
                    st(trace_base, m_tile * fx.Int32(n) + src, read_clock())
        more = arith.select(unit < fx.Int32(num_units), fx.Int32(1), fx.Int32(0))


@ASTRewriter.transform
def _q_ready(q, stash, **c):
    """The body's hook before its first dO / delta fetch of q block ``q``: until the block has seen every
    pull unit counted, wait for the ready flag of the unit holding row ``q``. Relaxed: the pull's release
    wrote the rows back, and no cache of this launch holds a line of them read before the flag."""
    n, cs, S = c["num_ranks"], c["seq_chunk"], c["S"]
    done = _readfirstlane_i32(ld(stash, fx.Int32(_DONE_AT), scope="workgroup", space=3))
    row = fx.Int32(q)
    # the body's prefetch past its last q block reads rows past the end (clamped) and never uses them
    if (done == fx.Int32(0)) & (row < fx.Int32(S)):
        src, t = _owner_row(row, cs, n)
        idx = t // fx.Int32(BM) * fx.Int32(n) + src
        ready_base = _region(stash, REGION_BWD_READY)
        epoch = _stashed(stash, _S_EPOCH)
        spin_start = read_clock()
        flag = ld(ready_base, idx, scope="agent", dtype=fx.T.i64)
        while flag < epoch:
            fx.rocdl.s_sleep(fx.Int32(1))
            # no printf: its hostcall code inside the body spills the body's main loop
            if spin_timed_out(spin_start, SPIN_TIMEOUT):
                _asm("s_trap 2")
            flag = ld(ready_base, idx, scope="agent", dtype=fx.T.i64)
        pulled = ld(_stashed(stash, _S_STATE), fx.Int32(STATE_PULLED), scope="agent", dtype=fx.T.i64)
        if pulled == fx.Int64(c["m_tiles"] * n):
            st(stash, fx.Int32(_DONE_AT), fx.Int32(1), scope="workgroup", space=3)


@ASTRewriter.transform
def _wait_bands(thread_index, band_base, last, epoch, **c):
    """Wait for the done flags of bands ``0..last``."""
    rank = c["rank"]
    idx = thread_index
    spin_start = read_clock()
    while idx <= last:
        flag = ld(band_base, fx.Int32(BAND_DONE_AT) + idx, scope="agent", dtype=fx.T.i64)
        while flag < epoch:
            fx.rocdl.s_sleep(fx.Int32(1))
            if spin_timed_out(spin_start, SPIN_TIMEOUT):
                fx.printf(f"[a2a_attn_bwd rank={rank}] band flag timeout: band={{}} flag={{}}\n", idx, flag)
                _asm("s_trap 2")
            flag = ld(band_base, fx.Int32(BAND_DONE_AT) + idx, scope="agent", dtype=fx.T.i64)
        idx = idx + fx.Int32(THREADS)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    memory_fence(order="acquire", scope="agent")


@ASTRewriter.transform
def _push_dkdv(thread_index, band, peers_base, recv_offset, wsk_res, wsv_res, epoch, **c):
    """Fold the band's ``q_split`` dK / dV slots (ascending, fp32, as the stock slot reduce) and push the
    rows into their owners' recv banks (this rank's column block), then the owners' flags."""
    n, d, rank, cs, hkv_r, S, qs = c["num_ranks"], c["head_dim"], c["rank"], c["seq_chunk"], c["hkv_r"], c["S"], c["q_split"]
    row_elems = hkv_r * d
    row_vecs = row_elems // VEC_BF16
    slot_elems = S * row_elems
    K_kv = n * row_elems
    per_iter = THREADS * 2
    for h in range_constexpr(BAND // BM):
        pos0 = band * fx.Int32(BAND) + fx.Int32(h * BM)
        owner, t0 = _owner_row(pos0, cs, n)
        owner_heap = ld(peers_base, owner, dtype=fx.T.i64)
        for x in range_constexpr(2):
            ws_res = (wsk_res, wsv_res)[x]
            recv_res = create_buffer_resource_from_addr(
                owner_heap + recv_offset + fx.Int64(c["recv_kv_at"][x]), num_records_bytes=c["recv_kv_bytes"]
            )
            for it in range(BM * row_vecs // per_iter):
                for j in range_constexpr(2):
                    v = fx.Int32(it * per_iter + j * THREADS) + thread_index
                    r = v // fx.Int32(row_vecs)
                    col = v % fx.Int32(row_vecs) * fx.Int32(VEC_BF16)
                    src = (pos0 + r) * fx.Int32(row_elems) + col
                    parts = [
                        buffer_load(ws_res, src + fx.Int32(s * slot_elems), vec_width=VEC_BF16, dtype=_BF16)
                        for s in range_constexpr(qs)
                    ]
                    acc = Vec(parts[0]).to(fx.Float32)
                    for s in range_constexpr(1, qs):
                        acc = acc + Vec(parts[s]).to(fx.Float32)
                    dst = (t0 + r) * fx.Int32(K_kv) + fx.Int32(rank * row_elems) + col
                    buffer_store(acc.to(_BF16).ir_value(), recv_res, dst, cache_modifier=SYS_NT)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        for h in range_constexpr(BAND // BM):
            owner, t0 = _owner_row(band * fx.Int32(BAND) + fx.Int32(h * BM), cs, n)
            owner_heap = ld(peers_base, owner, dtype=fx.T.i64)
            slot = (fx.Int32(c["g_tiles"]) + t0 // fx.Int32(BM)) * fx.Int32(n) + fx.Int32(rank)
            st(owner_heap + fx.Int64(REGION_BWD_RECV * FLAG_REGION_BYTES), slot, epoch, scope="sys", order="relaxed")


@ASTRewriter.transform
def _finish_item(thread_index, band, claim_slot, count_base, band_base, peers_base, recv_offset, wsk_res, wsv_res, epoch, **c):
    """Count the item on its band; the band's last item folds and pushes its dK / dV and marks it done."""
    ipb = c["items_per_band"]
    # this item's dK / dV slot stores and dQ atomics retire before its count releases them
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        # release only: an agent-scope acquire invalidates the XCD's L2 under the other resident items
        old = atomic_add(count_base, band, fx.Int64(1), scope="agent", order="release")
        st_lds(claim_slot, 1, arith.select(old == fx.Int64(ipb - 1), fx.Int32(1), fx.Int32(0)))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    last = _readfirstlane_i32(ld_lds(claim_slot, 1))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if last == fx.Int32(1):
        memory_fence(order="acquire", scope="agent")
        if thread_index == fx.Int32(0):
            # every item of the band has counted: reset for the next call
            st(count_base, band, fx.Int64(0))
        _push_dkdv(thread_index, band, peers_base, recv_offset, wsk_res, wsv_res, epoch, **c)
        if thread_index == fx.Int32(0):
            st(band_base, fx.Int32(BAND_DONE_AT) + band, epoch, scope="agent", order="release")


@ASTRewriter.transform
def _push_dq(thread_index, j, band_base, peers_base, recv_offset, img_res, epoch, **c):
    """Un-permute and scale q block ``j``'s rows of the dQ image (the stock dqa16 map; bands ``0..j``
    done) into their owners' recv banks (this rank's column block), then the owners' flags."""
    n, d, rank, cs, hq_r = c["num_ranks"], c["head_dim"], c["rank"], c["seq_chunk"], c["hq_r"]
    npair = d // 32
    K_q = n * hq_r * d
    # wave-uniform, so the owner's address and descriptor stay scalar
    wave = _readfirstlane_i32(thread_index // fx.Int32(WARP))
    lane = thread_index % fx.Int32(WARP)
    lane16 = lane & fx.Int32(15)
    w1 = (lane >> fx.Int32(5)) & fx.Int32(1)
    kgh = (lane >> fx.Int32(4)) & fx.Int32(1)
    q_in = (
        ((lane16 >> fx.Int32(2)) & fx.Int32(1)) * fx.Int32(16)
        + (lane16 >> fx.Int32(3)) * fx.Int32(4)
        + (lane16 & fx.Int32(3))
    )
    sm_vec = Vec.filled(8, c["dq_scale"], fx.Float32)
    for it in range(BAND // 16 * hq_r // (THREADS // WARP)):
        w = fx.Int32(it * (THREADS // WARP)) + wave
        qt = j * fx.Int32(BAND // 16) + w // fx.Int32(hq_r)
        qh = w % fx.Int32(hq_r)
        ibase = (
            (qt * fx.Int32(hq_r) + qh) * fx.Int32(npair * 512) + w1 * fx.Int32(256) + kgh * fx.Int32(64) + lane16 * fx.Int32(2)
        )
        parts = [
            [
                buffer_load(
                    img_res,
                    ibase + fx.Int32(p * 512 + w0 * 128 + kgl * 32),
                    vec_width=2,
                    dtype=_BF16,
                    cache_modifier=SYS_NT,
                )
                for kgl in range_constexpr(2)
                for w0 in range_constexpr(2)
            ]
            for p in range_constexpr(npair)
        ]
        # the tile pair (qt, qt ^ 1) holds one 32-row group, inside one chunk (cs % BM == 0)
        pos = (qt >> fx.Int32(1)) * fx.Int32(32)
        owner, t = _owner_row(pos, cs, n)
        owner_heap = ld(peers_base, owner, dtype=fx.T.i64)
        recv_res = create_buffer_resource_from_addr(owner_heap + recv_offset, num_records_bytes=c["recv_q_bytes"])
        row = t + q_in + (qt & fx.Int32(1)) * fx.Int32(8)
        obase = row * fx.Int32(K_q) + fx.Int32(rank * hq_r * d) + qh * fx.Int32(d) + w1 * fx.Int32(16) + kgh * fx.Int32(8)
        for p in range_constexpr(npair):
            lo = Vec(parts[p][0]).shuffle(Vec(parts[p][1]), [0, 1, 2, 3])
            hi = Vec(parts[p][2]).shuffle(Vec(parts[p][3]), [0, 1, 2, 3])
            v8 = lo.shuffle(hi, [0, 1, 2, 3, 4, 5, 6, 7])
            buffer_store(
                (v8.to(fx.Float32) * sm_vec).to(_BF16).ir_value(),
                recv_res,
                obase + fx.Int32(p * 32),
                cache_modifier=SYS_NT,
            )
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    # the unit's image rows are one contiguous span: zero it for the next call's atomics
    span = BAND * hq_r * d
    zero = Vec.filled(VEC_BF16, 0.0, _BF16).ir_value()
    for it in range_constexpr(span // VEC_BF16 // THREADS):
        buffer_store(zero, img_res, j * fx.Int32(span) + (fx.Int32(it * THREADS) + thread_index) * fx.Int32(VEC_BF16))
    if thread_index == fx.Int32(0):
        for h in range_constexpr(BAND // BM):
            owner, t0 = _owner_row(j * fx.Int32(BAND) + fx.Int32(h * BM), cs, n)
            owner_heap = ld(peers_base, owner, dtype=fx.T.i64)
            slot = t0 // fx.Int32(BM) * fx.Int32(n) + fx.Int32(rank)
            st(owner_heap + fx.Int64(REGION_BWD_RECV * FLAG_REGION_BYTES), slot, epoch, scope="sys", order="relaxed")


@ASTRewriter.transform
def _gather_unit(thread_index, g, src, flag_base, recv_addr, out_res, epoch, **c):
    """``src``'s column blocks of recv tile ``g`` (dq, dk, dv) into the outputs once ``src`` pushed them."""
    n, d = c["num_ranks"], c["head_dim"]
    if thread_index == fx.Int32(0):
        flag_idxs = [g * fx.Int32(n) + src, (fx.Int32(c["g_tiles"]) + g) * fx.Int32(n) + src]
        spin_flags(flag_base, flag_idxs, epoch, c["rank"], src, "a2a_attn_bwd gather")
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    # recv rows are rewritten by peers every 2 epochs: drop what this CU / XCD still caches of them
    memory_fence(order="acquire", scope="sys")
    row0 = g * fx.Int32(BM)
    for x in range_constexpr(3):
        blk_vecs = (c["hq_r"], c["hkv_r"], c["hkv_r"])[x] * d // VEC_BF16
        row_vecs = n * blk_vecs
        recv_res = create_buffer_resource_from_addr(
            recv_addr + fx.Int64(c["recv_at"][x]), num_records_bytes=c["recv_x_bytes"][x]
        )
        col0 = src * fx.Int32(blk_vecs)
        per_iter = THREADS * _BATCH

        def word(v, blk_vecs=blk_vecs, row_vecs=row_vecs, col0=col0):
            r = v // fx.Int32(blk_vecs)
            return ((row0 + r) * fx.Int32(row_vecs) + col0 + v % fx.Int32(blk_vecs)) * fx.Int32(4)

        for it in range(BM * blk_vecs // per_iter):
            v0 = fx.Int32(it * per_iter) + thread_index
            words = [word(v0 + fx.Int32(k * THREADS)) for k in range_constexpr(_BATCH)]
            vals = [buffer_load(recv_res, w, vec_width=4, dtype=fx.T.i32, cache_modifier=SYS_NT) for w in words]
            for w, val in zip(words, vals):
                buffer_store(val, out_res[x], w)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()


def _stashed(stash, i):
    """Stash entry ``i``, read where it is used: a base or descriptor held across the stock body raises
    its SGPR pressure, and the spills land in its main loop. An LDS load is not hoisted past the body."""
    v = fx.arith.ArithValue(ld(stash, fx.Int32(i), scope="workgroup", space=3, dtype=fx.T.i64), signed=True)
    lo = _readfirstlane_i32(cast(v, fx.T.i32))
    hi = _readfirstlane_i32(cast(v >> fx.Int64(32), fx.T.i32))
    return (_i64(hi) << fx.Int64(32)) | (_i64(lo) & fx.Int64(0xFFFFFFFF))


def _region(stash, region):
    return _stashed(stash, _S_MY) + fx.Int64(region * FLAG_REGION_BYTES)


def _recv_offset(epoch, c):
    """This epoch's recv rows in the heap (behind the flags and the dO staging rows of its bank)."""
    return fx.Int64(FLAG_BYTES + c["stage_bytes"]) + (epoch & fx.Int64(1)) * fx.Int64(c["bank_bytes"])


def _ws_res(stash, i, c):
    return create_buffer_resource_from_addr(
        _stashed(stash, i), num_records_bytes=c["q_split"] * c["S"] * c["hkv_r"] * c["head_dim"] * 2
    )


@ASTRewriter.transform
def _trace(stash, at, thread_index):
    tb = _stashed(stash, _S_TRACE)
    if thread_index == fx.Int32(0):
        st(tb, at, read_clock())


@ASTRewriter.transform
def _work_loop(thread_index, claim_slot, stash, run_item, **c):
    """Claim items (the top ``num_early`` bands from the top down, then band 0 up), then dQ push units in
    sequence order, then gather units ``(m-tile, src)``. Every base comes from ``stash`` at its use."""
    nb, ipb, ni, early, qs, n = (
        c["num_bands"],
        c["items_per_band"],
        c["num_items"],
        c["num_early"],
        c["q_split"],
        c["num_ranks"],
    )
    attn, comm = c["mode"] != "comm_only", c["mode"] != "attn_only"
    num_claims = ni + ((nb + c["g_tiles"] * n) if comm else 0)
    trace, trace_at = c["trace"], c["trace_item_at"]
    # items, then (a second loop: what the dQ push / gather code holds stays off the body's registers)
    # the comm units; claims are ordered, so the unit that ends the first loop starts the second
    unit = claim_unit(opaque_tid(thread_index), _stashed(stash, _S_STATE), claim_slot, counter=STATE_TILE)
    while unit < fx.Int32(ni):
        k = unit // fx.Int32(ipb)
        band = arith.select(k < fx.Int32(early), fx.Int32(nb - 1) - k, k - fx.Int32(early))
        sub = unit % fx.Int32(ipb)
        if const_expr(trace):
            _trace(stash, fx.Int32(trace_at) + unit * fx.Int32(3), thread_index)
            _trace(stash, fx.Int32(trace_at) + unit * fx.Int32(3) + fx.Int32(1), thread_index)
        if const_expr(attn):
            run_item(band, sub % fx.Int32(qs), sub // fx.Int32(qs))
        if const_expr(comm):
            epoch = _stashed(stash, _S_EPOCH)
            _finish_item(
                opaque_tid(thread_index),
                band,
                claim_slot,
                _stashed(stash, _S_COUNT),
                _region(stash, REGION_BWD_BAND),
                _stashed(stash, _S_PEERS),
                _recv_offset(epoch, c),
                _ws_res(stash, _S_WSK, c),
                _ws_res(stash, _S_WSV, c),
                epoch,
                **c,
            )
        if const_expr(trace):
            _trace(stash, fx.Int32(trace_at) + unit * fx.Int32(3) + fx.Int32(2), thread_index)
        # the next unit's LDS writes wait for this one's last reads
        fx.gpu.barrier()
        unit = claim_unit(opaque_tid(thread_index), _stashed(stash, _S_STATE), claim_slot, counter=STATE_TILE)
    if const_expr(comm):
        while unit < fx.Int32(num_claims):
            jj = unit - fx.Int32(ni)
            epoch = _stashed(stash, _S_EPOCH)
            if const_expr(trace):
                # the gather starts follow the dQ push ones: one stamp serves both
                _trace(stash, fx.Int32(c["trace_dq_at"]) + jj * fx.Int32(2), thread_index)
            if jj < fx.Int32(nb):
                _wait_bands(opaque_tid(thread_index), _region(stash, REGION_BWD_BAND), jj, epoch, **c)
                if const_expr(trace):
                    _trace(stash, fx.Int32(c["trace_dqw_at"]) + jj, thread_index)
                _push_dq(
                    opaque_tid(thread_index),
                    jj,
                    _region(stash, REGION_BWD_BAND),
                    _stashed(stash, _S_PEERS),
                    _recv_offset(epoch, c),
                    create_buffer_resource_from_addr(
                        _stashed(stash, _S_IMG), num_records_bytes=c["S"] * c["hq_r"] * c["head_dim"] * 2
                    ),
                    epoch,
                    **c,
                )
                if const_expr(trace):
                    _trace(stash, fx.Int32(c["trace_dq_at"]) + jj * fx.Int32(2) + fx.Int32(1), thread_index)
            else:
                gg = jj - fx.Int32(nb)
                g = gg // fx.Int32(n)
                src = gg % fx.Int32(n)
                _gather_unit(
                    opaque_tid(thread_index),
                    g,
                    src,
                    _region(stash, REGION_BWD_RECV),
                    _stashed(stash, _S_MY) + _recv_offset(epoch, c),
                    [
                        create_buffer_resource_from_addr(_stashed(stash, _S_DQ + x), num_records_bytes=c["recv_x_bytes"][x])
                        for x in range_constexpr(3)
                    ],
                    epoch,
                    **c,
                )
                if const_expr(trace):
                    _trace(stash, fx.Int32(c["trace_gather_at"]) + gg * fx.Int32(2) + fx.Int32(1), thread_index)
            unit = claim_unit(opaque_tid(thread_index), _stashed(stash, _S_STATE), claim_slot, counter=STATE_TILE)


TRACE_MAX_BLOCKS = 1024


def trace_layout(M, num_ranks, hq_r, hkv_r, head_dim):
    """Offsets of the debug timeline (s_memrealtime, 100 MHz): pull unit ends ``[m_tiles * n]``, then
    ``[claim, ready, end]`` per item, ``[start, end]`` per dQ push unit ``[num_bands]`` and per gather unit
    ``[g_tiles * n]``, stage piece ends ``[m_tiles * STAGE_SPLIT * n]``, block starts
    ``[TRACE_MAX_BLOCKS]``, the dQ push units' band-wait ends ``[num_bands]``; and the length."""
    S = M * num_ranks
    q_split, _ = _plan(S, hq_r, hkv_r, head_dim)
    nb = S // BAND
    item_at = M // BM * num_ranks
    dq_at = item_at + 3 * nb * q_split * hkv_r
    gather_at = dq_at + 2 * nb
    stage_at = gather_at + 2 * (M // BM * num_ranks)
    start_at = stage_at + M // BM * STAGE_SPLIT * num_ranks
    dqw_at = start_at + TRACE_MAX_BLOCKS
    return item_at, dq_at, gather_at, stage_at, start_at, dqw_at, dqw_at + nb


@functools.lru_cache(maxsize=32)
def _make_kernel(
    M, rank, num_ranks, hq_r, hkv_r, head_dim, seq_chunk, num_pull, num_blocks, num_early, num_prefix, bank_bytes, trace, mode
):
    n, d = num_ranks, head_dim
    S = M * n
    cs = seq_chunk if seq_chunk else M
    assert M % cs == 0 and cs % BM == 0 and S % BAND == 0, f"need M % seq_chunk == 0, seq_chunk % {BM} == 0"
    assert d in (64, 128)
    q_split, dkdv_l = _plan(S, hq_r, hkv_r, d)
    emb = dkdv_l.embed
    assert emb["block_size"] == THREADS and emb["block_kv"] == BAND and emb["q_split"] == q_split
    m_tiles = M // BM
    g_tiles = M // BM
    nb = S // BAND
    ipb = q_split * hkv_r
    num_items = nb * ipb
    stage_bytes = M * n * hq_r * d * 2
    recv_x_bytes = (M * n * hq_r * d * 2, M * n * hkv_r * d * 2, M * n * hkv_r * d * 2)
    recv_at = (0, recv_x_bytes[0], recv_x_bytes[0] + recv_x_bytes[1])
    assert stage_bytes + sum(recv_x_bytes) <= bank_bytes
    assert q_split * S * hkv_r * d * 2 < 2**31 and S * hq_r * d * 2 < 2**31, "i32 byte offsets overflow"
    assert m_tiles * STAGE_SPLIT * n <= FLAGS_PER_REGION and 2 * g_tiles * n <= FLAGS_PER_REGION
    assert nb <= BAND_DONE_AT and BM * hq_r % THREADS == 0
    assert (BM * hq_r * d // VEC_BF16) % (THREADS * _BATCH) == 0 and (BM * hkv_r * d // VEC_BF16) % (THREADS * _BATCH) == 0
    grid = max(num_blocks, num_pull)
    trace_item_at, trace_dq_at, trace_gather_at, trace_stage_at, trace_start_at, trace_dqw_at, _ = trace_layout(
        M, n, hq_r, hkv_r, d
    )
    assert not trace or grid <= TRACE_MAX_BLOCKS
    body = emb["body"]
    assert emb["q_ready"], "the dO readiness hook needs the body's q-block prefetch"
    num_early = min(num_early, nb)
    if num_prefix < 0:
        # the first wave's bands below the top ones start on their diagonal q blocks
        num_prefix = _ceildiv(grid, ipb) - num_early
    prefix_rows = min(max(num_prefix, 0), nb - num_early) * BAND
    prefix_tiles = prefix_rows // BM
    # every source's local rows of the sequence rows [0, prefix_rows)
    prefix_m_tiles = min(_ceildiv(prefix_rows, cs * n) * cs // BM, m_tiles)

    @fx.struct
    class SharedStorage:
        claim: fx.Array[fx.Int32, 4, 16]
        stash: fx.Array[fx.Int32, 2 * _STASH_LEN + 4, 16]

    c = dict(
        M=M,
        S=S,
        rank=rank,
        num_ranks=n,
        head_dim=d,
        hq_r=hq_r,
        hkv_r=hkv_r,
        seq_chunk=cs,
        m_tiles=m_tiles,
        g_tiles=g_tiles,
        stage_bytes=stage_bytes,
        recv_x_bytes=recv_x_bytes,
        recv_at=recv_at,
        recv_q_bytes=recv_x_bytes[0],
        recv_kv_at=recv_at[1:],
        recv_kv_bytes=recv_x_bytes[1],
        num_bands=nb,
        q_split=q_split,
        items_per_band=ipb,
        num_items=num_items,
        num_early=num_early,
        prefix_tiles=prefix_tiles,
        prefix_m_tiles=prefix_m_tiles,
        dq_scale=dkdv_l.dq_scale,
        mode=mode,
        trace=trace,
        bank_bytes=bank_bytes,
        trace_item_at=trace_item_at,
        trace_dq_at=trace_dq_at,
        trace_dqw_at=trace_dqw_at,
        trace_gather_at=trace_gather_at,
        trace_stage_at=trace_stage_at,
    )

    @flyc.kernel(name="ulysses_a2a_attn_bwd_kernel", known_block_size=[THREADS, 1, 1])
    def kernel(
        DO_IN: fx.Tensor,
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        LSE_IN: fx.Tensor,
        WSK: fx.Tensor,
        WSV: fx.Tensor,
        IMG: fx.Tensor,
        DQ: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        PEER_PTRS: fx.Tensor,
        STATE: fx.Tensor,
        COUNT: fx.Tensor,
        TRACE: fx.Tensor,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        trace_base = i64_base(TRACE) if trace else None
        if const_expr(trace):
            if thread_index == fx.Int32(0):
                st(trace_base, fx.Int32(trace_start_at) + block_index, read_clock())
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        claim_slot = fx.Int32(fx.ptrtoint(lds.claim.ptr))
        state_base = i64_base(STATE)
        peers_base = i64_base(PEER_PTRS)
        epoch = ld(state_base, fx.Int32(0), dtype=fx.T.i64)
        bank_offset = fx.Int64(FLAG_BYTES) + (epoch & fx.Int64(1)) * fx.Int64(bank_bytes)
        my_base = ld(peers_base, fx.Int32(rank), dtype=fx.T.i64)
        stash = fx.Int32(fx.ptrtoint(lds.stash.ptr))
        if thread_index == fx.Int32(0):
            for i, v in (
                (_S_PEERS, peers_base),
                (_S_MY, my_base),
                (_S_EPOCH, epoch),
                (_S_COUNT, i64_base(COUNT)),
                (_S_WSK, i64_base(WSK)),
                (_S_WSV, i64_base(WSV)),
                (_S_IMG, i64_base(IMG)),
                (_S_DQ, i64_base(DQ)),
                (_S_DQ + 1, i64_base(DK)),
                (_S_DQ + 2, i64_base(DV)),
                (_S_STATE, state_base),
                (_S_TRACE, trace_base if trace else fx.Int64(0)),
            ):
                st(stash, fx.Int32(i), v, scope="workgroup", space=3)
            st(stash, fx.Int32(_DONE_AT), fx.Int32(1 if mode == "attn_only" else 0), scope="workgroup", space=3)
        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()

        if const_expr(mode != "attn_only"):
            in_res = create_buffer_resource(DO_IN, num_records_bytes=stage_bytes)
            if block_index >= fx.Int32(0 if grid <= num_pull else num_pull):
                _stage_role(thread_index, state_base, claim_slot, my_base, bank_offset, in_res, epoch, trace_base, **c)
            if block_index < fx.Int32(num_pull):
                _pull_role(
                    thread_index,
                    state_base,
                    claim_slot,
                    peers_base,
                    my_base,
                    bank_offset,
                    create_buffer_resource(DO, num_records_bytes=S * hq_r * d * 2),
                    create_buffer_resource(O, num_records_bytes=S * hq_r * d * 2),
                    create_buffer_resource(DELTA, num_records_bytes=S * hq_r * 4),
                    [
                        create_buffer_resource(LSE_IN, num_records_bytes=S * hq_r * 4),
                        create_buffer_resource(LSE, num_records_bytes=S * hq_r * 4),
                    ],
                    epoch,
                    trace_base,
                    **c,
                )

        q_ready = None if mode == "attn_only" else functools.partial(_q_ready, stash=stash, **c)

        def run_item(band, split, kvh):
            body(
                Q,
                K,
                V,
                DO,
                LSE,
                DELTA,
                WSK,
                WSV,
                STATE,
                STATE,
                IMG,
                fx.Int32(S),
                fx.Int32(S),
                fx.Int32(S),
                ids=(fx.Index(band), fx.Index(kvh), fx.Index(split), fx.Index(0)),
                q_ready=q_ready,
            )

        _work_loop(thread_index, claim_slot, stash, run_item, **c)

        exit_and_bump_epoch(
            thread_index, _stashed(stash, _S_STATE), _stashed(stash, _S_EPOCH), grid, extra_counters=(STATE_ASSIST, STATE_PULLED)
        )

    return kernel, grid, emb["value_attrs"], emb["allocator"]


@flyc.jit
def _launch(
    DO_IN,
    Q,
    K,
    V,
    O,  # noqa: E741
    DO,
    LSE,
    DELTA,
    LSE_IN,
    WSK,
    WSV,
    IMG,
    DQ,
    DK,
    DV,
    PEER_PTRS,
    STATE,
    COUNT,
    TRACE,
    M: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    hq_r: fx.Constexpr[int],
    hkv_r: fx.Constexpr[int],
    head_dim: fx.Constexpr[int],
    seq_chunk: fx.Constexpr[int],
    num_pull: fx.Constexpr[int],
    num_blocks: fx.Constexpr[int],
    num_early: fx.Constexpr[int],
    num_prefix: fx.Constexpr[int],
    bank_bytes: fx.Constexpr[int],
    trace: fx.Constexpr[bool],
    mode: fx.Constexpr[str],
    stream: fx.Stream,
):
    kernel, grid, value_attrs, allocator = _make_kernel(
        M,
        rank,
        num_ranks,
        hq_r,
        hkv_r,
        head_dim,
        seq_chunk,
        num_pull,
        num_blocks,
        num_early,
        num_prefix,
        bank_bytes,
        trace,
        mode,
    )
    # the dkdv body's LDS lives in its builder's allocator
    allocator.finalized = False
    with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
        allocator.finalize()
    kernel(
        DO_IN, Q, K, V, O, DO, LSE, DELTA, LSE_IN, WSK, WSV, IMG, DQ, DK, DV, PEER_PTRS, STATE, COUNT, TRACE,
        value_attrs=value_attrs,
    ).launch(grid=(grid, 1, 1), block=(THREADS, 1, 1), stream=stream)


_BUFS: dict = {}
# buffer keys whose dQ image an attn_only call (no dQ push) left non-zero
_IMG_DIRTY: set = set()


def _buffers(S, hq_r, hkv_r, d, q_split, device):
    """dO and delta of this rank's heads, the dK / dV slots, the dQ image (zero between calls: its dQ push
    units zero it), the band item counts (zero between calls: a band's last item resets its count) and
    the scaled LSE; reused across calls."""
    key = (S, hq_r, hkv_r, d, q_split, device)
    bufs = _BUFS.get(key)
    if bufs is None:
        _BUFS.clear()
        bufs = (
            torch.empty(S, hq_r, d, device=device, dtype=torch.bfloat16),
            torch.empty(hq_r, S, device=device, dtype=torch.float32),
            torch.empty(q_split, S, hkv_r, d, device=device, dtype=torch.bfloat16),
            torch.empty(q_split, S, hkv_r, d, device=device, dtype=torch.bfloat16),
            torch.zeros(S * hq_r * d, device=device, dtype=torch.bfloat16),
            torch.zeros(S // BAND, device=device, dtype=torch.int64),
            torch.empty(hq_r * S, device=device, dtype=torch.float32),
        )
        _IMG_DIRTY.clear()
        _BUFS[key] = bufs
    return bufs


def ulysses_a2a_attn_bwd(
    dout,
    q,
    k,
    v,
    o,
    lse,
    dq,
    dk,
    dv,
    workspace,
    *,
    seq_chunk=0,
    num_pull=32,
    num_blocks=None,
    num_early=0,
    num_prefix=-1,
    trace=None,
    mode="full",
):
    """``dout`` ``[M, n * hq_r, d]``: dO for this rank's tokens with every head (batch 1); ``q`` / ``k`` /
    ``v`` / ``o`` ``[S, h_r, d]`` and the natural-log ``lse`` ``[S, hq_r]`` fp32 of this rank's heads over
    the sequence, as the fused forward leaves them. Writes ``dq`` ``[M, n * hq_r, d]`` and ``dk`` / ``dv``
    ``[M, n * hkv_r, d]`` for this rank's tokens. The bank must hold ``a2a_attn_bwd_bank_bytes``.

    ``mode`` (measurement only): ``"attn_only"`` runs the backward items on the dO / delta of the previous
    call (no exchange, no waits, no folds, pushes or gathers; LSE prescaled and the dQ image zeroed by
    separate launches); ``"comm_only"`` runs every step but the items' attention body."""
    n = workspace.num_ranks
    M, H, d = dout.shape[0], dout.shape[-2], dout.shape[-1]
    S, hq_r, hkv_r = M * n, H // n, k.shape[-2]
    assert dout.dtype == torch.bfloat16 and H % n == 0 and q.shape == (S, hq_r, d) and o.shape == q.shape
    assert all(t.is_contiguous() for t in (dout, q, k, v, o, lse, dq, dk, dv))
    assert dq.shape == dout.shape and dk.shape == (M, n * hkv_r, d) and dv.shape == dk.shape
    q_split, _ = _plan(S, hq_r, hkv_r, d)
    if num_blocks is None:
        num_blocks = torch.cuda.get_device_properties(dout.device).multi_processor_count
    stream = torch.cuda.current_stream()
    key = (S, hq_r, hkv_r, d, q_split, dout.device)
    do_h, delta, ws_dk, ws_dv, img, counts, lsef = _buffers(*key)
    qf = _bwd._prescale_q(q, 1.0 / math.sqrt(d))
    if mode == "attn_only":
        # no pull units to write it
        lsef = _bwd._prescale_lse(lse.view(1, S, hq_r).permute(0, 2, 1), stream).reshape(-1)
    if mode == "attn_only" or key in _IMG_DIRTY:
        img.zero_()
    if mode == "attn_only":
        _IMG_DIRTY.add(key)
    else:
        _IMG_DIRTY.discard(key)
    with CompilationContext.compile_hints(_plan(S, hq_r, hkv_r, d)[1].embed["hints"]):
        run_launcher(
            _launch,
            dout,
            qf,
            k,
            v,
            o,
            do_h,
            lsef,
            delta,
            lse,
            ws_dk,
            ws_dv,
            img,
            dq,
            dk,
            dv,
            workspace.peer_ptrs,
            workspace.state,
            counts,
            workspace.state if trace is None else trace,
            M,
            workspace.rank,
            n,
            hq_r,
            hkv_r,
            d,
            seq_chunk,
            num_pull,
            num_blocks,
            num_early,
            num_prefix,
            workspace.bank_bytes,
            trace is not None,
            mode,
            stream,
        )
