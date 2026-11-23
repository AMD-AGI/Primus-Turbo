###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses all-to-all + causal attention + all-to-all in one kernel (single node, symmetric memory).

Every rank holds its ``M`` tokens with all heads (q / k / v ``[M, n * h_r, d]``, batch 1). One launch:

* every block but the pull blocks (all of them when the grid has no others) first stages this rank's q / k / v
  rows into its symmetric bank as one column block per destination rank (``[q heads | k heads | v heads]`` of that
  rank), raising one flag per (row piece, destination);
* pull blocks (the first ``num_pull``) meanwhile claim pull units ``(m_tile, src)`` m-major, copy ``src``'s rows of this
  rank's column block into q / k / v over the full sequence (``[S, h_r, d]``, sequence order) and raise
  a local ready flag per unit, and then join the workers;
* every block runs as a persistent attention worker that claims ``(head group, q block)`` items, waits for the ready
  flags of every unit holding rows of its causal KV prefix, runs the dual-wave tile, and pushes the O
  tile into the recv bank of the rank owning those tokens, raising that rank's flag
  ``(m-tile, src, head group)``;
* once every attention item is claimed, the workers claim gather units ``(m-tile, src)``: a unit waits
  for the flags of ``src``'s head groups, then copies ``src``'s column block of the m-tile's recv rows
  into ``out`` ``[M, n * hq_r, d]``.

Every pull unit is claimed by a running block before that block turns worker, and every attention
item is claimed by a running worker before any worker waits on a gather flag, so with the pull blocks
dispatched first no wait can deadlock whatever the grid size.

Sequence split: with ``seq_chunk`` cs, chunk ``c`` of the global sequence lives on rank ``c % n`` (local
token ``t`` of rank ``r`` is global ``((t // cs) * n + r) * cs + t % cs``); ``cs = M`` is the contiguous
split. The pull units land m-major, i.e. in sequence order under a striped split, so the KV prefix of
the early q blocks is complete while the exchange still runs: workers take ``num_ascending`` q blocks
in sequence order, then the rest longest first.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr

from primus_turbo.flydsl.attention.flash_attn_fwd_stock import (
    DUALWAVE_SWP_COMPILE_HINTS,
    make_dualwave_swp_tile,
)
from primus_turbo.flydsl.attention.gfx950_mla.attn_helper_mla import DualwaveKernelContext as MlaKernelContext
from primus_turbo.flydsl.attention.gfx950_mla.flash_attn_fwd_mla import make_mla_tile, make_mla_traits
from primus_turbo.flydsl.deep_ep.common import _asm
from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.ulysses.a2a_gemm_kernel import _copy_rows
from primus_turbo.flydsl.ulysses.common import (
    BLOCK_THREADS,
    SPIN_TIMEOUT,
    STATE_ASSIST,
    STATE_CLAIM,
    STATE_TILE,
    SYS_NT,
    VEC_BF16,
    VEC_I32,
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
from primus_turbo.flydsl.utils.attn_helper_stock import (
    DualwaveKernelContext,
    _make_dualwave_swp_traits,
    dtype_to_elem_type,
)
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)
from primus_turbo.flydsl.utils.prims import (
    ld,
    memory_fence,
    read_clock,
    spin_timed_out,
    st,
)

REGION_FWD_STAGE = 0
REGION_FWD_READY = 1
REGION_FWD_O = 2

# rows of one pull unit
BM = 128
# stage pieces per pull unit: many small pieces in flight finish the first m-tiles early, so the pulls
# start while the later ones are still staged
STAGE_SPLIT = 4
# 16 B vectors per thread in flight in the gather copy
_GATHER_BATCH = 8


MLA_DIMS = (192, 128)


def is_mla(d, d_v):
    return (d, d_v) == MLA_DIMS


def make_traits(hq, hkv, d, d_v=None, sm_scale=None):
    """The production SBHD causal forward configuration: ``attention_flydsl_impl._fwd_module``, or for
    the MLA head dims (q / k 192, v 128) ``gfx950_mla.ops.mla_fwd_sbhd``'s persistent build.
    """
    if is_mla(d, d if d_v is None else d_v):
        return make_mla_traits(
            num_heads=hq,
            head_dim=d,
            head_dim_v=d_v,
            causal=True,
            dtype_str="bf16",
            num_kv_heads=hkv,
            emit_lse=True,
            sbhd=True,
            block_m=256,
            waves_per_eu=1,
            dualwave_swp_enable_stagger=False,
            k_head=1,
            rows_per_wave=64,
            fold_barriers=True,
            persistent=1,
            sm_scale=sm_scale,
        )
    assert d_v in (None, d) and sm_scale is None
    return _make_dualwave_swp_traits(
        hq,
        hkv,
        d,
        causal=True,
        dtype_str="bf16",
        waves_per_eu=2,
        dualwave_swp_enable_stagger=False,
        emit_lse=True,
        block_m=64 if d == 64 else 128,
        sbhd=True,
    )


def a2a_attn_fwd_bank_bytes(M, num_ranks, hq_r, hkv_r, head_dim, head_dim_v=None):
    """Bank bytes: the q / k / v staging rows, then the O recv rows (O and v ``head_dim_v`` wide)."""
    d_v = head_dim if head_dim_v is None else head_dim_v
    stage = M * num_ranks * ((hq_r + hkv_r) * head_dim + hkv_r * d_v) * 2
    return stage + M * num_ranks * hq_r * d_v * 2


def _global_row(t, src, cs, n):
    """Sequence position of local token ``t`` of rank ``src``."""
    return (t // fx.Int32(cs) * fx.Int32(n) + src) * fx.Int32(cs) + t % fx.Int32(cs)


@ASTRewriter.transform
def _stage_unit(thread_index, m_sub, dst, in_res, stage_res, **c):
    """Rows ``[m_sub * rows, +rows)`` (``rows = BM // STAGE_SPLIT``) of ``dst``'s heads of this rank's tensors
    into ``dst``'s column block of the staging bank, retired on return."""
    M, n, threads = c["M"], c["num_ranks"], c.get("threads", BLOCK_THREADS)
    widths, col_offs = c["widths"], c["col_offs"]
    row_vecs_out = n * c["N_r"] // VEC_BF16
    blk_vecs = c["N_r"] // VEC_BF16
    rows = BM // STAGE_SPLIT
    t0 = m_sub * fx.Int32(rows)
    for x in range_constexpr(len(widths)):
        h_vecs = widths[x] // VEC_BF16
        row_vecs_in = n * h_vecs
        off = col_offs[x] // VEC_BF16

        def coords(v, h_vecs=h_vecs, row_vecs_in=row_vecs_in, off=off):
            r = v // fx.Int32(h_vecs)
            col = v % fx.Int32(h_vecs)
            t = t0 + r
            src_word = (t * fx.Int32(row_vecs_in) + dst * fx.Int32(h_vecs) + col) * fx.Int32(VEC_I32)
            dst_word = (
                t * fx.Int32(row_vecs_out) + dst * fx.Int32(blk_vecs) + fx.Int32(off) + col
            ) * fx.Int32(VEC_I32)
            return src_word, dst_word, (r < fx.Int32(rows)) & (t < fx.Int32(M))

        _copy_rows(thread_index, rows * h_vecs, coords, in_res[x], stage_res, 0, SYS_NT, threads=threads)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()


@ASTRewriter.transform
def _pull_unit(thread_index, src, m_tile, src_heap, bank_offset, out_res, epoch, **c):
    """``src``'s staged rows ``[m_tile * BM, +BM)`` of this rank's column block into q / k / v."""
    M, n, rank, cs = c["M"], c["num_ranks"], c["rank"], c["seq_chunk"]
    threads = c.get("threads", BLOCK_THREADS)
    widths, col_offs = c["widths"], c["col_offs"]
    row_vecs = n * c["N_r"] // VEC_BF16
    blk_vecs = c["N_r"] // VEC_BF16
    if thread_index == fx.Int32(0):
        flag_base = src_heap + fx.Int64(c["stage_region"] * FLAG_REGION_BYTES)
        flag0 = m_tile * fx.Int32(STAGE_SPLIT * n) + fx.Int32(rank)
        flag_idxs = [flag0 + fx.Int32(i * n) for i in range(STAGE_SPLIT)]
        spin_flags(flag_base, flag_idxs, epoch, rank, src, "a2a_attn pull", report=False)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    stage_res = create_buffer_resource_from_addr(src_heap + bank_offset, num_records_bytes=c["stage_bytes"])
    t0 = m_tile * fx.Int32(BM)
    # a unit lies inside one sequence chunk (cs % BM == 0): its rows land contiguously
    g0 = _global_row(t0, src, cs, n)
    for x in range_constexpr(len(widths)):
        h_vecs = widths[x] // VEC_BF16
        off = rank * blk_vecs + col_offs[x] // VEC_BF16

        def coords(v, h_vecs=h_vecs, off=off):
            r = v // fx.Int32(h_vecs)
            col = v % fx.Int32(h_vecs)
            src_word = ((t0 + r) * fx.Int32(row_vecs) + fx.Int32(off) + col) * fx.Int32(VEC_I32)
            dst_word = ((g0 + r) * fx.Int32(h_vecs) + col) * fx.Int32(VEC_I32)
            return src_word, dst_word, (r < fx.Int32(BM)) & (t0 + r < fx.Int32(M))

        _copy_rows(thread_index, BM * h_vecs, coords, stage_res, out_res[x], SYS_NT, 0, threads=threads)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()


@ASTRewriter.transform
def _stage_role(thread_index, state_base, claim_slot, my_base, bank_offset, in_res, epoch, trace_base, **c):
    """Stage pieces ``(m_tile, dst)`` m-major until none are left."""
    n, rank, m_tiles = c["num_ranks"], c["rank"], c["m_tiles"]
    stage_res = create_buffer_resource_from_addr(my_base + bank_offset, num_records_bytes=c["stage_bytes"])
    stage_flags = my_base + fx.Int64(c["stage_region"] * FLAG_REGION_BYTES)
    more_stage = fx.Int32(1)
    num_pieces = m_tiles * STAGE_SPLIT * n
    while more_stage == fx.Int32(1):
        piece = claim_unit(opaque_tid(thread_index), state_base, claim_slot, STATE_ASSIST)
        if piece < fx.Int32(num_pieces):
            m_sub = piece // fx.Int32(n)
            # the pullers of m-tile i start with src = dst - i: stage the other ranks' blocks first
            dst = (fx.Int32(rank + 1) + piece % fx.Int32(n)) % fx.Int32(n)
            _stage_unit(thread_index, m_sub, dst, in_res, stage_res, **c)
            if thread_index == fx.Int32(0):
                # the staging stores are system scope (SYS_NT) and retired above
                st(stage_flags, m_sub * fx.Int32(n) + dst, epoch, scope="sys", order="relaxed")
                if const_expr(trace_base is not None):
                    st(
                        trace_base,
                        fx.Int32(c.get("trace_stage_at", 0)) + m_sub * fx.Int32(n) + dst,
                        read_clock(),
                    )
        more_stage = arith.select(piece < fx.Int32(num_pieces), fx.Int32(1), fx.Int32(0))


@ASTRewriter.transform
def _pull_role(
    thread_index,
    state_base,
    claim_slot,
    peers_base,
    my_base,
    bank_offset,
    out_res,
    epoch,
    trace_base,
    **c,
):
    """Pull units ``(m_tile, src = rank + i)`` m-major until none are left."""
    n, rank, m_tiles = c["num_ranks"], c["rank"], c["m_tiles"]
    ready_base = my_base + fx.Int64(c["ready_region"] * FLAG_REGION_BYTES)
    num_units = m_tiles * n
    more_pull = fx.Int32(1)
    while more_pull == fx.Int32(1):
        unit = claim_unit(opaque_tid(thread_index), state_base, claim_slot, STATE_CLAIM)
        if unit < fx.Int32(num_units):
            m_tile = unit // fx.Int32(n)
            src = (fx.Int32(rank) + unit % fx.Int32(n)) % fx.Int32(n)
            src_heap = ld(peers_base, src, dtype=fx.T.i64)
            _pull_unit(thread_index, src, m_tile, src_heap, bank_offset, out_res, epoch, **c)
            if thread_index == fx.Int32(0):
                st(ready_base, m_tile * fx.Int32(n) + src, epoch, scope="agent", order="release")
                if const_expr(trace_base is not None):
                    st(trace_base, m_tile * fx.Int32(n) + src, read_clock())
        more_pull = arith.select(unit < fx.Int32(num_units), fx.Int32(1), fx.Int32(0))


@ASTRewriter.transform
def _wait_prefix(thread_index, ready_base, kv_end, epoch, **c):
    """Wait for the ready flag of every pull unit holding rows below ``kv_end``: each thread polls its
    own share of the units (relaxed), then the block rendezvous and acquires once."""
    n, cs, rank = c["num_ranks"], c["seq_chunk"], c["rank"]
    total = c["m_tiles"] * n
    idx = thread_index
    spin_start = read_clock()
    while idx < fx.Int32(total):
        g0 = _global_row(idx // fx.Int32(n) * fx.Int32(BM), idx % fx.Int32(n), cs, n)
        need = g0 < kv_end
        flag = ld(ready_base, idx, scope="agent", dtype=fx.T.i64)
        while need & (flag < epoch):
            fx.rocdl.s_sleep(fx.Int32(1))
            if spin_timed_out(spin_start, SPIN_TIMEOUT):
                fx.printf(f"[a2a_attn rank={rank}] ready flag timeout: idx={{}} flag={{}}\n", idx, flag)
                _asm("s_trap 2")
            flag = ld(ready_base, idx, scope="agent", dtype=fx.T.i64)
        idx = idx + fx.Int32(c["threads"])
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    memory_fence(order="acquire", scope="agent")


@ASTRewriter.transform
def _push_o(thread_index, qb, h, peers_base, recv_offset, o_res, epoch, **c):
    """O rows of q block ``qb`` for work-item head group ``h`` into the owner's recv bank (this rank's
    column block, head order), then the owner's flag ``(m-tile, rank, h)``."""
    bm, cs, n, rank, hg = (c["block_m"], c["seq_chunk"], c["num_ranks"], c["rank"], c["head_groups"])
    d, hq_r, hkv_r, qpw, K_o = (c["d_o"], c["hq_r"], c["hkv_r"], c["heads_per_wg"], c["K_o"])
    # O was just stored by this CU's waves: drop stale L1 lines before reading it back
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    memory_fence(order="acquire", scope="agent")
    pos0 = qb * fx.Int32(bm)
    chunk = pos0 // fx.Int32(cs)
    owner = chunk % fx.Int32(n)
    t0 = chunk // fx.Int32(n) * fx.Int32(cs) + pos0 % fx.Int32(cs)
    owner_heap = ld(peers_base, owner, dtype=fx.T.i64)
    recv_res = create_buffer_resource_from_addr(owner_heap + recv_offset, num_records_bytes=c["recv_bytes"])
    vecs_per_row = qpw * d // VEC_BF16
    # the q heads of work-item head group h (attn_helper: kv head h % hkv, sharer group h // hkv)
    head0 = (h % fx.Int32(hkv_r)) * fx.Int32(hq_r // hkv_r) + h // fx.Int32(hkv_r) * fx.Int32(qpw)
    src_col0 = head0 * fx.Int32(d // VEC_BF16)
    dst_col0 = fx.Int32(rank * hq_r * d // VEC_BF16) + src_col0

    def coords(v):
        r = v // fx.Int32(vecs_per_row)
        col = v % fx.Int32(vecs_per_row)
        src_word = ((pos0 + r) * fx.Int32(hq_r * d // VEC_BF16) + src_col0 + col) * fx.Int32(VEC_I32)
        dst_word = ((t0 + r) * fx.Int32(K_o // VEC_BF16) + dst_col0 + col) * fx.Int32(VEC_I32)
        return src_word, dst_word, r < fx.Int32(bm)

    _copy_rows(thread_index, bm * vecs_per_row, coords, o_res, recv_res, 0, SYS_NT, threads=c["threads"])
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        flag_base = owner_heap + fx.Int64(REGION_FWD_O * FLAG_REGION_BYTES)
        slot = t0 // fx.Int32(bm) * fx.Int32(n * hg) + fx.Int32(rank * hg) + h
        # the O stores are system scope (SYS_NT) and retired above; a release here would write back the
        # XCD's whole L2 for every item
        st(flag_base, slot, epoch, scope="sys", order="relaxed")


@ASTRewriter.transform
def _gather_unit(thread_index, g, src, flag_base, recv_addr, out_res, epoch, **c):
    """``src``'s column block of the recv rows of m-tile ``g`` into ``out`` once every head group of
    ``src`` has pushed it."""
    bm, K_o, hg = c["block_m"], c["K_o"], c["head_groups"]
    blk_vecs = c["hq_r"] * c["d_o"] // VEC_BF16
    if thread_index == fx.Int32(0):
        flag0 = g * fx.Int32(c["num_ranks"] * hg) + src * fx.Int32(hg)
        flag_idxs = [flag0 + fx.Int32(i) for i in range(hg)]
        spin_flags(flag_base, flag_idxs, epoch, c["rank"], src, "a2a_attn gather")
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    # recv rows are rewritten by peers every 2 epochs: drop what this CU / XCD still caches of them
    memory_fence(order="acquire", scope="sys")
    recv_res = create_buffer_resource_from_addr(recv_addr, num_records_bytes=c["recv_bytes"])
    num_vecs = bm * blk_vecs
    threads = c["threads"]
    per_iter = threads * _GATHER_BATCH
    row0 = g * fx.Int32(bm)
    col0 = src * fx.Int32(blk_vecs)

    def word(v):
        r = v // fx.Int32(blk_vecs)
        return ((row0 + r) * fx.Int32(K_o // VEC_BF16) + col0 + v % fx.Int32(blk_vecs)) * fx.Int32(VEC_I32)

    # a unit is many copy steps: a loop rather than _copy_rows' unrolled ones
    for it in range(num_vecs // per_iter):
        v0 = fx.Int32(it * per_iter) + thread_index
        words = [word(v0 + fx.Int32(j * threads)) for j in range_constexpr(_GATHER_BATCH)]
        vals = [
            buffer_load(recv_res, w, vec_width=VEC_I32, dtype=fx.T.i32, cache_modifier=SYS_NT) for w in words
        ]
        for w, val in zip(words, vals):
            buffer_store(val, out_res, w)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()


@ASTRewriter.transform
def _attn_loop(
    thread_index,
    state_base,
    claim_slot,
    my_base,
    peers_base,
    recv_offset,
    o_res,
    out_res,
    epoch,
    run_tile,
    trace_base,
    **c,
):
    """Claim ``(head group, q block)`` items (``num_ascending`` q blocks in sequence order, then the rest
    longest first), wait for each item's KV prefix, run its tile and push its O; then the gather units
    ``(m-tile, src)`` (the m-tiles of the ascending q blocks first, the rest from the end)."""
    nqb, hg, asc, bm, bn = c["num_q_blocks"], c["head_groups"], c["num_ascending"], c["block_m"], c["block_n"]
    m_tiles_o, o_asc = c["m_tiles_o"], c["o_ascending"]
    num_units = nqb * hg
    attn, comm = c["mode"] != "comm_only", c["mode"] != "attn_only"
    n = c["num_ranks"]
    num_claims = num_units + (m_tiles_o * n if comm else 0)
    ready_base = my_base + fx.Int64(REGION_FWD_READY * FLAG_REGION_BYTES)
    o_flags = my_base + fx.Int64(REGION_FWD_O * FLAG_REGION_BYTES)
    trace_at = c["trace_attn_at"]
    unit = claim_unit(thread_index, state_base, claim_slot, counter=STATE_TILE)
    while unit < fx.Int32(num_claims):
        if unit < fx.Int32(num_units):
            k = unit // fx.Int32(hg)
            qb = arith.select(k < fx.Int32(asc), k, fx.Int32(nqb - 1 + asc) - k)
            h = unit % fx.Int32(hg)
            if const_expr(trace_base is not None):
                if thread_index == fx.Int32(0):
                    st(trace_base, fx.Int32(trace_at) + unit * fx.Int32(3), read_clock())
            # the KV the tile reads, masked or not (attn_helper: max(4, even(ceil(q_end / BLOCK_N)))
            # tiles); masked columns still multiply V, so those rows must be the landed ones
            kv_tiles = (
                ((qb + fx.Int32(1)) * fx.Int32(bm) + fx.Int32(2 * bn - 1)) // fx.Int32(2 * bn) * fx.Int32(2)
            )
            kv_tiles = arith.select(kv_tiles < fx.Int32(4), fx.Int32(4), kv_tiles)
            if const_expr(comm):
                _wait_prefix(thread_index, ready_base, kv_tiles * fx.Int32(bn), epoch, **c)
            if const_expr(trace_base is not None):
                if thread_index == fx.Int32(0):
                    st(trace_base, fx.Int32(trace_at) + unit * fx.Int32(3) + fx.Int32(1), read_clock())
            if const_expr(attn):
                run_tile(h, qb)
            if const_expr(comm):
                _push_o(thread_index, qb, h, peers_base, recv_offset, o_res, epoch, **c)
            if const_expr(trace_base is not None):
                if thread_index == fx.Int32(0):
                    st(trace_base, fx.Int32(trace_at) + unit * fx.Int32(3) + fx.Int32(2), read_clock())
        else:
            j = unit - fx.Int32(num_units)
            k = j // fx.Int32(n)
            src = j % fx.Int32(n)
            g = arith.select(k < fx.Int32(o_asc), k, fx.Int32(m_tiles_o - 1 + o_asc) - k)
            _gather_unit(thread_index, g, src, o_flags, my_base + recv_offset, out_res, epoch, **c)
            if const_expr(trace_base is not None):
                if thread_index == fx.Int32(0):
                    st(trace_base, fx.Int32(c["trace_gather_at"]) + g * fx.Int32(n) + src, read_clock())
        unit = claim_unit(thread_index, state_base, claim_slot, counter=STATE_TILE)


TRACE_MAX_BLOCKS = 1024


def trace_layout(M, num_ranks, hq_r, hkv_r, head_dim, head_dim_v=None):
    """Offsets of the debug timeline (s_memrealtime, 100 MHz): pull unit ends ``[m_tiles * n]``, then
    ``[claim, ready, end]`` per attention item, the gather unit ends ``[m_tiles_o * n]``, the stage piece ends
    ``[m_tiles * STAGE_SPLIT * n]`` and the block start times ``[TRACE_MAX_BLOCKS]``; and the total length.
    """
    traits = make_traits(hq_r, hkv_r, head_dim, head_dim_v)
    bm = traits.BLOCK_M
    hg = hq_r // traits.Q_HEADS_PER_WG
    attn_at = M // BM * num_ranks
    gather_at = attn_at + 3 * (M * num_ranks // bm) * hg
    stage_at = gather_at + M // bm * num_ranks
    start_at = stage_at + M // BM * STAGE_SPLIT * num_ranks
    return attn_at, gather_at, stage_at, start_at, start_at + TRACE_MAX_BLOCKS


@functools.lru_cache(maxsize=32)
def _make_kernel(
    M,
    rank,
    num_ranks,
    hq_r,
    hkv_r,
    head_dim,
    head_dim_v,
    sm_scale,
    seq_chunk,
    num_pull,
    num_blocks,
    num_ascending,
    bank_bytes,
    trace,
    mode,
):
    n, d, d_v = num_ranks, head_dim, head_dim_v
    mla = is_mla(d, d_v)
    traits = make_traits(hq_r, hkv_r, d, d_v, sm_scale)
    # the MLA tile is one 4-wave head per CTA (a block_m 256 q block)
    threads = traits.BLOCK_SIZE
    assert threads == (256 if mla else BLOCK_THREADS), (
        f"the fused kernel runs {BLOCK_THREADS}-thread blocks; this head config builds {threads} "
        "(an even q-per-kv group is needed for the two-head CTA)"
    )
    attn_tile = make_mla_tile(traits) if mla else make_dualwave_swp_tile(traits)
    bm = traits.BLOCK_M
    cs = seq_chunk if seq_chunk else M
    # a q block lies inside one sequence chunk, so its O rows have one owner
    assert M % cs == 0 and cs % max(BM, bm) == 0, (
        f"need M % seq_chunk == 0 and seq_chunk % {max(BM, bm)} == 0"
    )
    S = M * n
    m_tiles = M // BM
    widths = (hq_r * d, hkv_r * d, hkv_r * d_v)
    N_r = sum(widths)
    stage_bytes = M * n * N_r * 2
    K_o = n * hq_r * d_v
    recv_bytes = M * K_o * 2
    assert stage_bytes + recv_bytes <= bank_bytes
    assert stage_bytes < 2**31 and recv_bytes < 2**31 and S * hq_r * d * 2 < 2**31, (
        "i32 byte offsets overflow"
    )
    hg = hq_r // traits.Q_HEADS_PER_WG
    nqb = S // bm
    m_tiles_o = M // bm
    assert m_tiles * STAGE_SPLIT * n <= FLAGS_PER_REGION and m_tiles_o * n * hg <= FLAGS_PER_REGION
    assert (bm * hq_r * d_v // VEC_BF16) % (threads * _GATHER_BATCH) == 0, (
        "gather unit must be whole copy steps"
    )
    num_ascending = min(num_ascending, nqb)
    grid = max(num_blocks, num_pull)
    trace_attn_at, trace_gather_at, trace_stage_at, trace_start_at, _ = trace_layout(
        M, n, hq_r, hkv_r, d, d_v
    )
    assert not trace or grid <= TRACE_MAX_BLOCKS

    elem = dtype_to_elem_type(traits.DTYPE_STR)
    o_stage = mla and traits.O_LDS_STAGE

    @fx.struct
    class SharedStorage:
        kv: fx.Array[elem, traits.LDS_KV_TOTAL_SIZE, 16]
        if o_stage:
            o: fx.Array[elem, traits.LDS_O_STAGE_SIZE, 16]
        claim: fx.Array[fx.Int32, 4, 16]

    c = dict(
        M=M,
        rank=rank,
        num_ranks=n,
        threads=threads,
        d_o=d_v,
        hq_r=hq_r,
        hkv_r=hkv_r,
        widths=widths,
        col_offs=(0, widths[0], widths[0] + widths[1]),
        stage_region=REGION_FWD_STAGE,
        ready_region=REGION_FWD_READY,
        N_r=N_r,
        K_o=K_o,
        seq_chunk=cs,
        m_tiles=m_tiles,
        stage_bytes=stage_bytes,
        recv_bytes=recv_bytes,
        block_m=bm,
        block_n=traits.BLOCK_N,
        head_groups=hg,
        heads_per_wg=traits.Q_HEADS_PER_WG,
        num_q_blocks=nqb,
        num_ascending=num_ascending,
        m_tiles_o=m_tiles_o,
        o_ascending=min(num_ascending // n, m_tiles_o),
        mode=mode,
        trace_attn_at=trace_attn_at,
        trace_gather_at=trace_gather_at,
        trace_stage_at=trace_stage_at,
    )

    @flyc.kernel(name="ulysses_a2a_attn_fwd_kernel", known_block_size=[threads, 1, 1])
    def kernel(
        Q_IN: fx.Tensor,
        K_IN: fx.Tensor,
        V_IN: fx.Tensor,
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        OUT: fx.Tensor,
        PEER_PTRS: fx.Tensor,
        STATE: fx.Tensor,
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

        if const_expr(mode != "attn_only"):
            in_res = [
                create_buffer_resource(t, num_records_bytes=M * n * w * 2)
                for t, w in zip((Q_IN, K_IN, V_IN), widths)
            ]
            # staging never waits, so the stagers finish it whatever the pull blocks wait for; the pull blocks
            # stay out of it to start on the first m-tiles while the later ones are staged
            if block_index >= fx.Int32(0 if grid <= num_pull else num_pull):
                _stage_role(
                    thread_index, state_base, claim_slot, my_base, bank_offset, in_res, epoch, trace_base, **c
                )
            if block_index < fx.Int32(num_pull):
                out_res = [
                    create_buffer_resource(t, num_records_bytes=S * w * 2) for t, w in zip((Q, K, V), widths)
                ]
                _pull_role(
                    thread_index,
                    state_base,
                    claim_slot,
                    peers_base,
                    my_base,
                    bank_offset,
                    out_res,
                    epoch,
                    trace_base,
                    **c,
                )

        def make_ctx(context):
            return context(
                traits,
                Q,
                K,
                V,
                O,
                LSE,
                O,
                O,
                O,
                fx.Int32(S),
                fx.Int32(S),
                fx.Int32(hq_r * d),
                fx.Int32(hkv_r * d),
                fx.Int32(d),
                fx.Int32(0),
                SINK=O,
                ScoreBound=None,
            )

        if const_expr(mla):
            # one setup per CTA, one bind per item: the persistent forward's job loop without its
            # cross-job prefetch (the next item is not known until it is claimed)
            mctx = make_ctx(MlaKernelContext)
            mctx._setup(None, bind_block=False, lds=lds)

            def run_tile(h, qb):
                mctx.bind_item(fx.Index(h), fx.Index(qb), fx.Index(0))
                attn_tile(mctx)

        else:

            def run_tile(h, qb):
                ctx = make_ctx(DualwaveKernelContext)
                # _setup walks causal q blocks descending from the grid y index
                ctx._setup(None, block_ids=(h, fx.Int32(nqb - 1) - qb, fx.Int32(0)), lds=lds)
                attn_tile(ctx)

        _attn_loop(
            thread_index,
            state_base,
            claim_slot,
            my_base,
            peers_base,
            bank_offset + fx.Int64(stage_bytes),
            create_buffer_resource(O, num_records_bytes=S * hq_r * d_v * 2),
            create_buffer_resource(OUT, num_records_bytes=recv_bytes),
            epoch,
            run_tile,
            trace_base,
            **c,
        )

        exit_and_bump_epoch(thread_index, state_base, epoch, grid, extra_counters=(STATE_ASSIST,))

    value_attrs = {
        "rocdl.waves_per_eu": traits.WAVES_PER_EU,
        "rocdl.flat_work_group_size": f"{threads},{threads}",
        "llvm.passthrough": [
            ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
            ["no-nans-fp-math", "true"],
            ["unsafe-fp-math", "true"],
        ],
    }
    return kernel, grid, threads, value_attrs


@flyc.jit
def _launch(
    Q_IN,
    K_IN,
    V_IN,
    Q,
    K,
    V,
    O,  # noqa: E741
    LSE,
    OUT,
    PEER_PTRS,
    STATE,
    TRACE,
    M: fx.Constexpr[int],
    rank: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    hq_r: fx.Constexpr[int],
    hkv_r: fx.Constexpr[int],
    head_dim: fx.Constexpr[int],
    head_dim_v: fx.Constexpr[int],
    sm_scale: fx.Constexpr[float],
    seq_chunk: fx.Constexpr[int],
    num_pull: fx.Constexpr[int],
    num_blocks: fx.Constexpr[int],
    num_ascending: fx.Constexpr[int],
    bank_bytes: fx.Constexpr[int],
    trace: fx.Constexpr[bool],
    mode: fx.Constexpr[str],
    stream: fx.Stream,
):
    kernel, grid, threads, value_attrs = _make_kernel(
        M,
        rank,
        num_ranks,
        hq_r,
        hkv_r,
        head_dim,
        head_dim_v,
        sm_scale or None,
        seq_chunk,
        num_pull,
        num_blocks,
        num_ascending,
        bank_bytes,
        trace,
        mode,
    )
    kernel(Q_IN, K_IN, V_IN, Q, K, V, O, LSE, OUT, PEER_PTRS, STATE, TRACE, value_attrs=value_attrs).launch(
        grid=(grid, 1, 1), block=(threads, 1, 1), stream=stream
    )


def ulysses_a2a_attn_fwd(
    q_in,
    k_in,
    v_in,
    q,
    k,
    v,
    o,
    lse,
    out,
    workspace,
    *,
    seq_chunk=0,
    num_pull=32,
    num_blocks=None,
    num_ascending=None,
    softmax_scale=None,
    trace=None,
    mode="full",
):
    """``q_in`` / ``k_in`` / ``v_in`` ``[M, n * h_r, d]``: this rank's tokens with every head (batch 1).
    Writes ``q`` / ``k`` / ``v`` ``[S, h_r, d]`` (this rank's heads over the sequence, sequence order),
    the causal attention output ``o`` (``[S, hq_r, d_v]``) with its natural-log LSE ``[S, hq_r]`` fp32, and
    ``out`` ``[M, n * hq_r, d_v]``: the output for this rank's tokens. The bank must hold
    ``a2a_attn_fwd_bank_bytes``. ``v_in`` may be narrower than q / k only with the MLA head dims (q / k
    192, v 128, ``Hq == Hkv``), whose forward also takes a ``softmax_scale`` (default ``d ** -0.5``).

    ``mode`` (measurement only): ``"attn_only"`` runs the attention items on the q / k / v already in place
    (no exchange, no waits, no O push or gather); ``"comm_only"`` runs every exchange step and wait but
    no attention tile."""
    n = workspace.num_ranks
    M, H, d = q_in.shape[0], q_in.shape[-2], q_in.shape[-1]
    d_v = v_in.shape[-1]
    hq_r, hkv_r = H // n, k_in.shape[-2] // n
    tensors = (q_in, k_in, v_in, q, k, v, o, lse, out)
    assert all(t.is_contiguous() for t in tensors)
    assert q_in.dtype == torch.bfloat16 and H % n == 0 and k_in.shape[-2] % n == 0
    assert d_v == d or (is_mla(d, d_v) and hq_r == hkv_r), f"head dims {d} / {d_v}"
    assert softmax_scale is None or is_mla(d, d_v), "the GQA forward bakes softmax_scale = d ** -0.5"
    assert (
        q.shape[0] == M * n
        and q.shape[-2] == hq_r
        and k.shape[-2] == hkv_r
        and v.shape[-1] == d_v
        and o.shape[-1] == d_v
        and out.shape == (M, H, d_v)
    )
    if num_blocks is None:
        num_blocks = torch.cuda.get_device_properties(q_in.device).multi_processor_count
    bm = make_traits(hq_r, hkv_r, d, d_v, softmax_scale).BLOCK_M
    if num_ascending is None:
        num_ascending = M * n // bm // 2
    with CompilationContext.compile_hints(DUALWAVE_SWP_COMPILE_HINTS):
        run_launcher(
            _launch,
            *tensors,
            workspace.peer_ptrs,
            workspace.state,
            workspace.state if trace is None else trace,
            M,
            workspace.rank,
            n,
            hq_r,
            hkv_r,
            d,
            d_v,
            0.0 if softmax_scale is None else float(softmax_scale),
            seq_chunk,
            num_pull,
            num_blocks,
            num_ascending,
            workspace.bank_bytes,
            trace is not None,
            mode,
            torch.cuda.current_stream(),
        )
