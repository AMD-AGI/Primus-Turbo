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
* after every item is claimed: dQ push units, ``dq_rows`` rows of 256-row q block ``j`` each in sequence
  order, wait for bands ``0..j`` (every band that adds into those rows of the atomic dQ image), un-permute
  and scale the rows, push them to their owners and zero them in the image for the next call; then gather units
  ``(m-tile, src)`` copy ``src``'s column blocks of the recv rows into dq / dk / dv ``[M, n * h_r, d]``.

Pull units are claimed by running blocks before those turn worker, every item is claimed before any dQ
push or gather unit, and an item's own waits are only on pull units, so no wait can deadlock whatever the
grid. Sequence split as in ``a2a_attn_kernel``.

MLA head dims (q / k ``d`` 192, v / O / dO ``d_v`` 128) run the gfx950 MLA dK / dV body instead: 192-row
bands (the last one may run past ``S``; its rows there are neither pushed nor gathered), dK / dV pushed in
64-row parts and dQ in 64-row units, the a16 dQ image possibly spread over several images (summed in the
push), Q unscaled (the body folds the softmax scale into its K packs).
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
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
    atomic_max,
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
WAVES = THREADS // WARP
# per (item, wave) below BAND_DONE_AT: ``epoch << 32 | p``, p = 1 + the unmasked trips whose dQ atomics
# have retired (PROG_DONE: the whole item)
PROG_AT = 0
PROG_DONE = 0x7FFFFFFF
# rows of one pull unit, of one push / gather tile
BM = 128
STAGE_SPLIT = 4
# kv rows of one backward band (the MLA body's: MLA_BAND)
BAND = 256
MLA_BAND = 192
MLA_BLOCK_Q = 32
# rows of one dK / dV push part and of one MLA dQ push unit: they divide BM and every MLA band
MLA_PART_ROWS = 64
# a band's dK / dV push: BM-row tile x (dK, dV)
KV_PARTS = 2 * (BAND // BM)
# dQ push unit bytes (BAND rows of 8 heads): one CU's push, so the last unit bounds the tail
DQ_UNIT_HEAD_ROWS = BAND * 8
PULL_UNIT_HEAD_ROWS = 1024
# 16 B vectors per thread in flight in the copies
_BATCH = 8
# 16-row dQ tiles per wave whose image loads a push step keeps in flight
_DQ_UNROLL = 4

_BF16 = fx.BFloat16

# LDS stash of the work loop's 64-bit bases (see _stashed)
_S_PEERS, _S_MY, _S_EPOCH, _S_COUNT, _S_WSK, _S_WSV, _S_IMG, _S_DQ, _S_STATE, _S_TRACE = 0, 1, 2, 3, 4, 5, 6, 7, 10, 11
_STASH_LEN = 12
# i32 word after the stash: the block has seen every pull unit counted
_DONE_AT = 2 * _STASH_LEN
# pull units done (reset at exit)
STATE_PULLED = STATE_ASSIST_DONE
# dQ push unit claim counter (reset at exit)
STATE_DQ = 6
# finished bands queued for the dK / dV helpers, and the helpers' unit counter (reset at exit)
STATE_KVQ = 7
STATE_KVH = 8


def _ceildiv(a, b):
    return (a + b - 1) // b


def _link_plan(num_ranks, S, mla=False):
    """``(num_pull, num_comm, local_first, num_top)``: on two ranks one link carries the whole exchange,
    and fewer pullers land the first dO rows sooner while dedicated pushers keep it busy as bands finish;
    with more links the exchange is short and every block is worth more on the items. Local first pays
    only while the start is a large share of the call (two ranks, up to 16K: A/B -1.0 / -1.5%, +0.8..+2.8%
    above). Four ranks keep pushers up to 32K (64K: A/B -1.5% without them) and pull 8 top bands right
    after the prefix (A/B 0.0..-2.0% up to 32K); eight ranks pull with 48 blocks (A/B -0.2..-1.3%).
    MLA moves 5-10x the GQA bytes per row of attention: on eight ranks 8 pullers, 4 of them pushers
    (DeepSeek-V3 / Kimi-K2 heads, 32K / 64K: A/B -2.6..-7.4% against 48 / 0)."""
    if mla and num_ranks == 8:
        return 8, 4, False, 0
    if num_ranks == 2:
        return 24, 8, S <= 16384, 0
    if num_ranks == 4:
        return 32, (4 if S <= 32768 else 0), False, (8 if S <= 32768 else 0)
    return 48, 0, False, 0


def _pull_rows(hq_r, pull_rows=None):
    """Sequence rows of one pull unit: ``PULL_UNIT_HEAD_ROWS`` (row, head) pairs, 32 to 128 rows. The pull
    blocks copy their units side by side over the link, so the first unit lands after all of them."""
    if pull_rows is None:
        pull_rows = BM
        while pull_rows > BM // STAGE_SPLIT and pull_rows * hq_r > PULL_UNIT_HEAD_ROWS:
            pull_rows //= 2
    return pull_rows


def _dq_rows(hq_r, dq_rows=None, mla=False):
    """Sequence rows of one dQ push unit: a divisor of BAND near DQ_UNIT_HEAD_ROWS / hq_r (MLA: of
    MLA_PART_ROWS, so a unit stays inside one band and one gather tile)."""
    if dq_rows is None:
        dq_rows = MLA_PART_ROWS if mla else BAND
        while dq_rows > 32 and dq_rows * hq_r > DQ_UNIT_HEAD_ROWS:
            dq_rows //= 2
    return dq_rows


def a2a_attn_bwd_bank_bytes(M, num_ranks, hq_r, hkv_r, head_dim, head_dim_v=None):
    """Bank bytes: the dO staging rows, then the dq / dk / dv recv rows."""
    d_v = head_dim if head_dim_v is None else head_dim_v
    return M * num_ranks * ((hq_r + hkv_r) * (head_dim + d_v)) * 2


@functools.lru_cache(maxsize=16)
def _plan(S, hq_r, hkv_r, d, d_v=None, softmax_scale=None):
    """The dense SBHD backward plan for one rank's heads: ``(q_split, dkdv launcher, dQ image scale)``,
    with the stock body (``d_v == d``, scale ``d ** -0.5``) or the gfx950 MLA one."""
    if d_v is None or d_v == d:
        p = _bwd._dense_plan(1, S, S, hq_r, hkv_r, d, 1.0 / math.sqrt(d), -1, True, False)
        q_split, block_kv, a16, a16_nat, dkdv_l, bat_plan = p[1], p[2], p[4], p[5], p[11], p[13]
        assert a16 and not a16_nat and block_kv == BAND and len(bat_plan) == 1, (
            "the fused backward runs the a16 dQ-image body with 256-row bands"
        )
        return q_split, dkdv_l, dkdv_l.dq_scale
    from primus_turbo.flydsl.attention.gfx950_mla import ops_bwd as _mla

    scale = 1.0 / math.sqrt(d) if softmax_scale is None else float(softmax_scale)
    n_img = _mla._a16_imgs_for(S, MLA_BAND, 1, hkv_r)
    kw = _mla._dkdv_kwargs(1, hq_r, hkv_r, d, d_v, scale, MLA_BAND, MLA_BLOCK_Q, 1, sbhd=True, a16_imgs=n_img)
    dkdv_l = _mla.fb.build_flash_attn_bwd_dkdv_module(**kw)
    assert kw["k_pre"], "the MLA body takes Q unscaled only with the score fold on its K packs"
    return 1, dkdv_l, 1.0 / _mla._LOG2E


def _layout_reason(hq_r, hkv_r, d, d_v, pull_rows, dq_rows):
    """Why the copy loops' whole steps do not tile these heads with these pull / dQ push units (``None``:
    they do)."""
    mla = d_v != d
    band = MLA_BAND if mla else BAND
    kv_rows = MLA_PART_ROWS if mla else BM
    if band % dq_rows or (dq_rows // 16 * hq_r) % (THREADS // WARP):
        return "dq_rows: 16-row tiles per wave"
    if mla and BM % dq_rows:
        return "MLA dQ push units: inside one gather tile"
    if (dq_rows * hq_r * d) % (VEC_BF16 * THREADS):
        return "dQ push units: whole zeroing steps"
    if pull_rows not in (BM // 4, BM // 2, BM) or BM // STAGE_SPLIT > pull_rows:
        return f"pull_rows {pull_rows}"
    if (pull_rows * hq_r * d_v // VEC_BF16) % (THREADS * _BATCH) or pull_rows * hq_r % THREADS:
        return "pull units: whole copy steps"
    if BM * hq_r % THREADS:
        return "stage pieces: whole rows per thread"
    for h_r, dd in ((hq_r, d), (hkv_r, d), (hkv_r, d_v)):
        if (BM * h_r * dd // VEC_BF16) % (THREADS * _BATCH):
            return "gather tiles: whole copy steps"
    for dd in (d, d_v):
        if (kv_rows * hkv_r * dd // VEC_BF16) % (THREADS * 2):
            return "dK / dV parts: whole push steps"
    return None


@functools.lru_cache(maxsize=64)
def a2a_attn_bwd_unsupported(M, num_ranks, hq_r, hkv_r, head_dim, head_dim_v=None, seq_chunk=0, softmax_scale=None):
    """Why the fused backward cannot run this shape (``None``: it can): the stock plan must be the a16 dQ
    image with 256-row bands (dense causal, ``S`` a multiple of the band); MLA takes 192-row bands and
    64-row dK / dV parts inside one chunk. Byte offsets stay in i32 and the flags in their regions."""
    n, d = num_ranks, head_dim
    d_v = d if head_dim_v is None else head_dim_v
    mla = d_v != d
    S = M * n
    cs = seq_chunk if seq_chunk else M
    if (d, d_v) not in ((64, 64), (128, 128), (192, 128)):
        return f"head dims {d} / {d_v}"
    if M % cs or cs % BM:
        return f"need M % seq_chunk == 0, seq_chunk % {BM} == 0"
    if mla and (cs % 256 or S % MLA_PART_ROWS):
        return f"MLA: S a multiple of {MLA_PART_ROWS}, seq_chunk of 256"
    if not mla and S % BAND:
        return f"S a multiple of {BAND}"
    try:
        q_split = _plan(S, hq_r, hkv_r, d, d_v if mla else None, softmax_scale if mla else None)[0]
    except AssertionError as e:
        return str(e) or "no fused backward plan"
    if q_split * S * hkv_r * d * 2 >= 2**31 or S * hq_r * d * 2 >= 2**31:
        return "i32 byte offsets overflow"
    kv_rows = MLA_PART_ROWS if mla else BM
    dq_flag_rows = min(_dq_rows(hq_r, None, mla), BM)
    if M // BM * STAGE_SPLIT * n > FLAGS_PER_REGION or (M // dq_flag_rows + 2 * (M // kv_rows)) * n > FLAGS_PER_REGION:
        return "flag regions too small"
    if _ceildiv(S, MLA_BAND if mla else BAND) > BAND_DONE_AT:
        return "too many bands for the band flags"
    return _layout_reason(hq_r, hkv_r, d, d_v, _pull_rows(hq_r), _dq_rows(hq_r, None, mla))


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
    h_vecs = c["hq_r"] * c["d_v"] // VEC_BF16
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
            top = c["top_m_tiles"] * STAGE_SPLIT
            # the pullers of an m-tile start with src = dst - i: stage the other ranks' blocks first
            dst = (fx.Int32(rank + 1) + piece % fx.Int32(n)) % fx.Int32(n)
            m_sub = arith.select(
                k < fx.Int32(top),
                fx.Int32(m_tiles * STAGE_SPLIT - 1) - k,
                arith.select(k < fx.Int32(top + pre), k - fx.Int32(top), fx.Int32(m_tiles * STAGE_SPLIT - 1 + pre) - k),
            )
            if const_expr(c["local_bands"] > 0):
                # only this rank's first wave starts on its own rows: the peers pull the top rows first
                m_sub = arith.select(dst == fx.Int32(rank), m_sub, fx.Int32(m_tiles * STAGE_SPLIT - 1) - k)
            _stage_piece(opaque_tid(thread_index), m_sub, dst, in_res, stage_res, **c)
            if thread_index == fx.Int32(0):
                st(stage_flags, m_sub * fx.Int32(n) + dst, epoch, scope="sys", order="relaxed")
                if const_expr(trace_base is not None):
                    st(trace_base, fx.Int32(c["trace_stage_at"]) + m_sub * fx.Int32(n) + dst, read_clock())
        more = arith.select(piece < fx.Int32(num_pieces), fx.Int32(1), fx.Int32(0))


@ASTRewriter.transform
def _pull_unit(thread_index, src, m_tile, src_heap, bank_offset, do_res, o_res, delta_res, prep, epoch, **c):
    """``src``'s staged dO rows ``[m_tile * pull_rows, +pull_rows)`` of this rank's column block into dO, with their delta;
    ``prep`` = (lse, scaled LSE) resources: those rows of the latter, as the stock ``_prescale_lse``
    rounds them (the Q prescale stays a launch: HBM-bound, it would slow the pull the items wait on)."""
    n, d, rank, cs, hq_r, S = c["num_ranks"], c["d_v"], c["rank"], c["seq_chunk"], c["hq_r"], c["S"]
    pr = c["pull_rows"]
    pieces = pr // (BM // STAGE_SPLIT)
    lanes = d // VEC_BF16
    h_vecs = hq_r * lanes
    row_vecs = n * h_vecs
    if thread_index == fx.Int32(0):
        flag_base = src_heap + fx.Int64(REGION_BWD_STAGE * FLAG_REGION_BYTES)
        flag0 = m_tile * fx.Int32(pieces * n) + fx.Int32(rank)
        spin_flags(flag_base, [flag0 + fx.Int32(i * n) for i in range(pieces)], epoch, rank, src, "bwd pull", report=False)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    stage_res = create_buffer_resource_from_addr(src_heap + bank_offset, num_records_bytes=c["stage_bytes"])
    t0 = m_tile * fx.Int32(pr)
    # a unit lies inside one sequence chunk (cs % BM == 0): its rows land contiguously
    g0 = _global_row(t0, src, cs, n)
    lane = thread_index % fx.Int32(WARP)
    first_lane = thread_index % fx.Int32(lanes) == fx.Int32(0)
    per_iter = THREADS * _BATCH
    # the lanes of one (row, head) are consecutive: their partial sums fold within the wave
    for it in range(pr * h_vecs // per_iter):
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
    for it in range_constexpr(pr * c["hq_r"] // THREADS):
        x = fx.Int32(it * THREADS) + thread_index
        r = g0 + x % fx.Int32(pr)
        h = x // fx.Int32(pr)
        v = buffer_load(lse_res, r * fx.Int32(c["hq_r"]) + h, vec_width=1, dtype=fx.Float32)
        buffer_store(fx.Float32(v) * fx.Float32(-_bwd._LOG2E), lsef_res, h * fx.Int32(S) + r)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()


@ASTRewriter.transform
def _pull_role(thread_index, state_base, claim_slot, peers_base, my_base, bank_offset, do_res, o_res, delta_res, prep, epoch, trace_base, **c):
    """Pull units ``(m_tile, src)``, one ``pull_rows`` block of the sequence each: the top ``top_tiles``,
    then the first ``prefix_tiles`` from the bottom up and then the rest from the top down, until none are
    left."""
    n, p_tiles, cs, pre, rank, pr = c["num_ranks"], c["p_tiles"], c["seq_chunk"], c["prefix_tiles"], c["rank"], c["pull_rows"]
    local = c["local_bands"] > 0
    ready_base = my_base + fx.Int64(REGION_BWD_READY * FLAG_REGION_BYTES)
    num_units = p_tiles * n
    # local first: this rank's own prefix tiles, then every tile from the top down (those again skipped)
    pre_m = c["prefix_m_tiles"] * BM // pr if local else 0
    num_claims = num_units + pre_m
    more = fx.Int32(1)
    while more == fx.Int32(1):
        unit = claim_unit(opaque_tid(thread_index), state_base, claim_slot, STATE_CLAIM)
        if const_expr(local):
            src, t = _owner_row((fx.Int32(num_claims - 1) - unit) * fx.Int32(pr), cs, n)
            src = arith.select(unit < fx.Int32(pre_m), fx.Int32(rank), src)
            t = arith.select(unit < fx.Int32(pre_m), unit * fx.Int32(pr), t)
            dup = (unit >= fx.Int32(pre_m)) & (src == fx.Int32(rank)) & (t < fx.Int32(pre_m * pr))
            go = arith.select(dup, fx.Int32(0), fx.Int32(1))
        else:
            top = c["top_tiles"]
            g = arith.select(
                unit < fx.Int32(top),
                fx.Int32(num_units - 1) - unit,
                arith.select(unit < fx.Int32(top + pre), unit - fx.Int32(top), fx.Int32(num_units - 1 + pre) - unit),
            )
            src, t = _owner_row(g * fx.Int32(pr), cs, n)
            go = fx.Int32(1)
        if (unit < fx.Int32(num_claims)) & (go == fx.Int32(1)):
            m_tile = t // fx.Int32(pr)
            src_heap = ld(peers_base, src, dtype=fx.T.i64)
            _pull_unit(opaque_tid(thread_index), src, m_tile, src_heap, bank_offset, do_res, o_res, delta_res, prep, epoch, **c)
            if thread_index == fx.Int32(0):
                st(ready_base, m_tile * fx.Int32(n) + src, epoch, scope="agent", order="release")
                atomic_add(state_base, fx.Int32(STATE_PULLED), fx.Int64(1))
                if const_expr(trace_base is not None):
                    st(trace_base, m_tile * fx.Int32(n) + src, read_clock())
        more = arith.select(unit < fx.Int32(num_claims), fx.Int32(1), fx.Int32(0))


def _min_flag(base, idx, count, stride):
    """The lowest of the flags ``idx + i * stride``, ``i < count``."""
    flag = ld(base, idx, scope="agent", dtype=fx.T.i64)
    for i in range(1, count):
        f = ld(base, idx + fx.Int32(i * stride), scope="agent", dtype=fx.T.i64)
        flag = arith.select(f < flag, f, flag)
    return flag


def _prog_tag(epoch):
    return epoch * fx.Int64(1 << 32)


def _desc_walk(band, split, **c):
    """The body's unmasked q walk of item ``(band, split)``: its bottom row, top row, trips and rotation
    (``_desc_q``: trip ``i`` visits ``top - ((i + rot) % trips) * step``)."""
    S, bq, step, rr, bk = c["S"], c["block_q"], c["block_q"] * c["q_split"], c["qdesc_r"], c["band_rows"]
    bottom = band * fx.Int32(bk) + fx.Int32(bk) + split * fx.Int32(bq)
    span = arith.select(fx.Int32(S) > bottom, fx.Int32(S) - bottom, fx.Int32(0))
    trips = (span + fx.Int32(step - 1)) // fx.Int32(step)
    trips = arith.select(trips > fx.Int32(0), trips, fx.Int32(1))
    top = bottom + (trips - fx.Int32(1)) * fx.Int32(step)
    return bottom, top, trips, band % fx.Int32(rr)


def _desc_trip(row, top, trips, rot, **c):
    step, rr = c["block_q"] * c["q_split"], c["qdesc_r"]
    return ((top - row) // fx.Int32(step) + trips * fx.Int32(rr) - rot) % trips


@ASTRewriter.transform
def _publish_progress(row, band, pid, stash, tid, **c):
    """At the hook for the block of trip ``i`` (run while trip ``i - 1`` starts), trips ``< i - 1`` have
    issued every dQ atomic (the last head of a trip emits in the next one), the masked ones too once
    ``i >= 2``; every ``dq_prog`` trips, each wave publishes that its own have retired. Relaxed: the
    atomics resolve past every L2 (the stock image takes them from all XCDs), so retiring is enough."""
    qs, K = c["q_split"], c["dq_prog"]
    bottom, top, trips, rot = _desc_walk(band, pid % fx.Int32(qs), **c)
    i = _desc_trip(row, top, trips, rot, **c)
    if (row >= bottom) & (row <= top) & (i >= fx.Int32(2)) & ((i - fx.Int32(2)) % fx.Int32(K) == fx.Int32(0)):
        fx.rocdl.s_waitcnt(0)
        if tid % fx.Int32(WARP) == fx.Int32(0):
            slot = fx.Int32(PROG_AT) + pid * fx.Int32(WAVES) + tid // fx.Int32(WARP)
            value = _prog_tag(_stashed(stash, _S_EPOCH)) + _i64(i - fx.Int32(1))
            st(_region(stash, REGION_BWD_BAND), slot, value, scope="agent", order="relaxed")


@ASTRewriter.transform
def _q_ready(q, stash, tid, band=None, pid=None, **c):
    """The body's hook before its first dO / delta fetch of q block ``q``: until the block has seen every
    pull unit counted, wait for the ready flag of the unit holding row ``q``. Relaxed: the pull's release
    wrote the rows back, and no cache of this launch holds a line of them read before the flag. With
    ``dq_prog``, also the item's dQ progress (``_publish_progress``)."""
    n, cs, S = c["num_ranks"], c["seq_chunk"], c["S"]
    row = fx.Int32(q)
    if const_expr(c["dq_prog"] > 0):
        _publish_progress(row, band, pid, stash, tid, **c)
    done = _readfirstlane_i32(ld(stash, fx.Int32(_DONE_AT), scope="workgroup", space=3))
    # the body's prefetch past its last q block reads rows past the end (clamped) and never uses them
    if (done == fx.Int32(0)) & (row < fx.Int32(S)):
        src, t = _owner_row(row, cs, n)
        # the q block's rows: one chunk of one source, every pull unit they span
        idx = t // fx.Int32(c["pull_rows"]) * fx.Int32(n) + src
        ready_base = _region(stash, REGION_BWD_READY)
        epoch = _stashed(stash, _S_EPOCH)
        spin_start = read_clock()
        nf = max(c["block_q"] // c["pull_rows"], 1)
        flag = _min_flag(ready_base, idx, nf, n)
        spun = flag < epoch
        while flag < epoch:
            fx.rocdl.s_sleep(fx.Int32(1))
            # no printf: its hostcall code inside the body spills the body's main loop
            if spin_timed_out(spin_start, SPIN_TIMEOUT):
                _asm("s_trap 2")
            flag = _min_flag(ready_base, idx, nf, n)
        if const_expr(c["trace"]):
            if spun & (tid == fx.Int32(0)):
                atomic_add(_stashed(stash, _S_TRACE), fx.Int32(c["trace_qwait_at"]), read_clock() - spin_start)
        pulled = ld(_stashed(stash, _S_STATE), fx.Int32(STATE_PULLED), scope="agent", dtype=fx.T.i64)
        if pulled == fx.Int64(c["p_tiles"] * n):
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
            fx.rocdl.s_sleep(fx.Int32(c["spin_sleep"]))
            if spin_timed_out(spin_start, SPIN_TIMEOUT):
                fx.printf(f"[a2a_attn_bwd rank={rank}] band flag timeout: band={{}} flag={{}}\n", idx, flag)
                _asm("s_trap 2")
            flag = ld(band_base, fx.Int32(BAND_DONE_AT) + idx, scope="agent", dtype=fx.T.i64)
        idx = idx + fx.Int32(THREADS)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    memory_fence(order="acquire", scope="agent")


@ASTRewriter.transform
def _wait_progress(thread_index, band_base, j, part, epoch, **c):
    """Wait until every item of bands ``0..j`` has retired its dQ atomics into rows ``[j * BAND + part *
    dq_rows, +dq_rows)``: band ``j``'s masked trips, and for each lower band the unmasked trips that visit
    those rows (``_publish_progress``)."""
    rank, ipb, qs, bq, rows = c["rank"], c["items_per_band"], c["q_split"], c["block_q"], c["dq_rows"]
    step = bq * qs
    r0 = j * fx.Int32(c["band_rows"]) + part * fx.Int32(rows)
    tag = _prog_tag(epoch)
    idx = thread_index
    spin_start = read_clock()
    while idx < (j + fx.Int32(1)) * fx.Int32(ipb):
        b = idx // fx.Int32(ipb)
        s = idx % fx.Int32(qs)
        _, top, trips, rot = _desc_walk(b, s, **c)
        need = fx.Int32(1)
        for m in range_constexpr(rows // step + 1):
            r = r0 // fx.Int32(step) * fx.Int32(step) + s * fx.Int32(bq) + fx.Int32(m * step)
            t2 = _desc_trip(r, top, trips, rot, **c) + fx.Int32(2)
            need = arith.select((b < j) & (r >= r0) & (r < r0 + fx.Int32(rows)) & (t2 > need), t2, need)
        want = tag + _i64(need)
        at = fx.Int32(PROG_AT) + idx * fx.Int32(WAVES)
        flag = _min_flag(band_base, at, WAVES, 1)
        while flag < want:
            fx.rocdl.s_sleep(fx.Int32(c["spin_sleep"]))
            if spin_timed_out(spin_start, SPIN_TIMEOUT):
                fx.printf(f"[a2a_attn_bwd rank={rank}] dQ progress timeout: item={{}} need={{}} flag={{}}\n", idx, need, flag)
                _asm("s_trap 2")
            flag = _min_flag(band_base, at, WAVES, 1)
        idx = idx + fx.Int32(THREADS)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    memory_fence(order="acquire", scope="agent")


@ASTRewriter.transform
def _push_kv_part(thread_index, band, part, peers_base, recv_offset, wsk_res, wsv_res, epoch, **c):
    """Fold part ``part`` (``kv_rows`` rows ``part // 2`` of the band, dK or dV by ``part % 2``) of the band's
    ``q_split`` slots (ascending, fp32, as the stock slot reduce) and push the rows into their owner's recv
    bank (this rank's column block), then the owner's flag. A part past ``S`` (the MLA tail band) is empty."""
    n, rank, cs, hkv_r, S, qs = c["num_ranks"], c["rank"], c["seq_chunk"], c["hkv_r"], c["S"], c["q_split"]
    kr = c["kv_rows"]
    h, x = part // 2, part % 2
    d = (c["head_dim"], c["d_v"])[x]
    row_elems = hkv_r * d
    row_vecs = row_elems // VEC_BF16
    slot_elems = S * row_elems
    K_kv = n * row_elems
    per_iter = THREADS * 2
    pos0 = band * fx.Int32(c["band_rows"]) + fx.Int32(h * kr)
    if pos0 < fx.Int32(S):
        owner, t0 = _owner_row(pos0, cs, n)
        owner_heap = ld(peers_base, owner, dtype=fx.T.i64)
        ws_res = (wsk_res, wsv_res)[x]
        recv_res = create_buffer_resource_from_addr(
            owner_heap + recv_offset + fx.Int64(c["recv_kv_at"][x]), num_records_bytes=c["recv_kv_bytes"][x]
        )
        for it in range(kr * row_vecs // per_iter):
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
            slot = (fx.Int32(c["dq_flags"] + x * c["kv_tiles"]) + t0 // fx.Int32(kr)) * fx.Int32(n) + fx.Int32(rank)
            st(owner_heap + fx.Int64(REGION_BWD_RECV * FLAG_REGION_BYTES), slot, epoch, scope="sys", order="relaxed")


@ASTRewriter.transform
def _claim_kv_part(thread_index, band, part, want, claim_slot, count_base, epoch, **c):
    """Whether this block takes part ``part`` of the band's dK / dV push (only tried if ``want``): the
    first claim of the call (epoch-stamped, no reset)."""
    if thread_index == fx.Int32(0):
        st_lds(claim_slot, 2, fx.Int32(0))
        if want:
            at = fx.Int32(c["kv_claim_at"]) + band * fx.Int32(c["kv_parts"]) + fx.Int32(part)
            old = atomic_max(count_base, at, epoch, scope="agent")
            st_lds(claim_slot, 2, arith.select(old < epoch, fx.Int32(1), fx.Int32(0)))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    mine = _readfirstlane_i32(ld_lds(claim_slot, 2))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    return mine


@ASTRewriter.transform
def _kv_parts(thread_index, band, claim_slot, count_base, peers_base, recv_offset, wsk_res, wsv_res, epoch, part=None, **c):
    """Push the band's dK / dV parts this block claims (``part``: that one only, else every one)."""
    for p in range_constexpr(c["kv_parts"]):
        if const_expr(not c["kv_help"]):
            _push_kv_part(thread_index, band, p, peers_base, recv_offset, wsk_res, wsv_res, epoch, **c)
        else:
            want = band >= fx.Int32(0)
            if const_expr(part is not None):
                want = part == fx.Int32(p)
            mine = _claim_kv_part(thread_index, band, p, want, claim_slot, count_base, epoch, **c)
            if mine == fx.Int32(1):
                _push_kv_part(thread_index, band, p, peers_base, recv_offset, wsk_res, wsv_res, epoch, **c)


@ASTRewriter.transform
def _finish_item(thread_index, band, pid, claim_slot, count_base, band_base, state_base, peers_base, recv_offset, wsk_res, wsv_res, epoch, **c):
    """Count the item ``pid`` on its band; the band's last item marks it done, queues it for the dK / dV
    helpers and pushes the parts none of them took."""
    ipb = c["items_per_band"]
    # this item's dK / dV slot stores and dQ atomics retire before its count releases them
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if const_expr(c["dq_prog"] > 0):
        if thread_index < fx.Int32(WAVES):
            done = _prog_tag(epoch) + fx.Int64(PROG_DONE)
            st(band_base, fx.Int32(PROG_AT) + pid * fx.Int32(WAVES) + thread_index, done, scope="agent", order="relaxed")
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
            # the dQ image rows are final here: the dQ push units need not wait for the dK / dV push
            st(band_base, fx.Int32(BAND_DONE_AT) + band, epoch, scope="agent", order="release")
            if const_expr(c["kv_help"]):
                q = atomic_add(state_base, fx.Int32(STATE_KVQ), fx.Int64(1), scope="agent")
                entry = _prog_tag(epoch) + _i64(band)
                st(count_base, fx.Int32(c["kv_queue_at"]) + cast(q, fx.T.i32), entry, scope="agent", order="release")
        _kv_parts(thread_index, band, claim_slot, count_base, peers_base, recv_offset, wsk_res, wsv_res, epoch, **c)


@ASTRewriter.transform
def _kv_help(thread_index, claim_slot, stash, **c):
    """Help units ``(q, part)`` in the order bands finish: part ``part`` of the ``q``-th finished band,
    unless its last item (or another helper) took it first."""
    nb, rank = c["num_bands"], c["rank"]
    u = claim_unit(opaque_tid(thread_index), _stashed(stash, _S_STATE), claim_slot, counter=STATE_KVH)
    while u < fx.Int32(nb * c["kv_parts"]):
        epoch = _stashed(stash, _S_EPOCH)
        count_base = _stashed(stash, _S_COUNT)
        if thread_index == fx.Int32(0):
            at = fx.Int32(c["kv_queue_at"]) + u // fx.Int32(c["kv_parts"])
            tag = _prog_tag(epoch)
            spin_start = read_clock()
            entry = ld(count_base, at, scope="agent", dtype=fx.T.i64)
            while entry < tag:
                fx.rocdl.s_sleep(fx.Int32(c["spin_sleep"]))
                if spin_timed_out(spin_start, SPIN_TIMEOUT):
                    fx.printf(f"[a2a_attn_bwd rank={rank}] dK/dV queue timeout: unit={{}} entry={{}}\n", u, entry)
                    _asm("s_trap 2")
                entry = ld(count_base, at, scope="agent", dtype=fx.T.i64)
            st_lds(claim_slot, 1, cast(entry - tag, fx.T.i32))
        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        band = _readfirstlane_i32(ld_lds(claim_slot, 1))
        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        # the band's items released their slots before counting; the queue entry orders after the count
        memory_fence(order="acquire", scope="agent")
        _kv_parts(
            opaque_tid(thread_index),
            band,
            claim_slot,
            count_base,
            _stashed(stash, _S_PEERS),
            _recv_offset(epoch, c),
            _ws_res(stash, _S_WSK, c),
            _ws_res(stash, _S_WSV, c),
            epoch,
            part=u % fx.Int32(c["kv_parts"]),
            **c,
        )
        u = claim_unit(opaque_tid(thread_index), _stashed(stash, _S_STATE), claim_slot, counter=STATE_KVH)


@ASTRewriter.transform
def _push_dq(thread_index, j, part, band_base, peers_base, recv_offset, img_res, epoch, **c):
    """Un-permute, sum (fp32, over the images ``img_res``) and scale rows ``[j * band + part * dq_rows,
    +dq_rows)`` of the dQ image (the stock dqa16 map; bands ``0..j`` done) into their owners' recv banks
    (this rank's column block), then the owners' flags."""
    n, d, rank, cs, hq_r, rows = c["num_ranks"], c["head_dim"], c["rank"], c["seq_chunk"], c["hq_r"], c["dq_rows"]
    gran = c["dq_flag_rows"]
    tiles = rows // 16
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
    waves = THREADS // WARP
    n_it = tiles * hq_r // waves
    # tiles per step: their image loads are all in flight before the first store (at most the GQA
    # D128 step's 64 dwords a lane)
    unroll = next(
        u for u in (_DQ_UNROLL, 2, 1) if n_it % u == 0 and (u * len(img_res) * npair * 4 <= 96 or u == 1)
    )
    for it in range(n_it // unroll):
        steps = []
        for u in range_constexpr(unroll):
            w = fx.Int32(it * (unroll * waves) + u * waves) + wave
            qt = j * fx.Int32(c["band_rows"] // 16) + part * fx.Int32(tiles) + w // fx.Int32(hq_r)
            qh = w % fx.Int32(hq_r)
            ibase = (
                (qt * fx.Int32(hq_r) + qh) * fx.Int32(npair * 512) + w1 * fx.Int32(256) + kgh * fx.Int32(64) + lane16 * fx.Int32(2)
            )
            parts = [
                [
                    [
                        buffer_load(
                            res,
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
                for res in img_res
            ]
            steps.append((qt, qh, parts))
        for qt, qh, img_parts in steps:
            # the tile pair (qt, qt ^ 1) holds one 32-row group, inside one chunk (cs % BM == 0)
            pos = (qt >> fx.Int32(1)) * fx.Int32(32)
            owner, t = _owner_row(pos, cs, n)
            owner_heap = ld(peers_base, owner, dtype=fx.T.i64)
            recv_res = create_buffer_resource_from_addr(owner_heap + recv_offset, num_records_bytes=c["recv_q_bytes"])
            row = t + q_in + (qt & fx.Int32(1)) * fx.Int32(8)
            obase = row * fx.Int32(K_q) + fx.Int32(rank * hq_r * d) + qh * fx.Int32(d) + w1 * fx.Int32(16) + kgh * fx.Int32(8)
            for p in range_constexpr(npair):
                acc = None
                for parts in img_parts:
                    lo = Vec(parts[p][0]).shuffle(Vec(parts[p][1]), [0, 1, 2, 3])
                    hi = Vec(parts[p][2]).shuffle(Vec(parts[p][3]), [0, 1, 2, 3])
                    v8 = lo.shuffle(hi, [0, 1, 2, 3, 4, 5, 6, 7]).to(fx.Float32)
                    acc = v8 if acc is None else acc + v8
                buffer_store(
                    (acc * sm_vec).to(_BF16).ir_value(),
                    recv_res,
                    obase + fx.Int32(p * 32),
                    cache_modifier=SYS_NT,
                )
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    # the unit's image rows are one contiguous span: zero it for the next call's atomics
    span = rows * hq_r * d
    zero = Vec.filled(VEC_BF16, 0.0, _BF16).ir_value()
    row0 = j * fx.Int32(c["band_rows"]) + part * fx.Int32(rows)
    for res in img_res:
        for it in range_constexpr(span // VEC_BF16 // THREADS):
            buffer_store(zero, res, row0 * fx.Int32(hq_r * d) + (fx.Int32(it * THREADS) + thread_index) * fx.Int32(VEC_BF16))
    if thread_index == fx.Int32(0):
        for h in range_constexpr(rows // gran):
            owner, t0 = _owner_row(row0 + fx.Int32(h * gran), cs, n)
            owner_heap = ld(peers_base, owner, dtype=fx.T.i64)
            slot = t0 // fx.Int32(gran) * fx.Int32(n) + fx.Int32(rank)
            st(owner_heap + fx.Int64(REGION_BWD_RECV * FLAG_REGION_BYTES), slot, epoch, scope="sys", order="relaxed")


@ASTRewriter.transform
def _gather_unit(thread_index, g, src, flag_base, recv_addr, out_res, epoch, **c):
    """``src``'s column blocks of recv tile ``g`` (dq, dk, dv) into the outputs once ``src`` pushed them."""
    n = c["num_ranks"]
    if thread_index == fx.Int32(0):
        per_g = BM // c["dq_flag_rows"]
        flag_idxs = [(g * fx.Int32(per_g) + fx.Int32(i)) * fx.Int32(n) + src for i in range(per_g)]
        per_kv = BM // c["kv_rows"]
        flag_idxs += [
            (fx.Int32(c["dq_flags"] + x * c["kv_tiles"]) + g * fx.Int32(per_kv) + fx.Int32(i)) * fx.Int32(n) + src
            for x in range(2)
            for i in range(per_kv)
        ]
        spin_flags(flag_base, flag_idxs, epoch, c["rank"], src, "a2a_attn_bwd gather", sleep=c["spin_sleep"])
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    # recv rows are rewritten by peers every 2 epochs: drop what this CU / XCD still caches of them
    memory_fence(order="acquire", scope="sys")
    row0 = g * fx.Int32(BM)
    for x in range_constexpr(3):
        blk_vecs = (c["hq_r"], c["hkv_r"], c["hkv_r"])[x] * (c["head_dim"], c["head_dim"], c["d_v"])[x] // VEC_BF16
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
    d = c["head_dim"] if i == _S_WSK else c["d_v"]
    return create_buffer_resource_from_addr(
        _stashed(stash, i), num_records_bytes=c["q_split"] * c["S"] * c["hkv_r"] * d * 2
    )


def _img_res(stash, c):
    """The dQ images, one descriptor each (the stack can outgrow a 32-bit buffer offset)."""
    img_bytes = c["S"] * c["hq_r"] * c["head_dim"] * 2
    return [
        create_buffer_resource_from_addr(_stashed(stash, _S_IMG) + fx.Int64(i * img_bytes), num_records_bytes=img_bytes)
        for i in range(c["n_img"])
    ]


def read_shader_clock():
    # shader-clock cycles (follow the core clock, unlike s_memrealtime): frequency traces
    op = llvm.inline_asm(fx.T.i64, [], "s_memtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", has_side_effects=True)
    return fx.arith.ArithValue(op, signed=False)


@ASTRewriter.transform
def _trace(stash, at, thread_index):
    tb = _stashed(stash, _S_TRACE)
    if thread_index == fx.Int32(0):
        st(tb, at, read_clock())


def _local_first_band(kk, **c):
    """The ``kk``-th band of the local-first order: this rank's own bands of ``[0, n * local_bands)``,
    then the other ones of that range, both ascending, then the rest."""
    n, rank, lb, bpc = c["num_ranks"], c["rank"], c["local_bands"], c["bands_per_chunk"]
    own = kk // fx.Int32(bpc) * fx.Int32(n * bpc) + fx.Int32(rank * bpc) + kk % fx.Int32(bpc)
    r = kk - fx.Int32(lb)
    rem = r % fx.Int32((n - 1) * bpc)
    pos = arith.select(rem < fx.Int32(rank * bpc), rem, rem + fx.Int32(bpc))
    other = r // fx.Int32((n - 1) * bpc) * fx.Int32(n * bpc) + pos
    return arith.select(kk < fx.Int32(lb), own, arith.select(kk < fx.Int32(n * lb), other, kk))


@ASTRewriter.transform
def _work_loop(thread_index, block_index, claim_slot, stash, run_item, **c):
    """Claim items (the top ``num_early`` bands from the top down, then band 0 up), then dQ push units in
    sequence order (their own counter), then gather units ``(m-tile, src)``; the first ``num_comm`` blocks
    claim dQ push units first. Every base comes from ``stash`` at its use."""
    nb, ipb, ni, early, qs, n = (
        c["num_bands"],
        c["items_per_band"],
        c["num_items"],
        c["num_early"],
        c["q_split"],
        c["num_ranks"],
    )
    attn, comm = c["mode"] != "comm_only", c["mode"] != "attn_only"
    num_claims = ni + c["g_tiles"] * n
    trace, trace_at = c["trace"], c["trace_item_at"]
    if const_expr(comm and c["num_comm"] > 0):
        if block_index < fx.Int32(c["num_comm"]):
            _dq_units(thread_index, claim_slot, stash, **c)
    # items, then (a second loop: what the dQ push / gather code holds stays off the body's registers)
    # the comm units; claims are ordered, so the unit that ends the first loop starts the second
    unit = claim_unit(opaque_tid(thread_index), _stashed(stash, _S_STATE), claim_slot, counter=STATE_TILE)
    while unit < fx.Int32(ni):
        k = unit // fx.Int32(ipb)
        band = arith.select(k < fx.Int32(early), fx.Int32(nb - 1) - k, k - fx.Int32(early))
        if const_expr(c["local_bands"] > 0):
            band = _local_first_band(band, **c)
        sub = unit % fx.Int32(ipb)
        if const_expr(trace):
            _trace(stash, fx.Int32(trace_at) + unit * fx.Int32(3), thread_index)
            cyc0 = read_shader_clock()
        if const_expr(attn):
            run_item(band, sub % fx.Int32(qs), sub // fx.Int32(qs))
        if const_expr(trace):
            _trace(stash, fx.Int32(trace_at) + unit * fx.Int32(3) + fx.Int32(1), thread_index)
            cyc = read_shader_clock() - cyc0
            if thread_index == fx.Int32(0):
                st(_stashed(stash, _S_TRACE), fx.Int32(c["trace_cyc_at"]) + unit, cyc)
        if const_expr(comm):
            epoch = _stashed(stash, _S_EPOCH)
            _finish_item(
                opaque_tid(thread_index),
                band,
                band * fx.Int32(ipb) + sub,
                claim_slot,
                _stashed(stash, _S_COUNT),
                _region(stash, REGION_BWD_BAND),
                _stashed(stash, _S_STATE),
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
        if const_expr(c["kv_help"]):
            _kv_help(thread_index, claim_slot, stash, **c)
        # every dQ push unit is claimed before this block waits on a gather
        _dq_units(thread_index, claim_slot, stash, **c)
        while unit < fx.Int32(num_claims):
            gg = unit - fx.Int32(ni)
            epoch = _stashed(stash, _S_EPOCH)
            if const_expr(trace):
                _trace(stash, fx.Int32(c["trace_gather_at"]) + gg * fx.Int32(2), thread_index)
            _gather_unit(
                opaque_tid(thread_index),
                gg // fx.Int32(n),
                gg % fx.Int32(n),
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


@ASTRewriter.transform
def _dq_units(thread_index, claim_slot, stash, **c):
    """dQ push units in sequence order, from their own counter, until none are left. ``dq_desc``: in the
    order their rows get ready by item progress: down from the top band of the first wave (the later
    waves' bands are the short top ones), then down from the top; in one wave, down from the band below
    the top, the top band last (with a rotated walk, the odd bands visit their top row on their last
    trip)."""
    nb = c["num_bands"]
    dq_per_band = c["band_rows"] // c["dq_rows"]
    nq = c["S"] // c["dq_rows"]
    trace = c["trace"]
    jj = claim_unit(opaque_tid(thread_index), _stashed(stash, _S_STATE), claim_slot, counter=STATE_DQ)
    while jj < fx.Int32(nq):
        u = jj
        if const_expr(c["dq_desc"]):
            jb = jj // fx.Int32(dq_per_band)
            mid = c["dq_mid"] if c["dq_mid"] < nb else (nb - 1 if c["qdesc_r"] > 1 else nb)
            ub = arith.select(jb < fx.Int32(mid), fx.Int32(mid - 1) - jb, fx.Int32(nb - 1 + mid) - jb)
            u = ub * fx.Int32(dq_per_band) + jj % fx.Int32(dq_per_band)
        epoch = _stashed(stash, _S_EPOCH)
        if const_expr(trace):
            _trace(stash, fx.Int32(c["trace_dq_at"]) + u * fx.Int32(2), thread_index)
        j = u // fx.Int32(dq_per_band)
        part = u % fx.Int32(dq_per_band)
        if const_expr(c["dq_prog"] > 0):
            _wait_progress(opaque_tid(thread_index), _region(stash, REGION_BWD_BAND), j, part, epoch, **c)
        else:
            _wait_bands(opaque_tid(thread_index), _region(stash, REGION_BWD_BAND), j, epoch, **c)
        if const_expr(trace):
            _trace(stash, fx.Int32(c["trace_dqw_at"]) + u, thread_index)
        _push_dq(
            opaque_tid(thread_index),
            j,
            part,
            _region(stash, REGION_BWD_BAND),
            _stashed(stash, _S_PEERS),
            _recv_offset(epoch, c),
            _img_res(stash, c),
            epoch,
            **c,
        )
        if const_expr(trace):
            _trace(stash, fx.Int32(c["trace_dq_at"]) + u * fx.Int32(2) + fx.Int32(1), thread_index)
        jj = claim_unit(opaque_tid(thread_index), _stashed(stash, _S_STATE), claim_slot, counter=STATE_DQ)


TRACE_MAX_BLOCKS = 1024


def trace_layout(M, num_ranks, hq_r, hkv_r, head_dim, dq_rows=None, pull_rows=None, head_dim_v=None):
    """Offsets of the debug timeline (s_memrealtime, 100 MHz): pull unit ends ``[M / pull_rows * n]``, then
    ``[claim, body end, end]`` per item, ``[start, end]`` per dQ push unit ``[S / dq_rows]``
    and per gather unit ``[g_tiles * n]``, stage piece ends ``[m_tiles * STAGE_SPLIT * n]``, block starts
    ``[TRACE_MAX_BLOCKS]``, the dQ push units' band-wait ends, the body's summed dO spin ticks ``[1]``,
    shader-clock cycles of each item's body ``[num_items]``; and the length."""
    S = M * num_ranks
    mla = head_dim_v is not None and head_dim_v != head_dim
    q_split = _plan(S, hq_r, hkv_r, head_dim, head_dim_v)[0]
    nq = S // _dq_rows(hq_r, dq_rows, mla)
    ni = _ceildiv(S, MLA_BAND if mla else BAND) * q_split * hkv_r
    item_at = M // _pull_rows(hq_r, pull_rows) * num_ranks
    dq_at = item_at + 3 * ni
    gather_at = dq_at + 2 * nq
    stage_at = gather_at + 2 * (M // BM * num_ranks)
    start_at = stage_at + M // BM * STAGE_SPLIT * num_ranks
    dqw_at = start_at + TRACE_MAX_BLOCKS
    return item_at, dq_at, gather_at, stage_at, start_at, dqw_at, dqw_at + nq + 1 + ni


@functools.lru_cache(maxsize=32)
def _make_kernel(
    M,
    rank,
    num_ranks,
    hq_r,
    hkv_r,
    head_dim,
    head_dim_v,
    softmax_scale,
    seq_chunk,
    num_pull,
    num_blocks,
    num_early,
    num_prefix,
    num_top,
    dq_rows,
    num_comm,
    local_first,
    pull_rows,
    dq_prog,
    dq_desc,
    kv_help,
    spin_sleep,
    bank_bytes,
    trace,
    mode,
):
    n, d, d_v = num_ranks, head_dim, head_dim_v
    mla = d_v != d
    S = M * n
    cs = seq_chunk if seq_chunk else M
    band = MLA_BAND if mla else BAND
    kv_rows = MLA_PART_ROWS if mla else BM
    reason = a2a_attn_bwd_unsupported(M, n, hq_r, hkv_r, d, d_v, seq_chunk, softmax_scale)
    assert reason is None, reason
    assert not (mla and (dq_prog or dq_desc or kv_help)), "MLA: the default dQ / dK / dV push"
    q_split, dkdv_l, dq_scale = _plan(S, hq_r, hkv_r, d, d_v if mla else None, softmax_scale if mla else None)
    emb = dkdv_l.embed
    assert emb["block_size"] == THREADS and emb["block_kv"] == band and emb["q_split"] == q_split
    n_img = emb.get("a16_imgs", 1)
    m_tiles = M // BM
    g_tiles = M // BM
    nb = _ceildiv(S, band)
    ipb = q_split * hkv_r
    num_items = nb * ipb
    kv_parts = 2 * (band // kv_rows)
    stage_bytes = M * n * hq_r * d_v * 2
    recv_x_bytes = (M * n * hq_r * d * 2, M * n * hkv_r * d * 2, M * n * hkv_r * d_v * 2)
    recv_at = (0, recv_x_bytes[0], recv_x_bytes[0] + recv_x_bytes[1])
    assert stage_bytes + sum(recv_x_bytes) <= bank_bytes
    # dQ recv flags: one per (dq_flag_rows rows, source); the dK / dV ones (per kv_rows rows) follow
    dq_flag_rows = min(dq_rows, BM)
    dq_flags = M // dq_flag_rows
    kv_tiles = M // kv_rows
    reason = _layout_reason(hq_r, hkv_r, d, d_v, pull_rows, dq_rows)
    assert reason is None, reason
    assert m_tiles * STAGE_SPLIT * n <= FLAGS_PER_REGION and (dq_flags + 2 * kv_tiles) * n <= FLAGS_PER_REGION
    assert 0 <= num_comm <= num_pull and num_comm < num_blocks, "comm blocks: some of the pull blocks"
    assert cs % max(emb["block_q"], pull_rows) == 0, "a q block's pull units: one chunk of one source"
    grid = max(num_blocks, num_pull)
    trace_item_at, trace_dq_at, trace_gather_at, trace_stage_at, trace_start_at, trace_dqw_at, trace_len = trace_layout(
        M, n, hq_r, hkv_r, d, dq_rows, pull_rows, d_v
    )
    assert not trace or grid <= TRACE_MAX_BLOCKS
    body = emb["body"]
    assert emb["q_ready"], "the dO readiness hook needs the body's q-block prefetch"
    if dq_prog:
        assert emb["qdesc_r"] > 0 and num_items * WAVES <= BAND_DONE_AT - PROG_AT, "dQ progress: QDESC walk, flags"
        assert band % (emb["block_q"] * q_split) == 0, "the band-relative split walk (_desc_walk)"
    num_early = min(num_early, nb)
    if num_prefix < 0:
        # the first wave's bands below the top ones start on their diagonal q blocks
        num_prefix = _ceildiv(grid, ipb) - num_early
    # local first: the first wave runs this rank's own bands of [0, n * local_bands), whose diagonal rows
    # need no link, and the pull starts on the top rows they go on to; the other bands follow
    bpc = cs // band if cs % band == 0 else 0
    local_bands = _ceildiv(_ceildiv(grid, ipb) - num_early, bpc) * bpc if local_first and bpc else 0
    if local_bands <= 0 or n * local_bands > nb - num_early:
        local_bands = 0
    if local_bands:
        num_prefix = n * local_bands
    prefix_rows = min(min(max(num_prefix, 0), nb - num_early) * band, S)
    prefix_tiles = prefix_rows // pull_rows
    # every source's local rows of the sequence rows [0, prefix_rows)
    prefix_m_tiles = min(_ceildiv(prefix_rows, cs * n) * cs // BM, m_tiles)
    # the top rows every first-wave item goes on to after its diagonal ones: whole chunks of every source
    top_rows = _ceildiv(num_top * band, cs * n) * cs * n if num_top > 0 and not local_bands else 0
    if top_rows // n // BM + prefix_m_tiles > m_tiles:
        top_rows = 0

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
        d_v=d_v,
        band_rows=band,
        kv_rows=kv_rows,
        kv_parts=kv_parts,
        kv_tiles=kv_tiles,
        n_img=n_img,
        hq_r=hq_r,
        hkv_r=hkv_r,
        seq_chunk=cs,
        m_tiles=m_tiles,
        pull_rows=pull_rows,
        p_tiles=M // pull_rows,
        block_q=emb["block_q"],
        g_tiles=g_tiles,
        stage_bytes=stage_bytes,
        recv_x_bytes=recv_x_bytes,
        recv_at=recv_at,
        recv_q_bytes=recv_x_bytes[0],
        recv_kv_at=recv_at[1:],
        recv_kv_bytes=recv_x_bytes[1:],
        num_bands=nb,
        q_split=q_split,
        items_per_band=ipb,
        num_items=num_items,
        num_early=num_early,
        prefix_tiles=prefix_tiles,
        prefix_m_tiles=prefix_m_tiles,
        top_tiles=top_rows // pull_rows,
        top_m_tiles=top_rows // n // BM,
        local_bands=local_bands,
        bands_per_chunk=bpc,
        dq_scale=dq_scale,
        dq_rows=dq_rows,
        dq_prog=dq_prog,
        dq_desc=bool(dq_desc),
        dq_mid=(grid - num_comm) // ipb,
        kv_help=bool(kv_help),
        spin_sleep=spin_sleep,
        kv_claim_at=nb,
        kv_queue_at=nb + nb * kv_parts,
        qdesc_r=emb.get("qdesc_r", 0),
        num_comm=num_comm,
        dq_flag_rows=dq_flag_rows,
        dq_flags=dq_flags,
        mode=mode,
        trace=trace,
        bank_bytes=bank_bytes,
        trace_item_at=trace_item_at,
        trace_dq_at=trace_dq_at,
        trace_dqw_at=trace_dqw_at,
        trace_qwait_at=trace_dqw_at + S // dq_rows,
        trace_cyc_at=trace_dqw_at + S // dq_rows + 1,
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
                    create_buffer_resource(DO, num_records_bytes=S * hq_r * d_v * 2),
                    create_buffer_resource(O, num_records_bytes=S * hq_r * d_v * 2),
                    create_buffer_resource(DELTA, num_records_bytes=S * hq_r * 4),
                    [
                        create_buffer_resource(LSE_IN, num_records_bytes=S * hq_r * 4),
                        create_buffer_resource(LSE, num_records_bytes=S * hq_r * 4),
                    ],
                    epoch,
                    trace_base,
                    **c,
                )

        def run_item(band, split, kvh):
            q_ready = None
            if const_expr(mode != "attn_only"):
                pid = band * fx.Int32(ipb) + kvh * fx.Int32(q_split) + split
                q_ready = functools.partial(_q_ready, stash=stash, tid=thread_index, band=band, pid=pid, **c)
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

        _work_loop(thread_index, block_index, claim_slot, stash, run_item, **c)

        exit_and_bump_epoch(
            thread_index, _stashed(stash, _S_STATE), _stashed(stash, _S_EPOCH), grid, extra_counters=(STATE_ASSIST, STATE_PULLED, STATE_DQ, STATE_KVQ, STATE_KVH)
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
    head_dim_v: fx.Constexpr[int],
    softmax_scale: fx.Constexpr[float],
    seq_chunk: fx.Constexpr[int],
    num_pull: fx.Constexpr[int],
    num_blocks: fx.Constexpr[int],
    num_early: fx.Constexpr[int],
    num_prefix: fx.Constexpr[int],
    num_top: fx.Constexpr[int],
    dq_rows: fx.Constexpr[int],
    num_comm: fx.Constexpr[int],
    local_first: fx.Constexpr[int],
    pull_rows: fx.Constexpr[int],
    dq_prog: fx.Constexpr[int],
    dq_desc: fx.Constexpr[int],
    kv_help: fx.Constexpr[int],
    spin_sleep: fx.Constexpr[int],
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
        head_dim_v,
        softmax_scale or None,
        seq_chunk,
        num_pull,
        num_blocks,
        num_early,
        num_prefix,
        num_top,
        dq_rows,
        num_comm,
        local_first,
        pull_rows,
        dq_prog,
        dq_desc,
        kv_help,
        spin_sleep,
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


def _buffers(S, hq_r, hkv_r, d, d_v, q_split, n_img, device):
    """dO and delta of this rank's heads, the dK / dV slots, the dQ images (zero between calls: its dQ
    push units zero them), the band item counts (zero between calls: a band's last item resets its count)
    and the scaled LSE; reused across calls."""
    key = (S, hq_r, hkv_r, d, d_v, q_split, n_img, device)
    bufs = _BUFS.get(key)
    if bufs is None:
        _BUFS.clear()
        mla = d_v != d
        nb = _ceildiv(S, MLA_BAND if mla else BAND)
        kv_parts = 2 * MLA_BAND // MLA_PART_ROWS if mla else KV_PARTS
        bufs = (
            torch.empty(S, hq_r, d_v, device=device, dtype=torch.bfloat16),
            torch.empty(hq_r, S, device=device, dtype=torch.float32),
            torch.empty(q_split, S, hkv_r, d, device=device, dtype=torch.bfloat16),
            torch.empty(q_split, S, hkv_r, d_v, device=device, dtype=torch.bfloat16),
            torch.zeros(n_img * S * hq_r * d, device=device, dtype=torch.bfloat16),
            # item counts, then the dK / dV part claims and the finished-band queue (epoch-stamped)
            torch.zeros(nb * (2 + kv_parts), device=device, dtype=torch.int64),
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
    num_pull=None,
    num_blocks=None,
    num_early=0,
    num_prefix=-1,
    num_top=None,
    dq_rows=None,
    num_comm=None,
    local_first=None,
    pull_rows=None,
    dq_prog=0,
    dq_desc=False,
    kv_help=False,
    spin_sleep=1,
    trace=None,
    mode="full",
    softmax_scale=None,
):
    """``dout`` ``[M, n * hq_r, d_v]``: dO for this rank's tokens with every head (batch 1); ``q`` / ``k``
    ``[S, h_r, d]``, ``v`` / ``o`` ``[S, h_r, d_v]`` and the natural-log ``lse`` ``[S, hq_r]`` fp32 of this
    rank's heads over the sequence, as the fused forward leaves them. Writes ``dq`` ``[M, n * hq_r, d]``,
    ``dk`` ``[M, n * hkv_r, d]`` and ``dv`` ``[M, n * hkv_r, d_v]`` for this rank's tokens. ``d_v != d``:
    the MLA shapes (``d`` 192, ``d_v`` 128) on the gfx950 MLA dK / dV body, with ``softmax_scale``
    (default ``d ** -0.5``; the GQA path bakes the default). The bank must hold
    ``a2a_attn_bwd_bank_bytes``.
    ``num_pull``: pull blocks (default: by ``n``, see ``_link_plan``). ``dq_rows``: sequence rows of one
    dQ push unit (default: by bytes, see ``_dq_rows``). ``num_comm``: of the pull blocks, how many push dQ
    units as their bands finish instead of joining the items first (default: by ``n``).
    ``local_first``: the first wave starts on this rank's own bands (needs ``seq_chunk % 256 == 0``;
    default: by ``n`` and ``S``).
    ``pull_rows``: sequence rows of one pull unit (default: by bytes, see ``_pull_rows``).
    ``num_top``: bands at the top of the sequence the pull and the stage take right after the prefix, the
    rows the first wave goes on to after its diagonal ones (default: by ``n`` and ``S``).
    ``dq_prog``: a dQ push unit waits on the items' published progress (every ``dq_prog`` q blocks)
    instead of whole bands; ``dq_desc``: dQ push units from the top block down. ``kv_help``: blocks out of
    items (but the dQ pushers) take parts of the finished bands' dK / dV push from their last items.
    ``spin_sleep``: s_sleep count between flag polls of the comm units.

    ``mode`` (measurement only): ``"attn_only"`` runs the backward items on the dO / delta of the previous
    call (no exchange, no waits, no folds, pushes or gathers; LSE prescaled and the dQ image zeroed by
    separate launches); ``"comm_only"`` runs every step but the items' attention body."""
    n = workspace.num_ranks
    M, H, d_v = dout.shape[0], dout.shape[-2], dout.shape[-1]
    S, hq_r, hkv_r, d = M * n, H // n, k.shape[-2], q.shape[-1]
    mla = d_v != d
    assert softmax_scale is None or mla, "the GQA backward bakes softmax_scale = d ** -0.5"
    assert dout.dtype == torch.bfloat16 and H % n == 0 and q.shape == (S, hq_r, d) and o.shape == (S, hq_r, d_v)
    assert k.shape == (S, hkv_r, d) and v.shape == (S, hkv_r, d_v)
    assert all(t.is_contiguous() for t in (dout, q, k, v, o, lse, dq, dk, dv))
    assert dq.shape == (M, H, d) and dk.shape == (M, n * hkv_r, d) and dv.shape == (M, n * hkv_r, d_v)
    plan = _plan(S, hq_r, hkv_r, d, d_v if mla else None, softmax_scale if mla else None)
    q_split, n_img = plan[0], plan[1].embed.get("a16_imgs", 1)
    dq_rows = _dq_rows(hq_r, dq_rows, mla)
    pull_rows = _pull_rows(hq_r, pull_rows)
    auto_pull, auto_comm, auto_local, auto_top = _link_plan(n, S, mla)
    num_top = auto_top if num_top is None else num_top
    num_pull = auto_pull if num_pull is None else num_pull
    num_comm = auto_comm if num_comm is None else num_comm
    local_first = (auto_local and not mla) if local_first is None else local_first
    if num_blocks is None:
        num_blocks = torch.cuda.get_device_properties(dout.device).multi_processor_count
    stream = torch.cuda.current_stream()
    key = (S, hq_r, hkv_r, d, d_v, q_split, n_img, dout.device)
    do_h, delta, ws_dk, ws_dv, img, counts, lsef = _buffers(*key)
    # the MLA body scales Q itself
    qf = q if mla else _bwd._prescale_q(q, 1.0 / math.sqrt(d))
    if mode == "attn_only":
        # no pull units to write it
        lsef = _bwd._prescale_lse(lse.view(1, S, hq_r).permute(0, 2, 1), stream).reshape(-1)
    if mode == "attn_only" or key in _IMG_DIRTY:
        img.zero_()
    if mode == "attn_only":
        _IMG_DIRTY.add(key)
    else:
        _IMG_DIRTY.discard(key)
    with CompilationContext.compile_hints(plan[1].embed["hints"]):
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
            d_v,
            0.0 if softmax_scale is None else float(softmax_scale),
            seq_chunk,
            num_pull,
            num_blocks,
            num_early,
            num_prefix,
            num_top,
            dq_rows,
            num_comm,
            int(local_first),
            pull_rows,
            int(dq_prog),
            int(dq_desc),
            int(kv_help),
            spin_sleep,
            workspace.bank_bytes,
            trace is not None,
            mode,
            stream,
        )
