###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Both zigzag chunks' dK / dV for the fused all-gather-KV attention backward in one persistent kernel,
the remote keys' gradients pushed to their owners from inside it.

Rank ``r`` holds q0 (its first chunk, ``c`` rows, keys ``keys0`` = the ``len0 = (r + 1) c`` rows from
``rem - r c`` on) and q1 (its second chunk, every key ``[remote | own first | own second]``, ``skv``
rows). Every resident work-group claims work items from per-XCD queues (see ``agkv_fused_fwd._claim``):
one item is one work-group of the flux dK / dV kernel, ``(segment, kv tile, kv head, q split, batch)``,
run through the flux body with that segment's tensors (one compiled body serves both: q0's plan is
q1's). An item stores its split's dK / dV slot and counts itself on its group ``(band, kv head,
batch)`` of the ``skv`` axis; the group's last item sums the group's slots (q1's splits, and q0's
when the band lies in ``keys0``) and stores the rows

    remote key chunk k  ->  slot (half, rank) of the owner's backward bank (system-scope stores)
    own chunk           ->  the local own-gradient buffer [own first | own second]

then counts the group on its chunk; a chunk's last group raises its flag: a remote chunk's in the
owner's backward flags (slot (half, rank)), an own chunk's in this rank's own slot of them. No item
waits on another, so the claims cannot deadlock.

The reduce-scatter sum of this rank's received gradients runs as reduce units after every item is
claimed: unit (K / V, half, piece) waits for the flags of the half's senders (every rank for the
first half, ranks ``0..r`` for the second; this rank's own slot is its own chunk), then sums their
slots in sender order, the own term at the rank's place (``mega``'s reduce, number for number). dQ
partials stay the flux kernel's, reduced after the launch.

Per-call addresses ride a small int64 table:

    [epoch, (dK destination, dV destination, flag address) per key chunk,
     backward bank, flag array, dK out, dV out, own dK, own dV]

where a chunk's destinations are the addresses of ITS first row (batch 0, kv head 0).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import range_constexpr

from primus_turbo.flydsl.attention import flash_attn_bwd_flux as _flux
from primus_turbo.flydsl.attention.agkv_comm_kernel import _spin
from primus_turbo.flydsl.attention.agkv_fused_fwd import (
    NUM_XCDS,
    SCHED_HEAD,
    STATE_LEN,
    _claim,
    _opaque_tid,
    _stash_bases,
    _stashed,
)
from primus_turbo.flydsl.deep_ep.common import _uniform_i64, ld_lds, st_lds
from primus_turbo.flydsl.fused_collective_gemm.common import SYS_NT, i64_base
from primus_turbo.flydsl.utils.buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from primus_turbo.flydsl.utils.prims import _readfirstlane_i32, atomic_add, cast, ld, memory_fence, st

P_EPOCH, P_CHUNKS = 0, 1
CHUNK_FIELDS = 3
RS_BANK, RS_FLAGS, RS_OUT, RS_OWN = 0, 1, 2, 4
# the reduce units' claim counter, after the attention queues' state
ST_TAIL = STATE_LEN
RS_ITERS = 4

# host table copies in flight per (device, length) before the oldest is reused
_RING = 8

_SEG = ("Q", "K", "V", "DO", "LSE", "DELTA", "DK", "DV", "WSQ")
_STASH = tuple(f"{name}{i}" for i in range(2) for name in _SEG) + ("SCHED", "STATE", "COUNTS", "PARAMS")


def plan(B, c, n, r, Hq, Hkv, D, scale):
    """The shared flux body of both segments: ``(dkdv launcher, odo launcher, q_split, block_kv, dQ
    interleave)``, or None when q1's flux plan is not one plain launch (a16, band groups, rings)."""
    rem, len0 = (2 * n - 2 - r) * c, (r + 1) * c
    skv = rem + 2 * c
    p = _flux._dense_plan(B, c, skv, Hq, Hkv, D, scale, -1, True, False)
    _, q_split, bkv, nb1, a16, _, ilv, ring, pair, span = p[:10]
    if a16 or ring or pair or span or c % bkv or skv * B * Hkv * D * 2 >= 1 << 31:
        return None
    ilv = min(ilv, _flux._wsq_ilv(len0 // bkv, B, c, Hq * D))
    dkdv_l, odo_l = _flux._get_bwd(
        Hq, Hkv, D, scale, -1, q_split, bkv, batch_size=B, sbhd=True, square=False, wsq_ilv=ilv, causal=True
    )
    return dkdv_l, odo_l, q_split, bkv, ilv


@ASTRewriter.transform
def _finish(tid, band, head, batch, slot, stash, c):
    """Count the item on its group; the group's last item folds the group's slots and stores them."""
    nb, hkv, qs, lo0, nb0 = c["nb"], c["hkv"], c["q_split"], c["lo0"], c["nb0"]
    group = (batch * fx.Int32(hkv) + head) * fx.Int32(nb) + band
    in0 = (band >= fx.Int32(lo0)) & (band < fx.Int32(lo0 + nb0))
    expected = in0.select(fx.Int32(2 * qs), fx.Int32(qs))
    # this item's slot stores retire before its count releases them
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    counts = _stashed(stash, _STASH.index("COUNTS"))
    if tid == fx.Int32(0):
        old = atomic_add(counts, group, fx.Int64(1), scope="agent", order="release")
        st_lds(slot, 1, (old == cast(expected, fx.T.i64) - fx.Int64(1)).select(fx.Int32(1), fx.Int32(0)))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    last = _readfirstlane_i32(ld_lds(slot, 1))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if last == fx.Int32(1):
        memory_fence(order="acquire", scope="agent")
        if tid == fx.Int32(0):
            st(counts, group, fx.Int64(0))
        _fold(tid, band, head, batch, in0, stash, c)


@ASTRewriter.transform
def _fold(tid, band, head, batch, in0, stash, c):
    """The group's rows: q1's slots plus q0's (none outside ``keys0``: their descriptors are empty), fp32,
    stored to the chunk's destination; then the chunk's count, whose last raises a remote owner's flag."""
    bkv, d, tok, qs, skv, len0, lo0 = c["bkv"], c["d"], c["tok"], c["q_split"], c["skv"], c["len0"], c["lo0"]
    block, cc, per_chunk, ng = c["block"], c["c"], c["per_chunk"], c["ngroups"]
    vecs_per_row = d // 8
    f32x8, bf16x8 = fx.T.vec(8, fx.T.f32), fx.T.vec(8, fx.T.bf16)
    col0 = cast((batch * fx.Int32(c["hkv"]) + head) * fx.Int32(d * 2), fx.T.i64)
    row0 = band * fx.Int32(bkv)
    chunk = row0 // fx.Int32(cc)
    params = _stashed(stash, _STASH.index("PARAMS"))
    entry = fx.Int32(P_CHUNKS) + chunk * fx.Int32(CHUNK_FIELDS)
    band_bytes = fx.Int64(bkv * tok)
    rows0 = in0.select(fx.Int32(bkv * tok), fx.Int32(0))
    off0 = cast(in0.select(band - fx.Int32(lo0), fx.Int32(0)), fx.T.i64) * band_bytes
    for t in range_constexpr(2):
        ws1 = _stashed(stash, _STASH.index(("DK1", "DV1")[t])) + col0 + cast(band, fx.T.i64) * band_bytes
        ws0 = _stashed(stash, _STASH.index(("DK0", "DV0")[t])) + col0 + off0
        terms = [
            create_buffer_resource_from_addr(ws1 + fx.Int64(s * skv * tok), num_records_bytes=bkv * tok)
            for s in range(qs)
        ] + [
            create_buffer_resource_from_addr(ws0 + fx.Int64(s * len0 * tok), num_records_bytes=rows0)
            for s in range(qs)
        ]
        dst = (
            _param(params, entry + fx.Int32(t))
            + col0
            + cast(row0 - chunk * fx.Int32(cc), fx.T.i64) * fx.Int64(tok)
        )
        dst_res = create_buffer_resource_from_addr(dst, num_records_bytes=bkv * tok)
        idxs = []
        for i in range_constexpr(bkv * vecs_per_row // block):
            vec = tid + fx.Int32(i * block)
            idxs.append(
                (vec // fx.Int32(vecs_per_row)) * fx.Int32(tok // 2)
                + vec % fx.Int32(vecs_per_row) * fx.Int32(8)
            )
        loads = [[buffer_load(res, idx, vec_width=8, dtype=fx.T.bf16) for res in terms] for idx in idxs]
        for idx, parts in zip(idxs, loads):
            acc = fx.arith.extf(f32x8, parts[0])
            for part in parts[1:]:
                acc = fx.arith.addf(acc, fx.arith.extf(f32x8, part))
            buffer_store(fx.arith.trunc_f(bf16x8, acc), dst_res, idx, cache_modifier=SYS_NT)
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    if tid == fx.Int32(0):
        counts = _stashed(stash, _STASH.index("COUNTS"))
        done = atomic_add(counts, fx.Int32(ng) + chunk, fx.Int64(1), scope="agent", order="release")
        if done == fx.Int64(per_chunk - 1):
            st(counts, fx.Int32(ng) + chunk, fx.Int64(0))
            memory_fence(order="release", scope="sys")
            flag = _param(params, entry + fx.Int32(2))
            st(flag, fx.Int32(0), _param(params, fx.Int32(P_EPOCH)), scope="sys", order="relaxed")


def _param(params, i):
    return _uniform_i64(ld(params, i, dtype=fx.T.i64))


@ASTRewriter.transform
def _run_item(body, segs, lens, stash, tid, seg, tile, head, split, batch):
    """One flux dK / dV work-group of segment ``seg`` (both segments inline the same body), its tensor
    addresses read from the stash where the body uses them."""
    ids = (fx.Index(tile), fx.Index(head), fx.Index(split), fx.Index(batch))
    sq, len0, skv = lens
    bases = [{name: _reader(stash, f"{name}{i}") for name in _SEG} for i in range(2)]
    if seg == fx.Int32(0):
        body(*segs[0], fx.Int32(sq), fx.Int32(len0), fx.Int32(0), ids=ids, tid=tid, bases=bases[0])
    else:
        body(*segs[1], fx.Int32(sq), fx.Int32(skv), fx.Int32(0), ids=ids, tid=tid, bases=bases[1])


@ASTRewriter.transform
def _rs_unit(tid, unit, stash, c):
    """Reduce unit ``unit``: ``RS_ITERS`` vectors per thread of one (K / V, half) output segment."""
    n, r, block, pieces, seg_bytes = c["n"], c["r"], c["block"], c["rs_pieces"], c["c"] * c["tok"]
    f32x8, bf16x8 = fx.T.vec(8, fx.T.f32), fx.T.vec(8, fx.T.bf16)
    params = _stashed(stash, _STASH.index("PARAMS"))
    rs = fx.Int32(c["rs_off"])
    seg = unit // fx.Int32(pieces)
    elem0 = (unit % fx.Int32(pieces)) * fx.Int32(block * 8 * RS_ITERS)
    epoch = _param(params, fx.Int32(P_EPOCH))
    for i in range_constexpr(4):
        t, h = i // 2, i % 2
        senders = list(range(n) if h == 0 else range(r + 1))
        if seg == fx.Int32(i):
            if tid == fx.Int32(0):
                flags = _param(params, rs + fx.Int32(RS_FLAGS))
                _spin(flags, [fx.Int32(c["flag_halves"][h] + s) for s in senders], epoch, r, "agkv_fused_bwd")
            fx.rocdl.s_waitcnt(0)
            fx.gpu.barrier()
            memory_fence(order="acquire", scope="sys")
            bank = _param(params, rs + fx.Int32(RS_BANK)) + fx.Int64(t * c["v_off"])
            own = _param(params, rs + fx.Int32(RS_OWN + t)) + fx.Int64(h * seg_bytes)
            srcs = [
                (own, 0) if s == r else (bank + fx.Int64((h * n + s) * seg_bytes), SYS_NT) for s in senders
            ]
            res = [(create_buffer_resource_from_addr(a, num_records_bytes=seg_bytes), cm) for a, cm in srcs]
            out = _param(params, rs + fx.Int32(RS_OUT + t)) + fx.Int64(h * seg_bytes)
            out_res = create_buffer_resource_from_addr(out, num_records_bytes=seg_bytes)
            idxs = [elem0 + (fx.Int32(it * block) + tid) * fx.Int32(8) for it in range(RS_ITERS)]
            vals = [
                [buffer_load(rs_, e, vec_width=8, dtype=fx.T.bf16, cache_modifier=cm) for rs_, cm in res]
                for e in idxs
            ]
            for e, parts in zip(idxs, vals):
                acc = fx.arith.extf(f32x8, parts[0])
                for part in parts[1:]:
                    acc = fx.arith.addf(acc, fx.arith.extf(f32x8, part))
                buffer_store(fx.arith.trunc_f(bf16x8, acc), out_res, e)


@ASTRewriter.transform
def _claim_tail(tid, stash, slot):
    if tid == fx.Int32(0):
        state = _stashed(stash, _STASH.index("STATE"))
        st_lds(slot, 2, cast(atomic_add(state, fx.Int32(ST_TAIL), fx.Int64(1), scope="agent"), fx.T.i32))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    unit = _readfirstlane_i32(ld_lds(slot, 2))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    return unit


@ASTRewriter.transform
def _rs_loop(thread_index, slot, stash, c):
    """Reduce units, claimed once every attention item is: an item never waits, so every flag a unit
    waits for comes up."""
    num = 4 * c["rs_pieces"]
    unit = _claim_tail(_opaque_tid(thread_index), stash, slot)
    while unit < fx.Int32(num):
        _rs_unit(_opaque_tid(thread_index), unit, stash, c)
        unit = _claim_tail(_opaque_tid(thread_index), stash, slot)


@ASTRewriter.transform
def _exit_all(thread_index, state, grid):
    """``_exit`` over the reduce units' counter too."""
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        arrived = atomic_add(state, fx.Int32(0), fx.Int64(1), scope="agent")
        if arrived == fx.Int64(grid - 1):
            for i in range_constexpr(ST_TAIL + 1):
                st(state, fx.Int32(i), fx.Int64(0))


def _reader(stash, name):
    return lambda: _stashed(stash, _STASH.index(name))


@ASTRewriter.transform
def _work_loop(thread_index, slot, stash, num_sched, run_item, c):
    """Claim items until none are left; entry ``((((seg * B + batch) * Hkv + head) * nb + tile) * q_split
    + split``, ``tile`` local to its segment."""
    nb, hkv, b, qs, lo0 = c["nb"], c["hkv"], c["batch"], c["q_split"], c["lo0"]
    unit = _claim(
        _opaque_tid(thread_index),
        _stashed(stash, _STASH.index("STATE")),
        slot,
        _stashed(stash, _STASH.index("SCHED")),
        num_sched,
    )
    while unit < fx.Int32(num_sched):
        sched = _stashed(stash, _STASH.index("SCHED"))
        e = _readfirstlane_i32(cast(ld(sched, unit + fx.Int32(SCHED_HEAD), dtype=fx.T.i64), fx.T.i32))
        split = e % fx.Int32(qs)
        rest = e // fx.Int32(qs)
        tile = rest % fx.Int32(nb)
        rest = rest // fx.Int32(nb)
        head = rest % fx.Int32(hkv)
        rest = rest // fx.Int32(hkv)
        batch = rest % fx.Int32(b)
        seg = rest // fx.Int32(b)
        run_item(seg, tile, head, split, batch)
        band = (seg == fx.Int32(0)).select(tile + fx.Int32(lo0), tile)
        _finish(_opaque_tid(thread_index), band, head, batch, slot, stash, c)
        # the next item's LDS writes wait for this one's last reads
        fx.gpu.barrier()
        unit = _claim(
            _opaque_tid(thread_index),
            _stashed(stash, _STASH.index("STATE")),
            slot,
            _stashed(stash, _STASH.index("SCHED")),
            num_sched,
        )


@functools.lru_cache(maxsize=32)
def _build(Hq, Hkv, D, scale, n, r, c, B, num_sched, grid, v_off, flag_halves):
    dkdv_l, _, q_split, bkv, _ = plan(B, c, n, r, Hq, Hkv, D, scale)
    emb = dkdv_l.embed
    assert emb["q_split"] == q_split and emb["n_qsp"] == q_split and emb["block_kv"] == bkv
    body, block = emb["body"], emb["block_size"]
    rem, len0 = (2 * n - 2 - r) * c, (r + 1) * c
    skv = rem + 2 * c
    nb = skv // bkv
    consts = dict(
        nb=nb,
        nb0=len0 // bkv,
        lo0=(rem - r * c) // bkv,
        hkv=Hkv,
        batch=B,
        q_split=q_split,
        bkv=bkv,
        d=D,
        tok=B * Hkv * D * 2,
        skv=skv,
        len0=len0,
        c=c,
        block=block,
        nrem=rem // c,
        per_chunk=c // bkv * Hkv * B,
        ngroups=nb * Hkv * B,
        n=n,
        r=r,
        rs_off=P_CHUNKS + CHUNK_FIELDS * (2 * n - r),
        rs_pieces=-(-(c * B * Hkv * D) // (block * 8 * RS_ITERS)),
        v_off=v_off,
        flag_halves=flag_halves,
    )
    assert bkv * (D // 8) % block == 0

    @fx.struct
    class SharedStorage:
        claim: fx.Array[fx.Int32, 4, 16]
        stash: fx.Array[fx.Int32, 2 * len(_STASH), 16]

    @flyc.kernel(name="agkv_fused_bwd_kernel", known_block_size=[block, 1, 1])
    def kernel(
        Q0: fx.Tensor,
        K0: fx.Tensor,
        V0: fx.Tensor,
        DO0: fx.Tensor,
        LSE0: fx.Tensor,
        DELTA0: fx.Tensor,
        WSK0: fx.Tensor,
        WSV0: fx.Tensor,
        WSQ0: fx.Tensor,
        Q1: fx.Tensor,
        K1: fx.Tensor,
        V1: fx.Tensor,
        DO1: fx.Tensor,
        LSE1: fx.Tensor,
        DELTA1: fx.Tensor,
        WSK1: fx.Tensor,
        WSV1: fx.Tensor,
        WSQ1: fx.Tensor,
        PH: fx.Tensor,
        SCHED: fx.Tensor,
        STATE: fx.Tensor,
        COUNTS: fx.Tensor,
        PARAMS: fx.Tensor,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        slot = fx.Int32(fx.ptrtoint(lds.claim.ptr))
        stash = fx.Int32(fx.ptrtoint(lds.stash.ptr))
        thread_index = cast(fx.thread_idx.x, fx.T.i32)
        segs = (
            (Q0, K0, V0, DO0, LSE0, DELTA0, WSK0, WSV0, PH, PH, WSQ0),
            (Q1, K1, V1, DO1, LSE1, DELTA1, WSK1, WSV1, PH, PH, WSQ1),
        )
        tensors = dict(SCHED=SCHED, STATE=STATE, COUNTS=COUNTS, PARAMS=PARAMS)
        for i in range_constexpr(2):
            for j in range_constexpr(len(_SEG)):
                tensors[f"{_SEG[j]}{i}"] = segs[i][(0, 1, 2, 3, 4, 5, 6, 7, 10)[j]]
        _stash_bases(thread_index, stash, [i64_base(tensors[name]) for name in _STASH])

        def run_item(seg, tile, head, split, batch):
            _run_item(
                body, segs, (c, len0, skv), stash, _opaque_tid(thread_index), seg, tile, head, split, batch
            )

        _work_loop(thread_index, slot, stash, num_sched, run_item, consts)
        _rs_loop(thread_index, slot, stash, consts)
        _exit_all(thread_index, _stashed(stash, _STASH.index("STATE")), grid)

    allocator, value_attrs = emb["allocator"], emb["value_attrs"]

    @flyc.jit
    def launch(
        Q0: fx.Tensor,
        K0: fx.Tensor,
        V0: fx.Tensor,
        DO0: fx.Tensor,
        LSE0: fx.Tensor,
        DELTA0: fx.Tensor,
        WSK0: fx.Tensor,
        WSV0: fx.Tensor,
        WSQ0: fx.Tensor,
        Q1: fx.Tensor,
        K1: fx.Tensor,
        V1: fx.Tensor,
        DO1: fx.Tensor,
        LSE1: fx.Tensor,
        DELTA1: fx.Tensor,
        WSK1: fx.Tensor,
        WSV1: fx.Tensor,
        WSQ1: fx.Tensor,
        PH: fx.Tensor,
        SCHED: fx.Tensor,
        STATE: fx.Tensor,
        COUNTS: fx.Tensor,
        PARAMS: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        # the body's LDS lives in its builder's allocator
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        kernel(
            Q0,
            K0,
            V0,
            DO0,
            LSE0,
            DELTA0,
            WSK0,
            WSV0,
            WSQ0,
            Q1,
            K1,
            V1,
            DO1,
            LSE1,
            DELTA1,
            WSK1,
            WSV1,
            WSQ1,
            PH,
            SCHED,
            STATE,
            COUNTS,
            PARAMS,
            value_attrs=value_attrs,
        ).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch, [None], emb["hints"], consts


def resident_blocks(dkdv_l, device):
    """Work-groups of the persistent kernel one GPU holds at once (waves per EU and LDS bound)."""
    emb = dkdv_l.embed
    waves = emb["waves_per_eu"] * 4 * 64 // emb["block_size"]
    lds = 160 * 1024 // (emb["lds_bytes"] + 256)
    return torch.cuda.get_device_properties(device).multi_processor_count * max(1, min(waves, lds))


@functools.lru_cache(maxsize=64)
def schedule(n, r, c, B, Hkv, q_split, bkv, device):
    """Claim order: every item longest first (its visible q rows), an item of (batch, kv head) ``g`` in
    the queue of XCD ``g`` (spread over ``NUM_XCDS // groups`` XCDs when there are fewer groups)."""
    rem, len0 = (2 * n - 2 - r) * c, (r + 1) * c
    skv = rem + 2 * c
    nb = skv // bkv
    items = []
    for seg, (tiles, offset) in enumerate(((len0 // bkv, len0 - c), (nb, skv - c))):
        for b in range(B):
            for h in range(Hkv):
                for tile in range(tiles):
                    # q row i sees key k when k <= i + offset
                    rows = min(c, max(0, c + offset - tile * bkv))
                    for s in range(q_split):
                        e = (((seg * B + b) * Hkv + h) * nb + tile) * q_split + s
                        items.append((-rows, len(items), e, b * Hkv + h))
    items.sort()
    groups = B * Hkv
    spread = max(1, NUM_XCDS // groups)
    queues = [[] for _ in range(NUM_XCDS)]
    seen = {}
    for _, _, e, g in items:
        k = seen.get(g, 0)
        seen[g] = k + 1
        queues[(g * spread + k % spread) % NUM_XCDS].append(e)
    head, flat = [], []
    for queue in queues:
        head += [len(flat), len(queue)]
        flat += queue
    return torch.tensor(head + flat, dtype=torch.int64, device=device), len(flat)


_RINGS = {}


def params_table(values, device):
    """``values`` as a device int64 tensor, copied from a pinned ring without a host synchronize."""
    key = (device, len(values))
    ring = _RINGS.get(key)
    if ring is None:
        ring = _RINGS[key] = [
            [
                torch.empty(len(values), dtype=torch.int64, pin_memory=True),
                torch.empty(len(values), dtype=torch.int64, device=device),
                torch.cuda.Event(),
            ]
            for _ in range(_RING)
        ] + [0]
    i = ring[-1]
    ring[-1] = (i + 1) % _RING
    host, dev, done = ring[i]
    done.synchronize()
    host.copy_(torch.tensor(values, dtype=torch.int64))
    dev.copy_(host, non_blocking=True)
    done.record()
    return dev


_STATES = {}
_COUNTS = {}


def backward(dout, q, k, v, out, lse0, lse1, scale, n, r, table, dq, v_off, flag_halves):
    """dQ of both chunks into ``dq``; dK / dV out through ``table`` (see the module doc). ``k`` / ``v`` are
    the saved ``[remote | own first | own second]`` keys of q1; ``v_off`` is the V region's byte offset
    in a backward bank, ``flag_halves`` the flag array indices of sender 0 per half."""
    S2, B, Hq, D = q.shape
    c = S2 // 2
    skv, _, Hkv, _ = k.shape
    rem, len0 = (2 * n - 2 - r) * c, (r + 1) * c
    keys0 = slice(rem - r * c, rem + c)
    dkdv_l, odo_l, q_split, bkv, ilv = plan(B, c, n, r, Hq, Hkv, D, scale)
    st_ = torch.cuda.current_stream()
    dev = q.device
    qf = _flux._prescale_q(q, scale)
    half = c * B * Hq * D
    deltas = [torch.empty(B, Hq, c, device=dev, dtype=torch.float32) for _ in range(2)]
    for i in range(2):
        rows = slice(i * c, (i + 1) * c)
        odo_l(out[rows].reshape(-1), dout[rows].reshape(-1), deltas[i].reshape(-1), B, c, st_)
    lses = [_flux._prescale_lse(lse.view(B, c, Hq).permute(0, 2, 1), st_).reshape(-1) for lse in (lse0, lse1)]
    lens = (len0, skv)
    ws_kv = [
        [torch.empty(q_split, kv_len, B, Hkv, D, device=dev, dtype=k.dtype).reshape(-1) for _ in range(2)]
        for kv_len in lens
    ]
    pad = _flux._WSQ_BAND_PAD if D == 128 else 0
    ws_dq = [_flux._dq_partial_ws(kv_len // bkv, B, c, Hq * D, dev, q.dtype, pad, ilv)[0] for kv_len in lens]
    ph = _flux._cu_placeholder(dev)
    sched, num_sched = schedule(n, r, c, B, Hkv, q_split, bkv, dev)
    grid = resident_blocks(dkdv_l, dev)
    state = _STATES.get(dev)
    if state is None:
        state = _STATES[dev] = torch.zeros(ST_TAIL + 1, dtype=torch.int64, device=dev)
    nb = skv // bkv
    ckey = (dev, nb * Hkv * B + skv // c)
    counts = _COUNTS.get(ckey)
    if counts is None:
        counts = _COUNTS[ckey] = torch.zeros(ckey[1], dtype=torch.int64, device=dev)
    params = params_table(table, dev)
    launch, compiled, hints, _ = _build(Hq, Hkv, D, scale, n, r, c, B, num_sched, grid, v_off, flag_halves)
    args = (
        qf[:half],
        k[keys0].reshape(-1),
        v[keys0].reshape(-1),
        dout[:c].reshape(-1),
        lses[0],
        deltas[0].reshape(-1),
        ws_kv[0][0],
        ws_kv[0][1],
        ws_dq[0][0, 0].reshape(-1),
        qf[half:],
        k.reshape(-1),
        v.reshape(-1),
        dout[c:].reshape(-1),
        lses[1],
        deltas[1].reshape(-1),
        ws_kv[1][0],
        ws_kv[1][1],
        ws_dq[1][0, 0].reshape(-1),
        ph,
        sched,
        state,
        counts,
        params,
        st_,
    )
    if compiled[0] is None:
        with CompilationContext.compile_hints(hints):
            compiled[0] = flyc.compile(launch, *args)
    else:
        compiled[0](*args)
    for i, kv_len in enumerate(lens):
        _flux._reduce_dq_partials(
            ws_dq[i],
            dq[i * c : (i + 1) * c],
            bkv,
            Hq,
            D,
            dkdv_l.dq_scale,
            st_,
            causal_offset=kv_len - c,
            window_left=-1,
            causal=True,
            n_bands=kv_len // bkv,
            sbhd=True,
            ph=ph,
        )
    return dq


def supported(B, c, n, r, Hq, Hkv, D, scale, deterministic):
    return not deterministic and plan(B, c, n, r, Hq, Hkv, D, scale) is not None
