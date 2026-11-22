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

"""Communication kernels of the all-gather-KV attention's ``mega`` mode (FlyDSL, single node).

The attention kernels stay the FlyDSL ones; these move K / V and their gradients between ranks
with compute-unit stores over XGMI into the peers' symmetric buffers, MegaMoE style:

``push``
    walks a table of copy units ``(source, source offset, destination rank, destination offset, bytes,
    group, group size, flag)``; a unit's stores are system scope and retired before the block counts the
    unit into its group, and the group's last unit raises the destination's flag to ``epoch``. A launch
    can store the sum of each source and a second tensor instead (two partial gradients of one slot).
``wait``
    holds the stream until a set of this rank's flags reach ``epoch``. One block per XCD, each ending
    with a system-scope acquire, so no XCD's L2 keeps lines of a bank a peer rewrote.
``reduce``
    sums the received partial dK / dV slots and the own ones (fp32, sender order; up to two own terms
    for the first half) into the final dK / dV.
``merge``
    combines two partial attention outputs over disjoint keys by their lse, in one pass.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.math as fmath
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr

from primus_turbo.flydsl.deep_ep.common import _asm, _uniform_i64
from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.fused_collective_gemm.common import (
    BLOCK_THREADS,
    NUM_XCDS,
    SYS_NT,
    VEC_I32,
    copy_rows,
    i64_base,
)
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from primus_turbo.flydsl.utils.prims import (
    _readfirstlane_i32,
    atomic_add,
    cast,
    ld,
    memory_fence,
    read_clock,
    spin_timed_out,
    st,
)

# i64 fields of a copy unit
UNIT_FIELDS = 8
U_SRC, U_SRC_OFF, U_DST, U_DST_OFF, U_BYTES, U_GROUP, U_GROUP_SIZE, U_FLAG = range(UNIT_FIELDS)
# bytes a unit moves at most: one pass of 16 B vectors, ``PUSH_BATCH`` loads in flight per thread
PUSH_BATCH = 8
UNIT_BYTES = BLOCK_THREADS * PUSH_BATCH * 16


# realtime ticks (100 MHz) a flag wait allows before it traps: a peer's first call of a shape JIT compiles
# its attention kernels for tens of seconds, which the 20 s of the collective GEMM waits do not cover
WAIT_TIMEOUT = 300 * 100_000_000


def _reached(flag_base, flag_idxs, expected):
    ready = None
    for idx in flag_idxs:
        ok = ld(flag_base, idx, scope="sys", dtype=fx.T.i64) >= expected
        ready = ok if ready is None else ready & ok
    return ready


@ASTRewriter.transform
def _spin(flag_base, flag_idxs, expected, rank, tag):
    """Spin until every flag of ``flag_idxs`` reaches ``expected``; trap after ``WAIT_TIMEOUT``."""
    spin_start = read_clock()
    fx.rocdl.s_waitcnt(0)
    ready = _reached(flag_base, flag_idxs, expected)
    while not ready:
        fx.rocdl.s_sleep(fx.Int32(1))
        if spin_timed_out(spin_start, WAIT_TIMEOUT):
            fx.printf(
                f"[agkv rank={rank} {tag}] flag wait timeout: first flag_idx={{}} expected={{}}\n",
                flag_idxs[0],
                expected,
            )
            _asm("s_trap 2")
        fx.rocdl.s_waitcnt(0)
        ready = _reached(flag_base, flag_idxs, expected)


def _unit_field(units, unit, field, wide=False):
    """Field ``field`` of the block-uniform ``unit``, in scalar registers: a vector load leaves it in VGPRs,
    and every buffer descriptor built from it then pays a waterfall (and spills inside a persistent loop)."""
    value = ld(units, unit * fx.Int32(UNIT_FIELDS) + fx.Int32(field), dtype=fx.T.i64)
    return _uniform_i64(value) if wide else _readfirstlane_i32(cast(value, fx.T.i32))


def _sum_rows(thread_index, nvec, a_res, b_res, dst_res):
    """``dst = a + b`` over ``nvec`` 8-element bf16 vectors (fp32 add), ``PUSH_BATCH`` per thread in flight."""
    f32_vec, bf16_vec = fx.T.vec(8, fx.T.f32), fx.T.vec(8, fx.T.bf16)
    num_vecs = UNIT_BYTES // 16
    per_iter = BLOCK_THREADS * PUSH_BATCH
    for it in range_constexpr(-(-num_vecs // per_iter)):
        vecs = [
            fx.Int32(it * per_iter + j * BLOCK_THREADS) + thread_index
            for j in range_constexpr(PUSH_BATCH)
            if it * per_iter + j * BLOCK_THREADS < num_vecs
        ]
        pairs = [
            tuple(
                buffer_load(res, v * fx.Int32(8), vec_width=8, dtype=fx.T.bf16, mask=v < nvec)
                for res in (a_res, b_res)
            )
            for v in vecs
        ]
        for v, (x, y) in zip(vecs, pairs):
            total = fx.arith.addf(fx.arith.extf(f32_vec, x), fx.arith.extf(f32_vec, y))
            buffer_store(
                fx.arith.trunc_f(bf16_vec, total),
                dst_res,
                v * fx.Int32(8),
                mask=v < nvec,
                cache_modifier=SYS_NT,
            )


def _copy_unit(
    thread_index, unit, srcs, heaps, units, bank_off, addends, add_delta, block_threads=BLOCK_THREADS
):
    """Issue the stores of ``unit`` (not waited). With ``addends``, store source + addend instead: the
    addend of source ``i`` is ``addends[i]``, read ``add_delta`` bytes before the unit's source offset.
    A block of ``block_threads`` < ``BLOCK_THREADS`` copies the unit as that many threads would."""
    sel = _unit_field(units, unit, U_SRC)
    src_base = srcs[0]
    for i in range(1, len(srcs)):
        src_base = arith.select(sel == fx.Int32(i), srcs[i], src_base)
    nbytes = _unit_field(units, unit, U_BYTES)
    dst_heap = _uniform_i64(ld(heaps, _unit_field(units, unit, U_DST), dtype=fx.T.i64))
    src_off = _unit_field(units, unit, U_SRC_OFF, wide=True)
    dst_addr = dst_heap + bank_off + _unit_field(units, unit, U_DST_OFF, wide=True)
    src_res = create_buffer_resource_from_addr(src_base + src_off, num_records_bytes=nbytes)
    dst_res = create_buffer_resource_from_addr(dst_addr, num_records_bytes=nbytes)
    nvec = nbytes // fx.Int32(16)
    if addends is not None:
        assert block_threads == BLOCK_THREADS
        add_base = addends[0]
        for i in range(1, len(addends)):
            add_base = arith.select(sel == fx.Int32(i), addends[i], add_base)
        add_res = create_buffer_resource_from_addr(add_base + src_off - add_delta, num_records_bytes=nbytes)
        _sum_rows(thread_index, nvec, src_res, add_res, dst_res)
        return

    def coords(v):
        word = v * fx.Int32(VEC_I32)
        return word, word, v < nvec

    for k in range_constexpr(BLOCK_THREADS // block_threads):
        copy_rows(
            thread_index + fx.Int32(k * block_threads),
            UNIT_BYTES // 16,
            coords,
            src_res,
            dst_res,
            0,
            SYS_NT,
            PUSH_BATCH,
        )


@ASTRewriter.transform
def _push_loop(
    thread_index,
    block_index,
    srcs,
    heaps,
    flags,
    units,
    counters,
    bank_off,
    epoch,
    num_units,
    grid,
    addends,
    add_delta,
    block_threads=BLOCK_THREADS,
):
    unit = block_index
    while unit < fx.Int32(num_units):
        _copy_unit(thread_index, unit, srcs, heaps, units, bank_off, addends, add_delta, block_threads)
        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        if thread_index == fx.Int32(0):
            group = _unit_field(units, unit, U_GROUP)
            # every unit's stores retired before its count, so the group's last count publishes them all
            done = atomic_add(counters, group, fx.Int64(1), scope="agent")
            if done == _unit_field(units, unit, U_GROUP_SIZE, wide=True) - fx.Int64(1):
                st(counters, group, fx.Int64(0), scope="agent")
                dst_flags = ld(flags, _unit_field(units, unit, U_DST), dtype=fx.T.i64)
                st(dst_flags, _unit_field(units, unit, U_FLAG), epoch, scope="sys", order="relaxed")
        unit = unit + fx.Int32(grid)


@functools.lru_cache(maxsize=64)
def _make_push_kernel(num_srcs, num_units, grid, summed):
    @flyc.kernel(name="agkv_push_kernel", known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        SRC0: fx.Tensor,
        SRC1: fx.Tensor,
        SRC2: fx.Tensor,
        SRC3: fx.Tensor,
        HEAPS: fx.Tensor,
        FLAGS: fx.Tensor,
        UNITS: fx.Tensor,
        COUNTERS: fx.Tensor,
        bank_off: fx.Int64,
        epoch: fx.Int64,
        add_delta: fx.Int64,
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        bases = [i64_base(t) for t in (SRC0, SRC1, SRC2, SRC3)]
        srcs, addends = (bases[:2], bases[2:]) if summed else (bases[:num_srcs], None)
        _push_loop(
            cast(thread_index, fx.T.i32),
            cast(block_index, fx.T.i32),
            srcs,
            i64_base(HEAPS),
            i64_base(FLAGS),
            i64_base(UNITS),
            i64_base(COUNTERS),
            cast(bank_off, fx.T.i64),
            cast(epoch, fx.T.i64),
            num_units,
            grid,
            addends,
            cast(add_delta, fx.T.i64),
        )

    return kernel


@flyc.jit
def _launch_push(
    SRC0,
    SRC1,
    SRC2,
    SRC3,
    HEAPS,
    FLAGS,
    UNITS,
    COUNTERS,
    bank_off: int,
    epoch: int,
    add_delta: int,
    num_srcs: fx.Constexpr[int],
    num_units: fx.Constexpr[int],
    grid: fx.Constexpr[int],
    summed: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel = _make_push_kernel(num_srcs, num_units, grid, summed)
    kernel(SRC0, SRC1, SRC2, SRC3, HEAPS, FLAGS, UNITS, COUNTERS, bank_off, epoch, add_delta).launch(
        grid=(grid, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
    )


def push(
    srcs, heaps, flags, units, num_units, counters, bank_off, epoch, grid, stream, addends=None, add_delta=0
):
    """Run the copy units of ``units`` (``[num_units, UNIT_FIELDS]`` int64) with ``grid`` blocks.

    ``srcs``: up to 4 contiguous tensors a unit's source selects; ``heaps`` / ``flags``: int64 tensors
    of every rank's symmetric buffer and flag array addresses; ``counters``: int64 group counters,
    zero between launches; ``bank_off``: byte offset added to every destination offset.
    ``addends``: one bf16 tensor per source (two sources at most); a unit then stores its source plus the
    addend's bytes ``add_delta`` before the unit's source offset, summed in fp32."""
    assert 1 <= len(srcs) <= 4 and all(t.is_contiguous() for t in srcs)
    if addends is not None:
        assert len(srcs) == len(addends) <= 2 and all(t.is_contiguous() for t in addends)
        padded = list(srcs) + [srcs[0]] * (2 - len(srcs)) + list(addends) + [addends[0]] * (2 - len(addends))
    else:
        padded = list(srcs) + [srcs[0]] * (4 - len(srcs))
    run_launcher(
        _launch_push,
        *padded,
        heaps,
        flags,
        units,
        counters,
        bank_off,
        epoch,
        add_delta,
        len(srcs),
        num_units,
        min(grid, num_units),
        int(addends is not None),
        stream,
    )


# bf16 elements a reduce unit sums: ``REDUCE_ITERS`` passes of one 16 B vector per thread
REDUCE_ITERS = 4
REDUCE_ELEMS = BLOCK_THREADS * 8 * REDUCE_ITERS


def _reduce_unit(thread_index, srcs, out_base, elem0, seg_elems):
    """``out[elem0 : elem0 + REDUCE_ELEMS] = sum(src[...] for src in srcs)``, fp32, in ``srcs`` order;
    ``srcs`` are ``(address, cache modifier)`` of bf16 segments of ``seg_elems`` elements."""
    f32_vec, bf16_vec = fx.T.vec(8, fx.T.f32), fx.T.vec(8, fx.T.bf16)
    res = [(create_buffer_resource_from_addr(a, num_records_bytes=seg_elems * 2), cm) for a, cm in srcs]
    out_res = create_buffer_resource_from_addr(out_base, num_records_bytes=seg_elems * 2)
    for it in range(REDUCE_ITERS):
        e = elem0 + (fx.Int32(it * BLOCK_THREADS) + thread_index) * fx.Int32(8)
        vals = [buffer_load(rs, e, vec_width=8, dtype=fx.T.bf16, cache_modifier=cm) for rs, cm in res]
        acc = None
        for val in vals:
            term = fx.arith.extf(f32_vec, val)
            acc = term if acc is None else fx.arith.addf(acc, term)
        buffer_store(fx.arith.trunc_f(bf16_vec, acc), out_res, e)


@ASTRewriter.transform
def _reduce_loop(thread_index, block_index, segs, flags_base, epoch, rank, pieces, grid, seg_elems):
    num_units = len(segs) * pieces
    unit = block_index
    while unit < fx.Int32(num_units):
        seg = unit // fx.Int32(pieces)
        elem0 = (unit % fx.Int32(pieces)) * fx.Int32(REDUCE_ELEMS)
        for i in range_constexpr(len(segs)):
            srcs, out_base, flag_idxs = segs[i]
            if seg == fx.Int32(i):
                if const_expr(len(flag_idxs) > 0):
                    if thread_index == fx.Int32(0):
                        _spin(flags_base, [fx.Int32(f) for f in flag_idxs], epoch, rank, "agkv_reduce")
                    fx.rocdl.s_waitcnt(0)
                    fx.gpu.barrier()
                    memory_fence(order="acquire", scope="sys")
                _reduce_unit(thread_index, srcs, out_base, elem0, seg_elems)
        unit = unit + fx.Int32(grid)


def _reduce_segs(
    bank, tensors, n, r, seg_elems, local_first, local_tail, extra_first, bank_v, first_flags, second_flags
):
    """``(sources, output address, flags)`` of the four output segments (K / V x half)."""
    segs = []
    for t, (local, extra, out) in enumerate(tensors):
        local_base, extra_base, out_base = i64_base(local), i64_base(extra), i64_base(out)
        region = bank + fx.Int64(t * bank_v * 2)
        for h in range(2):
            srcs = []
            for s in range(n) if h == 0 else range(r + 1):
                if s == r:
                    own = local_first if h == 0 else local_tail
                    srcs.append((local_base + fx.Int64(own * 2), 0))
                    if h == 0 and extra_first >= 0:
                        srcs.append((extra_base + fx.Int64(extra_first * 2), 0))
                else:
                    srcs.append((region + fx.Int64((h * n + s) * seg_elems * 2), SYS_NT))
            segs.append(
                (srcs, out_base + fx.Int64(h * seg_elems * 2), first_flags if h == 0 else second_flags)
            )
    return segs


@functools.lru_cache(maxsize=64)
def _make_reduce_kernel(
    rank, num_ranks, seg_elems, local_first, local_tail, extra_first, bank_v, first_flags, second_flags, grid
):
    n, r = num_ranks, rank
    pieces = -(-seg_elems // REDUCE_ELEMS)

    @flyc.kernel(name="agkv_reduce_kernel", known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        HEAPS: fx.Tensor,
        FLAGS: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        XK: fx.Tensor,
        XV: fx.Tensor,
        OUT_K: fx.Tensor,
        OUT_V: fx.Tensor,
        bank_off: fx.Int64,
        epoch: fx.Int64,
    ):
        thread_index = cast(fx.thread_idx.x, fx.T.i32)
        block_index, _b, _c = fx.block_idx
        bank = ld(i64_base(HEAPS), fx.Int32(r), dtype=fx.T.i64) + cast(bank_off, fx.T.i64)
        flags_base = ld(i64_base(FLAGS), fx.Int32(r), dtype=fx.T.i64)
        segs = _reduce_segs(
            bank,
            ((DK, XK, OUT_K), (DV, XV, OUT_V)),
            n,
            r,
            seg_elems,
            local_first,
            local_tail,
            extra_first,
            bank_v,
            first_flags,
            second_flags,
        )
        _reduce_loop(
            thread_index,
            cast(block_index, fx.T.i32),
            segs,
            flags_base,
            cast(epoch, fx.T.i64),
            r,
            pieces,
            grid,
            seg_elems,
        )

    return kernel, 4 * pieces


@flyc.jit
def _launch_reduce(
    HEAPS,
    FLAGS,
    DK,
    DV,
    XK,
    XV,
    OUT_K,
    OUT_V,
    bank_off: int,
    epoch: int,
    rank: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    seg_elems: fx.Constexpr[int],
    local_first: fx.Constexpr[int],
    local_tail: fx.Constexpr[int],
    extra_first: fx.Constexpr[int],
    bank_v: fx.Constexpr[int],
    first_flags: fx.Constexpr[int],
    second_flags: fx.Constexpr[int],
    grid: fx.Constexpr[int],
    stream: fx.Stream,
):
    unpack = lambda mask: tuple(i for i in range(mask.bit_length()) if mask >> i & 1)  # noqa: E731
    kernel, _ = _make_reduce_kernel(
        rank,
        num_ranks,
        seg_elems,
        local_first,
        local_tail,
        extra_first,
        bank_v,
        unpack(first_flags),
        unpack(second_flags),
        grid,
    )
    kernel(HEAPS, FLAGS, DK, DV, XK, XV, OUT_K, OUT_V, bank_off, epoch).launch(
        grid=(grid, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
    )


def reduce(
    heaps,
    flags,
    dk,
    dv,
    out_k,
    out_v,
    rank,
    num_ranks,
    seg_elems,
    local_first,
    local_tail,
    bank_off,
    bank_v,
    first_flags,
    second_flags,
    epoch,
    stream,
    extra=None,
):
    """Sum this rank's dK / dV, ``out_k`` / ``out_v`` = ``[first half | second half]`` of
    ``seg_elems`` bf16 elements each.

    Half 0 sums, in sender order, the first-half slots of every sender (``(0 * n + s) * seg_elems`` of
    the K / V region of the bank at ``bank_off``; the V region starts ``bank_v`` elements in), with the
    own term read from ``dk`` / ``dv`` at element ``local_first``; half 1 sums the second-half slots of
    senders ``0..rank``, the own term at ``local_tail``. A half waits for its flags (``first_flags`` /
    ``second_flags``, indices into this rank's flag array) first. ``extra = (xk, xv, first)`` adds a
    second own term to half 0, read from ``xk`` / ``xv`` at element ``first``."""
    pieces = -(-seg_elems // REDUCE_ELEMS)
    mask = lambda idxs: sum(1 << i for i in set(idxs))  # noqa: E731
    xk, xv, extra_first = extra if extra is not None else (dk, dv, -1)
    run_launcher(
        _launch_reduce,
        heaps,
        flags,
        dk,
        dv,
        xk,
        xv,
        out_k,
        out_v,
        bank_off,
        epoch,
        rank,
        num_ranks,
        seg_elems,
        local_first,
        local_tail,
        extra_first,
        bank_v,
        mask(first_flags),
        mask(second_flags),
        min(4 * pieces, 256),
        stream,
    )


@ASTRewriter.transform
def _wait_flags(thread_index, base, flag_idxs, epoch, rank):
    if thread_index == fx.Int32(0):
        _spin(base, flag_idxs, epoch, rank, "agkv_wait")


@functools.lru_cache(maxsize=256)
def _make_wait_kernel(rank, flag_mask):
    flag_idxs = [i for i in range(flag_mask.bit_length()) if flag_mask >> i & 1]

    @flyc.kernel(name="agkv_wait_kernel", known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(FLAGS: fx.Tensor, epoch: fx.Int64):
        thread_index = fx.thread_idx.x
        base = ld(i64_base(FLAGS), fx.Int32(rank), dtype=fx.T.i64)
        _wait_flags(
            cast(thread_index, fx.T.i32), base, [fx.Int32(i) for i in flag_idxs], cast(epoch, fx.T.i64), rank
        )
        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        memory_fence(order="acquire", scope="sys")

    return kernel


@flyc.jit
def _launch_wait(
    FLAGS,
    epoch: int,
    rank: fx.Constexpr[int],
    flag_mask: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel = _make_wait_kernel(rank, flag_mask)
    kernel(FLAGS, epoch).launch(grid=(NUM_XCDS, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)


def wait(flags, rank, flag_idxs, epoch, stream):
    """Hold ``stream`` until ``rank``'s flags ``flag_idxs`` (indices into its flag array) reach ``epoch``."""
    mask = sum(1 << i for i in set(flag_idxs))
    if mask:
        run_launcher(_launch_wait, flags, epoch, rank, mask, stream)


def _merge_vectors(thread_index, block_index, oa, la, ob, lb, out, lse, sq, b, hq, d):
    """One 8-element vector of ``out`` = softmax-weighted sum of ``oa`` / ``ob`` (fp32), and the row's
    merged lse from its first vector; vectors past the end fall outside the buffer ranges."""
    f32, f32_vec, bf16_vec = fx.T.f32, fx.T.vec(8, fx.T.f32), fx.T.vec(8, fx.T.bf16)
    vpr = d // 8
    rows = sq * b * hq
    o_res = [create_buffer_resource_from_addr(t, num_records_bytes=rows * d * 2) for t in (oa, ob, out)]
    l_res = [create_buffer_resource_from_addr(t, num_records_bytes=rows * 4) for t in (la, lb, lse)]
    vec = block_index * fx.Int32(BLOCK_THREADS) + thread_index
    row = vec // fx.Int32(vpr)
    # o rows are (token, batch, head), lse rows (batch, token, head)
    token, batch, head = row // fx.Int32(b * hq), (row // fx.Int32(hq)) % fx.Int32(b), row % fx.Int32(hq)
    li = (batch * fx.Int32(sq) + token) * fx.Int32(hq) + head
    a, z = (buffer_load(res, li, vec_width=1, dtype=f32) for res in l_res[:2])
    m = fx.arith.maximumf(a, z)
    ea, ez = fmath.exp(fx.arith.subf(a, m)), fmath.exp(fx.arith.subf(z, m))
    total = fx.arith.addf(ea, ez)
    wa, wz = fx.arith.divf(ea, total), fx.arith.divf(ez, total)
    va, vz = (
        fx.arith.extf(f32_vec, buffer_load(res, vec * fx.Int32(8), vec_width=8, dtype=fx.T.bf16))
        for res in o_res[:2]
    )
    acc = fx.arith.addf(
        fx.arith.mulf(va, _vector.broadcast(f32_vec, wa)), fx.arith.mulf(vz, _vector.broadcast(f32_vec, wz))
    )
    buffer_store(fx.arith.trunc_f(bf16_vec, acc), o_res[2], vec * fx.Int32(8))
    merged = fx.arith.addf(m, fmath.log(total))
    buffer_store(merged, l_res[2], li, mask=(vec % fx.Int32(vpr)) == fx.Int32(0))


@functools.lru_cache(maxsize=64)
def _make_merge_kernel(sq, b, hq, d):
    @flyc.kernel(name="agkv_merge_kernel", known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(OA: fx.Tensor, LA: fx.Tensor, OB: fx.Tensor, LB: fx.Tensor, OUT: fx.Tensor, LSE: fx.Tensor):
        block_index, _b, _c = fx.block_idx
        _merge_vectors(
            cast(fx.thread_idx.x, fx.T.i32),
            cast(block_index, fx.T.i32),
            *(i64_base(t) for t in (OA, LA, OB, LB, OUT, LSE)),
            sq,
            b,
            hq,
            d,
        )

    return kernel


@flyc.jit
def _launch_merge(
    OA,
    LA,
    OB,
    LB,
    OUT,
    LSE,
    sq: fx.Constexpr[int],
    b: fx.Constexpr[int],
    hq: fx.Constexpr[int],
    d: fx.Constexpr[int],
    stream: fx.Stream,
):
    grid = -(-sq * b * hq * d // (8 * BLOCK_THREADS))
    _make_merge_kernel(sq, b, hq, d)(OA, LA, OB, LB, OUT, LSE).launch(
        grid=(grid, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
    )


def merge(out, lse, oa, la, ob, lb, stream):
    """``out`` / ``lse`` = the merge of two attentions of the same queries over disjoint keys.

    ``o*`` / ``out``: contiguous bf16 ``[sq, b, hq, d]``; ``l*`` / ``lse``: fp32 natural-log ``[b * sq, hq]``."""
    sq, b, hq, d = out.shape
    assert d % 8 == 0 and all(t.is_contiguous() for t in (out, lse, oa, la, ob, lb))
    run_launcher(_launch_merge, oa, la, ob, lb, out, lse, sq, b, hq, d, stream)
