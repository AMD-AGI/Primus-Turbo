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

The attention kernels stay the stock FlyDSL ones; these move K / V and their gradients between ranks
with compute-unit stores over XGMI into the peers' symmetric buffers, MegaMoE style:

``push``
    walks a table of copy units ``(source, source offset, destination rank, destination offset, bytes,
    group, group size, flag)``; a unit's stores are system scope and retired before the block counts the
    unit into its group, and the group's last unit raises the destination's flag to ``epoch``.
``wait``
    holds the stream until a set of this rank's flags reach ``epoch``. One block per XCD, each ending
    with a system-scope acquire, so no XCD's L2 keeps lines of a bank a peer rewrote.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr

from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.fused_collective_gemm.common import (
    BLOCK_THREADS,
    NUM_XCDS,
    SYS_NT,
    VEC_I32,
    copy_rows,
    i64_base,
    spin_flags,
)
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from primus_turbo.flydsl.utils.prims import atomic_add, cast, ld, memory_fence, st

# i64 fields of a copy unit
UNIT_FIELDS = 8
U_SRC, U_SRC_OFF, U_DST, U_DST_OFF, U_BYTES, U_GROUP, U_GROUP_SIZE, U_FLAG = range(UNIT_FIELDS)
# bytes a unit moves at most: one pass of 16 B vectors, ``PUSH_BATCH`` loads in flight per thread
PUSH_BATCH = 8
UNIT_BYTES = BLOCK_THREADS * PUSH_BATCH * 16


def _unit_field(units, unit, field, wide=False):
    value = ld(units, unit * fx.Int32(UNIT_FIELDS) + fx.Int32(field), dtype=fx.T.i64)
    return value if wide else cast(value, fx.T.i32)


def _copy_unit(thread_index, unit, srcs, heaps, units, bank_off):
    """Issue the stores of ``unit`` (not waited)."""
    sel = _unit_field(units, unit, U_SRC)
    src_base = srcs[0]
    for i in range(1, len(srcs)):
        src_base = arith.select(sel == fx.Int32(i), srcs[i], src_base)
    nbytes = _unit_field(units, unit, U_BYTES)
    dst_heap = ld(heaps, _unit_field(units, unit, U_DST), dtype=fx.T.i64)
    src_addr = src_base + _unit_field(units, unit, U_SRC_OFF, wide=True)
    dst_addr = dst_heap + bank_off + _unit_field(units, unit, U_DST_OFF, wide=True)
    src_res = create_buffer_resource_from_addr(src_addr, num_records_bytes=nbytes)
    dst_res = create_buffer_resource_from_addr(dst_addr, num_records_bytes=nbytes)
    nvec = nbytes // fx.Int32(16)

    def coords(v):
        word = v * fx.Int32(VEC_I32)
        return word, word, v < nvec

    copy_rows(thread_index, UNIT_BYTES // 16, coords, src_res, dst_res, 0, SYS_NT, PUSH_BATCH)


@ASTRewriter.transform
def _push_loop(
    thread_index, block_index, srcs, heaps, flags, units, counters, bank_off, epoch, num_units, grid
):
    unit = block_index
    while unit < fx.Int32(num_units):
        _copy_unit(thread_index, unit, srcs, heaps, units, bank_off)
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
def _make_push_kernel(num_srcs, num_units, grid):
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
    ):
        thread_index = fx.thread_idx.x
        block_index, _b, _c = fx.block_idx
        srcs = [i64_base(t) for t in (SRC0, SRC1, SRC2, SRC3)[:num_srcs]]
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
    num_srcs: fx.Constexpr[int],
    num_units: fx.Constexpr[int],
    grid: fx.Constexpr[int],
    stream: fx.Stream,
):
    kernel = _make_push_kernel(num_srcs, num_units, grid)
    kernel(SRC0, SRC1, SRC2, SRC3, HEAPS, FLAGS, UNITS, COUNTERS, bank_off, epoch).launch(
        grid=(grid, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
    )


def push(srcs, heaps, flags, units, num_units, counters, bank_off, epoch, grid, stream):
    """Run the copy units of ``units`` (``[num_units, UNIT_FIELDS]`` int64) with ``grid`` blocks.

    ``srcs``: up to 4 contiguous tensors a unit's source selects; ``heaps`` / ``flags``: int64 tensors
    of every rank's symmetric buffer and flag array addresses; ``counters``: int64 group counters,
    zero between launches; ``bank_off``: byte offset added to every destination offset."""
    assert 1 <= len(srcs) <= 4 and all(t.is_contiguous() for t in srcs)
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
        len(srcs),
        num_units,
        min(grid, num_units),
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
                        spin_flags(
                            flags_base, [fx.Int32(f) for f in flag_idxs], epoch, rank, -1, "agkv_reduce"
                        )
                    fx.rocdl.s_waitcnt(0)
                    fx.gpu.barrier()
                    memory_fence(order="acquire", scope="sys")
                _reduce_unit(thread_index, srcs, out_base, elem0, seg_elems)
        unit = unit + fx.Int32(grid)


def _reduce_segs(bank, tensors, n, r, seg_elems, local_first, local_tail, bank_v, first_flags, second_flags):
    """``(sources, output address, flags)`` of the four output segments (K / V x half)."""
    segs = []
    for t, (local, out) in enumerate(tensors):
        local_base, out_base = i64_base(local), i64_base(out)
        region = bank + fx.Int64(t * bank_v * 2)
        for h in range(2):
            srcs = []
            for s in range(n) if h == 0 else range(r + 1):
                if s == r:
                    own = local_first if h == 0 else local_tail
                    srcs.append((local_base + fx.Int64(own * 2), 0))
                else:
                    srcs.append((region + fx.Int64((h * n + s) * seg_elems * 2), SYS_NT))
            segs.append(
                (srcs, out_base + fx.Int64(h * seg_elems * 2), first_flags if h == 0 else second_flags)
            )
    return segs


@functools.lru_cache(maxsize=64)
def _make_reduce_kernel(
    rank, num_ranks, seg_elems, local_first, local_tail, bank_v, first_flags, second_flags, grid
):
    n, r = num_ranks, rank
    pieces = -(-seg_elems // REDUCE_ELEMS)

    @flyc.kernel(name="agkv_reduce_kernel", known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        HEAPS: fx.Tensor,
        FLAGS: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
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
            ((DK, OUT_K), (DV, OUT_V)),
            n,
            r,
            seg_elems,
            local_first,
            local_tail,
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
    OUT_K,
    OUT_V,
    bank_off: int,
    epoch: int,
    rank: fx.Constexpr[int],
    num_ranks: fx.Constexpr[int],
    seg_elems: fx.Constexpr[int],
    local_first: fx.Constexpr[int],
    local_tail: fx.Constexpr[int],
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
        bank_v,
        unpack(first_flags),
        unpack(second_flags),
        grid,
    )
    kernel(HEAPS, FLAGS, DK, DV, OUT_K, OUT_V, bank_off, epoch).launch(
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
):
    """Sum this rank's dK / dV, ``out_k`` / ``out_v`` = ``[first half | second half]`` of
    ``seg_elems`` bf16 elements each.

    Half 0 sums, in sender order, the first-half slots of every sender (``(0 * n + s) * seg_elems`` of
    the K / V region of the bank at ``bank_off``; the V region starts ``bank_v`` elements in), with the
    own term read from ``dk`` / ``dv`` at element ``local_first``; half 1 sums the second-half slots of
    senders ``0..rank``, the own term at ``local_tail``. A half waits for its flags (``first_flags`` /
    ``second_flags``, indices into this rank's flag array) first."""
    pieces = -(-seg_elems // REDUCE_ELEMS)
    mask = lambda idxs: sum(1 << i for i in set(idxs))  # noqa: E731
    run_launcher(
        _launch_reduce,
        heaps,
        flags,
        dk,
        dv,
        out_k,
        out_v,
        bank_off,
        epoch,
        rank,
        num_ranks,
        seg_elems,
        local_first,
        local_tail,
        bank_v,
        mask(first_flags),
        mask(second_flags),
        min(4 * pieces, 256),
        stream,
    )


@ASTRewriter.transform
def _wait_flags(thread_index, base, flag_idxs, epoch, rank):
    if thread_index == fx.Int32(0):
        spin_flags(base, flag_idxs, epoch, rank, -1, "agkv_wait")


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
