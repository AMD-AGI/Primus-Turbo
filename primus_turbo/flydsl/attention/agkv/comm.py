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

"""Device-side exchange pieces of the fused all-gather-KV attention (FlyDSL, single node).

K / V move between ranks with compute-unit stores over XGMI into the peers' symmetric buffers, MegaMoE
style, from blocks of the attention kernels themselves:

``_push_loop``
    walks a table of copy units ``(source, source offset, destination rank, destination offset, bytes,
    group, group size, flag)``; a unit's stores are system scope and retired before the block counts the
    unit into its group, and the group's last unit raises the destination's flag to ``epoch``.
``_spin``
    spins until a set of flags reach ``epoch`` (trapping after ``WAIT_TIMEOUT``).
"""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, range_constexpr

from primus_turbo.flydsl.deep_ep.common import _asm, _uniform_i64
from primus_turbo.flydsl.fused_collective_gemm.common import (
    BLOCK_THREADS,
    SYS_NT,
    VEC_I32,
    copy_rows,
)
from primus_turbo.flydsl.utils.buffer_ops import create_buffer_resource_from_addr
from primus_turbo.flydsl.utils.prims import (
    _readfirstlane_i32,
    atomic_add,
    cast,
    ld,
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


def _copy_unit(thread_index, unit, srcs, heaps, units, bank_off, block_threads=BLOCK_THREADS):
    """Issue the stores of ``unit`` (not waited). A block of ``block_threads`` < ``BLOCK_THREADS`` copies
    the unit as that many threads would."""
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
    block_threads=BLOCK_THREADS,
):
    unit = block_index
    while unit < fx.Int32(num_units):
        _copy_unit(thread_index, unit, srcs, heaps, units, bank_off, block_threads)
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
