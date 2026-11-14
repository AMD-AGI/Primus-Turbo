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

"""Device helpers shared by the Ulysses fused kernels: flag waits, block-wide unit claims and the
epoch bump of the last block out (see ``workspace.py`` for the protocol)."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith

from primus_turbo.flydsl.deep_ep.common import _asm, ld_lds, st_lds
from primus_turbo.flydsl.utils.buffer_ops import extract_base_index
from primus_turbo.flydsl.utils.prims import (
    _readfirstlane_i32,
    atomic_add,
    cast,
    ld,
    read_clock,
    spin_timed_out,
    st,
)

WARP = 64
BLOCK_THREADS = 512
NUM_WARPS = BLOCK_THREADS // WARP
# 16 B per lane per copy step
VEC_I32 = 4
VEC_BF16 = 8
# sc0|sc1|nt: system-coherent, streaming; staging is read by peers, recv reuses a bank every 2 epochs
SYS_NT = 19
# s_memrealtime ticks at 100 MHz: 5 s
SPIN_TIMEOUT = 500_000_000

# state: [epoch, exit counter, unit claim counter, tile claim counter]
STATE_EPOCH = 0
STATE_EXIT = 1
STATE_CLAIM = 2
STATE_TILE = 3
# AgGemm: own-row copy pieces claimed / done
STATE_ASSIST = 4
STATE_ASSIST_DONE = 5
# AgGemm: tile claim counter of each XCD (MI355X: 8 XCDs, block i runs on XCD i % 8)
NUM_XCDS = 8
STATE_XCD_TILE = 6


def i64_base(tensor):
    return fx.arith.ArithValue(arith.index_cast(fx.T.i64, extract_base_index(tensor)), signed=True)


def opaque_tid(thread_index):
    """``thread_index`` through an opaque move. Inside a claim loop over several unit kinds LICM
    hoists every unit body's lane offsets out of their branches, and all of them live across the
    loop and spill."""
    return fx.arith.ArithValue(
        _llvm.inline_asm(
            fx.T.i32,
            [fx.as_ir_value(cast(thread_index, fx.T.i32))],
            "v_mov_b32 $0, $1",
            "=v,v",
            has_side_effects=True,
        )
    )


def opaque_base(tensor):
    """``i64_base(tensor)`` through an opaque move, for the same reason as ``opaque_tid``."""
    return fx.arith.ArithValue(
        _llvm.inline_asm(
            fx.T.i64, [fx.as_ir_value(i64_base(tensor))], "s_mov_b64 $0, $1", "=s,s", has_side_effects=True
        )
    )


@ASTRewriter.transform
def spin_flag(flag_base, flag_idx, expected, rank, src, tag):
    """Spin until the i64 flag reaches ``expected``; on timeout report and trap (no silent hang)."""
    spin_start = read_clock()
    fx.rocdl.s_waitcnt(0)
    flag = ld(flag_base, flag_idx, scope="sys", dtype=fx.T.i64)
    while flag < expected:
        fx.rocdl.s_sleep(fx.Int32(1))
        if spin_timed_out(spin_start, SPIN_TIMEOUT):
            fx.printf(
                f"[ulysses rank={rank} {tag}] flag wait timeout: src={{}} flag_idx={{}} flag={{}} expected={{}}\n",
                src,
                flag_idx,
                flag,
                expected,
            )
            _asm("s_trap 2")
        fx.rocdl.s_waitcnt(0)
        flag = ld(flag_base, flag_idx, scope="sys", dtype=fx.T.i64)


def _all_reached(flag_base, flag_idxs, expected):
    ready = None
    for idx in flag_idxs:
        ok = ld(flag_base, idx, scope="sys", dtype=fx.T.i64) >= expected
        ready = ok if ready is None else ready & ok
    return ready


@ASTRewriter.transform
def spin_flags(flag_base, flag_idxs, expected, rank, src, tag):
    """``spin_flag`` over several flags, polled together (one round trip per poll)."""
    spin_start = read_clock()
    fx.rocdl.s_waitcnt(0)
    ready = _all_reached(flag_base, flag_idxs, expected)
    while not ready:
        fx.rocdl.s_sleep(fx.Int32(1))
        if spin_timed_out(spin_start, SPIN_TIMEOUT):
            fx.printf(
                f"[ulysses rank={rank} {tag}] flag wait timeout: src={{}} first flag_idx={{}} expected={{}}\n",
                src,
                flag_idxs[0],
                expected,
            )
            _asm("s_trap 2")
        fx.rocdl.s_waitcnt(0)
        ready = _all_reached(flag_base, flag_idxs, expected)


@ASTRewriter.transform
def claim_unit(thread_index, state_base, claim_slot, counter=STATE_CLAIM):
    """Next work unit for the whole block, from the call's shared counter ``state[counter]``."""
    if thread_index == fx.Int32(0):
        claimed = atomic_add(state_base, fx.Int32(counter), fx.Int64(1), scope="agent")
        st_lds(claim_slot, 0, cast(claimed, fx.T.i32))
    # s_barrier does not wait for LDS traffic: drain the store, then the reads
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    unit = _readfirstlane_i32(ld_lds(claim_slot, 0))
    fx.rocdl.s_waitcnt(0)
    fx.gpu.barrier()
    return unit


@ASTRewriter.transform
def exit_and_bump_epoch(thread_index, state_base, epoch, grid, order="acq_rel", extra_counters=()):
    """The last block out resets the counters and bumps the epoch; every block read it on entry.
    ``order="relaxed"`` skips the L2 writeback / invalidate around the count; enough when every
    claim of the block has returned before it exits and its flags carry their own release."""
    fx.gpu.barrier()
    if thread_index == fx.Int32(0):
        arrived = atomic_add(state_base, fx.Int32(STATE_EXIT), fx.Int64(1), scope="agent", order=order)
        if arrived == fx.Int64(grid - 1):
            st(state_base, fx.Int32(STATE_EXIT), fx.Int64(0))
            st(state_base, fx.Int32(STATE_CLAIM), fx.Int64(0))
            st(state_base, fx.Int32(STATE_TILE), fx.Int64(0))
            for counter in extra_counters:
                st(state_base, fx.Int32(counter), fx.Int64(0))
            st(state_base, fx.Int32(STATE_EPOCH), epoch + fx.Int64(1))
