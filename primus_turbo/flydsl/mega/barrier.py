###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter

from primus_turbo.flydsl.mega.symm_buffer import SymBuffer, Workspace
from primus_turbo.flydsl.utils.prims import (
    atomic_add,
    ld,
    memory_fence,
    read_clock,
    spin_timed_out,
)

# grid_sync counter: low bits count per-block arrivals, bit 25 is the phase flag. Requires num_blocks < 2^25.
_PHASE_BIT = 25
_PHASE_MASK = fx.Int32(1 << _PHASE_BIT)


@ASTRewriter.transform
def grid_sync(
    workspace: Workspace,
    thread_id: fx.Int32,
    block_id: fx.Int32,
    num_blocks: int,
    rank: int = -1,
    tag: str = "grid_sync",
):
    """Device-wide barrier over all blocks via a split counter."""
    grid_sync_count_ptr = workspace.get_grid_sync_count_ptr(0)
    fx.gpu.barrier()
    memory_fence(order="release", scope="agent")
    if thread_id == fx.Int32(0):
        # last block folds the phase-bit flip into its increment
        add_value = fx.arith.select(
            block_id == fx.Int32(0),
            fx.Int32((1 << _PHASE_BIT) - (num_blocks - 1)),
            fx.Int32(1),
        )
        old_value = atomic_add(grid_sync_count_ptr, fx.Int32(0), add_value, scope="agent")
        spin_start = read_clock()
        new_value = ld(grid_sync_count_ptr, fx.Int32(0), scope="agent")
        # spin until the phase bit toggles relative to our arrival snapshot
        while ((new_value ^ old_value) & _PHASE_MASK) == fx.Int32(0):
            if spin_timed_out(spin_start):
                # tag/rank are compile-time constants, baked into the format string
                fx.printf(
                    "[MEGA rank=" + str(rank) + " " + tag + "] grid_sync stuck: waiting on peer blocks; "
                    "this block={} arrived_count={} expected_num_blocks={}\n",
                    block_id,
                    new_value,
                    fx.Int32(num_blocks),
                )
                spin_start = read_clock()
            new_value = ld(grid_sync_count_ptr, fx.Int32(0), scope="agent")
    fx.gpu.barrier()
    memory_fence(order="acquire", scope="agent")


@ASTRewriter.transform
def xgmi_barrier(
    workspace: Workspace,
    sym: SymBuffer,
    rank: int,
    world_size: int,
    thread_id: fx.Int32,
    block_id: fx.Int32,
    skip_fence: bool = False,
    tag: str = "xgmi_barrier",
):
    """Cross-rank arrival barrier over XGMI."""
    # hoist all workspace-derived values before dynamic control flow (rewriter can't carry Workspace)
    counter_ptr = workspace.get_xgmi_barrier_counter_ptr()
    status = ld(counter_ptr, fx.Int32(0), scope="agent") & fx.Int32(3)
    sign_is_pos = (status & fx.Int32(2)) == fx.Int32(0)  # bit 1: 0 -> +1/world, 1 -> -1/0
    signal_ptr = fx.arith.select(
        (status & fx.Int32(1)) == fx.Int32(0),  # bit 0: which of the two signal buffers
        workspace.get_xgmi_barrier_signal_ptr(0),
        workspace.get_xgmi_barrier_signal_ptr(1),
    )
    add_value = fx.arith.select(sign_is_pos, fx.Int32(1), fx.Int32(-1))
    target = fx.arith.select(sign_is_pos, fx.Int32(world_size), fx.Int32(0))

    if not skip_fence:
        memory_fence(order="release", scope="sys")
    fx.gpu.barrier()
    if block_id == fx.Int32(0):
        # thread t (t < world) bumps peer t's signal; our own signal is bumped by every rank
        if thread_id < fx.Int32(world_size):
            atomic_add(sym.map(signal_ptr, thread_id), fx.Int32(0), add_value, scope="sys")
        fx.gpu.barrier()  # local sends land before thread 0 waits on our own signal
        if thread_id == fx.Int32(0):
            atomic_add(counter_ptr, fx.Int32(0), fx.Int32(1), scope="agent")  # advance phase/sign
            spin_start = read_clock()
            signal_value = ld(signal_ptr, fx.Int32(0), scope="sys", order="acquire")
            while signal_value != target:
                if spin_timed_out(spin_start):
                    # rank/tag are compile-time constants, baked into the format string
                    fx.printf(
                        "[MEGA rank=" + str(rank) + " " + tag + "] xgmi_barrier stuck: "
                        "signal={} != target={}\n",
                        signal_value,
                        target,
                    )
                    spin_start = read_clock()
                signal_value = ld(signal_ptr, fx.Int32(0), scope="sys", order="acquire")
    fx.gpu.barrier()
