###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter

from primus_turbo.flydsl.mega.prims import (
    atomic_add,
    ld,
    memory_fence,
    read_clock,
    spin_timed_out,
)
from primus_turbo.flydsl.mega.symm_buffer import SymLayout, sym_map

# grid_sync counter layout: the low bits accumulate per-block arrivals and
# bit 25 is the phase flag the last block flips. Requires num_blocks < 2^25.
_PHASE_BIT = 25
_PHASE_MASK = fx.Int32(1 << _PHASE_BIT)


@ASTRewriter.transform
def grid_sync(
    sym_layout: SymLayout,
    thread_id: fx.Int32,
    block_id: fx.Int32,
    num_blocks: int,
    rank: int = -1,
    tag: str = "grid_sync",
):
    """Device-wide barrier over all blocks via a split counter."""
    fx.gpu.barrier()
    memory_fence(order="release", scope="agent")
    if thread_id == fx.Int32(0):
        # last block folds the phase-bit flip into its increment
        add_value = fx.arith.select(
            block_id == fx.Int32(0),
            fx.Int32((1 << _PHASE_BIT) - (num_blocks - 1)),
            fx.Int32(1),
        )
        old_value = atomic_add(
            sym_layout.grid_sync_count, fx.Int32(0), add_value, scope="agent", order="release"
        )
        spin_start = read_clock()
        new_value = ld(sym_layout.grid_sync_count, fx.Int32(0), scope="agent", order="acquire")
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
            new_value = ld(sym_layout.grid_sync_count, fx.Int32(0), scope="agent", order="acquire")
    fx.gpu.barrier()
    memory_fence(order="acquire", scope="agent")


@ASTRewriter.transform
def xgmi_barrier(
    sym_layout: SymLayout,
    rank: int,
    world_size: int,
    thread_id: fx.Int32,
    block_id: fx.Int32,
    skip_fence: bool = False,
    tag: str = "xgmi_barrier",
):
    """Cross-rank arrival barrier over XGMI."""
    if not skip_fence:
        memory_fence(order="release", scope="sys")
    fx.gpu.barrier()
    if block_id == fx.Int32(0):
        if thread_id < fx.Int32(world_size):
            atomic_add(sym_layout.signal, thread_id, fx.Int32(1), scope="sys")
            atomic_add(
                sym_map(sym_layout, sym_layout.signal, thread_id),
                fx.Int32(rank),
                fx.Int32(-1),
                scope="sys",
            )
            spin_start = read_clock()
            my_signal_value = ld(sym_layout.signal, thread_id, scope="sys")
            while my_signal_value > fx.Int32(0):
                if spin_timed_out(spin_start):
                    # rank/tag are compile-time constants, baked into the format string
                    fx.printf(
                        "[MEGA rank=" + str(rank) + " " + tag + "] xgmi_barrier stuck: "
                        "peer={} has not arrived (outstanding signal={})\n",
                        thread_id,
                        my_signal_value,
                    )
                    spin_start = read_clock()
                my_signal_value = ld(sym_layout.signal, thread_id, scope="sys")
    fx.gpu.barrier()
