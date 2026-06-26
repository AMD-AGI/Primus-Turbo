###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Device-wide + cross-rank barrier/sync primitives for the mega-MoE FlyDSL kernels.

The grid-wide sense-reversing barrier ``grid_sync`` plus the real acquire/release
fences. The low-level pointer / atomic / scalar load-store prims live in ``prims``
(``elem_ptr`` / ``atomic_add`` / ``ld`` / ``st`` for LDS + global); this module just
adds the fences and grid barrier on top. Depends on ``flydsl`` + ``prims`` + ``sym_layout``.
"""

import flydsl.expr as fx
import flydsl.expr.buffer_ops as bo
from flydsl.compiler.ast_rewriter import ASTRewriter

from primus_turbo.flydsl.mega.prims import _unwrap_scope, atomic_add, ld
from primus_turbo.flydsl.mega.sym_layout import SymLayout

_llvm = bo.llvm
_ORD = _llvm.AtomicOrdering
# Single high tag bit (DeepEP). Each round adds exactly this to the counter (block 0
# absorbs the offset), so the tag bit toggles per round -> self-resetting, no rewrite.
# Must exceed num_blocks (arrival residue stays below it). Overflow-safe (mod 2^32).
_FINISH_SUM_TAG = 1 << 25


def fence_acquire(*, scope="agent"):
    """REAL acquire fence: invalidates L1 so a reused-buffer consumer reads fresh.
    Load-bearing for grid_sync table handoff (cheap fence leaves L1 stale)."""
    _llvm.fence(_ORD.acquire, syncscope=_unwrap_scope(scope))


def fence_release(*, scope="agent"):
    """REAL release fence: L2 writeback so other CUs see prior stores; pair w/ fence_acquire."""
    _llvm.fence(_ORD.release, syncscope=_unwrap_scope(scope))


def grid_sync(sym_layout: SymLayout, thread_id, block_id, num_blocks):
    count_ptr = sym_layout.grid_sync_count_ptr  # i64 base ptr of the grid-sync counter
    fx.gpu.barrier()  # align threads in this block
    fence_release(scope="agent")  # publish writes to other CUs
    if thread_id == fx.Int32(0):
        add_value = fx.arith.select(
            block_id == fx.Int32(0),
            fx.Int32(_FINISH_SUM_TAG - (num_blocks - 1)),
            fx.Int32(1),
        )
        old_value = atomic_add(count_ptr, fx.Int32(0), add_value, scope="agent", release=True)
        new_value = ld(count_ptr, fx.Int32(0), scope="agent", order="acquire")
        while ((new_value ^ old_value) & fx.Int32(_FINISH_SUM_TAG)) == fx.Int32(0):
            new_value = ld(count_ptr, fx.Int32(0), scope="agent", order="acquire")
    fx.gpu.barrier()  # broadcast "all arrived"
    fence_acquire(scope="agent")  # invalidate L1 for fresh reads


# Lower if/while in grid_sync to scf.
grid_sync = ASTRewriter.transform(grid_sync)
