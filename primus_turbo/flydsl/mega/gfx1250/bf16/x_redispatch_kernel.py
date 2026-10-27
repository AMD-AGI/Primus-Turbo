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

"""Cross-rank completion flags for the copy-engine x re-dispatch of the stage-1 backward.

Every rank copies its whole x into each rank's x buffer at rows ``rank * T_max`` with peer copies on
a side stream. ``x_copy_signal`` runs behind those copies: it bumps this rank's epoch and stores it
into ``flags[RANK]`` of every rank. ``x_copy_wait`` runs on the consumer stream after it has waited on
the local side stream, and holds until ``flags[s]`` of this rank reached that epoch for every source s.

Flags region (i64): ``[0, NUM_RANKS)`` the per-source flags, ``NUM_RANKS`` this rank's epoch.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.typing import Constexpr

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import _as_ptr
from primus_turbo.flydsl.mega.gfx1250.ep_intranode import spin_until_flag_reaches
from primus_turbo.flydsl.mega.gfx1250.prims import ld, st
from primus_turbo.pytorch.core.symm_mem import SymmetricMemory


def _addr(p):
    return fx.Int64(fx.ptrtoint(p))


@flyc.jit
def _launch_x_copy_signal(
    arg_peer_flags: fx.Pointer,
    arg_local_flags: fx.Pointer,
    stream: fx.Stream,
    NUM_RANKS: Constexpr[int],
    RANK: Constexpr[int],
):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def x_copy_signal(arg_peer_flags: fx.Pointer, arg_local_flags: fx.Pointer):
        lane = fx.Int32(fx.thread_idx.x)
        if lane < fx.Int32(NUM_RANKS):
            epoch = ld(_addr(arg_local_flags), fx.Int32(NUM_RANKS), dtype=fx.T.i64) + fx.Int64(1)
            flag_base = ld(_addr(arg_peer_flags), lane, dtype=fx.T.i64)
            st(flag_base, fx.Int32(RANK), epoch, scope="sys")
            if lane == fx.Int32(0):
                st(_addr(arg_local_flags), fx.Int32(NUM_RANKS), epoch)

    x_copy_signal(arg_peer_flags, arg_local_flags).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.jit
def _launch_x_copy_wait(
    arg_local_flags: fx.Pointer,
    stream: fx.Stream,
    NUM_RANKS: Constexpr[int],
    RANK: Constexpr[int],
):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def x_copy_wait(arg_local_flags: fx.Pointer):
        lane = fx.Int32(fx.thread_idx.x)
        if lane < fx.Int32(NUM_RANKS):
            flags = _addr(arg_local_flags)
            epoch = ld(flags, fx.Int32(NUM_RANKS), dtype=fx.T.i64)
            spin_until_flag_reaches(flags, lane, epoch, "sys", RANK, "x_redispatch")

    x_copy_wait(arg_local_flags).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


_compiled: dict = {}


def _run(launcher, args, num_ranks: int, rank: int) -> None:
    key = (id(launcher), num_ranks, rank)
    fn = _compiled.get(key)
    if fn is None:
        fn = flyc.compile(launcher, *args)
        _compiled[key] = fn
    fn(*args)


def x_copy_signal(
    peer_flag_table: torch.Tensor, local_flags: torch.Tensor, num_ranks: int, rank: int
) -> None:
    """Publish this call's epoch to every rank; ``peer_flag_table[d]`` is rank d's flags address."""
    args = (_as_ptr(peer_flag_table), _as_ptr(local_flags), torch.cuda.current_stream(), num_ranks, rank)
    _run(_launch_x_copy_signal, args, num_ranks, rank)


def x_copy_wait(local_flags: torch.Tensor, num_ranks: int, rank: int) -> None:
    """Hold the current stream until every source published this rank's current epoch."""
    args = (_as_ptr(local_flags), torch.cuda.current_stream(), num_ranks, rank)
    _run(_launch_x_copy_wait, args, num_ranks, rank)


_FLAGS_ALIGN = 256


class XRedispatch:
    """A second x receive buffer of one SymmBuffer, filled by peer copies on a side stream.

    ``push`` must run after the stage-2 backward dispatch of dy on this rank: that dispatch waits on
    every peer's dy, so every peer has finished the dW1 that last read its buffer."""

    def __init__(self, symm):
        self.rank, self.num_ranks = symm.rank, symm.num_ranks
        self.rows_per_rank = symm.num_max_tokens_per_rank
        rows = self.num_ranks * self.rows_per_rank
        flags_offset = -(-rows * symm.hidden * 2 // _FLAGS_ALIGN) * _FLAGS_ALIGN
        self.mem = SymmetricMemory(symm.group, flags_offset + _FLAGS_ALIGN, signal_pad_size=0)
        self.buffers = [
            self.mem.get_buffer(r, (rows, symm.hidden), torch.bfloat16) for r in range(self.num_ranks)
        ]
        self.local_flags = self.mem.get_buffer(
            self.rank, (self.num_ranks + 1,), torch.int64, storage_offset=flags_offset // 8
        )
        self.peer_flag_table = torch.tensor(
            [base + flags_offset for base in self.mem.buffer_ptrs], dtype=torch.int64, device="cuda"
        )
        self.stream = torch.cuda.Stream()

    def push(self, x: torch.Tensor) -> None:
        """Copy x into rows ``rank * T_max`` of every rank's buffer, behind the current stream's work."""
        self.stream.wait_stream(torch.cuda.current_stream())
        start = self.rank * self.rows_per_rank
        with torch.cuda.stream(self.stream):
            for i in range(self.num_ranks):
                self.buffers[(self.rank + i) % self.num_ranks][start : start + x.shape[0]].copy_(x)
            x_copy_signal(self.peer_flag_table, self.local_flags, self.num_ranks, self.rank)
        x.record_stream(self.stream)

    def wait(self) -> torch.Tensor:
        """This rank's buffer once every rank's copy into it landed; orders the current stream behind it."""
        torch.cuda.current_stream().wait_stream(self.stream)
        x_copy_wait(self.local_flags, self.num_ranks, self.rank)
        return self.buffers[self.rank]


_x_redispatch: dict = {}


def get_x_redispatch(symm) -> XRedispatch:
    state = _x_redispatch.get(symm.key)
    if state is None:
        state = XRedispatch(symm)
        _x_redispatch[symm.key] = state
    return state
