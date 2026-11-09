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

"""Symmetric workspace shared by the Ulysses GemmA2A / A2AGemm kernels of one group.

Heap layout (identical on every rank):

    [flag regions: NUM_FLAG_REGIONS x FLAGS_PER_REGION i64][bank 0][bank 1]

Every fused kernel reads ``epoch`` from ``state[0]`` (starts at 1), uses bank ``epoch & 1`` and
publishes flags whose value is ``epoch``; waiters spin until ``flag >= epoch``. The last block to
exit bumps ``state[0]``. A rank's call ``e`` cannot finish before every peer has produced its call-``e``
data, so when a rank starts call ``e + 2`` every peer has finished call ``e`` and bank ``e & 1`` is
free again: two banks need no acknowledgement or barrier. This holds as long as, on every rank, all
calls on one group are stream-ordered.
"""

from typing import Optional

import torch
import torch.distributed as dist

from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

FLAGS_PER_REGION = 1 << 15
NUM_FLAG_REGIONS = 4
FLAG_REGION_BYTES = FLAGS_PER_REGION * 8
FLAG_BYTES = NUM_FLAG_REGIONS * FLAG_REGION_BYTES
BANK_ALIGN = 1 << 16

# flag region of each fused op; ops never share flag slots
REGION_QKV_GEMM_A2A = 0
REGION_O_GEMM_A2A = 1
REGION_O_A2A_GEMM = 2
REGION_QKV_A2A_GEMM = 3


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class UlyssesWorkspace:
    def __init__(self, group: dist.ProcessGroup, bank_bytes: int):
        self.group = group
        self.rank = group.rank()
        self.num_ranks = group.size()
        self.bank_bytes = _align(bank_bytes, BANK_ALIGN)
        self.num_bytes = FLAG_BYTES + 2 * self.bank_bytes
        self.symm_mem = SymmetricMemory(group, self.num_bytes, signal_pad_size=0)
        self.peer_ptrs = torch.tensor(self.symm_mem.buffer_ptrs, dtype=torch.int64, device="cuda")
        # [0] epoch, [1] exit counter, [2] / [3] push-unit / tile claim counters of the running fused kernel
        self.state = torch.tensor([1, 0, 0, 0], dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()
        dist.barrier(group=group, device_ids=[torch.cuda.current_device()])

    def flags(self, rank: int, region: int) -> torch.Tensor:
        """Host view of one flag region of ``rank``'s heap (tests and diagnostics)."""
        return self.symm_mem.get_buffer(
            rank, (FLAGS_PER_REGION,), torch.int64, storage_offset=region * FLAGS_PER_REGION
        )

    def destroy(self) -> None:
        if self.symm_mem is not None:
            # peers may still read this heap until every rank is past its last fused kernel
            torch.cuda.synchronize()
            dist.barrier(group=self.group, device_ids=[torch.cuda.current_device()])
            self.symm_mem.destroy()
        self.symm_mem = None


_WORKSPACES: dict = {}


def get_ulysses_workspace(group: dist.ProcessGroup, bank_bytes: int) -> UlyssesWorkspace:
    """The group's workspace, regrown (collectively, outside graph capture) when a bank is too small."""
    ws: Optional[UlyssesWorkspace] = _WORKSPACES.get(group.group_name)
    if ws is not None and ws.bank_bytes >= bank_bytes:
        return ws
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"the Ulysses fused-a2a workspace needs {bank_bytes} bytes per bank but has "
            f"{0 if ws is None else ws.bank_bytes}; run one eager step before CUDA graph capture"
        )
    old = ws
    # grow geometrically; allocate before freeing, since a heap that reuses just-freed memory
    # cannot be exported over IPC
    ws = UlyssesWorkspace(group, max(bank_bytes, 0 if old is None else 2 * old.bank_bytes))
    if old is not None:
        old.destroy()
    _WORKSPACES[group.group_name] = ws
    return ws
