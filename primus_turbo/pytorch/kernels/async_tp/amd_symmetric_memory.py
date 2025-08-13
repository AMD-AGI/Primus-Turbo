###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import operator
from functools import reduce
from typing import Tuple

import torch
import torch.distributed.distributed_c10d as c10d

from .common_ops import barrier_all_on_stream, ipc_create_tensor_lists


class AMDSymmetricMemory:
    def __init__(self, group_name: str, mem_bytes: int):
        group = c10d._resolve_process_group(group_name)

        self._buffers = ipc_create_tensor_lists(group, [mem_bytes], torch.uint8)

        self._comm_bufs = ipc_create_tensor_lists(group, [group.size()], torch.int32)
        self._comm_bufs[group.rank()].fill_(0)
        self._comm_buf_ptr = torch.tensor(
            [t.data_ptr() for t in self._comm_bufs], device=torch.cuda.current_device(), requires_grad=False
        )
        self._group = group
        self._buffer_size = mem_bytes

        torch.cuda.synchronize()
        self.barrier()

    def get_buffer(
        self,
        rank: int,
        sizes: torch.types._size,
        dtype: torch.dtype,
        storage_offset: int | None = 0,
    ) -> torch.Tensor:
        required_len = reduce(operator.mul, sizes, 1)
        start = storage_offset * dtype.itemsize
        end = start + required_len * dtype.itemsize
        assert (
            end <= self._buffer_size
        ), f"request size {end} with storage_offset {storage_offset} exceed buffer size: {self._buffer_size}"
        buffer = self._buffers[rank][start:end].view(dtype).view(*sizes)
        return buffer

    def barrier(self) -> None:
        barrier_all_on_stream(self.rank, self.world_size, self._comm_buf_ptr, torch.cuda.current_stream())

    @property
    def rank(self):
        return self._group.rank()

    @property
    def world_size(self):
        return self._group.size()

    @property
    def buffer_size(self) -> int:
        return self._buffer_size


_group_name_to_workspace_tensor: dict[Tuple[str, int], AMDSymmetricMemory] = {}


def get_amd_symm_mem_workspace(group_name: str, min_size) -> AMDSymmetricMemory:
    global _group_name_to_workspace_tensor

    symm = _group_name_to_workspace_tensor.get(group_name, None)

    if symm is None or symm.buffer_size < min_size:
        symm = AMDSymmetricMemory(group_name, min_size)
        _group_name_to_workspace_tensor[group_name] = symm

    return symm
