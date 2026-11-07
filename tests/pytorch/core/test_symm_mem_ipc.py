###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Real two-GPU IPC views, visibility barriers and workspace replacement."""

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize

from primus_turbo.pytorch.core import symm_mem


@instantiate_parametrized_tests
class SymmetricMemoryIPCTest(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(2)
    @parametrize("uncached", [False, True])
    def test_peer_write_and_workspace_growth(self, uncached):
        torch.cuda.set_device(self.rank)
        dist.init_process_group("nccl", store=dist.FileStore(self.file_name, 2), rank=self.rank, world_size=2)
        group = dist.group.WORLD
        try:
            with symm_mem.SymmetricMemory(
                group, 4096, signal_pad_size=128, uncached_signal_pad=uncached, uncached_buffer=uncached
            ) as memory:
                for dtype in (torch.float32, torch.bfloat16, torch.int32):
                    own = memory.get_buffer(self.rank, (16,), dtype, storage_offset=4)
                    peer = memory.get_buffer(1 - self.rank, (16,), dtype, storage_offset=4)
                    own.fill_(self.rank + 1)
                    memory.barrier()
                    torch.testing.assert_close(peer, torch.full_like(peer, 2 - self.rank), rtol=0, atol=0)
                    memory.barrier()
                    peer.fill_(10 + self.rank)
                    memory.barrier()
                    torch.testing.assert_close(own, torch.full_like(own, 11 - self.rank), rtol=0, atol=0)
                    memory.barrier()
            assert memory.is_destroyed
            memory.destroy()
            first = symm_mem.get_symm_mem_workspace(group, 1024, 128)
            assert symm_mem.get_symm_mem_workspace(group, 512, 64) is first
            grown = symm_mem.get_symm_mem_workspace(group, 2048, 256)
            assert grown is not first and first.is_destroyed
            assert grown.buffer_size >= 2048 and grown.signal_pad_size >= 256
            grown.barrier()
            grown.destroy()
            symm_mem._group_name_to_workspace_tensor.pop(group.group_name, None)
        finally:
            dist.destroy_process_group()
