###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from primus_turbo.pytorch.dist import dma_all_gather_into_tensor


@instantiate_parametrized_tests
class AllGatherTestCase(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("dtype", [torch.float32])
    def test_dma_all_gather(
        self,
        dtype: torch.dtype,
    ) -> None:
        self._init_process()

        def get_profiler_context():
            ENABLE_PROFILER = False
            if not ENABLE_PROFILER or self.rank != 0:
                from contextlib import nullcontext

                return nullcontext()
            schedule = torch.profiler.schedule(
                wait=5,
                warmup=5,
                active=2,
                repeat=1,
            )
            profiler = torch.profiler.profile(
                schedule=schedule,
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=False,
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join("./profile_results/")),
            )
            return profiler

        num_elems = 16 * 1024 * 1024
        input_tensor = torch.full([num_elems], self.rank, dtype=dtype, device=self.device)
        output_tensor = torch.zeros([num_elems * self.world_size], dtype=dtype, device=self.device)
        output_tensor_ref = torch.ones([num_elems * self.world_size], dtype=dtype, device=self.device)

        with get_profiler_context() as prof:
            for i in range(64):
                dist.all_gather_into_tensor(output_tensor_ref, input_tensor)
                dma_all_gather_into_tensor(output_tensor, input_tensor)
                if prof:
                    prof.step()

        torch.testing.assert_close(output_tensor, output_tensor_ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    run_tests()
