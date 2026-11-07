###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two-GPU all-gather GEMM with caller-owned outputs and optional gathered input."""

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize

from primus_turbo.pytorch.ops import fused_all_gather_matmul
from tests.pytorch.test_utils import get_tolerances


@instantiate_parametrized_tests
class AsyncTPOutputBufferTest(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(2)
    @parametrize("method", ["pipeline", "ring_exchange"])
    @parametrize("return_a", [False, True])
    def test_external_buffers(self, method, return_a):
        torch.cuda.set_device(self.rank)
        dist.init_process_group("nccl", store=dist.FileStore(self.file_name, 2), rank=self.rank, world_size=2)
        try:
            torch.manual_seed(811 + self.rank)
            a = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)
            bs = [torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
            shards = [torch.empty_like(a) for _ in range(2)]
            dist.all_gather(shards, a)
            expected_a = torch.cat(shards, dim=1)
            expected = [expected_a @ b.T for b in bs]
            a_buffer = torch.empty(256, 128, device="cuda", dtype=a.dtype)
            c_buffers = [torch.full((256, 64), float("nan"), device="cuda", dtype=a.dtype) for _ in bs]
            if method == "ring_exchange":
                with pytest.raises(NotImplementedError, match="not support yet"):
                    fused_all_gather_matmul(
                        a,
                        bs,
                        ["NT", "NT"],
                        1,
                        dist.group.WORLD.group_name,
                        [torch.cuda.current_stream()],
                        [torch.cuda.Stream()],
                        [torch.cuda.Stream()],
                        comm_method=method,
                    )
                return
            result_a, result_c = fused_all_gather_matmul(
                a,
                bs,
                ["NT", "NT"],
                1,
                dist.group.WORLD.group_name,
                [torch.cuda.current_stream()],
                [torch.cuda.Stream()],
                [torch.cuda.Stream()],
                comm_method=method,
                return_A=return_a,
                A_out=a_buffer,
                outputs=c_buffers,
                out_dtypes=[a.dtype, a.dtype],
            )
            torch.cuda.synchronize()
            if return_a:
                assert result_a.data_ptr() == a_buffer.data_ptr()
                torch.testing.assert_close(result_a, expected_a, rtol=0, atol=0)
            else:
                assert result_a is None
            for actual, want, buffer in zip(result_c, expected, c_buffers):
                assert actual.data_ptr() == buffer.data_ptr()
                torch.testing.assert_close(actual, want, **get_tolerances(a.dtype))
        finally:
            dist.destroy_process_group()
