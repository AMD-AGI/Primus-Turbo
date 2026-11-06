###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize

from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, Format
from primus_turbo.pytorch.dist import FP8AllToAll


@instantiate_parametrized_tests
class FP8AllToAllSmallTest(MultiProcessTestCase):
    """Small transport and autograd checks, collected outside pytest's ignored dist/ directory."""

    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(2)
    @parametrize("format", [Format.E4M3, Format.HYBRID])
    @parametrize("quantize", [(False, False), (True, False), (False, True), (True, True)])
    @parametrize("unequal", [False, True])
    def test_forward_and_backward(self, format, quantize, unequal):
        torch.cuda.set_device(self.rank)
        dist.init_process_group(
            "nccl",
            store=dist.FileStore(self.file_name, self.world_size),
            rank=self.rank,
            world_size=self.world_size,
        )
        try:
            # Rank-distinct permutations share an amax, keeping scale agreement
            # separate from byte transport and the inverse backward split layout.
            base = torch.arange(128, device="cuda", dtype=torch.float32).reshape(16, 8) / 16
            x = (base if self.rank == 0 else base.flip(1)).clone().requires_grad_()
            input_splits = [4, 12] if unequal else None
            output_splits = ([4, 4] if self.rank == 0 else [12, 12]) if unequal else None
            expected = x.new_empty((sum(output_splits), 8)) if unequal else torch.empty_like(x)
            dist.all_to_all_single(expected, x.detach(), output_splits, input_splits)
            actual = FP8AllToAll.apply(
                dist.group.WORLD, x, output_splits, input_splits, *quantize, Float8QuantConfig(format=format)
            )
            assert actual.shape == expected.shape and actual.dtype == x.dtype
            torch.testing.assert_close(
                actual, expected, rtol=0.07 if quantize[0] else 0, atol=0.02 if quantize[0] else 0
            )
            grad = torch.ones_like(actual)
            expected_dx = torch.empty_like(x)
            dist.all_to_all_single(expected_dx, grad, input_splits, output_splits)
            # coverage.py's thread tracer misses Python callbacks on autograd's
            # native worker threads. Run this small gradient check on the caller.
            with torch.autograd.set_multithreading_enabled(False):
                actual.backward(grad)
            torch.testing.assert_close(
                x.grad, expected_dx, rtol=0.15 if quantize[1] else 0, atol=0.01 if quantize[1] else 0
            )
        finally:
            dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    @parametrize("zero_input", [False, True])
    def test_empty_peer_split(self, zero_input):
        torch.cuda.set_device(self.rank)
        dist.init_process_group(
            "nccl",
            store=dist.FileStore(self.file_name, self.world_size),
            rank=self.rank,
            world_size=self.world_size,
        )
        try:
            if zero_input:
                input_splits = [0, 0] if self.rank == 0 else [16, 16]
                output_splits = [0, 16]
            else:
                input_splits = [0, 16]
                output_splits = [0, 0] if self.rank == 0 else [16, 16]
            x = (
                torch.arange(sum(input_splits) * 8, device="cuda", dtype=torch.float32)
                .reshape(-1, 8)
                .requires_grad_()
            )
            expected = x.new_empty((sum(output_splits), 8))
            dist.all_to_all_single(expected, x.detach(), output_splits, input_splits)
            actual = FP8AllToAll.apply(
                dist.group.WORLD, x, output_splits, input_splits, False, False, Float8QuantConfig()
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            grad = torch.ones_like(actual)
            expected_dx = torch.empty_like(x)
            dist.all_to_all_single(expected_dx, grad, input_splits, output_splits)
            with torch.autograd.set_multithreading_enabled(False):
                actual.backward(grad)
            torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)
        finally:
            dist.destroy_process_group()
