###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Tuple

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

from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.dist import FP8AllToAll
from tests.pytorch.test_utils import get_tolerances

shapes = [
    # (b, s, h, topk)
    (1, 8192, 7168, 6)
]


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = group.size()
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None):
    """Wrapper for autograd function"""
    assert group is not None, "group should not be None"
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes)


def fp8_all_to_all(
    group,
    input_,
    output_split_sizes_=None,
    input_split_sizes=None,
    fwd_quant=True,
    bwd_quant=True,
    config=None,
):
    """Wrapper for autograd function"""

    if config is None:
        config = Float8QuantConfig()

    args = (
        group,
        input_,
        output_split_sizes_,
        input_split_sizes,
        fwd_quant,
        bwd_quant,
        config,
    )

    return FP8AllToAll.apply(*args)


@instantiate_parametrized_tests
class FP8AlltoAllTestCase(MultiProcessTestCase):
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
        torch.manual_seed(42 + self.rank)

    @skip_if_lt_x_gpu(2)
    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @parametrize("shape", shapes)
    @parametrize("fwd_bwd_quant", [(True, True)])
    def test_fp8_alltoall(
        self,
        dtype: torch.dtype,
        shape: Tuple[int, int, int, int, int],
        fwd_bwd_quant: bool,
    ) -> None:
        self._init_process()

        group = dist.group.WORLD
        fwd_quant, bwd_quant = fwd_bwd_quant

        # check forward pass
        inp = torch.rand(shape, dtype=dtype, device=self.device, requires_grad=True)
        inp_ref = inp.detach().clone().requires_grad_()

        config = Float8QuantConfig()
        out = fp8_all_to_all(
            group,
            inp,
            fwd_quant=fwd_quant,
            bwd_quant=bwd_quant,
            config=config,
        )
        out_ref = all_to_all(group, inp_ref)

        fwd_fp8_dtype = float8_e4m3
        if config.format == Format.HYBRID:
            bwd_fp8_dtype = float8_e5m2
        elif config.format == Format.E4M3:
            bwd_fp8_dtype = float8_e4m3

        torch.testing.assert_close(out, out_ref, **get_tolerances(fwd_fp8_dtype))

        grad_out = torch.ones_like(out)
        out.backward(grad_out)

        grad_out_ref = torch.ones_like(out_ref)
        out_ref.backward(grad_out_ref)

        torch.testing.assert_close(inp.grad, inp_ref.grad, **get_tolerances(bwd_fp8_dtype))


if __name__ == "__main__":
    run_tests()
