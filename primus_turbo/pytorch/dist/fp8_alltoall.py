from typing import List, Tuple, Union

import torch
import torch.distributed as dist

from primus_turbo.pytorch.core.float8 import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    ScalingStrategy,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.quantize import (
    dequant_fp8_tensorwise_impl,
    quant_fp8_tensorwise_impl,
)

__all__ = ["FP8AllToAll"]


@torch.compile
def calc_scale_and_scale_inv(x: torch.Tensor, fp8_max: float):
    amax = x.abs().amax()
    scale = torch.full([1], fill_value=fp8_max, dtype=torch.float32, device=x.device) / amax
    scale_inv = 1.0 / scale

    return scale, scale_inv


class FP8AllToAll(torch.autograd.Function):
    """
    Split input tensor and then scatter the split list to all processes in a group.

    Later the received tensors are concatenated from all the processes in the group
    and returned as a single output tensor.

    Support fp8 precision.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        input_ (Tensor): Input tensor to scatter.
        output_split_sizes (Union[List, None]): Output split sizes for dim 0
            if specified None or empty, dim 0 of ``output`` tensor must divide
            equally by ``world_size``.
        input_split_sizes (Union[List, None]): Input split sizes for dim 0
            if specified None or empty, dim 0 of ``input`` tensor must divide
            equally by ``world_size``.
        fwd_quant (bool): Apply FP8 quantize on forward pass.
        bwd_quant (bool): Apply FP8 quantize on backward pass.
        config (Float8QuantConfig): Primus-Turbo Float8Config. Only support strategy is ScalingStrategy.DYNAMIC and granularity is ScalingGranularity.TENSORWISE.

    Returns:
        output (Tensor): Gathered concatenated output tensor.
    """

    @staticmethod
    def forward(
        ctx,
        group: dist.group,
        input_: torch.Tensor,
        output_split_sizes: Union[List, None],
        input_split_sizes: Union[List, None],
        fwd_quant: bool,
        bwd_quant: bool,
        config: Float8QuantConfig,
    ) -> torch.Tensor:
        assert group is not None, "group should not be None."
        assert config.strategy == ScalingStrategy.DYNAMIC
        assert config.granularity == ScalingGranularity.TENSORWISE

        allreduce_amax = True

        if config.format == Format.E4M3:
            fwd_fp8_dtype = float8_e4m3
        elif config.format == Format.HYBRID:
            fwd_fp8_dtype = float8_e4m3
        else:
            raise ValueError("FP8AlltoAll only support E4M3 and HYBRID format.")

        orig_dtype = input_.dtype

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_

        input_ = input_.contiguous()
        # quant
        if fwd_quant:
            # pertensor scale. scale = FP8_MAX / amax
            scale, scale_inv = calc_scale_and_scale_inv(input_, torch.finfo(fwd_fp8_dtype).max)
            dist.all_reduce(scale, op=dist.ReduceOp.MIN, group=group)

            input_ = quant_fp8_tensorwise_impl(input_, scale, fwd_fp8_dtype)
            input_ = input_.view(torch.uint8)

        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input_)
        else:
            # Unequal split (all2all-v)
            output = torch.empty(
                size=[sum(output_split_sizes)] + list(input_.size()[1:]),
                dtype=input_.dtype,
                device=input_.device,
            )

        if fwd_quant:
            output = output.view(torch.uint8)

        torch.distributed.all_to_all_single(
            output,
            input_,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )

        if output.nelement() == 0:
            output = torch.empty_like(output, dtype=orig_dtype)
        else:
            # dequant
            if fwd_quant:
                output = output.view(fwd_fp8_dtype)

                if allreduce_amax:
                    # recompute scale_inv
                    scale_inv = 1 / scale

                output = dequant_fp8_tensorwise_impl(output, scale_inv, orig_dtype)

        ctx.config = config
        ctx.group = group
        ctx.orig_dtype = orig_dtype
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.bwd_quant = bwd_quant

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        config = ctx.config
        bwd_quant = ctx.bwd_quant

        allreduce_amax = True

        if config.format == Format.E4M3:
            bwd_fp8_dtype = float8_e4m3
        elif config.format == Format.HYBRID:
            bwd_fp8_dtype = float8_e5m2
        else:
            raise ValueError("FP8AlltoAll only support E4M3 and HYBRID format.")

        orig_dtype = grad_output.dtype

        world_size = torch.distributed.get_world_size(group=ctx.group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return grad_output

        grad_output = grad_output.contiguous()
        # quant
        if bwd_quant:
            scale, scale_inv = calc_scale_and_scale_inv(grad_output, torch.finfo(bwd_fp8_dtype).max)
            dist.all_reduce(scale, op=dist.ReduceOp.MIN, group=ctx.group)

            grad_output = quant_fp8_tensorwise_impl(grad_output, scale, bwd_fp8_dtype)
            grad_output = grad_output.view(torch.uint8)

        if ctx.input_split_sizes is None:
            # Equal split (all2all)
            dgrad = torch.empty_like(grad_output)
        else:
            # Unequal split (all2all-v)
            dgrad = torch.empty(
                size=[sum(ctx.input_split_sizes)] + list(grad_output.size()[1:]),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

        if bwd_quant:
            dgrad = dgrad.view(torch.uint8)

        torch.distributed.all_to_all_single(
            dgrad,
            grad_output,
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group,
        )

        if dgrad.nelement() == 0:
            dgrad = torch.empty_like(dgrad, dtype=orig_dtype)
        else:
            # dequant
            if bwd_quant:
                dgrad = dgrad.view(bwd_fp8_dtype)

                if allreduce_amax:
                    # recompute sclae_inv
                    scale_inv = 1 / scale

                dgrad = dequant_fp8_tensorwise_impl(dgrad, scale_inv, orig_dtype)

        return (
            None,
            dgrad,
            None,
            None,
            None,
            None,
            None,
        )
