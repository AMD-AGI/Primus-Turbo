from typing import List, Tuple, Union

import torch
import torch.distributed as dist

from primus_turbo.pytorch.core import float8

__all__ = ["FP8AllToAll"]


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
        fp8_format (Format): Format of fp8 quantize.
        allreduce_amax (bool): Applying allreduce-max of input_'s amax.

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
        fp8_format: float8.Format,
        allreduce_amax: bool,
    ) -> torch.Tensor:
        assert group is not None, "group should not be None."

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_

        orig_dtype = input_.dtype

        # quant
        input_ = input_.contiguous()
        if not allreduce_amax:
            # scale = 1.0
            scale = torch.full([1], fill_value=1.0, device=input_.device)
        else:
            # pertensor scale. scale = FP8_MAX / amax
            amax = input_.abs().max()
            dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=group)
            scale = torch.full(
                [1], fill_value=torch.finfo(fp8_format.value.fwd_dtype).max / amax, device=input_.device
            )
        input_fp8 = torch.ops.primus_turbo_cpp_extension.fp8_quantize(
            input_, scale, fp8_format.value.fwd_dtype
        )

        if output_split_sizes is None:
            # Equal split (all2all)
            output_fp8 = torch.empty_like(input_fp8)
        else:
            # Unequal split (all2all-v)
            output_fp8 = torch.empty(
                size=[sum(output_split_sizes)] + list(input_fp8.size()[1:]),
                dtype=input_fp8.dtype,
                device=input_fp8.device,
            )
        input_fp8 = input_fp8.view(torch.uint8)
        output_fp8 = output_fp8.view(torch.uint8)
        torch.distributed.all_to_all_single(
            output_fp8,
            input_fp8,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        output_fp8 = output_fp8.view(fp8_format.value.fwd_dtype)

        # dequant
        if output_fp8.nelement() == 0:
            output = torch.empty_like(output_fp8, dtype=orig_dtype)
        else:
            if not allreduce_amax:
                scale_inv = torch.full([1], fill_value=1.0, device=output_fp8.device)
            else:
                scale_inv = torch.full(
                    [1],
                    fill_value=amax / torch.finfo(fp8_format.value.fwd_dtype).max,
                    device=output_fp8.device,
                )
            output = torch.ops.primus_turbo_cpp_extension.fp8_dequantize(output_fp8, scale_inv, orig_dtype)

        ctx.fp8_format = fp8_format
        ctx.group = group
        ctx.orig_dtype = orig_dtype
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.allreduce_amax = allreduce_amax

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        world_size = torch.distributed.get_world_size(group=ctx.group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return grad_output

        orig_dtype = grad_output.dtype

        grad_output = grad_output.contiguous()
        if not ctx.allreduce_amax:
            # scale = 1.0
            scale = torch.full([1], fill_value=1.0, device=grad_output.device)
        else:
            # pertensor scale. scale = FP8_MAX / amax
            amax = grad_output.abs().max()
            dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=ctx.group)
            scale = torch.full(
                [1],
                fill_value=torch.finfo(ctx.fp8_format.value.bwd_dtype).max / amax,
                device=grad_output.device,
            )
        grad_output_fp8 = torch.ops.primus_turbo_cpp_extension.fp8_quantize(
            grad_output, scale, ctx.fp8_format.value.bwd_dtype
        )

        if ctx.input_split_sizes is None:
            # Equal split (all2all)
            dgrad_fp8 = torch.empty_like(grad_output_fp8)
        else:
            # Unequal split (all2all-v)
            dgrad_fp8 = torch.empty(
                size=[sum(ctx.input_split_sizes)] + list(grad_output_fp8.size()[1:]),
                dtype=grad_output_fp8.dtype,
                device=grad_output_fp8.device,
            )
        grad_output_fp8 = grad_output_fp8.view(torch.uint8)
        dgrad_fp8 = dgrad_fp8.view(torch.uint8)
        torch.distributed.all_to_all_single(
            dgrad_fp8,
            grad_output_fp8,
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group,
        )
        dgrad_fp8 = dgrad_fp8.view(ctx.fp8_format.value.bwd_dtype)

        if dgrad_fp8.nelement() == 0:
            dgrad = torch.empty_like(dgrad_fp8, dtype=orig_dtype)
        else:
            if not ctx.allreduce_amax:
                scale_inv = torch.full([1], fill_value=1.0, device=dgrad_fp8.device)
            else:
                scale_inv = torch.full(
                    [1],
                    fill_value=amax / torch.finfo(ctx.fp8_format.value.bwd_dtype).max,
                    device=dgrad_fp8.device,
                )
            dgrad = torch.ops.primus_turbo_cpp_extension.fp8_dequantize(dgrad_fp8, scale_inv, orig_dtype)

        return (
            None,
            dgrad,
            None,
            None,
            None,
            None,
            None,
        )
