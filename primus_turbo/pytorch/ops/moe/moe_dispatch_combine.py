###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from typing import List, Tuple, Union

import torch

from primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl import (
    moe_combine_impl,
    moe_dispatch_impl,
)


class MoEDispatch(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        token_indices: torch.Tensor,
        token_probs: torch.Tensor,
        num_experts: int,
        group,
        async_finish,
        allocate_on_comm_stream,
        num_worst_tokens,
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        Union[List, torch.Tensor],
        Tuple,
    ]:
        recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle = moe_dispatch_impl(
            x,
            group,
            None,
            topk_idx=token_indices,
            token_weight=token_probs,
            num_experts=num_experts,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            num_worst_tokens=num_worst_tokens,
        )
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream

        tokens_per_expert = torch.tensor(tokens_per_expert)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle):
        combined_x, combined_topk_weights = moe_combine_impl(
            grad_output,
            ctx.group,
            ctx.handle,
            grad_token_probs,
            ctx.async_finish,
            ctx.allocate_on_comm_stream,
        )
        return combined_x, None, combined_topk_weights, None, None, None, None, None, None


class MoECombine(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, group, handle: Tuple, async_finish=False, allocate_on_comm_stream=False
    ) -> torch.Tensor:
        combined_x, _ = moe_combine_impl(
            x,
            group,
            handle,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x

    @staticmethod
    def backward(ctx, grad_output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        grad_x, _, _, _, _ = moe_dispatch_impl(
            grad_output,
            ctx.group,
            ctx.handle,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        return grad_x, None, None, None, None


def moe_dispatch(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    token_indices: torch.Tensor,
    token_probs: torch.Tensor,
    num_experts: int,
    group,
    async_finish=False,
    allocate_on_comm_stream=False,
    num_worst_tokens: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
    """Perform fused dispatch operation if deep_ep is available.

    Args:
        x: Input tensor [num_tokens, hidden_size]
        token_indices: Token routing indices [num_tokens, topk]
        token_probs: Token routing probabilities [num_tokens, topk]
        num_experts: Number of experts
        group: Process group
        num_worst_tokens: set num_worst_token for deepep dispatch which can elimite host sync
    Returns:
        Result of FusedDispatch
    """
    return MoEDispatch.apply(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish,
        allocate_on_comm_stream,
        num_worst_tokens,
    )


def moe_combine(
    x,
    group,
    handle,
    async_finish=False,
    allocate_on_comm_stream=False,
) -> torch.Tensor:
    """Perform fused combine operation if deep_ep is available.

    Args:
        x: Input tensor
        group: Process group
        handle: Communication handle

    Returns:
        Result of FusedCombine
    """
    return MoECombine.apply(
        x,
        group,
        handle,
        async_finish,
        allocate_on_comm_stream,
    )
