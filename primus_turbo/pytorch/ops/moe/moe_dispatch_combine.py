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
            token_weights=token_probs,
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
    """
    MoE dispatch operation: distributes input tokens to their assigned experts.

    This function is the first stage of MoE forward pass, routing tokens to the experts
    they have been assigned to based on the router's decisions.
    Supports autograd; the backward pass calls moe_combine to gather gradients.

    Args:
        x: Input tensor of shape (num_tokens, hidden_size), or a tuple
           (hidden_states, auxiliary_states) for mixed-precision scenarios.
        token_indices: Token-to-expert mapping indices of shape (num_tokens, top_k),
                       indicating which top-k experts each token is routed to.
        token_probs: Routing probabilities/weights of shape (num_tokens, top_k),
                     representing the weight assigned to each expert for each token.
        num_experts: Total number of experts.
        group: Distributed communication group for cross-device/node All-to-All communication.
        async_finish: Whether to enable async completion mode, allowing communication
                      to overlap with computation. Defaults to False.
        allocate_on_comm_stream: Whether to allocate memory on the communication stream,
                                 which can optimize memory access patterns. Defaults to False.
        num_worst_tokens: Number of low-quality tokens to drop (for load balancing optimization).
                          Defaults to 0.

    Returns:
        A tuple containing:
            - recv_x: Dispatched token tensor received by this rank, to be processed by local experts.
            - recv_token_indices: Received token index information.
            - recv_token_probs: Received token probability weights.
            - tokens_per_expert: Statistics of token count received by each expert.
            - handle: Communication handle that must be passed to moe_combine to complete the combine operation.
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
    """
    MoE combine operation: merges expert outputs back to the original token order.

    This function is the final stage of MoE forward pass. After all experts have completed
    their computations, it collects and combines the outputs scattered across different devices
    back to their original token positions.
    Supports autograd; the backward pass calls moe_dispatch to distribute gradients.

    Args:
        x: Expert output tensor, shape depends on the number of tokens processed by local experts.
        group: Distributed communication group for cross-device/node All-to-All communication.
               Must be the same communication group used in moe_dispatch.
        handle: Communication handle from moe_dispatch, containing metadata from the dispatch stage
                used to correctly route outputs back to their original positions.
        async_finish: Whether to enable async completion mode, allowing communication
                      to overlap with computation. Defaults to False.
        allocate_on_comm_stream: Whether to allocate memory on the communication stream,
                                 which can optimize memory access patterns. Defaults to False.

    Returns:
        torch.Tensor: Combined output tensor with the same shape as the input x to moe_dispatch,
                      i.e., (num_tokens, hidden_size). Each token's output is the weighted sum
                      of outputs from the experts it was routed to (weights from token_probs).
    """
    return MoECombine.apply(
        x,
        group,
        handle,
        async_finish,
        allocate_on_comm_stream,
    )
