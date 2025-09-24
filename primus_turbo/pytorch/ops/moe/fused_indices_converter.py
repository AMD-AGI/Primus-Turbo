###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import math

import torch

from primus_turbo.triton.moe.multihot_to_indices import (
    _indices_to_multihot_kernel,
    _multihot_to_indices_kernel,
)


class IndicesToMultihot(torch.autograd.Function):
    """Convert moe topk indices to multihot representation.

    This class implements a custom forward and backward propagation
    operation for efficiently converting indices to multihot
    representation.
    It is an experimental feature and may change in future versions.
    """

    @staticmethod
    def forward(ctx, indices, probs_indices, num_of_local_experts):
        """Forward function for IndicesToMultihot

        Convert indices to multihot representation.

        Args:
            indices: [num_of_tokens, topk]
            probs_indices: [num_of_tokens, topk]
            num_of_local_experts: int

        Returns:
            multihot_indices: [num_of_tokens, num_of_local_experts]
            probs_in_multihot: [num_of_tokens, num_of_local_experts]
        """
        num_of_tokens = indices.shape[0]
        assert indices.shape == probs_indices.shape, "indices and probs_indices must have the same shape"
        topk = indices.shape[1]
        multihot_indices = torch.empty((num_of_tokens, num_of_local_experts), dtype=torch.bool, device="cuda")
        probs_in_multihot = torch.empty(
            (num_of_tokens, num_of_local_experts), dtype=probs_indices.dtype, device="cuda"
        )
        position_map = torch.empty((num_of_tokens, num_of_local_experts), dtype=torch.int32, device="cuda")
        # Compute the next power of 2 for the topk and num_of_local_experts
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(num_of_local_experts)))
        grid = (num_of_tokens,)
        _indices_to_multihot_kernel[grid](
            indices,
            probs_indices,
            multihot_indices,
            probs_in_multihot,
            position_map,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,  # use only 1 warp per block
            num_warps=1,
        )

        ctx.save_for_backward(position_map)
        ctx.num_of_tokens = num_of_tokens
        ctx.num_of_local_experts = num_of_local_experts
        ctx.topk = topk
        return multihot_indices, probs_in_multihot

    @staticmethod
    def backward(ctx, grad_multihot_indices, grad_probs_in_multihot):
        """Backward function for IndicesToMultihot

        Convert multihot probs representation to indices.
        indices is ignored in the backward function.

        Args:
            grad_multihot_indices: [num_of_tokens, num_of_local_experts]
            grad_probs_in_multihot: [num_of_tokens, num_of_local_experts]

        Returns:
            grad_probs_indices: [num_of_tokens, topk]
        """
        position_map = ctx.saved_tensors[0]
        num_of_tokens = ctx.num_of_tokens
        num_of_local_experts = ctx.num_of_local_experts
        topk = ctx.topk

        # Initialize the gradient of the indices and probs_indices
        grad_probs_indices = torch.empty(
            (num_of_tokens, topk), dtype=grad_probs_in_multihot.dtype, device="cuda"
        )
        # Compute the next power of 2 for the topk and num_of_local_experts
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(num_of_local_experts)))

        grid = (num_of_tokens,)
        _multihot_to_indices_kernel[grid](
            # if the grad_probs_in_multihot is all-one/all-zero,
            # overlapping stride will cause error without contiguous()
            grad_probs_in_multihot.contiguous(),
            position_map,
            grad_probs_indices,
            num_of_local_experts,
            num_of_local_experts_next_power_of_2,
            topk,
            topk_next_power_of_2,
            BLOCK_SIZE=32,  # use only 1 warp per block
            num_warps=1,
        )
        return None, grad_probs_indices, None, None


def fused_indices_to_multihot(indices, probs_indices, num_of_local_experts):
    return IndicesToMultihot.apply(indices, probs_indices, num_of_local_experts)
