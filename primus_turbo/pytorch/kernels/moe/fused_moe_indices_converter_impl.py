###############################################################################
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Adapted from NVIDIA Megatron-LM (https://github.com/NVIDIA/Megatron-LM),
#   file megatron/core/fusions/fused_indices_converter.py.
# Modified by the Primus-Turbo team.
#
# This file is distributed under the 3-clause BSD license used by Megatron-LM,
# not the MIT license that covers the rest of Primus-Turbo. Both texts are in
# LICENSE.
###############################################################################

import math
from typing import Optional, Tuple

import torch

from primus_turbo.triton.moe.multihot_to_indices import (
    _indices_to_multihot_kernel,
    _multihot_to_indices_kernel,
)

__all__ = [
    "fused_moe_indices_converter_forward_impl",
    "fused_moe_indices_converter_backward_impl",
]


def fused_moe_indices_converter_forward_impl(
    topk_indices: torch.Tensor,
    topk_probs: Optional[torch.Tensor],
    num_local_experts: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Convert top-k indices and optional probabilities to multihot layout."""
    num_tokens, topk = topk_indices.shape
    assert topk > 0 and num_local_experts > 0
    if topk_probs is not None:
        assert topk_probs.shape == topk_indices.shape

    multihot_indices = torch.empty(
        (num_tokens, num_local_experts),
        dtype=torch.bool,
        device=topk_indices.device,
    )
    probs_dtype = topk_probs.dtype if topk_probs is not None else torch.float32
    multihot_probs = torch.empty(
        (num_tokens, num_local_experts),
        dtype=probs_dtype,
        device=topk_indices.device,
    )
    position_map = torch.empty(
        (num_tokens, num_local_experts),
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if num_tokens == 0:
        return (
            multihot_indices,
            multihot_probs if topk_probs is not None else None,
            position_map,
        )

    kernel_probs = (
        topk_probs
        if topk_probs is not None
        else torch.empty(
            topk_indices.shape,
            dtype=probs_dtype,
            device=topk_indices.device,
        )
    )
    topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
    experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(num_local_experts)))
    _indices_to_multihot_kernel[(num_tokens,)](
        topk_indices,
        kernel_probs,
        multihot_indices,
        multihot_probs,
        position_map,
        num_local_experts,
        experts_next_power_of_2,
        topk,
        topk_next_power_of_2,
        BLOCK_SIZE=32,
        num_warps=1,
    )
    return (
        multihot_indices,
        multihot_probs if topk_probs is not None else None,
        position_map,
    )


def fused_moe_indices_converter_backward_impl(
    grad_multihot_probs: torch.Tensor,
    position_map: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Convert multihot probability gradients back to their top-k slots."""
    num_tokens, num_local_experts = grad_multihot_probs.shape
    assert topk > 0 and num_local_experts > 0
    grad_topk_probs = torch.empty(
        (num_tokens, topk),
        dtype=grad_multihot_probs.dtype,
        device=grad_multihot_probs.device,
    )
    if num_tokens == 0:
        return grad_topk_probs

    topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
    experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(num_local_experts)))
    _multihot_to_indices_kernel[(num_tokens,)](
        grad_multihot_probs.contiguous(),
        position_map,
        grad_topk_probs,
        num_local_experts,
        experts_next_power_of_2,
        topk,
        topk_next_power_of_2,
        BLOCK_SIZE=32,
        num_warps=1,
    )
    return grad_topk_probs
