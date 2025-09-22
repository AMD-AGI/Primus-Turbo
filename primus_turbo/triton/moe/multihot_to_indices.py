###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import triton
import triton.language as tl


# Assign a block to a row([1,topk]), generate a local routing map([1,num_of_local_experts])
@triton.jit
def _indices_to_multihot_kernel(
    indices_ptr,
    probs_in_indices_ptr,
    multihot_indices_ptr,  # bool
    probs_in_multihot_ptr,
    position_map_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for converting indices to multihot representation.

    Input:
        indices: [num_of_tokens, topk]
        probs_in_indices: [num_of_tokens, topk]
    Output:
        multihot_indices: [num_of_tokens, num_of_local_experts]
        probs_in_multihot: [num_of_tokens, num_of_local_experts]

    Assume that topk = 2 , num_of_local_experts = 4, num_of_tokens = 2,
    then the kernel can process the following conversion:

    Input Example:
        indices = [
                [0, 1],
                [1, 2]
            ]
        probs_in_indices = [
                [0.1, 0.2],
                [0.3, 0.4]
            ]
    Output Example:
        multihot_indices = [
                [1, 1, -1, -1],
                [-1, 1, 1, -1]
            ]
        probs_in_multihot = [
                [0.1, 0.2, 0.0, 0.0],
                [0.0, 0.3, 0.4, 0.0]
            ]
    """
    # Prepare the [0, topk) row
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    # Prepare the [0, num_of_local_experts) row
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    # Load a [1, topk] row from the indices buffer
    row_idx = tl.program_id(0)
    indices_row = tl.load(indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)
    indices_row = tl.where(topk_row_mask, indices_row, -1)
    probs_row = tl.load(probs_in_indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)

    # Get the position of the each index in the indices_row, which is saved for backwards
    position_row = tl.where(indices_row != -1, topk_row, -1)
    # Mask of the valid indices
    mask = (indices_row != -1) & (indices_row < num_of_local_experts)

    row_idx_offset = row_idx * num_of_local_experts
    # Store to initialize
    tl.store(multihot_indices_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask)
    tl.store(position_map_ptr + row_idx_offset + num_exp_row, -1, mask=num_exp_row_mask)
    # Use barrier to make sure the initialization is done
    tl.debug_barrier()
    # Store the indices and probs_in_indices
    tl.store(multihot_indices_ptr + row_idx_offset + indices_row, 1, mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + indices_row, probs_row, mask)
    # Store the position of the position_row for backwards
    tl.store(position_map_ptr + row_idx_offset + indices_row, position_row, mask)


# Assign a block to a row([1,topk]), generate a probs_indices([1,topk])
@triton.jit
def _multihot_to_indices_kernel(
    probs_in_multihot_ptr,
    position_map_ptr,
    probs_indices_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for converting multihot representation to indices.

    Input:
        probs_in_multihot: [num_of_tokens, num_of_local_experts]
        position_map: [num_of_tokens, num_of_local_experts]
    Output:
        probs_indices: [num_of_tokens, topk]

    Assume that topk = 2 , num_of_local_experts = 4, num_of_tokens = 2,
    then the kernel can process the following conversion:

    Input Example:
        probs_in_multihot = [
                [0.7, 0.8, 0.0, 0.0],
                [0.0, 0.1, 0.9, 0.0]
            ]
        position_map = [
                [1, 1, -1, -1],
                [-1, 1, 1, -1]
            ]
    Output Example:
        probs_indices = [
                [0.7, 0.8],
                [0.1, 0.9]
            ]
    """
    # Prepare the [0, topk) row
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    # Prepare the [0, num_of_local_experts) row
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    # Load a [1, num_of_local_experts] row from the local routing map
    row_idx = tl.program_id(0)
    ptr_offset = row_idx * num_of_local_experts + num_exp_row
    probs_in_multihot_row = tl.load(probs_in_multihot_ptr + ptr_offset, mask=num_exp_row_mask)

    # Get the original position of the valid value in the the indices
    position_map_row = tl.load(position_map_ptr + ptr_offset, mask=num_exp_row_mask)
    position_map_row = tl.where(num_exp_row_mask, position_map_row, -1)
    mask = position_map_row != -1

    # Store to initialize
    tl.store(probs_indices_ptr + row_idx * topk + topk_row, 0, mask=topk_row_mask)
    # Use barrier to make sure the initialization is done
    tl.debug_barrier()
    # Restore the indices and probs_indices
    tl.store(probs_indices_ptr + row_idx * topk + position_map_row, probs_in_multihot_row, mask)
