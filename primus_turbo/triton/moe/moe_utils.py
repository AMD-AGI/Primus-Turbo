###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl

from primus_turbo.triton.utils.argsort import argsort


###############################################################################
# Fused Router Kernel
###############################################################################


@triton.jit
def fused_scaling_group_sum_routing_kernel(
    input_logit_ptr,  # [s, e]
    output_scores_ptr,  # [s, e]
    output_topk_idx_ptr,  # [s, k]
    output_raw_topk_logits_ptr,  # [s, k]
    output_probs_ptr,  # [s, e]
    output_routing_map_ptr,  # [s, e]
    s: tl.constexpr,  # seq len
    e: tl.constexpr,  # how many experts
    g: tl.constexpr,  # how many groups
    k: tl.constexpr,  # topk
    selected_groups: tl.constexpr,
    E_ALIGNED: tl.constexpr,
    INNER_GROUP_K_ALIGNED: tl.constexpr,  # align of (k // selected_groups)
    num_stages: tl.constexpr,
    score_function: tl.constexpr,  # 0 sigmoid 1 softmax
    scaling_factor: float = 1.0,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    # offset and mask
    col_offsets = tl.arange(0, E_ALIGNED)
    col_mask = col_offsets < e

    # padding for groups topk expert-6 groups-2 will padded to [0, 1, 2, P, 3, 4, 5, P]
    padded_e: tl.constexpr = (E_ALIGNED - e) // g
    padded_group_row_size: tl.constexpr = E_ALIGNED // g
    col_padded_offsets = col_offsets - ((col_offsets // padded_group_row_size) * padded_e)
    col_padded_mask = (col_offsets % padded_group_row_size) < (E_ALIGNED // g - padded_e)

    k_mask = col_offsets < k
    sort_mask = col_offsets < selected_groups * (E_ALIGNED // g)

    ones = tl.full((E_ALIGNED,), 1, tl.int32)

    inner_group_gather_idx = (
        tl.arange(0, INNER_GROUP_K_ALIGNED)
        .reshape(1, INNER_GROUP_K_ALIGNED)
        .broadcast_to(g, INNER_GROUP_K_ALIGNED)
    )
    inner_group_mask = inner_group_gather_idx < (k // selected_groups)

    for row_idx in tl.range(row_start, s, row_step, num_stages=num_stages):
        # load row
        input_logit_row_ptr = input_logit_ptr + row_idx * e + col_offsets

        # cal score for aux loss
        if score_function == 0:
            input_logit_row = tl.load(input_logit_row_ptr, mask=col_mask, other=-float("inf"))
            row_logit = tl.sigmoid(input_logit_row.to(tl.float32))
            row_sum = tl.sum(row_logit, dtype=tl.float32)
            row_scores = (row_logit / (row_sum + 1e-20)).to(input_logit_row.dtype)
        else:
            input_logit_row = tl.load(input_logit_row_ptr, mask=col_mask, other=-float("inf"))
            row_logit = tl.softmax(input_logit_row.to(tl.float32))
            row_scores = row_logit.to(input_logit_row.dtype)

        row_output_scores_ptr = output_scores_ptr + row_idx * e + col_offsets
        tl.store(row_output_scores_ptr, row_scores, mask=col_mask)

        if padded_e > 0:
            input_logit_row_ptr = input_logit_ptr + row_idx * e + col_padded_offsets
            # reload input due to the padding
            if score_function == 0:
                input_logit_row = tl.load(input_logit_row_ptr, mask=col_padded_mask, other=-float("inf"))
                row_logit = tl.sigmoid(input_logit_row.to(tl.float32))
            else:
                input_logit_row = tl.load(input_logit_row_ptr, mask=col_padded_mask, other=-float("inf"))
                row_logit = tl.softmax(input_logit_row.to(tl.float32))

        # sort inner groups
        input_logit_groups = tl.reshape(row_logit, (g, E_ALIGNED // g))  # [g, e // g]
        inner_groups_idx = tl.arange(0, E_ALIGNED).reshape(g, E_ALIGNED // g)
        sorted_groups_logits, sorted_inner_groups_idx = argsort(input_logit_groups, inner_groups_idx, 1, True)

        # gather inner groups top_(k // selected_groups)
        sorted_groups_logits = sorted_groups_logits.gather(inner_group_gather_idx, axis=1)
        sorted_inner_groups_idx = sorted_inner_groups_idx.gather(inner_group_gather_idx, axis=1)

        groups_topk_sum = tl.sum((sorted_groups_logits * inner_group_mask), axis=1)
        groups_idx = tl.arange(0, g)
        _, sorted_groups_idx = argsort(groups_topk_sum, groups_idx, 0, True)

        # gather topk
        sorted_groups_idx_for_gather = tl.broadcast_to(sorted_groups_idx.reshape(g, 1), (g, E_ALIGNED // g))
        sorted_raw_topk_logits = tl.gather(input_logit_groups, sorted_groups_idx_for_gather, axis=0).reshape(
            E_ALIGNED
        )
        sorted_topk_idxs = tl.gather(inner_groups_idx, sorted_groups_idx_for_gather, axis=0).reshape(
            E_ALIGNED
        )

        minus_ones = tl.full(sorted_raw_topk_logits.shape, -1.0, dtype=sorted_raw_topk_logits.dtype)
        sorted_raw_topk_logits = tl.where(sort_mask, sorted_raw_topk_logits, minus_ones)
        sorted_raw_topk_logits, sorted_topk_idxs = argsort(sorted_raw_topk_logits, sorted_topk_idxs, 0, True)

        sorted_topk_idxs = sorted_topk_idxs - ((sorted_topk_idxs // padded_group_row_size) * padded_e)

        # cal scaled probs
        if score_function == 0:
            sorted_topk_logits = sorted_raw_topk_logits / (tl.sum(sorted_raw_topk_logits * k_mask) + 1e-20)
            row_output_raw_topk_logits = output_raw_topk_logits_ptr + row_idx * k + col_offsets
            tl.store(row_output_raw_topk_logits, sorted_raw_topk_logits, mask=k_mask)
        else:
            sorted_topk_logits = sorted_raw_topk_logits

        sorted_topk_logits = scaling_factor * sorted_topk_logits

        # save results
        row_output_sorted_inner_groups_idx = output_topk_idx_ptr + row_idx * k + col_offsets
        tl.store(row_output_sorted_inner_groups_idx, sorted_topk_idxs, mask=k_mask)

        # scatter to the routing map
        row_scattered_probs = output_probs_ptr + row_idx * e + sorted_topk_idxs
        row_scattered_routing_map = output_routing_map_ptr + row_idx * e + sorted_topk_idxs
        tl.store(row_scattered_probs, sorted_topk_logits, mask=k_mask)
        tl.store(row_scattered_routing_map, ones, mask=k_mask)


@triton.jit
def fused_scaling_group_sum_routing_backward_kernel(
    input_g_probs,  # [s, e]
    input_g_score,  # [s, e]
    input_logits,  # [s, e]
    input_probs,  # [s, e]
    input_topk_indices,  # [s, k]
    input_raw_topk_logits,  # [s, k]
    input_out_scores,  # [s, e]
    input_routing_map,  # [s, e]
    output_g_probs,  # [s, e]
    output_g_scores,  # [s, e]
    s: tl.constexpr,  # seq len
    e: tl.constexpr,  # how many experts
    k: tl.constexpr,  # topk
    K_ALIGNED: tl.constexpr,
    E_ALIGNED: tl.constexpr,  # cols
    num_stages: tl.constexpr,
    score_function: tl.constexpr,  # 0 sigmoid 1 softmax
    scaling_factor: float = 1.0,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    # offset and mask
    col_offsets = tl.arange(0, E_ALIGNED)
    col_mask = col_offsets < e

    k_offsets = tl.arange(0, K_ALIGNED)
    k_mask = k_offsets < k

    for row_idx in tl.range(row_start, s, row_step, num_stages=num_stages):
        # topk indices
        input_topk_indices_ptr = input_topk_indices + row_idx * k + k_offsets
        input_topk_indices_offset = tl.load(input_topk_indices_ptr, mask=k_mask, other=0)

        ## handle g_probs
        input_g_probs_ptr = input_g_probs + row_idx * e + input_topk_indices_offset
        input_topk_g_probs_row = tl.load(input_g_probs_ptr, mask=k_mask, other=float(0.0))
        input_topk_g_probs_row = scaling_factor * input_topk_g_probs_row

        input_topk_probs_ptr = input_probs + row_idx * e + input_topk_indices_offset
        input_topk_probs_row = tl.load(input_topk_probs_ptr, mask=k_mask, other=-float(0.0))

        # out scores
        input_out_score_ptr = input_out_scores + row_idx * e + col_offsets
        input_out_score_row = tl.load(input_out_score_ptr, mask=col_mask, other=float(0.0))

        if score_function == 0:  # sigmoid
            # topk raw logits
            input_raw_topk_logits_ptr = input_raw_topk_logits + row_idx * k + k_offsets
            input_raw_topk_logits_row = tl.load(input_raw_topk_logits_ptr, mask=k_mask, other=float(1.0))

            unscaled_topk_logits_row = input_topk_probs_row / scaling_factor

            sum_t = (
                (-1)
                * input_topk_g_probs_row
                * unscaled_topk_logits_row
                * unscaled_topk_logits_row
                / input_raw_topk_logits_row
            )
            sum_t = tl.sum(sum_t).broadcast_to(K_ALIGNED)

            g_probs_row = (
                input_topk_g_probs_row * unscaled_topk_logits_row / input_raw_topk_logits_row + sum_t
            )
            g_probs_row = g_probs_row * input_raw_topk_logits_row * (1 - input_raw_topk_logits_row)

            output_g_probs_ptr = output_g_probs + row_idx * e + input_topk_indices_offset
            tl.store(output_g_probs_ptr, g_probs_row, mask=k_mask)
        else:
            input_topk_score_ptr = input_out_scores + row_idx * e + input_topk_indices_offset
            input_topk_score_raw = tl.load(input_topk_score_ptr, mask=k_mask, other=float(0.0))
            sum_t = tl.sum(input_topk_g_probs_row * input_topk_score_raw).broadcast_to(E_ALIGNED)

            input_routing_map_ptr = input_routing_map + row_idx * e + col_offsets
            input_routing_map_row = tl.load(input_routing_map_ptr, mask=col_mask, other=0).to(
                input_topk_g_probs_row.dtype
            )

            input_g_probs_ptr = input_g_probs + row_idx * e + col_offsets
            input_g_probs_row = tl.load(input_g_probs_ptr, mask=col_mask, other=float(0.0))
            input_g_probs_row = scaling_factor * input_g_probs_row * input_routing_map_row

            g_probs_row = input_out_score_row * (input_g_probs_row - sum_t)

            output_g_probs_ptr = output_g_probs + row_idx * e + col_offsets
            tl.store(output_g_probs_ptr, g_probs_row, mask=col_mask)

        ## handle g_score
        input_g_score_ptr = input_g_score + row_idx * e + col_offsets
        input_g_score_row = tl.load(input_g_score_ptr, mask=col_mask, other=float(0.0))

        input_logits_ptr = input_logits + row_idx * e + col_offsets
        input_logits_row = tl.load(input_logits_ptr, mask=col_mask, other=float(0.0))

        if score_function == 0:  # sigmoid
            sigmoid_logit_row = tl.sigmoid(input_logits_row.to(tl.float32))
            sum_t = (-1) * (input_g_score_row * input_out_score_row * input_out_score_row / sigmoid_logit_row)
            sum_t = tl.sum(sum_t).broadcast_to(E_ALIGNED)

            g_score_row = input_g_score_row * input_out_score_row / sigmoid_logit_row + sum_t
            g_score_row = g_score_row * sigmoid_logit_row * (1 - sigmoid_logit_row)
        else:
            sum_t = tl.sum(input_g_score_row * input_out_score_row).broadcast_to(E_ALIGNED)
            g_score_row = (input_out_score_row) * (input_g_score_row - sum_t)

        output_g_scores_ptr = output_g_scores + row_idx * e + col_offsets
        tl.store(output_g_scores_ptr, g_score_row, mask=col_mask)


###############################################################################
# Indices <-> Multihot Conversion Kernels
###############################################################################


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


###############################################################################
# Tokens Per Expert to Mask Kernel
###############################################################################


@triton.jit
def tokens_per_expert_to_mask_kernel(
    # pointers
    tokens_per_expert_ptr,
    mask_ptr,
    # sizes
    num_expert: tl.constexpr,
    # metas
    TOKENS_PER_EXPERT_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    tokens_per_expert_off = tl.arange(0, TOKENS_PER_EXPERT_LOAD_WIDTH)
    real_num_tokens = tl.load(
        tokens_per_expert_ptr + tokens_per_expert_off, mask=(tokens_per_expert_off < num_expert)
    )
    real_num_tokens = tl.sum(real_num_tokens)

    ones = tl.full((BLOCK_SIZE,), 1, dtype=tl.int64)

    off = tl.arange(0, BLOCK_SIZE)
    mask = (pid * BLOCK_SIZE + off) < real_num_tokens

    tl.store(mask_ptr + pid * BLOCK_SIZE + off, ones, mask=mask)
