###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch


def pytorch_permute_mask_map(tokens, routing_map):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
    """
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[1]

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool().T.contiguous()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices


def pytorch_unpermute_mask_map(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The indices used to sort the tokens.
        restore_shape (torch.Size): The shape of the unpermuted tensor.
        probs (torch.Tensor, optional): The unpermuted probs tensor,
        routing_map (torch.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].

    Returns:
        torch.Tensor: The tokens restored to their original order.
    """
    _, hidden = restore_shape

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(restore_shape, device=permuted_tokens.device, dtype=permuted_tokens.dtype)
    # Scatter add the permuted_input back to the original positions
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens
