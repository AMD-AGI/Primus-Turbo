###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Friendly Python API for the MoE permute / unpermute HIP kernels.

These wrap the raw ``torch.ops.primus_turbo_cpp_extension.*`` ops that live
in ``csrc/kernels/permute/permute.cu`` + ``csrc/pytorch/permute/permute.cpp``.
The goal is to keep the C++ ABI stable (flat, IValue-friendly arguments) while
giving Python callers a small, opinionated surface.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "permute_preprocessing",
    "permute",
    "unpermute",
]


_OPS = torch.ops.primus_turbo_cpp_extension


def permute_preprocessing(
    routing_map: torch.Tensor,                 # bool [num_dispatched, num_local_experts]
    num_dispatched_token_tensor: torch.Tensor, # int32 scalar (device)
    max_num_dispatched_tokens: int,
    num_of_local_experts: int,
    pad_multiple: int = 0,
    num_permuted_tokens: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``(row_id_map, tokens_per_expert, overflow_flag)`` from the routing map.

    ``num_permuted_tokens < 0`` means "no cap".
    """
    return _OPS.permute_preprocessing(
        routing_map,
        num_dispatched_token_tensor,
        int(max_num_dispatched_tokens),
        int(num_of_local_experts),
        int(pad_multiple),
        int(num_permuted_tokens),
    )


def permute(
    tokens: torch.Tensor,                       # [num_dispatched, hidden]
    output_tokens: torch.Tensor,                # [num_permuted, hidden] (preallocated)
    row_id_map: torch.Tensor,                   # int32 [num_dispatched + pad, num_local_experts]
    num_dispatched_token_tensor: torch.Tensor,  # int32 scalar (device)
    *,
    pad_multiple: int = 0,
    num_of_local_experts: int,
    hidden_size: Optional[int] = None,
    scaling_factor: Optional[torch.Tensor] = None,
    output_scaling_factor: Optional[torch.Tensor] = None,
    probs: Optional[torch.Tensor] = None,
    output_probs: Optional[torch.Tensor] = None,
    scales_per_token: int = 0,
    local_rank: int = 0,
    num_ranks_per_node: int = 1,
    use_fp8: bool = False,
    num_permuted_token: Optional[int] = None,
    num_of_blocks_permute: int = 0,
) -> torch.Tensor:
    """Gather ``tokens`` into expert-grouped order, writing to ``output_tokens``.

    Returns ``output_tokens`` for chaining.
    """
    if hidden_size is None:
        hidden_size = int(tokens.shape[-1])
    if num_permuted_token is None:
        num_permuted_token = int(output_tokens.shape[0])
    with_probs = (probs is not None) and (output_probs is not None)

    _OPS.permute_launcher(
        tokens,
        output_tokens,
        scaling_factor,
        output_scaling_factor,
        probs,
        output_probs,
        row_id_map,
        num_dispatched_token_tensor,
        int(pad_multiple),
        int(num_of_local_experts),
        int(hidden_size),
        int(scales_per_token),
        int(local_rank),
        int(num_ranks_per_node),
        bool(use_fp8),
        bool(with_probs),
        int(num_permuted_token),
        int(num_of_blocks_permute),
    )
    return output_tokens


def unpermute(
    permuted_tokens: torch.Tensor,                # [num_permuted, hidden]   (bf16)
    output_tokens: torch.Tensor,                  # [num_dispatched, hidden] (bf16, preallocated)
    row_id_map: torch.Tensor,                     # int32 [num_dispatched + pad, num_local_experts]
    num_dispatched_tokens_tensor: torch.Tensor,   # int32 scalar (device)
    *,
    num_of_local_experts: int,
    hidden_size: Optional[int] = None,
    permuted_probs: Optional[torch.Tensor] = None,
    output_probs: Optional[torch.Tensor] = None,
    local_rank: int = 0,
    num_ranks_per_node: int = 1,
    num_of_blocks_unpermute: int = 0,
) -> torch.Tensor:
    """Reduce ``permuted_tokens`` back to per-source rows in ``output_tokens``."""
    if hidden_size is None:
        hidden_size = int(permuted_tokens.shape[-1])
    with_probs = (permuted_probs is not None) and (output_probs is not None)

    _OPS.unpermute_launcher(
        permuted_tokens,
        output_tokens,
        permuted_probs,
        output_probs,
        row_id_map,
        num_dispatched_tokens_tensor,
        int(num_of_local_experts),
        int(hidden_size),
        int(local_rank),
        int(num_ranks_per_node),
        bool(with_probs),
        int(num_of_blocks_unpermute),
    )
    return output_tokens
