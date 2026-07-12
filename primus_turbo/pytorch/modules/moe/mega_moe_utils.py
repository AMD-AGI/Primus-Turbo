###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helpers for ``MegaMoE`` routing: random load-balancing STE +  aux loss."""

from typing import Optional

import torch

__all__ = ["apply_random_logits", "switch_load_balancing_loss_func"]


class _RandomSTE(torch.autograd.Function):
    """STE: forward returns random logits, backward passes grad through."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
        return logits.clone().normal_()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def apply_random_logits(logits: torch.Tensor) -> torch.Tensor:
    """Replace logits with random ones via the straight-through estimator (Megatron parity)."""
    return _RandomSTE.apply(logits)


def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
    fused: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Switch-Transformer load-balancing aux loss: E * Σ_i (f_i * P_i), Megatron-parity."""
    # mask out padding tokens before aggregating probs
    if padding_mask is not None:
        mask_expanded = padding_mask.unsqueeze(-1)
        probs = probs * mask_expanded

    if fused:
        raise NotImplementedError(
            "fused aux loss is not available; use the primus_turbo fused router kernel instead."
        )

    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss
