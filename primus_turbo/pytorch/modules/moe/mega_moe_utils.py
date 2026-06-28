###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helpers for ``MegaMoE`` routing (random load-balancing STE)."""

import torch

__all__ = ["apply_random_logits"]


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
