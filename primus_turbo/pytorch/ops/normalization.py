###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Triton-backed RMSNorm ops (standard + fused residual variant).

Public API:
    - ``rmsnorm(x, gamma, eps=1e-6, zero_centered=False) -> y``
    - ``rmsnorm_residual(x, residual, gamma, eps=1e-6) -> (y, x_plus_r)``
"""
from __future__ import annotations

from typing import Tuple

import torch

from primus_turbo.pytorch.kernels.normalization.rmsnorm_impl import (
    rmsnorm_bwd_impl,
    rmsnorm_bwd_residual_impl,
    rmsnorm_fwd_impl,
    rmsnorm_fwd_residual_impl,
)

__all__ = ["rmsnorm", "rmsnorm_residual"]


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6, zero_centered: bool = False):
        assert x.is_cuda and gamma.is_cuda, "rmsnorm: x and gamma must be CUDA tensors"
        orig_shape = x.shape
        H = gamma.shape[0]
        assert (
            orig_shape[-1] == H
        ), f"rmsnorm: last dim of x ({orig_shape[-1]}) must equal gamma.shape[0] ({H})"

        y, x2, rstd, BLOCK_H, ROWS, num_warps, num_stages = rmsnorm_fwd_impl(x, gamma, eps, zero_centered)

        ctx.save_for_backward(x2, gamma, rstd)
        ctx.eps = eps
        ctx.zero_centered = zero_centered
        ctx.orig_shape = orig_shape
        ctx.BLOCK_H = BLOCK_H
        ctx.ROWS = ROWS
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x2, gamma, rstd = ctx.saved_tensors
        dx, dg = rmsnorm_bwd_impl(
            grad_out,
            x2,
            gamma,
            rstd,
            ctx.BLOCK_H,
            ctx.ROWS,
            ctx.num_warps,
            ctx.num_stages,
            ctx.zero_centered,
        )
        return dx.reshape(ctx.orig_shape), dg, None, None


class _RMSNormResidualFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6):
        assert (
            x.is_cuda and residual.is_cuda and gamma.is_cuda
        ), "rmsnorm_residual: x, residual and gamma must be CUDA tensors"
        assert (
            x.shape == residual.shape
        ), f"rmsnorm_residual: shape mismatch {tuple(x.shape)} vs {tuple(residual.shape)}"
        orig_shape = x.shape
        H = gamma.shape[0]
        assert orig_shape[-1] == H

        y, x_plus_r, rstd, BLOCK_H, ROWS, num_warps, num_stages = rmsnorm_fwd_residual_impl(
            x, residual, gamma, eps
        )

        ctx.save_for_backward(x_plus_r, gamma, rstd)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.BLOCK_H = BLOCK_H
        ctx.ROWS = ROWS
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return y.reshape(orig_shape), x_plus_r.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_xpr: torch.Tensor):
        x_plus_r, gamma, rstd = ctx.saved_tensors
        dx, dg = rmsnorm_bwd_residual_impl(
            grad_y,
            grad_xpr,
            x_plus_r,
            gamma,
            rstd,
            ctx.BLOCK_H,
            ctx.ROWS,
            ctx.num_warps,
            ctx.num_stages,
        )
        dx_out = dx.reshape(ctx.orig_shape)
        # Jacobian of add() is [I, I] -> both x and residual get the same grad.
        return dx_out, dx_out, dg, None


def rmsnorm(
    x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6, zero_centered: bool = False
) -> torch.Tensor:
    """RMSNorm.

    Args:
        x: input tensor; normalization is over the last dim.
        gamma: learnable gain of shape ``[x.shape[-1]]``.
        eps: variance epsilon.
        zero_centered: if True, the effective gain is ``(1 + gamma)`` (computed in
            fp32). Initialize ``gamma`` to zeros in this mode.
    """
    return _RMSNormFunction.apply(x, gamma, eps, zero_centered)


def rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _RMSNormResidualFunction.apply(x, residual, gamma, eps)
