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
from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    attach_fp8_amax_partials,
)

__all__ = ["rmsnorm", "rmsnorm_residual"]


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6, zero_centered: bool = False):
        assert x.is_cuda and gamma.is_cuda, "rmsnorm: x and gamma must be CUDA tensors"
        orig_shape = x.shape
        H = gamma.shape[0]
        assert orig_shape[-1] == H, (
            f"rmsnorm: last dim of x ({orig_shape[-1]}) must equal gamma.shape[0] ({H})"
        )

        y, x2, rstd, BLOCK_H, ROWS, num_warps, num_stages, _ = rmsnorm_fwd_impl(x, gamma, eps, zero_centered)

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
        assert x.is_cuda and residual.is_cuda and gamma.is_cuda, (
            "rmsnorm_residual: x, residual and gamma must be CUDA tensors"
        )
        assert x.shape == residual.shape, (
            f"rmsnorm_residual: shape mismatch {tuple(x.shape)} vs {tuple(residual.shape)}"
        )
        orig_shape = x.shape
        H = gamma.shape[0]
        assert orig_shape[-1] == H
        # Leave an unused output's cotangent as None instead of a zero tensor: callers that
        # consume only ``y`` are the common case, and the backward would otherwise allocate,
        # fill and read back a whole tensor of zeros just to add it to dx.
        ctx.set_materialize_grads(False)

        # Reduce the output's abs-max while it is still in registers and publish it, so a
        # tensorwise fp8 cast downstream finalises those partials instead of streaming the
        # whole tensor again for the same scalar. A max is exact and order-independent and
        # the reduction rides the store's own predicate, so the scale is the same float
        # either way; a consumer that is not such a cast just ignores them.
        #
        # Only this variant folds. It opens a transformer block, so its output is the
        # projection input a quantiser reads next, and its forward already moves four
        # tensors per row, which hides the reduction: 147.2 -> 153.1 us against 34.7 us of
        # `tensorwise_amax_partial` deleted. The plain forward moves two, so the same fold
        # shows up in full (95.0 -> 111.4 us) and wants a consumer to be worth it.
        y, x_plus_r, rstd, BLOCK_H, ROWS, num_warps, num_stages, amax = rmsnorm_fwd_residual_impl(
            x, residual, gamma, eps, amax_out=True
        )

        ctx.save_for_backward(x_plus_r, gamma, rstd)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.BLOCK_H = BLOCK_H
        ctx.ROWS = ROWS
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        out = y.reshape(orig_shape)
        attach_fp8_amax_partials(out, amax)
        return out, x_plus_r.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_xpr: torch.Tensor):
        x_plus_r, gamma, rstd = ctx.saved_tensors
        if grad_y is None:
            # Nothing flows through the norm itself, so dgamma is zero and dx is whatever
            # came back through x_plus_r.
            return grad_xpr, grad_xpr, torch.zeros_like(gamma), None
        # Jacobian of add() is [I, I], so x and residual take the same gradient. Handing
        # back one tensor twice makes autograd clone it for the second consumer, a full
        # read plus write of [B, H]; the kernel emits both while the block is in registers.
        dx, dxr, dg = rmsnorm_bwd_residual_impl(
            grad_y,
            grad_xpr,
            x_plus_r,
            gamma,
            rstd,
            ctx.BLOCK_H,
            ctx.ROWS,
            ctx.num_warps,
            ctx.num_stages,
            dual_dx=True,
        )
        return dx.reshape(ctx.orig_shape), dxr.reshape(ctx.orig_shape), dg, None


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
