###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

__all__ = ["rmsnorm"]


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6):
        y = torch.ops.primus_turbo_cpp_extension.rmsnorm_fwd(x, gamma, eps)

        ctx.save_for_backward(x, gamma)
        ctx.eps = eps
        return y

    @staticmethod
    def backward_torch(ctx, grad_out: torch.Tensor):
        x, gamma = ctx.saved_tensors
        eps = ctx.eps

        N = x.size(-1)
        x_squared = x * x
        x_squared_sum = x_squared.sum(dim=-1, keepdim=True)
        x_norm = torch.rsqrt(x_squared_sum / N + eps)

        grad_x_norm = grad_out * gamma  # scale by g
        grad_x_part1 = grad_x_norm * x_norm  # apply normalized scaling

        grad_x_squared_sum = (-0.5 * (x_squared_sum / N + eps) ** (-1.5)) * (2 * x / N)
        grad_x_part2 = grad_x_squared_sum * (x * grad_x_norm).sum(dim=-1, keepdim=True)

        grad_x = grad_x_part1 + grad_x_part2

        # Gradient w.r.t. g
        grad_g = (grad_out * x * x_norm).sum(dim=0)

        return grad_x, grad_g, None

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, gamma = ctx.saved_tensors
        eps = ctx.eps
        grad_x, grad_g = torch.ops.primus_turbo_cpp_extension.rmsnorm_bwd(x, gamma, grad_out, eps)
        return grad_x, grad_g.sum(dim=0), None


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return RMSNormFunction.apply(x, gamma, eps)
