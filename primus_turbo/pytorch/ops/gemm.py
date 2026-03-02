###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.gemm.gemm_impl import gemm_impl

__all__ = ["gemm"]


class GemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
    ):
        assert a.dim() == 2 and b.dim() == 2, "Only 2D GEMM is supported"
        # FWD
        # out    = a * b
        # [M, N] = [M, K] * [K, N]
        out = gemm_impl(a, trans_a, b, trans_b, out_dtype, False, default_backend=BackendType.HIPBLASLT.value)
        # Save for bwd
        if a.requires_grad or b.requires_grad:
            ctx.save_for_backward(a, b)
            ctx.trans_a = trans_a
            ctx.trans_b = trans_b
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b = ctx.saved_tensors

        # AGrad
        # grad_a = grad_out * b^T
        grad_a = gemm_impl(
            grad_out,
            False,
            b,
            not ctx.trans_b,
            a.dtype,
            ctx.trans_a,
            default_backend=BackendType.HIPBLASLT.value,
        )

        # BGrad
        # grad_b = a^T * grad_out
        grad_b = gemm_impl(
            a,
            not ctx.trans_a,
            grad_out,
            False,
            b.dtype,
            ctx.trans_b,
            default_backend=BackendType.HIPBLASLT.value,
        )

        return grad_a, grad_b, None, None, None


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2, "Only 2D tensors are supported"
    if out_dtype is None:
        out_dtype = torch.result_type(a, b)
    return GemmFunction.apply(a, b, trans_a, trans_b, out_dtype)
