###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus_turbo.pytorch.kernels.gemm.gemm_csrc_impl import gemm_impl
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_csrc_impl import (
    grouped_gemm_csrc_impl,
    grouped_gemm_hipblaslt_csrc_impl,
    grouped_gemm_variable_k_csrc_impl,
    grouped_gemm_variable_k_hipblaslt_csrc_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
)

__all__ = ["grouped_gemm", "grouped_gemm_hipblaslt"]


class GroupedGemmFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        num_cu: int | None,
    ):
        if len(group_lens) == 1:
            assert b.size(0) == 1, f"Expected first dimension to be 1, got {b.size(0)}"
            b_2d = b.squeeze(0)
            out = gemm_impl(a, False, b_2d, trans_b, a.dtype, False)
        else:
            out = grouped_gemm_csrc_impl(
                a,
                b,
                group_lens,
                group_offs,
                trans_a=False,
                trans_b=trans_b,
                num_cu=num_cu,
            )
        ctx.save_for_backward(a, b, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b, group_lens, group_offs = ctx.saved_tensors
        if len(group_lens) == 1:
            assert b.size(0) == 1, f"Expected first dimension to be 1, got {b.size(0)}"
            b_2d = b.squeeze(0)
            grad_a = gemm_impl(grad_out, False, b_2d, not ctx.trans_b, a.dtype, ctx.trans_a)
            grad_b = gemm_impl(a, True, grad_out, False, b.dtype, ctx.trans_b).view(b.size())
        else:
            grad_a = grouped_gemm_csrc_impl(
                grad_out,
                b,
                group_lens,
                group_offs,
                trans_a=False,
                trans_b=not ctx.trans_b,
                num_cu=ctx.num_cu,
            )

            lhs, rhs = (grad_out, a) if ctx.trans_b else (a, grad_out)
            grad_b = grouped_gemm_variable_k_csrc_impl(
                lhs,
                rhs,
                group_lens,
                group_offs,
                trans_a=True,
                trans_b=False,
                num_cu=ctx.num_cu,
            )
        return grad_a, grad_b, None, None, None, None


def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_b: bool = False,
    num_cu: int | None = None,
) -> torch.Tensor:
    """
    Grouped GEMM.

    Args:
        a (torch.Tensor): Shape [sum(group_lens), K], DType float16/bfloat16.
        b (torch.Tensor): Shape [G, K, N] (or [G, N, K] if trans_b=True), DType float16/bfloat16.
        group_lens (torch.Tensor): Rows per expert of shape [G], int64. sum(group_lens) == a.size(0).
        group_offs (torch.Tensor | None): Exclusive prefix-sum of group_lens, shape [G+1].
                                          If None, it will be computed internally.
        trans_b (bool): If True, treat each b[g] as transposed.
        num_cu (int | None): Limit the number of CUs to use. None = default.

    Returns:
        torch.Tensor: Output of shape [sum(group_lens), N], same dtype/device as `a`.

    Example:
        >>> G, K, N = 3, 128, 64
        >>> group_lens = torch.tensor([32, 16, 48], dtype=torch.long, device="cuda")
        >>> a = torch.randn(group_lens.sum().item(), K, device="cuda", dtype=torch.bfloat16)
        >>> b = torch.randn(G, K, N, device="cuda", dtype=torch.bfloat16)  # or [G, N, K] with trans_b=True
        >>> out = grouped_gemm(a, b, group_lens)  # [96, 64]
        >>> out.shape
        torch.Size([96, 64])
    """
    if group_offs is None:
        group_offs = grouped_gemm_compute_offs(group_lens)

    return GroupedGemmFunc.apply(a, b, group_lens, group_offs, trans_b, num_cu)


class GroupedGemmHipblasltFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        num_cu: int | None,
    ):
        out = grouped_gemm_hipblaslt_csrc_impl(
            a,
            b,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            num_cu=num_cu,
        )
        ctx.save_for_backward(a, b, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b, group_lens, group_offs = ctx.saved_tensors
        grad_a = grouped_gemm_hipblaslt_csrc_impl(
            grad_out,
            b,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            num_cu=ctx.num_cu,
        )

        lhs, rhs = (grad_out, a) if ctx.trans_b else (a, grad_out)
        grad_b = grouped_gemm_variable_k_hipblaslt_csrc_impl(
            lhs,
            rhs,
            group_lens,
            group_offs,
            trans_a=True,
            trans_b=False,
            num_cu=ctx.num_cu,
        )
        return grad_a, grad_b, None, None, None, None


def grouped_gemm_hipblaslt(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_b: bool = False,
    num_cu: int | None = None,
) -> torch.Tensor:
    """
    Grouped GEMM using hipBLASLt backend.

    Args:
        a (torch.Tensor): Shape [sum(group_lens), K], DType float16/bfloat16.
        b (torch.Tensor): Shape [G, K, N] (or [G, N, K] if trans_b=True), DType float16/bfloat16.
        group_lens (torch.Tensor): Rows per expert of shape [G], int64. sum(group_lens) == a.size(0).
        group_offs (torch.Tensor | None): Exclusive prefix-sum of group_lens, shape [G+1].
                                          If None, it will be computed internally.
        trans_b (bool): If True, treat each b[g] as transposed.
        num_cu (int | None): Limit the number of CUs to use. None = default.

    Returns:
        torch.Tensor: Output of shape [sum(group_lens), N], same dtype/device as `a`.

    Example:
        >>> G, K, N = 3, 128, 64
        >>> group_lens = torch.tensor([32, 16, 48], dtype=torch.long, device="cuda")
        >>> a = torch.randn(group_lens.sum().item(), K, device="cuda", dtype=torch.bfloat16)
        >>> b = torch.randn(G, K, N, device="cuda", dtype=torch.bfloat16)  # or [G, N, K] with trans_b=True
        >>> out = grouped_gemm_hipblaslt(a, b, group_lens)  # [96, 64]
        >>> out.shape
        torch.Size([96, 64])
    """
    if group_offs is None:
        group_offs = grouped_gemm_compute_offs(group_lens)

    return GroupedGemmHipblasltFunc.apply(a, b, group_lens, group_offs, trans_b, num_cu)
