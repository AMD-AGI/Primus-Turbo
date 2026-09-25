###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Union

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.gemm.gemm_impl import gemm_accum_impl, gemm_impl
from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, _setup_fused_grad_accum

__all__ = ["gemm"]


def _bgrad_gemm_impl_wrapper(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    default_backend: int,
    inplace_add_to_out: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the wgrad GEMM, accumulating into ``out`` when asked to.

    Returns the weight gradient for autograd, or a dummy buffer when the wgrad went
    straight into ``out``: forward already flagged the weight, so the training
    framework's own accumulation step stands down. Megatron still expects a tensor
    rather than None there, so its backward hooks stay on the main thread; the
    contents are never read. It is handed back in the weight's own dtype, since a
    mismatch would make autograd allocate and cast a full-size copy.
    """
    inputs = (a, trans_a, b, trans_b, out_dtype, trans_c)

    if not inplace_add_to_out:
        return gemm_impl(*inputs, default_backend=default_backend)

    assert out is not None, "out should not be None when inplace_add_to_out is True"
    # The wgrad keeps the caller's default backend: hipBLASLt and Triton both carry the
    # beta=1 epilogue. Backends without it report `inplace_add_to_out` as unsupported,
    # which keeps an explicitly pinned backend or auto-tune from silently landing
    # somewhere that ignores `out`.
    gemm_accum_impl(*inputs, out=out, default_backend=default_backend)

    return _get_dummy_wgrad(out.shape, out_dtype)


class GemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        fuse_bgrad_accum_pattern: Union[None, str] = None,
    ):
        assert a.dim() == 2 and b.dim() == 2, "Only 2D GEMM is supported"
        fuse_bgrad_accum, main_grad = _setup_fused_grad_accum(b, fuse_bgrad_accum_pattern)
        # FWD
        # out    = a * b
        # [M, N] = [M, K] * [K, N]
        out = gemm_impl(a, trans_a, b, trans_b, out_dtype, False, default_backend=BackendType.HIPBLASLT.value)
        # Save for bwd
        if a.requires_grad or b.requires_grad:
            ctx.save_for_backward(a, b)
            ctx.trans_a = trans_a
            ctx.trans_b = trans_b
            ctx.fuse_bgrad_accum = fuse_bgrad_accum
            ctx.main_grad = main_grad
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
        grad_b = _bgrad_gemm_impl_wrapper(
            a,
            not ctx.trans_a,
            grad_out,
            False,
            b.dtype,
            ctx.trans_b,
            default_backend=BackendType.HIPBLASLT.value,
            inplace_add_to_out=ctx.fuse_bgrad_accum,
            out=ctx.main_grad,
        )

        return grad_a, grad_b, None, None, None, None


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    out_dtype: torch.dtype | None = None,
    fuse_bgrad_accum_pattern: Union[None, str] = None,
) -> torch.Tensor:
    """General matrix multiplication (GEMM) for BF16/FP16, supporting autograd.

    Args:
        a: Input matrix A with shape (M, K), must be 2D tensor
        b: Input matrix B with shape (K, N) or (N, K), must be 2D tensor
        trans_a: Whether to transpose matrix A
        trans_b: Whether to transpose matrix B, if True B shape is (N, K)
        out_dtype: Output data type, defaults to None (auto-inferred)
        fuse_bgrad_accum_pattern: Enables fusing the weight-gradient accumulation
            into the wgrad GEMM epilogue, so backward writes ``b.main_grad``
            directly instead of returning a gradient the framework then adds.
            ``"megatron"`` is the only supported pattern; ``b`` must carry
            ``main_grad`` / ``grad_added_to_main_grad``. Defaults to None (no fusion).

    Returns:
        torch.Tensor: Output matrix with shape (M, N)
    """
    assert a.ndim == 2 and b.ndim == 2, "Only 2D tensors are supported"
    if out_dtype is None:
        out_dtype = torch.promote_types(a.dtype, b.dtype)
    return GemmFunction.apply(a, b, trans_a, trans_b, out_dtype, fuse_bgrad_accum_pattern)
