###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Union

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.gemm.gemm_impl import gemm_accum_impl, gemm_impl
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_impl,
    grouped_gemm_variable_k_accum_impl,
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    group_offs_from_lens,
)
from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, _setup_fused_grad_accum

__all__ = ["grouped_gemm"]


def _bgrad_grouped_gemm_impl_wrapper(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    trans_c: bool,
    num_cu: int | None,
    default_backend: int,
    schedule: str = "static",
    inplace_add_to_out: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the variable-K wgrad GEMM, accumulating into ``out`` when asked to.

    Returns the weight gradient for autograd, or a dummy buffer when the wgrad went
    straight into ``out``: forward already flagged the weight, so the training
    framework's own accumulation step stands down. Megatron still expects a tensor
    rather than None there, so its backward hooks stay on the main thread; the
    contents are never read. It is handed back in the weight's own dtype, since a
    mismatch would make autograd allocate and cast a full-size copy.
    """
    inputs = (a, b, group_lens, group_offs)
    options = dict(
        trans_a=trans_a,
        trans_b=trans_b,
        trans_c=trans_c,
        num_cu=num_cu,
        schedule=schedule,
    )

    if not inplace_add_to_out:
        return grouped_gemm_variable_k_impl(*inputs, default_backend=default_backend, **options)

    assert out is not None, "out should not be None when inplace_add_to_out is True"
    # Name a backend that carries the beta=1 epilogue as the default, rather than
    # letting the dispatcher fall back to it and log a spurious "may hurt performance"
    # warning. Backends without the epilogue still report it as unsupported, which
    # keeps an explicitly pinned backend or auto-tune from silently landing somewhere
    # that ignores `out`.
    grouped_gemm_variable_k_accum_impl(*inputs, default_backend=BackendType.TRITON.value, out=out, **options)

    return _get_dummy_wgrad(out.shape, b.dtype)


class GroupedGemmFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B + 1,] int64
        trans_b: bool,
        num_cu: int | None,
        schedule: str = "static",
        fuse_bgrad_accum_pattern: Union[None, str] = None,
    ):
        fuse_bgrad_accum, main_grad = _setup_fused_grad_accum(b, fuse_bgrad_accum_pattern)
        if len(group_lens) == 1:
            assert b.size(0) == 1, f"Expected first dimension to be 1, got {b.size(0)}"
            b_2d = b.squeeze(0)
            out = gemm_impl(
                a, False, b_2d, trans_b, a.dtype, False, default_backend=BackendType.HIPBLASLT.value
            )
        else:
            out = grouped_gemm_impl(
                a,
                b,
                group_lens,
                group_offs,
                trans_a=False,
                trans_b=trans_b,
                num_cu=num_cu,
                default_backend=BackendType.TRITON.value,
                maybe_pre_sync=True,
                schedule=schedule,
            )
        ctx.save_for_backward(a, b, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.num_cu = num_cu
        ctx.schedule = schedule
        ctx.fuse_bgrad_accum = fuse_bgrad_accum
        ctx.main_grad = main_grad
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        a, b, group_lens, group_offs = ctx.saved_tensors
        if len(group_lens) == 1:
            assert b.size(0) == 1, f"Expected first dimension to be 1, got {b.size(0)}"
            b_2d = b.squeeze(0)
            grad_a = gemm_impl(
                grad_out,
                False,
                b_2d,
                not ctx.trans_b,
                a.dtype,
                ctx.trans_a,
                default_backend=BackendType.HIPBLASLT.value,
            )
            if ctx.fuse_bgrad_accum:
                # main_grad matches b's [1, ...] shape; the dense GEMM writes 2D, and
                # squeeze(0) is a view so the accumulation lands in the real buffer.
                gemm_accum_impl(
                    a,
                    True,
                    grad_out,
                    False,
                    b.dtype,
                    ctx.trans_b,
                    out=ctx.main_grad.squeeze(0),
                    default_backend=BackendType.TRITON.value,
                )
                grad_b = _get_dummy_wgrad(b.shape, b.dtype)
            else:
                grad_b = gemm_impl(
                    a,
                    True,
                    grad_out,
                    False,
                    b.dtype,
                    ctx.trans_b,
                    default_backend=BackendType.HIPBLASLT.value,
                ).view(b.size())
        else:
            grad_a = grouped_gemm_impl(
                grad_out,
                b,
                group_lens,
                group_offs,
                trans_a=False,
                trans_b=not ctx.trans_b,
                num_cu=ctx.num_cu,
                default_backend=BackendType.TRITON.value,
                schedule=ctx.schedule,
            )
            grad_b = _bgrad_grouped_gemm_impl_wrapper(
                a,
                grad_out,
                group_lens,
                group_offs,
                trans_a=not ctx.trans_a,
                trans_b=False,
                trans_c=ctx.trans_b,
                num_cu=ctx.num_cu,
                default_backend=BackendType.TRITON.value,
                schedule=ctx.schedule,
                inplace_add_to_out=ctx.fuse_bgrad_accum,
                out=ctx.main_grad,
            )
        return grad_a, grad_b, None, None, None, None, None, None


def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_b: bool = False,
    num_cu: int | None = None,
    schedule: str = "static",
    fuse_bgrad_accum_pattern: Union[None, str] = None,
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
            Must be None when ``schedule="work_steal"`` -- the work-stealing
            kernel was designed and tested for full-device launches, and the
            heuristic / per-XCD slot layout assume the persistent grid spans
            every XCD. Mixing the two raises ``ValueError``.
        schedule (str): Persistent-loop scheduling. One of:
            * ``"static"`` (default): static-stride persistent kernel.
            * ``"work_steal"``: work-stealing persistent kernel with a kernel-
              aware heuristic that picks per-XCD / global / hierarchical tile
              claims based on tensor metadata. Supported on the Triton and CK
              backends; HIPBLASLT advertises only ``"static"``. Internal WS
              sub-modes are not exposed at this layer; tune via the kernel-
              level entry points when needed. Requires ``num_cu=None`` (see
              ``num_cu`` above).
        fuse_bgrad_accum_pattern (str | None): Enables fusing the weight-gradient
            accumulation into the wgrad GEMM epilogue, so backward writes
            ``b.main_grad`` directly instead of returning a gradient the framework
            then adds. ``"megatron"`` is the only supported pattern; ``b`` must
            carry ``main_grad`` / ``grad_added_to_main_grad``. Defaults to None
            (no fusion).

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
    if schedule == "work_steal" and num_cu is not None:
        raise ValueError(
            f'schedule="work_steal" requires num_cu=None, got num_cu={num_cu}. '
            "Work-stealing is designed and tested for full-device launches; the "
            "heuristic and per-XCD slot layout assume the persistent grid spans "
            'every XCD. Pass num_cu=None, or use schedule="static".'
        )

    if group_offs is None:
        group_offs = group_offs_from_lens(group_lens)

    return GroupedGemmFunc.apply(
        a,
        b,
        group_lens,
        group_offs,
        trans_b,
        num_cu,
        schedule,
        fuse_bgrad_accum_pattern,
    )
