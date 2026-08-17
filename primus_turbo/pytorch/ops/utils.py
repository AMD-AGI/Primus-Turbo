###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Helpers shared across the GEMM / grouped-GEMM op implementations."""

from typing import Optional

import torch

from primus_turbo.pytorch.core.low_precision import (
    Format,
    float8_e4m3,
    float8_e5m2,
)


def _get_fp8_dtype(format: Format, is_fwd_stage: bool):
    if format == Format.E4M3:
        return float8_e4m3
    elif format == Format.E5M2:
        return float8_e5m2
    elif format == Format.HYBRID:
        return float8_e4m3 if is_fwd_stage else float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")


def _ensure_contiguous_grad_out(grad_out: torch.Tensor) -> torch.Tensor:
    # Some upstream reductions can produce expanded zero-stride grad_out views.
    # Custom grouped GEMM kernels expect dense layouts.
    return grad_out if grad_out.is_contiguous() else grad_out.contiguous()


def _setup_fused_grad_accum(b, fuse_bgrad_accum_pattern: Optional[str]):
    """Resolve the weight's gradient-accumulation buffer for the fused wgrad path.

    Returns ``(enabled, main_grad)``. When enabled, the wgrad GEMM writes straight
    into ``main_grad`` and the Function must return no gradient for ``b``.

    Call this at the top of ``forward``: the flag has to be set while the weight
    object is still in hand, because the backward pass only sees the saved tensors.
    """
    if fuse_bgrad_accum_pattern is None:
        return False, None

    assert fuse_bgrad_accum_pattern in ["megatron"], (
        "Only megatron support gradient accumulation fusion currently"
    )

    assert hasattr(b, "grad_added_to_main_grad"), (
        "b.grad_added_to_main_grad must be set up before the backward pass."
    )
    assert hasattr(b, "main_grad"), "b.main_grad must be set up before the backward pass."
    assert isinstance(b.main_grad, torch.Tensor) and (b.main_grad.shape == b.shape), (
        "b.main_grad must be a tensor with the same shape as b"
    )

    # Set in forward, not backward: autograd hands backward the saved tensors, not
    # the parameter object that carries this attribute.
    b.grad_added_to_main_grad = True
    return True, b.main_grad


_dummy_wgrads = {}


def _get_dummy_wgrad(shape: list, dtype: torch.dtype, zero=False) -> torch.Tensor:
    """Returns a dummy tensor of given shape.

    Supports arbitrary rank (2D for plain Linear weights, 3D for stacked
    grouped-linear weights ``(num_gemms, out_features, in_features)``, etc.).
    Tensors are cached by ``(shape, dtype)`` so each distinct weight layout
    only allocates one persistent buffer that gets reused across steps.
    """
    global _dummy_wgrads
    key = (tuple(shape), dtype)
    if key not in _dummy_wgrads:
        _dummy_wgrads[key] = torch.empty(
            shape,
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
    if zero:
        _dummy_wgrads[key].fill_(0)
    return _dummy_wgrads[key].detach()
