###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Python-side wrappers that launch the Triton RMSNorm kernels."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from primus_turbo.triton.normalization.rmsnorm_kernel import (
    rmsnorm_bwd_kernel,
    rmsnorm_bwd_kernel_multi_row,
    rmsnorm_bwd_residual_kernel,
    rmsnorm_bwd_residual_kernel_multi_row,
    rmsnorm_fwd_kernel,
    rmsnorm_fwd_kernel_multi_row,
    rmsnorm_fwd_residual_kernel,
    rmsnorm_fwd_residual_kernel_multi_row,
)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _reshape_batch_hidden(x: torch.Tensor, H: int) -> torch.Tensor:
    """Flatten to [B, H] without forcing a contiguous copy.

    Original strides are kept so the Triton kernels can read/write strided rows
    directly, avoiding implicit ``_to_copy`` kernels on the autograd hot path.
    """
    if x.shape[-1] != H:
        raise ValueError(f"last dim mismatch: expected H={H}, got shape={tuple(x.shape)}")
    return x.reshape(-1, H)


def _pick_config(H: int, B: int) -> Tuple[int, int, int, int]:
    """Return (BLOCK_H, ROWS_PER_BLOCK, num_warps, num_stages).

    Multi-row mode (ROWS_PER_BLOCK > 1) wins when H is small AND B is huge,
    because the grid size of one program per row becomes the bottleneck.
    """
    BLOCK_H = _next_pow2(H)
    if BLOCK_H <= 256 and B >= 4096:
        ROWS = 16 if BLOCK_H <= 128 else 8
        return BLOCK_H, ROWS, 4, 2
    if BLOCK_H <= 256:
        return BLOCK_H, 1, 1, 1
    if BLOCK_H <= 1024:
        return BLOCK_H, 1, 4, 2
    if BLOCK_H <= 4096:
        return BLOCK_H, 1, 8, 2
    return BLOCK_H, 1, 16, 2


def rmsnorm_fwd_impl(
    x: torch.Tensor, gamma: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    """Forward launcher.

    Returns ``(y, x2, rstd, BLOCK_H, ROWS, num_warps, num_stages)``. ``x2`` is a
    [B, H] view of ``x`` (no copy) saved for backward. ``rstd`` is the per-row
    reciprocal std needed by backward.
    """
    H = gamma.shape[0]
    x2 = _reshape_batch_hidden(x, H)
    B = x2.shape[0]
    y = torch.empty_like(x2)
    rstd = torch.empty(B, device=x.device, dtype=torch.float32)
    BLOCK_H, ROWS, num_warps, num_stages = _pick_config(H, B)
    if ROWS == 1:
        rmsnorm_fwd_kernel[(B,)](
            x2,
            gamma,
            y,
            rstd,
            x2.stride(0),
            x2.stride(1),
            y.stride(0),
            y.stride(1),
            H=H,
            eps=eps,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        grid = ((B + ROWS - 1) // ROWS,)
        rmsnorm_fwd_kernel_multi_row[grid](
            x2,
            gamma,
            y,
            rstd,
            x2.stride(0),
            x2.stride(1),
            y.stride(0),
            y.stride(1),
            B=B,
            H=H,
            eps=eps,
            BLOCK_H=BLOCK_H,
            ROWS_PER_BLOCK=ROWS,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return y, x2, rstd, BLOCK_H, ROWS, num_warps, num_stages


def rmsnorm_bwd_impl(
    dy: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_H: int,
    ROWS: int,
    num_warps: int,
    num_stages: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward launcher. Returns ``(dx [B, H], dgamma [H])``."""
    H = gamma.shape[0]
    B = x2.shape[0]
    dy2 = _reshape_batch_hidden(dy, H)
    dx = torch.empty_like(x2)
    if ROWS == 1:
        dg_partial = torch.empty(B, H, device=x2.device, dtype=torch.float32)
        rmsnorm_bwd_kernel[(B,)](
            dy2,
            x2,
            gamma,
            rstd,
            dx,
            dg_partial,
            x2.stride(0),
            x2.stride(1),
            dy2.stride(0),
            dy2.stride(1),
            dx.stride(0),
            dx.stride(1),
            dg_partial.stride(0),
            H=H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        num_programs = (B + ROWS - 1) // ROWS
        dg_partial = torch.empty(num_programs, H, device=x2.device, dtype=torch.float32)
        grid = (num_programs,)
        rmsnorm_bwd_kernel_multi_row[grid](
            dy2,
            x2,
            gamma,
            rstd,
            dx,
            dg_partial,
            x2.stride(0),
            x2.stride(1),
            dy2.stride(0),
            dy2.stride(1),
            dx.stride(0),
            dx.stride(1),
            dg_partial.stride(0),
            B=B,
            H=H,
            BLOCK_H=BLOCK_H,
            ROWS_PER_BLOCK=ROWS,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    dg = dg_partial.sum(dim=0).to(gamma.dtype)
    return dx, dg


def rmsnorm_fwd_residual_impl(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    """Fused (x + residual) -> rmsnorm forward.

    Returns ``(y, x_plus_r, rstd, BLOCK_H, ROWS, num_warps, num_stages)``. Both
    ``y`` and ``x_plus_r`` are returned in [B, H] layout (caller is expected to
    reshape back to the original logical shape if needed).
    """
    H = gamma.shape[0]
    x2 = _reshape_batch_hidden(x, H)
    r2 = _reshape_batch_hidden(residual, H)
    B = x2.shape[0]
    y = torch.empty_like(x2)
    x_plus_r = torch.empty_like(x2)
    rstd = torch.empty(B, device=x.device, dtype=torch.float32)
    BLOCK_H, ROWS, num_warps, num_stages = _pick_config(H, B)
    if ROWS == 1:
        rmsnorm_fwd_residual_kernel[(B,)](
            x2,
            r2,
            gamma,
            y,
            x_plus_r,
            rstd,
            x2.stride(0),
            x2.stride(1),
            r2.stride(0),
            r2.stride(1),
            y.stride(0),
            y.stride(1),
            x_plus_r.stride(0),
            x_plus_r.stride(1),
            H=H,
            eps=eps,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        grid = ((B + ROWS - 1) // ROWS,)
        rmsnorm_fwd_residual_kernel_multi_row[grid](
            x2,
            r2,
            gamma,
            y,
            x_plus_r,
            rstd,
            x2.stride(0),
            x2.stride(1),
            r2.stride(0),
            r2.stride(1),
            y.stride(0),
            y.stride(1),
            x_plus_r.stride(0),
            x_plus_r.stride(1),
            B=B,
            H=H,
            eps=eps,
            BLOCK_H=BLOCK_H,
            ROWS_PER_BLOCK=ROWS,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return y, x_plus_r, rstd, BLOCK_H, ROWS, num_warps, num_stages


def rmsnorm_bwd_residual_impl(
    dy: torch.Tensor,
    dxpr: Optional[torch.Tensor],
    x_plus_r: torch.Tensor,
    gamma: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_H: int,
    ROWS: int,
    num_warps: int,
    num_stages: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward launcher for the residual variant. Returns ``(dx [B, H], dgamma [H])``.

    Caller is expected to return the same ``dx`` for both ``x`` and ``residual``
    inputs because the upstream ``+`` has Jacobian ``[I, I]``.
    """
    H = gamma.shape[0]
    B = x_plus_r.shape[0]
    dy2 = _reshape_batch_hidden(dy, H)
    if dxpr is None:
        # When ``x_plus_r`` is unused downstream, autograd may hand us None.
        # Substitute zeros so the kernel sees a valid pointer.
        dxpr2 = torch.zeros_like(x_plus_r)
    else:
        dxpr2 = _reshape_batch_hidden(dxpr, H)
    dx = torch.empty_like(x_plus_r)
    if ROWS == 1:
        dg_partial = torch.empty(B, H, device=x_plus_r.device, dtype=torch.float32)
        rmsnorm_bwd_residual_kernel[(B,)](
            dy2,
            dxpr2,
            x_plus_r,
            gamma,
            rstd,
            dx,
            dg_partial,
            x_plus_r.stride(0),
            x_plus_r.stride(1),
            dy2.stride(0),
            dy2.stride(1),
            dxpr2.stride(0),
            dxpr2.stride(1),
            dx.stride(0),
            dx.stride(1),
            dg_partial.stride(0),
            H=H,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        num_programs = (B + ROWS - 1) // ROWS
        dg_partial = torch.empty(num_programs, H, device=x_plus_r.device, dtype=torch.float32)
        grid = (num_programs,)
        rmsnorm_bwd_residual_kernel_multi_row[grid](
            dy2,
            dxpr2,
            x_plus_r,
            gamma,
            rstd,
            dx,
            dg_partial,
            x_plus_r.stride(0),
            x_plus_r.stride(1),
            dy2.stride(0),
            dy2.stride(1),
            dxpr2.stride(0),
            dxpr2.stride(1),
            dx.stride(0),
            dx.stride(1),
            dg_partial.stride(0),
            B=B,
            H=H,
            BLOCK_H=BLOCK_H,
            ROWS_PER_BLOCK=ROWS,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    dg = dg_partial.sum(dim=0).to(gamma.dtype)
    return dx, dg
