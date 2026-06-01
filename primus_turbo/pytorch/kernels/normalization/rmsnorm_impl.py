###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Python-side wrappers that launch the Triton RMSNorm kernels."""
from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

from primus_turbo.triton.normalization.rmsnorm_kernel import (
    rmsnorm_bwd_finalize_kernel,
    rmsnorm_bwd_kernel,
    rmsnorm_bwd_kernel_grid_stride,
    rmsnorm_bwd_kernel_multi_row,
    rmsnorm_bwd_residual_kernel,
    rmsnorm_bwd_residual_kernel_grid_stride,
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
    """Return (BLOCK_H, ROWS_PER_BLOCK, num_warps, num_stages) for fwd.

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


@functools.lru_cache(maxsize=None)
def _num_cus(device_index: int = 0) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _pick_bwd_config(H: int, B: int) -> Tuple[str, int, int, int, int]:
    """Pick (mode, BLOCK_H, GRID_OR_ROWS, num_warps, num_stages) for bwd.

    Modes: "multi" (2D-tile for small H + huge B), "single" (one row per
    program for tiny H/B), "grid" (persistent grid-stride for everything
    wider; num_warps/num_stages are unused — picked by @triton.autotune).
    """
    BLOCK_H = _next_pow2(H)
    if BLOCK_H <= 256 and B >= 4096:
        # ROWS=64 at BLOCK_H<=128 keeps the program count <= ~2 waves even at
        # B=32768, which slashes the dg_partial volume and finalize time.
        if BLOCK_H <= 128:
            return "multi", BLOCK_H, 64, 4, 3
        return "multi", BLOCK_H, 8, 4, 2
    if BLOCK_H <= 256:
        return "single", BLOCK_H, 1, 1, 1

    # Half-wave grid wins when B is small AND (H==8192 or H>=16384): each
    # program covers more rows, amortising the dgamma store and shrinking
    # dg_partial. H=12288 (BLOCK_H rounds to 16384) prefers full wave.
    half_wave = B <= 4096 and (H >= 16384 or (BLOCK_H == 8192 and H <= 8192))
    grid = min(B, _num_cus() // 2 if half_wave else _num_cus())
    return "grid", BLOCK_H, grid, 0, 0


def _finalize_dgamma(dg_partial: torch.Tensor, gamma_dtype: torch.dtype) -> torch.Tensor:
    """Reduce [n_parts, H] fp32 partials to dgamma[H] via a coalesced col-tile."""
    n_parts, H = dg_partial.shape
    dg = torch.empty(H, device=dg_partial.device, dtype=gamma_dtype)
    BLOCK_H = 64 if H >= 64 else _next_pow2(H)
    BLOCK_N = 64 if n_parts >= 64 else _next_pow2(max(n_parts, 1))
    grid = ((H + BLOCK_H - 1) // BLOCK_H,)
    rmsnorm_bwd_finalize_kernel[grid](
        dg_partial,
        dg,
        n_parts,
        H=H,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        num_warps=2,
        num_stages=1,
    )
    return dg


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
    """Backward launcher. Returns ``(dx [B, H], dgamma [H])``.

    The fwd's chosen ``(BLOCK_H, ROWS, num_warps, num_stages)`` are accepted
    for API compatibility but ignored — bwd re-picks a config tuned for its
    own arithmetic intensity (see ``_pick_bwd_config``).
    """
    H = gamma.shape[0]
    B = x2.shape[0]
    dy2 = _reshape_batch_hidden(dy, H)
    dx = torch.empty_like(x2)
    mode, BLOCK_H, GR, num_warps, num_stages = _pick_bwd_config(H, B)
    if mode == "single":
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
    elif mode == "multi":
        ROWS = GR
        num_programs = (B + ROWS - 1) // ROWS
        dg_partial = torch.empty(num_programs, H, device=x2.device, dtype=torch.float32)
        rmsnorm_bwd_kernel_multi_row[(num_programs,)](
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
    else:  # "grid": persistent grid-stride
        num_programs = GR
        dg_partial = torch.empty(num_programs, H, device=x2.device, dtype=torch.float32)
        rmsnorm_bwd_kernel_grid_stride[(num_programs,)](
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
            num_programs=num_programs,
        )
    dg = _finalize_dgamma(dg_partial, gamma.dtype)
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
    mode, BLOCK_H, GR, num_warps, num_stages = _pick_bwd_config(H, B)
    if mode == "single":
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
    elif mode == "multi":
        ROWS = GR
        num_programs = (B + ROWS - 1) // ROWS
        dg_partial = torch.empty(num_programs, H, device=x_plus_r.device, dtype=torch.float32)
        rmsnorm_bwd_residual_kernel_multi_row[(num_programs,)](
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
    else:  # "grid": persistent grid-stride
        num_programs = GR
        dg_partial = torch.empty(num_programs, H, device=x_plus_r.device, dtype=torch.float32)
        rmsnorm_bwd_residual_kernel_grid_stride[(num_programs,)](
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
            num_programs=num_programs,
        )
    dg = _finalize_dgamma(dg_partial, gamma.dtype)
    return dx, dg
