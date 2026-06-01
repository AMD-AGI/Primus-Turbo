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


# MI300/MI325X have 304 CUs. One wave per CU saturates the device for this
# grid-stride kernel; pushing to 2 waves spreads dg accumulators thinner and
# inflates the finalize input.
_NUM_CUS = 304


def _pick_bwd_config(H: int, B: int) -> Tuple[str, int, int, int, int]:
    """Return (mode, BLOCK_H, GRID_OR_ROWS, num_warps, num_stages) for bwd.

    Modes:
    - ``"multi"``: multi-row 2D-tile kernel. GRID_OR_ROWS is ROWS_PER_BLOCK.
      Good for small H + huge B (q-norm in MoE attention) where launch cost
      and 2D-tile coalescing both pay off.
    - ``"grid"``: persistent grid-stride kernel. GRID_OR_ROWS is the number
      of programs. Each program loops over rows of stride num_programs while
      keeping the dgamma accumulator live in registers across the loop. This
      matches the HIP stage-0 approach and minimises the (n_parts, H)
      dg_partial buffer (n_parts == grid_size, not B).
    - ``"single"``: one-program-per-row fallback for tiny H/B.
    """
    BLOCK_H = _next_pow2(H)

    # Small-H + huge-B: multi-row is great here (tile 2D in one shot).
    if BLOCK_H <= 256 and B >= 4096:
        ROWS = 16 if BLOCK_H <= 128 else 8
        return "multi", BLOCK_H, ROWS, 4, 2
    if BLOCK_H <= 256:
        return "single", BLOCK_H, 1, 1, 1

    # Grid-stride for everything wider. Tuned empirically on MI325X:
    # - grid=304 (= NUM_CUS, 1 wave) is the default — saturates the device.
    # - grid=152 (= NUM_CUS/2) wins for H=8192 with small B (4096): each
    #   program gets more rows to chew so the dgamma write amortises and the
    #   dg_partial cache footprint shrinks.
    if H >= 16384 and B <= 4096:
        # Native huge H + small B: half-wave grid (152) wins on MI325X because
        # each program covers ~27 rows so the dgamma store amortises better
        # and dg_partial halves to (152, H), making the finalize cheaper.
        # Sweep on N=4096,C=16384 showed grid=152/nw=8/ns=1 at ~120µs vs
        # ~126µs for full-wave (grid=304). Gated on the *actual* H (not the
        # pow2-rounded BLOCK_H) because H=12288 prefers full wave.
        grid = min(B, _NUM_CUS // 2)
        num_warps = 8
        num_stages = 1
    elif BLOCK_H >= 16384 and B <= 4096:
        # H in (8192, 16384) and small B (e.g. H=12288). Full wave wins here.
        grid = min(B, _NUM_CUS)
        num_warps = 8
        num_stages = 2
    elif BLOCK_H >= 8192 and B <= 4096:
        # H=8192, small B: half-wave gives each program more rows to chew so
        # the dgamma write amortises better.
        grid = min(B, _NUM_CUS // 2)
        num_warps = 8
        num_stages = 1
    elif BLOCK_H >= 8192:
        # Wide H, larger B: full wave, drop ns to relieve register pressure.
        grid = min(B, _NUM_CUS)
        num_warps = 4
        num_stages = 2
    else:
        # H in (256, 4096]: full wave, nw=8 amortises BLOCK_H per program.
        grid = min(B, _NUM_CUS)
        num_warps = 8 if BLOCK_H >= 2048 else 4
        num_stages = 2

    return "grid", BLOCK_H, grid, num_warps, num_stages


def _finalize_dgamma(dg_partial: torch.Tensor, gamma_dtype: torch.dtype) -> torch.Tensor:
    """Reduce a [n_parts, H] fp32 partial buffer to dgamma[H] in ``gamma_dtype``.

    Uses a column-major coalesced Triton reduction (one program per BLOCK_H
    tile of H, walks n_parts rows in BLOCK_N-sized chunks with tl.sum tree
    reduction). Tuned on MI325X: a (64,64) tile with nw=2 dominates the
    previous (256,16) tile across every shape we ship — the smaller column
    tile lets more programs run concurrently (more CUs occupied) while the
    larger BLOCK_N=64 row chunk amortises the launch + dgamma write per
    program. Also beats ``dg_partial.sum(dim=0).to(...)`` at every (n_parts,
    H) we care about.
    """
    n_parts, H = dg_partial.shape
    dg = torch.empty(H, device=dg_partial.device, dtype=gamma_dtype)
    # (BLOCK_H=64, BLOCK_N=64, nw=2) is the universal winner on MI300/325X
    # across every (n_parts, H) we sweep. Clamp BLOCK_H to next_pow2(H) so
    # tiny H doesn't over-pad, and clamp BLOCK_N to a power of two >= 1 so
    # tl.sum's tree reduction stays legal when n_parts is small.
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
            num_warps=num_warps,
            num_stages=num_stages,
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
            num_warps=num_warps,
            num_stages=num_stages,
        )
    dg = _finalize_dgamma(dg_partial, gamma.dtype)
    return dx, dg
