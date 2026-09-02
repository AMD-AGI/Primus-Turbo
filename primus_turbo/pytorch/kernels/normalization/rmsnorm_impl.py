###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Python-side wrappers that launch the Triton RMSNorm kernels."""

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import torch

from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    FP8_AMAX_MAX_PARTIALS,
)
from primus_turbo.triton.normalization.rmsnorm_kernel import (
    amax_reduce_kernel,
    dgamma_reduce_kernel,
    rmsnorm_bwd_finalize_kernel,
    rmsnorm_bwd_kernel_grid_stride,
    rmsnorm_bwd_kernel_multi_row,
    rmsnorm_bwd_residual_kernel_grid_stride,
    rmsnorm_bwd_residual_kernel_multi_row,
    rmsnorm_fwd_kernel,
    rmsnorm_fwd_kernel_multi_row,
    rmsnorm_fwd_residual_kernel,
    rmsnorm_fwd_residual_kernel_multi_row,
)

# Cache policy for the passes that stream. A norm reads each input row once and writes
# each output row once, and nothing in the launch re-reads either, so the lines they
# would otherwise keep resident only evict what the surrounding operators are still
# using. Raced inside the projection chain rather than on the kernels alone, which
# mis-ranks this class of change.
_FWD_LD_CM = ""
_FWD_ST_CM = ".cs"
_BWD_LD_CM = ".cg"
_BWD_ST_CM = ".cs"

# Programs per CU for the grid-stride backward.
_BWD_GRID_MULT = 2


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _amax_finalize(partials: torch.Tensor) -> torch.Tensor:
    """Bring per-program partials within what the cpp tensorwise cast accepts.

    Untouched when the grid already fits, so only the one-row-per-program widths pay the
    extra pass -- over a few hundred KB, not over the tensor itself.
    """
    n = partials.numel()
    if n <= FP8_AMAX_MAX_PARTIALS:
        return partials
    BLOCK = _next_pow2((n + FP8_AMAX_MAX_PARTIALS - 1) // FP8_AMAX_MAX_PARTIALS)
    grid = (n + BLOCK - 1) // BLOCK
    out = torch.empty(grid, device=partials.device, dtype=torch.float32)
    amax_reduce_kernel[(grid,)](partials, out, n, BLOCK=BLOCK, num_warps=1, num_stages=1)
    return out


def _reshape_batch_hidden(x: torch.Tensor, H: int) -> torch.Tensor:
    """Flatten to [B, H] without forcing a contiguous copy.

    Original strides are kept so the Triton kernels can read/write strided rows
    directly, avoiding implicit ``_to_copy`` kernels on the autograd hot path.
    """
    if x.shape[-1] != H:
        raise ValueError(f"last dim mismatch: expected H={H}, got shape={tuple(x.shape)}")
    return x.reshape(-1, H)


def _row_layout(x: torch.Tensor, H: int) -> Tuple[torch.Tensor, int, int, int, int]:
    """Address ``x``'s rows where they lie, so a strided view needs no flatten copy.

    Returns ``(base, B, stride_g, stride_row, row_group)`` as ``_row_off`` wants them:
    ``base`` is the tensor the kernels index from, and ``row_group`` is 0 whenever a single
    row stride reaches every row.

    ``reshape(-1, H)`` only returns a view when the leading dims are contiguous among
    themselves. The per-head norms break that -- they read ``qkv[:, :AO].view(S, B, HQ, HD)``
    straight out of the projection, whose rows come in slabs -- so the flatten silently
    materialises the whole tensor before the kernel has read a byte.
    """
    if x.shape[-1] != H:
        raise ValueError(f"last dim mismatch: expected H={H}, got shape={tuple(x.shape)}")
    dims, strides = list(x.shape[:-1]), list(x.stride()[:-1])
    if not dims:
        return x, 1, 0, 0, 0
    B = math.prod(dims)
    # Rows collapse onto one stride exactly where each dim's stride spans the dim inside it.
    breaks = [i for i in range(len(dims) - 1) if strides[i] != strides[i + 1] * dims[i + 1]]
    if not breaks:
        return x, B, 0, strides[-1], 0
    if len(breaks) == 1:
        j = breaks[0] + 1
        group = math.prod(dims[j:])
        if group > 1:
            return x, B, strides[j - 1], strides[-1], group
    x2 = _reshape_batch_hidden(x, H)
    return x2, B, 0, x2.stride(0), 0


def _pick_config(H: int, B: int) -> Tuple[int, int, int, int]:
    """Return (BLOCK_H, ROWS_PER_BLOCK, num_warps, num_stages) for fwd."""
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
    """Return (mode, BLOCK_H, GRID_OR_ROWS, num_warps, num_stages) for bwd."""
    BLOCK_H = _next_pow2(H)
    if BLOCK_H <= 256:
        # ROWS targets ~half-wave program count; cap by register budget at BLOCK_H=256.
        cap = 64 if BLOCK_H <= 128 else 8
        target = max(1, _num_cus() // 2)
        ROWS = min(cap, max(1, _next_pow2(B // target)))
        ns = 3 if BLOCK_H <= 128 else 2
        return "multi", BLOCK_H, ROWS, 4, ns

    # Half-wave grid when each program would otherwise get few rows and H is wide.
    rows_per_program = max(1, B // _num_cus())
    wide_row = H >= 16384 or (BLOCK_H == 8192 and H <= 8192)
    half_wave = rows_per_program <= 13 and wide_row
    # Otherwise several programs a CU: with one apiece nothing covers its own tail.
    full_wave = _BWD_GRID_MULT * _num_cus()
    grid = min(B, _num_cus() // 2 if half_wave else full_wave)
    return "grid", BLOCK_H, grid, 0, 0


# Above this many partials the finalize is handed a walk it does serially: it
# parallelizes over H only (ceil(H/BLOCK_H) programs) and steps n_parts inside one
# program, so a tall, narrow buffer leaves the GPU idle. The multi-row backward emits
# ceil(B / ROWS_PER_BLOCK) partials, which reaches the tall regime for q_norm/k_norm
# (B in the millions, H=64). `_narrow_dgamma_partials` folds those down first.
_FINALIZE_TRITON_MAX_PARTS = 512


def _narrow_dgamma_partials(dg_partial: torch.Tensor) -> torch.Tensor:
    """Fold a tall partial buffer down to what the finalize parallelises well over.

    The multi-row backward emits one slab per program, which for the per-head norms
    (B in the millions, H=64) is tens of thousands of them. One extra pass over a few MB
    is worth far more than handing the finalize a walk it does serially, and it keeps the
    reduction inside the launch instead of a torch reduce that also has to cast.
    """
    n_parts, H = dg_partial.shape
    rows = _next_pow2((n_parts + _FINALIZE_TRITON_MAX_PARTS - 1) // _FINALIZE_TRITON_MAX_PARTS)
    n_out = (n_parts + rows - 1) // rows
    BLOCK_H = min(256, _next_pow2(H))
    out = torch.empty(n_out, H, device=dg_partial.device, dtype=torch.float32)
    dgamma_reduce_kernel[((H + BLOCK_H - 1) // BLOCK_H, n_out)](
        dg_partial,
        out,
        n_parts,
        H=H,
        BLOCK_H=BLOCK_H,
        BLOCK_N=min(64, rows),
        ROWS_PER_PROG=rows,
        num_warps=4,
        num_stages=1,
    )
    return out


def _finalize_dgamma(dg_partial: torch.Tensor, gamma_dtype: torch.dtype) -> torch.Tensor:
    """Reduce (n_parts, H) fp32 partials to dgamma[H]."""
    if dg_partial.shape[0] > _FINALIZE_TRITON_MAX_PARTS:
        dg_partial = _narrow_dgamma_partials(dg_partial)
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
    x: torch.Tensor, gamma: torch.Tensor, eps: float, zero_centered: bool = False, amax_out: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int, Optional[torch.Tensor]]:
    """Forward launcher.

    Returns ``(y, x2, rstd, BLOCK_H, ROWS, num_warps, num_stages, amax)``. ``x2`` is the
    tensor the kernel read ``x``'s rows from -- usually ``x`` itself, addressed in place --
    and is saved for backward, which re-derives the same layout from it. ``rstd`` is the
    per-row reciprocal std needed by backward. ``amax`` is the fp32 abs-max partials of
    ``y`` when ``amax_out``, else None.
    """
    H = gamma.shape[0]
    x2, B, sxg, sxb, row_group = _row_layout(x, H)
    y = torch.empty((B, H), device=x.device, dtype=x.dtype)
    rstd = torch.empty(B, device=x.device, dtype=torch.float32)
    BLOCK_H, ROWS, num_warps, num_stages = _pick_config(H, B)
    grid = (B if ROWS == 1 else (B + ROWS - 1) // ROWS,)
    # One partial per program, every slot written, so it needs no pre-zeroing.
    amax = torch.empty(grid[0], device=x.device, dtype=torch.float32) if amax_out else None
    if ROWS == 1:
        rmsnorm_fwd_kernel[grid](
            x2,
            gamma,
            y,
            rstd,
            amax,
            sxg,
            sxb,
            x2.stride(-1),
            y.stride(0),
            y.stride(1),
            H=H,
            eps=eps,
            BLOCK_H=BLOCK_H,
            ROW_GROUP=row_group,
            AMAX=amax_out,
            ZERO_CENTERED=zero_centered,
            LD_CM=_FWD_LD_CM,
            ST_CM=_FWD_ST_CM,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        rmsnorm_fwd_kernel_multi_row[grid](
            x2,
            gamma,
            y,
            rstd,
            amax,
            sxg,
            sxb,
            x2.stride(-1),
            y.stride(0),
            y.stride(1),
            B=B,
            H=H,
            eps=eps,
            BLOCK_H=BLOCK_H,
            ROWS_PER_BLOCK=ROWS,
            ROW_GROUP=row_group,
            AMAX=amax_out,
            ZERO_CENTERED=zero_centered,
            LD_CM=_FWD_LD_CM,
            ST_CM=_FWD_ST_CM,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if amax is not None:
        amax = _amax_finalize(amax)
    return y, x2, rstd, BLOCK_H, ROWS, num_warps, num_stages, amax


def rmsnorm_bwd_impl(
    dy: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_H: int,
    ROWS: int,
    num_warps: int,
    num_stages: int,
    zero_centered: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward launcher. Returns (dx [B, H], dgamma [H]).

    ``x2`` is what the forward saved; ``_row_layout`` is idempotent on its own output, so
    re-deriving here reproduces the forward's addressing without threading it through ctx.
    """
    H = gamma.shape[0]
    x2, B, sxg, sxb, row_group = _row_layout(x2, H)
    dy2 = _reshape_batch_hidden(dy, H)
    dx = torch.empty((B, H), device=x2.device, dtype=x2.dtype)
    mode, BLOCK_H, GR, num_warps, num_stages = _pick_bwd_config(H, B)
    if mode == "multi":
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
            sxg,
            sxb,
            x2.stride(-1),
            dy2.stride(0),
            dy2.stride(1),
            dx.stride(0),
            dx.stride(1),
            dg_partial.stride(0),
            B=B,
            H=H,
            BLOCK_H=BLOCK_H,
            ROWS_PER_BLOCK=ROWS,
            ROW_GROUP=row_group,
            ZERO_CENTERED=zero_centered,
            LD_CM=_BWD_LD_CM,
            ST_CM=_BWD_ST_CM,
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
            sxg,
            sxb,
            x2.stride(-1),
            dy2.stride(0),
            dy2.stride(1),
            dx.stride(0),
            dx.stride(1),
            dg_partial.stride(0),
            B=B,
            H=H,
            BLOCK_H=BLOCK_H,
            num_programs=num_programs,
            ROW_GROUP=row_group,
            ZERO_CENTERED=zero_centered,
            LD_CM=_BWD_LD_CM,
            ST_CM=_BWD_ST_CM,
        )
    dg = _finalize_dgamma(dg_partial, gamma.dtype)
    return dx, dg


def rmsnorm_fwd_residual_impl(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, eps: float, amax_out: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int, Optional[torch.Tensor]]:
    """Fused (x + residual) -> rmsnorm forward.

    Returns ``(y, x_plus_r, rstd, BLOCK_H, ROWS, num_warps, num_stages, amax)``. Both
    ``y`` and ``x_plus_r`` are returned in [B, H] layout (caller is expected to
    reshape back to the original logical shape if needed). ``amax`` is the fp32 abs-max
    partials of ``y`` when ``amax_out``, else None.
    """
    H = gamma.shape[0]
    x2 = _reshape_batch_hidden(x, H)
    r2 = _reshape_batch_hidden(residual, H)
    B = x2.shape[0]
    y = torch.empty_like(x2)
    x_plus_r = torch.empty_like(x2)
    rstd = torch.empty(B, device=x.device, dtype=torch.float32)
    BLOCK_H, ROWS, num_warps, num_stages = _pick_config(H, B)
    grid = (B if ROWS == 1 else (B + ROWS - 1) // ROWS,)
    # One partial per program, every slot written, so it needs no pre-zeroing.
    amax = torch.empty(grid[0], device=x.device, dtype=torch.float32) if amax_out else None
    if ROWS == 1:
        rmsnorm_fwd_residual_kernel[grid](
            x2,
            r2,
            gamma,
            y,
            x_plus_r,
            rstd,
            amax,
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
            AMAX=amax_out,
            LD_CM=_FWD_LD_CM,
            ST_CM=_FWD_ST_CM,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        rmsnorm_fwd_residual_kernel_multi_row[grid](
            x2,
            r2,
            gamma,
            y,
            x_plus_r,
            rstd,
            amax,
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
            AMAX=amax_out,
            LD_CM=_FWD_LD_CM,
            ST_CM=_FWD_ST_CM,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if amax is not None:
        amax = _amax_finalize(amax)
    return y, x_plus_r, rstd, BLOCK_H, ROWS, num_warps, num_stages, amax


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
    dual_dx: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Backward launcher for the residual variant. Returns ``(dx, dx_residual, dgamma)``.

    ``x`` and ``residual`` take the same gradient, the upstream ``+`` having Jacobian
    ``[I, I]``. Returning one tensor for both makes autograd copy it before it can hand
    it to the second consumer, and at this width the copy costs more than storing the
    block twice while it is still in registers, so ``dual_dx`` writes a second output
    instead. ``dx_residual`` is None when it is off.
    """
    H = gamma.shape[0]
    B = x_plus_r.shape[0]
    dy2 = _reshape_batch_hidden(dy, H)
    # When ``x_plus_r`` is unused downstream, autograd hands us None. Rather than filling a
    # whole tensor with zeros for the kernel to read back and add, drop the term at compile
    # time and pass ``x_plus_r`` as a stand-in pointer the kernel never dereferences.
    has_dxpr = dxpr is not None
    dxpr2 = _reshape_batch_hidden(dxpr, H) if has_dxpr else x_plus_r
    dx = torch.empty_like(x_plus_r)
    # Same stand-in trick: off, the kernel never dereferences the second output pointer.
    dxr = torch.empty_like(x_plus_r) if dual_dx else dx
    mode, BLOCK_H, GR, num_warps, num_stages = _pick_bwd_config(H, B)
    if mode == "multi":
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
            dxr,
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
            HAS_DXPR=has_dxpr,
            DUAL_DX=dual_dx,
            LD_CM=_BWD_LD_CM,
            ST_CM=_BWD_ST_CM,
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
            dxr,
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
            HAS_DXPR=has_dxpr,
            DUAL_DX=dual_dx,
            LD_CM=_BWD_LD_CM,
            ST_CM=_BWD_ST_CM,
        )
    dg = _finalize_dgamma(dg_partial, gamma.dtype)
    return dx, (dxr if dual_dx else None), dg
