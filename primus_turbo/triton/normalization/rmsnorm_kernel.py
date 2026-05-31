###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Triton RMSNorm kernels (single-row + multi-row fwd/bwd, plus residual variant).

The kernels are stride-aware on both batch and hidden dims so callers can pass
non-contiguous views (e.g. ``hidden_states.reshape(-1, H)`` on a strided
fp16/bf16 tensor) without forcing a ``.contiguous()`` copy.

Backward formulation (standard):
    grad_x = (grad_out * gamma * rstd) - x * rstd^3 * mean(grad_out * gamma * x) / H
    grad_g = sum_over_batch(grad_out * x * rstd)

For the residual variant the bwd additionally folds the gradient flowing through
``x_plus_r`` (consumed by the next residual-add) into ``dx``. The autograd
function returns the same gradient for both ``x`` and ``residual`` since their
sum has Jacobian ``[I, I]``.

A 2-stage bwd is used. The multi-row variants reduce ``dgamma`` *inside* each
program over its ``ROWS_PER_BLOCK`` rows, so the partial buffer is
``(num_programs, H)`` instead of ``(B, H)``. This is essential at small-H,
huge-B shapes (e.g. q_norm in MoE attention) where ``(B, H)`` would otherwise
cost an unreasonable amount of workspace memory.
"""
from __future__ import annotations

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel — one row per program. Used when H is large.
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_fwd_kernel(
    X_ptr,
    G_ptr,
    Y_ptr,
    RSTD_ptr,
    stride_xb,
    stride_xh,
    stride_yb,
    stride_yh,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    x_ptrs = X_ptr + row * stride_xb + offs * stride_xh
    y_ptrs = Y_ptr + row * stride_yb + offs * stride_yh
    g_ptrs = G_ptr + offs
    mask = offs < H

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd * g).to(Y_ptr.dtype.element_ty)
    tl.store(y_ptrs, y, mask=mask)
    tl.store(RSTD_ptr + row, rstd)


# ---------------------------------------------------------------------------
# Forward kernel — N rows per program (small H, huge B). Reduces grid size,
# which is critical when the launch / scheduling cost dominates.
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_fwd_kernel_multi_row(
    X_ptr,
    G_ptr,
    Y_ptr,
    RSTD_ptr,
    stride_xb,
    stride_xh,
    stride_yb,
    stride_yh,
    B,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B

    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_ptrs = X_ptr + row_offs[:, None] * stride_xb + h_offs[None, :] * stride_xh
    y_ptrs = Y_ptr + row_offs[:, None] * stride_yb + h_offs[None, :] * stride_yh
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=1) / H
    rstd = tl.rsqrt(var + eps)
    y = (x * rstd[:, None] * g[None, :]).to(Y_ptr.dtype.element_ty)
    tl.store(y_ptrs, y, mask=full_mask)
    tl.store(RSTD_ptr + row_offs, rstd, mask=row_mask)


# ---------------------------------------------------------------------------
# Forward kernel — fused residual add. Computes
#     x_plus_r = x + residual
#     y        = rmsnorm(x_plus_r) * gamma
# in one pass and exposes ``x_plus_r`` for the next residual add. Saves
# ``x_plus_r`` (input dtype) and ``rstd`` (fp32) for backward.
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_fwd_residual_kernel(
    X_ptr,
    R_ptr,
    G_ptr,
    Y_ptr,
    XPR_ptr,
    RSTD_ptr,
    stride_xb,
    stride_xh,
    stride_rb,
    stride_rh,
    stride_yb,
    stride_yh,
    stride_xprb,
    stride_xprh,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    x_ptrs = X_ptr + row * stride_xb + offs * stride_xh
    r_ptrs = R_ptr + row * stride_rb + offs * stride_rh
    y_ptrs = Y_ptr + row * stride_yb + offs * stride_yh
    xpr_ptrs = XPR_ptr + row * stride_xprb + offs * stride_xprh
    g_ptrs = G_ptr + offs
    mask = offs < H

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptrs, mask=mask, other=0.0).to(tl.float32)
    xpr = x + r
    tl.store(xpr_ptrs, xpr.to(XPR_ptr.dtype.element_ty), mask=mask)

    var = tl.sum(xpr * xpr, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    y = (xpr * rstd * g).to(Y_ptr.dtype.element_ty)
    tl.store(y_ptrs, y, mask=mask)
    tl.store(RSTD_ptr + row, rstd)


@triton.jit
def rmsnorm_fwd_residual_kernel_multi_row(
    X_ptr,
    R_ptr,
    G_ptr,
    Y_ptr,
    XPR_ptr,
    RSTD_ptr,
    stride_xb,
    stride_xh,
    stride_rb,
    stride_rh,
    stride_yb,
    stride_yh,
    stride_xprb,
    stride_xprh,
    B,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B

    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_ptrs = X_ptr + row_offs[:, None] * stride_xb + h_offs[None, :] * stride_xh
    r_ptrs = R_ptr + row_offs[:, None] * stride_rb + h_offs[None, :] * stride_rh
    y_ptrs = Y_ptr + row_offs[:, None] * stride_yb + h_offs[None, :] * stride_yh
    xpr_ptrs = XPR_ptr + row_offs[:, None] * stride_xprb + h_offs[None, :] * stride_xprh
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)

    xpr = x + r
    tl.store(xpr_ptrs, xpr.to(XPR_ptr.dtype.element_ty), mask=full_mask)

    var = tl.sum(xpr * xpr, axis=1) / H
    rstd = tl.rsqrt(var + eps)
    y = (xpr * rstd[:, None] * g[None, :]).to(Y_ptr.dtype.element_ty)
    tl.store(y_ptrs, y, mask=full_mask)
    tl.store(RSTD_ptr + row_offs, rstd, mask=row_mask)


# ---------------------------------------------------------------------------
# Backward — stage 0: per-row dx + per-row (or per-program) partial dgamma.
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_bwd_kernel(
    DY_ptr,
    X_ptr,
    G_ptr,
    RSTD_ptr,
    DX_ptr,
    DG_PART_ptr,
    stride_xb,
    stride_xh,
    stride_dyb,
    stride_dyh,
    stride_dxb,
    stride_dxh,
    stride_dgb,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x_ptrs = X_ptr + row * stride_xb + offs * stride_xh
    dy_ptrs = DY_ptr + row * stride_dyb + offs * stride_dyh
    dx_ptrs = DX_ptr + row * stride_dxb + offs * stride_dxh
    dgp_ptrs = DG_PART_ptr + row * stride_dgb + offs
    g_ptrs = G_ptr + offs

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row).to(tl.float32)

    x_hat = x * rstd
    dxhat = dy * g
    m = tl.sum(dxhat * x_hat, axis=0) / H
    dx = (dxhat - x_hat * m) * rstd
    dgp = dy * x_hat

    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=mask)
    tl.store(dgp_ptrs, dgp, mask=mask)


@triton.jit
def rmsnorm_bwd_kernel_multi_row(
    DY_ptr,
    X_ptr,
    G_ptr,
    RSTD_ptr,
    DX_ptr,
    DG_PART_ptr,
    stride_xb,
    stride_xh,
    stride_dyb,
    stride_dyh,
    stride_dxb,
    stride_dxh,
    stride_dgp,
    B,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_ptrs = X_ptr + row_offs[:, None] * stride_xb + h_offs[None, :] * stride_xh
    dy_ptrs = DY_ptr + row_offs[:, None] * stride_dyb + h_offs[None, :] * stride_dyh
    dx_ptrs = DX_ptr + row_offs[:, None] * stride_dxb + h_offs[None, :] * stride_dxh
    dgp_ptrs = DG_PART_ptr + pid * stride_dgp + h_offs
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)

    x_hat = x * rstd[:, None]
    dxhat = dy * g[None, :]
    m = tl.sum(dxhat * x_hat, axis=1) / H
    dx = (dxhat - x_hat * m[:, None]) * rstd[:, None]

    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=full_mask)

    # Per-program dgamma reduction — mask out-of-range rows to zero so any
    # padding tail contributes nothing. Writes one fp32 [H] slab per program
    # instead of ROWS_PER_BLOCK rows.
    dgp_block = (dy * x_hat) * row_mask[:, None].to(tl.float32)
    dgp_row = tl.sum(dgp_block, axis=0)
    tl.store(dgp_ptrs, dgp_row, mask=h_mask)


# ---------------------------------------------------------------------------
# Backward — fused residual variant. Adds the gradient that flows through
# ``x_plus_r`` (consumed by the next residual add) to the standard RMSNorm dx.
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_bwd_residual_kernel(
    DY_ptr,
    DXPR_ptr,
    XPR_ptr,
    G_ptr,
    RSTD_ptr,
    DX_ptr,
    DG_PART_ptr,
    stride_xprb,
    stride_xprh,
    stride_dyb,
    stride_dyh,
    stride_dxprb,
    stride_dxprh,
    stride_dxb,
    stride_dxh,
    stride_dgb,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    xpr_ptrs = XPR_ptr + row * stride_xprb + offs * stride_xprh
    dy_ptrs = DY_ptr + row * stride_dyb + offs * stride_dyh
    dxpr_ptrs = DXPR_ptr + row * stride_dxprb + offs * stride_dxprh
    dx_ptrs = DX_ptr + row * stride_dxb + offs * stride_dxh
    dgp_ptrs = DG_PART_ptr + row * stride_dgb + offs
    g_ptrs = G_ptr + offs

    xpr = tl.load(xpr_ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
    dxpr = tl.load(dxpr_ptrs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row).to(tl.float32)

    x_hat = xpr * rstd
    dxhat = dy * g
    m = tl.sum(dxhat * x_hat, axis=0) / H
    dx_norm = (dxhat - x_hat * m) * rstd
    dx = dx_norm + dxpr
    dgp = dy * x_hat

    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=mask)
    tl.store(dgp_ptrs, dgp, mask=mask)


# ---------------------------------------------------------------------------
# Backward — grid-stride variant. Each program processes one row per loop
# step and grid-strides (row += num_programs) until it has covered its
# share of B. A per-program fp32 dgamma accumulator is kept live in
# registers across all rows and a single partial slab (one per program) is
# written at exit. This matches the HIP stage-0 pattern and drastically
# shrinks the dg_partial buffer (n_parts == grid_size, not B).
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_bwd_kernel_grid_stride(
    DY_ptr,
    X_ptr,
    G_ptr,
    RSTD_ptr,
    DX_ptr,
    DG_PART_ptr,
    stride_xb,
    stride_xh,
    stride_dyb,
    stride_dyh,
    stride_dxb,
    stride_dxh,
    stride_dgp,
    B,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    num_programs: tl.constexpr,
):
    pid = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # Load gamma once; reused across every row this program touches.
    g = tl.load(G_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)

    # Per-program dgamma accumulator stays in registers/LDS for the whole loop.
    dg_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Grid-stride over rows. Each program covers ceil(B / num_programs) rows.
    for row in range(pid, B, num_programs):
        x_ptrs = X_ptr + row * stride_xb + h_offs * stride_xh
        dy_ptrs = DY_ptr + row * stride_dyb + h_offs * stride_dyh
        dx_ptrs = DX_ptr + row * stride_dxb + h_offs * stride_dxh

        x = tl.load(x_ptrs, mask=h_mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptrs, mask=h_mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + row).to(tl.float32)

        x_hat = x * rstd
        dxhat = dy * g
        m = tl.sum(dxhat * x_hat, axis=0) / H
        dx = (dxhat - x_hat * m) * rstd
        tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=h_mask)

        dg_acc += dy * x_hat

    dgp_ptrs = DG_PART_ptr + pid * stride_dgp + h_offs
    tl.store(dgp_ptrs, dg_acc, mask=h_mask)


@triton.jit
def rmsnorm_bwd_residual_kernel_grid_stride(
    DY_ptr,
    DXPR_ptr,
    XPR_ptr,
    G_ptr,
    RSTD_ptr,
    DX_ptr,
    DG_PART_ptr,
    stride_xprb,
    stride_xprh,
    stride_dyb,
    stride_dyh,
    stride_dxprb,
    stride_dxprh,
    stride_dxb,
    stride_dxh,
    stride_dgp,
    B,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    num_programs: tl.constexpr,
):
    pid = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    g = tl.load(G_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    dg_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for row in range(pid, B, num_programs):
        xpr_ptrs = XPR_ptr + row * stride_xprb + h_offs * stride_xprh
        dy_ptrs = DY_ptr + row * stride_dyb + h_offs * stride_dyh
        dxpr_ptrs = DXPR_ptr + row * stride_dxprb + h_offs * stride_dxprh
        dx_ptrs = DX_ptr + row * stride_dxb + h_offs * stride_dxh

        xpr = tl.load(xpr_ptrs, mask=h_mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptrs, mask=h_mask, other=0.0).to(tl.float32)
        dxpr = tl.load(dxpr_ptrs, mask=h_mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + row).to(tl.float32)

        x_hat = xpr * rstd
        dxhat = dy * g
        m = tl.sum(dxhat * x_hat, axis=0) / H
        dx_norm = (dxhat - x_hat * m) * rstd
        dx = dx_norm + dxpr
        tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=h_mask)

        dg_acc += dy * x_hat

    dgp_ptrs = DG_PART_ptr + pid * stride_dgp + h_offs
    tl.store(dgp_ptrs, dg_acc, mask=h_mask)


# ---------------------------------------------------------------------------
# Backward — finalize: reduce the (n_parts, H) fp32 partial buffer down to
# the final dgamma[H] in the gamma dtype. Each program owns BLOCK_H columns
# and walks the n_parts rows. Coalesced column loads, single launch.
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_bwd_finalize_kernel(
    DGP_ptr,
    DG_ptr,
    n_parts,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tree-reduce a (n_parts, H) fp32 partial buffer to dgamma[H].

    Tiling: each program owns one BLOCK_H column tile. It walks the rows in
    chunks of BLOCK_N, loads a (BLOCK_N, BLOCK_H) tile, and uses ``tl.sum``
    along axis=0 — that gives a tree reduction inside the tile and keeps the
    error chain depth ~ log2(n_parts) instead of linear, matching PyTorch's
    ``sum(dim=0)`` fp32 accuracy.
    """
    pid = tl.program_id(0)
    h_offs = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    n_offs = tl.arange(0, BLOCK_N)
    for p in range(0, n_parts, BLOCK_N):
        rows = p + n_offs
        row_mask = rows < n_parts
        ptrs = DGP_ptr + rows[:, None] * H + h_offs[None, :]
        tile = tl.load(ptrs, mask=row_mask[:, None] & h_mask[None, :], other=0.0)
        acc += tl.sum(tile, axis=0)
    tl.store(DG_ptr + h_offs, acc.to(DG_ptr.dtype.element_ty), mask=h_mask)


@triton.jit
def rmsnorm_bwd_residual_kernel_multi_row(
    DY_ptr,
    DXPR_ptr,
    XPR_ptr,
    G_ptr,
    RSTD_ptr,
    DX_ptr,
    DG_PART_ptr,
    stride_xprb,
    stride_xprh,
    stride_dyb,
    stride_dyh,
    stride_dxprb,
    stride_dxprh,
    stride_dxb,
    stride_dxh,
    stride_dgp,
    B,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    xpr_ptrs = XPR_ptr + row_offs[:, None] * stride_xprb + h_offs[None, :] * stride_xprh
    dy_ptrs = DY_ptr + row_offs[:, None] * stride_dyb + h_offs[None, :] * stride_dyh
    dxpr_ptrs = DXPR_ptr + row_offs[:, None] * stride_dxprb + h_offs[None, :] * stride_dxprh
    dx_ptrs = DX_ptr + row_offs[:, None] * stride_dxb + h_offs[None, :] * stride_dxh
    dgp_ptrs = DG_PART_ptr + pid * stride_dgp + h_offs
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    xpr = tl.load(xpr_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    dxpr = tl.load(dxpr_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)

    x_hat = xpr * rstd[:, None]
    dxhat = dy * g[None, :]
    m = tl.sum(dxhat * x_hat, axis=1) / H
    dx_norm = (dxhat - x_hat * m[:, None]) * rstd[:, None]
    dx = dx_norm + dxpr

    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=full_mask)

    dgp_block = (dy * x_hat) * row_mask[:, None].to(tl.float32)
    dgp_row = tl.sum(dgp_block, axis=0)
    tl.store(dgp_ptrs, dgp_row, mask=h_mask)
