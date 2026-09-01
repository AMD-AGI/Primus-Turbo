###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton RMSNorm kernels (single-row + multi-row fwd/bwd, plus residual variant).

The kernels are stride-aware on both batch and hidden dims so callers can pass
non-contiguous views (e.g. ``hidden_states.reshape(-1, H)`` on a strided
fp16/bf16 tensor) without forcing a ``.contiguous()`` copy. ``ROW_GROUP`` widens
that to layouts a flatten cannot express at all -- see ``_row_off``.

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

The forwards can also leave the abs-max of ``y`` behind as one non-negative fp32
partial per program, reduced in registers under the store's own predicate. A
tensorwise fp8 quantiser downstream finalises those instead of re-reading ``y``;
a max is exact and order-independent, so the scale is unchanged.
"""

from __future__ import annotations

import triton
import triton.language as tl

# Autotune candidates for the grid-stride bwd kernels.
_GRID_STRIDE_BWD_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=1),
    triton.Config({}, num_warps=8, num_stages=2),
]


@triton.jit
def _row_off(row, stride_g, stride_r, ROW_GROUP: tl.constexpr):
    """Element offset of ``row``, allowing rows to come in groups.

    ``ROW_GROUP == 0`` is the plain case and lowers to exactly ``row * stride_r``.
    Otherwise ``ROW_GROUP`` consecutive rows sit ``stride_r`` apart inside a slab and
    the slabs are ``stride_g`` apart: the layout of e.g. ``qkv[:, :AO].view(S, B, HQ, HD)``,
    which is a legal view of the projection but has no single row stride, so
    ``reshape(-1, HD)`` would materialise a full contiguous copy of it.
    """
    if ROW_GROUP == 0:
        return row * stride_r
    return (row // ROW_GROUP) * stride_g + (row % ROW_GROUP) * stride_r


@triton.jit
def _amax_fold(acc, Y_ptr, AMAX_ptr, slot, AMAX: tl.constexpr):
    """Leave the abs-max of this program's output at ``AMAX_ptr[slot]``, one fp32 each.

    Takes the fp32 ``acc`` that is about to be stored rather than the narrowed value, so the
    fold adds a reduction and nothing else -- no second copy of the block. Rounding to the
    store dtype is monotone over magnitudes and commutes with abs, so rounding the max of
    ``acc`` is the same float as the max of the values that land in memory, which is what a
    streaming pass over the output would have found. Lanes outside ``H`` hold 0 here, having
    been loaded with ``other=0.0``, and a max over magnitudes ignores those.

    A plain store, not an atomic into a shared slot: these grids run one program per row, so
    a norm dispatches tens of thousands of them, and an atomic per program costs 30-45 ns
    each at that width (measured 148 -> 1629 us on the residual forward). The cross-program
    max is left to `amax_reduce_kernel`, which is the shape the cpp finalise wants anyway.
    """
    if AMAX:
        a = tl.max(tl.abs(acc))
        tl.store(AMAX_ptr + slot, a.to(Y_ptr.dtype.element_ty).to(tl.float32))


@triton.jit
def amax_reduce_kernel(SRC_ptr, DST_ptr, n, BLOCK: tl.constexpr):
    """Fold ``n`` per-program abs-max partials down to ``ceil(n / BLOCK)`` of them.

    The cpp tensorwise cast finalises a bounded number of partials and a forward leaves one
    per program, so this brings the count under that bound. One contiguous chunk per program:
    a single coalesced pass over a few hundred KB, against the whole tensor the streaming
    amax would have re-read.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(SRC_ptr + offs, mask=offs < n, other=0.0)
    tl.store(DST_ptr + pid, tl.max(v))


# ---------------------------------------------------------------------------
# Forward — one row per program.
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_fwd_kernel(
    X_ptr,
    G_ptr,
    Y_ptr,
    RSTD_ptr,
    AMAX_ptr,
    stride_xg,
    stride_xb,
    stride_xh,
    stride_yb,
    stride_yh,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
    ROW_GROUP: tl.constexpr = 0,
    AMAX: tl.constexpr = False,
    ZERO_CENTERED: tl.constexpr = False,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    x_ptrs = X_ptr + _row_off(row, stride_xg, stride_xb, ROW_GROUP) + offs * stride_xh
    y_ptrs = Y_ptr + row * stride_yb + offs * stride_yh
    g_ptrs = G_ptr + offs
    mask = offs < H

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    if ZERO_CENTERED:
        # zero-centered gamma: effective gain is (1 + g), computed in fp32.
        g = g + 1.0
    acc = x * rstd * g
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=mask)
    tl.store(RSTD_ptr + row, rstd)
    _amax_fold(acc, Y_ptr, AMAX_ptr, row, AMAX)


# ---------------------------------------------------------------------------
# Forward — N rows per program.
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_fwd_kernel_multi_row(
    X_ptr,
    G_ptr,
    Y_ptr,
    RSTD_ptr,
    AMAX_ptr,
    stride_xg,
    stride_xb,
    stride_xh,
    stride_yb,
    stride_yh,
    B,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    ROW_GROUP: tl.constexpr = 0,
    AMAX: tl.constexpr = False,
    ZERO_CENTERED: tl.constexpr = False,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B

    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_row = _row_off(row_offs, stride_xg, stride_xb, ROW_GROUP)
    x_ptrs = X_ptr + x_row[:, None] + h_offs[None, :] * stride_xh
    y_ptrs = Y_ptr + row_offs[:, None] * stride_yb + h_offs[None, :] * stride_yh
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    if ZERO_CENTERED:
        # zero-centered gamma: effective gain is (1 + g), computed in fp32.
        g = g + 1.0

    var = tl.sum(x * x, axis=1) / H
    rstd = tl.rsqrt(var + eps)
    acc = x * rstd[:, None] * g[None, :]
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=full_mask)
    tl.store(RSTD_ptr + row_offs, rstd, mask=row_mask)
    _amax_fold(acc, Y_ptr, AMAX_ptr, pid, AMAX)


# ---------------------------------------------------------------------------
# Forward — fused residual add.
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_fwd_residual_kernel(
    X_ptr,
    R_ptr,
    G_ptr,
    Y_ptr,
    XPR_ptr,
    RSTD_ptr,
    AMAX_ptr,
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
    AMAX: tl.constexpr = False,
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
    acc = xpr * rstd * g
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=mask)
    tl.store(RSTD_ptr + row, rstd)
    _amax_fold(acc, Y_ptr, AMAX_ptr, row, AMAX)


# ---------------------------------------------------------------------------
# Forward — fused residual add, N rows per program.
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_fwd_residual_kernel_multi_row(
    X_ptr,
    R_ptr,
    G_ptr,
    Y_ptr,
    XPR_ptr,
    RSTD_ptr,
    AMAX_ptr,
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
    AMAX: tl.constexpr = False,
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
    acc = xpr * rstd[:, None] * g[None, :]
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=full_mask)
    tl.store(RSTD_ptr + row_offs, rstd, mask=row_mask)
    _amax_fold(acc, Y_ptr, AMAX_ptr, pid, AMAX)


# ---------------------------------------------------------------------------
# Backward — 2D tile over (ROWS_PER_BLOCK, BLOCK_H). Writes one partial
# dgamma slab per program.
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_bwd_kernel_multi_row(
    DY_ptr,
    X_ptr,
    G_ptr,
    RSTD_ptr,
    DX_ptr,
    DG_PART_ptr,
    stride_xg,
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
    ROW_GROUP: tl.constexpr = 0,
    ZERO_CENTERED: tl.constexpr = False,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_row = _row_off(row_offs, stride_xg, stride_xb, ROW_GROUP)
    x_ptrs = X_ptr + x_row[:, None] + h_offs[None, :] * stride_xh
    dy_ptrs = DY_ptr + row_offs[:, None] * stride_dyb + h_offs[None, :] * stride_dyh
    dx_ptrs = DX_ptr + row_offs[:, None] * stride_dxb + h_offs[None, :] * stride_dxh
    dgp_ptrs = DG_PART_ptr + pid * stride_dgp + h_offs
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    if ZERO_CENTERED:
        # zero-centered gamma: dx flows through the effective gain (1 + g).
        # dgamma is unchanged: d(1 + g)/dg = 1, so it still accumulates dy * x_hat.
        g = g + 1.0
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
# Backward — persistent grid-stride. dgamma accumulator stays in registers
# across the row loop; n_parts == num_programs (not B).
# ---------------------------------------------------------------------------
@triton.autotune(configs=_GRID_STRIDE_BWD_CONFIGS, key=["BLOCK_H", "B", "num_programs"])
@triton.jit
def rmsnorm_bwd_kernel_grid_stride(
    DY_ptr,
    X_ptr,
    G_ptr,
    RSTD_ptr,
    DX_ptr,
    DG_PART_ptr,
    stride_xg,
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
    ROW_GROUP: tl.constexpr = 0,
    ZERO_CENTERED: tl.constexpr = False,
):
    pid = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    g = tl.load(G_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    if ZERO_CENTERED:
        # zero-centered gamma: dx flows through the effective gain (1 + g).
        # dgamma is unchanged: d(1 + g)/dg = 1, so it still accumulates dy * x_hat.
        g = g + 1.0
    dg_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for row in range(pid, B, num_programs):
        x_ptrs = X_ptr + _row_off(row, stride_xg, stride_xb, ROW_GROUP) + h_offs * stride_xh
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


# ---------------------------------------------------------------------------
# Backward — persistent grid-stride, fused residual variant.
# ---------------------------------------------------------------------------
@triton.autotune(configs=_GRID_STRIDE_BWD_CONFIGS, key=["BLOCK_H", "B", "num_programs"])
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
    HAS_DXPR: tl.constexpr = True,
):
    pid = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    g = tl.load(G_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    dg_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for row in range(pid, B, num_programs):
        xpr_ptrs = XPR_ptr + row * stride_xprb + h_offs * stride_xprh
        dy_ptrs = DY_ptr + row * stride_dyb + h_offs * stride_dyh
        dx_ptrs = DX_ptr + row * stride_dxb + h_offs * stride_dxh

        xpr = tl.load(xpr_ptrs, mask=h_mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptrs, mask=h_mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + row).to(tl.float32)

        x_hat = xpr * rstd
        dxhat = dy * g
        m = tl.sum(dxhat * x_hat, axis=0) / H
        dx = (dxhat - x_hat * m) * rstd
        if HAS_DXPR:
            dxpr_ptrs = DXPR_ptr + row * stride_dxprb + h_offs * stride_dxprh
            dx += tl.load(dxpr_ptrs, mask=h_mask, other=0.0).to(tl.float32)
        tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=h_mask)

        dg_acc += dy * x_hat

    dgp_ptrs = DG_PART_ptr + pid * stride_dgp + h_offs
    tl.store(dgp_ptrs, dg_acc, mask=h_mask)


# ---------------------------------------------------------------------------
# Backward finalize — reduces (n_parts, H) fp32 partials to dgamma[H].
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


# ---------------------------------------------------------------------------
# Backward — 2D tile, fused residual variant.
# ---------------------------------------------------------------------------
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
    HAS_DXPR: tl.constexpr = True,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    xpr_ptrs = XPR_ptr + row_offs[:, None] * stride_xprb + h_offs[None, :] * stride_xprh
    dy_ptrs = DY_ptr + row_offs[:, None] * stride_dyb + h_offs[None, :] * stride_dyh
    dx_ptrs = DX_ptr + row_offs[:, None] * stride_dxb + h_offs[None, :] * stride_dxh
    dgp_ptrs = DG_PART_ptr + pid * stride_dgp + h_offs
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    xpr = tl.load(xpr_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)

    x_hat = xpr * rstd[:, None]
    dxhat = dy * g[None, :]
    m = tl.sum(dxhat * x_hat, axis=1) / H
    dx = (dxhat - x_hat * m[:, None]) * rstd[:, None]
    if HAS_DXPR:
        dxpr_ptrs = DXPR_ptr + row_offs[:, None] * stride_dxprb + h_offs[None, :] * stride_dxprh
        dx += tl.load(dxpr_ptrs, mask=full_mask, other=0.0).to(tl.float32)

    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=full_mask)

    dgp_block = (dy * x_hat) * row_mask[:, None].to(tl.float32)
    dgp_row = tl.sum(dgp_block, axis=0)
    tl.store(dgp_ptrs, dgp_row, mask=h_mask)
