###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MX-FP8 quantization kernels (e4m3 values + e8m0 scales @ K-group=32).

Produces the scale layouts that ``tl.dot_scaled`` expects (see
``grouped_gemm_mxfp8_kernel.py``):

  Forward-path activations:   A  [M, K]       -> A_fp8  [M, K]       + A_scale  [M, K//32]
  Forward-path weight:        B  [G, K, N]    -> B_fp8  [G, K, N]    + B_scale  [G, N, K//32]  (N-first)
  Dgrad-path weight:          B  [G, K, N]    -> B_fp8  [G, K, N]    + B_scale  [G, K, N//32]  (K-first)

All three produce the fp8 values and the scale layout in a **single Triton
launch** with a 3D grid — no per-group Python loop, no transpose-and-copy.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_FP8_MAX = 448.0
MX_GROUP_SIZE = 32


class MXFP8WeightPrequant:
    """Container holding the forward + dgrad MX-FP8 quantisations of a weight.

    Produced by :func:`prequant_mxfp8_weights`; consumed by ``grouped_gemm_fp8``
    when ``b`` is a ``MXFP8WeightPrequant`` instance instead of a bf16 tensor.

    Real training loops with gradient accumulation have constant weights
    across micro-batches within one optimiser step. Pre-quantising the
    weight once per step instead of every forward saves ~0.6 ms/step on
    the gpt_oss_20B shape (both ``quant_mxfp8_weight_fwd`` and
    ``quant_mxfp8_weight_dgrad`` are lifted out of the hot path).
    """

    __slots__ = (
        "b_bf16",
        "b_fp8_fwd", "b_scale_fwd",
        "b_fp8_dgrad", "b_scale_dgrad",
    )

    def __init__(self, b_bf16, b_fp8_fwd, b_scale_fwd, b_fp8_dgrad, b_scale_dgrad):
        self.b_bf16 = b_bf16            # [G, K, N] bf16 — source (held for autograd grad flow)
        self.b_fp8_fwd = b_fp8_fwd      # [G, K, N] fp8
        self.b_scale_fwd = b_scale_fwd  # [G, N, K//32] e8m0
        self.b_fp8_dgrad = b_fp8_dgrad  # [G, K, N] fp8
        self.b_scale_dgrad = b_scale_dgrad  # [G, K, N//32] e8m0

    @property
    def shape(self):
        return self.b_bf16.shape

    @property
    def dtype(self):
        return self.b_bf16.dtype

    @property
    def device(self):
        return self.b_bf16.device


def prequant_mxfp8_weights(b: torch.Tensor) -> MXFP8WeightPrequant:
    """Pre-quantise a bf16 weight for both forward and dgrad layouts.

    Call once per optimiser step (or whenever weights change). The
    returned :class:`MXFP8WeightPrequant` can be passed in place of the
    bf16 weight to ``grouped_gemm_fp8(config=MX_BLOCKWISE, ...)``.

    The bf16 source is retained internally so autograd's ``grad_b`` flows
    back to the caller's bf16 weight as usual.
    """
    b_fp8_fwd, b_scale_fwd = quant_mxfp8_weight_fwd(b)
    b_fp8_dgrad, b_scale_dgrad = quant_mxfp8_weight_dgrad(b)
    return MXFP8WeightPrequant(b, b_fp8_fwd, b_scale_fwd, b_fp8_dgrad, b_scale_dgrad)


@triton.jit
def _mxfp8_quant_rowwise_kernel(
    X, X_fp8, X_scale,
    M, K,
    stride_xm, stride_xk,
    stride_qm, stride_qk,
    stride_sm, stride_sk,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP: tl.constexpr, FP8_MAX: tl.constexpr,
):
    """Row-wise mxfp8 quant: scales grouped along K (last axis)."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = rm < M
    mask_k = rk < K
    mask = mask_m[:, None] & mask_k[None, :]

    x = tl.load(X + rm[:, None] * stride_xm + rk[None, :] * stride_xk, mask=mask, other=0.0)
    x = x.to(tl.float32)

    x2 = tl.reshape(x, (BLOCK_M, BLOCK_K // GROUP, GROUP))
    amax = tl.max(tl.abs(x2), axis=-1, keep_dims=True)
    amax = tl.maximum(amax, 1e-8)
    e_int = tl.clamp(tl.ceil(tl.log2(amax / FP8_MAX)), -127.0, 127.0)
    scale = tl.exp2(e_int)
    e_u8 = (e_int.to(tl.int32) + 127).to(tl.uint8)
    xq = tl.clamp(x2 / scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
    xq_out = tl.reshape(xq, (BLOCK_M, BLOCK_K))
    sc_out = tl.reshape(e_u8, (BLOCK_M, BLOCK_K // GROUP))

    rks = pid_k * (BLOCK_K // GROUP) + tl.arange(0, BLOCK_K // GROUP)
    tl.store(X_fp8 + rm[:, None] * stride_qm + rk[None, :] * stride_qk, xq_out, mask=mask)
    tl.store(X_scale + rm[:, None] * stride_sm + rks[None, :] * stride_sk,
             sc_out, mask=mask_m[:, None] & (rks[None, :] < (K // GROUP)))


def quant_mxfp8_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[M, K]`` bf16 -> ``([M, K] fp8_e4m3, [M, K//32] uint8 e8m0)``.

    Scales grouped along the last (K) axis. Used for activations on the
    forward path and for ``grad_out`` on the dgrad path.
    """
    assert x.ndim == 2, f"x must be 2D, got {x.shape}"
    M, K = x.shape
    assert K % MX_GROUP_SIZE == 0, f"K={K} must be multiple of {MX_GROUP_SIZE}"
    x_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    x_scale = torch.empty((M, K // MX_GROUP_SIZE), dtype=torch.uint8, device=x.device)
    BLOCK_M, BLOCK_K = 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    _mxfp8_quant_rowwise_kernel[grid](
        x, x_fp8, x_scale, M, K,
        x.stride(0), x.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        x_scale.stride(0), x_scale.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, GROUP=MX_GROUP_SIZE, FP8_MAX=_FP8_MAX,
    )
    return x_fp8, x_scale


# ─────────────────────────────────────────────────────────────────────────────
# Weight quant (single 3D-grid kernels; no Python loop, no .T.contiguous())
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _mxfp8_quant_weight_fwd_kernel(
    W, W_fp8, W_scale,
    G, K, N,
    stride_wg, stride_wk, stride_wn,
    stride_qg, stride_qk, stride_qn,
    stride_sg, stride_sn, stride_sk,   # W_scale layout: [G, N, K//32] (N-first)
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP: tl.constexpr, FP8_MAX: tl.constexpr,
):
    """Forward weight quant: scales grouped along K, laid out N-first per group."""
    pid_g = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_n = rn < N
    mask_k = rk < K

    # Load a (BLOCK_K, BLOCK_N) tile of the weight for this group.
    off = pid_g * stride_wg + rk[:, None] * stride_wk + rn[None, :] * stride_wn
    mask_kn = mask_k[:, None] & mask_n[None, :]
    w = tl.load(W + off, mask=mask_kn, other=0.0)
    w = w.to(tl.float32)

    # Compute amax per K-group along axis 0 (K), independently for each N-col.
    # reshape (BLOCK_K, BLOCK_N) -> (BLOCK_K // GROUP, GROUP, BLOCK_N)
    w3 = tl.reshape(w, (BLOCK_K // GROUP, GROUP, BLOCK_N))
    amax = tl.max(tl.abs(w3), axis=1, keep_dims=True)
    amax = tl.maximum(amax, 1e-8)
    e_int = tl.clamp(tl.ceil(tl.log2(amax / FP8_MAX)), -127.0, 127.0)
    scale = tl.exp2(e_int)
    e_u8 = (e_int.to(tl.int32) + 127).to(tl.uint8)
    wq3 = tl.clamp(w3 / scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)

    wq = tl.reshape(wq3, (BLOCK_K, BLOCK_N))
    sc = tl.reshape(e_u8, (BLOCK_K // GROUP, BLOCK_N))  # (K-groups, N)

    # Store fp8 values.
    tl.store(W_fp8 + pid_g * stride_qg + rk[:, None] * stride_qk + rn[None, :] * stride_qn,
             wq, mask=mask_kn)

    # Store scales as [G, N, K//32]: per N-col, per K-group.
    rks = pid_k * (BLOCK_K // GROUP) + tl.arange(0, BLOCK_K // GROUP)
    mask_ks = rks < (K // GROUP)
    # sc has shape (K-groups, N); transpose indexing to write [N, K-groups].
    sc_t = tl.trans(sc, (1, 0))  # (BLOCK_N, BLOCK_K//GROUP)
    tl.store(
        W_scale
        + pid_g * stride_sg
        + rn[:, None] * stride_sn
        + rks[None, :] * stride_sk,
        sc_t,
        mask=mask_n[:, None] & mask_ks[None, :],
    )


def quant_mxfp8_weight_fwd(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[G, K, N]`` -> ``([G, K, N] fp8, [G, N, K//32] uint8 e8m0)``.

    Scales grouped along K, laid out N-first — the layout the forward kernel
    consumes. Single 3D-grid Triton launch; no per-group Python loop.
    """
    assert w.ndim == 3, f"w must be 3D, got {w.shape}"
    G, K, N = w.shape
    assert K % MX_GROUP_SIZE == 0
    w_fp8 = torch.empty((G, K, N), dtype=torch.float8_e4m3fn, device=w.device)
    w_scale = torch.empty((G, N, K // MX_GROUP_SIZE), dtype=torch.uint8, device=w.device)
    BLOCK_N, BLOCK_K = 64, 128
    grid = (G, triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _mxfp8_quant_weight_fwd_kernel[grid](
        w, w_fp8, w_scale, G, K, N,
        w.stride(0), w.stride(1), w.stride(2),
        w_fp8.stride(0), w_fp8.stride(1), w_fp8.stride(2),
        w_scale.stride(0), w_scale.stride(1), w_scale.stride(2),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP=MX_GROUP_SIZE, FP8_MAX=_FP8_MAX,
    )
    return w_fp8, w_scale


@triton.jit
def _mxfp8_quant_weight_dgrad_kernel(
    W, W_fp8, W_scale,
    G, K, N,
    stride_wg, stride_wk, stride_wn,
    stride_qg, stride_qk, stride_qn,
    stride_sg, stride_sk, stride_sn,   # W_scale layout: [G, K, N//32] (K-first)
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP: tl.constexpr, FP8_MAX: tl.constexpr,
):
    """Dgrad weight quant: scales grouped along N, laid out K-first per group."""
    pid_g = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_n = tl.program_id(2)

    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_k = rk < K
    mask_n = rn < N

    off = pid_g * stride_wg + rk[:, None] * stride_wk + rn[None, :] * stride_wn
    mask_kn = mask_k[:, None] & mask_n[None, :]
    w = tl.load(W + off, mask=mask_kn, other=0.0)
    w = w.to(tl.float32)

    # Group along N axis. reshape (BLOCK_K, BLOCK_N) -> (BLOCK_K, BLOCK_N//GROUP, GROUP)
    w3 = tl.reshape(w, (BLOCK_K, BLOCK_N // GROUP, GROUP))
    amax = tl.max(tl.abs(w3), axis=-1, keep_dims=True)
    amax = tl.maximum(amax, 1e-8)
    e_int = tl.clamp(tl.ceil(tl.log2(amax / FP8_MAX)), -127.0, 127.0)
    scale = tl.exp2(e_int)
    e_u8 = (e_int.to(tl.int32) + 127).to(tl.uint8)
    wq = tl.reshape(tl.clamp(w3 / scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv),
                    (BLOCK_K, BLOCK_N))
    sc = tl.reshape(e_u8, (BLOCK_K, BLOCK_N // GROUP))

    tl.store(W_fp8 + pid_g * stride_qg + rk[:, None] * stride_qk + rn[None, :] * stride_qn,
             wq, mask=mask_kn)

    rns = pid_n * (BLOCK_N // GROUP) + tl.arange(0, BLOCK_N // GROUP)
    mask_ns = rns < (N // GROUP)
    tl.store(
        W_scale
        + pid_g * stride_sg
        + rk[:, None] * stride_sk
        + rns[None, :] * stride_sn,
        sc,
        mask=mask_k[:, None] & mask_ns[None, :],
    )


def quant_mxfp8_weight_dgrad(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[G, K, N]`` -> ``([G, K, N] fp8, [G, K, N//32] uint8 e8m0)``.

    Scales grouped along N, laid out K-first — the layout the dgrad path
    consumes (via ``grouped_gemm_mxfp8_triton_kernel`` with ``trans_b=True``).
    Single 3D-grid Triton launch.
    """
    assert w.ndim == 3, f"w must be 3D, got {w.shape}"
    G, K, N = w.shape
    assert N % MX_GROUP_SIZE == 0
    w_fp8 = torch.empty((G, K, N), dtype=torch.float8_e4m3fn, device=w.device)
    w_scale = torch.empty((G, K, N // MX_GROUP_SIZE), dtype=torch.uint8, device=w.device)
    BLOCK_K, BLOCK_N = 64, 128
    grid = (G, triton.cdiv(K, BLOCK_K), triton.cdiv(N, BLOCK_N))
    _mxfp8_quant_weight_dgrad_kernel[grid](
        w, w_fp8, w_scale, G, K, N,
        w.stride(0), w.stride(1), w.stride(2),
        w_fp8.stride(0), w_fp8.stride(1), w_fp8.stride(2),
        w_scale.stride(0), w_scale.stride(1), w_scale.stride(2),
        BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, GROUP=MX_GROUP_SIZE, FP8_MAX=_FP8_MAX,
    )
    return w_fp8, w_scale


# ─────────────────────────────────────────────────────────────────────────────
# Column-wise quant for variable-K (wgrad) path — scales grouped along M axis
#
# wgrad computes   dB[g][k, n] = sum_{m in g} A[m, k] * grad_out[m, n]
# Reduction axis = M. tl.dot_scaled expects MX groups along the reduction axis,
# so A and grad_out each need 32-element groups taken along M (column-wise
# within each [M, L] tensor, where L = K for A or N for grad_out).
#
# Scale layout produced:  [L, M // 32] uint8 e8m0.
# Indexing the scale of column `l`, group `mg` is scale[l, mg]. That maps
# directly onto the operand-row-first / scale-group-second layout that
# ``tl.dot_scaled`` consumes for both the a-operand (LHS = A^T) and the
# b-operand (RHS = grad_out) when the reduction axis is M.
#
# Constraint: M must be a multiple of GROUP=32. For our target training shape
# (M_total = 65536 split into 32 × 2048 groups) this holds trivially and each
# per-group segment stays aligned to the scale-grid, so no scale group ever
# spans two expert groups.
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _mxfp8_quant_colwise_for_variable_k_kernel(
    X, X_fp8, X_scale,
    M, L,
    stride_xm, stride_xl,
    stride_qm, stride_ql,
    stride_sl, stride_sm,        # X_scale layout: [L, M//32]
    BLOCK_M: tl.constexpr, BLOCK_L: tl.constexpr,
    GROUP: tl.constexpr, FP8_MAX: tl.constexpr,
):
    """Column-wise mxfp8 quant: scales grouped along the M axis.

    Produces scales at layout ``[L, M//32]`` ready for ``tl.dot_scaled``
    consumption in the wgrad kernel (reduction axis = M).
    """
    pid_m = tl.program_id(0)
    pid_l = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rl = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_m = rm < M
    mask_l = rl < L
    mask = mask_m[:, None] & mask_l[None, :]

    x = tl.load(X + rm[:, None] * stride_xm + rl[None, :] * stride_xl, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # Group along M: reshape (BLOCK_M, BLOCK_L) -> (BLOCK_M // GROUP, GROUP, BLOCK_L).
    x3 = tl.reshape(x, (BLOCK_M // GROUP, GROUP, BLOCK_L))
    amax = tl.max(tl.abs(x3), axis=1, keep_dims=True)
    amax = tl.maximum(amax, 1e-8)
    e_int = tl.clamp(tl.ceil(tl.log2(amax / FP8_MAX)), -127.0, 127.0)
    scale = tl.exp2(e_int)
    e_u8 = (e_int.to(tl.int32) + 127).to(tl.uint8)
    xq3 = tl.clamp(x3 / scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)

    xq = tl.reshape(xq3, (BLOCK_M, BLOCK_L))
    sc = tl.reshape(e_u8, (BLOCK_M // GROUP, BLOCK_L))  # (M-groups, L)

    # Store fp8 values (same [M, L] layout as input).
    tl.store(X_fp8 + rm[:, None] * stride_qm + rl[None, :] * stride_ql, xq, mask=mask)

    # Store scales as [L, M//32]. sc has shape (M-groups, L); transpose to [L, M-groups].
    rms = pid_m * (BLOCK_M // GROUP) + tl.arange(0, BLOCK_M // GROUP)
    mask_ms = rms < (M // GROUP)
    sc_t = tl.trans(sc, (1, 0))  # (BLOCK_L, BLOCK_M//GROUP)
    tl.store(
        X_scale + rl[:, None] * stride_sl + rms[None, :] * stride_sm,
        sc_t,
        mask=mask_l[:, None] & mask_ms[None, :],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dual quant (rowwise + colwise scales in one HBM read)
#
# For a single 2D tensor (A or grad_out) we need both scale orientations:
#   - rowwise (scales along L)  : consumed by forward/dgrad kernels
#   - colwise (scales along M)  : consumed by wgrad variable-K kernel
#
# Doing them as two separate kernels costs 2× HBM read of the input. Fusing
# them costs only 1× read + a per-tile extra amax reduction and one more
# store. Typical saving on gpt_oss_20B shape: ~0.4 ms per fused call.
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _mxfp8_quant_dual_kernel(
    X,
    X_fp8_row, X_scale_row,
    X_fp8_col, X_scale_col,
    M, L,
    stride_xm, stride_xl,
    stride_qm_row, stride_ql_row,
    stride_sm_row, stride_sl_row,        # row_scale:  [M, L//32]
    stride_qm_col, stride_ql_col,
    stride_sl_col, stride_sm_col,        # col_scale:  [L, M//32]
    BLOCK_M: tl.constexpr, BLOCK_L: tl.constexpr,
    GROUP: tl.constexpr, FP8_MAX: tl.constexpr,
):
    """Row + column mxfp8 quant in one HBM read of ``x``."""
    pid_m = tl.program_id(0)
    pid_l = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rl = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_m = rm < M
    mask_l = rl < L
    mask = mask_m[:, None] & mask_l[None, :]

    x = tl.load(X + rm[:, None] * stride_xm + rl[None, :] * stride_xl, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # ── rowwise path: group along L, one scale per (m, L-group).
    x_row = tl.reshape(x, (BLOCK_M, BLOCK_L // GROUP, GROUP))
    amax_r = tl.maximum(tl.max(tl.abs(x_row), axis=-1, keep_dims=True), 1e-8)
    e_r = tl.clamp(tl.ceil(tl.log2(amax_r / FP8_MAX)), -127.0, 127.0)
    sc_r = tl.exp2(e_r)
    u8_r = (e_r.to(tl.int32) + 127).to(tl.uint8)
    xq_r = tl.clamp(x_row / sc_r, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
    xq_row_out = tl.reshape(xq_r, (BLOCK_M, BLOCK_L))
    sc_row_out = tl.reshape(u8_r, (BLOCK_M, BLOCK_L // GROUP))

    # ── colwise path: group along M, one scale per (M-group, l).
    x_col = tl.reshape(x, (BLOCK_M // GROUP, GROUP, BLOCK_L))
    amax_c = tl.maximum(tl.max(tl.abs(x_col), axis=1, keep_dims=True), 1e-8)
    e_c = tl.clamp(tl.ceil(tl.log2(amax_c / FP8_MAX)), -127.0, 127.0)
    sc_c = tl.exp2(e_c)
    u8_c = (e_c.to(tl.int32) + 127).to(tl.uint8)
    xq_c = tl.clamp(x_col / sc_c, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
    xq_col_out = tl.reshape(xq_c, (BLOCK_M, BLOCK_L))
    sc_col_out = tl.reshape(u8_c, (BLOCK_M // GROUP, BLOCK_L))

    # ── writes.
    tl.store(X_fp8_row + rm[:, None] * stride_qm_row + rl[None, :] * stride_ql_row,
             xq_row_out, mask=mask)
    rls = pid_l * (BLOCK_L // GROUP) + tl.arange(0, BLOCK_L // GROUP)
    mask_ls = rls < (L // GROUP)
    tl.store(
        X_scale_row + rm[:, None] * stride_sm_row + rls[None, :] * stride_sl_row,
        sc_row_out,
        mask=mask_m[:, None] & mask_ls[None, :],
    )

    tl.store(X_fp8_col + rm[:, None] * stride_qm_col + rl[None, :] * stride_ql_col,
             xq_col_out, mask=mask)
    rms = pid_m * (BLOCK_M // GROUP) + tl.arange(0, BLOCK_M // GROUP)
    mask_ms = rms < (M // GROUP)
    sc_col_t = tl.trans(sc_col_out, (1, 0))  # (BLOCK_L, BLOCK_M // GROUP)
    tl.store(
        X_scale_col + rl[:, None] * stride_sl_col + rms[None, :] * stride_sm_col,
        sc_col_t,
        mask=mask_l[:, None] & mask_ms[None, :],
    )


def quant_mxfp8_dual(x: torch.Tensor):
    """Fused rowwise + colwise mxfp8 quant of a 2D tensor.

    Returns ``(row_fp8, row_scale, col_fp8, col_scale)`` where:
      - ``row_fp8``  : ``[M, L]`` fp8_e4m3 (scales along L)
      - ``row_scale``: ``[M, L//32]`` uint8 e8m0
      - ``col_fp8``  : ``[M, L]`` fp8_e4m3 (scales along M, different values)
      - ``col_scale``: ``[L, M//32]`` uint8 e8m0

    Drop-in replacement for calling ``quant_mxfp8_rowwise(x)`` and
    ``quant_mxfp8_colwise_for_variable_k(x)`` separately; reads ``x``
    from HBM once instead of twice.

    Constraints: ``M % 32 == 0`` and ``L % 32 == 0``.
    """
    assert x.ndim == 2, f"x must be 2D, got {x.shape}"
    M, L = x.shape
    assert M % MX_GROUP_SIZE == 0, f"M={M} must be multiple of {MX_GROUP_SIZE}"
    assert L % MX_GROUP_SIZE == 0, f"L={L} must be multiple of {MX_GROUP_SIZE}"

    row_fp8 = torch.empty((M, L), dtype=torch.float8_e4m3fn, device=x.device)
    row_scale = torch.empty((M, L // MX_GROUP_SIZE), dtype=torch.uint8, device=x.device)
    col_fp8 = torch.empty((M, L), dtype=torch.float8_e4m3fn, device=x.device)
    col_scale = torch.empty((L, M // MX_GROUP_SIZE), dtype=torch.uint8, device=x.device)

    BLOCK_M, BLOCK_L = 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(L, BLOCK_L))
    _mxfp8_quant_dual_kernel[grid](
        x,
        row_fp8, row_scale,
        col_fp8, col_scale,
        M, L,
        x.stride(0), x.stride(1),
        row_fp8.stride(0), row_fp8.stride(1),
        row_scale.stride(0), row_scale.stride(1),
        col_fp8.stride(0), col_fp8.stride(1),
        col_scale.stride(0), col_scale.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_L=BLOCK_L, GROUP=MX_GROUP_SIZE, FP8_MAX=_FP8_MAX,
    )
    return row_fp8, row_scale, col_fp8, col_scale


# ─────────────────────────────────────────────────────────────────────────────
# Jagged per-group colwise quant (Option C — no padding of A required)
#
# For unbalanced MoE the padding approach wastes HBM + adds a per-group copy.
# The jagged layout stores scales per-group consecutively in one flat tensor:
#
#    x_scale shape = [L, total_scale_cols]
#    total_scale_cols = sum(ceil(M_g / 32)) across experts
#    scale_offs[g]    = sum_{h<g} ceil(M_h / 32)
#
# Per expert g, the scale columns are ``x_scale[:, scale_offs[g]:scale_offs[g+1]]``.
# The last scale within each expert may cover fewer than 32 real M elements
# (partial group); the remaining 32-M_g%32 slots quantise to fp8 zero so
# the partial group contributes exactly the right value to any downstream
# reduction over M.
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _mxfp8_quant_colwise_jagged_kernel(
    X, X_fp8, X_scale,
    group_offs_ptr, scale_offs_ptr, block_to_expert_ptr,
    L,
    stride_xm, stride_xl,
    stride_qm, stride_ql,
    stride_sl, stride_sc,           # [L, total_scale_cols]
    BLOCK_L: tl.constexpr,
    GROUP: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Flat-grid jagged quant.

    Grid = (total_scale_cols, cdiv(L, BLOCK_L)).
    Each program processes one scale-group (32 M-rows) × BLOCK_L cols.

    ``block_to_expert_ptr[pid_flat]`` tells us which expert owns this
    scale-group; ``scale_offs_ptr[expert]`` gives the expert's scale base.
    This launches exactly ``total_scale_cols`` programs — zero wasted work
    from an oversized grid on heavily unbalanced MoE.
    """
    pid_flat = tl.program_id(0)
    pid_l = tl.program_id(1)

    expert_g = tl.load(block_to_expert_ptr + pid_flat).to(tl.int32)
    sc_base = tl.load(scale_offs_ptr + expert_g)
    sg_within = pid_flat - sc_base              # 0-based within expert

    m_start = tl.load(group_offs_ptr + expert_g)
    m_end = tl.load(group_offs_ptr + expert_g + 1)
    M_g = (m_end - m_start).to(tl.int32)

    # Row range within this scale group (mask the partial last group of expert).
    rm_local = sg_within * GROUP + tl.arange(0, GROUP)
    rm = m_start + rm_local
    mask_m = rm_local < M_g

    rl = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = rl < L
    mask = mask_m[:, None] & mask_l[None, :]

    x = tl.load(X + rm[:, None] * stride_xm + rl[None, :] * stride_xl, mask=mask, other=0.0)
    x = x.to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-8)  # (BLOCK_L,)
    e_int = tl.clamp(tl.ceil(tl.log2(amax / FP8_MAX)), -127.0, 127.0)
    scale = tl.exp2(e_int)
    e_u8 = (e_int.to(tl.int32) + 127).to(tl.uint8)

    xq = tl.clamp(x / scale[None, :], -FP8_MAX, FP8_MAX).to(tl.float8e4nv)

    tl.store(X_fp8 + rm[:, None] * stride_qm + rl[None, :] * stride_ql, xq, mask=mask)
    tl.store(
        X_scale + rl * stride_sl + pid_flat * stride_sc,
        e_u8,
        mask=mask_l,
    )


def quant_mxfp8_colwise_jagged(
    x: torch.Tensor,
    group_offs: torch.Tensor,
    group_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Jagged per-group colwise MX-FP8 quant (flat-grid, no wasted launches).

    Input:
      ``x``          : ``[M_total, L]`` bf16
      ``group_offs`` : ``[G+1]`` int64 prefix sums of expert sizes
      ``group_lens`` : ``[G]`` int64 per-expert M sizes

    Returns ``(x_fp8, x_scale, scale_offs)``:
      ``x_fp8``      : ``[M_total, L]`` fp8_e4m3
      ``x_scale``    : ``[L, total_scale_cols]`` uint8 e8m0, jagged per-group
      ``scale_offs`` : ``[G+1]`` int64 scale-space prefix sums

    Works on arbitrary per-group sizes — no ``M_g % 32`` constraint. Partial
    scale groups (``M_g % 32 != 0``) are masked during quant; the trailing
    fp8 rows inside that partial group are zero-initialised.
    """
    assert x.ndim == 2, f"x must be 2D, got {x.shape}"
    M_total, L = x.shape
    G = group_lens.shape[0]
    GROUP = MX_GROUP_SIZE

    # Scale-space layout.
    sg_per_group = (group_lens + GROUP - 1) // GROUP                   # [G]
    scale_offs = torch.cat([
        torch.zeros(1, dtype=group_offs.dtype, device=group_offs.device),
        torch.cumsum(sg_per_group, dim=0),
    ])                                                                  # [G+1]
    total_sc = int(scale_offs[-1].item())

    # Flat block → expert lookup: for each scale-group index in [0, total_sc),
    # record the expert g it belongs to. G=32 experts expanded by per-expert
    # sg count — one-shot repeat_interleave (~10 µs on GPU).
    block_to_expert = torch.repeat_interleave(
        torch.arange(G, dtype=torch.int32, device=x.device), sg_per_group.to(torch.int32),
    )

    x_fp8 = torch.zeros((M_total, L), dtype=torch.float8_e4m3fn, device=x.device)
    x_scale = torch.empty((L, total_sc), dtype=torch.uint8, device=x.device)

    # BLOCK_L chosen to keep the launch-grid count close to the dense kernel
    # (balanced case) while still fitting in LDS. 128 halves the grid vs 64
    # and recovers the ~5% balanced regression without hurting unbalanced.
    BLOCK_L = 128
    grid = (total_sc, triton.cdiv(L, BLOCK_L))
    _mxfp8_quant_colwise_jagged_kernel[grid](
        x, x_fp8, x_scale,
        group_offs, scale_offs, block_to_expert,
        L,
        x.stride(0), x.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        x_scale.stride(0), x_scale.stride(1),
        BLOCK_L=BLOCK_L, GROUP=GROUP, FP8_MAX=_FP8_MAX,
    )
    return x_fp8, x_scale, scale_offs


# ─────────────────────────────────────────────────────────────────────────────
# Dual jagged quant (rowwise + colwise jagged in one HBM read)
#
# Replaces two separate kernels (rowwise for fwd/dgrad + jagged colwise for
# wgrad) with a single kernel that reads the input tensor once and writes
# both scale layouts plus both fp8 outputs. Saves one ~0.1 ms HBM read of A
# per forward pass + one ~0.1 ms HBM read of grad_out per backward pass.
#
# Grid shape mirrors the jagged colwise grid: (total_sc, cdiv(L, BLOCK_L)).
# Each tile covers 32 M-rows × BLOCK_L L-cols — exactly one M-scale-group
# for the colwise direction, and (BLOCK_L // 32) K-scale-groups along the L
# direction for the rowwise direction.
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _mxfp8_quant_dual_jagged_kernel(
    X,
    X_fp8_row, X_scale_row,
    X_fp8_col, X_scale_col,
    group_offs_ptr, scale_offs_ptr, block_to_expert_ptr,
    L,
    stride_xm, stride_xl,
    stride_qm_row, stride_ql_row,
    stride_sm_row, stride_sl_row,        # row_scale: [M, L//32]
    stride_qm_col, stride_ql_col,
    stride_sl_col, stride_sc_col,        # col_scale: [L, total_sc]
    BLOCK_L: tl.constexpr,
    GROUP: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Rowwise + jagged-colwise mxfp8 quant in one HBM read of ``x``.

    Grid: (total_sc, cdiv(L, BLOCK_L)). Each program handles one M-scale-group
    for colwise (32 rows) × BLOCK_L L-cols.
    """
    pid_flat = tl.program_id(0)
    pid_l = tl.program_id(1)

    expert_g = tl.load(block_to_expert_ptr + pid_flat).to(tl.int32)
    sc_base = tl.load(scale_offs_ptr + expert_g)
    sg_within = pid_flat - sc_base

    m_start = tl.load(group_offs_ptr + expert_g)
    m_end = tl.load(group_offs_ptr + expert_g + 1)
    M_g = (m_end - m_start).to(tl.int32)

    rm_local = sg_within * GROUP + tl.arange(0, GROUP)
    rm = m_start + rm_local
    mask_m = rm_local < M_g

    rl = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = rl < L
    mask = mask_m[:, None] & mask_l[None, :]

    # Single HBM read of the (32, BLOCK_L) tile.
    x = tl.load(X + rm[:, None] * stride_xm + rl[None, :] * stride_xl, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # ── Rowwise path: group along L (last axis), one scale per (row, L-group).
    x_row = tl.reshape(x, (GROUP, BLOCK_L // GROUP, GROUP))
    amax_r = tl.maximum(tl.max(tl.abs(x_row), axis=-1, keep_dims=True), 1e-8)
    e_r = tl.clamp(tl.ceil(tl.log2(amax_r / FP8_MAX)), -127.0, 127.0)
    sc_r = tl.exp2(e_r)
    u8_r = (e_r.to(tl.int32) + 127).to(tl.uint8)
    xq_r = tl.clamp(x_row / sc_r, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
    xq_row_out = tl.reshape(xq_r, (GROUP, BLOCK_L))
    sc_row_out = tl.reshape(u8_r, (GROUP, BLOCK_L // GROUP))

    # ── Colwise path: group along M axis (the 32 rows in this tile).
    amax_c = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-8)  # (BLOCK_L,)
    e_c = tl.clamp(tl.ceil(tl.log2(amax_c / FP8_MAX)), -127.0, 127.0)
    sc_c = tl.exp2(e_c)
    u8_c = (e_c.to(tl.int32) + 127).to(tl.uint8)
    xq_c = tl.clamp(x / sc_c[None, :], -FP8_MAX, FP8_MAX).to(tl.float8e4nv)

    # ── Stores: two fp8 tensors (same positions, different values) + two scale
    # layouts.
    tl.store(X_fp8_row + rm[:, None] * stride_qm_row + rl[None, :] * stride_ql_row,
             xq_row_out, mask=mask)
    rls = pid_l * (BLOCK_L // GROUP) + tl.arange(0, BLOCK_L // GROUP)
    mask_ls = rls < ((L + GROUP - 1) // GROUP)
    tl.store(
        X_scale_row + rm[:, None] * stride_sm_row + rls[None, :] * stride_sl_row,
        sc_row_out,
        mask=mask_m[:, None] & mask_ls[None, :],
    )

    tl.store(X_fp8_col + rm[:, None] * stride_qm_col + rl[None, :] * stride_ql_col,
             xq_c, mask=mask)
    tl.store(
        X_scale_col + rl * stride_sl_col + pid_flat * stride_sc_col,
        u8_c,
        mask=mask_l,
    )


def quant_mxfp8_dual_jagged(
    x: torch.Tensor,
    group_offs: torch.Tensor,
    group_lens: torch.Tensor,
):
    """Fused rowwise + jagged colwise MX-FP8 quant of a 2D tensor.

    Returns ``(row_fp8, row_scale, col_fp8, col_scale, scale_offs)`` —
    drop-in replacement for calling ``quant_mxfp8_rowwise(x)`` and
    ``quant_mxfp8_colwise_jagged(x, group_offs, group_lens)`` separately;
    reads ``x`` from HBM once instead of twice.

    - ``row_fp8``   : ``[M, L]`` fp8_e4m3 (scales along L)
    - ``row_scale`` : ``[M, L//32]`` uint8 e8m0
    - ``col_fp8``   : ``[M, L]`` fp8_e4m3 (scales along M; different values)
    - ``col_scale`` : ``[L, total_sc]`` uint8 e8m0 (jagged per-expert)
    - ``scale_offs``: ``[G+1]`` int64 scale-space prefix sums
    """
    assert x.ndim == 2, f"x must be 2D, got {x.shape}"
    M_total, L = x.shape
    assert L % MX_GROUP_SIZE == 0, f"L={L} must be multiple of {MX_GROUP_SIZE}"
    G = group_lens.shape[0]
    GROUP = MX_GROUP_SIZE

    sg_per_group = (group_lens + GROUP - 1) // GROUP
    scale_offs = torch.cat([
        torch.zeros(1, dtype=group_offs.dtype, device=group_offs.device),
        torch.cumsum(sg_per_group, dim=0),
    ])
    total_sc = int(scale_offs[-1].item())

    block_to_expert = torch.repeat_interleave(
        torch.arange(G, dtype=torch.int32, device=x.device),
        sg_per_group.to(torch.int32),
    )

    # Colwise fp8 needs zero-init (masked rows inside a partial scale group
    # must read back as fp8 zero); rowwise is fully written where x is valid.
    row_fp8 = torch.zeros((M_total, L), dtype=torch.float8_e4m3fn, device=x.device)
    row_scale = torch.empty((M_total, L // GROUP), dtype=torch.uint8, device=x.device)
    col_fp8 = torch.zeros((M_total, L), dtype=torch.float8_e4m3fn, device=x.device)
    col_scale = torch.empty((L, total_sc), dtype=torch.uint8, device=x.device)

    BLOCK_L = 128
    grid = (total_sc, triton.cdiv(L, BLOCK_L))
    _mxfp8_quant_dual_jagged_kernel[grid](
        x,
        row_fp8, row_scale,
        col_fp8, col_scale,
        group_offs, scale_offs, block_to_expert,
        L,
        x.stride(0), x.stride(1),
        row_fp8.stride(0), row_fp8.stride(1),
        row_scale.stride(0), row_scale.stride(1),
        col_fp8.stride(0), col_fp8.stride(1),
        col_scale.stride(0), col_scale.stride(1),
        BLOCK_L=BLOCK_L, GROUP=GROUP, FP8_MAX=_FP8_MAX,
    )
    return row_fp8, row_scale, col_fp8, col_scale, scale_offs


# ── Dual-jagged with col output pre-permuted to [G, L, M_g] ────────────────
# Attempted 2026-04-24 to eliminate the runtime fp8 permute for HIP wgrad.
# Result: SLOWER than dual_jagged + post-permute at gpt_oss_20B shape (1301 vs
# 1092 us), bit-exact correctness. Root cause: Triton's blocked layout for the
# [GROUP=32, BLOCK_L=128] tile distributes threads along the OUTER axis, so
# even with tl.trans(xq_c), stores to the permuted [G, L, M_g] layout (inner
# = M_g) are not coalesced. The existing fp8_permute_M_to_GN post-pass hits
# ~85% HBM peak and is cheaper than the layout penalty. Kept below for future
# revisits (e.g., with explicit tl.BlockedEncoding tuned for inner-M_g stores).

@triton.jit
def _mxfp8_quant_dual_jagged_permuted_kernel(
    X,
    X_fp8_row, X_scale_row,
    X_fp8_col_perm, X_scale_col_perm,
    group_offs_ptr, scale_offs_ptr, block_to_expert_ptr,
    L, M_g,
    stride_xm, stride_xl,
    stride_qm_row, stride_ql_row,
    stride_sm_row, stride_sl_row,
    # col_fp8_perm  [G, L, M_g]: stride_cg, stride_cl, stride_cm
    stride_cg_perm, stride_cl_perm, stride_cm_perm,
    # col_scale_perm [G, L, sc_g]: stride_ssg, stride_ssl, stride_sss
    stride_ssg_perm, stride_ssl_perm, stride_sss_perm,
    BLOCK_L: tl.constexpr,
    GROUP: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Dual-jagged quant where col outputs are PRE-PERMUTED to [G, L, M_g]
    contiguous (M_g uniform balanced MoE). Row outputs unchanged."""
    pid_flat = tl.program_id(0)
    pid_l = tl.program_id(1)

    expert_g = tl.load(block_to_expert_ptr + pid_flat).to(tl.int32)
    sc_base = tl.load(scale_offs_ptr + expert_g)
    sg_within = pid_flat - sc_base

    m_start = tl.load(group_offs_ptr + expert_g)
    m_end = tl.load(group_offs_ptr + expert_g + 1)
    M_g_runtime = (m_end - m_start).to(tl.int32)

    rm_local = sg_within * GROUP + tl.arange(0, GROUP)
    rm = m_start + rm_local
    mask_m = rm_local < M_g_runtime

    rl = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = rl < L
    mask = mask_m[:, None] & mask_l[None, :]

    # Single HBM read.
    x = tl.load(X + rm[:, None] * stride_xm + rl[None, :] * stride_xl, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # Rowwise path (unchanged).
    x_row = tl.reshape(x, (GROUP, BLOCK_L // GROUP, GROUP))
    amax_r = tl.maximum(tl.max(tl.abs(x_row), axis=-1, keep_dims=True), 1e-8)
    e_r = tl.clamp(tl.ceil(tl.log2(amax_r / FP8_MAX)), -127.0, 127.0)
    sc_r = tl.exp2(e_r)
    u8_r = (e_r.to(tl.int32) + 127).to(tl.uint8)
    xq_r = tl.clamp(x_row / sc_r, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
    xq_row_out = tl.reshape(xq_r, (GROUP, BLOCK_L))
    sc_row_out = tl.reshape(u8_r, (GROUP, BLOCK_L // GROUP))

    # Colwise path.
    amax_c = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-8)  # (BLOCK_L,)
    e_c = tl.clamp(tl.ceil(tl.log2(amax_c / FP8_MAX)), -127.0, 127.0)
    sc_c = tl.exp2(e_c)
    u8_c = (e_c.to(tl.int32) + 127).to(tl.uint8)
    xq_c = tl.clamp(x / sc_c[None, :], -FP8_MAX, FP8_MAX).to(tl.float8e4nv)

    # Row outputs (unchanged).
    tl.store(X_fp8_row + rm[:, None] * stride_qm_row + rl[None, :] * stride_ql_row,
             xq_row_out, mask=mask)
    rls = pid_l * (BLOCK_L // GROUP) + tl.arange(0, BLOCK_L // GROUP)
    mask_ls = rls < ((L + GROUP - 1) // GROUP)
    tl.store(X_scale_row + rm[:, None] * stride_sm_row + rls[None, :] * stride_sl_row,
             sc_row_out, mask=mask_m[:, None] & mask_ls[None, :])

    # ── Col outputs: PERMUTED to [G, L, M_g] / [G, L, sc_g] ──────────────
    # Target layout [G, L, M_g] has M_g as the innermost (contiguous) dim.
    # To keep stores coalesced we transpose xq_c (shape [m_local, l]) into
    # [l, m_local] and emit with m_local as the inner axis of the address
    # expression. Adjacent lanes then write adjacent m_local bytes.
    col_base = X_fp8_col_perm + expert_g.to(tl.int64) * stride_cg_perm
    col_addr = col_base + rl[:, None] * stride_cl_perm + rm_local[None, :] * stride_cm_perm
    tl.store(col_addr, tl.trans(xq_c), mask=mask_l[:, None] & mask_m[None, :])

    # Scale: one scale byte per (l_idx, m_scale_group). sg_within is the
    # current m-scale-group index within this expert.
    scale_base = X_scale_col_perm + expert_g.to(tl.int64) * stride_ssg_perm
    scale_addr = scale_base + rl * stride_ssl_perm + sg_within.to(tl.int64) * stride_sss_perm
    tl.store(scale_addr, u8_c, mask=mask_l)


def quant_mxfp8_dual_jagged_permuted(
    x: torch.Tensor,
    group_offs: torch.Tensor,
    group_lens: torch.Tensor,
):
    """Fused rowwise + colwise MX-FP8 quant with PRE-PERMUTED col output.

    Same as ``quant_mxfp8_dual_jagged`` except the col outputs are in the
    ``[G, L, M_g]`` / ``[G, L, M_g//32]`` layout that HIP wgrad consumes,
    eliminating the runtime fp8 permute (~240 µs per step).

    **Requires balanced MoE (M_g uniform across all experts).** Caller must
    verify. Returns 5-tuple matching the base function's shape:
      - ``row_fp8``  : ``[M, L]``
      - ``row_scale``: ``[M, L//32]``
      - ``col_fp8``  : ``[G, L, M_g]``    ← pre-permuted
      - ``col_scale``: ``[G, L, M_g//32]`` ← pre-permuted
      - ``scale_offs``: ``[G+1]`` (same as base for API compatibility)
    """
    assert x.ndim == 2, f"x must be 2D, got {x.shape}"
    M_total, L = x.shape
    assert L % MX_GROUP_SIZE == 0, f"L={L} must be multiple of {MX_GROUP_SIZE}"
    G = group_lens.shape[0]
    GROUP = MX_GROUP_SIZE
    assert M_total % G == 0, f"balanced MoE required: M_total={M_total} not div G={G}"
    M_g = M_total // G
    assert M_g % GROUP == 0, f"M_g={M_g} must be multiple of {GROUP} for balanced quant"

    sg_per_group = (group_lens + GROUP - 1) // GROUP
    scale_offs = torch.cat([
        torch.zeros(1, dtype=group_offs.dtype, device=group_offs.device),
        torch.cumsum(sg_per_group, dim=0),
    ])
    total_sc = int(scale_offs[-1].item())
    block_to_expert = torch.repeat_interleave(
        torch.arange(G, dtype=torch.int32, device=x.device),
        sg_per_group.to(torch.int32),
    )

    sc_g = M_g // GROUP
    row_fp8   = torch.zeros((M_total, L), dtype=torch.float8_e4m3fn, device=x.device)
    row_scale = torch.empty((M_total, L // GROUP), dtype=torch.uint8, device=x.device)
    col_fp8_perm   = torch.zeros((G, L, M_g), dtype=torch.float8_e4m3fn, device=x.device)
    col_scale_perm = torch.empty((G, L, sc_g), dtype=torch.uint8, device=x.device)

    BLOCK_L = 128
    grid = (total_sc, triton.cdiv(L, BLOCK_L))
    _mxfp8_quant_dual_jagged_permuted_kernel[grid](
        x,
        row_fp8, row_scale,
        col_fp8_perm, col_scale_perm,
        group_offs, scale_offs, block_to_expert,
        L, M_g,
        x.stride(0), x.stride(1),
        row_fp8.stride(0), row_fp8.stride(1),
        row_scale.stride(0), row_scale.stride(1),
        col_fp8_perm.stride(0), col_fp8_perm.stride(1), col_fp8_perm.stride(2),
        col_scale_perm.stride(0), col_scale_perm.stride(1), col_scale_perm.stride(2),
        BLOCK_L=BLOCK_L, GROUP=GROUP, FP8_MAX=_FP8_MAX,
    )
    return row_fp8, row_scale, col_fp8_perm, col_scale_perm, scale_offs


def quant_mxfp8_colwise_for_variable_k(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[M, L]`` bf16 -> ``([M, L] fp8_e4m3, [L, M//32] uint8 e8m0)``.

    Column-wise mxfp8 quant with 32-element groups taken along the M (reduction)
    axis. Produces the scale layout consumed by the wgrad variable-K kernel via
    ``tl.dot_scaled`` (see ``grouped_gemm_mxfp8_variable_k_kernel.py``).

    Used for:
      - A / activations: M = token count, L = hidden (K)
      - grad_out:         M = token count, L = output (N)

    Requires ``M % 32 == 0`` AND, for correctness at group boundaries, every
    per-expert segment length must itself be a multiple of 32 so that no MX
    scale group ever spans two expert groups.
    """
    assert x.ndim == 2, f"x must be 2D, got {x.shape}"
    M, L = x.shape
    assert M % MX_GROUP_SIZE == 0, f"M={M} must be multiple of {MX_GROUP_SIZE}"
    x_fp8 = torch.empty((M, L), dtype=torch.float8_e4m3fn, device=x.device)
    x_scale = torch.empty((L, M // MX_GROUP_SIZE), dtype=torch.uint8, device=x.device)

    # BLOCK_M must be a multiple of GROUP (32). 128 gives 4 K-groups per tile,
    # BLOCK_L=64 gives a 128×64 fp8 tile which matches the rowwise/dgrad sizing
    # and fits in a single wavefront pass.
    BLOCK_M, BLOCK_L = 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(L, BLOCK_L))
    _mxfp8_quant_colwise_for_variable_k_kernel[grid](
        x, x_fp8, x_scale, M, L,
        x.stride(0), x.stride(1),
        x_fp8.stride(0), x_fp8.stride(1),
        x_scale.stride(0), x_scale.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_L=BLOCK_L, GROUP=MX_GROUP_SIZE, FP8_MAX=_FP8_MAX,
    )
    return x_fp8, x_scale
