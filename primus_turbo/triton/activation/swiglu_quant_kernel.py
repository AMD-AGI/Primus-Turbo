###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Fused SwiGLU + FP8 rowwise quant kernel.

Does in one pass what the separate ``swiglu_fwd_kernel`` + ``quantize_fp8_rowwise``
path does in three:
  (1)  read  gate||up  (bf16, [M, 2N])
  (2)  write bf16 out  (bf16, [M, N])   ← eliminated
  (3)  read  bf16 out ×2 (two-scan amax+quant)  ← eliminated
  (4)  write fp8 out + scale

Bytes saved per call on gpt-oss-20B gate_up (M=65536, N=2880):
  baseline: ~1.98 GB HBM
  fused:    ~1.26 GB HBM
  savings:  ~720 MB  → ~140 µs at 5 TB/s HBM

Designed for the MoE gate_up epilogue where the GEMM produces [M, 2N] bf16
and the downstream op is silu(gate) * up → rowwise-fp8 quant.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_quant_rowwise_fwd_kernel(
    # pointers
    x_ptr,              # [M, 2N] bf16/fp16 — gate in [:, :N], up in [:, N:]
    probs_ptr,          # [M] fp32/bf16/None
    out_fp8_ptr,        # [M, N]   fp8_e4m3
    out_scale_inv_ptr,  # [M]      fp32 (1/scale, matches Primus quant convention)
    # strides (in elements)
    stride_x_token,
    stride_probs_token,
    stride_out_token,
    stride_scale_token,
    # sizes
    M,
    N,                  # half-width (output dim); input is 2N wide
    # constexpr
    FP8_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,   # must be >= N (power of 2, cols outside N are masked)
    USE_PROBS: tl.constexpr,
):
    """One program per row; 1-pass fused silu(gate)*up * (probs) → rowwise fp8."""
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_row = x_ptr + row.to(tl.int64) * stride_x_token
    gate = tl.load(x_row + cols,         mask=mask, other=0.0).to(tl.float32)
    up   = tl.load(x_row + N + cols,     mask=mask, other=0.0).to(tl.float32)

    y = (gate * tl.sigmoid(gate)) * up
    if USE_PROBS:
        probs = tl.load(probs_ptr + row * stride_probs_token).to(tl.float32)
        y = y * probs

    # Rowwise amax in fp32. `tl.max` over the fp32 row vector uses LDS reduction.
    amax = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-8)
    scale     = FP8_MAX / amax
    scale_inv = 1.0 / scale

    # Scale + clamp + cast in registers. No bf16 intermediate hits HBM.
    y_q = tl.clamp(y * scale, -FP8_MAX, FP8_MAX).to(out_fp8_ptr.dtype.element_ty)

    tl.store(out_fp8_ptr + row.to(tl.int64) * stride_out_token + cols, y_q, mask=mask)
    tl.store(out_scale_inv_ptr + row * stride_scale_token, scale_inv)


def swiglu_quant_rowwise_fwd(
    x: torch.Tensor,                         # [M, 2N]
    out_fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    probs: torch.Tensor | None = None,       # [M] or None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU + rowwise-FP8 quant. Returns (out_fp8, scale_inv).

    Args:
        x:     Input activations, shape [M, 2N]. Gate=x[:, :N], up=x[:, N:].
        out_fp8_dtype: fp8 dtype (default e4m3).
        probs: Per-token probs; if provided, output is silu(gate)*up*probs.

    Returns:
        out_fp8:   [M, N] fp8 quantized activations.
        scale_inv: [M]    fp32 per-row inverse scale (multiply to dequant).
    """
    assert x.ndim == 2, f"x must be 2D, got {x.shape}"
    M, two_N = x.shape
    assert two_N % 2 == 0, f"x last dim {two_N} must be even (gate || up packed)"
    N = two_N // 2
    assert x.is_contiguous(), "x must be contiguous"

    # BLOCK_N is the smallest power of 2 >= N. Cols outside N are masked.
    BLOCK_N = max(64, triton.next_power_of_2(N))
    assert BLOCK_N <= 16384, f"N={N} too large for 1-pass; use 2-pass kernel instead"

    out_fp8 = torch.empty((M, N), dtype=out_fp8_dtype, device=x.device)
    scale_inv = torch.empty((M,), dtype=torch.float32, device=x.device)

    fp8_max = torch.finfo(out_fp8_dtype).max
    use_probs = probs is not None
    probs_ptr = probs if use_probs else x  # any valid ptr; unused when USE_PROBS=False
    stride_probs = probs.stride(0) if use_probs else 0

    swiglu_quant_rowwise_fwd_kernel[(M,)](
        x, probs_ptr, out_fp8, scale_inv,
        x.stride(0),
        stride_probs,
        out_fp8.stride(0),
        scale_inv.stride(0),
        M, N,
        FP8_MAX=fp8_max,
        BLOCK_N=BLOCK_N,
        USE_PROBS=use_probs,
        num_warps=8,
    )
    return out_fp8, scale_inv


# ─────────────────────────────────────────────────────────────────────────────
# MX-FP8 variant — same fusion, e8m0 per-32-K-element scales instead of rowwise.
#
# Output scale layout matches what ``grouped_gemm_mxfp8_triton_kernel`` and its
# variable-K sibling consume: ``[M, N // 32]`` uint8 e8m0 (bias 127).
#
# Saves the bf16 intermediate roundtrip AND produces the exact scale format
# needed by the MX-FP8 grouped GEMM — so a full MoE forward becomes:
#   [gate||up] bf16 → (this kernel) → ([M, N] fp8 + [M, N//32] e8m0)
#                                     → grouped_gemm_mxfp8  → [M, N'] bf16
# ─────────────────────────────────────────────────────────────────────────────


_MX_GROUP_SIZE = 32  # OCP MX-FP8 mandatory K-group size for e8m0 scales


@triton.jit
def swiglu_quant_mxfp8_fwd_kernel(
    # pointers
    x_ptr,              # [M, 2N] bf16/fp16 — gate in [:, :N], up in [:, N:]
    probs_ptr,          # [M] fp32/bf16 (unused if USE_PROBS=False)
    out_fp8_ptr,        # [M, N]          fp8_e4m3
    out_scale_ptr,      # [M, N // GROUP] uint8 e8m0
    # strides (in elements)
    stride_x_token,
    stride_probs_token,
    stride_out_token,
    stride_scale_token,
    # sizes
    M,
    N,                  # half-width (output dim); input is 2N wide
    # constexpr
    FP8_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,   # must be >= N; must be a multiple of GROUP
    GROUP: tl.constexpr,     # 32 for MX-FP8
    USE_PROBS: tl.constexpr,
):
    """One program per row; 1-pass fused silu(gate)*up * (probs) → MX-FP8 rowwise.

    Scales are grouped along N with group_size=32 (OCP MX-FP8), laid out as
    ``[M, N//32]`` uint8 e8m0 (bias 127). Tail groups past ``N`` are masked
    during the store so ``N`` need not be a multiple of 32 at launch, but for
    the shipping MoE shapes (N=2880, 5760 …) it always is.
    """
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_row = x_ptr + row.to(tl.int64) * stride_x_token
    gate = tl.load(x_row + cols,         mask=mask, other=0.0).to(tl.float32)
    up   = tl.load(x_row + N + cols,     mask=mask, other=0.0).to(tl.float32)

    y = (gate * tl.sigmoid(gate)) * up
    if USE_PROBS:
        probs = tl.load(probs_ptr + row * stride_probs_token).to(tl.float32)
        y = y * probs

    # Group along N: reshape (BLOCK_N,) -> (BLOCK_N // GROUP, GROUP).
    y2 = tl.reshape(y, (BLOCK_N // GROUP, GROUP))
    amax = tl.maximum(tl.max(tl.abs(y2), axis=-1, keep_dims=True), 1e-8)
    # e8m0 exponent: ceil(log2(amax / FP8_MAX)), clipped to valid range.
    e_int = tl.clamp(tl.ceil(tl.log2(amax / FP8_MAX)), -127.0, 127.0)
    scale = tl.exp2(e_int)
    e_u8 = (e_int.to(tl.int32) + 127).to(tl.uint8)  # (BLOCK_N // GROUP, 1)
    yq = tl.clamp(y2 / scale, -FP8_MAX, FP8_MAX).to(out_fp8_ptr.dtype.element_ty)

    yq_flat = tl.reshape(yq, (BLOCK_N,))
    sc_flat = tl.reshape(e_u8, (BLOCK_N // GROUP,))

    # Store fp8 (masked on N).
    tl.store(out_fp8_ptr + row.to(tl.int64) * stride_out_token + cols, yq_flat, mask=mask)

    # Store scales (masked on N//GROUP).
    scale_cols = tl.arange(0, BLOCK_N // GROUP)
    scale_mask = scale_cols < ((N + GROUP - 1) // GROUP)
    tl.store(
        out_scale_ptr + row.to(tl.int64) * stride_scale_token + scale_cols,
        sc_flat,
        mask=scale_mask,
    )


def swiglu_quant_mxfp8_fwd(
    x: torch.Tensor,                         # [M, 2N] bf16/fp16
    out_fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    probs: torch.Tensor | None = None,       # [M] or None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU + MX-FP8 rowwise quant.

    Produces the exact scale layout that ``grouped_gemm_mxfp8_triton_kernel``
    expects for its ``a`` operand: uint8 e8m0 at ``[M, N//32]``.

    Args:
        x:             Input activations, shape [M, 2N]. Gate=x[:, :N], up=x[:, N:].
        out_fp8_dtype: fp8 dtype (default e4m3fn).
        probs:         Per-token probs; if provided, output is silu(gate)*up*probs.

    Returns:
        out_fp8:   ``[M, N]``      fp8_e4m3 quantised activations.
        out_scale: ``[M, N // 32]`` uint8 e8m0.

    Constraint: ``N`` must be a multiple of 32 (OCP MX-FP8 group size). For
    ``N`` not divisible by 32, either pad manually or fall back to the
    rowwise fusion ``swiglu_quant_rowwise_fwd``.
    """
    assert x.ndim == 2, f"x must be 2D, got {x.shape}"
    M, two_N = x.shape
    assert two_N % 2 == 0, f"x last dim {two_N} must be even (gate || up packed)"
    N = two_N // 2
    assert N % _MX_GROUP_SIZE == 0, f"N={N} must be multiple of {_MX_GROUP_SIZE} (MX-FP8)"
    assert x.is_contiguous(), "x must be contiguous"

    # BLOCK_N must be both ≥ N and a multiple of GROUP. next_power_of_2 ≥ N
    # already satisfies "multiple of 32" as long as N ≥ 32 (which we've
    # asserted above). Cols outside N are masked at load/store time.
    BLOCK_N = max(_MX_GROUP_SIZE, triton.next_power_of_2(N))
    assert BLOCK_N <= 16384, f"N={N} too large for 1-pass; use 2-pass kernel instead"

    out_fp8 = torch.empty((M, N), dtype=out_fp8_dtype, device=x.device)
    out_scale = torch.empty((M, N // _MX_GROUP_SIZE), dtype=torch.uint8, device=x.device)

    fp8_max = torch.finfo(out_fp8_dtype).max
    use_probs = probs is not None
    probs_ptr = probs if use_probs else x  # any valid ptr; unused when USE_PROBS=False
    stride_probs = probs.stride(0) if use_probs else 0

    swiglu_quant_mxfp8_fwd_kernel[(M,)](
        x, probs_ptr, out_fp8, out_scale,
        x.stride(0),
        stride_probs,
        out_fp8.stride(0),
        out_scale.stride(0),
        M, N,
        FP8_MAX=fp8_max,
        BLOCK_N=BLOCK_N,
        GROUP=_MX_GROUP_SIZE,
        USE_PROBS=use_probs,
        num_warps=8,
    )
    return out_fp8, out_scale
