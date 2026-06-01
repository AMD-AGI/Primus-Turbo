###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Fused FP8 cast + transpose + amax Triton kernel.

Given a 2D bf16/f16 input [M, N] and a pre-computed scale, produces:
  - out_fp8  [M, N]  -- quantized row-major output
  - out_t_fp8 [N, M] -- quantized transposed output (contiguous)
  - amax_out (scalar) -- abs-max of the *unscaled* input (for delayed scaling)

Single kernel replaces: quantize kernel + .t().contiguous() copy.
Follows TE's _cast_transpose_triton pattern with 2D grouped tiling.

Registered as a @triton_op so that:
  - torch.compile / Inductor can see through the op
  - Output tensors are standard torch.empty (no triton.reinterpret metadata)
  - register_fake provides correct shapes/strides for compile-time validation
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from primus_turbo.triton.utils.triton_lang_helper import tl_extra_shim

_tanh = tl_extra_shim.tanh


def _fp8_max(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).max


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 1}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _cast_transpose_amax_kernel(
    X_ptr,
    C_ptr,
    T_ptr,
    stride_xm, stride_xn,
    stride_cm, stride_cn,
    stride_tm, stride_tn,
    M, N,
    scale_ptr,
    amax_ptr,
    scale_inv_ptr,
    max_fp8: tl.constexpr,
    COMPUTE_AMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    scale = tl.load(scale_ptr)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    x_ptrs = X_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn
    a = tl.load(x_ptrs, mask=mask)
    val = a.to(tl.float32)

    scaled = val * scale
    scaled = tl.clamp(scaled, -max_fp8, max_fp8)
    fp8_val = scaled.to(C_ptr.dtype.element_ty)

    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, fp8_val, mask=mask)

    # Transpose the tile so the store to T is coalesced (stride-1 in the fast dim).
    # Without this, the transpose store scatters with stride=M which kills
    # write throughput on MI355X under memory pressure.
    fp8_val_t = tl.trans(fp8_val)
    rn2 = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    rm2 = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_t = (rn2 < N)[:, None] & (rm2 < M)[None, :]
    t_ptrs = T_ptr + rn2[:, None] * stride_tm + rm2[None, :] * stride_tn
    tl.store(t_ptrs, fp8_val_t, mask=mask_t)

    if COMPUTE_AMAX:
        tile_amax = tl.max(tl.abs(val))
        tl.atomic_max(amax_ptr, tile_amax, sem="relaxed")

    if pid == 0:
        tl.store(scale_inv_ptr, tl.fdiv(1.0, scale))


# ---------------------------------------------------------------------------
# @triton_op wrapper -- produces clean PyTorch tensors, no reinterpret
# ---------------------------------------------------------------------------

@triton_op("primus_turbo::cast_transpose_fp8_triton", mutates_args=())
def cast_transpose_fp8_triton(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    scale: torch.Tensor,
    amax_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused FP8 cast + transpose + optional amax.

    Args:
        x: 2D input tensor [M, N] (bf16 or f16), must be contiguous.
        fp8_dtype: Target FP8 dtype (e.g. torch.float8_e4m3fn).
        scale: Scalar float32 tensor with the quantization scale.
        amax_out: Optional scalar float32 tensor. If provided, the kernel
                  writes the abs-max of x into it (atomically reduced).

    Returns:
        (cast_out, transpose_out, scale_inv) where:
          cast_out:      [M, N] FP8 tensor (same layout as input)
          transpose_out: [N, M] FP8 tensor (contiguous transpose)
          scale_inv:     scalar float32, 1/scale
    """
    assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"
    if not x.is_contiguous():
        x = x.contiguous()

    M, N = x.shape
    max_fp8 = _fp8_max(fp8_dtype)

    cast_out = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    transpose_out = torch.empty((N, M), dtype=fp8_dtype, device=x.device)
    scale_inv = torch.empty((), dtype=torch.float32, device=x.device)

    compute_amax = amax_out is not None
    if not compute_amax:
        amax_out = torch.empty((), dtype=torch.float32, device=x.device)
    else:
        amax_out.zero_()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    wrap_triton(_cast_transpose_amax_kernel)[grid](
        x, cast_out, transpose_out,
        x.stride(0), x.stride(1),
        cast_out.stride(0), cast_out.stride(1),
        transpose_out.stride(0), transpose_out.stride(1),
        M, N,
        scale,
        amax_out,
        scale_inv,
        max_fp8,
        compute_amax,
    )

    return cast_out, transpose_out, scale_inv


@cast_transpose_fp8_triton.register_fake
def _cast_transpose_fp8_triton_meta(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    scale: torch.Tensor,
    amax_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = x.shape
    return (
        torch.empty((M, N), dtype=fp8_dtype, device=x.device),
        torch.empty((N, M), dtype=fp8_dtype, device=x.device),
        torch.empty((), dtype=torch.float32, device=x.device),
    )


# Backward-compat alias for existing benchmark / call sites
cast_transpose_amax = cast_transpose_fp8_triton


# ---------------------------------------------------------------------------
# Fused bias + GELU(tanh) + cast_transpose + amax Triton kernel
#
# Identical 2D tiling as _cast_transpose_amax_kernel, but inserts
#   val = (x + bias); val = 0.5*val*(1+tanh(k*val*(1+0.044715*val^2)))
# before the FP8 quantize + transpose store.  Eliminates one bf16 GMEM
# round-trip per MLP fc2 input in the Flux double-stream block.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 1}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _bias_gelu_cast_transpose_kernel(
    X_ptr,
    Bias_ptr,
    C_ptr,
    T_ptr,
    stride_xm, stride_xn,
    stride_cm, stride_cn,
    stride_tm, stride_tn,
    M, N,
    scale_ptr,
    amax_ptr,
    scale_inv_ptr,
    max_fp8: tl.constexpr,
    COMPUTE_AMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    scale = tl.load(scale_ptr)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    x_ptrs = X_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn
    a = tl.load(x_ptrs, mask=mask)
    val = a.to(tl.float32)

    # bias add -- bias is [N], broadcast over M
    bias_mask = rn < N
    bias = tl.load(Bias_ptr + rn, mask=bias_mask).to(tl.float32)
    val = val + bias[None, :]

    # GELU tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*x*(1+0.044715*x^2)))
    # Constants match openai_gelu_no_jit used by Flux.
    inner = 0.7978845608028654 * val * (1.0 + 0.044715 * val * val)
    val = 0.5 * val * (1.0 + _tanh(inner))

    # FP8 quantize + row-major store
    scaled = val * scale
    scaled = tl.clamp(scaled, -max_fp8, max_fp8)
    fp8_val = scaled.to(C_ptr.dtype.element_ty)

    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, fp8_val, mask=mask)

    # Transposed store (coalesced via tl.trans)
    fp8_val_t = tl.trans(fp8_val)
    rn2 = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    rm2 = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_t = (rn2 < N)[:, None] & (rm2 < M)[None, :]
    t_ptrs = T_ptr + rn2[:, None] * stride_tm + rm2[None, :] * stride_tn
    tl.store(t_ptrs, fp8_val_t, mask=mask_t)

    if COMPUTE_AMAX:
        tile_amax = tl.max(tl.abs(val))
        tl.atomic_max(amax_ptr, tile_amax, sem="relaxed")

    if pid == 0:
        tl.store(scale_inv_ptr, tl.fdiv(1.0, scale))


# ---------------------------------------------------------------------------
# @triton_op wrapper for bias_gelu_cast_transpose
# ---------------------------------------------------------------------------

@triton_op("primus_turbo::bias_gelu_cast_transpose_fp8", mutates_args=())
def bias_gelu_cast_transpose_fp8(
    x: torch.Tensor,
    bias: torch.Tensor,
    fp8_dtype: torch.dtype,
    scale: torch.Tensor,
    amax_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused bias + GELU(tanh) + FP8 cast + transpose + optional amax.

    Computes GELU(x + bias) then quantizes to FP8, producing both row-major
    and transposed outputs.  Eliminates the bf16 intermediate that would
    otherwise be written by separate bias+GELU then read by cast_transpose.

    Args:
        x: 2D input tensor [M, N] (bf16 or f16), must be contiguous.
        bias: 1D bias tensor [N].
        fp8_dtype: Target FP8 dtype (e.g. torch.float8_e4m3fn).
        scale: Scalar float32 tensor with the quantization scale.
        amax_out: Optional scalar float32 tensor for amax capture.

    Returns:
        (cast_out, transpose_out, scale_inv) where:
          cast_out:      [M, N] FP8 tensor
          transpose_out: [N, M] FP8 tensor (contiguous transpose)
          scale_inv:     scalar float32, 1/scale
    """
    assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"
    assert bias.ndim == 1, f"Expected 1D bias, got {bias.ndim}D"
    assert bias.shape[0] == x.shape[1], (
        f"Bias size {bias.shape[0]} != input columns {x.shape[1]}"
    )
    if not x.is_contiguous():
        x = x.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()

    M, N = x.shape
    max_fp8 = _fp8_max(fp8_dtype)

    cast_out = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    transpose_out = torch.empty((N, M), dtype=fp8_dtype, device=x.device)
    scale_inv = torch.empty((), dtype=torch.float32, device=x.device)

    compute_amax = amax_out is not None
    if not compute_amax:
        amax_out = torch.empty((), dtype=torch.float32, device=x.device)
    else:
        amax_out.zero_()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    wrap_triton(_bias_gelu_cast_transpose_kernel)[grid](
        x, bias, cast_out, transpose_out,
        x.stride(0), x.stride(1),
        cast_out.stride(0), cast_out.stride(1),
        transpose_out.stride(0), transpose_out.stride(1),
        M, N,
        scale,
        amax_out,
        scale_inv,
        max_fp8,
        compute_amax,
    )

    return cast_out, transpose_out, scale_inv


@bias_gelu_cast_transpose_fp8.register_fake
def _bias_gelu_cast_transpose_fp8_meta(
    x: torch.Tensor,
    bias: torch.Tensor,
    fp8_dtype: torch.dtype,
    scale: torch.Tensor,
    amax_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = x.shape
    return (
        torch.empty((M, N), dtype=fp8_dtype, device=x.device),
        torch.empty((N, M), dtype=fp8_dtype, device=x.device),
        torch.empty((), dtype=torch.float32, device=x.device),
    )


# ---------------------------------------------------------------------------
# Fused dGELU(tanh) backward + bias_grad + FP8 cast_transpose + amax
#
# Backward counterpart of _bias_gelu_cast_transpose_kernel.  Given
# grad_output and the pre-activation input x (before bias+GELU), computes:
#   val = x + bias
#   dgelu = GELU'(val) * grad_output          (fp32)
#   -> FP8 quantize + transpose + amax        (same tiling as forward)
#   -> partial bias grad = row-sum of dgelu   (fp32, reduced outside kernel)
#
# Eliminates the bf16 intermediate between FusedBiasGeluCastTranspose.backward
# and DelayedFP8LinearTensorwiseFunction.backward.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 1}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _dbias_dgelu_cast_transpose_kernel(
    GradOut_ptr,
    X_ptr,
    Bias_ptr,
    C_ptr,
    T_ptr,
    Dact_bf16_ptr,
    PBias_ptr,
    stride_gm, stride_gn,
    stride_xm, stride_xn,
    stride_cm, stride_cn,
    stride_tm, stride_tn,
    stride_dm, stride_dn,
    stride_pbm, stride_pbn,
    M, N,
    scale_ptr,
    amax_ptr,
    scale_inv_ptr,
    max_fp8: tl.constexpr,
    COMPUTE_AMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    scale = tl.load(scale_ptr)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    # Load grad_output
    g_ptrs = GradOut_ptr + rm[:, None] * stride_gm + rn[None, :] * stride_gn
    grad = tl.load(g_ptrs, mask=mask).to(tl.float32)

    # Load x (pre-activation input, before bias)
    x_ptrs = X_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn
    x_val = tl.load(x_ptrs, mask=mask).to(tl.float32)

    # bias add
    bias_mask = rn < N
    bias = tl.load(Bias_ptr + rn, mask=bias_mask).to(tl.float32)
    val = x_val + bias[None, :]

    # dGELU(tanh): d/dx[0.5*x*(1+tanh(k*x*(1+0.044715*x^2)))]
    K: tl.constexpr = 0.7978845608028654
    C1: tl.constexpr = 0.044715
    val_sq = val * val
    inner = K * val * (1.0 + C1 * val_sq)
    tanh_inner = _tanh(inner)
    dtanh = 1.0 - tanh_inner * tanh_inner
    d_inner = K * (1.0 + 3.0 * C1 * val_sq)
    dgelu = 0.5 * (1.0 + tanh_inner + val * dtanh * d_inner)

    dact = grad * dgelu

    # Partial bias grad: sum over rows in this tile
    partial_dbias = tl.sum(dact, axis=0)
    pb_ptrs = PBias_ptr + pid_m.to(tl.int64) * stride_pbm + rn * stride_pbn
    tl.store(pb_ptrs, partial_dbias, mask=bias_mask)

    # bf16 dact store (high-precision gradient for backprop to prior layers)
    d_ptrs = Dact_bf16_ptr + rm[:, None] * stride_dm + rn[None, :] * stride_dn
    tl.store(d_ptrs, dact.to(tl.bfloat16), mask=mask)

    # FP8 quantize + row-major store
    scaled = dact * scale
    scaled = tl.clamp(scaled, -max_fp8, max_fp8)
    fp8_val = scaled.to(C_ptr.dtype.element_ty)

    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, fp8_val, mask=mask)

    # Transposed store (coalesced via tl.trans)
    fp8_val_t = tl.trans(fp8_val)
    rn2 = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    rm2 = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_t = (rn2 < N)[:, None] & (rm2 < M)[None, :]
    t_ptrs = T_ptr + rn2[:, None] * stride_tm + rm2[None, :] * stride_tn
    tl.store(t_ptrs, fp8_val_t, mask=mask_t)

    if COMPUTE_AMAX:
        tile_amax = tl.max(tl.abs(dact))
        tl.atomic_max(amax_ptr, tile_amax, sem="relaxed")

    if pid == 0:
        tl.store(scale_inv_ptr, tl.fdiv(1.0, scale))


# ---------------------------------------------------------------------------
# @triton_op wrapper for dbias_dgelu_cast_transpose
# ---------------------------------------------------------------------------

@triton_op("primus_turbo::dbias_dgelu_cast_transpose_fp8", mutates_args=())
def dbias_dgelu_cast_transpose_fp8(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
    fp8_dtype: torch.dtype,
    scale: torch.Tensor,
    amax_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused dGELU(tanh) backward + bias_grad + FP8 cast + transpose + amax.

    Computes GELU'(x + bias) * grad_output in fp32, then produces:
      - bf16 dact for high-precision gradient backprop to prior layers
      - FP8 dact (row-major + transposed) for backward GEMMs
      - bias gradient

    Args:
        grad_output: 2D gradient tensor [M, N] (bf16 or f16).
        x: 2D pre-activation input [M, N] (before bias+GELU was applied).
        bias: 1D bias tensor [N].
        fp8_dtype: Target FP8 dtype (e.g. torch.float8_e5m2 for backward).
        scale: Scalar float32 tensor with the quantization scale.
        amax_out: Optional scalar float32 tensor for amax capture.

    Returns:
        (dact_fp8, dact_t_fp8, scale_inv, grad_bias, dact_bf16) where:
          dact_fp8:   [M, N] FP8 tensor (row-major)
          dact_t_fp8: [N, M] FP8 tensor (contiguous transpose)
          scale_inv:  scalar float32, 1/scale
          grad_bias:  [N] float32 bias gradient
          dact_bf16:  [M, N] bf16 tensor (high-precision dGELU output)
    """
    assert grad_output.ndim == 2, f"Expected 2D grad_output, got {grad_output.ndim}D"
    assert x.ndim == 2, f"Expected 2D x, got {x.ndim}D"
    assert bias.ndim == 1, f"Expected 1D bias, got {bias.ndim}D"
    assert x.shape == grad_output.shape, (
        f"Shape mismatch: x {x.shape} vs grad_output {grad_output.shape}"
    )
    assert bias.shape[0] == x.shape[1], (
        f"Bias size {bias.shape[0]} != input columns {x.shape[1]}"
    )
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()

    M, N = x.shape
    max_fp8 = _fp8_max(fp8_dtype)

    dact_fp8 = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    dact_t_fp8 = torch.empty((N, M), dtype=fp8_dtype, device=x.device)
    dact_bf16 = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    scale_inv = torch.empty((), dtype=torch.float32, device=x.device)

    compute_amax = amax_out is not None
    if not compute_amax:
        amax_out = torch.empty((), dtype=torch.float32, device=x.device)
    else:
        amax_out.zero_()

    max_grid_m = triton.cdiv(M, 64)
    partial_dbias = torch.zeros(
        (max_grid_m, N), dtype=torch.float32, device=x.device,
    )

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    wrap_triton(_dbias_dgelu_cast_transpose_kernel)[grid](
        grad_output, x, bias, dact_fp8, dact_t_fp8, dact_bf16, partial_dbias,
        grad_output.stride(0), grad_output.stride(1),
        x.stride(0), x.stride(1),
        dact_fp8.stride(0), dact_fp8.stride(1),
        dact_t_fp8.stride(0), dact_t_fp8.stride(1),
        dact_bf16.stride(0), dact_bf16.stride(1),
        partial_dbias.stride(0), partial_dbias.stride(1),
        M, N,
        scale,
        amax_out,
        scale_inv,
        max_fp8,
        compute_amax,
    )

    grad_bias = partial_dbias.sum(dim=0)

    return dact_fp8, dact_t_fp8, scale_inv, grad_bias, dact_bf16


@dbias_dgelu_cast_transpose_fp8.register_fake
def _dbias_dgelu_cast_transpose_fp8_meta(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
    fp8_dtype: torch.dtype,
    scale: torch.Tensor,
    amax_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = x.shape
    return (
        torch.empty((M, N), dtype=fp8_dtype, device=x.device),
        torch.empty((N, M), dtype=fp8_dtype, device=x.device),
        torch.empty((), dtype=torch.float32, device=x.device),
        torch.empty((N,), dtype=torch.float32, device=x.device),
        torch.empty((M, N), dtype=torch.bfloat16, device=x.device),
    )
