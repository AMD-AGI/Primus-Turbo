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
    stride_xm,
    stride_xn,
    stride_cm,
    stride_cn,
    stride_tm,
    stride_tn,
    M,
    N,
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

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    wrap_triton(_cast_transpose_amax_kernel)[grid](
        x,
        cast_out,
        transpose_out,
        x.stride(0),
        x.stride(1),
        cast_out.stride(0),
        cast_out.stride(1),
        transpose_out.stride(0),
        transpose_out.stride(1),
        M,
        N,
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
