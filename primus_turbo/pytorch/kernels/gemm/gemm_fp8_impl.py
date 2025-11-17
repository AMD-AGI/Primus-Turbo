###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Tuple

import torch
import triton
from torch.library import custom_op, triton_op, wrap_triton

from primus_turbo.pytorch.core.float8 import ScalingGranularity
from primus_turbo.triton.gemm.gemm_fp8_kernel import (
    gemm_fp8_blockwise_nn_kernel,
    gemm_fp8_blockwise_nt_kernel,
    gemm_fp8_blockwise_tn_kernel,
)
from primus_turbo.triton.quantization.quant_blockwise import (
    quant_fp8_blockwise_for_act_grad_kernel,
    quant_fp8_blockwise_for_weight_kernel,
)


def ceil_div(a, b):
    return (a + b - 1) // b


def get_gemm_logical_shape(
    a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> Tuple[int, int, int]:
    assert (
        a.ndim == 2 and b.ndim == 2
    ), f"Expected both a and b to be 2D tensors, but got a.ndim={a.ndim}, b.ndim={b.ndim}"
    M = a.shape[1] if trans_a else a.shape[0]
    Ka = a.shape[0] if trans_a else a.shape[1]
    Kb = b.shape[1] if trans_b else b.shape[0]
    N = b.shape[0] if trans_b else b.shape[1]
    assert Ka == Kb, f"GEMM K mismatch: a has K={Ka}, b has K={Kb}"
    return M, N, Ka


@triton_op("primus_turbo::quant_fp8_blockwise_for_weight_impl", mutates_args=())
def quant_fp8_blockwise_for_weight_impl(
    w: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantization for fp8 blockwise (weight).

    Quantizes a 2D or 3D weight tensor using blockwise scales along both axes.
    Assumes `w` is contiguous and 2D or 3D.

    Returns:
        w_fp8: FP8-quantized weight tensor.
        w_scales: Per-block scale tensor in float32.
    """

    assert w.dim() in (2, 3)
    if not w.is_contiguous():
        w = w.contiguous()

    ori_dims = w.dim()
    if ori_dims == 2:
        B, M, N = 1, *w.shape
        w = w.unsqueeze(0)
    else:
        B, M, N = w.shape
    w_fp8 = torch.empty((B, M, N), dtype=dtype, device=w.device)
    w_scales = torch.empty(
        (B, ceil_div(M, block_size), ceil_div(N, block_size)),
        dtype=torch.float32,
        device=w.device,
    )
    grid = (B, triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    wrap_triton(quant_fp8_blockwise_for_weight_kernel)[grid](
        w,
        w_fp8,
        w_scales,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
    )

    if ori_dims == 2:
        w_fp8 = w_fp8.squeeze(0)
        w_scales = w_scales.squeeze(0)
    return w_fp8, w_scales


@quant_fp8_blockwise_for_weight_impl.register_fake
def quant_fp8_blockwise_for_weight_impl_meta(
    w: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert w.dim() in (2, 3)
    ori_dims = w.dim()
    if ori_dims == 2:
        B, M, N = 1, *w.shape
        w = w.unsqueeze(0)
    else:
        B, M, N = w.shape
    w_fp8 = torch.empty((B, M, N), dtype=dtype, device=w.device)
    w_scales = torch.empty(
        (B, ceil_div(M, block_size), ceil_div(N, block_size)),
        dtype=torch.float32,
        device=w.device,
    )
    if ori_dims == 2:
        w_fp8 = w_fp8.squeeze(0)
        w_scales = w_scales.squeeze(0)
    return w_fp8, w_scales


@triton_op("primus_turbo::quant_fp8_blockwise_for_act_grad_impl", mutates_args=())
def quant_fp8_blockwise_for_act_grad_impl(
    x: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantization for fp8 blockwise (activation grad).

    Quantizes a 2D tensor both row-wise and column-wise for activation gradients.
    Assumes `x` is contiguous and 2D.

    Returns:
        x_fp8_row: Row-wise quantized tensor.
        x_scales_row: Row-wise scale tensor (float32).
        x_fp8_col: Column-wise quantized tensor.
        x_scales_col: Column-wise scale tensor (float32).
    """
    assert x.is_contiguous()
    M, N = x.shape
    x_fp8_row = torch.empty(M, N, dtype=dtype, device=x.device)
    x_scales_row = torch.empty(M, ceil_div(N, block_size), dtype=torch.float32, device=x.device)
    x_fp8_col = torch.empty(M, N, dtype=dtype, device=x.device)
    x_scales_col = torch.empty(ceil_div(M, block_size), N, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    wrap_triton(quant_fp8_blockwise_for_act_grad_kernel)[grid](
        x,
        x_fp8_row,
        x_scales_row,
        x_fp8_col,
        x_scales_col,
        M,
        N,
        block_size,
        torch.finfo(dtype).max,
    )
    return x_fp8_row, x_scales_row, x_fp8_col, x_scales_col


@quant_fp8_blockwise_for_act_grad_impl.register_fake
def quant_fp8_blockwise_for_act_grad_impl_meta(
    x: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = x.shape
    x_fp8_row = torch.empty(M, N, dtype=dtype, device=x.device)
    x_scales_row = torch.empty(M, ceil_div(N, block_size), dtype=torch.float32, device=x.device)
    x_fp8_col = torch.empty(M, N, dtype=dtype, device=x.device)
    x_scales_col = torch.empty(ceil_div(M, block_size), N, dtype=torch.float32, device=x.device)
    return x_fp8_row, x_scales_row, x_fp8_col, x_scales_col


@custom_op("primus_turbo::gemm_fp8_blockwise_impl", mutates_args=(), device_types="cuda")
def gemm_fp8_blockwise_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype,
    scale_group_size_m: int,
    scale_group_size_n: int,
    scale_group_size_k: int,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    """
    Blockwise FP8 GEMM.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    assert a_scales.is_contiguous() and b_scales.is_contiguous(), "Scales must be contiguous"

    M, N, K = get_gemm_logical_shape(a, b, trans_a, trans_b)
    c = torch.empty(M, N, dtype=out_dtype, device=a.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    if trans_a == False and trans_b == True:
        # NT
        wrap_triton(gemm_fp8_blockwise_nt_kernel)[grid](
            a, b, c, a_scales, b_scales, M, N, K, BLOCK_SIZE_K=scale_group_size_k
        )
    elif trans_a == False and trans_b == False:
        # NN
        wrap_triton(gemm_fp8_blockwise_nn_kernel)[grid](
            a, b, c, a_scales, b_scales, M, N, K, BLOCK_SIZE_K=scale_group_size_k
        )
    elif trans_a == True and trans_b == False:
        # TN
        wrap_triton(gemm_fp8_blockwise_tn_kernel)[grid](
            a, b, c, a_scales, b_scales, M, N, K, BLOCK_SIZE_K=scale_group_size_k
        )
    else:
        raise NotImplementedError(f"Unsupported trans_a={trans_a}, trans_b={trans_b}")

    return c


@gemm_fp8_blockwise_impl.register_fake
def gemm_fp8_blockwise_impl_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype,
    scale_group_size_m: int,
    scale_group_size_n: int,
    scale_group_size_k: int,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    M, N, _ = get_gemm_logical_shape(a, b, trans_a, trans_b)
    c = torch.empty(M, N, dtype=out_dtype, device=a.device)
    return c


def gemm_fp8_impl(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    backend: str = "hipblaslt",
    granularity: ScalingGranularity = ScalingGranularity.TENSORWISE,
) -> torch.Tensor:
    assert backend in ("hipblaslt", "ck"), f"Unsupported backend: {backend}"

    if backend == "hipblaslt":
        out = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp8(
            a, a_scale_inv, b, b_scale_inv, out_dtype, trans_a, trans_b, trans_c, granularity.name
        )
    elif backend == "ck":
        out = torch.ops.primus_turbo_cpp_extension.ck_gemm_fp8(
            a, b, a_scale_inv, b_scale_inv, trans_a, trans_b, out_dtype, granularity.name
        )

    return out
