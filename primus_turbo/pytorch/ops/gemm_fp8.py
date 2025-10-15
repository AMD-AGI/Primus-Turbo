###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Union

import torch

from primus_turbo.pytorch.core.float8 import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import (
    gemm_fp8_blockwise_impl,
    gemm_fp8_impl,
    quant_fp8_blockwise_for_weight_impl,
)
from primus_turbo.pytorch.kernels.quantize import quant_fp8_blockwise_impl
from primus_turbo.pytorch.ops.quantization import quantize_fp8

__all__ = ["gemm_fp8", "gemm_fp8_blockwise"]


# TODO: opt and refact
class BlockwiseFP8GemmFunction(torch.autograd.Function):
    """
    Autograd function for FP8 blockwise GEMM.

    This class implements the forward and backward pass for a GEMM using blockwise FP8 quantization.

    Quantization granularity:
        - Input `a` is quantized with shape [1, blocksize]
        - Input `b` is quantized with shape [blocksize, blocksize]

    Forward:
        - Quantize input `a`.
        - Quantize input `b`.
        - Perform matrix multiplication: Y = a @ b.
        - Output in high-precision dtype (same as `a` or explicitly set via `out_dtype`).

    Backward:
        - Quantize `grad_out` both row-wise and column-wise for dgrad/wgrad respectively.
        - Re-quantize input `a` column-wise for use in wgrad.
        - Compute:
            * grad_a = grad_out @ b
            * grad_b = grad_out.T @ x
        - Both gradients are returned in high precision.

    Returns:
        out (Tensor): Output tensor of shape [..., M, N], same dtype as `a` or `out_dtype`.
    """

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        # Check
        assert config != None
        assert config.granularity == ScalingGranularity.BLOCKWISE
        assert config.block_size != None
        assert a.ndim >= 2, "Input tensor a must have at least 2 dimensions."
        assert b.ndim == 2, "Weight tensor must be 2-dimensional."

        # TODO: more layout
        assert trans_a == False and trans_b == True, "Currently only support NT layout"
        # TODO: more format
        assert config.format == Format.E4M3

        a_dtype = float8_e4m3
        b_dtype = float8_e4m3

        block_size = config.block_size

        *batch_shape, M, K = a.shape
        a = a.view(-1, K)  # flatten shape [BM, K]

        # Quantize input activation (row): shape [M, K] → FP8
        a_fp8_row, a_scales_row = quant_fp8_blockwise_impl(a, a_dtype, axis=1, block_size=block_size)
        # Quantize weight blockwise: shape [N, K] → FP8
        b_fp8, b_scales = quant_fp8_blockwise_for_weight_impl(b, b_dtype, block_size)

        # Perform NT GEMM in quantized domain:
        # out = a_fp8_row @ w_fp8.T → shape [M, N]
        out = gemm_fp8_blockwise_impl(
            a_fp8_row,
            b_fp8,
            a_scales_row,
            b_scales,
            out_dtype=out_dtype,
            scale_group_size_m=1,
            scale_group_size_n=block_size,
            scale_group_size_k=block_size,
            trans_a=False,
            trans_b=True,
        )
        out = out.view(*batch_shape, M, -1)  # restore shape

        # Save tensors for backward
        ctx.save_for_backward(a, b_fp8, b_scales)
        ctx.config = config
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        a, b_fp8, b_scales = ctx.saved_tensors
        config = ctx.config
        block_size = config.block_size

        out_dtype = float8_e4m3
        a_dtype = float8_e4m3

        *batch_shape, M, N = grad_out.shape
        grad_out = grad_out.view(-1, N)

        # Quantize grad_out in both row-wise and column-wise directions:
        # - row-wise: for dgrad (grad_x)
        # - col-wise: for wgrad (grad_w)
        grad_out_fp8_row, grad_out_scales_row = quant_fp8_blockwise_impl(grad_out, out_dtype, -1, block_size)
        grad_out_fp8_col, grad_out_scales_col = quant_fp8_blockwise_impl(grad_out, out_dtype, -2, block_size)
        # TODO: dequant + quant kernel
        a_fp8_col, a_scales_col = quant_fp8_blockwise_impl(a, a_dtype, axis=0, block_size=block_size)

        # DGrad NN:
        # grad_x = grad_out @ weight
        # [m, k] = [m, n] * [n, k]
        grad_a = gemm_fp8_blockwise_impl(
            grad_out_fp8_row,
            b_fp8,
            grad_out_scales_row,
            b_scales,
            out_dtype=grad_out.dtype,
            scale_group_size_m=1,
            scale_group_size_n=block_size,
            scale_group_size_k=block_size,
            trans_a=False,
            trans_b=False,
        )

        # WGrad TN:
        # grad_w = grad_out.T @ x
        # [n,k] = [m, n] * [m, k]
        grad_b = gemm_fp8_blockwise_impl(
            grad_out_fp8_col,
            a_fp8_col,
            grad_out_scales_col,
            a_scales_col,
            out_dtype=grad_out.dtype,
            scale_group_size_m=1,
            scale_group_size_n=1,
            scale_group_size_k=block_size,
            trans_a=True,
            trans_b=False,
        )

        grad_a = grad_a.view(*batch_shape, M, -1)
        return grad_a, grad_b, None, None, None, None


# TODO: weight support Float8Tensor
def gemm_fp8_blockwise(
    a,
    b,
    trans_a: bool = False,
    trans_b: bool = False,
    out_dtype: Union[torch.dtype, None] = None,
    config: Optional[Float8QuantConfig] = None,
):
    """
    Blockwise GEMM using FP8 quantization.

    This function applies blockwise FP8 quantization to both the input `a` and the `b`,
    performs matrix multiplication in the FP8 domain, and returns the result in the original
    input precision (e.g., bf16 or fp16).

    Args:
        a (torch.Tensor): Input tensor of shape [M, K] (if trans_a=False), typically in bf16 or fp16.
            Blockwise quantized with shape [1, blocksize].
        b (torch.Tensor): Weight tensor of shape [K, N] (if trans_b=False), typically in bf16 or fp16.
            Blockwise quantized with shape [blocksize, blocksize].
        trans_a (bool): Default: False.
        trans_b (bool): Default: False.
        out_dtype (torch.dtype, optional): Output dtype. If None, inferred via `torch.result_type(a, b)`.
        config (Float8QuantConfig, optional): Quantization configuration controlling FP8 dtype,
            scaling strategy, granularity, and block size. If None, a default config is used.

    Returns:
        torch.Tensor: Output tensor of shape [M, N] (assuming trans_a=False and trans_b=False),
        in the same dtype as `a` unless `out_dtype` is explicitly set.

    Example:
        >>> a = torch.randn(512, 1024, dtype=torch.bfloat16).cuda()
        >>> b = torch.randn(768, 1024, dtype=torch.bfloat16).cuda()
        >>> y = gemm_fp8_blockwise(a, b, trans_a=False, trans_b=True)
    """

    if out_dtype is None:
        out_dtype = torch.result_type(a, b)

    if config is None:
        config = Float8QuantConfig(granularity=ScalingGranularity.BLOCKWISE, block_size=128)

    return BlockwiseFP8GemmFunction.apply(a, b, trans_a, trans_b, out_dtype, config)


@torch.compile
def calc_scale_and_scale_inv(x: torch.Tensor, fp8_max: float):
    amax = x.abs().amax()
    scale = torch.full([1], fill_value=fp8_max, dtype=torch.float32, device=x.device) / amax
    scale_inv = 1.0 / scale

    return scale, scale_inv


@torch.compile
def quantize_fp8_rowwise(x: torch.Tensor, dtype: torch.dtype, dim: int = -1, eps: float = 1e-12):
    fp8_max = torch.finfo(dtype).max
    max_abs = torch.amax(x.to(torch.float32).abs(), dim=dim, keepdim=True)
    scale_inv = torch.clamp(max_abs / fp8_max, min=eps)
    scale = 1.0 / scale_inv
    x_fp8 = x * scale
    return x_fp8.to(dtype), scale_inv.view(-1).to(torch.float)


class FP8GemmFunction(torch.autograd.Function):

    @staticmethod
    def get_fp8_dtype(format: Format, is_fwd_stage: bool):
        if format == Format.E4M3:
            return float8_e4m3
        elif format == Format.E5M2:
            return float8_e5m2
        elif format == Format.HYBRID:
            return float8_e4m3 if is_fwd_stage else float8_e5m2
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        a_dtype = FP8GemmFunction.get_fp8_dtype(config.format, True)
        b_dtype = FP8GemmFunction.get_fp8_dtype(config.format, True)

        a_t_fp8, b_t_fp8 = None, None
        a_t_scale_inv, b_t_scale_inv = None, None
        if config.granularity == ScalingGranularity.TENSORWISE:
            a_fp8, a_scale_inv = quantize_fp8(a, a_dtype, config.granularity)
            b_fp8, b_scale_inv = quantize_fp8(b, b_dtype, config.granularity)
        # elif config.granularity == ScalingGranularity.ROWWISE:
        #     assert (
        #         trans_a == False and trans_b == True
        #     ), f"{ScalingGranularity.ROWWISE} only support NT layout."
        #     assert config.format == Format.E4M3, f"{ScalingGranularity.ROWWISE} only support E4M3 format."
        #     assert (
        #         out_dtype == torch.bfloat16
        #     ), f"{ScalingGranularity.ROWWISE} only support bfloat16 as out_dtype."

        #     a_fp8, a_scale_inv = quantize_fp8_rowwise(a, a_dtype, dim=(-1))
        #     b_fp8, b_scale_inv = quantize_fp8_rowwise(b, b_dtype, dim=(-1))

        #     a_t_fp8, a_t_scale_inv = quantize_fp8_rowwise(a.transpose(0, 1).contiguous(), a_dtype, dim=(-1))
        #     b_t_fp8, b_t_scale_inv = quantize_fp8_rowwise(b.transpose(0, 1).contiguous(), b_dtype, dim=(-1))
        else:
            raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")

        # NN
        out = gemm_fp8_impl(a_fp8, a_scale_inv, trans_a, b_fp8, b_scale_inv, trans_b, out_dtype, False)

        ctx.save_for_backward(
            a_fp8, a_scale_inv, b_fp8, b_scale_inv, a_t_fp8, a_t_scale_inv, b_t_fp8, b_t_scale_inv
        )
        ctx.trans_a = trans_a
        ctx.trans_b = trans_b
        ctx.out_dtype = out_dtype
        ctx.config = config

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (a_fp8, a_scale_inv, b_fp8, b_scale_inv, a_t_fp8, a_t_scale_inv, b_t_fp8, b_t_scale_inv) = (
            ctx.saved_tensors
        )
        grad_out_dtype = FP8GemmFunction.get_fp8_dtype(ctx.config.format, False)

        if ctx.config.granularity == ScalingGranularity.TENSORWISE:
            grad_out_fp8, grad_out_scale_inv = quantize_fp8(grad_out, grad_out_dtype, ctx.config.granularity)

            # NT,  change to use NT layout for high performance in hipblaslt gemm
            a_grad = gemm_fp8_impl(
                grad_out_fp8,
                grad_out_scale_inv,
                False,
                b_fp8.transpose(0, 1).contiguous(),
                b_scale_inv,
                True,
                ctx.out_dtype,
                ctx.trans_a,
            )

            # NT, Use NT layout for high performance in hipblaslt gemm
            b_grad = gemm_fp8_impl(
                grad_out_fp8.transpose(0, 1).contiguous(),
                grad_out_scale_inv,
                False, # False
                a_fp8.transpose(0, 1).contiguous(),
                a_scale_inv,
                True, # True
                ctx.out_dtype,
                False
            )
        # elif ctx.config.granularity == ScalingGranularity.ROWWISE:
        #     # NOTE: rowwise only support NT layout.
        #     grad_out_fp8, grad_out_scale_inv = quantize_fp8_rowwise(grad_out, grad_out_dtype, dim=(-1))
        #     grad_out_t_fp8, grad_out_t_scale_inv = quantize_fp8_rowwise(
        #         grad_out.transpose(0, 1).contiguous(), grad_out_dtype, dim=(-1)
        #     )

        #     # NT
        #     a_grad = gemm_fp8_impl(
        #         grad_out_fp8, grad_out_scale_inv, False, b_t_fp8, b_t_scale_inv, True, ctx.out_dtype, False
        #     )

        #     # TN
        #     b_grad = gemm_fp8_impl(
        #         a_t_fp8, a_t_scale_inv, False, grad_out_t_fp8, grad_out_t_scale_inv, True, ctx.out_dtype, True
        #     )
        else:
            raise ValueError(f"Unsupported FP8 ScalingGranularity: {ctx.config.granularity}")

        return (a_grad, b_grad, None, None, None, None)


def gemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float8QuantConfig, None] = None,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2, "Only 2D tensors are supported"

    if out_dtype is None:
        out_dtype = torch.result_type(a, b)

    if config is None:
        config = Float8QuantConfig()

    args = (a, b, trans_a, trans_b, out_dtype, config)

    return FP8GemmFunction.apply(*args)
