import torch

from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import (
    gemm_fp8_blockwise_nn_impl,
    gemm_fp8_blockwise_nt_impl,
    gemm_fp8_blockwise_tn_impl,
    quant_fp8_blockwise_for_act_grad_impl,
    quant_fp8_blockwise_for_weight_impl,
)
from primus_turbo.pytorch.kernels.quantize import quant_fp8_blockwise_impl

__all__ = ["gemm_fp8_blockwise"]


class BlockwiseFP8GemmFunction(torch.autograd.Function):
    """
    Autograd function for FP8 blockwise GEMM.

    This class implements the forward and backward pass for a GEMM using blockwise FP8 quantization.
    It is not intended to be used directly — use `gemm_fp8_blockwise` instead.

    Forward:
        - Quantize input `x` row-wise into FP8 blockwise.
        - Quantize weight `weight` blockwise.
        - Perform matrix multiplication: Y = x @ weight.T (NT layout).
        - Output result in high precision (same dtype as `x`).

    Backward Pass:
        - Quantize `grad_out` both row-wise and column-wise for dgrad/wgrad respectively.
        - Re-quantize input `x` column-wise for use in wgrad.
        - Compute:
            * grad_x = grad_out @ weight      (NN layout)
            * grad_w = grad_out.T @ x         (TN layout)
        - Both gradients are returned in high precision.

    Inputs:
        x (Tensor): Input tensor of shape [M, K], typically bf16/fp16.
        weight (Tensor): Weight tensor of shape [N, K], typically bf16/fp16.
        block_size (int): Block size for quantization. Default: 128.
        dtype (torch.dtype): FP8 quantization dtype. Default: float8_e4m3fnuz.

    Returns:
        out (Tensor): Output tensor of shape [M, N], same dtype as `x`.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        block_size: int = 128,
        dtype=torch.float8_e4m3fnuz,
    ):
        # Quantize input activation (row): shape [M, K] → FP8
        x_fp8_row, x_scales_row = quant_fp8_blockwise_impl(x, dtype, axis=1, block_size=block_size)
        # Quantize weight blockwise: shape [N, K] → FP8
        w_fp8, w_scales = quant_fp8_blockwise_for_weight_impl(weight, dtype, block_size)

        # Perform NT GEMM in quantized domain:
        # out = x_fp8_row @ w_fp8.T → shape [M, N]
        out = gemm_fp8_blockwise_nt_impl(
            x_fp8_row,
            w_fp8,
            x_scales_row,
            w_scales,
            out_dtype=x.dtype,
            block_size=block_size,
        )

        # Save tensors for backward
        ctx.save_for_backward(x, w_fp8, w_scales)
        ctx.block_size = block_size
        ctx.dtype = dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_fp8, w_scales = ctx.saved_tensors
        block_size = ctx.block_size
        dtype = ctx.dtype

        # Quantize grad_out in both row-wise and column-wise directions:
        # - row-wise: for dgrad (grad_x)
        # - col-wise: for wgrad (grad_w)
        grad_out_fp8_row, grad_out_scales_row, grad_out_fp8_col, grad_out_scales_col = (
            quant_fp8_blockwise_for_act_grad_impl(grad_out, dtype, block_size)
        )

        # TODO: dequant + quant kernel
        x_fp8_col, x_scales_col = quant_fp8_blockwise_impl(x, dtype, axis=0, block_size=block_size)

        # DGrad NN:
        # grad_x = grad_out @ weight
        # [m, k] = [m, n] * [n, k]
        grad_x = gemm_fp8_blockwise_nn_impl(
            grad_out_fp8_row,
            w_fp8,
            grad_out_scales_row,
            w_scales,
            out_dtype=grad_out.dtype,
            block_size=block_size,
        )

        # WGrad TN:
        # grad_w = grad_out.T @ x
        # [n,k] = [m, n] * [m, k]
        grad_w = gemm_fp8_blockwise_tn_impl(
            grad_out_fp8_col,
            x_fp8_col,
            grad_out_scales_col,
            x_scales_col,
            out_dtype=grad_out.dtype,
            block_size=block_size,
        )
        return grad_x, grad_w, None, None


def gemm_fp8_blockwise(x, weight, block_size=128, dtype=torch.float8_e4m3fnuz):
    """
    Blockwise GEMM using FP8 quantization.

    This function applies blockwise FP8 quantization to both the input `x` and the `weight`,
    performs matrix multiplication in the FP8 domain, and output the result in a higher
    precision (e.g., bf16 or fp16).

    Args:
        x (torch.Tensor): Input tensor of shape [M, K], typically in bf16 or fp16.
        weight (torch.Tensor): Weight tensor of shape [N, K], typically in bf16 or fp16.
        block_size (int): Block size used for quantization. Default: 128.
        dtype (torch.dtype): FP8 dtype to use for quantization. Default: torch.float8_e4m3fnuz.

    Returns:
        torch.Tensor: Result tensor of shape [M, N], in the same dtype as input.

    Example:
        >>> x = torch.randn(512, 1024, dtype=torch.bfloat16).cuda()
        >>> w = torch.randn(768, 1024, dtype=torch.bfloat16).cuda()
        >>> y = gemm_fp8_blockwise(x, w)
    """
    return BlockwiseFP8GemmFunction.apply(x, weight, block_size, dtype)
