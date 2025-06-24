import torch

from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import (
    quant_fp8_blockwise_for_weight_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_fp8_blockwise_impl,
    grouped_gemm_variable_k_fp8_blockwise_impl,
)
from primus_turbo.pytorch.kernels.quantize import (
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
)

__all__ = ["grouped_gemm_fp8_blockwise"]


class BlockwiseFP8GroupedGemmFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        seg_lens: torch.Tensor,
        block_size: int = 128,
        dtype=torch.float8_e4m3fnuz,
    ):
        batch_size = seg_lens.size(0)
        assert batch_size == weight.size(0)
        seg_indptr = torch.cat([torch.tensor([0], device=seg_lens.device), seg_lens.cumsum(0)])

        # Quantize input activation (row): shape [M, K] → FP8
        x_fp8_row, x_scales_row = quant_fp8_blockwise_impl(x, dtype, axis=1, block_size=block_size)
        # Quantize weight blockwise: shape [B, N, K] → FP8
        w_fp8, w_scales = quant_fp8_blockwise_for_weight_impl(weight, dtype, block_size)

        # TODO: Opt
        out = grouped_gemm_fp8_blockwise_impl(
            x_fp8_row,
            w_fp8,
            x_scales_row,
            w_scales,
            batch_size,
            seg_indptr,
            out_dtype=x.dtype,
            scale_group_size_m=1,
            scale_group_size_n=block_size,
            scale_group_size_k=block_size,
            transA=False,
            transB=True,
        )

        ctx.save_for_backward(x, w_fp8, w_scales, seg_lens, seg_indptr)
        ctx.batch_size = batch_size
        ctx.block_size = block_size
        ctx.dtype = dtype

        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_fp8, w_scales, seg_lens, seg_indptr = ctx.saved_tensors
        batch_size = ctx.batch_size
        block_size = ctx.block_size
        dtype = ctx.dtype

        # quant grad_out
        grad_out_fp8_row, grad_out_scales_row = quant_fp8_blockwise_impl(
            grad_out, dtype, axis=1, block_size=block_size
        )

        # TODO: Opt
        # DGrad NN:
        grad_x = grouped_gemm_fp8_blockwise_impl(
            grad_out_fp8_row,
            w_fp8,
            grad_out_scales_row,
            w_scales,
            batch_size,
            seg_indptr,
            out_dtype=grad_out.dtype,
            scale_group_size_m=1,
            scale_group_size_n=block_size,
            scale_group_size_k=block_size,
            transA=False,
            transB=False,
        )

        # TODO: Opt
        # WGrad TN
        # grad_w = grad_out.T @ x
        # [b, n, k] = [m, n] * [m, k]
        act_scales_col_seg_lens = torch.ceil(seg_lens.float() / block_size).to(seg_lens.dtype)
        act_scales_col_seg_indptr = torch.cat(
            [
                torch.tensor([0], device=seg_lens.device),
                act_scales_col_seg_lens.cumsum(0),
            ]
        )

        x_fp8_col, x_scales_col = quant_fp8_blockwise_segment_m_impl(
            x,
            batch_size,
            seg_lens,
            seg_indptr,
            act_scales_col_seg_indptr,
            dtype,
            block_size,
        )
        grad_out_fp8_col, grad_out_scales_col = quant_fp8_blockwise_segment_m_impl(
            grad_out,
            batch_size,
            seg_lens,
            seg_indptr,
            act_scales_col_seg_indptr,
            dtype,
            block_size,
        )

        grad_w = grouped_gemm_variable_k_fp8_blockwise_impl(
            grad_out_fp8_col,
            x_fp8_col,
            grad_out_scales_col,
            x_scales_col,
            batch_size,
            seg_indptr,
            act_scales_col_seg_indptr,
            out_dtype=grad_out.dtype,
            scale_group_size_m=1,
            scale_group_size_n=1,
            scale_group_size_k=block_size,
            transA=True,
            transB=False,
        )

        return (
            grad_x,
            grad_w,
            None,
            None,
            None,
            None,
            None,
        )


def grouped_gemm_fp8_blockwise(
    x: torch.Tensor,
    weight: torch.Tensor,
    seg_lens: torch.Tensor,
    block_size: int = 128,
    dtype=torch.float8_e4m3fnuz,
):
    """
    Grouped GEMM with FP8 blockwise quantization.

    Args:
        x (torch.Tensor): Input tensor of shape [M, K], float16/bfloat16.
        weight (torch.Tensor): Weight tensor of shape [B, N, K], float16/bfloat16.
        seg_lens (torch.Tensor): Segment lengths of shape [B], int64. Sum should equal M.
        block_size (int): Block size for quantization. Default: 128.
        dtype (torch.dtype): FP8 type. Default: torch.float8_e4m3fnuz.

    Returns:
        torch.Tensor: Output tensor of shape [M, N], same dtype as x.

    Example:
        >>> x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
        >>> weight = torch.randn(4, 64, 128, device="cuda", dtype=torch.bfloat16)
        >>> seg_lens = torch.tensor([64, 64, 64, 64], dtype=torch.long, device="cuda")
        >>> out = grouped_gemm_fp8_blockwise(x, weight, seg_lens)
        >>> print(out.shape)  # torch.Size([256, 64])
    """
    return BlockwiseFP8GroupedGemmFunc.apply(x, weight, seg_lens, block_size, dtype)


# TODO grouped_gemm_fp8 tensorwise/rowwise
