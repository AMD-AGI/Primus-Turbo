import torch

from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import (
    gemm_fp8_blockwise_nn_impl,
    gemm_fp8_blockwise_nt_impl,
    gemm_fp8_blockwise_tn_impl,
    quant_fp8_blockwise_for_weight_impl,
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
        x: torch.Tensor,  # [M, K]
        weight: torch.Tensor,  # [B, N, K]
        seg_lens: torch.Tensor,  # [B,] int64
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
        out = []
        for bs in range(batch_size):
            out_local = gemm_fp8_blockwise_nt_impl(
                x_fp8_row[seg_indptr[bs] : seg_indptr[bs + 1], :],
                w_fp8[bs, :, :],
                x_scales_row[seg_indptr[bs] : seg_indptr[bs + 1], :],
                w_scales[bs, :, :],
                out_dtype=x.dtype,
                block_size=block_size,
            )
            out.append(out_local)

        ctx.save_for_backward(x, w_fp8, w_scales, seg_lens, seg_indptr)
        ctx.batch_size = batch_size
        ctx.block_size = block_size
        ctx.dtype = dtype

        return torch.cat(out)

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
        grad_x = []
        for bs in range(batch_size):
            grad_x_local = gemm_fp8_blockwise_nn_impl(
                grad_out_fp8_row[seg_indptr[bs] : seg_indptr[bs + 1], :],
                w_fp8[bs, :, :],
                grad_out_scales_row[seg_indptr[bs] : seg_indptr[bs + 1], :],
                w_scales[bs, :, :],
                out_dtype=grad_out.dtype,
                block_size=block_size,
            )
            grad_x.append(grad_x_local)

        # TODO: Opt
        # WGrad TN
        # grad_w = grad_out.T @ x
        # [n,k] = [m, n] * [m, k]
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

        grad_w = []
        for bs in range(batch_size):
            grad_w_local = gemm_fp8_blockwise_tn_impl(
                grad_out_fp8_col[seg_indptr[bs] : seg_indptr[bs + 1], :],
                x_fp8_col[seg_indptr[bs] : seg_indptr[bs + 1], :],
                grad_out_scales_col[act_scales_col_seg_indptr[bs] : act_scales_col_seg_indptr[bs + 1], :],
                x_scales_col[act_scales_col_seg_indptr[bs] : act_scales_col_seg_indptr[bs + 1], :],
                out_dtype=grad_out.dtype,
                block_size=block_size,
            )
            grad_w.append(grad_w_local)

        return (
            torch.cat(grad_x),
            torch.stack(grad_w, dim=0),
            None,
            None,
            None,
            None,
            None,
        )


def grouped_gemm_fp8_blockwise(
    x: torch.Tensor,  # [M, K]
    weight: torch.Tensor,  # [B, N, K]
    seg_lens: torch.Tensor,  # [B,] int64
    block_size: int = 128,
    dtype=torch.float8_e4m3fnuz,
):
    return BlockwiseFP8GroupedGemmFunc.apply(x, weight, seg_lens, block_size, dtype)


# TODO grouped_gemm_fp8 tensorwise/rowwise
