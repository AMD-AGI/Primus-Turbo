import torch


def grouped_gemm_fp8_blockwise_nt_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int = 128,
):
    pass
