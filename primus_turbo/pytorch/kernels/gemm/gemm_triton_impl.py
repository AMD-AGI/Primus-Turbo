import torch
import triton

from primus_turbo.triton.gemm.gemm_kernel import gemm_triton_kernel


def gemm_triton_imlp(a: torch.Tensor, b: torch.Tensor, layout: str = "NN"):
    assert layout in ["NN", "NT", "TN"], f"Unsupported layout: {layout}"
    if layout == "NN":
        a_mat, b_mat = a, b
    elif layout == "NT":
        a_mat, b_mat = a, b.transpose(-1, -2)
    elif layout == "TN":
        a_mat, b_mat = a.transpose(-1, -2), b

    M, K = a_mat.shape
    K, N = b_mat.shape
    out_dtype = torch.result_type(a, b)  # TODO
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    gemm_triton_kernel[grid](
        a_mat,
        b_mat,
        out,
        M,
        N,
        K,
        a_mat.stride(0),
        a_mat.stride(1),
        b_mat.stride(0),
        b_mat.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out
