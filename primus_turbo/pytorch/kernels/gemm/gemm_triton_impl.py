import torch
import triton
from torch.library import triton_op, wrap_triton

from primus_turbo.triton.gemm.gemm_kernel import gemm_triton_kernel


def gemm_triton_imlp(a: torch.Tensor, b: torch.Tensor, layout: str = "NN") -> torch.Tensor:
    return torch.ops.primus_turbo.gemm_triton.default(a, b, layout)


@triton_op("primus_turbo::gemm_triton", mutates_args={})
def gemm_triton(a: torch.Tensor, b: torch.Tensor, layout: str = "NN") -> torch.Tensor:
    assert layout in ["NN", "NT", "TN"], f"Unsupported layout: {layout}"
    if layout == "NN":
        a_mat, b_mat = a, b
    elif layout == "NT":
        a_mat, b_mat = a, b.transpose(-1, -2)
    elif layout == "TN":
        a_mat, b_mat = a.transpose(-1, -2), b

    M, K = a_mat.shape
    K1, N = b_mat.shape
    assert K == K1
    out_dtype = torch.result_type(a, b)  # TODO
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    wrap_triton(gemm_triton_kernel)[grid](
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


@gemm_triton.register_fake
def gemm_triton_meta(a: torch.Tensor, b: torch.Tensor, layout: str = "NN") -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2, f"Expect 2D tensors, got {a.shape}, {b.shape}"
    if layout == "NN":
        m, k1 = a.shape
        k2, n = b.shape
    elif layout == "NT":
        m, k1 = a.shape
        n, k2 = b.shape
    elif layout == "TN":
        k1, m = a.shape
        k2, n = b.shape
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    assert k1 == k2, f"Incompatible matmul dims: k1={k1}, k2={k2}"
    out_dtype = torch.result_type(a, b)
    return torch.empty((m, n), device=a.device, dtype=out_dtype)
