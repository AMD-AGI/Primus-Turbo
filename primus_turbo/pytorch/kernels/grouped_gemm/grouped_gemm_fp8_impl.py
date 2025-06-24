import torch
import triton

from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
    compute_m_num_tiles_indptr,
    grouped_gemm_fp8_blockwise_kernel,
    grouped_gemm_variable_k_fp8_blockwise_kernel,
)


def grouped_gemm_fp8_blockwise_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    batch_size: int,
    seg_indptr: torch.Tensor,  # [B+1,] int64
    out_dtype: torch.dtype,
    scale_group_size_m: int,
    scale_group_size_n: int,
    scale_group_size_k: int,
    transA: bool,
    transB: bool,
):
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a_scales.dim() == 2, f"a scales must be 2D, got {a_scales.shape}"
    assert b_scales.dim() == 3, f"b scales must be 3D, got {b_scales.shape}"

    a_view = a.transpose(-1, -2) if transA else a
    a_scales_view = a_scales.transpose(-1, -2) if transA else a_scales
    b_view = b.transpose(-1, -2) if transB else b
    b_scales_view = b_scales.transpose(-1, -2) if transB else b_scales

    M, K = a_view.shape
    B, KB, N = b_view.shape
    assert K == KB, f"K mismatch: K={K}, KB={KB}"
    assert B == batch_size

    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": scale_group_size_k,
    }

    c = torch.empty(M, N, dtype=out_dtype, device=a.device)

    m_num_tiles_indptr = torch.zeros(batch_size + 1, device=a.device, dtype=torch.int64)
    compute_m_num_tiles_indptr[(1,)](m_num_tiles_indptr, seg_indptr, batch_size, config["BLOCK_SIZE_M"])

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) + batch_size,
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    grouped_gemm_fp8_blockwise_kernel[grid](
        a_view,
        b_view,
        c,
        a_scales_view,
        b_scales_view,
        batch_size,
        N,
        K,
        seg_indptr,
        m_num_tiles_indptr,
        a_view.stride(0),
        a_view.stride(1),
        b_view.stride(0),
        b_view.stride(1),
        b_view.stride(2),
        c.stride(0),
        c.stride(1),
        a_scales_view.stride(0),
        a_scales_view.stride(1),
        b_scales_view.stride(0),
        b_scales_view.stride(1),
        b_scales_view.stride(2),
        scale_group_size_m,
        scale_group_size_n,
        scale_group_size_k,
        **config,
    )
    return c


def grouped_gemm_variable_k_fp8_blockwise_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    batch_size: int,
    seg_indptr: torch.Tensor,  # [B+1,] int64
    scales_seg_indptr: torch.Tensor,
    out_dtype: torch.dtype,
    scale_group_size_m: int,
    scale_group_size_n: int,
    scale_group_size_k: int,
    transA: bool,
    transB: bool,
):
    assert transA == True and transB == False, "Only transA=True and transB=False are supported."
    assert (
        seg_indptr.shape[0] == batch_size + 1
    ), f"Expected seg_indptr shape [{batch_size + 1}], got {seg_indptr.shape}"
    assert (
        scales_seg_indptr.shape[0] == batch_size + 1
    ), f"Expected scales_seg_indptr shape [{batch_size + 1}], got {scales_seg_indptr.shape}"

    assert (
        scale_group_size_m == 1 and scale_group_size_n == 1
    ), f"Only scale_group_size_m == 1 and scale_group_size_n == 1 are supported, got {scale_group_size_m}, {scale_group_size_n}"

    a_view = a.transpose(-1, -2) if transA else a
    a_scales_view = a_scales.transpose(-1, -2) if transA else a_scales
    b_view = b.transpose(-1, -2) if transB else b
    b_scales_view = b_scales.transpose(-1, -2) if transB else b_scales

    M, K = a_view.shape
    KB, N = b_view.shape
    assert K == KB, f"K mismatch: KA={K}, KB={KB}"

    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": scale_group_size_k,
    }

    c = torch.empty(batch_size, M, N, dtype=out_dtype, device=a.device)

    grid = lambda META: (
        batch_size,
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    grouped_gemm_variable_k_fp8_blockwise_kernel[grid](
        a_view,
        b_view,
        c,
        a_scales_view,
        b_scales_view,
        batch_size,
        M,
        N,
        K,
        seg_indptr,
        scales_seg_indptr,
        a_view.stride(0),
        a_view.stride(1),
        b_view.stride(0),
        b_view.stride(1),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        a_scales_view.stride(0),
        a_scales_view.stride(1),
        b_scales_view.stride(0),
        b_scales_view.stride(1),
        scale_group_size_m,
        scale_group_size_n,
        scale_group_size_k,
        **config,
    )
    return c
