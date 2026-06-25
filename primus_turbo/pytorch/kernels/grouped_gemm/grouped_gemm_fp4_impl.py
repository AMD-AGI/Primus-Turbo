###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP4 grouped GEMM dispatch (Triton-only, gfx950).

Thin ``torch.library`` custom-op wrappers around the Triton MXFP4 grouped
kernels. Unlike the FP8 path (which fans out to CK / hipBLASLt / FlyDSL), MXFP4
grouped GEMM currently has a single Triton block-scaled backend, so the ops call
it directly. The forward op over-allocates the output to the (padded) input rows
and the caller slices ``[:total_m]``; ``group_offs_out`` packs each group tight.
"""

import torch

from primus_turbo.pytorch.core.low_precision import float4_e2m1fn_x2
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp4_kernel import (
    grouped_gemm_mxfp4_triton_kernel,
    grouped_gemm_mxfp4_variable_k_triton_kernel,
)

_torch_custom_op_wrapper = torch.library.custom_op


@_torch_custom_op_wrapper("primus_turbo::grouped_gemm_fp4_impl", mutates_args=(), device_types="cuda")
def grouped_gemm_fp4_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_offs: torch.Tensor,
    N: int,
    K: int,
    num_cu: int | None,
    out_dtype: torch.dtype,
    group_offs_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward (NT): C[g] = A[g] @ B[g]^T.

    a (total_M, K/2) fp4, a_scales (total_M, K/32) e8m0,
    b (G, N, K/2) fp4, b_scales (G, N, K/32) e8m0. K is logical.
    """
    return grouped_gemm_mxfp4_triton_kernel(
        a,
        a_scales,
        b,
        b_scales,
        group_offs,
        N,
        K,
        group_offs_out=group_offs_out,
        out_dtype=out_dtype,
        num_cu=num_cu,
    )


@grouped_gemm_fp4_impl.register_fake
def grouped_gemm_fp4_impl_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_offs: torch.Tensor,
    N: int,
    K: int,
    num_cu: int | None,
    out_dtype: torch.dtype,
    group_offs_out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype == float4_e2m1fn_x2, f"a must be fp4, got {a.dtype}"
    assert b.dtype == float4_e2m1fn_x2, f"b must be fp4, got {b.dtype}"
    assert out_dtype in (torch.float16, torch.bfloat16)
    # Output over-allocated to padded input rows; caller slices [:total_m].
    return torch.empty((a.shape[0], N), device=a.device, dtype=out_dtype)


@_torch_custom_op_wrapper(
    "primus_turbo::grouped_gemm_fp4_variable_k_impl", mutates_args=(), device_types="cuda"
)
def grouped_gemm_fp4_variable_k_impl(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs_scales: torch.Tensor,
    group_offs: torch.Tensor,
    OUT_M: int,
    OUT_N: int,
    G: int,
    num_cu: int | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Backward wgrad: C[g] (OUT_M, OUT_N) = lhs[:,g] @ rhs[:,g]^T, reduction over M_g.

    lhs (OUT_M, M_total/2) fp4, rhs (OUT_N, M_total/2) fp4; scales (.., M_total/32) e8m0.
    group_offs: padded per-group offsets along M (each M_g multiple of 128).
    """
    return grouped_gemm_mxfp4_variable_k_triton_kernel(
        lhs,
        lhs_scales,
        rhs,
        rhs_scales,
        group_offs,
        OUT_M,
        OUT_N,
        G,
        out_dtype=out_dtype,
        num_cu=num_cu,
    )


@grouped_gemm_fp4_variable_k_impl.register_fake
def grouped_gemm_fp4_variable_k_impl_meta(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs_scales: torch.Tensor,
    group_offs: torch.Tensor,
    OUT_M: int,
    OUT_N: int,
    G: int,
    num_cu: int | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    assert lhs.dim() == 2 and rhs.dim() == 2
    assert lhs.dtype == float4_e2m1fn_x2 and rhs.dtype == float4_e2m1fn_x2
    assert out_dtype in (torch.float16, torch.bfloat16)
    return torch.empty((G, OUT_M, OUT_N), device=lhs.device, dtype=out_dtype)
