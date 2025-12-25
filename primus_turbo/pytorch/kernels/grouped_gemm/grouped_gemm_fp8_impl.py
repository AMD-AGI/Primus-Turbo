###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
import triton

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    TuneCache,
)
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
    compute_m_num_tiles_indptr,
    grouped_gemm_fp8_blockwise_kernel,
    grouped_gemm_variable_k_fp8_blockwise_tn_kernel,
)

_COMMON_SUPPORTED_DTYPES = (
    (float8_e4m3, float8_e4m3, torch.float16),
    (float8_e4m3, float8_e4m3, torch.bfloat16),
    (float8_e5m2, float8_e5m2, torch.float16),
    (float8_e5m2, float8_e5m2, torch.bfloat16),
)


class GroupedGEMMFP8CKBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
    }

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES)

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= (a.dtype, b.dtype, out_dtype) in _COMMON_SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8CKBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
    ):
        return torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm_fp8(
            a,
            b,
            a_scales,
            b_scales,
            group_lens,
            group_offs,
            trans_a,
            trans_b,
            out_dtype,
            granularity.name,
            num_cu,
        )


class GroupedGEMMFP8KernelDispatcher(AutoKernelDispatcher):
    _backends = {
        BackendType.CK: GroupedGEMMFP8CKBackend,
    }
    _cache = TuneCache(1024)

    @classmethod
    def make_key(
        cls,
        a,
        b,
        a_scales,
        b_scales,
        group_lens,
        group_offs,
        trans_a,
        trans_b,
        out_dtype,
        granularity,
        num_cu,
    ):
        def get_grouped_gemm_logical_shape(a, b, trans_a, trans_b):
            bs = b.shape[0]
            m = a.shape[1] if trans_a else a.shape[0]
            n = b.shape[-2] if trans_b else b.shape[-1]
            k = a.shape[0] if trans_a else a.shape[1]
            return bs, m, n, k

        bs, m, n, k = get_grouped_gemm_logical_shape(a, b, trans_a, trans_b)
        # bs, m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c
        return (bs, m, n, k, a.dtype, b.dtype, a.dtype, trans_a, trans_b, False)


def grouped_gemm_fp8_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    num_cu: int | None,
    default_backend: int,
    maybe_pre_sync: bool = False,
) -> torch.Tensor:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_grouped_gemm_backend()

    kwargs = dict(
        a=a,
        b=b,
        a_scales=a_scales,
        b_scales=b_scales,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=trans_a,
        trans_b=trans_b,
        out_dtype=out_dtype,
        granularity=granularity,
        num_cu=num_cu,
    )

    return GroupedGEMMFP8KernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


def grouped_gemm_fp8_variable_k_csrc_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    granularity: ScalingGranularity,
    num_cu: int | None,
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 2, f"b must be 2D, got {b.shape}"
    assert a.dtype in [
        turbo.float8_e4m3,
        turbo.float8_e5m2,
    ], f"a must be float8, got {a.dtype}"
    assert b.dtype in [
        turbo.float8_e4m3,
        turbo.float8_e5m2,
    ], f"b must be float8, got {b.dtype}"
    assert trans_a == True and trans_b == False, "Only trans_a=True and trans_b=False are supported."

    return torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm_fp8_variable_k(
        a,
        b,
        a_scales,
        b_scales,
        group_lens,
        group_offs,
        trans_a,
        trans_b,
        out_dtype,
        granularity.name,
        num_cu,
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
    trans_a: bool,
    trans_b: bool,
):
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a_scales.dim() == 2, f"a scales must be 2D, got {a_scales.shape}"
    assert b_scales.dim() == 3, f"b scales must be 3D, got {b_scales.shape}"

    M = a.shape[1] if trans_a else a.shape[0]
    Ka = a.shape[0] if trans_a else a.shape[1]
    Kb = b.shape[-1] if trans_b else b.shape[-2]
    N = b.shape[-2] if trans_b else b.shape[-1]
    B = b.shape[0]

    assert Ka == Kb, f"K mismatch: Ka={Ka}, Kb={Kb}"
    assert B == batch_size

    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": scale_group_size_k,
        "num_stages": 2,
        "num_warps": 4,
    }

    c = torch.empty(M, N, dtype=out_dtype, device=a.device)

    m_num_tiles_indptr = torch.zeros(batch_size + 1, device=a.device, dtype=torch.int64)
    compute_m_num_tiles_indptr[(1,)](m_num_tiles_indptr, seg_indptr, batch_size, config["BLOCK_SIZE_M"])

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) + batch_size,
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    grouped_gemm_fp8_blockwise_kernel[grid](
        a,
        b,
        c,
        a_scales,
        b_scales,
        batch_size,
        M,
        N,
        Ka,
        seg_indptr,
        m_num_tiles_indptr,
        trans_a,
        trans_b,
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
    trans_a: bool,
    trans_b: bool,
):
    assert trans_a == True and trans_b == False, "Only trans_a=True and trans_b=False are supported."
    assert (
        seg_indptr.shape[0] == batch_size + 1
    ), f"Expected seg_indptr shape [{batch_size + 1}], got {seg_indptr.shape}"
    assert (
        scales_seg_indptr.shape[0] == batch_size + 1
    ), f"Expected scales_seg_indptr shape [{batch_size + 1}], got {scales_seg_indptr.shape}"

    assert (
        scale_group_size_m == 1 and scale_group_size_n == 1
    ), f"Only scale_group_size_m == 1 and scale_group_size_n == 1 are supported, got {scale_group_size_m}, {scale_group_size_n}"

    # a_view = a.transpose(-1, -2) if trans_a else a
    # a_scales_view = a_scales.transpose(-1, -2) if trans_a else a_scales
    # b_view = b.transpose(-1, -2) if trans_b else b
    # b_scales_view = b_scales.transpose(-1, -2) if trans_b else b_scales

    M = a.shape[1] if trans_a else a.shape[0]
    Ka = a.shape[0] if trans_a else a.shape[1]
    Kb = b.shape[-1] if trans_b else b.shape[-2]
    N = b.shape[-2] if trans_b else b.shape[-1]
    assert Ka == Kb, f"K mismatch: KA={Ka}, KB={Kb}"

    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": scale_group_size_k,
        "num_stages": 2,
        "num_warps": 4,
    }

    c = torch.empty(batch_size, M, N, dtype=out_dtype, device=a.device)

    grid = lambda META: (
        batch_size,
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    grouped_gemm_variable_k_fp8_blockwise_tn_kernel[grid](
        a,
        b,
        c,
        a_scales,
        b_scales,
        batch_size,
        M,
        N,
        Ka,
        seg_indptr,
        scales_seg_indptr,
        scale_group_size_m,
        scale_group_size_n,
        scale_group_size_k,
        **config,
    )
    return c


def grouped_gemm_compute_offs(group_lens: torch.Tensor) -> torch.Tensor:
    group_offs = torch.ops.primus_turbo_cpp_extension.grouped_gemm_compute_offs(group_lens)
    return group_offs
