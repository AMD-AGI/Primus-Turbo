###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Tuple

import aiter
import torch

_torch_custom_op_wrapper = torch.library.custom_op

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float4_e2m1fn_x2


def ceil_div(a, b):
    return (a + b - 1) // b


def get_gemm_logical_shape(
    a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> Tuple[int, int, int]:
    assert (
        a.ndim == 2 and b.ndim == 2
    ), f"Expected both a and b to be 2D tensors, but got a.ndim={a.ndim}, b.ndim={b.ndim}"
    M = a.shape[1] if trans_a else a.shape[0]
    Ka = a.shape[0] if trans_a else a.shape[1]
    Kb = b.shape[1] if trans_b else b.shape[0]
    N = b.shape[0] if trans_b else b.shape[1]
    assert Ka == Kb, f"GEMM K mismatch: a has K={Ka}, b has K={Kb}"
    return M, N, Ka


_COMMON_SUPPORTED_DTYPES = (
    (float4_e2m1fn_x2, float4_e2m1fn_x2, torch.float16),
    (float4_e2m1fn_x2, float4_e2m1fn_x2, torch.bfloat16),
)


class GEMMFP4HipBLASLtBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.MX_BLOCKWISE,
    }

    # (a_dtype, b_dtype, c_dtype)
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES)

    # (trans_a, trans_b, trans_c)
    SUPPORTED_LAYOUTS = ((False, True, False),)

    HIPBLASLT_M_MULTIPLE = 16
    HIPBLASLT_N_MULTIPLE = 16
    HIPBLASLT_K_MULTIPLE = 128

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
    ) -> bool:
        supported = True
        # check ScalingGranularity
        supported &= granularity in GEMMFP4HipBLASLtBackend.SUPPORTED_GRANULARITIES
        # check dtype
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP4HipBLASLtBackend.SUPPORTED_DTYPES
        supported &= (trans_a, trans_b, trans_c) in GEMMFP4HipBLASLtBackend.SUPPORTED_LAYOUTS

        # check dimension
        supported &= (
            a.size(0) % GEMMFP4HipBLASLtBackend.HIPBLASLT_M_MULTIPLE == 0
            and b.size(0) % GEMMFP4HipBLASLtBackend.HIPBLASLT_N_MULTIPLE == 0
        )

        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
    ):
        # TODO(ruibin): Add padding
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp4(
            a, a_scale_inv, b, b_scale_inv, out_dtype, trans_a, trans_b, trans_c, granularity.name
        )


def enable_preshuffle() -> bool:
    # NOTE: When auto-tune backend is disabled and gemm backend is AITER, we enable preshuffle for better performance.
    is_aiter_backend = (
        GlobalBackendManager.get_gemm_backend(PrecisionType.FP4) == BackendType.AITER
        and GlobalBackendManager.auto_tune_enabled() is False
    )

    return is_aiter_backend


class GEMMFP4AITERBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.MX_BLOCKWISE,
    }

    # (a_dtype, b_dtype, c_dtype)
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES)

    # (trans_a, trans_b, trans_c)
    SUPPORTED_LAYOUTS = ((False, True, False),)

    AITER_FP4GEMM_M_MULTIPLE = 16
    AITER_FP4GEMM_N_MULTIPLE = 16

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
    ) -> bool:
        supported = True
        # check ScalingGranularity
        supported &= granularity in GEMMFP4AITERBackend.SUPPORTED_GRANULARITIES
        # check dtype
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP4AITERBackend.SUPPORTED_DTYPES
        supported &= (trans_a, trans_b, trans_c) in GEMMFP4AITERBackend.SUPPORTED_LAYOUTS

        # check dimension
        supported &= (
            a.size(0) % GEMMFP4AITERBackend.AITER_FP4GEMM_M_MULTIPLE == 0
            and b.size(0) % GEMMFP4AITERBackend.AITER_FP4GEMM_N_MULTIPLE == 0
        )

        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
    ):
        if enable_preshuffle():
            # NOTE: When enable preshuffle, the input tensor is already shuffled.
            return aiter.gemm_a4w4(a, b, a_scale_inv, b_scale_inv, dtype=out_dtype, bpreshuffle=True)

        a_scale_inv_shuffled = torch.ops.primus_turbo_cpp_extension.shuffle_scale(a_scale_inv, [16, 16])
        b_scale_inv_shuffled = torch.ops.primus_turbo_cpp_extension.shuffle_scale(b_scale_inv, [16, 16])
        b_shuffled = torch.ops.primus_turbo_cpp_extension.shuffle_weight(b, [16, 16])
        return aiter.gemm_a4w4(
            a, b_shuffled, a_scale_inv_shuffled, b_scale_inv_shuffled, dtype=out_dtype, bpreshuffle=True
        )


_GEMM_FP4_BACKENDS = {
    BackendType.AITER: BackendEntry(GEMMFP4AITERBackend),
    BackendType.HIPBLASLT: BackendEntry(GEMMFP4HipBLASLtBackend),
}


class GEMMFP4KernelDispatcher(AutoKernelDispatcher):
    _backends = _GEMM_FP4_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, a, b, trans_a, trans_b, trans_c, out_dtype, granularity, **kwargs):
        m, n, k = get_gemm_logical_shape(a, b, trans_a, trans_b)
        return (m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c, granularity)


@_torch_custom_op_wrapper("primus_turbo::gemm_fp4_impl", mutates_args=(), device_types="cuda")
def gemm_fp4_impl(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    granularity: int,
    default_backend: int,
) -> torch.Tensor:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_gemm_backend(PrecisionType.FP4)
    granularity_enum = ScalingGranularity(granularity)

    kwargs = dict(
        a=a,
        b=b,
        a_scale_inv=a_scale_inv,
        b_scale_inv=b_scale_inv,
        out_dtype=out_dtype,
        trans_a=trans_a,
        trans_b=trans_b,
        trans_c=trans_c,
        granularity=granularity_enum,
    )

    return GEMMFP4KernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


@gemm_fp4_impl.register_fake
def gemm_fp4_impl_meta(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    granularity: int,
    default_backend: int,
) -> torch.Tensor:
    m, n, _ = get_gemm_logical_shape(a, b, trans_a, trans_b)
    if trans_c:
        m, n = n, m
    return torch.empty(m, n, dtype=out_dtype, device=a.device)
