###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

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
from primus_turbo.triton.gemm.gemm_kernel import gemm_triton_kernel

_COMMON_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_HIPBLASLT_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class GEMMHipBLASLtBackend(KernelBackend):

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.ndim == 2 and b.ndim == 2
        supported &= a.dtype in _HIPBLASLT_SUPPORTED_DTYPES and b.dtype in _HIPBLASLT_SUPPORTED_DTYPES
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        **kwargs,
    ) -> torch.Tensor:
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm(a, b, out_dtype, trans_a, trans_b, trans_c)


class GEMMTritonBackend(KernelBackend):

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.ndim == 2 and b.ndim == 2
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        **kwargs,
    ) -> torch.Tensor:
        return gemm_triton_kernel(a, b, trans_a, trans_b, out_dtype, trans_c)


_GEMM_BACKENDS = {
    BackendType.HIPBLASLT: BackendEntry(GEMMHipBLASLtBackend),
    BackendType.TRITON: BackendEntry(GEMMTritonBackend),
}


class GEMMKernelDispatcher(AutoKernelDispatcher):
    _backends = _GEMM_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, a, b, trans_a, trans_b, out_dtype, trans_c, **kwargs):
        M = a.shape[1] if trans_a else a.shape[0]
        Ka = a.shape[0] if trans_a else a.shape[1]
        N = b.shape[0] if trans_b else b.shape[1]
        return (M, N, Ka, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c)


def _shape_preferred_backend(a, trans_a, b, trans_b):
    """Select the statistically best backend based on (M, N, K) shape.

    Rules derived from MI300X benchmarks (hipBLASLt vs Triton, BF16):
      - Very large K (>=40000): Triton persistent kernel hides latency better
      - Very large N (>=65536) with M>=8192: Triton tile scheduling wins
    Only overrides the default when the expected gain is significant (>5%).
    """
    if a.dtype not in _COMMON_SUPPORTED_DTYPES:
        return None
    M = a.shape[1] if trans_a else a.shape[0]
    K = a.shape[0] if trans_a else a.shape[1]
    N = b.shape[0] if trans_b else b.shape[1]
    if K >= 40000:
        return BackendType.TRITON
    if N >= 65536 and M >= 8192:
        return BackendType.TRITON
    return None


@_torch_custom_op_wrapper("primus_turbo::gemm_impl", mutates_args=(), device_types="cuda")
def gemm_impl(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    default_backend: int,
) -> torch.Tensor:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_gemm_backend(PrecisionType.BF16_FP16_FP32)

    if user_backend_enum is None and not GlobalBackendManager.auto_tune_enabled():
        shape_hint = _shape_preferred_backend(a, trans_a, b, trans_b)
        if shape_hint is not None:
            default_backend_enum = shape_hint

    kwargs = dict(
        a=a,
        trans_a=trans_a,
        b=b,
        trans_b=trans_b,
        out_dtype=out_dtype,
        trans_c=trans_c,
    )

    return GEMMKernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


@gemm_impl.register_fake
def gemm_impl_meta(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    default_backend: int,
) -> torch.Tensor:
    assert (
        a.ndim == 2 and b.ndim == 2
    ), f"Expected both a and b to be 2D tensors, but got a.ndim={a.ndim}, b.ndim={b.ndim}"
    M = a.shape[1] if trans_a else a.shape[0]
    N = b.shape[0] if trans_b else b.shape[1]
    if trans_c:
        M, N = N, M
    return torch.empty(M, N, dtype=out_dtype, device=a.device)
