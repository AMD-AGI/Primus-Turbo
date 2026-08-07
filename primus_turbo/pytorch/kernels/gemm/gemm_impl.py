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
        inplace_add_to_out: bool = False,
        **kwargs,
    ) -> bool:
        supported = True
        # This backend has no beta=1 accumulate epilogue.
        supported &= not inplace_add_to_out
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
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        beta = 1.0 if inplace_add_to_out else 0.0
        if inplace_add_to_out and trans_c:
            # C^T = (A @ B)^T = B^T @ A^T. A trans_c store writes `out` with swapped
            # strides, which makes the beta=1 read-back column-major; Triton stages
            # that through LDS (a 256x256 fp32 tile needs 256KB, over the 160KB
            # limit). Swapping the operands produces the exact same result with a
            # row-major, coalesced accumulate instead.
            return gemm_triton_kernel(b, a, not trans_b, not trans_a, out_dtype, False, beta=beta, out=out)
        return gemm_triton_kernel(a, b, trans_a, trans_b, out_dtype, trans_c, beta=beta, out=out)


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
    assert a.ndim == 2 and b.ndim == 2, (
        f"Expected both a and b to be 2D tensors, but got a.ndim={a.ndim}, b.ndim={b.ndim}"
    )
    M = a.shape[1] if trans_a else a.shape[0]
    N = b.shape[0] if trans_b else b.shape[1]
    if trans_c:
        M, N = N, M
    return torch.empty(M, N, dtype=out_dtype, device=a.device)


@_torch_custom_op_wrapper("primus_turbo::gemm_accum_impl", mutates_args={"out"}, device_types="cuda")
def gemm_accum_impl(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    out: torch.Tensor,
    default_backend: int,
) -> None:
    """BF16/FP16 GEMM that accumulates into ``out`` instead of returning.

    Computes ``out += op(A) @ op(B)``, folding the accumulation into the GEMM
    epilogue (beta=1) so the caller does not have to run a separate elementwise add
    over the whole weight-gradient buffer.

    Split out from :func:`gemm_impl` rather than added as a flag on it because a
    ``torch.library`` custom op may not return a tensor that aliases one of its
    inputs; this one mutates ``out`` and returns nothing.

    A backend without the beta=1 epilogue reports ``inplace_add_to_out`` as
    unsupported in ``can_handle``, so the dispatcher routes to one that has it rather
    than landing somewhere that would quietly ignore ``out``.
    """
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_gemm_backend(PrecisionType.BF16_FP16_FP32)

    kwargs = dict(
        a=a,
        trans_a=trans_a,
        b=b,
        trans_b=trans_b,
        out_dtype=out_dtype,
        trans_c=trans_c,
        inplace_add_to_out=True,
        out=out,
    )

    # The tuner benchmarks a backend by launching it repeatedly, so letting it tune on
    # the caller's buffer would accumulate the wgrad once per warmup and timing
    # iteration. Prime the cache on a scratch buffer first: the tune key ignores `out`,
    # so the dispatch below hits that cache and runs exactly once on the real buffer.
    # Zeroed, not empty -- beta=1 reads the buffer back and NaNs would skew the timings.
    if GlobalBackendManager.auto_tune_enabled() and not GEMMKernelDispatcher._is_graph_capturing():
        GEMMKernelDispatcher.tune(**{**kwargs, "out": torch.zeros_like(out)})

    GEMMKernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


@gemm_accum_impl.register_fake
def gemm_accum_impl_meta(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    out: torch.Tensor,
    default_backend: int,
) -> None:
    assert a.ndim == 2 and b.ndim == 2, (
        f"Expected both a and b to be 2D tensors, but got a.ndim={a.ndim}, b.ndim={b.ndim}"
    )
    M = a.shape[1] if trans_a else a.shape[0]
    N = b.shape[0] if trans_b else b.shape[1]
    if trans_c:
        M, N = N, M
    assert tuple(out.shape) == (M, N), f"out shape {tuple(out.shape)} must equal {(M, N)}"
    return None
