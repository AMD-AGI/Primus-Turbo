###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

_torch_custom_op_wrapper = torch.library.custom_op

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendChoice,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.core.utils import is_gfx1250
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
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.ndim == 2 and b.ndim == 2
        supported &= a.dtype in _HIPBLASLT_SUPPORTED_DTYPES and a.dtype == b.dtype

        d_dtype = out.dtype if (inplace_add_to_out and out is not None) else out_dtype
        supported &= d_dtype == a.dtype or d_dtype == torch.float32

        if inplace_add_to_out:
            supported &= out is not None and out.is_contiguous()

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
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm(
            a, b, out_dtype, trans_a, trans_b, trans_c, beta, out
        )


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
        return gemm_triton_kernel(a, b, trans_a, trans_b, out_dtype, trans_c, beta=beta, out=out)


# (trans_a, trans_b) -> the FlyDSL kernel's layout name, for out = op(a) @ op(b).
# trans_a picks A[M,K] or A[K,M]; trans_b picks B[K,N] or B[N,K]. The three
# reachable combinations are exactly the three GEMMs a Linear issues per
# training step -- forward, dgrad and wgrad. TT has no consumer and no kernel.
_FLYDSL_LAYOUTS = {
    (False, True): "nt",  # Y  = X  @ W^T
    (False, False): "nn",  # dX = dY @ W
    (True, False): "tn",  # dW = dY^T @ X
}


def _flydsl_call(a, trans_a, b, trans_b, trans_c):
    """Normalise a gemm_impl call to ``(a, b, layout)`` for the FlyDSL kernel.

    A transposed output is folded into the operands rather than rejected:
    ``(op(a) @ op(b))^T == op(b)^T @ op(a)^T``, so ``trans_c`` is the same call
    with the operands swapped and both flags flipped. That matters because the
    wgrad of a Linear arrives as ``trans_c=True`` (dW has to come out [N, K] to
    match the weight), and without this it could never reach this backend.

    Returns ``None`` when the combination has no layout here.
    """
    if trans_c:
        a, b, trans_a, trans_b = b, a, not trans_b, not trans_a
    layout = _FLYDSL_LAYOUTS.get((trans_a, trans_b))
    return None if layout is None else (a, b, layout)


def _flydsl_gemm():
    """Import the gfx1250 FlyDSL GEMM lazily.

    This module is imported on every arch, and importing FlyDSL pulls in MLIR.
    """
    from primus_turbo.flydsl.gemm.gemm_gfx1250_kernel import can_run, gemm_gfx1250

    return can_run, gemm_gfx1250


class GEMMFlyDSLBackend(KernelBackend):
    """gfx1250 WMMA GEMM. See primus_turbo/flydsl/gemm/gemm_gfx1250_kernel.py."""

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        if not is_gfx1250():
            return False
        if inplace_add_to_out and (out is None or not out.is_contiguous()):
            return False
        call = _flydsl_call(a, trans_a, b, trans_b, trans_c)
        if call is None:
            return False
        if out_dtype not in _COMMON_SUPPORTED_DTYPES:
            return False
        ka, kb, layout = call
        can_run, _ = _flydsl_gemm()
        # Every remaining constraint (dtype pairing, K divisibility, LDS budget)
        # lives with the kernel, so this cannot drift away from what it accepts.
        return can_run(ka, kb, layout=layout, out_dtype=out_dtype)

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
        ka, kb, layout = _flydsl_call(a, trans_a, b, trans_b, trans_c)
        _, gemm_gfx1250 = _flydsl_gemm()
        return gemm_gfx1250(
            ka,
            kb,
            out,
            layout=layout,
            out_dtype=out_dtype,
            accumulate=inplace_add_to_out,
        )


_GEMM_BACKENDS = {
    BackendType.HIPBLASLT: BackendEntry(GEMMHipBLASLtBackend),
    BackendType.TRITON: BackendEntry(GEMMTritonBackend),
    BackendType.FLYDSL: BackendEntry(GEMMFlyDSLBackend),
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
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_gemm_backend(PrecisionType.BF16_FP16_FP32)

    kwargs = dict(
        a=a,
        trans_a=trans_a,
        b=b,
        trans_b=trans_b,
        out_dtype=out_dtype,
        trans_c=trans_c,
    )

    return GEMMKernelDispatcher.dispatch(default_backend_choice, user_backend_choice, **kwargs)


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
    epilogue (beta=1)
    """
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_gemm_backend(PrecisionType.BF16_FP16_FP32)

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

    GEMMKernelDispatcher.dispatch(default_backend_choice, user_backend_choice, **kwargs)


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
