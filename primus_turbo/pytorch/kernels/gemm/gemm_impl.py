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
from primus_turbo.pytorch.kernels import hipkitten
from primus_turbo.triton.gemm.gemm_kernel import gemm_triton_kernel

_COMMON_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_HIPBLASLT_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _load_hipkitten_module():
    """Backward-compat shim for tests/pytorch/ops/test_gemm.py.

    The historical API (pre-refactor) lived at this module path and
    returned the loaded BF16 extension. The rule-based dispatcher
    refactor in commit 1970d91 moved the actual loader to
    :func:`primus_turbo.pytorch.kernels.hipkitten.load_bf16`, but the
    test file (which is FROZEN per project policy) still imports from
    here. We forward to the new loader so the import — and the tests'
    "skip on missing extension" guard — keep working.
    """
    return hipkitten.load_bf16().module


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


class GEMMHipKittenBackend(KernelBackend):

    @staticmethod
    def _logical_shape(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
    ) -> tuple[int, int, int, int]:
        m = a.shape[1] if trans_a else a.shape[0]
        k_a = a.shape[0] if trans_a else a.shape[1]
        n = b.shape[0] if trans_b else b.shape[1]
        k_b = b.shape[1] if trans_b else b.shape[0]
        return m, n, k_a, k_b

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
        # Hard constraints only — alignment, dtype, layout, device. No
        # shape-table or autotune-cache lookup. Aligned shapes always run;
        # the kernel handles them via the binding defaults (group_m=4,
        # num_xcds=8) selected by ``hipkitten.select_default_config``.
        if a.ndim != 2 or b.ndim != 2:
            return False
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or out_dtype != torch.bfloat16:
            return False
        if not a.is_cuda or not b.is_cuda or a.device != b.device:
            return False
        if not hipkitten.has_bf16():
            return False
        layout = hipkitten.layout_of(trans_a, trans_b)
        if layout is None:
            return False
        m, n, k_a, k_b = GEMMHipKittenBackend._logical_shape(a, trans_a, b, trans_b)
        if k_a != k_b:
            return False
        return hipkitten.aligned_for(m, n, k_a, "bf16")

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
        hk = hipkitten.load_bf16()
        m, n, k, _ = GEMMHipKittenBackend._logical_shape(a, trans_a, b, trans_b)
        layout = hipkitten.layout_of(trans_a, trans_b)
        if layout is None:
            raise ValueError("HipKitten BF16 backend supports RCR, RRR, and CRR layouts only.")

        # CRR + trans_c shortcut. The HK ``gemm_crr`` kernel computes
        # ``c[m, n] = first^T @ second``; the algebraic identity
        # ``(A^T @ B)^T == B^T @ A`` lets us produce the trans_c-transposed
        # output [n, m] directly by swapping (a, b). This avoids the
        # post-GEMM ``out.t().contiguous()`` copy — for the canonical
        # 4096x4096x4096 BF16 backward dB the saved copy is 32 MB / ~21 us
        # at 3 TB/s HBM, which is ~12 % of the kernel time. The CRR path
        # is the dominant trans_c=True hit on this branch (autograd dW
        # always lands on it for trans_b=True forwards).
        a_in = a if a.is_contiguous() else a.contiguous()
        b_in = b if b.is_contiguous() else b.contiguous()
        if trans_c and layout == "crr":
            out = torch.empty((n, m), dtype=out_dtype, device=a.device)
            cfg = hipkitten.select_default_config(n, m, k, layout, "bf16")
            hipkitten.dense_run(hk, cfg, b_in, a_in, out)
            return out

        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
        cfg = hipkitten.select_default_config(m, n, k, layout, "bf16")
        hipkitten.dense_run(hk, cfg, a_in, b_in, out)
        return out.t().contiguous() if trans_c else out


_GEMM_BACKENDS = {
    BackendType.HIPBLASLT: BackendEntry(GEMMHipBLASLtBackend),
    BackendType.HIPKITTEN: BackendEntry(GEMMHipKittenBackend, autotune=False),
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
    assert (
        a.ndim == 2 and b.ndim == 2
    ), f"Expected both a and b to be 2D tensors, but got a.ndim={a.ndim}, b.ndim={b.ndim}"
    M = a.shape[1] if trans_a else a.shape[0]
    N = b.shape[0] if trans_b else b.shape[1]
    if trans_c:
        M, N = N, M
    return torch.empty(M, N, dtype=out_dtype, device=a.device)
