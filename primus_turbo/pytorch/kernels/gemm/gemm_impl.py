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


def _hk_layout(trans_a: bool, trans_b: bool) -> str | None:
    """Map (trans_a, trans_b) to the HK dense layout id (rcr/rrr/crr).

    HK exposes three layouts: RCR (B is column-major / B^T), RRR (both
    row-major), CRR (A is column-major / A^T). Mirrors the resolution
    that lived in the deleted ``hipkitten.layout_of`` helper.
    """
    if not trans_a and trans_b:
        return "rcr"
    if not trans_a and not trans_b:
        return "rrr"
    if trans_a and not trans_b:
        return "crr"
    return None


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
        if a.ndim != 2 or b.ndim != 2:
            return False
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or out_dtype != torch.bfloat16:
            return False
        if not a.is_cuda or not b.is_cuda or a.device != b.device:
            return False
        layout = _hk_layout(trans_a, trans_b)
        if layout is None:
            return False
        m, n, k_a, k_b = GEMMHipKittenBackend._logical_shape(a, trans_a, b, trans_b)
        if k_a != k_b:
            return False
        # HK BF16 dense main kernel needs M%256, N%256, K%128. Tail
        # kernel handles small misalignment but we keep the gate
        # conservative to mirror the pre-refactor ``aligned_for`` rule.
        return (m > 0 and n > 0 and k_a > 0
                and (m % 256 == 0) and (n % 256 == 0) and (k_a % 128 == 0))

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
        layout = _hk_layout(trans_a, trans_b)
        if layout is None:
            raise ValueError("HipKitten BF16 backend supports RCR, RRR, and CRR layouts only.")

        # CRR + trans_c shortcut. ``(A^T @ B)^T == B^T @ A``: swapping
        # (a, b) into the CRR kernel yields the trans_c-transposed
        # output [n, m] directly and saves the post-GEMM
        # ``out.t().contiguous()`` copy (~12 % of kernel time on
        # 4096x4096x4096). Autograd dW always lands here for
        # trans_b=True forwards.
        a_in = a if a.is_contiguous() else a.contiguous()
        b_in = b if b.is_contiguous() else b.contiguous()
        op = torch.ops.primus_turbo_cpp_extension.hk_gemm_bf16
        if trans_c and layout == "crr":
            return op(b_in, a_in, layout, 4, 8)
        out = op(a_in, b_in, layout, 4, 8)
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
