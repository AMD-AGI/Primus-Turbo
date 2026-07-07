###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Tuple

import torch

_torch_custom_op_wrapper = torch.library.custom_op

from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel
from primus_turbo.flydsl.gemm.mxfp8_gemm_kernel import gemm_mxfp8_flydsl_kernel
from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.core.utils import get_device_compute_capability
from primus_turbo.triton.gemm.gemm_fp8_kernel import (
    gemm_fp8_blockwise_triton_kernel,
    gemm_fp8_rowwise_triton_kernel,
    gemm_fp8_tensorwise_triton_kernel,
)


def get_gemm_logical_shape(
    a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> Tuple[int, int, int]:
    assert a.ndim == 2 and b.ndim == 2, (
        f"Expected both a and b to be 2D tensors, but got a.ndim={a.ndim}, b.ndim={b.ndim}"
    )
    M = a.shape[1] if trans_a else a.shape[0]
    Ka = a.shape[0] if trans_a else a.shape[1]
    Kb = b.shape[1] if trans_b else b.shape[0]
    N = b.shape[0] if trans_b else b.shape[1]
    assert Ka == Kb, f"GEMM K mismatch: a has K={Ka}, b has K={Kb}"
    return M, N, Ka


_COMMON_SUPPORTED_DTYPES = (
    (float8_e4m3, float8_e4m3, torch.float16),
    (float8_e4m3, float8_e4m3, torch.bfloat16),
    (float8_e5m2, float8_e5m2, torch.float16),
    (float8_e5m2, float8_e5m2, torch.bfloat16),
)

_HYBRID_SUPPORTED_DTYPES = (
    (float8_e4m3, float8_e5m2, torch.float16),
    (float8_e4m3, float8_e5m2, torch.bfloat16),
    (float8_e5m2, float8_e4m3, torch.float16),
    (float8_e5m2, float8_e4m3, torch.bfloat16),
)


class GEMMFP8HipBLASLtBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.MX_BLOCKWISE,
    }

    # (a_dtype, b_dtype, c_dtype)
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

    # (trans_a, trans_b, trans_c)
    SUPPORTED_LAYOUTS = (
        (False, False, False),
        (False, True, False),
        (True, False, False),
    )

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
        supported &= granularity in GEMMFP8HipBLASLtBackend.SUPPORTED_GRANULARITIES
        # check dtype
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8HipBLASLtBackend.SUPPORTED_DTYPES

        # TODO:
        # check layout
        # supported &= (trans_a, trans_b, trans_c) in GEMMFP8HipBLASLtBackend.SUPPORTED_LAYOUTS
        # TODO:
        # check shape

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
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp8(
            a, a_scale_inv, b, b_scale_inv, out_dtype, trans_a, trans_b, trans_c, granularity.name
        )


class GEMMFP8CKBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
    }

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

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
        supported &= granularity in GEMMFP8CKBackend.SUPPORTED_GRANULARITIES
        # check dtype
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8CKBackend.SUPPORTED_DTYPES

        if trans_c:
            lhs, rhs = b, a
            trans_lhs, trans_rhs = (not trans_b), (not trans_a)
        else:
            lhs, rhs = a, b
            trans_lhs, trans_rhs = trans_a, trans_b

        k = lhs.shape[0] if trans_lhs else lhs.shape[1]
        n = rhs.shape[0] if trans_rhs else rhs.shape[1]

        # NT / NN layout (transA == False): the contraction dim k must be a
        # multiple of 32.
        if not trans_lhs:
            supported &= k % 32 == 0
            # BLOCKWISE additionally requires k, n multiples of 128 and k >= 128.
            if granularity == ScalingGranularity.BLOCKWISE:
                supported &= (k % 128 == 0) and (n % 128 == 0) and (k >= 128)

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
        if trans_c:
            lhs, rhs = b, a
            lhs_scale_inv, rhs_scale_inv = b_scale_inv, a_scale_inv
            trans_lhs = not trans_b
            trans_rhs = not trans_a
        else:
            lhs, rhs = a, b
            lhs_scale_inv, rhs_scale_inv = a_scale_inv, b_scale_inv
            trans_lhs = trans_a
            trans_rhs = trans_b

        return torch.ops.primus_turbo_cpp_extension.ck_gemm_fp8(
            lhs, rhs, lhs_scale_inv, rhs_scale_inv, trans_lhs, trans_rhs, out_dtype, granularity.name
        )


class GEMMFP8TritonBackend(KernelBackend):
    """Triton persistent-kernel backend for FP8 GEMM.

    Supports:
      - TENSORWISE: per-tensor scaling (all layouts), including HYBRID format
      - ROWWISE: per-row/per-col vector scaling (all layouts)
      - BLOCKWISE: block-wise scaling with three layouts:
          NT/RCR (forward), NN/RRR (grad_X), TN/CRR (grad_W)
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
    }

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

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
        supported &= granularity in GEMMFP8TritonBackend.SUPPORTED_GRANULARITIES
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8TritonBackend.SUPPORTED_DTYPES
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
        if granularity == ScalingGranularity.TENSORWISE:
            return gemm_fp8_tensorwise_triton_kernel(
                a,
                a_scale_inv,
                b,
                b_scale_inv,
                trans_a=trans_a,
                trans_b=trans_b,
                out_dtype=out_dtype,
                trans_c=trans_c,
            )
        elif granularity == ScalingGranularity.ROWWISE:
            return gemm_fp8_rowwise_triton_kernel(
                a,
                a_scale_inv,
                b,
                b_scale_inv,
                trans_a=trans_a,
                trans_b=trans_b,
                out_dtype=out_dtype,
                trans_c=trans_c,
            )
        elif granularity == ScalingGranularity.BLOCKWISE:
            return gemm_fp8_blockwise_triton_kernel(
                a,
                a_scale_inv,
                b,
                b_scale_inv,
                trans_a=trans_a,
                trans_b=trans_b,
                out_dtype=out_dtype,
                trans_c=trans_c,
            )
        else:
            raise ValueError(f"Unsupported granularity for FP8 Triton: {granularity}")


class GEMMFP8TurboBackend(KernelBackend):
    """Hand-tuned MXFP8 GEMM kernel for GFX950 (MI350/MI355).

    Supports MX_BLOCKWISE only. NT layout. Tile 256x256x128.
    Shape constraints: m,n % 16 == 0, k % 128 == 0, k >= 384.
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.MX_BLOCKWISE,
    }

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

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
        supported &= granularity in GEMMFP8TurboBackend.SUPPORTED_GRANULARITIES
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8TurboBackend.SUPPORTED_DTYPES
        supported &= not trans_a and trans_b and not trans_c
        m, n, k = get_gemm_logical_shape(a, b, trans_a, trans_b)
        supported &= m % 16 == 0 and n % 16 == 0 and k % 128 == 0 and k >= 384
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
        return torch.ops.primus_turbo_cpp_extension.turbo_gemm_fp8(
            a, a_scale_inv, b, b_scale_inv, out_dtype, trans_a, trans_b, trans_c, granularity.name
        )


class GEMMFP8FlyDSLBackend(KernelBackend):
    """FlyDSL 8-wave fp8 dense GEMM backend (gfx950 only).

    TENSORWISE: scalar a_scale/b_scale, bf16/fp16 out, arbitrary M/N/K, layouts
    NT/NN/TN (TT unsupported). trans_c via post-hoc transpose.

    MX_BLOCKWISE: NT only, per-operand E4M3/E5M2 (incl. hybrid), bf16/fp16 out,
    per-1x32 raw E8M0 2D scales [M,K//32]/[N,K//32]. The kernel repacks the scales
    to its preshuffled layout, so execute() does no host-side padding.
    """

    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE, ScalingGranularity.MX_BLOCKWISE}
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

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
        # gfx950 (CDNA4) only: kernel uses mfma_f32_16x16x128_f8f6f4, absent on gfx942-.
        supported &= get_device_compute_capability() >= (9, 5)
        supported &= granularity in GEMMFP8FlyDSLBackend.SUPPORTED_GRANULARITIES
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8FlyDSLBackend.SUPPORTED_DTYPES
        m, n, k = get_gemm_logical_shape(a, b, trans_a, trans_b)

        if granularity == ScalingGranularity.MX_BLOCKWISE:
            # NT only; per-operand E4M3/E5M2; raw E8M0 2D scales [M,K//32]/[N,K//32].
            supported &= (not trans_a) and trans_b
            supported &= k % 128 == 0 and k >= 256
            supported &= a_scale_inv.shape == (m, k // 32) and b_scale_inv.shape == (n, k // 32)
            supported &= a_scale_inv.element_size() == 1 and b_scale_inv.element_size() == 1
            return supported

        # TENSORWISE: NT/NN/TN native (TT unsupported), scalar per-tensor scales.
        supported &= not (trans_a and trans_b)
        supported &= k > 128  # software pipeline needs >= 2 K tiles: ceil(K/128) >= 2
        supported &= a_scale_inv.numel() == 1 and b_scale_inv.numel() == 1
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
        if granularity == ScalingGranularity.MX_BLOCKWISE:
            out = gemm_mxfp8_flydsl_kernel(
                a, a_scale_inv.view(torch.uint8), b, b_scale_inv.view(torch.uint8), out_dtype=out_dtype
            )
            return out.t().contiguous() if trans_c else out

        if trans_c:
            lhs, rhs = b, a
            lhs_scale_inv, rhs_scale_inv = b_scale_inv, a_scale_inv
            trans_lhs, trans_rhs = (not trans_b), (not trans_a)
        else:
            lhs, rhs = a, b
            lhs_scale_inv, rhs_scale_inv = a_scale_inv, b_scale_inv
            trans_lhs, trans_rhs = trans_a, trans_b
        return gemm_fp8_tensorwise_flydsl_kernel(
            lhs,
            lhs_scale_inv,
            rhs,
            rhs_scale_inv,
            trans_a=trans_lhs,
            trans_b=trans_rhs,
            out_dtype=out_dtype,
            trans_c=False,
        )


_GEMM_FP8_BACKENDS = {
    BackendType.TURBO: BackendEntry(GEMMFP8TurboBackend),
    BackendType.HIPBLASLT: BackendEntry(GEMMFP8HipBLASLtBackend),
    BackendType.CK: BackendEntry(GEMMFP8CKBackend),
    BackendType.TRITON: BackendEntry(GEMMFP8TritonBackend),
    BackendType.FLYDSL: BackendEntry(GEMMFP8FlyDSLBackend),
}


class GEMMFP8KernelDispatcher(AutoKernelDispatcher):
    _backends = _GEMM_FP8_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, a, b, trans_a, trans_b, trans_c, out_dtype, granularity, **kwargs):
        m, n, k = get_gemm_logical_shape(a, b, trans_a, trans_b)
        return (m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c, granularity)


@_torch_custom_op_wrapper("primus_turbo::gemm_fp8_impl", mutates_args=(), device_types="cuda")
def gemm_fp8_impl(
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
    user_backend_enum = GlobalBackendManager.get_gemm_backend(PrecisionType.FP8)
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

    return GEMMFP8KernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


@gemm_fp8_impl.register_fake
def gemm_fp8_impl_meta(
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
