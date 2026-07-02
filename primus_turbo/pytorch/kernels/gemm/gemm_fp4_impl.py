###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Tuple

import torch

_torch_custom_op_wrapper = torch.library.custom_op

from primus_turbo.common.aiter_utils import get_aiter
from primus_turbo.flydsl.gemm.mxfp4_gemm_kernel import gemm_mxfp4_flydsl_kernel
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
from primus_turbo.pytorch.core.utils import get_device_compute_capability


def ceil_div(a, b):
    return (a + b - 1) // b


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
        preshuffled: bool = False,
    ) -> bool:
        # HipBLASLt vendor wrapper has no preshuffle plumbing (see
        # csrc/kernels/gemm/hipblaslt_gemm.cu) and would silently produce
        # garbage on AITER-preshuffled inputs. Refuse the layout cleanly.
        if preshuffled:
            return False

        supported = True
        # check ScalingGranularity
        supported &= granularity in GEMMFP4HipBLASLtBackend.SUPPORTED_GRANULARITIES
        # check dtype
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP4HipBLASLtBackend.SUPPORTED_DTYPES
        supported &= (trans_a, trans_b, trans_c) in GEMMFP4HipBLASLtBackend.SUPPORTED_LAYOUTS

        # check dimension. Assume layout is NT.
        supported &= (
            a.size(0) % GEMMFP4HipBLASLtBackend.HIPBLASLT_M_MULTIPLE == 0
            and b.size(0) % GEMMFP4HipBLASLtBackend.HIPBLASLT_N_MULTIPLE == 0
        )

        # NOTE: The k dim is packed for FP4. So it need to multiply 2.
        supported &= (a.size(1) * 2) % GEMMFP4HipBLASLtBackend.HIPBLASLT_K_MULTIPLE == 0 and (
            b.size(1) * 2
        ) % GEMMFP4HipBLASLtBackend.HIPBLASLT_K_MULTIPLE == 0

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
        preshuffled: bool = False,
    ):
        # preshuffled is accepted only so the dispatcher's uniform
        # execute(**kwargs) call works; can_handle already rejected the
        # preshuffled=True case.
        del preshuffled
        # TODO(ruibin): Add padding
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp4(
            a, a_scale_inv, b, b_scale_inv, out_dtype, trans_a, trans_b, trans_c, granularity.name
        )


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
    AITER_FP4GEMM_K_MULTIPLE = 32

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
        preshuffled: bool = False,
    ) -> bool:
        del preshuffled  # AITER handles both layouts
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
            and a.size(1) * 2 % GEMMFP4AITERBackend.AITER_FP4GEMM_K_MULTIPLE == 0
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
        preshuffled: bool = False,
    ):
        if preshuffled:
            # Fast path: caller guarantees a_scale_inv, b_scale_inv, and b
            # were already produced in the AITER 16x16-tile layout (e.g. via
            # quantize_mxfp4{,_dual}(shuffle_scale=True, shuffle_out=True)).
            # Skip the 3 standalone shuffle kernel launches.
            return get_aiter().gemm_a4w4(a, b, a_scale_inv, b_scale_inv, dtype=out_dtype, bpreshuffle=True)

        # NOTE: AITER FP4 GEMM requires shuffled scale and B
        a_scale_inv_shuffled = torch.ops.primus_turbo_cpp_extension.shuffle_scale(a_scale_inv, [16, 16])
        b_scale_inv_shuffled = torch.ops.primus_turbo_cpp_extension.shuffle_scale(b_scale_inv, [16, 16])
        b_shuffled = torch.ops.primus_turbo_cpp_extension.shuffle_weight(b, [16, 16])
        return get_aiter().gemm_a4w4(
            a, b_shuffled, a_scale_inv_shuffled, b_scale_inv_shuffled, dtype=out_dtype, bpreshuffle=True
        )


class GEMMFP4FlyDSLBackend(KernelBackend):
    """FlyDSL 4-wave MXFP4 dense GEMM backend (gfx950 / CDNA4 only).

    NT only (trans_a=F, trans_b=T), E2M1 fp4 operands, bf16/fp16 out. Per-1x32 E8M0
    block scales [M,K//32]/[N,K//32] are passed straight to ``gemm_mxfp4_flydsl_kernel``,
    which repacks them into the lane-contiguous VGPR-direct layout via a separate
    FlyDSL preshuffle kernel (quant stays generic, mirroring the mxfp8 GEMM). The
    kernel is a whole-loop bare-asm path tuned for 256x256x256 tiles, so M/N/K must
    all be multiples of 256.
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.MX_BLOCKWISE,
    }

    # (a_dtype, b_dtype, c_dtype)
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES)

    # (trans_a, trans_b, trans_c)
    SUPPORTED_LAYOUTS = ((False, True, False),)

    FLYDSL_FP4_MNK_MULTIPLE = 256

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
        preshuffled: bool = False,
    ) -> bool:

        supported = True
        # gfx950 (CDNA4) only: mfma_scale_f32_16x16x128_f8f6f4 is absent below.
        supported &= get_device_compute_capability() >= (9, 5)
        supported &= granularity in GEMMFP4FlyDSLBackend.SUPPORTED_GRANULARITIES
        supported &= a.ndim == 2 and b.ndim == 2
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP4FlyDSLBackend.SUPPORTED_DTYPES
        supported &= (trans_a, trans_b, trans_c) in GEMMFP4FlyDSLBackend.SUPPORTED_LAYOUTS
        if not supported:
            return False

        m, n = a.size(0), b.size(0)
        k = a.size(1) * 2  # FP4 K dim is packed (2 values / byte)
        mul = GEMMFP4FlyDSLBackend.FLYDSL_FP4_MNK_MULTIPLE
        supported &= (m % mul == 0) and (n % mul == 0) and (k % mul == 0)

        # Raw E8M0 block scales, 1 byte/elem, [DIM, K//32] (the GEMM wrapper preshuffles
        # them into its lane-contiguous layout).
        def _scale_ok(s, dim):
            return s.ndim == 2 and s.shape == (dim, k // 32) and s.element_size() == 1

        supported &= _scale_ok(a_scale_inv, m)
        supported &= _scale_ok(b_scale_inv, n)
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
        preshuffled: bool = False,
    ):
        # preshuffled accepted only so the dispatcher's uniform execute(**kwargs)
        # call works; can_handle already rejected the preshuffled=True case.
        del preshuffled
        # Raw E8M0 block scales ([DIM, K/32], 1 byte/elem) are passed straight through;
        # the FlyDSL GEMM wrapper repacks them into its lane-contiguous layout via a
        # separate preshuffle kernel on the same stream (quant stays generic). The
        # whole-loop kernel consumes any K % 256 (KI//2 pairs + MFMA-only odd tail).
        return gemm_mxfp4_flydsl_kernel(a, a_scale_inv, b, b_scale_inv, out_dtype=out_dtype, trans_c=trans_c)


_GEMM_FP4_BACKENDS = {
    BackendType.AITER: BackendEntry(GEMMFP4AITERBackend, autotune=False),
    BackendType.HIPBLASLT: BackendEntry(GEMMFP4HipBLASLtBackend),
    BackendType.FLYDSL: BackendEntry(GEMMFP4FlyDSLBackend, autotune=False),
}


class GEMMFP4KernelDispatcher(AutoKernelDispatcher):
    _backends = _GEMM_FP4_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, a, b, trans_a, trans_b, trans_c, out_dtype, granularity, preshuffled=False, **kwargs):
        m, n, k = get_gemm_logical_shape(a, b, trans_a, trans_b)
        return (m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c, granularity, preshuffled)


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
    preshuffled: bool = False,
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
        preshuffled=preshuffled,
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
    preshuffled: bool = False,
) -> torch.Tensor:
    m, n, _ = get_gemm_logical_shape(a, b, trans_a, trans_b)
    if trans_c:
        m, n = n, m
    return torch.empty(m, n, dtype=out_dtype, device=a.device)
