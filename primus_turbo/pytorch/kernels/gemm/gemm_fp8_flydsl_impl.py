###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL backend for dense FP8 GEMM.

Layer-2 backend class; the underlying kernel wrapper lives in
``primus_turbo.flydsl.gemm.gemm_fp8_kernel``. Registered into
``GEMMFP8KernelDispatcher`` via ``_GEMM_FP8_BACKENDS`` in
``gemm_fp8_impl.py``.

Supports NT, NN, TN natively (no host transpose); TT is not supported. trans_c
is supported via a post-hoc output transpose.
"""

from __future__ import annotations

import torch

from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel
from primus_turbo.pytorch.core.backend import KernelBackend
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.core.utils import get_device_compute_capability

# Mirror the dtype tuples from gemm_fp8_impl.py so we don't tightly couple.
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


class GEMMFP8FlyDSLBackend(KernelBackend):
    """FlyDSL 8-wave fp8 dense GEMM backend.

    Layout support:
      - NT (native):  trans_a=F, trans_b=T
      - NN (native):  trans_a=F, trans_b=F
      - TN (native):  trans_a=T, trans_b=F
      - TT:           trans_a=T, trans_b=T   (not supported)

    Constraints:
      - TENSORWISE per-tensor scaling (a_scale / b_scale scalar)
      - out_dtype in {bf16, fp16}
      - arbitrary contraction K, M and N
      (trans_c=True is supported via post-hoc output transpose; extra mem copy vs Triton.)
    """

    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE}
    # E4M3 / E5M2 / hybrid, bf16 or fp16 output. Per-operand fp8 format is threaded
    # into the MFMA via cbsz(srcA)/blgp(srcB) (0=E4M3, 1=E5M2) and the FlyDSL
    # MFMA_Scale atom dtype; fp16 output is produced from the f32 accumulator.
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES) | set(_HYBRID_SUPPORTED_DTYPES)

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
        # gfx950 (CDNA4) only: the kernel uses mfma_f32_16x16x128_f8f6f4, absent
        # on gfx942 and below. Gate here so the dispatcher never picks FlyDSL off
        # gfx950 (the backend still imports fine on other archs).
        supported &= get_device_compute_capability() >= (9, 5)
        supported &= granularity in GEMMFP8FlyDSLBackend.SUPPORTED_GRANULARITIES
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8FlyDSLBackend.SUPPORTED_DTYPES
        supported &= out_dtype in (torch.bfloat16, torch.float16)
        # NT / NN / TN native; TT (trans_a and trans_b) is not supported.
        supported &= not (trans_a and trans_b)
        # Contraction K: any value handled by the native K-tail, but the software
        # pipeline needs K_ITERS = ceil(K/128) >= 2, i.e. K >= 129. (M / N are
        # arbitrary: the partial last output tile is bounded by the c_m / c_n
        # StoreC clamp + the global SRD.)
        k = a.shape[0] if trans_a else a.shape[1]
        supported &= k >= 129
        # per-tensor scalar scale (wrapper broadcasts to vector internally)
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
        return gemm_fp8_tensorwise_flydsl_kernel(
            a,
            a_scale_inv,
            b,
            b_scale_inv,
            trans_a=trans_a,
            trans_b=trans_b,
            out_dtype=out_dtype,
            trans_c=trans_c,
        )
