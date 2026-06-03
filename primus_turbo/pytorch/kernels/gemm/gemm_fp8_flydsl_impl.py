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

Supports NT, NN, TN, TT layouts (NT is native; others go through a host
transpose of the non-canonical operand). trans_c is not supported.
"""

from __future__ import annotations

import torch

from primus_turbo.flydsl.gemm.gemm_fp8_kernel import (
    flydsl_available,
    gemm_fp8_tensorwise_flydsl_kernel,
)
from primus_turbo.pytorch.core.backend import KernelBackend
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)

from .gemm_fp8_impl import get_gemm_logical_shape

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

    Layout support (all four combos):
      - NT (native):  trans_a=F, trans_b=T
      - NN:           trans_a=F, trans_b=F   (host transposes B)
      - TN:           trans_a=T, trans_b=T   (host transposes A)
      - TT:           trans_a=T, trans_b=F   (host transposes both)

    Constraints:
      - TENSORWISE per-tensor scaling (a_scale / b_scale scalar)
      - out_dtype = bf16
      - K (contraction) % 128 == 0
      (trans_c=True is supported via post-hoc output transpose; extra mem copy vs Triton.)
    """

    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE}
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
        if not flydsl_available():
            return False
        supported = True
        supported &= granularity in GEMMFP8FlyDSLBackend.SUPPORTED_GRANULARITIES
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8FlyDSLBackend.SUPPORTED_DTYPES
        supported &= out_dtype == torch.bfloat16
        # K (contraction) must be a multiple of BLOCK_K=128
        _m, _n, k = get_gemm_logical_shape(a, b, trans_a, trans_b)
        supported &= (k % 128) == 0
        # Tile constraints: BLOCK_M min 128, BLOCK_N 256. Off-tile M/N is not
        # masked, so gate (fall back to another backend) rather than crash.
        # TODO(flydsl): support arbitrary M (and N) via byte-level addressing,
        # as done for the bf16 grouped path; then relax these gates.
        supported &= (_m % 128) == 0
        supported &= (_n % 256) == 0
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
