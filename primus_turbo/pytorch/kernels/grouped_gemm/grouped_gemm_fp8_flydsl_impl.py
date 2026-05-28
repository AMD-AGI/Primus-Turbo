###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL backend for grouped FP8 GEMM (forward + variable-K backward).

Layer-2 backends; the underlying kernel wrappers live in
``primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_kernel``. Both backend
classes are registered into the respective dispatchers in
``grouped_gemm_fp8_impl.py``.

Supports:
  - TENSORWISE per-tensor scaling only.
  - out_dtype = bf16.
  - fwd: K (contraction) % 128 == 0.
  - variable_k (wgrad): each group's M_g % 128 == 0 (per-group dense fallback).
"""

from __future__ import annotations

import torch

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_kernel import (
    flydsl_available,
    grouped_gemm_fp8_tensorwise_flydsl_kernel,
    grouped_gemm_fp8_tensorwise_variable_k_flydsl_kernel,
)
from primus_turbo.pytorch.core.backend import KernelBackend
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)


# Mirror the dtype tuples from grouped_gemm_fp8_impl.py.
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


class GroupedGEMMFP8FlyDSLBackend(KernelBackend):
    """FlyDSL 8-wave fp8 grouped GEMM backend (fwd + dgrad-via-trans_b).

    Constraints:
      - TENSORWISE per-tensor scale (a_scale / b_scale scalar)
      - out_dtype = bf16
      - K (contraction) % 128 == 0
      - trans_a = False (A must be row-major K-contig)
    """

    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE}
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        if not flydsl_available():
            return False
        supported = True
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8FlyDSLBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8FlyDSLBackend.SUPPORTED_GRANULARITIES
        supported &= out_dtype == torch.bfloat16
        supported &= not trans_a
        # K (contraction) divisible by BLOCK_K=128
        k_contraction = a.shape[1] if not trans_a else a.shape[0]
        supported &= (k_contraction % 128) == 0
        # per-tensor scalar scales
        supported &= a_scales.numel() == 1 and b_scales.numel() == 1
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ):
        return grouped_gemm_fp8_tensorwise_flydsl_kernel(
            a,
            b,
            a_scales,
            b_scales,
            group_offs,
            trans_b=trans_b,
            out_dtype=out_dtype,
        )


class GroupedGEMMFP8VariableKFlyDSLBackend(KernelBackend):
    """FlyDSL 8-wave fp8 variable-K grouped GEMM backend (wgrad).

    P0 implementation: per-group dense GEMM fallback. Each group is dispatched
    to the single-launch dense FlyDSL kernel with M_g as the contraction dim.
    Triton's true single-launch variable-K kernel is faster on multi-group
    workloads; a native FlyDSL grouped variable-K kernel is a separate effort.

    Constraints:
      - TENSORWISE per-tensor scale
      - out_dtype = bf16
      - Each group's M_g % 128 == 0
      - trans_a = True, trans_b = False (standard wgrad layout)
    """

    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE}
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ) -> bool:
        if not flydsl_available():
            return False
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8VariableKFlyDSLBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8VariableKFlyDSLBackend.SUPPORTED_GRANULARITIES
        supported &= out_dtype == torch.bfloat16
        supported &= trans_a and not trans_b
        supported &= a_scales.numel() == 1 and b_scales.numel() == 1
        # Per-group dense fallback requires each group's M_g (= K_inner of the
        # per-group gemm) % 128 == 0. Note: with `balance=False` test inputs
        # this gate will reject -- choose `balance=True` for FlyDSL coverage,
        # or fall back to CK/HipBLASLt/Triton for uneven distributions.
        try:
            offs = group_offs.tolist()
            for i in range(len(offs) - 1):
                if (offs[i + 1] - offs[i]) % 128 != 0:
                    return False
        except Exception:
            return False
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        group_lens: torch.Tensor,
        group_offs: torch.Tensor,
        trans_a: bool,
        trans_b: bool,
        trans_c: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        **kwargs,
    ):
        # Mirror Triton var_k dispatcher's lhs/rhs swap on trans_c. The wrapper
        # always returns [G, lhs.shape[1], rhs.shape[1]] so the swap alone is
        # what aligns the orientation with what the autograd Function expects.
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales
        return grouped_gemm_fp8_tensorwise_variable_k_flydsl_kernel(
            lhs,
            rhs,
            lhs_scales,
            rhs_scales,
            group_offs,
            out_dtype=out_dtype,
        )
