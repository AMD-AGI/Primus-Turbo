###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""KernelBackend subclasses for FP8 grouped-GEMM ASM_CO kernels.

``GroupedGEMMFP8ASMCOBackend`` handles the FWD (trans_b=False) and DGRAD
(trans_b=True) passes via per-shape .hsaco files.
``GroupedGEMMFP8VariableKASMCOBackend`` handles the WGRAD (variable-K) pass
via dot_scaled_v2_{fixed,beta1}.co files.

Both backends only accept E=32 experts and tensorwise FP8 scaling — shapes
outside the supported set cause ``can_handle()`` to return ``False``, which
allows the dispatcher to fall back to Triton automatically.
"""

import torch

from primus_turbo.pytorch.core.backend import KernelBackend
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e4m3, float8_e5m2
from primus_turbo.asm_co.grouped_gemm.launcher import (
    launch_asm_co_fwd,
    launch_asm_co_fwd_dgrad,
    launch_asm_co_wgrad_variable_k,
    launch_asm_co_wgrad_variable_k_beta1,
)

__all__ = [
    "GroupedGEMMFP8ASMCOBackend",
    "GroupedGEMMFP8VariableKASMCOBackend",
]

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

# (K, N) shapes supported by the FWD/DGRAD .hsaco files (gpt-oss MoE on MI355X)
_ASM_CO_FWD_SITES_TRANSB = {
    (2880, 5760),  # gate_up_dgrad
    (2880, 2880),  # down_dgrad
    (5760, 2880),  # gate_up_dgrad (alternate)
}
_ASM_CO_FWD_SITES_NOTRANSB = {
    (2880, 5760),  # gate_up_fwd
    (2880, 2880),  # down_fwd
}

# (OUT_M, OUT_N) shapes supported by the WGRAD .co files
_ASM_CO_WGRAD_SITES = {
    (2880, 5760),  # gate_up_wgrad
    (2880, 2880),  # down_wgrad
}


class GroupedGEMMFP8ASMCOBackend(KernelBackend):
    """Hand-tuned AMDGCN assembly (.hsaco) backend for FP8 grouped GEMM FWD and DGRAD.

    Activated when ``PRIMUS_TURBO_GROUPED_GEMM_BACKEND=ASM_CO``.
    Requires tensorwise scaling, E=32 experts, gpt-oss MoE shapes on MI355X (gfx950).
    Unsupported shapes return ``can_handle() == False`` so the dispatcher falls
    back to Triton silently.
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
        supported = True
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8ASMCOBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8ASMCOBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
        supported &= b.shape[0] == 32  # E=32 experts
        k = a.shape[1]
        if trans_b:
            n = b.shape[-2]
            supported &= (k, n) in _ASM_CO_FWD_SITES_TRANSB
        else:
            n = b.shape[-1]
            supported &= (k, n) in _ASM_CO_FWD_SITES_NOTRANSB
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
        if trans_b:
            return launch_asm_co_fwd_dgrad(a, b, a_scales, b_scales, group_offs, trans_b, out_dtype, num_cu)
        return launch_asm_co_fwd(a, b, a_scales, b_scales, group_offs, out_dtype, num_cu)


class GroupedGEMMFP8VariableKASMCOBackend(KernelBackend):
    """Hand-tuned AMDGCN assembly (.co) backend for FP8 variable-K grouped GEMM (WGRAD).

    Activated when ``PRIMUS_TURBO_GROUPED_GEMM_BACKEND=ASM_CO`` and the call
    arrives with ``is_bwd=True``.  Tensorwise scaling only, E=32 experts,
    gpt-oss MoE shapes on MI355X (gfx950).
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
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8VariableKASMCOBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8VariableKASMCOBackend.SUPPORTED_GRANULARITIES
        supported &= trans_a and not trans_b
        out_m = a.shape[1]
        out_n = b.shape[1]
        if trans_c:
            out_m, out_n = out_n, out_m
        supported &= (out_m, out_n) in _ASM_CO_WGRAD_SITES
        supported &= group_lens.shape[0] == 32  # E=32 experts
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
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales
        return launch_asm_co_wgrad_variable_k(
            lhs, rhs, lhs_scales, rhs_scales, group_lens, group_offs, out_dtype, num_cu
        )

    @staticmethod
    def execute_beta1(
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
        out: torch.Tensor,
        **kwargs,
    ):
        """In-place fused accumulation: ``out += A^T @ B * scale``."""
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales
        launch_asm_co_wgrad_variable_k_beta1(
            lhs, rhs, lhs_scales, rhs_scales, group_lens, group_offs, out, num_cu
        )
