###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus_turbo.pytorch.core.backend import (
    BackendChoice,
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
from primus_turbo.pytorch.core.utils import (
    build_ck,
    is_gfx942,
    is_gfx950,
    is_gfx1250,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    BaseGroupedGEMMKernelDispatcher,
    BaseGroupedGEMMVariableKKernelDispatcher,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
    grouped_gemm_fp8_blockwise_triton_kernel,
    grouped_gemm_fp8_blockwise_variable_k_triton_kernel,
    grouped_gemm_fp8_rowwise_triton_kernel,
    grouped_gemm_fp8_rowwise_variable_k_triton_kernel,
    grouped_gemm_fp8_tensorwise_triton_kernel,
    grouped_gemm_fp8_tensorwise_variable_k_triton_kernel,
    grouped_gemm_mxfp8_triton_kernel,
    grouped_gemm_mxfp8_variable_k_triton_kernel,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_helper import (
    grouped_gemm_output_tail_kernel,
)

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


class GroupedGEMMFP8CKBackend(KernelBackend):
    # BLOCKWISE intentionally excluded: the Triton path (with pshuffled scales +
    # HIP fused quant) is the production blockwise backend; CK adds no value here.
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
    }

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
        # check the CK backend was compiled into this build
        supported &= build_ck()
        supported &= not is_gfx1250()
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8CKBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8CKBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
        # This backend writes the full operand pitch, so it can't produce a tight output;
        # decline only when a shrinking tight output is requested (n_real < operand free dim).
        n_real = kwargs.get("n_real", None)
        if n_real is not None:
            n_operand = b.shape[-2] if trans_b else b.shape[-1]
            supported &= n_real == n_operand
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
        return torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm_fp8(
            a,
            b,
            a_scales,
            b_scales,
            group_lens,
            group_offs,
            trans_a,
            trans_b,
            out_dtype,
            granularity.name,
            num_cu,
        )


class GroupedGEMMFP8VariableKCKBackend(KernelBackend):
    # BLOCKWISE intentionally excluded: variable-K BLOCKWISE wgrad runs on Triton.
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
    }

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
        inplace_add_to_out: bool = False,
        **kwargs,
    ) -> bool:
        supported = True
        # This backend has no beta=1 accumulate epilogue.
        supported &= not inplace_add_to_out
        # No N/K-pad tight-output path here.
        supported &= kwargs.get("m_real", None) is None and kwargs.get("n_real", None) is None
        # check the CK backend was compiled into this build
        supported &= build_ck()
        supported &= not is_gfx1250()
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8VariableKCKBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8VariableKCKBackend.SUPPORTED_GRANULARITIES
        supported &= trans_a and not trans_b
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
            trans_lhs, trans_rhs = not trans_b, not trans_a
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales
            trans_lhs, trans_rhs = trans_a, trans_b
        return torch.ops.primus_turbo_cpp_extension.ck_grouped_gemm_fp8_variable_k(
            lhs,
            rhs,
            lhs_scales,
            rhs_scales,
            group_lens,
            group_offs,
            trans_lhs,
            trans_rhs,
            out_dtype,
            granularity.name,
            num_cu,
        )


class GroupedGEMMFP8HipblasltBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
    }

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
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8HipblasltBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8HipblasltBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
        # Can't produce a tight output; decline only when one is requested (n_real < operand pitch).
        n_real = kwargs.get("n_real", None)
        if n_real is not None:
            n_operand = b.shape[-2] if trans_b else b.shape[-1]
            supported &= n_real == n_operand
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
        maybe_pre_sync: bool = False,
        **kwargs,
    ):
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_grouped_gemm_fp8(
            a,
            b,
            a_scales,
            b_scales,
            group_lens,
            group_offs,
            trans_a,
            trans_b,
            out_dtype,
            granularity.name,
            maybe_pre_sync,
        )


class GroupedGEMMFP8VariableKHipblasltBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
    }

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
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        supported = True
        # No N/K-pad tight-output path here.
        supported &= kwargs.get("m_real", None) is None and kwargs.get("n_real", None) is None
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8VariableKHipblasltBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8VariableKHipblasltBackend.SUPPORTED_GRANULARITIES
        supported &= trans_a and not trans_b

        if inplace_add_to_out:
            supported &= out is not None and out.is_contiguous()
            supported &= out is not None and out.dtype in (
                torch.float32,
                torch.bfloat16,
                torch.float16,
            )

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
        maybe_pre_sync: bool = False,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ):
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
            trans_lhs, trans_rhs = not trans_b, not trans_a
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales
            trans_lhs, trans_rhs = trans_a, trans_b
        beta = 1.0 if inplace_add_to_out else 0.0
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_grouped_gemm_fp8(
            lhs,
            rhs,
            lhs_scales,
            rhs_scales,
            group_lens,
            group_offs,
            trans_lhs,
            trans_rhs,
            out_dtype,
            granularity.name,
            maybe_pre_sync,
            beta,
            out,
        )


class GroupedGEMMFP8TritonBackend(KernelBackend):
    """Triton persistent-kernel backend for FP8 grouped GEMM (CPU-sync-free).

    Supports:
      - TENSORWISE: per-tensor scaling, including HYBRID format
      - ROWWISE: per-row/per-col vector scaling
      - BLOCKWISE: block-wise scaling (2D B_scales per group)
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
        ScalingGranularity.MX_BLOCKWISE,
    }

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
        supported &= granularity in GroupedGEMMFP8TritonBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
        if granularity != ScalingGranularity.MX_BLOCKWISE:
            supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8TritonBackend.SUPPORTED_DTYPES
        else:
            # MXFP8: both operands must be fp8 (e4m3/e5m2) — the kernel infers the
            # format from a.dtype — and the layout is NT only (trans_b=True).
            supported &= not is_gfx942()
            supported &= a.dtype in (float8_e4m3, float8_e5m2)
            supported &= b.dtype in (float8_e4m3, float8_e5m2)
            supported &= out_dtype in (torch.float16, torch.bfloat16)
            supported &= trans_b
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
        if granularity == ScalingGranularity.MX_BLOCKWISE:
            # b is (G, N, K) NT.  group_offs = padded read offsets; group_offs_out
            # = real write offsets (output over-allocated to padded rows, sliced
            # by the caller).
            N = b.shape[-2]
            K = b.shape[-1]
            group_offs_out = kwargs.get("group_offs_out", None)
            return grouped_gemm_mxfp8_triton_kernel(
                a,
                a_scales,
                b,
                b_scales,
                group_offs,
                N,
                K,
                group_offs_out=group_offs_out,
                out_dtype=out_dtype,
                num_cu=num_cu,
            )
        if granularity == ScalingGranularity.BLOCKWISE:
            return grouped_gemm_fp8_blockwise_triton_kernel(
                a,
                b,
                a_scales,
                b_scales,
                group_offs,
                trans_b=trans_b,
                out_dtype=out_dtype,
            )
        elif granularity == ScalingGranularity.ROWWISE:
            return grouped_gemm_fp8_rowwise_triton_kernel(
                a,
                b,
                a_scales,
                b_scales,
                group_offs,
                trans_b=trans_b,
                out_dtype=out_dtype,
            )
        return grouped_gemm_fp8_tensorwise_triton_kernel(
            a,
            b,
            a_scales,
            b_scales,
            group_offs,
            trans_b=trans_b,
            out_dtype=out_dtype,
            n_real=kwargs.get("n_real", None),
        )


class GroupedGEMMFP8FlyDSLBackend(KernelBackend):
    """FlyDSL fp8 grouped GEMM backend (gfx950).

    M-grouped operator. Granularities:
      - TENSORWISE: forward (trans_b=True, NT) + dgrad (trans_b=False, NN).
      - MX_BLOCKWISE: NT only (trans_b=True), per-1x32 raw E8M0 scales.
    Uses the FlyDSL mfma[_scale]_f32_16x16x128_f8f6f4 kernel (gfx950-only).
    """

    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE, ScalingGranularity.MX_BLOCKWISE}
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
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8FlyDSLBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8FlyDSLBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
        # gfx950 (CDNA4) only: kernel uses mfma[_scale]_f32_16x16x128_f8f6f4.
        supported &= is_gfx950()

        if granularity == ScalingGranularity.MX_BLOCKWISE:
            # NT only; per-1x32 raw E8M0 scales; K % 128 == 0 and K >= 256.
            supported &= trans_b
            supported &= a.shape[1] % 128 == 0 and a.shape[1] >= 256
            return supported

        # TENSORWISE: per-tensor scalar scales; contraction K >= 129.
        supported &= a_scales.numel() == 1 and b_scales.numel() == 1
        supported &= a.shape[1] >= 129
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
        from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_kernel import (
            grouped_gemm_fp8_tensorwise_flydsl_kernel,
        )

        if granularity == ScalingGranularity.MX_BLOCKWISE:
            from primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp8_kernel import (
                grouped_gemm_mxfp8_flydsl_kernel,
            )

            N = b.shape[-2]
            K = b.shape[-1]
            group_offs_out = kwargs.get("group_offs_out", None)
            return grouped_gemm_mxfp8_flydsl_kernel(
                a,
                a_scales,
                b,
                b_scales,
                group_offs,
                N,
                K,
                group_offs_out,
                out_dtype=out_dtype,
                num_cu=num_cu,
            )

        return grouped_gemm_fp8_tensorwise_flydsl_kernel(
            a,
            b,
            a_scales,
            b_scales,
            group_offs,
            trans_b=trans_b,
            out_dtype=out_dtype,
            num_cu=num_cu,
            n_real=kwargs.get("n_real", None),
        )


class GroupedGEMMFP8KernelDispatcher(BaseGroupedGEMMKernelDispatcher):
    _backends = {
        BackendType.CK: BackendEntry(GroupedGEMMFP8CKBackend),
        BackendType.HIPBLASLT: BackendEntry(GroupedGEMMFP8HipblasltBackend, autotune=False),
        BackendType.TRITON: BackendEntry(GroupedGEMMFP8TritonBackend),
        BackendType.FLYDSL: BackendEntry(GroupedGEMMFP8FlyDSLBackend),
    }
    _cache = TuneCache(1024)

    @classmethod
    def make_key(
        cls,
        a,
        b,
        a_scales,
        b_scales,
        group_lens,
        group_offs,
        trans_a,
        trans_b,
        out_dtype,
        granularity,
        num_cu,
        **kwargs,
    ):
        bs = b.shape[0]
        m = a.shape[1] if trans_a else a.shape[0]
        n = b.shape[-2] if trans_b else b.shape[-1]
        k = a.shape[0] if trans_a else a.shape[1]
        # bs, m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c, granularity
        return (bs, m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, False, granularity)


class GroupedGEMMFP8VariableKTritonBackend(KernelBackend):
    """Triton persistent-kernel backend for FP8 variable-K grouped GEMM (backward).

    Supports:
      - TENSORWISE: per-tensor scaling, including HYBRID format
      - ROWWISE: per-row/per-col vector scaling
      - BLOCKWISE: 1D+1D block-wise scaling (TN/CRR layout)
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
        ScalingGranularity.MX_BLOCKWISE,
    }

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
        inplace_add_to_out: bool = False,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= granularity in GroupedGEMMFP8VariableKTritonBackend.SUPPORTED_GRANULARITIES
        if inplace_add_to_out:
            supported &= granularity in (
                ScalingGranularity.TENSORWISE,
                ScalingGranularity.MX_BLOCKWISE,
            )
        if granularity != ScalingGranularity.MX_BLOCKWISE:
            supported &= (
                a.dtype,
                b.dtype,
                out_dtype,
            ) in GroupedGEMMFP8VariableKTritonBackend.SUPPORTED_DTYPES
            supported &= trans_a and not trans_b
        else:
            # MXFP8 variable-K wgrad: both operands fp8 (e4m3/e5m2), and the kernel
            # expects the non-transposed (OUT_M, M_total) / (OUT_N, M_total) layout.
            supported &= not is_gfx942()
            supported &= a.dtype in (float8_e4m3, float8_e5m2)
            supported &= b.dtype in (float8_e4m3, float8_e5m2)
            supported &= out_dtype in (torch.float16, torch.bfloat16)
            supported &= not trans_a and not trans_b
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
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ):
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales

        beta = 1.0 if inplace_add_to_out else 0.0

        if granularity == ScalingGranularity.MX_BLOCKWISE:
            # wgrad: C[g](OUT_M,OUT_N) = lhs[:,g](OUT_M,M_g) @ rhs[:,g](OUT_N,M_g)^T
            # lhs = grad_out_col (OUT_M=N, M_total), rhs = a_col (OUT_N=K, M_total).
            # group_offs = padded per-group offsets along M.
            OUT_M = lhs.shape[0]
            OUT_N = rhs.shape[0]
            G = group_lens.shape[0]
            return grouped_gemm_mxfp8_variable_k_triton_kernel(
                lhs,
                lhs_scales,
                rhs,
                rhs_scales,
                group_offs,
                OUT_M,
                OUT_N,
                G,
                out_dtype=out_dtype,
                num_cu=num_cu,
                out=out,
                beta=beta,
            )
        if granularity == ScalingGranularity.BLOCKWISE:
            return grouped_gemm_fp8_blockwise_variable_k_triton_kernel(
                lhs,
                rhs,
                lhs_scales,
                rhs_scales,
                group_offs,
                out_dtype=out_dtype,
                out=out,
                beta=beta,
            )
        elif granularity == ScalingGranularity.ROWWISE:
            return grouped_gemm_fp8_rowwise_variable_k_triton_kernel(
                lhs,
                rhs,
                lhs_scales,
                rhs_scales,
                group_offs,
                out_dtype=out_dtype,
                out=out,
                beta=beta,
            )
        return grouped_gemm_fp8_tensorwise_variable_k_triton_kernel(
            lhs,
            rhs,
            lhs_scales,
            rhs_scales,
            group_offs,
            out_dtype=out_dtype,
            out=out,
            beta=beta,
            m_real=kwargs.get("m_real", None),
            n_real=kwargs.get("n_real", None),
        )


class GroupedGEMMFP8VariableKFlyDSLBackend(KernelBackend):
    """FlyDSL fp8 variable-K grouped GEMM backend (gfx950).

    wgrad: C[g] = lhs[offs[g]:offs[g+1]]^T @ rhs[offs[g]:offs[g+1]], contraction
    = m_g (variable per group) via a runtime scf.for K-loop. Supports TENSORWISE
    (trans_a=True, TN) and MX_BLOCKWISE (non-transposed operands, TN) via the
    FlyDSL mfma[_scale]_f32_16x16x128_f8f6f4 TN kernel (gfx950-only).
    """

    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE, ScalingGranularity.MX_BLOCKWISE}
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
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        supported = True
        if inplace_add_to_out:
            # Fused bgrad-accum: out_dtype stays the (16-bit) weight dtype and gates the fp8
            # input pair below; the accumulate target ``out`` (Megatron main_grad) may be fp32.
            # The store dtype is derived from ``out.dtype`` in execute -- fp32 is written
            # natively by the scalar epilogue (tensorwise only; MX stays 16-bit).
            _out_allowed = (
                (torch.bfloat16, torch.float16)
                if granularity == ScalingGranularity.MX_BLOCKWISE
                else (torch.bfloat16, torch.float16, torch.float32)
            )
            supported &= out is not None and out.dtype in _out_allowed
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8VariableKFlyDSLBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8VariableKFlyDSLBackend.SUPPORTED_GRANULARITIES
        # gfx950 (CDNA4) only: kernel uses mfma[_scale]_f32_16x16x128_f8f6f4.
        supported &= is_gfx950()

        if granularity == ScalingGranularity.MX_BLOCKWISE:
            # MXFP8 wgrad: non-transposed operands (TN), contraction M_total % 128 == 0.
            supported &= (not trans_a) and (not trans_b)
            supported &= a.shape[1] % 128 == 0
            return supported

        # TENSORWISE: variable-K contraction along shared rows; per-tensor scalars.
        supported &= trans_a and not trans_b
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
        trans_c: bool,
        out_dtype: torch.dtype,
        granularity: ScalingGranularity,
        num_cu: int | None,
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ):
        from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_kernel import (
            grouped_gemm_fp8_variable_k_tensorwise_flydsl_kernel,
        )

        # trans_c swaps which operand is lhs (output transpose): out[g] = lhs[g]^T @ rhs[g].
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales

        beta = 1.0 if inplace_add_to_out else 0.0
        accum_out = out if inplace_add_to_out else None

        # tensorwise N/K-pad: m_real=OUT_M (real N), n_real=OUT_N (real K) shrink the C
        # extents from the padded operand pitch to the tight output. tight now accumulates
        # natively in the scalar epilogue (beta=1 read-back rides the tight band SRD), so
        # fused bgrad-accum folds straight into the tight fp32 main_grad -- no host add_.
        m_real = kwargs.get("m_real", None)
        n_real = kwargs.get("n_real", None)
        c_tight = m_real is not None or n_real is not None

        if granularity == ScalingGranularity.MX_BLOCKWISE:
            from primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp8_kernel import (
                grouped_gemm_mxfp8_variable_k_flydsl_kernel,
            )

            OUT_M = lhs.shape[0]
            OUT_N = rhs.shape[0]
            G = group_lens.shape[0]
            return grouped_gemm_mxfp8_variable_k_flydsl_kernel(
                lhs,
                lhs_scales,
                rhs,
                rhs_scales,
                group_offs,
                OUT_M,
                OUT_N,
                G,
                out_dtype=out_dtype,
                num_cu=num_cu,
                beta=beta,
                out=accum_out,
            )

        # When accumulating, the store width follows the real target (main_grad, possibly
        # fp32); out_dtype only carried the 16-bit weight dtype for input gating.
        store_dtype = accum_out.dtype if accum_out is not None else out_dtype
        return grouped_gemm_fp8_variable_k_tensorwise_flydsl_kernel(
            lhs,
            rhs,
            lhs_scales,
            rhs_scales,
            group_offs,
            out_dtype=store_dtype,
            num_cu=num_cu,
            beta=beta,
            out=accum_out,
            m_real=m_real,
            n_real=n_real,
            c_tight=c_tight,
        )


class GroupedGEMMFP8VariableKKernelDispatcher(BaseGroupedGEMMVariableKKernelDispatcher):
    _backends = {
        BackendType.CK: BackendEntry(GroupedGEMMFP8VariableKCKBackend),
        BackendType.HIPBLASLT: BackendEntry(GroupedGEMMFP8VariableKHipblasltBackend),
        BackendType.TRITON: BackendEntry(GroupedGEMMFP8VariableKTritonBackend),
        BackendType.FLYDSL: BackendEntry(GroupedGEMMFP8VariableKFlyDSLBackend),
    }
    _cache = TuneCache(1024)

    @classmethod
    def make_key(
        cls,
        a,
        b,
        a_scales,
        b_scales,
        group_lens,
        group_offs,
        trans_a,
        trans_b,
        trans_c,
        out_dtype,
        granularity,
        num_cu,
        **kwargs,
    ):
        bs = group_lens.shape[0]
        m = a.shape[1] if trans_a else a.shape[0]
        n = b.shape[-2] if trans_b else b.shape[-1]
        k = a.shape[0] if trans_a else a.shape[1]
        if trans_c:
            m, n = n, m
        return (bs, m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c, granularity)


_torch_custom_op_wrapper = torch.library.custom_op


@_torch_custom_op_wrapper("primus_turbo::grouped_gemm_fp8_impl", mutates_args=(), device_types="cuda")
def grouped_gemm_fp8_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    granularity: int,
    num_cu: int | None,
    default_backend: int,
    maybe_pre_sync: bool = False,
    group_offs_out: torch.Tensor | None = None,
    n_real: int | None = None,
) -> torch.Tensor:
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8)
    granularity_enum = ScalingGranularity(granularity)

    kwargs = dict(
        a=a,
        b=b,
        a_scales=a_scales,
        b_scales=b_scales,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=trans_a,
        trans_b=trans_b,
        out_dtype=out_dtype,
        granularity=granularity_enum,
        num_cu=num_cu,
        maybe_pre_sync=maybe_pre_sync,
        group_offs_out=group_offs_out,
        n_real=n_real,
    )

    out = GroupedGEMMFP8KernelDispatcher.dispatch(default_backend_choice, user_backend_choice, **kwargs)
    # Over-allocated output: zero the unwritten tail past the tight write bound
    # (group_offs_out for MX; group_offs otherwise) so the caller's [:total_m]
    # slice never exposes uninitialized rows.
    out = grouped_gemm_output_tail_kernel(out, group_offs_out if group_offs_out is not None else group_offs)
    return out


@_torch_custom_op_wrapper(
    "primus_turbo::grouped_gemm_fp8_variable_k_impl", mutates_args=(), device_types="cuda"
)
def grouped_gemm_fp8_variable_k_impl(
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
    granularity: int,
    num_cu: int | None,
    default_backend: int,
    maybe_pre_sync: bool = False,
    m_real: int | None = None,
    n_real: int | None = None,
) -> torch.Tensor:
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8)
    granularity_enum = ScalingGranularity(granularity)

    kwargs = dict(
        a=a,
        b=b,
        a_scales=a_scales,
        b_scales=b_scales,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=trans_a,
        trans_b=trans_b,
        trans_c=trans_c,
        out_dtype=out_dtype,
        granularity=granularity_enum,
        num_cu=num_cu,
        maybe_pre_sync=maybe_pre_sync,
        m_real=m_real,
        n_real=n_real,
    )

    return GroupedGEMMFP8VariableKKernelDispatcher.dispatch(
        default_backend_choice, user_backend_choice, **kwargs
    )


@_torch_custom_op_wrapper(
    "primus_turbo::grouped_gemm_fp8_variable_k_accum_impl", mutates_args={"out"}, device_types="cuda"
)
def grouped_gemm_fp8_variable_k_accum_impl(
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
    granularity: int,
    num_cu: int | None,
    default_backend: int,
    out: torch.Tensor,
    maybe_pre_sync: bool = False,
    m_real: int | None = None,
    n_real: int | None = None,
) -> None:
    """Variable-K grouped FP8 GEMM that accumulates into ``out`` instead of returning.

    Computes ``out += A^T @ B`` per group, folding the accumulation into the GEMM
    epilogue (beta=1)
    """
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8)
    granularity_enum = ScalingGranularity(granularity)

    kwargs = dict(
        a=a,
        b=b,
        a_scales=a_scales,
        b_scales=b_scales,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=trans_a,
        trans_b=trans_b,
        trans_c=trans_c,
        out_dtype=out_dtype,
        granularity=granularity_enum,
        num_cu=num_cu,
        maybe_pre_sync=maybe_pre_sync,
        inplace_add_to_out=True,
        out=out,
        m_real=m_real,
        n_real=n_real,
    )

    if (
        GlobalBackendManager.auto_tune_enabled()
        and not GroupedGEMMFP8VariableKKernelDispatcher._is_graph_capturing()
    ):
        GroupedGEMMFP8VariableKKernelDispatcher.tune(**{**kwargs, "out": torch.zeros_like(out)})

    GroupedGEMMFP8VariableKKernelDispatcher.dispatch(default_backend_choice, user_backend_choice, **kwargs)


@grouped_gemm_fp8_variable_k_accum_impl.register_fake
def grouped_gemm_fp8_variable_k_accum_impl_meta(
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
    granularity: int,
    num_cu: int | None,
    default_backend: int,
    out: torch.Tensor,
    maybe_pre_sync: bool = False,
    m_real: int | None = None,
    n_real: int | None = None,
) -> None:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 2, f"b must be 2D, got {b.shape}"
    assert out.dim() == 3, f"out must be 3D, got {out.shape}"
    return None


@grouped_gemm_fp8_impl.register_fake
def grouped_gemm_fp8_impl_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    granularity: int,
    num_cu: int | None,
    default_backend: int,
    maybe_pre_sync: bool = False,
    group_offs_out: torch.Tensor | None = None,
    n_real: int | None = None,
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype in [float8_e4m3, float8_e5m2], f"a must be fp8, got {a.dtype}"
    assert b.dtype in [float8_e4m3, float8_e5m2], f"b must be fp8, got {b.dtype}"
    assert out_dtype in [
        torch.float16,
        torch.bfloat16,
    ], f"out_dtype must be float16 or bfloat16, got {out_dtype}"
    assert trans_a == False, "Only trans_a=False is supported."

    # MX over-allocates to the padded input rows; group_offs_out maps each group
    # into the tight layout and the caller slices [:total_m]. n_real (tensorwise
    # N/K-pad) shrinks the output free dim from the padded operand pitch to the
    # real extent, so the returned tensor is already tight.
    m = a.shape[1] if trans_a else a.shape[0]
    n = n_real if n_real is not None else (b.shape[-2] if trans_b else b.shape[-1])
    return torch.empty((m, n), device=a.device, dtype=out_dtype)


@grouped_gemm_fp8_variable_k_impl.register_fake
def grouped_gemm_fp8_variable_k_impl_meta(
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
    granularity: int,
    num_cu: int | None,
    default_backend: int,
    maybe_pre_sync: bool = False,
    m_real: int | None = None,
    n_real: int | None = None,
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 2, f"b must be 2D, got {b.shape}"
    assert a.dtype in [float8_e4m3, float8_e5m2], f"a must be fp8, got {a.dtype}"
    assert b.dtype in [float8_e4m3, float8_e5m2], f"b must be fp8, got {b.dtype}"
    assert out_dtype in [
        torch.float16,
        torch.bfloat16,
    ], f"out_dtype must be float16 or bfloat16, got {out_dtype}"

    bs = group_lens.shape[0]
    if ScalingGranularity(granularity) == ScalingGranularity.MX_BLOCKWISE:
        # MX wgrad: C[g] (OUT_M, OUT_N) = lhs[:,g] @ rhs[:,g]^T, lhs/rhs swapped by
        # trans_c (matches the eager path). Output (G, OUT_M, OUT_N).
        lhs, rhs = (b, a) if trans_c else (a, b)
        return torch.empty((bs, lhs.shape[0], rhs.shape[0]), device=a.device, dtype=out_dtype)

    assert trans_a and not trans_b, "Only trans_a=True and trans_b=False are supported."
    m = a.shape[1] if trans_a else a.shape[0]
    n = b.shape[-2] if trans_b else b.shape[-1]
    if trans_c:
        m, n = n, m
    # tensorwise N/K-pad: C output free dims shrink from the padded operand pitch
    # to the real (m_real=N, n_real=K) extents so grad_b comes back tight [G, N, K].
    if m_real is not None:
        m = m_real
    if n_real is not None:
        n = n_real
    return torch.empty((bs, m, n), device=a.device, dtype=out_dtype)


def _check_glu_dispatch(granularity: int, trans_a: bool) -> None:
    """The fused GLU epilogue has one scaling mode and one A layout."""
    assert ScalingGranularity(granularity) == ScalingGranularity.TENSORWISE, (
        f"Fused GLU grouped GEMM is tensorwise-only, got {ScalingGranularity(granularity)}"
    )
    assert not trans_a, "Fused GLU grouped GEMM does not support trans_a"


def _alloc_grad_probs_partial(spec, device: torch.device) -> torch.Tensor:
    """The grad_probs partial buffer a fused dgrad asked for.

    The kernels allocate nothing, and both the shape and ``needs_zero`` follow
    from a tiling only they know, so the spec is asked for rather than
    reconstructed here.
    """
    fill = torch.zeros if spec.needs_zero else torch.empty
    return fill(spec.shape, device=device, dtype=torch.float32)


@_torch_custom_op_wrapper("primus_turbo::grouped_gemm_fp8_glu_impl", mutates_args=(), device_types="cuda")
def grouped_gemm_fp8_glu_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,  # not used
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    granularity: int,
    num_cu: int | None,
    probs: torch.Tensor,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """fc1 grouped GEMM with the GLU activation fused into its epilogue.

    Returns ``(act, intermediate)``: the activation [M, N // 2] that feeds fc2,
    and the pre-activation [M, N] that the backward needs. ``probs`` is folded
    into ``act`` only -- ``intermediate`` is always the unscaled GEMM output.
    """
    _check_glu_dispatch(granularity, trans_a)

    assert is_gfx950(), "grouped_gemm_fp8_glu_impl is only supported on gfx950"
    assert trans_b == True, "trans_b must be True for grouped_gemm_fp8_glu_impl"
    assert a.shape[1] >= 129, "a.shape[1] must be >= 129 for grouped_gemm_fp8_glu_impl"

    M = a.shape[0]
    N = b.shape[1] if trans_b else b.shape[2]

    act = torch.empty((M, N // 2), device=a.device, dtype=out_dtype)
    intermediate = torch.empty((M, N), device=a.device, dtype=out_dtype)

    from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_glu_kernel import (
        grouped_gemm_fp8_glu_tensorwise_flydsl_kernel,
    )

    return grouped_gemm_fp8_glu_tensorwise_flydsl_kernel(
        a,
        b,
        a_scales,
        b_scales,
        probs,
        group_offs,
        act,
        intermediate,
        trans_b=trans_b,
        activation=activation,
        out_dtype=out_dtype,
        num_cu=num_cu,
    )


@_torch_custom_op_wrapper("primus_turbo::grouped_gemm_fp8_dglu_impl", mutates_args=(), device_types="cuda")
def grouped_gemm_fp8_dglu_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,  # not used
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    granularity: int,
    num_cu: int | None,
    probs: torch.Tensor,
    intermediate: torch.Tensor,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """fc2 dgrad with the GLU activation gradient fused into its epilogue.

    ``a`` is the fc2 output gradient and ``b`` the fc2 weight oriented for the
    dgrad, so ``a @ b`` is the gradient wrt the activation; the epilogue consumes
    it in registers against ``intermediate`` (the pre-activation the forward
    wrote).

    Returns ``(grad_intermediate, grad_probs)``: the pre-activation gradient
    [M, 2 * N], and [M] fp32 the gradient wrt the routing probabilities -- the
    gradient wrt the activation, taken before the probs scaling, times the
    activation, summed over the hidden dimension. The kernels leave that sum as
    per-tile partials and this function folds them.
    """
    _check_glu_dispatch(granularity, trans_a)

    assert is_gfx950(), "grouped_gemm_fp8_dglu_impl is only supported on gfx950"
    assert trans_b == False, "trans_b must be False for grouped_gemm_fp8_dglu_impl"
    assert a.shape[1] >= 129, "a.shape[1] must be >= 129 for grouped_gemm_fp8_dglu_impl"

    M = a.shape[0]
    N = b.shape[1] if trans_b else b.shape[2]

    from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_glu_kernel import (
        grouped_gemm_fp8_dglu_grad_probs_partial_spec,
        grouped_gemm_fp8_dglu_tensorwise_flydsl_kernel,
    )

    out = torch.empty((M, N * 2), device=a.device, dtype=out_dtype)
    grad_probs_partial = _alloc_grad_probs_partial(
        grouped_gemm_fp8_dglu_grad_probs_partial_spec(a, b), a.device
    )
    grouped_gemm_fp8_dglu_tensorwise_flydsl_kernel(
        a,
        b,
        a_scales,
        b_scales,
        intermediate,
        group_offs,
        probs,
        out,
        grad_probs_partial,
        trans_b=trans_b,
        activation=activation,
        num_cu=num_cu,
    )

    return out, torch.sum(grad_probs_partial, dim=0)


@grouped_gemm_fp8_glu_impl.register_fake
def grouped_gemm_fp8_glu_impl_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    granularity: int,
    num_cu: int | None,
    probs: torch.Tensor,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_glu_dispatch(granularity, trans_a)

    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype in [float8_e4m3, float8_e5m2], f"a must be fp8, got {a.dtype}"
    assert b.dtype in [float8_e4m3, float8_e5m2], f"b must be fp8, got {b.dtype}"
    assert out_dtype in [
        torch.float16,
        torch.bfloat16,
    ], f"out_dtype must be float16 or bfloat16, got {out_dtype}"

    # b holds gate||up, so its N is twice the activation's width.
    m = a.shape[0]
    n = b.shape[1] if trans_b else b.shape[2]
    act = torch.empty((m, n // 2), device=a.device, dtype=out_dtype)
    intermediate = torch.empty((m, n), device=a.device, dtype=out_dtype)
    return act, intermediate


@grouped_gemm_fp8_dglu_impl.register_fake
def grouped_gemm_fp8_dglu_impl_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    granularity: int,
    num_cu: int | None,
    probs: torch.Tensor,
    intermediate: torch.Tensor,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_glu_dispatch(granularity, trans_a)

    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype in [float8_e4m3, float8_e5m2], f"a must be fp8, got {a.dtype}"
    assert b.dtype in [float8_e4m3, float8_e5m2], f"b must be fp8, got {b.dtype}"
    assert out_dtype in [
        torch.float16,
        torch.bfloat16,
    ], f"out_dtype must be float16 or bfloat16, got {out_dtype}"

    # This is the dgrad wrt the activation, so the gate||up gradient it writes
    # is twice as wide. grad_probs is per-row and always fp32, whatever out_dtype is.
    m = a.shape[0]
    n = b.shape[1] if trans_b else b.shape[2]
    grad_intermediate = torch.empty((m, n * 2), device=a.device, dtype=out_dtype)
    grad_probs = torch.empty((m,), device=a.device, dtype=torch.float32)
    return grad_intermediate, grad_probs
