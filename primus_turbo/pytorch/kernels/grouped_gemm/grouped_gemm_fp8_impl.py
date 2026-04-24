###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus_turbo.pytorch.core.backend import (
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
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_kernel import (
    grouped_gemm_mxfp8_triton_kernel,
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
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
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
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8CKBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8CKBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
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
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
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
        **kwargs,
    ) -> bool:
        supported = True
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
    # NOTE: MX_BLOCKWISE works correctness-wise via the patched C++ binding
    # (hipblaslt_grouped_gemm.cpp accepts MX_BLOCKWISE → VEC32_UE8M0 scale
    # mode). Disabled here because Primus-Turbo's grouped wrapper does
    # per-expert hipblasLtMatmul calls with 4-stream pingpong → ~6× slower
    # than Triton at gpt_oss_20B fwd (9.3 vs 1.5 ms). To re-enable, port the
    # binding to use hipblaslt_ext::GroupedGemm (single-launch Tensile).
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
    # NOTE: MX_BLOCKWISE for variable-K (wgrad) does not work via hipBLASLt's
    # VEC32_UE8M0 scale mode. The MX spec needs scales along the reduction
    # axis (M for wgrad), but VEC32_UE8M0 expects scales along the innermost
    # dim of the operand. With opA=T (the only opA supported by MX-FP8 in
    # hipBLASLt), the innermost dim of A equals the OUTPUT row dim, NOT the
    # reduction. Verified empirically: routing MX wgrad through this backend
    # produces 7.4 dB SNR (garbage). hipBLASLt would need to support a
    # "scales along K" / "scales along reduction" mode for wgrad to work.
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
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8VariableKHipblasltBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8VariableKHipblasltBackend.SUPPORTED_GRANULARITIES
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
        maybe_pre_sync: bool = False,
    ):
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
            trans_lhs, trans_rhs = not trans_b, not trans_a
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales
            trans_lhs, trans_rhs = trans_a, trans_b
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
        )


class GroupedGEMMFP8TritonBackend(KernelBackend):
    """Triton persistent-kernel backend for FP8 grouped GEMM (CPU-sync-free).

    Supports:
      - TENSORWISE: per-tensor scaling
      - ROWWISE: per-row/per-col vector scaling
      - BLOCKWISE: 128-block FP32 scaling (post-MFMA apply)
      - MX_BLOCKWISE: 32-block E8M0 scaling via ``tl.dot_scaled`` (native
        ``v_mfma_scale_f32_32x32x64_f8f6f4`` on gfx950)
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
        ScalingGranularity.MX_BLOCKWISE,
    }

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES)

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
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8TritonBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8TritonBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
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
            return grouped_gemm_mxfp8_triton_kernel(
                a,
                b,
                a_scales,
                b_scales,
                group_offs,
                trans_b=trans_b,
                out_dtype=out_dtype,
            )
        elif granularity == ScalingGranularity.BLOCKWISE:
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
        )


class GroupedGEMMFP8KernelDispatcher(BaseGroupedGEMMKernelDispatcher):
    _backends = {
        BackendType.CK: BackendEntry(GroupedGEMMFP8CKBackend),
        BackendType.HIPBLASLT: BackendEntry(GroupedGEMMFP8HipblasltBackend, autotune=False),
        BackendType.TRITON: BackendEntry(GroupedGEMMFP8TritonBackend),
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


class GroupedGEMMFP8VariableKHipBackend(KernelBackend):
    """HIP-native variable-K grouped-GEMM backend (MX-FP8 only, v1).

    Drop-in counterpart to ``GroupedGEMMFP8VariableKHipblasltBackend`` for the
    ``MX_BLOCKWISE`` granularity that the C++ hipBLASLt binding does not yet
    handle. Wraps the HIP fwd kernel (reused with operand role-swap) via a
    single-launch stacked dispatch.

    See ``primus_turbo/hip/grouped_gemm_mxfp8/__init__.py::grouped_gemm_mxfp8_hip_variable_k``
    for implementation details and constraints (balanced MoE, M_g >= 384,
    M_g % 128 == 0).
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.MX_BLOCKWISE,
    }

    # MX-FP8 is e4m3 only in current Primus-Turbo pipeline.
    SUPPORTED_DTYPES = {
        (float8_e4m3, float8_e4m3, torch.float16),
        (float8_e4m3, float8_e4m3, torch.bfloat16),
    }

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
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8VariableKHipBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8VariableKHipBackend.SUPPORTED_GRANULARITIES
        supported &= trans_a and not trans_b
        if not supported:
            return False
        # v1 only runs on gfx950 (MFMA scale instruction availability).
        supported &= torch.cuda.is_available() and (
            "gfx950" in torch.cuda.get_device_properties(a.device).gcnArchName
        )
        if not supported:
            return False
        # v1 requires balanced MoE with M_g multiple-of-128 and >= 384.
        g = int(group_lens.numel())
        M_total = int(a.size(0)) if not trans_a else int(a.size(0))  # M = reduction axis = a.size(0)
        if M_total % g != 0:
            return False
        M_g = M_total // g
        if M_g < 384 or M_g % 128 != 0:
            return False
        return True

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
    ):
        # Match the other VariableK backends' role-swap convention.
        from primus_turbo.hip.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_variable_k
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales
        # Our HIP entry point treats "lhs" as the operand whose cols become
        # output rows; trans_c already handled above, so pass trans_c=False.
        return grouped_gemm_mxfp8_hip_variable_k(
            lhs, rhs, lhs_scales, rhs_scales,
            group_offs,
            out_dtype=out_dtype,
            trans_c=False,
        )


class GroupedGEMMFP8VariableKTritonBackend(KernelBackend):
    """Triton persistent-kernel backend for FP8 variable-K grouped GEMM (backward).

    Supports:
      - TENSORWISE: per-tensor scaling
      - ROWWISE: per-row/per-col vector scaling
      - BLOCKWISE: 1D+1D block-wise scaling (TN/CRR layout)
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
    }

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES)

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
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8VariableKTritonBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8VariableKTritonBackend.SUPPORTED_GRANULARITIES
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
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales

        if granularity == ScalingGranularity.BLOCKWISE:
            return grouped_gemm_fp8_blockwise_variable_k_triton_kernel(
                lhs,
                rhs,
                lhs_scales,
                rhs_scales,
                group_offs,
                out_dtype=out_dtype,
            )
        elif granularity == ScalingGranularity.ROWWISE:
            return grouped_gemm_fp8_rowwise_variable_k_triton_kernel(
                lhs,
                rhs,
                lhs_scales,
                rhs_scales,
                group_offs,
                out_dtype=out_dtype,
            )
        return grouped_gemm_fp8_tensorwise_variable_k_triton_kernel(
            lhs,
            rhs,
            lhs_scales,
            rhs_scales,
            group_offs,
            out_dtype=out_dtype,
        )


class GroupedGEMMFP8VariableKKernelDispatcher(BaseGroupedGEMMVariableKKernelDispatcher):
    _backends = {
        BackendType.CK: BackendEntry(GroupedGEMMFP8VariableKCKBackend),
        BackendType.HIPBLASLT: BackendEntry(GroupedGEMMFP8VariableKHipblasltBackend, autotune=False),
        BackendType.TURBO: BackendEntry(GroupedGEMMFP8VariableKHipBackend, autotune=False),
        BackendType.TRITON: BackendEntry(GroupedGEMMFP8VariableKTritonBackend),
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
) -> torch.Tensor:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8)
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
    )

    return GroupedGEMMFP8KernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


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
) -> torch.Tensor:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP8)
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
    )

    return GroupedGEMMFP8VariableKKernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


def grouped_gemm_compute_offs(group_lens: torch.Tensor) -> torch.Tensor:
    group_offs = torch.ops.primus_turbo_cpp_extension.grouped_gemm_compute_offs(group_lens)
    return group_offs


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

    m = a.shape[1] if trans_a else a.shape[0]
    n = b.shape[-2] if trans_b else b.shape[-1]
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
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 2, f"b must be 2D, got {b.shape}"
    assert a.dtype in [float8_e4m3, float8_e5m2], f"a must be fp8, got {a.dtype}"
    assert b.dtype in [float8_e4m3, float8_e5m2], f"b must be fp8, got {b.dtype}"
    assert out_dtype in [
        torch.float16,
        torch.bfloat16,
    ], f"out_dtype must be float16 or bfloat16, got {out_dtype}"
    assert trans_a and not trans_b, "Only trans_a=True and trans_b=False are supported."

    bs = group_lens.shape[0]
    m = a.shape[1] if trans_a else a.shape[0]
    n = b.shape[-2] if trans_b else b.shape[-1]
    if trans_c:
        m, n = n, m
    return torch.empty((bs, m, n), device=a.device, dtype=out_dtype)
