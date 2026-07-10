###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP4 grouped GEMM dispatch (Triton-only, gfx950).

Mirrors the FP8 grouped GEMM dispatch (``grouped_gemm_fp8_impl``): the
``torch.library`` custom ops keep the same argument lists and select the kernel
through an :class:`AutoKernelDispatcher`, so the backend is controlled the same
way (env / ``GlobalBackendManager`` / autotune / default). Unlike FP8 (which
fans out to CK / hipBLASLt / FlyDSL), MXFP4 grouped GEMM currently has a single
Triton block-scaled backend, so only ``BackendType.TRITON`` is registered.

The forward op over-allocates the output to the (padded) input rows and the
caller slices ``[:total_m]``; ``group_offs_out`` packs each group tight.
"""

import torch

from primus_turbo.pytorch.core.backend import (
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float4_e2m1fn_x2
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    BaseGroupedGEMMKernelDispatcher,
    BaseGroupedGEMMVariableKKernelDispatcher,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp4_kernel import (
    grouped_gemm_mxfp4_triton_kernel,
    grouped_gemm_mxfp4_variable_k_triton_kernel,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    grouped_gemm_output_tail_kernel,
)


class GroupedGEMMFP4TritonBackend(KernelBackend):
    """Triton persistent-kernel backend for MXFP4 grouped GEMM (forward / dgrad).

    MX_BLOCKWISE only, NT layout (trans_b=True). ``b`` is FP4-packed (G, N, K/2),
    so the logical contraction ``K`` is ``b.shape[-1] * 2`` (already zero-padded
    to BLOCK_SIZE_K=128 by the quantizer) and the free dim ``N`` is ``b.shape[-2]``.
    """

    SUPPORTED_GRANULARITIES = {ScalingGranularity.MX_BLOCKWISE}

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
        supported &= granularity in GroupedGEMMFP4TritonBackend.SUPPORTED_GRANULARITIES
        supported &= a.dtype == float4_e2m1fn_x2 and b.dtype == float4_e2m1fn_x2
        supported &= out_dtype in (torch.float16, torch.bfloat16)
        # NT only: the kernel infers the contraction from the packed last dim.
        supported &= trans_b and not trans_a
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
        # b is (G, N, K/2) NT, FP4-packed. group_offs = padded read offsets;
        # group_offs_out = real write offsets (output over-allocated to padded
        # rows, sliced by the caller).
        N = b.shape[-2]
        K = b.shape[-1] * 2
        group_offs_out = kwargs.get("group_offs_out", None)
        return grouped_gemm_mxfp4_triton_kernel(
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


class GroupedGEMMFP4KernelDispatcher(BaseGroupedGEMMKernelDispatcher):
    _backends = {
        BackendType.TRITON: BackendEntry(GroupedGEMMFP4TritonBackend),
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
        return (bs, m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, False, granularity)


class GroupedGEMMFP4VariableKTritonBackend(KernelBackend):
    """Triton persistent-kernel backend for MXFP4 variable-K grouped GEMM (wgrad).

    MX_BLOCKWISE only. Both operands are FP4-packed 2D ``(OUT_*, M_total/2)`` and
    the kernel reduces over the (padded) per-group M, so it expects the
    non-transposed layout (``not trans_a and not trans_b``).
    """

    SUPPORTED_GRANULARITIES = {ScalingGranularity.MX_BLOCKWISE}

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
        supported &= granularity in GroupedGEMMFP4VariableKTritonBackend.SUPPORTED_GRANULARITIES
        supported &= a.dtype == float4_e2m1fn_x2 and b.dtype == float4_e2m1fn_x2
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
        **kwargs,
    ):
        # wgrad: C[g] (OUT_M, OUT_N) = lhs[:,g] @ rhs[:,g]^T, reduction over M.
        # lhs/rhs swapped by trans_c (matches the FP8 variable-K backend).
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales
        OUT_M = lhs.shape[0]
        OUT_N = rhs.shape[0]
        G = group_lens.shape[0]
        return grouped_gemm_mxfp4_variable_k_triton_kernel(
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
        )


class GroupedGEMMFP4VariableKKernelDispatcher(BaseGroupedGEMMVariableKKernelDispatcher):
    _backends = {
        BackendType.TRITON: BackendEntry(GroupedGEMMFP4VariableKTritonBackend),
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


@_torch_custom_op_wrapper("primus_turbo::grouped_gemm_fp4_impl", mutates_args=(), device_types="cuda")
def grouped_gemm_fp4_impl(
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
) -> torch.Tensor:
    """Forward / dgrad (NT): C[g] = A[g] @ B[g]^T, contraction over (packed) K."""
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP4)
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
    )

    out = GroupedGEMMFP4KernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)
    # Over-allocated output: zero the unwritten tail past the tight write bound
    # (group_offs_out for MX; group_offs otherwise) so the caller's [:total_m]
    # slice never exposes uninitialized rows.
    out = grouped_gemm_output_tail_kernel(out, group_offs_out if group_offs_out is not None else group_offs)
    return out


@_torch_custom_op_wrapper(
    "primus_turbo::grouped_gemm_fp4_variable_k_impl", mutates_args=(), device_types="cuda"
)
def grouped_gemm_fp4_variable_k_impl(
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
    """Backward wgrad: C[g] (OUT_M, OUT_N) = lhs[:,g] @ rhs[:,g]^T, reduction over M."""
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP4)
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

    return GroupedGEMMFP4VariableKKernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


@grouped_gemm_fp4_impl.register_fake
def grouped_gemm_fp4_impl_meta(
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
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype == float4_e2m1fn_x2, f"a must be fp4, got {a.dtype}"
    assert b.dtype == float4_e2m1fn_x2, f"b must be fp4, got {b.dtype}"
    assert out_dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"out_dtype must be float16 or bfloat16, got {out_dtype}"
    assert trans_a == False, "Only trans_a=False is supported."

    # MX over-allocates to the padded input rows; group_offs_out maps each group
    # into the tight layout and the caller slices [:total_m].
    m = a.shape[1] if trans_a else a.shape[0]
    n = b.shape[-2] if trans_b else b.shape[-1]
    return torch.empty((m, n), device=a.device, dtype=out_dtype)


@grouped_gemm_fp4_variable_k_impl.register_fake
def grouped_gemm_fp4_variable_k_impl_meta(
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
    assert a.dtype == float4_e2m1fn_x2, f"a must be fp4, got {a.dtype}"
    assert b.dtype == float4_e2m1fn_x2, f"b must be fp4, got {b.dtype}"
    assert out_dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"out_dtype must be float16 or bfloat16, got {out_dtype}"

    # wgrad: C[g] (OUT_M, OUT_N) = lhs[:,g] @ rhs[:,g]^T, lhs/rhs swapped by
    # trans_c (matches the eager path). Output (G, OUT_M, OUT_N).
    bs = group_lens.shape[0]
    lhs, rhs = (b, a) if trans_c else (a, b)
    return torch.empty((bs, lhs.shape[0], rhs.shape[0]), device=a.device, dtype=out_dtype)
