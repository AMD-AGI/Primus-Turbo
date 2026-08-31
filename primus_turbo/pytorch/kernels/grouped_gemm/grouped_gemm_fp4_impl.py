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
    BackendChoice,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.core.low_precision import (
    MXFP4_BLOCK_SIZE,
    Float4QuantConfig,
    ScalingGranularity,
    ScalingRecipe,
    float4_e2m1fn_x2,
)
from primus_turbo.pytorch.core.utils import is_gfx942, is_gfx950
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils import (
    BaseGroupedGEMMKernelDispatcher,
    BaseGroupedGEMMVariableKKernelDispatcher,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp4_kernel import (
    grouped_gemm_mxfp4_triton_kernel,
    grouped_gemm_mxfp4_variable_k_triton_kernel,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_helper import (
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
        if granularity == ScalingGranularity.MX_BLOCKWISE:
            supported &= not is_gfx942()
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


class GroupedGEMMFP4FlyDSLBackend(KernelBackend):
    """FlyDSL MXFP4 grouped GEMM backend (fwd / dgrad, gfx950).

    MX_BLOCKWISE, NT (trans_b=True). Reuses the dense mxfp4 whole-loop compute
    (mfma_scale_f32_16x16x128_f8f6f4) with the fp8-grouped addressing; the host
    wrapper zero-pads K scales and the packed N-scale workspace to 256 while
    retaining the real FP4 operand strides.
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
        supported &= is_gfx950()
        supported &= a.dim() == 2 and b.dim() == 3
        supported &= a.dtype == float4_e2m1fn_x2 and b.dtype == float4_e2m1fn_x2
        supported &= out_dtype in (torch.float16, torch.bfloat16)
        supported &= granularity in GroupedGEMMFP4FlyDSLBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a

        supported &= trans_b
        supported &= b.shape[-2] % 64 == 0
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
        from primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp4_kernel import (
            grouped_gemm_mxfp4_flydsl_kernel,
        )

        N = b.shape[-2]
        K = b.shape[-1] * 2
        group_offs_out = kwargs.get("group_offs_out", None)
        return grouped_gemm_mxfp4_flydsl_kernel(
            a,
            a_scales,
            b,
            b_scales,
            group_offs,
            N,
            K,
            group_offs_out=group_offs_out,
            out_dtype=out_dtype,
            num_cu=num_cu if num_cu is not None else -1,
        )


class GroupedGEMMFP4KernelDispatcher(BaseGroupedGEMMKernelDispatcher):
    _backends = {
        BackendType.FLYDSL: BackendEntry(GroupedGEMMFP4FlyDSLBackend),
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
        inplace_add_to_out: bool = False,
        **kwargs,
    ) -> bool:
        supported = True
        # TODO: this backend has no beta=1 accumulate epilogue yet.
        supported &= not inplace_add_to_out
        supported &= not is_gfx942()
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


class GroupedGEMMFP4VariableKFlyDSLBackend(KernelBackend):
    """FlyDSL MXFP4 variable-K grouped GEMM backend (wgrad, gfx950).

    MX_BLOCKWISE, non-transposed FP4-packed 2D operands ``(OUT_*, M_total/2)``,
    contraction over the (128-padded) per-group M. Reuses the dense whole-loop
    with a runtime nval; the host repacks the contraction to 512-aligned per group.
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
        inplace_add_to_out: bool = False,
        out: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        supported = True
        if inplace_add_to_out:
            supported &= out is not None and out.dtype == out_dtype
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= granularity in GroupedGEMMFP4VariableKFlyDSLBackend.SUPPORTED_GRANULARITIES
        supported &= a.dtype == float4_e2m1fn_x2 and b.dtype == float4_e2m1fn_x2
        # bf16 and fp16: this kernel consumes the FlyDSL dual-quant's 512-aligned colwise
        # layout, which the grouped quant now produces for both dtypes.
        supported &= out_dtype in (torch.bfloat16, torch.float16)
        supported &= not trans_a and not trans_b
        # OUT_M/OUT_N (=a.shape[0]/b.shape[0], swapped by trans_c) must be 64-multiples for
        # the packed-scale preshuffle; non-64 (tiny test shapes) falls back to Triton.
        supported &= a.shape[0] % 64 == 0 and b.shape[0] % 64 == 0
        supported &= is_gfx950()
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
        from primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp4_kernel import (
            grouped_gemm_mxfp4_variable_k_flydsl_kernel,
        )

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
        return grouped_gemm_mxfp4_variable_k_flydsl_kernel(
            lhs,
            lhs_scales,
            rhs,
            rhs_scales,
            group_offs,
            OUT_M,
            OUT_N,
            G,
            out_dtype=out_dtype,
            num_cu=num_cu if num_cu is not None else -1,
            beta=1.0 if inplace_add_to_out else 0.0,
            out=out if inplace_add_to_out else None,
        )


class GroupedGEMMFP4VariableKKernelDispatcher(BaseGroupedGEMMVariableKKernelDispatcher):
    _backends = {
        BackendType.FLYDSL: BackendEntry(GroupedGEMMFP4VariableKFlyDSLBackend),
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
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP4)
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

    out = GroupedGEMMFP4KernelDispatcher.dispatch(default_backend_choice, user_backend_choice, **kwargs)
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
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP4)
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

    return GroupedGEMMFP4VariableKKernelDispatcher.dispatch(
        default_backend_choice, user_backend_choice, **kwargs
    )


@_torch_custom_op_wrapper(
    "primus_turbo::grouped_gemm_fp4_variable_k_accum_impl", mutates_args={"out"}, device_types="cuda"
)
def grouped_gemm_fp4_variable_k_accum_impl(
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
) -> None:
    """Variable-K grouped MXFP4 GEMM that accumulates into ``out`` instead of returning.

    Computes ``out += lhs[:,g] @ rhs[:,g]^T`` per group, folding the accumulation into
    the GEMM epilogue (beta=1)
    """
    default_backend_choice = BackendChoice(backend=BackendType(default_backend))
    user_backend_choice = GlobalBackendManager.get_grouped_gemm_backend(PrecisionType.FP4)
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
    )

    # The tuner benchmarks a backend by launching it repeatedly, so letting it tune on
    # the caller's buffer would accumulate the wgrad once per warmup and timing
    # iteration.
    if (
        GlobalBackendManager.auto_tune_enabled()
        and not GroupedGEMMFP4VariableKKernelDispatcher._is_graph_capturing()
    ):
        GroupedGEMMFP4VariableKKernelDispatcher.tune(**{**kwargs, "out": torch.zeros_like(out)})

    GroupedGEMMFP4VariableKKernelDispatcher.dispatch(default_backend_choice, user_backend_choice, **kwargs)


@grouped_gemm_fp4_variable_k_accum_impl.register_fake
def grouped_gemm_fp4_variable_k_accum_impl_meta(
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
) -> None:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 2, f"b must be 2D, got {b.shape}"
    assert out.dim() == 3, f"out must be 3D, got {out.shape}"
    return None


def _check_glu_dispatch(config: Float4QuantConfig, trans_a: bool, trans_b: bool) -> None:
    """The fused GLU epilogue rides the FlyDSL NT kernel, so it has one layout.

    The whole config is held against ``mxfp4_scaling``, not just its granularity: the
    epilogue emits E8M0 block scales, so anything else would be a different kernel even
    at the same granularity.
    """
    assert config.mxfp4_scaling(), f"Fused GLU grouped MXFP4 GEMM is MX_BLOCKWISE-only, got {config}"
    assert is_gfx950(), "Fused GLU grouped MXFP4 GEMM is only supported on gfx950"
    assert not trans_a and trans_b, "Fused GLU grouped MXFP4 GEMM is NT only"


def _check_out_recipes(row: ScalingRecipe, col: ScalingRecipe) -> tuple[bool, bool]:
    """Hold the caller's recipes against the geometry the fused epilogue is built on.

    The epilogue is not a general quantiser. Which axis gets the RHT is baked into the
    kernel -- the col-wise operand, which is what a wgrad contracts over -- and it has
    no 2-D block scaling and no shuffled layouts, so those three are contract checks
    rather than knobs. Shapes wanting them belong on the standalone quantiser.

    Returns the per-operand stochastic-rounding flags, which the epilogue does honour
    independently.
    """
    assert not row.use_rht, "the fused epilogue puts the RHT on the col-wise operand only"
    assert col.use_rht, "the fused epilogue always RHTs the col-wise operand"
    for name, recipe in (("row", row), ("col", col)):
        assert not recipe.use_2d_block, f"the fused epilogue has no 2-D block scaling, got {name}={recipe}"
        assert not recipe.shuffle_scale and not recipe.shuffle_out, (
            f"the fused epilogue writes unshuffled operands, got {name}={recipe}"
        )
    return row.use_sr, col.use_sr


_COL_ALIGN = 256  # the col-wise operand's per-group M alignment, as the quantizer pads it
_ROW_ALIGN = 128  # and the row-wise operand's N alignment


def _alloc_act_buffers(M: int, I: int, G: int, device):
    """The four MXFP4 operands ``grouped_quantize_fp4_with_trans`` would have made.

    Byte-identical geometry to the standalone quantizer, because the wgrad and fc2
    consume them under the same group tables it builds: the row-wise pair is tight-M
    and padded to :data:`_ROW_ALIGN` columns, the col-wise pair is feature-major over
    an M rounded up per group to :data:`_COL_ALIGN`.

    Only the row-wise pad columns are zeroed. The epilogue writes every real column
    and every padded col-wise row (a group's ragged tail included, as zeros), but the
    columns past ``I`` belong to no tile and fc2 contracts over them.
    """
    I_pad = -(-I // _ROW_ALIGN) * _ROW_ALIGN
    M_pad_col = -(-(M + G * _COL_ALIGN) // _COL_ALIGN) * _COL_ALIGN
    row_out = torch.empty((M, I_pad // 2), dtype=torch.uint8, device=device)
    row_sc = torch.empty((M, I_pad // MXFP4_BLOCK_SIZE), dtype=torch.uint8, device=device)
    if I_pad != I:
        row_out[:, I // 2 :].zero_()
        row_sc[:, I // MXFP4_BLOCK_SIZE :].zero_()
    col_out = torch.empty((I, M_pad_col // 2), dtype=torch.uint8, device=device)
    col_sc = torch.empty((I, M_pad_col // MXFP4_BLOCK_SIZE), dtype=torch.uint8, device=device)
    e8 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    return (
        row_out.view(float4_e2m1fn_x2),
        row_sc.view(e8),
        col_out.view(float4_e2m1fn_x2),
        col_sc.view(e8),
    )


@_torch_custom_op_wrapper("primus_turbo::grouped_gemm_fp4_glu_impl", mutates_args=(), device_types="cuda")
def grouped_gemm_fp4_glu_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,  # not used
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    num_cu: int | None,
    probs: torch.Tensor,
    config: Float4QuantConfig,
    out_row_scaling_recipe: ScalingRecipe,
    out_col_scaling_recipe: ScalingRecipe,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fc1 grouped MXFP4 GEMM with the GLU activation and its quantisation fused in.

    The activation is quantised from the accumulators instead of being written and read
    back: what comes out is the row-wise pair fc2 contracts against and the col-wise
    (RHT) pair the wgrad does, byte-for-byte what ``grouped_quantize_fp4_with_trans``
    would have produced from the bf16 activation under the same two recipes.

    Returns ``(intermediate, act_row, act_row_scale, act_col, act_col_scale)``, where
    ``intermediate`` is the [M, N] pre-activation backward needs, still in
    ``out_dtype``. The row-wise pair keeps the tight-M layout, so its group table is
    ``group_offs``; the col-wise pair's is the per-group 256-aligned one the wgrad
    already builds from the same ``group_lens``.

    ``config`` and the recipes ride through as opaque value arguments, so torch.compile
    bakes them into the graph and guards on their equality rather than tracing into
    them. That is also why they carry no defaults: the schema admits an opaque type
    only as a required parameter.
    """
    _check_glu_dispatch(config, trans_a, trans_b)
    row_sr, col_sr = _check_out_recipes(out_row_scaling_recipe, out_col_scaling_recipe)

    from primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp4_glu_kernel import (
        glu_epi_quant_supported,
        grouped_gemm_mxfp4_epi_glu_quant_flydsl_kernel,
    )

    M, N, K = a.shape[0], b.shape[1], b.shape[2] * 2
    assert glu_epi_quant_supported(K, N // 2, out_dtype), (
        f"the fused GLU quant epilogue does not cover K={K} I={N // 2} out_dtype={out_dtype}"
    )
    intermediate = torch.empty((M, N), device=a.device, dtype=out_dtype)
    row_out, row_sc, col_out, col_sc = _alloc_act_buffers(M, N // 2, group_lens.shape[0], a.device)
    grouped_gemm_mxfp4_epi_glu_quant_flydsl_kernel(
        a,
        a_scales,
        b,
        b_scales,
        probs,
        group_offs,
        intermediate,
        row_out,
        row_sc,
        col_out,
        col_sc,
        N,
        K,
        activation=activation,
        row_use_sr=row_sr,
        col_use_sr=col_sr,
        out_dtype=out_dtype,
    )
    return intermediate, row_out, row_sc, col_out, col_sc


@_torch_custom_op_wrapper("primus_turbo::grouped_gemm_fp4_dglu_impl", mutates_args=(), device_types="cuda")
def grouped_gemm_fp4_dglu_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    num_cu: int | None,
    probs: torch.Tensor,
    intermediate: torch.Tensor,
    config: Float4QuantConfig,
    out_row_scaling_recipe: ScalingRecipe,
    out_col_scaling_recipe: ScalingRecipe,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fc2 dgrad with the GLU activation gradient and its quantisation fused in.

    ``a`` is the fc2 output gradient and ``b`` its weight quantised for the dgrad's
    contraction, so ``a @ b^T`` is the gradient wrt the activation; the epilogue
    consumes that against ``intermediate`` in registers, so it never reaches HBM.
    ``grad_l1`` is quantized from the epilogue's values rather than written and read
    back -- it is [M, 2I], the largest tensor the MLP touches, and backward reads it
    exactly once, to quantize it, so that round trip is pure overhead.

    Returns ``(grad_probs, row, row_scale, col, col_scale)``: [M] fp32 wrt the routing
    probabilities, folded here from the per-tile partials the kernel leaves, then the
    four operands, byte-for-byte what ``grouped_quantize_fp4_with_trans`` would have
    produced from the bf16 ``grad_l1`` under the same two recipes. The row-wise pair
    keeps the tight-M layout, so its group table is ``group_offs``; the col-wise pair's
    is the per-group 256-aligned one the wgrad builds from the same ``group_lens``.

    ``config`` and the recipes are opaque value arguments; see the forward's note on
    why they carry no defaults.
    """
    _check_glu_dispatch(config, trans_a, trans_b)
    row_sr, col_sr = _check_out_recipes(out_row_scaling_recipe, out_col_scaling_recipe)

    from primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp4_glu_kernel import (
        dglu_epi_quant_supported,
        grouped_gemm_mxfp4_dglu_grad_probs_partial_spec,
        grouped_gemm_mxfp4_epi_dglu_quant_flydsl_kernel,
    )

    M, I, K = a.shape[0], b.shape[1], b.shape[2] * 2
    assert dglu_epi_quant_supported(K, I, out_dtype), (
        f"the fused dglu quant epilogue does not cover I={I} out_dtype={out_dtype}"
    )
    spec = grouped_gemm_mxfp4_dglu_grad_probs_partial_spec(a, b)
    fill = torch.zeros if spec.needs_zero else torch.empty
    grad_probs_partial = fill(spec.shape, device=a.device, dtype=torch.float32)
    row_out, row_sc, col_out, col_sc = _alloc_act_buffers(M, 2 * I, group_lens.shape[0], a.device)
    grouped_gemm_mxfp4_epi_dglu_quant_flydsl_kernel(
        a,
        a_scales,
        b,
        b_scales,
        intermediate,
        group_offs,
        probs,
        grad_probs_partial,
        row_out,
        row_sc,
        col_out,
        col_sc,
        I,
        K,
        activation=activation,
        row_use_sr=row_sr,
        col_use_sr=col_sr,
        out_dtype=out_dtype,
    )
    return torch.sum(grad_probs_partial, dim=0), row_out, row_sc, col_out, col_sc


@grouped_gemm_fp4_dglu_impl.register_fake
def grouped_gemm_fp4_dglu_impl_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    num_cu: int | None,
    probs: torch.Tensor,
    intermediate: torch.Tensor,
    config: Float4QuantConfig,
    out_row_scaling_recipe: ScalingRecipe,
    out_col_scaling_recipe: ScalingRecipe,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert a.dim() == 2 and b.dim() == 3, f"a must be 2D and b 3D, got {a.shape} {b.shape}"
    assert a.dtype == float4_e2m1fn_x2 and b.dtype == float4_e2m1fn_x2, "operands must be fp4"
    assert out_dtype in (torch.float16, torch.bfloat16), f"bad out_dtype {out_dtype}"
    m, i = a.shape[0], b.shape[1]
    return (
        torch.empty((m,), device=a.device, dtype=torch.float32),
        *_alloc_act_buffers(m, 2 * i, group_lens.shape[0], a.device),
    )


@grouped_gemm_fp4_glu_impl.register_fake
def grouped_gemm_fp4_glu_impl_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
    out_dtype: torch.dtype,
    num_cu: int | None,
    probs: torch.Tensor,
    config: Float4QuantConfig,
    out_row_scaling_recipe: ScalingRecipe,
    out_col_scaling_recipe: ScalingRecipe,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert a.dim() == 2 and b.dim() == 3, f"a must be 2D and b 3D, got {a.shape} {b.shape}"
    assert a.dtype == float4_e2m1fn_x2 and b.dtype == float4_e2m1fn_x2, "operands must be fp4"
    assert out_dtype in (torch.float16, torch.bfloat16), f"bad out_dtype {out_dtype}"
    # b holds gate||up, so its N is twice the activation's width.
    m, n = a.shape[0], b.shape[1]
    return (
        torch.empty((m, n), device=a.device, dtype=out_dtype),
        *_alloc_act_buffers(m, n // 2, group_lens.shape[0], a.device),
    )


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
