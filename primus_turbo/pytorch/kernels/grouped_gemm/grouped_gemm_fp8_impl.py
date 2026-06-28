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
from primus_turbo.pytorch.core.utils import get_device_compute_capability
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

# MXFP8 e8m0 block scale constants for the hipBLASLt VEC32_UE8M0 swizzle.
_MX_BLOCK = 32  # MXFP8 1x32 block size (e8m0 scale covers 32 elements along K).
_MX_M_PAD = 128  # hipBLASLt pads the scale/data m-dim to a 128 multiple.
_MX_K_GROUP = 4  # e8m0 scales are swizzled in groups of 4 along K-blocks.


def _swizzle_mxfp8_scale_groups4(scale: torch.Tensor, m_pad: int) -> torch.Tensor:
    """Reorder a row-wise e8m0 scale ``(M, Ks)`` into hipBLASLt's GEMM-swizzled
    (groups-of-4) layout, returned as a ``(m_pad, Ks_pad)`` e8m0 tensor.

    Index map (matches TE's mxfp8_colwise_scale_to_rowwise with output_swizzled):
        out_idx = (k // 4) * (m_pad * 4) + m * 4 + (k % 4)
    Padded entries (m >= M, or k >= Ks) are 127 (e8m0 bias 0 -> native scale 1.0).
    hipBLASLt requires the data matrix m-dim padded to the same ``m_pad``.
    """
    M, Ks = scale.shape
    ks_pad = ((Ks + _MX_K_GROUP - 1) // _MX_K_GROUP) * _MX_K_GROUP
    dev = scale.device
    out = torch.full((m_pad * ks_pad,), 127, dtype=torch.uint8, device=dev)
    m_idx = torch.arange(M, device=dev)
    k_idx = torch.arange(Ks, device=dev)
    mm, kk = torch.meshgrid(m_idx, k_idx, indexing="ij")
    out_idx = (kk // _MX_K_GROUP) * (m_pad * _MX_K_GROUP) + mm * _MX_K_GROUP + (kk % _MX_K_GROUP)
    out[out_idx.reshape(-1)] = scale.reshape(-1).view(torch.uint8)
    return out.view(m_pad, ks_pad).view(scale.dtype)


def _grouped_mxfp8_hipblaslt(
    a: torch.Tensor,  # [M_in (padded), K] fp8
    b: torch.Tensor,  # [E, N, K] fp8 (NT)
    a_scales: torch.Tensor,  # [M_in (padded), K//32] e8m0
    b_scales: torch.Tensor,  # [E, N, K//32] e8m0
    group_lens: torch.Tensor,  # [E] int64
    group_offs: torch.Tensor,  # [E+1] int64 padded read offsets (A / A_scale)
    group_offs_out: torch.Tensor | None,  # [E+1] int64 real write offsets (C)
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Grouped MXFP8 forward (NT) on hipBLASLt: per-expert pad-to-128 + groups-of-4
    scale swizzle + the proven dense ``hipblaslt_gemm_fp8`` MX op.

    Mirrors the Triton MX backend's contract: the output is over-allocated to the
    padded input rows (``a.shape[0]``), each group's real rows are written at the
    ``group_offs_out`` (tight) offsets, and the caller slices ``[:total_m]``.
    """
    if group_offs_out is None:
        group_offs_out = group_offs
    E = int(group_lens.shape[0])
    K = a.shape[1]
    Ks = a_scales.shape[1]
    N = b.shape[1]
    # Over-allocate to the padded input rows (matches Triton + the registered fake).
    out = torch.empty((a.shape[0], N), dtype=out_dtype, device=a.device)

    lens = group_lens.tolist()
    rd = group_offs.tolist()
    oo = group_offs_out.tolist()
    for g in range(E):
        ln = lens[g]
        if ln <= 0:
            continue
        m_pad = ((ln + _MX_M_PAD - 1) // _MX_M_PAD) * _MX_M_PAD
        # Pad A data + scale for this group to m_pad rows (padding rows = 0).
        a_g = a.new_zeros((m_pad, K))
        a_g[:ln] = a[rd[g] : rd[g] + ln]
        as_g = torch.zeros((m_pad, Ks), dtype=torch.uint8, device=a.device).view(a_scales.dtype)
        as_g[:ln] = a_scales[rd[g] : rd[g] + ln]
        sa = _swizzle_mxfp8_scale_groups4(as_g, m_pad)
        # Weight: N rows (a fixed 128-multiple in MoE), swizzled with m_pad = N.
        n_pad = ((N + _MX_M_PAD - 1) // _MX_M_PAD) * _MX_M_PAD
        sb = _swizzle_mxfp8_scale_groups4(b_scales[g].contiguous(), n_pad)
        b_g = b[g].contiguous()
        if n_pad != N:
            b_pad = b.new_zeros((n_pad, K))
            b_pad[:N] = b_g
            b_g = b_pad
        c_g = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp8(
            a_g, sa, b_g, sb, out_dtype, False, True, False, "MX_BLOCKWISE"
        )  # [m_pad, n_pad]
        out[oo[g] : oo[g] + ln] = c_g[:ln, :N]
    return out


def _grouped_mxfp8_variable_k_hipblaslt(
    lhs: torch.Tensor,  # [OUT_M, M_total] fp8 (col-wise grad_out)
    rhs: torch.Tensor,  # [OUT_N, M_total] fp8 (col-wise activation)
    lhs_scales: torch.Tensor,  # [OUT_M, M_total//32] e8m0
    rhs_scales: torch.Tensor,  # [OUT_N, M_total//32] e8m0
    group_lens: torch.Tensor,  # [G] int64 (colwise-padded lens; each M_g % 128 == 0)
    group_offs: torch.Tensor,  # [G+1] int64 colwise-padded offsets along M
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Grouped MXFP8 wgrad (variable-K) on hipBLASLt, fully on the GEMM (no Triton).

    Per group g:  C[g] (OUT_M, OUT_N) = lhs[:, g] (OUT_M, M_g) @ rhs[:, g] (OUT_N, M_g)^T,
    reduction over M_g (the token dim). Per group this is the SAME NT GEMM as the
    forward -- the contraction M_g is the contiguous trailing dim and the e8m0
    scales are blocked along it -- so the forward recipe applies directly: slice
    the M_g columns, pad the OUT_M / OUT_N output rows to a 128-multiple, swizzle
    both scales (groups-of-4) with the matching m_pad, run the dense hipBLASLt MX
    op, and trim the padding. No physical operand transpose is needed because the
    col-wise operands already arrive in (feat, M) layout. M_g arrives 128-padded
    (group_offs are the colwise-padded offsets), so the contraction needs no mask.
    Output is (G, OUT_M, OUT_N), matching the (G, N, K) grad_b the model expects.
    """
    G = int(group_lens.shape[0])
    OUT_M = lhs.shape[0]
    OUT_N = rhs.shape[0]
    out = torch.empty((G, OUT_M, OUT_N), dtype=out_dtype, device=lhs.device)

    m_pad_M = ((OUT_M + _MX_M_PAD - 1) // _MX_M_PAD) * _MX_M_PAD
    m_pad_N = ((OUT_N + _MX_M_PAD - 1) // _MX_M_PAD) * _MX_M_PAD
    gp = group_offs.tolist()
    for g in range(G):
        ms, me = gp[g], gp[g + 1]
        m_g = me - ms  # 128-padded contraction
        if m_g <= 0:
            out[g].zero_()
            continue
        ks_lo, ks_hi = ms // _MX_BLOCK, me // _MX_BLOCK
        a_g = lhs[:, ms:me].contiguous()  # [OUT_M, M_g]
        b_g = rhs[:, ms:me].contiguous()  # [OUT_N, M_g]
        as_g = lhs_scales[:, ks_lo:ks_hi].contiguous()  # [OUT_M, M_g//32]
        bs_g = rhs_scales[:, ks_lo:ks_hi].contiguous()  # [OUT_N, M_g//32]
        if m_pad_M != OUT_M:
            a_p = a_g.new_zeros((m_pad_M, m_g))
            a_p[:OUT_M] = a_g
            a_g = a_p
            asp = torch.zeros((m_pad_M, as_g.shape[1]), dtype=torch.uint8, device=lhs.device).view(
                lhs_scales.dtype
            )
            asp[:OUT_M] = as_g
            as_g = asp
        if m_pad_N != OUT_N:
            b_p = b_g.new_zeros((m_pad_N, m_g))
            b_p[:OUT_N] = b_g
            b_g = b_p
            bsp = torch.zeros((m_pad_N, bs_g.shape[1]), dtype=torch.uint8, device=lhs.device).view(
                rhs_scales.dtype
            )
            bsp[:OUT_N] = bs_g
            bs_g = bsp
        sa = _swizzle_mxfp8_scale_groups4(as_g, m_pad_M)
        sb = _swizzle_mxfp8_scale_groups4(bs_g, m_pad_N)
        c_g = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp8(
            a_g, sa, b_g, sb, out_dtype, False, True, False, "MX_BLOCKWISE"
        )  # [m_pad_M, m_pad_N]
        out[g] = c_g[:OUT_M, :OUT_N]
    return out


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
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
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
        supported &= granularity in GroupedGEMMFP8HipblasltBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
        if granularity == ScalingGranularity.MX_BLOCKWISE:
            # MXFP8 (e8m0 1x32 block scales): both operands must be fp8, NT only
            # (trans_b=True), matching the hipBLASLt VEC32_UE8M0 dense path.
            supported &= a.dtype in (float8_e4m3, float8_e5m2)
            supported &= b.dtype in (float8_e4m3, float8_e5m2)
            supported &= out_dtype in (torch.float16, torch.bfloat16)
            supported &= trans_b
        else:
            supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8HipblasltBackend.SUPPORTED_DTYPES
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
        if granularity == ScalingGranularity.MX_BLOCKWISE:
            # hipBLASLt VEC32_UE8M0 on gfx1250 is TN-only AND requires the e8m0
            # block scales in the GEMM-swizzled (groups-of-4) layout with the data
            # m-dimension padded to a 128-multiple (see _grouped_mxfp8_hipblaslt
            # for the empirical derivation). The grouped forward (NT, b=[E,N,K]) is
            # a set of independent per-expert GEMMs, so we pad+swizzle each group
            # and run the proven dense hipBLASLt MX op. group_offs are the padded
            # read offsets (32-aligned per group), group_offs_out the real write
            # offsets (defaults to group_offs).
            group_offs_out = kwargs.get("group_offs_out", None)
            if group_offs_out is None:
                group_offs_out = group_offs
            N = b.shape[1]
            # Fast path: the C++ multi-stream kernel (batched pad+swizzle HIP kernels
            # + per-group hipblaslt_gemm_impl on the stream pool) requires N % 128 == 0
            # (true for MoE). Falls back to the Python reference otherwise.
            if N % 128 == 0:
                return torch.ops.primus_turbo_cpp_extension.hipblaslt_grouped_gemm_mxfp8(
                    a, b, a_scales, b_scales, group_lens, group_offs, group_offs_out, out_dtype
                )
            return _grouped_mxfp8_hipblaslt(
                a, b, a_scales, b_scales, group_lens, group_offs, group_offs_out, out_dtype
            )
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
    # TENSORWISE wgrad runs on the C++ hipBLASLt grouped kernel. MX_BLOCKWISE wgrad
    # also runs fully on hipBLASLt: per group it is the SAME NT GEMM as the forward
    # (C[g] = grad_out_col[:,g] @ a_col[:,g]^T, contraction = M_g over the token dim,
    # which is the contiguous trailing dim), so the forward pad-128 + groups-of-4
    # scale swizzle applies directly to the per-group column slices -- no physical
    # transpose, no Triton (see _grouped_mxfp8_variable_k_hipblaslt).
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
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
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= granularity in GroupedGEMMFP8VariableKHipblasltBackend.SUPPORTED_GRANULARITIES
        if granularity == ScalingGranularity.MX_BLOCKWISE:
            # MoE wgrad layout: both operands fp8 (e4m3/e5m2), non-transposed
            # (OUT_M, M_total) / (OUT_N, M_total). Delegated to Triton in execute().
            supported &= a.dtype in (float8_e4m3, float8_e5m2)
            supported &= b.dtype in (float8_e4m3, float8_e5m2)
            supported &= out_dtype in (torch.float16, torch.bfloat16)
            supported &= not trans_a and not trans_b
        else:
            supported &= (
                a.dtype,
                b.dtype,
                out_dtype,
            ) in GroupedGEMMFP8VariableKHipblasltBackend.SUPPORTED_DTYPES
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
        **kwargs,
    ):
        if granularity == ScalingGranularity.MX_BLOCKWISE:
            # wgrad: C[g] = lhs[:,g] @ rhs[:,g]^T over the token dim M_g. trans_c
            # swaps which operand is lhs (mirrors the eager/Triton path). Runs the
            # per-group NT pad-128 + groups-of-4 swizzle on hipBLASLt.
            if trans_c:
                lhs, rhs = b, a
                lhs_scales, rhs_scales = b_scales, a_scales
            else:
                lhs, rhs = a, b
                lhs_scales, rhs_scales = a_scales, b_scales
            return _grouped_mxfp8_variable_k_hipblaslt(
                lhs, rhs, lhs_scales, rhs_scales, group_lens, group_offs, out_dtype
            )
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
        )


class GroupedGEMMFP8FlyDSLBackend(KernelBackend):
    """FlyDSL fp8 grouped GEMM backend (gfx950, per-tensor / TENSORWISE only).

    M-grouped operator: forward (trans_b=True, NT) + dgrad (trans_b=False, NN).
    Uses the FlyDSL mfma_f32_16x16x128_f8f6f4 kernel (gfx950-only).
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
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8FlyDSLBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8FlyDSLBackend.SUPPORTED_GRANULARITIES
        supported &= not trans_a
        # per-tensor scaling = single scalar each
        supported &= a_scales.numel() == 1 and b_scales.numel() == 1
        # gfx950 (CDNA4) only: kernel uses mfma_f32_16x16x128_f8f6f4.
        supported &= get_device_compute_capability() >= (9, 5)
        # K-loop needs ceil(K/128) >= 2, i.e. contraction K >= 129.
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
        from primus_turbo.flydsl.grouped_gemm.gemm_fp8_grouped_kernel import (
            grouped_gemm_fp8_tensorwise_flydsl_kernel,
        )

        return grouped_gemm_fp8_tensorwise_flydsl_kernel(
            a, b, a_scales, b_scales, group_offs, trans_b=trans_b, out_dtype=out_dtype, num_cu=num_cu
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
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.dim() == 2 and b.dim() == 2
        supported &= granularity in GroupedGEMMFP8VariableKTritonBackend.SUPPORTED_GRANULARITIES
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
        **kwargs,
    ):
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales

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
            )
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


class GroupedGEMMFP8VariableKFlyDSLBackend(KernelBackend):
    """FlyDSL fp8 variable-K grouped GEMM backend (gfx950, per-tensor only).

    wgrad: C[g] = lhs[offs[g]:offs[g+1]]^T @ rhs[offs[g]:offs[g+1]], contraction
    = m_g (variable per group) via a runtime scf.for K-loop. Uses the FlyDSL
    mfma_f32_16x16x128_f8f6f4 TN kernel (gfx950-only).
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
        supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8VariableKFlyDSLBackend.SUPPORTED_DTYPES
        supported &= granularity in GroupedGEMMFP8VariableKFlyDSLBackend.SUPPORTED_GRANULARITIES
        # variable-K contract: contraction along the shared (rows) dim.
        supported &= trans_a and not trans_b
        # per-tensor scaling = single scalar each
        supported &= a_scales.numel() == 1 and b_scales.numel() == 1
        # gfx950 (CDNA4) only: kernel uses mfma_f32_16x16x128_f8f6f4.
        supported &= get_device_compute_capability() >= (9, 5)
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
        from primus_turbo.flydsl.grouped_gemm.gemm_fp8_grouped_kernel import (
            grouped_gemm_fp8_variable_k_tensorwise_flydsl_kernel,
        )

        # trans_c swaps which operand is lhs (output transpose), mirroring the
        # Triton variable-K backend: out[g] = lhs[g]^T @ rhs[g].
        if trans_c:
            lhs, rhs = b, a
            lhs_scales, rhs_scales = b_scales, a_scales
        else:
            lhs, rhs = a, b
            lhs_scales, rhs_scales = a_scales, b_scales

        return grouped_gemm_fp8_variable_k_tensorwise_flydsl_kernel(
            lhs, rhs, lhs_scales, rhs_scales, group_offs, out_dtype=out_dtype, num_cu=num_cu
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
        group_offs_out=group_offs_out,
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
    # into the tight layout and the caller slices [:total_m].
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
    return torch.empty((bs, m, n), device=a.device, dtype=out_dtype)
