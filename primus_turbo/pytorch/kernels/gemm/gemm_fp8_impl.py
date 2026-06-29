###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Tuple

import torch

_torch_custom_op_wrapper = torch.library.custom_op

from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel
from primus_turbo.flydsl.gemm.mxfp8_gemm_kernel import gemm_mxfp8_flydsl_kernel
from primus_turbo.flydsl.utils.gemm_helper import preshuffle_ab_flydsl
from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
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
from primus_turbo.triton.gemm.gemm_fp8_kernel import (
    gemm_fp8_blockwise_triton_kernel,
    gemm_fp8_rowwise_triton_kernel,
    gemm_fp8_tensorwise_triton_kernel,
)


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


class GEMMFP8HipBLASLtBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.MX_BLOCKWISE,
    }

    # (a_dtype, b_dtype, c_dtype)
    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)

    # (trans_a, trans_b, trans_c)
    SUPPORTED_LAYOUTS = (
        (False, False, False),
        (False, True, False),
        (True, False, False),
    )

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
        # check ScalingGranularity
        supported &= granularity in GEMMFP8HipBLASLtBackend.SUPPORTED_GRANULARITIES
        # check dtype
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8HipBLASLtBackend.SUPPORTED_DTYPES

        # TODO:
        # check layout
        # supported &= (trans_a, trans_b, trans_c) in GEMMFP8HipBLASLtBackend.SUPPORTED_LAYOUTS
        # TODO:
        # check shape

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
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm_fp8(
            a, a_scale_inv, b, b_scale_inv, out_dtype, trans_a, trans_b, trans_c, granularity.name
        )


class GEMMFP8CKBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
    }

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
        supported = True
        # check ScalingGranularity
        supported &= granularity in GEMMFP8CKBackend.SUPPORTED_GRANULARITIES
        # check dtype
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8CKBackend.SUPPORTED_DTYPES

        # TODO: check layout
        # supported &= (trans_a, trans_b, trans_c) in GEMMFP8CKBackend.SUPPORTED_LAYOUTS

        # TODO: check shape

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
        if trans_c:
            lhs, rhs = b, a
            lhs_scale_inv, rhs_scale_inv = b_scale_inv, a_scale_inv
            trans_lhs = not trans_b
            trans_rhs = not trans_a
        else:
            lhs, rhs = a, b
            lhs_scale_inv, rhs_scale_inv = a_scale_inv, b_scale_inv
            trans_lhs = trans_a
            trans_rhs = trans_b

        return torch.ops.primus_turbo_cpp_extension.ck_gemm_fp8(
            lhs, rhs, lhs_scale_inv, rhs_scale_inv, trans_lhs, trans_rhs, out_dtype, granularity.name
        )


class GEMMFP8TritonBackend(KernelBackend):
    """Triton persistent-kernel backend for FP8 GEMM.

    Supports:
      - TENSORWISE: per-tensor scaling (all layouts), including HYBRID format
      - ROWWISE: per-row/per-col vector scaling (all layouts)
      - BLOCKWISE: block-wise scaling with three layouts:
          NT/RCR (forward), NN/RRR (grad_X), TN/CRR (grad_W)
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        ScalingGranularity.BLOCKWISE,
    }

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
        supported = True
        supported &= granularity in GEMMFP8TritonBackend.SUPPORTED_GRANULARITIES
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8TritonBackend.SUPPORTED_DTYPES
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
        if granularity == ScalingGranularity.TENSORWISE:
            return gemm_fp8_tensorwise_triton_kernel(
                a,
                a_scale_inv,
                b,
                b_scale_inv,
                trans_a=trans_a,
                trans_b=trans_b,
                out_dtype=out_dtype,
                trans_c=trans_c,
            )
        elif granularity == ScalingGranularity.ROWWISE:
            return gemm_fp8_rowwise_triton_kernel(
                a,
                a_scale_inv,
                b,
                b_scale_inv,
                trans_a=trans_a,
                trans_b=trans_b,
                out_dtype=out_dtype,
                trans_c=trans_c,
            )
        elif granularity == ScalingGranularity.BLOCKWISE:
            return gemm_fp8_blockwise_triton_kernel(
                a,
                a_scale_inv,
                b,
                b_scale_inv,
                trans_a=trans_a,
                trans_b=trans_b,
                out_dtype=out_dtype,
                trans_c=trans_c,
            )
        else:
            raise ValueError(f"Unsupported granularity for FP8 Triton: {granularity}")


class GEMMFP8TurboBackend(KernelBackend):
    """Hand-tuned MXFP8 GEMM kernel for GFX950 (MI350/MI355).

    Supports MX_BLOCKWISE only. NT layout. Tile 256x256x128.
    Shape constraints: m,n % 16 == 0, k % 128 == 0, k >= 384.
    """

    SUPPORTED_GRANULARITIES = {
        ScalingGranularity.MX_BLOCKWISE,
    }

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
        supported = True
        supported &= granularity in GEMMFP8TurboBackend.SUPPORTED_GRANULARITIES
        supported &= (a.dtype, b.dtype, out_dtype) in GEMMFP8TurboBackend.SUPPORTED_DTYPES
        supported &= not trans_a and trans_b and not trans_c
        m, n, k = get_gemm_logical_shape(a, b, trans_a, trans_b)
        supported &= m % 16 == 0 and n % 16 == 0 and k % 128 == 0 and k >= 384
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
        return torch.ops.primus_turbo_cpp_extension.turbo_gemm_fp8(
            a, a_scale_inv, b, b_scale_inv, out_dtype, trans_a, trans_b, trans_c, granularity.name
        )


class GEMMFP8FlyDSLBackend(KernelBackend):
    """FlyDSL 8-wave fp8 dense GEMM backend.

    The underlying kernel wrapper lives in
    ``primus_turbo.flydsl.gemm.gemm_fp8_kernel``.

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

    MX_BLOCKWISE (compute-only): NT only, per-operand E4M3/E5M2 (incl. hybrid),
    bf16/fp16 out. Per-1x32 raw E8M0 2D block scales [M,K//32]/[N,K//32] (M%16,
    K%32); execute() zero-pads (M->64, K->kernel tile) and LDS-repacks them to the
    preshuffled int32 layout the kernel consumes (no host-side preshuffle).
    Routes to gemm_mxfp8_flydsl_kernel.
    """

    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE, ScalingGranularity.MX_BLOCKWISE}
    # E4M3 / E5M2 / hybrid, bf16 or fp16 output. Per-operand fp8 format is threaded
    # into the MFMA via cbsz(srcA)/blgp(srcB) (0=E4M3, 1=E5M2) and the FlyDSL
    # MFMA_Scale atom dtype; fp16 output is produced from the f32 accumulator.
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
        supported = True
        # gfx950 (CDNA4) only: the kernel uses mfma_f32_16x16x128_f8f6f4, absent
        # on gfx942 and below. Gate here so the dispatcher never picks FlyDSL off
        # gfx950 (the backend still imports fine on other archs).
        supported &= get_device_compute_capability() >= (9, 5)
        supported &= granularity in GEMMFP8FlyDSLBackend.SUPPORTED_GRANULARITIES

        if granularity == ScalingGranularity.MX_BLOCKWISE:
            # NT. Per-operand E4M3/E5M2 (cbsz/blgp in the MFMA), bf16/fp16 out. Scales
            # are raw E8M0 2D [M,K//32] / [N,K//32]; execute() zero-pads (M->64,
            # K->kernel tile) and LDS-repacks them to the preshuffled int32 layout the
            # kernel consumes (no host-side preshuffle).
            supported &= a.ndim == 2 and b.ndim == 2
            supported &= (not trans_a) and trans_b
            supported &= a.dtype in (float8_e4m3, float8_e5m2) and b.dtype in (float8_e4m3, float8_e5m2)
            supported &= out_dtype in (torch.bfloat16, torch.float16)
            if not supported:
                return supported
            m, k, n = a.shape[0], a.shape[1], b.shape[0]
            supported &= n % 16 == 0
            # raw E8M0 block scales, 1 byte/elem, [M,K//32] / [N,K//32]. execute()
            # zero-pads M up to 64 and K up to the kernel tile, so only the MX
            # minimums (M % 16, K % 32) are required here.
            supported &= m % 16 == 0
            supported &= k % 32 == 0
            supported &= a_scale_inv.ndim == 2 and a_scale_inv.shape == (m, k // 32)
            supported &= b_scale_inv.ndim == 2 and b_scale_inv.shape == (n, k // 32)
            supported &= a_scale_inv.element_size() == 1 and b_scale_inv.element_size() == 1
            return supported

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
        # No size cap: foldable operands (NT both, NN-A) fold their per-tile base into
        # the i64 SRD; the traversal operands (NN-B k*n, TN k*m & k*n) that would wrap a
        # 32-bit soffset past 2^32 fp8 are re-based per load in i64 by the wrapper (it
        # auto-selects i64 at/above 2^32, keeping the cheaper int32 path below).
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
        if granularity == ScalingGranularity.MX_BLOCKWISE:
            # NT only. Raw E8M0 -> one fused FlyDSL LDS repack to the preshuffled
            # int32 layout the kernel consumes (A layout-1, B-comb layout-3).
            k = a.shape[1]
            # K-tail: the FlyDSL kernel tiles K by 128 and needs K>=256. Zero-pad K
            # to the next valid extent; padded fp8 operands are 0 -> contribute 0
            # (scale padded with 127=2^0, finite). MX requires K % 32 == 0.
            k_pad = max(256, ((k + 127) // 128) * 128)
            if k_pad != k:
                M_, N_ = a.shape[0], b.shape[0]
                ap = torch.zeros(M_, k_pad, dtype=a.dtype, device=a.device)
                ap[:, :k] = a
                a = ap
                bp = torch.zeros(N_, k_pad, dtype=b.dtype, device=b.device)
                bp[:, :k] = b
                b = bp
                asc = a_scale_inv.view(torch.uint8)
                bsc = b_scale_inv.view(torch.uint8)
                asp_ = torch.full((M_, k_pad // 32), 127, dtype=torch.uint8, device=a.device)
                asp_[:, : k // 32] = asc
                bsp_ = torch.full((N_, k_pad // 32), 127, dtype=torch.uint8, device=b.device)
                bsp_[:, : k // 32] = bsc
                a_scale_inv, b_scale_inv, k = asp_, bsp_, k_pad
            # General-M: the A-operand scale preshuffle (layout 1) groups rows by 64,
            # so M must be a 64-multiple. Pad A rows (fp8 0 -> contributes 0, scale
            # 127=2^0) up to the next 64; the extra output rows are sliced off below.
            # (B / general-N is handled by the combined-B preshuffle's cdiv(N,256)*4
            # sizing + source-row mask, so only M needs padding.)
            m_orig = a.shape[0]
            m_pad = ((m_orig + 63) // 64) * 64
            if m_pad != m_orig:
                ap = torch.zeros(m_pad, k, dtype=a.dtype, device=a.device)
                ap[:m_orig] = a
                a = ap
                asc = a_scale_inv.view(torch.uint8)
                asp_ = torch.full((m_pad, k // 32), 127, dtype=torch.uint8, device=a.device)
                asp_[:m_orig] = asc
                a_scale_inv = asp_
            # Fused LDS repack (FlyDSL). A needs M % 64 (padded above); B accepts any
            # N % 16 (general-N handled by the kernel's cdiv(N,256)*4 sizing + row mask).
            # preshuffle_ab_flydsl emits broadcast layout (pack=1).
            a_sp, b_sp = preshuffle_ab_flydsl(
                a_scale_inv.view(torch.uint8), b_scale_inv.view(torch.uint8), k, 4
            )
            out = gemm_mxfp8_flydsl_kernel(a, a_sp, b, b_sp, out_dtype=out_dtype)
            if m_pad != m_orig:
                out = out[:m_orig]
            return out.t().contiguous() if trans_c else out
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


_GEMM_FP8_BACKENDS = {
    BackendType.TURBO: BackendEntry(GEMMFP8TurboBackend),
    BackendType.HIPBLASLT: BackendEntry(GEMMFP8HipBLASLtBackend),
    BackendType.CK: BackendEntry(GEMMFP8CKBackend),
    BackendType.TRITON: BackendEntry(GEMMFP8TritonBackend),
    BackendType.FLYDSL: BackendEntry(GEMMFP8FlyDSLBackend),
}


class GEMMFP8KernelDispatcher(AutoKernelDispatcher):
    _backends = _GEMM_FP8_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, a, b, trans_a, trans_b, trans_c, out_dtype, granularity, **kwargs):
        m, n, k = get_gemm_logical_shape(a, b, trans_a, trans_b)
        return (m, n, k, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c, granularity)


@_torch_custom_op_wrapper("primus_turbo::gemm_fp8_impl", mutates_args=(), device_types="cuda")
def gemm_fp8_impl(
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
) -> torch.Tensor:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_gemm_backend(PrecisionType.FP8)
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
    )

    return GEMMFP8KernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


@gemm_fp8_impl.register_fake
def gemm_fp8_impl_meta(
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
) -> torch.Tensor:
    m, n, _ = get_gemm_logical_shape(a, b, trans_a, trans_b)
    if trans_c:
        m, n = n, m
    return torch.empty(m, n, dtype=out_dtype, device=a.device)
