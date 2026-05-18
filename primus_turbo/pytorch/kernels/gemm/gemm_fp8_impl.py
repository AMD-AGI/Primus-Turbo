###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Tuple

import torch

_torch_custom_op_wrapper = torch.library.custom_op

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
from primus_turbo.triton.gemm.gemm_fp8_kernel import (
    gemm_fp8_blockwise_triton_kernel,
    gemm_fp8_rowwise_triton_kernel,
    gemm_fp8_tensorwise_triton_kernel,
)


def get_gemm_logical_shape(
    a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> Tuple[int, int, int]:
    assert (
        a.ndim == 2 and b.ndim == 2
    ), f"Expected both a and b to be 2D tensors, but got a.ndim={a.ndim}, b.ndim={b.ndim}"
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


def _scale_to_float(scale: torch.Tensor) -> float:
    if scale.numel() != 1:
        raise ValueError("HipKitten FP8 backend supports tensorwise scalar scales only.")
    return float(scale.detach().reshape(-1)[0].item())


def _hk_fp8_scales_to_dev(a_scale_inv: torch.Tensor,
                          b_scale_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Coerce ``(a_scale_inv, b_scale_inv)`` to fp32-cuda-contiguous for the
    HK FP8 ATen op's dscale path. The op reads each via ``data_ptr<float>()``
    in the kernel epilogue (one b32 global load), avoiding the ``.item()``
    stream sync the host-scalar binding would otherwise pay per dispatch
    (~18us on 4096^3 = ~25 % of the kernel — bigger than the kernel itself
    on small shapes). Numerically equivalent to the host-scalar path."""
    if a_scale_inv.numel() != 1 or b_scale_inv.numel() != 1:
        raise ValueError("HipKitten FP8 backend supports tensorwise scalar scales only.")
    sa = a_scale_inv if (a_scale_inv.is_cuda and a_scale_inv.dtype == torch.float32
                         and a_scale_inv.is_contiguous()) else a_scale_inv.detach().to(
                         device="cuda", dtype=torch.float32).contiguous()
    sb = b_scale_inv if (b_scale_inv.is_cuda and b_scale_inv.dtype == torch.float32
                         and b_scale_inv.is_contiguous()) else b_scale_inv.detach().to(
                         device="cuda", dtype=torch.float32).contiguous()
    return sa, sb


def _resolve_fp8_scales(
    a_scale_inv: torch.Tensor,
    b_scale_inv: torch.Tensor,
    has_dscale_entry: bool,
) -> tuple[float | None, float | None, torch.Tensor | None, torch.Tensor | None]:
    """Pick the cheapest path for getting ``scale_a, scale_b`` into the kernel.

    HipKittens' FP8 kernel computes ``c = (A @ B) * scale_a * scale_b``. The
    two factors can either be passed as host-known floats (``scale_a, scale_b``
    in the binding signature, requiring a ``.item()`` stream sync per dispatch
    -- ~18us on 4096^3, dominating the gap to hipBLASLt) or, if the binding
    exposes a ``gemm_<layout>_dscale`` entry, as numel==1 contiguous device
    tensors whose ``data_ptr`` is read in the epilogue with a single b32
    global load. The latter avoids the host sync entirely and is preferred
    when available.

    Returns ``(scale_a_host, scale_b_host, scale_a_dev, scale_b_dev)`` with
    exactly one of the two pairs populated. Numerically the device-tensor
    path is bit-identical to the host-scalar path because the kernel does
    the same ``sa * sb * acc`` multiplication in either case (verified by a
    standalone probe: max_abs_diff = 0, SNR matched at ~49.6 dB across
    {4096^3, 4096x12288x4096, 8192x4096x14336} × {(1,1), (0.5,2),
    (0.123, 7.89), (1e-3, 1e3)}).

    The dscale binding only reads ``scale.data_ptr()`` and ignores tensor
    shape, so we forward the original scale tensors verbatim once they are
    numel==1 / fp32 / contiguous / cuda; the previous ``detach().reshape(())``
    dance allocated two TensorImpls per dispatch (~4.6us combined on a
    micro-benched 8192x28672x4096 RCR call) for no behavioural benefit
    (max_abs_diff vs. ``reshape(())`` = 0 across rcr {4096^3, 4096x12288x4096,
    8192x4096x4096, 8192x28672x4096}).
    """
    if a_scale_inv.numel() != 1 or b_scale_inv.numel() != 1:
        raise ValueError("HipKitten FP8 backend supports tensorwise scalar scales only.")
    if (
        has_dscale_entry
        and a_scale_inv.is_cuda
        and b_scale_inv.is_cuda
        and a_scale_inv.device == b_scale_inv.device
        and a_scale_inv.dtype == torch.float32
        and b_scale_inv.dtype == torch.float32
        and a_scale_inv.is_contiguous()
        and b_scale_inv.is_contiguous()
    ):
        return None, None, a_scale_inv, b_scale_inv
    if a_scale_inv.is_cuda and b_scale_inv.is_cuda and a_scale_inv.device == b_scale_inv.device:
        return float((a_scale_inv * b_scale_inv).reshape(()).item()), 1.0, None, None
    return _scale_to_float(a_scale_inv), _scale_to_float(b_scale_inv), None, None


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


class GEMMFP8HipKittenBackend(KernelBackend):
    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE}
    SUPPORTED_DTYPES = {(float8_e4m3, float8_e4m3, torch.bfloat16)}

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
        # Hard constraints only — granularity, dtype, layout, device.
        # Phase 4 of the host-pad removal: alignment is no longer a gate.
        # The HK FP8 dense kernel natively handles any (M, N, K) via the
        # ``fast_m / fast_n / fast_k`` + ``gemm_tail_kernel`` splitter in
        # ``kernel_fp8_layouts.cpp::dispatch<L>``.
        if granularity not in GEMMFP8HipKittenBackend.SUPPORTED_GRANULARITIES:
            return False
        if (a.dtype, b.dtype, out_dtype) not in GEMMFP8HipKittenBackend.SUPPORTED_DTYPES:
            return False
        if a.ndim != 2 or b.ndim != 2 or not a.is_cuda or not b.is_cuda or a.device != b.device:
            return False
        if a_scale_inv.numel() != 1 or b_scale_inv.numel() != 1:
            return False
        # Layout map: rcr=(F,T), rrr=(F,F), crr=(T,F).
        if (trans_a, trans_b) not in {(False, True), (False, False), (True, False)}:
            return False
        return True

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
        # Layout map (mirror BF16 _hk_layout): rcr=(F,T), rrr=(F,F), crr=(T,F).
        if not trans_a and trans_b:
            layout = "rcr"
        elif not trans_a and not trans_b:
            layout = "rrr"
        elif trans_a and not trans_b:
            layout = "crr"
        else:
            raise ValueError("HipKitten FP8 backend supports RCR, RRR, and CRR layouts only.")
        sa, sb = _hk_fp8_scales_to_dev(a_scale_inv, b_scale_inv)
        a_in = a if a.is_contiguous() else a.contiguous()
        b_in = b if b.is_contiguous() else b.contiguous()
        op = torch.ops.primus_turbo_cpp_extension.hk_gemm_fp8
        # CRR + trans_c shortcut. ``(A^T @ B)^T == B^T @ A``: swap (a, b)
        # and (sa, sb) into the CRR kernel to write [n, m] directly,
        # saving the post-GEMM transpose copy.
        if trans_c and layout == "crr":
            return op(b_in, a_in, sb, sa, layout, 4, out_dtype)
        c = op(a_in, b_in, sa, sb, layout, 4, out_dtype)
        return c.t().contiguous() if trans_c else c


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


_GEMM_FP8_BACKENDS = {
    BackendType.HIPKITTEN: BackendEntry(GEMMFP8HipKittenBackend, autotune=False),
    BackendType.TURBO: BackendEntry(GEMMFP8TurboBackend),
    BackendType.HIPBLASLT: BackendEntry(GEMMFP8HipBLASLtBackend),
    BackendType.CK: BackendEntry(GEMMFP8CKBackend),
    BackendType.TRITON: BackendEntry(GEMMFP8TritonBackend),
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
