###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib
import os
import sys
from typing import Optional, Tuple

import torch

_torch_custom_op_wrapper = torch.library.custom_op

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
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

    SUPPORTED_DTYPES = set(_COMMON_SUPPORTED_DTYPES)

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
      - TENSORWISE: per-tensor scaling (all layouts)
      - ROWWISE: per-row/per-col vector scaling (all layouts)
      - BLOCKWISE: block-wise scaling with three layouts:
          NT/RCR (forward), NN/RRR (grad_X), TN/CRR (grad_W)
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


_tk_module = None
_tk_module_key = None


def _read_optional_int_env(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _get_hipkittens_build_shape() -> Optional[Tuple[int, int, int]]:
    build_shape = (
        _read_optional_int_env("PRIMUS_TURBO_HIPKITTENS_BUILD_M"),
        _read_optional_int_env("PRIMUS_TURBO_HIPKITTENS_BUILD_N"),
        _read_optional_int_env("PRIMUS_TURBO_HIPKITTENS_BUILD_K"),
    )
    if any(dim is None for dim in build_shape):
        return None
    return build_shape


def _get_tk_module():
    global _tk_module, _tk_module_key

    module_name = os.environ.get("PRIMUS_TURBO_HIPKITTENS_MODULE", "tk_fp8_layouts")
    module_dir = os.environ.get("PRIMUS_TURBO_HIPKITTENS_MODULE_DIR")
    module_key = (module_name, module_dir)

    if _tk_module is not None and _tk_module_key == module_key:
        return _tk_module

    if module_dir and module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None

    _tk_module = module
    _tk_module_key = module_key
    return module


def _pad_fp8_tensor(x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if x.shape == (rows, cols):
        return x if x.is_contiguous() else x.contiguous()

    padded = torch.zeros((rows, cols), dtype=x.dtype, device=x.device)
    padded[: x.shape[0], : x.shape[1]].copy_(x)
    return padded


class GEMMFP8HipKittensBackend(KernelBackend):
    """Conservative HipKittens bridge for strict tensorwise RCR/RRR/CRR paths."""

    SUPPORTED_GRANULARITIES = {ScalingGranularity.TENSORWISE}
    SUPPORTED_DTYPES = {
        (float8_e4m3, float8_e4m3, torch.bfloat16),
    }
    BLK = 256
    BK = 128

    @staticmethod
    def _resolve_request(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        trans_c: bool,
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

        if not trans_lhs and trans_rhs:
            layout = "rcr"
        elif not trans_lhs and not trans_rhs:
            layout = "rrr"
        elif trans_lhs and not trans_rhs:
            layout = "crr"
        else:
            return None

        return layout, lhs, lhs_scale_inv, trans_lhs, rhs, rhs_scale_inv, trans_rhs

    @staticmethod
    def _pad_inputs_for_layout(
        layout: str,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        build_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        build_m, build_n, build_k = build_shape

        if layout == "rcr":
            lhs_padded = _pad_fp8_tensor(lhs, build_m, build_k)
            rhs_padded = _pad_fp8_tensor(rhs, build_n, build_k)
        elif layout == "rrr":
            lhs_padded = _pad_fp8_tensor(lhs, build_m, build_k)
            rhs_padded = _pad_fp8_tensor(rhs, build_k, build_n)
        elif layout == "crr":
            lhs_padded = _pad_fp8_tensor(lhs, build_k, build_m)
            rhs_padded = _pad_fp8_tensor(rhs, build_k, build_n)
        else:
            raise ValueError(f"Unsupported HipKittens layout: {layout}")

        return lhs_padded, rhs_padded

    @staticmethod
    def _get_runtime_plan(
        a: torch.Tensor,
        a_scale_inv: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        b_scale_inv: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        granularity: ScalingGranularity,
    ) -> Optional[tuple]:
        if granularity not in GEMMFP8HipKittensBackend.SUPPORTED_GRANULARITIES:
            return None
        if (a.dtype, b.dtype, out_dtype) not in GEMMFP8HipKittensBackend.SUPPORTED_DTYPES:
            return None

        if a.device != b.device or a.device.type != "cuda":
            return None
        if a.ndim != 2 or b.ndim != 2:
            return None
        if a_scale_inv.numel() != 1 or b_scale_inv.numel() != 1:
            return None

        request = GEMMFP8HipKittensBackend._resolve_request(
            a, a_scale_inv, trans_a, b, b_scale_inv, trans_b, trans_c
        )
        if request is None:
            return None
        layout, lhs, lhs_scale_inv, trans_lhs, rhs, rhs_scale_inv, trans_rhs = request

        tk = _get_tk_module()
        if tk is None:
            return None
        gemm_fn = getattr(tk, f"gemm_{layout}", None)
        if gemm_fn is None:
            return None

        build_shape = _get_hipkittens_build_shape()
        if build_shape is None:
            return None
        build_m, build_n, build_k = build_shape
        if build_m <= 0 or build_n <= 0 or build_k <= 0:
            return None
        if build_m % GEMMFP8HipKittensBackend.BLK != 0:
            return None
        if build_n % GEMMFP8HipKittensBackend.BLK != 0:
            return None
        if build_k % GEMMFP8HipKittensBackend.BK != 0:
            return None

        m, n, k = get_gemm_logical_shape(lhs, rhs, trans_lhs, trans_rhs)
        if k % GEMMFP8HipKittensBackend.BK != 0:
            return None
        if m > build_m or n > build_n or k > build_k:
            return None

        return gemm_fn, layout, (m, n, k), build_shape, lhs, rhs, lhs_scale_inv, rhs_scale_inv

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
        plan = GEMMFP8HipKittensBackend._get_runtime_plan(
            a,
            a_scale_inv,
            trans_a,
            b,
            b_scale_inv,
            trans_b,
            out_dtype,
            trans_c,
            granularity,
        )
        return plan is not None

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
        plan = GEMMFP8HipKittensBackend._get_runtime_plan(
            a,
            a_scale_inv,
            trans_a,
            b,
            b_scale_inv,
            trans_b,
            out_dtype,
            trans_c,
            granularity,
        )
        if plan is None:
            raise ValueError(
                "HipKittens only supports tensorwise E4M3 strict RCR/RRR/CRR with a configured build shape"
            )

        gemm_fn, layout, (m, n, k), (build_m, build_n, build_k), lhs, rhs, lhs_scale_inv, rhs_scale_inv = plan
        lhs, rhs = GEMMFP8HipKittensBackend._pad_inputs_for_layout(
            layout, lhs, rhs, (build_m, build_n, build_k)
        )
        out = torch.empty((build_m, build_n), dtype=torch.bfloat16, device=a.device)

        scale = lhs_scale_inv.reshape(()).to(dtype=torch.float32) * rhs_scale_inv.reshape(()).to(
            dtype=torch.float32
        )
        scaled_in_kernel = False

        try:
            gemm_fn(lhs, rhs, out, float(scale.item()))
            scaled_in_kernel = True
        except TypeError:
            gemm_fn(lhs, rhs, out)

        out = out if (m, n) == (build_m, build_n) else out[:m, :n].contiguous()
        if not scaled_in_kernel:
            # Older tk_fp8_layouts builds do not accept an explicit scale argument.
            # Apply the tensorwise scale in fp32 here to avoid losing extra dB to
            # bf16 scalar rounding before the final cast.
            out = out.float() * scale
        if out.dtype != out_dtype:
            out = out.to(out_dtype)
        return out


_GEMM_FP8_BACKENDS = {
    BackendType.HIPBLASLT: BackendEntry(GEMMFP8HipBLASLtBackend),
    BackendType.CK: BackendEntry(GEMMFP8CKBackend),
    BackendType.TRITON: BackendEntry(GEMMFP8TritonBackend),
    BackendType.HIPKITTENS: BackendEntry(GEMMFP8HipKittensBackend, autotune=False),
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
    user_backend_enum = GlobalBackendManager.get_gemm_backend()
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
