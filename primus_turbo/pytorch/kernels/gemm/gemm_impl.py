###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

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
from primus_turbo.triton.gemm.gemm_kernel import gemm_triton_kernel

_COMMON_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_HIPBLASLT_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_HIPKITTEN_MODULE_ENV = "PRIMUS_TURBO_HIPKITTEN_MODULE"
_HIPKITTEN_BF16_PATH_ENV = "PRIMUS_TURBO_HIPKITTEN_BF16_PATH"
_HIPKITTEN_PATH_ENV = "PRIMUS_TURBO_HIPKITTEN_PATH"


@lru_cache(maxsize=1)
def _load_hipkitten_module():
    module_name = os.environ.get(_HIPKITTEN_MODULE_ENV, "tk_bf16_layouts")
    try:
        module = importlib.import_module(module_name)
        return module, _load_hipkitten_bf16_cache(None)
    except ImportError as import_error:
        search_paths = []
        if bf16_path := os.environ.get(_HIPKITTEN_BF16_PATH_ENV):
            search_paths.append(Path(bf16_path))
        if hipkitten_root := os.environ.get(_HIPKITTEN_PATH_ENV):
            root_path = Path(hipkitten_root)
            search_paths.extend(
                [
                    root_path,
                    root_path / "analysis" / "bf16_gemm" / "mi350x",
                ]
            )
        for search_path in search_paths:
            if search_path.is_dir():
                sys.path.insert(0, str(search_path))
                try:
                    module = importlib.import_module(module_name)
                    return module, _load_hipkitten_bf16_cache(search_path)
                except ImportError:
                    pass

        raise ImportError(
            f"HipKitten GEMM backend requires importable '{module_name}'. "
            f"Build HipKittens analysis/bf16_gemm/mi350x and set {_HIPKITTEN_BF16_PATH_ENV}, "
            f"{_HIPKITTEN_PATH_ENV}, or {_HIPKITTEN_MODULE_ENV}."
        ) from import_error


def _load_hipkitten_bf16_cache(search_path: Path | None) -> dict:
    candidate_paths = []
    if search_path is not None:
        candidate_paths.append(search_path / "bench_bf16_no_jit_final.json")
    if bf16_path := os.environ.get(_HIPKITTEN_BF16_PATH_ENV):
        candidate_paths.append(Path(bf16_path) / "bench_bf16_no_jit_final.json")
    if hipkitten_root := os.environ.get(_HIPKITTEN_PATH_ENV):
        candidate_paths.append(
            Path(hipkitten_root) / "analysis" / "bf16_gemm" / "mi350x" / "bench_bf16_no_jit_final.json"
        )

    for candidate_path in candidate_paths:
        if candidate_path.is_file():
            with candidate_path.open() as f:
                data = json.load(f)
            cache = {}
            for row in data.get("rows", []):
                key = (int(row["M"]), int(row["N"]), int(row["K"]))
                cache[key] = {
                    "rcr": (int(row.get("rcr_gm", 4)), int(row.get("rcr_xcd", 8))),
                    "rrr": (int(row.get("rrr_gm", 4)), int(row.get("rrr_xcd", 8))),
                    "crr": (int(row.get("crr_gm", 4)), int(row.get("crr_xcd", 8))),
                }
            return cache
    return {}


def _has_hipkitten_module() -> bool:
    try:
        _load_hipkitten_module()
        return True
    except ImportError:
        return False


class GEMMHipBLASLtBackend(KernelBackend):

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.ndim == 2 and b.ndim == 2
        supported &= a.dtype in _HIPBLASLT_SUPPORTED_DTYPES and b.dtype in _HIPBLASLT_SUPPORTED_DTYPES
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        **kwargs,
    ) -> torch.Tensor:
        return torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm(a, b, out_dtype, trans_a, trans_b, trans_c)


class GEMMTritonBackend(KernelBackend):

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= a.ndim == 2 and b.ndim == 2
        supported &= a.dtype in _COMMON_SUPPORTED_DTYPES and b.dtype in _COMMON_SUPPORTED_DTYPES
        return supported

    @staticmethod
    def execute(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        **kwargs,
    ) -> torch.Tensor:
        return gemm_triton_kernel(a, b, trans_a, trans_b, out_dtype, trans_c)


class GEMMHipKittenBackend(KernelBackend):
    TILE_MN = 256
    TILE_K = 64

    @staticmethod
    def _layout_name(trans_a: bool, trans_b: bool) -> str | None:
        if not trans_a and trans_b:
            return "rcr"
        if not trans_a and not trans_b:
            return "rrr"
        if trans_a and not trans_b:
            return "crr"
        return None

    @staticmethod
    def _logical_shape(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
    ) -> tuple[int, int, int, int]:
        m = a.shape[1] if trans_a else a.shape[0]
        k_a = a.shape[0] if trans_a else a.shape[1]
        n = b.shape[0] if trans_b else b.shape[1]
        k_b = b.shape[1] if trans_b else b.shape[0]
        return m, n, k_a, k_b

    @staticmethod
    def _can_use_hipkitten_kernel(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        trans_c: bool,
    ) -> bool:
        m, n, k_a, k_b = GEMMHipKittenBackend._logical_shape(a, trans_a, b, trans_b)
        if k_a != k_b:
            return False
        if GEMMHipKittenBackend._layout_name(trans_a, trans_b) is None:
            return False

        return (
            m % GEMMHipKittenBackend.TILE_MN == 0
            and n % GEMMHipKittenBackend.TILE_MN == 0
            and k_a % GEMMHipKittenBackend.TILE_K == 0
        )

    @staticmethod
    def can_handle(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        **kwargs,
    ) -> bool:
        if a.ndim != 2 or b.ndim != 2:
            return False
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or out_dtype != torch.bfloat16:
            return False
        if not a.is_cuda or not b.is_cuda or a.device != b.device:
            return False
        if not _has_hipkitten_module():
            return False

        if not GEMMHipKittenBackend._can_use_hipkitten_kernel(a, trans_a, b, trans_b, trans_c):
            return False

        _, cache = _load_hipkitten_module()
        m, n, k, _ = GEMMHipKittenBackend._logical_shape(a, trans_a, b, trans_b)
        layout = GEMMHipKittenBackend._layout_name(trans_a, trans_b)
        return (m, n, k) in cache and layout in cache[(m, n, k)]

    @staticmethod
    def execute(
        a: torch.Tensor,
        trans_a: bool,
        b: torch.Tensor,
        trans_b: bool,
        out_dtype: torch.dtype,
        trans_c: bool,
        **kwargs,
    ) -> torch.Tensor:
        hipkitten, cache = _load_hipkitten_module()
        m, n, k, _ = GEMMHipKittenBackend._logical_shape(a, trans_a, b, trans_b)
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
        shape_cfg = cache[(m, n, k)]

        if not trans_a and trans_b:
            group_m, num_xcds = shape_cfg["rcr"]
            hipkitten.gemm_rcr(a.contiguous(), b.contiguous(), out, group_m, num_xcds)
        elif not trans_a and not trans_b:
            group_m, num_xcds = shape_cfg["rrr"]
            hipkitten.gemm_rrr(a.contiguous(), b.contiguous(), out, group_m, num_xcds)
        elif trans_a and not trans_b:
            group_m, num_xcds = shape_cfg["crr"]
            hipkitten.gemm_crr(a.contiguous(), b.contiguous(), out, group_m, num_xcds)
        else:
            raise ValueError("HipKitten BF16 backend supports RCR, RRR, and CRR layouts only.")
        return out.t().contiguous() if trans_c else out


_GEMM_BACKENDS = {
    BackendType.HIPBLASLT: BackendEntry(GEMMHipBLASLtBackend),
    BackendType.HIPKITTEN: BackendEntry(GEMMHipKittenBackend, autotune=False),
    BackendType.TRITON: BackendEntry(GEMMTritonBackend),
}


class GEMMKernelDispatcher(AutoKernelDispatcher):
    _backends = _GEMM_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, a, b, trans_a, trans_b, out_dtype, trans_c, **kwargs):
        M = a.shape[1] if trans_a else a.shape[0]
        Ka = a.shape[0] if trans_a else a.shape[1]
        N = b.shape[0] if trans_b else b.shape[1]
        return (M, N, Ka, a.dtype, b.dtype, out_dtype, trans_a, trans_b, trans_c)


@_torch_custom_op_wrapper("primus_turbo::gemm_impl", mutates_args=(), device_types="cuda")
def gemm_impl(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    default_backend: int,
) -> torch.Tensor:
    default_backend_enum = BackendType(default_backend)
    user_backend_enum = GlobalBackendManager.get_gemm_backend(PrecisionType.BF16_FP16_FP32)

    kwargs = dict(
        a=a,
        trans_a=trans_a,
        b=b,
        trans_b=trans_b,
        out_dtype=out_dtype,
        trans_c=trans_c,
    )

    return GEMMKernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


@gemm_impl.register_fake
def gemm_impl_meta(
    a: torch.Tensor,
    trans_a: bool,
    b: torch.Tensor,
    trans_b: bool,
    out_dtype: torch.dtype,
    trans_c: bool,
    default_backend: int,
) -> torch.Tensor:
    assert (
        a.ndim == 2 and b.ndim == 2
    ), f"Expected both a and b to be 2D tensors, but got a.ndim={a.ndim}, b.ndim={b.ndim}"
    M = a.shape[1] if trans_a else a.shape[0]
    N = b.shape[0] if trans_b else b.shape[1]
    if trans_c:
        M, N = N, M
    return torch.empty(M, N, dtype=out_dtype, device=a.device)
