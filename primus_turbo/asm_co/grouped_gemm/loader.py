###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Lazy loaders for grouped-GEMM FP8 ASM .co/.hsaco kernel binaries.

Each function loads the kernel on first call and caches the HIP module and
function handles for all subsequent calls.  Path constants are configurable
via the ``PRIMUS_TURBO_ASM_CO_DIR`` environment variable.
"""

import ctypes
import os
from pathlib import Path

from primus_turbo.asm_co.hip_utils import load_co_func

__all__ = [
    "get_asm_co_fwd_hsaco_func",
    "get_asm_co_dgrad_func",
    "get_asm_co_wgrad_func",
    "get_asm_co_wgrad_beta1_func",
]

# ── Directory / path constants ────────────────────────────────────────────────
# Package-relative default: asm_co/asm_kernels/ sits next to asm_co/grouped_gemm/.
_PKG_ASM_KERNELS_DIR = Path(__file__).resolve().parent.parent / "asm_kernels"
_ENV_OVERRIDE = os.environ.get("PRIMUS_TURBO_ASM_CO_DIR")

# ── Shared kernel name (baked into both FWD and DGRAD .hsaco files) ───────────
_PERSISTENT_GEMM_KERNEL_NAME = "_grouped_fp8_persistent_gemm_kernel"


def _resolve_path(subdir: str, filename: str, legacy_filename: str | None = None) -> str:
    """Return the kernel binary path, honoring PRIMUS_TURBO_ASM_CO_DIR override.

    When the env var is set, files are expected under a flat directory with
    the legacy Dockerfile-era names.  Otherwise, the package-relative tree
    structure is used directly.
    """
    if _ENV_OVERRIDE is not None:
        return os.path.join(_ENV_OVERRIDE, legacy_filename or filename)
    return str(_PKG_ASM_KERNELS_DIR / subdir / filename)


# ── FWD .hsaco paths keyed by N dimension ────────────────────────────────────
_ASM_CO_FWD_HSACO_PATHS: dict[int, str] = {
    5760: _resolve_path("grouped_gemm_fwd", "base_gate_up_5760.hsaco", "reference_grouped_gemm_fwd_5760.hsaco"),
    2880: _resolve_path("grouped_gemm_fwd", "base_down_2880.hsaco", "reference_grouped_gemm_fwd_2880.hsaco"),
}

_ASM_CO_FWD_MODULES: dict[int, ctypes.c_void_p] = {}
_ASM_CO_FWD_FUNCS:   dict[int, ctypes.c_void_p] = {}


def get_asm_co_fwd_hsaco_func(n: int) -> ctypes.c_void_p:
    """Return the FWD .hsaco function handle for output N dimension, loading on first call."""
    if n not in _ASM_CO_FWD_FUNCS:
        path = _ASM_CO_FWD_HSACO_PATHS[n]
        mod, func = load_co_func(path, _PERSISTENT_GEMM_KERNEL_NAME)
        _ASM_CO_FWD_MODULES[n] = mod
        _ASM_CO_FWD_FUNCS[n] = func
    return _ASM_CO_FWD_FUNCS[n]


# ── DGRAD .hsaco paths keyed by K dimension ───────────────────────────────────
_ASM_CO_DGRAD_HSACO_PATHS: dict[int, str] = {
    2880: _resolve_path("grouped_gemm_dgrad", "final_down_2880.hsaco", "dgrad_down_2880.hsaco"),
    5760: _resolve_path("grouped_gemm_dgrad", "final_gate_up_5760.hsaco", "dgrad_gate_up_5760.hsaco"),
}

_ASM_CO_DGRAD_MODULES: dict[int, ctypes.c_void_p] = {}
_ASM_CO_DGRAD_FUNCS:   dict[int, ctypes.c_void_p] = {}


def get_asm_co_dgrad_func(k: int) -> ctypes.c_void_p:
    """Return the DGRAD .hsaco function handle for K dimension, loading on first call."""
    if k not in _ASM_CO_DGRAD_FUNCS:
        path = _ASM_CO_DGRAD_HSACO_PATHS[k]
        mod, func = load_co_func(path, _PERSISTENT_GEMM_KERNEL_NAME)
        _ASM_CO_DGRAD_MODULES[k] = mod
        _ASM_CO_DGRAD_FUNCS[k] = func
    return _ASM_CO_DGRAD_FUNCS[k]


# ── WGRAD .co paths (variable-K, dot_scaled MFMA 32x32x64) ──────────────────
_ASM_CO_WGRAD_KERNEL_NAME = "grouped_variable_k_dot_scaled_kernel"

_ASM_CO_WGRAD_CO_PATH       = _resolve_path("grouped_gemm_wgrad", "intermediate.co", "dot_scaled_v2_fixed.co")
_ASM_CO_WGRAD_BETA1_CO_PATH = _resolve_path("grouped_gemm_wgrad", "final.co", "dot_scaled_v2_beta1.co")

_ASM_CO_WGRAD_MODULE: ctypes.c_void_p | None = None
_ASM_CO_WGRAD_FUNC:   ctypes.c_void_p | None = None

_ASM_CO_WGRAD_BETA1_MODULE: ctypes.c_void_p | None = None
_ASM_CO_WGRAD_BETA1_FUNC:   ctypes.c_void_p | None = None


def get_asm_co_wgrad_func() -> ctypes.c_void_p:
    """Return the variable-K wgrad .co function handle (beta=0), loading on first call."""
    global _ASM_CO_WGRAD_MODULE, _ASM_CO_WGRAD_FUNC
    if _ASM_CO_WGRAD_FUNC is None:
        _ASM_CO_WGRAD_MODULE, _ASM_CO_WGRAD_FUNC = load_co_func(
            _ASM_CO_WGRAD_CO_PATH, _ASM_CO_WGRAD_KERNEL_NAME
        )
    return _ASM_CO_WGRAD_FUNC


def get_asm_co_wgrad_beta1_func() -> ctypes.c_void_p:
    """Return the variable-K wgrad .co function handle (beta=1), loading on first call."""
    global _ASM_CO_WGRAD_BETA1_MODULE, _ASM_CO_WGRAD_BETA1_FUNC
    if _ASM_CO_WGRAD_BETA1_FUNC is None:
        _ASM_CO_WGRAD_BETA1_MODULE, _ASM_CO_WGRAD_BETA1_FUNC = load_co_func(
            _ASM_CO_WGRAD_BETA1_CO_PATH, _ASM_CO_WGRAD_KERNEL_NAME
        )
    return _ASM_CO_WGRAD_BETA1_FUNC
