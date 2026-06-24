###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Lazy loader for the SwiGLU BWD ASM .co kernel binary."""

import ctypes
import os
from pathlib import Path

from primus_turbo.asm_co.hip_utils import load_co_func

__all__ = [
    "ASM_CO_SWIGLU_BWD_PATH",
    "get_asm_co_swiglu_bwd_func",
]

_PKG_ASM_KERNELS_DIR = Path(__file__).resolve().parent.parent / "asm_kernels"
_ENV_OVERRIDE = os.environ.get("PRIMUS_TURBO_ASM_CO_DIR")

ASM_CO_SWIGLU_BWD_PATH = (
    os.path.join(_ENV_OVERRIDE, "swiglu_bwd_opt.co")
    if _ENV_OVERRIDE is not None
    else str(_PKG_ASM_KERNELS_DIR / "swiglu_bwd" / "final.co")
)
_ASM_CO_SWIGLU_BWD_KERNEL_NAME = "swiglu_with_mask_bwd_kernel"

_ASM_CO_SWIGLU_BWD_MODULE: ctypes.c_void_p | None = None
_ASM_CO_SWIGLU_BWD_FUNC:   ctypes.c_void_p | None = None


def get_asm_co_swiglu_bwd_func() -> ctypes.c_void_p:
    """Return the SwiGLU BWD ASM function handle, loading the .co on first call."""
    global _ASM_CO_SWIGLU_BWD_MODULE, _ASM_CO_SWIGLU_BWD_FUNC
    if _ASM_CO_SWIGLU_BWD_FUNC is None:
        _ASM_CO_SWIGLU_BWD_MODULE, _ASM_CO_SWIGLU_BWD_FUNC = load_co_func(
            ASM_CO_SWIGLU_BWD_PATH, _ASM_CO_SWIGLU_BWD_KERNEL_NAME
        )
    return _ASM_CO_SWIGLU_BWD_FUNC
