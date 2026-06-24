###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Lazy loader for the fused MoE router ASM .co kernel binary."""

import ctypes
import os
from pathlib import Path

from primus_turbo.asm_co.hip_utils import load_co_func

__all__ = [
    "ASM_CO_ROUTER_PATH",
    "get_asm_co_router_func",
]

_PKG_ASM_KERNELS_DIR = Path(__file__).resolve().parent.parent / "asm_kernels"
_ENV_OVERRIDE = os.environ.get("PRIMUS_TURBO_ASM_CO_DIR")

ASM_CO_ROUTER_PATH = (
    os.path.join(_ENV_OVERRIDE, "fused_router_opt.co")
    if _ENV_OVERRIDE is not None
    else str(_PKG_ASM_KERNELS_DIR / "fused_router" / "final.co")
)
_ASM_CO_ROUTER_KERNEL_NAME = "fused_scaling_group_sum_routing_kernel"

_ASM_CO_ROUTER_MODULE: ctypes.c_void_p | None = None
_ASM_CO_ROUTER_FUNC:   ctypes.c_void_p | None = None


def get_asm_co_router_func() -> ctypes.c_void_p:
    """Return the fused router ASM function handle, loading the .co on first call."""
    global _ASM_CO_ROUTER_MODULE, _ASM_CO_ROUTER_FUNC
    if _ASM_CO_ROUTER_FUNC is None:
        _ASM_CO_ROUTER_MODULE, _ASM_CO_ROUTER_FUNC = load_co_func(
            ASM_CO_ROUTER_PATH, _ASM_CO_ROUTER_KERNEL_NAME
        )
    return _ASM_CO_ROUTER_FUNC
