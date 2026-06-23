###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Lazy loader for the fused MoE router ASM .co kernel binary."""

import ctypes

from primus_turbo.asm_co.hip_utils import load_co_func

__all__ = [
    "ASM_CO_ROUTER_PATH",
    "get_asm_co_router_func",
]

ASM_CO_ROUTER_PATH        = "/opt/asm_gemm/fused_router_opt.co"
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
