###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

from .runtime import (
    DispatchState,
    MegaMoEPrecision,
    MegaPrologueFacade,
    MegaRuntime,
    MegaRuntimeRegistry,
    MegaShape,
)

_LAZY_EXPORTS = {
    "dispatch_grouped_gemm_bf16_flydsl_kernel": (
        ".dispatch_grouped_gemm_bf16_kernel",
        "dispatch_grouped_gemm_bf16_flydsl_kernel",
    ),
    "dispatch_prologue_flydsl_kernel": (
        ".dispatch_prologue_kernel",
        "dispatch_prologue_flydsl_kernel",
    ),
    "grouped_gemm_combine_bf16_flydsl_kernel": (
        ".grouped_gemm_combine_bf16_kernel",
        "grouped_gemm_combine_bf16_flydsl_kernel",
    ),
}


def __getattr__(name):
    """Keep runtime contracts importable without importing the optional FlyDSL compiler."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    import importlib

    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "DispatchState",
    "dispatch_grouped_gemm_bf16_flydsl_kernel",
    "dispatch_prologue_flydsl_kernel",
    "grouped_gemm_combine_bf16_flydsl_kernel",
    "MegaMoEPrecision",
    "MegaPrologueFacade",
    "MegaRuntime",
    "MegaRuntimeRegistry",
    "MegaShape",
]
