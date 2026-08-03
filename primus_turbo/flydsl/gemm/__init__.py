###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
import importlib


@functools.lru_cache(maxsize=1)
def _blockwise_module():
    return importlib.import_module("primus_turbo.flydsl.gemm.gemm_fp8_blockwise_kernel")


def flydsl_blockwise_gemm_supported(
    m: int,
    n: int,
    k: int,
    *,
    allow_partial_n: bool = False,
) -> bool:
    try:
        return _blockwise_module().flydsl_blockwise_gemm_supported(
            m,
            n,
            k,
            allow_partial_n=allow_partial_n,
        )
    except (ImportError, OSError):
        return False


def flydsl_blockwise_wgrad_supported(m: int, n: int, k: int) -> bool:
    try:
        return _blockwise_module().flydsl_blockwise_wgrad_supported(m, n, k)
    except (ImportError, OSError):
        return False


def gemm_fp8_blockwise_flydsl(*args, **kwargs):
    return _blockwise_module().gemm_fp8_blockwise_flydsl(*args, **kwargs)


def gemm_fp8_blockwise_flydsl_dgrad(*args, **kwargs):
    return _blockwise_module().gemm_fp8_blockwise_flydsl_dgrad(*args, **kwargs)


def gemm_fp8_blockwise_flydsl_wgrad(*args, **kwargs):
    return _blockwise_module().gemm_fp8_blockwise_flydsl_wgrad(*args, **kwargs)


__all__ = [
    "gemm_fp8_blockwise_flydsl",
    "gemm_fp8_blockwise_flydsl_dgrad",
    "gemm_fp8_blockwise_flydsl_wgrad",
    "flydsl_blockwise_gemm_supported",
    "flydsl_blockwise_wgrad_supported",
]
