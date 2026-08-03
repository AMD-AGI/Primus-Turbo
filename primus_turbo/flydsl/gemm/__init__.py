###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
import importlib


@functools.lru_cache(maxsize=1)
def _blockwise_module():
    return importlib.import_module("primus_turbo.flydsl.gemm.blockscale_fp8_gemm")


def flydsl_blockwise_gemm_supported(
    m: int,
    n: int,
    k: int,
    *,
    allow_partial_n: bool = False,
) -> bool:
    try:
        if allow_partial_n:
            return _blockwise_module().flydsl_blockwise_4wave_forward_supported(m, n, k)
        return _blockwise_module().flydsl_blockwise_4wave_dgrad_supported(m, k, n)
    except (ImportError, OSError):
        return False


def flydsl_blockwise_wgrad_supported(m: int, n: int, k: int) -> bool:
    try:
        return _blockwise_module().flydsl_blockwise_4wave_wgrad_supported(k, n, m)
    except (ImportError, OSError):
        return False


def flydsl_blockwise_4wave_dgrad_supported(m: int, n: int, k: int) -> bool:
    try:
        return _blockwise_module().flydsl_blockwise_4wave_dgrad_supported(m, n, k)
    except (ImportError, OSError):
        return False


def flydsl_blockwise_4wave_forward_supported(m: int, n: int, k: int) -> bool:
    try:
        return _blockwise_module().flydsl_blockwise_4wave_forward_supported(m, n, k)
    except (ImportError, OSError):
        return False


def flydsl_blockwise_4wave_wgrad_supported(m: int, n: int, k: int) -> bool:
    try:
        return _blockwise_module().flydsl_blockwise_4wave_wgrad_supported(m, n, k)
    except (ImportError, OSError):
        return False


def gemm_fp8_blockwise_flydsl(*args, **kwargs):
    return _blockwise_module().gemm_fp8_blockwise_forward(*args, **kwargs)


def gemm_fp8_blockwise_flydsl_dgrad(*args, **kwargs):
    return _blockwise_module().gemm_fp8_blockwise_dgrad(*args, **kwargs)


def gemm_fp8_blockwise_flydsl_wgrad(*args, **kwargs):
    return _blockwise_module().gemm_fp8_blockwise_wgrad(*args, **kwargs)


__all__ = [
    "gemm_fp8_blockwise_flydsl",
    "gemm_fp8_blockwise_flydsl_dgrad",
    "gemm_fp8_blockwise_flydsl_wgrad",
    "flydsl_blockwise_gemm_supported",
    "flydsl_blockwise_wgrad_supported",
    "flydsl_blockwise_4wave_dgrad_supported",
    "flydsl_blockwise_4wave_forward_supported",
    "flydsl_blockwise_4wave_wgrad_supported",
]
