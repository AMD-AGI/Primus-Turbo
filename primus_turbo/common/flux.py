###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Import-time FLUX kernel selection; never rewrites installed package files.

Keep the original implementations available for flag-off runs and ablations.
Environment choices are resolved once in each training process, before compilation.
"""

import os


def enabled(name, default="0"):
    return os.getenv(name, default) == "1"


def mxfp4_enabled():
    return os.getenv("FLUX_FP4_PASSES", "off") not in ("", "off")


def fused_quant_enabled():
    dispatch = os.getenv("FLUX_FP4_DISPATCH") or "fusion"
    return dispatch.lower() in ("fusion", "host_dispatch", "mxfp4_mm") or any(
        enabled(name) for name in ("FLUX_FP4_FUSED_H16_QUANT", "FLUX_FP4_HOST_DISPATCH", "FLUX_FP4_MXFP4_MM")
    )


def p3_enabled():
    # The base revision has no native packed-scale API, so auto selects P3.
    return (
        mxfp4_enabled()
        and fused_quant_enabled()
        and not enabled("FLUX_FP4_H16_STOCK")
        and os.getenv("FLUX_MXFP4_P3_GEMM", "auto") in ("auto", "1")
    )


def fp8_bias_enabled():
    active = mxfp4_enabled() and enabled("FLUX_FP8_FUSE_BIAS_EPILOGUE")
    if active and enabled("FLUX_FP8_BIAS_EPILOGUE_PERROW"):
        raise ValueError("FLUX_FP8_BIAS_EPILOGUE_PERROW=1 is not supported by the migrated recipe")
    return active
