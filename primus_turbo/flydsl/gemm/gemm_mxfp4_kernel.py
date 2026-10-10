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

"""Select a packaged implementation once, without modifying installed sources."""

from primus_turbo.common.flux import enabled, p3_enabled

if p3_enabled():
    # P3 reached the packed-scale layout that the base revision had no API for; upstream has
    # since grown that API natively, and its kernel measures faster on the recipe's shapes, so
    # it is what `auto` selects. FLUX_MXFP4_P3_GEMM=1 still pins P3 for the ablation.
    if enabled("FLUX_MXFP4_P3_GEMM"):
        from . import gemm_mxfp4_kernel_p3 as _implementation
    else:
        from . import gemm_mxfp4_kernel_main as _implementation
else:
    from . import gemm_mxfp4_kernel_stock as _implementation

__all__ = [name for name in dir(_implementation) if not name.startswith("_")]


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
