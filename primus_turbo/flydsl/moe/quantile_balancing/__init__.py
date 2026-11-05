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

"""Fused FlyDSL Quantile Balancing backend, gfx950 / CDNA4 only.

Importing this package pulls in ``flydsl``, so it must only ever be reached
through :func:`primus_turbo.pytorch.ops.moe.quantile_balancing.load_flydsl_qb_backend`, never from module scope of
anything on the default import path.
"""

from primus_turbo.flydsl.moe.quantile_balancing.histogram import (
    KERNEL_MAX_TOKENS,
    flydsl_compute_margin_histogram,
    inject_defect,
    kernel_beats_eager,
    supports_histogram_inputs,
)
from primus_turbo.flydsl.moe.quantile_balancing.qb_margin_histogram_kernel import (
    INJECTIONS,
    WIDTH_DEPENDENT_VARIANTS,
)

__all__ = [
    "flydsl_compute_margin_histogram",
    "supports_histogram_inputs",
    "kernel_beats_eager",
    "KERNEL_MAX_TOKENS",
    # test-only: proving the bit-exactness assertion has discrimination power
    "inject_defect",
    "INJECTIONS",
    "WIDTH_DEPENDENT_VARIANTS",
]
