###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Kernel-level autotune for the FlyDSL builder ops, switched by ``PRIMUS_TURBO_AUTO_TUNE``. The
``@flyc.jit`` ops (rope, swiglu) keep ``mega/tune_utils.autotune`` instead."""

from primus_turbo.flydsl.autotune import runtime
from primus_turbo.flydsl.autotune.engine import NoWorkingConfig
from primus_turbo.flydsl.autotune.op import Compile, Op
from primus_turbo.flydsl.autotune.race import Race, Reference, Refine, staged

__all__ = [
    "runtime",
    "Op",
    "Race",
    "Refine",
    "Reference",
    "staged",
    "Compile",
    "NoWorkingConfig",
]
