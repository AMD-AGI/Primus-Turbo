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

"""Launch ``@flyc.jit`` functions through cached ``flyc.compile`` objects (see
:mod:`primus_turbo.flydsl.utils.launch_cache`)."""

from primus_turbo.flydsl.utils.launch_cache import run_launcher

__all__ = ["run_launcher"]
