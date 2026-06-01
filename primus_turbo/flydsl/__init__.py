###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""FlyDSL-backed kernels for Primus-Turbo.

FlyDSL (https://github.com/ROCm/FlyDSL) is a Python DSL + MLIR stack for
authoring high-performance AMD GPU kernels. This subpackage wraps FlyDSL
kernels so they can plug into Primus-Turbo's backend dispatch system.

All FlyDSL imports are performed lazily inside the launchers, so importing
``primus_turbo`` never fails when FlyDSL is not installed.
"""
