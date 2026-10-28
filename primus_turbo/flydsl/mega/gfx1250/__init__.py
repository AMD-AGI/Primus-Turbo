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

"""gfx1250 Mega MoE building blocks (MI455X): wave32, WMMA, TDM, gfx12 waits.

Self-contained under this package: device primitives, EP intranode tiles, the
symmetric buffer and dispatch prologue live here, not in the gfx9 ``mega.bf16``
or ``flydsl.utils`` trees. Precision-specific kernels are under ``bf16`` (and
later ``fp8``). Import only after ``is_gfx1250()``.
"""
