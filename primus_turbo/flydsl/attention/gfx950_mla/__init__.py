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

"""gfx950 FlyDSL attention for D_QK != D_V (DeepSeek-V3 MLA 192 / 128): dual-wave forward and fused
backward, standalone (not wired into ``turbo.ops`` dispatch)."""

from .ops import mla_fwd_sbhd
from .ops_bwd import mla_bwd_sbhd

__all__ = ["mla_bwd_sbhd", "mla_fwd_sbhd"]
