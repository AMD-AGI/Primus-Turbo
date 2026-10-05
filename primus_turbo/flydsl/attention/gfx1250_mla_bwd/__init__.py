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

"""FlyDSL flash-attention backward for gfx1250 (MI455X), DeepSeek-V3 MLA head dims (192 / 128).

Runs on gfx1250 only; the entry point is ``impl.flydsl_attn_bwd``. Importing the package raises
ImportError unless flydsl satisfies ``gfx1250_mla_version.FLYDSL_REQUIREMENT`` (0.3.4.x).
"""

from ..gfx1250_mla_version import require_flydsl

require_flydsl()

from .impl import flydsl_attn_bwd  # noqa: E402

__all__ = ["flydsl_attn_bwd"]
