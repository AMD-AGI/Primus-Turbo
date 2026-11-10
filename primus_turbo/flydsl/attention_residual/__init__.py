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

"""Fused FlyDSL attention-residual backend, gfx950 / CDNA4 only.

Importing this package pulls in ``flydsl``, so it must only ever be reached
through :func:`primus_turbo.pytorch.ops.attention_residual.load_flydsl_attn_res_backend`, never from
module scope of anything on the default import path. Same lazy-loader rationale
as :mod:`primus_turbo.flydsl.attention.kda`.
"""

from primus_turbo.flydsl.attention_residual.attn_res_mixer_bwd_kernel import (
    BWD_INJECTIONS,
)
from primus_turbo.flydsl.attention_residual.attn_res_mixer_kernel import (
    FWD_INJECTIONS,
    FWD_NEUTRAL_VARIANTS,
)
from primus_turbo.flydsl.attention_residual.mixer import (
    flydsl_attn_res_mix,
    flydsl_mix_kernel,
    inject_defect,
    supports_mixer_inputs,
)

__all__ = [
    "flydsl_attn_res_mix",
    "flydsl_mix_kernel",
    "supports_mixer_inputs",
    # test-only: proving the parity assertions have discrimination power
    "inject_defect",
    "FWD_INJECTIONS",
    "FWD_NEUTRAL_VARIANTS",
    "BWD_INJECTIONS",
]
