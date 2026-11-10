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

"""Native FlyDSL KDA backend, gfx950 / CDNA4 only.

Importing this package pulls in ``flydsl``, so it must only ever be reached
through :func:`primus_turbo.pytorch.ops.attention.kda.load_flydsl_kda_backend`, never from module
scope of anything on the default import path. See the lazy-loader rationale
in :mod:`primus_turbo.pytorch.ops.attention.kda` and its DeepSeek-V4 precedent
(``primus_turbo/pytorch/ops/attention/deepseek_v4/__init__.py``).
"""

from primus_turbo.flydsl.attention.kda.chunk import (
    flydsl_chunk_kda,
    flydsl_chunk_kda_bwd,
    flydsl_chunk_kda_fwd,
)

__all__ = ["flydsl_chunk_kda", "flydsl_chunk_kda_fwd", "flydsl_chunk_kda_bwd"]
