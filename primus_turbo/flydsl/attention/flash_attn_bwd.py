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

from primus_turbo.common.flux import enabled

if enabled("FLUX_ATTN_FLYDSL"):
    from . import flash_attn_bwd_flux as _implementation
else:
    from . import flash_attn_bwd_stock as _implementation

from . import flash_attn_bwd_stock as _stock

# The flux variant is a fork of an older stock snapshot. Names it lacks (APIs added to
# stock since) resolve to the current stock module, so public imports keep working.
__all__ = sorted(
    {n for n in dir(_implementation) if not n.startswith("_")}
    | {n for n in dir(_stock) if not n.startswith("_")}
)


def __getattr__(name):
    try:
        return getattr(_implementation, name)
    except AttributeError:
        return getattr(_stock, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)) | set(dir(_stock)))
