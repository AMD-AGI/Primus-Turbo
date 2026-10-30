###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Environment overrides of the gfx1250 Mega MoE launch knobs, to sweep them without code changes.

Every knob keeps its shipped default while its variable is unset. With ``nn`` (the backward launches,
which read the weight in its NN layout) a ``<NAME>_NN`` variable wins over ``<NAME>``, so the forward
and backward launches of one kernel can be tuned apart. Knobs change compile-time constants, so a
sweep sets them per process.
"""

import os

__all__ = ["env_int"]


def env_int(name: str, default: int, nn: bool = False) -> int:
    for key in (name + "_NN", name) if nn else (name,):
        value = os.environ.get(key, "")
        if value != "":
            return int(value)
    return default
