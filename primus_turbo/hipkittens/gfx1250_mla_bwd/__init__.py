###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gfx1250 (MI455X) MLA attention backward on HIP kernels (head dims qk 192 / v 128).

Selected with ``PRIMUS_TURBO_MLA_BWD_IMPL=hk`` (default ``flydsl``); the entry point is
``impl.hk_attn_bwd``. The kernels are HIP sources (``hip/``) built at install time into
``bin/*.hsaco`` and launched through ``hiprt`` (ctypes, torch's HIP runtime); ``hsaco`` builds and
verifies them against ``manifest.json``. ``hip/include/cdna5`` vendors a subset of HipKittens (MIT).
Importing this package imports nothing; ``hk_attn_bwd`` imports torch and the FlyDSL backward
package (flydsl 0.3.4.x) on first access.
"""

__all__ = ["hk_attn_bwd"]


def __getattr__(name):
    if name == "hk_attn_bwd":
        from .impl import hk_attn_bwd

        return hk_attn_bwd
    raise AttributeError(name)
