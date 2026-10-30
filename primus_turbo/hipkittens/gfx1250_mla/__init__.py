###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gfx1250 (MI455X) MLA attention forward and backward on HIP kernels (head dims qk 192 / v 128).

Selected with ``PRIMUS_TURBO_MLA_FWD_IMPL=hk`` / ``PRIMUS_TURBO_MLA_BWD_IMPL=hk`` (default ``flydsl``); the entry
points are ``fwd.hk_attn_fwd`` and ``bwd.hk_attn_bwd``. The kernels are HIP sources (``hip/``) built at install time
into ``bin/*.hsaco`` and launched through ``hiprt`` (ctypes, torch's HIP runtime); ``hsaco`` builds and verifies them
against ``manifest.json``. Importing this package imports nothing; the entry points import torch on first access.
"""

__all__ = ["hk_attn_fwd", "hk_attn_bwd"]


def __getattr__(name):
    if name == "hk_attn_fwd":
        from .fwd import hk_attn_fwd

        return hk_attn_fwd
    if name == "hk_attn_bwd":
        from .bwd import hk_attn_bwd

        return hk_attn_bwd
    raise AttributeError(name)
