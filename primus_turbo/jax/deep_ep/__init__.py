###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Low-level DeepEP runtime constants.

Most users want :mod:`primus_turbo.jax.lax.moe` instead, which provides
the high-level ``setup`` / ``moe_dispatch`` / ``moe_combine`` API.

This package's ``__init__`` exposes only the small set of constants and
enums that framework code (e.g. MaxText mode-validation checks, custom
shardings) needs at import time.  Advanced callers can reach into the
underlying module with::

    from primus_turbo.jax.deep_ep import runtime as deep_ep_runtime
    deep_ep_runtime.pin_ep_group_from_jax_mesh(mesh)  # etc.
"""

from .runtime import MODE_INPROC, MODE_PER_PROCESS, NUM_MAX_NVL_PEERS, LaunchMode

__all__ = [
    "LaunchMode",
    "MODE_INPROC",
    "MODE_PER_PROCESS",
    "NUM_MAX_NVL_PEERS",
]
