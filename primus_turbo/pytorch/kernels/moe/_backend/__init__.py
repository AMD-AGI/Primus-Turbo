###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""EP backend implementations split out from ``moe_dispatch_combine_impl``.

Each concrete backend lives in its own module so that optional dependencies
are only touched when the backend is actually imported / used. The public
entry points are re-exported here for convenience.
"""

from .base import (
    EPBackend,
    _broadcast_from_rank0_float,
    _broadcast_from_rank0_int,
    _DeepEPLikeBackend,
    _DeepEPLikeKernelName,
)
from .deep_ep import DeepEPBackend
from .mori import MoriEPBackend
from .turbo import TurboEPBackend
from .uccl_ep import UCCLEPBackend

__all__ = [
    "EPBackend",
    "_DeepEPLikeBackend",
    "_DeepEPLikeKernelName",
    "_broadcast_from_rank0_float",
    "_broadcast_from_rank0_int",
    "TurboEPBackend",
    "DeepEPBackend",
    "UCCLEPBackend",
    "MoriEPBackend",
]
