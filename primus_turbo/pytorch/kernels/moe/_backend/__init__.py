###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""EP backend implementations; submodules touch optional deps only when imported."""

from ._config import EPBufferConfig
from .base import (
    EPBackend,
    _broadcast_from_rank0_float,
    _broadcast_from_rank0_int,
    _DeepEPLikeBackend,
    bench,
    detect_group_topology,
)
from .deep_ep import DeepEPBackend
from .mori import MoriEPBackend
from .turbo import TurboEPBackend
from .uccl_ep import UCCLEPBackend

__all__ = [
    "EPBackend",
    "EPBufferConfig",
    "_DeepEPLikeBackend",
    "_broadcast_from_rank0_float",
    "_broadcast_from_rank0_int",
    "bench",
    "detect_group_topology",
    "TurboEPBackend",
    "DeepEPBackend",
    "UCCLEPBackend",
    "MoriEPBackend",
]
