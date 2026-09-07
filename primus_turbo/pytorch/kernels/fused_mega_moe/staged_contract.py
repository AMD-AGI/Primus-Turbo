###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compatibility exports for the shared MegaMoE runtime contract.

The canonical definitions live in :mod:`primus_turbo.flydsl.mega.runtime` so
FlyDSL workspace/prologue code and PyTorch autograd use one ABI source.
"""

from primus_turbo.flydsl.mega.runtime import (
    BF16_HANDLE_SCHEMA,
    MXFP8_HANDLE_SCHEMA,
    DispatchState,
    HandleSchema,
    MegaMoEPrecision,
    MegaShape,
    RouteState,
    StageState,
    WorkspaceAdapter,
    WorkspaceRegistry,
    WorkspaceRequest,
    handle_schema,
    make_route_state,
)

__all__ = [
    "BF16_HANDLE_SCHEMA",
    "MXFP8_HANDLE_SCHEMA",
    "DispatchState",
    "HandleSchema",
    "MegaMoEPrecision",
    "MegaShape",
    "RouteState",
    "StageState",
    "WorkspaceAdapter",
    "WorkspaceRegistry",
    "WorkspaceRequest",
    "handle_schema",
    "make_route_state",
]
