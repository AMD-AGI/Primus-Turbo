###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Select a packaged implementation once, without modifying installed sources."""

from primus_turbo.common.flux import p3_enabled

if p3_enabled():
    from . import gemm_fp4_impl_p3 as _implementation
else:
    from . import gemm_fp4_impl_stock as _implementation

__all__ = [name for name in dir(_implementation) if not name.startswith("_")]


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
