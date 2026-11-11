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

from . import gemm_fp4_impl_stock as _stock

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
