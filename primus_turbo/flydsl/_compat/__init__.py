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

"""Compatibility shims that let the Primus-Turbo FlyDSL kernels run on newer flydsl releases.

``install()`` is called from ``primus_turbo/flydsl/__init__.py`` before any kernel module is
imported. It only fills gaps: anything the installed flydsl still ships is left untouched,
so on the pinned flydsl (``setup.py``) it is a no-op.
"""

import importlib.util
import sys

__all__ = ["install"]

_BUFFER_OPS = "flydsl.expr.buffer_ops"


def _ships(module_name: str) -> bool:
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def install() -> None:
    try:
        import flydsl.expr as fx_expr
    except ImportError:  # flydsl is optional; its absence is reported where it is used
        return

    # flydsl 0.3.x removed flydsl.expr.buffer_ops (moved to the repo-level, unshipped
    # kernels/common/ with a different resource type); provide the 0.2.4 API.
    if not _ships(_BUFFER_OPS):
        from primus_turbo.flydsl._compat import buffer_ops

        sys.modules[_BUFFER_OPS] = buffer_ops
        fx_expr.buffer_ops = buffer_ops

    # flydsl 0.3.x dropped the deprecated top-level ``fx.ArithValue`` re-export (still
    # ``flydsl.expr.arith.ArithValue``); Turbo uses it in annotations evaluated at import.
    if "ArithValue" not in vars(fx_expr):
        from flydsl.expr.arith import ArithValue

        fx_expr.ArithValue = ArithValue
