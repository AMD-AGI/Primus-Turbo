# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-dispatch helpers for the vendored gfx1250 forward kernel.

Trimmed to what this package uses: ``_run_compiled`` (fwd host entry) and ``_to_raw``
(fmha_b16_buffer_managers). The aiter original also carries the stable-API
buffer-view helpers (``ptr_buf_tensor`` / ``buf_copy_*``, built on
``fx.rocdl.make_buffer_tensor``) and ``GTensor``; see ../PROVENANCE.md.
"""

import flydsl.compiler as flyc
from flydsl._mlir import ir

# One CompiledFunction per @flyc.jit launcher. Kept here rather than as an
# attribute on the JitFunction so no FlyDSL object carries a field it does not
# define.
_COMPILED = {}


def _run_compiled(exe, *args):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.
    """
    cf = _COMPILED.get(exe)
    if cf is not None:
        cf(*args)
        return
    try:
        cf = flyc.compile(exe, *args)
        _COMPILED[exe] = cf
    except Exception:
        # flyc.compile leaks ir.Context on failure; pop it so a retry takes the right path.
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:  # noqa: BLE001, S110
            pass
        raise


def _to_raw(v):
    """Convert ArithValue / Numeric (Int32, Boolean, …) to raw ir.Value.

    Legacy shim kept for fmha_b16_buffer_managers; new code should call
    ``v.ir_value()`` (or ``fx.as_ir_value(v)``) at the raw-IR boundary.
    """
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "ir_value"):
        return _to_raw(v.ir_value())
    return ir.Value._CAPICreate(v._CAPIPtr)
