###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""One check for every gfx1250 Gluon (cdna5) kernel in Primus-Turbo.

The gfx1250 Gluon kernels (TDM loads, scaled WMMA, partitioned shared layouts) need the
``gluon.language.amd.cdna5`` API of a Triton release from 3.9 on. Older releases can still import
parts of that namespace but fail or hang when a kernel is compiled, so an import test is not a
reliable gate. Every caller asks :func:`gluon_ok` instead and keeps a non-Gluon path for the case
where it answers False.
"""

import functools
import logging
import re

_log = logging.getLogger("primus_turbo.gluon")

MIN_TRITON = (3, 9)


def triton_version() -> tuple:
    """``(major, minor)`` of the installed Triton, ``(0, 0)`` when it is missing or unparsable."""
    try:
        import triton

        m = re.match(r"(\d+)\.(\d+)", triton.__version__)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    except Exception:
        return (0, 0)


def why_not() -> str:
    """The reason the Gluon kernels are off, or "" when they are on."""
    from primus_turbo.common import envs

    if not envs.PRIMUS_GLUON:
        return "PRIMUS_GLUON=0"
    from primus_turbo.pytorch.core.utils import is_gfx1250

    if not is_gfx1250():
        return "the device is not gfx1250"
    if triton_version() < MIN_TRITON:
        return f"Triton {triton_version()} is older than {MIN_TRITON}"
    try:
        from triton.experimental.gluon.language.amd.cdna5 import tdm  # noqa: F401
    except Exception as e:  # a Triton without gluon cdna5
        return f"gluon cdna5 is not importable ({type(e).__name__}: {e})"
    return ""


@functools.lru_cache(maxsize=None)
def gluon_ok() -> bool:
    """True when the gfx1250 Gluon kernels can be used: a gfx1250 device, Triton >= 3.9 with
    ``gluon.language.amd.cdna5``, and ``PRIMUS_GLUON`` not set to 0."""
    reason = why_not()
    if reason:
        _log.info("gfx1250 Gluon kernels are off: %s; using the non-Gluon kernels", reason)
    return not reason


def require_gluon(what: str) -> None:
    """Raise with the reason when the Gluon kernels are off (for entry points with no fallback)."""
    if not gluon_ok():
        raise RuntimeError(f"{what} needs the gfx1250 Gluon kernels, which are off: {why_not()}")
