###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helpers for the optional ``aiter`` dependency.

aiter is imported lazily (only when an AITER-backed op runs). Primus-Turbo
requires the amd-aiter release below; it is not on PyPI, so install from git tag.
"""

import importlib.metadata
from typing import NoReturn, Optional

from primus_turbo.common.logger import logger

# Required aiter release. Keep in sync with AITER_VERSION in the ci / benchmark
# / release workflows.
AITER_VERSION = "0.1.14.post1"
AITER_GIT_TAG = "v0.1.14.post1"
_AITER_DIST_NAME = "amd-aiter"
_AITER_GIT_URL = "https://github.com/ROCm/aiter.git"

_AITER_PIP_INSTALL = f'pip install "amd-aiter @ git+{_AITER_GIT_URL}@{AITER_GIT_TAG}"'

AITER_INSTALL_HINT = (
    f"Primus-Turbo requires amd-aiter=={AITER_VERSION} for this operator. Install it with:\n"
    f"  {_AITER_PIP_INSTALL}"
)

_version_checked = False


def _installed_aiter_version():
    try:
        return importlib.metadata.version(_AITER_DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_order(installed: str, expected: str) -> Optional[int]:
    """-1, 0 or 1 for installed older than / equal to / newer than expected.

    None when the two cannot be ordered, either because a version does not parse or
    because ``packaging`` is absent, in which case only equality is decidable. Any
    local or dev suffix is ignored (e.g. an editable checkout's "+g1234567").
    """
    try:
        from packaging.version import InvalidVersion, Version

        try:
            a = Version(Version(installed).public)
            b = Version(Version(expected).public)
        except InvalidVersion:
            return None
    except ImportError:
        return 0 if installed.split("+")[0] == expected else None
    return (a > b) - (a < b)


def check_aiter_version_once():
    """Warn once if the installed aiter is not the pinned release.

    Only an *older* aiter is told to install the pin. Operators whose kernels landed
    after the pinned release need a newer one -- MXFP6's A6W6 entry points are the
    current example -- and handing that user the pin would be telling them to install
    an aiter their operator cannot run on.

    This never decides anything: whether a given operator's kernels are actually
    present is settled by probing for its symbols, because a version string cannot
    express "contains commit X" and a fork can carry any version it likes.
    """
    global _version_checked
    if _version_checked:
        return
    _version_checked = True

    installed = _installed_aiter_version()
    if installed is None:
        return
    order = _version_order(installed, AITER_VERSION)
    if order == 0:
        return
    if order is None or order < 0:
        logger.warning(
            "aiter version mismatch: installed=%s, expected=%s; behavior/perf may differ. "
            "To match, run:\n  %s",
            installed,
            AITER_VERSION,
            _AITER_PIP_INSTALL,
            once=True,
        )
    else:
        logger.info(
            "aiter %s is newer than the pinned %s; behavior/perf may differ from what CI covers.",
            installed,
            AITER_VERSION,
            once=True,
        )


def raise_aiter_missing(exc: Exception) -> NoReturn:
    logger.error(AITER_INSTALL_HINT, once=True)
    raise ImportError(AITER_INSTALL_HINT) from exc


_aiter_module = None


def get_aiter():
    """Import and return the ``aiter`` module, lazily and with a clear error."""
    global _aiter_module
    if _aiter_module is None:
        try:
            import aiter
        except ImportError as exc:
            raise_aiter_missing(exc)
        check_aiter_version_once()
        _aiter_module = aiter
    return _aiter_module
