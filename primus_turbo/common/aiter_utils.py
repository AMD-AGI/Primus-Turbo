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
from typing import NoReturn

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


def _versions_match(installed: str, expected: str) -> bool:
    # Ignore any local/dev suffix (e.g. an editable checkout's "+g1234567").
    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(installed).public == Version(expected).public
        except InvalidVersion:
            return False
    except ImportError:
        return installed.split("+")[0] == expected


def check_aiter_version_once():
    """Warn once if the installed aiter version differs from the pin."""
    global _version_checked
    if _version_checked:
        return
    _version_checked = True

    installed = _installed_aiter_version()
    if installed and not _versions_match(installed, AITER_VERSION):
        logger.warning(
            "aiter version mismatch: installed=%s, expected=%s; behavior/perf may differ. "
            "To match, run:\n  %s",
            installed,
            AITER_VERSION,
            _AITER_PIP_INSTALL,
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
