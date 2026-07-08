###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helpers for the optional ``mori`` dependency.

mori is imported lazily (only when the MORI EP backend runs). Primus-Turbo
requires the amd_mori release below; it is not on PyPI, so install from git tag.
``BUILD_UMBP=OFF`` skips mori's gRPC-dependent umbp component (unused here).
"""

import importlib.metadata
from typing import NoReturn

from primus_turbo.common.logger import logger

# Required mori release. Keep in sync with MORI_VERSION in the ci / benchmark
# / release workflows.
MORI_VERSION = "1.2.0"
MORI_GIT_TAG = "v1.2.0"
_MORI_DIST_NAME = "amd_mori"
_MORI_GIT_URL = "https://github.com/ROCm/mori.git"

_MORI_PIP_INSTALL = f'BUILD_UMBP=OFF pip install "amd_mori @ git+{_MORI_GIT_URL}@{MORI_GIT_TAG}"'

MORI_INSTALL_HINT = (
    f"Primus-Turbo requires amd_mori=={MORI_VERSION} for the MORI EP backend. Install it with:\n"
    f"  {_MORI_PIP_INSTALL}"
)

_version_checked = False


def _installed_mori_version():
    try:
        return importlib.metadata.version(_MORI_DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None


def _versions_match(installed: str, expected: str) -> bool:
    # Ignore any local/dev suffix (e.g. a source build's "1.2.1.dev6+g1234567").
    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(installed).public == Version(expected).public
        except InvalidVersion:
            return False
    except ImportError:
        return installed.split("+")[0] == expected


def check_mori_version_once():
    """Warn once if the installed mori version differs from the pin."""
    global _version_checked
    if _version_checked:
        return
    _version_checked = True

    installed = _installed_mori_version()
    if installed and not _versions_match(installed, MORI_VERSION):
        logger.warning(
            "mori version mismatch: installed=%s, expected=%s; behavior/perf may differ. "
            "To match, run:\n  %s",
            installed,
            MORI_VERSION,
            _MORI_PIP_INSTALL,
            once=True,
        )


def raise_mori_missing(exc: Exception) -> NoReturn:
    logger.error(MORI_INSTALL_HINT, once=True)
    raise ImportError(MORI_INSTALL_HINT) from exc


_mori_module = None


def get_mori():
    """Import and return the ``mori`` module, lazily and with a clear error."""
    global _mori_module
    if _mori_module is None:
        try:
            import mori
        except ImportError as exc:
            raise_mori_missing(exc)
        check_mori_version_once()
        _mori_module = mori
    return _mori_module
