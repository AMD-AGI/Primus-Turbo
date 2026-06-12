###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helpers for the optional ``uccl`` dependency.

uccl is imported lazily (only when the UCCL EP backend runs). Primus-Turbo
requires the uccl release below; it is not on PyPI, so install from git tag.
"""

import importlib.metadata
from typing import NoReturn

from primus_turbo.common.logger import logger

# Required uccl release. uccl has no usable release tag, so pin a main commit.
# Keep in sync with UCCL_VERSION/UCCL_GIT_TAG in the ci / benchmark / release workflows.
UCCL_VERSION = "0.1.1"
UCCL_GIT_TAG = "34bdf4f49c8e33d618b983afc34fc5bec8686cfa"  # latest main commit
_UCCL_DIST_NAME = "uccl"
_UCCL_GIT_URL = "https://github.com/uccl-project/uccl.git"

_UCCL_PIP_INSTALL = f'pip install "uccl @ git+{_UCCL_GIT_URL}@{UCCL_GIT_TAG}"'

UCCL_INSTALL_HINT = (
    f"Primus-Turbo requires uccl=={UCCL_VERSION} for the UCCL EP backend. Install it with:\n"
    f"  {_UCCL_PIP_INSTALL}"
)

_version_checked = False


def _installed_uccl_version():
    try:
        return importlib.metadata.version(_UCCL_DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None


def _versions_match(installed: str, expected: str) -> bool:
    # Ignore any local/dev suffix (e.g. a source build's "+g1234567").
    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(installed).public == Version(expected).public
        except InvalidVersion:
            return False
    except ImportError:
        return installed.split("+")[0] == expected


def check_uccl_version_once():
    """Warn once if the installed uccl version differs from the pin."""
    global _version_checked
    if _version_checked:
        return
    _version_checked = True

    installed = _installed_uccl_version()
    if installed and not _versions_match(installed, UCCL_VERSION):
        logger.warning(
            "uccl version mismatch: installed=%s, expected=%s; behavior/perf may differ. "
            "To match, run:\n  %s",
            installed,
            UCCL_VERSION,
            _UCCL_PIP_INSTALL,
            once=True,
        )


def raise_uccl_missing(exc: Exception) -> NoReturn:
    logger.error(UCCL_INSTALL_HINT, once=True)
    raise ImportError(UCCL_INSTALL_HINT) from exc


_uccl_module = None


def get_uccl():
    """Import and return the ``uccl`` module, lazily and with a clear error."""
    global _uccl_module
    if _uccl_module is None:
        try:
            import uccl
        except ImportError as exc:
            raise_uccl_missing(exc)
        check_uccl_version_once()
        _uccl_module = uccl
    return _uccl_module
