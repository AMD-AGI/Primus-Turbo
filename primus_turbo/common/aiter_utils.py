###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helpers for the optional ``aiter`` dependency.

aiter is imported lazily (only when an AITER-backed op runs), so it stays out of
the package metadata while users still get a clear error when it is needed. The
pinned commit is usually newer than PyPI's, so it is installed from git.
"""

import importlib.metadata
import re
from typing import NoReturn

from primus_turbo.common.logger import logger

# Single source of truth for the aiter commit; CI installs this exact commit.
EXPECTED_AITER_COMMIT = "b5e03ed191fca11ee423226537ef8d9435e432a6"
_AITER_DIST_NAME = "amd-aiter"  # pip dist name; importable module is ``aiter``

AITER_INSTALL_HINT = (
    "Primus-Turbo requires 'aiter' for this operator, but it is not installed. "
    "Install the pinned commit (not available on PyPI):\n"
    f'  pip install "amd-aiter @ git+https://github.com/ROCm/aiter.git@{EXPECTED_AITER_COMMIT}"'
)

_version_checked = False


def _installed_aiter_commit():
    # aiter's setuptools_scm version embeds the commit, e.g. "0.1.1.dev1611+gf299f579a".
    try:
        version = importlib.metadata.version(_AITER_DIST_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None
    match = re.search(r"\+g([0-9a-fA-F]+)", version)
    return match.group(1) if match else None


def check_aiter_version_once():
    """Warn once if the installed aiter commit differs from the pin."""
    global _version_checked
    if _version_checked:
        return
    _version_checked = True

    installed = _installed_aiter_commit()
    if installed and not EXPECTED_AITER_COMMIT.lower().startswith(installed.lower()):
        logger.warning(
            "aiter commit mismatch: installed=%s, expected=%s; behavior/perf may differ.",
            installed,
            EXPECTED_AITER_COMMIT,
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
