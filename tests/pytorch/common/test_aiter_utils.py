###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import importlib.metadata
from unittest.mock import Mock

import pytest

from primus_turbo.common import aiter_utils as impl


@pytest.mark.parametrize(
    "installed, expected",
    [
        (impl.AITER_VERSION, True),
        (impl.AITER_VERSION + "+local", True),
        ("0.0.1", False),
        ("invalid version", False),
    ],
)
def test_release_pin_allows_local_suffix(installed, expected):
    assert impl._versions_match(installed, impl.AITER_VERSION) == expected


def test_missing_distribution(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata, "version", Mock(side_effect=importlib.metadata.PackageNotFoundError)
    )
    assert impl._installed_aiter_version() is None


@pytest.mark.parametrize("installed, warnings", [(None, 0), (impl.AITER_VERSION, 0), ("0.0.1", 1)])
def test_version_warning_is_emitted_once(monkeypatch, installed, warnings):
    monkeypatch.setattr(impl, "_version_checked", False)
    monkeypatch.setattr(impl, "_installed_aiter_version", lambda: installed)
    logger = Mock()
    monkeypatch.setattr(impl, "logger", logger)
    impl.check_aiter_version_once()
    impl.check_aiter_version_once()
    assert logger.warning.call_count == warnings
    if warnings:
        assert impl.AITER_VERSION in logger.warning.call_args.args


def test_import_error_retains_cause_and_install_hint(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(impl, "logger", logger)
    cause = ModuleNotFoundError("aiter")
    with pytest.raises(ImportError, match="Primus-Turbo requires amd-aiter") as error:
        impl.raise_aiter_missing(cause)
    assert error.value.__cause__ is cause
    assert impl.AITER_GIT_TAG in str(error.value)
    logger.error.assert_called_once_with(impl.AITER_INSTALL_HINT, once=True)
