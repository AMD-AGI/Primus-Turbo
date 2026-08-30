###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The aiter version check, which advises and never decides.

Whether a given operator's kernels are present is settled by probing for its symbols --
see ``_check_aiter_a6w6`` for MXFP6 -- because no version string can express "contains
commit X". All this check does is tell the user how their aiter relates to the pin, so
what matters here is that the advice is right in both directions.

Nothing below touches a GPU or imports aiter.
"""

import pytest

from primus_turbo.common import aiter_utils


class _RecordingLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args, **kwargs):
        self.infos.append(msg % args if args else msg)


def _run_check(monkeypatch, installed):
    """Run the once-only check as if ``installed`` were the installed aiter."""
    recorder = _RecordingLogger()
    monkeypatch.setattr(aiter_utils, "logger", recorder)
    monkeypatch.setattr(aiter_utils, "_version_checked", False)
    monkeypatch.setattr(aiter_utils, "_installed_aiter_version", lambda: installed)
    aiter_utils.check_aiter_version_once()
    return recorder


@pytest.mark.parametrize(
    "installed,expected,order",
    [
        ("0.1.14.post1", "0.1.14.post1", 0),
        ("0.1.14.post1+g1234567", "0.1.14.post1", 0),  # editable checkout of the pin
        ("0.1.21.dev51+gb2acdf98c", "0.1.14.post1", 1),  # a source build past the pin
        ("0.1.9", "0.1.14.post1", -1),
        ("not-a-version", "0.1.14.post1", None),
    ],
)
def test_version_order(installed, expected, order):
    assert aiter_utils._version_order(installed, expected) == order


def test_a_newer_aiter_is_reported_without_advising_a_downgrade(monkeypatch):
    """The case MXFP6 puts everyone in, and the reason this check was reworked.

    A6W6 merged after the pinned release, so every MXFP6 user necessarily runs an aiter
    newer than the pin. Warning them with the install command would be handing them, on
    every run, an aiter their operator cannot run on.
    """
    recorder = _run_check(monkeypatch, "0.1.21.dev51+gb2acdf98c")
    assert recorder.warnings == []
    assert len(recorder.infos) == 1
    assert "newer" in recorder.infos[0]
    assert aiter_utils._AITER_PIP_INSTALL not in recorder.infos[0]


def test_an_older_aiter_is_told_how_to_reach_the_pin(monkeypatch):
    recorder = _run_check(monkeypatch, "0.1.9")
    assert len(recorder.warnings) == 1
    assert aiter_utils._AITER_PIP_INSTALL in recorder.warnings[0]


def test_an_unorderable_version_is_warned_about(monkeypatch):
    """Unable to compare is not the same as fine, so it takes the cautious branch."""
    recorder = _run_check(monkeypatch, "not-a-version")
    assert len(recorder.warnings) == 1
    assert aiter_utils._AITER_PIP_INSTALL in recorder.warnings[0]


@pytest.mark.parametrize("installed", ["0.1.14.post1", "0.1.14.post1+g1234567", None])
def test_the_pin_and_an_absent_aiter_say_nothing(monkeypatch, installed):
    """An absent aiter is not this check's business -- ``get_aiter`` reports that."""
    recorder = _run_check(monkeypatch, installed)
    assert recorder.warnings == [] and recorder.infos == []


def test_the_check_runs_at_most_once(monkeypatch):
    """It sits on an import path, so a repeat would warn once per operator call."""
    recorder = _run_check(monkeypatch, "0.1.9")
    aiter_utils.check_aiter_version_once()
    assert len(recorder.warnings) == 1
