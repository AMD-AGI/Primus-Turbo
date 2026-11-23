###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``PRIMUS_OVERLAP`` is the one switch of the side-stream overlap (``primus_turbo.pytorch.core.overlap``): it turns
every branch on or off together, and the former per-branch ``PRIMUS_OVERLAP_<branch>`` variables are not read."""

import importlib

import pytest

import primus_turbo.common.envs as envs
from primus_turbo.pytorch.core import overlap

_BRANCHES = ("WGRAD", "MOE_WGRAD", "SHARED", "INDEXER", "COMPRESSOR")


@pytest.mark.parametrize("value, want", [("1", 1), ("0", 0), (None, 0)], ids=["on", "off", "unset"])
def test_master_switch_sets_every_branch(monkeypatch, value, want):
    for b in _BRANCHES:  # set to the opposite value: must be ignored
        monkeypatch.setenv(f"PRIMUS_OVERLAP_{b}", "0" if want else "1")
    if value is None:
        monkeypatch.delenv("PRIMUS_OVERLAP", raising=False)
    else:
        monkeypatch.setenv("PRIMUS_OVERLAP", value)
    envs.reset_cache("PRIMUS_OVERLAP")
    mod = importlib.reload(overlap)
    try:
        assert set(mod._SWITCHES) == set(_BRANCHES)
        assert mod._FLAGS == {b: want for b in _BRANCHES}
        assert all(mod.enabled(b) == bool(want) for b in _BRANCHES)
    finally:
        monkeypatch.delenv("PRIMUS_OVERLAP", raising=False)
        for b in _BRANCHES:
            monkeypatch.delenv(f"PRIMUS_OVERLAP_{b}", raising=False)
        envs.reset_cache("PRIMUS_OVERLAP")
        importlib.reload(overlap)
