###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import pytest

from primus_turbo.pytorch.core.backend import TuneCache


def test_lru_access_and_update_protect_the_most_recent_entry():
    cache = TuneCache(capacity=2)
    first, second, replacement = object(), object(), object()
    cache.put("first", first)
    cache.put("second", second)
    assert cache.get("first") is first
    with pytest.warns(UserWarning, match="capacity.*exceeded"):
        cache.put("third", replacement)
    assert "second" not in cache and cache.get("second") is None
    assert cache.get("third") is replacement
    cache.put("first", replacement)
    with pytest.warns(UserWarning, match="capacity.*exceeded"):
        cache.put("fourth", second)
    assert "third" not in cache
    assert cache.get("first") is replacement
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0 and "first" not in cache
