###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An omitted gradient clear must be matched by a producer in the same step."""

import pytest
import torch

from primus_turbo.pytorch.core import grad_ownership as ownership


@pytest.fixture(autouse=True)
def isolated_log(monkeypatch):
    for name, value in [
        ("_written", set()),
        ("_skipped", set()),
        ("_previous", frozenset()),
        ("_ever_recorded", False),
    ]:
        monkeypatch.setattr(ownership, name, value)


def identity(x):
    return x.data_ptr(), x.numel(), x.dtype


def test_overwrite_is_valid_only_in_current_step():
    slab = torch.zeros(32)
    part = slab[8:24]
    key = identity(part)
    assert not ownership.was_enabled()
    ownership.record_overwrite(part)
    assert ownership.was_enabled()
    assert ownership.begin_step() == frozenset({key})
    ownership.note_skipped([key])
    with pytest.raises(RuntimeError, match="no overwrite arrived"):
        ownership.validate_overwritten([key])
    ownership.record_overwrite(part)
    ownership.validate_overwritten([key])
    assert ownership.begin_step() == frozenset({key})
    # An abandoned warmup step does not poison the following real step.
    ownership.note_skipped([key])
    assert ownership.begin_step() == frozenset()
    ownership.validate_overwritten([key])


@pytest.mark.parametrize("mismatch", ["offset", "length", "dtype"])
def test_overlapping_storage_is_not_the_same_gradient_slice(mismatch):
    slab = torch.zeros(64)
    part = slab[8:24]
    key = identity(part)
    ownership.note_skipped([key])
    other = {"offset": slab[9:25], "length": slab[8:25], "dtype": part.view(torch.int32)}[mismatch]
    ownership.record_overwrite(other)
    # Only the actual skipped slice is relevant to this communication bucket.
    ownership.validate_overwritten([identity(other)])
    with pytest.raises(RuntimeError, match="gradient-buffer slice"):
        ownership.validate_overwritten([key])
