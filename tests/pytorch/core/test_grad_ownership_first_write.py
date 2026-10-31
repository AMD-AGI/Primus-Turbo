###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""grad_ownership's first-write claims: one overwrite per slice and iteration, none before a consumer
rotates the log, and the slices they cover reported for the next iteration."""

import pytest
import torch

from primus_turbo.pytorch.core import grad_ownership


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """An empty log and no consumer rotating it; monkeypatch restores the module's own state."""
    for name, value in (
        ("_written", set()),
        ("_skipped", set()),
        ("_previous", frozenset()),
        ("_first_writes", set()),
        ("_previous_first_writes", frozenset()),
        ("_ever_recorded", False),
        ("_stepping", False),
    ):
        monkeypatch.setattr(grad_ownership, name, value)


def _key(t):
    return (t.data_ptr(), t.numel(), t.dtype)


def test_no_claim_without_a_consumer():
    g = torch.zeros(16, dtype=torch.bfloat16)
    assert not grad_ownership.claim_first_write(g)
    assert not grad_ownership.was_enabled()


def test_first_write_per_iteration():
    buf = torch.zeros(64, dtype=torch.bfloat16)
    w1, w2 = buf[:32], buf[32:]
    grad_ownership.begin_step()
    # micro-batch 1 overwrites, the later ones accumulate
    assert grad_ownership.claim_first_write(w1)
    assert grad_ownership.claim_first_write(w2)
    for _ in range(7):
        assert not grad_ownership.claim_first_write(w1)
        assert not grad_ownership.claim_first_write(w2)
    assert grad_ownership.was_enabled()
    previous = grad_ownership.begin_step()
    assert previous == {_key(w1), _key(w2)}
    assert grad_ownership.previous_first_writes() == {_key(w1), _key(w2)}
    # a new iteration starts with an overwrite again
    assert grad_ownership.claim_first_write(w1)


def test_record_overwrite_is_not_a_first_write_claim():
    a, b = torch.zeros(8, dtype=torch.bfloat16), torch.zeros(8, dtype=torch.bfloat16)
    grad_ownership.begin_step()
    grad_ownership.record_overwrite(a)
    assert grad_ownership.claim_first_write(b)
    assert not grad_ownership.claim_first_write(a)  # already overwritten this iteration
    grad_ownership.begin_step()
    assert grad_ownership.previous_first_writes() == {_key(b)}


def test_skipped_first_write_slices_validate():
    g = torch.zeros(8, dtype=torch.bfloat16)
    grad_ownership.begin_step()
    grad_ownership.note_skipped([_key(g)])
    with pytest.raises(RuntimeError, match="no overwrite"):
        grad_ownership.validate_overwritten([_key(g)])
    assert grad_ownership.claim_first_write(g)
    grad_ownership.validate_overwritten([_key(g)])
