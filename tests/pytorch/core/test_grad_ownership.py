###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for ``primus_turbo.pytorch.core.grad_ownership``.

This module is the single source of truth for which ``main_grad`` slices a
beta=0 wgrad epilogue fully overwrote during the iteration that just ran, and
it is the only safety net that catches a producer silently falling back to
beta=1 (or stopping) after a framework decided -- based on the *previous*
iteration's behaviour -- that it could skip zeroing a slice. None of the
existing GEMM-level tests exercise this state machine directly (they all
construct their weight as something other than a real ``torch.nn.Parameter``,
which makes the beta=0 path structurally unreachable), so this file tests it
in isolation with plain CPU tensors -- no GPU/HIP build is required.
"""

import pytest
import torch

from primus_turbo.pytorch.core import grad_ownership


@pytest.fixture(autouse=True)
def _clean_grad_ownership_state():
    """``grad_ownership`` is a module-level singleton log. Reset it around every
    test so tests cannot leak ``_written`` / ``_skipped`` / ``_previous`` /
    ``_ever_recorded`` state into each other."""
    grad_ownership._written = set()
    grad_ownership._skipped = set()
    grad_ownership._previous = frozenset()
    grad_ownership._ever_recorded = False
    yield
    grad_ownership._written = set()
    grad_ownership._skipped = set()
    grad_ownership._previous = frozenset()
    grad_ownership._ever_recorded = False


def _slice_of(tensor: torch.Tensor):
    return (tensor.data_ptr(), tensor.numel(), tensor.dtype)


class TestBeginStepRotation:
    def test_first_call_returns_empty_and_does_not_raise(self):
        """Nothing has ever been written or skipped: begin_step() must report
        "zero everything" (empty set), which is what makes iteration 0 correct
        without a special case."""
        assert grad_ownership.begin_step() == frozenset()

    def test_written_slices_become_the_returned_owned_set(self):
        t1 = torch.empty(8)
        t2 = torch.empty(4)
        grad_ownership.record_overwrite(t1)
        grad_ownership.record_overwrite(t2)

        owned = grad_ownership.begin_step()

        assert owned == frozenset({_slice_of(t1), _slice_of(t2)})

    def test_written_log_is_cleared_after_rotation(self):
        t1 = torch.empty(8)
        grad_ownership.record_overwrite(t1)
        grad_ownership.begin_step()  # rotates t1's write into "previous"

        # Nothing new was written since the rotation: the next call must not
        # keep re-reporting a stale claim.
        owned = grad_ownership.begin_step()
        assert owned == frozenset()

    def test_only_the_most_recently_completed_iterations_writes_are_returned(self):
        t1 = torch.empty(8)
        t2 = torch.empty(4)
        grad_ownership.record_overwrite(t1)
        grad_ownership.begin_step()  # iteration 1's write rotates in as "previous"

        grad_ownership.record_overwrite(t2)
        owned = grad_ownership.begin_step()  # iteration 2's write rotates in

        assert owned == frozenset({_slice_of(t2)})

    def test_record_overwrite_keys_by_address_numel_dtype_not_object_identity(self):
        """Two different Python tensor objects describing the same underlying
        memory region and element count must count as the same slice -- this is
        what lets the consumer re-derive its own claim purely from buffer
        offsets, per the module's docstring."""
        t = torch.empty(8)
        written_view = t[:]
        rederived_view = t.view(8)
        assert written_view.data_ptr() == rederived_view.data_ptr()

        grad_ownership.record_overwrite(written_view)
        owned = grad_ownership.begin_step()

        assert _slice_of(rederived_view) in owned

    def test_dtype_prevents_same_address_and_numel_from_colliding(self):
        tensor = torch.empty(8, dtype=torch.float32)
        typed_alias = tensor.view(torch.int32)
        assert tensor.data_ptr() == typed_alias.data_ptr()
        assert tensor.numel() == typed_alias.numel()

        grad_ownership.record_overwrite(tensor)
        owned = grad_ownership.begin_step()

        assert _slice_of(typed_alias) not in owned


class TestBeginStepSafetyNet:
    def test_raises_when_a_skipped_slice_was_never_rewritten(self):
        """A slice the framework predicted would get a beta=0 write (and so
        skipped zeroing) must actually receive one. If not, the reduced
        gradient for that slice is stale -- this is the mechanical detector
        for a wgrad producer that stopped running or fell back to beta=1."""
        phantom_slice = (0x1000, 16, torch.float32)
        grad_ownership.note_skipped([phantom_slice])

        with pytest.raises(RuntimeError, match="left unzeroed"):
            grad_ownership.begin_step()

    def test_does_not_raise_when_the_skipped_slice_was_rewritten(self):
        t = torch.empty(8)
        grad_ownership.note_skipped([_slice_of(t)])
        grad_ownership.record_overwrite(t)

        owned = grad_ownership.begin_step()

        assert owned == frozenset({_slice_of(t)})

    def test_raises_unless_every_predicted_slice_was_rewritten(self):
        """Two slices were predicted; only one actually got rewritten. This
        must still raise -- a partial match is not good enough."""
        t1 = torch.empty(8)
        t2 = torch.empty(4)
        grad_ownership.note_skipped([_slice_of(t1), _slice_of(t2)])
        grad_ownership.record_overwrite(t1)

        with pytest.raises(RuntimeError):
            grad_ownership.begin_step()

    def test_unrelated_written_slices_do_not_satisfy_a_different_prediction(self):
        """Writing some unrelated slice must not be mistaken for satisfying a
        prediction about a completely different slice."""
        predicted = torch.empty(8)
        unrelated = torch.empty(8)
        grad_ownership.note_skipped([_slice_of(predicted)])
        grad_ownership.record_overwrite(unrelated)

        with pytest.raises(RuntimeError, match="left unzeroed"):
            grad_ownership.begin_step()


class TestWasEnabled:
    def test_false_until_the_first_overwrite_ever_recorded(self):
        assert grad_ownership.was_enabled() is False
        grad_ownership.record_overwrite(torch.empty(4))
        assert grad_ownership.was_enabled() is True

    def test_stays_true_across_rotations_even_with_nothing_new_written(self):
        """``was_enabled`` answers "has this path ever been live", not "was it
        used this iteration" -- ``begin_step`` must never clear it."""
        grad_ownership.record_overwrite(torch.empty(4))
        grad_ownership.begin_step()
        grad_ownership.begin_step()

        assert grad_ownership.was_enabled() is True


class TestSteadyStateIterationCycle:
    def test_three_iteration_cycle_matches_the_documented_owner_rotation(self):
        """Mirrors steady-state single-microbatch training: the wgrad
        producer's beta=0 write on iteration N becomes the owned
        (skip-zeroing) set consumed at the *start* of iteration N+1, and a
        repeat of the same write pattern on iteration N+1 validates cleanly at
        the start of iteration N+2 -- exactly the cycle the module's docstring
        describes."""
        w1 = torch.empty(4, 4)
        w2 = torch.empty(4, 4)

        # Iteration 1 starts: nothing is owned yet, so the framework's own
        # reset would fall back to a full zero_().
        assert grad_ownership.begin_step() == frozenset()

        # Iteration 1's backward: both expert weights get a beta=0 first write.
        grad_ownership.record_overwrite(w1)
        grad_ownership.record_overwrite(w2)

        # Iteration 2 starts: the framework may skip zeroing exactly those two
        # slices, and (per the real consumer's contract) records that it did.
        owned = grad_ownership.begin_step()
        assert owned == frozenset({_slice_of(w1), _slice_of(w2)})
        grad_ownership.note_skipped(owned)

        # Iteration 2's backward: steady-state routing repeats, so both
        # weights get a beta=0 write again.
        grad_ownership.record_overwrite(w1)
        grad_ownership.record_overwrite(w2)

        # Iteration 3 starts: no RuntimeError, because both predicted slices
        # were in fact rewritten during iteration 2.
        owned_again = grad_ownership.begin_step()
        assert owned_again == frozenset({_slice_of(w1), _slice_of(w2)})

    def test_a_producer_that_stops_writing_is_caught_one_iteration_later(self):
        """If a param that owned a slice on iteration N unexpectedly does not
        get a beta=0 write on iteration N+1 (e.g. it fell back to beta=1),
        the framework's skip-zeroing decision for iteration N+1 was wrong.
        ``begin_step`` at the start of iteration N+2 must catch it."""
        w = torch.empty(4, 4)

        grad_ownership.begin_step()
        grad_ownership.record_overwrite(w)  # iteration 1: beta=0 write

        owned = grad_ownership.begin_step()  # iteration 2 starts
        assert owned == frozenset({_slice_of(w)})
        grad_ownership.note_skipped(owned)  # iteration 2 skips zeroing `w`'s slice

        # Iteration 2's backward falls back to beta=1 (or the producer stops
        # running): no record_overwrite call for `w` this time.

        with pytest.raises(RuntimeError, match="left unzeroed"):
            grad_ownership.begin_step()  # iteration 3 starts: caught
