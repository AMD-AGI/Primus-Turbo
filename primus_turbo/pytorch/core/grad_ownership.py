###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Which slices of the framework's gradient buffer a Turbo wgrad fully owns.

A wgrad epilogue that runs at beta=0 overwrites every element of the
``main_grad`` view it was handed, so the framework's per-step zeroing of that
slice is dead work. This module is the contract the framework side reads to
find those slices: the producer calls :func:`record_overwrite` from the branch
that actually issues the beta=0 write, and the consumer calls
:func:`begin_step` once per iteration to rotate the log.

Recording happens at the write, never at the forward. The beta=0 producer is
also selected in actual backward order, so activation recomputation and a
schedule that stages several forwards before backward cannot assign ownership
to a discarded or later-executing forward.

A slice skipped for iteration N is justified by iteration N-1 having
overwritten it, so :func:`begin_step` re-checks that prediction against what
iteration N actually wrote and raises if a skipped slice went unwritten.

Entries are ``(data_ptr, numel, dtype)`` rather than tensor or parameter references:
the consumer re-derives the same pair from its own buffer offsets and only
honours an exact match, so a stale or aliased entry cannot widen a claim, and
nothing here keeps a tensor alive.
"""

from typing import FrozenSet, Iterable, Set, Tuple

import torch

__all__ = ["record_overwrite", "note_skipped", "begin_step", "was_enabled"]

Slice = Tuple[int, int, torch.dtype]

_written: Set[Slice] = set()
_skipped: Set[Slice] = set()
_previous: FrozenSet[Slice] = frozenset()
_ever_recorded = False


def record_overwrite(main_grad: torch.Tensor) -> None:
    """Log that ``main_grad`` was fully overwritten by a beta=0 wgrad epilogue."""
    global _ever_recorded
    _ever_recorded = True
    _written.add((main_grad.data_ptr(), main_grad.numel(), main_grad.dtype))


def note_skipped(slices: Iterable[Slice]) -> None:
    """Log slices the consumer left unzeroed for the iteration now running."""
    _skipped.update(slices)


def begin_step() -> FrozenSet[Slice]:
    """Rotate the log at the top of an iteration and return the previous one.

    The returned set is what the iteration that just ended overwrote, which is
    what the framework may skip zeroing for the iteration about to start. An
    empty set means "zero everything" -- that is what makes iteration 0 and
    every multi-microbatch configuration correct without a special case, since
    neither takes any beta=0 write.
    """
    global _written, _skipped, _previous
    unwritten = _skipped - _written
    if unwritten:
        raise RuntimeError(
            f"{len(unwritten)} gradient-buffer slice(s) were left unzeroed on the "
            "prediction that a beta=0 wgrad would overwrite them, but no overwrite "
            f"arrived: {sorted(unwritten)[:4]}. The reduced gradient for those "
            "slices is stale. This means a wgrad producer stopped running or fell "
            "back to beta=1 mid-run."
        )
    _previous = frozenset(_written)
    _written = set()
    _skipped = set()
    return _previous


def was_enabled() -> bool:
    """Whether any beta=0 wgrad has ever run, i.e. whether this path is live."""
    return _ever_recorded
