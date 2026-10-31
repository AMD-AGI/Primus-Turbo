###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Track gradient-buffer slices fully overwritten by a fused wgrad.

The beta=0 grouped-MXFP4 wgrad producer calls :func:`record_overwrite` only
after it has launched a full replacement of ``main_grad``.  Primus consumes
that exact ``(data_ptr, numel, dtype)`` identity on the following step to skip
the otherwise redundant clear of the same slice.

The consumer calls :func:`begin_step` once per iteration. Any slice it skips is
checked by :func:`validate_overwritten` before gradient communication, so a
missing producer fails instead of silently reducing stale gradients. A reset
that is abandoned without communication (for example, synthetic-warmup
cleanup) is safely discarded by the next :func:`begin_step`.

A producer that runs once per micro-batch (the Mega MoE wgrad) asks
:func:`claim_first_write` instead: it overwrites only on the slice's first
write of the iteration and accumulates on the others, so its slices
(:func:`previous_first_writes`) may stay uncleared whatever the number of
micro-batches. The beta=0 MXFP4 producer overwrites on every call, so its
claims hold only for single-micro-batch steps.
"""

from typing import FrozenSet, Iterable, Set, Tuple

import torch

__all__ = [
    "record_overwrite",
    "claim_first_write",
    "note_skipped",
    "validate_overwritten",
    "begin_step",
    "previous_first_writes",
    "was_enabled",
]

Slice = Tuple[int, int, torch.dtype]

_written: Set[Slice] = set()
_skipped: Set[Slice] = set()
_previous: FrozenSet[Slice] = frozenset()
_ever_recorded = False
# slices claimed by first-write producers this iteration / the last one
_first_writes: Set[Slice] = set()
_previous_first_writes: FrozenSet[Slice] = frozenset()
# a consumer rotates the log every iteration (claims mean nothing before that)
_stepping = False


def record_overwrite(main_grad: torch.Tensor) -> None:
    """Log a slice fully replaced by a beta=0 wgrad epilogue."""
    global _ever_recorded
    _ever_recorded = True
    _written.add((main_grad.data_ptr(), main_grad.numel(), main_grad.dtype))


def claim_first_write(main_grad: torch.Tensor) -> bool:
    """Whether a wgrad producer should overwrite ``main_grad`` (beta=0) instead of adding into it.

    True for the slice's first write of the iteration, which is then logged as an
    overwrite: the producer must replace the whole slice. False for later writes,
    and always False until a consumer calls :func:`begin_step` every iteration, so
    without one the producer keeps accumulating into the cleared buffer.
    """
    global _ever_recorded
    if not _stepping:
        return False
    key = (main_grad.data_ptr(), main_grad.numel(), main_grad.dtype)
    if key in _written:
        return False
    _ever_recorded = True
    _written.add(key)
    _first_writes.add(key)
    return True


def note_skipped(slices: Iterable[Slice]) -> None:
    """Log slices the consumer left uncleared in the current iteration."""
    _skipped.update(slices)


def _raise_if_unwritten(unwritten: Set[Slice]) -> None:
    if not unwritten:
        return
    preview = sorted(unwritten, key=lambda entry: (entry[0], entry[1], str(entry[2])))[:4]
    raise RuntimeError(
        f"{len(unwritten)} gradient-buffer slice(s) were left unzeroed on the "
        "prediction that a beta=0 wgrad would overwrite them, but no overwrite "
        f"arrived: {preview}. The reduced gradient for those slices is stale. "
        "This means a wgrad producer stopped running or fell back to beta=1 mid-run."
    )


def validate_overwritten(slices: Iterable[Slice]) -> None:
    """Fail before communication if a relevant skipped slice remains stale."""
    relevant = _skipped.intersection(slices)
    _raise_if_unwritten(relevant - _written)


def begin_step() -> FrozenSet[Slice]:
    """Rotate the producer log and return slices overwritten in the prior step."""
    global _written, _skipped, _previous, _first_writes, _previous_first_writes, _stepping
    _previous = frozenset(_written)
    _previous_first_writes = frozenset(_first_writes)
    _written = set()
    _skipped = set()
    _first_writes = set()
    _stepping = True
    return _previous


def previous_first_writes() -> FrozenSet[Slice]:
    """The prior step's slices overwritten by first-write producers (:func:`claim_first_write`)."""
    return _previous_first_writes


def was_enabled() -> bool:
    """Return whether a beta=0 producer has recorded an overwrite."""
    return _ever_recorded
