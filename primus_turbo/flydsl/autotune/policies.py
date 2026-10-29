###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Probe / check / bench / select policies shared by the FlyDSL autotuners.

A tune runs on a list of :class:`Point` (one argument tuple per probe shape). Every
candidate is a callable ``launch(*point.args)``; a search whose candidates are only built,
never run on arguments, uses ``Point(())`` (see :class:`NoArgs`). Each policy implements one
of four roles, whose base classes carry the contract the engine relies on: :class:`Probe`,
:class:`Check` (or :class:`ReferenceCheck`), :class:`Bench` and :class:`Select`.

The engine checks, times the survivors, then selects; the stage's ``on_empty`` decides what
happens when none survives (see :class:`Stage`). All timing goes through :func:`_window` and
:func:`_settle`, which the CPU tests replace with a fake clock.
"""

import contextlib
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, ContextManager, List, NamedTuple, Optional, Sequence, Tuple

import torch

from primus_turbo.common.logger import logger
from primus_turbo.flydsl.autotune import runtime

__all__ = [
    "Point",
    "Probe",
    "Check",
    "ReferenceCheck",
    "Bench",
    "Select",
    "RealArgs",
    "Synthetic",
    "NoArgs",
    "Finite",
    "Reproducible",
    "MatchBase",
    "Interleaved",
    "Remember",
    "Fastest",
    "RAISE",
    "Stage",
    "robust_time",
    "geomean",
    "collect_faults",
    "note_fault",
]

INF = float("inf")

_faults = threading.local()


@contextlib.contextmanager
def collect_faults():
    """Every launch a bench scored ``inf`` because it raised, as ``(launch, exception)``, while held."""
    prev = getattr(_faults, "log", None)
    _faults.log = log = []
    try:
        yield log
    finally:
        _faults.log = prev


def note_fault(launch, ex: BaseException) -> None:
    """A bench's record of a launch that raised; kept by the innermost :func:`collect_faults`."""
    log = getattr(_faults, "log", None)
    if log is not None:
        log.append((launch, ex))


class Point(NamedTuple):
    """One probe: the launch arguments and the output the checks read (or None)."""

    args: tuple
    out: Optional[torch.Tensor] = None


# ── roles ───────────────────────────────────────────────────────────────────


class Probe(ABC):
    """Where a tune runs: :meth:`points` gives a context manager that yields the points, or None
    when it cannot probe (the tune then takes the base); the points are used inside the ``with``
    only.

    ``release``: the caching allocator is emptied once the points are dead (see
    :class:`Synthetic`).
    """

    release: bool = False

    @abstractmethod
    def points(self) -> ContextManager[Optional[List[Point]]]: ...


class Check(ABC):
    """Whether a built candidate may race. The engine calls :meth:`check` on each one before
    timing, with ``ref=None``; a candidate that fails it or raises leaves the race."""

    @abstractmethod
    def check(self, launch, points: Sequence[Point], ref=None) -> bool: ...


class ReferenceCheck(Check):
    """A :class:`Check` against the stage's first candidate (the base or the previous pick).

    The engine first takes :meth:`reference` on that candidate, passes it to every ``check`` as
    ``ref`` and does not check that candidate itself. When it did not build or gives None, the
    stage keeps it, unverified, and times nothing. A reference stage has no base, so it skips
    such a check and building has to suffice.
    """

    @abstractmethod
    def reference(self, launch, points: Sequence[Point]):
        """What ``check`` compares against, or None when the base is not usable."""


class Bench(ABC):
    """How the candidates are timed: ``times(launches, points)`` is a ``[candidate][point]``
    matrix in ms, one row per launch in order. A launch that raises scores ``inf`` and is kept
    with :func:`note_fault` rather than raised. A wrapper (:class:`Remember`, an op's own) is a
    bench around another one."""

    @abstractmethod
    def times(self, launches: Sequence[Callable], points: Sequence[Point]) -> List[List[float]]: ...


class Select(ABC):
    """Which candidate wins: ``pick(t, configs) -> (index or None, scores)`` from a bench's matrix
    ``t``, one row per entry of ``configs``, with no device work; None when no candidate scored,
    ``scores`` (one per row) go to the report. Row 0 is the stage's first surviving candidate: the
    base or the previous pick, unless that one did not build or failed a check it is held to, in
    which case it is the first challenger that passed."""

    @abstractmethod
    def pick(self, t: List[List[float]], configs: Sequence) -> Tuple[Optional[int], List[float]]: ...


# ── probes ──────────────────────────────────────────────────────────────────


class RealArgs(Probe):
    """Tune on the caller's own arguments (already redirected to a scratch output if needed)."""

    def __init__(self, *points: Point):
        self._points = list(points)

    @contextlib.contextmanager
    def points(self):
        try:
            yield self._points
        finally:
            self._points = None


class NoArgs(Probe):
    """The one empty point of a search whose candidates are only built, never run on arguments
    (e.g. a single ``Reference`` step on a heuristic's config)."""

    @contextlib.contextmanager
    def points(self):
        yield [Point(())]


class Synthetic(Probe):
    """Tune on probe tensors built by ``make() -> list[Point] | None`` (None = cannot probe).

    ``release``: the caching allocator is emptied once the probes are dead, which is when the
    ``Tuner.entry`` the pick builds for returns (``engine.request_release``): gigabyte probes left
    in it were measured to make a later race in the same process mis-pick. A call site keeps its
    probe tensors in the points or in locals of that entry's ``make``, nowhere longer-lived.
    """

    release: bool = True

    def __init__(self, make: Callable[[], Optional[List[Point]]]):
        self._make = make

    @contextlib.contextmanager
    def points(self):
        yield self._make()


# ── checks ──────────────────────────────────────────────────────────────────


# Elements per slice of a whole-output comparison: its fp32 temporaries stay at this size rather
# than a few copies of the whole output.
_SLICE = 1 << 24


def _slices(t: torch.Tensor):
    flat = t.reshape(-1)
    for i in range(0, flat.numel(), _SLICE):
        yield flat[i : i + _SLICE]


def _finite(t: torch.Tensor, n: Optional[int]) -> bool:
    flat = t.reshape(-1)
    if n is not None:
        flat = flat[:n]
    return all(bool(torch.isfinite(s.float()).all().item()) for s in _slices(flat))


class Finite(Check):
    """Keep a candidate whose output sample is finite. The base gets no special role."""

    def __init__(self, n: Optional[int] = 1024):
        self.n = n

    def check(self, launch, points: Sequence[Point], ref=None) -> bool:
        for p in points:
            launch(*p.args)
            torch.cuda.synchronize()
            if p.out is not None and not _finite(p.out, self.n):
                return False
        return True


# (device, megabytes) -> the cache flush every Reproducible check on that device writes, or None
# where it could not be allocated. Kept for the process, until runtime.clear().
_FLUSH: dict = {}
runtime.register(clear_hook=_FLUSH.clear)


def _alloc_flush(device, mb):
    return torch.empty(mb << 20, dtype=torch.uint8, device=device)


def _flush_buffer(device, mb):
    """One flush buffer per device for the process: allocated per search, it competes with the
    probe's own scratch on the largest shapes, and losing it there would fail every candidate."""
    key = (str(device), mb)
    if key not in _FLUSH:
        try:
            _FLUSH[key] = _alloc_flush(device, mb)
        except torch.OutOfMemoryError:
            logger.warning(
                f"[flydsl-autotune] no room for a {mb} MB cache flush on {device}: the "
                "reproducibility check repeats warm there, which misses a race that needs a cold cache",
                once=True,
            )
            _FLUSH[key] = None
    return _FLUSH[key]


class Reproducible(Finite):
    """:class:`Finite`, then the same bytes, compared exactly, on each of ``reps`` repeats on one
    set of operands: a candidate whose output moves between calls is not a candidate at any
    speed, and a race by time alone ranks such a wrong answer first. The defects this catches do
    not raise and keep the whole-output SNR high; single tiles fall, in different places each
    call, so repetition is the instrument (#570).

    The repeats run cold: before each one a ``flush_mb`` buffer is written over, since back to
    back the operands stay resident and the window a racing schedule needs never opens -- a warm
    check passed configs that then differed on 31 of 32 real calls (#570). The buffer is one per
    device for the process (see :func:`_flush_buffer`); where it cannot be allocated the repeats
    run warm, with a warning, rather than failing every candidate on an out-of-memory error.

    Probabilistic, not a proof: a candidate that differs on one call in sixteen passes 24 repeats
    about a fifth of the time and 64 about 1.6% of it. The repeats cost first-call tune time only.
    """

    def __init__(self, reps: int = 64, flush_mb: int = 512, n: Optional[int] = 1024):
        super().__init__(n)
        self.reps, self.flush_mb = reps, flush_mb

    def check(self, launch, points: Sequence[Point], ref=None) -> bool:
        if not super().check(launch, points):
            return False
        for p in points:
            if p.out is None:
                continue
            launch(*p.args)
            torch.cuda.synchronize()
            try:
                first = p.out.clone()
            except torch.OutOfMemoryError:
                logger.warning(
                    f"[flydsl-autotune] no room to copy a {tuple(p.out.shape)} output: the "
                    "reproducibility check is skipped there and the candidate kept on its finite check",
                    once=True,
                )
                continue
            flush = _flush_buffer(p.out.device, self.flush_mb)
            for _ in range(self.reps):
                if flush is not None:
                    flush.zero_()
                launch(*p.args)
                torch.cuda.synchronize()
                if not torch.equal(first, p.out):
                    return False
        return True


class MatchBase(ReferenceCheck):
    """Keep a candidate whose output matches the base's within ``rel_mse`` at every point.

    The base runs first and must itself be finite, or the whole tune falls back to base. The
    output is zeroed before every run, so a candidate that skips tiles cannot inherit the
    previous run's values there, and an accumulating (beta=1) build still sees ``C = 0 + acc``.
    ``n``: how many leading output elements the finite checks read (None = all of them).
    """

    def __init__(self, rel_mse: float = 2e-2**2, n: Optional[int] = 1024):
        self.rel_mse = rel_mse
        self.n = n

    def reference(self, launch, points: Sequence[Point]):
        refs = []
        for p in points:
            p.out.zero_()
            launch(*p.args)
            torch.cuda.synchronize()
            r = p.out.detach().to(torch.float32, copy=True)
            if not _finite(r, self.n):
                return None
            rn = sum(float(torch.dot(s, s)) for s in _slices(r))
            refs.append((r, rn or 1.0))
        return refs

    def check(self, launch, points: Sequence[Point], ref=None) -> bool:
        for p, (r, rn) in zip(points, ref):
            p.out.zero_()
            launch(*p.args)
            torch.cuda.synchronize()
            o = p.out.detach()
            err = 0.0
            for os_, rs in zip(_slices(o), _slices(r)):
                d = os_.float() - rs
                err += float(torch.dot(d, d))
            if not (err / rn < self.rel_mse and _finite(o, self.n)):
                return False
        return True


# ── timing primitives ───────────────────────────────────────────────────────


def _settle(launch, args, n):
    """``n`` untimed launches, then a device sync."""
    for _ in range(n):
        launch(*args)
    torch.cuda.synchronize()


def _window(launch, args, iters):
    """One timed window of ``iters`` back-to-back launches; ms per launch."""
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        launch(*args)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def robust_time(launch, args, warmup=250, reps=5, iters=50):
    """Median-of-``reps`` per-launch time after ``warmup`` launches. The long warmup reaches boost
    clock; short-K kernels mis-pick configs otherwise."""
    _settle(launch, args, warmup)
    ts = sorted(_window(launch, args, iters) for _ in range(reps))
    return ts[len(ts) // 2]


# ── bench ───────────────────────────────────────────────────────────────────


class Interleaved(Bench):
    """Every candidate timed inside one shared set of windows, each kept at its fastest.

    The candidates sit closer together than the window-to-window drift, so a private window per
    candidate lets the drift do the ranking. Per point:

    * ``ramp`` launches of the first candidate only (the leading slot would otherwise pay the
      clock ramp), then ``warmup`` launches of every candidate -- ``warmup_rounds=True``
      interleaves them round by round, else each candidate runs its own ``warmup`` back to back;
    * ``passes`` rounds over the candidates, the order reversed on the way back when
      ``palindrome`` (the ramp term is odd in the slot index, so it cancels); each visit is
      ``visit_warmup`` launches plus the median of ``visit_reps`` windows of ``iters`` launches.
    """

    def __init__(
        self,
        passes,
        iters,
        *,
        warmup=0,
        warmup_rounds=True,
        ramp=0,
        palindrome=True,
        visit_warmup=0,
        visit_reps=1,
    ):
        self.passes, self.iters = passes, iters
        self.warmup, self.warmup_rounds, self.ramp = warmup, warmup_rounds, ramp
        self.palindrome, self.visit_warmup, self.visit_reps = palindrome, visit_warmup, visit_reps

    def _visit(self, launch, args):
        if self.visit_reps == 1 and self.visit_warmup == 0:
            return _window(launch, args, self.iters)
        return robust_time(launch, args, self.visit_warmup, self.visit_reps, self.iters)

    def times(self, launches, points):
        cols = [self._at(launches, p.args) for p in points]
        return [list(row) for row in zip(*cols)] if cols else [[] for _ in launches]

    def _at(self, launches, args):
        n = len(launches)
        best = [INF] * n
        dead = set()

        def run(i, fn, *a):
            if i in dead:
                return None
            try:
                return fn(*a)
            except Exception as ex:  # noqa: BLE001 - a launch that faults loses the race
                note_fault(launches[i], ex)
                dead.add(i)
                return None

        if self.ramp and n:
            run(0, _settle, launches[0], args, self.ramp)
        if self.warmup_rounds:
            for _ in range(self.warmup):
                for i in range(n):
                    run(i, launches[i], *args)
            torch.cuda.synchronize()
        else:
            for i in range(n):
                run(i, _settle, launches[i], args, self.warmup)
        order = list(range(n))
        if self.palindrome:
            order = order + order[::-1]
        for _ in range(self.passes):
            for i in order:
                t = run(i, self._visit, launches[i], args)
                if t is not None and t < best[i]:
                    best[i] = t
        return [INF if i in dead else t for i, t in enumerate(best)]


class Remember(Bench):
    """``inner``, but a launch already timed through this object keeps that time: a stage that
    leads with the previous stage's pick compares against the time that pick won with rather
    than a fresh sample. Build one per search -- it holds the launches it timed."""

    def __init__(self, inner: Bench):
        self.inner = inner
        self._seen = {}

    def times(self, launches, points):
        todo = [l for l in launches if id(l) not in self._seen]
        if todo:
            for l, row in zip(todo, self.inner.times(todo, points)):
                self._seen[id(l)] = (l, row)
        return [list(self._seen[id(l)][1]) for l in launches]


# ── selection ───────────────────────────────────────────────────────────────


def geomean(xs):
    if any(x == INF for x in xs):
        return INF
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _argmin(scores):
    best = min(range(len(scores)), key=scores.__getitem__)
    return None if scores[best] == INF else best


def _worst_ratio(row, ref):
    return INF if any(x == INF for x in row) else max(x / r for x, r in zip(row, ref))


class Fastest(Select):
    """Lowest score over the points; ties go to the earlier candidate.

    ``over``: ``"geomean"`` scores the geomean time; ``"worst"`` scores the worst point's time
    relative to the first candidate that timed at every point, so a challenger has to win at every
    point rather than on average. ``margin`` < 1 keeps row 0 (the stage's first surviving
    candidate, see :class:`Select`) unless another beats it by that factor, so a noise-tie
    never displaces it; the challengers race each other without it. ``charge(config) -> factor``
    inflates a speculative candidate's time, so it has to win by more than the noise.
    """

    def __init__(self, margin: float = 1.0, over: str = "geomean", charge: Optional[Callable] = None):
        assert 0.0 < margin <= 1.0 and over in ("geomean", "worst")
        self.margin, self.over, self.charge = margin, over, charge

    def pick(self, t, configs):
        """``t``: a bench's matrix, one row per entry of ``configs``."""
        ref = None
        if self.over == "worst":
            ref = next((row for row in t if row and all(x != INF for x in row)), None)
        scores = []
        for i, (cfg, row) in enumerate(zip(configs, t)):
            s = geomean(row) if ref is None else _worst_ratio(row, ref)
            if s != INF:
                if self.charge is not None:
                    s *= self.charge(cfg)
                if i > 0:
                    s /= self.margin
            scores.append(s)
        return _argmin(scores), scores


RAISE = object()


@dataclass(frozen=True)
class Stage:
    """One step of a staged search.

    ``configs(prev_best) -> list``: the candidates, the previous pick first (prev_best is None in
    the first stage). ``bench=None`` is a reference search: no timing and no ``select``, the first
    candidate that builds and passes ``check`` is the pick, and ``fallback`` is taken when none
    does (:data:`RAISE`: raise instead). A timed stage none of whose candidates survives raises
    when ``on_empty`` is ``"raise"``, and with ``"base"`` keeps what it started from.
    """

    configs: Callable
    bench: Optional[Bench]
    select: Optional[Select] = None
    check: Optional[Check] = None
    fallback: object = RAISE
    on_empty: str = "raise"

    def __post_init__(self):
        for name, role in (("bench", Bench), ("select", Select), ("check", Check)):
            v = getattr(self, name)
            if v is not None and not isinstance(v, role):
                raise TypeError(f"Stage.{name} must be a {role.__name__} or None, not {type(v).__name__}")
        if self.on_empty not in ("raise", "base"):
            raise ValueError(f'Stage.on_empty must be "raise" or "base", not {self.on_empty!r}')
