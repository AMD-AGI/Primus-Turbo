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

"""The selection engine every FlyDSL autotune front end runs on.

* :class:`Space` -- the candidates: a flat list or a staged search, each with the policies that
  race them and the base used untuned.
* :class:`Tuner` -- one op's picks: switch, capture guard, search, report and an in-process
  cache. ``pick(key, space, probe=..., build=...)`` returns a :class:`Pick`.
* :class:`CompiledLaunch` -- a launch run through its compile, with the one eager/capture split;
  :class:`Entry` is one an op runs for a pick.
"""

import contextlib
import threading
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence

from primus_turbo.common.logger import logger
from primus_turbo.flydsl.autotune import runtime
from primus_turbo.flydsl.autotune.policies import (
    RAISE,
    Bench,
    Check,
    Probe,
    ReferenceCheck,
    Select,
    Stage,
    collect_faults,
)

__all__ = ["Space", "Pick", "Tuner", "CompiledLaunch", "Entry", "NoWorkingConfig"]


class Pick(NamedTuple):
    """``config``: the pick. ``launch``: its built launch when this call built it, else None.
    ``tuned``: True for a search's pick. False for an untimed one while the switch is off (base,
    or ``off="first_valid"``'s pick): final until tuning can run. None for base taken under
    capture/compile, or for a fallback taken because every candidate that built failed on this
    call's probe: provisional, picked again at the next call outside capture whatever the switch.
    An op caches its artifact with the same flag, and :meth:`Tuner.stale` reads it back.
    ``built``: every launch the search built, for callers whose artifact caches key on launch
    identity."""

    config: Any
    launch: Optional[Callable] = None
    tuned: Optional[bool] = True
    built: tuple = ()


class NoWorkingConfig(RuntimeError):
    """No candidate builds and holds for a key. ``data_dependent``: some candidate built but none
    held on this call's probe (e.g. non-finite inputs), so the next call searches again; otherwise
    nothing built and it is remembered until ``runtime.clear()``."""

    def __init__(self, msg, data_dependent=False):
        super().__init__(msg)
        self.data_dependent = data_dependent


def _no_working(op, cfgs, data_dependent):
    return NoWorkingConfig(f"[flydsl-autotune] {op}: no working config among {cfgs}", data_dependent)


_UNSET = object()


class Space:
    """What a :class:`Tuner` searches: ``stages`` (see :class:`Stage`) and the untuned ``base``."""

    def __init__(self, stages: Sequence[Stage], base=_UNSET):
        self.stages = list(stages)
        self._base = base

    @property
    def base(self):
        if self._base is _UNSET:
            self._base = list(self.stages[0].configs(None))[0]
        return self._base

    @classmethod
    def of(
        cls,
        configs,
        *,
        bench: Optional[Bench],
        select: Optional[Select],
        check: Optional[Check] = None,
        on_empty: str = "raise",
        base=_UNSET,
    ) -> "Space":
        """One race over ``configs`` -- a list (``configs[0]`` = base unless ``base`` is given), or a
        zero-argument callable evaluated only when a search runs (``base`` then required)."""
        if callable(configs):
            if base is _UNSET:
                raise ValueError("a lazy Space needs an explicit base")
            fn = configs
        else:
            cands = list(configs)
            fn = lambda _c=cands: _c  # noqa: E731
            if base is _UNSET:
                base = cands[0]
        return cls([Stage(lambda _prev: fn(), bench, select, check, on_empty=on_empty)], base)

    @classmethod
    def staged(cls, *stages: Stage, base=_UNSET) -> "Space":
        return cls(stages, base)


def _failed(op, errors, key, cfg, what, ex):
    """A candidate that raised while checked or timed: it leaves the race, and the reason is kept
    in ``errors`` (a build failure is kept there as the bare ``repr``)."""
    if errors is not None:
        errors[(key, cfg)] = f"{what}: {ex!r}"
    if what == "bench":
        logger.warning(f"[flydsl-autotune] {op}: {cfg} raised while timed, scored inf: {ex!r}", once=True)


def _reference(op, st, cfgs, points, build, errors, key):
    """A stage without a bench: build and check ``cfgs`` in order, the first that holds wins, else
    the stage's ``fallback``. Returns (config, launch, built); a fallback taken although a
    candidate built (it then failed its check) is ``launch=None`` with ``built`` non-empty."""
    built = []
    for cfg in cfgs:
        try:
            launch = build(cfg)
        except Exception as ex:  # noqa: BLE001
            if errors is not None:
                errors[(key, cfg)] = repr(ex)
            launch = None
        if launch is not None:
            built.append(launch)
        good = launch is not None
        # A base-relative check has no base to hold the reference here: building has to suffice.
        if good and st.check is not None and not isinstance(st.check, ReferenceCheck):
            try:
                good = bool(st.check.check(launch, points, None))
            except Exception as ex:  # noqa: BLE001
                _failed(op, errors, key, cfg, "check", ex)
                good = False
        if good:
            return cfg, launch, built
    if st.fallback is RAISE:
        raise _no_working(op, cfgs, bool(built))
    return st.fallback, None, built


def _empty(op, on_empty, cfgs, si, base, launches):
    """A stage where nothing survived: raise, or keep what it started from. Data-dependent when
    a candidate built (it then failed its check or its timing on this probe)."""
    data_dependent = any(l is not None for l in launches)
    if on_empty == "raise":
        raise _no_working(op, cfgs, data_dependent)
    return cfgs[0] if si > 0 else base, data_dependent


def _search(op, stages, points, build, base, *, errors=None, key=None):
    """Run ``stages`` on ``points``; returns (config, launch, scores, n_candidates, built,
    unverified). ``unverified``: a stage fell back although a candidate of it built, so the pick
    reflects this probe's data rather than the candidates and must not be cached."""
    best, best_launch = base, None
    scores_all, n_cands, built, unverified = [], 0, [], False
    for si, st in enumerate(stages):
        cfgs = list(st.configs(None if si == 0 else best))
        if si > 0 and cfgs and cfgs[0] != best:
            cfgs = [best] + [c for c in cfgs if c != best]
        if not cfgs:
            continue
        if len(cfgs) == 1 and si == 0 and st.check is None:
            best = cfgs[0]
            continue
        if len(cfgs) == 1 and si > 0:
            continue
        n_cands += len(cfgs)
        if st.bench is None:
            best, best_launch, b = _reference(op, st, cfgs, points, build, errors, key)
            built.extend(b)
            unverified |= best_launch is None and bool(b)
            continue
        launches = []
        for cfg in cfgs:
            if si > 0 and cfg == best and best_launch is not None:
                launches.append(best_launch)
                continue
            try:
                launches.append(build(cfg))
            except Exception as ex:  # noqa: BLE001 - a config that does not build just leaves the race
                if errors is not None:
                    errors[(key, cfg)] = repr(ex)
                launches.append(None)
        built.extend(l for l in launches if l is not None)

        c = st.check
        by_ref = isinstance(c, ReferenceCheck)
        ref = None
        if by_ref:
            if launches[0] is not None:
                try:
                    ref = c.reference(launches[0], points)
                except Exception as ex:  # noqa: BLE001
                    _failed(op, errors, key, cfgs[0], "check", ex)
                    ref = None
                if ref is None:
                    # The base built but gave no usable reference (e.g. non-finite on this probe):
                    # nothing can be compared, so the stage keeps it, unverified.
                    if errors is not None:
                        errors.setdefault((key, cfgs[0]), "check: no usable reference")
                    unverified = True
            if ref is None:
                best = cfgs[0] if si > 0 else base
                continue
        alive = []
        for i, launch in enumerate(launches):
            if launch is None:
                alive.append(False)
                continue
            good = True
            if c is not None and not (by_ref and i == 0):
                try:
                    good = bool(c.check(launch, points, ref))
                except Exception as ex:  # noqa: BLE001
                    _failed(op, errors, key, cfgs[i], "check", ex)
                    good = False
            alive.append(good)
        alive = [i for i, k in enumerate(alive) if k]
        if not alive:
            (best, dd), best_launch = _empty(op, st.on_empty, cfgs, si, base, launches), None
            unverified |= dd
            continue
        if len(alive) == 1:
            i = alive[0]
            best, best_launch = cfgs[i], launches[i]
            continue
        sub_l = [launches[i] for i in alive]
        sub_c = [cfgs[i] for i in alive]
        with collect_faults() as faults:
            t = st.bench.times(sub_l, points)
        for launch, ex in faults:
            cfg = next((c for c, l in zip(sub_c, sub_l) if l is launch), launch)
            _failed(op, errors, key, cfg, "bench", ex)
        idx, scores = st.select.pick(t, sub_c)
        scores_all.append(list(zip(sub_c, scores)))
        if idx is None:
            (best, dd), best_launch = _empty(op, st.on_empty, cfgs, si, base, launches), None
            unverified |= dd
            continue
        best, best_launch = sub_c[idx], sub_l[idx]
    return best, best_launch, scores_all, n_cands, tuple(built), unverified


class CompiledLaunch:
    """A launch an op runs through its ``compile``: ``raw`` (the ``@flyc.jit`` launch),
    ``compiled`` (what the compile made of it), ``config`` and ``extra`` op data.

    The raw launch is called at most once, by the compile (whose own launch is that run); every
    later run, eager or under CUDA-graph capture, takes the compiled object, which skips the raw
    call's per-launch key build and globals check. A launch the search did not build compiles on
    its first run, under capture too (a raw call would trace just the same). A ``compile`` that
    hands back a raw-launching wrapper (``Compile(raw=True)``) launches raw on every run.

    This bounds the steady state only: a race may time candidates raw, and ``runtime.clear()``
    keeps the launch-level caches, so a re-tune compiles a kept launch again. Both are safe
    because no launch reads a mutable global (``test_no_launch_reads_a_mutable_global``).
    """

    __slots__ = ("config", "raw", "compiled", "extra", "compile")

    def __init__(self, raw, compile: Optional[Callable] = None, *, config=None, compiled=None, extra=None):
        self.config, self.raw, self.compiled, self.extra, self.compile = config, raw, compiled, extra, compile

    def run(self, args, compile: Optional[Callable] = None):
        """``compile(raw, args) -> (compiled, ran)``: how the launch compiles (this one's own unless
        given); ``ran`` when that compile's own launch was this run, on ``args``."""
        if self.compiled is None:
            self.compiled, ran = (compile or self.compile)(self.raw, args)
            if ran:
                return
        self.compiled(*args)


class Entry(CompiledLaunch):
    """The launch an op runs for one pick, with how final that pick is (see :attr:`Pick.tuned`)."""

    __slots__ = ("tuned",)

    def __init__(self, config, raw, *, tuned=True, compiled=None, extra=None, compile=None):
        super().__init__(raw, compile, config=config, compiled=compiled, extra=extra)
        self.tuned = tuned


def least_final(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    """How final an artifact built from two picks is: the less final of the two (None < False < True)."""
    return None if a is None or b is None else a and b


# Per thread, the enclosing Tuner.entry and Tuner.pick calls.
_scopes = threading.local()


class _Scope:
    """One enclosing call: for a Tuner.entry the picks its ``make`` made (``seen``; None for a
    Tuner.pick), and whether a pick inside asked for the caching allocator to be emptied."""

    __slots__ = ("seen", "release")

    def __init__(self, seen):
        self.seen, self.release = seen, False


@contextlib.contextmanager
def _scope(seen=None):
    """Push a scope. On exit an entry's release runs (its ``make`` and every probe frame under it
    have returned); a pick's passes to the enclosing scope, whose frames may still hold the
    probe, and runs only when none encloses it."""
    stack = _scopes.__dict__.setdefault("stack", [])
    s = _Scope(seen)
    stack.append(s)
    try:
        yield s
    finally:
        stack.pop()
        if s.release:
            if seen is None and stack:
                stack[-1].release = True
            else:
                runtime.empty_cache()


def request_release() -> None:
    """Empty the caching allocator once the innermost enclosing :meth:`Tuner.entry` (else
    :meth:`Tuner.pick`) returns -- earlier, a call site's closures and the engine's frames still
    hold the probe tensors and nothing would be returned. Right away outside both."""
    stack = getattr(_scopes, "stack", None)
    if stack:
        stack[-1].release = True
    else:
        runtime.empty_cache()


@contextlib.contextmanager
def _probed(probe: Probe):
    """``probe.points()``; a probe with ``release`` set requests a release (see
    :func:`request_release`)."""
    try:
        with probe.points() as points:
            yield points
    finally:
        if probe.release:
            request_release()


def _note(p: "Pick") -> "Pick":
    for s in getattr(_scopes, "stack", ()):
        if s.seen is not None:
            s.seen.append(p.tuned)
    return p


class Tuner:
    """One op's picks, in process and per rank only: no rank waits on another. Not thread-safe:
    two threads missing one key at once both search, and the last write wins (dispatch is
    single-threaded).

    ``off``: what the switch-off path returns -- ``"base"`` (untouched), or ``"first_valid"``
    (the first stage's candidates in order, the first that builds and passes its check, else that
    stage's ``fallback`` when it is a reference stage, or base; cached as untuned). The latter is
    for ops whose base may not hold for every shape; under capture it can not check anything, so
    it takes base provisionally and checks at the next call outside capture.

    A key where nothing builds raises :class:`NoWorkingConfig`, and keeps raising it without
    searching again until ``runtime.clear()``. When candidates build but none holds on the call's
    probe, the result follows the data, not the candidates: nothing is cached, and the next call
    outside capture searches again.

    Every artifact an op caches stays alive until ``runtime.clear()``, a replaced one too: a CUDA
    graph captured on it still launches its code. Nothing is evicted before then, so the caches
    grow with the distinct keys a process runs.
    """

    def __init__(self, op: str, *, off: str = "base"):
        if off not in ("base", "first_valid"):
            raise ValueError(f"unknown autotune off mode {off!r}")
        self.op, self.off = op, off
        self.cache: Dict[Any, Any] = {}
        self.errors: Dict[Any, str] = {}
        self.entries: Dict[Any, Entry] = {}
        self._untuned: Dict[Any, Any] = {}
        self._derived: Dict[Any, Any] = {}
        self._failed: Dict[Any, NoWorkingConfig] = {}
        self._retired: list = []
        runtime.register(self)

    def clear(self) -> None:
        self.cache.clear()
        self.errors.clear()
        self.entries.clear()
        self._untuned.clear()
        self._derived.clear()
        self._failed.clear()
        self._retired.clear()

    def pick(self, key, space: Space, *, probe: Probe, build: Callable) -> Pick:
        """Cached pick, else base (switch off / capture; see ``off``), else a search. A probe with
        ``release`` set has the caching allocator emptied once its tensors are dead (see
        :func:`request_release`)."""
        if not isinstance(probe, Probe):
            raise TypeError(f"probe must be a Probe, not {type(probe).__name__}")
        with _scope():
            p = self._pick(key, space, probe, build)
        return _note(p)

    def _pick(self, key, space, probe, build) -> Pick:
        cfg = self.cache.get(key, _UNSET)
        if cfg is not _UNSET:
            return Pick(cfg)
        if not runtime.enabled():
            if self.off == "base":
                return Pick(space.base, tuned=False)
            cfg = self._untuned.get(key, _UNSET)
            if cfg is not _UNSET:
                return Pick(cfg, tuned=False)
            reason = runtime.blocked()
            if reason is None:
                return self._first_valid(key, space, probe, build)
            runtime.warn_blocked(self.op, key, reason)
            return Pick(space.base, tuned=None)
        reason = runtime.blocked()
        if reason is not None:
            runtime.warn_blocked(self.op, key, reason)
            return Pick(space.base, tuned=None)
        self._raise_if_failed(key)
        t0 = time.time()
        launch, scores, n_cands, built, unverified = None, None, 0, (), False
        with _probed(probe) as points:
            if points is None:
                best = space.base
            else:
                best, launch, scores, n_cands, built, unverified = self._search(
                    key, space.stages, points, build, space.base
                )
        if unverified:
            runtime.warn_unverified(self.op, key, best)
            return Pick(best, launch, None, built)
        self.cache[key] = best
        runtime.record(self.op, key, best, candidates=n_cands, scores=scores, secs=time.time() - t0)
        return Pick(best, launch, True, built)

    def _raise_if_failed(self, key):
        failed = self._failed.get(key)
        if failed is not None:
            raise NoWorkingConfig(*failed.args)

    def _search(self, key, stages, points, build, base):
        try:
            return _search(self.op, stages, points, build, base, errors=self.errors, key=key)
        except NoWorkingConfig as ex:
            if not ex.data_dependent:
                self._failed[key] = ex
            raise

    def _first_valid(self, key, space, probe, build) -> Pick:
        self._raise_if_failed(key)
        st = space.stages[0]
        fallback = st.fallback if st.bench is None else space.base
        ref = Stage(st.configs, None, check=st.check, fallback=fallback)
        unverified = False
        with _probed(probe) as points:
            if points is None:
                best, launch, built = space.base, None, ()
            else:
                best, launch, _s, _n, built, unverified = self._search(key, [ref], points, build, space.base)
        if unverified:
            runtime.warn_unverified(self.op, key, best)
            return Pick(best, launch, None, built)
        self._untuned[key] = best
        return Pick(best, launch, False, built)

    @staticmethod
    def stale(e: Optional[Entry]) -> bool:
        """Whether an op's cached artifact has to be picked again now: it is missing, provisional
        (made under capture) and this call is outside capture, or untuned and tuning can run.
        Outside those cases it is reused, under capture too, so nothing is rebuilt there."""
        if e is None:
            return True
        if e.tuned:
            return False
        if e.tuned is None:
            return runtime.blocked() is None
        return runtime.enabled() and runtime.blocked() is None

    def hit(self, ekey) -> Optional[Entry]:
        """The cached artifact for ``ekey``, or None when it is :meth:`stale` -- an op checks this
        before it assembles its candidates."""
        e = self.entries.get(ekey)
        return None if self.stale(e) else e

    def _retire(self, old) -> None:
        """Keep a replaced artifact until ``clear()``: a graph captured on it still runs its code,
        which unloads once nothing holds the compile."""
        self._retired.append(old)

    def entry(self, ekey, make: Callable[[], Entry]) -> Entry:
        """The op's artifact for ``ekey``: :meth:`hit`, else ``make()``, cached. An artifact is as
        final as the least final pick ``make`` made, on any tuner (see :func:`least_final`); a
        release a pick in ``make`` requested runs once ``make`` returns."""
        e = self.hit(ekey)
        if e is None:
            seen = []
            with _scope(seen):
                e = make()
            for tuned in seen:
                e.tuned = least_final(e.tuned, tuned)
            old = self.entries.get(ekey)
            if old is not None and old is not e:
                self._retire(old)
            self.entries[ekey] = e
        return e

    def derived(self, ekey, tag, base: Entry, make: Callable[[Entry], Entry]) -> Entry:
        """``make(base)``: an artifact built from the entry ``base`` (e.g. its accumulate
        variant), cached by (``ekey``, ``tag``) for as long as ``base`` is the one passed in."""
        k = (ekey, tag)
        d = self._derived.get(k)
        if d is None or d[0] is not base:
            if d is not None:
                self._retire(d[1])
            d = self._derived[k] = (base, make(base))
        return d[1]
