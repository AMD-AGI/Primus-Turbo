###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""How an op races its candidates: a :class:`Race` is one timing method with its selection and
check; a staged search is a :class:`Reference` or a first :class:`Refine`, then more
:class:`Refine` steps, each narrowing around the previous step's pick."""

from typing import Callable, NamedTuple, Optional, Union

from primus_turbo.flydsl.autotune.engine import Space
from primus_turbo.flydsl.autotune.policies import (
    RAISE,
    Bench,
    Check,
    Fastest,
    Finite,
    Interleaved,
    MatchBase,
    Remember,
    Reproducible,
    Stage,
)

__all__ = ["Race", "Refine", "Reference", "staged"]

_CHECKS = {"finite": Finite, "reproducible": Reproducible, "match": MatchBase}


def _check(check: Union[None, str, Check]) -> Optional[Check]:
    if check is None or not isinstance(check, str):
        return check
    return _CHECKS[check]()


class Race:
    """One race: ``bench`` times the candidates, then the fastest that passes ``check`` wins.

    Build one with :meth:`interleaved` (every candidate inside one shared set of windows) or around
    any :class:`~primus_turbo.flydsl.autotune.policies.Bench`. Selection and check:

    * ``margin`` < 1: the first candidate (the base, or the previous step's pick) stays unless
      another beats it by that factor;
    * ``compare``: ``"geomean"`` over the points, or ``"worst"`` -- the worst point's time relative
      to the first candidate, so a challenger has to win everywhere;
    * ``check``: None, ``"finite"`` (the output sample stays finite), ``"reproducible"`` (finite,
      and the same bytes on repeated cold calls; see ``Reproducible``), ``"match"`` (matches the
      base's output), or a ``Check`` such as ``MatchBase(n=None)``;
    * ``charge(config) -> factor``: inflates a speculative candidate's time;
    * ``on_empty``: ``"base"`` or ``"raise"`` when no candidate survives; by default ``"base"``
      when a margin or a charge is set, else ``"raise"``. :class:`Stage` refuses any other value
      when the race is staged.
    """

    __slots__ = ("bench", "select", "check", "on_empty")

    def __init__(
        self,
        bench: Bench,
        *,
        margin: float = 1.0,
        compare: str = "geomean",
        check: Union[None, str, Check] = None,
        charge: Optional[Callable] = None,
        on_empty: Optional[str] = None,
    ):
        if on_empty is None:
            on_empty = "base" if margin < 1.0 or charge is not None else "raise"
        self.bench = bench
        self.select = Fastest(margin, compare, charge)
        self.check = _check(check)
        self.on_empty = on_empty

    @classmethod
    def interleaved(
        cls, passes, iters, *, warmup=0, warmup_rounds=True, ramp=0, palindrome=True, visit=(0, 1), **select
    ) -> "Race":
        """:class:`~primus_turbo.flydsl.autotune.policies.Interleaved`; ``visit`` = its
        ``(visit_warmup, visit_reps)``."""
        bench = Interleaved(
            passes,
            iters,
            warmup=warmup,
            warmup_rounds=warmup_rounds,
            ramp=ramp,
            palindrome=palindrome,
            visit_warmup=visit[0],
            visit_reps=visit[1],
        )
        return cls(bench, **select)

    def timed_by(self, bench: Bench) -> "Race":
        """This race with its candidates timed by ``bench`` instead."""
        r = object.__new__(Race)
        r.bench, r.select, r.check, r.on_empty = bench, self.select, self.check, self.on_empty
        return r

    def checked_by(self, check: Union[None, str, Check]) -> "Race":
        """This race with its candidates held to ``check`` instead."""
        r = object.__new__(Race)
        r.bench, r.select, r.check, r.on_empty = self.bench, self.select, _check(check), self.on_empty
        return r

    def __repr__(self):
        return f"Race({self.bench!r}, {self.select!r}, {self.check!r}, on_empty={self.on_empty!r})"


class Refine(NamedTuple):
    """One step of a staged search: ``configs(prev_best) -> list``, the previous pick first
    (``prev_best`` is None in the first step), raced by ``race``. ``reuse_winner_time``: a launch
    already timed by an earlier step of the same race keeps that time, so a step that leads with
    the previous pick compares against the time it won with rather than a fresh sample."""

    configs: Callable
    race: Race
    reuse_winner_time: bool = False


class Reference(NamedTuple):
    """A step that times nothing: the first of ``configs(prev_best)`` that builds and passes
    ``check`` is the pick, and ``fallback`` is taken when none does (``RAISE``: raise instead)."""

    configs: Callable
    check: Union[None, str, Check] = "finite"
    fallback: object = RAISE


def staged(*steps, base) -> Space:
    """A staged search over ``steps`` (:class:`Reference` / :class:`Refine`), ``base`` untuned."""
    remembered = {}
    stages = []
    for s in steps:
        if isinstance(s, Reference):
            stages.append(Stage(s.configs, bench=None, check=_check(s.check), fallback=s.fallback))
            continue
        bench = s.race.bench
        if s.reuse_winner_time:
            bench = remembered.get(id(s.race))
            if bench is None:
                bench = remembered[id(s.race)] = Remember(s.race.bench)
        stages.append(Stage(s.configs, bench, s.race.select, s.race.check, on_empty=s.race.on_empty))
    return Space.staged(*stages, base=base)
