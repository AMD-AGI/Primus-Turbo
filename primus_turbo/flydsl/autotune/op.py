###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""What a FlyDSL op declares to be tuned: an :class:`Op` (its picks, entries and how its launches
compile), its :class:`Compile`, and the :class:`Build` a search builds its candidates through. The
search itself -- the candidates, the race, the probe -- is the call site's (see :class:`Op`)."""

import contextlib
from typing import Any, Callable, Optional

import flydsl.compiler as flyc
import torch
from flydsl.compiler.jit_function import JitFunction
from flydsl.compiler.kernel_function import CompilationContext

from primus_turbo.flydsl.autotune import runtime
from primus_turbo.flydsl.autotune.engine import Entry, Space, Tuner, request_release
from primus_turbo.flydsl.autotune.policies import Probe

__all__ = ["Build", "Compile", "Op"]


# One compile per (launch, hints, layout, argument geometry), shared by the race that builds a
# launch and the entry that runs it, so the winner takes the race's compile instead of tracing
# and running it again. A clear() empties it with the picks, and nothing else does (a captured
# graph may run any of them); each value holds its launch, so the id in the key is never reused.
_COMPILED_ONCE: dict = {}
runtime.register(clear_hook=_COMPILED_ONCE.clear)


def _compile_once_key(launch, args, hints, static=False):
    """(launch identity, hints, layout, then per argument its shape/stride/dtype or int value).
    The trailing argument is the stream: it selects where a launch goes, so it is not in the key."""
    parts = [id(launch), None if hints is None else repr(hints), static]
    for a in args[:-1]:
        if isinstance(a, torch.Tensor):
            parts.append((tuple(a.shape), a.stride(), a.dtype))
        elif isinstance(a, int):
            parts.append(a)
        else:
            parts.append(type(a).__name__)
    return tuple(parts)


def _static_layout(args):
    """The tensor arguments as static-layout memrefs: a bare torch.Tensor compiles layout-dynamic
    and re-reads shape/stride per launch, while a compile kept per geometry is already one per
    operand layout."""
    return tuple(flyc.from_torch_tensor(a) if isinstance(a, torch.Tensor) else a for a in args)


def _scratch_out(args, out_index):
    scratch = torch.zeros_like(args[out_index])
    return args[:out_index] + (scratch,) + tuple(args[out_index + 1 :])


class Compile:
    """How an op's launches compile; ``compile(raw, args) -> (compiled, ran)`` is what
    :meth:`CompiledLaunch.run` calls.

    * ``Compile()`` -- ``flyc.compile`` on the caller's own arguments, so its launch is that run
      (an accumulate build folds in once).
    * ``hints`` -- the compile hints the launch is traced under (and raced under, with ``raw``).
      A compiled op races its launches raw, without the hints: hinting the race would change
      what it picks, and that is unmeasured.
    * ``once`` -- through the shared per-geometry cache: a hit is ``(compiled, False)``.
    * ``static_layout`` -- compiled once per geometry with the operand layouts pinned; the race
      builds compile too, onto a scratch output (:meth:`race`), and an entry takes that compile.
    * ``raw`` -- no compile: every run launches raw under ``hints``.

    A launch that is a plain closure, not ``@flyc.jit``, runs itself every time.
    """

    __slots__ = ("hints", "once", "static_layout", "raw", "out_index")

    def __init__(self, *, hints=None, once=False, static_layout=False, raw=False, out_index=2):
        if raw and (once or static_layout):
            raise ValueError("a raw Compile compiles nothing, so it can not be once or static")
        self.hints, self.once, self.static_layout, self.raw = hints, once or static_layout, static_layout, raw
        self.out_index = out_index

    def __repr__(self):
        flags = [n for n in ("once", "static_layout", "raw") if getattr(self, n)]
        if self.hints is not None:
            flags.append(f"hints={self.hints!r}")
        return f"Compile({', '.join(flags)})"

    def _hinted(self):
        return CompilationContext.compile_hints(self.hints) if self.hints else contextlib.nullcontext()

    def racing(self):
        """The context a race runs in: a raw op times its launches under its hints."""
        return self._hinted() if self.raw else contextlib.nullcontext()

    def _compile(self, launch, args):
        with self._hinted():
            return flyc.compile(launch, *(_static_layout(args) if self.static_layout else args))

    def __call__(self, launch, args):
        if self.raw:
            hints = self.hints

            def run(*a):
                with CompilationContext.compile_hints(hints):
                    launch(*a)

            return run, False
        if not isinstance(launch, JitFunction) and not self.static_layout:
            return launch, False
        if not self.once:
            return self._compile(launch, args), True
        key = _compile_once_key(launch, args, self.hints, self.static_layout)
        hit = _COMPILED_ONCE.get(key)
        if hit is not None:
            return hit[1], False
        compiled = self._compile(launch, args)
        _COMPILED_ONCE[key] = (launch, compiled)
        return compiled, True

    def scratch(self, launch, args):
        """A race build: compiled through the shared cache onto a scratch output, so the timed
        launches leave the caller's alone and the entry that later runs the winner on the same
        geometry takes this compile."""
        key = _compile_once_key(launch, args, self.hints, self.static_layout)
        hit = _COMPILED_ONCE.get(key)
        if hit is None:
            hit = _COMPILED_ONCE[key] = (launch, self._compile(launch, _scratch_out(args, self.out_index)))
        return hit[1]

    def race(self, launch, args):
        """What the race times for ``launch``: its static compile (see :meth:`scratch`), else the
        raw launch itself."""
        return self.scratch(launch, args) if self.static_layout else launch


class Build:
    """A search's ``build`` for :meth:`Op.pick`: ``entry_of(config) -> Entry``, timed as the op's
    compile races its launch on ``args``. It keeps what it built, so :meth:`winner` hands back the
    pick's entry instead of building it again."""

    __slots__ = ("compile", "entry_of", "args", "built")

    def __init__(self, compile: Compile, entry_of: Callable[[Any], Entry], args=None):
        self.compile, self.entry_of, self.args, self.built = compile, entry_of, args, {}

    def __call__(self, cfg):
        e = self.built[cfg] = self.entry_of(cfg)
        return self.compile.race(e.raw, self.args)

    def winner(self, res) -> Entry:
        """The entry of the pick ``res``, as final as it is; a static-layout entry takes the
        compile its race built."""
        e = self.built.get(res.config)
        if e is None:
            e = self.entry_of(res.config)
        e.tuned = res.tuned
        if self.compile.static_layout and res.launch is not None:
            e.compiled = res.launch
        return e


class Op(Tuner):
    """One tuned op: a :class:`Tuner` with how its launches compile and its switch-off mode.

    A call site runs one flow: ``entry(key, make)`` -- on a miss ``make`` builds the search
    (``staged(...)``), :meth:`pick` s it with a :meth:`build` and returns ``build.winner(res)`` --
    then ``derived(key, True, entry, ...)`` for a beta=1 call that rebuilds the pick with the
    accumulate epilogue, then :meth:`launch`. ``release``: hand a search's buffers back to the
    allocator once it built anything, when the entry returns (as a ``Synthetic`` probe does).
    """

    def __init__(
        self, op: str, *, compile: Optional[Compile] = None, off: str = "base", release: bool = False
    ):
        super().__init__(op, off=off)
        self.compile, self.release = compile or Compile(), release

    def build(self, entry_of: Callable[[Any], Entry], args=None) -> Build:
        """The :class:`Build` of ``entry_of`` on ``args`` (needed by a static-layout compile only)."""
        return Build(self.compile, entry_of, args)

    def pick(self, key, space: Space, *, probe: Probe, build: Callable):
        """:meth:`Tuner.pick`, raced under the op's compile."""
        with self.compile.racing():
            res = super().pick(key, space, probe=probe, build=build)
        if self.release and res.built:
            request_release()
        return res

    def launch(self, entry: Entry, args) -> None:
        """Run ``entry``, one this op made, on ``args`` through the op's compile."""
        entry.run(args, self.compile)
