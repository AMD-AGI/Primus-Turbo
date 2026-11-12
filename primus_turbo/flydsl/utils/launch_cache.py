###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Launch ``@flyc.jit`` functions through cached ``flyc.compile`` objects.

Since flydsl 0.5 a direct call of a ``@flyc.jit`` function retraces the whole launcher to MLIR and
hashes the module to find its artifact: several milliseconds of host time on every launch. A
``flyc.compile``'d object launches in ~10-20 us, but it never re-checks anything: it runs the one
specialization it was compiled for with whatever it is called with. So a host-side launch must go
through a compiled object, and its cache key must cover everything the trace bakes in.

``run_launcher`` derives that key itself: the launcher object (one built per call by a factory
closes over its own Python values), the value of every ``Constexpr`` / ``Type`` argument, the type
of every other argument and, for a tensor, its dtype, rank and unit-stride axis (a bare
``torch.Tensor`` compiles layout-dynamic), the ``__cache_signature__`` of an argument that has one
(a ``flyc.from_torch_tensor`` wrapper, a custom ``JitArgument``), and the compile hints in effect. ``run_compiled`` takes a key from the caller,
for a launch whose key the caller already holds.
"""

import inspect

import flydsl.compiler as flyc
import torch
from flydsl.compiler.jit_function import (
    JitFunction,
    is_type_param_annotation,
    resolve_signature,
)
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr.typing import Constexpr

__all__ = ["bind_args", "launch_key", "run_compiled", "run_launcher", "run_launcher_cached"]

_COMPILED = {}
# launcher -> (signature, positions of its trace-time arguments, default of each parameter)
_SPECS = {}
_EMPTY = inspect.Parameter.empty


def _spec(launcher):
    spec = _SPECS.get(launcher)
    if spec is None:
        sig = resolve_signature(launcher.func)
        params = list(sig.parameters.values())
        baked = tuple(
            i
            for i, p in enumerate(params)
            if p.annotation is not _EMPTY
            and (Constexpr.is_constexpr_annotation(p.annotation) or is_type_param_annotation(p.annotation))
        )
        spec = _SPECS[launcher] = (sig, baked, tuple(p.default for p in params))
    return spec


def _arg_key(a):
    if isinstance(a, torch.Tensor):
        stride = a.stride()
        return (a.dtype, len(stride), stride.index(1) if 1 in stride else None)
    signature = getattr(a, "__cache_signature__", None)
    if signature is not None:
        return (type(a), signature())
    return type(a)


def _hashable(key):
    try:
        hash(key)
        return key
    except TypeError:
        return repr(key)


def bind_args(launcher, args, kwargs):
    """``args`` / ``kwargs`` of a call of ``launcher`` as the full positional tuple a compiled
    object takes."""
    bound = _spec(launcher)[0].bind(*args, **kwargs)
    bound.apply_defaults()
    return tuple(bound.arguments.values())


def launch_key(launcher, args):
    """The ``run_launcher`` cache key of ``launcher(*args)``; ``args`` are complete and positional."""
    _, baked, _ = _spec(launcher)
    hints = CompilationContext.get_compile_hints()
    key = (
        launcher,
        tuple(args[i] for i in baked),
        tuple(map(_arg_key, args)),
        tuple(sorted(hints.items())) if hints else None,
    )
    return _hashable(key)


def run_launcher(launcher, *args, **kwargs):
    """Run ``launcher(*args, **kwargs)`` through a ``flyc.compile``'d object, one per
    ``launch_key``. A launcher that is not ``@flyc.jit`` (a plain Python closure) is called as is.

    A positional call with no ``kwargs`` skips the signature bind; trailing parameters with a
    default may be left out of it. The process-wide cache keeps every launcher it ran alive: a
    launcher built per shape by an evicting builder takes ``run_launcher_cached`` instead."""
    run_launcher_cached(_COMPILED, launcher, *args, **kwargs)


def run_launcher_cached(cache: dict, launcher, *args, **kwargs):
    """``run_launcher`` into the caller's ``cache``, which goes when the caller drops it."""
    if not isinstance(launcher, JitFunction):
        return launcher(*args, **kwargs)
    _, _, defaults = _spec(launcher)
    if kwargs:
        args = bind_args(launcher, args, kwargs)
    elif len(args) < len(defaults):
        missing = defaults[len(args) :]
        if _EMPTY in missing:
            raise TypeError(f"{launcher.func.__name__}() missing positional arguments")
        args = args + missing
    key = launch_key(launcher, args)
    compiled = cache.get(key)
    if compiled is None:
        # the compile runs this launch; COMPILE_ONLY returns None and runs nothing
        compiled = flyc.compile(launcher, *args)
        if compiled is not None:
            cache[key] = compiled
    else:
        compiled(*args)


def run_compiled(cache: dict, key, launch, *args):
    """``flyc.compile`` the launch once per ``key``, then call the compiled object.

    The caller owns the key: it must cover the launch's identity (some callers build the closure
    per call) plus any argument value the compile bakes in, such as an ``M`` outside the
    builder's key."""
    compiled = cache.get(key)
    if compiled is None:
        compiled = flyc.compile(launch, *args)
        if compiled is not None:
            cache[key] = compiled
    else:
        compiled(*args)
