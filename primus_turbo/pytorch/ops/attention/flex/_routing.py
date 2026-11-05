###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backend selection (performance routing layer).

Once a variant is recognised (mask classified, ``score_mod`` mapped) it still passes
through :func:`choose_backend` before dispatch. By default this returns ``"turbo"``,
so behaviour is identical to dispatching directly; a tuner or a test can bias
specific shapes/kinds towards the ``_dispatch_custom`` hook via
:func:`register_backend_override` without touching the classifier.
"""

import inspect
from typing import Any, Callable, Dict, Optional, Tuple


def _dispatch_custom(*_args, **_kwargs):
    """Hook for the future high-performance arbitrary score_mod/mask_mod kernel.

    Not implemented yet: this is intentionally a hard stop so callers never get a
    silently wrong result from an unrecognised programmable modification. It is
    the shared landing point for (a) arbitrary score_mod/mask_mod and (b) a variant
    explicitly routed away from Turbo by :func:`choose_backend`. A recognised /
    explicit logits soft-cap is gated *before* this hook (single point in
    :func:`flex_attention`), so ``softcap`` never reaches here as ``> 0``.
    """
    raise NotImplementedError(
        "custom/arbitrary score_mod+mask_mod fast path not implemented yet (planned: codegen 'path B')"
    )


# Valid return values / backend identifiers for the routing layer.
#   "turbo"  -> dispatch the recognised variant to ``flash_attn_func``.
#   "custom" -> route to the ``_dispatch_custom`` hook (currently a stub).
_VALID_BACKENDS: Tuple[str, ...] = ("turbo", "custom")


# Ordered registry of ``(matcher, backend)`` overrides. ``matcher(ctx) -> bool``
# inspects a routing-context dict; the first match wins. An empty registry means
# ``choose_backend`` always returns "turbo", i.e. the historical behaviour where
# every recognised variant goes straight to Turbo.
_BACKEND_OVERRIDES: list = []


def register_backend_override(matcher: Callable[[Dict[str, Any]], bool], backend: str) -> None:
    """Register a performance-routing override consulted by :func:`choose_backend`.

    ``matcher`` receives the routing context dict (keys: ``kind``, ``causal``,
    ``window_size``, ``shape``, ``dtype``, ``has_alibi``, ``has_softcap``,
    ``has_dropout``, ``has_sink``, ``has_bias`` and the full ``mask_cfg``) and
    returns ``True`` to force ``backend`` for that call.
    Overrides are evaluated in registration order and the first match wins, so a
    tuner can steer specific shapes/kinds to the custom hook without disturbing
    the classifier. Registering nothing keeps the default all-Turbo routing.
    """
    if not callable(matcher):
        raise TypeError("register_backend_override: matcher must be a callable(ctx)->bool.")
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"register_backend_override: backend must be one of {_VALID_BACKENDS}, got {backend!r}."
        )
    _BACKEND_OVERRIDES.append((matcher, backend))


def clear_backend_overrides() -> None:
    """Remove all registered overrides, restoring the default all-Turbo routing."""
    _BACKEND_OVERRIDES.clear()


def choose_backend(
    mask_cfg: Dict[str, Any],
    *,
    shape: Tuple[int, ...],
    dtype: Any,
    has_alibi: bool,
    has_softcap: bool = False,
    has_dropout: bool = False,
    has_sink: bool = False,
    has_bias: bool = False,
) -> str:
    """Pick the execution backend for an already-recognised (mask, score_mod) combo.

    Returns ``"turbo"`` (dispatch to ``flash_attn_func``) or ``"custom"`` (route to
    the ``_dispatch_custom`` hook). The default is always ``"turbo"`` so every
    variant the compat layer accelerates keeps the fast path and existing
    behaviour is unchanged. Registered overrides (see
    :func:`register_backend_override`) are consulted first, in order; the first
    matching one decides. ``has_dropout`` / ``has_sink`` / ``has_bias`` reflect the
    Turbo-extension passthrough args and are surfaced in the routing context
    alongside ``has_alibi`` / ``has_softcap`` so a tuner can key on them too.
    """
    ctx: Dict[str, Any] = {
        "kind": mask_cfg.get("kind"),
        "causal": mask_cfg.get("causal"),
        "window_size": mask_cfg.get("window_size"),
        "shape": tuple(shape),
        "dtype": dtype,
        "has_alibi": bool(has_alibi),
        "has_softcap": bool(has_softcap),
        "has_dropout": bool(has_dropout),
        "has_sink": bool(has_sink),
        "has_bias": bool(has_bias),
        "mask_cfg": mask_cfg,
    }
    for matcher, backend in _BACKEND_OVERRIDES:
        try:
            hit = matcher(ctx)
        except Exception as exc:  # a broken matcher must fail loud, never silently reroute
            raise RuntimeError(f"choose_backend: backend override matcher raised: {exc!r}") from exc
        if hit:
            return backend
    return "turbo"


# id(fn) -> (fn, params). The callable itself is retained deliberately: keying on
# id() alone is unsound because CPython recycles ids of collected objects, and a
# recycled id would hand back another function's parameter set. Backends are
# module-level functions that live for the process anyway, so pinning a handful of
# them costs nothing.
_BACKEND_PARAM_CACHE: Dict[int, Tuple[Callable, Optional[frozenset]]] = {}


def _backend_accepts(fn: Callable, name: str) -> bool:
    """Does ``fn`` take a keyword argument called ``name``?

    Used to tell "this Primus-Turbo build predates the parameter" apart from "the
    caller asked for something we support". Getting that wrong in either direction is
    bad: passing an unknown kwarg surfaces a bare ``TypeError`` from inside the backend
    binding (unreadable, and it names the wrong culprit), while silently dropping the
    argument changes the numbers without telling anyone. A backend that takes
    ``**kwargs`` is assumed to accept everything, and an unintrospectable callable
    (C extension, no signature) is given the benefit of the doubt rather than
    pre-emptively rejected.
    """
    key = id(fn)
    cached = _BACKEND_PARAM_CACHE.get(key)
    if cached is not None and cached[0] is fn:
        params = cached[1]
    else:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover - C callables
            params = None
        else:
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                params = None
            else:
                params = frozenset(sig.parameters)
        _BACKEND_PARAM_CACHE[key] = (fn, params)
    return True if params is None else name in params
